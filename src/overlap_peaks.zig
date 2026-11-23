//! Bounded GPU pages of normalized native SO(3) local maxima for overlap.
//! This module contains scalar grid ranking only; orientations and symmetry
//! remain with spherical.zig and the existing native symmetry implementation.
const std = @import("std");
const gpu = @import("gpu/gpu.zig");
const shaders = @import("shaders");
pub const Key = extern struct {
    value_bits: u32,
    index: u32,

    pub fn before(_: void, a: Key, b: Key) bool {
        const av: f32 = @bitCast(a.value_bits);
        const bv: f32 = @bitCast(b.value_bits);
        return av > bv or (av == bv and a.index < b.index);
    }
};
pub const missing = Key{ .value_bits = @bitCast(@as(f32, -3.4e38)), .index = 0xffffffff };
pub const Params = extern struct {
    n: u32,
    nc: u32,
    nb: u32,
    first: u32,
    rows: u32,
    core_first: u32,
    core_rows: u32,
    grid_x: u32 = 1,
    input_count: u32 = 0,
    cursor_set: u32 = 0,
    item: u32 = 0,
    capacity: u32 = 0,
};
pub const Selector = struct {
    g: *gpu.Gpu,
    scores: gpu.Buf = .{},
    pole: gpu.Buf = .{},
    cursor: gpu.Buf = .{},
    counts: gpu.Buf = .{},
    records: [2]gpu.Buf = .{ .{}, .{} },
    pub fn init(g: *gpu.Gpu) Selector {
        return .{ .g = g };
    }
    pub fn deinit(self: *Selector) void {
        self.g.release(&self.scores);
        self.g.release(&self.pole);
        self.g.release(&self.cursor);
        self.g.release(&self.counts);
        for (&self.records) |*r| self.g.release(r);
    }
    fn pipe(self: *Selector, comptime entry: []const u8) u32 {
        return self.g.pipeline("overlap_peaks/" ++ entry, shaders.overlap_peaks, entry) catch gpu.NO_PIPE;
    }
    pub fn setCursor(self: *Selector, after: ?Key) void {
        self.g.ensure(&self.cursor, 8);
        self.g.writeSlice(self.cursor, 0, Key, &.{after orelse missing});
    }
    pub fn beginCollection(self: *Selector, n_items: u32) void {
        const bytes = @as(u64, n_items) * @sizeOf(u32);
        self.g.ensure(&self.counts, bytes);
        self.g.clearRange(self.counts, 0, bytes);
    }
    /// Record one core plus its complete j halo. The numerator and reciprocal
    /// buffers contain only these rows, including complete a/c lines at poles.
    /// With a collection capacity, append each local maximum to its item's
    /// bounded list. Otherwise eight ordered records land at out_offset.
    pub fn slab(self: *Selector, params: Params, numerator: gpu.Buf, reciprocal: gpu.Buf, out: gpu.Buf, out_offset: u64) void {
        const g = self.g;
        var p = params;
        const pixels = @as(u64, p.rows) * p.nc * p.n;
        g.ensure(&self.scores, pixels * 4);
        g.ensure(&self.pole, @as(u64, 6) * p.nc * 8);
        var f = gpu.Gpu.flat(@intCast((pixels + 255) / 256));
        p.grid_x = f[0];
        g.dispatch(self.pipe("normalize_slab"), f[0], f[1], 1, &.{ g.uniform(0, std.mem.asBytes(&p)), self.scores.at(1), reciprocal.at(2), numerator.at(7) });
        f = gpu.Gpu.flat(4 * p.nc);
        p.grid_x = f[0];
        g.dispatch(self.pipe("pole_groups"), f[0], f[1], 1, &.{ g.uniform(0, std.mem.asBytes(&p)), self.scores.at(1), self.pole.at(3) });
        const blocks = (p.core_rows * p.nc * p.n + 255) / 256;
        f = gpu.Gpu.flat(blocks);
        p.grid_x = f[0];
        if (p.capacity != 0) {
            g.dispatch(self.pipe("collect_maxima"), f[0], f[1], 1, &.{ g.uniform(0, std.mem.asBytes(&p)), self.scores.at(1), self.pole.at(3), out.at(5), self.counts.at(8) });
            return;
        }
        g.ensure(&self.records[0], @as(u64, blocks) * 8 * 8);
        g.dispatch(self.pipe("block_maxima"), f[0], f[1], 1, &.{ g.uniform(0, std.mem.asBytes(&p)), self.scores.at(1), self.pole.at(3), self.records[0].at(5), self.cursor.at(6) });
        self.reduce(self.records[0], blocks * 8, out, out_offset);
    }
    /// Exact sorted top-eight tree. Each source block contributes eight entries;
    /// reducing 32 lists cannot discard a member of the global top eight.
    pub fn reduce(self: *Selector, input: gpu.Buf, count: u32, out: gpu.Buf, out_offset: u64) void {
        const g = self.g;
        var source = input;
        var n = count;
        while (n > 8) {
            const next = (n + 255) / 256;
            const target: usize = if (source.id == self.records[0].id) 1 else 0;
            g.ensure(&self.records[target], @as(u64, next) * 8 * 8);
            var p: Params = .{ .n = 0, .nc = 0, .nb = 0, .first = 0, .rows = 0, .core_first = 0, .core_rows = 0, .input_count = n };
            const f = gpu.Gpu.flat(next);
            p.grid_x = f[0];
            g.dispatch(self.pipe("reduce_pages"), f[0], f[1], 1, &.{ g.uniform(0, std.mem.asBytes(&p)), source.at(4), self.records[target].at(5) });
            source = self.records[target];
            n = next * 8;
        }
        g.copy(source, 0, out, out_offset, 64);
    }
};
