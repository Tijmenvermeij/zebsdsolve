//! GPU fixed prepared-domain NNLS. No physical fraction or blur/gain parity is implied.
const std = @import("std");
const gpu = @import("gpu/gpu.zig");
const shaders = @import("shaders");
pub const capacity = 4;
/// ABI: 16 f32, matching WGSL State. Flags: 1 ill-conditioned, 2 flat target,
/// 4 full-Gram spectral condition unassessed, 8 unresolved attribution (NaN
/// joint outputs), 16 unresolved discovery candidate skipped (prior fit retained).
pub const State = extern struct {
    sse: f32 = 0,
    initial_sse: f32 = 0,
    phase: f32 = -1,
    flags: f32 = 0,
    amplitude: [4]f32 = @splat(0),
    single_sse: [4]f32 = @splat(0),
    correlation: f32 = 0,
    condition: f32 = 1,
    best_single_sse: f32 = 0,
    trial_sse: f32 = 0,
};
/// V2 per-cell controller state; old 64-byte State ABI is unchanged.
pub const Progress = extern struct {
    retained_order: u32 = 0,
    proposed_count: u32 = 0,
    active_count: u32 = 0,
    stop_reason: u32 = 0,
    previous_sse: f32 = 0,
    incremental_improvement: f32 = 0,
    boundary_seen: u32 = 0,
    _pad: u32 = 0,
};
pub const Component = extern struct {
    q: [4]f32 = @splat(0),
    amplitude: f32 = 0,
    phase: i32 = -1,
    proposal_ncc: f32 = std.math.nan(f32),
    proposal_nmi: f32 = std.math.nan(f32),
};
const Params = extern struct {
    n: u32,
    px: u32,
    k: u32 = 0,
    phase: u32 = 0,
    mode: u32 = 0,
    grid_x: u32 = 1,
    width: u32 = 1,
    pad1: u32 = 0,
    mask: [4]f32 = @splat(0),
};
pub const Fit = struct {
    g: *gpu.Gpu,
    n: u32,
    px: u32,
    width: u32,
    mask: [3]f32,
    contrast: bool = false,
    original: gpu.Buf = .{},
    bank: gpu.Buf = .{},
    state: gpu.Buf = .{},
    progress: gpu.Buf = .{},
    pub fn init(g: *gpu.Gpu, n: u32, px: u32, width: u32, mask: [3]f32, target: gpu.Buf) Fit {
        var f = allocate(g, n, px, width, mask);
        f.resetTarget(target);
        return f;
    }

    /// Reserve space for n lanes. Set the active length and reset the target
    /// before recording work if fewer lanes are populated.
    pub fn allocate(g: *gpu.Gpu, n: u32, px: u32, width: u32, mask: [3]f32) Fit {
        const f: Fit = .{
            .g = g,
            .n = n,
            .px = px,
            .width = width,
            .mask = mask,
            .original = g.storage(@as(u64, n) * px * 4),
            .bank = g.storage(@as(u64, capacity) * n * px * 4),
            .state = g.storage(@as(u64, n) * @sizeOf(State)),
            .progress = g.storage(@as(u64, n) * @sizeOf(Progress)),
        };
        g.clear(f.bank);
        return f;
    }
    /// Reuse this allocation for another batch of n prepared target rows.
    pub fn resetTarget(self: *Fit, target: gpu.Buf) void {
        if (target.id != self.original.id) self.g.copy(target, 0, self.original, 0, @as(u64, self.n) * self.px * 4);
        self.run("initialize", .{ .n = self.n, .px = self.px }, target, target);
    }
    pub fn deinit(self: *Fit) void {
        self.g.release(&self.original);
        self.g.release(&self.bank);
        self.g.release(&self.state);
        self.g.release(&self.progress);
    }
    fn run(self: *Fit, comptime entry: []const u8, params: Params, proposed: gpu.Buf, out: gpu.Buf) void {
        var p = params;
        p.width = self.width;
        if (comptime std.mem.eql(u8, entry, "fit_nuisance")) p.pad1 = @intFromBool(self.contrast);
        p.mask = .{ self.mask[0], self.mask[1], self.mask[2], 0 };
        const grid = gpu.Gpu.flat(self.n);
        p.grid_x = grid[0];
        const pipe = self.g.pipeline("overlap/" ++ entry, shaders.overlap, entry) catch gpu.NO_PIPE;
        // Bind only the resources reachable from each entry point: WebGPU infers layouts.
        const u = self.g.uniform(0, std.mem.asBytes(&p));
        if (comptime std.mem.eql(u8, entry, "initialize")) {
            self.g.dispatch(pipe, grid[0], grid[1], 1, &.{ u, self.original.at(1), self.state.at(4), self.progress.at(6) });
        } else if (comptime std.mem.eql(u8, entry, "begin_step")) {
            self.g.dispatch(pipe, grid[0], grid[1], 1, &.{ u, self.state.at(4), self.progress.at(6) });
        } else if (comptime std.mem.eql(u8, entry, "fit") or std.mem.eql(u8, entry, "fit_nuisance")) {
            self.g.dispatch(pipe, grid[0], grid[1], 1, &.{ u, self.original.at(1), self.bank.at(2), proposed.at(3), self.state.at(4), self.progress.at(6) });
        } else {
            self.g.dispatch(pipe, grid[0], grid[1], 1, &.{ u, self.original.at(1), self.bank.at(2), self.state.at(4), out.at(5), self.progress.at(6) });
        }
    }
    pub fn begin(self: *Fit) void {
        self.run("begin_step", .{ .n = self.n, .px = self.px }, self.original, self.original);
    }
    pub fn beginManaged(self: *Fit) void {
        self.run("begin_step", .{ .n = self.n, .px = self.px, .mode = 1 }, self.original, self.original);
    }
    pub fn candidateManaged(self: *Fit, patterns: gpu.Buf, phase: u32, seed: bool) void {
        self.run("fit", .{ .n = self.n, .px = self.px, .phase = phase, .mode = if (seed) 4 else 2 }, patterns, self.original);
    }
    pub fn readProgress(self: *Fit, out: []Progress) void {
        self.g.read(self.progress, 0, std.mem.sliceAsBytes(out));
    }
    pub fn writeProgress(self: *Fit, values: []const Progress) void {
        self.g.writeSlice(self.progress, 0, Progress, values);
    }
    pub fn candidate(self: *Fit, patterns: gpu.Buf, k: u32, phase: u32) void {
        self.run("fit", .{ .n = self.n, .px = self.px, .k = k, .phase = phase }, patterns, self.original);
    }
    pub fn nuisanceTrial(self: *Fit, singleton: bool, k: u32) void {
        self.run("fit_nuisance", .{ .n = self.n, .px = self.px, .mode = if (singleton) 7 else 6, .k = k }, self.original, self.original);
    }
    pub fn fixed(self: *Fit, k: u32) void {
        self.run("fit", .{ .n = self.n, .px = self.px, .k = k, .mode = 1 }, self.original, self.original);
    }
    pub fn residual(self: *Fit, k: u32, out: gpu.Buf, normalized: bool) void {
        self.run("make_residual", .{ .n = self.n, .px = self.px, .k = k, .mode = @intFromBool(normalized) }, self.original, out);
    }
    pub fn read(self: *Fit, out: []State) void {
        self.g.read(self.state, 0, std.mem.sliceAsBytes(out));
    }
};
