//! Spherical indexing on the GPU: the schedule over shaders/sphere.wgsl, fft.wgsl and so3.wgsl.
//!
//! `Tables` holds what depends only on the bandwidth L (grid, Legendre and Wigner tables);
//! `PhaseGpu` what a phase adds (its master's weight tables, and the denominator volume of the
//! normalized correlation for the reference geometry). `Spherical` owns the work buffers and
//! records the two stages for a tile of items (an item is one pattern under one geometry):
//!   coefficients   patterns → sphere → harmonic coefficients
//!   correlate      coefficients × a master → correlation over SO(3) → best orientation each
//! The correlation's three transforms run b, c, a (so3.wgsl says why), and what an item keeps
//! between them is half a volume of complex numbers: (n / 2 + 1) · L · nc.
//! Nothing here reads back: results land in `peaks` (and `stats`) for the caller to read with
//! everything else it queued.
const std = @import("std");
const gpu_mod = @import("gpu/gpu.zig");
const shaders = @import("shaders");
const sh = @import("sh.zig");
const sym = @import("sym.zig");
const rot = @import("rot.zig");
const peak_select = @import("overlap_peaks.zig");
const master_mod = @import("master.zig");

const Gpu = gpu_mod.Gpu;
const Buf = gpu_mod.Buf;
const Bind = gpu_mod.Bind;
const Allocator = std.mem.Allocator;

pub const TW_LEN = 1024;

const FftP = extern struct {
    n: u32,
    log2n: u32,
    keep: u32 = 0,
    n_lines: u32,
    stored: u32,
    grid_x: u32 = 1,
    tw_len: u32 = TW_LEN,
    pad0: u32 = 0,
    sign: f32 = -1,
    scale: f32 = 1,
    pad1: f32 = 0,
    pad2: f32 = 0,
};

const SphP = extern struct {
    n_items: u32,
    n_theta: u32,
    n_phi: u32,
    bw: u32,
    pat_h: u32,
    pat_w: u32,
    n_pairs: u32,
    grid_x: u32 = 1,
    flags: u32 = 0,
    pad0: u32 = 0,
    pad1: u32 = 0,
    pad2: u32 = 0,
    c0: [4]f32,
    c1: [4]f32,
    c2: [4]f32,
    mask: [4]f32,
};

const SoP = extern struct {
    n_items: u32,
    bw: u32,
    n: u32,
    nc: u32,
    mc: u32,
    mu_max: u32,
    fold: u32,
    l_step: u32,
    n_pairs: u32,
    grid_x: u32 = 1,
    rden_div: u32 = 1,
    nb: u32,
    log2nc: u32,
    pad0: u32 = 0,
    pad1: u32 = 0,
    pad2: u32 = 0,
};

/// The detector's attitude and mask, as the kernels take them (pixels of the binned pattern).
pub const Frame = struct {
    /// detector row direction, column direction and normal in the sample frame
    c0: [3]f32,
    c1: [3]f32,
    c2: [3]f32,
    pat_h: u32,
    pat_w: u32,
    /// mask centre (row, col), squared radius (≤ 0: none)
    mask: [3]f32,
};

/// One pattern under one geometry: the detector position for it (pixels) and the pattern's
/// index in the pattern buffer.
pub const Item = [4]f32;

pub const FLAG_CENTER: u32 = 1;
pub const FLAG_NORM: u32 = 2;
pub const FLAG_WINDOW: u32 = 4;

pub fn fftSize(L: u32) u32 {
    var n: u32 = 16;
    while (n < 2 * L - 1) n *= 2;
    return n;
}

pub const Tables = struct {
    L: u32,
    /// orientation grid points per turn; the sphere grid has n rings of 2n points
    n: u32,
    n_pairs: u32,
    ring: Buf,
    wplm: Buf,
    pairs: Buf,
    dq: Buf,
    tw: Buf,
    /// d^l_{k,m}(π/2), host copy for the phases' weight tables
    wigner: []f64,

    /// Pure allocation preflight, before any grid/Wigner generation. Sizes are
    /// individual storage bindings, not their combined resident memory.
    pub fn checkSize(L: u32, limit: u64) !void {
        if (L == 0 or L > TW_LEN / 2) return error.BandwidthExceedsDeviceLimit;
        const l: u64 = L;
        const n: u64 = fftSize(L);
        const pairs = l * (l + 1) / 2;
        for ([_]u64{ 2 * n * 4, pairs * (n / 2) * 4, pairs * 4, l * l * l * 4, TW_LEN * 8 }) |bytes| {
            if (bytes > limit) return error.BandwidthExceedsDeviceLimit;
        }
    }
    pub fn init(g: *Gpu, L: u32) !Tables {
        try checkSize(L, @min(g.lim.max_storage_binding, g.lim.max_buffer));
        const gpa = g.gpa;
        const n = fftSize(L);
        const n_theta = n;
        const n_phi = 2 * n;
        const np = sh.pairCount(L);
        var grid = try sh.Grid.init(gpa, L, n_theta, n_phi);
        defer grid.deinit(gpa);

        const ring_h = try gpa.alloc(f32, 2 * n_theta);
        defer gpa.free(ring_h);
        for (0..n_theta) |y| {
            ring_h[2 * y] = @floatCast(grid.x[y]);
            ring_h[2 * y + 1] = @floatCast(grid.w[y]);
        }
        // the northern rings only (the kernel folds each with its mirror), the pair innermost
        const wplm_h = try gpa.alloc(f32, @as(usize, np) * (n_theta / 2));
        defer gpa.free(wplm_h);
        const dphi = 2.0 * std.math.pi / @as(f64, @floatFromInt(n_phi));
        for (0..n_theta / 2) |y| for (0..np) |p| {
            wplm_h[y * np + p] = @floatCast(grid.w[y] * grid.plm[p * n_theta + y] * dphi);
        };
        const pairs_h = try gpa.alloc(u32, np);
        defer gpa.free(pairs_h);
        for (0..L) |m| for (m..L) |l| {
            pairs_h[sh.pair(L, @intCast(m), @intCast(l))] = (@as(u32, @intCast(m)) << 16) | @as(u32, @intCast(l));
        };
        const wig = try sh.wignerHalfPi(gpa, L);
        errdefer gpa.free(wig);
        // d^l_{k,m}(π/2) for k, m ≥ 0 at [(m · L + l) · L + k], zero where l < max(k, m): k
        // innermost, as in the masters' weight tables (master.weights says why)
        const dq_h = try gpa.alloc(f32, wig.len);
        defer gpa.free(dq_h);
        for (0..L) |m| for (0..L) |l| for (0..L) |k| {
            dq_h[(m * L + l) * L + k] = @floatCast(wig[(k * L + m) * L + l]);
        };
        var tw_h: [2 * TW_LEN]f32 = undefined;
        for (0..TW_LEN) |t| {
            const ang = -2.0 * std.math.pi * @as(f64, @floatFromInt(t)) / TW_LEN;
            tw_h[2 * t] = @floatCast(@cos(ang));
            tw_h[2 * t + 1] = @floatCast(@sin(ang));
        }

        var self: Tables = .{
            .L = L,
            .n = n,
            .n_pairs = np,
            .ring = g.storage(ring_h.len * 4),
            .wplm = g.storage(wplm_h.len * 4),
            .pairs = g.storage(pairs_h.len * 4),
            .dq = g.storage(dq_h.len * 4),
            .tw = g.storage(tw_h.len * 4),
            .wigner = wig,
        };
        g.writeSlice(self.ring, 0, f32, ring_h);
        g.writeSlice(self.wplm, 0, f32, wplm_h);
        g.writeSlice(self.pairs, 0, u32, pairs_h);
        g.writeSlice(self.dq, 0, f32, dq_h);
        g.writeSlice(self.tw, 0, f32, &tw_h);
        _ = &self;
        return self;
    }

    pub fn deinit(self: *Tables, g: *Gpu) void {
        g.release(&self.ring);
        g.release(&self.wplm);
        g.release(&self.pairs);
        g.release(&self.dq);
        g.release(&self.tw);
        g.gpa.free(self.wigner);
    }
};

pub const PhaseGpu = struct {
    fold: u32,
    nc: u32,
    mc: u32,
    mu_max: u32,
    l_step: u32,
    /// weight tables of the master and of its square
    w_f: Buf,
    w_f2: Buf,
    /// reciprocal denominator volume for the reference geometry
    rden: Buf = .{},

    /// Effective fold is the native FFT fold, which can differ from the
    /// master's metadata order. Reject before either CPU weight allocation.
    pub fn checkSize(L: u32, n: u32, fold: u32, limit: u64) !void {
        if (L == 0 or L > TW_LEN / 2 or fold == 0 or n == 0 or n % fold != 0) return error.BandwidthExceedsDeviceLimit;
        const l: u64 = L;
        const grid: u64 = n;
        const nc = grid / fold;
        const mc: u64 = 2 * ((L - 1) / fold) + 1;
        const weights = l * mc * l * 8;
        const spectrum = (grid / 2 + 1) * l * nc * 8;
        const denominator = (grid / 2 + 1) * grid * nc * 4;
        if (weights > limit or spectrum > limit or denominator > limit) return error.BandwidthExceedsDeviceLimit;
    }
    pub fn init(g: *Gpu, t: *const Tables, hm: *const master_mod.Harmonics) !PhaseGpu {
        const gpa = g.gpa;
        const fold = master_mod.foldOf(hm.rules);
        try checkSize(t.L, t.n, fold, @min(g.lim.max_storage_binding, g.lim.max_buffer));
        const mu_max = (t.L - 1) / fold;
        const wf = try master_mod.weights(gpa, t.L, hm.f, t.wigner, fold);
        defer gpa.free(wf);
        const wf2 = try master_mod.weights(gpa, t.L, hm.f2, t.wigner, fold);
        defer gpa.free(wf2);
        const self: PhaseGpu = .{
            .fold = fold,
            .nc = t.n / fold,
            .mc = 2 * mu_max + 1,
            .mu_max = mu_max,
            // the square of a master without odd l has none either
            .l_step = if (hm.rules.inversion) 2 else 1,
            .w_f = g.storage(wf.len * 4),
            .w_f2 = g.storage(wf2.len * 4),
        };
        g.writeSlice(self.w_f, 0, f32, wf);
        g.writeSlice(self.w_f2, 0, f32, wf2);
        return self;
    }

    pub fn deinit(self: *PhaseGpu, g: *Gpu) void {
        g.release(&self.w_f);
        g.release(&self.w_f2);
        g.release(&self.rden);
    }

    /// Cells of one (half) correlation volume.
    pub fn volumeCells(self: *const PhaseGpu, t: *const Tables) u64 {
        return @as(u64, t.n / 2 + 1) * t.n * self.nc;
    }

    /// Bytes of spectrum one item needs: half the b axis, m' ≥ 0, the master's orders.
    pub fn specBytes(self: *const PhaseGpu, t: *const Tables) u64 {
        return @as(u64, t.n / 2 + 1) * t.L * self.nc * 8;
    }
};

/// An FFT's parameters and its workgroup grid.
const Lines = struct { u: FftP, f: [2]u32 };

pub const Spherical = struct {
    g: *Gpu,
    /// the bandwidth's tables in use: the caller points this at them before recording (the
    /// work buffers below serve every bandwidth)
    t: *const Tables = undefined,
    /// invocations per workgroup of the FFT kernels (settings.wg_fft)
    wg: u32 = shaders.wg_default,
    items: Buf = .{},
    grid: Buf = .{},
    alm: Buf = .{},
    /// per item: window area, window mean, harmonic norm, a_00
    stats: Buf = .{},
    spec: Buf = .{},
    cand_v: Buf = .{},
    cand_i: Buf = .{},
    best: Buf = .{},
    /// per item: quaternion (w, x, y, z), score, a, b, c
    peaks: Buf = .{},
    candidate_cancel: ?*const @import("cancellation.zig").Flag = null,
    seed_selector: peak_select.Selector,
    seed_numerator: Buf = .{},
    seed_denominator: Buf = .{},
    seed_pages: Buf = .{},
    seed_page: Buf = .{},
    seed_interpolated: Buf = .{},
    // the window's side (denominators): its own coefficient and volume buffers
    w_alm: Buf = .{},
    w_stats: Buf = .{},
    vol_f: Buf = .{},
    vol_f2: Buf = .{},

    pub fn init(g: *Gpu) Spherical {
        return .{ .g = g, .seed_selector = peak_select.Selector.init(g) };
    }

    pub fn deinit(self: *Spherical) void {
        self.seed_selector.deinit();
        inline for (.{ "items", "grid", "alm", "stats", "spec", "cand_v", "cand_i", "best", "peaks", "w_alm", "w_stats", "vol_f", "vol_f2", "seed_numerator", "seed_denominator", "seed_pages", "seed_page", "seed_interpolated" }) |f| self.g.release(&@field(self, f));
    }

    fn pipe(self: *Spherical, comptime module: []const u8, comptime entry: []const u8) u32 {
        return self.g.pipeline(module ++ "/" ++ entry, @field(shaders, module), entry) catch gpu_mod.NO_PIPE;
    }

    /// A kernel built on the line FFT, with `wg` invocations to a workgroup.
    fn fftPipe(self: *Spherical, comptime module: []const u8, comptime entry: []const u8) u32 {
        return self.g.pipelineWith(module ++ "/" ++ entry, shaders.fft_wg, self.wg, @field(shaders, module ++ "_body"), entry) catch gpu_mod.NO_PIPE;
    }

    /// Forward FFT of `n_lines` lines of length n lying back to back in `data`, in place; the
    /// first `keep` points of each are written back.
    fn fftLines(self: *Spherical, data: Buf, n: u32, n_lines: u32, keep: u32) void {
        const g = self.g;
        const k = self.lineParams(n, n_lines);
        var u = k.u;
        u.keep = keep;
        g.dispatch(self.fftPipe("fft", "fft_lines"), k.f[0], k.f[1], 1, &.{ g.uniform(0, std.mem.asBytes(&u)), data.at(1), self.t.tw.at(2) });
    }

    /// The FFT's parameters and workgroup grid for `n_lines` lines of length n, as many whole
    /// lines to a workgroup as its memory holds.
    fn lineParams(self: *const Spherical, n: u32, n_lines: u32) Lines {
        _ = self;
        const lpw = TW_LEN / n;
        const f = Gpu.flat((n_lines + lpw - 1) / lpw);
        return .{ .u = .{ .n = n, .log2n = std.math.log2_int(u32, n), .n_lines = n_lines, .stored = n, .grid_x = f[0] }, .f = f };
    }

    /// Items a tile may hold for this phase within `budget` bytes of spectrum.
    pub fn tileItems(self: *const Spherical, ph: *const PhaseGpu, budget: u64) u32 {
        const lim = @min(budget, @min(self.g.lim.max_storage_binding, self.g.lim.max_buffer));
        const per = ph.specBytes(self.t);
        return @intCast(@max(1, @min(lim / per, 4096)));
    }

    /// Record: the harmonic coefficients of `items` (patterns from `pats`, or the detector's
    /// window itself with FLAG_WINDOW) into `alm` / `stats`.
    pub fn coefficients(self: *Spherical, pats: Buf, items: []const Item, fr: Frame, flags: u32, alm: *Buf, stats: *Buf) void {
        const g = self.g;
        const t = self.t;
        const n_items: u32 = @intCast(items.len);
        const n_theta = t.n;
        const n_phi = 2 * t.n;
        const per: u64 = @as(u64, n_theta) * n_phi;
        g.ensure(&self.items, @as(u64, n_items) * 16);
        g.ensure(&self.grid, n_items * per * 8);
        g.ensure(alm, @as(u64, n_items) * t.n_pairs * 8);
        g.ensure(stats, @as(u64, n_items) * 16);
        g.writeSlice(self.items, 0, Item, items);
        var p: SphP = .{
            .n_items = n_items,
            .n_theta = n_theta,
            .n_phi = n_phi,
            .bw = t.L,
            .pat_h = fr.pat_h,
            .pat_w = fr.pat_w,
            .n_pairs = t.n_pairs,
            .flags = flags,
            .c0 = .{ fr.c0[0], fr.c0[1], fr.c0[2], 0 },
            .c1 = .{ fr.c1[0], fr.c1[1], fr.c1[2], 0 },
            .c2 = .{ fr.c2[0], fr.c2[1], fr.c2[2], 0 },
            .mask = .{ fr.mask[0], fr.mask[1], fr.mask[2], 0 },
        };
        var f = Gpu.flat(@intCast((n_items * per + 255) / 256));
        p.grid_x = f[0];
        g.dispatch(self.pipe("sphere", "backproject"), f[0], f[1], 1, &.{ g.uniform(0, std.mem.asBytes(&p)), self.items.at(1), pats.at(2), t.ring.at(3), self.grid.at(4) });
        f = Gpu.flat(n_items);
        p.grid_x = f[0];
        g.dispatch(self.pipe("sphere", "window_center"), f[0], f[1], 1, &.{ g.uniform(0, std.mem.asBytes(&p)), t.ring.at(3), self.grid.at(4), stats.at(8) });
        // (the Legendre sums read orders below L only)
        self.fftLines(self.grid, n_phi, n_items * n_theta, t.L);
        f = Gpu.flat((n_items * t.n_pairs + 255) / 256);
        p.grid_x = f[0];
        g.dispatch(self.pipe("sphere", "legendre"), f[0], f[1], 1, &.{ g.uniform(0, std.mem.asBytes(&p)), self.grid.at(4), t.wplm.at(5), t.pairs.at(6), alm.at(7) });
        f = Gpu.flat(n_items);
        p.grid_x = f[0];
        g.dispatch(self.pipe("sphere", "sh_norm"), f[0], f[1], 1, &.{ g.uniform(0, std.mem.asBytes(&p)), t.pairs.at(6), alm.at(7), stats.at(8) });
    }

    fn soParams(self: *const Spherical, ph: *const PhaseGpu, n_items: u32) SoP {
        return .{ .n_items = n_items, .bw = self.t.L, .n = self.t.n, .nc = ph.nc, .mc = ph.mc, .mu_max = ph.mu_max, .fold = ph.fold, .l_step = ph.l_step, .n_pairs = self.t.n_pairs, .nb = self.t.n / 2 + 1, .log2nc = std.math.log2_int(u32, ph.nc) };
    }

    /// Record the spectrum of `n_items` coefficient sets against the weights `w`, transformed
    /// along b and then c, leaving V[j; m', c] in `spec`.
    fn spectrum(self: *Spherical, ph: *const PhaseGpu, w: Buf, alm: Buf, n_items: u32) void {
        const g = self.g;
        const t = self.t;
        g.ensure(&self.spec, n_items * ph.specBytes(t));
        const so = self.soParams(ph, n_items);
        const k = self.lineParams(t.n, n_items * t.L * ph.nc);
        const u = g.uniformPair(0, std.mem.asBytes(&k.u), 3, std.mem.asBytes(&so));
        g.dispatch(self.fftPipe("so3", "spectrum"), k.f[0], k.f[1], 1, &.{ u[0], t.tw.at(2), u[1], w.at(4), t.dq.at(5), alm.at(6), self.spec.at(7) });
        self.fftLines(self.spec, ph.nc, n_items * so.nb * t.L, ph.nc);
    }

    /// The last transform's parameters: the `n_items · nb · nc` lines along a, as a transform
    /// of half the grid's length.
    fn aParams(self: *const Spherical, ph: *const PhaseGpu, n_items: u32) Lines {
        return self.lineParams(self.t.n / 2, n_items * (self.t.n / 2 + 1) * ph.nc);
    }

    /// Record: correlate the `n_items` coefficient sets in `self.alm` with the phase and leave
    /// each item's best orientation in `peaks`. Item b is normalized by volume b / rden_div of
    /// `rden`.
    pub fn correlate(self: *Spherical, ph: *const PhaseGpu, n_items: u32, rden: Bind, rden_div: u32) void {
        const g = self.g;
        const t = self.t;
        const per = (t.n / 2 + 1) * ph.nc;
        self.spectrum(ph, ph.w_f, self.alm, n_items);
        g.ensure(&self.cand_v, @as(u64, n_items) * per * 4);
        g.ensure(&self.cand_i, @as(u64, n_items) * per * 4);
        g.ensure(&self.best, @as(u64, n_items) * 8);
        g.ensure(&self.peaks, @as(u64, n_items) * 32);
        var so = self.soParams(ph, n_items);
        so.rden_div = @max(rden_div, 1);
        var rd = rden;
        rd.slot = 8;
        const k = self.aParams(ph, n_items);
        const u = g.uniformPair(0, std.mem.asBytes(&k.u), 3, std.mem.asBytes(&so));
        g.dispatch(self.fftPipe("so3", "a_peak"), k.f[0], k.f[1], 1, &.{ u[0], t.tw.at(2), u[1], self.spec.at(7), rd, self.cand_v.at(9), self.cand_i.at(10) });
        var f = Gpu.flat(n_items);
        so.grid_x = f[0];
        g.dispatch(self.pipe("so3", "peak_reduce"), f[0], f[1], 1, &.{ g.uniform(3, std.mem.asBytes(&so)), self.cand_v.at(9), self.cand_i.at(10), self.best.at(11) });
        f = Gpu.flat((n_items + 63) / 64);
        so.grid_x = f[0];
        g.dispatch(self.pipe("so3", "peak_finish"), f[0], f[1], 1, &.{ g.uniform(3, std.mem.asBytes(&so)), self.spec.at(7), rd, self.best.at(11), self.peaks.at(17) });
    }

    /// Record: the correlation volumes of the `n_items` window coefficient sets in `w_alm`
    /// against weights `w`, into `out`.
    fn volume(self: *Spherical, ph: *const PhaseGpu, w: Buf, n_items: u32, out: Buf) void {
        const g = self.g;
        const t = self.t;
        self.spectrum(ph, w, self.w_alm, n_items);
        const so = self.soParams(ph, n_items);
        const k = self.aParams(ph, n_items);
        const u = g.uniformPair(0, std.mem.asBytes(&k.u), 3, std.mem.asBytes(&so));
        g.dispatch(self.fftPipe("so3", "a_volume"), k.f[0], k.f[1], 1, &.{ u[0], t.tw.at(2), u[1], self.spec.at(7), out.at(12) });
    }

    /// Record: the reciprocal denominator volumes of the normalized correlation for each window
    /// (a detector position in `windows`, pattern index unused) into `rden` (grown to hold them).
    pub fn denominators(self: *Spherical, ph: *const PhaseGpu, windows: []const Item, fr: Frame, rden: *Buf) void {
        const g = self.g;
        const t = self.t;
        const n: u32 = @intCast(windows.len);
        const cells = ph.volumeCells(t);
        // the pattern buffer is not read for a window, but the kernel binds one
        self.coefficients(t.ring, windows, fr, FLAG_WINDOW, &self.w_alm, &self.w_stats);
        g.ensure(&self.vol_f, n * cells * 4);
        g.ensure(&self.vol_f2, n * cells * 4);
        g.ensure(rden, n * cells * 4);
        self.volume(ph, ph.w_f, n, self.vol_f);
        self.volume(ph, ph.w_f2, n, self.vol_f2);
        var so = self.soParams(ph, n);
        const f = Gpu.flat(@intCast((n * cells + 255) / 256));
        so.grid_x = f[0];
        g.dispatch(self.pipe("so3", "rden_build"), f[0], f[1], 1, &.{ g.uniform(3, std.mem.asBytes(&so)), self.vol_f.at(13), self.vol_f2.at(14), self.w_stats.at(15), rden.at(16) });
    }
    fn candidateCancelled(self: *Spherical) !void {
        if (self.candidate_cancel) |flag| if (flag.get()) {
            try self.g.wait();
            return error.Cancelled;
        };
    }
    fn candidateSlab(self: *Spherical, ph: *const PhaseGpu, item: u32, first: u32, rows: u32, out: Buf) void {
        var so = self.soParams(ph, 1);
        so.pad0 = item;
        so.pad1 = first;
        so.pad2 = rows;
        const k = self.lineParams(self.t.n / 2, rows * ph.nc);
        const u = self.g.uniformPair(0, std.mem.asBytes(&k.u), 3, std.mem.asBytes(&so));
        self.g.dispatch(self.fftPipe("so3", "a_volume_slab"), k.f[0], k.f[1], 1, &.{ u[0], self.t.tw.at(2), u[1], self.spec.at(7), out.at(12) });
    }
    fn peakQuat(p: [8]f32) rot.Quat {
        return .{ p[0], p[1], p[2], p[3] };
    }
    /// Overlap-only seed pool. Rank-major native peak rows; rank zero is the
    /// exact ordinary SI peak. Extra ranks are normalized quotient-grid local
    /// maxima, refined with the ordinary interpolation, separated under the
    /// phase's proper LEFT rotation group. Missing ranks have score -2 and q=0.
    pub fn candidatePeaks(self: *Spherical, ph: *const PhaseGpu, n_items: u32, rden: Bind, rden_div: u32, ops: []const rot.Quat, keep_n: u32, out: [][8]f32) !void {
        if (keep_n == 0 or keep_n > 8 or out.len != @as(usize, n_items) * keep_n) return error.BadCandidateCount;
        const g = self.g;
        const t = self.t;
        const lim = @min(g.lim.max_storage_binding, g.lim.max_buffer);
        if (ph.specBytes(t) > lim or ph.volumeCells(t) * 4 > lim) return error.BandwidthExceedsDeviceLimit;
        if (@as(u64, n_items) * ph.specBytes(t) > lim) return error.SphericalCandidateTileTooLarge;
        try self.candidateCancelled();
        for (out) |*p| p.* = .{ 0, 0, 0, 0, -2, 0, 0, 0 };
        self.correlate(ph, n_items, rden, rden_div);
        g.read(self.peaks, 0, std.mem.sliceAsBytes(out[0..n_items]));
        try g.wait();
        if (keep_n == 1) return;
        const row_bytes = @as(u64, ph.nc) * t.n * 4;
        if (3 * row_bytes > lim) return error.BandwidthExceedsDeviceLimit;
        const core_rows: u32 = @intCast(@min(16, lim / row_bytes - 2));
        const nb = t.n / 2 + 1;
        const nslabs = (nb + core_rows - 1) / core_rows;
        g.ensure(&self.seed_pages, @as(u64, nslabs) * 64);
        g.ensure(&self.seed_page, 64);
        g.ensure(&self.seed_interpolated, 8 * 32);
        const rdiv = @max(1, rden_div);
        const spacing = 2 * std.math.tau / @as(f64, @floatFromInt(t.n));
        for (0..n_items) |item_index| {
            const item: u32 = @intCast(item_index);
            var count: u32 = 1;
            var after: ?peak_select.Key = null;
            while (count < keep_n) {
                try self.candidateCancelled();
                self.seed_selector.setCursor(after);
                var j: u32 = 0;
                var slab_index: u32 = 0;
                while (j < nb) : (j += core_rows) {
                    try self.candidateCancelled();
                    const first = j -| 1;
                    const end = @min(nb, j + core_rows + 1);
                    const rows = end - first;
                    const bytes = @as(u64, rows) * row_bytes;
                    g.ensure(&self.seed_numerator, bytes);
                    g.ensure(&self.seed_denominator, bytes);
                    self.candidateSlab(ph, item, first, rows, self.seed_numerator);
                    const offset = rden.offset + (@as(u64, item / rdiv) * nb + first) * row_bytes;
                    g.copy(.{ .id = rden.buf, .size = 0 }, offset, self.seed_denominator, 0, bytes);
                    self.seed_selector.slab(.{ .n = t.n, .nc = ph.nc, .nb = nb, .first = first, .rows = rows, .core_first = j, .core_rows = @min(core_rows, nb - j), .cursor_set = @intFromBool(after != null) }, self.seed_numerator, self.seed_denominator, self.seed_pages, @as(u64, slab_index) * 64);
                    slab_index += 1;
                }
                self.seed_selector.reduce(self.seed_pages, nslabs * 8, self.seed_page, 0);
                var page: [8]peak_select.Key = undefined;
                g.read(self.seed_page, 0, std.mem.sliceAsBytes(&page));
                try g.wait();
                var valid: usize = 0;
                for (page) |key| {
                    if (key.index == 0xffffffff) break;
                    valid += 1;
                }
                if (valid == 0) break;
                var so = self.soParams(ph, n_items);
                so.rden_div = rdiv;
                const grid = Gpu.flat((n_items + 63) / 64);
                so.grid_x = grid[0];
                var rd = rden;
                rd.slot = 8;
                for (page[0..valid], 0..) |key, rank| {
                    g.writeSlice(self.best, @as(u64, item) * 8, peak_select.Key, &.{key});
                    g.dispatch(self.pipe("so3", "peak_finish"), grid[0], grid[1], 1, &.{ g.uniform(3, std.mem.asBytes(&so)), self.spec.at(7), rd, self.best.at(11), self.peaks.at(17) });
                    g.copy(self.peaks, @as(u64, item) * 32, self.seed_interpolated, @as(u64, rank) * 32, 32);
                }
                var proposed: [8][8]f32 = undefined;
                g.read(self.seed_interpolated, 0, std.mem.sliceAsBytes(proposed[0..valid]));
                try g.wait();
                for (proposed[0..valid], page[0..valid]) |peak, key| {
                    after = key;
                    var distinct = true;
                    for (0..count) |rank| {
                        if (sym.disorientation(peakQuat(peak), peakQuat(out[rank * n_items + item]), ops) <= spacing) {
                            distinct = false;
                            break;
                        }
                    }
                    if (distinct) {
                        out[count * n_items + item] = peak;
                        count += 1;
                    }
                    if (count == keep_n) break;
                }
                if (valid < 8) break;
            }
        }
    }
};
