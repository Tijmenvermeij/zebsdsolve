//! Bounded shared blur/gain variable projection. Host schedules trials; images,
//! trial decisions, NNLS and parameter updates remain on the GPU.
const std = @import("std");
const gpu = @import("gpu/gpu.zig");
const shaders = @import("shaders");
const overlap = @import("overlap.zig");
const Prep = @import("prep.zig").Prep;
pub const Theta = [7]f32;
pub const identity: Theta = .{ 0, 0, 1, 1, 1, 0, 0 };
pub const max_evaluations: u32 = 104;
const P = extern struct {
    n: u32,
    px: u32,
    width: u32,
    grid_x: u32 = 1,
    mode: u32 = 0,
    coord: u32 = 0,
    phase: u32 = 0,
    seed: u32 = 0,
    step: f32 = 0,
    dc: f32 = 0,
    k: u32 = 0,
    contrast: u32 = 0,
    mask: [4]f32 = @splat(0),
};
pub const Model = struct {
    g: *gpu.Gpu,
    prep: *Prep,
    fit: *overlap.Fit,
    trial: overlap.Fit,
    raw: gpu.Buf,
    committed: gpu.Buf,
    centered: gpu.Buf,
    mid: gpu.Buf,
    blurred: gpu.Buf,
    cells: gpu.Buf,
    best: gpu.Buf,
    best_bank: gpu.Buf,
    theta: gpu.Buf,
    norms: gpu.Buf,
    contrast: bool = false,
    optimize_enabled: bool = true,
    k: u32 = 0,
    single_column: u32 = 0,
    pub fn init(g: *gpu.Gpu, prep: *Prep, fit: *overlap.Fit) Model {
        const bytes = @as(u64, 4) * fit.n * fit.px * 4;
        var m: Model = .{ .g = g, .prep = prep, .fit = fit, .trial = overlap.Fit.init(g, fit.n, fit.px, fit.width, fit.mask, fit.original), .raw = g.storage(bytes), .committed = g.storage(bytes), .centered = g.storage(bytes), .mid = g.storage(bytes), .blurred = g.storage(bytes), .cells = g.storage(@as(u64, fit.n) * 196 * 4), .best = g.storage(@as(u64, fit.n) * @sizeOf(overlap.State)), .best_bank = g.storage(bytes), .theta = g.storage(@as(u64, fit.n) * 7 * 4), .norms = g.storage(@as(u64, fit.n) * 4 * 4) };
        g.clear(m.cells);
        g.clear(m.raw);
        g.clear(m.committed);
        m.run("initialize", .{}, &.{ m.cells.at(5), fit.state.at(8) }, false);
        return m;
    }
    pub fn deinit(self: *Model) void {
        self.trial.deinit();
        inline for (.{ "raw", "committed", "centered", "mid", "blurred", "cells", "best", "best_bank", "theta", "norms" }) |field| self.g.release(&@field(self, field));
    }
    const Args = struct { mode: u32 = 0, coord: u32 = 0, phase: u32 = 0, seed: u32 = 0, step: f32 = 0, dc: f32 = 0, reuse_blur: bool = false };
    fn run(self: *Model, comptime entry: []const u8, a: Args, binds: []const gpu.Bind, pixels: bool) void {
        const f = self.fit;
        const count = if (pixels) (@as(u64, 4) * f.n * f.px + 255) / 256 else f.n;
        const grid = gpu.Gpu.flat(@intCast(count));
        const p: P = .{ .n = f.n, .px = f.px, .width = f.width, .grid_x = grid[0], .mode = a.mode, .coord = a.coord, .phase = a.phase, .seed = a.seed, .step = a.step, .dc = a.dc, .k = self.k, .contrast = @intFromBool(self.contrast), .mask = .{ f.mask[0], f.mask[1], f.mask[2], 0 } };
        var all: [10]gpu.Bind = undefined;
        all[0] = self.g.uniform(0, std.mem.asBytes(&p));
        @memcpy(all[1..][0..binds.len], binds);
        const pipe = self.g.pipeline("overlap_nuisance/" ++ entry, shaders.overlap_nuisance, entry) catch gpu.NO_PIPE;
        self.g.dispatch(pipe, grid[0], grid[1], 1, all[0 .. binds.len + 1]);
    }
    pub fn begin(self: *Model) void {
        self.run("begin_order", .{}, &.{ self.cells.at(5), self.fit.state.at(8) }, false);
    }
    fn transform(self: *Model, singleton: bool) void {
        self.transformWithArgs(.{ .mode = @intFromBool(singleton), .coord = self.single_column }, true);
    }
    fn centerRaw(self: *Model) void {
        // A full-image constant shift is exact preconditioning for the
        // constant-preserving Gaussian followed by masked centering. Keep a
        // separate dispatch/buffer so large DC cannot be reassociated into gain.
        if (self.contrast) {
            self.run("raw_median", .{}, &.{ self.raw.at(1), self.cells.at(5) }, false);
            self.run("center_raw", .{}, &.{ self.raw.at(1), self.centered.at(15), self.cells.at(5) }, true);
        }
    }
    fn transformWithArgs(self: *Model, a: Args, normalize: bool) void {
        if (!a.reuse_blur) {
            const source = if (self.contrast) self.centered else self.raw;
            self.run("blur_rows", a, &.{ source.at(1), self.mid.at(3), self.cells.at(5), self.fit.progress.at(9) }, true);
            self.run("blur_cols", a, &.{ self.mid.at(3), self.blurred.at(4), self.cells.at(5), self.fit.progress.at(9) }, true);
            self.run("cache_sigma", a, &.{self.cells.at(5)}, false);
            if (self.contrast) self.run("mean_blurred", a, &.{ self.blurred.at(4), self.cells.at(5), self.fit.progress.at(9) }, false);
        }
        self.run("gain", a, &.{ self.trial.bank.at(2), self.blurred.at(4), self.cells.at(5), self.fit.progress.at(9) }, true);
        if (normalize) {
            // Profiled trials have a known column count and use only this prefix.
            // Batched discovery may have a different count for each cell.
            const columns = if (self.k == 0) 4 else if (a.mode == 1) 1 else self.k;
            self.prep.filter(self.trial.bank, columns * self.fit.n, self.fit.px / self.fit.width, self.fit.width, .{ .mask = self.fit.mask });
        }
    }
    fn evaluate(self: *Model, a: Args, singleton: bool) void {
        self.run("propose", a, &.{ self.cells.at(5), self.trial.state.at(6), self.fit.state.at(8), self.fit.progress.at(9) }, false);
        self.transformWithArgs(.{ .mode = @intFromBool(singleton), .coord = self.single_column, .reuse_blur = a.reuse_blur }, true);
        self.trial.contrast = self.contrast;
        self.trial.nuisanceTrial(singleton, self.k);
        self.run("retain", .{ .mode = @intFromBool(singleton), .seed = a.seed }, &.{ self.trial.bank.at(2), self.cells.at(5), self.trial.state.at(6), self.best.at(7), self.fit.progress.at(9), self.best_bank.at(10) }, false);
    }
    // A sigma trial may leave its rejected blur in the cache. Refresh at
    // the first gain trial, then every remaining gain trial uses the same sigma
    // whether or not its gain is retained. Starting from new seeds also refreshes.
    fn coordinatePair(self: *Model, singleton: bool, coord: u32, step: f32, reuse_first: bool) void {
        self.evaluate(.{ .seed = 8, .coord = coord, .step = step, .reuse_blur = reuse_first }, singleton);
        self.evaluate(.{ .seed = 8, .coord = coord, .step = -step, .reuse_blur = coord != 0 }, singleton);
    }
    fn optimize(self: *Model, singleton: bool) void {
        self.run("reset_search", .{}, &.{ self.cells.at(5), self.best.at(7), self.fit.state.at(8) }, false);
        if (!self.optimize_enabled) {
            self.evaluate(.{ .seed = 0 }, singleton);
            return;
        }
        const seeds: u32 = if (singleton) 4 else 8;
        for (0..seeds) |seed| self.evaluate(.{ .seed = @intCast(seed) }, singleton);
        const steps: Theta = .{ 1.5, 0.35, 2, 0.2, 0.2, 0.06, 0.06 };
        for ([_]f32{ 1, 0.5, 0.25 }) |scale| for (0..2) |_| for (steps, 0..) |step, j| {
            self.coordinatePair(singleton, @intCast(j), step * scale, j > 1);
        };
        if (singleton) {
            // Keep the current search's coarse singleton seeds for the joint
            // trajectory; polished singleton scores still define its baseline.
            self.run("save_coarse_single", .{ .coord = self.single_column }, &.{self.cells.at(5)}, false);
        } else {
            // Include every polished singleton as a feasible joint candidate,
            // after preserving the complete coarse joint trajectory.
            for (9..13) |seed| self.evaluate(.{ .seed = @intCast(seed) }, false);
        }
        // Resolve weak radial gain without resetting the achieved NNLS incumbent.
        for ([_]f32{ 0.04375, 0.021875, 0.0109375, 0.00546875 }, 0..) |step, j| {
            self.coordinatePair(singleton, 1, step, j > 0);
        }
    }
    pub fn candidate(self: *Model, patterns: gpu.Buf, phase: u32, dc: f32, seed: bool) void {
        self.candidateWithID(patterns, phase, dc, seed, phase);
    }
    pub fn candidateWithID(self: *Model, patterns: gpu.Buf, phase: u32, dc: f32, seed: bool, candidate_id: u32) void {
        self.g.copy(self.fit.progress, 0, self.trial.progress, 0, @as(u64, self.fit.n) * @sizeOf(overlap.Progress));
        self.run("stage", .{ .dc = dc }, &.{ self.raw.at(1), self.cells.at(5), self.fit.progress.at(9), self.committed.at(11), patterns.at(13) }, false);
        self.run("validate", .{}, &.{ self.raw.at(1), self.cells.at(5) }, false);
        self.centerRaw();
        self.single_column = 0;
        // Discovery singleton is the proposed final column.
        self.run("set_single_column", .{}, &.{ self.cells.at(5), self.fit.progress.at(9) }, false);
        self.optimize(true);
        self.run("save_single", .{}, &.{ self.cells.at(5), self.best.at(7) }, false);
        self.optimize(false);
        self.run("commit", .{ .phase = phase, .seed = @intFromBool(seed), .coord = candidate_id }, &.{ self.raw.at(1), self.fit.bank.at(2), self.cells.at(5), self.best.at(7), self.fit.state.at(8), self.fit.progress.at(9), self.best_bank.at(10), self.committed.at(11) }, false);
    }
    /// Recompute the unnormalized final contrast norms without touching fit.bank.
    /// Discovery uses only the committed raw bank; rejected staged columns cannot leak.
    pub fn readNorms(self: *Model, out: [][4]f32, committed: bool) void {
        self.run("prepare_final", .{ .seed = @intFromBool(committed) }, &.{ self.raw.at(1), self.cells.at(5), self.fit.progress.at(9), self.committed.at(11) }, false);
        self.run("validate", .{}, &.{ self.raw.at(1), self.cells.at(5) }, false);
        self.centerRaw();
        self.transformWithArgs(.{ .mode = 2 }, false);
        self.run("component_norms", .{ .mode = 2 }, &.{ self.trial.bank.at(2), self.fit.progress.at(9), self.norms.at(14) }, false);
        self.g.read(self.norms, 0, std.mem.sliceAsBytes(out));
    }
    pub fn fixed(self: *Model, templates: []const f32, k: u32) void {
        self.k = k;
        self.g.writeSlice(self.raw, 0, f32, templates);
        self.g.copy(self.fit.progress, 0, self.trial.progress, 0, @as(u64, self.fit.n) * @sizeOf(overlap.Progress));
        self.run("validate", .{}, &.{ self.raw.at(1), self.cells.at(5) }, false);
        self.centerRaw();
        for (0..k) |j| {
            self.single_column = @intCast(j);
            self.optimize(true);
            self.run("store_fixed_single", .{ .coord = @intCast(j) }, &.{ self.cells.at(5), self.best.at(7) }, false);
        }
        self.optimize(false);
        self.run("finish_fixed", .{}, &.{ self.cells.at(5), self.best.at(7), self.fit.state.at(8) }, false);
        self.g.copy(self.best_bank, 0, self.fit.bank, 0, @as(u64, 4) * self.fit.n * self.fit.px * 4);
    }
    pub fn readWinnerIDs(self: *Model, out: []f32) void {
        self.run("read_winner_ids", .{}, &.{ self.cells.at(5), self.theta.at(12) }, false);
        self.g.read(self.theta, 0, std.mem.sliceAsBytes(out));
    }
    pub fn readTheta(self: *Model, out: []Theta) void {
        self.run("read_theta", .{}, &.{ self.cells.at(5), self.theta.at(12) }, false);
        self.g.read(self.theta, 0, std.mem.sliceAsBytes(out));
    }
    // These V6 entry points operate on one cell, with a fixed positive k. The
    // controller projects into raw with Detector.project. Raw and blur-cache
    // invalidation form one operation even if the next theta has the same sigma.
    pub fn profileRawChanged(self: *Model, k: u32) void {
        std.debug.assert(self.contrast and k > 0 and k <= 4);
        self.k = k;
        if (self.fit.n == 1) {
            self.g.writeSlice(self.cells, 65 * 4, f32, &.{-1});
        } else {
            self.run("invalidate_sigma", .{}, &.{self.cells.at(5)}, false);
        }
        self.run("validate", .{}, &.{ self.raw.at(1), self.cells.at(5) }, false);
        self.centerRaw();
    }

    /// Exactly one shared-theta evaluation, with the V4 GPU active-set NNLS.
    /// Both its state and its normalized columns belong to this particular pose.
    pub fn profileFixedTheta(self: *Model, theta: Theta) void {
        self.g.writeSlice(self.cells, 24 * 4, f32, &theta);
        self.fixedWarmTheta();
    }

    fn fixedWarmTheta(self: *Model) void {
        self.run("propose", .{ .seed = 1 }, &.{ self.cells.at(5), self.trial.state.at(6), self.fit.state.at(8), self.fit.progress.at(9) }, false);
        self.transform(false);
        self.trial.contrast = true;
        self.trial.nuisanceTrial(false, self.k);
    }

    fn optimizePrimaryTheta(self: *Model) void {
        self.single_column = 0;
        self.optimize(true);
        self.run("store_fixed_single", .{}, &.{ self.cells.at(5), self.best.at(7) }, false);
        // K1 uses its own singleton seed in every unoccupied slot.
        self.run("share_primary_theta", .{}, &.{self.cells.at(5)}, false);
        self.optimize(false);
    }

    /// Independent singleton seeds start with identity theta. Launch each
    /// trial across the batch with one independent reduction per detector pattern.
    pub fn singletonSeeds(self: *Model, fixed_states: []overlap.State, optimized_states: []overlap.State, theta: []Theta) void {
        std.debug.assert(self.fit.n > 1);
        self.profileRawChanged(1);
        self.fixedWarmTheta();
        self.trial.read(fixed_states);
        if (self.optimize_enabled) {
            self.optimizePrimaryTheta();
            self.run("read_theta", .{ .mode = 1 }, &.{ self.cells.at(5), self.theta.at(12) }, false);
            self.g.read(self.best, 0, std.mem.sliceAsBytes(optimized_states));
            self.g.read(self.theta, 0, std.mem.sliceAsBytes(theta));
        }
    }

    /// Full V4 bounded shared-theta search at a new support or at the final
    /// pose. A short search between angular levels always starts at the achieved
    /// theta. Per-member singleton nuisance seeds are isolated to this support.
    pub fn profileNuisance(self: *Model, warm: Theta, singles: [4]Theta, full: bool) void {
        self.g.writeSlice(self.cells, 24 * 4, f32, &warm);
        if (full) {
            self.g.writeSlice(self.cells, 32 * 4, Theta, &singles);
            self.g.writeSlice(self.cells, 90 * 4, Theta, &singles);
            if (self.k == 1) self.optimizePrimaryTheta() else self.optimize(false);
        } else {
            self.run("reset_search", .{}, &.{ self.cells.at(5), self.best.at(7), self.fit.state.at(8) }, false);
            self.evaluate(.{ .seed = 1 }, false);
            const steps: Theta = .{ 0.375, 0.04375, 0.5, 0.05, 0.05, 0.015, 0.015 };
            for (steps, 0..) |step, j| {
                self.coordinatePair(false, @intCast(j), step, j > 1);
            }
        }
    }

    /// Read a single achieved state and theta together, after profileNuisance.
    pub fn readProfile(self: *Model, state: *overlap.State, theta: *Theta) void {
        self.g.read(self.best, 0, std.mem.asBytes(state));
        self.g.read(self.cells, 8 * 4, std.mem.asBytes(theta));
    }

    pub fn readProfileNorms(self: *Model, theta: Theta, out: *[4]f32) void {
        self.g.writeSlice(self.cells, 0, f32, &theta);
        self.transformWithArgs(.{}, false);
        self.run("component_norms", .{}, &.{ self.trial.bank.at(2), self.fit.progress.at(9), self.norms.at(14) }, false);
        self.g.read(self.norms, 0, std.mem.asBytes(out));
    }

    /// Exact inspector transformation with the exported fixed parameters.
    pub fn apply(self: *Model, patterns: gpu.Buf, dc: f32, theta: Theta) gpu.Buf {
        self.k = 1;
        self.g.copy(patterns, 0, self.raw, 0, @as(u64, self.fit.n) * self.fit.px * 4);
        self.g.writeSlice(self.cells, 0, f32, &theta);
        self.g.writeSlice(self.cells, 64 * 4, f32, &.{ dc, -1 });
        self.run("validate", .{}, &.{ self.raw.at(1), self.cells.at(5) }, false);
        self.centerRaw();
        self.transform(false);
        return self.trial.bank;
    }
};
