//! Independent fixed-support profiles share dispatches and scratch, while each
//! lane retains its own target, optical parameters and nonnegative fit.
const std = @import("std");
const core = @import("engine.zig");
const detector = @import("detector.zig");
const spherical = @import("spherical.zig");
const gpu = @import("gpu/gpu.zig");
const overlap = @import("overlap.zig");
const prep = @import("prep.zig");
const nuisance = @import("overlap_nuisance.zig");

pub const Mode = enum { fixed, optimize, both };
pub const Job = struct {
    row: usize,
    k: u32,
    poses: [4]detector.Pose,
    phases: [4]u32,
    seeds: nuisance.ProfileSeed,
    state: overlap.State,
    mode: Mode,
    optimized: ?struct { state: overlap.State, theta: nuisance.Theta } = null,
};

const Flight = struct {
    lanes: []usize,
    fixed: []overlap.State,
    optimized: []overlap.State,
    theta: []nuisance.Theta,
    ticket: u64 = 0,
};

/// Shared ceiling for the GPU workspace and the browser's progress tiles.
pub fn maxLanes(e: *const core.Engine, px: u32) u32 {
    const bytes = @as(u64, px) * @sizeOf(f32);
    const binding = @min(e.g.lim.max_storage_binding, e.g.lim.max_buffer);
    const budget = @as(u64, e.set.tile_mb) << 20;
    const clahe = if (e.set.clahe) prep.claheImageBytes(e.set.clahe_grid, e.set.clahe_bins) else 0;
    // Use the tuned (or default) batch as a ceiling. Both flights share
    // one workspace. Include the frozen targets, projections and CLAHE
    // histogram/CDF scratch from preparing the browser tile.
    const per_lane = @as(u64, px) * nuisance.profile_bytes_per_pixel +
        nuisance.profile_bytes_per_lane + 2 * bytes + @sizeOf(detector.Pose) + 2 * clahe;
    const largest_binding = @max(overlap.capacity * bytes, 196 * @sizeOf(f32), clahe);
    const limit = (binding / 16 * 16) / largest_binding;
    // As for spherical search, a single lane can exceed the soft budget
    // if it fits the hard allocation and binding limits.
    return @intCast(@min(@max(1, e.set.batch), limit, @max(1, budget / per_lane)));
}

pub const Batch = struct {
    engine: *core.Engine,
    lv: core.Level,
    frame: spherical.Frame,
    frozen: gpu.Buf,
    fit: *overlap.Fit,
    model: nuisance.Model,
    lanes: usize,
    optimize: bool,
    arena: std.heap.ArenaAllocator,
    poses: []detector.Pose,
    source_lanes: []usize,
    seeds: []nuisance.ProfileSeed,
    flights: [2]Flight,

    pub fn init(e: *core.Engine, lv: core.Level, frame: spherical.Frame, frozen: gpu.Buf, max_lanes: usize, optimize: bool) !Batch {
        const g = &e.g;
        const px = frame.pat_h * frame.pat_w;
        const lanes = @min(max_lanes, maxLanes(e, px));
        if (lanes == 0) return error.OverlapBatchTooLargeUseSmallerBatches;

        var arena = std.heap.ArenaAllocator.init(e.gpa);
        errdefer arena.deinit();
        const a = arena.allocator();
        const poses = try a.alloc(detector.Pose, lanes);
        const source_lanes = try a.alloc(usize, lanes);
        const seeds = try a.alloc(nuisance.ProfileSeed, lanes);
        var flights: [2]Flight = undefined;
        for (&flights) |*f| f.* = .{
            .lanes = try a.alloc(usize, lanes),
            .fixed = try a.alloc(overlap.State, lanes),
            .optimized = try a.alloc(overlap.State, lanes),
            .theta = try a.alloc(nuisance.Theta, lanes),
        };
        const fit = try a.create(overlap.Fit);
        fit.* = overlap.Fit.allocate(g, @intCast(lanes), px, frame.pat_w, frame.mask);
        const model = nuisance.Model.allocate(g, &e.prep, fit);
        return .{
            .engine = e,
            .lv = lv,
            .frame = frame,
            .frozen = frozen,
            .fit = fit,
            .model = model,
            .lanes = lanes,
            .optimize = optimize,
            .arena = arena,
            .poses = poses,
            .source_lanes = source_lanes,
            .seeds = seeds,
            .flights = flights,
        };
    }

    pub fn deinit(self: *Batch) void {
        self.model.deinit();
        self.fit.deinit();
        self.arena.deinit();
    }

    fn stage(self: *Batch, jobs: []const Job, lanes: []const usize, k: u32) void {
        const e = self.engine;
        const g = &e.g;
        const n = lanes.len;
        self.fit.n = @intCast(n);
        self.model.trial.n = @intCast(n);
        self.model.contrast = true;
        self.model.optimize_enabled = self.optimize;
        const bytes = @as(u64, self.fit.px) * @sizeOf(f32);
        for (lanes, 0..) |job, lane| {
            g.copy(self.frozen, jobs[job].row * bytes, self.fit.original, lane * bytes, bytes);
            self.seeds[lane] = jobs[job].seeds;
        }
        self.fit.resetTarget(self.fit.original);
        self.model.resetProfileTarget();

        for (0..k) |column| {
            for (e.phases.items, 0..) |phase, pi| {
                var count: usize = 0;
                for (lanes, 0..) |job, lane| {
                    if (jobs[job].phases[column] != pi) continue;
                    self.poses[count] = jobs[job].poses[column];
                    self.source_lanes[count] = lane;
                    count += 1;
                }
                if (count == 0) continue;
                e.det.setPoses(self.poses[0..count]);
                e.det.project(phase.det[@intFromEnum(self.lv)], self.frame, @intCast(count));
                for (self.source_lanes[0..count], 0..) |lane, source| {
                    g.copy(e.det.sims, source * bytes, self.model.raw, (column * n + lane) * bytes, bytes);
                }
            }
        }

        self.model.profileRawChanged(k);
        self.model.stageProfileSeeds(self.seeds[0..n]);
    }

    /// Generate signed normalized residuals from each lane's accepted support.
    /// The caller compacts live cells and searches these prepared rows next.
    pub fn residual(self: *Batch, jobs: []const Job, out: gpu.Buf) void {
        const n = jobs.len;
        std.debug.assert(n > 0 and n <= self.lanes);
        const k = jobs[0].k;
        const f = &self.flights[0];
        for (jobs, 0..) |job, i| {
            std.debug.assert(job.k == k);
            f.lanes[i] = i;
            f.fixed[i] = job.state;
        }
        self.stage(jobs, f.lanes[0..n], k);
        self.model.prepareWarmTheta();
        const g = &self.engine.g;
        g.copy(self.model.trial.bank, 0, self.fit.bank, 0, @as(u64, k) * n * self.fit.px * @sizeOf(f32));
        g.writeSlice(self.fit.state, 0, overlap.State, f.fixed[0..n]);
        self.fit.residual(k, out, true);
    }

    fn record(self: *Batch, jobs: []const Job, flight: *Flight, n: usize, k: u32, mode: Mode) void {
        self.stage(jobs, flight.lanes[0..n], k);
        const g = &self.engine.g;
        if (mode != .optimize) self.model.fixedProfileBatch(flight.fixed[0..n]);
        if (mode != .fixed and self.optimize) self.model.optimizeProfileBatch(flight.optimized[0..n], flight.theta[0..n]);
        flight.ticket = g.readTicket();
        g.submit();
    }

    fn land(self: *Batch, jobs: []Job, flight: Flight, n: usize, mode: Mode) !void {
        try self.engine.g.waitFor(flight.ticket);
        for (flight.lanes[0..n], 0..) |index, lane| {
            const job = &jobs[index];
            if (mode != .optimize) job.state = flight.fixed[lane];
            if (mode != .fixed and self.optimize) job.optimized = .{ .state = flight.optimized[lane], .theta = flight.theta[lane] };
        }
    }

    /// Compact one support order and operation at a time, recording the next
    /// chunk before waiting for the previous read ticket, as in primary indexing.
    pub fn run(self: *Batch, jobs: []Job) !void {
        const e = self.engine;
        errdefer e.g.wait() catch {};
        for (1..overlap.capacity + 1) |k| {
            inline for (.{ Mode.fixed, Mode.optimize, Mode.both }) |mode| {
                if (mode != .optimize or self.optimize) try self.runGroup(jobs, @intCast(k), mode);
            }
        }
    }

    fn runGroup(self: *Batch, jobs: []Job, k: u32, mode: Mode) !void {
        var at: usize = 0;
        var slot: usize = 0;
        var pending: usize = 0;
        while (at < jobs.len) {
            if (self.engine.cancel.get()) return error.Cancelled;
            const f = &self.flights[slot];
            var n: usize = 0;
            while (at < jobs.len and n < self.lanes) : (at += 1) {
                const job = &jobs[at];
                if (job.k != k or job.mode != mode) continue;
                job.optimized = null;
                f.lanes[n] = at;
                n += 1;
            }
            if (n == 0) break;
            self.record(jobs, f, n, k, mode);
            slot = 1 - slot;
            if (pending != 0) try self.land(jobs, self.flights[slot], pending, mode);
            pending = n;
        }
        if (pending != 0) try self.land(jobs, self.flights[1 - slot], pending, mode);
        if (self.engine.cancel.get()) return error.Cancelled;
    }
};
