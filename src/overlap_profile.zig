//! Opt-in V6 overlap model competition. All measured pixels, detector
//! projections, contrast transfer, NNLS, residuals and nuisance fits stay on
//! the GPU. The host compares small, complete achieved support records.
const std = @import("std");
const core = @import("engine.zig");
const detector = @import("detector.zig");
const spherical = @import("spherical.zig");
const gpu = @import("gpu/gpu.zig");
const shaders = @import("shaders");
const overlap = @import("overlap.zig");
const nuisance = @import("overlap_nuisance.zig");
const batched = @import("overlap_profile_batch.zig");
const Theta = nuisance.Theta;
const Quat = [4]f32;
const nan = std.math.nan(f32);
const unselected = std.math.maxInt(u32);

pub const WireOptions = extern struct {
    size: u32,
    flags: u32,
    radius_deg: f32,
    min_residual_ncc: f32,
};

pub const WireStageOptions = extern struct {
    base: WireOptions,
    min_stage_improvement: [3]f32,
};

pub const refit_after_polish: u32 = 8;

pub const Options = struct {
    flags: u32 = 1, // 1 polish pose, 2 screen residual NCC, 4 emit publication witness, 8 refit blur/gain after polishing
    radius_deg: f32 = 1,
    min_residual_ncc: f32 = 0,
    min_stage_improvement: ?[3]f32 = null,

    pub fn decode(wire: *const WireOptions, k: u32) ?Options {
        var options: Options = .{
            .flags = wire.flags,
            .radius_deg = wire.radius_deg,
            .min_residual_ncc = wire.min_residual_ncc,
        };
        switch (wire.size) {
            @sizeOf(WireOptions) => {},
            @sizeOf(WireStageOptions) => {
                if (k < 2 or k > 4) return null;
                const staged: *const WireStageOptions = @ptrCast(wire);
                options.min_stage_improvement = staged.min_stage_improvement;
                for (staged.min_stage_improvement[k - 1 ..]) |v| if (v != 0) return null;
            },
            else => return null,
        }
        return if (options.valid()) options else null;
    }

    pub fn valid(self: Options) bool {
        if (self.min_stage_improvement) |limits| {
            if (self.flags & 2 == 0) return false;
            for (limits) |v| if (!std.math.isFinite(v) or v < 0 or v > 1) return false;
        }
        return self.flags & 3 != 0 and self.flags & ~@as(u32, 15) == 0 and
            (self.flags & refit_after_polish == 0 or self.flags & 1 != 0) and
            std.math.isFinite(self.radius_deg) and self.radius_deg > 0 and self.radius_deg <= 5 and
            std.math.isFinite(self.min_residual_ncc) and @abs(self.min_residual_ncc) <= 1;
    }
};

pub const Event = extern struct {
    cell_index: u32,
    branch: u32, // 0 exact N=1, 1 pooled
    order: u32,
    attempt: u32,
    phase: i32,
    rank: i32,
    flags: u32 = 0,
    result_order: u32 = 0,
    source_q: Quat,
    residual_ncc: f32 = nan,
    baseline_sse: f32,
    lower_sse: f32 = nan,
    joint_sse: f32 = nan,
};
pub const flag = struct {
    pub const native: u32 = 1;
    pub const screened: u32 = 2;
    pub const lower: u32 = 4;
    pub const joint: u32 = 8;
    pub const replace: u32 = 16;
    pub const add: u32 = 32;
    pub const boundary: u32 = 64;
    pub const chosen_branch: u32 = 128;
    pub const published: u32 = 256;
    pub const published_lower: u32 = 512;
    pub const stage_checked: u32 = 1024;
    pub const stage_passed: u32 = 2048;
};
comptime {
    if (@sizeOf(WireOptions) != 16 or @sizeOf(WireStageOptions) != 28 or @sizeOf(Event) != 64 or @offsetOf(Event, "source_q") != 32 or @offsetOf(Event, "residual_ncc") != 48)
        @compileError("V6 overlap ABI layout mismatch");
}

const max_replacements = 2;

/// Each lower-order replacement can cause a return to a previously visited
/// order. There are at most 2*(K-1) replacements, each possibly followed by a
/// fresh traversal to K. Include both histories, an entry for *every* attempted
/// phase/rank, and check overflow before allocation or numerical work.
pub fn traceCapacity(n: usize, phases: usize, k: u32, keep_n: u32) !usize {
    if (k == 0 or k > 4 or keep_n == 0 or keep_n > 8 or phases == 0 or n > std.math.maxInt(u32)) return error.BadLength;
    const levels: usize = k - 1;
    const residual_rounds = levels * (1 + max_replacements * levels);
    const per_phase = if (keep_n > 1 and k > 1)
        2 + residual_rounds * (keep_n + 1)
    else
        1 + residual_rounds;
    return std.math.mul(usize, try std.math.mul(usize, n, phases), per_phase);
}

pub const Request = struct {
    options: Options,
    source_q: []Quat,
    events: []Event,
    used: usize = 0,
};

fn finiteScore(x: f32) bool {
    return std.math.isFinite(x) and x >= 0 and x < 1e20;
}

/// A rounding tie on a direct pixel-space SSE is resolved by keeping the
/// incumbent. The energy-proportional floor matters only very close to zero.
pub fn improves(a: f32, b: f32, initial: f32) bool {
    if (!finiteScore(a)) return false;
    if (!finiteScore(b)) return true;
    const scale = @max(@abs(a), @abs(b), initial * 1e-12);
    return b - a > 8 * std.math.floatEps(f32) * scale;
}

fn assessed(s: overlap.State) bool {
    return finiteScore(s.sse) and finiteScore(s.initial_sse) and s.initial_sse > 1e-12 and
        std.math.isFinite(s.flags) and s.flags >= 0 and s.flags < 32 and
        (@as(u32, @intFromFloat(s.flags)) & (2 | 8)) == 0;
}

const Member = struct {
    source: overlap.Component = .{},
    rank: i32 = -1,
    refined: ?Quat = null,
    single_theta: Theta = nuisance.identity,
    single_sse: f32 = nan,

    fn indexed(record: core.Record, phase: usize, rank: i32) Member {
        return .{
            .source = .{ .q = record.q, .phase = @intCast(phase), .proposal_ncc = record.ncc, .proposal_nmi = record.nmi },
            .rank = rank,
        };
    }

    fn valid(self: Member) bool {
        var squared: f32 = 0;
        for (self.source.q) |v| squared += v * v;
        const ncc = self.source.proposal_ncc;
        return self.source.phase >= 0 and std.math.isFinite(squared) and @abs(squared - 1) <= 0.01 and
            std.math.isFinite(ncc) and ncc >= -1 and ncc <= 1.00001 and std.math.isFinite(self.source.proposal_nmi);
    }

    fn pose(self: Member) Quat {
        return self.refined orelse self.source.q;
    }

    fn sameProjection(self: Member, other: Member) bool {
        const a = self.pose();
        const b = other.pose();
        return self.source.phase == other.source.phase and std.mem.eql(f32, &a, &b);
    }
};

const Support = struct {
    k: u32 = 0,
    members: [4]Member = @splat(.{}),
    state: overlap.State = .{ .sse = nan },
    theta: Theta = nuisance.identity,

    fn append(self: Support, member: Member) Support {
        var s = self;
        s.members[s.k] = member;
        s.k += 1;
        s.state.sse = nan;
        return s;
    }

    fn subset(self: Support, mask: u32) Support {
        var s: Support = .{ .theta = self.theta };
        for (0..self.k) |j| if (mask & (@as(u32, 1) << @intCast(j)) != 0) {
            s = s.append(self.members[j]);
        };
        return s;
    }

    fn activeMask(self: Support) u32 {
        var mask: u32 = 0;
        for (self.state.amplitude[0..self.k], 0..) |a, j| if (a > 0) {
            mask |= @as(u32, 1) << @intCast(j);
        };
        return mask;
    }

    fn positive(self: Support) bool {
        return self.k > 0 and assessed(self.state) and @popCount(self.activeMask()) == self.k;
    }
};

const Achieved = struct {
    support: Support,
    event: usize,
    from_lower: bool,
};

pub const Archive = struct {
    by_order: [5]?Achieved = @splat(null),
    lower_scores: [5]f32 = @splat(nan),

    fn note(self: *Archive, k: usize, sse: f32) void {
        if (finiteScore(sse) and (!finiteScore(self.lower_scores[k]) or sse < self.lower_scores[k])) self.lower_scores[k] = sse;
    }

    fn remember(self: *Archive, support: Support, event: usize, from_lower: bool) void {
        if (!support.positive() or !improves(support.state.sse, support.state.initial_sse, support.state.initial_sse)) return;
        self.note(support.k, support.state.sse);
        const prior = self.by_order[support.k];
        if (prior == null or support.state.sse < prior.?.support.state.sse) {
            self.by_order[support.k] = .{ .support = support, .event = event, .from_lower = from_lower };
        }
    }

    pub fn best(self: Archive, pooled: ?Archive) ?Achieved {
        var combined = self;
        if (pooled) |archive| combined.include(archive);
        var best_sse = nan;
        for (combined.by_order) |entry| if (entry) |candidate| {
            const sse = candidate.support.state.sse;
            if (!finiteScore(best_sse) or sse < best_sse) best_sse = sse;
        };
        if (!finiteScore(best_sse)) return null;
        // Compare every tie to the actual global minimum. Lower order comes
        // first, followed by the exact N=1 branch at that order.
        for (self.by_order, 0..) |entry, order| {
            if (entry) |candidate| {
                if (!improves(best_sse, candidate.support.state.sse, candidate.support.state.initial_sse)) return candidate;
            }
            if (pooled) |archive| if (archive.by_order[order]) |candidate| {
                if (!improves(best_sse, candidate.support.state.sse, candidate.support.state.initial_sse)) return candidate;
            };
        }
        unreachable;
    }

    fn include(self: *Archive, other: Archive) void {
        for (other.lower_scores, 0..) |sse, k| self.note(k, sse);
        for (other.by_order) |entry| if (entry) |candidate| {
            self.remember(candidate.support, candidate.event, candidate.from_lower);
        };
    }

    fn score(self: Archive, order: usize) f32 {
        return self.lower_scores[order];
    }
};

pub const Outcome = struct {
    support: Support,
    progress: overlap.Progress,
    archive: Archive = .{},

    pub fn updateEvidence(self: *Outcome) void {
        const k = self.support.k;
        if (k == 0) return;
        var lower = self.support.state.initial_sse;
        for (0..k) |order| {
            const sse = self.archive.score(order);
            if (finiteScore(sse)) lower = @min(lower, sse);
        }
        self.progress.previous_sse = lower;
        self.progress.incremental_improvement = @max(0, lower - self.support.state.sse);
    }

    fn useAchieved(self: *Outcome, achieved: Achieved) void {
        self.support = achieved.support;
        self.progress.retained_order = achieved.support.k;
        self.progress.active_count = achieved.support.k;
        self.updateEvidence();
    }
};

pub const Competition = struct {
    archive: *Archive,
    lower: Support,
    joint: Support = .{},
    lower_event: ?usize = null,
    joint_event: ?usize = null,
    valid: u32 = 0,
    screened: u32 = 0,
    unresolved: bool = false,
    // Present only for a gated addition. Includes all scored smaller models
    // in this round, even those that cannot replace the screened lineage.
    stage_reference_sse: ?f32 = null,

    pub fn offerLower(self: *Competition, s: Support, event_idx: usize) void {
        if (!s.positive()) return;
        if (self.stage_reference_sse) |score| {
            self.stage_reference_sse = @min(score, s.state.sse);
            self.archive.note(s.k, s.state.sse);
            if (s.k > 1) return;
        }
        self.archive.remember(s, event_idx, true);
    }

    /// Consider the strongest joint fit only after all phase/rank trials have
    /// contributed their smaller-model competitors.
    pub fn gateJoint(self: *Competition, minimum: f32, events: []Event) bool {
        const idx = self.joint_event orelse return false;
        const reference = self.stage_reference_sse.?;
        const s = self.joint.state;
        events[idx].lower_sse = reference;
        events[idx].joint_sse = s.sse;
        events[idx].flags |= flag.stage_checked;
        self.archive.note(self.joint.k, s.sse);
        if (improves(s.sse, reference, s.initial_sse) and (reference - s.sse) / s.initial_sse >= minimum) {
            events[idx].flags |= flag.stage_passed;
            self.archive.remember(self.joint, idx, false);
            return false;
        }
        self.joint_event = null;
        return true;
    }

    pub fn offerJoint(self: *Competition, baseline_sse: f32, s: Support, event_idx: usize) bool {
        if (!s.positive()) return false;
        if (self.stage_reference_sse == null) self.archive.remember(s, event_idx, false);
        if ((self.stage_reference_sse != null or improves(s.state.sse, baseline_sse, s.state.initial_sse)) and
            (!self.joint.positive() or if (self.stage_reference_sse != null) s.state.sse < self.joint.state.sse else improves(s.state.sse, self.joint.state.sse, s.state.initial_sse)))
        {
            self.joint = s;
            self.joint_event = event_idx;
        }
        return true;
    }

    pub fn selection(self: Competition) ?struct { support: Support, event: usize, replacement: bool } {
        if (self.lower_event) |idx| return .{ .support = self.lower, .event = idx, .replacement = true };
        if (self.joint_event) |idx| return .{ .support = self.joint, .event = idx, .replacement = false };
        return null;
    }
};

const ConditionalParams = extern struct {
    px: u32,
    components: u32,
    selected: u32,
    grid_x: u32 = 1,
    amplitudes: [4]f32,
};

const Refinement = struct {
    bank: gpu.Buf,
    target: gpu.Buf,
    isolated: gpu.Buf = .{},
    scores: gpu.Buf = .{},

    fn init(g: *gpu.Gpu, px: u32, k: u32, refit: bool) Refinement {
        const bytes = @as(u64, px) * @sizeOf(f32);
        return .{
            .bank = g.storage(k * bytes),
            .target = g.storage(bytes),
            .isolated = if (refit) g.storage(k * bytes) else .{},
            .scores = if (refit) g.storage(2 * @sizeOf(f32)) else .{},
        };
    }

    fn deinit(self: *Refinement, g: *gpu.Gpu) void {
        inline for (.{ "bank", "target", "isolated", "scores" }) |field| g.release(&@field(self, field));
    }

    fn isolate(self: *const Refinement, fit: *const overlap.Fit, s: Support, j: u32) void {
        const g = fit.g;
        const grid = gpu.Gpu.flat((fit.px + 255) / 256);
        const p: ConditionalParams = .{ .px = fit.px, .components = s.k, .selected = j, .grid_x = grid[0], .amplitudes = s.state.amplitude };
        const pipe = g.pipeline("overlap_profile/conditional_target", shaders.overlap_profile, "conditional_target") catch gpu.NO_PIPE;
        g.dispatch(pipe, grid[0], grid[1], 1, &.{ g.uniform(0, std.mem.asBytes(&p)), fit.original.at(1), self.bank.at(2), self.target.at(3) });
    }

    fn score(self: *const Refinement, model: *nuisance.Model, s: Support, theta: Theta, slot: u32) void {
        model.prepareFixedTheta(theta);
        const g = model.g;
        const p: ConditionalParams = .{ .px = model.fit.px, .components = s.k, .selected = slot, .amplitudes = s.state.amplitude };
        const pipe = g.pipeline("overlap_profile/conditional_score", shaders.overlap_profile, "conditional_score") catch gpu.NO_PIPE;
        g.dispatch(pipe, 1, 1, 1, &.{ g.uniform(0, std.mem.asBytes(&p)), model.trial.bank.at(2), self.isolated.at(4), self.scores.at(5) });
    }
};

// Only the selected poses/nuisance parameters are persisted. Each GPU trial
// is rebuilt from the same frozen measured target and fixed per-cell PC. These
// single-cell workspaces keep allocations bounded independently of trace size.
const Work = struct {
    engine: *core.Engine,
    model: *nuisance.Model,
    options: Options,
    lv: core.Level,
    frame: spherical.Frame,
    t: [3]f32,
    refinement: ?*Refinement,
    cache: [4]Member = @splat(.{}),

    fn check(self: *Work) !void {
        if (self.engine.cancel.get()) return error.Cancelled;
    }

    fn wait(self: *Work) !void {
        try self.engine.g.wait();
        try self.check();
    }

    fn setPose(self: *Work, m: Member) void {
        const e = self.engine;
        e.det.setPoses(&.{.{ .q = m.pose(), .t = self.t, .pat = 0 }});
    }

    fn stage(self: *Work, s: Support) void {
        const e = self.engine;
        const bytes = @as(u64, self.model.fit.px) * 4;
        var changed = self.model.k != s.k;
        for (s.members[0..s.k], 0..) |member, j| {
            if (member.sameProjection(self.cache[j])) continue;
            self.setPose(member);
            e.det.project(e.phases.items[@intCast(member.source.phase)].det[@intFromEnum(self.lv)], self.frame, 1);
            e.g.copy(e.det.sims, 0, self.model.raw, j * bytes, bytes);
            self.cache[j] = member;
            changed = true;
        }
        if (changed) self.model.profileRawChanged(s.k);
    }

    fn queueFixed(self: *Work, s: Support) void {
        self.stage(s);
        self.model.profileFixedTheta(s.theta);
    }

    fn queueFixedRead(self: *Work, s: *Support) void {
        self.queueFixed(s.*);
        self.model.trial.read(@as(*[1]overlap.State, @ptrCast(&s.state)));
    }

    fn fixed(self: *Work, s: *Support) !void {
        try self.check();
        self.queueFixedRead(s);
        try self.wait();
    }

    fn refineSelected(self: *Work, seed: Support) !Support {
        const work = self.refinement orelse return seed;
        const e = self.engine;
        if (!seed.positive()) return seed;

        self.stage(seed);
        self.model.prepareFixedTheta(seed.theta);
        const fit = self.model.fit;
        const bytes = @as(u64, fit.px) * @sizeOf(f32);
        e.g.copy(self.model.trial.bank, 0, work.bank, 0, seed.k * bytes);
        // The Jacobian uses the model's scratch bank. All component targets
        // use the same frozen fit and are independent of refinement order.
        self.cache = @splat(.{});
        const settings: detector.RefineOptions = .{
            .max_step = @floatCast(std.math.degreesToRadians(e.set.refine_step)),
            .max_total = std.math.degreesToRadians(self.options.radius_deg),
        };
        var result = seed;
        var poses: [overlap.capacity]Quat = undefined;
        errdefer e.g.wait() catch {};
        const shape = [2]u32{ self.frame.pat_h, self.frame.pat_w };
        const transfer = seed.theta[0] != 0 or seed.theta[1] != 0;
        for (seed.members[0..seed.k], 0..) |member, j| {
            try self.check();
            work.isolate(fit, seed, @intCast(j));
            e.prep.filter(work.target, 1, shape[0], shape[1], .{ .mask = self.frame.mask });
            self.setPose(member);
            e.det.beginConditionalRefine(self.frame, settings);
            const master = e.phases.items[@intCast(member.source.phase)].det[@intFromEnum(self.lv)];
            for (0..@as(usize, e.set.refine_iters) + 1) |iteration| {
                e.det.projectRotationJacobian(master, self.frame);
                const jacobian = if (transfer) self.model.transferJacobian(e.det.sims, seed.theta) else e.det.sims;
                e.det.conditionalRefineStep(self.frame, work.target, jacobian, settings, iteration == e.set.refine_iters);
            }
            e.g.read(e.det.poses, 16, std.mem.asBytes(&poses[j]));
        }
        try self.wait();
        var changed = false;
        for (result.members[0..result.k], poses[0..result.k]) |*member, q| {
            var norm: f32 = 0;
            for (q) |x| norm += x * x;
            if (!std.math.isFinite(norm) or @abs(norm - 1) > 0.01 or std.mem.eql(f32, &q, &member.source.q)) continue;
            member.refined = q;
            changed = true;
        }
        if (changed and !try self.fitRefinedSupport(&result)) result = seed;
        if (work.isolated.id != 0) try self.refitIsolated(&result, work);
        return result;
    }

    fn fitRefinedSupport(self: *Work, support: *Support) !bool {
        // A valid NNLS solution may give a component zero weight. Keep the
        // refined orientations of the surviving components at the same theta.
        while (true) {
            try self.fixed(support);
            if (support.positive()) return true;
            if (!assessed(support.state)) return false;
            const active = support.activeMask();
            if (active == 0) return false;
            support.* = support.subset(active);
        }
    }

    fn refitIsolated(self: *Work, support: *Support, work: *const Refinement) !void {
        const seed = support.*;
        const g = &self.engine.g;
        const fit = self.model.fit;
        const bytes = @as(u64, fit.px) * @sizeOf(f32);
        // Rebuild the selected columns even if the orientations were unchanged.
        // Freeze the isolated targets and weights before comparing profiles.
        self.stage(seed);
        self.model.prepareFixedTheta(seed.theta);
        g.copy(self.model.trial.bank, 0, work.bank, 0, seed.k * bytes);
        for (0..seed.k) |j| {
            work.isolate(fit, seed, @intCast(j));
            g.copy(work.target, 0, work.isolated, j * bytes, bytes);
        }
        var scores: [2]f32 = undefined;
        var theta = seed.theta;
        work.score(self.model, seed, theta, 0);
        g.read(work.scores, 0, std.mem.asBytes(&scores[0]));
        try self.wait();
        var best_score = scores[0];
        if (!finiteScore(best_score)) return;
        // Local bounded search from the existing profile. It ranks trials by
        // isolated-component error with fixed poses and fixed weights.
        const steps: Theta = .{ 1.5, 0.35, 2, 0.2, 0.2, 0.06, 0.06 };
        for ([_]f32{ 1, 0.5, 0.25 }) |scale| {
            for (steps, 0..) |step, j| {
                var trials = [2]Theta{ theta, theta };
                trials[0][j] = nuisance.clampParameter(theta[j] + step * scale, j);
                trials[1][j] = nuisance.clampParameter(theta[j] - step * scale, j);
                for (trials, 0..) |trial, slot| work.score(self.model, seed, trial, @intCast(slot));
                g.read(work.scores, 0, std.mem.asBytes(&scores));
                try self.wait();
                for (scores, trials) |score, trial| {
                    if (improves(score, best_score, seed.state.initial_sse)) {
                        theta = trial;
                        best_score = score;
                    }
                }
            }
        }
        if (std.mem.eql(f32, &theta, &seed.theta)) return;
        var trial = seed;
        trial.theta = theta;
        try self.fixed(&trial);
        if (trial.positive()) support.* = trial;
    }

    fn publish(self: *Work, out: Outcome, slots: []overlap.Component, source_q: []Quat, ranks: []i32, norms: []f32, state: *overlap.State) !void {
        const s = out.support;
        state.* = s.state;
        if (s.k == 0) return;
        self.stage(s);
        var magnitudes: [4]f32 = undefined;
        self.model.readProfileNorms(s.theta, &magnitudes);
        try self.wait();
        state.phase = @floatFromInt(s.members[s.k - 1].source.phase);
        state.trial_sse = state.sse;
        const best_single = out.archive.score(1);
        state.best_single_sse = if (finiteScore(best_single)) best_single else state.initial_sse;
        for (s.members[0..s.k], 0..) |member, j| {
            slots[j] = member.source;
            slots[j].q = member.pose();
            slots[j].amplitude = s.state.amplitude[j];
            source_q[j] = member.source.q;
            ranks[j] = member.rank;
            norms[j] = magnitudes[j];
            const single = if (finiteScore(member.single_sse)) @min(state.single_sse[j], member.single_sse) else state.single_sse[j];
            state.single_sse[j] = if (s.k == 1) s.state.sse else single;
            state.best_single_sse = @min(state.best_single_sse, state.single_sse[j]);
        }
    }
};

const History = struct {
    outcome: Outcome,
    events: []Event,
    used: usize = 0,
    replacements: [5]u32 = @splat(0),

    fn log(self: *History, event: Event) !usize {
        if (self.used == self.events.len) return error.ProfileTraceCapacityExceeded;
        const index = self.used;
        self.events[index] = event;
        self.used += 1;
        return index;
    }

    fn nextOrder(self: *const History) ?u32 {
        return if (self.outcome.progress.stop_reason == 0) self.outcome.support.k + 1 else null;
    }

    fn begin(self: *History, order: u32, options: Options) Competition {
        const baseline = self.outcome.support;
        var reference: ?f32 = null;
        if (order > 1 and options.min_stage_improvement != null) {
            reference = baseline.state.sse;
            for (1..order) |k_lower| {
                const score = self.outcome.archive.score(k_lower);
                if (finiteScore(score)) reference = @min(reference.?, score);
            }
        }
        return .{ .lower = baseline, .archive = &self.outcome.archive, .stage_reference_sse = reference };
    }

    fn finish(self: *History, battle: *Competition, order: u32, k: u32, options: Options) void {
        const out = &self.outcome;
        if (battle.valid > 0 and battle.screened == battle.valid) {
            // The accepted prefix stays intact when every candidate is screened.
            out.progress.stop_reason = 11;
            return;
        }
        const stage_rejected = if (battle.stage_reference_sse != null)
            battle.gateJoint(options.min_stage_improvement.?[order - 2], self.events)
        else
            false;
        out.progress.proposed_count += 1;
        if (stage_rejected) {
            out.progress.stop_reason = 13;
            return;
        }
        const selected = battle.selection() orelse {
            out.progress.stop_reason = if (battle.unresolved) 4 else if (battle.valid == 0) 3 else 5;
            out.progress.boundary_seen = if (battle.unresolved) 2 else if (battle.valid != 0) 1 else 0;
            out.updateEvidence();
            return;
        };
        out.support = selected.support;
        const event = &self.events[selected.event];
        if (selected.replacement) {
            event.flags |= flag.replace;
            self.replacements[order] += 1;
            if (self.replacements[order] >= max_replacements) out.progress.stop_reason = 12;
        } else event.flags |= flag.add;
        event.result_order = out.support.k;
        out.progress.retained_order = out.support.k;
        out.progress.active_count = out.support.k;
        out.updateEvidence();
        out.progress.boundary_seen = 0;
        if (out.progress.stop_reason == 0) {
            if (out.support.state.sse <= 1e-8 * out.support.state.initial_sse) {
                out.progress.stop_reason = 2;
            } else if (out.support.k == k) out.progress.stop_reason = 1;
        }
    }
};

const Row = struct {
    t: [3]f32,
    trace: []Event,
    histories: [2]History,

    fn sharePrimary(self: *Row) void {
        const exact = &self.histories[0];
        const pooled = &self.histories[1];
        pooled.outcome = exact.outcome;
        pooled.used = exact.used;
        const events = pooled.events[0..pooled.used];
        @memcpy(events, exact.events[0..exact.used]);
        for (events) |*event| event.branch = 1;
    }

    fn select(self: *Row, compare_pooled: bool, options: Options) struct { outcome: Outcome, events: []Event } {
        const exact = &self.histories[0];
        const pooled = &self.histories[1];
        const exact_used = exact.used;
        const pooled_used = pooled.used;
        if (compare_pooled) {
            std.mem.copyForwards(Event, self.trace[exact_used..][0..pooled_used], pooled.events[0..pooled_used]);
            for (&pooled.outcome.archive.by_order) |*entry| {
                if (entry.*) |*candidate| candidate.event += exact_used;
            }
        }
        const events = self.trace[0 .. exact_used + pooled_used];
        const archive = if (compare_pooled) pooled.outcome.archive else null;
        var selected = exact.outcome;
        var source: i32 = -1;
        if (exact.outcome.archive.best(archive)) |achieved| {
            const event = &events[achieved.event];
            source = @intCast(event.branch);
            if (source == 1) selected = pooled.outcome;
            if (archive) |other| {
                selected.archive = exact.outcome.archive;
                selected.archive.include(other);
            }
            selected.useAchieved(achieved);
            if (options.flags & 4 != 0) {
                event.flags |= flag.published;
                if (achieved.from_lower) event.flags |= flag.published_lower;
            }
        }
        for (events) |*event| if (source >= 0 and event.branch == @as(u32, @intCast(source))) {
            event.flags |= flag.chosen_branch;
        };
        selected.progress._pad = if (source < 0) unselected else @intCast(source);
        return .{ .outcome = selected, .events = events };
    }
};

fn fitJob(row: usize, t: [3]f32, support: Support, mode: batched.Mode) batched.Job {
    var job: batched.Job = .{
        .row = row,
        .k = support.k,
        .poses = undefined,
        .phases = undefined,
        .seeds = .{ .warm = support.theta },
        .state = support.state,
        .mode = mode,
    };
    for (support.members[0..support.k], 0..) |member, j| {
        job.poses[j] = .{ .q = member.pose(), .t = t, .pat = 0 };
        job.phases[j] = @intCast(member.source.phase);
    }
    for (&job.seeds.singles, 0..) |*seed, j| seed.* = support.members[@min(j, support.k - 1)].single_theta;
    return job;
}

const Wave = struct {
    jobs: []batched.Job,
    destinations: []*Support,
    used: usize = 0,

    fn add(self: *Wave, row: usize, t: [3]f32, support: *Support, mode: batched.Mode) void {
        self.jobs[self.used] = fitJob(row, t, support.*, mode);
        self.destinations[self.used] = support;
        self.used += 1;
    }

    fn fit(self: *Wave, batch: *batched.Batch) !void {
        if (self.used == 0) return;
        const jobs = self.jobs[0..self.used];
        try batch.run(jobs);
        for (jobs, self.destinations[0..self.used]) |job, support| {
            if (job.mode != .optimize) support.state = job.state;
            if (job.optimized) |trial| {
                if (assessed(trial.state) and (!assessed(support.state) or improves(trial.state.sse, support.state.sse, trial.state.initial_sse))) {
                    support.state = trial.state;
                    support.theta = trial.theta;
                }
            }
        }
        self.used = 0;
    }
};

fn saveSingleton(s: *Support) void {
    s.members[0].single_sse = s.state.sse;
    s.members[0].single_theta = s.theta;
}

const Proposal = struct {
    lane: usize,
    event: usize,
    single: Support,
    joint: Support = .{},
    lower: Support = .{},
    // At most four drop-one models and one smaller active face.
    faces: [5]Support = undefined,
    face_count: usize = 0,

    fn faceForMask(self: *Proposal, mask: u32) ?*Support {
        const active = if (assessed(self.joint.state)) self.joint.activeMask() else 0;
        if (@popCount(mask) != self.joint.k - 1 and active != mask) return null;
        return &self.faces[self.face_count];
    }

    fn offer(self: *const Proposal, baseline: Support, battle: *Competition, event: *Event) void {
        const single = self.single;
        if (baseline.k == 0) {
            battle.archive.remember(single, self.event, false);
            event.joint_sse = if (assessed(single.state)) single.state.sse else nan;
            if (single.positive() and improves(single.state.sse, baseline.state.sse, baseline.state.initial_sse)) {
                event.flags |= flag.joint;
                if (!battle.joint.positive() or improves(single.state.sse, battle.joint.state.sse, single.state.initial_sse)) {
                    battle.joint = single;
                    battle.joint_event = self.event;
                }
            } else {
                event.flags |= flag.boundary;
                battle.unresolved = battle.unresolved or !assessed(single.state);
            }
            return;
        }

        battle.offerLower(single, self.event);
        const joint = self.joint;
        if (assessed(joint.state)) {
            // The accepted support is a feasible nested fit with zero added amplitude.
            event.joint_sse = @min(baseline.state.sse, joint.state.sse);
            event.flags |= if (battle.offerJoint(baseline.state.sse, joint, self.event)) flag.joint else flag.boundary;
        } else {
            battle.unresolved = true;
            event.flags |= flag.boundary;
        }
        for (self.faces[0..self.face_count]) |reduced| battle.offerLower(reduced, self.event);
        const lower = self.lower;
        event.flags |= flag.lower;
        event.lower_sse = lower.state.sse;
        if (battle.stage_reference_sse == null and lower.positive()) {
            const prior = battle.lower;
            if (improves(lower.state.sse, prior.state.sse, lower.state.initial_sse) or
                (battle.lower_event != null and !improves(prior.state.sse, lower.state.sse, prior.state.initial_sse) and lower.k < prior.k))
            {
                battle.lower = lower;
                battle.lower_event = self.event;
            }
        }
    }
};

const Round = struct {
    row: usize,
    battle: Competition,
};

const Controller = struct {
    batch: *batched.Batch,
    cells: []const u64,
    rows: []Row,
    primary: []const core.Record,
    k: u32,
    keep_n: u32,
    options: Options,
    band: ?*core.Band,
    search: core.Search,
    rounds: []Round,
    ids: []u64,
    records: []core.Record,
    proposals: []Proposal,
    wave: Wave,

    fn readInitial(self: *Controller, states: []overlap.State, progress: []overlap.Progress) !void {
        const batch = self.batch;
        const e = batch.engine;
        const fit = batch.fit;
        const bytes = @as(u64, fit.px) * @sizeOf(f32);
        var first: usize = 0;
        while (first < states.len) {
            if (e.cancel.get()) return error.Cancelled;
            const n = @min(batch.lanes, states.len - first);
            fit.n = @intCast(n);
            e.g.copy(batch.frozen, first * bytes, fit.original, 0, n * bytes);
            fit.resetTarget(fit.original);
            fit.read(states[first..][0..n]);
            fit.readProgress(progress[first..][0..n]);
            try e.g.wait();
            first += n;
        }
    }

    fn runStages(self: *Controller) !void {
        try self.runStage(1, 0, 1);
        const branches: usize = if (self.k > 1 and self.keep_n > 1) 2 else 1;
        if (branches == 2) for (self.rows) |*row| row.sharePrimary();

        while (true) {
            var order = self.k + 1;
            var branch: usize = 0;
            for (0..branches) |b| {
                for (self.rows) |*row| {
                    if (row.histories[b].nextOrder()) |next| {
                        if (next >= order) continue;
                        order = next;
                        branch = b;
                    }
                }
            }
            if (order > self.k) return;
            try self.runStage(order, branch, if (branch == 0) 1 else self.keep_n);
        }
    }

    fn runStage(self: *Controller, order: u32, branch: usize, ranks: u32) !void {
        const batch = self.batch;
        const e = batch.engine;
        var at: usize = 0;
        while (at < self.rows.len) {
            if (e.cancel.get()) return error.Cancelled;
            var n: usize = 0;
            while (at < self.rows.len and n < self.rounds.len) : (at += 1) {
                const row = &self.rows[at];
                const history = &row.histories[branch];
                if (history.nextOrder() != order) continue;
                self.rounds[n] = .{ .row = at, .battle = history.begin(order, self.options) };
                if (order > 1) {
                    self.wave.jobs[n] = fitJob(at, row.t, history.outcome.support, .fixed);
                    self.ids[n] = self.cells[at];
                }
                n += 1;
            }
            if (n == 0) break;
            const records = self.records[0 .. n * e.phases.items.len * ranks];
            if (order > 1) {
                batch.residual(self.wave.jobs[0..n], e.pats[@intFromEnum(batch.lv)]);
                try e.indexPreparedCandidates(self.ids[0..n], self.search, self.band, ranks, records);
            }
            try self.scoreRound(self.rounds[0..n], order, branch, ranks, records);
        }
    }

    fn scoreRound(self: *Controller, rounds: []Round, order: u32, branch: usize, ranks: u32, records: []const core.Record) !void {
        const phase_count = self.batch.engine.phases.items.len;
        var used: usize = 0;
        for (rounds, 0..) |*round, lane| {
            const row = &self.rows[round.row];
            const history = &row.histories[branch];
            const baseline = history.outcome.support;
            for (0..phase_count) |pi| for (0..ranks) |rank| {
                const record = if (order == 1)
                    self.primary[pi * self.rows.len + round.row]
                else
                    records[(pi * ranks + rank) * rounds.len + lane];
                const member = Member.indexed(record, pi, if (order == 1) 0 else @intCast(rank + 1));
                const valid = member.valid();
                const event = try history.log(.{
                    .cell_index = @intCast(round.row),
                    .branch = @intCast(branch),
                    .order = order,
                    .attempt = history.replacements[order],
                    .phase = @intCast(pi),
                    .rank = member.rank,
                    .source_q = record.q,
                    .residual_ncc = if (order == 1 or !valid) nan else record.ncc,
                    .baseline_sse = baseline.state.sse,
                    .result_order = baseline.k,
                });
                if (!valid) continue;
                history.events[event].flags |= flag.native;
                round.battle.valid += 1;
                if (order > 1 and self.options.flags & 2 != 0 and record.ncc < self.options.min_residual_ncc) {
                    history.events[event].flags |= flag.screened;
                    round.battle.screened += 1;
                    continue;
                }
                const proposal = &self.proposals[used];
                proposal.* = .{ .lane = lane, .event = event, .single = (Support{}).append(member) };
                self.wave.add(round.row, row.t, &proposal.single, .both);
                used += 1;
            };
        }
        try self.wave.fit(self.batch);
        const proposals = self.proposals[0..used];
        for (proposals) |*proposal| {
            if (assessed(proposal.single.state)) saveSingleton(&proposal.single);
        }
        if (order > 1 and used > 0) try self.scoreJointAndFaces(rounds, proposals, branch, order);
        // GPU work is grouped by fit shape; offer achieved results in exactly
        // the original phase, rank and face-mask order within each history.
        for (proposals) |*proposal| {
            const round = &rounds[proposal.lane];
            const history = &self.rows[round.row].histories[branch];
            proposal.offer(history.outcome.support, &round.battle, &history.events[proposal.event]);
        }
        if (self.batch.engine.cancel.get()) return error.Cancelled;
        for (rounds) |*round| {
            self.rows[round.row].histories[branch].finish(&round.battle, order, self.k, self.options);
        }
    }

    fn scoreJointAndFaces(self: *Controller, rounds: []const Round, proposals: []Proposal, branch: usize, order: u32) !void {
        for (proposals) |*proposal| {
            const i = rounds[proposal.lane].row;
            const row = &self.rows[i];
            const baseline = row.histories[branch].outcome.support;
            const single = proposal.single;
            proposal.lower = if (single.positive() and improves(single.state.sse, baseline.state.sse, single.state.initial_sse)) single else baseline;
            proposal.joint = baseline.append(single.members[0]);
            self.wave.add(i, row.t, &proposal.joint, .both);
        }
        try self.wave.fit(self.batch);

        const all = (@as(u32, 1) << @intCast(order)) - 1;
        for (1..all) |m| {
            const mask: u32 = @intCast(m);
            for (proposals) |*proposal| {
                const face = proposal.faceForMask(mask) orelse continue;
                face.* = proposal.joint.subset(mask);
                const i = rounds[proposal.lane].row;
                // For K2 both singleton poses have already had a nuisance fit.
                // Re-optimize from joint theta only when its fixed fit improves.
                self.wave.add(i, self.rows[i].t, face, if (order == 2) .fixed else .both);
            }
            try self.wave.fit(self.batch);
            if (order == 2 and self.batch.optimize) {
                for (proposals) |*proposal| {
                    const face = proposal.faceForMask(mask) orelse continue;
                    if (!assessed(face.state) or !improves(face.state.sse, proposal.lower.state.sse, face.state.initial_sse)) continue;
                    const i = rounds[proposal.lane].row;
                    self.wave.add(i, self.rows[i].t, face, .optimize);
                }
                try self.wave.fit(self.batch);
            }

            while (true) {
                for (proposals) |*proposal| {
                    const face = proposal.faceForMask(mask) orelse continue;
                    if (!assessed(face.state) or face.positive()) continue;
                    const active = face.activeMask();
                    if (active == 0) {
                        face.* = .{};
                        continue;
                    }
                    face.* = face.subset(active);
                    const i = rounds[proposal.lane].row;
                    self.wave.add(i, self.rows[i].t, face, .both);
                }
                if (self.wave.used == 0) break;
                try self.wave.fit(self.batch);
            }

            for (proposals) |*proposal| {
                const face = proposal.faceForMask(mask) orelse continue;
                proposal.face_count += 1;
                if (!face.positive()) continue;
                if (face.k == 1) saveSingleton(face);
                const lower = proposal.lower;
                if (improves(face.state.sse, lower.state.sse, face.state.initial_sse) or
                    (lower.k != order - 1 and !improves(lower.state.sse, face.state.sse, lower.state.initial_sse) and face.k < lower.k))
                    proposal.lower = face.*;
            }
        }
        for (proposals) |*proposal| {
            if (proposal.lower.k == 1) saveSingleton(&proposal.lower);
        }
    }
};

pub fn batchWidth(e: *const core.Engine) u32 {
    const shape = e.patternShape(e.levelOf(.search)) catch return 0;
    const px = std.math.mul(u32, shape[0], shape[1]) catch return 0;
    return batched.maxLanes(e, px);
}

pub fn run(e: *core.Engine, cells: []const u64, k: u32, primary: []const core.Record, missing: []const u32, hw: [2]u32, outputs: core.Engine.OverlapTrajectoryPayload, optimize: bool, keep_n: u32, prm: core.Search, req: *Request) !void {
    const g = &e.g;
    // Retire per-call scratch before the next tile allocates its workspace.
    defer g.submit();
    const lv = e.levelOf(.search);
    const frame = try e.frame(lv);
    const px = hw[0] * hw[1];
    const bytes = @as(u64, px) * @sizeOf(f32);
    const n = cells.len;
    const phase_count = e.phases.items.len;
    const compare_pooled = k > 1 and keep_n > 1;
    const saved_ahead = e.ahead;
    e.ahead = &.{};
    defer e.ahead = saved_ahead;
    var frozen = g.storage(n * bytes);
    defer g.release(&frozen);
    g.copy(e.pats[@intFromEnum(lv)], 0, frozen, 0, n * bytes);

    var arena = std.heap.ArenaAllocator.init(e.gpa);
    defer arena.deinit();
    const a = arena.allocator();
    const rows = try a.alloc(Row, n);
    {
        var scratch = std.heap.ArenaAllocator.init(e.gpa);
        defer scratch.deinit();
        const temp = scratch.allocator();
        const candidates_per_row = phase_count * keep_n;
        // A small request can still fill GPU lanes with independent proposals.
        var batch = try batched.Batch.init(e, lv, frame, frozen, @max(n, candidates_per_row), optimize);
        defer batch.deinit();
        // An indexing failure can leave reads targeting controller scratch.
        errdefer g.wait() catch {};
        const row_lanes = @min(n, batch.lanes);
        const capacity = try std.math.mul(usize, row_lanes, candidates_per_row);
        var controller: Controller = .{
            .batch = &batch,
            .cells = cells,
            .rows = rows,
            .primary = primary,
            .k = k,
            .keep_n = keep_n,
            .options = req.options,
            .band = if (k > 1 and prm.method == .spherical) try e.band(prm.bandwidth, lv) else null,
            .search = prm,
            .rounds = try temp.alloc(Round, row_lanes),
            .ids = try temp.alloc(u64, row_lanes),
            .records = try temp.alloc(core.Record, capacity),
            .proposals = try temp.alloc(Proposal, capacity),
            .wave = .{
                .jobs = try temp.alloc(batched.Job, capacity),
                .destinations = try temp.alloc(*Support, capacity),
            },
        };
        try controller.readInitial(outputs.states, outputs.progress);
        const exact_capacity = try traceCapacity(1, phase_count, k, 1);
        const row_capacity = try traceCapacity(1, phase_count, k, keep_n);
        for (rows, cells, missing, outputs.states, outputs.progress, 0..) |*row, cell, absent, *state, *progress, i| {
            const events = req.events[req.used + i * row_capacity ..][0..row_capacity];
            if (absent != 0) {
                state.* = .{ .sse = nan, .initial_sse = nan, .single_sse = @splat(nan), .correlation = nan, .condition = nan, .best_single_sse = nan, .trial_sse = nan };
                progress.* = .{ .stop_reason = absent, .previous_sse = nan, .incremental_improvement = nan, ._pad = unselected };
            }
            const initial: Outcome = .{ .support = .{ .state = state.* }, .progress = progress.* };
            row.* = .{
                .t = if (absent == 0) try e.positionOf(&e.geo, cell, lv) else @splat(0),
                .trace = events,
                .histories = .{
                    .{ .outcome = initial, .events = events[0..exact_capacity] },
                    .{ .outcome = initial, .events = events[exact_capacity..] },
                },
            };
        }
        try controller.runStages();
    }
    // Retire GPU buffers before reserving scalar polishing scratch.
    g.submit();

    errdefer g.wait() catch {};
    var fit = overlap.Fit.init(g, 1, px, hw[1], frame.mask, frozen);
    defer fit.deinit();
    var model = nuisance.Model.init(g, &e.prep, &fit);
    defer model.deinit();
    model.contrast = true;
    model.optimize_enabled = optimize;
    var refinement: ?Refinement = if (req.options.flags & 1 != 0)
        Refinement.init(g, px, k, req.options.flags & refit_after_polish != 0)
    else
        null;
    defer if (refinement) |*workspace| workspace.deinit(g);
    for (rows, missing, 0..) |*row, absent, i| {
        if (e.cancel.get()) return error.Cancelled;
        if (absent != 0) continue;
        const result = row.select(compare_pooled, req.options);
        std.mem.copyForwards(Event, req.events[req.used..][0..result.events.len], result.events);
        req.used += result.events.len;
        var selected = result.outcome;
        if (selected.support.k > 0) {
            if (i != 0) {
                g.copy(frozen, i * bytes, fit.original, 0, bytes);
                fit.resetTarget(fit.original);
                model.resetProfileTarget();
            }
            var work: Work = .{
                .engine = e,
                .model = &model,
                .options = req.options,
                .lv = lv,
                .frame = frame,
                .t = row.t,
                .refinement = if (refinement) |*workspace| workspace else null,
            };
            // A singleton in an overlap run has no other signal to isolate.
            if (k == 1 or selected.support.k > 1) selected.support = try work.refineSelected(selected.support);
            const first = i * k;
            try work.publish(selected, outputs.components[first..][0..k], req.source_q[first..][0..k], outputs.ranks[first..][0..k], outputs.norms[first..][0..k], &outputs.states[i]);
        } else outputs.states[i] = selected.support.state;
        selected.progress.retained_order = selected.support.k;
        selected.progress.active_count = selected.support.k;
        selected.updateEvidence();
        outputs.progress[i] = selected.progress;
        outputs.theta[i] = selected.support.theta;
    }
    if (e.cancel.get()) return error.Cancelled;
}
