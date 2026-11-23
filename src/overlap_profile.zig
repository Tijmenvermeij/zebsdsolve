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
const Theta = nuisance.Theta;
const Quat = [4]f32;
const Offset = [3]f32;
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

pub const Options = struct {
    flags: u32 = 1, // 1 polish pose, 2 screen residual NCC, 4 emit publication witness
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
        return self.flags & 3 != 0 and self.flags & ~@as(u32, 7) == 0 and
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

    fn log(self: *Request, event: Event) !usize {
        if (self.used == self.events.len) return error.ProfileTraceCapacityExceeded;
        const index = self.used;
        self.events[index] = event;
        self.used += 1;
        return index;
    }
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

/// Apply the same crystal-frame exponential used by Detector.refine. The
/// caller uploads the immutable source pose before each independent trial.
pub fn applyPoseOffset(g: *gpu.Gpu, poses: gpu.Buf, delta: Offset) void {
    if (delta[0] == 0 and delta[1] == 0 and delta[2] == 0) return;
    const packed_step = [4]f32{ delta[0], delta[1], delta[2], 0 };
    const pipe = g.pipeline("overlap_profile/profile_pose", shaders.overlap_profile, "profile_pose") catch gpu.NO_PIPE;
    const u = g.uniform(0, std.mem.asBytes(&packed_step));
    g.dispatch(pipe, 1, 1, 1, &.{ u, poses.at(1) });
}

const Member = struct {
    source: overlap.Component = .{},
    rank: i32 = -1,
    // Coordinate of the current pose in a fixed exponential chart. Every
    // revisit, drop-one fit and replacement retains the immutable source q.
    delta: Offset = @splat(0),
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

    fn sameProjection(self: Member, other: Member) bool {
        return self.source.phase == other.source.phase and std.mem.eql(f32, &self.source.q, &other.source.q) and
            std.mem.eql(f32, &self.delta, &other.delta);
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

const SingletonSeed = struct {
    state: overlap.State,
    theta: Theta = nuisance.identity,
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
    cell: u64,
    row: u32,
    band: ?*core.Band,
    pool: []core.Record,
    trial_states: gpu.Buf,
    seed_target: gpu.Buf,
    residual_seeds: []?SingletonSeed,
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
        e.det.setPoses(&.{.{ .q = m.source.q, .t = self.t, .pat = 0 }});
        applyPoseOffset(&e.g, e.det.poses, m.delta);
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

    fn queueTheta(self: *Work, s: Support, full: bool, trial: *overlap.State, theta: *Theta) void {
        self.stage(s);
        var seeds: [4]Theta = undefined;
        for (&seeds, 0..) |*t, j| t.* = s.members[@min(j, s.k - 1)].single_theta;
        self.model.profileNuisance(s.theta, seeds, full);
        self.model.readProfile(trial, theta);
    }

    fn retainTheta(s: *Support, trial: overlap.State, theta: Theta) void {
        if (assessed(trial) and (!assessed(s.state) or improves(trial.sse, s.state.sse, trial.initial_sse))) {
            s.state = trial;
            s.theta = theta;
        }
    }

    fn optimizeTheta(self: *Work, s: *Support, full: bool) !void {
        if (!self.model.optimize_enabled) return;
        try self.check();
        var trial: overlap.State = undefined;
        var theta: Theta = undefined;
        self.queueTheta(s.*, full, &trial, &theta);
        try self.wait();
        retainTheta(s, trial, theta);
    }

    // One six-neighbour poll, all at the same theta and incumbent support.
    // Return only six small states from the GPU. The original q remains the
    // centre of the trust ball even when a component persists across orders.
    fn poll(self: *Work, s: *Support, column: usize, step: f32, radius: f32) !bool {
        try self.check();
        var deltas: [6]Offset = undefined;
        var states: [6]overlap.State = undefined;
        for (&deltas, 0..) |*delta, i| {
            delta.* = s.members[column].delta;
            delta[i / 2] += if (i % 2 == 0) step else -step;
            const norm = @sqrt(delta[0] * delta[0] + delta[1] * delta[1] + delta[2] * delta[2]);
            if (norm > radius) for (delta) |*v| {
                v.* *= radius / norm;
            };
            var trial = s.*;
            trial.members[column].delta = delta.*;
            self.queueFixed(trial);
            self.engine.g.copy(self.model.trial.state, 0, self.trial_states, i * @sizeOf(overlap.State), @sizeOf(overlap.State));
        }
        self.engine.g.read(self.trial_states, 0, std.mem.asBytes(&states));
        try self.wait();
        var moved = false;
        for (states, deltas) |state, delta| {
            if (!assessed(state) or (assessed(s.state) and !improves(state.sse, s.state.sse, state.initial_sse))) continue;
            s.state = state;
            s.members[column].delta = delta;
            moved = true;
        }
        return moved;
    }

    fn polishStart(self: *Work, s: *Support) !void {
        const radius = std.math.degreesToRadians(self.options.radius_deg);
        for ([_]f32{ 0.5, 0.25, 0.125 }) |scale| {
            var moved = false;
            for (0..2) |_| {
                var changed = false;
                for (0..s.k) |j| if (try self.poll(s, j, scale * radius, radius)) {
                    changed = true;
                };
                moved = moved or changed;
                if (!changed) break;
            }
            if (moved) try self.optimizeTheta(s, false);
        }
    }

    fn polish(self: *Work, input: Support, prefit: bool) !Support {
        var seed = input;
        try self.check();
        if (!prefit) self.queueFixedRead(&seed);
        // The nuisance search does not consume the fixed-fit result. Queue both
        // reads before waiting, then compare once both host destinations are ready.
        if (self.model.optimize_enabled) {
            var trial: overlap.State = undefined;
            var theta: Theta = undefined;
            self.queueTheta(seed, true, &trial, &theta);
            try self.wait();
            retainTheta(&seed, trial, theta);
        } else if (!prefit) try self.wait();
        return self.polishFromSeed(seed);
    }

    fn polishFromSeed(self: *Work, seed: Support) !Support {
        if (self.options.flags & 1 == 0) return seed;
        var best = seed;
        const starts: usize = if (seed.k == 1) 3 else 1;
        for (0..starts) |i| {
            var current = seed;
            if (i != 0) {
                // Fixed secondary singleton starts, independent of pooling,
                // storage mode and phase. Profile new and original singletons
                // using precisely the same angular budget.
                const half = std.math.degreesToRadians(self.options.radius_deg) * 0.5;
                current.members[0].delta = .{ if (i == 1) half else -half, 0, 0 };
                try self.fixed(&current);
            }
            try self.polishStart(&current);
            if (assessed(current.state) and (!assessed(best.state) or improves(current.state.sse, best.state.sse, current.state.initial_sse))) best = current;
        }
        var changed = false;
        for (best.members[0..best.k], seed.members[0..seed.k]) |a, b| if (!a.sameProjection(b)) {
            changed = true;
        };
        if (changed) try self.optimizeTheta(&best, true);
        return best;
    }

    fn residual(self: *Work, s: Support) !void {
        try self.check();
        self.queueFixed(s);
        const g = &self.engine.g;
        const f = self.model.fit;
        g.copy(self.model.trial.bank, 0, f.bank, 0, @as(u64, s.k) * f.px * 4);
        // The unchanged accepted amplitude vector is the explicit nested
        // zero-new-amplitude witness. Rebuild its own residual, never a rejected
        // candidate's bank, coefficients or theta.
        g.writeSlice(f.state, 0, overlap.State, &.{s.state});
        f.residual(s.k, self.engine.pats[@intFromEnum(self.lv)], true);
    }

    fn singleton(self: *Work, native: Member, precomputed: ?SingletonSeed) !Support {
        var seed = (Support{}).append(native);
        var result = if (precomputed) |pre| result: {
            seed.state = pre.state;
            seed.theta = pre.theta;
            break :result try self.polishFromSeed(seed);
        } else try self.polish(seed, false);
        if (assessed(result.state)) {
            result.members[0].single_theta = result.theta;
            result.members[0].single_sse = result.state.sse;
        }
        return result;
    }

    fn compactFace(self: *Work, input: Support) !Support {
        var s = input;
        while (assessed(s.state) and !s.positive()) {
            const mask = s.activeMask();
            if (mask == 0) return .{};
            s = try self.polish(s.subset(mask), false);
        }
        return s;
    }

    fn saveLower(battle: *Competition, s: Support, event_idx: usize) void {
        if (!s.positive()) return;
        const b = battle.lower;
        if (improves(s.state.sse, b.state.sse, s.state.initial_sse) or
            (battle.lower_event != null and !improves(b.state.sse, s.state.sse, b.state.initial_sse) and s.k < b.k))
        {
            battle.lower = s;
            battle.lower_event = event_idx;
        }
    }

    fn evaluate(self: *Work, baseline: Support, native: Member, battle: *Competition, req: *Request, index: usize, primary_fit: ?*?Support, precomputed: ?SingletonSeed) !void {
        var event = &req.events[index];
        if (!native.valid()) return;
        event.flags |= flag.native;
        battle.valid += 1;
        if (baseline.k > 0 and self.options.flags & 2 != 0 and native.source.proposal_ncc < self.options.min_residual_ncc) {
            event.flags |= flag.screened;
            battle.screened += 1;
            return;
        }
        const cached = if (primary_fit) |fit| fit.* else null;
        const single = cached orelse try self.singleton(native, precomputed);
        if (primary_fit) |fit| fit.* = single;
        if (baseline.k == 0) {
            battle.archive.remember(single, index, false);
        } else {
            battle.offerLower(single, index);
        }
        if (baseline.k == 0) {
            event.joint_sse = if (assessed(single.state)) single.state.sse else nan;
            if (single.positive() and improves(single.state.sse, baseline.state.sse, baseline.state.initial_sse)) {
                event.flags |= flag.joint;
                if (!battle.joint.positive() or improves(single.state.sse, battle.joint.state.sse, single.state.initial_sse)) {
                    battle.joint = single;
                    battle.joint_event = index;
                }
            } else {
                event.flags |= flag.boundary;
                battle.unresolved = battle.unresolved or !assessed(single.state);
            }
            return;
        }
        // The prior support, embedded with zero amplitude for the new pose, is
        // feasible even when a reprofiled joint design is rank deficient.
        var lower = baseline;
        if (single.positive() and improves(single.state.sse, lower.state.sse, lower.state.initial_sse)) lower = single;

        // Include the achieved strengthened singleton as a starting pose for
        // the alternative. Also evaluate the original residual-native pose at
        // the same theta; a weak real component may fit only in the mixture.
        var seed = baseline.append(single.members[0]);
        seed.theta = baseline.theta;
        try self.check();
        self.queueFixedRead(&seed);
        const compare_poses = !single.members[0].sameProjection(native);
        var alt = seed;
        if (compare_poses) {
            alt.members[baseline.k].delta = native.delta;
            self.queueFixedRead(&alt);
        }
        try self.wait();
        if (compare_poses and assessed(alt.state) and (!assessed(seed.state) or improves(alt.state.sse, seed.state.sse, alt.state.initial_sse))) seed = alt;
        const joint = try self.polish(seed, true);
        if (assessed(joint.state)) {
            // The prior embedded model is an achieved feasible nested trial.
            event.joint_sse = @min(baseline.state.sse, joint.state.sse);
            event.flags |= if (battle.offerJoint(baseline.state.sse, joint, index)) flag.joint else flag.boundary;
        } else {
            battle.unresolved = true;
            event.flags |= flag.boundary;
        }

        // At K3/K4 each possible drop-one support is independently reprofiled.
        // Include any smaller exact active face the joint happened to reach.
        const all = (@as(u32, 1) << @intCast(joint.k)) - 1;
        const active = if (assessed(joint.state)) joint.activeMask() else 0;
        for (1..all) |m| {
            const mask: u32 = @intCast(m);
            const count: u32 = @popCount(mask);
            if (count != joint.k - 1 and !(active != 0 and active == mask)) continue;
            const subset = joint.subset(mask);
            // K2's independently strengthened singletons are already achieved.
            // An identical pose can still benefit from the joint theta. If the
            // joint moved a member, reprofile that new singleton as well.
            var reduced = subset;
            if (joint.k == 2 and
                (subset.members[0].sameProjection(baseline.members[0]) or subset.members[0].sameProjection(single.members[0])))
            {
                try self.fixed(&reduced);
                if (assessed(reduced.state) and improves(reduced.state.sse, lower.state.sse, reduced.state.initial_sse))
                    try self.optimizeTheta(&reduced, true);
            } else reduced = try self.polish(subset, false);
            reduced = try self.compactFace(reduced);
            if (reduced.positive() and reduced.k == 1) {
                reduced.members[0].single_sse = reduced.state.sse;
                reduced.members[0].single_theta = reduced.theta;
            }
            battle.offerLower(reduced, index);
            if (reduced.positive() and (improves(reduced.state.sse, lower.state.sse, reduced.state.initial_sse) or
                (lower.k != baseline.k and !improves(lower.state.sse, reduced.state.sse, reduced.state.initial_sse) and reduced.k < lower.k))) lower = reduced;
        }
        if (lower.k == 1) {
            lower.members[0].single_sse = lower.state.sse;
            lower.members[0].single_theta = lower.theta;
        }
        event.flags |= flag.lower;
        event.lower_sse = lower.state.sse;
        if (battle.stage_reference_sse == null) saveLower(battle, lower, index);
    }

    fn prepareResidualSeeds(self: *Work, ranks: usize) !void {
        @memset(self.residual_seeds, null);
        if (ranks < 2 or self.residual_seeds.len == 0) return;
        const g = &self.engine.g;
        const bytes = @as(u64, self.model.fit.px) * 4;
        const capacity: usize = @intCast(@min(ranks, self.seed_target.size / bytes));
        if (capacity < 2) return;
        var poses: [8]detector.Pose = undefined;
        var rows: [8]usize = undefined;
        var seeds: [8]?SingletonSeed = undefined;
        for (0..self.engine.phases.items.len) |phase| {
            var first: usize = 0;
            while (first < ranks) : (first += capacity) {
                const end = @min(first + capacity, ranks);
                var n: usize = 0;
                for (first..end) |rank| {
                    const i = phase * ranks + rank;
                    const member = Member.indexed(self.pool[i], phase, @intCast(rank + 1));
                    if (!member.valid()) continue;
                    if (self.options.flags & 2 != 0 and member.source.proposal_ncc < self.options.min_residual_ncc) continue;
                    poses[n] = .{ .q = member.source.q, .t = self.t, .pat = 0 };
                    rows[n] = i;
                    n += 1;
                }
                if (n < 2) continue;
                for (0..n) |i| g.copy(self.model.fit.original, 0, self.seed_target, i * bytes, bytes);
                try fitSingletonSeeds(self.engine, self.lv, self.frame, phase, poses[0..n], self.seed_target, self.model.optimize_enabled, seeds[0..n]);
                for (rows[0..n], seeds[0..n]) |row, seed| self.residual_seeds[row] = seed;
            }
        }
    }

    fn branch(self: *Work, primary: []const core.Record, stride: usize, k: u32, keep_n: u32, branch_id: u32, initial: Outcome, req: *Request, primary_fits: []?Support, primary_seeds: []const ?SingletonSeed) !Outcome {
        var out = initial;
        var replacements = [_]u32{0} ** 5;
        while (out.progress.stop_reason == 0) {
            try self.check();
            const baseline = out.support;
            const order = baseline.k + 1;
            const ranks = if (order == 1) 1 else keep_n;
            if (order != 1) {
                try self.residual(baseline);
                const nrec = self.engine.phases.items.len * ranks;
                try self.engine.indexPreparedCandidates(&.{self.cell}, self.engine.searchOf(), self.band, ranks, self.pool[0..nrec]);
                try self.check();
                try self.prepareResidualSeeds(ranks);
            }
            var reference: ?f32 = null;
            if (order > 1 and self.options.min_stage_improvement != null) {
                reference = baseline.state.sse;
                for (1..order) |k_lower| {
                    const score = out.archive.score(k_lower);
                    if (finiteScore(score)) reference = @min(reference.?, score);
                }
            }
            var battle: Competition = .{ .lower = baseline, .archive = &out.archive, .stage_reference_sse = reference };
            for (0..self.engine.phases.items.len) |pi| for (0..ranks) |rank| {
                const record = if (order == 1) primary[pi * stride + self.row] else self.pool[pi * ranks + rank];
                const member = Member.indexed(record, pi, if (order == 1) 0 else @intCast(rank + 1));
                const idx = try req.log(.{
                    .cell_index = self.row,
                    .branch = branch_id,
                    .order = order,
                    .attempt = replacements[order],
                    .phase = @intCast(pi),
                    .rank = member.rank,
                    .source_q = record.q,
                    .residual_ncc = if (order == 1 or !member.valid()) nan else record.ncc,
                    .baseline_sse = baseline.state.sse,
                    .result_order = baseline.k,
                });
                // Both trajectories start from exactly the same primary fits.
                // Each still records its own candidates, ties and selection.
                const primary_fit = if (order == 1 and primary_fits.len != 0) &primary_fits[pi] else null;
                const precomputed = if (order == 1)
                    if (primary_seeds.len != 0) primary_seeds[pi * stride + self.row] else null
                else if (self.residual_seeds.len != 0)
                    self.residual_seeds[pi * ranks + rank]
                else
                    null;
                try self.evaluate(baseline, member, &battle, req, idx, primary_fit, precomputed);
            };
            try self.check();
            if (battle.valid > 0 and battle.screened == battle.valid) {
                // Every byte of the accepted prefix remains intact. The stop
                // reason and the separate proposal log describe the screen.
                out.progress.stop_reason = 11;
                break;
            }
            const stage_rejected = if (reference != null)
                battle.gateJoint(self.options.min_stage_improvement.?[order - 2], req.events)
            else
                false;
            out.progress.proposed_count += 1;
            if (stage_rejected) {
                out.progress.stop_reason = 13;
                break;
            }
            const selected = battle.selection() orelse {
                out.progress.stop_reason = if (battle.unresolved) 4 else if (battle.valid == 0) 3 else 5;
                out.progress.boundary_seen = if (battle.unresolved) 2 else if (battle.valid != 0) 1 else 0;
                out.updateEvidence();
                break;
            };
            out.support = selected.support;
            if (selected.replacement) {
                req.events[selected.event].flags |= flag.replace;
                replacements[order] += 1;
                if (replacements[order] >= max_replacements) out.progress.stop_reason = 12;
            } else req.events[selected.event].flags |= flag.add;
            req.events[selected.event].result_order = out.support.k;
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
        return out;
    }

    fn publish(self: *Work, out: Outcome, slots: []overlap.Component, source_q: []Quat, ranks: []i32, norms: []f32, state: *overlap.State) !void {
        const s = out.support;
        state.* = s.state;
        if (s.k == 0) return;
        self.stage(s);
        var magnitudes: [4]f32 = undefined;
        var qs: [4]Quat = undefined;
        self.model.readProfileNorms(s.theta, &magnitudes);
        for (s.members[0..s.k], 0..) |member, j| {
            self.setPose(member);
            self.engine.g.read(self.engine.det.poses, 16, std.mem.asBytes(&qs[j]));
        }
        try self.wait();
        state.phase = @floatFromInt(s.members[s.k - 1].source.phase);
        state.trial_sse = state.sse;
        const best_single = out.archive.score(1);
        state.best_single_sse = if (finiteScore(best_single)) best_single else state.initial_sse;
        for (s.members[0..s.k], 0..) |member, j| {
            slots[j] = member.source;
            slots[j].q = qs[j];
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

/// Shared fixed-identity and blur/gain prefit. All lanes in a dispatch
/// project the same master, each with its own pose and measured target.
fn fitSingletonSeeds(e: *core.Engine, lv: core.Level, frame: spherical.Frame, phase: usize, poses: []const detector.Pose, target: gpu.Buf, optimize: bool, seeds: []?SingletonSeed) !void {
    if (e.cancel.get()) return error.Cancelled;
    const g = &e.g;
    const px = frame.pat_h * frame.pat_w;
    const count: u32 = @intCast(poses.len);
    var fixed_states: [64]overlap.State = undefined;
    var optimized_states: [64]overlap.State = undefined;
    var theta: [64]Theta = undefined;
    {
        var fit = overlap.Fit.init(g, count, px, frame.pat_w, frame.mask, target);
        defer fit.deinit();
        var model = nuisance.Model.init(g, &e.prep, &fit);
        defer model.deinit();
        model.contrast = true;
        model.optimize_enabled = optimize;
        e.det.setPoses(poses);
        e.det.project(e.phases.items[phase].det[@intFromEnum(lv)], frame, count);
        g.copy(e.det.sims, 0, model.raw, 0, @as(u64, count) * px * 4);
        model.singletonSeeds(fixed_states[0..count], optimized_states[0..count], theta[0..count]);
    } // Retire the fit buffers with this submission; host read destinations stay live.
    try g.wait();
    if (e.cancel.get()) return error.Cancelled;
    for (seeds, 0..) |*out, i| {
        var seed: Support = .{ .state = fixed_states[i] };
        if (optimize) Work.retainTheta(&seed, optimized_states[i], theta[i]);
        // Let the scalar path retain its exact unavailable-state diagnostics.
        out.* = if (assessed(seed.state)) .{ .state = seed.state, .theta = seed.theta } else null;
    }
}

fn preparePrimarySeeds(e: *core.Engine, cells: []const u64, primary: []const core.Record, missing: []const u32, lv: core.Level, frame: spherical.Frame, frozen: gpu.Buf, target: gpu.Buf, optimize: bool, seeds: []?SingletonSeed) !void {
    if (seeds.len == 0) return;
    const g = &e.g;
    const bytes = @as(u64, frame.pat_h) * frame.pat_w * 4;
    const capacity: usize = @intCast(target.size / bytes);
    var poses: [64]detector.Pose = undefined;
    var rows: [64]usize = undefined;
    var fit_seeds: [64]?SingletonSeed = undefined;
    for (0..e.phases.items.len) |phase| {
        const phase_records = primary[phase * cells.len ..][0..cells.len];
        var first: usize = 0;
        while (first < cells.len) : (first += capacity) {
            const end = @min(first + capacity, cells.len);
            var n: usize = 0;
            for (first..end) |row| {
                const record = phase_records[row];
                if (missing[row] != 0 or !Member.indexed(record, phase, 0).valid()) continue;
                poses[n] = .{ .q = record.q, .t = try e.positionOf(&e.geo, cells[row], lv), .pat = 0 };
                rows[n] = row;
                n += 1;
            }
            if (n < 2) continue;
            for (rows[0..n], 0..) |row, i| g.copy(frozen, row * bytes, target, i * bytes, bytes);
            try fitSingletonSeeds(e, lv, frame, phase, poses[0..n], target, optimize, fit_seeds[0..n]);
            for (rows[0..n], fit_seeds[0..n]) |row, seed| seeds[phase * cells.len + row] = seed;
        }
    }
}

pub fn run(e: *core.Engine, cells: []const u64, k: u32, primary: []const core.Record, missing: []const u32, hw: [2]u32, outputs: core.Engine.OverlapTrajectoryPayload, optimize: bool, keep_n: u32, req: *Request) !void {
    const g = &e.g;
    const lv = e.levelOf(.search);
    const frame = try e.frame(lv);
    const px = hw[0] * hw[1];
    const bytes = @as(u64, px) * 4;
    const n = cells.len;
    const saved_ahead = e.ahead;
    e.ahead = &.{};
    defer e.ahead = saved_ahead;
    var frozen = g.storage(n * bytes);
    defer g.release(&frozen);
    g.copy(e.pats[@intFromEnum(lv)], 0, frozen, 0, n * bytes);
    const compare_pooled = k > 1 and keep_n > 1;
    // The model, two fits and shared scratch hold about forty full
    // patterns per lane. Check the largest individual binding too.
    const lim = @min(g.lim.max_storage_binding, g.lim.max_buffer);
    const budget = @as(u64, e.set.tile_mb) << 20;
    const capacity = @min(64, lim / (4 * bytes), budget / (40 * bytes), @max(n, if (compare_pooled) keep_n else 1));
    var seed_target: gpu.Buf = if (capacity >= 2) g.storage(capacity * bytes) else .{};
    defer g.release(&seed_target);
    const primary_seeds = try e.gpa.alloc(?SingletonSeed, if (capacity >= 2 and n > 1) primary.len else 0);
    defer e.gpa.free(primary_seeds);
    @memset(primary_seeds, null);
    try preparePrimarySeeds(e, cells, primary, missing, lv, frame, frozen, seed_target, optimize, primary_seeds);
    var target = g.storage(bytes);
    defer g.release(&target);
    var trial_states = g.storage(6 * @sizeOf(overlap.State));
    defer g.release(&trial_states);
    const pool = try e.gpa.alloc(core.Record, e.phases.items.len * keep_n);
    defer e.gpa.free(pool);
    const residual_seeds = try e.gpa.alloc(?SingletonSeed, if (capacity >= 2 and compare_pooled) pool.len else 0);
    defer e.gpa.free(residual_seeds);
    const primary_fits = try e.gpa.alloc(?Support, if (compare_pooled) e.phases.items.len else 0);
    defer e.gpa.free(primary_fits);
    const prm = e.searchOf();
    const band = if (prm.method == .spherical) try e.band(prm.bandwidth, lv) else null;
    for (cells, 0..) |cell, i| {
        if (e.cancel.get()) return error.Cancelled;
        const first = i * k;
        if (missing[i] != 0) {
            outputs.states[i] = .{ .sse = nan, .initial_sse = nan, .single_sse = @splat(nan), .correlation = nan, .condition = nan, .best_single_sse = nan, .trial_sse = nan };
            outputs.progress[i] = .{ .stop_reason = missing[i], .previous_sse = nan, .incremental_improvement = nan, ._pad = unselected };
            continue;
        }
        g.copy(frozen, i * bytes, target, 0, bytes);
        var fit = overlap.Fit.init(g, 1, px, hw[1], frame.mask, target);
        defer fit.deinit();
        var model = nuisance.Model.init(g, &e.prep, &fit);
        defer model.deinit();
        model.contrast = true;
        model.optimize_enabled = optimize;
        var work: Work = .{ .engine = e, .model = &model, .options = req.options, .lv = lv, .frame = frame, .t = try e.positionOf(&e.geo, cell, lv), .row = @intCast(i), .cell = cell, .band = band, .pool = pool, .trial_states = trial_states, .seed_target = seed_target, .residual_seeds = residual_seeds };
        var states: [1]overlap.State = undefined;
        var progress: [1]overlap.Progress = undefined;
        fit.read(&states);
        fit.readProgress(&progress);
        try work.wait();
        const initial: Outcome = .{ .support = .{ .state = states[0] }, .progress = progress[0] };
        const begin = req.used;
        @memset(primary_fits, null);
        const exact = try work.branch(primary, n, k, 1, 0, initial, req, primary_fits, primary_seeds);
        var selected = exact;
        var pooled: ?Outcome = null;
        if (compare_pooled) {
            pooled = try work.branch(primary, n, k, keep_n, 1, initial, req, primary_fits, primary_seeds);
        }
        var source: i32 = -1;
        if (exact.archive.best(if (pooled) |branch| branch.archive else null)) |achieved| {
            const event = &req.events[achieved.event];
            source = @intCast(event.branch);
            if (source == 1) selected = pooled.?;
            if (pooled) |branch| {
                selected.archive = exact.archive;
                selected.archive.include(branch.archive);
            }
            selected.useAchieved(achieved);
            if (req.options.flags & 4 != 0) {
                event.flags |= flag.published;
                if (achieved.from_lower) event.flags |= flag.published_lower;
            }
        }
        for (req.events[begin..req.used]) |*event| if (source >= 0 and event.branch == @as(u32, @intCast(source))) {
            event.flags |= flag.chosen_branch;
        };
        selected.progress._pad = if (source < 0) unselected else @intCast(source);
        outputs.progress[i] = selected.progress;
        outputs.theta[i] = selected.support.theta;
        try work.publish(selected, outputs.components[first..][0..k], req.source_q[first..][0..k], outputs.ranks[first..][0..k], outputs.norms[first..][0..k], &outputs.states[i]);
    }
    if (e.cancel.get()) return error.Cancelled;
}
