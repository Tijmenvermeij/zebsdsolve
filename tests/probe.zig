//! Native GPU probe: open wgpu-native, report the adapter and limits, run one kernel.
//!   zig build probe            (the path to wgpu-native's lib/ is passed by build.zig)
const std = @import("std");
const ze = @import("zebsdsolve");
const gpu = ze.gpu;

pub fn main(init: std.process.Init) !void {
    const gpa = init.gpa;
    var args = try init.minimal.args.iterateAllocator(gpa);
    defer args.deinit();
    _ = args.next();
    const lib_path = args.next() orelse ".deps/wgpu-native/win_amd64/lib";

    const dev = try gpu.Backend.Device.init(gpa, lib_path);
    defer dev.deinit();
    var g = try gpu.Gpu.init(gpa, dev);
    defer g.deinit();
    const lim = g.lim;
    std.debug.print("adapter: {s}\nlimits: storage binding {d} MiB, buffer {d} MiB, workgroup storage {d} B, storage/stage {d}, uniform align {d}\n", .{
        dev.info(), lim.max_storage_binding >> 20, lim.max_buffer >> 20, lim.max_workgroup_storage, lim.max_storage_per_stage, lim.uniform_align,
    });

    // fft.wgsl `fft_lines`: the forward FFT of a unit impulse at point 1 of an 8-point line is
    // e^{−2πi k / 8}.
    const pipe = try g.pipeline("fft/fft_lines", ze.shaders.fft, "fft_lines");
    if (dev.takeError()) |e| std.debug.print("error: {s}\n", .{e});
    var data = g.storage(16 * 4);
    defer g.release(&data);
    var tw = g.storage(1024 * 8);
    defer g.release(&tw);
    var line: [16]f32 = @splat(0);
    line[2] = 1;
    g.writeSlice(data, 0, f32, &line);
    var tw_h: [2048]f32 = undefined;
    for (0..1024) |t| {
        const ang = -2.0 * std.math.pi * @as(f64, @floatFromInt(t)) / 1024.0;
        tw_h[2 * t] = @floatCast(@cos(ang));
        tw_h[2 * t + 1] = @floatCast(@sin(ang));
    }
    g.writeSlice(tw, 0, f32, &tw_h);
    const u = [8]u32{ 8, 3, 1, 1, 8, 1, 1024, 0 };
    var ub: [48]u8 = undefined;
    @memcpy(ub[0..32], std.mem.asBytes(&u));
    @memcpy(ub[32..48], std.mem.asBytes(&[4]f32{ -1, 1, 0, 0 }));
    g.dispatch(pipe, 1, 1, 1, &.{ g.uniform(0, &ub), data.at(1), tw.at(2) });
    var host: [16]f32 = undefined;
    g.read(data, 0, std.mem.sliceAsBytes(&host));
    try g.wait();
    if (dev.takeError()) |e| std.debug.print("error: {s}\n", .{e});
    std.debug.print("fft of an impulse: X[1] = {d:.4} {d:.4}i (expect 0.7071 -0.7071i), X[2] = {d:.4} {d:.4}i (expect 0 -1i)\n", .{ host[2], host[3], host[4], host[5] });
    try checkOverlapRejection(&g);
    try checkManagedOverlapOrders(&g);
    try checkNearPerfectPhaseCompetition(&g);
    try checkNuisanceCandidateIsolation(&g);
    try checkNuisanceCandidateTies(&g);
    try checkOverlapDictionaryCandidates(&g);
    try checkSphericalSeedPages(&g);
    try checkSphericalAllocationGuards(&g);
    try checkOverlapTrajectoryChoice();
    try checkOverlapTrajectoryFinalization();
    try checkProfilePose(&g);
    try checkProfileRawCache(&g);
    try checkProfileSupersededJointEvidence();
    try checkOverlapStageComparison();
}

/// Exercise the SAME candidate-ranking seam the overlap controller uses: an
/// unresolved addition must preserve the previous fit, and later safe phases
/// must still be able to win. No extra numerical test ABI is needed.
fn checkOverlapRejection(g: *gpu.Gpu) !void {
    const model = ze.engine.overlap;
    const a = [_]f32{ 0.5, -0.5, 0.5, -0.5 };
    const b = [_]f32{ 0.5, 0.5, -0.5, -0.5 };
    var original = g.storage(16);
    defer g.release(&original);
    var candidate = g.storage(16);
    defer g.release(&candidate);
    g.writeSlice(original, 0, f32, &b);
    var fit = model.Fit.init(g, 1, 4, 2, .{ 0, 0, 0 }, original);
    defer fit.deinit();
    fit.begin();
    g.writeSlice(candidate, 0, f32, &a);
    fit.candidate(candidate, 1, 0);
    var states: [1]model.State = undefined;
    fit.read(&states);
    try g.wait();
    const previous = states[0];
    fit.begin();
    var opposing: [4]f32 = undefined;
    for (0..4) |i| opposing[i] = -a[i] + b[i] / 4096;
    g.writeSlice(candidate, 0, f32, &opposing);
    fit.candidate(candidate, 2, 1);
    fit.read(&states);
    var bank: [8]f32 = undefined;
    g.read(fit.bank, 0, std.mem.sliceAsBytes(&bank));
    try g.wait();
    if ((@as(u32, @intFromFloat(states[0].flags)) & 16) == 0 or states[0].phase != -1 or states[0].sse != previous.sse or states[0].amplitude[0] != previous.amplitude[0]) return error.OverlapFailedToRetainPriorFit;
    for (0..4) |i| if (bank[i] != a[i] or bank[i + 4] != 0) {
        return error.OverlapRejectedCandidateChangedBank;
    };
    // A flat candidate must also be excluded from ranking.
    g.writeSlice(candidate, 0, f32, &[_]f32{ 0.5, 0.5, 0.5, 0.5 });
    fit.candidate(candidate, 2, 3);
    fit.read(&states);
    try g.wait();
    if (states[0].phase != -1 or states[0].sse != previous.sse) return error.OverlapFlatCandidateWasRanked;
    // Do not restart the step: a later safe phase must still win this same order.
    g.writeSlice(candidate, 0, f32, &b);
    fit.candidate(candidate, 2, 2);
    fit.read(&states);
    try g.wait();
    if (states[0].phase != 2 or states[0].sse > 1e-6 or @abs(states[0].amplitude[1] - 1) > 1e-6 or (@as(u32, @intFromFloat(states[0].flags)) & 16) == 0) return error.OverlapSafeCandidateDidNotWin;
    g.writeSlice(original, 0, f32, &[_]f32{ 0.5, 0.5, 0.5, 0.5 });
    var flat_fit = model.Fit.init(g, 1, 4, 2, .{ 0, 0, 0 }, original);
    defer flat_fit.deinit();
    flat_fit.fixed(2);
    flat_fit.read(&states);
    try g.wait();
    if ((@as(u32, @intFromFloat(states[0].flags)) & 2) == 0 or !std.math.isNan(states[0].condition)) return error.OverlapFlatConditionWasAssessed;
    g.writeSlice(original, 0, f32, &[_]f32{ std.math.nan(f32), 0, 0, 0 });
    var invalid = model.Fit.init(g, 1, 4, 2, .{ 0, 0, 0 }, original);
    defer invalid.deinit();
    var invalid_progress: [1]model.Progress = undefined;
    invalid.read(&states);
    invalid.readProgress(&invalid_progress);
    try g.wait();
    if (invalid_progress[0].stop_reason != 9 or !std.math.isNan(states[0].sse) or @as(u32, @intFromFloat(states[0].flags)) != 8) return error.OverlapInvalidTargetWasAssessed;
    std.debug.print("overlap rank rejection: prior fit/bank retained, flat excluded, later safe phase won; flat/invalid target fail closed\n", .{});
}

/// Independent stopped cells keep their order while the rest of the SAME batch
/// continues. Exact orthogonal scalar vectors isolate GPU/controller semantics.
fn checkManagedOverlapOrders(g: *gpu.Gpu) !void {
    const model = ze.engine.overlap;
    const signs = [4][8]f32{ .{ 1, -1, 1, -1, 1, -1, 1, -1 }, .{ 1, 1, -1, -1, 1, 1, -1, -1 }, .{ 1, 1, 1, 1, -1, -1, -1, -1 }, .{ 1, -1, -1, 1, 1, -1, -1, 1 } };
    var basis: [4][8]f32 = undefined;
    for (0..4) |j| for (0..8) |i| {
        basis[j][i] = signs[j][i] / @sqrt(@as(f32, 8));
    };
    var target: [40]f32 = @splat(0);
    for (1..5) |b| for (0..8) |i| {
        for (0..b) |j| target[b * 8 + i] += basis[j][i] / @sqrt(@as(f32, @floatFromInt(b)));
    };
    var input = g.storage(40 * 4);
    defer g.release(&input);
    var candidate = g.storage(40 * 4);
    defer g.release(&candidate);
    g.writeSlice(input, 0, f32, &target);
    var fit = model.Fit.init(g, 5, 8, 4, .{ 0, 0, 0 }, input);
    defer fit.deinit();
    var states: [5]model.State = undefined;
    var progress: [5]model.Progress = undefined;
    for (0..4) |pass| {
        fit.beginManaged();
        var patterns: [40]f32 = undefined;
        for (0..5) |b| @memcpy(patterns[b * 8 ..][0..8], &basis[pass]);
        g.writeSlice(candidate, 0, f32, &patterns);
        fit.candidateManaged(candidate, @intCast(pass), false);
        fit.read(&states);
        fit.readProgress(&progress);
        try g.wait();
        for (&states, &progress) |*state, *m| {
            if (m.stop_reason != 0) continue;
            if (state.phase < 0) return error.ManagedOverlapUnexpectedMissingProposal;
            m.retained_order += 1;
            m.active_count = 0;
            for (0..m.retained_order) |j| if (state.amplitude[j] > 1e-6) {
                m.active_count += 1;
            };
            if (state.sse <= 1e-8 * state.initial_sse) m.stop_reason = 2;
        }
        fit.writeProgress(&progress);
    }
    for (progress, 0..) |m, b| if (m.retained_order != b or m.active_count != b or m.proposed_count != b or states[b].sse > 1e-6) {
        return error.ManagedOverlapPerCellOrderFailed;
    };
    // A new zero-weight trial is only a boundary candidate. A later phase in
    // that same pass can still supply a positive, improving component.
    var one_target: [8]f32 = undefined;
    for (0..8) |i| one_target[i] = (basis[0][i] + basis[1][i]) / @sqrt(@as(f32, 2));
    g.writeSlice(input, 0, f32, &one_target);
    var boundary = model.Fit.init(g, 1, 8, 4, .{ 0, 0, 0 }, input);
    defer boundary.deinit();
    boundary.beginManaged();
    g.writeSlice(candidate, 0, f32, &basis[0]);
    boundary.candidateManaged(candidate, 0, false);
    boundary.read(states[0..1]);
    boundary.readProgress(progress[0..1]);
    try g.wait();
    progress[0].retained_order = 1;
    boundary.writeProgress(progress[0..1]);
    boundary.beginManaged();
    var negative: [8]f32 = undefined;
    for (0..8) |i| negative[i] = -basis[1][i];
    g.writeSlice(candidate, 0, f32, &negative);
    boundary.candidateManaged(candidate, 1, false);
    boundary.read(states[0..1]);
    boundary.readProgress(progress[0..1]);
    try g.wait();
    if (states[0].phase != -1 or progress[0].boundary_seen != 1 or @abs(states[0].sse - 0.5) > 1e-5) return error.ManagedOverlapBoundaryWasCommitted;
    g.writeSlice(candidate, 0, f32, &basis[1]);
    boundary.candidateManaged(candidate, 2, false);
    boundary.read(states[0..1]);
    try g.wait();
    if (states[0].phase != 2 or states[0].sse > 1e-6) return error.ManagedOverlapBoundaryBlockedLaterPhase;
    std.debug.print("overlap managed orders: 0/1/2/3/4 stable within one batch; zero boundary permits later phase\n", .{});
}

/// A provisional near-perfect fit must not stop competitors in the same order.
fn checkNearPerfectPhaseCompetition(g: *gpu.Gpu) !void {
    const model = ze.engine.overlap;
    const a = [_]f32{ 0.5, -0.5, 0.5, -0.5 };
    const b = [_]f32{ 0.5, 0.5, -0.5, -0.5 };
    var original = g.storage(16);
    defer g.release(&original);
    g.writeSlice(original, 0, f32, &b);
    var candidate = g.storage(16);
    defer g.release(&candidate);
    for (0..2) |managed| {
        var fit = model.Fit.init(g, 1, 4, 2, .{ 0, 0, 0 }, original);
        defer fit.deinit();
        if (managed == 1) fit.beginManaged() else fit.begin();
        var near: [4]f32 = undefined;
        for (0..4) |i| near[i] = b[i] + 5e-5 * a[i];
        g.writeSlice(candidate, 0, f32, &near);
        if (managed == 1) fit.candidateManaged(candidate, 0, false) else fit.candidate(candidate, 1, 0);
        var states: [1]model.State = undefined;
        fit.read(&states);
        try g.wait();
        if (!(states[0].sse > 0 and states[0].sse < 1e-8)) return error.OverlapNearPerfectFixtureInvalid;
        g.writeSlice(candidate, 0, f32, &b);
        if (managed == 1) fit.candidateManaged(candidate, 1, false) else fit.candidate(candidate, 1, 1);
        fit.read(&states);
        try g.wait();
        if (states[0].phase != 1 or states[0].sse != 0) return error.OverlapSkippedSuperiorLaterPhase;
    }
    std.debug.print("overlap phase competition: exact later phase beats near-perfect provisional phase in legacy/managed paths\n", .{});
}

/// Real variable-projection candidates, all cloned from ONE stored-primary K1
/// fit. Independent A-only/B-only runs are the oracle for same-order sequences.
fn checkNuisanceCandidateIsolation(g: *gpu.Gpu) !void {
    const overlap = ze.engine.overlap;
    const Nuisance = ze.engine.overlap_nuisance;
    const Prep = @typeInfo(@FieldType(Nuisance.Model, "prep")).pointer.child;
    const n = 12;
    const px = 256;
    const Result = struct {
        state: [n]overlap.State,
        theta: [n]Nuisance.Theta,
        norms: [n][4]f32,
        residual: [n * px]f32,
        cache: [n][196]f32,
        raw: [4 * n * px]f32,
    };
    var primary: [n * px]f32 = undefined;
    var a: [n * px]f32 = undefined;
    var b: [n * px]f32 = undefined;
    var target: [n * px]f32 = undefined;
    for (0..n) |cell| for (0..px) |i| {
        const x: f32 = @floatFromInt(i % 16);
        const y: f32 = @floatFromInt(i / 16);
        const c: f32 = @floatFromInt(cell);
        const ix = cell * px + i;
        primary[ix] = @sin(x * (0.37 + c * 0.019) + y * 0.41) + 0.4 * @cos(y * 0.7 - x * 0.21);
        b[ix] = @cos(x * 0.91 - y * (0.31 + c * 0.013)) + 0.3 * @sin(x * 0.4 + y * 1.1);
        a[ix] = b[ix] + (0.12 + c * 0.012) * @sin(x * 1.31 + y * 0.83);
        const radial = @max(1.0 - @sqrt((x - 8) * (x - 8) + (y - 8) * (y - 8)) / 8, 0);
        target[ix] = (1 + (0.08 + c * 0.025) * radial) * (0.8 * primary[ix] + 0.6 * b[ix]);
    };
    var prep = Prep.init(g);
    defer prep.deinit();
    var input = g.storage(n * px * 4);
    defer g.release(&input);
    var candidate = g.storage(n * px * 4);
    defer g.release(&candidate);
    var residual = g.storage(n * px * 4);
    defer g.release(&residual);
    g.writeSlice(input, 0, f32, &target);
    prep.filter(input, n, 16, 16, .{});
    var base_fit = overlap.Fit.init(g, n, px, 16, .{ 0, 0, 0 }, input);
    defer base_fit.deinit();
    var base = Nuisance.Model.init(g, &prep, &base_fit);
    defer base.deinit();
    base.contrast = true;
    // Known identity primary assessment; later additions optimize shared theta.
    // This deliberately exposes achieved earlier-single improvements.
    base.optimize_enabled = false;
    base_fit.beginManaged();
    base.begin();
    g.writeSlice(candidate, 0, f32, &primary);
    base.candidate(candidate, 31, 0, true);
    var base_state: [n]overlap.State = undefined;
    var progress: [n]overlap.Progress = undefined;
    base_fit.read(&base_state);
    base_fit.readProgress(&progress);
    try g.wait();
    for (&progress) |*p| p.retained_order = 1;
    base_fit.writeProgress(&progress);
    var results: [5]Result = undefined;
    // Last sequence includes a flat rejected phase and relabels physical A/B.
    for (0..5) |sequence| {
        var fit = overlap.Fit.init(g, n, px, 16, .{ 0, 0, 0 }, input);
        defer fit.deinit();
        var model = Nuisance.Model.init(g, &prep, &fit);
        defer model.deinit();
        model.contrast = true;
        g.copy(base_fit.state, 0, fit.state, 0, n * @sizeOf(overlap.State));
        g.copy(base_fit.progress, 0, fit.progress, 0, n * @sizeOf(overlap.Progress));
        g.copy(base_fit.bank, 0, fit.bank, 0, 4 * n * px * 4);
        g.copy(base.cells, 0, model.cells, 0, n * 196 * 4);
        g.copy(base.committed, 0, model.committed, 0, 4 * n * px * 4);
        fit.beginManaged();
        model.begin();
        const orders = [5][3]u32{ .{ 0, 3, 3 }, .{ 1, 3, 3 }, .{ 0, 1, 3 }, .{ 1, 0, 3 }, .{ 2, 0, 1 } };
        for (orders[sequence]) |which| {
            if (which == 3) continue;
            if (which == 2) {
                g.clear(candidate);
            } else {
                g.writeSlice(candidate, 0, f32, if (which == 0) &a else &b);
            }
            const phase: u32 = if (sequence == 4) 90 - which else 10 + which;
            model.candidate(candidate, phase, 0, false);
        }
        const out = &results[sequence];
        fit.read(&out.state);
        model.readTheta(&out.theta);
        // Finalize controller order before computing the next proposal residual.
        for (&progress) |*p| p.retained_order = 2;
        fit.writeProgress(&progress);
        // The next proposal uses the signed, normalized committed residual.
        fit.residual(0, residual, true);
        g.read(residual, 0, std.mem.sliceAsBytes(&out.residual));
        model.readNorms(&out.norms, true);
        g.read(model.cells, 0, std.mem.asBytes(&out.cache));
        g.read(model.committed, 0, std.mem.sliceAsBytes(&out.raw));
        try g.wait();
        for (0..n) |cell| {
            var energy: f32 = 0;
            var negative = false;
            for (out.residual[cell * px ..][0..px]) |value| {
                energy += value * value;
                negative = negative or value < 0;
            }
            if (!(out.state[cell].sse > 1e-8 * out.state[cell].initial_sse) or @abs(energy - 1) > 2e-4 or !negative) return error.IsolationNextResidualNotSignedUnitEnergy;
        }
    }
    var improved: usize = 0;
    var unique: usize = 0;
    for (0..n) |cell| {
        if (results[0].state[cell].single_sse[0] < base_state[cell].single_sse[0]) improved += 1;
        if (results[0].state[cell].sse == results[1].state[cell].sse) return error.IsolationFixtureHasTiedCandidates;
        unique += 1;
        const winner: usize = if (results[0].state[cell].sse < results[1].state[cell].sse) 0 else 1;
        const expected = &results[winner];
        for (2..5) |sequence| {
            const actual = &results[sequence];
            var state = actual.state[cell];
            const expected_phase: f32 = @floatFromInt(if (sequence == 4) 90 - @as(u32, @intCast(winner)) else 10 + @as(u32, @intCast(winner)));
            if (state.phase != expected_phase) return error.IsolationChangedPhysicalWinner;
            state.phase = expected.state[cell].phase;
            if (sequence == 4) {
                if ((@as(u32, @intFromFloat(state.flags)) & 16) == 0) return error.IsolationLostHistoricalRejection;
                state.flags = @floatFromInt(@as(u32, @intFromFloat(state.flags)) & ~@as(u32, 16));
            }
            if (!std.mem.eql(u8, std.mem.asBytes(&state), std.mem.asBytes(&expected.state[cell]))) return error.IsolationChangedWinningState;
            if (!std.mem.eql(u8, std.mem.asBytes(&actual.theta[cell]), std.mem.asBytes(&expected.theta[cell]))) return error.IsolationChangedWinningTheta;
            if (!std.mem.eql(u8, std.mem.asBytes(&actual.norms[cell]), std.mem.asBytes(&expected.norms[cell]))) return error.IsolationChangedWinningNorms;
            const begin = cell * px;
            if (!std.mem.eql(f32, actual.residual[begin..][0..px], expected.residual[begin..][0..px])) return error.IsolationChangedNextResidual;
            for (0..14) |j| if (actual.cache[cell][32 + j] != expected.cache[cell][32 + j]) return error.IsolationChangedSingletonTheta;
            for (0..2) |j| if (actual.cache[cell][80 + j] != expected.cache[cell][80 + j]) return error.IsolationChangedSingletonSse;
            if (!std.mem.eql(f32, actual.raw[begin..][0..px], primary[begin..][0..px])) return error.IsolationChangedStoredPrimary;
        }
    }
    if (improved == 0) return error.IsolationFixtureDidNotImproveEarlierSingleton;
    std.debug.print("nuisance candidate isolation: {d} unique winners; {d} provisional earlier-single improvements; A/B/AB/BA/rejected+relabeled payloads and next residual bit-identical\n", .{ unique, improved });
}

/// Exact equal-SSE alternatives use raw-projection identity, never phase labels.
fn checkNuisanceCandidateTies(g: *gpu.Gpu) !void {
    const overlap = ze.engine.overlap;
    const Nuisance = ze.engine.overlap_nuisance;
    const Prep = @typeInfo(@FieldType(Nuisance.Model, "prep")).pointer.child;
    const primary = [_]f32{ 1, -1, 1, -1, 1, -1, 1, -1 };
    const a = [_]f32{ 1, 1, -1, -1, 1, 1, -1, -1 };
    const b = [_]f32{ 1, 1, 1, 1, -1, -1, -1, -1 };
    var target: [8]f32 = undefined;
    for (0..8) |i| target[i] = primary[i] + a[i] + b[i];
    var prep = Prep.init(g);
    defer prep.deinit();
    var input = g.storage(32);
    defer g.release(&input);
    var candidate = g.storage(32);
    defer g.release(&candidate);
    g.writeSlice(input, 0, f32, &target);
    prep.filter(input, 1, 2, 4, .{});
    var states: [2][1]overlap.State = undefined;
    for (0..2) |sequence| {
        var fit = overlap.Fit.init(g, 1, 8, 4, .{ 0, 0, 0 }, input);
        defer fit.deinit();
        var model = Nuisance.Model.init(g, &prep, &fit);
        defer model.deinit();
        model.contrast = true;
        model.optimize_enabled = false;
        fit.beginManaged();
        model.begin();
        g.writeSlice(candidate, 0, f32, &primary);
        model.candidate(candidate, 7, 0, true);
        var progress: [1]overlap.Progress = undefined;
        fit.readProgress(&progress);
        try g.wait();
        progress[0].retained_order = 1;
        fit.writeProgress(&progress);
        fit.beginManaged();
        model.begin();
        for (0..2) |j| {
            const is_a = (j == sequence);
            g.writeSlice(candidate, 0, f32, if (is_a) &a else &b);
            model.candidate(candidate, if (is_a) 900 else 100, 0, false);
        }
        fit.read(&states[sequence]);
        var raw: [16]f32 = undefined;
        g.read(model.committed, 0, std.mem.sliceAsBytes(&raw));
        try g.wait();
        if (states[sequence][0].phase != 900 or !std.mem.eql(f32, raw[8..16], &a)) return error.IsolationTieUsedPhaseOrder;
    }
    if (!std.mem.eql(u8, std.mem.asBytes(&states[0]), std.mem.asBytes(&states[1]))) return error.IsolationTieChangedPayload;
    std.debug.print("nuisance exact-SSE ties: raw identity beats phase label and insertion order\n", .{});
}

/// Exact independent score/index oracle, including chunk and lane boundaries,
/// equal scores and a completely tied experimental pattern. No test ABI.
fn checkOverlapDictionaryCandidates(g: *gpu.Gpu) !void {
    const Detector = @FieldType(ze.engine.Engine, "det");
    const total = 517;
    const n = 3;
    const px = 17;
    var pats: [n * px]f32 = @splat(0);
    pats[0] = 1;
    pats[px + 1] = -1;
    var sims: [total * px]f32 = @splat(0);
    var oracle: [n][total]u32 = undefined;
    for (0..total) |d| {
        sims[d * px] = @as(f32, @floatFromInt(@as(i32, @intCast(d % 7)) - 3)) / 16;
        sims[d * px + 1] = @as(f32, @floatFromInt(@as(i32, @intCast(d % 11)) - 5)) / 32;
        for (0..n) |e| oracle[e][d] = @intCast(d);
    }
    const Context = struct {
        images: []const f32,
        e: usize,
        fn score(self: @This(), d: u32) f32 {
            return if (self.e == 0) self.images[d * px] else if (self.e == 1) -self.images[d * px + 1] else 0;
        }
        fn less(self: @This(), a: u32, b: u32) bool {
            const av = self.score(a);
            const bv = self.score(b);
            return av > bv or (av == bv and a < b);
        }
    };
    for (0..n) |e| std.mem.sort(u32, &oracle[e], Context{ .images = &sims, .e = e }, Context.less);
    var detector = Detector.init(g);
    defer detector.deinit();
    var input = g.storage(n * px * 4);
    defer g.release(&input);
    var chunk = g.storage(total * px * 4);
    defer g.release(&chunk);
    g.writeSlice(input, 0, f32, &pats);
    const top_pipe = try g.pipeline("probe/dict_score_candidates", ze.shaders.detector, "dict_score_candidates");
    const one_pipe = try g.pipeline("probe/dict_score", ze.shaders.detector, "dict_score");
    for ([_]u32{ 1, 5, 8 }) |keep| for ([_]u32{ 1, 255, 256, 257, 517 }) |chunk_size| {
        try detector.dictBeginCandidates(n, keep);
        var first: u32 = 0;
        while (first < total) : (first += chunk_size) {
            const count = @min(chunk_size, total - first);
            g.writeSlice(chunk, 0, f32, sims[first * px ..][0 .. count * px]);
            var params: [36]u32 = @splat(0);
            params[1] = 1;
            params[2] = px;
            params[4] = n;
            params[6] = keep;
            params[8] = count;
            params[9] = first;
            params[10] = n;
            g.dispatch(top_pipe, n, 1, 1, &.{ g.uniform(0, std.mem.asBytes(&params)), input.at(3), chunk.at(12), detector.dict_best.at(13) });
        }
        var actual: [n * 8][2]u32 = undefined;
        g.read(detector.dict_best, 0, std.mem.sliceAsBytes(actual[0 .. n * keep]));
        try g.wait();
        for (0..keep) |rank| for (0..n) |e| {
            const hit = actual[rank * n + e];
            if (hit[1] != oracle[e][rank] or @as(f32, @bitCast(hit[0])) != (Context{ .images = &sims, .e = e }).score(oracle[e][rank])) return error.DictionaryCandidatePoolDiffersFromOracle;
        };
        if (keep == 1) {
            try detector.dictBegin(n);
            first = 0;
            while (first < total) : (first += chunk_size) {
                const count = @min(chunk_size, total - first);
                g.writeSlice(chunk, 0, f32, sims[first * px ..][0 .. count * px]);
                var params: [36]u32 = @splat(0);
                params[1] = 1;
                params[2] = px;
                params[4] = n;
                params[8] = count;
                params[9] = first;
                params[10] = n;
                g.dispatch(one_pipe, n, 1, 1, &.{ g.uniform(0, std.mem.asBytes(&params)), input.at(3), chunk.at(12), detector.dict_best.at(13) });
            }
            var ordinary: [n][2]u32 = undefined;
            g.read(detector.dict_best, 0, std.mem.asBytes(&ordinary));
            try g.wait();
            if (!std.mem.eql(u8, std.mem.asBytes(&ordinary), std.mem.sliceAsBytes(actual[0..n]))) return error.DictionaryCandidateOneChangedOrdinaryWinner;
        }
    };
    // Fewer dictionary entries than requested ranks stay explicitly missing.
    try detector.dictBeginCandidates(n, 8);
    g.writeSlice(chunk, 0, f32, sims[0 .. 3 * px]);
    var short_params: [36]u32 = @splat(0);
    short_params[1] = 1;
    short_params[2] = px;
    short_params[4] = n;
    short_params[6] = 8;
    short_params[8] = 3;
    short_params[10] = n;
    g.dispatch(top_pipe, n, 1, 1, &.{ g.uniform(0, std.mem.asBytes(&short_params)), input.at(3), chunk.at(12), detector.dict_best.at(13) });
    var short: [n * 8][2]u32 = undefined;
    g.read(detector.dict_best, 0, std.mem.asBytes(&short));
    try g.wait();
    for (3..8) |rank| for (0..n) |e| {
        if (short[rank * n + e][1] != std.math.maxInt(u32)) return error.DictionaryMissingRankWasFilled;
    };
    std.debug.print("overlap DI candidates: exact top1/5/8 CPU oracle and ordinary top1 parity across chunks1/255/256/257/517, ties and zero scores\n", .{});
}
/// Native scalar-grid regression for bounded SO(3) pages. The four isolated
/// extrema are specified independently of the GPU implementation, including
/// source aliases at each pole and an exact seam tie. No crystal math here.
fn checkSphericalSeedPages(g: *gpu.Gpu) !void {
    const peak = ze.engine.overlap_peaks;
    for ([_]u32{ 16, 64 }) |n| {
        const nc = n / 2;
        const nb = n / 2 + 1;
        const pixels = @as(usize, nb) * nc * n;
        const num = try g.gpa.alloc(f32, pixels);
        defer g.gpa.free(num);
        const den = try g.gpa.alloc(f32, pixels);
        defer g.gpa.free(den);
        var raw = g.storage(pixels * 4);
        defer g.release(&raw);
        var reciprocal = g.storage(pixels * 4);
        defer g.release(&reciprocal);
        var slab_num: gpu.Buf = .{};
        defer g.release(&slab_num);
        var slab_den: gpu.Buf = .{};
        defer g.release(&slab_den);
        var pages: gpu.Buf = .{};
        defer g.release(&pages);
        var answer = g.storage(64);
        defer g.release(&answer);
        var selector = peak.Selector.init(g);
        defer selector.deinit();
        const ja = n / 4 - 1;
        const jb = nb - 4;
        const indices = [_]u32{ 3 * n + nc - 1, ja * nc * n, ((nb - 1) * nc + 6) * n + (11 % nc), (jb * nc + 4) * n + n / 2 };
        for (0..7) |fixture| {
            @memset(num, if (fixture >= 2 and fixture <= 4) @as(f32, 0.2) else -1);
            @memset(den, if (fixture == 3 or fixture == 4) @as(f32, 0) else 1);
            var expected: [4]peak.Key = undefined;
            const values = if (fixture == 1) [_]f32{ -0.05, -0.1, -0.15, -0.2 } else [_]f32{ 0.95, 0.9, 0.85, 0.8 };
            const count: usize = if (fixture == 6) 2 else if (fixture >= 2 and fixture <= 4) 0 else 4;
            if (fixture == 6) {
                @memset(num, 0.5);
                num[nc * n] = 0.4; // first interior ring, c=a=0
                num[indices[1]] = 0.9;
                expected[0] = .{ .value_bits = @bitCast(@as(f32, 0.9)), .index = indices[1] };
                expected[1] = .{ .value_bits = @bitCast(@as(f32, 0.5)), .index = 0 };
            } else for (indices, values, 0..) |index, v, i| {
                expected[i] = .{ .value_bits = @bitCast(v), .index = index };
                if (count != 0) num[index] = v;
            }
            if (fixture == 4) den[indices[1]] = 1;
            if (fixture == 5) num[(ja * nc + nc - 1) * n + n - 1] = values[1];
            if (count != 0 and fixture != 6) {
                const invalid = (jb * nc + nc - 1) * n + n / 4;
                num[invalid] = std.math.inf(f32);
            }
            g.writeSlice(raw, 0, f32, num);
            g.writeSlice(reciprocal, 0, f32, den);
            for ([_]u32{ 1, 3, nb }) |core| {
                const slabs = (nb + core - 1) / core;
                g.ensure(&pages, @as(u64, slabs) * 64);
                for (0..2) |cursor_pass| {
                    const skip: usize = if (cursor_pass != 0 and count > 0) 2 else 0;
                    const after: ?peak.Key = if (skip > 0) expected[skip - 1] else null;
                    selector.setCursor(after);
                    var first: u32 = 0;
                    var si: u32 = 0;
                    while (first < nb) : (first += core) {
                        const lo = first -| 1;
                        const end = @min(nb, first + core + 1);
                        const rows = end - lo;
                        const bytes = @as(u64, rows) * nc * n * 4;
                        g.ensure(&slab_num, bytes);
                        g.ensure(&slab_den, bytes);
                        g.copy(raw, @as(u64, lo) * nc * n * 4, slab_num, 0, bytes);
                        g.copy(reciprocal, @as(u64, lo) * nc * n * 4, slab_den, 0, bytes);
                        selector.slab(.{ .n = n, .nc = nc, .nb = nb, .first = lo, .rows = rows, .core_first = first, .core_rows = @min(core, nb - first), .cursor_set = @intFromBool(after != null) }, slab_num, slab_den, pages, @as(u64, si) * 64);
                        si += 1;
                    }
                    selector.reduce(pages, slabs * 8, answer, 0);
                    var actual: [8]peak.Key = undefined;
                    g.read(answer, 0, std.mem.sliceAsBytes(&actual));
                    try g.wait();
                    for (actual, 0..) |key, i| {
                        if (i < count - skip) {
                            if (key.index != expected[skip + i].index or key.value_bits != expected[skip + i].value_bits) return error.SphericalSeedPageWrongExtremum;
                        } else if (key.index != 0xffffffff) return error.SphericalSeedPageInventedExtremum;
                    }
                }
            }
        }
    }
    std.debug.print("overlap SI seed pages: poles/seams/ties/negative/flat/undefined and cursor/slab reductions pass\n", .{});
}

/// Pure guard checks run before table generation or weight allocations. The
/// independent byte boundaries use native layouts, without allocating them.
fn checkSphericalAllocationGuards(g: *gpu.Gpu) !void {
    const Spherical = @FieldType(ze.engine.Engine, "sp");
    const Tables = @typeInfo(@FieldType(Spherical, "t")).pointer.child;
    const Phase = @typeInfo(@typeInfo(@TypeOf(Spherical.candidatePeaks)).@"fn".params[1].type.?).pointer.child;
    const cap: u64 = 128 << 20;
    try Tables.checkSize(256, cap);
    if (Tables.checkSize(512, cap)) |_| return error.SphericalOversizedTablesWereAllowed else |err| {
        if (err != error.BandwidthExceedsDeviceLimit) return err;
    }
    if (Phase.checkSize(256, 512, 1, cap)) |_| return error.SphericalOversizedPhaseWasAllowed else |err| {
        if (err != error.BandwidthExceedsDeviceLimit) return err;
    }
    // Fold-four Cu fits at L256: spectrum/rden 67,371,008 bytes; each
    // weight table 66,584,576. Equality is supported, one byte less rejected.
    try Phase.checkSize(256, 512, 4, 67_371_008);
    if (Phase.checkSize(256, 512, 4, 67_371_007)) |_| return error.SphericalPhaseBoundaryWasAllowed else |err| {
        if (err != error.BandwidthExceedsDeviceLimit) return err;
    }
    try Tables.checkSize(256, 67_108_864);
    if (Tables.checkSize(256, 67_108_863)) |_| return error.SphericalTableBoundaryWasAllowed else |err| {
        if (err != error.BandwidthExceedsDeviceLimit) return err;
    }
    const actual = @min(g.lim.max_storage_binding, g.lim.max_buffer);
    if (actual >= 269_484_032) try Phase.checkSize(256, 512, 1, actual);
    std.debug.print("overlap SI allocation guards: fake128MiB rejects L256fold1 and L512tables before allocation; foldedCu/equality boundaries and actual limits pass\n", .{});
}

/// Final objective choice is independent of phase labels and stop history.
fn checkOverlapTrajectoryChoice() !void {
    const Engine = ze.engine.Engine;
    const State = ze.engine.overlap.State;
    const Progress = ze.engine.overlap.Progress;
    const valid: Progress = .{ .retained_order = 3, .stop_reason = 1 };
    const prior: Progress = .{ .retained_order = 2, .stop_reason = 4 };
    const missing: Progress = .{ .stop_reason = 6 };
    const reference: State = .{ .sse = 0.2, .initial_sse = 1 };
    if (Engine.overlapTrajectorySource(.{ .sse = 0.1, .initial_sse = 1 }, valid, reference, valid) != 1) return error.TrajectoryMissedLowerPool;
    if (Engine.overlapTrajectorySource(.{ .sse = 0.3, .initial_sse = 1 }, valid, reference, valid) != 0) return error.TrajectoryLostReference;
    if (Engine.overlapTrajectorySource(reference, valid, reference, valid) != 0) return error.TrajectoryTieDidNotChooseReference;
    if (Engine.overlapTrajectorySource(.{ .sse = 0, .initial_sse = 1, .flags = 8 }, valid, reference, valid) != 0) return error.TrajectoryChoseUnresolvedPool;
    if (Engine.overlapTrajectorySource(reference, valid, .{ .sse = std.math.nan(f32), .initial_sse = 1 }, valid) != 1) return error.TrajectoryMissedOnlyValidPool;
    if (Engine.overlapTrajectorySource(reference, missing, reference, missing) != -1) return error.TrajectoryInventedUnavailableObjective;
    if (Engine.overlapTrajectorySource(.{ .sse = -1, .initial_sse = 1 }, valid, .{ .sse = 0, .initial_sse = 0, .flags = 2 }, .{}) != -1) return error.TrajectoryAcceptedInvalidSse;
    if (Engine.overlapTrajectorySource(.{ .sse = 0.1, .initial_sse = 1, .flags = 16 }, prior, reference, valid) != 1) return error.TrajectoryDiscardedValidPriorFit;
    std.debug.print("overlap trajectory choice: valid/tie/one-invalid/both-invalid/prior-valid stop semantics pass\n", .{});
}

/// Deterministic finalization seam: three distinct payloads force pool,
/// reference and unavailable. Cancellation preserves all destination bytes.
fn checkOverlapTrajectoryFinalization() !void {
    const E = ze.engine.Engine;
    const O = ze.engine.overlap;
    const k = 4;
    var pc: [12]O.Component = undefined;
    var rc: [12]O.Component = undefined;
    var pn: [12]f32 = undefined;
    var rn: [12]f32 = undefined;
    var pr: [12]i32 = undefined;
    var rr: [12]i32 = undefined;
    for (0..12) |i| {
        const value: f32 = @floatFromInt(i);
        pc[i] = .{ .q = @splat(value + 1), .amplitude = value + 2, .phase = 0, .proposal_ncc = value + 3, .proposal_nmi = value + 4 };
        rc[i] = .{ .q = @splat(value + 11), .amplitude = value + 12, .phase = 1, .proposal_ncc = value + 13, .proposal_nmi = value + 14 };
        pn[i] = value + 10;
        rn[i] = value + 20;
        pr[i] = @intCast(i);
        rr[i] = -@as(i32, @intCast(i)) - 1;
    }
    var ps = [_]O.State{ .{ .sse = 0.1, .initial_sse = 1 }, .{ .sse = 0.3, .initial_sse = 1 }, .{ .sse = std.math.nan(f32), .flags = 8 } };
    var rs = [_]O.State{ .{ .sse = 0.2, .initial_sse = 1 }, .{ .sse = 0.2, .initial_sse = 1 }, .{ .sse = std.math.nan(f32), .flags = 8 } };
    var pp = [_]O.Progress{ .{ .retained_order = 3, ._pad = 123 }, .{ .retained_order = 3, ._pad = 124 }, .{ .stop_reason = 6, ._pad = 125 } };
    var rp = [_]O.Progress{ .{ .retained_order = 2, ._pad = 223 }, .{ .retained_order = 2, ._pad = 224 }, .{ .stop_reason = 9, ._pad = 225 } };
    var pt: [3][7]f32 = .{ @splat(10), @splat(11), @splat(12) };
    var rt: [3][7]f32 = .{ @splat(20), @splat(21), @splat(22) };
    const before = .{ pc, ps, pp, pt, pn, pr };
    const pooled: E.OverlapTrajectoryPayload = .{ .components = &pc, .states = &ps, .progress = &pp, .theta = &pt, .norms = &pn, .ranks = &pr };
    const reference: E.OverlapTrajectoryPayload = .{ .components = &rc, .states = &rs, .progress = &rp, .theta = &rt, .norms = &rn, .ranks = &rr };
    if (E.finishOverlapTrajectories(true, k, pooled, reference)) |_| return error.TrajectoryFinalizationIgnoredCancel else |err| {
        if (err != error.Cancelled) return err;
    }
    const after_cancel = .{ pc, ps, pp, pt, pn, pr };
    inline for (0..6) |field| if (!std.mem.eql(u8, std.mem.asBytes(&before[field]), std.mem.asBytes(&after_cancel[field]))) return error.TrajectoryCancelledFinalizationChangedPayload;
    try E.finishOverlapTrajectories(false, k, pooled, reference);
    for (0..3) |row| {
        const first = row * k;
        const ec = if (row == 0) before[0][first..][0..k] else rc[first..][0..k];
        const en = if (row == 0) before[4][first..][0..k] else rn[first..][0..k];
        const er = if (row == 0) before[5][first..][0..k] else rr[first..][0..k];
        if (!std.mem.eql(u8, std.mem.sliceAsBytes(pc[first..][0..k]), std.mem.sliceAsBytes(ec)) or !std.mem.eql(f32, pn[first..][0..k], en) or !std.mem.eql(i32, pr[first..][0..k], er)) return error.TrajectoryFinalizationMixedComponents;
        const es = if (row == 0) before[1][row] else rs[row];
        const et = if (row == 0) before[3][row] else rt[row];
        var ep = if (row == 0) before[2][row] else rp[row];
        ep._pad = if (row == 0) 1 else if (row == 1) 0 else 0xffffffff;
        if (!std.mem.eql(u8, std.mem.asBytes(&ps[row]), std.mem.asBytes(&es)) or !std.mem.eql(f32, &pt[row], &et) or !std.mem.eql(u8, std.mem.asBytes(&pp[row]), std.mem.asBytes(&ep))) return error.TrajectoryFinalizationMixedState;
    }
    std.debug.print("overlap trajectory finalization: forced mixed winners/unavailable, whole payload and cancellation sentinel pass\n", .{});
}

/// Actual production pose shader readback for a noncommuting sample-to-crystal
/// seed. Print sufficient f32 precision for independent MTEX verification.
fn checkProfilePose(g: *gpu.Gpu) !void {
    const profile = ze.engine.overlap_profile;
    const seed = [4]f32{ 0.8, 0.4, -0.4, 0.2 };
    var poses = g.storage(32);
    defer g.release(&poses);
    var result: [7][4]f32 = undefined;
    var deltas: [7][3]f32 = @splat(@splat(0));
    for (0..7) |i| {
        if (i > 0) deltas[i][(i - 1) / 2] = std.math.degreesToRadians(@as(f32, if (i % 2 == 1) 0.5 else -0.5));
        g.writeSlice(poses, 16, f32, &seed);
        profile.applyPoseOffset(g, poses, deltas[i]);
        g.read(poses, 16, std.mem.asBytes(&result[i]));
    }
    try g.wait();
    if (!std.mem.eql(f32, &result[0], &seed)) return error.IdentityProfileStepChangedSeed;
    for (result, deltas, 0..) |q, delta, i| {
        var norm: f32 = 0;
        for (q) |v| norm += v * v;
        if (@abs(norm - 1) > 1e-6 or (i > 0 and std.mem.eql(f32, &q, &seed))) return error.InvalidProfilePose;
        std.debug.print("PROFILE_POSE,{d},{d:.9},{d:.9},{d:.9},{d:.9},{d:.9},{d:.9},{d:.9},{d:.9},{d:.9},{d:.9},{d:.9}\n", .{
            i, seed[0], seed[1], seed[2], seed[3], delta[0], delta[1], delta[2], q[0], q[1], q[2], q[3],
        });
    }
}

fn checkProfileRawCache(g: *gpu.Gpu) !void {
    const O = ze.engine.overlap;
    const N = ze.engine.overlap_nuisance;
    const Prep = @typeInfo(@FieldType(N.Model, "prep")).pointer.child;
    const a = [_]f32{ 0.5, -0.5, 0.5, -0.5 };
    const b = [_]f32{ 0.5, 0.5, -0.5, -0.5 };
    const negative = [_]f32{ -0.5, -0.5, 0.5, 0.5 };
    var target = g.storage(16);
    defer g.release(&target);
    g.writeSlice(target, 0, f32, &b);
    var prep = Prep.init(g);
    defer prep.deinit();
    var fit = O.Fit.init(g, 1, 4, 2, .{ 0, 0, 0 }, target);
    defer fit.deinit();
    var model = N.Model.init(g, &prep, &fit);
    defer model.deinit();
    model.contrast = true;
    var states: [4]O.State = undefined;
    for ([_][4]f32{ a, b, negative, b }, 0..) |raw, i| {
        g.writeSlice(model.raw, 0, f32, &raw);
        if (i == 3) g.writeSlice(model.raw, 16, f32, &b);
        model.profileRawChanged(if (i == 3) 2 else 1);
        model.profileFixedTheta(N.identity);
        g.read(model.trial.state, 0, std.mem.asBytes(&states[i]));
    }
    try g.wait();
    if (@abs(states[0].sse - 1) > 1e-6 or states[1].sse > 1e-10 or states[1].amplitude[0] != 1 or
        states[2].amplitude[0] != 0 or @abs(states[2].sse - 1) > 1e-6 or
        !std.math.isNan(states[3].sse) or (@as(u32, @intFromFloat(states[3].flags)) & 8) == 0)
        return error.ProfileRawInvalidationOrCertificateFailed;
    std.debug.print("V6 fixed-theta raw invalidation, NNLS zero face and rank deficiency passed\n", .{});
}

/// A restart follows the replacement, while publication keeps the best
/// complete support from every eligible fit encountered during the search.
fn checkProfileSupersededJointEvidence() !void {
    const profile = ze.engine.overlap_profile;
    const Support = @FieldType(profile.Outcome, "support");
    const single: Support = .{ .k = 1, .state = .{ .initial_sse = 1, .sse = 0.04, .amplitude = .{ 1, 0, 0, 0 } } };
    const first_pair: Support = .{ .k = 2, .state = .{ .initial_sse = 1, .sse = 0.011, .amplitude = .{ 0.8, 0.2, 0, 0 } } };
    var out: profile.Outcome = .{ .support = single, .progress = .{} };
    var initial: profile.Competition = .{ .lower = single, .archive = &out.archive };
    if (!initial.offerJoint(1, single, 6)) return error.ProfileSingletonUnassessed;
    var first: profile.Competition = .{ .lower = single, .lower_event = 7, .archive = &out.archive };
    if (!first.offerJoint(0.97, first_pair, 8)) return error.ProfilePositiveProposalUnassessed;
    const replaced = first.selection().?;
    if (!replaced.replacement or replaced.event != 7 or replaced.support.k != 1) return error.ProfileReplacementNotPrioritized;

    var fresh = first_pair;
    fresh.state.sse = 0.021;
    fresh.state.amplitude = .{ 0.7, 0.3, 0, 0 };
    var reopened: profile.Competition = .{ .lower = replaced.support, .archive = &out.archive };
    if (!reopened.offerJoint(replaced.support.state.sse, fresh, 9)) return error.ProfileFreshPairUnassessed;
    const pair = reopened.selection().?;
    if (pair.replacement or pair.support.state.sse != fresh.state.sse) return error.ProfileRestartReusedPriorResidual;
    var invalid = first_pair;
    invalid.state.flags = 8;
    invalid.state.sse = 0.001;
    if (reopened.offerJoint(0.04, invalid, 11)) return error.ProfileUnresolvedDesignEnteredArchive;
    invalid.state.flags = 0;
    invalid.state.amplitude[1] = 0;
    if (reopened.offerJoint(0.04, invalid, 12)) return error.ProfileZeroFaceEnteredWrongArchiveOrder;

    var triple: Support = .{ .k = 3, .state = .{ .initial_sse = 1, .sse = 0.015, .amplitude = .{ 0.6, 0.3, 0.1, 0 } } };
    var next: profile.Competition = .{ .lower = pair.support, .archive = &out.archive };
    if (!next.offerJoint(pair.support.state.sse, triple, 13) or next.selection().?.support.k != 3)
        return error.ProfileThirdOrderUnassessed;
    const winner = out.archive.best(null).?;
    if (winner.event != 8 or winner.support.k != 2 or winner.support.state.sse != first_pair.state.sse or
        winner.support.state.amplitude[1] != first_pair.state.amplitude[1]) return error.ProfileDiscardedBetterPair;

    triple.state.sse = 0.007;
    if (!next.offerJoint(pair.support.state.sse, triple, 14)) return error.ProfileBetterTripleUnassessed;
    const best = out.archive.best(null).?;
    if (best.event != 14 or best.support.k != 3) return error.ProfileDiscardedBetterTriple;
    out.support = best.support;
    out.updateEvidence();
    if (out.progress.previous_sse != first_pair.state.sse or @abs(out.progress.incremental_improvement - 0.004) > 1e-7)
        return error.ProfileSupersededPairMissingFromEvidence;
    triple.state.sse = first_pair.state.sse;
    var exact: profile.Archive = .{};
    var pooled: profile.Archive = .{};
    var exact_trial: profile.Competition = .{ .lower = single, .archive = &exact };
    var pooled_trial: profile.Competition = .{ .lower = single, .archive = &pooled };
    if (!exact_trial.offerJoint(1, first_pair, 20) or !pooled_trial.offerJoint(1, first_pair, 21) or
        !pooled_trial.offerJoint(1, triple, 22)) return error.ProfileTiedSupportUnassessed;
    if (exact.best(pooled).?.event != 20) return error.ProfileArchiveTieBreakFailed;
    exact = .{};
    pooled = .{};
    var tied_single = single;
    tied_single.state.sse = 0.5000007152557373;
    var tied_pair = first_pair;
    tied_pair.state.sse = 0.5000003576278687;
    if (!exact_trial.offerJoint(1, tied_single, 23) or !exact_trial.offerJoint(1, tied_pair, 24))
        return error.ProfileTiedSupportUnassessed;
    tied_pair.state.sse = 0.5;
    if (!pooled_trial.offerJoint(1, tied_pair, 25)) return error.ProfileTiedSupportUnassessed;
    if (exact.best(pooled).?.event != 24) return error.ProfileToleranceChainAdmittedWorseSupport;
    std.debug.print("V6 restarts independently, publishes the best archived payload, and compares against the best lower order\n", .{});
}

fn checkOverlapStageComparison() !void {
    const profile = ze.engine.overlap_profile;
    const Support = @FieldType(profile.Outcome, "support");
    const single: Support = .{ .k = 1, .state = .{ .initial_sse = 2, .sse = 1.0, .amplitude = .{ 1, 0, 0, 0 } } };
    const pair: Support = .{ .k = 2, .state = .{ .initial_sse = 2, .sse = 0.5, .amplitude = .{ 0.8, 0.2, 0, 0 } } };
    const other_single: Support = .{ .k = 1, .state = .{ .initial_sse = 2, .sse = 0.75, .amplitude = .{ 1, 0, 0, 0 } } };
    const event: profile.Event = .{ .cell_index = 0, .branch = 0, .order = 2, .attempt = 0, .phase = 0, .rank = 1, .source_q = .{ 1, 0, 0, 0 }, .baseline_sse = 1 };
    var denied: profile.Archive = .{};
    var battle: profile.Competition = .{ .archive = &denied, .lower = single, .stage_reference_sse = single.state.sse };
    var events = [_]profile.Event{event};
    if (!battle.offerJoint(single.state.sse, pair, 0)) return error.StagePairInvalid;
    battle.offerLower(other_single, 0);
    if (!battle.gateJoint(0.25, &events) or denied.best(null).?.support.k != 1 or
        events[0].lower_sse != 0.75 or events[0].joint_sse != 0.5 or events[0].flags != profile.flag.stage_checked)
        return error.StageThresholdUsesIncorrectReferenceOrUnits;
    var allowed: profile.Archive = .{};
    var equality: profile.Competition = .{ .archive = &allowed, .lower = single, .stage_reference_sse = single.state.sse };
    events[0] = event;
    if (!equality.offerJoint(single.state.sse, pair, 0)) return error.StagePairInvalid;
    equality.offerLower(other_single, 0);
    if (equality.gateJoint(0.125, &events) or allowed.best(null).?.support.k != 2 or events[0].flags & profile.flag.stage_passed == 0)
        return error.StageThresholdEqualityNotAccepted;
    std.debug.print("Overlap gate uses normalized original energy and the stronger lower comparator; equality passes\n", .{});
}
