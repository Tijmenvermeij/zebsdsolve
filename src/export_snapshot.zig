//! Shared export payload. HDF5 containers are written by established HDF5 runtimes.
const std = @import("std");
const engine = @import("engine.zig");
const rot = @import("rot.zig");
const oxford = @import("oxford.zig");

fn value(w: *std.Io.Writer, v: anytype) !void {
    try std.json.Stringify.value(v, .{}, w);
}

fn finite(w: *std.Io.Writer, v: f64) !void {
    if (std.math.isFinite(v)) try value(w, v) else try w.writeAll("null");
}

fn numbers(w: *std.Io.Writer, data: anytype) !void {
    try w.writeByte('[');
    for (data, 0..) |n, i| {
        if (i != 0) try w.writeByte(',');
        if (@typeInfo(@TypeOf(n)) == .float and !std.math.isFinite(n)) {
            try w.writeAll("null");
        } else try value(w, n);
    }
    try w.writeByte(']');
}

fn quat(q: anytype, transform: ?rot.Quat, crystal: rot.Quat) rot.Quat {
    const raw: rot.Quat = .{ q[0], q[1], q[2], q[3] };
    return if (transform) |t| rot.qstd(rot.qmul(rot.qconj(crystal), rot.qmul(raw, rot.qconj(t)))) else raw;
}

pub fn snapshot(e: *engine.Engine, transform: ?rot.Quat) ![]u8 {
    return snapshotImpl(e, transform, false);
}

pub fn snapshotOxford(e: *engine.Engine, transform: ?rot.Quat) ![]u8 {
    return snapshotImpl(e, transform, true);
}

fn snapshotImpl(e: *engine.Engine, requested: ?rot.Quat, is_oxford: bool) ![]u8 {
    const info = try e.info();
    if (!e.res.context_consistent) return error.MixedIndexingContextsReindexScan;
    const context = e.res.context orelse return error.NoIndexedResults;
    if (context.phase_generation != e.phase_generation) return error.PhasesChangedReindexScan;
    const geometry = context.geometry;
    if (e.phases.items.len == 0) return error.NoPhases;
    for (e.phases.items) |p| if (p.master.lattice == null) return error.MissingPhaseLattice;
    var transform = requested;
    const crystal = try e.gpa.alloc(rot.Quat, e.phases.items.len);
    defer e.gpa.free(crystal);
    @memset(crystal, rot.quat_identity);
    if (is_oxford) {
        if (transform == null or context.calibration.metadata != null) {
            try context.calibration.requireSupported();
            if (geometry.plane_kind == 0 and geometry.px_um != 0) return error.OxfordScanTranslationNotValidated;
        }
        for (geometry.pc) |x| if (!std.math.isFinite(x)) return error.InvalidOxfordGeometry;
        for (geometry.rot) |x| if (!std.math.isFinite(x)) return error.InvalidOxfordGeometry;
        if (!std.math.isFinite(geometry.sample_tilt_deg) or !std.math.isFinite(geometry.detector_tilt_deg)) return error.InvalidOxfordGeometry;
        if (transform == null) {
            transform = oxford.sampleBasis();
        }
        for (e.phases.items, crystal) |p, *c| c.* = try oxford.crystalBasis(p.master.crystal_frame, p.master.lattice.?);
    }
    if (transform) |q| {
        var norm: f64 = 0;
        for (q) |x| {
            if (!std.math.isFinite(x)) return error.InvalidExportFrame;
            norm += x * x;
        }
        if (@abs(norm - 1) > 1e-6) return error.InvalidExportFrame;
        if (is_oxford and requested != null and context.calibration.metadata != null) {
            const fixed = oxford.sampleBasis();
            var dot: f64 = 0;
            for (q, fixed) |a, b| dot += a * b;
            if (@abs(dot) / @sqrt(norm) < 1 - 1e-10) return error.OxfordFrameOverrideConflictsWithAcquisition;
        }
    }
    var out: std.Io.Writer.Allocating = .init(e.gpa);
    errdefer out.deinit();
    const w = &out.writer;
    try w.writeAll("{\"schema_version\":1,\"version\":");
    try value(w, engine.version);
    try w.writeAll(",\"scan\":");
    try value(w, .{ .rows = info.rows, .cols = info.cols, .pat_h = info.pat_h, .pat_w = info.pat_w, .format = switch (e.source) {
        .file => |s| s.formatName(),
        else => "memory",
    } });
    try w.writeAll(",\"geometry\":");
    try value(w, geometry);
    try w.writeAll(",\"calibration\":{\"kind\":");
    try value(w, if (context.calibration.metadata != null) "oxford" else "legacy");
    try w.writeAll(",\"status\":");
    try value(w, @tagName(context.calibration.status()));
    if (context.calibration.metadata) |m| {
        try w.writeAll(",\"source_detector_euler_radians\":");
        try numbers(w, m.detector_euler);
        try w.writeAll(",\"source_specimen_euler_radians\":");
        try numbers(w, m.specimen_euler);
        try w.writeAll(",\"source_tilt_radians\":");
        try finite(w, m.original_tilt);
        try w.writeAll(",\"tilt_axis_radians\":");
        try finite(w, m.tilt_axis);
        try w.writeAll(",\"scanning_rotation_radians\":");
        try finite(w, m.scanning_rotation);
        if (context.calibration.status() == .supported) {
            try w.writeAll(",\"effective_detector_euler_radians\":");
            try numbers(w, try context.calibration.effectiveDetectorEuler(&geometry));
            try w.writeAll(",\"effective_tilt_radians\":");
            try value(w, std.math.degreesToRadians(geometry.sample_tilt_deg));
        }
    }
    try w.writeByte('}');
    if (is_oxford) {
        try w.writeAll(",\"oxford_frame\":{\"convention_version\":1,\"engine_to_cs1\":");
        try numbers(w, transform.?);
        try w.writeAll(",\"provenance\":");
        try value(w, if (requested == null) "Inherited Oxford acquisition: fixed engine-to-CS1 basis, full detector calibration, phase-specific master crystal conversion" else "Explicit engine-to-CS1 override; phase-specific master crystal conversion");
        try w.writeAll(",\"automatic\":");
        try value(w, requested == null);
        try w.writeAll(",\"phase_generation\":");
        try value(w, context.phase_generation);
        const preserve_source_pc = geometry.plane_kind != 0;
        try w.writeAll(",\"preserve_source_pattern_centres\":");
        try value(w, preserve_source_pc);
        try w.writeAll(",\"warnings\":[");
        if (preserve_source_pc) try value(w, "Orientations were indexed with a fitted source plane. Standard Oxford pattern-centre fields retain acquisition values. Fitted values are stored in /zebsdsolve/PatternCentres.");
        try w.writeAll("],\"phase_crystal_to_master\":[");
        for (crystal, 0..) |c, i| {
            if (i != 0) try w.writeByte(',');
            try numbers(w, c);
        }
        try w.writeAll("],\"master_crystal_convention\":\"X||a,Z||c*\",\"output_crystal_convention\":\"X||a*,Z||c\"}");
    }
    try w.writeAll(",\"pattern_centres\":[");
    for (0..e.res.phase.len) |i| {
        const t = geometry.positionForRotation(context.calibration.rotation(&geometry), @floatFromInt(i / info.cols), @floatFromInt(i % info.cols), 0.5 * @as(f64, @floatFromInt(info.rows - 1)), 0.5 * @as(f64, @floatFromInt(info.cols - 1)), context.settings.scan_x_sign, context.settings.scan_y_sign, 1, @floatFromInt(info.pat_h), @floatFromInt(info.pat_w));
        const pc = engine.Geometry.pcOf(context.calibration.rotation(&geometry), t, @floatFromInt(info.pat_h), @floatFromInt(info.pat_w));
        for (pc, 0..) |x, k| {
            if (i != 0 or k != 0) try w.writeByte(',');
            try value(w, x);
        }
    }
    try w.writeAll("],\"settings\":");
    const settings = try context.settings.toJson(e.gpa);
    defer e.gpa.free(settings);
    try w.writeAll(settings);
    try w.writeAll(",\"phases\":[");
    for (e.phases.items, 0..) |p, i| {
        if (i != 0) try w.writeByte(',');
        try value(w, .{ .name = p.master.name, .space_group = p.master.space_group, .laue = @intFromEnum(p.master.laue), .lattice = p.master.lattice.?, .master_source = p.master.source, .master_source_format_version = p.master.source_format_version, .master_crystal_convention = p.master.crystal_frame.label() });
    }
    try w.writeAll("],\"results\":{\"quaternions\":[");
    const n = e.res.phase.len;
    for (0..n) |i| {
        const q = if (e.res.phase[i] >= 0) quat(e.res.quat[4 * i ..][0..4], transform, crystal[@intCast(e.res.phase[i])]) else rot.quat_identity;
        for (q, 0..) |x, k| {
            if (i != 0 or k != 0) try w.writeByte(',');
            try value(w, if (e.res.phase[i] >= 0) x else 0);
        }
    }
    try w.writeAll("],\"euler\":[");
    for (0..n) |i| {
        const b = if (e.res.phase[i] >= 0) rot.toBunge(quat(e.res.quat[4 * i ..][0..4], transform, crystal[@intCast(e.res.phase[i])])) else .{ 0.0, 0.0, 0.0 };
        for (b, 0..) |x, k| {
            if (i != 0 or k != 0) try w.writeByte(',');
            try value(w, x);
        }
    }
    try w.writeAll("],\"phase\":");
    try numbers(w, e.res.phase);
    inline for (.{ "ncc", "nmi", "z", "sweep" }) |field| {
        try w.print(",\"{s}\":", .{field});
        try numbers(w, @field(e.res, field));
    }
    try w.print("}},\"candidates\":{{\"phase_count\":{d},\"quaternions\":[", .{e.res.candidate_phases});
    for (e.res.candidate_sweep, 0..) |sw, i| {
        const q = if (sw >= 0) quat(e.res.candidates[i].q, transform, crystal[i / n]) else .{ 0.0, 0.0, 0.0, 0.0 };
        for (q, 0..) |x, k| {
            if (i != 0 or k != 0) try w.writeByte(',');
            try value(w, x);
        }
    }
    try w.writeAll("],\"euler\":[");
    for (e.res.candidate_sweep, 0..) |sw, i| {
        const b = if (sw >= 0) rot.toBunge(quat(e.res.candidates[i].q, transform, crystal[i / n])) else .{ 0.0, 0.0, 0.0 };
        for (b, 0..) |x, k| {
            if (i != 0 or k != 0) try w.writeByte(',');
            try value(w, x);
        }
    }
    try w.writeByte(']');
    inline for (.{ "ncc", "nmi", "z" }) |field| {
        try w.print(",\"{s}\":[", .{field});
        for (e.res.candidate_sweep, 0..) |sw, i| {
            if (i != 0) try w.writeByte(',');
            const x: f32 = if (sw >= 0) @field(e.res.candidates[i], field) else 0;
            try value(w, if (std.math.isFinite(x)) x else 0);
        }
        try w.writeByte(']');
    }
    try w.writeAll(",\"sweep\":");
    try numbers(w, e.res.candidate_sweep);
    try w.writeAll("}}");
    return out.toOwnedSlice();
}
