//! Oxford acquisition calibration and crystal-frame conventions.
//! Runtime implementation of source-defined transforms, checked against native MTEX.
//! Engine q: sample -> master crystal. Oxford q: CS1 -> Oxford crystal.
const std = @import("std");
const rot = @import("rot.zig");
const Geometry = @import("geometry.zig").Geometry;

pub const Metadata = struct {
    detector_euler: [3]f64 = @splat(std.math.nan(f64)),
    specimen_euler: [3]f64 = @splat(std.math.nan(f64)),
    tilt_axis: f64 = std.math.nan(f64),
    scanning_rotation: f64 = std.math.nan(f64),
    original_tilt: f64 = std.math.nan(f64),
};

pub const Status = enum { not_oxford, supported, missing_detector_orientation, missing_specimen_orientation, missing_sample_tilt, unknown_tilt_axis, unsupported_tilt_axis, unknown_scan_rotation, unsupported_scan_rotation };

/// Internal context, deliberately separate from the public Geometry ABI and its
/// fitted residual. Opening a different source replaces this context.
pub const Calibration = struct {
    metadata: ?Metadata = null,

    pub fn status(self: Calibration) Status {
        const m = self.metadata orelse return .not_oxford;
        for (m.detector_euler) |x| if (!std.math.isFinite(x)) return .missing_detector_orientation;
        for (m.specimen_euler) |x| if (!std.math.isFinite(x)) return .missing_specimen_orientation;
        if (!std.math.isFinite(m.original_tilt)) return .missing_sample_tilt;
        if (!std.math.isFinite(m.tilt_axis)) return .unknown_tilt_axis;
        if (@abs(m.tilt_axis) > 1e-6) return .unsupported_tilt_axis;
        if (!std.math.isFinite(m.scanning_rotation)) return .unknown_scan_rotation;
        if (@abs(m.scanning_rotation) > 1e-6) return .unsupported_scan_rotation;
        return .supported;
    }

    pub fn requireSupported(self: Calibration) !void {
        return switch (self.status()) {
            .supported => {},
            .not_oxford => error.OxfordFrameRequiresOriginalAcquisition,
            .missing_detector_orientation => error.MissingOxfordDetectorOrientation,
            .missing_specimen_orientation => error.MissingOxfordSpecimenOrientation,
            .missing_sample_tilt => error.MissingOxfordSampleTilt,
            .unknown_tilt_axis => error.UnknownOxfordTiltAxis,
            .unsupported_tilt_axis => error.UnsupportedOxfordTiltAxis,
            .unknown_scan_rotation => error.UnknownOxfordScanRotation,
            .unsupported_scan_rotation => error.UnsupportedOxfordScanRotation,
        };
    }

    /// Calibrated baseline plus an optional local detector-column tilt. Sample
    /// tilt edits recompute the Oxford chain; the sample basis B stays fixed.
    pub fn nominal(self: Calibration, g: *const Geometry) rot.Mat3 {
        if (self.status() != .supported) return g.nominal();
        const m = self.metadata.?;
        const tilt = std.math.degreesToRadians(g.sample_tilt_deg);
        // fromBunge is passive: this is exactly native MTEX D.inv().
        const dinv = rot.toMat(rot.fromBunge(m.detector_euler));
        const a = rot.toMat(rot.so3Exp(.{ -tilt, 0, 0 }));
        const base = rot.matMul(rot.matMul(rot.matMul(rot.transpose(rot.toMat(sampleBasis())), a), dinv), rasterBasis());
        return rot.matMul(base, rot.toMat(rot.so3Exp(.{ 0, std.math.degreesToRadians(g.detector_tilt_deg), 0 })));
    }

    pub fn rotation(self: Calibration, g: *const Geometry) rot.Mat3 {
        return rot.matMul(self.nominal(g), rot.toMat(rot.so3Exp(g.rot)));
    }

    pub fn translation(self: Calibration, g: *const Geometry, rows: f64, cols: f64) rot.Vec3 {
        return rot.matVec(self.rotation(g), .{ 0.5 - g.pc[1] * rows, 0.5 - g.pc[0] * cols, g.pc[2] * rows });
    }

    /// Write the full effective pose back into Oxford's detector Euler field.
    /// This includes fitted residuals without changing the fixed sample basis.
    pub fn effectiveDetectorEuler(self: Calibration, g: *const Geometry) !rot.Vec3 {
        try self.requireSupported();
        const a = rot.toMat(rot.so3Exp(.{ std.math.degreesToRadians(g.sample_tilt_deg), 0, 0 }));
        const dinv = rot.matMul(rot.matMul(rot.matMul(a, rot.toMat(sampleBasis())), self.rotation(g)), rot.transpose(rasterBasis()));
        return rot.toBunge(rot.fromMat(dinv));
    }

    pub fn eql(a: Calibration, b: Calibration) bool {
        const x = a.metadata orelse return b.metadata == null;
        const y = b.metadata orelse return false;
        for (x.detector_euler, y.detector_euler) |p, q| if (!same(p, q)) return false;
        for (x.specimen_euler, y.specimen_euler) |p, q| if (!same(p, q)) return false;
        return same(x.tilt_axis, y.tilt_axis) and same(x.scanning_rotation, y.scanning_rotation) and same(x.original_tilt, y.original_tilt);
    }

    pub fn hash(self: Calibration, h: anytype) void {
        h.update(@tagName(self.status()));
        if (self.metadata) |m| {
            h.update(std.mem.asBytes(&m.detector_euler));
            h.update(std.mem.asBytes(&m.specimen_euler));
            h.update(std.mem.asBytes(&m.tilt_axis));
            h.update(std.mem.asBytes(&m.scanning_rotation));
            h.update(std.mem.asBytes(&m.original_tilt));
        }
    }
};

fn same(a: f64, b: f64) bool {
    return a == b or (std.math.isNan(a) and std.math.isNan(b));
}

/// Fixed engine-sample -> Oxford CS1 components for the verified convention.
pub fn sampleBasis() rot.Quat {
    return rot.so3Exp(.{ 0, 0, std.math.pi / 2.0 });
}

/// Engine detector row/column/ray-depth -> Oxford CS3 components.
pub fn rasterBasis() rot.Mat3 {
    return .{ 0, 1, 0, 1, 0, 0, 0, 0, -1 };
}

/// Oxford crystal -> master crystal. Only source-verified Cartesian conventions
/// and validated orthogonal/conventional hexagonal cells are enabled initially.
pub fn crystalBasis(frame: @import("master.zig").CrystalFrame, lattice: [6]f64) !rot.Quat {
    if (frame != .x_a_z_cstar) return error.UnknownMasterCrystalConvention;
    for (lattice) |x| if (!std.math.isFinite(x) or x <= 0) return error.InvalidMasterLattice;
    const a90 = @abs(lattice[3] - 90) < 1e-4;
    const b90 = @abs(lattice[4] - 90) < 1e-4;
    if (a90 and b90 and @abs(lattice[5] - 90) < 1e-4) return rot.quat_identity;
    if (a90 and b90 and @abs(lattice[5] - 120) < 1e-4 and @abs(lattice[0] - lattice[1]) < 1e-5 * lattice[0])
        return rot.so3Exp(.{ 0, 0, std.math.pi / 6.0 });
    return error.UnsupportedMasterCrystalSetting;
}

test "Oxford residual pose and detector Euler roundtrip keep sample basis fixed" {
    const c: Calibration = .{ .metadata = .{ .detector_euler = .{ 0.00722032, 1.5679878, 0.00296268 }, .specimen_euler = .{ 0.2, 0.1, 0.3 }, .tilt_axis = 0, .scanning_rotation = 0, .original_tilt = -std.math.pi / 9.0 } };
    var g: Geometry = .{ .sample_tilt_deg = -20, .rot = .{ 0, 0, 0 } };
    const d = try c.effectiveDetectorEuler(&g);
    for (rot.toMat(rot.fromBunge(d)), rot.toMat(rot.fromBunge(c.metadata.?.detector_euler))) |a, b| try std.testing.expectApproxEqAbs(a, b, 1e-12);
    g.rot = .{ 0.01, -0.02, 0.005 };
    const changed = try c.effectiveDetectorEuler(&g);
    var copy = c;
    copy.metadata.?.detector_euler = changed;
    var baseline = g;
    baseline.rot = .{ 0, 0, 0 };
    for (copy.rotation(&baseline), c.rotation(&g)) |a, b| try std.testing.expectApproxEqAbs(a, b, 1e-12);
}

test "Oxford crystal conversion is per convention and cell setting" {
    const q = try crystalBasis(.x_a_z_cstar, .{ 2.95, 2.95, 4.68, 90, 90, 120 });
    const v = rot.qapply(q, .{ 1, 0, 0 });
    try std.testing.expectApproxEqAbs(@sqrt(3.0) / 2.0, v[0], 1e-12);
    try std.testing.expectApproxEqAbs(0.5, v[1], 1e-12);
    try std.testing.expectEqual(rot.quat_identity, try crystalBasis(.x_a_z_cstar, .{ 4.1, 4.1, 4.1, 90, 90, 90 }));
    try std.testing.expectError(error.UnknownMasterCrystalConvention, crystalBasis(.unknown, .{ 2.95, 2.95, 4.68, 90, 90, 120 }));
    try std.testing.expectError(error.UnsupportedMasterCrystalSetting, crystalBasis(.x_a_z_cstar, .{ 2.95, 3.0, 4.68, 90, 105, 90 }));
}

test "Oxford calibrated detector matrices agree with independent native MTEX cases" {
    const c: Calibration = .{ .metadata = .{ .detector_euler = .{ 0.00722032, 1.5679878, 0.00296268 }, .specimen_euler = .{ 0, 0, 0 }, .tilt_axis = 0, .scanning_rotation = 0, .original_tilt = -std.math.pi / 9.0 } };
    // Generated with pyMTEX 1.0.0 by native byEuler/byAxisAngle compositions,
    // independently of this implementation. Rows/columns are detector raster axes.
    const negative: rot.Mat3 = .{
        0.3446288460628502, -0.0052724508239519, -0.9387242192058842,
        -0.0072285461023605, -0.9999694849061044, 0.0029626639813385,
        -0.9387111944483426, 0.0057645918267715, -0.3446564418351069,
    };
    const positive: rot.Mat3 = .{
        -0.9387111944483427, 0.0057645918267715, -0.3446564418351067,
        -0.0072285461023606, -0.9999694849061044, 0.0029626639813387,
        -0.3446288460628500, 0.0052724508239522, 0.9387242192058836,
    };
    for ([_]f64{ -20, 70 }, [_]rot.Mat3{ negative, positive }) |tilt, expected| {
        const g: Geometry = .{ .sample_tilt_deg = tilt };
        for (c.rotation(&g), expected) |a, b| try std.testing.expectApproxEqAbs(a, b, 1e-12);
    }
}
