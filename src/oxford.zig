//! Oxford acquisition calibration and crystal-frame conventions.
//! Runtime implementation of source-defined transforms, checked against native MTEX.
//! Engine q: sample -> master crystal. Oxford q: CS1 -> Oxford crystal.
const std = @import("std");
const rot = @import("rot.zig");
const Geometry = @import("geometry.zig").Geometry;

/// Oxford H5OINA camera attitude. This version is independent of the CS1
/// orientation-export convention. Version 2 uses detector gnomonic axes.
pub const detector_attitude_convention = "oxford_detector_euler_gnomonic";
pub const detector_attitude_convention_version: u8 = 2;

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
        if (scanDirection(m.scanning_rotation) == null) return .unsupported_scan_rotation;
        return .supported;
    }

    /// Settings signs describe the zero-angle raster basis. Oxford's scan
    /// rotation changes beam travel, not the detector pose or CS1 orientations.
    /// Apply the acquisition half turn once, retaining both caller preferences.
    pub fn scanSigns(self: Calibration, x: i32, y: i32) [2]i32 {
        const direction = if (self.status() == .supported) scanDirection(self.metadata.?.scanning_rotation).? else 1;
        return .{ direction * x, direction * y };
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
        // Oxford D maps detector CS3 to the microscope frame. fromBunge
        // is passive, so transpose it to obtain the source-defined active D.
        const detector = rot.transpose(rot.toMat(rot.fromBunge(m.detector_euler)));
        const a = rot.toMat(rot.so3Exp(.{ -tilt, 0, 0 }));
        const base = rot.matMul(rot.matMul(rot.matMul(rot.transpose(rot.toMat(sampleBasis())), a), detector), rasterBasis());
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
        const detector = rot.matMul(rot.matMul(rot.matMul(a, rot.toMat(sampleBasis())), self.rotation(g)), rot.transpose(rasterBasis()));
        return rot.toBunge(rot.fromMat(rot.transpose(detector)));
    }

    pub fn eql(a: Calibration, b: Calibration) bool {
        const x = a.metadata orelse return b.metadata == null;
        const y = b.metadata orelse return false;
        for (x.detector_euler, y.detector_euler) |p, q| if (!same(p, q)) return false;
        for (x.specimen_euler, y.specimen_euler) |p, q| if (!same(p, q)) return false;
        return same(x.tilt_axis, y.tilt_axis) and same(x.scanning_rotation, y.scanning_rotation) and same(x.original_tilt, y.original_tilt);
    }

    pub fn detectorAttitudeVersion(self: Calibration) u32 {
        return if (self.status() == .supported) detector_attitude_convention_version else 0;
    }

    /// Only checkpoints and geometry-dependent caches that actually used the
    /// Oxford attitude change identity when its interpretation changes.
    pub fn hashDetectorAttitude(self: Calibration, h: anytype) void {
        if (self.detectorAttitudeVersion() != 0) {
            h.update("oxford-detector-cs3-v2");
        }
    }

    pub fn hash(self: Calibration, h: anytype) void {
        h.update(@tagName(self.status()));
        self.hashDetectorAttitude(h);
        if (self.metadata) |m| {
            h.update(std.mem.asBytes(&m.detector_euler));
            h.update(std.mem.asBytes(&m.specimen_euler));
            h.update(std.mem.asBytes(&m.tilt_axis));
            h.update(std.mem.asBytes(&m.scanning_rotation));
            h.update(std.mem.asBytes(&m.original_tilt));
        }
    }
};

/// Only zero and a half turn, modulo one full turn, have validated raster
/// geometry. Oxford stores radians in float32; tolerate that rounding only.
fn scanDirection(angle: f64) ?i32 {
    if (!std.math.isFinite(angle)) return null;
    // Preserve the complete previously supported interval at its exact edges.
    if (@abs(angle) <= 1e-6) return 1;
    const full = 2 * std.math.pi;
    const wrapped = @mod(angle, full);
    if (@min(wrapped, full - wrapped) <= 1e-6) return 1;
    if (@abs(wrapped - std.math.pi) <= 1e-6) return -1;
    return null;
}

fn same(a: f64, b: f64) bool {
    return a == b or (std.math.isNan(a) and std.math.isNan(b));
}

/// Fixed engine-sample -> Oxford CS1 components for the verified convention.
pub fn sampleBasis() rot.Quat {
    return rot.so3Exp(.{ 0, 0, std.math.pi / 2.0 });
}

/// Engine detector row/column/ray-depth -> Oxford CS3: (column,-row,depth).
/// Oxford x points right, y up from the bottom-left origin, z into the screen.
pub fn rasterBasis() rot.Mat3 {
    return .{ 0, 1, 0, -1, 0, 0, 0, 0, 1 };
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
    g.sample_tilt_deg = 38;
    g.detector_tilt_deg = -4.3;
    g.rot = .{ 0.01, -0.02, 0.005 };
    const changed = try c.effectiveDetectorEuler(&g);
    var reopened = c;
    reopened.metadata.?.detector_euler = changed;
    reopened.metadata.?.original_tilt = std.math.degreesToRadians(g.sample_tilt_deg);
    // Native export writes the full attitude into Detector Orientation Euler.
    // On opening that acquisition adoptInfo resets both local refinements.
    const baseline: Geometry = .{ .sample_tilt_deg = std.math.radiansToDegrees(reopened.metadata.?.original_tilt) };
    for (reopened.rotation(&baseline), c.rotation(&g)) |a, b| try std.testing.expectApproxEqAbs(a, b, 1e-12);
    for (rot.toMat(rot.fromBunge(try reopened.effectiveDetectorEuler(&baseline))), rot.toMat(rot.fromBunge(changed))) |a, b|
        try std.testing.expectApproxEqAbs(a, b, 1e-12);
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

test "Oxford camera axes and cropped detector rays match kikuchipy 0.14.2" {
    // Independent oracle: the OxfordH5EBSDReader and get_detector_angles in
    // released kikuchipy 0.14.2, followed by its detector direction cosines.
    // Their sample basis is mapped by diag(-1,-1,1). No fitted alignment.
    // Real HSR6 site 774 file has 128 rows by 156 columns. At bin 8 the
    // raster drops four right columns; PC fractions refer to the full extent.
    // The reader converts the original sample tilt through f32 degrees.
    const Ray = struct { bin: f64, row: f64, col: f64, expected: rot.Vec3 };
    const Case = struct { euler: rot.Vec3, tilt: f64, pc: rot.Vec3, matrix: rot.Mat3, rays: [6]Ray };
    const cases = [_]Case{
        // HSR6/site774
        .{ .euler = .{ 0.007220315281301737, 1.606059074401855, 0.002962682629004121 }, .tilt = 69.99999967119774, .pc = .{ 0.4875591695308685, 0.6206509126350284, 1.102802725508809 }, .matrix = .{ -0.9270392544621672, 0.00521601468329036, -0.374928011592963, -0.002708049004768343, -0.9999702991694756, -0.007215763957993809, -0.3749545134503879, -0.005673973011339428, 0.9270258458497528 }, .rays = .{
            .{ .bin = 1, .row = 0, .col = 0, .expected = .{ 0.1112805288152942, 0.4187483793422502, 0.9012582530578167 } },
            .{ .bin = 1, .row = 64, .col = 78, .expected = .{ -0.2751215876650284, -0.02408138603775137, 0.9611077977246751 } },
            .{ .bin = 1, .row = 127, .col = 155, .expected = .{ -0.5744722488419919, -0.4769715101110011, 0.6651915617721376 } },
            .{ .bin = 4, .row = 31, .col = 38, .expected = .{ -0.5700751450245706, -0.4711997598691662, 0.6730416891429941 } },
            .{ .bin = 8, .row = 0, .col = 18, .expected = .{ 0.09909885681342888, -0.414595483247895, 0.9045938325285641 } },
            .{ .bin = 8, .row = 15, .col = 0, .expected = .{ -0.5746522646628668, 0.4307665275668941, 0.6958555693862043 } },
        } },
        // nontrivial_negative_tilt
        .{ .euler = .{ 0.173, 1.641, -0.089 }, .tilt = -20.0, .pc = .{ 0.47, -0.32, 0.74 }, .matrix = .{ 0.3901246956251316, 0.1972129504946318, -0.899394114957731, 0.09958281133707025, -0.980100738532462, -0.1717143151118387, -0.915361123027193, -0.02257419955002304, -0.4020005223442774 }, .rays = .{
            .{ .bin = 1, .row = 0, .col = 0, .expected = .{ -0.6592864001358774, 0.4683869171801582, -0.5881795120627233 } },
            .{ .bin = 1, .row = 64, .col = 78, .expected = .{ -0.3033306977654711, -0.07641833574831822, -0.9498161536605755 } },
            .{ .bin = 1, .row = 127, .col = 155, .expected = .{ -0.01553851715180198, -0.3810948818403861, -0.9244053469770636 } },
            .{ .bin = 4, .row = 31, .col = 38, .expected = .{ -0.01990295130593279, -0.3780225690760772, -0.9255824165348194 } },
            .{ .bin = 8, .row = 0, .col = 18, .expected = .{ -0.4111270620547584, -0.6601732487354621, -0.6286062523553901 } },
            .{ .bin = 8, .row = 15, .col = 0, .expected = .{ -0.1704381576226368, 0.3363820274896678, -0.9261738314205069 } },
        } },
        // nominal_axes
        .{ .euler = .{ 0, 1.570796326794897, 0 }, .tilt = 70.0, .pc = .{ 0.5, 0.5, 0.65 }, .matrix = .{ -0.9396926207859086, -1.561251128379126e-16, -0.3420201433256685, 2.081668171172169e-16, -1, -1.387778780781446e-16, -0.3420201433256685, -1.942890293094024e-16, 0.9396926207859088 }, .rays = .{
            .{ .bin = 1, .row = 0, .col = 0, .expected = .{ 0.2396804820608705, 0.5950854121686913, 0.7670897071028183 } },
            .{ .bin = 1, .row = 64, .col = 78, .expected = .{ -0.3476547790648707, -0.006009398356246112, 0.9376033498898946 } },
            .{ .bin = 1, .row = 127, .col = 155, .expected = .{ -0.6766816606505753, -0.5950854121686916, 0.4335611633474541 } },
            .{ .bin = 4, .row = 31, .col = 38, .expected = .{ -0.6742286117560236, -0.5909033247564554, 0.4430000450127655 } },
            .{ .bin = 8, .row = 0, .col = 18, .expected = .{ 0.2248676279417733, -0.5636692085684755, 0.7948028517913286 } },
            .{ .bin = 8, .row = 15, .col = 0, .expected = .{ -0.6707372718667522, 0.5850535459611201, 0.45587702342525 } },
        } },
    };
    for (cases) |case| {
        const cal: Calibration = .{ .metadata = .{
            .detector_euler = case.euler,
            .specimen_euler = .{ 0, 0, 0 },
            .original_tilt = std.math.degreesToRadians(case.tilt),
            .tilt_axis = 0,
            .scanning_rotation = 0,
        } };
        const geo: Geometry = .{ .sample_tilt_deg = case.tilt, .pc = case.pc };
        const R = cal.rotation(&geo);
        for (R, case.matrix) |actual, expected| try std.testing.expectApproxEqAbs(expected, actual, 2e-7);
        for (case.rays) |ray| {
            const origin = cal.translation(&geo, 128 / ray.bin, 156 / ray.bin);
            const direction = rot.normalize(rot.add(origin, rot.matVec(R, .{ ray.row, ray.col, 0 })));
            for (direction, ray.expected) |actual, expected| try std.testing.expectApproxEqAbs(expected, actual, 3e-7);
        }
    }
}

test "only supported Oxford calibrations acquire the corrected detector attitude fingerprint" {
    const Sha = std.crypto.hash.sha2.Sha256;
    const valid: Calibration = .{ .metadata = .{ .detector_euler = .{ 0, 1, 0 }, .specimen_euler = .{ 0, 0, 0 }, .tilt_axis = 0, .scanning_rotation = 0, .original_tilt = 0 } };
    const unsupported: Calibration = .{ .metadata = .{} };
    var corrected = Sha.init(.{});
    valid.hashDetectorAttitude(&corrected);
    var legacy = Sha.init(.{});
    (Calibration{}).hashDetectorAttitude(&legacy);
    unsupported.hashDetectorAttitude(&legacy);
    var empty = Sha.init(.{});
    try std.testing.expectEqual(@as(u32, 2), valid.detectorAttitudeVersion());
    try std.testing.expectEqual(@as(u32, 0), unsupported.detectorAttitudeVersion());
    const unchanged = empty.finalResult();
    try std.testing.expectEqual(unchanged, legacy.finalResult());
    try std.testing.expect(!std.mem.eql(u8, &unchanged, &corrected.finalResult()));
}

test "Oxford per-cell positions and plane frame use the calibrated baseline" {
    const c: Calibration = .{ .metadata = .{ .detector_euler = .{ 0.00722032, 1.5679878, 0.00296268 }, .specimen_euler = .{ 0, 0, 0 }, .tilt_axis = 0, .scanning_rotation = 0, .original_tilt = -std.math.pi / 9.0 } };
    for ([_]f64{ 0, std.math.pi }) |angle| {
        var calibrated = c;
        calibrated.metadata.?.scanning_rotation = angle;
        const signs = calibrated.scanSigns(-1, 1);
        var g: Geometry = .{ .pc = .{ 0.42, 0.21, 0.5 }, .sample_tilt_deg = -20, .rot = .{ 0.02, -0.03, 0.04 }, .px_um = 70, .step_x_um = 2, .step_y_um = 3 };
        const R = calibrated.rotation(&g);
        const t = g.positionForRotation(R, 4, 6, 3, 5, signs[0], signs[1], 2.5, 24.4, 32.4);
        const want = rot.sub(calibrated.translation(&g, 24.4, 32.4), g.scanOffset(4, 6, 3, 5, signs[0], signs[1], 2.5));
        for (t, want) |a, b| try std.testing.expectApproxEqAbs(a, b, 1e-12);
        // Construct an affine plane from calibrated step-model PCs. Fitting and decoding it
        // must recover the same residual relative to Oxford's full nominal attitude.
        const plane = @import("plane.zig");
        const rows = 30;
        const cols = 40;
        const fr = plane.Frame.of(rows, cols, 1.25);
        var points: [30]plane.Point = undefined;
        for (0..30) |k| {
            const row: f64 = @floatFromInt((k * 7) % rows);
            const col: f64 = @floatFromInt((k * 11) % cols);
            const p = g.positionForRotation(R, row, col, fr.r0, fr.c0, signs[0], signs[1], 1, 60, 75);
            const pc = Geometry.pcOf(R, p, 60, 75);
            points[k] = .{ .col = col, .row = row, .y = .{ pc[0] * fr.aspect, pc[1], pc[2] } };
        }
        g.plane = plane.fit(.affine, &points, fr).?;
        g.plane_kind = 1;
        const pc = plane.at(g.plane, 4, 6);
        const got = Geometry.pcOf(R, g.positionForRotation(R, 4, 6, fr.r0, fr.c0, signs[0], signs[1], 2.5, 24.4, 30.5), 24.4, 30.5);
        for (pc, got) |a, b| try std.testing.expectApproxEqAbs(a, b, 1e-12);
        const said = plane.saidWithNominal(g.plane, &g, calibrated.nominal(&g), rows, cols, 1.25, 60, signs[0], signs[1]);
        for (g.rot, said.rot) |a, b| try std.testing.expectApproxEqAbs(a, b, 1e-9);
        for (said.step) |step| try std.testing.expectApproxEqAbs(@as(f64, 1), step, 1e-9);
    }
}

test "Oxford zero and half-turn scan rotations preserve pose and caller signs" {
    const base: Calibration = .{ .metadata = .{ .detector_euler = .{ 6.252466, 1.594263, 6.263398 }, .specimen_euler = .{ 0, 0, 0 }, .tilt_axis = 0, .scanning_rotation = 0, .original_tilt = 1.2217305 } };
    const g: Geometry = .{ .sample_tilt_deg = 70, .px_um = 70, .step_x_um = 2, .step_y_um = 3 };
    // Fixed expectations do not derive their answer from scanDirection.
    try std.testing.expectEqual([2]i32{ -1, -1 }, base.scanSigns(-1, -1));
    var half = base;
    half.metadata.?.scanning_rotation = @as(f64, @as(f32, std.math.pi));
    try std.testing.expectEqual([2]i32{ 1, 1 }, half.scanSigns(-1, -1));
    try std.testing.expectEqual([2]i32{ 1, -1 }, half.scanSigns(-1, 1));
    const signed = half.scanSigns(-1, 1);
    const offset = g.scanOffset(4, 6, 3, 5, signed[0], signed[1], 1);
    for (offset, [_]f64{ -3.0 / 70.0, 2.0 / 70.0, 0 }) |a, expected| try std.testing.expectApproxEqAbs(expected, a, 1e-12);
    for ([_]f64{ 0, -1e-6, 1e-6, 2 * std.math.pi, -4 * std.math.pi, @as(f64, @as(f32, 4 * std.math.pi)), std.math.pi, -std.math.pi, @as(f64, @as(f32, std.math.pi)) }) |angle| {
        var c = base;
        c.metadata.?.scanning_rotation = angle;
        try std.testing.expectEqual(Status.supported, c.status());
        for (c.nominal(&g), base.nominal(&g)) |a, b| try std.testing.expectApproxEqAbs(a, b, 1e-12);
        for (try c.effectiveDetectorEuler(&g), try base.effectiveDetectorEuler(&g)) |a, b| try std.testing.expectApproxEqAbs(a, b, 1e-12);
        for ([_]i32{ -1, 1 }) |x| for ([_]i32{ -1, 1 }) |y| {
            const signs = c.scanSigns(x, y);
            const is_half = scanDirection(angle).? == -1;
            try std.testing.expectEqual(if (is_half) -x else x, signs[0]);
            try std.testing.expectEqual(if (is_half) -y else y, signs[1]);
            const turned_offset = g.scanOffset(4, 6, 3, 5, signs[0], signs[1], 1);
            const zero = g.scanOffset(4, 6, 3, 5, x, y, 1);
            for (turned_offset, zero) |a, b| try std.testing.expectApproxEqAbs(if (is_half) -b else b, a, 1e-12);
        };
    }
    for ([_]f64{ std.math.pi / 2.0, std.math.pi + 2e-6, 2 * std.math.pi + 2e-6 }) |angle| {
        var c = base;
        c.metadata.?.scanning_rotation = angle;
        try std.testing.expectEqual(Status.unsupported_scan_rotation, c.status());
        try std.testing.expectEqual([2]i32{ -1, 1 }, c.scanSigns(-1, 1));
    }
}

test "Oxford nonideal detector rays follow source CS3 to microscope calibration" {
    // Independent native MTEX matrix columns are H*(-y), H*x, H*z,
    // where H=B.inv()*Rx(-tilt)*D uses S05's float32 header.
    // No production helper supplies the expected columns.
    const c: Calibration = .{ .metadata = .{ .detector_euler = .{ 6.252466201782227, 1.594262957572937, 6.263398170471191 }, .specimen_euler = .{ 0, 0, 0 }, .tilt_axis = 0, .scanning_rotation = 3.1415927410125732, .original_tilt = 1.2217304706573486 } };
    const g: Geometry = .{ .sample_tilt_deg = 69.99999967119774 };
    const expected: rot.Mat3 = .{
        -0.9310221761103453, -0.0289316252125052, -0.3638140578005752,
        0.0190559573659648,  -0.9993467982882921, 0.0307058176866571,
        -0.3644647830444304, 0.0216549720272926,  0.9309653506478496,
    };
    for (c.nominal(&g), expected) |actual, want| try std.testing.expectApproxEqAbs(want, actual, 1e-12);
    const effective = try c.effectiveDetectorEuler(&g);
    for (rot.toMat(rot.fromBunge(effective)), rot.toMat(rot.fromBunge(c.metadata.?.detector_euler))) |actual, want| try std.testing.expectApproxEqAbs(want, actual, 1e-12);
}
