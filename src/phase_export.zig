//! Crystal-coordinate adapter for text formats. The engine's sample frame is unchanged.
//! MTEX/pyMTEX are not available in the native/Wasm engine: this small metric adapter mirrors
//! crystalFrame alignment and is checked against native pyMTEX reference-frame transforms.
const std = @import("std");
const master = @import("master.zig");
const rot = @import("rot.zig");
const sym = @import("sym.zig");

pub const Format = enum { ang, ctf };

pub fn rotation(m: *const master.Master, format: Format) !rot.Quat {
    if (m.crystal_basis != .a_cstar) return error.UnsupportedCrystalBasis;
    if (m.invalid_lattice) return error.InvalidPhaseLattice;
    return latticeRotation(m.lattice orelse return error.MissingPhaseLattice, m.laue, format);
}

fn near(a: f64, b: f64) bool {
    return @abs(a - b) <= 1e-5 * @max(@max(@abs(a), @abs(b)), 1);
}

pub fn latticeRotation(cell: [6]f64, laue: sym.Laue, format: Format) !rot.Quat {
    if (!master.validLattice(cell)) return error.InvalidPhaseLattice;
    const trihex = switch (laue) {
        .c3, .d3, .c6, .d6 => true,
        else => false,
    };
    const orthogonal = switch (laue) {
        .d2, .c4, .d4, .t, .o => true,
        else => false,
    };
    // Native importers fix conventional symmetry axes. Refuse incompatible or unresolved
    // settings rather than silently exporting a different crystal metric.
    if (trihex and !(near(cell[0], cell[1]) and near(cell[3], 90) and near(cell[4], 90) and near(cell[5], 120))) return error.UnsupportedCrystalBasis;
    if (orthogonal and !(near(cell[3], 90) and near(cell[4], 90) and near(cell[5], 90))) return error.UnsupportedCrystalBasis;
    if ((laue == .c4 or laue == .d4 or laue == .t or laue == .o) and !near(cell[0], cell[1])) return error.InvalidPhaseLattice;
    if ((laue == .t or laue == .o) and !near(cell[0], cell[2])) return error.InvalidPhaseLattice;
    if (laue == .c2 and !(near(cell[3], 90) and near(cell[5], 90))) return error.UnsupportedCrystalBasis;
    // EDAX uses a/c* for triclinic, trigonal and hexagonal crystals, default a*/c elsewhere.
    if (orthogonal or (format == .ang and (laue == .c1 or trihex))) return rot.quat_identity;
    const ca = @cos(std.math.degreesToRadians(cell[3]));
    const cb = @cos(std.math.degreesToRadians(cell[4]));
    const cg = @cos(std.math.degreesToRadians(cell[5]));
    const sg = @sin(std.math.degreesToRadians(cell[5]));
    // Direct axes in source coordinates X||a, Z||c*. Lengths cancel in axis normalization.
    const b: rot.Vec3 = .{ cg, sg, 0 };
    const c: rot.Vec3 = .{ cb, (ca - cb * cg) / sg, @sqrt(1 + 2 * ca * cb * cg - ca * ca - cb * cb - cg * cg) / sg };
    const x = rot.normalize(rot.cross(b, c)); // a*
    const z = rot.normalize(c);
    const y = rot.normalize(rot.cross(z, x));
    // Rows are the target's orthonormal axes expressed in the master's coordinates.
    return rot.fromMat(.{ x[0], x[1], x[2], y[0], y[1], y[2], z[0], z[1], z[2] });
}

pub fn orientation(q: rot.Quat, crystal_rotation: rot.Quat) rot.Quat {
    // q maps sample to master; changing only its crystal coordinates multiplies on the left.
    return rot.qmul(crystal_rotation, q);
}

test "native pyMTEX crystalFrame reference transforms" {
    const Case = struct { cell: [6]f64, laue: sym.Laue, format: Format, expected: rot.Mat3 };
    // pyMTEX 1.0.0 upstream 3323a312: orientation.id(sourceCS).transformReferenceFrame(targetCS)
    // source X||a,Z||c*; target EDAX (ANG), X||a*,Z||c (CTF); inverse.matrix gives these rows.
    const cases = [_]Case{
        .{ .cell = .{ 3.47514587, 3.47514587, 3.47514587, 90, 90, 90 }, .laue = .o, .format = .ctf, .expected = .{ 1, 0, 0, 0, 1, 0, 0, 0, 1 } },
        .{ .cell = .{ 2.95111, 2.95111, 4.68433, 90, 90, 120 }, .laue = .d6, .format = .ang, .expected = .{ 1, 0, 0, 0, 1, 0, 0, 0, 1 } },
        .{ .cell = .{ 2.95111, 2.95111, 4.68433, 90, 90, 120 }, .laue = .d6, .format = .ctf, .expected = .{ 0.8660254037844388, 0.5, 0, -0.5, 0.8660254037844388, 0, 0, 0, 1 } },
        .{ .cell = .{ 4, 5, 6, 90, 104, 90 }, .laue = .c2, .format = .ang, .expected = .{ 0.9702957262759968, 0, 0.24192189559966776, 0, 1, 0, -0.24192189559966776, 0, 0.9702957262759968 } },
        .{ .cell = .{ 4, 5, 6, 73, 88, 102 }, .laue = .c1, .format = .ang, .expected = .{ 1, 0, 0, 0, 1, 0, 0, 0, 1 } },
        .{ .cell = .{ 4, 5, 6, 73, 88, 102 }, .laue = .c1, .format = .ctf, .expected = .{ 0.9730163953108699, 0.20682101943583758, -0.10229447871591912, -0.22808138807485556, 0.9291889774533888, -0.29083796965487907, 0.034899496702501066, 0.3063215795510432, 0.9512881346002722 } },
        .{ .cell = .{ 3, 3, 5, 90, 90, 120 }, .laue = .d3, .format = .ctf, .expected = .{ 0.8660254037844388, 0.5, 0, -0.5, 0.8660254037844388, 0, 0, 0, 1 } },
    };
    for (cases) |case| {
        const basis = try latticeRotation(case.cell, case.laue, case.format);
        const got = rot.toMat(basis);
        for (got, case.expected) |a, b| try std.testing.expectApproxEqAbs(b, a, 1e-12);
        const q = rot.fromBunge(.{ 0.37, 0.83, 1.21 });
        const target = orientation(q, basis);
        const sample: rot.Vec3 = .{ 0.31, -0.27, 0.91 };
        const expected = rot.matVec(case.expected, rot.qapply(q, sample));
        const actual = rot.qapply(target, sample);
        for (actual, expected) |a, b| try std.testing.expectApproxEqAbs(b, a, 1e-12);
    }
}

test "export refuses absent invalid and unresolved crystal metrics" {
    var m: master.Master = .{ .gpa = std.testing.allocator, .name = &.{}, .side = 0, .raw = &.{} };
    try std.testing.expectError(error.MissingPhaseLattice, rotation(&m, .ang));
    m.invalid_lattice = true;
    try std.testing.expectError(error.InvalidPhaseLattice, rotation(&m, .ctf));
    m.crystal_basis = .emsoft_rhombohedral;
    try std.testing.expectError(error.UnsupportedCrystalBasis, rotation(&m, .ang));
    try std.testing.expectError(error.UnsupportedCrystalBasis, latticeRotation(.{ 3, 3, 3, 80, 80, 80 }, .d3, .ctf));
}
