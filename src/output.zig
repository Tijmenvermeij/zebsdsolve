//! What leaves the engine besides numbers: maps as images, and the results as the text files
//! other EBSD software reads.
const std = @import("std");
const engine = @import("engine.zig");
const rot = @import("rot.zig");
const sym = @import("sym.zig");
const phase_export = @import("phase_export.zig");

const Engine = engine.Engine;

const palette = [_][3]u8{
    .{ 0x3b, 0x82, 0xf6 }, .{ 0xf5, 0x9e, 0x0b }, .{ 0x10, 0xb9, 0x81 }, .{ 0xef, 0x44, 0x44 },
    .{ 0x8b, 0x5c, 0xf6 }, .{ 0x14, 0xb8, 0xa6 }, .{ 0xec, 0x48, 0x99 }, .{ 0x84, 0xcc, 0x16 },
};

/// A map as RGBA8 (rows × cols, row-major): kind 0, 1, 2 the inverse pole figure colour of
/// sample x, y, z; 3 NCC, 4 NMI and 6 the NMI's distance above chance as gray between the
/// indexed cells' extremes; 5 phase; 7 the sweep that accepted the cell (dark gray: none did).
/// Cells not indexed are transparent black.
pub fn map(e: *Engine, kind: u32, out: []u8) !void {
    const n = e.res.ncc.len;
    if (out.len != 4 * n) return error.BadLength;
    if (kind > 7) return error.UnknownKind;
    var lo: f32 = std.math.inf(f32);
    var hi: f32 = -std.math.inf(f32);
    const gray_of: ?[]const f32 = switch (kind) {
        3 => e.res.ncc,
        4 => e.res.nmi,
        6 => e.res.z,
        else => null,
    };
    if (gray_of) |v| {
        for (0..n) |i| if (e.res.phase[i] >= 0) {
            lo = @min(lo, v[i]);
            hi = @max(hi, v[i]);
        };
        if (!(hi > lo)) hi = lo + 1;
    }
    // one key per phase, made when its first cell comes
    const keys = try e.gpa.alloc(?sym.Key, if (kind <= 2) e.phases.items.len else 0);
    defer e.gpa.free(keys);
    @memset(keys, null);
    for (0..n) |i| {
        const px = out[4 * i ..][0..4];
        const ph = e.res.phase[i];
        if (ph < 0) {
            px.* = .{ 0, 0, 0, 0 };
            continue;
        }
        px[3] = 255;
        switch (kind) {
            0, 1, 2 => {
                const q = e.res.quat[4 * i ..][0..4];
                var v: rot.Vec3 = .{ 0, 0, 0 };
                v[kind] = 1;
                const key = &keys[@intCast(ph)];
                if (key.* == null) key.* = e.ipfKeyOf(@intCast(ph));
                const c = key.*.?.color(rot.normalize(rot.qapply(.{ q[0], q[1], q[2], q[3] }, v)));
                for (0..3) |k| px[k] = @intFromFloat(@round(255 * std.math.clamp(c[k], 0, 1)));
            },
            3, 4, 6 => {
                const v = gray_of.?[i];
                const gray: u8 = @intFromFloat(@round(255 * std.math.clamp((v - lo) / (hi - lo), 0, 1)));
                px[0] = gray;
                px[1] = gray;
                px[2] = gray;
            },
            7 => {
                const sw = e.res.sweep[i];
                const c: [3]u8 = if (sw < 0) .{ 0x3a, 0x3a, 0x3e } else palette[@as(usize, @intCast(sw)) % palette.len];
                px[0] = c[0];
                px[1] = c[1];
                px[2] = c[2];
            },
            else => {
                const c = palette[@as(usize, @intCast(ph)) % palette.len];
                px[0] = c[0];
                px[1] = c[1];
                px[2] = c[2];
            },
        }
    }
}

/// A key to inverse pole figure colours as an image: the directions that stand for all the
/// others (key.shape), in stereographic projection with the crystal's x to the right, as RGBA8
/// size × size, transparent outside them. `lower` asks for their part in the lower hemisphere
/// (projected from the north pole, the same x and y: seen through the sphere from above), which
/// only the keys of a single rotation axis taken without the inversion have; otherwise the
/// upper one, [001] at the origin. `box` receives the part's bounding box in the projection
/// (x0, y0, x1, y1; all zero where there is none) and, for a wedge about the c axis, its two
/// azimuths in degrees (NaN for the cubic classes' polygons); the part is drawn from the image's
/// lower left corner at (size − 1) / the box's longer side pixels per unit.
pub fn ipfKey(key: *const sym.Key, lower: bool, size: u32, out: []u8, box: *[6]f32) !void {
    if (size < 2 or out.len != 4 * @as(usize, size) * size) return error.BadLength;
    const b = key.shape.box(lower);
    for (0..4) |i| box[i] = @floatCast(b[i]);
    const wedge = key.shape.kind == .wedge;
    box[4] = if (wedge) @floatCast(std.math.radiansToDegrees(key.shape.a0)) else std.math.nan(f32);
    box[5] = if (wedge) @floatCast(std.math.radiansToDegrees(key.shape.a1)) else std.math.nan(f32);
    @memset(out, 0);
    const side = @max(b[2] - b[0], b[3] - b[1]);
    if (!(side > 0)) return;
    const scale = @as(f64, @floatFromInt(size - 1)) / side;
    for (0..size) |py| for (0..size) |px| {
        const o = out[4 * (py * size + px) ..][0..4];
        const x = b[0] + @as(f64, @floatFromInt(px)) / scale;
        const y = b[1] + @as(f64, @floatFromInt(size - 1 - py)) / scale;
        const r2 = x * x + y * y;
        if (r2 > 1) continue;
        // the direction this point of the projection stands for
        const z = (1 - r2) / (1 + r2);
        const d: rot.Vec3 = .{ 2 * x / (1 + r2), 2 * y / (1 + r2), if (lower) -z else z };
        if (!key.shape.contains(d)) continue;
        const c = key.color(d);
        for (0..3) |k| o[k] = @intFromFloat(@round(255 * std.math.clamp(c[k], 0, 1)));
        o[3] = 255;
    };
}

/// TSL symmetry code of a Laue class (the .ang header's `# Symmetry`).
fn tslSymmetry(laue: sym.Laue) u32 {
    return switch (laue) {
        .c1 => 1,
        .c2 => 2,
        .d2 => 22,
        .c4 => 4,
        .d4 => 42,
        .c3 => 3,
        .d3 => 32,
        .c6 => 6,
        .d6 => 62,
        .t => 23,
        .o => 43,
    };
}

/// EDAX distinguishes settings that the master's numeric PG alone cannot disambiguate.
/// Keep those groups in Info/Symmetry rather than assert an unverified PointGroupID setting.
fn edaxPointGroupId(pg: u32) ?u32 {
    if (pg == 0 or pg > 32) return null;
    return switch (pg) {
        14, 18, 19, 20, 26 => null,
        else => 99 + pg,
    };
}

/// The results as text (caller frees): kind 0 EDAX/TSL .ang, 1 HKL .ctf. Orientations are
/// Bunge angles in the engine's sample frame (x down the tilted surface toward the detector,
/// z the surface normal), those of the stored quaternions (each in its phase's fundamental
/// zone), with the crystal basis converted to each format's native convention. The sample
/// frame and specimen tilt are unchanged; cells not indexed carry the formats' own markers.
pub fn exportText(e: *Engine, kind: u32) ![]u8 {
    const format: phase_export.Format = switch (kind) {
        0 => .ang,
        1 => .ctf,
        else => return error.UnknownKind,
    };
    // Validate every phase before writing any result; missing cells never become unit cubes.
    const crystal_rotations = try e.gpa.alloc(rot.Quat, e.phases.items.len);
    defer e.gpa.free(crystal_rotations);
    for (e.phases.items, crystal_rotations) |*p, *q| q.* = try phase_export.rotation(&p.master, format);
    const i = try e.info();
    const gpa = e.gpa;
    var out: std.Io.Writer.Allocating = .init(gpa);
    errdefer out.deinit();
    const w = &out.writer;
    const g = &e.geo;
    const n_ph = e.phases.items.len;
    const ratio = @as(f64, @floatFromInt(i.pat_w)) / @as(f64, @floatFromInt(i.pat_h));
    switch (kind) {
        0 => {
            // EDAX measures the pattern centre's y from the bottom and y, z in fractions of the width
            try w.print("# TEM_PIX_PER_UM          1.000000\n# x-star                 {d:.6}\n# y-star                 {d:.6}\n# z-star                 {d:.6}\n# WorkingDistance        0.000000\n#\n", .{ g.pc[0], (1 - g.pc[1]) / ratio, g.pc[2] / ratio });
            for (e.phases.items, 0..) |*p, pi| {
                try w.print("# Phase {d}\n# MaterialName  \t{s}\n# Formula     \t{s}\n# Info \t\tSpaceGroup={d}; SourcePointGroup={d}\n# Symmetry              {d}\n# LatticeConstants      {d:.8} {d:.8} {d:.8}  {d:.8}  {d:.8}  {d:.8}\n# NumberFamilies        0\n", .{ pi + 1, p.master.name, p.master.formula, p.master.space_group, p.master.point_group, tslSymmetry(p.master.laue), p.master.lattice.?[0], p.master.lattice.?[1], p.master.lattice.?[2], p.master.lattice.?[3], p.master.lattice.?[4], p.master.lattice.?[5] });
                if (edaxPointGroupId(p.master.point_group)) |pgid| try w.print("# PointGroupID           {d}\n", .{pgid});
                try w.writeAll("#\n");
            }
            try w.print("# GRID: SqrGrid\n# XSTEP: {d:.6}\n# YSTEP: {d:.6}\n# NCOLS_ODD: {d}\n# NCOLS_EVEN: {d}\n# NROWS: {d}\n#\n# SOFTWARE: zebsdsolve {s}\n# COLUMNS: phi1 PHI phi2 x y IQ(NMI) CI(NCC) Phase SEM Fit\n", .{ g.step_x_um, g.step_y_um, i.cols, i.cols, i.rows, engine.version });
            for (0..i.rows) |r| for (0..i.cols) |c| {
                const at = r * i.cols + c;
                const x = @as(f64, @floatFromInt(c)) * g.step_x_um;
                const y = @as(f64, @floatFromInt(r)) * g.step_y_um;
                const ph = e.res.phase[at];
                if (ph < 0) {
                    try w.print("  12.56637   12.56637   12.56637 {d:12.5} {d:12.5}      0.0  -1.000  0      0  0.000\n", .{ x, y });
                    continue;
                }
                const q = e.res.quat[4 * at ..][0..4];
                const b = rot.toBunge(phase_export.orientation(.{ q[0], q[1], q[2], q[3] }, crystal_rotations[@intCast(ph)]));
                const pid: u32 = @intCast(ph + 1);
                try w.print("{d:10.5} {d:10.5} {d:10.5} {d:12.5} {d:12.5} {d:8.4} {d:7.4} {d:2} {d:6} {d:6.3}\n", .{ b[0], b[1], b[2], x, y, e.res.nmi[at], e.res.ncc[at], pid, 0, 0.0 });
            };
        },
        1 => {
            try w.print("Channel Text File\nPrj\tzebsdsolve\nAuthor\t\nJobMode\tGrid\nXCells\t{d}\nYCells\t{d}\nXStep\t{d:.6}\nYStep\t{d:.6}\nAcqE1\t0\nAcqE2\t0\nAcqE3\t0\n", .{ i.cols, i.rows, g.step_x_um, g.step_y_um });
            try w.print("Euler angles refer to Sample Coordinate system (CS0)!\tMag\t0\tCoverage\t100\tDevice\t0\tKV\t{d:.1}\tTiltAngle\t{d:.2}\tTiltAxis\t0\n", .{ if (std.math.isNan(i.voltage_kv)) 0 else i.voltage_kv, g.sample_tilt_deg });
            try w.print("Phases\t{d}\n", .{n_ph});
            for (e.phases.items) |*p| {
                try w.print("{d:.8};{d:.8};{d:.8}\t{d:.8};{d:.8};{d:.8}\t{s}\t{d}\t{d}\t\t\t\n", .{ p.master.lattice.?[0], p.master.lattice.?[1], p.master.lattice.?[2], p.master.lattice.?[3], p.master.lattice.?[4], p.master.lattice.?[5], p.master.name, @intFromEnum(p.master.laue), p.master.space_group });
            }
            try w.writeAll("Phase\tX\tY\tBands\tError\tEuler1\tEuler2\tEuler3\tMAD\tBC\tBS\n");
            const deg = 180.0 / std.math.pi;
            for (0..i.rows) |r| for (0..i.cols) |c| {
                const at = r * i.cols + c;
                const x = @as(f64, @floatFromInt(c)) * g.step_x_um;
                const y = @as(f64, @floatFromInt(r)) * g.step_y_um;
                const ph = e.res.phase[at];
                if (ph < 0) {
                    try w.print("0\t{d:.4}\t{d:.4}\t0\t3\t0.0000\t0.0000\t0.0000\t0.0000\t0\t0\n", .{ x, y });
                    continue;
                }
                const q = e.res.quat[4 * at ..][0..4];
                const b = rot.toBunge(phase_export.orientation(.{ q[0], q[1], q[2], q[3] }, crystal_rotations[@intCast(ph)]));
                const bc: u32 = @intFromFloat(@round(255 * std.math.clamp(e.res.ncc[at], 0, 1)));
                const bs: u32 = @intFromFloat(@round(255 * std.math.clamp(e.res.nmi[at], 0, 1)));
                try w.print("{d}\t{d:.4}\t{d:.4}\t0\t0\t{d:.4}\t{d:.4}\t{d:.4}\t0.0000\t{d}\t{d}\n", .{ @as(u32, @intCast(ph + 1)), x, y, b[0] * deg, b[1] * deg, b[2] * deg, bc, bs });
            };
        },
        else => return error.UnknownKind,
    }
    return out.toOwnedSlice();
}

// An optional fixture destination lets the native MTEX oracle read actual exportText output,
// without a GPU, production result injection, or changes to the public ABI.
fn fixtureEnvironment(key: []const u8) !?[]u8 {
    return std.testing.environ.getAlloc(std.testing.allocator, key) catch |err| switch (err) {
        error.EnvironmentVariableMissing => null,
        else => return err,
    };
}

fn saveExportFixture(name: []const u8, format: []const u8, text: []const u8) !void {
    const destination = (try fixtureEnvironment("ZEBSDSOLVE_EXPORT_FIXTURE_DIR")) orelse return;
    defer std.testing.allocator.free(destination);
    const path = try std.fmt.allocPrint(std.testing.allocator, "{s}/{s}.{s}", .{ destination, name, format });
    defer std.testing.allocator.free(path);
    try std.Io.Dir.cwd().writeFile(std.Io.Threaded.global_single_threaded.io(), .{ .sub_path = path, .data = text });
}

test "text export real cells indexed IDs and crystal frames" {
    const gpa = std.testing.allocator;
    const cells = [_][6]f64{
        .{ 3.47514587, 3.47514587, 3.47514587, 90, 90, 90 },
        .{ 2.95111, 2.95111, 4.68433, 90, 90, 120 },
        .{ 4, 5, 6, 90, 104, 90 },
        .{ 4, 5, 6, 73, 88, 102 },
        .{ 3, 3, 5, 90, 90, 120 },
    };
    const laues = [_]sym.Laue{ .o, .d6, .c2, .c1, .d3 };
    const sgs = [_]u32{ 225, 194, 10, 2, 164 };
    const pgs = [_]u32{ 32, 27, 5, 2, 20 };
    const names = [_][]const u8{ "cubic", "hexagonal", "monoclinic", "triclinic", "trigonal" };
    var phases: [5]engine.Phase = undefined;
    for (&phases, 0..) |*p, j| p.* = .{ .master = .{
        .gpa = gpa,
        .name = @constCast(names[j]),
        .formula = @constCast(if (j == 1) "Ti" else ""),
        .side = 0,
        .raw = &.{},
        .lattice = cells[j],
        .laue = laues[j],
        .space_group = sgs[j],
        .point_group = pgs[j],
    } };
    // Only the members read by exportText are initialized. No GPU object is constructed.
    var e: Engine = undefined;
    e.gpa = gpa;
    e.geo = .{ .step_x_um = 0.25, .step_y_um = 0.5 };
    e.source = .{ .memory = .{ .data = &.{}, .info = .{ .rows = 1, .cols = 2, .pat_h = 8, .pat_w = 8, .voltage_kv = 20 } } };
    const q = rot.fromBunge(.{ 0.37, 0.83, 1.21 });
    var quats: [24]f32 = undefined;
    for (0..6) |j| for (0..4) |k| {
        quats[4 * j + k] = @floatCast(q[k]);
    };
    var ncc = [_]f32{0.8} ** 6;
    var nmi = [_]f32{0.7} ** 6;
    var ids = [_]i32{ 0, -1, 0, 0, 0, 0 };
    e.res = .{ .quat = &quats, .ncc = &ncc, .nmi = &nmi, .phase = &ids };
    for (0..5) |j| {
        e.phases = .{ .items = phases[j..][0..1], .capacity = 1 };
        for (0..2) |kind| {
            const text = try exportText(&e, @intCast(kind));
            defer gpa.free(text);
            try saveExportFixture(names[j], if (kind == 0) "ang" else "ctf", text);
            if (kind == 0) {
                try std.testing.expect(std.mem.indexOf(u8, text, "# Phase 1\n") != null);
                try std.testing.expect(std.mem.indexOf(u8, text, " 1      0  0.000\n") != null);
                try std.testing.expect(std.mem.indexOf(u8, text, "-1.000  0      0  0.000\n") != null);
                if (j == 4) {
                    try std.testing.expect(std.mem.indexOf(u8, text, "PointGroupID") == null);
                    try std.testing.expect(std.mem.indexOf(u8, text, "SpaceGroup=164; SourcePointGroup=20") != null);
                }
                if (j == 1) {
                    try std.testing.expect(std.mem.indexOf(u8, text, "2.95111000 2.95111000 4.68433000  90.00000000  90.00000000  120.00000000") != null);
                    try std.testing.expect(std.mem.indexOf(u8, text, "# Formula     \tTi\n") != null);
                }
            } else if (j == 1) try std.testing.expect(std.mem.indexOf(u8, text, "2.95111000;2.95111000;4.68433000\t90.00000000;90.00000000;120.00000000\thexagonal\t9\t194") != null);
        }
    }
    e.phases = .{ .items = &phases, .capacity = phases.len };
    e.source.memory.info.cols = 6;
    ids = .{ 0, 1, 2, 3, 4, -1 };
    for (0..2) |kind| {
        const text = try exportText(&e, @intCast(kind));
        defer gpa.free(text);
        try saveExportFixture("multiphase", if (kind == 0) "ang" else "ctf", text);
        if (kind == 0) try std.testing.expect(std.mem.indexOf(u8, text, " 5      0  0.000\n") != null);
    }
    // Optional original master fixture, for testing the complete importer-to-export boundary.
    if (try fixtureEnvironment("ZEBSDSOLVE_EXPORT_MASTER")) |path| {
        defer gpa.free(path);
        var loaded = try @import("master.zig").load(gpa, path);
        defer loaded.deinit();
        var loaded_phase: [1]engine.Phase = .{.{ .master = loaded }};
        e.phases = .{ .items = &loaded_phase, .capacity = 1 };
        e.source.memory.info.cols = 2;
        ids[0] = 0;
        ids[1] = -1;
        for (0..2) |kind| {
            const text = try exportText(&e, @intCast(kind));
            defer gpa.free(text);
            try saveExportFixture("loaded", if (kind == 0) "ang" else "ctf", text);
        }
        e.phases = .{ .items = &phases, .capacity = phases.len };
    }
    phases[4].master.lattice = null;
    try std.testing.expectError(error.MissingPhaseLattice, exportText(&e, 0));
    phases[4].master.invalid_lattice = true;
    try std.testing.expectError(error.InvalidPhaseLattice, exportText(&e, 1));
    phases[4].master.crystal_basis = .emsoft_rhombohedral;
    try std.testing.expectError(error.UnsupportedCrystalBasis, exportText(&e, 0));
}

test "text export does not claim unresolved EDAX point-group settings" {
    for ([_]u32{ 0, 14, 18, 19, 20, 26, 33 }) |pg| try std.testing.expect(edaxPointGroupId(pg) == null);
    try std.testing.expectEqual(@as(?u32, 124), edaxPointGroupId(25));
    try std.testing.expectEqual(@as(?u32, 126), edaxPointGroupId(27));
    try std.testing.expectEqual(@as(?u32, 131), edaxPointGroupId(32));
}
