//! ANG scalar/grid/producer-specific PC sidecar. Euler angles, phases and crystal metadata are not imported.
//! EDAX positions and XSTEP/YSTEP are micrometres by default (native pyMTEX ANG convention).
//! This streaming adapter proves row-major registration before joining IQ to UP pixel IDs.
const std = @import("std");
const File = @import("io/file.zig").File;
const Allocator = std.mem.Allocator;

pub const Metadata = struct {
    rows: u32,
    cols: u32,
    step_x_um: f64,
    step_y_um: f64,
    iq: []f32,
    pc: ?[3]f64 = null,
    pc_note: []const u8 = "",
};

const Lines = struct {
    f: *File,
    buf: [65536]u8 = undefined,
    pos: usize = 0,
    len: usize = 0,
    offset: u64 = 0,
    line: std.ArrayList(u8) = .empty,
    fn next(self: *Lines, gpa: Allocator) !?[]const u8 {
        self.line.clearRetainingCapacity();
        while (true) {
            if (self.pos == self.len) {
                if (self.offset == self.f.size) return if (self.line.items.len > 0) self.line.items else null;
                self.len = @intCast(@min(self.buf.len, self.f.size - self.offset));
                try self.f.read(self.offset, self.buf[0..self.len]);
                self.offset += self.len;
                self.pos = 0;
            }
            const remaining = self.buf[self.pos..self.len];
            const end = std.mem.indexOfScalar(u8, remaining, '\n') orelse remaining.len;
            if (self.line.items.len + end > 1 << 20) return error.ANGLineTooLong;
            try self.line.appendSlice(gpa, remaining[0..end]);
            self.pos += end;
            if (self.pos < self.len) {
                self.pos += 1;
                return self.line.items;
            }
        }
    }
};

const Columns = struct {
    text: []const u8,
    at: usize = 0,
    comma: bool,
    fn init(text: []const u8) Columns {
        return .{ .text = text, .comma = std.mem.indexOfScalar(u8, text, ',') != null };
    }
    fn next(self: *Columns) ?[]const u8 {
        if (self.comma) {
            if (self.at > self.text.len) return null;
            const start = self.at;
            const end = if (std.mem.indexOfScalar(u8, self.text[start..], ',')) |k| start + k else self.text.len;
            self.at = end + 1;
            return std.mem.trim(u8, self.text[start..end], " \t\"'");
        }
        while (self.at < self.text.len and (self.text[self.at] == ' ' or self.text[self.at] == '\t' or self.text[self.at] == ';')) self.at += 1;
        if (self.at == self.text.len) return null;
        const start = self.at;
        while (self.at < self.text.len and self.text[self.at] != ' ' and self.text[self.at] != '\t' and self.text[self.at] != ';') self.at += 1;
        return self.text[start..self.at];
    }
};

fn eq(a: []const u8, b: []const u8) bool {
    return std.ascii.eqlIgnoreCase(a, b);
}
fn micro(value: []const u8) bool {
    const v = std.mem.trim(u8, value, " \t[]()");
    for ([_][]const u8{ "um", "µm", "μm", "micron", "microns", "micrometer", "micrometers", "micrometre", "micrometres" }) |unit| if (eq(v, unit)) return true;
    return false;
}
fn positive(v: []const u8) !f64 {
    const x = std.fmt.parseFloat(f64, v) catch return error.InvalidANGStep;
    if (!std.math.isFinite(x) or x <= 0) return error.InvalidANGStep;
    return x;
}
fn dimension(v: []const u8) !u32 {
    const n = std.fmt.parseInt(u32, v, 10) catch return error.InvalidANGDimensions;
    if (n == 0) return error.InvalidANGDimensions;
    return n;
}
fn checkCoordinate(actual: f64, expected: f64, origin: f64, travel: f64, step: f64) !void {
    if (!std.math.isFinite(actual) or !std.math.isFinite(expected)) return error.ANGCoordinatesNotRowMajor;
    const residual = @abs(actual - expected);
    const strict = @max(step * 1e-6, @min(5e-6, step * 0.01));
    if (residual <= strict) return; // also accepts well-resolved f64 coordinates at large origins
    // EDAX may compute positions in float32 and print five decimal places. Account
    // for origin, point, step/multiply rounding, plus two half decimal quanta.
    // This per-cell budget never grows a decimal allowance with the pixel index.
    const budget = @min(1e-5, step * 0.01) + 0x1p-24 * (@abs(origin) + @abs(expected) + 2 * @abs(travel));
    if (!std.math.isFinite(budget) or budget >= step / 8) return error.ANGCoordinatePrecisionInsufficient;
    if (residual > budget) return error.ANGCoordinatesNotRowMajor;
}

const PcDecision = struct { value: ?[3]f64 = null, note: []const u8 = "" };
const LegacyPhase = struct {
    material_seen: bool = false,
    blank_material: bool = false,
    zero_families_before_lattice: bool = false,
    lattice: bool = false,
    elastic: usize = 0,
    categories: bool = false,
    fn matches(self: LegacyPhase) bool {
        return self.material_seen and self.blank_material and self.zero_families_before_lattice and self.lattice and self.elastic == 6 and self.categories;
    }
};
const PcHeader = struct {
    raw: [3]?f64 = .{ null, null, null },
    width_producer: bool = false,
    software_seen: bool = false,
    edax_producer: bool = false,
    header_start: bool = false,
    numeric_version: bool = false,
    version_seen: bool = false,
    phases: usize = 0,
    previous_legacy_phases: bool = true,
    phase: LegacyPhase = .{},
    operator_unknown: bool = false,
    sample_empty: bool = false,
    scan_empty: bool = false,

    fn header(self: *PcHeader, key: []const u8, value: []const u8) !void {
        const index: ?usize = if (eq(key, "x-star")) 0 else if (eq(key, "y-star")) 1 else if (eq(key, "z-star")) 2 else null;
        if (index) |k| {
            if (self.raw[k] != null) return error.DuplicateANGPatternCentre;
            const x = std.fmt.parseFloat(f64, value) catch return error.InvalidANGPatternCentre;
            if (!std.math.isFinite(x)) return error.InvalidANGPatternCentre;
            self.raw[k] = x;
        } else if (eq(key, "SOFTWARE")) {
            self.software_seen = true;
            var words = std.mem.tokenizeAny(u8, value, " \t");
            const producer = words.next() orelse return;
            if (eq(producer, "zebsdsolve") or eq(producer, "EMSphInx")) self.width_producer = true;
            if (eq(producer, "EDAX") or eq(producer, "OIM")) self.edax_producer = true;
        } else if (eq(key, "HEADER") and eq(value, "Start")) {
            self.header_start = true;
        } else if (eq(key, "VERSION")) {
            self.version_seen = true;
            if (std.ascii.indexOfIgnoreCase(value, "Overlap-GUI positive phase IDs") == 0) self.width_producer = true;
            _ = std.fmt.parseInt(u32, value, 10) catch return;
            self.numeric_version = true;
        } else if (eq(key, "Phase")) {
            if (self.phases != 0) self.previous_legacy_phases = self.previous_legacy_phases and self.phase.matches();
            self.phases += 1;
            self.phase = .{};
        } else if (self.phases != 0 and eq(key, "MaterialName")) {
            self.phase.material_seen = true;
            self.phase.blank_material = value.len == 0;
        } else if (self.phases != 0 and eq(key, "NumberFamilies")) {
            self.phase.zero_families_before_lattice = !self.phase.lattice and eq(value, "0");
        } else if (self.phases != 0 and eq(key, "LatticeConstants")) {
            self.phase.lattice = true;
        } else if (self.phases != 0 and eq(key, "ElasticConstants")) {
            self.phase.elastic += 1;
        } else if (self.phases != 0 and eq(key, "Categories")) {
            self.phase.categories = true;
        } else if (eq(key, "OPERATOR")) {
            self.operator_unknown = eq(value, "unknown");
        } else if (eq(key, "SAMPLEID")) {
            self.sample_empty = value.len == 0;
        } else if (eq(key, "SCANID")) {
            self.scan_empty = value.len == 0;
        }
    }

    fn resolve(self: PcHeader, width: u32, height: u32) !PcDecision {
        var present: usize = 0;
        for (self.raw) |v| present += @intFromBool(v != null);
        if (present == 0) return .{};
        if (present != 3) return error.IncompleteANGPatternCentre;
        const raw: [3]f64 = .{ self.raw[0].?, self.raw[1].?, self.raw[2].? };
        if (raw[0] == 0 and raw[1] == 0 and raw[2] == 0) return .{}; // producer's uncalibrated sentinel
        if (raw[2] <= 0 or width == 0 or height == 0) return error.InvalidANGPatternCentre;
        // The classic EMSphInx writer's phase-block ordering and trailer distinguish
        // it from an unknown ANG; blank material/phase IDs alone are not a convention.
        const legacy = self.phases > 0 and self.previous_legacy_phases and self.phase.matches() and !self.version_seen and !self.software_seen and self.operator_unknown and self.sample_empty and self.scan_empty;
        const standard_edax = self.header_start and self.numeric_version and !self.software_seen;
        // Explicit recognized writers outrank inherited structural headers. Conflicting
        // explicit writers are ambiguous only when the detector normalizations differ.
        const conflict = self.width_producer and self.edax_producer;
        if (conflict and width != height) return .{ .note = "ANG pattern centre was not applied: explicit producer markers conflict." };
        const explicit = self.width_producer or self.edax_producer;
        const width_scaled = self.width_producer or (!explicit and legacy and !standard_edax);
        const edax = self.edax_producer or (!explicit and standard_edax and !legacy);
        if (!width_scaled and !edax and width != height) return .{ .note = "ANG pattern centre was not applied: producer convention is unknown for a rectangular detector." };
        const w: f64 = @floatFromInt(width);
        const h: f64 = @floatFromInt(height);
        // Existing native detector conventions, not crystallographic rotations:
        // EDAX/OIM: x/width, y/height from bottom, distance/min(width,height).
        // EMSphInx/zebsdsolve/Overlap-GUI: all three normalized by width.
        const pc: [3]f64 = if (width_scaled) .{ raw[0], 1 - raw[1] * w / h, raw[2] * w / h } else .{ raw[0], 1 - raw[1], raw[2] * @min(w, h) / h };
        for (pc) |x| if (!std.math.isFinite(x)) return error.InvalidANGPatternCentre;
        if (!(pc[2] > 0)) return error.InvalidANGPatternCentre;
        return .{ .value = pc, .note = if (conflict) "Imported ANG pattern centre: square detector; conflicting producer normalizations coincide." else if (width_scaled) "Imported ANG pattern centre: EMSphInx/zebsdsolve width-normalized convention." else if (edax) "Imported ANG pattern centre: EDAX/OIM convention." else "Imported ANG pattern centre: square detector; ANG normalizations coincide." };
    }
};

pub fn load(gpa: Allocator, file: *File, expected_count: u64, pattern_width: u32, pattern_height: u32) !Metadata {
    var lines: Lines = .{ .f = file };
    defer lines.line.deinit(gpa);
    var rows: u32 = 0;
    var odd: u32 = 0;
    var even: u32 = 0;
    var dx: f64 = 0;
    var dy: f64 = 0;
    var square = false;
    var pc_header: PcHeader = .{};
    var xcol: usize = 3;
    var ycol: usize = 4;
    var iqcol: usize = 5;
    var names_seen = false;
    var units: ?[]u8 = null;
    defer if (units) |v| gpa.free(v);
    var iq: []f32 = &.{};
    errdefer gpa.free(iq);
    var count: usize = 0;
    var x0: f64 = 0;
    var y0: f64 = 0;
    while (try lines.next(gpa)) |raw| {
        const line = std.mem.trim(u8, raw, " \t\r");
        if (line.len == 0) continue;
        if (line[0] == '#') {
            if (count != 0) continue;
            const content = std.mem.trim(u8, line[1..], " \t");
            if (std.ascii.indexOfIgnoreCase(content, "ACOM") != null or std.ascii.indexOfIgnoreCase(content, "TEM data") != null) return error.UnsupportedANGConvention;
            var pair = std.mem.tokenizeAny(u8, content, ": \t");
            const key = pair.next() orelse continue;
            const value = std.mem.trim(u8, pair.rest(), ": \t");
            try pc_header.header(key, value);
            if (eq(key, "GRID")) {
                if (!eq(value, "SqrGrid")) return error.UnsupportedANGGrid;
                square = true;
            } else if (eq(key, "NROWS")) {
                rows = try dimension(value);
            } else if (eq(key, "NCOLS_ODD")) {
                odd = try dimension(value);
            } else if (eq(key, "NCOLS_EVEN")) {
                even = try dimension(value);
            } else if (eq(key, "XSTEP")) {
                dx = try positive(value);
            } else if (eq(key, "YSTEP")) {
                dy = try positive(value);
            } else if (eq(key, "COLUMN_HEADERS")) {
                if (names_seen) return error.DuplicateANGColumns;
                names_seen = true;
                xcol = std.math.maxInt(usize);
                ycol = xcol;
                iqcol = xcol;
                var names = Columns.init(value);
                var k: usize = 0;
                while (names.next()) |name| : (k += 1) {
                    const out = if (eq(name, "x")) &xcol else if (eq(name, "y")) &ycol else if (eq(name, "IQ") or eq(name, "ImageQuality") or eq(name, "Image Quality") or eq(name, "image_quality")) &iqcol else continue;
                    if (out.* != std.math.maxInt(usize)) return error.DuplicateANGColumns;
                    out.* = k;
                }
                if (xcol == std.math.maxInt(usize) or ycol == std.math.maxInt(usize) or iqcol == std.math.maxInt(usize)) return error.UnsupportedANGColumns;
            } else if (eq(key, "COLUMN_UNITS")) {
                if (units != null) return error.DuplicateANGUnits;
                units = try gpa.dupe(u8, value);
            } else if (eq(key, "SCAN_UNIT") or eq(key, "SCANUNIT") or eq(key, "XUNITS") or eq(key, "YUNITS")) {
                if (!micro(value)) return error.UnsupportedANGUnits;
            }
            continue;
        }
        if (count == 0) {
            if (!square or rows == 0 or odd == 0 or even == 0) return error.ANGGridMetadataRequired;
            if (odd != even) return error.UnsupportedANGGrid;
            if (!(dx > 0) or !(dy > 0)) return error.InvalidANGStep;
            if (@as(u64, rows) * odd != expected_count or expected_count > std.math.maxInt(usize)) return error.ANGPatternCountMismatch;
            if (units) |u| {
                var values = Columns.init(u);
                var k: usize = 0;
                var found: usize = 0;
                while (values.next()) |v| : (k += 1) {
                    if (k == xcol or k == ycol) {
                        if (!micro(v)) return error.UnsupportedANGUnits;
                        found += 1;
                    }
                }
                if (found != 2) return error.UnsupportedANGUnits;
            }
            iq = try gpa.alloc(f32, @intCast(expected_count));
        }
        if (count >= iq.len) return error.ANGPatternCountMismatch;
        var values = std.mem.tokenizeAny(u8, line, " \t");
        var k: usize = 0;
        var x: f64 = std.math.nan(f64);
        var y = x;
        var quality = x;
        while (values.next()) |v| : (k += 1) {
            if (k == xcol or k == ycol or k == iqcol) {
                const n = std.fmt.parseFloat(f64, v) catch return error.InvalidANGRecord;
                if (k == xcol) x = n else if (k == ycol) y = n else quality = n;
            }
        }
        if (k < 8 or k <= @max(xcol, @max(ycol, iqcol))) return error.InvalidANGRecord;
        if (count == 0) {
            x0 = x;
            y0 = y;
        }
        const xt = @as(f64, @floatFromInt(count % odd)) * dx;
        const yt = @as(f64, @floatFromInt(count / odd)) * dy;
        try checkCoordinate(x, x0 + xt, x0, xt, dx);
        try checkCoordinate(y, y0 + yt, y0, yt, dy);
        iq[count] = if (std.math.isFinite(quality) and @abs(quality) <= std.math.floatMax(f32)) @floatCast(quality) else std.math.nan(f32);
        count += 1;
    }
    if (count != expected_count or count == 0) return error.ANGPatternCountMismatch;
    const pc = try pc_header.resolve(pattern_width, pattern_height);
    return .{ .rows = rows, .cols = odd, .step_x_um = dx, .step_y_um = dy, .iq = iq, .pc = pc.value, .pc_note = pc.note };
}
