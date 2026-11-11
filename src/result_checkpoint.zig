//! Same-page, versioned native result state. No rotations or source frames are converted.
//! Explicit little-endian scalars: no ABI padding, pointers, or old phase generation.
const std = @import("std");
const engine = @import("engine.zig");
const Sha = std.crypto.hash.sha2.Sha256;
const header = 112;
const magic = "ZERESULT";

fn hashValue(h: *Sha, v: anytype) void {
    const T = @TypeOf(v);
    switch (@typeInfo(T)) {
        .bool => h.update(&.{@intFromBool(v)}),
        .int => |i| {
            const U = std.meta.Int(.unsigned, i.bits);
            var b: [@sizeOf(T)]u8 = undefined;
            std.mem.writeInt(U, &b, @bitCast(v), .little);
            h.update(&b);
        },
        .float => |f| {
            const U = std.meta.Int(.unsigned, f.bits);
            // NaNs in unavailable calibration fields are a single semantic value.
            hashValue(h, @as(U, @bitCast(if (std.math.isNan(v)) std.math.nan(T) else v)));
        },
        .@"enum" => hashValue(h, @as(u32, @intCast(@intFromEnum(v)))),
        .optional => {
            hashValue(h, v != null);
            if (v) |x| hashValue(h, x);
        },
        .array => for (v) |x| hashValue(h, x),
        .@"struct" => |s| inline for (s.fields) |f| {
            if (!std.mem.eql(u8, f.name, "_pad")) hashValue(h, @field(v, f.name));
        },
        .pointer => |p| {
            if (p.size != .slice) @compileError("checkpoint hash cannot contain pointers");
            hashValue(h, @as(u64, v.len));
            for (v) |x| hashValue(h, x);
        },
        else => @compileError("unsupported checkpoint context field"),
    }
}

fn fingerprint(e: *const engine.Engine) [32]u8 {
    var h = Sha.init(.{});
    hashValue(&h, e.set);
    hashValue(&h, e.geo);
    hashValue(&h, e.calibration);
    for (e.phases.items) |p| {
        const m = p.master;
        hashValue(&h, m.name);
        hashValue(&h, m.side);
        hashValue(&h, m.lattice);
        hashValue(&h, m.point_group);
        hashValue(&h, m.space_group);
        hashValue(&h, m.laue);
        hashValue(&h, m.rotation);
        hashValue(&h, m.voltage_kv);
        hashValue(&h, m.source);
        hashValue(&h, m.crystal_frame);
        hashValue(&h, m.source_format_version);
        hashValue(&h, m.raw);
        hashValue(&h, m.ops);
    }
    return h.finalResult();
}

const Codec = struct {
    bytes: []u8,
    pos: usize = 0,
    fn put(self: *Codec, v: anytype) void {
        const T = @TypeOf(v);
        const U = std.meta.Int(.unsigned, @bitSizeOf(T));
        std.mem.writeInt(U, self.bytes[self.pos..][0..@sizeOf(T)], @bitCast(v), .little);
        self.pos += @sizeOf(T);
    }
    fn get(self: *Codec, comptime T: type) T {
        const U = std.meta.Int(.unsigned, @bitSizeOf(T));
        const v = std.mem.readInt(U, self.bytes[self.pos..][0..@sizeOf(T)], .little);
        self.pos += @sizeOf(T);
        return @bitCast(v);
    }
};

fn payloadSize(n: usize, phases: usize) !usize {
    const winners = try std.math.mul(usize, n, 37);
    const candidates = try std.math.mul(usize, try std.math.mul(usize, n, phases), 36);
    return std.math.add(usize, winners, candidates);
}

pub fn byteSize(e: *engine.Engine) !usize {
    _ = try e.info();
    const context = e.res.context orelse return error.NoIndexedResults;
    if (!e.res.context_consistent or !engine.sameExportContext(context, .{ .settings = e.set, .geometry = e.geo, .calibration = e.calibration, .phase_generation = e.phase_generation })) return error.CheckpointContextChanged;
    if (e.res.candidate_phases != e.phases.items.len or e.res.candidates.len != e.res.phase.len * e.phases.items.len) return error.CheckpointCandidatesMissing;
    return std.math.add(usize, header, try payloadSize(e.res.phase.len, e.phases.items.len));
}

pub fn write(e: *engine.Engine, destination: []u8) !void {
    const total = try byteSize(e);
    if (destination.len < total) return error.BufferTooSmall;
    const bytes = destination[0..total];
    const info = try e.info();
    const size = total - header;
    @memset(bytes[0..header], 0);
    @memcpy(bytes[0..8], magic);
    var c: Codec = .{ .bytes = bytes, .pos = 8 };
    c.put(@as(u32, 1));
    c.put(@as(u32, header));
    c.put(info.rows);
    c.put(info.cols);
    c.put(info.pat_h);
    c.put(info.pat_w);
    c.put(@as(u32, @intCast(e.phases.items.len)));
    c.put(@as(u32, 0));
    c.put(@as(u64, size));
    @memcpy(bytes[48..80], &fingerprint(e));
    c.pos = header;
    for (e.res.phase, 0..) |phase, i| {
        for (e.res.quat[4 * i ..][0..4]) |q| c.put(q);
        c.put(e.res.ncc[i]);
        c.put(e.res.nmi[i]);
        c.put(e.res.z[i]);
        c.put(phase);
        c.put(e.res.sweep[i]);
        c.put(@as(u8, @intFromBool(e.res.settled[i])));
    }
    for (e.res.candidate_sweep, 0..) |sweep, i| {
        // Missing candidates have uninitialized native Record storage. Never read it.
        const r: engine.Record = if (sweep < 0) std.mem.zeroes(engine.Record) else e.res.candidates[i];
        for (r.q) |q| c.put(q);
        c.put(r.ncc);
        c.put(r.nmi);
        c.put(r.step);
        c.put(r.z);
        c.put(sweep);
    }
    Sha.hash(bytes[header..], bytes[80..112], .{});
}

pub fn restore(e: *engine.Engine, input: []const u8) !void {
    if (input.len < header or !std.mem.eql(u8, input[0..8], magic)) return error.InvalidResultCheckpoint;
    // Codec only reads; the ABI input remains const.
    var c: Codec = .{ .bytes = @constCast(input), .pos = 8 };
    if (c.get(u32) != 1 or c.get(u32) != header) return error.InvalidResultCheckpoint;
    const info = try e.info();
    if (c.get(u32) != info.rows or c.get(u32) != info.cols or c.get(u32) != info.pat_h or c.get(u32) != info.pat_w) return error.CheckpointShapeMismatch;
    const phases = c.get(u32);
    if (phases == 0 or phases != e.phases.items.len or c.get(u32) != 0) return error.CheckpointPhaseMismatch;
    const n = try std.math.mul(usize, info.rows, info.cols);
    const size = try payloadSize(n, phases);
    if (c.get(u64) != size or input.len != try std.math.add(usize, header, size)) return error.InvalidResultCheckpoint;
    if (!std.mem.eql(u8, input[48..80], &fingerprint(e))) return error.CheckpointContextMismatch;
    var checksum: [32]u8 = undefined;
    Sha.hash(input[header..], &checksum, .{});
    if (!std.mem.eql(u8, input[80..112], &checksum)) return error.InvalidResultCheckpoint;
    var replacement: engine.Results = .{};
    errdefer replacement.free(e.gpa);
    try replacement.resize(e.gpa, n);
    try replacement.prepareCandidates(e.gpa, phases);
    c.pos = header;
    for (0..n) |i| {
        for (replacement.quat[4 * i ..][0..4]) |*q| q.* = c.get(f32);
        replacement.ncc[i] = c.get(f32);
        replacement.nmi[i] = c.get(f32);
        replacement.z[i] = c.get(f32);
        const phase = c.get(i32);
        const sweep = c.get(i32);
        const settled = c.get(u8);
        if (phase < -1 or phase >= phases or sweep < -1 or (sweep >= 0 and sweep >= e.set.sweeps) or settled > 1) return error.InvalidResultCheckpoint;
        replacement.phase[i] = phase;
        replacement.sweep[i] = sweep;
        replacement.settled[i] = settled == 1;
    }
    for (replacement.candidates, replacement.candidate_sweep) |*r, *sweep| {
        for (&r.q) |*q| q.* = c.get(f32);
        r.ncc = c.get(f32);
        r.nmi = c.get(f32);
        r.step = c.get(f32);
        r.z = c.get(f32);
        sweep.* = c.get(i32);
        if (sweep.* < -1 or (sweep.* >= 0 and sweep.* >= e.set.sweeps)) return error.InvalidResultCheckpoint;
    }
    replacement.context = .{ .settings = e.set, .geometry = e.geo, .calibration = e.calibration, .phase_generation = e.phase_generation };
    e.res.free(e.gpa);
    e.res = replacement;
}
