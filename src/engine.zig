//! The indexing engine both front ends drive. Operations are written as ordinary blocking
//! code: record the GPU schedule, queue the reads of its results, `wait`, compute the result.
//! Natively `wait` polls the device and a file read is a system call; in the browser the module
//! suspends at both (JS Promise Integration) while the adapter awaits the GPU or a file slice.
//! Same code, same schedule.
//!
//! State: a scan (a file read in ranges, or patterns handed over in memory), candidate phases
//! (master patterns), the detector geometry, the settings, and per map cell the result.
//!
//! Patterns exist at two resolutions (`Level`): reduced by settings.binning on the host as
//! they are read, and as stored. Each stage of indexing (the search, the refinement, the final
//! scores) works at the level settings.full_resolution gives it; everything a level needs (the
//! filtered patterns, the static background, each phase's filtered master) is kept per level.
//!
//! Indexing keeps the file and the device busy at once: two batches are in flight, each recorded
//! and submitted while the device works on the one before (`indexBatches`), and meanwhile the
//! host reads the batch after's patterns (into a cache by map cell) and reduces them if a level
//! is reduced. With settings.nlpar every pattern is first averaged with its
//! neighbours in the map, on the device, in `uploadCells`: a batch is then read with a margin
//! of cells around it, batches are square blocks of the map, and the cache holds what
//! neighbouring blocks share.
const std = @import("std");
const gpu_mod = @import("gpu/gpu.zig");
const rot = @import("rot.zig");
const sym = @import("sym.zig");
const sh = @import("sh.zig");
const scan_mod = @import("scan.zig");
const master_mod = @import("master.zig");
const geometry_mod = @import("geometry.zig");
const settings_mod = @import("settings.zig");
const events_mod = @import("events.zig");
const prep_mod = @import("prep.zig");
const reduce = @import("reduce.zig");
const spherical = @import("spherical.zig");
const detector = @import("detector.zig");
const pca_mod = @import("pca.zig");
const nlpar_mod = @import("nlpar.zig");
const crystal_mod = @import("sim/crystal.zig");
const simulate_mod = @import("simulate.zig");
pub const overlap = @import("overlap.zig");
pub const overlap_peaks = @import("overlap_peaks.zig");
pub const overlap_nuisance = @import("overlap_nuisance.zig");
pub const overlap_profile = @import("overlap_profile.zig");
const convention_mod = @import("convention.zig");

const Gpu = gpu_mod.Gpu;
const Buf = gpu_mod.Buf;
const Allocator = std.mem.Allocator;
pub const Settings = settings_mod.Settings;
pub const Geometry = geometry_mod.Geometry;

pub const version = "0.2.8";

/// A pattern resolution: reduced by settings.binning, or as stored.
pub const Level = enum(u1) { reduced = 0, full = 1 };
pub const Stage = enum { search, refine, score };

/// What the scan is, for the front ends (`ze_scan_info`).
pub const ScanInfo = extern struct {
    rows: u32 = 0,
    cols: u32 = 0,
    pat_h: u32 = 0,
    pat_w: u32 = 0,
    /// 0: u8, 1: u16, 2: f32
    pixel: u32 = 0,
    has_background: u32 = 0,
    /// 1 when the file gave a pattern centre
    has_pc: u32 = 0,
    /// index into scan.zig's Format, or 100 for patterns in memory
    format: u32 = 0,
    voltage_kv: f64 = 0,
    bytes: f64 = 0,
    /// 1 when the file gave a pixel size
    has_px: u32 = 0,
    /// the kinds of patterns the file stores side by side (scan.zig's KIND_*: 1 processed,
    /// 2 unprocessed, 3 both; 0: one kind it does not name), and (4) when those read are the
    /// unprocessed
    pattern_kinds: u32 = 0,
    /// Source-derived geometry convention: 0 generic, 1 legacy Oxford (reserved),
    /// 2 Oxford detector CS3-to-microscope v2, 3 Oxford fallback.
    geometry_frame: u32 = 0,
};

pub const PhaseInfo = extern struct {
    side: u32 = 0,
    point_group: u32 = 0,
    space_group: u32 = 0,
    laue: u32 = 0,
    /// harmonic selection rules at the bandwidth last used: rotation order about z, mirror, inversion
    n_fold: u32 = 0,
    mirror: u32 = 0,
    inversion: u32 = 0,
    pad: u32 = 0,
    voltage_kv: f64 = 0,
    name: [64]u8 = @splat(0),
};

/// Where indexing stands (`ze_index_status`).
pub const IndexStatus = extern struct {
    /// the sweep in progress (0-based) of `sweeps`, and what it searches with
    sweep: u32 = 0,
    sweeps: u32 = 1,
    bandwidth: u32 = 0,
    dict_grid: u32 = 0,
    /// patterns of this sweep done, and in it
    done: u64 = 0,
    total: u64 = 0,
    /// cells of the region, and how many have an accepted match so far
    region: u64 = 0,
    accepted: u64 = 0,
    /// the sweep's method (settings.Method's order)
    method: u32 = 0,
    pad: u32 = 0,
};

/// What a phase's dictionary components were learned for.
const PcaKey = struct {
    grid: u32,
    k: u32,
    epochs: u32,
    level: Level,
    serial: u32,
    h: u32,
    w: u32,
    mask: [3]f32,
    t: [3]f32,
    c: [9]f32,
};

/// What a phase's master was filtered for at one level.
const BakeKey = struct {
    /// a pattern pixel of that level on the master's sheets, in sheet pixels, as a step of a
    /// 5 % ladder (so a pattern centre that moves a little does not rebuild everything);
    /// i32 minimum: unknown (no scan), the master stays as loaded
    px_q: i32,
    /// in pattern pixels of that level
    smooth: f64,
    bandpass: bool,
    fine: f64,
    coarse: f64,
    clahe: bool,
    tiles: u32,
    bins: u32,
    clip: f64,
};

const RdenKey = struct {
    c: [9]f32,
    t: [3]f32,
    h: u32,
    w: u32,
    mask: [3]f32,
};

const BgKey = struct {
    kind: settings_mod.Static,
    n: u32,
    bin: f64,
};

/// A dictionary of one grid, with what was learned of it.
const Dictionary = struct {
    dict: []f32 = &.{},
    dict_gpu: Buf = .{},
    dict_grid: u32 = 0,
    pca_w: Buf = .{},
    pca_coef: Buf = .{},
    pca_key: ?PcaKey = null,

    fn release(self: *Dictionary, g: *Gpu) void {
        g.gpa.free(self.dict);
        g.release(&self.dict_gpu);
        g.release(&self.pca_w);
        g.release(&self.pca_coef);
        self.* = .{};
    }
};

pub const Phase = struct {
    master: master_mod.Master,
    /// the filtered master per level: on the device, on the host, what it was filtered for,
    /// and a count of its rebuilds (what was derived from it knows by that when it is stale)
    det: [2]detector.MasterGpu = .{ .{}, .{} },
    work: [2][]f32 = .{ &.{}, &.{} },
    bake_mean: [2]f64 = .{ 0, 0 },
    bake_sd: [2]f64 = .{ 1, 1 },
    baked: [2]?BakeKey = .{ null, null },
    serial: [2]u32 = .{ 0, 0 },
    /// dictionary orientations (4 f32 each) in the phase's fundamental zone, and on the device
    dict: []f32 = &.{},
    dict_gpu: Buf = .{},
    dict_grid: u32 = 0,
    /// the dictionary's leading components (k rows of pixels) and every entry's coefficients
    pca_w: Buf = .{},
    pca_coef: Buf = .{},
    pca_key: ?PcaKey = null,
    /// dictionaries of other grids (sweeps alternate between a few), set aside whole
    spare: [3]Dictionary = .{ .{}, .{}, .{} },
    spare_next: usize = 0,

    /// Make the dictionary of grid N the current one if it was set aside; the current one is
    /// set aside in its place (or over the oldest). False: none was kept, and the current
    /// fields are empty for a new one.
    fn takeDictionary(self: *Phase, g: *Gpu, N: u32) bool {
        var cur: Dictionary = .{ .dict = self.dict, .dict_gpu = self.dict_gpu, .dict_grid = self.dict_grid, .pca_w = self.pca_w, .pca_coef = self.pca_coef, .pca_key = self.pca_key };
        var found: ?Dictionary = null;
        var slot: usize = self.spare_next;
        for (&self.spare, 0..) |*d, i| {
            if (d.dict.len != 0 and d.dict_grid == N) {
                found = d.*;
                slot = i;
            }
        }
        if (found == null) {
            for (&self.spare, 0..) |*d, i| {
                if (d.dict.len == 0) slot = i;
            }
            self.spare[slot].release(g);
            self.spare_next = (slot + 1) % self.spare.len;
        }
        self.spare[slot] = if (cur.dict.len != 0) cur else .{};
        if (cur.dict.len == 0) cur.release(g);
        const now: Dictionary = found orelse .{};
        self.dict = now.dict;
        self.dict_gpu = now.dict_gpu;
        self.dict_grid = now.dict_grid;
        self.pca_w = now.pca_w;
        self.pca_coef = now.pca_coef;
        self.pca_key = now.pca_key;
        return found != null;
    }

    fn deinit(self: *Phase, g: *Gpu) void {
        for (&self.spare) |*d| d.release(g);
        g.release(&self.pca_w);
        g.release(&self.pca_coef);
        const gpa = g.gpa;
        for (0..2) |k| {
            g.release(&self.det[k].buf);
            g.release(&self.det[k].grad);
            gpa.free(self.work[k]);
        }
        gpa.free(self.dict);
        g.release(&self.dict_gpu);
        self.master.deinit();
    }
};

/// A phase's harmonic side at one bandwidth, from its master as filtered for `level`.
const BandPhase = struct {
    hm: master_mod.Harmonics,
    gpu: spherical.PhaseGpu,
    rden_key: ?RdenKey = null,
    level: Level,
    serial: u32,
};

/// What spherical indexing needs at one bandwidth: its tables and each phase's harmonic side.
/// Sweeps and the fits alternate between bandwidths, so a few are kept.
pub const Band = struct {
    L: u32,
    tables: spherical.Tables,
    ph: std.ArrayList(?BandPhase) = .empty,
    /// last use
    stamp: u64 = 0,

    fn dropPhase(self: *Band, g: *Gpu, pi: usize) void {
        if (self.ph.items[pi]) |*x| {
            x.hm.deinit(g.gpa);
            x.gpu.deinit(g);
        }
        self.ph.items[pi] = null;
    }

    fn destroy(self: *Band, g: *Gpu) void {
        for (0..self.ph.items.len) |pi| self.dropPhase(g, pi);
        self.ph.deinit(g.gpa);
        self.tables.deinit(g);
        g.gpa.destroy(self);
    }
};

const MAX_BANDS = 6;

const Memory = struct {
    data: []u8,
    info: scan_mod.Info,
};

/// Tuning how the work is cut up (settings.batch, refine_pixels, wg_refine, wg_fft) to this
/// machine and this scan. A sample of the region is indexed as indexing would index it, the
/// front end times that (the engine has no clock of its own), and a few moves are tried one
/// after another, each climbing from the values held: doubled while that is faster, else halved while
/// that is. About ten trials, not a search of every value. Nothing of the map's results is
/// touched, and neither the orientations indexing finds nor which matches are accepted
/// depend on these settings.
pub const Tune = struct {
    /// The settings tuned.
    pub const count = 4;
    pub const names = [count][]const u8{ "batch", "refine_pixels", "wg_refine", "wg_fft" };
    /// The moves, in the order they are tried: the settings each scales together.
    ///   refine_pixels          how many invocations a batch's refinement makes
    ///   batch, refine_pixels   the batch at that many invocations: the two are not
    ///                          independent, and the fastest pairs lie along this line
    ///   wg_refine              the refinement's workgroup
    ///   wg_fft                 the correlation's workgroup
    /// All but the last are timed on the first sweep, where most patterns go and reading,
    /// filtering and refinement weigh most; wg_fft shapes the correlation alone and is timed
    /// on the last sweep, at the full bandwidth.
    pub const moves = [_][count]bool{
        .{ false, true, false, false },
        .{ true, true, false, false },
        .{ false, false, true, false },
        .{ false, false, false, true },
    };
    pub const on_last = 3;
    /// How far each setting may climb.
    pub const lo = [count]u32{ 64, 16, 16, 16 };
    pub const hi = [count]u32{ 4096, 2048, 256, 256 };
    /// Timed runs of a trial (the faster counts). Before them one batch runs untimed: a
    /// workgroup size not used before is compiled in it, and a larger batch gets its buffers.
    pub const runs = 2;
    /// A move is kept only when it is faster by more than this share.
    pub const worth = 0.03;

    /// Patterns a trial indexes, by what the search costs, so that a trial stays a fraction of
    /// a second: the spherical correlation's cost follows the cube of its orientation grid
    /// (64 points per 360° up to bandwidth 32, 128 up to 64, 256 up to 128, 512 beyond).
    pub fn sampleOf(method: settings_mod.Method, bandwidth: u32, dict_grid: u32) usize {
        return switch (method) {
            .spherical => if (bandwidth <= 32) 8192 else if (bandwidth <= 64) 4096 else if (bandwidth <= 128) 1024 else 128,
            .dictionary => if (dict_grid <= 30) 1024 else 256,
            .pca => 4096,
        };
    }

    /// the samples timed on the first sweep and on the last: rectangles of the region from
    /// its first cell
    cells: [2][]u64 = .{ &.{}, &.{} },
    /// the values held, their seconds on the sweep being timed, and what tuning began with
    best: [count]u32 = @splat(0),
    best_time: f64 = std.math.inf(f64),
    first: [count]u32 = @splat(0),
    /// seconds before and after, on the first sweep and on the last (0: not timed)
    was: [2]f64 = .{ 0, 0 },
    now: [2]f64 = .{ 0, 0 },
    /// the trial in hand: the move (`moves.len`: none is left), whether it times the values
    /// held (what a sweep's candidates must beat) or a candidate, that candidate, which way
    /// the move is climbing and whether a step that way was kept
    move: u32 = 0,
    base: bool = true,
    cand: [count]u32 = @splat(0),
    up: bool = true,
    kept: bool = false,
    /// whether the last sweep is the one timed; the run of the trial and its fastest so far
    last: bool = false,
    run: u32 = 0,
    time: f64 = std.math.inf(f64),
    trials: u32 = 0,
    active: bool = false,
};

/// A cell a read must fetch, and where in the read it was asked for.
const Miss = struct {
    cell: u64,
    at: usize,

    fn before(_: void, a: Miss, b: Miss) bool {
        return a.cell < b.cell;
    }
};

/// Stored patterns kept on the host by map cell, the oldest dropped first: the neighbours that
/// consecutive batches share when patterns are averaged with their neighbours (NLPAR). Only
/// then: a pattern that passes through here is copied twice more than one read straight into
/// its batch, and on a fast device those copies are what a batch waits for.
const PatternCache = struct {
    /// Bytes of patterns kept at most: enough for the rows of the map between a block and the
    /// block below it on a scan of ordinary size.
    const budget: usize = if (gpu_mod.is_web) 96 << 20 else 384 << 20;

    /// bytes of one pattern
    pb: usize = 0,
    /// the cell each slot holds; slots fill in order and then wrap around from `next`
    cells: std.ArrayList(u64) = .empty,
    data: std.ArrayList(u8) = .empty,
    /// The slot each map cell is in (NO_SLOT: none), by cell: a table, not a hash map. A hash
    /// map that drops an entry for every one it takes fills with tombstones, and once none of
    /// its places is free a cell it does not hold costs a walk over all of them: on a scan
    /// larger than the cache that halved the speed of indexing in the browser.
    slot_of: std.ArrayList(u32) = .empty,
    next: u32 = 0,

    const NO_SLOT: u32 = 0xffffffff;

    fn deinit(self: *PatternCache, gpa: Allocator) void {
        self.cells.deinit(gpa);
        self.data.deinit(gpa);
        self.slot_of.deinit(gpa);
        self.* = .{};
    }

    /// Empty the cache for patterns of `pb` bytes.
    fn reset(self: *PatternCache, gpa: Allocator, pb: usize) void {
        self.deinit(gpa);
        self.pb = pb;
    }

    fn slots(self: *const PatternCache) usize {
        return if (self.pb == 0) 0 else budget / self.pb;
    }

    fn get(self: *const PatternCache, cell: u64) ?[]const u8 {
        if (cell >= self.slot_of.items.len) return null;
        const slot = self.slot_of.items[@intCast(cell)];
        if (slot == NO_SLOT) return null;
        return self.data.items[@as(usize, slot) * self.pb ..][0..self.pb];
    }

    /// Keep a pattern (a failed allocation only means it is not kept).
    fn put(self: *PatternCache, gpa: Allocator, cell: u64, bytes: []const u8) void {
        const max = self.slots();
        if (max == 0 or self.get(cell) != null) return;
        if (cell >= self.slot_of.items.len) {
            // (the table grows to the cell asked for and half as much again: a scan's cells come in order)
            const old = self.slot_of.items.len;
            const want: usize = @intCast(cell + cell / 2 + 1024);
            self.slot_of.resize(gpa, want) catch return;
            @memset(self.slot_of.items[old..], NO_SLOT);
        }
        var slot: u32 = undefined;
        if (self.cells.items.len < max) {
            self.cells.append(gpa, cell) catch return;
            self.data.appendSlice(gpa, bytes) catch {
                self.cells.items.len -= 1;
                return;
            };
            slot = @intCast(self.cells.items.len - 1);
        } else {
            slot = self.next;
            self.next = @intCast((@as(usize, slot) + 1) % max);
            self.slot_of.items[@intCast(self.cells.items[slot])] = NO_SLOT;
            self.cells.items[slot] = cell;
            @memcpy(self.data.items[@as(usize, slot) * self.pb ..][0..self.pb], bytes);
        }
        self.slot_of.items[@intCast(cell)] = slot;
    }
};

const Source = union(enum) {
    none,
    file: *scan_mod.Scan,
    memory: Memory,
};

/// Per map cell (row-major): the chosen phase's orientation and scores; phase −1: not indexed.
pub const ExportContext = struct {
    settings: settings_mod.Settings,
    geometry: Geometry,
    calibration: @import("oxford.zig").Calibration,
    phase_generation: u64,
    scan_signs: [2]i32,
};

pub fn sameExportContext(a: ExportContext, b: ExportContext) bool {
    var x = a.settings;
    var y = b.settings;
    inline for (.{ "batch", "tile_mb", "refine_pixels", "wg_fft", "wg_refine" }) |key| {
        @field(x, key) = 0;
        @field(y, key) = 0;
    }
    x.ipf_colors = .classic;
    y.ipf_colors = .classic;
    return std.meta.eql(x, y) and std.meta.eql(a.geometry, b.geometry) and @import("oxford.zig").Calibration.eql(a.calibration, b.calibration) and a.phase_generation == b.phase_generation and std.meta.eql(a.scan_signs, b.scan_signs);
}

/// Seed conditioning excludes only region selection. Raw poses/NCC/NMI do not
/// depend on ROI; chance-z, acceptance labels and export guards remain strict.
pub fn sameSeedContext(a: ExportContext, b: ExportContext) bool {
    var x = a;
    var y = b;
    inline for (.{ "roi_x", "roi_y", "roi_w", "roi_h" }) |field| {
        @field(x.settings, field) = 0;
        @field(y.settings, field) = 0;
    }
    return sameExportContext(x, y);
}

pub const Results = struct {
    seed_context: ?ExportContext = null,
    seed_valid: []bool = &.{},
    context: ?ExportContext = null,
    context_consistent: bool = true,
    /// Phase-major, best stored candidate per phase; no within-phase peak search.
    candidates: []Record = &.{},
    candidate_sweep: []i32 = &.{},
    candidate_phases: usize = 0,
    quat: []f32 = &.{},
    ncc: []f32 = &.{},
    nmi: []f32 = &.{},
    phase: []i32 = &.{},
    /// how far the match's NMI stands above chance, in chance's standard deviations
    z: []f32 = &.{},
    /// the sweep that found an accepted match (−1: none did)
    sweep: []i32 = &.{},
    /// whether the stored match's refinement had settled
    settled: []bool = &.{},

    pub fn resize(self: *Results, gpa: Allocator, n: usize) !void {
        const n_quat = std.math.mul(usize, 4, n) catch return error.ScanShapeTooLarge;
        _ = std.math.mul(usize, n_quat, @sizeOf(f32)) catch return error.ScanShapeTooLarge;
        self.free(gpa);
        self.seed_valid = try gpa.alloc(bool, n);
        self.quat = try gpa.alloc(f32, n_quat);
        self.ncc = try gpa.alloc(f32, n);
        self.nmi = try gpa.alloc(f32, n);
        self.phase = try gpa.alloc(i32, n);
        self.z = try gpa.alloc(f32, n);
        self.sweep = try gpa.alloc(i32, n);
        self.settled = try gpa.alloc(bool, n);
        self.clear();
    }

    fn clear(self: *Results) void {
        self.seed_context = null;
        @memset(self.seed_valid, false);
        self.context = null;
        self.context_consistent = true;
        @memset(self.candidate_sweep, -1);
        @memset(self.quat, 0);
        @memset(self.ncc, 0);
        @memset(self.nmi, 0);
        @memset(self.phase, -1);
        @memset(self.z, 0);
        @memset(self.sweep, -1);
        @memset(self.settled, false);
    }

    pub fn free(self: *Results, gpa: Allocator) void {
        gpa.free(self.seed_valid);
        gpa.free(self.candidates);
        gpa.free(self.candidate_sweep);
        gpa.free(self.quat);
        gpa.free(self.ncc);
        gpa.free(self.nmi);
        gpa.free(self.phase);
        gpa.free(self.z);
        gpa.free(self.sweep);
        gpa.free(self.settled);
        self.* = .{};
    }

    pub fn prepareCandidates(self: *Results, gpa: Allocator, phases: usize) !void {
        const count = try std.math.mul(usize, self.phase.len, phases);
        if (self.candidates.len == count and self.candidate_phases == phases) return;
        const records = try gpa.alloc(Record, count);
        errdefer gpa.free(records);
        const sweeps = try gpa.alloc(i32, count);
        @memset(sweeps, -1);
        gpa.free(self.candidates);
        gpa.free(self.candidate_sweep);
        self.candidates = records;
        self.candidate_sweep = sweeps;
        self.candidate_phases = phases;
    }
};

/// One pattern's outcome for one phase: what the device records, and (z) what the host adds.
pub const Record = extern struct {
    q: [4]f32,
    ncc: f32,
    nmi: f32,
    /// the rotation of the last refinement step that improved the match, radians: small when
    /// the refinement had settled, large when its iterations ran out on the way (0 without
    /// refinement)
    step: f32,
    /// (NMI − chance) / chance's spread: the NMI on a scale measured once for the scan (a
    /// sample of the region's patterns against the phase at random orientations), the same
    /// for every pattern. As the device writes it, the number of the batch that wrote it
    /// (`finishCells` checks it, then puts z there).
    z: f32,
};

comptime {
    // (the layout detector.wgsl's collect writes)
    std.debug.assert(@sizeOf(Record) == detector.RECORD * @sizeOf(f32));
}

/// What a search runs with: its method, and the bandwidth of the spherical correlation or the
/// dictionary's grid.
pub const Search = struct {
    method: settings_mod.Method = .spherical,
    bandwidth: u32,
    dict_grid: u32,
};

pub const Engine = struct {
    gpa: Allocator,
    dev: *gpu_mod.Backend.Device,
    g: Gpu,
    set: Settings = .{},
    /// Manual and restored project mask choices persist across scans; automatic choices
    /// are recalculated from every newly opened format and never become explicit.
    mask_explicit: bool = false,
    geo: Geometry = .{},
    calibration: @import("oxford.zig").Calibration = .{},
    phase_generation: u64 = 0,
    /// the geometry in use while `cellUnder` runs under another (reading and setting the
    /// geometry go to it meanwhile)
    geo_home: ?Geometry = null,
    /// The geometry as it was given: by the scan when it was opened, and by the caller since
    /// (exports.ze_set_geometry). The fits change `geo` and never this, and start from this, so
    /// that a fit is a function of the scan, the settings and what was given, not of the fits
    /// run before it.
    geo_given: Geometry = .{},
    /// The frame the exports write orientations in (convention.zig): the engine's own until
    /// one is identified from the maker's own orientations or set; a new scan resets it.
    convention: convention_mod.Frame = .{},
    /// an export made by a call that asked its size, for the call that takes it (exports.zig)
    export_held: []u8 = &.{},
    /// the source points the last plane fit measured, a fit pattern each (planefit.plane)
    plane_points: std.ArrayList(@import("planefit.zig").PlanePoint) = .empty,
    /// the crystal a master pattern is simulated for, and the last simulation (simulate.zig)
    crystal: ?crystal_mod.Cell = null,
    sim: ?simulate_mod.Simulation = null,
    /// the masters the app shows on a globe, kept on the device (sphereview.zig): a phase's,
    /// the simulation's
    view: [2]@import("sphereview.zig").Cache = .{ .{}, .{} },
    /// the crystal's cell the app shows, kept on the device (cellview.zig)
    cell_view: @import("cellview.zig").Cache = .{},
    events: events_mod.Sink,
    source: Source = .none,
    phases: std.ArrayList(Phase) = .empty,
    bands: std.ArrayList(*Band) = .empty,
    band_clock: u64 = 0,
    /// the spherical work buffers, pointed at the band in use
    sp: spherical.Spherical,
    prep: prep_mod.Prep,
    det: detector.Detector,
    pca: pca_mod.Pca,
    /// per level: the batch as uploaded, the batch filtered (n · h · w f32), its items
    /// ((t, pattern index) each), the static background and what it was made for
    raw: [2]Buf = .{ .{}, .{} },
    pats: [2]Buf = .{ .{}, .{} },
    items_gpu: [2]Buf = .{ .{}, .{} },
    bg_gpu: [2]Buf = .{ .{}, .{} },
    bg_key: [2]?BgKey = .{ null, null },
    raw_host: std.ArrayList(u8) = .empty,
    red_host: std.ArrayList(f32) = .empty,
    /// stored patterns by cell (see PatternCache), and scratch for what a read finds missing
    cache: PatternCache = .{},
    miss_host: std.ArrayList(u8) = .empty,
    miss_cells: std.ArrayList(u64) = .empty,
    miss: std.ArrayList(Miss) = .empty,
    /// averaging with neighbours: the kernels' buffers; the window in hand (its cells, the
    /// wanted ones first and then ring after ring around them, where each lies in that list,
    /// how many lie within each ring, and every cell's neighbours); what the first batch of an
    /// indexing run reports
    nl: nlpar_mod.Nlpar,
    nl_cells: std.ArrayList(u64) = .empty,
    nl_index: std.AutoHashMapUnmanaged(u64, u32) = .empty,
    nl_counts: std.ArrayList(u32) = .empty,
    nl_nbr: std.ArrayList(u32) = .empty,
    nl_report: enum { off, wanted, queued } = .off,
    /// for the report: every listed cell's noise, the wanted cells' shares of the stored
    /// patterns and where those lie in the list
    nl_noise: []f32 = &.{},
    nl_shares: []f32 = &.{},
    nl_where: []u32 = &.{},
    nl_stats_n: usize = 0,
    /// what the averaging did to the pattern last looked at (`pattern`): its summary and its
    /// shares of the stored patterns around it (`nl_view_side`² of them; 0: it was not averaged)
    nl_view: NlparSummary = .{ .noise = 0, .left = 1, .own = 1 },
    nl_view_shares: []f32 = &.{},
    nl_view_side: u32 = 0,
    /// the cells of the batch after the one being indexed, and what the host has ready of them
    /// by the time the device is done with this one: the cells read (`ahead_cells`), their
    /// stored bytes when `ahead_bytes` (a buffer the next read takes over whole), and their
    /// reduction by `ahead_bin` when `ahead_reduced`
    ahead: []const u64 = &.{},
    ahead_cells: std.ArrayList(u64) = .empty,
    ahead_host: std.ArrayList(u8) = .empty,
    ahead_bytes: bool = false,
    ahead_red: std.ArrayList(f32) = .empty,
    ahead_reduced: bool = false,
    ahead_bin: f64 = 0,
    /// the average pattern at stored size, and how many patterns it was asked over
    avg: []f32 = &.{},
    avg_n: u32 = 0,
    told_no_background: bool = false,
    /// each phase's spherical peaks; the refinement records; chance's mutual information
    peaks: std.ArrayList(Buf) = .empty,
    records: Buf = .{},
    /// the last recorded detector batch's number, which its records carry back
    batch_stamp: u32 = 0,
    chance: Buf = .{},
    /// what mutual information chance gives on this scan, per phase: measured once on a sample
    /// of the region (`calibrate`), the same for every pattern, and what it was measured for
    chance_med: []f32 = &.{},
    chance_spread: []f32 = &.{},
    chance_key: u64 = 0,
    /// counts the scans opened, made or given (closeScan), so that what is measured on one is
    /// not taken for the next's
    scan_serial: u64 = 0,
    /// −1 where the scan open went the other way from what the settings' scan signs say
    /// (Info.scan_turn, or the file's own pattern centres): `scanSigns`
    scan_turn: i32 = 1,
    res: Results = .{},
    /// indexing in progress: the cells of the sweep in progress and how many are done
    index_cells: []u64 = &.{},
    index_done: usize = 0,
    index_region: usize = 0,
    index_sweep: u32 = 0,
    index_accepted: usize = 0,
    /// the last pattern-centre search: its score volume (z, y, x) and axes
    pc_scores: []f32 = &.{},
    /// every grid that search scored, in order (pcfit.zig)
    pc_history: std.ArrayList(@import("pcfit.zig").Level) = .empty,
    pc_axes: [3][]f64 = .{ &.{}, &.{}, &.{} },
    /// the fit patterns' cells and, per cell, phase and orientation from the last fit
    fit_cells: []u64 = &.{},
    fit_quats: []f32 = &.{},
    fit_phase: []i32 = &.{},
    fit_ncc: []f32 = &.{},
    tune: Tune = .{},
    cancel: @import("cancellation.zig").Flag = .{},
    // One-shot browser request, consumed before overlap can suspend.
    next_overlap_bandwidth: u32 = 0,
    err_buf: [1024]u8 = undefined,
    err_len: usize = 0,

    pub fn create(gpa: Allocator, dev: *gpu_mod.Backend.Device) !*Engine {
        const self = try gpa.create(Engine);
        errdefer gpa.destroy(self);
        self.* = .{ .gpa = gpa, .dev = dev, .g = undefined, .events = .{ .gpa = gpa }, .sp = undefined, .prep = undefined, .det = undefined, .pca = undefined, .nl = undefined };
        self.g = try Gpu.init(gpa, dev);
        self.sp = spherical.Spherical.init(&self.g);
        self.prep = prep_mod.Prep.init(&self.g);
        self.det = detector.Detector.init(&self.g);
        self.pca = pca_mod.Pca.init(&self.g);
        self.nl = nlpar_mod.Nlpar.init(&self.g);
        return self;
    }

    pub fn destroy(self: *Engine) void {
        const gpa = self.gpa;
        self.closeScan();
        for (self.bands.items) |b| b.destroy(&self.g);
        self.bands.deinit(gpa);
        self.sp.deinit();
        for (self.phases.items) |*p| p.deinit(&self.g);
        self.phases.deinit(gpa);
        self.prep.deinit();
        self.det.deinit();
        self.pca.deinit();
        self.nl.deinit();
        self.nl_cells.deinit(gpa);
        self.nl_index.deinit(gpa);
        self.nl_counts.deinit(gpa);
        self.nl_nbr.deinit(gpa);
        gpa.free(self.nl_noise);
        gpa.free(self.nl_shares);
        gpa.free(self.nl_where);
        gpa.free(self.nl_view_shares);
        for (self.peaks.items) |*b| self.g.release(b);
        self.peaks.deinit(gpa);
        for (0..2) |k| inline for (.{ "raw", "pats", "items_gpu", "bg_gpu" }) |f| self.g.release(&@field(self, f)[k]);
        self.g.release(&self.records);
        self.g.release(&self.chance);
        gpa.free(self.chance_med);
        gpa.free(self.chance_spread);
        self.raw_host.deinit(gpa);
        self.red_host.deinit(gpa);
        self.cache.deinit(gpa);
        self.miss_host.deinit(gpa);
        self.miss_cells.deinit(gpa);
        self.miss.deinit(gpa);
        self.ahead_cells.deinit(gpa);
        self.ahead_host.deinit(gpa);
        self.ahead_red.deinit(gpa);
        self.res.free(gpa);
        gpa.free(self.pc_scores);
        for (self.pc_axes) |a| gpa.free(a);
        @import("pcfit.zig").clearHistory(self);
        self.pc_history.deinit(gpa);
        self.plane_points.deinit(gpa);
        if (self.crystal) |*c| c.deinit();
        if (self.sim) |*x| x.deinit();
        for (&self.view) |*v| v.deinit(&self.g);
        self.cell_view.deinit(&self.g);
        self.events.deinit();
        self.g.deinit();
        gpa.destroy(self);
    }

    // ── errors ──

    pub fn fail(self: *Engine, comptime fmt: []const u8, args: anytype) void {
        const m = std.fmt.bufPrint(&self.err_buf, fmt, args) catch self.err_buf[0..0];
        self.err_len = m.len;
    }

    /// The first engine or GPU error since the last call (copied into `buf`), or 0.
    pub fn takeError(self: *Engine, buf: []u8) usize {
        if (self.g.oversize) |o| {
            const tail = std.fmt.bufPrint(self.err_buf[self.err_len..], " ({s}: {d}×{d}×{d} workgroups, limit {d} per dimension)", .{
                self.g.pipeName(o.pipe), o.n[0], o.n[1], o.n[2], gpu_mod.MAX_GROUPS,
            }) catch "";
            self.err_len += tail.len;
            self.g.oversize = null;
        }
        var m: []const u8 = self.err_buf[0..self.err_len];
        if (m.len == 0) m = self.dev.takeError() orelse "";
        const n = @min(m.len, buf.len);
        @memcpy(buf[0..n], m[0..n]);
        self.err_len = 0;
        return n;
    }

    // ── settings ──

    pub fn configure(self: *Engine, json: []const u8) !void {
        const explicit = try self.set.applyTrackingMask(self.gpa, json);
        self.mask_explicit = self.mask_explicit or explicit;
    }

    // ── scan ──

    pub fn closeScan(self: *Engine) void {
        switch (self.source) {
            .file => |s| s.close(),
            .memory => |m| self.gpa.free(m.data),
            .none => {},
        }
        self.source = .none;
        self.calibration = .{};
        self.geo.plane_kind = 0;
        self.geo.plane = (Geometry{}).plane;
        self.plane_points.clearRetainingCapacity();
        self.scan_serial += 1;
        self.chance_key = 0;
        self.scan_turn = 1;
        self.convention = .{};
        self.gpa.free(self.export_held);
        self.export_held = &.{};
        self.tuneFree();
        self.cache.deinit(self.gpa);
        self.ahead = &.{};
        self.ahead_cells.clearRetainingCapacity();
        self.ahead_bytes = false;
        self.ahead_reduced = false;
        self.res.free(self.gpa);
        self.gpa.free(self.index_cells);
        self.index_cells = &.{};
        self.index_done = 0;
        self.index_region = 0;
        self.index_sweep = 0;
        self.index_accepted = 0;
        self.clearFit();
        self.gpa.free(self.avg);
        self.avg = &.{};
        self.avg_n = 0;
        self.bg_key = .{ null, null };
        self.told_no_background = false;
    }

    /// The frame the maker's orientations in `ref` are in (with none, the scan file's own),
    /// found against this engine's results (convention.identify). The exports take it when it
    /// was told clearly: its median under 5°, and the best other convention's at least twice
    /// that and 5° more. The misorientation left is reported, never applied.
    pub fn identifyFrame(self: *Engine, ref: ?convention_mod.Reference) !struct { found: convention_mod.Found, installed: bool } {
        if (self.phases.items.len == 0) return error.NoPhases;
        var stored: ?scan_mod.Stored = null;
        defer if (stored) |*st| st.deinit(self.gpa);
        const r = ref orelse blk: {
            const s = switch (self.source) {
                .file => |s| s,
                else => return error.NoStoredOrientations,
            };
            stored = (try s.storedOrientations(self.gpa)) orelse return error.NoStoredOrientations;
            break :blk convention_mod.Reference{ .euler = stored.?.euler, .phase = stored.?.phase };
        };
        var views: [convention_mod.max_phases]convention_mod.PhaseView = undefined;
        const n_ph = @min(self.phases.items.len, convention_mod.max_phases);
        for (self.phases.items[0..n_ph], 0..) |*p, i| views[i] = .{ .rotations = p.master.rotations(), .lattice = p.master.lattice };
        const cols: usize = if (self.info()) |i| i.cols else |_| 0;
        const found = try convention_mod.identify(self.gpa, self.res.quat, self.res.phase, cols, r, views[0..n_ph]);
        const holds = found.median_deg < 5 and found.runner_up_deg >= 2 * found.median_deg + 5;
        if (holds) self.convention = found.frame;
        var line: [256]u8 = undefined;
        var w = std.Io.Writer.fixed(&line);
        found.frame.describe(&w) catch {};
        const deg = std.math.radiansToDegrees;
        const m = found.misorientation;
        const th = rot.norm(m);
        const u = if (th > 0) rot.scale(m, 1 / th) else rot.Vec3{ 0, 0, 0 };
        self.events.log("frame of the stored orientations: {s}; {d} cells, median {d:.2}°, {d:.0} % within 2°, {d:.1}° under the next convention{s}; misorientation left {d:.2}° about ({d:.3}, {d:.3}, {d:.3}) (x, y, z: toward the detector, the tilt axis, out of the surface), changing {d:.2}° from the first row to the last and {d:.2}° from the first column to the last", .{ w.buffered(), found.cells, found.median_deg, 100 * found.within, found.runner_up_deg, if (holds) "; the exports take it" else "; not told clearly, the exports keep theirs", deg(th), u[0], u[1], u[2], deg(rot.norm(found.misorientation_rows)), deg(rot.norm(found.misorientation_cols)) });
        return .{ .found = found, .installed = holds };
    }

    fn clearFit(self: *Engine) void {
        const gpa = self.gpa;
        gpa.free(self.fit_cells);
        gpa.free(self.fit_quats);
        gpa.free(self.fit_phase);
        gpa.free(self.fit_ncc);
        self.fit_cells = &.{};
        self.fit_quats = &.{};
        self.fit_phase = &.{};
        self.fit_ncc = &.{};
    }

    /// Open a scan from files (the pattern file first). The geometry takes whatever the file
    /// records; the rest keeps its current value.
    pub fn openScan(self: *Engine, paths: []const []const u8) !void {
        var settings = self.set;
        wholeMap(&settings);
        try self.openScanPrepared(paths, settings, settings, self.mask_explicit, .auto);
    }

    /// Project/read patches are prospective until parsing and result allocation succeed.
    /// Read hints win while opening; the project patch wins in committed settings.
    pub fn openScanConfigured(self: *Engine, paths: []const []const u8, settings_json: []const u8, hints_json: []const u8) !void {
        var read_settings = self.set;
        wholeMap(&read_settings);
        var explicit = self.mask_explicit;
        const project_mask = try read_settings.applyTrackingMask(self.gpa, settings_json);
        explicit = explicit or project_mask;
        var hints = try std.json.parseFromSlice(std.json.Value, self.gpa, hints_json, .{});
        defer hints.deinit();
        if (hints.value != .object) return error.ExpectedObject;
        const hints_obj = &hints.value.object;
        const oxford_patterns: scan_mod.OxfordPatternChoice = if (hints_obj.fetchOrderedRemove("oxford_patterns")) |entry|
            switch (entry.value) {
                .string => |name| std.meta.stringToEnum(scan_mod.OxfordPatternChoice, name) orelse return error.InvalidOxfordPatternChoice,
                else => return error.InvalidOxfordPatternChoice,
            }
        else
            .auto;
        const hints_mask = try read_settings.applyObjectTrackingMask(self.gpa, hints_obj.*);
        explicit = explicit or hints_mask;
        var final_settings = read_settings;
        _ = try final_settings.applyTrackingMask(self.gpa, settings_json);
        try self.openScanPrepared(paths, read_settings, final_settings, explicit, oxford_patterns);
    }

    fn openScanPrepared(self: *Engine, paths: []const []const u8, read_settings: Settings, final_settings: Settings, explicit: bool, oxford_patterns: scan_mod.OxfordPatternChoice) !void {
        const s = try scan_mod.Scan.open(self.gpa, paths, .{
            .rows = read_settings.map_rows,
            .cols = read_settings.map_cols,
            .pat_h = read_settings.pattern_height,
            .pat_w = read_settings.pattern_width,
            .oxford_patterns = oxford_patterns,
            .unprocessed = read_settings.stored_patterns == .unprocessed,
        });
        errdefer s.close();
        // Count and quaternion byte sizes must fit wasm32 before any resize arithmetic.
        const n = std.math.cast(usize, s.count()) orelse return error.ScanShapeTooLarge;
        var replacement: Results = .{};
        errdefer replacement.free(self.gpa);
        try replacement.resize(self.gpa, n);
        // No fallible operations below: only now may the current scan be replaced.
        self.closeScan();
        self.source = .{ .file = s };
        self.res = replacement;
        self.set = final_settings;
        self.mask_explicit = explicit;
        if (!self.mask_explicit) self.set.mask = s.format.defaultMask();
        self.adoptInfo(s.info, read_settings.detector_width_mm);
        self.geo_given = self.geo;
        self.events.log("opened {s}: {d} × {d} patterns of {d} × {d}", .{ s.formatName(), s.info.rows, s.info.cols, s.info.pat_h, s.info.pat_w });
        if (s.oxford_patterns) |source| self.events.log("Oxford pattern source: {s}", .{source.datasetName()});
        const kinds = s.info.pattern_kinds;
        if (kinds == scan_mod.KIND_PROCESSED | scan_mod.KIND_UNPROCESSED) {
            self.events.log("the file stores processed and unprocessed patterns: reading the {s} (stored_patterns)", .{if (s.info.unprocessed) "unprocessed" else "processed"});
        } else if (kinds != 0 and s.info.unprocessed != (self.set.stored_patterns == .unprocessed)) {
            self.events.log("stored_patterns asks for {s} patterns; the file stores only {s} ones, read instead", .{ @tagName(self.set.stored_patterns), if (s.info.unprocessed) "unprocessed" else "processed" });
        }
        if (!std.math.isNan(s.info.detector_rot[0])) self.events.log("the detector's orientation as the file stores it: {d:.2}° from the nominal one", .{std.math.radiansToDegrees(rot.norm(s.info.detector_rot))});
        if (self.calibration.status() != .supported) self.takeTurn(s.info);
        // (as #13 found: a tilt about another axis than the scan's x is not what the geometry is)
        if (@abs(s.info.tilt_axis_deg) > 1e-4) self.events.log("the file tilts the sample about an axis {d:.2}° from the scan's x, which is not read: the geometry tilts it about x", .{s.info.tilt_axis_deg});
    }

    /// Which way the beam went: as the scan signs say, or the other way along both. The file's
    /// own pattern centres decide where they move measurably from cell to cell (a maker's
    /// calibration moves them with the beam), else what the file says of its scan's turn.
    fn takeTurn(self: *Engine, si: scan_mod.Info) void {
        self.scan_turn = 1;
        const said = si.scan_turn;
        if (si.turn_note.len != 0) self.events.log("the scan's turn as the file stores it: {s}", .{si.turn_note});
        const stored = [2]rot.Vec3{ si.pc_per_col, si.pc_per_row };
        var ok = si.rows >= 2 and si.cols >= 2;
        for (stored) |v| for (v) |x| {
            ok = ok and std.math.isFinite(x);
        };
        if (ok) ok = rot.norm(stored[0]) > 1e-7 and rot.norm(stored[1]) > 1e-7;
        if (ok) {
            // the engine's own change from the middle cell to the next column and the next row,
            // the beam as the signs say
            // Infer the acquisition's turn before applying the caller's requested scan signs.
            const defaults: Settings = .{};
            const signs = [2]i32{ defaults.scan_x_sign, defaults.scan_y_sign };
            const R = self.calibration.rotation(&self.geo);
            const sz = self.sizeOf(.full) catch {
                self.scan_turn = said;
                return;
            };
            const mid = @as(u64, (si.rows - 1) / 2) * si.cols + (si.cols - 1) / 2;
            const at = [3]u64{ mid, mid + 1, mid + si.cols };
            var pc: [3]rot.Vec3 = undefined;
            for (at, &pc) |c, *p| p.* = Geometry.pcOf(R, self.placeWithSigns(&self.geo, c, .full, signs) catch {
                self.scan_turn = said;
                return;
            }, sz[0], sz[1]);
            var agree: [2]f64 = undefined;
            for (0..2) |k| {
                const ours = rot.sub(pc[k + 1], pc[0]);
                agree[k] = rot.dot(ours, stored[k]) / @max(rot.norm(ours) * rot.norm(stored[k]), 1e-300);
            }
            if (agree[0] > 0.5 and agree[1] > 0.5) {
                self.scan_turn = 1;
            } else if (agree[0] < -0.5 and agree[1] < -0.5) {
                self.scan_turn = -1;
            } else ok = false;
            if (ok) {
                self.events.log("the file's own pattern centres move {s} the scan signs say{s}", .{ if (self.scan_turn == 1) "as the beam does where" else "as the beam does going the other way from where", if (self.scan_turn != said) " (taken over what it says of its scan's turn)" else "" });
                return;
            }
        }
        self.scan_turn = said;
    }

    /// The scan's signs (settings.scan_x_sign, scan_y_sign) as the scan open went.
    pub fn scanSigns(self: *const Engine) [2]i32 {
        if (self.calibration.status() == .supported) return self.calibration.scanSigns(self.set.scan_x_sign, self.set.scan_y_sign);
        return .{ self.set.scan_x_sign * self.scan_turn, self.set.scan_y_sign * self.scan_turn };
    }

    pub fn exportContext(self: *const Engine) ExportContext {
        return .{ .settings = self.set, .geometry = self.geo, .calibration = self.calibration, .phase_generation = self.phase_generation, .scan_signs = self.scanSigns() };
    }

    /// Take patterns from memory: rows · cols patterns of h × w pixels, row-major, copied.
    pub fn setPatterns(self: *Engine, data: []const u8, pixel: scan_mod.Pixel, rows: u32, cols: u32, h: u32, w: u32) !void {
        const bpp: usize = switch (pixel) {
            .u8 => 1,
            .u16 => 2,
            .f32 => 4,
        };
        if (rows == 0 or cols == 0 or h == 0 or w == 0) return error.BadShape;
        if (data.len != @as(usize, rows) * cols * h * w * bpp) return error.BadLength;
        const copy = try self.gpa.dupe(u8, data);
        self.closeScan();
        self.source = .{ .memory = .{ .data = copy, .info = .{ .rows = rows, .cols = cols, .pat_h = h, .pat_w = w, .pixel = pixel } } };
        wholeMap(&self.set);
        try self.res.resize(self.gpa, @as(usize, rows) * cols);
        if (!self.mask_explicit) self.set.mask = true;
        // (patterns from memory come with no geometry: a pixel size set by hand stays)
        if (!(self.geo.px_um > 0)) self.geo.px_um = self.defaultPx();
    }

    /// A new scan (or new patterns) starts at the whole map: a region drawn on another scan
    /// says nothing of this one, and would silently keep the fits and indexing to it.
    fn wholeMap(set: *Settings) void {
        set.roi_x = 0;
        set.roi_y = 0;
        set.roi_w = 0;
        set.roi_h = 0;
    }

    fn adoptInfo(self: *Engine, si: scan_mod.Info, detector_width_mm: f64) void {
        if (si.pc_note.len != 0) self.events.log("{s}", .{si.pc_note});
        self.calibration = si.calibration;
        const g = &self.geo;
        if (!std.math.isNan(si.pc[0]) and !std.math.isNan(si.pc[1]) and !std.math.isNan(si.pc[2])) {
            g.pc = si.pc;
            g.rot = .{ 0, 0, 0 };
        }
        if (!std.math.isNan(si.detector_rot[0])) g.rot = si.detector_rot;
        if (!std.math.isNan(si.sample_tilt_deg)) g.sample_tilt_deg = si.sample_tilt_deg;
        if (!std.math.isNan(si.detector_tilt_deg)) g.detector_tilt_deg = si.detector_tilt_deg;
        g.px_um = if (std.math.isFinite(si.px_um) and si.px_um > 0) si.px_um else 1000.0 * detector_width_mm / @as(f64, @floatFromInt(@max(si.pat_w, 1)));
        if (!std.math.isNan(si.step_x_um)) g.step_x_um = si.step_x_um;
        if (!std.math.isNan(si.step_y_um)) g.step_y_um = si.step_y_um;
        if (si.calibration.status() == .supported) {
            // Oxford has a full calibrated baseline; `rot` is only its fitted
            // residual. Never inherit unknown pixel size from a previous scan.
            g.rot = .{ 0, 0, 0 };
            g.detector_tilt_deg = 0;
            g.px_um = if (std.math.isFinite(si.px_um)) si.px_um else 0;
        } else if (si.calibration.metadata != null) {
            self.events.log("Oxford automatic frame unavailable: {s}", .{@tagName(si.calibration.status())});
        }
    }

    /// The detector's pixel size when none is set by hand: the file's, or for a scan that
    /// records none the detector's width (settings.detector_width_mm) over the pattern's
    /// stored pixels across. 0 with no scan open.
    pub fn defaultPx(self: *const Engine) f64 {
        const i = self.info() catch return 0;
        if (!std.math.isNan(i.px_um) and i.px_um > 0) return i.px_um;
        return 1000.0 * self.set.detector_width_mm / @as(f64, @floatFromInt(@max(i.pat_w, 1)));
    }

    pub fn info(self: *const Engine) !scan_mod.Info {
        return switch (self.source) {
            .file => |s| s.info,
            .memory => |m| m.info,
            .none => error.NoScan,
        };
    }

    pub fn scanInfo(self: *const Engine) ScanInfo {
        const i = self.info() catch return .{};
        var out: ScanInfo = .{ .rows = i.rows, .cols = i.cols, .pat_h = i.pat_h, .pat_w = i.pat_w, .pixel = @intFromEnum(i.pixel) };
        out.has_pc = @intFromBool(!std.math.isNan(i.pc[0]));
        out.has_px = @intFromBool(!std.math.isNan(i.px_um));
        if (self.calibration.metadata != null) {
            out.geometry_frame = if (self.calibration.status() == .supported) 2 else 3;
        }
        if (!std.math.isNan(i.voltage_kv)) out.voltage_kv = i.voltage_kv;
        switch (self.source) {
            .file => |s| {
                out.has_background = @intFromBool(s.background.len != 0);
                out.format = @intFromEnum(s.format);
                out.pattern_kinds = s.info.pattern_kinds | @as(u32, if (s.info.unprocessed) 4 else 0);
                out.bytes = @floatFromInt(s.files[0].size);
            },
            .memory => |m| {
                out.format = 100;
                out.bytes = @floatFromInt(m.data.len);
            },
            .none => {},
        }
        return out;
    }

    // ── levels ──

    /// The factor a level's patterns are reduced by.
    pub fn binOf(self: *const Engine, lv: Level) f64 {
        return if (lv == .full) 1 else self.set.binning;
    }

    /// The level a stage works at (the one level there is when nothing is reduced).
    pub fn levelOf(self: *const Engine, stage: Stage) Level {
        if (self.set.binning == 1) return .reduced;
        return switch (self.set.full_resolution) {
            .none => .reduced,
            .scores => if (stage == .score) .full else .reduced,
            .refine => if (stage == .search) .reduced else .full,
        };
    }

    /// The level patterns are looked at: the reduced one, or as stored.
    pub fn viewLevel(self: *const Engine, full: bool) Level {
        return if (full and self.set.binning != 1) .full else .reduced;
    }

    /// Size of a pattern at a level.
    pub fn patternShape(self: *const Engine, lv: Level) ![2]u32 {
        const i = try self.info();
        const hw = reduce.shape(i.pat_h, i.pat_w, self.binOf(lv));
        if (hw[0] < 4 or hw[1] < 4) return error.PatternTooSmall;
        return hw;
    }

    /// The stored pattern's size in a level's pixels (rows, cols): what the pattern centre's
    /// fractions are fractions of. Not whole numbers when the reduction dropped a remainder.
    pub fn sizeOf(self: *const Engine, lv: Level) ![2]f64 {
        const i = try self.info();
        const f = self.binOf(lv);
        return .{ @as(f64, @floatFromInt(i.pat_h)) / f, @as(f64, @floatFromInt(i.pat_w)) / f };
    }

    /// Angular width of one of a level's pixels, radians: settings.pixel_angle_deg for a stored
    /// pixel when set, else the pixel at the pattern centre. 0 without a scan.
    pub fn pixelAngle(self: *const Engine, lv: Level) f64 {
        const i = self.info() catch return 0;
        const stored = if (self.set.pixel_angle_deg > 0)
            std.math.degreesToRadians(self.set.pixel_angle_deg)
        else
            self.geo.pixelAngle(@floatFromInt(i.pat_h));
        return stored * self.binOf(lv);
    }

    // ── reading ──

    fn patternBytes(self: *const Engine) !usize {
        const i = try self.info();
        return @as(usize, i.pat_h) * i.pat_w * bytesPerPixel(i.pixel);
    }

    /// The stored bytes of the given cells, back to back (padded to whole words). Ascending
    /// cells of a file are fetched together. `cached`: for indexing, which reads ahead: what
    /// was read ahead is handed over as it lies, and with neighbours shared between batches
    /// (NLPAR) the rest goes by way of the pattern cache.
    fn readCells(self: *Engine, cells: []const u64, cached: bool) ![]u8 {
        const i = try self.info();
        const pb = try self.patternBytes();
        const count = @as(u64, i.rows) * i.cols;
        for (cells) |c| if (c >= count) return error.OutOfRange;
        if (cached and self.ahead_bytes and std.mem.eql(u64, self.ahead_cells.items, cells)) {
            std.mem.swap(std.ArrayList(u8), &self.raw_host, &self.ahead_host);
            self.ahead_bytes = false;
            return self.raw_host.items;
        }
        const total = std.mem.alignForward(usize, cells.len * pb, 4);
        try self.raw_host.resize(self.gpa, total);
        const buf = self.raw_host.items;
        @memset(buf[cells.len * pb ..], 0);
        const out = buf[0 .. cells.len * pb];
        switch (self.source) {
            .file => |s| try self.fetch(s, cells, out, cached and self.set.nlpar and self.cacheHolds(pb, cells.len)),
            .memory => |m| for (cells, 0..) |c, k| {
                @memcpy(out[k * pb ..][0..pb], m.data[@intCast(c * pb)..][0..pb]);
            },
            .none => return error.NoScan,
        }
        return buf;
    }

    /// Whether the cache can serve a read of `n` patterns of `pb` bytes (it is emptied for a
    /// new pattern size).
    fn cacheHolds(self: *Engine, pb: usize, n: usize) bool {
        if (self.cache.pb != pb) self.cache.reset(self.gpa, pb);
        return n <= self.cache.slots();
    }

    /// Read cells of a file, in any order and with repeats: `out` (when given) gets them back
    /// to back as asked. `through`: by way of the cache, which serves what it holds and keeps
    /// what it does not. What must be read is read in ascending order, neighbours together.
    fn fetch(self: *Engine, s: *scan_mod.Scan, cells: []const u64, out: ?[]u8, through: bool) !void {
        const gpa = self.gpa;
        const pb = try self.patternBytes();
        if (!through) {
            // already in file order (a preview, a plain batch): straight to where it goes
            var ascending = true;
            for (1..cells.len) |k| {
                if (cells[k] <= cells[k - 1]) ascending = false;
            }
            if (ascending) return if (out) |o| s.readCells(cells, o);
        }
        self.miss.clearRetainingCapacity();
        // what is held first: keeping what is missing may push it out
        for (cells, 0..) |c, k| {
            if (through) if (self.cache.get(c)) |bytes| {
                if (out) |o| @memcpy(o[k * pb ..][0..pb], bytes);
                continue;
            };
            try self.miss.append(gpa, .{ .cell = c, .at = k });
        }
        const miss = self.miss.items;
        if (miss.len == 0) return;
        std.mem.sort(Miss, miss, {}, Miss.before);
        self.miss_cells.clearRetainingCapacity();
        for (miss, 0..) |m, j| {
            if (j == 0 or m.cell != miss[j - 1].cell) try self.miss_cells.append(gpa, m.cell);
        }
        try self.miss_host.resize(gpa, self.miss_cells.items.len * pb);
        try s.readCells(self.miss_cells.items, self.miss_host.items);
        var u: usize = 0;
        for (miss, 0..) |m, j| {
            if (j != 0 and m.cell != miss[j - 1].cell) u += 1;
            const bytes = self.miss_host.items[u * pb ..][0..pb];
            if (out) |o| @memcpy(o[m.at * pb ..][0..pb], bytes);
            if (through) self.cache.put(gpa, m.cell, bytes);
        }
    }

    /// Do on the host what indexing `cells` will need of it, while the device is busy with
    /// the batch before: read their patterns and, when a level is reduced, reduce them.
    /// Whatever goes wrong here is left for the upload proper to report.
    fn readAhead(self: *Engine, cells: []const u64) void {
        const gpa = self.gpa;
        const i = self.info() catch return;
        const pb = self.patternBytes() catch return;
        self.ahead_cells.clearRetainingCapacity();
        self.ahead_bytes = false;
        self.ahead_reduced = false;
        var list = cells;
        if (self.set.nlpar) {
            _ = self.nlparWindow(cells) catch return;
            list = self.nl_cells.items;
        } else if (self.cache.pb != 0) {
            // (no batch shares neighbours with the next any more)
            self.cache.deinit(gpa);
        }
        if (list.len == 0) return;
        const bin = self.binOf(.reduced);
        var reduced = false;
        for ([_]Stage{ .search, .refine, .score }) |st| {
            if (bin != 1 and self.levelOf(st) == .reduced) reduced = true;
        }
        // With neighbours shared, by way of the cache, which then has them for the batches
        // after as well. Otherwise straight into a buffer the batch takes over as it lies.
        const through = self.set.nlpar and self.source == .file and self.cacheHolds(pb, 2 * list.len);
        if (through and !reduced) {
            self.fetch(self.source.file, list, null, true) catch {};
            return;
        }
        // (patterns in memory need no reading, but gathering the batch's out of the map is a copy
        // as long as a read, so it is made here too, while the device works; averaged ones are
        // gathered window by window)
        if (self.source != .file and !reduced and self.set.nlpar) return;
        self.ahead_host.resize(gpa, std.mem.alignForward(usize, list.len * pb, 4)) catch return;
        @memset(self.ahead_host.items[list.len * pb ..], 0);
        const bytes = self.ahead_host.items[0 .. list.len * pb];
        switch (self.source) {
            .file => |s| self.fetch(s, list, bytes, through) catch return,
            .memory => |m| for (list, 0..) |c, k| {
                @memcpy(bytes[k * pb ..][0..pb], m.data[@intCast(c * pb)..][0..pb]);
            },
            .none => return,
        }
        self.ahead_cells.appendSlice(gpa, list) catch return;
        self.ahead_bytes = true;
        if (!reduced) return;
        const hw = self.patternShape(.reduced) catch return;
        self.ahead_red.resize(gpa, list.len * @as(usize, hw[0]) * hw[1]) catch return;
        reduce.patterns(gpa, bytes, list.len, i.pat_h, i.pat_w, @intCast(bytesPerPixel(i.pixel)), bin, self.ahead_red.items) catch return;
        self.ahead_bin = bin;
        self.ahead_reduced = true;
    }

    /// The patterns of `cells` (their stored bytes in `bytes`) reduced for a level, into
    /// `red_host`: what was reduced ahead, when that was these cells.
    fn reduceLevel(self: *Engine, lv: Level, cells: []const u64, bytes: []const u8) !void {
        const i = try self.info();
        const bin = self.binOf(lv);
        if (self.ahead_reduced and self.ahead_bin == bin and std.mem.eql(u64, self.ahead_cells.items, cells)) {
            std.mem.swap(std.ArrayList(f32), &self.red_host, &self.ahead_red);
            self.ahead_reduced = false;
            return;
        }
        const hw = try self.patternShape(lv);
        const bpp = bytesPerPixel(i.pixel);
        try self.red_host.resize(self.gpa, cells.len * @as(usize, hw[0]) * hw[1]);
        try reduce.patterns(self.gpa, bytes[0 .. cells.len * i.pat_h * i.pat_w * bpp], cells.len, i.pat_h, i.pat_w, @intCast(bpp), bin, self.red_host.items);
    }

    // ── averaging with neighbours ──

    /// Bytes of one copy of a window's patterns on the device at most: a batch whose window
    /// would take more is averaged in parts.
    const nlpar_bytes: u64 = 256 << 20;

    /// The window averaging `wanted` needs: the wanted cells, then ring after ring of the
    /// map's cells around them, as far out as the passes reach, with every cell's neighbours.
    /// Left in the engine's lists; the result points into them.
    fn nlparWindow(self: *Engine, wanted: []const u64) !nlpar_mod.Window {
        const gpa = self.gpa;
        const i = try self.info();
        const rows: i64 = i.rows;
        const cols: i64 = i.cols;
        const radius = self.set.nlpar_radius;
        const passes = self.set.nlpar_iterations;
        const cells = &self.nl_cells;
        cells.clearRetainingCapacity();
        self.nl_index.clearRetainingCapacity();
        self.nl_counts.clearRetainingCapacity();
        for (wanted) |c| {
            const at = try self.nl_index.getOrPut(gpa, c);
            if (!at.found_existing) at.value_ptr.* = @intCast(cells.items.len);
            try cells.append(gpa, c);
        }
        try self.nl_counts.append(gpa, @intCast(cells.items.len));
        var first: usize = 0;
        for (0..nlpar_mod.Window.ringsOf(radius, passes)) |_| {
            const end = cells.items.len;
            for (first..end) |k| {
                const y: i64 = @intCast(cells.items[k] / i.cols);
                const x: i64 = @intCast(cells.items[k] % i.cols);
                var dy: i64 = -1;
                while (dy <= 1) : (dy += 1) {
                    var dx: i64 = -1;
                    while (dx <= 1) : (dx += 1) {
                        if (y + dy < 0 or y + dy >= rows or x + dx < 0 or x + dx >= cols) continue;
                        const at = try self.nl_index.getOrPut(gpa, @intCast((y + dy) * cols + x + dx));
                        if (at.found_existing) continue;
                        at.value_ptr.* = @intCast(cells.items.len);
                        try cells.append(gpa, at.key_ptr.*);
                    }
                }
            }
            first = end;
            try self.nl_counts.append(gpa, @intCast(cells.items.len));
        }
        const r: i64 = nlpar_mod.Window.reachOf(radius, passes);
        const side: usize = @intCast(2 * r + 1);
        try self.nl_nbr.resize(gpa, cells.items.len * side * side);
        for (cells.items, 0..) |c, k| {
            const y: i64 = @intCast(c / i.cols);
            const x: i64 = @intCast(c % i.cols);
            var dy: i64 = -r;
            while (dy <= r) : (dy += 1) {
                var dx: i64 = -r;
                while (dx <= r) : (dx += 1) {
                    const slot: usize = @intCast((dy + r) * @as(i64, @intCast(side)) + dx + r);
                    const inside = y + dy >= 0 and y + dy < rows and x + dx >= 0 and x + dx < cols;
                    self.nl_nbr.items[k * side * side + slot] = if (inside) self.nl_index.get(@intCast((y + dy) * cols + x + dx)) orelse nlpar_mod.NONE else nlpar_mod.NONE;
                }
            }
        }
        return .{ .nbr = self.nl_nbr.items, .counts = self.nl_counts.items, .radius = radius, .passes = passes };
    }

    /// Record: the window's patterns (their stored bytes in `bytes`, in the window's order)
    /// at a level, averaged, the wanted ones into the level's pattern buffer from pattern
    /// `out0` on.
    fn averageLevel(self: *Engine, lv: Level, bytes: []const u8, win: nlpar_mod.Window, out0: u32) !void {
        const g = &self.g;
        const i = try self.info();
        const k = @intFromEnum(lv);
        const hw = try self.patternShape(lv);
        const px = hw[0] * hw[1];
        const n_win: u32 = @intCast(self.nl_cells.items.len);
        g.ensure(&self.nl.a, @as(u64, n_win) * px * 4);
        if (self.binOf(lv) == 1) {
            g.ensure(&self.raw[k], bytes.len);
            g.upload(self.raw[k], 0, bytes);
            self.prep.load(self.raw[k], n_win, hw[0], hw[1], @intFromEnum(i.pixel), self.nl.a);
        } else {
            try self.reduceLevel(lv, self.nl_cells.items, bytes);
            g.writeSlice(self.nl.a, 0, f32, self.red_host.items);
        }
        self.nl.average(win, px, self.set.nlpar_lambda, self.pats[k], out0);
    }

    /// What the averaging did to the cells last read back with `account`: the stored patterns'
    /// noise (standard deviation a pixel), the share of it the averages keep, and the share of
    /// an average that is the pattern itself, each a mean over those cells.
    pub const NlparSummary = struct { noise: f64, left: f64, own: f64 };

    fn nlparSummary(self: *const Engine) NlparSummary {
        const n = self.nl_stats_n;
        const f = self.nl_shares.len / @max(n, 1);
        var noise: f64 = 0;
        var kept: f64 = 0;
        var own: f64 = 0;
        var counted: f64 = 0;
        for (0..n) |c| {
            const before: f64 = self.nl_noise[c];
            // an average's noise: Σ share² · the noise of the pattern it is a share of
            var after: f64 = 0;
            for (self.nl_shares[c * f ..][0..f], self.nl_where[c * f ..][0..f]) |a, at| {
                if (at != nlpar_mod.NONE) after += @as(f64, a) * a * self.nl_noise[at];
            }
            own += self.nl_shares[c * f + f / 2];
            if (!(before > 0)) continue;
            noise += @sqrt(before);
            kept += @sqrt(after / before);
            counted += 1;
        }
        return .{ .noise = noise / @max(counted, 1), .left = kept / @max(counted, 1), .own = own / @as(f64, @floatFromInt(@max(n, 1))) };
    }

    /// Log what the averaging did to the first batch of an indexing run (read with its
    /// results).
    fn reportNlpar(self: *Engine) void {
        if (self.nl_report != .queued) return;
        self.nl_report = .off;
        const a = self.nlparSummary();
        const s = &self.set;
        self.events.log("neighbour averaging: {d} pass{s} over {d} × {d} cells (λ {d:.2}); noise about {d:.2} a pixel, {d:.0} % of it left; an average is {d:.0} % the pattern itself", .{
            s.nlpar_iterations,     if (s.nlpar_iterations == 1) "" else "es",
            2 * s.nlpar_radius + 1, 2 * s.nlpar_radius + 1,
            s.nlpar_lambda,         a.noise,
            100 * a.left,           100 * a.own,
        });
    }

    /// About `want` cells on a lattice over the whole map, each at the centre of its share
    /// (every cell when the map has no more), ascending. Caller frees.
    pub fn lattice(self: *const Engine, want: u32) ![]u64 {
        const i = try self.info();
        const rows: u64 = i.rows;
        const cols: u64 = i.cols;
        var gy = rows;
        var gx = cols;
        if (rows * cols > want) {
            const ratio = @as(f64, @floatFromInt(rows)) / @as(f64, @floatFromInt(cols));
            gy = std.math.clamp(@as(u64, @intFromFloat(@round(@sqrt(@as(f64, @floatFromInt(want)) * ratio)))), 1, rows);
            gx = std.math.clamp(want / gy, 1, cols);
        }
        const cells = try self.gpa.alloc(u64, @intCast(gy * gx));
        for (0..@intCast(gy)) |r| for (0..@intCast(gx)) |c| {
            cells[r * @as(usize, @intCast(gx)) + c] = ((2 * r + 1) * rows / (2 * gy)) * cols + (2 * c + 1) * cols / (2 * gx);
        };
        return cells;
    }

    /// The average pattern at stored size, over settings.static_patterns patterns of a lattice.
    fn ensureAverage(self: *Engine) !void {
        if (self.avg.len != 0 and self.avg_n == self.set.static_patterns) return;
        const gpa = self.gpa;
        const i = try self.info();
        const px = @as(usize, i.pat_h) * i.pat_w;
        const bpp = bytesPerPixel(i.pixel);
        const cells = try self.lattice(self.set.static_patterns);
        defer gpa.free(cells);
        const sum = try gpa.alloc(f64, px);
        defer gpa.free(sum);
        @memset(sum, 0);
        var done: usize = 0;
        while (done < cells.len) {
            const n = @min(cells.len - done, 256);
            const bytes = try self.readCells(cells[done..][0..n], false);
            for (0..n) |k| {
                const pat = bytes[k * px * bpp ..][0 .. px * bpp];
                switch (i.pixel) {
                    .u8 => for (sum, pat) |*a, v| {
                        a.* += @floatFromInt(v);
                    },
                    .u16 => for (sum, 0..) |*a, j| {
                        a.* += @floatFromInt(std.mem.readInt(u16, pat[2 * j ..][0..2], .little));
                    },
                    .f32 => for (sum, 0..) |*a, j| {
                        a.* += @as(f32, @bitCast(std.mem.readInt(u32, pat[4 * j ..][0..4], .little)));
                    },
                }
            }
            done += n;
            if (cells.len > 256) self.events.emit(.{ .kind = "progress", .stage = "background", .done = done, .total = cells.len });
        }
        if (self.avg.len != px) {
            gpa.free(self.avg);
            self.avg = &.{};
            self.avg = try gpa.alloc(f32, px);
        }
        const k = 1.0 / @as(f64, @floatFromInt(cells.len));
        for (self.avg, sum) |*a, v| a.* = @floatCast(v * k);
        self.avg_n = self.set.static_patterns;
        self.bg_key = .{ null, null };
        self.events.log("static background: the average of {d} patterns over the map", .{cells.len});
    }

    /// The static background of a level on the device (null: none is subtracted).
    fn staticBackground(self: *Engine, lv: Level) !?Buf {
        const kind = self.set.static_background;
        var src: []const f32 = &.{};
        switch (kind) {
            .none => return null,
            .average => {
                try self.ensureAverage();
                src = self.avg;
            },
            .file => {
                src = switch (self.source) {
                    .file => |s| s.background,
                    else => &.{},
                };
                if (src.len == 0) {
                    if (!self.told_no_background) self.events.log("static background: this scan stores none, so none is subtracted (static_background: \"average\" takes the patterns' own)", .{});
                    self.told_no_background = true;
                    return null;
                }
            },
        }
        const i = try self.info();
        const k = @intFromEnum(lv);
        const key: BgKey = .{ .kind = kind, .n = if (kind == .average) self.avg_n else 0, .bin = self.binOf(lv) };
        if (self.bg_key[k] == null or !std.meta.eql(self.bg_key[k].?, key)) {
            const hw = try self.patternShape(lv);
            const out = try self.gpa.alloc(f32, @as(usize, hw[0]) * hw[1]);
            defer self.gpa.free(out);
            try reduce.patterns(self.gpa, std.mem.sliceAsBytes(src), 1, i.pat_h, i.pat_w, 4, key.bin, out);
            self.g.ensure(&self.bg_gpu[k], out.len * 4);
            self.g.writeSlice(self.bg_gpu[k], 0, f32, out);
            self.bg_key[k] = key;
        }
        return self.bg_gpu[k];
    }

    /// The pattern filters at a level; the band pass keeps its size on the detector.
    pub fn prepOptions(self: *const Engine, lv: Level, hw: [2]u32, static: ?Buf) prep_mod.Options {
        const s = &self.set;
        const f = self.binOf(lv);
        return .{
            .static = static,
            .dynamic = s.dynamic_background,
            .dynamic_sigma = s.dynamic_sigma * @as(f64, @floatFromInt(hw[1])),
            .clahe = s.clahe,
            .clahe_grid = s.clahe_grid,
            .clahe_bins = s.clahe_bins,
            .clahe_clip = s.clahe_clip,
            .bandpass = s.bandpass,
            .bp_fine = s.bp_fine / f,
            .bp_coarse = s.bp_coarse / f,
            .mask = self.maskOf(hw[0], hw[1]),
        };
    }

    fn maskOf(self: *const Engine, h: u32, w: u32) [3]f32 {
        if (!self.set.mask) return .{ 0, 0, 0 };
        const r = self.set.mask_radius * 0.5 * @as(f64, @floatFromInt(@min(h, w)));
        return .{ 0.5 * @as(f32, @floatFromInt(h - 1)), 0.5 * @as(f32, @floatFromInt(w - 1)), @floatCast(r * r) };
    }

    /// Record: unpack `n` patterns from the host bytes last read (as stored, or reduced here
    /// for a reduced level) into the level's pattern buffer, filtered when `opts` is given.
    fn uploadLevel(self: *Engine, lv: Level, cells: []const u64, bytes: []const u8, opts: ?prep_mod.Options) !void {
        const g = &self.g;
        const i = try self.info();
        const k = @intFromEnum(lv);
        const hw = try self.patternShape(lv);
        const px = @as(usize, hw[0]) * hw[1];
        const n: u32 = @intCast(cells.len);
        const o = opts orelse prep_mod.Options{ .normalize = false };
        g.ensure(&self.pats[k], @as(u64, n) * px * 4);
        // (uploads, not writes: made while the device may still work on the batch before)
        if (self.binOf(lv) == 1) {
            g.ensure(&self.raw[k], bytes.len);
            g.upload(self.raw[k], 0, bytes);
            self.prep.run(self.raw[k], n, hw[0], hw[1], @intFromEnum(i.pixel), o, self.pats[k]);
        } else {
            try self.reduceLevel(lv, cells, bytes);
            g.ensure(&self.raw[k], self.red_host.items.len * 4);
            g.upload(self.raw[k], 0, std.mem.sliceAsBytes(self.red_host.items));
            self.prep.run(self.raw[k], n, hw[0], hw[1], 2, o, self.pats[k]);
        }
    }

    /// Record: read the cells once and leave them, filtered, in the pattern buffer of each of
    /// the given levels.
    pub fn uploadCells(self: *Engine, cells: []const u64, levels: []const Level) !void {
        return self.uploadCellsAs(cells, levels, .{});
    }

    const Upload = struct {
        /// false: averaged with their neighbours (when that is on) and nothing more
        filter: bool = true,
        /// also read back what the averaging did (`nlparSummary` after the wait)
        account: bool = false,
    };

    fn uploadCellsAs(self: *Engine, cells: []const u64, levels: []const Level, how: Upload) !void {
        var need = [2]bool{ false, false };
        for (levels) |lv| need[@intFromEnum(lv)] = true;
        // the backgrounds first: averaging reads patterns through the same host buffer
        var bg: [2]?Buf = .{ null, null };
        for (0..2) |k| {
            if (need[k] and how.filter) bg[k] = try self.staticBackground(@enumFromInt(k));
        }
        if (!self.set.nlpar) {
            const bytes = try self.readCells(cells, true);
            for (0..2) |k| {
                if (!need[k]) continue;
                const lv: Level = @enumFromInt(k);
                try self.uploadLevel(lv, cells, bytes, if (how.filter) self.prepOptions(lv, try self.patternShape(lv), bg[k]) else null);
            }
            return;
        }
        // averaged with their neighbours first: the cells' windows are read, averaged on the
        // device, and the filters then run on the averages
        const n: u32 = @intCast(cells.len);
        var px_max: u64 = 1;
        for (0..2) |k| {
            if (!need[k]) continue;
            const hw = try self.patternShape(@enumFromInt(k));
            const px = @as(u64, hw[0]) * hw[1];
            self.g.ensure(&self.pats[k], n * px * 4);
            px_max = @max(px_max, px);
        }
        const max_win: usize = @intCast(@max(1, nlpar_bytes / (px_max * 4)));
        var t0: u32 = 0;
        // How many cells to try at once: all of them first, and from then on twice what the
        // last part held. (Trying all that remain every time builds their whole window again
        // for every part: for cells far apart, each with a window of its own, that was most
        // of the time an upload took, on the host.)
        var guess: u32 = n;
        while (t0 < n) {
            // as many of the cells at once as their window allows
            var m: u32 = @min(n - t0, guess);
            var win = try self.nlparWindow(cells[t0..][0..m]);
            while (m > 1 and self.nl_cells.items.len > max_win) {
                m = (m + 1) / 2;
                win = try self.nlparWindow(cells[t0..][0..m]);
            }
            guess = m *| 2;
            const bytes = try self.readCells(self.nl_cells.items, true);
            var reported = false;
            for (0..2) |k| {
                if (!need[k]) continue;
                try self.averageLevel(@enumFromInt(k), bytes, win, t0);
                if (how.account and t0 == 0 and !reported) {
                    reported = true;
                    const gpa = self.gpa;
                    const f = @as(usize, win.side()) * win.side();
                    gpa.free(self.nl_noise);
                    self.nl_noise = &.{};
                    self.nl_noise = try gpa.alloc(f32, self.nl_cells.items.len);
                    gpa.free(self.nl_shares);
                    self.nl_shares = &.{};
                    self.nl_shares = try gpa.alloc(f32, m * f);
                    gpa.free(self.nl_where);
                    self.nl_where = &.{};
                    self.nl_where = try gpa.dupe(u32, win.nbr[0 .. m * f]);
                    self.nl_stats_n = m;
                    self.g.read(self.nl.sig, 0, std.mem.sliceAsBytes(self.nl_noise));
                    self.g.read(self.nl.final, 0, std.mem.sliceAsBytes(self.nl_shares));
                }
            }
            t0 += m;
        }
        if (!how.filter) return;
        for (0..2) |k| {
            if (!need[k]) continue;
            const lv: Level = @enumFromInt(k);
            const hw = try self.patternShape(lv);
            self.prep.filter(self.pats[k], n, hw[0], hw[1], self.prepOptions(lv, hw, bg[k]));
        }
    }

    // ── geometry on the device ──

    pub fn frame(self: *const Engine, lv: Level) !spherical.Frame {
        const hw = try self.patternShape(lv);
        const R = self.calibration.rotation(&self.geo);
        return .{
            .c0 = .{ @floatCast(R[0]), @floatCast(R[3]), @floatCast(R[6]) },
            .c1 = .{ @floatCast(R[1]), @floatCast(R[4]), @floatCast(R[7]) },
            .c2 = .{ @floatCast(R[2]), @floatCast(R[5]), @floatCast(R[8]) },
            .pat_h = hw[0],
            .pat_w = hw[1],
            .mask = self.maskOf(hw[0], hw[1]),
        };
    }

    /// The detector's position under `geo` in a level's pixels (the reference cell's).
    pub fn translationOf(self: *const Engine, geo: *const Geometry, lv: Level) !rot.Vec3 {
        const sz = try self.sizeOf(lv);
        return self.calibration.translation(geo, sz[0], sz[1]);
    }

    /// Beam offset of a cell from the map's reference (its centre), in a level's pixels: by
    /// the scan's steps, or (with a plane of source points) what the plane says.
    pub fn offsetOf(self: *const Engine, geo: *const Geometry, cell: u64, lv: Level) rot.Vec3 {
        if (geo.plane_kind != 0) {
            const t0 = self.translationOf(geo, lv) catch return .{ 0, 0, 0 };
            return rot.sub(t0, self.placeOf(geo, cell, lv) catch return .{ 0, 0, 0 });
        }
        const i = self.info() catch return .{ 0, 0, 0 };
        const row: f64 = @floatFromInt(cell / i.cols);
        const col: f64 = @floatFromInt(cell % i.cols);
        const signs = self.scanSigns();
        return geo.scanOffset(row, col, 0.5 * @as(f64, @floatFromInt(i.rows - 1)), 0.5 * @as(f64, @floatFromInt(i.cols - 1)), signs[0], signs[1], self.binOf(lv));
    }

    /// The detector's position for a cell under `geo`, in a level's pixels.
    pub fn placeOf(self: *const Engine, geo: *const Geometry, cell: u64, lv: Level) !rot.Vec3 {
        return self.placeWithSigns(geo, cell, lv, self.scanSigns());
    }

    fn placeWithSigns(self: *const Engine, geo: *const Geometry, cell: u64, lv: Level, signs: [2]i32) !rot.Vec3 {
        const i = try self.info();
        const sz = try self.sizeOf(lv);
        const row: f64 = @floatFromInt(cell / i.cols);
        const col: f64 = @floatFromInt(cell % i.cols);
        return geo.positionForRotation(self.calibration.rotation(geo), row, col, 0.5 * @as(f64, @floatFromInt(i.rows - 1)), 0.5 * @as(f64, @floatFromInt(i.cols - 1)), signs[0], signs[1], self.binOf(lv), sz[0], sz[1]);
    }

    pub fn positionOf(self: *const Engine, geo: *const Geometry, cell: u64, lv: Level) ![3]f32 {
        const t = try self.placeOf(geo, cell, lv);
        return .{ @floatCast(t[0]), @floatCast(t[1]), @floatCast(t[2]) };
    }

    // ── phases ──

    pub fn addPhase(self: *Engine, path: []const u8) !u32 {
        var m = try master_mod.load(self.gpa, path);
        errdefer m.deinit();
        return self.addMaster(m);
    }

    /// Add a candidate phase whose master is already loaded (the engine owns it from here).
    pub fn addMaster(self: *Engine, m: master_mod.Master) !u32 {
        try self.peaks.ensureUnusedCapacity(self.gpa, 1);
        for (self.bands.items) |b| try b.ph.ensureUnusedCapacity(self.gpa, 1);
        try self.phases.append(self.gpa, .{ .master = m });
        self.peaks.appendAssumeCapacity(.{});
        for (self.bands.items) |b| b.ph.appendAssumeCapacity(null);
        self.events.log("phase {d}: {s} ({s}, {d} px, point group {s})", .{ self.phases.items.len - 1, m.name, m.source, m.side, if (m.point_group >= 1 and m.point_group <= 32) sym.point_group_names[m.point_group - 1] else "?" });
        self.res.clear();
        self.phase_generation += 1;
        self.chance_key = 0;
        // the symmetry it obeys, as measured, and where its axes are when not the tables' own
        const st = m.setting;
        const z = rot.qapply(st.frame, .{ 0, 0, 1 });
        const x = rot.qapply(st.frame, .{ 1, 0, 0 });
        if (rot.angleBetween(st.frame, rot.quat_identity) < 1e-6)
            self.events.log("phase {d}: orientations reduced by the rotations {s} (Laue class {s}) of the master, {d} of its point-group operators holding", .{ self.phases.items.len - 1, st.rotation.rotationName(), st.laue.name(), m.ops.len })
        else
            self.events.log("phase {d}: orientations reduced by the rotations {s} (Laue class {s}) of the master, {d} of its point-group operators holding; its unique axis along ({d:.3}, {d:.3}, {d:.3}), its x along ({d:.3}, {d:.3}, {d:.3})", .{ self.phases.items.len - 1, st.rotation.rotationName(), st.laue.name(), m.ops.len, z[0], z[1], z[2], x[0], x[1], x[2] });
        return @intCast(self.phases.items.len - 1);
    }

    pub fn clearPhases(self: *Engine) void {
        for (self.bands.items) |b| {
            for (0..b.ph.items.len) |pi| b.dropPhase(&self.g, pi);
            b.ph.clearRetainingCapacity();
        }
        for (self.phases.items) |*p| p.deinit(&self.g);
        self.phases.clearRetainingCapacity();
        for (self.peaks.items) |*b| self.g.release(b);
        self.peaks.clearRetainingCapacity();
        self.clearFit();
        self.res.clear();
        self.phase_generation += 1;
        self.chance_key = 0;
    }

    pub fn phaseInfo(self: *Engine, index: u32) !PhaseInfo {
        if (index >= self.phases.items.len) return error.NoSuchPhase;
        const p = &self.phases.items[index];
        var out: PhaseInfo = .{ .side = p.master.side, .point_group = p.master.point_group, .space_group = p.master.space_group, .laue = @intFromEnum(p.master.laue), .voltage_kv = p.master.voltage_kv };
        // the selection rules as the band last used found them
        var newest: u64 = 0;
        for (self.bands.items) |b| {
            if (b.stamp < newest) continue;
            if (b.ph.items[index]) |x| {
                newest = b.stamp;
                out.n_fold = x.hm.rules.n_fold;
                out.mirror = @intFromBool(x.hm.rules.mirror);
                out.inversion = @intFromBool(x.hm.rules.inversion);
            }
        }
        const n = @min(p.master.name.len, out.name.len - 1);
        @memcpy(out.name[0..n], p.master.name[0..n]);
        return out;
    }

    /// What a phase's master must be filtered for at a level: the patterns' own filters, at
    /// the sizes they have on the sphere when a pattern pixel has its angular width.
    fn bakeKeyOf(self: *const Engine, p: *const Phase, lv: Level) BakeKey {
        const s = &self.set;
        var key: BakeKey = .{ .px_q = std.math.minInt(i32), .smooth = s.master_smooth, .bandpass = s.bandpass, .fine = 0, .coarse = 0, .clahe = s.clahe, .tiles = 0, .bins = 0, .clip = 0 };
        const angle = self.pixelAngle(lv);
        const hw = self.patternShape(lv) catch return key;
        if (!(angle > 0)) return key;
        // the sheet spreads 2π steradians over side² pixels
        const px = angle * @as(f64, @floatFromInt(p.master.side)) / @sqrt(2.0 * std.math.pi);
        key.px_q = @intFromFloat(@round(@log(px) / @log(1.05)));
        const f = self.binOf(lv);
        if (s.bandpass) {
            key.fine = s.bp_fine / f;
            key.coarse = s.bp_coarse / f;
        }
        if (s.clahe) {
            // tiles as wide on the sphere as the patterns' are: pattern height / grid pixels
            const tile = @as(f64, @floatFromInt(hw[0])) / @as(f64, @floatFromInt(s.clahe_grid)) * px;
            key.tiles = @intFromFloat(std.math.clamp(@round(@as(f64, @floatFromInt(p.master.side)) / tile), 1, 64));
            key.bins = s.clahe_bins;
            key.clip = s.clahe_clip;
        }
        return key;
    }

    /// Bring every phase's filtered master up to date with the settings and the geometry, at
    /// each of the given levels.
    pub fn prepareMasters(self: *Engine, levels: []const Level) !void {
        if (self.phases.items.len == 0) return error.NoPhases;
        for (self.phases.items) |*p| for (levels) |lv| {
            const k = @intFromEnum(lv);
            const key = self.bakeKeyOf(p, lv);
            if (p.baked[k] != null and std.meta.eql(p.baked[k].?, key)) continue;
            try self.bakeMaster(p, lv, key);
            self.prep.gradient(p.det[k].buf, p.master.side, &p.det[k].grad);
            p.baked[k] = key;
            p.serial[k] +%= 1;
        };
    }

    /// Filter the master's sheets for a level (on the device), normalize them together and
    /// keep them on both sides. The order is the patterns': what a pixel averages (smoothing),
    /// CLAHE, band pass.
    fn bakeMaster(self: *Engine, p: *Phase, lv: Level, key: BakeKey) !void {
        const g = &self.g;
        const gpa = self.gpa;
        const k = @intFromEnum(lv);
        const m = &p.master;
        const side = m.side;
        const n = m.raw.len;
        if (p.work[k].len != n) {
            gpa.free(p.work[k]);
            p.work[k] = &.{};
            p.work[k] = try gpa.alloc(f32, n);
        }
        const work = p.work[k];
        p.bake_mean[k] = 0;
        p.bake_sd[k] = 1;
        @memcpy(work, m.raw);
        g.ensure(&p.det[k].buf, n * 4);
        p.det[k].side = side;
        const buf = p.det[k].buf;
        g.writeSlice(buf, 0, f32, m.raw);
        if (key.px_q == std.math.minInt(i32)) return;
        const px = std.math.pow(f64, 1.05, @floatFromInt(key.px_q));
        const smooth = key.smooth * px;
        if (!(smooth > 0.05 or key.bandpass or key.clahe)) return;
        var fine = key.fine * px;
        if (key.clahe) {
            self.prep.sphereBlur(buf, side, smooth);
            self.prep.filter(buf, 2, side, side, .{ .clahe = true, .clahe_grid = key.tiles, .clahe_bins = key.bins, .clahe_clip = key.clip, .normalize = false });
            // CLAHE works on the sheets as images, which the crystal's symmetry does not
            // respect: average the result over the master's own point group
            if (m.ops.len > 1) {
                const rows = try gpa.alloc([4]f32, m.ops.len * 3);
                defer gpa.free(rows);
                for (m.ops, 0..) |op, j| for (0..3) |r| {
                    rows[j * 3 + r] = .{ @floatCast(op[r * 3]), @floatCast(op[r * 3 + 1]), @floatCast(op[r * 3 + 2]), 0 };
                };
                var ops = g.storage(rows.len * 16);
                defer g.release(&ops);
                g.writeSlice(ops, 0, [4]f32, rows);
                self.prep.symmetrize(buf, side, ops, @intCast(m.ops.len));
            }
        } else {
            // what the pixel averages and the band pass's fine Gaussian compose into one
            fine = @sqrt(smooth * smooth + fine * fine);
            if (!key.bandpass) self.prep.sphereBlur(buf, side, smooth);
        }
        if (key.bandpass) {
            // the difference of two smoothings on the sphere
            var low = g.storage(n * 4);
            defer g.release(&low);
            g.copy(buf, 0, low, 0, n * 4);
            self.prep.sphereBlur(buf, side, fine);
            self.prep.sphereBlur(low, side, key.coarse * px);
            const lo_h = try gpa.alloc(f32, n);
            defer gpa.free(lo_h);
            g.read(buf, 0, std.mem.sliceAsBytes(work));
            g.read(low, 0, std.mem.sliceAsBytes(lo_h));
            try g.wait();
            for (work, lo_h) |*v, l| v.* -= l;
        } else {
            g.read(buf, 0, std.mem.sliceAsBytes(work));
            try g.wait();
        }
        // one scale for both sheets again
        var sum: f64 = 0;
        for (work) |v| sum += v;
        const mean = sum / @as(f64, @floatFromInt(n));
        var ss: f64 = 0;
        for (work) |v| ss += (v - mean) * (v - mean);
        const scale = 1.0 / @max(@sqrt(ss / @as(f64, @floatFromInt(n))), 1e-30);
        p.bake_mean[k] = mean;
        p.bake_sd[k] = 1.0 / scale;
        for (work) |*v| v.* = @floatCast((v.* - mean) * scale);
        g.writeSlice(buf, 0, f32, work);
    }

    /// Spherical indexing at bandwidth L against every phase's master as filtered for `lv`
    /// (built on first use, kept for the next). The work buffers (`self.sp`) are left pointing
    /// at this band's tables.
    pub fn band(self: *Engine, L: u32, lv: Level) !*Band {
        const g = &self.g;
        const gpa = self.gpa;
        try self.prepareMasters(&.{lv});
        var found: ?*Band = null;
        for (self.bands.items) |b| {
            if (b.L == L) found = b;
        }
        if (found == null) {
            if (self.bands.items.len >= MAX_BANDS) {
                var oldest: usize = 0;
                for (self.bands.items, 0..) |b, i| {
                    if (b.stamp < self.bands.items[oldest].stamp) oldest = i;
                }
                self.bands.swapRemove(oldest).destroy(g);
            }
            try self.bands.ensureUnusedCapacity(gpa, 1);
            const b = try gpa.create(Band);
            errdefer gpa.destroy(b);
            b.* = .{ .L = L, .tables = try spherical.Tables.init(g, L) };
            self.bands.appendAssumeCapacity(b);
            found = b;
        }
        const b = found.?;
        self.band_clock += 1;
        b.stamp = self.band_clock;
        self.sp.t = &b.tables;
        self.sp.wg = self.set.wg_fft;
        while (b.ph.items.len < self.phases.items.len) try b.ph.append(gpa, null);
        const k = @intFromEnum(lv);
        for (self.phases.items, 0..) |*p, pi| {
            if (b.ph.items[pi]) |x| {
                if (x.level == lv and x.serial == p.serial[k]) continue;
                b.dropPhase(g, pi);
            }
            var hm = try master_mod.Harmonics.init(gpa, &p.master, p.work[k], L);
            errdefer hm.deinit(gpa);
            const gp = try spherical.PhaseGpu.init(g, &b.tables, &hm);
            b.ph.items[pi] = .{ .hm = hm, .gpu = gp, .level = lv, .serial = p.serial[k] };
            const r = hm.rules;
            self.events.log("phase {d}: bandwidth {d}, rotation order {d} about z{s}{s}", .{ pi, L, r.n_fold, if (r.mirror) ", mirror" else "", if (r.inversion) ", inversion" else "" });
        }
        return b;
    }

    /// Record (when stale): each phase's denominator volume for the reference geometry.
    fn prepareDenominators(self: *Engine, b: *Band, lv: Level, fr: spherical.Frame) !void {
        const t64 = try self.translationOf(&self.geo, lv);
        const t: [3]f32 = .{ @floatCast(t64[0]), @floatCast(t64[1]), @floatCast(t64[2]) };
        const key: RdenKey = .{ .c = .{ fr.c0[0], fr.c0[1], fr.c0[2], fr.c1[0], fr.c1[1], fr.c1[2], fr.c2[0], fr.c2[1], fr.c2[2] }, .t = t, .h = fr.pat_h, .w = fr.pat_w, .mask = fr.mask };
        for (b.ph.items) |*slot| {
            const x = &slot.*.?;
            if (x.rden_key != null and std.meta.eql(x.rden_key.?, key)) continue;
            self.sp.denominators(&x.gpu, &.{.{ t[0], t[1], t[2], 0 }}, fr, &x.gpu.rden);
            x.rden_key = key;
        }
    }

    /// The phase's dictionary: points of a cubochoric grid of N per half edge whose
    /// orientation is its own fundamental-zone representative.
    fn prepareDictionary(self: *Engine, p: *Phase, N: u32) !void {
        const gpa = self.gpa;
        if (p.dict_grid == N and p.dict.len != 0) return;
        if (p.takeDictionary(&self.g, N)) return;
        var list: std.ArrayList(f32) = .empty;
        errdefer list.deinit(gpa);
        const ops = p.master.rotations();
        const a = rot.cubochoric_half_edge;
        const nf: f64 = @floatFromInt(N);
        var i: i64 = -@as(i64, N);
        while (i <= N) : (i += 1) {
            var j: i64 = -@as(i64, N);
            while (j <= N) : (j += 1) {
                var k: i64 = -@as(i64, N);
                while (k <= N) : (k += 1) {
                    const q = rot.qstd(rot.fromCubochoric(.{ a * @as(f64, @floatFromInt(i)) / nf, a * @as(f64, @floatFromInt(j)) / nf, a * @as(f64, @floatFromInt(k)) / nf }));
                    var in_fz = true;
                    for (ops[1..]) |g| {
                        if (@abs(rot.qmul(g, q)[0]) > q[0] + 1e-9) {
                            in_fz = false;
                            break;
                        }
                    }
                    if (!in_fz) continue;
                    for (q) |c| try list.append(gpa, @floatCast(c));
                }
            }
        }
        p.dict = try list.toOwnedSlice(gpa);
        p.dict_grid = N;
        // in a shuffled order (the same every time): a run of grid neighbours is no sample of
        // the dictionary, and Oja's rule learns from one run at a time
        var prng = std.Random.DefaultPrng.init(0x6469_6374);
        const r = prng.random();
        var left = p.dict.len / 4;
        while (left > 1) : (left -= 1) {
            const j = r.uintLessThan(usize, left);
            const hi = p.dict[4 * (left - 1) ..][0..4];
            const lo = p.dict[4 * j ..][0..4];
            const tmp = hi.*;
            hi.* = lo.*;
            lo.* = tmp;
        }
        self.g.ensure(&p.dict_gpu, p.dict.len * 4);
        self.g.writeSlice(p.dict_gpu, 0, f32, p.dict);
        self.events.log("dictionary: {d} orientations for the rotations {s} (grid {d})", .{ p.dict.len / 4, p.master.rotation.rotationName(), N });
    }

    /// The phase's dictionary as its leading components: learned by Oja's rule while the
    /// dictionary is simulated (at the reference detector position `t`, the search's level),
    /// then every entry's coefficients. Kept until the dictionary, the geometry, the filters
    /// or the settings change.
    fn preparePca(self: *Engine, p: *Phase, N: u32, lv: Level, fr: spherical.Frame, t: [3]f32) !void {
        const g = &self.g;
        const s = &self.set;
        try self.prepareDictionary(p, N);
        const total: u32 = @intCast(p.dict.len / 4);
        const px: u32 = fr.pat_h * fr.pat_w;
        const k: u32 = @min(s.pca_components, @min(px, total));
        const key: PcaKey = .{ .grid = N, .k = k, .epochs = s.pca_epochs, .level = lv, .serial = p.serial[@intFromEnum(lv)], .h = fr.pat_h, .w = fr.pat_w, .mask = fr.mask, .t = t, .c = .{ fr.c0[0], fr.c0[1], fr.c0[2], fr.c1[0], fr.c1[1], fr.c1[2], fr.c2[0], fr.c2[1], fr.c2[2] } };
        if (p.pca_key != null and std.meta.eql(p.pca_key.?, key)) return;
        p.pca_key = null;
        if (@as(u64, total) * k * 4 > @min(g.lim.max_storage_binding, g.lim.max_buffer)) return error.DictionaryTooLargeForPca;
        const m = p.det[@intFromEnum(lv)];
        // dictionary entries per step: about 64 MiB of simulated patterns
        const chunk: u32 = @min(total, @as(u32, @intCast(std.math.clamp((64 << 20) / (@as(u64, px) * 4), 256, 4096))));
        try self.pca.begin(&p.pca_w, k, px);
        var steps: u32 = 0;
        const all_steps = s.pca_epochs * ((total + chunk - 1) / chunk);
        for (0..s.pca_epochs) |_| {
            var d0: u32 = 0;
            while (d0 < total) : (d0 += chunk) {
                if (self.cancel.get()) return error.Cancelled;
                const nb = @min(chunk, total - d0);
                steps += 1;
                // a short tail is no sample: it would pull the basis toward its few patterns
                if (nb < chunk / 2 and d0 != 0) continue;
                self.det.dictSims(m, fr, t, p.dict_gpu, nb, d0);
                try self.pca.learn(p.pca_w, self.det.sims, nb, k, px);
                self.events.emit(.{ .kind = "progress", .stage = "pca", .done = steps, .total = all_steps });
            }
        }
        g.ensure(&p.pca_coef, @as(u64, total) * k * 4);
        var d0: u32 = 0;
        while (d0 < total) : (d0 += chunk) {
            const nb = @min(chunk, total - d0);
            self.det.dictSims(m, fr, t, p.dict_gpu, nb, d0);
            self.pca.project(p.pca_w, self.det.sims, nb, k, px, p.pca_coef, d0);
        }
        // how much of the dictionary the components hold: a unit pattern's coefficients have
        // squared norm equal to the share of it they keep
        const ns: usize = @min(total, 2048);
        const head = try self.gpa.alloc(f32, ns * k);
        defer self.gpa.free(head);
        g.read(p.pca_coef, 0, std.mem.sliceAsBytes(head));
        try g.wait();
        var kept: f64 = 0;
        for (head) |v| kept += @as(f64, v) * v;
        p.pca_key = key;
        self.events.log("dictionary components: {d} of {d} pixels hold {d:.1} % of {d} patterns' variance ({d} steps of Oja's rule)", .{ k, px, 100 * kept / @as(f64, @floatFromInt(ns)), total, self.pca.steps });
    }

    // ── the batch ──

    /// What a search runs with outside a sweep: the settings' method, bandwidth and dictionary.
    /// One cell indexed and simulated under another geometry than the one in use, to set two
    /// pattern centres side by side; the one in use stays as it is (kept in `geo_home` for the
    /// call). `out` gets a record per phase (the spherical correlation at `bandwidth`, then the
    /// refinement, as the fits index their patterns), `sim` the best phase's simulation at
    /// `lv`, the phase chosen as the phase_by setting says. Returns that phase.
    pub fn cellUnder(self: *Engine, cell: u64, geo: Geometry, bandwidth: u32, lv: Level, out: []Record, sim: []f32) !u32 {
        if (self.geo_home != null) return error.Busy;
        self.geo_home = self.geo;
        self.geo = geo;
        defer {
            self.geo = self.geo_home.?;
            self.geo_home = null;
        }
        try self.indexCells(&[_]u64{cell}, .{ .method = .spherical, .bandwidth = bandwidth, .dict_grid = self.set.dict_grid }, out);
        var p: usize = 0;
        for (out, 0..) |r, k| {
            const v = if (self.set.phase_by == .nmi) r.nmi else r.ncc;
            const b = if (self.set.phase_by == .nmi) out[p].nmi else out[p].ncc;
            if (v > b) p = k;
        }
        try self.simulate(@intCast(p), &[_]u64{cell}, &out[p].q, lv, sim);
        return @intCast(p);
    }

    pub fn searchOf(self: *const Engine) Search {
        return .{ .method = self.set.method, .bandwidth = self.set.bandwidth, .dict_grid = self.set.dict_grid };
    }

    /// Index the given cells: per phase, find each pattern's orientation (spherical correlation
    /// or the dictionary) at the search's level, refine it on the detector at the refinement's,
    /// score it (NCC, NMI) at the scores', and measure there what NMI chance gives. `out` gets
    /// one record per phase per cell ([phase][cell]). One readback.
    pub fn indexCells(self: *Engine, cells: []const u64, prm: Search, out: []Record) !void {
        const rec = try self.recordCells(cells, prm, out);
        // the device starts on this batch; the host reads the next one meanwhile
        if (self.ahead.len != 0) {
            self.g.submit();
            self.readAhead(self.ahead);
        }
        try self.finishCells(rec, cells.len, out);
    }

    /// A batch recorded: the ticket of its read, and the number its records carry back.
    const Recorded = struct { ticket: u64, stamp: u32 };

    fn nextBatchStamp(self: *Engine) u32 {
        // The record carries the stamp as f32, which holds each value exactly.
        self.batch_stamp = self.batch_stamp % (1 << 24) + 1;
        return self.batch_stamp;
    }

    /// Record indexCells through its readback. The caller keeps `out` alive until
    /// `finishCells` completes that read.
    fn recordCells(self: *Engine, cells: []const u64, prm: Search, out: []Record) !Recorded {
        const s = &self.set;
        const n_ph = self.phases.items.len;
        const n: u32 = @intCast(cells.len);
        if (n_ph == 0) return error.NoPhases;
        if (out.len != n_ph * cells.len) return error.BadLength;
        if (n == 0) return .{ .ticket = self.g.readTicket(), .stamp = 0 };
        const ls = self.levelOf(.search);
        const lr = self.levelOf(.refine);
        const lc = self.levelOf(.score);
        self.det.chunk_px = s.refine_pixels;
        self.det.wg = s.wg_refine;
        const spherical_method = prm.method == .spherical;
        var bd: ?*Band = null;
        if (spherical_method) bd = try self.band(prm.bandwidth, ls);
        try self.prepareMasters(&.{ ls, lr, lc });
        // a dictionary is simulated at one detector position: the map's reference
        const t64 = try self.translationOf(&self.geo, ls);
        const t_avg: [3]f32 = .{ @floatCast(t64[0]), @floatCast(t64[1]), @floatCast(t64[2]) };
        // (its components are learned before the batch is recorded: learning waits on the device)
        if (prm.method == .pca) {
            for (self.phases.items) |*p| try self.preparePca(p, prm.dict_grid, ls, try self.frame(ls), t_avg);
        }
        // (what chance gives is measured once for the scan, before this batch's patterns go up:
        // it uses the same buffers)
        try self.calibrate();
        const account = self.set.nlpar and self.nl_report == .wanted;
        try self.uploadCellsAs(cells, &.{ ls, lr, lc }, .{ .account = account });
        if (account) self.nl_report = .queued;

        return self.recordPrepared(cells, prm, bd, out);
    }

    fn indexPrepared(self: *Engine, cells: []const u64, prm: Search, bd: ?*Band, out: []Record) !void {
        const rec = try self.recordPrepared(cells, prm, bd, out);
        try self.waitForRecords(rec, out);
        for (out) |*r| r.z = 0;
    }

    /// Search/refine/score a batch already in pats at the three stage levels.
    /// Never uploads, preprocesses, calibrates chance, or assigns chance-z.
    /// Masters/band/PCA must already have been prepared by indexCells.
    fn recordPrepared(self: *Engine, cells: []const u64, prm: Search, bd: ?*Band, out: []Record) !Recorded {
        const g = &self.g;
        const gpa = self.gpa;
        const s = &self.set;
        const n_ph = self.phases.items.len;
        const n: u32 = @intCast(cells.len);
        const ls = self.levelOf(.search);
        const lr = self.levelOf(.refine);
        const lc = self.levelOf(.score);
        const spherical_method = prm.method == .spherical;
        const t64 = try self.translationOf(&self.geo, ls);
        const t_avg: [3]f32 = .{ @floatCast(t64[0]), @floatCast(t64[1]), @floatCast(t64[2]) };
        // the items at each level in use: the detector's position for each pattern, in pixels
        const items = try gpa.alloc(spherical.Item, n);
        defer gpa.free(items);
        var have = [2]bool{ false, false };
        for ([_]Level{ lr, lc, ls }) |lv| {
            const k = @intFromEnum(lv);
            if (have[k] and lv != ls) continue;
            for (cells, 0..) |c, b| {
                const t = try self.positionOf(&self.geo, c, lv);
                items[b] = .{ t[0], t[1], t[2], @floatFromInt(b) };
            }
            if (!have[k]) {
                g.ensure(&self.items_gpu[k], @as(u64, n) * 16);
                g.writeSlice(self.items_gpu[k], 0, spherical.Item, items);
            }
            have[k] = true;
        }
        // (the search's items were built last and stay on the host for its tiles)
        const fr_s = try self.frame(ls);
        const fr_r = try self.frame(lr);
        const fr_c = try self.frame(lc);
        g.ensure(&self.records, @as(u64, n_ph) * n * @sizeOf(Record));
        const refine: detector.RefineOptions = .{
            .iters = if (s.refine) s.refine_iters else 0,
            .max_step = @floatCast(std.math.degreesToRadians(s.refine_step)),
            .nmi_bins = s.nmi_bins,
            .stamp = self.nextBatchStamp(),
        };

        if (spherical_method) {
            const b = bd.?;
            const sp = &self.sp;
            try self.prepareDenominators(b, ls, fr_s);
            var tile: u32 = n;
            for (b.ph.items) |*x| tile = @min(tile, sp.tileItems(&x.*.?.gpu, @as(u64, s.tile_mb) << 20));
            for (self.peaks.items[0..n_ph]) |*pk| g.ensure(pk, @as(u64, n) * 32);
            var t0: u32 = 0;
            while (t0 < n) : (t0 += tile) {
                const nt = @min(tile, n - t0);
                sp.coefficients(self.pats[@intFromEnum(ls)], items[t0..][0..nt], fr_s, spherical.FLAG_CENTER | spherical.FLAG_NORM, &sp.alm, &sp.stats);
                for (b.ph.items, 0..) |*x, pi| {
                    sp.correlate(&x.*.?.gpu, nt, x.*.?.gpu.rden.at(8), nt);
                    g.copy(sp.peaks, 0, self.peaks.items[pi], @as(u64, t0) * 32, @as(u64, nt) * 32);
                }
            }
        }
        for (self.phases.items, 0..) |*p, pi| {
            const first: u32 = @intCast(pi * n);
            if (spherical_method) {
                self.det.seed(fr_r, self.items_gpu[@intFromEnum(lr)], self.peaks.items[pi], 0, n);
            } else if (prm.method == .pca) {
                const k = p.pca_key.?.k;
                const total: u32 = @intCast(p.dict.len / 4);
                g.ensure(&self.pca.y, @as(u64, n) * k * 4);
                self.pca.project(p.pca_w, self.pats[@intFromEnum(ls)], n, k, fr_s.pat_h * fr_s.pat_w, self.pca.y, 0);
                try self.det.dictBegin(n);
                self.pca.match(p.pca_coef.at(1), self.pca.y, n, k, total, 0, self.det.dict_best);
                self.det.seedDict(fr_r, self.items_gpu[@intFromEnum(lr)], p.dict_gpu, n);
            } else {
                // entries per chunk: about 64 MiB of simulated patterns
                const chunk: u32 = @intCast(std.math.clamp((64 << 20) / (@as(u64, fr_s.pat_h) * fr_s.pat_w * 4), 64, 16384));
                try self.prepareDictionary(p, prm.dict_grid);
                const total: u32 = @intCast(p.dict.len / 4);
                try self.det.dictBegin(n);
                var d0: u32 = 0;
                while (d0 < total) : (d0 += chunk) {
                    // (a submission a chunk: no one submission keeps a slow device busy for long)
                    if (d0 > 0) g.submit();
                    self.det.dictChunk(p.det[@intFromEnum(ls)], fr_s, t_avg, self.pats[@intFromEnum(ls)], p.dict_gpu, @min(chunk, total - d0), d0, n);
                }
                self.det.seedDict(fr_r, self.items_gpu[@intFromEnum(lr)], p.dict_gpu, n);
            }
            self.det.refine(p.det[@intFromEnum(lr)], fr_r, self.pats[@intFromEnum(lr)], n, refine, self.records, first);
            if (lc != lr) {
                // the refined orientations scored again where the scores are taken
                self.det.seed(fr_c, self.items_gpu[@intFromEnum(lc)], self.records, first, n);
                var score = refine;
                score.iters = 0;
                self.det.refine(p.det[@intFromEnum(lc)], fr_c, self.pats[@intFromEnum(lc)], n, score, self.records, first);
            }
        }
        g.read(self.records, 0, std.mem.sliceAsBytes(out));
        return .{ .ticket = g.readTicket(), .stamp = refine.stamp };
    }

    /// Wait for the read and reject any records the device left from another batch.
    fn waitForRecords(self: *Engine, rec: Recorded, out: []const Record) !void {
        try self.g.waitFor(rec.ticket);
        const stamp: f32 = @floatFromInt(rec.stamp);
        for (out) |r| {
            if (r.z != stamp) return error.DeviceSkippedBatch;
        }
    }

    /// Finish the batch and put each match's NMI on the scan's own scale.
    fn finishCells(self: *Engine, rec: Recorded, n: usize, out: []Record) !void {
        try self.waitForRecords(rec, out);
        self.reportNlpar();
        for (0..self.phases.items.len) |pi| {
            for (out[pi * n ..][0..n]) |*r| r.z = (r.nmi - self.chance_med[pi]) / self.chance_spread[pi];
        }
    }

    /// Overlap-private residual pool. Output is immutable [phase][rank][cell].
    /// Every valid pose is refined against the same signed normalized residual;
    /// missing ranks are compacted out before detector projection/refinement.
    pub fn indexPreparedCandidates(self: *Engine, cells: []const u64, prm: Search, bd: ?*Band, keep_n: u32, out: []Record) !void {
        if (keep_n == 1) return self.indexPrepared(cells, prm, bd, out);
        if (prm.method == .pca) return error.MultiCandidatePcaUnsupported;
        const g = &self.g;
        const n: u32 = @intCast(cells.len);
        const lv = self.levelOf(.search);
        const fr = try self.frame(lv);
        const t64 = try self.translationOf(&self.geo, lv);
        const t_avg: [3]f32 = .{ @floatCast(t64[0]), @floatCast(t64[1]), @floatCast(t64[2]) };
        if (out.len != self.phases.items.len * keep_n * n) return error.BadLength;
        @memset(out, .{ .q = @splat(0), .ncc = -2, .nmi = 0, .step = 0, .z = 0 });
        const items = try self.gpa.alloc(spherical.Item, n);
        defer self.gpa.free(items);
        // Keep at least the former per-rank capacity. Small requests can
        // refine several ranks at once within the detector scratch budget.
        const scratch_per_pose = @as(u64, detector.NS) * 1024 * @sizeOf(f32);
        const limit = @min(g.lim.max_storage_binding, g.lim.max_buffer, @as(u64, self.set.tile_mb) << 20);
        const capacity = @max(n, @as(u32, @intCast(@min(limit / scratch_per_pose, keep_n))));
        const poses = try self.gpa.alloc(detector.Pose, capacity);
        defer self.gpa.free(poses);
        const source_slots = try self.gpa.alloc(usize, capacity);
        defer self.gpa.free(source_slots);
        const refined = try self.gpa.alloc(Record, capacity);
        defer self.gpa.free(refined);
        const peaks = try self.gpa.alloc([8]f32, @as(usize, n) * keep_n);
        defer self.gpa.free(peaks);
        const scores = try self.gpa.alloc([2]u32, @as(usize, n) * keep_n);
        defer self.gpa.free(scores);
        var records = g.storage(@as(u64, capacity) * @sizeOf(Record));
        defer g.release(&records);
        for (cells, 0..) |cell, b| {
            const t = try self.positionOf(&self.geo, cell, lv);
            items[b] = .{ t[0], t[1], t[2], @floatFromInt(b) };
        }
        const options: detector.RefineOptions = .{ .iters = if (self.set.refine) self.set.refine_iters else 0, .max_step = @floatCast(std.math.degreesToRadians(self.set.refine_step)), .nmi_bins = self.set.nmi_bins };
        if (prm.method == .spherical) try self.prepareDenominators(bd.?, lv, fr);
        for (self.phases.items, 0..) |*phase, pi| {
            if (self.cancel.get()) return error.Cancelled;
            @memset(peaks, @splat(0));
            if (prm.method == .spherical) {
                if (comptime @hasDecl(spherical.Spherical, "candidatePeaks")) {
                    const ph = &bd.?.ph.items[pi].?.gpu;
                    const tile = @min(n, self.sp.tileItems(ph, @as(u64, self.set.tile_mb) << 20));
                    var first: u32 = 0;
                    while (first < n) : (first += tile) {
                        if (self.cancel.get()) return error.Cancelled;
                        const nt = @min(tile, n - first);
                        self.sp.coefficients(self.pats[@intFromEnum(lv)], items[first..][0..nt], fr, spherical.FLAG_CENTER | spherical.FLAG_NORM, &self.sp.alm, &self.sp.stats);
                        const tile_peaks = try self.gpa.alloc([8]f32, @as(usize, nt) * keep_n);
                        defer self.gpa.free(tile_peaks);
                        if (comptime @hasField(spherical.Spherical, "candidate_cancel")) self.sp.candidate_cancel = &self.cancel;
                        defer {
                            if (comptime @hasField(spherical.Spherical, "candidate_cancel")) self.sp.candidate_cancel = null;
                        }
                        try self.sp.candidatePeaks(ph, nt, ph.rden.at(8), nt, phase.master.rotations(), keep_n, tile_peaks);
                        for (0..keep_n) |rank| @memcpy(peaks[rank * n + first ..][0..nt], tile_peaks[rank * nt ..][0..nt]);
                    }
                } else return error.MultiCandidateSphericalUnavailable;
            } else {
                try self.prepareDictionary(phase, prm.dict_grid);
                const total: u32 = @intCast(phase.dict.len / 4);
                const chunk: u32 = @intCast(std.math.clamp((64 << 20) / (@as(u64, fr.pat_h) * fr.pat_w * 4), 64, 16384));
                try self.det.dictBeginCandidates(n, keep_n);
                var d0: u32 = 0;
                while (d0 < total) : (d0 += chunk) {
                    if (self.cancel.get()) return error.Cancelled;
                    if (d0 > 0) g.submit();
                    self.det.dictChunkCandidates(phase.det[@intFromEnum(lv)], fr, t_avg, self.pats[@intFromEnum(lv)], phase.dict_gpu, @min(chunk, total - d0), d0, n, keep_n);
                }
                g.read(self.det.dict_best, 0, std.mem.sliceAsBytes(scores));
                try g.wait();
                for (scores, 0..) |hit, i| {
                    const index = hit[1];
                    if (index >= total) {
                        peaks[i][4] = -2;
                        continue;
                    }
                    @memcpy(peaks[i][0..4], phase.dict[@as(usize, index) * 4 ..][0..4]);
                    peaks[i][4] = @bitCast(hit[0]);
                }
            }
            const phase_out = out[pi * peaks.len ..][0..peaks.len];
            var first: usize = 0;
            while (first < peaks.len) {
                if (self.cancel.get()) return error.Cancelled;
                var count: usize = 0;
                while (first < peaks.len and count < capacity) : (first += 1) {
                    const peak = peaks[first];
                    var norm: f32 = 0;
                    for (peak[0..4]) |v| norm += v * v;
                    if (!std.math.isFinite(norm) or @abs(norm - 1) > 0.01 or !std.math.isFinite(peak[4]) or peak[4] < -1) continue;
                    const b = first % n;
                    poses[count] = .{ .t = items[b][0..3].*, .pat = @floatFromInt(b), .q = peak[0..4].* };
                    source_slots[count] = first;
                    count += 1;
                }
                if (count == 0) continue;
                var refine = options;
                refine.stamp = self.nextBatchStamp();
                self.det.setPoses(poses[0..count]);
                self.det.refine(phase.det[@intFromEnum(lv)], fr, self.pats[@intFromEnum(lv)], @intCast(count), refine, records, 0);
                g.read(records, 0, std.mem.sliceAsBytes(refined[0..count]));
                try self.waitForRecords(.{ .ticket = g.readTicket(), .stamp = refine.stamp }, refined[0..count]);
                for (source_slots[0..count], refined[0..count]) |slot, record| {
                    phase_out[slot] = record;
                    phase_out[slot].z = 0;
                }
            }
        }
    }

    /// Candidate overlap components, in [cell][component] order. This is a fixed
    /// prepared-domain projection model, not a physical raw-intensity decomposition.
    /// Ordinary stored indexing results and the source acquisition are untouched.
    pub fn overlapCells(self: *Engine, cells: []const u64, max_components: u32, out: []overlap.Component, states: []overlap.State) !void {
        const progress = try self.gpa.alloc(overlap.Progress, cells.len);
        defer self.gpa.free(progress);
        return self.overlapCellsV2(cells, max_components, false, out, states, progress);
    }

    /// Discovery slots retain their identity even if a later joint fit gives a
    /// previous distinct template zero weight. No physical acceptance is implied.
    pub fn overlapCellsV2(self: *Engine, cells: []const u64, max_components: u32, stored: bool, out: []overlap.Component, states: []overlap.State, progress: []overlap.Progress) !void {
        return self.overlapCellsV3(cells, max_components, stored, out, states, progress, null);
    }

    pub fn overlapDC(self: *const Engine, phase: *const Phase, lv: Level) f32 {
        const k = @intFromEnum(lv);
        const source_dc = if (!self.set.bandpass and !self.set.clahe and phase.master.source_sd > 0) phase.master.source_mean / phase.master.source_sd else 0;
        return @floatCast((phase.bake_mean[k] + source_dc) / phase.bake_sd[k]);
    }

    pub fn overlapCellsV3(self: *Engine, cells: []const u64, max_components: u32, stored: bool, out: []overlap.Component, states: []overlap.State, progress: []overlap.Progress, theta: ?[]overlap_nuisance.Theta) !void {
        return self.overlapCellsModel(cells, max_components, stored, out, states, progress, theta, false, theta != null, null, 1, null, null);
    }
    pub fn overlapCellsV4(self: *Engine, cells: []const u64, max_components: u32, stored: bool, out: []overlap.Component, states: []overlap.State, progress: []overlap.Progress, theta: []overlap_nuisance.Theta, optimize: bool, norms: []f32) !void {
        if (norms.len != cells.len * max_components) return error.BadLength;
        @memset(norms, std.math.nan(f32));
        return self.overlapCellsModel(cells, max_components, stored, out, states, progress, theta, true, optimize, norms, 1, null, null);
    }
    /// V5 adds residual-order candidate pools; K1 and keep_n=1 call V4 exactly.
    pub fn overlapCellsV5(self: *Engine, cells: []const u64, max_components: u32, stored: bool, out: []overlap.Component, states: []overlap.State, progress: []overlap.Progress, theta: []overlap_nuisance.Theta, optimize: bool, norms: []f32, keep_n: u32, winner_ranks: []i32) !void {
        if (keep_n < 1 or keep_n > 8) return error.OverlapKeepNMustBeOneThroughEight;
        if (winner_ranks.len != out.len or norms.len != out.len) return error.BadLength;
        @memset(winner_ranks, -1);
        if (keep_n == 1 or max_components == 1) {
            try self.overlapCellsV4(cells, max_components, stored, out, states, progress, theta, optimize, norms);
            for (out, winner_ranks, 0..) |component, *rank, i| if (component.phase >= 0) {
                rank.* = if (i % max_components == 0) 0 else 1;
            };
            return;
        }
        @memset(norms, std.math.nan(f32));
        return self.overlapCellsModel(cells, max_components, stored, out, states, progress, theta, true, optimize, norms, keep_n, winner_ranks, null);
    }
    /// Transactional V6 boundary: no partial components, ranks or diagnostics
    /// reach the caller if cancellation, allocation or device work fails.
    pub fn overlapCellsV6(self: *Engine, cells: []const u64, k: u32, stored: bool, outputs: OverlapTrajectoryPayload, optimize: bool, keep_n: u32, options: overlap_profile.Options, source_q: [][4]f32, events: []overlap_profile.Event) !usize {
        if (!options.valid() or (options.flags & overlap_profile.refit_after_polish != 0 and !optimize)) return error.BadProfileOptions;
        const n = cells.len;
        const nk = try std.math.mul(usize, n, k);
        if (outputs.components.len != nk or outputs.states.len != n or outputs.progress.len != n or outputs.theta.len != n or
            outputs.norms.len != nk or outputs.ranks.len != nk or source_q.len != nk) return error.BadLength;
        const capacity = try overlap_profile.traceCapacity(n, self.phases.items.len, k, keep_n);
        if (events.len < capacity) return error.ProfileTraceTooSmall;
        var arena = std.heap.ArenaAllocator.init(self.gpa);
        defer arena.deinit();
        const alloc = arena.allocator();
        const temp: OverlapTrajectoryPayload = .{
            .components = try alloc.alloc(overlap.Component, nk),
            .states = try alloc.alloc(overlap.State, n),
            .progress = try alloc.alloc(overlap.Progress, n),
            .theta = try alloc.alloc(overlap_nuisance.Theta, n),
            .norms = try alloc.alloc(f32, nk),
            .ranks = try alloc.alloc(i32, nk),
        };
        var req: overlap_profile.Request = .{ .options = options, .source_q = try alloc.alloc([4]f32, nk), .events = try alloc.alloc(overlap_profile.Event, capacity) };
        @memset(temp.norms, std.math.nan(f32));
        @memset(temp.ranks, -1);
        @memset(req.source_q, @splat(0));
        // K1 only requests the ordinary primary. In particular, PCA is valid
        // here and SI must not allocate a volume for a nonexistent residual.
        const residual_keep_n = if (k == 1) 1 else keep_n;
        try self.overlapCellsModel(cells, k, stored, temp.components, temp.states, temp.progress, temp.theta, true, optimize, temp.norms, residual_keep_n, temp.ranks, &req);
        if (n > 0 and self.cancel.get()) return error.Cancelled;
        @memcpy(outputs.components, temp.components);
        @memcpy(outputs.states, temp.states);
        @memcpy(outputs.progress, temp.progress);
        @memcpy(outputs.theta, temp.theta);
        @memcpy(outputs.norms, temp.norms);
        @memcpy(outputs.ranks, temp.ranks);
        @memcpy(source_q, req.source_q);
        @memcpy(events[0..req.used], req.events[0..req.used]);
        return req.used;
    }
    const StoredPrimaryStatus = enum(u32) { valid = 0, missing = 6, stale = 10 };

    pub fn storedPrimaryContextCurrent(self: *const Engine) bool {
        const context = self.res.seed_context orelse return false;
        return sameSeedContext(context, self.exportContext());
    }

    pub fn storedPrimaryStatus(self: *const Engine, cell: usize, context_current: bool) StoredPrimaryStatus {
        const phase = self.res.phase[cell];
        if (phase < 0) return .missing;
        if (!context_current or self.res.seed_valid.len != self.res.phase.len or !self.res.seed_valid[cell]) return .stale;
        var norm: f32 = 0;
        for (self.res.quat[4 * cell ..][0..4]) |v| norm += v * v;
        const ncc = self.res.ncc[cell];
        if (phase >= self.phases.items.len or !std.math.isFinite(norm) or @abs(norm - 1) > 0.01 or !std.math.isFinite(ncc) or ncc < -1 or ncc > 1.00001 or !std.math.isFinite(self.res.nmi[cell])) return .missing;
        return .valid;
    }

    fn overlapCellsModel(self: *Engine, cells: []const u64, max_components: u32, stored: bool, out: []overlap.Component, states: []overlap.State, progress: []overlap.Progress, theta: ?[]overlap_nuisance.Theta, contrast: bool, optimize: bool, norms: ?[]f32, keep_n: u32, winner_ranks: ?[]i32, profile_request: ?*overlap_profile.Request) !void {
        const bandwidth = self.next_overlap_bandwidth;
        self.next_overlap_bandwidth = 0;
        if (theta) |values| {
            if (values.len != cells.len) return error.BadLength;
            @memset(values, overlap_nuisance.identity);
        }
        if (max_components < 1 or max_components > overlap.capacity) return error.OverlapComponentsMustBeOneThroughFour;
        const n = cells.len;
        if (out.len != n * max_components or states.len != n or progress.len != n) return error.BadLength;
        if (self.phases.items.len == 0) return error.NoPhases;
        const lv = self.levelOf(.search);
        if (self.levelOf(.refine) != lv or self.levelOf(.score) != lv) return error.OverlapRequiresSameResolutionUseFullSizeForNothing;
        const info_ = try self.info();
        for (cells) |cell| if (cell >= @as(u64, info_.rows) * info_.cols) {
            return error.BadCell;
        };
        const stored_context_current = stored and self.storedPrimaryContextCurrent();
        if (stored and self.res.seed_context != null and !stored_context_current) return error.StoredPrimaryContextStaleRerunIndex;
        for (out) |*c| c.* = .{};
        for (progress) |*m| m.* = .{};
        for (states) |*state| state.* = .{};
        if (n == 0) return;
        const hw = try self.patternShape(lv);
        const px = hw[0] * hw[1];
        if (px > 131072) return error.OverlapPatternTooLargeUseBinning;
        const ni: u32 = @intCast(n);
        const g = &self.g;
        const scratch_bytes = @max(@as(u64, overlap.capacity) * ni * px * 4, @as(u64, ni) * @sizeOf(overlap.State));
        if (scratch_bytes > @min(g.lim.max_storage_binding, g.lim.max_buffer)) return error.OverlapBatchTooLargeUseSmallerBatches;
        const gpa = self.gpa;
        var prm = self.searchOf();
        if (bandwidth != 0 and prm.method == .spherical) prm.bandwidth = bandwidth;
        if (max_components > 1 and keep_n > 1 and prm.method == .pca) return error.MultiCandidatePcaUnsupported;
        if (self.phases.items.len * keep_n > (1 << 24)) return error.TooManyCandidateIdentities;
        // Reset before preflight can suspend; never clear cancellation after a wait.
        self.cancel.set(false);
        // Fail on unsupported one-item SI allocations before ordinary fresh K1.
        if (max_components > 1 and keep_n > 1 and prm.method == .spherical) {
            if (comptime !@hasDecl(spherical.Spherical, "candidatePeaks")) return error.MultiCandidateSphericalUnavailable;
            const candidate_band = try self.band(prm.bandwidth, lv);
            const limit = @min(g.lim.max_storage_binding, g.lim.max_buffer);
            for (candidate_band.ph.items) |slot| {
                const ph = &slot.?.gpu;
                if (ph.specBytes(&candidate_band.tables) > limit or ph.volumeCells(&candidate_band.tables) * 4 > limit) return error.BandwidthExceedsDeviceLimit;
            }
        }
        if (self.cancel.get()) return error.Cancelled;
        const recs = try gpa.alloc(Record, self.phases.items.len * n);
        defer gpa.free(recs);
        const missing = try gpa.alloc(u32, n);
        defer gpa.free(missing);
        @memset(missing, 0);
        const primary_gates = try gpa.alloc(u32, n);
        defer gpa.free(primary_gates);
        @memset(primary_gates, 0);
        if (stored) {
            @memset(recs, .{ .q = @splat(0), .ncc = -2, .nmi = 0, .step = 0, .z = 0 });
            var valid_count: usize = 0;
            for (cells, 0..) |cell, i| {
                const c: usize = @intCast(cell);
                const primary_status = self.storedPrimaryStatus(c, stored_context_current);
                missing[i] = @intFromEnum(primary_status);
                if (primary_status != .valid) continue;
                const phase = self.res.phase[c];
                const q = self.res.quat[4 * c ..][0..4].*;
                valid_count += 1;
                primary_gates[i] = @as(u32, @intCast(phase)) + 1;
                // sweep=-1 may still hold a computed valid best winner.
                recs[@as(usize, @intCast(phase)) * n + i] = .{ .q = q, .ncc = self.res.ncc[c], .nmi = self.res.nmi[c], .step = 0, .z = 0 };
            }
            if (valid_count == 0) {
                for (progress, states, missing) |*m, *state, reason| {
                    m.stop_reason = reason;
                    m.previous_sse = std.math.nan(f32);
                    m.incremental_improvement = std.math.nan(f32);
                    state.correlation = std.math.nan(f32);
                    state.trial_sse = std.math.nan(f32);
                    state.sse = std.math.nan(f32);
                    state.initial_sse = std.math.nan(f32);
                    state.condition = std.math.nan(f32);
                    state.best_single_sse = std.math.nan(f32);
                    state.single_sse = @splat(std.math.nan(f32));
                    if (profile_request != null or (contrast and keep_n > 1 and max_components >= 3)) m._pad = std.math.maxInt(u32);
                }
                if (self.cancel.get()) return error.Cancelled;
                return;
            }
            // Stored winner reuse never calls indexCells/calibrate/refine. Prepare
            // only resources needed for future residual proposals; upload once.
            self.det.chunk_px = self.set.refine_pixels;
            self.det.wg = self.set.wg_refine;
            if (max_components > 1 and prm.method == .spherical) _ = try self.band(prm.bandwidth, lv);
            try self.prepareMasters(&.{lv});
            if (max_components > 1 and prm.method == .pca) {
                const t = try self.translationOf(&self.geo, lv);
                for (self.phases.items) |*phase| try self.preparePca(phase, prm.dict_grid, lv, try self.frame(lv), .{ @floatCast(t[0]), @floatCast(t[1]), @floatCast(t[2]) });
            }
            try self.uploadCellsAs(cells, &.{lv}, .{});
        } else {
            // Fresh discovery preserves the existing first-search path exactly.
            try self.indexCells(cells, self.searchOf(), recs);
        }
        if (self.cancel.get()) return error.Cancelled;
        if (profile_request) |req| {
            return overlap_profile.run(self, cells, max_components, recs, missing, hw, .{
                .components = out,
                .states = states,
                .progress = progress,
                .theta = theta.?,
                .norms = norms.?,
                .ranks = winner_ranks.?,
            }, optimize, keep_n, prm, req);
        }
        const protected = contrast and keep_n > 1 and max_components >= 3;
        if (!protected) return self.runPreparedTrajectory(cells, max_components, stored, recs, missing, primary_gates, hw, prm, out, states, progress, theta, contrast, optimize, norms, keep_n, winner_ranks);
        // Ordinary indexing's optional read-ahead is unrelated to this frozen
        // request; residual histories must never trigger acquisition I/O.
        const saved_ahead = self.ahead;
        self.ahead = &.{};
        defer self.ahead = saved_ahead;
        // Freeze the once-prepared input before either branch overwrites pats
        // with residuals. No acquisition I/O or preprocessing occurs in a branch.
        var target = g.storage(@as(u64, ni) * px * 4);
        defer g.release(&target);
        g.copy(self.pats[@intFromEnum(lv)], 0, target, 0, @as(u64, ni) * px * 4);
        g.copy(target, 0, self.pats[@intFromEnum(lv)], 0, @as(u64, ni) * px * 4);
        try self.runPreparedTrajectory(cells, max_components, stored, recs, missing, primary_gates, hw, prm, out, states, progress, theta, contrast, optimize, norms, keep_n, winner_ranks);
        if (self.cancel.get()) return error.Cancelled;
        const reference = try gpa.alloc(overlap.Component, out.len);
        defer gpa.free(reference);
        const reference_states = try gpa.alloc(overlap.State, n);
        defer gpa.free(reference_states);
        const reference_progress = try gpa.alloc(overlap.Progress, n);
        defer gpa.free(reference_progress);
        const reference_theta = try gpa.alloc(overlap_nuisance.Theta, n);
        defer gpa.free(reference_theta);
        const reference_norms = try gpa.alloc(f32, out.len);
        defer gpa.free(reference_norms);
        const reference_ranks = try gpa.alloc(i32, out.len);
        defer gpa.free(reference_ranks);
        g.copy(target, 0, self.pats[@intFromEnum(lv)], 0, @as(u64, ni) * px * 4);
        try self.runPreparedTrajectory(cells, max_components, stored, recs, missing, primary_gates, hw, prm, reference, reference_states, reference_progress, reference_theta, contrast, optimize, reference_norms, 1, reference_ranks);
        if (self.cancel.get()) return error.Cancelled;
        try finishOverlapTrajectories(self.cancel.get(), max_components, .{ .components = out, .states = states, .progress = progress, .theta = theta.?, .norms = norms.?, .ranks = winner_ranks.? }, .{ .components = reference, .states = reference_states, .progress = reference_progress, .theta = reference_theta, .norms = reference_norms, .ranks = reference_ranks });
        if (self.cancel.get()) return error.Cancelled;
    }

    pub const OverlapTrajectoryPayload = struct {
        components: []overlap.Component,
        states: []overlap.State,
        progress: []overlap.Progress,
        theta: []overlap_nuisance.Theta,
        norms: []f32,
        ranks: []i32,
    };
    /// Host finalization boundary: cancellation observed here cannot publish a
    /// selected payload. The SDK also discards the whole operation on any error.
    pub fn finishOverlapTrajectories(cancelled: bool, k: u32, pooled: OverlapTrajectoryPayload, reference: OverlapTrajectoryPayload) !void {
        if (cancelled) return error.Cancelled;
        for (0..pooled.states.len) |i| {
            const source = overlapTrajectorySource(pooled.states[i], pooled.progress[i], reference.states[i], reference.progress[i]);
            if (source != 1) {
                const first = i * k;
                @memcpy(pooled.components[first..][0..k], reference.components[first..][0..k]);
                pooled.states[i] = reference.states[i];
                pooled.progress[i] = reference.progress[i];
                pooled.theta[i] = reference.theta[i];
                @memcpy(pooled.norms[first..][0..k], reference.norms[first..][0..k]);
                @memcpy(pooled.ranks[first..][0..k], reference.ranks[first..][0..k]);
            }
            // No GPU operation consumes progress after this host-only stamp.
            pooled.progress[i]._pad = if (source < 0) std.math.maxInt(u32) else @intCast(source);
        }
    }

    fn overlapTrajectoryValid(state: overlap.State, progress: overlap.Progress) bool {
        if (progress.retained_order == 0 or progress.stop_reason == 6 or progress.stop_reason == 9 or progress.stop_reason == 10) return false;
        if (!std.math.isFinite(state.sse) or state.sse < 0 or !std.math.isFinite(state.initial_sse) or state.initial_sse <= 1e-12) return false;
        if (!std.math.isFinite(state.flags) or state.flags < 0 or state.flags >= 4294967296.0) return false;
        return (@as(u32, @intFromFloat(state.flags)) & (8 | 2)) == 0;
    }
    /// Per-cell final-objective choice. Exact ties and both invalid choose the
    /// complete reference payload; -1 distinguishes unavailable from reference.
    pub fn overlapTrajectorySource(pooled: overlap.State, pooled_progress: overlap.Progress, reference: overlap.State, reference_progress: overlap.Progress) i32 {
        const p = overlapTrajectoryValid(pooled, pooled_progress);
        const r = overlapTrajectoryValid(reference, reference_progress);
        if (p and (!r or pooled.sse < reference.sse)) return 1;
        return if (r) 0 else -1;
    }

    /// Run one history from immutable primary records and the current prepared
    /// GPU target. Each branch owns all fit/model/record scratch; never uploads
    /// inputs, preprocesses targets, searches a primary, or resets cancellation.
    fn runPreparedTrajectory(self: *Engine, cells: []const u64, max_components: u32, stored: bool, primary: []const Record, missing: []const u32, primary_gates: []const u32, hw: [2]u32, prm: Search, out: []overlap.Component, states: []overlap.State, progress: []overlap.Progress, theta: ?[]overlap_nuisance.Theta, contrast: bool, optimize: bool, norms: ?[]f32, keep_n: u32, winner_ranks: ?[]i32) !void {
        const n = cells.len;
        const ni: u32 = @intCast(n);
        const px = hw[0] * hw[1];
        const lv = self.levelOf(.search);
        const g = &self.g;
        const gpa = self.gpa;
        for (out) |*component| component.* = .{};
        @memset(states, .{});
        @memset(progress, .{});
        if (theta) |values| @memset(values, overlap_nuisance.identity);
        if (norms) |values| @memset(values, std.math.nan(f32));
        if (winner_ranks) |values| @memset(values, -1);
        const recs = try gpa.dupe(Record, primary);
        defer gpa.free(recs);
        const pool = if (keep_n > 1) try gpa.alloc(Record, self.phases.items.len * keep_n * n) else recs;
        defer if (keep_n > 1) gpa.free(pool);
        const winner_ids = try gpa.alloc(f32, n);
        defer gpa.free(winner_ids);
        var fit = overlap.Fit.init(g, ni, px, hw[1], self.maskOf(hw[0], hw[1]), self.pats[@intFromEnum(lv)]);
        defer fit.deinit();
        var nuisance: ?overlap_nuisance.Model = if (theta != null) overlap_nuisance.Model.init(g, &self.prep, &fit) else null;
        if (nuisance) |*model| {
            model.contrast = contrast;
            model.optimize_enabled = optimize;
        }
        defer if (nuisance) |*model| model.deinit();
        fit.readProgress(progress);
        fit.read(states);
        try g.wait();
        if (self.cancel.get()) return error.Cancelled;
        for (missing, progress, primary_gates) |is_missing, *m, gate| {
            if (is_missing != 0) m.stop_reason = is_missing;
            if (stored and is_missing == 0) m._pad = gate;
        }
        fit.writeProgress(progress);
        const poses = try gpa.alloc(detector.Pose, n);
        defer gpa.free(poses);
        const bd = if (max_components > 1 and prm.method == .spherical) try self.band(prm.bandwidth, lv) else null;
        const fr = try self.frame(lv);
        for (0..max_components) |pass| {
            if (self.cancel.get()) return error.Cancelled;
            if (pass != 0) {
                fit.residual(0, self.pats[@intFromEnum(lv)], true);
                if (keep_n > 1) try self.indexPreparedCandidates(cells, prm, bd, keep_n, pool) else try self.indexPrepared(cells, prm, bd, recs);
                if (self.cancel.get()) return error.Cancelled;
            }
            fit.beginManaged();
            if (nuisance) |*model| model.begin();
            for (self.phases.items, 0..) |*phase, pi| {
                const rank_count: usize = if (pass == 0) 1 else keep_n;
                for (0..rank_count) |rank| {
                    if (self.cancel.get()) return error.Cancelled;
                    const candidates = if (pass == 0 or keep_n == 1) recs[pi * n ..][0..n] else pool[(pi * keep_n + rank) * n ..][0..n];
                    for (cells, 0..) |cell, i| {
                        var q = candidates[i].q;
                        if (keep_n > 1) {
                            var qnorm: f32 = 0;
                            for (q) |v| qnorm += v * v;
                            if (!std.math.isFinite(qnorm) or @abs(qnorm - 1) > 0.01) q = .{ 1, 0, 0, 0 };
                        }
                        poses[i] = .{ .t = try self.positionOf(&self.geo, cell, lv), .pat = 0, .q = q };
                    }
                    self.det.setPoses(poses);
                    self.det.project(phase.det[@intFromEnum(lv)], fr, ni);
                    for (poses, 0..) |pose, i| {
                        var norm: f32 = 0;
                        for (pose.q) |v| norm += v * v;
                        const r = candidates[i];
                        var source_norm: f32 = 0;
                        for (r.q) |v| source_norm += v * v;
                        const valid = (keep_n == 1 or (std.math.isFinite(source_norm) and @abs(source_norm - 1) <= 0.01)) and std.math.isFinite(norm) and @abs(norm - 1) <= 0.01 and std.math.isFinite(r.ncc) and r.ncc >= -1 and r.ncc <= 1.00001 and std.math.isFinite(r.nmi);
                        const gate: u32 = if (valid) @as(u32, @intCast(pi)) + 1 else std.math.maxInt(u32);
                        g.writeSlice(fit.progress, @as(u64, i) * @sizeOf(overlap.Progress) + 28, u32, &.{gate});
                        if (!valid) g.clearRange(self.det.sims, @as(u64, i) * px * 4, @as(u64, px) * 4);
                    }
                    if (nuisance) |*model| {
                        if (keep_n > 1) model.candidateWithID(self.det.sims, @intCast(pi), self.overlapDC(phase, lv), stored and pass == 0, @intCast(pi * keep_n + rank)) else model.candidate(self.det.sims, @intCast(pi), self.overlapDC(phase, lv), stored and pass == 0);
                    } else {
                        self.prep.filter(self.det.sims, ni, hw[0], hw[1], .{ .mask = fr.mask });
                        fit.candidateManaged(self.det.sims, @intCast(pi), stored and pass == 0);
                    }
                }
            }
            fit.read(states);
            fit.readProgress(progress);
            if (keep_n > 1) nuisance.?.readWinnerIDs(winner_ids);
            try g.wait();
            if (self.cancel.get()) return error.Cancelled;
            var any = false;
            for (states, progress, 0..) |state, *m, i| {
                if (m.stop_reason != 0) continue;
                if (state.phase < 0) {
                    m.stop_reason = if ((m.boundary_seen & 2) != 0) 4 else if ((m.boundary_seen & 1) != 0) 5 else 3;
                    continue;
                }
                const pi: usize = @intFromFloat(state.phase);
                const winner_rank: usize = if (pass == 0 or keep_n == 1) 0 else @as(usize, @intFromFloat(winner_ids[i])) % keep_n;
                if (keep_n > 1 and (winner_ids[i] < 0 or @as(usize, @intFromFloat(winner_ids[i])) / keep_n != pi)) return error.IncoherentOverlapWinnerIdentity;
                const r = if (pass == 0 or keep_n == 1) recs[pi * n + i] else pool[(pi * keep_n + winner_rank) * n + i];
                const slot = m.retained_order;
                if (winner_ranks) |ranks| ranks[i * max_components + slot] = if (pass == 0) 0 else @intCast(winner_rank + 1);
                out[i * max_components + slot] = .{ .q = r.q, .phase = @intCast(pi), .proposal_ncc = r.ncc, .proposal_nmi = r.nmi, .amplitude = state.amplitude[slot] };
                m.retained_order += 1;
                m.active_count = 0;
                for (0..m.retained_order) |j| {
                    out[i * max_components + j].amplitude = state.amplitude[j];
                    if (state.amplitude[j] > 1e-6) m.active_count += 1;
                }
                m.incremental_improvement = @max(0, m.previous_sse - state.sse);
                if (state.sse <= 1e-8 * state.initial_sse) m.stop_reason = 2 else if (m.retained_order >= max_components) m.stop_reason = 1 else any = true;
            }
            fit.writeProgress(progress);
            if (!any) break;
        }
        if (nuisance) |*model| {
            if (norms) |values| {
                // Allocate before queuing either host destination: allocation
                // failure must not unwind a still-pending theta readback.
                const final_norms = try gpa.alloc([4]f32, n);
                defer gpa.free(final_norms);
                model.readTheta(theta.?);
                model.readNorms(final_norms, true);
                try g.wait();
                if (self.cancel.get()) return error.Cancelled;
                for (final_norms, 0..) |row, b| for (0..max_components) |j| {
                    values[b * max_components + j] = row[j];
                };
            } else {
                model.readTheta(theta.?);
                try g.wait();
                if (self.cancel.get()) return error.Cancelled;
            }
        }
        for (missing, states, progress) |reason, *state, *m| if (reason != 0) {
            state.* = .{ .sse = std.math.nan(f32), .initial_sse = std.math.nan(f32), .correlation = std.math.nan(f32), .condition = std.math.nan(f32), .single_sse = @splat(std.math.nan(f32)), .best_single_sse = std.math.nan(f32), .trial_sse = std.math.nan(f32) };
            m.previous_sse = std.math.nan(f32);
            m.incremental_improvement = std.math.nan(f32);
        };
    }

    /// Five images in the original prepared-target units, regenerated for one
    /// cell: measured, conditional-on-others, selected contribution, composite,
    /// signed final residual. Caller freezes/validates the current context.
    pub fn overlapInspect(self: *Engine, cell: u64, components: []const overlap.Component, selected: u32, out: []f32) !void {
        return self.overlapInspectV3(cell, components, selected, null, out);
    }
    pub fn overlapInspectV3(self: *Engine, cell: u64, components: []const overlap.Component, selected: u32, theta: ?overlap_nuisance.Theta, out: []f32) !void {
        return self.overlapInspectModel(cell, components, selected, theta, false, out);
    }
    pub fn overlapInspectV4(self: *Engine, cell: u64, components: []const overlap.Component, selected: u32, theta: overlap_nuisance.Theta, out: []f32) !void {
        return self.overlapInspectModel(cell, components, selected, theta, true, out);
    }
    fn overlapInspectModel(self: *Engine, cell: u64, components: []const overlap.Component, selected: u32, theta: ?overlap_nuisance.Theta, contrast: bool, out: []f32) !void {
        if (components.len == 0 or components.len > 4 or selected >= components.len) return error.BadLength;
        const info_ = try self.info();
        if (cell >= @as(u64, info_.rows) * info_.cols) return error.BadCell;
        const lv = self.levelOf(.search);
        if (self.levelOf(.refine) != lv or self.levelOf(.score) != lv) return error.OverlapRequiresSameResolutionUseFullSizeForNothing;
        const hw = try self.patternShape(lv);
        const px = @as(usize, hw[0]) * hw[1];
        if (out.len != 5 * px or px > 131072) return error.BadLength;
        try self.pattern(cell, .filtered, lv, out[0..px]);
        const mask = self.maskOf(hw[0], hw[1]);
        var lo: f32 = std.math.inf(f32);
        var hi: f32 = -std.math.inf(f32);
        for (out[0..px], 0..) |v, i| {
            const dr = @as(f32, @floatFromInt(i / hw[1])) - mask[0];
            const dc = @as(f32, @floatFromInt(i % hw[1])) - mask[1];
            if (mask[2] <= 0 or dr * dr + dc * dc <= mask[2]) {
                lo = @min(lo, v);
                hi = @max(hi, v);
            }
        }
        if (hi - lo <= 1e-6 * @max(@max(@abs(lo), @abs(hi)), 1e-30)) @memset(out[0..px], 0);
        @memcpy(out[px .. 2 * px], out[0..px]);
        @memset(out[2 * px ..], 0);
        try self.prepareMasters(&.{lv});
        const fr = try self.frame(lv);
        const sim = try self.gpa.alloc(f32, px);
        defer self.gpa.free(sim);
        var fit: ?overlap.Fit = if (theta != null) overlap.Fit.init(&self.g, 1, @intCast(px), hw[1], mask, self.pats[@intFromEnum(lv)]) else null;
        defer if (fit) |*f| f.deinit();
        var nuisance: ?overlap_nuisance.Model = if (fit) |*f| overlap_nuisance.Model.init(&self.g, &self.prep, f) else null;
        defer if (nuisance) |*m| m.deinit();
        if (nuisance) |*model| model.contrast = contrast;
        for (components, 0..) |c, j| {
            if (!std.math.isFinite(c.amplitude) or c.amplitude < 0) return error.BadComponent;
            if (c.phase < 0) {
                if (c.amplitude != 0) return error.BadComponent;
                continue;
            }
            if (c.phase >= self.phases.items.len) return error.NoSuchPhase;
            var norm: f32 = 0;
            for (c.q) |v| norm += v * v;
            if (!std.math.isFinite(norm) or @abs(norm - 1) > 0.01) return error.BadQuaternion;
            self.det.setPoses(&.{.{ .t = try self.positionOf(&self.geo, cell, lv), .pat = 0, .q = c.q }});
            self.det.project(self.phases.items[@intCast(c.phase)].det[@intFromEnum(lv)], fr, 1);
            var prepared = self.det.sims;
            if (nuisance) |*m| {
                prepared = m.apply(self.det.sims, self.overlapDC(&self.phases.items[@intCast(c.phase)], lv), theta.?);
            } else self.prep.filter(prepared, 1, hw[0], hw[1], .{ .mask = fr.mask });
            self.g.read(prepared, 0, std.mem.sliceAsBytes(sim));
            try self.g.wait();
            for (sim, 0..) |v, i| {
                const a = c.amplitude * v;
                out[3 * px + i] += a;
                if (j == selected) out[2 * px + i] = a else out[px + i] -= a;
            }
        }
        for (0..px) |i| out[4 * px + i] = out[i] - out[3 * px + i];
    }

    /// Fixed supplied projections: a small public numerical seam for known-answer
    /// checks and candidate assessment, normalized on the GPU with no hidden filters.
    /// Templates are [component][pattern][pixel], target is [pattern][pixel].
    pub fn fitOverlapProjections(self: *Engine, target: []const f32, templates: []const f32, n: u32, k: u32, h: u32, w: u32, amplitudes: []f32, residual: []f32, states: []overlap.State) !void {
        return self.fitOverlapProjectionsV3(target, templates, n, k, h, w, amplitudes, residual, states, null, .{ 0, 0, 0 });
    }
    pub fn fitOverlapProjectionsV3(self: *Engine, target: []const f32, templates: []const f32, n: u32, k: u32, h: u32, w: u32, amplitudes: []f32, residual: []f32, states: []overlap.State, theta: ?[]overlap_nuisance.Theta, mask: [3]f32) !void {
        return self.fitOverlapProjectionsModel(target, templates, n, k, h, w, amplitudes, residual, states, theta, mask, false, theta != null, null);
    }
    pub fn fitOverlapProjectionsV4(self: *Engine, target: []const f32, templates: []const f32, n: u32, k: u32, h: u32, w: u32, amplitudes: []f32, residual: []f32, states: []overlap.State, theta: []overlap_nuisance.Theta, mask: [3]f32, optimize: bool, norms: []f32) !void {
        if (norms.len != n * k) return error.BadLength;
        return self.fitOverlapProjectionsModel(target, templates, n, k, h, w, amplitudes, residual, states, theta, mask, true, optimize, norms);
    }
    fn fitOverlapProjectionsModel(self: *Engine, target: []const f32, templates: []const f32, n: u32, k: u32, h: u32, w: u32, amplitudes: []f32, residual: []f32, states: []overlap.State, theta: ?[]overlap_nuisance.Theta, mask: [3]f32, contrast: bool, optimize: bool, norms: ?[]f32) !void {
        if (n == 0 or h == 0 or w == 0 or k == 0 or k > overlap.capacity) return error.BadLength;
        const px = @as(usize, h) * w;
        if (px > 131072) return error.OverlapPatternTooLargeUseBinning;
        if (target.len != n * px or templates.len != k * n * px or amplitudes.len != n * k or residual.len != target.len or states.len != n) return error.BadLength;
        for (target) |v| if (!std.math.isFinite(v)) {
            return error.NonFinitePattern;
        };
        for (templates) |v| if (!std.math.isFinite(v)) {
            return error.NonFinitePattern;
        };
        const g = &self.g;
        const scratch_bytes = @max(@as(u64, overlap.capacity) * n * px * 4, @as(u64, n) * @sizeOf(overlap.State));
        if (scratch_bytes > @min(g.lim.max_storage_binding, g.lim.max_buffer)) return error.OverlapBatchTooLargeUseSmallerBatches;
        var original = g.storage(target.len * 4);
        defer g.release(&original);
        var output = g.storage(target.len * 4);
        defer g.release(&output);
        if (contrast) {
            // Scalar centering only: protect this supplied-image seam from a
            // representable target pedestal. Experimental index preparation and
            // every spatial/background operator remain unchanged.
            const stable_target = try self.gpa.alloc(f32, target.len);
            defer self.gpa.free(stable_target);
            for (0..n) |b| {
                const image = target[b * px ..][0..px];
                var anchor: f64 = 0;
                var total: f64 = 0;
                var count: usize = 0;
                for (image, 0..) |v, i| {
                    const dy = @as(f64, @floatFromInt(i / w)) - mask[0];
                    const dx = @as(f64, @floatFromInt(i % w)) - mask[1];
                    if (mask[2] > 0 and dy * dy + dx * dx > mask[2]) continue;
                    if (count == 0) anchor = v;
                    total += @as(f64, v) - anchor;
                    count += 1;
                }
                const mean_delta = total / @as(f64, @floatFromInt(@max(count, 1)));
                for (image, 0..) |v, i| stable_target[b * px + i] = @floatCast((@as(f64, v) - anchor) - mean_delta);
            }
            g.writeSlice(original, 0, f32, stable_target);
        } else g.writeSlice(original, 0, f32, target);
        self.prep.filter(original, n, h, w, .{ .mask = mask });
        var fit = overlap.Fit.init(g, n, @intCast(px), w, mask, original);
        defer fit.deinit();
        if (theta) |values| {
            if (values.len != n) return error.BadLength;
            var model = overlap_nuisance.Model.init(g, &self.prep, &fit);
            defer model.deinit();
            model.contrast = contrast;
            model.optimize_enabled = optimize;
            model.fixed(templates, k);
            if (norms) |norm_values| {
                // Reserve both host destinations before queuing either readback.
                const final_norms = try self.gpa.alloc([4]f32, n);
                defer self.gpa.free(final_norms);
                model.readTheta(values);
                model.readNorms(final_norms, false);
                try g.wait();
                for (final_norms, 0..) |row, b| for (0..k) |j| {
                    norm_values[b * k + j] = row[j];
                };
            } else model.readTheta(values);
        } else {
            g.writeSlice(fit.bank, 0, f32, templates);
            self.prep.filter(fit.bank, k * n, h, w, .{ .mask = mask });
            fit.fixed(k);
        }
        fit.residual(k, output, false);
        fit.read(states);
        g.read(output, 0, std.mem.sliceAsBytes(residual));
        try g.wait();
        for (states, 0..) |state, b_| for (0..k) |j| {
            amplitudes[b_ * k + j] = state.amplitude[j];
        };
    }

    /// A batch in flight (indexBatches): its cells, the records it lands in, what was recorded.
    const Flight = struct { cells: []const u64, recs: []Record, rec: Recorded };

    /// Index `cells` a batch at a time, two batches in flight: each batch is recorded and
    /// submitted while the device still works on the one before, which is waited on only then,
    /// so the device does not stand idle between batches; the host reads the batch after (from
    /// `cells`, and then `after`) meanwhile. `land(ctx, cells, records)` takes each batch's
    /// records once they land, in order. Stops between batches when cancelled. Nothing is left
    /// in flight when it returns, an error included (what is in flight lands first).
    fn indexBatches(self: *Engine, cells: []const u64, after: []const u64, prm: Search, ctx: anytype, comptime land: fn (@TypeOf(ctx), []const u64, []Record) anyerror!void) !void {
        const gpa = self.gpa;
        const batch: usize = @max(self.set.batch, 1);
        const per = self.phases.items.len * @min(batch, cells.len);
        var recs: [2][]Record = .{ &.{}, &.{} };
        defer for (recs) |r| gpa.free(r);
        recs[0] = try gpa.alloc(Record, per);
        recs[1] = try gpa.alloc(Record, per);
        errdefer self.g.wait() catch {};
        var cur: ?Flight = null;
        var at: usize = 0;
        var slot: usize = 0;
        while (true) {
            const more = at < cells.len and !self.cancel.get();
            // (a batch whose averaging is still to be reported lands before the next is recorded:
            // the report reads what its averaging left)
            if (cur) |c| if (!more or self.nl_report == .queued) {
                cur = null;
                try self.finishCells(c.rec, c.cells.len, c.recs);
                try land(ctx, c.cells, c.recs);
            };
            if (!more) break;
            const n = @min(batch, cells.len - at);
            const mine = cells[at..][0..n];
            const out = recs[slot][0 .. self.phases.items.len * n];
            const rec = try self.recordCells(mine, prm, out);
            self.g.submit();
            at += n;
            slot = 1 - slot;
            const next = if (at < cells.len) cells[at..][0..@min(batch, cells.len - at)] else after;
            if (next.len != 0 and !self.cancel.get()) self.readAhead(next);
            if (cur) |c| {
                try self.finishCells(c.rec, c.cells.len, c.recs);
                try land(ctx, c.cells, c.recs);
            }
            cur = .{ .cells = mine, .recs = out, .rec = rec };
        }
    }

    fn landNowhere(_: void, _: []const u64, _: []Record) anyerror!void {}

    /// The key to a phase's inverse pole figure colours, as the settings have them.
    pub fn ipfKeyOf(self: *const Engine, phase: usize) sym.Key {
        // a direction's colour follows the Laue class; an orientation's, the point group's own rotations
        const m = &self.phases.items[phase].master;
        return sym.Key.init(if (self.set.ipf_colors == .perfect_rotation) m.rotation else m.laue, self.set.ipf_colors);
    }

    /// Measure, once for the scan as it is prepared now, what mutual information chance gives:
    /// a sample of the region's patterns (about 256 on a lattice) against each phase at random
    /// orientations, at the level the scores are taken. Its median and spread (1.4826 × the
    /// median deviation) per phase are the scale every match's NMI is then read on; they are
    /// measured again only when what the scores depend on changes (the scan, the patterns'
    /// preparation, the geometry, the phases, the region).
    pub fn calibrate(self: *Engine) !void {
        const gpa = self.gpa;
        const g = &self.g;
        const s = &self.set;
        const n_ph = self.phases.items.len;
        if (n_ph == 0) return error.NoPhases;
        // the key: the settings with those that do not touch a score put aside
        var k = s.*;
        inline for (.{ "batch", "tile_mb", "refine_pixels", "wg_fft", "wg_refine", "sweeps", "sweep_bandwidth", "sweep_dict_grid", "bandwidth", "dict_grid", "fit_bandwidth", "fit_grid", "refine_iters", "sim_halfw", "sim_rank", "pc_steps", "pc_steps_z", "pc_levels", "pc_top_k", "pca_components", "pca_epochs", "pc_nms", "toy_seed", "toy_rows", "toy_cols", "toy_height", "toy_width", "toy_grains", "toy_noise_seed" }) |f| @field(k, f) = 0;
        inline for (.{ "toy_counts", "toy_spread_deg", "toy_blank", "toy_background_sigma" }) |f| @field(k, f) = 0;
        k.toy_contrast = 1;
        inline for (.{ "accept_z", "accept_nmi", "accept_step_deg", "refine_step", "pc_span", "pc_span_z", "pc_shrink", "pc_gate", "pc_kde_sigma", "pc_inlier_radius", "pc_z_min" }) |f| @field(k, f) = 0;
        k.method = .spherical;
        k.sweep_method = .same;
        k.pc_vote = .ballots;
        inline for (.{ "sim_voltage_kv", "sim_tilt_deg", "sim_bin_kev", "sim_dmin_nm", "sim_c_strong", "sim_c_cutoff", "sim_dbdiff_cutoff", "sim_stop" }) |f| @field(k, f) = 0;
        k.sim_correct = false;
        k.plane_interp = .bilinear;
        k.plane_master_px = 0;
        k.plane_master_auto = false;
        k.plane_kind = .affine;
        k.plane_frame = .nominal;
        k.plane_weights = .equal;
        k.plane_stride = 0;
        k.plane_patterns_max = 0;
        k.plane_trust = 0;
        k.plane_outlier = 0;
        k.plane_iters = 0;
        k.refine = false;
        k.ipf_colors = .classic;
        const json = try k.toJson(gpa);
        defer gpa.free(json);
        var h = std.hash.Wyhash.init(n_ph);
        h.update(json);
        h.update(std.mem.asBytes(&self.geo));
        self.calibration.hash(&h);
        h.update(std.mem.asBytes(&self.scan_serial));
        for (self.phases.items) |*p| h.update(std.mem.asBytes(&p.serial));
        const key = h.final() | 1;
        if (key == self.chance_key and self.chance_med.len == n_ph) return;
        self.chance_key = 0;

        const region = try self.regionCells();
        defer gpa.free(region);
        const n: u32 = @intCast(@min(region.len, 256));
        const cells = try gpa.alloc(u64, n);
        defer gpa.free(cells);
        for (cells, 0..) |*c, j| c.* = region[(2 * j + 1) * region.len / (2 * @as(usize, n))];
        const draws: u32 = 8;
        const lc = self.levelOf(.score);
        try self.prepareMasters(&.{lc});
        // (not a batch of indexing: nothing read ahead is its, and none of its account)
        const ahead = self.ahead;
        self.ahead = &.{};
        defer self.ahead = ahead;
        try self.uploadCellsAs(cells, &.{lc}, .{});
        const fr = try self.frame(lc);
        // each sample pattern where the detector stood for it; chance supplies the orientations
        const poses = try gpa.alloc(detector.Pose, n);
        defer gpa.free(poses);
        for (cells, 0..) |c, b| poses[b] = .{ .t = try self.positionOf(&self.geo, c, lc), .pat = @floatFromInt(b), .q = .{ 1, 0, 0, 0 } };
        self.det.setPoses(poses);
        const per = @as(usize, n) * draws;
        g.ensure(&self.chance, @as(u64, n_ph) * per * 4);
        for (self.phases.items, 0..) |*p, pi| {
            self.det.chanceOf(p.det[@intFromEnum(lc)], fr, self.pats[@intFromEnum(lc)], n, draws, s.nmi_bins);
            g.copy(self.det.chance, 0, self.chance, @as(u64, pi) * per * 4, per * 4);
        }
        const host = try gpa.alloc(f32, n_ph * per);
        defer gpa.free(host);
        g.read(self.chance, 0, std.mem.sliceAsBytes(host));
        try g.wait();
        if (self.chance_med.len != n_ph) {
            gpa.free(self.chance_med);
            gpa.free(self.chance_spread);
            self.chance_med = &.{};
            self.chance_spread = &.{};
            self.chance_med = try gpa.alloc(f32, n_ph);
            self.chance_spread = try gpa.alloc(f32, n_ph);
        }
        for (0..n_ph) |pi| {
            const c = host[pi * per ..][0..per];
            std.mem.sort(f32, c, {}, std.sort.asc(f32));
            const med = c[per / 2];
            for (c) |*v| v.* = @abs(v.* - med);
            std.mem.sort(f32, c, {}, std.sort.asc(f32));
            self.chance_med[pi] = med;
            self.chance_spread[pi] = @max(1.4826 * c[per / 2], 1e-6);
            self.events.log("chance, phase {d}: NMI {d:.4} ± {d:.4} over {d} patterns at random orientations; z {d:.0} is NMI {d:.4}", .{ pi, med, self.chance_spread[pi], n, s.accept_z, med + s.accept_z * self.chance_spread[pi] });
        }
        self.chance_key = key;
    }

    /// The phase a pattern is assigned among its per-phase records.
    pub fn choosePhase(self: *const Engine, recs: []const Record, n: usize, b: usize) usize {
        var best: usize = 0;
        var best_v: f32 = -std.math.inf(f32);
        for (0..recs.len / n) |p| {
            const r = recs[p * n + b];
            const v = if (self.set.phase_by == .nmi) r.nmi else r.ncc;
            if (v > best_v) {
                best_v = v;
                best = p;
            }
        }
        return best;
    }

    // ── indexing a region, a few batches per step, in sweeps ──

    /// The cells of the region of interest (the whole map when it is empty), row-major.
    pub fn regionCells(self: *Engine) ![]u64 {
        const i = try self.info();
        const s = &self.set;
        var x0: u32 = 0;
        var y0: u32 = 0;
        var w: u32 = i.cols;
        var h: u32 = i.rows;
        if (s.roi_w != 0 and s.roi_h != 0) {
            x0 = @min(s.roi_x, i.cols - 1);
            y0 = @min(s.roi_y, i.rows - 1);
            w = @min(s.roi_w, i.cols - x0);
            h = @min(s.roi_h, i.rows - y0);
        }
        const cells = try self.gpa.alloc(u64, @as(usize, w) * h);
        for (0..h) |r| for (0..w) |c| {
            cells[r * w + c] = @as(u64, y0 + r) * i.cols + x0 + c;
        };
        return cells;
    }

    pub fn indexBegin(self: *Engine) !void {
        if (self.phases.items.len == 0) return error.NoPhases;
        try self.res.prepareCandidates(self.gpa, self.phases.items.len);
        const cells = try self.regionCells();
        const included = self.gpa.alloc(bool, self.res.phase.len) catch |err| {
            self.gpa.free(cells);
            return err;
        };
        defer self.gpa.free(included);
        @memset(included, false);
        for (cells) |cell| included[@intCast(cell)] = true;
        var outside = false;
        for (self.res.phase, 0..) |phase, cell| if (phase >= 0 and !included[cell]) {
            outside = true;
            break;
        };
        const context: ExportContext = self.exportContext();
        if (!outside) {
            self.res.context_consistent = true;
        } else if (self.res.context) |old| {
            if (!sameExportContext(old, context)) self.res.context_consistent = false;
        }
        self.res.context = context;
        self.gpa.free(self.index_cells);
        self.index_cells = cells;
        self.nl_report = if (self.set.nlpar) .wanted else .off;
        if (self.set.nlpar) try self.orderInBlocks(cells);
        if (self.res.seed_context) |old| {
            if (!sameSeedContext(old, context)) {
                @memset(self.res.seed_valid, false);
                self.res.seed_context = context;
            }
        } else {
            @memset(self.res.seed_valid, false);
            self.res.seed_context = context;
        }
        self.index_done = 0;
        self.index_region = cells.len;
        self.index_sweep = 0;
        self.index_accepted = 0;
        self.cancel.set(false);
        for (cells) |c| {
            self.res.seed_valid[@intCast(c)] = false;
            for (0..self.res.candidate_phases) |p| self.res.candidate_sweep[p * self.res.phase.len + @as(usize, @intCast(c))] = -1;
            self.res.phase[@intCast(c)] = -1;
            self.res.sweep[@intCast(c)] = -1;
            self.res.z[@intCast(c)] = 0;
        }
    }

    /// Reorder the region's cells (a rectangle, row by row) into square blocks of about a
    /// batch each: a batch averaged with its neighbours reads a margin of cells around itself,
    /// which is small beside a block and many times a strip of rows.
    fn orderInBlocks(self: *Engine, cells: []u64) !void {
        const i = try self.info();
        if (cells.len == 0) return;
        const w: usize = @intCast(cells[cells.len - 1] % i.cols - cells[0] % i.cols + 1);
        const h = cells.len / w;
        const side: usize = @intFromFloat(@ceil(@sqrt(@as(f64, @floatFromInt(self.set.batch)))));
        const rowwise = try self.gpa.dupe(u64, cells);
        defer self.gpa.free(rowwise);
        var k: usize = 0;
        var by: usize = 0;
        while (by < h) : (by += side) {
            var bx: usize = 0;
            while (bx < w) : (bx += side) {
                for (by..@min(by + side, h)) |y| for (bx..@min(bx + side, w)) |x| {
                    cells[k] = rowwise[y * w + x];
                    k += 1;
                };
            }
        }
    }

    pub fn indexStatus(self: *const Engine) IndexStatus {
        const sw = self.set.sweep(self.index_sweep);
        return .{
            .sweep = self.index_sweep,
            .sweeps = self.set.sweeps,
            .bandwidth = sw.bandwidth,
            .dict_grid = sw.dict_grid,
            .done = self.index_done,
            .total = self.index_cells.len,
            .region = self.index_region,
            .accepted = self.index_accepted,
            .method = @intFromEnum(sw.method),
        };
    }

    /// A batch of the sweep landed (indexBatches): each cell keeps the better match, the
    /// accepted ones their sweep.
    fn landIndexed(self: *Engine, cells: []const u64, recs: []Record) anyerror!void {
        const s = &self.set;
        const n_ph = self.phases.items.len;
        const n = cells.len;
        const landed_context = self.exportContext();
        if (self.res.context) |context| {
            if (!sameExportContext(context, landed_context)) self.res.context_consistent = false;
        }
        const coherent_seed = if (self.res.seed_context) |seed| sameSeedContext(seed, landed_context) else false;
        const settle: f32 = @floatCast(std.math.degreesToRadians(s.accept_step_deg));
        for (cells, 0..) |c, b| {
            const cell: usize = @intCast(c);
            for (0..n_ph) |pi| {
                const slot = pi * self.res.phase.len + cell;
                const candidate = recs[pi * n + b];
                const value = if (s.phase_by == .nmi) candidate.nmi else candidate.ncc;
                const previous = if (self.res.candidate_sweep[slot] < 0) -std.math.inf(f32) else if (s.phase_by == .nmi) self.res.candidates[slot].nmi else self.res.candidates[slot].ncc;
                if (self.res.candidate_sweep[slot] < 0 or value > previous) {
                    self.res.candidates[slot] = candidate;
                    self.res.candidate_sweep[slot] = @intCast(self.index_sweep);
                }
            }
            const p = self.choosePhase(recs, n, b);
            const r = recs[p * n + b];
            const at: usize = @intCast(c);
            // an acceptable match replaces whatever was kept; otherwise the better one stays
            const good = if (s.accept_nmi > 0) r.nmi >= s.accept_nmi else r.z >= s.accept_z;
            if (self.res.phase[at] < 0 or (good and r.step <= settle) or r.z > self.res.z[at]) {
                const q = sym.toFz(.{ r.q[0], r.q[1], r.q[2], r.q[3] }, self.phases.items[p].master.rotations());
                for (0..4) |k| self.res.quat[4 * at + k] = @floatCast(q[k]);
                self.res.ncc[at] = r.ncc;
                self.res.nmi[at] = r.nmi;
                self.res.z[at] = r.z;
                self.res.phase[at] = @intCast(p);
                self.res.settled[at] = r.step <= settle;
                var norm: f32 = 0;
                for (r.q) |v| norm += v * v;
                self.res.seed_valid[at] = coherent_seed and std.math.isFinite(norm) and @abs(norm - 1) <= 0.01 and std.math.isFinite(r.ncc) and r.ncc >= -1 and r.ncc <= 1.00001 and std.math.isFinite(r.nmi);
            }
            const kept_good = if (s.accept_nmi > 0) self.res.nmi[at] >= s.accept_nmi else self.res.z[at] >= s.accept_z;
            if (self.res.sweep[at] < 0 and kept_good and self.res.settled[at]) {
                self.res.sweep[at] = @intCast(self.index_sweep);
                self.index_accepted += 1;
            }
        }
        self.index_done += n;
        self.events.emit(.{ .kind = "progress", .stage = "index", .done = self.index_done, .total = self.index_cells.len, .sweep = self.index_sweep, .sweeps = s.sweeps });
    }

    /// Index the next few batches of the sweep in progress; when the sweep ends, the cells
    /// without an accepted match become the next one's. Returns the cells left to visit (0:
    /// done). A match is accepted when it stands far enough above chance (settings.accept_z) and
    /// its refinement had settled (settings.accept_step_deg). Every cell keeps the best match
    /// any sweep gave it, accepted or not.
    pub fn indexStep(self: *Engine) !usize {
        const s = &self.set;
        const left = self.index_cells.len - self.index_done;
        if (left == 0 or self.cancel.get()) return 0;
        if (self.res.context) |context| {
            if (!sameExportContext(context, self.exportContext())) self.res.context_consistent = false;
        }
        // A few batches a call, about 4096 patterns, two at a time in flight (indexBatches); a
        // call ends with nothing in flight, so whatever comes between calls finds the device
        // idle.
        const batch: usize = @max(s.batch, 1);
        const n = @min(left, batch * std.math.clamp(4096 / batch, 2, 16));
        const cells = self.index_cells[self.index_done..][0..n];
        const after = self.index_cells[self.index_done + n ..][0..@min(left - n, batch)];
        const sw = s.sweep(self.index_sweep);
        try self.indexBatches(cells, after, .{ .method = sw.method, .bandwidth = sw.bandwidth, .dict_grid = sw.dict_grid }, self, landIndexed);
        if (self.index_done == self.index_cells.len and self.index_sweep + 1 < s.sweeps) {
            // the next sweep takes what this one left
            var keep: usize = 0;
            for (self.index_cells) |c| {
                if (self.res.sweep[@intCast(c)] >= 0) continue;
                self.index_cells[keep] = c;
                keep += 1;
            }
            self.events.log("sweep {d} of {d} ({s} {d}): {d} of {d} patterns accepted, {d} left", .{
                self.index_sweep + 1,                                        s.sweeps,
                switch (sw.method) {
                    .spherical => "spherical, bandwidth",
                    .dictionary => "dictionary, grid",
                    .pca => "dictionary components, grid",
                },
                if (sw.method == .spherical) sw.bandwidth else sw.dict_grid, self.index_cells.len - keep,
                self.index_cells.len,                                        keep,
            });
            if (keep != 0) {
                self.index_cells = try self.gpa.realloc(self.index_cells, keep);
                self.index_done = 0;
                self.index_sweep += 1;
            }
        }
        return self.index_cells.len - self.index_done;
    }

    // ── tuning ──

    fn tuneFree(self: *Engine) void {
        for (self.tune.cells) |c| self.gpa.free(c);
        self.tune = .{};
    }

    fn tuneValues(self: *const Engine) [Tune.count]u32 {
        return .{ self.set.batch, self.set.refine_pixels, self.set.wg_refine, self.set.wg_fft };
    }

    fn tuneApply(self: *Engine, v: [Tune.count]u32) void {
        self.set.batch = v[0];
        self.set.refine_pixels = v[1];
        self.set.wg_refine = v[2];
        self.set.wg_fft = v[3];
    }

    /// Whether the last sweep searches otherwise than the first (and so is timed on its own).
    fn tuneTwoSweeps(self: *const Engine) bool {
        const a = self.set.sweep(0);
        const b = self.set.sweep(self.set.sweeps - 1);
        return a.method != b.method or a.bandwidth != b.bandwidth or a.dict_grid != b.dict_grid;
    }

    /// The first `want` or so cells of the region as a rectangle from its first cell: whole
    /// rows, as indexing reads them; or, when patterns are averaged with their neighbours, as
    /// square as the region lets it be, as indexing's blocks are. Caller frees.
    fn tuneCells(self: *Engine, want: usize) ![]u64 {
        const gpa = self.gpa;
        const region = try self.regionCells();
        defer gpa.free(region);
        const i = try self.info();
        const w: usize = @intCast(region[region.len - 1] % i.cols - region[0] % i.cols + 1);
        const h = region.len / w;
        const side: usize = @intFromFloat(@ceil(@sqrt(@as(f64, @floatFromInt(want)))));
        const across = @min(w, if (self.set.nlpar) side else want);
        const down = std.math.clamp(want / across, 1, h);
        const cells = try gpa.alloc(u64, across * down);
        for (0..down) |y| @memcpy(cells[y * across ..][0..across], region[y * w ..][0..across]);
        return cells;
    }

    /// Whether a set of values can be tried on this scan: each in its range, and a batch other
    /// than the one held only if the sample holds it twice over (a single batch never waits on
    /// the next one's reading) and one buffer holds its patterns.
    fn tuneFits(self: *const Engine, v: [Tune.count]u32) bool {
        for (v, Tune.lo, Tune.hi) |x, a, b| if (x < a or x > b) return false;
        if (v[0] == self.tune.best[0]) return true;
        const i = self.info() catch return false;
        const bytes = @as(u64, v[0]) * i.pat_h * i.pat_w * 4;
        return 2 * @as(usize, v[0]) <= self.tune.cells[0].len and bytes <= @min(self.g.lim.max_storage_binding, self.g.lim.max_buffer) / 2;
    }

    /// Start tuning: take the samples. The first trial times the settings as they are.
    pub fn tuneBegin(self: *Engine) !void {
        if (self.phases.items.len == 0) return error.NoPhases;
        self.tuneFree();
        const t = &self.tune;
        const a = self.set.sweep(0);
        const b = self.set.sweep(self.set.sweeps - 1);
        t.cells[0] = try self.tuneCells(Tune.sampleOf(a.method, a.bandwidth, a.dict_grid));
        t.cells[1] = try self.tuneCells(Tune.sampleOf(b.method, b.bandwidth, b.dict_grid));
        t.best = self.tuneValues();
        t.first = t.best;
        t.active = true;
        self.cancel.set(false);
    }

    /// Run the trial in hand: index its sample with its settings as indexing would a sweep of
    /// it (batches, two in flight, reading ahead), keeping nothing. The front end times this call. A trial's
    /// first call indexes one batch only, to settle what a new value needs built.
    pub fn tuneTrial(self: *Engine) !void {
        const t = &self.tune;
        if (!t.active) return error.NotTuning;
        self.tuneApply(if (t.base) t.best else t.cand);
        defer self.tuneApply(t.best);
        const gpa = self.gpa;
        const batch: usize = self.set.batch;
        const sample = t.cells[@intFromBool(t.last)];
        const cells = try gpa.dupe(u64, if (t.run == 0) sample[0..@min(batch, sample.len)] else sample);
        defer gpa.free(cells);
        if (self.set.nlpar and t.run != 0) try self.orderInBlocks(cells);
        const sw = self.set.sweep(if (t.last) self.set.sweeps - 1 else 0);
        if (self.cancel.get()) return error.Cancelled;
        try self.indexBatches(cells, &.{}, .{ .method = sw.method, .bandwidth = sw.bandwidth, .dict_grid = sw.dict_grid }, {}, landNowhere);
        if (self.cancel.get()) return error.Cancelled;
    }

    /// The next candidate worth a trial, into `cand`: the values held with the move in hand
    /// applied (its settings doubled, or halved once doubling did not pay), on to the next
    /// move when neither can be. False: no move is left.
    fn tuneNext(self: *Engine) bool {
        const t = &self.tune;
        while (t.move < Tune.moves.len) {
            if (t.move >= Tune.on_last and !t.last and self.tuneTwoSweeps()) {
                // from here on the last sweep is what is timed: first the values held, on it
                t.last = true;
                t.base = true;
                return true;
            }
            var c = t.best;
            for (&c, Tune.moves[t.move]) |*x, takes_part| {
                if (takes_part) x.* = if (t.up) x.* * 2 else x.* / 2;
            }
            if (self.tuneFits(c)) {
                t.cand = c;
                return true;
            }
            self.tuneReverse();
        }
        return false;
    }

    /// The candidate in this direction was not kept: try the other way if this was the move's
    /// first step up, else leave the move.
    fn tuneReverse(self: *Engine) void {
        const t = &self.tune;
        if (t.up and !t.kept) {
            t.up = false;
            return;
        }
        t.move += 1;
        t.up = true;
        t.kept = false;
    }

    /// Take the seconds the last call of `tuneTrial` took (negative: it failed) and move on.
    /// Returns whether a call is left; when none is, the settings hold the fastest values found.
    pub fn tuneReport(self: *Engine, seconds: f64) bool {
        const t = &self.tune;
        if (!t.active) return false;
        const failed = !(seconds >= 0);
        if (!failed) {
            // (the first call of a trial is one batch, untimed)
            if (t.run > 0) t.time = @min(t.time, seconds);
            t.run += 1;
            if (t.run <= Tune.runs) return true;
        }
        const which: usize = @intFromBool(t.last);
        t.trials += 1;
        if (t.base) {
            if (failed) {
                // the settings as they are do not run: nothing to tune from
                t.active = false;
                return false;
            }
            t.best_time = t.time;
            t.was[which] = t.time;
            t.base = false;
        } else if (!failed and t.time < t.best_time * (1 - Tune.worth)) {
            t.best_time = t.time;
            t.best = t.cand;
            t.kept = true;
        } else {
            self.tuneReverse();
        }
        t.now[which] = t.best_time;
        t.run = 0;
        t.time = std.math.inf(f64);
        const more = self.tuneNext();
        self.tuneApply(t.best);
        self.events.emit(.{ .kind = "progress", .stage = "tune", .done = @min(t.move, Tune.moves.len), .total = Tune.moves.len });
        if (more) return true;
        t.active = false;
        var buf: [320]u8 = undefined;
        var w: std.Io.Writer = .fixed(&buf);
        for (Tune.names, t.first, t.best) |name, a, b| {
            if (a == b) w.print(", {s} {d}", .{ name, a }) catch {} else w.print(", {s} {d} → {d}", .{ name, a, b }) catch {};
        }
        const n0: f64 = @floatFromInt(t.cells[0].len);
        const n1: f64 = @floatFromInt(t.cells[1].len);
        if (t.was[1] > 0) w.print("; first sweep {d:.0} → {d:.0} patterns per second, last (on {d}) {d:.0} → {d:.0}", .{ n0 / t.was[0], n0 / t.now[0], t.cells[1].len, n1 / t.was[1], n1 / t.now[1] }) catch {} else w.print("; {d:.0} → {d:.0} patterns per second", .{ n0 / t.was[0], n0 / t.now[0] }) catch {};
        self.events.log("tuned on {d} patterns in {d} trials{s}", .{ t.cells[0].len, t.trials, w.buffered() });
        return false;
    }

    // ── looking at things ──

    /// How far along the pipeline a pattern is looked at.
    pub const View = enum(u32) {
        /// as stored (reduced, at a reduced level)
        stored = 0,
        /// averaged with its neighbours (when that is on) and filtered: what is indexed
        filtered = 1,
        /// averaged with its neighbours (when that is on), before any filter
        averaged = 2,
        /// not a pattern: the static background, as it is subtracted at this size
        background = 3,
    };

    /// One pattern as f32 at a level, at a stage of the pipeline. When it was averaged with
    /// its neighbours, `nl_view` and `nl_view_shares` then say what that did to it.
    pub fn pattern(self: *Engine, cell: u64, view: View, lv: Level, out: []f32) !void {
        const hw = try self.patternShape(lv);
        if (out.len != @as(usize, hw[0]) * hw[1]) return error.BadLength;
        const g = &self.g;
        const averaged = self.set.nlpar and (view == .filtered or view == .averaged);
        self.nl_view_side = 0;
        if (view == .background) {
            const bg = try self.staticBackground(lv) orelse return error.NoBackground;
            g.read(bg, 0, std.mem.sliceAsBytes(out));
            try g.wait();
            return;
        }
        if (view == .stored) {
            const bytes = try self.readCells(&.{cell}, false);
            try self.uploadLevel(lv, &.{cell}, bytes, null);
        } else {
            try self.uploadCellsAs(&.{cell}, &.{lv}, .{ .filter = view == .filtered, .account = averaged });
        }
        g.read(self.pats[@intFromEnum(lv)], 0, std.mem.sliceAsBytes(out));
        try g.wait();
        if (!averaged) return;
        self.nl_view = self.nlparSummary();
        const side = 2 * nlpar_mod.Window.reachOf(self.set.nlpar_radius, self.set.nlpar_iterations) + 1;
        const f = @as(usize, side) * side;
        if (self.nl_view_shares.len != f) {
            self.gpa.free(self.nl_view_shares);
            self.nl_view_shares = &.{};
            self.nl_view_shares = try self.gpa.alloc(f32, f);
        }
        // (a cell outside the map has no share: NaN, to tell it from a neighbour given none)
        for (self.nl_view_shares, self.nl_shares[0..f], self.nl_where[0..f]) |*o, a, at| o.* = if (at == nlpar_mod.NONE) std.math.nan(f32) else a;
        self.nl_view_side = side;
    }

    /// Mean and standard deviation of each given cell's stored pattern (2 f32 per cell), on the
    /// host: a map of either shows the microstructure before anything is indexed.
    pub fn patternStats(self: *Engine, cells: []const u64, out: []f32) !void {
        if (out.len != 2 * cells.len) return error.BadLength;
        const i = try self.info();
        const total = @as(u64, i.rows) * i.cols;
        for (cells) |c| if (c >= total) return error.OutOfRange;
        const bytes = try self.readCells(cells, false);
        const px = @as(usize, i.pat_h) * i.pat_w;
        for (0..cells.len) |k| {
            var sum: f64 = 0;
            var sq: f64 = 0;
            switch (i.pixel) {
                .u8 => for (bytes[k * px ..][0..px]) |v| {
                    const f: f64 = @floatFromInt(v);
                    sum += f;
                    sq += f * f;
                },
                .u16 => for (0..px) |j| {
                    const f: f64 = @floatFromInt(std.mem.readInt(u16, bytes[(k * px + j) * 2 ..][0..2], .little));
                    sum += f;
                    sq += f * f;
                },
                .f32 => for (0..px) |j| {
                    const f: f64 = @as(f32, @bitCast(std.mem.readInt(u32, bytes[(k * px + j) * 4 ..][0..4], .little)));
                    sum += f;
                    sq += f * f;
                },
            }
            const n: f64 = @floatFromInt(px);
            const mean = sum / n;
            out[2 * k] = @floatCast(mean);
            out[2 * k + 1] = @floatCast(@sqrt(@max(sq / n - mean * mean, 0)));
        }
    }

    pub fn storedOverviewKind(self: *const Engine) scan_mod.OverviewKind {
        return switch (self.source) {
            .file => |s| s.overview_kind,
            else => .none,
        };
    }

    pub fn storedOverviewCount(self: *const Engine) usize {
        return switch (self.source) {
            .file => |s| if (s.overview_kind != .none) @intCast(s.count()) else 0,
            else => 0,
        };
    }

    pub fn readStoredOverview(self: *Engine, out: []f32) !void {
        switch (self.source) {
            .file => |s| try s.readStoredOverview(out),
            else => return error.NoStoredOverview,
        }
    }

    /// Simulated patterns of a phase at the given orientations, each as cell `cells[k]` would
    /// record it at a level, from the master as filtered for that level.
    pub fn simulate(self: *Engine, phase: u32, cells: []const u64, quats: []const f32, lv: Level, out: []f32) !void {
        if (phase >= self.phases.items.len) return error.NoSuchPhase;
        const hw = try self.patternShape(lv);
        const n = cells.len;
        if (quats.len != 4 * n or out.len != n * hw[0] * hw[1]) return error.BadLength;
        try self.prepareMasters(&.{lv});
        const fr = try self.frame(lv);
        const poses = try self.gpa.alloc(detector.Pose, n);
        defer self.gpa.free(poses);
        for (cells, 0..) |c, b| {
            poses[b] = .{ .t = try self.positionOf(&self.geo, c, lv), .pat = 0, .q = quats[4 * b ..][0..4].* };
        }
        self.det.setPoses(poses);
        self.det.project(self.phases.items[phase].det[@intFromEnum(lv)], fr, @intCast(n));
        self.g.read(self.det.sims, 0, std.mem.sliceAsBytes(out));
        try self.g.wait();
    }

    /// A toy scan from the phases loaded (settings toy_*), in place of whatever scan was open:
    /// grains by nearest of toy_grains random points, each with a random orientation and
    /// phase, every cell simulated as it would be recorded under the present geometry, laid on
    /// a background and counted as electrons. The microstructure comes from toy_seed and the
    /// counting from toy_noise_seed; the patterns themselves are the device's arithmetic.
    /// The answer goes to `quats` (4 per cell: w, x, y, z), `phase` (−1: a blank cell) and
    /// `grain` (the cell's grain).
    pub fn synthesize(self: *Engine, quats: []f32, phase: []i32, grain: []i32) !void {
        const gpa = self.gpa;
        const s = self.set;
        const n_ph = self.phases.items.len;
        if (n_ph == 0) return error.NoPhases;
        const n: usize = @as(usize, s.toy_rows) * s.toy_cols;
        const px: usize = @as(usize, s.toy_height) * s.toy_width;
        if (quats.len != 4 * n or phase.len != n or grain.len != n) return error.BadLength;
        const data = try gpa.alloc(f32, n * px);
        defer gpa.free(data);
        @memset(data, 0);
        // an empty scan of the shape first: a cell's place in it is part of its geometry
        const calibration = self.calibration;
        const scan_turn = self.scan_turn;
        var geometry = self.geo;
        try self.setPatterns(std.mem.sliceAsBytes(data), .f32, s.toy_rows, s.toy_cols, s.toy_height, s.toy_width);
        // A generic scan without a known pixel size takes the default for this new
        // detector shape. Oxford's unknown scale remains unknown. Resolve once so
        // simulation and the final scan use exactly the same geometry.
        if (calibration.metadata == null and !(geometry.px_um > 0)) geometry.px_um = self.geo.px_um;
        geometry.adoptPlaneReference(s.toy_rows, s.toy_cols);
        self.calibration = calibration;
        self.scan_turn = scan_turn;
        self.geo = geometry;

        // the microstructure
        var structure = std.Random.DefaultPrng.init(s.toy_seed);
        const r = structure.random();
        const Grain = struct { x: f64, y: f64, q: rot.Quat, phase: i32 };
        const grains = try gpa.alloc(Grain, @min(@as(usize, s.toy_grains), n));
        defer gpa.free(grains);
        for (grains) |*gr| {
            gr.x = r.float(f64) * @as(f64, @floatFromInt(s.toy_cols));
            gr.y = r.float(f64) * @as(f64, @floatFromInt(s.toy_rows));
            // four normal numbers, normalized: uniform over the rotations
            gr.q = rot.qstd(.{ r.floatNorm(f64), r.floatNorm(f64), r.floatNorm(f64), r.floatNorm(f64) });
            gr.phase = @intCast(r.uintLessThan(usize, n_ph));
        }
        // (a rotation vector of three equal normal components has their root-sum-square for its angle)
        const spread = std.math.degreesToRadians(s.toy_spread_deg) / @sqrt(3.0);
        for (0..s.toy_rows) |row| for (0..s.toy_cols) |col| {
            const x = @as(f64, @floatFromInt(col)) + 0.5;
            const y = @as(f64, @floatFromInt(row)) + 0.5;
            var best: usize = 0;
            var best_d: f64 = std.math.inf(f64);
            for (grains, 0..) |gr, i| {
                const d = (gr.x - x) * (gr.x - x) + (gr.y - y) * (gr.y - y);
                if (d < best_d) {
                    best_d = d;
                    best = i;
                }
            }
            const at = row * s.toy_cols + col;
            var q = grains[best].q;
            if (spread > 0) q = rot.qstd(rot.qmul(rot.so3Exp(.{ spread * r.floatNorm(f64), spread * r.floatNorm(f64), spread * r.floatNorm(f64) }), q));
            quats[4 * at ..][0..4].* = .{ @floatCast(q[0]), @floatCast(q[1]), @floatCast(q[2]), @floatCast(q[3]) };
            grain[at] = @intCast(best);
            phase[at] = if (s.toy_blank > 0 and r.float(f64) < s.toy_blank) -1 else grains[best].phase;
        };

        // each phase's cells, a readback at a time
        const lv = self.viewLevel(true);
        const chunk: usize = @max(1, (4 << 20) / px);
        const cells = try gpa.alloc(u64, chunk);
        defer gpa.free(cells);
        const qs = try gpa.alloc(f32, 4 * chunk);
        defer gpa.free(qs);
        const sims = try gpa.alloc(f32, chunk * px);
        defer gpa.free(sims);
        var made: usize = 0;
        for (0..n_ph) |p| {
            var at: usize = 0;
            while (at < n) {
                var k: usize = 0;
                while (at < n and k < chunk) : (at += 1) {
                    if (phase[at] != @as(i32, @intCast(p))) continue;
                    cells[k] = at;
                    qs[4 * k ..][0..4].* = quats[4 * at ..][0..4].*;
                    k += 1;
                }
                if (k == 0) continue;
                if (self.cancel.get()) return error.Cancelled;
                try self.simulate(@intCast(p), cells[0..k], qs[0 .. 4 * k], lv, sims[0 .. k * px]);
                for (cells[0..k], 0..) |c, i| @memcpy(data[@intCast(c * px)..][0..px], sims[i * px ..][0..px]);
                made += k;
                self.events.emit(.{ .kind = "progress", .stage = "synthesize", .done = made, .total = n });
            }
        }

        // the detector: the simulation (its darkest pixel 0, its mean 1) on a background (mean 1)
        var mean: f64 = 0;
        var low: f64 = std.math.inf(f64);
        var crystals: usize = 0;
        for (0..n) |c| {
            if (phase[c] < 0) continue;
            crystals += 1;
            for (data[c * px ..][0..px]) |v| {
                mean += v;
                low = @min(low, v);
            }
        }
        if (crystals > 0) mean /= @floatFromInt(crystals * px);
        const gain = if (crystals > 0 and mean > low) 1.0 / (mean - low) else 0;
        const back = try gpa.alloc(f64, px);
        defer gpa.free(back);
        @memset(back, 1);
        if (s.toy_background_sigma > 0) {
            const sg = s.toy_background_sigma * @as(f64, @floatFromInt(s.toy_width));
            var sum: f64 = 0;
            for (0..s.toy_height) |yy| for (0..s.toy_width) |xx| {
                const dx = @as(f64, @floatFromInt(xx)) + 0.5 - 0.5 * @as(f64, @floatFromInt(s.toy_width));
                const dy = @as(f64, @floatFromInt(yy)) + 0.5 - 0.5 * @as(f64, @floatFromInt(s.toy_height));
                const v = @exp(-(dx * dx + dy * dy) / (2 * sg * sg));
                back[yy * s.toy_width + xx] = v;
                sum += v;
            };
            for (back) |*v| v.* *= @as(f64, @floatFromInt(px)) / sum;
        }
        // counted: the mean pixel toy_counts electrons, every pixel a Poisson draw at its own intensity
        var counting = std.Random.DefaultPrng.init(if (s.toy_noise_seed != 0) s.toy_noise_seed else @as(u64, s.toy_seed) ^ 0x9E3779B97F4A7C15);
        const rn = counting.random();
        for (0..n) |c| {
            const blank = phase[c] < 0 or gain == 0;
            for (data[c * px ..][0..px], back) |*v, bg| {
                const signal = if (blank) 1.0 else (v.* - low) * gain;
                const intensity = (1 - s.toy_contrast) * bg + s.toy_contrast * signal;
                v.* = if (s.toy_counts > 0) @floatFromInt(poisson(rn, s.toy_counts * intensity)) else @floatCast(intensity);
            }
        }
        try self.setPatterns(std.mem.sliceAsBytes(data), .f32, s.toy_rows, s.toy_cols, s.toy_height, s.toy_width);
        self.calibration = calibration;
        self.scan_turn = scan_turn;
        self.geo = geometry;
        self.events.log("toy scan: {d} × {d} patterns of {d} × {d}, {d} grains of {d} phase(s), {d} blank cells, contrast {d:.2}, {d:.4} electrons a pixel, seed {d}", .{ s.toy_rows, s.toy_cols, s.toy_height, s.toy_width, grains.len, n_ph, n - crystals, s.toy_contrast, s.toy_counts, s.toy_seed });
    }

    /// A phase's master as the search compares patterns with it (filtered for the search's
    /// level): [north | south], side × side each.
    pub fn masterImage(self: *Engine, phase: u32, out: []f32) !void {
        if (phase >= self.phases.items.len) return error.NoSuchPhase;
        const lv = self.levelOf(.search);
        try self.prepareMasters(&.{lv});
        const work = self.phases.items[phase].work[@intFromEnum(lv)];
        if (out.len != work.len) return error.BadLength;
        @memcpy(out, work);
    }
};

/// A Poisson draw of mean `lam`: by multiplying uniform numbers for small means (Knuth), by
/// transformed rejection for larger ones (Hörmann 1993, PTRS).
fn poisson(r: std.Random, lam: f64) u32 {
    if (!(lam > 0)) return 0;
    if (lam < 10) {
        const limit = @exp(-lam);
        var k: u32 = 0;
        var p: f64 = r.float(f64);
        while (p > limit) : (k += 1) p *= r.float(f64);
        return k;
    }
    const slam = @sqrt(lam);
    const b = 0.931 + 2.53 * slam;
    const a = -0.059 + 0.02483 * b;
    const inv_alpha = 1.1239 + 1.1328 / (b - 3.4);
    const vr = 0.9277 - 3.6224 / (b - 2);
    while (true) {
        const u = r.float(f64) - 0.5;
        const v = r.float(f64);
        const us = 0.5 - @abs(u);
        const kf = @floor((2 * a / us + b) * u + lam + 0.43);
        if (us >= 0.07 and v <= vr) return @intFromFloat(kf);
        if (kf < 0 or (us < 0.013 and v > us)) continue;
        if (@log(v) + @log(inv_alpha) - @log(a / (us * us) + b) <= -lam + kf * @log(lam) - std.math.lgamma(f64, kf + 1)) return @intFromFloat(kf);
    }
}

fn bytesPerPixel(pixel: scan_mod.Pixel) usize {
    return switch (pixel) {
        .u8 => 1,
        .u16 => 2,
        .f32 => 4,
    };
}
