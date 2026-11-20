"""ctypes bindings to the zebsdsolve shared library (built from Zig; see src/exports.zig, src/native_api.zig).

The structures mirror the engine's `extern struct`s field for field (and web/zebsdsolve.js mirrors
their byte offsets): append fields at the end and update all three together.
"""
import ctypes
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

U32, I32, I64, U64, F64, SZ, P = ctypes.c_uint32, ctypes.c_int32, ctypes.c_int64, ctypes.c_uint64, ctypes.c_double, ctypes.c_size_t, ctypes.c_void_p
f32p, f64p, u8p, u32p, i32p, u64p = (ctypes.POINTER(t) for t in (ctypes.c_float, ctypes.c_double, ctypes.c_uint8, ctypes.c_uint32, ctypes.c_int32, ctypes.c_uint64))


class ScanInfo(ctypes.Structure):
    _fields_ = [
        ("rows", U32), ("cols", U32), ("pat_h", U32), ("pat_w", U32),
        ("pixel", U32), ("has_background", U32), ("has_pc", U32), ("format", U32),
        ("voltage_kv", F64), ("bytes", F64), ("has_px", U32), ("_pad", U32),
    ]


class PhaseInfo(ctypes.Structure):
    _fields_ = [
        ("side", U32), ("point_group", U32), ("space_group", U32), ("laue", U32),
        ("n_fold", U32), ("mirror", U32), ("inversion", U32), ("_pad", U32),
        ("voltage_kv", F64), ("name", ctypes.c_char * 64),
    ]


class Geometry(ctypes.Structure):
    _fields_ = [
        ("pc", F64 * 3), ("sample_tilt_deg", F64), ("detector_tilt_deg", F64), ("rot", F64 * 3),
        ("px_um", F64), ("step_x_um", F64), ("step_y_um", F64),
        ("plane_kind", U32), ("_pad", U32), ("plane", F64 * 12),
    ]


class PcResult(ctypes.Structure):
    _fields_ = [
        ("pc", F64 * 3), ("start", F64 * 3), ("score", F64), ("start_score", F64),
        ("n_patterns", U32), ("n_candidates", U32), ("levels", U32), ("at_border", U32),
    ]


class PlaneResult(ctypes.Structure):
    _fields_ = [
        ("plane", F64 * 12), ("pc", F64 * 3), ("start_pc", F64 * 3), ("rot", F64 * 3),
        ("said_rot", F64 * 3), ("said_rot_se", F64 * 3), ("step", F64 * 2), ("step_se", F64 * 2), ("skew", F64), ("skew_se", F64),
        ("pc_se", F64 * 3), ("rmsd", F64 * 3), ("ncc", F64), ("start_ncc", F64),
        ("kind", U32), ("frame", U32), ("n_patterns", U32), ("n_used", U32), ("iterations", U32), ("blocks", U32),
    ]


class PlanePoint(ctypes.Structure):
    _fields_ = [("cell", U64), ("pc", F64 * 3), ("fit", F64 * 3), ("ncc", F64), ("used", U32), ("_pad", U32)]


class SimSummary(ctypes.Structure):
    _fields_ = [
        ("n_k", U32), ("n_sites", U32), ("n_g", U32), ("n_bins", U32), ("n_bins_run", U32), ("max_beams", U32),
        ("failed", U32), ("stopped", U32), ("halfw", U32), ("rank", U32), ("max_weak", U32), ("overflow", U32), ("last_change", F64),
    ]


class IndexStatus(ctypes.Structure):
    _fields_ = [
        ("sweep", U32), ("sweeps", U32), ("bandwidth", U32), ("dict_grid", U32),
        ("done", U64), ("total", U64), ("region", U64), ("accepted", U64),
        ("method", U32), ("_pad", U32),
    ]


class OverlapComponent(ctypes.Structure):
    _fields_ = [("q", ctypes.c_float * 4), ("amplitude", ctypes.c_float),
                ("phase", I32), ("proposal_ncc", ctypes.c_float), ("proposal_nmi", ctypes.c_float)]


class OverlapProgress(ctypes.Structure):
    _fields_ = [("retained_order",U32),("proposed_count",U32),("active_count",U32),("stop_reason",U32),
                ("previous_sse",ctypes.c_float),("incremental_improvement",ctypes.c_float),("boundary_seen",U32),("_pad",U32)]


class OverlapProfileOptions(ctypes.Structure):
    _fields_ = [("size", U32), ("flags", U32), ("radius_deg", ctypes.c_float), ("min_residual_ncc", ctypes.c_float)]


class OverlapStageOptions(ctypes.Structure):
    _fields_ = [("base", OverlapProfileOptions), ("min_stage_improvement", ctypes.c_float * 3)]


class OverlapProfileEvent(ctypes.Structure):
    _fields_ = [("cell_index", U32), ("branch", U32), ("order", U32), ("attempt", U32),
                ("phase", I32), ("rank", I32), ("flags", U32), ("result_order", U32),
                ("source_q", ctypes.c_float * 4), ("residual_ncc", ctypes.c_float),
                ("baseline_sse", ctypes.c_float), ("lower_sse", ctypes.c_float), ("joint_sse", ctypes.c_float)]


class OverlapState(ctypes.Structure):
    _fields_ = [("sse", ctypes.c_float), ("initial_sse", ctypes.c_float),
                ("phase", ctypes.c_float), ("flags", ctypes.c_float),
                ("amplitude", ctypes.c_float * 4), ("single_sse", ctypes.c_float * 4),
                ("correlation", ctypes.c_float), ("condition", ctypes.c_float),
                ("best_single_sse", ctypes.c_float), ("trial_sse", ctypes.c_float)]


def _names():
    if sys.platform == "win32":
        return "zebsdsolve.dll", "wgpu_native.dll"
    if sys.platform == "darwin":
        return "libzebsdsolve.dylib", "libwgpu_native.dylib"
    return "libzebsdsolve.so", "libwgpu_native.so"


def _find(env, name, dev_dirs):
    """ZEBSDSOLVE_LIB / ZEBSDSOLVE_WGPU, then the package's _lib/ (wheels), then a source checkout."""
    if os.environ.get(env):
        return Path(os.environ[env])
    for d in [HERE / "_lib", *dev_dirs]:
        if (d / name).exists():
            return d / name
    raise FileNotFoundError(f"{name} not found (set {env}, or build: zig build)")


_SIGS = {
    # name: (argtypes, restype)
    "ze_version": ([], ctypes.c_char_p),
    "ze_create": ([ctypes.c_char_p], P),
    "ze_create_error": ([ctypes.c_char_p, SZ], SZ),
    "ze_destroy": ([P], None),
    "ze_adapter": ([P, ctypes.c_char_p, SZ], SZ),
    "ze_error": ([P, ctypes.c_char_p, SZ], SZ),
    "ze_configure": ([P, ctypes.c_char_p, SZ], I32),
    "ze_settings": ([P, ctypes.c_char_p, SZ], SZ),
    "ze_export_snapshot": ([P, f64p, P, SZ], I64),
    "ze_export_oxford_snapshot": ([P, f64p, P, SZ], I64),
    "ze_scan_overview_kind": ([P], U32),
    "ze_scan_overview": ([P, f32p, SZ], I64),
    "ze_events": ([P, ctypes.c_char_p, SZ], SZ),
    "ze_cancel": ([P], None),
    "ze_open_scan": ([P, ctypes.c_char_p, SZ], I32),
    "ze_open_scan_configured": ([P, ctypes.c_char_p, SZ, ctypes.c_char_p, SZ, ctypes.c_char_p, SZ], I32),
    "ze_set_patterns": ([P, P, SZ, U32, U32, U32, U32, U32], I32),
    "ze_synthesize": ([P, f32p, ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_int32), SZ], I32),
    "ze_chunk_index_status": ([P, ctypes.POINTER(U64)], None),
    "ze_prepare_index_step": ([P, U32, U32], I32),
    "ze_close_scan": ([P], None),
    "ze_scan_info": ([P, ctypes.POINTER(ScanInfo)], None),
    "ze_scan_format": ([P, ctypes.c_char_p, SZ], SZ),
    "ze_has_acquisition_calibration": ([P], U32),
    "ze_get_geometry": ([P, ctypes.POINTER(Geometry)], None),
    "ze_set_geometry": ([P, ctypes.POINTER(Geometry)], None),
    "ze_add_phase": ([P, ctypes.c_char_p, SZ], I32),
    "ze_clear_phases": ([P], None),
    "ze_seed_source_roi": ([P,u32p],U32),
    "ze_phase_generation": ([P],U64),
    "ze_phase_count": ([P], U32),
    "ze_phase_info": ([P, U32, ctypes.POINTER(PhaseInfo)], I32),
    "ze_master_image": ([P, U32, f32p, SZ], I32),
    "ze_pattern_shape": ([P, U32, u32p], I32),
    "ze_pattern": ([P, U64, U32, U32, f32p, SZ], I32),
    "ze_nlpar_view": ([P, f32p, f32p, SZ], U32),
    "ze_lattice": ([P, U32, u64p, SZ], I64),
    "ze_pixel_angle": ([P], F64),
    "ze_pattern_stats": ([P, u64p, SZ, f32p, SZ], I32),
    "ze_simulate": ([P, U32, u64p, f32p, SZ, U32, f32p, SZ], I32),
    "ze_fit_pc": ([P, ctypes.POINTER(PcResult)], I32),
    "ze_pc_scores": ([P, u32p, f64p, SZ, f32p, SZ], I32),
    "ze_pc_level_count": ([P], U32),
    "ze_pc_level": ([P, U32, u32p, u32p, f64p, SZ, f32p, SZ, u32p, f32p, SZ], I32),
    "ze_pc_level_scores": ([P, U32, f32p, SZ], I32),
    "ze_pc_level_ballots": ([P, U32, f32p, SZ, f64p], I32),
    "ze_fit_auto": ([P, ctypes.POINTER(PlaneResult)], I32),
    "ze_sim_read_cif": ([P, ctypes.c_char_p, SZ], I32),
    "ze_sim_set_crystal": ([P, ctypes.c_char_p, SZ], I32),
    "ze_sim_edit_crystal": ([P, ctypes.c_char_p, SZ], I32),
    "ze_sim_crystal": ([P, ctypes.c_void_p, SZ], ctypes.c_int64),
    "ze_sim_run": ([P, ctypes.POINTER(SimSummary)], I32),
    "ze_sim_sheets": ([P, f32p, SZ], I32),
    "ze_sim_add_phase": ([P, ctypes.c_char_p, SZ], I32),
    "ze_sim_npz": ([P, ctypes.c_void_p, SZ], ctypes.c_int64),
    "ze_fit_plane": ([P, ctypes.POINTER(PlaneResult)], I32),
    "ze_plane_points": ([P, ctypes.POINTER(PlanePoint), SZ], SZ),
    "ze_fit_patterns": ([P, u64p, i32p, f32p, f32p, SZ], I32),
    "ze_tune_begin": ([P], I32),
    "ze_tune_trial": ([P], I32),
    "ze_tune_report": ([P, ctypes.c_double], I32),
    "ze_chance": ([P, U32, f32p], I32),
    "ze_index_begin": ([P], I32),
    "ze_index_step": ([P], I64),
    "ze_index_status": ([P, ctypes.POINTER(IndexStatus)], None),
    "ze_sweep": ([P, U32, u32p], None),
    "ze_overlap_cells": ([P, u64p, SZ, U32, ctypes.POINTER(OverlapComponent), SZ, ctypes.POINTER(OverlapState), SZ], I32),
    "ze_overlap_cells_v2": ([P,u64p,SZ,U32,U32,ctypes.POINTER(OverlapComponent),SZ,ctypes.POINTER(OverlapState),SZ,ctypes.POINTER(OverlapProgress),SZ],I32),
    "ze_overlap_phase_moments": ([P,f64p,SZ],I32),
    "ze_overlap_cells_v3": ([P,u64p,SZ,U32,U32,ctypes.POINTER(OverlapComponent),SZ,ctypes.POINTER(OverlapState),SZ,ctypes.POINTER(OverlapProgress),SZ,f32p,SZ],I32),
    "ze_overlap_inspect_v3": ([P,U64,ctypes.POINTER(OverlapComponent),SZ,U32,f32p,SZ,f32p,SZ],I32),
    "ze_overlap_fit_v3": ([P,f32p,f32p,U32,U32,U32,U32,f32p,f32p,ctypes.POINTER(OverlapState),f32p,f32p],I32),
    "ze_overlap_inspect": ([P,U64,ctypes.POINTER(OverlapComponent),SZ,U32,f32p,SZ],I32),
    "ze_orientation_colors": ([P,f32p,i32p,SZ,U32,u8p,SZ],I32),
    "ze_overlap_fit": ([P, f32p, f32p, U32, U32, U32, U32, f32p, f32p, ctypes.POINTER(OverlapState)], I32),
    "ze_index_cells": ([P, u64p, SZ, f32p, SZ], I32),
    "ze_index_cells_by": ([P, u64p, SZ, U32, U32, U32, f32p, SZ], I32),
    "ze_results": ([P, U32, P, SZ], I64),
    "ze_map": ([P, U32, u8p, SZ], I32),
    "ze_ipf_key": ([P, U32, U32, U32, u8p, SZ, f32p], I32),
    "ze_export": ([P, U32, P, SZ], I64),
    "ze_bunge": ([f32p, f64p], None),
    "ze_reduce": ([P, U32, f32p, f64p], None),
    "ze_disorientation": ([P, U32, f32p, f32p], F64),
    "ze_compile_wgsl": ([P, ctypes.c_char_p, SZ, ctypes.c_char_p, SZ, ctypes.c_char_p, SZ], I32),
    "ze_shader": ([U32, ctypes.c_char_p, SZ, P, SZ], SZ),
    "ze_gpu_counts": ([P, u32p], None),
    "ze_profile": ([P, U32], I32),
    "ze_profile_report": ([P, ctypes.c_char_p, SZ], SZ),
}


# Optional capability exports preserve loading older libraries with the V3 ABI.
_OPTIONAL_SIGS = {
    "ze_overlap_optimizer_version": ([], U32),
    "ze_overlap_profile_version": ([], U32),
    "ze_overlap_stage_gate_version": ([], U32),
    "ze_overlap_profile_selection_version": ([], U32),
    "ze_oxford_detector_attitude_version": ([P], U32),
    "ze_overlap_profile_trace_capacity": ([P,SZ,U32,U32],SZ),
    "ze_overlap_cells_v6": ([P,u64p,SZ,U32,U32,U32,U32,ctypes.POINTER(OverlapComponent),SZ,ctypes.POINTER(OverlapState),SZ,ctypes.POINTER(OverlapProgress),SZ,f32p,SZ,f32p,SZ,i32p,SZ,ctypes.POINTER(OverlapProfileOptions),f32p,SZ,ctypes.POINTER(OverlapProfileEvent),SZ,ctypes.POINTER(SZ)],I32),
    "ze_overlap_trajectory_version": ([], U32),
    "ze_overlap_cells_v4": ([P,u64p,SZ,U32,U32,U32,ctypes.POINTER(OverlapComponent),SZ,ctypes.POINTER(OverlapState),SZ,ctypes.POINTER(OverlapProgress),SZ,f32p,SZ,f32p,SZ],I32),
    "ze_overlap_cells_v5": ([P,u64p,SZ,U32,U32,U32,U32,ctypes.POINTER(OverlapComponent),SZ,ctypes.POINTER(OverlapState),SZ,ctypes.POINTER(OverlapProgress),SZ,f32p,SZ,f32p,SZ,i32p,SZ],I32),
    "ze_overlap_inspect_v4": ([P,U64,ctypes.POINTER(OverlapComponent),SZ,U32,f32p,SZ,f32p,SZ],I32),
    "ze_overlap_fit_v4": ([P,f32p,f32p,U32,U32,U32,U32,f32p,f32p,ctypes.POINTER(OverlapState),f32p,SZ,f32p,SZ,U32,f32p,SZ],I32),
}


def load():
    lib_name, wgpu_name = _names()
    repo = HERE.parent.parent  # source checkout: <repo>/python/zebsdsolve
    lib_path = _find("ZEBSDSOLVE_LIB", lib_name, [repo / "zig-out" / "bin", repo / "zig-out" / "lib"])
    wgpu_dirs = sorted((repo / ".deps" / "wgpu-native").glob("*/lib")) if (repo / ".deps").exists() else []
    if sys.platform == "win32":
        wgpu_dirs = [d for d in wgpu_dirs if "win" in d.parent.name]
    elif sys.platform == "darwin":
        wgpu_dirs = [d for d in wgpu_dirs if "macos" in d.parent.name]
    else:
        wgpu_dirs = [d for d in wgpu_dirs if "linux" in d.parent.name]
    import platform
    arm = platform.machine().lower() in ("arm64", "aarch64")
    wgpu_dirs = [d for d in wgpu_dirs if ("arm64" in d.parent.name or "aarch64" in d.parent.name) == arm] or wgpu_dirs
    wgpu_path = _find("ZEBSDSOLVE_WGPU", wgpu_name, wgpu_dirs)
    lib = ctypes.CDLL(str(lib_path))
    for name, (args, res) in _SIGS.items():
        fn = getattr(lib, name)
        fn.argtypes, fn.restype = args, res
    for name, (args, res) in _OPTIONAL_SIGS.items():
        fn = getattr(lib, name, None)
        if fn is not None:
            fn.argtypes, fn.restype = args, res
    return lib, str(wgpu_path)
