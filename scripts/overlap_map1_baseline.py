#!/usr/bin/env python3
"""Freeze selected experimental/reference records for overlap development.

Acquisitions are opened by zebsdsolve's native format reader. h5py reads only
the custom GUI Step-4 reference schema. Euler arrays are copied unchanged as
opaque reference records: this script does not construct or compare rotations.
No private acquisition or output is added to the repository.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import h5py
import numpy as np


FIELDS = (
    "computed_mask", "fit_success", "primary_phase", "secondary_phase",
    "primary_euler_rad", "secondary_euler_rad", "pattern_center",
    "primary_coefficient", "secondary_coefficient", "primary_fraction",
    "secondary_fraction", "primary_ncc", "secondary_ncc", "mixture_ncc",
    "residual_rms", "component_correlation", "fitted_sigma", "gain_params",
    "ellipse_params", "orientation_refined",
)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def json_value(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, bytes):
        return value.decode("utf-8")
    raise TypeError(type(value).__name__)


def read_stage(path):
    with h5py.File(path) as source:
        attrs = {key: value for key, value in source.attrs.items()}
        if attrs.get("format") != "overlap-ebsd-step4-results-v1":
            raise ValueError(f"Unsupported reference format: {path}")
        shape = (int(attrs["map_rows"]), int(attrs["map_columns"]))
        arrays = {name: source[f"Maps/{name}"][...] for name in FIELDS}
        if any(value.shape[:2] != shape for value in arrays.values()):
            raise ValueError(f"Reference map shape mismatch: {path}")
        mask = arrays["computed_mask"].astype(bool)
        cells = np.asarray(source["Point Results/index"], dtype=np.int64)
        if not np.array_equal(cells, np.flatnonzero(mask)):
            raise ValueError(f"Reference point IDs disagree with C-order map mask: {path}")
        if len(cells) != int(attrs["computed_point_count"]):
            raise ValueError(f"Reference point count mismatch: {path}")
        coefficients = np.stack([arrays[f"{c}_coefficient"] for c in ("primary", "secondary")], -1)[mask]
        fractions = np.stack([arrays[f"{c}_fraction"] for c in ("primary", "secondary")], -1)[mask]
        totals = coefficients.sum(axis=-1)
        valid = totals > 1e-12
        if np.any(coefficients < 0) or not np.all(np.isfinite(coefficients)):
            raise ValueError(f"Invalid nonnegative reference coefficients: {path}")
        if not np.allclose(fractions[valid], coefficients[valid] / totals[valid, None], atol=2e-7, rtol=0):
            raise ValueError(f"Reference fractions disagree with amplitudes: {path}")
        summary = {
            "path": str(path), "sha256": sha256(path), "attributes": attrs,
            "computed_pixels": int(mask.sum()), "orientation_refined_pixels": int(np.count_nonzero(arrays["orientation_refined"][mask])),
        }
        return shape, arrays, summary


def sample_cells(stages, per_group):
    first = stages[0]
    mask = first["computed_mask"].astype(bool).ravel()
    success = first["fit_success"].astype(bool).ravel()
    base = mask & success
    minority = np.minimum(first["primary_fraction"], first["secondary_fraction"]).ravel()
    correlation = np.abs(first["component_correlation"]).ravel()
    groups = {"stage1_fitted": np.flatnonzero(base), "stage1_missing": np.flatnonzero(~mask)}
    for i, stage in enumerate(stages[1:], 2):
        later = stage["computed_mask"].astype(bool).ravel()
        groups[f"stage{i}_with_stage1"] = np.flatnonzero(base & later)
        groups[f"stage{i}_without_stage1"] = np.flatnonzero(~mask & later)
    for name, values, descending in (("small_minor_share", minority, False), ("balanced_share", minority, True), ("correlated_templates", correlation, True)):
        available = np.flatnonzero(base & np.isfinite(values))
        order = np.argsort(-values[available] if descending else values[available], kind="stable")
        groups[name] = available[order[:per_group]]
    selected = {}
    for name, cells in groups.items():
        indices = np.linspace(0, len(cells) - 1, min(per_group, len(cells)), dtype=int) if len(cells) else np.empty(0, int)
        selected[name] = cells[indices].tolist()
    union = sorted({cell for cells in selected.values() for cell in cells})
    return np.asarray(union, np.uint64), selected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scan", type=Path, required=True)
    parser.add_argument("--stage", type=Path, action="append", required=True, help="Original Step-4 HDF5 files, in discovery order")
    parser.add_argument("--master", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True, help="New output directory outside the repository")
    parser.add_argument("--per-group", type=int, default=4)
    parser.add_argument("--audit", type=Path, help="Optional historical audit/summary.json; require matching reference hashes")
    parser.add_argument("--hash-scan", action="store_true", help="Read and hash the full acquisition; otherwise identify by file stat and selected patterns")
    args = parser.parse_args()
    if args.per_group < 1:
        parser.error("--per-group must be positive")
    for path in [args.scan, args.master, *args.stage]:
        if not path.is_file():
            parser.error(f"Missing file: {path}")
    if args.out.exists():
        parser.error("--out must be a new directory to preserve frozen baselines")

    references = [read_stage(path.resolve()) for path in args.stage]
    shape = references[0][0]
    if any(record[0] != shape for record in references):
        raise ValueError("Reference stages describe different map shapes")
    if args.audit:
        expected = {stage["file"]: stage["sha256"] for stage in json.loads(args.audit.read_text())["stages"]}
        for _, _, summary in references:
            if expected.get(Path(summary["path"]).name) != summary["sha256"]:
                raise ValueError(f"Reference differs from historical audit: {summary['path']}")
    cells, groups = sample_cells([record[1] for record in references], args.per_group)

    # Native importer owns acquisition schema, calibration and scan geometry.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
    import zebsdsolve as ze
    with ze.Solver() as solver:
        solver.open(args.scan)
        scan = solver.scan
        if (scan["rows"], scan["cols"]) != shape:
            raise ValueError("Native acquisition shape disagrees with saved GUI maps")
        patterns = np.stack([solver.pattern(int(cell), processed=False, full=True) for cell in cells])
        geometry = solver.geometry
        native_settings = solver.settings
        native_library = Path(solver._lib._name).resolve()

    arrays = {"cells": cells, "row": cells // shape[1], "column": cells % shape[1], "patterns_as_stored_f32": patterns}
    for i, (_, stage, _) in enumerate(references, 1):
        for name, values in stage.items():
            arrays[f"stage{i}_{name}"] = values.reshape((-1, *values.shape[2:]))[cells]
    args.out.mkdir(parents=True)
    fixture = args.out / "selected.npz"
    np.savez_compressed(fixture, **arrays)
    stat = args.scan.stat()
    manifest = {
        "schema": "zebsdsolve-overlap-baseline-v1",
        "map_shape": shape, "selected_pixels": len(cells), "selection_groups": groups,
        "scan": {"path": str(args.scan.resolve()), "size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns,
                 "sha256": sha256(args.scan) if args.hash_scan else None},
        "master": {"path": str(args.master.resolve()), "sha256": sha256(args.master)},
        "importer": {"name": "zebsdsolve native acquisition importer", "version": ze.__version__, "scan": scan,
                     "source_revision": subprocess.check_output(["git", "-C", str(Path(__file__).resolve().parents[1]), "rev-parse", "HEAD"], text=True).strip(),
                     "library": str(native_library), "library_sha256": sha256(native_library),
                     "geometry": geometry, "settings": native_settings, "pattern_options": {"processed": False, "full": True}},
        "reference_stages": [record[2] for record in references],
        "fixture": {"file": fixture.name, "sha256": sha256(fixture),
                    "patterns_sha256": hashlib.sha256(patterns.tobytes()).hexdigest()},
        "conventions": {
            "pixel_ids": "Zero-based original acquisition cell IDs; C-order row * map_columns + column",
            "reference_orientations": "Opaque saved GUI Bunge Euler radians; copied unchanged, no rotation construction or alignment",
            "pattern_center": "Saved GUI pattern-center records retained unchanged; equivalence to native geometry is not asserted",
            "spatial_units": "Reference stage scan_unit and step_x/step_y retained verbatim in attributes; native geometry records its own units",
            "missing_fit": "Unknown, never evidence of a single orientation",
            "contributions": "Saved fitted pattern-intensity weights; later stages split residual budgets, not simultaneous original-pattern fits",
            "acceptance": "Historical fit success does not establish physical overlap or newer acceptance diagnostics",
        },
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2, default=json_value, allow_nan=False) + "\n")
    print(json.dumps({"output": str(args.out), "selected_pixels": len(cells), "reference_computed_pixels": [r[2]["computed_pixels"] for r in references]}))


if __name__ == "__main__":
    main()
