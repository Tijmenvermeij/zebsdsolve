#!/usr/bin/env python3
"""Locked native-MP virtual overlap study. See docs/overlap-virtual-study.md."""
from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
SEEDS = (315, 714)
FRACTIONS = (0., .1, .25, .5, .75, .9, 1.)
PAIRS = ("A_A", "A_B")
CONDITIONS = ("flat_noiseless", "flat_poisson", "bg_gain_noiseless", "bg_gain_poisson")
METHODS = ("spherical", "dictionary")
POSE_THRESHOLD_DEG = 2.
SPURIOUS_WEIGHT_THRESHOLD = .05
SETTINGS = dict(binning=1, full_resolution="none", static_background="none",
                dynamic_background="none", dynamic_sigma=.15, mask=False,
                clahe=False, bandpass=False, nlpar=False, master_smooth=0,
                bandwidth=64, dict_grid=40, refine=True, refine_iters=16, sweeps=1)
GEOMETRY = dict(pc=(.44, .24, .53), sample_tilt_deg=70, detector_tilt_deg=0,
                rot=(0, 0, 0), px_um=560, step_x_um=0, step_y_um=0)


def safe(value):
    if isinstance(value, np.ndarray):
        return safe(value.tolist())
    if isinstance(value, np.generic):
        return safe(value.item())
    if isinstance(value, dict):
        return {str(k): safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [safe(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path, value):
    path.write_text(json.dumps(safe(value), indent=2, allow_nan=False) + "\n")


def write_csv(path, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(safe(v)) if isinstance(v, (list, tuple, dict)) else safe(v)
                             for k, v in row.items()})


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def standardized(patterns):
    """Equal full-rectangle population-standard-deviation contrast basis."""
    a = np.asarray(patterns, np.float64)
    mean = a.mean(axis=(-2, -1), keepdims=True)
    std = a.std(axis=(-2, -1), keepdims=True)
    if np.any(std <= 1e-12) or not np.isfinite(a).all():
        raise ValueError("A supplied MP produced a flat or nonfinite truth projection")
    return (a - mean) / std


def choose_assignment(pred_phase, truth_phase, present, errors, threshold=POSE_THRESHOLD_DEG):
    """Distinct phase-compatible matching; maximize recovered poses, then coverage.

    errors[t,p] is computed externally using the engine's native disorientation.
    Missing and wrong-phase candidates cannot make recovery denominators disappear.
    """
    best = None
    for assignment in itertools.product((-1, 0, 1), repeat=len(present)):
        used = [p for p in assignment if p >= 0]
        if len(set(used)) != len(used):
            continue
        if any(p >= 0 and (pred_phase[p] < 0 or pred_phase[p] != truth_phase[t]
                          or not np.isfinite(errors[t, p])) for t, p in zip(present, assignment)):
            continue
        count = sum(p >= 0 and errors[t, p] <= threshold for t, p in zip(present, assignment))
        key = (-count, -len(used), sum(errors[t, p] for t, p in zip(present, assignment) if p >= 0), assignment)
        if best is None or key < best[0]:
            best = (key, assignment, count)
    return dict(zip(present, best[1])), best[2]


def check_metrics():
    assignment, count = choose_assignment([0, 0], [0, 0], [0, 1], np.array([[30, .2], [.1, 40]]))
    assert assignment == {0: 1, 1: 0} and count == 2
    _, count = choose_assignment([0, -1], [0, 0], [0, 1], np.array([[.1, np.inf], [20, np.inf]]))
    assert count == 1
    assignment, count = choose_assignment([1, 0], [0, 1], [1], np.array([[0, 0], [.2, 0]]))
    assert assignment == {1: 0} and count == 1  # absent pure component is not scored
    _, count = choose_assignment([1, 1], [0, 0], [0, 1], np.zeros((2, 2)))
    assert count == 0
    # A discovery exception is a failed recovery, not an observed unresolved fit.
    case = dict(index=0, case_id="simulated_engine_failure", mixed=True, f_a=.5,
                phase_a=0, phase_b=0, pose_a=[1,0,0,0], pose_b=[0,1,0,0])
    class NoNativeCall:
        def disorientation(self, *args, **kwargs):
            raise AssertionError("An engine-failure metric must not invent native poses")
    fixed = dict(prepared_fractions=np.array([[.5,.5]]), objective_valid=np.array([True]), sse=np.array([0.]))
    failed = metrics(NoNativeCall(), [case], {"engine_error":np.array(["simulated GPU exception"])}, fixed, "none", "spherical")[0]
    assert failed["addition_unresolved"] is None and failed["attribution_unresolved"] is None
    assert not failed["present_pose_recovered"] and not failed["fraction_eligible"]
    summary = aggregate([failed],["policy","method"])[0]
    assert summary["mixed_cases"] == 1 and summary["both_pose_recovered"] == 0 and summary["engine_errors"] == 1
    assert summary["addition_unresolved"] == 0 and summary["addition_assessed"] == 0
    assert summary["attribution_unresolved"] == 0 and summary["attribution_assessed"] == 0


def make_design(args, policies):
    return dict(schema="overlap_virtual_study_v1", seeds=SEEDS, fractions_a=FRACTIONS,
                pair_types=PAIRS, conditions=CONDITIONS, methods=METHODS,
                policies=policies, n_patterns=112, shape=[args.size, args.size],
                counts=args.counts, requested_contrast=args.contrast,
                positivity_margin=.2, pose_threshold_deg=POSE_THRESHOLD_DEG,
                spurious_weight_threshold=SPURIOUS_WEIGHT_THRESHOLD,
                settings=SETTINGS, geometry=GEOMETRY, batch=args.batch,
                master_a=str(args.master_a.resolve()), master_b=str(args.master_b.resolve()),
                pose_rng="PCG64 SeedSequence([seed,pair_index,0x504f5345]); two normalized Gaussian quaternions",
                noise_rng="PCG64 SeedSequence([seed,pair_index,fraction_index,condition_index,0x504f4953])",
                forward_model="lambda=N*(B+C*g*(f*zA+(1-f)*zB))",
                spatial_fields="x,y linspace[-1,1]; B=1+.10*x+.05*y^2; g=1+.20*x",
                gain_convention="contrast-only spatial multiplier; neither photon nor digital gain",
                fraction_convention="injected equal-contrast contribution; not physical volume fraction",
                baseline_policy="none disables existing dynamic background correction to isolate model bias",
                secondary_policy="subtract uses existing dynamic subtraction at sigma=.15*width; no settings tuning",
                selection="all cases retained, including missing, unresolved, misassigned, and ambiguous poses")


def generate(solver, args, out):
    n, h = 112, args.size
    solver.set_patterns(np.zeros((1, n, h, h), np.float32))
    solver.set_geometry(**GEOMETRY)
    bases, qbase, pbase, cases = [], [], [], []
    for seed in SEEDS:
        for pair_i, pair in enumerate(PAIRS):
            rng = np.random.default_rng(np.random.SeedSequence([seed, pair_i, 0x504F5345]))
            q = rng.normal(size=(2, 4))
            q = (q / np.linalg.norm(q, axis=1, keepdims=True)).astype(np.float32)
            phase = np.array([0, 0 if pair == "A_A" else 1], np.int32)
            base_i = len(bases)
            # Same declared native geometry at every cell: both scan steps are zero.
            sims = np.stack([solver.simulate(q[j:j+1], cells=[len(cases)], phase=int(phase[j]))[0]
                             for j in range(2)])
            bases.append(sims)
            qbase.append(q)
            pbase.append(phase)
            for fi, fraction in enumerate(FRACTIONS):
                for ci, condition in enumerate(CONDITIONS):
                    index = len(cases)
                    cases.append(dict(case_id=f"s{seed}_{pair}_f{fraction:.2f}_{condition}",
                                      index=index, cell=index, seed=seed, pair=pair, base_index=base_i,
                                      fraction_index=fi, condition_index=ci, condition=condition,
                                      f_a=fraction, f_b=1-fraction, minor_fraction=min(fraction, 1-fraction),
                                      mixed=0 < fraction < 1, phase_a=int(phase[0]), phase_b=int(phase[1]),
                                      pose_a=q[0].tolist(), pose_b=q[1].tolist(),
                                      noise_seed_entropy=[seed, pair_i, fi, ci, 0x504F4953]))
    bases, qbase, pbase = np.stack(bases), np.stack(qbase), np.stack(pbase)
    z = standardized(bases)
    ygrid, xgrid = np.meshgrid(np.linspace(-1, 1, h), np.linspace(-1, 1, h), indexing="ij")
    background = 1 + .10*xgrid + .05*ygrid*ygrid
    gain = 1 + .20*xgrid
    negative_extent = max(0., -float(z.min()))
    contrast_bound = .8*min(1., float(background.min()))/(max(1., float(gain.max()))*negative_extent)
    contrast = min(args.contrast, contrast_bound)
    lock = dict(chosen_contrast=contrast, conservative_contrast_bound=contrast_bound,
                minimum_background=min(1., float(background.min())), maximum_gain=max(1., float(gain.max())),
                z_min=float(z.min()), z_max=float(z.max()),
                guaranteed_lambda_lower_bound=args.counts*(min(1., float(background.min())) - contrast*max(1., float(gain.max()))*negative_extent))
    # Written BEFORE any noisy/noiseless measurement is generated or any result is inspected.
    write_json(out/"contrast-lock.json", lock)
    measurements, expectation = [], []
    for case in cases:
        zi = z[case["base_index"]]
        signal = case["f_a"]*zi[0] + case["f_b"]*zi[1]
        nuisance = case["condition"].startswith("bg_gain")
        lam = args.counts*((background if nuisance else 1.) + contrast*(gain if nuisance else 1.)*signal)
        if not np.isfinite(lam).all() or np.any(lam <= 0):
            raise ValueError("Global contrast bound failed; no clipping or per-pattern rescaling is allowed")
        noisy = case["condition"].endswith("poisson")
        rng = np.random.default_rng(np.random.SeedSequence(case["noise_seed_entropy"]))
        measured = rng.poisson(lam) if noisy else lam
        case.update(expectation_min=float(lam.min()), expectation_max=float(lam.max()),
                    measurement_min=float(measured.min()), measurement_max=float(measured.max()), poisson=noisy)
        measurements.append(measured.astype(np.float32))
        expectation.append(lam)
    measurements, expectation = np.stack(measurements), np.stack(expectation)
    np.savez_compressed(out/"truth.npz", measurements=measurements, expectation=expectation,
                        base_projections=bases, base_standardized=z, base_quaternions=qbase,
                        base_phase=pbase, case_base=np.array([c["base_index"] for c in cases]),
                        case_cells=np.arange(n, dtype=np.uint64), fractions_a=np.array([c["f_a"] for c in cases]),
                        background=background, contrast_gain=gain, x=xgrid, y=ygrid)
    write_json(out/"cases.json", cases)
    write_csv(out/"cases.csv", cases)
    lock.update(actual_lambda_min=float(expectation.min()), actual_lambda_max=float(expectation.max()))
    return cases, measurements, qbase, pbase, lock


def discovery(solver, policy, method, n, batch, destination):
    solver.configure(dynamic_background=policy, method=method)
    arrays, metadata = {}, None
    errors = np.full(n, "", dtype="U2048")
    started = time.perf_counter()
    for first in range(0, n, batch):
        cells = np.arange(first, min(first+batch, n), dtype=np.uint64)
        try:
            result = solver.overlap_cells(cells)
            for key, value in result.items():
                if isinstance(value, np.ndarray):
                    if key not in arrays:
                        fill = np.nan if value.dtype.kind == "f" else (False if value.dtype.kind == "b" else (-1 if value.dtype.kind == "i" else ("engine_error" if value.dtype.kind in "US" else 0)))
                        arrays[key] = np.full((n, *value.shape[1:]), fill, dtype=value.dtype)
                    arrays[key][cells] = value
            current = {key:value for key,value in result.items() if not isinstance(value, np.ndarray)}
            if metadata is None:
                metadata = current
            elif safe(current) != safe(metadata):
                raise RuntimeError("Batch metadata changed within a locked method/policy")
        except Exception as error:
            errors[cells] = f"{type(error).__name__}: {error}"
            print(f"ERROR {policy}/{method} cells {cells.tolist()}: {errors[cells[0]]}", flush=True)
        print(f"{policy}/{method}: {min(first+batch,n)}/{n}", flush=True)
    arrays["engine_error"] = errors
    np.savez_compressed(destination/f"discovery-{method}.npz", **arrays)
    metadata = dict(metadata or {}, wall_seconds=time.perf_counter()-started, attempted=n,
                    engine_error_count=int(np.count_nonzero(errors)), timing_is_benchmark=False)
    write_json(destination/f"discovery-{method}.json", metadata)
    return arrays, metadata


def truth_diagnostic(solver, policy, cases, qbase, pbase, destination):
    # Runs AFTER discovery. Truth poses/B/g never seed the indexing searches.
    solver.configure(dynamic_background=policy)
    prepared = np.stack([solver.pattern(c["cell"], processed=True) for c in cases])
    base_proj = []
    for bi, (quats, phases) in enumerate(zip(qbase, pbase)):
        cell = next(c["cell"] for c in cases if c["base_index"] == bi)
        base_proj.append(np.stack([solver.simulate(quats[j:j+1], cells=[cell], phase=int(phases[j]))[0]
                                   for j in range(2)]))
    templates = np.stack(base_proj)[[c["base_index"] for c in cases]].transpose(1,0,2,3)
    result = solver.fit_overlap_projections(prepared, templates)
    arrays = {key:value for key,value in result.items() if isinstance(value, np.ndarray)}
    np.savez_compressed(destination/"truth-pose-diagnostic.npz", prepared_targets=prepared,
                        projected_templates=templates, **arrays)
    write_json(destination/"truth-pose-diagnostic.json", dict(
        label="Fixed true-pose numerical diagnostic; never a discovery result or search seed",
        target="actual Solver.pattern(processed=True) at this policy",
        templates="actual Solver.simulate at true phase/pose/cell under this policy",
        numerical_seam="centers/unit-normalizes supplied prepared targets/templates; no nonlinear reprocessing",
        geometry=solver.geometry, settings=solver.settings,
        metadata={key:value for key,value in result.items() if not isinstance(value, np.ndarray)}))
    return arrays


def metrics(solver, cases, result, truth_fit, policy, method):
    rows = []
    for case in cases:
        i = case["index"]
        row = dict(policy=policy, method=method, **case)
        row["engine_error"] = str(result["engine_error"][i])
        truth_phase = [case["phase_a"], case["phase_b"]]
        truth_q = [case["pose_a"], case["pose_b"]]
        present = [0,1] if case["mixed"] else ([0] if case["f_a"] == 1 else [1])
        valid = not row["engine_error"]
        phases = result["phase"][i] if valid else np.array([-1,-1])
        fractions = result["prepared_fractions"][i] if valid else np.array([np.nan,np.nan])
        errors = np.full((2,2), np.inf)
        if valid:
            for t in present:
                for p in range(2):
                    if phases[p] == truth_phase[t] and np.isfinite(result["quaternions"][i,p]).all():
                        errors[t,p] = float(np.degrees(solver.disorientation(result["quaternions"][i,p:p+1], np.array(truth_q[t])[None], phase=truth_phase[t]))[0])
        assignment, recovered = choose_assignment(phases, truth_phase, present, errors)
        complete = recovered == len(present)
        matched = {p for t,p in assignment.items() if p >= 0 and errors[t,p] <= POSE_THRESHOLD_DEG}
        objective_valid = bool(result["objective_valid"][i]) if valid else False
        finite_fraction = objective_valid and np.isfinite(fractions).all()
        fraction_eligible = complete and finite_fraction
        estimate_a = np.nan
        if fraction_eligible:
            estimate_a = float(fractions[assignment[0]]) if 0 in present else 1-float(fractions[assignment[1]])
        spurious = float(sum(fractions[p] for p in range(2) if p not in matched)) if finite_fraction else np.nan
        for t,label in enumerate(("a","b")):
            p = assignment.get(t, -1)
            row[f"assigned_candidate_{label}"] = p
            row[f"pose_error_{label}_deg"] = float(errors[t,p]) if p >= 0 else np.nan
        tf = truth_fit["prepared_fractions"][i]
        truth_valid = bool(truth_fit["objective_valid"][i]) and np.isfinite(tf).all()
        row.update(present_components=len(present), pose_recovered_count=recovered,
                   present_pose_recovered=complete, both_pose_candidates_recovered=bool(case["mixed"] and complete),
                   unrecovered_present_count=len(present)-recovered,
                   missing_candidate_count=max(0,len(present)-int(np.count_nonzero(phases>=0))),
                   false_proposal_count=int(np.count_nonzero(phases>=0))-len(matched),
                   fraction_eligible=fraction_eligible, estimated_f_a=estimate_a,
                   fraction_error_a=estimate_a-case["f_a"], absolute_fraction_error_a=abs(estimate_a-case["f_a"]),
                   spurious_prepared_weight=spurious, spurious_weight_assessed=finite_fraction,
                   spurious_weight_gt05=bool(spurious > SPURIOUS_WEIGHT_THRESHOLD) if finite_fraction else None,
                   objective_valid=objective_valid,
                   addition_unresolved=bool(result["addition_unresolved"][i]) if valid else None,
                   attribution_unresolved=bool(result["attribution_unresolved"][i]) if valid else None,
                   conditioned=bool(result["conditioned"][i]) if valid else None,
                   candidate_count=int(result["candidate_count"][i]) if valid else 0,
                   fitted_count=int(result["fitted_count"][i]) if valid else 0,
                   solve_status=str(result["solve_status"][i]) if valid else "engine_error",
                   mixture_sse=float(result["sse"][i]) if valid else np.nan,
                   truth_pose_fraction_eligible=truth_valid,
                   truth_pose_estimated_f_a=float(tf[0]) if truth_valid else np.nan,
                   truth_pose_fraction_error_a=float(tf[0]-case["f_a"]) if truth_valid else np.nan,
                   truth_pose_sse=float(truth_fit["sse"][i]),
                   truth_pose_objective_valid=bool(truth_fit["objective_valid"][i]))
        rows.append(row)
    return rows


def mean_finite(values):
    a = np.asarray(values, float)
    return float(a[np.isfinite(a)].mean()) if np.isfinite(a).any() else np.nan


def aggregate(rows, keys):
    groups = {}
    for row in rows:
        key = tuple(row[k] for k in keys)
        groups.setdefault(key, []).append(row)
    output = []
    for key, group in groups.items():
        mixed = [r for r in group if r["mixed"]]
        pure = [r for r in group if not r["mixed"]]
        eligible = [r for r in group if r["fraction_eligible"]]
        mixed_eligible = [r for r in mixed if r["fraction_eligible"]]
        pure_weight = [r for r in pure if r["spurious_weight_assessed"]]
        output.append(dict(zip(keys,key), cases=len(group), mixed_cases=len(mixed),
            both_pose_recovered=sum(r["both_pose_candidates_recovered"] for r in mixed),
            both_pose_recovery_rate=sum(r["both_pose_candidates_recovered"] for r in mixed)/len(mixed) if mixed else np.nan,
            pure_cases=len(pure), pure_present_pose_recovered=sum(r["present_pose_recovered"] for r in pure),
            pure_false_proposals=sum(r["false_proposal_count"] for r in pure),
            pure_spurious_weight_assessed=len(pure_weight),
            pure_spurious_gt05=sum(r["spurious_weight_gt05"] for r in pure_weight),
            fraction_scope="all_cases_including_pure_controls",
            fraction_eligible=len(eligible), fraction_coverage=len(eligible)/len(group),
            mixed_fraction_eligible=len(mixed_eligible),
            mixed_fraction_coverage=len(mixed_eligible)/len(mixed) if mixed else np.nan,
            mixed_fraction_mae=mean_finite([r["absolute_fraction_error_a"] for r in mixed_eligible]),
            mixed_fraction_bias=mean_finite([r["fraction_error_a"] for r in mixed_eligible]),
            fraction_mae=mean_finite([r["absolute_fraction_error_a"] for r in eligible]),
            fraction_bias=mean_finite([r["fraction_error_a"] for r in eligible]),
            truth_pose_fraction_eligible=sum(r["truth_pose_fraction_eligible"] for r in group),
            truth_pose_fraction_bias=mean_finite([r["truth_pose_fraction_error_a"] for r in group]),
            truth_pose_fraction_mae=mean_finite([abs(r["truth_pose_fraction_error_a"]) for r in group]),
            engine_errors=sum(bool(r["engine_error"]) for r in group),
            addition_unresolved=sum(r["addition_unresolved"] is True for r in group),
            addition_assessed=sum(r["addition_unresolved"] is not None for r in group),
            attribution_unresolved=sum(r["attribution_unresolved"] is True for r in group),
            attribution_assessed=sum(r["attribution_unresolved"] is not None for r in group),
            conditioned=sum(r["conditioned"] is True for r in group),
            conditioning_assessed=sum(r["conditioned"] is not None for r in group),
            unrecovered_present_components=sum(r["unrecovered_present_count"] for r in group),
            missing_candidate_components=sum(r["missing_candidate_count"] for r in group)))
    return output


def plot(rows, cases, measurements, policies, out):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
    except ImportError:
        return dict(generated=False, reason="matplotlib unavailable; numerical artifacts complete")
    plt.rcParams.update({"figure.facecolor":"white", "axes.facecolor":"white", "savefig.facecolor":"white",
                         "text.color":"#17212b", "axes.labelcolor":"#17212b", "xtick.color":"#17212b",
                         "ytick.color":"#17212b", "font.size":10, "axes.spines.top":False, "axes.spines.right":False})
    colors = dict(zip(CONDITIONS, ["#2166ac","#67a9cf","#b2182b","#ef8a62"]))
    styles = {"spherical":"-", "dictionary":"--", "truth_pose":":"}
    fig, axes = plt.subplots(2,len(policies),figsize=(9*len(policies),9),squeeze=False,sharey="row")
    for col, policy in enumerate(policies):
        for condition in CONDITIONS:
            for method in METHODS:
                subset = [r for r in rows if r["policy"] == policy and r["method"] == method and r["condition"] == condition]
                mixed = [r for r in subset if r["mixed"]]
                x = sorted(set(round(r["minor_fraction"],8) for r in mixed))
                y = [np.mean([r["both_pose_candidates_recovered"] for r in mixed if abs(r["minor_fraction"]-v)<1e-9]) for v in x]
                axes[0,col].plot(x,y,styles[method],color=colors[condition],marker="o",ms=4)
                y = [mean_finite([r["fraction_error_a"] for r in subset if r["f_a"] == v and r["fraction_eligible"]]) for v in FRACTIONS]
                axes[1,col].plot(FRACTIONS,y,styles[method],color=colors[condition],marker="o",ms=3)
            fixed = [r for r in rows if r["policy"]==policy and r["method"]==METHODS[0] and r["condition"]==condition]
            y = [mean_finite([r["truth_pose_fraction_error_a"] for r in fixed if r["f_a"]==v]) for v in FRACTIONS]
            axes[1,col].plot(FRACTIONS,y,styles["truth_pose"],color=colors[condition],lw=2)
        axes[0,col].set(title=f"Dynamic background: {policy}\nBoth pose candidates within 2° — all mixed cases", xlabel="Injected minor equal-contrast contribution",ylabel="Recovery fraction",ylim=(-.03,1.03))
        axes[1,col].axhline(0,color="#888888",lw=.7)
        axes[1,col].set(title="Fraction bias: discovery matched cases only; truth poses all cases",xlabel="Injected equal-contrast contribution of A",ylabel="Mean estimated A fraction − injected A fraction")
        for ax in axes[:,col]:
            ax.grid(alpha=.18)
    handles = [Line2D([0],[0],color=colors[c],label=c) for c in CONDITIONS]
    cohort_labels = {"spherical":"Spherical: matched cases only", "dictionary":"Dictionary: matched cases only", "truth_pose":"Truth poses: all cases"}
    handles += [Line2D([0],[0],color="#17212b",ls=styles[m],label=cohort_labels[m]) for m in (*METHODS,"truth_pose")]
    fig.legend(handles=handles,loc="lower center",ncol=4 if len(policies)>1 else 2,frameon=False)
    fig.suptitle("Virtual MP study · 112 frozen patterns · two seeds · A/A and A/B pairs\nDifferent fraction cohorts are explicit; coverage and all failures in summary.csv",fontsize=13)
    fig.tight_layout(rect=(0,.10,1,.92))
    fig.savefig(out/"summary.png",dpi=170)
    plt.close(fig)
    selected = [c for c in cases if c["seed"]==SEEDS[0] and c["pair"]=="A_B" and c["f_a"]==.5]
    fig, axes = plt.subplots(1,4,figsize=(14,4),layout="constrained")
    vmax = max(float(measurements[c["index"]].max()) for c in selected)
    for ax, case in zip(axes,selected):
        im = ax.imshow(measurements[case["index"]],cmap="gray",vmin=0,vmax=vmax)
        ax.set_title(case["condition"],fontsize=10)
        ax.set_axis_off()
    fig.colorbar(im,ax=axes,shrink=.72,label="Recorded counts / noiseless expectation")
    fig.suptitle(f"Same seed {SEEDS[0]}, A/B poses and fA=0.5 · shared count scale · no clipping",fontsize=12)
    fig.savefig(out/"example-patterns.png",dpi=170)
    plt.close(fig)
    return dict(generated=True, files=["summary.png","example-patterns.png"], example_vmin=0,example_vmax=vmax)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,default=Path("/Users/tvermeij/EMsoftData/overlap_virtual_study_20261116"))
    parser.add_argument("--master-a",type=Path,default=ROOT/"testdata/nickel/Ni_master_20kV.npz")
    parser.add_argument("--master-b",type=Path,default=ROOT/"testdata/gan/GaN_master_20kV.npz")
    parser.add_argument("--library",type=Path)
    parser.add_argument("--reference-study",type=Path,help="assert every frozen truth array matches an earlier locked study byte-for-byte before inference")
    parser.add_argument("--size",type=int,default=48)
    parser.add_argument("--counts",type=float,default=200.)
    parser.add_argument("--contrast",type=float,default=.2)
    parser.add_argument("--batch",type=int,default=28)
    parser.add_argument("--dynamic-background",choices=("none","subtract"),default="none",
                        help="none is an isolated baseline; subtract uses existing background correction")
    parser.add_argument("--also-subtract",action="store_true",help="also run subtract on the SAME frozen measurements")
    parser.add_argument("--check-metrics",action="store_true")
    args = parser.parse_args()
    check_metrics()
    if args.check_metrics:
        print("metric known-answer checks passed")
        return
    if args.size < 8 or args.size*args.size > 65536 or args.counts<=0 or not 0<args.contrast<=1 or args.batch<=0:
        parser.error("size 8..256, positive counts/batch, and contrast in (0,1] required")
    if args.library:
        os.environ["ZEBSDSOLVE_LIB"] = str(args.library.resolve())
    import zebsdsolve as ze
    policies = list(dict.fromkeys([args.dynamic_background]+(["subtract"] if args.also_subtract else [])))
    args.output.mkdir(parents=True,exist_ok=False)
    design = make_design(args,policies)
    write_json(args.output/"design.json",design)
    started = time.perf_counter()
    with ze.Solver() as solver:
        solver.configure(**SETTINGS,batch=args.batch)
        solver.add_phase(args.master_a)
        solver.add_phase(args.master_b)
        library = Path(solver._lib._name).resolve()
        manifest = dict(design=design, status="generating", native_library=str(library), native_library_sha256=sha256(library),
                        native_version=ze.__version__, adapter=solver.adapter, numpy_version=np.__version__,python=sys.version,
                        master_importer="native Solver.add_phase (native NPZ/EMsoft importer by content)",
                        master_hashes=[dict(path=str(p.resolve()),sha256=sha256(p)) for p in (args.master_a,args.master_b)],
                        script_sha256=sha256(Path(__file__)),
                        conventions="native w,x,y,z sample-to-crystal; native proper phase symmetry; geometry tilt once in native projection",
                        worktree_head=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),
                        limits=["Same native forward model for generation and inference: internal consistency only, not physical validation",
                                "Equal-contrast injection is not physical volume fraction; contrast-only gain is not photon/digital gain",
                                "Only two pose pairs per pair type across two locked seeds; no statistical generalization",
                                "Existing GaN polar competing poses may fail native proper-phase pose matching",
                                "No-background baseline deliberately disables existing correction; nuisance bias is not all missing gain fitting",
                                "No search tuning, blur/gain parity, GUI, acceptance calibration, or backend changes"])
        write_json(args.output/"manifest.json",manifest)
        cases, measured, qbase, pbase, contrast = generate(solver,args,args.output)
        manifest.update(contrast_lock=contrast,geometry=solver.geometry,phases=solver.phases,truth_sha256=sha256(args.output/"truth.npz"))
        if args.reference_study:
            reference_manifest = json.loads((args.reference_study/"manifest.json").read_text())
            assert manifest["native_library_sha256"] == reference_manifest["native_library_sha256"], "Reference native library changed"
            assert manifest["master_hashes"] == reference_manifest["master_hashes"], "Reference masters changed"
            assert safe(design) == reference_manifest["design"], "Locked design changed"
            matched_arrays = []
            with np.load(args.reference_study/"truth.npz",allow_pickle=False) as prior, np.load(args.output/"truth.npz",allow_pickle=False) as current:
                assert prior.files == current.files, "Truth schema changed"
                for name in prior.files:
                    assert prior[name].dtype == current[name].dtype and prior[name].shape == current[name].shape, name
                    assert prior[name].tobytes() == current[name].tobytes(), f"Frozen truth array changed: {name}"
                    matched_arrays.append(name)
            manifest["reference_validation"] = dict(path=str(args.reference_study.resolve()),
                arrays_byte_identical=matched_arrays, measurements_byte_identical=True,
                source_truth_sha256=reference_manifest["truth_sha256"],
                npz_archive_byte_identical=manifest["truth_sha256"] == reference_manifest["truth_sha256"],
                design_unchanged=True, native_library_unchanged=True, masters_unchanged=True)
            print(f"Reference study: all {len(matched_arrays)} frozen arrays byte-identical; design/library/masters unchanged",flush=True)
        # Freeze one original acquisition array for BOTH preprocessing policies.
        solver.set_patterns(measured[None])
        assert solver.geometry == manifest["geometry"]
        rows, method_metadata = [], {}
        for policy in policies:
            destination = args.output/policy
            destination.mkdir()
            results = {}
            for method in METHODS:
                results[method], method_metadata[f"{policy}/{method}"] = discovery(solver,policy,method,len(cases),args.batch,destination)
            fixed = truth_diagnostic(solver,policy,cases,qbase,pbase,destination)
            for method in METHODS:
                rows.extend(metrics(solver,cases,results[method],fixed,policy,method))
        assert sha256(library) == manifest["native_library_sha256"], "Native library changed during locked study"
        assert sha256(args.output/"truth.npz") == manifest["truth_sha256"]
        manifest["method_metadata"] = method_metadata
    write_json(args.output/"metrics.json",rows)
    write_csv(args.output/"metrics.csv",rows)
    summary = aggregate(rows,["policy","method","condition","pair","seed"])
    pooled = aggregate(rows,["policy","method","condition"])
    overall = aggregate(rows,["policy","method"])
    write_json(args.output/"summary.json",dict(by_condition_pair_seed=summary,pooled=pooled,overall=overall))
    write_csv(args.output/"summary.csv",summary)
    write_csv(args.output/"summary-pooled.csv",pooled)
    manifest["plots"] = plot(rows,cases,measured,policies,args.output)
    manifest.update(status="complete",measurements=len(cases),discovery_case_rows=len(rows),
                    wall_seconds=time.perf_counter()-started,
                    artifact_hashes={str(p.relative_to(args.output)):sha256(p) for p in sorted(args.output.rglob("*")) if p.is_file() and p.name!="manifest.json"})
    write_json(args.output/"manifest.json",manifest)
    for item in overall:
        print(json.dumps(safe(item),sort_keys=True),flush=True)
    print(f"Saved {len(cases)} frozen measurements, {len(rows)} discovery rows to {args.output}",flush=True)


if __name__ == "__main__":
    main()
