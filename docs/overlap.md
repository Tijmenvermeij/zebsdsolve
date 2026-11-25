# Prepared-domain overlap candidates

Python `Solver.overlap_cells(cells, max_components=2, primary="search")` and browser
`await ze.overlapCells(cells, {maxComponents: 2, primary: "search"})` discover one
through four candidate components. Existing spherical, dictionary and PCA search
schedules supply proposals; spherical is the current development focus. Detector
refinement supplies candidate poses. Every amplitude fit uses the original,
once-prepared pattern, on the GPU, with nonnegative coefficients. This does not
replace ordinary stored indexing results or change the acquisition.

## Optional orientation profiling and residual screening (V6)

The Overlap tab exposes **Polish orientations** and **Screen residual NCC**.
**Polish orientations** and **Refit blur / gain after polishing** are off by
default. **Screen residual NCC** is on by default in the Overlap tab.
Direct C V6 calls require polishing, screening, or both;
use the existing V4/V5 entry points when both controls are off.
Screening requires an explicitly entered threshold;
**Show only** remains a separate display filter. The numerical controls are also
available as:

```python
result = solver.overlap_cells(
    cells, max_components=2, keep_n=5, blur_gain=True,
    polish=True, min_residual_ncc=None, polish_radius_deg=1.0,
)
```

```javascript
const result = await ze.overlapCells(cells, {
  maxComponents: 2, keepN: 5, fitBlurGain: true,
  polish: true, minResidualNcc: null, polishRadiusDeg: 1,
});
```

V6 keeps the detector geometry fixed and evaluates orientation changes using the
same shared gain/blur and nonnegative amplitude objective as the final mixture.
The experimental target is prepared once. No new background filter is applied
while polishing or forming residuals. Each support gets its initial shared
blur/gain fit before angular trials. During polishing, blur/gain stays fixed and
the nonnegative amplitudes are refitted at each trial. When both fitting and
polishing are enabled, **Refit blur / gain after polishing** optionally runs one
more shared fit at the best moved pose. The equivalent API options are
`refit_blur_gain_after_polish` in Python, `refitBlurGainAfterPolish` in JavaScript,
and flag value 8 in C. Orientation polishing with blur/gain requires V6 profile
version 2. The GPU scores the six neighbors of an angle together and reuses
the unchanged prepared components and their inner products. Initial primary
polishing can batch independent scan points. Later residual searches still
follow each point's accepted support.
V6 compacts zero coefficients: `active_count` is the strictly positive support
size, while `fitted_count` retains the legacy `>1e-6` display convention. Inner
nuisance-search evaluation bounds are not bounds on the whole profiled run.
The search is bounded around immutable
native proposal seeds; the radius is in degrees. Existing spherical/dictionary
proposal routines are reused. A better achieved lower-order explanation replaces
the current support and triggers a fresh residual search; it is not counted as an
extra component. Higher-order comparisons include drop-one supports. These are
bounded best-achieved comparisons, not global-optimum guarantees. Selection version 2
archives the complete best fit at each order. A restart can change the next
search residual while publication retains any stronger fit already evaluated.

`min_residual_ncc` / `minResidualNcc` screens each native refined proposal against
its pre-addition signed residual **before** expensive joint fitting. It requires a
finite score greater than or equal to the configured float32 threshold. If every
proposal fails, that search branch stops without adding a component.
The screen uses the residual-indexing NCC in the sense of [Cios et al., Section 2.3](https://arxiv.org/html/2601.14155v3#S2.SS3). Their empirical 0.175–0.20 thresholds refer to their own DTM and detector processing. The companion subtraction code also records residual-versus-primary NCC, a separate diagnostic.
There is no automatic minimum physical fraction or universal NCC threshold.
The exact executed threshold and requested value are exported separately.

`quaternions` contains the final fitted poses. `source_quaternions` retains the
native proposal poses; `proposal_ncc` is the historical discovery score before
joint polishing. A replaced primary can have a residual-derived score, so V6
component slots do not imply discovery order. `profile_events` preserves rejected
proposals, replacements, their scan-cell IDs and the selected trajectory. Event bit
256 marks the event that produced the published fit when Options bit 4 is
requested (the current Python and browser SDKs request it automatically).
Bit 512 indicates that
the fit came from a singleton or a reduced support in that event. Add and
replace bits and the stop reason describe the search path. Final fit arrays
hold the complete selected pose, coefficients and gain/blur.
The profiled `best_single_sse` can describe a different achieved pose from the
final mixture; incremental improvement compares the strongest achieved strictly
lower-order model across both histories, without transplanting its components. The
screen is a computational rule, not calibrated statistical confidence. All
solutions remain **unassessed candidates** until a matching independent
calibration is established.

Validate quality strata separately: good Cu TKD, moderate Ti TKD, poor Re TKD
(including possible thickness-related band inversion), and Re EBSD require
separate geometry, master-energy and preprocessing provenance. A high contribution
or a lower in-sample error does not establish a second physical component. The
coarse spherical PC estimator also needs independent detector-plane validation;
a higher fitted score by itself does not certify the recovered PC.


## Optional ΔSSE gate for successive additions

Leave all stage fields blank to use ordinary overlap discovery. To use the gate,
enable **Screen residual NCC** and enter a minimum ΔSSE for each addition through
**Maximum components**. A value of 0.01 requires a reduction of at least 1% of the
original prepared pattern's squared energy. A value of 0 still requires a positive
improvement beyond the numerical fit tolerance.

Each search history proceeds from its own accepted support. After the candidate
NCC screen, the best positive joint fit in that round competes with the strongest
valid smaller fit evaluated in that history, including smaller fits evaluated
during the round. A failed stage ends that history. Drop-one fits can
strengthen the comparison without becoming a new search seed. The exact and
pooled histories are independent. The final result selects from singletons
and fits that passed their own chain of gates. A newly evaluated singleton
can win publication if it fits better than every eligible joint fit.

These settings do not estimate statistical confidence. Use trusted K−1 component
controls to set the K threshold, with matched preparation, geometry, NCC screen,
candidate search, blur/gain, and polishing settings. For a calibration run,
explicitly enter zero thresholds, then examine the round-level traces of false
additions, counting histories and trials as part of the same pixel-level
procedure. Use additional held-out controls to verify the resulting cutoff for
each stage. Thresholds are intentionally blank by default.

Python accepts `min_stage_improvement=[k2, k3, k4]`; use `None` for stages above
`max_components`. Browser SDK uses `minStageImprovement` with `null` for unused
stages. Both require a residual NCC minimum. `stage_gate` records executed
float32 limits and the comparison definition. Event mask 1024 identifies the
best joint trial that was checked, with the actual comparison reference in
`lower_sse`; mask 2048 marks a pass. The executed score is
`(lower_sse - joint_sse) / initial_sse`. Stop code 13 is insufficient improvement.
The map's incremental ΔSSE retains its separate cross-history diagnostic meaning.


The model is `fixed_prepared_projection_nnls`, minimizing
`||y - sum(a[k] s[k])||²` with `a[k]>=0`. The original target and projected templates
have zero mean and unit L2 norm under the native detector mask. Amplitudes and
normalized fractions describe that **prepared-pattern domain**, not raw intensity
or physical volume fractions. Negative residual pixels remain signed f32. A
separate normalized residual copy goes directly into the existing prepared GPU
search/refinement seam. Background removal, CLAHE and NLPAR never run a second
time on residuals. Chance-z and ordinary primary acceptance are not residual
scores and are absent from overlap output.

Every phase proposal is reprojected at the actual original cell geometry and
compared through its joint fit against the original target. Invalid/sentinel
proposals are excluded. The full normalized Gram matrix passes a conservative
f32 rank guard (diagonal-pivoted Schur complements, pivot margin `1e-5`). Each
feasible subset, independent single-template fit and final objective is scored
directly from signed pixel residuals to avoid energy cancellation.

The spectral condition ratio is `lambda_min/lambda_max` of the **whole** normalized
Gram matrix. K1 is exactly one; K2 uses `(1-|rho|)/(1+|rho|)`; K3/4 use a bounded
10-sweep symmetric cyclic Jacobi calculation on the GPU. It checks finite output,
remaining off-diagonal magnitude and PSD tolerance and fails closed. No suitable
existing GPU eigensolver was available in this backend; this small linear-algebra
operation adds no crystallographic mathematics. Fractions are withheld when the
ratio is at or below `(1-.995)/(1+.995)`, including collective conditioning that
pairwise correlations alone miss. Component identities and feasible amplitudes
remain available for otherwise valid but ill-conditioned fits.

Supplied-projection `Solver.fit_overlap_projections(patterns, projections)` is the
numerical seam for one through four fixed templates. Unreliable designs return
`objective_valid=False`, `solve_status="numerically_unresolved"` and NaN joint
amplitudes/fractions/SSE/NCC/residual/improvement, retaining independently valid
`single_sse` and `best_single_sse`. Discovery excludes unreliable provisional
objectives from ranking, tries other phases and retains the last valid order when
no reliable addition exists. Historical skipped-candidate flags and the current
addition's stop reason remain separate. Flat targets report unassessed
conditioning and NaN condition ratios. Nonfinite prepared targets fail closed.

Legacy V3–V5 discovery slots have stable identities: `[cell, discovery_slot]`, with a trailing
quaternion axis `(w,x,y,z)` in the engine's native sample-to-crystal convention.
Browser arrays are flat cell-major typed arrays. `discovery_id` is the fixed slot
index, or -1 for an absent record. Refitting can set an earlier distinct template's
amplitude to zero; that record retains its phase/quaternion/slot, participates in
later full-design fits and can reactivate. Zero amplitudes do not create a zero
Gram column. In V3–V5, `active_count` counts amplitudes above `1e-6` for display only; that
threshold never modifies coefficients or SSE. A new zero-weight or non-improving
proposal is a boundary trial: all phases are compared before stopping that cell.
A failed addition never creates a hole followed by a later component. Flat-residual
stopping happens only between orders: every competing phase in the current order
is compared, even after a provisional near-perfect match.

Cells stop independently within a batch. `retained_order` counts retained proposal
records, legacy `active_count` counts displayed positive contributors, and
`proposed_count` counts attempted component additions. Discovery is bounded to
`max_components` passes. `previous_sse` and `incremental_improvement` describe the
last attempted order; `best_single_sse` and `improvement_over_best_single` compare
against the better independent single-template fit. These are numerical
candidate diagnostics, not statistical or physical component acceptance.

Stop codes (`OVERLAP_STOP_REASONS` / browser `overlapStopReasons`) are:

| Code | Name |
| ---: | --- |
| 0 | running |
| 1 | component_limit |
| 2 | flat_residual |
| 3 | no_proposal |
| 4 | unresolved_addition |
| 5 | boundary |
| 6 | missing_primary |
| 7 | flat_target |
| 8 | attempt_limit (reserved) |
| 9 | invalid_target |
| 10 | stale_primary |

`primary="stored"` reuses the exact ordinary stored winner (`res.quat`/`phase`),
without reindexing, chance calibration or primary refinement. A computed valid
best winner remains usable when its accepted sweep is -1. Stored seed conditioning is tracked separately from ordinary export validity.
`Results.seed_context` plus a per-cell proof bitmap records the context in which
a winning pose/phase/raw NCC/NMI payload was actually committed. Only ROI
coordinates are ignored by `sameSeedContext`; every other `sameExportContext`
field remains strict. Whole-map background preparation is independent of ROI.
Source ROI is preserved as `seed_source_roi`, the conditioning cohort's anchor
ROI (zero width/height means full map); current requested ROI remains in settings.
Changing only selection neither invalidates a valid raw seed nor changes its
payload. Chance-z, sweep acceptance labels and all normal export/checkpoint
context guards remain strict and unchanged.

A real geometry/preprocessing/phase change fails stored reuse up front with
`StoredPrimaryContextStaleRerunIndex`. Starting normal indexing under new
conditioning invalidates only proof bits globally, preserving all primary
payloads outside the fresh ROI. Requested fresh cells lose proof until an actual
winner commits; an unvisited or cancelled cell never gains proof. A nonwinning
later sweep preserves the proof of its retained earlier winner. Uncomputed cells
return `missing_primary`; positive old payloads without current proof return
`stale_primary`; neither gets a usable objective or fresh-search fallback. If no
requested seed has valid proof, no patterns are uploaded. Missing/stale/non-evaluated rows carry NaN fit, rank,
condition and improvement diagnostics, with assessment flags false; an absent
component's proposal NCC/NMI are NaN. The same unavailable row has identical
semantics alone or in a mixed-validity batch. A valid retained lower-order fit
keeps its objective when only a new addition is unresolved. Verified homogeneous
checkpoint restore rebinds proof after all transactional validation succeeds,
without changing the serialized format. Failed restore leaves existing proof
unchanged; mixed-context checkpoints remain rejected by the existing guards.

Original primary arrays remain unchanged by overlap. `phase_generation` / browser
`phaseGeneration` is a cheap synchronous native phase-asset epoch for caller
context checks, independent of warm-up/index scheduling. Stored mode prepares
residual-search resources and uploads original patterns once, without primary
search, calibration or refinement.

Browser helpers:

```js
const r = await ze.overlapCells(cells, {maxComponents: 4, primary: "stored"});
const rgba = ze.orientationColors(r.quaternions, r.phase, "z");
const images = await ze.overlapInspect(cell, r, row, selectedComponent);
```

`orientationColors` calls existing native `Key`/rotation logic, returning RGBA and
transparent absent slots. It uses the same IPF settings as ordinary native maps;
there is no browser crystal-color implementation. `overlapInspect` regenerates
five images at the current native context: `measured`, `conditional` (original y
minus all other contributions), `selected_contribution`, `composite`, and signed
final `residual`, plus `[height,width]`. All share original prepared-target units.
An absent selected slot has zero contribution and conditional equals final
residual. Zero-weight retained records preserve their identities. No residual
stack is cached. The caller must freeze/validate result settings, geometry,
acquisition and phase identity before inspection. Python equivalents are
`orientation_colors(...)` and `overlap_inspect(cell, result, row, selected_component)`.

All stages must use one effective resolution. Set `binning=1` for stored
pixels, or choose **Full size for → Nothing** (`full_resolution="none"`) with a
reduction factor to use reduced pixels. Mixed levels fail up front.
Patterns are bounded to 131,072 pixels (including the 256 × 311 Re acquisition). The four-component
bank and state allocations are checked against device storage binding/buffer
limits; use smaller batches for an oversized request. Moments use 64 lanes per
pattern, residual writes 256; small solves, explicit subset SSEs and selected-bank
copies use one lane. No production-scale throughput claim is made. Cancellation
is checked between passes/readbacks and throws without returning a partial batch.

The old `ze_overlap_cells` ABI and 32-byte Component / 64-byte State layouts remain.
`ze_overlap_cells_v2` adds primary mode and 32-byte per-cell Progress. A separate
UI result map can own batch commits/cancellation without persisting an overlap
map in the engine. Tests cover float64 NNLS/eigvalsh oracles, permutations,
collective conditioning, rank failure, native spherical three/four-pose cases,
heterogeneous stopped orders, zero-weight reactivation, stored-context coverage,
primary preservation, native IPF parity and inspection identities.

The app's Overlap workflow uses the existing normal Index settings and can reuse
compatible stored primary winners or run normal indexing first. ROI selection is
context-safe and does not invalidate physically compatible seed conditioning.
Legacy component maps keep discovery slots stable; selecting a pixel regenerates the
prepared-domain inspection images above. Result batches commit only on successful
completion; cancelled or stale work does not become a partial result tile.
Component export is versioned results JSON with provenance/arrays and nested
metadata, distinct from settings-project JSON. There is no component-result
reimport yet. The result context is frozen against real input, preprocessing,
geometry and phase-asset changes before inspection/export.

This remains a prepared-domain candidate model. V3–V5 use fixed poses; V6
optionally profiles orientations as described above. Calibrated physical
component acceptance and legacy GUI objective parity are not claimed. The retained GaN seed315/cell2
regression has a dictionary polar competing pose near 180° under native proper
symmetry despite correct phase and useful reconstruction. Such search ambiguity
remains visible; DI has not been tuned for overlap. The separate Map1 and virtual
study artifacts document their original scope and binary fingerprints.


## V4 contrast transfer and basis weights

`overlap_cells` / `overlapCells` default to the explicit V4 contrast model,
`fit_model="contrast_gain_v1"`, `gain_domain="contrast"`. Python accepts
`gain_domain="source_dc"`; JS accepts `gainDomain:"source_dc"` to request the
legacy source-DC path. The supplied-projection numerical seam preserves its legacy
source-DC default and requires `gain_domain="contrast"` for V4. Older native/WASM
builds without V4 reject a requested contrast fit clearly.

For each reference T_j and shared theta, V4 computes:

```
v_j = gaussian_sigma(T_j)                    # full image, including outside mask
mu_j = masked_mean(v_j)
x_j = masked_center(G_theta * (v_j - mu_j))
d_j = masked_L2(x_j)
z_j = x_j / d_j
prepared_prediction = sum_j a_j * z_j        # a_j >= 0, existing NNLS
beta_j = a_j / d_j
contrast_basis_weights_j = beta_j / sum(beta)
```

The original experimental target is prepared once. There is no additional
background filter on the gained templates, residuals or refinement targets.
SI search preparation remains independent. Its harmonic/master preparation
supplies orientation seeds; it does not define the final mixture objective.
Detector-plane zero-normalized NCC removes a constant offset and a positive
uniform scale, not a spatial gain or background. Scalar masked centering and L2
normalization are therefore distinct from dynamic background subtraction. The
existing orientation proposal refinement is followed by blur/gain and amplitude
fitting; the current pipeline does not jointly refine all component orientations
with those fitted profiles. Gain does not restore the removed source DC. Blur never receives a premasked image; masked means are recomputed for
the actual blurred source column and sigma. A separate private GPU buffer first
subtracts the actual upper masked median over the full image for numerical
stability. The GPU selects it with four 8-bit radix passes over ordered finite
f32 values, once per raw stage. Post-blur masked means use short accumulation
streams and tree reductions, without an exterior-pixel anchor. Empty or one-pixel
masks have no valid contrast fit. Since the Gaussian preserves constants and masked centering follows
it, this changes no contrast field or declared reference units; it prevents large
representable pedestals from contaminating blur or gain arithmetic. Original raw
and committed templates remain unchanged. Final `component_norms` are recomputed
from committed raw templates and theta in private scratch buffers, after candidate
competition finishes. Rejected/stopped additions cannot change retained norms.

`amplitudes` and `prepared_fractions` retain their original unit-template meaning.
`contrast_basis_weights` undo numerical column normalization; discovery declares
`weight_basis="globally_standardized_baked_master_contrast"`, while supplied
projections declare `weight_basis="supplied_projection_contrast"`. Reference
scaling changes beta inversely and therefore changes basis weights; adding an
independent constant pedestal does not change the contrast model. A common
mean-one versus maximum-one gain gauge cancels in normalized basis weights.

Discovery records master smoothing, CLAHE, bandpass and per-phase source/bake
moments in `contrast_basis_preparation`. These weights refer to that declared
baked-master basis. Cross-phase absolute intensity transfer and experimental
contrast calibration remain unresolved; the weights are not raw-camera, volume
or physical phase fractions. Any invalid retained norm, flat target or unreliable
rank makes the whole weight row NaN. Absent slots are NaN; retained zero amplitudes
have zero weight when the remaining valid coefficients have a positive sum.

With blur/gain optimization off, V4 evaluates identity theta through the same
contrast transform and NNLS path as optimized fitting, norm output and inspection:
one identity evaluation per required independent singleton and joint fit. It does
not route through the legacy normalization/NNLS implementation. V3's ordinary and
source-DC fit paths retain their own exact behavior. Inspection selects V4
from `fit_model`, including identity fits. Frozen results without that tag, or
with `source_dc_gain_v1`, continue through legacy V3/ordinary inspection. V3's
source-DC transformation is unchanged. V4 uses the same optimizer version 2 and
104/96 joint/singleton evaluation budgets described below; it changes the declared
transfer model and certifies nonnegative amplitude faces without changing the
orientation search/refinement procedures. Improved coefficients and residuals may
legitimately change subsequent proposed poses.


V4 checks KKT conditions on every final clamped candidate coefficient vector,
using `g=c-G*a` and
`tau_j=32*eps32*(sqrt(G_jj*yTy)+abs(c_j)+sum_l(abs(G_jl)*a_l))`.
Positive coefficients require `abs(g_j)<=tau_j`; zero coefficients require
`g_j<=tau_j`. Materially negative/nonfinite solutions are rejected, the zero face
is not feasible by default, and no certified face yields unresolved NaN outputs.
The existing whole-Gram rank guard remains. Eight short accumulation streams and
fixed tree reductions compute V4 moments and direct pixel SSE; feasible ties use
dual error, then stable subset order. V3's face selection and reductions remain
unchanged. Metadata records the V4 solver, tolerance and anchor method.

Only at the supplied-projection V4 seam, an f64 masked scalar centering pass before
GPU normalization protects independently shifted, representable target values.
This introduces no spatial background operator and does not change experimental
index preparation. Reference/source anchoring, target centering and normalization
retain their declared intensity/contrast units and do not calibrate physical
fractions.


## Legacy V3 source-DC blur and spatial gain

Explicit Python `gain_domain="source_dc"` and JavaScript `gainDomain:"source_dc"`
use V3 when blur/gain is enabled, and V2 when disabled. Ordinary indexing and
proposal NCC/NMI are unchanged.
V3 retains the 32-byte Component, 64-byte State and 32-byte Progress layouts and
returns a separate seven-float theta per cell. The model name is
`shared_blur_gain_prepared_projection_nnls`.

For fixed discovered poses and geometry, the same theta transforms every template:

```
theta = (sigma, log_rho, power, axis_y, axis_x, offset_y, offset_x)
r = sqrt(((row - h/2 - offset_y*h)/(axis_y*h/2))^2
       + ((col - w/2 - offset_x*w)/(axis_x*w/2))^2)
shape = max(1-r,0)^power
G = 1 + (exp(log_rho)-1)*shape
s[k] = masked_center_unit_norm(G * gaussian(z[k])
                            + (exp(log_rho)-1)*shape*dc[k])
```

The edge gain is fixed at one, removing global gain/weight degeneracy. Bounds are
sigma [0,5] **effective fit pixels**, rho [.25,4], power [.25,8], both axes [.6,1.4],
and both offsets [-.15,.15] of the corresponding image dimension. These are initial
engineering bounds, not calibrated detector constraints. Gaussian filtering uses
the same separable implementation as GPU preparation, radius ceil(3*sigma),
truncated-edge renormalization and exact sigma-zero bypass. Same-sigma trials reuse
the cached blurred images. Raw flat or nonfinite columns are rejected before gain
can create artificial contrast.

Both native master loaders now record the mean and standard deviation removed
from their expanded Lambert sheets, without changing normalized values. Each
bake also records its pre-final-normalization mean and standard deviation.
For smoothing-only preparation `dc=(bake_mean+source_mean/source_sd)/bake_sd`;
for bandpass or CLAHE, `dc=bake_mean/bake_sd` in the filtered master domain.
This stable formula restores the spatially varying DC contribution without
subtracting a large constant in f32. Identity reproduces the existing prepared
template. Under nonlinear preparation or NLPAR, this gain is a phenomenological
prepared-master-domain adjustment; it is not a raw detector calibration or a
physical mixture fraction.

All raw projections, Gaussian/gain transforms, detector-mask normalization,
NNLS amplitudes, direct pixel SSE comparisons, parameters and trial decisions stay
on the GPU. The host schedules a deterministic coordinate pattern search and
reads final records/parameters. Each joint search evaluates identity, previous
order theta, positive/negative log-rho seeds and cached independent singleton
seeds, followed by two sweeps at each of three decreasing coordinate scales
then four polished singleton seeds and eight final log-rho-only trials at steps ±0.04375, ±0.021875, ±0.0109375,
and ±0.00546875 (104 joint evaluations; 96 per singleton; optimizer version 2).
Wrappers query the loaded library/WASM optimizer version independently of the V3
ABI. Older V3 builds without that capability export retain version-1 metadata
(92 joint / 88 singleton evaluations); unknown versions are rejected.
The fine trials retain the incumbent and accept only strictly smaller actual
NNLS SSE. Current-search coarse singleton theta snapshots seed the joint search,
so polishing the singleton baseline cannot replace its coarse joint trajectory.
The four polished singleton seeds are evaluated after the coarse coordinate
search, retaining both that incumbent and every feasible finer singleton fit.
At fixed supplied poses this retains the version-1 joint incumbent before
polishing; discovery can follow different later poses after an improved earlier
fit. Every trial solves all K amplitudes
against the original immutable prepared target using the existing full-Gram rank
guard and exhaustive nonnegative active sets. A fixed budget yields a local fit;
global optimality or convergence to the true nuisance parameters is not claimed.

Each proposed pose also gets its own independent theta optimization. Any better
independent singleton score achieved at a winning joint theta is folded back into
its score/parameter cache. K1 therefore has exactly zero improvement over best
single; later additions cannot discard that achieved K1 baseline. Proposed-slot
caches reset for each winning phase competitor. The cached single scores form `single_sse` / `best_single_sse`; the shared joint fit is seeded
from their coarse snapshots for the current search (and earlier cached singleton
parameters during discovery), then from the polished singleton parameters after
the coarse coordinate search. Improvement is not measured
against singles conditionally evaluated only at the joint theta. Candidate raw
banks, transformed banks, parameters and phase competition are transactional.
Rejected proposals preserve the last accepted order. K1 uses this fitted model
before its signed normalized residual feeds the native K2 proposal search.
Earlier zero-weight discovery slots stay in later joint fits and may reactivate.

The result includes `blur_gain`, `blur_gain_theta`, `blur_gain_parameter_names`
and `blur_gain_optimizer`. The optimizer metadata records algorithm version,
engineering bounds, evaluation limits, fit-pixel sigma units and prepared-master
interpretation. Inspector V3 consumes the exported theta unchanged and regenerates
the same transformed templates, composite and residual rather than reoptimizing.
The caller must freeze theta with the existing geometry/settings/phase context.

`fit_overlap_projections(patterns, projections, blur_gain=True, mask=None)` is the
scalar numerical seam. Optional mask is `(row, col, radius)` in effective fit
pixels. Its input projections retain their original affine intensity; no hidden
pre-normalization discards their DC. Tests compare returned-parameter transforms,
NNLS coefficients and direct signed residuals with independent SciPy float64
Gaussian/NNLS calculations, including offset ellipses, nonzero DC, masks,
rectangular images, K1–K4, identity and flat/rank-deficient designs. These are scalar
image and linear-algebra tests. No new crystallographic mathematics or alternate
orientation import path was introduced; native master projection and proper
symmetry paths are preserved, so MTEX has no applicable replacement in this new
scalar optimizer.


For the opt-in nuisance solver, feasible subset coefficients are solved in one
lane, while every direct signed-pixel residual square and singleton score is
accumulated over 64 lanes with a fixed reduction tree. Subsets retain ascending
order and strict score comparisons for deterministic ties. This avoids serial
full-image SSE loops during repeated nuisance trials without using cancellation-
prone energy subtraction. The original V2 kernel is unchanged. On the local
Apple M4 Pro, an eight-cell 256 × 310 K2 supplied-projection fit (including two
independent singleton optimizations and the joint fit) took approximately 1.05 s;
throughput depends on the GPU and proposal-search settings.


## Focused maps and display filtering

The Overlap tab enables **Fit blur / gain** and **Screen residual NCC** by
default, using a minimum residual NCC of 0.10. Spherical residual indexing uses
the primary indexing bandwidth by default. Clear **Use same as primary
indexing** to set a separate bandwidth for residual search. The export records
the executed value as `residual_bandwidth`. The controls are specific to spherical
search. The API's blur/gain and screening defaults are unchanged.

The linked maps default to **Original primary** and **First accepted residual
component**. Original primary is available before overlap indexing and covers
the accepted indexed points across the scan, including outside the overlap ROI.
When both maps show IPF colours, choosing X, Y or Z on either map updates both.
Each map can also show its own metric.

**Fit component 1**, **Fit component 2**, and subsequent numbered maps show active
components in the final fitted order using their final orientations. Component 1
includes single-pattern fits and usually retains the original primary. The
original primary stays available on its own map when a fit replaces it.

The **First accepted residual component** map shows the first eligible
residual-derived component in fit order at each point of a multi-component fit.
It uses that component's discovery orientation. **Residual map NCC minimum**
filters this map using the score measured against the earlier search residual,
after residual-only refinement and before joint fitting. **Source NCC** on a
numbered fit map reports the score on that component's original search target.

**Primary NCC minimum** controls the original primary map and which points in
the ROI are eligible for the next overlap run. The execution cutoff is frozen in
the result metadata. **Show only** applies additional weight or fit-improvement
filters to the numbered and residual maps. Display filtering leaves the fitted
results and exported records intact. Reindexing invalidates the earlier overlap
results because it changes their primary snapshot. Mixture NCC measures the
quality of the fitted reconstruction.
The separate computational **Screen residual NCC** control retains its own
frozen execution threshold; lowering a display threshold cannot restore
proposals skipped during computation. Thresholds are user choices, not calibrated
physical-presence confidence.

Hover over map selectors, patterns and scores for definitions and units. The
inspector shows the target, simulation, signed residual and **Gain profile** in a
2 × 2 grid, with the fit details below. The gain caption reports the range of the
shared spatial envelope and the fitted Gaussian sigma in effective fit pixels.
The fitted contribution uses the saved blur, gain and amplitude of the selected
component. The reconstruction sums those contributions. Original primary
inspection uses the index renderer without live reindexing, with a signed
difference independent of the ordinary split-view control. In this view the gain
profile still belongs to the computed mixture.

### Experimental checks and interpretation

The final contrast implementation was checked on 192 frozen Re pixels, 128
Cu comparison pixels, the 347 Cu four-component pixels, and all 1,820 frozen Ti
fits. Orientation comparisons use native MTEX 7.2 imports and established scan
frames. Re uses the old GUI optimized constant PC and the native integrated
master; its old GUI poses are also refitted with the same contrast objective for
a fair pose comparison. These references are comparisons, not ground truth.

A separate measured-image blend control preserves the acquired intensity scale
of both endpoints. All 56 blend weights are recovered within 6e-8 using the
measured endpoints as references. Substituting native simulated references at
frozen indexed poses leaves mean weight errors of 4.75 percentage points without
the fitted transfer and 4.45 points with it. Lower image SSE does not guarantee
more accurate weights. Reference/experimental contrast response, orientation, PC
and preprocessing mismatch remain possible causes; endpoint purity is not
certified.

Controlled pure noise still yields large relative weights, despite little
explained energy. Correlated noise mixed with one real simulated pattern can also
yield a sizeable second weight. A ratio among nonnegative fitted coefficients is
not a component-presence test. Signed float range is not the cause: an affine
positive-range control preserves scores and weights to numerical precision.
Calibrated component acceptance and conditional orientation refinement remain
separate work; no universal NCC cutoff is imposed.


## Local regression checks

Run the native Overlap tests with
`PYTHONPATH=python:tests python -m unittest discover -s tests -p 'test_overlap*.py'`
after building the native library. The optional historical trajectory comparison
reads frozen `old_dictionary.npz` and `old_spherical.npz` from
`ZEBSDSOLVE_REFERENCE_BASELINE`. Use references captured with matching geometry,
settings and primary seeds. `ZEBSDSOLVE_REFERENCE_ARTIFACTS` can retain the current
capture for diagnosis.

## Residual candidate pools (V5)

`Solver.overlap_cells(..., keep_n=N)` and JavaScript `overlapCells(..., {keepN:N})` accept integer N from 1 through 8. N applies **per phase, per residual order**. The default is 1. The ordinary first search or stored primary remains unchanged; a one-component request follows V4 regardless of N. N=1 follows the corrected V4 path exactly. Existing V2/V3 exports retain their legacy behavior.

For dictionary proposals, the overlap-private search keeps the exact top N raw inner products across all native dictionary chunks, ordered by descending score then ascending global shuffled dictionary index. It uses the normal configured search resolution and dictionary grid, without angular filtering or a new coarse/full resolution schedule. Every seed is refined at the actual cell position against the same signed normalized residual, using the normal detector mask. The immutable refined pool is arranged internally as `[phase][rank][cell]`. Missing ranks never enter refinement.

Spherical pools preserve the ordinary native first peak and fill additional ranks from normalized 26-neighbor local maxima. The overlap-private selector streams slabs, collapses pole aliases, applies proper left-crystal-symmetry separation at a radius of two native grid steps, and refills paged results after suppression. It reuses native peak interpolation and float correlation data. It does not quantize the correlation volume or replace the ordinary search.

Each candidate is then fitted jointly with the retained K−1 raw projections against the original once-prepared experimental target. The retained singleton seed/SSE baseline is frozen before alternatives are evaluated. No target or residual receives another background filter. The selected pose, phase, proposal scores, shared parameters, amplitudes and component norms come from the same winning alternative. Exact-SSE ties use raw projection identity; numerically identical projections remain indistinguishable at this seam.

Results include `keep_n`, `candidate_method`, `candidate_seed_policy` and the cell/component array `winner_rank`: −1 means missing, 0 means the ordinary primary, and 1…N identifies the residual seed's rank within its phase before refinement. Phase labels remain in the separate `phase` array. `proposed_count` counts attempted orders, never candidate ranks. These fields describe candidate generation and do not supply physical phase acceptance criteria.

N>1 with more than one requested component requires V5 support in the loaded native library or WASM module. Legacy `source_dc` and PCA proposals reject that configuration explicitly. Small same-resolution batches remain appropriate. Spherical candidate pools additionally require that a single item's spectrum and normalization volume fit the actual adapter's buffer and binding limits; unsupported bandwidths fail explicitly before first-search work, without reducing bandwidth or changing method.


## Rank-one reference protection

For V5 requests with `keep_n > 1` and at least three requested components, trajectory version 1 runs two independent histories from the same prepared target and frozen primary records: the requested pooled search and the exact N=1 reference. Acquisition reading, target upload/preprocessing and ordinary primary search happen once. Each history owns its fit, residual, candidate and nuisance state. The complete valid payload with the lower final prepared-target SSE is selected per cell; exact ties select the reference. If both histories lack a valid retained fit, the reference unavailable payload is returned. A rejected addition can leave a valid earlier fit eligible for comparison. This preserves the N=1 final-SSE bound; it supplies no physical component acceptance or orientation-truth guarantee.

This is a final choice between two complete histories, without mixing their components. A K3 request and a K4 request can select different histories, so their published prefixes need not agree. Prefixes remain immutable within each individual history. K1, N=1, K2 and legacy V2/V3/V4 numerical paths retain their existing behavior.

`trajectory_source` is a per-cell signed integer: −1 unavailable, 0 ordinary/rank-one reference, 1 pooled. `trajectory_selection` declares `best_valid_final_sse_of_pooled_and_rank1_v1` only for eligible requests when the loaded runtime reports `ze_overlap_trajectory_version() == 1`. Older V5 runtimes are labeled `greedy_pooled_v5`; their reserved progress word is never interpreted as provenance. Unknown nonzero eligible protocol versions fail explicitly. The V5 C signature remains unchanged: after all GPU work, version 1 uses progress offset 28 solely for final host provenance (0 reference, 1 pooled, max-u32 unavailable). That word remains a phase gate while kernels run.

Versioned seed policy identifies the requested candidate generation: `spherical_localmax26_pole_quotient_2_grid_steps_v1`, `dictionary_raw_top_n_v1`, `native_single_winner_v1` for N=1, or `ordinary_primary_only_v1` for K1. Use `trajectory_source` to distinguish the selected reference from the requested pooled history.

Native cancellation uses acquire/release atomic flag access even when Zig is compiled single-threaded, because foreign callers such as Python ctypes can request cancellation from another thread. The current unshared wasm32-freestanding browser runtime uses its single event loop. Cancellation reset points are unchanged.


## Dictionary sampling resolution

`dict_grid` counts cubochoric steps per half edge, not degrees. The conventional
nominal mean spacing estimate is `131.97049 / dict_grid + 0.03732` degrees.
The default grid 30 is therefore about 4.44 degrees. Grid 90 matches the integer
step count selected by the legacy GUI/orix for a requested 1.5-degree resolution;
grid 91 gives an estimated mean of 1.49 degrees, below 1.5. This conversion is
an average-spacing convention, not a measured maximum angular gap or covering
radius. Fundamental-zone boundary samples and ordering can differ between
implementations even at equal step counts.

The final Ti/Re/Cu overlap comparisons used spherical indexing at bandwidth 158,
so `dict_grid` did not affect them. Its padded 512-point correlation grid has
0.703125-degree Euler-axis increments before peak interpolation and detector-plane
orientation refinement. Those increments are not equivalent to a uniform SO(3)
dictionary spacing or a worst-case angular coverage bound.


## Interactive and batch runs

With **Screen residual NCC** enabled, Overlap uses the higher of the primary
NCC minimum and the residual screen as the effective primary minimum. This
sets eligibility from the stored primary results and filters the original
primary map. It does not change the primary indexer's search. The residual
screen applies again at each proposed residual component. The separate
**Residual map NCC minimum** only filters the residual-source map and does
not remove fits from the export.

Set the Overlap options, then enable **Include overlap indexing** before
adding a scan to Batch. Each queued job captures its own Overlap options.
It indexes its primary patterns in its selected region, runs Overlap from
those results, then saves the ordinary outputs and a separate
`_overlap.json` file. Loading a completed Batch job restores the primary
results and queued settings. The JSON contains the published overlap fits
and diagnostics. It does not contain the search state needed to resume a fit
at a higher component order.
