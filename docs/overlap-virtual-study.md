# Virtual master-pattern study

`scripts/overlap_virtual_study.py` generates a locked 112-pattern pilot and runs
both native spherical and dictionary overlap discovery. It tests internal
consistency of this backend with its own master-pattern projection model. It does
not validate physical overlap fractions, real detector statistics, or the legacy
GUI blur/gain objective. The separate real Map1 baseline remains necessary.

```sh
PYTHONPATH=python ZEBSDSOLVE_LIB="$PWD/zig-out/lib/libzebsdsolve.dylib" \
  python scripts/overlap_virtual_study.py --also-subtract \
  --output "$STUDY_OUT"
```

Set `STUDY_OUT` to a new output directory before running the example.
NumPy and the built native library are required. Matplotlib is optional and writes
white scientific PNGs when installed. There are no new core dependencies. The
output directory must be new. `--check-metrics` runs the pure assignment
known-answer checks without a GPU. `--master-a` and `--master-b` accept native
master files; defaults are the repository nickel and GaN NPZ masters.

The pilot has two locked seeds (315, 714), two pair types (A/A and A/B), seven
injected A contributions (0, .1, .25, .5, .75, .9, 1), and four conditions (flat
noiseless, flat Poisson, varying background/contrast gain noiseless, and varying
background/contrast gain Poisson). All 112 measurements are generated once and
retained. The same frozen measurements can run under two separate preprocessing
policies. `--dynamic-background none` is the default **isolated baseline**, which
deliberately disables the framework's existing background correction.
`--dynamic-background subtract` uses that existing correction; `--also-subtract`
adds it as a second policy without regenerating measurements or tuning settings.
Policy results and denominators remain separate. Background-case bias in the
isolated baseline must not all be attributed to a missing gain model.

Each MP is loaded natively. Two seeded unit quaternions define each pose pair in
the existing native sample-to-crystal convention; no Euler/frame conversion or
symmetry folding is performed. Native `Solver.simulate` projects at the declared
geometry. Both spatial scan steps are zero, so every case has the same detector
position. Generation and inference share this geometry and the native masters.
The full rectangular projection `Pj` becomes `zj=(Pj-mean(Pj))/std(Pj)` with
population standard deviation. The injected contribution is an **equal-contrast
basis weight**, not a physical volume fraction.

The expected counts are

`lambda = N * (B + C * g * (f*zA + (1-f)*zB))`.

Defaults are N=200 and requested C=.2. Flat cases use B=g=1. Nuisance cases use
x spanning left-to-right columns [-1,1] and y spanning top-to-bottom rows [-1,1], `B=1+.10*x+.05*y²`, and `g=1+.20*x`. This is a spatial
multiplier of the contrast term only; it is neither photon gain nor digital gain.
One C is fixed over the whole declared suite before measurements are generated:

`C=min(requested_C, .8*min(B)/(max(g)*max(0,-min(zj))))`.

The extrema cover every individual standardized truth projection and both field
policies. Convex mixtures therefore retain a positive expectation with a declared
20% background margin. There is no clipping, per-image scaling, or result-based
contrast change. Poisson seeds are independent per case and separate from pose
seeds. Noiseless cases retain their count expectation. All float32 measurements
and float64 expectations are saved, along with native projections, standardized
bases, phases, poses, fields and case IDs; NPZ files load with `allow_pickle=False`.

Both searches use locked bandwidth 64, dictionary grid 40, 16 refinement
iterations, full pattern resolution, no mask, CLAHE, bandpass, NLPAR or static
background, and zero master smoothing. The selected dynamic-background policy is
the only preprocessing difference. Default patterns are 48×48; settings and full
native metadata are retained. Background B, gain g, fractions and truth poses are
never supplied to discovery. All cases, including failures, remain in its
outputs. Batch errors are recorded for every affected case and are not removed
from recovery denominators. Unresolved/conditioned flags from failed engine calls
are unavailable, not asserted true; summaries count only observed flags and report
assessment counts separately.

A separate truth-pose numerical diagnostic runs **after discovery**, once per
preprocessing policy. It uses the actual native prepared targets from
`Solver.pattern(processed=True)` and native true-pose projections under that
policy, then the existing supplied-projection numerical fit. That seam only
centers/unit-normalizes its supplied vectors again; it does not rerun nonlinear
filters. It receives truth poses solely to assess the model, never to seed the
searches. Flat noiseless results under policy `none` check the injected
contributions; nuisance cases and policy `subtract` quantify the prepared-model
bias under the actual preprocessing convention.

Recovery uses native, phase-aware proper-symmetry `Solver.disorientation`, with
a declared 2° threshold and distinct permutation matching for A/A pairs. Matching
maximizes recovered poses, then phase-compatible assignment coverage, then minimizes
angular error. All mixed cases remain in the both-pose recovery denominator.
Recovery is **pose-candidate recovery**, without a physical detection/acceptance
claim. Pure endpoints score only the present component. Additional/misassigned
proposals and unmatched prepared weight above 5% are diagnostic; they are not
physical false detections. Spurious-weight assessment coverage is reported.

Fraction MAE/bias is conditional on correct present-component pose assignment,
a valid retained objective, and finite fractions. `fraction_*` summaries include
pure controls and mixed cases; `mixed_fraction_*` separately reports conditional
MAE/bias and eligible count/coverage over **all mixed cases**. Eligible count and
coverage are always reported alongside recovery, missing components, unresolved additions,
conditioned fits and engine errors. For pure endpoints, the matched present
component's weight estimates its injected contribution; unmatched weight counts
as contamination. The truth-pose diagnostic has its own eligibility and is not
conditioned on search success. GaN polar competing poses may reconstruct well
while failing the native proper-phase angular criterion; those failures remain.

Artifacts include preregistered `design.json`, `contrast-lock.json`, provenance
and hashes in `manifest.json`, `truth.npz`, case JSON/CSV, full native per-policy
and per-method outputs, the separate truth-pose diagnostics, all-case metric
JSON/CSV, and grouped summary JSON/CSV by policy, method, condition, pair and seed.
`summary.png` shows both-pose recovery versus minor contribution and conditional
fraction bias with explicitly different cohorts: discovery means use matched
cases only, while fixed truth poses use all valid cases. Their mean gap cannot be
attributed solely to search error; cohort selection also differs. `example-patterns.png` uses the
same seed, poses, f=.5, and shared count scale across all four conditions.

Only two pose pairs per pair type are sampled. This pilot characterizes named
cases and model/preprocessing effects; it does not estimate population-wide
performance. There are no backend changes or settings tuning in this study.

For a reporting-only rerun, pass `--reference-study "$PRIOR_STUDY_OUT"` with a new
output directory. Before inference this asserts byte-identical frozen arrays and
unchanged design, native library and MP hashes. The first pilot is preserved.

The 2026-11-16 pilot used the locked 112 measurements under both policies, producing
448 discovery rows with no engine errors or unresolved fits. Over the **80 mixed
cases** per method/policy, the named-case results were:

| Background policy | Search | Both pose candidates recovered | Conditional mixed fraction MAE | Eligible mixed fraction fits |
| --- | --- | ---: | ---: | ---: |
| none | spherical | 77/80 | .004818 | 77/80 |
| none | dictionary | 56/80 | .005078 | 56/80 |
| subtract | spherical | 80/80 | .004283 | 80/80 |
| subtract | dictionary | 58/80 | .004191 | 58/80 |

For the 32 pure controls, unmatched prepared weight exceeded 5% in 9 and 12 cases
without dynamic correction (spherical/dictionary), and 1 and 4 with subtraction.
These are extra-proposal diagnostics, not physical false detections. The clean,
no-background fixed-pose check had maximum fraction error 5.96e-8. The v2 report
retains the first pilot and verifies all 13 frozen truth arrays byte-for-byte,
with unchanged settings, native binary and master hashes. The report's mixed-only
MAEs above exclude the pure controls; the original all-case MAEs remain labeled
in the summaries. Discovery and all-case truth-pose means use different cohorts.
