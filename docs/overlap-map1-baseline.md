# Map 1 overlap development baseline

The experimental reference is Cu reference AD, Site 1, Map 1: 210 rows by
454 columns, 5 nm scan steps, 128 by 156 stored pattern pixels. The source is
Oxford H5OINA. The saved overlap GUI Step-4 files contain 38,354 first-stage,
12,862 second-stage and 2,500 third-stage computed fits. These are saved fit
records, not independently validated physical component counts.

`scripts/overlap_map1_baseline.py` creates a small, deterministic private
fixture without copying the acquisition into this repository. Supply paths
explicitly; nothing depends on the developer's home directory:

```sh
python scripts/overlap_map1_baseline.py \
  --scan "$MAP1_SCAN" --master "$MAP1_MASTER" \
  --stage "$MAP1_STAGE1" --stage "$MAP1_STAGE2" --stage "$MAP1_STAGE3" \
  --audit "$MAP1_REPORT/audit/summary.json" --out "$BASELINE_OUT"
```

The output directory must not already exist. Dependencies are NumPy, h5py,
the built zebsdsolve native library and a local GPU. The audit argument is
optional; when supplied, the complete reference-file hashes must match it.
`--hash-scan` additionally hashes the entire acquisition, which otherwise is
identified by path, file size, modification time and selected pattern data.
All masters and Step-4 reference files are hashed in full.

## What is frozen

- `manifest.json`: acquisition identity, master and reference hashes, native
  reader/version/library hash, source revision, native geometry/settings,
  original saved GUI metadata and deterministic selection groups.
- `selected.npz`: original acquisition cell IDs, rows/columns, selected
  unprocessed full-size patterns returned as float32 by the native reader,
  and unchanged reference records for each stage. No pickle is used.

The default selection takes four points per group, with duplicates removed:
first-stage fits, missing first-stage fits, deeper-stage availability with
and without a first-stage fit, smallest minor contributions, balanced
contributions and most correlated first-stage templates. On this dataset it
selects 35 cells. Selections from deeper stages establish saved processing
coverage only; they are not claims of three or four independently identified
orientations. Separate known-answer simulations cover that requirement.

The fixture validates point IDs against each saved C-order computed mask,
map shapes and coefficient/fraction consistency. Missing reference fits
remain unknown. They must not be used as single-component negative controls.

## Import and coordinate conventions

The acquisition uses zebsdsolve's native Oxford importer, including its
calibration and spatial metadata. h5py reads only unsupported custom Step-4
reference fields. Reference Bunge Euler arrays (radians), phase labels and
pattern centers are copied verbatim as opaque records; the fixture generator
does not build rotations, transform frames or compare orientations. Native
and GUI pattern-center conventions are not asserted equivalent.

The manifest retains spatial units and steps from each source. Original
pixel identity is `row * 454 + column`, zero based. No fitted alignment or
additional tilt correction is applied. Before an orientation comparison,
verify the recorded conventions using the existing native crystallographic
tools. This fixture introduces no new crystallographic mathematics and no
MTEX/pyMTEX testing dependency.

The recorded GUI master is `Cu_30kv_EMsOO.h5`, SHA-256
`2875b70c0eac3f9ad7de3fd4f5c86226c47d7f280ef518ce764fd1de887bfaa7`.
It is a constituent Cu master, not a prescribed twin-composite master.

## Comparison gates

1. Verify identical input records and selected raw patterns on regeneration.
2. Check projected intensities at fixed poses and geometry before comparing
   fitted residuals. Master normalization and spatial gain do not commute.
3. Verify GPU signed residuals and original-pattern nonnegative amplitude
   fits against an independent float64 numerical reference.
4. Run both spherical and dictionary residual searches. Preserve original
   cell geometry and distinguish search failures from fitting errors.
5. Assess recovered orientations, reconstruction error and contributions
   together. First match physical components independent of discovery order.

The first GPU pipeline is an explicitly fixed **prepared-pattern projection
fit**. It does not reproduce the GUI's shared blur and spatial gain model.
Its output components are candidates; numerical fit improvement alone is
not a physical-overlap acceptance test. The local smoke run must record its
preparation, search settings, scores and limitations rather than present its
contributions as GUI-equivalent fractions.

The historical deeper-stage weights recursively split earlier residual
budgets. They are not a simultaneous fit of all components against the
original pattern. The new pipeline may therefore produce different fitted
weights. The saved first-stage runs also had no optional mixture orientation
refinement, which provides a baseline before evaluating that extension.

## Initial local run, 16 November 2026

The three source reference files matched the historical audit hashes. The
native importer verified 210 by 454 cells, 128 by 156 uint8 patterns and
5 nm scan steps. The frozen selection contains 35 cells. Private artifacts
remain outside the repository; the manifest records their exact locations.

The source acquisition had moved relative to its recorded GUI metadata;
the current location is recorded explicitly rather than rewriting the saved
source provenance. No GUI agreement or physical overlap validation is
claimed by this baseline creation step.

On four fixed original cells (0, 2, 54946, 79646), ordinary spherical and
dictionary indexing returned bit-identical quaternions, NCC, NMI, refinement
steps and chance-z before and after the prepared-buffer extraction. This
compares the starting local library to the new build, with identical native
geometry and explicit 160 micrometre stored detector pixel pitch, binning 2,
bandwidth 64, dictionary grid 30 and 12 refinement iterations.

The two-component candidate pipeline completed on all 35 baseline cells
through both search methods. An independent float64 SciPy NNLS calculation
used the original prepared targets and native projections at each returned
pose and original cell, with the same circular mask. Maximum amplitude
differences were below 1.1e-7; maximum SSE differences were below 1.5e-6.
These checks validate the numerical prepared-pattern fit, not component
identity or agreement with the legacy blur/gain objective. All output
components remain unassessed candidates.

The existing local GPU API regression suite also passed, covering ordinary
indexing, pattern filters, geometry fitting, master simulation and exports.


## Four-component Cu cohort

The report's `classes/tables.zip` selects 347 pixels with `active_count == 4`
under its 10% support and 5 degree identity criteria. Pixel IDs are converted
from the report's one-based raster IDs to native zero-based map cells, without
orientation conversion. Native acquisition and master import, geometry and
preparation use the settings above.

The four-component spherical search completed with four positive candidate
amplitudes and finite assessed objectives on all 347 pixels. These counts use
the backend's numerical activity threshold; they do not establish 10% support,
distinct crystallographic identities or agreement with the report's orientations.
Independent float64 SciPy NNLS checks at every returned pose gave maximum
amplitude error 1.10e-7 and SSE error 1.96e-6. Regenerated inspection images on
12 distributed pixels reproduced SSE within 1.25e-6.

Ordinary primary indexing on 12 distributed cohort pixels followed by stored
primary reuse preserved all original primary payload arrays bit for bit. The
first component's quaternion was exactly the original stored primary pose in
each case. Sparse ROI changes exercised per-cell conditioning proof reuse.

The private `four_components/` directory contains the complete selected IDs,
search and stored results, reproducible `validation.py`, source-table SHA256,
settings, geometry and native library fingerprint in `summary.json`. This is
four-component pipeline and fixed-pose fit validation. GUI orientation/fraction
agreement, the legacy blur/gain objective and physical acceptance remain to be
validated separately; no frame alignment has been fitted or assumed.
