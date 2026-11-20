# Changelog

What changed in each release, newest first. Work lands under an **Unreleased** heading; a release renames that
heading to the version and date, and the release workflow publishes the section as the release's
notes.

## Unreleased

### Overlap indexing

- Correct the Oxford detector Euler-to-ray conversion and effective-camera
  writeback. Version calibration provenance and checkpoints; reindex and refit
  geometry for pre-fix Oxford results and projects.

- Add opt-in gain-aware orientation profiling at fixed geometry and a separate
  computational residual-NCC screen. Retain proposal poses and rejected events,
  compare lower-order replacements before additions, and preserve legacy defaults.

- Retain 1–8 residual-search candidates per phase with native spherical or
  dictionary indexing. Refine each against the same signed residual before
  comparing joint fits to the original pattern.
- For three or four components, compare pooled and single-candidate histories
  and return the complete lower-error fit. Prepare the input and primary once;
  export the selected history and candidate ranks.
- Make competing overlap fits independent of phase iteration order and make
  native cancellation thread-safe.
- Add one-through-four-component native, Python and browser candidate discovery
  using existing spherical and dictionary searches, signed GPU residuals and
  original-pattern nonnegative projection fitting. Preserve stable component slots
  and support compatible stored-primary reuse.
- Add an Overlap tab with normal primary indexing, component maps, native IPF
  colors, regenerated pattern inspection and versioned result export. Cancelled
  tiles cannot replace completed results; input changes invalidate inspection.
- Add optional GPU fitting of one shared Gaussian blur and positive spatial gain
  for all retained components. Apply gain to simulated contrast after blur, fit
  amplitudes jointly against the once-prepared target, and regenerate inspection
  images from frozen fit parameters. Keep the legacy source-DC model explicit.
- Report model weights after undoing per-template numerical normalization; retain
  original contrast shares for compatibility and label the reference basis.
- Add primary/residual search NCC and mixture NCC maps, focused map choices,
  hover explanations and display-only minimum filters that preserve all results.
- Expose fit diagnostics and a supplied-projection numerical API. Outputs remain
  prepared-pattern candidates; physical overlap acceptance is not implemented.
- Add a reproducible Map 1 baseline tool and local GPU numerical/regression checks,
  including the Cu report's 347 four-supported-component pixels.
- Add reproducible virtual master-pattern mixtures with counting noise and background/gain profiles, evaluated with both searches and preprocessing policies.

## 0.2.0 — 2026-11-10

### Geometry

- The detector's geometry is fitted as a vote on the pattern centre and then a plane of source
  points: how the beam's source moves over the scan. The SE(3) pose fit is removed.
- The bundled nickel scan's `detector_pixel_size` is set to 70 µm (the file had 1, kikuchipy's
  value for unknown); see `testdata/README.md`.

### Simulator

- Master patterns are simulated natively from a CIF file (the app's Simulate tab and the Python
  API's `simulate`): a CIF reader, the Joy Monte Carlo energy and depth distribution from a small
  surrogate network, and a dynamical solve per energy bin.
- Bethe weak beams enter both the strong-beam matrix and the intensity. The weak-beam cutoff c₂ is
  exposed and defaults to 40, and the correction is on by default.
- The number of beams is limited by the device's workgroup memory (512, 1024 or 1536 beams for 16,
  32 or 48 KiB), not fixed at 512; each energy bin is compiled at the smallest size that fits, and
  strong beams past capacity are followed as weak beams.
- Per-site Debye–Waller factors and the Bethe cutoffs are editable; `edit_crystal` in the Python
  API.
- A simulated master can be added as a phase or saved as `.npz`.

### App

- Master view: globe, Lambert, stereographic and joint (single-square) Lambert projections, with
  zoom and pan; one hemisphere for centrosymmetric crystals and an N − S′ difference for the
  others; a stereogram mode (distance and eye separation, default 0.5 m and 60 mm); several masters
  can be switched between and toggled on or off; Ctrl + scroll turns about the north pole.
- Crystal pane: the unit cell drawn by the engine, with an optional 3 × 3 × 3 supercell, a, b and c
  views over the picture, and a lock to the master's rotation.
- Energy and depth charts run from the beam's energy down to 0, with arrow-key navigation and an
  integrated view.
- The region of interest is drawn as a box on the map; the map takes arrow keys.
- The pattern stage (stored, filtered, background) and difference or split are chosen in place;
  the static background can be viewed.
- The Patterns tab is laid out as one card per processing step; the Simulate panel is reorganised
  with the run and its result at the top.
- What the file reader found and the simulator's summaries go to the log.

### Fixed

- Spherical indexing could give a pattern a wrong orientation and a low NCC depending on the GPU
  work before it: a dispatch's two uniform blocks could be split by the arena filling up, the
  first then overwritten before it ran (#2, by [Tijmenvermeij](https://github.com/Tijmenvermeij)). The master view's dispatch is
  guarded the same way.
- ANG and CTF exports write each phase's lattice parameters, read from ebsdsim and EMsoft masters
  and taken from the crystal for simulated ones, in place of `1 1 1 90 90 90`; a phase whose
  master carries none is refused (#1, by [Tijmenvermeij](https://github.com/Tijmenvermeij)).

## 0.1.0 — 2026-10-08

First release: indexing EBSD scans by dictionary matching against master patterns on the GPU
(Zig and WebGPU), as a browser app and a Python package.
