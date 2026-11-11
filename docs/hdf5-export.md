# HDF5 exports

## EMSphInx-compatible `.h5`

The browser's `.h5` button and Python `solver.export("results.h5")` write:

- `Scan 1/EBSD`: the selected result, using the same raw Bunge angles, raster
  order and phase metadata as ANG. Native MTEX default imports agree with ANG
  setting 2. Euler angles are radians; map steps are micrometres; lattice
  lengths are angstroms and lattice angles are degrees.
- Further scan groups: alternate-phase candidates ranked per pixel by
  `phase_by`, excluding the selected phase. These are the best stored result
  for each phase, not additional orientation peaks within the same phase.
- `zebsdsolve`: producer, settings, geometry, selected results and all retained
  phase candidates. The root `Manufacturer=EMSphInx` identifies the compatible
  layout; the producer is zebsdsolve.

Batch H5 exports additionally contain `zebsdsolve/Maps/IPFZ_RGBA` and
`NCC_RGBA` as row-major `[rows, columns, 4]` byte arrays, copied from the native
GUI maps. `zebsdsolve/Maps/NCC` stores the numeric detector NCC raster, with NaN
where no solution was computed. The maps use the existing native GUI frame and
colour settings; this adds no orientation conversion. Batch output comprises
this H5, the vendor result (H5OINA or ANG), a reopenable project/status JSON,
and standalone `_IPF-Z.png` and `_NCC.png` images, with the same stem in one
per-scan directory. The PNGs preserve native map dimensions and pixels.
**Save to Downloads** packages those five files in one ZIP per scan without an
external folder-write grant.

Export does not run another search. All computed solver solutions remain
exported, including low-scoring or unconverged matches and results whose
acceptance sweep is −1. No quality threshold is applied; users can filter
after export. Missing quality metrics are empty values, without discarding
the orientation. Pixels without a computed solution use the format's
unindexed marker. Incomplete or cancelled runs preserve the full raster.

Compatibility here targets the native MTEX EMSphInx importer and the scan-group
layout. It does not reproduce every native EMSphInx metadata field or guarantee
loading in EMSphInx's own C++ reader. `Metric` and `CI` contain detector NCC;
native EMSphInx's `Metric` is its spherical-correlation score. `IQ` currently
aliases NMI, also stored explicitly as `NMI`; it is not native EMSphInx's DCT
image-quality measurement. Compare the explicitly named quantities.

`Sample Tilt` currently retains the acquisition/engine tilt, which is not
necessarily the master simulation tilt written by native EMSphInx. Use the
recorded calibration and reference-frame provenance for coordinate comparisons;
do not infer a general TKD conversion from this scalar header alone. The native
EMSphInx comparison and current calibrated three-point check were approximately
consistent after source-derived frame conversion, without fitted alignment.

Settings and calibration are captured during indexing. Changing controls
after indexing does not relabel the old results. If a later ROI retains
results from an incompatible indexing context, export requires reindexing
the retained map with one context. Performance and IPF-colour controls are
excluded from that compatibility check.

## Oxford `.h5oina`

Oxford export creates a new file with the original acquisition's metadata,
images, map datasets and phase definitions. Diffraction patterns are omitted
by default, including hard/soft aliases to their datasets. Static backgrounds
are retained. The original file is never changed.
All map rows are replaced from solver results. Pixels without a computed
solver solution have phase 0 and empty orientations; no original acquisition
solutions are used as fallback. Original indexing and Data Processing maps
are omitted, including aliases, rather than archived as additional solutions.
New master phases receive new phase IDs. Updated pixels get the fitted pattern
centres; old indexing-specific band counts/MAD are cleared rather than reused
as solver quality measurements. Solver NCC/NMI are stored under `zebsdsolve`.

Oxford export automatically inherits the original acquisition calibration. In
Python, after loading the original scan and indexing, use:

```python
solver.export("results.h5oina")
# Or select the original baseline explicitly:
solver.export("results.h5oina", reference="original.h5oina")
```

In the browser, open the original Oxford file, index, and click **.h5oina**.
No frame JSON or MTEX runtime is needed. Python HDF5 export requires
`pip install "zebsdsolve[hdf5]"`.

The native importer retains the complete original detector Euler calibration
and tilt. Indexing uses this baseline plus any small detector-pose refinement.
The engine-to-CS1 basis stays frozen at import; refinement changes effective
calibration without changing that basis. Export uses the native conversion
`Q_oxford = inverse(C_phase) * Q_engine * inverse(B)`, where `B` is the fixed
engine-to-CS1 rotation and `C_phase` converts Oxford crystal axes to the
master axes. Selected results and every retained phase candidate receive the
same native conversion. Python and JavaScript only serialize these values.
Raw snapshots, ANG and EMSphInx-compatible H5 retain their existing conventions.

The importer records crystal-frame provenance per master, separately from the
producer/container name. Verified EMsoft masters and recognized ebsdsim
`ebsdsim-master-pattern` version-3 NPZ archives use X||a, Z||c*, while Oxford
uses X||a*, Z||c. NPZ format version and crystal axes are included in the
phase provenance. Unverified legacy/future NPZ versions remain unknown;
their producer name alone does not authorize a conversion.
The native per-phase conversion is identity for supported orthogonal lattices
and a 30° crystal-axis conversion for supported hexagonal lattices. Unknown
master provenance and unsupported crystal frames fail explicitly. Automatic
export initially requires source Tilt Axis and Scanning Rotation both zero,
complete original detector calibration, and `px_um=0`; unverified scan
translation and axis configurations fail with a native error.

`zebsdsolve/ProvenanceJSON` records the full native `oxford_frame`, including
per-phase crystal rotations and convention version, and the native calibration
status and source. When native export supplies effective detector Euler angles
and tilt, those Oxford Header values are updated in radians and their original
datasets and attributes are copied to `zebsdsolve/OriginalCalibration`.
Original Specimen Orientation Euler is preserved unchanged: CS1-to-CS0 must
not be applied again to exported CS1 orientations. Tilt Axis and Scanning
Rotation remain preserved.

Advanced Python callers may override `engine_to_cs1`, but must supply a
nonempty `frame_provenance`. The override still passes through the native
Oxford conversion and retains the per-phase crystal-basis checks. On a
calibrated Oxford acquisition it must equal the inherited basis (either
quaternion sign); it cannot redefine CS1 or bypass unsupported-calibration
checks. Arbitrary verified overrides remain available for legacy/manual
input contexts. Optional
`phase_id_map` maps master phases to verified existing Oxford phase IDs;
ambiguous phase metadata requires this explicit mapping.

Missing frames, unsupported calibration and incompatible pixel layouts fail
before an output is delivered. The original source is never overwritten.
The browser reads the original file in pieces in a worker and copies metadata
into a fresh output; omitted patterns do not impose a source-size limit.
Retained output data still needs browser memory. Python uses native HDF5
copies. Existing Data Processing maps and their aliases are omitted so readers
select only computed solver results.
Unsupported HDF5 references fail explicitly; Python supports more metadata
datatypes than the browser writer.

Validation uses native MTEX/pyMTEX importers and rotations, without a runtime
MTEX dependency. It checks candidate conversion, phase IDs, empty uncomputed pixels,
pattern omission, original metadata and source integrity. Sparse Oxford phase
IDs and some partial multiphase maps also need raw file-ID checks because the
installed pyMTEX 1.0.0 importer has phase-numbering limitations.

Reopen the original Oxford acquisition and reindex after upgrading. Old
Oxford project geometry used a different baseline convention; it is not
silently restored. New projects record `geometry_convention=oxford-calibrated-v1`.

The earlier P6 hexagonal reference-point diagnostic found a 29.88878°
original/export difference caused by omitted crystal-basis conversion.
The native per-phase conversion now addresses that omission. Actual automatic
P6 exports at pixels 83377, 57094 and 94780, without frame or detector-rotation
overrides, differ from the original Oxford orientations by 0.904°, 0.805°
and 1.187°. Serialization error is below 0.000011°. The earlier diagnostic
correct-basis comparison was about 0.90324°, with serialization error about
0.00000397°. That point comparison does not validate every detector geometry
or scan translation configuration. The ten fixed-orientation forward checks
used `px_um=0`; source/master c lengths differed (4.729 versus 4.684 angstroms).
The preservation check retained all 144 non-pattern datasets and attributes,
reducing a 2,457,095,971-byte source to about 62 MB.

The same P6 check with the supplied ebsdsim version-3 Ti NPZ master produced
native original/export differences of 0.976°, 0.766° and 1.196°. An actual
browser-produced H5OINA file using that NPZ was reimported with native MTEX
and differed by 0.965° at pixel 83377. The master is recognized through its
format/version metadata; its producer is not relabelled as EMsoft.

References: [Oxford H5OINA specification](https://github.com/oinanoanalysis/h5oina/blob/master/H5OINAFile.md)
and [EMsoft crystal matrices](https://github.com/EMsoft-org/EMsoft/blob/develop/Source/EMsoftLib/crystal.f90).
The ebsdsim v3 convention was verified against pinned revision
`52afdeba84792b2a962f39d582aa0cea6182a341`:
[lattice construction](https://github.com/ZacharyVarley/ebsdsim/blob/52afdeba84792b2a962f39d582aa0cea6182a341/ebsdsim/crystal/build.py),
[Cartesian propagation directions](https://github.com/ZacharyVarley/ebsdsim/blob/52afdeba84792b2a962f39d582aa0cea6182a341/ebsdsim/lambert/kgrid.py),
and [NPZ writer](https://github.com/ZacharyVarley/ebsdsim/blob/52afdeba84792b2a962f39d582aa0cea6182a341/ebsdsim/io/save.py).
CIF setting transforms recorded as P/p have already been accounted for upstream;
they are not applied again to the stored directions or exported orientations.


## Upstream 0.2 source-plane integration

Ordinary Oxford acquisition geometry retains its calibrated detector basis and
unknown detector pixel size as zero. Saved settings and batch replay preserve
the complete source-plane geometry, including all 12 coefficients.

H5OINA export after source-plane fitting displays a warning and retains the
original acquisition's `Pattern Center X`, `Pattern Center Y` and `Detector
Distance` fields unchanged, including unindexed pixels, attributes and absent
fields. The fitted per-pixel PCs remain in `/zebsdsolve/PatternCentres`, with the
full plane in `GeometryJSON` and the preservation policy in `ProvenanceJSON`.
The warning also appears on batch jobs and in Python exports. It does not block
export. Orientations and effective detector attitude use the frozen indexing
context and the existing native Oxford frame conversion; no extra rotation is
introduced. The standard Oxford PCs therefore describe the acquisition rather
than the fitted geometry used for these results.
