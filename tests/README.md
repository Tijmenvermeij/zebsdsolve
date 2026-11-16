# Local checks

From the repository root, build the native library with the matching Zig version
and wgpu-native headers, then run the affected Python suites:

```sh
zig build -Doptimize=ReleaseFast -Dwgpu=/path/to/wgpu-native
python tests/test_fitted_pose.py
python tests/test_hdf5_exports.py
python tests/test_up_ang.py
zig build test -Doptimize=ReleaseFast -Dwgpu=/path/to/wgpu-native
```

Python checks require NumPy and h5py; the solver also requires a GPU and the
wgpu-native runtime. Set `ZEBSDSOLVE_LIB` and `ZEBSDSOLVE_WGPU` to explicit shared
library paths when they are outside the checkout's standard discovery locations.

For ZIPs saved by the browser download check:

```sh
python web/tests/check-download-zips.py /path/to/downloads
```

That check also requires Pillow and checks archive structure, native maps,
serialized vendor records and project state. Repository checks have no MTEX
dependency. Independent MTEX importer/frame oracles are private development
validation maintained outside the repository.
