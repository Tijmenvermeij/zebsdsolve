"""Native H5OINA pattern-source regressions (numpy, h5py, native build)."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
from zebsdsolve import Error, Solver


class OxfordPatterns(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory(prefix="zebsdsolve-oxford-patterns-")
        self.addCleanup(tmp.cleanup)
        self.path = Path(tmp.name) / "scan.h5oina"
        self.processed = np.arange(480, dtype=np.uint8).reshape(6, 8, 10)
        self.raw = (np.arange(480, dtype=np.int16) * 100 - 20000).reshape(6, 8, 10)
        self.raw[0, 0, :2] = [-32768, 32767]

    def fixture(self, *, processed=True, raw=True, processed_bg=True, raw_bg=True, endian="<", raw_bg_shape=(8, 10)):
        with h5py.File(self.path, "w") as f:
            header = f.create_group("1/EBSD/Header")
            header["Y Cells"], header["X Cells"] = [2], [3]
            if processed_bg:
                header["Processed Static Background"] = np.full((8, 10), 23, dtype="u1")
            if raw_bg:
                header["Unprocessed Static Background"] = np.full(raw_bg_shape, -51, dtype=endian + "i2")
            data = f.create_group("1/EBSD/Data")
            if processed:
                data["Processed Patterns"] = self.processed
            if raw:
                data.create_dataset("Unprocessed Patterns", data=self.raw.astype(endian + "i2"),
                                    chunks=(2, 8, 10), compression="lzf", shuffle=True)

    def configured_open(self, solver, source, **settings):
        name = str(self.path).encode()
        patch = json.dumps(settings).encode()
        hints = json.dumps({"oxford_patterns": source}).encode()
        status = solver._lib.ze_open_scan_configured(
            solver._e, name, len(name), patch, len(patch), hints, len(hints),
        )
        solver._check(status, "open_scan_configured")

    def test_two_sources_choose_without_replacing_the_current_scan(self):
        with Solver() as solver:
            for endian in ("<", ">"):
                with self.subTest(endian=endian):
                    self.fixture(endian=endian)
                    solver.configure(static_background="file")
                    solver.open(self.path)
                    settings, scan, geometry = solver.settings, solver.scan, solver.geometry
                    self.assertEqual(scan["dtype"], "uint8")
                    np.testing.assert_array_equal(solver.pattern(0, processed=False), self.processed[0])
                    np.testing.assert_array_equal(solver.pattern(0, processed="background"), np.full((8, 10), 23))
                    self.assertTrue(any("Oxford pattern source: Processed Patterns" in event.get("text", "")
                                        for event in solver.events()))

                    with self.assertRaisesRegex(Error, "OxfordPatternChoiceRequired"):
                        self.configured_open(solver, "ask", bandwidth=16, mask=True)
                    self.assertEqual(solver.settings, settings)
                    self.assertEqual(solver.scan, scan)
                    self.assertEqual(solver.geometry, geometry)
                    np.testing.assert_array_equal(solver.pattern(0, processed=False), self.processed[0])

                    self.configured_open(solver, "unprocessed")
                    self.assertEqual(solver.scan["dtype"], "float32")
                    self.assertNotIn("oxford_patterns", solver.settings)
                    np.testing.assert_array_equal(solver.pattern(0, processed=False), self.raw[0])
                    np.testing.assert_array_equal(solver.pattern(0, processed="background"), np.full((8, 10), -51))
                    cells = [4, 0, 5, 0]
                    means, stddevs = solver.pattern_stats(cells)
                    np.testing.assert_allclose(means, self.raw[cells].mean(axis=(1, 2)), rtol=1e-6)
                    np.testing.assert_allclose(stddevs, self.raw[cells].std(axis=(1, 2)), rtol=1e-6)
                    self.assertTrue(any("Oxford pattern source: Unprocessed Patterns" in event.get("text", "")
                                        for event in solver.events()))

                    self.configured_open(solver, "processed")
                    np.testing.assert_array_equal(solver.pattern(0, processed=False), self.processed[0])
                    solver._lib.ze_close_scan(solver._e)

    def test_single_source_opens_without_a_choice(self):
        with Solver() as solver:
            for is_raw in (False, True):
                with self.subTest(is_raw=is_raw):
                    # Keep only the opposite source's static background.
                    self.fixture(processed=not is_raw, raw=is_raw, processed_bg=is_raw, raw_bg=not is_raw)
                    expected = self.raw[0] if is_raw else self.processed[0]
                    solver.open(self.path)
                    np.testing.assert_array_equal(solver.pattern(0, processed=False), expected)
                    self.configured_open(solver, "ask")
                    self.assertFalse(solver.scan["has_background"])
                    np.testing.assert_array_equal(solver.pattern(0, processed=False), expected)
                    with self.assertRaisesRegex(Error, "OxfordPatternSourceUnavailable"):
                        self.configured_open(solver, "processed" if is_raw else "unprocessed")
                    np.testing.assert_array_equal(solver.pattern(0, processed=False), expected)
                    solver._lib.ze_close_scan(solver._e)

            self.fixture(processed=False, raw_bg_shape=(10, 8))
            self.configured_open(solver, "unprocessed")
            self.assertFalse(solver.scan["has_background"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
