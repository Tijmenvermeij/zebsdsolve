"""Native symmetry-reduced fitted poses survive calibration and exports.

Run locally with a rebuilt native library, GPU, NumPy and h5py:
    python tests/test_fitted_pose.py

The existing monoclinic unique-b fixture checks upstream reduction about Y,
forward-pattern agreement, unchanged raw candidate records, and exported angles
of the stored equivalent pose. Independent scientific oracles are private
development validation, outside the repository.
"""
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import h5py

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
from zebsdsolve import Solver


class FittedPose(unittest.TestCase):
    def setUp(self):
        self.s = Solver()
        self.addCleanup(self.s.close)
        self.s.configure(
            master_smooth=0, static_background="none", dynamic_background="none",
            mask=False, clahe=False, bandwidth=32, refine_iters=8,
            sim_halfw=32, sim_dmin_nm=.12, sim_bin_kev=5, sim_c_weak=0,
            toy_seed=17, toy_rows=3, toy_cols=4, toy_height=40, toy_width=40,
            toy_grains=12, toy_counts=0, toy_contrast=1, batch=12, sweeps=1,
        )
        self.s.set_crystal({
            "space_group": 10, "cell": [3.5, 4.5, 5.5, 90, 105, 90],
            "sites": [
                {"symbol": "Al", "xyz": [.17, .23, .31], "b_iso": .5},
                {"symbol": "Cu", "xyz": [.29, .11, .43], "b_iso": .5},
            ],
        })
        self.assertEqual(self.s.simulate_master()["failed"], 0)
        self.s.add_simulated_phase("C2 unique-b fixture")
        self.s.set_geometry(pc=[.45, .3, .6], px_um=0)
        self.s.synthesize()

    def assert_same_pose(self, a, b, tolerance=2e-6):
        # Compare stored components, allowing only quaternion sign equivalence.
        a, b = np.broadcast_arrays(a, b)
        error = np.minimum(np.max(np.abs(a-b), axis=-1), np.max(np.abs(a+b), axis=-1))
        self.assertLess(float(np.max(error)), tolerance)

    def test_bunge_preserves_shapes_and_rejects_bad_shapes(self):
        np.testing.assert_allclose(self.s.bunge([1, 0, 0, 0]), np.zeros(3))
        batch = np.zeros((2, 3, 4), dtype=np.float32)
        batch[..., 0] = 1
        np.testing.assert_allclose(self.s.bunge(batch), np.zeros((2, 3, 3)))
        self.assertEqual(self.s.bunge(np.empty((0, 4))).shape, (0, 3))
        for invalid in (1, [], [1, 0, 0], np.zeros((4, 1)), np.zeros(8)):
            with self.subTest(shape=np.shape(invalid)):
                with self.assertRaisesRegex(ValueError, "shape"):
                    self.s.bunge(invalid)

    def test_selected_pose_is_correct_equivalent_and_exports_stored_pose(self):
        s = self.s
        # Repeat the same one-phase search in one batch, without result storage. This
        # public API is also available upstream, where candidate snapshots are absent.
        raw = s.index_cells(np.arange(12))
        candidate = raw["quaternions"][0]
        result = s.index()
        snapshot = s.export_snapshot() if callable(getattr(s, "export_snapshot", None)) else None
        q = result["quaternions"].reshape(-1, 4)
        # One phase, one batch and one sweep ensure the same search settings and raw
        # winner. Never use a candidate from another sweep to reconstruct selected poses.
        self.assertTrue(np.all(result["phase"] == 0))
        self.assert_same_pose(q, s.reduce(candidate)[0])
        np.testing.assert_allclose(np.linalg.norm(q, axis=-1), 1, atol=1e-6)
        self.assertTrue(np.all(q[:, 0] >= 0))
        np.testing.assert_allclose(result["ncc"].reshape(-1), raw["ncc"][0], atol=1e-5)
        angles = [result["euler"].reshape(-1, 3), s.bunge(q)]
        if snapshot is not None:
            self.assert_same_pose(np.asarray(snapshot["candidates"]["quaternions"]).reshape(-1, 4), candidate)
            self.assert_same_pose(np.asarray(snapshot["results"]["quaternions"]).reshape(-1, 4), q)
            angles.append(np.asarray(snapshot["results"]["euler"]).reshape(-1, 3))
        for euler in angles:
            np.testing.assert_allclose(euler, s.bunge(q), atol=1e-6, rtol=0)
        # The fixture must exercise reduction; simulate the actual stored poses
        # to check that native measured symmetry preserves their patterns.
        folded = np.minimum(np.max(np.abs(candidate-q), axis=-1),
                            np.max(np.abs(candidate+q), axis=-1)) > 1e-3
        self.assertGreater(int(folded.sum()), 0, "fixture must exercise native symmetry reduction")
        original, equivalent = s.simulate(candidate), s.simulate(q)
        correlation = np.array([np.corrcoef(a.ravel(), b.ravel())[0, 1] for a, b in zip(original, equivalent)])
        self.assertGreater(float(correlation.min()), .99)
        self.assert_same_pose(s.reduce(q)[0], q)
        with tempfile.TemporaryDirectory(prefix="zebsdsolve-pose-") as tmp:
            for suffix in (("ang", "h5") if snapshot is not None else ("ang",)):
                with self.subTest(format=suffix):
                    path = Path(tmp) / f"poses.{suffix}"
                    s.export(path)
                    if suffix == "ang":
                        np.testing.assert_allclose(np.loadtxt(path)[:, :3], s.bunge(q), atol=5.1e-6, rtol=0)
                    else:
                        with h5py.File(path) as output:
                            data = output["Scan 1/EBSD/Data"]
                            np.testing.assert_array_equal(data["Quaternion"][()], q)
                            np.testing.assert_allclose(np.column_stack([data[key][()] for key in ("Phi1", "Phi", "Phi2")]),
                                                       s.bunge(q), atol=1e-6, rtol=0)

    def test_vote_returns_native_symmetry_equivalent_search_pose(self):
        s = self.s
        s.configure(fit_grid=4, fit_bandwidth=32, pc_steps=1, pc_steps_z=1,
                    pc_levels=1, refine=False)
        raw = s.index_cells(np.arange(12), bandwidth=32)["quaternions"][0]
        s.fit_pc()
        cells, phases, q, _ = s.fit_patterns()
        self.assertTrue(np.all(phases == 0))
        self.assert_same_pose(q, s.reduce(raw[cells])[0])
        self.assert_same_pose(q, s.reduce(q)[0])
        original, equivalent = s.simulate(raw[cells], cells=cells), s.simulate(q, cells=cells)
        self.assertGreater(min(np.corrcoef(a.ravel(), b.ravel())[0, 1] for a, b in zip(original, equivalent)), .99)

    def test_plane_fit_returns_poses_that_reproduce_patterns(self):
        s = self.s
        s.configure(fit_bandwidth=32, plane_stride=1, plane_iters=3,
                    accept_z=0, accept_step_deg=180)
        s.fit_plane()
        cells, phases, q, _ = s.fit_patterns()
        self.assertTrue(np.all(phases == 0))
        sims = s.simulate(q, cells=cells)
        correlation = [np.corrcoef(s.pattern(int(c), processed=False).ravel(), a.ravel())[0, 1]
                       for c, a in zip(cells, sims)]
        self.assertGreater(min(correlation), .99)
        self.assert_same_pose(q, s.reduce(q)[0])

    def test_cubic_and_polar_phases_store_native_equivalent_poses(self):
        s = self.s
        for folder, name in (("nickel", "Ni"), ("gan", "GaN")):
            with self.subTest(phase=name):
                s.clear_phases()
                s.add_phase(ROOT / "testdata" / folder / f"{name}_master_20kV.npz")
                s.synthesize()
                candidate = s.index_cells(np.arange(12))["quaternions"][0]
                result = s.index()
                if callable(getattr(s, "export_snapshot", None)):
                    snapshot = s.export_snapshot()
                    self.assert_same_pose(np.asarray(snapshot["candidates"]["quaternions"]).reshape(-1, 4), candidate)
                q = result["quaternions"].reshape(-1, 4)
                self.assert_same_pose(q, s.reduce(candidate)[0])
                np.testing.assert_allclose(result["euler"].reshape(-1, 3), s.bunge(q), atol=1e-6, rtol=0)
                self.assert_same_pose(s.reduce(q)[0], q)


if __name__ == "__main__":
    unittest.main()
