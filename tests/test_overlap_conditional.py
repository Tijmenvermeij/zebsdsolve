"""End-to-end overlap checks with known detector-domain orientation mixtures."""
from itertools import permutations
import unittest

import numpy as np
from scipy.optimize import nnls

from test_overlap import ROOT, Solver
from test_overlap_blur_gain import unit
from test_overlap_contrast_gain import centered_and_norm, contrast_transform, mask_pixels


def mixture_fixture(solver, blur_gain):
    """Simulate known native wxyz sample-to-crystal poses at each original cell."""
    weights = np.array([[1, 0, 0], [.8, .55, 0], [.75, .5, .35], [.9, .16, 0]])
    height, width = 56, 68
    cells = np.arange(len(weights))
    solver.configure(method="spherical", bandwidth=64, refine_iters=16, refine_step=.5,
                     static_background="none", dynamic_background="none", clahe=False,
                     bandpass=False, master_smooth=0, mask=blur_gain, mask_radius=.85,
                     binning=1, full_resolution="none", batch=8)
    solver.add_phase(ROOT/"testdata/nickel/Ni_master_20kV.npz")
    solver.set_patterns(np.zeros((1, len(cells), height, width), np.float32))
    solver.set_geometry(pc=(.44, .24, .53), sample_tilt_deg=70, detector_tilt_deg=0,
                        px_um=560, step_x_um=5, step_y_um=5)
    rng = np.random.default_rng(714)
    poses = rng.normal(size=(3, 4))
    poses = (poses / np.linalg.norm(poses, axis=1, keepdims=True)).astype(np.float32)
    raw = np.stack([solver.simulate(np.tile(pose, (len(cells), 1)), cells=cells)
                    for pose in poses], axis=1)
    mask = ((height - 1) * .5, (width - 1) * .5, .85 * height * .5) if blur_gain else None
    theta = (.8, .15, 1.2, .92, 1.04, .03, -.02) if blur_gain else (0, 0, 1, 1, 1, 0, 0)
    templates = unit(contrast_transform(raw, theta, mask), mask)
    observed = np.einsum("nj,njhw->nhw", weights, templates)
    solver.set_patterns(observed[None])
    return cells, poses, weights, mask


def matched_errors(solver, found, truth):
    """Assign poses with native phase symmetry, keeping the test suite MTEX-independent."""
    count = len(truth)
    distance = np.rad2deg(solver.disorientation(
        np.repeat(found, count, axis=0), np.tile(truth, (count, 1)))).reshape(count, count)
    return min((distance[order, np.arange(count)] for order in permutations(range(count))),
               key=np.sum)


class ConditionalOrientationPolish(unittest.TestCase):
    def setUp(self):
        self.s = Solver()
        self.addCleanup(self.s.close)

    def test_singleton_matches_ordinary_indexing_and_does_not_change_stored_results(self):
        cells, truth, _, _ = mixture_fixture(self.s, blur_gain=False)
        primary = self.s.index()
        ordinary = self.s.index_cells(cells[:1])
        result = self.s.overlap_cells(cells[:1], max_components=1, primary="stored", polish=True)
        self.assertEqual(result["profile_version"], 3)
        self.assertEqual(result["retained_order"][0], 1)
        np.testing.assert_array_equal(result["source_quaternions"][0, 0],
                                      primary["quaternions"][0, 0])
        self.assertLess(matched_errors(self.s, result["quaternions"][0], truth[:1])[0], .15)
        self.assertLess(matched_errors(self.s, result["quaternions"][0],
                                       ordinary["quaternions"][0])[0], .15)
        self.assertAlmostEqual(float(result["mixture_ncc"][0]),
                               float(ordinary["ncc"][0, 0]), delta=2e-5)
        for key, value in primary.items():
            np.testing.assert_array_equal(self.s.results()[key], value, err_msg=key)

    def test_known_two_three_and_weak_component(self):
        cells, truth, weights, _ = mixture_fixture(self.s, blur_gain=False)
        for cell in cells[1:]:
            order = int(np.count_nonzero(weights[cell]))
            with self.subTest(cell=cell, order=order):
                options = dict(max_components=order, min_residual_ncc=.05)
                raw = self.s.overlap_cells([cell], **options)
                result = self.s.overlap_cells([cell], **options, polish=True)
                self.assertEqual(raw["retained_order"][0], order)
                self.assertEqual(result["retained_order"][0], order)
                np.testing.assert_array_equal(result["source_quaternions"], raw["source_quaternions"])
                initial_error = matched_errors(self.s, raw["quaternions"][0], truth[:order])
                error = matched_errors(self.s, result["quaternions"][0], truth[:order])
                self.assertLess(float(error.max()), .1, error)
                self.assertLess(float(error.sum()), float(initial_error.sum()))
                self.assertLess(float(result["sse"][0]), float(raw["sse"][0]))

    def test_masked_blur_gain_final_fit_and_fixed_nuisance(self):
        cells, truth, weights, mask = mixture_fixture(self.s, blur_gain=True)
        for cell in cells:
            order = int(np.count_nonzero(weights[cell]))
            with self.subTest(cell=cell, order=order):
                options = dict(max_components=order, blur_gain=True, min_residual_ncc=.05)
                raw = self.s.overlap_cells([cell], **options)
                fixed = self.s.overlap_cells([cell], **options, polish=True)
                refit = self.s.overlap_cells([cell], **options, polish=True, refit_blur_gain_after_polish=True)
                self.assertEqual(refit["retained_order"][0], order)
                np.testing.assert_array_equal(fixed["blur_gain_theta"], raw["blur_gain_theta"])
                np.testing.assert_array_equal(refit["quaternions"], fixed["quaternions"])
                np.testing.assert_array_equal(refit["source_quaternions"], raw["source_quaternions"])
                self.assertGreater(float(fixed["blur_gain_theta"][0, 0]), .2)
                self.assertGreater(abs(float(fixed["blur_gain_theta"][0, 1])), .03)
                self.assertLess(float(fixed["sse"][0]), float(raw["sse"][0]))
                poses = refit["quaternions"][0]
                initial_error = matched_errors(self.s, raw["quaternions"][0], truth[:order])
                error = matched_errors(self.s, poses, truth[:order])
                self.assertLess(float(error.max()), .2, error)
                self.assertLess(float(error.sum()), float(initial_error.sum()))
                projected = self.s.simulate(poses, cells=[cell] * order)
                transformed = contrast_transform(projected, refit["blur_gain_theta"][0], mask)
                centered, norms = centered_and_norm(transformed, mask)
                design = (centered / norms[:, None, None]).reshape(order, -1).T
                target = unit(self.s.pattern(cell), mask)
                # The optional theta update ranks isolated targets at the fixed
                # orientations and amplitudes, before the final NNLS.
                frozen = unit(contrast_transform(projected, fixed["blur_gain_theta"][0], mask), mask)
                weights_fixed = fixed["amplitudes"][0, :, None]
                contributions = weights_fixed * frozen.reshape(order, -1)
                isolated = target.ravel() - contributions.sum(axis=0) + contributions
                initial_error = np.sum((isolated - contributions)**2)
                final_error = np.sum((isolated - weights_fixed * design.T)**2)
                self.assertLessEqual(final_error, initial_error + 4e-6)
                if cell == 1:
                    self.assertLess(final_error, initial_error - 2e-6)
                    self.assertFalse(np.array_equal(refit["blur_gain_theta"], fixed["blur_gain_theta"]))
                amplitudes, _ = nnls(design, target.ravel())
                residual = (target.ravel() - design @ amplitudes).reshape(target.shape)
                np.testing.assert_allclose(refit["amplitudes"][0], amplitudes, atol=3e-6)
                np.testing.assert_allclose(refit["component_norms"][0], norms, rtol=3e-5)
                self.assertAlmostEqual(float(refit["sse"][0]),
                                       float(np.sum(residual**2)), delta=4e-6)
                images = self.s.overlap_inspect(cell, refit)
                np.testing.assert_allclose(images["residual"], residual, atol=3e-6)
                np.testing.assert_array_equal(images["residual"][~mask_pixels(target.shape, mask)], 0)


if __name__ == "__main__":
    unittest.main()
