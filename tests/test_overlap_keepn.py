"""Overlap-private V5 candidate pools; scalar tests do not convert orientations."""
from pathlib import Path
import os
import subprocess
import unittest
import threading
import time
import numpy as np
import test_overlap as fixtures
from zebsdsolve import Error

ROOT = Path(__file__).resolve().parents[1]


class KeepNProduction(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.NativeDiscovery()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.s = self.fixture.s
        self.s.configure(method='dictionary')

    def test_default_one_and_primary_only_are_bit_identical(self):
        for stored in (False, True):
            if stored:
                self.s.index()
            options = dict(primary='stored' if stored else 'search', blur_gain=True)
            default = self.s.overlap_cells([1, 2], **options)
            explicit = self.s.overlap_cells([1, 2], keep_n=1, **options)
            for key, value in default.items():
                if isinstance(value, np.ndarray):
                    np.testing.assert_array_equal(explicit[key], value, err_msg=key)
            one = self.s.overlap_cells([1, 2], max_components=1, keep_n=1, **options)
            eight = self.s.overlap_cells([1, 2], max_components=1, keep_n=8, **options)
            for key, value in one.items():
                if isinstance(value, np.ndarray):
                    np.testing.assert_array_equal(eight[key], value, err_msg='K1 ' + key)

    def test_nested_pools_same_primary_joint_sse_and_winner_rank(self):
        self.s.index()
        stored = {key: value.copy() for key, value in self.s.results().items() if isinstance(value, np.ndarray)}
        runs = [self.s.overlap_cells([1, 2], primary='stored', blur_gain=True, keep_n=n) for n in (1, 2, 5)]
        for small, large in zip(runs, runs[1:]):
            np.testing.assert_array_equal(large['quaternions'][:, 0], small['quaternions'][:, 0])
            self.assertTrue(np.all(large['sse'] <= small['sse'] + 1e-7), (small['sse'], large['sse']))
        result = runs[-1]
        np.testing.assert_array_equal(result['proposed_count'], [2, 2])
        np.testing.assert_array_equal(result['winner_rank'][:, 0], [0, 0])
        if self.s.settings['method'] == 'dictionary':
            self.assertTrue(np.any(result['winner_rank'][:, 1] > 1))
        self.assertTrue(np.all((result['winner_rank'][:, 1] >= 1) & (result['winner_rank'][:, 1] <= 5)))
        if self.s.settings['method'] == 'dictionary':
            np.testing.assert_array_equal(result['phase'], [[0, 0], [0, 1]])
        for key, value in stored.items():
            np.testing.assert_array_equal(self.s.results()[key], value, err_msg='stored primary ' + key)
        for row, cell in enumerate((1, 2)):
            images = self.s.overlap_inspect(cell, result, row=row)
            self.assertAlmostEqual(float(np.sum(images['residual'].astype(float)**2)), float(result['sse'][row]), delta=5e-5)
            self.assertTrue(np.any(images['residual'] < 0))

    def test_later_orders_preserve_prefix_and_replay_with_independent_cells(self):
        # Reuse the existing native four-pose projection fixture; no new
        # crystallographic conversion or claim about ground-truth recovery.
        self.s.clear_phases()
        self.s.add_phase(ROOT / 'testdata/nickel/Ni_master_20kV.npz')
        self.s.set_geometry(step_x_um=0, step_y_um=0)
        q = np.random.default_rng(714).normal(size=(4, 4))
        q = (q / np.linalg.norm(q, axis=1, keepdims=True)).astype(np.float32)
        base = fixtures.unit(self.s.simulate(q, cells=[0] * 4))
        patterns = np.zeros((6, 48, 48), np.float32)
        patterns[1] = base[0]
        patterns[2] = np.einsum('i,ihw->hw', [.7, .5, .3], base[:3])
        patterns[3] = np.einsum('i,ihw->hw', [.7, .5, .3, .2], base)
        self.s.set_patterns(patterns[None])
        self.s.index()
        stored = {key: value.copy() for key, value in self.s.results().items() if isinstance(value, np.ndarray)}
        cells = [0, 1, 2, 3]
        runs = [self.s.overlap_cells(cells, max_components=k, primary='stored', keep_n=3, blur_gain=True) for k in (2, 3, 4)]
        for previous, following in zip(runs, runs[1:]):
            k = previous['max_components']
            # Final best-of-two may change histories across requested K. Only
            # a shared selected trajectory has a common committed prefix.
            for row in range(len(cells)):
                if following['trajectory_source'][row] == previous['trajectory_source'][row]:
                    for field in ('quaternions', 'phase', 'winner_rank'):
                        np.testing.assert_array_equal(following[field][row, :k], previous[field][row], err_msg='same-history prefix ' + field)
            self.assertTrue(np.all(following['sse'][1:] <= previous['sse'][1:] + 1e-7))
        result = runs[-1]
        self.assertEqual(int(result['retained_order'][0]), 0)
        self.assertEqual(int(result['proposed_count'][0]), 0)
        self.assertEqual(int(result['retained_order'][3]), 4)
        self.assertEqual(int(result['proposed_count'][3]), 4)
        self.assertTrue(np.all(result['proposed_count'] >= result['retained_order']))
        self.assertTrue(np.all(result['proposed_count'] <= 4))
        np.testing.assert_array_equal(result['winner_rank'][0], -1)
        np.testing.assert_array_equal(result['winner_rank'][3, 0], 0)
        self.assertTrue(np.all((result['winner_rank'][3, 1:] >= 1) & (result['winner_rank'][3, 1:] <= 3)))
        np.testing.assert_array_equal(result['phase'][3], 0)
        # Replay every retained contribution, not only the winning final slot.
        images = [self.s.overlap_inspect(3, result, row=3, selected_component=j) for j in range(4)]
        composite = sum(image['selected_contribution'] for image in images)
        np.testing.assert_allclose(composite, images[0]['composite'], atol=3e-7)
        np.testing.assert_allclose(images[0]['measured'] - composite, images[0]['residual'], atol=3e-7)
        self.assertAlmostEqual(float(np.sum(images[0]['residual'].astype(float)**2)), float(result['sse'][3]), delta=5e-5)
        self.assertTrue(np.any(images[0]['residual'] < 0))
        singleton = self.s.overlap_cells([3], max_components=4, primary='stored', keep_n=3, blur_gain=True)
        for field, values in result.items():
            if isinstance(values, np.ndarray) and values.shape and values.shape[0] == len(cells):
                np.testing.assert_array_equal(singleton[field][0], values[3], err_msg='singleton vs heterogeneous batch ' + field)
        for field, values in stored.items():
            np.testing.assert_array_equal(self.s.results()[field], values, err_msg='stored primary ' + field)

    def test_missing_flat_stale_and_invalid_configuration_fail_closed(self):
        missing = self.s.overlap_cells([1], primary='stored', keep_n=5)
        np.testing.assert_array_equal(missing['winner_rank'], [[-1, -1]])
        self.assertEqual(int(missing['stop_reason'][0]), 6)
        flat = self.s.overlap_cells([3], keep_n=5)
        self.assertEqual(int(flat['retained_order'][0]), 0)
        np.testing.assert_array_equal(flat['winner_rank'], [[-1, -1]])
        for invalid in (0, 9, True, 1.5):
            with self.assertRaisesRegex(ValueError, 'keep_n'):
                self.s.overlap_cells([1], keep_n=invalid)
        with self.assertRaisesRegex(ValueError, 'contrast'):
            self.s.overlap_cells([1], keep_n=2, gain_domain='source_dc')
        self.s.configure(method='pca')
        with self.assertRaisesRegex(ValueError, 'PCA'):
            self.s.overlap_cells([1], keep_n=2)
        self.s.configure(method='dictionary')
        self.s.index()
        self.s.configure(refine_iters=7)
        with self.assertRaisesRegex(Error, 'StoredPrimaryContextStale'):
            self.s.overlap_cells([1], primary='stored', keep_n=5)

    def test_cancel_keeps_stored_primary_and_next_call_works(self):
        self.s.index()
        stored = {key: value.copy() for key, value in self.s.results().items() if isinstance(value, np.ndarray)}
        started = threading.Event()
        outcome = []
        def run():
            started.set()
            try:
                outcome.append(self.s.overlap_cells([1, 2], primary='stored', keep_n=8, blur_gain=True))
            except Error as error:
                outcome.append(error)
        worker = threading.Thread(target=run)
        worker.start()
        started.wait(1)
        deadline = time.monotonic() + 10
        while worker.is_alive() and time.monotonic() < deadline:
            self.s.cancel()
            time.sleep(.002)
        worker.join(1)
        self.assertFalse(worker.is_alive(), 'cancellation must reach a bounded native wait')
        self.assertEqual(len(outcome), 1)
        self.assertIsInstance(outcome[0], Error)
        self.assertIn('Cancelled', str(outcome[0]))
        for key, value in stored.items():
            np.testing.assert_array_equal(self.s.results()[key], value, err_msg='cancelled stored primary ' + key)
        again = self.s.overlap_cells([1], primary='stored', keep_n=2)
        self.assertGreater(int(again['retained_order'][0]), 0)

    def test_old_runtime_guard_only_applies_to_residual_pools(self):
        native = self.s._lib
        class OldRuntime:
            def __getattr__(self, name):
                if name == 'ze_overlap_cells_v5':
                    raise AttributeError(name)
                return getattr(native, name)
        self.s._lib = OldRuntime()
        try:
            with self.assertRaisesRegex(Error, 'native V5'):
                self.s.overlap_cells([1], keep_n=5)
            self.assertEqual(int(self.s.overlap_cells([1], max_components=1, keep_n=8)['winner_rank'][0, 0]), 0)
            self.assertEqual(self.s.overlap_cells([1], keep_n=1)['keep_n'], 1)
        finally:
            self.s._lib = native

    def test_native_v5_rejects_bad_flags_and_lengths(self):
        from zebsdsolve import _native as nt
        n = 1
        cells = np.array([1], np.uint64)
        out = (nt.OverlapComponent * 2)()
        states = (nt.OverlapState * n)()
        progress = (nt.OverlapProgress * n)()
        theta = np.empty((1, 7), np.float32)
        norms = np.empty((1, 2), np.float32)
        ranks = np.empty((1, 2), np.int32)
        args = [self.s._e, cells.ctypes.data_as(nt.u64p), n, 2, 0, 1, 5,
                out, 2, states, 1, progress, 1, theta.ctypes.data_as(nt.f32p), 1,
                norms.ctypes.data_as(nt.f32p), 2, ranks.ctypes.data_as(nt.i32p), 2]
        # Direct C V5 keep_n=1 must reproduce V4, independently of the wrapper's
        # intentional default-V4 dispatch.
        args[6] = 1
        self.assertEqual(self.s._lib.ze_overlap_cells_v5(*args), 0, self.s._error())
        expected = (bytes(out), bytes(states), bytes(progress), theta.copy(), norms.copy())
        v4 = args[:6] + args[7:17]
        self.assertEqual(self.s._lib.ze_overlap_cells_v4(*v4), 0, self.s._error())
        self.assertEqual(bytes(out), expected[0])
        self.assertEqual(bytes(states), expected[1])
        self.assertEqual(bytes(progress), expected[2])
        np.testing.assert_array_equal(theta, expected[3])
        np.testing.assert_array_equal(norms, expected[4])
        np.testing.assert_array_equal(ranks, [[0, 1]])
        args[6] = 5
        for index, value in ((3, 0), (4, 2), (5, 2), (6, 0), (6, 9), (8, 1), (18, 1)):
            bad = args.copy(); bad[index] = value
            self.assertLess(self.s._lib.ze_overlap_cells_v5(*bad), 0)
            self.s._error()


class SphericalKeepNProduction(KeepNProduction):
    def setUp(self):
        super().setUp()
        self.s.configure(method='spherical')


class DictionaryKernelOracle(unittest.TestCase):
    def test_exact_scores_ties_and_chunk_merging(self):
        probe = ROOT / 'zig-out/bin/probe'
        if not probe.exists():
            self.skipTest('build native probe first')
        wgpu = Path(os.environ.get('ZEBSDSOLVE_WGPU', ROOT / '.deps/wgpu-native/macosx_11_0_arm64/lib/libwgpu_native.dylib'))
        run = subprocess.run([str(probe), str(wgpu.parent)], capture_output=True, text=True, timeout=180)
        self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
        self.assertIn('exact top1/5/8 CPU oracle and ordinary top1 parity', run.stderr)


if __name__ == '__main__':
    unittest.main()
