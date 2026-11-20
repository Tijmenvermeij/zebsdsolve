"""Candidate-order isolation at the real GPU seam and native two-phase wiring.

Orientations are frozen native wxyz values; no crystallographic conversion is
performed. Scalar projection normalization only constructs synthetic mixtures.
"""
from pathlib import Path
import json
import os
import subprocess
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'python'))
from zebsdsolve import Solver


class CandidateIsolation(unittest.TestCase):
    def test_real_model_candidate_sequences(self):
        probe = Path(os.environ.get('ZEBSDSOLVE_PROBE', ROOT / 'zig-out/bin/probe'))
        if not probe.exists():
            self.skipTest('build the native probe before this GPU integration test')
        wgpu = Path(os.environ.get('ZEBSDSOLVE_WGPU', ROOT / '.deps/wgpu-native/macosx_11_0_arm64/lib/libwgpu_native.dylib'))
        run = subprocess.run([str(probe), str(wgpu.parent)], capture_output=True, text=True, timeout=180)
        self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
        self.assertIn('12 unique winners;', run.stderr)
        self.assertIn('provisional earlier-single improvements;', run.stderr)
        self.assertIn('exact-SSE ties: raw identity beats phase label and insertion order', run.stderr)

    def test_ti_phase_insertion_reversal_preserves_physical_winner(self):
        source = Path('/Users/tvermeij/EMsoftData/overlap_ti_gain_audit_20261116/live_overlap.json')
        assets = Path('/Users/tvermeij/Dropbox/Work/EMPA/Research/Collaborations/Kamila/cryo Ti')
        masters = [assets / 'Ti_30kv_HCP.h5', assets / 'Ti_30kv_FCC.h5']
        if not source.exists() or not all(p.exists() for p in masters):
            self.skipTest('local frozen native Ti poses and HCP/FCC master assets required')
        frozen = json.loads(source.read_text())
        count = len(frozen['cells'])
        phase = np.asarray(frozen['arrays']['phase']['values']).reshape(count, 2)
        quats = np.asarray(frozen['arrays']['quaternions']['values'], np.float32).reshape(count, 2, 4)
        row = next(i for i, p in enumerate(phase) if sorted(p) == [0, 1])
        poses = [quats[row, np.flatnonzero(phase[row] == p)[0]] for p in (0, 1)]
        settings = dict(static_background='none', dynamic_background='none', mask=False,
                        clahe=False, master_smooth=.3, bandwidth=64, dict_grid=90,
                        refine_iters=12, batch=1, binning=1, full_resolution='none',
                        method='spherical')
        geometry = dict(pc=(.48, -.478, .719), sample_tilt_deg=-20,
                        detector_tilt_deg=0, px_um=0, step_x_um=.025, step_y_um=.025)
        def prepared(image):
            image = image.astype(np.float64)
            image -= image.mean()
            return image / np.linalg.norm(image)
        results = []
        for order in ([0, 1], [1, 0]):
            with Solver() as s:
                s.configure(**settings)
                for p in order:
                    s.add_phase(masters[p])
                s.set_patterns(np.zeros((1, 1, 64, 78), np.float32))
                s.set_geometry(**geometry)
                images = [s.simulate(poses[p][None], cells=[0], phase=order.index(p))[0] for p in (0, 1)]
                target = (.8 * prepared(images[0]) + .6 * prepared(images[1])).astype(np.float32)
                s.set_patterns(target[None, None])
                s.index()
                stored = {k: v.copy() for k, v in s.results().items() if isinstance(v, np.ndarray)}
                result = s.overlap_cells([0], max_components=2, primary='stored', blur_gain=True, gain_domain='contrast')
                self.assertEqual(int(result['retained_order'][0]), 2)
                self.assertEqual(sorted(result['phase'][0].tolist()), [0, 1])
                self.assertTrue(result['objective_valid'][0])
                for key, value in stored.items():
                    np.testing.assert_array_equal(s.results()[key], value, err_msg='stored primary ' + key)
                physical = np.asarray(order, np.int32)[result['phase']]
                result['phase'] = physical
                results.append(result)
        self.assertEqual(int(results[0]['phase'][0, 0]), int(results[1]['phase'][0, 0]))
        compared = []
        for key, value in results[0].items():
            if isinstance(value, np.ndarray):
                np.testing.assert_array_equal(results[1][key], value, err_msg=key)
                compared.append(key)
        out = os.environ.get('ZEBSDSOLVE_ISOLATION_ARTIFACTS')
        if out:
            Path(out, 'ti_phase_reversal.json').write_text(json.dumps(dict(
                source=str(source), masters=[str(p) for p in masters], frozen_row=row,
                quaternion_convention='native wxyz sample-to-crystal, used unchanged',
                geometry=geometry, settings=settings, physical_phase=results[0]['phase'].tolist(),
                sse=results[0]['sse'].tolist(), arrays_bit_identical=compared,
                stored_primary_unchanged=True), indent=2))


if __name__ == '__main__':
    unittest.main()
