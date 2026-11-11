"""Native same-page checkpoint: exact maps, candidates, exports and atomic rejection.

No scientific file is imported or converted here: native result state is restored
under the identical acquisition context. Run against a locally rebuilt library.
"""
from pathlib import Path
import ctypes as C
import hashlib
import json
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'python'))
from zebsdsolve import Solver


class ResultCheckpoint(unittest.TestCase):
    def setUp(self):
        self.s = Solver()
        self.addCleanup(self.s.close)
        self.lib = self.s._lib
        self.lib.ze_result_checkpoint.argtypes = [C.c_void_p, C.c_void_p, C.c_size_t]
        self.lib.ze_result_checkpoint.restype = C.c_int64
        self.lib.ze_restore_result_checkpoint.argtypes = [C.c_void_p, C.c_void_p, C.c_size_t]
        self.lib.ze_restore_result_checkpoint.restype = C.c_int32
        self.s.configure(bandwidth=16, fit_bandwidth=16, refine_iters=1,
                         static_background='none', dynamic_background='none',
                         toy_rows=2, toy_cols=3, toy_height=24, toy_width=32,
                         toy_counts=0, master_smooth=0)
        for file in ['nickel/Ni_master_20kV.npz', 'gan/GaN_master_20kV.npz']:
            self.s.add_phase(ROOT / 'testdata' / file)
        self.s.synthesize()
        self.s.configure(roi_x=1, roi_y=0, roi_w=1, roi_h=2)
        self.s.index()

    def capture(self):
        n = self.lib.ze_result_checkpoint(self.s._e, None, 0)
        self.assertGreater(n, 112, self.s._error())
        out = C.create_string_buffer(n)
        self.assertEqual(self.lib.ze_result_checkpoint(self.s._e, out, n), n)
        return out.raw

    def restore(self, data):
        rc = self.lib.ze_restore_result_checkpoint(self.s._e, data, len(data))
        self.last_error = self.s._error()
        return rc

    def state(self):
        def text(kind):
            n = self.lib.ze_export(self.s._e, kind, None, 0)
            self.assertGreater(n, 0, self.s._error())
            out = C.create_string_buffer(n)
            self.assertEqual(self.lib.ze_export(self.s._e, kind, out, n), n)
            return out.raw
        return (text(0), text(1),
                json.dumps(self.s.export_snapshot(), sort_keys=True),
                [self.s.map(kind).tobytes() for kind in ['ipf_x', 'ipf_y', 'ipf_z', 'ncc', 'nmi', 'phase', 'z', 'sweep']])

    def test_exact_state_and_atomic_rejection(self):
        before = self.state()
        checkpoint = self.capture()
        # Both unvisited candidates are canonical zeros, never allocator bytes.
        n, phases = 6, 2
        candidates = checkpoint[112 + n * 37:]
        self.assertEqual(len(candidates), n * phases * 36)
        for phase in range(phases):
            for cell in [0, 2, 3, 5]:
                record = candidates[(phase*n+cell)*36:][:36]
                self.assertEqual(record, bytes(32) + b'\xff'*4)
        # Change generation by reopening masters; replay exact settings/geometry.
        settings, geometry = self.s.settings, self.s.geometry
        self.s.clear_phases()
        for file in ['nickel/Ni_master_20kV.npz', 'gan/GaN_master_20kV.npz']:
            self.s.add_phase(ROOT / 'testdata' / file)
        self.assertEqual(self.restore(checkpoint), 0, self.s._error())
        self.assertEqual(self.state(), before)
        self.assertEqual(self.capture(), checkpoint)
        for corrupted in [checkpoint[:-1], b'BADMAGIC'+checkpoint[8:], checkpoint[:150]+bytes([checkpoint[150]^1])+checkpoint[151:]]:
            self.assertLess(self.restore(corrupted), 0)
            self.assertEqual(self.state(), before)
        # Even a checksummed payload cannot smuggle invalid indices into results.
        corrupted = bytearray(checkpoint)
        corrupted[112+28:112+32] = (999).to_bytes(4, 'little')
        corrupted[80:112] = hashlib.sha256(corrupted[112:]).digest()
        self.assertLess(self.restore(bytes(corrupted)), 0)
        self.assertEqual(self.state(), before)
        self.s.configure(roi_x=0)
        self.assertLess(self.restore(checkpoint), 0)
        self.s.configure(**settings)
        self.s.set_geometry(pc=[.1,.2,.3])
        self.assertLess(self.restore(checkpoint), 0)
        self.s.set_geometry(**geometry)
        self.assertEqual(self.restore(checkpoint), 0, self.s._error())
        self.assertEqual(self.state(), before)


if __name__ == '__main__':
    unittest.main()
