"""h5py oracle for targeted, interleaved and complete HDF5 chunk lookup.

Uses fresh native library through ZEBSDSOLVE_LIB; no crystallographic changes.
"""
from pathlib import Path
import sys
import tempfile
import unittest
import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'python'))
from zebsdsolve import Solver, Error


def fixture(path, chunked=True):
    n = 2048
    expected = np.zeros((n, 5, 7), dtype=np.uint16)
    with h5py.File(path, 'w', libver='earliest') as f:
        h = f.create_group('Scan 1/EBSD/Header')
        for key, value in [('nRows', 32), ('nColumns', 64), ('Pattern Height', 5), ('Pattern Width', 7)]: h.create_dataset(key,data=[value])
        options = dict(chunks=(1,3,4), compression='gzip', shuffle=True, fletcher32=True) if chunked else {}
        d = f.create_dataset('Scan 1/EBSD/Data/Pattern', shape=expected.shape, dtype='u2', **options)
        for i in range(n):
            # Missing leading/interior/trailing chunks, and edge chunks on both image axes.
            if i % 7 in (0, 3) or i == n-1: continue
            expected[i] = (i+np.arange(35).reshape(5,7)) % 65535
            d[i] = expected[i]
    return expected


class ChunkIndex(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory(prefix='zebsdsolve-chunks-')
        cls.path = Path(cls.tmp.name)/'split-compressed.h5'
        cls.expected = fixture(cls.path)

    @classmethod
    def tearDownClass(cls): cls.tmp.cleanup()

    def solver(self, path=None):
        s = Solver(); self.addCleanup(s.close)
        s.configure(binning=1, static_background='none', dynamic_background='none', mask=False)
        s.open(path or self.path)
        return s

    def test_targeted_interleaved_and_complete_are_identical(self):
        s = self.solver()
        before = s.index_preparation
        for cell in [0,1,2,3,511,1024,2046,2047]:
            np.testing.assert_array_equal(s.pattern(cell,processed=False,full=True), self.expected[cell])
        targeted = s.index_preparation
        self.assertFalse(targeted['ready'])
        self.assertEqual(targeted['visited'], 0)
        self.assertLess(targeted['reads']-before['reads'], 40)
        steps = 0
        while True:
            before = s.index_preparation
            complete = s.prepare_index_step(max_nodes=2,max_bytes=16384)
            after = s.index_preparation
            self.assertLessEqual(after['reads']-before['reads'], 1)
            self.assertLessEqual(after['bytes']-before['bytes'], 16384)
            self.assertLessEqual(after['visited']-before['visited'], 2)
            steps += 1
            if complete: break
            self.assertFalse(after['ready'])
            cell = (steps*79) % 2048
            np.testing.assert_array_equal(s.pattern(cell,processed=False,full=True), self.expected[cell])
            self.assertLess(steps,1000)
        self.assertGreater(steps, 10)
        self.assertGreater(s.index_preparation['visited'], 10)
        self.assertTrue(s.index_preparation['ready'])
        self.assertEqual(s.index_preparation['pending'], 0)
        for cell in [0,1,127,500,1000,2046,2047]:
            before = s.index_preparation
            np.testing.assert_array_equal(s.pattern(cell,processed=False,full=True), self.expected[cell])
            # Complete lookup needs only the payload read (zero for missing chunks).
            self.assertLessEqual(s.index_preparation['reads']-before['reads'], 1)
        self.assertTrue(s.prepare_index_step())

    def test_small_budget_does_not_poison_previews_and_scan_switch_resets(self):
        s = self.solver()
        with self.assertRaisesRegex(Error, 'ChunkIndexBudgetTooSmall'): s.prepare_index_step(max_bytes=1)
        self.assertFalse(s.index_preparation['failed'])
        np.testing.assert_array_equal(s.pattern(21,processed=False,full=True), self.expected[21])
        s.prepare_index()
        self.assertTrue(s.index_preparation['ready'])
        s.open(self.path)
        self.assertFalse(s.index_preparation['ready'])
        self.assertEqual(s.index_preparation['visited'],0)

    def test_contiguous_preparation_reads_no_pattern_data(self):
        path = Path(self.tmp.name)/'contiguous.h5'
        expected = fixture(path,False)
        s = self.solver(path)
        before = s.index_preparation
        self.assertTrue(s.prepare_index_step())
        self.assertEqual(s.index_preparation['reads'], before['reads'])
        self.assertEqual(s.index_preparation['visited'],0)
        np.testing.assert_array_equal(s.pattern(91,processed=False,full=True),expected[91])

    def test_rank4_scan_cells_before_during_and_after_preparation(self):
        path = Path(self.tmp.name)/'rank4.h5'
        expected = self.expected.reshape(32,64,5,7)
        with h5py.File(self.path) as src, h5py.File(path,'w',libver='earliest') as dst:
            src.copy('Scan 1/EBSD/Header',dst.create_group('Scan 1/EBSD'),name='Header')
            ds=dst.create_dataset('Scan 1/EBSD/Data/Pattern', shape=expected.shape, dtype='u2', chunks=(1,2,3,4), compression='gzip',shuffle=True)
            for row in range(0,32,2): ds[row] = expected[row]
        oracle = expected.copy(); oracle[1::2] = 0
        s=self.solver(path)
        for cell in [0,1,63,64,2047]: np.testing.assert_array_equal(s.pattern(cell,processed=False,full=True),oracle.reshape(2048,5,7)[cell])
        while not s.prepare_index_step(max_nodes=1):
            np.testing.assert_array_equal(s.pattern(129,processed=False,full=True),oracle.reshape(2048,5,7)[129])
        for cell in [0,1,63,64,2047]: np.testing.assert_array_equal(s.pattern(cell,processed=False,full=True),oracle.reshape(2048,5,7)[cell])

    def test_cyclic_tree_fails_instead_of_finishing_or_spinning(self):
        path = Path(self.tmp.name)/'cyclic.h5'
        data = bytearray(self.path.read_bytes())
        roots = [i for i in range(len(data)-8) if data[i:i+5] == b'TREE\x01' and data[i+5]>0]
        self.assertTrue(roots)
        root = max(roots, key=lambda i:data[i+5])
        # superblock-v0 8-byte addresses, rank3 chunk key40; first child at head24+key40.
        data[root+64:root+72] = root.to_bytes(8,'little')
        path.write_bytes(data)
        s = self.solver(path)
        with self.assertRaisesRegex(Error,'BadHdf5'):
            for _ in range(1000):
                if s.prepare_index_step(): self.fail('Corrupt lookup became ready')
        self.assertTrue(s.index_preparation['failed'])
        self.assertFalse(s.index_preparation['ready'])


if __name__ == '__main__': unittest.main(verbosity=2)
