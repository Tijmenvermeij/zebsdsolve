"""Stored acquisition quality fields: shape, preference and pixel-order checks."""
from pathlib import Path
import sys,tempfile,unittest
import h5py
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'python'))
from zebsdsolve import Solver

class StoredOverview(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(prefix='zebsdsolve-overview-')
        self.path=Path(self.tmp.name)/'scan.h5'
    def tearDown(self): self.tmp.cleanup()
    def fixture(self,fields):
        with h5py.File(self.path,'w') as f:
            for scan,values in fields.items():
                g=f.create_group(scan+'/EBSD');h=g.create_group('Header');d=g.create_group('Data')
                for k,v in [('n_rows',2),('n_columns',3),('pattern_height',8),('pattern_width',10)]:h.create_dataset(k,data=[v])
                d.create_dataset('patterns',data=np.arange(6*8*10,dtype='u1').reshape(6,8,10))
                for name,array in values.items():d.create_dataset(name,data=array)
    def test_native_scalar_order_types_and_layouts(self):
        for shape in [(6,),(2,3),(6,1),(1,6)]:
            with self.subTest(shape=shape):
                self.fixture({'1':{'Band Contrast':np.array([5,1,8,3,6,2],dtype='u2').reshape(shape)}})
                with Solver() as s:
                    s.open(self.path);r=s.stored_overview()
                    self.assertEqual(r['label'],'Band contrast')
                    self.assertTrue(np.array_equal(r['values'],[[5,1,8],[3,6,2]]))
    def test_preference_and_invalid_field_fallback(self):
        self.fixture({'1':{'Band Contrast':np.arange(6,dtype='u1'),'IQ':np.full(6,77,dtype='f4')}})
        with Solver() as s:
            s.open(self.path);self.assertEqual(s.stored_overview()['kind'],1)
        self.fixture({'1':{'Band Contrast':np.arange(6).reshape(3,2),'Image Quality':np.array([1,2,np.nan,4,5,6],dtype='f4')}})
        with Solver() as s:
            s.open(self.path);r=s.stored_overview()
            self.assertEqual(r['kind'],2);self.assertTrue(np.isnan(r['values'][0,2]))
            self.assertEqual(r['values'][1,2],6)
    def test_optional_and_other_acquisition_fields_not_used(self):
        self.fixture({'1':{},'2':{'IQ':np.full(6,77,dtype='f4')}})
        with Solver() as s:
            s.open(self.path);self.assertIsNone(s.stored_overview())
            s.set_patterns(np.zeros((2,3,8,10),dtype='u1'));self.assertIsNone(s.stored_overview())
    def test_non_numeric_field_is_not_a_quality_map(self):
        self.fixture({'1':{'Band Contrast':np.array([b'a']*6)}})
        with Solver() as s:
            s.open(self.path);self.assertIsNone(s.stored_overview())

if __name__=='__main__':unittest.main(verbosity=2)
