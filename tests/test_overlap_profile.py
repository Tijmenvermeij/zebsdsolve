"""V6 profile contract and GPU integration against once-prepared native fixtures."""
import ctypes
import unittest
from types import SimpleNamespace
import numpy as np
from zebsdsolve import Solver, Error, _native as nt
import test_overlap as fixtures

class ProfileContract(unittest.TestCase):
    def test_layout(self):
        self.assertEqual(ctypes.sizeof(nt.OverlapProfileOptions),16)
        self.assertEqual(ctypes.sizeof(nt.OverlapProfileEvent),64)
        self.assertEqual(nt.OverlapProfileEvent.source_q.offset,32)
        self.assertEqual(nt.OverlapProfileEvent.joint_sse.offset,60)

    def test_options_and_old_runtime_fail_before_native_work(self):
        old=SimpleNamespace(_lib=object())
        for value in [float('nan'),float('inf'),2,-2,True]:
            with self.assertRaises(ValueError):Solver.overlap_cells(old,[1],min_residual_ncc=value)
        for value in [0,6,-1,float('nan'),True]:
            with self.assertRaises(ValueError):Solver.overlap_cells(old,[1],polish_radius_deg=value)
        with self.assertRaisesRegex(Error,'native V6'):Solver.overlap_cells(old,[1],polish=True)
        with self.assertRaisesRegex(Error,'native V6'):Solver.overlap_cells(old,[1],min_residual_ncc=0)
        with self.assertRaisesRegex(ValueError,'contrast'):Solver.overlap_cells(old,[1],polish=True,gain_domain='source_dc')

class ProfileProduction(unittest.TestCase):
    def setUp(self):
        self.fixture=fixtures.NativeDiscovery();self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups);self.s=self.fixture.s
        self.s.configure(method='spherical')
    def test_monotone_polish_and_inspection(self):
        old=self.s.overlap_cells([1,2],max_components=1,blur_gain=True)
        polished_one=self.s.overlap_cells([1,2],max_components=1,blur_gain=True,polish=True)
        new=self.s.overlap_cells([1,2],blur_gain=True,polish=True)
        self.assertEqual(new['profile_version'],1)
        self.assertTrue(np.all(polished_one['sse']<=old['sse']+3e-6),(old['sse'],polished_one['sse']))
        self.assertTrue(np.all(new['sse']<=polished_one['sse']+3e-6),(polished_one['sse'],new['sse']))
        self.assertEqual(new['source_quaternions'].shape,(2,2,4))
        for row,cell in enumerate([1,2]):
            images=self.s.overlap_inspect(cell,new,row=row)
            self.assertAlmostEqual(float(np.sum(images['residual'].astype(float)**2)),float(new['sse'][row]),delta=5e-5)
    def test_batched_primary_seeds_match_independent_stored_cells(self):
        self.s.index()
        cells = [4, 2, 1, 0]
        options = dict(primary='stored', max_components=1, blur_gain=True, min_residual_ncc=0)
        batch = self.s.overlap_cells(cells, **options)
        self.assertEqual(batch['cells'].tolist(), cells)
        self.assertTrue(np.all(batch['active_count'] == 1))
        for row, cell in enumerate(cells):
            independent = self.s.overlap_cells([cell], **options)
            for field in ('phase', 'source_quaternions', 'quaternions', 'amplitudes',
                          'component_norms', 'blur_gain_theta', 'sse'):
                with self.subTest(cell=cell, field=field):
                    np.testing.assert_array_equal(batch[field][row], independent[field][0])

    def test_screen_rollback_and_proposal_trace(self):
        one=self.s.overlap_cells([1,2],max_components=1,blur_gain=True,min_residual_ncc=1)
        screened=self.s.overlap_cells([1,2],max_components=2,blur_gain=True,min_residual_ncc=1)
        np.testing.assert_array_equal(screened['sse'],one['sse'])
        np.testing.assert_array_equal(screened['quaternions'][:,0],one['quaternions'][:,0])
        np.testing.assert_array_equal(screened['amplitudes'][:,0],one['amplitudes'][:,0])
        np.testing.assert_array_equal(screened['blur_gain_theta'],one['blur_gain_theta'])
        self.assertTrue(np.all(screened['retained_order']==1))
        self.assertTrue(any(e['flags']&2 for e in screened['profile_events']))
        self.assertTrue(all(e['cell'] in [1,2] for e in screened['profile_events']))
    def test_higher_orders_keep_coherent_complete_histories(self):
        from pathlib import Path
        self.s.clear_phases()
        self.s.add_phase(Path(__file__).resolve().parents[1]/'testdata/nickel/Ni_master_20kV.npz')
        self.s.set_geometry(step_x_um=0,step_y_um=0)
        # Native forward projections only; no custom symmetry or angle math.
        q=np.random.default_rng(714).normal(size=(4,4));q=(q/np.linalg.norm(q,axis=1,keepdims=True)).astype(np.float32)
        base=fixtures.unit(self.s.simulate(q,cells=[0]*4))
        patterns=np.zeros((2,48,48),np.float32)
        patterns[0]=np.einsum('i,ihw->hw',[.7,.5,.3],base[:3])
        patterns[1]=np.einsum('i,ihw->hw',[.7,.5,.3,.2],base)
        self.s.set_patterns(patterns)
        previous=None
        for k in (2,3,4):
            z=self.s.overlap_cells([0,1],max_components=k,keep_n=2,polish=True,blur_gain=False)
            self.assertTrue(np.all(z['objective_valid']))
            if previous is not None:self.assertTrue(np.all(z['sse']<=previous+3e-6))
            previous=z['sse']
            self.assertTrue(np.all(z['retained_order']<=k))
            self.assertEqual(int(z['retained_order'][1]),k,'four-source fixture must enter each requested order')
            for row,cell in enumerate([0,1]):
                im=self.s.overlap_inspect(cell,z,row=row)
                self.assertAlmostEqual(float(np.sum(im['residual'].astype(float)**2)),float(z['sse'][row]),delta=5e-5)
                events=[e for e in z['profile_events'] if e['cell']==cell and e['flags']&128]
                self.assertTrue(events)
                self.assertTrue(all(e['branch']==z['trajectory_source'][row] for e in events))
                if z['selection_version']==2:
                    witness, = [e for e in events if e['flags'] & 256]
                    self.assertFalse(witness['flags'] & 2)

    def test_stage_gate_stops_before_searching_the_next_order(self):
        from pathlib import Path
        self.s.clear_phases()
        self.s.add_phase(Path(__file__).resolve().parents[1]/'testdata/nickel/Ni_master_20kV.npz')
        self.s.set_geometry(step_x_um=0,step_y_um=0)
        q=np.random.default_rng(714).normal(size=(4,4))
        q=(q/np.linalg.norm(q,axis=1,keepdims=True)).astype(np.float32)
        base=fixtures.unit(self.s.simulate(q,cells=[0]*4))
        self.s.set_patterns(np.einsum('i,ihw->hw',[.7,.5,.3,.2],base)[None].astype(np.float32))
        for rejected_order in (2,3,4):
            limits = [float(n==rejected_order) for n in (2,3,4)]
            result = self.s.overlap_cells([0],max_components=4,keep_n=2,polish=True,
                                          min_residual_ncc=-1,min_stage_improvement=limits)
            self.assertEqual(result['stage_gate']['minimum'], limits)
            self.assertEqual(int(result['retained_order'][0]), rejected_order-1)
            self.assertEqual(result['stop_reason_name'][0], 'stage_improvement')
            events = result['profile_events']
            self.assertTrue(all(e['order']<=rejected_order for e in events))
            examined = [e for e in events if e['order']==rejected_order and e['flags']&1024]
            self.assertTrue(examined)
            self.assertTrue(all(not e['flags']&2048 and (e['lower_sse']-e['joint_sse'])/result['initial_sse'][0]<1 for e in examined))
            if rejected_order > 2:
                witness, = [e for e in events if e['flags']&256]
                self.assertTrue(witness['flags']&2048)
            image = self.s.overlap_inspect(0,result)
            self.assertAlmostEqual(float(np.sum(image['residual'].astype(float)**2)),float(result['sse'][0]),delta=5e-5)

    def test_identity_gain_stays_identity(self):
        result=self.s.overlap_cells([1],polish=True,blur_gain=False)
        np.testing.assert_array_equal(result['blur_gain_theta'],[[0,0,1,1,1,0,0]])

if __name__=='__main__':unittest.main()
