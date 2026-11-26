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
        older = SimpleNamespace(_lib=SimpleNamespace(
            ze_overlap_cells_v6=True, ze_overlap_profile_version=lambda: 1,
            ze_overlap_profile_trace_capacity=True))
        with self.assertRaisesRegex(Error,'updated native library'):
            Solver.overlap_cells(older,[1],polish=True,blur_gain=True)
        for polish, blur_gain in [(False, True), (True, False)]:
            with self.assertRaisesRegex(ValueError,'requires both'):
                Solver.overlap_cells(old,[1],polish=polish,blur_gain=blur_gain,
                                     refit_blur_gain_after_polish=True)

class ProfileProduction(unittest.TestCase):
    def setUp(self):
        self.fixture=fixtures.NativeDiscovery();self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups);self.s=self.fixture.s
        self.s.configure(method='spherical')
    def test_conditional_polish_and_inspection(self):
        old=self.s.overlap_cells([1,2],max_components=1,blur_gain=True)
        polished_one=self.s.overlap_cells([1,2],max_components=1,blur_gain=True,polish=True)
        new=self.s.overlap_cells([1,2],blur_gain=True,polish=True)
        self.assertEqual(new['profile_version'],3)
        self.assertTrue(np.all(polished_one['sse']<=old['sse']+3e-6),(old['sse'],polished_one['sse']))
        self.assertEqual(new['profile_optimizer']['method'], 'conditional_detector_gauss_newton')
        self.assertEqual(new['source_quaternions'].shape,(2,2,4))
        np.testing.assert_allclose(new['incremental_improvement'],
                                   np.maximum(0, new['previous_sse'] - new['sse']),
                                   atol=3e-6, rtol=0)
        for row,cell in enumerate([1,2]):
            images=self.s.overlap_inspect(cell,new,row=row)
            self.assertAlmostEqual(float(np.sum(images['residual'].astype(float)**2)),float(new['sse'][row]),delta=5e-5)
    def test_polish_holds_entry_blur_gain_fixed(self):
        self.s.configure(refine=False)  # Leave the coarse search pose for polishing to move.
        options = dict(max_components=1, blur_gain=True, min_residual_ncc=0)
        entry = self.s.overlap_cells([1,2], **options)
        fixed = self.s.overlap_cells([1,2], polish=True, **options)
        refit = self.s.overlap_cells([1,2], polish=True, refit_blur_gain_after_polish=True, **options)
        self.assertTrue(np.any(fixed['sse'] < entry['sse'] - 1e-4))
        np.testing.assert_array_equal(fixed['blur_gain_theta'], entry['blur_gain_theta'])
        np.testing.assert_array_equal(refit['quaternions'], fixed['quaternions'])
        self.assertTrue(np.all(refit['sse'] <= fixed['sse'] + 3e-6))
        self.assertTrue(np.any(refit['sse'] < fixed['sse'] - 1e-5))
        self.assertFalse(np.array_equal(refit['blur_gain_theta'], fixed['blur_gain_theta']))
        self.assertFalse(fixed['refit_blur_gain_after_polish'])
        self.assertTrue(refit['refit_blur_gain_after_polish'])
        self.assertEqual(fixed['profile_optimizer']['nuisance_schedule'], 'discovery_only')
        self.assertEqual(refit['profile_optimizer']['nuisance_schedule'], 'discovery_then_isolated_endpoint')

    def test_refinement_is_stable_across_tile_budgets_and_scan_positions(self):
        self.s.configure(refine=False, mask=True, mask_radius=.85)
        self.s.set_patterns(np.zeros((1, 4, 96, 96), np.float32))
        cells = [2, 1, 3, 0, 1]
        q0, q1 = self.fixture.q[:2]
        primary = fixtures.unit(self.s.simulate(np.tile(q0, (4, 1)), cells=np.arange(4), phase=0))
        secondary = fixtures.unit(self.s.simulate(np.tile(q1, (4, 1)), cells=np.arange(4), phase=0))
        mixture = primary + np.array([0, .6, .4, .2], np.float32)[:, None, None] * secondary
        self.s.set_patterns(mixture[None])
        self.s.index()
        options = dict(primary='stored', max_components=2, min_residual_ncc=.1, blur_gain=True,
                       polish=True, refit_blur_gain_after_polish=True)
        batched = self.s.overlap_cells(cells, **options)
        self.s.configure(tile_mb=8)
        scalar = self.s.overlap_cells(cells, **options)
        self.assertTrue(np.any(batched['retained_order'] == 2))
        self.assertTrue(np.any(abs(batched['quaternions'] - batched['source_quaternions']) > 1e-4))
        for field in ('phase', 'retained_order', 'stop_reason', 'winner_rank'):
            np.testing.assert_array_equal(batched[field], scalar[field], err_msg=field)
        for field in ('quaternions', 'amplitudes', 'blur_gain_theta', 'sse'):
            np.testing.assert_allclose(batched[field], scalar[field], atol=1e-6, rtol=0, err_msg=field)
        self.assertEqual(len(batched['profile_events']), len(scalar['profile_events']))
        for a, b in zip(batched['profile_events'], scalar['profile_events']):
            for key in ('cell', 'branch', 'order', 'attempt', 'phase', 'rank', 'flags', 'result_order'):
                self.assertEqual(a[key], b[key])

    def test_single_component_does_not_prepare_residual_bandwidth(self):
        self.s.index()
        set_band = self.s._lib.ze_overlap_set_next_bandwidth
        set_band.argtypes = (ctypes.c_void_p, ctypes.c_uint32)
        set_band.restype = ctypes.c_int32
        for screen in (None, .1):
            options = dict(max_components=1, primary='stored', keep_n=3,
                           blur_gain=True, min_residual_ncc=screen)
            ordinary = self.s.overlap_cells([1], **options)
            self.s.events()
            self.assertEqual(set_band(self.s._e, 256), 256)
            custom = self.s.overlap_cells([1], **options)
            self.assertFalse(any('bandwidth 256,' in event.get('text', '') for event in self.s.events()))
            for field in ('sse', 'phase', 'quaternions', 'amplitudes', 'blur_gain_theta'):
                np.testing.assert_array_equal(custom[field], ordinary[field])

    def test_first_residual_batch_matches_independent_calls(self):
        from pathlib import Path
        self.s.clear_phases()
        self.s.add_phase(Path(__file__).resolve().parents[1]/'testdata/nickel/Ni_master_20kV.npz')
        cells = [1, 0, 4, 1]
        for method in ('spherical', 'dictionary'):
            self.s.configure(method=method)
            self.s.index()
            options = dict(primary='stored', max_components=2, keep_n=1, blur_gain=True,
                           min_residual_ncc=.1, min_stage_improvement=[.001, None, None])
            batch = self.s.overlap_cells(cells, **options)
            self.assertTrue(np.any(batch['retained_order'] == 2))
            self.assertTrue(np.any(batch['retained_order'] == 1))
            for row, cell in enumerate(cells):
                independent = self.s.overlap_cells([cell], **options)
                for field in ('phase', 'quaternions', 'amplitudes', 'proposal_ncc', 'winner_rank',
                              'blur_gain_theta', 'sse', 'stop_reason', 'retained_order'):
                    with self.subTest(method=method, row=row, field=field):
                        np.testing.assert_array_equal(batch[field][row], independent[field][0])

    def test_batched_mixed_phase_profiles_match_single_lane_with_partial_batches(self):
        # Exercise shuffled and repeated cells, a missing primary, and both
        # rank-one and pooled histories on independently positioned patterns.
        q0, q1, q2 = self.fixture.q
        q3 = np.array([.5, -.5, .5, -.5], np.float32)
        ids = np.arange(6)
        patterns = [
            fixtures.unit(self.s.simulate(np.tile(q, (6, 1)), cells=ids, phase=phase))
            for q, phase in [(q0, 0), (q1, 0), (q2, 1), (q3, 1)]
        ]
        mixture = np.einsum('k,khij->hij', [.8, .6, .5, .4], patterns).astype(np.float32)
        mixture[3] = 0
        self.s.set_patterns(mixture[None])
        self.s.index()
        cells = [4, 2, 3, 1, 4]
        options = dict(primary='stored', max_components=4, keep_n=2, blur_gain=True,
                       min_residual_ncc=.1, min_stage_improvement=[.001, .001, .001])
        self.s.configure(batch=1)
        scalar = self.s.overlap_cells(cells, **options)
        self.s.configure(batch=3)
        batched = self.s.overlap_cells(cells, **options)
        self.assertTrue(any(event['order'] == 4 for event in scalar['profile_events']))
        self.assertTrue(any(event['branch'] == 1 for event in scalar['profile_events']))
        self.assertTrue(np.any(scalar['phase'] == 1))
        # One row can still fill multiple fit lanes with phase/rank candidates.
        self.s.configure(batch=4)
        one = self.s.overlap_cells(cells[:1], **options)
        for field in ('phase', 'retained_order', 'blur_gain_theta', 'quaternions', 'sse'):
            np.testing.assert_array_equal(one[field][0], scalar[field][0], err_msg=field)
        for field in ('phase', 'retained_order', 'stop_reason', 'winner_rank', 'trajectory_source'):
            np.testing.assert_array_equal(batched[field], scalar[field], err_msg=field)
        for field in ('quaternions', 'amplitudes', 'proposal_ncc', 'blur_gain_theta', 'sse', 'component_norms'):
            np.testing.assert_allclose(batched[field], scalar[field], atol=1e-6, rtol=0, err_msg=field)
        self.assertEqual(len(batched['profile_events']), len(scalar['profile_events']))
        for actual, expected in zip(batched['profile_events'], scalar['profile_events']):
            for key in ('cell_index', 'branch', 'order', 'attempt', 'phase', 'rank', 'flags', 'result_order'):
                self.assertEqual(actual[key], expected[key], key)
            for key in ('source_q', 'residual_ncc', 'baseline_sse', 'lower_sse', 'joint_sse'):
                np.testing.assert_allclose(actual[key], expected[key], atol=1e-6, rtol=0, err_msg=key)

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
        options = dict(blur_gain=True, polish=True, refit_blur_gain_after_polish=True, min_residual_ncc=1)
        one=self.s.overlap_cells([1,2],max_components=1,blur_gain=True,min_residual_ncc=1)
        screened=self.s.overlap_cells([1,2],max_components=2,**options)
        np.testing.assert_array_equal(screened['sse'],one['sse'])
        np.testing.assert_array_equal(screened['quaternions'][:,0],one['quaternions'][:,0])
        np.testing.assert_array_equal(screened['amplitudes'][:,0],one['amplitudes'][:,0])
        np.testing.assert_array_equal(screened['blur_gain_theta'],one['blur_gain_theta'])
        self.assertTrue(np.all(screened['retained_order']==1))
        rejected = [e for e in screened['profile_events'] if e['flags'] & 2]
        self.assertTrue(rejected)
        self.assertTrue(all(np.isnan(e['joint_sse']) and np.isnan(e['lower_sse']) for e in rejected))
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
