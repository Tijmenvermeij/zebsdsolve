"""Fixed prepared-domain GPU mixture checks; run locally after zig build.

No external crystallographic conversion: native simulations define the fixtures and
native disorientation compares discovery poses. NumPy is only the float64 linear
algebra oracle for the explicitly non-crystallographic projection fit.
"""
from itertools import combinations
from pathlib import Path
import sys
import unittest
import ctypes as C
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'python'))
from zebsdsolve import Solver, Error


def unit(x):
    x = np.asarray(x, np.float64)
    x = x - x.mean(axis=(-2, -1), keepdims=True)
    norm = np.sqrt((x*x).sum(axis=(-2, -1), keepdims=True))
    return np.divide(x, norm, out=np.zeros_like(x), where=norm > 1e-15)


def oracle(y, templates):
    """Independent float64 active-set oracle (up to four fixed vectors)."""
    y = unit(y).ravel()
    a = unit(templates).reshape(len(templates), -1).T
    best, coeff = float(y @ y), np.zeros(a.shape[1])
    for count in range(1, a.shape[1] + 1):
        for active in combinations(range(a.shape[1]), count):
            candidate = np.linalg.lstsq(a[:, active], y, rcond=1e-10)[0]
            if candidate.min() < -1e-10:
                continue
            x = np.zeros(a.shape[1])
            x[list(active)] = np.maximum(candidate, 0)
            ss = float(np.sum((y - a @ x)**2))
            if ss < best:
                best, coeff = ss, x
    return coeff, y - a @ coeff, best


class FixedProjectionFit(unittest.TestCase):
    def setUp(self):
        self.s = Solver()
        self.addCleanup(self.s.close)

    def test_gpu_float64_oracle_signed_residual_and_affine_invariance(self):
        rng = np.random.default_rng(74)
        for k in [1, 2, 4]:
            templates = unit(rng.normal(size=(k, 3, 11, 17)))
            coefficients = np.arange(1, k+1) / k
            patterns = np.einsum('knhw,k->nhw', templates, coefficients) + .02*rng.normal(size=(3, 11, 17))
            originals = (patterns.copy(), templates.copy())
            r = self.s.fit_overlap_projections(patterns, templates)
            for b in range(3):
                amp, residual, ss = oracle(patterns[b], templates[:, b])
                np.testing.assert_allclose(r['amplitudes'][b], amp, atol=3e-6)
                np.testing.assert_allclose(r['residual'][b].ravel(), residual, atol=2e-6)
                self.assertAlmostEqual(float(r['sse'][b]), ss, delta=3e-6)
            self.assertTrue(np.any(r['residual'] < 0), 'residual must retain negative pixels')
            affine = self.s.fit_overlap_projections(patterns*3 + 27, templates*2 - 19)
            np.testing.assert_allclose(affine['amplitudes'], r['amplitudes'], atol=8e-6)
            np.testing.assert_array_equal(patterns, originals[0])
            np.testing.assert_array_equal(templates, originals[1])
            if k > 2:
                self.assertTrue(r['conditioning_assessed'].all())
                self.assertTrue(np.isfinite(r['prepared_fractions']).all())

    def test_negative_projection_coefficient_is_constrained_to_zero(self):
        x = np.arange(24, dtype=float).reshape(1, 4, 6)
        r = self.s.fit_overlap_projections(x, -x[None])
        np.testing.assert_array_equal(r['amplitudes'], 0)
        np.testing.assert_allclose(r['sse'], 1, atol=1e-6)
        np.testing.assert_allclose(r['residual'], unit(x), atol=1e-6)

    def test_flat_duplicate_near_duplicate_and_dependent_three(self):
        rng = np.random.default_rng(91)
        a, b = unit(rng.normal(size=(2, 1, 10, 14)))
        flat = self.s.fit_overlap_projections(np.ones_like(a), np.stack([a, b]))
        self.assertTrue(flat['flat'].all())
        self.assertFalse(flat['conditioning_assessed'].any())
        self.assertTrue(np.isnan(flat['relative_gram_condition']).all())
        np.testing.assert_array_equal(flat['residual'], 0)
        for flat_design in [np.stack([a,a]), np.ones((2,*a.shape))]:
            r_flat = self.s.fit_overlap_projections(np.ones_like(a), flat_design)
            self.assertTrue(r_flat['flat'][0])
            self.assertFalse(r_flat['conditioning_assessed'][0])
            self.assertTrue(np.isnan(r_flat['relative_gram_condition'][0]))
        flat_template = self.s.fit_overlap_projections(a, np.stack([np.ones_like(a), a]))
        self.assertFalse(flat_template['objective_valid'][0])
        self.assertTrue(np.isnan(flat_template['amplitudes']).all())
        self.assertEqual(float(flat_template['relative_gram_condition'][0]), 0)
        self.assertLess(float(flat_template['best_single_sse'][0]), 1e-6)
        for second in [a, unit(a + .001*b)]:
            r = self.s.fit_overlap_projections(a, np.stack([a, second]))
            self.assertTrue(r['conditioned'].all())
            self.assertTrue(np.isnan(r['prepared_fractions']).all())
            self.assertFalse(r['objective_valid'][0])
            self.assertTrue(np.isnan(r['sse']).all())
            self.assertTrue(np.isnan(r['amplitudes']).all())
            self.assertLess(float(r['best_single_sse'][0]), 1e-6)
            self.assertLess(float(r['relative_gram_condition'][0]), 1e-6)
        templates = np.stack([a, b, unit(a+b)])
        r = self.s.fit_overlap_projections(a+b, templates)
        self.assertTrue(r['conditioning_assessed'].all())
        self.assertTrue(np.isnan(r['prepared_fractions']).all())
        self.assertLess(float(r['relative_gram_condition'][0]),1e-6)
        self.assertFalse(r['objective_valid'][0])
        self.assertTrue(np.isnan(r['sse']).all())
        self.assertLess(float(r['best_single_sse'][0]), 1e-6)

    def test_rank_rejection_preserves_single_fits_and_valid_boundary(self):
        a = np.array([[[1,-1],[1,-1]]], np.float32)
        b = np.array([[[1,1],[-1,-1]]], np.float32)
        # Independent nearly opposing templates have a valid positive solution
        # around (4096,4096); their Gram matrix loses that direction in f32.
        for templates in [np.stack([a, -a+b/4096]), np.stack([-a+b/4096, a])]:
            amplitude, _, ss = oracle(b[0], templates[:,0])
            self.assertTrue(np.all(amplitude > 4000))
            self.assertLess(ss, 1e-15)
            r = self.s.fit_overlap_projections(b, templates)
            self.assertEqual(r['solve_status'][0], 'numerically_unresolved')
            self.assertFalse(r['objective_valid'][0])
            for key in ['amplitudes','prepared_fractions','sse','mixture_ncc','improvement_over_best_single','residual']:
                self.assertTrue(np.isnan(r[key]).all(), key)
            self.assertTrue(np.isfinite(r['single_sse']).all())
        # Adding independent directions cannot cure the unresolved opposing pair.
        a = np.tile(a, (1,1,2))
        b = np.tile(b, (1,1,2))
        c = np.array([[[1,1,-1,-1],[1,1,-1,-1]]], np.float32)
        d = a*b
        for templates in [np.stack([a,-a+b/4096,c]), np.stack([d,a,c,-a+b/4096])]:
            r = self.s.fit_overlap_projections(b, templates)
            self.assertFalse(r['objective_valid'][0])
            self.assertTrue(np.isnan(r['sse']).all())
        # Both placements of a flat column are explicitly rank-deficient.
        for templates in [np.stack([b,np.ones_like(b)]),np.stack([np.ones_like(b),b])]:
            r = self.s.fit_overlap_projections(b, templates)
            self.assertTrue(r['conditioned'][0])
            self.assertFalse(r['objective_valid'][0])
            self.assertEqual(float(r['relative_gram_condition'][0]), 0)
            self.assertTrue(np.isnan(r['prepared_fractions']).all())
            self.assertLess(float(r['best_single_sse'][0]), 1e-6)
        # A zero NNLS coefficient in a full-rank design is a valid boundary fit.
        r = self.s.fit_overlap_projections(b, np.stack([b,a]))
        self.assertTrue(r['objective_valid'][0])
        self.assertEqual(r['solve_status'][0], 'solved')
        np.testing.assert_allclose(r['amplitudes'], [[1,0]], atol=1e-6)
        self.assertLess(float(r['sse'][0]), 1e-6)

    def test_full_spectral_ratio_and_permutations(self):
        rng=np.random.default_rng(406)
        for k in (3,4):
            templates=unit(rng.normal(size=(k,1,9,13)))
            target=np.einsum('knhw,k->nhw',templates,np.arange(1,k+1))
            expected=np.linalg.eigvalsh(templates[:,0].reshape(k,-1)@templates[:,0].reshape(k,-1).T)
            expected=expected[0]/expected[-1]
            from itertools import permutations
            reference=None
            for order in permutations(range(k)):
                r=self.s.fit_overlap_projections(target,templates[list(order)])
                self.assertTrue(r['conditioning_assessed'][0])
                self.assertAlmostEqual(float(r['relative_gram_condition'][0]),expected,delta=3e-6)
                amp=np.empty(k);amp[list(order)]=r['amplitudes'][0]
                if reference is None:reference=amp
                np.testing.assert_allclose(amp,reference,atol=4e-6)
                self.assertTrue(r['objective_valid'][0])
        # Zero coefficients do not remove distinct columns from the design.
        basis=unit(np.array([[[1,-1,1,-1,1,-1,1,-1]],[[1,1,-1,-1,1,1,-1,-1]],[[1,1,1,1,-1,-1,-1,-1]]],float))
        a=basis[0];b=(basis[0]+basis[1])/np.sqrt(2);c=(basis[1]+basis[2])/np.sqrt(2)
        y=(basis[0]+basis[1]+2*basis[2])[None]
        two=self.s.fit_overlap_projections(y,np.stack([a,b])[:,None])
        three=self.s.fit_overlap_projections(y,np.stack([a,b,c])[:,None])
        self.assertLess(float(two['amplitudes'][0,0]),1e-6)
        self.assertGreater(float(three['amplitudes'][0,0]),.1)
        self.assertLess(float(three['amplitudes'][0,1]),1e-6)
        self.assertLess(float(three['sse'][0]),float(two['sse'][0]))
        self.assertGreater(float(three['relative_gram_condition'][0]),.00251)

    def test_full_gram_withholds_fractions_without_pair_duplicates(self):
        basis=unit(np.array([[[1,-1,1,-1,1,-1,1,-1]],[[1,1,-1,-1,1,1,-1,-1]],[[1,1,1,1,-1,-1,-1,-1]]],float))
        third=np.sqrt(1-.05**2)*(basis[0]+basis[1])/np.sqrt(2)+.05*basis[2]
        templates=np.stack([basis[0],basis[1],third])[:,None]
        target=templates.sum(axis=0)
        r=self.s.fit_overlap_projections(target,templates)
        self.assertTrue(r['objective_valid'][0])
        self.assertTrue(r['conditioned'][0])
        self.assertLess(abs(float(r['component_correlation'][0])),.995)
        self.assertLess(float(r['relative_gram_condition'][0]),(1-.995)/(1+.995))
        self.assertTrue(np.isnan(r['prepared_fractions']).all())
        self.assertLess(float(r['sse'][0]),1e-4)

    def test_shapes_and_nonfinite(self):
        x = np.ones((1, 4, 5), np.float32)
        for a in [np.ones((0, 1, 4, 5)), np.ones((5, 1, 4, 5)), np.ones((2, 4, 5))]:
            with self.assertRaises(ValueError):
                self.s.fit_overlap_projections(x, a)
        with self.assertRaisesRegex(ValueError, 'finite'):
            self.s.fit_overlap_projections(x*np.nan, x[None])
        oversized = np.zeros((1,363,363), np.float32)
        with self.assertRaisesRegex(Error, 'PatternTooLargeUseBinning'):
            self.s.fit_overlap_projections(oversized, oversized[None])


class NativeDiscovery(unittest.TestCase):
    def setUp(self):
        self.s = Solver()
        self.addCleanup(self.s.close)
        self.s.configure(static_background='none', dynamic_background='none',
                         mask=False, clahe=False, master_smooth=0,
                         bandwidth=64, dict_grid=40, refine_iters=16, batch=6,
                         binning=1, full_resolution='none')
        for filename in ['nickel/Ni_master_20kV.npz', 'gan/GaN_master_20kV.npz']:
            self.s.add_phase(ROOT/'testdata'/filename)
        self.s.set_patterns(np.zeros((1, 6, 48, 48), np.float32))
        self.s.set_geometry(pc=(.44, .24, .53), sample_tilt_deg=70, detector_tilt_deg=0,
                            px_um=560, step_x_um=5, step_y_um=5)
        rng = np.random.default_rng(315)
        q = rng.normal(size=(3, 4))
        self.q = (q/np.linalg.norm(q, axis=1, keepdims=True)).astype(np.float32)
        # Simulate at each original cell's actual source position, with native poses.
        s0 = unit(self.s.simulate(np.tile(self.q[0], (6, 1)), cells=np.arange(6), phase=0))
        s1 = unit(self.s.simulate(np.tile(self.q[1], (6, 1)), cells=np.arange(6), phase=0))
        s2 = unit(self.s.simulate(np.tile(self.q[2], (6, 1)), cells=np.arange(6), phase=1))
        self.patterns = np.stack([s0[0], .8*s0[1]+.6*s1[1], .8*s0[2]+.6*s2[2],
                                  np.zeros_like(s0[3]), s0[4]+.01*rng.normal(size=(48,48)),
                                  rng.normal(size=(48,48))]).astype(np.float32)
        self.s.set_patterns(self.patterns[None])

    def test_dictionary_and_spherical_original_conditioned_components(self):
        for method in ['spherical', 'dictionary']:
            with self.subTest(method=method):
                self.s.configure(method=method)
                before = self.s.index_cells(np.arange(6))
                stored = self.s.index()
                geometry = self.s.geometry
                settings = self.s.settings
                raw = self.s.pattern(1, processed=False).copy()
                r = self.s.overlap_cells(np.arange(6))
                self.assertEqual(r['model'], 'fixed_prepared_projection_nnls')
                self.assertEqual(r['assessment'], 'unassessed_candidates')
                self.assertFalse(r['blur_gain_parity'])
                self.assertNotIn('accepted', r)
                self.assertNotIn('z', r)
                self.assertEqual(r['phase'].shape, (6, 2))
                self.assertTrue(np.all(r['amplitudes'] >= 0))
                evaluated=~r['flat']
                self.assertTrue(np.all(r['retained_order'][evaluated]>0))
                self.assertTrue(np.isfinite(r['best_single_sse'][evaluated]).all())
                self.assertTrue(np.isfinite(r['improvement_over_best_single'][evaluated]).all())
                self.assertTrue(np.all(r['sse'][evaluated] <= r['best_single_sse'][evaluated] + 2e-5))
                self.assertTrue(np.all(r['improvement_over_best_single'][evaluated] >= 0))
                self.assertTrue(np.isnan(r['best_single_sse'][r['flat']]).all())
                self.assertTrue(np.isnan(r['mixture_ncc'][r['flat']]).all())
                self.assertTrue(r['flat'][3])
                self.assertTrue(np.all(r['phase'][3] == -1))
                self.assertGreater(float(r['mixture_ncc'][0]), .99)
                self.assertGreater(float(r['mixture_ncc'][4]), .85)
                # Known same-phase and mixed-phase mixtures: assess identities using
                # the native symmetry operation, not custom quaternion mathematics.
                for b, phases in [(1, [0, 0]), (2, [0, 1])]:
                    self.assertEqual(sorted(r['phase'][b].tolist()), phases)
                    truth = [self.q[0], self.q[1] if b == 1 else self.q[2]]
                    for j, phase in enumerate(phases):
                        candidates = r['quaternions'][b][r['phase'][b] == phase]
                        errors = [float(np.degrees(self.s.disorientation(c[None], truth[j][None], phase=phase))[0]) for c in candidates]
                        if b == 2 and phase == 1 and method == 'dictionary':
                            # Named retained case: seed315/cell2 has a polar competing
                            # pose (~180 deg at grid40), so test phase + reconstruction
                            # and leave physical orientation assessment explicit.
                            print(f' seed315/cell2 polar dictionary native error: {min(errors):.6f} deg', flush=True)
                            self.assertGreater(float(r['mixture_ncc'][b]), .95)
                            self.assertFalse(r['physical_fractions'])
                        else:
                            self.assertLess(min(errors), 3, f'{method}: cell {b}, phase {phase}, errors {errors}')
                    self.assertGreater(float(r['improvement_over_best_single'][b]), .05)
                np.testing.assert_array_equal(self.s.pattern(1, processed=False), raw)
                self.assertEqual(self.s.geometry, geometry)
                self.assertEqual(self.s.settings, settings)
                after = self.s.index_cells(np.arange(6))
                for key in before:
                    np.testing.assert_array_equal(after[key], before[key], err_msg=f'ordinary {method} {key}')
                for key, value in stored.items():
                    np.testing.assert_array_equal(self.s.results()[key], value, err_msg=f'stored {method} {key}')

    def test_well_conditioned_mixed_phase_known_pose(self):
        # Keep the seed315/cell2 polar case above. This separate, fixed seed714/0
        # fixture exercises strict pose recovery with both native proposal methods.
        rng = np.random.default_rng(714)
        q = rng.normal(size=4)
        q = (q / np.linalg.norm(q)).astype(np.float32)
        a = unit(self.s.simulate(self.q[0:1], cells=[2], phase=0))[0]
        b = unit(self.s.simulate(q[None], cells=[2], phase=1))[0]
        patterns = self.patterns.copy()
        patterns[2] = .8*a + .6*b
        self.s.set_patterns(patterns[None])
        for method in ['spherical', 'dictionary']:
            self.s.configure(method=method)
            r = self.s.overlap_cells([2])
            self.assertEqual(sorted(r['phase'][0].tolist()), [0,1])
            for phase, truth in [(0, self.q[0]), (1, q)]:
                candidate = r['quaternions'][0][r['phase'][0] == phase]
                error = np.degrees(self.s.disorientation(candidate, truth[None], phase=phase))[0]
                self.assertLess(float(error), 3, f'{method}: known pose phase{phase}: {error}')
            self.assertGreater(float(r['mixture_ncc'][0]), .98)
            self.assertGreater(float(r['improvement_over_best_single'][0]), .05)

    def test_historical_skipped_phase_does_not_mark_flat_stop_unresolved(self):
        # Native event summary: a flat/invalid phase was skipped, then a later
        # phase fit exactly at order one. GPU probe exercises skip/continue;
        # this API fixture verifies the final diagnostic interpretation.
        def exact_fit_after_skip(_engine, _cells, n, k, _primary, components, _length, states, _state_length, progress, _progress_length):
            self.assertEqual((n,k), (1,2))
            components[0].phase = 1
            components[0].q[:] = [1,0,0,0]
            components[0].amplitude = 1
            components[1].phase = -1
            states[0].sse = 0
            states[0].initial_sse = 1
            states[0].flags = 16
            states[0].amplitude[:] = [1,0,0,0]
            states[0].single_sse[:] = [0,1,1,1]
            states[0].condition = 1
            states[0].best_single_sse = 0
            progress[0].retained_order=1;progress[0].active_count=1;progress[0].proposed_count=1;progress[0].stop_reason=2
            return 0
        for domain, entry in [('source_dc', 'ze_overlap_cells_v2'), ('contrast', 'ze_overlap_cells_v4')]:
            with self.subTest(gain_domain=domain):
                def native_fit(*args):
                    if domain == 'source_dc':
                        return exact_fit_after_skip(*args)
                    engine,cells,n,k,primary,_optimize,components,length,states,state_length,progress,progress_length,theta,_theta_length,norms,_norms_length = args
                    result = exact_fit_after_skip(engine,cells,n,k,primary,components,length,states,state_length,progress,progress_length)
                    for b in range(n):
                        for j,value in enumerate([0,0,1,1,1,0,0]):theta[b*7+j]=value
                    for j in range(n*k):norms[j]=1 if components[j].phase >= 0 else float('nan')
                    return result
                with patch.object(self.s._lib, entry, side_effect=native_fit) as mocked:
                    r = self.s.overlap_cells([0], gain_domain=domain)
                mocked.assert_called_once()
                self.assertTrue(r['unresolved_candidate_skipped'][0])
                self.assertTrue(r['residual_flat'][0])
                self.assertFalse(r['addition_unresolved'][0])
                self.assertEqual(r['candidate_count'][0], 1)
                self.assertNotEqual(r['solve_status'][0], 'unresolved_addition')
                self.assertTrue(r['objective_valid'][0])

    def test_zero_order_unavailable_status_names_reason(self):
        for reason in (3,5):
            def no_design(_engine,_cells,n,k,_primary,components,_length,states,_state_length,progress,_progress_length):
                for c in components:c.phase=-1
                states[0].sse=1;states[0].initial_sse=1;states[0].condition=1;states[0].best_single_sse=1
                progress[0].stop_reason=reason;progress[0].proposed_count=1
                return 0
            for domain,entry in [('source_dc','ze_overlap_cells_v2'),('contrast','ze_overlap_cells_v4')]:
                with self.subTest(reason=reason,gain_domain=domain):
                    def native_fit(*args):
                        if domain == 'source_dc':return no_design(*args)
                        engine,cells,n,k,primary,_optimize,components,length,states,state_length,progress,progress_length,theta,_theta_length,norms,_norms_length=args
                        result=no_design(engine,cells,n,k,primary,components,length,states,state_length,progress,progress_length)
                        for b in range(n):
                            for j,value in enumerate([0,0,1,1,1,0,0]):theta[b*7+j]=value
                        for j in range(n*k):norms[j]=float('nan')
                        return result
                    with patch.object(self.s._lib,entry,side_effect=native_fit) as mocked:
                        result=self.s.overlap_cells([0],gain_domain=domain)
                    mocked.assert_called_once()
                    self.assertFalse(result['objective_valid'][0])
                    self.assertTrue(np.isnan(result['sse'][0]))
                    self.assertEqual(result['solve_status'][0],result['stop_reason_name'][0])
                    self.assertNotEqual(result['solve_status'][0],'solved')

    def test_masked_nonzero_constant_has_no_components(self):
        self.s.configure(mask=True)
        self.s.set_patterns(np.ones((1,1,48,48), np.float32)*17)
        r = self.s.overlap_cells([0])
        self.assertTrue(r['flat'][0])
        np.testing.assert_array_equal(r['phase'], -1)
        np.testing.assert_array_equal(r['amplitudes'], 0)
        self.assertEqual(float(r['sse'][0]), 0)

    def test_stored_primary_context_missing_and_inspection(self):
        self.s.configure(roi_x=0,roi_y=0,roi_w=1,roi_h=1,accept_z=1e9)
        original=self.s.index()
        self.assertEqual(int(original['sweep'][0,0]),-1)
        before={key:value.copy() for key,value in self.s.results().items()}
        result=self.s.overlap_cells([0,1],max_components=4,primary='stored')
        np.testing.assert_array_equal(result['quaternions'][0,0],original['quaternions'][0,0])
        self.assertEqual(result['stop_reason_name'][1],'missing_primary')
        self.assertFalse(result['objective_valid'][1])
        self.assertTrue(np.isnan(result['sse'][1]))
        for direction in ('x','y','z'):
            colors=self.s.orientation_colors(original['quaternions'],original['phase'],direction)
            np.testing.assert_array_equal(colors,self.s.map('ipf_'+direction))
        image=self.s.overlap_inspect(0,result,0,0)
        np.testing.assert_allclose(image['measured']-image['composite'],image['residual'],atol=1e-7)
        np.testing.assert_allclose(image['conditional']-image['selected_contribution'],image['residual'],atol=1e-7)
        self.assertAlmostEqual(float(np.sum(image['residual']**2)),float(result['sse'][0]),delta=4e-6)
        for key,value in before.items():np.testing.assert_array_equal(self.s.results()[key],value)
        from zebsdsolve import _native
        counters=np.empty(4,np.uint32)
        self.s._lib.ze_gpu_counts(self.s._e,counters.ctypes.data_as(_native.u32p))
        self.s.configure(refine_iters=17)
        with self.assertRaisesRegex(Error,'StoredPrimaryContextStaleRerunIndex'):
            self.s.overlap_cells([0],primary='stored')
        self.s._lib.ze_gpu_counts(self.s._e,counters.ctypes.data_as(_native.u32p))
        np.testing.assert_array_equal(counters,0)
        self.s.configure(refine_iters=16)
        self.s.set_geometry(pc=(.43,.24,.53))
        with self.assertRaisesRegex(Error,'StoredPrimaryContextStaleRerunIndex'):
            self.s.overlap_cells([0],primary='stored')
        generation=self.s.phase_generation
        self.s.add_phase(ROOT/'testdata/nickel/Ni_master_20kV.npz')
        self.assertGreater(self.s.phase_generation,generation)
        self.s._lib.ze_gpu_counts(self.s._e,counters.ctypes.data_as(_native.u32p))
        missing=self.s.overlap_cells([0,1],primary='stored')
        self.assertTrue(np.all(missing['stop_reason']==6))
        self.s._lib.ze_gpu_counts(self.s._e,counters.ctypes.data_as(_native.u32p))
        np.testing.assert_array_equal(counters,0)

    def test_unavailable_diagnostics_match_alone_and_mixed_batch(self):
        scalar_scores=('sse','initial_sse','residual_norm','mixture_ncc','best_single_sse','improvement_over_best_single','component_correlation','relative_gram_condition','previous_sse','incremental_improvement')
        def compare(reason):
            alone=self.s.overlap_cells([1],max_components=4,primary='stored')
            mixed=self.s.overlap_cells([0,1],max_components=4,primary='stored')
            self.assertTrue(mixed['objective_valid'][0])
            for result,index in ((alone,0),(mixed,1)):
                self.assertEqual(result['stop_reason'][index],reason)
                for key in scalar_scores:self.assertTrue(np.isnan(result[key][index]),key)
                for key in ('prepared_fractions','proposal_ncc','proposal_nmi','single_sse'):
                    self.assertTrue(np.isnan(result[key][index]).all(),key)
                for key in ('objective_valid','conditioning_assessed','design_rank_reliable','residual_flat','attribution_unresolved','conditioned','unresolved_candidate_skipped'):
                    self.assertFalse(result[key][index],key)
                for key in ('candidate_count','fitted_count','retained_order','proposed_count','active_count','objective_order'):
                    self.assertEqual(result[key][index],0,key)
                np.testing.assert_array_equal(result['phase'][index],-1)
                np.testing.assert_array_equal(result['discovery_id'][index],-1)
            self.assertEqual(alone['solve_status'][0],mixed['solve_status'][1])
            self.assertEqual(alone['seed_source_roi'],mixed['seed_source_roi'])
        self.s.configure(roi_x=0,roi_y=0,roi_w=1,roi_h=1)
        self.s.index();compare(6)
        self.s.configure(roi_w=6);self.s.index()
        self.s.configure(mask=True,roi_w=1);self.s.index();compare(10)

    def test_stored_multi_phase_winners_are_not_researched(self):
        patterns=self.patterns.copy()
        patterns[2]=self.s.simulate(self.q[2:3],cells=[2],phase=1)[0]
        self.s.set_patterns(patterns[None])
        original=self.s.index()
        self.assertEqual(original['phase'][0,0],0)
        self.assertEqual(original['phase'][0,2],1)
        result=self.s.overlap_cells([0,2],max_components=1,primary='stored')
        np.testing.assert_array_equal(result['phase'][:,0],original['phase'].reshape(-1)[[0,2]])
        np.testing.assert_array_equal(result['quaternions'][:,0],original['quaternions'].reshape(-1,4)[[0,2]])
        self.assertFalse(result['unresolved_candidate_skipped'].any())
        np.testing.assert_array_equal(result['retained_order'],1)
        np.testing.assert_array_equal(result['proposed_count'],1)

    def test_roi_reuse_and_changed_conditioning_proof(self):
        original=self.s.index()
        before={k:v.copy() for k,v in original.items()}
        self.s.configure(roi_x=0,roi_y=0,roi_w=1,roi_h=1)
        reused=self.s.overlap_cells([0],max_components=1,primary='stored')
        np.testing.assert_array_equal(reused['quaternions'][0,0],original['quaternions'][0,0])
        self.assertEqual(reused['seed_source_roi'],[0,0,0,0])
        self.s.configure(mask=True)
        with self.assertRaisesRegex(Error,'StoredPrimaryContextStaleRerunIndex'):
            self.s.overlap_cells([0],primary='stored')
        self.s.index_begin()
        pending=self.s.overlap_cells([0,1],max_components=1,primary='stored')
        np.testing.assert_array_equal(pending['stop_reason'],[6,10])
        self.s.index_step()
        fresh=self.s.overlap_cells([0,1],max_components=1,primary='stored')
        self.assertTrue(fresh['objective_valid'][0]);self.assertFalse(fresh['objective_valid'][1])
        self.assertEqual(fresh['stop_reason_name'][1],'stale_primary')
        self.assertEqual(fresh['seed_source_roi'],[0,0,1,1])
        after=self.s.results()
        for key,value in before.items():
            np.testing.assert_array_equal(after[key].reshape((6,)+after[key].shape[2:])[1:],value.reshape((6,)+value.shape[2:])[1:])
        with self.assertRaisesRegex(Error,'MixedIndexingContexts'):
            self.s.export_snapshot()

    def test_seed_proof_partial_cancel_later_sweep_and_checkpoint(self):
        self.s.configure(roi_x=0,roi_y=0,roi_w=1,roi_h=1,sweeps=2,sweep_bandwidth=64,accept_z=1e9,batch=1)
        self.s.index_begin()
        self.assertEqual(self.s.index_step(),1)
        first=self.s.results()['quaternions'].copy()
        self.assertEqual(self.s.index_step(),0)
        np.testing.assert_array_equal(first,self.s.results()['quaternions'])
        seed=self.s.overlap_cells([0],max_components=1,primary='stored')
        np.testing.assert_array_equal(seed['quaternions'][0,0],first[0,0])
        self.assertEqual(self.s.results()['sweep'][0,0],-1)
        capture=self.s._lib.ze_result_checkpoint;capture.argtypes=[C.c_void_p,C.c_void_p,C.c_size_t];capture.restype=C.c_int64
        restore=self.s._lib.ze_restore_result_checkpoint;restore.argtypes=[C.c_void_p,C.c_void_p,C.c_size_t];restore.restype=C.c_int32
        size=capture(self.s._e,None,0);self.assertGreater(size,112,self.s._error())
        data=C.create_string_buffer(size);self.assertEqual(capture(self.s._e,data,size),size)
        # Clear in-progress proof, then a verified homogeneous restore rebinds it.
        self.s.index_begin()
        self.assertEqual(self.s.overlap_cells([0],primary='stored')['stop_reason'][0],6)
        self.assertEqual(restore(self.s._e,data,size),0,self.s._error())
        np.testing.assert_array_equal(self.s.overlap_cells([0],max_components=1,primary='stored')['quaternions'][0,0],first[0,0])
        bad=bytearray(data.raw);bad[-1]^=1;bad_buffer=C.create_string_buffer(bytes(bad))
        self.assertLess(restore(self.s._e,bad_buffer,size),0);self.s._error()
        np.testing.assert_array_equal(self.s.overlap_cells([0],max_components=1,primary='stored')['quaternions'][0,0],first[0,0])
        # New conditioning invalidates all old proofs; only committed cells regain
        # proof if normal indexing is cancelled after its first group of batches.
        patterns = np.tile(self.patterns[None], (1,4,1,1))
        self.s.set_patterns(patterns)
        self.s.configure(mask=True,roi_w=24,roi_h=1,sweeps=1)
        self.s.index_begin()
        self.assertGreater(self.s.index_step(), 0)
        visited = int(np.count_nonzero(self.s.results()['phase'] >= 0))
        self.assertGreater(visited, 1)
        self.s.cancel()
        self.s.index_step()
        partial = self.s.overlap_cells([0,visited],max_components=1,primary='stored')
        self.assertTrue(partial['objective_valid'][0])
        self.assertEqual(partial['stop_reason'][1],6)

    def test_checkpoint_requires_the_opened_pattern_kind(self):
        from tempfile import TemporaryDirectory
        import h5py

        with TemporaryDirectory() as temp, Solver() as solver:
            path = Path(temp) / "scan.h5oina"
            with h5py.File(path, "w") as file:
                group = file.create_group("1/EBSD")
                header, data = group.create_group("Header"), group.create_group("Data")
                header.create_dataset("Y Cells", data=[2])
                header.create_dataset("X Cells", data=[3])
                data.create_dataset("Processed Patterns", data=np.arange(480, dtype=np.uint8).reshape(6,8,10))
                data.create_dataset("Unprocessed Patterns", data=np.arange(480, dtype=np.int16).reshape(6,8,10) - 240)
            solver.configure(bandwidth=16, batch=2)
            solver.add_phase(ROOT / "testdata/nickel/Ni_master_20kV.npz")
            lib = solver._lib
            open_configured = lib.ze_open_scan_configured
            open_configured.argtypes = [C.c_void_p, C.c_char_p, C.c_size_t, C.c_char_p,
                                        C.c_size_t, C.c_char_p, C.c_size_t]
            open_configured.restype = C.c_int32
            capture = lib.ze_result_checkpoint
            capture.argtypes = [C.c_void_p, C.c_void_p, C.c_size_t]
            capture.restype = C.c_int64
            restore = lib.ze_restore_result_checkpoint
            restore.argtypes = [C.c_void_p, C.c_void_p, C.c_size_t]
            restore.restype = C.c_int32
            filename = str(path).encode()

            def reopen(kind):
                hints = b'{"oxford_patterns":"' + kind + b'"}'
                self.assertEqual(open_configured(solver._e, filename, len(filename), b"{}", 2, hints, len(hints)), 0)
                self.assertEqual(solver.scan["unprocessed"], kind == b"unprocessed")

            reopen(b"processed")
            solver.index()
            size = capture(solver._e, None, 0)
            self.assertGreater(size, 112)
            data = C.create_string_buffer(size)
            self.assertEqual(capture(solver._e, data, size), size)
            reopen(b"processed")
            self.assertEqual(restore(solver._e, data, size), 0)
            reopen(b"unprocessed")
            self.assertLess(restore(solver._e, data, size), 0)
            self.assertIn("CheckpointContextMismatch", solver._error())
            self.assertTrue(np.all(solver.results()["phase"] == -1))

    def test_spherical_three_and_four_known_native_poses(self):
        self.s.clear_phases();self.s.add_phase(ROOT/'testdata/nickel/Ni_master_20kV.npz')
        self.s.configure(method='spherical')
        self.s.set_geometry(step_x_um=0,step_y_um=0)
        q=np.random.default_rng(714).normal(size=(4,4));q=(q/np.linalg.norm(q,axis=1,keepdims=True)).astype(np.float32)
        base=unit(self.s.simulate(q,cells=[0]*4))
        patterns=np.zeros((6,48,48),np.float32)
        for k in range(1,5):patterns[k]=np.einsum('i,ihw->hw',np.array([.7,.5,.3,.2][:k]),base[:k])
        self.s.set_patterns(patterns[None])
        result=self.s.overlap_cells(np.arange(6),max_components=4)
        np.testing.assert_array_equal(result['phase'][0],-1)
        self.assertEqual(result['retained_order'][0],0)
        for k in (3,4):
            self.assertEqual(result['retained_order'][k],4)
            for truth in q[:k]:
                error=np.degrees(self.s.disorientation(result['quaternions'][k],np.tile(truth,(4,1))))
                self.assertLess(float(error.min()),2)
            self.assertLess(float(result['sse'][k]),.02)
        for b in range(6):
            order=int(result['retained_order'][b])
            np.testing.assert_array_equal(result['phase'][b,order:],-1)
            np.testing.assert_array_equal(result['discovery_id'][b,:order],np.arange(order))
        absent=self.s.overlap_inspect(0,result,0,3)
        np.testing.assert_array_equal(absent['selected_contribution'],0)
        np.testing.assert_array_equal(absent['conditional'],absent['residual'])

    def test_resolution_guard_invalid_cells_and_empty(self):
        self.s.configure(binning=2, full_resolution='refine')
        with self.assertRaisesRegex(Error, 'SameResolution'):
            self.s.overlap_cells([0])
        self.s.configure(binning=1)
        for cells in [[-1], [1.5]]:
            with self.assertRaises(ValueError):
                self.s.overlap_cells(cells)
        with self.assertRaisesRegex(Error, 'BadCell'):
            self.s.overlap_cells([6])
        for k in [0, 5]:
            with self.assertRaises(ValueError):
                self.s.overlap_cells([0], max_components=k)
        self.assertEqual(self.s.overlap_cells([])['quaternions'].shape, (0,2,4))
        # Full-size and reduced fits both inspect at the resolution used for search.
        for factor in [1, 2]:
            self.s.configure(binning=factor, full_resolution='none')
            result = self.s.overlap_cells([0], max_components=1)
            self.assertEqual(result['phase'].shape, (1,1))
            self.assertEqual(self.s.overlap_inspect(0, result)['residual'].shape, self.s.pattern_shape)


if __name__ == '__main__':
    unittest.main()
