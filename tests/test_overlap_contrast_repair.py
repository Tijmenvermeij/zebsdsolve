"""Reviewed V4 regressions: exact transfer consistency, robust mask anchors and KKT."""
from pathlib import Path
import sys
import unittest
import numpy as np
from scipy.optimize import nnls
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'python'),str(ROOT/'tests')]
from zebsdsolve import Solver
from test_overlap_blur_gain import unit
from test_overlap_contrast_gain import contrast_transform,centered_and_norm


class ContrastRepair(unittest.TestCase):
    def setUp(self):
        self.s=Solver();self.addCleanup(self.s.close)

    def fit(self,y,a,enabled=False,mask=None):
        return self.s.fit_overlap_projections(np.asarray(y)[None],np.asarray(a)[:,None],blur_gain=enabled,mask=mask,gain_domain='contrast')

    def test_source_and_target_pedestals_independently_off_and_on(self):
        grid=np.arange(23*29).reshape(23,29)
        a=np.stack([(grid%5==1).astype(np.float32),(grid%7==2).astype(np.float32)])
        y=2*a[0]+3*a[1]
        for enabled in (False,True):
            base=self.fit(y,a,enabled)
            variants=[self.fit(y,a+np.array([2**23,-2**23],np.float32)[:,None,None],enabled),
                      self.fit(y+np.float32(2**23),a,enabled),self.fit(y-np.float32(2**23),a,enabled)]
            for shifted in variants:
                for field in ('amplitudes','residual','sse','component_norms','contrast_basis_weights','blur_gain_theta'):
                    np.testing.assert_array_equal(shifted[field],base[field],err_msg=f'{enabled}: {field}')
            np.testing.assert_allclose(base['contrast_basis_weights'][0],[.4,.6],atol=2e-6)

    def test_full_image_contrast_can_enter_mask_after_blur(self):
        a=np.zeros((25,25),np.float32);a[12,19]=1;mask=(12,12,6)
        y=contrast_transform(a,[1.5,0,1,1,1,0,0],mask)
        zero=self.fit(y,a[None],False,mask)
        self.assertFalse(zero['objective_valid'][0]);self.assertTrue(np.isnan(zero['contrast_basis_weights']).all())
        blurred=self.fit(y,a[None],True,mask)
        self.assertTrue(blurred['objective_valid'][0]);self.assertLess(float(blurred['sse'][0]),1e-10)
        self.assertAlmostEqual(float(blurred['blur_gain_theta'][0,0]),1.5,places=6)
        _,norm=centered_and_norm(y,mask)
        np.testing.assert_allclose(blurred['component_norms'][0,0],norm,rtol=2e-6)

    def test_literal_distant_corner_outlier_and_masked_median_edge_cases(self):
        # Literal independent-review fixture: row-major i % 5, not column % 5.
        a=(np.arange(48*48).reshape(48,48)%5==1).astype(np.float32);mask=(24,24,5)
        base=self.fit(a,a[None],False,mask);corner=a.copy();corner[0,0]=2**28
        outlier=self.fit(a,corner[None],False,mask)
        self.assertTrue(base['objective_valid'][0]);self.assertTrue(outlier['objective_valid'][0])
        _,expected_norm=centered_and_norm(a,mask)
        np.testing.assert_allclose(base['component_norms'][0,0],expected_norm,rtol=2e-6)
        for field in ('amplitudes','residual','sse','component_norms','contrast_basis_weights'):
            np.testing.assert_array_equal(outlier[field],base[field])
        # A center outlier is legitimate image contrast; it must not become a
        # pathological scalar anchor or cause an otherwise exact identity fit to fail.
        center=a.copy();center[24,24]=2**16
        for image,active_mask in ((center,mask),(a,(0,0,3))):
            fitted=self.fit(image,image[None],False,active_mask)
            self.assertTrue(fitted['objective_valid'][0]);self.assertLess(float(fitted['sse'][0]),1e-10)
            _,norm=centered_and_norm(image,active_mask)
            np.testing.assert_allclose(fitted['component_norms'][0,0],norm,rtol=3e-6)
        for tiny in ((0,0,.1),(-10,-10,1)):
            fitted=self.fit(a,a[None],False,tiny)
            self.assertTrue(np.isnan(fitted['contrast_basis_weights']).all())

    def test_weak_faces_kkt_large_orthogonal_residual_k1_to4_permutations(self):
        # Balanced Walsh columns give exact zero dual products, avoiding an
        # undefined basis-weight comparison for a numerically tiny QR dot.
        pixels=128*128;index=np.arange(pixels)
        q=np.stack([np.where((index>>j)&1,-1.,1.)/128 for j in range(6)],axis=1)
        for k in (1,2,3,4):
            for weak in (.000782085,0,-.000782085):
                a=q[:,:k].T.reshape(k,128,128).astype(np.float32)
                coefficients=np.full(k,.00035);coefficients[0]=.24;coefficients[-1]=weak
                y=(q[:,:k]@coefficients+.97*q[:,5]).reshape(128,128).astype(np.float32)
                for order in (np.arange(k),np.arange(k)[::-1]):
                    with self.subTest(k=k,weak=weak,order=order.tolist()):
                        refs=a[order];result=self.fit(y,refs)
                        self.assertTrue(result['objective_valid'][0])
                        design=unit(refs).reshape(k,-1).T;target=unit(y).ravel()
                        oracle,_=nnls(design,target);actual=result['amplitudes'][0]
                        np.testing.assert_allclose(actual,oracle,atol=2e-6,rtol=2e-6)
                        residual=target-design@actual
                        np.testing.assert_allclose(result['residual'][0].ravel(),residual,atol=2e-7)
                        self.assertAlmostEqual(float(result['sse'][0]),float(residual@residual),delta=2e-6)
                        gram=design.T@design;rhs=design.T@target;gradient=rhs-gram@actual
                        tau=32*np.finfo(np.float32).eps*(np.sqrt(np.diag(gram)*(target@target))+abs(rhs)+abs(gram)@actual)
                        error=np.where(actual>0,abs(gradient),np.maximum(gradient,0))
                        self.assertTrue(np.all(error<=tau+1e-7),(gradient,tau,actual))
                        if oracle.sum()>0:
                            expected=oracle/result['component_norms'][0];expected/=expected.sum()
                            np.testing.assert_allclose(result['contrast_basis_weights'][0],expected,atol=5e-6)
                        else:self.assertTrue(np.isnan(result['contrast_basis_weights']).all())

    def test_near_collinear_rank_and_nonfinite_stay_fail_closed(self):
        rng=np.random.default_rng(819);a=rng.normal(size=(17,23)).astype(np.float32)
        b=a+1e-5*rng.normal(size=a.shape).astype(np.float32)
        result=self.fit(a,np.stack([a,b]));self.assertFalse(result['objective_valid'][0])
        self.assertTrue(np.isnan(result['contrast_basis_weights']).all())
        invalid=np.stack([a,b]);invalid[0,0,0]=np.nan
        with self.assertRaisesRegex(ValueError,'finite'):self.fit(a,invalid)


if __name__=='__main__':unittest.main()
