"""V4 image-domain oracles. Native projection geometry and frames stay unchanged."""
from pathlib import Path
import sys
import unittest
import numpy as np
from scipy.ndimage import convolve1d
from scipy.optimize import nnls
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'python'),str(ROOT/'tests')]
from zebsdsolve import Solver
from test_overlap_blur_gain import transform,unit


def mask_pixels(shape,mask=None):
    if mask is None:return np.ones(shape,bool)
    y,x=np.indices(shape);return (y-mask[0])**2+(x-mask[1])**2<=mask[2]**2


def contrast_transform(images,theta,mask=None):
    a=np.asarray(images,np.float64)
    sigma=float(theta[0])
    if sigma>0:
        radius=int(np.ceil(3*sigma));x=np.arange(-radius,radius+1,dtype=float)
        kernel=np.exp(-x*x/(2*sigma*sigma))
        # Never mask before blur: exterior pixels contribute across the boundary.
        for axis in (-1,-2):
            a=convolve1d(a,kernel,axis=axis,mode='constant')/convolve1d(np.ones_like(a),kernel,axis=axis,mode='constant')
    keep=mask_pixels(a.shape[-2:],mask)
    a=a-a[...,keep].mean(axis=-1)[...,None,None]
    return transform(a,[0,*np.asarray(theta)[1:]])


def centered_and_norm(images,mask=None):
    a=np.asarray(images,np.float64);keep=mask_pixels(a.shape[-2:],mask)
    x=np.where(keep,a-a[...,keep].mean(axis=-1)[...,None,None],0)
    d=np.linalg.norm(x.reshape(*x.shape[:-2],-1),axis=-1)
    return x,d


class ContrastGainFit(unittest.TestCase):
    def setUp(self):
        self.s=Solver();self.addCleanup(self.s.close)

    def check_oracle(self,r,y,a,mask=None):
        k=len(a);v=contrast_transform(a,r['blur_gain_theta'][0],mask)
        x,d=centered_and_norm(v,mask);design=(x/d[:,None,None]).reshape(k,-1).T
        target=unit(y,mask).ravel();coef,_=nnls(design,target);res=target-design@coef
        np.testing.assert_allclose(r['component_norms'][0],d,rtol=2e-5,atol=2e-6)
        np.testing.assert_allclose(r['amplitudes'][0],coef,atol=3e-6)
        np.testing.assert_allclose(r['residual'][0].ravel(),res,atol=3e-6)
        self.assertAlmostEqual(float(r['sse'][0]),float(res@res),delta=3e-6)
        expected=coef/d;expected/=expected.sum()
        np.testing.assert_allclose(r['contrast_basis_weights'][0],expected,atol=3e-6)

    def test_unstandardized_known_weights_k1_through4_nonflat_gain_mask(self):
        rng=np.random.default_rng(810)
        for k in (1,2,3,4):
            for mask in (None,(10,14,9)):
                with self.subTest(k=k,mask=mask):
                    a=rng.normal(size=(k,23,31))*np.arange(1,k+1)[:,None,None]+np.arange(k)[:,None,None]*7
                    truth=[0,.25,1,1,1,0,0]
                    weights=np.linspace(.8,.2,k);weights/=weights.sum()
                    # Mix unnormalized contrast fields, retaining their unequal norms.
                    y=np.einsum('k,khw->hw',weights,contrast_transform(a,truth,mask))
                    r=self.s.fit_overlap_projections(y[None],a[:,None],blur_gain=True,gain_domain='contrast',mask=mask)
                    self.assertLess(float(r['sse'][0]),1e-9)
                    np.testing.assert_allclose(r['contrast_basis_weights'][0],weights,atol=2e-6)
                    self.check_oracle(r,y,a,mask)
                    self.assertEqual(r['fit_model'],'contrast_gain_v1')
                    self.assertEqual(r['weight_basis'],'supplied_projection_contrast')
                    self.assertFalse(r['physical_fractions'])
                    self.assertEqual(r['blur_gain_optimizer']['version'],2)

    def test_offset_invariance_and_reference_scale_conversion(self):
        rng=np.random.default_rng(811)
        a=rng.normal(size=(2,21,27))*np.array([1,3])[:,None,None]
        weights=np.array([.7,.3]);truth=[0,.25,1,1,1,0,0]
        y=np.einsum('k,khw->hw',weights,contrast_transform(a,truth))
        fit=lambda aa:self.s.fit_overlap_projections(y[None],aa[:,None],blur_gain=True,gain_domain='contrast')
        base=fit(a);offset=fit(a+np.array([12,-9])[:,None,None])
        np.testing.assert_allclose(offset['amplitudes'],base['amplitudes'],atol=2e-6)
        np.testing.assert_allclose(offset['component_norms'],base['component_norms'],rtol=2e-6)
        np.testing.assert_allclose(offset['contrast_basis_weights'],base['contrast_basis_weights'],atol=2e-6)
        np.testing.assert_allclose(offset['residual'],base['residual'],atol=2e-6)
        scaled=fit(a*np.array([4,1])[:,None,None])
        np.testing.assert_allclose(scaled['amplitudes'],base['amplitudes'],atol=2e-6)
        np.testing.assert_allclose(scaled['component_norms'],base['component_norms']*[[4,1]],rtol=2e-6)
        expected=weights/[4,1];expected/=expected.sum()
        np.testing.assert_allclose(scaled['contrast_basis_weights'][0],expected,atol=2e-6)

    def test_representable_large_pedestals_do_not_invalidate_contrast(self):
        rng=np.random.default_rng(815)
        a=rng.integers(-3,4,size=(2,17,19)).astype(np.float32)
        for sigma in (0,1.5):
            y=np.einsum('k,khw->hw',[.7,.3],contrast_transform(a,[sigma,.25,1,1,1,0,0]))
            fit=lambda aa:self.s.fit_overlap_projections(y[None],aa[:,None],blur_gain=True,gain_domain='contrast')
            base=fit(a);shifted=fit(a+np.array([2**23,-2**23],np.float32)[:,None,None])
            self.assertTrue(shifted['objective_valid'][0])
            for field in ('component_norms','amplitudes','contrast_basis_weights','residual','sse','blur_gain_theta'):
                np.testing.assert_array_equal(shifted[field],base[field],err_msg=f'sigma={sigma}, {field}')

    def test_blur_full_image_before_mask_and_independent_norm_oracle(self):
        rng=np.random.default_rng(812);mask=(9,12,8)
        a=rng.normal(size=(2,19,25))*np.array([1,2])[:,None,None]
        a[:,~mask_pixels(a.shape[-2:],mask)]+=15
        y=np.einsum('k,khw->hw',[.65,.35],contrast_transform(a,[1.5,.25,1,1,1,0,0],mask))
        r=self.s.fit_overlap_projections(y[None],a[:,None],blur_gain=True,gain_domain='contrast',mask=mask)
        self.assertGreater(float(r['blur_gain_theta'][0,0]),.2)
        self.check_oracle(r,y,a,mask)
        premasked=a.copy();premasked[:,~mask_pixels(a.shape[-2:],mask)]=0
        wrong=unit(contrast_transform(premasked,r['blur_gain_theta'][0],mask),mask)
        correct=unit(contrast_transform(a,r['blur_gain_theta'][0],mask),mask)
        self.assertGreater(float(np.linalg.norm(wrong-correct)),.1)

    def test_v4_c_export_rejects_bad_shapes_flags_and_mask(self):
        from zebsdsolve import Error,_native as native
        values=np.ones(4,np.float32);amp=np.empty(1,np.float32);theta=np.array([0,0,1,1,1,0,0],np.float32)
        norms=np.empty(1,np.float32);mask=np.zeros(3,np.float32);states=(native.OverlapState*1)()
        ptr=lambda a:a.ctypes.data_as(native.f32p)
        def call(optimize=0,theta_len=1,mask_len=3,norms_len=1):
            rc=self.s._lib.ze_overlap_fit_v4(self.s._e,ptr(values),ptr(values),1,1,2,2,ptr(amp),ptr(values),states,ptr(theta),theta_len,ptr(mask),mask_len,optimize,ptr(norms),norms_len)
            self.s._check(rc,'fit_v4_shape_check')
        for args in ({'optimize':2},{'theta_len':0},{'mask_len':2},{'norms_len':0}):
            with self.subTest(args=args),self.assertRaises(Error):call(**args)
        mask[0]=np.nan
        with self.assertRaisesRegex(Error,'BadMask'):call()

    def test_identity_off_parity_and_flat_rank_fail_closed(self):
        rng=np.random.default_rng(813);a=rng.normal(size=(3,1,17,19)).astype(np.float32)
        y=np.einsum('knhw,k->nhw',a,[.5,.3,.2])
        new=self.s.fit_overlap_projections(y,a,gain_domain='contrast')
        self.check_oracle(new,y[0],a[:,0])
        np.testing.assert_allclose(new['contrast_basis_weights'][0],[.5,.3,.2],atol=2e-6)
        for target,templates in ((np.ones_like(y),a),(y,np.ones_like(a)),(y,a[[0,0]])):
            r=self.s.fit_overlap_projections(target,templates,blur_gain=True,gain_domain='contrast')
            self.assertTrue(np.isnan(r['contrast_basis_weights']).all())
        # Invalid one retained norm invalidates the whole row; absent slots stay NaN.
        result=dict(objective_valid=np.array([True]),design_rank_reliable=np.array([True]),flat=np.array([False]),conditioned=np.array([False]))
        m=Solver._contrast_metadata(result,np.array([[0,.5,0]],np.float32),np.array([[2,3,np.nan]],np.float32),np.array([[True,True,False]]),np.zeros((1,7)),"test")
        np.testing.assert_array_equal(m['contrast_basis_weights'][0,:2],[0,1]);self.assertTrue(np.isnan(m['contrast_basis_weights'][0,2]))
        m=Solver._contrast_metadata(result,np.array([[0,.5]],np.float32),np.array([[np.nan,3]],np.float32),np.ones((1,2),bool),np.zeros((1,7)),"test")
        self.assertTrue(np.isnan(m['contrast_basis_weights']).all())


class ContrastGainNative(unittest.TestCase):
    def setUp(self):
        from test_overlap import NativeDiscovery
        fixture=NativeDiscovery();fixture.setUp();self.fixture=fixture;self.s=fixture.s;self.addCleanup(self.s.close)

    def test_native_master_raw_known_same_and_mixed_phase_weights(self):
        # Native globally standardized baked-master projections, never per-pose standardized.
        for mixed in (False,True):
            aa=self.s.simulate(self.fixture.q[0:1],cells=[1],phase=0)[0]
            bb=self.s.simulate(self.fixture.q[1:2] if not mixed else self.fixture.q[2:3],cells=[1],phase=int(mixed))[0]
            a=np.stack([aa,bb]);weights=np.array([.3,.7]);truth=[0,.25,1,1,1,0,0]
            y=np.einsum('k,khw->hw',weights,contrast_transform(a,truth))
            r=self.s.fit_overlap_projections(y[None],a[:,None],blur_gain=True,gain_domain='contrast')
            self.assertLess(float(r['sse'][0]),1e-9)
            np.testing.assert_allclose(r['contrast_basis_weights'][0],weights,atol=1e-5)

    def test_discovery_inspection_norms_transactional_and_off_parity(self):
        self.s.index();before={key:v.copy() for key,v in self.s.results().items()}
        cells=[0,1,2,3]
        old=self.s.overlap_cells(cells,max_components=3,primary='stored',gain_domain='source_dc')
        off=self.s.overlap_cells(cells,max_components=3,primary='stored')
        # Stored primary/proposal context is unchanged; later residual proposals
        # may legitimately change after the corrected V4 NNLS fit.
        np.testing.assert_array_equal(off['quaternions'][:,0],old['quaternions'][:,0])
        np.testing.assert_array_equal(off['phase'][:,0],old['phase'][:,0])
        for blur_gain in (False,True):
            r=off if not blur_gain else self.s.overlap_cells(cells,max_components=3,primary='stored',blur_gain=True)
            for row,cell in enumerate(cells):
                keep=r['phase'][row]>=0
                if not np.any(keep):
                    self.assertTrue(np.isnan(r['contrast_basis_weights'][row]).all());continue
                raw=np.stack([self.s.simulate(q[None],cells=[cell],phase=int(ph))[0] for q,ph in zip(r['quaternions'][row,keep],r['phase'][row,keep])])
                x,d=centered_and_norm(contrast_transform(raw,r['blur_gain_theta'][row]))
                np.testing.assert_allclose(r['component_norms'][row,keep],d,rtol=3e-5,atol=2e-6)
                coef=r['amplitudes'][row,keep];des=(x/d[:,None,None]).reshape(len(d),-1).T
                residual=unit(self.s.pattern(cell,processed=True)).ravel()-des@coef
                im=self.s.overlap_inspect(cell,r,row,0)
                np.testing.assert_allclose(im['residual'].ravel(),residual,atol=3e-6)
                self.assertAlmostEqual(float(r['sse'][row]),float(residual@residual),delta=4e-6)
                self.assertTrue(np.isnan(r['contrast_basis_weights'][row,~keep]).all())
            self.assertTrue(np.any(r['retained_order']<3)) # staged extra slots cannot affect final retained norms
        for key,value in before.items():np.testing.assert_array_equal(self.s.results()[key],value)


if __name__=='__main__':unittest.main()
