"""Scalar image-domain oracle; native crystallographic import/projection stays unchanged."""
from pathlib import Path
import sys
import unittest
import numpy as np
from scipy.ndimage import convolve1d, gaussian_filter
from scipy.optimize import nnls
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'python'))
from zebsdsolve import Solver, Error


def transform(image, theta, dc=0):
    sigma,logrho,power,ay,ax,dy,dx=np.asarray(theta,dtype=np.float64)
    image=np.asarray(image,np.float64)
    if sigma>0:
        radius=int(np.ceil(3*sigma));xx=np.arange(-radius,radius+1,dtype=float)
        kernel=np.exp(-xx*xx/(2*sigma*sigma))
        for axis in (-1,-2):
            image=convolve1d(image,kernel,axis=axis,mode='constant')/convolve1d(np.ones_like(image),kernel,axis=axis,mode='constant')
    h,w=image.shape[-2:];y,x=np.indices((h,w),dtype=float)
    radius=np.sqrt(((y-h/2-dy*h)/(ay*h/2))**2+((x-w/2-dx*w)/(ax*w/2))**2)
    shape=np.maximum(1-radius,0)**power;delta=np.expm1(logrho)
    return (1+delta*shape)*image+delta*shape*dc


def unit(image,mask=None):
    image=np.asarray(image,np.float64)
    keep=np.ones(image.shape[-2:],bool)
    if mask is not None:
        y,x=np.indices(keep.shape);keep=(y-mask[0])**2+(x-mask[1])**2<=mask[2]**2
    out=np.where(keep,image-image[...,keep].mean(axis=-1)[...,None,None],0)
    norm=np.sqrt(np.sum(out*out,axis=(-1,-2),keepdims=True))
    return np.divide(out,norm,out=np.zeros_like(out),where=norm>1e-14)


class BlurGainMetadata(unittest.TestCase):
    def test_loaded_optimizer_capability_and_unknown_version(self):
        from types import SimpleNamespace
        solver=Solver.__new__(Solver)
        theta=np.array([[0,0,1,1,1,0,0]],np.float32)
        for version in (None,1,2,3):
            solver._lib=SimpleNamespace() if version is None else SimpleNamespace(ze_overlap_optimizer_version=lambda:version)
            if version==3:
                with self.assertRaisesRegex(Error,"unsupported overlap optimizer metadata version 3"):
                    solver._blur_gain_metadata(True,theta)
                continue
            m=solver._blur_gain_metadata(True,theta)['blur_gain_optimizer']
            self.assertEqual(m['version'],version or 1)
            self.assertEqual(m['max_evaluations'],104 if version==2 else 92)
            self.assertEqual(m['singleton_max_evaluations'],96 if version==2 else 88)
            self.assertEqual('log_rho_polish_steps' in m,version==2)
            self.assertEqual('joint_singleton_seeds' in m,version==2)
        self.assertEqual(solver._blur_gain_metadata(False,None),{'blur_gain':False})


class BlurGainProjectionFit(unittest.TestCase):
    def setUp(self):
        self.s=Solver();self.addCleanup(self.s.close)

    def test_returned_theta_independent_float64_gaussian_mask_nnls_k1_to_4(self):
        rng=np.random.default_rng(805)
        for k in (1,2,3,4):
            for mask in (None,(9.5,14.5,9)):
                with self.subTest(k=k,mask=mask):
                    a=np.stack([gaussian_filter(rng.normal(size=(23,31)),.45)+2.3+j for j in range(k)]).astype(np.float32)
                    truth=[1.1,np.log(1.7),1.4,.9,1.2,.07,-.05]
                    templates=unit(transform(a,truth),mask)
                    y=np.einsum('k,khw->hw',np.linspace(.8,.2,k),templates)+.001*rng.normal(size=(23,31))
                    original=(y.copy(),a.copy())
                    r=self.s.fit_overlap_projections(y[None],a[:,None],blur_gain=True,mask=mask)
                    theta=r['blur_gain_theta'][0];design=unit(transform(a,theta),mask).reshape(k,-1).T;target=unit(y,mask).ravel()
                    coef,_=nnls(design,target);res=target-design@coef
                    np.testing.assert_allclose(r['amplitudes'][0],coef,atol=1.5e-5)
                    np.testing.assert_allclose(r['residual'][0].ravel(),res,atol=2e-6)
                    self.assertAlmostEqual(float(r['sse'][0]),float(res@res),delta=3e-6)
                    baseline,_=nnls(unit(a,mask).reshape(k,-1).T,target)
                    self.assertLessEqual(float(r['sse'][0]),float(np.sum((target-unit(a,mask).reshape(k,-1).T@baseline)**2))+3e-6)
                    self.assertGreater(np.exp(theta[1]),.24999);self.assertLess(np.exp(theta[1]),4.00001)
                    self.assertGreater(float(r['best_single_sse'][0])+1e-5,float(r['sse'][0]) if k>1 else -1)
                    np.testing.assert_array_equal(y,original[0]);np.testing.assert_array_equal(a,original[1])

    def test_small_gain_polish_k1_to_4_and_independent_nnls(self):
        rng=np.random.default_rng(806)
        for k in (1,2,3,4):
            with self.subTest(k=k):
                a=(rng.normal(size=(k,23,31))+2.5).astype(np.float32)
                truth=[0,.015,1,1,1,0,0]
                y=np.einsum('k,khw->hw',np.linspace(.8,.2,k),unit(transform(a,truth)))
                target=unit(y).ravel();identity_design=unit(a).reshape(k,-1).T
                identity_coef,_=nnls(identity_design,target)
                identity_residual=target-identity_design@identity_coef
                r=self.s.fit_overlap_projections(y[None],a[:,None],blur_gain=True)
                theta=r['blur_gain_theta'][0]
                self.assertTrue(r['objective_valid'][0])
                self.assertTrue(np.isfinite(r['sse'][0]))
                self.assertLess(abs(float(theta[1])-.015),.003)
                self.assertLess(float(r['sse'][0]),float(identity_residual@identity_residual)*.02)
                design=unit(transform(a,theta)).reshape(k,-1).T
                coef,_=nnls(design,target);res=target-design@coef
                np.testing.assert_allclose(r['amplitudes'][0],coef,atol=2e-6)
                np.testing.assert_allclose(r['residual'][0].ravel(),res,atol=2e-6)
                self.assertAlmostEqual(float(r['sse'][0]),float(res@res),delta=3e-6)
                self.assertTrue(np.all(r['sse'][0]<=r['single_sse'][0]+3e-6))
                if k==1:
                    np.testing.assert_array_equal(r['sse'],r['best_single_sse'])
                metadata=r['blur_gain_optimizer']
                self.assertEqual(metadata['version'],2)
                self.assertEqual(metadata['max_evaluations'],104)
                self.assertEqual(metadata['singleton_max_evaluations'],96)
                self.assertEqual(metadata['log_rho_polish_steps'],[.04375,.021875,.0109375,.00546875])

    def test_identity_and_flat_fail_closed_and_collective_rank(self):
        rng=np.random.default_rng(802)
        a=rng.normal(size=(4,1,15,19)).astype(np.float32)
        y=np.einsum('knhw,k->nhw',a,[.7,.5,.3,.2])
        old=self.s.fit_overlap_projections(y,a)
        new=self.s.fit_overlap_projections(y,a,blur_gain=True)
        np.testing.assert_array_equal(new['blur_gain_theta'],[[0,0,1,1,1,0,0]])
        np.testing.assert_allclose(new['sse'],old['sse'],atol=1e-10)
        for field in ('amplitudes','residual'):
            np.testing.assert_allclose(new[field],old[field],atol=2e-6)
        flat=self.s.fit_overlap_projections(np.full((1,15,19),17,np.float32),a,blur_gain=True)
        self.assertTrue(flat['flat'][0]);self.assertEqual(flat['sse'][0],0)
        invalid=self.s.fit_overlap_projections(y,np.ones_like(a),blur_gain=True)
        self.assertFalse(invalid['objective_valid'][0]);self.assertTrue(np.isnan(invalid['sse'][0]))
        # Duplicate and collective rank deficiency must not be concealed by a nuisance fit.
        for aa in (a[[0,0]],np.concatenate([a[:2],(a[:1]+a[1:2])],axis=0)):
            r=self.s.fit_overlap_projections(y,aa,blur_gain=True)
            self.assertFalse(r['objective_valid'][0]);self.assertTrue(np.isnan(r['amplitudes']).all())
        duplicate=self.s.fit_overlap_projections(-a[0],a[[0,0]],blur_gain=True)
        self.assertAlmostEqual(float(duplicate['component_correlation'][0]),1,delta=2e-6)
        self.assertEqual(float(duplicate['relative_gram_condition'][0]),0)
        np.testing.assert_allclose(duplicate['single_sse'][0],[1,1],atol=3e-6)
        self.assertAlmostEqual(float(duplicate['best_single_sse'][0]),1,delta=3e-6)

    def test_zero_composite_has_no_ncc(self):
        a=np.tile(np.array([-1,-1,1,1],np.float32),(4,1))[None]
        for enabled in (False,True):
            r=self.s.fit_overlap_projections(-a,a[None],blur_gain=enabled)
            np.testing.assert_array_equal(r['amplitudes'],0)
            self.assertTrue(r['objective_valid'][0])
            self.assertAlmostEqual(float(r['sse'][0]),1,delta=2e-6)
            self.assertTrue(np.isnan(r['mixture_ncc'][0]))

    def test_nonzero_dc_gain_sigma_is_used_and_full_resolution(self):
        rng=np.random.default_rng(803)
        a=(gaussian_filter(rng.normal(size=(19,29)),.5)+6).astype(np.float32)
        y=transform(a,[1.2,np.log(1.8),1,.8,1.2,.08,-.05])
        old=self.s.fit_overlap_projections(y[None],a[None,None])
        r=self.s.fit_overlap_projections(y[None],a[None,None],blur_gain=True)
        self.assertLess(float(r['sse'][0]),float(old['sse'][0])*.15)
        self.assertGreater(float(r['blur_gain_theta'][0,0]),.2)
        self.assertGreater(abs(float(r['blur_gain_theta'][0,1])),.05)
        # The old API also supports the actual 256 x 310 Re resolution.
        full=rng.normal(size=(1,256,310)).astype(np.float32)
        full_result=self.s.fit_overlap_projections(full,full[None])
        self.assertLess(float(full_result['sse'][0]),1e-9)


class BlurGainNativeDiscovery(unittest.TestCase):
    def test_native_affine_chain_inspector_and_primary_preservation(self):
        from test_overlap import NativeDiscovery
        fixture=NativeDiscovery();fixture.setUp();s=fixture.s;self.addCleanup(s.close)
        from zebsdsolve import _native
        moments=np.empty((2,5),np.float64)
        raw=s.simulate(fixture.q[:1],cells=[0],phase=0)[0]
        s._lib.ze_overlap_phase_moments(s._e,moments.ctypes.data_as(_native.f64p),moments.size)
        patterns=fixture.patterns.copy()
        patterns[0]=transform(raw,[1.1,np.log(1.1),1.4,.9,1.2,.06,-.03],moments[0,4])
        s.set_patterns(patterns[None])
        original=s.index();source=s.pattern(1,processed=False).copy()
        for preparation in ({},{"master_smooth":.3},{"clahe":True},{"bandpass":True}):
            with self.subTest(preparation=preparation):
                if preparation:
                    s.configure(**dict(dict(master_smooth=0,clahe=False,bandpass=False),**preparation))
                    s.index()
                before={key:value.copy() for key,value in s.results().items()}
                r=s.overlap_cells([0,1,3],max_components=2,primary="stored",blur_gain=True,gain_domain="source_dc")
                if not preparation:self.assertGreater(abs(float(r['blur_gain_theta'][0,1])),.01)
                moments=r['blur_gain_optimizer']['phase_moments']
                self.assertEqual(len(moments),2)
                for row,cell in enumerate([0,1]):
                    raw=[];transformed=[]
                    for j,phase in enumerate(r['phase'][row]):
                        if phase<0:continue
                        image=s.simulate(r['quaternions'][row,j][None],cells=[cell],phase=int(phase))[0]
                        dc=moments[int(phase)][4]
                        transformed.append(unit(transform(image,r['blur_gain_theta'][row],dc)))
                        raw.append(image)
                    design=np.stack(transformed).reshape(len(transformed),-1).T
                    y=unit(s.pattern(cell,processed=True)).ravel()
                    coef,_=nnls(design,y);res=y-design@coef
                    np.testing.assert_allclose(r['amplitudes'][row,:len(coef)],coef,atol=2e-5)
                    self.assertAlmostEqual(float(r['sse'][row]),float(res@res),delta=5e-6)
                    im=s.overlap_inspect(cell,r,row,0)
                    np.testing.assert_allclose(im['residual'].ravel(),res,atol=3e-6)
                    if not preparation and row==0:
                        frozen=dict(r,blur_gain_optimizer=dict(r['blur_gain_optimizer'],version=1,
                                    max_evaluations=92,singleton_max_evaluations=88))
                        legacy=s.overlap_inspect(cell,frozen,row,0)
                        for key in im:np.testing.assert_array_equal(legacy[key],im[key])
                    np.testing.assert_allclose(im['conditional']-im['selected_contribution'],im['residual'],atol=1e-7)
                self.assertEqual(r['retained_order'][2],0);self.assertTrue(r['flat'][2])
                for key,value in before.items():np.testing.assert_array_equal(s.results()[key],value)
                np.testing.assert_array_equal(s.pattern(1,processed=False),source)


    def test_joint_orders_are_monotone_and_singletons_independent(self):
        from test_overlap import NativeDiscovery
        fixture=NativeDiscovery();fixture.setUp();s=fixture.s;self.addCleanup(s.close)
        s.index();previous=None
        for k in (1,2,3,4):
            r=s.overlap_cells([1,2],max_components=k,primary="stored",blur_gain=True,gain_domain="source_dc")
            if previous is not None:
                self.assertTrue(np.all(r['sse']<=previous['sse']+2e-6))
                np.testing.assert_array_equal(r['quaternions'][:,:k-1],previous['quaternions'])
            self.assertTrue(np.all(r['sse']<=r['best_single_sse']+3e-6))
            if k==1:
                np.testing.assert_array_equal(r['sse'],r['best_single_sse'])
                np.testing.assert_array_equal(r['improvement_over_best_single'],0)
            if k==2:self.assertTrue(np.all(r['single_sse'][:,0]<=r['previous_sse']+3e-6))
            previous=r


if __name__=='__main__':unittest.main()
