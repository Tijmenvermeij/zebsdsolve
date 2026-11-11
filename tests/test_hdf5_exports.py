"""Stored-candidate export, baseline preservation and native MTEX frame round-trips.

Run with h5py and native pyMTEX installed, against a rebuilt native library.
"""
from pathlib import Path
from copy import deepcopy
import hashlib
import json
import os
import sys
import tempfile
import shutil
import warnings
import unittest

import h5py
import numpy as np
from mtex import loadEBSD, rotation, zvector, degree

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'python'))
from zebsdsolve import Solver, Error
from zebsdsolve.hdf5_export import write_emsphinx, write_oxford

class Exports(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory(prefix='zebsdsolve-hdf5-')
        cls.out = Path(cls.tmp.name)
        cls.master_paths = []
        # Synthetic EMsoft-format fixtures exercise the native importer contract;
        # this recontainerization does not establish the original NPZ crystal axes.
        with Solver() as converter:
            converter.configure(master_smooth=0)
            for folder, name in [('nickel', 'Ni'), ('gan', 'GaN')]:
                source = ROOT/f'testdata/{folder}/{name}_master_20kV.npz'
                converter.add_phase(source)
                hemispheres = converter.master(len(cls.master_paths))
                with np.load(source) as npz:
                    metadata = json.loads(npz['meta_json'].tobytes())
                cell = metadata['cell']
                lattice = [cell[k] for k in ['a_angstrom', 'b_angstrom', 'c_angstrom', 'alpha_deg', 'beta_deg', 'gamma_deg']]
                target = cls.out/f'{name}_emsoft_fixture.h5'
                with h5py.File(target, 'w') as f:
                    f.attrs['TestFixture'] = 'Synthetic EMsoft container; not a verified NPZ-axis conversion'
                    f.create_dataset('EMData/EBSDmaster/mLPNH', data=hemispheres[0])
                    f.create_dataset('EMData/EBSDmaster/mLPSH', data=hemispheres[1])
                    f.create_dataset('CrystalData/LatticeParameters', data=np.r_[np.asarray(lattice[:3])/10, lattice[3:]])
                    f.create_dataset('CrystalData/SpaceGroupNumber', data=[cell['space_group']])
                    f.create_dataset('NMLparameters/MCCLNameList/EkeV', data=[metadata['voltage_kv']])
                cls.master_paths.append(target)
        cls.s = Solver()
        cls.s.configure(static_background='none', dynamic_background='none', mask=False,
                        master_smooth=0, bandwidth=24, refine_iters=3,
                        toy_rows=2, toy_cols=3, toy_height=32, toy_width=40,
                        toy_counts=0, toy_grains=3, batch=4)
        cls.s.set_geometry(pc=[.45,.3,.6], sample_tilt_deg=70, detector_tilt_deg=0,
                           step_x_um=.2, step_y_um=.25, px_um=0)
        for master in cls.master_paths: cls.s.add_phase(master)
        cls.s.synthesize()
        cls.s.configure(roi_x=1, roi_y=0, roi_w=1, roi_h=2)
        cls.s.index()
        cls.snapshot = cls.s.export_snapshot()
        cls.baseline = cls.out/'original.h5oina'
        cls.make_oxford(cls.baseline)
        cls.original_hash = hashlib.sha256(cls.baseline.read_bytes()).hexdigest()

    @classmethod
    def tearDownClass(cls):
        cls.s.close()
        cls.tmp.cleanup()

    @classmethod
    def make_oxford(cls, path):
        with h5py.File(path,'w') as f:
            f.create_dataset('Manufacturer',data='Oxford Instruments')
            f.create_dataset('Format Version',data='8.0')
            f.attrs['Unknown root metadata']='preserved'
            g=f.create_group('1/EBSD');h=g.create_group('Header');d=g.create_group('Data')
            for key,value,unit in [('X Cells',3,'px'),('Y Cells',2,'px'),('X Step',.2,'um'),('Y Step',.25,'um'),('Pattern Width',40,'px'),('Pattern Height',32,'px')]:
                h.create_dataset(key,data=[value]).attrs['Unit']=unit
            h.create_dataset('Specimen Orientation Euler',data=[[.2,.1,.3]]).attrs['Unit']='rad'
            h.create_dataset('Detector Orientation Euler',data=[[.01,1.5,.02]]).attrs['Unit']='rad'
            h.create_dataset('Scanning Rotation Angle',data=[0.]).attrs['Unit']='rad'
            h.create_dataset('Tilt Angle',data=[70*degree]).attrs['Unit']='rad'
            h.create_dataset('Tilt Axis',data=[0.]).attrs['Unit']='rad'
            phase=cls.snapshot['phases'][0]
            p=h.create_group('Phases/1')
            p.create_dataset('Phase Name',data=phase['name'])
            p.create_dataset('Space Group',data=[phase['space_group']])
            p.create_dataset('Laue Group',data=[phase['laue']])
            p.create_dataset('Lattice Dimensions',data=[phase['lattice'][:3]]).attrs['Unit']='angstrom'
            p.create_dataset('Lattice Angles',data=[np.deg2rad(phase['lattice'][3:])]).attrs['Unit']='rad'
            d.create_dataset('X',data=np.tile(np.arange(3)*.2,2)).attrs['Unit']='um'
            d.create_dataset('Y',data=np.repeat(np.arange(2)*.25,3)).attrs['Unit']='um'
            d.create_dataset('Euler',data=np.zeros((6,3),dtype='f4')).attrs['Unit']='rad'
            d.create_dataset('Phase',data=np.full(6,1,dtype='u1'))
            d.create_dataset('Mean Angular Deviation',data=np.ones(6,dtype='f4')).attrs['Unit']='rad'
            d.create_dataset('Bands',data=np.full(6,8,dtype='u1'))
            d.create_dataset('Processed Patterns',data=np.arange(6*32*40,dtype='u1').reshape(6,32,40))
            h.create_dataset('Processed Static Background',data=np.ones((32,40),dtype='u1'))
            for key,v in [('Pattern Center X',.4),('Pattern Center Y',.4),('Detector Distance',.5)]:
                d.create_dataset(key,data=np.full(6,v,dtype='f4')).attrs['Unit']='fraction'
            unknown=f.create_dataset('Unknown/Values',data=np.array([3,5,8],dtype='i2'))
            f['Unknown/Pattern alias']=d['Processed Patterns']
            f['Unknown/Pattern soft link']=h5py.SoftLink('/1/EBSD/Data/Processed Patterns')
            unknown.attrs['Untouched']='yes'
            f['Hard link']=unknown
            f['Unknown/Soft link']=h5py.SoftLink('/Unknown/Values')
            f['Unknown/External link']=h5py.ExternalLink('other.h5','/Unknown')
            f.copy(g['Data/Euler'],f.create_group('1/Data Processing/Data'),name='Euler')
            f['1/Data Processing/SharedData']=d
            f['1/Data Processing/SourcePhase']=h5py.SoftLink('/1/EBSD/Data/Phase')
            f['1/Data Processing/RelativeEuler']=h5py.SoftLink('Data/Euler')
            f['Unknown/Old processed Euler']=h5py.SoftLink('/1/Data Processing/Data/Euler')

    @staticmethod
    def expected_oxford(raw, sample_basis, valid, phase=None):
        values = raw['results'] if phase is None else raw['candidates']
        quaternions = np.asarray(values['quaternions']).reshape((-1, 6, 4))
        selected = np.asarray(raw['results']['phase'])
        output = []
        for cell in np.flatnonzero(valid):
            p = int(selected[cell]) if phase is None else phase
            crystal = rotation.id if p == 0 else rotation.byAxisAngle(zvector, 30*degree)
            q = rotation(quaternions[0 if phase is None else phase, cell])
            output.append((sample_basis*q.inv()*crystal).data)
        return rotation(np.asarray(output))

    def oxford_solver(self, source=None, masters=None):
        s = Solver()
        self.addCleanup(s.close)
        s.configure(static_background='none', dynamic_background='none', mask=False,
                    master_smooth=0, bandwidth=24, refine_iters=3, batch=4)
        s.open(source or self.baseline)
        for master in masters or self.master_paths: s.add_phase(master)
        s.configure(roi_x=1, roi_y=0, roi_w=1, roi_h=2)
        return s

    def test_automatic_oxford_native_frames_calibration_and_serialization(self):
        s = self.oxford_solver()
        self.assertEqual(s.geometry['rot'], [0,0,0])
        self.assertEqual(s.geometry['detector_tilt_deg'], 0)
        self.assertEqual(s.geometry['px_um'], 0)
        # A nonzero residual exercises effective header serialization once only.
        s.set_geometry(rot=[.002,-.003,.001])
        s.index()
        raw = s.export_snapshot(); native = s.export_oxford_snapshot()
        frame = native['oxford_frame']; b = rotation.byAxisAngle(zvector, 90*degree)
        with self.assertRaisesRegex(Error, 'OxfordFrameOverrideConflictsWithAcquisition'):
            s.export_oxford_snapshot([1,0,0,0])
        for override in [b.data, -b.data]:
            equivalent = s.export_oxford_snapshot(override)
            self.assertTrue(np.allclose(equivalent['results']['quaternions'], native['results']['quaternions']))
        self.assertEqual(frame['convention_version'], 1)
        self.assertFalse(frame['preserve_source_pattern_centres'])
        self.assertEqual(frame['warnings'], [])
        self.assertLess(float(rotation(frame['engine_to_cs1']).angle(b))/degree, .0001)
        for p, angle in enumerate([0,30]):
            c = rotation.id if not angle else rotation.byAxisAngle(zvector, angle*degree)
            self.assertLess(float(rotation(frame['phase_crystal_to_master'][p]).angle(c))/degree, .0001)
        valid = np.asarray(raw['results']['phase']) >= 0
        expected = self.expected_oxford(raw, b, valid)
        actual = rotation(np.asarray(native['results']['quaternions']).reshape(6,4)[valid]).inv()
        self.assertLess(float(np.max(actual.angle(expected)))/degree, .0001)
        for p in range(2):
            cells = np.asarray(raw['candidates']['sweep']).reshape(2,6)[p] >= 0
            expected_candidate = self.expected_oxford(raw, b, cells, phase=p)
            actual_candidate = rotation(np.asarray(native['candidates']['quaternions']).reshape(2,6,4)[p,cells]).inv()
            self.assertLess(float(np.max(actual_candidate.angle(expected_candidate)))/degree, .0001)
        target = self.out/'automatic.h5oina'
        s.export(target)
        imported = loadEBSD(str(target), grid=False, eulerCorrection=rotation.id)
        for p in range(2):
            cells = valid & (np.asarray(raw['results']['phase']) == p)
            if cells.any():
                self.assertLess(float(np.max(imported.rotations[cells].angle(self.expected_oxford(raw,b,cells))))/degree, .002)
        with h5py.File(target) as f, h5py.File(self.baseline) as original:
            provenance = json.loads(f['zebsdsolve/ProvenanceJSON'][()])
            self.assertEqual(provenance['oxford_frame'], frame)
            self.assertEqual(provenance['calibration'], native['calibration'])
            for key, field in [('Detector Orientation Euler','effective_detector_euler_radians'), ('Tilt Angle','effective_tilt_radians')]:
                self.assertTrue(np.allclose(f['1/EBSD/Header/'+key][()].reshape(-1), np.asarray(native['calibration'][field]).reshape(-1)))
                backup = f['zebsdsolve/OriginalCalibration/'+key]
                self.assertTrue(np.array_equal(backup[()], original['1/EBSD/Header/'+key][()]))
                self.assertEqual(dict(backup.attrs), dict(original['1/EBSD/Header/'+key].attrs))
            for key in ['Specimen Orientation Euler','Tilt Axis','Scanning Rotation Angle']:
                self.assertTrue(np.array_equal(f['1/EBSD/Header/'+key][()], original['1/EBSD/Header/'+key][()]))
        # Export captures the indexed pose, even after controls change.
        s.set_geometry(rot=[.1,.2,.3], sample_tilt_deg=60)
        self.assertEqual(s.export_oxford_snapshot(), native)
        # Replacing the source must discard both old results and Oxford metadata.
        s.set_patterns(np.arange(6*32*40, dtype=np.uint8).reshape(2,3,32,40))
        s.index()
        blocked = self.out/'replaced-source.h5oina'
        with self.assertRaisesRegex(Error, 'OxfordFrameRequiresOriginalAcquisition'):
            s.export(blocked, reference=self.baseline)
        self.assertFalse(blocked.exists())

    def test_automatic_oxford_rejects_unverified_inputs_before_output(self):
        import shutil
        for key, error in [('Tilt Axis','UnsupportedOxfordTiltAxis'), ('Scanning Rotation Angle','UnsupportedOxfordScanRotation')]:
            with self.subTest(header=key):
                source = self.out/(key.replace(' ','-')+'.h5oina')
                target = self.out/(key.replace(' ','-')+'-blocked.h5oina')
                shutil.copyfile(self.baseline, source)
                with h5py.File(source,'r+') as f: f['1/EBSD/Header/'+key][...] = .1
                s = self.oxford_solver(source); s.index()
                with self.assertRaisesRegex(Error, error): s.export(target)
                with self.assertRaisesRegex(Error, error): s.export(target,engine_to_cs1=[1,0,0,0],frame_provenance='Cannot bypass acquisition support')
                self.assertFalse(target.exists())
        unknown = self.oxford_solver(masters=[ROOT/'testdata/nickel/Ni_master_20kV.npz'])
        unknown.index()
        target = self.out/'unknown-master-blocked.h5oina'
        with self.assertRaisesRegex(Error,'UnknownMasterCrystalConvention'): unknown.export(target)
        self.assertFalse(target.exists())
        spatial = self.oxford_solver()
        spatial.set_geometry(px_um=100)
        spatial.index()
        target = self.out/'unverified-scan-translation.h5oina'
        with self.assertRaisesRegex(Error,'OxfordScanTranslationNotValidated'): spatial.export(target)
        self.assertFalse(target.exists())

    def test_ebsdsim_v3_crystal_frame_is_supported(self):
        s = self.oxford_solver(masters=[ROOT/'testdata/gan/GaN_master_20kV.npz'])
        s.index()
        raw = s.export_snapshot(); native = s.export_oxford_snapshot()
        self.assertEqual(native['phases'][0]['master_source'], 'ebsdsim')
        self.assertEqual(native['phases'][0]['master_source_format_version'], 3)
        self.assertEqual(native['phases'][0]['master_crystal_convention'], 'X||a,Z||c*')
        b = rotation.byAxisAngle(zvector,90*degree)
        c = rotation.byAxisAngle(zvector,30*degree)
        self.assertLess(float(rotation(native['oxford_frame']['phase_crystal_to_master'][0]).angle(c))/degree,.0001)
        valid = np.asarray(raw['results']['phase']) >= 0
        expected = b*rotation(np.asarray(raw['results']['quaternions']).reshape(6,4)[valid]).inv()*c
        target=self.out/'ebsdsim-v3.h5oina';s.export(target)
        imported=loadEBSD(str(target),grid=False,eulerCorrection=rotation.id)
        self.assertLess(float(np.max(imported.rotations[valid].angle(expected)))/degree,.002)

    def test_emsphinx_default_import_equals_ang(self):
        ang=self.out/'results.ang'; h5=self.out/'results.h5'
        self.s.export(ang); self.s.export(h5)
        a=loadEBSD(str(ang),grid=False)
        h=loadEBSD(str(h5),grid=False,dataSet='Scan 1')
        self.assertTrue(np.array_equal(a.phaseId,h.phaseId))
        self.assertTrue(np.allclose(a.pos.data,h.pos.data))
        valid=a.phaseId>0
        self.assertLess(float(np.max(a.rotations[valid].angle(h.rotations[valid]))/degree),.002)
        s=self.snapshot; n=6; c=s['candidates']; cp=np.asarray(c['sweep']).reshape(2,n)
        self.assertEqual(cp.shape,(2,n))
        self.assertTrue(np.all(cp[:,[1,4]]>=0))
        self.assertTrue(np.all(cp[:,[0,2,3,5]]==-1))
        with h5py.File(h5) as f:
            p=f['Scan 2/EBSD/Data/Phase'][()]
            chosen=np.asarray(s['results']['phase'])
            self.assertTrue(np.all(p[[1,4]]!=chosen[[1,4]]))
            for cell in [1,4]:
                phase=int(p[cell]); i=phase*n+cell
                expected=np.asarray(c['euler']).reshape(2,n,3)[phase,cell]
                actual=[f['Scan 2/EBSD/Data/'+key][cell] for key in ['Phi1','Phi','Phi2']]
                self.assertTrue(np.array_equal(expected,actual))
                self.assertEqual(float(f['Scan 2/EBSD/Data/CI'][cell]),c['ncc'][i])

    def test_oxford_preserves_baseline_and_explicit_frame(self):
        target=self.out/'reindexed.h5oina'
        t=rotation.byAxisAngle(zvector,90*degree)
        self.s.export(target,reference=self.baseline,engine_to_cs1=t.data,
                      frame_provenance='Known-axis controlled test: engine→CS1 +90deg z')
        self.assertEqual(self.original_hash,hashlib.sha256(self.baseline.read_bytes()).hexdigest())
        valid=np.asarray(self.snapshot['results']['phase'])>=0
        expected=self.expected_oxford(self.snapshot, t, valid)
        # Explicit identity avoids a known importer acquisition-surface convention ambiguity;
        # H5OINA Euler is CS2→CS1, radians. Original nonzero specimen metadata is preserved.
        e=loadEBSD(str(target),grid=False,eulerCorrection=rotation.id)
        self.assertLess(float(np.max(e.rotations[valid].angle(expected))/degree),.002)
        with h5py.File(target) as f,h5py.File(self.baseline) as original:
            self.assertNotIn('1/Data Processing',f)
            self.assertNotIn('zebsdsolve/OriginalDataProcessing',f)
            self.assertNotIn('zebsdsolve/OriginalIndexing',f)
            self.assertNotIn('Unknown/Old processed Euler',f)
            for key in ['1/EBSD/Header/Specimen Orientation Euler','1/EBSD/Header/Detector Orientation Euler','Unknown/Values','1/EBSD/Header/Processed Static Background']:
                self.assertTrue(np.array_equal(f[key][()],original[key][()]))
                self.assertEqual(dict(f[key].attrs),dict(original[key].attrs))
            self.assertEqual(f['Hard link'].id,f['Unknown/Values'].id)
            for key in ['1/EBSD/Data/Processed Patterns','Unknown/Pattern alias','Unknown/Pattern soft link']:
                self.assertNotIn(key,f)
            self.assertEqual(f.get('Unknown/Soft link',getlink=True).path,'/Unknown/Values')
            self.assertEqual(f.get('Unknown/External link',getlink=True).filename,'other.h5')
            self.assertLess(target.stat().st_size,self.baseline.stat().st_size+100000)
            self.assertIn('2',f['1/EBSD/Header/Phases'])
            for key in ['Euler','Phase','Mean Angular Deviation','Bands']:
                empty = f['1/EBSD/Data/'+key][()][~valid]
                self.assertTrue(np.isnan(empty).all() if key == 'Mean Angular Deviation' else np.all(empty == 0))
            self.assertTrue(np.isnan(f['1/EBSD/Data/Mean Angular Deviation'][()][valid]).all())
            pc=np.asarray(self.snapshot['pattern_centres']).reshape(6,3)
            self.assertTrue(np.allclose(f['1/EBSD/Data/Pattern Center X'][()][valid],pc[valid,0]))

    def test_computed_solutions_survive_missing_quality_and_unaccepted_sweep(self):
        s = deepcopy(self.snapshot)
        cell = 1
        phase = int(s['results']['phase'][cell]); other = 1-phase
        s['results']['sweep'][cell] = -1
        s['results']['ncc'][cell] = None
        s['results']['nmi'][cell] = -10
        s['results']['z'][cell] = None
        for key in ('ncc', 'nmi', 'z'): s['candidates'][key][other*6+cell] = None
        h5 = self.out/'missing-quality.h5'
        write_emsphinx(h5, s)
        with h5py.File(h5) as f:
            self.assertEqual(f['Scan 1/EBSD/Data/Phase'][cell], phase)
            self.assertEqual(f['Scan 1/EBSD/Data/Sweep'][cell], -1)
            self.assertTrue(np.isnan(f['Scan 1/EBSD/Data/CI'][cell]))
            self.assertEqual(f['Scan 2/EBSD/Data/Phase'][cell], other)
            self.assertTrue(np.isnan(f['zebsdsolve/Selected/ncc'][cell]))
            self.assertTrue(np.isnan(f['zebsdsolve/Candidates/ncc'][other,cell]))
        oxford = self.out/'missing-quality.h5oina'
        write_oxford(oxford, self.baseline, s, [1,0,0,0], 'Serialization-only identity fixture')
        with h5py.File(oxford) as f:
            self.assertGreater(f['1/EBSD/Data/Phase'][cell], 0)
            self.assertTrue(np.isnan(f['zebsdsolve/NCC'][cell]))
            self.assertEqual(f['zebsdsolve/SWEEP'][cell], -1)
            self.assertNotIn('zebsdsolve/OriginalIndexing', f)
            self.assertTrue(np.all(f['1/EBSD/Data/Phase'][()][[0,2,3,5]] == 0))
            self.assertTrue(np.all(f['1/EBSD/Data/Euler'][()][[0,2,3,5]] == 0))

    def test_all_attempted_candidates_export_without_selected_phase(self):
        s = deepcopy(self.snapshot)
        s['results']['phase'][1] = -1
        target = self.out/'candidates-without-selected.h5'
        write_emsphinx(target, s)
        with h5py.File(target) as f:
            self.assertEqual(f['Scan 1/EBSD/Data/Phase'][1], 255)
            self.assertEqual(set([int(f['Scan 2/EBSD/Data/Phase'][1]), int(f['Scan 3/EBSD/Data/Phase'][1])]), {0,1})
            self.assertTrue(np.all(f['zebsdsolve/Selected/quaternions'][1] == 0))
        # The same rule applies to a sole retained phase candidate.
        s['phases'] = s['phases'][:1]
        s['candidates'] = {key: 1 if key == 'phase_count' else value[:6*(4 if key == 'quaternions' else 3 if key == 'euler' else 1)] for key,value in s['candidates'].items()}
        write_emsphinx(target, s)
        with h5py.File(target) as f: self.assertEqual(f['Scan 2/EBSD/Data/Phase'][1], 0)

    def test_unresolved_frame_and_source_overwrite_fail_before_write(self):
        target=self.out/'blocked.h5oina'
        with self.assertRaises(Error): self.s.export(target,reference=self.baseline)
        self.assertFalse(target.exists())
        with self.assertRaises(ValueError):
            self.s.export(self.baseline,reference=self.baseline,engine_to_cs1=[1,0,0,0],frame_provenance='test')
        self.assertEqual(self.original_hash,hashlib.sha256(self.baseline.read_bytes()).hexdigest())

    def test_sparse_original_phase_ids_are_preserved(self):
        import shutil
        source=self.out/'sparse.h5oina';target=self.out/'sparse-output.h5oina'
        shutil.copyfile(self.baseline,source)
        with h5py.File(source,'r+') as f:
            f.move('1/EBSD/Header/Phases/1','1/EBSD/Header/Phases/7')
            f['1/EBSD/Data/Phase'][...]=7
        self.s.export(target,reference=source,engine_to_cs1=[1,0,0,0],frame_provenance='Controlled identity frame')
        with h5py.File(target) as f:
            self.assertIn('7',f['1/EBSD/Header/Phases'])
            self.assertIn('8',f['1/EBSD/Header/Phases'])
            valid=np.asarray(self.snapshot['results']['phase'])>=0
            self.assertTrue(np.all(f['1/EBSD/Data/Phase'][()][~valid]==0))

    def test_metadata_reference_rejected_without_output(self):
        import shutil
        source=self.out/'references.h5oina';target=self.out/'references-output.h5oina'
        shutil.copyfile(self.baseline,source)
        with h5py.File(source,'r+') as f:
            f.create_dataset('Unknown/Reference',data=f['Unknown/Values'].ref,dtype=h5py.ref_dtype)
        with self.assertRaisesRegex(ValueError,'reference datasets'):
            self.s.export(target,reference=source,engine_to_cs1=[1,0,0,0],frame_provenance='Controlled identity')
        self.assertFalse(target.exists())

    def test_soft_pattern_dataset_omits_target_payload(self):
        import shutil
        source=self.out/'soft-pattern.h5oina';target=self.out/'soft-pattern-output.h5oina'
        shutil.copyfile(self.baseline,source)
        with h5py.File(source,'r+') as f:
            del f['1/EBSD/Data/Processed Patterns']
            f['1/EBSD/Data/Processed Patterns']=h5py.SoftLink('/Unknown/Pattern alias')
        self.s.export(target,reference=source,engine_to_cs1=[1,0,0,0],frame_provenance='Controlled identity')
        with h5py.File(target) as f:
            self.assertNotIn('Unknown/Pattern alias',f)
            self.assertNotIn('Unknown/Pattern soft link',f)
            self.assertNotIn('1/EBSD/Data/Processed Patterns',f)

    def test_processing_link_through_acquisition_alias(self):
        import shutil
        source=self.out/'alias-processing.h5oina';target=self.out/'alias-processing-output.h5oina'
        shutil.copyfile(self.baseline,source)
        with h5py.File(source,'r+') as f:
            f['2']=f['1']
            f['Unknown/Processing through alias']=h5py.SoftLink('/2/Data Processing/Data/Euler')
        self.s.export(target,reference=source,engine_to_cs1=[1,0,0,0],frame_provenance='Controlled identity')
        with h5py.File(target) as f:
            self.assertNotIn('1/Data Processing',f)
            self.assertNotIn('2/Data Processing',f)
            self.assertNotIn('Unknown/Processing through alias',f)
            self.assertNotIn('zebsdsolve/OriginalDataProcessing',f)

    def test_large_origin_reordered_pixels_rejected(self):
        import shutil
        source=self.out/'reordered.h5oina';target=self.out/'reordered-output.h5oina'
        shutil.copyfile(self.baseline,source)
        with h5py.File(source,'r+') as f:
            x=100000+np.tile(np.arange(3)*.2,2)
            x[[1,2]]=x[[2,1]]
            f['1/EBSD/Data/X'][...]=x
        with self.assertRaisesRegex(ValueError,'positions'):
            self.s.export(target,reference=source,engine_to_cs1=[1,0,0,0],frame_provenance='Controlled identity')
        self.assertFalse(target.exists())

    def test_z_frozen_context_and_incompatible_roi_retention(self):
        old=self.s.export_snapshot()
        moved=list(old['geometry']['pc']);moved[0]+=.1
        self.s.set_geometry(pc=moved)
        self.s.configure(phase_by='ncc')
        frozen=self.s.export_snapshot()
        self.assertEqual(frozen['geometry'],old['geometry'])
        self.assertEqual(frozen['settings'],old['settings'])
        self.assertEqual(frozen['pattern_centres'],old['pattern_centres'])
        self.s.configure(roi_x=0,roi_y=0,roi_w=1,roi_h=1)
        self.s.index()
        with self.assertRaisesRegex(Error,'MixedIndexingContexts'):
            self.s.export_snapshot()
        self.s.configure(roi_x=0,roi_y=0,roi_w=0,roi_h=0)
        self.s.index()
        self.assertEqual(self.s.export_snapshot()['geometry']['pc'],moved)

    def test_plane_snapshot_uses_frozen_calibrated_pc_and_resets_on_source_change(self):
        from zebsdsolve._native import Geometry
        import ctypes
        self.assertEqual((ctypes.sizeof(Geometry), Geometry.plane_kind.offset, Geometry.plane.offset), (192, 88, 96))
        s = self.oxford_solver()
        # Unrelated field edits/replay must not invent an Oxford detector pixel size.
        s.set_geometry(rot=[.002, -.003, .001])
        self.assertEqual(s.geometry['px_um'], 0)
        P = np.array([[.001,.002,.4],[.003,.004,.2],[.005,.006,.5],[0,0,1.]])
        s.set_geometry(plane_kind='affine', plane=P)
        self.assertTrue(np.array_equal(s.geometry['plane'], P))
        centre_pc = (P @ [1., .5, 1.])[:3].tolist()
        self.assertEqual(s.geometry['pc'], centre_pc)
        s.index()
        frozen = s.export_snapshot()
        self.assertEqual(frozen['geometry']['pc'], centre_pc)
        target = self.out/'frozen-plane.h5'
        s.export(target)
        with h5py.File(target, 'r') as f:
            self.assertTrue(np.array_equal(f['zebsdsolve/PatternCentres'][()], np.asarray(frozen['pattern_centres']).reshape(6,3)))
            self.assertIn('Bruker', f['zebsdsolve/PatternCentres'].attrs['Convention'])
        expected = np.array([P @ [c,r,1.] for r in range(2) for c in range(3)])[:,:3]
        self.assertTrue(np.allclose(np.asarray(frozen['pattern_centres']).reshape(-1,3), expected, atol=1e-12))
        oxford = s.export_oxford_snapshot()
        self.assertTrue(oxford['oxford_frame']['preserve_source_pattern_centres'])
        self.assertEqual(len(oxford['oxford_frame']['warnings']), 1)
        s.set_geometry(plane_kind='none', pc=[.2,.3,.4])
        self.assertEqual(s.export_snapshot(), frozen)
        s.set_geometry(plane_kind='affine', plane=P)
        s.set_patterns(np.zeros((2,3,32,40), dtype=np.float32))
        self.assertEqual(s.geometry['plane_kind'], 'none')
        s.set_geometry(plane_kind='affine', plane=P)
        s.open(self.baseline)
        self.assertEqual(s.geometry['plane_kind'], 'none')
        self.assertEqual(s.geometry['px_um'], 0)

    def test_oxford_plane_preserves_acquisition_pc_and_native_frames(self):
        pc_keys = ['Pattern Center X', 'Pattern Center Y', 'Detector Distance']
        message = ('Orientations were indexed with a fitted source plane. Standard Oxford '
                   'pattern-centre fields retain acquisition values. Fitted values are stored '
                   'in /zebsdsolve/PatternCentres.')
        source = self.out/'plane-acquisition.h5oina'
        shutil.copyfile(self.baseline, source)
        with h5py.File(source, 'r+') as f:
            data = f['1/EBSD/Data']
            for k, key in enumerate(pc_keys):
                del data[key]
                dtype = '<f8' if k == 1 else '<f4'
                d = data.create_dataset(key, data=np.array([.11+k*.05, .2, np.nan, .4, .5, .6], dtype=dtype))
                d.attrs['Unit'] = 'fraction'
                d.attrs['Acquisition metadata'] = np.array([7, 13], dtype='i2')
        for kind in ['affine', 'projective']:
            for frame_mode in ['nominal', 'follow']:
                with self.subTest(kind=kind, frame=frame_mode):
                    s = self.oxford_solver(source)
                    s.configure(plane_frame=frame_mode)
                    # These are final indexed poses: a retained nominal residual and a
                    # nonzero follow-equivalent residual. Export adds no plane-dependent
                    # frame rotation to either one.
                    residual = [.002,-.003,.001] if frame_mode == 'nominal' else [.015,-.02,.01]
                    P = np.array([[.001,.002,.4],[.003,.004,.2],[.005,.006,.5],[0,0,1.]])
                    if kind == 'projective': P[3,:2] = [.001, -.002]
                    s.set_geometry(rot=residual, plane_kind=kind, plane=P, px_um=350)
                    s.index()
                    raw = s.export_snapshot()
                    native = s.export_oxford_snapshot()
                    policy = native['oxford_frame']
                    self.assertTrue(policy['preserve_source_pattern_centres'])
                    self.assertEqual(policy['warnings'], [message])
                    b = rotation.byAxisAngle(zvector, 90*degree)
                    valid = np.asarray(raw['results']['phase']) >= 0
                    expected = self.expected_oxford(raw, b, valid)
                    actual = rotation(np.asarray(native['results']['quaternions']).reshape(6,4)[valid]).inv()
                    self.assertLess(float(np.max(actual.angle(expected)))/degree, .0001)
                    # The indexed frozen context controls policy and effective calibration,
                    # including after current controls drop the plane and change the pose.
                    s.set_geometry(plane_kind='none', rot=[.2,.3,.1], sample_tilt_deg=60, px_um=0)
                    self.assertEqual(s.export_oxford_snapshot(), native)
                    target = self.out/f'plane-{kind}-{frame_mode}.h5oina'
                    with warnings.catch_warnings(record=True) as caught:
                        warnings.simplefilter('always')
                        s.export(target)
                    self.assertEqual([str(w.message) for w in caught], [message])
                    imported = loadEBSD(str(target), grid=False, eulerCorrection=rotation.id)
                    self.assertLess(float(np.max(imported.rotations[valid].angle(expected)))/degree, .002)
                    with h5py.File(target) as f, h5py.File(source) as original:
                        for key in pc_keys:
                            a, b0 = f['1/EBSD/Data/'+key], original['1/EBSD/Data/'+key]
                            self.assertEqual(a.dtype, b0.dtype)
                            self.assertEqual(a.shape, b0.shape)
                            self.assertEqual(a[()].tobytes(), b0[()].tobytes())
                            self.assertEqual(set(a.attrs), set(b0.attrs))
                            for attr in a.attrs: np.testing.assert_equal(a.attrs[attr], b0.attrs[attr])
                        self.assertTrue(np.array_equal(f['zebsdsolve/PatternCentres'][()], np.asarray(native['pattern_centres']).reshape(6,3)))
                        self.assertEqual(json.loads(f['zebsdsolve/ProvenanceJSON'][()])['oxford_frame'], policy)
                        for key, field in [('Detector Orientation Euler','effective_detector_euler_radians'), ('Tilt Angle','effective_tilt_radians')]:
                            self.assertTrue(np.allclose(f['1/EBSD/Header/'+key][()].reshape(-1), np.asarray(native['calibration'][field]).reshape(-1)))

    def test_oxford_plane_requires_native_pattern_centres_before_warning(self):
        for flag in [False, True]:
            with self.subTest(native_flag=flag):
                snapshot = deepcopy(self.snapshot)
                del snapshot['pattern_centres']
                snapshot['geometry']['plane_kind'] = 0 if flag else 1
                snapshot['oxford_frame'] = {'preserve_source_pattern_centres': flag}
                target = self.out/f'missing-plane-pc-{flag}.h5oina'
                original = b'Existing output must survive an incomplete snapshot'
                target.write_bytes(original)
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter('always')
                    with self.assertRaisesRegex(ValueError, 'requires pattern_centres'):
                        write_oxford(target, self.baseline, snapshot, [1,0,0,0], 'Controlled identity')
                self.assertEqual(caught, [])
                self.assertEqual(target.read_bytes(), original)
                self.assertEqual(list(self.out.glob('.'+target.name+'.*.h5')), [])

    def test_oxford_plane_warning_as_error_keeps_existing_output(self):
        snapshot = deepcopy(self.snapshot)
        snapshot['geometry']['plane_kind'] = 1
        snapshot['oxford_frame'] = {'preserve_source_pattern_centres': True}
        target = self.out/'warning-error-existing.h5oina'
        original_bytes = b'Existing output must survive a warning promoted to an error'
        target.write_bytes(original_bytes)
        with warnings.catch_warnings():
            warnings.simplefilter('error', UserWarning)
            with self.assertRaisesRegex(UserWarning, 'Standard Oxford pattern-centre fields retain acquisition values'):
                write_oxford(target, self.baseline, snapshot, [1,0,0,0], 'Controlled identity')
        self.assertEqual(target.read_bytes(), original_bytes)
        self.assertEqual(list(self.out.glob('.'+target.name+'.*.h5')), [])

    def test_direct_flagless_plane_snapshot_preserves_pc_and_normalizes_policy(self):
        from zebsdsolve.hdf5_export import OXFORD_PLANE_PC_WARNING
        for native_flag in [None, False]:
            with self.subTest(native_flag=native_flag):
                snapshot = deepcopy(self.snapshot)
                snapshot['geometry']['plane_kind'] = 1
                if native_flag is not None:
                    snapshot['oxford_frame'] = {'preserve_source_pattern_centres': native_flag, 'warnings': []}
                original = deepcopy(snapshot)
                target = self.out/f'direct-flagless-plane-{native_flag}.h5oina'
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter('always')
                    write_oxford(target, self.baseline, snapshot, [1,0,0,0], 'Controlled identity')
                self.assertEqual([str(w.message) for w in caught], [OXFORD_PLANE_PC_WARNING])
                self.assertEqual(snapshot, original)
                with h5py.File(target) as f, h5py.File(self.baseline) as acquisition:
                    for key in ['Pattern Center X', 'Pattern Center Y', 'Detector Distance']:
                        a, b = f['1/EBSD/Data/'+key], acquisition['1/EBSD/Data/'+key]
                        self.assertEqual(a.dtype, b.dtype)
                        self.assertEqual(a[()].tobytes(), b[()].tobytes())
                        for attr in b.attrs: np.testing.assert_equal(a.attrs[attr], b.attrs[attr])
                    policy = json.loads(f['zebsdsolve/ProvenanceJSON'][()])['oxford_frame']
                    self.assertTrue(policy['preserve_source_pattern_centres'])
                    self.assertEqual(policy['warnings'], [OXFORD_PLANE_PC_WARNING])

    def test_oxford_plane_keeps_absent_acquisition_pc_fields_absent(self):
        source = self.out/'plane-no-acquisition-pc.h5oina'
        shutil.copyfile(self.baseline, source)
        keys = ['Pattern Center X', 'Pattern Center Y', 'Detector Distance']
        with h5py.File(source, 'r+') as f:
            for key in keys: del f['1/EBSD/Data/'+key]
        s = self.oxford_solver(source)
        P = np.array([[.001,.002,.4],[.003,.004,.2],[.005,.006,.5],[0,0,1.]])
        s.set_geometry(plane_kind='affine', plane=P, px_um=350)
        s.index()
        target = self.out/'plane-no-acquisition-pc-export.h5oina'
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            s.export(target)
        self.assertEqual(len(caught), 1)
        with h5py.File(target) as f:
            for key in keys: self.assertNotIn('1/EBSD/Data/'+key, f)
            self.assertIn('zebsdsolve/PatternCentres', f)

    def test_native_pattern_centres_are_stored_and_validated(self):
        target = self.out/'ordinary-native-pcs.h5'
        write_emsphinx(target, self.snapshot)
        with h5py.File(target, 'r') as f:
            self.assertTrue(np.array_equal(f['zebsdsolve/PatternCentres'][()], np.asarray(self.snapshot['pattern_centres']).reshape(6,3)))
        for invalid in [[0,0,0], [float('nan')]*18]:
            bad = deepcopy(self.snapshot)
            bad['pattern_centres'] = invalid
            with self.assertRaisesRegex(ValueError, 'pattern_centres'):
                write_emsphinx(self.out/'invalid-native-pcs.h5', bad)
            self.assertFalse((self.out/'invalid-native-pcs.h5').exists())

    def test_high_level_plane_kind_routes_to_fit_unless_matrix_is_given(self):
        from zebsdsolve.api import _open
        class Recorder:
            def __init__(self): self.settings = {}; self.geometry = {}
            def configure(self, **kwargs): self.settings.update(kwargs)
            def set_patterns(self, scan): pass
            def set_geometry(self, **kwargs): self.geometry.update(kwargs)
        setting = Recorder()
        _open(setting, np.zeros((1,1,4,4)), [], {'plane_kind': 'projective'})
        self.assertEqual(setting.settings, {'plane_kind': 'projective'})
        self.assertEqual(setting.geometry, {})
        P = np.array([[.001,.002,.4],[.003,.004,.2],[.005,.006,.5],[0,0,1.]])
        geometry = Recorder()
        _open(geometry, np.zeros((1,1,4,4)), [], {'plane_kind': 'projective', 'plane': P})
        self.assertEqual(geometry.settings, {})
        self.assertEqual(geometry.geometry['plane_kind'], 'projective')
        self.assertTrue(np.array_equal(geometry.geometry['plane'], P))

    def test_synthesis_preserves_effective_geometry_and_calibration(self):
        P = np.array([[.001,.002,.4],[.003,.004,.2],[.005,.006,.5],[0,0,1.]])
        for oxford, plane in [(False, False), (False, True), (True, False), (True, True)]:
            with self.subTest(oxford=oxford, plane=plane), Solver() as s:
                s.configure(static_background='none', dynamic_background='none', master_smooth=0,
                            toy_rows=2, toy_cols=3, toy_height=32, toy_width=40,
                            toy_counts=0, toy_contrast=1, toy_background_sigma=0,
                            toy_blank=0, toy_spread_deg=0, bandwidth=16, refine_iters=1)
                if oxford:
                    s.open(self.baseline)
                s.add_phase(self.master_paths[0])
                s.set_geometry(rot=[.002,-.003,.001], plane_kind='affine' if plane else 'none', plane=P)
                before = s.geometry
                self.assertEqual(before['px_um'], 0)
                self.assertEqual(s.has_acquisition_calibration, oxford)
                truth = s.synthesize()
                self.assertEqual(s.has_acquisition_calibration, oxford)
                expected = deepcopy(before)
                if not oxford:
                    expected['px_um'] = 1000*s.settings['detector_width_mm']/40
                if plane:
                    expected['pc'] = (P @ [1., .5, 1.])[:3].tolist()
                self.assertEqual(truth['geometry'], expected)
                self.assertEqual(s.geometry, expected)
                qs = truth['quaternions'].reshape(-1,4)
                sim = s.simulate(qs, cells=np.arange(6), full=True)
                raw = np.array([s.pattern(c, processed=False, full=True) for c in range(6)])
                expected_raw = (sim-sim.min())/(np.mean(sim, dtype=np.float64)-sim.min())
                self.assertTrue(np.allclose(raw, expected_raw, rtol=2e-6, atol=2e-6))
                # Both internal pattern replacements must retain the calibration and plane,
                # so replay with the same seed reproduces the same quaternions and pixels.
                again = s.synthesize()
                self.assertEqual(s.has_acquisition_calibration, oxford)
                self.assertTrue(np.array_equal(again['quaternions'], truth['quaternions']))
                self.assertTrue(np.array_equal(raw, [s.pattern(c, processed=False, full=True) for c in range(6)]))
                s.index()
                snapshot = s.export_snapshot()
                wire_geometry = dict(expected, plane_kind=int(plane), plane=P.ravel().tolist(), _pad=0)
                self.assertEqual(snapshot['geometry'], wire_geometry)
                self.assertEqual(snapshot['calibration']['kind'], 'oxford' if oxford else 'legacy')
                from zebsdsolve._native import PlaneResult
                report = s._plane_result(PlaneResult())
                self.assertEqual('sample_tilt_deg' in report['said'], not oxford)
                if oxford:
                    self.assertEqual(snapshot['calibration']['status'], 'supported')

    def test_native_simulation_and_npz_reload_preserve_crystal_convention(self):
        with Solver() as s:
            s.configure(sim_halfw=8, sim_rank=2, sim_bin_kev=100,
                        static_background='none', dynamic_background='none',
                        master_smooth=0, bandwidth=16, refine_iters=1)
            s.set_crystal(ROOT/'web/app/cif/Ni.cif')
            result = s.simulate_master()
            self.assertEqual(result['failed'], 0)
            s.add_simulated_phase('native Ni')
            direct = s.master(0)
            path = self.out/'native-ni.npz'
            s.save_simulation(path)
            s.add_phase(path)
            self.assertTrue(np.allclose(direct, s.master(1), atol=1e-6))
            s.set_patterns(np.arange(32*40, dtype=np.float32).reshape(1,1,32,40))
            s.index()
            phases = s.export_snapshot()['phases']
            for p in phases:
                self.assertEqual(p['master_source'], 'zebsdsolve')
                self.assertEqual(p['master_crystal_convention'], 'X||a,Z||c*')
            self.assertEqual(phases[1]['master_source_format_version'], 3)

    def test_mask_auto_switch_and_override(self):
        edax=self.out/'edax.h5'
        with h5py.File(edax,'w') as f:
            h=f.create_group('Scan 1/EBSD/Header')
            for key,v in [('nRows',2),('nColumns',3),('Pattern Height',32),('Pattern Width',40)]: h.create_dataset(key,data=[v])
            f.create_dataset('Scan 1/EBSD/Data/Pattern',data=np.zeros((6,32,40),dtype='u1'))
        with Solver() as s:
            s.open(self.baseline); self.assertFalse(s.settings['mask'])
            s.open(edax); self.assertTrue(s.settings['mask'])
            s.configure(mask=False); s.open(edax); self.assertFalse(s.settings['mask'])
            s.configure(mask=True); s.open(self.baseline); self.assertTrue(s.settings['mask'])

if __name__=='__main__': unittest.main(verbosity=2)
