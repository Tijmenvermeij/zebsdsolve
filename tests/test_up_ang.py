"""UP stream + ANG scalar/grid adapter; native pyMTEX ANG import is the metadata oracle.

ANG uses EDAX micrometres, x-fast/y-next increasing raster order. No ANG orientations,
phases or acquisition calibration enter the engine. Pattern centre is a scalar-only,
producer-specific detector normalization, independent of MTEX orientation import. All temporary fixtures are local.
"""
from pathlib import Path
import struct
import json
import h5py
import sys
import tempfile
import unittest
import numpy as np
from mtex import loadEBSD
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'python'))
from zebsdsolve import Solver, Error

class UpAng(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='zebsdsolve-up-ang-')
        self.root = Path(self.tmp.name)
    def tearDown(self): self.tmp.cleanup()
    def up(self, bits=8, version=1, rows=2, cols=3, count=None, extra=0, hex=0, dx=.025, dy=.05, partial=False, pat_w=10, pat_h=8):
        count = rows*cols if count is None else count
        offset = 16 if version == 1 else 58 if version == 4 else 42
        head = bytearray(offset)
        struct.pack_into('<4I', head, 0, version, pat_w, pat_h, offset)
        if version >= 3:
            struct.pack_into('<BIIBdd', head, 16, extra, cols, rows, hex, dx, dy)
        path = self.root/f'scan-v{version}-{bits}-{extra}-{hex}.up{1 if bits == 8 else 2}'
        patterns = np.arange(count*pat_w*pat_h, dtype='u1' if bits == 8 else '<u2').reshape(count,pat_h,pat_w)
        path.write_bytes(head + patterns.tobytes() + (b'\xff' if partial else b''))
        return path, patterns
    def ang(self, rows=2, cols=3, dx=.025, dy=.05, grid='SqrGrid', records=None, headers=None, units=None, extra_cols=0):
        path = self.root/'scan.ang'
        head = ['# VERSION 7', '# Phase 1', '# MaterialName Nickel', '# Symmetry 43',
                '# LatticeConstants 3.52 3.52 3.52 90 90 90', f'# GRID: {grid}',
                f'# XSTEP: {dx}', f'# YSTEP: {dy}', f'# NCOLS_ODD: {cols}',
                f'# NCOLS_EVEN: {cols}', f'# NROWS: {rows}']
        if headers is not None: head.append('# COLUMN_HEADERS: '+headers)
        if units is not None: head.append('# COLUMN_UNITS: '+units)
        if records is None:
            records = [[.1,.2,.3, 10+c*dx,20+r*dy, float(10*r+c), .9,1,123,0]+[5000+k for k in range(extra_cols)]
                       for r in range(rows) for c in range(cols)]
        path.write_text('\r\n'.join(head+[' '.join(str(v) for v in row) for row in records]))
        return path, np.asarray(records)
    def test_version_pixel_matrix_and_native_mtex_metadata_oracle(self):
        ang, records = self.ang(extra_cols=3)
        oracle = loadEBSD(str(ang), grid=False, setting=0)
        self.assertEqual(str(oracle.scanUnit), 'um')
        np.testing.assert_allclose(oracle.x, records[:,3])
        np.testing.assert_allclose(oracle.y, records[:,4])
        np.testing.assert_allclose(oracle.prop['iq'], records[:,5])
        for version in [1,3,4]:
            for bits in [8,16]:
                with self.subTest(version=version,bits=bits), Solver() as s:
                    up, patterns = self.up(bits,version)
                    s.open(up,ang)
                    self.assertEqual((s.scan['rows'],s.scan['cols']), (2,3))
                    self.assertEqual(s.scan['pattern_shape'], (8,10))
                    self.assertAlmostEqual(s.geometry['step_x_um'],.025)
                    self.assertAlmostEqual(s.geometry['step_y_um'],.05)
                    self.assertFalse(s.scan['has_pc'])
                    self.assertFalse(s.has_acquisition_calibration)
                    self.assertEqual(s.phases, [])
                    stats = s.index_preparation
                    self.assertEqual(stats['bytes'],24 if version == 1 else 50)
                    self.assertEqual(stats['reads'],2 if version == 1 else 3)
                    overview = s.stored_overview()
                    self.assertEqual(overview['kind'],2)
                    self.assertEqual(overview['label'],'Image quality')
                    np.testing.assert_array_equal(overview['values'], records[:,5].reshape(2,3))
                    self.assertEqual(s.index_preparation,stats)
                    for cell in [0,4,5]:
                        np.testing.assert_array_equal(s.pattern(cell,processed=False,full=True),patterns[cell])
                    self.assertGreater(s.index_preparation['bytes'],stats['bytes'])
                    self.assertGreater(s.index_preparation['reads'],stats['reads'])
    def test_standalone_raw_and_embedded_geometry(self):
        for version in [1,3,4]:
            with self.subTest(version=version), Solver() as s:
                up,_ = self.up(version=version)
                s.open(up)
                self.assertEqual((s.scan['rows'],s.scan['cols']), (1,6) if version == 1 else (2,3))
                self.assertIsNone(s.stored_overview())
        up,_ = self.up(version=3)
        with Solver() as s:
            s.configure(map_rows=1,map_cols=3)
            s.open(up)
            self.assertEqual((s.scan['rows'],s.scan['cols']), (1,3))
    def test_extra_flag_never_skips_a_pattern(self):
        ang,_ = self.ang()
        up,patterns = self.up(version=3,extra=1)
        with Solver() as s:
            s.open(up,ang)
            np.testing.assert_array_equal(s.pattern(0,processed=False,full=True),patterns[0])
        up,_ = self.up(version=3,extra=1,count=7)
        with Solver() as s:
            with self.assertRaisesRegex(Error,'AmbiguousUPExtraPatterns'):s.open(up,ang)
            s.open(up)
            self.assertEqual((s.scan['rows'],s.scan['cols']), (1,7))
    def test_bad_up_payload_grid_and_steps(self):
        ang,_ = self.ang()
        for options,error in [({'partial':True},'UPIncompletePattern'),({'version':3,'count':5},'UPPatternCountMismatch'),
                              ({'version':3,'hex':1},'UnsupportedUPHexGrid'),({'version':3,'dx':0},'InvalidUPStep'),
                              ({'version':3,'dx':float('nan')},'InvalidUPStep'),({'version':3,'dx':.1},'UPANGStepMismatch'),
                              ({'version':3,'rows':3,'cols':2},'ScanShapeMismatch')]:
            with self.subTest(options=options),Solver() as s:
                up,_ = self.up(**options)
                with self.assertRaisesRegex(Error,error):s.open(up,ang)
        up,_ = self.up(version=3,hex=1)
        with Solver() as s:
            s.open(up)
            self.assertEqual((s.scan['rows'],s.scan['cols']), (1,6))
    def test_duplicate_sidecar_and_hint_conflicts(self):
        ang,_ = self.ang();up,_ = self.up()
        with Solver() as s:
            with self.assertRaisesRegex(Error,'DuplicateANGSidecar'):s.open(up,ang,ang)
            s.open(up,ang)
        with Solver() as s:
            s.configure(map_rows=1,map_cols=6)
            with self.assertRaisesRegex(Error,'ScanShapeMismatch'):s.open(up,ang)
    def test_coordinate_registration_and_count(self):
        up,_ = self.up(); _,original = self.ang()
        mutations = [('reorder', original[[0,2,1,3,4,5]], 'ANGCoordinatesNotRowMajor'),
                     ('serpentine', original[[0,1,2,5,4,3]], 'ANGCoordinatesNotRowMajor'),
                     ('short',original[:5], 'ANGPatternCountMismatch'),
                     ('extra',np.concatenate([original,original[:1]]), 'ANGPatternCountMismatch')]
        shifted = original.copy();shifted[3:,3]+=.0125
        mutations.append(('stagger',shifted,'ANGCoordinatesNotRowMajor'))
        for name,records,error in mutations:
            with self.subTest(name=name),Solver() as s:
                ang,_ = self.ang(records=records)
                with self.assertRaisesRegex(Error,error):s.open(up,ang)
    def test_unsupported_headers_and_units(self):
        up,_=self.up()
        for options,error in [({'grid':'HexGrid'},'UnsupportedANGGrid'),({'dx':-1},'InvalidANGStep'),
                              ({'dx':0},'InvalidANGStep'),({'units':'rad rad rad nm nm au au au au au'},'UnsupportedANGUnits'),
                              ({'headers':'phi1 Phi phi2 x y CI CI Phase SEM Fit'},'UnsupportedANGColumns')]:
            with self.subTest(options=options),Solver() as s:
                ang,_=self.ang(**options)
                with self.assertRaisesRegex(Error,error):s.open(up,ang)
    def test_explicit_columns_units_and_missing_iq_values(self):
        up,_=self.up(); _,standard=self.ang()
        reordered=standard[:,[0,1,2,6,5,4,3,7,8,9]]
        reordered[2,4]=float('nan')
        names='phi1 Phi phi2 CI IQ y x Phase SEM Fit'
        ang,_=self.ang(records=reordered,headers=names,units='rad rad rad au au microns µm au au au')
        with Solver() as s:
            s.open(up,ang);quality=s.stored_overview()['values']
            expected=standard[:,5].copy();expected[2]=np.nan
            np.testing.assert_equal(quality,expected.reshape(2,3))
    def test_comma_columns_preserve_spaces_and_empty_unit_slots(self):
        up,_=self.up();_,standard=self.ang()
        records=standard[:,[0,1,2,7,3,4,5,6,8,9]]
        ang,_=self.ang(records=records,
            headers='phi1, Phi, phi2, Phase index, x, y, Image Quality, CI, SEM, Fit',
            units='radians, radians, radians, , microns, microns, , , , degrees')
        with Solver() as s:
            s.open(up,ang)
            np.testing.assert_array_equal(s.stored_overview()['values'],standard[:,5].reshape(2,3))

    def test_float32_positions_printed_to_five_decimals_remain_registered(self):
        for origin, step in [(0., .2), (1000., .025)]:
            with self.subTest(origin=origin, step=step):
                up,_ = self.up(rows=2, cols=1000)
                _, records = self.ang(rows=2, cols=1000, dx=step)
                positions = np.float32(origin) + np.arange(1000, dtype=np.float32)*np.float32(step)
                records[:,3] = np.tile(np.round(positions.astype(float),5),2)
                ang,_ = self.ang(rows=2, cols=1000, dx=step, records=records)
                oracle = loadEBSD(str(ang), grid=False, setting=0)
                np.testing.assert_array_equal(oracle.x, records[:,3])
                with Solver() as s:
                    s.open(up,ang)
                    np.testing.assert_array_equal(s.stored_overview()['values'],records[:,5].reshape(2,1000))
                bad = records.copy();bad[[642,643]]=bad[[643,642]]
                ang,_ = self.ang(rows=2, cols=1000, dx=step, records=bad)
                with Solver() as s:
                    with self.assertRaisesRegex(Error,'ANGCoordinatesNotRowMajor'):s.open(up,ang)
    def test_ambiguous_coordinate_precision_is_rejected(self):
        up,_ = self.up()
        _, records = self.ang()
        records[:,3] = np.float32(1e9) + (records[:,3]-10).astype('f4')
        ang,_ = self.ang(records=records)
        with Solver() as s:
            with self.assertRaisesRegex(Error,'ANGCoordinatePrecisionInsufficient'):s.open(up,ang)
        # Double precision positions with the same large origin remain identifiable.
        records[:,3] = 1e9 + np.tile(np.arange(3)*.025,2)
        ang,_ = self.ang(records=records)
        with Solver() as s:s.open(up,ang)

    def pc_ang(self, pc_lines, producer='edax'):
        ang,records=self.ang()
        text=ang.read_text()
        if producer == 'edax': text='# HEADER: Start\n'+text
        elif producer == 'zebsdsolve': pass # explicit marker below wins over inherited VERSION
        elif producer == 'legacy':
            text=text.replace('# VERSION 7\n','').replace('# MaterialName Nickel','# MaterialName')
            text=text.replace('# LatticeConstants', '# NumberFamilies 0\n# LatticeConstants')
            text=text.replace('# GRID:', ('# ElasticConstants 0 0 0 0 0 0\n'*6)+'# Categories\n# OPERATOR: unknown\n# SAMPLEID:\n# SCANID:\n# GRID:')
        elif producer == 'unknown': text=text.replace('# VERSION 7\n','')
        elif producer != 'overlap': raise AssertionError(producer)
        marker={'zebsdsolve':'# SOFTWARE: zebsdsolve 0.2.0','overlap':'# VERSION Overlap-GUI positive phase IDs'}.get(producer,'')
        # Keep all metadata before records; the scalar parser never reads Euler columns.
        lines=text.splitlines();first=next(i for i,line in enumerate(lines) if not line.startswith('#'))
        lines[first:first]=([marker] if marker else [])+pc_lines
        ang.write_text('\n'.join(lines)+'\n')
        return ang,records
    def test_ang_pc_producer_normalizations_use_stored_detector_shape(self):
        raw=np.array([.43,.27,.61])
        headers=[f'# {axis}-star: {value}' for axis,value in zip('xyz',raw)]
        for producer in ['edax','zebsdsolve','overlap','legacy']:
            for width,height in [(16,8),(8,16),(8,8)]:
                with self.subTest(producer=producer,width=width,height=height),Solver() as s:
                    up,_=self.up(pat_w=width,pat_h=height)
                    ang,records=self.pc_ang(headers,producer)
                    # pyMTEX exposes the original stars, but has no typed PC conversion.
                    oracle=loadEBSD(str(ang),grid=False,setting=0)
                    np.testing.assert_array_equal([oracle.meta['header'][axis+'_star'] for axis in 'xyz'],raw)
                    s.configure(binning=2)
                    s.set_geometry(sample_tilt_deg=-20,detector_tilt_deg=4)
                    s.open(up,ang)
                    expected=[raw[0],1-raw[1],raw[2]*min(width,height)/height] if producer=='edax' else [raw[0],1-raw[1]*width/height,raw[2]*width/height]
                    np.testing.assert_allclose(s.geometry['pc'],expected,atol=1e-14,rtol=0)
                    self.assertTrue(s.scan['has_pc'])
                    self.assertFalse(s.has_acquisition_calibration)
                    self.assertEqual(s.geometry['sample_tilt_deg'],-20)
                    self.assertEqual(s.geometry['detector_tilt_deg'],4)
                    self.assertEqual(s.phases,[])
                    np.testing.assert_array_equal(s.stored_overview()['values'],records[:,5].reshape(2,3))
                    self.assertTrue(any('Imported ANG pattern centre:' in event.get('text','') for event in s.events()))
    def test_ang_pc_explicit_producers_outrank_structural_hints(self):
        raw=[.43,.27,.61]
        pc=[f'# {axis}-star {value}' for axis,value in zip('xyz',raw)]
        cases=[('legacy',['# SOFTWARE: EDAX OIM Analysis'],'edax'),
               ('edax',['# SOFTWARE: zebsdsolve 0.2.0'],'width'),
               ('edax',['# SOFTWARE: EMSphInx'],'width'),
               ('legacy',['# SOFTWARE: OtherTool'],None),
               ('edax',['# SOFTWARE: OtherTool'],None),
               ('edax',['# SOFTWARE: EDAX OIM Analysis','# SOFTWARE: zebsdsolve 0.2.0'],None),
               ('unknown',['# HEADER: Start','# Version 1: header notes are not an exact version'],None)]
        for base,markers,convention in cases:
            with self.subTest(base=base,markers=markers),Solver() as s:
                up,_=self.up(pat_w=16,pat_h=8)
                ang,_=self.pc_ang(markers+pc,base)
                s.open(up,ang)
                self.assertEqual(s.scan['has_pc'],convention is not None)
                if convention:
                    expected=[.43,.73,.61] if convention=='edax' else [.43,.46,1.22]
                    np.testing.assert_allclose(s.geometry['pc'],expected,atol=1e-14)
                else:self.assertTrue(any('ANG pattern centre was not applied' in e.get('text','') for e in s.events()))
        # Square normalization is unambiguous even when explicit producers conflict.
        with Solver() as s:
            up,_=self.up(pat_w=8,pat_h=8)
            ang,_=self.pc_ang(['# SOFTWARE: EDAX OIM Analysis',*pc],'zebsdsolve')
            s.open(up,ang);self.assertTrue(s.scan['has_pc'])
            np.testing.assert_allclose(s.geometry['pc'],[.43,.73,.61],atol=1e-14)
            self.assertTrue(any('conflicting producer normalizations coincide' in e.get('text','') for e in s.events()))
    def test_legacy_re_pattern_centre_uses_full_stored_shape(self):
        up,_=self.up(pat_w=311,pat_h=256)
        ang,_=self.pc_ang(['# x-star .459861','# y-star 1.146244','# z-star .582882'],'legacy')
        with Solver() as s:
            s.configure(binning=4)
            s.open(up,ang)
            self.assertTrue(s.scan['has_pc'])
            np.testing.assert_allclose(s.geometry['pc'],[.459861,-.392507359375,.7081105546875],rtol=0,atol=1e-14)

    def test_ang_pc_missing_zero_ambiguous_and_square_policy(self):
        for headers in [[],['# x-star 0','# y-star 0','# z-star 0']]:
            with self.subTest(headers=headers),Solver() as s:
                up,_=self.up();ang,_=self.pc_ang(headers)
                s.set_geometry(pc=[.42,.22,.55]);s.open(up,ang)
                self.assertFalse(s.scan['has_pc'])
                self.assertEqual(s.geometry['pc'],[.42,.22,.55])
        headers=['# x-star .43','# y-star 1.27','# z-star .61']
        for size in [(16,8),(8,8)]:
            with self.subTest(size=size),Solver() as s:
                up,_=self.up(pat_w=size[0],pat_h=size[1]);ang,_=self.pc_ang(headers,'unknown')
                s.set_geometry(pc=[.42,.22,.55]);s.open(up,ang)
                notes=[e.get('text','') for e in s.events()]
                if size[0]!=size[1]:
                    self.assertFalse(s.scan['has_pc'])
                    self.assertEqual(s.geometry['pc'],[.42,.22,.55])
                    self.assertTrue(any('producer convention is unknown for a rectangular detector' in text for text in notes))
                else:
                    self.assertTrue(s.scan['has_pc'])
                    np.testing.assert_allclose(s.geometry['pc'],[.43,-.27,.61],atol=1e-14)
    def test_ang_invalid_pc_is_transactional_and_off_detector_pc_is_valid(self):
        up,_=self.up(pat_w=16,pat_h=12)
        good=['# x-star .43','# y-star .27','# z-star .61']
        ang,_=self.pc_ang(good)
        with Solver() as s:
            s.configure(static_background='none',dynamic_background='none',bandwidth=16,refine_iters=1)
            s.open(up,ang);s.add_phase(ROOT/'testdata/nickel/Ni_master_20kV.npz');s.index()
            frozen=s.export_snapshot();geometry=s.geometry;overview=s.stored_overview()['values'].copy()
            invalid=[(good[:2],'IncompleteANGPatternCentre'),(good+['# x-star .43'],'DuplicateANGPatternCentre'),
                     (['# x-star nan',*good[1:]],'InvalidANGPatternCentre'),
                     ([good[0],'# y-star inf',good[2]],'InvalidANGPatternCentre'),
                     ([*good[:2],'# z-star -1'],'InvalidANGPatternCentre'),
                     ([*good[:2],'# z-star 0'],'InvalidANGPatternCentre')]
            for headers,error in invalid:
                with self.subTest(headers=headers):
                    ang,_=self.pc_ang(headers)
                    with self.assertRaisesRegex(Error,error):self.configured_open(s,[up,ang],{'bandwidth':32})
                    self.assertEqual(s.export_snapshot(),frozen)
                    self.assertEqual(s.geometry,geometry)
                    np.testing.assert_array_equal(s.stored_overview()['values'],overview)
            s.set_geometry(pc=[.1,.2,.3])
            self.assertEqual(s.export_snapshot(),frozen) # imported PC is frozen for batch/export context
        ang,_=self.pc_ang(['# x-star -1.4','# y-star 2.3','# z-star .61'])
        with Solver() as s:
            s.open(up,ang)
            np.testing.assert_allclose(s.geometry['pc'],[-1.4,-1.3,.61],atol=1e-14)

    def configured_open(self, solver, paths, settings=None, hints=None):
        names='\n'.join(str(p) for p in paths).encode()
        patch=json.dumps(settings or {}).encode()
        read=json.dumps(hints or {}).encode()
        return solver._check(solver._lib.ze_open_scan_configured(solver._e,names,len(names),patch,len(patch),read,len(read)), 'open_scan_configured')
    def oxford_fixture(self):
        path=self.root/'scan.h5oina'
        with h5py.File(path,'w') as f:
            f.create_dataset('Manufacturer',data='Oxford Instruments')
            h=f.create_group('1/EBSD/Header')
            for key,value in [('X Cells',3),('Y Cells',2),('Pattern Height',8),('Pattern Width',10),('X Step',.025),('Y Step',.05)]:
                h.create_dataset(key,data=[value])
            f.create_dataset('1/EBSD/Data/Processed Patterns',data=np.zeros((6,8,10),dtype='u1'))
        return path
    def test_configured_open_failure_preserves_source_results_settings_and_auto_mask(self):
        up,_=self.up();ang,records=self.ang()
        with Solver() as s:
            s.configure(bandwidth=16,refine_iters=1,static_background='none',dynamic_background='none')
            s.open(up,ang)
            s.add_phase(ROOT/'testdata/nickel/Ni_master_20kV.npz')
            s.index()
            before=s.export_snapshot()
            settings=s.settings;geometry=s.geometry;scan=s.scan;overview=s.stored_overview()['values'].copy()
            stats=s.index_preparation
            patch={'bandwidth':32,'mask':False,'roi_x':2,'roi_w':1,'static_background':'average'}
            with self.assertRaisesRegex(Error,'ScanShapeMismatch'):
                self.configured_open(s,[up,ang],patch,{'map_rows':3,'map_cols':2})
            self.assertEqual(s.settings,settings);self.assertEqual(s.geometry,geometry);self.assertEqual(s.scan,scan)
            self.assertEqual(s.export_snapshot(),before)
            np.testing.assert_array_equal(s.stored_overview()['values'],overview)
            self.assertEqual(s.index_preparation,stats)
            # A bad settings patch must fail before even attempting a missing input.
            with self.assertRaisesRegex(Error,'BadSetting'):
                self.configured_open(s,[self.root/'missing.up1'],{'bandwidth':3})
            self.assertEqual(s.export_snapshot(),before);self.assertEqual(s.settings,settings)
            bad=records.copy();bad[[1,2]]=bad[[2,1]]
            ang,_=self.ang(records=bad)
            with self.assertRaisesRegex(Error,'ANGCoordinatesNotRowMajor'):
                self.configured_open(s,[up,ang],patch)
            self.assertEqual(s.export_snapshot(),before);self.assertEqual(s.settings,settings)
            # Ordinary open shares the staged path and must preserve the same live state.
            with self.assertRaisesRegex(Error,'ANGCoordinatesNotRowMajor'):s.open(up,ang)
            self.assertEqual(s.export_snapshot(),before);self.assertEqual(s.settings,settings)
            self.assertEqual(s.index_preparation,stats)
            # No failed full-settings restore may have made the automatic mask explicit.
            s.open(self.oxford_fixture())
            self.assertFalse(s.settings['mask'])
    def test_configured_open_keeps_existing_explicit_mask_on_failure(self):
        up,_=self.up();ang,_=self.ang()
        with Solver() as s:
            s.configure(mask=True)
            s.open(up,ang)
            with self.assertRaisesRegex(Error,'ScanShapeMismatch'):
                self.configured_open(s,[up,ang],{'mask':False},{'map_rows':3,'map_cols':2})
            s.open(self.oxford_fixture())
            self.assertTrue(s.settings['mask'])
    def test_configured_open_commits_read_hints_and_final_settings_in_order(self):
        up,_=self.up()
        with Solver() as s:
            self.configured_open(s,[up],{'bandwidth':32,'map_rows':1,'map_cols':6,'detector_width_mm':40},
                                 {'map_rows':2,'map_cols':3,'detector_width_mm':80})
            self.assertEqual((s.scan['rows'],s.scan['cols']),(2,3))
            self.assertEqual((s.settings['map_rows'],s.settings['map_cols']),(1,6))
            self.assertEqual(s.settings['bandwidth'],32)
            self.assertEqual(s.settings['detector_width_mm'],40)
            self.assertEqual(s.geometry['px_um'],8000)

    def test_streaming_line_boundaries_and_replacement_cleanup(self):
        rows,cols=101,103
        up,_=self.up(rows=rows,cols=cols)
        ang,records=self.ang(rows=rows,cols=cols)
        self.assertGreater(ang.stat().st_size,65536)
        with Solver() as s:
            s.open(up,ang)
            stats=s.index_preparation
            np.testing.assert_array_equal(s.stored_overview()['values'],records[:,5].reshape(rows,cols))
            self.assertEqual(s.index_preparation,stats)
            s.open(up)
            self.assertIsNone(s.stored_overview())
            s.set_patterns(np.zeros((1,1,8,10),dtype='u1'))
            self.assertIsNone(s.stored_overview())

if __name__=='__main__':unittest.main(verbosity=2)
