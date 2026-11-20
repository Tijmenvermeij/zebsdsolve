"""Independent whole-payload reference protection and versioned provenance."""
from pathlib import Path
import json
import os
import re
import subprocess
import sys
import tempfile
import unittest
import numpy as np
import test_overlap as fixtures
from zebsdsolve import Error

ROOT=Path(__file__).resolve().parents[1]
OLD=Path('/Users/tvermeij/EMsoftData/overlap_keepn_production_20261117/native/libzebsdsolve_268330f84d0c.dylib')


def compare_histories(new,old):
    counts={-1:0,0:0,1:0};fields=0
    # Prefix invariance belongs to each individual trajectory, not to final
    # best-of-two results whose selected history may change with requested K.
    for primary in ('search','stored'):
        for keep in (1,3):
            for field in ('quaternions','phase','winner_rank'):
                np.testing.assert_array_equal(old[f'{primary}_K4_N{keep}_{field}'][:,:3],old[f'{primary}_K3_N{keep}_{field}'],err_msg='individual-history prefix '+field)
    for primary in ('search','stored'):
        for k in (3,4):
            reference=f'{primary}_K{k}_N1';pooled=f'{primary}_K{k}_N3'
            def valid(data,prefix):
                return (data[prefix+'_retained_order']>0)&data[prefix+'_objective_valid']&np.isfinite(data[prefix+'_sse'])&(data[prefix+'_sse']>=0)
            choice=np.where(valid(old,pooled)&(~valid(new,reference)|(old[pooled+'_sse']<new[reference+'_sse'])),1,np.where(valid(new,reference),0,-1))
            np.testing.assert_array_equal(new[pooled+'_trajectory_source'],choice)
            for source in choice:counts[int(source)]+=1
            for key in new.files:
                if key.startswith(reference+'_'):
                    np.testing.assert_array_equal(new[key],old[key],err_msg='N1 parity '+key)
                if key.startswith(pooled+'_'):
                    suffix=key[len(pooled):]
                    if suffix.startswith('_inspect_'):
                        expected=old[key] if choice[3]==1 else new[reference+suffix]
                    else:
                        expected=np.stack([old[key][i] if source==1 else new[reference+suffix][i] for i,source in enumerate(choice)])
                    np.testing.assert_array_equal(new[key],expected,err_msg='whole selected payload '+key)
                    fields+=1
    return counts,fields


class IndependentReferenceHistories(unittest.TestCase):
    def test_selected_payload_equals_independent_greedy_or_rank_one(self):
        if not OLD.exists():self.skipTest('frozen independent original V5 library required')
        requested=os.environ.get('ZEBSDSOLVE_REFERENCE_ARTIFACTS')
        with tempfile.TemporaryDirectory() as temporary:
            output=Path(requested or temporary);output.mkdir(parents=True,exist_ok=True)
            summary={}
            for method in ('dictionary','spherical'):
                for mode in ('old','new'):
                    env=os.environ.copy()
                    if mode=='old':env['ZEBSDSOLVE_LIB']=str(OLD)
                    run=subprocess.run([sys.executable,str(ROOT/'tests/overlap_reference_capture.py'),str(output/f'{mode}_{method}.npz'),method],env=env,capture_output=True,text=True,timeout=180)
                    (output/f'{mode}_{method}.log').write_text(run.stdout+run.stderr)
                    self.assertEqual(run.returncode,0,run.stdout+run.stderr)
                counts,fields=compare_histories(np.load(output/f'new_{method}.npz'),np.load(output/f'old_{method}.npz'))
                for mode in ('old','new'):
                    info=json.loads((output/f'{mode}_{method}.json').read_text())
                    for profile in info['profiles'].values():self.assertEqual(profile['prep_load_dispatches'],1)
                summary[method]={'selected_source_counts':counts,'whole_payload_array_checks':fields}
            self.assertGreater(sum(case['selected_source_counts'][1] for case in summary.values()),0)
            self.assertGreater(sum(case['selected_source_counts'][0] for case in summary.values()),0)
            self.assertGreater(sum(case['selected_source_counts'][-1] for case in summary.values()),0)
            (output/'independent_payload_summary.json').write_text(json.dumps(summary,indent=2))


class TrajectoryProtocol(unittest.TestCase):
    def setUp(self):
        self.fixture=fixtures.NativeDiscovery();self.fixture.setUp();self.addCleanup(self.fixture.doCleanups)
        self.s=self.fixture.s;self.s.configure(method='dictionary')

    def fake_runtime(self,version,pad):
        native=self.s._lib;calls={'fit':0,'version':0}
        def fake(*args):
            calls['fit']+=1;n=int(args[2]);k=int(args[3])
            out,states,progress,theta,norms,ranks=args[7],args[9],args[11],args[13],args[15],args[17]
            for b in range(n):
                states[b].initial_sse=1;states[b].sse=.25;states[b].condition=1;states[b].best_single_sse=.5
                states[b].single_sse[:]=[.5]*4;states[b].amplitude[:]=[.5]*4
                progress[b].retained_order=k;progress[b].proposed_count=k;progress[b].active_count=k;progress[b].stop_reason=1;progress[b]._pad=pad
                for j,value in enumerate([0,0,1,1,1,0,0]):theta[b*7+j]=value
                for j in range(k):
                    out[b*k+j].q[:]=[1,0,0,0];out[b*k+j].phase=0;out[b*k+j].amplitude=.5
                    norms[b*k+j]=1;ranks[b*k+j]=0 if j==0 else 1
            return 0
        class Runtime:
            def __getattr__(self,name):
                if name=='ze_overlap_trajectory_version':
                    if version is None:raise AttributeError(name)
                    def getter():calls['version']+=1;return version
                    return getter
                if name=='ze_overlap_cells_v5':return fake
                return getattr(native,name)
        return Runtime(),calls

    def test_old_phase_gate_is_not_decoded_as_provenance(self):
        native=self.s._lib
        try:
            for pad in (1,0xffffffff):
                self.s._lib,calls=self.fake_runtime(None,pad)
                result=self.s.overlap_cells([1],max_components=3,keep_n=3)
                np.testing.assert_array_equal(result['trajectory_source'],[1])
                self.assertEqual(result['trajectory_selection'],'greedy_pooled_v5')
                self.assertEqual(result['candidate_seed_policy'],'dictionary_raw_top_n_v1')
                self.s._lib=native
            self.s._lib,calls=self.fake_runtime(1,0)
            result=self.s.overlap_cells([1],max_components=3,keep_n=3)
            np.testing.assert_array_equal(result['trajectory_source'],[0])
            self.assertEqual(result['trajectory_selection'],'best_valid_final_sse_of_pooled_and_rank1_v1')
        finally:self.s._lib=native

    def test_unknown_version_fails_before_fit_and_sole_does_not_decode(self):
        native=self.s._lib
        try:
            self.s._lib,calls=self.fake_runtime(99,1)
            with self.assertRaisesRegex(Error,'unsupported overlap trajectory version 99'):
                self.s.overlap_cells([1],max_components=4,keep_n=3)
            self.assertEqual(calls['fit'],0)
            result=self.s.overlap_cells([1],max_components=1,keep_n=3)
            self.assertEqual(result['trajectory_selection'],'ordinary_primary_only_v1')
            np.testing.assert_array_equal(result['trajectory_source'],[0])
            self.assertEqual(calls['version'],1)
        finally:self.s._lib=native

    def test_unavailable_and_flat_rows_are_explicitly_tagged(self):
        missing=self.s.overlap_cells([0],max_components=4,primary='stored',keep_n=3)
        flat=self.s.overlap_cells([3],max_components=4,keep_n=3)
        for result in (missing,flat):
            np.testing.assert_array_equal(result['trajectory_source'],[-1])
            self.assertEqual(result['trajectory_selection'],'best_valid_final_sse_of_pooled_and_rank1_v1')

    def test_cancel_protected_operations_preserves_stored_and_recovers(self):
        import threading
        import time
        self.s.index()
        saved={key:value.copy() for key,value in self.s.results().items() if isinstance(value,np.ndarray)}
        for k in (3,4):
            started=threading.Event();outcome=[]
            def run():
                started.set()
                try:outcome.append(self.s.overlap_cells([1,2],max_components=k,primary='stored',keep_n=3,blur_gain=True))
                except Error as error:outcome.append(error)
            worker=threading.Thread(target=run);worker.start();started.wait(1)
            deadline=time.monotonic()+10
            while worker.is_alive() and time.monotonic()<deadline:
                self.s.cancel();time.sleep(.002)
            worker.join(1)
            self.assertFalse(worker.is_alive())
            self.assertEqual(len(outcome),1);self.assertIsInstance(outcome[0],Error)
            self.assertIn('Cancelled',str(outcome[0]))
            for field,value in saved.items():np.testing.assert_array_equal(self.s.results()[field],value)
            retry=self.s.overlap_cells([1],max_components=k,primary='stored',keep_n=3)
            self.assertGreater(int(retry['retained_order'][0]),0)

    def test_preprocessing_and_raw_upload_occur_once(self):
        self.s.configure(dynamic_background='subtract');self.s.index()
        if not self.s.profile(True):self.skipTest('native pipeline profiling unavailable')
        try:
            self.s.profile_report()
            result=self.s.overlap_cells([1,2],max_components=4,primary='stored',keep_n=3)
            report=self.s.profile_report()
        finally:self.s.profile(False)
        for pipeline in ('prep/load','prep/blur_rows','prep/blur_cols','prep/combine'):
            match=re.search(r'^\s*[\d.]+\s+ms\s+(\d+)\s+'+re.escape(pipeline)+r'\s*$',report,re.M)
            self.assertIsNotNone(match,report)
            self.assertEqual(int(match[1]),1,report)
        self.assertEqual(result['trajectory_selection'],'best_valid_final_sse_of_pooled_and_rank1_v1')


if __name__=='__main__':unittest.main()
