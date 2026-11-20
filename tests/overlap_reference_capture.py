"""Subprocess evidence: independent legacy pooled and current rank-one histories."""
from pathlib import Path
import ctypes as C
import json
import os
import re
import sys
import numpy as np
import test_overlap as fixtures

output = Path(sys.argv[1]); method = sys.argv[2]
f = fixtures.NativeDiscovery(); f.setUp()
try:
    s = f.s
    s.clear_phases(); s.add_phase(fixtures.ROOT/'testdata/nickel/Ni_master_20kV.npz')
    s.configure(method=method, tile_mb=8, mask=True, mask_radius=.9)
    s.set_geometry(step_x_um=0, step_y_um=0)
    q = np.random.default_rng(714).normal(size=(4,4)); q = (q/np.linalg.norm(q,axis=1,keepdims=True)).astype(np.float32)
    base = fixtures.unit(s.simulate(q,cells=[0]*4))
    patterns = np.zeros((6,48,48),np.float32)
    patterns[1] = base[0]
    patterns[2] = np.einsum('i,ihw->hw',[.7,.5,.3],base[:3])
    patterns[3] = np.einsum('i,ihw->hw',[.7,.5,.3,.2],base)
    s.set_patterns(patterns[None]); s.index()
    saved = {key:value.copy() for key,value in s.results().items() if isinstance(value,np.ndarray)}
    arrays={}; profiles={}
    for primary in ('search','stored'):
        for k in (3,4):
            for keep in (1,3):
                if os.environ.get('ZEBSDSOLVE_REFERENCE_FAST') and (primary,k,keep)!=('stored',4,3):continue
                name=f'{primary}_K{k}_N{keep}'
                tracked = 'ze_overlap_cells_v4' if keep == 1 else 'ze_overlap_cells_v5'
                native = getattr(s._lib,tracked)
                raw={}
                def call(*args):
                    rc=native(*args)
                    if rc==0:
                        n=int(args[2]); shift=int(keep>1)
                        for key,index,width in (('_raw_components',6+shift,k*32),('_raw_states',8+shift,64),('_raw_progress',10+shift,32)):
                            raw[key]=np.frombuffer(C.string_at(C.addressof(args[index]),n*width),np.uint32).copy().reshape(n,-1)
                        # Versioned final-host provenance replaces an ephemeral
                        # GPU phase gate; every other raw byte must match.
                        if keep==1:raw['_raw_phase_gate']=raw['_raw_progress'][:,7].copy()
                        raw['_raw_progress'][:,7]=0
                    return rc
                profile=primary=='stored' and k==4 and keep==3 and s.profile(True)
                if profile:s.profile_report()
                setattr(s._lib,tracked,call)
                try:r=s.overlap_cells([0,1,2,3,3,3,3,3,3],max_components=k,primary=primary,keep_n=keep,blur_gain=True)
                finally:setattr(s._lib,tracked,native)
                if profile:
                    report=s.profile_report();s.profile(False)
                    load=re.search(r'^\s*[\d.]+\s+ms\s+(\d+)\s+prep/load\s*$',report,re.M)
                    profiles[name]={'prep_load_dispatches':int(load[1]) if load else None,'report':report}
                for key,value in r.items():
                    if isinstance(value,np.ndarray):arrays[name+'_'+key]=value
                for key,value in raw.items():arrays[name+key]=value
                images=s.overlap_inspect(3,r,row=3,selected_component=0)
                for key,value in images.items():arrays[name+'_inspect_'+key]=value
                print(name,r['sse'].tolist(),r['trajectory_source'].tolist(),flush=True)
    for key,value in saved.items():np.testing.assert_array_equal(s.results()[key],value)
    np.savez_compressed(output,**arrays)
    output.with_suffix('.json').write_text(json.dumps({'profiles':profiles,'stored_primary_unchanged':True,'method':method,'duplicate_cells':[0,1,2,3,3,3,3,3,3]},indent=2))
finally:f.doCleanups()
