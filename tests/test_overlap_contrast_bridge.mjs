import assert from 'node:assert/strict';
import {Zebsdsolve,contrastBasisMetadata} from '../web/zebsdsolve.js';
const base={sse:new Float32Array([.2]),amplitudes:new Float32Array([0,.8,0]),phase:new Int32Array([0,0,-1]),max_components:3,objective_valid:[1],design_rank_reliable:[1],flat:[0],conditioned:[0]};
const theta=[0,.25,1,1,1,0,0];
const good=contrastBasisMetadata(base,[2,4,NaN],theta);
assert.deepEqual([...good.contrast_basis_weights.slice(0,2)],[0,1]);
assert.ok(Number.isNaN(good.contrast_basis_weights[2]));
assert.ok(contrastBasisMetadata(base,[NaN,4,NaN],theta).contrast_basis_weights.every(Number.isNaN));
assert.ok(contrastBasisMetadata({...base,flat:[1]},[2,4,NaN],theta).contrast_basis_weights.every(Number.isNaN));
assert.ok(contrastBasisMetadata({...base,design_rank_reliable:[0]},[2,4,NaN],theta).contrast_basis_weights.every(Number.isNaN));
const memory={buffer:new ArrayBuffer(1<<20)};let offset=16,calls=[];
const frame={alloc:n=>{const p=offset;offset+=(n+7)&~7;return p;},u64:values=>{const p=frame.alloc(values.length*8);new BigUint64Array(memory.buffer,p,values.length).set(values.map(BigInt));return p;},f32:values=>{const p=frame.alloc(values.length*4);new Float32Array(memory.buffer,p,values.length).set(values);return p;}};
const x={ze_create:()=>1,ze_overlap_optimizer_version:()=>2,ze_seed_source_roi:()=>0,
  ze_overlap_cells_v4(e,c,n,k,primary,optimize,cp,cl,sp,sl,mp,ml,tp,tl,np,nl){
    calls.push(['cells_v4',optimize,nl,tl]);assert.equal(n,1);assert.equal(k,2);assert.equal(nl,2);
    const records=new DataView(memory.buffer,cp,k*32);
    for(let j=0;j<k;j++){records.setFloat32(j*32,1,true);records.setFloat32(j*32+16,j===0?.6:.8,true);records.setInt32(j*32+20,0,true);}
    new Float32Array(memory.buffer,sp,16).set([.2,1,0,0,.6,.8,0,0,.8,.7,1,1,.1,.9,.7,.2]);
    const progress=new DataView(memory.buffer,mp,32);progress.setUint32(0,2,true);progress.setUint32(4,2,true);progress.setUint32(8,2,true);progress.setUint32(12,1,true);
    new Float32Array(memory.buffer,tp,7).set(theta);new Float32Array(memory.buffer,np,2).set([2,4]);return 0;
  },
  ze_overlap_phase_moments(e,p,n){new Float64Array(memory.buffer,p,n).set([2,3,4,5,6]);return 0;},
  ze_overlap_inspect_v4(){calls.push(['inspect_v4']);return 0;},
  ze_overlap_inspect_v3(){calls.push(['inspect_v3']);return 0;}};
const solver=new Zebsdsolve(x,{memory});solver._serial=task=>task();solver._frame=task=>task(frame);solver._check=rc=>rc;solver.shape=()=>[2,2];
Object.defineProperties(solver,{settings:{get:()=>({binning:1,full_resolution:'none',master_smooth:0,clahe:false,bandpass:false})},geometry:{get:()=>({})},phases:{get:()=>[{}]}});
const result=await solver.overlapCells([0],{maxComponents:2,fitBlurGain:true});
assert.equal(result.fit_model,'contrast_gain_v1');assert.equal(result.gain_domain,'contrast');assert.equal(result.blur_gain_optimizer.version,2);
assert.deepEqual(calls[0],['cells_v4',1,2,1]);assert.deepEqual([...result.contrast_basis_weights].map(x=>Math.round(x*10)),[6,4]);
assert.equal(result.contrast_basis_preparation.source_dc_restored,false);assert.equal(result.contrast_basis_preparation.phase_moments[0].length,4);assert.ok(!('phase_dc' in result.blur_gain_optimizer));
await solver.overlapInspect(0,{...result,blur_gain:false});assert.deepEqual(calls.at(-1),['inspect_v4']);
await solver.overlapInspect(0,{...result,fit_model:'source_dc_gain_v1'});assert.deepEqual(calls.at(-1),['inspect_v3']);
const legacy={...result};delete legacy.fit_model;await solver.overlapInspect(0,legacy);assert.deepEqual(calls.at(-1),['inspect_v3']);
await assert.rejects(()=>solver.overlapInspect(0,{...result,fit_model:'future_model'}),/unsupported overlap fit_model/);
const old=new Zebsdsolve({ze_create:()=>1},{memory});assert.throws(()=>old.overlapCells([0]),/V4 overlap support/);
console.log('V4 contrast bridge: invalid weights fail closed, explicit V4 dispatch and norm lengths, declared basis provenance, identity V4 and legacy frozen inspection passed');
