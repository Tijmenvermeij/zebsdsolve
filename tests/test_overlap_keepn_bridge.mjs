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
Object.defineProperties(solver,{settings:{get:()=>({binning:1,full_resolution:'none',master_smooth:0,clahe:false,bandpass:false,method:'dictionary'})},geometry:{get:()=>({})},phases:{get:()=>[{}]}});

x.ze_overlap_cells_v5=function(e,c,n,k,primary,optimize,keep,cp,cl,sp,sl,mp,ml,tp,tl,np,nl,wp,wl){
  calls.push(['cells_v5',keep,wl]);assert.equal(keep,5);assert.equal(wl,n*k);
  const result=x.ze_overlap_cells_v4(e,c,n,k,primary,optimize,cp,cl,sp,sl,mp,ml,tp,tl,np,nl);
  new Int32Array(memory.buffer,wp,n*k).set([0,4]);return result;
};
const original=await solver.overlapCells([0]);
assert.deepEqual(calls[0],['cells_v4',0,2,1]);
assert.deepEqual([...original.winner_rank],[0,1]);assert.equal(original.keep_n,1);
calls=[];
const pooled=await solver.overlapCells([0],{keepN:5,fitBlurGain:true});
assert.deepEqual(calls[0],['cells_v5',5,2]);assert.deepEqual([...pooled.winner_rank],[0,4]);
assert.equal(pooled.keep_n,5);assert.equal(pooled.candidate_method,'dictionary');
assert.equal(pooled.candidate_seed_policy,'dictionary_raw_top_n_v1');
for(const keepN of [0,9,1.5,true])assert.throws(()=>solver.overlapCells([0],{keepN}),/keepN/);
assert.throws(()=>solver.overlapCells([0],{keepN:2,gainDomain:'source_dc'}),/contrast/);
delete x.ze_overlap_cells_v5;
assert.throws(()=>solver.overlapCells([0],{keepN:5}),/WASM V5/);
console.log('V5 keep-N bridge: additive dispatch, separate rank, policy metadata, parameter and old-runtime guards passed');
