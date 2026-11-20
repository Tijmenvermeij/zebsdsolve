import assert from 'node:assert/strict';
import {decodeOverlapProfileEvents,Zebsdsolve} from '../web/zebsdsolve.js';
const bytes=new Uint8Array(128),v=new DataView(bytes.buffer);
for(let i=0;i<2;i++){
 const o=i*64;v.setUint32(o,i,true);v.setUint32(o+4,i,true);v.setUint32(o+8,2,true);
 v.setUint32(o+12,1,true);v.setInt32(o+16,1,true);v.setInt32(o+20,3,true);
 v.setUint32(o+24,i?163:3,true);v.setUint32(o+28,1,true);v.setFloat32(o+32,1,true);
 v.setFloat32(o+48,.125,true);v.setFloat32(o+52,.8,true);v.setFloat32(o+56,.7,true);v.setFloat32(o+60,.6,true);
}
const events=decodeOverlapProfileEvents(bytes,[500,900]);
assert.deepEqual(events.map(e=>e.cell),[500,900]);assert.equal(events[1].branch,1);
assert.equal(events[1].rank,3);assert.equal(events[1].residual_ncc,.125);
assert.deepEqual(events[0].source_q,[1,0,0,0]);
assert.throws(()=>decodeOverlapProfileEvents(bytes.subarray(1),[1,2]),/complete/);
v.setUint32(0,2,true);assert.throws(()=>decodeOverlapProfileEvents(bytes,[1,2]),/invalid/);v.setUint32(0,0,true);
v.setUint32(24,256,true);assert.throws(()=>decodeOverlapProfileEvents(bytes,[1,2]),/invalid/);
assert.equal(decodeOverlapProfileEvents(bytes,[1,2],2)[0].flags,256);
const ze=Object.create(Zebsdsolve.prototype);ze.x={};
for(const polish of [1,'yes',null])assert.throws(()=>ze.overlapCells([0],{polish}),/boolean/);
for(const minResidualNcc of [NaN,Infinity,2,-2,true,'0.1'])assert.throws(()=>ze.overlapCells([0],{minResidualNcc}),/minResidualNcc/);
for(const polishRadiusDeg of [0,-1,6,NaN,true])assert.throws(()=>ze.overlapCells([0],{polishRadiusDeg}),/polishRadiusDeg/);
assert.throws(()=>ze.overlapCells([0],{polish:true}),/V6/);
assert.throws(()=>ze.overlapCells([0],{minResidualNcc:0}),/V6/);
assert.throws(()=>ze.overlapCells([0],{polish:true,gainDomain:'source_dc'}),/contrast/);
console.log('V6 trace decoding, scan-ID mapping, finite options and capability guards passed');
const memory={buffer:new ArrayBuffer(1<<20)};let offset=16,called=0;
const frame={alloc:n=>{const p=offset;offset+=(n+7)&~7;return p;},u64:values=>{const p=frame.alloc(values.length*8);new BigUint64Array(memory.buffer,p,values.length).set(values.map(BigInt));return p;}};
const x={ze_create:()=>1,ze_overlap_profile_version:()=>1,ze_overlap_profile_selection_version:()=>2,ze_overlap_profile_trace_capacity:()=>4,ze_overlap_optimizer_version:()=>2,ze_seed_source_roi:()=>0,ze_overlap_cells_v4:()=>{throw Error('V4 used for enabled V6');},
 ze_overlap_cells_v6(e,c,n,k,primary,optimize,keep,cp,cl,sp,sl,mp,ml,tp,tl,np,nl,wp,wl,op,qp,ql,ep,capacity,countp){
  called++;assert.equal(n,1);assert.equal(k,2);assert.equal(ql,2);assert.equal(capacity,4);
  const ov=new DataView(memory.buffer,op,16);assert.equal(ov.getUint32(0,true),16);assert.equal(ov.getUint32(4,true),7);assert.equal(ov.getFloat32(12,true),1);
  const rec=new DataView(memory.buffer,cp,64);rec.setFloat32(0,1,true);rec.setFloat32(16,.6,true);rec.setInt32(20,0,true);rec.setInt32(52,-1,true);
  new Float32Array(memory.buffer,sp,16).set([.64,1,0,0,.6,0,0,0,.64,0,1,1,.64,.64,.64,.64]);
  const pv=new DataView(memory.buffer,mp,32);pv.setUint32(0,1,true);pv.setUint32(8,1,true);pv.setUint32(12,11,true);
  new Float32Array(memory.buffer,tp,7).set([0,0,1,1,1,0,0]);new Float32Array(memory.buffer,np,2).set([2,NaN]);
  new Float32Array(memory.buffer,qp,8).set([1,0,0,0,0,0,0,0]);new Int32Array(memory.buffer,wp,2).set([0,-1]);
  const ev=new DataView(memory.buffer,ep,64);ev.setUint32(8,2,true);ev.setInt32(16,0,true);ev.setInt32(20,1,true);ev.setUint32(24,128|256|512|1|4,true);ev.setUint32(28,1,true);ev.setFloat32(32,1,true);ev.setFloat32(48,.1,true);
  new DataView(memory.buffer,countp,4).setUint32(0,1,true);return 0;
 },ze_overlap_phase_moments(e,p,n){new Float64Array(memory.buffer,p,n).set([2,3,4,5,6]);return 0;}};
const legacyV6=x.ze_overlap_cells_v6;
const solver=new Zebsdsolve(x,{memory});solver._serial=task=>task();solver._frame=task=>task(frame);solver._check=rc=>{assert.equal(rc,0);};
Object.defineProperties(solver,{settings:{get:()=>({method:'spherical',binning:1,full_resolution:'none',master_smooth:0,clahe:false,bandpass:false})},geometry:{get:()=>({})},phases:{get:()=>[{}]}});
const result=await solver.overlapCells([123],{polish:true,minResidualNcc:1});
assert.equal(called,1);assert.equal(result.profile_version,1);assert.equal(result.stop_reason_name[0],'residual_screen');
assert.equal(result.profile_events[0].cell,123);assert.equal(result.source_quaternions[0],1);assert.equal(result.residual_screen.confidence_calibrated,false);
assert.equal(result.selection_version,2);assert.equal(result.trajectory_selection,'profiled_achieved_supports_v2');
assert.equal(result.profile_events[0].flags&768,768);assert.deepEqual([...result.winner_rank],[0,-1]);
console.log('V6 exact call layout, independent controls, source poses, trace, stops and frozen metadata passed');

assert.throws(()=>solver.overlapCells([123],{minStageImprovement:[.01,null,null]}),/requires residual NCC/);
assert.throws(()=>solver.overlapCells([123],{maxComponents:3,minResidualNcc:.1,minStageImprovement:[.01,null,null]}),/each requested addition/);
assert.throws(()=>solver.overlapCells([123],{minResidualNcc:.1,minStageImprovement:[.01,null,null]}),/current WASM/);
x.ze_overlap_stage_gate_version=()=>1;
x.ze_overlap_cells_v6=(...args)=>{
  const op=args[19], ev=args[22];
  const options=new DataView(memory.buffer,op,28);
  assert.equal(options.getUint32(0,true),28);
  assert.equal(options.getFloat32(16,true),Math.fround(.01));
  assert.equal(options.getFloat32(20,true),0);
  options.setUint32(0,16,true);
  const rc=legacyV6(...args);
  new DataView(memory.buffer,ev,64).setUint32(24,128|1024,true);
  return rc;
};
const gated=await solver.overlapCells([123],{polish:true,minResidualNcc:1,minStageImprovement:[.01,null,null]});
assert.deepEqual(gated.stage_gate.minimum,[Math.fround(.01),null,null]);
assert.equal(gated.profile_events[0].flags&1024,1024);
console.log('Optional stage limits, extended wire options and stage diagnostics passed');
