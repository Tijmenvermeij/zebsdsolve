import assert from 'node:assert/strict';
import {decodeOverlapArrays,overlapStopReasons,overlapBlurGainMetadata} from '../web/zebsdsolve.js';
const n=3,k=4,records=new Uint8Array(n*k*32),r=new DataView(records.buffer),state=new Float32Array(n*16),progress=new Uint8Array(n*32),p=new DataView(progress.buffer);
for(let b=0;b<n;b++)for(let j=0;j<k;j++){r.setInt32((b*k+j)*32+20,-1,true);state[b*16+8+j]=1;}
for(let j=0;j<3;j++){r.setFloat32(j*32,1,true);r.setInt32(j*32+20,0,true);r.setFloat32(j*32+16,j===1?0:.5,true);}
state[0]=.1;state[1]=1;state[13]=.3;state[14]=.5;
p.setUint32(0,3,true);p.setUint32(4,4,true);p.setUint32(8,2,true);p.setUint32(12,5,true);p.setFloat32(16,.2,true);p.setFloat32(20,.1,true);
state[16]=NaN;state[17]=1;state[29]=NaN;p.setUint32(32+12,6,true);
state[32]=0;state[33]=0;state[35]=2;state[45]=NaN;p.setUint32(64+12,7,true);
const out=decodeOverlapArrays(records,state,progress,[4,7,9],k);
assert.deepEqual([...out.discovery_id.slice(0,4)],[0,1,2,-1]);assert.deepEqual([...out.amplitudes.slice(0,4)],[.5,0,.5,0]);
assert.equal(out.active_count[0],2);assert.equal(out.retained_order[0],3);assert.equal(out.proposed_count[0],4);assert.equal(out.stop_reason_name[0],'boundary');
assert.equal(out.prepared_fractions[1],0);assert.equal(out.objective_valid[1],0);assert.equal(out.solve_status[1],'missing_primary');
assert.equal(out.conditioning_assessed[2],0);assert.ok(Number.isNaN(out.relative_gram_condition[2]));
assert.equal(overlapStopReasons[6],'missing_primary');assert.equal(overlapStopReasons[10],'stale_primary');console.log('overlap bridge decoding: stable IDs, zero inactive component, missing/flat statuses passed');

const scalarScores=['sse','initial_sse','residual_norm','mixture_ncc','best_single_sse','improvement_over_best_single','component_correlation','relative_gram_condition','previous_sse','incremental_improvement'];
for(const stop of [6,10]) {
  p.setUint32(32+12,stop,true);
  const mixed=decodeOverlapArrays(records,state,progress,[4,7,9],k);
  const single=decodeOverlapArrays(records.slice(k*32,2*k*32),state.slice(16,32),progress.slice(32,64),[7],k);
  for(const [result,index]of [[single,0],[mixed,1]]){
    for(const key of scalarScores)assert.ok(Number.isNaN(result[key][index]),key);
    for(const key of ['prepared_fractions','single_sse','proposal_ncc','proposal_nmi'])assert.ok([...result[key].slice(index*k,(index+1)*k)].every(Number.isNaN),key);
    for(const key of ['objective_valid','conditioning_assessed','design_rank_reliable','residual_flat'])assert.equal(result[key][index],0,key);
    for(const key of ['candidate_count','fitted_count','retained_order','proposed_count','active_count'])assert.equal(result[key][index],0,key);
  }
}
p.setUint32(12,4,true);state[3]=16;
const retained=decodeOverlapArrays(records,state,progress,[4,7,9],k);
assert.equal(retained.objective_valid[0],1);assert.equal(retained.addition_unresolved[0],1);assert.ok(Math.abs(retained.sse[0]-.1)<1e-6);
console.log('overlap unavailable diagnostics: single/mixed missing and stale rows agree; valid retained unresolved addition keeps SSE');

for(const stop of [3,5]) {
  p.setUint32(32+12,stop,true);state[16]=1;state[17]=1;state[29]=1;
  const result=decodeOverlapArrays(records,state,progress,[4,7,9],k);
  assert.equal(result.objective_valid[1],0);assert.ok(Number.isNaN(result.sse[1]));
  assert.equal(result.solve_status[1],overlapStopReasons[stop]);
}
p.setUint32(12,5,true);state[3]=0;
assert.equal(decodeOverlapArrays(records,state,progress,[4,7,9],k).solve_status[0],'solved');
console.log('overlap zero-order unavailable status follows stop reason; valid retained boundary stays solved');

const theta=new Float32Array([1.2,.2,1,.8,1.2,.05,-.03]);
const nuisance=overlapBlurGainMetadata(true,theta);
assert.equal(nuisance.model,'shared_blur_gain_prepared_projection_nnls');
assert.equal(nuisance.blur_gain,true);assert.deepEqual([...nuisance.blur_gain_theta],[...theta]);
theta[0]=5;assert.notEqual(nuisance.blur_gain_theta[0],5);
assert.equal(nuisance.blur_gain_optimizer.max_evaluations,92);
assert.equal(nuisance.blur_gain_optimizer.best_single,'independently_optimized_per_discovered_pose');
assert.equal(nuisance.blur_gain_parameter_names.length,7);
assert.deepEqual(overlapBlurGainMetadata(false,null),{blur_gain:false});
assert.equal(nuisance.blur_gain_optimizer.version,1);
assert.equal(nuisance.blur_gain_optimizer.singleton_max_evaluations,88);
assert.ok(!('log_rho_polish_steps' in nuisance.blur_gain_optimizer));
assert.ok(!('joint_singleton_seeds' in nuisance.blur_gain_optimizer));
for(const version of [1,2]) {
  const metadata=overlapBlurGainMetadata(true,theta,{ze_overlap_optimizer_version:()=>version}).blur_gain_optimizer;
  assert.equal(metadata.version,version);
  assert.equal(metadata.max_evaluations,version===2?104:92);
  assert.equal(metadata.singleton_max_evaluations,version===2?96:88);
  assert.equal('log_rho_polish_steps' in metadata,version===2);
  assert.equal('joint_singleton_seeds' in metadata,version===2);
}
assert.throws(()=>overlapBlurGainMetadata(true,theta,{ze_overlap_optimizer_version:()=>3}),/unsupported overlap optimizer metadata version 3/);
console.log('overlap blur/gain metadata: explicit opt-in, copied theta, bounds and optimizer interpretation passed');

const noCompositeRecords=records.slice(0,k*32),noCompositeView=new DataView(noCompositeRecords.buffer);
for(let j=0;j<k;j++)noCompositeView.setFloat32(j*32+16,0,true);
const noCompositeState=state.slice(0,16);noCompositeState[0]=.9999998;noCompositeState[1]=1;
const noComposite=decodeOverlapArrays(noCompositeRecords,noCompositeState,progress.slice(0,32),[4],k);
assert.ok(Number.isNaN(noComposite.mixture_ncc[0]));
console.log('zero composite has undefined NCC despite rounded positive 1-SSE');
