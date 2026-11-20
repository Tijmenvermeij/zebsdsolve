import assert from 'node:assert/strict';
import {overlapTrajectoryMetadata} from '../web/zebsdsolve.js';
const result={retained_order:[3,3,0,3],objective_valid:[1,1,0,1],sse:[.1,.2,NaN,.3],initial_sse:[1,1,1,1]};
const bytes=new Uint8Array(4*32),view=new DataView(bytes.buffer);
for(const [i,value]of [1,0,0xffffffff,0xffffffff].entries())view.setUint32(i*32+28,value,true);
const current=overlapTrajectoryMetadata(result,bytes,5,4,1);
assert.deepEqual([...current.trajectory_source],[1,0,-1,-1]);
assert.equal(current.trajectory_selection,'best_valid_final_sse_of_pooled_and_rank1_v1');
// Old V5 used this word as an ephemeral phase gate, including 1 and max-u32.
const old=overlapTrajectoryMetadata(result,bytes,5,4,0);
assert.deepEqual([...old.trajectory_source],[1,1,-1,1]);assert.equal(old.trajectory_selection,'greedy_pooled_v5');
assert.throws(()=>overlapTrajectoryMetadata(result,bytes,5,4,2),/unsupported overlap trajectory version/);
assert.deepEqual([...overlapTrajectoryMetadata(result,bytes,1,4,0).trajectory_source],[0,0,-1,0]);
assert.equal(overlapTrajectoryMetadata(result,bytes,8,1,0).trajectory_selection,'ordinary_primary_only_v1');
view.setUint32(28,2,true);assert.throws(()=>overlapTrajectoryMetadata(result,bytes,5,4,1),/invalid V5/);
console.log('trajectory bridge: versioned provenance, old phase gates ignored, unknown versions fail, sole/rank1 policies pass');
