import assert from "node:assert/strict";
import { pairUPScan, validateUPProjectSources } from "../app/js/up-scan-files.js";
const up = {name:"patterns.up2",size:120}, ang = {name:"scan.ang",size:220};
let calls=0;
const choose=async pattern=>{assert.equal(pattern,up);calls++;return ang;};
assert.deepEqual(await pairUPScan([up],choose),[up,ang]); assert.equal(calls,1);
assert.deepEqual(await pairUPScan([ang,up],choose),[up,ang]); assert.equal(calls,1);
assert.equal(await pairUPScan([up],async()=>null),null);
assert.deepEqual(await pairUPScan([{name:"Oxford.h5oina"}],choose),[{name:"Oxford.h5oina"}]);
await assert.rejects(pairUPScan([up,{...up,name:"second.up1"}],choose),/one UP1/);
await assert.rejects(pairUPScan([up,ang,{...ang,name:"second.ang"}],choose),/one UP1/);
await assert.rejects(pairUPScan([ang],choose),/together/);
await assert.rejects(pairUPScan([up],async()=>({name:"wrong.osc"})),/ANG/);
assert.deepEqual(await pairUPScan([up,ang],choose,[up,ang]),[up,ang]);
await assert.rejects(pairUPScan([up,{...ang,size:1}],choose,[up,ang]),/project requires the original scan.ang/);
assert.deepEqual(await pairUPScan([up,ang],choose,null),[up,ang]);
console.log("UP/ANG selection: prompt, cancellation, supplied pairs, ambiguity and saved-project identity passed.");

await assert.rejects(pairUPScan([up,{name:"scan.h5"}],choose),/one UP1/);

await assert.rejects(pairUPScan([{name:"other.h5"}],choose,[up,ang]),/original UP1/);
assert.throws(()=>validateUPProjectSources([up,ang],[{name:"other.h5"}]),/original UP1/);
assert.doesNotThrow(()=>validateUPProjectSources([up,ang],[up,ang]));
assert.doesNotThrow(()=>validateUPProjectSources(undefined,[{name:"other.h5"}]));
assert.throws(()=>validateUPProjectSources([{...up,lastModified:10},ang],[{...up,lastModified:11},ang]),/original patterns.up2/);
