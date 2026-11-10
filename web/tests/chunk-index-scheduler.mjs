// Artificial IO latency exercises the real adapter queue, without GPU/network dependencies.
import assert from "node:assert/strict";
import { Zebsdsolve } from "../zebsdsolve.js";
const delay = ms => new Promise(resolve => setTimeout(resolve,ms));
const until = async fn => { for(let i=0;i<300;i++){ if(fn()) return; await delay(2); } throw new Error("scheduler timeout"); };
function engine(){
  const memory=new WebAssembly.Memory({initial:2}), log=[];
  let next=4096, active=0, source=0, visited=0, destroyed=false;
  const operation=async(kind,fn)=>{ assert.equal(active,0,"engine calls must not overlap");assert.equal(destroyed,false);active++;log.push(kind+":start");await delay(20);fn?.();active--;log.push(kind+":end"); };
  const x={ze_create:()=>1,ze_alloc:n=>{const p=next;next+=Math.ceil(n/8)*8;return p;},ze_free:()=>{log.push("free");},ze_error:()=>0,
    ze_pattern_shape(e,full,p){new Uint32Array(memory.buffer,p,2).set([2,2]);return 0;},ze_nlpar_view:()=>0,
    async ze_pattern(e,cell,k,full,p){await operation("pattern",()=>new Float32Array(memory.buffer,p,4).fill(source));return 0;},
    async ze_prepare_index_step(){await operation("warm",()=>visited++);return visited>=5?1:0;},
    ze_close_scan(){assert.equal(active,0);log.push("close");source++;visited=0;},
    async ze_open_scan(){await operation("open",()=>{source++;visited=0;});return 0;},
    ze_cancel(){assert.equal(destroyed,false);log.push("cancel");},ze_clear_phases(){assert.equal(destroyed,false);},ze_pixel_angle(){assert.equal(destroyed,false);return 1;},
    ze_destroy(){assert.equal(active,0);destroyed=true;log.push("destroy");},
  };
  const state={memory,listeners:[],files:[],deviceLoss:null};
  const z=new Zebsdsolve(x,state); Object.defineProperty(z,"scan",{get:()=>({source})});
  return {z,log,state,get visited(){return visited;}};
}
const a=engine();
assert.deepEqual(Array.from(await a.z.pattern(0,false)),[0,0,0,0]);
assert.equal(a.visited,0,"first preview must resolve before warm-up starts");
await until(()=>a.log.includes("warm:start"));
const preview=await a.z.pattern(1,false);
assert.equal(preview[0],0);
assert.ok(a.visited<5,"foreground preview must not wait for full warm-up");
assert.ok(a.log.indexOf("pattern:start",a.log.indexOf("warm:start"))>a.log.indexOf("warm:end"));
await until(()=>a.z.chunkIndexWarmup.state==="ready");
assert.equal(a.visited,5);
// Scan switching waits for an in-flight warm read, then cancels its generation.
const b=engine();await b.z.pattern(0,false);await until(()=>b.log.includes("warm:start"));
await b.z.closeScan();const stopped=b.visited;await delay(45);assert.equal(b.visited,stopped);
await b.z.openScan(new Blob(["new scan"]));await b.z.pattern(0,false);await until(()=>b.visited>0);
await b.z.dispose();assert.equal(b.log.at(-1),"destroy");const length=b.log.length;await delay(45);assert.equal(b.log.length,length);
b.z.cancel();assert.throws(()=>b.z.clearPhases(),/disposed/);assert.throws(()=>b.z.pixelAngle,/disposed/);await assert.rejects(b.z.pattern(0,false),/disposed/);
// Dispose while a foreground JSPI call is suspended: destroy follows frame release.
const c=engine();const running=c.z.pattern(0,false);const rejected=assert.rejects(running,/disposed/);await delay(2);await c.z.dispose();await rejected;
assert.ok(c.log.lastIndexOf("free")<c.log.indexOf("destroy"));
// GPU loss stops pending automatic work, and errors become observable without retry loops.
const d=engine();await d.z.pattern(0,false);d.state.deviceLoss={reason:"unknown",message:"test"};await delay(45);assert.equal(d.visited,0);await d.z.dispose();
const e=engine();let errors=0;e.state.listeners.push(v=>{if(v.kind==="chunkIndexError")errors++;});e.z.x.ze_prepare_index_step=async()=>{throw new Error("broken tree");};await e.z.pattern(0,false);await until(()=>e.z.chunkIndexWarmup.state==="failed");await delay(35);assert.equal(errors,1);await e.z.dispose();await a.z.dispose();
console.log("Chunk lookup scheduler passed: first preview, foreground priority, full transition, suspended switch/dispose, GPU loss and failure reporting.");
