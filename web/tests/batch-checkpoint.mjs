import assert from 'node:assert/strict';
import { BatchQueue } from '../app/js/batch.js';

let captures = 0, reclaimed = 0, released = 0;
const downloadsAdapter = {
  archive: async () => ({blob: new Blob(['archive'])}),
  reclaim: async () => { reclaimed++; },
  release: () => { released++; },
  clear: () => {},
};
const queue = new BatchQueue({createEngine: () => {}, artifacts: async function* () {}, downloadsAdapter});
queue._engine = {exportResultCheckpoint() { return new Uint8Array([++captures]); }, resultChance() { return {median:0.1,spread:0.02}; }};
queue._validateCompleted = async () => {};
let failJournal = true;
queue._saveJournal = async () => { if (failJournal) {failJournal=false;throw Error('journal failed');} };
const job = {status:'exporting', _transport:'downloads', outputs:[{name:'job.json',complete:false}], journalName:'job.json'};
await assert.rejects(queue._export(job), /journal failed/);
const retained = job.resultCheckpoint;
assert.ok(retained instanceof Blob);
await queue._export(job);
assert.strictEqual(job.resultCheckpoint, retained, 'export retries retain the first immutable result checkpoint');
assert.equal(captures,1);
assert.deepEqual(job.resultChance,{median:0.1,spread:0.02});assert.ok(Object.isFrozen(job.resultChance));
assert.equal(reclaimed,1);
assert.deepEqual(new Uint8Array(await retained.arrayBuffer()),new Uint8Array([1]),'Downloads reclamation must not touch native results');
job.id='completed';queue.jobs.push(job);
queue.remove(job.id);
assert.equal(job.resultCheckpoint,null);assert.equal(job.resultChance,null);assert.equal(released,1);
const second={...job,id:'second',resultCheckpoint:retained,status:'complete'};
queue.jobs.push(second);queue._engine=null;
await queue.clear();assert.equal(second.resultCheckpoint,null);
const failed={...job,id:'failed',status:'failed',resultCheckpoint:retained};queue.jobs.push(failed);
queue.retry(failed.id);assert.equal(failed.resultCheckpoint,null);
console.log('Batch checkpoint lifecycle passed: one capture, retry retention, Downloads independence, remove/clear/retry release.');
