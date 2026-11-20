import assert from 'node:assert/strict';
import { Zebsdsolve } from '../zebsdsolve.js';
import { clearPhasesWhenIdle, replacePhaseEntries } from '../app/js/phase-state.js';

const deferred = () => { let resolve; const promise = new Promise(r => resolve = r); return {promise, resolve}; };
const engine = Object.create(Zebsdsolve.prototype);
Object.assign(engine, {_queue: Promise.resolve(), _disposed: false, state: {deviceLoss: null}});
let active = 0, cleared = 0;
engine.clearPhases = () => { assert.equal(active, 0, 'native phase storage survives suspended consumers'); cleared++; };
const started = deferred(), release = deferred(), lateStarted = deferred(), lateRelease = deferred();
const consumer = engine._serial(async () => { active++; started.resolve(); await release.promise; active--; });
await started.promise;
const clearing = clearPhasesWhenIdle(engine, () => true);
const lateConsumer = engine._serial(async () => { active++; lateStarted.resolve(); await lateRelease.promise; active--; });
assert.equal(cleared, 0);
release.resolve(); await consumer; await lateStarted.promise;
assert.equal(cleared, 0, 'a consumer queued during the drain is also awaited');
lateRelease.resolve(); await lateConsumer;
assert.equal(await clearing, true); assert.equal(cleared, 1);

for (const invalidation of ['generation', 'device']) {
  let current = true;
  const began = deferred(), end = deferred();
  const pending = engine._serial(async () => { active++; began.resolve(); await end.promise; active--; });
  await began.promise;
  const clear = clearPhasesWhenIdle(engine, () => current && !engine.deviceLost);
  if (invalidation === 'generation') current = false; else engine.state.deviceLoss = {reason: 'test'};
  end.resolve(); await pending;
  assert.equal(await clear, false); assert.equal(cleared, 1);
  engine.state.deviceLoss = null;
}
console.log('Phase replacement drains in-flight native consumers and rejects stale generations/device loss.');

const entries = [{file: 'good', on: true}, {file: 'broken', on: true}, {file: 'later', on: true}, {file: 'off', on: false}];
let native = [], source = 'new acquisition';
engine.clearPhases = () => { native = []; };
engine.addPhase = async file => { if (file === 'broken') throw Error('bad master'); native.push(file); };
await assert.rejects(replacePhaseEntries(engine, entries, () => true), /bad master/);
assert.deepEqual(native, ['good']);
assert.deepEqual(entries.filter(p => p.on).map(p => p.file), native);
assert.equal(source, 'new acquisition', 'failed imports do not restore the previous acquisition');
console.log('Failed phase imports retain exactly the successfully loaded active prefix.');
