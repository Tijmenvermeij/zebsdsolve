import assert from "node:assert/strict";
import { BatchQueue, outputStem } from "../app/js/batch.js";

const missing = () => Object.assign(new Error("missing"), { name: "NotFoundError" });
const mismatch = () => Object.assign(new Error("wrong kind"), { name: "TypeMismatchError" });
class Directory {
  constructor(root = null, path = "") { this.entries = new Map(); this.root = root || this; this.path = path; this.permission = "granted"; this.fault = null; this.clock = 0; }
  async queryPermission(options) { assert.deepEqual(options, { mode: "readwrite" }); return this.permission; }
  async getDirectoryHandle(name, options = {}) {
    if (this.entries.has(name)) { const item = this.entries.get(name); if (!(item instanceof Directory)) throw mismatch(); return item; }
    if (!options.create) throw missing();
    const directory = new Directory(this.root, `${this.path}/${name}`); this.entries.set(name, directory); return directory;
  }
  async getFileHandle(name, options = {}) {
    if (this.entries.has(name)) { const item = this.entries.get(name); if (item instanceof Directory) throw mismatch(); return item; }
    if (!options.create) throw missing();
    const path = `${this.path}/${name}`, root = this.root;
    const file = { data: null, closes: 0, root, path, lastModified: ++root.clock,
      async isSameEntry(other) { return other?.root === root && other?.path === path; },
      async getFile() {
        if (root.fault?.(path, "getFile")) throw new Error(`getFile ${path}`);
        return { size: file.data == null ? 0 : new TextEncoder().encode(String(file.data)).byteLength, lastModified: file.lastModified };
      },
      async createWritable() {
      let data;
      return { async write(value) { if (root.fault?.(path, "write", value)) throw new Error(`write ${path}`); data = value; },
        async close() { if (root.fault?.(path, "close", data)) throw new Error(`close ${path}`); file.data = data; file.lastModified = ++root.clock; file.closes++; }, async abort() {} };
    } };
    this.entries.set(name, file); return file;
  }
}
function file(name, text = "012345678901234567890123456789") {
  const blob = new Blob([text]); blob.name = name; blob.lastModified = 123;
  const slice = blob.slice.bind(blob); blob.reads = [];
  blob.slice = (start, end) => { blob.reads.push([start, end]); return slice(start, end); };
  return blob;
}
const scan = { rows: 2, cols: 3, patH: 4, patW: 5, dtype: "uint8", format: "test" };
function spec() {
  return { sourceFiles: [file("scan.ebsp"), file("Setting.txt")], masterFiles: [file("phase1.h5"), file("phase2.h5")],
    settings: { batch: 2, mask: "none", map_rows: 99, map_cols: 98, pattern_height: 97, pattern_width: 96 },
    openHints: { map_rows: 2, map_cols: 3, pattern_height: 4, pattern_width: 5 }, scanInfo: { ...scan },
    geometry: { pc: [0.4, 0.5, 0.6], sampleTiltDeg: 70 }, openGeometry: { pc: [0.3, 0.5, 0.6], sampleTiltDeg: 70 }, phases: [{ name: "iron" }], version: "1" };
}
function fixture({ engineOptions = {}, artifacts: makeArtifacts } = {}) {
  const directory = new Directory(), calls = [], engines = [], counts = { index: 0, capture: 0 };
  let queue;
  const createEngine = async () => {
    let done = 0;
    const engine = { scan: { ...scan }, deviceLost: null,
      device: { destroy() { calls.push("destroy"); } },
      async closeScan() { calls.push("closeScan"); },
      configure(settings) { calls.push(["configure", settings]); },
      async openScan(files) { calls.push(["open", files]); if (engineOptions.mismatch) this.scan.rows++; },
      setGeometry(geometry) { calls.push(["geometry", geometry]); },
      async clearPhases() { calls.push("clearPhases"); }, async addPhase(file) { calls.push(["phase", file]); },
      async indexBegin() { calls.push("begin"); counts.index++; done = 0; await engineOptions.begin?.(queue, engine); },
      get indexStatus() { return { total: 6, done, sweep: 0, sweeps: 1 }; },
      async indexStep() { calls.push("step"); await engineOptions.step?.(queue, engine); if (engineOptions.fail) throw new Error("native failed"); done += 2; return { total: 6, done, sweep: 0, sweeps: 1, left: queue._stop || done >= 6 ? 0 : 6 - done }; },
      cancel() { calls.push("cancel"); }, async dispose() { calls.push("dispose"); },
    };
    engines.push(engine); return engine;
  };
  const artifacts = makeArtifacts || (async function* (engine, job, completed) {
    for (const suffix of [".h5", "_settings.json", "_IPF-Z.png", "_NCC.png"]) {
      const name = job.outputStem + suffix;
      if (completed.has(name)) continue;
      counts.capture++; yield { name, data: `data ${name}` };
    }
  });
  queue = new BatchQueue({ createEngine, artifacts }); queue.setDirectory(directory);
  return { queue, directory, calls, engines, counts };
}

assert.equal(outputStem("CON"), "_CON");
assert.equal(outputStem('bad:/name. '), "bad__name");
{
  const { queue, directory, calls, counts } = fixture();
  const original = spec(), job = queue.add(original), second = queue.add(spec());
  original.geometry.pc[0] = 88; original.openGeometry.pc[0] = 99; original.settings.mask = "circle"; original.phases[0].name = "changed";
  original.sourceFiles.reverse();
  assert.equal(job.sourceFiles[0].name, "scan.ebsp"); assert.equal(job.geometry.pc[0], .4);
  assert.equal(job.settings.mask, "none"); assert.equal(job.phases[0].name, "iron");
  assert.ok(Object.isFrozen(job.settings)); assert.ok(Object.isFrozen(job.geometry.pc));
  assert.equal(job.openGeometry.pc[0], .3); assert.ok(Object.isFrozen(job.openGeometry.pc));
  await directory.getDirectoryHandle(job.outputStem, { create: true });
  await directory.getFileHandle("batch.json", { create: true });
  await queue.run();
  assert.equal(queue.error, null); assert.equal(job.status, "complete"); assert.equal(second.status, "complete");
  assert.equal(job.directoryName, "scan_zebsdsolve_02"); assert.equal(second.directoryName, "scan_zebsdsolve_03");
  assert.equal(queue.manifestName, "batch-02.json"); assert.equal(counts.index, 2); assert.equal(calls.filter(x => x === "dispose").length, 1);
  assert.deepEqual(calls.slice(-2), ["dispose", "destroy"]);
  const firstOpen = calls.findIndex(x => Array.isArray(x) && x[0] === "open");
  assert.deepEqual(calls[firstOpen - 1], ["configure", job.openHints]);
  assert.deepEqual(calls[firstOpen + 1], ["configure", job.settings]);
  assert.deepEqual(calls[firstOpen + 2], ["geometry", job.geometry]);
  assert.equal(calls[firstOpen + 3], "clearPhases");
  assert.strictEqual(calls[firstOpen][1][0], job.sourceFiles[0]);
  assert.deepEqual(calls.slice(firstOpen + 4, firstOpen + 6).map(x => x[1].name), ["phase1.h5", "phase2.h5"]);
  for (const input of [...job.sourceFiles, ...job.masterFiles]) assert.deepEqual(input.reads, [[0, 16]]);
  const manifest = JSON.parse(directory.entries.get(queue.manifestName).data);
  assert.equal(manifest.jobs[0].status, "complete"); assert.equal(manifest.jobs[0].sources[0].lastModified, 123);
  assert.equal(manifest.jobs[0].openGeometry.pc[0], .3);
  assert.ok(manifest.jobs[0].outputs.every(item => item.complete));
}
{
  const { queue, counts } = fixture({ engineOptions: { mismatch: true } }); const job = queue.add(spec());
  await queue.run(); assert.equal(job.status, "failed"); assert.match(job.error, /rows changed/); assert.equal(counts.index, 0);
}
for (const faultStage of ["write", "close"]) {
  const { queue, directory, counts, calls } = fixture(); const job = queue.add(spec()); queue.add(spec());
  let failed = false;
  directory.fault = (path, stage) => { if (!failed && stage === faultStage && path.endsWith("_settings.json")) { failed = true; return true; } return false; };
  await queue.run(); assert.equal(job.status, "export-pending"); assert.equal(queue.jobs[1].status, "queued");
  assert.equal(counts.index, 1); assert.equal(job.outputs[0].complete, true); assert.ok(!calls.includes("dispose"));
  const captures = counts.capture; queue.retry(job.id); directory.fault = null; await queue.run();
  assert.equal(job.status, "complete"); assert.equal(counts.index, 2); // only the second, previously queued job indexed
  assert.equal(counts.capture - captures, 7); // retry's 3 missing outputs plus the next job's 4
  assert.equal(job._folder.entries.get(job.outputStem + ".h5").closes, 1);
}
{
  const { queue, directory, counts, calls } = fixture(); const job = queue.add(spec());
  let failed = false;
  directory.fault = (path, stage, data) => {
    if (path.endsWith("/batch.json") && stage === "close" && JSON.parse(data).jobs[0].status === "complete" && !failed) {
      assert.equal(job.status, "exporting"); failed = true; return true;
    }
    return false;
  };
  await queue.run(); assert.equal(job.status, "export-pending"); assert.equal(counts.index, 1);
  assert.ok(!calls.includes("dispose")); directory.fault = null; await queue.run();
  assert.equal(job.status, "complete"); assert.equal(counts.index, 1); assert.equal(counts.capture, 4);
}
for (const when of ["begin", "step"]) {
  const engineOptions = { [when]: queue => queue.stop() };
  const { queue, counts, calls } = fixture({ engineOptions }); const job = queue.add(spec()); queue.add(spec());
  await queue.run(); assert.equal(job.status, "cancelled"); assert.equal(queue.jobs[1].status, "queued");
  assert.equal(counts.capture, 0); assert.ok(calls.includes("cancel"));
  if (when === "begin") assert.ok(!calls.includes("step"));
}
{
  let queue;
  const f = fixture({ artifacts: async function* (engine, job) { queue.stop(); yield { name: job.outputStem + ".h5", data: "results" }; } });
  queue = f.queue; const job = queue.add(spec()); queue.add(spec());
  await queue.run(); assert.equal(job.status, "complete"); assert.equal(queue.jobs[1].status, "queued"); assert.equal(f.counts.index, 1);
}
{
  const { queue, calls } = fixture({ engineOptions: { fail: true } }); const job = queue.add(spec()); queue.add(spec());
  await queue.run(); assert.equal(job.status, "failed"); assert.equal(queue.jobs[1].status, "queued"); assert.deepEqual(calls.slice(-2), ["dispose", "destroy"]);
}
{
  const { queue, directory, counts } = fixture(); const job = queue.add(spec()); directory.permission = "prompt";
  await queue.run(); assert.match(queue.error, /grant write permission/); assert.equal(counts.index, 0); assert.equal(job.status, "queued");
  queue.setDirectory({}); await queue.run(); assert.match(queue.error, /File System Access/);
}
{
  const { queue, directory, counts } = fixture(); const job = queue.add(spec());
  directory.fault = (path, stage) => path.endsWith("/batch.json") && stage === "close";
  await queue.run(); assert.equal(job.status, "failed"); assert.equal(counts.index, 0); assert.match(queue.error, /close/);
}
{
  const { queue, counts } = fixture(); const input = spec(); input.sourceFiles[0].slice = () => ({ async arrayBuffer() { throw new Error("File no longer available"); } });
  const job = queue.add(input); await queue.run();
  assert.equal(job.status, "queued"); assert.equal(counts.index, 0); assert.match(queue.error, /File no longer available/);
}
{
  const { queue, calls } = fixture({ artifacts: async function* () { throw new Error("snapshot capture"); } });
  const job = queue.add(spec()); const id = job.id; queue.replace(id, spec()); queue.rename(id, "updated");
  assert.equal(job.outputStem, "updated"); await queue.run(); assert.equal(job.status, "export-pending");
  assert.throws(() => queue.remove(id), /Clear the batch/); await queue.clear();
  assert.equal(queue.jobs.length, 0); assert.deepEqual(calls.slice(-2), ["dispose", "destroy"]);
}
{
  const { queue, counts } = fixture({ engineOptions: { step: (queue, engine) => { engine.deviceLost = { message: "removed" }; } } });
  const job = queue.add(spec()); queue.add(spec()); await queue.run();
  assert.equal(job.status, "failed"); assert.match(job.error, /GPU device lost/); assert.equal(counts.index, 1); assert.equal(queue.jobs[1].status, "queued");
}
for (const initialState of ["failed", "cancelled"]) {
  let firstBegin = true, firstExport = true;
  const exports = [];
  const { queue, counts, calls } = fixture({
    engineOptions: { begin: (queue, engine) => {
      engine.resultIdentity = queue._active.id;
      if (firstBegin) { firstBegin = false; if (initialState === "cancelled") queue.stop(); else throw new Error("A failed"); }
    } },
    artifacts: async function* (engine, job) {
      exports.push([job.id, engine.resultIdentity]);
      assert.equal(engine.resultIdentity, job.id, "an export must use its own indexed results");
      if (firstExport) { firstExport = false; throw new Error("B export failed"); }
      yield { name: job.outputStem + ".h5", data: engine.resultIdentity };
    },
  });
  const a = queue.add(spec()), b = queue.add(spec());
  await queue.run(); assert.equal(a.status, initialState); assert.equal(b.status, "queued");
  await queue.run(); assert.equal(b.status, "export-pending"); assert.equal(counts.index, 2);
  queue.retry(a.id);
  const originalSlice = a.sourceFiles[0].slice;
  a.sourceFiles[0].slice = b.sourceFiles[0].slice = () => { throw new Error("queued input unavailable"); };
  await queue.run();
  assert.equal(b.status, "complete", "an unrelated unreadable queued input must not block retained exports");
  assert.equal(b._folder.entries.get(b.outputStem + ".h5").data, b.id);
  assert.equal(a.status, "queued"); assert.equal(counts.index, 2);
  assert.match(queue.error, /queued input unavailable/);
  assert.deepEqual(exports, [[b.id, b.id], [b.id, b.id]]);
  assert.deepEqual(calls.slice(-2), ["dispose", "destroy"]);
  a.sourceFiles[0].slice = originalSlice;
  await queue.run(); assert.equal(a.status, "complete"); assert.equal(counts.index, 3);
  assert.equal(b._folder.entries.get(b.outputStem + ".h5").data, b.id);
  assert.deepEqual(exports[2], [a.id, a.id]);
}
{
  const { queue, counts } = fixture(); const job = queue.add(spec());
  job.status = "export-pending";
  await assert.rejects(queue.run(), /Pending export state is invalid/);
  assert.equal(counts.index, 0); assert.equal(queue.running, false);
}
for (const useWebLocks of [false, true]) {
  const previousNavigator = Object.getOwnPropertyDescriptor(globalThis, "navigator");
  const lockNames = []; let lockTail = Promise.resolve();
  if (useWebLocks) Object.defineProperty(globalThis, "navigator", { configurable: true, value: { locks: {
    request(name, options, operation) {
      lockNames.push(name); assert.deepEqual(options, { mode: "exclusive" });
      const result = lockTail.then(operation); lockTail = result.catch(() => {}); return result;
    },
  } } });
  try {
    const left = fixture(), right = fixture(); right.queue.setDirectory(left.directory);
    const a = left.queue.add({ ...spec(), version: "left" }), b = right.queue.add({ ...spec(), version: "right" });
    await Promise.all([left.queue.run(), right.queue.run()]);
    assert.equal(a.status, "complete"); assert.equal(b.status, "complete");
    assert.notEqual(left.queue.manifestName, right.queue.manifestName); assert.notEqual(a.directoryName, b.directoryName);
    assert.equal(JSON.parse(left.directory.entries.get(left.queue.manifestName).data).jobs[0].version, "left");
    assert.equal(JSON.parse(left.directory.entries.get(right.queue.manifestName).data).jobs[0].version, "right");
    assert.equal(a._folder.entries.get(a.outputStem + ".h5").data, `data ${a.outputStem}.h5`);
    assert.equal(b._folder.entries.get(b.outputStem + ".h5").data, `data ${b.outputStem}.h5`);
    if (useWebLocks) { assert.ok(lockNames.length >= 12); assert.equal(new Set(lockNames).size, 1); }
  } finally {
    if (previousNavigator) Object.defineProperty(globalThis, "navigator", previousNavigator); else delete globalThis.navigator;
  }
}
{
  const { queue, directory, counts } = fixture(); const job = queue.add(spec());
  let failed = false;
  directory.fault = (path, stage, data) => {
    if (!failed && path.endsWith("/batch.json") && stage === "close" && JSON.parse(data).jobs[0].outputs.length) { failed = true; return true; }
    return false;
  };
  await queue.run(); assert.equal(job.status, "export-pending"); assert.equal(job.outputs[0].handle, undefined);
  const foreign = await job._folder.getFileHandle(job.outputs[0].name, { create: true }); foreign.data = "foreign output";
  directory.fault = null; await queue.run();
  assert.equal(job.status, "export-pending"); assert.match(job.error, /already exists/);
  assert.equal(foreign.data, "foreign output"); assert.equal(counts.index, 1);
}
{
  const { queue, directory, counts } = fixture(); const job = queue.add(spec());
  directory.fault = (path, stage) => path.endsWith(".h5") && stage === "close";
  await queue.run(); assert.equal(job.status, "export-pending"); assert.ok(job.outputs[0].handle);
  const name = job.outputs[0].name; job._folder.entries.delete(name);
  const foreign = await job._folder.getFileHandle(name, { create: true }); foreign.data = "replacement output";
  assert.equal(await job.outputs[0].handle.isSameEntry(foreign), true, "native handles compare the root, path and kind");
  directory.fault = null; await queue.run();
  assert.equal(job.status, "export-pending"); assert.match(job.error, /changed since/);
  assert.equal(foreign.data, "replacement output"); assert.equal(counts.index, 1);
  const manifest = JSON.parse(directory.entries.get(queue.manifestName).data);
  assert.deepEqual(Object.keys(manifest.jobs[0].outputs[0]).sort(), ["complete", "name"]);
}
{
  const { queue, directory, counts } = fixture(); const job = queue.add(spec());
  directory.fault = (path, stage) => path.endsWith(".h5") && stage === "close";
  await queue.run(); assert.equal(job.status, "export-pending");
  const name = job.outputs[0].name; job._folder.entries.delete(name);
  const replacement = await job._folder.getFileHandle(name, { create: true });
  assert.equal((await replacement.getFile()).size, 0);
  directory.fault = null; await queue.run();
  assert.equal(job.status, "export-pending"); assert.match(job.error, /changed since/);
  assert.equal(replacement.data, null); assert.equal(counts.index, 1);
}
{
  const { queue, directory, counts } = fixture(); const job = queue.add(spec());
  let failed = false;
  directory.fault = (path, stage, data) => {
    if (!failed && path.endsWith(".h5") && stage === "close") {
      failed = true;
      // A close error can have an uncertain commit outcome; preserve any resulting content.
      job.outputs[0].handle.data = data; job.outputs[0].handle.lastModified = ++directory.clock;
      return true;
    }
    return false;
  };
  await queue.run(); assert.equal(job.status, "export-pending");
  const original = job.outputs[0].handle.data;
  directory.fault = null; await queue.run();
  assert.equal(job.status, "export-pending"); assert.match(job.error, /changed since/);
  assert.equal(job.outputs[0].handle.data, original); assert.equal(counts.index, 1);
}
for (const foreignContent of [false, true]) {
  const { queue, directory, counts } = fixture(); const job = queue.add(spec());
  let failed = false;
  directory.fault = (path, stage) => {
    if (!failed && path.endsWith(".h5") && stage === "getFile") { failed = true; return true; }
    return false;
  };
  await queue.run(); assert.equal(job.status, "export-pending"); assert.ok(job.outputs[0].handle);
  assert.equal(job.outputs[0].placeholder, undefined); assert.equal(counts.index, 1);
  if (foreignContent) job.outputs[0].handle.data = "foreign content";
  directory.fault = null; await queue.run();
  assert.equal(counts.index, 1);
  if (foreignContent) {
    assert.equal(job.status, "export-pending"); assert.match(job.error, /inspect it before continuing/);
    assert.equal(job.outputs[0].handle.data, "foreign content");
  } else {
    assert.equal(job.status, "complete"); assert.ok(job.outputs[0].placeholder);
    assert.equal(job.outputs[0].handle.data, `data ${job.outputStem}.h5`);
  }
}
console.log("Batch runtime: frozen inputs, replay, output collisions, cancellation, failure/retry and cleanup passed.");
