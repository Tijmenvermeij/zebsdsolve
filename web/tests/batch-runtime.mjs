import assert from "node:assert/strict";
import { BatchQueue, outputStem, defaultOutputStem } from "../app/js/batch.js";

const missing = () => Object.assign(new Error("missing"), { name: "NotFoundError" });
const mismatch = () => Object.assign(new Error("wrong kind"), { name: "TypeMismatchError" });
const dataSize = (data) => typeof data === "string" ? new TextEncoder().encode(data).byteLength : data?.size ?? data?.byteLength ?? 0;
class Directory {
  constructor(root = null, path = "") { this.entries = new Map(); this.root = root || this; this.path = path; this.permission = "granted"; this.fault = null; this.clock = 0; }
  async queryPermission(options) { assert.deepEqual(options, { mode: "readwrite" }); return this.permission; }
  async getDirectoryHandle(name, options = {}) {
    assert.ok(new TextEncoder().encode(name).byteLength <= 255, "directory name must fit the native component limit");
    if (this.entries.has(name)) { const item = this.entries.get(name); if (!(item instanceof Directory)) throw mismatch(); return item; }
    if (!options.create) throw missing();
    const directory = new Directory(this.root, `${this.path}/${name}`); this.entries.set(name, directory); return directory;
  }
  async getFileHandle(name, options = {}) {
    assert.ok(new TextEncoder().encode(name).byteLength <= 255, "file name must fit the native component limit");
    if (this.root.fault?.(`${this.path}/${name}`, "getFileHandle")) throw new Error(`getFileHandle ${name}`);
    if (this.entries.has(name)) { const item = this.entries.get(name); if (item instanceof Directory) throw mismatch(); return item; }
    if (!options.create) throw missing();
    const path = `${this.path}/${name}`, root = this.root;
    const file = { data: null, closes: 0, root, path, lastModified: ++root.clock,
      async isSameEntry(other) {
        if (root.fault?.(path, "isSameEntry")) throw new Error(`isSameEntry ${path}`);
        return other?.root === root && other?.path === path;
      },
      async getFile() {
        if (root.fault?.(path, "getFile")) throw new Error(`getFile ${path}`);
        return { size: dataSize(file.data), lastModified: file.lastModified, text: async () => String(file.data ?? "") };
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
    settings: { bandwidth: 123, batch: 2, mask: "none", map_rows: 99, map_cols: 98, pattern_height: 97, pattern_width: 96 },
    openHints: { map_rows: 2, map_cols: 3, pattern_height: 4, pattern_width: 5 }, scanInfo: { ...scan },
    geometry: { pc: [0.4, 0.5, 0.6], sampleTiltDeg: 70 }, openGeometry: { pc: [0.3, 0.5, 0.6], sampleTiltDeg: 70 }, phases: [{ name: "iron" }], version: "1" };
}
function fixture({ engineOptions = {}, artifacts: makeArtifacts } = {}) {
  const directory = new Directory(), calls = [], engines = [], counts = { index: 0, capture: 0 };
  let queue;
  const createEngine = async () => {
    let done = 0;
    const engine = { exportResultCheckpoint() { return new Uint8Array([1]); }, scan: { ...scan, ...engineOptions.scan }, deviceLost: null,
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
    for (const suffix of [".h5", /Oxford h5oina/.test(job.scanInfo.format) ? ".h5oina" : ".ang", "_IPF-Z.png", "_NCC.png"]) {
      const name = job.outputStem + suffix;
      if (completed.has(name)) continue;
      counts.capture++; yield { name, data: `data ${name}` };
    }
  });
  queue = new BatchQueue({ createEngine, artifacts }); queue.setDirectory(directory);
  return { queue, directory, calls, engines, counts };
}


const journal = (job) => JSON.parse(job._folder.entries.get(job.journalName).data);
function assertLeaf(job, vendor = ".ang") {
  assert.deepEqual([...job._folder.entries.keys()].sort(), [job.outputStem + ".h5", job.outputStem + vendor, job.outputStem + "_IPF-Z.png", job.outputStem + "_NCC.png", job.outputStem + ".json"].sort());
  assert.equal(journal(job).batch.status, "complete");
  assert.ok(journal(job).batch.outputs.every(entry => entry.complete));
}
assert.equal(outputStem("CON"), "_CON");
assert.equal(outputStem('bad:/name. '), "bad__name");
assert.equal(outputStem(".".repeat(200) + "a"), "scan");
assert.ok(new TextEncoder().encode(outputStem("custom_ZEBSDsolve_BW" + "1".repeat(300))).byteLength <= 200);
assert.equal(defaultOutputStem(file("scan.ebsp"), 123), "scan_ZEBSDsolve_BW123");
assert.ok(defaultOutputStem(file("a".repeat(250) + ".h5"), 123).endsWith("_ZEBSDsolve_BW123"));
assert.ok(defaultOutputStem(file("a".repeat(250) + ".h5"), 123).length <= 160);
for (const base of ["漢".repeat(80), "🧪".repeat(70), "長🧪".repeat(80)]) {
  for (const bandwidth of [123, 999999999999999]) {
    const name = defaultOutputStem(file(base + ".up1"), bandwidth);
    assert.ok(new TextEncoder().encode(name).byteLength <= 200);
    assert.ok(name.endsWith(`_ZEBSDsolve_BW${bandwidth}`)); assert.ok(name.isWellFormed());
    assert.ok(new TextEncoder().encode(name + ".h5oina").byteLength <= 255);
  }
  const custom = outputStem(base + '<>:"/\\|?* .');
  assert.ok(new TextEncoder().encode(custom).byteLength <= 200); assert.ok(custom.isWellFormed());
  assert.doesNotMatch(custom, /[<>:"/\\|?*]/);
}
for (const custom of [false, true]) {
  const { queue } = fixture(); const input = spec(); input.sourceFiles[0] = file("漢🧪".repeat(80) + ".up1");
  if (custom) input.outputStem = "🧪".repeat(70);
  const jobs = [queue.add(input), queue.add(input), queue.add(input)]; await queue.run();
  for (const job of jobs) {
    assertLeaf(job); assert.ok(new TextEncoder().encode(job.outputStem).byteLength <= 200); assert.ok(job.outputStem.isWellFormed());
    if (!custom) assert.match(job.outputStem, /_ZEBSDsolve_BW123(?:_\d+)?$/);
  }
  assert.ok(jobs[1].outputStem.endsWith("_02")); assert.ok(jobs[2].outputStem.endsWith("_03"));
}
{
  const { queue, directory, calls, counts } = fixture();
  const original = spec(), job = queue.add(original), second = queue.add(spec());
  original.geometry.pc[0] = 88; original.openGeometry.pc[0] = 99; original.settings.mask = "circle"; original.phases[0].name = "changed"; original.sourceFiles.reverse();
  assert.equal(job.sourceFiles[0].name, "scan.ebsp"); assert.equal(job.geometry.pc[0], .4); assert.equal(job.settings.mask, "none"); assert.equal(job.phases[0].name, "iron");
  assert.ok(Object.isFrozen(job.settings)); assert.ok(Object.isFrozen(job.geometry.pc)); assert.equal(job.openGeometry.pc[0], .3);
  await directory.getDirectoryHandle(job.outputStem, { create: true }); await queue.run(); assert.equal(queue.error, null);
  assert.equal(job.directoryName, "scan_ZEBSDsolve_BW123_02"); assert.equal(second.directoryName, "scan_ZEBSDsolve_BW123_03"); assert.equal(counts.index, 2);
  assert.equal(calls.filter(x => x === "dispose").length, 1); assert.deepEqual(calls.slice(-2), ["dispose", "destroy"]);
  const open = calls.findIndex(x => Array.isArray(x) && x[0] === "open");
  assert.deepEqual(calls[open - 1], ["configure", job.openHints]); assert.deepEqual(calls[open + 1], ["configure", job.settings]);
  assert.deepEqual(calls[open + 2], ["geometry", job.geometry]); assert.equal(calls[open + 3], "clearPhases"); assert.strictEqual(calls[open][1][0], job.sourceFiles[0]);
  assert.deepEqual(calls.slice(open + 4, open + 6).map(x => x[1].name), ["phase1.h5", "phase2.h5"]);
  for (const input of [...job.sourceFiles, ...job.masterFiles]) assert.deepEqual(input.reads, [[0, 16]]);
  assertLeaf(job); assertLeaf(second); assert.equal(journal(job).source_files[0].lastModified, 123); assert.equal(journal(job).acquisition_geometry.pc[0], .3);
  assert.deepEqual(journal(job).settings, job.settings); assert.deepEqual(journal(job).opening_hints, job.openHints);
  assert.ok([...directory.entries.values()].every(entry => entry instanceof Directory), "no files outside output leaves");
}
{
  const { queue } = fixture(); const a = new Directory(), b = new Directory(), input = spec();
  const job = queue.add({ ...input, directory: a }); const updated = spec(); updated.settings.bandwidth = 64;
  queue.replace(job.id, updated); assert.equal(job.outputStem, "scan_ZEBSDsolve_BW64"); assert.strictEqual(job.directory, a);
  queue.rename(job.id, "custom"); updated.settings.bandwidth = 96; queue.replace(job.id, updated); assert.equal(job.outputStem, "custom");
  queue.setJobDirectory(job.id, null); assert.equal(job.directory, null); queue.setJobDirectory(job.id, b);
  await queue.run(); assert.strictEqual(queue.directoryFor(job), b); assertLeaf(job); assert.throws(() => queue.setJobDirectory(job.id, a), /unstarted/);
}
{
  const { queue, directory: fallback } = fixture(); const a = new Directory(), b = new Directory(), input = { ...spec(), directory: a };
  const first = queue.add(input); input.directory = b; const second = queue.add({ ...spec(), directory: b });
  assert.strictEqual(first.directory, a); assert.equal(Object.isFrozen(a), false); await queue.run(); assertLeaf(first); assertLeaf(second);
  assert.equal(first.outputStem, second.outputStem); assert.equal(a.entries.size, 1); assert.equal(b.entries.size, 1); assert.equal(fallback.entries.size, 0);
}
{
  const { queue, counts, engines } = fixture(); queue.setDirectory(null); const a = queue.add(spec()), b = queue.add(spec());
  assert.equal(queue.hasDestinations, false); await queue.run(); assert.match(queue.error, /output folder/); assert.equal(engines.length, 0);
  queue.setJobDirectory(a.id, new Directory()); const denied = new Directory(); denied.permission = "prompt"; queue.setJobDirectory(b.id, denied);
  assert.equal(queue.hasDestinations, true); await queue.run(); assert.match(queue.error, /grant write permission/); assert.equal(counts.index, 0); assert.equal(engines.length, 0);
  denied.permission = "granted"; await queue.run(); assertLeaf(a); assertLeaf(b);
}
{
  const { queue, counts, engines } = fixture(), input = spec(); input.sourceFiles[0].slice = () => { throw new Error("input unavailable"); };
  const job = queue.add(input); await queue.run(); assert.equal(job.status, "queued"); assert.equal(counts.index, 0); assert.equal(engines.length, 0);
}
{
  const { queue, counts } = fixture({ engineOptions: { mismatch: true } }); const job = queue.add(spec());
  await queue.run(); assert.equal(job.status, "failed"); assert.match(job.error, /rows changed/); assert.equal(counts.index, 0);
}
for (const stage of ["write", "close"]) {
  const { queue, directory, counts, calls } = fixture(); const job = queue.add(spec()); queue.add(spec()); let failed = false;
  directory.fault = (path, at) => { if (!failed && at === stage && path.endsWith(".ang")) { failed = true; return true; } return false; };
  await queue.run(); assert.equal(job.status, "export-pending"); assert.equal(queue.jobs[1].status, "queued"); assert.equal(counts.index, 1);
  assert.ok(!calls.includes("dispose")); assert.equal(job.outputs[0].complete, true); const captures = counts.capture;
  directory.fault = null; await queue.run(); assertLeaf(job); assert.equal(counts.index, 2); assert.equal(counts.capture - captures, 7);
  assert.equal(job._folder.entries.get(job.outputStem + ".h5").closes, 1);
}
for (const [stage, errorName] of [["getFileHandle", "NotReadableError"], ["isSameEntry", "NotReadableError"], ["getFile", "NotReadableError"], ["getFile", "NotAllowedError"]]) {
  const { queue, directory, counts } = fixture(); const job = queue.add(spec()); let failed = false;
  directory.fault = (path, at) => { if (!failed && at === "close" && path.endsWith(".ang")) { failed = true; return true; } return false; };
  await queue.run(); assert.equal(job.status, "export-pending"); const saved = { ...job.outputs[0].saved };
  let unreadable = false;
  directory.fault = (path, at) => {
    if (!unreadable && path.endsWith(".h5") && at === stage) {
      unreadable = true; throw Object.assign(new Error("temporarily unreadable"), { name: errorName });
    }
    return false;
  };
  const captures = counts.capture; await queue.run();
  assert.equal(job.status, "export-pending"); assert.match(job.error, /temporarily unreadable/);
  assert.equal(job.outputs[0].complete, true); assert.equal(job.outputs[0].invalid, undefined); assert.deepEqual(job.outputs[0].saved, saved);
  assert.equal(counts.capture, captures); directory.fault = null; await queue.run();
  assertLeaf(job); assert.equal(counts.index, 1); assert.equal(counts.capture - captures, 3);
  assert.equal(job.outputs[0].handle.closes, 1);
}
for (const data of ["漢🧪", new Uint8Array([0, 1, 255]), new Uint8Array([5, 7]).buffer, new Blob(["abc"])]) {
  let captures = 0;
  const { queue, directory, counts } = fixture({ artifacts: async function* (engine, job, completed) {
    for (const suffix of [".h5", ".ang", "_IPF-Z.png", "_NCC.png"]) {
      if (completed.has(job.outputStem + suffix)) continue;
      captures++; yield { name: job.outputStem + suffix, data: suffix === ".h5" ? data : "ANG result" };
    }
  } });
  const job = queue.add(spec()); let failed = false;
  directory.fault = (path, stage) => {
    if (!failed && path.endsWith(".h5") && stage === "getFile" && job.outputs[0]?.handle?.closes) {
      failed = true; throw Object.assign(new Error("post-close stat unavailable"), { name: "NotReadableError" });
    }
    return false;
  };
  await queue.run(); assert.equal(job.status, "export-pending"); assert.equal(job.outputs[0].complete, true);
  assert.equal(job.outputs[0].saved, undefined); assert.equal(job.outputs[0].closed.size, dataSize(data));
  assert.equal(captures, 1); directory.fault = null; await queue.run();
  assertLeaf(job); assert.equal(counts.index, 1); assert.equal(captures, 4); assert.equal(job.outputs[0].handle.closes, 1);
  assert.equal(job.outputs[0].saved.size, dataSize(data));
}
{
  const { queue, directory, counts } = fixture(); const job = queue.add(spec()); let failed = false;
  directory.fault = (path, stage, data) => {
    if (!failed && path.endsWith(".h5") && stage === "close") {
      failed = true; job.outputs[0].handle.data = data; job.outputs[0].handle.lastModified = ++directory.clock; return true;
    }
    return false;
  };
  await queue.run(); assert.equal(job.status, "export-pending"); assert.equal(job.outputs[0].closed, undefined);
  const committed = job.outputs[0].handle.data; directory.fault = null; await queue.run();
  assert.equal(job.status, "export-pending"); assert.match(job.error, /changed since/);
  assert.equal(job.outputs[0].handle.data, committed); assert.equal(job.outputs[0].closed, undefined); assert.equal(counts.index, 1);
}
for (const change of ["delete", "replace"]) {
  const { queue, directory, counts } = fixture(); const job = queue.add(spec()); let failed = false;
  directory.fault = (path, stage) => { if (!failed && path.endsWith(".ang") && stage === "close") { failed = true; return true; } return false; };
  await queue.run(); assert.equal(job.status, "export-pending"); assert.equal(job.outputs[0].complete, true);
  const name = job.outputs[0].name; job._folder.entries.delete(name);
  let replacement;
  if (change === "replace") {
    replacement = await job._folder.getFileHandle(name, { create: true });
    replacement.data = "foreign H5 result";
  }
  const captures = counts.capture; directory.fault = null; await queue.run();
  assert.equal(job.status, "export-pending"); assert.match(job.error, /Saved output is missing or changed/);
  assert.equal(job.outputs[0].complete, false); assert.equal(journal(job).batch.outputs[0].complete, false);
  assert.match(journal(job).batch.outputs[0].error, /missing or changed/);
  assert.equal(journal(job).batch.status, "export-pending"); assert.equal(counts.index, 1); assert.equal(counts.capture, captures);
  if (replacement) assert.equal(replacement.data, "foreign H5 result"); else assert.equal(job._folder.entries.has(name), false);
  await queue.run(); assert.equal(counts.capture, captures); assert.equal(counts.index, 1); assert.equal(job.status, "export-pending");
}
{
  const { queue, counts } = fixture({ artifacts: async function* (engine, job) {
    yield { name: job.outputStem + ".h5", data: "H5 results" };
    yield { name: job.outputStem + ".ang", data: "ANG results" };
    for (const suffix of ["_IPF-Z.png", "_NCC.png"]) yield { name: job.outputStem + suffix, data: "pixels" };
    job._folder.entries.delete(job.outputStem + ".h5");
  } });
  const job = queue.add(spec()); await queue.run();
  assert.equal(job.status, "export-pending"); assert.equal(job.outputs[0].complete, false);
  assert.equal(journal(job).batch.outputs[0].complete, false); assert.equal(journal(job).batch.status, "export-pending");
  assert.equal(counts.index, 1);
}
{
  const { queue, directory, counts, calls } = fixture(); const job = queue.add(spec()); let failed = false;
  directory.fault = (path, stage, data) => {
    if (!failed && path.endsWith(".json") && stage === "close" && JSON.parse(data).batch.status === "complete") {
      assert.equal(job.status, "exporting"); failed = true; return true;
    } return false;
  };
  await queue.run(); assert.equal(job.status, "export-pending"); assert.equal(counts.index, 1); assert.ok(!calls.includes("dispose"));
  directory.fault = null; await queue.run(); assertLeaf(job); assert.equal(counts.index, 1); assert.equal(counts.capture, 4);
}
{
  const { queue, directory, counts, engines } = fixture(); const job = queue.add(spec());
  directory.fault = (path, stage) => path.endsWith(".json") && stage === "close";
  await queue.run(); assert.equal(job.status, "failed"); assert.equal(counts.index, 0); assert.equal(engines.length, 0);
}
for (const when of ["begin", "step"]) {
  const { queue, counts, calls } = fixture({ engineOptions: { [when]: queue => queue.stop() } }); const job = queue.add(spec()); queue.add(spec());
  await queue.run(); assert.equal(job.status, "cancelled"); assert.equal(queue.jobs[1].status, "queued"); assert.equal(counts.capture, 0);
  assert.ok(calls.includes("cancel")); if (when === "begin") assert.ok(!calls.includes("step"));
}
{
  let queue; const f = fixture({ artifacts: async function* (engine, job) {
    queue.stop(); yield { name: job.outputStem + ".h5", data: "results" }; yield { name: job.outputStem + ".ang", data: "results" }; for (const suffix of ["_IPF-Z.png", "_NCC.png"]) yield { name: job.outputStem + suffix, data: "pixels" };
  } }); queue = f.queue; const job = queue.add(spec()); queue.add(spec()); await queue.run(); assertLeaf(job); assert.equal(queue.jobs[1].status, "queued");
}
for (const initialState of ["failed", "cancelled"]) {
  let firstBegin = true, firstExport = true; const exports = [], rootA = new Directory(), rootB = new Directory();
  const { queue, counts } = fixture({ engineOptions: { begin: (queue, engine) => {
    engine.resultIdentity = queue._active.id;
    if (firstBegin) { firstBegin = false; if (initialState === "cancelled") queue.stop(); else throw new Error("A failed"); }
  } }, artifacts: async function* (engine, job, completed) {
    exports.push([job.id, engine.resultIdentity]); assert.equal(engine.resultIdentity, job.id);
    if (firstExport) { firstExport = false; throw new Error("B export failed"); }
    for (const suffix of [".h5", ".ang", "_IPF-Z.png", "_NCC.png"]) if (!completed.has(job.outputStem + suffix)) yield { name: job.outputStem + suffix, data: engine.resultIdentity };
  } });
  const a = queue.add({ ...spec(), directory: rootA }), b = queue.add({ ...spec(), directory: rootB });
  await queue.run(); assert.equal(a.status, initialState); await queue.run(); assert.equal(b.status, "export-pending");
  queue.retry(a.id); rootA.permission = "denied"; b.sourceFiles[0].slice = () => { throw new Error("B input unavailable"); };
  await queue.run(); assertLeaf(b); assert.equal(b._folder.entries.get(b.outputStem + ".h5").data, b.id);
  assert.equal(a.status, "queued"); assert.equal(counts.index, 2); assert.match(queue.error, /grant write permission/); assert.deepEqual(exports, [[b.id, b.id], [b.id, b.id]]);
  rootA.permission = "granted"; await queue.run(); assertLeaf(a); assert.equal(counts.index, 3);
}
{
  const { queue, directory, counts } = fixture(); const job = queue.add(spec()); let failed = false;
  directory.fault = (path, stage) => { if (!failed && path.endsWith(".ang") && stage === "close") { failed = true; return true; } return false; };
  await queue.run(); assert.equal(job.status, "export-pending"); directory.permission = "denied"; await queue.run(); assert.equal(job.status, "export-pending");
  directory.permission = "granted"; directory.fault = null; await queue.run(); assertLeaf(job); assert.equal(counts.index, 1);
}
{
  const { queue, counts } = fixture(); const job = queue.add(spec()); job.status = "export-pending";
  await assert.rejects(queue.run(), /Pending export state is invalid/); assert.equal(counts.index, 0);
}
for (const useWebLocks of [false, true]) {
  const old = Object.getOwnPropertyDescriptor(globalThis, "navigator"); const names = []; let tail = Promise.resolve();
  if (useWebLocks) Object.defineProperty(globalThis, "navigator", { configurable: true, value: { locks: { request(name, options, operation) {
    names.push(name); assert.equal(options.mode, "exclusive"); const result = tail.then(operation); tail = result.catch(() => {}); return result;
  } } } });
  try {
    const left = fixture(), right = fixture(); right.queue.setDirectory(left.directory);
    const a = left.queue.add({ ...spec(), version: "left" }), b = right.queue.add({ ...spec(), version: "right" });
    await Promise.all([left.queue.run(), right.queue.run()]); assertLeaf(a); assertLeaf(b); assert.notEqual(a.directoryName, b.directoryName);
    assert.equal(journal(a).zebsdsolve, "left"); assert.equal(journal(b).zebsdsolve, "right"); assert.equal(left.directory.entries.size, 2);
    assert.ok([...left.directory.entries.values()].every(value => value instanceof Directory));
    if (useWebLocks) { assert.ok(names.length > 8); assert.equal(new Set(names).size, 1); }
  } finally { if (old) Object.defineProperty(globalThis, "navigator", old); else delete globalThis.navigator; }
}
{
  const { queue, directory, counts } = fixture(); const job = queue.add(spec()); let failed = false;
  directory.fault = (path, stage, data) => {
    if (!failed && path.endsWith(".json") && stage === "close" && JSON.parse(data).batch.status === "exporting") { failed = true; return true; } return false;
  };
  await queue.run(); assert.equal(job.status, "export-pending"); assert.equal(job.outputs[0].handle, undefined);
  const foreign = await job._folder.getFileHandle(job.outputs[0].name, { create: true }); foreign.data = "foreign output";
  directory.fault = null; await queue.run(); assert.equal(job.status, "export-pending"); assert.match(job.error, /already exists/);
  assert.equal(foreign.data, "foreign output"); assert.equal(counts.index, 1);
}
for (const empty of [false, true]) {
  const { queue, directory, counts } = fixture(); const job = queue.add(spec()); directory.fault = (path, stage) => path.endsWith(".h5") && stage === "close";
  await queue.run(); assert.equal(job.status, "export-pending"); const name = job.outputs[0].name; job._folder.entries.delete(name);
  const foreign = await job._folder.getFileHandle(name, { create: true }); if (!empty) foreign.data = "replacement";
  assert.equal(await job.outputs[0].handle.isSameEntry(foreign), true); directory.fault = null; await queue.run();
  assert.equal(job.status, "export-pending"); assert.match(job.error, /changed since/); assert.equal(foreign.data, empty ? null : "replacement"); assert.equal(counts.index, 1);
}
for (const foreignContent of [false, true]) {
  const { queue, directory, counts } = fixture(); const job = queue.add(spec()); let failed = false;
  directory.fault = (path, stage) => { if (!failed && path.endsWith(".h5") && stage === "getFile") { failed = true; return true; } return false; };
  await queue.run(); assert.equal(job.status, "export-pending"); assert.equal(job.outputs[0].placeholder, undefined);
  if (foreignContent) job.outputs[0].handle.data = "foreign content"; directory.fault = null; await queue.run(); assert.equal(counts.index, 1);
  if (foreignContent) { assert.equal(job.status, "export-pending"); assert.equal(job.outputs[0].handle.data, "foreign content"); } else assertLeaf(job);
}
{
  const { queue, directory, counts } = fixture(); const job = queue.add(spec()); let failed = false;
  directory.fault = (path, stage) => { if (!failed && path.endsWith(".ang") && stage === "close") { failed = true; return true; } return false; };
  await queue.run(); assert.equal(job.status, "export-pending"); const handle = job._journal.handle;
  const expected = handle.data, foreign = expected.replace('"none"', '"oops"');
  assert.notEqual(foreign, expected); assert.equal(foreign.length, expected.length); handle.data = foreign;
  directory.fault = null; await queue.run(); assert.equal(job.status, "export-pending"); assert.match(job.error, /journal changed/); assert.equal(handle.data, foreign); assert.equal(counts.index, 1);
}
{
  const { queue, directory, counts } = fixture(); const job = queue.add(spec()); let failed = false;
  directory.fault = (path, stage, data) => {
    if (!failed && path.endsWith(".json") && stage === "close" && JSON.parse(data).batch.status === "complete") {
      failed = true; job._journal.handle.data = data; job._journal.handle.lastModified = ++directory.clock; return true;
    } return false;
  };
  await queue.run(); assert.equal(job.status, "export-pending"); const committed = job._journal.handle.data;
  directory.fault = null; await queue.run(); assert.equal(job.status, "export-pending"); assert.match(job.error, /inspect it/); assert.equal(job._journal.handle.data, committed); assert.equal(counts.index, 1);
}
{
  const { queue, calls } = fixture({ artifacts: async function* () { throw new Error("snapshot capture"); } }); const job = queue.add(spec());
  await queue.run(); assert.equal(job.status, "export-pending"); assert.throws(() => queue.remove(job.id), /Clear the batch/);
  await queue.clear(); assert.equal(queue.jobs.length, 0); assert.deepEqual(calls.slice(-2), ["dispose", "destroy"]);
}
{
  const { queue, counts } = fixture({ engineOptions: { step: (queue, engine) => { engine.deviceLost = { message: "removed" }; } } });
  const job = queue.add(spec()); queue.add(spec()); await queue.run(); assert.equal(job.status, "failed"); assert.match(job.error, /GPU device lost/); assert.equal(counts.index, 1);
}
{
  const { queue } = fixture({ engineOptions: { scan: { format: "Oxford h5oina" } } }); const input = spec(); input.scanInfo.format = "Oxford h5oina"; input.geometry.planeKind = 2;
  const job = queue.add(input); assert.equal(job.warnings.length, 1); assert.ok(Object.isFrozen(job.warnings)); input.geometry.planeKind = 0;
  await queue.run(); assertLeaf(job, ".h5oina"); assert.equal(journal(job).warnings.length, 1);
}
console.log("Batch runtime: per-job destinations, five-file journals, frozen replay, concurrency, cancellation and guarded retries passed.");
