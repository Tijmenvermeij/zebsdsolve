import assert from "node:assert/strict";
import { BatchQueue } from "../app/js/batch.js";
import { BatchDownloads } from "../app/js/batch-downloads.js";
import { createStoreZip } from "../app/js/zip-store.js";

const absent = () => Object.assign(new Error("missing"), { name: "NotFoundError" });
const wrongKind = () => Object.assign(new Error("wrong kind"), { name: "TypeMismatchError" });
class Locks {
  constructor() { this.held = new Map(); this.requests = []; }
  async request(name, options, operation) {
    this.requests.push({ name, ...options });
    while (this.held.has(name)) {
      if (options.ifAvailable) return operation(null);
      await this.held.get(name).finished;
    }
    let release;
    const finished = new Promise(resolve => { release = resolve; });
    this.held.set(name, { finished });
    try { return await operation({ name, mode: "exclusive" }); }
    finally { this.held.delete(name); release(); }
  }
}
class Directory {
  constructor(root = null, name = "origin", path = "") { this.root = root || this; this.name = name; this.path = path; this.children = new Map(); this.kind = "directory"; this.clock = 1; this.permissionCalls = 0; this.reads = []; }
  async *entries() { yield* this.children.entries(); }
  async queryPermission() { this.root.permissionCalls++; throw new Error("External permission API must not be called for Downloads."); }
  async getDirectoryHandle(name, options = {}) {
    const existing = this.children.get(name);
    if (existing) { if (!(existing instanceof Directory)) throw wrongKind(); return existing; }
    if (!options.create) throw absent();
    const directory = new Directory(this.root, name, `${this.path}/${name}`); this.children.set(name, directory); return directory;
  }
  async getFileHandle(name, options = {}) {
    const existing = this.children.get(name);
    if (existing) { if (existing instanceof Directory) throw wrongKind(); return existing; }
    if (!options.create) throw absent();
    const root = this.root, path = `${this.path}/${name}`;
    const file = { name, path, root, kind: "file", data: new Blob(), lastModified: ++root.clock, closes: 0,
      async isSameEntry(other) { return other.root === root && other.path === path; },
      async getFile() {
        const blob = new Blob([file.data]); blob.name = name; blob.lastModified = file.lastModified;
        const slice = blob.slice.bind(blob);
        blob.slice = (start, end) => { root.reads.push([path, start, end]); return slice(start, end); };
        return blob;
      },
      async createWritable() {
        let pending;
        return { async write(data) { root.fault?.(path, "write", data); pending = new Blob([data]); },
          async close() { root.fault?.(path, "close", pending); file.data = pending; file.lastModified = ++root.clock; file.closes++; }, async abort() {} };
      },
    };
    this.children.set(name, file); return file;
  }
  async removeEntry(name) { this.root.fault?.(`${this.path}/${name}`, "remove"); if (!this.children.has(name)) throw absent(); this.children.delete(name); }
}
function file(name) { const blob = new Blob(["012345678901234567890123456789"]); blob.name = name; blob.lastModified = 123; return blob; }
const scan = { rows: 1, cols: 1, patH: 2, patW: 2, dtype: "uint8", format: "UP" };
function spec(directory = null) { return { directory, sourceFiles: [file("漢🧪.up1")], masterFiles: [file("iron.h5")], settings: { bandwidth: 42, batch: 1, mask: "none" }, geometry: { pc: [.4, .5, .6] }, scanInfo: scan, version: "test", phases: [{ name: "iron" }] }; }
function fixture({ adapterOptions = {}, onDownloadReady } = {}) {
  const origin = new Directory(), external = new Directory(), cleanup = [], revoked = [], urls = new Map(), counters = { engines: 0, indices: 0, captures: 0, callbacks: 0, disposed: 0, destroyed: 0, directory: 0 };
  let unique = 0;
  const adapter = new BatchDownloads({ getDirectory: async () => { counters.directory++; return origin; }, randomId: () => `test-${++unique}`,
    locks: null, // Generic fixtures are isolated; lifetime-lock cases inject their own manager.
    createObjectURL: blob => { const url = `blob:test/${urls.size + 1}`; urls.set(url, blob); return url; }, revokeObjectURL: url => revoked.push(url),
    scheduleCleanup: (operation, delay) => cleanup.push({ operation, delay }), ...adapterOptions });
  const queue = new BatchQueue({ downloadsAdapter: adapter,
    createEngine: async () => {
      counters.engines++;
      return { scan, configure() {}, setGeometry() {}, async openScan() { return scan; }, async closeScan() {}, async clearPhases() {}, async addPhase() {},
        async indexBegin() { counters.indices++; }, get indexStatus() { return { total: 1, done: 0, sweep: 0, sweeps: 1 }; },
        async indexStep() { return { total: 1, done: 1, left: 0, sweep: 0, sweeps: 1 }; },
        async dispose() { counters.disposed++; }, exportResultCheckpoint() { return new Uint8Array([1]); }, device: { destroy() { counters.destroyed++; } } };
    }, artifacts: async function* (engine, job, completed) {
      for (const suffix of [".h5", ".ang", "_IPF-Z.png", "_NCC.png"]) {
        const name = job.outputStem + suffix; if (completed.has(name)) continue;
        counters.captures++; yield { name, data: suffix === ".h5" ? new Uint8Array([137, 72, 68, 70, 13, 10, 26, 10]) : "ANG results" };
      }
    }, onDownloadReady: async job => { counters.callbacks++; await onDownloadReady?.(job); },
  });
  return { queue, adapter, origin, external, cleanup, revoked, urls, counters };
}
async function readZip(blob) {
  const bytes = new Uint8Array(await blob.arrayBuffer()), view = new DataView(bytes.buffer);
  assert.equal(view.getUint32(bytes.length - 22, true), 0x06054b50);
  const count = view.getUint16(bytes.length - 12, true), start = view.getUint32(bytes.length - 6, true);
  let offset = start; const entries = [];
  for (let index = 0; index < count; index++) {
    assert.equal(view.getUint32(offset, true), 0x02014b50); assert.equal(view.getUint16(offset + 8, true), 0x0800); assert.equal(view.getUint16(offset + 10, true), 0);
    const length = view.getUint16(offset + 28, true), size = view.getUint32(offset + 24, true), crc = view.getUint32(offset + 16, true), local = view.getUint32(offset + 42, true);
    const name = new TextDecoder().decode(bytes.subarray(offset + 46, offset + 46 + length));
    assert.equal(view.getUint32(local, true), 0x04034b50); assert.equal(view.getUint16(local + 6, true), 0x0800); assert.equal(view.getUint16(local + 8, true), 0);
    assert.equal(view.getUint32(local + 14, true), crc); assert.equal(view.getUint32(local + 22, true), size);
    const dataStart = local + 30 + view.getUint16(local + 26, true) + view.getUint16(local + 28, true);
    entries.push({ name, bytes: bytes.slice(dataStart, dataStart + size), crc, date: view.getUint16(local + 12, true), time: view.getUint16(local + 10, true) });
    offset += 46 + length + view.getUint16(offset + 30, true) + view.getUint16(offset + 32, true);
  }
  assert.equal(offset, bytes.length - 22); return entries;
}
{
  const f = fixture({ onDownloadReady: job => {
    assert.equal(job.status, "complete"); assert.ok(job.archive.url); assert.equal(job._folder.children.size, 5, "archive is published before cache reclamation");
    throw new Error("automatic download blocked");
  } }), a = f.queue.add(spec(f.external)), b = f.queue.add(spec(f.external));
  assert.equal(f.queue.downloadsEnabled, false); assert.equal(f.queue.modeFor(a), "folder");
  f.queue.setDownloads(true); assert.equal(f.queue.modeFor(a), "downloads"); assert.equal(f.queue.hasDestinations, true);
  f.queue.setDownloads(false); assert.strictEqual(f.queue.directoryFor(a), f.external); f.queue.setDownloads(true);
  await f.queue.run(); assert.equal(f.queue.error, null); assert.equal(a.status, "complete"); assert.equal(b.status, "complete");
  assert.equal(f.counters.indices, 2); assert.equal(f.counters.engines, 1); assert.equal(f.counters.callbacks, 2); assert.equal(f.counters.disposed, 1); assert.equal(f.counters.destroyed, 1);
  assert.equal(f.external.permissionCalls, 0); assert.equal(f.origin.permissionCalls, 0); assert.equal(f.external.children.size, 0);
  assert.notEqual(a.archive.name, b.archive.name); assert.ok(b.archive.name.endsWith("_02.zip")); assert.match(a.downloadError, /automatic download blocked/);
  for (const job of [a, b]) {
    const entries = await readZip(job.archive.blob), prefix = job.outputStem + "/";
    assert.deepEqual(entries.map(entry => entry.name).sort(), [".h5", ".ang", "_IPF-Z.png", "_NCC.png", ".json"].map(suffix => prefix + job.outputStem + suffix).sort());
    const project = JSON.parse(new TextDecoder().decode(entries.find(entry => entry.name.endsWith(".json")).bytes));
    assert.equal(project.batch.transport, "downloads"); assert.equal(project.batch.status, "archive-ready"); assert.ok(project.batch.outputs.every(entry => entry.complete));
    assert.deepEqual(project.settings, job.settings); assert.equal(project.source_files[0].name, "漢🧪.up1");
    assert.strictEqual(await f.adapter.archive(job), job.archive); assert.ok(f.urls.has(job.archive.url));
    assert.equal(job._folder.children.size, 0, "successful archives reclaim staged members");
    assert.ok(job.archive.members.every(member => !("blob" in member)), "archive retains metadata, never source File references");
    assert.equal((await readZip(job.archive.blob)).length, 5, "ZIP remains readable after member removal");
  }
  assert.equal(f.revoked.length, 0); f.queue.setDownloads(false); assert.equal(f.queue.modeFor(a), "downloads"); assert.equal(f.queue.modeFor(f.queue.add(spec(f.external))), "folder");
  const url = a.archive.url; f.queue.remove(a.id); assert.equal(f.revoked.length, 0); assert.equal(f.cleanup.length, 1); assert.ok(f.cleanup[0].delay >= 60_000);
  await f.cleanup.shift().operation(); assert.ok(f.revoked.includes(url));
  assert.ok(await a._root.getDirectoryHandle(a.directoryName), "empty collision reservation survives row removal");
  assert.equal(b._folder.children.size, 0); assert.ok(!f.revoked.includes(b.archive.url));
  const bUrl = b.archive.url; await f.queue.clear(); assert.ok(!f.revoked.includes(bUrl)); assert.ok(f.cleanup.length); assert.equal(f.queue.jobs.length, 0);
  for (const { operation } of f.cleanup.splice(0)) await operation(); assert.ok(f.revoked.includes(bUrl));
}
{
  const f = fixture(), job = f.queue.add(spec()); f.queue.setDownloads(true); let failed = false;
  f.origin.fault = (path, stage) => { if (!failed && path.endsWith(".ang") && stage === "write") { failed = true; throw Object.assign(new Error("private staging quota exceeded"), { name: "QuotaExceededError" }); } };
  await f.queue.run(); assert.equal(job.status, "export-pending"); assert.match(job.error, /quota/); assert.equal(f.counters.indices, 1);
  assert.throws(() => f.queue.setDownloads(false), /reserved jobs/); assert.equal(f.queue.modeFor(job), "downloads");
  f.origin.fault = null; await f.queue.run(); assert.equal(job.status, "complete"); assert.equal(f.counters.indices, 1); assert.equal(f.counters.captures, 5);
  assert.equal(job.outputs[0].handle.closes, 1); assert.equal((await readZip(job.archive.blob)).length, 5);
}
{
  let packs = 0;
  const f = fixture({ adapterOptions: { pack: async members => { if (++packs === 1) throw new Error("packing interrupted"); return createStoreZip(members); } } });
  const job = f.queue.add(spec()); f.queue.setDownloads(true); await f.queue.run(); assert.equal(job.status, "export-pending"); assert.equal(f.counters.indices, 1);
  await f.queue.run(); assert.equal(job.status, "complete"); assert.equal(f.counters.indices, 1); assert.equal(f.counters.captures, 4); assert.equal(packs, 2);
  assert.equal(JSON.parse(new TextDecoder().decode((await readZip(job.archive.blob)).find(entry => entry.name.endsWith(".json")).bytes)).batch.status, "archive-ready");
}
{
  const f = fixture({ adapterOptions: { getDirectory: async () => { throw new Error("Downloads private storage is unavailable"); } } }), job = f.queue.add(spec());
  f.queue.setDownloads(true); await f.queue.run(); assert.equal(job.status, "queued"); assert.match(f.queue.error, /unavailable/); assert.equal(f.counters.engines, 0);
}
{
  const f = fixture(); await f.origin.getDirectoryHandle("another-tab", { create: true });
  const job = f.queue.add(spec()); f.queue.setDownloads(true); await f.queue.run(); const oldRoot = job._root, oldName = [...f.origin.children.keys()].find(name => name.startsWith(".zebsdsolve"));
  await f.queue.clear(); for (const { operation } of f.cleanup.splice(0)) operation(); await new Promise(resolve => setTimeout(resolve, 0));
  assert.equal(f.origin.children.has(oldName), false); assert.equal(f.origin.children.has("another-tab"), true);
  f.queue.add(spec()); await f.queue.run(); assert.notStrictEqual(f.queue.jobs[0]._root, oldRoot);
}
{
  let clock = 10;
  const f = fixture({ adapterOptions: { now: () => clock }, onDownloadReady: () => { clock = 20; } });
  const job = f.queue.add(spec()); f.queue.setDownloads(true);
  f.origin.fault = (path, stage) => { if (path.endsWith(".h5") && stage === "remove") throw new Error("cache removal temporarily unavailable"); };
  await f.queue.run(); assert.equal(job.status, "complete"); assert.match(job.cleanupError, /temporarily unavailable/);
  assert.equal((await readZip(job.archive.blob)).length, 5); assert.equal(f.counters.indices, 1);
  const workspace = [...f.origin.children.values()][0];
  assert.equal(JSON.parse(await workspace.children.get(".owner.json").data.text()).lastUsed, 20);
  f.origin.fault = null; await f.adapter.reclaim(job); assert.equal(job._folder.children.size, 0);
  assert.equal((await readZip(job.archive.blob)).length, 5);
}
{
  const origin = new Directory(), locks = new Locks(), scheduled = [], activeId = "11111111-1111-4111-8111-111111111111", nextId = "22222222-2222-4222-8222-222222222222";
  const abandonedId = "33333333-3333-4333-8333-333333333333", freshId = "44444444-4444-4444-8444-444444444444";
  const wrongId = "55555555-5555-4555-8555-555555555555", oversizedId = "66666666-6666-4666-8666-666666666666";
  const prefix = ".zebsdsolve-downloads-", lockName = id => "zebsdsolve-downloads-workspace:" + id;
  const marker = (id, time) => ({ magic: "zebsdsolve-downloads", version: 1, id, lockProtected: true, created: time, lastUsed: time });
  async function stage(id, data) {
    const directory = await origin.getDirectoryHandle(prefix + id, { create: true });
    const handle = await directory.getFileHandle(".owner.json", { create: true }), stream = await handle.createWritable();
    await stream.write(typeof data === "string" ? data : JSON.stringify(data)); await stream.close(); return directory;
  }
  const originalCreate = origin.getDirectoryHandle.bind(origin);
  origin.getDirectoryHandle = async (name, options = {}) => {
    if (options.create && name === prefix + activeId) assert.ok(locks.held.has(lockName(activeId)), "the live lock is acquired before creating the workspace");
    return originalCreate(name, options);
  };
  const active = new BatchDownloads({ getDirectory: async () => origin, locks, now: () => 0, randomId: () => activeId, scheduleCleanup: operation => scheduled.push(operation) });
  await active.directory(); assert.ok(locks.held.has(lockName(activeId)));
  await stage(abandonedId, marker(abandonedId, 0)); await stage(freshId, marker(freshId, 2000));
  await stage(wrongId, { ...marker(wrongId, 0), id: "foreign" }); await stage(oversizedId, "x".repeat(5000));
  await origin.getDirectoryHandle(prefix + "legacy", { create: true }); await origin.getDirectoryHandle("other-application", { create: true });
  const unmarkedId = "77777777-7777-4777-8777-777777777777"; await origin.getDirectoryHandle(prefix + unmarkedId, { create: true });
  origin.fault = (path, stage) => {
    if (stage === "remove" && path === "/" + prefix + abandonedId) assert.ok(locks.held.has(lockName(abandonedId)), "the stale lock stays held across deletion");
  };
  const next = new BatchDownloads({ getDirectory: async () => origin, locks, now: () => 2000, staleAfterMs: 1000, randomId: () => nextId, scheduleCleanup: operation => scheduled.push(operation) });
  await next.directory();
  assert.equal(origin.children.has(prefix + abandonedId), false); assert.equal(origin.children.has(prefix + activeId), true);
  for (const id of [freshId, wrongId, oversizedId, unmarkedId]) assert.equal(origin.children.has(prefix + id), true);
  assert.equal(origin.children.has(prefix + "legacy"), true); assert.equal(origin.children.has("other-application"), true);
  assert.ok(locks.requests.some(request => request.name === lockName(activeId) && request.ifAvailable === true));
  next.clear([]); assert.ok(locks.held.has(lockName(nextId))); for (const operation of scheduled.splice(0)) await operation();
  await new Promise(resolve => setTimeout(resolve, 0)); assert.equal(locks.held.has(lockName(nextId)), false); assert.equal(origin.children.has(prefix + nextId), false);
  assert.ok(locks.held.has(lockName(activeId))); active.clear([]); for (const operation of scheduled.splice(0)) await operation();
  await new Promise(resolve => setTimeout(resolve, 0)); assert.equal(locks.held.has(lockName(activeId)), false);
}
{
  const origin = new Directory(), id = "88888888-8888-4888-8888-888888888888", name = ".zebsdsolve-downloads-" + id;
  const stale = await origin.getDirectoryHandle(name, { create: true }), stream = await (await stale.getFileHandle(".owner.json", { create: true })).createWritable();
  await stream.write(JSON.stringify({ magic: "zebsdsolve-downloads", version: 1, id, lockProtected: true, created: 0, lastUsed: 0 })); await stream.close();
  const adapter = new BatchDownloads({ getDirectory: async () => origin, locks: null, now: () => 1e9, staleAfterMs: 1, randomId: () => "99999999-9999-4999-8999-999999999999" });
  await adapter.directory(); assert.equal(origin.children.has(name), true, "cross-session cleanup is skipped without Web Locks");
}
{
  const origin = new Directory(), locks = new Locks(), scheduled = [], id = "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa", nextId = "bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb";
  const active = new BatchDownloads({ getDirectory: async () => origin, locks: null, now: () => 0, randomId: () => id, scheduleCleanup: operation => scheduled.push(operation) });
  await active.directory(); const name = ".zebsdsolve-downloads-" + id;
  assert.equal(JSON.parse(await origin.children.get(name).children.get(".owner.json").data.text()).lockProtected, false);
  const next = new BatchDownloads({ getDirectory: async () => origin, locks, now: () => 2000, staleAfterMs: 1000, randomId: () => nextId, scheduleCleanup: operation => scheduled.push(operation) });
  await next.directory(); assert.equal(origin.children.has(name), true, "a later lock-capable context cannot reap an active no-lock workspace");
  active.clear([]); next.clear([]); for (const operation of scheduled.splice(0)) await operation();
  assert.equal(origin.children.has(name), false, "the creating session can explicitly clear its own no-lock workspace");
}
for (const failure of ["marker", "remove"]) {
  const origin = new Directory(), locks = new Locks(), scheduled = [], id = "cccccccc-cccc-4ccc-8ccc-cccccccccccc", name = ".zebsdsolve-downloads-" + id;
  const adapter = new BatchDownloads({ getDirectory: async () => origin, locks, now: () => 0, randomId: () => id, scheduleCleanup: operation => scheduled.push(operation) });
  await adapter.directory(); const marker = origin.children.get(name).children.get(".owner.json"), originalGetFile = marker.getFile;
  if (failure === "marker") marker.getFile = async () => { throw new Error("marker temporarily unreadable"); };
  else origin.fault = (path, stage) => { if (path === "/" + name && stage === "remove") throw new Error("remove temporarily unavailable"); };
  adapter.clear([]); for (const operation of scheduled.splice(0)) await operation(); await new Promise(resolve => setTimeout(resolve, 0));
  assert.equal(locks.held.has("zebsdsolve-downloads-workspace:" + id), false, "failed cleanup always releases the lifetime lease");
  marker.getFile = originalGetFile; origin.fault = null;
  const next = new BatchDownloads({ getDirectory: async () => origin, locks, now: () => 2000, staleAfterMs: 1000, randomId: () => "dddddddd-dddd-4ddd-8ddd-dddddddddddd", scheduleCleanup: operation => scheduled.push(operation) });
  await next.directory(); assert.equal(origin.children.has(name), false, "a later visit can reclaim the abandoned cache after cleanup recovers");
  next.clear([]); for (const operation of scheduled.splice(0)) await operation();
}
{
  const bytes = new TextEncoder().encode("owned ZIP payload"), original = bytes.slice(); let accessible = true;
  const source = { size: bytes.length, slice(start, end) {
    if (!accessible) throw new Error("source file was removed");
    return { arrayBuffer: async () => bytes.slice(start, end).buffer };
  } };
  const zip = await createStoreZip([{ name: "job/result.h5", blob: source }], { chunkSize: 4, yieldControl: async () => {} });
  bytes.fill(0); accessible = false;
  assert.deepEqual((await readZip(zip))[0].bytes, original, "ZIP payload owns copied chunks, not source-file references");
}
{
  const blob = new Blob(["123456789"]); blob.lastModified = new Date(1800, 0, 1).getTime();
  const entries = await readZip(await createStoreZip([{ name: "漢🧪/CRC.txt", blob }], { chunkSize: 2, yieldControl: async () => {} }));
  assert.equal(entries[0].name, "漢🧪/CRC.txt"); assert.equal(entries[0].crc, 0xcbf43926); assert.equal(entries[0].date, 0x21);
  blob.lastModified = new Date(2200, 0, 1).getTime(); const future = (await readZip(await createStoreZip([{ name: "future", blob }])))[0];
  assert.equal(future.date >>> 9, 127); assert.equal(future.time, 23 << 11 | 59 << 5 | 29);
}
{
  const blob = new Blob([new Uint8Array(2 * 1024 * 1024 + 17)]), slices = [], nativeSlice = blob.slice.bind(blob); let yields = 0;
  blob.arrayBuffer = () => { throw new Error("Do not materialize a whole ZIP payload."); };
  blob.slice = (start, end) => { slices.push([start, end]); return nativeSlice(start, end); };
  await createStoreZip([{ name: "big", blob }], { yieldControl: async () => { yields++; } });
  assert.equal(yields, 3); assert.equal(slices.length, 3); assert.ok(slices.every(([start, end]) => end - start <= 1024 * 1024));
}
for (const size of [0xffffffff, 0xfffffffe, 0xffffffff - 100]) {
  let reads = 0; const blob = { size, slice() { reads++; throw new Error("oversize read"); } };
  const members = size === 0xffffffff - 100 ? [{ name: "a", blob }, { name: "b", blob }] : [{ name: "a", blob }];
  await assert.rejects(createStoreZip(members), /ZIP32 limits/); assert.equal(reads, 0);
}
await assert.rejects(createStoreZip([{ name: "../unsafe", blob: new Blob() }]), /Invalid ZIP member path/);
console.log("Downloads transport: private staging, five-member ZIPs, persistent URLs, retries, cleanup, CRC/UTF-8/chunk/ZIP32 checks passed.");
