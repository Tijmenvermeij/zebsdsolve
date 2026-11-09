// Device loss must not let native temporary memory outlive pending readback writes.
import assert from "node:assert/strict";
import { Zebsdsolve } from "../zebsdsolve.js";

const deferred = () => {
  let resolve, reject;
  const promise = new Promise((a, b) => { resolve = a; reject = b; });
  return { promise, resolve, reject };
};
const tick = async () => { for (let i = 0; i < 12; i++) await Promise.resolve(); };
const originals = new Map();
const replace = (key, value, target = globalThis) => {
  originals.set([target, key], Object.getOwnPropertyDescriptor(target, key));
  Object.defineProperty(target, key, { configurable: true, writable: true, value });
};
const memory = new WebAssembly.Memory({ initial: 1 });
const buffers = [], loss = deferred();
let uncaptured, imports, next = 4096, frees = 0, nativeError = "";
const adapterInfo = { vendor: "NVIDIA", architecture: "turing", device: "RTX 4000", description: "Mock adapter" };
const device = {
  lost: loss.promise,
  queue: { submit() {} },
  addEventListener(kind, fn) { if (kind === "uncapturederror") uncaptured = fn; },
  createCommandEncoder() { return { copyBufferToBuffer() {}, finish() { return {}; } }; },
  createBuffer({ size }) {
    const mapping = deferred();
    const b = { mapping, data: new Uint8Array(size), destroyed: 0, unmapped: 0,
      mapAsync: () => mapping.promise, getMappedRange() { return this.data.buffer; },
      unmap() { this.unmapped++; }, destroy() { this.destroyed++; } };
    buffers.push(b);
    return b;
  },
};
const encode = new TextEncoder();
const snapshot = encode.encode('{"completed":17}');
const exports = {
  memory, ze_create: () => 1,
  ze_alloc(n) { const p = next; next += Math.ceil(n / 8) * 8; return p; },
  ze_free() { frees++; },
  ze_error(e, p, cap) {
    if (!nativeError) return imports.zegpu.takeError(p, cap);
    const bytes = encode.encode(nativeError).subarray(0, cap);
    new Uint8Array(memory.buffer, p, bytes.length).set(bytes);
    nativeError = "";
    return bytes.length;
  },
  ze_export_snapshot(e, frame, p, cap) {
    if (p) new Uint8Array(memory.buffer, p, cap).set(snapshot);
    return BigInt(snapshot.length);
  },
};

try {
  replace("GPUBufferUsage", { MAP_READ: 1, COPY_DST: 2, COPY_SRC: 4, STORAGE: 8, INDIRECT: 16, UNIFORM: 32 });
  replace("GPUMapMode", { READ: 1 });
  replace("location", { href: "https://example.test/" });
  replace("navigator", { gpu: { requestAdapter: async () => ({ info: adapterInfo, limits: {}, features: new Set(), requestDevice: async () => device }) } });
  replace("fetch", async () => ({ arrayBuffer: async () => new ArrayBuffer(0) }));
  replace("Suspending", function (fn) { return fn; }, WebAssembly);
  replace("promising", (fn) => fn, WebAssembly);
  replace("instantiate", async (bytes, value) => { imports = value; return { instance: { exports } }; }, WebAssembly);
  const z = await Zebsdsolve.create({ wasm: "https://example.test/test.wasm", warm: false });
  assert.equal(z.deviceLost, null);
  assert.deepEqual(z.adapterInfo, adapterInfo);
  assert(Object.isFrozen(z.adapterInfo));
  adapterInfo.vendor = "changed";
  assert.equal(z.adapterInfo.vendor, "NVIDIA");

  const gpu = imports.zegpu;
  const src = gpu.createBuffer(16, 0);
  let nativeReturned = false, frameFreed = false, freedAt;
  const operation = z._serial(() => z._frame(async (f) => {
    f.alloc(8);
    gpu.readBuffer(src, 0, 4, 128);
    gpu.readBuffer(src, 4, 4, 132);
    gpu.submit();
    const status = await gpu.wait();
    nativeReturned = true;
    assert.equal(status, 1);
    nativeError = "index_step: ReadFailed";
    z._check(-1, "index");
  })).finally(() => { frameFreed = true; freedAt = frees; });
  const failure = assert.rejects(operation, /index_step: ReadFailed.*first map failed/);
  let queuedRan = false;
  const queued = z._serial(() => { queuedRan = true; });
  const queuedFailure = assert.rejects(queued, (e) => e.name === "GPUDeviceLost" && /unknown.*driver reset/.test(e.message));
  await tick();
  assert.equal(buffers.length, 3);
  buffers[1].mapping.reject(new Error("first map failed"));
  await tick();
  assert.equal(buffers[1].destroyed, 1);
  assert.equal(buffers[2].destroyed, 0);
  assert.equal(nativeReturned, false);
  assert.equal(frameFreed, false);

  const events = [];
  z.onEvent((ev) => events.push(ev));
  loss.resolve({ reason: "unknown", message: "driver reset" });
  await tick();
  assert.deepEqual(z.deviceLost, { reason: "unknown", message: "driver reset" });
  assert(Object.isFrozen(z.deviceLost));
  assert.deepEqual(events, [{ kind: "deviceLost", reason: "unknown", message: "driver reset" }]);
  assert.equal(nativeReturned, false, "device.lost must not race the native call");
  assert.equal(frameFreed, false);
  buffers[2].data.set([4, 3, 2, 1]);
  buffers[2].mapping.resolve();
  await failure;
  await queuedFailure;
  assert.equal(queuedRan, false, "an already queued call must check loss when it starts");
  assert.equal(frameFreed, true);
  assert.equal(buffers[2].destroyed, 1);
  assert.equal(buffers[2].unmapped, 1);
  assert.deepEqual(Array.from(new Uint8Array(memory.buffer, 132, 4)), [4, 3, 2, 1]);
  new Uint8Array(memory.buffer, 132, 4).fill(9);
  await tick();
  assert.equal(frees, freedAt);
  assert.deepEqual(Array.from(new Uint8Array(memory.buffer, 132, 4)), [9, 9, 9, 9], "no late writes after native unwind");

  uncaptured({ error: { message: "late lost-device validation error" } });
  assert.deepEqual(z.exportSnapshot(), { completed: 17 }, "CPU export must survive residual GPU errors");
  nativeError = "export_snapshot: MissingPhaseLattice";
  assert.throws(() => z.exportSnapshot(), /MissingPhaseLattice/, "real native CPU errors must remain visible");
  await assert.rejects(z._serial(() => assert.fail("GPU work after loss")), { name: "GPUDeviceLost" });
  assert.equal(events.length, 1);

  // Native failure arrives before device.lost: consume both its own error and adapter detail
  // with that operation, leaving the next CPU call clean without changing success checks.
  z.state.deviceLoss = null;
  z.state.errors.push("adapter failure before loss notification");
  nativeError = "ReadFailed";
  assert.throws(() => z._check(-1, "index"), /ReadFailed.*adapter failure before loss notification/);
  assert.deepEqual(z.exportSnapshot(), { completed: 17 });
  z.state.errors.push("validation failure");
  assert.throws(() => z._check(0, "normal call"), /validation failure/);

  const supplied = await Zebsdsolve.create({ wasm: "https://example.test/test.wasm", device, warm: false });
  assert.deepEqual(supplied.adapterInfo, {}, "a supplied device must not invent adapter identity");

  // Tuning normally reports failed trials and keeps searching. A fatal loss during the
  // suspended trial must instead stop before any report, retry or progress callback.
  for (const outcome of [-1, 0, "throw"]) {
    const tuneLoss = deferred();
    const tuneDevice = { ...device, lost: tuneLoss.promise };
    let trials = 0, reports = 0, progress = 0;
    exports.ze_tune_begin = () => 0;
    exports.ze_tune_report = () => { reports++; return 1; };
    exports.ze_tune_trial = async () => {
      assert.equal(++trials, 1, "no trial may follow a fatal loss");
      tuneLoss.resolve({ reason: "unknown", message: "lost during tune trial" });
      await tick();
      nativeError = "ReadFailed during tuning";
      if (outcome === "throw") throw new Error("mapping failed");
      return outcome;
    };
    const tuning = await Zebsdsolve.create({ wasm: "https://example.test/test.wasm", device: tuneDevice, warm: false });
    Object.defineProperty(tuning, "settings", { value: { batch: 64, refine_pixels: 64, wg_refine: 64, wg_fft: 64 } });
    await assert.rejects(tuning.tune(() => { progress++; }), (e) => e.name === "GPUDeviceLost" && /lost during tune trial.*ReadFailed during tuning/.test(e.message)
      && (outcome !== "throw" || e.cause?.message === "mapping failed"));
    assert.equal(trials, 1);
    assert.equal(reports, 0);
    assert.equal(progress, 0);
    assert.deepEqual(tuning.exportSnapshot(), { completed: 17 }, "the failed trial's native error must not contaminate CPU export");
  }
  const trialError = new Error("ordinary trial exception");
  exports.ze_tune_trial = async () => { throw trialError; };
  const ordinaryTune = await Zebsdsolve.create({ wasm: "https://example.test/test.wasm", device: { ...device, lost: deferred().promise }, warm: false });
  Object.defineProperty(ordinaryTune, "settings", { value: { batch: 64, refine_pixels: 64, wg_refine: 64, wg_fft: 64 } });
  await assert.rejects(ordinaryTune.tune(), (e) => e === trialError, "an exception without known loss must be preserved");
  console.log("GPU loss monitoring, readback lifetime, queue guard and CPU export checks passed.");
} finally {
  for (const [[target, key], descriptor] of [...originals].reverse()) {
    if (descriptor) Object.defineProperty(target, key, descriptor);
    else delete target[key];
  }
}
