// Hold a file-backed operation at its read to verify close and disposal order.
import assert from "node:assert/strict";
import { Zebsdsolve } from "../zebsdsolve.js";

function deferred() {
  let resolve;
  const promise = new Promise((r) => { resolve = r; });
  return { promise, resolve };
}

function engine() {
  const memory = new WebAssembly.Memory({ initial: 1 });
  const state = { memory, files: [null] };
  const entered = deferred(), release = deferred();
  const log = [];
  let next = 4096, fileId = 0;
  const x = {
    ze_create: () => 1,
    ze_alloc(n) { const p = next; next += Math.ceil(n / 8) * 8; return p; },
    ze_free() { log.push("free"); },
    ze_error: () => 0,
    ze_open_scan(e, p, n) {
      const token = new TextDecoder().decode(new Uint8Array(memory.buffer, p, n));
      assert.match(token, /^@\d+:file$/);
      fileId = Number(token.slice(1, token.indexOf(":")));
      return 0;
    },
    ze_pattern_shape(e, full, p) {
      new Uint32Array(memory.buffer, p, 2).set([1, 1]);
      return 0;
    },
    ze_nlpar_view: () => 0,
    async ze_pattern(e, cell, stage, full, p) {
      entered.resolve();
      await release.promise;
      const bytes = await state.files[fileId].arrayBuffer();
      new Float32Array(memory.buffer, p, 1)[0] = new Uint8Array(bytes)[0];
      log.push("read");
      return 0;
    },
    ze_close_scan() { log.push("close"); fileId = 0; },
    ze_destroy() { log.push("destroy"); },
  };
  const ze = new Zebsdsolve(x, state);
  Object.defineProperty(ze, "scan", { get: () => fileId });
  return { ze, entered, release, log };
}

{
  const { ze, entered, release, log } = engine();
  await ze.openScan(new Blob([Uint8Array.of(17)]));
  const preview = ze.pattern(0, "stored");
  await entered.promise;
  const closed = ze.closeScan();
  await Promise.resolve();
  assert.equal(log.includes("close"), false);
  release.resolve();
  assert.equal((await preview)[0], 17);
  await closed;
  assert.deepEqual(log.slice(-2), ["free", "close"]);
  assert.equal(ze.scan, 0);
  await ze.dispose();
}

{
  const { ze, entered, release, log } = engine();
  await ze.openScan(new Blob([Uint8Array.of(23)]));
  const preview = ze.pattern(0, "stored");
  await entered.promise;
  const rejected = assert.rejects(preview, /disposed/);
  const disposed = ze.dispose();
  await Promise.resolve();
  assert.equal(log.includes("destroy"), false);
  release.resolve();
  await rejected;
  await disposed;
  assert.deepEqual(log.slice(-3), ["read", "free", "destroy"]);
  await assert.rejects(ze.pattern(0, "stored"), /disposed/);
}
console.log("Queued scan close and engine disposal wait for suspended file reads.");
