// The adapter's synchronous native ABI, including i64 return values and f64 frame storage.
import assert from "node:assert/strict";
import { Zebsdsolve } from "../zebsdsolve.js";
import { geometryConvention, acceptsGeometryConvention } from "../app/js/oxford-export-policy.js";
const memory = new WebAssembly.Memory({ initial: 1 }), bytes = new TextEncoder().encode('{"schema_version":1,"value":3}');
let next = 64, calls = [];
const x = {
  ze_create: () => 1, ze_alloc: (n) => { const p = next; next += Math.ceil(n / 8) * 8; return p; }, ze_free: () => {}, ze_error: () => 0,
  ze_export_snapshot: (e, frame, p, cap) => {
    assert.equal(e, 1); calls.push(frame ? Array.from(new Float64Array(memory.buffer, frame, 4)) : null);
    if (p) { assert.equal(cap, bytes.length); new Uint8Array(memory.buffer, p, cap).set(bytes); }
    return BigInt(bytes.length);
  },
};
x.ze_export_oxford_snapshot = x.ze_export_snapshot;
const z = new Zebsdsolve(x, { memory });
assert.deepEqual(z.exportSnapshot(), { schema_version: 1, value: 3 });
assert.deepEqual(z.exportSnapshot([1, 0, 0, 0]), { schema_version: 1, value: 3 });
assert.deepEqual(z.exportOxfordSnapshot(), { schema_version: 1, value: 3 });
assert.deepEqual(z.exportOxfordSnapshot([1, 0, 0, 0]), { schema_version: 1, value: 3 });
assert.deepEqual(calls, [null, null, [1, 0, 0, 0], [1, 0, 0, 0], null, null, [1, 0, 0, 0], [1, 0, 0, 0]]);
assert.throws(() => z.exportOxfordSnapshot([1, 0, 0]), /quaternion/);
assert.throws(() => z.exportSnapshot([1, 0, 0]), /quaternion/);
console.log("Browser export snapshot ABI checks passed.");

// A project fitted with the former fallback pose must not restore onto S05 calibration.
const oxford = { format: "Oxford h5oina" };
assert.equal(geometryConvention({ ...oxford, geometryFrame: 1 }), "oxford-calibrated-v1");
assert.equal(geometryConvention({ ...oxford, geometryFrame: 2 }), "oxford-calibrated-v3");
assert.notEqual(geometryConvention({ ...oxford, geometryFrame: 2 }), "oxford-calibrated-v1");
assert.equal(geometryConvention({ ...oxford, geometryFrame: 3 }), "oxford-fallback-v1");
assert.equal(geometryConvention({ format: "EDAX up1/up2", geometryFrame: 0 }), "generic-v1");

for (const [frame, saved, accepted] of [
  [1, "oxford-calibrated-v1", true],
  [2, "oxford-calibrated-v1", false],
  [3, "oxford-calibrated-v1", true],
  [2, "oxford-calibrated-v2", false],
  [2, "oxford-calibrated-v3", true],
]) assert.equal(acceptsGeometryConvention({ ...oxford, geometryFrame: frame }, saved), accepted);
