import assert from 'node:assert/strict';
import { batchArtifacts } from '../app/js/batch-artifacts.js';

const pixels = new Uint8ClampedArray([255, 0, 2, 255, 0, 128, 7, 255]);
const calls = [];
class ImageDataStub {
  constructor(data, width, height) { Object.assign(this, { data, width, height }); }
}
const context = { putImageData(image, x, y) { assert.deepEqual(image.data, pixels); assert.equal(image.width, 2); assert.equal(image.height, 1); assert.deepEqual([x, y], [0, 0]); calls.push('pixels'); } };
globalThis.ImageData = ImageDataStub;
globalThis.OffscreenCanvas = class {
  constructor(width, height) { assert.deepEqual([width, height], [2, 1]); }
  getContext(kind) { assert.equal(kind, '2d'); return context; }
  async convertToBlob(options) { assert.deepEqual(options, { type: 'image/png' }); return new Blob(['encoded'], options); }
};
const engine = { scan: { rows: 1, cols: 2 }, map(kind) { calls.push(kind); return pixels; } };
const job = { outputStem: 'scan_02', scanInfo: { format: 'test' } };
const completed = new Set(['scan_02.h5', 'scan_02.ang', 'scan_02_IPF-Z.png']);
let artifacts = [];
for await (const artifact of batchArtifacts(engine, job, completed)) artifacts.push(artifact);
assert.deepEqual(artifacts.map(item => item.name), ['scan_02_NCC.png']);
assert.deepEqual(calls, ['ncc', 'pixels']);
assert.equal(artifacts[0].data.type, 'image/png');
completed.add('scan_02_NCC.png');
for await (const artifact of batchArtifacts(engine, job, completed)) assert.fail('Completed artifact was regenerated');
assert.deepEqual(calls, ['ncc', 'pixels']);
// Exercise HTMLCanvasElement's asynchronous encoder fallback as well.
delete globalThis.OffscreenCanvas;
globalThis.document = { createElement(kind) {
  assert.equal(kind, 'canvas');
  return { getContext() { assert.deepEqual([this.width, this.height], [2, 1]); return context; }, toBlob(callback, type) { callback(new Blob(['fallback'], { type })); } };
} };
completed.delete('scan_02_IPF-Z.png');
artifacts = [];
for await (const artifact of batchArtifacts(engine, job, completed)) artifacts.push(artifact);
assert.deepEqual(artifacts.map(item => item.name), ['scan_02_IPF-Z.png']);
assert.equal(artifacts[0].data.type, 'image/png');
console.log('Batch PNG artifacts: native RGBA dimensions, canvas paths and completed-map retries passed.');
