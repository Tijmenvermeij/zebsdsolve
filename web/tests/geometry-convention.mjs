import assert from 'node:assert/strict';
import {geometryConvention, acceptsGeometryConvention} from '../app/js/oxford-export-policy.js';

for (const format of ['Oxford h5oina', 'h5_oxford']) {
  const calibrated = {format, geometryFrame:2};
  assert.equal(geometryConvention(calibrated), 'oxford-calibrated-v3');
  assert.ok(acceptsGeometryConvention(calibrated, 'oxford-calibrated-v3'));
  for (const old of [undefined, 'oxford-calibrated-v1', 'oxford-calibrated-v2', 'generic-v1', 'oxford-fallback-v1']) {
    assert.ok(!acceptsGeometryConvention(calibrated, old));
  }
  const fallback = {format, geometryFrame:3};
  assert.equal(geometryConvention(fallback), 'oxford-fallback-v1');
  assert.ok(acceptsGeometryConvention(fallback, 'oxford-fallback-v1'));
  assert.ok(acceptsGeometryConvention(fallback, 'oxford-calibrated-v1'));
  assert.ok(!acceptsGeometryConvention(fallback, 'oxford-calibrated-v3'));
}
const generic = {format:'EMsoft H5', geometryFrame:0};
assert.equal(geometryConvention(generic), 'generic-v1');
assert.ok(acceptsGeometryConvention(generic, undefined));
for (const saved of ['oxford-calibrated-v1','oxford-calibrated-v2','oxford-calibrated-v3','oxford-unversioned','oxford-fallback-v1']) {
  assert.ok(!acceptsGeometryConvention(generic, saved));
}
console.log('Published Oxford geometry markers, fallback migration, and cross-format guards pass');
