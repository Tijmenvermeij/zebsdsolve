import assert from 'node:assert/strict';
import {geometryConvention,canRestoreGeometry} from '../app/js/oxford-export-policy.js';
for(const format of ['Oxford h5oina','h5_oxford']){
 assert.equal(geometryConvention(format,2),'oxford-calibrated-v2');
 assert.ok(canRestoreGeometry('oxford-calibrated-v2',format,2));
 for(const old of [undefined,'oxford-calibrated-v1','generic-v1','oxford-fallback-v1'])assert.ok(!canRestoreGeometry(old,format,2));
 assert.equal(geometryConvention(format,0),'oxford-fallback-v1');assert.ok(canRestoreGeometry('oxford-fallback-v1',format,0));
 assert.ok(!canRestoreGeometry('oxford-calibrated-v2',format,0));
 assert.equal(geometryConvention(format),'oxford-unversioned');assert.ok(!canRestoreGeometry('oxford-unversioned',format));
}
for(const oxford of ['oxford-calibrated-v1','oxford-calibrated-v2','oxford-unversioned','oxford-fallback-v1'])assert.ok(!canRestoreGeometry(oxford,'EMsoft H5',0));
assert.equal(geometryConvention('EMsoft H5',2),'generic-v1');assert.ok(canRestoreGeometry(undefined,'EMsoft H5',2));
console.log('geometry persistence: supported v2, explicit fallback, old fitted residuals blocked, generic unaffected');
