// Native Engine.regionCells ROI semantics, including either-zero = full scan.
import assert from 'node:assert/strict';
import {roiCells} from '../app/js/overlap.js';
const scan={cols:5,rows:2};
const cells=options=>Array.from(roiCells(scan,{roi_x:0,roi_y:0,roi_w:0,roi_h:0,...options}));
assert.deepEqual(cells({}),[0,1,2,3,4,5,6,7,8,9]);
assert.deepEqual(cells({roi_x:3,roi_w:0,roi_h:1}),[0,1,2,3,4,5,6,7,8,9]);
assert.deepEqual(cells({roi_x:3,roi_w:1,roi_h:0}),[0,1,2,3,4,5,6,7,8,9]);
assert.deepEqual(cells({roi_x:1,roi_y:0,roi_w:2,roi_h:2}),[1,2,6,7]);
assert.deepEqual(cells({roi_x:4,roi_y:1,roi_w:9,roi_h:9}),[9]);
assert.deepEqual(cells({roi_x:99,roi_y:99,roi_w:1,roi_h:1}),[9]);
assert.deepEqual(Array.from(roiCells({cols:0,rows:0},{roi_w:1,roi_h:1})),[]);
console.log('Overlap ROI matches native selection');
