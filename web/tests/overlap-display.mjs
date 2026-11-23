// Controller acceptance and source selection with native operations stubbed; no crystallographic math.
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {createOverlap} from '../app/js/overlap.js';
class Canvas {
  constructor(){this.width=1;this.height=1;this.data=new Uint8ClampedArray(4);}
  getContext(){const canvas=this;return {createImageData:(w,h)=>({data:new Uint8ClampedArray(w*h*4)}),putImageData(image){canvas.data=image.data.slice();},drawImage(source){for(let i=0;i<source.data.length;i+=4)if(source.data[i+3])canvas.data.set(source.data.subarray(i,i+4),i);},clearRect(){canvas.data.fill(0);}};}
}
class Element extends Canvas {
  constructor(id){super();this.id=id;this.value='';this.checked=false;this.children=[];this.options=[];this.listeners={};this.title='';this.textContent='';this.hidden=false;this.dataset={};}
  get valueAsNumber(){return this.value===''?NaN:Number(this.value);}
  get selectedOptions(){return this.options.filter(o=>o.value===this.value);}
  addEventListener(name,fn){this.listeners[name]=fn;}
  replaceChildren(...children){this.children=children;if(this.options.length)this.options=children;}
  append(...children){this.children.push(...children);if(this.id.includes('filter-metric'))this.options.push(...children);}
  querySelector(selector){return selector==='option[data-weight]'?this.options.find(o=>o.dataset?.weight!==undefined):this.options.find(o=>o.value===/value="([^"]+)"/.exec(selector)?.[1]);}
}
globalThis.Option=class {constructor(text,value){this.textContent=text;this.value=value;this.dataset={};}};
globalThis.OffscreenCanvas=Canvas;
const html=readFileSync(new URL('../app/index.html',import.meta.url),'utf8'),elements=new Map();
for(const [,id] of html.matchAll(/id="([^"]+)"/g))elements.set(id,new Element(id));
for(const [,id,body] of html.matchAll(/<select[^>]*id="([^"]+)"[^>]*>([\s\S]*?)<\/select>/g)) {
  const e=elements.get(id);for(const [,attrs,text] of body.matchAll(/<option([^>]*)>(.*?)<\/option>/g)){const o=new Option(text,/value="([^"]*)"/.exec(attrs)?.[1]??text);e.options.push(o);if(!e.value||attrs.includes('selected'))e.value=o.value;}
}
for(const [,id,attrs] of html.matchAll(/<input[^>]*id="([^"]+)"([^>]*)>/g)){const e=elements.get(id);e.value=/value="([^"]*)"/.exec(attrs)?.[1]??'';e.checked=/\bchecked\b/.test(attrs);}
globalThis.document={getElementById:id=>elements.get(id),createElement:tag=>new Element(tag)};
const $=id=>elements.get(id),qs=new Float32Array([.7,0,0,0,.6,0,0,0,.5,0,0,0]);
const requests=[], requestOptions=[];
const seeds=Uint8Array.of(1,1,1);
const st={scan:{rows:1,cols:3},sourceFiles:[],phaseFiles:[{on:true}],settings:{batch:64,roi_x:0,roi_y:0,roi_w:2,roi_h:1},overview:new Float32Array([.1,.2,.3]),results:{phase:new Int32Array([0,0,0]),ncc:new Float32Array([.8,.2,.9]),quaternions:qs},cell:-1,showRun:0};
st.ze={
  settings:st.settings,geometry:{},phases:[{name:'phase'}],version:'stub',
  storedPrimaryValidity(first=0,count=seeds.length-first){return seeds.slice(first,first+count);},
  orientationColors(q,p){return Uint8ClampedArray.from(Array.from(p).flatMap((phase,i)=>[Math.round(q[4*i]*100),40,80,phase>=0?255:0]));},
  async overlapCells(cells, options){
    requests.push(Array.from(cells)); requestOptions.push(structuredClone(options));
    const support = (Type, multiple, singleton) => Type.from(Array.from(cells).flatMap(cell=>cell===0?multiple:singleton));
    return {
      max_components:2,cells,stage_gate:{minimum:options.minStageImprovement},
      stop_reason:support(Uint32Array,[13],[1]),
      phase:support(Int32Array,[0,0],[0,-1]),
      quaternions:support(Float32Array,[.2,0,0,0,.9,0,0,0],[.9,0,0,0,0,0,0,0]),
      source_quaternions:support(Float32Array,[.3,0,0,0,.1,0,0,0],[.8,0,0,0,0,0,0,0]),
      winner_rank:support(Int32Array,[1,0],[1,-1]),
      amplitudes:support(Float32Array,[.4,1],[1,0]),
      proposal_ncc:support(Float32Array,[.5,.8],[.8,NaN]),
      prepared_fractions:support(Float32Array,[.3,.7],[1,NaN]),
      retained_order:support(Uint32Array,[2],[1]),active_count:support(Uint32Array,[2],[1]),
      objective_valid:support(Uint32Array,[1],[1]),
    };
  },
};
let ui;const redraw=()=>{ui.drawMap($('map'),0);ui.drawMap($('overlap-map-canvas'),1);};
ui=createOverlap({st,busy:async(_,fn)=>{st.busy=true;try{await fn();}finally{st.busy=false;}},runPrimary:async()=>true,selectTab(){},inspectPrimary:async()=>false,redraw,status(message,error){if(error)throw Error(message);},log(){}});
ui.sync();redraw();
assert.equal($('overlap-map-component').value,'primary');assert.equal($('overlap-map-component-2').value,'residual');
assert.equal($('map').data[0],70);assert.equal($('map').data[8],50,'primary outside overlap ROI visible before overlap');
$('overlap-polish').checked=true;
$('overlap-primary-cutoff').value='.5';await $('btn-overlap-run').listeners.click();
assert.deepEqual(requests,[[0]],'next overlap eligibility intersects primary acceptance and ROI');
assert.equal(ui.result.metadata.primary_ncc_minimum,.5);assert.equal($('map').data[8],50,'outside ROI primary survives overlap');
assert.equal($('overlap-map-canvas').data[0],30,'residual map uses the moved residual’s native source pose');
const fitSource=$('overlap-map-component-2');
fitSource.value='1';redraw();
assert.equal($('overlap-map-canvas').data[0],20,'numbered map uses the final fitted orientation');
fitSource.value='2';redraw();
assert.equal($('overlap-map-canvas').data[0],90,'numbered map includes the retained original primary');
fitSource.value='residual';redraw();

assert.match(ui.hover(0,1),/Fit component 1/);
ui.inspectMap(1,0);assert.equal($('overlap-component').value,'1','automatic source resolves to the clicked fit slot');
const residual=$('overlap-map-canvas').data.slice();
$('overlap-primary-cutoff').value='.95';$('overlap-primary-cutoff').listeners.input();
assert.notEqual($('map').data[0],70);
assert.deepEqual($('overlap-map-canvas').data,residual,'future-run primary cutoff leaves historical residual map intact');
$('overlap-primary-cutoff').value='.5';$('overlap-primary-cutoff').listeners.input();
const original=$('map').data.slice();$('overlap-residual-cutoff').value='.6';$('overlap-residual-cutoff').listeners.input();
assert.deepEqual($('map').data,original,'residual cutoff never masks original primary');
assert.notEqual($('overlap-map-canvas').data[0],30);assert.ok(ui.result.context.includes('primary_snapshot'));
assert.ok($('overlap-readout').textContent.startsWith('Complete'),'display cutoff preserves fitted context');
$('overlap-primary-cutoff').value='0';$('overlap-primary-cutoff').listeners.input();
assert.equal($('map').data[4],60);assert.equal(ui.result.metadata.primary_ncc_minimum,.5,'execution cutoff remains frozen');
assert.ok($('overlap-readout').textContent.startsWith('Complete'));
st.results={...st.results};ui.sync();redraw();assert.ok($('overlap-readout').textContent.startsWith('Context changed'),'new primary snapshot invalidates prior overlap at same geometry');
assert.equal($('map').data[8],50);
st.ze.geometry={pc:[.4,.5,.6]};
seeds[2]=0;
st.settings.roi_w=0;st.settings.roi_h=0;
$('overlap-residual-cutoff').value='0';$('overlap-residual-cutoff').listeners.input();
assert.notEqual($('map').data[8],50,'stale outside-ROI primary is hidden');
assert.equal(ui.hover(2),'Primary unavailable');
await $('btn-overlap-run').listeners.click();
assert.deepEqual(requests,[[0],[0,1]],'next run excludes the same stale cell even when the new ROI includes it');
assert.equal(ui.hover(1,1),'No multi-component fit');
assert.notEqual($('overlap-map-canvas').data[4],80,'replacement singleton stays off the residual map');
ui.inspectMap(1,1);assert.equal($('overlap-component').value,'primary');
$('overlap-map-component-2').value='1';redraw();
assert.equal($('overlap-map-canvas').data[4],90,'numbered map includes a singleton fit');
assert.match(ui.hover(1,1),/Fit component 1/);
$('overlap-residual-cutoff').value='1';$('overlap-residual-cutoff').listeners.input();
assert.equal($('overlap-map-canvas').data[4],90,'residual NCC cutoff only filters the residual-source map');
$('overlap-residual-cutoff').value='0';$('overlap-residual-cutoff').listeners.input();
ui.inspectMap(1,1);assert.equal($('overlap-component').value,'1','explicit fit slot remains inspectable');
console.log('Overlap controller: native primary eligibility, residual provenance, singleton masking, independent cutoffs and frozen results pass');

// A skipped or failed native primary render must never copy a previous cell's canvases.
st.cell=0;$('overlap-component').value='primary';$('pat-exp').width=9;$('pat-exp').height=9;
await ui.showCell(0);
assert.equal($('overlap-target').width,1);
assert.ok($('overlap-target').data.every(v=>v===0));
console.log('Skipped primary renders leave the inspector clear');

// Computational stage gates are opt-in, frozen in the request and copied to export metadata.
assert.equal(requestOptions[0].minStageImprovement,null);
assert.equal($('overlap-stage-3-row').hidden,true);
$('overlap-stage-2').value='0.01';
$('overlap-screen').checked=false;
await assert.rejects($('btn-overlap-run').listeners.click(),/Enable Screen residual NCC/);
$('overlap-screen').checked=true;$('overlap-screen').listeners.change();
$('overlap-stage-2').value='0.01';$('overlap-stage-2').listeners.change();
await $('btn-overlap-run').listeners.click();
assert.deepEqual(requestOptions.at(-1).minStageImprovement,[0.01,null,null]);
assert.deepEqual(ui.result.metadata.requested_min_stage_improvement,[0.01,null,null]);
assert.deepEqual(ui.result.metadata.stage_gate.minimum,[0.01,null,null]);
assert.equal(ui.result.metadata.stop_reasons[13],'Insufficient ΔSSE');
$('overlap-stage-2').value='0.02';$('overlap-stage-2').listeners.change();
assert.ok($('overlap-readout').textContent.startsWith('Context changed'));
assert.deepEqual(ui.result.metadata.stage_gate.minimum,[0.01,null,null]);
$('overlap-limit').value='4';$('overlap-limit').listeners.change();
assert.equal($('overlap-stage-3-row').hidden,false);
await assert.rejects($('btn-overlap-run').listeners.click(),/Fill every ΔSSE threshold/);
console.log('Overlap stage thresholds: opt-in, request, frozen metadata, stop label and context pass');
