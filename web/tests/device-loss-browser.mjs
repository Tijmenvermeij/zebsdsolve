#!/usr/bin/env node
// Inject real WebGPU loss while indexing; completed CPU results remain exportable.
// Node 22+: node web/tests/device-loss-browser.mjs [active|between]
import assert from "node:assert/strict";
import { spawn } from "node:child_process";
import { createServer } from "node:http";
import { createReadStream, existsSync, mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { extname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
const root = resolve(fileURLToPath(new URL("../..", import.meta.url)));
const mode = process.argv[2] || "active";
assert.ok(["active", "between"].includes(mode));
const chromePath = process.env.CHROME || ["/Applications/Google Chrome.app/Contents/MacOS/Google Chrome", "/usr/bin/google-chrome", "/usr/bin/chromium"].find(existsSync);
assert.ok(chromePath, "Chrome is required; set CHROME");
const server = createServer((req, res) => {
  let path = decodeURIComponent(req.url.split("?")[0]);
  for (const [from,to] of [["/web/app/zebsdsolve/zebsdsolve.js","/web/zebsdsolve.js"],["/web/app/zebsdsolve/","/zig-out/web/"],["/web/app/presets/nickel/","/testdata/nickel/"]]) {
    if (path.startsWith(from)) { path = to + path.slice(from.length); break; }
  }
  if (path.endsWith("/")) path += "index.html";
  const file = resolve(root, `.${path}`);
  if (!file.startsWith(`${root}/`) || !existsSync(file)) { res.writeHead(404).end(); return; }
  res.writeHead(200, { "Content-Type": ({ ".js": "text/javascript", ".wasm": "application/wasm", ".html": "text/html", ".css": "text/css" })[extname(file)] || "application/octet-stream" });
  createReadStream(file).pipe(res);
});
const profile = mkdtempSync(join(tmpdir(), "device-loss-chrome-"));
let chrome, ws;
const wallTimeout = setTimeout(() => {
  console.error('Device-loss browser test exceeded 180 seconds');
  clearTimeout(wallTimeout);ws?.close();chrome?.kill();server.closeAllConnections();server.close();process.exit(1);
},180000);
try {
  await new Promise((resolve, reject) => { server.once("error", reject); server.listen(0, "127.0.0.1", resolve); });
  const base = `http://127.0.0.1:${server.address().port}`;
  chrome = spawn(chromePath, ["--headless=new", "--remote-debugging-port=0", `--user-data-dir=${profile}`, "--no-first-run", "--no-default-browser-check", "--enable-unsafe-webgpu", "--ignore-gpu-blocklist", "--enable-features=Vulkan", "--window-size=1500,900", "about:blank"], { stdio: ["ignore", "ignore", "pipe"] });
  const wsURL = await new Promise((resolve, reject) => {
    let stderr = "";
    const timer = setTimeout(() => reject(new Error(`Chrome did not start: ${stderr.slice(-1000)}`)), 30000);
    chrome.once("error", reject);
    chrome.once("exit", (code) => { clearTimeout(timer); reject(new Error(`Chrome exited ${code}: ${stderr.slice(-1000)}`)); });
    chrome.stderr.on("data", (data) => { stderr += data; const match = /DevTools listening on (ws:\/\/\S+)/.exec(stderr); if (match) { clearTimeout(timer); resolve(match[1]); } });
  });
  ws = new WebSocket(wsURL);
  await new Promise((resolve, reject) => { ws.onopen = resolve; ws.onerror = reject; });
  let serial = 0, pageLoaded;
  const pending = new Map();
  ws.onmessage = ({ data }) => {
    const message = JSON.parse(data);
    if (message.method === "Page.loadEventFired") pageLoaded?.();
    const request = pending.get(message.id);
    if (!request) return;
    pending.delete(message.id);
    message.error ? request.reject(new Error(message.error.message)) : request.resolve(message.result);
  };
  function send(method, params = {}, sessionId) {
    return new Promise((resolve, reject) => { const id = ++serial; pending.set(id, { resolve, reject }); ws.send(JSON.stringify({ id, method, params, ...(sessionId ? { sessionId } : {}) })); });
  }
  const { targetId } = await send("Target.createTarget", { url: "about:blank" });
  const { sessionId } = await send("Target.attachToTarget", { targetId, flatten: true });
  async function evaluate(expression) {
    const result = await send("Runtime.evaluate", { expression: `(async () => { ${expression} })()`, awaitPromise: true, returnByValue: true, timeout: 300000 }, sessionId);
    if (result.exceptionDetails) throw new Error(result.exceptionDetails.exception?.description || result.exceptionDetails.text);
    return result.result.value;
  }
  await send("Page.enable", {}, sessionId);
  const loaded = new Promise(resolve => { pageLoaded = resolve; });
  await send("Page.navigate", { url: `${base}/web/app/` }, sessionId);
  await Promise.race([loaded,new Promise((_,reject)=>setTimeout(()=>reject(new Error("Page load timed out")),30000).unref())]);
  console.error("device loss: app loaded");
  await evaluate(`
    for(let i=0;i<600 && !window.ze;i++) await new Promise(r=>setTimeout(r,100));
    if(!window.ze) throw Error('No WebGPU engine');
    document.getElementById('pat-live').checked=false;
    window.waitFor=async(p)=>{for(let i=0;i<1200&&!p();i++) await new Promise(r=>setTimeout(r,50));if(!p())throw Error('Condition timed out: '+document.getElementById('status').textContent);};
    document.getElementById('btn-example').click();
    await waitFor(()=>ze.phases.length===1&&document.getElementById('app').dataset.busy==='0');
    await ze._queue;
    ze.configure({static_background:'none',dynamic_background:'none',bandwidth:8,fit_bandwidth:8,master_smooth:0,refine_iters:1,batch:2,roi_x:0,roi_y:0,roi_w:4,roi_h:2});
    const {FIELDS}=await import('/web/app/js/fields.js');
    const i=FIELDS.indexmore.findIndex(f=>f.key==='batch');
    if(i<0)throw Error('Missing batch field');
    const field=document.querySelectorAll('.fields[data-group="indexmore"] .field')[i].querySelector('input');
    field.value='2';field.dispatchEvent(new Event('change'));
    await ze._queue;
    window.probe={steps:0,nativePatternCalls:0,nativeShapeCalls:0,mappings:0};
    const nativePattern=ze.x.ze_pattern,nativeShape=ze.x.ze_pattern_shape;
    ze.x.ze_pattern=(...args)=>{probe.nativePatternCalls++;return nativePattern(...args);};
    ze.x.ze_pattern_shape=(...args)=>{probe.nativeShapeCalls++;return nativeShape(...args);};
    ze.onEvent(ev=>{if(ev.kind==='deviceLost')probe.atLoss={patterns:probe.nativePatternCalls,shapes:probe.nativeShapeCalls};});
    const create=ze.device.createBuffer.bind(ze.device);
    ze.device.createBuffer=(descriptor)=>{
      const buffer=create(descriptor);
      if(descriptor.usage&GPUBufferUsage.MAP_READ){
        const map=buffer.mapAsync.bind(buffer);
        buffer.mapAsync=(...args)=>{
          probe.mappings++;
          const result=map(...args);
          if(window.killOnMapping){window.killOnMapping=false;ze.device.destroy();window.queuedPreview=ze.pattern(0,'stored').then(()=>{probe.preview='ran';},e=>{probe.preview=e.name;});}
          return result;
        };
      }
      return buffer;
    };
    const step=ze.indexStep.bind(ze);
    ze.indexStep=async()=>{
      if(++probe.steps===3){
        const r=ze.results();window.committed=Object.fromEntries(Object.entries(r).map(([k,v])=>[k,Array.from(v)]));
        if('${mode}'==='active')window.killOnMapping=true;
        else {ze.device.destroy();await ze.device.lost;await Promise.resolve();}
      }
      return step();
    };
    document.getElementById('btn-index').click();
    await waitFor(()=>!!ze.deviceLost&&document.getElementById('app').dataset.busy==='0');
    if(window.queuedPreview)await queuedPreview;
  `);
  console.error(`device loss: ${mode} operation settled`);
  const result=await evaluate(`
    const actual=ze.results();
    const same=Object.entries(committed).every(([key,value])=>value.every((v,i)=>Object.is(v,actual[key][i])));
    const snapshot=ze.exportSnapshot(), ang=ze.exportText('ang');
    const {exportH5}=await import('/web/app/js/hdf5-export.js');
    const bytes=await exportH5(snapshot);
    const h=await import('/web/app/js/vendor/h5wasm/hdf5_hl.js');await h.ready;
    h.FS.writeFile('/after-loss.h5',bytes);const f=new h.File('/after-loss.h5');
    const exported=Array.from(f.get('Scan 1/EBSD/Data/Phase').value).filter(v=>v!==255).length;
    f.close();h.FS.unlink('/after-loss.h5');
    await new Promise(r=>setTimeout(r,250));
    return {same,retained:actual.phase.filter(p=>p>=0).length,exported,ang:ang.length,h5:bytes.length,
      probe,loss:ze.deviceLost,status:document.getElementById('status').textContent,
      readout:document.getElementById('index-readout').textContent,
      indexDisabled:document.getElementById('btn-index').disabled,
      exportEnabled:!document.getElementById('btn-ang').disabled&&!document.getElementById('btn-h5').disabled,
      done:document.getElementById('tab-index').dataset.done,barHidden:document.getElementById('index-progress').hidden};
  `);
  assert.ok(result.same,'Every committed result field must survive loss unchanged');
  assert.equal(result.retained,4);assert.equal(result.exported,4);
  assert.ok(result.ang>100&&result.h5>1000&&result.exportEnabled);
  assert.ok(result.indexDisabled&&result.barHidden);assert.notEqual(result.done,'1');
  assert.match(result.status,/GPU device lost.*4 indexed results retained/);
  assert.match(result.readout,/Interrupted.*4 indexed results retained/);
  assert.doesNotMatch(result.status,/pattern_shape/);
  assert.equal(result.probe.nativePatternCalls,result.probe.atLoss.patterns);
  assert.equal(result.probe.nativeShapeCalls,result.probe.atLoss.shapes);
  if(mode==='active')assert.equal(result.probe.preview,'GPUDeviceLost');
  console.log(JSON.stringify(result));
  console.log(`Real WebGPU ${mode} loss: committed results unchanged, CPU H5/ANG export works, no subsequent GPU preview, original failure retained.`);
} finally {
  clearTimeout(wallTimeout);ws?.close();chrome?.kill();server.closeAllConnections();await new Promise(resolve=>server.close(resolve));
  try {rmSync(profile,{recursive:true,force:true});} catch {}
}
