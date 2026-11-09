#!/usr/bin/env node
// Stored acquisition quality and opt-in contrast in the actual app, no npm.
// Node 22+: node web/tests/stored-overview-browser.mjs source.h5oina
import assert from "node:assert/strict";
import { spawn } from "node:child_process";
import { createServer } from "node:http";
import { createReadStream, existsSync, mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { extname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
const root = resolve(fileURLToPath(new URL("../..", import.meta.url))), acquisition = process.argv[2] && resolve(process.argv[2]);
assert.ok(acquisition && existsSync(acquisition), "Pass the original P6 Oxford acquisition path");
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
const profile = mkdtempSync(join(tmpdir(), "oxford-export-chrome-"));
let chrome, ws;
const wallTimeout = setTimeout(() => {
  console.error('Stored overview browser test exceeded 180 seconds');
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
  console.error("preview: app loaded");
  await evaluate(`
    for(let i=0;i<600 && !window.ze;i++) await new Promise(r=>setTimeout(r,100));
    if(!window.ze) throw new Error('Application did not initialize WebGPU');
    window.previewCalls = {contrast:0,stored:[],maps:[],open:[],patterns:[]};
    const ze = window.ze, native = ze.x.ze_scan_overview;
    if(typeof native !== 'function') throw new Error('Served WASM lacks ze_scan_overview');
    ze.x.ze_scan_overview = async (...args) => {
      const before = ze.state.io.bytes, n = await native(...args);
      previewCalls.stored.push({query:args[1]===0,n:Number(n),bytes:ze.state.io.bytes-before});return n;
    };
    for (const [name,records] of [['ze_open_scan',previewCalls.open],['ze_pattern',previewCalls.patterns]]) {
      const original=ze.x[name];ze.x[name]=async(...args)=>{
        const before=ze.state.io.bytes,value=await original(...args);
        records.push({bytes:ze.state.io.bytes-before,...(name==='ze_pattern'?{stage:args[2]}:{})});return value;
      };
    }
    const stored = ze.storedOverview.bind(ze), contrast = ze.patternStats.bind(ze);
    ze.storedOverview = async (...args) => {
      const result = await stored(...args);
      previewCalls.maps.push(result && {kind:result.kind,label:result.label,length:result.values.length,samples:[result.values[0],result.values[83377],result.values.at(-1)]});
      if(window.holdStored) {window.holdStored=false;await new Promise(r=>window.releaseStored=r);}
      return result;
    };
    ze.patternStats = async (...args) => {
      previewCalls.contrast++;
      const result = await contrast(...args);
      if(window.holdContrast) {window.holdContrast=false;await new Promise(r=>window.releaseContrast=r);}
      return result;
    };
    window.waitPreview = async (predicate) => {for(let i=0;i<300 && !predicate();i++) await new Promise(r=>setTimeout(r,50));if(!predicate())throw new Error('Preview condition timed out');};
  `);
  async function openSource() {
    const { root: documentRoot } = await send("DOM.getDocument", {}, sessionId);
    const { nodeId } = await send("DOM.querySelector", { nodeId:documentRoot.nodeId,selector:'#file-scan'},sessionId);
    await send('DOM.setFileInputFiles',{nodeId,files:[acquisition]},sessionId);
    await evaluate(`document.getElementById('file-scan').dispatchEvent(new Event('change'));`);
  }
  await openSource();
  console.error("preview: P6 opening, awaiting stored quality");
  const first = await evaluate(`
    await waitPreview(()=>document.getElementById('map-legend').textContent==='Band contrast');
    await ze._queue;
    const before = ze.stats();await new Promise(r=>setTimeout(r,250));const idle=ze.stats();
    return {calls:previewCalls,initialBytes:before.fileBytes,idleBytes:idle.fileBytes,live:document.getElementById('pat-live').checked,stage:document.querySelector('#pat-stage [aria-pressed=true]').dataset.stage};
  `);
  console.error("preview: P6 stored quality loaded",JSON.stringify(first));
  assert.equal(first.calls.contrast,0);assert.equal(first.idleBytes,0);
  assert.ok(first.initialBytes<16*1024*1024,`Opening exceeded bounded metadata traffic: ${first.initialBytes}`);
  assert.ok(first.calls.patterns.length<=2 && first.calls.patterns.every(p=>p.stage===0), "Opening may display the selected stored pattern only");
  assert.equal(first.live,true);assert.equal(first.stage,'stored');
  const controls = await evaluate(`
    const select=document.getElementById('overview-source'), button=document.getElementById('btn-overview');
    const visible=el=>{const r=el.getBoundingClientRect();return !el.hidden && r.width>0 && r.height>0 && r.left>=0 && r.right<=innerWidth && r.top>=0 && r.bottom<=document.getElementById('map-well').getBoundingClientRect().top;};
    return {choices:Array.from(select.options,o=>o.textContent),visible:visible(select)&&visible(button)};
  `);
  assert.deepEqual(controls.choices,['Band contrast','Pattern contrast']);assert.ok(controls.visible,'Contrast controls must be visible above the map before calculation');
  assert.equal(first.calls.maps[0].kind,1);assert.equal(first.calls.maps[0].length,415*293);
  assert.equal(first.calls.stored[0].query,true);assert.equal(first.calls.stored[0].bytes,0);
  assert.ok(first.calls.stored[1].bytes>0 && first.calls.stored[1].bytes<=415*293*4+65536);
  // Delay a real contrast result, then replace the source. Its callback must not
  // paint the new map or schedule another batch against the new acquisition.
  console.error("preview: checking explicit contrast and source replacement");
  const cancelled = await evaluate(`
    window.holdContrast=true;const picker=document.getElementById('overview-source');picker.value='calculated';picker.dispatchEvent(new Event('change'));
    await waitPreview(()=>!!window.releaseContrast);
    const started = previewCalls.contrast, selector=document.getElementById('overview-source');
    const choices=Array.from(selector.options,o=>o.textContent);
    selector.value='stored';selector.dispatchEvent(new Event('change'));
    const selected=document.getElementById('map-legend').textContent;
    document.getElementById('btn-example').click();
    await waitPreview(()=>ze.scan.rows===55 && document.getElementById('app').dataset.busy==='0' && !document.getElementById('btn-overview').disabled);
    releaseContrast();await new Promise(r=>setTimeout(r,300));
    return {started,after:previewCalls.contrast,choices,selected,neutral:document.getElementById('map-legend').hidden,button:document.getElementById('btn-overview').textContent,sourceHidden:document.getElementById('overview-source').hidden};
  `);
  assert.equal(cancelled.started,1);assert.equal(cancelled.after,1);
  assert.deepEqual(cancelled.choices,['Band contrast','Pattern contrast']);assert.equal(cancelled.selected,'Band contrast');
  assert.ok(cancelled.neutral && !cancelled.sourceHidden);assert.equal(cancelled.button,'Calculate pattern contrast');
  console.error("preview: contrast replacement guard passed");
  // A late stored scalar result must also be discarded after a new source opens.
  await evaluate(`window.holdStored=true;window.releaseStored=null;document.getElementById('pat-live').checked=false;`);
  await openSource();
  console.error("preview: checking late stored result and Stop");
  const stale = await evaluate(`
    await waitPreview(()=>!!window.releaseStored);
    document.getElementById('btn-example').click();
    await waitPreview(()=>ze.scan.rows===55 && document.getElementById('app').dataset.busy==='0' && !document.getElementById('btn-overview').disabled);
    releaseStored();await new Promise(r=>setTimeout(r,300));
    const neutral=document.getElementById('map-legend').hidden;
    window.holdContrast=true;window.releaseContrast=null;document.getElementById('btn-overview').click();
    await waitPreview(()=>!!window.releaseContrast);
    const before=previewCalls.contrast;document.getElementById('btn-overview').click();releaseContrast();await new Promise(r=>setTimeout(r,300));
    return {neutral,before,after:previewCalls.contrast,button:document.getElementById('btn-overview').textContent,live:document.getElementById('pat-live').checked};
  `);
  assert.ok(stale.neutral);assert.equal(stale.after,stale.before);assert.equal(stale.button,'Calculate pattern contrast');
  assert.equal(stale.live,false,'An explicit on-spot opt-out persists across scan changes');
  console.log(JSON.stringify({first,cancelled,stale}));
  console.log('Stored overview browser checks passed: bounded scalar IO, zero automatic contrast, neutral fallback, explicit calculation, source switching, stop and stale callback guards.');
} finally {
  clearTimeout(wallTimeout);ws?.close();chrome?.kill();server.closeAllConnections();await new Promise(resolve=>server.close(resolve));
  try {rmSync(profile,{recursive:true,force:true});} catch { /* Chrome may still be shutting down. */ }
}
