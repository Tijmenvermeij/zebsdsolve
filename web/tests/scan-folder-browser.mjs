#!/usr/bin/env node
// Individual scan loading and batch source-folder setup. Node22+: node web/tests/scan-folder-browser.mjs
import assert from "node:assert/strict";
import { spawn } from "node:child_process";
import { createServer } from "node:http";
import { createReadStream, existsSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { extname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
const root = resolve(fileURLToPath(new URL("../..", import.meta.url)));
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
  console.error('Scan folder browser test exceeded 300 seconds');
  clearTimeout(wallTimeout);ws?.close();chrome?.kill();server.closeAllConnections();server.close();process.exit(1);
},300000);
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



  await evaluate(`
    for(let i=0;i<600&&!window.ze;i++)await new Promise(resolve=>setTimeout(resolve,50));
    if(!window.ze)throw Error('No engine');
    window.pause=ms=>new Promise(resolve=>setTimeout(resolve,ms));
    window.waitFor=async predicate=>{for(let i=0;i<2400;i++){if(predicate())return;await pause(50);}throw Error('Timeout: '+document.getElementById('status').textContent);};
    window.require=(ok,message)=>{if(!ok)throw Error(message);};
    require(!document.getElementById('btn-open-folder'),'Folder-first action remains');
    document.getElementById('pat-live').checked=false;document.getElementById('btn-example').click();
    await waitFor(()=>ze.scan.rows===55&&ze.phases.length===1&&document.getElementById('app').dataset.busy==='0');await ze._queue;
    const source=ze.state.files.find(file=>file?.name==='patterns_v2.h5');
    const fs=await navigator.storage.getDirectory(),folder=await fs.getDirectoryHandle('batch-setup-'+Date.now(),{create:true});
    Object.defineProperties(folder,{queryPermission:{value:async()=> 'granted'},requestPermission:{value:async()=> 'granted'}});
    const handles=[];for(const name of ['first.h5','second.h5']){const handle=await folder.getFileHandle(name,{create:true}),stream=await handle.createWritable();await stream.write(source);await stream.close();handles.push(handle);}
    let index=0;window.showOpenFilePicker=async()=>[handles[index++]];
    window.pickerCalls=0;window.showDirectoryPicker=async options=>{require(options.mode==='readwrite','Wrong folder permission');pickerCalls++;return folder;};
    for(let n=1;n<=2;n++){
      const generation=ze._scanGeneration;document.getElementById('btn-open').click();
      await waitFor(()=>ze._scanGeneration>generation&&document.getElementById('app').dataset.busy==='0');await ze._queue;
      ze.configure({bandwidth:16,fit_bandwidth:16,refine_iters:1,roi_x:10,roi_y:10,roi_w:1,roi_h:1});
      document.getElementById('btn-batch-current').click();await waitFor(()=>document.querySelectorAll('#batch-jobs tr').length===n);
    }
    require(pickerCalls===0,'Opening individual scans requested a folder');
    const before={generation:ze._scanGeneration,geometry:JSON.stringify(ze.geometry),roi:JSON.stringify([ze.settings.roi_x,ze.settings.roi_y,ze.settings.roi_w,ze.settings.roi_h])};
    document.getElementById('btn-batch-setup').click();await waitFor(()=>document.getElementById('batch-folder-dialog'));
    require(document.getElementById('btn-batch-setup').disabled,'Setup did not guard picking');
    require([...document.querySelectorAll('#batch-jobs button')].every(button=>button.disabled),'Row actions remained enabled during setup');
    [...document.querySelectorAll('#batch-folder-dialog button')].find(b=>b.textContent==='Cancel').click();
    await waitFor(()=>!document.getElementById('btn-batch-setup').disabled);
    require(ze._scanGeneration===before.generation&&JSON.stringify(ze.geometry)===before.geometry&&JSON.stringify([ze.settings.roi_x,ze.settings.roi_y,ze.settings.roi_w,ze.settings.roi_h])===before.roi,'Cancellation changed scan state');
    document.getElementById('btn-batch-setup').click();await waitFor(()=>document.getElementById('batch-folder-dialog'));
    document.querySelector('#batch-folder-dialog .primary').click();await waitFor(()=>!document.getElementById('batch-folder-dialog')&&!document.getElementById('btn-batch-setup').disabled);
    require(pickerCalls===1,'Shared source folder was not reused: '+pickerCalls);
    require([...document.querySelectorAll('#batch-jobs tr')].every(row=>row.children[1].textContent===folder.name),'Jobs did not receive the source folder');
    require(ze._scanGeneration===before.generation&&JSON.stringify(ze.geometry)===before.geometry&&JSON.stringify([ze.settings.roi_x,ze.settings.roi_y,ze.settings.roi_w,ze.settings.roi_h])===before.roi,'Setup changed scan state');
    document.getElementById('batch-downloads').click();
    require(document.getElementById('btn-batch-setup').disabled&&document.getElementById('btn-batch-folder').disabled,'Downloads did not disable folder actions');
  `);
  console.log('Batch setup browser passed: individual file picker, shared-folder reuse, cancellation and scan/ROI preservation, Downloads controls.');

} finally {
  clearTimeout(wallTimeout);ws?.close();chrome?.kill();server.closeAllConnections();await new Promise(resolve=>server.close(resolve));
  try{rmSync(profile,{recursive:true,force:true});}catch{}
}
