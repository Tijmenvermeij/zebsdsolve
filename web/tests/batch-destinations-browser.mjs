#!/usr/bin/env node
// Per-scan destinations in the actual app. Node22+: node web/tests/batch-destinations-browser.mjs
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
  console.error('Batch destination browser test exceeded 300 seconds');
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
    document.getElementById('pat-live').checked=false;document.getElementById('btn-example').click();
    await waitFor(()=>ze.scan.rows===55&&ze.phases.length===1&&document.getElementById('app').dataset.busy==='0');await ze._queue;
    const source=ze.state.files.find(file=>file?.name==='patterns_v2.h5');
    const fs=await navigator.storage.getDirectory();window.container=await fs.getDirectoryHandle('batch-source-folders-'+Date.now(),{create:true});
    window.parents=[];window.inputs=[];window.expected=[];
    const {rememberFileHandle}=await import('/web/app/js/source-folders.js');
    for(const name of ['source-A','source-B']){
      const directory=await container.getDirectoryHandle(name,{create:true});
      Object.defineProperties(directory,{queryPermission:{value:async()=> 'granted'},requestPermission:{value:async()=> 'granted'}});
      const handle=await directory.getFileHandle('same-scan.h5',{create:true});const stream=await handle.createWritable();await stream.write(source);await stream.close();
      parents.push(directory);inputs.push(handle);
    }
    window.loadSource=async index=>{
      const file=await inputs[index].getFile();const transfer=new DataTransfer();transfer.items.add(file);const input=document.getElementById('file-scan');input.files=transfer.files;
      const generation=ze._scanGeneration;rememberFileHandle(input.files[0],inputs[index]);input.dispatchEvent(new Event('change'));
      await waitFor(()=>ze._scanGeneration>generation&&document.getElementById('app').dataset.busy==='0'&&document.getElementById('scan-meta').textContent.startsWith('same-scan.h5'));await ze._queue;
    };
    window.addSource=async(index,count)=>{
      await loadSource(index);ze.configure({bandwidth:16,fit_bandwidth:16,refine_iters:1,roi_x:10,roi_y:10,roi_w:1,roi_h:1});
      await ze.index();expected.push(ze.exportSnapshot());document.getElementById('btn-batch-current').click();await waitFor(()=>document.querySelectorAll('#batch-jobs tr').length===count);
    };
    await addSource(0,1);await addSource(1,2);
    require(!document.getElementById('btn-batch-run').disabled,'Missing folder disabled Run');
    const second=document.querySelectorAll('#batch-jobs tr')[1];[...second.querySelectorAll('button')].find(button=>button.textContent==='Output…').click();
    await waitFor(()=>document.querySelector('#batch-output-dialog')?.open);
    require(document.getElementById('batch-output-name').value==='same-scan_ZEBSDsolve_BW16','Default stem');
    window.pickerCalls=0;
    window.showDirectoryPicker=async options=>{pickerCalls++;require(await options.startIn.isSameEntry(inputs[1]),'Output picker starts at wrong source');return parents[1];};
    document.getElementById('batch-output-parent').click();await waitFor(()=>document.getElementById('batch-output-parent').textContent==='source-B');
  `);
  const dialog=await send('Page.captureScreenshot',{format:'png'},sessionId);writeFileSync('/tmp/zebsdsolve-batch-output-dialog.png',Buffer.from(dialog.data,'base64'));
  await evaluate(`
    document.getElementById('batch-output-save').click();await waitFor(()=>!document.getElementById('batch-output-dialog'));
    // Cancel must win even if a folder identity check is awaiting I/O.
    const row=document.querySelectorAll('#batch-jobs tr')[1];[...row.querySelectorAll('button')].find(button=>button.textContent==='Output…').click();await waitFor(()=>document.querySelector('#batch-output-dialog')?.open);
    const {sourceFolders}=await import('/web/app/js/source-folders.js');const accept=sourceFolders.accept;
    sourceFolders.accept=async function(...args){await new Promise(resolve=>window.resumeOutputSave=resolve);return accept.apply(this,args);};
    document.getElementById('batch-output-name').value='cancelled-name';document.getElementById('batch-output-save').click();await waitFor(()=>!!window.resumeOutputSave);
    [...document.querySelectorAll('#batch-output-dialog button')].find(button=>button.textContent==='Cancel').click();resumeOutputSave();await pause(100);sourceFolders.accept=accept;
    require(document.querySelectorAll('#batch-jobs tr')[1].querySelector('.batch-output-name').textContent==='same-scan_ZEBSDsolve_BW16','Cancelled output edit changed job');
    window.showDirectoryPicker=async options=>{pickerCalls++;require(await options.startIn.isSameEntry(inputs[0]),'Scan-folder picker starts at wrong source');return parents[0];};
    document.getElementById('btn-batch-run').click();await waitFor(()=>document.querySelector('#batch-folder-dialog')?.open);
    require(document.getElementById('btn-batch-clear').disabled,'Clear allowed during preparation');document.getElementById('btn-batch-clear').dispatchEvent(new Event('click'));
    require(document.querySelectorAll('#batch-jobs tr').length===2,'Clear removed pending queue');
    document.querySelector('#batch-folder-dialog .primary').click();
    await waitFor(()=>document.querySelectorAll('#batch-jobs tr[data-status=complete]').length===2&&document.getElementById('app').dataset.batchRunning==='0');
    require(pickerCalls===2,'Unexpected folder requests');
    window.stem='same-scan_ZEBSDsolve_BW16';window.originalJSON=[];
    const h=await import('/web/app/js/vendor/h5wasm/hdf5_hl.js');await h.ready;
    for(let i=0;i<2;i++){
      const rootNames=[];for await(const name of parents[i].keys())rootNames.push(name);
      require(rootNames.sort().join(',')===['same-scan.h5',stem].sort().join(','),'Outputs outside per-scan folder');
      const dir=await parents[i].getDirectoryHandle(stem);const names=[];for await(const name of dir.keys())names.push(name);
      require(names.sort().join(',')===[stem+'.ang',stem+'.h5',stem+'_IPF-Z.png',stem+'_NCC.png',stem+'.json'].sort().join(','),'Five matching file names');
      const text=await(await(await dir.getFileHandle(stem+'.json')).getFile()).text();originalJSON.push(text);const config=JSON.parse(text);
      require(config.batch.status==='complete'&&config.batch.outputs.every(item=>item.complete),'Final JSON ledger');
      const file=await(await dir.getFileHandle(stem+'.h5')).getFile();h.FS.writeFile('/out.h5',new Uint8Array(await file.arrayBuffer()));const out=new h.File('/out.h5');
      try{const actual=out.get('zebsdsolve/Selected/quaternions').value;require(actual.every((value,k)=>Math.abs(value-expected[i].results.quaternions[k])<1e-6),'Mixed-folder output orientation');}
      finally{out.close();h.FS.unlink('/out.h5');}
    }
    // The same source parent grant is reused; repeated results get one common collision suffix.
    await addSource(0,3);window.showDirectoryPicker=async()=>{throw Error('Cached source folder was not reused');};document.getElementById('btn-batch-run').click();
    await waitFor(()=>document.querySelectorAll('#batch-jobs tr[data-status=complete]').length===3&&document.getElementById('app').dataset.batchRunning==='0');
    const prefix=stem+'_02',dir=await parents[0].getDirectoryHandle(prefix);const files=[];for await(const name of dir.keys())files.push(name);
    require(files.sort().join(',')===[prefix+'.ang',prefix+'.h5',prefix+'_IPF-Z.png',prefix+'_NCC.png',prefix+'.json'].sort().join(','),'Collision suffix not applied to all files');
    const original=await parents[0].getDirectoryHandle(stem);require(await(await(await original.getFileHandle(stem+'.json')).getFile()).text()===originalJSON[0],'Earlier result changed');
  `);
  const screen=await send('Page.captureScreenshot',{format:'png'},sessionId);writeFileSync('/tmp/zebsdsolve-batch-destinations.png',Buffer.from(screen.data,'base64'));
  console.log('Batch destination browser passed: per-scan Output dialog, cancel during save, source parent grant, two same-basename source folders, five-file outputs, grant reuse and collision protection.');
} finally {
  clearTimeout(wallTimeout);ws?.close();chrome?.kill();server.closeAllConnections();await new Promise(resolve=>server.close(resolve));
  try{rmSync(profile,{recursive:true,force:true});}catch{}
}
