#!/usr/bin/env node
// Downloads mode in the actual app. Node22+: node web/tests/batch-downloads-browser.mjs [P6.h5oina]
import assert from "node:assert/strict";
import { spawn } from "node:child_process";
import { createServer } from "node:http";
import { createReadStream, existsSync, mkdtempSync, rmSync, writeFileSync, readdirSync } from "node:fs";
import { tmpdir } from "node:os";
import { extname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
const root = resolve(fileURLToPath(new URL("../..", import.meta.url)));
const acquisition=process.argv[2]&&resolve(process.argv[2]);
const downloads=mkdtempSync(join(tmpdir(),"zebsdsolve-downloads-browser-"));
writeFileSync("/tmp/zebsdsolve-downloads-browser-path.txt",downloads);
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
  console.error('Downloads browser test exceeded 300 seconds');
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



  await send('Browser.setDownloadBehavior',{behavior:'allow',downloadPath:downloads,eventsEnabled:true});
  const waitDownloads=async count=>{for(let i=0;i<1200;i++){const names=readdirSync(downloads);if(names.filter(name=>name.endsWith('.zip')).length>=count&&!names.some(name=>name.endsWith('.crdownload')))return;await new Promise(resolve=>setTimeout(resolve,50));}throw Error('Browser downloads timed out');};
  await evaluate(`
    for(let i=0;i<600&&!window.ze;i++)await new Promise(resolve=>setTimeout(resolve,50));
    if(!window.ze)throw Error('No engine');
    window.pause=ms=>new Promise(resolve=>setTimeout(resolve,ms));
    window.waitFor=async predicate=>{for(let i=0;i<2400;i++){if(predicate())return;await pause(50);}throw Error('Timeout: '+document.getElementById('status').textContent);};
    window.require=(ok,message)=>{if(!ok)throw Error(message);};
    // A ZIP owns copied chunks, so a slow reader survives OPFS deletion and URL revocation.
    const {createStoreZip}=await import('/web/app/js/zip-store.js');
    const privateRoot=await navigator.storage.getDirectory(),probe=await privateRoot.getDirectoryHandle('zip-snapshot-probe',{create:true});
    const probeHandle=await probe.getFileHandle('member.bin',{create:true}),probeWriter=await probeHandle.createWritable();
    await probeWriter.write(Uint8Array.from({length:2*1024*1024+17},(_,index)=>index%251));await probeWriter.close();
    const ownedZip=await createStoreZip([{name:'probe/member.bin',blob:await probeHandle.getFile()}]);
    const expectedZip=new Uint8Array(await ownedZip.arrayBuffer()),probeURL=URL.createObjectURL(ownedZip),response=await fetch(probeURL),reader=response.body.getReader();
    await probe.removeEntry('member.bin');URL.revokeObjectURL(probeURL);let offset=0;
    for(;;){const chunk=await reader.read();if(chunk.done)break;require(chunk.value.every((value,index)=>value===expectedZip[offset+index]),'Deleted backing file/revoked URL corrupted opened ZIP stream');offset+=chunk.value.length;await pause(1);}
    require(offset===expectedZip.length,'Opened ZIP stream was truncated');
    require(!document.getElementById('batch-downloads').checked,'Downloads default is on');
    document.getElementById('pat-live').checked=false;document.getElementById('btn-example').click();
    await waitFor(()=>ze.scan.rows===55&&ze.phases.length===1&&document.getElementById('app').dataset.busy==='0');await ze._queue;
    window.exampleSource=ze.state.files.find(file=>file?.name==='patterns_v2.h5');
    for(const [index,bandwidth] of [16,24].entries()){
      ze.configure({bandwidth,fit_bandwidth:16,refine_iters:1,roi_x:10,roi_y:10,roi_w:1,roi_h:1});document.getElementById('btn-batch-current').click();await waitFor(()=>document.querySelectorAll('#batch-jobs tr').length===index+1);
    }
    document.getElementById('batch-downloads').checked=true;document.getElementById('batch-downloads').dispatchEvent(new Event('change'));
    require([...document.querySelectorAll('#batch-jobs tr')].every(row=>row.children[1].textContent==='Downloads'),'After-queue mode toggle did not apply');
    require(document.getElementById('btn-batch-folder').disabled,'External folder action still enabled');
    window.showDirectoryPicker=async()=>{throw Error('Downloads requested external directory write access');};
    const click=HTMLAnchorElement.prototype.click;window.autoDownloads=0;
    HTMLAnchorElement.prototype.click=function(){if(this.download?.endsWith('.zip')){autoDownloads++;return;}return click.call(this);};
    const Engine=ze.constructor,create=Engine.create;window.restoreCreate=()=>{Engine.create=create;};
    Engine.create=async function(...args){const engine=await create.apply(this,args);let begins=0,held=false;const begin=engine.indexBegin.bind(engine),step=engine.indexStep.bind(engine);
      engine.indexBegin=async(...args)=>{begins++;return begin(...args);};engine.indexStep=async(...args)=>{if(begins===2&&!held){held=true;await new Promise(resolve=>window.releaseSecondStep=resolve);}return step(...args);};return engine;};
    document.getElementById('btn-batch-run').click();await waitFor(()=>document.querySelectorAll('#batch-jobs tr[data-status=complete]').length===1&&!!window.releaseSecondStep);
    window.firstArchiveLink=document.querySelector('a.batch-download');require(!!firstArchiveLink,'Missing manual archive link');firstArchiveLink.focus();
    releaseSecondStep();await waitFor(()=>document.querySelectorAll('#batch-jobs tr[data-status=complete]').length===2&&document.getElementById('app').dataset.batchRunning==='0');restoreCreate();
    require(firstArchiveLink.isConnected&&firstArchiveLink===document.querySelector('a.batch-download')&&document.activeElement===firstArchiveLink,'Progress destroyed/focused manual ZIP link');
    require(autoDownloads===2,'Automatic archive dispatch not attempted once per scan');
    require([...document.querySelectorAll('#batch-jobs .batch-status')].every(cell=>cell.textContent==='Download ready'),'UI claims disk saving was confirmed');
    require(!document.getElementById('batch-folder-dialog'),'Unexpected folder permission dialog');
  `);
  assert.equal(readdirSync(downloads).filter(name=>name.endsWith('.zip')).length,0,'Simulated blocked auto-download was not blocked');
  // A trusted Enter activates the focused persistent anchor, bypassing the simulated auto block.
  await send('Input.dispatchKeyEvent',{type:'keyDown',key:'Enter',code:'Enter',windowsVirtualKeyCode:13},sessionId);await send('Input.dispatchKeyEvent',{type:'keyUp',key:'Enter',code:'Enter',windowsVirtualKeyCode:13},sessionId);
  await waitDownloads(1);
  async function clickArchive(index){const point=await evaluate(`const element=document.querySelectorAll('a.batch-download')[${index}],rect=element.getBoundingClientRect();return {x:rect.x+rect.width/2,y:rect.y+rect.height/2};`);await send('Input.dispatchMouseEvent',{type:'mousePressed',button:'left',clickCount:1,...point},sessionId);await send('Input.dispatchMouseEvent',{type:'mouseReleased',button:'left',clickCount:1,...point},sessionId);}
  await clickArchive(1);await waitDownloads(2);
  await evaluate(`
    // Individual file opening and Downloads do not request directory access.
    const fs=await navigator.storage.getDirectory(),directory=await fs.getDirectoryHandle('downloads-source-'+Date.now(),{create:true});
    const input=await directory.getFileHandle('readonly.h5',{create:true}),stream=await input.createWritable();await stream.write(exampleSource);await stream.close();
    window.showDirectoryPicker=async()=>{throw Error('Downloads requested directory access');};
    window.showOpenFilePicker=async()=>[input];
    const generation=ze._scanGeneration;document.getElementById('btn-open').click();await waitFor(()=>ze._scanGeneration>generation&&document.getElementById('app').dataset.busy==='0');
    require(document.getElementById('btn-batch-setup').disabled,'Downloads enabled output setup');
    ze.configure({bandwidth:16,fit_bandwidth:16,refine_iters:1,roi_x:10,roi_y:10,roi_w:1,roi_h:1});document.getElementById('btn-batch-current').click();await waitFor(()=>document.querySelectorAll('#batch-jobs tr').length===3);
    window.showDirectoryPicker=async()=>{throw Error('Batch tried to upgrade a read-only source grant');};document.getElementById('btn-batch-run').click();await waitFor(()=>document.querySelectorAll('#batch-jobs tr[data-status=complete]').length===3&&document.getElementById('app').dataset.batchRunning==='0');
    require(autoDownloads===3,'Read-only source archive not prepared');
  `);
  await clickArchive(2);await waitDownloads(3);
  async function inputFile(selector,path){const {root:doc}=await send('DOM.getDocument',{},sessionId);const {nodeId}=await send('DOM.querySelector',{nodeId:doc.nodeId,selector},sessionId);await send('DOM.setFileInputFiles',{nodeId,files:[path]},sessionId);}
  if(acquisition){
    await evaluate(`document.querySelector('#phases button').click();await waitFor(()=>ze.phases.length===0&&document.getElementById('app').dataset.busy==='0');`);
    await inputFile('#file-scan',acquisition);
    await evaluate(`await waitFor(()=>ze.scan.rows===415&&document.getElementById('app').dataset.busy==='0');`);
    await inputFile('#file-phase',join('/Users/tvermeij/EMsoftData/TiOverlap','Ti_30kv_HCP.h5'));
    const reference=await evaluate(`
      await waitFor(()=>ze.phases.length===1&&document.getElementById('app').dataset.busy==='0');
      ze.configure({bandwidth:16,fit_bandwidth:16,refine_iters:1,roi_x:165,roi_y:284,roi_w:1,roi_h:1,static_background:'none',dynamic_background:'subtract'});
      await ze.index();const snapshot=ze.exportOxfordSnapshot(),cell=284*293+165;
      document.getElementById('btn-batch-current').click();await waitFor(()=>document.querySelectorAll('#batch-jobs tr').length===4);
      document.getElementById('btn-batch-run').click();await waitFor(()=>document.querySelectorAll('#batch-jobs tr[data-status=complete]').length===4&&document.getElementById('app').dataset.batchRunning==='0');
      return {cell,euler:snapshot.results.euler.slice(cell*3,cell*3+3),phase:snapshot.results.phase[cell],frame:snapshot.oxford_frame};
    `);
    writeFileSync(join(downloads,'p6-reference.json'),JSON.stringify(reference));
    const readyScreen=await send('Page.captureScreenshot',{format:'png'},sessionId);writeFileSync('/tmp/zebsdsolve-downloads-ready.png',Buffer.from(readyScreen.data,'base64'));
    await clickArchive(3);
    // Clear immediately after a real download starts; its immutable ZIP must stay readable.
    await evaluate(`document.getElementById('btn-batch-clear').click();await waitFor(()=>document.querySelectorAll('#batch-jobs tr').length===0);`);
    await waitDownloads(4);
  }
  const screenshot=await send('Page.captureScreenshot',{format:'png'},sessionId);writeFileSync('/tmp/zebsdsolve-downloads-option.png',Buffer.from(screenshot.data,'base64'));
  console.log(JSON.stringify({downloads,archives:readdirSync(downloads).filter(name=>name.endsWith('.zip'))}));
  console.log('Downloads browser passed: default off, after-queue toggle, no external write grants, blocked automatic/manual download, persistent focused links, read-only source folder and real browser ZIP files.');
} finally {
  clearTimeout(wallTimeout);ws?.close();chrome?.kill();server.closeAllConnections();await new Promise(resolve=>server.close(resolve));
  try{rmSync(profile,{recursive:true,force:true});}catch{}
}
