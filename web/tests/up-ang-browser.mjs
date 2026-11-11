#!/usr/bin/env node
// UP1/UP2 + ANG in the real app: picker, stored IQ, project guards and batch replay.
// Node 22+: node web/tests/up-ang-browser.mjs
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
const fixtures = mkdtempSync(join(tmpdir(), "up-ang-fixtures-"));
function makeUP(name, sixteen=false) {
 const header=Buffer.alloc(16);header.writeUInt32LE(1,0);header.writeUInt32LE(32,4);header.writeUInt32LE(16,8);header.writeUInt32LE(16,12);
 const pixels=Buffer.alloc(6*512*(sixteen?2:1));for(let cell=0;cell<6;cell++)for(let p=0;p<512;p++) {const index=cell*512+p;if(sixteen)pixels.writeUInt16LE(1000+index,index*2);else pixels[index]=10+index;}
 const path=join(fixtures,name);writeFileSync(path,Buffer.concat([header,pixels]));return path;
}
const up1=makeUP("map.up1"),up2=makeUP("map.up2",true),ang=join(fixtures,"map.ang"),bad=join(fixtures,"wrong.ang");
const angHeader="# HEADER: Start\n# VERSION 7\n# x-star 0.43\n# y-star 0.72\n# z-star 0.61\n# CameraElevationAngle 10\n# GRID: SqrGrid\n# XSTEP: 0.025\n# YSTEP: 0.05\n# NCOLS_ODD: 3\n# NCOLS_EVEN: 3\n# NROWS: 2\n";
const points=Array.from({length:6},(_,i)=>`0 0 0 ${(i%3)*.025} ${Math.floor(i/3)*.05} ${(i+1)*10} .8 1 1 .2`).join("\n")+"\n";
writeFileSync(ang,angHeader+points);writeFileSync(bad,angHeader.replace("NROWS: 2","NROWS: 3")+points);
const profile = mkdtempSync(join(tmpdir(), "up-ang-chrome-"));
let chrome, ws;
const wallTimeout = setTimeout(() => {
  console.error('UP/ANG browser test exceeded 300 seconds');
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
    for(let i=0;i<600 && !window.ze;i++)await new Promise(r=>setTimeout(r,50));
    if(!window.ze)throw Error('No engine');
    document.getElementById('pat-live').checked=false;
    window.pause=ms=>new Promise(r=>setTimeout(r,ms));
    window.waitFor=async predicate=>{for(let i=0;i<600;i++){if(predicate())return;await pause(50);}throw Error('Timeout: '+document.getElementById('status').textContent);};
    window.require=(ok,message)=>{if(!ok)throw Error(message);};
    window.angPickerCalls=0;
    window.showOpenFilePicker=async()=>{window.angPickerCalls++;throw new DOMException('Use standard input fallback','SecurityError');};
  `);
  await send('Page.setInterceptFileChooserDialog',{enabled:true},sessionId);
  async function inputFiles(selector,files) {
    const {root:doc}=await send('DOM.getDocument',{},sessionId);
    const {nodeId}=await send('DOM.querySelector',{nodeId:doc.nodeId,selector},sessionId);
    assert.ok(nodeId,selector);
    await send('DOM.setFileInputFiles',{nodeId,files},sessionId);
  }
  await inputFiles('#file-scan',[up1]);
  await evaluate(`await waitFor(()=>document.querySelector('#scan-info-dialog')?.open);await pause(150);require(angPickerCalls===0,'ANG picker hid prompt automatically');require(document.getElementById('scan-info-title').textContent==='Select the ANG scan file for map.up1','ANG prompt is unclear');`);
  const prompt=await send('Page.captureScreenshot',{format:'png'},sessionId);
  writeFileSync('/tmp/zebsdsolve-up-ang-prompt.png',Buffer.from(prompt.data,'base64'));
  await evaluate(`document.querySelector('#scan-info-dialog .primary').click();await pause(50);require(angPickerCalls===1,'Choose ANG did not open picker');`);
  await inputFiles('#file-scan-info',[ang]);
  await evaluate(`
    await waitFor(()=>document.getElementById('app').dataset.busy==='0' && ze.scan.rows===2 && ze.scan.cols===3);
    require(!document.getElementById('scan-info-dialog'),'Picker did not close');
    const stored=await ze.storedOverview();require(stored.label==='Image quality','Stored map kind');
    require(JSON.stringify(Array.from(stored.values))==='[10,20,30,40,50,60]','IQ changed');
    require(ze.geometry.stepXUm===.025 && ze.geometry.stepYUm===.05,'ANG scan steps missing');
    require(ze.scan.hasPc,'ANG PC was not imported');
    require(ze.geometry.pc.every((v,i)=>Math.abs(v-[.43,.28,.61][i])<1e-12),'EDAX ANG PC convention was wrong on rectangular detector');
    const pcFact=[...document.querySelectorAll('#scan-facts dt')].find(el=>el.textContent==='pattern centre');require(pcFact?.nextElementSibling.textContent==='from the file','Imported PC not reported in GUI');
    await waitFor(()=>document.getElementById('overview-source').textContent.toLowerCase().includes('image quality'));
  `);
  console.error('UP1 + ANG: second chooser, grid, scan steps and stored IQ passed');
  await evaluate(`ze.configure({roi_x:1,roi_y:1,roi_w:1,roi_h:1});`);
  await inputFiles('#file-scan',[up2]);
  await evaluate(`await waitFor(()=>document.querySelector('#scan-info-dialog')?.open);document.querySelector('#scan-info-dialog button:not(.primary)').click();await pause(50);require(ze.scan.dtype==='uint8' && ze.scan.rows===2,'Cancel replaced scan');require(ze.settings.roi_w===1 && ze.settings.roi_x===1,'Cancel changed prior ROI');`);
  await inputFiles('#file-scan',[bad,up2]);
  await evaluate(`await waitFor(()=>document.getElementById('status').classList.contains('err'));require(!document.querySelector('#scan-info-dialog'),'Already-paired selection prompted');require(ze.settings.roi_w===1,'Rejected scan changed prior ROI');`);
  await inputFiles('#file-scan',[ang,up2]);
  await evaluate(`
    await waitFor(()=>document.getElementById('app').dataset.busy==='0' && ze.scan.dtype==='uint16');
    require(['roi_x','roi_y','roi_w','roi_h'].every(key=>ze.settings[key]===0),'New scan retained previous ROI');
    require(document.getElementById('roi-text').textContent==='region','Full-map ROI controls stale');
    const values=await ze.pattern(5,'stored',true);require(values[0]===3560 && values[511]===4071,'UP2 pattern-to-ANG pixel order changed');
    const stored=await ze.storedOverview();require(stored.values[5]===60,'UP2 stored IQ missing');
  `);
  const savedProject=await evaluate(`
    ze.configure({roi_x:1,roi_y:1,roi_w:1,roi_h:1});
    const makeURL=URL.createObjectURL;let captured;
    URL.createObjectURL=blob=>{captured=blob;return makeURL.call(URL,blob);};
    try { document.getElementById('btn-json').click(); }
    finally { URL.createObjectURL=makeURL; }
    const project=JSON.parse(await captured.text());
    require(project.source_files.length===2 && project.source_files[0].name==='map.up2' && project.source_files[1].name==='map.ang','Saved project lost source pair');
    return project;
  `);
  const projectPath=join(fixtures,'up-project.json');
  writeFileSync(projectPath,JSON.stringify({...savedProject,settings:{...savedProject.settings,bandwidth:42},geometry:{...savedProject.geometry,pc:[.2,.3,.4]}}));
  console.error('UP2 + ANG: cancel, mismatch rejection, reverse selection and pattern order passed');
  // Queue/reload proves the companion is retained with the pattern source.
  await inputFiles('#file-phase',[join(root,'testdata/nickel/Ni_master_20kV.npz')]);
  await evaluate(`await waitFor(()=>document.getElementById('app').dataset.busy==='0' && ze.phases.length===1);document.getElementById('btn-batch-current').click();await waitFor(()=>document.querySelectorAll('#batch-jobs tr').length===1);`);
  await inputFiles('#file-scan',[up1,ang]);
  await evaluate(`
    await waitFor(()=>document.getElementById('app').dataset.busy==='0' && ze.scan.dtype==='uint8');
  `);
  await inputFiles('#file-scan',[projectPath]);
  await evaluate(`await waitFor(()=>document.getElementById('status').textContent.includes('project requires'));require(ze.settings.bandwidth===88,'Wrong project changed settings');require(ze.geometry.pc[0]!==.2,'Wrong project changed geometry');`);
  await inputFiles('#file-scan',[projectPath,join(root,'testdata/nickel/patterns_v2.h5')]);
  await evaluate(`await waitFor(()=>document.getElementById('status').textContent.includes('project requires'));require(ze.scan.dtype==='uint8' && ze.scan.cols===3 && ze.settings.bandwidth===88,'Wrong selected source bypassed project validation');`);
  await evaluate(`
    const row=document.querySelector('#batch-jobs tr');[...row.querySelectorAll('button')].find(b=>b.textContent==='Load').click();
    await waitFor(()=>document.getElementById('app').dataset.busy==='0' && ze.scan.dtype==='uint16');
    require((await ze.storedOverview()).values[0]===10,'Batch replay lost ANG');
    require(ze.settings.roi_x===1 && ze.settings.roi_y===1 && ze.settings.roi_w===1 && ze.settings.roi_h===1,'Batch Load lost saved ROI');
  `);
  const validProjectPath=join(fixtures,'valid-project.json');writeFileSync(validProjectPath,JSON.stringify(savedProject));
  await evaluate(`window.projectGeneration=ze._scanGeneration;ze.configure({roi_x:0,roi_y:0,roi_w:0,roi_h:0});`);
  await inputFiles('#file-scan',[up2,ang,validProjectPath]);
  await evaluate(`await waitFor(()=>ze._scanGeneration>projectGeneration && document.getElementById('app').dataset.busy==='0');require(ze.settings.roi_x===1 && ze.settings.roi_w===1,'Project open lost saved ROI');`);
  const badHintsPath=join(fixtures,'bad-hints-project.json');
  writeFileSync(badHintsPath,JSON.stringify({...savedProject,settings:{...savedProject.settings,bandwidth:42,mask:false,roi_x:1,roi_y:1,roi_w:1,roi_h:1,static_background:'average',dynamic_background:'divide'},opening_hints:{map_rows:3,map_cols:3,pattern_height:0,pattern_width:0}}));
  await evaluate(`window.beforeRejectedOpen={settings:JSON.stringify(ze.settings),geometry:JSON.stringify(ze.geometry),generation:ze._scanGeneration};`);
  await inputFiles('#file-scan',[up2,ang,badHintsPath]);
  await evaluate(`
    await waitFor(()=>document.getElementById('app').dataset.busy==='0' && document.getElementById('status').classList.contains('err'));
    require(JSON.stringify(ze.settings)===beforeRejectedOpen.settings,'Rejected native open changed settings');
    require(JSON.stringify(ze.geometry)===beforeRejectedOpen.geometry,'Rejected native open changed geometry');
    require(ze._scanGeneration===beforeRejectedOpen.generation,'Rejected open invalidated prior scan warmup');
    require(ze.scan.dtype==='uint16' && (await ze.storedOverview()).values[5]===60,'Rejected native open replaced source');
    require(document.getElementById('overview-source').textContent.toLowerCase().includes('image quality'),'Rejected open lost old map');
  `);
  const screenshot=await send('Page.captureScreenshot',{format:'png'},sessionId);
  writeFileSync('/tmp/zebsdsolve-up-ang-map.png',Buffer.from(screenshot.data,'base64'));
  console.log('UP/ANG browser checks passed: prompt, cancel, malformed pair, map/steps/IQ/PC, UP1/UP2 pixel order, project transactions and batch replay.');
} finally {
  clearTimeout(wallTimeout);ws?.close();chrome?.kill();server.closeAllConnections();await new Promise(resolve=>server.close(resolve));
  try {rmSync(profile,{recursive:true,force:true});rmSync(fixtures,{recursive:true,force:true});} catch {}
}
