#!/usr/bin/env node
// Batch queue in the actual app: native indexing, OPFS output and export validation.
// Node 22+: node web/tests/stored-overview-browser.mjs source.h5oina
import assert from "node:assert/strict";
import { spawn } from "node:child_process";
import { createServer } from "node:http";
import { createReadStream, existsSync, mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { extname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
const root = resolve(fileURLToPath(new URL("../..", import.meta.url))), acquisition = process.argv[2] && resolve(process.argv[2]);

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
    for(let i=0;i<600 && !window.ze;i++) await new Promise(r=>setTimeout(r,50));
    if(!window.ze)throw Error('No engine');
    document.getElementById('pat-live').checked=false;
    window.pause=ms=>new Promise(r=>setTimeout(r,ms));
    window.waitBatch=async predicate=>{for(let i=0;i<2400;i++){if(predicate())return;await pause(50);}throw Error('Batch timeout: '+document.getElementById('batch-readout').textContent+' / '+document.getElementById('status').textContent);};
    window.require=(ok,message)=>{if(!ok)throw Error(message);};
    document.getElementById('btn-example').click();
    await waitBatch(()=>document.getElementById('app').dataset.busy==='0' && ze.phases.length===1 && ze.scan.rows===55);
    await ze._queue;
    window.editOutputName=async(row,value)=>{
      [...row.querySelectorAll('button')].find(button=>button.textContent==='Output…').click();
      await waitBatch(()=>document.querySelector('#batch-output-dialog')?.open);
      document.getElementById('batch-output-name').value=value;
      document.getElementById('batch-output-save').click();
      await waitBatch(()=>!document.getElementById('batch-output-dialog'));
    };
    const masterData=await(await fetch('/testdata/gan/GaN_master_20kV.npz')).blob();
    const transfer=new DataTransfer();transfer.items.add(new File([masterData],'GaN_master_20kV.npz'));
    const phaseInput=document.getElementById('file-phase');phaseInput.files=transfer.files;phaseInput.dispatchEvent(new Event('change'));
    await waitBatch(()=>document.getElementById('app').dataset.busy==='0' && ze.phases.length===2);
    window.expected=[];window.fixtureBaseGeo=ze.geometry;
    window.fixtureSource=ze.state.files.find(file=>file?.name==='patterns_v2.h5');
    window.addConfigured=async(settings,geometry=null)=>{
      await ze.openScan([fixtureSource]);
      ze.configure(settings);if(geometry)ze.setGeometry(geometry);
      document.getElementById('btn-batch-current').click();
      await waitBatch(()=>document.querySelectorAll('#batch-jobs tr').length===expected.length+1);
      const row=[...document.querySelectorAll('#batch-jobs tr')].at(-1);
      require(row.querySelector('.batch-output-name').textContent.includes('_ZEBSDsolve_BW'+settings.bandwidth),'Default name lacks bandwidth');
      await editOutputName(row,'Ni_test');
      await ze.index();
      expected.push({chance:ze.resultChance(),snapshot:ze.exportSnapshot(),ang:new TextDecoder().decode(ze.exportText('ang')),ipf:Array.from(ze.map('ipf_z')),ncc:Array.from(ze.map('ncc')),settings:ze.settings,geometry:ze.geometry});
    };
    await addConfigured({bandwidth:16,fit_bandwidth:16,refine_iters:1,accept_nmi:0.2,roi_x:10,roi_y:10,roi_w:2,roi_h:1});
    const geo=ze.geometry;geo.pc=[geo.pc[0]+0.001,geo.pc[1],geo.pc[2]];
    await addConfigured({bandwidth:24,binning:2,roi_x:20,roi_y:20,roi_w:1,roi_h:2},geo);
    window.beforeBatch={settings:ze.settings,geometry:ze.geometry,selected:ze.exportSnapshot().results.quaternions};
    const root=await navigator.storage.getDirectory();
    window.testOutput=await root.getDirectoryHandle('zebsdsolve-batch-'+Date.now(),{create:true});
    // OPFS uses the browser's already granted origin permission.
    const access={name:testOutput.name,getDirectoryHandle:testOutput.getDirectoryHandle.bind(testOutput),getFileHandle:testOutput.getFileHandle.bind(testOutput),queryPermission:async()=> 'granted',requestPermission:async()=> 'granted'};
    window.showDirectoryPicker=async()=>access;
    document.getElementById('btn-batch-folder').click();
    await waitBatch(()=>[...document.querySelectorAll('#batch-jobs tr')].every(row=>row.children[1].textContent===testOutput.name));
    document.getElementById('btn-batch-run').click();
    await waitBatch(()=>document.querySelectorAll('#batch-jobs tr[data-status=complete]').length===2 && document.getElementById('app').dataset.busy==='0');
    require(JSON.stringify(ze.settings)===JSON.stringify(beforeBatch.settings),'Interactive settings changed');
    require(JSON.stringify(ze.geometry)===JSON.stringify(beforeBatch.geometry),'Interactive geometry changed');
    require(JSON.stringify(ze.exportSnapshot().results.quaternions)===JSON.stringify(beforeBatch.selected),'Interactive result changed');
  `);
  console.error('batch: two jobs indexed, interactive state retained');
  const first = await evaluate(`
    const h=await import('/web/app/js/vendor/h5wasm/hdf5_hl.js');await h.ready;
    const configs=[];
    for(const prefix of ['Ni_test','Ni_test_02']){
      const dir=await testOutput.getDirectoryHandle(prefix);
      const config=JSON.parse(await(await(await dir.getFileHandle(prefix+'.json')).getFile()).text());
      require(config.batch.status==='complete' && config.batch.outputs.length===5,'Journal completion');configs.push(config);
    }
    require(configs[0].batch.directory==='Ni_test' && configs[1].batch.directory==='Ni_test_02','Collision suffix');
    for(let k=0;k<2;k++){
      const job=configs[k].batch,dir=await testOutput.getDirectoryHandle(job.directory),prefix=job.outputStem;
      const entries=[];for await(const name of dir.keys())entries.push(name);
      require(entries.sort().join(',')===[prefix+'.ang',prefix+'.h5',prefix+'_IPF-Z.png',prefix+'_NCC.png',prefix+'.json'].sort().join(','),'Expected exactly five same-stem files');
      const read=async name=>(await (await dir.getFileHandle(name)).getFile());
      const bytes=new Uint8Array(await (await read(prefix+'.h5')).arrayBuffer());
      h.FS.writeFile('/batch-result.h5',bytes);const file=new h.File('/batch-result.h5','r');
      try{
        const selected=file.get('zebsdsolve/Selected/quaternions').value,expectedQ=expected[k].snapshot.results.quaternions;
        require(selected.length===expectedQ.length,'Export length');
        let max=0;for(let i=0;i<selected.length;i++)max=Math.max(max,Math.abs(selected[i]-expectedQ[i]));
        require(max<1e-6,'Batch orientations differ from individual run: '+max);
        const pixels=file.get('zebsdsolve/Maps/IPFZ_RGBA').value;
        require(pixels.every((v,i)=>v===expected[k].ipf[i]),'Native IPF map changed');
        require(file.get('zebsdsolve/Maps/NCC').shape.join(',')==='55,75','Quality raster shape');
      }finally{file.close();h.FS.unlink('/batch-result.h5');}
      for (const [suffix, reference] of [['_IPF-Z.png',expected[k].ipf],['_NCC.png',expected[k].ncc]]) {
        const bitmap=await createImageBitmap(await read(prefix+suffix));
        require(bitmap.width===75 && bitmap.height===55,'PNG scan dimensions changed');
        const canvas=new OffscreenCanvas(75,55),context=canvas.getContext('2d');context.drawImage(bitmap,0,0);bitmap.close();
        const pixels=context.getImageData(0,0,75,55).data;
        require(pixels.length===reference.length && pixels.every((v,i)=>v===reference[i]),'PNG differs from native '+suffix);
      }
      require(await (await read(prefix+'.ang')).text()===expected[k].ang,'ANG differs from individual run');
      const config=JSON.parse(await (await read(prefix+'.json')).text());
      require(config.settings.bandwidth===expected[k].settings.bandwidth,'Frozen bandwidth');
      require(JSON.stringify(config.geometry)===JSON.stringify(expected[k].geometry),'Frozen geometry');

    }
    return {jobs:configs.map(c=>({status:c.batch.status,folder:c.batch.directory,bandwidth:c.settings.bandwidth,outputs:c.batch.outputs.length})),interactiveUnchanged:true};
  `);

  const restoration = await evaluate(`
    let indexes=0;
    const indexCells=ze.indexCells.bind(ze);ze.indexCells=(...args)=>{indexes++;return indexCells(...args);};
    const indexBegin=ze.indexBegin.bind(ze);ze.indexBegin=(...args)=>{indexes++;return indexBegin(...args);};
    document.getElementById('pat-live').checked=true;
    for (const k of [0,1,0]) {
      const row=document.querySelectorAll('#batch-jobs tr')[k];
      [...row.querySelectorAll('button')].find(button=>button.textContent==='Load').click();
      await waitBatch(()=>document.getElementById('app').dataset.busy==='0');
      require(!document.getElementById('status').classList.contains('error'),'Load failed: '+document.getElementById('status').textContent);
      require(document.getElementById('roi-text').textContent==='region '+expected[k].settings.roi_w+' × '+expected[k].settings.roi_h,'Restored ROI label stale');
      const snapshot=ze.exportSnapshot();
      require(JSON.stringify(snapshot.results)===JSON.stringify(expected[k].snapshot.results),'Restored winners differ');
      require(JSON.stringify(snapshot.candidates)===JSON.stringify(expected[k].snapshot.candidates),'Restored secondary candidates differ');
      require(new TextDecoder().decode(ze.exportText('ang'))===expected[k].ang,'Restored ANG differs');
      require(Array.from(ze.map('ipf_z')).every((v,i)=>v===expected[k].ipf[i]),'Restored IPF map differs');
      require(Array.from(ze.map('ncc')).every((v,i)=>v===expected[k].ncc[i]),'Restored NCC map differs');
      require(document.getElementById('btn-ang').disabled===false,'Restored export disabled');
      const chance=expected[k].chance;
      const threshold=(expected[k].settings.accept_nmi-chance.median)/chance.spread;
      require(document.getElementById('z-hint').textContent.includes('(z '+threshold.toFixed(1)+')'),'Restored absolute NMI threshold lost measured chance: '+document.getElementById('z-hint').textContent);
      await waitBatch(()=>document.getElementById('pat-sim-cap').textContent.includes('indexed'));
      require(!document.getElementById('pat-sim-cap').textContent.includes('live'),'Overlay reindexed');
    }
    require(indexes===0,'Completed Load performed indexing: '+indexes);
    const before=ze.exportResultCheckpoint(),state=JSON.stringify(ze.results());
    before[before.length-1]^=1;
    let rejected=false;try{ze.restoreResultCheckpoint(before);}catch{rejected=true;}
    require(rejected,'Corrupt checkpoint accepted');require(JSON.stringify(ze.results())===state,'Corrupt restore changed results');
    return {loads:3,indexes,secondaryCandidates:true,overlay:true,corruptionAtomic:true};
  `);
  console.log(JSON.stringify({first,restoration}));
  console.log('Batch checkpoint browser checks passed.');
} finally {
  clearTimeout(wallTimeout);ws?.close();chrome?.kill();server.closeAllConnections();await new Promise(resolve=>server.close(resolve));
  try {rmSync(profile,{recursive:true,force:true});} catch {}
}
