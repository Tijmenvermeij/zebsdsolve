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
      expected.push({snapshot:ze.exportSnapshot(),ang:new TextDecoder().decode(ze.exportText('ang')),ipf:Array.from(ze.map('ipf_z')),ncc:Array.from(ze.map('ncc')),settings:ze.settings,geometry:ze.geometry});
    };
    await addConfigured({bandwidth:16,fit_bandwidth:16,refine_iters:1,roi_x:10,roi_y:10,roi_w:2,roi_h:1});
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

  const loading = await evaluate(`
    ze.configure({binning:1,sweeps:2});
    const row=document.querySelectorAll('#batch-jobs tr')[1];[...row.querySelectorAll('button')].find(b=>b.textContent==='Load').click();
    await waitBatch(()=>document.getElementById('app').dataset.busy==='0' && document.getElementById('tab-index').getAttribute('aria-selected')==='true');
    require(JSON.stringify(ze.geometry.pc)===JSON.stringify(expected[1].geometry.pc),'Load did not restore fitted geometry');
    require(document.getElementById('bin-choices').value==='2' && !document.getElementById('pat-full-box').hidden,'Saved binning controls stale');
    require(document.getElementById('sweep-readout').textContent.includes('one pass'),'Saved sweeps stale');
    document.getElementById('btn-geo-file').click();
    require(JSON.stringify(ze.geometry.pc)===JSON.stringify(fixtureBaseGeo.pc),'As-opened geometry became fitted geometry');
    document.getElementById('btn-batch-clear').click();await pause(100);
    document.getElementById('btn-batch-current').click();await waitBatch(()=>document.querySelectorAll('#batch-jobs tr').length===1);
    let permissionCalls=0;const granted=async()=> 'granted',prompt=async()=> 'prompt';
    const access=requestPermission=>({name:testOutput.name,getDirectoryHandle:testOutput.getDirectoryHandle.bind(testOutput),getFileHandle:testOutput.getFileHandle.bind(testOutput),queryPermission:prompt,requestPermission});
    window.showDirectoryPicker=async()=>access(async()=>{permissionCalls++;return 'denied';});
    document.getElementById('btn-batch-folder').click();await pause(100);
    document.getElementById('btn-batch-run').click();await waitBatch(()=>document.querySelector('#batch-folder-dialog')?.open);
    require(document.getElementById('btn-batch-clear').disabled,'Clear allowed during folder preparation');
    document.getElementById('btn-batch-clear').dispatchEvent(new Event('click'));
    require(document.querySelectorAll('#batch-jobs tr').length===1,'Clear discarded pending queue');
    document.querySelector('#batch-folder-dialog .primary').click();
    await waitBatch(()=>document.querySelector('#batch-folder-dialog .folder-error')?.textContent.includes('not granted'));
    require(document.querySelector('#batch-jobs tr').dataset.status==='queued','Permission rejection indexed a job');
    require(permissionCalls===1,'Unexpected permission requests');
    [...document.querySelectorAll('#batch-folder-dialog button')].find(button=>button.textContent==='Cancel').click();
    await waitBatch(()=>document.getElementById('app').dataset.batchRunning==='0');
    window.showDirectoryPicker=async()=>access(()=>{permissionCalls++;return new Promise(resolve=>window.resolvePermission=resolve);});
    document.getElementById('btn-batch-folder').click();await pause(100);
    document.getElementById('btn-batch-run').click();await waitBatch(()=>document.querySelector('#batch-folder-dialog')?.open);
    document.querySelector('#batch-folder-dialog .primary').click();await waitBatch(()=>!!window.resolvePermission);
    document.getElementById('btn-batch-run').dispatchEvent(new Event('click'));
    [...document.querySelectorAll('#batch-folder-dialog button')].find(button=>button.textContent==='Cancel').click();
    resolvePermission('granted');await waitBatch(()=>document.getElementById('app').dataset.batchRunning==='0');
    require(permissionCalls===2,'Duplicate launch entered permission path');
    require(document.querySelector('#batch-jobs tr').dataset.status==='queued','Cancelled preparation started indexing');
    require([...document.querySelectorAll('#tabs button')].every(button=>!button.disabled),'Tabs stayed locked');
    document.getElementById('btn-batch-clear').click();await pause(100);
    window.showDirectoryPicker=async()=>({name:testOutput.name,getDirectoryHandle:testOutput.getDirectoryHandle.bind(testOutput),getFileHandle:testOutput.getFileHandle.bind(testOutput),queryPermission:granted,requestPermission:granted});
    return {geometryBaseline:true,denialVisible:true,launchSerialized:true,stopBeforeIndex:true};
  `);
  console.error('batch: job load and launch guards passed');
  async function inputFile(selector,path) {
    const {root:doc}=await send('DOM.getDocument',{},sessionId);
    const {nodeId}=await send('DOM.querySelector',{nodeId:doc.nodeId,selector},sessionId);
    await send('DOM.setFileInputFiles',{nodeId,files:[path]},sessionId);
    await evaluate(`document.querySelector(${JSON.stringify(selector)}).dispatchEvent(new Event('change'));await pause(100);await waitBatch(()=>document.getElementById('app').dataset.busy==='0');`);
  }
  await inputFile('#file-phase',join(root,'testdata/nickel/Ni_master_20kV.npz'));
  const failedLoad=await evaluate(`
    require(ze.phases.length===2,'Second master fixture missing');
    window.dropFixture=async file=>{const dt=new DataTransfer();dt.items.add(file);window.dispatchEvent(new DragEvent('drop',{dataTransfer:dt}));await pause(100);await waitBatch(()=>document.getElementById('app').dataset.busy==='0');await ze._queue;};
    const other=new File([await fixtureSource.arrayBuffer()],'other-acquisition.h5');
    await dropFixture(other);document.getElementById('btn-batch-current').click();await waitBatch(()=>document.querySelectorAll('#batch-jobs tr').length===1);
    await dropFixture(fixtureSource);
    const add=ze.addPhase.bind(ze);let calls=0;ze.addPhase=async(...args)=>{if(++calls===2)throw Error('Test missing second master');return add(...args);};
    const row=document.querySelector('#batch-jobs tr');[...row.querySelectorAll('button')].find(b=>b.textContent==='Load').click();
    await waitBatch(()=>document.getElementById('status').textContent.includes('Test missing second master') && document.getElementById('app').dataset.busy==='0');ze.addPhase=add;
    require(ze.phases.length===1 && document.querySelectorAll('#phases li').length===1,'Partial phase list stale');
    require(document.getElementById('scan-meta').textContent.includes('other-acquisition.h5'),'Loaded source label stale');
    require([...document.querySelectorAll('.needs-results')].every(b=>b.disabled),'Previous results still exportable');
    document.getElementById('btn-batch-current').click();await waitBatch(()=>document.querySelectorAll('#batch-jobs tr').length===2);
    require([...document.querySelectorAll('#batch-jobs tr')].at(-1).textContent.includes('other-acquisition.h5'),'Current source file is stale');
    document.getElementById('btn-batch-clear').click();await pause(100);
    document.querySelector('#phases button').click();await pause(100);await waitBatch(()=>document.getElementById('app').dataset.busy==='0' && ze.phases.length===0);
    return {sourceSynchronized:true,partialPhasesSynchronized:true,oldResultsInvalidated:true};
  `);
  console.error('batch: failed master load leaves coherent new acquisition');
  await inputFile('#file-scan',acquisition);
  await inputFile('#file-phase',process.argv[3] || '/Users/tvermeij/EMsoftData/TiOverlap/Ti_30kv_HCP.h5');
  await evaluate(`
    require(ze.scan.rows===415 && ze.phases.length===1,'P6 acquisition fixture failed');
    ze.configure({bandwidth:16,fit_bandwidth:16,refine_iters:1,roi_x:165,roi_y:284,roi_w:1,roi_h:1,static_background:'none',dynamic_background:'subtract'});
    const originalPC=ze.geometry.pc;
    ze.setGeometry({planeKind:1,plane:[1e-7,0,originalPC[0],0,1e-7,originalPC[1],0,0,originalPC[2],0,0,1]});
    document.getElementById('btn-batch-current').click();await waitBatch(()=>document.querySelectorAll('#batch-jobs tr').length===1);
    require(document.querySelector('.batch-warning')?.textContent.includes('Oxford PCs retained'),'Plane warning not visible');
    await editOutputName(document.querySelector('#batch-jobs tr'),'Ti_P6_test');
    document.getElementById('btn-batch-folder').click();await waitBatch(()=>document.querySelector('#batch-jobs tr').children[1].textContent===testOutput.name);
    await ze.index();window.expectedOxford=ze.exportOxfordSnapshot();window.expectedP6=ze.exportSnapshot();
    document.getElementById('btn-batch-run').click();await waitBatch(()=>document.querySelector('#batch-jobs tr').dataset.status==='complete' && document.getElementById('app').dataset.busy==='0');
  `);
  const oxford=await evaluate(`
    const dir=await testOutput.getDirectoryHandle('Ti_P6_test'),file=await(await dir.getFileHandle('Ti_P6_test.h5oina')).getFile();
    require(file.size<acquisitionSize,'Oxford export unexpectedly contains patterns');
    const h=await import('/web/app/js/vendor/h5wasm/hdf5_hl.js');await h.ready;h.FS.writeFile('/p6-result.h5oina',new Uint8Array(await file.arrayBuffer()));const out=new h.File('/p6-result.h5oina','r');
    try{
      const group=out.keys().find(key=>out.get(key) instanceof h.Group && out.get(key).keys().includes('EBSD'));
      const data=out.get(group+'/EBSD/Data'),header=out.get(group+'/EBSD/Header'),cell=284*293+165;
      require(!data.get('Processed Patterns')&&!data.get('Unprocessed Patterns'),'Oxford patterns must be absent');
      const euler=data.get('Euler').value;
      for(let j=0;j<3;j++)require(Math.abs(euler[cell*3+j]-expectedOxford.results.euler[cell*3+j])<1e-6,'Oxford frame differs from individual run');
      const sourcePCs=[['Pattern Center X',[0.480416476726532,0.48041751980781555,0.4804185628890991]],['Pattern Center Y',[1.2126353979110718,1.2126353979110718,1.2126353979110718]],['Detector Distance',[0.5900739431381226,0.5900739431381226,0.5900739431381226]]];
      for(const [field,values] of sourcePCs) for(let j=0;j<3;j++)require(data.get(field).value[j]===values[j],'Original Oxford PC changed: '+field);
      const provenance=JSON.parse(out.get('zebsdsolve/ProvenanceJSON').value);
      require(provenance.oxford_frame.preserve_source_pattern_centres===true,'Plane preservation policy missing');
      const frame={...provenance.oxford_frame},reference={...expectedOxford.oxford_frame};delete frame.phase_generation;delete reference.phase_generation;
      require(JSON.stringify(frame)===JSON.stringify(reference),'Oxford frame provenance changed');
      const tilt=header.get('Tilt Angle').value[0];require(Math.abs(tilt-expectedOxford.calibration.effective_tilt_radians)<1e-6,'Oxford tilt changed');
      const ids=out.get('zebsdsolve/PhaseIDMap').value;require(data.get('Phase').value[cell]===ids[expectedOxford.results.phase[cell]],'Oxford phase mapping wrong');
    }finally{out.close();h.FS.unlink('/p6-result.h5oina');}
    require(JSON.parse(await(await(await(await testOutput.getDirectoryHandle('Ni_test_02')).getFileHandle('Ni_test_02.json')).getFile()).text()).batch.status==='complete','Earlier job journal overwritten');
    return {status:'complete',outputBytes:file.size,patternsOmitted:true,frameMatchesIndividual:true};
  `.replace('acquisitionSize',String((await import('node:fs')).statSync(acquisition).size)));
  console.error('batch: original P6 Oxford output validated');
  const ownership=await evaluate(`
    const {BatchQueue}=await import('/web/app/js/batch.js');
    const dir=await testOutput.getDirectoryHandle('native-retry-ownership',{create:true});
    let fail=true,indexes=0;const handles=new Map();
    const proxy={name:dir.name,queryPermission:async()=> 'granted',getDirectoryHandle:dir.getDirectoryHandle.bind(dir),getFileHandle:dir.getFileHandle.bind(dir)};
    const getDir=proxy.getDirectoryHandle;
    proxy.getDirectoryHandle=async(...args)=>{const sub=await getDir(...args);return {name:sub.name,getDirectoryHandle:sub.getDirectoryHandle.bind(sub),getFileHandle:async(...a)=>{
      const actual=await sub.getFileHandle(...a);if(!a[0].endsWith('.h5'))return actual;
      const handle={native:actual,name:actual.name,getFile:actual.getFile.bind(actual),isSameEntry:other=>actual.isSameEntry(other.native||other),createWritable:async()=>{
        const stream=await actual.createWritable();return {write:async data=>{await stream.write(data);if(fail){fail=false;throw Error('Injected native write failure');}},close:stream.close.bind(stream),abort:stream.abort.bind(stream)};
      }};handles.set('output',actual);return handle;
    }};};
    const info={rows:1,cols:1,patH:1,patW:1,dtype:'uint8',format:'fixture'};
    const queue=new BatchQueue({createEngine:async()=>({exportResultCheckpoint(){return new Uint8Array([1]);},scan:info,device:{destroy(){}},configure(){},openScan:async()=>info,setGeometry(){},clearPhases(){},addPhase:async()=>{},indexBegin:async()=>{indexes++;},get indexStatus(){return {total:1,done:0};},indexStep:async()=>({left:0,total:1,done:1,sweep:0,sweeps:1}),dispose:async()=>{}}),artifacts:async function*(engine,job){yield{name:job.outputStem+'.h5',data:'own result'};yield{name:job.outputStem+'.ang',data:'own angles'};for(const suffix of ['_IPF-Z.png','_NCC.png'])yield{name:job.outputStem+suffix,data:'pixels'};}});
    queue.setDirectory(proxy);queue.add({sourceFiles:[new File(['x'],'scan.h5')],masterFiles:[new File(['x'],'master.npz')],settings:{batch:1},geometry:{},scanInfo:info});
    await queue.run();require(queue.jobs[0].status==='export-pending','Expected native write failure');
    const folder=await dir.getDirectoryHandle(queue.jobs[0].folderName),original=handles.get('output');
    const outputName=queue.jobs[0].outputStem+'.h5';await folder.removeEntry(outputName);const foreign=await folder.getFileHandle(outputName,{create:true});const writer=await foreign.createWritable();await writer.write('foreign content');await writer.close();
    const sameLocator=await original.isSameEntry(foreign);
    await queue.run();
    require(queue.jobs[0].status==='export-pending' && indexes===1,'Native retry must not reindex');
    require(await(await foreign.getFile()).text()==='foreign content','Native foreign file overwritten');
    await queue.clear();return {sameLocator,foreignPreserved:true,indexes};
  `);
  console.error('batch: native OPFS replacement guard passed');

  const screenshot=await send('Page.captureScreenshot',{format:'png'},sessionId);
  const {writeFileSync}=await import('node:fs');writeFileSync('/tmp/zebsdsolve-batch.png',Buffer.from(screenshot.data,'base64'));
  console.log(JSON.stringify({first,loading,failedLoad,oxford,ownership}));
  console.log('Batch browser checks passed: two frozen jobs, collision-safe five-file exports, exact native maps/ANG, independent orientation comparison and preserved interactive state.');
} finally {
  clearTimeout(wallTimeout);ws?.close();chrome?.kill();server.closeAllConnections();await new Promise(resolve=>server.close(resolve));
  try {rmSync(profile,{recursive:true,force:true});} catch {}
}
