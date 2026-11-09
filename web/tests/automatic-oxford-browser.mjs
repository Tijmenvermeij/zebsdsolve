#!/usr/bin/env node
// Actual WebGPU/WASM indexing and automatic Oxford WORKERFS export, no npm.
// Node 22+: node web/tests/automatic-oxford-browser.mjs source.h5oina master.h5 [output.h5oina]
import assert from "node:assert/strict";
import { spawn } from "node:child_process";
import { createServer } from "node:http";
import { createReadStream, existsSync, mkdtempSync, rmSync, statSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { basename, dirname, extname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
const root = resolve(fileURLToPath(new URL("../..", import.meta.url))), acquisition = process.argv[2] && resolve(process.argv[2]), master = process.argv[3] && resolve(process.argv[3]), outputPath = process.argv[4] && resolve(process.argv[4]);
if (outputPath) {
  assert.equal(extname(outputPath).toLowerCase(), ".h5oina");
  assert.ok(existsSync(dirname(outputPath)), "Output directory must exist");
  assert.ok(!existsSync(outputPath), "Output file already exists");
  assert.notEqual(outputPath, acquisition); assert.notEqual(outputPath, master);
}
assert.ok(acquisition && master && existsSync(acquisition) && existsSync(master), "Pass existing Oxford source and EMsoft master paths");
const chromePath = process.env.CHROME || ["/Applications/Google Chrome.app/Contents/MacOS/Google Chrome", "/usr/bin/google-chrome", "/usr/bin/chromium"].find(existsSync);
assert.ok(chromePath, "Chrome is required; set CHROME");
const server = createServer((req, res) => {
  const path = decodeURIComponent(req.url.split("?")[0]);
  if (path === "/") { res.writeHead(200, { "Content-Type": "text/html" }).end('<!doctype html><html style="background:white;color:black"><body style="background:white;color:black"><input id="source" type="file" hidden><input id="master" type="file" hidden></body></html>'); return; }
  const file = resolve(root, `.${path}`);
  if (!file.startsWith(`${root}/`) || !existsSync(file)) { res.writeHead(404).end(); return; }
  res.writeHead(200, { "Content-Type": ({ ".js": "text/javascript", ".wasm": "application/wasm" })[extname(file)] || "application/octet-stream" });
  createReadStream(file).pipe(res);
});
const profile = mkdtempSync(join(tmpdir(), "oxford-export-chrome-"));
let chrome, ws;
try {
  await new Promise((resolve, reject) => { server.once("error", reject); server.listen(0, "127.0.0.1", resolve); });
  const base = `http://127.0.0.1:${server.address().port}`;
  chrome = spawn(chromePath, ["--headless=new", "--remote-debugging-port=0", `--user-data-dir=${profile}`, "--no-first-run", "--no-default-browser-check", "--enable-unsafe-webgpu", "--ignore-gpu-blocklist", "--enable-features=Vulkan", "about:blank"], { stdio: ["ignore", "ignore", "pipe"] });
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
  if (outputPath) await send("Browser.setDownloadBehavior", { behavior: "allow", downloadPath: dirname(outputPath) });
  await send("Page.enable", {}, sessionId);
  const loaded = new Promise(resolve => { pageLoaded = resolve; });
  await send("Page.navigate", { url: `${base}/` }, sessionId);
  await loaded;
  await evaluate('while (!document.getElementById("source")) await new Promise(r => setTimeout(r, 20));');
  const { root: documentRoot } = await send("DOM.getDocument", {}, sessionId);
  for (const [selector, file] of [["#source", acquisition], ["#master", master]]) {
    const { nodeId } = await send("DOM.querySelector", { nodeId: documentRoot.nodeId, selector }, sessionId);
    await send("DOM.setFileInputFiles", { nodeId, files: [file] }, sessionId);
  }
  const result = await evaluate(`
    const require = (ok, message) => { if (!ok) throw new Error(message); };
    const { Zebsdsolve } = await import('/web/zebsdsolve.js');
    const { exportOxford } = await import('/web/app/js/hdf5-export.js');
    const ze = await Zebsdsolve.create({wasm:'/zig-out/web/zebsdsolve.wasm',warm:false});
    let output;
    try {
      const source = document.getElementById('source').files[0], master = document.getElementById('master').files[0];
      await ze.openScan([source]); await ze.addPhase(master);
      const originalGeometry = ze.geometry;
      require(originalGeometry.rot.every(v => v === 0), 'Imported detector residual must be zero');
      require(originalGeometry.detectorTiltDeg === 0 && originalGeometry.pxUm === 0, 'Imported tilt residual and scan translation must be zero');
      require(ze.settings.mask === false, 'Oxford pattern mask must default to false');
      ze.configure({roi_x:165,roi_y:284,roi_w:1,roi_h:1,bandwidth:88,static_background:'average',dynamic_background:'subtract'});
      await ze.index();
      const raw = ze.exportSnapshot(), ox = ze.exportOxfordSnapshot();
      const cell = 284 * ox.scan.cols + 165;
      require(cell === 83377, 'Expected P6 reference pixel');
      require(ox.results.phase[cell] === 0, 'Reference point must be indexed by the Ti master');
      const frame = ox.oxford_frame;
      require(frame.automatic === true && frame.convention_version === 1, 'Automatic native Oxford frame missing');
      const close = (a,b,t=1e-12) => a.length===b.length && a.every((v,i)=>Math.abs(v-b[i])<=t);
      require(close(frame.engine_to_cs1,[Math.SQRT1_2,0,0,Math.SQRT1_2]), 'Unexpected frozen sample basis');
      require(close(frame.phase_crystal_to_master[0],[Math.cos(Math.PI/12),0,0,Math.sin(Math.PI/12)]), 'Missing hexagonal crystal basis');
      source.arrayBuffer = () => { throw new Error('Whole-source read is forbidden'); };
      const bytes = await exportOxford(ox,source,frame);
      const h = await import('/web/app/js/vendor/h5wasm/hdf5_hl.js'); await h.ready;
      h.FS.writeFile('/automatic.h5oina',bytes); output = new h.File('/automatic.h5oina','r');
      const scanRoot = output.keys().find(key => output.get(key) instanceof h.Group && output.get(key).keys().includes('EBSD'));
      require(!!scanRoot,'No Oxford output scan');
      const path = scanRoot+'/EBSD', data = output.get(path+'/Data'), header = output.get(path+'/Header');
      const euler = data.get('Euler'), saved = Array.from(euler.value.slice(cell*3,cell*3+3));
      require(['rad','radians'].includes(euler.attrs.Unit.value),'Oxford Euler must declare radians');
      const expected = ox.results.euler.slice(cell*3,cell*3+3);
      require(close(saved,expected,1e-6),'Oxford Euler serialization changed orientation');
      const phaseMap = output.get('zebsdsolve/PhaseIDMap').value;
      require(data.get('Phase').value[cell]===phaseMap[0],'Oxford phase mapping differs from native selected phase');
      for (const name of ['Processed Patterns','Unprocessed Patterns','Raw Patterns']) require(!data.get(name),'Diffraction patterns must be omitted');
      const backup = output.get('zebsdsolve/OriginalCalibration');
      require(backup?.get('Detector Orientation Euler') && backup?.get('Tilt Angle'),'Original calibration backup missing');
      require(close(Array.from(header.get('Detector Orientation Euler').value),ox.calibration.effective_detector_euler_radians,1e-6),'Effective detector Euler differs from native snapshot');
      require(Math.abs(header.get('Tilt Angle').value[0]-ox.calibration.effective_tilt_radians)<1e-6,'Effective tilt differs from native snapshot');
      require(close(Array.from(header.get('Specimen Orientation Euler').value),ox.calibration.source_specimen_euler_radians,1e-6),'Specimen orientation changed');
      const provenance = JSON.parse(output.get('zebsdsolve/ProvenanceJSON').value);
      require(JSON.stringify(provenance.oxford_frame)===JSON.stringify(frame),'Native frame provenance changed');
      require(JSON.stringify(provenance.calibration)===JSON.stringify(ox.calibration),'Native calibration provenance changed');
      ${outputPath ? `const downloadURL = URL.createObjectURL(new Blob([bytes],{type:'application/x-hdf5'}));
      const link = document.createElement('a'); link.href = downloadURL; link.download = ${JSON.stringify(basename(outputPath))}; document.body.append(link); link.click(); link.remove();` : ''}
      return {pixel:cell,rows:ox.scan.rows,cols:ox.scan.cols,sourceBytes:source.size,outputBytes:bytes.length,geometry:originalGeometry,frame,calibration:ox.calibration,raw:{quaternion:raw.results.quaternions.slice(cell*4,cell*4+4),euler:raw.results.euler.slice(cell*3,cell*3+3)},oxford:{quaternion:ox.results.quaternions.slice(cell*4,cell*4+4),euler:expected,savedEuler:saved,phase:data.get('Phase').value[cell]},quality:{ncc:ox.results.ncc[cell],nmi:ox.results.nmi[cell],z:ox.results.z[cell]}};
    } finally { output?.close(); ze.x.ze_destroy(ze.e); }
  `);
  assert.ok(result.outputBytes < result.sourceBytes);
  if (outputPath) {
    const deadline = Date.now() + 30000;
    while ((!existsSync(outputPath) || existsSync(outputPath + ".crdownload")) && Date.now() < deadline)
      await new Promise(resolve => setTimeout(resolve, 100));
    assert.ok(existsSync(outputPath) && !existsSync(outputPath + ".crdownload"), "Browser download did not complete within 30 seconds");
    assert.equal(statSync(outputPath).size, result.outputBytes, "Downloaded size differs from browser export");
    result.outputPath = outputPath;
    writeFileSync(outputPath + ".json", JSON.stringify(result, null, 2) + "\n");
  }
  console.log(JSON.stringify(result));
  console.log('Actual WASM automatic Oxford export passed: WebGPU indexing, native sample/crystal conversion, WORKERFS export, calibration/provenance, phase/Euler serialization and pattern omission.');
} finally {
  ws?.close(); chrome?.kill(); await new Promise(resolve => server.close(resolve));
  try { rmSync(profile, { recursive: true, force: true }); } catch { /* Chrome may still be shutting down. */ }
}
