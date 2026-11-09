#!/usr/bin/env node
// Real browser WORKERFS test, no npm dependencies. Optional acquisition path tests
// large-file metadata roundtrip without reading any diffraction pattern values.
// Node 22+: node web/tests/export-oxford-browser.mjs /path/to/source.h5oina
import assert from "node:assert/strict";
import { spawn } from "node:child_process";
import { createServer } from "node:http";
import { createReadStream, existsSync, mkdtempSync, rmSync, statSync } from "node:fs";
import { tmpdir } from "node:os";
import { extname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
const root = resolve(fileURLToPath(new URL("../..", import.meta.url))), acquisition = process.argv[2] && resolve(process.argv[2]);
const chromePath = process.env.CHROME || ["/Applications/Google Chrome.app/Contents/MacOS/Google Chrome", "/usr/bin/google-chrome", "/usr/bin/chromium"].find(existsSync);
assert.ok(chromePath, "Chrome is required; set CHROME");
const server = createServer((req, res) => {
  const path = decodeURIComponent(req.url.split("?")[0]);
  if (path === "/") { res.writeHead(200, { "Content-Type": "text/html" }).end('<!doctype html><html style="background:white;color:black"><body style="background:white;color:black"><input id="source" type="file"></body></html>'); return; }
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
  chrome = spawn(chromePath, ["--headless=new", "--remote-debugging-port=0", `--user-data-dir=${profile}`, "--no-first-run", "--no-default-browser-check", "about:blank"], { stdio: ["ignore", "ignore", "pipe"] });
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
  await send("Page.navigate", { url: `${base}/` }, sessionId);
  await loaded;
  await evaluate('while (!document.getElementById("source")) await new Promise(r => setTimeout(r, 20));');
  // First exercise the production export worker on a small complete fixture.
  const production = await evaluate(`
    const h = await import('/web/app/js/vendor/h5wasm/hdf5_hl.js'); await h.ready;
    const { exportOxford } = await import('/web/app/js/hdf5-export.js');
    const put = (g, name, data, shape) => g.create_dataset({name, data, ...(shape ? {shape} : {})});
    const f = new h.File('/fixture.h5oina','w'), root = f.create_group('1'), eb = root.create_group('EBSD'), hd = eb.create_group('Header'), d = eb.create_group('Data');
    for (const [key,v] of [['X Cells',1],['Y Cells',1],['Pattern Width',10],['Pattern Height',8]]) put(hd,key,new Uint32Array([v]));
    for (const key of ['X Step','Y Step']) put(hd,key,new Float64Array([1])).create_attribute('Unit','um');
    const pg = hd.create_group('Phases').create_group('1');
    put(pg,'Space Group',new Uint32Array([229])); put(pg,'Laue Group',new Uint32Array([11]));
    put(pg,'Lattice Dimensions',new Float64Array([3,3,3]),[1,3]).create_attribute('Unit','angstrom');
    put(pg,'Lattice Angles',new Float64Array(3).fill(Math.PI/2),[1,3]).create_attribute('Unit','rad');
    put(d,'Euler',new Float64Array([.8,.8,.8]),[1,3]).create_attribute('Unit','rad'); put(d,'Phase',new Uint8Array([1]));
    for (const key of ['X','Y']) put(d,key,new Float64Array([0])).create_attribute('Unit','um');
    put(d,'Processed Patterns',new Uint8Array(80),[1,8,10]);
    put(d,'Processed Static Background',new Uint8Array(80).fill(17),[8,10]); f.close();
    const source = new File([h.FS.readFile('/fixture.h5oina')],'fixture.h5oina'); h.FS.unlink('/fixture.h5oina');
    source.arrayBuffer = () => { throw new Error('Whole-source arrayBuffer is forbidden'); };
    const snapshot = {schema_version:1,version:'browser-worker-test',scan:{rows:1,cols:1,pat_w:10,pat_h:8,format:'h5_oxford'},geometry:{step_x_um:1,step_y_um:1},settings:{phase_by:'nmi'},phases:[{name:'Fe',space_group:229,laue:11,lattice:[3,3,3,90,90,90]}],results:{euler:[.1,.2,.3],quaternions:[1,0,0,0],phase:[0],ncc:[.8],nmi:[.7],z:[8],sweep:[0]},pattern_centres:[.4,.5,.6]};
    const result = await exportOxford(snapshot, source, {engine_to_cs1:[1,0,0,0],provenance:'serialization fixture'});
    h.FS.writeFile('/result.h5oina',result); const output = new h.File('/result.h5oina','r');
    const checked = {bytes:result.length,euler:output.get('1/EBSD/Data/Euler').value[0],patterns:!!output.get('1/EBSD/Data/Processed Patterns'),background:output.get('1/EBSD/Data/Processed Static Background').value[0]}; output.close();h.FS.unlink('/result.h5oina');return checked;
  `);
  assert.equal(production.euler, .1); assert.equal(production.patterns, false); assert.equal(production.background, 17);
  console.log(`Production Oxford worker passed: ${JSON.stringify(production)}`);
  if (acquisition) {
    const { root: documentRoot } = await send("DOM.getDocument", {}, sessionId);
    const { nodeId } = await send("DOM.querySelector", { nodeId: documentRoot.nodeId, selector: "#source" }, sessionId);
    await send("DOM.setFileInputFiles", { nodeId, files: [acquisition] }, sessionId);
    const workerCode = `
      import * as h from '${base}/web/app/js/vendor/h5wasm/hdf5_hl.js';
      import {copyOxfordMetadata} from '${base}/web/app/js/hdf5-metadata-copy.js';
      self.onmessage = async ({data:source}) => {
        let input, output;
        try {
          await h.ready;
          Blob.prototype.arrayBuffer = () => { throw new Error('Whole-source arrayBuffer is forbidden'); };
          let readBytes = 0, maxRead = 0;
          const read = FileReaderSync.prototype.readAsArrayBuffer;
          FileReaderSync.prototype.readAsArrayBuffer = function(blob) { readBytes += blob.size; maxRead = Math.max(maxRead, blob.size); return read.call(this,blob); };
          h.FS.mkdir('/source');h.FS.mount(h.FS.filesystems.WORKERFS,{blobs:[{name:'acquisition.h5oina',data:source}]},'/source');
          input = new h.File('/source/acquisition.h5oina','r'); output = new h.File('/metadata.h5oina','w');
          copyOxfordMetadata(input,output,h); output.flush();
          const excluded = new Set(['Processed Patterns','Unprocessed Patterns','Raw Patterns']), visited = new Set();
          let datasets = 0, attributes = 0, omitted = 0;
          function same(a,b) { if (ArrayBuffer.isView(a)) return a.constructor.name===b?.constructor?.name && a.length===b.length && a.every((v,i)=>Object.is(v,b[i])); if (Array.isArray(a)) return Array.isArray(b) && a.length===b.length && a.every((v,i)=>same(v,b[i])); return Object.is(a,b); }
          function require(ok,path) { if(!ok) throw new Error('Metadata mismatch: '+path); }
          function visit(g,path) {
            const id = Array.from(g.create_reference().ref_data).join(','); if (visited.has(id)) return; visited.add(id);
            const target = output.get(path);
            for (const [key,a] of Object.entries(g.attrs)) {const b=target.attrs[key];require(b && JSON.stringify(a.dtype)===JSON.stringify(b.dtype) && JSON.stringify(a.shape)===JSON.stringify(b.shape) && same(a.value,b.value),path+'@'+key);attributes++;}
            for(const name of g.keys()) {
              const p=(path==='/'?'':path)+'/'+name, d=g.get(name);
              if(excluded.has(name)) {require(!output.get(p),p+' omitted');omitted++;continue;}
              if(d instanceof h.Group) visit(d,p);
              else if(d instanceof h.Dataset) {const b=output.get(p);require(b && JSON.stringify(d.dtype)===JSON.stringify(b.dtype) && JSON.stringify(d.shape)===JSON.stringify(b.shape) && same(d.value,b.value),p);for(const [key,a] of Object.entries(d.attrs)){const c=b.attrs[key];require(c && JSON.stringify(a.dtype)===JSON.stringify(c.dtype) && JSON.stringify(a.shape)===JSON.stringify(c.shape) && same(a.value,c.value),p+'@'+key);attributes++;}datasets++;}
            }
          }
          visit(input,'/');
          self.postMessage({sourceBytes:source.size,outputBytes:h.FS.stat('/metadata.h5oina').size,datasets,attributes,omitted,readBytes,maxRead});
        } catch(error) {self.postMessage({error:error.message});}
        finally {input?.close();output?.close();self.close();}
      };
    `;
    const result = await evaluate(`
      const url = URL.createObjectURL(new Blob([${JSON.stringify(workerCode)}],{type:'text/javascript'}));
      try { return await new Promise((resolve,reject)=>{const w=new Worker(url,{type:'module'});w.onmessage=({data})=>{w.terminate();data.error?reject(new Error(data.error)):resolve(data)};w.onerror=e=>{w.terminate();reject(new Error(e.message))};w.postMessage(document.getElementById('source').files[0]);}); }
      finally {URL.revokeObjectURL(url);}
    `);
    assert.equal(result.sourceBytes, statSync(acquisition).size); assert.ok(result.datasets > 0); assert.ok(result.omitted > 0);
    assert.ok(result.outputBytes < result.sourceBytes); assert.ok(result.maxRead < result.sourceBytes / 10, "must use bounded random reads");
    console.log(`WORKERFS metadata roundtrip passed: ${JSON.stringify(result)}`);
  }
} finally {
  ws?.close(); chrome?.kill(); await new Promise(resolve => server.close(resolve));
  try { rmSync(profile, { recursive: true, force: true }); } catch { /* Chrome may still be shutting down. */ }
}
