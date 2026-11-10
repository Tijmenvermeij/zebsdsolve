#!/usr/bin/env node
// Example-specific background settings in the actual app, no npm.
// Node 22+: node web/tests/example-background-browser.mjs
import assert from "node:assert/strict";
import { spawn } from "node:child_process";
import { createServer } from "node:http";
import { createReadStream, existsSync, mkdtempSync, rmSync } from "node:fs";
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
  console.error('Example background browser test exceeded 180 seconds');
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

  await evaluate(`for(let i=0;i<600 && !window.ze;i++) await new Promise(r=>setTimeout(r,50));if(!window.ze)throw new Error('No engine');await ze._queue;`);
  const result = await evaluate(`
    const pause = ms => new Promise(r => setTimeout(r, ms));
    const wait = async predicate => { for (let i=0; i<2400; i++) { if (predicate()) return; await pause(50); } throw Error('timed out: '+document.querySelector('#status').textContent); };
    const bg = () => [ze.settings.static_background, ze.settings.dynamic_background];
    const check = (actual, expected, context) => { if (JSON.stringify(actual)!==JSON.stringify(expected)) throw Error(context+': '+JSON.stringify(actual)); };
    const clickExample = async () => { document.querySelector('#btn-example').click(); await wait(() => document.querySelector('#app').dataset.busy === '1'); await wait(() => document.querySelector('#app').dataset.busy === '0'); };
    const drop = async files => { const dt = new DataTransfer(); files.forEach(f => dt.items.add(f)); window.dispatchEvent(new DragEvent('drop', {dataTransfer: dt})); await pause(100); await wait(() => document.querySelector('#app').dataset.busy !== '1'); };
    const scan = new File([await (await fetch('./presets/nickel/patterns_v2.h5')).blob()], 'other.h5');
    ze.configure({static_background:'average', dynamic_background:'divide'});
    await clickExample(); check(bg(), ['file','subtract'], 'example');
    const selects = [...document.querySelectorAll('label.field')];
    const value = name => selects.find(e => e.querySelector('span')?.textContent === name)?.querySelector('select')?.value;
    check([value('static background'), value('dynamic background')], ['file','subtract'], 'controls');
    const result = await ze.indexCells([0], {method:'spherical', bandwidth:8});
    if (!result || !result.length) throw Error('indexing returned no result');
    await clickExample(); check(bg(), ['file','subtract'], 'repeat');
    const originalFetch = window.fetch;
    window.fetch = async (...args) => String(args[0]).includes('Ni_master_20kV.npz') ? new Response('', {status:404}) : originalFetch(...args);
    await clickExample(); window.fetch = originalFetch;
    check(bg(), ['file','subtract'], 'failed example fetch');
    ze.configure({static_background:'none', dynamic_background:'none', fit_bandwidth: 16});
    await drop([new File(['invalid'],'bad.h5')]); check(bg(), ['none','none'], 'failed scan');
    await drop([scan]); check(bg(), ['average','divide'], 'restore');
    if (ze.settings.fit_bandwidth !== 16) throw Error('unrelated setting lost');
    await clickExample();
    const project = new File([JSON.stringify({settings:{static_background:'none',dynamic_background:'subtract',sweeps:2,ipf_colors:'hue'}})], 'settings.json');
    await drop([scan, project]); check(bg(), ['none','subtract'], 'project precedence');
    if (!document.querySelector('#sweep-readout').textContent.includes('→')) throw Error('project sweep readout was not refreshed');
    if (ze.settings.ipf_colors !== 'hue') throw Error('project IPF colors were not applied');
    await clickExample();
    await drop([new File([JSON.stringify({settings:{static_background:'file', dynamic_background:'divide'}})],'only.json')]);
    await drop([scan]); check(bg(), ['none','subtract'], 'project only baseline');
    await clickExample();
    ze.configure({toy_rows:1, toy_cols:1, toy_width:16, toy_height:16, toy_grains:1});
    document.querySelector('#btn-toy').click(); await wait(() => document.querySelector('#app').dataset.busy === '1'); await wait(() => document.querySelector('#app').dataset.busy === '0');
    check(bg(), ['none','subtract'], 'toy restore');
    if (!document.querySelector('#scan-meta').textContent.includes('toy scan')) throw Error('toy scan failed');
    return 'example preset, controls, indexing, repeat, failed scan, restore, unrelated settings, project precedence, project-only edits and toy passed';
  `);
  console.log(result);
} finally {
  clearTimeout(wallTimeout);ws?.close();chrome?.kill();server.closeAllConnections();await new Promise(resolve=>server.close(resolve));
  try {rmSync(profile,{recursive:true,force:true});} catch {}
}
