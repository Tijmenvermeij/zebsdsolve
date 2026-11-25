#!/usr/bin/env node
// Overlap in the actual app: primary reuse, inspection, frozen export and tile cancellation.
// Node 22+: node web/tests/overlap-browser.mjs
import assert from "node:assert/strict";
import { spawn } from "node:child_process";
import { createServer } from "node:http";
import { createReadStream, existsSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { extname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
const root = process.env.ZEBSDSOLVE_REPO || resolve(fileURLToPath(new URL("../..", import.meta.url)));
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
const profile = mkdtempSync(join(tmpdir(), "overlap-chrome-"));
let chrome, ws;
const wallTimeout = setTimeout(() => {
  console.error('Overlap browser test exceeded 300 seconds');
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
    if(message.method === "Runtime.exceptionThrown")console.error("PAGE",JSON.stringify(message.params.exceptionDetails));
    if(message.method === "Runtime.consoleAPICalled")console.log("PAGE",message.params.args.map(a=>a.value??a.description).join(" "));
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
  await send("Runtime.enable",{},sessionId);
  const loaded = new Promise(resolve => { pageLoaded = resolve; });
  await send("Page.navigate", { url: `${base}/web/app/` }, sessionId);
  await Promise.race([loaded,new Promise((_,reject)=>setTimeout(()=>reject(new Error("Page load timed out")),30000).unref())]);

  console.log("Testing overlap workflow");
  const report = await evaluate(`
    window.pause=ms=>new Promise(r=>setTimeout(r,ms));
    window.require=(ok,message)=>{if(!ok)throw Error(message);};
    window.waitFor=async(predicate,label)=>{for(let i=0;i<2400;i++){if(predicate())return;if(document.getElementById('app').dataset.busy==='0' && document.getElementById('status').classList.contains('err'))throw Error(label+': '+document.getElementById('status').textContent);await pause(25);}throw Error(label+': '+document.getElementById('status').textContent);};
    await waitFor(()=>window.ze,'engine');console.log('engine ready');
    const code=ze._frame(f=>{const name=f.alloc(64);for(let i=0;i<40;i++){const n=ze.x.ze_shader(i,name,64,0,0);if(!n)break;const bytes=new Uint8Array(ze.mem,name,64);const text=new TextDecoder().decode(bytes.subarray(0,bytes.indexOf(0)));if(text==='overlap'){const p=f.alloc(n);ze.x.ze_shader(i,name,64,p,n);return new TextDecoder().decode(new Uint8Array(ze.mem,p,n));}}throw Error('No overlap shader');});
    const compilation=await ze.state.device.createShaderModule({code}).getCompilationInfo();
    require(!compilation.messages.some(m=>m.type==='error'),'Overlap WGSL: '+compilation.messages.map(m=>m.message+' line '+m.lineNum).join('; '));
    console.log('overlap shader compiles');

    document.getElementById('pat-live').checked=false;
    document.getElementById('btn-example').click();
    await waitFor(()=>document.getElementById('app').dataset.busy==='0' && ze.scan.rows===55,'example');console.log('example ready');
    await waitFor(()=>document.getElementById('log').textContent.includes('overview: 4,125 patterns'),'scan overview');
    window.configureUI=async(settings)=>{
      ze.configure(settings);
      const label=document.querySelector('.field[data-key="bandwidth"]');
      const input=label.querySelector('input');input.value=String(ze.settings.bandwidth);input.dispatchEvent(new Event('change',{bubbles:true}));
      await ze._queue;await pause(10);
    };
    await configureUI({method:'spherical',bandwidth:32,refine_iters:4,binning:1,full_resolution:'none',roi_x:0,roi_y:0,roi_w:4,roi_h:2});
    document.getElementById('btn-index').click();
    await waitFor(()=>document.getElementById('app').dataset.busy==='0' && ze.results().phase.filter(x=>x>=0).length===8,'primary');console.log('primary ready');
    window.primary=ze.results();
    await configureUI({roi_x:0,roi_y:0,roi_w:3,roi_h:2});
    document.getElementById('tab-overlap').click();
    const screen=document.getElementById('overlap-screen'),cutoff=document.getElementById('overlap-screen-value');
    const blur=document.getElementById('overlap-blur-gain');
    const link=document.getElementById('overlap-link-bandwidth'),bw=document.getElementById('overlap-bandwidth');
    require(blur.checked && screen.checked && cutoff.valueAsNumber===0.1 && !cutoff.disabled,'new overlap defaults');
    const polish=document.getElementById('overlap-polish'),refit=document.getElementById('overlap-refit-after-polish');
    require(!refit.checked && refit.disabled,'endpoint refit must default off and require polishing');
    polish.checked=true;polish.dispatchEvent(new Event('change'));
    require(!refit.disabled,'endpoint refit must be available with blur/gain and polishing enabled');
    blur.checked=false;blur.dispatchEvent(new Event('change'));
    require(refit.disabled,'endpoint refit must require blur/gain fitting');
    polish.checked=false;polish.dispatchEvent(new Event('change'));
    blur.checked=true;blur.dispatchEvent(new Event('change'));
    require(link.checked && bw.disabled && bw.valueAsNumber===ze.settings.bandwidth,'linked residual bandwidth default');
    // Keep the existing unscreened/no-gain fixture for map and cancellation checks.
    blur.checked=false;blur.dispatchEvent(new Event('change'));
    screen.checked=false;screen.dispatchEvent(new Event('change'));
    require(document.getElementById('overlap-map-component').value==='primary' && document.getElementById('overlap-map-component-2').value==='residual','default components');
    require(JSON.stringify(Array.from(ze.storedPrimaryValidity(0,5)))==='[1,1,1,1,0]','ROI-only change preserved native seed provenance');
    let invalidSeedRange;
    try { ze.storedPrimaryValidity(ze.scan.rows*ze.scan.cols,1); } catch (error) { invalidSeedRange=error; }
    require(/BadCell/.test(invalidSeedRange?.message),'invalid native seed range must fail synchronously');
    require(document.getElementById('overlap-map').value==='ipf_z' && document.getElementById('overlap-map-2').value==='ipf_z','default IPF Z maps');
    window.primaryBeforeOverlap=document.getElementById('map').toDataURL();
    require(primaryBeforeOverlap!==document.getElementById('overlap-map-canvas').toDataURL(),'primary must be visible before overlap');
    const primaryPixels=document.getElementById('map').getContext('2d').getImageData(3,0,1,1).data;
    require(primaryPixels[0]!==primaryPixels[1] || primaryPixels[1]!==primaryPixels[2],'primary outside overlap ROI must be native colored');
    const limit=document.getElementById('overlap-limit');limit.value='4';limit.dispatchEvent(new Event('change'));
    for(const id of ['overlap-map-component','overlap-map-component-2']) {
      const selector=document.getElementById(id);
      require(selector.options[1].textContent==='First accepted residual component','automatic residual label');
      require(selector.options[4].textContent==='Fit component 3' && selector.options[5].textContent==='Fit component 4','numbered component labels');
      require(selector.options[4].title.includes('final fit'),'component option tooltip');
      selector.value='3';selector.dispatchEvent(new Event('change'));
      require(selector.title===selector.options[4].title,'selected component tooltip');
      selector.value=id.endsWith('-2')?'residual':'primary';selector.dispatchEvent(new Event('change'));
    }
    document.getElementById('btn-overlap-run').click();
    await waitFor(()=>document.getElementById('app').dataset.busy==='0' && document.getElementById('overlap-readout').textContent.startsWith('Complete'),'overlap');console.log('overlap ready');
    require(document.getElementById('overlap-readout').textContent.includes('6 / 6'),'all ROI cells committed');
    const after=ze.results();
    for(const key of Object.keys(primary))require(primary[key].every((v,i)=>Object.is(v,after[key][i])),'primary changed '+key);
    window.exportOverlap=async()=>{
      let payload;const original=URL.createObjectURL,click=HTMLAnchorElement.prototype.click;
      URL.createObjectURL=blob=>{payload=blob;return 'blob:overlap-test';};HTMLAnchorElement.prototype.click=function(){};
      document.getElementById('btn-overlap-export').click();
      URL.createObjectURL=original;HTMLAnchorElement.prototype.click=click;
      require(payload,'export produced no data');return JSON.parse(await payload.text());
    };
    let saved=await exportOverlap();
    require(saved.schema==='zebsdsolve-overlap-candidates-v1' && saved.cells.length===6 && saved.max_components===4,'compact export shape');
    require(saved.arrays.quaternions.stride===16 && saved.arrays.phase.stride===4,'component axis');
    require(saved.assessment==='unassessed_candidates','candidate qualification');
    const component=document.getElementById('overlap-component');component.value='1';component.dispatchEvent(new Event('change'));
    const leftSource=document.getElementById('overlap-map-component');leftSource.value='2';leftSource.dispatchEvent(new Event('change'));
    const map=document.getElementById('overlap-map');map.value='fraction';map.dispatchEvent(new Event('change'));
    require(document.getElementById('map-legend').textContent.includes('100%'),'contribution legend');
    component.value='1';component.dispatchEvent(new Event('change'));
    map.value='ipf_z';leftSource.value='primary';leftSource.dispatchEvent(new Event('change'));
    require(document.getElementById('map').toDataURL()===primaryBeforeOverlap,'overlap replaced original primary map');
    return {primary:true,roi:6,export:true,components:4};
  `);
  // Select an actually computed cell through the shared map interaction.
  const point = await evaluate(`const b=document.getElementById('map').getBoundingClientRect();return {x:b.left+b.width/75*.5,y:b.top+b.height/55*.5};`);
  await send("Input.dispatchMouseEvent",{type:"mousePressed",x:point.x,y:point.y,button:"left",clickCount:1},sessionId);
  await send("Input.dispatchMouseEvent",{type:"mouseReleased",x:point.x,y:point.y,button:"left",clickCount:1},sessionId);
  await evaluate(`
    document.getElementById('overlap-component').value='1';document.getElementById('overlap-component').dispatchEvent(new Event('change'));
    await waitFor(()=>document.getElementById('overlap-target').width>1,'inspector');
    require(document.querySelectorAll('#overlap-components tr').length===4,'component table');
    const view=document.getElementById('overlap-inspection');view.value='component';view.dispatchEvent(new Event('change'));
    await waitFor(()=>document.getElementById('overlap-target-caption').textContent==='Conditional target','conditional inspector');
    const c=document.getElementById('overlap-component');c.value='1';c.dispatchEvent(new Event('change'));
    await waitFor(()=>document.getElementById('overlap-target').width>1,'primary fit slot inspector');
    c.value='2';c.dispatchEvent(new Event('change'));
    const map=document.getElementById('overlap-map');
    require(map.options.length===8,'map selector grew beyond focused set');
    map.value='proposal_ncc';map.dispatchEvent(new Event('change'));
    const mapComponent=document.getElementById('overlap-map-component');mapComponent.value='2';mapComponent.dispatchEvent(new Event('change'));
    require(map.selectedOptions[0].textContent==='Source NCC','numbered fit source NCC map label');
    require(document.getElementById('map-well').title.includes('target used to discover'),'numbered fit NCC hover meaning');
    mapComponent.value='residual';mapComponent.dispatchEvent(new Event('change'));
    require(map.selectedOptions[0].textContent==='Residual source NCC','residual NCC map label');
    require(document.getElementById('map-well').title.includes('before this component was added'),'residual NCC hover meaning');
    require([...document.querySelectorAll('#overlap-facts dt')].every(el=>el.title.length>30),'metric hover explanations');
    await waitFor(()=>document.getElementById('overlap-residual').title.includes('Final signed residual'),'pattern hover meaning');
    const filter=document.getElementById('overlap-filter-metric');
    require(filter.options.length===3 && ![...filter.options].some(o=>o.value==='mixture_ncc'),'mixture NCC must not be an acceptance filter');
    map.value='ipf_z';mapComponent.value='primary';mapComponent.dispatchEvent(new Event('change'));
    const primaryMap=document.getElementById('map').toDataURL(),beforeFilter=await exportOverlap();
    const threshold=document.getElementById('overlap-residual-cutoff');threshold.value='1';threshold.dispatchEvent(new Event('input'));
    require(document.getElementById('overlap-readout').textContent.includes('0 shown'),'residual cutoff did not hide low NCC');
    require(document.getElementById('map').toDataURL()===primaryMap,'residual cutoff changed original primary');
    const filtered=await exportOverlap();
    require(JSON.stringify(filtered.arrays)===JSON.stringify(beforeFilter.arrays) && filtered.cells.length===6,'display cutoff changed exported fits');
    require(filtered.view_filter.residual_ncc_minimum===1 && filtered.context_current,'residual cutoff invalidated frozen fit');
    threshold.value='0';threshold.dispatchEvent(new Event('input'));
    const residualMap=document.getElementById('overlap-map-canvas').toDataURL();
    const primaryCutoff=document.getElementById('overlap-primary-cutoff');primaryCutoff.value='1';primaryCutoff.dispatchEvent(new Event('input'));
    require((await exportOverlap()).context_current && document.getElementById('map').toDataURL()!==primaryMap,'primary cutoff must change display without invalidating fit');
    require(document.getElementById('overlap-map-canvas').toDataURL()===residualMap,'primary cutoff changed historical residual display');
    primaryCutoff.value='0';primaryCutoff.dispatchEvent(new Event('input'));threshold.value='0';threshold.dispatchEvent(new Event('input'));
    c.value='1';c.dispatchEvent(new Event('change'));
    require(document.getElementById('map').toDataURL()===primaryMap,'primary cutoff restoration');
    await pause(100);
  `);
  const layout=await evaluate(`
    const $=id=>document.getElementById(id);
    window.mapGeometry=()=>['map','overlap-map-canvas'].map((id,i)=>{
      const c=$(id).getBoundingClientRect(),w=$(i?'overlap-map-well':'map-well').getBoundingClientRect();
      return {width:c.width,height:c.height,left:c.left-w.left,top:c.top-w.top};
    });
    window.assertLinked=label=>{const [a,b]=mapGeometry();for(const key of ['width','height','left','top'])require(Math.abs(a[key]-b[key])<1.1,label+' '+key);};
    assertLinked('initial linked maps');
    const [target,simulation,residual,gain,facts]=[...$('overlap-pats').children].map(el=>el.getBoundingClientRect());
    require(target.top===simulation.top && residual.top===gain.top && target.left===residual.left && simulation.left===gain.left && simulation.left>target.left && gain.top>target.top,'inspector images must form a 2 × 2 grid');
    require(facts.top>=gain.bottom && facts.left===target.left && facts.right===gain.right,'fit facts must span both columns below the images');
    require($('pat-pane').getBoundingClientRect().width>380,'the 2 × 2 inspector needs a wider default pane');
    const a=$('overlap-map'),b=$('overlap-map-2'),component=$('overlap-map-component-2');
    require(a.options.length===8 && b.options.length===8,'map metrics must stay focused');
    b.value='mixture_ncc';b.dispatchEvent(new Event('change'));
    require(a.value==='ipf_z' && $('overlap-map-legend').textContent.includes('Mixture NCC'),'independent metric');
    a.value='ipf_y';a.dispatchEvent(new Event('change'));
    require(b.value==='mixture_ncc','IPF axis change must preserve a non-IPF metric');
    b.value='ipf_x';b.dispatchEvent(new Event('change'));
    require(a.value==='ipf_x','right IPF axis must update the left map');
    a.value='ipf_y';a.dispatchEvent(new Event('change'));
    require(b.value==='ipf_y' && $('map-legend').textContent.includes('IPF Y') && $('overlap-map-legend').textContent.includes('IPF Y'),'left IPF Y must update both maps and legends');
    b.value='mixture_ncc';b.dispatchEvent(new Event('change'));

    $('overlap-map-component').value='2';$('overlap-map-component').dispatchEvent(new Event('change'));
    a.value='mixture_ncc';a.dispatchEvent(new Event('change'));
    require(a.value===b.value,'same metric must be allowed');
    component.value='3';component.dispatchEvent(new Event('change'));
    require($('overlap-map-component').value==='2' && component.value==='3','independent component');
    component.value='2';component.dispatchEvent(new Event('change'));
    a.value='ipf_z';a.dispatchEvent(new Event('change'));b.value='ipf_z';b.dispatchEvent(new Event('change'));
    const well=$('overlap-map-well'),rect=well.getBoundingClientRect();
    window.beforeZoom=mapGeometry()[0];
    well.dispatchEvent(new WheelEvent('wheel',{deltaY:-500,clientX:rect.left+rect.width/2,clientY:rect.top+rect.height/2,bubbles:true,cancelable:true}));
    require(mapGeometry()[0].width>beforeZoom.width*1.5,'right map wheel did not zoom');assertLinked('right map zoom');
    const canvas=$('overlap-map-canvas'),x=rect.left+rect.width/2,y=rect.top+rect.height/2;
    const beforePan=mapGeometry()[0];
    canvas.dispatchEvent(new MouseEvent('mousedown',{clientX:x,clientY:y,button:0,bubbles:true}));
    window.dispatchEvent(new MouseEvent('mousemove',{clientX:x+25,clientY:y+20,button:0,bubbles:true}));
    window.dispatchEvent(new MouseEvent('mouseup',{clientX:x+25,clientY:y+20,button:0,bubbles:true}));
    require(Math.abs(mapGeometry()[0].left-beforePan.left)>10,'right map pan did not move');assertLinked('right map pan');
    $('btn-zoom').click();assertLinked('reset zoom');
    const c=canvas.getBoundingClientRect(),px=c.left+c.width/75*.5,py=c.top+c.height/55*.5;
    canvas.dispatchEvent(new MouseEvent('mousedown',{clientX:px,clientY:py,button:0,bubbles:true}));
    window.dispatchEvent(new MouseEvent('mouseup',{clientX:px,clientY:py,button:0,bubbles:true}));
    require($('overlap-component').value==='2','right map click did not inspect its component');
    await waitFor(()=>$('overlap-target').width>1,'right map inspection');
    require($('map-over').innerHTML===$('overlap-map-over').innerHTML,'selection and ROI overlays differ');
    const inspector=$('overlap-component');
    inspector.value='1';inspector.dispatchEvent(new Event('change'));
    well.focus();
    well.dispatchEvent(new KeyboardEvent('keydown',{key:'ArrowRight',bubbles:true,cancelable:true}));
    require($('pat-cell').textContent==='row 0, col 1' && inspector.value==='2','focused map arrows should inspect its numbered component');

    // Residual display filtering leaves the original primary source unchanged.
    $('overlap-map-component').value='primary';$('overlap-map-component').dispatchEvent(new Event('change'));
    a.value='ipf_z';a.dispatchEvent(new Event('change'));b.value='ipf_z';b.dispatchEvent(new Event('change'));
    const primaryMap=$('map').toDataURL(),cutoff=$('overlap-residual-cutoff');
    cutoff.value='1';cutoff.dispatchEvent(new Event('input'));
    require($('map').toDataURL()===primaryMap,'residual threshold changed primary map');
    cutoff.value='0';cutoff.dispatchEvent(new Event('input'));
    $('overlap-component').value='1';$('overlap-component').dispatchEvent(new Event('change'));
    await waitFor(()=>$('overlap-target').width>1,'restore joint inspector');
    const originalROI=[ze.settings.roi_x,ze.settings.roi_y,ze.settings.roi_w,ze.settings.roi_h];
    const m=canvas.getBoundingClientRect(),x0=m.left+m.width/75*.5,y0=m.top+m.height/55*.5,x1=m.left+m.width/75*2.5,y1=m.top+m.height/55*1.5;
    canvas.dispatchEvent(new MouseEvent('mousedown',{clientX:x0,clientY:y0,shiftKey:true,button:0,bubbles:true}));
    window.dispatchEvent(new MouseEvent('mousemove',{clientX:x1,clientY:y1,shiftKey:true,button:0,bubbles:true}));
    window.dispatchEvent(new MouseEvent('mouseup',{clientX:x1,clientY:y1,shiftKey:true,button:0,bubbles:true}));
    require(JSON.stringify([ze.settings.roi_x,ze.settings.roi_y,ze.settings.roi_w,ze.settings.roi_h])===JSON.stringify(originalROI),'right map ROI mismatch');
    require($('map-over').innerHTML===$('overlap-map-over').innerHTML,'ROI overlays differ');
    $('tab-index').click();await pause(50);
    require($('overlap-map-view').hidden && $('overlap-map-control').hidden && !$('map-kind').hidden,'ordinary map controls not restored');
    require(getComputedStyle(document.querySelector('.map-views')).display==='flex','ordinary map layout not restored');
    $('tab-overlap').click();await pause(50);assertLinked('return to overlap');
    return {defaults:true,independentSelectors:true,patternGrid:true,linkedZoom:true,linkedPan:true,selection:true,sharedROI:true,independentAcceptance:true,indexRestore:true};
  `);
  if(process.env.OVERLAP_SCREENSHOT){
    const shot=await send("Page.captureScreenshot",{format:"png"},sessionId);
    writeFileSync(process.env.OVERLAP_SCREENSHOT,Buffer.from(shot.data,"base64"));
  }
  await send("Emulation.setDeviceMetricsOverride",{width:700,height:560,deviceScaleFactor:1,mobile:false},sessionId);
  const narrow=await evaluate(`
    await pause(80);
    const stage=document.getElementById('stage'),pats=document.getElementById('overlap-pats');
    const views=[...document.querySelectorAll('.map-view')].map(el=>el.getBoundingClientRect());
    require(views[1].top>=views[0].bottom-1,'narrow maps did not stack');
    require(getComputedStyle(stage).overflowY==='auto' && stage.scrollHeight>stage.clientHeight,'short viewport needs a scrollable stage');
    stage.scrollTop=stage.scrollHeight;pats.scrollTop=pats.scrollHeight;await pause(30);
    const facts=pats.querySelector('.facts-cell').getBoundingClientRect(),box=stage.getBoundingClientRect();
    require(facts.bottom<=box.bottom+1 && facts.top<box.bottom,'fit information unreachable on short viewport');
    assertLinked('narrow maps');
    return {stackedMaps:true,shortViewportFactsReachable:true};
  `);
  if(process.env.OVERLAP_SCREENSHOT){
    const shot=await send("Page.captureScreenshot",{format:"png"},sessionId);
    writeFileSync(process.env.OVERLAP_SCREENSHOT.replace(/\.png$/,'-narrow.png'),Buffer.from(shot.data,"base64"));
  }
  await send("Emulation.clearDeviceMetricsOverride",{},sessionId);
  await evaluate('await pause(80);');
  const primaryFallback=await evaluate(`
    const $=id=>document.getElementById(id),comparison=$('pat-cmp'),previousComparison=comparison.value;
    comparison.value='split';comparison.dispatchEvent(new Event('change',{bubbles:true}));
    const inspector=$('overlap-component');inspector.value='primary';inspector.dispatchEvent(new Event('change'));
    await waitFor(()=>$('overlap-target').width>1 && $('overlap-simulation-caption').textContent==='Primary simulation','primary fallback after split selection');
    require($('overlap-residual').title.includes('Signed primary difference'),'primary difference semantics');
    require(![...$('overlap-facts').querySelectorAll('dt')].some(dt=>dt.textContent==='accepted'),'ordinary z acceptance leaked into NCC-gated overlap primary view');
    const signed=$('overlap-residual').toDataURL();
    require(signed!==$('pat-diff').toDataURL(),'primary fallback copied the split image');
    comparison.value='diff';comparison.dispatchEvent(new Event('change',{bubbles:true}));
    require(signed===$('pat-diff').toDataURL(),'primary fallback differs from existing native signed comparison');
    window.clickPrimaryCell=(row,col)=>{const c=$('map'),b=c.getBoundingClientRect(),x=b.left+b.width*(col+.5)/ze.scan.cols,y=b.top+b.height*(row+.5)/ze.scan.rows;c.dispatchEvent(new MouseEvent('mousedown',{clientX:x,clientY:y,button:0,bubbles:true}));window.dispatchEvent(new MouseEvent('mouseup',{clientX:x,clientY:y,button:0,bubbles:true}));};
    require(ze.results().phase[ze.scan.rows*ze.scan.cols-1]<0,'fallback fixture requires an unindexed final cell');
    clickPrimaryCell(ze.scan.rows-1,ze.scan.cols-1);
    await waitFor(()=>$('pat-cell').textContent==='row '+(ze.scan.rows-1)+', col '+(ze.scan.cols-1) && $('overlap-target').width>1 && $('overlap-point-status').textContent==='Primary unavailable','primary fallback without orientation');
    const image=$('overlap-residual'),pixels=image.getContext('2d').getImageData(0,0,image.width,image.height).data;
    require(pixels.every(v=>v===0),'missing primary orientation retained a previous signed difference');
    clickPrimaryCell(0,0);await waitFor(()=>$('pat-cell').textContent==='row 0, col 0' && $('overlap-target').width>1 && $('overlap-simulation-caption').textContent==='Primary simulation','restore indexed primary');
    comparison.value=previousComparison;comparison.dispatchEvent(new Event('change',{bubbles:true}));
    inspector.value='1';inspector.dispatchEvent(new Event('change'));
    return {splitViewIndependent:true,missingOrientationCleared:true};
  `);
  const lifecycle = await evaluate(`
    const pc=document.querySelector('input[data-geo="pc"][data-i="0"]');
    pc.value=String(Number(pc.value)+.01);pc.dispatchEvent(new Event('change',{bubbles:true}));
    await configureUI({roi_w:1,roi_h:1});
    require(ze.results().phase[1]>=0 && ze.storedPrimaryValidity(0,4).every(v=>v===0),'geometry edit must mark the old cohort unavailable');
    require(document.getElementById('overlap-readout').textContent.startsWith('Context changed'),'stale context not marked');
    require(!document.getElementById('btn-overlap-export').disabled,'stale results cannot export');
    let saved=await exportOverlap();require(saved.cells.length===6 && saved.context_current===false,'frozen stale export');
    const primary=document.getElementById('overlap-primary');primary.value='index';primary.dispatchEvent(new Event('change'));
    document.getElementById('btn-overlap-run').click();
    await waitFor(()=>document.getElementById('app').dataset.busy==='0' && document.getElementById('overlap-readout').textContent.startsWith('Complete · 1 / 1 cells'),'index first');
    require((await exportOverlap()).cells.length===1,'index-first replaced results incorrectly');
    require(ze.results().phase[1]>=0 && JSON.stringify(Array.from(ze.storedPrimaryValidity(0,4)))==='[1,0,0,0]','partial indexing must mark only the new geometry’s cohort');
    const oldPixel=document.getElementById('map').getContext('2d').getImageData(1,0,1,1).data;
    require(oldPixel[0]===oldPixel[1] && oldPixel[1]===oldPixel[2],'stale outside-ROI primary must remain grayscale');
    await configureUI({roi_w:4,roi_h:2});
    primary.value='stored';primary.dispatchEvent(new Event('change'));
    document.getElementById('btn-overlap-run').click();
    await waitFor(()=>document.getElementById('app').dataset.busy==='0' && document.getElementById('overlap-readout').textContent.startsWith('Complete · 1 / 1 cells'),'reuse after expanding ROI');
    require(JSON.stringify((await exportOverlap()).requested_cells)==='[0]','next run must exclude stale primary even inside expanded ROI');
    // Stop at the first complete tile, before another tile can begin.
    await configureUI({bandwidth:16,refine_iters:1,roi_x:0,roi_y:0,roi_w:16,roi_h:32,batch:2});
    document.getElementById('btn-overlap-primary').click();
    await waitFor(()=>document.getElementById('app').dataset.busy==='1','cancellation primary indexing started');
    await waitFor(()=>document.getElementById('app').dataset.busy==='0','cancellation primary indexing completed');
    const {roiCells}=await import('/web/app/js/overlap.js'),indexed=ze.results(),primaryCutoff=document.getElementById('overlap-primary-cutoff').valueAsNumber;
    const seeds=ze.storedPrimaryValidity();
    const eligible=Array.from(roiCells(ze.scan,ze.settings)).filter(cell=>seeds[cell] && indexed.phase[cell]>=0 && Number.isFinite(indexed.ncc[cell]) && indexed.ncc[cell]>=primaryCutoff);
    const batchSize=Math.max(1,Math.min(64,ze.settings.batch));
    console.log('cancellation fixture',JSON.stringify({eligible:eligible.length,batch:batchSize,cutoff:primaryCutoff,roi:[ze.settings.roi_x,ze.settings.roi_y,ze.settings.roi_w,ze.settings.roi_h]}));
    require(eligible.length>batchSize,'cancellation fixture needs more accepted primary cells than one batch: '+eligible.length+' eligible, batch '+batchSize);
    primary.value='stored';primary.dispatchEvent(new Event('change'));
    const first=eligible[0];
    clickPrimaryCell(Math.floor(first/ze.scan.cols),first%ze.scan.cols);
    let liveTable=false,liveElapsed=false;
    const observer=new MutationObserver(()=>{
      if(!document.getElementById('overlap-readout').textContent.includes(batchSize.toLocaleString()+' / '+eligible.length.toLocaleString()+' cells'))return;
      observer.disconnect();
      liveTable=document.getElementById('app').dataset.busy==='1' && !!document.querySelector('#overlap-components button:not([disabled])');
      liveElapsed=!document.getElementById('overlap-elapsed').hidden;
      document.getElementById('btn-stop').click();
    });
    observer.observe(document.getElementById('overlap-readout'),{childList:true,subtree:true,characterData:true});
    document.getElementById('btn-overlap-run').click();
    await waitFor(()=>document.getElementById('app').dataset.busy==='0' && document.getElementById('overlap-readout').textContent.startsWith('Stopped'),'stop after first tile ('+eligible.length+' eligible, batch '+batchSize+')');
    require(liveTable && liveElapsed,'live component table or elapsed time missing');
    saved=await exportOverlap();
    require(saved.state==='Stopped' && saved.cells.length===batchSize && saved.requested_cells.length===eligible.length,'incomplete tile committed or completed tile lost; eligible '+eligible.length+', batch '+batchSize);
    // Hold the serial queue so Stop arrives before the native operation starts.
    primary.value='stored';primary.dispatchEvent(new Event('change'));
    let release;const gate=new Promise(resolve=>{release=resolve;});
    const queued=ze._serial(()=>gate);
    document.getElementById('btn-overlap-run').click();
    await waitFor(()=>document.getElementById('app').dataset.busy==='1','queued overlap');
    document.getElementById('btn-stop').click();release();await queued;
    await waitFor(()=>document.getElementById('app').dataset.busy==='0','queued stop');
    const unchanged=await exportOverlap();
    require(JSON.stringify(unchanged)===JSON.stringify(saved),'queued stopped tile replaced prior result');
    document.getElementById('tab-index').click();
    require(!document.getElementById('pats').hidden && document.getElementById('overlap-pats').hidden,'primary view not restored');
    return {stale:true,indexFirst:true,stopRetained:saved.cells.length,cancellationEligible:eligible.length,queuedStop:true,primaryView:true};
  `);
  const nuisance=await evaluate(`
    document.getElementById('tab-overlap').click();
    const previousInspect=ze.overlapInspect.bind(ze);window.lastGuiReplay=null;
    ze.overlapInspect=async(...args)=>{const images=await previousInspect(...args);window.lastGuiReplay={fit_model:args[1].fit_model,blur_gain:args[1].blur_gain,sse:images.residual.reduce((a,v)=>a+v*v,0),images};return images;};
    await configureUI({roi_x:0,roi_y:0,roi_w:1,roi_h:1});
    const checkbox=document.getElementById('overlap-blur-gain');checkbox.checked=true;checkbox.dispatchEvent(new Event('change'));
    require(document.getElementById('overlap-readout').textContent.startsWith('Context changed'),'fit model toggle did not invalidate inspection');
    document.getElementById('overlap-limit').value='2';document.getElementById('overlap-limit').dispatchEvent(new Event('change'));
    const keep=document.getElementById('overlap-keep-n');require(keep.value==='1','unexpected candidate default');keep.value='5';keep.dispatchEvent(new Event('change'));
    document.getElementById('btn-overlap-run').click();
    await waitFor(()=>document.getElementById('app').dataset.busy==='0' && document.getElementById('overlap-readout').textContent.startsWith('Complete'),'blur gain fit');
    const fitted=await exportOverlap();
    require(fitted.metadata.residual_bandwidth===ze.settings.bandwidth,'linked bandwidth not exported');
    require(fitted.metadata.fit_model==='contrast_gain_v1' && fitted.metadata.blur_gain===true,'fitted model provenance');
    require(fitted.metadata.keep_n===5 && fitted.metadata.candidate_method==='spherical' && fitted.metadata.candidate_seed_policy,'candidate search provenance');
    require(fitted.arrays.winner_rank.stride===2 && fitted.arrays.winner_rank.values[0]===0,'primary candidate rank missing');
    const secondaryRank=fitted.arrays.winner_rank.values[1];require(fitted.arrays.phase.values[1]<0 ? secondaryRank===-1 : secondaryRank>=1 && secondaryRank<=5,'residual candidate rank outside pool');
    require(fitted.metadata.contrast_basis_preparation?.source_dc_restored===false,'contrast preparation provenance missing');
    require(fitted.metadata.component_norm_definition && fitted.arrays.contrast_basis_weights.stride===2,'model-weight provenance missing');
    require(fitted.arrays.blur_gain_theta.stride===7 && fitted.arrays.blur_gain_theta.values.length===7,'frozen fit parameters missing');
    require(fitted.metadata.blur_gain_optimizer.sigma_units==='effective_fit_pixels','fit parameter units missing');
    document.getElementById('overlap-component').value='1';document.getElementById('overlap-component').dispatchEvent(new Event('change'));
    const inspection=document.getElementById('overlap-inspection');
    inspection.value='component';inspection.dispatchEvent(new Event('change'));
    await waitFor(()=>!document.getElementById('overlap-gain').hidden && document.getElementById('overlap-simulation-caption').textContent==='Fitted contribution','fitted component and gain profile');
    require([...document.querySelectorAll('#overlap-facts dt')].some(el=>el.textContent==='Blur σ'),'fitted parameter details missing');
    require(!document.getElementById('overlap-gain-panel').hidden && document.getElementById('overlap-gain-empty').hidden,'gain panel must display directly');
    require(JSON.stringify(JSON.parse(document.getElementById('overlap-gain').dataset.theta))===JSON.stringify(fitted.arrays.blur_gain_theta.values),'gain panel must use frozen theta');
    require(document.getElementById('overlap-gain-range').textContent.includes('σ') && document.getElementById('overlap-gain-panel').title.includes('The displayed fit includes this gain and blur.'),'gain/blur display semantics');
    await waitFor(()=>window.lastGuiReplay?.blur_gain===true && window.lastGuiReplay.fit_model==='contrast_gain_v1','GUI contrast replay model');
    require(Math.abs(window.lastGuiReplay.sse-fitted.arrays.sse.values[0])<5e-5,'GUI model replay SSE disagrees');
    const {drawGray,quantiles}=await import('/web/app/js/draw.js');
    const {shape:[h,w],conditional,selected_contribution}=window.lastGuiReplay.images;
    const intensities=new Float32Array(2*h*w);
    intensities.set(conditional);intensities.set(selected_contribution,h*w);
    const [lo,hi]=quantiles(intensities,.005,.995);
    const contribution=document.createElement('canvas');
    drawGray(contribution,selected_contribution,w,h,{lo,hi});
    require(document.getElementById('overlap-simulation').toDataURL()===contribution.toDataURL(),'display must use the corrected native contribution');
    const image=await ze.overlapInspect(0,{fit_model:fitted.metadata.fit_model,max_components:2,maxComponents:2,cells:[0],blur_gain:true,blur_gain_theta:Float32Array.from(fitted.arrays.blur_gain_theta.values),quaternions:Float32Array.from(fitted.arrays.quaternions.values),phase:Int32Array.from(fitted.arrays.phase.values),amplitudes:Float32Array.from(fitted.arrays.amplitudes.values)},0,1);
    const ss=image.residual.reduce((a,v)=>a+v*v,0);require(Math.abs(ss-fitted.arrays.sse.values[0])<5e-5,'fitted browser inspector SSE disagrees');
    keep.value='1';keep.dispatchEvent(new Event('change'));
    const frozen=await exportOverlap();require(frozen.metadata.keep_n===5 && frozen.context_current,'future search setting rewrote or invalidated completed fit');
    return {frozenParameters:true,inspector:true,hover:true,modelReplay:true,weightProvenance:true,candidatePool:true,frozenCandidateMetadata:true};
  `);
  if(process.env.OVERLAP_GAIN_SCREENSHOT){
    const shot=await send("Page.captureScreenshot",{format:"png"},sessionId);
    writeFileSync(process.env.OVERLAP_GAIN_SCREENSHOT,Buffer.from(shot.data,"base64"));
  }
  await evaluate(`
    await waitFor(()=>{
      const box=document.getElementById('overlap-gain').getBoundingClientRect();
      return box.width>0 && box.height>0 && box.top>=0 && box.bottom<=innerHeight+1;
    },'gain profile visible in viewport');
  `);
  const histories=await evaluate(`
    const keep=document.getElementById('overlap-keep-n');keep.value='5';keep.dispatchEvent(new Event('change'));
    const limit=document.getElementById('overlap-limit');limit.value='4';limit.dispatchEvent(new Event('change'));
    document.getElementById('btn-overlap-run').click();
    await waitFor(()=>document.getElementById('app').dataset.busy==='0' && document.getElementById('overlap-readout').textContent.startsWith('Complete'),'protected four-component fit');
    const result=await exportOverlap();
    require(result.max_components===4 && result.metadata.keep_n===5,'protected fit settings');
    require(result.metadata.trajectory_selection==='best_valid_final_sse_of_pooled_and_rank1_v1','reference strategy not exported');
    require(result.arrays.trajectory_source.stride===1 && [-1,0,1].includes(result.arrays.trajectory_source.values[0]),'trajectory source not exported');
    require(result.arrays.winner_rank.stride===4,'four-component ranks missing');
    for(let j=0;j<4;j++){const rank=result.arrays.winner_rank.values[j],phase=result.arrays.phase.values[j];require(phase<0?rank===-1:j===0?rank===0:rank>=1&&rank<=5,'incoherent protected rank');}
    if(result.arrays.trajectory_source.values[0]===0)for(let j=1;j<4;j++)if(result.arrays.phase.values[j]>=0)require(result.arrays.winner_rank.values[j]===1,'reference fit has pooled rank');
    return {fourComponents:true,referenceProvenance:true,frozenRanks:true};
  `);
  const profiling=await evaluate(`
    const limit=document.getElementById('overlap-limit');limit.value='2';limit.dispatchEvent(new Event('change'));
    const keep=document.getElementById('overlap-keep-n');keep.value='1';keep.dispatchEvent(new Event('change'));
    const polish=document.getElementById('overlap-polish'),screen=document.getElementById('overlap-screen'),cutoff=document.getElementById('overlap-screen-value');
    require(!polish.checked && !screen.checked,'fixture must start with profiling disabled');
    const link=document.getElementById('overlap-link-bandwidth'),bw=document.getElementById('overlap-bandwidth');
    const primaryBandwidth=ze.settings.bandwidth;
    link.checked=false;link.dispatchEvent(new Event('change'));
    require(!bw.disabled,'manual bandwidth not editable');
    bw.value=String(primaryBandwidth+4);bw.dispatchEvent(new Event('change'));
    const validity=ze.storedPrimaryValidity();
    const cells=[];
    for(let i=0;i<validity.length && cells.length<2;i++)if(validity[i]===1)cells.push(i);
    require(cells.length===2 && ze.phases.length===1,'fixture requires two stored seeds in one phase');
    const screenOptions={primary:'stored',maxComponents:2,fitBlurGain:true,minResidualNcc:.1};
    const batched=await ze.overlapCells(cells,screenOptions);
    require(batched.profile_events.some(event=>event.order===2),'batched residual searches were not exercised');
    for(const [row,cell] of cells.entries()){
      const scalar=await ze.overlapCells([cell],screenOptions);
      require(batched.stop_reason[row]===scalar.stop_reason[0] && Math.abs(batched.sse[row]-scalar.sse[0])<1e-6,
        'WebGPU residual batch differs from the one-cell fit');
    }
    const primaryScores=ze.results().ncc;
    const strongest=Array.from(validity.keys()).filter(i=>validity[i]===1 && Number.isFinite(primaryScores[i]))
      .sort((a,b)=>primaryScores[b]-primaryScores[a]).slice(0,8);
    let screenCell,threshold;
    for(const cell of strongest){
      const control=await ze.overlapCells([cell],{primary:'stored',maxComponents:2,fitBlurGain:true,
        polish:true,minResidualNcc:1,residualBandwidth:primaryBandwidth+4});
      const scores=control.profile_events.filter(e=>e.order===2).map(e=>e.residual_ncc).filter(Number.isFinite);
      if(scores.length && primaryScores[cell]>Math.max(0,...scores)){
        screenCell=cell;threshold=(primaryScores[cell]+Math.max(0,...scores))/2;break;
      }
    }
    require(screenCell!==undefined,'fixture needs a primary that outscores its residual');
    const row=Math.floor(screenCell/ze.scan.cols),col=screenCell%ze.scan.cols;
    await configureUI({roi_x:col,roi_y:row,roi_w:1,roi_h:1});
    clickPrimaryCell(row,col);
    await waitFor(()=>document.getElementById('pat-cell').textContent==='row '+row+', col '+col,'screened test cell selected');
    screen.checked=true;screen.dispatchEvent(new Event('change'));cutoff.value=String(threshold);cutoff.dispatchEvent(new Event('change'));
    require(document.getElementById('overlap-primary-effective').textContent.includes(String(threshold)),'effective primary cutoff missing');
    polish.checked=true;polish.dispatchEvent(new Event('change'));
    document.getElementById('btn-overlap-run').click();
    await waitFor(()=>document.getElementById('app').dataset.busy==='0' && document.getElementById('overlap-readout').textContent.startsWith('Complete'),'V6 profile and screen');
    const fitted=await exportOverlap();
    require(fitted.elapsed_ms>=0 && !document.getElementById('overlap-elapsed').hidden,'completed elapsed time missing');
    const left=document.getElementById('overlap-map-component');left.value='primary';left.dispatchEvent(new Event('change'));
    require(document.getElementById('map').toDataURL()!==document.getElementById('overlap-map-canvas').toDataURL(),'screened residual must leave original primary visible');
    require(fitted.metadata.primary_ncc_minimum===Math.fround(threshold) && fitted.metadata.requested_primary_ncc_minimum===0,'primary execution cutoff missing');
    require(fitted.metadata.residual_bandwidth===primaryBandwidth+4 && fitted.metadata.settings.bandwidth===primaryBandwidth,
      'manual bandwidth must be distinct from the primary setting in the export');
    require(ze.settings.bandwidth===primaryBandwidth && ze.storedPrimaryValidity(fitted.cells[0],1)[0]===1,
      'residual search changed the indexed primary context');
    link.checked=true;link.dispatchEvent(new Event('change'));
    require(bw.disabled && bw.valueAsNumber===primaryBandwidth,'relink did not restore primary bandwidth');
    link.checked=false;link.dispatchEvent(new Event('change'));
    require(bw.valueAsNumber===primaryBandwidth+4,'custom bandwidth was not retained');
    require(fitted.metadata.profile_version===3 && fitted.metadata.orientation_polish===true,'profile provenance not exported');
    require(fitted.metadata.refit_blur_gain_after_polish===false,'disabled endpoint refit must be frozen');
    require(fitted.metadata.oxford_detector_attitude_version===0,'generic camera provenance not exported');
    require(fitted.metadata.residual_screen.enabled && Math.abs(fitted.metadata.min_residual_ncc-threshold)<1e-6,'computational screen not frozen');
    require(fitted.arrays.source_quaternions.stride===8,'native proposal poses missing');
    require(fitted.arrays.retained_order.values[0]===1,'screen did not preserve singleton');
    require(fitted.profile_events.some(e=>e.flags&2),'screened proposal diagnostic missing');
    require(fitted.profile_events.every(e=>e.cell===fitted.cells[0]),'tile events lost scan IDs');
    const fitSelector=document.getElementById('overlap-component');
    fitSelector.value='1';fitSelector.dispatchEvent(new Event('change'));
    await waitFor(()=>document.getElementById('overlap-target').width>1 && document.querySelector('#overlap-components button'),'screened component inspector');
    const buttons=document.querySelectorAll('#overlap-components button');
    require(!buttons[0].disabled && buttons[1].disabled && fitSelector.options[2].disabled,'screened component disabled in both inspector controls');
    buttons[1].click();
    require(fitSelector.value==='1' && document.querySelector('#overlap-components button')===buttons[0],'unavailable component click cleared the fitted table');
    fitSelector.value='2';fitSelector.dispatchEvent(new Event('change'));
    require(fitSelector.value==='1' && document.querySelectorAll('#overlap-components tr').length===2,'unavailable selection must keep the fitted table');

    const target=await ze.overlapInspect(fitted.cells[0],{fit_model:fitted.metadata.fit_model,max_components:2,cells:fitted.cells,blur_gain:true,blur_gain_theta:Float32Array.from(fitted.arrays.blur_gain_theta.values),quaternions:Float32Array.from(fitted.arrays.quaternions.values),phase:Int32Array.from(fitted.arrays.phase.values),amplitudes:Float32Array.from(fitted.arrays.amplitudes.values)},0,0);
    require(Math.abs(target.residual.reduce((sum,x)=>sum+x*x,0)-fitted.arrays.sse.values[0])<5e-5,'profile inspector replay mismatch');
    polish.checked=false;polish.dispatchEvent(new Event('change'));
    cutoff.value='-1';cutoff.dispatchEvent(new Event('change'));
    const stage2=document.getElementById('overlap-stage-2');
    stage2.value='1';stage2.dispatchEvent(new Event('change'));
    document.getElementById('btn-overlap-run').click();
    await waitFor(()=>document.getElementById('app').dataset.busy==='0' && document.getElementById('overlap-readout').textContent.startsWith('Complete'),'staged overlap');
    const staged=await exportOverlap();
    require(staged.metadata.stage_gate?.minimum[0]===1,'executed stage threshold not exported');
    require(staged.arrays.retained_order.values[0]===1,'stage gate accepted a joint fit below its threshold');
    require(staged.profile_events.some(e=>(e.flags&1024)&&!(e.flags&2048)),'missing rejected gate comparison');
    const inspector = document.getElementById('overlap-component');
    inspector.value='primary';inspector.dispatchEvent(new Event('change'));
    await waitFor(()=>document.getElementById('overlap-point-status').textContent.includes('K2 stopped'),'stage stop in primary inspection');
    stage2.value='';stage2.dispatchEvent(new Event('change'));
    document.getElementById('overlap-component').value='1';document.getElementById('overlap-component').dispatchEvent(new Event('change'));await pause(100);
    document.getElementById('btn-overlap-primary').click();
    await waitFor(()=>document.getElementById('app').dataset.busy==='0' && document.getElementById('overlap-readout').textContent.startsWith('Context changed'),'same-geometry primary rerun invalidates overlap');
    require(!(await exportOverlap()).context_current,'reindexed primary retained old overlap');
    const initialSettings=JSON.stringify(ze.settings);
    const runUnscreened=(residualBandwidth)=>ze.overlapCells(fitted.cells,{primary:'stored',residualBandwidth});
    const linkedFit=await runUnscreened(0), manualFit=await runUnscreened(8);
    require(linkedFit.residual_bandwidth===primaryBandwidth && manualFit.residual_bandwidth===8,
      'native unscreened bandwidth was not recorded');
    require(manualFit.proposal_ncc[0]===linkedFit.proposal_ncc[0],'manual bandwidth changed the primary proposal');
    require(!Object.is(manualFit.proposal_ncc[1],linkedFit.proposal_ncc[1]),
      'changing bandwidth did not change the unscreened residual proposal on the fixture');
    require(JSON.stringify(ze.settings)===initialSettings,'custom residual bandwidth leaked into project settings');
    return {primarySnapshotInvalidation:true,optIn:true,screenRollback:true,stageGate:true,events:true,proposalPoses:true,replay:true,cameraProvenance:true};
  `);
  if(process.env.OVERLAP_SCREENSHOT){
    const shot=await send("Page.captureScreenshot",{format:"png"},sessionId);
    writeFileSync(process.env.OVERLAP_SCREENSHOT,Buffer.from(shot.data,"base64"));
  }
  const geometryCompatibility=await evaluate(`
    const before=JSON.stringify(ze.geometry),input=document.getElementById('file-scan');
    const project={geometry_convention:'oxford-calibrated-v1',geometry:{...ze.geometry,rot:[1,0,0]}};
    const dt=new DataTransfer();dt.items.add(new File([JSON.stringify(project)],'old-oxford-project.json',{type:'application/json'}));
    input.files=dt.files;input.dispatchEvent(new Event('change',{bubbles:true}));
    await waitFor(()=>document.getElementById('status').textContent.includes('different camera convention'),'old Oxford project rejection');
    require(JSON.stringify(ze.geometry)===before,'Oxford project mutated generic detector geometry');
    return {oldOxfordOnGenericRejected:true,geometryUnchanged:true};
  `);
  console.log(JSON.stringify({overlap:report,layout,narrow,primaryFallback,lifecycle,nuisance,histories,profiling,geometryCompatibility}));
} finally {
  clearTimeout(wallTimeout);
  ws?.close();chrome?.kill();server.closeAllConnections();await new Promise(resolve=>server.close(resolve));
  try {rmSync(profile,{recursive:true,force:true,maxRetries:10,retryDelay:100});} catch(error){console.error("Profile cleanup:",error.message);}
}

