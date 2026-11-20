import { METRICS, MAPS, metricValue, passesFilter, mapDescription, weightMetricKey, acceptsPrimary, acceptsResidual, gainProfile } from "./overlap-metrics.js";
import { drawGray, drawSigned, quantiles, viridis, download } from "./draw.js";

const $ = id => document.getElementById(id);
const COLORS = ["#3b82f6", "#f59e0b", "#10b981", "#ef4444", "#8b5cf6", "#14b8a6"];
const STOP = ["Running", "Component limit", "Residual exhausted", "No proposal", "Unresolved addition", "Boundary fit", "Missing primary", "Flat pattern", "Attempt limit", "Invalid target", "Stale primary", "Residual screened", "Replacement limit", "Insufficient ΔSSE"];
const IGNORE = new Set(["batch", "tile_mb", "refine_pixels", "wg_fft", "wg_refine", "ipf_colors", "ipf_lightness", "ipf_gray", "roi_x", "roi_y", "roi_w", "roi_h"]);
const PRIMARY_MAP_DESCRIPTION = "Original indexed primary with a current stored seed and NCC at or above the primary minimum.";
const value = x => Number.isFinite(x) ? x.toFixed(4) : "—";

export function roiCells(scan, settings) {
  if (!scan.cols || !scan.rows) return new Uint32Array();
  const limited = settings.roi_w !== 0 && settings.roi_h !== 0;
  const x = limited ? Math.min(scan.cols - 1, settings.roi_x) : 0;
  const y = limited ? Math.min(scan.rows - 1, settings.roi_y) : 0;
  const w = limited ? Math.min(scan.cols - x, settings.roi_w) : scan.cols;
  const h = limited ? Math.min(scan.rows - y, settings.roi_h) : scan.rows;
  return Uint32Array.from({length: w * h}, (_, i) => (y + Math.floor(i / w)) * scan.cols + x + i % w);
}

/** Export indices refer to the combined cells array, preserving tile provenance. */
export function profileEventsForExport(events, cells) {
  const rows=new Map(cells.map((cell,row)=>[cell,row]));
  if(rows.size!==cells.length)throw Error("Duplicate exported overlap cells");
  return events.map(event=>{
    if(!rows.has(event.cell))throw Error("Overlap event refers to an uncommitted cell");
    return {...event,tile_cell_index:event.cell_index,cell_index:rows.get(event.cell)};
  });
}

/** Separate client-side result ownership. A completed tile is the commit unit. */
export function createOverlap({ st, busy, runPrimary, selectTab, redraw, status, log, renderIpfKey, inspectPrimary }) {
  let active = false, run = null, launching = false, inspectSerial = 0, lastRanges = [[0,1],[0,1]], primaryColors=null;
  const ordinaryMapTitle = $("map-well").title;
  const filterState = () => ({key:$("overlap-filter-metric").value,operator:"min",cutoff:$("overlap-filter-value").valueAsNumber});
  const primaryCutoff = () => $("overlap-primary-cutoff").valueAsNumber;
  const residualCutoff = () => $("overlap-residual-cutoff").valueAsNumber;
  const stageThresholds = () => [2,3,4].map(k => {
    const input = $(`overlap-stage-${k}`);
    return k > +$("overlap-limit").value || (input.value === "" && !input.validity?.badInput) ? null : input.valueAsNumber;
  });
  const primarySeed = cell => st.results?.phase[cell]>=0 ? st.ze.storedPrimaryValidity(cell,1)[0] : 0;
  const primaryVisible = (cell,seedValid=primarySeed(cell)) => acceptsPrimary(st.results,cell,primaryCutoff(),seedValid);
  const visible = (cell,component=selected(),cutoff=residualCutoff(),filter=filterState()) => acceptsResidual(run,cell,component,cutoff) && passesFilter(run,cell,component,filter);
  const identities = new WeakMap(); let nextIdentity = 1;
  const identity = object => {
    if (!object || typeof object !== "object") return 0;
    if (!identities.has(object)) identities.set(object, nextIdentity++);
    return identities.get(object);
  };
  const context = () => {if(!st.ze || !st.scan)return "";try{return JSON.stringify({
    scan: identity(st.scan),
    primary_snapshot: identity(st.results),
    files: st.sourceFiles.map(f => identity(f)),
    assets: st.phaseFiles.map(p => [identity(p), p.on]),
    phases: st.ze.phaseGeneration?.toString() ?? st.ze.phases,
    settings: Object.fromEntries(Object.entries(st.ze.settings).filter(([k]) => !IGNORE.has(k))),
    geometry: st.ze.geometry,
    geometry_frame: st.scan.geometryFrame,
    oxford_detector_attitude_version: st.ze.detectorAttitudeVersion ?? null,
    fit_blur_gain: $("overlap-blur-gain").checked,
    orientation_polish: $("overlap-polish").checked,
    min_residual_ncc: $("overlap-screen").checked ? $("overlap-screen-value").valueAsNumber : null,
    min_stage_improvement: stageThresholds(),
  });}catch{return "unavailable";}};
  const fresh = () => !!run && run.context === context();
  const selected = () => $("overlap-component").value === "primary" ? -1 : Math.max(0, +$("overlap-component").value - 1);
  const mapSelect = i => $(i ? "overlap-map-2" : "overlap-map");
  const componentSelect = i => $(i ? "overlap-map-component-2" : "overlap-map-component");
  const mapSource = i => componentSelect(i).value;
  const mapSlot = i => +mapSource(i)-1;
  const primaryMap = i => mapSource(i)==="primary";
  const mapComponent = (source,cell,cutoff=residualCutoff(),filter=filterState()) => {
    if(!run?.completed[cell])return -1;
    if(source!=="residual") {
      const component = +source-1;
      return visible(cell,component,cutoff,filter)?component:-1;
    }
    for(let component=0;component<run.k;component++)if(visible(cell,component,cutoff,filter))return component;
    return -1;
  };
  const describeMap = i => primaryMap(i) ? PRIMARY_MAP_DESCRIPTION : mapDescription(mode(i),mapSlot(i),run);
  const mode = (i=0) => mapSelect(i).value;
  const colorContext = () => JSON.stringify([st.settings.ipf_colors,st.settings.ipf_lightness,st.settings.ipf_gray]);
  const clearCanvas = id => { const c = $(id); c.width = c.height = 1; c.getContext("2d").clearRect(0, 0, 1, 1); };

  function sync() {
    const ready = !!st.scan && !st.gpuLost && !!st.phaseFiles.some(p=>p.on) && !st.batchRunning;
    $("btn-overlap-run").disabled = !ready || !!st.busy || launching || ($("overlap-primary").value === "stored" && !st.results);
    $("btn-overlap-primary").disabled = !ready || !!st.busy || launching;
    $("btn-overlap-settings").disabled = !!st.busy || launching;
    $("btn-overlap-export").disabled = !run?.completedCount;
    for (const id of ["overlap-primary", "overlap-limit", "overlap-keep-n", "overlap-blur-gain", "overlap-polish", "overlap-screen", "overlap-primary-cutoff"]) $(id).disabled = !!st.busy || launching;
    $("overlap-screen-value").disabled = !!st.busy || launching || !$("overlap-screen").checked;
    for (const k of [2,3,4]) {
      const excluded = k > +$("overlap-limit").value;
      $(`overlap-stage-${k}-row`).hidden = excluded;
      $(`overlap-stage-${k}`).disabled = excluded || !!st.busy || launching;
    }
    const weightKey=weightMetricKey(run), weight=METRICS[weightKey], filterSelect=$("overlap-filter-metric");
    const weightOption=filterSelect.querySelector('option[data-weight]');
    if(weightOption) {
      const wasSelected=filterSelect.value===weightOption.value;
      weightOption.value=weightKey; weightOption.textContent=weight.label; weightOption.title=weight.description;
      if(wasSelected){filterSelect.value=weightKey;$("overlap-filter-value").title=weight.description;}
    }
    $("overlap-weight-heading").textContent=weight.label.replace(" (%)","");
    $("overlap-weight-heading").title=weight.description;
    for(let i=0;i<2;i++) {const option=mapSelect(i).querySelector('option[value="fraction"]');if(option){option.textContent=weight.label.replace(" (%)","");option.title=weight.description;}}
    const k = run?.k || +$("overlap-limit").value;
    for (const choice of [$("overlap-component"),componentSelect(0),componentSelect(1)]) {
      const map = choice!==$("overlap-component");
      const old=choice.value;
      if (choice.options.length !== k+(map?2:1)) {
        const sources = [new Option("Primary","primary")];
        if(map)sources.push(new Option("Residual","residual"));
        choice.replaceChildren(...sources,...Array.from({length:k}, (_,i) => new Option(`${map?"Residual slot":"Fit slot"} ${i+1}`,String(i+1))));
        choice.value=sources.some(s=>s.value===old) || (+old>=1 && +old<=k)?old:choice===componentSelect(1)?"residual":"primary";
      }
    }
    $("overlap-readout").textContent = !run ? "" : `${fresh() ? run.state : "Context changed"} · ${run.completedCount.toLocaleString()} / ${run.requested.length.toLocaleString()} cells\n${run.fittedCount.toLocaleString()} candidate fits${run.unseededCount ? ` · ${run.unseededCount.toLocaleString()} missing primary` : ""}`;
    if(run && fresh() && !primaryMap(1)) {
      const source=mapSource(1), cutoff=residualCutoff(),filter=filterState();
      let shown=0;
      for(const cell of run.requested)if(mapComponent(source,cell,cutoff,filter)>=0)shown++;
      $("overlap-readout").textContent+=`\n${shown.toLocaleString()} shown`;
    }
    for(let i=0;i<2;i++) {
      const primary=primaryMap(i),desc=describeMap(i),select=mapSelect(i);
      for(const option of select.options)option.disabled=primary && !["ipf_x","ipf_y","ipf_z","phase","proposal_ncc"].includes(option.value);
      if(primary && select.selectedOptions[0]?.disabled)select.value="ipf_z";
      select.title=desc;$(i?"overlap-map-legend":"map-legend").title=desc;
      const searchOption=select.querySelector('option[value="proposal_ncc"]');
      if(searchOption)searchOption.textContent=primary?"Primary NCC":"Residual source NCC";
    }
    $("overlap-progress").hidden = !st.busy || !run || run.state !== "Running";
    $("overlap-progress").value = run ? run.completedCount / Math.max(1, run.requested.length) : 0;
  }

  function newRun(k, requested, keepN, minStageImprovement) {
    const settings = structuredClone(st.ze.settings);
    return { k, requested, context: context(), colorContext:colorContext(), state: "Running", completedCount: 0, fittedCount:0, unseededCount:0,
      completed: new Uint8Array(st.scan.rows * st.scan.cols), arrays: {}, rgba: {}, profileEvents: {},
      metadata: {schema:"zebsdsolve-overlap-candidates-v1", model:null, keep_n:keepN, blur_gain:$("overlap-blur-gain").checked, assessment:"unassessed_candidates",
        primary_ncc_minimum:primaryCutoff(), primary_snapshot:identity(st.results), orientation_polish:$("overlap-polish").checked, min_residual_ncc:$("overlap-screen").checked?$("overlap-screen-value").valueAsNumber:null,
        requested_min_stage_improvement:structuredClone(minStageImprovement), component_axis:"stable discovery order", stop_reasons:STOP, engine_version:st.ze.version, seed_source_roi:null, scan_name:st.scanName, quaternion_convention:"wxyz, sample to crystal", geometry:structuredClone(st.ze.geometry),
        settings, phases:structuredClone(st.ze.phases), scan:structuredClone(st.scan), primary:"stored",
        sources:st.sourceFiles.map(f => ({name:f.name,size:f.size,lastModified:f.lastModified})),
        masters:st.phaseFiles.filter(p => p.on).map(p => ({name:p.file?.name,size:p.file?.size,lastModified:p.file?.lastModified})),
      },
    };
  }

  function commit(target, result, cells) {
    const n = cells.length, total = target.completed.length;
    if(result.keep_n !== undefined && result.keep_n !== target.metadata.keep_n)throw Error("Overlap candidate count changed between tiles");
    if(result.blur_gain!==undefined && result.blur_gain!==target.metadata.blur_gain)throw Error("Overlap fit model changed between tiles");
    if(result.blur_gain_optimizer)target.metadata.blur_gain_optimizer=structuredClone(result.blur_gain_optimizer);
    if(result.blur_gain_parameter_names)target.metadata.blur_gain_parameter_names=Array.from(result.blur_gain_parameter_names);
    if(result.model)target.metadata.model=result.model;
    if(result.fit_model !== undefined)target.metadata.model=result.fit_model;
    for(const key of ["oxford_detector_attitude_version","fit_model","gain_domain","weight_basis","contrast_basis_preparation","component_norm_definition","target_preparation","contrast_nnls_solver","contrast_anchor","contrast_target_centering","keep_n","candidate_method","candidate_seed_policy","trajectory_selection","trajectory_version","trajectory_source_names","profile_optimizer","active_count_definition","fitted_count_definition","profile_version","selection_version","orientation_polish","polish_radius_deg","min_residual_ncc","requested_min_residual_ncc","profile_precision","residual_screen","stage_gate","component_axis"])if(result[key] !== undefined)target.metadata[key]=structuredClone(result[key]);
    // Prepare every copied field and color before changing the visible result map.
    const seedROI=result.seed_source_roi ? Array.from(result.seed_source_roi) : null;
    if(seedROI && target.metadata.seed_source_roi && JSON.stringify(seedROI)!==JSON.stringify(target.metadata.seed_source_roi))throw Error("Primary source context changed between tiles");
    const updates = [];
    for (const [key, array] of Object.entries(result)) {
      if (!ArrayBuffer.isView(array) || ["cells","seed_source_roi"].includes(key) || array.length % n) continue;
      const stride = array.length / n;
      if (!stride) continue;
      if (!target.arrays[key]) {
        const all = new array.constructor(total * stride);
        all.fill(key === "phase" || key === "discovery_id" || key === "winner_rank" || key === "trajectory_source" ? -1 : array instanceof Float32Array || array instanceof Float64Array ? NaN : 0);
        target.arrays[key] = {values:all,stride};
      }
      updates.push([target.arrays[key],array]);
    }
    const colors = {};
    for (const axis of ["x", "y", "z"]) colors[axis] = st.ze.orientationColors(result.source_quaternions || result.quaternions, result.phase, axis);
    if(seedROI)target.metadata.seed_source_roi=seedROI;
    for (const [field,array] of updates) cells.forEach((cell,i) => field.values.set(array.subarray(i*field.stride,(i+1)*field.stride),cell*field.stride));
    for (const axis of ["x", "y", "z"]) {
      target.rgba[axis] ||= new Uint8ClampedArray(total * target.k * 4);
      cells.forEach((cell,i) => target.rgba[axis].set(colors[axis].subarray(i*target.k*4,(i+1)*target.k*4),cell*target.k*4));
    }
    if(result.profile_events){
      for(const cell of cells)target.profileEvents[cell]=[];
      for(const event of result.profile_events)target.profileEvents[event.cell].push(structuredClone(event));
    }
    for (const cell of cells) {
      if (!target.completed[cell]) {
        target.completedCount++;
        if(target.arrays.retained_order?.values[cell]>0)target.fittedCount++;
        if([6,10].includes(target.arrays.stop_reason?.values[cell]))target.unseededCount++;
      }
      target.completed[cell] = 1;
    }
  }

  async function start() {
    if (st.busy || launching || !st.scan || st.gpuLost) return;
    const keepN=$("overlap-keep-n").valueAsNumber;
    const minResidualNcc=$("overlap-screen").checked?$("overlap-screen-value").valueAsNumber:null;
    const stages=stageThresholds();
    const minStageImprovement=stages.some(v=>v!==null)?stages:null;
    if(stages.some(v=>v!==null&&(!Number.isFinite(v)||v<0||v>1))){status("Minimum ΔSSE must be between 0 and 1",true);return;}
    if(minStageImprovement) {
      if(!$('overlap-screen').checked){status("Enable Screen residual NCC for ΔSSE thresholds",true);return;}
      if(minStageImprovement.slice(0,+$('overlap-limit').value-1).some(v=>v===null)){status("Fill every ΔSSE threshold through Maximum components",true);return;}
    }
    if(minResidualNcc!==null&&(!Number.isFinite(minResidualNcc)||minResidualNcc< -1||minResidualNcc>1)){status("Minimum residual NCC must be between −1 and 1",true);return;}

    if(!Number.isInteger(keepN) || keepN<1 || keepN>8){status("Search candidates must be an integer from 1 to 8",true);return;}
    launching = true; sync();
    try {
      if ($("overlap-primary").value === "index") {
        const indexed = await runPrimary();
        if (!indexed || st.stop || st.gpuLost) return;
      }
      const cutoff=primaryCutoff();
      if(!Number.isFinite(cutoff)||cutoff< -1||cutoff>1){status("Primary NCC minimum must be between −1 and 1",true);return;}
      const seeds = st.results ? st.ze.storedPrimaryValidity() : [];
      const requested = roiCells(st.scan, st.settings).filter(cell=>acceptsPrimary(st.results,cell,cutoff,seeds[cell])), k = +$("overlap-limit").value;
      if (!requested.length) { status("No accepted primary cells in overlap region", true); return; }
      const candidate = newRun(k, requested, keepN, minStageImprovement);
      await busy("overlap indexing…", async () => {
        const batch = Math.max(1, Math.min(64, st.settings.batch));
        try {
          for (let first = 0; first < requested.length && !st.stop; first += batch) {
            if (context() !== candidate.context) throw Error("Overlap context changed; completed tiles retained");
            const cells = requested.slice(first, first + batch);
            const result = await st.ze.overlapCells(cells, {maxComponents:k, primary:"stored",keepN,fitBlurGain:$("overlap-blur-gain").checked,polish:$("overlap-polish").checked,minResidualNcc,minStageImprovement});
            if (st.stop) { candidate.state = "Stopped"; break; }
            if (context() !== candidate.context) throw Error("Overlap context changed; tile not committed");
            commit(candidate, result, cells);
            run = candidate;
            sync(); redraw();
            await new Promise(resolve => setTimeout(resolve, 0));
          }
          candidate.state = st.stop ? "Stopped" : "Complete";
        } catch (error) {
          candidate.state = /Cancelled/.test(error.message) ? "Stopped" : "Interrupted";
          if (candidate.completedCount) run = candidate;
          throw error;
        } finally { sync(); redraw(); }
      });
    } finally { launching = false; sync(); }
  }

  function drawMap(canvas,i=0) {
    if (!st.scan) return;
    const {rows,cols} = st.scan, n = rows * cols;
    $(i?"overlap-map-well":"map-well").title=describeMap(i);
    drawGray(canvas, st.overview || new Float32Array(n), cols, rows);
    if(primaryMap(i)) { drawPrimary(canvas,i);return; }
    if (!run || run.completed.length !== n || !fresh()) return;
    if(run.colorContext!==colorContext() && !st.busy) {
      for(const axis of ["x","y","z"])run.rgba[axis]=st.ze.orientationColors((run.arrays.source_quaternions||run.arrays.quaternions).values,run.arrays.phase.values,axis);
      run.colorContext=colorContext();
    }
    const kind = mode(i), source=mapSource(i),cutoff=residualCutoff(),filter=filterState(),ctx = canvas.getContext("2d"), image = ctx.createImageData(cols, rows);
    const field = kind === "fraction" ? weightMetricKey(run) : kind === "count" ? "active_count" : kind === "proposals" ? "retained_order" : kind === "improvement" ? "incremental_improvement" : kind === "single_improvement" ? "improvement_over_best_single" : kind === "condition" ? "relative_gram_condition" : kind === "residual" ? "sse" : kind;
    const data = run.arrays[field];
    let lo = kind === "proposal_ncc" ? -1 : 0, hi = kind === "fraction" ? 1 : kind === "count" || kind === "proposals" ? run.k : 1;
    if (["residual","improvement","single_improvement"].includes(kind) && data) [lo, hi] = quantiles(Float32Array.from(run.completed, (v,i) => v ? data.values[i*data.stride] : NaN), 0, 1);
    lastRanges[i]=[lo,hi];
    for (const cell of run.requested) {
      const component = mapComponent(source,cell,cutoff,filter);
      if (component<0) continue;
      const at = cell*4, slot = cell*run.k + component;
      let color;
      if (kind.startsWith("ipf_")) {
        color = run.rgba[kind.slice(-1)]?.subarray(slot*4,slot*4+4);
        if (!color?.[3]) continue;
      } else if (kind === "phase") {
        const phase = run.arrays.phase?.values[slot];
        if (!(phase >= 0)) continue;
        color = COLORS[phase % COLORS.length].slice(1).match(/../g).map(v => parseInt(v,16));
      } else if (kind === "status") {
        const stop = run.arrays.stop_reason?.values[cell];
        color = [4,6,9,10].includes(stop) || run.arrays.conditioned?.values[cell] ? [226,166,71] : stop === 0 ? [100,100,105] : [78,156,139];
      } else {
        const v = kind==="fraction" ? metricValue(run,cell,component,weightMetricKey(run))/100 : data?.values[cell*data.stride + (data.stride === run.k ? component : 0)];
        if (!Number.isFinite(v)) { color = [80,80,86]; } else color = viridis((v-lo)/Math.max(hi-lo,1e-20));
      }
      image.data.set([color[0],color[1],color[2],255],at);
    }
    const overlay = new OffscreenCanvas(cols,rows); overlay.getContext("2d").putImageData(image,0,0); ctx.drawImage(overlay,0,0);
  }

  function drawPrimary(canvas,i) {
    const results=st.results;if(!results)return;
    const seeds=st.ze.storedPrimaryValidity(),cutoff=primaryCutoff();
    const key=colorContext();
    if(!primaryColors || primaryColors.results!==results || primaryColors.context!==key) {
      const rgba={};for(const axis of ["x","y","z"])rgba[axis]=st.ze.orientationColors(results.quaternions,results.phase,axis);
      primaryColors={results,context:key,rgba};
    }
    const {cols,rows}=st.scan,ctx=canvas.getContext("2d"),image=ctx.createImageData(cols,rows),kind=mode(i);
    for(let cell=0;cell<rows*cols;cell++) {
      if(!acceptsPrimary(results,cell,cutoff,seeds[cell]))continue;
      let color;
      if(kind.startsWith("ipf_")) { color=primaryColors.rgba[kind.slice(-1)].subarray(cell*4,cell*4+4);if(!color[3])continue; }
      else if(kind==="phase")color=COLORS[results.phase[cell]%COLORS.length].slice(1).match(/../g).map(v=>parseInt(v,16));
      else color=viridis((results.ncc[cell]+1)/2);
      image.data.set([color[0],color[1],color[2],255],cell*4);
    }
    lastRanges[i]=[-1,1];
    const overlay=new OffscreenCanvas(cols,rows);overlay.getContext("2d").putImageData(image,0,0);ctx.drawImage(overlay,0,0);
  }

  function drawLegend(box,i=0) {
    box.replaceChildren(); const primary=primaryMap(i);box.hidden = primary?!st.results:!run;
    if (box.hidden) return;
    if(!primary && !fresh()){box.textContent="Context changed";return;}
    const title = document.createElement("span");
    title.textContent = (primary?"Primary · ":"Residual source · ")+mapSelect(i).selectedOptions[0].textContent + (mode(i) === "fraction" ? " · 0–100%" : "");
    box.append(title);
    if(["fraction","proposal_ncc","mixture_ncc","count","proposals","residual","improvement","single_improvement","condition"].includes(mode(i))) {
      const scale=document.createElement("span");scale.className="overlap-scale";box.append(scale);
      if(mode(i)!=="fraction"){const limits=document.createElement("span");limits.textContent=`${value(lastRanges[i][0])} – ${value(lastRanges[i][1])}`;box.append(limits);}
    }
    if(mode(i)==="status")for(const [color,label] of [["#4e9c8b","Stopped / limit"],["#e2a647","Unresolved / unavailable"],["#646469","Running"]]){const chip=document.createElement("span");chip.className="chip";const marker=document.createElement("i");marker.style.background=color;chip.append(marker,label);box.append(chip);}
    if(mode(i).startsWith("ipf_") && renderIpfKey) {
      const seen=new Set();
      (primary?st.ze.phases:run.metadata.phases).forEach((phase,i)=>{const key=`${phase.laue}/${phase.pointGroup}`;if(!seen.has(key)){seen.add(key);box.append(renderIpfKey(i,phase.laue,key));}});
    }
    if (mode(i) === "phase") (primary?st.ze.phases:run.metadata.phases).forEach((phase,i) => {
      const chip = document.createElement("span"); chip.className="chip";
      const square = document.createElement("i"); square.style.background=COLORS[i%COLORS.length];
      chip.append(square,phase.name); box.append(chip);
    });
  }

  function drawGain(cell,w,h,originalPrimary=false) {
    const gainValue=x=>Number.isFinite(x)?x.toPrecision(7):"—";
    $("overlap-gain-panel").hidden=true;
    if(!fresh() || !run.completed[cell] || !run.metadata.blur_gain || run.arrays.objective_valid?.values[cell]!==1)return;
    const theta=run.arrays.blur_gain_theta?.values.subarray(cell*7,cell*7+7),gain=gainProfile(theta,w,h);
    if(!gain)return;
    $("overlap-gain-panel").hidden=false;
    // A constant envelope uses a range anchored at zero to avoid a false black image.
    drawGray($("overlap-gain"),gain.values,w,h,{lo:gain.min===gain.max?0:gain.min,hi:gain.max});
    $("overlap-gain-range").textContent=`G ${gainValue(gain.min)} – ${gainValue(gain.max)} · σ ${value(gain.sigma)} px`;
    $("overlap-gain").dataset.theta=JSON.stringify(Array.from(theta));
    $("overlap-gain").dataset.range=JSON.stringify([gain.min,gain.max]);
    $("overlap-gain-panel").title=`Frozen fitted mixture gain G(x,y), black ${gainValue(gain.min===gain.max?0:gain.min)}, white ${gainValue(gain.max)}. Edge G = 1. Applied to each fitted blurred, zero-mean simulation; ${originalPrimary?"The original primary simulation above is unchanged.":"The fitted reconstruction above includes this gain."} Gaussian σ = ${value(gain.sigma)} effective fit pixels is applied separately to fitted simulations, never to this gain image.`;
  }

  function stageStop(cell) {
    if (!fresh() || run.arrays.stop_reason?.values[cell] !== 13) return null;
    const stopped = run.profileEvents[cell]?.findLast(e => (e.flags & 128) && (e.flags & 1024) && !(e.flags & 2048));
    if (!stopped) return STOP[13];
    const energy = run.arrays.initial_sse?.values[cell];
    const measured = (stopped.lower_sse - stopped.joint_sse) / energy;
    const required = run.metadata.stage_gate?.minimum?.[stopped.order - 2];
    return `K${stopped.order} stopped · ΔSSE ${Number.isFinite(measured)?measured.toPrecision(3):"—"} / ${Number.isFinite(required)?required.toPrecision(3):"—"} min`;
  }

  async function showCell(cell) {
    const serial = ++inspectSerial;++st.showRun;
    $("pat-cell").textContent = st.scan ? `row ${Math.floor(cell/st.scan.cols)}, col ${cell%st.scan.cols}` : "pattern";
    const table = $("overlap-components"); table.replaceChildren();
    for (const id of ["overlap-target", "overlap-simulation", "overlap-residual"]) clearCanvas(id);
    $("overlap-facts").replaceChildren();$("overlap-gain-panel").hidden=true;
    const fallback=async()=>{
      if(serial!==inspectSerial)return;
      const seed=primarySeed(cell);
      $("overlap-point-status").textContent=stageStop(cell) ?? (primaryVisible(cell,seed)?"Primary":seed?"Below primary minimum":"Primary unavailable");
      const rendered=await inspectPrimary?.(seed===1);
      if(rendered!==true || serial!==inspectSerial || st.cell!==cell)return;
      for(const [from,to] of [["pat-exp","overlap-target"],["pat-sim","overlap-simulation"]]) {
        const source=$(from),target=$(to);target.width=source.width;target.height=source.height;
        target.getContext("2d").drawImage(source,0,0);
        target.title=source.title || (from==="pat-sim"?"Simulation at the original indexed primary orientation. No mixture blur or gain is applied.":from==="pat-diff"?"Original primary pattern comparison from the index inspector.":"Measured pattern from the index inspector.");
        target.closest("figure").title=target.title;
      }
      const comparison=st.cmp;
      if(comparison?.exp && comparison.diff)drawSigned($("overlap-residual"),comparison.diff,comparison.w,comparison.h);
      else clearCanvas("overlap-residual");
      $("overlap-residual").title="Signed primary difference: the existing index inspector's measured and simulated patterns, each zero-mean and unit-norm under its scoring mask. Independent of split-view selection.";
      $("overlap-residual").closest("figure").title=$("overlap-residual").title;
      $("overlap-target-caption").textContent="Measured";$("overlap-simulation-caption").textContent="Primary simulation";
      $("overlap-residual").closest("figure").querySelector("figcaption").textContent="Primary difference";
      const facts=Array.from($("pat-facts").children),displayFacts=[];
      for(let i=0;i<facts.length;i+=2)if(facts[i].textContent!=="accepted") {
        displayFacts.push(facts[i].cloneNode(true));if(facts[i+1])displayFacts.push(facts[i+1].cloneNode(true));
      }
      $("overlap-facts").replaceChildren(...displayFacts);
      const [h,w]=st.ze.shape(st.settings.binning!==1 && st.settings.full_resolution==="all");
      drawGain(cell,w,h,true);
    };
    $("pats").hidden=true;$("overlap-pats").hidden=false;$("overlap-pattern-control").hidden=selected()<0;
    const slot=cell*(run?.k||1)+selected(), retainedFit=run?.completed[cell] && run.arrays.phase?.values[slot]>=0 && run.arrays.objective_valid?.values[cell]!==0;
    if(selected()<0 || (!visible(cell) && !retainedFit && primaryVisible(cell))) { if(!st.busy && !st.gpuLost)await fallback();return; }
    if (!run?.completed[cell] || !fresh() || st.busy || st.gpuLost) {
      $("overlap-point-status").textContent = !run?.completed[cell] ? "Uncomputed" : !fresh() ? "Context changed" : st.gpuLost ? "GPU unavailable" : "Running";
      if(primaryVisible(cell))await fallback();return;
    }
    const k = run.k, component = selected(), get = key => run.arrays[key]?.values;
    for (let j=0; j<k; j++) {
      const slot=cell*k+j, phase=get("phase")?.[slot], row=document.createElement("tr"), button=document.createElement("button");
      button.type="button"; button.textContent=String(j+1); button.setAttribute("aria-pressed",String(j===component));
      button.addEventListener("click",()=>{$("overlap-component").value=String(j+1);$("overlap-inspection").value="component";redraw();showCell(cell);});
      const rank=get("winner_rank")?.[slot];
      button.title=`Fit ${j+1}: ${rank===0?"original primary source":rank>0?`residual source, candidate rank ${rank}`:"source unavailable"}. Pattern inspection uses the final joint-fit orientation; residual maps use its discovery orientation.`;
      const first=document.createElement("td"); first.append(button); row.append(first);
      const amplitude=get("amplitudes")?.[slot], fraction=get(weightMetricKey(run))?.[slot];
      for (const text of [phase>=0 ? run.metadata.phases[phase]?.name || String(phase) : "—", phase>=0 && Number.isFinite(fraction) ? `${(100*fraction).toFixed(1)}%` : "—", phase >= 0 ? value(get("proposal_ncc")?.[slot]) : "—"]) { const td=document.createElement("td");td.textContent=text;td.title=row.children.length===2?METRICS[weightMetricKey(run)].description:row.children.length===3?METRICS.proposal_ncc.description:"Candidate phase assigned to this retained component.";row.append(td); }
      if (!(amplitude>1e-6)) row.classList.add("inactive"); table.append(row);
    }

    const stop=get("stop_reason")?.[cell];
    $("overlap-point-status").textContent=stageStop(cell) ?? ((STOP[stop] || "Candidate fit")+(visible(cell)?"":" · Outside residual display acceptance"));
    for (const [label,key] of [["Mixture NCC","mixture_ncc"],["Residual SSE","sse"],["Incremental ΔSSE","incremental_improvement"],["ΔSSE over single","improvement_over_best_single"],["Condition ratio","relative_gram_condition"],["Active fits","active_count"]]) {
      const dt=document.createElement("dt"),dd=document.createElement("dd");dt.textContent=label;dd.textContent=key==="active_count"?String(get(key)?.[cell]??"—"):value(get(key)?.[cell]);dt.title=dd.title=key==="active_count"?mapDescription("count",selected(),run):METRICS[key]?.description||MAPS.count.description;$("overlap-facts").append(dt,dd);
    }
    if(run.metadata.blur_gain && get("retained_order")?.[cell]>0 && get("objective_valid")?.[cell]){
      const theta=get("blur_gain_theta")?.subarray(cell*7,cell*7+7);
      for(const [label,v,description] of [["Blur σ",theta?.[0],"Shared fitted Gaussian blur standard deviation in effective fit pixels. The same blur is applied to all component projections before gain and normalization."],["Gain centre / edge",theta?Math.exp(theta[1]):NaN,(run.metadata.fit_model==="contrast_gain_v1" ? "Shared contrast-envelope gain at the elliptical centre divided by the edge. Applied to each blurred, zero-mean simulation; it is not a camera-gain calibration. Edge gain is 1 to remove the arbitrary overall scale." : "Fitted positive gain at the elliptical centre divided by gain at the outer edge. Shared by all components; edge gain is fixed to 1 to remove an unidentifiable overall scale.")]]){
        const dt=document.createElement("dt"),dd=document.createElement("dd");dt.textContent=label;dd.textContent=value(v);dt.title=dd.title=description;$("overlap-facts").append(dt,dd);
      }
    }
    if(!(get("retained_order")?.[cell]>0))return;
    try {
      const result = Object.fromEntries(Object.entries(run.arrays).map(([key,field])=>[key,field.values]));
      result.max_components=k;result.maxComponents=k;result.blur_gain=run.metadata.blur_gain;result.blur_gain_optimizer=run.metadata.blur_gain_optimizer;result.model=run.metadata.model;result.fit_model=run.metadata.fit_model;result.gain_domain=run.metadata.gain_domain;
      result.cells=Uint32Array.from({length:run.completed.length},(_,i)=>i);
      const images=await st.ze.overlapInspect(cell,result,cell,component);
      if (!active || serial!==inspectSerial || !fresh() || st.cell!==cell) return;
      const [h,w]=images.shape, individual=$("overlap-inspection").value==="component";
      const target=individual?images.conditional:images.measured, simulation=individual?images.selected_contribution:images.composite;
      const pair=new Float32Array(target.length+simulation.length);pair.set(target);pair.set(simulation,target.length);
      const [lo,hi]=quantiles(pair,.005,.995);
      drawGray($("overlap-target"),target,w,h,{lo,hi});drawGray($("overlap-simulation"),simulation,w,h,{lo,hi});
      drawSigned($("overlap-residual"),images.residual,w,h);
      $("overlap-target").title=$("overlap-target-caption").title=individual?"Conditional target: the original prepared pattern minus every other fitted component. This differs from the earlier residual used to discover the selected orientation.":"Original measured pattern after the configured preparation. Shared intensity units with the fitted reconstruction.";
      $("overlap-simulation").title=$("overlap-simulation-caption").title=individual?"Selected component projection multiplied by its fitted amplitude, including shared blur and gain when enabled.":"Sum of every retained fitted component contribution, including shared blur and gain when enabled.";
      $("overlap-residual").title="Final signed residual: original prepared pattern minus the complete fitted reconstruction. Orange and blue show opposite residual signs. Residual search NCC instead refers to the earlier residual used for discovery.";
      drawGain(cell,w,h);
      for(const id of ["overlap-target","overlap-simulation","overlap-residual"])$(id).closest("figure").title=$(id).title;
      $("overlap-target-caption").textContent=individual?"Conditional target":"Measured";
      $("overlap-simulation-caption").textContent=individual?"Fitted contribution":"Reconstruction";
      $("overlap-residual").closest("figure").querySelector("figcaption").textContent="Signed residual";
    } catch(error) { if(serial===inspectSerial){status(error.message,true);log(error.message,true);} }
  }

  function exportRun() {
    if (!run?.completedCount) return;
    const cells=Array.from(run.completed.keys()).filter(cell=>run.completed[cell]);
    const arrays=Object.fromEntries(Object.entries(run.arrays).map(([key,field])=>[key,{stride:field.stride,values:cells.flatMap(cell=>Array.from(field.values.subarray(cell*field.stride,(cell+1)*field.stride)))}]));
    const payload={schema:run.metadata.schema,model:run.metadata.model,assessment:run.metadata.assessment,metadata:run.metadata,state:run.state,context_current:fresh(),max_components:run.k,view_filter:{...filterState(),component:selected()<0?"primary":selected()+1,primary_ncc_minimum:primaryCutoff(),residual_ncc_minimum:residualCutoff(),scope:"display_only_all_computed_results_exported"},requested_cells:Array.from(run.requested),cells,arrays,profile_event_index_scope:"exported_cells_row; original request row in tile_cell_index",profile_events:profileEventsForExport(Object.values(run.profileEvents).flat(),cells)};
    download(run.metadata.scan_name.replace(/\.[^.]+$/,"")+"_overlap.json",JSON.stringify(payload),"application/json");
  }

  $("btn-overlap-run").addEventListener("click",()=>start().catch(error=>{status(error.message,true);log(error.message,true);}));
  $("btn-overlap-primary").addEventListener("click",async()=>{await runPrimary();sync();redraw();});
  $("btn-overlap-settings").addEventListener("click",()=>selectTab("index"));
  $("btn-overlap-export").addEventListener("click",exportRun);
  for(const key of ["prepared_fractions","improvement_over_best_single"]){const m=METRICS[key],option=new Option(m.label,key);option.title=m.description;if(key==="prepared_fractions")option.dataset.weight="";$("overlap-filter-metric").append(option);}
  for(let i=0;i<2;i++)for(const option of mapSelect(i).options)option.title=MAPS[option.value]?.description||"";
  $("overlap-filter-metric").title="Additional display minimum for residual sources.";
  $("overlap-filter-value").title="Inclusive minimum for the selected display metric. NCC and ΔSSE are dimensionless; model weight or contrast share is in percent.";
  const updateFilter = reset => {
    const m=METRICS[$("overlap-filter-metric").value],input=$("overlap-filter-value");input.disabled=!m;
    if(m){input.min=m.min??"";input.max=m.max??"";input.step=m.step;if(reset)input.value=m.default;input.title=m.description;}
    sync();redraw();if(active)showCell(st.cell);
  };
  for(const id of ["overlap-primary-cutoff","overlap-residual-cutoff"])$(id).addEventListener("input",()=>{sync();redraw();if(active)showCell(st.cell);});
  $("overlap-filter-metric").addEventListener("change",()=>updateFilter(true));
  $("overlap-filter-value").addEventListener("input",()=>updateFilter(false));
  for(const id of ["overlap-primary","overlap-limit","overlap-keep-n","overlap-blur-gain","overlap-polish","overlap-screen","overlap-screen-value","overlap-stage-2","overlap-stage-3","overlap-stage-4"])$(id).addEventListener("change",()=>{sync();redraw();if(active)showCell(st.cell);});
  for(const id of ["overlap-component","overlap-map","overlap-map-2","overlap-map-component","overlap-map-component-2"])$(id).addEventListener("change",()=>{sync();redraw();if(active)showCell(st.cell);});
  $("overlap-inspection").addEventListener("change",()=>showCell(st.cell));
  return {get active(){return active;},get result(){return run;},sync,drawMap,drawLegend,showCell,
    inspectMap(i,cell){
      const source=mapSource(i);
      const component=source==="residual" && fresh()?mapComponent(source,cell):-1;
      $("overlap-component").value=source==="residual"?(component>=0?String(component+1):"primary"):source;
      sync();redraw();
    },
    hover(cell,i=0){
      if(primaryMap(i)) {
        const seed=primarySeed(cell);
        return seed?`Primary · NCC ${value(st.results.ncc[cell])}${primaryVisible(cell,seed)?"":" · Below primary minimum"}`:"Primary unavailable";
      }
      if(!run?.completed[cell])return "Uncomputed";if(!fresh())return "Context changed";
      if(run.arrays.active_count?.values[cell]<2)return "No multi-component fit";
      const component=mapComponent(mapSource(i),cell);
      if(component<0)return "No accepted residual source";
      const f=metricValue(run,cell,component,weightMetricKey(run))/100,info=MAPS[mode(i)],field=run.arrays[info?.field||mode(i)];
      const reading=field?.values[cell*field.stride+(field.stride===run.k?component:0)];
      return `Residual source slot ${component+1}${Number.isFinite(f)?` · ${(100*f).toFixed(1)}%`:""}${mode(i)!=="fraction" && Number.isFinite(reading)?` · ${mapSelect(i).selectedOptions[0].textContent} ${value(reading)}`:""}`;
    },
    activate(on){active=on;++inspectSerial;++st.showRun;if(!on)$("map-well").title=ordinaryMapTitle;$("pats").hidden=on;$("overlap-pats").hidden=!on;$("overlap-pattern-control").hidden=!on;$("map-kind").hidden=on;$("overlap-map-control").hidden=!on;$("overlap-map-view").hidden=!on;sync();redraw();if(st.cell>=0){if(on)showCell(st.cell);}},
  };
}
