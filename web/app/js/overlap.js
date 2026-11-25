import { METRICS, MAPS, metricValue, passesFilter, mapDescription, weightMetricKey, acceptsPrimary, acceptsFitComponent, acceptsResidual, gainProfile } from "./overlap-metrics.js";
import { drawGray, drawSigned, quantiles, viridis, download } from "./draw.js";

const $ = id => document.getElementById(id);
const COLORS = ["#3b82f6", "#f59e0b", "#10b981", "#ef4444", "#8b5cf6", "#14b8a6"];
const STOP = ["Running", "Component limit", "Residual exhausted", "No proposal", "Unresolved addition", "Boundary fit", "Missing primary", "Flat pattern", "Attempt limit", "Invalid target", "Stale primary", "Residual screened", "Replacement limit", "Insufficient ΔSSE"];
const IGNORE = new Set(["batch", "tile_mb", "refine_pixels", "wg_fft", "wg_refine", "ipf_colors", "ipf_lightness", "ipf_gray", "roi_x", "roi_y", "roi_w", "roi_h"]);
const PRIMARY_MAP_DESCRIPTION = "Original indexed primary with a current stored seed and NCC at or above the primary minimum.";
const RESIDUAL_MAP_LABEL = "First accepted residual component";
const RESIDUAL_MAP_DESCRIPTION = "Shows the first component in fit order found by residual indexing that passes the display thresholds at each pixel.";
const FIT_COMPONENT_MAP_DESCRIPTION = "Shows this component in the final fit, including original primary orientations retained by the fit.";
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
  let ipfKind = "ipf_z", customBandwidth = null;
  let rightMapCount = null, pendingRun = null, renderedPreview = null;
  const ordinaryMapTitle = $("map-well").title;
  const filterState = () => ({key:$("overlap-filter-metric").value,operator:"min",cutoff:$("overlap-filter-value").valueAsNumber});
  const requestedPrimaryCutoff = () => $("overlap-primary-cutoff").valueAsNumber;
  const screenCutoff = () => $("overlap-screen").checked ? $("overlap-screen-value").valueAsNumber : null;
  const effectivePrimaryCutoff = (primary, residual) => Math.fround(Math.max(primary, residual ?? -1));
  const primaryCutoff = () => effectivePrimaryCutoff(requestedPrimaryCutoff(), screenCutoff());
  const residualCutoff = () => $("overlap-residual-cutoff").valueAsNumber;
  const bandwidthInput = $("overlap-bandwidth");
  const linkedBandwidth = () => $("overlap-link-bandwidth").checked;
  const residualBandwidth = () => st.settings?.method === "spherical" && !linkedBandwidth() ? bandwidthInput.valueAsNumber : 0;
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
  const canRefitAfterPolish = () => $("overlap-polish").checked && $("overlap-blur-gain").checked;
  const refitAfterPolish = () => canRefitAfterPolish() && $("overlap-refit-after-polish").checked;
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
    refit_blur_gain_after_polish: refitAfterPolish(),
    min_residual_ncc: $("overlap-screen").checked ? $("overlap-screen-value").valueAsNumber : null,
    min_stage_improvement: stageThresholds(),
    residual_bandwidth: residualBandwidth(),
  });}catch{return "unavailable";}};
  const fresh = () => !!run && run.context === context();
  const selected = () => $("overlap-component").value === "primary" ? -1 : Math.max(0, +$("overlap-component").value - 1);
  const mapSelect = i => $(i ? "overlap-map-2" : "overlap-map");
  const componentSelect = i => $(i ? "overlap-map-component-2" : "overlap-map-component");
  const mapSource = i => componentSelect(i).value;
  const primaryMap = i => mapSource(i)==="primary";
  const mapComponent = (source,cell,cutoff=residualCutoff(),filter=filterState()) => {
    if(!run?.completed[cell])return -1;
    if(source!=="residual") {
      const component = +source-1;
      return acceptsFitComponent(run,cell,component) && passesFilter(run,cell,component,filter)?component:-1;
    }
    for(let component=0;component<run.k;component++)if(visible(cell,component,cutoff,filter))return component;
    return -1;
  };
  const describeMap = i => {
    if(primaryMap(i))return PRIMARY_MAP_DESCRIPTION;
    if(mapSource(i)!=="residual" && mode(i)==="proposal_ncc")return METRICS.proposal_ncc.description;
    return mapDescription(mode(i),-1,run);
  };
  const mode = (i=0) => mapSelect(i).value;
  const colorContext = () => JSON.stringify([st.settings.ipf_colors,st.settings.ipf_lightness,st.settings.ipf_gray]);
  const clearCanvas = id => { const c = $(id); c.width = c.height = 1; c.getContext("2d").clearRect(0, 0, 1, 1); };

  function showElapsed() {
    const output = $("overlap-elapsed");
    const attempt = pendingRun ?? run;
    output.hidden = !attempt;
    if (!attempt) return;
    const seconds = Math.floor(((attempt.finishedAt ?? performance.now()) - attempt.startedAt) / 1000);
    const units = [Math.floor(seconds / 3600), Math.floor(seconds / 60) % 60, seconds % 60];
    output.textContent = "Elapsed " + units.map(unit => String(unit).padStart(2,"0")).join(":");
  }

  function sync() {
    showElapsed();
    const effective = $("overlap-primary-effective");
    const floor = screenCutoff();
    effective.textContent = Number.isFinite(floor) && floor > requestedPrimaryCutoff() ? `Effective ${floor}` : "";
    const ready = !!st.scan && !st.gpuLost && !!st.phaseFiles.some(p=>p.on) && !st.batchRunning;
    $("btn-overlap-run").disabled = !ready || !!st.busy || launching || ($("overlap-primary").value === "stored" && !st.results);
    $("btn-overlap-primary").disabled = !ready || !!st.busy || launching;
    $("btn-overlap-settings").disabled = !!st.busy || launching;
    $("btn-overlap-export").disabled = !run?.completedCount;
    for (const id of ["overlap-primary", "overlap-limit", "overlap-keep-n", "overlap-blur-gain", "overlap-polish", "overlap-screen", "overlap-primary-cutoff"]) $(id).disabled = !!st.busy || launching;
    $("overlap-refit-after-polish").disabled = !!st.busy || launching || !canRefitAfterPolish();
    const spherical = st.settings?.method === "spherical";
    $("overlap-spherical-bandwidth").hidden = !spherical;
    $("overlap-link-bandwidth").disabled = !spherical || !!st.busy || launching;
    bandwidthInput.disabled = $("overlap-link-bandwidth").disabled || linkedBandwidth();
    if (linkedBandwidth()) bandwidthInput.value = st.settings?.bandwidth ?? "";
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
        const sources = [new Option("Original primary","primary")];
        if(map)sources.push(new Option(RESIDUAL_MAP_LABEL,"residual"));
        choice.replaceChildren(...sources,...Array.from({length:k}, (_,i) => new Option(`Fit component ${i+1}`,String(i+1))));
        choice.value=sources.some(s=>s.value===old) || (+old>=1 && +old<=k)?old:choice===componentSelect(1)?"residual":"primary";
      }
      if(map) {
        for(const option of choice.options) {
          option.title=option.value==="primary"?PRIMARY_MAP_DESCRIPTION:option.value==="residual"?RESIDUAL_MAP_DESCRIPTION:FIT_COMPONENT_MAP_DESCRIPTION;
        }
        choice.title=choice.selectedOptions[0]?.title || "";
      }
    }
    const current = fresh();
    const progressRun = pendingRun ?? run;
    $("overlap-readout").textContent = !progressRun ? "" : `${pendingRun || current ? progressRun.state : "Context changed"} · ${progressRun.completedCount.toLocaleString()} / ${progressRun.requested.length.toLocaleString()} cells\n${progressRun.fittedCount.toLocaleString()} candidate fits${progressRun.unseededCount ? ` · ${progressRun.unseededCount.toLocaleString()} missing primary` : ""}`;
    if(current && (!pendingRun || pendingRun === run) && !primaryMap(1)) {
      const source=mapSource(1), cutoff=residualCutoff(),filter=filterState();
      if (rightMapCount?.run !== run || rightMapCount.source !== source || !Object.is(rightMapCount.cutoff, cutoff) ||
          rightMapCount.filter.key !== filter.key || rightMapCount.filter.operator !== filter.operator ||
          !Object.is(rightMapCount.filter.cutoff, filter.cutoff)) {
        rightMapCount = {run, source, cutoff, filter, counted:0, shown:0};
      }
      // Tiles commit in request order, so completedCount bounds the committed prefix.
      for (let i = rightMapCount.counted; i < run.completedCount; i++) {
        if (mapComponent(source,run.requested[i],cutoff,filter)>=0) rightMapCount.shown++;
      }
      rightMapCount.counted = run.completedCount;
      $("overlap-readout").textContent+=`\n${rightMapCount.shown.toLocaleString()} shown in right map`;
    } else {
      rightMapCount = null;
    }
    for(let i=0;i<2;i++) {
      const primary=primaryMap(i),select=mapSelect(i);
      for(const option of select.options)option.disabled=primary && !["ipf_x","ipf_y","ipf_z","phase","proposal_ncc"].includes(option.value);
      if(primary && select.selectedOptions[0]?.disabled)select.value=ipfKind;
      const desc=describeMap(i);
      select.title=desc;$(i?"overlap-map-legend":"map-legend").title=desc;
      const searchOption=select.querySelector('option[value="proposal_ncc"]');
      if(searchOption)searchOption.textContent=primary?"Primary NCC":mapSource(i)==="residual"?"Residual source NCC":"Source NCC";
    }
    $("overlap-progress").hidden = !st.busy || !progressRun || progressRun.state !== "Running";
    $("overlap-progress").value = progressRun ? progressRun.completedCount / Math.max(1, progressRun.requested.length) : 0;
  }

  function newRun(engine, requested, options, provenance) {
    const scan = provenance.scan ?? engine.scan;
    const settings = structuredClone(engine.settings);
    return { k:options.maxComponents, requested, state:"Running", startedAt:performance.now(), completedCount:0, fittedCount:0, unseededCount:0,
      completed:new Uint8Array(scan.rows * scan.cols), arrays:{}, rgba:{}, sourceRgba:{}, profileEvents:{},
      metadata:{schema:"zebsdsolve-overlap-candidates-v1", model:null, keep_n:options.keepN, blur_gain:options.fitBlurGain, assessment:"unassessed_candidates",
        primary_ncc_minimum:effectivePrimaryCutoff(options.primaryMinimum, options.minResidualNcc), requested_primary_ncc_minimum:options.primaryMinimum,
        primary_snapshot:provenance.snapshot ?? null, orientation_polish:options.polish, refit_blur_gain_after_polish:options.refitBlurGainAfterPolish,
        min_residual_ncc:options.minResidualNcc, requested_min_stage_improvement:structuredClone(options.minStageImprovement),
        component_axis:"stable discovery order", stop_reasons:STOP, engine_version:engine.version, seed_source_roi:null,
        scan_name:provenance.scanName, quaternion_convention:"wxyz, sample to crystal", geometry:structuredClone(engine.geometry),
        residual_bandwidth:settings.method==="spherical" ? options.residualBandwidth || settings.bandwidth : null,
        settings, phases:structuredClone(engine.phases), scan:structuredClone(scan), primary:"stored",
        sources:provenance.sources, masters:provenance.masters, opening_hints:structuredClone(provenance.openHints ?? null),
      },
    };
  }

  function commit(target, result, cells, colorEngine) {
    const n = cells.length, total = target.completed.length;
    if(result.keep_n !== undefined && result.keep_n !== target.metadata.keep_n)throw Error("Overlap candidate count changed between tiles");
    if(result.blur_gain!==undefined && result.blur_gain!==target.metadata.blur_gain)throw Error("Overlap fit model changed between tiles");
    if(result.residual_bandwidth!==undefined && result.residual_bandwidth!==target.metadata.residual_bandwidth)throw Error("Residual bandwidth changed between tiles");
    if(result.blur_gain_optimizer)target.metadata.blur_gain_optimizer=structuredClone(result.blur_gain_optimizer);
    if(result.blur_gain_parameter_names)target.metadata.blur_gain_parameter_names=Array.from(result.blur_gain_parameter_names);
    if(result.model)target.metadata.model=result.model;
    if(result.fit_model !== undefined)target.metadata.model=result.fit_model;
    for(const key of ["oxford_detector_attitude_version","fit_model","gain_domain","weight_basis","contrast_basis_preparation","component_norm_definition","target_preparation","contrast_nnls_solver","contrast_anchor","contrast_target_centering","keep_n","candidate_method","candidate_seed_policy","trajectory_selection","trajectory_version","trajectory_source_names","profile_optimizer","active_count_definition","fitted_count_definition","profile_version","selection_version","orientation_polish","refit_blur_gain_after_polish","polish_radius_deg","min_residual_ncc","requested_min_residual_ncc","profile_precision","residual_screen","stage_gate","component_axis"])if(result[key] !== undefined)target.metadata[key]=structuredClone(result[key]);
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
    const colors = {}, sourceColors = {};
    const separateSource = target.metadata.orientation_polish && result.source_quaternions;
    if (colorEngine) for (const axis of ["x", "y", "z"]) {
      colors[axis] = colorEngine.orientationColors(result.quaternions, result.phase, axis);
      if(separateSource)sourceColors[axis]=colorEngine.orientationColors(result.source_quaternions,result.phase,axis);
    }
    if(seedROI)target.metadata.seed_source_roi=seedROI;
    for (const [field,array] of updates) cells.forEach((cell,i) => field.values.set(array.subarray(i*field.stride,(i+1)*field.stride),cell*field.stride));
    if (colorEngine) for (const axis of ["x", "y", "z"]) {
      target.rgba[axis] ||= new Uint8ClampedArray(total * target.k * 4);
      cells.forEach((cell,i) => target.rgba[axis].set(colors[axis].subarray(i*target.k*4,(i+1)*target.k*4),cell*target.k*4));
      if(separateSource) {
        target.sourceRgba[axis] ||= new Uint8ClampedArray(total * target.k * 4);
        cells.forEach((cell,i) => target.sourceRgba[axis].set(sourceColors[axis].subarray(i*target.k*4,(i+1)*target.k*4),cell*target.k*4));
      }
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

  function readOptions(settings) {
    const keepN=$("overlap-keep-n").valueAsNumber;
    const minResidualNcc=$("overlap-screen").checked?$("overlap-screen-value").valueAsNumber:null;
    const bandwidth=settings.method === "spherical" && !linkedBandwidth() ? bandwidthInput.valueAsNumber : 0;
    if (!Number.isInteger(bandwidth) || (bandwidth!==0 && (bandwidth<8 || bandwidth>256)) ||
        (settings.method==="spherical" && !linkedBandwidth() && bandwidth===0)) {
      throw new Error("Residual bandwidth must be an integer from 8 to 256");
    }
    const stages=stageThresholds();
    const minStageImprovement=stages.some(v=>v!==null)?stages:null;
    if(stages.some(v=>v!==null&&(!Number.isFinite(v)||v<0||v>1))){throw new Error("Minimum ΔSSE must be between 0 and 1");}
    if(minStageImprovement) {
      if(!$('overlap-screen').checked){throw new Error("Enable Screen residual NCC for ΔSSE thresholds");}
      if(minStageImprovement.slice(0,+$('overlap-limit').value-1).some(v=>v===null)){throw new Error("Fill every ΔSSE threshold through Maximum components");}
    }
    if(minResidualNcc!==null&&(!Number.isFinite(minResidualNcc)||minResidualNcc< -1||minResidualNcc>1)){throw new Error("Minimum residual NCC must be between −1 and 1");}

    if(!Number.isInteger(keepN) || keepN<1 || keepN>8){throw new Error("Search candidates must be an integer from 1 to 8");}
    const primaryMinimum = requestedPrimaryCutoff();
    if (!Number.isFinite(primaryMinimum) || primaryMinimum < -1 || primaryMinimum > 1) throw new Error("Primary NCC minimum must be between −1 and 1");
    return {
      maxComponents: +$("overlap-limit").value, primary:"stored", keepN, fitBlurGain:$("overlap-blur-gain").checked,
      polish:$("overlap-polish").checked, refitBlurGainAfterPolish:refitAfterPolish(),
      minResidualNcc, minStageImprovement, residualBandwidth:bandwidth,
      primaryMinimum, residualMapMinimum:residualCutoff(),
    };
  }

  function restoreOptions(options) {
    $("overlap-limit").value = options.maxComponents;
    $("overlap-keep-n").value = options.keepN;
    $("overlap-blur-gain").checked = options.fitBlurGain;
    $("overlap-polish").checked = options.polish;
    $("overlap-refit-after-polish").checked = options.refitBlurGainAfterPolish;
    $("overlap-screen").checked = options.minResidualNcc !== null;
    if (options.minResidualNcc !== null) $("overlap-screen-value").value = options.minResidualNcc;
    $("overlap-primary-cutoff").value = options.primaryMinimum;
    $("overlap-residual-cutoff").value = options.residualMapMinimum;
    $("overlap-link-bandwidth").checked = options.residualBandwidth === 0;
    if (options.residualBandwidth) bandwidthInput.value = customBandwidth = options.residualBandwidth;
    for (let k=2; k<=4; k++) $(`overlap-stage-${k}`).value = options.minStageImprovement?.[k-2] ?? "";
    sync();
  }

  function eligibleCells(engine, results, options, scan=engine.scan) {
    const cutoff = effectivePrimaryCutoff(options.primaryMinimum, options.minResidualNcc);
    const seeds = engine.storedPrimaryValidity();
    return roiCells(scan, engine.settings).filter(cell=>acceptsPrimary(results,cell,cutoff,seeds[cell]));
  }

  async function computeTiles(engine, candidate, options, check, onTile, colors) {
    const batch = Math.max(1, Math.min(64, engine.settings.batch));
    const {primaryMinimum, residualMapMinimum, ...fitOptions} = options;
    for (let first = 0; first < candidate.requested.length; first += batch) {
      if (!check()) return false;
      const cells = candidate.requested.slice(first, first + batch);
      const result = await engine.overlapCells(cells, fitOptions);
      if (!check()) return false;
      commit(candidate, result, cells, colors ? engine : null);
      await onTile(cells);
      await new Promise(resolve => setTimeout(resolve, 0));
    }
    return true;
  }

  async function runBatchOverlap(engine, job, check, onTile) {
    const options = job.overlap;
    const requested = eligibleCells(engine, engine.results(), options);
    const candidate = newRun(engine, requested, options, {
      scanName:job.sourceFiles[0].name, sources:job.sources, masters:job.masters, openHints:job.openHints,
    });
    await onTile({done:0,total:requested.length});
    const finished = await computeTiles(engine, candidate, options, check,
      () => onTile({done:candidate.completedCount,total:requested.length}), false);
    if (!finished) return null;
    candidate.state = "Complete";
    candidate.finishedAt = performance.now();
    const payload = exportPayload(candidate,true,{component:"primary",
      primary_ncc_minimum:candidate.metadata.primary_ncc_minimum,residual_ncc_minimum:options.residualMapMinimum,
      scope:"display_only_all_computed_results_exported"});
    return new Blob([JSON.stringify(payload)], {type:"application/json"});
  }

  async function start() {
    if (st.busy || launching || !st.scan || st.gpuLost) return;
    const options = readOptions(st.settings);
    launching = true; sync();
    try {
      if ($("overlap-primary").value === "index") {
        const indexed = await runPrimary();
        if (!indexed || st.stop || st.gpuLost) return;
      }
      const requested = eligibleCells(st.ze, st.results, options, st.scan);
      if (!requested.length) { status("No accepted primary cells in overlap region", true); return; }
      const descriptor = f => ({name:f.name,size:f.size,lastModified:f.lastModified});
      const candidate = newRun(st.ze, requested, options, {
        scan:st.scan, scanName:st.scanName, openHints:st.sourceOpenHints, sources:st.sourceFiles.map(descriptor),
        masters:st.phaseFiles.filter(p => p.on).map(p => descriptor(p.file || {})),
        snapshot:identity(st.results),
      });
      candidate.context = context();
      candidate.colorContext = colorContext();
      await busy("overlap indexing…", async () => {
        pendingRun = candidate;
        sync();
        const timer = setInterval(showElapsed, 1000);
        try {
          const finished = await computeTiles(st.ze, candidate, options, () => {
            if (st.stop) return false;
            if (context() !== candidate.context) throw Error("Overlap context changed; completed tiles retained");
            return true;
          }, cells => {
            run = candidate;
            sync(); redraw();
            if (active && cells.includes(st.cell)) void showCell(st.cell);
          }, true);
          candidate.state = finished ? "Complete" : "Stopped";
        } catch (error) {
          candidate.state = /Cancelled/.test(error.message) ? "Stopped" : "Interrupted";
          if (candidate.completedCount) run = candidate;
          throw error;
        } finally {
          clearInterval(timer);
          candidate.finishedAt = performance.now();
          pendingRun = null;
          run ??= candidate;
          sync(); redraw();
        }
      });
    } finally {
      launching = false; sync();
      if (active && st.cell >= 0 && !st.busy) void showCell(st.cell);
    }
  }

  function drawMap(canvas,i=0) {
    if (!st.scan) return;
    const {rows,cols} = st.scan, n = rows * cols;
    $(i?"overlap-map-well":"map-well").title=describeMap(i);
    drawGray(canvas, st.overview || new Float32Array(n), cols, rows);
    if(primaryMap(i)) { drawPrimary(canvas,i);return; }
    if (!run || run.completed.length !== n || !fresh()) return;
    if(run.colorContext!==colorContext() && !st.busy) {
      for(const axis of ["x","y","z"]) {
        run.rgba[axis]=st.ze.orientationColors(run.arrays.quaternions.values,run.arrays.phase.values,axis);
        if(run.metadata.orientation_polish && run.arrays.source_quaternions)run.sourceRgba[axis]=st.ze.orientationColors(run.arrays.source_quaternions.values,run.arrays.phase.values,axis);
      }
      run.colorContext=colorContext();
    }
    const kind = mode(i), source=mapSource(i),cutoff=residualCutoff(),filter=filterState(),ctx = canvas.getContext("2d"), image = ctx.createImageData(cols, rows);
    const field = kind === "fraction" ? weightMetricKey(run) : kind === "count" ? "active_count" : kind === "proposals" ? "retained_order" : kind === "improvement" ? "incremental_improvement" : kind === "single_improvement" ? "improvement_over_best_single" : kind === "condition" ? "relative_gram_condition" : kind === "residual" ? "sse" : kind;
    const data = run.arrays[field];
    let lo = kind === "proposal_ncc" ? -1 : 0, hi = kind === "fraction" ? 1 : kind === "count" || kind === "proposals" ? run.k : 1;
    if (["residual","improvement","single_improvement"].includes(kind) && data) [lo, hi] = quantiles(Float32Array.from(run.completed, (v,i) => v ? data.values[i*data.stride] : NaN), 0, 1);
    lastRanges[i]=[lo,hi];
    for (let row = 0; row < run.completedCount; row++) {
      const cell = run.requested[row];
      const component = mapComponent(source,cell,cutoff,filter);
      if (component<0) continue;
      const at = cell*4, slot = cell*run.k + component;
      let color;
      if (kind.startsWith("ipf_")) {
        const axis=kind.slice(-1), rgba=source==="residual"?run.sourceRgba[axis]||run.rgba[axis]:run.rgba[axis];
        color = rgba?.subarray(slot*4,slot*4+4);
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
    title.textContent = (primary?"Original primary · ":mapSource(i)==="residual"?"Residual source · ":`Fit component ${mapSource(i)} · `)+mapSelect(i).selectedOptions[0].textContent + (mode(i) === "fraction" ? " · 0–100%" : "");
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
    if(!fresh() || !run.completed[cell] || run.arrays.objective_valid?.values[cell]!==1)return;
    if(!run.metadata.blur_gain) {
      $("overlap-gain-empty").textContent="Fit disabled";
      return;
    }
    const theta=run.arrays.blur_gain_theta?.values.subarray(cell*7,cell*7+7),gain=gainProfile(theta,w,h);
    if(!gain)return;
    const canvas=$("overlap-gain"), lo=gain.min===gain.max?0:gain.min;
    canvas.hidden=false;
    $("overlap-gain-empty").hidden=true;
    // A constant envelope uses a range anchored at zero to avoid a false black image.
    drawGray(canvas,gain.values,w,h,{lo,hi:gain.max});
    $("overlap-gain-range").textContent=` · ${gain.min.toFixed(2)}–${gain.max.toFixed(2)}× · σ ${gain.sigma.toFixed(2)} px`;
    canvas.dataset.theta=JSON.stringify(Array.from(theta));
    canvas.dataset.range=JSON.stringify([gain.min,gain.max]);
    const view=originalPrimary?"Select a fitted component to see it with this gain and blur.":"The displayed fit includes this gain and blur.";
    $("overlap-gain-panel").title=`Shared fitted gain. Black = ${lo.toPrecision(7)}×, white = ${gain.max.toPrecision(7)}×. Edge gain = 1. Gaussian blur σ = ${value(gain.sigma)} effective fit pixels is applied to each simulation before gain. ${view}`;
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
    $("overlap-facts").replaceChildren();
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
      const [h,w]=st.ze.shape();
      drawGain(cell,w,h,true);
      renderedPreview = key;
    };
    const inspector=$("overlap-component");
    const fitCurrent=run?.completed[cell] && fresh();
    const hasFit=j=>fitCurrent && acceptsFitComponent(run,cell,j);
    for (const option of inspector.options) option.disabled=option.value!=="primary" && !hasFit(+option.value-1);
    if (selected()>=0 && !hasFit(selected()) && !(st.busy && !fitCurrent)) {
      const residual=fitCurrent?mapComponent("residual",cell):-1;
      const retained=Array.from(inspector.options).find(option=>option.value!=="primary" && !option.disabled);
      inspector.value=hasFit(residual)?String(residual+1):retained?.value ?? "primary";
    }
    const component = selected();
    const key = JSON.stringify([context(), cell, component, $("overlap-inspection").value,
      identity(fitCurrent ? run : null)]);
    if (!st.busy || renderedPreview !== key) {
      renderedPreview = null;
      for (const id of ["overlap-target", "overlap-simulation", "overlap-residual", "overlap-gain"]) clearCanvas(id);
      $("overlap-gain").hidden = true;
      delete $("overlap-gain").dataset.theta;
      delete $("overlap-gain").dataset.range;
      $("overlap-gain-empty").hidden = false;
      $("overlap-gain-empty").textContent = "Unavailable";
      $("overlap-gain-panel").title = "";
      $("overlap-gain-range").textContent = "";
    }
    $("pats").hidden=true;$("overlap-pats").hidden=false;$("overlap-pattern-control").hidden=selected()<0;
    if(component<0 && !st.busy) { if(!st.gpuLost)await fallback();return; }
    if (!fitCurrent) {
      $("overlap-point-status").textContent = pendingRun ? "Awaiting fit" : run?.completed[cell] ? "Context changed" : "Uncomputed";
      if(!st.busy && !st.gpuLost && primaryVisible(cell))await fallback();return;
    }
    const k = run.k, get = key => run.arrays[key]?.values;
    for (let j=0; j<k; j++) {
      const slot=cell*k+j, phase=get("phase")?.[slot], row=document.createElement("tr"), button=document.createElement("button");
      button.type="button"; button.textContent=String(j+1); button.setAttribute("aria-pressed",String(j===component));
      button.disabled=!hasFit(j);
      button.addEventListener("click",()=>{if(button.disabled)return;inspector.value=String(j+1);$("overlap-inspection").value="component";redraw();showCell(cell);});
      const rank=get("winner_rank")?.[slot];
      button.title=button.disabled?`No fitted component ${j+1} at this point.`:`Fit ${j+1}: ${rank===0?"original primary source":rank>0?`residual source, candidate rank ${rank}`:"source unavailable"}. The fit map and inspector use the final fitted orientation. The residual-source map uses its discovery orientation.`;
      const first=document.createElement("td"); first.append(button); row.append(first);
      const fraction=get(weightMetricKey(run))?.[slot];
      for (const text of [phase>=0 ? run.metadata.phases[phase]?.name || String(phase) : "—", phase>=0 && Number.isFinite(fraction) ? `${(100*fraction).toFixed(1)}%` : "—", phase >= 0 ? value(get("proposal_ncc")?.[slot]) : "—"]) { const td=document.createElement("td");td.textContent=text;td.title=row.children.length===2?METRICS[weightMetricKey(run)].description:row.children.length===3?METRICS.proposal_ncc.description:"Candidate phase assigned to this retained component.";row.append(td); }
      if (button.disabled) row.classList.add("inactive"); table.append(row);
    }

    const stop=get("stop_reason")?.[cell];
    const residualInMixture=component>=0 && get("winner_rank")?.[cell*k+component]>0 && get("active_count")?.[cell]>=2;
    $("overlap-point-status").textContent=stageStop(cell) ?? ((STOP[stop] || "Candidate fit")+(residualInMixture && !visible(cell)?" · Hidden by residual map filters":""));
    for (const [label,key] of [["Mixture NCC","mixture_ncc"],["Residual SSE","sse"],["Incremental ΔSSE","incremental_improvement"],["ΔSSE over single","improvement_over_best_single"],["Condition ratio","relative_gram_condition"],["Active fits","active_count"]]) {
      const dt=document.createElement("dt"),dd=document.createElement("dd");dt.textContent=label;dd.textContent=key==="active_count"?String(get(key)?.[cell]??"—"):value(get(key)?.[cell]);dt.title=dd.title=key==="active_count"?mapDescription("count",selected(),run):METRICS[key]?.description||MAPS.count.description;$("overlap-facts").append(dt,dd);
    }
    if(run.metadata.blur_gain && get("retained_order")?.[cell]>0 && get("objective_valid")?.[cell]){
      const theta=get("blur_gain_theta")?.subarray(cell*7,cell*7+7);
      for(const [label,v,description] of [["Blur σ",theta?.[0],"Shared fitted Gaussian blur standard deviation in effective fit pixels. The same blur is applied to all component projections before gain and normalization."],["Gain centre / edge",theta?Math.exp(theta[1]):NaN,(run.metadata.fit_model==="contrast_gain_v1" ? "Shared contrast-envelope gain at the elliptical centre divided by the edge. Applied to each blurred, zero-mean simulation; it is not a camera-gain calibration. Edge gain is 1 to remove the arbitrary overall scale." : "Fitted positive gain at the elliptical centre divided by gain at the outer edge. Shared by all components; edge gain is fixed to 1 to remove an unidentifiable overall scale.")]]){
        const dt=document.createElement("dt"),dd=document.createElement("dd");dt.textContent=label;dd.textContent=value(v);dt.title=dd.title=description;$("overlap-facts").append(dt,dd);
      }
    }
    if(!(get("retained_order")?.[cell]>0) || st.busy || st.gpuLost)return;
    try {
      const {blur_gain, blur_gain_optimizer, model, fit_model, gain_domain} = run.metadata;
      const result = {cells:Uint32Array.of(cell), max_components:k, blur_gain, blur_gain_optimizer, model, fit_model, gain_domain};
      for (const key of ["quaternions","amplitudes","phase","proposal_ncc","proposal_nmi","blur_gain_theta"]) {
        const field = run.arrays[key];
        if (field) result[key] = field.values.subarray(cell*field.stride,(cell+1)*field.stride);
      }
      const images=await st.ze.overlapInspect(cell,result,0,component);
      if (!active || serial!==inspectSerial || !fresh() || st.cell!==cell) return;
      const [h,w]=images.shape, individual=$("overlap-inspection").value==="component";
      // Native inspection replays the saved nuisance parameters before applying
      // the fitted amplitudes. Display that contribution in the target's units.
      const target=individual?images.conditional:images.measured;
      const simulation=individual?images.selected_contribution:images.composite;
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
      renderedPreview = key;
    } catch(error) { if(serial===inspectSerial){status(error.message,true);log(error.message,true);} }
  }

  function exportPayload(target, contextCurrent, viewFilter) {
    const cells=Array.from(target.requested.subarray(0,target.completedCount));
    const arrays=Object.fromEntries(Object.entries(target.arrays).map(([key,field])=>[key,{stride:field.stride,values:cells.flatMap(cell=>Array.from(field.values.subarray(cell*field.stride,(cell+1)*field.stride)))}]));
    return {schema:target.metadata.schema,model:target.metadata.model,assessment:target.metadata.assessment,metadata:target.metadata,state:target.state,
      elapsed_ms:(target.finishedAt ?? performance.now())-target.startedAt,context_current:contextCurrent,max_components:target.k,view_filter:viewFilter,
      requested_cells:Array.from(target.requested),cells,arrays,profile_event_index_scope:"exported_cells_row; original request row in tile_cell_index",
      profile_events:profileEventsForExport(Object.values(target.profileEvents).flat(),cells)};
  }

  function exportRun() {
    if (!run?.completedCount) return;
    const payload=exportPayload(run,fresh(),{...filterState(),component:selected()<0?"primary":selected()+1,primary_ncc_minimum:primaryCutoff(),
      residual_ncc_minimum:residualCutoff(),scope:"display_only_all_computed_results_exported"});
    download(run.metadata.scan_name.replace(/\.[^.]+$/,"")+"_overlap.json",JSON.stringify(payload),"application/json");
  }

  $("btn-overlap-run").addEventListener("click",()=>start().catch(error=>{status(error.message,true);log(error.message,true);}));
  $("btn-overlap-primary").addEventListener("click",async()=>{await runPrimary();sync();redraw();});
  $("btn-overlap-settings").addEventListener("click",()=>selectTab("index"));
  $("btn-overlap-export").addEventListener("click",exportRun);
  for(const key of ["prepared_fractions","improvement_over_best_single"]){const m=METRICS[key],option=new Option(m.label,key);option.title=m.description;if(key==="prepared_fractions")option.dataset.weight="";$("overlap-filter-metric").append(option);}
  for(let i=0;i<2;i++)for(const option of mapSelect(i).options)option.title=MAPS[option.value]?.description||"";
  $("overlap-filter-metric").title="Filters the numbered fit maps and the residual map.";
  $("overlap-filter-value").title="Inclusive minimum for the selected display metric. NCC and ΔSSE are dimensionless; model weight or contrast share is in percent.";
  const updateFilter = reset => {
    const m=METRICS[$("overlap-filter-metric").value],input=$("overlap-filter-value");input.disabled=!m;
    if(m){input.min=m.min??"";input.max=m.max??"";input.step=m.step;if(reset)input.value=m.default;input.title=m.description;}
    sync();redraw();if(active)showCell(st.cell);
  };
  for(const id of ["overlap-primary-cutoff","overlap-residual-cutoff"])$(id).addEventListener("input",()=>{sync();redraw();if(active)showCell(st.cell);});
  $("overlap-filter-metric").addEventListener("change",()=>updateFilter(true));
  $("overlap-filter-value").addEventListener("input",()=>updateFilter(false));
  $("overlap-link-bandwidth").addEventListener("change",()=>{
    if (!linkedBandwidth() && customBandwidth !== null) bandwidthInput.value=customBandwidth;
    sync();redraw();if(active)showCell(st.cell);
  });
  bandwidthInput.addEventListener("change",()=>{
    customBandwidth=bandwidthInput.value;
    sync();redraw();if(active)showCell(st.cell);
  });
  for(const id of ["overlap-primary","overlap-limit","overlap-keep-n","overlap-blur-gain","overlap-polish","overlap-refit-after-polish","overlap-screen","overlap-screen-value","overlap-stage-2","overlap-stage-3","overlap-stage-4"])$(id).addEventListener("change",()=>{sync();redraw();if(active)showCell(st.cell);});
  const updateView = () => {sync();redraw();if(active)showCell(st.cell);};
  for(const id of ["overlap-component","overlap-map-component","overlap-map-component-2"])$(id).addEventListener("change",updateView);
  for(let i=0;i<2;i++)mapSelect(i).addEventListener("change",()=>{
    if(mode(i).startsWith("ipf_")) {
      ipfKind=mode(i);
      if(mode(1-i).startsWith("ipf_"))mapSelect(1-i).value=ipfKind;
    }
    updateView();
  });
  $("overlap-inspection").addEventListener("change",()=>showCell(st.cell));
  return {readOptions, restoreOptions, runBatchOverlap, get active(){return active;},get result(){return run;},sync,drawMap,drawLegend,showCell,
    inspectMap(i,cell){
      const source=mapSource(i);
      const component=source==="residual" && fresh()?mapComponent(source,cell):-1;
      $("overlap-component").value=source==="residual"?(component>=0?String(component+1):"primary"):source;
    },
    hover(cell,i=0){
      if(primaryMap(i)) {
        const seed=primarySeed(cell);
        return seed?`Original primary · NCC ${value(st.results.ncc[cell])}${primaryVisible(cell,seed)?"":" · Below primary minimum"}`:"Primary unavailable";
      }
      if(!run?.completed[cell])return "Uncomputed";if(!fresh())return "Context changed";
      const source=mapSource(i);
      if(source==="residual" && run.arrays.active_count?.values[cell]<2)return "No multi-component fit";
      const component=mapComponent(source,cell);
      if(component<0)return source==="residual"?"No accepted residual source":`No visible fit component ${source}`;
      const f=metricValue(run,cell,component,weightMetricKey(run))/100,info=MAPS[mode(i)],field=run.arrays[info?.field||mode(i)];
      const reading=field?.values[cell*field.stride+(field.stride===run.k?component:0)];
      return `Fit component ${component+1}${Number.isFinite(f)?` · ${(100*f).toFixed(1)}%`:""}${mode(i)!=="fraction" && Number.isFinite(reading)?` · ${mapSelect(i).selectedOptions[0].textContent} ${value(reading)}`:""}`;
    },
    activate(on){active=on;++inspectSerial;++st.showRun;if(!on)$("map-well").title=ordinaryMapTitle;$("pats").hidden=on;$("overlap-pats").hidden=!on;$("overlap-pattern-control").hidden=!on;$("map-kind").hidden=on;$("overlap-map-control").hidden=!on;$("overlap-map-view").hidden=!on;sync();redraw();if(st.cell>=0){if(on)showCell(st.cell);}},
  };
}
