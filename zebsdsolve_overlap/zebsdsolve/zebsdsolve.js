/**
 * zebsdsolve in the browser: loads zebsdsolve.wasm (Zig) and gives it WebGPU and files.
 *
 * The module schedules every GPU operation and every file read itself; this adapter only carries
 * out its requests (buffers, pipelines, dispatches, copies, submits through the browser's WebGPU
 * API; byte ranges of Blobs) and resolves them asynchronously. No indexing logic lives here.
 *
 *   import { Zebsdsolve } from "./zebsdsolve.js";
 *   const ze = await Zebsdsolve.create({ wasm: "zebsdsolve.wasm" });
 *   await ze.openScan([scanFile]);               // a File of any size: it is read in ranges
 *   await ze.addPhase(masterFile);
 *   await ze.fitGeometry();                     // the pattern-centre vote, then the plane of source points
 *   await ze.index(({ done, total }) => …);
 *   const { quaternions, ncc, phase } = ze.results();
 *
 * The module's blocking code suspends on GPU readbacks and file reads through WebAssembly
 * JavaScript Promise Integration (Chrome / Edge 137+). Calls are serialized; only cancel() may
 * run while one is suspended.
 */

const LIMIT_KEYS = ["maxStorageBufferBindingSize", "maxBufferSize", "maxComputeWorkgroupStorageSize", "maxStorageBuffersPerShaderStage"];
const utf8 = new TextDecoder();
const newStats = () => ({ dispatches: 0, submits: 0, reads: 0, waits: 0, waitMs: 0, compileWaitMs: 0, pipelines: 0, pipelineMs: 0 });
const utf8enc = new TextEncoder();
// ── pipeline warm-up across page loads ──
// Browsers compile a WGSL pipeline the first time a page asks for it, which costs 20–1000 ms
// each. The adapter remembers (in localStorage) which pipelines this origin created, and at
// the next load starts compiling all of them in parallel while the page sets up, so the first
// fit or indexing step finds them ready. The list belongs to one build of zebsdsolve.wasm (its
// fingerprint), so a new build starts a new list. `warm: false` in Zebsdsolve.create turns this off.
// A first visit starts from a list shipped with the site (`pipelines.json` beside the wasm,
// recorded by tests/test_web.mjs --save-pipelines) when it belongs to the same build.
// sphereView's projections, in the engine's order (sphereview.zig Projection)
const PROJECTIONS = ["globe", "lambert", "stereographic", "joint"];
const WARM_KEY = "zebsdsolve.pipelines.v1";
const WARM_MAX = 400;

export const overlapStopReasons = Object.freeze(["running","component_limit","flat_residual","no_proposal","unresolved_addition","boundary","missing_primary","flat_target","attempt_limit","invalid_target","stale_primary","residual_screen","replacement_limit","stage_improvement"]);

export function overlapBlurGainMetadata(enabled, theta, runtimeExports={}) {
  if(!enabled)return {blur_gain:false};
  const version=runtimeExports.ze_overlap_optimizer_version?.()??1;
  if(version!==1 && version!==2)throw new RangeError(`unsupported overlap optimizer metadata version ${version}`);
  const fine=version===2?{log_rho_polish_steps:[.04375,.021875,.0109375,.00546875],joint_singleton_seeds:"coarse_snapshot_then_polished_after_coordinates"}:{};
  return {blur_gain:true,blur_gain_theta:Float32Array.from(theta),model:"shared_blur_gain_prepared_projection_nnls",
    blur_gain_parameter_names:["sigma","log_rho","power","axis_y","axis_x","offset_y","offset_x"],
    blur_gain_optimizer:{method:"bounded_coordinate_pattern_search_variable_projection",version,max_evaluations:version===2?104:92,singleton_max_evaluations:version===2?96:88,initial_steps:[1.5,.35,2,.2,.2,.06,.06],scales:[1,.5,.25],sweeps_per_scale:2,...fine,
      bounds:[[0,5],[-Math.log(4),Math.log(4)],[.25,8],[.6,1.4],[.6,1.4],[-.15,.15],[-.15,.15]],sigma_units:"effective_fit_pixels",gain_domain:"prepared_master",physical_fractions:false,
      convergence:"fixed_budget_local_search",best_single:"independently_optimized_per_discovered_pose",best_single_refinement:"winning_joint_theta_single_projection",pixel_sse:"direct_parallel_64_lanes"}};
}

/** Undo only numerical column normalization; these are declared basis weights. */
export function contrastBasisMetadata(result, norms, theta, basis="globally_standardized_baked_master_contrast") {
  const k=result.max_components??result.maxComponents??(norms.length/result.sse.length),n=result.sse.length;
  const component_norms=Float32Array.from(norms),weights=new Float32Array(n*k).fill(NaN);
  for(let b=0;b<n;b++){
    let valid=!!result.objective_valid[b]&&!!result.design_rank_reliable[b]&&!result.flat[b]&&!result.conditioned[b],sum=0,count=0;
    for(let j=0;j<k;j++){
      const i=b*k+j,present=result.phase?result.phase[i]>=0:true,d=component_norms[i],a=result.amplitudes[i];
      if(!present){component_norms[i]=NaN;continue;}
      count++;if(!Number.isFinite(d)||d<=0||!Number.isFinite(a)||a<0)valid=false;
      sum+=a/d;
    }
    valid=valid&&count>0&&Number.isFinite(sum)&&sum>0;
    if(valid)for(let j=0;j<k;j++){const i=b*k+j;if(!result.phase||result.phase[i]>=0)weights[i]=(result.amplitudes[i]/component_norms[i])/sum;}
  }
  if(result.blur_gain_optimizer)Object.assign(result.blur_gain_optimizer,{gain_domain:"blurred_mask_centered_contrast",target_preparation:"original_once_no_residual_background_filter",nnls_solver:"kkt_enumerated_nnls_f32_tree",pixel_sse:"tiled_tree_parallel_64_lanes",kkt_tolerance_factor:32,kkt_float_epsilon:2**-23});
  return {component_norms,contrast_basis_weights:weights,blur_gain_theta:Float32Array.from(theta),fit_model:"contrast_gain_v1",gain_domain:"contrast",weight_basis:basis,contrast_nnls_solver:"kkt_enumerated_nnls_f32_tree",contrast_anchor:"masked_upper_median_radix_f32",contrast_target_centering:basis==="supplied_projection_contrast"?"provided_masked_f64_scalar_then_gpu_normalize":"existing_once_prepared_target",component_norm_definition:"masked_l2_center_gain_blur_centered_reference"};
}

/** Decode native records only; no crystal mathematics is implemented in JS. */
export function decodeOverlapArrays(records, state, progress, cells, k) {
  const n=cells.length, cv=new DataView(records.buffer,records.byteOffset,records.byteLength);
  const pv=new DataView(progress.buffer,progress.byteOffset,progress.byteLength);
  const out={cells:Float64Array.from(cells,Number),max_components:k,maxComponents:k,
    quaternions:new Float32Array(n*k*4),phase:new Int32Array(n*k),amplitudes:new Float32Array(n*k),
    prepared_fractions:new Float32Array(n*k).fill(NaN),proposal_ncc:new Float32Array(n*k),proposal_nmi:new Float32Array(n*k),
    discovery_id:new Int32Array(n*k).fill(-1),single_sse:new Float32Array(n*k),
    model:"fixed_prepared_projection_nnls",assessment:"unassessed_candidates",domain:"zero_mean_unit_norm_prepared_pattern",physical_fractions:false,blur_gain_parity:false};
  for(const key of ["sse","initial_sse","residual_norm","mixture_ncc","best_single_sse","improvement_over_best_single","component_correlation","relative_gram_condition","previous_sse","incremental_improvement"])out[key]=new Float32Array(n);
  for(const key of ["flags","candidate_count","fitted_count","retained_order","proposed_count","active_count","stop_reason","objective_order"])out[key]=new Uint32Array(n);
  for(const key of ["conditioned","flat","conditioning_assessed","residual_flat","objective_valid","attribution_unresolved","unresolved_candidate_skipped","design_rank_reliable","addition_unresolved"])out[key]=new Uint8Array(n);
  out.solve_status=[];out.stop_reason_name=[];
  for(let b=0;b<n;b++) {
    const offset=b*16, flags=Math.round(state[offset+3]), stop=pv.getUint32(b*32+12,true);
    let sum=0;
    for(let j=0;j<k;j++) {
      const a=b*k+j,o=a*32;
      for(let t=0;t<4;t++)out.quaternions[a*4+t]=cv.getFloat32(o+t*4,true);
      out.amplitudes[a]=cv.getFloat32(o+16,true);sum+=out.amplitudes[a];
      out.phase[a]=cv.getInt32(o+20,true);out.proposal_ncc[a]=out.phase[a]>=0?cv.getFloat32(o+24,true):NaN;out.proposal_nmi[a]=out.phase[a]>=0?cv.getFloat32(o+28,true):NaN;
      out.single_sse[a]=out.phase[a]>=0?state[offset+8+j]:NaN;
      if(out.phase[a]>=0){out.discovery_id[a]=j;out.candidate_count[b]++;}
      if(out.amplitudes[a]>1e-6)out.fitted_count[b]++;
    }
    if(sum>1e-12 && !(flags&1))for(let j=0;j<k;j++)out.prepared_fractions[b*k+j]=out.amplitudes[b*k+j]/sum;
    const ss=state[offset],initial=state[offset+1],condition=state[offset+13];
    out.sse[b]=ss;out.initial_sse[b]=initial;out.residual_norm[b]=Math.sqrt(Math.max(ss,0));
    out.mixture_ncc[b]=Math.sqrt(Math.max(0,Math.min(1,1-(initial>1e-12&&sum>0?ss/initial:NaN))));
    out.best_single_sse[b]=state[offset+14];out.improvement_over_best_single[b]=Math.max(0,state[offset+14]-ss);
    out.flags[b]=flags;out.conditioned[b]=!!(flags&1);out.flat[b]=!!(flags&2);
    out.conditioning_assessed[b]=!(flags&(2|4))&&Number.isFinite(condition);out.relative_gram_condition[b]=out.conditioning_assessed[b]?condition:NaN;
    out.component_correlation[b]=state[offset+12];out.residual_flat[b]=ss<=1e-8*Math.max(initial,1e-30);
    out.objective_valid[b]=!(flags&8)&&stop!==6&&stop!==10;out.attribution_unresolved[b]=!!(flags&8);out.unresolved_candidate_skipped[b]=!!(flags&16);
    out.design_rank_reliable[b]=!(flags&(8|2));out.addition_unresolved[b]=stop===4;
    for(const [key,o]of [["retained_order",0],["proposed_count",4],["active_count",8],["stop_reason",12]])out[key][b]=pv.getUint32(b*32+o,true);
    out.objective_order[b]=out.retained_order[b];out.previous_sse[b]=pv.getFloat32(b*32+16,true);out.incremental_improvement[b]=pv.getFloat32(b*32+20,true);
    out.stop_reason_name[b]=overlapStopReasons[stop]??"unknown";
    const unavailable=[6,9,10].includes(stop)||(out.retained_order[b]===0&&!out.flat[b]);
    if(unavailable) {
      out.objective_valid[b]=0;out.conditioning_assessed[b]=0;out.design_rank_reliable[b]=0;out.residual_flat[b]=0;
      for(const key of ["sse","initial_sse","residual_norm","mixture_ncc","best_single_sse","improvement_over_best_single","component_correlation","relative_gram_condition","previous_sse","incremental_improvement"])out[key][b]=NaN;
      for(let j=0;j<k;j++){out.single_sse[b*k+j]=NaN;out.prepared_fractions[b*k+j]=NaN;}
    }
    if(out.retained_order[b]===0) {
      out.conditioning_assessed[b]=0;out.design_rank_reliable[b]=0;
      for(const key of ["best_single_sse","improvement_over_best_single","component_correlation","relative_gram_condition","previous_sse","incremental_improvement"])out[key][b]=NaN;
    }
    out.solve_status[b]=unavailable?out.stop_reason_name[b]:flags&8?"numerically_unresolved":flags&2?"flat_target":flags&16?"unresolved_candidate_skipped":flags&4?"conditioning_unassessed":flags&1?"ill_conditioned":"solved";
  }
  return out;
}

function hashCode(str) {
  let h = 0x811c9dc5;
  for (let i = 0; i < str.length; i++) h = Math.imul(h ^ str.charCodeAt(i), 0x01000193);
  return (h >>> 0).toString(36) + "." + str.length.toString(36);
}

function hashBytes(bytes) {
  const u = new Uint8Array(bytes);
  let h = 0x811c9dc5;
  for (let i = 0; i < u.length; i++) h = Math.imul(h ^ u[i], 0x01000193);
  return (h >>> 0).toString(36) + "." + u.length.toString(36);
}

function loadWarm() {
  try { return JSON.parse(localStorage.getItem(WARM_KEY)); } catch { return null; }
}

async function fetchWarm(url) {
  try {
    const r = await fetch(url);
    return r.ok ? await r.json() : null;
  } catch { return null; }
}

function warmRecord(w) {
  const list = w.list.slice(-WARM_MAX);
  const codes = {};
  for (const [h] of list) codes[h] = w.codes.get(h);
  return { fp: w.fp, codes, list };
}

function saveWarm(state) {
  const w = state.warm;
  if (!w || w.timer) return;
  w.timer = setTimeout(() => {
    w.timer = null;
    try { localStorage.setItem(WARM_KEY, JSON.stringify(warmRecord(w))); } catch { /* quota: skip */ }
  }, 1500);
}

function gpuImports(state) {
  const { device } = state;
  const bufs = [null];
  const freeIds = [];
  // pipelines: { pipeline, layout } once compiled (pipeline null while compiling or failed)
  const pipes = [null];
  const modules = new Map();
  // (code hash, entry) → compile promise, shared by the warm-up and createPipeline
  const compiled = new Map();
  let enc = null;
  let pass = null;
  let recorded = [];
  // Pipelines compile asynchronously and in parallel (createComputePipelineAsync). While a
  // pipeline that recorded work uses is still compiling, queue operations are held back in
  // order (`backlog`) and replayed once it is ready, so the module keeps recording ahead and
  // only a wait (a readback) blocks on the compiler.
  let backlog = null;
  let blocking = new Set();
  let drained = null;
  const mem = () => state.memory.buffer;
  const bytes = (ptr, len) => new Uint8Array(mem(), ptr, len);
  const endPass = () => { if (pass) { pass.end(); pass = null; } };
  const encoder = () => (enc ||= device.createCommandEncoder());
  // counters for ze.stats(): what an operation cost in GPU round trips
  const st = state.stats;
  device.addEventListener?.("uncapturederror", (e) => state.errors.push(String(e.error?.message || e.error)));

  const defer = (op) => {
    if (!backlog) {
      backlog = [];
      let resolve;
      drained = { promise: new Promise((r) => (resolve = r)), resolve };
    }
    backlog.push(op);
  };
  const replay = () => {
    const ops = backlog;
    backlog = null;
    for (const op of ops) op();
    drained.resolve();
    drained = null;
  };

  // ── operations, issued directly (or replayed from the backlog) ──
  // indirect: [buffer, byte offset] of the workgroup counts (3 × u32) instead of x, y, z
  const doDispatch = (p, x, y, z, entries, indirect) => {
    if (!p.pipeline) return; // failed to compile: reported through takeError
    if (!pass) pass = encoder().beginComputePass();
    pass.setPipeline(p.pipeline);
    pass.setBindGroup(0, device.createBindGroup({ layout: p.layout, entries }));
    if (indirect) pass.dispatchWorkgroupsIndirect(indirect[0], indirect[1]);
    else pass.dispatchWorkgroups(Math.max(1, x), Math.max(1, y), Math.max(1, z));
  };
  const bindEntries = (bindsPtr, n) => {
    const v = new DataView(mem(), bindsPtr, n * 24);
    const entries = [];
    for (let i = 0; i < n; i++) {
      const o = i * 24;
      const size = Number(v.getBigUint64(o + 16, true));
      const resource = { buffer: bufs[v.getUint32(o + 4, true)], offset: Number(v.getBigUint64(o + 8, true)) };
      if (size) resource.size = size;
      entries.push({ binding: v.getUint32(o, true), resource });
    }
    return entries;
  };
  const doSubmit = () => {
    endPass();
    if (enc) { st.submits++; device.queue.submit([enc.finish()]); enc = null; }
    for (const b of recorded) b.destroy();
    recorded = [];
    for (const r of state.reads) {
      if (r.promise || !r.issued) continue;
      r.promise = (async () => {
        let mapped = false;
        try {
          await r.staging.mapAsync(GPUMapMode.READ);
          mapped = true;
          // wasm memory may have grown since the read was requested: take a fresh view.
          new Uint8Array(mem(), r.dst, r.size).set(new Uint8Array(r.staging.getMappedRange(), 0, r.size));
        } finally {
          try { if (mapped) r.staging.unmap(); } finally { r.staging.destroy(); }
        }
      })();
      // A submit may precede wait while the host reads the next batch's file ranges.
      r.promise.catch(() => {});
    }
  };
  const doDestroy = (b) => { if (enc) recorded.push(b); else b.destroy(); }; // never under an open batch

  const moduleOf = (code) => {
    let module = modules.get(code);
    if (!module) { module = device.createShaderModule({ code }); modules.set(code, module); }
    return module;
  };
  // Compile progress for onCompile listeners: counts restart once everything started has finished.
  const cp = state.compiling;
  const told = () => { for (const fn of cp.listeners) try { fn({ done: cp.done, total: cp.total }); } catch (e) { console.error(e); } };
  const compile = (code, h, entryPoint) => {
    const key = h + "/" + entryPoint;
    let c = compiled.get(key);
    if (!c) {
      if (cp.done === cp.total) cp.done = cp.total = 0;
      cp.total++;
      told();
      c = device.createComputePipelineAsync({ label: entryPoint, layout: "auto", compute: { module: moduleOf(code), entryPoint } });
      c.finally(() => { cp.done++; told(); }).catch(() => {});
      compiled.set(key, c);
    }
    return c;
  };
  // Start compiling what this origin used before. Entries from an older build may no longer
  // compile: their errors stay inside this error scope (not reported against a later call) and
  // they leave the list.
  const w = state.warm;
  if (w?.saved) {
    w.list = w.saved.list.slice();
    device.pushErrorScope("validation");
    for (const [h, entry] of w.saved.list) {
      const code = w.saved.codes[h];
      if (!code) continue;
      w.codes.set(h, code);
      compile(code, h, entry).catch(() => {
        const i = w.list.findIndex(([a, b]) => a === h && b === entry);
        if (i >= 0) w.list.splice(i, 1);
        saveWarm(state);
      });
    }
    device.popErrorScope().catch(() => {});
    w.saved = null;
  }

  return {
    limits(out) {
      const v = new DataView(mem(), out, 40);
      const L = device.limits;
      v.setBigUint64(0, BigInt(L.maxStorageBufferBindingSize), true);
      v.setBigUint64(8, BigInt(L.maxBufferSize), true);
      v.setUint32(16, L.maxComputeWorkgroupStorageSize, true);
      v.setUint32(20, L.maxStorageBuffersPerShaderStage, true);
      v.setUint32(24, L.minUniformBufferOffsetAlignment, true);
      v.setUint32(28, L.minStorageBufferOffsetAlignment, true);
      v.setUint32(32, device.features.has("shader-f16") ? 1 : 0, true);
      v.setUint32(36, navigator.gpu?.wgslLanguageFeatures?.has?.("packed_4x8_integer_dot_product") ? 1 : 0, true);
    },
    info(ptr, cap) {
      const b = utf8enc.encode(state.info).subarray(0, cap);
      bytes(ptr, b.length).set(b);
      return b.length;
    },
    createBuffer(size, usage) {
      const u = usage === 1
        ? GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
        : GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST | GPUBufferUsage.INDIRECT;
      const b = device.createBuffer({ size, usage: u });
      const id = freeIds.length ? freeIds.pop() : bufs.length;
      bufs[id] = b;
      return id;
    },
    destroyBuffer(id) {
      const b = bufs[id];
      if (!b) return;
      bufs[id] = null;
      freeIds.push(id);
      if (backlog) defer(() => doDestroy(b)); else doDestroy(b);
    },
    writeBuffer(id, offset, ptr, len) {
      const b = bufs[id];
      if (backlog) { const data = bytes(ptr, len).slice(); defer(() => device.queue.writeBuffer(b, offset, data)); }
      else device.queue.writeBuffer(b, offset, mem(), ptr, len);
    },
    // Starts compiling and returns at once; the first dispatch that needs it holds the queue
    // until it is ready.
    createPipeline(codePtr, codeLen, entryPtr, entryLen) {
      const t0 = performance.now();
      st.pipelines++;
      const code = utf8.decode(bytes(codePtr, codeLen));
      const entryPoint = utf8.decode(bytes(entryPtr, entryLen));
      const h = hashCode(code);
      if (w) {
        w.codes.set(h, code);
        const i = w.list.findIndex(([a, b]) => a === h && b === entryPoint);
        if (i >= 0) w.list.splice(i, 1);
        w.list.push([h, entryPoint]); // most recent last
        saveWarm(state);
      }
      const p = { pipeline: null, layout: null, ready: false };
      p.promise = compile(code, h, entryPoint)
        .then((pl) => { p.pipeline = pl; p.layout = pl.getBindGroupLayout(0); })
        .catch((e) => state.errors.push(`pipeline ${entryPoint}: ${e.message || e}`))
        .finally(() => {
          p.ready = true;
          if (blocking.delete(p) && backlog && blocking.size === 0) replay();
        });
      pipes.push(p);
      st.pipelineMs += performance.now() - t0;
      return pipes.length - 1;
    },
    dispatch(pipe, x, y, z, bindsPtr, n) {
      st.dispatches++;
      const p = pipes[pipe];
      const entries = bindEntries(bindsPtr, n);
      if (!p.ready) blocking.add(p);
      if (backlog || !p.ready) defer(() => doDispatch(p, x, y, z, entries));
      else doDispatch(p, x, y, z, entries);
    },
    dispatchIndirect(pipe, buf, offset, bindsPtr, n) {
      st.dispatches++;
      const p = pipes[pipe];
      const entries = bindEntries(bindsPtr, n);
      const ind = [bufs[buf], offset];
      if (!p.ready) blocking.add(p);
      if (backlog || !p.ready) defer(() => doDispatch(p, 0, 0, 0, entries, ind));
      else doDispatch(p, 0, 0, 0, entries, ind);
    },
    clearBuffer(id, offset, size) {
      const b = bufs[id];
      const op = () => { endPass(); encoder().clearBuffer(b, offset, size); };
      if (backlog) defer(op); else op();
    },
    copyBuffer(src, soff, dst, doff, size) {
      const a = bufs[src];
      const b = bufs[dst];
      const op = () => { endPass(); encoder().copyBufferToBuffer(a, soff, b, doff, size); };
      if (backlog) defer(op); else op();
    },
    readBuffer(src, offset, size, dst) {
      st.reads++;
      const a = bufs[src];
      const padded = Math.ceil(size / 4) * 4;
      const staging = device.createBuffer({ size: padded, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
      const r = { staging, size, dst, promise: null, issued: false };
      state.reads.push(r);
      const op = () => { endPass(); encoder().copyBufferToBuffer(a, offset, staging, 0, padded); r.issued = true; };
      if (backlog) defer(op); else op();
    },
    submit() {
      if (backlog) defer(doSubmit); else doSubmit();
    },
    takeError(ptr, cap) {
      // Fatal GPU diagnostics are retained separately. Residual asynchronous adapter errors
      // must not turn a successful CPU-only results/export call into another GPU failure.
      if (state.deviceLoss) { state.errors.length = 0; return 0; }
      if (!state.errors.length) return 0;
      const b = utf8enc.encode(state.errors.shift()).subarray(0, cap);
      bytes(ptr, b.length).set(b);
      return b.length;
    },
    // Suspending import (JS Promise Integration): the module's stack waits here while pipelines
    // still compiling and the submitted reads resolve, then resumes as if the call had blocked.
    wait: new WebAssembly.Suspending(async () => {
      const t0 = performance.now();
      st.waits++;
      try {
        if (drained) { await drained.promise; st.compileWaitMs += performance.now() - t0; }
        const reads = state.reads;
        state.reads = [];
        // Never unwind the native stack (and free its destinations) while another mapping
        // can still copy into wasm memory, even when an earlier mapping has failed.
        const settled = await Promise.allSettled(reads.map((r) => r.promise));
        const failed = settled.find((r) => r.status === "rejected");
        if (failed) throw failed.reason;
        return 0;
      } catch (e) { state.errors.push(String(e)); return 1; } finally { st.waitMs += performance.now() - t0; }
    }),
  };
}

// ── files ──
// The module reads scans and master patterns itself, in ranges (src/io/file.zig). Here a file is
// a Blob (a File the user picked, or a fetched example): `open` resolves the token `addFile`
// gave it, `read` slices it. `read` is a suspending import: the module's stack waits while the
// slice loads, so a scan far larger than memory is read in place. `readv` loads many slices in
// one suspension: the module asks for scattered patterns (a preview lattice, the chunks of an
// HDF5 dataset) together, because each suspension costs about as much as half a megabyte read.
function ioImports(state) {
  const files = state.files;
  // Load ranges of a file into wasm memory back to back; 0 on success. Slices load in
  // parallel, a large range as several pieces (Chrome reads four pieces of a range together
  // about three times as fast as the range whole), a few hundred at a time.
  const PIECE = 4 << 20;
  const load = async (id, off, len, dst) => {
    try {
      const t0 = performance.now();
      const file = files[id];
      const pieces = [];
      for (let i = 0; i < off.length; i++) {
        for (let a = 0; a < len[i]; a += PIECE) pieces.push([off[i] + a, Math.min(PIECE, len[i] - a)]);
      }
      let at = dst;
      for (let i0 = 0; i0 < pieces.length; i0 += 512) {
        const bufs = await Promise.all(pieces.slice(i0, i0 + 512).map(([o, n]) => file.slice(o, o + n).arrayBuffer()));
        // wasm memory may have grown while the slices loaded: take a fresh view
        const mem = new Uint8Array(state.memory.buffer);
        for (let k = 0; k < bufs.length; k++) {
          if (bufs[k].byteLength !== pieces[i0 + k][1]) return 1;
          mem.set(new Uint8Array(bufs[k]), at);
          at += bufs[k].byteLength;
        }
      }
      state.io.reads++;
      state.io.slices += pieces.length;
      state.io.bytes += at - dst;
      state.io.ms += performance.now() - t0;
      if (state.patternStatsIo) {
        state.patternStatsIo.fileReads++;
        state.patternStatsIo.fileBytes += at - dst;
      }
      return 0;
    } catch (e) { state.errors.push(`file read: ${e.message || e}`); return 1; }
  };
  return {
    open(ptr, len) {
      const name = utf8.decode(new Uint8Array(state.memory.buffer, ptr, len));
      const m = /^@(\d+):/.exec(name);
      const id = m ? +m[1] : 0;
      return files[id] ? id : 0;
    },
    size(id) { return files[id] ? files[id].size : 0; },
    read: new WebAssembly.Suspending((id, offset, dst, len) => load(id, [offset], [len], dst)),
    readv: new WebAssembly.Suspending((id, n, offs, lens, dst) =>
      // the ranges, copied out: wasm memory may grow while the slices load
      load(id, new Float64Array(state.memory.buffer, offs, n).slice(), new Uint32Array(state.memory.buffer, lens, n).slice(), dst)),
    // the engine closes a file when it is done with it; the Blob stays for a later open
    close() {},
  };
}


/** Decode host provenance only for the capability/version that owns offset 28. */
export function overlapTrajectoryMetadata(out, progress, keepN, k, version=0) {
  const n=out.retained_order.length;
  if(keepN>1&&k>=3&&![0,1].includes(version))throw new Error(`unsupported overlap trajectory version ${version}`);
  if(keepN>1&&k>=3&&version===1){
    const pv=new DataView(progress.buffer,progress.byteOffset,progress.byteLength);
    return {trajectory_selection:"best_valid_final_sse_of_pooled_and_rank1_v1",
      trajectory_source:Int32Array.from({length:n},(_,b)=>{
        const source=pv.getUint32(b*32+28,true);
        if(![0,1,0xffffffff].includes(source))throw new Error("invalid V5 overlap trajectory provenance");
        return source===0xffffffff?-1:source;
      })};
  }
  return {trajectory_selection:k===1?"ordinary_primary_only_v1":keepN===1?"rank1_only_v1":"greedy_pooled_v5",
    trajectory_source:Int32Array.from({length:n},(_,b)=>out.retained_order[b]>0&&out.objective_valid[b]&&Number.isFinite(out.sse[b])&&out.sse[b]>=0&&Number.isFinite(out.initial_sse[b])&&out.initial_sse[b]>1e-12?(keepN>1&&k>1?1:0):-1)};
}

/** V6 request-local trace; cells map local indices back to immutable scan IDs. */
export function decodeOverlapProfileEvents(bytes, cells, selectionVersion=1, staged=false) {
  const flagMask = staged ? 4095 : selectionVersion === 2 ? 1023 : 255;
  if(bytes.byteLength%64)throw new RangeError("V6 overlap trace must contain complete 64-byte events");
  const v=new DataView(bytes.buffer,bytes.byteOffset,bytes.byteLength),events=[];
  for(let o=0;o<bytes.byteLength;o+=64){
    const cell_index=v.getUint32(o,true),branch=v.getUint32(o+4,true),flags=v.getUint32(o+24,true);
    if(cell_index>=cells.length||branch>1||(flags&~flagMask))throw new Error("invalid V6 overlap event");
    events.push({cell_index,cell:Number(cells[cell_index]),branch,order:v.getUint32(o+8,true),attempt:v.getUint32(o+12,true),
      phase:v.getInt32(o+16,true),rank:v.getInt32(o+20,true),flags,result_order:v.getUint32(o+28,true),
      source_q:Array.from({length:4},(_,j)=>v.getFloat32(o+32+j*4,true)),residual_ncc:v.getFloat32(o+48,true),
      baseline_sse:v.getFloat32(o+52,true),lower_sse:v.getFloat32(o+56,true),joint_sse:v.getFloat32(o+60,true)});
  }
  return events;
}

// Exports that never wait (plain calls); every other function export may suspend (on the GPU
// or on a file read) and is called through WebAssembly.promising.
const SYNC_EXPORTS = new Set(["ze_alloc", "ze_free", "ze_version", "ze_overlap_optimizer_version", "ze_overlap_trajectory_version", "ze_overlap_profile_version", "ze_overlap_profile_selection_version", "ze_overlap_stage_gate_version", "ze_oxford_detector_attitude_version", "ze_overlap_profile_trace_capacity", "ze_create", "ze_destroy", "ze_adapter", "ze_error",
  "ze_has_acquisition_calibration", "ze_configure", "ze_settings", "ze_events", "ze_cancel", "ze_close_scan", "ze_scan_info", "ze_scan_format", "ze_scan_overview_kind",
  "ze_get_geometry", "ze_set_geometry", "ze_clear_phases", "ze_phase_count", "ze_phase_info", "ze_pattern_shape", "ze_nlpar_view",
  "ze_seed_source_roi", "ze_stored_primary_validity", "ze_phase_generation", "ze_orientation_colors", "ze_pc_scores", "ze_fit_patterns", "ze_results", "ze_map", "ze_ipf_key", "ze_export", "ze_export_snapshot", "ze_export_oxford_snapshot", "ze_result_checkpoint", "ze_restore_result_checkpoint", "ze_result_chance", "ze_bunge", "ze_reduce", "ze_disorientation", "ze_shader",
  "ze_index_status", "ze_tune_begin", "ze_tune_report", "ze_pixel_angle", "ze_lattice", "ze_sweep", "ze_pc_level_count", "ze_pc_level", "ze_pc_level_ballots",
  "ze_gpu_counts", "ze_profile", "ze_profile_report", "ze_compile_wgsl", "ze_plane_points",
  "ze_sim_read_cif", "ze_sim_set_crystal", "ze_sim_edit_crystal", "ze_sim_crystal", "ze_sim_sheets", "ze_sim_add_phase", "ze_sim_npz",
  "ze_cell_info", "ze_sim_energy"]);

// Struct layouts (src/engine.zig, geometry.zig, pcfit.zig, planefit.zig): [name, type, byte offset].
const SCAN = [["rows", "u32", 0], ["cols", "u32", 4], ["patH", "u32", 8], ["patW", "u32", 12], ["pixel", "u32", 16],
  ["hasBackground", "bool", 20], ["hasPc", "bool", 24], ["formatId", "u32", 28], ["voltageKv", "f64", 32], ["bytes", "f64", 40], ["hasPx", "bool", 48], ["geometryFrame", "u32", 52]];
const SCAN_SIZE = 56;
const PHASE = [["side", "u32", 0], ["pointGroup", "u32", 4], ["spaceGroup", "u32", 8], ["laueId", "u32", 12],
  ["nFold", "u32", 16], ["mirror", "bool", 20], ["inversion", "bool", 24], ["voltageKv", "f64", 32]];
const PHASE_SIZE = 104;
const GEOMETRY = [["pc", "f64x3", 0], ["sampleTiltDeg", "f64", 24], ["detectorTiltDeg", "f64", 32], ["rot", "f64x3", 40],
  ["pxUm", "f64", 64], ["stepXUm", "f64", 72], ["stepYUm", "f64", 80], ["planeKind", "u32", 88], ["plane", "f64x12", 96]];
const GEOMETRY_SIZE = 192;
/** What a geometry's plane of source points is (its `planeKind`), and a plane fit's `kind`. */
export const PLANE_KINDS = ["none", "affine", "projective"];
const PLANE_RESULT = [["plane", "f64x12", 0], ["pc", "f64x3", 96], ["startPc", "f64x3", 120], ["rot", "f64x3", 144], ["saidRot", "f64x3", 168], ["saidRotSe", "f64x3", 192],
  ["step", "f64x2", 216], ["stepSe", "f64x2", 232], ["skew", "f64", 248], ["skewSe", "f64", 256], ["pcSe", "f64x3", 264], ["rmsd", "f64x3", 288], ["ncc", "f64", 312], ["startNcc", "f64", 320],
  ["kind", "u32", 328], ["frame", "u32", 332], ["nPatterns", "u32", 336], ["nUsed", "u32", 340], ["iterations", "u32", 344], ["blocks", "u32", 348],
  ["model", "u32", 352], ["compared", "u32", 356], ["cv", "f64x4", 360], ["cvSe", "f64x4", 392], ["method", "u32", 424]];
const PLANE_RESULT_SIZE = 432;
/** How the source moves over the scan, a plane fit's `model`: the automatic fit takes the simplest the map supports. */
export const PLANE_MODELS = ["centre", "line", "affine", "projective"];
const PLANE_POINT_SIZE = 72;
const PC_RESULT = [["pc", "f64x3", 0], ["start", "f64x3", 24], ["score", "f64", 48], ["startScore", "f64", 56],
  ["nPatterns", "u32", 64], ["nCandidates", "u32", 68], ["levels", "u32", 72], ["atBorder", "bool", 76]];
const PC_RESULT_SIZE = 80;
const SIM_SUMMARY = [["nK", "u32", 0], ["nSites", "u32", 4], ["nG", "u32", 8], ["nBins", "u32", 12], ["nBinsRun", "u32", 16], ["maxBeams", "u32", 20],
  ["failed", "u32", 24], ["stopped", "bool", 28], ["halfw", "u32", 32], ["rank", "u32", 36], ["maxWeak", "u32", 40], ["overflow", "u32", 44], ["lastChange", "f64", 48]];
const SIM_SUMMARY_SIZE = 56;
const INDEX_STATUS = [["sweep", "u32", 0], ["sweeps", "u32", 4], ["bandwidth", "u32", 8], ["dictGrid", "u32", 12],
  ["done", "u64", 16], ["total", "u64", 24], ["region", "u64", 32], ["accepted", "u64", 40], ["methodId", "u32", 48]];
const INDEX_STATUS_SIZE = 56;
export const METHODS = ["spherical", "dictionary", "pca"];

export const LAUE = [null, "-1", "2/m", "mmm", "4/m", "4/mmm", "-3", "-3m", "6/m", "6/mmm", "m-3", "m-3m"];
export const POINT_GROUPS = [null, "1", "-1", "2", "m", "2/m", "222", "mm2", "mmm", "4", "-4", "4/m", "422", "4mm", "-42m", "4/mmm",
  "3", "-3", "32", "3m", "-3m", "6", "-6", "6/m", "622", "6mm", "-6m2", "6/mmm", "23", "m-3", "432", "-43m", "m-3m"];
export const MAPS = { ipf_x: 0, ipf_y: 1, ipf_z: 2, ncc: 3, nmi: 4, phase: 5, z: 6, sweep: 7 };

export class Zebsdsolve {
  /** Load the module and open a high-performance device with the limits the engine needs. */
  static async create({ wasm = new URL("../zig-out/web/zebsdsolve.wasm", import.meta.url), device = null, warm = true, onCompile = null } = {}) {
    let info = "";
    let adapterInfo = {};
    if (!device) {
      const adapter = await navigator.gpu?.requestAdapter({ powerPreference: "high-performance" });
      if (!adapter) throw new Error("WebGPU is not available");
      const requiredLimits = {};
      for (const k of LIMIT_KEYS) requiredLimits[k] = adapter.limits[k];
      const requiredFeatures = adapter.features.has("shader-f16") ? ["shader-f16"] : [];
      device = await adapter.requestDevice({ requiredLimits, requiredFeatures });
      const ai = adapter.info || {};
      adapterInfo = Object.fromEntries(["vendor", "architecture", "device", "description"].map((k) => [k, String(ai[k] || "")]));
      info = `${ai.vendor || "gpu"} ${ai.architecture || ai.device || ""}`.trim();
    }
    const compiling = { done: 0, total: 0, listeners: onCompile ? [onCompile] : [] };
    const state = { device, memory: null, reads: [], errors: [], info, adapterInfo: Object.freeze(adapterInfo), deviceLoss: null, listeners: [], stats: newStats(), compiling,
      files: [null], io: { reads: 0, slices: 0, bytes: 0, ms: 0 }, patternStatsIo: null };
    // This callback only reports JS state. It must never enter wasm while an operation is
    // suspended, nor race that operation's lifetime against device.lost.
    device.lost?.then((loss) => {
      if (state.deviceLoss) return;
      state.deviceLoss = Object.freeze({ reason: String(loss.reason || "unknown"), message: String(loss.message || "") });
      const ev = { kind: "deviceLost", ...state.deviceLoss };
      for (const fn of state.listeners) try { fn(ev); } catch (e) { console.error(e); }
    });
    const warmOn = warm && typeof localStorage !== "undefined";
    // This origin's own list when it belongs to this build; else the shipped one, downloaded
    // alongside the wasm on a first visit (or after the wasm changed).
    const own = warmOn ? loadWarm() : null;
    const shippedUrl = new URL("pipelines.json", new URL(wasm, location.href));
    const shipped = warmOn && !own ? fetchWarm(shippedUrl) : null;
    const bytes = await (await fetch(wasm)).arrayBuffer();
    if (warmOn) {
      const fp = hashBytes(bytes);
      let saved = own?.fp === fp ? own : null;
      if (!saved) {
        const ship = await (shipped || fetchWarm(shippedUrl));
        saved = ship?.fp === fp ? ship : null;
      }
      state.warm = { fp, saved, codes: new Map(), list: [], timer: null };
    }
    const env = {
      ze_log(ptr, len) { console.warn("zebsdsolve:", utf8.decode(new Uint8Array(state.memory.buffer, ptr, len))); },
      // progress events from long operations (fits, indexing): one JSON object each
      ze_event(ptr, len) {
        const ev = JSON.parse(utf8.decode(new Uint8Array(state.memory.buffer, ptr, len)));
        for (const fn of state.listeners) try { fn(ev); } catch (e) { console.error(e); }
      },
    };
    if (typeof WebAssembly.Suspending !== "function") throw new Error("this browser lacks WebAssembly JS Promise Integration (Chrome / Edge 137+)");
    const { instance } = await WebAssembly.instantiate(bytes, { zegpu: gpuImports(state), zeio: ioImports(state), env });
    state.memory = instance.exports.memory;
    const x = {};
    for (const [k, v] of Object.entries(instance.exports)) {
      x[k] = typeof v === "function" && !SYNC_EXPORTS.has(k) ? WebAssembly.promising(v) : v;
    }
    return new Zebsdsolve(x, state);
  }

  constructor(x, state) {
    this.x = x;
    this.state = state;
    this._queue = Promise.resolve();
    this._disposed = false;
    this._scanGeneration = 0;
    this.e = x.ze_create();
    if (!this.e) throw new Error("zebsdsolve: engine creation failed");
  }

  /** Cheap native asset epoch, independent of warm-up/index scheduling. */
  get phaseGeneration() { return this._frame(()=>Number(this.x.ze_phase_generation(this.e))); }
  get detectorAttitudeVersion() { return this.x.ze_oxford_detector_attitude_version?.(this.e) ?? null; }

  get hasAcquisitionCalibration() { return !!this._frame(() => this.x.ze_has_acquisition_calibration(this.e)); }
  get device() { return this.state.device; }
  get deviceLost() { return this.state.deviceLoss || null; }
  get adapterInfo() { return this.state.adapterInfo; }
  get mem() { return this.state.memory.buffer; }

  get version() {
    const p = this.x.ze_version();
    const u = new Uint8Array(this.mem, p, 32);
    return utf8.decode(u.subarray(0, u.indexOf(0)));
  }

  /** Call fn({ done, total }) as pipelines compile (total counts the current burst). */
  onCompile(fn) {
    this.state.compiling.listeners.push(fn);
    fn({ done: this.state.compiling.done, total: this.state.compiling.total });
  }

  /** The pipelines this page created, as the warm-up list (pipelines.json) records them. */
  warmList() { return this.state.warm ? warmRecord(this.state.warm) : null; }

  /** GPU and file traffic since the last call, then reset (fileReads: round trips; fileSlices: ranges read). */
  stats() {
    const s = { ...this.state.stats, fileReads: this.state.io.reads, fileSlices: this.state.io.slices, fileBytes: this.state.io.bytes, fileMs: this.state.io.ms };
    Object.assign(this.state.stats, newStats());
    this.state.io = { reads: 0, slices: 0, bytes: 0, ms: 0 };
    return s;
  }

  /** Listen to progress and deviceLost events; returns an unsubscribe. */
  onEvent(fn) {
    this.state.listeners.push(fn);
    return () => { this.state.listeners = this.state.listeners.filter((f) => f !== fn); };
  }

  // ── plumbing ──

  /** Temporary wasm allocations for one call, freed afterwards (each at least 8 bytes, so 8-aligned). */
  _frame(fn) {
    if (this._disposed) throw new Error("zebsdsolve: engine disposed");
    const held = [];
    const f = {
      alloc: (n) => { const p = this.x.ze_alloc(Math.max(n, 8)); if (!p) throw new Error("zebsdsolve: out of memory"); held.push([p, Math.max(n, 8)]); return p; },
      f32: (arr) => { const p = f.alloc(arr.length * 4); new Float32Array(this.mem, p, arr.length).set(arr); return p; },
      u64: (arr) => { const p = f.alloc(arr.length * 8); const v = new BigUint64Array(this.mem, p, arr.length); for (let i = 0; i < arr.length; i++) v[i] = BigInt(arr[i]); return p; },
      str: (s) => { const b = utf8enc.encode(s); const p = f.alloc(b.length); new Uint8Array(this.mem, p, b.length).set(b); return [p, b.length]; },
    };
    const done = () => { for (const [p, n] of held) this.x.ze_free(p, n); };
    let out;
    try { out = fn(f); } catch (e) { done(); throw e; }
    if (out instanceof Promise) return out.finally(done);
    done();
    return out;
  }

  _read(layout, p) {
    const v = new DataView(this.mem);
    const o = {};
    for (const [k, t, off] of layout) {
      const a = p + off;
      const many = /^(u32|i32|f64)x(\d+)$/.exec(t);
      if (many) {
        const size = many[1] === "f64" ? 8 : 4;
        o[k] = Array.from({ length: +many[2] }, (_, i) => (many[1] === "f64" ? v.getFloat64(a + size * i, true) : many[1] === "i32" ? v.getInt32(a + size * i, true) : v.getUint32(a + size * i, true)));
        continue;
      }
      o[k] = t === "u32" ? v.getUint32(a, true) : t === "bool" ? !!v.getUint32(a, true) : t === "u64" ? Number(v.getBigUint64(a, true)) : v.getFloat64(a, true);
    }
    return o;
  }

  _error() {
    return this._frame((f) => {
      const p = f.alloc(1024);
      const n = this.x.ze_error(this.e, p, 1024);
      return utf8.decode(new Uint8Array(this.mem, p, n));
    });
  }

  _check(rc, what) {
    const err = this._error();
    if (Number(rc) < 0 || err) {
      // Native failure text takes precedence over takeError's adapter queue. Consume the
      // remaining adapter diagnostics with this failed call, not the next results/export.
      const extra = Number(rc) < 0 ? this.state.errors?.splice(0) || [] : [];
      if (this.deviceLost) extra.push(this._deviceLossError().message);
      throw new Error(`${what}: ${[err || rc, ...extra].join("; ")}`);
    }
    return Number(rc);
  }

  _deviceLossError() {
    const loss = this.deviceLost;
    const e = new Error(`GPU device lost (${loss.reason})${loss.message ? `: ${loss.message}` : ""}. Reload to resume GPU work; completed results remain available for export.`);
    e.name = "GPUDeviceLost";
    return e;
  }

  /** Calls are serialized: one operation on the module at a time (only cancel() may interleave). */
  _serial(fn, allowDisposed = false) {
    const guarded = () => {
      if (this._disposed && !allowDisposed) throw new Error("zebsdsolve: engine disposed");
      if (this.deviceLost && !allowDisposed) throw this._deviceLossError();
      return fn();
    };
    const run = this._queue.then(guarded);
    this._queue = run.catch(() => {});
    return run;
  }

  /** Drain in-flight work before freeing the engine. */
  dispose() {
    if (this._disposed) return this._queue;
    this._disposed = true;
    return this._serial(() => { if (this.e) { this.x.ze_destroy(this.e); this.e = 0; } }, true);
  }

  _text(call, cap = 256) {
    return this._frame((f) => {
      const p = f.alloc(cap);
      return utf8.decode(new Uint8Array(this.mem, p, call(p, cap)));
    });
  }

  get adapter() { return this._text((p, cap) => this.x.ze_adapter(this.e, p, cap)); }

  /** Ask the running fit or indexing to stop at its next step (it keeps what it has). */
  cancel() { if (!this._disposed) this.x.ze_cancel(this.e); }

  // ── settings and geometry ──

  /** Apply settings overrides (see src/settings.zig for the keys). */
  configure(obj) {
    return this._frame((f) => {
      const [p, n] = f.str(JSON.stringify(obj));
      this._check(this.x.ze_configure(this.e, p, n), "configure");
    });
  }

  get settings() {
    return this._frame((f) => {
      let cap = 8192, p = f.alloc(cap), n = this.x.ze_settings(this.e, p, cap);
      if (n > cap) { cap = n; p = f.alloc(cap); n = this.x.ze_settings(this.e, p, cap); }
      return JSON.parse(utf8.decode(new Uint8Array(this.mem, p, n)));
    });
  }

  /** The detector geometry: { pc[3], sampleTiltDeg, detectorTiltDeg, rot[3], pxUm, stepXUm, stepYUm }.
   *  pxUm is the size of a stored pattern pixel on the detector: the scan's, or one set here, or
   *  (when neither says) the detector's width, the setting detector_width_mm, over the pattern's
   *  pixels across. No fit changes it. */
  get geometry() {
    return this._frame((f) => {
      const p = f.alloc(GEOMETRY_SIZE);
      this.x.ze_get_geometry(this.e, p);
      return this._read(GEOMETRY, p);
    });
  }

  /** Set geometry fields. Oxford unknown detector pixel size stays zero; other
   *  formats may use the scan pixel size or detector-width default. */
  setGeometry(fields) {
    this._frame((f) => {
      const p = f.alloc(GEOMETRY_SIZE);
      this.x.ze_get_geometry(this.e, p);
      const v = new DataView(this.mem);
      for (const [k, t, off] of GEOMETRY) {
        if (!(k in fields)) continue;
        const many = /^f64x(\d+)$/.exec(t);
        if (many) for (let i = 0; i < +many[1]; i++) v.setFloat64(p + off + 8 * i, +fields[k][i], true);
        else if (t === "u32") v.setUint32(p + off, +fields[k], true);
        else v.setFloat64(p + off, +fields[k], true);
      }
      this.x.ze_set_geometry(this.e, p);
    });
  }

  // ── scan ──

  /** Register a Blob (a picked File, or fetched bytes) the module may open; returns its token. */
  addFile(blob, name = blob.name || "file") {
    this.state.files.push(blob);
    return `@${this.state.files.length - 1}:${name}`;
  }

  /** Open pattern and sidecar Blobs. Optional settings/hints commit together with the new scan. */
  openScan(blobs, options = null) {
    return this._serial(() => this._frame(async (f) => {
      const tokens = [].concat(blobs).map((b) => this.addFile(b));
      const [p, n] = f.str(tokens.join("\n"));
      if (options) {
        const [sp, sn] = f.str(JSON.stringify(options.settings || {}));
        const [hp, hn] = f.str(JSON.stringify(options.hints || {}));
        this._check(await this.x.ze_open_scan_configured(this.e, p, n, sp, sn, hp, hn), "open");
      } else this._check(await this.x.ze_open_scan(this.e, p, n), "open");
      ++this._scanGeneration;
      return this.scan;
    }));
  }

  /** Make a toy scan from the phases loaded (settings toy_*), in place of any scan open: grains
   *  with random orientations and phases, simulated under the present geometry (its true one),
   *  laid on a background and counted as electrons. Resolves to its answer: { quaternions (4 per
   *  cell), phase (−1: a blank cell), grain, geometry }. */
  synthesize() {
    return this._serial(() => this._frame(async (f) => {
      const s = this.settings, n = s.toy_rows * s.toy_cols;
      const qp = f.alloc(16 * n), pp = f.alloc(4 * n), gp = f.alloc(4 * n);
      this._check(await this.x.ze_synthesize(this.e, qp, pp, gp, n), "synthesize");
      ++this._scanGeneration;
      const geometry = this.geometry;
      return { quaternions: new Float32Array(this.mem, qp, 4 * n).slice(), phase: new Int32Array(this.mem, pp, n).slice(), grain: new Int32Array(this.mem, gp, n).slice(), geometry };
    }));
  }

  /** Take rows · cols patterns of h × w from a Uint8Array, Uint16Array or Float32Array (row-major). */
  setPatterns(data, rows, cols, h, w) {
    return this._serial(() => this._frame(async (f) => {
      const pixel = data instanceof Uint8Array ? 0 : data instanceof Uint16Array ? 1 : 2;
      const bytes = new Uint8Array(data.buffer, data.byteOffset, data.byteLength);
      const p = f.alloc(bytes.length);
      new Uint8Array(this.mem, p, bytes.length).set(bytes);
      this._check(await this.x.ze_set_patterns(this.e, p, bytes.length, pixel, rows, cols, h, w), "set_patterns");
      ++this._scanGeneration;
    }));
  }

  /** Close the scan, after the calls queued before (one may be waiting on the file's bytes). */
  closeScan() {
    return this._serial(() => {
      this.x.ze_close_scan(this.e);
      ++this._scanGeneration;
    });
  }

  /** What the scan is: { rows, cols, patH, patW, pixel, format, hasBackground, hasPc, voltageKv, bytes }. */
  get scan() {
    const s = this._frame((f) => {
      const p = f.alloc(SCAN_SIZE);
      this.x.ze_scan_info(this.e, p);
      return this._read(SCAN, p);
    });
    s.format = this._text((p, cap) => this.x.ze_scan_format(this.e, p, cap), 64);
    s.dtype = ["uint8", "uint16", "float32"][s.pixel];
    return s;
  }

  /** The acquisition's stored scalar map, without reading diffraction patterns. */
  storedOverview() {
    return this._serial(() => this._frame(async (f) => {
      const n = this._check(await this.x.ze_scan_overview(this.e, 0, 0), "stored overview");
      if (!n) return null;
      const kind = this.x.ze_scan_overview_kind(this.e);
      const label = [null, "Band contrast", "Image quality", "Radon quality", "Pattern quality"][kind];
      if (!label) throw new Error("Unknown stored overview kind.");
      const p = f.alloc(n * 4);
      const count = this._check(await this.x.ze_scan_overview(this.e, p, n), "stored overview");
      if (count !== n) throw new Error("Stored overview size changed while reading.");
      return { kind, label, values: new Float32Array(this.mem, p, n).slice() };
    }));
  }

  /** [height, width] of a pattern: reduced by the binning setting, or (full) as stored. */
  shape(full = false) {
    return this._frame((f) => {
      const p = f.alloc(8);
      this._check(this.x.ze_pattern_shape(this.e, full ? 1 : 0, p), "pattern_shape");
      return Array.from(new Uint32Array(this.mem, p, 2));
    });
  }

  /** [height, width] of a pattern as the search sees it (reduced by the binning setting). */
  get patternShape() { return this.shape(false); }

  /** One pattern (map cell, row-major) as Float32Array(h·w), at the reduced size or (full) the
   *  stored one. `stage`: "filtered" (or true), as it is indexed: averaged with its neighbours
   *  when the nlpar setting is on, then filtered; "stored" (or false), as it is stored;
   *  "averaged", averaged with its neighbours and not filtered; "background", not the cell's
   *  pattern but the static background subtracted from every pattern (an error when the
   *  static_background setting subtracts none). */
  pattern(cell, stage = true, full = false) {
    return this._serial(() => this._frame(async (f) => {
      const [h, w] = this.shape(full);
      const p = f.alloc(h * w * 4);
      const k = stage === "background" ? 3 : stage === "averaged" ? 2 : stage === "stored" || !stage ? 0 : 1;
      this._check(await this.x.ze_pattern(this.e, BigInt(cell), k, full ? 1 : 0, p, h * w), "pattern");
      const out = new Float32Array(this.mem, p, h * w).slice();
      // what the averaging did to it, kept with it (another pattern() replaces the engine's)
      out.nlpar = this._nlparView();
      return out;
    }));
  }

  /** What averaging with its neighbours did to the pattern last read: { noise (as stored,
   *  standard deviation a pixel), left (the share of it the average keeps), own (the share of
   *  the average that is the pattern itself), side, shares (side², the pattern in the middle,
   *  NaN outside the map) }, or null when it was not averaged. */
  _nlparView() {
    return this._frame((f) => {
      const op = f.alloc(12);
      const side = this.x.ze_nlpar_view(this.e, op, 0, 0);
      if (!side) return null;
      const sp = f.alloc(side * side * 4);
      this.x.ze_nlpar_view(this.e, op, sp, side * side);
      const o = new Float32Array(this.mem, op, 3);
      return { noise: o[0], left: o[1], own: o[2], side, shares: new Float32Array(this.mem, sp, side * side).slice() };
    });
  }

  /** About n map cells on a lattice over the whole map (every cell when it has no more). */
  lattice(n) {
    return this._frame((f) => {
      const count = this._check(this.x.ze_lattice(this.e, n, 0, 0), "lattice");
      const p = f.alloc(count * 8);
      this._check(this.x.ze_lattice(this.e, n, p, count), "lattice");
      return Array.from(new BigUint64Array(this.mem, p, count), Number);
    });
  }

  /** Angular width of one stored detector pixel as the master filters take it, degrees. */
  get pixelAngle() { if (this._disposed) throw new Error("zebsdsolve: engine disposed"); return this.x.ze_pixel_angle(this.e); }

  /** Mean, standard deviation and file traffic of this batch of stored patterns: { mean, std, io }. */
  patternStats(cells) {
    return this._serial(() => this._frame(async (f) => {
      const n = cells.length;
      const out = f.alloc(n * 8);
      const io = { fileReads: 0, fileBytes: 0 };
      this.state.patternStatsIo = io;
      try {
        this._check(await this.x.ze_pattern_stats(this.e, f.u64(cells), n, out, 2 * n), "pattern_stats");
      } finally {
        this.state.patternStatsIo = null;
      }
      const v = new Float32Array(this.mem, out, 2 * n);
      const mean = new Float32Array(n), std = new Float32Array(n);
      for (let i = 0; i < n; i++) { mean[i] = v[2 * i]; std[i] = v[2 * i + 1]; }
      return { mean, std, io };
    }));
  }

  // ── phases ──

  /** Add a candidate phase from a master pattern Blob (ebsdsim .npz or EMsoft .h5); returns its index. */
  addPhase(blob) {
    return this._serial(() => this._frame(async (f) => {
      const [p, n] = f.str(this.addFile(blob));
      return this._check(await this.x.ze_add_phase(this.e, p, n), "add_phase");
    }));
  }

  clearPhases() {
    if (this._disposed) throw new Error("zebsdsolve: engine disposed");
    this.x.ze_clear_phases(this.e);
  }

  // ── master patterns simulated (src/simulate.zig) ──

  /** Set the crystal to simulate from a CIF's text, which the engine reads into its International Tables standard
   *  setting (throwing, with the reason, on a file that disagrees with itself). Returns crystal(). */
  readCif(text) {
    return this._frame((f) => {
      const [p, n] = f.str(text);
      this._check(this.x.ze_sim_read_cif(this.e, p, n), "sim_read_cif");
      return this.crystal();
    });
  }

  /** Set the crystal to simulate, already in its standard setting: { space_group, cell: [a, b, c, α, β, γ] (Å,
   *  degrees), sites: [{ z or symbol, xyz, occupancy, b_iso (Å²) }] }. Returns crystal(). */
  setCrystal(crystal) {
    return this._frame((f) => {
      const [p, n] = f.str(JSON.stringify(crystal));
      this._check(this.x.ze_sim_set_crystal(this.e, p, n), "sim_set_crystal");
      return this.crystal();
    });
  }

  /** Change the crystal set: `cell` [a, b, c, α, β, γ] (Å, degrees; what its family ties follows the free
   *  parameters), `sites` one { occupancy, b_iso (Å²) } a site in order (either left out to keep it). How it was
   *  read is kept. Returns crystal(). */
  editCrystal({ cell, sites } = {}) {
    return this._frame((f) => {
      const [p, n] = f.str(JSON.stringify({ ...(cell ? { cell } : {}), ...(sites ? { sites } : {}) }));
      this._check(this.x.ze_sim_edit_crystal(this.e, p, n), "sim_edit_crystal");
      return this.crystal();
    });
  }

  /** The crystal set: { space_group, name (short H-M), point_group (as the master is folded by), centring,
   *  centrosymmetric, cell (Å, degrees), free (which of its six parameters are free), volume_nm3, density (g/cm³), average_z, average_a, sites: [{ symbol, z, xyz,
   *  occupancy, b_iso (Å²), multiplicity }], reading (from a CIF: symmetry_provenance, setting, setting_note,
   *  declared, cell_as_read, sites_as_read, sites_given_default_b, …; else null) }. */
  crystal() {
    return this._frame((f) => {
      const n = Number(this.x.ze_sim_crystal(this.e, 0, 0));
      this._check(n < 0 ? -1 : 0, "sim_crystal");
      const p = f.alloc(n);
      this.x.ze_sim_crystal(this.e, p, n);
      return JSON.parse(utf8.decode(new Uint8Array(this.mem, p, n)));
    });
  }

  /** A JSON export's text, parsed: `fn(e, out, cap)` returns the length (and copies when cap suffices) or −1. */
  _json(fn, name) {
    return this._frame((f) => {
      const n = Number(fn(this.e, 0, 0));
      this._check(n < 0 ? -1 : 0, name);
      const p = f.alloc(n);
      fn(this.e, p, n);
      return JSON.parse(utf8.decode(new Uint8Array(this.mem, p, n)));
    });
  }

  /** The crystal's cell as the app draws it: { basis (the lattice vectors a, b, c as rows, Å, Cartesian, a along
   *  x), centre (the middle of the cell, Å), radius (Å: the picture's shorter side is twice this at zoom 1), atoms
   *  (how many are drawn), sites: [{ symbol, z, colour, ball (radius, Å) }] }. */
  cellInfo(cells = 1) { return this._json((e, p, n) => this.x.ze_cell_info(e, cells, p, n), "cell_info"); }

  /** The crystal's cell in a picture, rendered by the engine: `rot` the view's rotation into the crystal's
   *  Cartesian frame (9 numbers, row-major: x right, y up, z toward the viewer), `zoom` 1 fitting the cell to the
   *  shorter side, `highlight` what to pick out ("a", "b", "c", "alpha", "beta", "gamma", a site's index, or
   *  nothing). Returns the `width` × `height` RGBA pixels as a Uint8ClampedArray. */
  cellView({ rot, zoom = 1, highlight = null, cells = 1, width, height = width }) {
    const hl = typeof highlight === "number" ? 16 + highlight : Math.max(0, ["a", "b", "c", "alpha", "beta", "gamma"].indexOf(highlight) + 1);
    return this._serial(() => this._frame(async (f) => {
      const n = width * height;
      const p = f.alloc(n * 4);
      this._check(await this.x.ze_cell_view(this.e, f.f32(rot), zoom, hl, cells, width, height, p, n), "cell_view");
      return new Uint8ClampedArray(this.mem, p, n * 4).slice();
    }));
  }

  /** How the energy model spreads the backscattered electrons over energy and depth, for the crystal under the
   *  settings sim_voltage_kv, sim_tilt_deg and sim_bin_kev: { beam_kv, bin_kev, tilt_deg, bins: [{ kv, weight,
   *  beta (nm⁻¹), amplitude }] (least loss first), mean_loss_kev, mean_depth_nm, depth90_nm, depth_nm (the depths
   *  sampled), depth (all bins' profile there, largest 1), depth_by_bin (each bin's, on that scale) }. */
  energyModel() { return this._json(this.x.ze_sim_energy, "sim_energy"); }

  /** Simulate the crystal's master pattern under the settings sim_* (progress events stage "simulate"). The engine
   *  keeps it for simulatedSheets, addSimulatedPhase and simulationNpz. Resolves to { nK (directions), nSites, nG
   *  (reflections), nBins and nBinsRun (energy bins given and solved), maxBeams, failed, stopped, halfw, rank,
   *  lastChange }. */
  simulateMaster() {
    return this._serial(() => this._frame(async (f) => {
      const o = f.alloc(SIM_SUMMARY_SIZE);
      this._check(await this.x.ze_sim_run(this.e, o), "sim_run");
      return this._read(SIM_SUMMARY, o);
    }));
  }

  /** The last simulation's master: { side, sheets: Float32Array(2 · side²) } (north, then south), normalized. */
  simulatedSheets() {
    return this._frame((f) => {
      const side = this.x.ze_sim_sheets(this.e, 0, 0);
      this._check(side < 0 ? -1 : 0, "sim_sheets");
      const n = 2 * side * side;
      const p = f.alloc(n * 4);
      this.x.ze_sim_sheets(this.e, p, n);
      return { side, sheets: new Float32Array(this.mem, p, n).slice() };
    });
  }

  /** Add the last simulation as a candidate phase named `name`; returns its index. */
  addSimulatedPhase(name) {
    return this._frame((f) => {
      const [p, n] = f.str(name);
      return this._check(this.x.ze_sim_add_phase(this.e, p, n), "sim_add_phase");
    });
  }

  /** The last simulation as an .npz file in ebsdsim's layout (Uint8Array). */
  simulationNpz() {
    return this._frame((f) => {
      const n = Number(this.x.ze_sim_npz(this.e, 0, 0));
      this._check(n < 0 ? -1 : 0, "sim_npz");
      const p = f.alloc(n);
      this.x.ze_sim_npz(this.e, p, n);
      return new Uint8Array(this.mem, p, n).slice();
    });
  }

  /** The candidate phases: [{ name, side, pointGroup, spaceGroup, laue, nFold, mirror, inversion, voltageKv }]. */
  get phases() {
    return this._frame((f) => {
      const out = [];
      const p = f.alloc(PHASE_SIZE);
      const n = this.x.ze_phase_count(this.e);
      for (let i = 0; i < n; i++) {
        this._check(this.x.ze_phase_info(this.e, i, p), "phase_info");
        const o = this._read(PHASE, p);
        const name = new Uint8Array(this.mem, p + 40, 64);
        // the token's "@id:" prefix is the adapter's, not the file's name
        o.name = utf8.decode(name.subarray(0, name.indexOf(0))).replace(/^@\d+:/, "");
        o.laue = LAUE[o.laueId] || "?";
        o.pointGroupName = POINT_GROUPS[o.pointGroup] || "?";
        out.push(o);
      }
      return out;
    });
  }

  /** The phase's master as the search compares patterns with it (smoothed, and filtered as the
   *  patterns are): { side, data: Float32Array(2·side²) }, north then south. */
  masterImage(phase = 0) {
    return this._serial(() => this._frame(async (f) => {
      const side = this.phases[phase].side;
      const n = 2 * side * side;
      const p = f.alloc(n * 4);
      this._check(await this.x.ze_master_image(this.e, phase, p, n), "master_image");
      return { side, data: new Float32Array(this.mem, p, n).slice() };
    }));
  }

  /** A master in a picture, rendered by the engine: `source` a phase index, or −1 for the
   *  last simulation; `stored`, a phase's master as loaded (else as patterns are compared with
   *  it); `bin` and `site`, the simulation's one energy bin and one site alone (null: all);
   *  `projection` "globe", "lambert" (the two squares), "stereographic" (the two discs) or
   *  "joint" (the whole sphere in one square); `rot` the view's rotation into the crystal's
   *  frame (9 numbers, row-major: x right, y up, z toward the viewer); `zoom` 1 fits the
   *  projection to the picture and scales the layout about its middle; `pan` moves it (across and
   *  down, in shorter sides of the picture); `shown`, for the Lambert and stereographic views,
   *  which hemispheres: 0 both, 1 the upper, 2 the lower; `difference`, the signed difference of
   *  the upper hemisphere and the lower one's inversion, I(d) − I(−d), in any projection. Returns the `width` × `height` RGBA pixels as a
   *  Uint8ClampedArray (ImageData's layout). */
  sphereView({ source, stored = false, bin = null, site = null, projection = "globe", rot, pan = [0, 0], shown = 0, difference = false, zoom = 1, width, height = width }) {
    const kind = PROJECTIONS.indexOf(projection);
    if (kind < 0) throw new Error(`sphereView: unknown projection ${projection}`);
    return this._serial(() => this._frame(async (f) => {
      const n = width * height;
      const p = f.alloc(n * 4);
      this._check(await this.x.ze_sphere_view(this.e, source, stored ? 1 : 0, bin ?? -1, site ?? -1, kind, f.f32([...rot, pan[0], pan[1], shown, difference ? 1 : 0]), zoom, width, height, p, n), "sphere_view");
      return new Uint8ClampedArray(this.mem, p, n * 4).slice();
    }));
  }

  /** Simulated patterns of a phase: cells[k] with orientation quats[4k..4k+4] (w, x, y, z), at
   *  the reduced size or (full) the stored one; Float32Array(n·h·w). */
  /** One cell indexed and simulated under another geometry than the one in use, which stays as
   *  it is (two pattern centres side by side): `geometry` holds the fields to change, as for
   *  setGeometry. The spherical correlation at `bandwidth`, then the refinement; returns the best
   *  phase's { phase, q, ncc, nmi, sim }. */
  cellUnder(cell, geometry, bandwidth, full = false) {
    return this._serial(() => this._frame(async (f) => {
      const [h, w] = this.shape(full);
      const nPh = this.phases.length;
      const g = f.alloc(GEOMETRY_SIZE);
      this.x.ze_get_geometry(this.e, g);
      const v = new DataView(this.mem);
      for (const [k, t, off] of GEOMETRY) {
        if (!(k in geometry)) continue;
        const many = /^f64x(\d+)$/.exec(t);
        if (many) for (let i = 0; i < +many[1]; i++) v.setFloat64(g + off + 8 * i, +geometry[k][i], true);
        else if (t === "u32") v.setUint32(g + off, +geometry[k], true);
        else v.setFloat64(g + off, +geometry[k], true);
      }
      const recs = f.alloc(nPh * 8 * 4), sim = f.alloc(h * w * 4);
      const p = this._check(await this.x.ze_cell_under(this.e, BigInt(cell), g, bandwidth, full ? 1 : 0, recs, nPh * 8, sim, h * w), "cell under");
      const r = new Float32Array(this.mem, recs + 32 * p, 8);
      return { phase: p, q: Array.from(r.subarray(0, 4)), ncc: r[4], nmi: r[5], sim: new Float32Array(this.mem, sim, h * w).slice() };
    }));
  }

  simulate(phase, cells, quats, full = false) {
    return this._serial(() => this._frame(async (f) => {
      const [h, w] = this.shape(full);
      const n = cells.length;
      const p = f.alloc(n * h * w * 4);
      this._check(await this.x.ze_simulate(this.e, phase, f.u64(cells), f.f32(quats), n, full ? 1 : 0, p, n * h * w), "simulate");
      return new Float32Array(this.mem, p, n * h * w).slice();
    }));
  }

  // ── fits ──

  /** Fit the pattern centre by the voting search; the geometry takes the result. */
  fitPc() {
    return this._serial(() => this._frame(async (f) => {
      const p = f.alloc(PC_RESULT_SIZE);
      this._check(await this.x.ze_fit_pc(this.e, p), "fit_pc");
      return this._read(PC_RESULT, p);
    }));
  }

  /** The last search level's votes: { shape: [nz, ny, nx], x, y, z, scores } (scores z-major). */
  pcScores() {
    return this._frame((f) => {
      const sp = f.alloc(12);
      const n = this.x.ze_pc_scores(this.e, sp, 0, 0, 0, 0);
      const [nz, ny, nx] = new Uint32Array(this.mem, sp, 3);
      if (n <= 0) return null;
      const ap = f.alloc((nx + ny + nz) * 8), cp = f.alloc(n * 4);
      this.x.ze_pc_scores(this.e, sp, ap, nx + ny + nz, cp, n);
      const axes = new Float64Array(this.mem, ap, nx + ny + nz);
      return { shape: [nz, ny, nx], x: Array.from(axes.subarray(0, nx)), y: Array.from(axes.subarray(nx, nx + ny)), z: Array.from(axes.subarray(nx + ny)),
        scores: new Float32Array(this.mem, cp, n).slice() };
    });
  }

  /** Every grid the last pattern-centre search scored, in order: [{ shape: [nz, ny, nx], x, y,
   *  z, scores (the fit patterns' mean score per candidate, z-major), winner (index into scores:
   *  the candidate nearest the grid's choice), walked (the grid then moved onto a winner at its
   *  border), patBest (per fit pattern, the candidate it scored highest under), patNcc, and,
   *  when ballots decided (the pc_vote setting), density (of the ballots, per candidate), peak
   *  (the choice in candidate steps along x, y, z), ballots and inliers (counts) }]. */
  pcLevels() {
    return this._frame((f) => {
      const out = [];
      const count = this.x.ze_pc_level_count(this.e);
      const nPat = Math.max(1, this.x.ze_fit_patterns(this.e, 0, 0, 0, 0, 0));
      const sp = f.alloc(12), ip = f.alloc(8);
      for (let i = 0; i < count; i++) {
        const n = this.x.ze_pc_level(this.e, i, sp, ip, 0, 0, 0, 0, 0, 0, 0);
        const [nz, ny, nx] = new Uint32Array(this.mem, sp, 3);
        const ap = f.alloc((nx + ny + nz) * 8), cp = f.alloc(n * 4), bp = f.alloc(nPat * 4), np = f.alloc(nPat * 4);
        this.x.ze_pc_level(this.e, i, sp, ip, ap, nx + ny + nz, cp, n, bp, np, nPat);
        const axes = new Float64Array(this.mem, ap, nx + ny + nz);
        const info = new Uint32Array(this.mem, ip, 2);
        const fp = f.alloc(40), dn = this.x.ze_pc_level_ballots(this.e, i, 0, 0, fp);
        const dp = f.alloc(Math.max(dn, 1) * 4);
        this.x.ze_pc_level_ballots(this.e, i, dp, dn, fp);
        const bal = new Float64Array(this.mem, fp, 5);
        out.push({ shape: [nz, ny, nx], x: Array.from(axes.subarray(0, nx)), y: Array.from(axes.subarray(nx, nx + ny)), z: Array.from(axes.subarray(nx + ny)),
          density: dn > 0 ? new Float32Array(this.mem, dp, dn).slice() : null, peak: [bal[0], bal[1], bal[2]], ballots: bal[3], inliers: bal[4],
          scores: new Float32Array(this.mem, cp, n).slice(), winner: info[0], walked: !!info[1],
          patBest: Array.from(new Uint32Array(this.mem, bp, nPat)), patNcc: new Float32Array(this.mem, np, nPat).slice() });
      }
      return out;
    });
  }

  /**
   * Fit the plane of source points: where the beam stood for every map cell, as the detector sees it (Winkelmann
   * et al. 2020; kikuchipy's fit_pc). Each fit pattern (settings plane_stride, plane_patterns_max) is fitted by
   * itself, its orientation and its own source point; then the affine or projective transformation of the scan's
   * indices through those points (plane_kind) by weighted least squares (plane_weights, plane_outlier). The geometry
   * takes the plane (its `planeKind`, `plane`: 4 × 3 row-major, plane · (column, row, 1) = w · (PCx, PCy, PCz, 1)), and
   * the scan's steps and the pixel size then say nothing of where the beam stood. Resolves to { plane, kind
   * (PLANE_KINDS), frame (0 nominal, 1 following the plane), pc, pcSe, rot, saidRot and saidRotSe (the sample's frame
   * by the plane against the nominal one: swing, tilt, the scan's turn in its plane; radians), step and stepSe (the
   * beam's travel for a row and a column against what the steps and pixel size say), skew, skewSe, rmsd (PCx, PCy,
   * PCz), ncc, startNcc, nPatterns, nUsed, iterations, blocks }.
   */
  fitPlane() {
    return this._serial(() => this._frame(async (f) => {
      const p = f.alloc(PLANE_RESULT_SIZE);
      this._check(await this.x.ze_fit_plane(this.e, p), "fit_plane");
      return this._read(PLANE_RESULT, p);
    }));
  }

  /** The automatic fit, from the geometry as it was given: the vote for the pattern centre (fitPc), then the plane of
   *  source points from there (fitPlane). Resolves to what fitPlane does; its startPc is the vote's pattern centre. */
  fitGeometry() {
    return this._serial(() => this._frame(async (f) => {
      const p = f.alloc(PLANE_RESULT_SIZE);
      this._check(await this.x.ze_fit_auto(this.e, p), "fit_geometry");
      return this._read(PLANE_RESULT, p);
    }));
  }

  /** The source points the last plane fit measured, a fit pattern each: { cells (Float64Array), pc and fit
   *  (Float64Array, 3 a point: the pattern's own, and the plane's), ncc, used (Uint8Array) }. */
  planePoints() {
    return this._frame((f) => {
      const n = this.x.ze_plane_points(this.e, 0, 0);
      const p = f.alloc(Math.max(n, 1) * PLANE_POINT_SIZE);
      this.x.ze_plane_points(this.e, p, n);
      const v = new DataView(this.mem);
      const out = { cells: new Float64Array(n), pc: new Float64Array(3 * n), fit: new Float64Array(3 * n), ncc: new Float64Array(n), used: new Uint8Array(n) };
      for (let b = 0; b < n; b++) {
        const a = p + b * PLANE_POINT_SIZE;
        out.cells[b] = Number(v.getBigUint64(a, true));
        for (let k = 0; k < 3; k++) { out.pc[3 * b + k] = v.getFloat64(a + 8 + 8 * k, true); out.fit[3 * b + k] = v.getFloat64(a + 32 + 8 * k, true); }
        out.ncc[b] = v.getFloat64(a + 56, true);
        out.used[b] = v.getUint32(a + 64, true);
      }
      return out;
    });
  }

  /** The patterns of the last fit: { cells, phases, quats (4 per pattern), ncc }. */
  fitPatterns() {
    return this._frame((f) => {
      const n = this.x.ze_fit_patterns(this.e, 0, 0, 0, 0, 0);
      if (n <= 0) return { cells: [], phases: new Int32Array(0), quats: new Float32Array(0), ncc: new Float32Array(0) };
      const cp = f.alloc(n * 8), pp = f.alloc(n * 4), qp = f.alloc(n * 16), sp = f.alloc(n * 4);
      this.x.ze_fit_patterns(this.e, cp, pp, qp, sp, n);
      return { cells: Array.from(new BigUint64Array(this.mem, cp, n), Number), phases: new Int32Array(this.mem, pp, n).slice(),
        quats: new Float32Array(this.mem, qp, 4 * n).slice(), ncc: new Float32Array(this.mem, sp, n).slice() };
    });
  }

  // ── indexing ──

  /** What normalized mutual information chance gives on this scan for a phase: { median,
   *  spread }, measured once on a sample of the region's patterns against simulations at
   *  random orientations, the same for every pattern. z in the results is the NMI on that
   *  scale; the accept_nmi setting, above 0, is an absolute threshold in its place. */
  chance(phase = 0) {
    return this._serial(() => this._frame(async (f) => {
      const p = f.alloc(8);
      this._check(await this.x.ze_chance(this.e, phase, p), "chance");
      const v = new Float32Array(this.mem, p, 2);
      return { median: v[0], spread: v[1] };
    }));
  }

  /** Start indexing the region of interest (settings roi_*; the whole map by default). */
  indexBegin() {
    return this._serial(async () => { this._check(await this.x.ze_index_begin(this.e), "index"); });
  }

  /** Where indexing stands: { sweep, sweeps, bandwidth, dictGrid, done, total (patterns of this
   *  sweep), region (cells of the region), accepted }. */
  get indexStatus() {
    return this._frame((f) => {
      const p = f.alloc(INDEX_STATUS_SIZE);
      this.x.ze_index_status(this.e, p);
      const s = this._read(INDEX_STATUS, p);
      s.method = METHODS[s.methodId];
      return s;
    });
  }

  /** What each sweep searches with, in order: [{ method, bandwidth, dictGrid }]. */
  get sweepPlan() {
    return this._frame((f) => {
      const p = f.alloc(12);
      const out = [];
      for (let i = 0; i < this.settings.sweeps; i++) {
        this.x.ze_sweep(this.e, i, p);
        const v = new Uint32Array(this.mem, p, 3);
        out.push({ method: METHODS[v[2]], bandwidth: v[0], dictGrid: v[1] });
      }
      return out;
    });
  }

  /** Index the next batch of the sweep in progress: the status, with `left` cells still to visit (0: done). */
  indexStep() {
    return this._serial(async () => {
      const left = this._check(await this.x.ze_index_step(this.e), "index");
      return { left, ...this.indexStatus };
    });
  }

  /** Index the region, calling progress(status) after each batch (return false to stop). */
  async index(progress = null) {
    await this.indexBegin();
    for (;;) {
      const st = await this.indexStep();
      if ((progress && progress(st) === false) || st.left === 0) return st;
    }
  }

  /** Fit how the work is cut up (settings batch, refine_pixels, wg_refine, wg_fft) to this
   *  machine and this scan: a sample of the region, sized by the bandwidth, is indexed and
   *  timed here, and each setting in turn climbs from its value (doubled while that is faster,
   *  else halved while that is). About ten trials. Nothing stored is touched.
   *  progress(callsDone) may return false to stop (the best so far stays). Resolves to
   *  { before, after } of the four settings. */
  tune(progress = null) {
    return this._serial(async () => {
      const pick = (s) => ({ batch: s.batch, refine_pixels: s.refine_pixels, wg_refine: s.wg_refine, wg_fft: s.wg_fft });
      const before = pick(this.settings);
      this._check(this.x.ze_tune_begin(this.e), "tune");
      const lostTrialError = () => {
        // The trial has settled: consume its native failure before a later CPU export can
        // mistake that stale error for its own. Keep the trial diagnostic with the loss.
        const detail = this._error();
        const error = this._deviceLossError();
        if (detail) error.message += `; ${detail}`;
        return error;
      };
      for (let done = 1; ; done++) {
        const t0 = performance.now();
        let rc;
        try { rc = await this.x.ze_tune_trial(this.e); }
        catch (error) {
          if (!this.deviceLost) throw error;
          const fatal = lostTrialError();
          fatal.cause = error;
          throw fatal;
        }
        if (this.deviceLost) throw lostTrialError();
        let seconds = (performance.now() - t0) / 1000;
        if (rc !== 0) {
          // a value this machine cannot run just loses
          const message = this._error();
          if (/Cancelled/.test(message)) throw new Error(`tune: ${message}`);
          seconds = -1;
        }
        if (this.deviceLost) throw lostTrialError();
        const more = this.x.ze_tune_report(this.e, seconds);
        if (!more || (progress && progress(done) === false)) break;
      }
      return { before, after: pick(this.settings) };
    });
  }

  /** Index given cells without storing: per phase and cell, 8 floats (w, x, y, z, NCC, NMI, the
   *  last improving refinement step in radians, and the NMI's distance above chance in standard
   *  deviations). The search is the settings' unless `search` gives one:
   *  { method, bandwidth, dictGrid }, each defaulting to its setting. */
  indexCells(cells, search = null) {
    return this._serial(() => this._frame(async (f) => {
      const n = cells.length, nPh = this.x.ze_phase_count(this.e);
      const p = f.alloc(nPh * n * 32);
      if (search) {
        const s = this.settings;
        this._check(await this.x.ze_index_cells_by(this.e, f.u64(cells), n, METHODS.indexOf(search.method ?? s.method), search.bandwidth ?? s.bandwidth, search.dictGrid ?? s.dict_grid, p, nPh * n * 8), "index_cells");
      } else this._check(await this.x.ze_index_cells(this.e, f.u64(cells), n, p, nPh * n * 8), "index_cells");
      return new Float32Array(this.mem, p, nPh * n * 8).slice();
    }));
  }

  /** Candidate overlap fit; flat arrays use [cell, discovery slot, quaternion].
   * Stored primary mode requires the exact current primary context. Missing cells
   * return stop_reason=6. Caller commits a batch only after successful resolution. */
  overlapCells(cells, {maxComponents=2, primary="search",fitBlurGain=false,gainDomain="contrast",keepN=1,polish=false,minResidualNcc=null,minStageImprovement=null,polishRadiusDeg=1}={}) {
    if(typeof polish!=="boolean")throw new RangeError("polish must be a boolean");
    if(typeof polishRadiusDeg!=="number"||!Number.isFinite(polishRadiusDeg)||polishRadiusDeg<=0||polishRadiusDeg>5)throw new RangeError("polishRadiusDeg must be >0 and <=5 degrees");
    if(minResidualNcc!==null&&(typeof minResidualNcc!=="number"||!Number.isFinite(minResidualNcc)||minResidualNcc< -1||minResidualNcc>1))throw new RangeError("minResidualNcc must be null or finite in [-1,1]");
    const staged = minStageImprovement !== null;
    let requestedStageImprovement = null, stageMinimum = null;
    if (staged) {
      if (minResidualNcc === null) throw new RangeError("Stage improvement requires residual NCC screening");
      if (![2, 3, 4].includes(maxComponents) || !Array.isArray(minStageImprovement) || minStageImprovement.length !== 3 ||
          Array.from(minStageImprovement).some((v, i) => i >= maxComponents - 1 ? v !== null : !Number.isFinite(v) || v < 0 || v > 1)) {
        throw new RangeError("Enter a ΔSSE minimum in [0, 1] for each requested addition, with null for unused stages");
      }
      if (this.x.ze_overlap_stage_gate_version?.() !== 1) throw new Error("Stage improvement requires a current WASM overlap module");
      requestedStageImprovement = [...minStageImprovement];
      stageMinimum = requestedStageImprovement.map(v => v === null ? null : Math.fround(v));
    }
    const requestedMinResidualNcc=minResidualNcc;
    if(minResidualNcc!==null)minResidualNcc=Math.fround(minResidualNcc);
    polishRadiusDeg=Math.fround(polishRadiusDeg);
    const profile=polish||minResidualNcc!==null;
    let selectionVersion = 1;
    if(profile){
      if(gainDomain!=="contrast")throw new RangeError("overlap profiling requires contrast gain");
      if(!this.x.ze_overlap_cells_v6||!this.x.ze_overlap_profile_version||!this.x.ze_overlap_profile_trace_capacity)throw new Error("overlap profiling requires WASM V6 support; load a current module");
      if(this.x.ze_overlap_profile_version()!==1)throw new Error("unsupported overlap profile version");
      selectionVersion = this.x.ze_overlap_profile_selection_version?.() ?? 1;
      if (![1, 2].includes(selectionVersion)) throw new Error("unsupported overlap selection version");
    }
    if(!["contrast","source_dc"].includes(gainDomain))throw new RangeError("gainDomain must be contrast or source_dc");
    if(!Number.isInteger(keepN)||keepN<1||keepN>8)throw new RangeError("keepN must be an integer from 1 through 8");
    if(keepN>1&&maxComponents>1){
      if(gainDomain!=="contrast")throw new RangeError("keepN > 1 requires contrast gain");
      if(this.settings.method==="pca")throw new RangeError("keepN > 1 does not support PCA proposals");
      if(!this.x.ze_overlap_cells_v5)throw new Error("keepN > 1 requires WASM V5 overlap support; load a current module");
    }
    const contrast=gainDomain==="contrast";
    if(contrast&&!this.x.ze_overlap_cells_v4)throw new Error("contrast gain requires WASM V4 overlap support; load a current module or explicitly use gainDomain:'source_dc'");
    if(!Number.isInteger(maxComponents)||maxComponents<1||maxComponents>4)throw new RangeError("maxComponents must be 1..4");
    if(!["search","stored"].includes(primary))throw new RangeError("primary must be search or stored");
    cells=Array.from(cells,Number);
    if(cells.some(c=>!Number.isSafeInteger(c)||c<0))throw new RangeError("cells must be nonnegative safe integers");
    return this._serial(()=>this._frame(async f=>{
      const n=cells.length,k=maxComponents,cp=f.alloc(n*k*32),sp=f.alloc(n*64),mp=f.alloc(n*32);
      const tp=fitBlurGain||contrast?f.alloc(n*7*4):0,np=contrast?f.alloc(n*k*4):0;
      const wp=profile||keepN>1&&k>1?f.alloc(n*k*4):0;
      let profileEvents=[],sourceQ=null;
      const trajectoryVersion=keepN>1&&k>=3&&this.x.ze_overlap_trajectory_version?this.x.ze_overlap_trajectory_version():0;
      if(![0,1].includes(trajectoryVersion))throw new Error(`unsupported overlap trajectory version ${trajectoryVersion}`);
      if(profile){
        const capacity=this.x.ze_overlap_profile_trace_capacity(this.e,n,k,keepN);
        if(n&&(!Number.isSafeInteger(capacity)||capacity<=0))throw new Error("invalid V6 overlap trace capacity");
        const optionsSize = staged ? 28 : 16;
        const options=f.alloc(optionsSize),ov=new DataView(this.mem,options,optionsSize);
        ov.setUint32(0,optionsSize,true);ov.setUint32(4,(polish?1:0)|(minResidualNcc!==null?2:0)|(selectionVersion===2?4:0),true);
        ov.setFloat32(8,polishRadiusDeg,true);ov.setFloat32(12,minResidualNcc??0,true);
        if (staged) stageMinimum.forEach((v, i) => ov.setFloat32(16+4*i,v??0,true));
        const qp=f.alloc(n*k*16),ep=f.alloc(capacity*64),countp=f.alloc(4);
        this._check(await this.x.ze_overlap_cells_v6(this.e,f.u64(cells),n,k,primary==="stored"?1:0,fitBlurGain?1:0,keepN,cp,n*k,sp,n,mp,n,tp,n,np,n*k,wp,n*k,options,qp,n*k,ep,capacity,countp),"overlap_cells_v6");
        const count=new DataView(this.mem,countp,4).getUint32(0,true);
        if(count>capacity)throw new Error("V6 overlap trace exceeds allocated capacity");
        sourceQ=new Float32Array(this.mem,qp,n*k*4).slice();
        profileEvents=decodeOverlapProfileEvents(new Uint8Array(this.mem,ep,count*64),cells,selectionVersion,staged);
      }else if(contrast&&keepN>1&&k>1)this._check(await this.x.ze_overlap_cells_v5(this.e,f.u64(cells),n,k,primary==="stored"?1:0,fitBlurGain?1:0,keepN,cp,n*k,sp,n,mp,n,tp,n,np,n*k,wp,n*k),"overlap_cells_v5");
      else if(contrast)this._check(await this.x.ze_overlap_cells_v4(this.e,f.u64(cells),n,k,primary==="stored"?1:0,fitBlurGain?1:0,cp,n*k,sp,n,mp,n,tp,n,np,n*k),"overlap_cells_v4");
      else if(fitBlurGain)this._check(await this.x.ze_overlap_cells_v3(this.e,f.u64(cells),n,k,primary==="stored"?1:0,cp,n*k,sp,n,mp,n,tp,n),"overlap_cells");
      else this._check(await this.x.ze_overlap_cells_v2(this.e,f.u64(cells),n,k,primary==="stored"?1:0,cp,n*k,sp,n,mp,n),"overlap_cells");
      const out=decodeOverlapArrays(new Uint8Array(this.mem,cp,n*k*32).slice(),new Float32Array(this.mem,sp,n*16).slice(),new Uint8Array(this.mem,mp,n*32).slice(),cells,k);
      out.winner_rank=wp?new Int32Array(this.mem,wp,n*k).slice():Int32Array.from(out.phase,(phase,i)=>phase<0?-1:i%k===0?0:1);
      Object.assign(out,{keep_n:keepN,candidate_method:this.settings.method,
        candidate_seed_policy:k===1?"ordinary_primary_only_v1":keepN===1?"native_single_winner_v1":this.settings.method==="spherical"?"spherical_localmax26_pole_quotient_2_grid_steps_v1":"dictionary_raw_top_n_v1"});
      Object.assign(out,overlapTrajectoryMetadata(out,new Uint8Array(this.mem,mp,n*32),keepN,k,trajectoryVersion));
      if(profile){
        const pv=new DataView(this.mem,mp,n*32);
        Object.assign(out,{profile_version:1,selection_version:selectionVersion,orientation_polish:polish,polish_radius_deg:polishRadiusDeg,min_residual_ncc:minResidualNcc,
          source_quaternions:sourceQ,profile_events:profileEvents,requested_min_residual_ncc:requestedMinResidualNcc,profile_precision:"f32",
          component_axis:"selected support order; source poses and events retain proposal provenance",
          trajectory_selection:selectionVersion===2?"profiled_achieved_supports_v2":"profiled_complete_trajectories_v1",
          trajectory_source:Int32Array.from({length:n},(_,b)=>{const p=pv.getUint32(b*32+28,true);if(![0,1,0xffffffff].includes(p))throw Error("invalid V6 overlap trajectory provenance");return p===0xffffffff?-1:p;}),
          residual_screen:{enabled:minResidualNcc!==null,minimum:minResidualNcc,comparison:"finite inclusive >= before fitting each proposal",confidence_calibrated:false}});
        if (staged) out.stage_gate = {
          version:1, minimum:stageMinimum, requested_minimum:requestedStageImprovement,
          orders:[2,3,4], comparison:"(strongest lower-order SSE in the same branch and round minus best joint SSE) divided by original prepared-pattern energy",
          joint_candidate:"positive addition only; residual NCC screened first",
          confidence_calibrated:false,
        };
      }
      const rp=f.alloc(16),hasROI=primary==="stored"&&this.x.ze_seed_source_roi(this.e,rp);
      const seed_source_roi=hasROI?Array.from(new Uint32Array(this.mem,rp,4)):null;
      const metadata=overlapBlurGainMetadata(fitBlurGain,fitBlurGain?new Float32Array(this.mem,tp,n*7):null,this.x);
      Object.assign(metadata,{fit_model:contrast?"contrast_gain_v1":"source_dc_gain_v1",gain_domain:gainDomain});
      if(contrast){
        Object.assign(out,metadata);
        Object.assign(metadata,contrastBasisMetadata(out,new Float32Array(this.mem,np,n*k),new Float32Array(this.mem,tp,n*7)));
        const settings=this.settings;
        metadata.contrast_basis_preparation=Object.fromEntries(["master_smooth","clahe","bandpass"].map(key=>[key,settings[key]]));
      }
      if(fitBlurGain||contrast){
        const phases=this.phases.length,moments=f.alloc(phases*5*8);
        this._check(this.x.ze_overlap_phase_moments(this.e,moments,phases*5),"overlap_phase_moments");
        const values=new Float64Array(this.mem,moments,phases*5),rows=Array.from({length:phases},(_,i)=>Array.from(values.slice(i*5,(i+1)*5)));
        if(contrast){
          const provenance={phase_moment_names:["source_mean","source_sd","bake_mean","bake_sd"],phase_moments:rows.map(row=>row.slice(0,4)),source_dc_restored:false};
          Object.assign(metadata.contrast_basis_preparation,provenance);
          if(fitBlurGain)Object.assign(metadata.blur_gain_optimizer,provenance);
        }else Object.assign(metadata.blur_gain_optimizer,{phase_moment_names:["source_mean","source_sd","bake_mean","bake_sd","effective_dc"],phase_moments:rows,phase_dc:rows.map(row=>row[4]),dc_units:"baked_master_standard_deviations"});
      }
      if(profile){
        Object.assign(metadata,{active_count_definition:"strictly_positive_compacted_support_size",fitted_count_definition:"amplitude_above_1e-6_display_only",
          profile_optimizer:{version:1,enabled:polish,method:"bounded_crystal_frame_axis_poll_variable_projection",geometry:"fixed",trust_centre:"immutable_native_source_pose",radius_deg:polishRadiusDeg,
            singleton_starts_x_radius_factors:[0,.5,-.5],joint_starts:1,angular_step_radius_factors:[.5,.25,.125],max_sweeps_per_scale:2,
            nuisance_schedule:"full_at_entry_and_after_final_pose_change; short_after_moving_scale",short_nuisance_max_evaluations:15,max_replacements_per_attempted_order:2,
            best_single_scope:"strongest_achieved_across_eligible_supports_and_both_complete_histories"}});
        if(fitBlurGain){const optimizer=metadata.blur_gain_optimizer;
          optimizer.max_evaluations_per_inner_joint_search=optimizer.max_evaluations;delete optimizer.max_evaluations;
          optimizer.max_evaluations_per_inner_singleton_search=optimizer.singleton_max_evaluations;delete optimizer.singleton_max_evaluations;
          Object.assign(optimizer,{evaluation_bound_scope:"one_inner_search_not_total_profile",best_single:"strongest_achieved_singleton_across_candidates_and_complete_histories",
            best_single_refinement:"same_objective_bounded_pose_and_nuisance_profiles_when_enabled",joint_singleton_seeds:{k1:"separate_coarse_and_polished_theta_tables",k_gt_1:"same_achieved_member_theta_in_both_seed_tables"}});
        }
      }
      return Object.assign(out,metadata,{primary,seed_source_roi,geometry:this.geometry,settings:this.settings,phases:this.phases,oxford_detector_attitude_version:this.detectorAttitudeVersion});
    }));
  }

  /** Arbitrary-component native IPF RGBA, matching native map colors exactly. */
  orientationColors(quaternions, phase, direction="z") {
    const q=Float32Array.from(quaternions),p=Int32Array.from(phase),axis=["x","y","z"].indexOf(direction);
    if(q.length!==p.length*4||axis<0)throw new RangeError("q length must be 4*phase length and direction x/y/z");
    return this._frame(f=>{
      const pp=f.alloc(p.length*4),out=f.alloc(p.length*4);new Int32Array(this.mem,pp,p.length).set(p);
      this._check(this.x.ze_orientation_colors(this.e,f.f32(q),pp,p.length,axis,out,p.length*4),"orientation_colors");
      return new Uint8ClampedArray(this.mem,out,p.length*4).slice();
    });
  }

  /** Regenerate at CURRENT native context. The caller must reject stale results. */
  overlapInspect(cell,result,row=0,selectedComponent=0) {
    const k=result.max_components??result.maxComponents;
    if(!Number.isSafeInteger(cell)||cell<0||!Number.isInteger(row)||row<0||row>=result.cells.length||Number(result.cells[row])!==cell||!Number.isInteger(selectedComponent)||selectedComponent<0||selectedComponent>=k)throw new RangeError("inspection row/cell/component mismatch");
    return this._serial(()=>this._frame(async f=>{
      const cp=f.alloc(k*32),v=new DataView(this.mem,cp,k*32);
      for(let j=0;j<k;j++){
        const index=row*k+j;
        for(let t=0;t<4;t++)v.setFloat32(j*32+t*4,result.quaternions[index*4+t],true);
        v.setFloat32(j*32+16,result.amplitudes[index],true);v.setInt32(j*32+20,result.phase[index],true);
        v.setFloat32(j*32+24,result.proposal_ncc?.[index]??0,true);v.setFloat32(j*32+28,result.proposal_nmi?.[index]??0,true);
      }
      const st=this.settings,[h,w]=this.shape(st.binning!==1&&st.full_resolution==="all"),px=h*w,out=f.alloc(5*px*4);
      const fitModel=result.fit_model??"source_dc_gain_v1";
      if(!["contrast_gain_v1","source_dc_gain_v1"].includes(fitModel))throw new RangeError(`unsupported overlap fit_model ${fitModel}`);
      if(fitModel==="contrast_gain_v1"||result.blur_gain){
        const theta=Float32Array.from(result.blur_gain_theta.slice(row*7,(row+1)*7));
        if(theta.length!==7)throw new RangeError("expected seven frozen blur/gain parameters");
        const inspect=fitModel==="contrast_gain_v1"?this.x.ze_overlap_inspect_v4:this.x.ze_overlap_inspect_v3;
        if(!inspect)throw new Error("contrast inspection requires WASM V4 overlap support");
        this._check(await inspect(this.e,BigInt(cell),cp,k,selectedComponent,f.f32(theta),7,out,5*px),"overlap_inspect");
      }else this._check(await this.x.ze_overlap_inspect(this.e,BigInt(cell),cp,k,selectedComponent,out,5*px),"overlap_inspect");
      const data=new Float32Array(this.mem,out,5*px).slice(),answer={shape:[h,w]};
      ["measured","conditional","selected_contribution","composite","residual"].forEach((key,i)=>answer[key]=data.slice(i*px,(i+1)*px));
      return answer;
    }));
  }

  /** Row-major native overlap seed validity under current settings, geometry and phases. */
  storedPrimaryValidity(first = 0, count = this.scan.rows * this.scan.cols - first) {
    if (!Number.isInteger(first) || first < 0 || first > 0xffffffff || !Number.isInteger(count) || count < 0 || count > 0xffffffff) throw new RangeError("Invalid primary seed range");
    if (!this.x.ze_stored_primary_validity) throw new Error("Stored primary validation requires a current WASM module");
    return this._frame(f => {
      const p = f.alloc(count);
      const rc = this.x.ze_stored_primary_validity(this.e, first, p, count);
      if (rc !== 0) this._check(rc, "stored_primary_validity");
      return new Uint8Array(this.mem, p, count).slice();
    });
  }

  /** The stored results per map cell: { quaternions (4 per cell), euler (3), ncc, nmi, phase
   *  (Int32, −1: not indexed), z (the NMI's distance above chance), sweep (Int32: the sweep that
   *  accepted the cell, −1: none) }. */
  results() {
    return this._frame((f) => {
      const s = this.scan, n = s.rows * s.cols;
      const get = (kind, k, T) => {
        const p = f.alloc(n * k * 4);
        this._check(this.x.ze_results(this.e, kind, p, n * k * 4), "results");
        return new T(this.mem, p, n * k).slice();
      };
      return { quaternions: get(0, 4, Float32Array), euler: get(4, 3, Float32Array), ncc: get(1, 1, Float32Array), nmi: get(2, 1, Float32Array), phase: get(3, 1, Int32Array),
        z: get(5, 1, Float32Array), sweep: get(6, 1, Int32Array) };
    });
  }

  /** A result map as RGBA (rows · cols · 4), for an ImageData: "ipf_x" | "ipf_y" | "ipf_z" | "ncc" | "nmi" | "phase" | "z" | "sweep". */
  map(kind = "ipf_z") {
    return this._frame((f) => {
      const s = this.scan, n = s.rows * s.cols * 4;
      const p = f.alloc(n);
      this._check(this.x.ze_map(this.e, MAPS[kind], p, n), "map");
      return new Uint8ClampedArray(this.mem, p, n).slice();
    });
  }

  /** The key to a phase's inverse pole figure colours, as the settings have them (ipf_colors,
   *  ipf_lightness, ipf_gray): { rgba (size × size × 4, for an ImageData), size, box, wedge }.
   *  The crystal directions that stand for all the others, in stereographic projection with the
   *  crystal's x to the right ([001] at the origin), transparent outside them; `box` is their
   *  bounding box in the projection [x0, y0, x1, y1], drawn from the image's lower left corner at
   *  (size − 1) / its longer side pixels per unit, and `wedge` the two azimuths (degrees) of a
   *  wedge about the c axis, null for the cubic classes' polygons. `lower` asks for the part in
   *  the lower hemisphere, which only "perfect_rotation" has for the classes of a single axis:
   *  elsewhere the box is all zero. */
  ipfKey(phase = 0, size = 128, lower = false) {
    return this._frame((f) => {
      const p = f.alloc(size * size * 4), bp = f.alloc(24);
      this._check(this.x.ze_ipf_key(this.e, phase, size, lower ? 1 : 0, p, size * size * 4, bp), "ipf_key");
      const b = Array.from(new Float32Array(this.mem, bp, 6));
      return { rgba: new Uint8ClampedArray(this.mem, p, size * size * 4).slice(), size, box: b.slice(0, 4), wedge: Number.isNaN(b[4]) ? null : [b[4], b[5]] };
    });
  }

  /** The results as a text file: "ang" (EDAX) or "ctf" (HKL); a Uint8Array. */
  exportText(kind = "ang") {
    return this._frame((f) => {
      const k = kind === "ctf" ? 1 : 0;
      const n = this._check(this.x.ze_export(this.e, k, 0, 0), "export");
      const p = f.alloc(n);
      this._check(this.x.ze_export(this.e, k, p, n), "export");
      return new Uint8Array(this.mem, p, n).slice();
    });
  }

  /** Lossless internal result state, including every stored phase candidate. */
  exportResultCheckpoint() {
    return this._frame((f) => {
      const n = this._check(this.x.ze_result_checkpoint(this.e, 0, 0), "result checkpoint");
      const p = f.alloc(n);
      this._check(this.x.ze_result_checkpoint(this.e, p, n), "result checkpoint");
      return new Uint8Array(this.mem, p, n).slice();
    });
  }

  /** Phase-0 chance already measured during indexing; never recalibrates. */
  resultChance() {
    return this._frame((f) => {
      const p = f.alloc(8);
      if (!this.x.ze_result_chance(this.e, p)) return null;
      const values = new Float32Array(this.mem, p, 2);
      return { median: values[0], spread: values[1] };
    });
  }

  /** Restore only after reopening the same retained sources, masters and settings. */
  restoreResultCheckpoint(bytes) {
    if (!(bytes instanceof Uint8Array)) throw new TypeError("Expected result checkpoint bytes.");
    return this._frame((f) => {
      const p = f.alloc(bytes.byteLength);
      new Uint8Array(this.mem, p, bytes.byteLength).set(bytes);
      this._check(this.x.ze_restore_result_checkpoint(this.e, p, bytes.byteLength), "restore result checkpoint");
    });
  }

  /** Native export data. Optional active wxyz engine→Oxford CS1 frame transformation. */
  exportSnapshot(engineToCs1 = null) {
    return this._frame((f) => {
      let frame = 0;
      if (engineToCs1 !== null) {
        if (!Array.isArray(engineToCs1) || engineToCs1.length !== 4 || !engineToCs1.every(Number.isFinite)) throw new Error("Expected engine→CS1 quaternion [w,x,y,z].");
        frame = f.alloc(32);
        new Float64Array(this.mem, frame, 4).set(engineToCs1);
      }
      const n = this._check(this.x.ze_export_snapshot(this.e, frame, 0, 0), "export snapshot");
      const p = f.alloc(n);
      this._check(this.x.ze_export_snapshot(this.e, frame, p, n), "export snapshot");
      return JSON.parse(utf8.decode(new Uint8Array(this.mem, p, n)));
    });
  }

  /** Oxford CS1 results, converted natively using frozen source and crystal frames. */
  exportOxfordSnapshot(engineToCs1 = null) {
    return this._frame((f) => {
      let frame = 0;
      if (engineToCs1 !== null) {
        if (!Array.isArray(engineToCs1) || engineToCs1.length !== 4 || !engineToCs1.every(Number.isFinite)) throw new Error("Expected engine→CS1 quaternion [w,x,y,z].");
        frame = f.alloc(32);
        new Float64Array(this.mem, frame, 4).set(engineToCs1);
      }
      const n = this._check(this.x.ze_export_oxford_snapshot(this.e, frame, 0, 0), "Oxford export snapshot");
      const p = f.alloc(n);
      this._check(this.x.ze_export_oxford_snapshot(this.e, frame, p, n), "Oxford export snapshot");
      return JSON.parse(utf8.decode(new Uint8Array(this.mem, p, n)));
    });
  }

  /** Bunge angles (radians) of a pose, without crystal symmetry reduction. */
  bunge(q) {
    if (!(Array.isArray(q) || ArrayBuffer.isView(q)) || q.length !== 4)
      throw new TypeError("Expected four quaternion components.");
    return this._frame((f) => {
      const qp = f.f32(q), ep = f.alloc(24);
      this.x.ze_bunge(qp, ep);
      return Array.from(new Float64Array(this.mem, ep, 3));
    });
  }

  /** Explicit symmetry reduction: { q, euler }, the representative in the phase's
   *  fundamental zone and that one's Bunge angles (radians). */
  reduced(phase, q) {
    return this._frame((f) => {
      const qp = f.f32(q), ep = f.alloc(24);
      this.x.ze_reduce(this.e, phase, qp, ep);
      return { q: Array.from(new Float32Array(this.mem, qp, 4)), euler: Array.from(new Float64Array(this.mem, ep, 3)) };
    });
  }

  /** Smallest rotation (radians) between two quaternions under a phase's symmetry. */
  disorientation(phase, a, b) {
    return this._frame((f) => this.x.ze_disorientation(this.e, phase, f.f32(a), f.f32(b)));
  }
}
