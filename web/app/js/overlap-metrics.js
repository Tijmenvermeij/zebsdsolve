// Presentation and scalar filtering only. All fit scores and IPF colours come from the engine.
export const METRICS = Object.freeze({
  proposal_ncc: {label:"Search NCC",component:true,lower:false,default:0,min:-1,max:1,step:.01,description:"NCC against the target used to discover the selected orientation: the original pattern for a primary proposal, or the previous model's signed residual for an overlap proposal. Components can move to a different slot after support replacement. Recorded after native detector orientation refinement against that source target, following Index settings, and before the joint mixture refit. Higher is better; this is not a calibrated overlap probability."},
  mixture_ncc: {label:"Mixture NCC",lower:false,default:.3,min:-1,max:1,step:.01,description:"Normalized cross-correlation of the original prepared pattern with the combined fitted reconstruction. Higher is better. A high value measures reconstruction quality, not whether every retained component is physically present."},
  contrast_basis_weights: {label:"Model weight (%)",component:true,scale:100,lower:false,default:10,min:0,max:100,step:1,description:"Relative coefficient in the declared simulation basis, corrected for numerical template normalization. This is a share among fitted components, not a fraction of total measured signal or a calibrated volume fraction. A poor or noisy fit can still give a large weight."},
  prepared_fractions: {label:"Contrast share (%)",component:true,scale:100,lower:false,default:10,min:0,max:100,step:1,description:"The selected component's nonnegative fitted amplitude divided by the sum of all amplitudes, shown as a percentage. All amplitudes are refitted jointly. This is a share in the prepared-pattern model, not a physical volume fraction. Unreliable attribution is left unavailable."},
  sse: {label:"Residual SSE",lower:true,default:.9,min:0,step:.01,description:"Sum of squared pixels remaining after subtracting the complete fitted reconstruction from the original prepared pattern. Lower is better. The target has unit squared energy, so 0.90 means about 90% remains unexplained. For an assessed optimal fit, SSE is approximately 1 minus Mixture NCC squared."},
  incremental_improvement: {label:"Incremental ΔSSE",lower:false,default:.01,min:0,step:.001,description:"Reduction in SSE relative to the recorded lower-order comparison. In the profiled path, this uses the strongest achieved strictly lower-order fit across both searched histories. Higher means the addition helped more. Includes refitting every amplitude and, when enabled, blur and gain. A small positive reduction alone does not establish a real extra component."},
  improvement_over_best_single: {label:"ΔSSE over single",lower:false,default:.01,min:0,step:.001,description:"Reduction in SSE relative to the best one-component explanation evaluated for this result. In the profiled path, its pose can differ from every final mixture pose. Higher means the mixture improves more over a single component. Uses the same fitting model, including independently fitted blur and gain when enabled. This is an improvement score, not statistical acceptance."},
  relative_gram_condition: {label:"Condition ratio",lower:false,default:.01,min:0,max:1,step:.01,description:"Smallest divided by largest eigenvalue of the complete normalized template Gram matrix. Values near 1 indicate less-collinear normalized template shapes; near 0 means their amplitudes are difficult to separate. This does not include experimental noise or uncertainty from weak template contrast, and does not measure match quality."},
});
export const MAPS = Object.freeze({
  ipf_z:{label:"IPF Z",description:"Native inverse pole figure colour of specimen Z for the selected component's orientation. Colour represents orientation, not fit quality. Grey scan background marks uncomputed, unavailable or filtered pixels."},
  ipf_x:{label:"IPF X",description:"Native inverse pole figure colour of specimen X for the selected component's orientation. Colour represents orientation, not fit quality."},
  ipf_y:{label:"IPF Y",description:"Native inverse pole figure colour of specimen Y for the selected component's orientation. Colour represents orientation, not fit quality."},
  phase:{label:"Phase",description:"Phase assigned to the selected retained component. This is a candidate phase, not an independently accepted phase identification."},
  fraction:{...METRICS.prepared_fractions,label:"Contrast share",field:"prepared_fractions"},
  proposal_ncc:METRICS.proposal_ncc,
  mixture_ncc:METRICS.mixture_ncc,
  improvement:{...METRICS.incremental_improvement,field:"incremental_improvement"},
  single_improvement:{...METRICS.improvement_over_best_single,field:"improvement_over_best_single"},
  residual:{...METRICS.sse,field:"sse"},
  condition:{...METRICS.relative_gram_condition,field:"relative_gram_condition"},
  count:{label:"Active fit count",description:"Number of fitted amplitudes above 1e-6. This is a numerical activity count, not an accepted count of physical overlap components."},
  proposals:{label:"Retained candidates",description:"Number of retained orientation proposals in discovery order. A retained candidate can later receive zero amplitude; stable component slots are preserved."},
  status:{label:"Stop status",description:"Why component discovery stopped: requested limit, exhausted residual, no valid addition, or an unavailable/unresolved fit. Reaching the requested limit is not proof that all those components are physically present."},
});
export function weightMetricKey(run) {
  return run?.arrays?.contrast_basis_weights ? "contrast_basis_weights" : "prepared_fractions";
}
export function metricValue(run,cell,component,key) {
  const m=METRICS[key],field=run?.arrays?.[key];
  if(!m || !field || !run.completed?.[cell])return NaN;
  if(m.component && !(run.arrays.phase?.values[cell*run.k+component]>=0))return NaN;
  return field.values[cell*field.stride+(m.component?component:0)]*(m.scale||1);
}
export function passesFilter(run,cell,component,filter) {
  if(!run?.completed?.[cell])return false;
  if(!filter?.key)return true;
  if(!(run.arrays.retained_order?.values[cell]>0) || run.arrays.objective_valid?.values[cell]===0)return false;
  const v=metricValue(run,cell,component,filter.key);
  return Number.isFinite(v) && Number.isFinite(filter.cutoff) && (filter.operator==="max" ? v<=filter.cutoff : v>=filter.cutoff);
}
export function mapDescription(kind,component,run) {
  if(kind==="count" && run?.metadata?.profile_version)return "Strictly positive fitted support size after zero-coefficient compaction. This is a numerical activity count, not an accepted number of physical components. The exported fitted_count separately uses the legacy amplitude >1e-6 display convention.";
  if(kind==="proposals" && run?.metadata?.profile_version)return "Number of orientations retained in the selected support after replacements and zero-coefficient compaction. Slots describe the final support; proposal identities and historical order are recorded in exported events.";
  if(kind==="fraction")return METRICS[weightMetricKey(run)].description;
  const info=MAPS[kind];
  if(kind==="proposal_ncc") {
    const slot = component >= 0 ? ` for fit slot ${component+1}` : "";
    return `Residual search NCC${slot}: correlation against the signed residual before this component was added. `+METRICS.proposal_ncc.description;
  }
  return info?.description||"";
}

/** Inclusive display/next-run acceptance of the original indexing snapshot. */
export function acceptsPrimary(results,cell,cutoff,seedValid) {
  return seedValid===1 && results?.phase?.[cell]>=0 && Number.isFinite(results.ncc?.[cell]) && Number.isFinite(cutoff) && results.ncc[cell]>=cutoff;
}
/** Active components of the final fitted model, including single-component fits. */
export function acceptsFitComponent(run,cell,component) {
  if(!run?.completed?.[cell] || component<0 || component>=run.k)return false;
  const slot=cell*run.k+component, a=run.arrays;
  const minActiveAmplitude = run.metadata?.profile_version ? 0 : 1e-6;
  return a.phase?.values[slot]>=0 && a.amplitudes?.values[slot]>minActiveAmplitude && a.objective_valid?.values[cell]!==0;
}
/** Historical residual-source score, never joint mixture NCC or a moved primary. */
export function acceptsResidual(run,cell,component,cutoff=0) {
  if(!acceptsFitComponent(run,cell,component) || !Number.isFinite(cutoff) || !(run.arrays.active_count?.values[cell]>=2))return false;
  const slot=cell*run.k+component, a=run.arrays;
  const score=a.proposal_ncc?.values[slot];
  return a.winner_rank?.values[slot]>0 && Number.isFinite(score) && score>=cutoff;
}
/** Display-only evaluation of the frozen GPU gain formula; sigma is not applied to G. */
export function gainProfile(theta,w,h) {
  if(theta?.length!==7 || !Array.from(theta).every(Number.isFinite) || !(theta[3]>0 && theta[4]>0) || !(w>0 && h>0))return null;
  const [,logrho,power,ay,ax,dy,dx]=theta, values=new Float32Array(w*h),delta=Math.expm1(logrho);
  let min=Infinity,max=-Infinity;
  for(let row=0;row<h;row++)for(let col=0;col<w;col++) {
    const r=Math.hypot((row-h/2-dy*h)/(ay*h/2),(col-w/2-dx*w)/(ax*w/2));
    values[row*w+col]=1+delta*Math.max(1-r,0)**power;
    const v=values[row*w+col];min=Math.min(min,v);max=Math.max(max,v);
  }
  return {values,min,max,sigma:theta[0]};
}
