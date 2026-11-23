// Shared prepared-master-domain blur/gain. All fitting and trial selection stay GPU resident.
// theta = sigma, log(rho), power, axis_y, axis_x, offset_y, offset_x.
struct P { n:u32, px:u32, width:u32, grid_x:u32, mode:u32, coord:u32, phase:u32, seed:u32, step:f32, dc:f32, k:u32, contrast:u32, mask:vec4<f32> }
struct State { sse:f32, initial_sse:f32, phase:f32, flags:f32, amplitude:vec4<f32>, single_sse:vec4<f32>, correlation:f32, condition:f32, best_single_sse:f32, trial_sse:f32 }
struct Progress { retained_order:u32, proposed_count:u32, active_count:u32, stop_reason:u32, previous_sse:f32, incremental_improvement:f32, boundary_seen:u32, proposal_phase:u32 }
@group(0) @binding(0) var<uniform> p:P;
@group(0) @binding(1) var<storage,read_write> raw:array<f32>;
@group(0) @binding(2) var<storage,read_write> transformed:array<f32>;
@group(0) @binding(3) var<storage,read_write> mid:array<f32>;
@group(0) @binding(4) var<storage,read_write> blurred:array<f32>;
@group(0) @binding(5) var<storage,read_write> cells:array<f32>;
@group(0) @binding(6) var<storage,read_write> trial:array<State>;
@group(0) @binding(7) var<storage,read_write> best:array<State>;
@group(0) @binding(8) var<storage,read_write> main:array<State>;
@group(0) @binding(9) var<storage,read_write> progress:array<Progress>;
@group(0) @binding(10) var<storage,read_write> bank:array<f32>;
@group(0) @binding(11) var<storage,read_write> committed:array<f32>;
@group(0) @binding(12) var<storage,read_write> theta_out:array<f32>;
@group(0) @binding(13) var<storage,read> candidate:array<f32>;
@group(0) @binding(14) var<storage,read_write> norms_out:array<f32>;
@group(0) @binding(15) var<storage,read_write> centered:array<f32>;
// 196 f32 per cell: working/best/committed/order theta at 0/8/16/24;
// singleton theta at32; DC at60; proposedDC64; cachedSigma65;
// proposed singleton SSE70/theta72; singleton SSE80; raw column validity84..87.
// Current-search coarse singleton seed snapshots90..117; proposed snapshot118..124.
// Masked blurred means128..131; reserved zero anchors132..135.
// Actual masked raw medians136..139; mask count144. Recomputed per raw stage.
// V4 discovery order baseline: singleton theta160..187, SSE188..191;
// historical rejection bits192; separate winning candidate ID193 (−1 missing).
fn ci(b:u32,j:u32)->u32 { return b*196u+j; }
// Mode 3 transforms one independently posed column per lane.
fn staged_columns()->u32 { return select(4u,1u,p.mode==3u); }
fn bounded_theta(x:f32,j:u32)->f32 {
 let lower=array<f32,7>(0.0,-1.38629436,0.25,0.6,0.6,-0.15,-0.15);
 let upper=array<f32,7>(5.0,1.38629436,8.0,1.4,1.4,0.15,0.15);
 return clamp(x,lower[j],upper[j]);
}
fn identity(j:u32)->f32 { return select(0.0,1.0,j>=2u && j<=4u); }
fn count(b:u32)->u32 { return select(select(progress[b].retained_order+1u,progress[b].retained_order,p.mode==2u),p.k,p.k>0u); }
fn column(b:u32,j:u32)->u32 { return select(j,select(p.coord,u32(cells[ci(b,88u)]),p.k==0u),p.mode==1u); }
fn source_index(b:u32,j:u32,i:u32)->u32 { return (column(b,j)*p.n+b)*p.px+i; }
fn image_id(w:vec3<u32>)->u32 { return w.y*p.grid_x+w.x; }
fn inside(i:u32)->bool { let d=vec2<f32>(f32(i/p.width),f32(i%p.width))-p.mask.xy; return p.mask.z<=0.0 || dot(d,d)<=p.mask.z; }
@compute @workgroup_size(1)
fn initialize(@builtin(workgroup_id) w:vec3<u32>) {
 let b=image_id(w); if(b>=p.n){return;}
 for(var j=0u;j<7u;j++){cells[ci(b,16u+j)]=identity(j);cells[ci(b,24u+j)]=identity(j);}
 for(var j=0u;j<4u;j++){cells[ci(b,80u+j)]=main[b].initial_sse;}
}
@compute @workgroup_size(1)
fn invalidate_sigma(@builtin(workgroup_id) w:vec3<u32>) {
 let b=image_id(w);if(b<p.n){cells[ci(b,65u)]=-1.0;}
}
@compute @workgroup_size(1)
fn stage_polish_theta(@builtin(workgroup_id) w:vec3<u32>) {
 let b=image_id(w);if(b>=p.n){return;}
 for(var j=0u;j<7u;j++){cells[ci(b,j)]=bounded_theta(theta_out[b*7u+j],j);}
 cells[ci(b,65u)]=-1.0;
}
@compute @workgroup_size(1)
fn share_primary_theta(@builtin(workgroup_id) w:vec3<u32>) {
 let b=image_id(w);if(b>=p.n){return;}
 for(var col=1u;col<4u;col++){
  for(var i=0u;i<7u;i++){
   cells[ci(b,90u+col*7u+i)]=cells[ci(b,90u+i)];
   cells[ci(b,32u+col*7u+i)]=cells[ci(b,32u+i)];
  }
 }
}
@compute @workgroup_size(1)
fn begin_order(@builtin(workgroup_id) w:vec3<u32>) {
 let b=image_id(w);if(b>=p.n){return;}
 cells[ci(b,193u)]=-1.0;
 for(var j=0u;j<7u;j++){cells[ci(b,24u+j)]=cells[ci(b,16u+j)];}
 if(p.contrast==1u){
  for(var j=0u;j<28u;j++){cells[ci(b,160u+j)]=cells[ci(b,32u+j)];}
  for(var j=0u;j<4u;j++){cells[ci(b,188u+j)]=cells[ci(b,80u+j)];}
  cells[ci(b,192u)]=f32(u32(main[b].flags)&16u);
 }
}
@compute @workgroup_size(64)
fn stage(@builtin(workgroup_id) w:vec3<u32>,@builtin(local_invocation_index) lane:u32) {
 let b=image_id(w);if(b>=p.n){return;}
 let k=count(b);
 for(var j=0u;j<k;j++){
  for(var i=lane;i<p.px;i+=64u){let ix=(j*p.n+b)*p.px+i;raw[ix]=select(committed[ix],candidate[b*p.px+i],j+1u==k);}
 }
 if(lane==0u){cells[ci(b,64u)]=p.dc;cells[ci(b,65u)]=-1.0;}
}
// V3 keeps its raw masked-flat guard. V4 checks full-image finite/nonconstant
// only; the actual blurred masked design determines contrast/rank downstream.
var<workgroup> lo:array<f32,64>;var<workgroup> hi:array<f32,64>;var<workgroup> valid:array<u32,64>;
@compute @workgroup_size(64)
fn validate(@builtin(workgroup_id) w:vec3<u32>,@builtin(local_invocation_index) lane:u32){
 let b=image_id(w);if(b>=p.n){return;}
 for(var j=0u;j<staged_columns();j++){
  var l=1e30;var h=-1e30;var ok=1u;
  for(var i=lane;i<p.px;i+=64u){
   let v=raw[(j*p.n+b)*p.px+i];
   if(p.contrast==1u){if(!finite(v)){ok=0u;}}else if(!(abs(v)<1e30)){ok=0u;}
   if(p.contrast==1u || inside(i)){l=min(l,v);h=max(h,v);}
  }
  lo[lane]=l;hi[lane]=h;valid[lane]=ok;workgroupBarrier();
  for(var d=32u;d>0u;d>>=1u){if(lane<d){lo[lane]=min(lo[lane],lo[lane+d]);hi[lane]=max(hi[lane],hi[lane+d]);valid[lane]&=valid[lane+d];}workgroupBarrier();}
  if(lane==0u){cells[ci(b,84u+j)]=select(0.0,1.0,valid[0]>0u && select(hi[0]-lo[0]>1e-6*max(max(abs(lo[0]),abs(hi[0])),1e-30),hi[0]>lo[0],p.contrast==1u));}workgroupBarrier();
 }
}
@compute @workgroup_size(1)
fn reset_search(@builtin(workgroup_id) w:vec3<u32>){
 let b=image_id(w);if(b>=p.n){return;}
 var s:State;s.initial_sse=main[b].initial_sse;s.sse=1e30;s.trial_sse=1e30;s.phase=-1.0;s.flags=12.0;s.condition=bitcast<f32>(0x7fc00000u | (p.n&0x7fu));s.correlation=s.condition;best[b]=s;
 cells[ci(b,65u)]=-1.0;
 for(var j=0u;j<7u;j++){cells[ci(b,8u+j)]=identity(j);}
}
@compute @workgroup_size(1)
fn propose(@builtin(workgroup_id) w:vec3<u32>){
 let b=image_id(w);if(b>=p.n){return;}
 for(var j=0u;j<7u;j++){
  var x=identity(j);
  if(p.seed==1u){x=cells[ci(b,24u+j)];}
  if(p.seed==2u && j==1u){x=0.25;}
  if(p.seed==3u && j==1u){x=-0.25;}
  if(p.seed>=4u && p.seed<8u){
   let c=p.seed-4u;var base=32u+c*7u;
   if(p.k>0u){base=90u+c*7u;}else if(c>=count(b)-1u){base=118u;}else if(p.contrast==1u){base=160u+c*7u;}
   x=cells[ci(b,base+j)];
  }
  if(p.seed==8u){x=cells[ci(b,8u+j)];if(j==p.coord){x+=p.step;}}
  if(p.seed>=9u && p.seed<=12u){
   let c=p.seed-9u;var base=32u+c*7u;
   if(p.k==0u && c>=count(b)-1u){base=72u;}else if(p.k==0u && p.contrast==1u){base=160u+c*7u;}
   x=cells[ci(b,base+j)];
  }
  cells[ci(b,j)]=bounded_theta(x,j);
 }
 var s:State;s.initial_sse=main[b].initial_sse;s.sse=s.initial_sse;s.trial_sse=1e30;trial[b]=s;
}
// Same separable Gaussian and truncated-edge renormalization as prep.wgsl;
// sigma is per cell here. Reuse blurred values throughout same-sigma gain trials.
@compute @workgroup_size(256)
fn blur_rows(@builtin(workgroup_id) w:vec3<u32>,@builtin(local_invocation_index) lane:u32){
 let tid=(w.y*p.grid_x+w.x)*256u+lane;if(tid>=staged_columns()*p.n*p.px){return;}
 let j=tid/(p.n*p.px);let b=(tid/p.px)%p.n;let i=tid%p.px;
 let k=select(count(b),1u,p.mode==1u);if(j>=k){return;}
 let sigma=cells[ci(b,0u)];if(sigma==cells[ci(b,65u)]){return;}
 if(sigma==0.0){mid[tid]=raw[source_index(b,j,i)];return;}
 let col=i32(i%p.width);let line=source_index(b,j,i)-u32(col);let radius=i32(ceil(3.0*sigma));var acc=0.0;var total=0.0;
 for(var d=-radius;d<=radius;d++){let c=col+d;if(c<0||c>=i32(p.width)){continue;}let weight=exp(-f32(d*d)/(2.0*sigma*sigma));acc+=weight*raw[line+u32(c)];total+=weight;}
 mid[tid]=acc/total;
}
@compute @workgroup_size(256)
fn blur_cols(@builtin(workgroup_id) w:vec3<u32>,@builtin(local_invocation_index) lane:u32){
 let tid=(w.y*p.grid_x+w.x)*256u+lane;if(tid>=staged_columns()*p.n*p.px){return;}
 let j=tid/(p.n*p.px);let b=(tid/p.px)%p.n;let i=tid%p.px;
 let k=select(count(b),1u,p.mode==1u);if(j>=k){return;}
 let sigma=cells[ci(b,0u)];if(sigma==cells[ci(b,65u)]){return;}
 if(sigma==0.0){blurred[tid]=mid[tid];return;}
 let row=i32(i/p.width);let height=p.px/p.width;let radius=i32(ceil(3.0*sigma));var acc=0.0;var total=0.0;
 for(var d=-radius;d<=radius;d++){let r=row+d;if(r<0||r>=i32(height)){continue;}let weight=exp(-f32(d*d)/(2.0*sigma*sigma));acc+=weight*mid[u32(i32(tid)+d*i32(p.width))];total+=weight;}
 blurred[tid]=acc/total;
}
@compute @workgroup_size(1)
fn cache_sigma(@builtin(workgroup_id) w:vec3<u32>){let b=image_id(w);if(b<p.n){cells[ci(b,65u)]=cells[ci(b,0u)];}}
@compute @workgroup_size(256)
fn gain(@builtin(workgroup_id) w:vec3<u32>,@builtin(local_invocation_index) lane:u32){
 let tid=(w.y*p.grid_x+w.x)*256u+lane;if(tid>=staged_columns()*p.n*p.px){return;}
 let j=tid/(p.n*p.px);let b=(tid/p.px)%p.n;let i=tid%p.px;let k=select(count(b),1u,p.mode==1u);
 if(j>=k){transformed[tid]=0.0;return;}
 let col=column(b,j);if(cells[ci(b,84u+col)]<0.5){transformed[tid]=0.0;return;}
 let h=f32(p.px/p.width);let ww=f32(p.width);let y=(f32(i/p.width)-(h*0.5+cells[ci(b,5u)]*h))/(cells[ci(b,3u)]*h*0.5);let x=(f32(i%p.width)-(ww*0.5+cells[ci(b,6u)]*ww))/(cells[ci(b,4u)]*ww*0.5);
 let shape=pow(max(1.0-sqrt(x*x+y*y),0.0),cells[ci(b,2u)]);let delta=exp(cells[ci(b,1u)])-1.0;
 let dc=select(cells[ci(b,60u+col)],cells[ci(b,64u)],col+1u==count(b));
 if(p.contrast==1u){
  transformed[tid]=(1.0+delta*shape)*((blurred[tid]-cells[ci(b,132u+j)])-cells[ci(b,128u+j)]);
 }else{
  transformed[tid]=(1.0+delta*shape)*blurred[tid]+delta*shape*dc;
 }
}
var<workgroup> keep:u32;
@compute @workgroup_size(64)
fn retain(@builtin(workgroup_id) w:vec3<u32>,@builtin(local_invocation_index) lane:u32){
 let b=image_id(w);if(b>=p.n){return;}
 let s=trial[b];
 if(lane==0u){
  keep=select(0u,1u,s.sse>=0.0 && s.sse<best[b].sse && (u32(s.flags)&8u)==0u);
  // Keep identity-design assessment if every optimized joint trial fails rank.
  if(p.seed==0u && (u32(s.flags)&8u)!=0u){best[b].condition=s.condition;best[b].correlation=s.correlation;best[b].flags=s.flags;}
 }
 workgroupBarrier();if(keep==0u){return;}
 if(lane==0u){best[b]=s;for(var j=0u;j<7u;j++){cells[ci(b,8u+j)]=cells[ci(b,j)];}}
 // Retain transformed best-bank; its raw counterpart stays unchanged.
 let k=select(count(b),1u,p.mode==1u);
 for(var j=0u;j<k;j++){for(var i=lane;i<p.px;i+=64u){let ix=(j*p.n+b)*p.px+i;bank[ix]=transformed[ix];}}
}
// Preserve the coarse seed before fine singleton polishing changes its theta.
// Fixed fits snapshot every supplied pose; discovery snapshots the proposed pose.
@compute @workgroup_size(1)
fn save_coarse_single(@builtin(workgroup_id) w:vec3<u32>){
 let b=image_id(w);if(b>=p.n){return;}
 let base=select(90u+p.coord*7u,118u,p.k==0u);
 for(var j=0u;j<7u;j++){cells[ci(b,base+j)]=cells[ci(b,8u+j)];}
}
@compute @workgroup_size(1)
fn save_single(@builtin(workgroup_id) w:vec3<u32>){
 let b=image_id(w);if(b>=p.n){return;}cells[ci(b,70u)]=min(best[b].sse,best[b].initial_sse);
 for(var j=0u;j<7u;j++){cells[ci(b,72u+j)]=cells[ci(b,8u+j)];}
}
// Collision-free tie key at the projection seam, independent of phase insertion
// labels. Identical raw projections are indistinguishable here; their numerical
// payload is identical because both were evaluated from the same order baseline.
fn candidate_precedes(b:u32,k:u32)->bool {
 let base=((k-1u)*p.n+b)*p.px;
 for(var i=0u;i<p.px;i++){
  let a=ordered_key(raw[base+i]);let z=ordered_key(committed[base+i]);
  if(a!=z){return a<z;}
 }
 return false;
}
@compute @workgroup_size(64)
fn commit(@builtin(workgroup_id) w:vec3<u32>,@builtin(local_invocation_index) lane:u32){
 let b=image_id(w);if(b>=p.n){return;}
 let s=best[b];let k=count(b);
 if(lane==0u){
  keep=0u;
  if(progress[b].stop_reason==0u && (progress[b].proposal_phase==0u || progress[b].proposal_phase==p.phase+1u)){
   if(!(s.sse<1e20) || (u32(s.flags)&8u)!=0u){main[b].flags=f32(u32(main[b].flags)|16u);if(p.contrast==1u){cells[ci(b,192u)]=16.0;}progress[b].boundary_seen|=2u;}
   else if(p.seed==0u && (s.amplitude[k-1u]<=0.0 || s.sse>=progress[b].previous_sse)){progress[b].boundary_seen|=1u;}
   else {keep=select(0u,1u,s.sse<main[b].trial_sse || (p.contrast==1u && s.sse==main[b].trial_sse && candidate_precedes(b,k)));}
  }
 }
 workgroupBarrier();if(keep==0u){return;}
 if(lane==0u){
  cells[ci(b,193u)]=f32(p.coord);
  var out=s;out.phase=f32(p.phase);out.trial_sse=s.sse;
  let rejection=select(u32(main[b].flags)&16u,u32(cells[ci(b,192u)]),p.contrast==1u);
  out.flags=f32(u32(out.flags)|rejection);
  // Every alternative owns a complete payload: previous provisional winners
  // must never supply earlier singleton seeds or diagnostics to this winner.
  if(p.contrast==1u){
   for(var j=0u;j+1u<k;j++){
    cells[ci(b,80u+j)]=cells[ci(b,188u+j)];
    for(var t=0u;t<7u;t++){cells[ci(b,32u+j*7u+t)]=cells[ci(b,160u+j*7u+t)];}
   }
  }
  // Initialize the proposed slot from THIS candidate, replacing any losing
  // phase's provisional slot cache. Earlier slots start from the order baseline.
  cells[ci(b,60u+k-1u)]=cells[ci(b,64u)];cells[ci(b,80u+k-1u)]=cells[ci(b,70u)];
  for(var j=0u;j<7u;j++){cells[ci(b,32u+(k-1u)*7u+j)]=cells[ci(b,72u+j)];}
  var singles=vec4<f32>(s.initial_sse);var bs=s.initial_sse;
  for(var j=0u;j<k;j++){
   // A winning shared theta is also a feasible independent singleton theta.
   // Fold this achieved improvement into the independently optimized cache.
   let score=select(s.single_sse[j],s.sse,k==1u);
   if(score<cells[ci(b,80u+j)] || k==1u){
    cells[ci(b,80u+j)]=score;
    for(var t=0u;t<7u;t++){cells[ci(b,32u+j*7u+t)]=cells[ci(b,8u+t)];}
   }
   singles[j]=cells[ci(b,80u+j)];bs=min(bs,singles[j]);
  }
  out.single_sse=singles;out.best_single_sse=bs;main[b]=out;
  for(var j=0u;j<7u;j++){cells[ci(b,16u+j)]=cells[ci(b,8u+j)];}
 }
 for(var j=0u;j<k;j++){for(var i=lane;i<p.px;i+=64u){let ix=(j*p.n+b)*p.px+i;committed[ix]=raw[ix];transformed[ix]=bank[ix];}}
}
@compute @workgroup_size(1)
fn read_theta(@builtin(workgroup_id) w:vec3<u32>){let b=image_id(w);if(b<p.n){for(var j=0u;j<7u;j++){theta_out[b*7u+j]=cells[ci(b,select(16u,8u,p.mode==1u)+j)];}}}
@compute @workgroup_size(1)
fn store_fixed_single(@builtin(workgroup_id) w:vec3<u32>){
 let b=image_id(w);if(b>=p.n){return;}cells[ci(b,80u+p.coord)]=min(best[b].sse,best[b].initial_sse);
 for(var j=0u;j<7u;j++){cells[ci(b,32u+p.coord*7u+j)]=cells[ci(b,8u+j)];}
}
@compute @workgroup_size(1)
fn finish_fixed(@builtin(workgroup_id) w:vec3<u32>){
 let b=image_id(w);if(b>=p.n){return;}
 if(main[b].initial_sse<=1e-12){for(var j=0u;j<7u;j++){cells[ci(b,16u+j)]=identity(j);}return;}
 var s=best[b];s.single_sse=vec4<f32>(s.initial_sse);s.best_single_sse=s.initial_sse;
 let joint_valid=s.sse<1e20 && (u32(s.flags)&8u)==0u;
 for(var j=0u;j<p.k;j++){
  s.single_sse[j]=cells[ci(b,80u+j)];
  if(joint_valid){
   s.single_sse[j]=min(s.single_sse[j],best[b].single_sse[j]);
   if(p.k==1u){s.single_sse[j]=s.sse;}
  }
  s.best_single_sse=min(s.best_single_sse,s.single_sse[j]);
 }
 if(!(s.sse<1e20)){let nan=bitcast<f32>(0x7fc00000u | (p.n&0x7fu));s.sse=nan;s.amplitude=vec4<f32>(nan);s.flags=f32(u32(s.flags)|8u);}
 main[b]=s;for(var j=0u;j<7u;j++){cells[ci(b,16u+j)]=cells[ci(b,8u+j)];}
}

@compute @workgroup_size(1)
fn set_single_column(@builtin(workgroup_id) w:vec3<u32>){let b=image_id(w);if(b<p.n){cells[ci(b,88u)]=f32(count(b)-1u);}}

// V4: full-image blur first, then mask-domain centering before gain. No H after G.
var<workgroup> sums:array<vec2<f32>,256>;
@compute @workgroup_size(256)
fn mean_blurred(@builtin(workgroup_id) w:vec3<u32>,@builtin(local_invocation_index) lane:u32){
 let b=image_id(w);if(b>=p.n){return;}
 let k=select(count(b),1u,p.mode==1u);
 for(var j=0u;j<staged_columns();j++){
  var bins:array<f32,8>;var count=0.0;
  for(var c=0u;c<8u;c++){bins[c]=0.0;}
  for(var i=lane;i<p.px;i+=256u){if(j<k && inside(i)){
   bins[(i/256u)&7u]+=blurred[(j*p.n+b)*p.px+i];count+=1.0;
  }}
  let total=((bins[0]+bins[1])+(bins[2]+bins[3]))+((bins[4]+bins[5])+(bins[6]+bins[7]));
  sums[lane]=vec2<f32>(total,count);workgroupBarrier();
  for(var d=128u;d>0u;d>>=1u){if(lane<d){sums[lane]+=sums[lane+d];}workgroupBarrier();}
  if(lane==0u){cells[ci(b,128u+j)]=sums[0].x/max(sums[0].y,1.0);cells[ci(b,132u+j)]=0.0;}workgroupBarrier();
 }
}
@compute @workgroup_size(64)
fn prepare_final(@builtin(workgroup_id) w:vec3<u32>,@builtin(local_invocation_index) lane:u32){
 let b=image_id(w);if(b>=p.n){return;}
 if(lane==0u){for(var j=0u;j<7u;j++){cells[ci(b,j)]=cells[ci(b,16u+j)];}cells[ci(b,65u)]=-1.0;}
 if(p.seed==1u){
  for(var j=0u;j<progress[b].retained_order;j++){for(var i=lane;i<p.px;i+=64u){let ix=(j*p.n+b)*p.px+i;raw[ix]=committed[ix];}}
 }
}
// Same two-pass masked reduction and 256-lane order as Prep.normalize, but
// preserve the norm before normalization. Output is cell-major, four slots.
@compute @workgroup_size(256)
fn component_norms(@builtin(workgroup_id) w:vec3<u32>,@builtin(local_invocation_index) lane:u32){
 let b=image_id(w);if(b>=p.n){return;}
 let nan=bitcast<f32>(0x7fc00000u | (p.n&0x7fu));
 for(var j=0u;j<4u;j++){
  var acc=vec2<f32>(0.0);
  if(j<count(b)){for(var i=lane;i<p.px;i+=256u){if(inside(i)){acc+=vec2<f32>(transformed[(j*p.n+b)*p.px+i],1.0);}}}
  sums[lane]=acc;workgroupBarrier();
  for(var d=128u;d>0u;d>>=1u){if(lane<d){sums[lane]+=sums[lane+d];}workgroupBarrier();}
  let mean=sums[0].x/max(sums[0].y,1.0);workgroupBarrier();
  var ss=0.0;
  if(j<count(b)){for(var i=lane;i<p.px;i+=256u){if(inside(i)){let v=transformed[(j*p.n+b)*p.px+i]-mean;ss+=v*v;}}}
  sums[lane]=vec2<f32>(ss,0.0);workgroupBarrier();
  for(var d=128u;d>0u;d>>=1u){if(lane<d){sums[lane]+=sums[lane+d];}workgroupBarrier();}
  if(lane==0u){let norm=sqrt(sums[0].x);norms_out[b*4u+j]=select(nan,norm,j<count(b) && norm>1e-15 && norm<1e30);}workgroupBarrier();
 }
}

// Numerically precondition the complete source image before the linear,
// constant-preserving Gaussian. This is not detector masking or background H.
// Original raw/committed references remain unchanged in their declared units.
@compute @workgroup_size(256)
fn center_raw(@builtin(workgroup_id) w:vec3<u32>,@builtin(local_invocation_index) lane:u32){
 let tid=(w.y*p.grid_x+w.x)*256u+lane;if(tid>=staged_columns()*p.n*p.px){return;}
 let image=tid/p.px;let j=image/p.n;let b=image%p.n;
 centered[tid]=raw[tid]-cells[ci(b,136u+j)];
}

fn finite(v:f32)->bool { return (bitcast<u32>(v)&0x7f800000u)!=0x7f800000u; }
fn ordered_key(v:f32)->u32 { let bits=bitcast<u32>(v);return select(bits^0x80000000u,~bits,(bits&0x80000000u)!=0u); }
fn ordered_value(key:u32)->f32 { return bitcast<f32>(select(~key,key^0x80000000u,(key&0x80000000u)!=0u)); }
var<workgroup> radix_bins:array<atomic<u32>,256>;
var<workgroup> radix_count:atomic<u32>;
var<workgroup> radix_prefix:u32;
var<workgroup> radix_mask:u32;
var<workgroup> radix_rank:u32;
// Actual upper masked median, selected exactly by four 8-bit radix passes over
// ordered finite f32 keys. Every barrier/column/pass is workgroup-uniform.
// Anchor selection is stage-only; no spatial H or outside-mask zeroing occurs.
@compute @workgroup_size(256)
fn raw_median(@builtin(workgroup_id) w:vec3<u32>,@builtin(local_invocation_index) lane:u32){
 let b=image_id(w);if(b>=p.n){return;}
 for(var j=0u;j<staged_columns();j++){
  if(lane==0u){atomicStore(&radix_count,0u);radix_prefix=0u;radix_mask=0u;}
  workgroupBarrier();
  var count=0u;
  for(var i=lane;i<p.px;i+=256u){let v=raw[(j*p.n+b)*p.px+i];if(inside(i)&&finite(v)){count++;}}
  atomicAdd(&radix_count,count);workgroupBarrier();
  if(lane==0u){radix_rank=atomicLoad(&radix_count)/2u;}
  workgroupBarrier();
  for(var digit=0u;digit<4u;digit++){
   let shift=24u-8u*digit;atomicStore(&radix_bins[lane],0u);workgroupBarrier();
   for(var i=lane;i<p.px;i+=256u){
    let v=raw[(j*p.n+b)*p.px+i];let key=ordered_key(v);
    if(inside(i)&&finite(v)&&(key&radix_mask)==radix_prefix){atomicAdd(&radix_bins[(key>>shift)&255u],1u);}
   }
   workgroupBarrier();
   if(lane==0u){
    var before=0u;var chosen=0u;
    for(var bin=0u;bin<256u;bin++){
     let n=atomicLoad(&radix_bins[bin]);
     if(radix_rank<before+n){chosen=bin;radix_rank-=before;break;}
     before+=n;
    }
    radix_prefix|=chosen<<shift;radix_mask|=255u<<shift;
   }
   workgroupBarrier();
  }
  if(lane==0u){
   let n=atomicLoad(&radix_count);
   cells[ci(b,136u+j)]=select(0.0,ordered_value(radix_prefix),n>1u);
   cells[ci(b,144u)]=f32(n);
   if(n<=1u){cells[ci(b,84u+j)]=0.0;}
  }
  workgroupBarrier();
 }
}

@compute @workgroup_size(1)
fn read_winner_ids(@builtin(workgroup_id) w:vec3<u32>){let b=image_id(w);if(b<p.n){theta_out[b]=cells[ci(b,193u)];}}
