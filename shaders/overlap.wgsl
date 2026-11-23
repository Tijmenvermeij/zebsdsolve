// Fixed prepared-domain mixture fit. Vectors have zero mean and unit L2 norm.
// The input observed is immutable. Residuals remain signed f32; search normalization
// is a separate output operation, never a second nonlinear preparation.
struct Params { n: u32, px: u32, k: u32, phase: u32, mode: u32, grid_x: u32, width: u32, pad1: u32, mask: vec4<f32> }
struct State {
    sse: f32, initial_sse: f32, phase: f32, flags: f32,
    amplitude: vec4<f32>, single_sse: vec4<f32>,
    correlation: f32, condition: f32, best_single_sse: f32, trial_sse: f32,
}
struct Progress {
    retained_order: u32, proposed_count: u32, active_count: u32, stop_reason: u32,
    previous_sse: f32, incremental_improvement: f32, boundary_seen: u32, proposal_phase: u32,
}
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> observed: array<f32>;
@group(0) @binding(2) var<storage, read_write> bank: array<f32>;
@group(0) @binding(3) var<storage, read> candidate: array<f32>;
@group(0) @binding(4) var<storage, read_write> state: array<State>;
@group(0) @binding(5) var<storage, read_write> residual: array<f32>;
@group(0) @binding(6) var<storage, read_write> progress: array<Progress>;
@group(0) @binding(7) var<storage, read_write> polish_cache: array<f32>;
// Runtime payload: Tint forbids compile-time nonfinite f32 expressions. All
// payload values remain quiet NaNs; native/browser missing-value semantics agree.
fn unavailable() -> f32 { return bitcast<f32>(0x7fc00000u | (p.n & 0x7fu)); }

fn managed() -> bool { return p.mode == 2u || p.mode == 4u; }
fn candidate_fit() -> bool { return p.mode == 0u || managed(); }
fn component_count(b: u32) -> u32 {
    if (p.mode == 7u) { return 1u; }
    if (p.mode == 6u && p.k > 0u) { return p.k; }
    return select(p.k, min(progress[b].retained_order + 1u, 4u), managed() || p.mode == 6u);
}

// Bounded cyclic symmetric Jacobi on the whole normalized Gram matrix. Returns
// (lambda_min/lambda_max, assessed). Only linear algebra; no crystal operations.
fn spectrum(matrixInput: array<f32,16>, k: u32) -> vec2<f32> {
    if (k == 1u) { return vec2<f32>(1.0,1.0); }
    var a = matrixInput;
    for (var sweep = 0u; sweep < 10u; sweep++) {
        for (var j = 0u; j < k; j++) {
            for (var l = j + 1u; l < k; l++) {
                let off = a[j*4u+l];
                if (abs(off) <= 1e-8) { continue; }
                let tau = (a[l*4u+l]-a[j*4u+j])/(2.0*off);
                let t = select(-1.0,1.0,tau>=0.0)/(abs(tau)+sqrt(1.0+tau*tau));
                let c = inverseSqrt(1.0+t*t); let sn = t*c;
                a[j*4u+j] -= t*off; a[l*4u+l] += t*off;
                a[j*4u+l] = 0.0; a[l*4u+j] = 0.0;
                for (var r = 0u; r < k; r++) {
                    if (r == j || r == l) { continue; }
                    let x = a[r*4u+j]; let y = a[r*4u+l];
                    a[r*4u+j] = c*x-sn*y; a[j*4u+r] = a[r*4u+j];
                    a[r*4u+l] = sn*x+c*y; a[l*4u+r] = a[r*4u+l];
                }
            }
        }
    }
    var lo = 1e30; var hi = 0.0; var off = 0.0;
    for (var j = 0u; j < k; j++) {
        let d = a[j*4u+j];
        if (!(abs(d)<1e20)) { return vec2<f32>(0.0,0.0); }
        lo = min(lo,d); hi = max(hi,d);
        for (var l = j+1u; l < k; l++) { off = max(off,abs(a[j*4u+l])); }
    }
    if (!(hi>0.0) || lo < -1e-6*hi || !(off<=2e-6*hi)) { return vec2<f32>(0.0,0.0); }
    return vec2<f32>(max(lo,0.0)/hi,1.0);
}


var<workgroup> moments: array<array<f32, 64>, 25>;
var<workgroup> energy: array<f32, 64>;
var<workgroup> range_lo: array<f32, 64>;
var<workgroup> range_hi: array<f32, 64>;
fn inside(i: u32) -> bool {
    let dr = f32(i / p.width) - p.mask.x;
    let dc = f32(i % p.width) - p.mask.y;
    return p.mask.z <= 0.0 || dr * dr + dc * dc <= p.mask.z;
}

fn value(k: u32, b: u32, i: u32) -> f32 {
    if (candidate_fit() && k + 1u == component_count(b)) { return candidate[b * p.px + i]; }
    return bank[(k * p.n + b) * p.px + i];
}

@compute @workgroup_size(64)
fn initialize(@builtin(workgroup_id) w: vec3<u32>, @builtin(local_invocation_index) lane: u32) {
    let b = w.y * p.grid_x + w.x;
    if (b >= p.n) { return; }
    var ss = 0.0;
    var lo = 1e30; var hi = -1e30;
    for (var i = lane; i < p.px; i += 64u) {
        let v = observed[b * p.px + i]; ss += v * v;
        if (inside(i)) { lo = min(lo, v); hi = max(hi, v); }
    }
    energy[lane] = ss; range_lo[lane] = lo; range_hi[lane] = hi;
    workgroupBarrier();
    for (var off = 32u; off > 0u; off >>= 1u) {
        if (lane < off) {
            energy[lane] += energy[lane + off];
            range_lo[lane] = min(range_lo[lane], range_lo[lane + off]);
            range_hi[lane] = max(range_hi[lane], range_hi[lane + off]);
        }
        workgroupBarrier();
    }
    if (lane != 0u) { return; }
    ss = energy[0];
    // Existing preprocessing can amplify the mean's f32 roundoff for a nonzero
    // constant image. Treat a flat field under the actual mask as no information.
    let valid = ss >= 0.0 && ss < 1e20;
    if (valid && range_hi[0] - range_lo[0] <= 1e-6 * max(max(abs(range_lo[0]), abs(range_hi[0])), 1e-30)) { ss = 0.0; }
    var s: State;
    s.sse = ss; s.initial_sse = ss; s.phase = -1.0;
    s.flags = select(8.0, select(0.0,2.0,ss<=1e-12), valid);
    // Until a proposal design is actually evaluated, its scores/conditioning
    // are unavailable; the original target energy itself is still known.
    s.single_sse = vec4<f32>(unavailable());
    s.correlation = unavailable(); s.condition = unavailable();
    s.best_single_sse = unavailable(); s.trial_sse = 1e30;
    state[b] = s;
    var m: Progress;
    m.stop_reason = select(9u,select(0u,7u,ss<=1e-12),valid);
    progress[b] = m;
}

@compute @workgroup_size(1)
fn begin_step(@builtin(workgroup_id) w: vec3<u32>) {
    let b = w.y * p.grid_x + w.x;
    if (b >= p.n) { return; }
    if (p.mode == 1u && progress[b].stop_reason != 0u) { return; }
    state[b].phase = -1.0; state[b].trial_sse = 1e30;
    if (p.mode == 1u) {
        progress[b].previous_sse = state[b].sse;
        progress[b].incremental_improvement = 0.0;
        progress[b].proposed_count += 1u;
        progress[b].boundary_seen = 0u;
    }
}

@compute @workgroup_size(64)
fn fit(@builtin(workgroup_id) w: vec3<u32>, @builtin(local_invocation_index) lane: u32) {
    let b = w.y * p.grid_x + w.x;
    if (b >= p.n) { return; }
    let old = state[b];
    let k = component_count(b);
    var gram: array<f32, 16>;
    var rhs: array<f32, 4>;
    var means: array<f32, 4>;
    var n_mask = 0.0;
    for (var i = lane; i < p.px; i += 64u) {
        let y = observed[b * p.px + i];
        if (inside(i)) { n_mask += 1.0; }
        var v: array<f32, 4>;
        for (var j = 0u; j < k; j++) { v[j] = value(j, b, i); rhs[j] += v[j] * y; means[j] += v[j]; }
        for (var j = 0u; j < k; j++) {
            for (var l = 0u; l < k; l++) { gram[j * 4u + l] += v[j] * v[l]; }
        }
    }
    for (var j = 0u; j < 16u; j++) { moments[j][lane] = gram[j]; }
    for (var j = 0u; j < 4u; j++) { moments[16u + j][lane] = rhs[j]; }
    for (var j = 0u; j < 4u; j++) { moments[20u + j][lane] = means[j]; }
    moments[24][lane] = n_mask;
    workgroupBarrier();
    for (var off = 32u; off > 0u; off >>= 1u) {
        if (lane < off) {
            for (var j = 0u; j < 25u; j++) { moments[j][lane] += moments[j][lane + off]; }
        }
        workgroupBarrier();
    }
    if (lane != 0u) { return; }
    if (managed() && progress[b].stop_reason != 0u) { return; }
    if (managed() && progress[b].proposal_phase != 0u && progress[b].proposal_phase != p.phase+1u) { return; }
    // Exhaustion belongs BETWEEN orders, never between phase competitors in
    // this order: an exact later phase must beat a near-perfect provisional one.
    if (old.initial_sse <= 1e-12) { return; }
    for (var j = 0u; j < 16u; j++) { gram[j] = moments[j][0]; }
    for (var j = 0u; j < 4u; j++) { rhs[j] = moments[16u + j][0]; }
    var flat_column = false;
    for (var j = 0u; j < k; j++) {
        let diag = gram[j * 4u + j];
        let sum = moments[20u + j][0];
        if (diag - sum * sum / max(moments[24][0], 1.0) <= 1e-6 * max(diag, 1e-30)) {
            flat_column = true;
            rhs[j] = 0.0;
            for (var l = 0u; l < k; l++) { gram[j * 4u + l] = 0.0; gram[l * 4u + j] = 0.0; }
        }
    }
    var best = old.initial_sse;
    var coef = vec4<f32>(0.0);
    var single = vec4<f32>(old.initial_sse);
    var best_single = old.initial_sse;
    var corr = 0.0;
    for (var j = 0u; j < k; j++) {
        let d = max(gram[j * 4u + j], 1e-30);
        let single_amp = max(rhs[j], 0.0) / d;
        var single_ss = 0.0;
        for (var i = 0u; i < p.px; i++) {
            let v = observed[b * p.px + i] - single_amp * value(j, b, i);
            single_ss += v * v;
        }
        single[j] = single_ss;
        best_single = min(best_single, single_ss);
        for (var l = 0u; l < j; l++) {
            let r = gram[j * 4u + l] / sqrt(max(gram[j * 4u + j] * gram[l * 4u + l], 1e-30));
            if (abs(r) > abs(corr)) { corr = r; }
        }
    }
    // Check the WHOLE normalized design before comparing mixture objectives.
    // Diagonal-pivoted Schur complements reject a conservative f32 rank margin;
    // pairwise correlation alone misses dependent triples. No ridge or caps.
    var rank_matrix: array<f32, 16>;
    for (var j = 0u; j < k; j++) {
        for (var l = 0u; l < k; l++) {
            rank_matrix[j * 4u + l] = gram[j * 4u + l] / sqrt(max(gram[j * 4u + j] * gram[l * 4u + l], 1e-30));
        }
    }
    let eig = spectrum(rank_matrix,k);
    var condition = eig.x;
    if (k == 2u && !flat_column) { condition = max(0.0,(1.0-abs(corr))/max(1.0+abs(corr),1e-30)); }
    if (flat_column) { condition = 0.0; }
    var reliable = !flat_column && eig.y > 0.5;
    for (var c = 0u; c < k; c++) {
        var pivot = c;
        for (var j = c + 1u; j < k; j++) {
            if (rank_matrix[j * 4u + j] > rank_matrix[pivot * 4u + pivot]) { pivot = j; }
        }
        if (rank_matrix[pivot * 4u + pivot] <= 1e-5) { reliable = false; break; }
        for (var j = 0u; j < k; j++) {
            let temp = rank_matrix[c * 4u + j]; rank_matrix[c * 4u + j] = rank_matrix[pivot * 4u + j]; rank_matrix[pivot * 4u + j] = temp;
        }
        for (var j = 0u; j < k; j++) {
            let temp = rank_matrix[j * 4u + c]; rank_matrix[j * 4u + c] = rank_matrix[j * 4u + pivot]; rank_matrix[j * 4u + pivot] = temp;
        }
        let diag = rank_matrix[c * 4u + c];
        for (var j = c + 1u; j < k; j++) {
            for (var l = c + 1u; l < k; l++) {
                rank_matrix[j * 4u + l] -= rank_matrix[j * 4u + c] * rank_matrix[c * 4u + l] / diag;
            }
        }
    }
    var active_sets_reliable = true;
    // Exhaustive subset sets: at most 15 systems of at most 4 unknowns.
    // Singular sets are skipped; their lower-dimensional faces are enumerated too.
    for (var subset = 1u; subset < (1u << k) && reliable; subset++) {
        var idx: array<u32, 4>; var m = 0u;
        for (var j = 0u; j < k; j++) { if ((subset & (1u << j)) != 0u) { idx[m] = j; m++; } }
        var a: array<f32, 20>;
        for (var r = 0u; r < m; r++) {
            for (var c = 0u; c < m; c++) { a[r * 5u + c] = gram[idx[r] * 4u + idx[c]]; }
            a[r * 5u + 4u] = rhs[idx[r]];
        }
        var valid = true;
        for (var c = 0u; c < m; c++) {
            var pivot = c;
            for (var r = c + 1u; r < m; r++) { if (abs(a[r * 5u + c]) > abs(a[pivot * 5u + c])) { pivot = r; } }
            if (abs(a[pivot * 5u + c]) < 1e-6) { valid = false; active_sets_reliable = false; break; }
            for (var j = 0u; j < 5u; j++) { let z = a[c * 5u + j]; a[c * 5u + j] = a[pivot * 5u + j]; a[pivot * 5u + j] = z; }
            let inv = 1.0 / a[c * 5u + c];
            for (var j = 0u; j < 5u; j++) { a[c * 5u + j] *= inv; }
            for (var r = 0u; r < m; r++) {
                if (r == c) { continue; }
                let factor = a[r * 5u + c];
                for (var j = 0u; j < 5u; j++) { a[r * 5u + j] -= factor * a[c * 5u + j]; }
            }
        }
        var x = vec4<f32>(0.0);
        for (var r = 0u; r < m; r++) { if (a[r * 5u + 4u] < -1e-6) { valid = false; } x[idx[r]] = max(a[r * 5u + 4u], 0.0); }
        if (!valid) { continue; }
        // Compare EACH feasible subset in pixel space, not by subtracting
        // cancellation-prone normal-equation energies. Final ranking uses this
        // same signed residual objective after the whole-design guard.
        var ss = 0.0;
        for (var i = 0u; i < p.px; i++) {
            var v = observed[b * p.px + i];
            for (var j = 0u; j < k; j++) { v -= x[j] * value(j, b, i); }
            ss += v * v;
        }
        if (ss < best) { best = ss; coef = x; }
    }
    reliable = reliable && active_sets_reliable;
    if (!reliable) {
        if (candidate_fit()) {
            // Discovery keeps the previous valid order and tries other phases.
            // This provisional objective must never enter candidate ranking.
            state[b].flags = f32(u32(old.flags) | 16u);
            if (managed()) { progress[b].boundary_seen |= 2u; }
        } else {
            // Supplied templates retain independently valid singleton diagnostics;
            // unresolved attribution and its claimed optimum are explicitly absent.
            let nan = unavailable();
            var s = old;
            s.sse = nan; s.trial_sse = nan; s.phase = -1.0;
            s.amplitude = vec4<f32>(nan); s.single_sse = single;
            s.best_single_sse = best_single; s.correlation = corr;
            s.condition = select(nan, condition, eig.y > 0.5);
            s.flags = 9.0 + select(4.0, 0.0, eig.y > 0.5);
            state[b] = s;
        }
        return;
    }
    // Recompute residual energy explicitly, avoiding cancellation in the normal equations.
    var ss = 0.0;
    for (var i = 0u; i < p.px; i++) {
        var v = observed[b * p.px + i];
        for (var j = 0u; j < k; j++) { v -= coef[j] * value(j, b, i); }
        ss += v * v;
    }
    if (managed() && p.mode != 4u && (coef[k-1u] <= 0.0 || ss >= progress[b].previous_sse)) {
        progress[b].boundary_seen |= 1u;
        return;
    }
    if (ss >= old.trial_sse) { return; }
    if (candidate_fit()) {
        for (var i = 0u; i < p.px; i++) { bank[((k - 1u) * p.n + b) * p.px + i] = candidate[b * p.px + i]; }
    }
    var s = old;
    s.sse = ss; s.trial_sse = ss; s.phase = f32(p.phase); s.amplitude = coef;
    s.single_sse = single; s.best_single_sse = best_single; s.correlation = corr;
    s.condition = condition;
    s.flags = f32(u32(old.flags) & 16u) + select(0.0,1.0,condition <= (1.0-.995)/(1.0+.995));
    state[b] = s;
}

@compute @workgroup_size(256)
fn make_residual(@builtin(workgroup_id) w: vec3<u32>, @builtin(local_invocation_index) lane: u32) {
    let b = w.y * p.grid_x + w.x;
    if (b >= p.n) { return; }
    let s = state[b];
    let k = select(p.k,progress[b].retained_order,p.k==0u);
    var scale = select(0.0, 1.0, s.initial_sse > 1e-12);
    if (p.k == 0u && progress[b].stop_reason != 0u) { scale = 0.0; }
    if (p.mode == 1u) { scale = select(0.0, inverseSqrt(max(s.sse, 1e-30)), s.sse > 1e-8 * max(s.initial_sse, 1e-30)); }
    if (p.k == 0u && progress[b].stop_reason != 0u) { scale = 0.0; }
    for (var i = lane; i < p.px; i += 256u) {
        var v = observed[b * p.px + i];
        for (var j = 0u; j < k; j++) { v -= s.amplitude[j] * bank[(j * p.n + b) * p.px + i]; }
        residual[b * p.px + i] = v * scale;
    }
}

// Nuisance-only NNLS: keep the same full-Gram guards and feasible active sets,
// but evaluate every signed pixel SSE across all 64 lanes. V2 fit stays unchanged.
var<workgroup> nuisance_coef: array<vec4<f32>,16>;
var<workgroup> nuisance_valid: array<u32,16>;
var<workgroup> nuisance_dual: array<f32,16>;
var<workgroup> nuisance_single_amp: vec4<f32>;
var<workgroup> nuisance_reliable: u32;
var<workgroup> nuisance_condition: f32;
var<workgroup> nuisance_correlation: f32;
var<workgroup> nuisance_assessed: f32;
// V4-only final-vector KKT certificate. No condition multiplier or coefficient
// waiver: stationarity/complementarity are checked after every clamp.
fn nuisance_kkt(gram:array<f32,16>,rhs:array<f32,4>,x:vec4<f32>,k:u32,yy:f32)->vec2<f32>{
 var valid=1.0;var dual=0.0;
 for(var j=0u;j<k;j++){
  if(!(abs(x[j])<1e30)||x[j]<0.0){valid=0.0;}
  var fitted=0.0;var scale=sqrt(max(gram[j*4u+j]*yy,0.0))+abs(rhs[j]);
  for(var l=0u;l<k;l++){fitted+=gram[j*4u+l]*x[l];scale+=abs(gram[j*4u+l])*x[l];}
  let gradient=rhs[j]-fitted;let tau=0.000003814697265625*scale;
  let error=select(max(gradient,0.0),abs(gradient),x[j]>0.0);
  if(!(abs(gradient)<1e30)||!(tau>=0.0)||error>tau){valid=0.0;}
  dual=max(dual,error);
 }
 return vec2<f32>(valid,dual);
}
// Eight independent short streams plus a fixed tree avoid long serial sums.
// This remains useful on GPU backends that reassociate compensation identities.
fn sum_eight(v:array<f32,8>)->f32 {
 return ((v[0]+v[1])+(v[2]+v[3]))+((v[4]+v[5])+(v[6]+v[7]));
}
fn nuisance_systems(k:u32) {
    var gram: array<f32,16>;
    var rhs: array<f32,4>;
    for(var j=0u;j<16u;j++){gram[j]=moments[j][0];nuisance_coef[j]=vec4<f32>(0.0);nuisance_valid[j]=0u;nuisance_dual[j]=1e30;}
    for(var j=0u;j<4u;j++){rhs[j]=moments[16u+j][0];}
    var flat_column = false;
    for (var j = 0u; j < k; j++) {
        let diag = gram[j * 4u + j];
        let sum = moments[20u + j][0];
        if (diag - sum * sum / max(moments[24][0], 1.0) <= 1e-6 * max(diag, 1e-30)) {
            flat_column = true;
            rhs[j] = 0.0;
            for (var l = 0u; l < k; l++) { gram[j * 4u + l] = 0.0; gram[l * 4u + j] = 0.0; }
        }
    }

    var corr=0.0;
    nuisance_single_amp=vec4<f32>(0.0);
    for(var j=0u;j<k;j++) {
        nuisance_single_amp[j]=max(rhs[j],0.0)/max(gram[j*4u+j],1e-30);
        for(var l=0u;l<j;l++){
            let r=gram[j*4u+l]/sqrt(max(gram[j*4u+j]*gram[l*4u+l],1e-30));
            if(abs(r)>abs(corr)){corr=r;}
        }
    }
    // Check the WHOLE normalized design before comparing mixture objectives.
    // Diagonal-pivoted Schur complements reject a conservative f32 rank margin;
    // pairwise correlation alone misses dependent triples. No ridge or caps.
    var rank_matrix: array<f32, 16>;
    for (var j = 0u; j < k; j++) {
        for (var l = 0u; l < k; l++) {
            rank_matrix[j * 4u + l] = gram[j * 4u + l] / sqrt(max(gram[j * 4u + j] * gram[l * 4u + l], 1e-30));
        }
    }
    let eig = spectrum(rank_matrix,k);
    var condition = eig.x;
    if (k == 2u && !flat_column) { condition = max(0.0,(1.0-abs(corr))/max(1.0+abs(corr),1e-30)); }
    if (flat_column) { condition = 0.0; }
    var reliable = !flat_column && eig.y > 0.5;
    for (var c = 0u; c < k; c++) {
        var pivot = c;
        for (var j = c + 1u; j < k; j++) {
            if (rank_matrix[j * 4u + j] > rank_matrix[pivot * 4u + pivot]) { pivot = j; }
        }
        if (rank_matrix[pivot * 4u + pivot] <= 1e-5) { reliable = false; break; }
        for (var j = 0u; j < k; j++) {
            let temp = rank_matrix[c * 4u + j]; rank_matrix[c * 4u + j] = rank_matrix[pivot * 4u + j]; rank_matrix[pivot * 4u + j] = temp;
        }
        for (var j = 0u; j < k; j++) {
            let temp = rank_matrix[j * 4u + c]; rank_matrix[j * 4u + c] = rank_matrix[j * 4u + pivot]; rank_matrix[j * 4u + pivot] = temp;
        }
        let diag = rank_matrix[c * 4u + c];
        for (var j = c + 1u; j < k; j++) {
            for (var l = c + 1u; l < k; l++) {
                rank_matrix[j * 4u + l] -= rank_matrix[j * 4u + c] * rank_matrix[c * 4u + l] / diag;
            }
        }
    }
    if(p.pad1==1u && reliable){
        let zero=nuisance_kkt(gram,rhs,vec4<f32>(0.0),k,energy[0]);
        nuisance_valid[0]=u32(zero.x);nuisance_dual[0]=zero.y;
    }
    var active_sets_reliable = true;
    // Exhaustive subset sets: at most 15 systems of at most 4 unknowns.
    // Singular sets are skipped; their lower-dimensional faces are enumerated too.
    for (var subset = 1u; subset < (1u << k) && reliable; subset++) {
        var idx: array<u32, 4>; var m = 0u;
        for (var j = 0u; j < k; j++) { if ((subset & (1u << j)) != 0u) { idx[m] = j; m++; } }
        var a: array<f32, 20>;
        for (var r = 0u; r < m; r++) {
            for (var c = 0u; c < m; c++) { a[r * 5u + c] = gram[idx[r] * 4u + idx[c]]; }
            a[r * 5u + 4u] = rhs[idx[r]];
        }
        var valid = true;
        for (var c = 0u; c < m; c++) {
            var pivot = c;
            for (var r = c + 1u; r < m; r++) { if (abs(a[r * 5u + c]) > abs(a[pivot * 5u + c])) { pivot = r; } }
            if (abs(a[pivot * 5u + c]) < 1e-6) { valid = false; active_sets_reliable = false; break; }
            for (var j = 0u; j < 5u; j++) { let z = a[c * 5u + j]; a[c * 5u + j] = a[pivot * 5u + j]; a[pivot * 5u + j] = z; }
            let inv = 1.0 / a[c * 5u + c];
            for (var j = 0u; j < 5u; j++) { a[c * 5u + j] *= inv; }
            for (var r = 0u; r < m; r++) {
                if (r == c) { continue; }
                let factor = a[r * 5u + c];
                for (var j = 0u; j < 5u; j++) { a[r * 5u + j] -= factor * a[c * 5u + j]; }
            }
        }
        var x = vec4<f32>(0.0);
        for (var r = 0u; r < m; r++) {
            let coefficient=a[r*5u+4u];
            if(p.pad1==1u){if(!(abs(coefficient)<1e30)||coefficient<0.0){valid=false;}}
            else if(coefficient < -1e-6){valid=false;}
            x[idx[r]]=max(coefficient,0.0);
        }
        if(p.pad1==1u){
            let certificate=nuisance_kkt(gram,rhs,x,k,energy[0]);
            valid=valid&&certificate.x>0.5;nuisance_dual[subset]=certificate.y;
        }
        if (!valid) { continue; }
        nuisance_coef[subset] = x;
        nuisance_valid[subset] = 1u;
    }
    nuisance_reliable = select(0u,1u,reliable && active_sets_reliable);
    nuisance_condition = condition;
    nuisance_correlation = corr;
    nuisance_assessed = eig.y;
}

@compute @workgroup_size(64)
fn fit_nuisance(@builtin(workgroup_id) w: vec3<u32>, @builtin(local_invocation_index) lane: u32) {
    let b = w.y * p.grid_x + w.x;
    if (b >= p.n) { return; }
    let old = state[b];
    let k = component_count(b);
    var gram: array<f32, 16>;
    var rhs: array<f32, 4>;
    var means: array<f32, 4>;
    var n_mask = 0.0;
    var moment_bins:array<array<f32,8>,25>;
    for (var i = lane; i < p.px; i += 64u) {
        let y = observed[b * p.px + i];
        if (inside(i)) { n_mask += 1.0; }
        var v: array<f32, 4>;
        let bin=(i/64u)&7u;
        for (var j = 0u; j < k; j++) {
            v[j]=value(j,b,i);
            if(p.pad1==1u){
                moment_bins[16u+j][bin]+=v[j]*y;
                moment_bins[20u+j][bin]+=v[j];
            }else{rhs[j]+=v[j]*y;means[j]+=v[j];}
        }
        for (var j = 0u; j < k; j++) {
            for (var l = 0u; l < k; l++) {
                let index=j*4u+l;
                if(p.pad1==1u){moment_bins[index][bin]+=v[j]*v[l];}
                else{gram[index]+=v[j]*v[l];}
            }
        }
        if(p.pad1==1u){moment_bins[24][bin]+=y*y;}
    }
    if(p.pad1==1u){
        for(var j=0u;j<16u;j++){gram[j]=sum_eight(moment_bins[j]);}
        for(var j=0u;j<4u;j++){rhs[j]=sum_eight(moment_bins[16u+j]);means[j]=sum_eight(moment_bins[20u+j]);}
        energy[lane]=sum_eight(moment_bins[24]);
    }
    for (var j = 0u; j < 16u; j++) { moments[j][lane] = gram[j]; }
    for (var j = 0u; j < 4u; j++) { moments[16u + j][lane] = rhs[j]; }
    for (var j = 0u; j < 4u; j++) { moments[20u + j][lane] = means[j]; }
    moments[24][lane] = n_mask;
    reduce_moments(lane,25u,p.pad1==1u);
    finish_nuisance(b,k,lane,old,false);
}

fn reduce_moments(lane:u32,count:u32,with_energy:bool) {
    workgroupBarrier();
    for (var off = 32u; off > 0u; off >>= 1u) {
        if (lane < off) {
            for (var j = 0u; j < count; j++) { moments[j][lane] += moments[j][lane + off]; }
            if(with_energy){energy[lane]+=energy[lane+off];}
        }
        workgroupBarrier();
    }
}

// A polish dispatch shares one target and one incumbent. Its bank is a single
// support (column-major), whereas ordinary fits have one support per target.
fn nuisance_value(j:u32,b:u32,i:u32,polish:bool)->f32 {
    if(!polish){return value(j,b,i);}
    if(j==p.phase){return candidate[b*p.px+i];}
    return bank[j*p.px+i];
}

fn finish_nuisance(b:u32,k:u32,lane:u32,old:State,polish:bool) {
    if(lane==0u){nuisance_systems(k);}
    workgroupBarrier();
    // Direct pixel-space scores: 15 nonempty subsets and four independent singles.
    var scores:array<f32,20>;var score_bins:array<array<f32,8>,20>;
    for(var i=lane;i<p.px;i+=64u){
        let y=observed[select(b,0u,polish)*p.px+i];let bin=(i/64u)&7u;
        var v:vec4<f32>;
        for(var j=0u;j<k;j++){v[j]=nuisance_value(j,b,i,polish);}
        for(var subset=select(1u,0u,p.pad1==1u);subset<(1u<<k);subset++){
            if(nuisance_valid[subset]==0u){continue;}
            var residual_value=y;
            for(var j=0u;j<k;j++){residual_value-=nuisance_coef[subset][j]*v[j];}
            if(p.pad1==1u){score_bins[subset][bin]+=residual_value*residual_value;}
            else{scores[subset]+=residual_value*residual_value;}
        }
        for(var j=0u;j<k;j++){
            let residual_value=y-nuisance_single_amp[j]*v[j];
            if(p.pad1==1u){score_bins[16u+j][bin]+=residual_value*residual_value;}
            else{scores[16u+j]+=residual_value*residual_value;}
        }
    }
    for(var j=0u;j<20u;j++){
        if(p.pad1==1u){moments[j][lane]=sum_eight(score_bins[j]);}
        else{moments[j][lane]=scores[j];}
    }
    reduce_moments(lane,20u,false);
    if(lane!=0u){return;}
    if(old.initial_sse<=1e-12){return;}
    var single=vec4<f32>(old.initial_sse);var best_single=old.initial_sse;
    for(var j=0u;j<k;j++){single[j]=moments[16u+j][0];best_single=min(best_single,single[j]);}
    var out=old;
    out.single_sse=single;out.best_single_sse=best_single;
    out.condition=nuisance_condition;out.correlation=nuisance_correlation;
    if(nuisance_reliable==0u){
        let nan=unavailable();out.sse=nan;out.trial_sse=nan;out.phase=-1.0;
        out.amplitude=vec4<f32>(nan);out.flags=9.0+select(4.0,0.0,nuisance_assessed>0.5);out.condition=select(nan,nuisance_condition,nuisance_assessed>0.5);
        state[b]=out;return;
    }
    var best_score=old.initial_sse;var coefficient=vec4<f32>(0.0);
    if(p.pad1==1u){
        best_score=1e30;var best_dual=1e30;var found=false;
        for(var subset=0u;subset<(1u<<k);subset++){
            let score=moments[subset][0];let dual=nuisance_dual[subset];
            if(nuisance_valid[subset]!=0u && score>=0.0 && score<1e30 &&
               (!found || score<best_score || (score==best_score && dual<best_dual))){
                found=true;best_score=score;best_dual=dual;coefficient=nuisance_coef[subset];
            }
        }
        if(!found){
            let nan=unavailable();out.sse=nan;out.trial_sse=nan;out.phase=-1.0;
            out.amplitude=vec4<f32>(nan);out.flags=9.0;state[b]=out;return;
        }
    }else{
        // Preserve the V3 ascending subset/strict SSE selection path exactly.
        for(var subset=1u;subset<(1u<<k);subset++){
            if(nuisance_valid[subset]!=0u && moments[subset][0]<best_score){best_score=moments[subset][0];coefficient=nuisance_coef[subset];}
        }
    }
    out.sse=best_score;out.trial_sse=best_score;out.amplitude=coefficient;
    out.flags=select(0.0,1.0,nuisance_condition<=(1.0-.995)/(1.0+.995));
    state[b]=out;
}


// Use the same contrast bins and reduction tree as fit_nuisance. Only entries
// independent of the replaced column need to be cached.
@compute @workgroup_size(64)
fn polish_moments(@builtin(local_invocation_index) lane:u32) {
    var bins:array<array<f32,8>,25>;
    var yy:array<f32,8>;
    var n_mask=0.0;
    for(var i=lane;i<p.px;i+=64u){
        let y=observed[i];let bin=(i/64u)&7u;
        if(inside(i)){n_mask+=1.0;}
        yy[bin]+=y*y;
        var v:array<f32,4>;
        for(var j=0u;j<p.k;j++){
            if(j==p.phase){continue;}
            v[j]=bank[j*p.px+i];
            bins[16u+j][bin]+=v[j]*y;
            bins[20u+j][bin]+=v[j];
        }
        for(var j=0u;j<p.k;j++){
            if(j==p.phase){continue;}
            for(var l=0u;l<p.k;l++){
                if(l!=p.phase){bins[j*4u+l][bin]+=v[j]*v[l];}
            }
        }
    }
    for(var j=0u;j<24u;j++){moments[j][lane]=sum_eight(bins[j]);}
    moments[24][lane]=n_mask;
    energy[lane]=sum_eight(yy);
    reduce_moments(lane,25u,true);
    if(lane==0u){
        for(var j=0u;j<25u;j++){polish_cache[j]=moments[j][0];}
        polish_cache[25]=energy[0];
    }
}

@compute @workgroup_size(64)
fn polish_trials(@builtin(workgroup_id) w:vec3<u32>,@builtin(local_invocation_index) lane:u32) {
    let b=w.y*p.grid_x+w.x;
    if(b>=p.n){return;}
    let k=p.k;let column=p.phase;let old=state[b];
    // Four Gram terms, the replaced RHS, and the replaced column's sum.
    var bins:array<array<f32,8>,6>;
    for(var i=lane;i<p.px;i+=64u){
        let bin=(i/64u)&7u;
        let v=candidate[b*p.px+i];
        bins[4][bin]+=v*observed[i];
        bins[5][bin]+=v;
        for(var j=0u;j<k;j++){
            var other=v;
            if(j!=column){other=bank[j*p.px+i];}
            bins[j][bin]+=v*other;
        }
    }
    for(var j=0u;j<6u;j++){moments[j][lane]=sum_eight(bins[j]);}
    reduce_moments(lane,6u,false);
    if(lane==0u){
        var changed:array<f32,6>;
        for(var j=0u;j<6u;j++){changed[j]=moments[j][0];}
        for(var j=0u;j<25u;j++){moments[j][0]=polish_cache[j];}
        for(var j=0u;j<k;j++){
            moments[column*4u+j][0]=changed[j];
            moments[j*4u+column][0]=changed[j];
        }
        moments[16u+column][0]=changed[4];
        moments[20u+column][0]=changed[5];
        energy[0]=polish_cache[25];
    }
    finish_nuisance(b,k,lane,old,true);
}
