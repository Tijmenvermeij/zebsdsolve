// The isolated target remains in the once-prepared experimental pattern's units.
struct ConditionalParams {
    px: u32,
    components: u32,
    selected: u32,
    grid_x: u32,
    amplitudes: vec4<f32>,
}
@group(0) @binding(0) var<uniform> p: ConditionalParams;
@group(0) @binding(1) var<storage, read> measured: array<f32>;
@group(0) @binding(2) var<storage, read> templates: array<f32>;
@group(0) @binding(3) var<storage, read_write> target_pattern: array<f32>;

@compute @workgroup_size(256)
fn conditional_target(@builtin(workgroup_id) group: vec3<u32>, @builtin(local_invocation_index) lane: u32) {
    let pixel = (group.y * p.grid_x + group.x) * 256u + lane;
    if (pixel >= p.px) { return; }
    var value = measured[pixel];
    for (var j = 0u; j < p.components; j++) {
        if (j != p.selected) {
            value -= p.amplitudes[j] * templates[j * p.px + pixel];
        }
    }
    target_pattern[pixel] = value;
}

@group(0) @binding(4) var<storage, read> isolated: array<f32>;
@group(0) @binding(5) var<storage, read_write> scores: array<f32>;
var<workgroup> partial: array<f32, 256>;

// Reduce the separate isolated-component squared errors directly so
// a near-exact fit does not lose precision through cancellation.
@compute @workgroup_size(256)
fn conditional_score(@builtin(local_invocation_index) lane: u32) {
    var score = 0.0;
    for (var j = 0u; j < p.components; j++) {
        let base = j * p.px;
        for (var pixel = lane; pixel < p.px; pixel += 256u) {
            let residual = isolated[base + pixel] - p.amplitudes[j] * templates[base + pixel];
            score += residual * residual;
        }
    }
    partial[lane] = score;
    workgroupBarrier();
    for (var stride = 128u; stride > 0u; stride >>= 1u) {
        if (lane < stride) { partial[lane] += partial[lane + stride]; }
        workgroupBarrier();
    }
    if (lane == 0u) { scores[p.selected] = partial[0]; }
}
