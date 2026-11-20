// V6 overlap-only pose update. Concatenated with common.wgsl so this is the
// same left crystal-frame exponential update as the detector refiner.
@group(0) @binding(0) var<uniform> step_rad: vec4<f32>;
@group(0) @binding(1) var<storage, read_write> poses: array<vec4<f32>>;

@compute @workgroup_size(1)
fn profile_pose() {
    poses[1] = qnormalize(qmul(qexp(step_rad.xyz), poses[1]));
}
