// V6 overlap-only pose update. Concatenated with common.wgsl so this is the
// same left crystal-frame exponential update as the detector refiner.
@group(0) @binding(0) var<uniform> step_rad: vec4<f32>;
@group(0) @binding(1) var<storage, read_write> poses: array<vec4<f32>>;

@compute @workgroup_size(1)
fn profile_pose() {
    poses[1] = qnormalize(qmul(qexp(step_rad.xyz), poses[1]));
}

@group(0) @binding(2) var<storage, read> angle_offsets: array<vec4<f32>>;

@compute @workgroup_size(64)
fn profile_poses(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if (i >= arrayLength(&angle_offsets)) { return; }
    let delta = angle_offsets[i].xyz;
    if (all(delta == vec3<f32>(0.0))) { return; }
    let q = 2u * i + 1u;
    poses[q] = qnormalize(qmul(qexp(delta), poses[q]));
}
