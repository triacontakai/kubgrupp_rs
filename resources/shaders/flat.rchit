#version 460

#extension GL_GOOGLE_include_directive : enable
#extension GL_EXT_nonuniform_qualifier : enable
#extension GL_EXT_ray_tracing : enable

#include "ray_common.glsl"
#include "hit_common.glsl"
#include "mesh_common.glsl"

layout(location = 0) rayPayloadInEXT RayPayload ray_info;

hitAttributeEXT vec2 bary_coord;

struct BrdfParams {
    vec3 albedo;
};

layout(scalar, set = 0, binding = BRDF_PARAMS_BINDING) readonly buffer Fields {
    BrdfParams params[];
} instance_info;

void main() {
    vec3 hit_pos = compute_mesh_hit_position(bary_coord);

    uint brdf_i = offsets.offsets[gl_InstanceID].brdf_i;
    ray_info.rad = instance_info.params[brdf_i].albedo;
    ray_info.hit_pos = hit_pos;
    ray_info.is_hit = true;
    ray_info.is_emitter = true;
}
