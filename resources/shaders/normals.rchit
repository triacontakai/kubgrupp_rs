#version 460

#extension GL_GOOGLE_include_directive : enable
#extension GL_EXT_ray_tracing : enable

#include "ray_common.glsl"
#include "hit_common.glsl"
#include "mesh_common.glsl"

layout(location = 0) rayPayloadInEXT RayPayload ray_info;

hitAttributeEXT vec2 bary_coord;

void main() {
    MeshHitInfo hit = compute_mesh_hit(bary_coord);

    ray_info.rad = abs(hit.normal);
    ray_info.is_hit = true;
    ray_info.is_emitter = true;
}
