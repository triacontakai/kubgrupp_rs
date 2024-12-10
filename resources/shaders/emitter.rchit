#version 460

#extension GL_GOOGLE_include_directive : enable
#extension GL_EXT_nonuniform_qualifier : enable
#extension GL_EXT_ray_tracing : enable

#include "ray_common.glsl"
#include "hit_common.glsl"

layout(location = 0) rayPayloadInEXT RayPayload ray_info;

void main() {
    ray_info.is_hit = true;
    ray_info.is_emitter = true;
    ray_info.rad = lights.lights[nonuniformEXT(gl_InstanceCustomIndexEXT)].color;
}
