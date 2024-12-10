#version 460

#extension GL_GOOGLE_include_directive : enable
#extension GL_EXT_nonuniform_qualifier : enable
#extension GL_EXT_ray_tracing : enable
#extension GL_EXT_scalar_block_layout : enable
#extension GL_EXT_debug_printf : enable

#include "raycommon.glsl"
#include "hitcommon.glsl"

layout(location = 0) rayPayloadInEXT RayPayload ray_info;

layout(scalar, set = 0, binding = 4) readonly buffer Lights {
    uint num_lights;
    Light lights[];
} lights;

void main() {
    ray_info.is_hit = true;
    ray_info.is_emitter = true;
    ray_info.rad = lights.lights[nonuniformEXT(gl_InstanceCustomIndexEXT)].color;
}
