#version 460

#extension GL_GOOGLE_include_directive : enable
#extension GL_EXT_nonuniform_qualifier : enable
#extension GL_EXT_ray_tracing : enable
#extension GL_EXT_scalar_block_layout : enable
#extension GL_EXT_debug_printf : enable

#include "raycommon.glsl"
#include "hitcommon.glsl"

layout(location = 0) rayPayloadInEXT RayPayload ray_info;

layout(scalar, set = 0, binding = 4) readonly buffer InstanceOffsets {
    Offsets offsets[];
} offsets;

layout(scalar, set = 0, binding = 5) readonly buffer Fields {
    BrdfParams params[];
} instance_info;

void main() {
    uint brdf_i = offsets.offsets[gl_InstanceID].brdf_i;
    ray_info.rad = instance_info.params[brdf_i].albedo;
}
