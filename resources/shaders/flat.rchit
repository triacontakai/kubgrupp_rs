#version 460

#extension GL_GOOGLE_include_directive : enable
#extension GL_EXT_nonuniform_qualifier : enable
#extension GL_EXT_ray_tracing : enable
#extension GL_EXT_scalar_block_layout : enable
#extension GL_EXT_debug_printf : enable

#include "raycommon.glsl"

layout(location = 0) rayPayloadInEXT RayPayload ray_info;

struct BrdfParams {
    vec3 albedo;
    float _padding;
};

layout(scalar, set = 0, binding = 6) readonly buffer Fields {
    BrdfParams params[];
} instance_info;

void main() {
    debugPrintfEXT("flat hit");
    ray_info.rad = instance_info.params[nonuniformEXT(gl_InstanceID)].albedo;
}
