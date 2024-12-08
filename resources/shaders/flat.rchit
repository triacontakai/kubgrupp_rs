#version 460

#extension GL_GOOGLE_include_directive : enable
#extension GL_EXT_nonuniform_qualifier : enable
#extension GL_EXT_ray_tracing : enable

#include "raycommon.glsl"

layout(location = 0) rayPayloadInEXT RayPayload ray_info;

struct BrdfParams {
    vec3 albedo;
};

layout(set = 0, binding = 5) readonly buffer Fields {
    BrdfParams params[];
} instance_info;

void main() {
    ray_info.rad = instance_info.params[nonuniformEXT(gl_InstanceID)].albedo;
}
