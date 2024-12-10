#version 460

#extension GL_GOOGLE_include_directive : enable
#extension GL_EXT_nonuniform_qualifier : enable
#extension GL_EXT_ray_tracing : enable

#include "raycommon.glsl"

layout(location = 0) rayPayloadInEXT RayPayload ray_info;

void main() {
    ray_info.rad = vec3(0);
    ray_info.is_hit = false;
}
