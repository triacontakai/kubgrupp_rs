#version 460

#extension GL_GOOGLE_include_directive : enable
#extension GL_EXT_nonuniform_qualifier : enable
#extension GL_EXT_ray_tracing : enable
#extension GL_EXT_scalar_block_layout : enable
#extension GL_EXT_debug_printf : enable

#include "raycommon.glsl"
#include "hitcommon.glsl"

struct BrdfParams {
    vec3 albedo;
};

layout(location = 0) rayPayloadInEXT RayPayload ray_info;

hitAttributeEXT vec2 bary_coord;

layout(scalar, set = 0, binding = 3) readonly buffer Vertices {
    Vertex vertices[];
} vertices;

layout(scalar, set = 0, binding = 5) readonly buffer InstanceOffsets {
    Offsets offsets[];
} offsets;

layout(scalar, set = 0, binding = 6) readonly buffer Fields {
    BrdfParams params[];
} instance_info;

void main() {
    Vertex a = vertices.vertices[gl_InstanceCustomIndexEXT + 3*gl_PrimitiveID];
    Vertex b = vertices.vertices[gl_InstanceCustomIndexEXT + 3*gl_PrimitiveID + 1];
    Vertex c = vertices.vertices[gl_InstanceCustomIndexEXT + 3*gl_PrimitiveID + 2];

    vec3 full_bary_coord = vec3(1 - bary_coord.x - bary_coord.y, bary_coord);

    vec3 hit_pos =
        a.position * full_bary_coord.x
        + b.position * full_bary_coord.y
        + c.position * full_bary_coord.z;
    vec3 final_pos = gl_ObjectToWorldEXT * vec4(hit_pos, 0);

    uint brdf_i = offsets.offsets[gl_InstanceID].brdf_i;
    ray_info.rad = instance_info.params[brdf_i].albedo;
    ray_info.hit_pos = final_pos;
    ray_info.is_hit = true;
    ray_info.is_emitter = true;
}
