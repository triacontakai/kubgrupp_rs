#version 460

#extension GL_GOOGLE_include_directive : enable
#extension GL_EXT_nonuniform_qualifier : enable
#extension GL_EXT_ray_tracing : enable
#extension GL_EXT_scalar_block_layout : enable
#extension GL_EXT_debug_printf : enable

#include "raycommon.glsl"
#include "hitcommon.glsl"
#include "random.glsl"

struct BrdfParams {
    vec3 albedo;
};

layout(location = 0) rayPayloadInEXT RayPayload ray_info;

hitAttributeEXT vec2 bary_coord;

layout(scalar, set = 0, binding = 3) readonly buffer Vertices {
    Vertex vertices[];
} vertices;

layout(scalar, set = 0, binding = 4) readonly buffer Lights {
    uint num_lights;
    Light lights[];
} lights;

layout(scalar, set = 0, binding = 5) readonly buffer InstanceOffsets {
    Offsets offsets[];
} offsets;

layout(scalar, set = 0, binding = 6) readonly buffer Fields {
    BrdfParams params[];
} instance_info;

void sample_brdf(vec3 hit_normal) {
    uint brdf_i = offsets.offsets[gl_InstanceID].brdf_i;
    BrdfParams brdf = instance_info.params[brdf_i];

    vec4 cos_sample = sample_cosine_hemisphere();
    vec3 wi = cos_sample.xyz;
    float pdf = cos_sample.w;

    ray_info.brdf_vals = brdf.albedo;
    ray_info.brdf_pdf = 1;

    // we need to convert the wi sample to be relative to the normal
    // do this by creating a rotation from (0, 0, 1) to the normal
    vec3 axis = vec3(hit_normal.y, -hit_normal.x, 0);
    float cos_t = hit_normal.z;
    float sin_t = length(axis);
    vec3 r = normalize(axis);
    ray_info.brdf_d =
        wi * cos_t
        + cross(wi, r) * sin_t
        + r * dot(r, wi) * (1 - cos_t);
}

vec3 eval_brdf(vec3 wi, vec3 hit_normal) {
    uint brdf_i = offsets.offsets[gl_InstanceID].brdf_i;
    BrdfParams brdf = instance_info.params[brdf_i];
    return brdf.albedo;
}

void sample_emitter(vec3 hit_pos, vec3 hit_normal) {
    ray_info.rad = vec3(0);
}

void main() {
    Vertex a = vertices.vertices[gl_InstanceCustomIndexEXT + 3*gl_PrimitiveID];
    Vertex b = vertices.vertices[gl_InstanceCustomIndexEXT + 3*gl_PrimitiveID + 1];
    Vertex c = vertices.vertices[gl_InstanceCustomIndexEXT + 3*gl_PrimitiveID + 2];

    vec3 full_bary_coord = vec3(1 - bary_coord.x - bary_coord.y, bary_coord);

    vec3 hit_pos =
        a.position * full_bary_coord.x
        + b.position * full_bary_coord.y
        + c.position * full_bary_coord.z;
    hit_pos = gl_ObjectToWorldEXT * vec4(hit_pos, 1);

    vec3 hit_normal =
        a.normal * full_bary_coord.x
        + b.normal * full_bary_coord.y
        + c.normal * full_bary_coord.z;
    hit_normal = normalize(gl_ObjectToWorldEXT * vec4(hit_normal, 0));

    sample_emitter(hit_pos, hit_normal);
    sample_brdf(hit_normal);

    uint brdf_i = offsets.offsets[gl_InstanceID].brdf_i;
    //ray_info.rad = instance_info.params[brdf_i].albedo;
    ray_info.hit_pos = hit_pos;
    ray_info.hit_normal = hit_normal;
    ray_info.is_hit = true;
    ray_info.is_emitter = false;
    ray_info.is_specular = true;
}
