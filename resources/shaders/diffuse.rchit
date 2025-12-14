#version 460

#extension GL_GOOGLE_include_directive : enable
#extension GL_EXT_nonuniform_qualifier : enable
#extension GL_EXT_ray_tracing : enable
#extension GL_EXT_debug_printf : enable

#include "ray_common.glsl"
#include "hit_common.glsl"
#include "random.glsl"
#include "sampling.glsl"
#include "emitter_sampling.glsl"

layout(location = 0) rayPayloadInEXT RayPayload ray_info;

hitAttributeEXT vec2 bary_coord;

struct BrdfParams {
    vec3 albedo;
};

layout(scalar, set = 0, binding = BRDF_PARAMS_BINDING) readonly buffer Fields {
    BrdfParams params[];
} instance_info;

vec4 eval_brdf(vec3 wi, vec3 hit_normal) {
    uint brdf_i = offsets.offsets[gl_InstanceID].brdf_i;
    BrdfParams brdf = instance_info.params[brdf_i];
    float cos_theta = max(0.0, dot(wi, hit_normal));
    float pdf = cos_theta / PI;
    return vec4(brdf.albedo, pdf);
}

void sample_brdf(vec3 hit_normal) {
    uint brdf_i = offsets.offsets[gl_InstanceID].brdf_i;
    BrdfParams brdf = instance_info.params[brdf_i];

    vec4 cos_sample = sample_cosine_hemisphere(rnd(ray_info.seed), rnd(ray_info.seed));
    ray_info.brdf_vals = brdf.albedo;
    ray_info.brdf_pdf = cos_sample.w;
    ray_info.brdf_d = frame_sample(cos_sample.xyz, hit_normal);
}

void sample_emitter(vec3 hit_pos, vec3 hit_normal) {
    EmitterSample light = sample_light(hit_pos, ray_info.seed);
    vec4 brdf_eval = eval_brdf(light.direction, hit_normal);

    ray_info.emitter_o = light.position;
    ray_info.emitter_pdf = light.pdf;
    ray_info.emitter_brdf_vals = brdf_eval.xyz;
    ray_info.emitter_brdf_pdf = brdf_eval.w;
    ray_info.emitter_normal = light.normal;
    ray_info.rad = light.radiance;
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

    vec3 edge1 = b.position - a.position;
    vec3 edge2 = c.position - a.position;
    vec3 face_normal = normalize(cross(edge1, edge2));
    face_normal = normalize(gl_ObjectToWorldEXT * vec4(face_normal, 0));

    bool is_backface = dot(gl_WorldRayDirectionEXT, face_normal) > 0.0;
    if (is_backface) {
        hit_normal = -hit_normal;
        face_normal = -face_normal;
    }

    sample_emitter(hit_pos, hit_normal);
    sample_brdf(hit_normal);

    ray_info.hit_pos = hit_pos;
    ray_info.hit_normal = hit_normal;
    ray_info.hit_geo_normal = face_normal;
    ray_info.is_hit = true;
    ray_info.is_emitter = false;
    ray_info.is_specular = false;
}
