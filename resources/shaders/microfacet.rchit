#version 460

#extension GL_GOOGLE_include_directive : enable
#extension GL_EXT_nonuniform_qualifier : enable
#extension GL_EXT_ray_tracing : enable

#include "ray_common.glsl"
#include "hit_common.glsl"
#include "mesh_common.glsl"
#include "random.glsl"
#include "sampling.glsl"
#include "emitter_sampling.glsl"

layout(location = 0) rayPayloadInEXT RayPayload ray_info;

hitAttributeEXT vec2 bary_coord;

struct BrdfParams {
    vec3 albedo;
    float ior;
    float roughness;
};

layout(scalar, set = 0, binding = BRDF_PARAMS_BINDING) readonly buffer Fields {
    BrdfParams params[];
} instance_info;

float g1(vec3 wv, vec3 wh, vec3 hit_normal, float roughness) {
    float cos_t = dot(wv, hit_normal);
    float tan_t = sqrt(1 - cos_t * cos_t) / cos_t;
    float b = 1 / (roughness * tan_t);
    float val = b < 1.6 ? (3.535 * b + 2.181 * b * b) / (1.0 + 2.276 * b + 2.577 * b * b) : 1.0;
    return dot(wv, wh) / cos_t > 0 ? val : 0.0;
}

vec4 eval_brdf(vec3 wi, vec3 hit_normal, float ks) {
    uint brdf_i = offsets.offsets[gl_InstanceID].brdf_i;
    BrdfParams brdf = instance_info.params[brdf_i];

    vec3 wh = normalize(wi - gl_WorldRayDirectionEXT);
    float g = g1(wi, wh, hit_normal, brdf.roughness) * g1(-gl_WorldRayDirectionEXT, wh, hit_normal, brdf.roughness);
    float f = fresnel(dot(wh, -gl_WorldRayDirectionEXT), 1/brdf.ior);

    float cos_wh = dot(wh, hit_normal);
    float cos_wi = dot(wi, hit_normal);
    float cos_wo = dot(-gl_WorldRayDirectionEXT, hit_normal);

    float d = pdf_beckmann(cos_wh, brdf.roughness);
    float jh = 1 / (4 * dot(wh, wi));
    float pdf = ks * d * jh + (1 - ks) * cos_wi / PI;
    vec3 vals = brdf.albedo / PI + ks * d * f * g / (4 * cos_wi * cos_wo * cos_wh);
    return vec4(vals, pdf);
}

void sample_brdf(vec3 hit_normal, float ks) {
    uint brdf_i = offsets.offsets[gl_InstanceID].brdf_i;
    BrdfParams brdf = instance_info.params[brdf_i];

    float u = rnd(ray_info.seed);
    float v = rnd(ray_info.seed);

    vec4 cos_sample = sample_cosine_hemisphere(u, v);

    if (rnd(ray_info.seed) < ks) {
        vec3 specular_normals = frame_sample(sample_beckmann(brdf.roughness, u, v), hit_normal);
        ray_info.brdf_d = reflect(gl_WorldRayDirectionEXT, specular_normals);
    } else {
        ray_info.brdf_d = frame_sample(cos_sample.xyz, hit_normal);
    }

    vec4 brdf_eval = eval_brdf(ray_info.brdf_d, hit_normal, ks);
    ray_info.brdf_pdf = brdf_eval.w;
    ray_info.brdf_vals = brdf_eval.xyz * dot(ray_info.brdf_d, hit_normal) / brdf_eval.w;
}

void sample_emitter(vec3 hit_pos, vec3 hit_normal, float ks) {
    EmitterSample light = sample_light(hit_pos, ray_info.seed);
    vec4 brdf_eval = eval_brdf(light.direction, hit_normal, ks);

    ray_info.emitter_o = light.position;
    ray_info.emitter_pdf = light.pdf;
    ray_info.emitter_brdf_vals = brdf_eval.xyz;
    ray_info.emitter_brdf_pdf = brdf_eval.w;
    ray_info.emitter_normal = light.normal;
    ray_info.rad = light.radiance;
}

void main() {
    MeshHitInfo hit = compute_mesh_hit(bary_coord);

    uint brdf_i = offsets.offsets[gl_InstanceID].brdf_i;
    BrdfParams brdf = instance_info.params[brdf_i];
    float ks = 1 - max(max(brdf.albedo.r, brdf.albedo.g), brdf.albedo.b);

    sample_emitter(hit.position, hit.normal, ks);
    sample_brdf(hit.normal, ks);

    ray_info.hit_pos = hit.position;
    ray_info.hit_normal = hit.normal;
    ray_info.hit_geo_normal = hit.geo_normal;
    ray_info.is_hit = true;
    ray_info.is_emitter = false;
    ray_info.is_specular = false;
}
