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
    MeshHitInfo hit = compute_mesh_hit(bary_coord);

    sample_emitter(hit.position, hit.normal);
    sample_brdf(hit.normal);

    ray_info.hit_pos = hit.position;
    ray_info.hit_normal = hit.normal;
    ray_info.hit_geo_normal = hit.geo_normal;
    ray_info.is_hit = true;
    ray_info.is_emitter = false;
    ray_info.is_specular = false;
}
