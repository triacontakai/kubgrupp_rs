#version 460

#extension GL_GOOGLE_include_directive : enable
#extension GL_EXT_nonuniform_qualifier : enable
#extension GL_EXT_ray_tracing : enable

#include "ray_common.glsl"
#include "hit_common.glsl"
#include "mesh_common.glsl"
#include "random.glsl"
#include "sampling.glsl"

layout(location = 0) rayPayloadInEXT RayPayload ray_info;

hitAttributeEXT vec2 bary_coord;

struct BrdfParams {
    float ior;
};

layout(scalar, set = 0, binding = BRDF_PARAMS_BINDING) readonly buffer Fields {
    BrdfParams params[];
} instance_info;

void main() {
    MeshHitInfo hit = compute_mesh_hit(bary_coord);

    uint brdf_i = offsets.offsets[gl_InstanceID].brdf_i;
    BrdfParams brdf = instance_info.params[brdf_i];

    vec3 normal = hit.normal;
    float eta = 1.0 / brdf.ior;
    if (dot(normal, -gl_WorldRayDirectionEXT) < 0.0) {
        normal = -normal;
        eta = brdf.ior;
    }

    vec3 reflected = reflect(gl_WorldRayDirectionEXT, normal);
    float f = fresnel(abs(dot(reflected, normal)), eta);

    if (rnd(ray_info.seed) < f) {
        ray_info.brdf_d = normalize(reflected);
    } else {
        ray_info.brdf_d = normalize(refract(gl_WorldRayDirectionEXT, normal, eta));
    }

    ray_info.brdf_vals = vec3(1);
    ray_info.brdf_pdf = 1.0;

    ray_info.rad = vec3(0);
    ray_info.emitter_brdf_pdf = 1.0;
    ray_info.emitter_pdf = 1.0;

    ray_info.hit_pos = hit.position;
    ray_info.hit_normal = hit.normal;
    ray_info.hit_geo_normal = hit.geo_normal;
    ray_info.is_hit = true;
    ray_info.is_emitter = false;
    ray_info.is_specular = true;
}
