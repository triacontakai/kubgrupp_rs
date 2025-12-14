#version 460

#extension GL_GOOGLE_include_directive : enable
#extension GL_EXT_nonuniform_qualifier : enable
#extension GL_EXT_ray_tracing : enable

#include "ray_common.glsl"
#include "hit_common.glsl"
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

void sample_brdf(vec3 hit_normal) {
    ray_info.brdf_vals = vec3(1);
    ray_info.brdf_pdf = 1;

    uint brdf_i = offsets.offsets[gl_InstanceID].brdf_i;
    BrdfParams brdf = instance_info.params[brdf_i];

    float eta = 1.0 / brdf.ior;
    if (dot(hit_normal, -gl_WorldRayDirectionEXT) < 0.0) {
        hit_normal = -hit_normal;
        eta = brdf.ior;
    }

    vec3 reflected = reflect(gl_WorldRayDirectionEXT, hit_normal);
    float f = fresnel(abs(dot(reflected, hit_normal)), eta);

    float r = rnd(ray_info.seed);
    if (r < f) {
        ray_info.brdf_d = normalize(reflected);
    } else {
        ray_info.brdf_d = normalize(refract(gl_WorldRayDirectionEXT, hit_normal, eta));
    }
}

void sample_emitter(vec3 hit_pos, vec3 hit_normal) {
    ray_info.rad = vec3(0);
    ray_info.emitter_brdf_pdf = 1.0;
    ray_info.emitter_pdf = 1.0;
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

    ray_info.hit_pos = hit_pos;
    ray_info.hit_normal = hit_normal;
    ray_info.hit_geo_normal = hit_normal;
    ray_info.is_hit = true;
    ray_info.is_emitter = false;
    ray_info.is_specular = true;
}
