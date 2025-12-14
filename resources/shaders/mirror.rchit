#version 460

#extension GL_GOOGLE_include_directive : enable
#extension GL_EXT_nonuniform_qualifier : enable
#extension GL_EXT_ray_tracing : enable

#include "ray_common.glsl"
#include "hit_common.glsl"
#include "random.glsl"

layout(location = 0) rayPayloadInEXT RayPayload ray_info;

hitAttributeEXT vec2 bary_coord;

void sample_brdf(vec3 hit_normal) {
    ray_info.brdf_vals = vec3(1);
    ray_info.brdf_pdf = 1.0;

    ray_info.brdf_d = reflect(gl_WorldRayDirectionEXT, hit_normal);
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
