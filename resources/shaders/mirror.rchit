#version 460

#extension GL_GOOGLE_include_directive : enable
#extension GL_EXT_nonuniform_qualifier : enable
#extension GL_EXT_ray_tracing : enable

#include "ray_common.glsl"
#include "hit_common.glsl"
#include "mesh_common.glsl"
#include "random.glsl"

layout(location = 0) rayPayloadInEXT RayPayload ray_info;

hitAttributeEXT vec2 bary_coord;

void main() {
    MeshHitInfo hit = compute_mesh_hit(bary_coord);

    ray_info.brdf_vals = vec3(1);
    ray_info.brdf_pdf = 1.0;
    ray_info.brdf_d = reflect(gl_WorldRayDirectionEXT, hit.normal);

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
