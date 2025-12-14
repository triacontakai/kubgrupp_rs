#version 460

#extension GL_GOOGLE_include_directive : enable
#extension GL_EXT_nonuniform_qualifier : enable
#extension GL_EXT_ray_tracing : enable
#extension GL_EXT_debug_printf : enable

#include "ray_common.glsl"
#include "hit_common.glsl"

layout(location = 0) rayPayloadInEXT RayPayload ray_info;

hitAttributeEXT vec2 bary_coord;

void main() {
    Light light = lights.lights[gl_InstanceCustomIndexEXT];
    vec3 a = light.data[0];
    vec3 b = light.data[1];
    vec3 c = light.data[2];

    vec3 full_bary_coord = vec3(1 - bary_coord.x - bary_coord.y, bary_coord);

    vec3 hit_pos =
        a * full_bary_coord.x
        + b * full_bary_coord.y
        + c * full_bary_coord.z;
    hit_pos = gl_ObjectToWorldEXT * vec4(hit_pos, 1);

    vec3 ab = light.data[1] - light.data[0];
    vec3 ac = light.data[2] - light.data[0];
    vec3 normal = cross(ab, ac);
    float area = length(normal) / 2;
    normal = normalize(normal);

    bool is_backface = dot(gl_WorldRayDirectionEXT, normal) >= 0.0;

    ray_info.is_hit = true;
    ray_info.hit_pos = hit_pos;
    ray_info.emitter_pdf = 1.0 / lights.num_lights / area;

    if (is_backface) {
        ray_info.is_emitter = true;
        ray_info.rad = vec3(0);
        ray_info.hit_normal = -normal;
    } else {
        ray_info.is_emitter = true;
        ray_info.rad = light.color;
        ray_info.hit_normal = normal;
    }
}
