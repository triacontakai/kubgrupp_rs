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

vec3 sample_cosine_hemisphere() {
    float u = rnd(ray_info.seed);
    float v = rnd(ray_info.seed);
    float theta = acos(sqrt(u));
    float phi = 2 * PI * v;

    return vec3(sin(theta) * cos(phi), sin(theta) * sin(phi), cos(theta));
}

void sample_brdf() {
}

vec3 eval_brdf(vec3 wi, vec3 hit_normal) {
    //return vec3(dot(wi, hit_normal) / PI);
    uint brdf_i = offsets.offsets[gl_InstanceID].brdf_i;
    BrdfParams brdf = instance_info.params[brdf_i];
    return brdf.albedo;
}

void sample_emitter(vec3 hit_pos, vec3 hit_normal) {
    // pick a random emitter
    uint light_i = uint(rnd(ray_info.seed) * lights.num_lights);

    Light light = lights.lights[light_i];
    vec3 dir_to_light = normalize(light.position - hit_pos);
    if (light.type == EMITTER_TYPE_POINT) {
        ray_info.emitter_o = light.position;
        ray_info.emitter_pdf = 1 / lights.num_lights;
        ray_info.emitter_brdf_vals = eval_brdf(dir_to_light, hit_normal);
        ray_info.emitter_normal = -dir_to_light;
        ray_info.rad = light.color;
        //debugPrintfEXT("%v3f", light.color);
    } else if (light.type == EMITTER_TYPE_AREA) {
        // TODO: implement area light
    }
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

    uint brdf_i = offsets.offsets[gl_InstanceID].brdf_i;
    //ray_info.rad = instance_info.params[brdf_i].albedo;
    ray_info.hit_pos = hit_pos;
    ray_info.hit_normal = hit_normal;
    ray_info.is_hit = true;
    ray_info.is_emitter = false;
}
