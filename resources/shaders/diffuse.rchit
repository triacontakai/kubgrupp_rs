#version 460

#extension GL_GOOGLE_include_directive : enable
#extension GL_EXT_nonuniform_qualifier : enable
#extension GL_EXT_ray_tracing : enable
#extension GL_EXT_debug_printf : enable

#include "ray_common.glsl"
#include "hit_common.glsl"
#include "random.glsl"
#include "sampling.glsl"

layout(location = 0) rayPayloadInEXT RayPayload ray_info;

hitAttributeEXT vec2 bary_coord;

struct BrdfParams {
    vec3 albedo;
};

layout(scalar, set = 0, binding = BRDF_PARAMS_BINDING) readonly buffer Fields {
    BrdfParams params[];
} instance_info;

void sample_brdf(vec3 hit_normal) {
    uint brdf_i = offsets.offsets[gl_InstanceID].brdf_i;
    BrdfParams brdf = instance_info.params[brdf_i];

    vec4 cos_sample = sample_cosine_hemisphere(rnd(ray_info.seed), rnd(ray_info.seed));
    vec3 wi = cos_sample.xyz;
    float pdf = cos_sample.w;

    ray_info.brdf_vals = brdf.albedo;
    ray_info.brdf_pdf = pdf;

    ray_info.brdf_d = frame_sample(wi, hit_normal);
}

vec4 eval_brdf(vec3 wi, vec3 hit_normal) {
    uint brdf_i = offsets.offsets[gl_InstanceID].brdf_i;
    BrdfParams brdf = instance_info.params[brdf_i];
    float pdf = abs(dot(wi, hit_normal)) / PI;
    return vec4(brdf.albedo, pdf);
}

void sample_emitter(vec3 hit_pos, vec3 hit_normal) {
    // pick a random emitter
    uint light_i = uint(rnd(ray_info.seed) * lights.num_lights);

    Light light = lights.lights[light_i];
    if (light.type == EMITTER_TYPE_POINT) {
        vec3 dir_to_light = normalize(light.position - hit_pos);
        vec4 brdf_eval = eval_brdf(dir_to_light, hit_normal);
        ray_info.emitter_o = light.position;
        ray_info.emitter_pdf = 1.0 / lights.num_lights;
        ray_info.emitter_brdf_vals = brdf_eval.xyz;
        ray_info.emitter_brdf_pdf = brdf_eval.w;
        ray_info.emitter_normal = -dir_to_light;
        ray_info.rad = light.color;
    } else if (light.type == EMITTER_TYPE_AREA) {
        // sample random point on triangle 
        float s = rnd(ray_info.seed);
        float t = sqrt(rnd(ray_info.seed));

        float a = 1 - t;
        float b = (1 - s) * t;
        float c = s * t;

        vec3 ab = light.vertices[1] - light.vertices[0];
        vec3 ac = light.vertices[2] - light.vertices[0];
        vec3 normal = cross(ab, ac);
        float area = length(normal) / 2;
        normal = normalize(normal);

        ray_info.emitter_o =
            a * light.vertices[0] + b * light.vertices[1] + c * light.vertices[2];

        vec3 dir_to_light = normalize(ray_info.emitter_o - hit_pos);
        vec4 brdf_eval = eval_brdf(dir_to_light, hit_normal);

        ray_info.emitter_pdf = 1.0 / lights.num_lights / area;
        ray_info.emitter_brdf_vals = brdf_eval.xyz;
        ray_info.emitter_brdf_pdf = brdf_eval.w;
        ray_info.emitter_normal = normal;
        ray_info.rad = light.color;
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
    sample_brdf(hit_normal);

    ray_info.hit_pos = hit_pos;
    ray_info.hit_normal = hit_normal;
    ray_info.is_hit = true;
    ray_info.is_emitter = false;
    ray_info.is_specular = false;
}
