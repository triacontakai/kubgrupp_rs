const float PI = 3.1415926535897932384626433832795;

struct RayPayload {
    // inputs
    uint seed;

    // outputs
    bool is_hit;
    bool is_emitter;
    bool is_specular;
    vec3 rad;
    vec3 hit_pos;
    vec3 hit_normal;
    vec3 hit_geo_normal;

    vec3 brdf_vals;
    vec3 brdf_d;
    float brdf_pdf;

    vec3 emitter_o;
    float emitter_pdf;
    float emitter_brdf_pdf;
    vec3 emitter_brdf_vals;
    vec3 emitter_normal;
};
