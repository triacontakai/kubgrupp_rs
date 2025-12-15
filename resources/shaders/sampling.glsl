vec4 sample_cosine_hemisphere(float u, float v) {
    float cos_theta = sqrt(u);
    float sin_theta = sqrt(1.0 - u);
    float phi = 2.0 * PI * v;

    return vec4(sin_theta * cos(phi), sin_theta * sin(phi), cos_theta, cos_theta / PI);
}

vec3 sample_beckmann(float a, float u, float v) {
    float cos_t = sqrt(1.0 / (1.0 - a * a * log(u)));
    float sin_t = sqrt(1.0 - cos_t * cos_t);

    float phi = 2.0 * PI * v;
    return vec3(sin_t * cos(phi), sin_t * sin(phi), cos_t);
}

float pdf_beckmann(float cos_t, float a) {
    float tan_t_2 = 1.0 / (cos_t * cos_t) - 1.0;
    return exp(-tan_t_2 / (a * a)) / (PI * a * a * cos_t * cos_t * cos_t);
}

float fresnel(float cos_i, float eta) {
    float sin_t_sq = eta * eta * (1.0 - cos_i * cos_i);

    if (sin_t_sq > 1.0) {
        return 1.0;
    }

    float cos_t = sqrt(max(1.0 - sin_t_sq, 0.0));

    float prl = (eta * cos_t - cos_i) / (eta * cos_t + cos_i);
    float ppd = (eta * cos_i - cos_t) / (eta * cos_i + cos_t);

    return 0.5 * (prl * prl + ppd * ppd);
}

vec3 frame_sample(vec3 wi, vec3 hit_normal) {
    vec3 axis = vec3(hit_normal.y, -hit_normal.x, 0.0);
    float axis_len_sq = dot(axis, axis);

    if (axis_len_sq < 0.0001) {
        return hit_normal.z < 0.0 ? vec3(wi.x, -wi.y, -wi.z) : wi;
    }

    float cos_t = hit_normal.z;
    float sin_t = sqrt(axis_len_sq);
    vec3 r = axis / sin_t;

    return wi * cos_t + cross(wi, r) * sin_t + r * dot(r, wi) * (1.0 - cos_t);
}
