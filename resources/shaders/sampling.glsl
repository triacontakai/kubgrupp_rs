vec4 sample_cosine_hemisphere(float u, float v) {
    float theta = acos(sqrt(u));
    float phi = 2 * PI * v;

    return vec4(sin(theta) * cos(phi), sin(theta) * sin(phi), cos(theta), cos(theta)/PI);
}

vec3 sample_beckmann(float a, float u, float v) {
    float cos_t = sqrt(1.0 / (1.0 - a * a * log(u)));
    float sin_t = sqrt(1.0 - cos_t * cos_t);

    float phi = 2 * PI * v;
    return vec3(sin_t * cos(phi), sin_t * sin(phi), cos_t);
}

float pdf_beckmann(float cos_t, float a) {
    float tan_t_2 = 1 / (cos_t * cos_t) - 1;
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
    // we need to convert the wi sample to be relative to the normal
    // do this by creating a rotation from (0, 0, 1) to the normal
    vec3 axis = vec3(hit_normal.y, -hit_normal.x, 0);
    if (length(axis) < 0.0001) {
        return wi;
    } else {
        float cos_t = hit_normal.z;
        float sin_t = length(axis);
        vec3 r = normalize(axis);
        return wi * cos_t
            + cross(wi, r) * sin_t
            + r * dot(r, wi) * (1 - cos_t);
    }
}
