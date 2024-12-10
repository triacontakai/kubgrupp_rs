struct Vertex {
    vec3 position;
    vec3 normal;
};

struct Light {
    uint type;
    vec3 color;
    vec3 position;
    vec3 vertices[3];
};

struct Offsets {
    uint brdf_i;
};

const uint EMITTER_TYPE_POINT = 0;
const uint EMITTER_TYPE_AREA = 1;

#extension GL_EXT_scalar_block_layout : enable

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

#define BRDF_PARAMS_BINDING 6