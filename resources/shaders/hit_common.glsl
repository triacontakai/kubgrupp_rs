struct Vertex {
    vec3 position;
    vec3 normal;
};

struct Light {
    uint type;
    vec3 color;
    vec3 position;
    vec3 data[3]; // area light: data = vertices, directional light: data[0] = direction and data[1].r = radius
};

struct Offsets {
    uint brdf_i;
};

const uint EMITTER_TYPE_POINT = 0;
const uint EMITTER_TYPE_AREA = 1;
const uint EMITTER_TYPE_DIRECTIONAL = 2;

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