struct Offsets {
    uint brdf_i;
};

struct Light {
    uint type;
    vec3 color;
    vec3 position;
    vec3 vertices[3];
};

struct Vertex {
    vec3 position;
    vec3 normal;
};

const uint EMITTER_TYPE_POINT = 0;
const uint EMITTER_TYPE_AREA = 1;

//layout(scalar, set = 0, binding = 2) readonly buffer Vertices {
//    Vertex vertices[];
//} vertices;
//
//layout(scalar, set = 0, binding = 3) readonly buffer Lights {
//    Light lights[];
//} lights;
//
//layout(scalar, set = 0, binding = 4) readonly buffer InstanceOffsets {
//    Offsets offsets[];
//} offsets;
//
//layout(scalar, set = 0, binding = 5) readonly buffer Fields {
//    BrdfParams params[];
//} instance_info;
