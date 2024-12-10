layout(set = 0, binding = 0) writeonly uniform image2D image;
layout(set = 0, binding = 1, rgba32f) uniform image2D accum_image;
layout(set = 0, binding = 2) uniform accelerationStructureEXT tlas;
layout(push_constant) uniform Constants {
    mat4 view_inverse;
    mat4 proj_inverse;
    uvec2 seed_offset;
    uint frame;
};
