struct MeshHitInfo {
    vec3 position;
    vec3 normal;
    vec3 geo_normal;
    bool is_backface;
};

MeshHitInfo compute_mesh_hit(vec2 bary_coord) {
    MeshHitInfo info;

    Vertex a = vertices.vertices[gl_InstanceCustomIndexEXT + 3*gl_PrimitiveID];
    Vertex b = vertices.vertices[gl_InstanceCustomIndexEXT + 3*gl_PrimitiveID + 1];
    Vertex c = vertices.vertices[gl_InstanceCustomIndexEXT + 3*gl_PrimitiveID + 2];

    vec3 bary = vec3(1.0 - bary_coord.x - bary_coord.y, bary_coord);

    vec3 local_pos = a.position * bary.x + b.position * bary.y + c.position * bary.z;
    info.position = vec3(gl_ObjectToWorldEXT * vec4(local_pos, 1.0));

    vec3 local_normal = a.normal * bary.x + b.normal * bary.y + c.normal * bary.z;
    info.normal = normalize(vec3(gl_ObjectToWorldEXT * vec4(local_normal, 0.0)));

    vec3 edge1 = b.position - a.position;
    vec3 edge2 = c.position - a.position;
    vec3 face_normal = normalize(cross(edge1, edge2));
    info.geo_normal = normalize(vec3(gl_ObjectToWorldEXT * vec4(face_normal, 0.0)));

    info.is_backface = dot(gl_WorldRayDirectionEXT, info.geo_normal) > 0.0;
    if (info.is_backface) {
        info.normal = -info.normal;
        info.geo_normal = -info.geo_normal;
    }

    return info;
}

vec3 compute_mesh_hit_position(vec2 bary_coord) {
    Vertex a = vertices.vertices[gl_InstanceCustomIndexEXT + 3*gl_PrimitiveID];
    Vertex b = vertices.vertices[gl_InstanceCustomIndexEXT + 3*gl_PrimitiveID + 1];
    Vertex c = vertices.vertices[gl_InstanceCustomIndexEXT + 3*gl_PrimitiveID + 2];

    vec3 bary = vec3(1.0 - bary_coord.x - bary_coord.y, bary_coord);
    vec3 local_pos = a.position * bary.x + b.position * bary.y + c.position * bary.z;
    return vec3(gl_ObjectToWorldEXT * vec4(local_pos, 1.0));
}

