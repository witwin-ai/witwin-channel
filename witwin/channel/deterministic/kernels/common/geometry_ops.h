#pragma once

#include <common/primitives.h>

namespace witwin::channel::native_ext::common {

WITWIN_KERNEL_DINLINE bool point_in_triangle_3d(
    Vec3f p,
    Vec3f v0,
    Vec3f v1,
    Vec3f v2,
    float bary_eps,
    float tri_eps
) {
    Vec3f edge1 = sub(v1, v0);
    Vec3f edge2 = sub(v2, v0);
    Vec3f vp = sub(p, v0);

    float dot00 = dot(edge1, edge1);
    float dot01 = dot(edge1, edge2);
    float dot02 = dot(edge1, vp);
    float dot11 = dot(edge2, edge2);
    float dot12 = dot(edge2, vp);

    float inv_denom = 1.0f / (dot00 * dot11 - dot01 * dot01 + bary_eps);
    float u = (dot11 * dot02 - dot01 * dot12) * inv_denom;
    float v = (dot00 * dot12 - dot01 * dot02) * inv_denom;

    return u >= -tri_eps && v >= -tri_eps && (u + v) <= (1.0f + tri_eps);
}

WITWIN_KERNEL_DINLINE bool surface_contains_point(
    Vec3f p,
    int prim_idx,
    const float* tri_v0_x,
    const float* tri_v0_y,
    const float* tri_v0_z,
    const float* tri_v1_x,
    const float* tri_v1_y,
    const float* tri_v1_z,
    const float* tri_v2_x,
    const float* tri_v2_y,
    const float* tri_v2_z,
    const int* tri_group_size,
    const int* tri_group_members,
    int max_group_size,
    float bary_eps,
    float tri_eps
) {
    if (prim_idx < 0) {
        return false;
    }

    if (tri_group_size == nullptr || tri_group_members == nullptr || max_group_size <= 0) {
        Vec3f v0 = make_vec3(tri_v0_x[prim_idx], tri_v0_y[prim_idx], tri_v0_z[prim_idx]);
        Vec3f v1 = make_vec3(tri_v1_x[prim_idx], tri_v1_y[prim_idx], tri_v1_z[prim_idx]);
        Vec3f v2 = make_vec3(tri_v2_x[prim_idx], tri_v2_y[prim_idx], tri_v2_z[prim_idx]);
        return point_in_triangle_3d(p, v0, v1, v2, bary_eps, tri_eps);
    }

    int group_size = tri_group_size[prim_idx];
    for (int slot = 0; slot < max_group_size; ++slot) {
        if (slot >= group_size) {
            break;
        }
        int member_idx = tri_group_members[prim_idx * max_group_size + slot];
        if (member_idx >= 0) {
            Vec3f v0 = make_vec3(tri_v0_x[member_idx], tri_v0_y[member_idx], tri_v0_z[member_idx]);
            Vec3f v1 = make_vec3(tri_v1_x[member_idx], tri_v1_y[member_idx], tri_v1_z[member_idx]);
            Vec3f v2 = make_vec3(tri_v2_x[member_idx], tri_v2_y[member_idx], tri_v2_z[member_idx]);
            if (point_in_triangle_3d(p, v0, v1, v2, bary_eps, tri_eps)) {
                return true;
            }
        }
    }
    return false;
}

} // namespace witwin::channel::native_ext::common
