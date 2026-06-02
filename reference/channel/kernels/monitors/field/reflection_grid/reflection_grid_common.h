#pragma once

#include <common/geometry_ops.h>
#include <common/grid_ops.h>

namespace witwin::channel::native_ext::reflection_grid_detail {

constexpr float RG_PI = 3.14159265358979323846f;
constexpr float RG_EPS = 1.0e-8f;
constexpr float RG_MIN_DIST = 1.0e-2f;
constexpr float RG_BARY_EPS = 1.0e-10f;
constexpr float RG_TRI_EPS = 1.0e-8f;

using common::Complexf;
using common::Vec3f;
using common::add;
using common::cadd;
using common::clamp_index;
using common::cmul;
using common::cross;
using common::dot;
using common::grid_cell_index;
using common::make_complex;
using common::make_vec3;
using common::mul;
using common::norm;
using common::point_on_plane;
using common::sub;
using common::tangential_components;

WITWIN_KERNEL_DINLINE Vec3f safe_unit(Vec3f a) {
    return common::safe_unit(a, RG_EPS);
}

WITWIN_KERNEL_DINLINE Complexf distance_factor(float wavelength, float k, float d) {
    return common::distance_factor(RG_PI, RG_MIN_DIST, wavelength, k, d);
}

WITWIN_KERNEL_DINLINE Complexf distance_factor_tangent(
    float wavelength,
    float k,
    float d,
    float d_tangent
) {
    return common::distance_factor_tangent(RG_PI, RG_MIN_DIST, wavelength, k, d, d_tangent);
}

WITWIN_KERNEL_DINLINE void accumulate_complex_backward(
    Complexf input,
    Complexf factor,
    Complexf grad_out,
    Complexf& grad_input,
    Complexf& grad_factor
) {
    common::accumulate_complex_backward(input, factor, grad_out, grad_input, grad_factor);
}

WITWIN_KERNEL_DINLINE float factor_distance_adjoint(
    float wavelength,
    float k,
    float d,
    Complexf grad_factor
) {
    return common::factor_distance_adjoint(RG_PI, RG_MIN_DIST, wavelength, k, d, grad_factor);
}

WITWIN_KERNEL_DINLINE Vec3f mirror_point(Vec3f prev_tx, Vec3f prev_refl_p, Vec3f prev_refl_n) {
    float d_to_plane = dot(sub(prev_tx, prev_refl_p), prev_refl_n);
    return sub(prev_tx, mul(prev_refl_n, 2.0f * d_to_plane));
}

WITWIN_KERNEL_DINLINE Vec3f mirror_tangent(
    Vec3f prev_tx,
    Vec3f prev_refl_p,
    Vec3f prev_refl_n,
    Vec3f t_prev_tx,
    Vec3f t_prev_refl_p,
    Vec3f t_prev_refl_n
) {
    Vec3f a = sub(prev_tx, prev_refl_p);
    Vec3f a_t = sub(t_prev_tx, t_prev_refl_p);
    float d_to_plane = dot(a, prev_refl_n);
    float d_to_plane_t = dot(a_t, prev_refl_n) + dot(a, t_prev_refl_n);
    Vec3f correction = add(mul(prev_refl_n, d_to_plane_t), mul(t_prev_refl_n, d_to_plane));
    return sub(t_prev_tx, mul(correction, 2.0f));
}

WITWIN_KERNEL_DINLINE void mirror_backward(
    Vec3f grad_mirror,
    Vec3f prev_tx,
    Vec3f prev_refl_p,
    Vec3f prev_refl_n,
    Vec3f& grad_prev_tx,
    Vec3f& grad_prev_refl_p,
    Vec3f& grad_prev_refl_n
) {
    Vec3f a = sub(prev_tx, prev_refl_p);
    float d_to_plane = dot(a, prev_refl_n);

    grad_prev_tx = add(grad_prev_tx, grad_mirror);

    float grad_d_to_plane = -2.0f * dot(grad_mirror, prev_refl_n);
    grad_prev_refl_n = add(grad_prev_refl_n, mul(grad_mirror, -2.0f * d_to_plane));
    grad_prev_refl_n = add(grad_prev_refl_n, mul(a, grad_d_to_plane));

    Vec3f grad_a = mul(prev_refl_n, grad_d_to_plane);
    grad_prev_tx = add(grad_prev_tx, grad_a);
    grad_prev_refl_p = sub(grad_prev_refl_p, grad_a);
}

WITWIN_KERNEL_DINLINE bool point_in_triangle_3d(Vec3f p, Vec3f v0, Vec3f v1, Vec3f v2) {
    return common::point_in_triangle_3d(p, v0, v1, v2, RG_BARY_EPS, RG_TRI_EPS);
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
    int max_group_size
) {
    return common::surface_contains_point(
        p,
        prim_idx,
        tri_v0_x,
        tri_v0_y,
        tri_v0_z,
        tri_v1_x,
        tri_v1_y,
        tri_v1_z,
        tri_v2_x,
        tri_v2_y,
        tri_v2_z,
        tri_group_size,
        tri_group_members,
        max_group_size,
        RG_BARY_EPS,
        RG_TRI_EPS
    );
}

WITWIN_KERNEL_DINLINE bool validate_reflection_path(
    int validate_paths,
    int has_mesh_data,
    Vec3f cell_pos,
    Vec3f mirror,
    Vec3f prev_refl_p,
    Vec3f prev_refl_n,
    int prev_prim_idx,
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
    int max_group_size
) {
    if (!validate_paths || !has_mesh_data) {
        return true;
    }

    Vec3f dir_to_rx = sub(cell_pos, mirror);
    float denom = dot(dir_to_rx, prev_refl_n);
    float t_intersect = dot(sub(prev_refl_p, mirror), prev_refl_n) / (denom + RG_EPS);
    Vec3f int_p = add(mirror, mul(dir_to_rx, t_intersect));
    return t_intersect > 0.0f
        && t_intersect < 1.0f
        && fabsf(denom) > RG_EPS
        && surface_contains_point(
            int_p,
            prev_prim_idx,
            tri_v0_x,
            tri_v0_y,
            tri_v0_z,
            tri_v1_x,
            tri_v1_y,
            tri_v1_z,
            tri_v2_x,
            tri_v2_y,
            tri_v2_z,
            tri_group_size,
            tri_group_members,
            max_group_size
        );
}

} // namespace witwin::channel::native_ext::reflection_grid_detail
