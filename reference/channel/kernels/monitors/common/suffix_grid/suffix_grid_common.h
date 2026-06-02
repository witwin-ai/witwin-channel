#pragma once

#include <common/grid_ops.h>

namespace witwin::channel::native_ext::suffix_grid_detail {

constexpr float SG_PI = 3.14159265358979323846f;
constexpr float SG_EPS = 1.0e-8f;
constexpr float SG_MIN_DIST = 1.0e-2f;

using common::Complexf;
using common::Vec3f;
using common::add;
using common::cadd;
using common::clamp_index;
using common::cmul;
using common::cscale;
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
    return common::safe_unit(a, SG_EPS);
}

WITWIN_KERNEL_DINLINE Complexf distance_factor(float wavelength, float k, float d) {
    return common::distance_factor(SG_PI, SG_MIN_DIST, wavelength, k, d);
}

WITWIN_KERNEL_DINLINE Complexf distance_factor_tangent(
    float wavelength,
    float k,
    float d,
    float d_tangent
) {
    return common::distance_factor_tangent(SG_PI, SG_MIN_DIST, wavelength, k, d, d_tangent);
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
    return common::factor_distance_adjoint(SG_PI, SG_MIN_DIST, wavelength, k, d, grad_factor);
}

} // namespace witwin::channel::native_ext::suffix_grid_detail
