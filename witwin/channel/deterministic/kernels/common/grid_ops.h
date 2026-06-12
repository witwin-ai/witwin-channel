#pragma once

#include <common/primitives.h>

namespace witwin::channel::native_ext::common {

WITWIN_KERNEL_DINLINE int clamp_index(int value, int hi) {
    if (value < 0) {
        return 0;
    }
    if (value > hi) {
        return hi;
    }
    return value;
}

WITWIN_KERNEL_DINLINE void tangential_components(
    int plane_axis,
    Vec3f value,
    float& coord_0,
    float& coord_1
) {
    if (plane_axis == 0) {
        coord_0 = value.y;
        coord_1 = value.z;
    } else if (plane_axis == 1) {
        coord_0 = value.x;
        coord_1 = value.z;
    } else {
        coord_0 = value.x;
        coord_1 = value.y;
    }
}

WITWIN_KERNEL_DINLINE void tangential_components(
    int plane_axis,
    Vec3f value,
    float& coord_0,
    float& coord_1,
    float& normal_coord
) {
    tangential_components(plane_axis, value, coord_0, coord_1);
    if (plane_axis == 0) {
        normal_coord = value.x;
    } else if (plane_axis == 1) {
        normal_coord = value.y;
    } else {
        normal_coord = value.z;
    }
}

WITWIN_KERNEL_DINLINE Vec3f point_on_plane(
    int plane_axis,
    float plane_position,
    float tangential_0,
    float tangential_1
) {
    if (plane_axis == 0) {
        return make_vec3(plane_position, tangential_0, tangential_1);
    }
    if (plane_axis == 1) {
        return make_vec3(tangential_0, plane_position, tangential_1);
    }
    return make_vec3(tangential_0, tangential_1, plane_position);
}

WITWIN_KERNEL_DINLINE int grid_cell_index(
    float coord_0,
    float coord_1,
    float coord_0_min,
    float coord_1_min,
    float cell_size_0,
    float cell_size_1,
    int n_coord_0,
    int n_coord_1
) {
    int ix = clamp_index(static_cast<int>((coord_0 - coord_0_min) / cell_size_0), n_coord_0 - 1);
    int iy = clamp_index(static_cast<int>((coord_1 - coord_1_min) / cell_size_1), n_coord_1 - 1);
    return iy * n_coord_0 + ix;
}

// Reduce k*d in double precision: the f32 product loses ~k*d*2^-24 of phase,
// which matters for coherent sums at mmWave ranges.
WITWIN_KERNEL_DINLINE float reduced_neg_kd(float k, float d) {
    return -(float)fmod((double)k * (double)d, 6.283185307179586476925287);
}

WITWIN_KERNEL_DINLINE Complexf distance_factor(
    float pi,
    float min_dist,
    float wavelength,
    float k,
    float d
) {
    float safe_d = fmaxf(d, min_dist);
    float fspl = (wavelength / (4.0f * pi)) / safe_d;
    float phase = reduced_neg_kd(k, d);
    return make_complex(fspl * cosf(phase), fspl * sinf(phase));
}

WITWIN_KERNEL_DINLINE Complexf distance_factor_tangent(
    float pi,
    float min_dist,
    float wavelength,
    float k,
    float d,
    float d_tangent
) {
    float safe_d = fmaxf(d, min_dist);
    float fspl = (wavelength / (4.0f * pi)) / safe_d;
    float dfspl_dd = d > min_dist ? -(wavelength / (4.0f * pi)) / (safe_d * safe_d) : 0.0f;
    float phase = reduced_neg_kd(k, d);
    float c = cosf(phase);
    float s = sinf(phase);
    float dphase_dd = -k;
    float dg_re_dd = dfspl_dd * c - fspl * s * dphase_dd;
    float dg_im_dd = dfspl_dd * s + fspl * c * dphase_dd;
    return make_complex(dg_re_dd * d_tangent, dg_im_dd * d_tangent);
}

WITWIN_KERNEL_DINLINE void accumulate_complex_backward(
    Complexf input,
    Complexf factor,
    Complexf grad_out,
    Complexf& grad_input,
    Complexf& grad_factor
) {
    grad_input.re += grad_out.re * factor.re + grad_out.im * factor.im;
    grad_input.im += -grad_out.re * factor.im + grad_out.im * factor.re;
    grad_factor.re += grad_out.re * input.re + grad_out.im * input.im;
    grad_factor.im += -grad_out.re * input.im + grad_out.im * input.re;
}

WITWIN_KERNEL_DINLINE float factor_distance_adjoint(
    float pi,
    float min_dist,
    float wavelength,
    float k,
    float d,
    Complexf grad_factor
) {
    float safe_d = fmaxf(d, min_dist);
    float fspl = (wavelength / (4.0f * pi)) / safe_d;
    float dfspl_dd = d > min_dist ? -(wavelength / (4.0f * pi)) / (safe_d * safe_d) : 0.0f;
    float phase = reduced_neg_kd(k, d);
    float c = cosf(phase);
    float s = sinf(phase);
    float dphase_dd = -k;
    float dg_re_dd = dfspl_dd * c - fspl * s * dphase_dd;
    float dg_im_dd = dfspl_dd * s + fspl * c * dphase_dd;
    return grad_factor.re * dg_re_dd + grad_factor.im * dg_im_dd;
}

} // namespace witwin::channel::native_ext::common

