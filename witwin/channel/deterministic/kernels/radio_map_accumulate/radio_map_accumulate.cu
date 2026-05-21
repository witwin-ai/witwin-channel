#include <cuda_runtime.h>
#include <climits>

#include <common/cuda_check.h>
#include <common/primitives.h>
#include <radio_map_accumulate/radio_map_accumulate.h>
#include <utd/utd_math.h>

namespace witwin::channel::native_ext {
namespace {

using common::Complexf;
using common::Vec3f;
using common::add;
using common::cadd;
using common::ceil_div_int;
using common::cmul;
using common::cscale;
using common::dot;
using common::make_complex;
using common::make_vec3;
using common::mul;
using common::norm;
using common::sub;
using common::throw_cuda;

constexpr float RM_EPS = 1.0e-10f;
constexpr float RM_SMALL_EPS = 1.0e-6f;
constexpr float RM_TRANSITION_EPS = 1.0e-12f;

__device__ __forceinline__ Vec3f safe_normalize_with_fallback(Vec3f value, Vec3f fallback) {
    float value_norm = norm(value);
    float fallback_norm = norm(fallback);
    Vec3f fallback_unit = (
        fallback_norm > RM_EPS
        ? mul(fallback, 1.0f / (fallback_norm + RM_EPS))
        : make_vec3(0.0f, 0.0f, 0.0f)
    );
    return (
        value_norm > RM_SMALL_EPS
        ? mul(value, 1.0f / (value_norm + RM_EPS))
        : fallback_unit
    );
}

__device__ __forceinline__ Vec3f stable_perpendicular_basis(Vec3f ray_dir, Vec3f preferred) {
    Vec3f proj = sub(preferred, mul(ray_dir, dot(preferred, ray_dir)));
    Vec3f alt_axis = (
        fabsf(ray_dir.z) < 0.9f
        ? make_vec3(0.0f, 0.0f, 1.0f)
        : make_vec3(0.0f, 1.0f, 0.0f)
    );
    Vec3f alt_proj = sub(alt_axis, mul(ray_dir, dot(alt_axis, ray_dir)));
    return safe_normalize_with_fallback(proj, alt_proj);
}

__device__ __forceinline__ Complexf complex_dot_real(
    Complexf x,
    Complexf y,
    Complexf z,
    Vec3f basis
) {
    return make_complex(
        x.re * basis.x + y.re * basis.y + z.re * basis.z,
        x.im * basis.x + y.im * basis.y + z.im * basis.z
    );
}

__device__ __forceinline__ float complex_abs_sqr(Complexf value) {
    return value.re * value.re + value.im * value.im;
}

__device__ __forceinline__ Complexf csub(Complexf a, Complexf b) {
    return make_complex(a.re - b.re, a.im - b.im);
}

__device__ __forceinline__ Complexf cconj(Complexf a) {
    return make_complex(a.re, -a.im);
}

__device__ __forceinline__ float complex_real_inner(Complexf grad_value, Complexf primal_value) {
    return grad_value.re * primal_value.re + grad_value.im * primal_value.im;
}

__device__ __forceinline__ void accumulate_vector_from_complex_grad(
    Complexf grad_value,
    Complexf coeff,
    float basis_component,
    Complexf* grad_coeff,
    float* grad_basis_component
) {
    grad_coeff->re += grad_value.re * basis_component;
    grad_coeff->im += grad_value.im * basis_component;
    *grad_basis_component += complex_real_inner(grad_value, coeff);
}

__device__ __forceinline__ void accumulate_complex_mul_backward(
    Complexf grad_output,
    Complexf lhs,
    Complexf rhs,
    Complexf* grad_lhs,
    Complexf* grad_rhs
) {
    *grad_lhs = cadd(*grad_lhs, cmul(grad_output, cconj(rhs)));
    *grad_rhs = cadd(*grad_rhs, cmul(cconj(lhs), grad_output));
}

struct MatchedIsbPrimal {
    Complexf smooth_coeff;
    Complexf hard_direct;
    Complexf raw_direct;
    Complexf excess;
    Complexf completion;
    Complexf coherent;
    Complexf vector_x;
    Complexf vector_y;
    Complexf vector_z;
    float scalar_factor;
    float power;
    bool completion_active;
};

__device__ __forceinline__ MatchedIsbPrimal compute_matched_isb_primal(
    Complexf continued_direct,
    Vec3f tx_basis,
    Vec3f rx_basis,
    float hard_visibility,
    bool interior_mask,
    float incident_weight,
    Complexf incident_response,
    Complexf raw_x,
    Complexf raw_y,
    Complexf raw_z
) {
    MatchedIsbPrimal primal{};
    float side_sign = hard_visibility > 0.0f ? 1.0f : -1.0f;
    primal.smooth_coeff = make_complex(
        0.5f * (1.0f + side_sign * incident_response.re),
        0.5f * (side_sign * incident_response.im)
    );
    primal.hard_direct = cscale(continued_direct, hard_visibility);
    primal.raw_direct = complex_dot_real(raw_x, raw_y, raw_z, tx_basis);
    primal.excess = csub(primal.raw_direct, primal.hard_direct);
    primal.completion = csub(
        csub(cmul(primal.smooth_coeff, continued_direct), primal.hard_direct),
        cscale(primal.excess, incident_weight)
    );
    primal.completion_active = !interior_mask;
    if (!primal.completion_active) {
        primal.completion = make_complex(0.0f, 0.0f);
    }
    primal.scalar_factor = dot(tx_basis, rx_basis);
    primal.coherent = cscale(primal.completion, primal.scalar_factor);
    primal.vector_x = cscale(primal.completion, tx_basis.x);
    primal.vector_y = cscale(primal.completion, tx_basis.y);
    primal.vector_z = cscale(primal.completion, tx_basis.z);
    primal.power = complex_abs_sqr(primal.vector_x)
        + complex_abs_sqr(primal.vector_y)
        + complex_abs_sqr(primal.vector_z);
    return primal;
}

struct ShadowFiniteFactorPrimal {
    float3a edge_hat;
    float source_axial;
    float target_axial;
    float s_prime_proj;
    float s_proj;
    float stationary_u;
    float source_offset;
    float target_offset;
    float source_range;
    float target_range;
    float curvature;
    float scale;
    Complex factor;
    float factor_magnitude;
    float factor_scale;
};

struct IncidentStatsPairPrimal {
    bool active;
    float phi;
    float phi_prime;
    float s;
    float s_prime;
    float kL;
    float a;
    float da_dif_phi;
    Complex transition;
    Complex transition_first;
    float transition_magnitude;
    float base_weight;
    Complex response;
    float weight;
    ShadowFiniteFactorPrimal finite_factor;
};

__device__ __forceinline__ float3a load_f3(
    const float* x,
    const float* y,
    const float* z,
    int idx
) {
    return make_f3(x[idx], y[idx], z[idx]);
}

__device__ __forceinline__ void atomic_max_nonnegative(float* dst, float value) {
    if (value <= 0.0f) {
        return;
    }
    atomicMax(reinterpret_cast<int*>(dst), __float_as_int(value));
}

__global__ void fill_int_kernel(
    int* __restrict__ values,
    int count,
    int value
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= count) {
        return;
    }
    values[tid] = value;
}

__device__ __forceinline__ float tangent_of_length(float3a value, float3a tangent) {
    float value_norm = safe_length(value);
    if (value_norm <= UTD_SMALL_EPS) {
        return 0.0f;
    }
    return f3_dot(value, tangent) / value_norm;
}

__device__ __forceinline__ float3a project_to_wedge_plane_tangent(float3a tangent, float3a edge_hat) {
    return project_to_wedge_plane(tangent, edge_hat);
}

__device__ __forceinline__ ShadowFiniteFactorPrimal compute_shadow_boundary_finite_factor_primal(
    float3a tx_pos,
    float3a rx_pos,
    float3a edge_pos,
    float3a edge_hat,
    float edge_line_min,
    float edge_line_max,
    float k,
    bool stationary_at_origin
) {
    ShadowFiniteFactorPrimal primal{};
    primal.edge_hat = edge_hat;
    float3a source_to_edge = f3_sub(edge_pos, tx_pos);
    float3a edge_to_target = f3_sub(rx_pos, edge_pos);
    float3a source_proj = project_to_wedge_plane(source_to_edge, edge_hat);
    float3a target_proj = project_to_wedge_plane(edge_to_target, edge_hat);
    primal.source_axial = f3_dot(f3_sub(tx_pos, edge_pos), edge_hat);
    primal.target_axial = f3_dot(f3_sub(rx_pos, edge_pos), edge_hat);
    primal.s_prime_proj = safe_length(source_proj) + UTD_EPS;
    primal.s_proj = safe_length(target_proj) + UTD_EPS;
    float stationary_u =
        (primal.s_prime_proj * primal.target_axial + primal.s_proj * primal.source_axial)
        / (primal.s_proj + primal.s_prime_proj + UTD_EPS);
    primal.stationary_u = stationary_at_origin ? 0.0f : stationary_u;
    primal.source_offset = primal.stationary_u - primal.source_axial;
    primal.target_offset = primal.target_axial - primal.stationary_u;
    primal.source_range = sqrtf(
        primal.s_prime_proj * primal.s_prime_proj + primal.source_offset * primal.source_offset + UTD_EPS
    );
    primal.target_range = sqrtf(
        primal.s_proj * primal.s_proj + primal.target_offset * primal.target_offset + UTD_EPS
    );
    primal.curvature =
        primal.s_prime_proj * primal.s_prime_proj
            / (primal.source_range * primal.source_range * primal.source_range + UTD_EPS)
        + primal.s_proj * primal.s_proj
            / (primal.target_range * primal.target_range * primal.target_range + UTD_EPS);
    primal.scale = sqrtf(fmaxf(k * primal.curvature, UTD_EPS) / UTD_PI);
    Complex f_min, f_min_first, f_min_second;
    Complex f_max, f_max_first, f_max_second;
    fresnel_boersma(primal.scale * (edge_line_min - primal.stationary_u), f_min, f_min_first, f_min_second);
    fresnel_boersma(primal.scale * (edge_line_max - primal.stationary_u), f_max, f_max_first, f_max_second);
    Complex delta = cplx_sub(f_max, f_min);
    primal.factor = cplx_mul(cplx(0.5f, 0.5f), cplx_conj(delta));
    primal.factor_magnitude = sqrtf(fmaxf(cplx_abs_sqr(primal.factor), 0.0f));
    primal.factor_scale = fminf(primal.factor_magnitude, 1.0f);
    return primal;
}

__device__ __forceinline__ void shadow_boundary_finite_factor_jvp(
    float3a tx_pos,
    float3a rx_pos,
    float3a edge_pos,
    float3a edge_hat,
    float edge_line_min,
    float edge_line_max,
    float k,
    float3a t_tx_pos,
    float3a t_rx_pos,
    float3a t_edge_pos,
    const ShadowFiniteFactorPrimal& primal,
    Complex& t_factor,
    float& t_factor_scale
) {
    float3a source_to_edge = f3_sub(edge_pos, tx_pos);
    float3a edge_to_target = f3_sub(rx_pos, edge_pos);
    float3a t_source_to_edge = f3_sub(t_edge_pos, t_tx_pos);
    float3a t_edge_to_target = f3_sub(t_rx_pos, t_edge_pos);
    float3a source_proj = project_to_wedge_plane(source_to_edge, edge_hat);
    float3a target_proj = project_to_wedge_plane(edge_to_target, edge_hat);
    float3a t_source_proj = project_to_wedge_plane_tangent(t_source_to_edge, edge_hat);
    float3a t_target_proj = project_to_wedge_plane_tangent(t_edge_to_target, edge_hat);
    float source_proj_norm = safe_length(source_proj);
    float target_proj_norm = safe_length(target_proj);
    float t_s_prime_proj = source_proj_norm > UTD_SMALL_EPS
        ? f3_dot(source_proj, t_source_proj) / source_proj_norm
        : 0.0f;
    float t_s_proj = target_proj_norm > UTD_SMALL_EPS
        ? f3_dot(target_proj, t_target_proj) / target_proj_norm
        : 0.0f;
    float t_source_axial = f3_dot(f3_sub(t_tx_pos, t_edge_pos), edge_hat);
    float t_target_axial = f3_dot(f3_sub(t_rx_pos, t_edge_pos), edge_hat);
    float denom = primal.s_proj + primal.s_prime_proj + UTD_EPS;
    float numer = primal.s_prime_proj * primal.target_axial + primal.s_proj * primal.source_axial;
    float t_numer =
        t_s_prime_proj * primal.target_axial
        + primal.s_prime_proj * t_target_axial
        + t_s_proj * primal.source_axial
        + primal.s_proj * t_source_axial;
    float t_denom = t_s_proj + t_s_prime_proj;
    float t_stationary_u = (t_numer * denom - numer * t_denom) / (denom * denom);
    float t_source_offset = t_stationary_u - t_source_axial;
    float t_target_offset = t_target_axial - t_stationary_u;
    float t_source_range = (
        primal.s_prime_proj * t_s_prime_proj + primal.source_offset * t_source_offset
    ) / primal.source_range;
    float t_target_range = (
        primal.s_proj * t_s_proj + primal.target_offset * t_target_offset
    ) / primal.target_range;
    float source_denom = primal.source_range * primal.source_range * primal.source_range + UTD_EPS;
    float target_denom = primal.target_range * primal.target_range * primal.target_range + UTD_EPS;
    float t_curvature =
        (2.0f * primal.s_prime_proj * t_s_prime_proj) / source_denom
        - (primal.s_prime_proj * primal.s_prime_proj)
            * (3.0f * primal.source_range * primal.source_range * t_source_range)
            / (source_denom * source_denom)
        + (2.0f * primal.s_proj * t_s_proj) / target_denom
        - (primal.s_proj * primal.s_proj)
            * (3.0f * primal.target_range * primal.target_range * t_target_range)
            / (target_denom * target_denom);
    float t_scale = 0.0f;
    if (k * primal.curvature > UTD_EPS && primal.scale > UTD_SMALL_EPS) {
        t_scale = 0.5f * k * t_curvature / (UTD_PI * primal.scale);
    }
    float x_min = primal.scale * (edge_line_min - primal.stationary_u);
    float x_max = primal.scale * (edge_line_max - primal.stationary_u);
    float t_x_min = t_scale * (edge_line_min - primal.stationary_u) - primal.scale * t_stationary_u;
    float t_x_max = t_scale * (edge_line_max - primal.stationary_u) - primal.scale * t_stationary_u;
    Complex f_min, f_min_first, f_min_second;
    Complex f_max, f_max_first, f_max_second;
    fresnel_boersma(x_min, f_min, f_min_first, f_min_second);
    fresnel_boersma(x_max, f_max, f_max_first, f_max_second);
    Complex t_delta = cplx_sub(
        cplx_mul_real(f_max_first, t_x_max),
        cplx_mul_real(f_min_first, t_x_min)
    );
    t_factor = cplx_mul(cplx(0.5f, 0.5f), cplx_conj(t_delta));
    t_factor_scale = 0.0f;
    if (primal.factor_magnitude < 1.0f && primal.factor_magnitude > UTD_SMALL_EPS) {
        t_factor_scale = cplx_adj_dot(t_factor, primal.factor) / primal.factor_magnitude;
    }
}

__device__ __forceinline__ void shadow_boundary_finite_factor_vjp(
    float3a tx_pos,
    float3a rx_pos,
    float3a edge_pos,
    float3a edge_hat,
    float edge_line_min,
    float edge_line_max,
    float k,
    const ShadowFiniteFactorPrimal& primal,
    Complex grad_factor,
    float grad_factor_scale,
    float3a& grad_tx_pos,
    float3a& grad_rx_pos,
    float3a& grad_edge_pos
) {
    if (!cplx_any_nonzero(grad_factor) && grad_factor_scale == 0.0f) {
        return;
    }
    if (primal.factor_magnitude < 1.0f && primal.factor_magnitude > UTD_SMALL_EPS) {
        float scale = grad_factor_scale / primal.factor_magnitude;
        grad_factor.re += scale * primal.factor.re;
        grad_factor.im += scale * primal.factor.im;
    }
    Complex grad_delta_conj = cplx_mul(cplx_conj(cplx(0.5f, 0.5f)), grad_factor);
    Complex grad_delta = cplx_conj(grad_delta_conj);
    float x_min = primal.scale * (edge_line_min - primal.stationary_u);
    float x_max = primal.scale * (edge_line_max - primal.stationary_u);
    Complex f_min, f_min_first, f_min_second;
    Complex f_max, f_max_first, f_max_second;
    fresnel_boersma(x_min, f_min, f_min_first, f_min_second);
    fresnel_boersma(x_max, f_max, f_max_first, f_max_second);
    float grad_x_max = cplx_adj_dot(grad_delta, f_max_first);
    float grad_x_min = -cplx_adj_dot(grad_delta, f_min_first);
    float grad_scale = 0.0f;
    float grad_stationary_u = 0.0f;
    grad_scale += grad_x_max * (edge_line_max - primal.stationary_u);
    grad_stationary_u -= grad_x_max * primal.scale;
    grad_scale += grad_x_min * (edge_line_min - primal.stationary_u);
    grad_stationary_u -= grad_x_min * primal.scale;
    float grad_curvature = 0.0f;
    if (k * primal.curvature > UTD_EPS && primal.scale > UTD_SMALL_EPS) {
        grad_curvature += grad_scale * 0.5f * k / (UTD_PI * primal.scale);
    }
    float source_denom = primal.source_range * primal.source_range * primal.source_range + UTD_EPS;
    float target_denom = primal.target_range * primal.target_range * primal.target_range + UTD_EPS;
    float grad_s_prime_proj = grad_curvature * (2.0f * primal.s_prime_proj / source_denom);
    float grad_source_range =
        grad_curvature
        * (
            -3.0f * primal.s_prime_proj * primal.s_prime_proj
            * primal.source_range * primal.source_range
            / (source_denom * source_denom)
        );
    float grad_s_proj = grad_curvature * (2.0f * primal.s_proj / target_denom);
    float grad_target_range =
        grad_curvature
        * (
            -3.0f * primal.s_proj * primal.s_proj
            * primal.target_range * primal.target_range
            / (target_denom * target_denom)
        );
    float grad_source_offset = grad_source_range * (primal.source_offset / primal.source_range);
    grad_s_prime_proj += grad_source_range * (primal.s_prime_proj / primal.source_range);
    float grad_target_offset = grad_target_range * (primal.target_offset / primal.target_range);
    grad_s_proj += grad_target_range * (primal.s_proj / primal.target_range);
    grad_stationary_u += grad_source_offset;
    float grad_source_axial = -grad_source_offset;
    float grad_target_axial = grad_target_offset;
    grad_stationary_u -= grad_target_offset;
    float denom = primal.s_proj + primal.s_prime_proj + UTD_EPS;
    float numer = primal.s_prime_proj * primal.target_axial + primal.s_proj * primal.source_axial;
    float grad_numer = grad_stationary_u / denom;
    float grad_denom = -grad_stationary_u * numer / (denom * denom);
    grad_s_prime_proj += grad_numer * primal.target_axial;
    grad_target_axial += grad_numer * primal.s_prime_proj;
    grad_s_proj += grad_numer * primal.source_axial;
    grad_source_axial += grad_numer * primal.s_proj;
    grad_s_proj += grad_denom;
    grad_s_prime_proj += grad_denom;

    float3a source_to_edge = f3_sub(edge_pos, tx_pos);
    float3a edge_to_target = f3_sub(rx_pos, edge_pos);
    float3a source_proj = project_to_wedge_plane(source_to_edge, edge_hat);
    float3a target_proj = project_to_wedge_plane(edge_to_target, edge_hat);
    float source_proj_norm = safe_length(source_proj);
    float target_proj_norm = safe_length(target_proj);
    float3a grad_source_proj = f3_zero();
    float3a grad_target_proj = f3_zero();
    if (source_proj_norm > UTD_SMALL_EPS) {
        grad_source_proj = f3_add(
            grad_source_proj,
            f3_mul(source_proj, grad_s_prime_proj / source_proj_norm)
        );
    }
    if (target_proj_norm > UTD_SMALL_EPS) {
        grad_target_proj = f3_add(
            grad_target_proj,
            f3_mul(target_proj, grad_s_proj / target_proj_norm)
        );
    }
    float3a grad_source_to_edge = project_to_wedge_plane(grad_source_proj, edge_hat);
    float3a grad_edge_to_target = project_to_wedge_plane(grad_target_proj, edge_hat);
    grad_tx_pos = f3_add(grad_tx_pos, f3_mul(edge_hat, grad_source_axial));
    grad_edge_pos = f3_sub(grad_edge_pos, f3_mul(edge_hat, grad_source_axial));
    grad_rx_pos = f3_add(grad_rx_pos, f3_mul(edge_hat, grad_target_axial));
    grad_edge_pos = f3_sub(grad_edge_pos, f3_mul(edge_hat, grad_target_axial));
    grad_edge_pos = f3_add(grad_edge_pos, grad_source_to_edge);
    grad_tx_pos = f3_sub(grad_tx_pos, grad_source_to_edge);
    grad_rx_pos = f3_add(grad_rx_pos, grad_edge_to_target);
    grad_edge_pos = f3_sub(grad_edge_pos, grad_edge_to_target);
}

__device__ __forceinline__ IncidentStatsPairPrimal compute_shadow_boundary_incident_pair_primal(
    float3a tx_pos,
    float3a rx_pos,
    float3a edge_pos,
    float3a edge_dir,
    float3a n0,
    float3a nn,
    float wedge_n,
    float edge_line_min,
    float edge_line_max,
    bool source_visible,
    float k
) {
    IncidentStatsPairPrimal primal{};
    float3a edge_hat = safe_normalize(edge_dir, make_f3(0.0f, 0.0f, 1.0f));
    float edge_length = edge_line_max - edge_line_min;
    float3a edge_origin = f3_add(edge_pos, f3_mul(edge_hat, edge_line_min));
    float parameter = first_order_diffraction_parameter(tx_pos, rx_pos, edge_origin, edge_hat);
    bool selected_valid = (edge_length > UTD_SMALL_EPS) && isfinite(parameter);
    float safe_parameter = isfinite(parameter) ? parameter : 0.0f;
    float3a eval_edge_pos = f3_add(edge_origin, f3_mul(edge_hat, safe_parameter));
    float eval_edge_line_min = -safe_parameter;
    float eval_edge_line_max = edge_length - safe_parameter;
    primal.finite_factor = compute_shadow_boundary_finite_factor_primal(
        tx_pos,
        rx_pos,
        eval_edge_pos,
        edge_hat,
        eval_edge_line_min,
        eval_edge_line_max,
        k,
        true
    );
    float s_proj = 0.0f;
    float s_prime_proj = 0.0f;
    compute_edge_angles(tx_pos, eval_edge_pos, edge_hat, n0, rx_pos, primal.phi, primal.phi_prime, s_proj, s_prime_proj);
    primal.s_prime = safe_length(f3_sub(eval_edge_pos, tx_pos)) + UTD_EPS;
    primal.s = safe_length(f3_sub(rx_pos, eval_edge_pos)) + UTD_EPS;
    bool source_exterior = wedge_exterior_mask(f3_sub(tx_pos, eval_edge_pos), edge_hat, n0, nn);
    bool target_exterior = wedge_exterior_mask(f3_sub(rx_pos, eval_edge_pos), edge_hat, n0, nn);
    primal.active =
        source_visible
        && selected_valid
        && (wedge_n > 1.01f)
        && source_exterior
        && target_exterior
        && (primal.s_prime > UTD_MIN_DISTANCE)
        && (primal.s > UTD_MIN_DISTANCE);
    if (!primal.active) {
        primal.transition = cplx_zero();
        primal.transition_first = cplx_zero();
        primal.transition_magnitude = 0.0f;
        primal.base_weight = 0.0f;
        primal.response = cplx_zero();
        primal.weight = 0.0f;
        primal.kL = 0.0f;
        primal.a = 0.0f;
        primal.da_dif_phi = 0.0f;
        return primal;
    }
    primal.kL = k * primal.s * primal.s_prime / (primal.s + primal.s_prime);
    float dif_phi = primal.phi - primal.phi_prime;
    float two_n_pi = 2.0f * wedge_n * UTD_PI;
    float round_plus = roundf((dif_phi + UTD_PI) / two_n_pi);
    float phase_offset_plus = two_n_pi * round_plus - dif_phi;
    float cosine_plus = cosf(0.5f * phase_offset_plus);
    float a_plus = 2.0f * cosine_plus * cosine_plus;
    float round_minus = roundf((dif_phi - UTD_PI) / two_n_pi);
    float phase_offset_minus = two_n_pi * round_minus - dif_phi;
    float cosine_minus = cosf(0.5f * phase_offset_minus);
    float a_minus = 2.0f * cosine_minus * cosine_minus;
    if (a_plus <= a_minus) {
        primal.a = a_plus;
        primal.da_dif_phi = sinf(phase_offset_plus);
    } else {
        primal.a = a_minus;
        primal.da_dif_phi = sinf(phase_offset_minus);
    }
    Complex transition_second;
    primal.transition = cplx_zero();
    primal.transition_first = cplx_zero();
    transition_second = cplx_zero();
    f_utd_with_derivatives(primal.kL * primal.a, primal.transition, primal.transition_first, transition_second);
    (void) transition_second;
    primal.transition_magnitude = sqrtf(fmaxf(cplx_abs_sqr(primal.transition), 0.0f));
    primal.base_weight = fmaxf(0.0f, 1.0f - fminf(primal.transition_magnitude, 1.0f));
    primal.response = cplx_mul(primal.transition, primal.finite_factor.factor);
    primal.weight = primal.base_weight * primal.finite_factor.factor_scale;
    return primal;
}

__device__ __forceinline__ void shadow_boundary_geometry_jvp(
    float3a tx_pos,
    float3a rx_pos,
    float3a edge_pos,
    float3a edge_hat,
    float3a n0,
    float3a t_tx_pos,
    float3a t_rx_pos,
    float3a t_edge_pos,
    float& t_phi,
    float& t_phi_prime,
    float& t_s,
    float& t_s_prime
) {
    float3a source_to_edge = f3_sub(edge_pos, tx_pos);
    float3a edge_to_target = f3_sub(rx_pos, edge_pos);
    float3a t_source_to_edge = f3_sub(t_edge_pos, t_tx_pos);
    float3a t_edge_to_target = f3_sub(t_rx_pos, t_edge_pos);
    float3a source_proj = project_to_wedge_plane(source_to_edge, edge_hat);
    float3a target_proj = project_to_wedge_plane(edge_to_target, edge_hat);
    float3a t_source_proj = project_to_wedge_plane_tangent(t_source_to_edge, edge_hat);
    float3a t_target_proj = project_to_wedge_plane_tangent(t_edge_to_target, edge_hat);
    float source_proj_norm = safe_length(source_proj);
    float target_proj_norm = safe_length(target_proj);
    float s_prime_proj = source_proj_norm + UTD_EPS;
    float s_proj = target_proj_norm + UTD_EPS;
    float t_s_prime_proj = source_proj_norm > UTD_SMALL_EPS
        ? f3_dot(source_proj, t_source_proj) / source_proj_norm
        : 0.0f;
    float t_s_proj = target_proj_norm > UTD_SMALL_EPS
        ? f3_dot(target_proj, t_target_proj) / target_proj_norm
        : 0.0f;
    float3a to_hat = safe_normalize(f3_cross(n0, edge_hat), make_f3(0.0f, 1.0f, 0.0f));
    float3a ki_proj = f3_div(source_proj, s_prime_proj);
    float3a t_ki_proj = f3_div(
        f3_sub(
            f3_mul(t_source_proj, s_prime_proj),
            f3_mul(source_proj, t_s_prime_proj)
        ),
        s_prime_proj * s_prime_proj
    );
    float c_phi_prime = -f3_dot(ki_proj, to_hat);
    float t_c_phi_prime = -f3_dot(t_ki_proj, to_hat);
    float phi_prime_denom = sqrtf(fmaxf(1.0f - c_phi_prime * c_phi_prime, 0.0f) + UTD_SMALL_EPS);
    float sign_phi_prime = ((-f3_dot(ki_proj, n0)) >= 0.0f ? 1.0f : -1.0f);
    t_phi_prime = (-sign_phi_prime) * (t_c_phi_prime / phi_prime_denom);

    float3a ko_proj = f3_div(target_proj, s_proj);
    float3a t_ko_proj = f3_div(
        f3_sub(
            f3_mul(t_target_proj, s_proj),
            f3_mul(target_proj, t_s_proj)
        ),
        s_proj * s_proj
    );
    float c_phi = f3_dot(ko_proj, to_hat);
    float t_c_phi = f3_dot(t_ko_proj, to_hat);
    float phi_denom = sqrtf(fmaxf(1.0f - c_phi * c_phi, 0.0f) + UTD_SMALL_EPS);
    float sign_phi = (f3_dot(ko_proj, n0) >= 0.0f ? 1.0f : -1.0f);
    t_phi = (-sign_phi) * (t_c_phi / phi_denom);

    float source_norm = safe_length(source_to_edge);
    float target_norm = safe_length(edge_to_target);
    t_s_prime = source_norm > UTD_SMALL_EPS
        ? f3_dot(source_to_edge, t_source_to_edge) / source_norm
        : 0.0f;
    t_s = target_norm > UTD_SMALL_EPS
        ? f3_dot(edge_to_target, t_edge_to_target) / target_norm
        : 0.0f;
}

__device__ __forceinline__ void shadow_boundary_incident_pair_jvp(
    float3a tx_pos,
    float3a rx_pos,
    float3a edge_pos,
    float3a edge_dir,
    float3a n0,
    float edge_line_min,
    float edge_line_max,
    float k,
    float3a t_tx_pos,
    float3a t_rx_pos,
    float3a t_edge_pos,
    const IncidentStatsPairPrimal& primal,
    float& t_weight,
    Complex& t_response
) {
    t_weight = 0.0f;
    t_response = cplx_zero();
    if (!primal.active) {
        return;
    }
    float t_phi = 0.0f;
    float t_phi_prime = 0.0f;
    float t_s = 0.0f;
    float t_s_prime = 0.0f;
    shadow_boundary_geometry_jvp(
        tx_pos,
        rx_pos,
        edge_pos,
        primal.finite_factor.edge_hat,
        n0,
        t_tx_pos,
        t_rx_pos,
        t_edge_pos,
        t_phi,
        t_phi_prime,
        t_s,
        t_s_prime
    );
    float denom = primal.s + primal.s_prime;
    float t_kL = k * (
        (t_s * primal.s_prime + primal.s * t_s_prime) * denom
        - primal.s * primal.s_prime * (t_s + t_s_prime)
    ) / (denom * denom);
    float t_dif_phi = t_phi - t_phi_prime;
    float t_a = primal.da_dif_phi * t_dif_phi;
    float t_x = t_kL * primal.a + primal.kL * t_a;
    Complex t_transition = cplx_mul_real(primal.transition_first, t_x);
    float t_transition_magnitude = 0.0f;
    if (primal.transition_magnitude > UTD_SMALL_EPS) {
        t_transition_magnitude = cplx_adj_dot(t_transition, primal.transition) / primal.transition_magnitude;
    }
    float t_base_weight = primal.transition_magnitude < 1.0f ? -t_transition_magnitude : 0.0f;
    Complex t_factor = cplx_zero();
    float t_factor_scale = 0.0f;
    shadow_boundary_finite_factor_jvp(
        tx_pos,
        rx_pos,
        edge_pos,
        primal.finite_factor.edge_hat,
        edge_line_min,
        edge_line_max,
        k,
        t_tx_pos,
        t_rx_pos,
        t_edge_pos,
        primal.finite_factor,
        t_factor,
        t_factor_scale
    );
    t_response = cplx_add(
        cplx_mul(t_transition, primal.finite_factor.factor),
        cplx_mul(primal.transition, t_factor)
    );
    t_weight = t_base_weight * primal.finite_factor.factor_scale + primal.base_weight * t_factor_scale;
}

__device__ __forceinline__ void shadow_boundary_incident_pair_vjp(
    float3a tx_pos,
    float3a rx_pos,
    float3a edge_pos,
    float3a edge_dir,
    float3a n0,
    float edge_line_min,
    float edge_line_max,
    float k,
    const IncidentStatsPairPrimal& primal,
    float grad_weight,
    Complex grad_response,
    float3a& grad_tx_pos,
    float3a& grad_rx_pos,
    float3a& grad_edge_pos
) {
    if (!primal.active) {
        return;
    }
    Complex grad_transition = cplx_zero();
    Complex grad_factor = cplx_zero();
    adj_cplx_mul(primal.transition, primal.finite_factor.factor, grad_response, grad_transition, grad_factor);
    float grad_base_weight = grad_weight * primal.finite_factor.factor_scale;
    float grad_factor_scale = grad_weight * primal.base_weight;
    if (primal.transition_magnitude < 1.0f && primal.transition_magnitude > UTD_SMALL_EPS) {
        float scale = -grad_base_weight / primal.transition_magnitude;
        grad_transition.re += scale * primal.transition.re;
        grad_transition.im += scale * primal.transition.im;
    }
    float grad_x = cplx_adj_dot(grad_transition, primal.transition_first);
    float grad_kL = grad_x * primal.a;
    float grad_dif_phi = grad_x * primal.kL * primal.da_dif_phi;
    float grad_phi = grad_dif_phi;
    float grad_phi_prime = -grad_dif_phi;
    float denom = primal.s + primal.s_prime;
    float grad_s = grad_kL * k * primal.s_prime * primal.s_prime / (denom * denom);
    float grad_s_prime = grad_kL * k * primal.s * primal.s / (denom * denom);
    float3a grad_edge_dir_ignored = f3_zero();
    float3a grad_n0_ignored = f3_zero();
    adj_compute_edge_geometry_3d(
        tx_pos,
        edge_pos,
        primal.finite_factor.edge_hat,
        n0,
        rx_pos,
        grad_phi,
        grad_phi_prime,
        grad_s,
        grad_s_prime,
        0.0f,
        grad_tx_pos,
        grad_edge_pos,
        grad_edge_dir_ignored,
        grad_n0_ignored,
        grad_rx_pos
    );
    shadow_boundary_finite_factor_vjp(
        tx_pos,
        rx_pos,
        edge_pos,
        primal.finite_factor.edge_hat,
        edge_line_min,
        edge_line_max,
        k,
        primal.finite_factor,
        grad_factor,
        grad_factor_scale,
        grad_tx_pos,
        grad_rx_pos,
        grad_edge_pos
    );
}

__global__ void radiomap_accumulate_vector_power_forward_kernel(
    const int* __restrict__ output_rx_idx,
    const float* __restrict__ pair_vec_x_re,
    const float* __restrict__ pair_vec_x_im,
    const float* __restrict__ pair_vec_y_re,
    const float* __restrict__ pair_vec_y_im,
    const float* __restrict__ pair_vec_z_re,
    const float* __restrict__ pair_vec_z_im,
    const float* __restrict__ arrival_x,
    const float* __restrict__ arrival_y,
    const float* __restrict__ arrival_z,
    float* __restrict__ coherent_re,
    float* __restrict__ coherent_im,
    float* __restrict__ power,
    float* __restrict__ vector_x_re,
    float* __restrict__ vector_x_im,
    float* __restrict__ vector_y_re,
    float* __restrict__ vector_y_im,
    float* __restrict__ vector_z_re,
    float* __restrict__ vector_z_im,
    float* __restrict__ valid_pair_count,
    int n_pairs,
    float rx_pol_x,
    float rx_pol_y,
    float rx_pol_z
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_pairs) {
        return;
    }

    int r_idx = output_rx_idx[tid];
    if (r_idx < 0) {
        return;
    }

    Complexf vx = make_complex(pair_vec_x_re[tid], pair_vec_x_im[tid]);
    Complexf vy = make_complex(pair_vec_y_re[tid], pair_vec_y_im[tid]);
    Complexf vz = make_complex(pair_vec_z_re[tid], pair_vec_z_im[tid]);
    Vec3f arrival = make_vec3(arrival_x[tid], arrival_y[tid], arrival_z[tid]);
    Vec3f rx_pol = make_vec3(rx_pol_x, rx_pol_y, rx_pol_z);
    Vec3f rx_basis = stable_perpendicular_basis(arrival, rx_pol);
    Complexf scalar = complex_dot_real(vx, vy, vz, rx_basis);
    float vector_power = complex_abs_sqr(vx) + complex_abs_sqr(vy) + complex_abs_sqr(vz);

    atomicAdd(&coherent_re[r_idx], scalar.re);
    atomicAdd(&coherent_im[r_idx], scalar.im);
    atomicAdd(&power[r_idx], vector_power);
    atomicAdd(&vector_x_re[r_idx], vx.re);
    atomicAdd(&vector_x_im[r_idx], vx.im);
    atomicAdd(&vector_y_re[r_idx], vy.re);
    atomicAdd(&vector_y_im[r_idx], vy.im);
    atomicAdd(&vector_z_re[r_idx], vz.re);
    atomicAdd(&vector_z_im[r_idx], vz.im);
    atomicAdd(valid_pair_count, 1.0f);
}

__global__ void radiomap_vector_power_forward_kernel(
    const float* __restrict__ vec_x_re,
    const float* __restrict__ vec_x_im,
    const float* __restrict__ vec_y_re,
    const float* __restrict__ vec_y_im,
    const float* __restrict__ vec_z_re,
    const float* __restrict__ vec_z_im,
    float* __restrict__ power,
    int n_rx
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_rx) {
        return;
    }
    Complexf vx = make_complex(vec_x_re[tid], vec_x_im[tid]);
    Complexf vy = make_complex(vec_y_re[tid], vec_y_im[tid]);
    Complexf vz = make_complex(vec_z_re[tid], vec_z_im[tid]);
    power[tid] = complex_abs_sqr(vx) + complex_abs_sqr(vy) + complex_abs_sqr(vz);
}

__global__ void radiomap_vector_power_jvp_kernel(
    const float* __restrict__ vec_x_re,
    const float* __restrict__ vec_x_im,
    const float* __restrict__ vec_y_re,
    const float* __restrict__ vec_y_im,
    const float* __restrict__ vec_z_re,
    const float* __restrict__ vec_z_im,
    const float* __restrict__ t_vec_x_re,
    const float* __restrict__ t_vec_x_im,
    const float* __restrict__ t_vec_y_re,
    const float* __restrict__ t_vec_y_im,
    const float* __restrict__ t_vec_z_re,
    const float* __restrict__ t_vec_z_im,
    float* __restrict__ t_power,
    int n_rx
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_rx) {
        return;
    }
    Complexf vx = make_complex(vec_x_re[tid], vec_x_im[tid]);
    Complexf vy = make_complex(vec_y_re[tid], vec_y_im[tid]);
    Complexf vz = make_complex(vec_z_re[tid], vec_z_im[tid]);
    Complexf tvx = make_complex(t_vec_x_re[tid], t_vec_x_im[tid]);
    Complexf tvy = make_complex(t_vec_y_re[tid], t_vec_y_im[tid]);
    Complexf tvz = make_complex(t_vec_z_re[tid], t_vec_z_im[tid]);
    t_power[tid] = 2.0f * (
        complex_real_inner(vx, tvx)
        + complex_real_inner(vy, tvy)
        + complex_real_inner(vz, tvz)
    );
}

__global__ void radiomap_vector_power_backward_kernel(
    const float* __restrict__ vec_x_re,
    const float* __restrict__ vec_x_im,
    const float* __restrict__ vec_y_re,
    const float* __restrict__ vec_y_im,
    const float* __restrict__ vec_z_re,
    const float* __restrict__ vec_z_im,
    const float* __restrict__ grad_power,
    float* __restrict__ grad_vec_x_re,
    float* __restrict__ grad_vec_x_im,
    float* __restrict__ grad_vec_y_re,
    float* __restrict__ grad_vec_y_im,
    float* __restrict__ grad_vec_z_re,
    float* __restrict__ grad_vec_z_im,
    int n_rx
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_rx) {
        return;
    }
    float gp = grad_power[tid];
    grad_vec_x_re[tid] = 2.0f * gp * vec_x_re[tid];
    grad_vec_x_im[tid] = 2.0f * gp * vec_x_im[tid];
    grad_vec_y_re[tid] = 2.0f * gp * vec_y_re[tid];
    grad_vec_y_im[tid] = 2.0f * gp * vec_y_im[tid];
    grad_vec_z_re[tid] = 2.0f * gp * vec_z_re[tid];
    grad_vec_z_im[tid] = 2.0f * gp * vec_z_im[tid];
}

__global__ void radiomap_matched_isb_completion_forward_kernel(
    const float* __restrict__ continued_direct_re,
    const float* __restrict__ continued_direct_im,
    const float* __restrict__ tx_basis_x,
    const float* __restrict__ tx_basis_y,
    const float* __restrict__ tx_basis_z,
    const float* __restrict__ rx_basis_x,
    const float* __restrict__ rx_basis_y,
    const float* __restrict__ rx_basis_z,
    const float* __restrict__ hard_visibility,
    const int* __restrict__ interior_mask,
    const float* __restrict__ incident_weight,
    const float* __restrict__ incident_response_re,
    const float* __restrict__ incident_response_im,
    const float* __restrict__ raw_vec_x_re,
    const float* __restrict__ raw_vec_x_im,
    const float* __restrict__ raw_vec_y_re,
    const float* __restrict__ raw_vec_y_im,
    const float* __restrict__ raw_vec_z_re,
    const float* __restrict__ raw_vec_z_im,
    float* __restrict__ coherent_re,
    float* __restrict__ coherent_im,
    float* __restrict__ power,
    float* __restrict__ vector_x_re,
    float* __restrict__ vector_x_im,
    float* __restrict__ vector_y_re,
    float* __restrict__ vector_y_im,
    float* __restrict__ vector_z_re,
    float* __restrict__ vector_z_im,
    int n_rx
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_rx) {
        return;
    }

    MatchedIsbPrimal primal = compute_matched_isb_primal(
        make_complex(continued_direct_re[tid], continued_direct_im[tid]),
        make_vec3(tx_basis_x[tid], tx_basis_y[tid], tx_basis_z[tid]),
        make_vec3(rx_basis_x[tid], rx_basis_y[tid], rx_basis_z[tid]),
        hard_visibility[tid],
        interior_mask[tid] != 0,
        incident_weight[tid],
        make_complex(incident_response_re[tid], incident_response_im[tid]),
        make_complex(raw_vec_x_re[tid], raw_vec_x_im[tid]),
        make_complex(raw_vec_y_re[tid], raw_vec_y_im[tid]),
        make_complex(raw_vec_z_re[tid], raw_vec_z_im[tid])
    );

    coherent_re[tid] = primal.coherent.re;
    coherent_im[tid] = primal.coherent.im;
    power[tid] = primal.power;
    vector_x_re[tid] = primal.vector_x.re;
    vector_x_im[tid] = primal.vector_x.im;
    vector_y_re[tid] = primal.vector_y.re;
    vector_y_im[tid] = primal.vector_y.im;
    vector_z_re[tid] = primal.vector_z.re;
    vector_z_im[tid] = primal.vector_z.im;
}

__global__ void radiomap_matched_isb_completion_jvp_kernel(
    const float* __restrict__ continued_direct_re,
    const float* __restrict__ continued_direct_im,
    const float* __restrict__ tx_basis_x,
    const float* __restrict__ tx_basis_y,
    const float* __restrict__ tx_basis_z,
    const float* __restrict__ rx_basis_x,
    const float* __restrict__ rx_basis_y,
    const float* __restrict__ rx_basis_z,
    const float* __restrict__ hard_visibility,
    const int* __restrict__ interior_mask,
    const float* __restrict__ incident_weight,
    const float* __restrict__ incident_response_re,
    const float* __restrict__ incident_response_im,
    const float* __restrict__ raw_vec_x_re,
    const float* __restrict__ raw_vec_x_im,
    const float* __restrict__ raw_vec_y_re,
    const float* __restrict__ raw_vec_y_im,
    const float* __restrict__ raw_vec_z_re,
    const float* __restrict__ raw_vec_z_im,
    const float* __restrict__ t_continued_direct_re,
    const float* __restrict__ t_continued_direct_im,
    const float* __restrict__ t_tx_basis_x,
    const float* __restrict__ t_tx_basis_y,
    const float* __restrict__ t_tx_basis_z,
    const float* __restrict__ t_rx_basis_x,
    const float* __restrict__ t_rx_basis_y,
    const float* __restrict__ t_rx_basis_z,
    const float* __restrict__ t_incident_weight,
    const float* __restrict__ t_incident_response_re,
    const float* __restrict__ t_incident_response_im,
    const float* __restrict__ t_raw_vec_x_re,
    const float* __restrict__ t_raw_vec_x_im,
    const float* __restrict__ t_raw_vec_y_re,
    const float* __restrict__ t_raw_vec_y_im,
    const float* __restrict__ t_raw_vec_z_re,
    const float* __restrict__ t_raw_vec_z_im,
    float* __restrict__ t_coherent_re,
    float* __restrict__ t_coherent_im,
    float* __restrict__ t_power,
    float* __restrict__ t_vector_x_re,
    float* __restrict__ t_vector_x_im,
    float* __restrict__ t_vector_y_re,
    float* __restrict__ t_vector_y_im,
    float* __restrict__ t_vector_z_re,
    float* __restrict__ t_vector_z_im,
    int n_rx
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_rx) {
        return;
    }

    Complexf continued_direct = make_complex(continued_direct_re[tid], continued_direct_im[tid]);
    Vec3f tx_basis = make_vec3(tx_basis_x[tid], tx_basis_y[tid], tx_basis_z[tid]);
    Vec3f rx_basis = make_vec3(rx_basis_x[tid], rx_basis_y[tid], rx_basis_z[tid]);
    bool local_interior_mask = interior_mask[tid] != 0;
    float local_hard_visibility = hard_visibility[tid];
    float local_incident_weight = incident_weight[tid];
    Complexf incident_response = make_complex(incident_response_re[tid], incident_response_im[tid]);
    Complexf raw_x = make_complex(raw_vec_x_re[tid], raw_vec_x_im[tid]);
    Complexf raw_y = make_complex(raw_vec_y_re[tid], raw_vec_y_im[tid]);
    Complexf raw_z = make_complex(raw_vec_z_re[tid], raw_vec_z_im[tid]);
    Complexf t_continued_direct = make_complex(t_continued_direct_re[tid], t_continued_direct_im[tid]);
    Vec3f t_tx_basis = make_vec3(t_tx_basis_x[tid], t_tx_basis_y[tid], t_tx_basis_z[tid]);
    Vec3f t_rx_basis = make_vec3(t_rx_basis_x[tid], t_rx_basis_y[tid], t_rx_basis_z[tid]);
    float t_local_incident_weight = t_incident_weight[tid];
    Complexf t_incident_response = make_complex(t_incident_response_re[tid], t_incident_response_im[tid]);
    Complexf t_raw_x = make_complex(t_raw_vec_x_re[tid], t_raw_vec_x_im[tid]);
    Complexf t_raw_y = make_complex(t_raw_vec_y_re[tid], t_raw_vec_y_im[tid]);
    Complexf t_raw_z = make_complex(t_raw_vec_z_re[tid], t_raw_vec_z_im[tid]);

    MatchedIsbPrimal primal = compute_matched_isb_primal(
        continued_direct,
        tx_basis,
        rx_basis,
        local_hard_visibility,
        local_interior_mask,
        local_incident_weight,
        incident_response,
        raw_x,
        raw_y,
        raw_z
    );

    if (local_interior_mask) {
        t_coherent_re[tid] = 0.0f;
        t_coherent_im[tid] = 0.0f;
        t_power[tid] = 0.0f;
        t_vector_x_re[tid] = 0.0f;
        t_vector_x_im[tid] = 0.0f;
        t_vector_y_re[tid] = 0.0f;
        t_vector_y_im[tid] = 0.0f;
        t_vector_z_re[tid] = 0.0f;
        t_vector_z_im[tid] = 0.0f;
        return;
    }

    float side_sign = local_hard_visibility > 0.0f ? 1.0f : -1.0f;
    Complexf t_smooth_coeff = make_complex(
        0.5f * side_sign * t_incident_response.re,
        0.5f * side_sign * t_incident_response.im
    );
    Complexf t_hard_direct = cscale(t_continued_direct, local_hard_visibility);
    Complexf t_raw_direct = cadd(
        cadd(cscale(t_raw_x, tx_basis.x), cscale(t_raw_y, tx_basis.y)),
        cscale(t_raw_z, tx_basis.z)
    );
    t_raw_direct.re += raw_x.re * t_tx_basis.x + raw_y.re * t_tx_basis.y + raw_z.re * t_tx_basis.z;
    t_raw_direct.im += raw_x.im * t_tx_basis.x + raw_y.im * t_tx_basis.y + raw_z.im * t_tx_basis.z;
    Complexf t_smooth = cadd(
        cmul(t_smooth_coeff, continued_direct),
        cmul(primal.smooth_coeff, t_continued_direct)
    );
    Complexf t_excess = csub(t_raw_direct, t_hard_direct);
    Complexf t_completion = csub(
        csub(t_smooth, t_hard_direct),
        cadd(
            cscale(primal.excess, t_local_incident_weight),
            cscale(t_excess, local_incident_weight)
        )
    );
    float t_scalar_factor = dot(t_tx_basis, rx_basis) + dot(tx_basis, t_rx_basis);
    Complexf t_coherent = cadd(
        cscale(t_completion, primal.scalar_factor),
        cscale(primal.completion, t_scalar_factor)
    );
    Complexf t_vec_x = cadd(cscale(t_completion, tx_basis.x), cscale(primal.completion, t_tx_basis.x));
    Complexf t_vec_y = cadd(cscale(t_completion, tx_basis.y), cscale(primal.completion, t_tx_basis.y));
    Complexf t_vec_z = cadd(cscale(t_completion, tx_basis.z), cscale(primal.completion, t_tx_basis.z));

    t_coherent_re[tid] = t_coherent.re;
    t_coherent_im[tid] = t_coherent.im;
    t_power[tid] = 2.0f * (
        complex_real_inner(primal.vector_x, t_vec_x)
        + complex_real_inner(primal.vector_y, t_vec_y)
        + complex_real_inner(primal.vector_z, t_vec_z)
    );
    t_vector_x_re[tid] = t_vec_x.re;
    t_vector_x_im[tid] = t_vec_x.im;
    t_vector_y_re[tid] = t_vec_y.re;
    t_vector_y_im[tid] = t_vec_y.im;
    t_vector_z_re[tid] = t_vec_z.re;
    t_vector_z_im[tid] = t_vec_z.im;
}

__global__ void radiomap_matched_isb_completion_backward_kernel(
    const float* __restrict__ continued_direct_re,
    const float* __restrict__ continued_direct_im,
    const float* __restrict__ tx_basis_x,
    const float* __restrict__ tx_basis_y,
    const float* __restrict__ tx_basis_z,
    const float* __restrict__ rx_basis_x,
    const float* __restrict__ rx_basis_y,
    const float* __restrict__ rx_basis_z,
    const float* __restrict__ hard_visibility,
    const int* __restrict__ interior_mask,
    const float* __restrict__ incident_weight,
    const float* __restrict__ incident_response_re,
    const float* __restrict__ incident_response_im,
    const float* __restrict__ raw_vec_x_re,
    const float* __restrict__ raw_vec_x_im,
    const float* __restrict__ raw_vec_y_re,
    const float* __restrict__ raw_vec_y_im,
    const float* __restrict__ raw_vec_z_re,
    const float* __restrict__ raw_vec_z_im,
    const float* __restrict__ grad_coherent_re,
    const float* __restrict__ grad_coherent_im,
    const float* __restrict__ grad_power,
    const float* __restrict__ grad_vector_x_re,
    const float* __restrict__ grad_vector_x_im,
    const float* __restrict__ grad_vector_y_re,
    const float* __restrict__ grad_vector_y_im,
    const float* __restrict__ grad_vector_z_re,
    const float* __restrict__ grad_vector_z_im,
    float* __restrict__ grad_continued_direct_re,
    float* __restrict__ grad_continued_direct_im,
    float* __restrict__ grad_tx_basis_x,
    float* __restrict__ grad_tx_basis_y,
    float* __restrict__ grad_tx_basis_z,
    float* __restrict__ grad_rx_basis_x,
    float* __restrict__ grad_rx_basis_y,
    float* __restrict__ grad_rx_basis_z,
    float* __restrict__ grad_incident_weight,
    float* __restrict__ grad_incident_response_re,
    float* __restrict__ grad_incident_response_im,
    float* __restrict__ grad_raw_vec_x_re,
    float* __restrict__ grad_raw_vec_x_im,
    float* __restrict__ grad_raw_vec_y_re,
    float* __restrict__ grad_raw_vec_y_im,
    float* __restrict__ grad_raw_vec_z_re,
    float* __restrict__ grad_raw_vec_z_im,
    int n_rx
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_rx) {
        return;
    }

    Complexf continued_direct = make_complex(continued_direct_re[tid], continued_direct_im[tid]);
    Vec3f tx_basis = make_vec3(tx_basis_x[tid], tx_basis_y[tid], tx_basis_z[tid]);
    Vec3f rx_basis = make_vec3(rx_basis_x[tid], rx_basis_y[tid], rx_basis_z[tid]);
    float local_hard_visibility = hard_visibility[tid];
    bool local_interior_mask = interior_mask[tid] != 0;
    float local_incident_weight = incident_weight[tid];
    Complexf incident_response = make_complex(incident_response_re[tid], incident_response_im[tid]);
    Complexf raw_x = make_complex(raw_vec_x_re[tid], raw_vec_x_im[tid]);
    Complexf raw_y = make_complex(raw_vec_y_re[tid], raw_vec_y_im[tid]);
    Complexf raw_z = make_complex(raw_vec_z_re[tid], raw_vec_z_im[tid]);

    MatchedIsbPrimal primal = compute_matched_isb_primal(
        continued_direct,
        tx_basis,
        rx_basis,
        local_hard_visibility,
        local_interior_mask,
        local_incident_weight,
        incident_response,
        raw_x,
        raw_y,
        raw_z
    );

    Complexf grad_cd = make_complex(0.0f, 0.0f);
    Vec3f grad_tx_basis = make_vec3(0.0f, 0.0f, 0.0f);
    Vec3f grad_rx_basis = make_vec3(0.0f, 0.0f, 0.0f);
    float grad_weight = 0.0f;
    Complexf grad_incident_response = make_complex(0.0f, 0.0f);
    Complexf grad_raw_x = make_complex(0.0f, 0.0f);
    Complexf grad_raw_y = make_complex(0.0f, 0.0f);
    Complexf grad_raw_z = make_complex(0.0f, 0.0f);

    if (!local_interior_mask) {
        Complexf grad_completion = make_complex(0.0f, 0.0f);
        Complexf g_coherent = make_complex(grad_coherent_re[tid], grad_coherent_im[tid]);
        grad_completion.re += g_coherent.re * primal.scalar_factor;
        grad_completion.im += g_coherent.im * primal.scalar_factor;
        float g_scalar = complex_real_inner(g_coherent, primal.completion);
        grad_tx_basis.x += g_scalar * rx_basis.x;
        grad_tx_basis.y += g_scalar * rx_basis.y;
        grad_tx_basis.z += g_scalar * rx_basis.z;
        grad_rx_basis.x += g_scalar * tx_basis.x;
        grad_rx_basis.y += g_scalar * tx_basis.y;
        grad_rx_basis.z += g_scalar * tx_basis.z;

        Complexf g_vec_x = make_complex(grad_vector_x_re[tid], grad_vector_x_im[tid]);
        Complexf g_vec_y = make_complex(grad_vector_y_re[tid], grad_vector_y_im[tid]);
        Complexf g_vec_z = make_complex(grad_vector_z_re[tid], grad_vector_z_im[tid]);
        float gp = grad_power[tid];
        if (gp != 0.0f) {
            g_vec_x.re += 2.0f * gp * primal.vector_x.re;
            g_vec_x.im += 2.0f * gp * primal.vector_x.im;
            g_vec_y.re += 2.0f * gp * primal.vector_y.re;
            g_vec_y.im += 2.0f * gp * primal.vector_y.im;
            g_vec_z.re += 2.0f * gp * primal.vector_z.re;
            g_vec_z.im += 2.0f * gp * primal.vector_z.im;
        }

        accumulate_vector_from_complex_grad(g_vec_x, primal.completion, tx_basis.x, &grad_completion, &grad_tx_basis.x);
        accumulate_vector_from_complex_grad(g_vec_y, primal.completion, tx_basis.y, &grad_completion, &grad_tx_basis.y);
        accumulate_vector_from_complex_grad(g_vec_z, primal.completion, tx_basis.z, &grad_completion, &grad_tx_basis.z);

        Complexf grad_smooth = grad_completion;
        Complexf grad_hard_direct = cscale(grad_completion, -1.0f);
        grad_weight -= complex_real_inner(grad_completion, primal.excess);
        Complexf grad_excess = cscale(grad_completion, -local_incident_weight);
        Complexf grad_raw_direct = grad_excess;
        grad_hard_direct = cadd(grad_hard_direct, cscale(grad_excess, -1.0f));

        grad_cd.re += grad_hard_direct.re * local_hard_visibility;
        grad_cd.im += grad_hard_direct.im * local_hard_visibility;

        Complexf grad_smooth_coeff = make_complex(0.0f, 0.0f);
        accumulate_complex_mul_backward(grad_smooth, primal.smooth_coeff, continued_direct, &grad_smooth_coeff, &grad_cd);
        float side_sign = local_hard_visibility > 0.0f ? 1.0f : -1.0f;
        grad_incident_response.re += 0.5f * side_sign * grad_smooth_coeff.re;
        grad_incident_response.im += 0.5f * side_sign * grad_smooth_coeff.im;

        grad_raw_x.re += grad_raw_direct.re * tx_basis.x;
        grad_raw_x.im += grad_raw_direct.im * tx_basis.x;
        grad_raw_y.re += grad_raw_direct.re * tx_basis.y;
        grad_raw_y.im += grad_raw_direct.im * tx_basis.y;
        grad_raw_z.re += grad_raw_direct.re * tx_basis.z;
        grad_raw_z.im += grad_raw_direct.im * tx_basis.z;
        grad_tx_basis.x += complex_real_inner(grad_raw_direct, raw_x);
        grad_tx_basis.y += complex_real_inner(grad_raw_direct, raw_y);
        grad_tx_basis.z += complex_real_inner(grad_raw_direct, raw_z);
    }

    grad_continued_direct_re[tid] = grad_cd.re;
    grad_continued_direct_im[tid] = grad_cd.im;
    grad_tx_basis_x[tid] = grad_tx_basis.x;
    grad_tx_basis_y[tid] = grad_tx_basis.y;
    grad_tx_basis_z[tid] = grad_tx_basis.z;
    grad_rx_basis_x[tid] = grad_rx_basis.x;
    grad_rx_basis_y[tid] = grad_rx_basis.y;
    grad_rx_basis_z[tid] = grad_rx_basis.z;
    grad_incident_weight[tid] = grad_weight;
    grad_incident_response_re[tid] = grad_incident_response.re;
    grad_incident_response_im[tid] = grad_incident_response.im;
    grad_raw_vec_x_re[tid] = grad_raw_x.re;
    grad_raw_vec_x_im[tid] = grad_raw_x.im;
    grad_raw_vec_y_re[tid] = grad_raw_y.re;
    grad_raw_vec_y_im[tid] = grad_raw_y.im;
    grad_raw_vec_z_re[tid] = grad_raw_z.re;
    grad_raw_vec_z_im[tid] = grad_raw_z.im;
}

__global__ void radiomap_shadow_boundary_incident_stats_forward_kernel(
    const float* __restrict__ tx_x,
    const float* __restrict__ tx_y,
    const float* __restrict__ tx_z,
    const float* __restrict__ rx_x,
    const float* __restrict__ rx_y,
    const float* __restrict__ rx_z,
    const float* __restrict__ edge_pos_x,
    const float* __restrict__ edge_pos_y,
    const float* __restrict__ edge_pos_z,
    const float* __restrict__ edge_dir_x,
    const float* __restrict__ edge_dir_y,
    const float* __restrict__ edge_dir_z,
    const float* __restrict__ n0_x,
    const float* __restrict__ n0_y,
    const float* __restrict__ n0_z,
    const float* __restrict__ nn_x,
    const float* __restrict__ nn_y,
    const float* __restrict__ nn_z,
    const float* __restrict__ wedge_n,
    const float* __restrict__ edge_line_min,
    const float* __restrict__ edge_line_max,
    const int* __restrict__ source_visible,
    float* __restrict__ sum_incident_weight,
    float* __restrict__ max_incident_weight,
    float* __restrict__ weighted_incident_response_real,
    float* __restrict__ weighted_incident_response_imag,
    int* __restrict__ support_edge_count,
    int n_rx,
    int n_edges,
    float k
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int n_pairs = n_rx * n_edges;
    if (tid >= n_pairs) {
        return;
    }
    int rx_idx = tid / n_edges;
    int edge_idx = tid - rx_idx * n_edges;
    float3a tx_pos = make_f3(tx_x[0], tx_y[0], tx_z[0]);
    float3a rx_pos = make_f3(rx_x[rx_idx], rx_y[rx_idx], rx_z[rx_idx]);
    float3a edge_pos = load_f3(edge_pos_x, edge_pos_y, edge_pos_z, edge_idx);
    float3a edge_dir = load_f3(edge_dir_x, edge_dir_y, edge_dir_z, edge_idx);
    float3a n0 = load_f3(n0_x, n0_y, n0_z, edge_idx);
    float3a nn = load_f3(nn_x, nn_y, nn_z, edge_idx);
    IncidentStatsPairPrimal primal = compute_shadow_boundary_incident_pair_primal(
        tx_pos,
        rx_pos,
        edge_pos,
        edge_dir,
        n0,
        nn,
        wedge_n[edge_idx],
        edge_line_min[edge_idx],
        edge_line_max[edge_idx],
        source_visible[edge_idx] != 0,
        k
    );
    atomicAdd(&sum_incident_weight[rx_idx], primal.weight);
    atomicAdd(&weighted_incident_response_real[rx_idx], primal.weight * primal.response.re);
    atomicAdd(&weighted_incident_response_imag[rx_idx], primal.weight * primal.response.im);
    if (primal.weight > 0.0f) {
        atomicAdd(&support_edge_count[rx_idx], 1);
    }
    atomic_max_nonnegative(&max_incident_weight[rx_idx], primal.weight);
}

__global__ void radiomap_shadow_boundary_incident_stats_argmax_kernel(
    const float* __restrict__ tx_x,
    const float* __restrict__ tx_y,
    const float* __restrict__ tx_z,
    const float* __restrict__ rx_x,
    const float* __restrict__ rx_y,
    const float* __restrict__ rx_z,
    const float* __restrict__ edge_pos_x,
    const float* __restrict__ edge_pos_y,
    const float* __restrict__ edge_pos_z,
    const float* __restrict__ edge_dir_x,
    const float* __restrict__ edge_dir_y,
    const float* __restrict__ edge_dir_z,
    const float* __restrict__ n0_x,
    const float* __restrict__ n0_y,
    const float* __restrict__ n0_z,
    const float* __restrict__ nn_x,
    const float* __restrict__ nn_y,
    const float* __restrict__ nn_z,
    const float* __restrict__ wedge_n,
    const float* __restrict__ edge_line_min,
    const float* __restrict__ edge_line_max,
    const int* __restrict__ source_visible,
    const float* __restrict__ max_incident_weight,
    int* __restrict__ argmax_edge_idx,
    int n_rx,
    int n_edges,
    float k
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int n_pairs = n_rx * n_edges;
    if (tid >= n_pairs) {
        return;
    }
    int rx_idx = tid / n_edges;
    int edge_idx = tid - rx_idx * n_edges;
    float max_weight = max_incident_weight[rx_idx];
    if (max_weight <= 0.0f) {
        return;
    }
    float3a tx_pos = make_f3(tx_x[0], tx_y[0], tx_z[0]);
    float3a rx_pos = make_f3(rx_x[rx_idx], rx_y[rx_idx], rx_z[rx_idx]);
    float3a edge_pos = load_f3(edge_pos_x, edge_pos_y, edge_pos_z, edge_idx);
    float3a edge_dir = load_f3(edge_dir_x, edge_dir_y, edge_dir_z, edge_idx);
    float3a n0 = load_f3(n0_x, n0_y, n0_z, edge_idx);
    float3a nn = load_f3(nn_x, nn_y, nn_z, edge_idx);
    IncidentStatsPairPrimal primal = compute_shadow_boundary_incident_pair_primal(
        tx_pos,
        rx_pos,
        edge_pos,
        edge_dir,
        n0,
        nn,
        wedge_n[edge_idx],
        edge_line_min[edge_idx],
        edge_line_max[edge_idx],
        source_visible[edge_idx] != 0,
        k
    );
    if (primal.weight > 0.0f && __float_as_uint(primal.weight) == __float_as_uint(max_weight)) {
        atomicMin(&argmax_edge_idx[rx_idx], edge_idx);
    }
}

__global__ void radiomap_shadow_boundary_incident_stats_finalize_argmax_kernel(
    int* __restrict__ argmax_edge_idx,
    int n_rx
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_rx) {
        return;
    }
    if (argmax_edge_idx[tid] == INT_MAX) {
        argmax_edge_idx[tid] = -1;
    }
}

__global__ void radiomap_shadow_boundary_incident_stats_second_max_kernel(
    const float* __restrict__ tx_x,
    const float* __restrict__ tx_y,
    const float* __restrict__ tx_z,
    const float* __restrict__ rx_x,
    const float* __restrict__ rx_y,
    const float* __restrict__ rx_z,
    const float* __restrict__ edge_pos_x,
    const float* __restrict__ edge_pos_y,
    const float* __restrict__ edge_pos_z,
    const float* __restrict__ edge_dir_x,
    const float* __restrict__ edge_dir_y,
    const float* __restrict__ edge_dir_z,
    const float* __restrict__ n0_x,
    const float* __restrict__ n0_y,
    const float* __restrict__ n0_z,
    const float* __restrict__ nn_x,
    const float* __restrict__ nn_y,
    const float* __restrict__ nn_z,
    const float* __restrict__ wedge_n,
    const float* __restrict__ edge_line_min,
    const float* __restrict__ edge_line_max,
    const int* __restrict__ source_visible,
    const float* __restrict__ max_incident_weight,
    const int* __restrict__ argmax_edge_idx,
    float* __restrict__ second_max_incident_weight,
    int n_rx,
    int n_edges,
    float k
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int n_pairs = n_rx * n_edges;
    if (tid >= n_pairs) {
        return;
    }
    int rx_idx = tid / n_edges;
    int edge_idx = tid - rx_idx * n_edges;
    int winning_edge_idx = argmax_edge_idx[rx_idx];
    float max_weight = max_incident_weight[rx_idx];
    if (winning_edge_idx < 0 || max_weight <= 0.0f || edge_idx == winning_edge_idx) {
        return;
    }
    float3a tx_pos = make_f3(tx_x[0], tx_y[0], tx_z[0]);
    float3a rx_pos = make_f3(rx_x[rx_idx], rx_y[rx_idx], rx_z[rx_idx]);
    float3a edge_pos = load_f3(edge_pos_x, edge_pos_y, edge_pos_z, edge_idx);
    float3a edge_dir = load_f3(edge_dir_x, edge_dir_y, edge_dir_z, edge_idx);
    float3a n0 = load_f3(n0_x, n0_y, n0_z, edge_idx);
    float3a nn = load_f3(nn_x, nn_y, nn_z, edge_idx);
    IncidentStatsPairPrimal primal = compute_shadow_boundary_incident_pair_primal(
        tx_pos,
        rx_pos,
        edge_pos,
        edge_dir,
        n0,
        nn,
        wedge_n[edge_idx],
        edge_line_min[edge_idx],
        edge_line_max[edge_idx],
        source_visible[edge_idx] != 0,
        k
    );
    atomic_max_nonnegative(&second_max_incident_weight[rx_idx], primal.weight);
}

__global__ void radiomap_shadow_boundary_incident_stats_jvp_kernel(
    const float* __restrict__ tx_x,
    const float* __restrict__ tx_y,
    const float* __restrict__ tx_z,
    const float* __restrict__ rx_x,
    const float* __restrict__ rx_y,
    const float* __restrict__ rx_z,
    const float* __restrict__ edge_pos_x,
    const float* __restrict__ edge_pos_y,
    const float* __restrict__ edge_pos_z,
    const float* __restrict__ edge_dir_x,
    const float* __restrict__ edge_dir_y,
    const float* __restrict__ edge_dir_z,
    const float* __restrict__ n0_x,
    const float* __restrict__ n0_y,
    const float* __restrict__ n0_z,
    const float* __restrict__ nn_x,
    const float* __restrict__ nn_y,
    const float* __restrict__ nn_z,
    const float* __restrict__ wedge_n,
    const float* __restrict__ edge_line_min,
    const float* __restrict__ edge_line_max,
    const int* __restrict__ source_visible,
    const int* __restrict__ argmax_edge_idx,
    const float* __restrict__ t_tx_x,
    const float* __restrict__ t_tx_y,
    const float* __restrict__ t_tx_z,
    const float* __restrict__ t_rx_x,
    const float* __restrict__ t_rx_y,
    const float* __restrict__ t_rx_z,
    const float* __restrict__ t_edge_pos_x,
    const float* __restrict__ t_edge_pos_y,
    const float* __restrict__ t_edge_pos_z,
    float* __restrict__ t_sum_incident_weight,
    float* __restrict__ t_max_incident_weight,
    float* __restrict__ t_weighted_incident_response_real,
    float* __restrict__ t_weighted_incident_response_imag,
    int n_rx,
    int n_edges,
    float k
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int n_pairs = n_rx * n_edges;
    if (tid >= n_pairs) {
        return;
    }
    int rx_idx = tid / n_edges;
    int edge_idx = tid - rx_idx * n_edges;
    float3a tx_pos = make_f3(tx_x[0], tx_y[0], tx_z[0]);
    float3a rx_pos = make_f3(rx_x[rx_idx], rx_y[rx_idx], rx_z[rx_idx]);
    float3a edge_pos = load_f3(edge_pos_x, edge_pos_y, edge_pos_z, edge_idx);
    float3a edge_dir = load_f3(edge_dir_x, edge_dir_y, edge_dir_z, edge_idx);
    float3a n0 = load_f3(n0_x, n0_y, n0_z, edge_idx);
    float3a nn = load_f3(nn_x, nn_y, nn_z, edge_idx);
    IncidentStatsPairPrimal primal = compute_shadow_boundary_incident_pair_primal(
        tx_pos,
        rx_pos,
        edge_pos,
        edge_dir,
        n0,
        nn,
        wedge_n[edge_idx],
        edge_line_min[edge_idx],
        edge_line_max[edge_idx],
        source_visible[edge_idx] != 0,
        k
    );
    float t_weight = 0.0f;
    Complex t_response = cplx_zero();
    shadow_boundary_incident_pair_jvp(
        tx_pos,
        rx_pos,
        edge_pos,
        edge_dir,
        n0,
        edge_line_min[edge_idx],
        edge_line_max[edge_idx],
        k,
        make_f3(t_tx_x[0], t_tx_y[0], t_tx_z[0]),
        make_f3(t_rx_x[rx_idx], t_rx_y[rx_idx], t_rx_z[rx_idx]),
        load_f3(t_edge_pos_x, t_edge_pos_y, t_edge_pos_z, edge_idx),
        primal,
        t_weight,
        t_response
    );
    atomicAdd(&t_sum_incident_weight[rx_idx], t_weight);
    if (argmax_edge_idx[rx_idx] == edge_idx) {
        atomicAdd(&t_max_incident_weight[rx_idx], t_weight);
    }
    atomicAdd(
        &t_weighted_incident_response_real[rx_idx],
        t_weight * primal.response.re + primal.weight * t_response.re
    );
    atomicAdd(
        &t_weighted_incident_response_imag[rx_idx],
        t_weight * primal.response.im + primal.weight * t_response.im
    );
}

__global__ void radiomap_shadow_boundary_incident_stats_backward_kernel(
    const float* __restrict__ tx_x,
    const float* __restrict__ tx_y,
    const float* __restrict__ tx_z,
    const float* __restrict__ rx_x,
    const float* __restrict__ rx_y,
    const float* __restrict__ rx_z,
    const float* __restrict__ edge_pos_x,
    const float* __restrict__ edge_pos_y,
    const float* __restrict__ edge_pos_z,
    const float* __restrict__ edge_dir_x,
    const float* __restrict__ edge_dir_y,
    const float* __restrict__ edge_dir_z,
    const float* __restrict__ n0_x,
    const float* __restrict__ n0_y,
    const float* __restrict__ n0_z,
    const float* __restrict__ nn_x,
    const float* __restrict__ nn_y,
    const float* __restrict__ nn_z,
    const float* __restrict__ wedge_n,
    const float* __restrict__ edge_line_min,
    const float* __restrict__ edge_line_max,
    const int* __restrict__ source_visible,
    const int* __restrict__ argmax_edge_idx,
    const float* __restrict__ grad_sum_incident_weight,
    const float* __restrict__ grad_max_incident_weight,
    const float* __restrict__ grad_weighted_incident_response_real,
    const float* __restrict__ grad_weighted_incident_response_imag,
    float* __restrict__ grad_tx_x,
    float* __restrict__ grad_tx_y,
    float* __restrict__ grad_tx_z,
    float* __restrict__ grad_rx_x,
    float* __restrict__ grad_rx_y,
    float* __restrict__ grad_rx_z,
    float* __restrict__ grad_edge_pos_x,
    float* __restrict__ grad_edge_pos_y,
    float* __restrict__ grad_edge_pos_z,
    int n_rx,
    int n_edges,
    float k
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int n_pairs = n_rx * n_edges;
    if (tid >= n_pairs) {
        return;
    }
    int rx_idx = tid / n_edges;
    int edge_idx = tid - rx_idx * n_edges;
    float local_grad_sum = grad_sum_incident_weight[rx_idx];
    float local_grad_max = argmax_edge_idx[rx_idx] == edge_idx ? grad_max_incident_weight[rx_idx] : 0.0f;
    float local_grad_wr = grad_weighted_incident_response_real[rx_idx];
    float local_grad_wi = grad_weighted_incident_response_imag[rx_idx];
    if (
        local_grad_sum == 0.0f
        && local_grad_max == 0.0f
        && local_grad_wr == 0.0f
        && local_grad_wi == 0.0f
    ) {
        return;
    }
    float3a tx_pos = make_f3(tx_x[0], tx_y[0], tx_z[0]);
    float3a rx_pos = make_f3(rx_x[rx_idx], rx_y[rx_idx], rx_z[rx_idx]);
    float3a edge_pos = load_f3(edge_pos_x, edge_pos_y, edge_pos_z, edge_idx);
    float3a edge_dir = load_f3(edge_dir_x, edge_dir_y, edge_dir_z, edge_idx);
    float3a n0 = load_f3(n0_x, n0_y, n0_z, edge_idx);
    float3a nn = load_f3(nn_x, nn_y, nn_z, edge_idx);
    IncidentStatsPairPrimal primal = compute_shadow_boundary_incident_pair_primal(
        tx_pos,
        rx_pos,
        edge_pos,
        edge_dir,
        n0,
        nn,
        wedge_n[edge_idx],
        edge_line_min[edge_idx],
        edge_line_max[edge_idx],
        source_visible[edge_idx] != 0,
        k
    );
    float grad_weight =
        local_grad_sum
        + local_grad_max
        + local_grad_wr * primal.response.re
        + local_grad_wi * primal.response.im;
    Complex grad_response = cplx(
        local_grad_wr * primal.weight,
        local_grad_wi * primal.weight
    );
    float3a grad_tx_pos = f3_zero();
    float3a grad_rx_pos = f3_zero();
    float3a grad_edge_pos = f3_zero();
    shadow_boundary_incident_pair_vjp(
        tx_pos,
        rx_pos,
        edge_pos,
        edge_dir,
        n0,
        edge_line_min[edge_idx],
        edge_line_max[edge_idx],
        k,
        primal,
        grad_weight,
        grad_response,
        grad_tx_pos,
        grad_rx_pos,
        grad_edge_pos
    );
    atomicAdd(&grad_tx_x[0], grad_tx_pos.x);
    atomicAdd(&grad_tx_y[0], grad_tx_pos.y);
    atomicAdd(&grad_tx_z[0], grad_tx_pos.z);
    atomicAdd(&grad_rx_x[rx_idx], grad_rx_pos.x);
    atomicAdd(&grad_rx_y[rx_idx], grad_rx_pos.y);
    atomicAdd(&grad_rx_z[rx_idx], grad_rx_pos.z);
    atomicAdd(&grad_edge_pos_x[edge_idx], grad_edge_pos.x);
    atomicAdd(&grad_edge_pos_y[edge_idx], grad_edge_pos.y);
    atomicAdd(&grad_edge_pos_z[edge_idx], grad_edge_pos.z);
}

} // namespace

void radiomap_accumulate_vector_power_forward(
    const int* output_rx_idx,
    const float* pair_vec_x_re,
    const float* pair_vec_x_im,
    const float* pair_vec_y_re,
    const float* pair_vec_y_im,
    const float* pair_vec_z_re,
    const float* pair_vec_z_im,
    const float* arrival_x,
    const float* arrival_y,
    const float* arrival_z,
    float* coherent_re,
    float* coherent_im,
    float* power,
    float* vector_x_re,
    float* vector_x_im,
    float* vector_y_re,
    float* vector_y_im,
    float* vector_z_re,
    float* vector_z_im,
    float* valid_pair_count,
    int n_pairs,
    float rx_pol_x,
    float rx_pol_y,
    float rx_pol_z
) {
    if (n_pairs <= 0) {
        return;
    }
    constexpr int block_size = 256;
    int grid_size = ceil_div_int(n_pairs, block_size);
    radiomap_accumulate_vector_power_forward_kernel<<<grid_size, block_size>>>(
        output_rx_idx,
        pair_vec_x_re,
        pair_vec_x_im,
        pair_vec_y_re,
        pair_vec_y_im,
        pair_vec_z_re,
        pair_vec_z_im,
        arrival_x,
        arrival_y,
        arrival_z,
        coherent_re,
        coherent_im,
        power,
        vector_x_re,
        vector_x_im,
        vector_y_re,
        vector_y_im,
        vector_z_re,
        vector_z_im,
        valid_pair_count,
        n_pairs,
        rx_pol_x,
        rx_pol_y,
        rx_pol_z
    );
    throw_cuda(cudaGetLastError(), "radiomap_accumulate_vector_power_forward_kernel launch");
}

void radiomap_vector_power_forward(
    const float* vec_x_re,
    const float* vec_x_im,
    const float* vec_y_re,
    const float* vec_y_im,
    const float* vec_z_re,
    const float* vec_z_im,
    float* power,
    int n_rx
) {
    if (n_rx <= 0) {
        return;
    }
    constexpr int block_size = 256;
    int grid_size = ceil_div_int(n_rx, block_size);
    radiomap_vector_power_forward_kernel<<<grid_size, block_size>>>(
        vec_x_re,
        vec_x_im,
        vec_y_re,
        vec_y_im,
        vec_z_re,
        vec_z_im,
        power,
        n_rx
    );
    throw_cuda(cudaGetLastError(), "radiomap_vector_power_forward_kernel launch");
}

void radiomap_vector_power_jvp(
    const float* vec_x_re,
    const float* vec_x_im,
    const float* vec_y_re,
    const float* vec_y_im,
    const float* vec_z_re,
    const float* vec_z_im,
    const float* t_vec_x_re,
    const float* t_vec_x_im,
    const float* t_vec_y_re,
    const float* t_vec_y_im,
    const float* t_vec_z_re,
    const float* t_vec_z_im,
    float* t_power,
    int n_rx
) {
    if (n_rx <= 0) {
        return;
    }
    constexpr int block_size = 256;
    int grid_size = ceil_div_int(n_rx, block_size);
    radiomap_vector_power_jvp_kernel<<<grid_size, block_size>>>(
        vec_x_re,
        vec_x_im,
        vec_y_re,
        vec_y_im,
        vec_z_re,
        vec_z_im,
        t_vec_x_re,
        t_vec_x_im,
        t_vec_y_re,
        t_vec_y_im,
        t_vec_z_re,
        t_vec_z_im,
        t_power,
        n_rx
    );
    throw_cuda(cudaGetLastError(), "radiomap_vector_power_jvp_kernel launch");
}

void radiomap_vector_power_backward(
    const float* vec_x_re,
    const float* vec_x_im,
    const float* vec_y_re,
    const float* vec_y_im,
    const float* vec_z_re,
    const float* vec_z_im,
    const float* grad_power,
    float* grad_vec_x_re,
    float* grad_vec_x_im,
    float* grad_vec_y_re,
    float* grad_vec_y_im,
    float* grad_vec_z_re,
    float* grad_vec_z_im,
    int n_rx
) {
    if (n_rx <= 0) {
        return;
    }
    constexpr int block_size = 256;
    int grid_size = ceil_div_int(n_rx, block_size);
    radiomap_vector_power_backward_kernel<<<grid_size, block_size>>>(
        vec_x_re,
        vec_x_im,
        vec_y_re,
        vec_y_im,
        vec_z_re,
        vec_z_im,
        grad_power,
        grad_vec_x_re,
        grad_vec_x_im,
        grad_vec_y_re,
        grad_vec_y_im,
        grad_vec_z_re,
        grad_vec_z_im,
        n_rx
    );
    throw_cuda(cudaGetLastError(), "radiomap_vector_power_backward_kernel launch");
}

void radiomap_matched_isb_completion_forward(
    const float* continued_direct_re,
    const float* continued_direct_im,
    const float* tx_basis_x,
    const float* tx_basis_y,
    const float* tx_basis_z,
    const float* rx_basis_x,
    const float* rx_basis_y,
    const float* rx_basis_z,
    const float* hard_visibility,
    const int* interior_mask,
    const float* incident_weight,
    const float* incident_response_re,
    const float* incident_response_im,
    const float* raw_vec_x_re,
    const float* raw_vec_x_im,
    const float* raw_vec_y_re,
    const float* raw_vec_y_im,
    const float* raw_vec_z_re,
    const float* raw_vec_z_im,
    float* coherent_re,
    float* coherent_im,
    float* power,
    float* vector_x_re,
    float* vector_x_im,
    float* vector_y_re,
    float* vector_y_im,
    float* vector_z_re,
    float* vector_z_im,
    int n_rx
) {
    if (n_rx <= 0) {
        return;
    }
    constexpr int block_size = 256;
    int grid_size = ceil_div_int(n_rx, block_size);
    radiomap_matched_isb_completion_forward_kernel<<<grid_size, block_size>>>(
        continued_direct_re,
        continued_direct_im,
        tx_basis_x,
        tx_basis_y,
        tx_basis_z,
        rx_basis_x,
        rx_basis_y,
        rx_basis_z,
        hard_visibility,
        interior_mask,
        incident_weight,
        incident_response_re,
        incident_response_im,
        raw_vec_x_re,
        raw_vec_x_im,
        raw_vec_y_re,
        raw_vec_y_im,
        raw_vec_z_re,
        raw_vec_z_im,
        coherent_re,
        coherent_im,
        power,
        vector_x_re,
        vector_x_im,
        vector_y_re,
        vector_y_im,
        vector_z_re,
        vector_z_im,
        n_rx
    );
    throw_cuda(cudaGetLastError(), "radiomap_matched_isb_completion_forward_kernel launch");
}

void radiomap_matched_isb_completion_jvp(
    const float* continued_direct_re,
    const float* continued_direct_im,
    const float* tx_basis_x,
    const float* tx_basis_y,
    const float* tx_basis_z,
    const float* rx_basis_x,
    const float* rx_basis_y,
    const float* rx_basis_z,
    const float* hard_visibility,
    const int* interior_mask,
    const float* incident_weight,
    const float* incident_response_re,
    const float* incident_response_im,
    const float* raw_vec_x_re,
    const float* raw_vec_x_im,
    const float* raw_vec_y_re,
    const float* raw_vec_y_im,
    const float* raw_vec_z_re,
    const float* raw_vec_z_im,
    const float* t_continued_direct_re,
    const float* t_continued_direct_im,
    const float* t_tx_basis_x,
    const float* t_tx_basis_y,
    const float* t_tx_basis_z,
    const float* t_rx_basis_x,
    const float* t_rx_basis_y,
    const float* t_rx_basis_z,
    const float* t_incident_weight,
    const float* t_incident_response_re,
    const float* t_incident_response_im,
    const float* t_raw_vec_x_re,
    const float* t_raw_vec_x_im,
    const float* t_raw_vec_y_re,
    const float* t_raw_vec_y_im,
    const float* t_raw_vec_z_re,
    const float* t_raw_vec_z_im,
    float* t_coherent_re,
    float* t_coherent_im,
    float* t_power,
    float* t_vector_x_re,
    float* t_vector_x_im,
    float* t_vector_y_re,
    float* t_vector_y_im,
    float* t_vector_z_re,
    float* t_vector_z_im,
    int n_rx
) {
    if (n_rx <= 0) {
        return;
    }
    constexpr int block_size = 256;
    int grid_size = ceil_div_int(n_rx, block_size);
    radiomap_matched_isb_completion_jvp_kernel<<<grid_size, block_size>>>(
        continued_direct_re,
        continued_direct_im,
        tx_basis_x,
        tx_basis_y,
        tx_basis_z,
        rx_basis_x,
        rx_basis_y,
        rx_basis_z,
        hard_visibility,
        interior_mask,
        incident_weight,
        incident_response_re,
        incident_response_im,
        raw_vec_x_re,
        raw_vec_x_im,
        raw_vec_y_re,
        raw_vec_y_im,
        raw_vec_z_re,
        raw_vec_z_im,
        t_continued_direct_re,
        t_continued_direct_im,
        t_tx_basis_x,
        t_tx_basis_y,
        t_tx_basis_z,
        t_rx_basis_x,
        t_rx_basis_y,
        t_rx_basis_z,
        t_incident_weight,
        t_incident_response_re,
        t_incident_response_im,
        t_raw_vec_x_re,
        t_raw_vec_x_im,
        t_raw_vec_y_re,
        t_raw_vec_y_im,
        t_raw_vec_z_re,
        t_raw_vec_z_im,
        t_coherent_re,
        t_coherent_im,
        t_power,
        t_vector_x_re,
        t_vector_x_im,
        t_vector_y_re,
        t_vector_y_im,
        t_vector_z_re,
        t_vector_z_im,
        n_rx
    );
    throw_cuda(cudaGetLastError(), "radiomap_matched_isb_completion_jvp_kernel launch");
}

void radiomap_matched_isb_completion_backward(
    const float* continued_direct_re,
    const float* continued_direct_im,
    const float* tx_basis_x,
    const float* tx_basis_y,
    const float* tx_basis_z,
    const float* rx_basis_x,
    const float* rx_basis_y,
    const float* rx_basis_z,
    const float* hard_visibility,
    const int* interior_mask,
    const float* incident_weight,
    const float* incident_response_re,
    const float* incident_response_im,
    const float* raw_vec_x_re,
    const float* raw_vec_x_im,
    const float* raw_vec_y_re,
    const float* raw_vec_y_im,
    const float* raw_vec_z_re,
    const float* raw_vec_z_im,
    const float* grad_coherent_re,
    const float* grad_coherent_im,
    const float* grad_power,
    const float* grad_vector_x_re,
    const float* grad_vector_x_im,
    const float* grad_vector_y_re,
    const float* grad_vector_y_im,
    const float* grad_vector_z_re,
    const float* grad_vector_z_im,
    float* grad_continued_direct_re,
    float* grad_continued_direct_im,
    float* grad_tx_basis_x,
    float* grad_tx_basis_y,
    float* grad_tx_basis_z,
    float* grad_rx_basis_x,
    float* grad_rx_basis_y,
    float* grad_rx_basis_z,
    float* grad_incident_weight,
    float* grad_incident_response_re,
    float* grad_incident_response_im,
    float* grad_raw_vec_x_re,
    float* grad_raw_vec_x_im,
    float* grad_raw_vec_y_re,
    float* grad_raw_vec_y_im,
    float* grad_raw_vec_z_re,
    float* grad_raw_vec_z_im,
    int n_rx
) {
    if (n_rx <= 0) {
        return;
    }
    constexpr int block_size = 256;
    int grid_size = ceil_div_int(n_rx, block_size);
    radiomap_matched_isb_completion_backward_kernel<<<grid_size, block_size>>>(
        continued_direct_re,
        continued_direct_im,
        tx_basis_x,
        tx_basis_y,
        tx_basis_z,
        rx_basis_x,
        rx_basis_y,
        rx_basis_z,
        hard_visibility,
        interior_mask,
        incident_weight,
        incident_response_re,
        incident_response_im,
        raw_vec_x_re,
        raw_vec_x_im,
        raw_vec_y_re,
        raw_vec_y_im,
        raw_vec_z_re,
        raw_vec_z_im,
        grad_coherent_re,
        grad_coherent_im,
        grad_power,
        grad_vector_x_re,
        grad_vector_x_im,
        grad_vector_y_re,
        grad_vector_y_im,
        grad_vector_z_re,
        grad_vector_z_im,
        grad_continued_direct_re,
        grad_continued_direct_im,
        grad_tx_basis_x,
        grad_tx_basis_y,
        grad_tx_basis_z,
        grad_rx_basis_x,
        grad_rx_basis_y,
        grad_rx_basis_z,
        grad_incident_weight,
        grad_incident_response_re,
        grad_incident_response_im,
        grad_raw_vec_x_re,
        grad_raw_vec_x_im,
        grad_raw_vec_y_re,
        grad_raw_vec_y_im,
        grad_raw_vec_z_re,
        grad_raw_vec_z_im,
        n_rx
    );
    throw_cuda(cudaGetLastError(), "radiomap_matched_isb_completion_backward_kernel launch");
}

void radiomap_shadow_boundary_incident_statistics_forward(
    const float* tx_x,
    const float* tx_y,
    const float* tx_z,
    const float* rx_x,
    const float* rx_y,
    const float* rx_z,
    const float* edge_pos_x,
    const float* edge_pos_y,
    const float* edge_pos_z,
    const float* edge_dir_x,
    const float* edge_dir_y,
    const float* edge_dir_z,
    const float* n0_x,
    const float* n0_y,
    const float* n0_z,
    const float* nn_x,
    const float* nn_y,
    const float* nn_z,
    const float* wedge_n,
    const float* edge_line_min,
    const float* edge_line_max,
    const int* source_visible,
    float* sum_incident_weight,
    float* max_incident_weight,
    float* weighted_incident_response_real,
    float* weighted_incident_response_imag,
    int* argmax_edge_idx,
    float* second_max_incident_weight,
    int* support_edge_count,
    int n_rx,
    int n_edges,
    float k
) {
    if (n_rx <= 0) {
        return;
    }
    constexpr int block_size = 256;
    int rx_grid = ceil_div_int(n_rx, block_size);
    fill_int_kernel<<<rx_grid, block_size>>>(argmax_edge_idx, n_rx, INT_MAX);
    throw_cuda(cudaGetLastError(), "fill_int_kernel launch");
    if (n_edges <= 0) {
        radiomap_shadow_boundary_incident_stats_finalize_argmax_kernel<<<rx_grid, block_size>>>(
            argmax_edge_idx,
            n_rx
        );
        throw_cuda(cudaGetLastError(), "radiomap_shadow_boundary_incident_stats_finalize_argmax_kernel launch");
        return;
    }
    int n_pairs = n_rx * n_edges;
    int pair_grid = ceil_div_int(n_pairs, block_size);
    radiomap_shadow_boundary_incident_stats_forward_kernel<<<pair_grid, block_size>>>(
        tx_x,
        tx_y,
        tx_z,
        rx_x,
        rx_y,
        rx_z,
        edge_pos_x,
        edge_pos_y,
        edge_pos_z,
        edge_dir_x,
        edge_dir_y,
        edge_dir_z,
        n0_x,
        n0_y,
        n0_z,
        nn_x,
        nn_y,
        nn_z,
        wedge_n,
        edge_line_min,
        edge_line_max,
        source_visible,
        sum_incident_weight,
        max_incident_weight,
        weighted_incident_response_real,
        weighted_incident_response_imag,
        support_edge_count,
        n_rx,
        n_edges,
        k
    );
    throw_cuda(cudaGetLastError(), "radiomap_shadow_boundary_incident_stats_forward_kernel launch");
    radiomap_shadow_boundary_incident_stats_argmax_kernel<<<pair_grid, block_size>>>(
        tx_x,
        tx_y,
        tx_z,
        rx_x,
        rx_y,
        rx_z,
        edge_pos_x,
        edge_pos_y,
        edge_pos_z,
        edge_dir_x,
        edge_dir_y,
        edge_dir_z,
        n0_x,
        n0_y,
        n0_z,
        nn_x,
        nn_y,
        nn_z,
        wedge_n,
        edge_line_min,
        edge_line_max,
        source_visible,
        max_incident_weight,
        argmax_edge_idx,
        n_rx,
        n_edges,
        k
    );
    throw_cuda(cudaGetLastError(), "radiomap_shadow_boundary_incident_stats_argmax_kernel launch");
    radiomap_shadow_boundary_incident_stats_finalize_argmax_kernel<<<rx_grid, block_size>>>(
        argmax_edge_idx,
        n_rx
    );
    throw_cuda(cudaGetLastError(), "radiomap_shadow_boundary_incident_stats_finalize_argmax_kernel launch");
    radiomap_shadow_boundary_incident_stats_second_max_kernel<<<pair_grid, block_size>>>(
        tx_x,
        tx_y,
        tx_z,
        rx_x,
        rx_y,
        rx_z,
        edge_pos_x,
        edge_pos_y,
        edge_pos_z,
        edge_dir_x,
        edge_dir_y,
        edge_dir_z,
        n0_x,
        n0_y,
        n0_z,
        nn_x,
        nn_y,
        nn_z,
        wedge_n,
        edge_line_min,
        edge_line_max,
        source_visible,
        max_incident_weight,
        argmax_edge_idx,
        second_max_incident_weight,
        n_rx,
        n_edges,
        k
    );
    throw_cuda(cudaGetLastError(), "radiomap_shadow_boundary_incident_stats_second_max_kernel launch");
}

void radiomap_shadow_boundary_incident_statistics_jvp(
    const float* tx_x,
    const float* tx_y,
    const float* tx_z,
    const float* rx_x,
    const float* rx_y,
    const float* rx_z,
    const float* edge_pos_x,
    const float* edge_pos_y,
    const float* edge_pos_z,
    const float* edge_dir_x,
    const float* edge_dir_y,
    const float* edge_dir_z,
    const float* n0_x,
    const float* n0_y,
    const float* n0_z,
    const float* nn_x,
    const float* nn_y,
    const float* nn_z,
    const float* wedge_n,
    const float* edge_line_min,
    const float* edge_line_max,
    const int* source_visible,
    const int* argmax_edge_idx,
    const float* t_tx_x,
    const float* t_tx_y,
    const float* t_tx_z,
    const float* t_rx_x,
    const float* t_rx_y,
    const float* t_rx_z,
    const float* t_edge_pos_x,
    const float* t_edge_pos_y,
    const float* t_edge_pos_z,
    float* t_sum_incident_weight,
    float* t_max_incident_weight,
    float* t_weighted_incident_response_real,
    float* t_weighted_incident_response_imag,
    int n_rx,
    int n_edges,
    float k
) {
    if (n_rx <= 0 || n_edges <= 0) {
        return;
    }
    constexpr int block_size = 256;
    int n_pairs = n_rx * n_edges;
    int pair_grid = ceil_div_int(n_pairs, block_size);
    radiomap_shadow_boundary_incident_stats_jvp_kernel<<<pair_grid, block_size>>>(
        tx_x,
        tx_y,
        tx_z,
        rx_x,
        rx_y,
        rx_z,
        edge_pos_x,
        edge_pos_y,
        edge_pos_z,
        edge_dir_x,
        edge_dir_y,
        edge_dir_z,
        n0_x,
        n0_y,
        n0_z,
        nn_x,
        nn_y,
        nn_z,
        wedge_n,
        edge_line_min,
        edge_line_max,
        source_visible,
        argmax_edge_idx,
        t_tx_x,
        t_tx_y,
        t_tx_z,
        t_rx_x,
        t_rx_y,
        t_rx_z,
        t_edge_pos_x,
        t_edge_pos_y,
        t_edge_pos_z,
        t_sum_incident_weight,
        t_max_incident_weight,
        t_weighted_incident_response_real,
        t_weighted_incident_response_imag,
        n_rx,
        n_edges,
        k
    );
    throw_cuda(cudaGetLastError(), "radiomap_shadow_boundary_incident_stats_jvp_kernel launch");
}

void radiomap_shadow_boundary_incident_statistics_backward(
    const float* tx_x,
    const float* tx_y,
    const float* tx_z,
    const float* rx_x,
    const float* rx_y,
    const float* rx_z,
    const float* edge_pos_x,
    const float* edge_pos_y,
    const float* edge_pos_z,
    const float* edge_dir_x,
    const float* edge_dir_y,
    const float* edge_dir_z,
    const float* n0_x,
    const float* n0_y,
    const float* n0_z,
    const float* nn_x,
    const float* nn_y,
    const float* nn_z,
    const float* wedge_n,
    const float* edge_line_min,
    const float* edge_line_max,
    const int* source_visible,
    const int* argmax_edge_idx,
    const float* grad_sum_incident_weight,
    const float* grad_max_incident_weight,
    const float* grad_weighted_incident_response_real,
    const float* grad_weighted_incident_response_imag,
    float* grad_tx_x,
    float* grad_tx_y,
    float* grad_tx_z,
    float* grad_rx_x,
    float* grad_rx_y,
    float* grad_rx_z,
    float* grad_edge_pos_x,
    float* grad_edge_pos_y,
    float* grad_edge_pos_z,
    int n_rx,
    int n_edges,
    float k
) {
    if (n_rx <= 0 || n_edges <= 0) {
        return;
    }
    constexpr int block_size = 256;
    int n_pairs = n_rx * n_edges;
    int pair_grid = ceil_div_int(n_pairs, block_size);
    radiomap_shadow_boundary_incident_stats_backward_kernel<<<pair_grid, block_size>>>(
        tx_x,
        tx_y,
        tx_z,
        rx_x,
        rx_y,
        rx_z,
        edge_pos_x,
        edge_pos_y,
        edge_pos_z,
        edge_dir_x,
        edge_dir_y,
        edge_dir_z,
        n0_x,
        n0_y,
        n0_z,
        nn_x,
        nn_y,
        nn_z,
        wedge_n,
        edge_line_min,
        edge_line_max,
        source_visible,
        argmax_edge_idx,
        grad_sum_incident_weight,
        grad_max_incident_weight,
        grad_weighted_incident_response_real,
        grad_weighted_incident_response_imag,
        grad_tx_x,
        grad_tx_y,
        grad_tx_z,
        grad_rx_x,
        grad_rx_y,
        grad_rx_z,
        grad_edge_pos_x,
        grad_edge_pos_y,
        grad_edge_pos_z,
        n_rx,
        n_edges,
        k
    );
    throw_cuda(cudaGetLastError(), "radiomap_shadow_boundary_incident_stats_backward_kernel launch");
}

} // namespace witwin::channel::native_ext
