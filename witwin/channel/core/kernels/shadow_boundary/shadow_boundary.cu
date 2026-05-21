#include <cuda_runtime.h>

#include <common/cuda_check.h>
#include <shadow_boundary/shadow_boundary.h>
#include <utd/utd_math.h>
#include <utd/utd_types.h>

#include <cmath>
#include <stdexcept>
#include <string>

namespace witwin::channel::native_ext {
namespace {

using common::throw_cuda;

constexpr int BLOCK_SIZE = 128;
constexpr float MIN_WEDGE_N = 1.01f;
constexpr float RESPONSE_WEIGHT_EPS = 1.0e-6f;

struct ShadowState {
    float3a edge_pos;
    float3a edge_dir;
    float3a n0;
    float3a nn;
    float wedge_n;
    float edge_line_min;
    float edge_line_max;
    float3a source_pos;
    int adjacent_group0;
    int adjacent_group1;
};

struct TransitionTerms {
    bool support;
    float incident_arg;
    float reflection_arg;
    float incident_weight;
    float reflection_weight;
    Complex incident_response;
    Complex reflection_response;
};

struct FiniteTerms {
    bool valid;
    Complex factor;
    float scale;
    float support_weight;
};

UTD_DINLINE float clamp_local(float value, float lo, float hi) {
    return fminf(fmaxf(value, lo), hi);
}

UTD_DINLINE float3a normalize_like_drjit(float3a vec, float3a fallback) {
    float vec_norm = f3_len(vec);
    float fallback_norm = f3_len(fallback);
    float3a safe_fallback = fallback_norm > UTD_EPS
        ? f3_div(fallback, fallback_norm)
        : make_f3(1.0f, 0.0f, 0.0f);
    return vec_norm > UTD_EPS ? f3_div(vec, vec_norm) : safe_fallback;
}

UTD_DINLINE bool finite_float(float value) {
    return isfinite(value);
}

UTD_DINLINE void atomic_max_positive_float(float *address, float value) {
    if (!(value > 0.0f) || !finite_float(value)) {
        return;
    }
    unsigned int *address_as_uint = reinterpret_cast<unsigned int *>(address);
    unsigned int old = *address_as_uint;
    while (__uint_as_float(old) < value) {
        unsigned int assumed = old;
        old = atomicCAS(address_as_uint, assumed, __float_as_uint(value));
        if (old == assumed) {
            break;
        }
    }
}

UTD_DINLINE Complex sanitize_complex(Complex value) {
    return cplx(
        finite_float(value.re) ? value.re : 0.0f,
        finite_float(value.im) ? value.im : 0.0f
    );
}

UTD_DINLINE Complex transition_response(float arg) {
    return sanitize_complex(f_utd_value(fmaxf(arg, 0.0f)));
}

UTD_DINLINE float transition_weight(float arg) {
    Complex value = transition_response(arg);
    float magnitude = sqrtf(fmaxf(cplx_abs_sqr(value), 0.0f));
    float weight = 1.0f - fminf(magnitude, 1.0f);
    weight = clamp_local(weight, 0.0f, 1.0f);
    return finite_float(weight) ? weight : 0.0f;
}

UTD_DINLINE float a_coeff_min(float beta, float exterior_angle) {
    float n_p = roundf((beta + UTD_PI) / (2.0f * exterior_angle));
    float n_m = roundf((beta - UTD_PI) / (2.0f * exterior_angle));
    float a_p_cos = cosf(exterior_angle * n_p - 0.5f * beta);
    float a_m_cos = cosf(exterior_angle * n_m - 0.5f * beta);
    float a_p = 2.0f * a_p_cos * a_p_cos;
    float a_m = 2.0f * a_m_cos * a_m_cos;
    return fminf(a_p, a_m);
}

UTD_DINLINE ShadowState load_state(
    int index,
    const float *edge_pos_x,
    const float *edge_pos_y,
    const float *edge_pos_z,
    const float *edge_dir_x,
    const float *edge_dir_y,
    const float *edge_dir_z,
    const float *edge_n0_x,
    const float *edge_n0_y,
    const float *edge_n0_z,
    const float *edge_nn_x,
    const float *edge_nn_y,
    const float *edge_nn_z,
    const float *edge_wedge_n,
    const float *edge_line_min,
    const float *edge_line_max,
    const float *source_pos_x,
    const float *source_pos_y,
    const float *source_pos_z,
    const int *edge_adjacent_group0,
    const int *edge_adjacent_group1
) {
    ShadowState state;
    state.edge_pos = make_f3(edge_pos_x[index], edge_pos_y[index], edge_pos_z[index]);
    state.edge_dir = make_f3(edge_dir_x[index], edge_dir_y[index], edge_dir_z[index]);
    state.n0 = make_f3(edge_n0_x[index], edge_n0_y[index], edge_n0_z[index]);
    state.nn = make_f3(edge_nn_x[index], edge_nn_y[index], edge_nn_z[index]);
    state.wedge_n = edge_wedge_n[index];
    state.edge_line_min = edge_line_min[index];
    state.edge_line_max = edge_line_max[index];
    state.source_pos = make_f3(source_pos_x[index], source_pos_y[index], source_pos_z[index]);
    state.adjacent_group0 = edge_adjacent_group0[index];
    state.adjacent_group1 = edge_adjacent_group1[index];
    return state;
}

UTD_DINLINE bool incident_cell_supported(
    ShadowState state,
    unsigned int direct_los_visible,
    int direct_blocker_group
) {
    if (direct_los_visible != 0u) {
        return true;
    }
    if (direct_blocker_group < 0) {
        return false;
    }
    return (
        (state.adjacent_group0 >= 0 && direct_blocker_group == state.adjacent_group0)
        || (state.adjacent_group1 >= 0 && direct_blocker_group == state.adjacent_group1)
    );
}

UTD_DINLINE ShadowState orient_state(ShadowState state) {
    float3a incident_dir = f3_sub(state.edge_pos, state.source_pos);
    if (f3_dot(incident_dir, state.n0) > 0.0f) {
        float3a old_n0 = state.n0;
        state.edge_dir = f3_neg(state.edge_dir);
        state.n0 = state.nn;
        state.nn = old_n0;
    }
    return state;
}

UTD_DINLINE bool transition_arguments(
    ShadowState oriented,
    float3a target_pos,
    float k,
    float& incident_arg,
    float& reflection_arg
) {
    float phi;
    float phi_prime;
    float s;
    float s_prime;
    float sin_beta0;
    compute_edge_geometry_3d(
        oriented.source_pos,
        oriented.edge_pos,
        oriented.edge_dir,
        oriented.n0,
        target_pos,
        phi,
        phi_prime,
        s,
        s_prime,
        sin_beta0
    );
    bool source_exterior = wedge_exterior_mask(
        f3_sub(oriented.source_pos, oriented.edge_pos),
        oriented.edge_dir,
        oriented.n0,
        oriented.nn
    );
    bool target_exterior = wedge_exterior_mask(
        f3_sub(target_pos, oriented.edge_pos),
        oriented.edge_dir,
        oriented.n0,
        oriented.nn
    );
    bool field_valid = source_exterior
        && (s_prime > UTD_MIN_DISTANCE)
        && (s > UTD_MIN_DISTANCE);
    bool pole_safe = field_valid
        && cot_pole_safe_mask(phi, phi_prime, oriented.wedge_n, 1.0e-6f);
    float phi_eval = pole_safe ? phi : 0.5f * oriented.wedge_n * UTD_PI;
    float phi_prime_eval = pole_safe ? phi_prime : 0.5f * oriented.wedge_n * UTD_PI;
    float exterior_angle = oriented.wedge_n * UTD_PI;
    float l = (
        s * s_prime / (s + s_prime + UTD_EPS)
    ) * sin_beta0 * sin_beta0;
    incident_arg = k * l * a_coeff_min(phi_eval - phi_prime_eval, exterior_angle);
    reflection_arg = k * l * a_coeff_min(phi_eval + phi_prime_eval, exterior_angle);
    return target_exterior
        && field_valid
        && pole_safe
        && (s > UTD_MIN_DISTANCE)
        && (s_prime > UTD_MIN_DISTANCE)
        && (oriented.wedge_n > MIN_WEDGE_N);
}

UTD_DINLINE TransitionTerms transition_terms(
    ShadowState oriented,
    float3a target_pos,
    float k
) {
    TransitionTerms terms;
    terms.incident_arg = 0.0f;
    terms.reflection_arg = 0.0f;
    terms.support = transition_arguments(
        oriented,
        target_pos,
        k,
        terms.incident_arg,
        terms.reflection_arg
    );
    if (!terms.support) {
        terms.incident_weight = 0.0f;
        terms.reflection_weight = 0.0f;
        terms.incident_response = cplx_zero();
        terms.reflection_response = cplx_zero();
        return terms;
    }
    terms.incident_response = transition_response(terms.incident_arg);
    terms.reflection_response = transition_response(terms.reflection_arg);
    terms.incident_weight = transition_weight(terms.incident_arg);
    terms.reflection_weight = transition_weight(terms.reflection_arg);
    return terms;
}

UTD_DINLINE FiniteTerms finite_edge_factor(
    ShadowState state,
    float3a target_pos,
    float k
) {
    FiniteTerms out;
    out.valid = false;
    out.factor = cplx_zero();
    out.scale = 0.0f;
    out.support_weight = 0.0f;

    float3a edge_hat = normalize_like_drjit(state.edge_dir, make_f3(0.0f, 0.0f, 1.0f));
    float source_axial = f3_dot(f3_sub(state.source_pos, state.edge_pos), edge_hat);
    float target_axial = f3_dot(f3_sub(target_pos, state.edge_pos), edge_hat);
    float3a source_to_edge = f3_sub(state.edge_pos, state.source_pos);
    float3a edge_to_target = f3_sub(target_pos, state.edge_pos);
    float3a source_proj = f3_sub(source_to_edge, f3_mul(edge_hat, f3_dot(source_to_edge, edge_hat)));
    float3a target_proj = f3_sub(edge_to_target, f3_mul(edge_hat, f3_dot(edge_to_target, edge_hat)));
    float s_prime_proj = f3_len(source_proj) + UTD_EPS;
    float s_proj = f3_len(target_proj) + UTD_EPS;
    float stationary_u = (
        s_prime_proj * target_axial + s_proj * source_axial
    ) / (s_proj + s_prime_proj + UTD_EPS);
    float source_offset = stationary_u - source_axial;
    float target_offset = target_axial - stationary_u;
    float source_range = sqrtf(
        s_prime_proj * s_prime_proj + source_offset * source_offset + UTD_EPS
    );
    float target_range = sqrtf(
        s_proj * s_proj + target_offset * target_offset + UTD_EPS
    );
    float curvature = (
        s_prime_proj * s_prime_proj
        / (source_range * source_range * source_range + UTD_EPS)
        + s_proj * s_proj
        / (target_range * target_range * target_range + UTD_EPS)
    );
    float scale = sqrtf(fmaxf(k * curvature, UTD_EPS) / UTD_PI);
    Complex fresnel_max;
    Complex fresnel_min;
    Complex unused_first;
    Complex unused_second;
    fresnel_boersma(scale * (state.edge_line_max - stationary_u), fresnel_max, unused_first, unused_second);
    fresnel_boersma(scale * (state.edge_line_min - stationary_u), fresnel_min, unused_first, unused_second);
    Complex delta_f = cplx_sub(fresnel_max, fresnel_min);
    Complex finite_factor = cplx_mul(cplx(0.5f, 0.5f), cplx_conj(delta_f));
    float finite_scale = sqrtf(fmaxf(cplx_abs_sqr(finite_factor), 0.0f));
    finite_scale = fminf(finite_scale, 1.0f);
    if (!finite_float(finite_scale)) {
        finite_scale = 0.0f;
    }
    bool line_length_valid = (state.edge_line_max - state.edge_line_min) > UTD_EPS;
    if (!line_length_valid) {
        return out;
    }
    bool stationary_on_segment = (
        stationary_u >= state.edge_line_min - UTD_EPS
        && stationary_u <= state.edge_line_max + UTD_EPS
    );
    float line_center = 0.5f * (state.edge_line_min + state.edge_line_max);
    float line_half_length = 0.5f * (state.edge_line_max - state.edge_line_min);
    float interior_weight = (
        line_half_length - fabsf(stationary_u - line_center)
    ) / fmaxf(line_half_length, UTD_EPS);
    interior_weight = clamp_local(interior_weight, 0.0f, 1.0f);
    float support_weight = 0.25f + 0.75f * interior_weight;
    support_weight = stationary_on_segment ? support_weight : 0.0f;
    out.valid = true;
    out.factor = sanitize_complex(finite_factor);
    out.scale = finite_scale;
    out.support_weight = support_weight;
    return out;
}

UTD_DINLINE bool cell_is_candidate(
    ShadowState state,
    float3a target_pos,
    float k,
    float arg_cutoff,
    bool incident_supported
) {
    ShadowState oriented = orient_state(state);
    float incident_arg;
    float reflection_arg;
    bool support = transition_arguments(oriented, target_pos, k, incident_arg, reflection_arg);
    if (!support) {
        return false;
    }
    FiniteTerms finite = finite_edge_factor(state, target_pos, k);
    if (!finite.valid || finite.support_weight <= 0.0f || finite.scale <= 0.0f) {
        return false;
    }
    bool incident_in_band = incident_arg <= arg_cutoff;
    bool reflection_in_band = reflection_arg <= arg_cutoff;
    if (!incident_in_band && !reflection_in_band) {
        return false;
    }
    float incident_weight = (incident_supported && incident_in_band)
        ? transition_weight(incident_arg)
        : 0.0f;
    float reflection_weight = reflection_in_band ? transition_weight(reflection_arg) : 0.0f;
    return incident_weight > 0.0f || reflection_weight > 0.0f;
}

__global__ void shadow_candidate_kernel(
    const float *edge_pos_x,
    const float *edge_pos_y,
    const float *edge_pos_z,
    const float *edge_dir_x,
    const float *edge_dir_y,
    const float *edge_dir_z,
    const float *edge_n0_x,
    const float *edge_n0_y,
    const float *edge_n0_z,
    const float *edge_nn_x,
    const float *edge_nn_y,
    const float *edge_nn_z,
    const float *edge_wedge_n,
    const float *edge_line_min,
    const float *edge_line_max,
    const float *source_pos_x,
    const float *source_pos_y,
    const float *source_pos_z,
    const unsigned int *direct_los_visible,
    const int *direct_blocker_group,
    const int *edge_adjacent_group0,
    const int *edge_adjacent_group1,
    const float *cell_x,
    const float *cell_y,
    const float *cell_z,
    int n_edges,
    int grid_nx,
    int grid_ny,
    int tile_nx,
    int tile_ny,
    int tiles_x,
    int tiles_y,
    float k,
    float arg_cutoff,
    unsigned int *candidate_mask,
    unsigned int *candidate_tile_count,
    unsigned int *candidate_cell_count
) {
    int tile_count = tiles_x * tiles_y;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = n_edges * tile_count;
    if (tid >= total) {
        return;
    }
    int edge_index = tid / tile_count;
    int tile_index = tid - edge_index * tile_count;
    int tile_x = tile_index % tiles_x;
    int tile_y = tile_index / tiles_x;
    int x0 = tile_x * tile_nx;
    int y0 = tile_y * tile_ny;
    int x1 = min(x0 + tile_nx, grid_nx);
    int y1 = min(y0 + tile_ny, grid_ny);

    ShadowState state = load_state(
        edge_index,
        edge_pos_x,
        edge_pos_y,
        edge_pos_z,
        edge_dir_x,
        edge_dir_y,
        edge_dir_z,
        edge_n0_x,
        edge_n0_y,
        edge_n0_z,
        edge_nn_x,
        edge_nn_y,
        edge_nn_z,
        edge_wedge_n,
        edge_line_min,
        edge_line_max,
        source_pos_x,
        source_pos_y,
        source_pos_z,
        edge_adjacent_group0,
        edge_adjacent_group1
    );

    unsigned int local_candidate_cells = 0u;
    for (int y = y0; y < y1; ++y) {
        for (int x = x0; x < x1; ++x) {
            int cell_index = y * grid_nx + x;
            float3a target_pos = make_f3(cell_x[cell_index], cell_y[cell_index], cell_z[cell_index]);
            bool incident_supported = incident_cell_supported(
                state,
                direct_los_visible[cell_index],
                direct_blocker_group[cell_index]
            );
            if (cell_is_candidate(state, target_pos, k, arg_cutoff, incident_supported)) {
                local_candidate_cells += 1u;
            }
        }
    }
    bool candidate = local_candidate_cells > 0u;
    candidate_mask[tid] = candidate ? 1u : 0u;
    if (candidate) {
        atomicAdd(candidate_tile_count, 1u);
        atomicAdd(candidate_cell_count, local_candidate_cells);
    }
}

__global__ void shadow_eval_kernel(
    const float *edge_pos_x,
    const float *edge_pos_y,
    const float *edge_pos_z,
    const float *edge_dir_x,
    const float *edge_dir_y,
    const float *edge_dir_z,
    const float *edge_n0_x,
    const float *edge_n0_y,
    const float *edge_n0_z,
    const float *edge_nn_x,
    const float *edge_nn_y,
    const float *edge_nn_z,
    const float *edge_wedge_n,
    const float *edge_line_min,
    const float *edge_line_max,
    const float *source_pos_x,
    const float *source_pos_y,
    const float *source_pos_z,
    const unsigned int *direct_los_visible,
    const int *direct_blocker_group,
    const int *edge_adjacent_group0,
    const int *edge_adjacent_group1,
    const float *cell_x,
    const float *cell_y,
    const float *cell_z,
    int n_edges,
    int grid_nx,
    int grid_ny,
    int tile_nx,
    int tile_ny,
    int tiles_x,
    int tiles_y,
    float k,
    const unsigned int *candidate_mask,
    float *out_incident_weight,
    float *out_reflection_weight,
    float *incident_weight_sum,
    float *reflection_weight_sum,
    float *out_incident_response_real,
    float *out_incident_response_imag,
    float *out_reflection_response_real,
    float *out_reflection_response_imag
) {
    int tile_count = tiles_x * tiles_y;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = n_edges * tile_count;
    if (tid >= total || candidate_mask[tid] == 0u) {
        return;
    }
    int edge_index = tid / tile_count;
    int tile_index = tid - edge_index * tile_count;
    int tile_x = tile_index % tiles_x;
    int tile_y = tile_index / tiles_x;
    int x0 = tile_x * tile_nx;
    int y0 = tile_y * tile_ny;
    int x1 = min(x0 + tile_nx, grid_nx);
    int y1 = min(y0 + tile_ny, grid_ny);

    ShadowState state = load_state(
        edge_index,
        edge_pos_x,
        edge_pos_y,
        edge_pos_z,
        edge_dir_x,
        edge_dir_y,
        edge_dir_z,
        edge_n0_x,
        edge_n0_y,
        edge_n0_z,
        edge_nn_x,
        edge_nn_y,
        edge_nn_z,
        edge_wedge_n,
        edge_line_min,
        edge_line_max,
        source_pos_x,
        source_pos_y,
        source_pos_z,
        edge_adjacent_group0,
        edge_adjacent_group1
    );
    ShadowState oriented = orient_state(state);

    for (int y = y0; y < y1; ++y) {
        for (int x = x0; x < x1; ++x) {
            int cell_index = y * grid_nx + x;
            float3a target_pos = make_f3(cell_x[cell_index], cell_y[cell_index], cell_z[cell_index]);
            TransitionTerms transition = transition_terms(oriented, target_pos, k);
            if (!transition.support) {
                continue;
            }
            FiniteTerms finite = finite_edge_factor(state, target_pos, k);
            if (!finite.valid || finite.support_weight <= 0.0f || finite.scale <= 0.0f) {
                continue;
            }
            float finite_amplitude = sqrtf(fmaxf(finite.scale, 0.0f));
            float finite_weight = finite_amplitude * finite.support_weight;
            float response_weight = finite.support_weight > 0.0f ? finite_amplitude : 0.0f;
            bool incident_supported = incident_cell_supported(
                state,
                direct_los_visible[cell_index],
                direct_blocker_group[cell_index]
            );
            float edge_incident = (
                incident_supported ? transition.incident_weight : 0.0f
            ) * finite_weight;
            float edge_reflection = transition.reflection_weight * finite_weight;
            float edge_incident_response = (
                incident_supported ? transition.incident_weight : 0.0f
            ) * response_weight;
            float edge_reflection_response = transition.reflection_weight * response_weight;
            if (edge_incident > 0.0f) {
                Complex response = cplx_mul(finite.factor, transition.incident_response);
                atomicAdd(incident_weight_sum + cell_index, edge_incident_response);
                atomic_max_positive_float(out_incident_weight + cell_index, edge_incident);
                atomicAdd(out_incident_response_real + cell_index, edge_incident_response * response.re);
                atomicAdd(out_incident_response_imag + cell_index, edge_incident_response * response.im);
            }
            if (edge_reflection > 0.0f) {
                Complex response = cplx_mul(finite.factor, transition.reflection_response);
                atomicAdd(reflection_weight_sum + cell_index, edge_reflection_response);
                atomic_max_positive_float(out_reflection_weight + cell_index, edge_reflection);
                atomicAdd(out_reflection_response_real + cell_index, edge_reflection_response * response.re);
                atomicAdd(out_reflection_response_imag + cell_index, edge_reflection_response * response.im);
            }
        }
    }
}

__global__ void shadow_finalize_kernel(
    int n_cells,
    float *out_incident_weight,
    float *out_reflection_weight,
    const float *incident_weight_sum,
    const float *reflection_weight_sum,
    float *out_incident_response_real,
    float *out_incident_response_imag,
    float *out_reflection_response_real,
    float *out_reflection_response_imag
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_cells) {
        return;
    }
    float incident_weight = out_incident_weight[tid];
    float reflection_weight = out_reflection_weight[tid];
    float incident_sum = incident_weight_sum[tid];
    float reflection_sum = reflection_weight_sum[tid];
    if (incident_sum > RESPONSE_WEIGHT_EPS) {
        float safe_weight = fmaxf(incident_sum, RESPONSE_WEIGHT_EPS);
        out_incident_response_real[tid] /= safe_weight;
        out_incident_response_imag[tid] /= safe_weight;
    } else {
        out_incident_response_real[tid] = 0.0f;
        out_incident_response_imag[tid] = 0.0f;
    }
    if (reflection_sum > RESPONSE_WEIGHT_EPS) {
        float safe_weight = fmaxf(reflection_sum, RESPONSE_WEIGHT_EPS);
        out_reflection_response_real[tid] /= safe_weight;
        out_reflection_response_imag[tid] /= safe_weight;
    } else {
        out_reflection_response_real[tid] = 0.0f;
        out_reflection_response_imag[tid] = 0.0f;
    }
    out_incident_weight[tid] = clamp_local(incident_weight, 0.0f, 1.0f);
    out_reflection_weight[tid] = clamp_local(reflection_weight, 0.0f, 1.0f);
}

} // namespace

void shadow_boundary_candidate_accumulate(
    const float *edge_pos_x,
    const float *edge_pos_y,
    const float *edge_pos_z,
    const float *edge_dir_x,
    const float *edge_dir_y,
    const float *edge_dir_z,
    const float *edge_n0_x,
    const float *edge_n0_y,
    const float *edge_n0_z,
    const float *edge_nn_x,
    const float *edge_nn_y,
    const float *edge_nn_z,
    const float *edge_wedge_n,
    const float *edge_line_min,
    const float *edge_line_max,
    const float *source_pos_x,
    const float *source_pos_y,
    const float *source_pos_z,
    const unsigned int *direct_los_visible,
    const int *direct_blocker_group,
    const int *edge_adjacent_group0,
    const int *edge_adjacent_group1,
    const float *cell_x,
    const float *cell_y,
    const float *cell_z,
    int n_edges,
    int grid_nx,
    int grid_ny,
    int tile_nx,
    int tile_ny,
    float k,
    float wavelength,
    float band_width_wavelengths,
    float max_candidate_factor,
    float *out_incident_weight,
    float *out_reflection_weight,
    float *out_incident_response_real,
    float *out_incident_response_imag,
    float *out_reflection_response_real,
    float *out_reflection_response_imag,
    unsigned int *out_candidate_tile_count,
    unsigned int *out_candidate_cell_count
) {
    if (n_edges <= 0 || grid_nx <= 0 || grid_ny <= 0) {
        return;
    }
    tile_nx = tile_nx > 0 ? tile_nx : 8;
    tile_ny = tile_ny > 0 ? tile_ny : 8;
    int tiles_x = (grid_nx + tile_nx - 1) / tile_nx;
    int tiles_y = (grid_ny + tile_ny - 1) / tile_ny;
    int tile_count = tiles_x * tiles_y;
    int edge_tile_count = n_edges * tile_count;
    if (edge_tile_count <= 0) {
        return;
    }

    unsigned int *candidate_mask = nullptr;
    throw_cuda(
        cudaMalloc(
            reinterpret_cast<void **>(&candidate_mask),
            static_cast<size_t>(edge_tile_count) * sizeof(unsigned int)
        ),
        "shadow_boundary candidate mask allocation"
    );
    throw_cuda(
        cudaMemset(candidate_mask, 0, static_cast<size_t>(edge_tile_count) * sizeof(unsigned int)),
        "shadow_boundary candidate mask memset"
    );

    float safe_wavelength = fmaxf(wavelength, UTD_SMALL_EPS);
    float arg_cutoff = fmaxf(
        k * safe_wavelength * band_width_wavelengths,
        UTD_TWO_PI * band_width_wavelengths
    );
    int blocks = (edge_tile_count + BLOCK_SIZE - 1) / BLOCK_SIZE;
    shadow_candidate_kernel<<<blocks, BLOCK_SIZE>>>(
        edge_pos_x,
        edge_pos_y,
        edge_pos_z,
        edge_dir_x,
        edge_dir_y,
        edge_dir_z,
        edge_n0_x,
        edge_n0_y,
        edge_n0_z,
        edge_nn_x,
        edge_nn_y,
        edge_nn_z,
        edge_wedge_n,
        edge_line_min,
        edge_line_max,
        source_pos_x,
        source_pos_y,
        source_pos_z,
        direct_los_visible,
        direct_blocker_group,
        edge_adjacent_group0,
        edge_adjacent_group1,
        cell_x,
        cell_y,
        cell_z,
        n_edges,
        grid_nx,
        grid_ny,
        tile_nx,
        tile_ny,
        tiles_x,
        tiles_y,
        k,
        arg_cutoff,
        candidate_mask,
        out_candidate_tile_count,
        out_candidate_cell_count
    );
    throw_cuda(cudaGetLastError(), "shadow_boundary candidate kernel launch");

    unsigned int host_candidate_tiles = 0;
    unsigned int host_candidate_cells = 0;
    throw_cuda(
        cudaMemcpy(
            &host_candidate_tiles,
            out_candidate_tile_count,
            sizeof(unsigned int),
            cudaMemcpyDeviceToHost
        ),
        "shadow_boundary candidate count copy"
    );
    throw_cuda(
        cudaMemcpy(
            &host_candidate_cells,
            out_candidate_cell_count,
            sizeof(unsigned int),
            cudaMemcpyDeviceToHost
        ),
        "shadow_boundary candidate cell count copy"
    );
    unsigned long long candidate_pairs =
        static_cast<unsigned long long>(host_candidate_cells);
    unsigned long long max_candidate_pairs =
        static_cast<unsigned long long>(
            ceilf(
                fmaxf(max_candidate_factor, UTD_SMALL_EPS)
                * static_cast<float>(grid_nx * grid_ny)
            )
        );
    if (candidate_pairs > max_candidate_pairs) {
        throw_cuda(cudaFree(candidate_mask), "shadow_boundary candidate mask free");
        throw std::runtime_error(
            "shadow_boundary native_candidate produced "
            + std::to_string(candidate_pairs)
            + " edge-cell candidates; max allowed is "
            + std::to_string(max_candidate_pairs)
            + ". "
            "increase shadow_boundary_max_candidate_factor, reduce band width, or use "
            "shadow_boundary_mode='none'."
        );
    }

    int n_cells = grid_nx * grid_ny;
    float *incident_weight_sum = nullptr;
    float *reflection_weight_sum = nullptr;
    throw_cuda(
        cudaMalloc(
            reinterpret_cast<void **>(&incident_weight_sum),
            static_cast<size_t>(n_cells) * sizeof(float)
        ),
        "shadow_boundary incident weight sum allocation"
    );
    throw_cuda(
        cudaMalloc(
            reinterpret_cast<void **>(&reflection_weight_sum),
            static_cast<size_t>(n_cells) * sizeof(float)
        ),
        "shadow_boundary reflection weight sum allocation"
    );
    throw_cuda(
        cudaMemset(incident_weight_sum, 0, static_cast<size_t>(n_cells) * sizeof(float)),
        "shadow_boundary incident weight sum memset"
    );
    throw_cuda(
        cudaMemset(reflection_weight_sum, 0, static_cast<size_t>(n_cells) * sizeof(float)),
        "shadow_boundary reflection weight sum memset"
    );

    shadow_eval_kernel<<<blocks, BLOCK_SIZE>>>(
        edge_pos_x,
        edge_pos_y,
        edge_pos_z,
        edge_dir_x,
        edge_dir_y,
        edge_dir_z,
        edge_n0_x,
        edge_n0_y,
        edge_n0_z,
        edge_nn_x,
        edge_nn_y,
        edge_nn_z,
        edge_wedge_n,
        edge_line_min,
        edge_line_max,
        source_pos_x,
        source_pos_y,
        source_pos_z,
        direct_los_visible,
        direct_blocker_group,
        edge_adjacent_group0,
        edge_adjacent_group1,
        cell_x,
        cell_y,
        cell_z,
        n_edges,
        grid_nx,
        grid_ny,
        tile_nx,
        tile_ny,
        tiles_x,
        tiles_y,
        k,
        candidate_mask,
        out_incident_weight,
        out_reflection_weight,
        incident_weight_sum,
        reflection_weight_sum,
        out_incident_response_real,
        out_incident_response_imag,
        out_reflection_response_real,
        out_reflection_response_imag
    );
    throw_cuda(cudaGetLastError(), "shadow_boundary evaluation kernel launch");

    int cell_blocks = (n_cells + BLOCK_SIZE - 1) / BLOCK_SIZE;
    shadow_finalize_kernel<<<cell_blocks, BLOCK_SIZE>>>(
        n_cells,
        out_incident_weight,
        out_reflection_weight,
        incident_weight_sum,
        reflection_weight_sum,
        out_incident_response_real,
        out_incident_response_imag,
        out_reflection_response_real,
        out_reflection_response_imag
    );
    throw_cuda(cudaGetLastError(), "shadow_boundary finalize kernel launch");
    throw_cuda(cudaFree(incident_weight_sum), "shadow_boundary incident weight sum free");
    throw_cuda(cudaFree(reflection_weight_sum), "shadow_boundary reflection weight sum free");
    throw_cuda(cudaFree(candidate_mask), "shadow_boundary candidate mask free");
}

} // namespace witwin::channel::native_ext
