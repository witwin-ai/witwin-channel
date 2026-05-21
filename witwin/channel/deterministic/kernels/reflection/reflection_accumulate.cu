#include <cuda_runtime.h>

#include <common/cuda_check.h>
#include <reflection/reflection_types.h>
#include <reflection/reflection_common.h>
#include <reflection/reflection_accumulate.h>

namespace witwin::channel::native_ext {
namespace {

using namespace reflection_detail;

using common::throw_cuda;

// Apply reflection to a field vector: decompose into s/p, apply Fresnel, recompose.
// This is the CUDA equivalent of reflect_field_vector() from polarization.py.
__device__ __forceinline__ Complex3 reflect_field_vector_material(
    Complex3 field_vec,
    float3a incoming_hat,
    float3a normal_hat,
    float eta_r, float mu_r, float sigma, float omega, float gain)
{
    float3a reflected_dir = reflect_direction(incoming_hat, normal_hat);

    // s-polarization basis (perpendicular to plane of incidence)
    float3a s_pref = f3_cross(normal_hat, incoming_hat);
    float3a s_hat = safe_normalize(s_pref, stable_perp_basis(incoming_hat, make_f3(0, 1, 0)));

    // p-polarization basis (in plane of incidence)
    float3a p_in_hat  = safe_normalize(f3_cross(s_hat, incoming_hat),
                                       stable_perp_basis(incoming_hat, make_f3(1, 0, 0)));
    float3a p_out_hat = safe_normalize(f3_cross(s_hat, reflected_dir),
                                       stable_perp_basis(reflected_dir, make_f3(1, 0, 0)));

    // Angle of incidence
    float cos_theta = fminf(fmaxf(fabsf(f3_dot(incoming_hat, normal_hat)), UTD_SMALL_EPS), 1.f);

    // Fresnel coefficients (reuse from utd_math.h)
    Complex rTE, rTM;
    fresnel_reflection_face(cos_theta, eta_r, mu_r, sigma, omega, rTE, rTM);

    // Apply gain
    Complex gain_c = cplx(gain, 0);
    rTE = cplx_mul(gain_c, rTE);
    rTM = cplx_mul(gain_c, rTM);

    // Sanitize non-finite values
    if (!isfinite(rTE.re)) rTE.re = 0.f;
    if (!isfinite(rTE.im)) rTE.im = 0.f;
    if (!isfinite(rTM.re)) rTM.re = 0.f;
    if (!isfinite(rTM.im)) rTM.im = 0.f;

    // Decompose incident field into s and p components
    Complex e_s = cplx_dot_real(field_vec, s_hat);
    Complex e_p = cplx_dot_real(field_vec, p_in_hat);

    // Recompose reflected field
    Complex3 s_part = cplx_scale_real(s_hat, cplx_mul(rTE, e_s));
    Complex3 p_part = cplx_scale_real(p_out_hat, cplx_mul(rTM, e_p));
    return c3_add(s_part, p_part);
}

struct ReflectionSlotData {
    float3a plane_pt;
    float3a plane_n;
    float eta_r;
    float mu_r;
    float sigma;
    float gain;
};

struct ReflectionChainEval {
    bool geom_valid;
    float3a image_source;
    float3a tx_pos;
    float3a hit_points[REFL_MAX_CHAIN_DEPTH];
    float3a normals[REFL_MAX_CHAIN_DEPTH];
    Complex3 chain_vec;
};

__device__ __forceinline__ ReflectionSlotData load_primary_slot(
    int slot,
    int pI,
    int n_paths,
    const float* __restrict__ slot_pp_x,
    const float* __restrict__ slot_pp_y,
    const float* __restrict__ slot_pp_z,
    const float* __restrict__ slot_pn_x,
    const float* __restrict__ slot_pn_y,
    const float* __restrict__ slot_pn_z,
    const float* __restrict__ slot_eta_r,
    const float* __restrict__ slot_mu_r,
    const float* __restrict__ slot_sigma,
    const float* __restrict__ slot_gain)
{
    int base = slot * n_paths + pI;
    return {
        make_f3(slot_pp_x[base], slot_pp_y[base], slot_pp_z[base]),
        make_f3(slot_pn_x[base], slot_pn_y[base], slot_pn_z[base]),
        slot_eta_r[base],
        slot_mu_r[base],
        slot_sigma[base],
        slot_gain[base],
    };
}

__device__ __forceinline__ ReflectionSlotData load_f_weight_slot(
    int slot,
    int pI,
    int tid,
    int override_slot,
    int n_paths,
    int n_pairs,
    const float* __restrict__ slot_pp_x,
    const float* __restrict__ slot_pp_y,
    const float* __restrict__ slot_pp_z,
    const float* __restrict__ slot_pn_x,
    const float* __restrict__ slot_pn_y,
    const float* __restrict__ slot_pn_z,
    const float* __restrict__ slot_eta_r,
    const float* __restrict__ slot_mu_r,
    const float* __restrict__ slot_sigma,
    const float* __restrict__ slot_gain,
    const float* __restrict__ adjacent_pp_x,
    const float* __restrict__ adjacent_pp_y,
    const float* __restrict__ adjacent_pp_z,
    const float* __restrict__ adjacent_pn_x,
    const float* __restrict__ adjacent_pn_y,
    const float* __restrict__ adjacent_pn_z,
    const float* __restrict__ adjacent_eta_r,
    const float* __restrict__ adjacent_mu_r,
    const float* __restrict__ adjacent_sigma,
    const float* __restrict__ adjacent_gain)
{
    if (slot == override_slot) {
        int base = slot * n_pairs + tid;
        return {
            make_f3(adjacent_pp_x[base], adjacent_pp_y[base], adjacent_pp_z[base]),
            make_f3(adjacent_pn_x[base], adjacent_pn_y[base], adjacent_pn_z[base]),
            adjacent_eta_r[base],
            adjacent_mu_r[base],
            adjacent_sigma[base],
            adjacent_gain[base],
        };
    }
    return load_primary_slot(
        slot, pI, n_paths,
        slot_pp_x, slot_pp_y, slot_pp_z,
        slot_pn_x, slot_pn_y, slot_pn_z,
        slot_eta_r, slot_mu_r, slot_sigma, slot_gain);
}

__device__ __forceinline__ float3a reflection_image_source_from_tx(
    float3a tx_pos,
    int pI,
    int tid,
    int override_slot,
    int n_paths,
    int n_pairs,
    int chain_depth,
    const float* __restrict__ slot_pp_x,
    const float* __restrict__ slot_pp_y,
    const float* __restrict__ slot_pp_z,
    const float* __restrict__ slot_pn_x,
    const float* __restrict__ slot_pn_y,
    const float* __restrict__ slot_pn_z,
    const float* __restrict__ slot_eta_r,
    const float* __restrict__ slot_mu_r,
    const float* __restrict__ slot_sigma,
    const float* __restrict__ slot_gain,
    const float* __restrict__ adjacent_pp_x,
    const float* __restrict__ adjacent_pp_y,
    const float* __restrict__ adjacent_pp_z,
    const float* __restrict__ adjacent_pn_x,
    const float* __restrict__ adjacent_pn_y,
    const float* __restrict__ adjacent_pn_z,
    const float* __restrict__ adjacent_eta_r,
    const float* __restrict__ adjacent_mu_r,
    const float* __restrict__ adjacent_sigma,
    const float* __restrict__ adjacent_gain)
{
    float3a source = tx_pos;
    for (int slot = 0; slot < chain_depth; ++slot) {
        ReflectionSlotData s = load_f_weight_slot(
            slot, pI, tid, override_slot, n_paths, n_pairs,
            slot_pp_x, slot_pp_y, slot_pp_z,
            slot_pn_x, slot_pn_y, slot_pn_z,
            slot_eta_r, slot_mu_r, slot_sigma, slot_gain,
            adjacent_pp_x, adjacent_pp_y, adjacent_pp_z,
            adjacent_pn_x, adjacent_pn_y, adjacent_pn_z,
            adjacent_eta_r, adjacent_mu_r, adjacent_sigma, adjacent_gain);
        source = reflect_point_across_plane(source, s.plane_pt, s.plane_n);
    }
    return source;
}

__device__ __forceinline__ ReflectionChainEval evaluate_reflection_chain(
    float3a image_source,
    float3a rx,
    int pI,
    int tid,
    int override_slot,
    int n_paths,
    int n_pairs,
    int chain_depth,
    float tx_pol_x,
    float tx_pol_y,
    float tx_pol_z,
    float omega,
    const float* __restrict__ slot_pp_x,
    const float* __restrict__ slot_pp_y,
    const float* __restrict__ slot_pp_z,
    const float* __restrict__ slot_pn_x,
    const float* __restrict__ slot_pn_y,
    const float* __restrict__ slot_pn_z,
    const float* __restrict__ slot_eta_r,
    const float* __restrict__ slot_mu_r,
    const float* __restrict__ slot_sigma,
    const float* __restrict__ slot_gain,
    const float* __restrict__ adjacent_pp_x,
    const float* __restrict__ adjacent_pp_y,
    const float* __restrict__ adjacent_pp_z,
    const float* __restrict__ adjacent_pn_x,
    const float* __restrict__ adjacent_pn_y,
    const float* __restrict__ adjacent_pn_z,
    const float* __restrict__ adjacent_eta_r,
    const float* __restrict__ adjacent_mu_r,
    const float* __restrict__ adjacent_sigma,
    const float* __restrict__ adjacent_gain)
{
    ReflectionChainEval out;
    out.geom_valid = true;
    out.image_source = image_source;
    out.tx_pos = image_source;
    out.chain_vec = c3_zero();

    if (chain_depth <= 0) {
        return out;
    }

    float3a current_source = image_source;
    float3a current_target = rx;

    for (int slot = chain_depth - 1; slot >= 0; --slot) {
        ReflectionSlotData s = load_f_weight_slot(
            slot, pI, tid, override_slot, n_paths, n_pairs,
            slot_pp_x, slot_pp_y, slot_pp_z,
            slot_pn_x, slot_pn_y, slot_pn_z,
            slot_eta_r, slot_mu_r, slot_sigma, slot_gain,
            adjacent_pp_x, adjacent_pp_y, adjacent_pp_z,
            adjacent_pn_x, adjacent_pn_y, adjacent_pn_z,
            adjacent_eta_r, adjacent_mu_r, adjacent_sigma, adjacent_gain);

        float3a seg = f3_sub(current_target, current_source);
        float denom = f3_dot(seg, s.plane_n);
        out.geom_valid = out.geom_valid && fabsf(denom) > UTD_EPS;
        float safe_denom = (fabsf(denom) < UTD_EPS) ? (denom >= 0.f ? UTD_EPS : -UTD_EPS) : denom;
        float t = f3_dot(f3_sub(s.plane_pt, current_source), s.plane_n) / safe_denom;
        out.geom_valid = out.geom_valid && (t > UTD_EPS) && (t < (1.f - UTD_EPS));

        float3a hit_p = f3_add(current_source, f3_mul(seg, t));
        out.hit_points[slot] = hit_p;
        out.normals[slot] = s.plane_n;

        current_target = hit_p;
        current_source = reflect_point_across_plane(current_source, s.plane_pt, s.plane_n);
    }

    out.tx_pos = current_source;

    float3a first_dir = safe_normalize(f3_sub(out.hit_points[0], out.tx_pos), make_f3(0, 0, 1));
    float3a pol_dir = project_polarization_to_ray(make_f3(tx_pol_x, tx_pol_y, tx_pol_z), first_dir);
    Complex3 chain_vec = {cplx(pol_dir.x, 0), cplx(pol_dir.y, 0), cplx(pol_dir.z, 0)};

    float3a prev_pt = out.tx_pos;
    for (int slot = 0; slot < chain_depth; ++slot) {
        ReflectionSlotData s = load_f_weight_slot(
            slot, pI, tid, override_slot, n_paths, n_pairs,
            slot_pp_x, slot_pp_y, slot_pp_z,
            slot_pn_x, slot_pn_y, slot_pn_z,
            slot_eta_r, slot_mu_r, slot_sigma, slot_gain,
            adjacent_pp_x, adjacent_pp_y, adjacent_pp_z,
            adjacent_pn_x, adjacent_pn_y, adjacent_pn_z,
            adjacent_eta_r, adjacent_mu_r, adjacent_sigma, adjacent_gain);
        float3a incoming = safe_normalize(f3_sub(out.hit_points[slot], prev_pt), make_f3(0, 0, 1));
        float3a normal = safe_normalize(out.normals[slot], make_f3(0, 1, 0));
        chain_vec = reflect_field_vector_material(
            chain_vec, incoming, normal, s.eta_r, s.mu_r, s.sigma, omega, s.gain);
        prev_pt = out.hit_points[slot];
    }

    out.chain_vec = chain_vec;
    return out;
}

__device__ __forceinline__ Complex point_source_field(
    float3a image_source,
    float3a rx,
    float k)
{
    float3a delta = f3_sub(rx, image_source);
    float dist = safe_length(delta) + UTD_EPS;
    Complex phase = cplx_exp_phase(-k * dist);
    float wavelength = UTD_TWO_PI / k;
    float fspl = wavelength / (4.f * UTD_PI * dist);
    return cplx_mul_real(phase, fspl);
}

__device__ __forceinline__ Complex safe_reflection_transition_weight(
    float edge_distance,
    float3a hit_p,
    float3a previous_point,
    float3a next_point,
    float k)
{
    float s_prev = safe_length(f3_sub(hit_p, previous_point)) + UTD_EPS;
    float s_next = safe_length(f3_sub(next_point, hit_p)) + UTD_EPS;
    float effective_distance = (s_prev * s_next) / (s_prev + s_next + UTD_EPS);
    float x = k * edge_distance * edge_distance / (effective_distance + UTD_EPS);
    float safe_x = fmaxf(x, UTD_SMALL_EPS);
    Complex transition = f_utd_value(safe_x);
    float ramp = fminf(1.f, fmaxf(x, 0.f) / UTD_SMALL_EPS);
    return cplx_mul_real(transition, ramp);
}

__device__ __forceinline__ void scatter_reflection_result(
    Complex3 result,
    int rI,
    float* __restrict__ out_vxr, float* __restrict__ out_vxi,
    float* __restrict__ out_vyr, float* __restrict__ out_vyi,
    float* __restrict__ out_vzr, float* __restrict__ out_vzi)
{
    atomicAdd(&out_vxr[rI], result.x.re);
    atomicAdd(&out_vxi[rI], result.x.im);
    atomicAdd(&out_vyr[rI], result.y.re);
    atomicAdd(&out_vyi[rI], result.y.im);
    atomicAdd(&out_vzr[rI], result.z.re);
    atomicAdd(&out_vzi[rI], result.z.im);
}

// =========================================================================
// REFLECTION ACCUMULATION FORWARD MEGA-KERNEL
// =========================================================================
__global__ void reflection_accumulate_forward_kernel(
    const int* __restrict__ path_idx,
    const int* __restrict__ rx_idx,
    const int* __restrict__ valid_mask,

    const float* __restrict__ img_src_x,
    const float* __restrict__ img_src_y,
    const float* __restrict__ img_src_z,

    const float* __restrict__ slot_pp_x,
    const float* __restrict__ slot_pp_y,
    const float* __restrict__ slot_pp_z,
    const float* __restrict__ slot_pn_x,
    const float* __restrict__ slot_pn_y,
    const float* __restrict__ slot_pn_z,

    const float* __restrict__ slot_eta_r,
    const float* __restrict__ slot_mu_r,
    const float* __restrict__ slot_sigma,
    const float* __restrict__ slot_gain,

    const float* __restrict__ rx_x,
    const float* __restrict__ rx_y,
    const float* __restrict__ rx_z,

    float tx_pol_x, float tx_pol_y, float tx_pol_z,

    float* __restrict__ out_vxr, float* __restrict__ out_vxi,
    float* __restrict__ out_vyr, float* __restrict__ out_vyi,
    float* __restrict__ out_vzr, float* __restrict__ out_vzi,

    int n_pairs,
    int n_paths,
    int chain_depth,
    float k,
    float omega)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_pairs) return;
    if (valid_mask[tid] == 0) return;

    int pI = path_idx[tid];
    int rI = rx_idx[tid];

    // Load image source and receiver position
    float3a img_src = make_f3(img_src_x[pI], img_src_y[pI], img_src_z[pI]);
    float3a rx = make_f3(rx_x[rI], rx_y[rI], rx_z[rI]);

    // Stack arrays for hit points and normals
    float3a hit_points[REFL_MAX_CHAIN_DEPTH];
    float3a normals[REFL_MAX_CHAIN_DEPTH];

    // --- Phase 1: Exact path calculation (backward walk) ---
    float3a current_source = img_src;
    float3a current_target = rx;

    for (int slot = chain_depth - 1; slot >= 0; --slot) {
        int base = slot * n_paths + pI;

        float3a plane_pt = make_f3(slot_pp_x[base], slot_pp_y[base], slot_pp_z[base]);
        float3a plane_n  = make_f3(slot_pn_x[base], slot_pn_y[base], slot_pn_z[base]);

        // Plane-segment intersection: find where segment hits the plane
        float3a seg = f3_sub(current_target, current_source);
        float denom = f3_dot(seg, plane_n);
        float safe_denom = (fabsf(denom) < UTD_EPS) ? (denom >= 0.f ? UTD_EPS : -UTD_EPS) : denom;
        float t = f3_dot(f3_sub(plane_pt, current_source), plane_n) / safe_denom;
        t = fminf(fmaxf(t, 0.f), 1.f);  // clamp to segment

        float3a hit_p = f3_add(current_source, f3_mul(seg, t));

        hit_points[slot] = hit_p;
        normals[slot] = plane_n;

        current_target = hit_p;
        current_source = reflect_point_across_plane(current_source, plane_pt, plane_n);
    }

    float3a tx_pos = current_source;

    // --- Phase 2: Forward Jones chain ---
    // Initialize field vector from TX polarization projected onto first ray
    float3a first_dir = f3_sub(hit_points[0], tx_pos);
    first_dir = safe_normalize(first_dir, make_f3(0, 0, 1));
    float3a pol_dir = project_polarization_to_ray(make_f3(tx_pol_x, tx_pol_y, tx_pol_z), first_dir);
    Complex3 chain_vec = {cplx(pol_dir.x, 0), cplx(pol_dir.y, 0), cplx(pol_dir.z, 0)};

    float3a prev_pt = tx_pos;
    for (int slot = 0; slot < chain_depth; ++slot) {
        float3a incoming = safe_normalize(f3_sub(hit_points[slot], prev_pt), make_f3(0, 0, 1));
        float3a normal = safe_normalize(normals[slot], make_f3(0, 1, 0));

        int base = slot * n_paths + pI;
        float eta_r = slot_eta_r[base];
        float mu_r = slot_mu_r[base];
        float sigma = slot_sigma[base];
        float gain  = slot_gain[base];

        chain_vec = reflect_field_vector_material(
            chain_vec, incoming, normal, eta_r, mu_r, sigma, omega, gain);

        prev_pt = hit_points[slot];
    }

    // --- Phase 3: Point-source field ---
    // Distance from image source to receiver (through the chain)
    float3a delta = f3_sub(rx, img_src);
    float dist = safe_length(delta) + UTD_EPS;
    Complex phase = cplx_exp_phase(-k * dist);
    // Free-space path loss: wavelength / (4*pi*dist)
    float wavelength = UTD_TWO_PI / k;
    float fspl = wavelength / (4.f * UTD_PI * dist);
    Complex unit_field = cplx_mul_real(phase, fspl);

    // Apply to chain vector
    Complex3 result = c3_scale(chain_vec, unit_field);

    // --- Phase 4: Atomic scatter to receiver ---
    atomicAdd(&out_vxr[rI], result.x.re);
    atomicAdd(&out_vxi[rI], result.x.im);
    atomicAdd(&out_vyr[rI], result.y.re);
    atomicAdd(&out_vyi[rI], result.y.im);
    atomicAdd(&out_vzr[rI], result.z.re);
    atomicAdd(&out_vzi[rI], result.z.im);
}

__global__ void reflection_accumulate_f_weight_forward_kernel(
    const int* __restrict__ path_idx,
    const int* __restrict__ rx_idx,
    const int* __restrict__ valid_mask,

    const float* __restrict__ img_src_x,
    const float* __restrict__ img_src_y,
    const float* __restrict__ img_src_z,

    const float* __restrict__ slot_pp_x,
    const float* __restrict__ slot_pp_y,
    const float* __restrict__ slot_pp_z,
    const float* __restrict__ slot_pn_x,
    const float* __restrict__ slot_pn_y,
    const float* __restrict__ slot_pn_z,
    const float* __restrict__ slot_eta_r,
    const float* __restrict__ slot_mu_r,
    const float* __restrict__ slot_sigma,
    const float* __restrict__ slot_gain,

    const int* __restrict__ transition_support_valid,
    const int* __restrict__ transition_primary_side,
    const float* __restrict__ transition_edge_distance,
    const int* __restrict__ adjacent_valid,
    const float* __restrict__ adjacent_pp_x,
    const float* __restrict__ adjacent_pp_y,
    const float* __restrict__ adjacent_pp_z,
    const float* __restrict__ adjacent_pn_x,
    const float* __restrict__ adjacent_pn_y,
    const float* __restrict__ adjacent_pn_z,
    const float* __restrict__ adjacent_eta_r,
    const float* __restrict__ adjacent_mu_r,
    const float* __restrict__ adjacent_sigma,
    const float* __restrict__ adjacent_gain,

    const float* __restrict__ rx_x,
    const float* __restrict__ rx_y,
    const float* __restrict__ rx_z,

    float tx_pos_x, float tx_pos_y, float tx_pos_z,
    float tx_pol_x, float tx_pol_y, float tx_pol_z,

    float* __restrict__ out_vxr, float* __restrict__ out_vxi,
    float* __restrict__ out_vyr, float* __restrict__ out_vyi,
    float* __restrict__ out_vzr, float* __restrict__ out_vzi,

    int n_pairs,
    int n_paths,
    int chain_depth,
    float k,
    float omega)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_pairs) return;
    if (valid_mask[tid] == 0) return;

    int pI = path_idx[tid];
    int rI = rx_idx[tid];
    float3a rx = make_f3(rx_x[rI], rx_y[rI], rx_z[rI]);
    float3a img_src = make_f3(img_src_x[pI], img_src_y[pI], img_src_z[pI]);

    ReflectionChainEval primary = evaluate_reflection_chain(
        img_src,
        rx,
        pI,
        tid,
        -1,
        n_paths,
        n_pairs,
        chain_depth,
        tx_pol_x,
        tx_pol_y,
        tx_pol_z,
        omega,
        slot_pp_x,
        slot_pp_y,
        slot_pp_z,
        slot_pn_x,
        slot_pn_y,
        slot_pn_z,
        slot_eta_r,
        slot_mu_r,
        slot_sigma,
        slot_gain,
        adjacent_pp_x,
        adjacent_pp_y,
        adjacent_pp_z,
        adjacent_pn_x,
        adjacent_pn_y,
        adjacent_pn_z,
        adjacent_eta_r,
        adjacent_mu_r,
        adjacent_sigma,
        adjacent_gain);

    if (!primary.geom_valid) return;

    bool any_support = false;
    bool all_primary_side = true;
    for (int slot = 0; slot < chain_depth; ++slot) {
        int base = slot * n_pairs + tid;
        any_support = any_support || (transition_support_valid[base] != 0);
        all_primary_side = all_primary_side && (transition_primary_side[base] != 0);
    }

    if (!any_support) {
        if (!all_primary_side) return;
        Complex unit_field = point_source_field(primary.image_source, rx, k);
        scatter_reflection_result(
            c3_scale(primary.chain_vec, unit_field),
            rI,
            out_vxr, out_vxi, out_vyr, out_vyi, out_vzr, out_vzi);
        return;
    }

    Complex primary_weights[REFL_MAX_CHAIN_DEPTH];
    Complex adjacent_weights[REFL_MAX_CHAIN_DEPTH];
    Complex chain_weight = cplx(1.f, 0.f);

    for (int slot = 0; slot < chain_depth; ++slot) {
        int base = slot * n_pairs + tid;
        bool support = transition_support_valid[base] != 0;
        bool primary_side = transition_primary_side[base] != 0;
        float3a prev_point = (slot == 0) ? primary.tx_pos : primary.hit_points[slot - 1];
        float3a next_point = (slot + 1 < chain_depth) ? primary.hit_points[slot + 1] : rx;
        Complex transition = support
            ? safe_reflection_transition_weight(
                transition_edge_distance[base],
                primary.hit_points[slot],
                prev_point,
                next_point,
                k)
            : cplx(1.f, 0.f);

        primary_weights[slot] = support
            ? (primary_side ? transition : cplx_zero())
            : (primary_side ? cplx(1.f, 0.f) : cplx_zero());
        adjacent_weights[slot] = (support && !primary_side && adjacent_valid[base] != 0)
            ? transition
            : cplx_zero();
        chain_weight = cplx_mul(chain_weight, primary_weights[slot]);
    }

    Complex3 total = c3_scale(
        c3_scale(primary.chain_vec, chain_weight),
        point_source_field(primary.image_source, rx, k));

    for (int slot = 0; slot < chain_depth; ++slot) {
        if (!cplx_any_nonzero(adjacent_weights[slot])) continue;
        Complex residual_weight = adjacent_weights[slot];
        for (int other = 0; other < chain_depth; ++other) {
            if (other == slot) continue;
            residual_weight = cplx_mul(residual_weight, primary_weights[other]);
        }
        if (!cplx_any_nonzero(residual_weight)) continue;

        float3a branch_image_source = reflection_image_source_from_tx(
            make_f3(tx_pos_x, tx_pos_y, tx_pos_z),
            pI,
            tid,
            slot,
            n_paths,
            n_pairs,
            chain_depth,
            slot_pp_x,
            slot_pp_y,
            slot_pp_z,
            slot_pn_x,
            slot_pn_y,
            slot_pn_z,
            slot_eta_r,
            slot_mu_r,
            slot_sigma,
            slot_gain,
            adjacent_pp_x,
            adjacent_pp_y,
            adjacent_pp_z,
            adjacent_pn_x,
            adjacent_pn_y,
            adjacent_pn_z,
            adjacent_eta_r,
            adjacent_mu_r,
            adjacent_sigma,
            adjacent_gain);

        ReflectionChainEval branch = evaluate_reflection_chain(
            branch_image_source,
            rx,
            pI,
            tid,
            slot,
            n_paths,
            n_pairs,
            chain_depth,
            tx_pol_x,
            tx_pol_y,
            tx_pol_z,
            omega,
            slot_pp_x,
            slot_pp_y,
            slot_pp_z,
            slot_pn_x,
            slot_pn_y,
            slot_pn_z,
            slot_eta_r,
            slot_mu_r,
            slot_sigma,
            slot_gain,
            adjacent_pp_x,
            adjacent_pp_y,
            adjacent_pp_z,
            adjacent_pn_x,
            adjacent_pn_y,
            adjacent_pn_z,
            adjacent_eta_r,
            adjacent_mu_r,
            adjacent_sigma,
            adjacent_gain);

        if (!branch.geom_valid) continue;

        Complex3 weighted_branch = c3_scale(branch.chain_vec, residual_weight);
        Complex unit_field = point_source_field(primary.image_source, rx, k);
        total = c3_add(total, c3_scale(weighted_branch, unit_field));
    }

    scatter_reflection_result(
        total,
        rI,
        out_vxr, out_vxi,
        out_vyr, out_vyi,
        out_vzr, out_vzi);
}

__global__ void reflection_epc_targets_forward_kernel(
    const int* __restrict__ path_idx,

    const float* __restrict__ img_src_x,
    const float* __restrict__ img_src_y,
    const float* __restrict__ img_src_z,

    const float* __restrict__ slot_pp_x,
    const float* __restrict__ slot_pp_y,
    const float* __restrict__ slot_pp_z,
    const float* __restrict__ slot_pn_x,
    const float* __restrict__ slot_pn_y,
    const float* __restrict__ slot_pn_z,

    const float* __restrict__ slot_eta_r,
    const float* __restrict__ slot_mu_r,
    const float* __restrict__ slot_sigma,
    const float* __restrict__ slot_gain,

    const float* __restrict__ target_x,
    const float* __restrict__ target_y,
    const float* __restrict__ target_z,

    float tx_pol_x, float tx_pol_y, float tx_pol_z,

    int* __restrict__ out_geom_valid,
    float* __restrict__ out_tx_pos_x,
    float* __restrict__ out_tx_pos_y,
    float* __restrict__ out_tx_pos_z,
    float* __restrict__ out_vxr,
    float* __restrict__ out_vxi,
    float* __restrict__ out_vyr,
    float* __restrict__ out_vyi,
    float* __restrict__ out_vzr,
    float* __restrict__ out_vzi,
    float* __restrict__ out_hit_x,
    float* __restrict__ out_hit_y,
    float* __restrict__ out_hit_z,

    int n_pairs,
    int n_paths,
    int chain_depth,
    float k,
    float omega)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_pairs) return;

    int pI = path_idx[tid];
    float3a current_source = make_f3(img_src_x[pI], img_src_y[pI], img_src_z[pI]);
    float3a current_target = make_f3(target_x[tid], target_y[tid], target_z[tid]);

    float3a hit_points[REFL_MAX_CHAIN_DEPTH];
    float3a normals[REFL_MAX_CHAIN_DEPTH];
    bool geom_valid = true;

    for (int slot = chain_depth - 1; slot >= 0; --slot) {
        int base = slot * n_paths + pI;
        float3a plane_pt = make_f3(slot_pp_x[base], slot_pp_y[base], slot_pp_z[base]);
        float3a plane_n = make_f3(slot_pn_x[base], slot_pn_y[base], slot_pn_z[base]);
        float3a seg = f3_sub(current_target, current_source);
        float denom = f3_dot(seg, plane_n);
        geom_valid = geom_valid && fabsf(denom) > UTD_EPS;
        float safe_denom = (fabsf(denom) < UTD_EPS) ? (denom >= 0.f ? UTD_EPS : -UTD_EPS) : denom;
        float t = f3_dot(f3_sub(plane_pt, current_source), plane_n) / safe_denom;
        geom_valid = geom_valid && (t > UTD_EPS) && (t < (1.f - UTD_EPS));

        float3a hit_p = f3_add(current_source, f3_mul(seg, t));
        hit_points[slot] = hit_p;
        normals[slot] = plane_n;

        int out_base = slot * n_pairs + tid;
        out_hit_x[out_base] = hit_p.x;
        out_hit_y[out_base] = hit_p.y;
        out_hit_z[out_base] = hit_p.z;

        current_target = hit_p;
        current_source = reflect_point_across_plane(current_source, plane_pt, plane_n);
    }

    out_geom_valid[tid] = geom_valid ? 1 : 0;
    out_tx_pos_x[tid] = current_source.x;
    out_tx_pos_y[tid] = current_source.y;
    out_tx_pos_z[tid] = current_source.z;

    if (chain_depth <= 0) {
        out_vxr[tid] = 0.f;
        out_vxi[tid] = 0.f;
        out_vyr[tid] = 0.f;
        out_vyi[tid] = 0.f;
        out_vzr[tid] = 0.f;
        out_vzi[tid] = 0.f;
        return;
    }

    float3a tx_pos = current_source;
    float3a first_dir = safe_normalize(f3_sub(hit_points[0], tx_pos), make_f3(0, 0, 1));
    float3a pol_dir = project_polarization_to_ray(make_f3(tx_pol_x, tx_pol_y, tx_pol_z), first_dir);
    Complex3 chain_vec = { cplx(pol_dir.x, 0.f), cplx(pol_dir.y, 0.f), cplx(pol_dir.z, 0.f) };

    float3a prev_pt = tx_pos;
    for (int slot = 0; slot < chain_depth; ++slot) {
        float3a incoming = safe_normalize(f3_sub(hit_points[slot], prev_pt), make_f3(0, 0, 1));
        float3a normal = safe_normalize(normals[slot], make_f3(0, 1, 0));
        int base = slot * n_paths + pI;
        chain_vec = reflect_field_vector_material(
            chain_vec,
            incoming,
            normal,
            slot_eta_r[base],
            slot_mu_r[base],
            slot_sigma[base],
            omega,
            slot_gain[base]
        );
        prev_pt = hit_points[slot];
    }

    out_vxr[tid] = chain_vec.x.re;
    out_vxi[tid] = chain_vec.x.im;
    out_vyr[tid] = chain_vec.y.re;
    out_vyi[tid] = chain_vec.y.im;
    out_vzr[tid] = chain_vec.z.re;
    out_vzi[tid] = chain_vec.z.im;
}

} // anonymous namespace

// =========================================================================
// Host launcher
// =========================================================================
void reflection_accumulate_forward(
    const int*   path_idx,
    const int*   rx_idx,
    const int*   valid_mask,
    const float* image_source_x, const float* image_source_y, const float* image_source_z,
    const float* slot_plane_point_x, const float* slot_plane_point_y, const float* slot_plane_point_z,
    const float* slot_plane_normal_x, const float* slot_plane_normal_y, const float* slot_plane_normal_z,
    const float* slot_eta_r, const float* slot_mu_r, const float* slot_sigma, const float* slot_gain,
    const float* rx_x, const float* rx_y, const float* rx_z,
    float tx_pol_x, float tx_pol_y, float tx_pol_z,
    float* out_vec_x_re, float* out_vec_x_im,
    float* out_vec_y_re, float* out_vec_y_im,
    float* out_vec_z_re, float* out_vec_z_im,
    int n_pairs, int n_paths, int chain_depth, float k, float omega)
{
    if (n_pairs <= 0) return;
    if (chain_depth > REFL_MAX_CHAIN_DEPTH)
        throw std::runtime_error("chain_depth exceeds REFL_MAX_CHAIN_DEPTH");

    constexpr int BLOCK = 256;
    int grid = (n_pairs + BLOCK - 1) / BLOCK;

    reflection_accumulate_forward_kernel<<<grid, BLOCK>>>(
        path_idx, rx_idx, valid_mask,
        image_source_x, image_source_y, image_source_z,
        slot_plane_point_x, slot_plane_point_y, slot_plane_point_z,
        slot_plane_normal_x, slot_plane_normal_y, slot_plane_normal_z,
        slot_eta_r, slot_mu_r, slot_sigma, slot_gain,
        rx_x, rx_y, rx_z,
        tx_pol_x, tx_pol_y, tx_pol_z,
        out_vec_x_re, out_vec_x_im,
        out_vec_y_re, out_vec_y_im,
        out_vec_z_re, out_vec_z_im,
        n_pairs, n_paths, chain_depth, k, omega);

    throw_cuda(cudaGetLastError(), "reflection_accumulate_forward_kernel launch");
}

void reflection_accumulate_f_weight_forward(
    const int*   path_idx,
    const int*   rx_idx,
    const int*   valid_mask,
    const float* image_source_x, const float* image_source_y, const float* image_source_z,
    const float* slot_plane_point_x, const float* slot_plane_point_y, const float* slot_plane_point_z,
    const float* slot_plane_normal_x, const float* slot_plane_normal_y, const float* slot_plane_normal_z,
    const float* slot_eta_r, const float* slot_mu_r, const float* slot_sigma, const float* slot_gain,
    const int*   transition_support_valid,
    const int*   transition_primary_side,
    const float* transition_edge_distance,
    const int*   adjacent_valid,
    const float* adjacent_plane_point_x, const float* adjacent_plane_point_y, const float* adjacent_plane_point_z,
    const float* adjacent_plane_normal_x, const float* adjacent_plane_normal_y, const float* adjacent_plane_normal_z,
    const float* adjacent_eta_r, const float* adjacent_mu_r, const float* adjacent_sigma, const float* adjacent_gain,
    const float* rx_x, const float* rx_y, const float* rx_z,
    float tx_pos_x, float tx_pos_y, float tx_pos_z,
    float tx_pol_x, float tx_pol_y, float tx_pol_z,
    float* out_vec_x_re, float* out_vec_x_im,
    float* out_vec_y_re, float* out_vec_y_im,
    float* out_vec_z_re, float* out_vec_z_im,
    int n_pairs, int n_paths, int chain_depth, float k, float omega)
{
    if (n_pairs <= 0) return;
    if (chain_depth > REFL_MAX_CHAIN_DEPTH)
        throw std::runtime_error("chain_depth exceeds REFL_MAX_CHAIN_DEPTH");

    constexpr int BLOCK = 256;
    int grid = (n_pairs + BLOCK - 1) / BLOCK;

    reflection_accumulate_f_weight_forward_kernel<<<grid, BLOCK>>>(
        path_idx, rx_idx, valid_mask,
        image_source_x, image_source_y, image_source_z,
        slot_plane_point_x, slot_plane_point_y, slot_plane_point_z,
        slot_plane_normal_x, slot_plane_normal_y, slot_plane_normal_z,
        slot_eta_r, slot_mu_r, slot_sigma, slot_gain,
        transition_support_valid,
        transition_primary_side,
        transition_edge_distance,
        adjacent_valid,
        adjacent_plane_point_x, adjacent_plane_point_y, adjacent_plane_point_z,
        adjacent_plane_normal_x, adjacent_plane_normal_y, adjacent_plane_normal_z,
        adjacent_eta_r, adjacent_mu_r, adjacent_sigma, adjacent_gain,
        rx_x, rx_y, rx_z,
        tx_pos_x, tx_pos_y, tx_pos_z,
        tx_pol_x, tx_pol_y, tx_pol_z,
        out_vec_x_re, out_vec_x_im,
        out_vec_y_re, out_vec_y_im,
        out_vec_z_re, out_vec_z_im,
        n_pairs, n_paths, chain_depth, k, omega);

    throw_cuda(cudaGetLastError(), "reflection_accumulate_f_weight_forward_kernel launch");
}

void reflection_epc_targets_forward(
    const int* path_idx,
    const float* image_source_x, const float* image_source_y, const float* image_source_z,
    const float* slot_plane_point_x, const float* slot_plane_point_y, const float* slot_plane_point_z,
    const float* slot_plane_normal_x, const float* slot_plane_normal_y, const float* slot_plane_normal_z,
    const float* slot_eta_r, const float* slot_mu_r, const float* slot_sigma, const float* slot_gain,
    const float* target_x, const float* target_y, const float* target_z,
    float tx_pol_x, float tx_pol_y, float tx_pol_z,
    int* out_geom_valid,
    float* out_tx_pos_x, float* out_tx_pos_y, float* out_tx_pos_z,
    float* out_vec_x_re, float* out_vec_x_im,
    float* out_vec_y_re, float* out_vec_y_im,
    float* out_vec_z_re, float* out_vec_z_im,
    float* out_hit_x,
    float* out_hit_y,
    float* out_hit_z,
    int n_pairs, int n_paths, int chain_depth, float k, float omega)
{
    if (n_pairs <= 0) return;
    if (chain_depth > REFL_MAX_CHAIN_DEPTH)
        throw std::runtime_error("chain_depth exceeds REFL_MAX_CHAIN_DEPTH");

    constexpr int BLOCK = 256;
    int grid = (n_pairs + BLOCK - 1) / BLOCK;

    reflection_epc_targets_forward_kernel<<<grid, BLOCK>>>(
        path_idx,
        image_source_x, image_source_y, image_source_z,
        slot_plane_point_x, slot_plane_point_y, slot_plane_point_z,
        slot_plane_normal_x, slot_plane_normal_y, slot_plane_normal_z,
        slot_eta_r, slot_mu_r, slot_sigma, slot_gain,
        target_x, target_y, target_z,
        tx_pol_x, tx_pol_y, tx_pol_z,
        out_geom_valid,
        out_tx_pos_x, out_tx_pos_y, out_tx_pos_z,
        out_vec_x_re, out_vec_x_im,
        out_vec_y_re, out_vec_y_im,
        out_vec_z_re, out_vec_z_im,
        out_hit_x, out_hit_y, out_hit_z,
        n_pairs, n_paths, chain_depth, k, omega
    );

    throw_cuda(cudaGetLastError(), "reflection_epc_targets_forward_kernel launch");
}

} // namespace witwin::channel::native_ext
