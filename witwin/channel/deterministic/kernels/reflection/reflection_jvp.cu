#include <cuda_runtime.h>

#include <common/cuda_check.h>
#include <reflection/reflection_types.h>
#include <reflection/reflection_common.h>
#include <reflection/reflection_jvp.h>

namespace witwin::channel::native_ext {
namespace {

using namespace reflection_detail;

using common::throw_cuda;

__device__ __forceinline__ float3a f3_zero()
{
    return make_f3(0.f, 0.f, 0.f);
}

__device__ __forceinline__ void safe_normalize_jvp(
    float3a value,
    float3a tangent,
    float3a fallback,
    float3a& out,
    float3a& t_out)
{
    float len = safe_length(value);
    out = safe_normalize(value, fallback);
    if (len <= UTD_SMALL_EPS) {
        t_out = f3_zero();
        return;
    }
    float along = f3_dot(out, tangent);
    t_out = f3_mul(f3_sub(tangent, f3_mul(out, along)), 1.f / len);
}

__device__ __forceinline__ float3a f3_cross_jvp(
    float3a a,
    float3a t_a,
    float3a b,
    float3a t_b)
{
    return f3_add(f3_cross(t_a, b), f3_cross(a, t_b));
}

__device__ __forceinline__ float3a reflect_direction_jvp(
    float3a direction,
    float3a t_direction,
    float3a normal,
    float3a t_normal)
{
    float d_dot_n = f3_dot(direction, normal);
    float t_dot = f3_dot(t_direction, normal) + f3_dot(direction, t_normal);
    return f3_sub(
        t_direction,
        f3_add(f3_mul(t_normal, 2.f * d_dot_n), f3_mul(normal, 2.f * t_dot))
    );
}

__device__ __forceinline__ Complex cplx_div_jvp(
    Complex num,
    Complex t_num,
    Complex den,
    Complex t_den)
{
    Complex den_sq = cplx_mul(den, den);
    if (cplx_abs_sqr(den_sq) <= UTD_EPS) {
        return cplx_zero();
    }
    return cplx_div(cplx_sub(cplx_mul(t_num, den), cplx_mul(num, t_den)), den_sq);
}

__device__ __forceinline__ Complex cplx_sqrt_jvp(Complex value, Complex tangent)
{
    Complex root = cplx_sqrt(value);
    if (cplx_abs_sqr(root) <= UTD_EPS) {
        return cplx_zero();
    }
    return cplx_div(tangent, cplx_mul_real(root, 2.f));
}

__device__ __forceinline__ void sanitize_complex_jvp(Complex& value, Complex& tangent)
{
    if (!isfinite(value.re) || !isfinite(value.im)) {
        value = cplx_zero();
        tangent = cplx_zero();
    }
    if (!isfinite(tangent.re) || !isfinite(tangent.im)) {
        tangent = cplx_zero();
    }
}

__device__ __forceinline__ void fresnel_reflection_face_jvp(
    float cos_theta,
    float t_cos_theta,
    float eta_r,
    float mu_r,
    float sigma,
    float omega,
    float gain,
    Complex& r_te,
    Complex& t_r_te,
    Complex& r_tm,
    Complex& t_r_tm)
{
    float ct = fminf(fmaxf(cos_theta, UTD_SMALL_EPS), 1.f);
    float t_ct = (cos_theta > UTD_SMALL_EPS && cos_theta < 1.f) ? t_cos_theta : 0.f;
    float sin_sq = 1.f - ct * ct;
    float t_sin_sq = -2.f * ct * t_ct;
    float so = fmaxf(omega, UTD_SMALL_EPS);
    Complex eta = cplx(eta_r, -sigma / (so * UTD_EPSILON_0));
    Complex mu = cplx(mu_r, 0.f);

    Complex tmp = cplx_sub(cplx_mul(mu, eta), cplx(sin_sq, 0.f));
    Complex t_tmp = cplx(-t_sin_sq, 0.f);
    Complex a = cplx_sqrt(tmp);
    Complex t_a = cplx_sqrt_jvp(tmp, t_tmp);

    Complex mu_ct = cplx_mul_real(mu, ct);
    Complex t_mu_ct = cplx_mul_real(mu, t_ct);
    Complex num_te = cplx_sub(mu_ct, a);
    Complex den_te = cplx_add(mu_ct, a);
    Complex t_num_te = cplx_sub(t_mu_ct, t_a);
    Complex t_den_te = cplx_add(t_mu_ct, t_a);
    r_te = cplx_div(num_te, den_te);
    t_r_te = cplx_div_jvp(num_te, t_num_te, den_te, t_den_te);

    Complex eta_ct = cplx_mul_real(eta, ct);
    Complex t_eta_ct = cplx_mul_real(eta, t_ct);
    Complex num_tm = cplx_sub(eta_ct, a);
    Complex den_tm = cplx_add(eta_ct, a);
    Complex t_num_tm = cplx_sub(t_eta_ct, t_a);
    Complex t_den_tm = cplx_add(t_eta_ct, t_a);
    r_tm = cplx_div(num_tm, den_tm);
    t_r_tm = cplx_div_jvp(num_tm, t_num_tm, den_tm, t_den_tm);

    Complex gain_c = cplx(gain, 0.f);
    r_te = cplx_mul(gain_c, r_te);
    r_tm = cplx_mul(gain_c, r_tm);
    t_r_te = cplx_mul(gain_c, t_r_te);
    t_r_tm = cplx_mul(gain_c, t_r_tm);
    sanitize_complex_jvp(r_te, t_r_te);
    sanitize_complex_jvp(r_tm, t_r_tm);
}

__device__ __forceinline__ void project_polarization_to_ray_jvp(
    float3a tx_pol,
    float3a ray_dir,
    float3a t_ray_dir,
    float3a& out,
    float3a& t_out)
{
    float3a ray_hat, t_ray_hat;
    safe_normalize_jvp(ray_dir, t_ray_dir, make_f3(0, 0, 1), ray_hat, t_ray_hat);
    float pol_dot_ray = f3_dot(tx_pol, ray_hat);
    float t_pol_dot_ray = f3_dot(tx_pol, t_ray_hat);
    float3a proj = f3_sub(tx_pol, f3_mul(ray_hat, pol_dot_ray));
    float3a t_proj = f3_sub(
        f3_zero(),
        f3_add(f3_mul(t_ray_hat, pol_dot_ray), f3_mul(ray_hat, t_pol_dot_ray))
    );
    safe_normalize_jvp(proj, t_proj, stable_perp_basis(ray_hat, make_f3(0, 1, 0)), out, t_out);
}

// Reflect field vector with material + simultaneous tangent propagation.
// Returns (primal_result, tangent_result).
__device__ __forceinline__ void reflect_field_jvp(
    Complex3 vec, Complex3 t_vec,
    float3a inc_hat, float3a t_inc_hat,
    float3a normal_hat, float3a t_normal_hat,
    float eta_r, float mu_r, float sigma, float omega, float gain,
    Complex3& out, Complex3& t_out)
{
    float3a ref_dir = reflect_direction(inc_hat, normal_hat);
    float3a t_ref_dir = reflect_direction_jvp(inc_hat, t_inc_hat, normal_hat, t_normal_hat);
    float3a s_pref = f3_cross(normal_hat, inc_hat);
    float3a t_s_pref = f3_cross_jvp(normal_hat, t_normal_hat, inc_hat, t_inc_hat);
    float3a s_hat, t_s_hat;
    safe_normalize_jvp(s_pref, t_s_pref, stable_perp_basis(inc_hat, make_f3(0,1,0)), s_hat, t_s_hat);
    float3a p_in_pref = f3_cross(s_hat, inc_hat);
    float3a t_p_in_pref = f3_cross_jvp(s_hat, t_s_hat, inc_hat, t_inc_hat);
    float3a p_in, t_p_in;
    safe_normalize_jvp(p_in_pref, t_p_in_pref, stable_perp_basis(inc_hat, make_f3(1,0,0)), p_in, t_p_in);
    float3a p_out_pref = f3_cross(s_hat, ref_dir);
    float3a t_p_out_pref = f3_cross_jvp(s_hat, t_s_hat, ref_dir, t_ref_dir);
    float3a p_out, t_p_out;
    safe_normalize_jvp(p_out_pref, t_p_out_pref, stable_perp_basis(ref_dir, make_f3(1,0,0)), p_out, t_p_out);

    float cos_raw = f3_dot(inc_hat, normal_hat);
    float cos_abs = fabsf(cos_raw);
    float t_cos_raw = f3_dot(t_inc_hat, normal_hat) + f3_dot(inc_hat, t_normal_hat);
    float t_cos_abs = (cos_raw >= 0.f) ? t_cos_raw : -t_cos_raw;
    Complex rTE, t_rTE, rTM, t_rTM;
    fresnel_reflection_face_jvp(
        cos_abs, t_cos_abs, eta_r, mu_r, sigma, omega, gain,
        rTE, t_rTE, rTM, t_rTM);

    // Primal
    Complex e_s = cplx_dot_real(vec, s_hat);
    Complex e_p = cplx_dot_real(vec, p_in);
    Complex amp_s = cplx_mul(rTE, e_s);
    Complex amp_p = cplx_mul(rTM, e_p);
    out = c3_add(cplx_scale_real(s_hat, amp_s),
                 cplx_scale_real(p_out, amp_p));

    Complex t_e_s = cplx_add(cplx_dot_real(t_vec, s_hat), cplx_dot_real(vec, t_s_hat));
    Complex t_e_p = cplx_add(cplx_dot_real(t_vec, p_in), cplx_dot_real(vec, t_p_in));
    Complex t_amp_s = cplx_add(cplx_mul(t_rTE, e_s), cplx_mul(rTE, t_e_s));
    Complex t_amp_p = cplx_add(cplx_mul(t_rTM, e_p), cplx_mul(rTM, t_e_p));
    t_out = c3_add(
        c3_add(cplx_scale_real(t_s_hat, amp_s), cplx_scale_real(s_hat, t_amp_s)),
        c3_add(cplx_scale_real(t_p_out, amp_p), cplx_scale_real(p_out, t_amp_p))
    );
}

// =========================================================================
// Reflection JVP kernel
// =========================================================================
__global__ void reflection_jvp_kernel(
    const int* __restrict__ pathIdx,
    const int* __restrict__ rxIdx_arr,
    const int* __restrict__ valid,
    const float* __restrict__ isx, const float* __restrict__ isy, const float* __restrict__ isz,
    const float* __restrict__ ppx, const float* __restrict__ ppy, const float* __restrict__ ppz,
    const float* __restrict__ pnx, const float* __restrict__ pny, const float* __restrict__ pnz,
    const float* __restrict__ s_eta, const float* __restrict__ s_mu, const float* __restrict__ s_sig, const float* __restrict__ s_gn,
    const float* __restrict__ rxx, const float* __restrict__ rxy, const float* __restrict__ rxz,
    float tx_px, float tx_py, float tx_pz,
    const float* __restrict__ t_isx, const float* __restrict__ t_isy, const float* __restrict__ t_isz,
    const float* __restrict__ t_ppx, const float* __restrict__ t_ppy, const float* __restrict__ t_ppz,
    const float* __restrict__ t_pnx, const float* __restrict__ t_pny, const float* __restrict__ t_pnz,
    const float* __restrict__ t_rxx, const float* __restrict__ t_rxy, const float* __restrict__ t_rxz,
    float* __restrict__ to_xr, float* __restrict__ to_xi,
    float* __restrict__ to_yr, float* __restrict__ to_yi,
    float* __restrict__ to_zr, float* __restrict__ to_zi,
    int nPairs, int nPaths, int chainDepth, float k, float omega)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= nPairs) return;
    if (valid[tid] == 0) return;

    int pI = pathIdx[tid];
    int rI = rxIdx_arr[tid];

    float3a img  = make_f3(isx[pI], isy[pI], isz[pI]);
    float3a rx   = make_f3(rxx[rI], rxy[rI], rxz[rI]);
    float3a t_img = make_f3(t_isx[pI], t_isy[pI], t_isz[pI]);
    float3a t_rx  = make_f3(t_rxx[rI], t_rxy[rI], t_rxz[rI]);

// --- Reflection-chain EPC (primal + tangent) ---
    float3a hitPts[REFL_MAX_CHAIN_DEPTH], normals[REFL_MAX_CHAIN_DEPTH];
    float3a t_hitPts[REFL_MAX_CHAIN_DEPTH], t_normals[REFL_MAX_CHAIN_DEPTH];

    float3a curSrc = img, curTgt = rx;
    float3a t_curSrc = t_img, t_curTgt = t_rx;

    for (int slot = chainDepth - 1; slot >= 0; --slot) {
        int base = slot * nPaths + pI;
        float3a plPt = make_f3(ppx[base], ppy[base], ppz[base]);
        float3a plN  = make_f3(pnx[base], pny[base], pnz[base]);
        float3a t_plPt = make_f3(t_ppx[base], t_ppy[base], t_ppz[base]);
        float3a t_plN  = make_f3(t_pnx[base], t_pny[base], t_pnz[base]);

        // Primal intersection
        float3a seg = f3_sub(curTgt, curSrc);
        float denom = f3_dot(seg, plN);
        float sd = (fabsf(denom) < UTD_EPS) ? (denom >= 0 ? UTD_EPS : -UTD_EPS) : denom;
        float t_val = f3_dot(f3_sub(plPt, curSrc), plN) / sd;
        t_val = fminf(fmaxf(t_val, 0.f), 1.f);
        float3a hitP = f3_add(curSrc, f3_mul(seg, t_val));

        // Tangent of intersection (first-order)
        float3a t_seg = f3_sub(t_curTgt, t_curSrc);
        float t_num = f3_dot(f3_sub(t_plPt, t_curSrc), plN) + f3_dot(f3_sub(plPt, curSrc), t_plN);
        float t_denom = f3_dot(t_seg, plN) + f3_dot(seg, t_plN);
        float dt = (t_num - t_val * t_denom) / sd;
        float3a t_hitP = f3_add(t_curSrc, f3_add(f3_mul(t_seg, t_val), f3_mul(seg, dt)));

        hitPts[slot] = hitP;
        normals[slot] = plN;
        t_hitPts[slot] = t_hitP;

        curTgt = hitP;
        t_curTgt = t_hitP;

        float3a oldSrc = curSrc;
        float3a oldTSrc = t_curSrc;
        float d = f3_dot(f3_sub(oldSrc, plPt), plN);
        float t_d = f3_dot(f3_sub(oldTSrc, t_plPt), plN) + f3_dot(f3_sub(oldSrc, plPt), t_plN);
        curSrc = reflect_point_across_plane(oldSrc, plPt, plN);
        t_curSrc = f3_sub(oldTSrc, f3_add(f3_mul(t_plN, 2.f * d), f3_mul(plN, 2.f * t_d)));
    }

    float3a txPos = curSrc;
    float3a t_txPos = t_curSrc;

    // --- Forward Jones chain (primal + tangent) ---
    float3a firstDir, t_firstDir;
    safe_normalize_jvp(
        f3_sub(hitPts[0], txPos),
        f3_sub(t_hitPts[0], t_txPos),
        make_f3(0,0,1),
        firstDir,
        t_firstDir
    );
    float3a polDir, t_polDir;
    project_polarization_to_ray_jvp(make_f3(tx_px, tx_py, tx_pz), firstDir, t_firstDir, polDir, t_polDir);
    Complex3 chain = {cplx(polDir.x,0), cplx(polDir.y,0), cplx(polDir.z,0)};
    Complex3 t_chain = {cplx(t_polDir.x,0), cplx(t_polDir.y,0), cplx(t_polDir.z,0)};

    float3a prev = txPos;
    float3a t_prev = t_txPos;
    for (int slot = 0; slot < chainDepth; ++slot) {
        float3a inc, t_inc;
        safe_normalize_jvp(
            f3_sub(hitPts[slot], prev),
            f3_sub(t_hitPts[slot], t_prev),
            make_f3(0,0,1),
            inc,
            t_inc
        );
        float3a nrm, t_nrm;
        safe_normalize_jvp(normals[slot], t_normals[slot], make_f3(0,1,0), nrm, t_nrm);
        int base = slot * nPaths + pI;

        Complex3 out_p, out_t;
        reflect_field_jvp(chain, t_chain, inc, t_inc, nrm, t_nrm,
                          s_eta[base], s_mu[base], s_sig[base], omega, s_gn[base],
                          out_p, out_t);
        chain = out_p;
        t_chain = out_t;
        prev = hitPts[slot];
        t_prev = t_hitPts[slot];
    }

    // --- Point-source field (primal + tangent) ---
    float3a delta = f3_sub(rx, img);
    float dist = safe_length(delta) + UTD_EPS;
    float wl = UTD_TWO_PI / k;
    float fspl = wl / (4.f * UTD_PI * dist);
    Complex phase = cplx_exp_phase(-k * dist);
    Complex unit_field = cplx_mul_real(phase, fspl);

    // Tangent of unit_field w.r.t. geometry
    float3a t_delta = f3_sub(t_rx, t_img);
    float t_dist = f3_dot(f3_mul(delta, 1.f/dist), t_delta);
    // d(unit_field)/d(dist) = (-jk * fspl - fspl/dist) * phase
    float dfspl = -wl / (4.f * UTD_PI * dist * dist);
    Complex d_unit = cplx_add(
        cplx_mul_real(cplx_mul(cplx(0, -k), phase), fspl),
        cplx_mul_real(phase, dfspl));
    Complex t_unit = cplx_mul_real(d_unit, t_dist);

    // Result tangent: d(chain * unit) = t_chain * unit + chain * t_unit
    Complex3 result_t = c3_add(c3_scale(t_chain, unit_field),
                               c3_scale(chain, t_unit));

    atomicAdd(&to_xr[rI], result_t.x.re);
    atomicAdd(&to_xi[rI], result_t.x.im);
    atomicAdd(&to_yr[rI], result_t.y.re);
    atomicAdd(&to_yi[rI], result_t.y.im);
    atomicAdd(&to_zr[rI], result_t.z.re);
    atomicAdd(&to_zi[rI], result_t.z.im);
}

__device__ __forceinline__ float length_jvp(float3a value, float3a tangent)
{
    float n = safe_length(value);
    if (n <= UTD_SMALL_EPS) return 0.f;
    return f3_dot(f3_div(value, n + UTD_EPS), tangent);
}

__device__ __forceinline__ void safe_transition_value_jvp(
    float x,
    float t_x,
    Complex& value,
    Complex& tangent)
{
    if (x <= 0.f) {
        value = cplx_zero();
        tangent = cplx_zero();
        return;
    }
    if (x < UTD_SMALL_EPS) {
        Complex small_value = f_utd_value(UTD_SMALL_EPS);
        value = cplx_mul_real(small_value, x / UTD_SMALL_EPS);
        tangent = cplx_mul_real(small_value, t_x / UTD_SMALL_EPS);
        return;
    }
    Complex first, second;
    f_utd_with_derivatives(x, value, first, second);
    tangent = cplx_mul_real(first, t_x);
}

__device__ __forceinline__ float3a reflect_point_jvp(
    float3a point,
    float3a t_point,
    float3a plane_point,
    float3a t_plane_point,
    float3a plane_normal,
    float3a t_plane_normal)
{
    float d = f3_dot(f3_sub(point, plane_point), plane_normal);
    float t_d = f3_dot(f3_sub(t_point, t_plane_point), plane_normal)
        + f3_dot(f3_sub(point, plane_point), t_plane_normal);
    return f3_sub(
        t_point,
        f3_add(f3_mul(t_plane_normal, 2.f * d), f3_mul(plane_normal, 2.f * t_d))
    );
}

__device__ __forceinline__ float segment_distance_jvp(
    float3a point,
    float3a t_point,
    float3a edge_v0,
    float3a t_edge_v0,
    float3a edge_v1,
    float3a t_edge_v1)
{
    float3a edge = f3_sub(edge_v1, edge_v0);
    float3a t_edge = f3_sub(t_edge_v1, t_edge_v0);
    float edge_len = safe_length(edge);
    if (edge_len <= UTD_SMALL_EPS) {
        return length_jvp(f3_sub(point, edge_v0), f3_sub(t_point, t_edge_v0));
    }

    float t_edge_len = length_jvp(edge, t_edge);
    float3a edge_dir;
    float3a t_edge_dir;
    safe_normalize_jvp(edge, t_edge, make_f3(0, 0, 1), edge_dir, t_edge_dir);
    float projection = f3_dot(f3_sub(point, edge_v0), edge_dir);
    float t_projection = f3_dot(f3_sub(t_point, t_edge_v0), edge_dir)
        + f3_dot(f3_sub(point, edge_v0), t_edge_dir);

    float3a closest;
    float3a t_closest;
    if (projection <= 0.f) {
        closest = edge_v0;
        t_closest = t_edge_v0;
    } else if (projection >= edge_len) {
        closest = edge_v1;
        t_closest = t_edge_v1;
    } else {
        closest = f3_add(edge_v0, f3_mul(edge_dir, projection));
        t_closest = f3_add(
            t_edge_v0,
            f3_add(f3_mul(t_edge_dir, projection), f3_mul(edge_dir, t_projection))
        );
    }

    (void) t_edge_len;
    return length_jvp(f3_sub(point, closest), f3_sub(t_point, t_closest));
}

struct FWeightSlotJvpData {
    float3a plane_point;
    float3a plane_normal;
    float3a t_plane_point;
    float3a t_plane_normal;
    float eta_r;
    float mu_r;
    float sigma;
    float gain;
};

struct FWeightChainJvpEval {
    bool geom_valid;
    float3a image_source;
    float3a t_image_source;
    float3a tx_pos;
    float3a t_tx_pos;
    float3a hit_points[REFL_MAX_CHAIN_DEPTH];
    float3a normals[REFL_MAX_CHAIN_DEPTH];
    float3a t_hit_points[REFL_MAX_CHAIN_DEPTH];
    float3a t_normals[REFL_MAX_CHAIN_DEPTH];
    Complex3 chain_vec;
    Complex3 t_chain_vec;
};

__device__ __forceinline__ FWeightSlotJvpData load_f_weight_jvp_slot(
    int slot,
    int pI,
    int tid,
    int override_slot,
    int nPaths,
    int nPairs,
    const float* __restrict__ ppx, const float* __restrict__ ppy, const float* __restrict__ ppz,
    const float* __restrict__ pnx, const float* __restrict__ pny, const float* __restrict__ pnz,
    const float* __restrict__ s_eta, const float* __restrict__ s_mu,
    const float* __restrict__ s_sig, const float* __restrict__ s_gn,
    const float* __restrict__ adjacent_ppx, const float* __restrict__ adjacent_ppy, const float* __restrict__ adjacent_ppz,
    const float* __restrict__ adjacent_pnx, const float* __restrict__ adjacent_pny, const float* __restrict__ adjacent_pnz,
    const float* __restrict__ adjacent_eta, const float* __restrict__ adjacent_mu,
    const float* __restrict__ adjacent_sig, const float* __restrict__ adjacent_gn,
    const float* __restrict__ t_ppx, const float* __restrict__ t_ppy, const float* __restrict__ t_ppz,
    const float* __restrict__ t_pnx, const float* __restrict__ t_pny, const float* __restrict__ t_pnz,
    const float* __restrict__ t_adjacent_ppx, const float* __restrict__ t_adjacent_ppy,
    const float* __restrict__ t_adjacent_ppz,
    const float* __restrict__ t_adjacent_pnx, const float* __restrict__ t_adjacent_pny,
    const float* __restrict__ t_adjacent_pnz)
{
    if (slot == override_slot) {
        int base = slot * nPairs + tid;
        return {
            make_f3(adjacent_ppx[base], adjacent_ppy[base], adjacent_ppz[base]),
            make_f3(adjacent_pnx[base], adjacent_pny[base], adjacent_pnz[base]),
            make_f3(t_adjacent_ppx[base], t_adjacent_ppy[base], t_adjacent_ppz[base]),
            make_f3(t_adjacent_pnx[base], t_adjacent_pny[base], t_adjacent_pnz[base]),
            adjacent_eta[base],
            adjacent_mu[base],
            adjacent_sig[base],
            adjacent_gn[base],
        };
    }

    int base = slot * nPaths + pI;
    return {
        make_f3(ppx[base], ppy[base], ppz[base]),
        make_f3(pnx[base], pny[base], pnz[base]),
        make_f3(t_ppx[base], t_ppy[base], t_ppz[base]),
        make_f3(t_pnx[base], t_pny[base], t_pnz[base]),
        s_eta[base],
        s_mu[base],
        s_sig[base],
        s_gn[base],
    };
}

__device__ __forceinline__ void f_weight_image_source_from_tx_jvp(
    float3a tx_pos,
    float3a t_tx_pos,
    int pI,
    int tid,
    int override_slot,
    int nPaths,
    int nPairs,
    int chainDepth,
    const float* __restrict__ ppx, const float* __restrict__ ppy, const float* __restrict__ ppz,
    const float* __restrict__ pnx, const float* __restrict__ pny, const float* __restrict__ pnz,
    const float* __restrict__ s_eta, const float* __restrict__ s_mu,
    const float* __restrict__ s_sig, const float* __restrict__ s_gn,
    const float* __restrict__ adjacent_ppx, const float* __restrict__ adjacent_ppy, const float* __restrict__ adjacent_ppz,
    const float* __restrict__ adjacent_pnx, const float* __restrict__ adjacent_pny, const float* __restrict__ adjacent_pnz,
    const float* __restrict__ adjacent_eta, const float* __restrict__ adjacent_mu,
    const float* __restrict__ adjacent_sig, const float* __restrict__ adjacent_gn,
    const float* __restrict__ t_ppx, const float* __restrict__ t_ppy, const float* __restrict__ t_ppz,
    const float* __restrict__ t_pnx, const float* __restrict__ t_pny, const float* __restrict__ t_pnz,
    const float* __restrict__ t_adjacent_ppx, const float* __restrict__ t_adjacent_ppy,
    const float* __restrict__ t_adjacent_ppz,
    const float* __restrict__ t_adjacent_pnx, const float* __restrict__ t_adjacent_pny,
    const float* __restrict__ t_adjacent_pnz,
    float3a& image_source,
    float3a& t_image_source)
{
    image_source = tx_pos;
    t_image_source = t_tx_pos;
    for (int slot = 0; slot < chainDepth; ++slot) {
        FWeightSlotJvpData data = load_f_weight_jvp_slot(
            slot, pI, tid, override_slot, nPaths, nPairs,
            ppx, ppy, ppz, pnx, pny, pnz,
            s_eta, s_mu, s_sig, s_gn,
            adjacent_ppx, adjacent_ppy, adjacent_ppz,
            adjacent_pnx, adjacent_pny, adjacent_pnz,
            adjacent_eta, adjacent_mu, adjacent_sig, adjacent_gn,
            t_ppx, t_ppy, t_ppz, t_pnx, t_pny, t_pnz,
            t_adjacent_ppx, t_adjacent_ppy, t_adjacent_ppz,
            t_adjacent_pnx, t_adjacent_pny, t_adjacent_pnz);
        float3a old_source = image_source;
        float3a old_t_source = t_image_source;
        image_source = reflect_point_across_plane(old_source, data.plane_point, data.plane_normal);
        t_image_source = reflect_point_jvp(
            old_source, old_t_source,
            data.plane_point, data.t_plane_point,
            data.plane_normal, data.t_plane_normal);
    }
}

__device__ __forceinline__ FWeightChainJvpEval evaluate_f_weight_chain_jvp(
    float3a image_source,
    float3a t_image_source,
    float3a rx,
    float3a t_rx,
    int pI,
    int tid,
    int override_slot,
    int nPaths,
    int nPairs,
    int chainDepth,
    float tx_px,
    float tx_py,
    float tx_pz,
    float omega,
    const float* __restrict__ ppx, const float* __restrict__ ppy, const float* __restrict__ ppz,
    const float* __restrict__ pnx, const float* __restrict__ pny, const float* __restrict__ pnz,
    const float* __restrict__ s_eta, const float* __restrict__ s_mu,
    const float* __restrict__ s_sig, const float* __restrict__ s_gn,
    const float* __restrict__ adjacent_ppx, const float* __restrict__ adjacent_ppy, const float* __restrict__ adjacent_ppz,
    const float* __restrict__ adjacent_pnx, const float* __restrict__ adjacent_pny, const float* __restrict__ adjacent_pnz,
    const float* __restrict__ adjacent_eta, const float* __restrict__ adjacent_mu,
    const float* __restrict__ adjacent_sig, const float* __restrict__ adjacent_gn,
    const float* __restrict__ t_ppx, const float* __restrict__ t_ppy, const float* __restrict__ t_ppz,
    const float* __restrict__ t_pnx, const float* __restrict__ t_pny, const float* __restrict__ t_pnz,
    const float* __restrict__ t_adjacent_ppx, const float* __restrict__ t_adjacent_ppy,
    const float* __restrict__ t_adjacent_ppz,
    const float* __restrict__ t_adjacent_pnx, const float* __restrict__ t_adjacent_pny,
    const float* __restrict__ t_adjacent_pnz)
{
    FWeightChainJvpEval out;
    out.geom_valid = true;
    out.image_source = image_source;
    out.t_image_source = t_image_source;
    out.tx_pos = image_source;
    out.t_tx_pos = t_image_source;
    out.chain_vec = c3_zero();
    out.t_chain_vec = c3_zero();
    if (chainDepth <= 0) {
        return out;
    }

    float3a curSrc = image_source;
    float3a curTgt = rx;
    float3a t_curSrc = t_image_source;
    float3a t_curTgt = t_rx;

    for (int slot = chainDepth - 1; slot >= 0; --slot) {
        FWeightSlotJvpData data = load_f_weight_jvp_slot(
            slot, pI, tid, override_slot, nPaths, nPairs,
            ppx, ppy, ppz, pnx, pny, pnz,
            s_eta, s_mu, s_sig, s_gn,
            adjacent_ppx, adjacent_ppy, adjacent_ppz,
            adjacent_pnx, adjacent_pny, adjacent_pnz,
            adjacent_eta, adjacent_mu, adjacent_sig, adjacent_gn,
            t_ppx, t_ppy, t_ppz, t_pnx, t_pny, t_pnz,
            t_adjacent_ppx, t_adjacent_ppy, t_adjacent_ppz,
            t_adjacent_pnx, t_adjacent_pny, t_adjacent_pnz);

        float3a seg = f3_sub(curTgt, curSrc);
        float denom = f3_dot(seg, data.plane_normal);
        out.geom_valid = out.geom_valid && fabsf(denom) > UTD_EPS;
        float sd = (fabsf(denom) < UTD_EPS) ? (denom >= 0.f ? UTD_EPS : -UTD_EPS) : denom;
        float t_val = f3_dot(f3_sub(data.plane_point, curSrc), data.plane_normal) / sd;
        out.geom_valid = out.geom_valid && (t_val > UTD_EPS) && (t_val < (1.f - UTD_EPS));
        t_val = fminf(fmaxf(t_val, 0.f), 1.f);
        float3a hitP = f3_add(curSrc, f3_mul(seg, t_val));

        float3a t_seg = f3_sub(t_curTgt, t_curSrc);
        float t_num = f3_dot(f3_sub(data.t_plane_point, t_curSrc), data.plane_normal)
            + f3_dot(f3_sub(data.plane_point, curSrc), data.t_plane_normal);
        float t_denom = f3_dot(t_seg, data.plane_normal) + f3_dot(seg, data.t_plane_normal);
        float dt = (t_num - t_val * t_denom) / sd;
        float3a t_hitP = f3_add(t_curSrc, f3_add(f3_mul(t_seg, t_val), f3_mul(seg, dt)));

        out.hit_points[slot] = hitP;
        out.normals[slot] = data.plane_normal;
        out.t_hit_points[slot] = t_hitP;
        out.t_normals[slot] = data.t_plane_normal;

        curTgt = hitP;
        t_curTgt = t_hitP;
        float3a oldSrc = curSrc;
        float3a oldTSrc = t_curSrc;
        curSrc = reflect_point_across_plane(oldSrc, data.plane_point, data.plane_normal);
        t_curSrc = reflect_point_jvp(
            oldSrc, oldTSrc,
            data.plane_point, data.t_plane_point,
            data.plane_normal, data.t_plane_normal);
    }

    out.tx_pos = curSrc;
    out.t_tx_pos = t_curSrc;

    float3a firstDir, t_firstDir;
    safe_normalize_jvp(
        f3_sub(out.hit_points[0], out.tx_pos),
        f3_sub(out.t_hit_points[0], out.t_tx_pos),
        make_f3(0, 0, 1),
        firstDir,
        t_firstDir);
    float3a polDir, t_polDir;
    project_polarization_to_ray_jvp(make_f3(tx_px, tx_py, tx_pz), firstDir, t_firstDir, polDir, t_polDir);
    Complex3 chain = {cplx(polDir.x, 0), cplx(polDir.y, 0), cplx(polDir.z, 0)};
    Complex3 t_chain = {cplx(t_polDir.x, 0), cplx(t_polDir.y, 0), cplx(t_polDir.z, 0)};

    float3a prev = out.tx_pos;
    float3a t_prev = out.t_tx_pos;
    for (int slot = 0; slot < chainDepth; ++slot) {
        FWeightSlotJvpData data = load_f_weight_jvp_slot(
            slot, pI, tid, override_slot, nPaths, nPairs,
            ppx, ppy, ppz, pnx, pny, pnz,
            s_eta, s_mu, s_sig, s_gn,
            adjacent_ppx, adjacent_ppy, adjacent_ppz,
            adjacent_pnx, adjacent_pny, adjacent_pnz,
            adjacent_eta, adjacent_mu, adjacent_sig, adjacent_gn,
            t_ppx, t_ppy, t_ppz, t_pnx, t_pny, t_pnz,
            t_adjacent_ppx, t_adjacent_ppy, t_adjacent_ppz,
            t_adjacent_pnx, t_adjacent_pny, t_adjacent_pnz);

        float3a inc, t_inc;
        safe_normalize_jvp(
            f3_sub(out.hit_points[slot], prev),
            f3_sub(out.t_hit_points[slot], t_prev),
            make_f3(0, 0, 1),
            inc,
            t_inc);
        float3a nrm, t_nrm;
        safe_normalize_jvp(out.normals[slot], out.t_normals[slot], make_f3(0, 1, 0), nrm, t_nrm);

        Complex3 out_p, out_t;
        reflect_field_jvp(
            chain, t_chain, inc, t_inc, nrm, t_nrm,
            data.eta_r, data.mu_r, data.sigma, omega, data.gain,
            out_p, out_t);
        chain = out_p;
        t_chain = out_t;
        prev = out.hit_points[slot];
        t_prev = out.t_hit_points[slot];
    }

    out.chain_vec = chain;
    out.t_chain_vec = t_chain;
    return out;
}

__global__ void reflection_f_weight_jvp_kernel(
    const int* __restrict__ pathIdx,
    const int* __restrict__ rxIdx_arr,
    const int* __restrict__ valid,
    const float* __restrict__ isx, const float* __restrict__ isy, const float* __restrict__ isz,
    const float* __restrict__ ppx, const float* __restrict__ ppy, const float* __restrict__ ppz,
    const float* __restrict__ pnx, const float* __restrict__ pny, const float* __restrict__ pnz,
    const float* __restrict__ s_eta, const float* __restrict__ s_mu, const float* __restrict__ s_sig, const float* __restrict__ s_gn,
    const int* __restrict__ transition_support_valid,
    const int* __restrict__ transition_primary_side,
    const float* __restrict__ transition_edge_distance,
    const float* __restrict__ transition_edge_v0_x,
    const float* __restrict__ transition_edge_v0_y,
    const float* __restrict__ transition_edge_v0_z,
    const float* __restrict__ transition_edge_v1_x,
    const float* __restrict__ transition_edge_v1_y,
    const float* __restrict__ transition_edge_v1_z,
    const int* __restrict__ adjacent_valid,
    const float* __restrict__ adjacent_ppx, const float* __restrict__ adjacent_ppy, const float* __restrict__ adjacent_ppz,
    const float* __restrict__ adjacent_pnx, const float* __restrict__ adjacent_pny, const float* __restrict__ adjacent_pnz,
    const float* __restrict__ adjacent_eta, const float* __restrict__ adjacent_mu,
    const float* __restrict__ adjacent_sig, const float* __restrict__ adjacent_gn,
    const float* __restrict__ rxx, const float* __restrict__ rxy, const float* __restrict__ rxz,
    float tx_px, float tx_py, float tx_pz,
    const float* __restrict__ t_isx, const float* __restrict__ t_isy, const float* __restrict__ t_isz,
    const float* __restrict__ t_ppx, const float* __restrict__ t_ppy, const float* __restrict__ t_ppz,
    const float* __restrict__ t_pnx, const float* __restrict__ t_pny, const float* __restrict__ t_pnz,
    const float* __restrict__ t_transition_edge_distance,
    const float* __restrict__ t_transition_edge_v0_x,
    const float* __restrict__ t_transition_edge_v0_y,
    const float* __restrict__ t_transition_edge_v0_z,
    const float* __restrict__ t_transition_edge_v1_x,
    const float* __restrict__ t_transition_edge_v1_y,
    const float* __restrict__ t_transition_edge_v1_z,
    const float* __restrict__ t_adjacent_ppx, const float* __restrict__ t_adjacent_ppy,
    const float* __restrict__ t_adjacent_ppz,
    const float* __restrict__ t_adjacent_pnx, const float* __restrict__ t_adjacent_pny,
    const float* __restrict__ t_adjacent_pnz,
    const float* __restrict__ t_rxx, const float* __restrict__ t_rxy, const float* __restrict__ t_rxz,
    float* __restrict__ to_xr, float* __restrict__ to_xi,
    float* __restrict__ to_yr, float* __restrict__ to_yi,
    float* __restrict__ to_zr, float* __restrict__ to_zi,
    int nPairs, int nPaths, int chainDepth, float k, float omega)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= nPairs) return;
    if (valid[tid] == 0) return;

    int pI = pathIdx[tid];
    int rI = rxIdx_arr[tid];

    float3a img = make_f3(isx[pI], isy[pI], isz[pI]);
    float3a rx = make_f3(rxx[rI], rxy[rI], rxz[rI]);
    float3a t_img = make_f3(t_isx[pI], t_isy[pI], t_isz[pI]);
    float3a t_rx = make_f3(t_rxx[rI], t_rxy[rI], t_rxz[rI]);

    float3a hitPts[REFL_MAX_CHAIN_DEPTH], normals[REFL_MAX_CHAIN_DEPTH];
    float3a t_hitPts[REFL_MAX_CHAIN_DEPTH], t_normals[REFL_MAX_CHAIN_DEPTH];

    float3a curSrc = img, curTgt = rx;
    float3a t_curSrc = t_img, t_curTgt = t_rx;

    for (int slot = chainDepth - 1; slot >= 0; --slot) {
        int base = slot * nPaths + pI;
        float3a plPt = make_f3(ppx[base], ppy[base], ppz[base]);
        float3a plN = make_f3(pnx[base], pny[base], pnz[base]);
        float3a t_plPt = make_f3(t_ppx[base], t_ppy[base], t_ppz[base]);
        float3a t_plN = make_f3(t_pnx[base], t_pny[base], t_pnz[base]);

        float3a seg = f3_sub(curTgt, curSrc);
        float denom = f3_dot(seg, plN);
        float sd = (fabsf(denom) < UTD_EPS) ? (denom >= 0 ? UTD_EPS : -UTD_EPS) : denom;
        float t_val = f3_dot(f3_sub(plPt, curSrc), plN) / sd;
        t_val = fminf(fmaxf(t_val, 0.f), 1.f);
        float3a hitP = f3_add(curSrc, f3_mul(seg, t_val));

        float3a t_seg = f3_sub(t_curTgt, t_curSrc);
        float t_num = f3_dot(f3_sub(t_plPt, t_curSrc), plN) + f3_dot(f3_sub(plPt, curSrc), t_plN);
        float t_denom = f3_dot(t_seg, plN) + f3_dot(seg, t_plN);
        float dt = (t_num - t_val * t_denom) / sd;
        float3a t_hitP = f3_add(t_curSrc, f3_add(f3_mul(t_seg, t_val), f3_mul(seg, dt)));

        hitPts[slot] = hitP;
        normals[slot] = plN;
        t_hitPts[slot] = t_hitP;
        t_normals[slot] = t_plN;

        curTgt = hitP;
        t_curTgt = t_hitP;

        float3a oldSrc = curSrc;
        float3a oldTSrc = t_curSrc;
        float d = f3_dot(f3_sub(oldSrc, plPt), plN);
        float t_d = f3_dot(f3_sub(oldTSrc, t_plPt), plN) + f3_dot(f3_sub(oldSrc, plPt), t_plN);
        curSrc = reflect_point_across_plane(oldSrc, plPt, plN);
        t_curSrc = f3_sub(oldTSrc, f3_add(f3_mul(t_plN, 2.f * d), f3_mul(plN, 2.f * t_d)));
    }

    float3a txPos = curSrc;
    float3a t_txPos = t_curSrc;

    float3a firstDir, t_firstDir;
    safe_normalize_jvp(
        f3_sub(hitPts[0], txPos),
        f3_sub(t_hitPts[0], t_txPos),
        make_f3(0,0,1),
        firstDir,
        t_firstDir
    );
    float3a polDir, t_polDir;
    project_polarization_to_ray_jvp(make_f3(tx_px, tx_py, tx_pz), firstDir, t_firstDir, polDir, t_polDir);
    Complex3 chain = {cplx(polDir.x,0), cplx(polDir.y,0), cplx(polDir.z,0)};
    Complex3 t_chain = {cplx(t_polDir.x,0), cplx(t_polDir.y,0), cplx(t_polDir.z,0)};

    float3a prev = txPos;
    float3a t_prev = t_txPos;
    for (int slot = 0; slot < chainDepth; ++slot) {
        float3a inc, t_inc;
        safe_normalize_jvp(
            f3_sub(hitPts[slot], prev),
            f3_sub(t_hitPts[slot], t_prev),
            make_f3(0,0,1),
            inc,
            t_inc
        );
        float3a nrm, t_nrm;
        safe_normalize_jvp(normals[slot], t_normals[slot], make_f3(0,1,0), nrm, t_nrm);
        int base = slot * nPaths + pI;

        Complex3 out_p, out_t;
        reflect_field_jvp(chain, t_chain, inc, t_inc, nrm, t_nrm,
                          s_eta[base], s_mu[base], s_sig[base], omega, s_gn[base],
                          out_p, out_t);
        chain = out_p;
        t_chain = out_t;
        prev = hitPts[slot];
        t_prev = t_hitPts[slot];
    }

    float3a delta = f3_sub(rx, img);
    float dist = safe_length(delta) + UTD_EPS;
    float wl = UTD_TWO_PI / k;
    float fspl = wl / (4.f * UTD_PI * dist);
    Complex phase = cplx_exp_phase(-k * dist);
    Complex unit_field = cplx_mul_real(phase, fspl);

    float3a t_delta = f3_sub(t_rx, t_img);
    float t_dist = f3_dot(f3_mul(delta, 1.f / dist), t_delta);
    float dfspl = -wl / (4.f * UTD_PI * dist * dist);
    Complex d_unit = cplx_add(
        cplx_mul_real(cplx_mul(cplx(0, -k), phase), fspl),
        cplx_mul_real(phase, dfspl));
    Complex t_unit = cplx_mul_real(d_unit, t_dist);

    Complex3 hard_result = c3_scale(chain, unit_field);
    Complex3 hard_tangent = c3_add(c3_scale(t_chain, unit_field), c3_scale(chain, t_unit));

    Complex primary_weights[REFL_MAX_CHAIN_DEPTH];
    Complex t_primary_weights[REFL_MAX_CHAIN_DEPTH];
    Complex adjacent_weights[REFL_MAX_CHAIN_DEPTH];
    Complex t_adjacent_weights[REFL_MAX_CHAIN_DEPTH];
    Complex chain_weight = cplx(1.f, 0.f);
    Complex t_chain_weight = cplx_zero();
    for (int slot = 0; slot < chainDepth; ++slot) {
        int tbase = slot * nPairs + tid;
        bool support = transition_support_valid[tbase] != 0;
        bool primary_side = transition_primary_side[tbase] != 0;
        Complex weight = primary_side ? cplx(1.f, 0.f) : cplx_zero();
        Complex t_weight = cplx_zero();
        Complex adjacent_weight = cplx_zero();
        Complex t_adjacent_weight = cplx_zero();
        if (support) {
            float d = transition_edge_distance[tbase];
            float3a edge_v0 = make_f3(
                transition_edge_v0_x[tbase],
                transition_edge_v0_y[tbase],
                transition_edge_v0_z[tbase]);
            float3a edge_v1 = make_f3(
                transition_edge_v1_x[tbase],
                transition_edge_v1_y[tbase],
                transition_edge_v1_z[tbase]);
            float3a t_edge_v0 = make_f3(
                t_transition_edge_v0_x[tbase],
                t_transition_edge_v0_y[tbase],
                t_transition_edge_v0_z[tbase]);
            float3a t_edge_v1 = make_f3(
                t_transition_edge_v1_x[tbase],
                t_transition_edge_v1_y[tbase],
                t_transition_edge_v1_z[tbase]);
            float t_d = segment_distance_jvp(
                hitPts[slot],
                t_hitPts[slot],
                edge_v0,
                t_edge_v0,
                edge_v1,
                t_edge_v1);
            if (!isfinite(t_d)) {
                t_d = t_transition_edge_distance[tbase];
            }
            float3a prev_pt = (slot == 0) ? txPos : hitPts[slot - 1];
            float3a next_pt = (slot + 1 < chainDepth) ? hitPts[slot + 1] : rx;
            float3a t_prev_pt = (slot == 0) ? t_txPos : t_hitPts[slot - 1];
            float3a t_next_pt = (slot + 1 < chainDepth) ? t_hitPts[slot + 1] : t_rx;

            float3a prev_vec = f3_sub(hitPts[slot], prev_pt);
            float3a next_vec = f3_sub(next_pt, hitPts[slot]);
            float s_prev = safe_length(prev_vec) + UTD_EPS;
            float s_next = safe_length(next_vec) + UTD_EPS;
            float t_s_prev = length_jvp(prev_vec, f3_sub(t_hitPts[slot], t_prev_pt));
            float t_s_next = length_jvp(next_vec, f3_sub(t_next_pt, t_hitPts[slot]));
            float den = s_prev + s_next + UTD_EPS;
            float eff = (s_prev * s_next) / den;
            float t_eff = ((t_s_prev * s_next + s_prev * t_s_next) * den -
                           (s_prev * s_next) * (t_s_prev + t_s_next)) / (den * den);
            float eff_safe = eff + UTD_EPS;
            float x = k * d * d / eff_safe;
            float t_x = k * ((2.f * d * t_d * eff_safe) - (d * d * t_eff)) /
                (eff_safe * eff_safe);

            Complex transition, t_transition;
            safe_transition_value_jvp(x, t_x, transition, t_transition);
            weight = primary_side ? transition : cplx_zero();
            t_weight = primary_side ? t_transition : cplx_zero();
            if (!primary_side && adjacent_valid[tbase] != 0) {
                adjacent_weight = transition;
                t_adjacent_weight = t_transition;
            }
        }
        primary_weights[slot] = weight;
        t_primary_weights[slot] = t_weight;
        adjacent_weights[slot] = adjacent_weight;
        t_adjacent_weights[slot] = t_adjacent_weight;
        t_chain_weight = cplx_add(cplx_mul(t_chain_weight, weight), cplx_mul(chain_weight, t_weight));
        chain_weight = cplx_mul(chain_weight, weight);
    }

    Complex3 result_t = c3_add(c3_scale(hard_tangent, chain_weight), c3_scale(hard_result, t_chain_weight));

    for (int slot = 0; slot < chainDepth; ++slot) {
        if (!cplx_any_nonzero(adjacent_weights[slot]) && !cplx_any_nonzero(t_adjacent_weights[slot])) {
            continue;
        }

        Complex residual_weight = adjacent_weights[slot];
        Complex t_residual_weight = t_adjacent_weights[slot];
        for (int other = 0; other < chainDepth; ++other) {
            if (other == slot) continue;
            Complex next_weight = cplx_mul(residual_weight, primary_weights[other]);
            Complex next_t_weight = cplx_add(
                cplx_mul(t_residual_weight, primary_weights[other]),
                cplx_mul(residual_weight, t_primary_weights[other])
            );
            residual_weight = next_weight;
            t_residual_weight = next_t_weight;
        }
        if (!cplx_any_nonzero(residual_weight) && !cplx_any_nonzero(t_residual_weight)) {
            continue;
        }

        float3a branch_img, t_branch_img;
        f_weight_image_source_from_tx_jvp(
            txPos, t_txPos,
            pI, tid, slot,
            nPaths, nPairs, chainDepth,
            ppx, ppy, ppz, pnx, pny, pnz,
            s_eta, s_mu, s_sig, s_gn,
            adjacent_ppx, adjacent_ppy, adjacent_ppz,
            adjacent_pnx, adjacent_pny, adjacent_pnz,
            adjacent_eta, adjacent_mu, adjacent_sig, adjacent_gn,
            t_ppx, t_ppy, t_ppz, t_pnx, t_pny, t_pnz,
            t_adjacent_ppx, t_adjacent_ppy, t_adjacent_ppz,
            t_adjacent_pnx, t_adjacent_pny, t_adjacent_pnz,
            branch_img, t_branch_img);

        FWeightChainJvpEval branch = evaluate_f_weight_chain_jvp(
            branch_img,
            t_branch_img,
            rx,
            t_rx,
            pI,
            tid,
            slot,
            nPaths,
            nPairs,
            chainDepth,
            tx_px,
            tx_py,
            tx_pz,
            omega,
            ppx, ppy, ppz, pnx, pny, pnz,
            s_eta, s_mu, s_sig, s_gn,
            adjacent_ppx, adjacent_ppy, adjacent_ppz,
            adjacent_pnx, adjacent_pny, adjacent_pnz,
            adjacent_eta, adjacent_mu, adjacent_sig, adjacent_gn,
            t_ppx, t_ppy, t_ppz, t_pnx, t_pny, t_pnz,
            t_adjacent_ppx, t_adjacent_ppy, t_adjacent_ppz,
            t_adjacent_pnx, t_adjacent_pny, t_adjacent_pnz);
        if (!branch.geom_valid) {
            continue;
        }

        Complex3 branch_weighted_t = c3_add(
            c3_scale(branch.t_chain_vec, residual_weight),
            c3_scale(branch.chain_vec, t_residual_weight)
        );
        result_t = c3_add(
            result_t,
            c3_add(
                c3_scale(branch_weighted_t, unit_field),
                c3_scale(c3_scale(branch.chain_vec, residual_weight), t_unit)
            )
        );
    }

    atomicAdd(&to_xr[rI], result_t.x.re);
    atomicAdd(&to_xi[rI], result_t.x.im);
    atomicAdd(&to_yr[rI], result_t.y.re);
    atomicAdd(&to_yi[rI], result_t.y.im);
    atomicAdd(&to_zr[rI], result_t.z.re);
    atomicAdd(&to_zi[rI], result_t.z.im);
}

} // anonymous namespace

// =========================================================================
// Host launcher
// =========================================================================
void reflection_accumulate_jvp(
    const int* path_idx, const int* rx_idx, const int* valid_mask,
    const float* isx, const float* isy, const float* isz,
    const float* ppx, const float* ppy, const float* ppz,
    const float* pnx, const float* pny, const float* pnz,
    const float* s_eta, const float* s_mu, const float* s_sig, const float* s_gn,
    const float* rxx, const float* rxy, const float* rxz,
    float tx_px, float tx_py, float tx_pz,
    const float* t_isx, const float* t_isy, const float* t_isz,
    const float* t_ppx, const float* t_ppy, const float* t_ppz,
    const float* t_pnx, const float* t_pny, const float* t_pnz,
    const float* t_rxx, const float* t_rxy, const float* t_rxz,
    float* to_xr, float* to_xi, float* to_yr, float* to_yi, float* to_zr, float* to_zi,
    int n_pairs, int n_paths, int chain_depth, float k, float omega)
{
    if (n_pairs <= 0) return;
    if (chain_depth > REFL_MAX_CHAIN_DEPTH)
        throw std::runtime_error("chain_depth exceeds REFL_MAX_CHAIN_DEPTH");

    constexpr int BLOCK = 256;
    int grid = (n_pairs + BLOCK - 1) / BLOCK;

    reflection_jvp_kernel<<<grid, BLOCK>>>(
        path_idx, rx_idx, valid_mask,
        isx,isy,isz, ppx,ppy,ppz, pnx,pny,pnz, s_eta,s_mu,s_sig,s_gn,
        rxx,rxy,rxz, tx_px,tx_py,tx_pz,
        t_isx,t_isy,t_isz, t_ppx,t_ppy,t_ppz, t_pnx,t_pny,t_pnz,
        t_rxx,t_rxy,t_rxz,
        to_xr,to_xi, to_yr,to_yi, to_zr,to_zi,
        n_pairs, n_paths, chain_depth, k, omega);

    throw_cuda(cudaGetLastError(), "reflection_jvp_kernel launch");
}

void reflection_accumulate_f_weight_jvp(
    const int* path_idx, const int* rx_idx, const int* valid_mask,
    const float* isx, const float* isy, const float* isz,
    const float* ppx, const float* ppy, const float* ppz,
    const float* pnx, const float* pny, const float* pnz,
    const float* s_eta, const float* s_mu, const float* s_sig, const float* s_gn,
    const int* transition_support_valid,
    const int* transition_primary_side,
    const float* transition_edge_distance,
    const float* transition_edge_v0_x, const float* transition_edge_v0_y, const float* transition_edge_v0_z,
    const float* transition_edge_v1_x, const float* transition_edge_v1_y, const float* transition_edge_v1_z,
    const int* adjacent_valid,
    const float* adjacent_ppx, const float* adjacent_ppy, const float* adjacent_ppz,
    const float* adjacent_pnx, const float* adjacent_pny, const float* adjacent_pnz,
    const float* adjacent_eta, const float* adjacent_mu,
    const float* adjacent_sig, const float* adjacent_gn,
    const float* rxx, const float* rxy, const float* rxz,
    float tx_px, float tx_py, float tx_pz,
    const float* t_isx, const float* t_isy, const float* t_isz,
    const float* t_ppx, const float* t_ppy, const float* t_ppz,
    const float* t_pnx, const float* t_pny, const float* t_pnz,
    const float* t_transition_edge_distance,
    const float* t_transition_edge_v0_x, const float* t_transition_edge_v0_y, const float* t_transition_edge_v0_z,
    const float* t_transition_edge_v1_x, const float* t_transition_edge_v1_y, const float* t_transition_edge_v1_z,
    const float* t_adjacent_ppx, const float* t_adjacent_ppy, const float* t_adjacent_ppz,
    const float* t_adjacent_pnx, const float* t_adjacent_pny, const float* t_adjacent_pnz,
    const float* t_rxx, const float* t_rxy, const float* t_rxz,
    float* to_xr, float* to_xi, float* to_yr, float* to_yi, float* to_zr, float* to_zi,
    int n_pairs, int n_paths, int chain_depth, float k, float omega)
{
    if (n_pairs <= 0) return;
    if (chain_depth > REFL_MAX_CHAIN_DEPTH)
        throw std::runtime_error("chain_depth exceeds REFL_MAX_CHAIN_DEPTH");

    constexpr int BLOCK = 256;
    int grid = (n_pairs + BLOCK - 1) / BLOCK;

    reflection_f_weight_jvp_kernel<<<grid, BLOCK>>>(
        path_idx, rx_idx, valid_mask,
        isx, isy, isz,
        ppx, ppy, ppz,
        pnx, pny, pnz,
        s_eta, s_mu, s_sig, s_gn,
        transition_support_valid,
        transition_primary_side,
        transition_edge_distance,
        transition_edge_v0_x, transition_edge_v0_y, transition_edge_v0_z,
        transition_edge_v1_x, transition_edge_v1_y, transition_edge_v1_z,
        adjacent_valid,
        adjacent_ppx, adjacent_ppy, adjacent_ppz,
        adjacent_pnx, adjacent_pny, adjacent_pnz,
        adjacent_eta, adjacent_mu, adjacent_sig, adjacent_gn,
        rxx, rxy, rxz,
        tx_px, tx_py, tx_pz,
        t_isx, t_isy, t_isz,
        t_ppx, t_ppy, t_ppz,
        t_pnx, t_pny, t_pnz,
        t_transition_edge_distance,
        t_transition_edge_v0_x, t_transition_edge_v0_y, t_transition_edge_v0_z,
        t_transition_edge_v1_x, t_transition_edge_v1_y, t_transition_edge_v1_z,
        t_adjacent_ppx, t_adjacent_ppy, t_adjacent_ppz,
        t_adjacent_pnx, t_adjacent_pny, t_adjacent_pnz,
        t_rxx, t_rxy, t_rxz,
        to_xr, to_xi,
        to_yr, to_yi,
        to_zr, to_zi,
        n_pairs, n_paths, chain_depth, k, omega);

    throw_cuda(cudaGetLastError(), "reflection_f_weight_jvp_kernel launch");
}

} // namespace witwin::channel::native_ext
