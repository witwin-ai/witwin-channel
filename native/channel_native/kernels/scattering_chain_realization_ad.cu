// ADR-021 Op B companions: VJP/JVP of the multi-bounce coherent phase-screen
// chain realization (forward in scattering_chain_realization.cu).
//
// The forward is total = sum_rows row_value, with per row
//   q = k0*(d_o - d_i); q_int = -q; q_int_n = n_hat . q_int
//   I = A2 * sum_t w_t exp(-j (pos_t . q_int + q_int_n h_t))     ('integral')
//   pref = |q|^2 / (4*pi * max(q . n, 1e-9))
//   E_rx = <A_2 . diag(r_te,r_tm) . A_1 . e_tx, rx_axis>          (Jones chain)
//   carrier = exp(-j (k0 (L1+L2) + q . centroid))
//   value = (j*pref) * E_rx * carrier * (sp1*sp2)
//   row_value = value * I
//
// The op-2 quadrature adjoint (heights / k0 / q-through-d_i,d_o) is reused
// verbatim (scattering_patch_integral_ad.cu); the Jones scalar E_rx replaces
// op-2's caller jones scalar, so its reverse/forward derivative rides the same
// per-bounce reflect_frame / slab_fresnel machinery as
// field_transport_reflection.cu and the em::stack_rt layer dual of
// rayd/torch/rf/field_transport_ad.cuh.  Both companions recompute the forward
// intermediates in primal expression order (ADR-004); this TU compiles
// --fmad=false in lockstep with the forward.
//
// Backward: one block per row, kQuadPoints threads. All threads reduce I and
// the phase-derivative vector S_phase and scatter the heights VJP (atomicAdd);
// thread 0 replays the chain, folds the row cotangent, reverse-mode
// differentiates the Jones sandwich (per-bounce direct stores for chain
// material/geometry grads, atomicAdd for the CSR layer stack and the
// frequency/k0 scalars).  JVP: block per row, dual reduction of I and t_I,
// thread 0 forms t_row_value from a fixed-order dual sweep, second block
// tree-reduces tangent_total.  No float atomics in the JVP.
//
// Endpoint positions source/vertex/target are frozen structural inputs (no
// tangent, no gradient), consistent with the frozen section 4.2/4.3 tangent
// and grad lists; see the owner's plan-10a section-4 gap note in the forward TU.

#include "field_transport_ad_common.cuh"

namespace {

constexpr int kQuadPoints = 256;
constexpr int kReduceBlock = 256;
constexpr float kPi = 3.14159265358979323846f;

using cfloat = c10::complex<float>;

__device__ __forceinline__ field::float3a load_leg3(
    const float* __restrict__ values, int64_t row, int bounce) {
    return load_sequence3f(values, row, bounce, kMaxAdDepth);
}

__device__ __forceinline__ float redot(field::Complex g, field::Complex x) {
    return g.re * x.re + g.im * x.im;  // Re(conj(g) * x)
}
__device__ __forceinline__ field::Complex cmulj(field::Complex a) {
    return field::cplx(-a.im, a.re);  // a * j
}

struct Texel4 { int idx[4]; float wgt[4]; };

// PhaseScreenRuntime.sample_height + the 4 bilinear texel indices/weights.
__device__ __forceinline__ float sample_height_tex(
    const float* __restrict__ heights, int h_rows, int w_cols, float u, float v,
    Texel4& tex) {
    const float tx = fminf(fmaxf(u * w_cols - 0.5f, 0.0f), static_cast<float>(w_cols - 1));
    const float ty = fminf(fmaxf(v * h_rows - 0.5f, 0.0f), static_cast<float>(h_rows - 1));
    const float x0 = floorf(tx);
    const float y0 = floorf(ty);
    const float wx = tx - x0;
    const float wy = ty - y0;
    const int ix0 = static_cast<int>(x0);
    const int iy0 = static_cast<int>(y0);
    const int ix1 = min(ix0 + 1, w_cols - 1);
    const int iy1 = min(iy0 + 1, h_rows - 1);
    const int i00 = iy0 * w_cols + ix0;
    const int i01 = iy0 * w_cols + ix1;
    const int i10 = iy1 * w_cols + ix0;
    const int i11 = iy1 * w_cols + ix1;
    const float t00 = heights[i00];
    const float t01 = heights[i01];
    const float t10 = heights[i10];
    const float t11 = heights[i11];
    const float top = t00 * (1.0f - wx) + t01 * wx;
    const float bot = t10 * (1.0f - wx) + t11 * wx;
    tex.idx[0] = i00; tex.wgt[0] = (1.0f - wx) * (1.0f - wy);
    tex.idx[1] = i01; tex.wgt[1] = wx * (1.0f - wy);
    tex.idx[2] = i10; tex.wgt[2] = (1.0f - wx) * wy;
    tex.idx[3] = i11; tex.wgt[3] = wx * wy;
    return top * (1.0f - wy) + bot * wy;
}

__device__ __forceinline__ field::float3a stable_tangent(field::float3a n) {
    const float ax = fabsf(n.x), ay = fabsf(n.y), az = fabsf(n.z);
    field::float3a axis = field::make_f3(0.0f, 0.0f, 0.0f);
    if (ax <= ay && ax <= az) axis.x = 1.0f;
    else if (ay <= az) axis.y = 1.0f;
    else axis.z = 1.0f;
    const float proj = field::f3_dot(axis, n);
    field::float3a t = field::make_f3(
        axis.x - proj * n.x, axis.y - proj * n.y, axis.z - proj * n.z);
    const float norm = fmaxf(sqrtf(field::f3_dot(t, t)), 1.0e-12f);
    return field::f3_mul(t, 1.0f / norm);
}

__device__ __forceinline__ void sp_basis(
    field::float3a n, field::float3a d, field::float3a backup,
    field::float3a& s, field::float3a& p) {
    const field::float3a raw = field::f3_cross(n, d);
    const float norm = sqrtf(field::f3_dot(raw, raw));
    if (norm < 1.0e-6f) s = backup;
    else s = field::f3_mul(raw, 1.0f / fmaxf(norm, 1.0e-12f));
    p = field::f3_cross(s, d);
}

// Dual of sp_basis (n and backup frozen; d carries the tangent).
__device__ __forceinline__ void dual_sp_basis(
    field::float3a n, ad::DualF3 d, field::float3a backup,
    ad::DualF3& s, ad::DualF3& p) {
    const ad::DualF3 nn = ad::df3_const(n);
    const ad::DualF3 raw = ad::df3_cross(nn, d);
    const float norm = sqrtf(field::f3_dot(raw.v, raw.v));
    if (norm < 1.0e-6f) {
        s.v = backup;
        s.d = field::f3_zero();
    } else {
        const float inv = 1.0f / fmaxf(norm, 1.0e-12f);
        s.v = field::f3_mul(raw.v, inv);
        const float rr = field::f3_dot(raw.v, raw.d);
        s.d = field::f3_sub(
            field::f3_mul(raw.d, inv), field::f3_mul(raw.v, rr * inv * inv * inv));
    }
    p.v = field::f3_cross(s.v, d.v);
    p.d = field::f3_add(field::f3_cross(s.d, d.v), field::f3_cross(s.v, d.d));
}

// Reverse of sp_basis into the direction d (n / backup frozen).
__device__ __forceinline__ field::float3a adj_sp_basis(
    field::float3a n, field::float3a d, field::float3a backup,
    field::float3a g_s_in, field::float3a g_p) {
    const field::float3a raw = field::f3_cross(n, d);
    const float norm = sqrtf(field::f3_dot(raw, raw));
    field::float3a s;
    if (norm < 1.0e-6f) s = backup;
    else s = field::f3_mul(raw, 1.0f / fmaxf(norm, 1.0e-12f));
    field::float3a g_d = field::f3_zero();
    // p = cross(s, d): g_s += d x g_p ; g_d += g_p x s.
    field::float3a g_s = field::f3_add(g_s_in, field::f3_cross(d, g_p));
    g_d = field::f3_add(g_d, field::f3_cross(g_p, s));
    // s = normalize(raw) on the unclamped branch, else constant.
    if (norm >= 1.0e-6f) {
        const float inv = 1.0f / norm;
        const float rg = field::f3_dot(raw, g_s);
        const field::float3a g_raw = field::f3_sub(
            field::f3_mul(g_s, inv), field::f3_mul(raw, rg * inv * inv * inv));
        // raw = cross(n, d): g_d += g_raw x n.
        g_d = field::f3_add(g_d, field::f3_cross(g_raw, n));
    }
    return g_d;
}

// dpref/dq (real 3-vector), verbatim op-2 (scattering_patch_integral_ad.cu).
__device__ __forceinline__ field::float3a pref_grad_q(
    field::float3a q, field::float3a n, float q_norm2) {
    const float qn = field::f3_dot(q, n);
    const float qn_c = fmaxf(qn, 1.0e-9f);
    const float flag = qn > 1.0e-9f ? 1.0f : 0.0f;
    const float inv = 1.0f / (4.0f * kPi * qn_c * qn_c);
    return field::make_f3(
        (2.0f * q.x * qn_c - q_norm2 * flag * n.x) * inv,
        (2.0f * q.y * qn_c - q_norm2 * flag * n.y) * inv,
        (2.0f * q.z * qn_c - q_norm2 * flag * n.z) * inv);
}

// ---------------------------------------------------------------------------
// Forward recording of one specular leg (mirrors transport_leg in the forward
// TU) so the backward can reverse the identical chain.
// ---------------------------------------------------------------------------

struct LegRecord {
    transport::ReflectFrame frames[kMaxAdDepth];
    field::Complex3 value_in[kMaxAdDepth];
    field::Complex e_s[kMaxAdDepth];
    field::Complex e_p[kMaxAdDepth];
    field::Complex r_te[kMaxAdDepth];
    field::Complex r_tm[kMaxAdDepth];
};

__device__ __forceinline__ field::Complex3 record_leg(
    field::Complex3 value,
    field::float3a start,
    field::float3a end,
    const float* positions, const float* normals,
    const float* eps_r, const float* sigma_e, const float* mu_r,
    const float* gain, const float* thickness,
    int64_t row, int depth, float frequency_hz,
    LegRecord& rec, field::float3a& last_dir) {
    const field::float3a ez = field::make_f3(0.0f, 0.0f, 1.0f);
    field::float3a previous = start;
    const field::float3a first_hit = depth > 0 ? load_leg3(positions, row, 0) : end;
    field::float3a outgoing = field::safe_normalize(
        field::f3_sub(first_hit, start), ez);
    for (int bounce = 0; bounce < depth; ++bounce) {
        const field::float3a hit = load_leg3(positions, row, bounce);
        const field::float3a incident = field::safe_normalize(
            field::f3_sub(hit, previous), outgoing);
        const int64_t slot = row * kMaxAdDepth + bounce;
        const transport::ReflectFrame frame = transport::reflect_frame(
            incident, load_leg3(normals, row, bounce));
        field::Complex r_te, r_tm;
        transport::slab_fresnel(
            frame.cos_theta, eps_r[slot], sigma_e[slot], mu_r[slot], gain[slot],
            thickness[slot], frequency_hz, r_te, r_tm);
        const field::Complex e_s = transport::complex3_dot_real(value, frame.s_axis);
        const field::Complex e_p = transport::complex3_dot_real(value, frame.p_in);
        rec.frames[bounce] = frame;
        rec.value_in[bounce] = value;
        rec.e_s[bounce] = e_s;
        rec.e_p[bounce] = e_p;
        rec.r_te[bounce] = r_te;
        rec.r_tm[bounce] = r_tm;
        value = field::c3_add(
            field::cplx_scale_real(frame.s_axis, field::cplx_mul(r_te, e_s)),
            field::cplx_scale_real(frame.p_out, field::cplx_mul(r_tm, e_p)));
        outgoing = frame.reflected_direction;
        previous = hit;
    }
    last_dir = field::safe_normalize(field::f3_sub(end, previous), outgoing);
    return value;
}

// Reverse one specular leg's bounce loop (no path-length term: the chain leg
// carries no propagation length). Writes per-bounce material grads (direct)
// and, under need_geometry, normals + hits[1..depth-1] (direct). Returns the
// cotangent on the leg INPUT field; g_bounce0_hit / g_incident_pre are the
// bounce-0 pieces the caller finalizes. g_carry_seed / g_outgoing_seed are the
// downstream cotangents on the leg's final previous hit / outgoing direction.
__device__ field::Complex3 reverse_leg(
    const LegRecord& rec, int depth,
    field::float3a start,
    const float* positions, const float* normals,
    const float* eps_r, const float* sigma_e, const float* mu_r,
    const float* gain, const float* thickness,
    float frequency_hz,
    field::Complex3 g_field_out,
    field::float3a g_carry_seed,
    field::float3a g_outgoing_seed,
    int64_t row, bool need_geometry,
    float* grad_eps, float* grad_sigma, float* grad_gain, float* grad_thick,
    float* grad_positions, float* grad_normals,
    float& g_freq,
    field::float3a& g_bounce0_hit,
    field::float3a& g_incident_pre) {
    const field::float3a ez = field::make_f3(0.0f, 0.0f, 1.0f);
    field::Complex3 g_chain = g_field_out;
    field::float3a g_carry = g_carry_seed;
    field::float3a g_outgoing = g_outgoing_seed;
    g_bounce0_hit = field::f3_zero();
    g_incident_pre = field::f3_zero();
    for (int bounce = depth - 1; bounce >= 0; --bounce) {
        const transport::ReflectFrame& frame = rec.frames[bounce];
        field::float3a g_s_axis = field::f3_zero();
        field::float3a g_p_in = field::f3_zero();
        field::float3a g_p_out = field::f3_zero();
        field::Complex gs = field::cplx_zero();
        field::Complex gp = field::cplx_zero();
        field::adj_cplx_scale_real(
            frame.s_axis, field::cplx_mul(rec.r_te[bounce], rec.e_s[bounce]),
            g_chain, g_s_axis, gs);
        field::adj_cplx_scale_real(
            frame.p_out, field::cplx_mul(rec.r_tm[bounce], rec.e_p[bounce]),
            g_chain, g_p_out, gp);
        field::Complex g_r_te = field::cplx_zero();
        field::Complex g_r_tm = field::cplx_zero();
        field::Complex g_e_s = field::cplx_zero();
        field::Complex g_e_p = field::cplx_zero();
        field::adj_cplx_mul(rec.r_te[bounce], rec.e_s[bounce], gs, g_r_te, g_e_s);
        field::adj_cplx_mul(rec.r_tm[bounce], rec.e_p[bounce], gp, g_r_tm, g_e_p);
        field::Complex3 g_value_in = field::c3_zero();
        field::adj_cplx_dot_real(
            rec.value_in[bounce], frame.s_axis, g_e_s, g_value_in, g_s_axis);
        field::adj_cplx_dot_real(
            rec.value_in[bounce], frame.p_in, g_e_p, g_value_in, g_p_in);
        g_chain = g_value_in;

        const int64_t slot = row * kMaxAdDepth + bounce;
        const float b_eps = eps_r[slot];
        const float b_sigma = sigma_e[slot];
        const float b_mu = mu_r[slot];
        const float b_gain = gain[slot];
        const float b_thick = thickness[slot];
        ad::DualC dte, dtm;
        if (grad_eps != nullptr) {
            ad::slab_fresnel_dual(frame.cos_theta, b_eps, b_sigma, b_mu, b_gain,
                b_thick, frequency_hz, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, dte, dtm);
            grad_eps[slot] = ad::adj_dot(g_r_te, dte.d) + ad::adj_dot(g_r_tm, dtm.d);
        }
        if (grad_sigma != nullptr) {
            ad::slab_fresnel_dual(frame.cos_theta, b_eps, b_sigma, b_mu, b_gain,
                b_thick, frequency_hz, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, dte, dtm);
            grad_sigma[slot] = ad::adj_dot(g_r_te, dte.d) + ad::adj_dot(g_r_tm, dtm.d);
        }
        if (grad_gain != nullptr) {
            ad::slab_fresnel_dual(frame.cos_theta, b_eps, b_sigma, b_mu, b_gain,
                b_thick, frequency_hz, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, dte, dtm);
            grad_gain[slot] = ad::adj_dot(g_r_te, dte.d) + ad::adj_dot(g_r_tm, dtm.d);
        }
        if (grad_thick != nullptr) {
            ad::slab_fresnel_dual(frame.cos_theta, b_eps, b_sigma, b_mu, b_gain,
                b_thick, frequency_hz, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, dte, dtm);
            grad_thick[slot] = ad::adj_dot(g_r_te, dte.d) + ad::adj_dot(g_r_tm, dtm.d);
        }
        {
            ad::slab_fresnel_dual(frame.cos_theta, b_eps, b_sigma, b_mu, b_gain,
                b_thick, frequency_hz, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, dte, dtm);
            g_freq += ad::adj_dot(g_r_te, dte.d) + ad::adj_dot(g_r_tm, dtm.d);
        }
        if (!need_geometry)
            continue;

        ad::slab_fresnel_dual(frame.cos_theta, b_eps, b_sigma, b_mu, b_gain,
            b_thick, frequency_hz, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, dte, dtm);
        const float g_cos_theta = ad::adj_dot(g_r_te, dte.d) + ad::adj_dot(g_r_tm, dtm.d);

        const field::float3a previous = bounce > 0
            ? load_leg3(positions, row, bounce - 1) : start;
        const field::float3a hit = load_leg3(positions, row, bounce);
        const field::float3a segment = field::f3_sub(hit, previous);
        const field::float3a outgoing_previous = bounce > 0
            ? rec.frames[bounce - 1].reflected_direction
            : field::safe_normalize(
                  field::f3_sub(load_leg3(positions, row, 0), start), ez);
        const field::float3a raw_normal = load_leg3(normals, row, bounce);
        field::float3a g_incident = field::f3_zero();
        field::float3a g_normal_raw = field::f3_zero();
        ad::adj_reflect_frame(
            field::safe_normalize(segment, outgoing_previous), raw_normal,
            g_s_axis, g_p_in, g_p_out, g_outgoing, g_cos_theta,
            g_incident, g_normal_raw);
        const int64_t normal_base = slot * 3;
        grad_normals[normal_base] = g_normal_raw.x;
        grad_normals[normal_base + 1] = g_normal_raw.y;
        grad_normals[normal_base + 2] = g_normal_raw.z;

        field::float3a g_segment = field::f3_zero();
        field::float3a g_outgoing_previous = field::f3_zero();
        field::adj_safe_normalize(
            segment, outgoing_previous, g_incident, g_segment, g_outgoing_previous);
        const field::float3a g_hit = field::f3_add(g_carry, g_segment);
        g_carry = field::f3_neg(g_segment);
        g_outgoing = g_outgoing_previous;
        if (bounce > 0) {
            const int64_t hit_base = slot * 3;
            grad_positions[hit_base] = g_hit.x;
            grad_positions[hit_base + 1] = g_hit.y;
            grad_positions[hit_base + 2] = g_hit.z;
        } else {
            g_bounce0_hit = g_hit;
            g_incident_pre = g_outgoing;  // cotangent on the pre-incident dir
        }
    }
    return g_chain;
}

struct BasisSeed {
    int slot; int param;  // 0 thickness, 1 eps, 2 sigma
    __device__ ad::LayerSeed operator()(int query) const {
        ad::LayerSeed seed{0.0f, 0.0f, 0.0f};
        if (query == slot) {
            if (param == 0) seed.d_thickness = 1.0f;
            else if (param == 1) seed.d_eps = 1.0f;
            else seed.d_sigma = 1.0f;
        }
        return seed;
    }
};
struct ZeroSeed {
    __device__ ad::LayerSeed operator()(int) const { return {0.0f, 0.0f, 0.0f}; }
};
struct TangentSeed {
    const float* t_thickness; const float* t_eps; const float* t_sigma;
    __device__ ad::LayerSeed operator()(int query) const {
        return {
            t_thickness != nullptr ? t_thickness[query] : 0.0f,
            t_eps != nullptr ? t_eps[query] : 0.0f,
            t_sigma != nullptr ? t_sigma[query] : 0.0f};
    }
};

// ============================ backward =====================================

__global__ void chain_realization_backward_kernel(
    int64_t row_count,
    const float* __restrict__ patch_tris,
    const float* __restrict__ patch_uvs,
    const int64_t* __restrict__ rows,
    const float* __restrict__ d_i,
    const float* __restrict__ d_o,
    const float* __restrict__ n_rows,
    const float* __restrict__ source,
    const float* __restrict__ vertex,
    const float* __restrict__ target,
    const float* __restrict__ c1_positions, const float* __restrict__ c1_normals,
    const float* __restrict__ c1_eps_r, const float* __restrict__ c1_sigma_e,
    const float* __restrict__ c1_mu_r, const float* __restrict__ c1_gain,
    const float* __restrict__ c1_thickness, const int* __restrict__ c1_depth,
    const float* __restrict__ c2_positions, const float* __restrict__ c2_normals,
    const float* __restrict__ c2_eps_r, const float* __restrict__ c2_sigma_e,
    const float* __restrict__ c2_mu_r, const float* __restrict__ c2_gain,
    const float* __restrict__ c2_thickness, const int* __restrict__ c2_depth,
    const float* __restrict__ tx_pol, const float* __restrict__ rx_pol,
    const float* __restrict__ l1_rows, const float* __restrict__ l2_rows,
    const float* __restrict__ sp1_rows, const float* __restrict__ sp2_rows,
    const float* __restrict__ centroids, const float* __restrict__ heights,
    int h_rows_dim, int w_cols_dim,
    const float* __restrict__ cos_spec, const int* __restrict__ material_id,
    const int* __restrict__ layer_offset, const int* __restrict__ layer_count,
    const float* __restrict__ layer_thickness_m, const float* __restrict__ layer_eps_r,
    const float* __restrict__ layer_sigma_e, const float* __restrict__ layer_mu_r,
    const float* __restrict__ quad_a, const float* __restrict__ quad_b,
    const float* __restrict__ quad_w,
    float k0, float frequency_hz,
    const cfloat* __restrict__ grad_total,
    const cfloat* __restrict__ grad_path_field,
    const float* __restrict__ grad_path_gain,
    float* __restrict__ grad_heights,
    float* __restrict__ grad_layer_thickness, float* __restrict__ grad_layer_eps_r,
    float* __restrict__ grad_layer_sigma_e,
    float* __restrict__ grad_c1_eps_r, float* __restrict__ grad_c1_sigma_e,
    float* __restrict__ grad_c1_gain, float* __restrict__ grad_c1_thickness,
    float* __restrict__ grad_c2_eps_r, float* __restrict__ grad_c2_sigma_e,
    float* __restrict__ grad_c2_gain, float* __restrict__ grad_c2_thickness,
    float* __restrict__ grad_d_i, float* __restrict__ grad_d_o,
    float* __restrict__ grad_c1_positions, float* __restrict__ grad_c1_normals,
    float* __restrict__ grad_c2_positions, float* __restrict__ grad_c2_normals,
    float* __restrict__ grad_l1, float* __restrict__ grad_l2,
    float* __restrict__ grad_sp1, float* __restrict__ grad_sp2,
    float* __restrict__ grad_centroids,
    float* __restrict__ grad_k0, float* __restrict__ grad_frequency,
    bool need_heights, bool need_layers, bool need_chain1, bool need_chain2,
    bool need_geometry, bool need_k0, bool need_frequency) {
    __shared__ float sh_I_re[kQuadPoints];
    __shared__ float sh_I_im[kQuadPoints];
    __shared__ float sh_Sp_re[3][kQuadPoints];
    __shared__ float sh_Sp_im[3][kQuadPoints];
    __shared__ float sh_value_re;
    __shared__ float sh_value_im;
    __shared__ float sh_G_re;
    __shared__ float sh_G_im;

    const int row = blockIdx.x;
    if (row >= row_count) return;
    const int64_t patch = rows[row];
    const int t = threadIdx.x;

    const field::float3a p0 = load3f(patch_tris, patch * 3 + 0);
    const field::float3a p1 = load3f(patch_tris, patch * 3 + 1);
    const field::float3a p2 = load3f(patch_tris, patch * 3 + 2);
    const field::float3a e1 = field::f3_sub(p1, p0);
    const field::float3a e2 = field::f3_sub(p2, p0);
    const field::float3a winding = field::f3_cross(e1, e2);
    const float double_area = sqrtf(field::f3_dot(winding, winding));
    const field::float3a n_hat = field::f3_mul(winding, 1.0f / fmaxf(double_area, 1.0e-30f));
    const field::float3a di = load3f(d_i, row);
    const field::float3a dov = load3f(d_o, row);
    const field::float3a kiv = field::f3_mul(di, k0);
    const field::float3a ksv = field::f3_mul(dov, k0);
    const field::float3a q = field::f3_sub(ksv, kiv);
    const field::float3a q_int = field::f3_sub(kiv, ksv);
    const float q_int_n = field::f3_dot(n_hat, q_int);

    const float a = quad_a[t];
    const float b = quad_b[t];
    const float w = quad_w[t];
    const field::float3a pos = field::make_f3(
        p0.x + a * e1.x + b * e2.x, p0.y + a * e1.y + b * e2.y,
        p0.z + a * e1.z + b * e2.z);
    const float u0 = patch_uvs[(patch * 3 + 0) * 2 + 0];
    const float v0 = patch_uvs[(patch * 3 + 0) * 2 + 1];
    const float u1 = patch_uvs[(patch * 3 + 1) * 2 + 0];
    const float v1 = patch_uvs[(patch * 3 + 1) * 2 + 1];
    const float u2 = patch_uvs[(patch * 3 + 2) * 2 + 0];
    const float v2 = patch_uvs[(patch * 3 + 2) * 2 + 1];
    const float uu = u0 + a * (u1 - u0) + b * (u2 - u0);
    const float vv = v0 + a * (v1 - v0) + b * (v2 - v0);
    Texel4 tex;
    const float h = sample_height_tex(heights, h_rows_dim, w_cols_dim, uu, vv, tex);
    const float phase = field::f3_dot(pos, q_int) + q_int_n * h;
    float e_im, e_re;  // exp(-j phase) = (cos phase, -sin phase)
    sincosf(-phase, &e_im, &e_re);
    sh_I_re[t] = e_re * w;
    sh_I_im[t] = e_im * w;
    const field::float3a pvec = field::make_f3(
        pos.x + h * n_hat.x, pos.y + h * n_hat.y, pos.z + h * n_hat.z);
    const float mj_re = e_im * w;
    const float mj_im = -e_re * w;
#pragma unroll
    for (int c = 0; c < 3; ++c) {
        const float pv = (c == 0) ? pvec.x : (c == 1) ? pvec.y : pvec.z;
        sh_Sp_re[c][t] = pv * mj_re;
        sh_Sp_im[c][t] = pv * mj_im;
    }
    __syncthreads();
#pragma unroll
    for (int stride = kQuadPoints / 2; stride > 0; stride >>= 1) {
        if (t < stride) {
            sh_I_re[t] += sh_I_re[t + stride];
            sh_I_im[t] += sh_I_im[t + stride];
#pragma unroll
            for (int c = 0; c < 3; ++c) {
                sh_Sp_re[c][t] += sh_Sp_re[c][t + stride];
                sh_Sp_im[c][t] += sh_Sp_im[c][t + stride];
            }
        }
        __syncthreads();
    }

    if (t == 0) {
        const float A2 = double_area;
        const field::Complex I = field::cplx(sh_I_re[0] * A2, sh_I_im[0] * A2);
        field::Complex S_phase[3];
#pragma unroll
        for (int c = 0; c < 3; ++c)
            S_phase[c] = field::cplx(sh_Sp_re[c][0] * A2, sh_Sp_im[c][0] * A2);

        const field::float3a n = load3f(n_rows, row);
        const field::float3a backup = stable_tangent(n);
        field::float3a s_i, p_i, s_o, p_o;
        sp_basis(n, di, backup, s_i, p_i);
        sp_basis(n, dov, backup, s_o, p_o);
        const field::float3a src = load3f(source, row);
        const field::float3a vtx = load3f(vertex, row);
        const field::float3a tgt = load3f(target, row);
        const int d1 = c1_depth[row];
        const int d2 = c2_depth[row];
        const field::float3a ez = field::make_f3(0.0f, 0.0f, 1.0f);

        // Replay C1, vertex, C2 with recording.
        const field::float3a first_hit1 = d1 > 0 ? load_leg3(c1_positions, row, 0) : vtx;
        const field::float3a incident_pre1 = field::safe_normalize(
            field::f3_sub(first_hit1, src), ez);
        const field::float3a tx_axis = field::project_to_wedge_plane(
            load3f(tx_pol, row), incident_pre1);
        const field::Complex3 e_tx = field::cplx_scale_real(tx_axis, field::cplx(1.0f, 0.0f));
        LegRecord rec1, rec2;
        field::float3a dump1;
        const field::Complex3 e_in = record_leg(
            e_tx, src, vtx, c1_positions, c1_normals, c1_eps_r, c1_sigma_e,
            c1_mu_r, c1_gain, c1_thickness, row, d1, frequency_hz, rec1, dump1);
        const field::Complex e_s_in = transport::complex3_dot_real(e_in, s_i);
        const field::Complex e_p_in = transport::complex3_dot_real(e_in, p_i);
        em::LayerView layers{layer_offset, layer_count, layer_thickness_m,
            layer_eps_r, layer_sigma_e, layer_mu_r, material_id[row]};
        const em::StackRT te = em::stack_rt(cos_spec[row], layers, frequency_hz, em::kPolTE);
        const em::StackRT tm = em::stack_rt(cos_spec[row], layers, frequency_hz, em::kPolTM);
        const field::Complex3 e_out = field::c3_add(
            field::cplx_scale_real(s_o, field::cplx_mul(te.r, e_s_in)),
            field::cplx_scale_real(p_o, field::cplx_mul(tm.r, e_p_in)));
        field::float3a last_dir;
        const field::Complex3 e_rx_field = record_leg(
            e_out, vtx, tgt, c2_positions, c2_normals, c2_eps_r, c2_sigma_e,
            c2_mu_r, c2_gain, c2_thickness, row, d2, frequency_hz, rec2, last_dir);
        const field::float3a rx_axis = field::project_to_wedge_plane(
            load3f(rx_pol, row), last_dir);
        const field::Complex e_rx = transport::complex3_dot_real(e_rx_field, rx_axis);

        // Row coefficient.
        const float q_norm2 = field::f3_dot(q, q);
        const float q_n = fmaxf(field::f3_dot(q, n), 1.0e-9f);
        const float pref = k0 * (q_norm2 / (k0 * q_n)) / (4.0f * kPi);
        const float l1v = l1_rows[row];
        const float l2v = l2_rows[row];
        const float sp1v = sp1_rows[row];
        const float sp2v = sp2_rows[row];
        const float sp = sp1v * sp2v;
        const field::float3a c_row = load3f(centroids, row);
        const float carrier_phase = -(k0 * (l1v + l2v) + field::f3_dot(q, c_row));
        float cc, cs;
        sincosf(carrier_phase, &cs, &cc);
        const field::Complex carrier = field::cplx(cc, cs);
        field::Complex value = field::cplx(0.0f, pref);
        value = field::cplx_mul(value, e_rx);
        value = field::cplx_mul(value, carrier);
        value = field::cplx_mul_real(value, sp);
        const field::Complex row_value = field::cplx_mul(value, I);
        sh_value_re = value.re;
        sh_value_im = value.im;

        // Folded row cotangent G (total broadcast + path_field + path_gain).
        field::Complex G = field::cplx(grad_total[0].real(), grad_total[0].imag());
        if (grad_path_field != nullptr)
            G = field::cplx_add(G, field::cplx(
                grad_path_field[row].real(), grad_path_field[row].imag()));
        if (grad_path_gain != nullptr)
            G = field::cplx_add(G, field::cplx_mul_real(row_value, 2.0f * grad_path_gain[row]));
        sh_G_re = G.re;
        sh_G_im = G.im;

        // ---- Jones scalar cotangent, reverse through the chain. ----
        // M = d row_value / d E_rx = (j pref) carrier sp I.
        field::Complex M = field::cplx(0.0f, pref);
        M = field::cplx_mul(M, carrier);
        M = field::cplx_mul_real(M, sp);
        M = field::cplx_mul(M, I);
        const field::Complex g_e_rx = field::cplx_mul(G, field::cplx_conj(M));

        float g_freq = 0.0f;
        // e_rx = <e_rx_field, rx_axis>.
        field::Complex3 g_e_rx_field = field::c3_zero();
        field::float3a g_rx_axis = field::f3_zero();
        field::adj_cplx_dot_real(e_rx_field, rx_axis, g_e_rx, g_e_rx_field, g_rx_axis);

        // C2 final segment / rx_axis geometry.
        field::float3a g_carry2 = field::f3_zero();
        field::float3a g_outgoing2 = field::f3_zero();
        if (need_geometry) {
            const field::float3a previous_last = d2 > 0
                ? load_leg3(c2_positions, row, d2 - 1) : vtx;
            const field::float3a outgoing_last = d2 > 0
                ? rec2.frames[d2 - 1].reflected_direction
                : field::safe_normalize(field::f3_sub(tgt, vtx), ez);
            const field::float3a final_offset = field::f3_sub(tgt, previous_last);
            field::float3a g_last_dir = field::f3_zero();
            field::float3a g_pol_dump = field::f3_zero();
            ad::adj_transverse_project(last_dir, load3f(rx_pol, row), g_rx_axis,
                g_last_dir, g_pol_dump);
            field::float3a g_final_offset = field::f3_zero();
            field::adj_safe_normalize(final_offset, outgoing_last, g_last_dir,
                g_final_offset, g_outgoing2);
            g_carry2 = field::f3_neg(g_final_offset);  // -> previous_last (c2 last hit)
        }

        // Reverse C2.
        field::float3a g_c2_hit0, g_c2_ipre;
        const field::Complex3 g_e_out = reverse_leg(
            rec2, d2, vtx, c2_positions, c2_normals, c2_eps_r, c2_sigma_e,
            c2_mu_r, c2_gain, c2_thickness, frequency_hz, g_e_rx_field,
            g_carry2, g_outgoing2, row, need_geometry,
            need_chain2 ? grad_c2_eps_r : nullptr,
            need_chain2 ? grad_c2_sigma_e : nullptr,
            need_chain2 ? grad_c2_gain : nullptr,
            need_chain2 ? grad_c2_thickness : nullptr,
            need_geometry ? grad_c2_positions : nullptr,
            need_geometry ? grad_c2_normals : nullptr,
            g_freq, g_c2_hit0, g_c2_ipre);
        if (need_geometry && d2 > 0) {
            // Finalize c2 bounce-0 position with the pre-incident contribution.
            field::float3a g_seg = field::f3_zero();
            field::float3a g_ez = field::f3_zero();
            field::adj_safe_normalize(
                field::f3_sub(load_leg3(c2_positions, row, 0), vtx), ez,
                g_c2_ipre, g_seg, g_ez);
            const field::float3a g_hit0 = field::f3_add(g_c2_hit0, g_seg);
            grad_c2_positions[row * kMaxAdDepth * 3 + 0] = g_hit0.x;
            grad_c2_positions[row * kMaxAdDepth * 3 + 1] = g_hit0.y;
            grad_c2_positions[row * kMaxAdDepth * 3 + 2] = g_hit0.z;
        }

        // ---- Vertex operator adjoint. ----
        field::float3a g_s_o = field::f3_zero();
        field::float3a g_p_o = field::f3_zero();
        field::Complex g_w_te = field::cplx_zero();
        field::Complex g_w_tm = field::cplx_zero();
        field::adj_cplx_scale_real(s_o, field::cplx_mul(te.r, e_s_in), g_e_out, g_s_o, g_w_te);
        field::adj_cplx_scale_real(p_o, field::cplx_mul(tm.r, e_p_in), g_e_out, g_p_o, g_w_tm);
        field::Complex g_te_r = field::cplx_zero();
        field::Complex g_tm_r = field::cplx_zero();
        field::Complex g_e_s_in = field::cplx_zero();
        field::Complex g_e_p_in = field::cplx_zero();
        field::adj_cplx_mul(te.r, e_s_in, g_w_te, g_te_r, g_e_s_in);
        field::adj_cplx_mul(tm.r, e_p_in, g_w_tm, g_tm_r, g_e_p_in);
        // e_s_in = <e_in, s_i>, e_p_in = <e_in, p_i>.
        field::Complex3 g_e_in = field::c3_zero();
        field::float3a g_s_i = field::f3_zero();
        field::float3a g_p_i = field::f3_zero();
        field::adj_cplx_dot_real(e_in, s_i, g_e_s_in, g_e_in, g_s_i);
        field::adj_cplx_dot_real(e_in, p_i, g_e_p_in, g_e_in, g_p_i);

        // Vertex layer stack grads (atomicAdd) + frequency.
        const int material = material_id[row];
        const int first = layer_offset[material];
        const int nlayers = layer_count[material];
        if (need_layers) {
            for (int layer = 0; layer < nlayers; ++layer) {
                const int slot = first + layer;
                for (int param = 0; param < 3; ++param) {
                    float* dst = param == 0 ? grad_layer_thickness
                               : param == 1 ? grad_layer_eps_r : grad_layer_sigma_e;
                    if (dst == nullptr) continue;
                    const BasisSeed seed{slot, param};
                    const ad::DualStackRT dte = ad::stack_rt_dual(
                        cos_spec[row], layers, frequency_hz, 0.0f, 0.0f, em::kPolTE, seed);
                    const ad::DualStackRT dtm = ad::stack_rt_dual(
                        cos_spec[row], layers, frequency_hz, 0.0f, 0.0f, em::kPolTM, seed);
                    atomicAdd(dst + slot,
                        ad::adj_dot(g_te_r, dte.r.d) + ad::adj_dot(g_tm_r, dtm.r.d));
                }
            }
        }
        {
            const ZeroSeed zero_seed;
            const ad::DualStackRT dte = ad::stack_rt_dual(
                cos_spec[row], layers, frequency_hz, 0.0f, 1.0f, em::kPolTE, zero_seed);
            const ad::DualStackRT dtm = ad::stack_rt_dual(
                cos_spec[row], layers, frequency_hz, 0.0f, 1.0f, em::kPolTM, zero_seed);
            g_freq += ad::adj_dot(g_te_r, dte.r.d) + ad::adj_dot(g_tm_r, dtm.r.d);
        }

        // ---- Reverse C1. ----
        field::float3a g_c1_hit0, g_c1_ipre;
        const field::Complex3 g_value0 = reverse_leg(
            rec1, d1, src, c1_positions, c1_normals, c1_eps_r, c1_sigma_e,
            c1_mu_r, c1_gain, c1_thickness, frequency_hz, g_e_in,
            field::f3_zero(), field::f3_zero(), row, need_geometry,
            need_chain1 ? grad_c1_eps_r : nullptr,
            need_chain1 ? grad_c1_sigma_e : nullptr,
            need_chain1 ? grad_c1_gain : nullptr,
            need_chain1 ? grad_c1_thickness : nullptr,
            need_geometry ? grad_c1_positions : nullptr,
            need_geometry ? grad_c1_normals : nullptr,
            g_freq, g_c1_hit0, g_c1_ipre);

        // C1 launch segment: value0 = tx_axis*(1+0j) built on incident_pre1.
        field::float3a g_d_i_basis = field::f3_zero();
        field::float3a g_d_o_basis = field::f3_zero();
        if (need_geometry) {
            g_d_i_basis = adj_sp_basis(n, di, backup, g_s_i, g_p_i);
            g_d_o_basis = adj_sp_basis(n, dov, backup, g_s_o, g_p_o);
            field::float3a g_tx_axis = field::make_f3(
                g_value0.x.re, g_value0.y.re, g_value0.z.re);
            field::float3a g_incident_pre1 = g_c1_ipre;
            field::float3a g_pol_dump = field::f3_zero();
            ad::adj_transverse_project(incident_pre1, load3f(tx_pol, row),
                g_tx_axis, g_incident_pre1, g_pol_dump);
            field::float3a g_seg_pre = field::f3_zero();
            field::float3a g_ez = field::f3_zero();
            field::adj_safe_normalize(
                field::f3_sub(first_hit1, src), ez, g_incident_pre1, g_seg_pre, g_ez);
            if (d1 > 0) {
                const field::float3a g_hit0 = field::f3_add(g_c1_hit0, g_seg_pre);
                grad_c1_positions[row * kMaxAdDepth * 3 + 0] = g_hit0.x;
                grad_c1_positions[row * kMaxAdDepth * 3 + 1] = g_hit0.y;
                grad_c1_positions[row * kMaxAdDepth * 3 + 2] = g_hit0.z;
            }
        }

        // ---- op-2 quadrature adjoint: heights value (stored), geometry q,
        //      L1/L2/sp1/sp2/centroids, k0. ----
        const field::Complex Iv = field::cplx_mul(I, value);
        const field::Complex base_c = field::cplx_mul_real(
            field::cplx_mul(cmulj(e_rx), carrier), sp);          // j*e_rx*carrier*sp
        const field::Complex Ibase = field::cplx_mul(I, base_c);
        const field::Complex jIv = cmulj(Iv);
        const field::Complex value_nosp = field::cplx_mul(
            field::cplx_mul(field::cplx(0.0f, pref), e_rx), carrier);
        const field::Complex base_amp = field::cplx_mul(value_nosp, I);

        if (need_geometry) {
            grad_sp1[row] = redot(G, field::cplx_mul_real(base_amp, sp2v));
            grad_sp2[row] = redot(G, field::cplx_mul_real(base_amp, sp1v));
            const float g_cphase = redot(G, cmulj(row_value));
            grad_l1[row] = g_cphase * (-k0);
            grad_l2[row] = g_cphase * (-k0);
            const field::float3a dprefdq = pref_grad_q(q, n, q_norm2);
            const float gIbase = redot(G, Ibase);
            const float gjIv = redot(G, jIv);
#pragma unroll
            for (int c = 0; c < 3; ++c) {
                const float qc = (c == 0) ? q.x : (c == 1) ? q.y : q.z;
                grad_centroids[row * 3 + c] =
                    redot(G, field::cplx_mul(Iv, field::cplx(0.0f, -qc)));
                const float ccrow = (c == 0) ? c_row.x : (c == 1) ? c_row.y : c_row.z;
                const float dpq = (c == 0) ? dprefdq.x : (c == 1) ? dprefdq.y : dprefdq.z;
                const float gVS = redot(G, field::cplx_mul(value, S_phase[c]));
                const float qterm_i = k0 * gVS - k0 * dpq * gIbase + k0 * ccrow * gjIv;
                const float qterm_o = -k0 * gVS + k0 * dpq * gIbase - k0 * ccrow * gjIv;
                const float bi = (c == 0) ? g_d_i_basis.x : (c == 1) ? g_d_i_basis.y : g_d_i_basis.z;
                const float bo = (c == 0) ? g_d_o_basis.x : (c == 1) ? g_d_o_basis.y : g_d_o_basis.z;
                grad_d_i[row * 3 + c] = qterm_i + bi;
                grad_d_o[row * 3 + c] = qterm_o + bo;
            }
        }

        if (need_k0) {
            const field::float3a Delta = field::f3_sub(dov, di);
            const field::float3a dprefdq = pref_grad_q(q, n, q_norm2);
            field::Complex S_k0 = field::cplx_zero();
#pragma unroll
            for (int c = 0; c < 3; ++c) {
                const float dl = (c == 0) ? Delta.x : (c == 1) ? Delta.y : Delta.z;
                S_k0 = field::cplx_add(S_k0, field::cplx_mul_real(S_phase[c], -dl));
            }
            const float dpref_dk0 = dprefdq.x * Delta.x + dprefdq.y * Delta.y + dprefdq.z * Delta.z;
            const float dcphase_dk0 = -(l1v + l2v) -
                (Delta.x * c_row.x + Delta.y * c_row.y + Delta.z * c_row.z);
            const field::Complex dvalue = field::cplx_add(
                field::cplx_mul_real(base_c, dpref_dk0),
                field::cplx_mul_real(cmulj(value), dcphase_dk0));
            const field::Complex drow = field::cplx_add(
                field::cplx_mul(value, S_k0), field::cplx_mul(I, dvalue));
            atomicAdd(grad_k0, redot(G, drow));
        }
        if (need_frequency)
            atomicAdd(grad_frequency, g_freq);
    }
    __syncthreads();

    if (need_heights) {
        // Row coefficient `value` and the fully-folded cotangent G are staged
        // in shared memory by thread 0; every node scatters its 4-texel VJP.
        // d row_value/d h_t = value * A2 * w_t * (-j q_int_n) * exp(-j phase_t).
        const field::Complex value = field::cplx(sh_value_re, sh_value_im);
        const field::Complex G = field::cplx(sh_G_re, sh_G_im);
        const field::Complex tmp = field::cplx(q_int_n * e_im, -q_int_n * e_re);
        const field::Complex drow = field::cplx_mul(
            value, field::cplx_mul_real(tmp, double_area * w));
        const float gcontrib = redot(G, drow);
#pragma unroll
        for (int k = 0; k < 4; ++k)
            atomicAdd(&grad_heights[tex.idx[k]], gcontrib * tex.wgt[k]);
    }
}

// ============================== jvp ========================================

// Dual transport of one specular leg (mirrors reflection_sequence_jvp).
__device__ void dual_transport_leg(
    field::Complex3 value, field::Complex3 d_value,
    field::float3a start, field::float3a end,
    const float* positions, const float* t_positions,
    const float* normals, const float* t_normals,
    const float* eps_r, const float* sigma_e, const float* mu_r,
    const float* gain, const float* thickness,
    const float* t_eps, const float* t_sigma, const float* t_gain, const float* t_thick,
    float tangent_frequency, int64_t row, int depth, float frequency_hz,
    field::Complex3& out_value, field::Complex3& out_d_value,
    ad::DualF3& out_last_dir) {
    const ad::DualF3 ez = ad::df3_const(field::make_f3(0.0f, 0.0f, 1.0f));
    ad::DualF3 previous = ad::df3_const(start);
    const ad::DualF3 first_hit = depth > 0
        ? load_dual_sequence3f(positions, t_positions, row, 0, kMaxAdDepth)
        : ad::df3_const(end);
    ad::DualF3 outgoing = ad::dual_safe_normalize(ad::df3_sub(first_hit, previous), ez);
    for (int bounce = 0; bounce < depth; ++bounce) {
        const ad::DualF3 hit = load_dual_sequence3f(
            positions, t_positions, row, bounce, kMaxAdDepth);
        const ad::DualF3 segment = ad::df3_sub(hit, previous);
        const ad::DualF3 incident = ad::dual_safe_normalize(segment, outgoing);
        const ad::DualF3 raw_normal = load_dual_sequence3f(
            normals, t_normals, row, bounce, kMaxAdDepth);
        const ad::DualReflectFrame frame = ad::dual_reflect_frame(incident, raw_normal);
        const int64_t slot = row * kMaxAdDepth + bounce;
        ad::DualC r_te, r_tm;
        ad::slab_fresnel_dual(
            frame.cos_theta.v, eps_r[slot], sigma_e[slot], mu_r[slot], gain[slot],
            thickness[slot], frequency_hz, frame.cos_theta.d,
            t_eps != nullptr ? t_eps[slot] : 0.0f,
            t_sigma != nullptr ? t_sigma[slot] : 0.0f,
            t_gain != nullptr ? t_gain[slot] : 0.0f,
            t_thick != nullptr ? t_thick[slot] : 0.0f,
            tangent_frequency, r_te, r_tm);
        const field::Complex e_s = transport::complex3_dot_real(value, frame.s_axis.v);
        const field::Complex e_p = transport::complex3_dot_real(value, frame.p_in.v);
        const field::Complex d_e_s = field::cplx_add(
            transport::complex3_dot_real(d_value, frame.s_axis.v),
            transport::complex3_dot_real(value, frame.s_axis.d));
        const field::Complex d_e_p = field::cplx_add(
            transport::complex3_dot_real(d_value, frame.p_in.v),
            transport::complex3_dot_real(value, frame.p_in.d));
        const field::Complex w_te = field::cplx_mul(r_te.v, e_s);
        const field::Complex w_tm = field::cplx_mul(r_tm.v, e_p);
        const field::Complex d_w_te = field::cplx_add(
            field::cplx_mul(r_te.d, e_s), field::cplx_mul(r_te.v, d_e_s));
        const field::Complex d_w_tm = field::cplx_add(
            field::cplx_mul(r_tm.d, e_p), field::cplx_mul(r_tm.v, d_e_p));
        value = field::c3_add(
            field::cplx_scale_real(frame.s_axis.v, w_te),
            field::cplx_scale_real(frame.p_out.v, w_tm));
        d_value = field::c3_add(
            field::c3_add(
                field::cplx_scale_real(frame.s_axis.d, w_te),
                field::cplx_scale_real(frame.s_axis.v, d_w_te)),
            field::c3_add(
                field::cplx_scale_real(frame.p_out.d, w_tm),
                field::cplx_scale_real(frame.p_out.v, d_w_tm)));
        outgoing = frame.reflected_direction;
        previous = hit;
    }
    out_value = value;
    out_d_value = d_value;
    out_last_dir = ad::dual_safe_normalize(ad::df3_sub(ad::df3_const(end), previous), outgoing);
}

__global__ void chain_realization_jvp_kernel(
    int64_t row_count,
    const float* __restrict__ patch_tris, const float* __restrict__ patch_uvs,
    const int64_t* __restrict__ rows,
    const float* __restrict__ d_i, const float* __restrict__ d_o,
    const float* __restrict__ n_rows,
    const float* __restrict__ source, const float* __restrict__ vertex,
    const float* __restrict__ target,
    const float* __restrict__ c1_positions, const float* __restrict__ c1_normals,
    const float* __restrict__ c1_eps_r, const float* __restrict__ c1_sigma_e,
    const float* __restrict__ c1_mu_r, const float* __restrict__ c1_gain,
    const float* __restrict__ c1_thickness, const int* __restrict__ c1_depth,
    const float* __restrict__ c2_positions, const float* __restrict__ c2_normals,
    const float* __restrict__ c2_eps_r, const float* __restrict__ c2_sigma_e,
    const float* __restrict__ c2_mu_r, const float* __restrict__ c2_gain,
    const float* __restrict__ c2_thickness, const int* __restrict__ c2_depth,
    const float* __restrict__ tx_pol, const float* __restrict__ rx_pol,
    const float* __restrict__ l1_rows, const float* __restrict__ l2_rows,
    const float* __restrict__ sp1_rows, const float* __restrict__ sp2_rows,
    const float* __restrict__ centroids, const float* __restrict__ heights,
    int h_rows_dim, int w_cols_dim,
    const float* __restrict__ cos_spec, const int* __restrict__ material_id,
    const int* __restrict__ layer_offset, const int* __restrict__ layer_count,
    const float* __restrict__ layer_thickness_m, const float* __restrict__ layer_eps_r,
    const float* __restrict__ layer_sigma_e, const float* __restrict__ layer_mu_r,
    const float* __restrict__ quad_a, const float* __restrict__ quad_b,
    const float* __restrict__ quad_w,
    float k0, float frequency_hz,
    const float* __restrict__ t_heights,
    const float* __restrict__ t_layer_thickness, const float* __restrict__ t_layer_eps_r,
    const float* __restrict__ t_layer_sigma_e,
    const float* __restrict__ t_c1_eps_r, const float* __restrict__ t_c1_sigma_e,
    const float* __restrict__ t_c1_gain, const float* __restrict__ t_c1_thickness,
    const float* __restrict__ t_c2_eps_r, const float* __restrict__ t_c2_sigma_e,
    const float* __restrict__ t_c2_gain, const float* __restrict__ t_c2_thickness,
    const float* __restrict__ t_d_i, const float* __restrict__ t_d_o,
    const float* __restrict__ t_c1_positions, const float* __restrict__ t_c1_normals,
    const float* __restrict__ t_c2_positions, const float* __restrict__ t_c2_normals,
    const float* __restrict__ t_l1, const float* __restrict__ t_l2,
    const float* __restrict__ t_sp1, const float* __restrict__ t_sp2,
    const float* __restrict__ t_centroids,
    float t_k0, float t_frequency,
    cfloat* __restrict__ out_t_row_value,
    cfloat* __restrict__ out_t_path_field,
    float* __restrict__ out_t_path_gain) {
    __shared__ float sh_I_re[kQuadPoints];
    __shared__ float sh_I_im[kQuadPoints];
    __shared__ float sh_tI_re[kQuadPoints];
    __shared__ float sh_tI_im[kQuadPoints];

    const int row = blockIdx.x;
    if (row >= row_count) return;
    const int64_t patch = rows[row];
    const int t = threadIdx.x;

    const field::float3a p0 = load3f(patch_tris, patch * 3 + 0);
    const field::float3a p1 = load3f(patch_tris, patch * 3 + 1);
    const field::float3a p2 = load3f(patch_tris, patch * 3 + 2);
    const field::float3a e1 = field::f3_sub(p1, p0);
    const field::float3a e2 = field::f3_sub(p2, p0);
    const field::float3a winding = field::f3_cross(e1, e2);
    const float double_area = sqrtf(field::f3_dot(winding, winding));
    const field::float3a n_hat = field::f3_mul(winding, 1.0f / fmaxf(double_area, 1.0e-30f));
    const field::float3a di = load3f(d_i, row);
    const field::float3a dov = load3f(d_o, row);
    const field::float3a Delta = field::f3_sub(dov, di);
    const field::float3a kiv = field::f3_mul(di, k0);
    const field::float3a ksv = field::f3_mul(dov, k0);
    const field::float3a q = field::f3_sub(ksv, kiv);
    const field::float3a q_int = field::f3_sub(kiv, ksv);
    const float q_int_n = field::f3_dot(n_hat, q_int);

    field::float3a t_di = field::f3_zero();
    field::float3a t_dov = field::f3_zero();
    if (t_d_i != nullptr) t_di = load3f(t_d_i, row);
    if (t_d_o != nullptr) t_dov = load3f(t_d_o, row);
    const field::float3a t_q = field::make_f3(
        t_k0 * Delta.x + k0 * (t_dov.x - t_di.x),
        t_k0 * Delta.y + k0 * (t_dov.y - t_di.y),
        t_k0 * Delta.z + k0 * (t_dov.z - t_di.z));
    const field::float3a t_q_int = field::f3_neg(t_q);

    const float a = quad_a[t];
    const float b = quad_b[t];
    const float w = quad_w[t];
    const field::float3a pos = field::make_f3(
        p0.x + a * e1.x + b * e2.x, p0.y + a * e1.y + b * e2.y,
        p0.z + a * e1.z + b * e2.z);
    const float u0 = patch_uvs[(patch * 3 + 0) * 2 + 0];
    const float v0 = patch_uvs[(patch * 3 + 0) * 2 + 1];
    const float u1 = patch_uvs[(patch * 3 + 1) * 2 + 0];
    const float v1 = patch_uvs[(patch * 3 + 1) * 2 + 1];
    const float u2 = patch_uvs[(patch * 3 + 2) * 2 + 0];
    const float v2 = patch_uvs[(patch * 3 + 2) * 2 + 1];
    const float uu = u0 + a * (u1 - u0) + b * (u2 - u0);
    const float vv = v0 + a * (v1 - v0) + b * (v2 - v0);
    Texel4 tex;
    const float h = sample_height_tex(heights, h_rows_dim, w_cols_dim, uu, vv, tex);
    const float phase = field::f3_dot(pos, q_int) + q_int_n * h;
    float e_im, e_re;
    sincosf(-phase, &e_im, &e_re);
    sh_I_re[t] = e_re * w;
    sh_I_im[t] = e_im * w;
    float t_h = 0.0f;
    if (t_heights != nullptr) {
#pragma unroll
        for (int k = 0; k < 4; ++k) t_h += tex.wgt[k] * t_heights[tex.idx[k]];
    }
    const field::float3a pvec = field::make_f3(
        pos.x + h * n_hat.x, pos.y + h * n_hat.y, pos.z + h * n_hat.z);
    const float t_phase = field::f3_dot(t_q_int, pvec) + q_int_n * t_h;
    sh_tI_re[t] = w * t_phase * e_im;
    sh_tI_im[t] = w * t_phase * (-e_re);
    __syncthreads();
#pragma unroll
    for (int stride = kQuadPoints / 2; stride > 0; stride >>= 1) {
        if (t < stride) {
            sh_I_re[t] += sh_I_re[t + stride];
            sh_I_im[t] += sh_I_im[t + stride];
            sh_tI_re[t] += sh_tI_re[t + stride];
            sh_tI_im[t] += sh_tI_im[t + stride];
        }
        __syncthreads();
    }
    if (t != 0) return;

    const float A2 = double_area;
    const field::Complex I = field::cplx(sh_I_re[0] * A2, sh_I_im[0] * A2);
    const field::Complex t_I = field::cplx(sh_tI_re[0] * A2, sh_tI_im[0] * A2);

    // Dual Jones scalar E_rx.
    const field::float3a n = load3f(n_rows, row);
    const field::float3a backup = stable_tangent(n);
    ad::DualF3 s_i, p_i, s_o, p_o;
    dual_sp_basis(n, ad::df3_make(di, t_di), backup, s_i, p_i);
    dual_sp_basis(n, ad::df3_make(dov, t_dov), backup, s_o, p_o);
    const field::float3a src = load3f(source, row);
    const field::float3a vtx = load3f(vertex, row);
    const field::float3a tgt = load3f(target, row);
    const int d1 = c1_depth[row];
    const int d2 = c2_depth[row];
    const ad::DualF3 ez = ad::df3_const(field::make_f3(0.0f, 0.0f, 1.0f));

    const ad::DualF3 first_hit1 = d1 > 0
        ? load_dual_sequence3f(c1_positions, t_c1_positions, row, 0, kMaxAdDepth)
        : ad::df3_const(vtx);
    const ad::DualF3 incident_pre1 = ad::dual_safe_normalize(
        ad::df3_sub(first_hit1, ad::df3_const(src)), ez);
    const ad::DualF3 tx_axis = ad::dual_transverse_project(
        incident_pre1, ad::df3_const(load3f(tx_pol, row)));
    field::Complex3 value_in = field::cplx_scale_real(tx_axis.v, field::cplx(1.0f, 0.0f));
    field::Complex3 d_value_in = field::cplx_scale_real(tx_axis.d, field::cplx(1.0f, 0.0f));
    field::Complex3 e_in, d_e_in; ad::DualF3 dump;
    dual_transport_leg(
        value_in, d_value_in, src, vtx, c1_positions, t_c1_positions,
        c1_normals, t_c1_normals, c1_eps_r, c1_sigma_e, c1_mu_r, c1_gain,
        c1_thickness, t_c1_eps_r, t_c1_sigma_e, t_c1_gain, t_c1_thickness,
        t_frequency, row, d1, frequency_hz, e_in, d_e_in, dump);

    const field::Complex e_s_in = transport::complex3_dot_real(e_in, s_i.v);
    const field::Complex e_p_in = transport::complex3_dot_real(e_in, p_i.v);
    const field::Complex d_e_s_in = field::cplx_add(
        transport::complex3_dot_real(d_e_in, s_i.v),
        transport::complex3_dot_real(e_in, s_i.d));
    const field::Complex d_e_p_in = field::cplx_add(
        transport::complex3_dot_real(d_e_in, p_i.v),
        transport::complex3_dot_real(e_in, p_i.d));

    em::LayerView layers{layer_offset, layer_count, layer_thickness_m,
        layer_eps_r, layer_sigma_e, layer_mu_r, material_id[row]};
    const TangentSeed layer_seed{t_layer_thickness, t_layer_eps_r, t_layer_sigma_e};
    const ad::DualStackRT te = ad::stack_rt_dual(
        cos_spec[row], layers, frequency_hz, 0.0f, t_frequency, em::kPolTE, layer_seed);
    const ad::DualStackRT tm = ad::stack_rt_dual(
        cos_spec[row], layers, frequency_hz, 0.0f, t_frequency, em::kPolTM, layer_seed);

    // e_out = s_o*(te.r*e_s_in) + p_o*(tm.r*e_p_in).
    const field::Complex w_te = field::cplx_mul(te.r.v, e_s_in);
    const field::Complex w_tm = field::cplx_mul(tm.r.v, e_p_in);
    const field::Complex d_w_te = field::cplx_add(
        field::cplx_mul(te.r.d, e_s_in), field::cplx_mul(te.r.v, d_e_s_in));
    const field::Complex d_w_tm = field::cplx_add(
        field::cplx_mul(tm.r.d, e_p_in), field::cplx_mul(tm.r.v, d_e_p_in));
    const field::Complex3 e_out = field::c3_add(
        field::cplx_scale_real(s_o.v, w_te), field::cplx_scale_real(p_o.v, w_tm));
    const field::Complex3 d_e_out = field::c3_add(
        field::c3_add(field::cplx_scale_real(s_o.d, w_te),
                      field::cplx_scale_real(s_o.v, d_w_te)),
        field::c3_add(field::cplx_scale_real(p_o.d, w_tm),
                      field::cplx_scale_real(p_o.v, d_w_tm)));

    field::Complex3 e_rx_field, d_e_rx_field; ad::DualF3 last_dir;
    dual_transport_leg(
        e_out, d_e_out, vtx, tgt, c2_positions, t_c2_positions,
        c2_normals, t_c2_normals, c2_eps_r, c2_sigma_e, c2_mu_r, c2_gain,
        c2_thickness, t_c2_eps_r, t_c2_sigma_e, t_c2_gain, t_c2_thickness,
        t_frequency, row, d2, frequency_hz, e_rx_field, d_e_rx_field, last_dir);

    const ad::DualF3 rx_axis = ad::dual_transverse_project(
        last_dir, ad::df3_const(load3f(rx_pol, row)));
    const field::Complex e_rx = transport::complex3_dot_real(e_rx_field, rx_axis.v);
    const field::Complex d_e_rx = field::cplx_add(
        transport::complex3_dot_real(d_e_rx_field, rx_axis.v),
        transport::complex3_dot_real(e_rx_field, rx_axis.d));

    // Row coefficient duals.
    const float q_norm2 = field::f3_dot(q, q);
    const float q_n = fmaxf(field::f3_dot(q, n), 1.0e-9f);
    const field::float3a dprefdq = pref_grad_q(q, n, q_norm2);
    const float pref = k0 * (q_norm2 / (k0 * q_n)) / (4.0f * kPi);
    const float t_pref = dprefdq.x * t_q.x + dprefdq.y * t_q.y + dprefdq.z * t_q.z;
    const float l1v = l1_rows[row];
    const float l2v = l2_rows[row];
    const float sp1v = sp1_rows[row];
    const float sp2v = sp2_rows[row];
    const float sp = sp1v * sp2v;
    const float t_l1v = t_l1 != nullptr ? t_l1[row] : 0.0f;
    const float t_l2v = t_l2 != nullptr ? t_l2[row] : 0.0f;
    const float t_sp1v = t_sp1 != nullptr ? t_sp1[row] : 0.0f;
    const float t_sp2v = t_sp2 != nullptr ? t_sp2[row] : 0.0f;
    const float t_sp = t_sp1v * sp2v + sp1v * t_sp2v;
    const field::float3a c_row = load3f(centroids, row);
    field::float3a t_c = field::f3_zero();
    if (t_centroids != nullptr) t_c = load3f(t_centroids, row);
    const float carrier_phase = -(k0 * (l1v + l2v) + field::f3_dot(q, c_row));
    float cc, cs;
    sincosf(carrier_phase, &cs, &cc);
    const field::Complex carrier = field::cplx(cc, cs);
    const float t_carrier_phase = -(t_k0 * (l1v + l2v) + k0 * (t_l1v + t_l2v)
        + (t_q.x * c_row.x + t_q.y * c_row.y + t_q.z * c_row.z)
        + (q.x * t_c.x + q.y * t_c.y + q.z * t_c.z));
    const field::Complex t_carrier = field::cplx_mul_real(cmulj(carrier), t_carrier_phase);

    // value = (j*pref) * e_rx * carrier * sp; product-rule tangent.
    const field::Complex A = field::cplx(0.0f, pref);
    const field::Complex t_A = field::cplx(0.0f, t_pref);
    const field::Complex Ae = field::cplx_mul(A, e_rx);
    const field::Complex t_Ae = field::cplx_add(
        field::cplx_mul(t_A, e_rx), field::cplx_mul(A, d_e_rx));
    const field::Complex Aec = field::cplx_mul(Ae, carrier);
    const field::Complex t_Aec = field::cplx_add(
        field::cplx_mul(t_Ae, carrier), field::cplx_mul(Ae, t_carrier));
    const field::Complex value = field::cplx_mul_real(Aec, sp);
    const field::Complex t_value = field::cplx_add(
        field::cplx_mul_real(t_Aec, sp), field::cplx_mul_real(Aec, t_sp));

    // t_row_value = t_value*I + value*t_I.
    const field::Complex row_value = field::cplx_mul(value, I);
    const field::Complex t_row_value = field::cplx_add(
        field::cplx_mul(t_value, I), field::cplx_mul(value, t_I));
    out_t_row_value[row] = cfloat(t_row_value.re, t_row_value.im);
    out_t_path_field[row] = cfloat(t_row_value.re, t_row_value.im);
    out_t_path_gain[row] = 2.0f * (row_value.re * t_row_value.re + row_value.im * t_row_value.im);
}

__global__ void chain_realization_jvp_total_kernel(
    int64_t row_count,
    const cfloat* __restrict__ row_values,
    cfloat* __restrict__ out_total) {
    __shared__ float sh_re[kReduceBlock];
    __shared__ float sh_im[kReduceBlock];
    const int t = threadIdx.x;
    float acc_re = 0.0f;
    float acc_im = 0.0f;
    for (int64_t index = t; index < row_count; index += kReduceBlock) {
        const cfloat value = row_values[index];
        acc_re += value.real();
        acc_im += value.imag();
    }
    sh_re[t] = acc_re;
    sh_im[t] = acc_im;
    __syncthreads();
#pragma unroll
    for (int stride = kReduceBlock / 2; stride > 0; stride >>= 1) {
        if (t < stride) {
            sh_re[t] += sh_re[t + stride];
            sh_im[t] += sh_im[t + stride];
        }
        __syncthreads();
    }
    if (t == 0) out_total[0] = cfloat(sh_re[0], sh_im[0]);
}

}  // namespace

// Declared in scattering_chain_realization.cu (shared forward validation).
int64_t cn_scattering_chain_realization_check(
    const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&,
    const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&,
    const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&,
    const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&,
    const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&,
    const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&,
    const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&,
    const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&,
    const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&,
    const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&,
    const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&);

namespace {
pybind11::object none_obj() { return pybind11::object(pybind11::none()); }
}  // namespace

pybind11::dict cn_scattering_chain_realization_eval_backward(
    at::Tensor patch_tris, at::Tensor patch_uvs, at::Tensor rows,
    at::Tensor d_i, at::Tensor d_o, at::Tensor n_rows,
    at::Tensor source, at::Tensor vertex, at::Tensor target,
    at::Tensor c1_positions, at::Tensor c1_normals, at::Tensor c1_eps_r,
    at::Tensor c1_sigma_e, at::Tensor c1_mu_r, at::Tensor c1_gain,
    at::Tensor c1_thickness, at::Tensor c1_depth,
    at::Tensor c2_positions, at::Tensor c2_normals, at::Tensor c2_eps_r,
    at::Tensor c2_sigma_e, at::Tensor c2_mu_r, at::Tensor c2_gain,
    at::Tensor c2_thickness, at::Tensor c2_depth,
    at::Tensor tx_pol, at::Tensor rx_pol, at::Tensor l1_rows, at::Tensor l2_rows,
    at::Tensor sp1_rows, at::Tensor sp2_rows, at::Tensor centroids,
    at::Tensor heights, at::Tensor cos_spec, at::Tensor material_id,
    at::Tensor layer_offset, at::Tensor layer_count, at::Tensor layer_thickness_m,
    at::Tensor layer_eps_r, at::Tensor layer_sigma_e, at::Tensor layer_mu_r,
    at::Tensor quad_a, at::Tensor quad_b, at::Tensor quad_w,
    double k0, double frequency_hz,
    at::Tensor grad_total,
    pybind11::object grad_path_field,
    pybind11::object grad_path_gain,
    bool need_grad_heights, bool need_grad_layers, bool need_grad_chain1,
    bool need_grad_chain2, bool need_grad_geometry, bool need_grad_k0,
    bool need_grad_frequency) {
    const int64_t row_count = cn_scattering_chain_realization_check(
        patch_tris, patch_uvs, rows, d_i, d_o, n_rows, source, vertex, target,
        c1_positions, c1_normals, c1_eps_r, c1_sigma_e, c1_mu_r, c1_gain,
        c1_thickness, c1_depth, c2_positions, c2_normals, c2_eps_r, c2_sigma_e,
        c2_mu_r, c2_gain, c2_thickness, c2_depth, tx_pol, rx_pol, l1_rows,
        l2_rows, sp1_rows, sp2_rows, centroids, heights, cos_spec, material_id,
        layer_offset, layer_count, layer_thickness_m, layer_eps_r, layer_sigma_e,
        layer_mu_r, quad_a, quad_b, quad_w);
    using channel_native::check_tensor;
    check_tensor(grad_total, "grad_total", at::kComplexFloat, 0);
    TORCH_CHECK(grad_total.get_device() == patch_tris.get_device(),
                "grad_total must share the primal device");
    grad_total = grad_total.contiguous();
    at::Tensor gpf_storage, gpg_storage;
    const at::Tensor* gpf = optional_grad(
        std::move(grad_path_field), gpf_storage, "grad_path_field",
        at::kComplexFloat, {row_count}, patch_tris);
    const at::Tensor* gpg = optional_grad(
        std::move(grad_path_gain), gpg_storage, "grad_path_gain",
        at::kFloat, {row_count}, patch_tris);

    const int64_t layer_total = layer_thickness_m.size(0);
    auto fopt = patch_tris.options();
    auto zero = [&](bool needed, at::IntArrayRef sizes) {
        return needed ? zero_filled(sizes, fopt) : at::Tensor();
    };
    at::Tensor grad_heights = zero(need_grad_heights, heights.sizes());
    at::Tensor grad_lt = zero(need_grad_layers, {layer_total});
    at::Tensor grad_le = zero(need_grad_layers, {layer_total});
    at::Tensor grad_ls = zero(need_grad_layers, {layer_total});
    at::Tensor grad_c1_eps = zero(need_grad_chain1, c1_eps_r.sizes());
    at::Tensor grad_c1_sig = zero(need_grad_chain1, c1_eps_r.sizes());
    at::Tensor grad_c1_gn = zero(need_grad_chain1, c1_eps_r.sizes());
    at::Tensor grad_c1_th = zero(need_grad_chain1, c1_eps_r.sizes());
    at::Tensor grad_c2_eps = zero(need_grad_chain2, c2_eps_r.sizes());
    at::Tensor grad_c2_sig = zero(need_grad_chain2, c2_eps_r.sizes());
    at::Tensor grad_c2_gn = zero(need_grad_chain2, c2_eps_r.sizes());
    at::Tensor grad_c2_th = zero(need_grad_chain2, c2_eps_r.sizes());
    at::Tensor grad_di = zero(need_grad_geometry, d_i.sizes());
    at::Tensor grad_do = zero(need_grad_geometry, d_o.sizes());
    at::Tensor grad_c1_pos = zero(need_grad_geometry, c1_positions.sizes());
    at::Tensor grad_c1_nrm = zero(need_grad_geometry, c1_normals.sizes());
    at::Tensor grad_c2_pos = zero(need_grad_geometry, c2_positions.sizes());
    at::Tensor grad_c2_nrm = zero(need_grad_geometry, c2_normals.sizes());
    at::Tensor grad_l1 = zero(need_grad_geometry, {row_count});
    at::Tensor grad_l2 = zero(need_grad_geometry, {row_count});
    at::Tensor grad_sp1 = zero(need_grad_geometry, {row_count});
    at::Tensor grad_sp2 = zero(need_grad_geometry, {row_count});
    at::Tensor grad_centroids = zero(need_grad_geometry, centroids.sizes());
    at::Tensor grad_k0 = zero(need_grad_k0, {1});
    at::Tensor grad_frequency = zero(need_grad_frequency, {1});

    const bool any_need = need_grad_heights || need_grad_layers ||
        need_grad_chain1 || need_grad_chain2 || need_grad_geometry ||
        need_grad_k0 || need_grad_frequency;
    if (row_count > 0 && any_need) {
        cudaStream_t stream =
            at::cuda::getCurrentCUDAStream(patch_tris.get_device()).stream();
        auto fp = [](at::Tensor& t) { return t.defined() ? t.data_ptr<float>() : nullptr; };
        chain_realization_backward_kernel<<<static_cast<int>(row_count), kQuadPoints, 0, stream>>>(
            row_count,
            patch_tris.data_ptr<float>(), patch_uvs.data_ptr<float>(),
            rows.data_ptr<int64_t>(), d_i.data_ptr<float>(), d_o.data_ptr<float>(),
            n_rows.data_ptr<float>(), source.data_ptr<float>(),
            vertex.data_ptr<float>(), target.data_ptr<float>(),
            c1_positions.data_ptr<float>(), c1_normals.data_ptr<float>(),
            c1_eps_r.data_ptr<float>(), c1_sigma_e.data_ptr<float>(),
            c1_mu_r.data_ptr<float>(), c1_gain.data_ptr<float>(),
            c1_thickness.data_ptr<float>(), c1_depth.data_ptr<int>(),
            c2_positions.data_ptr<float>(), c2_normals.data_ptr<float>(),
            c2_eps_r.data_ptr<float>(), c2_sigma_e.data_ptr<float>(),
            c2_mu_r.data_ptr<float>(), c2_gain.data_ptr<float>(),
            c2_thickness.data_ptr<float>(), c2_depth.data_ptr<int>(),
            tx_pol.data_ptr<float>(), rx_pol.data_ptr<float>(),
            l1_rows.data_ptr<float>(), l2_rows.data_ptr<float>(),
            sp1_rows.data_ptr<float>(), sp2_rows.data_ptr<float>(),
            centroids.data_ptr<float>(), heights.data_ptr<float>(),
            static_cast<int>(heights.size(0)), static_cast<int>(heights.size(1)),
            cos_spec.data_ptr<float>(), material_id.data_ptr<int>(),
            layer_offset.data_ptr<int>(), layer_count.data_ptr<int>(),
            layer_thickness_m.data_ptr<float>(), layer_eps_r.data_ptr<float>(),
            layer_sigma_e.data_ptr<float>(), layer_mu_r.data_ptr<float>(),
            quad_a.data_ptr<float>(), quad_b.data_ptr<float>(),
            quad_w.data_ptr<float>(),
            static_cast<float>(k0), static_cast<float>(frequency_hz),
            grad_total.data_ptr<cfloat>(),
            gpf ? gpf->data_ptr<cfloat>() : nullptr,
            grad_ptr<float>(gpg),
            fp(grad_heights), fp(grad_lt), fp(grad_le), fp(grad_ls),
            fp(grad_c1_eps), fp(grad_c1_sig), fp(grad_c1_gn), fp(grad_c1_th),
            fp(grad_c2_eps), fp(grad_c2_sig), fp(grad_c2_gn), fp(grad_c2_th),
            fp(grad_di), fp(grad_do), fp(grad_c1_pos), fp(grad_c1_nrm),
            fp(grad_c2_pos), fp(grad_c2_nrm), fp(grad_l1), fp(grad_l2),
            fp(grad_sp1), fp(grad_sp2), fp(grad_centroids),
            fp(grad_k0), fp(grad_frequency),
            need_grad_heights, need_grad_layers, need_grad_chain1,
            need_grad_chain2, need_grad_geometry, need_grad_k0,
            need_grad_frequency);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    auto opt = [](bool needed, at::Tensor& t) {
        return needed ? pybind11::cast(t) : none_obj();
    };
    pybind11::dict out;
    out["grad_heights"] = opt(need_grad_heights, grad_heights);
    out["grad_layer_thickness"] = opt(need_grad_layers, grad_lt);
    out["grad_layer_eps_r"] = opt(need_grad_layers, grad_le);
    out["grad_layer_sigma_e"] = opt(need_grad_layers, grad_ls);
    out["grad_c1_eps_r"] = opt(need_grad_chain1, grad_c1_eps);
    out["grad_c1_sigma_e"] = opt(need_grad_chain1, grad_c1_sig);
    out["grad_c1_gain"] = opt(need_grad_chain1, grad_c1_gn);
    out["grad_c1_thickness"] = opt(need_grad_chain1, grad_c1_th);
    out["grad_c2_eps_r"] = opt(need_grad_chain2, grad_c2_eps);
    out["grad_c2_sigma_e"] = opt(need_grad_chain2, grad_c2_sig);
    out["grad_c2_gain"] = opt(need_grad_chain2, grad_c2_gn);
    out["grad_c2_thickness"] = opt(need_grad_chain2, grad_c2_th);
    out["grad_d_i"] = opt(need_grad_geometry, grad_di);
    out["grad_d_o"] = opt(need_grad_geometry, grad_do);
    out["grad_c1_positions"] = opt(need_grad_geometry, grad_c1_pos);
    out["grad_c1_normals"] = opt(need_grad_geometry, grad_c1_nrm);
    out["grad_c2_positions"] = opt(need_grad_geometry, grad_c2_pos);
    out["grad_c2_normals"] = opt(need_grad_geometry, grad_c2_nrm);
    out["grad_L1"] = opt(need_grad_geometry, grad_l1);
    out["grad_L2"] = opt(need_grad_geometry, grad_l2);
    out["grad_sp1"] = opt(need_grad_geometry, grad_sp1);
    out["grad_sp2"] = opt(need_grad_geometry, grad_sp2);
    out["grad_centroids"] = opt(need_grad_geometry, grad_centroids);
    out["grad_k0"] = opt(need_grad_k0, grad_k0);
    out["grad_frequency"] = opt(need_grad_frequency, grad_frequency);
    return out;
}

pybind11::dict cn_scattering_chain_realization_eval_jvp(
    at::Tensor patch_tris, at::Tensor patch_uvs, at::Tensor rows,
    at::Tensor d_i, at::Tensor d_o, at::Tensor n_rows,
    at::Tensor source, at::Tensor vertex, at::Tensor target,
    at::Tensor c1_positions, at::Tensor c1_normals, at::Tensor c1_eps_r,
    at::Tensor c1_sigma_e, at::Tensor c1_mu_r, at::Tensor c1_gain,
    at::Tensor c1_thickness, at::Tensor c1_depth,
    at::Tensor c2_positions, at::Tensor c2_normals, at::Tensor c2_eps_r,
    at::Tensor c2_sigma_e, at::Tensor c2_mu_r, at::Tensor c2_gain,
    at::Tensor c2_thickness, at::Tensor c2_depth,
    at::Tensor tx_pol, at::Tensor rx_pol, at::Tensor l1_rows, at::Tensor l2_rows,
    at::Tensor sp1_rows, at::Tensor sp2_rows, at::Tensor centroids,
    at::Tensor heights, at::Tensor cos_spec, at::Tensor material_id,
    at::Tensor layer_offset, at::Tensor layer_count, at::Tensor layer_thickness_m,
    at::Tensor layer_eps_r, at::Tensor layer_sigma_e, at::Tensor layer_mu_r,
    at::Tensor quad_a, at::Tensor quad_b, at::Tensor quad_w,
    double k0, double frequency_hz,
    pybind11::object t_heights,
    pybind11::object t_layer_thickness, pybind11::object t_layer_eps_r,
    pybind11::object t_layer_sigma_e,
    pybind11::object t_c1_eps_r, pybind11::object t_c1_sigma_e,
    pybind11::object t_c1_gain, pybind11::object t_c1_thickness,
    pybind11::object t_c2_eps_r, pybind11::object t_c2_sigma_e,
    pybind11::object t_c2_gain, pybind11::object t_c2_thickness,
    pybind11::object t_d_i, pybind11::object t_d_o,
    pybind11::object t_c1_positions, pybind11::object t_c1_normals,
    pybind11::object t_c2_positions, pybind11::object t_c2_normals,
    pybind11::object t_l1, pybind11::object t_l2,
    pybind11::object t_sp1, pybind11::object t_sp2, pybind11::object t_centroids,
    double tangent_k0, double tangent_frequency) {
    const int64_t row_count = cn_scattering_chain_realization_check(
        patch_tris, patch_uvs, rows, d_i, d_o, n_rows, source, vertex, target,
        c1_positions, c1_normals, c1_eps_r, c1_sigma_e, c1_mu_r, c1_gain,
        c1_thickness, c1_depth, c2_positions, c2_normals, c2_eps_r, c2_sigma_e,
        c2_mu_r, c2_gain, c2_thickness, c2_depth, tx_pol, rx_pol, l1_rows,
        l2_rows, sp1_rows, sp2_rows, centroids, heights, cos_spec, material_id,
        layer_offset, layer_count, layer_thickness_m, layer_eps_r, layer_sigma_e,
        layer_mu_r, quad_a, quad_b, quad_w);
    const int64_t layer_total = layer_thickness_m.size(0);
    at::Tensor s[23];
    const at::Tensor* th = optional_grad(std::move(t_heights), s[0], "t_heights", at::kFloat, heights.sizes(), patch_tris);
    const at::Tensor* lt = optional_grad(std::move(t_layer_thickness), s[1], "t_layer_thickness", at::kFloat, {layer_total}, patch_tris);
    const at::Tensor* le = optional_grad(std::move(t_layer_eps_r), s[2], "t_layer_eps_r", at::kFloat, {layer_total}, patch_tris);
    const at::Tensor* ls = optional_grad(std::move(t_layer_sigma_e), s[3], "t_layer_sigma_e", at::kFloat, {layer_total}, patch_tris);
    const at::Tensor* c1e = optional_grad(std::move(t_c1_eps_r), s[4], "t_c1_eps_r", at::kFloat, c1_eps_r.sizes(), patch_tris);
    const at::Tensor* c1s = optional_grad(std::move(t_c1_sigma_e), s[5], "t_c1_sigma_e", at::kFloat, c1_eps_r.sizes(), patch_tris);
    const at::Tensor* c1g = optional_grad(std::move(t_c1_gain), s[6], "t_c1_gain", at::kFloat, c1_eps_r.sizes(), patch_tris);
    const at::Tensor* c1t = optional_grad(std::move(t_c1_thickness), s[7], "t_c1_thickness", at::kFloat, c1_eps_r.sizes(), patch_tris);
    const at::Tensor* c2e = optional_grad(std::move(t_c2_eps_r), s[8], "t_c2_eps_r", at::kFloat, c2_eps_r.sizes(), patch_tris);
    const at::Tensor* c2s = optional_grad(std::move(t_c2_sigma_e), s[9], "t_c2_sigma_e", at::kFloat, c2_eps_r.sizes(), patch_tris);
    const at::Tensor* c2g = optional_grad(std::move(t_c2_gain), s[10], "t_c2_gain", at::kFloat, c2_eps_r.sizes(), patch_tris);
    const at::Tensor* c2t = optional_grad(std::move(t_c2_thickness), s[11], "t_c2_thickness", at::kFloat, c2_eps_r.sizes(), patch_tris);
    const at::Tensor* tdi = optional_grad(std::move(t_d_i), s[12], "t_d_i", at::kFloat, {row_count, 3}, patch_tris);
    const at::Tensor* tdo = optional_grad(std::move(t_d_o), s[13], "t_d_o", at::kFloat, {row_count, 3}, patch_tris);
    const at::Tensor* c1p = optional_grad(std::move(t_c1_positions), s[14], "t_c1_positions", at::kFloat, c1_positions.sizes(), patch_tris);
    const at::Tensor* c1n = optional_grad(std::move(t_c1_normals), s[15], "t_c1_normals", at::kFloat, c1_normals.sizes(), patch_tris);
    const at::Tensor* c2p = optional_grad(std::move(t_c2_positions), s[16], "t_c2_positions", at::kFloat, c2_positions.sizes(), patch_tris);
    const at::Tensor* c2n = optional_grad(std::move(t_c2_normals), s[17], "t_c2_normals", at::kFloat, c2_normals.sizes(), patch_tris);
    const at::Tensor* tl1 = optional_grad(std::move(t_l1), s[18], "t_l1", at::kFloat, {row_count}, patch_tris);
    const at::Tensor* tl2 = optional_grad(std::move(t_l2), s[19], "t_l2", at::kFloat, {row_count}, patch_tris);
    const at::Tensor* ts1 = optional_grad(std::move(t_sp1), s[20], "t_sp1", at::kFloat, {row_count}, patch_tris);
    const at::Tensor* ts2 = optional_grad(std::move(t_sp2), s[21], "t_sp2", at::kFloat, {row_count}, patch_tris);
    const at::Tensor* tc = optional_grad(std::move(t_centroids), s[22], "t_centroids", at::kFloat, {row_count, 3}, patch_tris);

    auto complex_options = patch_tris.options().dtype(at::kComplexFloat);
    auto tangent_total = at::empty({}, complex_options);
    auto t_path_field = at::empty({row_count}, complex_options);
    auto t_path_gain = at::empty({row_count}, patch_tris.options());
    cudaStream_t stream =
        at::cuda::getCurrentCUDAStream(patch_tris.get_device()).stream();
    if (row_count > 0) {
        auto t_row_value = at::empty({row_count}, complex_options);
        chain_realization_jvp_kernel<<<static_cast<int>(row_count), kQuadPoints, 0, stream>>>(
            row_count,
            patch_tris.data_ptr<float>(), patch_uvs.data_ptr<float>(),
            rows.data_ptr<int64_t>(), d_i.data_ptr<float>(), d_o.data_ptr<float>(),
            n_rows.data_ptr<float>(), source.data_ptr<float>(),
            vertex.data_ptr<float>(), target.data_ptr<float>(),
            c1_positions.data_ptr<float>(), c1_normals.data_ptr<float>(),
            c1_eps_r.data_ptr<float>(), c1_sigma_e.data_ptr<float>(),
            c1_mu_r.data_ptr<float>(), c1_gain.data_ptr<float>(),
            c1_thickness.data_ptr<float>(), c1_depth.data_ptr<int>(),
            c2_positions.data_ptr<float>(), c2_normals.data_ptr<float>(),
            c2_eps_r.data_ptr<float>(), c2_sigma_e.data_ptr<float>(),
            c2_mu_r.data_ptr<float>(), c2_gain.data_ptr<float>(),
            c2_thickness.data_ptr<float>(), c2_depth.data_ptr<int>(),
            tx_pol.data_ptr<float>(), rx_pol.data_ptr<float>(),
            l1_rows.data_ptr<float>(), l2_rows.data_ptr<float>(),
            sp1_rows.data_ptr<float>(), sp2_rows.data_ptr<float>(),
            centroids.data_ptr<float>(), heights.data_ptr<float>(),
            static_cast<int>(heights.size(0)), static_cast<int>(heights.size(1)),
            cos_spec.data_ptr<float>(), material_id.data_ptr<int>(),
            layer_offset.data_ptr<int>(), layer_count.data_ptr<int>(),
            layer_thickness_m.data_ptr<float>(), layer_eps_r.data_ptr<float>(),
            layer_sigma_e.data_ptr<float>(), layer_mu_r.data_ptr<float>(),
            quad_a.data_ptr<float>(), quad_b.data_ptr<float>(),
            quad_w.data_ptr<float>(),
            static_cast<float>(k0), static_cast<float>(frequency_hz),
            grad_ptr<float>(th), grad_ptr<float>(lt), grad_ptr<float>(le),
            grad_ptr<float>(ls), grad_ptr<float>(c1e), grad_ptr<float>(c1s),
            grad_ptr<float>(c1g), grad_ptr<float>(c1t), grad_ptr<float>(c2e),
            grad_ptr<float>(c2s), grad_ptr<float>(c2g), grad_ptr<float>(c2t),
            grad_ptr<float>(tdi), grad_ptr<float>(tdo), grad_ptr<float>(c1p),
            grad_ptr<float>(c1n), grad_ptr<float>(c2p), grad_ptr<float>(c2n),
            grad_ptr<float>(tl1), grad_ptr<float>(tl2), grad_ptr<float>(ts1),
            grad_ptr<float>(ts2), grad_ptr<float>(tc),
            static_cast<float>(tangent_k0), static_cast<float>(tangent_frequency),
            t_row_value.data_ptr<cfloat>(), t_path_field.data_ptr<cfloat>(),
            t_path_gain.data_ptr<float>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        chain_realization_jvp_total_kernel<<<1, kReduceBlock, 0, stream>>>(
            row_count, t_row_value.data_ptr<cfloat>(), tangent_total.data_ptr<cfloat>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    } else {
        C10_CUDA_CHECK(cudaMemsetAsync(
            tangent_total.data_ptr(), 0, tangent_total.element_size(), stream));
    }
    pybind11::dict out;
    out["tangent_total"] = tangent_total;
    out["tangent_path_field"] = t_path_field;
    out["tangent_path_gain"] = t_path_gain;
    return out;
}
