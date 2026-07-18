// ADR-014 op 2 companions: VJP/JVP of the realization-coherent phase-screen
// patch integral (forward in scattering_patch_integral.cu).
//
// The forward is output = total = sum_rows row_value, with per row
//   q = k0*(d_o - d_i); q_int = -q; q_int_n = n_hat . q_int
//   phase_t = pos_t . q_int + q_int_n * h_t
//   I = A2 * sum_t w_t exp(-j phase_t)                  ('integral')
//   pref = |q|^2 / (4*pi * max(q . n, 1e-9))
//   jones = r_te*(a_te*g_te) + r_tm*(a_tm*g_tm)
//   carrier = exp(j*cphase), cphase = -(k0*(r1+r2) + q . c)
//   value = (j*pref) * jones * carrier / (r1*r2)
//   row_value = value * I
//
// Both companions recompute the forward intermediates in lockstep with the
// primal expression order (ADR-004). This TU must be compiled with
// --fmad=false, exactly like the forward, so the recomputed primal values
// round identically (the CMake owner adds the matching COMPILE_OPTIONS).
//
// Backward: one block per row, kQuadPoints threads. Every row sees the same
// 0-dim complex cotangent g = grad_total (total is a plain sum). Per-node the
// threads (a) scatter the heights VJP into the 4 bilinear texels (atomicAdd)
// and (b) tree-reduce the phasor sum I and the phase-derivative vector S_phase
// in the forward's fixed order. Thread 0 assembles the per-row geometry/Jones
// grads (direct stores) and atomicAdds the k0 contribution.
//
// JVP: same launch shape; primal and tangent phasors reduce together in the
// forward's fixed order, thread 0 forms t_row_value, then a second single
// block fixed-order stage reduces tangent_total. No atomics in the JVP.

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/util/complex.h>
#include <torch/extension.h>

#include "../tensor_checks.h"

namespace {

constexpr int kQuadPoints = 256;  // 16 x 16 Duffy-mapped Gauss-Legendre nodes
constexpr int kReduceBlock = 256;
constexpr float kPi = 3.14159265358979323846f;

using cfloat = c10::complex<float>;

struct V3 { float x, y, z; };
struct Texel4 { int idx[4]; float wgt[4]; };

__device__ __forceinline__ V3 load3(const float* __restrict__ p, int64_t i) {
    return {p[i * 3 + 0], p[i * 3 + 1], p[i * 3 + 2]};
}
__device__ __forceinline__ float dot3(V3 a, V3 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}
__device__ __forceinline__ V3 cross3(V3 a, V3 b) {
    return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}
__device__ __forceinline__ V3 sub3(V3 a, V3 b) {
    return {a.x - b.x, a.y - b.y, a.z - b.z};
}
__device__ __forceinline__ V3 scale3(V3 a, float s) {
    return {a.x * s, a.y * s, a.z * s};
}

__device__ __forceinline__ cfloat cmul(cfloat a, cfloat b) {
    return cfloat(a.real() * b.real() - a.imag() * b.imag(),
                  a.real() * b.imag() + a.imag() * b.real());
}
__device__ __forceinline__ cfloat cadd(cfloat a, cfloat b) {
    return cfloat(a.real() + b.real(), a.imag() + b.imag());
}
__device__ __forceinline__ cfloat cscalef(cfloat a, float s) {
    return cfloat(a.real() * s, a.imag() * s);
}
__device__ __forceinline__ cfloat cconj(cfloat a) {
    return cfloat(a.real(), -a.imag());
}
__device__ __forceinline__ cfloat cmulj(cfloat a) {  // a * j
    return cfloat(-a.imag(), a.real());
}
__device__ __forceinline__ float reconjmul(cfloat g, cfloat x) {  // Re(conj(g)*x)
    return g.real() * x.real() + g.imag() * x.imag();
}

// PhaseScreenRuntime.sample_height: bilinear with half-texel edge clamp. Also
// returns the 4 texel flat indices and bilinear weights for the heights VJP.
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

// Deterministic unit tangent (_stable_tangent): one-hot at the FIRST smallest
// |component| of n, Gram-Schmidt against n, normalized.
__device__ __forceinline__ V3 stable_tangent(V3 n) {
    const float ax = fabsf(n.x), ay = fabsf(n.y), az = fabsf(n.z);
    V3 axis = {0.0f, 0.0f, 0.0f};
    if (ax <= ay && ax <= az) axis.x = 1.0f;
    else if (ay <= az) axis.y = 1.0f;
    else axis.z = 1.0f;
    const float proj = dot3(axis, n);
    V3 t = {axis.x - proj * n.x, axis.y - proj * n.y, axis.z - proj * n.z};
    const float norm = fmaxf(sqrtf(dot3(t, t)), 1.0e-12f);
    return scale3(t, 1.0f / norm);
}

// _sp_basis plus the analytic gradients of the two projections w.r.t. the
// direction ``d`` (n fixed). s = normalize(n x d) with the deterministic
// backup axis at normal incidence; p = s x d; perp = pol - (pol.d) d;
// val_te = perp.s, val_tm = perp.p. Gradients (ADR-014 op-1 frame formulas):
//   grad(val_te) = -(d.s) pol - (pol.d) s + [unclamped ? (w_vec x n) : 0]
//   grad(val_tm) = -(d.p) pol - (pol.d) p + (perp x s) + [unclamped ? (r_vec x n) : 0]
// with w_vec = (perp - (perp.s) s)/|u|, r_vec = ((d x perp) - ((d x perp).s) s)/|u|.
__device__ __forceinline__ void basis_and_grads(
    V3 n, V3 d, V3 backup, V3 pol,
    V3& s, V3& p, float& val_te, float& val_tm, V3& grad_te, V3& grad_tm) {
    const V3 raw = cross3(n, d);
    const float norm = sqrtf(dot3(raw, raw));
    const bool unclamped = norm >= 1.0e-6f;
    if (unclamped) s = scale3(raw, 1.0f / fmaxf(norm, 1.0e-12f));
    else s = backup;
    p = cross3(s, d);
    const float pol_d = dot3(pol, d);
    const V3 perp = {pol.x - pol_d * d.x, pol.y - pol_d * d.y, pol.z - pol_d * d.z};
    val_te = dot3(perp, s);
    val_tm = dot3(perp, p);
    const float d_s = dot3(d, s);
    const float d_p = dot3(d, p);
    grad_te = {-d_s * pol.x - pol_d * s.x,
               -d_s * pol.y - pol_d * s.y,
               -d_s * pol.z - pol_d * s.z};
    const V3 perp_x_s = cross3(perp, s);
    grad_tm = {-d_p * pol.x - pol_d * p.x + perp_x_s.x,
               -d_p * pol.y - pol_d * p.y + perp_x_s.y,
               -d_p * pol.z - pol_d * p.z + perp_x_s.z};
    if (unclamped) {
        const float inv = 1.0f / norm;
        const float perp_s = dot3(perp, s);
        const V3 w_vec = {(perp.x - perp_s * s.x) * inv,
                          (perp.y - perp_s * s.y) * inv,
                          (perp.z - perp_s * s.z) * inv};
        const V3 wxn = cross3(w_vec, n);
        grad_te.x += wxn.x; grad_te.y += wxn.y; grad_te.z += wxn.z;
        const V3 dxperp = cross3(d, perp);
        const float dxperp_s = dot3(dxperp, s);
        const V3 r_vec = {(dxperp.x - dxperp_s * s.x) * inv,
                          (dxperp.y - dxperp_s * s.y) * inv,
                          (dxperp.z - dxperp_s * s.z) * inv};
        const V3 rxn = cross3(r_vec, n);
        grad_tm.x += rxn.x; grad_tm.y += rxn.y; grad_tm.z += rxn.z;
    }
}

// Per-row triangle frame + q vectors, shared by both companions.
struct RowFrame {
    V3 p0, e1, e2, n_hat;
    float double_area;
    V3 q, q_int;
    float q_int_n;
};
__device__ __forceinline__ RowFrame load_frame(
    const float* __restrict__ patch_tris, int64_t patch,
    const float* __restrict__ d_i, const float* __restrict__ d_o,
    int row, float k0) {
    RowFrame f;
    f.p0 = load3(patch_tris, patch * 3 + 0);
    const V3 p1 = load3(patch_tris, patch * 3 + 1);
    const V3 p2 = load3(patch_tris, patch * 3 + 2);
    f.e1 = sub3(p1, f.p0);
    f.e2 = sub3(p2, f.p0);
    const V3 winding = cross3(f.e1, f.e2);
    f.double_area = sqrtf(dot3(winding, winding));
    f.n_hat = scale3(winding, 1.0f / fmaxf(f.double_area, 1.0e-30f));
    const V3 di = load3(d_i, row);
    const V3 dov = load3(d_o, row);
    const V3 kiv = {di.x * k0, di.y * k0, di.z * k0};
    const V3 ksv = {dov.x * k0, dov.y * k0, dov.z * k0};
    f.q = sub3(ksv, kiv);
    f.q_int = sub3(kiv, ksv);
    f.q_int_n = dot3(f.n_hat, f.q_int);
    return f;
}

// Interpolated (u, v) of a Duffy node inside the patch triangle.
__device__ __forceinline__ void node_uv(
    const float* __restrict__ patch_uvs, int64_t patch, float a, float b,
    float& uu, float& vv) {
    const float u0 = patch_uvs[(patch * 3 + 0) * 2 + 0];
    const float v0 = patch_uvs[(patch * 3 + 0) * 2 + 1];
    const float u1 = patch_uvs[(patch * 3 + 1) * 2 + 0];
    const float v1 = patch_uvs[(patch * 3 + 1) * 2 + 1];
    const float u2 = patch_uvs[(patch * 3 + 2) * 2 + 0];
    const float v2 = patch_uvs[(patch * 3 + 2) * 2 + 1];
    uu = u0 + a * (u1 - u0) + b * (u2 - u0);
    vv = v0 + a * (v1 - v0) + b * (v2 - v0);
}

// prefactor gradient d pref/d q (real 3-vector); pref matches the forward.
__device__ __forceinline__ V3 pref_grad_q(V3 q, V3 n, float q_norm2) {
    const float qn = dot3(q, n);
    const float qn_c = fmaxf(qn, 1.0e-9f);
    const float flag = qn > 1.0e-9f ? 1.0f : 0.0f;
    const float inv = 1.0f / (4.0f * kPi * qn_c * qn_c);
    return {(2.0f * q.x * qn_c - q_norm2 * flag * n.x) * inv,
            (2.0f * q.y * qn_c - q_norm2 * flag * n.y) * inv,
            (2.0f * q.z * qn_c - q_norm2 * flag * n.z) * inv};
}

// Assemble the forward row coefficient ``value`` and its complex building
// blocks (all in the forward expression order). base_c = value/pref,
// coeff2 = value/jones, both division-free.
struct RowCoef {
    cfloat value, base_c, coeff2, jones, carrier;
    float pref, inv_rr, cc, cs;
    V3 s_i, p_i, s_o, p_o;
    float a_te, a_tm, g_te, g_tm;
    V3 grad_ate, grad_atm, grad_gte, grad_gtm;
    cfloat te, tm;
};
__device__ __forceinline__ RowCoef assemble_coef(
    const RowFrame& f, V3 n, V3 di, V3 dov, cfloat te, cfloat tm,
    V3 pt, V3 pr, V3 c_row, float r1v, float r2v, float k0) {
    RowCoef rc;
    rc.te = te; rc.tm = tm;
    const float q_norm2 = dot3(f.q, f.q);
    const float q_n = fmaxf(dot3(f.q, n), 1.0e-9f);
    // Forward-order prefactor (purely imaginary weight 1j*pref).
    rc.pref = k0 * (q_norm2 / (k0 * q_n)) / (4.0f * kPi);

    const V3 backup = stable_tangent(n);
    basis_and_grads(n, di, backup, pt, rc.s_i, rc.p_i, rc.a_te, rc.a_tm,
                    rc.grad_ate, rc.grad_atm);
    basis_and_grads(n, dov, backup, pr, rc.s_o, rc.p_o, rc.g_te, rc.g_tm,
                    rc.grad_gte, rc.grad_gtm);
    rc.jones = cfloat(
        te.real() * (rc.a_te * rc.g_te) + tm.real() * (rc.a_tm * rc.g_tm),
        te.imag() * (rc.a_te * rc.g_te) + tm.imag() * (rc.a_tm * rc.g_tm));

    const float carrier_phase = -(k0 * (r1v + r2v) + dot3(f.q, c_row));
    sincosf(carrier_phase, &rc.cs, &rc.cc);
    rc.carrier = cfloat(rc.cc, rc.cs);
    rc.inv_rr = 1.0f / (r1v * r2v);

    // value = (j*pref)*jones*carrier/(r1*r2) in forward order.
    cfloat v = cfloat(-rc.pref * rc.jones.imag(), rc.pref * rc.jones.real());
    v = cmul(v, rc.carrier);
    rc.value = cscalef(v, rc.inv_rr);
    // base_c = value/pref = (j*1)*jones*carrier/(r1*r2).
    cfloat b = cfloat(-rc.jones.imag(), rc.jones.real());
    b = cmul(b, rc.carrier);
    rc.base_c = cscalef(b, rc.inv_rr);
    // coeff2 = value/jones = (j*pref)*carrier/(r1*r2).
    cfloat c2 = cfloat(0.0f, rc.pref);
    c2 = cmul(c2, rc.carrier);
    rc.coeff2 = cscalef(c2, rc.inv_rr);
    return rc;
}

// ----------------------------- backward -----------------------------------

__global__ void patch_integral_backward_kernel(
    int64_t row_count,
    const float* __restrict__ patch_tris,
    const float* __restrict__ patch_uvs,
    const int64_t* __restrict__ rows,
    const float* __restrict__ d_i,
    const float* __restrict__ d_o,
    const float* __restrict__ n_rows,
    const cfloat* __restrict__ r_te,
    const cfloat* __restrict__ r_tm,
    const float* __restrict__ pol_t,
    const float* __restrict__ pol_r,
    const float* __restrict__ r1_rows,
    const float* __restrict__ r2_rows,
    const float* __restrict__ centroids,
    const float* __restrict__ heights,
    int h_rows_dim, int w_cols_dim,
    const float* __restrict__ quad_a,
    const float* __restrict__ quad_b,
    const float* __restrict__ quad_w,
    float k0,
    const cfloat* __restrict__ grad_total,
    float* __restrict__ grad_heights,
    cfloat* __restrict__ grad_r_te,
    cfloat* __restrict__ grad_r_tm,
    float* __restrict__ grad_d_i,
    float* __restrict__ grad_d_o,
    float* __restrict__ grad_r1,
    float* __restrict__ grad_r2,
    float* __restrict__ grad_centroids,
    float* __restrict__ grad_k0,
    bool need_heights, bool need_jones, bool need_geometry, bool need_k0) {
    __shared__ float sh_I_re[kQuadPoints];
    __shared__ float sh_I_im[kQuadPoints];
    __shared__ float sh_Sp_re[3][kQuadPoints];
    __shared__ float sh_Sp_im[3][kQuadPoints];
    __shared__ float sh_value_re;
    __shared__ float sh_value_im;

    const int row = blockIdx.x;
    if (row >= row_count) return;
    const int64_t patch = rows[row];
    const int t = threadIdx.x;

    const RowFrame f = load_frame(patch_tris, patch, d_i, d_o, row, k0);
    const cfloat g = grad_total[0];

    // Node phasor and its phase-derivative contribution.
    const float a = quad_a[t];
    const float b = quad_b[t];
    const float w = quad_w[t];
    const V3 pos = {f.p0.x + a * f.e1.x + b * f.e2.x,
                    f.p0.y + a * f.e1.y + b * f.e2.y,
                    f.p0.z + a * f.e1.z + b * f.e2.z};
    float uu, vv;
    node_uv(patch_uvs, patch, a, b, uu, vv);
    Texel4 tex;
    const float h = sample_height_tex(heights, h_rows_dim, w_cols_dim, uu, vv, tex);
    const float phase = dot3(pos, f.q_int) + f.q_int_n * h;
    float e_im, e_re;  // exp(-j phase) = (cos phase, -sin phase)
    sincosf(-phase, &e_im, &e_re);
    sh_I_re[t] = e_re * w;
    sh_I_im[t] = e_im * w;
    // S_phase node term: w_t*(-j)*(pos + h*n_hat)*exp(-j phase).
    const V3 pvec = {pos.x + h * f.n_hat.x, pos.y + h * f.n_hat.y, pos.z + h * f.n_hat.z};
    // (-j)*(e_re + j e_im) = (e_im, -e_re)
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
        const float A2 = f.double_area;
        const cfloat I = cfloat(sh_I_re[0] * A2, sh_I_im[0] * A2);
        cfloat S_phase[3];
#pragma unroll
        for (int c = 0; c < 3; ++c)
            S_phase[c] = cfloat(sh_Sp_re[c][0] * A2, sh_Sp_im[c][0] * A2);

        const V3 n = load3(n_rows, row);
        const V3 di = load3(d_i, row);
        const V3 dov = load3(d_o, row);
        const V3 pt = {pol_t[0], pol_t[1], pol_t[2]};
        const V3 pr = {pol_r[0], pol_r[1], pol_r[2]};
        const V3 c_row = load3(centroids, row);
        const float r1v = r1_rows[row];
        const float r2v = r2_rows[row];
        const RowCoef rc = assemble_coef(
            f, n, di, dov, r_te[row], r_tm[row], pt, pr, c_row, r1v, r2v, k0);

        sh_value_re = rc.value.real();
        sh_value_im = rc.value.imag();

        const cfloat Iv = cmul(I, rc.value);            // I * value
        const cfloat jIv = cmulj(Iv);                   // j * (I*value)
        const cfloat Ibase = cmul(I, rc.base_c);        // I * base_c
        const cfloat coeff2I = cmul(rc.coeff2, I);      // coeff2 * I

        if (need_jones) {
            // grad_z = g * conj(K), K = coeff2*(proj)*I.
            const cfloat K_te = cscalef(coeff2I, rc.a_te * rc.g_te);
            const cfloat K_tm = cscalef(coeff2I, rc.a_tm * rc.g_tm);
            grad_r_te[row] = cmul(g, cconj(K_te));
            grad_r_tm[row] = cmul(g, cconj(K_tm));
        }

        if (need_geometry) {
            // r1, r2 : d row_value/d r = I*value*(-j*k0 - 1/r).
            grad_r1[row] = reconjmul(g, cmul(Iv, cfloat(-1.0f / r1v, -k0)));
            grad_r2[row] = reconjmul(g, cmul(Iv, cfloat(-1.0f / r2v, -k0)));
            // centroids : d row_value/d c_j = I*value*(-j*q_j).
#pragma unroll
            for (int c = 0; c < 3; ++c) {
                const float qc = (c == 0) ? f.q.x : (c == 1) ? f.q.y : f.q.z;
                grad_centroids[row * 3 + c] = reconjmul(g, cmul(Iv, cfloat(0.0f, -qc)));
            }
            // d_i / d_o : phase + prefactor + carrier + Jones chains.
            const V3 dprefdq = pref_grad_q(f.q, n, dot3(f.q, f.q));
            const float gIbase = reconjmul(g, Ibase);
            const float gjIv = reconjmul(g, jIv);
            const float w_ate = reconjmul(g, cscalef(cmul(coeff2I, rc.te), rc.g_te));
            const float w_atm = reconjmul(g, cscalef(cmul(coeff2I, rc.tm), rc.g_tm));
            const float w_gte = reconjmul(g, cscalef(cmul(coeff2I, rc.te), rc.a_te));
            const float w_gtm = reconjmul(g, cscalef(cmul(coeff2I, rc.tm), rc.a_tm));
#pragma unroll
            for (int c = 0; c < 3; ++c) {
                const float cc = (c == 0) ? c_row.x : (c == 1) ? c_row.y : c_row.z;
                const float dpq = (c == 0) ? dprefdq.x : (c == 1) ? dprefdq.y : dprefdq.z;
                const float gVS = reconjmul(g, cmul(rc.value, S_phase[c]));
                const float ate = (c == 0) ? rc.grad_ate.x : (c == 1) ? rc.grad_ate.y : rc.grad_ate.z;
                const float atm = (c == 0) ? rc.grad_atm.x : (c == 1) ? rc.grad_atm.y : rc.grad_atm.z;
                const float gte = (c == 0) ? rc.grad_gte.x : (c == 1) ? rc.grad_gte.y : rc.grad_gte.z;
                const float gtm = (c == 0) ? rc.grad_gtm.x : (c == 1) ? rc.grad_gtm.y : rc.grad_gtm.z;
                grad_d_i[row * 3 + c] =
                    k0 * gVS - k0 * dpq * gIbase + k0 * cc * gjIv + w_ate * ate + w_atm * atm;
                grad_d_o[row * 3 + c] =
                    -k0 * gVS + k0 * dpq * gIbase - k0 * cc * gjIv + w_gte * gte + w_gtm * gtm;
            }
        }

        if (need_k0) {
            const V3 Delta = sub3(dov, di);
            const V3 dprefdq = pref_grad_q(f.q, n, dot3(f.q, f.q));
            // dI/dk0 = -dot(Delta, S_phase).
            cfloat S_k0 = cfloat(0.0f, 0.0f);
#pragma unroll
            for (int c = 0; c < 3; ++c) {
                const float dl = (c == 0) ? Delta.x : (c == 1) ? Delta.y : Delta.z;
                S_k0 = cadd(S_k0, cscalef(S_phase[c], -dl));
            }
            const float dpref_dk0 = dprefdq.x * Delta.x + dprefdq.y * Delta.y + dprefdq.z * Delta.z;
            const float dcphase_dk0 = -(r1v + r2v) - (Delta.x * c_row.x + Delta.y * c_row.y + Delta.z * c_row.z);
            // d value/d k0 = base_c*dpref/dk0 + value*(j*dcphase/dk0).
            const cfloat dvalue = cadd(cscalef(rc.base_c, dpref_dk0),
                                       cscalef(cmulj(rc.value), dcphase_dk0));
            const cfloat drow = cadd(cmul(rc.value, S_k0), cmul(I, dvalue));
            atomicAdd(grad_k0, reconjmul(g, drow));
        }
    }
    __syncthreads();

    if (need_heights) {
        const cfloat value = cfloat(sh_value_re, sh_value_im);
        // d row_value/d h_t = value * A2 * w_t * (-j q_int_n) * exp(-j phase_t).
        // (-j q_int_n)*(e_re + j e_im) = (q_int_n*e_im, -q_int_n*e_re)
        const cfloat tmp = cfloat(f.q_int_n * e_im, -f.q_int_n * e_re);
        const cfloat drow = cmul(value, cscalef(tmp, f.double_area * w));
        const float gcontrib = reconjmul(g, drow);
#pragma unroll
        for (int k = 0; k < 4; ++k)
            atomicAdd(&grad_heights[tex.idx[k]], gcontrib * tex.wgt[k]);
    }
}

// ------------------------------- jvp --------------------------------------

__global__ void patch_integral_jvp_kernel(
    int64_t row_count,
    const float* __restrict__ patch_tris,
    const float* __restrict__ patch_uvs,
    const int64_t* __restrict__ rows,
    const float* __restrict__ d_i,
    const float* __restrict__ d_o,
    const float* __restrict__ n_rows,
    const cfloat* __restrict__ r_te,
    const cfloat* __restrict__ r_tm,
    const float* __restrict__ pol_t,
    const float* __restrict__ pol_r,
    const float* __restrict__ r1_rows,
    const float* __restrict__ r2_rows,
    const float* __restrict__ centroids,
    const float* __restrict__ heights,
    int h_rows_dim, int w_cols_dim,
    const float* __restrict__ quad_a,
    const float* __restrict__ quad_b,
    const float* __restrict__ quad_w,
    float k0,
    const float* __restrict__ t_heights,
    const cfloat* __restrict__ t_r_te,
    const cfloat* __restrict__ t_r_tm,
    const float* __restrict__ t_d_i,
    const float* __restrict__ t_d_o,
    const float* __restrict__ t_r1,
    const float* __restrict__ t_r2,
    const float* __restrict__ t_centroids,
    float t_k0,
    cfloat* __restrict__ out_t_row_value) {
    __shared__ float sh_I_re[kQuadPoints];
    __shared__ float sh_I_im[kQuadPoints];
    __shared__ float sh_tI_re[kQuadPoints];
    __shared__ float sh_tI_im[kQuadPoints];

    const int row = blockIdx.x;
    if (row >= row_count) return;
    const int64_t patch = rows[row];
    const int t = threadIdx.x;

    const RowFrame f = load_frame(patch_tris, patch, d_i, d_o, row, k0);
    const V3 di = load3(d_i, row);
    const V3 dov = load3(d_o, row);
    const V3 Delta = sub3(dov, di);
    // t_q = t_k0*Delta + k0*(t_dov - t_di); t_q_int = -t_q.
    V3 t_di = {0.0f, 0.0f, 0.0f}, t_dov = {0.0f, 0.0f, 0.0f};
    if (t_d_i != nullptr) t_di = load3(t_d_i, row);
    if (t_d_o != nullptr) t_dov = load3(t_d_o, row);
    const V3 t_q = {t_k0 * Delta.x + k0 * (t_dov.x - t_di.x),
                    t_k0 * Delta.y + k0 * (t_dov.y - t_di.y),
                    t_k0 * Delta.z + k0 * (t_dov.z - t_di.z)};
    const V3 t_q_int = {-t_q.x, -t_q.y, -t_q.z};

    const float a = quad_a[t];
    const float b = quad_b[t];
    const float w = quad_w[t];
    const V3 pos = {f.p0.x + a * f.e1.x + b * f.e2.x,
                    f.p0.y + a * f.e1.y + b * f.e2.y,
                    f.p0.z + a * f.e1.z + b * f.e2.z};
    float uu, vv;
    node_uv(patch_uvs, patch, a, b, uu, vv);
    Texel4 tex;
    const float h = sample_height_tex(heights, h_rows_dim, w_cols_dim, uu, vv, tex);
    const float phase = dot3(pos, f.q_int) + f.q_int_n * h;
    float e_im, e_re;  // exp(-j phase) = (cos phase, -sin phase)
    sincosf(-phase, &e_im, &e_re);
    sh_I_re[t] = e_re * w;
    sh_I_im[t] = e_im * w;

    // t_phase = t_q_int.(pos + h*n_hat) + q_int_n * t_h.
    float t_h = 0.0f;
    if (t_heights != nullptr) {
#pragma unroll
        for (int k = 0; k < 4; ++k) t_h += tex.wgt[k] * t_heights[tex.idx[k]];
    }
    const V3 pvec = {pos.x + h * f.n_hat.x, pos.y + h * f.n_hat.y, pos.z + h * f.n_hat.z};
    const float t_phase = dot3(t_q_int, pvec) + f.q_int_n * t_h;
    // node tangent phasor = w_t*(-j t_phase)*exp(-j phase) = w_t*t_phase*(e_im, -e_re).
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

    const float A2 = f.double_area;
    const cfloat I = cfloat(sh_I_re[0] * A2, sh_I_im[0] * A2);
    const cfloat t_I = cfloat(sh_tI_re[0] * A2, sh_tI_im[0] * A2);

    const V3 n = load3(n_rows, row);
    const V3 pt = {pol_t[0], pol_t[1], pol_t[2]};
    const V3 pr = {pol_r[0], pol_r[1], pol_r[2]};
    const V3 c_row = load3(centroids, row);
    const float r1v = r1_rows[row];
    const float r2v = r2_rows[row];
    const cfloat te = r_te[row];
    const cfloat tm = r_tm[row];
    const RowCoef rc = assemble_coef(f, n, di, dov, te, tm, pt, pr, c_row, r1v, r2v, k0);

    // Tangent of value via product rule on value = A*B*C*D with
    // A = j*pref, B = jones, C = carrier, D = inv_rr.
    const V3 dprefdq = pref_grad_q(f.q, n, dot3(f.q, f.q));
    const float t_pref = dprefdq.x * t_q.x + dprefdq.y * t_q.y + dprefdq.z * t_q.z;

    const float t_a_te = dot3(rc.grad_ate, t_di);
    const float t_a_tm = dot3(rc.grad_atm, t_di);
    const float t_g_te = dot3(rc.grad_gte, t_dov);
    const float t_g_tm = dot3(rc.grad_gtm, t_dov);
    cfloat t_r_te_v = cfloat(0.0f, 0.0f), t_r_tm_v = cfloat(0.0f, 0.0f);
    if (t_r_te != nullptr) t_r_te_v = t_r_te[row];
    if (t_r_tm != nullptr) t_r_tm_v = t_r_tm[row];
    // t_jones = t_r_te*(a_te g_te) + te*(t_a_te g_te + a_te t_g_te) + (tm terms).
    const float d_pte = t_a_te * rc.g_te + rc.a_te * t_g_te;
    const float d_ptm = t_a_tm * rc.g_tm + rc.a_tm * t_g_tm;
    cfloat t_jones = cadd(cscalef(t_r_te_v, rc.a_te * rc.g_te), cscalef(rc.te, d_pte));
    t_jones = cadd(t_jones, cadd(cscalef(t_r_tm_v, rc.a_tm * rc.g_tm), cscalef(rc.tm, d_ptm)));

    // t_cphase = -(t_k0*(r1+r2) + k0*(t_r1+t_r2) + t_q.c + q.t_c).
    float t_r1_v = 0.0f, t_r2_v = 0.0f;
    if (t_r1 != nullptr) t_r1_v = t_r1[row];
    if (t_r2 != nullptr) t_r2_v = t_r2[row];
    V3 t_c = {0.0f, 0.0f, 0.0f};
    if (t_centroids != nullptr) t_c = load3(t_centroids, row);
    const float t_q_dot_c = t_q.x * c_row.x + t_q.y * c_row.y + t_q.z * c_row.z;
    const float q_dot_tc = f.q.x * t_c.x + f.q.y * t_c.y + f.q.z * t_c.z;
    const float t_cphase = -(t_k0 * (r1v + r2v) + k0 * (t_r1_v + t_r2_v) + t_q_dot_c + q_dot_tc);
    // t_inv_rr = -inv_rr*(t_r1/r1 + t_r2/r2).
    const float t_inv_rr = -rc.inv_rr * (t_r1_v / r1v + t_r2_v / r2v);

    const cfloat A = cfloat(0.0f, rc.pref);
    const cfloat t_A = cfloat(0.0f, t_pref);
    const cfloat B = rc.jones;
    const cfloat C = rc.carrier;
    const cfloat t_C = cscalef(cmulj(rc.carrier), t_cphase);  // j*carrier*t_cphase
    const float D = rc.inv_rr;
    const cfloat AB = cmul(A, B);
    const cfloat ABC = cmul(AB, C);
    cfloat t_value = cscalef(cmul(cmul(t_A, B), C), D);
    t_value = cadd(t_value, cscalef(cmul(cmul(A, t_jones), C), D));
    t_value = cadd(t_value, cscalef(cmul(AB, t_C), D));
    t_value = cadd(t_value, cscalef(ABC, t_inv_rr));

    // t_row_value = t_value*I + value*t_I.
    out_t_row_value[row] = cadd(cmul(t_value, I), cmul(rc.value, t_I));
}

__global__ void patch_integral_jvp_total_kernel(
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

// --------------------------- validation -----------------------------------

int64_t check_patch_inputs(
    const at::Tensor& patch_tris, const at::Tensor& patch_uvs,
    const at::Tensor& rows, const at::Tensor& d_i, const at::Tensor& d_o,
    const at::Tensor& n_rows, const at::Tensor& r_te, const at::Tensor& r_tm,
    const at::Tensor& pol_t, const at::Tensor& pol_r, const at::Tensor& r1_rows,
    const at::Tensor& r2_rows, const at::Tensor& centroids,
    const at::Tensor& heights, const at::Tensor& quad_a, const at::Tensor& quad_b,
    const at::Tensor& quad_w) {
    using channel_native::check_tensor;
    using channel_native::check_flat_tensor;
    using channel_native::check_vec3_table;
    check_tensor(patch_tris, "patch_tris", at::kFloat, 3);
    TORCH_CHECK(patch_tris.size(1) == 3 && patch_tris.size(2) == 3,
                "patch_tris must have shape (P, 3, 3)");
    check_tensor(patch_uvs, "patch_uvs", at::kFloat, 3);
    TORCH_CHECK(patch_uvs.size(0) == patch_tris.size(0) && patch_uvs.size(1) == 3 &&
                    patch_uvs.size(2) == 2,
                "patch_uvs must have shape (P, 3, 2)");
    check_flat_tensor(rows, "rows", at::kLong);
    const int64_t row_count = rows.size(0);
    check_vec3_table(d_i, "d_i");
    check_vec3_table(d_o, "d_o");
    check_vec3_table(n_rows, "n_rows");
    check_tensor(r_te, "r_te", at::kComplexFloat, 1);
    check_tensor(r_tm, "r_tm", at::kComplexFloat, 1);
    check_flat_tensor(pol_t, "pol_t", at::kFloat);
    check_flat_tensor(pol_r, "pol_r", at::kFloat);
    TORCH_CHECK(pol_t.size(0) == 3 && pol_r.size(0) == 3,
                "pol_t and pol_r must have shape (3,)");
    check_flat_tensor(r1_rows, "r1_rows", at::kFloat);
    check_flat_tensor(r2_rows, "r2_rows", at::kFloat);
    check_vec3_table(centroids, "centroids");
    check_tensor(heights, "heights", at::kFloat, 2);
    check_flat_tensor(quad_a, "quad_a", at::kFloat);
    check_flat_tensor(quad_b, "quad_b", at::kFloat);
    check_flat_tensor(quad_w, "quad_w", at::kFloat);
    TORCH_CHECK(quad_a.size(0) == kQuadPoints && quad_b.size(0) == kQuadPoints &&
                    quad_w.size(0) == kQuadPoints,
                "quadrature arrays must hold 16x16 Duffy-mapped nodes");
    TORCH_CHECK(d_i.size(0) == row_count && d_o.size(0) == row_count &&
                    n_rows.size(0) == row_count && r_te.size(0) == row_count &&
                    r_tm.size(0) == row_count && r1_rows.size(0) == row_count &&
                    r2_rows.size(0) == row_count && centroids.size(0) == row_count,
                "per-row arrays must match rows");
    for (const auto& tref : {patch_uvs, rows, d_i, d_o, n_rows, r_te, r_tm, pol_t,
                             pol_r, r1_rows, r2_rows, centroids, heights, quad_a,
                             quad_b, quad_w}) {
        TORCH_CHECK(tref.get_device() == patch_tris.get_device(),
                    "patch-integral tensors must share device");
    }
    return row_count;
}

const at::Tensor* optional_arg(
    pybind11::object value, at::Tensor& storage, const char* name,
    c10::ScalarType dtype, at::IntArrayRef sizes, const at::Tensor& reference) {
    if (value.is_none()) return nullptr;
    storage = value.cast<at::Tensor>().contiguous();
    TORCH_CHECK(storage.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(storage.scalar_type() == dtype, name, " has the wrong dtype");
    TORCH_CHECK(storage.sizes() == sizes, name, " has the wrong shape");
    TORCH_CHECK(storage.get_device() == reference.get_device(),
                name, " must share the primal device");
    return &storage;
}

template <typename T>
const T* opt_ptr(const at::Tensor* tensor) {
    return tensor == nullptr ? nullptr : tensor->data_ptr<T>();
}

at::Tensor zero_filled(at::IntArrayRef sizes, const at::TensorOptions& options) {
    auto tensor = at::empty(sizes, options);
    if (tensor.numel() > 0) {
        cudaStream_t stream =
            at::cuda::getCurrentCUDAStream(tensor.get_device()).stream();
        C10_CUDA_CHECK(cudaMemsetAsync(
            tensor.data_ptr(), 0,
            static_cast<size_t>(tensor.numel()) * tensor.element_size(), stream));
    }
    return tensor;
}

}  // namespace

pybind11::dict cn_scattering_patch_integral_eval_backward(
    at::Tensor patch_tris,
    at::Tensor patch_uvs,
    at::Tensor rows,
    at::Tensor d_i,
    at::Tensor d_o,
    at::Tensor n_rows,
    at::Tensor r_te,
    at::Tensor r_tm,
    at::Tensor pol_t,
    at::Tensor pol_r,
    at::Tensor r1_rows,
    at::Tensor r2_rows,
    at::Tensor centroids,
    at::Tensor heights,
    at::Tensor quad_a,
    at::Tensor quad_b,
    at::Tensor quad_w,
    double k0,
    at::Tensor grad_total,
    bool need_grad_heights,
    bool need_grad_jones,
    bool need_grad_geometry,
    bool need_grad_k0) {
    const int64_t row_count = check_patch_inputs(
        patch_tris, patch_uvs, rows, d_i, d_o, n_rows, r_te, r_tm, pol_t, pol_r,
        r1_rows, r2_rows, centroids, heights, quad_a, quad_b, quad_w);
    using channel_native::check_tensor;
    check_tensor(grad_total, "grad_total", at::kComplexFloat, 0);
    TORCH_CHECK(grad_total.get_device() == patch_tris.get_device(),
                "grad_total must share the primal device");

    at::Tensor grad_heights, grad_r_te, grad_r_tm, grad_d_i, grad_d_o, grad_r1,
        grad_r2, grad_centroids, grad_k0;
    if (need_grad_heights)
        grad_heights = zero_filled(
            {heights.size(0), heights.size(1)}, heights.options());
    if (need_grad_jones) {
        grad_r_te = at::empty_like(r_te);
        grad_r_tm = at::empty_like(r_tm);
    }
    if (need_grad_geometry) {
        grad_d_i = at::empty_like(d_i);
        grad_d_o = at::empty_like(d_o);
        grad_r1 = at::empty_like(r1_rows);
        grad_r2 = at::empty_like(r2_rows);
        grad_centroids = at::empty_like(centroids);
    }
    if (need_grad_k0)
        grad_k0 = zero_filled({1}, r1_rows.options());

    const bool any_need = need_grad_heights || need_grad_jones ||
                          need_grad_geometry || need_grad_k0;
    if (row_count > 0 && any_need) {
        cudaStream_t stream =
            at::cuda::getCurrentCUDAStream(patch_tris.get_device()).stream();
        patch_integral_backward_kernel<<<static_cast<int>(row_count), kQuadPoints, 0, stream>>>(
            row_count,
            patch_tris.data_ptr<float>(), patch_uvs.data_ptr<float>(),
            rows.data_ptr<int64_t>(), d_i.data_ptr<float>(), d_o.data_ptr<float>(),
            n_rows.data_ptr<float>(), r_te.data_ptr<cfloat>(), r_tm.data_ptr<cfloat>(),
            pol_t.data_ptr<float>(), pol_r.data_ptr<float>(),
            r1_rows.data_ptr<float>(), r2_rows.data_ptr<float>(),
            centroids.data_ptr<float>(), heights.data_ptr<float>(),
            static_cast<int>(heights.size(0)), static_cast<int>(heights.size(1)),
            quad_a.data_ptr<float>(), quad_b.data_ptr<float>(), quad_w.data_ptr<float>(),
            static_cast<float>(k0), grad_total.data_ptr<cfloat>(),
            need_grad_heights ? grad_heights.data_ptr<float>() : nullptr,
            need_grad_jones ? grad_r_te.data_ptr<cfloat>() : nullptr,
            need_grad_jones ? grad_r_tm.data_ptr<cfloat>() : nullptr,
            need_grad_geometry ? grad_d_i.data_ptr<float>() : nullptr,
            need_grad_geometry ? grad_d_o.data_ptr<float>() : nullptr,
            need_grad_geometry ? grad_r1.data_ptr<float>() : nullptr,
            need_grad_geometry ? grad_r2.data_ptr<float>() : nullptr,
            need_grad_geometry ? grad_centroids.data_ptr<float>() : nullptr,
            need_grad_k0 ? grad_k0.data_ptr<float>() : nullptr,
            need_grad_heights, need_grad_jones, need_grad_geometry, need_grad_k0);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

    auto none = []() { return pybind11::object(pybind11::none()); };
    pybind11::dict out;
    out["grad_heights"] = need_grad_heights ? pybind11::cast(grad_heights) : none();
    out["grad_r_te"] = need_grad_jones ? pybind11::cast(grad_r_te) : none();
    out["grad_r_tm"] = need_grad_jones ? pybind11::cast(grad_r_tm) : none();
    out["grad_d_i"] = need_grad_geometry ? pybind11::cast(grad_d_i) : none();
    out["grad_d_o"] = need_grad_geometry ? pybind11::cast(grad_d_o) : none();
    out["grad_r1_rows"] = need_grad_geometry ? pybind11::cast(grad_r1) : none();
    out["grad_r2_rows"] = need_grad_geometry ? pybind11::cast(grad_r2) : none();
    out["grad_centroids"] = need_grad_geometry ? pybind11::cast(grad_centroids) : none();
    out["grad_k0"] = need_grad_k0 ? pybind11::cast(grad_k0) : none();
    return out;
}

pybind11::dict cn_scattering_patch_integral_eval_jvp(
    at::Tensor patch_tris,
    at::Tensor patch_uvs,
    at::Tensor rows,
    at::Tensor d_i,
    at::Tensor d_o,
    at::Tensor n_rows,
    at::Tensor r_te,
    at::Tensor r_tm,
    at::Tensor pol_t,
    at::Tensor pol_r,
    at::Tensor r1_rows,
    at::Tensor r2_rows,
    at::Tensor centroids,
    at::Tensor heights,
    at::Tensor quad_a,
    at::Tensor quad_b,
    at::Tensor quad_w,
    double k0,
    pybind11::object t_heights,
    pybind11::object t_r_te,
    pybind11::object t_r_tm,
    pybind11::object t_d_i,
    pybind11::object t_d_o,
    pybind11::object t_r1_rows,
    pybind11::object t_r2_rows,
    pybind11::object t_centroids,
    double tangent_k0) {
    const int64_t row_count = check_patch_inputs(
        patch_tris, patch_uvs, rows, d_i, d_o, n_rows, r_te, r_tm, pol_t, pol_r,
        r1_rows, r2_rows, centroids, heights, quad_a, quad_b, quad_w);

    at::Tensor storage[8];
    const at::Tensor* th = optional_arg(
        std::move(t_heights), storage[0], "t_heights", at::kFloat,
        {heights.size(0), heights.size(1)}, patch_tris);
    const at::Tensor* tte = optional_arg(
        std::move(t_r_te), storage[1], "t_r_te", at::kComplexFloat,
        {row_count}, patch_tris);
    const at::Tensor* ttm = optional_arg(
        std::move(t_r_tm), storage[2], "t_r_tm", at::kComplexFloat,
        {row_count}, patch_tris);
    const at::Tensor* tdi = optional_arg(
        std::move(t_d_i), storage[3], "t_d_i", at::kFloat, {row_count, 3}, patch_tris);
    const at::Tensor* tdo = optional_arg(
        std::move(t_d_o), storage[4], "t_d_o", at::kFloat, {row_count, 3}, patch_tris);
    const at::Tensor* tr1 = optional_arg(
        std::move(t_r1_rows), storage[5], "t_r1_rows", at::kFloat, {row_count}, patch_tris);
    const at::Tensor* tr2 = optional_arg(
        std::move(t_r2_rows), storage[6], "t_r2_rows", at::kFloat, {row_count}, patch_tris);
    const at::Tensor* tc = optional_arg(
        std::move(t_centroids), storage[7], "t_centroids", at::kFloat,
        {row_count, 3}, patch_tris);

    auto tangent_total = at::empty(
        {}, patch_tris.options().dtype(at::kComplexFloat));
    cudaStream_t stream =
        at::cuda::getCurrentCUDAStream(patch_tris.get_device()).stream();
    if (row_count > 0) {
        auto t_row_value = at::empty(
            {row_count}, patch_tris.options().dtype(at::kComplexFloat));
        patch_integral_jvp_kernel<<<static_cast<int>(row_count), kQuadPoints, 0, stream>>>(
            row_count,
            patch_tris.data_ptr<float>(), patch_uvs.data_ptr<float>(),
            rows.data_ptr<int64_t>(), d_i.data_ptr<float>(), d_o.data_ptr<float>(),
            n_rows.data_ptr<float>(), r_te.data_ptr<cfloat>(), r_tm.data_ptr<cfloat>(),
            pol_t.data_ptr<float>(), pol_r.data_ptr<float>(),
            r1_rows.data_ptr<float>(), r2_rows.data_ptr<float>(),
            centroids.data_ptr<float>(), heights.data_ptr<float>(),
            static_cast<int>(heights.size(0)), static_cast<int>(heights.size(1)),
            quad_a.data_ptr<float>(), quad_b.data_ptr<float>(), quad_w.data_ptr<float>(),
            static_cast<float>(k0),
            opt_ptr<float>(th), opt_ptr<cfloat>(tte), opt_ptr<cfloat>(ttm),
            opt_ptr<float>(tdi), opt_ptr<float>(tdo), opt_ptr<float>(tr1),
            opt_ptr<float>(tr2), opt_ptr<float>(tc),
            static_cast<float>(tangent_k0),
            t_row_value.data_ptr<cfloat>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        patch_integral_jvp_total_kernel<<<1, kReduceBlock, 0, stream>>>(
            row_count, t_row_value.data_ptr<cfloat>(), tangent_total.data_ptr<cfloat>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    } else {
        C10_CUDA_CHECK(cudaMemsetAsync(
            tangent_total.data_ptr(), 0, tangent_total.element_size(), stream));
    }
    pybind11::dict out;
    out["tangent_total"] = tangent_total;
    return out;
}
