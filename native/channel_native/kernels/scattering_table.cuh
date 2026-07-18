#pragma once

// Shared device primitives for the resident Kirchhoff BSDF table lookup.
// Extracted verbatim from kernels/scattering.cu so both the single-table
// evaluation kernel and the ADR-010 ensemble kernel evaluate the table with
// exactly the same interpolation math (no behavior fork).

#include <cstdint>

namespace channel_native {
namespace scattering_tables {

constexpr float kTwoPi = 6.2831853071795864769f;

__device__ __forceinline__ float positive_phi(float y, float x) {
    float p = atan2f(y, x);
    return p < 0.0f ? p + kTwoPi : p;
}

__device__ __forceinline__ void linear_axis(
    float coord, int n, float period, bool periodic, int& i0, int& i1, float& w) {
    if (n == 1) { i0 = i1 = 0; w = 0.0f; return; }
    float t = coord * (static_cast<float>(n) / period) - 0.5f;
    if (periodic) {
        const int base = static_cast<int>(floorf(t));
        w = t - floorf(t);
        i0 = base % n; if (i0 < 0) i0 += n;
        i1 = (i0 + 1) % n;
    } else {
        t = fminf(fmaxf(t, 0.0f), static_cast<float>(n - 1));
        i0 = min(static_cast<int>(floorf(t)), n - 2);
        i1 = i0 + 1;
        w = t - static_cast<float>(i0);
    }
}

__device__ __forceinline__ int nearest_axis(float coord, int n, float period, bool periodic) {
    int i = static_cast<int>(floorf(coord * (static_cast<float>(n) / period)));
    if (periodic) { i %= n; return i < 0 ? i + n : i; }
    return min(max(i, 0), n - 1);
}

__device__ __forceinline__ float interp4(
    const float* __restrict__ table, int npi, int nto, int npo,
    int ti0, int ti1, float tw, int pi0, int pi1, float pw,
    int to0, int to1, float ow, int po0, int po1, float qw) {
    float out = 0.0f;
#pragma unroll
    for (int a = 0; a < 2; ++a) {
        const int ti = a ? ti1 : ti0; const float wa = a ? tw : 1.0f - tw;
#pragma unroll
        for (int b = 0; b < 2; ++b) {
            const int pi = b ? pi1 : pi0; const float wb = b ? pw : 1.0f - pw;
#pragma unroll
            for (int c = 0; c < 2; ++c) {
                const int to = c ? to1 : to0; const float wc = c ? ow : 1.0f - ow;
#pragma unroll
                for (int d = 0; d < 2; ++d) {
                    const int po = d ? po1 : po0; const float wd = d ? qw : 1.0f - qw;
                    const int64_t idx = ((static_cast<int64_t>(ti) * npi + pi) * nto + to) * npo + po;
                    out = fmaf(wa * wb * wc * wd, table[idx], out);
                }
            }
        }
    }
    return out;
}

// Per-pair (f_te, f_tm) lookup: identical to the body of scattering_eval_kernel
// in scattering.cu. Directions below the horizon return 0.
__device__ __forceinline__ void eval_te_tm(
    const float* __restrict__ fte, const float* __restrict__ ftm,
    int nti, int npi, int nto, int npo,
    const float* __restrict__ wi, const float* __restrict__ wo,
    float& out_te, float& out_tm) {
    if (wi[2] <= 0.0f || wo[2] <= 0.0f) { out_te = out_tm = 0.0f; return; }
    const float phi_i = positive_phi(wi[1], wi[0]);
    float phi_o = positive_phi(wo[1], wo[0]);
    if (npi == 1) { phi_o -= phi_i; if (phi_o < 0.0f) phi_o += kTwoPi; }
    int ti0, ti1, pi0, pi1, to0, to1, po0, po1; float tw, pw, ow, qw;
    linear_axis(wi[2], nti, 1.0f, false, ti0, ti1, tw);
    linear_axis(phi_i, npi, kTwoPi, true, pi0, pi1, pw);
    linear_axis(wo[2], nto, 1.0f, false, to0, to1, ow);
    linear_axis(phi_o, npo, kTwoPi, true, po0, po1, qw);
    out_te = interp4(fte, npi, nto, npo, ti0, ti1, tw, pi0, pi1, pw, to0, to1, ow, po0, po1, qw);
    out_tm = interp4(ftm, npi, nto, npo, ti0, ti1, tw, pi0, pi1, pw, to0, to1, ow, po0, po1, qw);
}

// ADR-014 derivative companion of ``linear_axis``: same i0/i1/w as the forward
// (bitwise, so recomputed table values round identically) plus the axis-weight
// derivative ``dw/dcoord``. Non-periodic axes gate the derivative to 0 when the
// pre-clamp coordinate leaves the open interval (0, n-1); periodic axes use the
// constant slope n/period; degenerate n==1 axes contribute 0.
__device__ __forceinline__ void linear_axis_grad(
    float coord, int n, float period, bool periodic,
    int& i0, int& i1, float& w, float& dwdc) {
    if (n == 1) { i0 = i1 = 0; w = 0.0f; dwdc = 0.0f; return; }
    const float scale = static_cast<float>(n) / period;
    float t = coord * scale - 0.5f;
    if (periodic) {
        const int base = static_cast<int>(floorf(t));
        w = t - floorf(t);
        i0 = base % n; if (i0 < 0) i0 += n;
        i1 = (i0 + 1) % n;
        dwdc = scale;
    } else {
        const float t_pre = t;
        t = fminf(fmaxf(t, 0.0f), static_cast<float>(n - 1));
        i0 = min(static_cast<int>(floorf(t)), n - 2);
        i1 = i0 + 1;
        w = t - static_cast<float>(i0);
        dwdc = (t_pre > 0.0f && t_pre < static_cast<float>(n - 1)) ? scale : 0.0f;
    }
}

// ADR-014 derivative companion of ``eval_te_tm``. Recomputes (f_te, f_tm) with
// the exact forward interpolation (same corner order, same fmaf accumulation)
// and, in the same pass, the partials of both values w.r.t. the raw ``wi`` and
// ``wo`` (= wo_local) 3-vectors, plus the 16 interpolation corners (flat index
// and weight ``wa*wb*wc*wd``) for the table-value VJP/JVP. ``active`` is false
// on the horizon gate, where value and all partials are zero.
struct TableEvalGrad {
    bool active;
    float te, tm;
    float dte_dwi[3], dtm_dwi[3];
    float dte_dwo[3], dtm_dwo[3];
    int64_t idx[16];
    float cw[16];
};

__device__ __forceinline__ void eval_te_tm_grad(
    const float* __restrict__ fte, const float* __restrict__ ftm,
    int nti, int npi, int nto, int npo,
    const float* __restrict__ wi, const float* __restrict__ wo,
    TableEvalGrad& g) {
    if (wi[2] <= 0.0f || wo[2] <= 0.0f) {
        g.active = false;
        g.te = g.tm = 0.0f;
        for (int i = 0; i < 3; ++i) {
            g.dte_dwi[i] = g.dtm_dwi[i] = 0.0f;
            g.dte_dwo[i] = g.dtm_dwo[i] = 0.0f;
        }
        for (int k = 0; k < 16; ++k) { g.idx[k] = 0; g.cw[k] = 0.0f; }
        return;
    }
    g.active = true;
    const float phi_i = positive_phi(wi[1], wi[0]);
    float phi_o = positive_phi(wo[1], wo[0]);
    if (npi == 1) { phi_o -= phi_i; if (phi_o < 0.0f) phi_o += kTwoPi; }
    int ti0, ti1, pi0, pi1, to0, to1, po0, po1;
    float tw, pw, ow, qw, dtw, dpw, dow, dqw;
    linear_axis_grad(wi[2], nti, 1.0f, false, ti0, ti1, tw, dtw);
    linear_axis_grad(phi_i, npi, kTwoPi, true, pi0, pi1, pw, dpw);
    linear_axis_grad(wo[2], nto, 1.0f, false, to0, to1, ow, dow);
    linear_axis_grad(phi_o, npo, kTwoPi, true, po0, po1, qw, dqw);

    // Value in forward order (fmaf, corner order a,b,c,d) so g.te/g.tm round
    // exactly like eval_te_tm; weight partials use plain arithmetic.
    float te = 0.0f, tm = 0.0f;
    float dte_tw = 0.0f, dte_pw = 0.0f, dte_ow = 0.0f, dte_qw = 0.0f;
    float dtm_tw = 0.0f, dtm_pw = 0.0f, dtm_ow = 0.0f, dtm_qw = 0.0f;
    int kk = 0;
#pragma unroll
    for (int a = 0; a < 2; ++a) {
        const int ti = a ? ti1 : ti0; const float wa = a ? tw : 1.0f - tw;
        const float sa = a ? 1.0f : -1.0f;
#pragma unroll
        for (int b = 0; b < 2; ++b) {
            const int pi = b ? pi1 : pi0; const float wb = b ? pw : 1.0f - pw;
            const float sb = b ? 1.0f : -1.0f;
#pragma unroll
            for (int cc = 0; cc < 2; ++cc) {
                const int to = cc ? to1 : to0; const float wc = cc ? ow : 1.0f - ow;
                const float sc = cc ? 1.0f : -1.0f;
#pragma unroll
                for (int d = 0; d < 2; ++d) {
                    const int po = d ? po1 : po0; const float wd = d ? qw : 1.0f - qw;
                    const float sd = d ? 1.0f : -1.0f;
                    const int64_t idx =
                        ((static_cast<int64_t>(ti) * npi + pi) * nto + to) * npo + po;
                    const float cw = wa * wb * wc * wd;
                    const float vte = fte[idx];
                    const float vtm = ftm[idx];
                    te = fmaf(cw, vte, te);
                    tm = fmaf(cw, vtm, tm);
                    dte_tw += (sa * wb * wc * wd) * vte;
                    dte_pw += (wa * sb * wc * wd) * vte;
                    dte_ow += (wa * wb * sc * wd) * vte;
                    dte_qw += (wa * wb * wc * sd) * vte;
                    dtm_tw += (sa * wb * wc * wd) * vtm;
                    dtm_pw += (wa * sb * wc * wd) * vtm;
                    dtm_ow += (wa * wb * sc * wd) * vtm;
                    dtm_qw += (wa * wb * wc * sd) * vtm;
                    g.idx[kk] = idx;
                    g.cw[kk] = cw;
                    ++kk;
                }
            }
        }
    }
    g.te = te; g.tm = tm;

    // Chain weight partials through dw/dcoord to the four lookup coordinates.
    const float df_te_wi2 = dte_tw * dtw;   // theta_i coord = wi[2]
    const float df_tm_wi2 = dtm_tw * dtw;
    const float df_te_wo2 = dte_ow * dow;   // theta_o coord = wo[2]
    const float df_tm_wo2 = dtm_ow * dow;
    const float df_te_phio = dte_qw * dqw;  // d/d phi_o' (post relative wrap)
    const float df_tm_phio = dtm_qw * dqw;
    // npi==1 couples phi_o' = wrap(phi_o - phi_i): d phi_o'/d phi_i = -1.
    const float coup = (npi == 1) ? -1.0f : 0.0f;
    const float df_te_phii = dte_pw * dpw + coup * df_te_phio;
    const float df_tm_phii = dtm_pw * dpw + coup * df_tm_phio;

    // atan2 chains: phi = atan2p(y, x), d phi/dx = -y/(x^2+y^2),
    // d phi/dy = x/(x^2+y^2); zero derivative at the (0,0) singularity.
    const float wi0 = wi[0], wi1 = wi[1];
    const float r2i = wi0 * wi0 + wi1 * wi1;
    const float inv_r2i = r2i > 0.0f ? 1.0f / r2i : 0.0f;
    const float dphii_dwi0 = -wi1 * inv_r2i;
    const float dphii_dwi1 = wi0 * inv_r2i;
    const float wo0 = wo[0], wo1 = wo[1];
    const float r2o = wo0 * wo0 + wo1 * wo1;
    const float inv_r2o = r2o > 0.0f ? 1.0f / r2o : 0.0f;
    const float dphio_dwo0 = -wo1 * inv_r2o;
    const float dphio_dwo1 = wo0 * inv_r2o;

    g.dte_dwi[0] = df_te_phii * dphii_dwi0;
    g.dte_dwi[1] = df_te_phii * dphii_dwi1;
    g.dte_dwi[2] = df_te_wi2;
    g.dtm_dwi[0] = df_tm_phii * dphii_dwi0;
    g.dtm_dwi[1] = df_tm_phii * dphii_dwi1;
    g.dtm_dwi[2] = df_tm_wi2;
    g.dte_dwo[0] = df_te_phio * dphio_dwo0;
    g.dte_dwo[1] = df_te_phio * dphio_dwo1;
    g.dte_dwo[2] = df_te_wo2;
    g.dtm_dwo[0] = df_tm_phio * dphio_dwo0;
    g.dtm_dwo[1] = df_tm_phio * dphio_dwo1;
    g.dtm_dwo[2] = df_tm_wo2;
}

}  // namespace scattering_tables
}  // namespace channel_native
