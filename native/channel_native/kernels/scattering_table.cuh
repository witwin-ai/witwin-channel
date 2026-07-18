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

}  // namespace scattering_tables
}  // namespace channel_native
