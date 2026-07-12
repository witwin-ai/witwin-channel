#pragma once

#include <rayd/shared/utd/utd_math.h>

// Shared complex helpers for the electromagnetic core (plan 05, contract
// section 2). This is a thin layer over utd::Complex: reuse the existing utd
// arithmetic helpers wherever they exist and only add the pieces the EM
// conventions require (passive-branch square root and precise phasors).

namespace channel_native::em {

namespace utd = witwin::channel::native_ext;

// Passive-branch complex square root.
//
// Contract section 2 fixes the branch Re(w) >= 0, Im(w) <= 0 so that with the
// e^{+j w t} time factor the propagation factor e^{-j w z} always decays in a
// passive medium. The branch is implemented explicitly from the
// magnitude/real-part form instead of relying on library sqrt defaults:
//   with r = |z|:  Re(w) = sqrt((r + Re z)/2) >= 0
//                  Im(w) = -sqrt((r - Re z)/2) <= 0
// Then w^2 = Re(z) - j*|Im(z)|, which is exact for every passive argument
// (Im(z) <= 0, including the negative real axis where the evanescent root
// -j*sqrt(|z|) is required) and maps float-noise arguments with slightly
// positive Im(z) onto the passive root of their conjugate.
__device__ __forceinline__ utd::Complex c_sqrt_passive(utd::Complex z) {
    const float magnitude = hypotf(z.re, z.im);
    const float real = sqrtf(fmaxf(0.5f * (magnitude + z.re), 0.0f));
    const float imag = -sqrtf(fmaxf(0.5f * (magnitude - z.re), 0.0f));
    return utd::cplx(real, imag);
}

// exp(-j*phase) with double-precision argument reduction (same technique as
// field_transport::precise_neg_kd) so long optical paths and thick layers do
// not lose carrier-phase accuracy in float32.
__device__ __forceinline__ utd::Complex c_exp_neg_j(double phase) {
    const double reduced = fmod(phase, 6.283185307179586476925287);
    float sine;
    float cosine;
    sincosf(static_cast<float>(reduced), &sine, &cosine);
    return utd::cplx(cosine, -sine);
}

// Exact complex division. utd::cplx_div adds UTD_EPS (1e-10) to |b|^2, which
// biases quotients by ~1e-5 relative when |b| ~ 1e-3 - exactly the magnitude
// of SI wave admittances - and that bias is visible in R+T energy budgets.
// The floor here only guards a fully degenerate denominator (passive
// admittance sums never come close to it).
__device__ __forceinline__ utd::Complex c_div(utd::Complex a, utd::Complex b) {
    const float denom = fmaxf(b.re * b.re + b.im * b.im, 1.0e-30f);
    return utd::cplx(
        (a.re * b.re + a.im * b.im) / denom,
        (a.im * b.re - a.re * b.im) / denom);
}

__device__ __forceinline__ float c_abs2(utd::Complex a) {
    return utd::cplx_abs_sqr(a);
}

}  // namespace channel_native::em
