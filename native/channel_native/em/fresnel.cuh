#pragma once

#include "medium.cuh"

// Single-interface Fresnel coefficients in the wave-admittance form
// (contract section 2 / plan section 4.1).
//
// The tangential wave number k_par is conserved across the interface;
// k_z,m = sqrt(k_m^2 - k_par^2) on the passive branch. Amplitudes are defined
// on the shared tangential electric field direction:
//   r = (Y1 - Y2)/(Y1 + Y2),  t = 2*Y1/(Y1 + Y2)
// with Y_TE = k_z/(omega*mu) and Y_TM = omega*eps/k_z (absolute eps/mu).
// The TM sign convention follows from these boundary conditions; do not
// hand-flip it to match scalar reference formulas.

namespace channel_native::em {

constexpr int kPolTE = 0;
constexpr int kPolTM = 1;

struct InterfaceRT {
    utd::Complex r;
    utd::Complex t;
};

__device__ __forceinline__ utd::Complex kz_from_kpar(
    utd::Complex k, float k_par) {
    return c_sqrt_passive(
        utd::cplx_sub(utd::cplx_mul(k, k), utd::cplx(k_par * k_par, 0.0f)));
}

__device__ __forceinline__ utd::Complex admittance_te(
    const Medium& medium, utd::Complex k_z, float omega) {
    return c_div(k_z, utd::cplx_mul_real(medium.mu_abs, omega));
}

__device__ __forceinline__ utd::Complex admittance_tm(
    const Medium& medium, utd::Complex k_z, float omega) {
    return c_div(utd::cplx_mul_real(medium.eps_abs, omega), k_z);
}

__device__ __forceinline__ utd::Complex admittance(
    const Medium& medium, utd::Complex k_z, float omega, int pol) {
    return pol == kPolTE ? admittance_te(medium, k_z, omega)
                         : admittance_tm(medium, k_z, omega);
}

__device__ __forceinline__ InterfaceRT interface_rt(
    utd::Complex y1, utd::Complex y2) {
    const utd::Complex denom = utd::cplx_add(y1, y2);
    InterfaceRT out;
    out.r = c_div(utd::cplx_sub(y1, y2), denom);
    out.t = c_div(utd::cplx_mul_real(y1, 2.0f), denom);
    return out;
}

}  // namespace channel_native::em
