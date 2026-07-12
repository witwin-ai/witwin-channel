#pragma once

#include "complex.cuh"

// Homogeneous isotropic medium parameters (contract section 2).
//
// Complex relative permittivity: eps_r_c = eps_r' - j*sigma_e/(omega*eps0)
// (time factor e^{+j w t}). v1 materials expose real mu_r. The wave number
// uses the passive branch k = omega*sqrt(eps_abs*mu_abs) with Re(k) >= 0 and
// Im(k) <= 0 so e^{-j k z} decays.

namespace channel_native::em {

constexpr float kVacuumPermittivity = 8.8541878128e-12f;
constexpr float kVacuumPermeability = 1.25663706212e-6f;
constexpr float kSpeedOfLight = 299792458.0f;

struct Medium {
    utd::Complex eps_abs;  // absolute permittivity eps0 * eps_r_complex [F/m]
    utd::Complex mu_abs;   // absolute permeability mu0 * mu_r_complex [H/m]
    utd::Complex k;        // wave number, passive branch [rad/m]
};

__device__ __forceinline__ Medium make_medium(
    float eps_r, float sigma_e, float mu_r, float omega) {
    Medium medium;
    const float safe_omega = fmaxf(omega, utd::UTD_SMALL_EPS);
    medium.eps_abs = utd::cplx(
        kVacuumPermittivity * fmaxf(eps_r, utd::UTD_SMALL_EPS),
        -fmaxf(sigma_e, 0.0f) / safe_omega);
    medium.mu_abs = utd::cplx(
        kVacuumPermeability * fmaxf(mu_r, utd::UTD_SMALL_EPS), 0.0f);
    medium.k = utd::cplx_mul_real(
        c_sqrt_passive(utd::cplx_mul(medium.eps_abs, medium.mu_abs)),
        safe_omega);
    return medium;
}

__device__ __forceinline__ Medium vacuum_medium(float omega) {
    Medium medium;
    medium.eps_abs = utd::cplx(kVacuumPermittivity, 0.0f);
    medium.mu_abs = utd::cplx(kVacuumPermeability, 0.0f);
    medium.k = utd::cplx(fmaxf(omega, utd::UTD_SMALL_EPS) / kSpeedOfLight, 0.0f);
    return medium;
}

}  // namespace channel_native::em
