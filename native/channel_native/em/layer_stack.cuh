#pragma once

#include "fresnel.cuh"

// Multilayer slab reflection/transmission via the numerically stable backward
// Airy (Rouard) recursion (contract sections 2 and 9; plan section 5.1 gives
// the transfer-matrix oracle this must match).
//
// Media are indexed 0 (entry, vacuum in v1), 1..N (layers from the CSR store),
// N+1 (exit/backing, vacuum in v1). Interface I_l sits between media l-1 and
// l. With per-interface admittance amplitudes
//   r_l = (Y_{l-1} - Y_l)/(Y_{l-1} + Y_l),  t_l = 2*Y_{l-1}/(Y_{l-1} + Y_l)
// the reverse-direction coefficients obey r'_l = -r_l and the Stokes relation
// t_l*t'_l = 1 - r_l^2 (t'_l = 2*Y_l/(Y_{l-1}+Y_l)).
//
// Define R_l, T_l as the composite reflection/transmission of everything from
// interface I_l down to the exit, referenced just above I_l and just below
// I_{N+1}. Base case (last interface): R_{N+1} = r_{N+1}, T_{N+1} = t_{N+1}.
// Between I_l and I_{l+1} lies layer l with one-way propagator
// p_l = exp(-j*k_{z,l}*d_l). Summing the interface geometric series
// (Airy summation):
//   R_l = r_l + t_l*t'_l*p_l^2*R_{l+1} * sum_{m>=0} (r'_l*p_l^2*R_{l+1})^m
//       = r_l + (1 - r_l^2)*p_l^2*R_{l+1} / (1 + r_l*p_l^2*R_{l+1})
//       = (r_l + p_l^2*R_{l+1}) / (1 + r_l*p_l^2*R_{l+1})
//   T_l = t_l*p_l*T_{l+1} * sum_{m>=0} (r'_l*p_l^2*R_{l+1})^m
//       = t_l*p_l*T_{l+1} / (1 + r_l*p_l^2*R_{l+1})
// This is algebraically identical to the transfer-matrix form
// (r = (Y0*B - C)/(Y0*B + C), t = 2*Y0/(Y0*B + C) with B, C from the product
// of layer matrices [[cos d, j sin d/Y],[j Y sin d, cos d]]): both resum the
// same interface Fresnel series; the transfer matrix carries cos/sin of the
// complex phase (mixing e^{+j k_z d} and e^{-j k_z d} terms and therefore
// growing exponentials for lossy/thick layers), while the recursion above
// only ever multiplies by p_l with |p_l| <= 1, so it stays finite for
// arbitrarily thick or conductive layers.
//
// t is defined interface-to-interface: it already contains all interior
// k_z*d phase and absorption but no exterior free-space carrier phase.

namespace channel_native::em {

// CSR view over the material layer store (ABI v3 tensors) plus the material
// row this evaluation targets. Layers of material m occupy
// [offset[m], offset[m] + count[m]) in the flat arrays.
struct LayerView {
    const int* layer_offset;      // int32[M]
    const int* layer_count;       // int32[M]
    const float* layer_thickness_m;  // f32[L]
    const float* layer_eps_r;        // f32[L]
    const float* layer_sigma_e;      // f32[L]
    const float* layer_mu_r;         // f32[L]
    int material;
};

struct StackRT {
    utd::Complex r;
    utd::Complex t;
    float cap_r;  // |r|^2
    float cap_t;  // Re(Y_exit)/Re(Y_entry) * |t|^2
};

// One-way layer propagator p = exp(-j*k_z*d).
// The passive branch guarantees Im(k_z) <= 0, so |p| = exp(Im(k_z)*d) <= 1.
// The exponent is clamped at 0 so float noise can never produce a growing
// exponential; strong decay (|Im(k_z)|*d large, e.g. sigma_e up to 1e9 S/m
// with thickness up to 10 m) underflows expf to 0 cleanly instead of
// overflowing anywhere in the recursion.
__device__ __forceinline__ utd::Complex layer_one_way_phase(
    utd::Complex k_z, float thickness_m) {
    const float amplitude = expf(
        fminf(k_z.im * thickness_m, 0.0f));
    const utd::Complex phasor = c_exp_neg_j(
        static_cast<double>(k_z.re) * static_cast<double>(thickness_m));
    return utd::cplx_mul_real(phasor, amplitude);
}

// Evaluate the full vacuum | layers | vacuum stack at incidence angle
// cos_theta_i (entry-medium angle; k_par = k0*sin(theta_i) is conserved).
// Zero-layer materials are transparent: r = 0, t = 1.
__device__ __forceinline__ StackRT stack_rt(
    float cos_theta_i,
    const LayerView& layers,
    float frequency_hz,
    int pol) {
    const float omega = fmaxf(
        2.0f * utd::UTD_PI * frequency_hz, utd::UTD_SMALL_EPS);
    const float ct = fminf(fmaxf(fabsf(cos_theta_i), utd::UTD_SMALL_EPS), 1.0f);
    const float sin2 = fmaxf(0.0f, 1.0f - ct * ct);

    const Medium entry = vacuum_medium(omega);
    // k_par is fixed by the entry medium (vacuum in v1).
    const float k_par = entry.k.re * sqrtf(sin2);
    const utd::Complex kz_entry = utd::cplx(entry.k.re * ct, 0.0f);
    const utd::Complex y_entry = admittance(entry, kz_entry, omega, pol);
    // Outside and backing media are vacuum in v1.
    const utd::Complex y_exit = y_entry;

    StackRT out;
    const int count = layers.layer_count[layers.material];
    const int offset = layers.layer_offset[layers.material];
    if (count <= 0) {
        out.r = utd::cplx_zero();
        out.t = utd::cplx(1.0f, 0.0f);
        out.cap_r = 0.0f;
        out.cap_t = 1.0f;
        return out;
    }

    // Backward pass: start at the last interface (layer N -> exit medium),
    // then fold layers N..1 in.
    const int last = offset + count - 1;
    Medium below = make_medium(
        layers.layer_eps_r[last],
        layers.layer_sigma_e[last],
        layers.layer_mu_r[last],
        omega);
    utd::Complex kz_below = kz_from_kpar(below.k, k_par);
    utd::Complex y_below = admittance(below, kz_below, omega, pol);
    InterfaceRT exit_interface = interface_rt(y_below, y_exit);
    utd::Complex r_total = exit_interface.r;
    utd::Complex t_total = exit_interface.t;

    for (int layer = count - 1; layer >= 0; --layer) {
        const int slot = offset + layer;
        const utd::Complex phase = layer_one_way_phase(
            kz_below, fmaxf(layers.layer_thickness_m[slot], 0.0f));
        const utd::Complex phase2 = utd::cplx_mul(phase, phase);

        // Medium above interface I_layer: previous layer, or entry medium.
        Medium above;
        utd::Complex kz_above;
        utd::Complex y_above;
        if (layer > 0) {
            const int above_slot = slot - 1;
            above = make_medium(
                layers.layer_eps_r[above_slot],
                layers.layer_sigma_e[above_slot],
                layers.layer_mu_r[above_slot],
                omega);
            kz_above = kz_from_kpar(above.k, k_par);
            y_above = admittance(above, kz_above, omega, pol);
        } else {
            kz_above = kz_entry;
            y_above = y_entry;
        }
        const InterfaceRT top = interface_rt(y_above, y_below);
        const utd::Complex loop = utd::cplx_mul(phase2, r_total);
        const utd::Complex denom = utd::cplx_add(
            utd::cplx(1.0f, 0.0f), utd::cplx_mul(top.r, loop));
        r_total = c_div(utd::cplx_add(top.r, loop), denom);
        t_total = c_div(
            utd::cplx_mul(top.t, utd::cplx_mul(phase, t_total)), denom);

        kz_below = kz_above;
        y_below = y_above;
    }

    out.r = r_total;
    out.t = t_total;
    out.cap_r = c_abs2(r_total);
    // Power transmittance carries the Re(Y_exit)/Re(Y_entry) flux factor
    // (unity for the v1 vacuum/vacuum surround, kept general on purpose).
    const float y_entry_re = fmaxf(y_entry.re, utd::UTD_SMALL_EPS * 1.0e-6f);
    out.cap_t = (y_exit.re / y_entry_re) * c_abs2(t_total);
    return out;
}

}  // namespace channel_native::em
