#pragma once

#include "em/layer_stack.cuh"
#include "field_transport.cuh"

#include <c10/util/complex.h>

// Companion derivative math for the field transport kernels (plan 07 AD-1
// materials/frequency, AD-2 geometry).
//
// Under the fixed-topology contract the discrete winner (face sequence,
// validity, normal-flip branches, polarizations, tx_power) is a constant of
// the differentiation; the differentiable inputs are the EM response
// parameters eps_r / sigma_e / gain / thickness (per bounce or per CSR
// layer), the carrier frequency, and since AD-2 the continuous hit geometry
// (source, target, interaction_positions, interaction_normals). Derivatives
// are propagated with forward-mode dual numbers that mirror the forward
// helpers step by step:
//
//   * slab_fresnel_dual mirrors field_transport::slab_fresnel,
//   * stack_rt_dual mirrors em::stack_rt (backward Rouard recursion),
//   * both take a cos_theta tangent seed: the incidence cosine is the only
//     door through which geometry enters the Fresnel response,
//   * the DualF3 helpers mirror utd::safe_normalize / stable_perp_basis and
//     transport::reflect_frame / wall_frame for the forward-mode frames,
//     while the reverse frames reuse utd::adj_safe_normalize /
//     adj_stable_perp_basis (same guards, shared source of truth),
//
// including every clamp and epsilon of the primal (a clamped branch carries a
// zero derivative, which matches what a central finite difference measures
// away from the kink). At the clamp boundary itself the subgradient follows
// the pass-through side of the forward fmaxf (gate with >=, not >), matching
// torch's clamp_min autograd in the tests/ad complex128 oracle so parameters
// initialized exactly at the boundary (e.g. sigma_e = 0) still receive a
// gradient. The mirroring is pinned by the tests/ad forward-parity and
// gradient-oracle suites; edit the primal helper and its dual TOGETHER.
//
// Real-pair convention: a complex cotangent g packs (dL/d out.re, dL/d out.im)
// exactly like torch's grad for complex tensors, so the adjoint of any
// C-linear step is applied on real pairs (see utd::adj_cplx_mul). Gradients
// for real inputs are plain real-pair dots against forward-mode duals.

namespace channel_native::field_transport_ad {

namespace utd = witwin::channel::native_ext;
namespace em = channel_native::em;
namespace transport = channel_native::field_transport;

// ---------------------------------------------------------------------------
// Dual scalars / complex numbers (value + derivative along one tangent).
// ---------------------------------------------------------------------------

struct DualF {
    float v;
    float d;
};

struct DualC {
    utd::Complex v;
    utd::Complex d;
};

__device__ __forceinline__ DualF df_const(float value) { return {value, 0.0f}; }

__device__ __forceinline__ DualC dc_const(utd::Complex value) {
    return {value, utd::cplx_zero()};
}

__device__ __forceinline__ DualC dc_make(float re, float im, float dre, float dim) {
    return {utd::cplx(re, im), utd::cplx(dre, dim)};
}

__device__ __forceinline__ DualC dc_add(DualC a, DualC b) {
    return {utd::cplx_add(a.v, b.v), utd::cplx_add(a.d, b.d)};
}

__device__ __forceinline__ DualC dc_sub(DualC a, DualC b) {
    return {utd::cplx_sub(a.v, b.v), utd::cplx_sub(a.d, b.d)};
}

__device__ __forceinline__ DualC dc_mul(DualC a, DualC b) {
    return {
        utd::cplx_mul(a.v, b.v),
        utd::cplx_add(utd::cplx_mul(a.d, b.v), utd::cplx_mul(a.v, b.d))};
}

__device__ __forceinline__ DualC dc_mul_real(DualC a, float b) {
    return {utd::cplx_mul_real(a.v, b), utd::cplx_mul_real(a.d, b)};
}

__device__ __forceinline__ DualC dc_mul_dualreal(DualC a, DualF b) {
    return {
        utd::cplx_mul_real(a.v, b.v),
        utd::cplx_add(
            utd::cplx_mul_real(a.d, b.v), utd::cplx_mul_real(a.v, b.d))};
}

// Dual of utd::cplx_div (denominator regularized with +UTD_EPS). The
// derivative treats the regularized quotient q = a*conj(b)/(|b|^2 + eps)
// exactly, so it matches the primal even near the floor.
__device__ __forceinline__ DualC dc_div_utd(DualC a, DualC b) {
    const float denom = b.v.re * b.v.re + b.v.im * b.v.im + utd::UTD_EPS;
    DualC out;
    out.v = utd::cplx_div(a.v, b.v);
    const float d_denom = 2.0f * (b.v.re * b.d.re + b.v.im * b.d.im);
    const utd::Complex d_num = utd::cplx_add(
        utd::cplx_mul(a.d, utd::cplx_conj(b.v)),
        utd::cplx_mul(a.v, utd::cplx_conj(b.d)));
    out.d = utd::cplx_div_real(
        utd::cplx_sub(d_num, utd::cplx_mul_real(out.v, d_denom)), denom);
    return out;
}

// Dual of em::c_div (denominator floored at 1e-30; the floor never binds for
// passive admittance sums, so the floored branch carries a zero denominator
// derivative).
__device__ __forceinline__ DualC dc_div_em(DualC a, DualC b) {
    const float mag2 = b.v.re * b.v.re + b.v.im * b.v.im;
    const float denom = fmaxf(mag2, 1.0e-30f);
    DualC out;
    out.v = em::c_div(a.v, b.v);
    const float d_denom =
        mag2 > 1.0e-30f ? 2.0f * (b.v.re * b.d.re + b.v.im * b.d.im) : 0.0f;
    const utd::Complex d_num = utd::cplx_add(
        utd::cplx_mul(a.d, utd::cplx_conj(b.v)),
        utd::cplx_mul(a.v, utd::cplx_conj(b.d)));
    out.d = utd::cplx_div_real(
        utd::cplx_sub(d_num, utd::cplx_mul_real(out.v, d_denom)), denom);
    return out;
}

// Dual of utd::cplx_sqrt: ds = dz / (2 sqrt(z)); the derivative is withheld
// (zero) at the branch point, mirroring utd::adj_cplx_sqrt.
__device__ __forceinline__ DualC dc_sqrt_utd(DualC a) {
    DualC out;
    out.v = utd::cplx_sqrt(a.v);
    if (utd::cplx_abs_sqr(out.v) <= utd::UTD_EPS) {
        out.d = utd::cplx_zero();
        return out;
    }
    out.d = utd::cplx_div(a.d, utd::cplx_mul_real(out.v, 2.0f));
    return out;
}

// Dual of em::c_sqrt_passive: same ds = dz / (2 w) rule on the passive branch.
__device__ __forceinline__ DualC dc_sqrt_passive(DualC a) {
    DualC out;
    out.v = em::c_sqrt_passive(a.v);
    if (utd::cplx_abs_sqr(out.v) <= 1.0e-30f) {
        out.d = utd::cplx_zero();
        return out;
    }
    out.d = em::c_div(a.d, utd::cplx_mul_real(out.v, 2.0f));
    return out;
}

// Dual of transport::exp_neg_2i (exp(-2j q) with growth clamp at exp(80)).
__device__ __forceinline__ DualC dc_exp_neg_2i(DualC q) {
    const float exponent = 2.0f * q.v.im;
    const float amplitude = expf(fminf(exponent, 80.0f));
    float sine;
    float cosine;
    sincosf(2.0f * q.v.re, &sine, &cosine);
    DualC out;
    out.v = utd::cplx(amplitude * cosine, -amplitude * sine);
    const float d_amplitude =
        exponent < 80.0f ? amplitude * 2.0f * q.d.im : 0.0f;
    const float d_theta = 2.0f * q.d.re;
    out.d = utd::cplx(
        d_amplitude * cosine - amplitude * sine * d_theta,
        -(d_amplitude * sine + amplitude * cosine * d_theta));
    return out;
}

// Real-pair dot: dL/dx contribution of a real input x whose forward tangent
// produced the complex dual derivative d, against the cotangent g.
__device__ __forceinline__ float adj_dot(utd::Complex g, utd::Complex d) {
    return g.re * d.re + g.im * d.im;
}

// ---------------------------------------------------------------------------
// slab_fresnel dual (mirrors field_transport::slab_fresnel step by step).
// Tangent seeds: d_cos_theta (geometry, plan 07 AD-2) and
// d_eps / d_sigma / d_gain / d_thickness / d_frequency.
// mu_r is a constant this phase (plan 07 AD-1).
// ---------------------------------------------------------------------------

// Derivative of fabsf matching torch.abs: sign(x), zero exactly at x == 0
// (the downstream SMALL_EPS clamp gates that corner off anyway).
__device__ __forceinline__ float d_fabsf(float x, float dx) {
    if (x > 0.0f)
        return dx;
    if (x < 0.0f)
        return -dx;
    return 0.0f;
}

__device__ __forceinline__ void slab_fresnel_dual(
    float cos_theta,
    float eps_r,
    float sigma_e,
    float mu_r,
    float gain,
    float thickness,
    float frequency_hz,
    float d_cos_theta,
    float d_eps,
    float d_sigma,
    float d_gain,
    float d_thickness,
    float d_frequency,
    DualC& r_te,
    DualC& r_tm) {
    const float omega_raw = 2.0f * utd::UTD_PI * frequency_hz;
    const float omega = fmaxf(omega_raw, utd::UTD_SMALL_EPS);
    const float d_omega =
        omega_raw > utd::UTD_SMALL_EPS ? 2.0f * utd::UTD_PI * d_frequency : 0.0f;
    const float wavelength = transport::kSpeedOfLight / frequency_hz;
    const float d_wavelength =
        -transport::kSpeedOfLight / (frequency_hz * frequency_hz) * d_frequency;
    const float ct_abs = fabsf(cos_theta);
    const float d_ct_abs = d_fabsf(cos_theta, d_cos_theta);
    const float ct = fminf(fmaxf(ct_abs, utd::UTD_SMALL_EPS), 1.0f);
    const float d_ct =
        (ct_abs >= utd::UTD_SMALL_EPS && ct_abs <= 1.0f) ? d_ct_abs : 0.0f;
    const float sin2_raw = 1.0f - ct * ct;
    const float sin2 = fmaxf(0.0f, sin2_raw);
    const float d_sin2 = sin2_raw >= 0.0f ? -2.0f * ct * d_ct : 0.0f;
    const float eps_clamped = fmaxf(eps_r, utd::UTD_SMALL_EPS);
    const float d_eps_clamped = eps_r > utd::UTD_SMALL_EPS ? d_eps : 0.0f;
    const float sigma_clamped = fmaxf(sigma_e, 0.0f);
    const float d_sigma_clamped = sigma_e >= 0.0f ? d_sigma : 0.0f;
    const float eta_im = -sigma_clamped / (omega * utd::UTD_EPSILON_0);
    const float d_eta_im =
        (-d_sigma_clamped / omega + sigma_clamped * d_omega / (omega * omega)) /
        utd::UTD_EPSILON_0;
    const DualC eta = dc_make(eps_clamped, eta_im, d_eps_clamped, d_eta_im);
    const float mu = fmaxf(mu_r, utd::UTD_SMALL_EPS);
    const DualC root = dc_sqrt_utd(dc_sub(
        dc_mul_real(eta, mu), dc_make(sin2, 0.0f, d_sin2, 0.0f)));
    const DualC mu_ct = dc_make(mu * ct, 0.0f, mu * d_ct, 0.0f);
    const DualC eta_ct = dc_mul_dualreal(eta, {ct, d_ct});
    const DualC interface_te = dc_div_utd(
        dc_sub(mu_ct, root), dc_add(mu_ct, root));
    const DualC interface_tm = dc_div_utd(
        dc_sub(eta_ct, root), dc_add(eta_ct, root));
    const float thickness_clamped = fmaxf(thickness, 0.0f);
    const float d_thickness_clamped = thickness >= 0.0f ? d_thickness : 0.0f;
    const float wavelength_clamped = fmaxf(wavelength, utd::UTD_SMALL_EPS);
    const float d_wavelength_clamped =
        wavelength > utd::UTD_SMALL_EPS ? d_wavelength : 0.0f;
    const float q_scale =
        2.0f * utd::UTD_PI * thickness_clamped / wavelength_clamped;
    const float d_q_scale = 2.0f * utd::UTD_PI *
                            (d_thickness_clamped * wavelength_clamped -
                             thickness_clamped * d_wavelength_clamped) /
                            (wavelength_clamped * wavelength_clamped);
    const DualC q = dc_mul_dualreal(root, {q_scale, d_q_scale});
    const DualC phase = dc_exp_neg_2i(q);
    const DualC one = dc_const(utd::cplx(1.0f, 0.0f));
    const DualC numerator = dc_sub(one, phase);
    const DualF gain_dual = {gain, d_gain};
    r_te = dc_mul_dualreal(
        dc_div_utd(
            dc_mul(interface_te, numerator),
            dc_sub(one, dc_mul(dc_mul(interface_te, interface_te), phase))),
        gain_dual);
    r_tm = dc_mul_dualreal(
        dc_div_utd(
            dc_mul(interface_tm, numerator),
            dc_sub(one, dc_mul(dc_mul(interface_tm, interface_tm), phase))),
        gain_dual);
}

// ---------------------------------------------------------------------------
// stack_rt dual (mirrors em::stack_rt step by step).
//
// SeedFn maps a flat CSR layer slot to its (thickness, eps_r, sigma_e)
// tangent triple; d_frequency seeds the carrier tangent. layer_mu_r is a
// constant this phase.
// ---------------------------------------------------------------------------

struct LayerSeed {
    float d_thickness;
    float d_eps;
    float d_sigma;
};

struct DualMedium {
    DualC eps_abs;
    utd::Complex mu_abs;  // constant this phase
    DualC k;
};

__device__ __forceinline__ DualMedium make_medium_dual(
    float eps_r,
    float sigma_e,
    float mu_r,
    DualF omega,
    float d_eps,
    float d_sigma) {
    DualMedium medium;
    const float safe_omega = fmaxf(omega.v, utd::UTD_SMALL_EPS);
    const float d_safe_omega = omega.v > utd::UTD_SMALL_EPS ? omega.d : 0.0f;
    const float eps_clamped = fmaxf(eps_r, utd::UTD_SMALL_EPS);
    const float d_eps_clamped = eps_r > utd::UTD_SMALL_EPS ? d_eps : 0.0f;
    const float sigma_clamped = fmaxf(sigma_e, 0.0f);
    const float d_sigma_clamped = sigma_e >= 0.0f ? d_sigma : 0.0f;
    medium.eps_abs = dc_make(
        em::kVacuumPermittivity * eps_clamped,
        -sigma_clamped / safe_omega,
        em::kVacuumPermittivity * d_eps_clamped,
        -d_sigma_clamped / safe_omega +
            sigma_clamped * d_safe_omega / (safe_omega * safe_omega));
    medium.mu_abs = utd::cplx(
        em::kVacuumPermeability * fmaxf(mu_r, utd::UTD_SMALL_EPS), 0.0f);
    medium.k = dc_mul_dualreal(
        dc_sqrt_passive(dc_mul(medium.eps_abs, dc_const(medium.mu_abs))),
        {safe_omega, d_safe_omega});
    return medium;
}

__device__ __forceinline__ DualC dc_kz_from_kpar(DualC k, DualF k_par) {
    return dc_sqrt_passive(dc_sub(
        dc_mul(k, k),
        dc_make(k_par.v * k_par.v, 0.0f, 2.0f * k_par.v * k_par.d, 0.0f)));
}

__device__ __forceinline__ DualC dc_admittance(
    const DualMedium& medium, DualC k_z, DualF omega, int pol) {
    if (pol == em::kPolTE) {
        const DualC mu_omega = dc_mul_dualreal(dc_const(medium.mu_abs), omega);
        return dc_div_em(k_z, mu_omega);
    }
    const DualC eps_omega = dc_mul_dualreal(medium.eps_abs, omega);
    return dc_div_em(eps_omega, k_z);
}

struct DualInterfaceRT {
    DualC r;
    DualC t;
};

__device__ __forceinline__ DualInterfaceRT dc_interface_rt(DualC y1, DualC y2) {
    const DualC denom = dc_add(y1, y2);
    DualInterfaceRT out;
    out.r = dc_div_em(dc_sub(y1, y2), denom);
    out.t = dc_div_em(dc_mul_real(y1, 2.0f), denom);
    return out;
}

// Dual of em::layer_one_way_phase (decay clamp at exponent 0; primal phasor
// keeps its double-precision argument reduction). The passive branch keeps
// exponent = Im(k_z)*d <= 0, so the fminf clamp only guards float noise; at
// the boundary itself (sigma_e = 0 or thickness = 0 both give
// exponent == -0.0) the subgradient follows the pass-through side of the
// fminf (gate with <=, mirroring the fmaxf >= convention above) so the decay
// derivative survives exactly where the tests/ad oracle differentiates it.
__device__ __forceinline__ DualC dc_layer_one_way_phase(
    DualC k_z, DualF thickness_m) {
    const float exponent = k_z.v.im * thickness_m.v;
    const float amplitude = expf(fminf(exponent, 0.0f));
    const float d_exponent = k_z.d.im * thickness_m.v + k_z.v.im * thickness_m.d;
    const float d_amplitude = exponent <= 0.0f ? amplitude * d_exponent : 0.0f;
    const utd::Complex phasor = em::c_exp_neg_j(
        static_cast<double>(k_z.v.re) * static_cast<double>(thickness_m.v));
    const float d_theta = k_z.d.re * thickness_m.v + k_z.v.re * thickness_m.d;
    // phasor = (cos t, -sin t); d/dt = (sin t and cos t read off the primal).
    const utd::Complex d_phasor = utd::cplx(
        phasor.im * d_theta, -phasor.re * d_theta);
    DualC out;
    out.v = utd::cplx_mul_real(phasor, amplitude);
    out.d = utd::cplx_add(
        utd::cplx_mul_real(d_phasor, amplitude),
        utd::cplx_mul_real(phasor, d_amplitude));
    return out;
}

struct DualStackRT {
    DualC r;
    DualC t;
};

template <typename SeedFn>
__device__ __forceinline__ DualStackRT stack_rt_dual(
    float cos_theta_i,
    const em::LayerView& layers,
    float frequency_hz,
    float d_cos_theta,
    float d_frequency,
    int pol,
    SeedFn&& seed) {
    const float omega_raw = 2.0f * utd::UTD_PI * frequency_hz;
    const DualF omega = {
        fmaxf(omega_raw, utd::UTD_SMALL_EPS),
        omega_raw > utd::UTD_SMALL_EPS ? 2.0f * utd::UTD_PI * d_frequency : 0.0f};
    const float ct_abs = fabsf(cos_theta_i);
    const float d_ct_abs = d_fabsf(cos_theta_i, d_cos_theta);
    const float ct = fminf(fmaxf(ct_abs, utd::UTD_SMALL_EPS), 1.0f);
    const float d_ct =
        (ct_abs >= utd::UTD_SMALL_EPS && ct_abs <= 1.0f) ? d_ct_abs : 0.0f;
    const float sin2_raw = 1.0f - ct * ct;
    const float sin2 = fmaxf(0.0f, sin2_raw);
    const float d_sin2 = sin2_raw >= 0.0f ? -2.0f * ct * d_ct : 0.0f;

    // Entry medium is vacuum (v1); its wave number is omega / c.
    const float k_entry = omega.v / em::kSpeedOfLight;
    const float d_k_entry = omega.d / em::kSpeedOfLight;
    const float sin_theta = sqrtf(sin2);
    // d sqrt at sin2 == 0 diverges like the oracle's; the d_sin2 == 0 gate
    // only keeps the untouched-tangent case (materials/frequency seeds)
    // exactly zero instead of 0 * inf.
    const float d_sin_theta =
        d_sin2 == 0.0f ? 0.0f : d_sin2 / (2.0f * sin_theta);
    const DualF k_par = {
        k_entry * sin_theta,
        d_k_entry * sin_theta + k_entry * d_sin_theta};
    const DualC kz_entry = dc_make(
        k_entry * ct, 0.0f, d_k_entry * ct + k_entry * d_ct, 0.0f);
    DualMedium entry;
    entry.eps_abs = dc_const(utd::cplx(em::kVacuumPermittivity, 0.0f));
    entry.mu_abs = utd::cplx(em::kVacuumPermeability, 0.0f);
    entry.k = dc_make(k_entry, 0.0f, d_k_entry, 0.0f);
    const DualC y_entry = dc_admittance(entry, kz_entry, omega, pol);
    const DualC y_exit = y_entry;

    DualStackRT out;
    const int count = layers.layer_count[layers.material];
    const int offset = layers.layer_offset[layers.material];
    if (count <= 0) {
        out.r = dc_const(utd::cplx_zero());
        out.t = dc_const(utd::cplx(1.0f, 0.0f));
        return out;
    }

    const int last = offset + count - 1;
    const LayerSeed last_seed = seed(last);
    DualMedium below = make_medium_dual(
        layers.layer_eps_r[last],
        layers.layer_sigma_e[last],
        layers.layer_mu_r[last],
        omega,
        last_seed.d_eps,
        last_seed.d_sigma);
    DualC kz_below = dc_kz_from_kpar(below.k, k_par);
    DualC y_below = dc_admittance(below, kz_below, omega, pol);
    const DualInterfaceRT exit_interface = dc_interface_rt(y_below, y_exit);
    DualC r_total = exit_interface.r;
    DualC t_total = exit_interface.t;

    for (int layer = count - 1; layer >= 0; --layer) {
        const int slot = offset + layer;
        const LayerSeed slot_seed = seed(slot);
        const float thickness_raw = layers.layer_thickness_m[slot];
        const DualF thickness = {
            fmaxf(thickness_raw, 0.0f),
            thickness_raw >= 0.0f ? slot_seed.d_thickness : 0.0f};
        const DualC phase = dc_layer_one_way_phase(kz_below, thickness);
        const DualC phase2 = dc_mul(phase, phase);

        DualC kz_above;
        DualC y_above;
        if (layer > 0) {
            const int above_slot = slot - 1;
            const LayerSeed above_seed = seed(above_slot);
            const DualMedium above = make_medium_dual(
                layers.layer_eps_r[above_slot],
                layers.layer_sigma_e[above_slot],
                layers.layer_mu_r[above_slot],
                omega,
                above_seed.d_eps,
                above_seed.d_sigma);
            kz_above = dc_kz_from_kpar(above.k, k_par);
            y_above = dc_admittance(above, kz_above, omega, pol);
        } else {
            kz_above = kz_entry;
            y_above = y_entry;
        }
        const DualInterfaceRT top = dc_interface_rt(y_above, y_below);
        const DualC loop = dc_mul(phase2, r_total);
        const DualC denom = dc_add(
            dc_const(utd::cplx(1.0f, 0.0f)), dc_mul(top.r, loop));
        r_total = dc_div_em(dc_add(top.r, loop), denom);
        t_total = dc_div_em(dc_mul(top.t, dc_mul(phase, t_total)), denom);

        kz_below = kz_above;
        y_below = y_above;
    }

    out.r = r_total;
    out.t = t_total;
    return out;
}

// ---------------------------------------------------------------------------
// Free-space carrier, templated over the real type so the same expression
// serves the float32 backward/jvp companions and the float64 gradcheck path
// (plan 07 section 9.1). Epsilon constants keep the float32 forward values so
// both precisions share one clamping contract.
// ---------------------------------------------------------------------------

template <typename T>
struct Vec3 {
    T x;
    T y;
    T z;
};

template <typename T>
__device__ __forceinline__ Vec3<T> v3_load(const T* values, int64_t index) {
    const int64_t base = index * 3;
    return {values[base], values[base + 1], values[base + 2]};
}

template <typename T>
__device__ __forceinline__ Vec3<T> v3_sub(Vec3<T> a, Vec3<T> b) {
    return {a.x - b.x, a.y - b.y, a.z - b.z};
}

template <typename T>
__device__ __forceinline__ Vec3<T> v3_scale(Vec3<T> a, T s) {
    return {a.x * s, a.y * s, a.z * s};
}

template <typename T>
__device__ __forceinline__ T v3_dot(Vec3<T> a, Vec3<T> b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

template <typename T>
__device__ __forceinline__ T v3_length(Vec3<T> a) {
    const T sq = v3_dot(a, a);
    return sqrt(sq > T(0) ? sq : T(0));
}

template <typename T>
__device__ __forceinline__ Vec3<T> v3_safe_normalize(Vec3<T> v, Vec3<T> alternate) {
    const T n = v3_length(v);
    if (n > T(utd::UTD_SMALL_EPS))
        return v3_scale(v, T(1) / (n + T(utd::UTD_EPS)));
    const T fn = v3_length(alternate);
    return v3_scale(alternate, T(1) / (fn + T(utd::UTD_EPS)));
}

template <typename T>
__device__ __forceinline__ Vec3<T> v3_stable_perp_basis(
    Vec3<T> ray_dir, Vec3<T> preferred) {
    const Vec3<T> proj = v3_sub(
        preferred, v3_scale(ray_dir, v3_dot(preferred, ray_dir)));
    const Vec3<T> alt_axis = (fabs(ray_dir.z) < T(0.9))
                                 ? Vec3<T>{T(0), T(0), T(1)}
                                 : Vec3<T>{T(0), T(1), T(0)};
    const Vec3<T> alt_proj = v3_sub(
        alt_axis, v3_scale(ray_dir, v3_dot(alt_axis, ray_dir)));
    return v3_safe_normalize(proj, alt_proj);
}

constexpr float kSpeedOfLight = transport::kSpeedOfLight;

// Fixed free-space carrier evaluation plus its frequency and distance
// derivatives. The primal mirrors free_space_kernel exactly (same clamps,
// same double-fmod phase reduction); the derivatives use
//   dP/df = P * (-1/k - j*d) * (2*pi/c)
//   dP/dd = P * (-[d >= EPS]/d_clamped - j*k)
// (the fmod phase reduction has unit slope, so the raw wave number drives
// the phase term; the amplitude term follows the clamp_min(EPS) subgradient
// convention of the tests/ad oracle).
template <typename T>
struct FreeSpaceEval {
    Vec3<T> direction;
    Vec3<T> tx_axis;
    Vec3<T> rx_axis;
    T distance;
    T amplitude_scale;      // sqrt(max(tx_power, 0))
    T projection;           // dot(tx_axis, rx_axis)
    c10::complex<T> carrier;        // P = exp(-j k d) / (2 k d)
    c10::complex<T> carrier_dfreq;  // dP/df
    c10::complex<T> carrier_ddist;  // dP/dd
};

template <typename T>
__device__ __forceinline__ FreeSpaceEval<T> free_space_eval(
    Vec3<T> source,
    Vec3<T> target,
    Vec3<T> tx_polarization,
    Vec3<T> rx_polarization,
    T tx_power,
    T frequency_hz) {
    FreeSpaceEval<T> out;
    const Vec3<T> offset = v3_sub(target, source);
    out.distance = v3_length(offset);
    out.direction = v3_safe_normalize(offset, Vec3<T>{T(0), T(0), T(1)});
    out.tx_axis = v3_stable_perp_basis(out.direction, tx_polarization);
    out.rx_axis = v3_stable_perp_basis(out.direction, rx_polarization);
    out.amplitude_scale = sqrt(tx_power > T(0) ? tx_power : T(0));
    out.projection = v3_dot(out.tx_axis, out.rx_axis);
    const T wave_number =
        T(2.0 * 3.14159265358979323846) * frequency_hz / T(kSpeedOfLight);
    const T k_clamped =
        wave_number > T(utd::UTD_SMALL_EPS) ? wave_number : T(utd::UTD_SMALL_EPS);
    const T d_clamped =
        out.distance > T(utd::UTD_EPS) ? out.distance : T(utd::UTD_EPS);
    const T amplitude = T(1) / (T(2) * k_clamped * d_clamped);
    const double phase_full = static_cast<double>(wave_number) *
                              static_cast<double>(out.distance);
    const double phase = -fmod(phase_full, 6.283185307179586476925287);
    const T phase_t = static_cast<T>(phase);
    out.carrier = c10::complex<T>(
        amplitude * cos(phase_t), amplitude * sin(phase_t));
    // dP/dk = P * (-1/k - j d); dk/df = 2 pi / c (clamps inactive for f > 0).
    const c10::complex<T> dlog(
        -T(1) / k_clamped, -out.distance);
    out.carrier_dfreq = out.carrier * dlog *
                        (T(2.0 * 3.14159265358979323846) / T(kSpeedOfLight));
    const T amplitude_gate =
        out.distance >= T(utd::UTD_EPS) ? T(1) / d_clamped : T(0);
    const c10::complex<T> dlog_dist(-amplitude_gate, -wave_number);
    out.carrier_ddist = out.carrier * dlog_dist;
    return out;
}

// ---------------------------------------------------------------------------
// Vec3<T> forward-mode duals and reverse-mode adjoints of the shared vector
// helpers (v3_length / v3_safe_normalize / v3_stable_perp_basis), used by the
// free-space geometry companions (plan 07 AD-2). Every gate mirrors the
// primal branch; length subgradients vanish at zero like torch's vector_norm.
// ---------------------------------------------------------------------------

template <typename T>
__device__ __forceinline__ Vec3<T> v3_add(Vec3<T> a, Vec3<T> b) {
    return {a.x + b.x, a.y + b.y, a.z + b.z};
}

template <typename T>
__device__ __forceinline__ Vec3<T> v3_neg(Vec3<T> a) {
    return {-a.x, -a.y, -a.z};
}

template <typename T>
struct DualV3 {
    Vec3<T> v;
    Vec3<T> d;
};

template <typename T>
__device__ __forceinline__ DualV3<T> dv3_const(Vec3<T> value) {
    return {value, {T(0), T(0), T(0)}};
}

template <typename T>
__device__ __forceinline__ DualV3<T> dv3_sub(DualV3<T> a, DualV3<T> b) {
    return {v3_sub(a.v, b.v), v3_sub(a.d, b.d)};
}

// Dual of v3_length: d|v| = (v . dv)/|v| on the positive branch, zero at the
// origin (the primal takes the max(., 0) zero branch there).
template <typename T>
__device__ __forceinline__ T dual_v3_length(DualV3<T> a, T& d_length) {
    const T sq = v3_dot(a.v, a.v);
    const T length = sqrt(sq > T(0) ? sq : T(0));
    d_length = sq > T(0) ? v3_dot(a.v, a.d) / length : T(0);
    return length;
}

// Dual of v3_safe_normalize: u = v/(n + EPS) on the main branch, otherwise
// the same map on the alternate. d u = dv * s - v * (v . dv) * s^2 / n.
template <typename T>
__device__ __forceinline__ DualV3<T> dual_v3_safe_normalize(
    DualV3<T> v, DualV3<T> alternate) {
    const T n = v3_length(v.v);
    const bool main_branch = n > T(utd::UTD_SMALL_EPS);
    const DualV3<T>& active = main_branch ? v : alternate;
    const T an = main_branch ? n : v3_length(alternate.v);
    const T s = T(1) / (an + T(utd::UTD_EPS));
    DualV3<T> out;
    out.v = v3_scale(active.v, s);
    if (an > T(0)) {
        const T dn = v3_dot(active.v, active.d) / an;
        out.d = v3_sub(
            v3_scale(active.d, s), v3_scale(active.v, dn * s * s));
    } else {
        out.d = {T(0), T(0), T(0)};
    }
    return out;
}

// Dual of v3_stable_perp_basis with a fixed preferred axis (the tx/rx
// polarizations are constants of the differentiation). The discrete
// alternate-axis pick |dir.z| < 0.9 is frozen.
template <typename T>
__device__ __forceinline__ DualV3<T> dual_v3_stable_perp_basis(
    DualV3<T> ray_dir, Vec3<T> preferred) {
    const T proj_dot = v3_dot(preferred, ray_dir.v);
    const T d_proj_dot = v3_dot(preferred, ray_dir.d);
    DualV3<T> proj;
    proj.v = v3_sub(preferred, v3_scale(ray_dir.v, proj_dot));
    proj.d = v3_neg(v3_add(
        v3_scale(ray_dir.d, proj_dot), v3_scale(ray_dir.v, d_proj_dot)));
    const Vec3<T> alt_axis = (fabs(ray_dir.v.z) < T(0.9))
                                 ? Vec3<T>{T(0), T(0), T(1)}
                                 : Vec3<T>{T(0), T(1), T(0)};
    const T alt_dot = v3_dot(alt_axis, ray_dir.v);
    const T d_alt_dot = v3_dot(alt_axis, ray_dir.d);
    DualV3<T> alt_proj;
    alt_proj.v = v3_sub(alt_axis, v3_scale(ray_dir.v, alt_dot));
    alt_proj.d = v3_neg(v3_add(
        v3_scale(ray_dir.d, alt_dot), v3_scale(ray_dir.v, d_alt_dot)));
    return dual_v3_safe_normalize(proj, alt_proj);
}

// Adjoint of v3_length into g_v (gate matches dual_v3_length).
template <typename T>
__device__ __forceinline__ void adj_v3_length(Vec3<T> v, T g_length, Vec3<T>& g_v) {
    const T sq = v3_dot(v, v);
    if (!(sq > T(0)))
        return;
    const T length = sqrt(sq);
    g_v = v3_add(g_v, v3_scale(v, g_length / length));
}

// Adjoint of the active v3_safe_normalize branch (mirror of
// utd::adj_normalize_branch on Vec3<T>).
template <typename T>
__device__ __forceinline__ void adj_v3_normalize_branch(
    Vec3<T> v, Vec3<T> g_out, Vec3<T>& g_v) {
    const T sq = v3_dot(v, v);
    const T n = sqrt(sq > T(0) ? sq : T(0));
    if (!(n > T(0)))
        return;
    const T denom = n + T(utd::UTD_EPS);
    const T dg = v3_dot(g_out, v);
    g_v = v3_add(
        g_v,
        v3_sub(v3_scale(g_out, T(1) / denom), v3_scale(v, dg / (n * denom * denom))));
}

template <typename T>
__device__ __forceinline__ void adj_v3_safe_normalize(
    Vec3<T> v, Vec3<T> alternate, Vec3<T> g_out, Vec3<T>& g_v, Vec3<T>& g_alternate) {
    if (v3_length(v) > T(utd::UTD_SMALL_EPS)) {
        adj_v3_normalize_branch(v, g_out, g_v);
    } else {
        adj_v3_normalize_branch(alternate, g_out, g_alternate);
    }
}

// Adjoint of v3_stable_perp_basis into the ray direction (preferred axis is
// fixed; its cotangent is discarded by the callers).
template <typename T>
__device__ __forceinline__ void adj_v3_stable_perp_basis(
    Vec3<T> ray_dir, Vec3<T> preferred, Vec3<T> g_out, Vec3<T>& g_ray_dir) {
    const T proj_dot = v3_dot(preferred, ray_dir);
    const Vec3<T> proj = v3_sub(preferred, v3_scale(ray_dir, proj_dot));
    const Vec3<T> alt_axis = (fabs(ray_dir.z) < T(0.9))
                                 ? Vec3<T>{T(0), T(0), T(1)}
                                 : Vec3<T>{T(0), T(1), T(0)};
    const T alt_dot = v3_dot(alt_axis, ray_dir);
    const Vec3<T> alt_proj = v3_sub(alt_axis, v3_scale(ray_dir, alt_dot));
    Vec3<T> g_proj = {T(0), T(0), T(0)};
    Vec3<T> g_alt_proj = {T(0), T(0), T(0)};
    adj_v3_safe_normalize(proj, alt_proj, g_out, g_proj, g_alt_proj);
    // proj = preferred - ray_dir * (preferred . ray_dir)
    g_ray_dir = v3_sub(g_ray_dir, v3_scale(g_proj, proj_dot));
    g_ray_dir = v3_sub(g_ray_dir, v3_scale(preferred, v3_dot(g_proj, ray_dir)));
    // alt_proj = alt_axis - ray_dir * (alt_axis . ray_dir)
    g_ray_dir = v3_sub(g_ray_dir, v3_scale(g_alt_proj, alt_dot));
    g_ray_dir = v3_sub(g_ray_dir, v3_scale(alt_axis, v3_dot(g_alt_proj, ray_dir)));
}

// ---------------------------------------------------------------------------
// float3a forward-mode duals of the utd vector helpers plus dual and adjoint
// interaction frames (plan 07 AD-2). The .v computations replay
// utd::safe_normalize / utd::stable_perp_basis / transport::reflect_frame /
// transport::wall_frame operation by operation so the primal values agree
// with the forward kernels; the reverse counterparts lean on RayD's own
// utd::adj_safe_normalize / adj_stable_perp_basis.
// ---------------------------------------------------------------------------

struct DualF3 {
    utd::float3a v;
    utd::float3a d;
};

__device__ __forceinline__ DualF3 df3_const(utd::float3a value) {
    return {value, utd::f3_zero()};
}

__device__ __forceinline__ DualF3 df3_make(utd::float3a value, utd::float3a d) {
    return {value, d};
}

__device__ __forceinline__ DualF3 df3_sub(DualF3 a, DualF3 b) {
    return {utd::f3_sub(a.v, b.v), utd::f3_sub(a.d, b.d)};
}

__device__ __forceinline__ DualF3 df3_neg(DualF3 a) {
    return {utd::f3_neg(a.v), utd::f3_neg(a.d)};
}

__device__ __forceinline__ DualF3 df3_cross(DualF3 a, DualF3 b) {
    return {
        utd::f3_cross(a.v, b.v),
        utd::f3_add(utd::f3_cross(a.d, b.v), utd::f3_cross(a.v, b.d))};
}

__device__ __forceinline__ DualF df3_dot(DualF3 a, DualF3 b) {
    return {
        utd::f3_dot(a.v, b.v),
        utd::f3_dot(a.d, b.v) + utd::f3_dot(a.v, b.d)};
}

// Dual of utd::safe_length (sqrt of the clamped square; zero tangent at the
// origin, matching torch vector_norm).
__device__ __forceinline__ DualF dual_safe_length(DualF3 a) {
    const float sq = utd::f3_dot(a.v, a.v);
    const float length = sqrtf(fmaxf(sq, 0.0f));
    return {length, sq > 0.0f ? utd::f3_dot(a.v, a.d) / length : 0.0f};
}

// Adjoint of utd::safe_length (same zero-at-origin subgradient).
__device__ __forceinline__ void adj_safe_length(
    utd::float3a v, float g_length, utd::float3a& g_v) {
    const float sq = utd::f3_dot(v, v);
    if (!(sq > 0.0f))
        return;
    g_v = utd::f3_add(g_v, utd::f3_mul(v, g_length / sqrtf(sq)));
}

// Dual of utd::safe_normalize: u = v / (n + EPS) on the active branch.
__device__ __forceinline__ DualF3 dual_safe_normalize(DualF3 v, DualF3 alternate) {
    const float n = utd::safe_length(v.v);
    const bool main_branch = n > utd::UTD_SMALL_EPS;
    const DualF3& active = main_branch ? v : alternate;
    const float an = main_branch ? n : utd::safe_length(alternate.v);
    const float denom = an + utd::UTD_EPS;
    DualF3 out;
    out.v = utd::f3_div(active.v, denom);
    if (an > 0.0f) {
        const float dn = utd::f3_dot(active.v, active.d) / an;
        out.d = utd::f3_sub(
            utd::f3_div(active.d, denom),
            utd::f3_mul(active.v, dn / (denom * denom)));
    } else {
        out.d = utd::f3_zero();
    }
    return out;
}

// Dual of utd::stable_perp_basis (both arguments may carry tangents; the
// discrete alternate-axis pick is frozen).
__device__ __forceinline__ DualF3 dual_stable_perp_basis(
    DualF3 ray_dir, DualF3 preferred) {
    const DualF proj_dot = df3_dot(preferred, ray_dir);
    const DualF3 proj = {
        utd::f3_sub(preferred.v, utd::f3_mul(ray_dir.v, proj_dot.v)),
        utd::f3_sub(
            preferred.d,
            utd::f3_add(
                utd::f3_mul(ray_dir.d, proj_dot.v),
                utd::f3_mul(ray_dir.v, proj_dot.d)))};
    const utd::float3a alt_axis = (fabsf(ray_dir.v.z) < 0.9f)
                                      ? utd::make_f3(0.0f, 0.0f, 1.0f)
                                      : utd::make_f3(0.0f, 1.0f, 0.0f);
    const float alt_dot = utd::f3_dot(alt_axis, ray_dir.v);
    const float d_alt_dot = utd::f3_dot(alt_axis, ray_dir.d);
    const DualF3 alt_proj = {
        utd::f3_sub(alt_axis, utd::f3_mul(ray_dir.v, alt_dot)),
        utd::f3_neg(utd::f3_add(
            utd::f3_mul(ray_dir.d, alt_dot),
            utd::f3_mul(ray_dir.v, d_alt_dot)))};
    return dual_safe_normalize(proj, alt_proj);
}

// ---------------------------------------------------------------------------
// Dual and adjoint specular reflection frames (transport::reflect_frame).
// The normal flip against the incident ray is a frozen discrete branch whose
// linear part is the sign: d(-n) = -dn.
// ---------------------------------------------------------------------------

struct DualReflectFrame {
    DualF3 incident;
    DualF3 s_axis;
    DualF3 p_in;
    DualF3 p_out;
    DualF3 reflected_direction;
    DualF cos_theta;
};

__device__ __forceinline__ DualReflectFrame dual_reflect_frame(
    DualF3 incident_direction,
    DualF3 normal) {
    DualReflectFrame frame;
    const DualF3 e_z = df3_const(utd::make_f3(0.0f, 0.0f, 1.0f));
    frame.incident = dual_safe_normalize(incident_direction, e_z);
    DualF3 oriented = dual_safe_normalize(normal, e_z);
    const bool flip = utd::f3_dot(frame.incident.v, oriented.v) > 0.0f;
    if (flip)
        oriented = df3_neg(oriented);
    const DualF dot_in = df3_dot(frame.incident, oriented);
    const DualF3 reflect_raw = {
        utd::f3_sub(
            frame.incident.v, utd::f3_mul(oriented.v, 2.0f * dot_in.v)),
        utd::f3_sub(
            frame.incident.d,
            utd::f3_add(
                utd::f3_mul(oriented.d, 2.0f * dot_in.v),
                utd::f3_mul(oriented.v, 2.0f * dot_in.d)))};
    frame.reflected_direction = dual_safe_normalize(
        reflect_raw, df3_neg(frame.incident));
    const DualF3 s_raw = df3_cross(oriented, frame.incident);
    frame.s_axis = dual_safe_normalize(
        s_raw, dual_stable_perp_basis(frame.incident, oriented));
    frame.p_in = dual_safe_normalize(
        df3_cross(frame.s_axis, frame.incident),
        dual_stable_perp_basis(frame.incident, frame.s_axis));
    frame.p_out = dual_safe_normalize(
        df3_cross(frame.s_axis, frame.reflected_direction),
        dual_stable_perp_basis(frame.reflected_direction, frame.s_axis));
    frame.cos_theta = {fabsf(dot_in.v), d_fabsf(dot_in.v, dot_in.d)};
    return frame;
}

// Reverse-mode adjoint of transport::reflect_frame. The frame outputs carry
// the cotangents g_s_axis / g_p_in / g_p_out / g_reflected / g_cos_theta;
// the results accumulate into the cotangents of the (already normalized)
// incident direction argument and the raw interaction normal. Intermediates
// are recomputed exactly like the primal so every branch matches.
__device__ __forceinline__ void adj_reflect_frame(
    utd::float3a incident_direction,
    utd::float3a normal,
    utd::float3a g_s_axis,
    utd::float3a g_p_in,
    utd::float3a g_p_out,
    utd::float3a g_reflected,
    float g_cos_theta,
    utd::float3a& g_incident_direction,
    utd::float3a& g_normal) {
    const utd::float3a e_z = utd::make_f3(0.0f, 0.0f, 1.0f);
    // Primal replay (mirrors transport::reflect_frame).
    const utd::float3a incident = utd::safe_normalize(incident_direction, e_z);
    const utd::float3a n_unit = utd::safe_normalize(normal, e_z);
    const bool flip = utd::f3_dot(incident, n_unit) > 0.0f;
    const utd::float3a oriented = flip ? utd::f3_neg(n_unit) : n_unit;
    const float dot_in = utd::f3_dot(incident, oriented);
    const utd::float3a reflect_raw = utd::f3_sub(
        incident, utd::f3_mul(oriented, 2.0f * dot_in));
    const utd::float3a reflected = utd::safe_normalize(
        reflect_raw, utd::f3_neg(incident));
    const utd::float3a s_raw = utd::f3_cross(oriented, incident);
    const utd::float3a s_alt = utd::stable_perp_basis(incident, oriented);
    const utd::float3a s_axis = utd::safe_normalize(s_raw, s_alt);
    const utd::float3a p_in_raw = utd::f3_cross(s_axis, incident);
    const utd::float3a p_out_raw = utd::f3_cross(s_axis, reflected);

    utd::float3a g_incident = utd::f3_zero();
    utd::float3a g_oriented = utd::f3_zero();
    utd::float3a g_reflected_total = g_reflected;
    utd::float3a g_s_total = g_s_axis;

    // p_out = safe_normalize(s x reflected, stable_perp_basis(reflected, s)).
    {
        utd::float3a g_raw = utd::f3_zero();
        utd::float3a g_alt = utd::f3_zero();
        utd::adj_safe_normalize(
            p_out_raw, utd::stable_perp_basis(reflected, s_axis), g_p_out,
            g_raw, g_alt);
        g_s_total = utd::f3_add(g_s_total, utd::f3_cross(reflected, g_raw));
        g_reflected_total = utd::f3_add(
            g_reflected_total, utd::f3_cross(g_raw, s_axis));
        utd::adj_stable_perp_basis(reflected, s_axis, g_alt, g_reflected_total, g_s_total);
    }
    // p_in = safe_normalize(s x incident, stable_perp_basis(incident, s)).
    {
        utd::float3a g_raw = utd::f3_zero();
        utd::float3a g_alt = utd::f3_zero();
        utd::adj_safe_normalize(
            p_in_raw, utd::stable_perp_basis(incident, s_axis), g_p_in,
            g_raw, g_alt);
        g_s_total = utd::f3_add(g_s_total, utd::f3_cross(incident, g_raw));
        g_incident = utd::f3_add(g_incident, utd::f3_cross(g_raw, s_axis));
        utd::adj_stable_perp_basis(incident, s_axis, g_alt, g_incident, g_s_total);
    }
    // s_axis = safe_normalize(oriented x incident, stable_perp_basis(incident,
    // oriented)).
    {
        utd::float3a g_raw = utd::f3_zero();
        utd::float3a g_alt = utd::f3_zero();
        utd::adj_safe_normalize(s_raw, s_alt, g_s_total, g_raw, g_alt);
        g_oriented = utd::f3_add(g_oriented, utd::f3_cross(incident, g_raw));
        g_incident = utd::f3_add(g_incident, utd::f3_cross(g_raw, oriented));
        utd::adj_stable_perp_basis(incident, oriented, g_alt, g_incident, g_oriented);
    }
    // reflected = safe_normalize(incident - 2*dot_in*oriented, -incident).
    float g_dot_in = 0.0f;
    {
        utd::float3a g_raw = utd::f3_zero();
        utd::float3a g_neg_incident = utd::f3_zero();
        utd::adj_safe_normalize(
            reflect_raw, utd::f3_neg(incident), g_reflected_total,
            g_raw, g_neg_incident);
        g_incident = utd::f3_sub(g_incident, g_neg_incident);
        g_incident = utd::f3_add(g_incident, g_raw);
        g_oriented = utd::f3_sub(g_oriented, utd::f3_mul(g_raw, 2.0f * dot_in));
        g_dot_in -= 2.0f * utd::f3_dot(g_raw, oriented);
    }
    // cos_theta = |dot_in| (torch.abs subgradient: zero exactly at 0).
    if (dot_in > 0.0f)
        g_dot_in += g_cos_theta;
    else if (dot_in < 0.0f)
        g_dot_in -= g_cos_theta;
    // dot_in = incident . oriented.
    g_incident = utd::f3_add(g_incident, utd::f3_mul(oriented, g_dot_in));
    g_oriented = utd::f3_add(g_oriented, utd::f3_mul(incident, g_dot_in));
    // oriented = flip ? -n_unit : n_unit (frozen branch, linear sign).
    const utd::float3a g_n_unit = flip ? utd::f3_neg(g_oriented) : g_oriented;
    // n_unit = safe_normalize(normal, e_z); incident = safe_normalize(arg, e_z).
    utd::float3a g_dump = utd::f3_zero();
    adj_safe_normalize(normal, e_z, g_n_unit, g_normal, g_dump);
    utd::adj_safe_normalize(
        incident_direction, e_z, g_incident, g_incident_direction, g_dump);
}

// ---------------------------------------------------------------------------
// Dual and adjoint thin-sheet wall frames (transport::wall_frame).
// ---------------------------------------------------------------------------

struct DualWallFrame {
    DualF3 s_axis;
    DualF3 p_axis;
    DualF cos_theta;
};

__device__ __forceinline__ DualWallFrame dual_wall_frame(
    DualF3 direction,
    DualF3 raw_normal) {
    const DualF3 e_z = df3_const(utd::make_f3(0.0f, 0.0f, 1.0f));
    DualF3 normal = dual_safe_normalize(raw_normal, e_z);
    if (utd::f3_dot(direction.v, normal.v) > 0.0f)
        normal = df3_neg(normal);
    DualWallFrame frame;
    const DualF dot = df3_dot(direction, normal);
    const float ct_abs = fabsf(dot.v);
    const float d_ct_abs = d_fabsf(dot.v, dot.d);
    frame.cos_theta.v = fminf(fmaxf(ct_abs, utd::UTD_SMALL_EPS), 1.0f);
    frame.cos_theta.d =
        (ct_abs >= utd::UTD_SMALL_EPS && ct_abs <= 1.0f) ? d_ct_abs : 0.0f;
    const DualF3 s_raw = df3_cross(normal, direction);
    frame.s_axis = dual_safe_normalize(
        s_raw, dual_stable_perp_basis(direction, normal));
    frame.p_axis = dual_safe_normalize(
        df3_cross(frame.s_axis, direction),
        dual_stable_perp_basis(direction, frame.s_axis));
    return frame;
}

// Reverse-mode adjoint of transport::wall_frame into the ray direction and
// the raw wall normal.
__device__ __forceinline__ void adj_wall_frame(
    utd::float3a direction,
    utd::float3a raw_normal,
    utd::float3a g_s_axis,
    utd::float3a g_p_axis,
    float g_cos_theta,
    utd::float3a& g_direction,
    utd::float3a& g_raw_normal) {
    const utd::float3a e_z = utd::make_f3(0.0f, 0.0f, 1.0f);
    // Primal replay (mirrors transport::wall_frame).
    const utd::float3a n_unit = utd::safe_normalize(raw_normal, e_z);
    const bool flip = utd::f3_dot(direction, n_unit) > 0.0f;
    const utd::float3a normal = flip ? utd::f3_neg(n_unit) : n_unit;
    const float dot = utd::f3_dot(direction, normal);
    const float ct_abs = fabsf(dot);
    const utd::float3a s_raw = utd::f3_cross(normal, direction);
    const utd::float3a s_alt = utd::stable_perp_basis(direction, normal);
    const utd::float3a s_axis = utd::safe_normalize(s_raw, s_alt);
    const utd::float3a p_raw = utd::f3_cross(s_axis, direction);

    utd::float3a g_normal = utd::f3_zero();
    utd::float3a g_s_total = g_s_axis;

    // p_axis = safe_normalize(s x direction, stable_perp_basis(direction, s)).
    {
        utd::float3a g_raw = utd::f3_zero();
        utd::float3a g_alt = utd::f3_zero();
        utd::adj_safe_normalize(
            p_raw, utd::stable_perp_basis(direction, s_axis), g_p_axis,
            g_raw, g_alt);
        g_s_total = utd::f3_add(g_s_total, utd::f3_cross(direction, g_raw));
        g_direction = utd::f3_add(g_direction, utd::f3_cross(g_raw, s_axis));
        utd::adj_stable_perp_basis(direction, s_axis, g_alt, g_direction, g_s_total);
    }
    // s_axis = safe_normalize(normal x direction, stable_perp_basis(direction,
    // normal)).
    {
        utd::float3a g_raw = utd::f3_zero();
        utd::float3a g_alt = utd::f3_zero();
        utd::adj_safe_normalize(s_raw, s_alt, g_s_total, g_raw, g_alt);
        g_normal = utd::f3_add(g_normal, utd::f3_cross(direction, g_raw));
        g_direction = utd::f3_add(g_direction, utd::f3_cross(g_raw, normal));
        utd::adj_stable_perp_basis(direction, normal, g_alt, g_direction, g_normal);
    }
    // cos_theta = clamp(|dot|, SMALL_EPS, 1).
    if (ct_abs >= utd::UTD_SMALL_EPS && ct_abs <= 1.0f) {
        float g_dot = 0.0f;
        if (dot > 0.0f)
            g_dot = g_cos_theta;
        else if (dot < 0.0f)
            g_dot = -g_cos_theta;
        g_direction = utd::f3_add(g_direction, utd::f3_mul(normal, g_dot));
        g_normal = utd::f3_add(g_normal, utd::f3_mul(direction, g_dot));
    }
    // Frozen flip sign, then the input normalize.
    const utd::float3a g_n_unit = flip ? utd::f3_neg(g_normal) : g_normal;
    utd::float3a g_dump = utd::f3_zero();
    adj_safe_normalize(raw_normal, e_z, g_n_unit, g_raw_normal, g_dump);
}

}  // namespace channel_native::field_transport_ad
