#pragma once

#include "em/layer_stack.cuh"
#include "field_transport.cuh"

#include <c10/util/complex.h>

// Companion derivative math for the field transport kernels (plan 07 AD-1).
//
// Under the fixed-topology contract the hit geometry (endpoints, interaction
// positions/normals, polarizations, tx_power) is a constant of the
// differentiation; the differentiable inputs are the EM response parameters
// eps_r / sigma_e / gain / thickness (per bounce or per CSR layer) and the
// carrier frequency. Derivatives are propagated with forward-mode dual
// numbers that mirror the forward helpers step by step:
//
//   * slab_fresnel_dual mirrors field_transport::slab_fresnel,
//   * stack_rt_dual mirrors em::stack_rt (backward Rouard recursion),
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
// Tangent seeds: d_eps / d_sigma / d_gain / d_thickness / d_frequency.
// mu_r is a constant this phase (plan 07 AD-1).
// ---------------------------------------------------------------------------

__device__ __forceinline__ void slab_fresnel_dual(
    float cos_theta,
    float eps_r,
    float sigma_e,
    float mu_r,
    float gain,
    float thickness,
    float frequency_hz,
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
    const float ct = fminf(fmaxf(fabsf(cos_theta), utd::UTD_SMALL_EPS), 1.0f);
    const float sin2 = fmaxf(0.0f, 1.0f - ct * ct);
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
        dc_mul_real(eta, mu), dc_const(utd::cplx(sin2, 0.0f))));
    const DualC mu_ct = dc_const(utd::cplx(mu * ct, 0.0f));
    const DualC eta_ct = dc_mul_real(eta, ct);
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
    float d_frequency,
    int pol,
    SeedFn&& seed) {
    const float omega_raw = 2.0f * utd::UTD_PI * frequency_hz;
    const DualF omega = {
        fmaxf(omega_raw, utd::UTD_SMALL_EPS),
        omega_raw > utd::UTD_SMALL_EPS ? 2.0f * utd::UTD_PI * d_frequency : 0.0f};
    const float ct = fminf(fmaxf(fabsf(cos_theta_i), utd::UTD_SMALL_EPS), 1.0f);
    const float sin2 = fmaxf(0.0f, 1.0f - ct * ct);

    // Entry medium is vacuum (v1); its wave number is omega / c.
    const float k_entry = omega.v / em::kSpeedOfLight;
    const float d_k_entry = omega.d / em::kSpeedOfLight;
    const float sin_theta = sqrtf(sin2);
    const DualF k_par = {k_entry * sin_theta, d_k_entry * sin_theta};
    const DualC kz_entry = dc_make(k_entry * ct, 0.0f, d_k_entry * ct, 0.0f);
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

// Fixed free-space carrier evaluation plus its frequency derivative. The
// primal mirrors free_space_kernel exactly (same clamps, same double-fmod
// phase reduction); the derivative uses dP/df = P * (-1/k - j*d) * (2*pi/c).
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
    return out;
}

}  // namespace channel_native::field_transport_ad
