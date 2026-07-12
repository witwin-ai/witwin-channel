#pragma once

#include <rayd/shared/utd/utd_math.h>

namespace channel_native::field_transport {

namespace utd = witwin::channel::native_ext;
constexpr float kSpeedOfLight = 299792458.0f;

__device__ __forceinline__ float precise_neg_kd(float wave_number, float distance) {
    const double phase = fmod(
        static_cast<double>(wave_number) * static_cast<double>(distance),
        6.283185307179586476925287);
    return -static_cast<float>(phase);
}

__device__ __forceinline__ utd::Complex3 free_space_complex3(
    utd::float3a source,
    utd::float3a target,
    float wave_number,
    utd::float3a polarization) {
    const utd::float3a offset = utd::f3_sub(target, source);
    const float distance = utd::safe_length(offset);
    const utd::float3a direction = utd::safe_normalize(
        offset, utd::make_f3(0.0f, 0.0f, 1.0f));
    const utd::float3a axis = utd::stable_perp_basis(direction, polarization);
    const float amplitude = 1.0f /
                            (2.0f * fmaxf(wave_number, utd::UTD_SMALL_EPS) *
                             fmaxf(distance, utd::UTD_EPS));
    const utd::Complex phase = utd::cplx_exp_phase(
        precise_neg_kd(wave_number, distance));
    return utd::cplx_scale_real(axis, utd::cplx_mul_real(phase, amplitude));
}

__device__ __forceinline__ utd::Complex project_receiver(
    utd::Complex3 value,
    utd::float3a direction,
    utd::float3a polarization) {
    const utd::float3a axis = utd::stable_perp_basis(direction, polarization);
    return utd::cplx_add(
        utd::cplx_add(
            utd::cplx_mul_real(value.x, axis.x),
            utd::cplx_mul_real(value.y, axis.y)),
        utd::cplx_mul_real(value.z, axis.z));
}

__device__ __forceinline__ utd::Complex exp_neg_2i(utd::Complex value) {
    const float amplitude = expf(fminf(2.0f * value.im, 80.0f));
    float sine;
    float cosine;
    sincosf(2.0f * value.re, &sine, &cosine);
    return utd::cplx(amplitude * cosine, -amplitude * sine);
}

// Frozen Sionna radiomap slab arithmetic. This intentionally preserves the
// original hypotf/sqrtf evaluation order and denominator floor used by the MC
// reflection estimator; changing it shifts seeded FP32 radiomap fingerprints.
struct LegacySlabComplex {
    float re;
    float im;
};

__device__ __forceinline__ LegacySlabComplex legacy_add(
    LegacySlabComplex a, LegacySlabComplex b) {
    return {a.re + b.re, a.im + b.im};
}

__device__ __forceinline__ LegacySlabComplex legacy_sub(
    LegacySlabComplex a, LegacySlabComplex b) {
    return {a.re - b.re, a.im - b.im};
}

__device__ __forceinline__ LegacySlabComplex legacy_mul(
    LegacySlabComplex a, LegacySlabComplex b) {
    return {a.re * b.re - a.im * b.im, a.re * b.im + a.im * b.re};
}

__device__ __forceinline__ LegacySlabComplex legacy_scale(
    LegacySlabComplex a, float value) {
    return {a.re * value, a.im * value};
}

__device__ __forceinline__ LegacySlabComplex legacy_div(
    LegacySlabComplex a, LegacySlabComplex b) {
    const float denominator = fmaxf(b.re * b.re + b.im * b.im, 1.0e-30f);
    return {
        (a.re * b.re + a.im * b.im) / denominator,
        (a.im * b.re - a.re * b.im) / denominator};
}

__device__ __forceinline__ LegacySlabComplex legacy_div_floor(
    LegacySlabComplex a, LegacySlabComplex b, float denominator_floor) {
    const float denominator = fmaxf(
        b.re * b.re + b.im * b.im, denominator_floor);
    return {
        (a.re * b.re + a.im * b.im) / denominator,
        (a.im * b.re - a.re * b.im) / denominator};
}

__device__ __forceinline__ LegacySlabComplex legacy_sqrt(
    LegacySlabComplex value) {
    const float magnitude = hypotf(value.re, value.im);
    const float real = sqrtf(fmaxf(0.0f, 0.5f * (magnitude + value.re)));
    const float imag = copysignf(
        sqrtf(fmaxf(0.0f, 0.5f * (magnitude - value.re))), value.im);
    return {real, imag};
}

__device__ __forceinline__ LegacySlabComplex legacy_interface_sqrt(
    LegacySlabComplex value) {
    const float magnitude = hypotf(value.re, value.im);
    const float real = sqrtf(fmaxf(0.0f, 0.5f * (magnitude + value.re)));
    const float imag_sign = value.im < 0.0f ? -1.0f : 1.0f;
    const float imag = imag_sign * sqrtf(
        fmaxf(0.0f, 0.5f * (magnitude - value.re)));
    return {real, imag};
}

__device__ __forceinline__ void legacy_interface_fresnel(
    float cos_theta_in,
    float eps_r,
    float sigma_e,
    float mu_r,
    float omega,
    float vacuum_permittivity,
    float epsilon,
    utd::Complex& r_te,
    utd::Complex& r_tm) {
    const float cos_theta = fminf(fmaxf(cos_theta_in, epsilon), 1.0f);
    const float sin2 = fmaxf(0.0f, 1.0f - cos_theta * cos_theta);
    const LegacySlabComplex eta = {
        eps_r, -sigma_e / (omega * vacuum_permittivity)};
    const LegacySlabComplex root = legacy_interface_sqrt(
        legacy_sub(legacy_scale(eta, mu_r), {sin2, 0.0f}));
    const LegacySlabComplex mu_cos = {mu_r * cos_theta, 0.0f};
    const LegacySlabComplex eta_cos = legacy_scale(eta, cos_theta);
    const LegacySlabComplex result_te = legacy_div_floor(
        legacy_sub(mu_cos, root), legacy_add(mu_cos, root), epsilon);
    const LegacySlabComplex result_tm = legacy_div_floor(
        legacy_sub(eta_cos, root), legacy_add(eta_cos, root), epsilon);
    r_te = utd::cplx(result_te.re, result_te.im);
    r_tm = utd::cplx(result_tm.re, result_tm.im);
}

__device__ __forceinline__ LegacySlabComplex legacy_exp_neg_2i(
    LegacySlabComplex value) {
    const float amplitude = expf(fminf(2.0f * value.im, 80.0f));
    float sine;
    float cosine;
    sincosf(2.0f * value.re, &sine, &cosine);
    return {amplitude * cosine, -amplitude * sine};
}

__device__ __forceinline__ void legacy_sionna_slab_fresnel(
    float cos_theta,
    float eps_r,
    float sigma_e,
    float gain,
    float thickness,
    float wavelength,
    utd::Complex& r_te,
    utd::Complex& r_tm) {
    const float omega = 2.0f * utd::UTD_PI * kSpeedOfLight /
                        fmaxf(wavelength, utd::UTD_SMALL_EPS);
    const LegacySlabComplex eta = {
        fmaxf(eps_r, utd::UTD_SMALL_EPS),
        -fmaxf(sigma_e, 0.0f) / (omega * utd::UTD_EPSILON_0)};
    const float ct = fminf(
        fmaxf(fabsf(cos_theta), utd::UTD_SMALL_EPS), 1.0f);
    const float sin2 = fmaxf(0.0f, 1.0f - ct * ct);
    const LegacySlabComplex root = legacy_sqrt(
        legacy_sub(eta, {sin2, 0.0f}));
    const LegacySlabComplex ct_complex = {ct, 0.0f};
    const LegacySlabComplex eta_ct = legacy_scale(eta, ct);
    const LegacySlabComplex interface_te = legacy_div(
        legacy_sub(ct_complex, root), legacy_add(ct_complex, root));
    const LegacySlabComplex interface_tm = legacy_div(
        legacy_sub(eta_ct, root), legacy_add(eta_ct, root));
    const LegacySlabComplex q = legacy_scale(
        root,
        2.0f * utd::UTD_PI * fmaxf(thickness, 0.0f) /
            fmaxf(wavelength, utd::UTD_SMALL_EPS));
    const LegacySlabComplex phase = legacy_exp_neg_2i(q);
    const LegacySlabComplex one = {1.0f, 0.0f};
    const LegacySlabComplex numerator = legacy_sub(one, phase);
    const LegacySlabComplex result_te = legacy_scale(
        legacy_div(
            legacy_mul(interface_te, numerator),
            legacy_sub(
                one,
                legacy_mul(legacy_mul(interface_te, interface_te), phase))),
        gain);
    const LegacySlabComplex result_tm = legacy_scale(
        legacy_div(
            legacy_mul(interface_tm, numerator),
            legacy_sub(
                one,
                legacy_mul(legacy_mul(interface_tm, interface_tm), phase))),
        gain);
    r_te = utd::cplx(result_te.re, result_te.im);
    r_tm = utd::cplx(result_tm.re, result_tm.im);
}

__device__ __forceinline__ void slab_fresnel(
    float cos_theta,
    float eps_r,
    float sigma_e,
    float mu_r,
    float gain,
    float thickness,
    float frequency_hz,
    utd::Complex& r_te,
    utd::Complex& r_tm) {
    const float omega = fmaxf(
        2.0f * utd::UTD_PI * frequency_hz, utd::UTD_SMALL_EPS);
    const float wavelength = kSpeedOfLight / frequency_hz;
    const float ct = fminf(fmaxf(fabsf(cos_theta), utd::UTD_SMALL_EPS), 1.0f);
    const float sin2 = fmaxf(0.0f, 1.0f - ct * ct);
    const utd::Complex eta = utd::cplx(
        fmaxf(eps_r, utd::UTD_SMALL_EPS),
        -fmaxf(sigma_e, 0.0f) / (omega * utd::UTD_EPSILON_0));
    const float mu = fmaxf(mu_r, utd::UTD_SMALL_EPS);
    const utd::Complex root = utd::cplx_sqrt(
        utd::cplx_sub(utd::cplx_mul_real(eta, mu), utd::cplx(sin2, 0.0f)));
    const utd::Complex mu_ct = utd::cplx(mu * ct, 0.0f);
    const utd::Complex eta_ct = utd::cplx_mul_real(eta, ct);
    const utd::Complex interface_te = utd::cplx_div(
        utd::cplx_sub(mu_ct, root), utd::cplx_add(mu_ct, root));
    const utd::Complex interface_tm = utd::cplx_div(
        utd::cplx_sub(eta_ct, root), utd::cplx_add(eta_ct, root));
    const utd::Complex q = utd::cplx_mul_real(
        root,
        2.0f * utd::UTD_PI * fmaxf(thickness, 0.0f) /
            fmaxf(wavelength, utd::UTD_SMALL_EPS));
    const utd::Complex phase = exp_neg_2i(q);
    const utd::Complex one = utd::cplx(1.0f, 0.0f);
    const utd::Complex numerator = utd::cplx_sub(one, phase);
    r_te = utd::cplx_mul_real(
        utd::cplx_div(
            utd::cplx_mul(interface_te, numerator),
            utd::cplx_sub(
                one,
                utd::cplx_mul(utd::cplx_mul(interface_te, interface_te), phase))),
        gain);
    r_tm = utd::cplx_mul_real(
        utd::cplx_div(
            utd::cplx_mul(interface_tm, numerator),
            utd::cplx_sub(
                one,
                utd::cplx_mul(utd::cplx_mul(interface_tm, interface_tm), phase))),
        gain);
}

__device__ __forceinline__ utd::Complex complex3_dot_real(
    utd::Complex3 value, utd::float3a axis) {
    return utd::cplx_add(
        utd::cplx_add(
            utd::cplx_mul_real(value.x, axis.x),
            utd::cplx_mul_real(value.y, axis.y)),
        utd::cplx_mul_real(value.z, axis.z));
}

__device__ __forceinline__ utd::Complex3 reflect_complex3(
    utd::Complex3 value,
    utd::float3a incident_direction,
    utd::float3a normal,
    float eps_r,
    float sigma_e,
    float mu_r,
    float gain,
    float thickness,
    float frequency_hz,
    utd::float3a& reflected_direction) {
    const utd::float3a incident = utd::safe_normalize(
        incident_direction, utd::make_f3(0.0f, 0.0f, 1.0f));
    utd::float3a oriented_normal = utd::safe_normalize(
        normal, utd::make_f3(0.0f, 0.0f, 1.0f));
    if (utd::f3_dot(incident, oriented_normal) > 0.0f)
        oriented_normal = utd::f3_neg(oriented_normal);
    const float dot_in = utd::f3_dot(incident, oriented_normal);
    reflected_direction = utd::safe_normalize(
        utd::f3_sub(incident, utd::f3_mul(oriented_normal, 2.0f * dot_in)),
        utd::f3_neg(incident));
    utd::float3a s_axis = utd::f3_cross(oriented_normal, incident);
    s_axis = utd::safe_normalize(
        s_axis, utd::stable_perp_basis(incident, oriented_normal));
    const utd::float3a p_in = utd::safe_normalize(
        utd::f3_cross(s_axis, incident),
        utd::stable_perp_basis(incident, s_axis));
    const utd::float3a p_out = utd::safe_normalize(
        utd::f3_cross(s_axis, reflected_direction),
        utd::stable_perp_basis(reflected_direction, s_axis));
    utd::Complex r_te;
    utd::Complex r_tm;
    slab_fresnel(
        fabsf(dot_in),
        eps_r,
        sigma_e,
        mu_r,
        gain,
        thickness,
        frequency_hz,
        r_te,
        r_tm);
    const utd::Complex e_s = complex3_dot_real(value, s_axis);
    const utd::Complex e_p = complex3_dot_real(value, p_in);
    return utd::c3_add(
        utd::cplx_scale_real(s_axis, utd::cplx_mul(r_te, e_s)),
        utd::cplx_scale_real(p_out, utd::cplx_mul(r_tm, e_p)));
}

__device__ __forceinline__ utd::JonesOperator slab_face_operator(
    float cos_theta,
    float eps_r,
    float sigma_e,
    float mu_r,
    float gain,
    float thickness,
    float frequency_hz,
    utd::float3a normal,
    utd::float3a in_direction,
    utd::float3a out_direction,
    utd::Basis3 in_edge,
    utd::Basis3 out_edge) {
    utd::Complex r_te;
    utd::Complex r_tm;
    slab_fresnel(
        cos_theta,
        eps_r,
        sigma_e,
        mu_r,
        gain,
        thickness,
        frequency_hz,
        r_te,
        r_tm);
    const utd::JonesOperator diagonal = {
        r_te,
        utd::cplx_zero(),
        utd::cplx_zero(),
        r_tm,
    };
    const utd::float3a face_in = utd::f3_cross(normal, in_direction);
    const utd::float3a raw_out = utd::f3_cross(normal, out_direction);
    const utd::float3a reference = utd::stable_perp_basis(out_direction, face_in);
    const utd::float3a face_out =
        utd::f3_dot(raw_out, reference) < 0.0f ? utd::f3_neg(raw_out) : raw_out;
    const utd::Basis3 input_basis = utd::basis_from_first_vector(
        in_direction,
        face_in,
        utd::stable_perp_basis(in_direction, utd::make_f3(0.0f, 0.0f, 1.0f)));
    const utd::Basis3 output_basis = utd::basis_from_first_vector(
        out_direction, face_out, reference);
    return utd::jop_in_basis(
        diagonal, input_basis, output_basis, in_edge, out_edge);
}

}  // namespace channel_native::field_transport
