// Copyright Xingyu Chen.
// Face reflection coefficients of single-slab and multilayer materials for the reflection kernels.
#pragma once

#include <rayd/field_transport.cuh>
#include <src/field_transport_ad.cuh>

namespace channel::layered_reflection {

namespace utd = ::rayd::shared::diffraction;
namespace em = ::rayd::shared::transmission;
namespace transport = ::rayd::shared::field_transport;
namespace ad = ::rayd::torch::field_transport_ad;

// CSR layer store of the compiled materials (ABI v3). A null `count` means the caller has no layer
// store; every face then uses its scalar single-slab view.
struct LayerTables {
    const int* offset = nullptr;
    const int* count = nullptr;
    const float* thickness_m = nullptr;
    const float* eps_r = nullptr;
    const float* sigma_e = nullptr;
    const float* mu_r = nullptr;
    int64_t material_count = 0;
};

// A material with more than one layer reflects as its whole stack; a single-layer material keeps the
// scalar slab model (the same physics, evaluated by the established single-slab expression).
__device__ __forceinline__ bool multilayer(const LayerTables& tables, int material) {
    return tables.count != nullptr && material >= 0 && material < tables.material_count &&
           tables.count[material] > 1;
}

__device__ __forceinline__ em::LayerView layer_view(const LayerTables& tables, int material) {
    return {tables.offset, tables.count, tables.thickness_m, tables.eps_r, tables.sigma_e, tables.mu_r, material};
}

// Reflection coefficients in the s / p = s x k convention of transport::slab_fresnel. The stack
// recursion reports the TM coefficient as a tangential-E ratio, the negative of that convention.
__device__ __forceinline__ void face_fresnel(float cos_theta, float eps_r, float sigma_e, float mu_r, float gain,
                                             float thickness, float frequency_hz, int material,
                                             const LayerTables& tables, utd::Complex& r_te, utd::Complex& r_tm) {
    if (!multilayer(tables, material)) {
        transport::slab_fresnel(cos_theta, eps_r, sigma_e, mu_r, gain, thickness, frequency_hz, r_te, r_tm);
        return;
    }
    const em::LayerView view = layer_view(tables, material);
    const em::StackRT te = em::stack_rt(cos_theta, view, frequency_hz, em::kPolTE);
    const em::StackRT tm = em::stack_rt(cos_theta, view, frequency_hz, em::kPolTM);
    r_te = utd::cplx_mul_real(te.r, gain);
    r_tm = utd::cplx_mul_real(tm.r, -gain);
}

struct ZeroLayerSeed {
    __device__ ad::LayerSeed operator()(int) const { return {0.0f, 0.0f, 0.0f}; }
};

// Forward-mode twin of face_fresnel. A multilayer face carries the incidence-angle, gain and
// frequency tangents; its layer parameters are not differentiable here, and callers refuse
// material derivatives on multilayer faces before launching.
__device__ __forceinline__ void face_fresnel_dual(float cos_theta, float eps_r, float sigma_e, float mu_r, float gain,
                                                  float thickness, float frequency_hz, int material,
                                                  const LayerTables& tables, float d_cos_theta, float d_eps,
                                                  float d_sigma, float d_gain, float d_thickness, float d_frequency,
                                                  ad::DualC& r_te, ad::DualC& r_tm) {
    if (!multilayer(tables, material)) {
        ad::slab_fresnel_dual(cos_theta, eps_r, sigma_e, mu_r, gain, thickness, frequency_hz, d_cos_theta, d_eps,
                              d_sigma, d_gain, d_thickness, d_frequency, r_te, r_tm);
        return;
    }
    const em::LayerView view = layer_view(tables, material);
    const ad::DualStackRT te =
        ad::stack_rt_dual(cos_theta, view, frequency_hz, d_cos_theta, d_frequency, em::kPolTE, ZeroLayerSeed{});
    const ad::DualStackRT tm =
        ad::stack_rt_dual(cos_theta, view, frequency_hz, d_cos_theta, d_frequency, em::kPolTM, ZeroLayerSeed{});
    r_te.v = utd::cplx_mul_real(te.r.v, gain);
    r_te.d = utd::cplx_add(utd::cplx_mul_real(te.r.d, gain), utd::cplx_mul_real(te.r.v, d_gain));
    r_tm.v = utd::cplx_mul_real(tm.r.v, -gain);
    r_tm.d = utd::cplx_add(utd::cplx_mul_real(tm.r.d, -gain), utd::cplx_mul_real(tm.r.v, -d_gain));
}

// Wavelength-parameterised twins for the Monte Carlo reflection family, whose single-slab law is
// transport::legacy_slab_fresnel. A multilayer face again reflects as its whole stack.
__device__ __forceinline__ void legacy_face_fresnel(float cos_theta, float eps_r, float sigma_e, float gain,
                                                    float thickness, float wavelength, int material,
                                                    const LayerTables& tables, utd::Complex& r_te,
                                                    utd::Complex& r_tm) {
    if (!multilayer(tables, material)) {
        transport::legacy_slab_fresnel(cos_theta, eps_r, sigma_e, gain, thickness, wavelength, r_te, r_tm);
        return;
    }
    face_fresnel(cos_theta, eps_r, sigma_e, 1.0f, gain, thickness,
                 transport::kSpeedOfLight / fmaxf(wavelength, utd::UTD_SMALL_EPS), material, tables, r_te, r_tm);
}

// Tangents of a multilayer face: gain and wavelength only (material tangents are refused upstream).
__device__ __forceinline__ void legacy_face_fresnel_dual(float cos_theta, float eps_r, float sigma_e, float gain,
                                                         float thickness, float wavelength, int material,
                                                         const LayerTables& tables, float d_eps, float d_sigma,
                                                         float d_gain, float d_thickness, float d_wavelength,
                                                         ad::DualC& r_te, ad::DualC& r_tm) {
    if (!multilayer(tables, material)) {
        ad::legacy_slab_fresnel_dual(cos_theta, eps_r, sigma_e, gain, thickness, wavelength, d_eps, d_sigma, d_gain,
                                     d_thickness, d_wavelength, r_te, r_tm);
        return;
    }
    const float lambda = fmaxf(wavelength, utd::UTD_SMALL_EPS);
    const float frequency = transport::kSpeedOfLight / lambda;
    const float d_frequency = wavelength > utd::UTD_SMALL_EPS ? -frequency / lambda * d_wavelength : 0.0f;
    face_fresnel_dual(cos_theta, eps_r, sigma_e, 1.0f, gain, thickness, frequency, material, tables, 0.0f, 0.0f,
                      0.0f, d_gain, 0.0f, d_frequency, r_te, r_tm);
}

// Multi-tangent form (tangent k of each array), matching ad::legacy_slab_fresnel_dual_multi.
template <int N>
__device__ __forceinline__ void legacy_face_fresnel_dual_multi(
    float cos_theta, float eps_r, float sigma_e, float gain, float thickness, float wavelength, int material,
    const LayerTables& tables, const float* d_eps, const float* d_sigma, const float* d_gain,
    const float* d_thickness, const float* d_wavelength, utd::Complex& r_te, utd::Complex& r_tm,
    utd::Complex* d_r_te, utd::Complex* d_r_tm) {
    if (!multilayer(tables, material)) {
        ad::legacy_slab_fresnel_dual_multi<N>(cos_theta, eps_r, sigma_e, gain, thickness, wavelength, d_eps, d_sigma,
                                              d_gain, d_thickness, d_wavelength, r_te, r_tm, d_r_te, d_r_tm);
        return;
    }
    for (int k = 0; k < N; ++k) {
        ad::DualC te, tm;
        legacy_face_fresnel_dual(cos_theta, eps_r, sigma_e, gain, thickness, wavelength, material, tables, 0.0f,
                                 0.0f, d_gain[k], 0.0f, d_wavelength[k], te, tm);
        r_te = te.v;
        r_tm = tm.v;
        d_r_te[k] = te.d;
        d_r_tm[k] = tm.d;
    }
}

// transport::reflect_complex3 with the layered coefficients.
__device__ __forceinline__ utd::Complex3 reflect_layered(utd::Complex3 value, utd::float3a incident_direction,
                                                         utd::float3a normal, float eps_r, float sigma_e, float mu_r,
                                                         float gain, float thickness, float frequency_hz, int material,
                                                         const LayerTables& tables,
                                                         utd::float3a& reflected_direction) {
    const transport::ReflectFrame frame = transport::reflect_frame(incident_direction, normal);
    reflected_direction = frame.reflected_direction;
    utd::Complex r_te;
    utd::Complex r_tm;
    face_fresnel(frame.cos_theta, eps_r, sigma_e, mu_r, gain, thickness, frequency_hz, material, tables, r_te, r_tm);
    const utd::Complex e_s = transport::complex3_dot_real(value, frame.s_axis);
    const utd::Complex e_p = transport::complex3_dot_real(value, frame.p_in);
    return utd::c3_add(utd::cplx_scale_real(frame.s_axis, utd::cplx_mul(r_te, e_s)),
                       utd::cplx_scale_real(frame.p_out, utd::cplx_mul(r_tm, e_p)));
}

// Reflection coefficients of the two faces of a diffraction wedge, for the pair evaluation
// (utd::compute_pair_vector_contribution_faces): each face reflects in the diffraction face term with the
// coefficient its specular reflection uses, a finite slab or the layer stack of a multilayer material
// (`material` < 0 or a null layer store: the slab). Faces whose FaceMaterialParams are absent keep the stored
// operator. The float form is the primal; the Dual form carries the incidence-angle, slab-parameter, gain and
// frequency tangents.
template <typename T> struct WedgeFaceFresnel;

template <> struct WedgeFaceFresnel<float> {
    float eps[2];
    float sigma[2];
    float mu[2];
    float gain[2];
    float thickness[2];
    int material[2] = {-1, -1};
    LayerTables tables;
    float frequency_hz;

    __device__ __forceinline__ bool evaluates(int, const utd::FaceMaterialParams& face) const {
        return face.present > 0.5f;
    }
    __device__ __forceinline__ utd::FaceFresnelT<float> operator()(int face, const utd::FaceMaterialParams&,
                                                                   float cos_theta) const {
        utd::FaceFresnelT<float> out;
        face_fresnel(cos_theta, eps[face], sigma[face], mu[face], gain[face], thickness[face], frequency_hz,
                     material[face], tables, out.rTE, out.rTM);
        return out;
    }
};

__device__ __forceinline__ utd::ComplexT<utd::Dual> dual_complex(const ad::DualC& value) {
    return {utd::Dual(value.v.re, value.d.re), utd::Dual(value.v.im, value.d.im)};
}

template <> struct WedgeFaceFresnel<utd::Dual> {
    utd::Dual eps[2];
    utd::Dual sigma[2];
    float mu[2];
    utd::Dual gain[2];
    utd::Dual thickness[2];
    int material[2] = {-1, -1};
    LayerTables tables;
    utd::Dual frequency_hz;

    __device__ __forceinline__ bool evaluates(int, const utd::FaceMaterialParamsT<utd::Dual>& face) const {
        return face.present > 0.5f;
    }
    __device__ __forceinline__ utd::FaceFresnelT<utd::Dual> operator()(int face,
                                                                       const utd::FaceMaterialParamsT<utd::Dual>&,
                                                                       utd::Dual cos_theta) const {
        ad::DualC r_te, r_tm;
        face_fresnel_dual(cos_theta.v, eps[face].v, sigma[face].v, mu[face], gain[face].v, thickness[face].v,
                          frequency_hz.v, material[face], tables, cos_theta.d, eps[face].d, sigma[face].d,
                          gain[face].d, thickness[face].d, frequency_hz.d, r_te, r_tm);
        return {dual_complex(r_te), dual_complex(r_tm)};
    }
};

} // namespace channel::layered_reflection
