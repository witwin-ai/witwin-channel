// Copyright Xingyu Chen.
// Implements diffraction CUDA operations.

// ==== Section: Diffraction kernels ====
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime_api.h>
#include <rayd/shared/utd/utd_math.h>

#include "../tensor_checks.h"
#include "math.cuh"
#include <rayd/shared/rf/field_transport.cuh>
#include <rayd/torch/rf/field_transport_ad.cuh>
#include <algorithm>
#include <cstdint>
#include <vector>

#define CHANNEL_DIFFRACTION_CHECK_STATE_PACK_TENSORS()                                     \
    check_tensor(edge_pos, "edge_pos", at::kFloat, 2);                               \
    check_tensor(edge_dir, "edge_dir", at::kFloat, 2);                               \
    check_tensor(line_min, "line_min", at::kFloat, 1);                               \
    check_tensor(line_max, "line_max", at::kFloat, 1);                               \
    check_tensor(n0, "n0", at::kFloat, 2);                                           \
    check_tensor(n1, "n1", at::kFloat, 2);                                           \
    check_tensor(face0, "face0", at::kInt, 1);                                       \
    check_tensor(face1, "face1", at::kInt, 1);                                       \
    check_tensor(exterior_angle, "exterior_angle", at::kFloat, 1);                   \
    check_tensor(tx, "tx", at::kFloat, 1)

#define CHANNEL_DIFFRACTION_CHECK_STATE_PACK_POWER()                                       \
    TORCH_CHECK(tx_power.is_cuda(), "tx_power must be a CUDA tensor");                \
    TORCH_CHECK(tx_power.scalar_type() == at::kFloat, "tx_power has the wrong dtype");\
    TORCH_CHECK(tx_power.is_contiguous(), "tx_power must be contiguous");             \
    TORCH_CHECK(tx_power.dim() == 0 || tx_power.dim() == 1, "tx_power must be scalar or 1-D");\
    if (tx_power.dim() == 0) {                                                         \
        TORCH_CHECK(tx_power_index == 0, "tx_power_index must be zero for scalar tx_power");\
    } else {                                                                           \
        TORCH_CHECK(tx_power_index >= 0 && tx_power_index < tx_power.size(0), "tx_power_index is out of range");\
    }

#define CHANNEL_DIFFRACTION_CHECK_STATE_PACK_SHAPES()                                      \
    TORCH_CHECK(edge_pos.size(1) == 3, "edge_pos must have shape (N, 3)");            \
    TORCH_CHECK(edge_dir.sizes() == edge_pos.sizes(), "edge_dir must match edge_pos");\
    TORCH_CHECK(n0.sizes() == edge_pos.sizes(), "n0 must match edge_pos");            \
    TORCH_CHECK(n1.sizes() == edge_pos.sizes(), "n1 must match edge_pos");            \
    TORCH_CHECK(line_min.size(0) == edge_pos.size(0), "line_min must match edge count");\
    TORCH_CHECK(line_max.size(0) == edge_pos.size(0), "line_max must match edge count");\
    TORCH_CHECK(face0.size(0) == edge_pos.size(0), "face0 must match edge count");    \
    TORCH_CHECK(face1.size(0) == edge_pos.size(0), "face1 must match edge count");    \
    TORCH_CHECK(exterior_angle.size(0) == edge_pos.size(0), "exterior_angle must match edge count")

#define CHANNEL_DIFFRACTION_ALLOCATE_STATE_PACK()                                          \
    auto state_edge_index = at::empty({state_count}, int_options);                    \
    auto state_edge_pos = at::empty({state_count, 3}, float_options);                  \
    auto state_edge_dir = at::empty({state_count, 3}, float_options);                  \
    auto state_line_min = at::empty({state_count}, float_options);                     \
    auto state_line_max = at::empty({state_count}, float_options);                     \
    auto state_n0 = at::empty({state_count, 3}, float_options);                        \
    auto state_n1 = at::empty({state_count, 3}, float_options);                        \
    auto state_face0 = at::empty({state_count}, int_options);                         \
    auto state_face1 = at::empty({state_count}, int_options);                         \
    auto state_exterior_angle = at::empty({state_count}, float_options);               \
    auto state_src = at::empty({state_count, 3}, float_options);                       \
    auto state_src_power = at::empty({state_count}, float_options)

#define CHANNEL_DIFFRACTION_STATE_PACK_INPUT_POINTERS()                                    \
    edge_pos.data_ptr<float>(),                                                       \
    edge_dir.data_ptr<float>(),                                                       \
    line_min.data_ptr<float>(),                                                       \
    line_max.data_ptr<float>(),                                                       \
    n0.data_ptr<float>(),                                                             \
    n1.data_ptr<float>(),                                                             \
    face0.data_ptr<int>(),                                                            \
    face1.data_ptr<int>(),                                                            \
    exterior_angle.data_ptr<float>(),                                                 \
    tx.data_ptr<float>(),                                                             \
    tx_power.data_ptr<float>() + tx_power_index

#define CHANNEL_DIFFRACTION_STATE_PACK_OUTPUT_POINTERS()                                   \
    state_edge_index.data_ptr<int>(),                                                 \
    state_edge_pos.data_ptr<float>(),                                                 \
    state_edge_dir.data_ptr<float>(),                                                 \
    state_line_min.data_ptr<float>(),                                                 \
    state_line_max.data_ptr<float>(),                                                 \
    state_n0.data_ptr<float>(),                                                       \
    state_n1.data_ptr<float>(),                                                       \
    state_face0.data_ptr<int>(),                                                      \
    state_face1.data_ptr<int>(),                                                      \
    state_exterior_angle.data_ptr<float>(),                                           \
    state_src.data_ptr<float>(),                                                      \
    state_src_power.data_ptr<float>()

#define CHANNEL_DIFFRACTION_STATE_PACK_RESULTS()                                           \
    state_edge_index,                                                                 \
    state_edge_pos,                                                                   \
    state_edge_dir,                                                                   \
    state_line_min,                                                                   \
    state_line_max,                                                                   \
    state_n0,                                                                         \
    state_n1,                                                                         \
    state_face0,                                                                      \
    state_face1,                                                                      \
    state_exterior_angle,                                                             \
    state_src,                                                                        \
    state_src_power

namespace {

constexpr int kDiffractionBlockSize = 256;
namespace utd = rayd::shared::utd;
namespace transport = rayd::shared::rf::field_transport;
namespace fad = rayd::torch::rf::field_transport_ad;

__device__ __forceinline__ unsigned int dfr_hash(unsigned int x) {
    x^=x>>16; x*=0x7feb352du; x^=x>>15; x*=0x846ca68bu; x^=x>>16; return x;
}
__device__ __forceinline__ float dfr_uniform(unsigned int lane,unsigned int stream,unsigned int seed) {
    const unsigned int h=dfr_hash(lane^(stream*0x9e3779b9u)^seed);
    return static_cast<float>(h&0x00ffffffu)*(1.0f/16777216.0f);
}

__device__ __forceinline__ utd::JonesOperator slab_face_operator(
    float ct,float er,float sg,float gain,float thickness,float wavelength,
    utd::float3a normal,utd::float3a in_hat,utd::float3a out_hat,
    utd::Basis3 in_edge,utd::Basis3 out_edge) {
    return transport::slab_face_operator(
        ct, er, sg, 1.0f, gain, thickness,
        transport::kSpeedOfLight / wavelength,
        normal, in_hat, out_hat, in_edge, out_edge);
}

// Templated twin of the local slab_face_operator: the float instantiation IS
// the primal operator; the Dual instantiation is the validated AD slab
// response on RayD duals (field_transport_ad.cuh).
template <typename T>
__device__ __forceinline__ utd::JonesOperatorT<T> slab_face_operator_t(
    T ct, T er, T sg, T gain, T thickness, T wavelength,
    utd::Vec3T<T> normal, utd::Vec3T<T> in_hat, utd::Vec3T<T> out_hat,
    utd::Basis3T<T> in_edge, utd::Basis3T<T> out_edge);
template <>
__device__ __forceinline__ utd::JonesOperatorT<float> slab_face_operator_t<float>(
    float ct, float er, float sg, float gain, float thickness, float wavelength,
    utd::Vec3T<float> normal, utd::Vec3T<float> in_hat, utd::Vec3T<float> out_hat,
    utd::Basis3T<float> in_edge, utd::Basis3T<float> out_edge) {
    return slab_face_operator(
        ct, er, sg, gain, thickness, wavelength, normal, in_hat, out_hat,
        in_edge, out_edge);
}
template <>
__device__ __forceinline__ utd::JonesOperatorT<utd::Dual> slab_face_operator_t<utd::Dual>(
    utd::Dual ct, utd::Dual er, utd::Dual sg, utd::Dual gain, utd::Dual thickness,
    utd::Dual wavelength, utd::Vec3T<utd::Dual> normal,
    utd::Vec3T<utd::Dual> in_hat, utd::Vec3T<utd::Dual> out_hat,
    utd::Basis3T<utd::Dual> in_edge, utd::Basis3T<utd::Dual> out_edge) {
    const utd::Dual frequency = transport::kSpeedOfLight / wavelength;
    return fad::slab_face_operator_dual(
        ct, er, sg, 1.0f, gain, thickness, frequency, normal, in_hat, out_hat,
        in_edge, out_edge);
}

__device__ __forceinline__ utd::float3a load_utd3(const float *p, int i) {
    return utd::make_f3(p[3*i], p[3*i+1], p[3*i+2]);
}

__device__ __forceinline__ float component_utd(utd::float3a v, int axis) {
    return axis == 0 ? v.x : (axis == 1 ? v.y : v.z);
}

template <typename T>
__device__ __forceinline__ T component_t(utd::Vec3T<T> v, int axis) {
    return axis == 0 ? v.x : (axis == 1 ? v.y : v.z);
}

// Truncation-factor freeze (diffraction AD policy, same as the coupled row in
// field_wedge_ad.cu): the tape rows evaluate the pair with +/-1e5
// pseudo-infinite edge bounds, where the Boersma endpoint ripple makes the
// factor's derivative float32 noise amplified by the 1e5 lever arm (the true
// infinite-edge derivative is zero). No-op for the float instantiation.
template <typename T>
__device__ __forceinline__ void freeze_complex_tangent(utd::ComplexT<T>&) {}
template <>
__device__ __forceinline__ void freeze_complex_tangent<utd::Dual>(
    utd::ComplexT<utd::Dual>& value) {
    value.re.d = 0.f;
    value.im.d = 0.f;
}

// ---------------------------------------------------------------------------
// Templated per-lane row of the UTD diffraction tape accumulator (the AD contract
// AD). The float instantiation is the primal deposit computed by
// utd_diffraction_tape_accumulate_kernel below; the Dual instantiation
// carries an exact directional derivative through the recomputed Keller-cone
// geometry, the incident spherical wave, the stored slab face operators and
// the UTD pair (fixed-point + stored-ops convention: selectStationaryPoint =
// 0, mat.omega = 0 at the pair call). Frozen winners: the RayD sampling tape
// (active / state / cell / u), the per-lane cone azimuth, the edge tables
// and the deposit binning. Differentiable: the per-face slab materials, the
// state source (transmitter) and the wavelength (chained to frequency by the
// host layer).
// ---------------------------------------------------------------------------

struct TapeRowContext {
    utd::float3a edge_pos;
    utd::float3a edge_dir_raw;
    float t_min;
    float t_max;
    utd::float3a n0;
    utd::float3a nn;
    float exterior_angle;
    int prim0;
    int prim1;
    bool valid0;
    bool valid1;
    float eps0, mu0, sigma0, gain0, thick0;
    float eps1, mu1, sigma1, gain1, thick1;
    utd::float3a src;
    utd::float3a tx_pol;
    float src_power;
    float u;
    float azimuth_sa;
    float azimuth_ca;
    float wavelength;
    int axis;
    float plane;
    float cell_area;
    float edge_weight;
};

struct TapeRowSeeds {
    utd::float3a src;
    float eps0, sigma0, gain0, thick0;
    float eps1, sigma1, gain1, thick1;
    float wavelength;
};

__device__ __forceinline__ TapeRowSeeds tape_seeds_zero() {
    TapeRowSeeds seeds;
    seeds.src = utd::f3_zero();
    seeds.eps0 = 0.f; seeds.sigma0 = 0.f; seeds.gain0 = 0.f; seeds.thick0 = 0.f;
    seeds.eps1 = 0.f; seeds.sigma1 = 0.f; seeds.gain1 = 0.f; seeds.thick1 = 0.f;
    seeds.wavelength = 0.f;
    return seeds;
}

__device__ __forceinline__ TapeRowContext load_tape_row(
    int64_t lane, int sidx,
    const float *edge_pos, const float *edge_dir, const float *t_min,
    const float *t_max, const float *n0, const float *nn, const int *prim0,
    const int *prim1, const float *exterior_angle, const float *source,
    const float *source_power, const float *tape_u,
    const float *eta_r, const float *sigma, const float *mu_r,
    const float *gain, const float *thickness, const bool *material_valid,
    int64_t sample_count, int axis, float plane, float wavelength,
    float cell_area, int seed, float total_edge_length,
    const float *tx_pol) {
    TapeRowContext c;
    c.edge_pos = load_utd3(edge_pos, sidx);
    c.edge_dir_raw = load_utd3(edge_dir, sidx);
    c.t_min = t_min[sidx];
    c.t_max = t_max[sidx];
    c.n0 = load_utd3(n0, sidx);
    c.nn = load_utd3(nn, sidx);
    c.exterior_angle = exterior_angle[sidx];
    c.prim0 = prim0[sidx];
    c.prim1 = prim1[sidx];
    c.valid0 = c.prim0 >= 0 && material_valid[c.prim0];
    c.valid1 = c.prim1 >= 0 && material_valid[c.prim1];
    c.eps0 = c.valid0 ? eta_r[c.prim0] : 1.f;
    c.mu0 = c.valid0 ? mu_r[c.prim0] : 1.f;
    c.sigma0 = c.valid0 ? sigma[c.prim0] : 0.f;
    c.gain0 = c.valid0 ? gain[c.prim0] : 1.f;
    c.thick0 = c.valid0 ? thickness[c.prim0] : 0.f;
    c.eps1 = c.valid1 ? eta_r[c.prim1] : 1.f;
    c.mu1 = c.valid1 ? mu_r[c.prim1] : 1.f;
    c.sigma1 = c.valid1 ? sigma[c.prim1] : 0.f;
    c.gain1 = c.valid1 ? gain[c.prim1] : 1.f;
    c.thick1 = c.valid1 ? thickness[c.prim1] : 0.f;
    c.src = load_utd3(source, sidx);
    // the true per-transmitter polarization (launch-wide constant vector),
    // fed into direct_source_vector's incident basis in place of the fabricated
    // (0, 0, 1). All states in one launch share the same transmitter.
    c.tx_pol = utd::make_f3(tx_pol[0], tx_pol[1], tx_pol[2]);
    c.src_power = source_power[sidx];
    c.u = tape_u[lane];
    sincosf(
        2.0f * utd::UTD_PI *
            dfr_uniform(
                static_cast<unsigned int>(lane), 1u,
                static_cast<unsigned int>(seed)),
        &c.azimuth_sa, &c.azimuth_ca);
    c.wavelength = wavelength;
    c.axis = axis;
    c.plane = plane;
    c.cell_area = cell_area;
    c.edge_weight =
        total_edge_length / fmaxf(static_cast<float>(sample_count), 1.0f);
    return c;
}

template <typename T>
__device__ bool tape_row_value(
    const TapeRowContext& c, const TapeRowSeeds& seeds, T& value_out) {
    const utd::float3a eh_f =
        utd::safe_normalize(c.edge_dir_raw, utd::make_f3(0, 0, 1));
    const float length = fmaxf(c.t_max - c.t_min, 0.0f);
    const float ell = c.t_min + c.u * length;
    const utd::float3a edge_point_f =
        utd::f3_add(c.edge_pos, utd::f3_mul(eh_f, ell));
    const utd::Vec3T<T> eh = fad::const3<T>(eh_f);
    const utd::Vec3T<T> edge_point = fad::const3<T>(edge_point_f);
    const utd::Vec3T<T> src = fad::seeded3<T>(c.src, seeds.src);
    const utd::Vec3T<T> incident = utd::safe_normalize(
        utd::f3_sub(edge_point, src), utd::v3_const<T>(0, 0, 1));
    const T axial =
        utd::fminf(utd::fmaxf(utd::f3_dot(incident, eh), T(-1.f)), T(1.f));
    const T radial = utd::sqrtf(utd::fmaxf(1.0f - axial * axial, T(0.f)));
    const utd::Vec3T<T> basis0 = utd::stable_perp_basis(eh, incident);
    const utd::Vec3T<T> basis1 = utd::safe_normalize(
        utd::f3_cross(eh, basis0), utd::v3_const<T>(0, 1, 0));
    const utd::Vec3T<T> ko_exact = utd::safe_normalize(
        utd::f3_add(
            utd::f3_mul(eh, axial),
            utd::f3_mul(
                utd::f3_add(
                    utd::f3_mul(basis0, c.azimuth_ca),
                    utd::f3_mul(basis1, c.azimuth_sa)),
                radial)),
        basis0);
    const T denom = component_t(ko_exact, c.axis);
    if (fabsf(utd::scalar_value(denom)) < 1.0e-8f) return false;
    const T distance = (c.plane - component_t(edge_point, c.axis)) / denom;
    if (!(utd::scalar_value(distance) > 0.0f)) return false;
    const utd::Vec3T<T> target =
        utd::f3_add(edge_point, utd::f3_mul(ko_exact, distance));

    utd::PairInputsT<T> p{};
    p.edgePos = edge_point;
    p.edgeDir = eh;
    p.n0 = fad::const3<T>(c.n0);
    p.nn = fad::const3<T>(c.nn);
    p.wedgeN = T(c.exterior_angle / utd::UTD_PI);
    p.edgeLineMin = T(-1.0e5f);
    p.edgeLineMax = T(1.0e5f);
    p.sourcePos = src;
    p.selectStationaryPoint = 0.0f;
    const T wavelength = fad::seeded<T>(c.wavelength, seeds.wavelength);
    const T k = 2.0f * utd::UTD_PI / wavelength;
    utd::MaterialParamsT<T> mat{};
    mat.useFresnel = 1;
    mat.etaR = T(1.f);
    mat.muR = T(1.f);
    mat.sigma = T(0.f);
    mat.gain = T(1.f);
    mat.omega = 2.0f * utd::UTD_PI * 299792458.0f / wavelength;
    mat.txPolX = T(c.tx_pol.x);
    mat.txPolY = T(c.tx_pol.y);
    mat.txPolZ = T(c.tx_pol.z);
    const utd::Vec3T<T> pol =
        utd::stable_perp_basis(incident, utd::v3_const<T>(0, 0, 1));
    p.incidentBasis =
        utd::basis_from_first_vector(incident, pol, utd::v3_const<T>(1, 0, 0));
    p.incidentJones = utd::jones_from_vector(
        utd::direct_source_vector(src, edge_point, k, mat), p.incidentBasis);
    if (c.valid0) {
        p.face0Material = {
            fad::seeded<T>(c.eps0, seeds.eps0), T(c.mu0),
            fad::seeded<T>(c.sigma0, seeds.sigma0),
            fad::seeded<T>(c.gain0, seeds.gain0), 1, 1};
    } else {
        p.face0Material = {T(1.f), T(1.f), T(0.f), T(1.f), 1, 0};
    }
    if (c.valid1) {
        p.face1Material = {
            fad::seeded<T>(c.eps1, seeds.eps1), T(c.mu1),
            fad::seeded<T>(c.sigma1, seeds.sigma1),
            fad::seeded<T>(c.gain1, seeds.gain1), 1, 1};
    } else {
        p.face1Material = {T(1.f), T(1.f), T(0.f), T(1.f), 1, 0};
    }
    T gphi, gphi_p, gs, gs_p, gsb;
    utd::compute_edge_geometry_3d(
        src, edge_point, eh, p.n0, target, gphi, gphi_p, gs, gs_p, gsb);
    const utd::Basis3T<T> in_edge = utd::diffraction_edge_basis(
        utd::f3_sub(edge_point, src), eh, false);
    const utd::Basis3T<T> out_edge = utd::diffraction_edge_basis(
        utd::f3_sub(target, edge_point), eh, true);
    if (c.valid0) {
        p.face0Operator = slab_face_operator_t<T>(
            utd::fabsf(utd::sinf(gphi_p)),
            fad::seeded<T>(c.eps0, seeds.eps0),
            fad::seeded<T>(c.sigma0, seeds.sigma0),
            fad::seeded<T>(c.gain0, seeds.gain0),
            fad::seeded<T>(c.thick0, seeds.thick0), wavelength, p.n0,
            in_edge.k, out_edge.k, in_edge, out_edge);
    }
    if (c.valid1) {
        p.face1Operator = slab_face_operator_t<T>(
            utd::fabsf(utd::sinf(p.wedgeN * utd::UTD_PI - gphi)),
            fad::seeded<T>(c.eps1, seeds.eps1),
            fad::seeded<T>(c.sigma1, seeds.sigma1),
            fad::seeded<T>(c.gain1, seeds.gain1),
            fad::seeded<T>(c.thick1, seeds.thick1), wavelength, p.nn,
            in_edge.k, out_edge.k, in_edge, out_edge);
    }
    mat.omega = T(0.0f);
    // compute_pair_vector_contribution with selectStationaryPoint = 0,
    // transcribed so the pseudo-infinite truncation factor can be frozen at
    // its primal value (see freeze_complex_tangent above). The reused
    // gphi/gs/basis values are the same calls the pair would make
    // internally, so the float instantiation stays value-identical to
    // utd::compute_pair_contribution.
    const bool src_ext = utd::wedge_exterior_mask(
        utd::f3_sub(p.sourcePos, p.edgePos), p.edgeDir, p.n0, p.nn);
    if (!src_ext ||
        !(utd::scalar_value(gs_p) > utd::UTD_MIN_DISTANCE) ||
        !(utd::scalar_value(gs) > utd::UTD_MIN_DISTANCE))
        return false;
    utd::ComplexT<T> finite_factor =
        utd::finite_wedge_truncation_factor(p, target, k);
    freeze_complex_tangent<T>(finite_factor);
    const utd::Complex3T<T> vector_field = utd::compute_pair_vector_at_angles(
        p, target, k, mat, gphi, gphi_p, gs, gs_p, gsb, in_edge, out_edge,
        finite_factor);
    const T field_power = utd::cplx_abs_sqr(vector_field.x) +
                          utd::cplx_abs_sqr(vector_field.y) +
                          utd::cplx_abs_sqr(vector_field.z);
    if (!(utd::scalar_value(field_power) > 0) ||
        !::isfinite(utd::scalar_value(field_power)))
        return false;

    const utd::Vec3T<T> t0v = utd::safe_normalize(
        utd::f3_cross(p.n0, eh), utd::v3_const<T>(1, 0, 0));
    const utd::Vec3T<T> ko = ko_exact;
    T phi = utd::atan2f(utd::f3_dot(ko, p.n0), utd::f3_dot(ko, t0v));
    if (utd::scalar_value(phi) < 0.0f) phi += T(2.0f * utd::UTD_PI);
    // RayD proposes the complete Keller cone, but only the wedge exterior is
    // lit, so a lane is accepted only while its edge azimuth lies in the
    // exterior angular interval [0, exterior_angle] = [0, 2pi - interior].
    // This is rejection, not reparameterization: the proposal density stays
    // the full-cone 1/(2pi), hence the accepted sample weight below remains
    // 2pi rather than the width of the accepted interval.
    if (utd::scalar_value(phi) > c.exterior_angle) return false;
    const utd::Vec3T<T> dko = utd::f3_mul(
        utd::f3_add(
            utd::f3_mul(basis0, -c.azimuth_sa),
            utd::f3_mul(basis1, c.azimuth_ca)),
        radial);
    const utd::Vec3T<T> je = utd::f3_sub(
        eh, utd::f3_mul(ko, component_t(eh, c.axis) / denom));
    const utd::Vec3T<T> jp = utd::f3_mul(
        utd::f3_sub(dko, utd::f3_mul(ko, component_t(dko, c.axis) / denom)),
        distance);
    const T jacobian = utd::safe_length(utd::f3_cross(jp, je));
    const T value = field_power * c.src_power * jacobian *
                    (2.0f * utd::UTD_PI) * c.edge_weight /
                    fmaxf(c.cell_area, 1.0e-8f);
    if (!(utd::scalar_value(value) > 0 &&
          ::isfinite(utd::scalar_value(value))))
        return false;
    value_out = value;
    return true;
}

#define TAPE_ROW_TABLE_ARGS                                                   \
    edge_pos, edge_dir, t_min, t_max, n0, nn, prim0, prim1, exterior_angle,   \
        source, source_power, tape_u, eta_r, sigma, mu_r, gain, thickness,    \
        material_valid

__global__ void utd_diffraction_tape_accumulate_kernel(
    const bool *tape_active, const int *tape_state, const int *tape_cell, const float *tape_u,
    const float *edge_pos, const float *edge_dir, const float *t_min, const float *t_max,
    const float *n0, const float *nn, const int *prim0, const int *prim1,
    const float *exterior_angle, const float *source, const float *source_power,
    const float *eta_r, const float *sigma, const float *mu_r, const float *gain, const float *thickness,
    const bool *material_valid, float *output, int64_t sample_count, int state_count,
    int axis, float plane, float c0min, float c0max, float c1min, float c1max,
    int r0, int r1, float wavelength, float cell_area, int seed, float total_edge_length,
    const float *tx_pol) {
    (void)c0min; (void)c0max; (void)c1min; (void)c1max;
    const int64_t stride = static_cast<int64_t>(blockDim.x)*gridDim.x;
    for (int64_t lane=static_cast<int64_t>(blockIdx.x)*blockDim.x+threadIdx.x;
         lane<sample_count; lane+=stride) {
        if (!tape_active[lane]) continue;
        const int sidx=tape_state[lane], cell=tape_cell[lane];
        if (sidx<0 || sidx>=state_count || cell<0 || cell>=r0*r1) continue;
        const TapeRowContext row = load_tape_row(
            lane, sidx, TAPE_ROW_TABLE_ARGS, sample_count, axis, plane,
            wavelength, cell_area, seed, total_edge_length, tx_pol);
        float value = 0.0f;
        if (tape_row_value<float>(row, tape_seeds_zero(), value)) {
            atomicAdd(output + cell, value);
        }
    }
}

__global__ void utd_diffraction_tape_accumulate_backward_kernel(
    const bool *tape_active, const int *tape_state, const int *tape_cell, const float *tape_u,
    const float *edge_pos, const float *edge_dir, const float *t_min, const float *t_max,
    const float *n0, const float *nn, const int *prim0, const int *prim1,
    const float *exterior_angle, const float *source, const float *source_power,
    const float *eta_r, const float *sigma, const float *mu_r, const float *gain, const float *thickness,
    const bool *material_valid, const float *grad_output,
    float *grad_eta_r, float *grad_sigma, float *grad_gain, float *grad_thickness,
    float *grad_source, float *grad_frequency,
    int64_t sample_count, int state_count,
    int axis, float plane, int r0, int r1, float wavelength, float cell_area,
    int seed, float total_edge_length, float wavelength_dfreq,
    int64_t grad_stride0, int64_t grad_stride1,
    const float *tx_pol) {
    const int64_t stride = static_cast<int64_t>(blockDim.x)*gridDim.x;
    for (int64_t lane=static_cast<int64_t>(blockIdx.x)*blockDim.x+threadIdx.x;
         lane<sample_count; lane+=stride) {
        if (!tape_active[lane]) continue;
        const int sidx=tape_state[lane], cell=tape_cell[lane];
        if (sidx<0 || sidx>=state_count || cell<0 || cell>=r0*r1) continue;
        const float cotangent = grad_output[
            static_cast<int64_t>(cell / r0) * grad_stride0 +
            static_cast<int64_t>(cell % r0) * grad_stride1];
        if (cotangent == 0.0f) continue;
        const TapeRowContext row = load_tape_row(
            lane, sidx, TAPE_ROW_TABLE_ARGS, sample_count, axis, plane,
            wavelength, cell_area, seed, total_edge_length, tx_pol);
        TapeRowSeeds seeds = tape_seeds_zero();
        utd::Dual value;
        if (grad_source != nullptr) {
            float* slots[3] = {&seeds.src.x, &seeds.src.y, &seeds.src.z};
            for (int axis_index = 0; axis_index < 3; ++axis_index) {
                *slots[axis_index] = 1.f;
                const bool accepted =
                    tape_row_value<utd::Dual>(row, seeds, value);
                *slots[axis_index] = 0.f;
                if (accepted)
                    atomicAdd(grad_source + axis_index, cotangent * value.d);
            }
        }
        if (grad_eta_r != nullptr) {
            struct MaterialSlot {
                float* seed;
                float* grad;
                int prim;
            };
            MaterialSlot slots[8] = {
                {&seeds.eps0, grad_eta_r, row.prim0},
                {&seeds.sigma0, grad_sigma, row.prim0},
                {&seeds.gain0, grad_gain, row.prim0},
                {&seeds.thick0, grad_thickness, row.prim0},
                {&seeds.eps1, grad_eta_r, row.prim1},
                {&seeds.sigma1, grad_sigma, row.prim1},
                {&seeds.gain1, grad_gain, row.prim1},
                {&seeds.thick1, grad_thickness, row.prim1},
            };
            for (int slot = 0; slot < 8; ++slot) {
                const bool face_valid = slot < 4 ? row.valid0 : row.valid1;
                if (!face_valid) continue;
                *slots[slot].seed = 1.f;
                const bool accepted =
                    tape_row_value<utd::Dual>(row, seeds, value);
                *slots[slot].seed = 0.f;
                if (accepted)
                    atomicAdd(
                        slots[slot].grad + slots[slot].prim,
                        cotangent * value.d);
            }
        }
        if (grad_frequency != nullptr) {
            seeds.wavelength = wavelength_dfreq;
            const bool accepted = tape_row_value<utd::Dual>(row, seeds, value);
            seeds.wavelength = 0.f;
            if (accepted) atomicAdd(grad_frequency, cotangent * value.d);
        }
    }
}

__global__ void utd_diffraction_tape_accumulate_jvp_kernel(
    const bool *tape_active, const int *tape_state, const int *tape_cell, const float *tape_u,
    const float *edge_pos, const float *edge_dir, const float *t_min, const float *t_max,
    const float *n0, const float *nn, const int *prim0, const int *prim1,
    const float *exterior_angle, const float *source, const float *source_power,
    const float *eta_r, const float *sigma, const float *mu_r, const float *gain, const float *thickness,
    const bool *material_valid,
    const float *tangent_eta_r, const float *tangent_sigma,
    const float *tangent_gain, const float *tangent_thickness,
    const float *tangent_source, float wavelength_tangent,
    float *output_tangent,
    int64_t sample_count, int state_count,
    int axis, float plane, int r0, int r1, float wavelength, float cell_area,
    int seed, float total_edge_length,
    const float *tx_pol) {
    const int64_t stride = static_cast<int64_t>(blockDim.x)*gridDim.x;
    for (int64_t lane=static_cast<int64_t>(blockIdx.x)*blockDim.x+threadIdx.x;
         lane<sample_count; lane+=stride) {
        if (!tape_active[lane]) continue;
        const int sidx=tape_state[lane], cell=tape_cell[lane];
        if (sidx<0 || sidx>=state_count || cell<0 || cell>=r0*r1) continue;
        const TapeRowContext row = load_tape_row(
            lane, sidx, TAPE_ROW_TABLE_ARGS, sample_count, axis, plane,
            wavelength, cell_area, seed, total_edge_length, tx_pol);
        TapeRowSeeds seeds = tape_seeds_zero();
        if (tangent_source != nullptr) {
            seeds.src = utd::make_f3(
                tangent_source[0], tangent_source[1], tangent_source[2]);
        }
        if (row.valid0) {
            if (tangent_eta_r != nullptr) seeds.eps0 = tangent_eta_r[row.prim0];
            if (tangent_sigma != nullptr) seeds.sigma0 = tangent_sigma[row.prim0];
            if (tangent_gain != nullptr) seeds.gain0 = tangent_gain[row.prim0];
            if (tangent_thickness != nullptr)
                seeds.thick0 = tangent_thickness[row.prim0];
        }
        if (row.valid1) {
            if (tangent_eta_r != nullptr) seeds.eps1 = tangent_eta_r[row.prim1];
            if (tangent_sigma != nullptr) seeds.sigma1 = tangent_sigma[row.prim1];
            if (tangent_gain != nullptr) seeds.gain1 = tangent_gain[row.prim1];
            if (tangent_thickness != nullptr)
                seeds.thick1 = tangent_thickness[row.prim1];
        }
        seeds.wavelength = wavelength_tangent;
        utd::Dual value;
        if (tape_row_value<utd::Dual>(row, seeds, value)) {
            atomicAdd(output_tangent + cell, value.d);
        }
    }
}

using channel::check_tensor;

__global__ void diffraction_state_wi_kernel(
    const float *__restrict__ state_edge_pos,
    const float *__restrict__ state_src,
    float *__restrict__ state_wi,
    int64_t state_count) {
    const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (int64_t state = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         state < state_count;
         state += stride) {
        const float *edge_pos = state_edge_pos + state * 3;
        const float *src = state_src + state * 3;
        float dx = edge_pos[0] - src[0];
        float dy = edge_pos[1] - src[1];
        float dz = edge_pos[2] - src[2];
        const float norm = sqrtf(dx * dx + dy * dy + dz * dz);
        const float scale = norm > 1.0e-6f ? 1.0f / norm : 1.0e6f;

        float *out = state_wi + state * 3;
        out[0] = dx * scale;
        out[1] = dy * scale;
        out[2] = dz * scale;
    }
}

__global__ void diffraction_state_pack_kernel(
    const int *__restrict__ edge_indices,
    const float *__restrict__ edge_pos,
    const float *__restrict__ edge_dir,
    const float *__restrict__ line_min,
    const float *__restrict__ line_max,
    const float *__restrict__ n0,
    const float *__restrict__ n1,
    const int *__restrict__ face0,
    const int *__restrict__ face1,
    const float *__restrict__ exterior_angle,
    const float *__restrict__ tx,
    const float *__restrict__ tx_power,
    int *__restrict__ state_edge_index,
    float *__restrict__ state_edge_pos,
    float *__restrict__ state_edge_dir,
    float *__restrict__ state_line_min,
    float *__restrict__ state_line_max,
    float *__restrict__ state_n0,
    float *__restrict__ state_n1,
    int *__restrict__ state_face0,
    int *__restrict__ state_face1,
    float *__restrict__ state_exterior_angle,
    float *__restrict__ state_src,
    float *__restrict__ state_src_power,
    int64_t state_count) {
    const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    const float tx_x = tx[0];
    const float tx_y = tx[1];
    const float tx_z = tx[2];
    const float power = tx_power[0];
    for (int64_t state = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         state < state_count;
         state += stride) {
        const int edge = edge_indices[state];
        state_edge_index[state] = edge;
        state_line_min[state] = line_min[edge];
        state_line_max[state] = line_max[edge];
        state_face0[state] = face0[edge];
        state_face1[state] = face1[edge];
        state_exterior_angle[state] = exterior_angle[edge];
        state_src_power[state] = power;

        const int64_t edge_base = static_cast<int64_t>(edge) * 3;
        const int64_t state_base = state * 3;
        state_edge_pos[state_base + 0] = edge_pos[edge_base + 0];
        state_edge_pos[state_base + 1] = edge_pos[edge_base + 1];
        state_edge_pos[state_base + 2] = edge_pos[edge_base + 2];
        state_edge_dir[state_base + 0] = edge_dir[edge_base + 0];
        state_edge_dir[state_base + 1] = edge_dir[edge_base + 1];
        state_edge_dir[state_base + 2] = edge_dir[edge_base + 2];
        state_n0[state_base + 0] = n0[edge_base + 0];
        state_n0[state_base + 1] = n0[edge_base + 1];
        state_n0[state_base + 2] = n0[edge_base + 2];
        state_n1[state_base + 0] = n1[edge_base + 0];
        state_n1[state_base + 1] = n1[edge_base + 1];
        state_n1[state_base + 2] = n1[edge_base + 2];
        state_src[state_base + 0] = tx_x;
        state_src[state_base + 1] = tx_y;
        state_src[state_base + 2] = tx_z;
    }
}

__global__ void diffraction_state_pack_selected_kernel(
    const bool *__restrict__ selected,
    const float *__restrict__ edge_pos,
    const float *__restrict__ edge_dir,
    const float *__restrict__ line_min,
    const float *__restrict__ line_max,
    const float *__restrict__ n0,
    const float *__restrict__ n1,
    const int *__restrict__ face0,
    const int *__restrict__ face1,
    const float *__restrict__ exterior_angle,
    const float *__restrict__ tx,
    const float *__restrict__ tx_power,
    int *__restrict__ state_edge_index,
    float *__restrict__ state_edge_pos,
    float *__restrict__ state_edge_dir,
    float *__restrict__ state_line_min,
    float *__restrict__ state_line_max,
    float *__restrict__ state_n0,
    float *__restrict__ state_n1,
    int *__restrict__ state_face0,
    int *__restrict__ state_face1,
    float *__restrict__ state_exterior_angle,
    float *__restrict__ state_src,
    float *__restrict__ state_src_power,
    int64_t edge_count) {
    const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    const float tx_x = tx[0];
    const float tx_y = tx[1];
    const float tx_z = tx[2];
    const float power = tx_power[0];
    for (int64_t edge = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         edge < edge_count;
         edge += stride) {
        state_edge_index[edge] = static_cast<int>(edge);
        state_line_min[edge] = line_min[edge];
        state_line_max[edge] = line_max[edge];
        state_face0[edge] = face0[edge];
        state_face1[edge] = face1[edge];
        state_exterior_angle[edge] = exterior_angle[edge];
        state_src_power[edge] = selected[edge] ? power : 0.0f;

        const int64_t base = edge * 3;
        state_edge_pos[base + 0] = edge_pos[base + 0];
        state_edge_pos[base + 1] = edge_pos[base + 1];
        state_edge_pos[base + 2] = edge_pos[base + 2];
        state_edge_dir[base + 0] = edge_dir[base + 0];
        state_edge_dir[base + 1] = edge_dir[base + 1];
        state_edge_dir[base + 2] = edge_dir[base + 2];
        state_n0[base + 0] = n0[base + 0];
        state_n0[base + 1] = n0[base + 1];
        state_n0[base + 2] = n0[base + 2];
        state_n1[base + 0] = n1[base + 0];
        state_n1[base + 1] = n1[base + 1];
        state_n1[base + 2] = n1[base + 2];
        state_src[base + 0] = tx_x;
        state_src[base + 1] = tx_y;
        state_src[base + 2] = tx_z;
    }
}



__device__ __forceinline__ float signf_like_torch(float value) {
    return (value > 0.0f) ? 1.0f : ((value < 0.0f) ? -1.0f : 0.0f);
}

__device__ __forceinline__ float unsigned_angle(float3 a, float3 b, float3 axis) {
    const float3 cross = channel::math::cross(a, b);
    const float signed_norm = signf_like_torch(channel::math::dot_rn_xzy(cross, axis)) * channel::math::length_rn_xzy(cross);
    float angle = atan2f(signed_norm, channel::math::dot_rn_xzy(a, b));
    return angle < 0.0f ? angle + 6.28318530717958647692f : angle;
}

__device__ __forceinline__ int opposite_vertex(const int *faces, int face, int shared0, int shared1) {
    const int *tri = faces + static_cast<int64_t>(face) * 3;
    const int v0 = tri[0];
    const int v1 = tri[1];
    const int v2 = tri[2];
    if (v0 != shared0 && v0 != shared1) {
        return v0;
    }
    if (v1 != shared0 && v1 != shared1) {
        return v1;
    }
    return v2;
}

__global__ void diffraction_edge_geometry_kernel(
    const float *__restrict__ vertices,
    const int *__restrict__ faces,
    const float *__restrict__ face_normals,
    const int *__restrict__ edge_v0,
    const int *__restrict__ edge_v1,
    const int *__restrict__ face0,
    const int *__restrict__ face1,
    bool *__restrict__ selected,
    float *__restrict__ edge_pos,
    float *__restrict__ edge_dir,
    float *__restrict__ lengths,
    float *__restrict__ line_min,
    float *__restrict__ line_max,
    float *__restrict__ n0,
    float *__restrict__ n1,
    float *__restrict__ exterior_angle,
    int64_t edge_count,
    float plane_tol) {
    constexpr float edge_epsilon = 1.0e-6f;
    constexpr float normal_cos_tol = 1.0f - 1.0e-5f;
    constexpr float two_pi = 6.28318530717958647692f;

    const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (int64_t edge = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         edge < edge_count;
         edge += stride) {
        const int v0 = edge_v0[edge];
        const int v1 = edge_v1[edge];
        const int f0 = face0[edge];
        const int f1 = face1[edge];
        const bool valid0 = f0 >= 0;
        const bool valid1 = f1 >= 0;
        const bool boundary = valid0 && !valid1;
        const bool interior = valid0 && valid1;
        const int safe0 = f0 >= 0 ? f0 : 0;
        const int safe1 = f1 >= 0 ? f1 : 0;

        const float3 start = channel::math::load_vec3(vertices, v0);
        const float3 end = channel::math::load_vec3(vertices, v1);
        const float3 vector = channel::math::sub(end, start);
        const float length = fmaxf(channel::math::length_rn_xzy(vector), 1.0e-12f);
        const float3 dir = make_float3(
            __fdiv_rn(vector.x, length),
            __fdiv_rn(vector.y, length),
            __fdiv_rn(vector.z, length));
        const float half_length = 0.5f * length;

        const float3 n0_cand = channel::math::normalize_rn_xzy(channel::math::load_vec3(face_normals, safe0), edge_epsilon);
        const float3 n1_cand = channel::math::normalize_rn_xzy(channel::math::load_vec3(face_normals, safe1), edge_epsilon);
        const float3 to1 = channel::math::normalize_rn_xzy(channel::math::cross(n0_cand, dir), edge_epsilon);
        const float3 tn1 = channel::math::normalize_rn_xzy(channel::math::cross(n1_cand, dir), edge_epsilon);
        const float3 to2 = channel::math::normalize_rn_xzy(channel::math::cross(n1_cand, dir), edge_epsilon);
        const float3 tn2 = channel::math::normalize_rn_xzy(channel::math::cross(n0_cand, dir), edge_epsilon);
        const bool choose_first = unsigned_angle(to1, tn1, dir) < unsigned_angle(to2, tn2, dir);
        const float3 ordered_n0 = choose_first ? n0_cand : n1_cand;
        const float3 ordered_n1 = choose_first ? n1_cand : n0_cand;
        float3 out_n0 = interior ? ordered_n0 : n0_cand;
        float3 out_n1 = interior ? ordered_n1 : n1_cand;
        if (f1 < 0) {
            out_n1 = channel::math::scale(n0_cand, -1.0f);
        }
        const float output_normal_dot = channel::math::dot_rn_xzy(out_n0, out_n1);
        const float output_clamped_neg_dot = fminf(fmaxf(-output_normal_dot, -1.0f), 1.0f);
        const float output_interior_angle = acosf(output_clamped_neg_dot);
        const float out_exterior_angle = interior ? (two_pi - output_interior_angle) : two_pi;

        bool coplanar = false;
        if (interior) {
            const float selected_normal_dot = channel::math::dot_rn_xzy(n0_cand, n1_cand);
            const bool aligned = fabsf(selected_normal_dot) >= normal_cos_tol;
            const int opp0 = opposite_vertex(faces, safe0, v0, v1);
            const int opp1 = opposite_vertex(faces, safe1, v0, v1);
            const float3 point_a = channel::math::load_vec3(vertices, opp0);
            const float3 point_b = channel::math::load_vec3(vertices, opp1);
            const float plane_dist_a = fabsf(channel::math::dot_rn_xzy(channel::math::sub(point_a, start), n0_cand));
            const float plane_dist_b = fabsf(channel::math::dot_rn_xzy(channel::math::sub(point_b, start), n0_cand));
            coplanar = aligned && plane_dist_a <= plane_tol && plane_dist_b <= plane_tol;
        }
        const float selected_normal_dot = channel::math::dot_rn_xzy(n0_cand, n1_cand);
        const bool selected_wedge_angle = boundary || (interior && selected_normal_dot < 1.0f);
        selected[edge] =
            (interior || boundary) && !coplanar && length > edge_epsilon && selected_wedge_angle;

        channel::math::store_vec3(edge_pos, edge, channel::math::scale(channel::math::add(start, end), 0.5f));
        channel::math::store_vec3(edge_dir, edge, dir);
        lengths[edge] = length;
        line_min[edge] = -half_length;
        line_max[edge] = half_length;
        channel::math::store_vec3(n0, edge, out_n0);
        channel::math::store_vec3(n1, edge, out_n1);
        exterior_angle[edge] = out_exterior_angle;
    }
}

__device__ __forceinline__ int find_root_const(const int *__restrict__ parent, int x) {
    int p = parent[x];
    while (p != parent[p]) {
        p = parent[p];
    }
    return p;
}

__global__ void init_parent_kernel(int *__restrict__ parent, int count) {
    const int stride = blockDim.x * gridDim.x;
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < count; idx += stride) {
        parent[idx] = idx;
    }
}

__global__ void count_surface_group_edges_kernel(
    const bool *__restrict__ selected,
    const int *__restrict__ face0,
    const int *__restrict__ face1,
    const int *__restrict__ parent,
    int *__restrict__ root_count,
    int *__restrict__ max_count,
    int64_t edge_count,
    int face_count) {
    if (blockIdx.x != 0 || threadIdx.x != 0) {
        return;
    }
    for (int face = 0; face < face_count; ++face) {
        root_count[face] = 0;
    }
    for (int64_t edge = 0; edge < edge_count; ++edge) {
        if (!selected[edge]) {
            continue;
        }
        const int f0 = face0[edge];
        const int f1 = face1[edge];
        const bool valid0 = f0 >= 0;
        const bool valid1 = f1 >= 0;
        const int root0 = valid0 ? find_root_const(parent, f0) : -1;
        if (valid0) {
            root_count[root0] += 1;
        }
        if (valid1) {
            const int root1 = find_root_const(parent, f1);
            if (!valid0 || root1 != root0) {
                root_count[root1] += 1;
            }
        }
    }
    int local_max = 0;
    for (int face = 0; face < face_count; ++face) {
        local_max = root_count[face] > local_max ? root_count[face] : local_max;
    }
    max_count[0] = local_max;
}

__global__ void fill_int_kernel(int *__restrict__ data, int value, int64_t count) {
    const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         idx < count;
         idx += stride) {
        data[idx] = value;
    }
}

__global__ void fill_surface_group_root_edges_kernel(
    const bool *__restrict__ selected,
    const int *__restrict__ face0,
    const int *__restrict__ face1,
    const int *__restrict__ parent,
    int *__restrict__ root_cursor,
    int *__restrict__ root_indices,
    int64_t edge_count,
    int face_count,
    int max_count) {
    if (blockIdx.x != 0 || threadIdx.x != 0) {
        return;
    }
    for (int face = 0; face < face_count; ++face) {
        root_cursor[face] = 0;
    }
    for (int64_t edge = 0; edge < edge_count; ++edge) {
        if (!selected[edge]) {
            continue;
        }
        const int edge_i = static_cast<int>(edge);
        const int f0 = face0[edge];
        const int f1 = face1[edge];
        const bool valid0 = f0 >= 0;
        const bool valid1 = f1 >= 0;
        const int root0 = valid0 ? find_root_const(parent, f0) : -1;
        if (valid0) {
            const int slot = root_cursor[root0]++;
            if (slot < max_count) {
                root_indices[static_cast<int64_t>(root0) * max_count + slot] = edge_i;
            }
        }
        if (valid1) {
            const int root1 = find_root_const(parent, f1);
            if (!valid0 || root1 != root0) {
                const int slot = root_cursor[root1]++;
                if (slot < max_count) {
                    root_indices[static_cast<int64_t>(root1) * max_count + slot] = edge_i;
                }
            }
        }
    }
}

__global__ void emit_surface_group_face_rows_kernel(
    const int *__restrict__ parent,
    const int *__restrict__ root_count,
    const int *__restrict__ root_indices,
    int *__restrict__ counts,
    int *__restrict__ indices,
    int face_count,
    int max_count) {
    const int stride = blockDim.x * gridDim.x;
    for (int face = blockIdx.x * blockDim.x + threadIdx.x; face < face_count; face += stride) {
        const int root = find_root_const(parent, face);
        const int count = root_count[root];
        counts[face] = count;
        for (int slot = 0; slot < count && slot < max_count; ++slot) {
            indices[static_cast<int64_t>(face) * max_count + slot] =
                root_indices[static_cast<int64_t>(root) * max_count + slot];
        }
    }
}

__global__ void count_selected_edge_indices_kernel(
    const bool *__restrict__ selected,
    int *__restrict__ count,
    int64_t edge_count) {
    if (blockIdx.x != 0 || threadIdx.x != 0) {
        return;
    }
    int local_count = 0;
    for (int64_t edge = 0; edge < edge_count; ++edge) {
        if (selected[edge]) {
            ++local_count;
        }
    }
    count[0] = local_count;
}

__global__ void fill_selected_edge_indices_kernel(
    const bool *__restrict__ selected,
    int *__restrict__ indices,
    int64_t edge_count) {
    if (blockIdx.x != 0 || threadIdx.x != 0) {
        return;
    }
    int cursor = 0;
    for (int64_t edge = 0; edge < edge_count; ++edge) {
        if (selected[edge]) {
            indices[cursor++] = static_cast<int>(edge);
        }
    }
}

}  // namespace

at::Tensor diffraction_state_wi_cuda_impl(at::Tensor state_edge_pos, at::Tensor state_src) {
    check_tensor(state_edge_pos, "state_edge_pos", at::kFloat, 2);
    check_tensor(state_src, "state_src", at::kFloat, 2);
    TORCH_CHECK(state_edge_pos.size(1) == 3, "state_edge_pos must have shape (N, 3)");
    TORCH_CHECK(state_src.size(1) == 3, "state_src must have shape (N, 3)");
    TORCH_CHECK(state_src.size(0) == state_edge_pos.size(0), "state_src must match state_edge_pos");

    const int64_t state_count = state_edge_pos.size(0);
    auto state_wi = at::empty({state_count, 3}, state_edge_pos.options());
    if (state_count > 0) {
        cudaStream_t stream = at::cuda::getCurrentCUDAStream(state_edge_pos.get_device()).stream();
        const int block_count = static_cast<int>((state_count + kDiffractionBlockSize - 1) / kDiffractionBlockSize);
        diffraction_state_wi_kernel<<<block_count, kDiffractionBlockSize, 0, stream>>>(
            state_edge_pos.data_ptr<float>(),
            state_src.data_ptr<float>(),
            state_wi.data_ptr<float>(),
            state_count);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return state_wi;
}

at::Tensor selected_edge_indices_cuda_impl(at::Tensor selected) {
    check_tensor(selected, "selected", at::kBool, 1);
    const int64_t edge_count = selected.size(0);
    auto int_options = selected.options().dtype(at::kInt);
    auto count_tensor = at::empty({1}, int_options);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(selected.get_device()).stream();
    count_selected_edge_indices_kernel<<<1, 1, 0, stream>>>(
        selected.data_ptr<bool>(),
        count_tensor.data_ptr<int>(),
        edge_count);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    int host_count = 0;
    C10_CUDA_CHECK(cudaMemcpyAsync(
        &host_count,
        count_tensor.data_ptr<int>(),
        sizeof(int),
        cudaMemcpyDeviceToHost,
        stream));
    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
    TORCH_CHECK(host_count >= 0, "selected edge count must be non-negative");

    auto indices = at::empty({host_count}, int_options);
    if (host_count > 0) {
        fill_selected_edge_indices_kernel<<<1, 1, 0, stream>>>(
            selected.data_ptr<bool>(),
            indices.data_ptr<int>(),
            edge_count);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return indices;
}

std::vector<at::Tensor> diffraction_state_pack_cuda_impl(
    at::Tensor edge_indices,
    at::Tensor edge_pos,
    at::Tensor edge_dir,
    at::Tensor line_min,
    at::Tensor line_max,
    at::Tensor n0,
    at::Tensor n1,
    at::Tensor face0,
    at::Tensor face1,
    at::Tensor exterior_angle,
    at::Tensor tx,
    at::Tensor tx_power,
    int64_t tx_power_index) {
    check_tensor(edge_indices, "edge_indices", at::kInt, 1);
    CHANNEL_DIFFRACTION_CHECK_STATE_PACK_TENSORS();
    CHANNEL_DIFFRACTION_CHECK_STATE_PACK_POWER();
    CHANNEL_DIFFRACTION_CHECK_STATE_PACK_SHAPES();
    TORCH_CHECK(tx.size(0) == 3, "tx must have shape (3,)");
    TORCH_CHECK(edge_pos.get_device() == edge_indices.get_device(), "edge tensors must be on the same device");

    const int64_t state_count = edge_indices.size(0);
    auto int_options = edge_indices.options();
    auto float_options = edge_pos.options();
    CHANNEL_DIFFRACTION_ALLOCATE_STATE_PACK();

    if (state_count > 0) {
        cudaStream_t stream = at::cuda::getCurrentCUDAStream(edge_pos.get_device()).stream();
        const int block_count = static_cast<int>((state_count + kDiffractionBlockSize - 1) / kDiffractionBlockSize);
        diffraction_state_pack_kernel<<<block_count, kDiffractionBlockSize, 0, stream>>>(
            edge_indices.data_ptr<int>(),
            CHANNEL_DIFFRACTION_STATE_PACK_INPUT_POINTERS(),
            CHANNEL_DIFFRACTION_STATE_PACK_OUTPUT_POINTERS(),
            state_count);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

    return {
        CHANNEL_DIFFRACTION_STATE_PACK_RESULTS(),
    };
}

std::vector<at::Tensor> diffraction_state_pack_selected_cuda_impl(
    at::Tensor selected,
    at::Tensor edge_pos,
    at::Tensor edge_dir,
    at::Tensor line_min,
    at::Tensor line_max,
    at::Tensor n0,
    at::Tensor n1,
    at::Tensor face0,
    at::Tensor face1,
    at::Tensor exterior_angle,
    at::Tensor tx,
    at::Tensor tx_power,
    int64_t tx_power_index) {
    check_tensor(selected, "selected", at::kBool, 1);
    CHANNEL_DIFFRACTION_CHECK_STATE_PACK_TENSORS();
    CHANNEL_DIFFRACTION_CHECK_STATE_PACK_POWER();
    CHANNEL_DIFFRACTION_CHECK_STATE_PACK_SHAPES();
    TORCH_CHECK(selected.size(0) == edge_pos.size(0), "selected must match edge count");
    TORCH_CHECK(tx.size(0) == 3, "tx must have shape (3,)");
    TORCH_CHECK(edge_pos.get_device() == selected.get_device(), "edge tensors must be on the same device");

    const int64_t state_count = edge_pos.size(0);
    auto int_options = face0.options();
    auto float_options = edge_pos.options();
    CHANNEL_DIFFRACTION_ALLOCATE_STATE_PACK();

    if (state_count > 0) {
        cudaStream_t stream = at::cuda::getCurrentCUDAStream(edge_pos.get_device()).stream();
        const int block_count = static_cast<int>((state_count + kDiffractionBlockSize - 1) / kDiffractionBlockSize);
        diffraction_state_pack_selected_kernel<<<block_count, kDiffractionBlockSize, 0, stream>>>(
            selected.data_ptr<bool>(),
            CHANNEL_DIFFRACTION_STATE_PACK_INPUT_POINTERS(),
            CHANNEL_DIFFRACTION_STATE_PACK_OUTPUT_POINTERS(),
            state_count);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

    return {
        CHANNEL_DIFFRACTION_STATE_PACK_RESULTS(),
    };
}

std::vector<at::Tensor> diffraction_edge_geometry_cuda_impl(
    at::Tensor vertices,
    at::Tensor faces,
    at::Tensor face_normals,
    at::Tensor edge_v0,
    at::Tensor edge_v1,
    at::Tensor face0,
    at::Tensor face1,
    double plane_tol) {
    check_tensor(vertices, "vertices", at::kFloat, 2);
    check_tensor(faces, "faces", at::kInt, 2);
    check_tensor(face_normals, "face_normals", at::kFloat, 2);
    check_tensor(edge_v0, "edge_v0", at::kInt, 1);
    check_tensor(edge_v1, "edge_v1", at::kInt, 1);
    check_tensor(face0, "face0", at::kInt, 1);
    check_tensor(face1, "face1", at::kInt, 1);
    TORCH_CHECK(vertices.size(1) == 3, "vertices must have shape (N, 3)");
    TORCH_CHECK(faces.size(1) == 3, "faces must have shape (F, 3)");
    TORCH_CHECK(face_normals.size(1) == 3, "face_normals must have shape (F, 3)");
    TORCH_CHECK(edge_v1.size(0) == edge_v0.size(0), "edge_v1 must match edge_v0");
    TORCH_CHECK(face0.size(0) == edge_v0.size(0), "face0 must match edge_v0");
    TORCH_CHECK(face1.size(0) == edge_v0.size(0), "face1 must match edge_v0");

    const int64_t edge_count = edge_v0.size(0);
    auto bool_options = edge_v0.options().dtype(at::kBool);
    auto float_options = vertices.options();
    auto selected = at::empty({edge_count}, bool_options);
    auto edge_pos = at::empty({edge_count, 3}, float_options);
    auto edge_dir = at::empty({edge_count, 3}, float_options);
    auto lengths = at::empty({edge_count}, float_options);
    auto line_min = at::empty({edge_count}, float_options);
    auto line_max = at::empty({edge_count}, float_options);
    auto n0 = at::empty({edge_count, 3}, float_options);
    auto n1 = at::empty({edge_count, 3}, float_options);
    auto exterior_angle = at::empty({edge_count}, float_options);

    if (edge_count > 0) {
        cudaStream_t stream = at::cuda::getCurrentCUDAStream(vertices.get_device()).stream();
        const int block_count = static_cast<int>((edge_count + kDiffractionBlockSize - 1) / kDiffractionBlockSize);
        diffraction_edge_geometry_kernel<<<block_count, kDiffractionBlockSize, 0, stream>>>(
            vertices.data_ptr<float>(),
            faces.data_ptr<int>(),
            face_normals.data_ptr<float>(),
            edge_v0.data_ptr<int>(),
            edge_v1.data_ptr<int>(),
            face0.data_ptr<int>(),
            face1.data_ptr<int>(),
            selected.data_ptr<bool>(),
            edge_pos.data_ptr<float>(),
            edge_dir.data_ptr<float>(),
            lengths.data_ptr<float>(),
            line_min.data_ptr<float>(),
            line_max.data_ptr<float>(),
            n0.data_ptr<float>(),
            n1.data_ptr<float>(),
            exterior_angle.data_ptr<float>(),
            edge_count,
            static_cast<float>(plane_tol));
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

    return {
        selected,
        edge_pos,
        edge_dir,
        lengths,
        line_min,
        line_max,
        n0,
        n1,
        face0,
        face1,
        exterior_angle,
    };
}

std::vector<at::Tensor> surface_group_edge_candidates_cuda_impl(
    at::Tensor vertices,
    at::Tensor faces,
    at::Tensor face_normals,
    at::Tensor edge_v0,
    at::Tensor edge_v1,
    at::Tensor face0,
    at::Tensor face1,
    at::Tensor selected,
    double plane_tol) {
    check_tensor(vertices, "vertices", at::kFloat, 2);
    check_tensor(faces, "faces", at::kInt, 2);
    check_tensor(face_normals, "face_normals", at::kFloat, 2);
    check_tensor(edge_v0, "edge_v0", at::kInt, 1);
    check_tensor(edge_v1, "edge_v1", at::kInt, 1);
    check_tensor(face0, "face0", at::kInt, 1);
    check_tensor(face1, "face1", at::kInt, 1);
    check_tensor(selected, "selected", at::kBool, 1);
    TORCH_CHECK(vertices.size(1) == 3, "vertices must have shape (N, 3)");
    TORCH_CHECK(faces.size(1) == 3, "faces must have shape (F, 3)");
    TORCH_CHECK(face_normals.size(1) == 3, "face_normals must have shape (F, 3)");
    TORCH_CHECK(edge_v1.size(0) == edge_v0.size(0), "edge_v1 must match edge_v0");
    TORCH_CHECK(face0.size(0) == edge_v0.size(0), "face0 must match edge_v0");
    TORCH_CHECK(face1.size(0) == edge_v0.size(0), "face1 must match edge_v0");
    TORCH_CHECK(selected.size(0) == edge_v0.size(0), "selected must match edge_v0");

    const int face_count = static_cast<int>(faces.size(0));
    const int64_t edge_count = edge_v0.size(0);
    auto int_options = faces.options();
    auto parent = at::empty({face_count}, int_options);
    auto root_count = at::empty({face_count}, int_options);
    auto root_cursor = at::empty({face_count}, int_options);
    auto max_count_tensor = at::empty({1}, int_options);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream(vertices.get_device()).stream();
    const int face_blocks = static_cast<int>((face_count + kDiffractionBlockSize - 1) / kDiffractionBlockSize);
    if (face_count > 0) {
        init_parent_kernel<<<face_blocks, kDiffractionBlockSize, 0, stream>>>(
            parent.data_ptr<int>(),
            face_count);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

    // The silhouette projection samples the perimeter of the single
    // intersected primitive, so the sampling domain is the per-triangle
    // perimeter set. Keep every triangle as its own root: merging coplanar
    // neighbours into one surface group replaces that domain with the merged
    // group's outline, which both drops interior wedges the per-triangle
    // domain contains and adds outline wedges it does not.
    (void)vertices;
    (void)face_normals;
    (void)edge_v0;
    (void)edge_v1;
    (void)plane_tol;

    count_surface_group_edges_kernel<<<1, 1, 0, stream>>>(
        selected.data_ptr<bool>(),
        face0.data_ptr<int>(),
        face1.data_ptr<int>(),
        parent.data_ptr<int>(),
        root_count.data_ptr<int>(),
        max_count_tensor.data_ptr<int>(),
        edge_count,
        face_count);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    int host_max_count = 0;
    C10_CUDA_CHECK(cudaMemcpyAsync(
        &host_max_count,
        max_count_tensor.data_ptr<int>(),
        sizeof(int),
        cudaMemcpyDeviceToHost,
        stream));
    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
    TORCH_CHECK(host_max_count >= 0, "surface group candidate max count must be non-negative");

    auto counts = at::empty({face_count}, int_options);
    auto indices = at::empty({face_count, host_max_count}, int_options);
    auto root_indices = at::empty({face_count, host_max_count}, int_options);
    const int64_t table_count = static_cast<int64_t>(face_count) * host_max_count;
    if (table_count > 0) {
        const int table_blocks = static_cast<int>((table_count + kDiffractionBlockSize - 1) / kDiffractionBlockSize);
        fill_int_kernel<<<table_blocks, kDiffractionBlockSize, 0, stream>>>(
            indices.data_ptr<int>(),
            -1,
            table_count);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        fill_int_kernel<<<table_blocks, kDiffractionBlockSize, 0, stream>>>(
            root_indices.data_ptr<int>(),
            -1,
            table_count);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    fill_surface_group_root_edges_kernel<<<1, 1, 0, stream>>>(
        selected.data_ptr<bool>(),
        face0.data_ptr<int>(),
        face1.data_ptr<int>(),
        parent.data_ptr<int>(),
        root_cursor.data_ptr<int>(),
        root_indices.data_ptr<int>(),
        edge_count,
        face_count,
        host_max_count);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    if (face_count > 0) {
        emit_surface_group_face_rows_kernel<<<face_blocks, kDiffractionBlockSize, 0, stream>>>(
            parent.data_ptr<int>(),
            root_count.data_ptr<int>(),
            root_indices.data_ptr<int>(),
            counts.data_ptr<int>(),
            indices.data_ptr<int>(),
            face_count,
            host_max_count);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

    return {counts, indices};
}

at::Tensor channel_mc_diffraction_state_wi_cuda(at::Tensor state_edge_pos, at::Tensor state_src) {
    return diffraction_state_wi_cuda_impl(state_edge_pos, state_src);
}

at::Tensor channel_mc_selected_edge_indices_cuda(at::Tensor selected) {
    return selected_edge_indices_cuda_impl(selected);
}

std::vector<at::Tensor> channel_mc_diffraction_state_pack_cuda(
    at::Tensor edge_indices,
    at::Tensor edge_pos,
    at::Tensor edge_dir,
    at::Tensor line_min,
    at::Tensor line_max,
    at::Tensor n0,
    at::Tensor n1,
    at::Tensor face0,
    at::Tensor face1,
    at::Tensor exterior_angle,
    at::Tensor tx,
    at::Tensor tx_power) {
    return diffraction_state_pack_cuda_impl(
        edge_indices,
        edge_pos,
        edge_dir,
        line_min,
        line_max,
        n0,
        n1,
        face0,
        face1,
        exterior_angle,
        tx,
        tx_power,
        0);
}

std::vector<at::Tensor> channel_deterministic_diffraction_state_pack_cuda(
    at::Tensor edge_indices,
    at::Tensor edge_pos,
    at::Tensor edge_dir,
    at::Tensor line_min,
    at::Tensor line_max,
    at::Tensor n0,
    at::Tensor n1,
    at::Tensor face0,
    at::Tensor face1,
    at::Tensor exterior_angle,
    at::Tensor tx,
    at::Tensor tx_power,
    int64_t tx_power_index) {
    return diffraction_state_pack_cuda_impl(
        edge_indices,
        edge_pos,
        edge_dir,
        line_min,
        line_max,
        n0,
        n1,
        face0,
        face1,
        exterior_angle,
        tx,
        tx_power,
        tx_power_index);
}

std::vector<at::Tensor> channel_deterministic_diffraction_state_pack_selected_cuda(
    at::Tensor selected,
    at::Tensor edge_pos,
    at::Tensor edge_dir,
    at::Tensor line_min,
    at::Tensor line_max,
    at::Tensor n0,
    at::Tensor n1,
    at::Tensor face0,
    at::Tensor face1,
    at::Tensor exterior_angle,
    at::Tensor tx,
    at::Tensor tx_power,
    int64_t tx_power_index) {
    return diffraction_state_pack_selected_cuda_impl(
        selected,
        edge_pos,
        edge_dir,
        line_min,
        line_max,
        n0,
        n1,
        face0,
        face1,
        exterior_angle,
        tx,
        tx_power,
        tx_power_index);
}

std::vector<at::Tensor> channel_mc_diffraction_edge_geometry_cuda(
    at::Tensor vertices,
    at::Tensor faces,
    at::Tensor face_normals,
    at::Tensor edge_v0,
    at::Tensor edge_v1,
    at::Tensor face0,
    at::Tensor face1,
    double plane_tol) {
    return diffraction_edge_geometry_cuda_impl(
        vertices,
        faces,
        face_normals,
        edge_v0,
        edge_v1,
        face0,
        face1,
        plane_tol);
}

std::vector<at::Tensor> channel_mc_surface_group_edge_candidates_cuda(
    at::Tensor vertices,
    at::Tensor faces,
    at::Tensor face_normals,
    at::Tensor edge_v0,
    at::Tensor edge_v1,
    at::Tensor face0,
    at::Tensor face1,
    at::Tensor selected,
    double plane_tol) {
    return surface_group_edge_candidates_cuda_impl(
        vertices,
        faces,
        face_normals,
        edge_v0,
        edge_v1,
        face0,
        face1,
        selected,
        plane_tol);
}

at::Tensor channel_mc_utd_diffraction_tape_accumulate_cuda(
    at::Tensor tape_active, at::Tensor tape_state, at::Tensor tape_cell, at::Tensor tape_u,
    at::Tensor edge_pos, at::Tensor edge_dir, at::Tensor t_min, at::Tensor t_max,
    at::Tensor n0, at::Tensor nn, at::Tensor prim0, at::Tensor prim1,
    at::Tensor exterior_angle, at::Tensor source, at::Tensor source_power,
    at::Tensor eta_r, at::Tensor sigma, at::Tensor mu_r, at::Tensor gain,
    at::Tensor material_valid, at::Tensor thickness, int64_t axis, double plane,
    double c0min, double c0max, double c1min, double c1max,
    int64_t r0, int64_t r1, double wavelength, double cell_area, int64_t seed, double total_edge_length,
    at::Tensor tx_pol) {
    auto output=at::empty({r1,r0},source.options());
    const auto stream=at::cuda::getCurrentCUDAStream();
    C10_CUDA_CHECK(cudaMemsetAsync(
        output.data_ptr<float>(),0,static_cast<size_t>(output.numel())*sizeof(float),stream));
    const int64_t samples=tape_active.numel();
    if(samples==0) return output;
    const int blocks=static_cast<int>(std::min<int64_t>((samples+kDiffractionBlockSize-1)/kDiffractionBlockSize,65535));
    utd_diffraction_tape_accumulate_kernel<<<blocks,kDiffractionBlockSize,0,stream>>>(
        tape_active.data_ptr<bool>(),tape_state.data_ptr<int>(),tape_cell.data_ptr<int>(),tape_u.data_ptr<float>(),
        edge_pos.data_ptr<float>(),edge_dir.data_ptr<float>(),t_min.data_ptr<float>(),t_max.data_ptr<float>(),
        n0.data_ptr<float>(),nn.data_ptr<float>(),prim0.data_ptr<int>(),prim1.data_ptr<int>(),
        exterior_angle.data_ptr<float>(),source.data_ptr<float>(),source_power.data_ptr<float>(),
        eta_r.data_ptr<float>(),sigma.data_ptr<float>(),mu_r.data_ptr<float>(),gain.data_ptr<float>(),thickness.data_ptr<float>(),
        material_valid.data_ptr<bool>(),output.data_ptr<float>(),samples,static_cast<int>(edge_pos.size(0)),
        static_cast<int>(axis),static_cast<float>(plane),static_cast<float>(c0min),static_cast<float>(c0max),
        static_cast<float>(c1min),static_cast<float>(c1max),static_cast<int>(r0),static_cast<int>(r1),
        static_cast<float>(wavelength),static_cast<float>(cell_area),static_cast<int>(seed),static_cast<float>(total_edge_length),
        tx_pol.data_ptr<float>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}

namespace {

// Zero-initialized gradient accumulator (memset on the current stream; same
// pattern as reflection.cu / los.cu).
at::Tensor diffraction_zero_filled(
    at::IntArrayRef sizes, const at::TensorOptions& options) {
    auto tensor = at::empty(sizes, options);
    if (tensor.numel() > 0) {
        cudaStream_t stream =
            at::cuda::getCurrentCUDAStream(tensor.get_device()).stream();
        C10_CUDA_CHECK(cudaMemsetAsync(
            tensor.data_ptr(),
            0,
            static_cast<size_t>(tensor.numel()) * tensor.element_size(),
            stream));
    }
    return tensor;
}

}  // namespace

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
channel_mc_utd_diffraction_tape_accumulate_backward_cuda(
    at::Tensor tape_active, at::Tensor tape_state, at::Tensor tape_cell, at::Tensor tape_u,
    at::Tensor edge_pos, at::Tensor edge_dir, at::Tensor t_min, at::Tensor t_max,
    at::Tensor n0, at::Tensor nn, at::Tensor prim0, at::Tensor prim1,
    at::Tensor exterior_angle, at::Tensor source, at::Tensor source_power,
    at::Tensor eta_r, at::Tensor sigma, at::Tensor mu_r, at::Tensor gain,
    at::Tensor material_valid, at::Tensor thickness,
    at::Tensor grad_output,
    bool need_materials, bool need_source, bool need_frequency,
    int64_t axis, double plane,
    int64_t r0, int64_t r1, double wavelength, double cell_area, int64_t seed,
    double total_edge_length, double wavelength_dfreq, at::Tensor tx_pol) {
    TORCH_CHECK(grad_output.is_cuda(), "grad_output must be a CUDA tensor");
    TORCH_CHECK(grad_output.scalar_type() == at::kFloat, "grad_output must be float32");
    TORCH_CHECK(grad_output.dim() == 2, "grad_output must have 2 dimensions");
    TORCH_CHECK(
        grad_output.size(0) == r1 && grad_output.size(1) == r0,
        "grad_output must match the (resolution1, resolution0) map");
    const auto face_options = eta_r.options();
    auto grad_eta_r = diffraction_zero_filled({eta_r.size(0)}, face_options);
    auto grad_sigma = diffraction_zero_filled({eta_r.size(0)}, face_options);
    auto grad_gain = diffraction_zero_filled({eta_r.size(0)}, face_options);
    auto grad_thickness = diffraction_zero_filled({eta_r.size(0)}, face_options);
    auto grad_source = diffraction_zero_filled({3}, source.options());
    auto grad_frequency = diffraction_zero_filled({1}, source.options());
    const int64_t samples = tape_active.numel();
    if (samples == 0 || !(need_materials || need_source || need_frequency)) {
        return {grad_eta_r, grad_sigma, grad_gain, grad_thickness, grad_source,
                grad_frequency};
    }
    const auto stream = at::cuda::getCurrentCUDAStream();
    const int blocks = static_cast<int>(std::min<int64_t>(
        (samples + kDiffractionBlockSize - 1) / kDiffractionBlockSize, 65535));
    utd_diffraction_tape_accumulate_backward_kernel<<<blocks, kDiffractionBlockSize, 0, stream>>>(
        tape_active.data_ptr<bool>(), tape_state.data_ptr<int>(),
        tape_cell.data_ptr<int>(), tape_u.data_ptr<float>(),
        edge_pos.data_ptr<float>(), edge_dir.data_ptr<float>(),
        t_min.data_ptr<float>(), t_max.data_ptr<float>(),
        n0.data_ptr<float>(), nn.data_ptr<float>(), prim0.data_ptr<int>(),
        prim1.data_ptr<int>(), exterior_angle.data_ptr<float>(),
        source.data_ptr<float>(), source_power.data_ptr<float>(),
        eta_r.data_ptr<float>(), sigma.data_ptr<float>(),
        mu_r.data_ptr<float>(), gain.data_ptr<float>(),
        thickness.data_ptr<float>(), material_valid.data_ptr<bool>(),
        grad_output.data_ptr<float>(),
        need_materials ? grad_eta_r.data_ptr<float>() : nullptr,
        need_materials ? grad_sigma.data_ptr<float>() : nullptr,
        need_materials ? grad_gain.data_ptr<float>() : nullptr,
        need_materials ? grad_thickness.data_ptr<float>() : nullptr,
        need_source ? grad_source.data_ptr<float>() : nullptr,
        need_frequency ? grad_frequency.data_ptr<float>() : nullptr,
        samples, static_cast<int>(edge_pos.size(0)),
        static_cast<int>(axis), static_cast<float>(plane),
        static_cast<int>(r0), static_cast<int>(r1),
        static_cast<float>(wavelength), static_cast<float>(cell_area),
        static_cast<int>(seed), static_cast<float>(total_edge_length),
        static_cast<float>(wavelength_dfreq),
        grad_output.stride(0), grad_output.stride(1),
        tx_pol.data_ptr<float>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {grad_eta_r, grad_sigma, grad_gain, grad_thickness, grad_source,
            grad_frequency};
}

at::Tensor channel_mc_utd_diffraction_tape_accumulate_jvp_cuda(
    at::Tensor tape_active, at::Tensor tape_state, at::Tensor tape_cell, at::Tensor tape_u,
    at::Tensor edge_pos, at::Tensor edge_dir, at::Tensor t_min, at::Tensor t_max,
    at::Tensor n0, at::Tensor nn, at::Tensor prim0, at::Tensor prim1,
    at::Tensor exterior_angle, at::Tensor source, at::Tensor source_power,
    at::Tensor eta_r, at::Tensor sigma, at::Tensor mu_r, at::Tensor gain,
    at::Tensor material_valid, at::Tensor thickness,
    at::Tensor tangent_eta_r, at::Tensor tangent_sigma, at::Tensor tangent_gain,
    at::Tensor tangent_thickness, at::Tensor tangent_source,
    bool has_tangent_eta_r, bool has_tangent_sigma, bool has_tangent_gain,
    bool has_tangent_thickness, bool has_tangent_source,
    int64_t axis, double plane,
    int64_t r0, int64_t r1, double wavelength, double cell_area, int64_t seed,
    double total_edge_length, double wavelength_tangent, at::Tensor tx_pol) {
    auto output_tangent = diffraction_zero_filled({r1, r0}, source.options());
    const int64_t samples = tape_active.numel();
    if (samples == 0) return output_tangent;
    const auto stream = at::cuda::getCurrentCUDAStream();
    const int blocks = static_cast<int>(std::min<int64_t>(
        (samples + kDiffractionBlockSize - 1) / kDiffractionBlockSize, 65535));
    utd_diffraction_tape_accumulate_jvp_kernel<<<blocks, kDiffractionBlockSize, 0, stream>>>(
        tape_active.data_ptr<bool>(), tape_state.data_ptr<int>(),
        tape_cell.data_ptr<int>(), tape_u.data_ptr<float>(),
        edge_pos.data_ptr<float>(), edge_dir.data_ptr<float>(),
        t_min.data_ptr<float>(), t_max.data_ptr<float>(),
        n0.data_ptr<float>(), nn.data_ptr<float>(), prim0.data_ptr<int>(),
        prim1.data_ptr<int>(), exterior_angle.data_ptr<float>(),
        source.data_ptr<float>(), source_power.data_ptr<float>(),
        eta_r.data_ptr<float>(), sigma.data_ptr<float>(),
        mu_r.data_ptr<float>(), gain.data_ptr<float>(),
        thickness.data_ptr<float>(), material_valid.data_ptr<bool>(),
        has_tangent_eta_r ? tangent_eta_r.data_ptr<float>() : nullptr,
        has_tangent_sigma ? tangent_sigma.data_ptr<float>() : nullptr,
        has_tangent_gain ? tangent_gain.data_ptr<float>() : nullptr,
        has_tangent_thickness ? tangent_thickness.data_ptr<float>() : nullptr,
        has_tangent_source ? tangent_source.data_ptr<float>() : nullptr,
        static_cast<float>(wavelength_tangent),
        output_tangent.data_ptr<float>(),
        samples, static_cast<int>(edge_pos.size(0)),
        static_cast<int>(axis), static_cast<float>(plane),
        static_cast<int>(r0), static_cast<int>(r1),
        static_cast<float>(wavelength), static_cast<float>(cell_area),
        static_cast<int>(seed), static_cast<float>(total_edge_length),
        tx_pol.data_ptr<float>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output_tangent;
}

#undef CHANNEL_DIFFRACTION_STATE_PACK_RESULTS
#undef CHANNEL_DIFFRACTION_STATE_PACK_OUTPUT_POINTERS
#undef CHANNEL_DIFFRACTION_STATE_PACK_INPUT_POINTERS
#undef CHANNEL_DIFFRACTION_ALLOCATE_STATE_PACK
#undef CHANNEL_DIFFRACTION_CHECK_STATE_PACK_SHAPES
#undef CHANNEL_DIFFRACTION_CHECK_STATE_PACK_POWER
#undef CHANNEL_DIFFRACTION_CHECK_STATE_PACK_TENSORS

// ==== Section: Diffraction discovery ====
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>

#include <cuda_runtime_api.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace {

constexpr int kBlockSize = 256;
constexpr float kEps = 1.0e-6f;
constexpr float kHalfPiMinusOffset = 1.52079632679f;

using Vec3 = channel::math::Vec3;
namespace cmath = channel::math;

__device__ __forceinline__ Vec3 silhouette_viewpoint(
    Vec3 hit_p, Vec3 shading_n, Vec3 geometric_n, Vec3 ray_dir) {
    Vec3 geo = cmath::length(geometric_n)>kEps ? geometric_n : shading_n;
    // Mitsuba's primitive_silhouette_projection uses the unmodified
    // geometric interaction normal (si.n), not a face-forward normal.
    Vec3 surface_n = geo;
    Vec3 tangent = cmath::sub(ray_dir,cmath::scale(surface_n,cmath::dot(ray_dir,surface_n)));
    if (cmath::length(tangent)<=kEps) {
        Vec3 fx=cmath::cross(surface_n,cmath::vec3(1.f,0.f,0.f));
        Vec3 fy=cmath::cross(surface_n,cmath::vec3(0.f,1.f,0.f));
        tangent=cmath::length(fx)>kEps?fx:fy;
    }
    tangent=cmath::normalize_or(tangent, kEps, cmath::vec3(1.f, 0.f, 0.f));
    Vec3 d=cmath::add(cmath::scale(surface_n,cosf(kHalfPiMinusOffset)),cmath::scale(tangent,sinf(kHalfPiMinusOffset)));
    // Fixed 0.1 scene-unit viewpoint displacement, deliberately absolute and
    // not scaled by primitive or scene size. Together with kHalfPiMinusOffset
    // (pi/2 - 0.05 rad, so `d` is the surface tangent lifted 0.05 rad off the
    // surface) it pushes the viewpoint just past the hit point and slightly
    // clear of the plane, which is what lets the silhouette projection below
    // see the primitive's perimeter edges. Both constants select which edges
    // are discovered, so they are pinned to these values rather than tuned.
    const float offset=0.1f;
    return cmath::add(hit_p,cmath::scale(d,offset));
}

__device__ __forceinline__ bool wedge_exterior(Vec3 from_edge,Vec3 edge_dir,Vec3 n0,Vec3 n1) {
    Vec3 eh=cmath::normalize_or(edge_dir, kEps, cmath::vec3(0.0f, 0.0f, 1.0f));
    Vec3 projected=cmath::sub(from_edge,cmath::scale(eh,cmath::dot(from_edge,eh)));
    return cmath::length(projected)>kEps && (cmath::dot(projected,n0)>=-kEps || cmath::dot(projected,n1)>=-kEps);
}

__device__ int sampled_edge(
    Vec3 tx, Vec3 ray_dir, Vec3 hit_p, Vec3 hit_n, Vec3 hit_geo_n, int prim,
    int sample_index,
    const int *tri_count,const int *tri_edges,int slots,int tri_n,
    const float *edge_pos,const float *edge_dir,const float *n0p,const float *n1p,
    const float *tminp,const float *tmaxp,const int *face1,int edge_n) {
    if(prim<0||prim>=tri_n) return -1;
    const int count=min(max(tri_count[prim],0),slots);
    Vec3 viewpoint=silhouette_viewpoint(hit_p,hit_n,hit_geo_n,ray_dir);
    int valid_count=0;
    for(int s=0;s<count;++s){
        int e=tri_edges[prim*slots+s]; if(e<0||e>=edge_n) continue;
        Vec3 ep=cmath::load_vec3(edge_pos,e), ed=cmath::load_vec3(edge_dir,e), eh=cmath::normalize_or(ed, kEps, cmath::vec3(0.0f, 0.0f, 1.0f));
        float ell=fminf(fmaxf(cmath::dot(cmath::sub(viewpoint,ep),eh),tminp[e]),tmaxp[e]);
        Vec3 point=cmath::add(ep,cmath::scale(eh,ell));
        Vec3 n0=cmath::load_vec3(n0p,e),n1=cmath::load_vec3(n1p,e);
        bool flip=cmath::dot(ray_dir,n0)>0.f;
        if(!wedge_exterior(cmath::sub(tx,point),ed,flip?n1:n0,flip?n0:n1)) continue;
        ++valid_count;
    }
    if(valid_count<=0) return -1;
    unsigned int h=static_cast<unsigned int>(sample_index)^0x9e3779b9u;
    h^=h>>16; h*=0x7feb352du; h^=h>>15; h*=0x846ca68bu; h^=h>>16;
    const int wanted=static_cast<int>(h%static_cast<unsigned int>(valid_count));
    int ordinal=0;
    for(int s=0;s<count;++s){
        int e=tri_edges[prim*slots+s]; if(e<0||e>=edge_n) continue;
        Vec3 ep=cmath::load_vec3(edge_pos,e), ed=cmath::load_vec3(edge_dir,e), eh=cmath::normalize_or(ed, kEps, cmath::vec3(0.0f, 0.0f, 1.0f));
        float ell=fminf(fmaxf(cmath::dot(cmath::sub(viewpoint,ep),eh),tminp[e]),tmaxp[e]);
        Vec3 point=cmath::add(ep,cmath::scale(eh,ell)); Vec3 en0=cmath::load_vec3(n0p,e),en1=cmath::load_vec3(n1p,e);
        bool flip=cmath::dot(ray_dir,en0)>0.f;
        if(!wedge_exterior(cmath::sub(tx,point),ed,flip?en1:en0,flip?en0:en1)) continue;
        if(ordinal++==wanted) return e;
    }
    return -1;
}

__global__ void discover_kernel(
    const float *tx,const float *ray_dir,const int *prim,const float *hit_p,
    const float *hit_n,const float *hit_geo_n,const int *hit_count,int capacity,
    const int *tri_count,const int *tri_edges,int slots,int tri_n,
    const float *edge_pos,const float *edge_dir,const float *n0,const float *n1,
    const float *tmin,const float *tmax,const int *face1,int edge_n,int *seen) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    int n=hit_count?min(max(hit_count[0],0),capacity):capacity;
    if(i>=n) return;
    // Discovery stores support, not a per-path contribution. Select one
    // exterior primitive-perimeter candidate per first hit with a reproducible
    // uniform draw; the subsequent estimator samples edge length explicitly.
    int e=sampled_edge(cmath::load_vec3(tx,0),cmath::load_vec3(ray_dir,i),cmath::load_vec3(hit_p,i),cmath::load_vec3(hit_n,i),
        cmath::load_vec3(hit_geo_n,i),prim[i],i,tri_count,tri_edges,slots,tri_n,edge_pos,edge_dir,
        n0,n1,tmin,tmax,face1,edge_n);
    if(e>=0) atomicExch(seen+e,1);
}

const at::Tensor &required(const at::Tensor *t,const char *name){
    if(!t) throw std::runtime_error(std::string("channel diffraction discovery received null ")+name);
    return *t;
}

at::Tensor discover(
    const at::Tensor *tx,const at::Tensor *ray_dir,const at::Tensor *prim,
    const at::Tensor *hit_p,const at::Tensor *hit_n,const at::Tensor *hit_geo_n,
    const at::Tensor *hit_count,const at::Tensor *tri_count,const at::Tensor *tri_edges,
    const at::Tensor *edge_pos,const at::Tensor *edge_dir,const at::Tensor *n0,
    const at::Tensor *n1,const at::Tensor *tmin,const at::Tensor *tmax,
    const at::Tensor *face1){
    const auto &rd=required(ray_dir,"ray_dir"); const auto &ep=required(edge_pos,"edge_pos");
    int64_t capacity=rd.size(0),edges=ep.size(0);
    at::Tensor seen=at::empty({edges},ep.options().dtype(at::kInt));
    if(edges>0){
        C10_CUDA_CHECK(cudaMemsetAsync(
            seen.data_ptr<int>(), 0, static_cast<size_t>(edges) * sizeof(int),
            at::cuda::getCurrentCUDAStream()));
    }
    if(capacity>0&&edges>0){
        int blocks=static_cast<int>((capacity+kBlockSize-1)/kBlockSize);
        const auto &te=required(tri_edges,"triangle_edge_indices");
        discover_kernel<<<blocks,kBlockSize,0,at::cuda::getCurrentCUDAStream()>>>(
            required(tx,"tx_pos").data_ptr<float>(),rd.data_ptr<float>(),required(prim,"prim_index").data_ptr<int>(),
            required(hit_p,"hit_p").data_ptr<float>(),required(hit_n,"hit_n").data_ptr<float>(),
            required(hit_geo_n,"hit_geo_n").data_ptr<float>(),hit_count?hit_count->data_ptr<int>():nullptr,
            static_cast<int>(capacity),required(tri_count,"triangle_edge_count").data_ptr<int>(),te.data_ptr<int>(),
            static_cast<int>(te.size(1)),static_cast<int>(required(tri_count,"triangle_edge_count").size(0)),
            ep.data_ptr<float>(),required(edge_dir,"edge_dir").data_ptr<float>(),required(n0,"edge_n0").data_ptr<float>(),
            required(n1,"edge_n1").data_ptr<float>(),required(tmin,"edge_t_min").data_ptr<float>(),
            required(tmax,"edge_t_max").data_ptr<float>(),required(face1,"edge_face1").data_ptr<int>(),
            static_cast<int>(edges),seen.data_ptr<int>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return at::nonzero(seen).reshape({-1}).to(at::kInt).contiguous();
}

} // namespace

at::Tensor channel_mc_diffraction_discover_edges_cuda(
    at::Tensor tx,at::Tensor ray_dir,at::Tensor prim,
    at::Tensor hit_p,at::Tensor hit_n,at::Tensor hit_geo_n,
    at::Tensor tri_count,at::Tensor tri_edges,at::Tensor edge_pos,
    at::Tensor edge_dir,at::Tensor n0,at::Tensor n1,
    at::Tensor tmin,at::Tensor tmax,at::Tensor face1){
    return discover(&tx,&ray_dir,&prim,&hit_p,&hit_n,&hit_geo_n,nullptr,&tri_count,
        &tri_edges,&edge_pos,&edge_dir,&n0,&n1,&tmin,&tmax,&face1);
}

at::Tensor channel_mc_diffraction_discover_edges_counted_cuda(
    at::Tensor tx,at::Tensor ray_dir,at::Tensor prim,
    at::Tensor hit_p,at::Tensor hit_n,at::Tensor hit_geo_n,
    at::Tensor hit_count,at::Tensor tri_count,at::Tensor tri_edges,
    at::Tensor edge_pos,at::Tensor edge_dir,at::Tensor n0,
    at::Tensor n1,at::Tensor tmin,at::Tensor tmax,
    at::Tensor face1){
    return discover(&tx,&ray_dir,&prim,&hit_p,&hit_n,&hit_geo_n,&hit_count,&tri_count,
        &tri_edges,&edge_pos,&edge_dir,&n0,&n1,&tmin,&tmax,&face1);
}
