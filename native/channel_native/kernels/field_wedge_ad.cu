#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/util/complex.h>
#include <torch/extension.h>

#include "../field_transport.cuh"
#include "../field_transport_ad.cuh"
#include "../tensor_checks.h"

#include <array>
#include <vector>

// Plan 07 AD-4a: differentiable UTD wedge diffraction and coupled
// reflection-diffraction.
//
// The wedge field is RayD's own templated forward
// (rayd/shared/utd/utd_math.h): instantiated with float it IS the production
// forward, instantiated with utd::Dual the same pass carries an exact
// directional derivative (host-FD validated in both channel conventions).
// Reverse mode runs one seeded dual pass per requested input scalar and
// contracts the output tangents with the cotangents; the
// torch.autograd.Function layer in ops.py is dispatch only.
//
// The coupled row dual below mirrors coupled_rd_field_kernel
// (field_transport.cu) step by step, reusing the validated AD-1/AD-2 duals
// (slab_fresnel_dual, dual_reflect_frame) for the slab legs and the RayD
// templates for everything else. Edit the primal kernel and this mirror
// TOGETHER.

namespace {

constexpr int kBlockSize = 128;
namespace field = witwin::channel::native_ext;
namespace transport = channel_native::field_transport;
namespace ad = channel_native::field_transport_ad;

using Dual = field::Dual;
using DualV3 = field::Vec3T<Dual>;
using DualCx = field::ComplexT<Dual>;
using DualC3 = field::Complex3T<Dual>;

__device__ __forceinline__ field::float3a load3f(const float* values, int64_t index) {
    const int64_t base = index * 3;
    return field::make_f3(values[base], values[base + 1], values[base + 2]);
}

__device__ __forceinline__ c10::complex<float> to_c10(field::Complex value) {
    return c10::complex<float>(value.re, value.im);
}

__device__ __forceinline__ field::Complex from_c10(c10::complex<float> value) {
    return field::cplx(value.real(), value.imag());
}

// ---------------------------------------------------------------------------
// Bridges between the channel_native AD-1/2 dual types (DualC / DualF3) and
// the RayD dual scalars, so the validated slab and frame duals feed the
// templated pair math directly.
// ---------------------------------------------------------------------------

__device__ __forceinline__ DualV3 from_dualf3(ad::DualF3 value) {
    return field::dual_seed(value.v, value.d);
}

__device__ __forceinline__ ad::DualF3 to_dualf3(DualV3 value) {
    return {
        field::make_f3(value.x.v, value.y.v, value.z.v),
        field::make_f3(value.x.d, value.y.d, value.z.d)};
}

// Scalar/vector seeding shims shared with the diffraction-map companions
// (field_transport_ad.cuh).
using ad::seeded;
using ad::seeded3;

// Slab Fresnel / slab face operator on RayD duals: shared with the
// diffraction-map companions in field_transport_ad.cuh (single copy).
using ad::slab_face_operator_dual;
using ad::slab_fresnel_dual_rd;

// Dual of transport::precise_neg_kd (fmod has unit slope).
__device__ __forceinline__ Dual precise_neg_kd_dual(Dual k, Dual d) {
    return {transport::precise_neg_kd(k.v, d.v), -(k.d * d.v + k.v * d.d)};
}

// Mirror of transport::reflect_complex3 on RayD duals, built from the
// validated dual_reflect_frame + slab_fresnel_dual.
__device__ DualC3 reflect_complex3_dual(
    DualC3 value,
    DualV3 incident_direction,
    DualV3 normal,
    Dual eps_r,
    Dual sigma_e,
    float mu_r,
    Dual gain,
    Dual thickness,
    Dual frequency,
    DualV3& reflected_direction) {
    const ad::DualReflectFrame frame = ad::dual_reflect_frame(
        to_dualf3(incident_direction), to_dualf3(normal));
    reflected_direction = from_dualf3(frame.reflected_direction);
    DualCx r_te;
    DualCx r_tm;
    slab_fresnel_dual_rd(
        {frame.cos_theta.v, frame.cos_theta.d}, eps_r, sigma_e, mu_r, gain,
        thickness, frequency, r_te, r_tm);
    const DualV3 s_axis = from_dualf3(frame.s_axis);
    const DualV3 p_in = from_dualf3(frame.p_in);
    const DualV3 p_out = from_dualf3(frame.p_out);
    const DualCx e_s = field::cplx_dot_real(value, s_axis);
    const DualCx e_p = field::cplx_dot_real(value, p_in);
    return field::c3_add(
        field::cplx_scale_real(s_axis, field::cplx_mul(r_te, e_s)),
        field::cplx_scale_real(p_out, field::cplx_mul(r_tm, e_p)));
}

// ---------------------------------------------------------------------------
// Pure diffraction (component 2): re-evaluate RayD's order-1 wedge export
// from the frozen topology. One templated row serves the forward (float) and
// the derivative (Dual). The conventions below reproduce what the export
// actually does (paths_optix.cu + rayd/diffraction.cpp): stationary-point
// selection inside the pair, half-space Fresnel from the face materials
// (mat.omega = 2*pi*f > 0), the +z hard-coded tx polarization the bridge
// hands to RayD, and the sqrt(tx_power) amplitude scale. The forward-parity
// test against topology.field_xyz is the gate for these conventions.
// ---------------------------------------------------------------------------

struct WedgeRowInputs {
    field::float3a source;
    field::float3a target;
    field::float3a edge_pos;
    field::float3a edge_dir;
    float t_min;
    float t_max;
    field::float3a n0;
    field::float3a n1;
    float exterior_angle;
    float eps0, sigma0, mu0, gain0;
    float eps1, sigma1, mu1, gain1;
    bool valid0;
    bool valid1;
    float tx_power;
    float frequency;
    // Optional winner vertices (plan 07 section 9.3 mesh-vertex x
    // diffraction). When present, the row rebuilds the edge tables from them
    // so vertex seeds reach the edge geometry; the frozen tables above stay
    // the winner reference for sign/plane assignment.
    bool has_vertices;
    bool edge_boundary;
    field::float3a v0, v1, opp0, opp1;
};

struct WedgeRowSeeds {
    field::float3a source;
    field::float3a target;
    float eps0, sigma0, gain0;
    float eps1, sigma1, gain1;
    float frequency;
    field::float3a v0, v1, opp0, opp1;
};

__device__ __forceinline__ WedgeRowSeeds wedge_seeds_zero() {
    WedgeRowSeeds seeds;
    seeds.source = field::f3_zero();
    seeds.target = field::f3_zero();
    seeds.eps0 = 0.f; seeds.sigma0 = 0.f; seeds.gain0 = 0.f;
    seeds.eps1 = 0.f; seeds.sigma1 = 0.f; seeds.gain1 = 0.f;
    seeds.frequency = 0.f;
    seeds.v0 = field::f3_zero();
    seeds.v1 = field::f3_zero();
    seeds.opp0 = field::f3_zero();
    seeds.opp1 = field::f3_zero();
    return seeds;
}

// acos on the templated scalar via atan2 (utd has no acos shim); equals
// ::acosf to float rounding on [-1, 1].
template <typename T>
__device__ __forceinline__ T wedge_acos(T x) {
    return field::atan2f(field::sqrtf(field::fmaxf(1.0f - x * x, T(0.f))), x);
}

// Primal part of a templated vector (identity for float).
template <typename T>
__device__ __forceinline__ field::float3a primal3(field::Vec3T<T> v) {
    return field::make_f3(
        field::scalar_value(v.x), field::scalar_value(v.y),
        field::scalar_value(v.z));
}

// Differentiable edge tables rebuilt from the winner vertices. The frozen
// discovery tables (in.n0 / in.n1) pick the plane assignment and the normal
// signs, so RayD's winding/ordering conventions cannot drift the primal
// values; the derivative flows through the aligned smooth normal. Mirrors
// diffraction_edge_geometry_kernel (kernels/diffraction.cu) row math; edit
// the discovery kernel and this rebuild TOGETHER.
template <typename T>
struct WedgeEdgeTables {
    field::Vec3T<T> edge_pos;
    field::Vec3T<T> edge_dir;
    T t_min;
    T t_max;
    field::Vec3T<T> n0;
    field::Vec3T<T> n1;
    T wedge_n;  // exterior_angle / pi
};

template <typename T>
__device__ WedgeEdgeTables<T> wedge_edge_tables_from_vertices(
    const WedgeRowInputs& in, const WedgeRowSeeds& seeds) {
    WedgeEdgeTables<T> tables;
    const field::Vec3T<T> v0 = seeded3<T>(in.v0, seeds.v0);
    const field::Vec3T<T> v1 = seeded3<T>(in.v1, seeds.v1);
    const field::Vec3T<T> vector = field::f3_sub(v1, v0);
    const T length = field::fmaxf(field::safe_length(vector), T(1.0e-12f));
    tables.edge_dir = field::f3_mul(vector, 1.0f / length);
    tables.edge_pos = field::f3_mul(field::f3_add(v0, v1), 0.5f);
    tables.t_min = -0.5f * length;
    tables.t_max = 0.5f * length;

    const field::Vec3T<T> opp0 = seeded3<T>(in.opp0, seeds.opp0);
    const field::Vec3T<T> candidate_a = field::safe_normalize(
        field::f3_cross(vector, field::f3_sub(opp0, v0)),
        field::v3_const<T>(0.f, 0.f, 1.f));
    field::Vec3T<T> pick0 = candidate_a;
    field::Vec3T<T> pick1 = candidate_a;
    if (!in.edge_boundary) {
        const field::Vec3T<T> opp1 = seeded3<T>(in.opp1, seeds.opp1);
        const field::Vec3T<T> candidate_b = field::safe_normalize(
            field::f3_cross(vector, field::f3_sub(opp1, v0)),
            field::v3_const<T>(0.f, 0.f, 1.f));
        // Frozen plane assignment: match each candidate to the discovery
        // table slot it is (anti)parallel to (primal values only).
        const field::float3a a_val = primal3<T>(candidate_a);
        const bool a_is_n0 =
            ::fabsf(field::f3_dot(a_val, in.n0)) >=
            ::fabsf(field::f3_dot(a_val, in.n1));
        pick0 = a_is_n0 ? candidate_a : candidate_b;
        pick1 = a_is_n0 ? candidate_b : candidate_a;
    }
    // Frozen sign alignment against the discovery normals.
    const float sign0 =
        field::f3_dot(primal3<T>(pick0), in.n0) < 0.f ? -1.f : 1.f;
    tables.n0 = field::f3_mul(pick0, sign0);
    if (in.edge_boundary) {
        tables.n1 = field::f3_neg(tables.n0);
        tables.wedge_n = T(2.f);
    } else {
        const float sign1 =
            field::f3_dot(primal3<T>(pick1), in.n1) < 0.f ? -1.f : 1.f;
        tables.n1 = field::f3_mul(pick1, sign1);
        const T neg_dot = field::fminf(
            field::fmaxf(-field::f3_dot(tables.n0, tables.n1), T(-1.f)), T(1.f));
        const T exterior = 2.0f * field::UTD_PI - wedge_acos(neg_dot);
        tables.wedge_n = exterior * (1.0f / field::UTD_PI);
    }
    return tables;
}

template <typename T>
struct WedgeRowOutputs {
    field::Complex3T<T> field_vector;  // includes the sqrt(tx_power) scale
    field::Vec3T<T> direction;         // arrival from the clamped edge point
};

// RayD's face_material_params: absent faces keep the default material and
// present = 0 (the pair then treats the face operator as zero).
template <typename T>
__device__ __forceinline__ field::FaceMaterialParamsT<T> wedge_face_material(
    bool valid, T eps, T sigma, float mu, T gain) {
    if (!valid) {
        return {T(1.f), T(1.f), T(0.f), T(1.f), 1.f, 0.f};
    }
    return {eps, T(mu), sigma, field::fmaxf(gain, T(0.f)), 1.f, 1.f};
}

template <typename T>
__device__ WedgeRowOutputs<T> wedge_row_eval(
    const WedgeRowInputs& in, const WedgeRowSeeds& seeds) {
    const field::Vec3T<T> source = seeded3<T>(in.source, seeds.source);
    const field::Vec3T<T> target = seeded3<T>(in.target, seeds.target);
    const T frequency = seeded<T>(in.frequency, seeds.frequency);
    const T wave_number =
        2.0f * field::UTD_PI * frequency / transport::kSpeedOfLight;

    const field::float3a zero3 = field::f3_zero();
    field::PairInputsT<T> pair{};
    T edge_t_min;
    T edge_t_max;
    if (in.has_vertices) {
        const WedgeEdgeTables<T> tables =
            wedge_edge_tables_from_vertices<T>(in, seeds);
        pair.edgePos = tables.edge_pos;
        pair.edgeDir = tables.edge_dir;
        pair.n0 = tables.n0;
        pair.nn = tables.n1;
        pair.wedgeN = tables.wedge_n;
        edge_t_min = tables.t_min;
        edge_t_max = tables.t_max;
    } else {
        pair.edgePos = seeded3<T>(in.edge_pos, zero3);
        pair.edgeDir = seeded3<T>(in.edge_dir, zero3);
        pair.n0 = seeded3<T>(in.n0, zero3);
        pair.nn = seeded3<T>(in.n1, zero3);
        pair.wedgeN = T(in.exterior_angle / field::UTD_PI);
        edge_t_min = T(in.t_min);
        edge_t_max = T(in.t_max);
    }
    pair.edgeLineMin = edge_t_min;
    pair.edgeLineMax = edge_t_max;
    pair.sourcePos = source;
    pair.selectStationaryPoint = 1.f;
    pair.face0Material = wedge_face_material(
        in.valid0, seeded<T>(in.eps0, seeds.eps0),
        seeded<T>(in.sigma0, seeds.sigma0), in.mu0,
        seeded<T>(in.gain0, seeds.gain0));
    pair.face1Material = wedge_face_material(
        in.valid1, seeded<T>(in.eps1, seeds.eps1),
        seeded<T>(in.sigma1, seeds.sigma1), in.mu1,
        seeded<T>(in.gain1, seeds.gain1));

    field::MaterialParamsT<T> mat{};
    mat.useFresnel = 1;
    mat.etaR = T(1.f);
    mat.muR = T(1.f);
    mat.sigma = T(0.f);
    mat.gain = T(1.f);
    mat.omega = 2.0f * field::UTD_PI * frequency;
    // rayd/diffraction.cpp hands RayD a hard-coded +z tx polarization for the
    // order-1 diffraction export; reproduce it (forward-parity gate).
    mat.txPolX = T(0.f);
    mat.txPolY = T(0.f);
    mat.txPolZ = T(1.f);

    WedgeRowOutputs<T> out;
    const field::Complex3T<T> vec =
        field::compute_pair_vector_contribution(pair, target, wave_number, mat);
    const T amplitude = sqrtf(T(fmaxf(in.tx_power, 0.f)));
    out.field_vector = field::c3_scale_real(vec, amplitude);

    // Arrival direction from the clamped stationary point (the export's p0).
    const field::Vec3T<T> edge_hat = field::safe_normalize(
        pair.edgeDir, field::v3_const<T>(0.f, 0.f, 1.f));
    const T edge_length = edge_t_max - edge_t_min;
    const field::Vec3T<T> edge_origin = field::f3_add(
        pair.edgePos, field::f3_mul(edge_hat, edge_t_min));
    const T parameter = field::first_order_diffraction_parameter(
        source, target, edge_origin, edge_hat);
    const T clamped = field::fminf(field::fmaxf(parameter, T(0.f)), edge_length);
    const field::Vec3T<T> point = field::f3_add(
        edge_origin, field::f3_mul(edge_hat, clamped));
    out.direction = field::safe_normalize(
        field::f3_sub(target, point), field::v3_const<T>(0.f, 0.f, 1.f));
    return out;
}

__device__ __forceinline__ WedgeRowInputs load_wedge_row(
    int64_t index,
    const float* source,
    const float* target,
    const float* edge_position,
    const float* edge_direction,
    const float* edge_t_min,
    const float* edge_t_max,
    const float* edge_n0,
    const float* edge_n1,
    const float* exterior_angle,
    const bool* face0_valid,
    const float* face0_eps_r,
    const float* face0_sigma_e,
    const float* face0_mu_r,
    const float* face0_gain,
    const bool* face1_valid,
    const float* face1_eps_r,
    const float* face1_sigma_e,
    const float* face1_mu_r,
    const float* face1_gain,
    const float* tx_power,
    float frequency_hz,
    const float* vertex_v0,
    const float* vertex_v1,
    const float* vertex_opp0,
    const float* vertex_opp1,
    const bool* edge_boundary) {
    WedgeRowInputs in;
    in.source = load3f(source, index);
    in.target = load3f(target, index);
    in.edge_pos = load3f(edge_position, index);
    in.edge_dir = load3f(edge_direction, index);
    in.t_min = edge_t_min[index];
    in.t_max = edge_t_max[index];
    in.n0 = load3f(edge_n0, index);
    in.n1 = load3f(edge_n1, index);
    in.exterior_angle = exterior_angle[index];
    in.valid0 = face0_valid[index];
    in.eps0 = face0_eps_r[index];
    in.sigma0 = face0_sigma_e[index];
    in.mu0 = face0_mu_r[index];
    in.gain0 = face0_gain[index];
    in.valid1 = face1_valid[index];
    in.eps1 = face1_eps_r[index];
    in.sigma1 = face1_sigma_e[index];
    in.mu1 = face1_mu_r[index];
    in.gain1 = face1_gain[index];
    in.tx_power = tx_power[index];
    in.frequency = frequency_hz;
    in.has_vertices = vertex_v0 != nullptr;
    if (in.has_vertices) {
        in.edge_boundary = edge_boundary[index];
        in.v0 = load3f(vertex_v0, index);
        in.v1 = load3f(vertex_v1, index);
        in.opp0 = load3f(vertex_opp0, index);
        in.opp1 = load3f(vertex_opp1, index);
    } else {
        in.edge_boundary = false;
        in.v0 = field::f3_zero();
        in.v1 = field::f3_zero();
        in.opp0 = field::f3_zero();
        in.opp1 = field::f3_zero();
    }
    return in;
}

#define WEDGE_ROW_PARAMS                                                      \
    const float* source, const float* target, const float* edge_position,    \
        const float* edge_direction, const float* edge_t_min,                 \
        const float* edge_t_max, const float* edge_n0, const float* edge_n1,  \
        const float* exterior_angle, const bool* face0_valid,                 \
        const float* face0_eps_r, const float* face0_sigma_e,                 \
        const float* face0_mu_r, const float* face0_gain,                     \
        const bool* face1_valid, const float* face1_eps_r,                    \
        const float* face1_sigma_e, const float* face1_mu_r,                  \
        const float* face1_gain, const float* tx_power, float frequency_hz,   \
        const float* vertex_v0, const float* vertex_v1,                       \
        const float* vertex_opp0, const float* vertex_opp1,                   \
        const bool* edge_boundary

#define WEDGE_ROW_ARGS(index)                                                 \
    index, source, target, edge_position, edge_direction, edge_t_min,         \
        edge_t_max, edge_n0, edge_n1, exterior_angle, face0_valid,            \
        face0_eps_r, face0_sigma_e, face0_mu_r, face0_gain, face1_valid,      \
        face1_eps_r, face1_sigma_e, face1_mu_r, face1_gain, tx_power,         \
        frequency_hz, vertex_v0, vertex_v1, vertex_opp0, vertex_opp1,         \
        edge_boundary

__global__ void diffraction_wedge_forward_kernel(
    int64_t count,
    WEDGE_ROW_PARAMS,
    c10::complex<float>* field_vector,
    float* direction) {
    for (int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < count;
         index += static_cast<int64_t>(blockDim.x) * gridDim.x) {
        const WedgeRowInputs in = load_wedge_row(WEDGE_ROW_ARGS(index));
        const WedgeRowOutputs<float> out =
            wedge_row_eval<float>(in, wedge_seeds_zero());
        const int64_t base = index * 3;
        field_vector[base] = to_c10(out.field_vector.x);
        field_vector[base + 1] = to_c10(out.field_vector.y);
        field_vector[base + 2] = to_c10(out.field_vector.z);
        direction[base] = out.direction.x;
        direction[base + 1] = out.direction.y;
        direction[base + 2] = out.direction.z;
    }
}

__device__ __forceinline__ float wedge_contract(
    int64_t index,
    const c10::complex<float>* grad_field_vector,
    const float* grad_direction,
    const WedgeRowOutputs<Dual>& out) {
    float acc = 0.f;
    const int64_t base = index * 3;
    if (grad_field_vector != nullptr) {
        const field::Complex3 tangent = field::dual_tangent(out.field_vector);
        acc += field::cplx_adj_dot(from_c10(grad_field_vector[base]), tangent.x);
        acc += field::cplx_adj_dot(from_c10(grad_field_vector[base + 1]), tangent.y);
        acc += field::cplx_adj_dot(from_c10(grad_field_vector[base + 2]), tangent.z);
    }
    if (grad_direction != nullptr) {
        acc += grad_direction[base] * out.direction.x.d;
        acc += grad_direction[base + 1] * out.direction.y.d;
        acc += grad_direction[base + 2] * out.direction.z.d;
    }
    return acc;
}

__global__ void diffraction_wedge_backward_kernel(
    int64_t count,
    WEDGE_ROW_PARAMS,
    const c10::complex<float>* grad_field_vector,
    const float* grad_direction,
    float* grad_source,
    float* grad_target,
    float* grad_face0_eps_r,
    float* grad_face0_sigma_e,
    float* grad_face0_gain,
    float* grad_face1_eps_r,
    float* grad_face1_sigma_e,
    float* grad_face1_gain,
    float* grad_frequency,
    float* grad_vertex_v0,
    float* grad_vertex_v1,
    float* grad_vertex_opp0,
    float* grad_vertex_opp1) {
    for (int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < count;
         index += static_cast<int64_t>(blockDim.x) * gridDim.x) {
        const WedgeRowInputs in = load_wedge_row(WEDGE_ROW_ARGS(index));
        WedgeRowSeeds seeds = wedge_seeds_zero();
        const int64_t base = index * 3;
        if (grad_source != nullptr) {
            float* slots[3] = {&seeds.source.x, &seeds.source.y, &seeds.source.z};
            for (int axis = 0; axis < 3; ++axis) {
                *slots[axis] = 1.f;
                const WedgeRowOutputs<Dual> out = wedge_row_eval<Dual>(in, seeds);
                *slots[axis] = 0.f;
                grad_source[base + axis] = wedge_contract(
                    index, grad_field_vector, grad_direction, out);
            }
        }
        if (grad_target != nullptr) {
            float* slots[3] = {&seeds.target.x, &seeds.target.y, &seeds.target.z};
            for (int axis = 0; axis < 3; ++axis) {
                *slots[axis] = 1.f;
                const WedgeRowOutputs<Dual> out = wedge_row_eval<Dual>(in, seeds);
                *slots[axis] = 0.f;
                grad_target[base + axis] = wedge_contract(
                    index, grad_field_vector, grad_direction, out);
            }
        }
        if (grad_vertex_v0 != nullptr) {
            struct VertexSlot {
                field::float3a* seed;
                float* grad;
            };
            VertexSlot vertex_slots[4] = {
                {&seeds.v0, grad_vertex_v0},
                {&seeds.v1, grad_vertex_v1},
                {&seeds.opp0, grad_vertex_opp0},
                {&seeds.opp1, grad_vertex_opp1},
            };
            for (int slot = 0; slot < 4; ++slot) {
                float* components[3] = {
                    &vertex_slots[slot].seed->x,
                    &vertex_slots[slot].seed->y,
                    &vertex_slots[slot].seed->z,
                };
                for (int axis = 0; axis < 3; ++axis) {
                    *components[axis] = 1.f;
                    const WedgeRowOutputs<Dual> out =
                        wedge_row_eval<Dual>(in, seeds);
                    *components[axis] = 0.f;
                    vertex_slots[slot].grad[base + axis] = wedge_contract(
                        index, grad_field_vector, grad_direction, out);
                }
            }
        }
        struct MaterialSlot {
            float* seed;
            float* grad;
        };
        MaterialSlot material_slots[6] = {
            {&seeds.eps0, grad_face0_eps_r},
            {&seeds.sigma0, grad_face0_sigma_e},
            {&seeds.gain0, grad_face0_gain},
            {&seeds.eps1, grad_face1_eps_r},
            {&seeds.sigma1, grad_face1_sigma_e},
            {&seeds.gain1, grad_face1_gain},
        };
        for (int slot = 0; slot < 6; ++slot) {
            if (material_slots[slot].grad == nullptr)
                continue;
            *material_slots[slot].seed = 1.f;
            const WedgeRowOutputs<Dual> out = wedge_row_eval<Dual>(in, seeds);
            *material_slots[slot].seed = 0.f;
            material_slots[slot].grad[index] = wedge_contract(
                index, grad_field_vector, grad_direction, out);
        }
        if (grad_frequency != nullptr) {
            seeds.frequency = 1.f;
            const WedgeRowOutputs<Dual> out = wedge_row_eval<Dual>(in, seeds);
            seeds.frequency = 0.f;
            atomicAdd(grad_frequency, wedge_contract(
                index, grad_field_vector, grad_direction, out));
        }
    }
}

__global__ void diffraction_wedge_jvp_kernel(
    int64_t count,
    WEDGE_ROW_PARAMS,
    const float* tangent_source,
    const float* tangent_target,
    const float* tangent_face0_eps_r,
    const float* tangent_face0_sigma_e,
    const float* tangent_face0_gain,
    const float* tangent_face1_eps_r,
    const float* tangent_face1_sigma_e,
    const float* tangent_face1_gain,
    float tangent_frequency,
    const float* tangent_vertex_v0,
    const float* tangent_vertex_v1,
    const float* tangent_vertex_opp0,
    const float* tangent_vertex_opp1,
    c10::complex<float>* tangent_field_vector,
    float* tangent_direction) {
    for (int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < count;
         index += static_cast<int64_t>(blockDim.x) * gridDim.x) {
        const WedgeRowInputs in = load_wedge_row(WEDGE_ROW_ARGS(index));
        WedgeRowSeeds seeds = wedge_seeds_zero();
        if (tangent_source != nullptr)
            seeds.source = load3f(tangent_source, index);
        if (tangent_target != nullptr)
            seeds.target = load3f(tangent_target, index);
        if (tangent_face0_eps_r != nullptr)
            seeds.eps0 = tangent_face0_eps_r[index];
        if (tangent_face0_sigma_e != nullptr)
            seeds.sigma0 = tangent_face0_sigma_e[index];
        if (tangent_face0_gain != nullptr)
            seeds.gain0 = tangent_face0_gain[index];
        if (tangent_face1_eps_r != nullptr)
            seeds.eps1 = tangent_face1_eps_r[index];
        if (tangent_face1_sigma_e != nullptr)
            seeds.sigma1 = tangent_face1_sigma_e[index];
        if (tangent_face1_gain != nullptr)
            seeds.gain1 = tangent_face1_gain[index];
        if (tangent_vertex_v0 != nullptr)
            seeds.v0 = load3f(tangent_vertex_v0, index);
        if (tangent_vertex_v1 != nullptr)
            seeds.v1 = load3f(tangent_vertex_v1, index);
        if (tangent_vertex_opp0 != nullptr)
            seeds.opp0 = load3f(tangent_vertex_opp0, index);
        if (tangent_vertex_opp1 != nullptr)
            seeds.opp1 = load3f(tangent_vertex_opp1, index);
        seeds.frequency = tangent_frequency;
        const WedgeRowOutputs<Dual> out = wedge_row_eval<Dual>(in, seeds);
        const field::Complex3 tangent = field::dual_tangent(out.field_vector);
        const int64_t base = index * 3;
        tangent_field_vector[base] = to_c10(tangent.x);
        tangent_field_vector[base + 1] = to_c10(tangent.y);
        tangent_field_vector[base + 2] = to_c10(tangent.z);
        tangent_direction[base] = out.direction.x.d;
        tangent_direction[base + 1] = out.direction.y.d;
        tangent_direction[base + 2] = out.direction.z.d;
    }
}

// ---------------------------------------------------------------------------
// Coupled reflection-diffraction dual row (components 3/4). Mirrors
// coupled_rd_field_kernel step by step on RayD duals.
//
// Truncation-factor policy: the primal evaluates the pair with +/-1e5
// pseudo-infinite edge bounds, where the Boersma endpoint ripple makes the
// truncation factor's derivative float32 noise amplified by the 1e5 lever
// arm (the true infinite-edge derivative is zero; see the plan 07 AD-4a
// notes). The dual therefore freezes that factor at its primal value.
// ---------------------------------------------------------------------------

struct CoupledRowInputs {
    field::float3a source;
    field::float3a target;
    field::float3a hit;
    field::float3a normal;
    field::float3a edge;
    field::float3a edge_dir_raw;
    field::float3a n0;
    field::float3a n1;
    float exterior_angle;
    field::float3a tx_pol;
    field::float3a rx_pol;
    float tx_power;
    float refl_eps, refl_sigma, refl_mu, refl_gain, refl_thickness;
    float w0_eps, w0_sigma, w0_mu, w0_gain, w0_thickness;
    float w1_eps, w1_sigma, w1_mu, w1_gain, w1_thickness;
    float frequency;
    bool reverse;
};

struct CoupledSeeds {
    field::float3a source;
    field::float3a target;
    field::float3a hit;
    field::float3a edge;
    float refl_eps, refl_sigma, refl_gain, refl_thickness;
    float w0_eps, w0_sigma, w0_gain, w0_thickness;
    float w1_eps, w1_sigma, w1_gain, w1_thickness;
    float frequency;
};

__device__ __forceinline__ CoupledSeeds coupled_seeds_zero() {
    CoupledSeeds seeds;
    seeds.source = field::f3_zero();
    seeds.target = field::f3_zero();
    seeds.hit = field::f3_zero();
    seeds.edge = field::f3_zero();
    seeds.refl_eps = 0.f; seeds.refl_sigma = 0.f;
    seeds.refl_gain = 0.f; seeds.refl_thickness = 0.f;
    seeds.w0_eps = 0.f; seeds.w0_sigma = 0.f;
    seeds.w0_gain = 0.f; seeds.w0_thickness = 0.f;
    seeds.w1_eps = 0.f; seeds.w1_sigma = 0.f;
    seeds.w1_gain = 0.f; seeds.w1_thickness = 0.f;
    seeds.frequency = 0.f;
    return seeds;
}

struct CoupledRowTangents {
    field::Complex3 field_vector;
    field::Complex coefficient;
    field::Complex path_field;
    float path_gain;
};

__device__ __forceinline__ DualV3 reflect_point_dual(
    DualV3 point, DualV3 plane_point, DualV3 raw_normal) {
    const DualV3 n = field::safe_normalize(
        raw_normal, field::v3_const<Dual>(0.f, 0.f, 1.f));
    return field::f3_sub(
        point,
        field::f3_mul(n, 2.0f * field::f3_dot(field::f3_sub(point, plane_point), n)));
}

__device__ CoupledRowTangents coupled_rd_row_dual(
    const CoupledRowInputs& in, const CoupledSeeds& seeds) {
    const DualV3 src = field::dual_seed(in.source, seeds.source);
    const DualV3 dst = field::dual_seed(in.target, seeds.target);
    const DualV3 hit = field::dual_seed(in.hit, seeds.hit);
    const DualV3 edge = field::dual_seed(in.edge, seeds.edge);
    const DualV3 normal = field::dual_const3(in.normal);
    const Dual frequency = {in.frequency, seeds.frequency};
    const Dual wave_number =
        2.0f * field::UTD_PI * frequency / transport::kSpeedOfLight;
    const DualV3 edge_axis = field::safe_normalize(
        field::dual_const3(in.edge_dir_raw), field::v3_const<Dual>(0.f, 0.f, 1.f));

    const Dual refl_eps = {in.refl_eps, seeds.refl_eps};
    const Dual refl_sigma = {in.refl_sigma, seeds.refl_sigma};
    const Dual refl_gain = {in.refl_gain, seeds.refl_gain};
    const Dual refl_thickness = {in.refl_thickness, seeds.refl_thickness};

    const DualV3 diffraction_source =
        in.reverse ? src : reflect_point_dual(src, hit, normal);
    const DualV3 diffraction_target =
        in.reverse ? reflect_point_dual(dst, hit, normal) : dst;
    const DualV3 incident_direction = field::safe_normalize(
        field::f3_sub(edge, diffraction_source),
        field::v3_const<Dual>(1.f, 0.f, 0.f));
    const DualV3 outgoing_direction = field::safe_normalize(
        field::f3_sub(diffraction_target, edge),
        field::v3_const<Dual>(1.f, 0.f, 0.f));
    const field::Basis3T<Dual> input_edge_basis = field::diffraction_edge_basis(
        field::f3_sub(edge, diffraction_source), edge_axis, false);
    const field::Basis3T<Dual> output_edge_basis = field::diffraction_edge_basis(
        field::f3_sub(diffraction_target, edge), edge_axis, true);

    DualC3 incident_field;
    if (in.reverse) {
        // Mirror of transport::free_space_complex3.
        const DualV3 offset = field::f3_sub(edge, src);
        const Dual distance = field::safe_length(offset);
        const DualV3 direction = field::safe_normalize(
            offset, field::v3_const<Dual>(0.f, 0.f, 1.f));
        const DualV3 axis = field::stable_perp_basis(
            direction, field::dual_const3(in.tx_pol));
        const Dual amplitude =
            1.0f / (2.0f * field::fmaxf(wave_number, Dual(field::UTD_SMALL_EPS)) *
                    field::fmaxf(distance, Dual(field::UTD_EPS)));
        const DualCx phase = field::cplx_exp_phase(
            precise_neg_kd_dual(wave_number, distance));
        incident_field = field::cplx_scale_real(
            axis, field::cplx_mul_real(phase, amplitude));
    } else {
        const DualV3 source_to_hit = field::safe_normalize(
            field::f3_sub(hit, src), incident_direction);
        const DualV3 tx_axis = field::stable_perp_basis(
            source_to_hit, field::dual_const3(in.tx_pol));
        DualV3 reflected_direction;
        const DualC3 reflected = reflect_complex3_dual(
            field::cplx_scale_real(tx_axis, field::c_const<Dual>(1.f, 0.f)),
            source_to_hit,
            normal,
            refl_eps,
            refl_sigma,
            in.refl_mu,
            refl_gain,
            refl_thickness,
            frequency,
            reflected_direction);
        const Dual unfolded_distance = field::safe_length(
            field::f3_sub(edge, diffraction_source));
        const Dual amplitude =
            1.0f / (2.0f * wave_number *
                    field::fmaxf(unfolded_distance, Dual(field::UTD_EPS)));
        const DualCx propagation = field::cplx_mul_real(
            field::cplx_exp_phase(
                precise_neg_kd_dual(wave_number, unfolded_distance)),
            amplitude);
        incident_field = field::c3_scale(reflected, propagation);
    }

    field::PairInputsT<Dual> pair{};
    pair.edgePos = edge;
    pair.edgeDir = edge_axis;
    pair.n0 = field::dual_const3(in.n0);
    pair.nn = field::dual_const3(in.n1);
    pair.wedgeN = Dual(in.exterior_angle / field::UTD_PI);
    pair.edgeLineMin = Dual(-1.0e5f);
    pair.edgeLineMax = Dual(1.0e5f);
    pair.sourcePos = diffraction_source;
    pair.incidentBasis = input_edge_basis;
    pair.incidentJones = field::jones_from_vector(incident_field, input_edge_basis);
    pair.incidentDerivativeJones = field::jones_zero<Dual>();
    pair.face0Material.present = 1.0f;
    pair.face1Material.present = 1.0f;
    pair.face0Operator = slab_face_operator_dual(
        field::fabsf(field::f3_dot(pair.n0, incident_direction)),
        {in.w0_eps, seeds.w0_eps},
        {in.w0_sigma, seeds.w0_sigma},
        in.w0_mu,
        {in.w0_gain, seeds.w0_gain},
        {in.w0_thickness, seeds.w0_thickness},
        frequency,
        pair.n0,
        incident_direction,
        outgoing_direction,
        input_edge_basis,
        output_edge_basis);
    pair.face1Operator = slab_face_operator_dual(
        field::fabsf(field::f3_dot(pair.nn, incident_direction)),
        {in.w1_eps, seeds.w1_eps},
        {in.w1_sigma, seeds.w1_sigma},
        in.w1_mu,
        {in.w1_gain, seeds.w1_gain},
        {in.w1_thickness, seeds.w1_thickness},
        frequency,
        pair.nn,
        incident_direction,
        outgoing_direction,
        input_edge_basis,
        output_edge_basis);
    pair.selectStationaryPoint = 0.0f;
    field::MaterialParamsT<Dual> material{};
    material.omega = Dual(-1.0f);

    // Pair vector path (compute_pair_vector_contribution with
    // selectStationaryPoint = 0), with the truncation factor frozen.
    DualC3 value = field::c3_zero<Dual>();
    const bool src_ext = field::wedge_exterior_mask(
        field::f3_sub(pair.sourcePos, pair.edgePos), pair.edgeDir, pair.n0, pair.nn);
    Dual phi, phi_p, s, s_p, sb;
    field::compute_edge_geometry_3d(
        pair.sourcePos, pair.edgePos, pair.edgeDir, pair.n0, diffraction_target,
        phi, phi_p, s, s_p, sb);
    const bool geom_valid =
        src_ext && (s_p > field::UTD_MIN_DISTANCE) && (s > field::UTD_MIN_DISTANCE);
    if (geom_valid) {
        DualCx finite_factor = field::finite_wedge_truncation_factor(
            pair, diffraction_target, wave_number);
        finite_factor.re.d = 0.f;
        finite_factor.im.d = 0.f;
        value = field::compute_pair_vector_at_angles(
            pair, diffraction_target, wave_number, material, phi, phi_p, s, s_p,
            sb, input_edge_basis, output_edge_basis, finite_factor, false);
    }

    DualV3 final_direction = outgoing_direction;
    if (in.reverse) {
        value = reflect_complex3_dual(
            value,
            field::safe_normalize(field::f3_sub(hit, edge), outgoing_direction),
            normal,
            refl_eps,
            refl_sigma,
            in.refl_mu,
            refl_gain,
            refl_thickness,
            frequency,
            final_direction);
    }

    const DualV3 rx_axis = field::stable_perp_basis(
        final_direction, field::dual_const3(in.rx_pol));
    const DualCx coefficient = field::cplx_dot_real(value, rx_axis);
    const float amplitude_scale = sqrtf(fmaxf(in.tx_power, 0.f));
    const DualCx path_field = field::cplx_mul_real(coefficient, amplitude_scale);

    CoupledRowTangents out;
    out.field_vector = field::dual_tangent(value);
    out.coefficient = field::dual_tangent(coefficient);
    out.path_field = field::dual_tangent(path_field);
    out.path_gain = 2.0f * (path_field.re.v * path_field.re.d +
                            path_field.im.v * path_field.im.d);
    return out;
}

__device__ __forceinline__ CoupledRowInputs load_coupled_row(
    int64_t index,
    const float* source,
    const float* target,
    const float* reflection_position,
    const float* reflection_normal,
    const float* edge_position,
    const float* edge_direction,
    const float* edge_n0,
    const float* edge_n1,
    const float* exterior_angle,
    const float* tx_power,
    const float* tx_polarization,
    const float* rx_polarization,
    const float* reflection_eps_r,
    const float* reflection_sigma_e,
    const float* reflection_mu_r,
    const float* reflection_gain,
    const float* reflection_thickness,
    const float* wedge_eps_r0,
    const float* wedge_sigma_e0,
    const float* wedge_mu_r0,
    const float* wedge_gain0,
    const float* wedge_thickness0,
    const float* wedge_eps_r1,
    const float* wedge_sigma_e1,
    const float* wedge_mu_r1,
    const float* wedge_gain1,
    const float* wedge_thickness1,
    float frequency_hz,
    bool reverse) {
    CoupledRowInputs in;
    in.source = load3f(source, index);
    in.target = load3f(target, index);
    in.hit = load3f(reflection_position, index);
    in.normal = load3f(reflection_normal, index);
    in.edge = load3f(edge_position, index);
    in.edge_dir_raw = load3f(edge_direction, index);
    in.n0 = load3f(edge_n0, index);
    in.n1 = load3f(edge_n1, index);
    in.exterior_angle = exterior_angle[index];
    in.tx_pol = load3f(tx_polarization, index);
    in.rx_pol = load3f(rx_polarization, index);
    in.tx_power = tx_power[index];
    in.refl_eps = reflection_eps_r[index];
    in.refl_sigma = reflection_sigma_e[index];
    in.refl_mu = reflection_mu_r[index];
    in.refl_gain = reflection_gain[index];
    in.refl_thickness = reflection_thickness[index];
    in.w0_eps = wedge_eps_r0[index];
    in.w0_sigma = wedge_sigma_e0[index];
    in.w0_mu = wedge_mu_r0[index];
    in.w0_gain = wedge_gain0[index];
    in.w0_thickness = wedge_thickness0[index];
    in.w1_eps = wedge_eps_r1[index];
    in.w1_sigma = wedge_sigma_e1[index];
    in.w1_mu = wedge_mu_r1[index];
    in.w1_gain = wedge_gain1[index];
    in.w1_thickness = wedge_thickness1[index];
    in.frequency = frequency_hz;
    in.reverse = reverse;
    return in;
}

#define COUPLED_ROW_PARAMS                                                    \
    const float* source, const float* target,                                 \
        const float* reflection_position, const float* reflection_normal,     \
        const float* edge_position, const float* edge_direction,              \
        const float* edge_n0, const float* edge_n1,                           \
        const float* exterior_angle, const float* tx_power,                   \
        const float* tx_polarization, const float* rx_polarization,           \
        const float* reflection_eps_r, const float* reflection_sigma_e,       \
        const float* reflection_mu_r, const float* reflection_gain,           \
        const float* reflection_thickness, const float* wedge_eps_r0,         \
        const float* wedge_sigma_e0, const float* wedge_mu_r0,                \
        const float* wedge_gain0, const float* wedge_thickness0,              \
        const float* wedge_eps_r1, const float* wedge_sigma_e1,               \
        const float* wedge_mu_r1, const float* wedge_gain1,                   \
        const float* wedge_thickness1, float frequency_hz, bool reverse

#define COUPLED_ROW_ARGS(index)                                               \
    index, source, target, reflection_position, reflection_normal,            \
        edge_position, edge_direction, edge_n0, edge_n1, exterior_angle,      \
        tx_power, tx_polarization, rx_polarization, reflection_eps_r,         \
        reflection_sigma_e, reflection_mu_r, reflection_gain,                 \
        reflection_thickness, wedge_eps_r0, wedge_sigma_e0, wedge_mu_r0,      \
        wedge_gain0, wedge_thickness0, wedge_eps_r1, wedge_sigma_e1,          \
        wedge_mu_r1, wedge_gain1, wedge_thickness1, frequency_hz, reverse

__device__ __forceinline__ float coupled_contract(
    int64_t index,
    const c10::complex<float>* grad_field_vector,
    const c10::complex<float>* grad_coefficient,
    const c10::complex<float>* grad_path_field,
    const float* grad_path_gain,
    const CoupledRowTangents& t) {
    float acc = 0.f;
    if (grad_field_vector != nullptr) {
        const int64_t base = index * 3;
        acc += field::cplx_adj_dot(from_c10(grad_field_vector[base]), t.field_vector.x);
        acc += field::cplx_adj_dot(from_c10(grad_field_vector[base + 1]), t.field_vector.y);
        acc += field::cplx_adj_dot(from_c10(grad_field_vector[base + 2]), t.field_vector.z);
    }
    if (grad_coefficient != nullptr)
        acc += field::cplx_adj_dot(from_c10(grad_coefficient[index]), t.coefficient);
    if (grad_path_field != nullptr)
        acc += field::cplx_adj_dot(from_c10(grad_path_field[index]), t.path_field);
    if (grad_path_gain != nullptr)
        acc += grad_path_gain[index] * t.path_gain;
    return acc;
}

__global__ void coupled_rd_backward_kernel(
    int64_t count,
    COUPLED_ROW_PARAMS,
    const c10::complex<float>* grad_field_vector,
    const c10::complex<float>* grad_coefficient,
    const c10::complex<float>* grad_path_field,
    const float* grad_path_gain,
    float* grad_source,
    float* grad_target,
    float* grad_reflection_position,
    float* grad_edge_position,
    float* grad_eps_r,        // (N, 3): reflection, wedge0, wedge1
    float* grad_sigma_e,
    float* grad_gain,
    float* grad_thickness,
    float* grad_frequency) {
    for (int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < count;
         index += static_cast<int64_t>(blockDim.x) * gridDim.x) {
        const CoupledRowInputs in = load_coupled_row(COUPLED_ROW_ARGS(index));
        CoupledSeeds seeds = coupled_seeds_zero();
        const int64_t base = index * 3;

        struct VectorSlot {
            field::float3a* seed;
            float* grad;
        };
        VectorSlot vector_slots[4] = {
            {&seeds.source, grad_source},
            {&seeds.target, grad_target},
            {&seeds.hit, grad_reflection_position},
            {&seeds.edge, grad_edge_position},
        };
        for (int slot = 0; slot < 4; ++slot) {
            if (vector_slots[slot].grad == nullptr)
                continue;
            float* axes[3] = {
                &vector_slots[slot].seed->x,
                &vector_slots[slot].seed->y,
                &vector_slots[slot].seed->z};
            for (int axis = 0; axis < 3; ++axis) {
                *axes[axis] = 1.f;
                const CoupledRowTangents t = coupled_rd_row_dual(in, seeds);
                *axes[axis] = 0.f;
                vector_slots[slot].grad[base + axis] = coupled_contract(
                    index, grad_field_vector, grad_coefficient, grad_path_field,
                    grad_path_gain, t);
            }
        }

        struct MaterialSlot {
            float* seed;
            float* grad;
            int column;
        };
        MaterialSlot material_slots[12] = {
            {&seeds.refl_eps, grad_eps_r, 0},
            {&seeds.w0_eps, grad_eps_r, 1},
            {&seeds.w1_eps, grad_eps_r, 2},
            {&seeds.refl_sigma, grad_sigma_e, 0},
            {&seeds.w0_sigma, grad_sigma_e, 1},
            {&seeds.w1_sigma, grad_sigma_e, 2},
            {&seeds.refl_gain, grad_gain, 0},
            {&seeds.w0_gain, grad_gain, 1},
            {&seeds.w1_gain, grad_gain, 2},
            {&seeds.refl_thickness, grad_thickness, 0},
            {&seeds.w0_thickness, grad_thickness, 1},
            {&seeds.w1_thickness, grad_thickness, 2},
        };
        for (int slot = 0; slot < 12; ++slot) {
            if (material_slots[slot].grad == nullptr)
                continue;
            *material_slots[slot].seed = 1.f;
            const CoupledRowTangents t = coupled_rd_row_dual(in, seeds);
            *material_slots[slot].seed = 0.f;
            material_slots[slot].grad[base + material_slots[slot].column] =
                coupled_contract(
                    index, grad_field_vector, grad_coefficient, grad_path_field,
                    grad_path_gain, t);
        }
        if (grad_frequency != nullptr) {
            seeds.frequency = 1.f;
            const CoupledRowTangents t = coupled_rd_row_dual(in, seeds);
            seeds.frequency = 0.f;
            atomicAdd(grad_frequency, coupled_contract(
                index, grad_field_vector, grad_coefficient, grad_path_field,
                grad_path_gain, t));
        }
    }
}

__global__ void coupled_rd_jvp_kernel(
    int64_t count,
    COUPLED_ROW_PARAMS,
    const float* tangent_source,
    const float* tangent_target,
    const float* tangent_reflection_position,
    const float* tangent_edge_position,
    const float* tangent_eps_r,     // (N, 3): reflection, wedge0, wedge1
    const float* tangent_sigma_e,
    const float* tangent_gain,
    const float* tangent_thickness,
    float tangent_frequency,
    c10::complex<float>* tangent_field_vector,
    c10::complex<float>* tangent_coefficient,
    c10::complex<float>* tangent_path_field,
    float* tangent_path_gain) {
    for (int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < count;
         index += static_cast<int64_t>(blockDim.x) * gridDim.x) {
        const CoupledRowInputs in = load_coupled_row(COUPLED_ROW_ARGS(index));
        CoupledSeeds seeds = coupled_seeds_zero();
        const int64_t base = index * 3;
        if (tangent_source != nullptr)
            seeds.source = load3f(tangent_source, index);
        if (tangent_target != nullptr)
            seeds.target = load3f(tangent_target, index);
        if (tangent_reflection_position != nullptr)
            seeds.hit = load3f(tangent_reflection_position, index);
        if (tangent_edge_position != nullptr)
            seeds.edge = load3f(tangent_edge_position, index);
        if (tangent_eps_r != nullptr) {
            seeds.refl_eps = tangent_eps_r[base];
            seeds.w0_eps = tangent_eps_r[base + 1];
            seeds.w1_eps = tangent_eps_r[base + 2];
        }
        if (tangent_sigma_e != nullptr) {
            seeds.refl_sigma = tangent_sigma_e[base];
            seeds.w0_sigma = tangent_sigma_e[base + 1];
            seeds.w1_sigma = tangent_sigma_e[base + 2];
        }
        if (tangent_gain != nullptr) {
            seeds.refl_gain = tangent_gain[base];
            seeds.w0_gain = tangent_gain[base + 1];
            seeds.w1_gain = tangent_gain[base + 2];
        }
        if (tangent_thickness != nullptr) {
            seeds.refl_thickness = tangent_thickness[base];
            seeds.w0_thickness = tangent_thickness[base + 1];
            seeds.w1_thickness = tangent_thickness[base + 2];
        }
        seeds.frequency = tangent_frequency;
        const CoupledRowTangents t = coupled_rd_row_dual(in, seeds);
        tangent_field_vector[base] = to_c10(t.field_vector.x);
        tangent_field_vector[base + 1] = to_c10(t.field_vector.y);
        tangent_field_vector[base + 2] = to_c10(t.field_vector.z);
        tangent_coefficient[index] = to_c10(t.coefficient);
        tangent_path_field[index] = to_c10(t.path_field);
        tangent_path_gain[index] = t.path_gain;
    }
}

// ---------------------------------------------------------------------------
// field_project_complex3 companions: coefficient = <field, axis(direction)>
// with axis = stable_perp_basis(direction, rx_pol); path_gain = |coeff|^2.
// Linear in the field vector; direction feeds the axis.
// ---------------------------------------------------------------------------

__global__ void project_complex3_backward_kernel(
    int64_t count,
    const c10::complex<float>* field_vector,
    const float* direction,
    const float* rx_polarization,
    const c10::complex<float>* grad_coefficient,
    const float* grad_path_gain,
    c10::complex<float>* grad_field_vector,
    float* grad_direction) {
    for (int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < count;
         index += static_cast<int64_t>(blockDim.x) * gridDim.x) {
        const int64_t base = index * 3;
        const field::Complex3 value = {
            from_c10(field_vector[base]),
            from_c10(field_vector[base + 1]),
            from_c10(field_vector[base + 2]),
        };
        const field::float3a dir = load3f(direction, index);
        const field::float3a pol = load3f(rx_polarization, index);
        const field::float3a axis = field::stable_perp_basis(dir, pol);
        const field::Complex coefficient = transport::complex3_dot_real(value, axis);
        field::Complex g_coeff = field::cplx_zero();
        if (grad_coefficient != nullptr)
            g_coeff = from_c10(grad_coefficient[index]);
        if (grad_path_gain != nullptr) {
            const float g_gain = grad_path_gain[index];
            g_coeff.re += 2.0f * coefficient.re * g_gain;
            g_coeff.im += 2.0f * coefficient.im * g_gain;
        }
        field::Complex3 g_value = field::c3_zero();
        field::float3a g_axis = field::f3_zero();
        field::adj_cplx_dot_real(value, axis, g_coeff, g_value, g_axis);
        if (grad_field_vector != nullptr) {
            grad_field_vector[base] = to_c10(g_value.x);
            grad_field_vector[base + 1] = to_c10(g_value.y);
            grad_field_vector[base + 2] = to_c10(g_value.z);
        }
        if (grad_direction != nullptr) {
            field::float3a g_dir = field::f3_zero();
            field::float3a g_pol = field::f3_zero();
            field::adj_stable_perp_basis(dir, pol, g_axis, g_dir, g_pol);
            grad_direction[base] = g_dir.x;
            grad_direction[base + 1] = g_dir.y;
            grad_direction[base + 2] = g_dir.z;
        }
    }
}

__global__ void project_complex3_jvp_kernel(
    int64_t count,
    const c10::complex<float>* field_vector,
    const float* direction,
    const float* rx_polarization,
    const c10::complex<float>* tangent_field_vector,
    const float* tangent_direction,
    c10::complex<float>* tangent_coefficient,
    float* tangent_path_gain) {
    for (int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < count;
         index += static_cast<int64_t>(blockDim.x) * gridDim.x) {
        const int64_t base = index * 3;
        const field::Complex3 value = {
            from_c10(field_vector[base]),
            from_c10(field_vector[base + 1]),
            from_c10(field_vector[base + 2]),
        };
        const field::float3a dir = load3f(direction, index);
        const field::float3a pol = load3f(rx_polarization, index);
        ad::DualF3 dir_dual = {
            dir,
            tangent_direction != nullptr ? load3f(tangent_direction, index)
                                         : field::f3_zero()};
        const ad::DualF3 axis = ad::dual_stable_perp_basis(
            dir_dual, ad::df3_const(pol));
        const field::Complex3 t_value = {
            tangent_field_vector != nullptr ? from_c10(tangent_field_vector[base])
                                            : field::cplx_zero(),
            tangent_field_vector != nullptr
                ? from_c10(tangent_field_vector[base + 1])
                : field::cplx_zero(),
            tangent_field_vector != nullptr
                ? from_c10(tangent_field_vector[base + 2])
                : field::cplx_zero(),
        };
        const field::Complex coefficient = transport::complex3_dot_real(value, axis.v);
        const field::Complex t_coefficient = field::cplx_add(
            transport::complex3_dot_real(t_value, axis.v),
            transport::complex3_dot_real(value, axis.d));
        tangent_coefficient[index] = to_c10(t_coefficient);
        tangent_path_gain[index] = 2.0f * (coefficient.re * t_coefficient.re +
                                           coefficient.im * t_coefficient.im);
    }
}

// ---------------------------------------------------------------------------
// Coupled stationary-geometry companions (fixed winner): the interaction
// points of a coupled path move with the endpoints. The primal re-solve is
// cn_coupled_rd_prepare_cuda (coupled_topology.cu); these duals mirror its
// row math with the wall plane and edge line frozen.
// ---------------------------------------------------------------------------

struct PrepareRowInputs {
    field::float3a source;
    field::float3a receiver;
    field::float3a plane_point;
    field::float3a plane_normal_unit;  // normalized like the primal kernel
    field::float3a edge_origin;
    field::float3a edge_dir_unit;
};

__device__ __forceinline__ PrepareRowInputs load_prepare_row(
    int64_t index,
    const float* source,
    const float* receiver,
    const float* plane_point,
    const float* plane_normal,
    const float* edge_pos,
    const float* edge_dir,
    const float* edge_t_min) {
    PrepareRowInputs in;
    in.source = load3f(source, index);
    in.receiver = load3f(receiver, index);
    in.plane_point = load3f(plane_point, index);
    // Primal normalize3: v / |v| when |v| > 1e-6, else zero.
    const field::float3a raw_n = load3f(plane_normal, index);
    const float n_len = sqrtf(field::f3_dot(raw_n, raw_n));
    in.plane_normal_unit = n_len > 1.0e-6f
                               ? field::f3_mul(raw_n, 1.0f / n_len)
                               : field::f3_zero();
    const field::float3a raw_d = load3f(edge_dir, index);
    const float d_len = sqrtf(field::f3_dot(raw_d, raw_d));
    in.edge_dir_unit = d_len > 1.0e-6f ? field::f3_mul(raw_d, 1.0f / d_len)
                                       : field::f3_zero();
    in.edge_origin = field::f3_add(
        load3f(edge_pos, index),
        field::f3_mul(in.edge_dir_unit, edge_t_min[index]));
    return in;
}

// Dual of one prepare row: edge stationary point and predicted reflection
// point as functions of (source, receiver) with the plane and edge frozen.
__device__ void prepare_row_dual(
    const PrepareRowInputs& in,
    field::float3a seed_source,
    field::float3a seed_receiver,
    DualV3& edge_point,
    DualV3& reflection_point) {
    const DualV3 src = field::dual_seed(in.source, seed_source);
    const DualV3 rcv = field::dual_seed(in.receiver, seed_receiver);
    const DualV3 normal = field::dual_const3(in.plane_normal_unit);
    const DualV3 p0 = field::dual_const3(in.plane_point);
    const DualV3 direction = field::dual_const3(in.edge_dir_unit);
    const DualV3 origin = field::dual_const3(in.edge_origin);
    const Dual signed_distance = field::f3_dot(field::f3_sub(src, p0), normal);
    const DualV3 image = field::f3_sub(
        src, field::f3_mul(normal, 2.0f * signed_distance));
    const Dual parameter = field::first_order_diffraction_parameter(
        image, rcv, origin, direction);
    edge_point = field::f3_add(origin, field::f3_mul(direction, parameter));
    const DualV3 image_to_edge = field::f3_sub(edge_point, image);
    const Dual plane_denominator = field::f3_dot(image_to_edge, normal);
    const Dual plane_parameter =
        field::f3_dot(field::f3_sub(p0, image), normal) / plane_denominator;
    reflection_point = field::f3_add(
        image, field::f3_mul(image_to_edge, plane_parameter));
}

__global__ void coupled_prepare_backward_kernel(
    int64_t count,
    const float* source,
    const float* receiver,
    const float* plane_point,
    const float* plane_normal,
    const float* edge_pos,
    const float* edge_dir,
    const float* edge_t_min,
    const float* grad_edge_point,
    const float* grad_reflection_point,
    float* grad_source,
    float* grad_receiver) {
    for (int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < count;
         index += static_cast<int64_t>(blockDim.x) * gridDim.x) {
        const PrepareRowInputs in = load_prepare_row(
            index, source, receiver, plane_point, plane_normal, edge_pos,
            edge_dir, edge_t_min);
        const int64_t base = index * 3;
        const field::float3a g_ep = grad_edge_point != nullptr
                                        ? load3f(grad_edge_point, index)
                                        : field::f3_zero();
        const field::float3a g_rp = grad_reflection_point != nullptr
                                        ? load3f(grad_reflection_point, index)
                                        : field::f3_zero();
        for (int slot = 0; slot < 2; ++slot) {
            float* out = slot == 0 ? grad_source : grad_receiver;
            if (out == nullptr)
                continue;
            for (int axis = 0; axis < 3; ++axis) {
                field::float3a seed_src = field::f3_zero();
                field::float3a seed_rcv = field::f3_zero();
                float* seed = slot == 0
                                  ? (axis == 0 ? &seed_src.x
                                               : axis == 1 ? &seed_src.y : &seed_src.z)
                                  : (axis == 0 ? &seed_rcv.x
                                               : axis == 1 ? &seed_rcv.y : &seed_rcv.z);
                *seed = 1.f;
                DualV3 edge_point;
                DualV3 reflection_point;
                prepare_row_dual(in, seed_src, seed_rcv, edge_point, reflection_point);
                out[base + axis] =
                    g_ep.x * edge_point.x.d + g_ep.y * edge_point.y.d +
                    g_ep.z * edge_point.z.d + g_rp.x * reflection_point.x.d +
                    g_rp.y * reflection_point.y.d + g_rp.z * reflection_point.z.d;
            }
        }
    }
}

__global__ void coupled_prepare_jvp_kernel(
    int64_t count,
    const float* source,
    const float* receiver,
    const float* plane_point,
    const float* plane_normal,
    const float* edge_pos,
    const float* edge_dir,
    const float* edge_t_min,
    const float* tangent_source,
    const float* tangent_receiver,
    float* tangent_edge_point,
    float* tangent_reflection_point) {
    for (int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < count;
         index += static_cast<int64_t>(blockDim.x) * gridDim.x) {
        const PrepareRowInputs in = load_prepare_row(
            index, source, receiver, plane_point, plane_normal, edge_pos,
            edge_dir, edge_t_min);
        const field::float3a seed_src = tangent_source != nullptr
                                            ? load3f(tangent_source, index)
                                            : field::f3_zero();
        const field::float3a seed_rcv = tangent_receiver != nullptr
                                            ? load3f(tangent_receiver, index)
                                            : field::f3_zero();
        DualV3 edge_point;
        DualV3 reflection_point;
        prepare_row_dual(in, seed_src, seed_rcv, edge_point, reflection_point);
        const int64_t base = index * 3;
        tangent_edge_point[base] = edge_point.x.d;
        tangent_edge_point[base + 1] = edge_point.y.d;
        tangent_edge_point[base + 2] = edge_point.z.d;
        tangent_reflection_point[base] = reflection_point.x.d;
        tangent_reflection_point[base + 1] = reflection_point.y.d;
        tangent_reflection_point[base + 2] = reflection_point.z.d;
    }
}

// ---------------------------------------------------------------------------
// Host-side plumbing.
// ---------------------------------------------------------------------------

int launch_blocks(int64_t count) {
    return static_cast<int>((count + kBlockSize - 1) / kBlockSize);
}

at::Tensor zero_scalar(const at::TensorOptions& options) {
    auto tensor = at::empty({1}, options);
    cudaStream_t stream =
        at::cuda::getCurrentCUDAStream(tensor.get_device()).stream();
    C10_CUDA_CHECK(cudaMemsetAsync(
        tensor.data_ptr(), 0, tensor.element_size(), stream));
    return tensor;
}

const at::Tensor* optional_tensor_arg(
    pybind11::object value,
    at::Tensor& storage,
    const char* name,
    c10::ScalarType dtype,
    at::IntArrayRef sizes,
    const at::Tensor& reference) {
    if (value.is_none())
        return nullptr;
    storage = value.cast<at::Tensor>().contiguous();
    TORCH_CHECK(storage.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(storage.scalar_type() == dtype, name, " has the wrong dtype");
    TORCH_CHECK(storage.sizes() == sizes, name, " has the wrong shape");
    TORCH_CHECK(
        storage.get_device() == reference.get_device(),
        name, " must share the primal device");
    return &storage;
}

template <typename T>
const T* opt_ptr(const at::Tensor* tensor) {
    return tensor == nullptr ? nullptr : tensor->data_ptr<T>();
}

template <typename T>
T* opt_mut_ptr(at::Tensor* tensor) {
    return tensor == nullptr ? nullptr : tensor->data_ptr<T>();
}

void check_wedge_primal(
    const at::Tensor& source,
    const at::Tensor& target,
    const at::Tensor& edge_position,
    const at::Tensor& edge_direction,
    const at::Tensor& edge_t_min,
    const at::Tensor& edge_t_max,
    const at::Tensor& edge_n0,
    const at::Tensor& edge_n1,
    const at::Tensor& exterior_angle,
    const at::Tensor& face0_valid,
    const at::Tensor& face0_eps_r,
    const at::Tensor& face0_sigma_e,
    const at::Tensor& face0_mu_r,
    const at::Tensor& face0_gain,
    const at::Tensor& face1_valid,
    const at::Tensor& face1_eps_r,
    const at::Tensor& face1_sigma_e,
    const at::Tensor& face1_mu_r,
    const at::Tensor& face1_gain,
    const at::Tensor& tx_power,
    double frequency_hz) {
    using channel_native::check_flat_tensor;
    using channel_native::check_tensor;
    using channel_native::check_vec3_table;
    const int64_t count = source.size(0);
    for (const auto& named : std::vector<std::pair<at::Tensor, const char*>>{
             {source, "source"},
             {target, "target"},
             {edge_position, "edge_position"},
             {edge_direction, "edge_direction"},
             {edge_n0, "edge_n0"},
             {edge_n1, "edge_n1"}}) {
        check_vec3_table(named.first, named.second);
        TORCH_CHECK(named.first.size(0) == count,
                    named.second, " must match source rows");
    }
    for (const auto& named : std::vector<std::pair<at::Tensor, const char*>>{
             {edge_t_min, "edge_t_min"},
             {edge_t_max, "edge_t_max"},
             {exterior_angle, "exterior_angle"},
             {face0_eps_r, "face0_eps_r"},
             {face0_sigma_e, "face0_sigma_e"},
             {face0_mu_r, "face0_mu_r"},
             {face0_gain, "face0_gain"},
             {face1_eps_r, "face1_eps_r"},
             {face1_sigma_e, "face1_sigma_e"},
             {face1_mu_r, "face1_mu_r"},
             {face1_gain, "face1_gain"},
             {tx_power, "tx_power"}}) {
        check_flat_tensor(named.first, named.second, at::kFloat);
        TORCH_CHECK(named.first.size(0) == count,
                    named.second, " must match source rows");
    }
    check_tensor(face0_valid, "face0_valid", at::kBool, 1);
    check_tensor(face1_valid, "face1_valid", at::kBool, 1);
    TORCH_CHECK(face0_valid.size(0) == count && face1_valid.size(0) == count,
                "face valid masks must match source rows");
    TORCH_CHECK(frequency_hz > 0.0, "frequency_hz must be positive");
}

// Optional winner-vertex group (mesh-vertex x diffraction). Supplied
// together or not at all; resolved into `out` so the tensor storage outlives
// the kernel launch.
struct WedgeVertexArgs {
    at::Tensor storage[5];
    const at::Tensor* v0 = nullptr;
    const at::Tensor* v1 = nullptr;
    const at::Tensor* opp0 = nullptr;
    const at::Tensor* opp1 = nullptr;
    const at::Tensor* boundary = nullptr;
};

void resolve_wedge_vertices(
    WedgeVertexArgs& out,
    pybind11::object vertex_v0,
    pybind11::object vertex_v1,
    pybind11::object vertex_opp0,
    pybind11::object vertex_opp1,
    pybind11::object edge_boundary,
    int64_t count,
    const at::Tensor& reference) {
    const bool any = !vertex_v0.is_none() || !vertex_v1.is_none() ||
                     !vertex_opp0.is_none() || !vertex_opp1.is_none() ||
                     !edge_boundary.is_none();
    if (!any)
        return;
    TORCH_CHECK(
        !vertex_v0.is_none() && !vertex_v1.is_none() &&
            !vertex_opp0.is_none() && !vertex_opp1.is_none() &&
            !edge_boundary.is_none(),
        "wedge vertex inputs must be supplied together");
    out.v0 = optional_tensor_arg(
        std::move(vertex_v0), out.storage[0], "vertex_v0", at::kFloat,
        {count, 3}, reference);
    out.v1 = optional_tensor_arg(
        std::move(vertex_v1), out.storage[1], "vertex_v1", at::kFloat,
        {count, 3}, reference);
    out.opp0 = optional_tensor_arg(
        std::move(vertex_opp0), out.storage[2], "vertex_opp0", at::kFloat,
        {count, 3}, reference);
    out.opp1 = optional_tensor_arg(
        std::move(vertex_opp1), out.storage[3], "vertex_opp1", at::kFloat,
        {count, 3}, reference);
    out.boundary = optional_tensor_arg(
        std::move(edge_boundary), out.storage[4], "edge_boundary", at::kBool,
        {count}, reference);
}

}  // namespace

#define WEDGE_HOST_ARGS                                                       \
    source.data_ptr<float>(), target.data_ptr<float>(),                       \
        edge_position.data_ptr<float>(), edge_direction.data_ptr<float>(),    \
        edge_t_min.data_ptr<float>(), edge_t_max.data_ptr<float>(),           \
        edge_n0.data_ptr<float>(), edge_n1.data_ptr<float>(),                 \
        exterior_angle.data_ptr<float>(), face0_valid.data_ptr<bool>(),       \
        face0_eps_r.data_ptr<float>(), face0_sigma_e.data_ptr<float>(),       \
        face0_mu_r.data_ptr<float>(), face0_gain.data_ptr<float>(),           \
        face1_valid.data_ptr<bool>(), face1_eps_r.data_ptr<float>(),          \
        face1_sigma_e.data_ptr<float>(), face1_mu_r.data_ptr<float>(),        \
        face1_gain.data_ptr<float>(), tx_power.data_ptr<float>(),             \
        static_cast<float>(frequency_hz), opt_ptr<float>(vertex_args.v0),     \
        opt_ptr<float>(vertex_args.v1), opt_ptr<float>(vertex_args.opp0),     \
        opt_ptr<float>(vertex_args.opp1), opt_ptr<bool>(vertex_args.boundary)

pybind11::dict cn_field_diffraction_wedge(
    at::Tensor source,
    at::Tensor target,
    at::Tensor edge_position,
    at::Tensor edge_direction,
    at::Tensor edge_t_min,
    at::Tensor edge_t_max,
    at::Tensor edge_n0,
    at::Tensor edge_n1,
    at::Tensor exterior_angle,
    at::Tensor face0_valid,
    at::Tensor face0_eps_r,
    at::Tensor face0_sigma_e,
    at::Tensor face0_mu_r,
    at::Tensor face0_gain,
    at::Tensor face1_valid,
    at::Tensor face1_eps_r,
    at::Tensor face1_sigma_e,
    at::Tensor face1_mu_r,
    at::Tensor face1_gain,
    at::Tensor tx_power,
    double frequency_hz,
    pybind11::object vertex_v0,
    pybind11::object vertex_v1,
    pybind11::object vertex_opp0,
    pybind11::object vertex_opp1,
    pybind11::object edge_boundary) {
    check_wedge_primal(
        source, target, edge_position, edge_direction, edge_t_min, edge_t_max,
        edge_n0, edge_n1, exterior_angle, face0_valid, face0_eps_r,
        face0_sigma_e, face0_mu_r, face0_gain, face1_valid, face1_eps_r,
        face1_sigma_e, face1_mu_r, face1_gain, tx_power, frequency_hz);
    const int64_t count = source.size(0);
    WedgeVertexArgs vertex_args;
    resolve_wedge_vertices(
        vertex_args, std::move(vertex_v0), std::move(vertex_v1),
        std::move(vertex_opp0), std::move(vertex_opp1),
        std::move(edge_boundary), count, source);
    auto field_vector = at::empty({count, 3}, source.options().dtype(at::kComplexFloat));
    auto direction = at::empty({count, 3}, source.options());
    if (count > 0) {
        cudaStream_t stream = at::cuda::getCurrentCUDAStream(source.get_device()).stream();
        diffraction_wedge_forward_kernel<<<launch_blocks(count), kBlockSize, 0, stream>>>(
            count,
            WEDGE_HOST_ARGS,
            field_vector.data_ptr<c10::complex<float>>(),
            direction.data_ptr<float>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    pybind11::dict out;
    out["field_vector"] = field_vector;
    out["direction"] = direction;
    return out;
}

pybind11::dict cn_field_diffraction_wedge_backward(
    at::Tensor source,
    at::Tensor target,
    at::Tensor edge_position,
    at::Tensor edge_direction,
    at::Tensor edge_t_min,
    at::Tensor edge_t_max,
    at::Tensor edge_n0,
    at::Tensor edge_n1,
    at::Tensor exterior_angle,
    at::Tensor face0_valid,
    at::Tensor face0_eps_r,
    at::Tensor face0_sigma_e,
    at::Tensor face0_mu_r,
    at::Tensor face0_gain,
    at::Tensor face1_valid,
    at::Tensor face1_eps_r,
    at::Tensor face1_sigma_e,
    at::Tensor face1_mu_r,
    at::Tensor face1_gain,
    at::Tensor tx_power,
    double frequency_hz,
    pybind11::object vertex_v0,
    pybind11::object vertex_v1,
    pybind11::object vertex_opp0,
    pybind11::object vertex_opp1,
    pybind11::object edge_boundary,
    pybind11::object grad_field_vector,
    pybind11::object grad_direction,
    bool need_grad_material,
    bool need_grad_frequency,
    bool need_grad_geometry,
    bool need_grad_vertices) {
    check_wedge_primal(
        source, target, edge_position, edge_direction, edge_t_min, edge_t_max,
        edge_n0, edge_n1, exterior_angle, face0_valid, face0_eps_r,
        face0_sigma_e, face0_mu_r, face0_gain, face1_valid, face1_eps_r,
        face1_sigma_e, face1_mu_r, face1_gain, tx_power, frequency_hz);
    const int64_t count = source.size(0);
    WedgeVertexArgs vertex_args;
    resolve_wedge_vertices(
        vertex_args, std::move(vertex_v0), std::move(vertex_v1),
        std::move(vertex_opp0), std::move(vertex_opp1),
        std::move(edge_boundary), count, source);
    TORCH_CHECK(
        !need_grad_vertices || vertex_args.v0 != nullptr,
        "vertex gradients require the wedge vertex inputs");
    at::Tensor grad_field_storage;
    at::Tensor grad_direction_storage;
    const at::Tensor* g_field = optional_tensor_arg(
        std::move(grad_field_vector), grad_field_storage, "grad_field_vector",
        at::kComplexFloat, {count, 3}, source);
    const at::Tensor* g_direction = optional_tensor_arg(
        std::move(grad_direction), grad_direction_storage, "grad_direction",
        at::kFloat, {count, 3}, source);

    const auto options = source.options();
    at::Tensor grad_source, grad_target;
    at::Tensor grad_face0_eps, grad_face0_sigma, grad_face0_gain;
    at::Tensor grad_face1_eps, grad_face1_sigma, grad_face1_gain;
    at::Tensor grad_frequency;
    at::Tensor grad_vertices[4];
    at::Tensor* grad_source_ptr = nullptr;
    at::Tensor* grad_target_ptr = nullptr;
    at::Tensor* material_ptrs[6] = {nullptr, nullptr, nullptr, nullptr, nullptr, nullptr};
    at::Tensor* grad_frequency_ptr = nullptr;
    at::Tensor* grad_vertex_ptrs[4] = {nullptr, nullptr, nullptr, nullptr};
    if (need_grad_geometry) {
        grad_source = at::empty({count, 3}, options);
        grad_target = at::empty({count, 3}, options);
        grad_source_ptr = &grad_source;
        grad_target_ptr = &grad_target;
    }
    if (need_grad_material) {
        grad_face0_eps = at::empty({count}, options);
        grad_face0_sigma = at::empty({count}, options);
        grad_face0_gain = at::empty({count}, options);
        grad_face1_eps = at::empty({count}, options);
        grad_face1_sigma = at::empty({count}, options);
        grad_face1_gain = at::empty({count}, options);
        material_ptrs[0] = &grad_face0_eps;
        material_ptrs[1] = &grad_face0_sigma;
        material_ptrs[2] = &grad_face0_gain;
        material_ptrs[3] = &grad_face1_eps;
        material_ptrs[4] = &grad_face1_sigma;
        material_ptrs[5] = &grad_face1_gain;
    }
    if (need_grad_frequency) {
        grad_frequency = zero_scalar(options);
        grad_frequency_ptr = &grad_frequency;
    }
    if (need_grad_vertices) {
        for (int slot = 0; slot < 4; ++slot) {
            grad_vertices[slot] = at::empty({count, 3}, options);
            grad_vertex_ptrs[slot] = &grad_vertices[slot];
        }
    }
    if (count > 0 && (g_field != nullptr || g_direction != nullptr)) {
        cudaStream_t stream = at::cuda::getCurrentCUDAStream(source.get_device()).stream();
        diffraction_wedge_backward_kernel<<<launch_blocks(count), kBlockSize, 0, stream>>>(
            count,
            WEDGE_HOST_ARGS,
            opt_ptr<c10::complex<float>>(g_field),
            opt_ptr<float>(g_direction),
            opt_mut_ptr<float>(grad_source_ptr),
            opt_mut_ptr<float>(grad_target_ptr),
            opt_mut_ptr<float>(material_ptrs[0]),
            opt_mut_ptr<float>(material_ptrs[1]),
            opt_mut_ptr<float>(material_ptrs[2]),
            opt_mut_ptr<float>(material_ptrs[3]),
            opt_mut_ptr<float>(material_ptrs[4]),
            opt_mut_ptr<float>(material_ptrs[5]),
            opt_mut_ptr<float>(grad_frequency_ptr),
            opt_mut_ptr<float>(grad_vertex_ptrs[0]),
            opt_mut_ptr<float>(grad_vertex_ptrs[1]),
            opt_mut_ptr<float>(grad_vertex_ptrs[2]),
            opt_mut_ptr<float>(grad_vertex_ptrs[3]));
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    } else if (count == 0 || (g_field == nullptr && g_direction == nullptr)) {
        // No cotangents: every requested gradient is exactly zero.
        for (at::Tensor* tensor : {grad_source_ptr, grad_target_ptr}) {
            if (tensor != nullptr)
                tensor->zero_();
        }
        for (at::Tensor* tensor : material_ptrs) {
            if (tensor != nullptr)
                tensor->zero_();
        }
        for (at::Tensor* tensor : grad_vertex_ptrs) {
            if (tensor != nullptr)
                tensor->zero_();
        }
    }
    pybind11::dict out;
    out["grad_source"] = grad_source_ptr != nullptr
                             ? pybind11::cast(grad_source)
                             : pybind11::object(pybind11::none());
    out["grad_target"] = grad_target_ptr != nullptr
                             ? pybind11::cast(grad_target)
                             : pybind11::object(pybind11::none());
    out["grad_face0_eps_r"] = material_ptrs[0] != nullptr
                                  ? pybind11::cast(grad_face0_eps)
                                  : pybind11::object(pybind11::none());
    out["grad_face0_sigma_e"] = material_ptrs[1] != nullptr
                                    ? pybind11::cast(grad_face0_sigma)
                                    : pybind11::object(pybind11::none());
    out["grad_face0_gain"] = material_ptrs[2] != nullptr
                                 ? pybind11::cast(grad_face0_gain)
                                 : pybind11::object(pybind11::none());
    out["grad_face1_eps_r"] = material_ptrs[3] != nullptr
                                  ? pybind11::cast(grad_face1_eps)
                                  : pybind11::object(pybind11::none());
    out["grad_face1_sigma_e"] = material_ptrs[4] != nullptr
                                    ? pybind11::cast(grad_face1_sigma)
                                    : pybind11::object(pybind11::none());
    out["grad_face1_gain"] = material_ptrs[5] != nullptr
                                 ? pybind11::cast(grad_face1_gain)
                                 : pybind11::object(pybind11::none());
    out["grad_frequency"] = grad_frequency_ptr != nullptr
                                ? pybind11::cast(grad_frequency)
                                : pybind11::object(pybind11::none());
    const char* vertex_grad_names[4] = {
        "grad_vertex_v0", "grad_vertex_v1", "grad_vertex_opp0",
        "grad_vertex_opp1"};
    for (int slot = 0; slot < 4; ++slot) {
        out[vertex_grad_names[slot]] =
            grad_vertex_ptrs[slot] != nullptr
                ? pybind11::cast(grad_vertices[slot])
                : pybind11::object(pybind11::none());
    }
    return out;
}

pybind11::dict cn_field_diffraction_wedge_jvp(
    at::Tensor source,
    at::Tensor target,
    at::Tensor edge_position,
    at::Tensor edge_direction,
    at::Tensor edge_t_min,
    at::Tensor edge_t_max,
    at::Tensor edge_n0,
    at::Tensor edge_n1,
    at::Tensor exterior_angle,
    at::Tensor face0_valid,
    at::Tensor face0_eps_r,
    at::Tensor face0_sigma_e,
    at::Tensor face0_mu_r,
    at::Tensor face0_gain,
    at::Tensor face1_valid,
    at::Tensor face1_eps_r,
    at::Tensor face1_sigma_e,
    at::Tensor face1_mu_r,
    at::Tensor face1_gain,
    at::Tensor tx_power,
    double frequency_hz,
    pybind11::object vertex_v0,
    pybind11::object vertex_v1,
    pybind11::object vertex_opp0,
    pybind11::object vertex_opp1,
    pybind11::object edge_boundary,
    pybind11::object tangent_source,
    pybind11::object tangent_target,
    pybind11::object tangent_face0_eps_r,
    pybind11::object tangent_face0_sigma_e,
    pybind11::object tangent_face0_gain,
    pybind11::object tangent_face1_eps_r,
    pybind11::object tangent_face1_sigma_e,
    pybind11::object tangent_face1_gain,
    double tangent_frequency,
    pybind11::object tangent_vertex_v0,
    pybind11::object tangent_vertex_v1,
    pybind11::object tangent_vertex_opp0,
    pybind11::object tangent_vertex_opp1) {
    check_wedge_primal(
        source, target, edge_position, edge_direction, edge_t_min, edge_t_max,
        edge_n0, edge_n1, exterior_angle, face0_valid, face0_eps_r,
        face0_sigma_e, face0_mu_r, face0_gain, face1_valid, face1_eps_r,
        face1_sigma_e, face1_mu_r, face1_gain, tx_power, frequency_hz);
    const int64_t count = source.size(0);
    WedgeVertexArgs vertex_args;
    resolve_wedge_vertices(
        vertex_args, std::move(vertex_v0), std::move(vertex_v1),
        std::move(vertex_opp0), std::move(vertex_opp1),
        std::move(edge_boundary), count, source);
    at::Tensor storage[12];
    const at::Tensor* t_source = optional_tensor_arg(
        std::move(tangent_source), storage[0], "tangent_source", at::kFloat,
        {count, 3}, source);
    const at::Tensor* t_target = optional_tensor_arg(
        std::move(tangent_target), storage[1], "tangent_target", at::kFloat,
        {count, 3}, source);
    const at::Tensor* t_f0_eps = optional_tensor_arg(
        std::move(tangent_face0_eps_r), storage[2], "tangent_face0_eps_r",
        at::kFloat, {count}, source);
    const at::Tensor* t_f0_sigma = optional_tensor_arg(
        std::move(tangent_face0_sigma_e), storage[3], "tangent_face0_sigma_e",
        at::kFloat, {count}, source);
    const at::Tensor* t_f0_gain = optional_tensor_arg(
        std::move(tangent_face0_gain), storage[4], "tangent_face0_gain",
        at::kFloat, {count}, source);
    const at::Tensor* t_f1_eps = optional_tensor_arg(
        std::move(tangent_face1_eps_r), storage[5], "tangent_face1_eps_r",
        at::kFloat, {count}, source);
    const at::Tensor* t_f1_sigma = optional_tensor_arg(
        std::move(tangent_face1_sigma_e), storage[6], "tangent_face1_sigma_e",
        at::kFloat, {count}, source);
    const at::Tensor* t_f1_gain = optional_tensor_arg(
        std::move(tangent_face1_gain), storage[7], "tangent_face1_gain",
        at::kFloat, {count}, source);
    const at::Tensor* t_v0 = optional_tensor_arg(
        std::move(tangent_vertex_v0), storage[8], "tangent_vertex_v0",
        at::kFloat, {count, 3}, source);
    const at::Tensor* t_v1 = optional_tensor_arg(
        std::move(tangent_vertex_v1), storage[9], "tangent_vertex_v1",
        at::kFloat, {count, 3}, source);
    const at::Tensor* t_opp0 = optional_tensor_arg(
        std::move(tangent_vertex_opp0), storage[10], "tangent_vertex_opp0",
        at::kFloat, {count, 3}, source);
    const at::Tensor* t_opp1 = optional_tensor_arg(
        std::move(tangent_vertex_opp1), storage[11], "tangent_vertex_opp1",
        at::kFloat, {count, 3}, source);
    TORCH_CHECK(
        (t_v0 == nullptr && t_v1 == nullptr && t_opp0 == nullptr &&
         t_opp1 == nullptr) ||
            vertex_args.v0 != nullptr,
        "vertex tangents require the wedge vertex inputs");
    auto tangent_field_vector = at::empty({count, 3}, source.options().dtype(at::kComplexFloat));
    auto tangent_direction = at::empty({count, 3}, source.options());
    if (count > 0) {
        cudaStream_t stream = at::cuda::getCurrentCUDAStream(source.get_device()).stream();
        diffraction_wedge_jvp_kernel<<<launch_blocks(count), kBlockSize, 0, stream>>>(
            count,
            WEDGE_HOST_ARGS,
            opt_ptr<float>(t_source),
            opt_ptr<float>(t_target),
            opt_ptr<float>(t_f0_eps),
            opt_ptr<float>(t_f0_sigma),
            opt_ptr<float>(t_f0_gain),
            opt_ptr<float>(t_f1_eps),
            opt_ptr<float>(t_f1_sigma),
            opt_ptr<float>(t_f1_gain),
            static_cast<float>(tangent_frequency),
            opt_ptr<float>(t_v0),
            opt_ptr<float>(t_v1),
            opt_ptr<float>(t_opp0),
            opt_ptr<float>(t_opp1),
            tangent_field_vector.data_ptr<c10::complex<float>>(),
            tangent_direction.data_ptr<float>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    pybind11::dict out;
    out["tangent_field_vector"] = tangent_field_vector;
    out["tangent_direction"] = tangent_direction;
    return out;
}

namespace {

void check_coupled_primal_rows(
    const std::vector<std::pair<at::Tensor, const char*>>& vectors,
    const std::vector<std::pair<at::Tensor, const char*>>& scalars,
    double frequency_hz) {
    using channel_native::check_flat_tensor;
    using channel_native::check_vec3_table;
    const int64_t count = vectors.front().first.size(0);
    for (const auto& named : vectors) {
        check_vec3_table(named.first, named.second);
        TORCH_CHECK(named.first.size(0) == count,
                    named.second, " must match source rows");
    }
    for (const auto& named : scalars) {
        check_flat_tensor(named.first, named.second, at::kFloat);
        TORCH_CHECK(named.first.size(0) == count,
                    named.second, " must match source rows");
    }
    TORCH_CHECK(frequency_hz > 0.0, "frequency_hz must be positive");
}

}  // namespace

#define COUPLED_HOST_ARGS                                                     \
    source.data_ptr<float>(), target.data_ptr<float>(),                       \
        reflection_position.data_ptr<float>(),                                \
        reflection_normal.data_ptr<float>(), edge_position.data_ptr<float>(), \
        edge_direction.data_ptr<float>(), edge_n0.data_ptr<float>(),          \
        edge_n1.data_ptr<float>(), exterior_angle.data_ptr<float>(),          \
        tx_power.data_ptr<float>(), tx_polarization.data_ptr<float>(),        \
        rx_polarization.data_ptr<float>(),                                    \
        reflection_eps_r.data_ptr<float>(),                                   \
        reflection_sigma_e.data_ptr<float>(),                                 \
        reflection_mu_r.data_ptr<float>(),                                    \
        reflection_gain.data_ptr<float>(),                                    \
        reflection_thickness.data_ptr<float>(),                               \
        wedge_eps_r0.data_ptr<float>(), wedge_sigma_e0.data_ptr<float>(),     \
        wedge_mu_r0.data_ptr<float>(), wedge_gain0.data_ptr<float>(),         \
        wedge_thickness0.data_ptr<float>(), wedge_eps_r1.data_ptr<float>(),   \
        wedge_sigma_e1.data_ptr<float>(), wedge_mu_r1.data_ptr<float>(),      \
        wedge_gain1.data_ptr<float>(), wedge_thickness1.data_ptr<float>(),    \
        static_cast<float>(frequency_hz), reverse

#define COUPLED_HOST_PARAMS                                                   \
    at::Tensor source, at::Tensor target, at::Tensor reflection_position,     \
        at::Tensor reflection_normal, at::Tensor edge_position,               \
        at::Tensor edge_direction, at::Tensor edge_n0, at::Tensor edge_n1,    \
        at::Tensor exterior_angle, at::Tensor tx_power,                       \
        at::Tensor tx_polarization, at::Tensor rx_polarization,               \
        at::Tensor reflection_eps_r, at::Tensor reflection_sigma_e,           \
        at::Tensor reflection_mu_r, at::Tensor reflection_gain,               \
        at::Tensor reflection_thickness, at::Tensor wedge_eps_r0,             \
        at::Tensor wedge_sigma_e0, at::Tensor wedge_mu_r0,                    \
        at::Tensor wedge_gain0, at::Tensor wedge_thickness0,                  \
        at::Tensor wedge_eps_r1, at::Tensor wedge_sigma_e1,                   \
        at::Tensor wedge_mu_r1, at::Tensor wedge_gain1,                       \
        at::Tensor wedge_thickness1, double frequency_hz, bool reverse

#define COUPLED_CHECK_LISTS                                                   \
    check_coupled_primal_rows(                                                \
        {{source, "source"},                                                  \
         {target, "target"},                                                  \
         {reflection_position, "reflection_position"},                        \
         {reflection_normal, "reflection_normal"},                            \
         {edge_position, "edge_position"},                                    \
         {edge_direction, "edge_direction"},                                  \
         {edge_n0, "edge_n0"},                                                \
         {edge_n1, "edge_n1"},                                                \
         {tx_polarization, "tx_polarization"},                                \
         {rx_polarization, "rx_polarization"}},                               \
        {{exterior_angle, "exterior_angle"},                                  \
         {tx_power, "tx_power"},                                              \
         {reflection_eps_r, "reflection_eps_r"},                              \
         {reflection_sigma_e, "reflection_sigma_e"},                          \
         {reflection_mu_r, "reflection_mu_r"},                                \
         {reflection_gain, "reflection_gain"},                                \
         {reflection_thickness, "reflection_thickness"},                      \
         {wedge_eps_r0, "wedge_eps_r0"},                                      \
         {wedge_sigma_e0, "wedge_sigma_e0"},                                  \
         {wedge_mu_r0, "wedge_mu_r0"},                                        \
         {wedge_gain0, "wedge_gain0"},                                        \
         {wedge_thickness0, "wedge_thickness0"},                              \
         {wedge_eps_r1, "wedge_eps_r1"},                                      \
         {wedge_sigma_e1, "wedge_sigma_e1"},                                  \
         {wedge_mu_r1, "wedge_mu_r1"},                                        \
         {wedge_gain1, "wedge_gain1"},                                        \
         {wedge_thickness1, "wedge_thickness1"}},                             \
        frequency_hz)

pybind11::dict cn_field_coupled_rd_backward(
    COUPLED_HOST_PARAMS,
    pybind11::object grad_field_vector,
    pybind11::object grad_coefficient,
    pybind11::object grad_path_field,
    pybind11::object grad_path_gain,
    bool need_grad_eps_r,
    bool need_grad_sigma_e,
    bool need_grad_gain,
    bool need_grad_thickness,
    bool need_grad_frequency,
    bool need_grad_geometry) {
    COUPLED_CHECK_LISTS;
    const int64_t count = source.size(0);
    at::Tensor grad_storage[4];
    const at::Tensor* g_field = optional_tensor_arg(
        std::move(grad_field_vector), grad_storage[0], "grad_field_vector",
        at::kComplexFloat, {count, 3}, source);
    const at::Tensor* g_coefficient = optional_tensor_arg(
        std::move(grad_coefficient), grad_storage[1], "grad_coefficient",
        at::kComplexFloat, {count}, source);
    const at::Tensor* g_path_field = optional_tensor_arg(
        std::move(grad_path_field), grad_storage[2], "grad_path_field",
        at::kComplexFloat, {count}, source);
    const at::Tensor* g_path_gain = optional_tensor_arg(
        std::move(grad_path_gain), grad_storage[3], "grad_path_gain",
        at::kFloat, {count}, source);

    const auto options = source.options();
    at::Tensor grad_source, grad_target, grad_hit, grad_edge;
    at::Tensor grad_eps, grad_sigma, grad_gain, grad_thickness, grad_frequency;
    at::Tensor* grad_source_ptr = nullptr;
    at::Tensor* grad_target_ptr = nullptr;
    at::Tensor* grad_hit_ptr = nullptr;
    at::Tensor* grad_edge_ptr = nullptr;
    at::Tensor* grad_eps_ptr = nullptr;
    at::Tensor* grad_sigma_ptr = nullptr;
    at::Tensor* grad_gain_ptr = nullptr;
    at::Tensor* grad_thickness_ptr = nullptr;
    at::Tensor* grad_frequency_ptr = nullptr;
    if (need_grad_geometry) {
        grad_source = at::empty({count, 3}, options);
        grad_target = at::empty({count, 3}, options);
        grad_hit = at::empty({count, 3}, options);
        grad_edge = at::empty({count, 3}, options);
        grad_source_ptr = &grad_source;
        grad_target_ptr = &grad_target;
        grad_hit_ptr = &grad_hit;
        grad_edge_ptr = &grad_edge;
    }
    if (need_grad_eps_r) {
        grad_eps = at::empty({count, 3}, options);
        grad_eps_ptr = &grad_eps;
    }
    if (need_grad_sigma_e) {
        grad_sigma = at::empty({count, 3}, options);
        grad_sigma_ptr = &grad_sigma;
    }
    if (need_grad_gain) {
        grad_gain = at::empty({count, 3}, options);
        grad_gain_ptr = &grad_gain;
    }
    if (need_grad_thickness) {
        grad_thickness = at::empty({count, 3}, options);
        grad_thickness_ptr = &grad_thickness;
    }
    if (need_grad_frequency) {
        grad_frequency = zero_scalar(options);
        grad_frequency_ptr = &grad_frequency;
    }
    const bool any_grad = g_field != nullptr || g_coefficient != nullptr ||
                          g_path_field != nullptr || g_path_gain != nullptr;
    if (count > 0 && any_grad) {
        cudaStream_t stream = at::cuda::getCurrentCUDAStream(source.get_device()).stream();
        coupled_rd_backward_kernel<<<launch_blocks(count), kBlockSize, 0, stream>>>(
            count,
            COUPLED_HOST_ARGS,
            opt_ptr<c10::complex<float>>(g_field),
            opt_ptr<c10::complex<float>>(g_coefficient),
            opt_ptr<c10::complex<float>>(g_path_field),
            opt_ptr<float>(g_path_gain),
            opt_mut_ptr<float>(grad_source_ptr),
            opt_mut_ptr<float>(grad_target_ptr),
            opt_mut_ptr<float>(grad_hit_ptr),
            opt_mut_ptr<float>(grad_edge_ptr),
            opt_mut_ptr<float>(grad_eps_ptr),
            opt_mut_ptr<float>(grad_sigma_ptr),
            opt_mut_ptr<float>(grad_gain_ptr),
            opt_mut_ptr<float>(grad_thickness_ptr),
            opt_mut_ptr<float>(grad_frequency_ptr));
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    } else {
        for (at::Tensor* tensor :
             {grad_source_ptr, grad_target_ptr, grad_hit_ptr, grad_edge_ptr,
              grad_eps_ptr, grad_sigma_ptr, grad_gain_ptr, grad_thickness_ptr}) {
            if (tensor != nullptr)
                tensor->zero_();
        }
    }
    auto pack = [](at::Tensor* tensor, const at::Tensor& value) {
        return tensor != nullptr ? pybind11::cast(value)
                                 : pybind11::object(pybind11::none());
    };
    pybind11::dict out;
    out["grad_source"] = pack(grad_source_ptr, grad_source);
    out["grad_target"] = pack(grad_target_ptr, grad_target);
    out["grad_reflection_position"] = pack(grad_hit_ptr, grad_hit);
    out["grad_edge_position"] = pack(grad_edge_ptr, grad_edge);
    out["grad_eps_r"] = pack(grad_eps_ptr, grad_eps);
    out["grad_sigma_e"] = pack(grad_sigma_ptr, grad_sigma);
    out["grad_gain"] = pack(grad_gain_ptr, grad_gain);
    out["grad_thickness"] = pack(grad_thickness_ptr, grad_thickness);
    out["grad_frequency"] = pack(grad_frequency_ptr, grad_frequency);
    return out;
}

pybind11::dict cn_field_coupled_rd_jvp(
    COUPLED_HOST_PARAMS,
    pybind11::object tangent_source,
    pybind11::object tangent_target,
    pybind11::object tangent_reflection_position,
    pybind11::object tangent_edge_position,
    pybind11::object tangent_eps_r,
    pybind11::object tangent_sigma_e,
    pybind11::object tangent_gain,
    pybind11::object tangent_thickness,
    double tangent_frequency) {
    COUPLED_CHECK_LISTS;
    const int64_t count = source.size(0);
    at::Tensor storage[8];
    const at::Tensor* t_source = optional_tensor_arg(
        std::move(tangent_source), storage[0], "tangent_source", at::kFloat,
        {count, 3}, source);
    const at::Tensor* t_target = optional_tensor_arg(
        std::move(tangent_target), storage[1], "tangent_target", at::kFloat,
        {count, 3}, source);
    const at::Tensor* t_hit = optional_tensor_arg(
        std::move(tangent_reflection_position), storage[2],
        "tangent_reflection_position", at::kFloat, {count, 3}, source);
    const at::Tensor* t_edge = optional_tensor_arg(
        std::move(tangent_edge_position), storage[3], "tangent_edge_position",
        at::kFloat, {count, 3}, source);
    const at::Tensor* t_eps = optional_tensor_arg(
        std::move(tangent_eps_r), storage[4], "tangent_eps_r", at::kFloat,
        {count, 3}, source);
    const at::Tensor* t_sigma = optional_tensor_arg(
        std::move(tangent_sigma_e), storage[5], "tangent_sigma_e", at::kFloat,
        {count, 3}, source);
    const at::Tensor* t_gain = optional_tensor_arg(
        std::move(tangent_gain), storage[6], "tangent_gain", at::kFloat,
        {count, 3}, source);
    const at::Tensor* t_thickness = optional_tensor_arg(
        std::move(tangent_thickness), storage[7], "tangent_thickness",
        at::kFloat, {count, 3}, source);
    auto tangent_field_vector = at::empty({count, 3}, source.options().dtype(at::kComplexFloat));
    auto tangent_coefficient = at::empty({count}, source.options().dtype(at::kComplexFloat));
    auto tangent_path_field = at::empty({count}, source.options().dtype(at::kComplexFloat));
    auto tangent_path_gain = at::empty({count}, source.options());
    if (count > 0) {
        cudaStream_t stream = at::cuda::getCurrentCUDAStream(source.get_device()).stream();
        coupled_rd_jvp_kernel<<<launch_blocks(count), kBlockSize, 0, stream>>>(
            count,
            COUPLED_HOST_ARGS,
            opt_ptr<float>(t_source),
            opt_ptr<float>(t_target),
            opt_ptr<float>(t_hit),
            opt_ptr<float>(t_edge),
            opt_ptr<float>(t_eps),
            opt_ptr<float>(t_sigma),
            opt_ptr<float>(t_gain),
            opt_ptr<float>(t_thickness),
            static_cast<float>(tangent_frequency),
            tangent_field_vector.data_ptr<c10::complex<float>>(),
            tangent_coefficient.data_ptr<c10::complex<float>>(),
            tangent_path_field.data_ptr<c10::complex<float>>(),
            tangent_path_gain.data_ptr<float>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    pybind11::dict out;
    out["tangent_field_vector"] = tangent_field_vector;
    out["tangent_coefficient"] = tangent_coefficient;
    out["tangent_path_field"] = tangent_path_field;
    out["tangent_path_gain"] = tangent_path_gain;
    return out;
}

pybind11::dict cn_field_project_complex3_backward(
    at::Tensor field_vector,
    at::Tensor direction,
    at::Tensor rx_polarization,
    pybind11::object grad_coefficient,
    pybind11::object grad_path_gain,
    bool need_grad_field_vector,
    bool need_grad_direction) {
    using channel_native::check_tensor;
    using channel_native::check_vec3_table;
    check_tensor(field_vector, "field_vector", at::kComplexFloat, 2);
    TORCH_CHECK(field_vector.size(1) == 3, "field_vector must have shape (N, 3)");
    check_vec3_table(direction, "direction");
    check_vec3_table(rx_polarization, "rx_polarization");
    const int64_t count = field_vector.size(0);
    TORCH_CHECK(direction.size(0) == count && rx_polarization.size(0) == count,
                "projection tensors must match field_vector rows");
    at::Tensor grad_storage[2];
    const at::Tensor* g_coefficient = optional_tensor_arg(
        std::move(grad_coefficient), grad_storage[0], "grad_coefficient",
        at::kComplexFloat, {count}, field_vector);
    const at::Tensor* g_path_gain = optional_tensor_arg(
        std::move(grad_path_gain), grad_storage[1], "grad_path_gain",
        at::kFloat, {count}, field_vector);
    at::Tensor grad_field_vector;
    at::Tensor grad_direction;
    at::Tensor* grad_field_ptr = nullptr;
    at::Tensor* grad_direction_ptr = nullptr;
    if (need_grad_field_vector) {
        grad_field_vector = at::empty(
            {count, 3}, field_vector.options());
        grad_field_ptr = &grad_field_vector;
    }
    if (need_grad_direction) {
        grad_direction = at::empty({count, 3}, direction.options());
        grad_direction_ptr = &grad_direction;
    }
    if (count > 0 && (g_coefficient != nullptr || g_path_gain != nullptr)) {
        cudaStream_t stream = at::cuda::getCurrentCUDAStream(direction.get_device()).stream();
        project_complex3_backward_kernel<<<launch_blocks(count), kBlockSize, 0, stream>>>(
            count,
            field_vector.data_ptr<c10::complex<float>>(),
            direction.data_ptr<float>(),
            rx_polarization.data_ptr<float>(),
            opt_ptr<c10::complex<float>>(g_coefficient),
            opt_ptr<float>(g_path_gain),
            opt_mut_ptr<c10::complex<float>>(grad_field_ptr),
            opt_mut_ptr<float>(grad_direction_ptr));
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    } else {
        if (grad_field_ptr != nullptr)
            grad_field_ptr->zero_();
        if (grad_direction_ptr != nullptr)
            grad_direction_ptr->zero_();
    }
    pybind11::dict out;
    out["grad_field_vector"] = grad_field_ptr != nullptr
                                   ? pybind11::cast(grad_field_vector)
                                   : pybind11::object(pybind11::none());
    out["grad_direction"] = grad_direction_ptr != nullptr
                                ? pybind11::cast(grad_direction)
                                : pybind11::object(pybind11::none());
    return out;
}

pybind11::dict cn_field_project_complex3_jvp(
    at::Tensor field_vector,
    at::Tensor direction,
    at::Tensor rx_polarization,
    pybind11::object tangent_field_vector,
    pybind11::object tangent_direction) {
    using channel_native::check_tensor;
    using channel_native::check_vec3_table;
    check_tensor(field_vector, "field_vector", at::kComplexFloat, 2);
    TORCH_CHECK(field_vector.size(1) == 3, "field_vector must have shape (N, 3)");
    check_vec3_table(direction, "direction");
    check_vec3_table(rx_polarization, "rx_polarization");
    const int64_t count = field_vector.size(0);
    TORCH_CHECK(direction.size(0) == count && rx_polarization.size(0) == count,
                "projection tensors must match field_vector rows");
    at::Tensor storage[2];
    const at::Tensor* t_field = optional_tensor_arg(
        std::move(tangent_field_vector), storage[0], "tangent_field_vector",
        at::kComplexFloat, {count, 3}, field_vector);
    const at::Tensor* t_direction = optional_tensor_arg(
        std::move(tangent_direction), storage[1], "tangent_direction",
        at::kFloat, {count, 3}, field_vector);
    auto tangent_coefficient = at::empty({count}, field_vector.options());
    auto tangent_path_gain = at::empty({count}, direction.options());
    if (count > 0) {
        cudaStream_t stream = at::cuda::getCurrentCUDAStream(direction.get_device()).stream();
        project_complex3_jvp_kernel<<<launch_blocks(count), kBlockSize, 0, stream>>>(
            count,
            field_vector.data_ptr<c10::complex<float>>(),
            direction.data_ptr<float>(),
            rx_polarization.data_ptr<float>(),
            opt_ptr<c10::complex<float>>(t_field),
            opt_ptr<float>(t_direction),
            tangent_coefficient.data_ptr<c10::complex<float>>(),
            tangent_path_gain.data_ptr<float>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    pybind11::dict out;
    out["tangent_coefficient"] = tangent_coefficient;
    out["tangent_path_gain"] = tangent_path_gain;
    return out;
}

namespace {

void check_prepare_rows(
    const at::Tensor& source,
    const at::Tensor& receiver,
    const at::Tensor& plane_point,
    const at::Tensor& plane_normal,
    const at::Tensor& edge_pos,
    const at::Tensor& edge_dir,
    const at::Tensor& edge_t_min) {
    using channel_native::check_flat_tensor;
    using channel_native::check_vec3_table;
    const int64_t count = source.size(0);
    for (const auto& named : std::vector<std::pair<at::Tensor, const char*>>{
             {source, "source"},
             {receiver, "receiver"},
             {plane_point, "plane_point"},
             {plane_normal, "plane_normal"},
             {edge_pos, "edge_pos"},
             {edge_dir, "edge_dir"}}) {
        check_vec3_table(named.first, named.second);
        TORCH_CHECK(named.first.size(0) == count,
                    named.second, " must match source rows");
    }
    check_flat_tensor(edge_t_min, "edge_t_min", at::kFloat);
    TORCH_CHECK(edge_t_min.size(0) == count, "edge_t_min must match source rows");
}

}  // namespace

pybind11::dict cn_coupled_rd_prepare_backward(
    at::Tensor source,
    at::Tensor receiver,
    at::Tensor plane_point,
    at::Tensor plane_normal,
    at::Tensor edge_pos,
    at::Tensor edge_dir,
    at::Tensor edge_t_min,
    pybind11::object grad_edge_point,
    pybind11::object grad_reflection_point,
    bool need_grad_source,
    bool need_grad_receiver) {
    check_prepare_rows(
        source, receiver, plane_point, plane_normal, edge_pos, edge_dir,
        edge_t_min);
    const int64_t count = source.size(0);
    at::Tensor grad_storage[2];
    const at::Tensor* g_edge_point = optional_tensor_arg(
        std::move(grad_edge_point), grad_storage[0], "grad_edge_point",
        at::kFloat, {count, 3}, source);
    const at::Tensor* g_reflection_point = optional_tensor_arg(
        std::move(grad_reflection_point), grad_storage[1],
        "grad_reflection_point", at::kFloat, {count, 3}, source);
    at::Tensor grad_source;
    at::Tensor grad_receiver;
    at::Tensor* grad_source_ptr = nullptr;
    at::Tensor* grad_receiver_ptr = nullptr;
    if (need_grad_source) {
        grad_source = at::empty({count, 3}, source.options());
        grad_source_ptr = &grad_source;
    }
    if (need_grad_receiver) {
        grad_receiver = at::empty({count, 3}, source.options());
        grad_receiver_ptr = &grad_receiver;
    }
    if (count > 0 && (g_edge_point != nullptr || g_reflection_point != nullptr)) {
        cudaStream_t stream = at::cuda::getCurrentCUDAStream(source.get_device()).stream();
        coupled_prepare_backward_kernel<<<launch_blocks(count), kBlockSize, 0, stream>>>(
            count,
            source.data_ptr<float>(),
            receiver.data_ptr<float>(),
            plane_point.data_ptr<float>(),
            plane_normal.data_ptr<float>(),
            edge_pos.data_ptr<float>(),
            edge_dir.data_ptr<float>(),
            edge_t_min.data_ptr<float>(),
            opt_ptr<float>(g_edge_point),
            opt_ptr<float>(g_reflection_point),
            opt_mut_ptr<float>(grad_source_ptr),
            opt_mut_ptr<float>(grad_receiver_ptr));
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    } else {
        if (grad_source_ptr != nullptr)
            grad_source_ptr->zero_();
        if (grad_receiver_ptr != nullptr)
            grad_receiver_ptr->zero_();
    }
    pybind11::dict out;
    out["grad_source"] = grad_source_ptr != nullptr
                             ? pybind11::cast(grad_source)
                             : pybind11::object(pybind11::none());
    out["grad_receiver"] = grad_receiver_ptr != nullptr
                               ? pybind11::cast(grad_receiver)
                               : pybind11::object(pybind11::none());
    return out;
}

pybind11::dict cn_coupled_rd_prepare_jvp(
    at::Tensor source,
    at::Tensor receiver,
    at::Tensor plane_point,
    at::Tensor plane_normal,
    at::Tensor edge_pos,
    at::Tensor edge_dir,
    at::Tensor edge_t_min,
    pybind11::object tangent_source,
    pybind11::object tangent_receiver) {
    check_prepare_rows(
        source, receiver, plane_point, plane_normal, edge_pos, edge_dir,
        edge_t_min);
    const int64_t count = source.size(0);
    at::Tensor storage[2];
    const at::Tensor* t_source = optional_tensor_arg(
        std::move(tangent_source), storage[0], "tangent_source", at::kFloat,
        {count, 3}, source);
    const at::Tensor* t_receiver = optional_tensor_arg(
        std::move(tangent_receiver), storage[1], "tangent_receiver", at::kFloat,
        {count, 3}, source);
    auto tangent_edge_point = at::empty({count, 3}, source.options());
    auto tangent_reflection_point = at::empty({count, 3}, source.options());
    if (count > 0) {
        cudaStream_t stream = at::cuda::getCurrentCUDAStream(source.get_device()).stream();
        coupled_prepare_jvp_kernel<<<launch_blocks(count), kBlockSize, 0, stream>>>(
            count,
            source.data_ptr<float>(),
            receiver.data_ptr<float>(),
            plane_point.data_ptr<float>(),
            plane_normal.data_ptr<float>(),
            edge_pos.data_ptr<float>(),
            edge_dir.data_ptr<float>(),
            edge_t_min.data_ptr<float>(),
            opt_ptr<float>(t_source),
            opt_ptr<float>(t_receiver),
            tangent_edge_point.data_ptr<float>(),
            tangent_reflection_point.data_ptr<float>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    pybind11::dict out;
    out["tangent_edge_point"] = tangent_edge_point;
    out["tangent_reflection_point"] = tangent_reflection_point;
    return out;
}
