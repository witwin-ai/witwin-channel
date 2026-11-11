// Copyright Xingyu Chen.
// Implements coupled path CUDA operations.

// ==== Section: Coupled wedge AD ====
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/util/complex.h>
#include "torch_cuda.h"
#include "constants.cuh"

#include <rayd/field_transport.cuh>
#include <src/field_transport_ad.cuh>
#include "../tensor_checks.h"
#include "ordered_sum.cuh"
#include "layered_reflection.cuh"

#include <array>
#include <type_traits>
#include <vector>

namespace {

constexpr int kBlockSize = 128;
namespace field = rayd::shared::diffraction;
namespace transport = rayd::shared::field_transport;
namespace ad = rayd::torch::field_transport_ad;
namespace layered = channel::layered_reflection;

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

}  // namespace

// Interaction-chain and double-diffraction fields with their derivative companions.
//
// The wedge field is RayD's own templated forward (rayd/utd.h): instantiated with float it is
// the production forward, instantiated with utd::Dual the same pass carries an exact directional
// derivative. Reverse mode runs one seeded dual pass per requested input scalar and contracts the
// output tangents with the cotangents; the torch.autograd.Function layer is dispatch only. The
// chain field below has one templated body for the primal and both derivative companions; its
// slab legs use the validated duals (slab_fresnel_dual, dual_reflect_frame) in the AD passes.

namespace {

// ---------------------------------------------------------------------------
// Bridges between the channel AD/2 dual types (DualC / DualF3) and
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

struct CoupledRowTangents {
    field::Complex3 field_vector;
    field::Complex coefficient;
    field::Complex path_field;
    float path_gain;
};

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

// ---------------------------------------------------------------------------
// Interaction chain field (components 3, 4 and 8 + slot): reflections, one
// diffraction at `slot`, reflections. One templated body serves the float
// primal and the dual JVP/VJP, so the three stay in lockstep by construction.
//
// The incident field reaches the edge through the prefix reflections as a
// spherical wave from the prefix image source (unfolded distance); the UTD
// pair sees the prefix image as its source and the suffix image of the
// receiver as its target, so its spreading uses the unfolded suffix length;
// the diffracted field is then reflected through the suffix planes. The edge
// geometry is frozen (chain rows carry no mesh-vertex gradient). In the dual
// passes the interaction points move with the endpoints through the image
// re-solve of the frozen winner (planes and edge line fixed); their primal
// values stay the discovered ones.
// ---------------------------------------------------------------------------

constexpr int kChainMaxDepth = 8;
constexpr int kChainWedge0 = kChainMaxDepth;
constexpr int kChainWedge1 = kChainMaxDepth + 1;
constexpr int kChainColumns = kChainMaxDepth + 2;

// Diffraction slot of a chain row: component 3 diffracts last, 4 first, 8 + s at slot s.
__device__ __forceinline__ int chain_diffraction_slot(int component, int depth) {
    if (component == channel::kComponentReflectionDiffraction)
        return depth - 1;
    if (component == channel::kComponentDiffractionReflection)
        return 0;
    return component - channel::kComponentChainInteriorBase;
}

struct ChainArgs {
    const float* source;
    const float* target;
    const float* positions;  // (N, W, 3)
    const float* normals;    // (N, W, 3)
    const int* component_id;
    const int* depth;
    int64_t width;
    const float* edge_direction;
    const float* edge_n0;
    const float* edge_n1;
    const float* exterior_angle;
    const float* edge_line_min;
    const float* edge_line_max;
    const float* tx_power;
    const float* tx_polarization;
    const float* rx_polarization;
    // (N, W + 2): reflection columns 0..W-1, wedge faces at W and W + 1.
    const float* eps_r;
    const float* sigma_e;
    const float* mu_r;
    const float* gain;
    const float* thickness;
    const int* bounce_material;  // (N, W + 2): layer-store row of each reflection slot, then both wedge faces
    layered::LayerTables layers;
    float frequency;
};

struct ChainRow {
    field::float3a source;
    field::float3a target;
    field::float3a position[kChainMaxDepth];
    field::float3a normal[kChainMaxDepth];
    int depth;
    int slot;
    bool regular;
    field::float3a edge_dir_raw;
    field::float3a n0;
    field::float3a n1;
    float exterior_angle;
    float edge_line_min;
    float edge_line_max;
    field::float3a tx_pol;
    field::float3a rx_pol;
    float tx_power;
    float eps[kChainColumns];
    float sigma[kChainColumns];
    float mu[kChainColumns];
    float gain[kChainColumns];
    float thickness[kChainColumns];
    int material[kChainColumns];
    float frequency;
};

__device__ __forceinline__ int chain_column(int64_t width, int column) {
    return column == kChainWedge0 ? static_cast<int>(width)
                                  : (column == kChainWedge1 ? static_cast<int>(width) + 1 : column);
}

__device__ ChainRow load_chain_row(const ChainArgs& a, int64_t index) {
    ChainRow row;
    row.source = load3f(a.source, index);
    row.target = load3f(a.target, index);
    row.depth = a.depth[index];
    row.slot = chain_diffraction_slot(a.component_id[index], row.depth);
    row.regular = row.depth >= 2 && row.depth <= a.width && row.depth <= kChainMaxDepth &&
                  row.slot >= 0 && row.slot < row.depth;
    const int depth = row.regular ? row.depth : 0;
    for (int i = 0; i < depth; ++i) {
        row.position[i] = load3f(a.positions, index * a.width + i);
        row.normal[i] = load3f(a.normals, index * a.width + i);
    }
    row.edge_dir_raw = load3f(a.edge_direction, index);
    row.n0 = load3f(a.edge_n0, index);
    row.n1 = load3f(a.edge_n1, index);
    row.exterior_angle = a.exterior_angle[index];
    row.edge_line_min = a.edge_line_min[index];
    row.edge_line_max = a.edge_line_max[index];
    row.tx_pol = load3f(a.tx_polarization, index);
    row.rx_pol = load3f(a.rx_polarization, index);
    row.tx_power = a.tx_power[index];
    const int64_t stride = a.width + 2;
    for (int column = 0; column < kChainColumns; ++column) {
        if (column < kChainWedge0 && column >= depth)
            continue;
        const int64_t at = index * stride + chain_column(a.width, column);
        row.eps[column] = a.eps_r[at];
        row.sigma[column] = a.sigma_e[at];
        row.mu[column] = a.mu_r[at];
        row.gain[column] = a.gain[at];
        row.thickness[column] = a.thickness[at];
        row.material[column] = a.bounce_material[at];
    }
    row.frequency = a.frequency;
    return row;
}

// The differentiable inputs of one chain row in scalar type T.
template <typename T>
struct ChainValues {
    field::Vec3T<T> source;
    field::Vec3T<T> target;
    field::Vec3T<T> position[kChainMaxDepth];
    T eps[kChainColumns];
    T sigma[kChainColumns];
    T gain[kChainColumns];
    T thickness[kChainColumns];
    T frequency;
};

template <typename T>
struct ChainOutput {
    field::Complex3T<T> field_vector;
    field::ComplexT<T> coefficient;
    field::Vec3T<T> direction;
};

__device__ __forceinline__ field::float3a chain_const3(field::float3a value, float) { return value; }
__device__ __forceinline__ DualV3 chain_const3(field::float3a value, Dual) { return field::dual_const3(value); }

__device__ __forceinline__ float chain_neg_kd(float wave_number, float distance) {
    return transport::precise_neg_kd(wave_number, distance);
}
__device__ __forceinline__ Dual chain_neg_kd(Dual wave_number, Dual distance) {
    return precise_neg_kd_dual(wave_number, distance);
}

template <typename T>
__device__ __forceinline__ field::Vec3T<T> chain_reflect_point(
    field::Vec3T<T> point, field::Vec3T<T> plane_point, field::Vec3T<T> raw_normal) {
    const field::Vec3T<T> n = field::safe_normalize(raw_normal, field::v3_const<T>(0.f, 0.f, 1.f));
    return field::f3_sub(point, field::f3_mul(n, 2.0f * field::f3_dot(field::f3_sub(point, plane_point), n)));
}

// Free-space field of the transmit dipole at the edge (the first leg of a chain that diffracts first).
__device__ __forceinline__ field::Complex3 chain_free_space(
    field::float3a source, field::float3a edge, float wave_number, field::float3a tx_pol) {
    return transport::free_space_complex3(source, edge, wave_number, tx_pol);
}
__device__ DualC3 chain_free_space(DualV3 source, DualV3 edge, Dual wave_number, field::float3a tx_pol) {
    const DualV3 offset = field::f3_sub(edge, source);
    const Dual distance = field::safe_length(offset);
    const DualV3 direction = field::safe_normalize(offset, field::v3_const<Dual>(0.f, 0.f, 1.f));
    const DualV3 axis = field::project_to_wedge_plane(field::dual_const3(tx_pol), direction);
    const Dual amplitude = 1.0f / (2.0f * field::fmaxf(wave_number, Dual(field::UTD_SMALL_EPS)) *
                                   field::fmaxf(distance, Dual(field::UTD_EPS)));
    const DualCx phase = field::cplx_exp_phase(precise_neg_kd_dual(wave_number, distance));
    return field::cplx_scale_real(axis, field::cplx_mul_real(phase, amplitude));
}

// Specular reflection at one chain slot: the layer store in the primal, the slab duals in the AD passes.
__device__ __forceinline__ field::Complex3 chain_reflect(
    field::Complex3 value, field::float3a direction, field::float3a normal, const ChainRow& row,
    const ChainValues<float>& v, int column, const layered::LayerTables& layers,
    field::float3a& reflected_direction) {
    return layered::reflect_layered(
        value, direction, normal, v.eps[column], v.sigma[column], row.mu[column], v.gain[column],
        v.thickness[column], v.frequency, row.material[column], layers, reflected_direction);
}
__device__ __forceinline__ DualC3 chain_reflect(
    DualC3 value, DualV3 direction, DualV3 normal, const ChainRow& row, const ChainValues<Dual>& v,
    int column, const layered::LayerTables&, DualV3& reflected_direction) {
    return reflect_complex3_dual(
        value, direction, normal, v.eps[column], v.sigma[column], row.mu[column], v.gain[column],
        v.thickness[column], v.frequency, reflected_direction);
}

// Reflection coefficients of the two wedge faces of a chain row: the layer store in the primal, the slab duals
// in the AD passes (as the chain's reflections).
template <typename T>
__device__ __forceinline__ layered::WedgeFaceFresnel<T> chain_face_fresnel(
    const ChainRow& row, const ChainValues<T>& v, const layered::LayerTables& layers, bool layered_store) {
    layered::WedgeFaceFresnel<T> faces;
    for (int face = 0; face < 2; ++face) {
        const int column = kChainWedge0 + face;
        faces.eps[face] = v.eps[column];
        faces.sigma[face] = v.sigma[column];
        faces.mu[face] = row.mu[column];
        faces.gain[face] = v.gain[column];
        faces.thickness[face] = v.thickness[column];
        faces.material[face] = layered_store ? row.material[column] : -1;
    }
    if (layered_store)
        faces.tables = layers;
    faces.frequency_hz = v.frequency;
    return faces;
}

template <typename T>
__device__ __forceinline__ field::ComplexT<T> chain_project(
    field::Complex3T<T> value, field::Vec3T<T> direction, field::Vec3T<T> polarization) {
    const field::Vec3T<T> axis = field::project_to_wedge_plane(polarization, direction);
    return field::cplx_add(
        field::cplx_add(field::cplx_mul_real(value.x, axis.x), field::cplx_mul_real(value.y, axis.y)),
        field::cplx_mul_real(value.z, axis.z));
}

template <typename T>
__device__ ChainOutput<T> chain_row_body(
    const ChainRow& row, const ChainValues<T>& v, const layered::LayerTables& layers) {
    using V3 = field::Vec3T<T>;
    const T tag{};
    const int depth = row.depth;
    const int slot = row.slot;
    const T wave_number = 2.0f * field::UTD_PI * v.frequency / transport::kSpeedOfLight;
    const V3 edge = v.position[slot];
    const V3 edge_axis = field::safe_normalize(chain_const3(row.edge_dir_raw, tag), field::v3_const<T>(0.f, 0.f, 1.f));

    V3 diffraction_source = v.source;
    for (int i = 0; i < slot; ++i)
        diffraction_source = chain_reflect_point(diffraction_source, v.position[i], chain_const3(row.normal[i], tag));
    V3 diffraction_target = v.target;
    for (int i = depth - 1; i > slot; --i)
        diffraction_target = chain_reflect_point(diffraction_target, v.position[i], chain_const3(row.normal[i], tag));

    const V3 incident_direction =
        field::safe_normalize(field::f3_sub(edge, diffraction_source), field::v3_const<T>(1.f, 0.f, 0.f));
    const V3 outgoing_direction =
        field::safe_normalize(field::f3_sub(diffraction_target, edge), field::v3_const<T>(1.f, 0.f, 0.f));
    const field::Basis3T<T> input_edge_basis =
        field::diffraction_edge_basis(field::f3_sub(edge, diffraction_source), edge_axis, false);

    field::Complex3T<T> incident_field;
    if (slot == 0) {
        incident_field = chain_free_space(v.source, edge, wave_number, row.tx_pol);
    } else {
        V3 direction = field::safe_normalize(field::f3_sub(v.position[0], v.source), incident_direction);
        // Unnormalized transverse projection of the transmit polarization.
        const V3 tx_axis = field::project_to_wedge_plane(chain_const3(row.tx_pol, tag), direction);
        field::Complex3T<T> value = field::cplx_scale_real(tx_axis, field::c_const<T>(1.f, 0.f));
        for (int i = 0; i < slot; ++i) {
            V3 reflected_direction;
            value = chain_reflect(
                value, direction, chain_const3(row.normal[i], tag), row, v, i, layers, reflected_direction);
            const V3 next = i + 1 < slot ? v.position[i + 1] : edge;
            direction = field::safe_normalize(field::f3_sub(next, v.position[i]), reflected_direction);
        }
        const T unfolded_distance = field::safe_length(field::f3_sub(edge, diffraction_source));
        const T amplitude = 1.0f / (2.0f * wave_number * field::fmaxf(unfolded_distance, T(field::UTD_EPS)));
        const field::ComplexT<T> propagation =
            field::cplx_mul_real(field::cplx_exp_phase(chain_neg_kd(wave_number, unfolded_distance)), amplitude);
        incident_field = field::c3_scale(value, propagation);
    }

    field::PairInputsT<T> pair{};
    pair.edgePos = edge;
    pair.edgeDir = edge_axis;
    pair.n0 = chain_const3(row.n0, tag);
    pair.nn = chain_const3(row.n1, tag);
    pair.wedgeN = T(row.exterior_angle / field::UTD_PI);
    // Real segment bounds let the stationary solver truncate the diffraction leg and apply its
    // corner correction around the supplied edge point; they carry no derivative.
    pair.edgeLineMin = T(row.edge_line_min);
    pair.edgeLineMax = T(row.edge_line_max);
    pair.sourcePos = diffraction_source;
    pair.incidentBasis = input_edge_basis;
    pair.incidentJones = field::jones_from_vector(incident_field, input_edge_basis);
    pair.incidentDerivativeJones = field::jones_zero<T>();
    pair.face0Material.present = 1.0f;
    pair.face1Material.present = 1.0f;
    pair.face0Material.useFresnel = 1.0f;
    pair.face1Material.useFresnel = 1.0f;
    // The stationary-path machinery (edge re-anchoring, monotone even truncation, corner
    // correction, boundary blend) runs on the diffraction leg; stationaryExternalIncident
    // re-extrapolates the image-source incident field stored in incidentJones to the re-anchored
    // point. The face terms reflect with the faces' own slab or layer-stack coefficients.
    pair.selectStationaryPoint = 1.0f;
    pair.stationaryExternalIncident = 1.0f;
    field::MaterialParamsT<T> material{};
    material.omega = T(-1.0f);
    const bool layered_store = std::is_same<T, float>::value;
    field::Complex3T<T> value = field::compute_pair_vector_contribution_faces(
        pair, diffraction_target, wave_number, material, chain_face_fresnel(row, v, layers, layered_store));

    V3 final_direction = outgoing_direction;
    V3 previous = edge;
    for (int i = slot + 1; i < depth; ++i) {
        const V3 direction = field::safe_normalize(field::f3_sub(v.position[i], previous), final_direction);
        value = chain_reflect(value, direction, chain_const3(row.normal[i], tag), row, v, i, layers, final_direction);
        previous = v.position[i];
    }
    ChainOutput<T> out;
    out.field_vector = value;
    out.coefficient = chain_project(value, final_direction, chain_const3(row.rx_pol, tag));
    out.direction = final_direction;
    return out;
}

__device__ ChainValues<float> chain_primal_values(const ChainRow& row) {
    ChainValues<float> v;
    v.source = row.source;
    v.target = row.target;
    for (int i = 0; i < row.depth; ++i)
        v.position[i] = row.position[i];
    for (int column = 0; column < kChainColumns; ++column) {
        if (column < kChainWedge0 && column >= row.depth)
            continue;
        v.eps[column] = row.eps[column];
        v.sigma[column] = row.sigma[column];
        v.gain[column] = row.gain[column];
        v.thickness[column] = row.thickness[column];
    }
    v.frequency = row.frequency;
    return v;
}

struct ChainSeeds {
    field::float3a source;
    field::float3a target;
    float eps[kChainColumns];
    float sigma[kChainColumns];
    float gain[kChainColumns];
    float thickness[kChainColumns];
    float frequency;
};

__device__ __forceinline__ ChainSeeds chain_seeds_zero() {
    ChainSeeds seeds;
    seeds.source = field::f3_zero();
    seeds.target = field::f3_zero();
    for (int column = 0; column < kChainColumns; ++column) {
        seeds.eps[column] = 0.f;
        seeds.sigma[column] = 0.f;
        seeds.gain[column] = 0.f;
        seeds.thickness[column] = 0.f;
    }
    seeds.frequency = 0.f;
    return seeds;
}

// Point where the segment start -> end crosses the plane through plane_point with normal raw_normal.
__device__ __forceinline__ DualV3 chain_plane_crossing(DualV3 start, DualV3 end, DualV3 plane_point, DualV3 raw_normal) {
    const DualV3 n = field::safe_normalize(raw_normal, field::v3_const<Dual>(0.f, 0.f, 1.f));
    const DualV3 span = field::f3_sub(end, start);
    const Dual t = field::f3_dot(field::f3_sub(plane_point, start), n) / field::f3_dot(span, n);
    return field::f3_add(start, field::f3_mul(span, t));
}

// Re-solve of the frozen winner: the edge point is the Keller point between the prefix image
// of the source and the suffix image of the target, and every reflection point is back-traced
// from it through the image chain. Only the tangents are taken; the primal values stay frozen.
__device__ void chain_resolve_dual(const ChainRow& row, const ChainSeeds& seeds, DualV3 (&position)[kChainMaxDepth]) {
    const int depth = row.depth;
    const int slot = row.slot;
    const bool moving = seeds.source.x != 0.f || seeds.source.y != 0.f || seeds.source.z != 0.f ||
                        seeds.target.x != 0.f || seeds.target.y != 0.f || seeds.target.z != 0.f;
    if (!moving) {
        for (int i = 0; i < depth; ++i)
            position[i] = field::dual_const3(row.position[i]);
        return;
    }
    DualV3 prefix_image[kChainMaxDepth + 1];
    DualV3 suffix_image[kChainMaxDepth + 1];
    prefix_image[0] = field::dual_seed(row.source, seeds.source);
    for (int i = 0; i < slot; ++i)
        prefix_image[i + 1] = chain_reflect_point(
            prefix_image[i], field::dual_const3(row.position[i]), field::dual_const3(row.normal[i]));
    suffix_image[0] = field::dual_seed(row.target, seeds.target);
    for (int i = depth - 1, j = 0; i > slot; --i, ++j)
        suffix_image[j + 1] = chain_reflect_point(
            suffix_image[j], field::dual_const3(row.position[i]), field::dual_const3(row.normal[i]));
    const DualV3 origin = field::dual_const3(row.position[slot]);
    const DualV3 axis = field::safe_normalize(field::dual_const3(row.edge_dir_raw), field::v3_const<Dual>(0.f, 0.f, 1.f));
    const Dual parameter =
        field::first_order_diffraction_parameter(prefix_image[slot], suffix_image[depth - 1 - slot], origin, axis);
    position[slot] = field::f3_add(origin, field::f3_mul(axis, parameter));
    DualV3 next = position[slot];
    for (int i = slot - 1; i >= 0; --i) {
        position[i] = chain_plane_crossing(
            prefix_image[i + 1], next, field::dual_const3(row.position[i]), field::dual_const3(row.normal[i]));
        next = position[i];
    }
    next = position[slot];
    for (int i = slot + 1; i < depth; ++i) {
        position[i] = chain_plane_crossing(
            suffix_image[depth - i], next, field::dual_const3(row.position[i]), field::dual_const3(row.normal[i]));
        next = position[i];
    }
    for (int i = 0; i < depth; ++i) {
        position[i].x.v = row.position[i].x;
        position[i].y.v = row.position[i].y;
        position[i].z.v = row.position[i].z;
    }
}

__device__ CoupledRowTangents chain_row_dual(const ChainRow& row, const ChainSeeds& seeds) {
    ChainValues<Dual> v;
    v.source = field::dual_seed(row.source, seeds.source);
    v.target = field::dual_seed(row.target, seeds.target);
    chain_resolve_dual(row, seeds, v.position);
    for (int column = 0; column < kChainColumns; ++column) {
        if (column < kChainWedge0 && column >= row.depth)
            continue;
        v.eps[column] = {row.eps[column], seeds.eps[column]};
        v.sigma[column] = {row.sigma[column], seeds.sigma[column]};
        v.gain[column] = {row.gain[column], seeds.gain[column]};
        v.thickness[column] = {row.thickness[column], seeds.thickness[column]};
    }
    v.frequency = {row.frequency, seeds.frequency};
    const layered::LayerTables no_layers{};
    const ChainOutput<Dual> body = chain_row_body(row, v, no_layers);
    const float amplitude_scale = sqrtf(fmaxf(row.tx_power, 0.f));
    const DualCx path_field = field::cplx_mul_real(body.coefficient, amplitude_scale);
    CoupledRowTangents out;
    out.field_vector = field::dual_tangent(body.field_vector);
    out.coefficient = field::dual_tangent(body.coefficient);
    out.path_field = field::dual_tangent(path_field);
    out.path_gain = 2.0f * (path_field.re.v * path_field.re.d + path_field.im.v * path_field.im.d);
    return out;
}

__global__ void coupled_chain_field_kernel(
    int64_t count, ChainArgs args, c10::complex<float>* field_vector, c10::complex<float>* coefficient,
    c10::complex<float>* path_field, float* path_gain, float* direction_out) {
    for (int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; index < count;
         index += static_cast<int64_t>(blockDim.x) * gridDim.x) {
        const ChainRow row = load_chain_row(args, index);
        const int64_t base = index * 3;
        if (!row.regular) {
            // A malformed row is inert: zero field, NaN direction.
            for (int axis = 0; axis < 3; ++axis) {
                field_vector[base + axis] = c10::complex<float>(0.f, 0.f);
                direction_out[base + axis] = NAN;
            }
            coefficient[index] = c10::complex<float>(0.f, 0.f);
            path_field[index] = c10::complex<float>(0.f, 0.f);
            path_gain[index] = 0.f;
            continue;
        }
        const ChainOutput<float> out = chain_row_body(row, chain_primal_values(row), args.layers);
        field_vector[base] = to_c10(out.field_vector.x);
        field_vector[base + 1] = to_c10(out.field_vector.y);
        field_vector[base + 2] = to_c10(out.field_vector.z);
        coefficient[index] = to_c10(out.coefficient);
        const field::Complex received = field::cplx_mul_real(out.coefficient, sqrtf(fmaxf(row.tx_power, 0.0f)));
        path_field[index] = to_c10(received);
        path_gain[index] = field::cplx_abs_sqr(received);
        direction_out[base] = out.direction.x;
        direction_out[base + 1] = out.direction.y;
        direction_out[base + 2] = out.direction.z;
    }
}

// Material parameters a chain row differentiates: the reflection slots and both wedge faces.
__device__ __forceinline__ bool chain_material_column(const ChainRow& row, int column) {
    return column >= kChainWedge0 || (column < row.depth && column != row.slot);
}

__global__ void coupled_chain_backward_kernel(
    int64_t count, ChainArgs args, const c10::complex<float>* grad_field_vector,
    const c10::complex<float>* grad_coefficient, const c10::complex<float>* grad_path_field,
    const float* grad_path_gain, float* grad_source, float* grad_target, float* grad_eps_r, float* grad_sigma_e,
    float* grad_gain, float* grad_thickness, float* grad_frequency) {
    for (int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; index < count;
         index += static_cast<int64_t>(blockDim.x) * gridDim.x) {
        const ChainRow row = load_chain_row(args, index);
        const int64_t base = index * 3;
        const int64_t stride = args.width + 2;
        ChainSeeds seeds = chain_seeds_zero();
        float* vector_grads[2] = {grad_source, grad_target};
        field::float3a* vector_seeds[2] = {&seeds.source, &seeds.target};
        for (int slot = 0; slot < 2; ++slot) {
            if (vector_grads[slot] == nullptr)
                continue;
            float* axes[3] = {&vector_seeds[slot]->x, &vector_seeds[slot]->y, &vector_seeds[slot]->z};
            for (int axis = 0; axis < 3; ++axis) {
                float value = 0.f;
                if (row.regular) {
                    *axes[axis] = 1.f;
                    value = coupled_contract(
                        index, grad_field_vector, grad_coefficient, grad_path_field, grad_path_gain,
                        chain_row_dual(row, seeds));
                    *axes[axis] = 0.f;
                }
                vector_grads[slot][base + axis] = value;
            }
        }
        float* material_grads[4] = {grad_eps_r, grad_sigma_e, grad_gain, grad_thickness};
        float* material_seeds[4] = {seeds.eps, seeds.sigma, seeds.gain, seeds.thickness};
        for (int parameter = 0; parameter < 4; ++parameter) {
            float* grad = material_grads[parameter];
            if (grad == nullptr)
                continue;
            for (int64_t column = 0; column < stride; ++column)
                grad[index * stride + column] = 0.f;
            if (!row.regular)
                continue;
            for (int column = 0; column < kChainColumns; ++column) {
                if (!chain_material_column(row, column))
                    continue;
                material_seeds[parameter][column] = 1.f;
                grad[index * stride + chain_column(args.width, column)] = coupled_contract(
                    index, grad_field_vector, grad_coefficient, grad_path_field, grad_path_gain,
                    chain_row_dual(row, seeds));
                material_seeds[parameter][column] = 0.f;
            }
        }
        if (grad_frequency != nullptr) {
            float value = 0.f;
            if (row.regular) {
                seeds.frequency = 1.f;
                value = coupled_contract(
                    index, grad_field_vector, grad_coefficient, grad_path_field, grad_path_gain,
                    chain_row_dual(row, seeds));
                seeds.frequency = 0.f;
            }
            grad_frequency[index] = value;
        }
    }
}

__global__ void coupled_chain_jvp_kernel(
    int64_t count, ChainArgs args, const float* tangent_source, const float* tangent_target,
    const float* tangent_eps_r, const float* tangent_sigma_e, const float* tangent_gain,
    const float* tangent_thickness, float tangent_frequency, c10::complex<float>* tangent_field_vector,
    c10::complex<float>* tangent_coefficient, c10::complex<float>* tangent_path_field, float* tangent_path_gain) {
    for (int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; index < count;
         index += static_cast<int64_t>(blockDim.x) * gridDim.x) {
        const ChainRow row = load_chain_row(args, index);
        const int64_t base = index * 3;
        const int64_t stride = args.width + 2;
        CoupledRowTangents t{};
        if (row.regular) {
            ChainSeeds seeds = chain_seeds_zero();
            if (tangent_source != nullptr)
                seeds.source = load3f(tangent_source, index);
            if (tangent_target != nullptr)
                seeds.target = load3f(tangent_target, index);
            const float* tangents[4] = {tangent_eps_r, tangent_sigma_e, tangent_gain, tangent_thickness};
            float* material_seeds[4] = {seeds.eps, seeds.sigma, seeds.gain, seeds.thickness};
            for (int parameter = 0; parameter < 4; ++parameter) {
                if (tangents[parameter] == nullptr)
                    continue;
                for (int column = 0; column < kChainColumns; ++column) {
                    if (chain_material_column(row, column))
                        material_seeds[parameter][column] =
                            tangents[parameter][index * stride + chain_column(args.width, column)];
                }
            }
            seeds.frequency = tangent_frequency;
            t = chain_row_dual(row, seeds);
        }
        tangent_field_vector[base] = to_c10(t.field_vector.x);
        tangent_field_vector[base + 1] = to_c10(t.field_vector.y);
        tangent_field_vector[base + 2] = to_c10(t.field_vector.z);
        tangent_coefficient[index] = to_c10(t.coefficient);
        tangent_path_field[index] = to_c10(t.path_field);
        tangent_path_gain[index] = t.path_gain;
    }
}

// ---------------------------------------------------------------------------
// coupled double diffraction: coupled double-diffraction dual row (component 7). Mirrors
// coupled_dd_field_kernel (fields.cu) step by step on RayD duals.
//
// Both legs run the stationary path so the truncation / corner-mend / boundary
// blend derivatives flow through the dual in lockstep with the primal (the
// dual calls compute_pair_vector_contribution directly, exactly like the
// order-1 diffraction dual and the coupled R->D dual after the coupled-path approximation). The
// gradient contract (coupled double diffraction): tx (source) and rx (target) flow through the
// live per-leg re-anchoring; Q1/Q2, the edge axes/normals/exterior, the edge
// bounds and the polarizations are frozen seeds (detached), so they are loaded
// as dual constants and carry no gradient. The wedge face materials and the
// frequency are differentiable. This file is a PRECISE-math translation unit
// (only RayD's pure-wedge translation unit is fast-math), matching the primal.
// Edit coupled_dd_field_kernel and this mirror TOGETHER.
// ---------------------------------------------------------------------------

struct CoupledDdRowInputs {
    field::float3a source;
    field::float3a target;
    field::float3a q1;
    field::float3a q2;
    field::float3a e1_dir_raw;
    field::float3a e2_dir_raw;
    field::float3a e1_n0, e1_n1;
    field::float3a e2_n0, e2_n1;
    float e1_exterior, e2_exterior;
    float e1_line_min, e1_line_max;
    float e2_line_min, e2_line_max;
    field::float3a tx_pol;
    field::float3a rx_pol;
    float tx_power;
    // wedge1 face0 (a) / face1 (b), wedge2 face0 (a) / face1 (b).
    float w1a_eps, w1a_sigma, w1a_mu, w1a_gain, w1a_thickness;
    float w1b_eps, w1b_sigma, w1b_mu, w1b_gain, w1b_thickness;
    float w2a_eps, w2a_sigma, w2a_mu, w2a_gain, w2a_thickness;
    float w2b_eps, w2b_sigma, w2b_mu, w2b_gain, w2b_thickness;
    float frequency;
};

struct CoupledDdSeeds {
    field::float3a source;
    field::float3a target;
    float w1a_eps, w1a_sigma, w1a_gain, w1a_thickness;
    float w1b_eps, w1b_sigma, w1b_gain, w1b_thickness;
    float w2a_eps, w2a_sigma, w2a_gain, w2a_thickness;
    float w2b_eps, w2b_sigma, w2b_gain, w2b_thickness;
    float frequency;
};

__device__ __forceinline__ CoupledDdSeeds coupled_dd_seeds_zero() {
    CoupledDdSeeds seeds;
    seeds.source = field::f3_zero();
    seeds.target = field::f3_zero();
    seeds.w1a_eps = 0.f; seeds.w1a_sigma = 0.f;
    seeds.w1a_gain = 0.f; seeds.w1a_thickness = 0.f;
    seeds.w1b_eps = 0.f; seeds.w1b_sigma = 0.f;
    seeds.w1b_gain = 0.f; seeds.w1b_thickness = 0.f;
    seeds.w2a_eps = 0.f; seeds.w2a_sigma = 0.f;
    seeds.w2a_gain = 0.f; seeds.w2a_thickness = 0.f;
    seeds.w2b_eps = 0.f; seeds.w2b_sigma = 0.f;
    seeds.w2b_gain = 0.f; seeds.w2b_thickness = 0.f;
    seeds.frequency = 0.f;
    return seeds;
}

// Reuses CoupledRowTangents (identical output shape) and coupled_contract.
__device__ CoupledRowTangents coupled_dd_row_dual(
    const CoupledDdRowInputs& in, const CoupledDdSeeds& seeds) {
    const DualV3 src = field::dual_seed(in.source, seeds.source);
    const DualV3 dst = field::dual_seed(in.target, seeds.target);
    // Q1/Q2 are the frozen discovery Fermat seeds (detached): dual constants.
    const DualV3 q1 = field::dual_const3(in.q1);
    const DualV3 q2 = field::dual_const3(in.q2);
    const Dual frequency = {in.frequency, seeds.frequency};
    const Dual wave_number =
        2.0f * field::UTD_PI * frequency / transport::kSpeedOfLight;
    const DualV3 e1_axis = field::safe_normalize(
        field::dual_const3(in.e1_dir_raw), field::v3_const<Dual>(0.f, 0.f, 1.f));
    const DualV3 e2_axis = field::safe_normalize(
        field::dual_const3(in.e2_dir_raw), field::v3_const<Dual>(0.f, 0.f, 1.f));

    // --- Leg 1: direct-source diffraction at e1, observed at frozen Q2 ---
    const field::Basis3T<Dual> input_edge_basis1 = field::diffraction_edge_basis(
        field::f3_sub(q1, src), e1_axis, false);

    field::PairInputsT<Dual> pair1{};
    pair1.edgePos = q1;
    pair1.edgeDir = e1_axis;
    pair1.n0 = field::dual_const3(in.e1_n0);
    pair1.nn = field::dual_const3(in.e1_n1);
    pair1.wedgeN = Dual(in.e1_exterior / field::UTD_PI);
    pair1.edgeLineMin = Dual(in.e1_line_min);
    pair1.edgeLineMax = Dual(in.e1_line_max);
    pair1.sourcePos = src;
    // Direct-source path ignores incidentBasis/Jones; set them consistently.
    pair1.incidentBasis = input_edge_basis1;
    pair1.incidentJones = field::jones_zero<Dual>();
    pair1.incidentDerivativeJones = field::jones_zero<Dual>();
    pair1.face0Material.present = 1.0f;
    pair1.face1Material.present = 1.0f;
    pair1.face0Material.useFresnel = 1.0f;
    pair1.face1Material.useFresnel = 1.0f;
    layered::WedgeFaceFresnel<Dual> faces1;
    faces1.eps[0] = {in.w1a_eps, seeds.w1a_eps};
    faces1.sigma[0] = {in.w1a_sigma, seeds.w1a_sigma};
    faces1.mu[0] = in.w1a_mu;
    faces1.gain[0] = {in.w1a_gain, seeds.w1a_gain};
    faces1.thickness[0] = {in.w1a_thickness, seeds.w1a_thickness};
    faces1.eps[1] = {in.w1b_eps, seeds.w1b_eps};
    faces1.sigma[1] = {in.w1b_sigma, seeds.w1b_sigma};
    faces1.mu[1] = in.w1b_mu;
    faces1.gain[1] = {in.w1b_gain, seeds.w1b_gain};
    faces1.thickness[1] = {in.w1b_thickness, seeds.w1b_thickness};
    faces1.frequency_hz = frequency;
    pair1.selectStationaryPoint = 1.0f;
    pair1.stationaryExternalIncident = 0.0f;
    field::MaterialParamsT<Dual> material1{};
    // The face terms reflect with the wedge faces' slab coefficients; the direct incident
    // spherical wave uses the (frozen) transmitter polarization, exactly like coupled_dd_field_kernel.
    material1.omega = Dual(-1.0f);
    material1.txPolX = Dual(in.tx_pol.x);
    material1.txPolY = Dual(in.tx_pol.y);
    material1.txPolZ = Dual(in.tx_pol.z);
    const DualC3 leg1_field = field::compute_pair_vector_contribution_faces(
        pair1, q2, wave_number, material1, faces1);

    // --- Leg 2: external-incident diffraction at e2, observed at rx ---
    const DualV3 outgoing_direction2 = field::safe_normalize(
        field::f3_sub(dst, q2), field::v3_const<Dual>(1.f, 0.f, 0.f));
    const field::Basis3T<Dual> input_edge_basis2 = field::diffraction_edge_basis(
        field::f3_sub(q2, q1), e2_axis, false);

    field::PairInputsT<Dual> pair2{};
    pair2.edgePos = q2;
    pair2.edgeDir = e2_axis;
    pair2.n0 = field::dual_const3(in.e2_n0);
    pair2.nn = field::dual_const3(in.e2_n1);
    pair2.wedgeN = Dual(in.e2_exterior / field::UTD_PI);
    pair2.edgeLineMin = Dual(in.e2_line_min);
    pair2.edgeLineMax = Dual(in.e2_line_max);
    pair2.sourcePos = q1;
    pair2.incidentBasis = input_edge_basis2;
    // Leg-1 diffracted field projected onto e2's incident basis at frozen Q2.
    pair2.incidentJones = field::jones_from_vector(leg1_field, input_edge_basis2);
    pair2.incidentDerivativeJones = field::jones_zero<Dual>();
    pair2.face0Material.present = 1.0f;
    pair2.face1Material.present = 1.0f;
    pair2.face0Material.useFresnel = 1.0f;
    pair2.face1Material.useFresnel = 1.0f;
    layered::WedgeFaceFresnel<Dual> faces2;
    faces2.eps[0] = {in.w2a_eps, seeds.w2a_eps};
    faces2.sigma[0] = {in.w2a_sigma, seeds.w2a_sigma};
    faces2.mu[0] = in.w2a_mu;
    faces2.gain[0] = {in.w2a_gain, seeds.w2a_gain};
    faces2.thickness[0] = {in.w2a_thickness, seeds.w2a_thickness};
    faces2.eps[1] = {in.w2b_eps, seeds.w2b_eps};
    faces2.sigma[1] = {in.w2b_sigma, seeds.w2b_sigma};
    faces2.mu[1] = in.w2b_mu;
    faces2.gain[1] = {in.w2b_gain, seeds.w2b_gain};
    faces2.thickness[1] = {in.w2b_thickness, seeds.w2b_thickness};
    faces2.frequency_hz = frequency;
    pair2.selectStationaryPoint = 1.0f;
    pair2.stationaryExternalIncident = 1.0f;
    field::MaterialParamsT<Dual> material2{};
    material2.omega = Dual(-1.0f);
    const DualC3 value = field::compute_pair_vector_contribution_faces(
        pair2, dst, wave_number, material2, faces2);

    const DualV3 final_direction = outgoing_direction2;
    const DualV3 rx_axis = field::project_to_wedge_plane(
        field::dual_const3(in.rx_pol), final_direction);
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

__device__ __forceinline__ CoupledDdRowInputs load_coupled_dd_row(
    int64_t index,
    const float* source,
    const float* target,
    const float* edge1_position,
    const float* edge1_direction,
    const float* edge1_n0,
    const float* edge1_n1,
    const float* edge1_exterior,
    const float* edge2_position,
    const float* edge2_direction,
    const float* edge2_n0,
    const float* edge2_n1,
    const float* edge2_exterior,
    const float* tx_power,
    const float* tx_polarization,
    const float* rx_polarization,
    const float* wedge1_eps_r0,
    const float* wedge1_sigma_e0,
    const float* wedge1_mu_r0,
    const float* wedge1_gain0,
    const float* wedge1_thickness0,
    const float* wedge1_eps_r1,
    const float* wedge1_sigma_e1,
    const float* wedge1_mu_r1,
    const float* wedge1_gain1,
    const float* wedge1_thickness1,
    const float* wedge2_eps_r0,
    const float* wedge2_sigma_e0,
    const float* wedge2_mu_r0,
    const float* wedge2_gain0,
    const float* wedge2_thickness0,
    const float* wedge2_eps_r1,
    const float* wedge2_sigma_e1,
    const float* wedge2_mu_r1,
    const float* wedge2_gain1,
    const float* wedge2_thickness1,
    const float* edge1_line_min,
    const float* edge1_line_max,
    const float* edge2_line_min,
    const float* edge2_line_max,
    float frequency_hz) {
    CoupledDdRowInputs in;
    in.source = load3f(source, index);
    in.target = load3f(target, index);
    in.q1 = load3f(edge1_position, index);
    in.q2 = load3f(edge2_position, index);
    in.e1_dir_raw = load3f(edge1_direction, index);
    in.e2_dir_raw = load3f(edge2_direction, index);
    in.e1_n0 = load3f(edge1_n0, index);
    in.e1_n1 = load3f(edge1_n1, index);
    in.e2_n0 = load3f(edge2_n0, index);
    in.e2_n1 = load3f(edge2_n1, index);
    in.e1_exterior = edge1_exterior[index];
    in.e2_exterior = edge2_exterior[index];
    in.e1_line_min = edge1_line_min[index];
    in.e1_line_max = edge1_line_max[index];
    in.e2_line_min = edge2_line_min[index];
    in.e2_line_max = edge2_line_max[index];
    in.tx_pol = load3f(tx_polarization, index);
    in.rx_pol = load3f(rx_polarization, index);
    in.tx_power = tx_power[index];
    in.w1a_eps = wedge1_eps_r0[index];
    in.w1a_sigma = wedge1_sigma_e0[index];
    in.w1a_mu = wedge1_mu_r0[index];
    in.w1a_gain = wedge1_gain0[index];
    in.w1a_thickness = wedge1_thickness0[index];
    in.w1b_eps = wedge1_eps_r1[index];
    in.w1b_sigma = wedge1_sigma_e1[index];
    in.w1b_mu = wedge1_mu_r1[index];
    in.w1b_gain = wedge1_gain1[index];
    in.w1b_thickness = wedge1_thickness1[index];
    in.w2a_eps = wedge2_eps_r0[index];
    in.w2a_sigma = wedge2_sigma_e0[index];
    in.w2a_mu = wedge2_mu_r0[index];
    in.w2a_gain = wedge2_gain0[index];
    in.w2a_thickness = wedge2_thickness0[index];
    in.w2b_eps = wedge2_eps_r1[index];
    in.w2b_sigma = wedge2_sigma_e1[index];
    in.w2b_mu = wedge2_mu_r1[index];
    in.w2b_gain = wedge2_gain1[index];
    in.w2b_thickness = wedge2_thickness1[index];
    in.frequency = frequency_hz;
    return in;
}

#define COUPLED_DD_ROW_PARAMS                                                  \
    const float* source, const float* target,                                 \
        const float* edge1_position, const float* edge1_direction,            \
        const float* edge1_n0, const float* edge1_n1,                         \
        const float* edge1_exterior, const float* edge2_position,             \
        const float* edge2_direction, const float* edge2_n0,                  \
        const float* edge2_n1, const float* edge2_exterior,                   \
        const float* tx_power, const float* tx_polarization,                  \
        const float* rx_polarization, const float* wedge1_eps_r0,             \
        const float* wedge1_sigma_e0, const float* wedge1_mu_r0,              \
        const float* wedge1_gain0, const float* wedge1_thickness0,            \
        const float* wedge1_eps_r1, const float* wedge1_sigma_e1,             \
        const float* wedge1_mu_r1, const float* wedge1_gain1,                 \
        const float* wedge1_thickness1, const float* wedge2_eps_r0,           \
        const float* wedge2_sigma_e0, const float* wedge2_mu_r0,              \
        const float* wedge2_gain0, const float* wedge2_thickness0,            \
        const float* wedge2_eps_r1, const float* wedge2_sigma_e1,             \
        const float* wedge2_mu_r1, const float* wedge2_gain1,                 \
        const float* wedge2_thickness1, const float* edge1_line_min,          \
        const float* edge1_line_max, const float* edge2_line_min,             \
        const float* edge2_line_max, float frequency_hz

#define COUPLED_DD_ROW_ARGS(index)                                            \
    index, source, target, edge1_position, edge1_direction, edge1_n0,         \
        edge1_n1, edge1_exterior, edge2_position, edge2_direction, edge2_n0,  \
        edge2_n1, edge2_exterior, tx_power, tx_polarization, rx_polarization, \
        wedge1_eps_r0, wedge1_sigma_e0, wedge1_mu_r0, wedge1_gain0,           \
        wedge1_thickness0, wedge1_eps_r1, wedge1_sigma_e1, wedge1_mu_r1,      \
        wedge1_gain1, wedge1_thickness1, wedge2_eps_r0, wedge2_sigma_e0,      \
        wedge2_mu_r0, wedge2_gain0, wedge2_thickness0, wedge2_eps_r1,         \
        wedge2_sigma_e1, wedge2_mu_r1, wedge2_gain1, wedge2_thickness1,       \
        edge1_line_min, edge1_line_max, edge2_line_min, edge2_line_max,       \
        frequency_hz

__global__ void coupled_dd_backward_kernel(
    int64_t count,
    COUPLED_DD_ROW_PARAMS,
    const c10::complex<float>* grad_field_vector,
    const c10::complex<float>* grad_coefficient,
    const c10::complex<float>* grad_path_field,
    const float* grad_path_gain,
    float* grad_source,
    float* grad_target,
    float* grad_eps_r,        // (N, 4): wedge1 face0/face1, wedge2 face0/face1
    float* grad_sigma_e,
    float* grad_gain,
    float* grad_thickness,
    float* grad_frequency) {
    for (int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < count;
         index += static_cast<int64_t>(blockDim.x) * gridDim.x) {
        const CoupledDdRowInputs in = load_coupled_dd_row(COUPLED_DD_ROW_ARGS(index));
        CoupledDdSeeds seeds = coupled_dd_seeds_zero();
        const int64_t base = index * 3;
        const int64_t mbase = index * 4;

        struct VectorSlot {
            field::float3a* seed;
            float* grad;
        };
        VectorSlot vector_slots[2] = {
            {&seeds.source, grad_source},
            {&seeds.target, grad_target},
        };
        for (int slot = 0; slot < 2; ++slot) {
            if (vector_slots[slot].grad == nullptr)
                continue;
            float* axes[3] = {
                &vector_slots[slot].seed->x,
                &vector_slots[slot].seed->y,
                &vector_slots[slot].seed->z};
            for (int axis = 0; axis < 3; ++axis) {
                *axes[axis] = 1.f;
                const CoupledRowTangents t = coupled_dd_row_dual(in, seeds);
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
        MaterialSlot material_slots[16] = {
            {&seeds.w1a_eps, grad_eps_r, 0},
            {&seeds.w1b_eps, grad_eps_r, 1},
            {&seeds.w2a_eps, grad_eps_r, 2},
            {&seeds.w2b_eps, grad_eps_r, 3},
            {&seeds.w1a_sigma, grad_sigma_e, 0},
            {&seeds.w1b_sigma, grad_sigma_e, 1},
            {&seeds.w2a_sigma, grad_sigma_e, 2},
            {&seeds.w2b_sigma, grad_sigma_e, 3},
            {&seeds.w1a_gain, grad_gain, 0},
            {&seeds.w1b_gain, grad_gain, 1},
            {&seeds.w2a_gain, grad_gain, 2},
            {&seeds.w2b_gain, grad_gain, 3},
            {&seeds.w1a_thickness, grad_thickness, 0},
            {&seeds.w1b_thickness, grad_thickness, 1},
            {&seeds.w2a_thickness, grad_thickness, 2},
            {&seeds.w2b_thickness, grad_thickness, 3},
        };
        for (int slot = 0; slot < 16; ++slot) {
            if (material_slots[slot].grad == nullptr)
                continue;
            *material_slots[slot].seed = 1.f;
            const CoupledRowTangents t = coupled_dd_row_dual(in, seeds);
            *material_slots[slot].seed = 0.f;
            material_slots[slot].grad[mbase + material_slots[slot].column] =
                coupled_contract(
                    index, grad_field_vector, grad_coefficient, grad_path_field,
                    grad_path_gain, t);
        }
        if (grad_frequency != nullptr) {
            seeds.frequency = 1.f;
            const CoupledRowTangents t = coupled_dd_row_dual(in, seeds);
            seeds.frequency = 0.f;
            grad_frequency[index] = coupled_contract(
                index, grad_field_vector, grad_coefficient, grad_path_field,
                grad_path_gain, t);
        }
    }
}

__global__ void coupled_dd_jvp_kernel(
    int64_t count,
    COUPLED_DD_ROW_PARAMS,
    const float* tangent_source,
    const float* tangent_target,
    const float* tangent_eps_r,     // (N, 4): wedge1 face0/face1, wedge2 face0/face1
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
        const CoupledDdRowInputs in = load_coupled_dd_row(COUPLED_DD_ROW_ARGS(index));
        CoupledDdSeeds seeds = coupled_dd_seeds_zero();
        const int64_t base = index * 3;
        const int64_t mbase = index * 4;
        if (tangent_source != nullptr)
            seeds.source = load3f(tangent_source, index);
        if (tangent_target != nullptr)
            seeds.target = load3f(tangent_target, index);
        if (tangent_eps_r != nullptr) {
            seeds.w1a_eps = tangent_eps_r[mbase];
            seeds.w1b_eps = tangent_eps_r[mbase + 1];
            seeds.w2a_eps = tangent_eps_r[mbase + 2];
            seeds.w2b_eps = tangent_eps_r[mbase + 3];
        }
        if (tangent_sigma_e != nullptr) {
            seeds.w1a_sigma = tangent_sigma_e[mbase];
            seeds.w1b_sigma = tangent_sigma_e[mbase + 1];
            seeds.w2a_sigma = tangent_sigma_e[mbase + 2];
            seeds.w2b_sigma = tangent_sigma_e[mbase + 3];
        }
        if (tangent_gain != nullptr) {
            seeds.w1a_gain = tangent_gain[mbase];
            seeds.w1b_gain = tangent_gain[mbase + 1];
            seeds.w2a_gain = tangent_gain[mbase + 2];
            seeds.w2b_gain = tangent_gain[mbase + 3];
        }
        if (tangent_thickness != nullptr) {
            seeds.w1a_thickness = tangent_thickness[mbase];
            seeds.w1b_thickness = tangent_thickness[mbase + 1];
            seeds.w2a_thickness = tangent_thickness[mbase + 2];
            seeds.w2b_thickness = tangent_thickness[mbase + 3];
        }
        seeds.frequency = tangent_frequency;
        const CoupledRowTangents t = coupled_dd_row_dual(in, seeds);
        tangent_field_vector[base] = to_c10(t.field_vector.x);
        tangent_field_vector[base + 1] = to_c10(t.field_vector.y);
        tangent_field_vector[base + 2] = to_c10(t.field_vector.z);
        tangent_coefficient[index] = to_c10(t.coefficient);
        tangent_path_field[index] = to_c10(t.path_field);
        tangent_path_gain[index] = t.path_gain;
    }
}


}  // namespace

namespace {

void check_coupled_primal_rows(
    const std::vector<std::pair<at::Tensor, const char*>>& vectors,
    const std::vector<std::pair<at::Tensor, const char*>>& scalars,
    double frequency_hz) {
    using channel::check_flat_tensor;
    using channel::check_vec3_table;
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

// ---------------------------------------------------------------------------
// Interaction chain field host entry points: primal, VJP and JVP share one
// argument list and one validated ChainArgs view.
// ---------------------------------------------------------------------------

#define CHAIN_HOST_PARAMS                                                                              \
    at::Tensor source, at::Tensor target, at::Tensor positions, at::Tensor normals,                    \
        at::Tensor component_id, at::Tensor depth, at::Tensor edge_direction, at::Tensor edge_n0,      \
        at::Tensor edge_n1, at::Tensor exterior_angle, at::Tensor edge_line_min,                       \
        at::Tensor edge_line_max, at::Tensor tx_power, at::Tensor tx_polarization,                     \
        at::Tensor rx_polarization, at::Tensor eps_r, at::Tensor sigma_e, at::Tensor mu_r,             \
        at::Tensor gain, at::Tensor thickness, at::Tensor bounce_material, at::Tensor layer_offset,    \
        at::Tensor layer_count, at::Tensor layer_thickness_m, at::Tensor layer_eps_r,                  \
        at::Tensor layer_sigma_e, at::Tensor layer_mu_r, double frequency_hz

#define CHAIN_HOST_ARGS                                                                                \
    source, target, positions, normals, component_id, depth, edge_direction, edge_n0, edge_n1,         \
        exterior_angle, edge_line_min, edge_line_max, tx_power, tx_polarization, rx_polarization,      \
        eps_r, sigma_e, mu_r, gain, thickness, bounce_material, layer_offset, layer_count,             \
        layer_thickness_m, layer_eps_r, layer_sigma_e, layer_mu_r, frequency_hz

namespace {

ChainArgs chain_args(CHAIN_HOST_PARAMS) {
    using channel::check_flat_tensor;
    using channel::check_tensor;
    using channel::check_vec3_table;
    const int64_t count = source.size(0);
    for (const auto& named : std::vector<std::pair<at::Tensor, const char*>>{
             {source, "source"},
             {target, "target"},
             {edge_direction, "edge_direction"},
             {edge_n0, "edge_n0"},
             {edge_n1, "edge_n1"},
             {tx_polarization, "tx_polarization"},
             {rx_polarization, "rx_polarization"}}) {
        check_vec3_table(named.first, named.second);
        TORCH_CHECK(named.first.size(0) == count, named.second, " must match source rows");
    }
    for (const auto& named : std::vector<std::pair<at::Tensor, const char*>>{
             {exterior_angle, "exterior_angle"},
             {edge_line_min, "edge_line_min"},
             {edge_line_max, "edge_line_max"},
             {tx_power, "tx_power"}}) {
        check_flat_tensor(named.first, named.second, at::kFloat);
        TORCH_CHECK(named.first.size(0) == count, named.second, " must match source rows");
    }
    for (const auto& named : std::vector<std::pair<at::Tensor, const char*>>{
             {component_id, "component_id"}, {depth, "depth"}}) {
        check_flat_tensor(named.first, named.second, at::kInt);
        TORCH_CHECK(named.first.size(0) == count, named.second, " must match source rows");
    }
    for (const auto& named : std::vector<std::pair<at::Tensor, const char*>>{
             {positions, "positions"}, {normals, "normals"}}) {
        check_tensor(named.first, named.second, at::kFloat, 3);
        TORCH_CHECK(named.first.size(0) == count && named.first.size(2) == 3,
                    named.second, " must have shape (N, W, 3)");
    }
    const int64_t width = positions.size(1);
    TORCH_CHECK(normals.size(1) == width, "normals must match positions");
    TORCH_CHECK(width >= 2 && width <= kChainMaxDepth, "chain sequence width must be in [2, ", kChainMaxDepth, "]");
    for (const auto& named : std::vector<std::pair<at::Tensor, const char*>>{
             {eps_r, "eps_r"}, {sigma_e, "sigma_e"}, {mu_r, "mu_r"}, {gain, "gain"}, {thickness, "thickness"}}) {
        check_tensor(named.first, named.second, at::kFloat, 2);
        TORCH_CHECK(named.first.size(0) == count && named.first.size(1) == width + 2,
                    named.second, " must have shape (N, W + 2)");
    }
    check_tensor(bounce_material, "bounce_material", at::kInt, 2);
    TORCH_CHECK(bounce_material.size(0) == count && bounce_material.size(1) == width + 2,
                "bounce_material must have shape (N, W + 2)");
    check_tensor(layer_offset, "layer_offset", at::kInt, 1);
    check_tensor(layer_count, "layer_count", at::kInt, 1);
    TORCH_CHECK(layer_count.size(0) == layer_offset.size(0), "layer_count must match layer_offset");
    for (const auto& tensor : {layer_thickness_m, layer_eps_r, layer_sigma_e, layer_mu_r}) {
        check_tensor(tensor, "layer table", at::kFloat, 1);
        TORCH_CHECK(tensor.size(0) == layer_thickness_m.size(0), "layer tables must share one length");
    }
    TORCH_CHECK(frequency_hz > 0.0, "frequency_hz must be positive");
    ChainArgs args;
    args.source = source.data_ptr<float>();
    args.target = target.data_ptr<float>();
    args.positions = positions.data_ptr<float>();
    args.normals = normals.data_ptr<float>();
    args.component_id = component_id.data_ptr<int>();
    args.depth = depth.data_ptr<int>();
    args.width = width;
    args.edge_direction = edge_direction.data_ptr<float>();
    args.edge_n0 = edge_n0.data_ptr<float>();
    args.edge_n1 = edge_n1.data_ptr<float>();
    args.exterior_angle = exterior_angle.data_ptr<float>();
    args.edge_line_min = edge_line_min.data_ptr<float>();
    args.edge_line_max = edge_line_max.data_ptr<float>();
    args.tx_power = tx_power.data_ptr<float>();
    args.tx_polarization = tx_polarization.data_ptr<float>();
    args.rx_polarization = rx_polarization.data_ptr<float>();
    args.eps_r = eps_r.data_ptr<float>();
    args.sigma_e = sigma_e.data_ptr<float>();
    args.mu_r = mu_r.data_ptr<float>();
    args.gain = gain.data_ptr<float>();
    args.thickness = thickness.data_ptr<float>();
    args.bounce_material = bounce_material.data_ptr<int>();
    args.layers.offset = layer_offset.data_ptr<int>();
    args.layers.count = layer_count.data_ptr<int>();
    args.layers.thickness_m = layer_thickness_m.data_ptr<float>();
    args.layers.eps_r = layer_eps_r.data_ptr<float>();
    args.layers.sigma_e = layer_sigma_e.data_ptr<float>();
    args.layers.mu_r = layer_mu_r.data_ptr<float>();
    args.layers.material_count = layer_count.size(0);
    args.frequency = static_cast<float>(frequency_hz);
    return args;
}

}  // namespace

pybind11::dict channel_field_coupled_chain(CHAIN_HOST_PARAMS) {
    const ChainArgs args = chain_args(CHAIN_HOST_ARGS);
    const int64_t count = source.size(0);
    auto complex_options = source.options().dtype(at::kComplexFloat);
    auto field_vector = at::empty({count, 3}, complex_options);
    auto coefficient = at::empty({count}, complex_options);
    auto path_field = at::empty({count}, complex_options);
    auto path_gain = at::empty({count}, source.options());
    auto direction = at::empty({count, 3}, source.options());
    if (count > 0) {
        cudaStream_t stream = at::cuda::getCurrentCUDAStream(source.get_device()).stream();
        coupled_chain_field_kernel<<<::channel::launch_blocks(count, kBlockSize),
                                     kBlockSize, 0, stream>>>(
            count, args, field_vector.data_ptr<c10::complex<float>>(),
            coefficient.data_ptr<c10::complex<float>>(), path_field.data_ptr<c10::complex<float>>(),
            path_gain.data_ptr<float>(), direction.data_ptr<float>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    pybind11::dict out;
    out["field_vector"] = field_vector;
    out["coefficient"] = coefficient;
    out["path_field"] = path_field;
    out["path_gain"] = path_gain;
    out["direction"] = direction;
    return out;
}

pybind11::dict channel_field_coupled_chain_backward(
    CHAIN_HOST_PARAMS,
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
    const ChainArgs args = chain_args(CHAIN_HOST_ARGS);
    const int64_t count = source.size(0);
    at::Tensor grad_storage[4];
    const at::Tensor* g_field = optional_tensor_arg(
        std::move(grad_field_vector), grad_storage[0], "grad_field_vector", at::kComplexFloat, {count, 3}, source);
    const at::Tensor* g_coefficient = optional_tensor_arg(
        std::move(grad_coefficient), grad_storage[1], "grad_coefficient", at::kComplexFloat, {count}, source);
    const at::Tensor* g_path_field = optional_tensor_arg(
        std::move(grad_path_field), grad_storage[2], "grad_path_field", at::kComplexFloat, {count}, source);
    const at::Tensor* g_path_gain = optional_tensor_arg(
        std::move(grad_path_gain), grad_storage[3], "grad_path_gain", at::kFloat, {count}, source);
    const auto options = source.options();
    const int64_t columns = args.width + 2;
    at::Tensor grads[6];
    at::Tensor* grad_ptrs[6] = {nullptr, nullptr, nullptr, nullptr, nullptr, nullptr};
    if (need_grad_geometry) {
        grads[0] = at::empty({count, 3}, options);
        grads[1] = at::empty({count, 3}, options);
        grad_ptrs[0] = &grads[0];
        grad_ptrs[1] = &grads[1];
    }
    const bool need_material[4] = {need_grad_eps_r, need_grad_sigma_e, need_grad_gain, need_grad_thickness};
    for (int parameter = 0; parameter < 4; ++parameter) {
        if (need_material[parameter]) {
            grads[2 + parameter] = at::empty({count, columns}, options);
            grad_ptrs[2 + parameter] = &grads[2 + parameter];
        }
    }
    // Per-row frequency cotangents, summed after the kernel in a fixed order.
    at::Tensor grad_frequency;
    at::Tensor row_frequency;
    if (need_grad_frequency) {
        grad_frequency = zero_scalar(options);
        row_frequency = at::empty({count}, options);
    }
    const bool any_grad =
        g_field != nullptr || g_coefficient != nullptr || g_path_field != nullptr || g_path_gain != nullptr;
    if (count > 0 && any_grad) {
        cudaStream_t stream = at::cuda::getCurrentCUDAStream(source.get_device()).stream();
        coupled_chain_backward_kernel<<<::channel::launch_blocks(count, kBlockSize),
                                        kBlockSize, 0, stream>>>(
            count, args, opt_ptr<c10::complex<float>>(g_field), opt_ptr<c10::complex<float>>(g_coefficient),
            opt_ptr<c10::complex<float>>(g_path_field), opt_ptr<float>(g_path_gain), opt_mut_ptr<float>(grad_ptrs[0]),
            opt_mut_ptr<float>(grad_ptrs[1]), opt_mut_ptr<float>(grad_ptrs[2]), opt_mut_ptr<float>(grad_ptrs[3]),
            opt_mut_ptr<float>(grad_ptrs[4]), opt_mut_ptr<float>(grad_ptrs[5]),
            need_grad_frequency ? row_frequency.data_ptr<float>() : nullptr);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        if (need_grad_frequency)
            channel::ordered_sum::add_sum<float>(
                row_frequency.data_ptr<float>(), count, grad_frequency.data_ptr<float>(), options, stream);
    } else {
        for (at::Tensor* tensor : grad_ptrs) {
            if (tensor != nullptr)
                tensor->zero_();
        }
    }
    auto pack = [](at::Tensor* tensor) {
        return tensor != nullptr ? pybind11::cast(*tensor) : pybind11::object(pybind11::none());
    };
    pybind11::dict out;
    out["grad_source"] = pack(grad_ptrs[0]);
    out["grad_target"] = pack(grad_ptrs[1]);
    out["grad_eps_r"] = pack(grad_ptrs[2]);
    out["grad_sigma_e"] = pack(grad_ptrs[3]);
    out["grad_gain"] = pack(grad_ptrs[4]);
    out["grad_thickness"] = pack(grad_ptrs[5]);
    out["grad_frequency"] = need_grad_frequency ? pybind11::cast(grad_frequency) : pybind11::object(pybind11::none());
    return out;
}

pybind11::dict channel_field_coupled_chain_jvp(
    CHAIN_HOST_PARAMS,
    pybind11::object tangent_source,
    pybind11::object tangent_target,
    pybind11::object tangent_eps_r,
    pybind11::object tangent_sigma_e,
    pybind11::object tangent_gain,
    pybind11::object tangent_thickness,
    double tangent_frequency) {
    const ChainArgs args = chain_args(CHAIN_HOST_ARGS);
    const int64_t count = source.size(0);
    const int64_t columns = args.width + 2;
    at::Tensor storage[6];
    const at::Tensor* t_source =
        optional_tensor_arg(std::move(tangent_source), storage[0], "tangent_source", at::kFloat, {count, 3}, source);
    const at::Tensor* t_target =
        optional_tensor_arg(std::move(tangent_target), storage[1], "tangent_target", at::kFloat, {count, 3}, source);
    const at::Tensor* t_eps = optional_tensor_arg(
        std::move(tangent_eps_r), storage[2], "tangent_eps_r", at::kFloat, {count, columns}, source);
    const at::Tensor* t_sigma = optional_tensor_arg(
        std::move(tangent_sigma_e), storage[3], "tangent_sigma_e", at::kFloat, {count, columns}, source);
    const at::Tensor* t_gain = optional_tensor_arg(
        std::move(tangent_gain), storage[4], "tangent_gain", at::kFloat, {count, columns}, source);
    const at::Tensor* t_thickness = optional_tensor_arg(
        std::move(tangent_thickness), storage[5], "tangent_thickness", at::kFloat, {count, columns}, source);
    auto tangent_field_vector = at::empty({count, 3}, source.options().dtype(at::kComplexFloat));
    auto tangent_coefficient = at::empty({count}, source.options().dtype(at::kComplexFloat));
    auto tangent_path_field = at::empty({count}, source.options().dtype(at::kComplexFloat));
    auto tangent_path_gain = at::empty({count}, source.options());
    if (count > 0) {
        cudaStream_t stream = at::cuda::getCurrentCUDAStream(source.get_device()).stream();
        coupled_chain_jvp_kernel<<<::channel::launch_blocks(count, kBlockSize),
                                   kBlockSize, 0, stream>>>(
            count, args, opt_ptr<float>(t_source), opt_ptr<float>(t_target), opt_ptr<float>(t_eps),
            opt_ptr<float>(t_sigma), opt_ptr<float>(t_gain), opt_ptr<float>(t_thickness),
            static_cast<float>(tangent_frequency), tangent_field_vector.data_ptr<c10::complex<float>>(),
            tangent_coefficient.data_ptr<c10::complex<float>>(), tangent_path_field.data_ptr<c10::complex<float>>(),
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

// ---------------------------------------------------------------------------
// coupled double diffraction: coupled double-diffraction AD host companions. Twins of the RD
// companions above (same output schema, same coupled_contract), driving the
// coupled_dd_row_dual mirror. Q1/Q2/bounds/axes/normals/polarizations are
// frozen (loaded as dual constants); tx/rx flow through live re-anchoring, the
// four wedge faces carry material gradients, and frequency is differentiable.
// ---------------------------------------------------------------------------

#define COUPLED_DD_HOST_PARAMS                                                 \
    at::Tensor source, at::Tensor target, at::Tensor edge1_position,          \
        at::Tensor edge1_direction, at::Tensor edge1_n0, at::Tensor edge1_n1, \
        at::Tensor edge1_exterior, at::Tensor edge2_position,                 \
        at::Tensor edge2_direction, at::Tensor edge2_n0, at::Tensor edge2_n1, \
        at::Tensor edge2_exterior, at::Tensor tx_power,                       \
        at::Tensor tx_polarization, at::Tensor rx_polarization,               \
        at::Tensor wedge1_eps_r0, at::Tensor wedge1_sigma_e0,                 \
        at::Tensor wedge1_mu_r0, at::Tensor wedge1_gain0,                     \
        at::Tensor wedge1_thickness0, at::Tensor wedge1_eps_r1,               \
        at::Tensor wedge1_sigma_e1, at::Tensor wedge1_mu_r1,                  \
        at::Tensor wedge1_gain1, at::Tensor wedge1_thickness1,               \
        at::Tensor wedge2_eps_r0, at::Tensor wedge2_sigma_e0,                 \
        at::Tensor wedge2_mu_r0, at::Tensor wedge2_gain0,                     \
        at::Tensor wedge2_thickness0, at::Tensor wedge2_eps_r1,               \
        at::Tensor wedge2_sigma_e1, at::Tensor wedge2_mu_r1,                  \
        at::Tensor wedge2_gain1, at::Tensor wedge2_thickness1,               \
        at::Tensor edge1_line_min, at::Tensor edge1_line_max,                 \
        at::Tensor edge2_line_min, at::Tensor edge2_line_max,                 \
        double frequency_hz

#define COUPLED_DD_HOST_ARGS                                                   \
    source.data_ptr<float>(), target.data_ptr<float>(),                       \
        edge1_position.data_ptr<float>(), edge1_direction.data_ptr<float>(),  \
        edge1_n0.data_ptr<float>(), edge1_n1.data_ptr<float>(),               \
        edge1_exterior.data_ptr<float>(), edge2_position.data_ptr<float>(),   \
        edge2_direction.data_ptr<float>(), edge2_n0.data_ptr<float>(),        \
        edge2_n1.data_ptr<float>(), edge2_exterior.data_ptr<float>(),         \
        tx_power.data_ptr<float>(), tx_polarization.data_ptr<float>(),        \
        rx_polarization.data_ptr<float>(), wedge1_eps_r0.data_ptr<float>(),   \
        wedge1_sigma_e0.data_ptr<float>(), wedge1_mu_r0.data_ptr<float>(),    \
        wedge1_gain0.data_ptr<float>(), wedge1_thickness0.data_ptr<float>(),  \
        wedge1_eps_r1.data_ptr<float>(), wedge1_sigma_e1.data_ptr<float>(),   \
        wedge1_mu_r1.data_ptr<float>(), wedge1_gain1.data_ptr<float>(),       \
        wedge1_thickness1.data_ptr<float>(), wedge2_eps_r0.data_ptr<float>(), \
        wedge2_sigma_e0.data_ptr<float>(), wedge2_mu_r0.data_ptr<float>(),    \
        wedge2_gain0.data_ptr<float>(), wedge2_thickness0.data_ptr<float>(),  \
        wedge2_eps_r1.data_ptr<float>(), wedge2_sigma_e1.data_ptr<float>(),   \
        wedge2_mu_r1.data_ptr<float>(), wedge2_gain1.data_ptr<float>(),       \
        wedge2_thickness1.data_ptr<float>(), edge1_line_min.data_ptr<float>(),\
        edge1_line_max.data_ptr<float>(), edge2_line_min.data_ptr<float>(),   \
        edge2_line_max.data_ptr<float>(), static_cast<float>(frequency_hz)

#define COUPLED_DD_CHECK_LISTS                                                 \
    check_coupled_primal_rows(                                                 \
        {{source, "source"},                                                  \
         {target, "target"},                                                  \
         {edge1_position, "edge1_position"},                                  \
         {edge1_direction, "edge1_direction"},                                \
         {edge1_n0, "edge1_n0"},                                              \
         {edge1_n1, "edge1_n1"},                                              \
         {edge2_position, "edge2_position"},                                  \
         {edge2_direction, "edge2_direction"},                                \
         {edge2_n0, "edge2_n0"},                                              \
         {edge2_n1, "edge2_n1"},                                              \
         {tx_polarization, "tx_polarization"},                                \
         {rx_polarization, "rx_polarization"}},                               \
        {{edge1_exterior, "edge1_exterior"},                                  \
         {edge2_exterior, "edge2_exterior"},                                  \
         {tx_power, "tx_power"},                                              \
         {wedge1_eps_r0, "wedge1_eps_r0"},                                    \
         {wedge1_sigma_e0, "wedge1_sigma_e0"},                                \
         {wedge1_mu_r0, "wedge1_mu_r0"},                                      \
         {wedge1_gain0, "wedge1_gain0"},                                      \
         {wedge1_thickness0, "wedge1_thickness0"},                            \
         {wedge1_eps_r1, "wedge1_eps_r1"},                                    \
         {wedge1_sigma_e1, "wedge1_sigma_e1"},                                \
         {wedge1_mu_r1, "wedge1_mu_r1"},                                      \
         {wedge1_gain1, "wedge1_gain1"},                                      \
         {wedge1_thickness1, "wedge1_thickness1"},                            \
         {wedge2_eps_r0, "wedge2_eps_r0"},                                    \
         {wedge2_sigma_e0, "wedge2_sigma_e0"},                                \
         {wedge2_mu_r0, "wedge2_mu_r0"},                                      \
         {wedge2_gain0, "wedge2_gain0"},                                      \
         {wedge2_thickness0, "wedge2_thickness0"},                            \
         {wedge2_eps_r1, "wedge2_eps_r1"},                                    \
         {wedge2_sigma_e1, "wedge2_sigma_e1"},                                \
         {wedge2_mu_r1, "wedge2_mu_r1"},                                      \
         {wedge2_gain1, "wedge2_gain1"},                                      \
         {wedge2_thickness1, "wedge2_thickness1"},                            \
         {edge1_line_min, "edge1_line_min"},                                  \
         {edge1_line_max, "edge1_line_max"},                                  \
         {edge2_line_min, "edge2_line_min"},                                  \
         {edge2_line_max, "edge2_line_max"}},                                 \
        frequency_hz)

pybind11::dict channel_field_coupled_dd_backward(
    COUPLED_DD_HOST_PARAMS,
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
    COUPLED_DD_CHECK_LISTS;
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
    at::Tensor grad_source, grad_target;
    at::Tensor grad_eps, grad_sigma, grad_gain, grad_thickness, grad_frequency;
    at::Tensor* grad_source_ptr = nullptr;
    at::Tensor* grad_target_ptr = nullptr;
    at::Tensor* grad_eps_ptr = nullptr;
    at::Tensor* grad_sigma_ptr = nullptr;
    at::Tensor* grad_gain_ptr = nullptr;
    at::Tensor* grad_thickness_ptr = nullptr;
    at::Tensor* grad_frequency_ptr = nullptr;
    if (need_grad_geometry) {
        grad_source = at::empty({count, 3}, options);
        grad_target = at::empty({count, 3}, options);
        grad_source_ptr = &grad_source;
        grad_target_ptr = &grad_target;
    }
    if (need_grad_eps_r) {
        grad_eps = at::empty({count, 4}, options);
        grad_eps_ptr = &grad_eps;
    }
    if (need_grad_sigma_e) {
        grad_sigma = at::empty({count, 4}, options);
        grad_sigma_ptr = &grad_sigma;
    }
    if (need_grad_gain) {
        grad_gain = at::empty({count, 4}, options);
        grad_gain_ptr = &grad_gain;
    }
    if (need_grad_thickness) {
        grad_thickness = at::empty({count, 4}, options);
        grad_thickness_ptr = &grad_thickness;
    }
    // Per-row frequency cotangents (every row writes one), summed after the kernel in a
    // fixed order.
    at::Tensor row_frequency;
    at::Tensor* row_frequency_ptr = nullptr;
    if (need_grad_frequency) {
        grad_frequency = zero_scalar(options);
        grad_frequency_ptr = &grad_frequency;
        row_frequency = at::empty({count}, options);
        row_frequency_ptr = &row_frequency;
    }
    const bool any_grad = g_field != nullptr || g_coefficient != nullptr ||
                          g_path_field != nullptr || g_path_gain != nullptr;
    if (count > 0 && any_grad) {
        cudaStream_t stream = at::cuda::getCurrentCUDAStream(source.get_device()).stream();
        coupled_dd_backward_kernel<<<::channel::launch_blocks(count, kBlockSize),
                                     kBlockSize, 0, stream>>>(
            count,
            COUPLED_DD_HOST_ARGS,
            opt_ptr<c10::complex<float>>(g_field),
            opt_ptr<c10::complex<float>>(g_coefficient),
            opt_ptr<c10::complex<float>>(g_path_field),
            opt_ptr<float>(g_path_gain),
            opt_mut_ptr<float>(grad_source_ptr),
            opt_mut_ptr<float>(grad_target_ptr),
            opt_mut_ptr<float>(grad_eps_ptr),
            opt_mut_ptr<float>(grad_sigma_ptr),
            opt_mut_ptr<float>(grad_gain_ptr),
            opt_mut_ptr<float>(grad_thickness_ptr),
            opt_mut_ptr<float>(row_frequency_ptr));
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        if (row_frequency_ptr != nullptr)
            channel::ordered_sum::add_sum<float>(
                row_frequency.data_ptr<float>(), count, grad_frequency.data_ptr<float>(),
                options, stream);
    } else {
        for (at::Tensor* tensor :
             {grad_source_ptr, grad_target_ptr, grad_eps_ptr, grad_sigma_ptr,
              grad_gain_ptr, grad_thickness_ptr}) {
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
    out["grad_eps_r"] = pack(grad_eps_ptr, grad_eps);
    out["grad_sigma_e"] = pack(grad_sigma_ptr, grad_sigma);
    out["grad_gain"] = pack(grad_gain_ptr, grad_gain);
    out["grad_thickness"] = pack(grad_thickness_ptr, grad_thickness);
    out["grad_frequency"] = pack(grad_frequency_ptr, grad_frequency);
    return out;
}

pybind11::dict channel_field_coupled_dd_jvp(
    COUPLED_DD_HOST_PARAMS,
    pybind11::object tangent_source,
    pybind11::object tangent_target,
    pybind11::object tangent_eps_r,
    pybind11::object tangent_sigma_e,
    pybind11::object tangent_gain,
    pybind11::object tangent_thickness,
    double tangent_frequency) {
    COUPLED_DD_CHECK_LISTS;
    const int64_t count = source.size(0);
    at::Tensor storage[6];
    const at::Tensor* t_source = optional_tensor_arg(
        std::move(tangent_source), storage[0], "tangent_source", at::kFloat,
        {count, 3}, source);
    const at::Tensor* t_target = optional_tensor_arg(
        std::move(tangent_target), storage[1], "tangent_target", at::kFloat,
        {count, 3}, source);
    const at::Tensor* t_eps = optional_tensor_arg(
        std::move(tangent_eps_r), storage[2], "tangent_eps_r", at::kFloat,
        {count, 4}, source);
    const at::Tensor* t_sigma = optional_tensor_arg(
        std::move(tangent_sigma_e), storage[3], "tangent_sigma_e", at::kFloat,
        {count, 4}, source);
    const at::Tensor* t_gain = optional_tensor_arg(
        std::move(tangent_gain), storage[4], "tangent_gain", at::kFloat,
        {count, 4}, source);
    const at::Tensor* t_thickness = optional_tensor_arg(
        std::move(tangent_thickness), storage[5], "tangent_thickness",
        at::kFloat, {count, 4}, source);
    auto tangent_field_vector = at::empty({count, 3}, source.options().dtype(at::kComplexFloat));
    auto tangent_coefficient = at::empty({count}, source.options().dtype(at::kComplexFloat));
    auto tangent_path_field = at::empty({count}, source.options().dtype(at::kComplexFloat));
    auto tangent_path_gain = at::empty({count}, source.options());
    if (count > 0) {
        cudaStream_t stream = at::cuda::getCurrentCUDAStream(source.get_device()).stream();
        coupled_dd_jvp_kernel<<<::channel::launch_blocks(count, kBlockSize),
                                kBlockSize, 0, stream>>>(
            count,
            COUPLED_DD_HOST_ARGS,
            opt_ptr<float>(t_source),
            opt_ptr<float>(t_target),
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

// ==== Section: Wedge AD projection ====


namespace {

// ---------------------------------------------------------------------------
// field_project_complex3 companions: coefficient = <field, axis(direction)>
// with axis = project_to_wedge_plane(rx_pol, direction) ( unnormalized
// transverse of p_rx); path_gain = |coeff|^2.
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
        // coefficient = p_rx . E via the unnormalized transverse of p_rx.
        const field::float3a axis = field::project_to_wedge_plane(pol, dir);
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
            ad::adj_transverse_project(dir, pol, g_axis, g_dir, g_pol);
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
        const ad::DualF3 axis = ad::dual_transverse_project(
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


}  // namespace

pybind11::dict channel_field_project_complex3_backward(
    at::Tensor field_vector,
    at::Tensor direction,
    at::Tensor rx_polarization,
    pybind11::object grad_coefficient,
    pybind11::object grad_path_gain,
    bool need_grad_field_vector,
    bool need_grad_direction) {
    using channel::check_tensor;
    using channel::check_vec3_table;
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
        project_complex3_backward_kernel<<<::channel::launch_blocks(count, kBlockSize),
                                           kBlockSize, 0, stream>>>(
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

pybind11::dict channel_field_project_complex3_jvp(
    at::Tensor field_vector,
    at::Tensor direction,
    at::Tensor rx_polarization,
    pybind11::object tangent_field_vector,
    pybind11::object tangent_direction) {
    using channel::check_tensor;
    using channel::check_vec3_table;
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
        project_complex3_jvp_kernel<<<::channel::launch_blocks(count, kBlockSize),
                                      kBlockSize, 0, stream>>>(
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

// ==== Section: Coupled topology ====
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime_api.h>
#include "math.cuh"
#include <rayd/utd.h>
#include "torch_cuda.h"

#include "../tensor_checks.h"

#include <cfloat>
#include <cmath>
#include <vector>


namespace {

constexpr int kCoupledTopologyBlockSize = 256;
constexpr float kGeometryEpsilon = 1.0e-6f;
using channel::kSpeedOfLightF;
namespace utd = rayd::shared::diffraction;



__device__ __forceinline__ utd::float3a to_utd(float3 value) {
    return utd::make_f3(value.x, value.y, value.z);
}

// Interaction chain rows from the two traced halves. Slots 0..a-1 hold the transmitter-side
// reflections in path order, slot a the diffraction point, and slots a+1..a+b the
// receiver-side reflections (traced from the receiver, so written in reverse).
__global__ void coupled_chain_finalize_kernel(
    int64_t count,
    int64_t prefix_depth,
    int64_t suffix_depth,
    const bool *__restrict__ prefix_valid,
    const bool *__restrict__ suffix_valid,
    const float *__restrict__ prefix_length,
    const float *__restrict__ suffix_length,
    const int *__restrict__ prefix_prims,
    const int *__restrict__ suffix_prims,
    const float *__restrict__ prefix_hits,
    const float *__restrict__ suffix_hits,
    const float *__restrict__ prefix_normals,
    const float *__restrict__ suffix_normals,
    const float *__restrict__ source,
    const float *__restrict__ receiver,
    const float *__restrict__ edge_point,
    const int *__restrict__ edge_id,
    float min_segment_length,
    int *__restrict__ primitive_sequence,
    float *__restrict__ interaction_positions,
    float *__restrict__ interaction_normals,
    float *__restrict__ path_length,
    float *__restrict__ delay,
    bool *__restrict__ valid_out) {
    const int64_t depth = prefix_depth + suffix_depth + 1;
    for (int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < count;
         index += static_cast<int64_t>(blockDim.x) * gridDim.x) {
        const int64_t base = index * depth;
        const float3 point = channel::math::load_vec3(edge_point, index);
        for (int64_t i = 0; i < prefix_depth; ++i) {
            primitive_sequence[base + i] = prefix_prims[index * prefix_depth + i];
            channel::math::store_vec3(
                interaction_positions, base + i, channel::math::load_vec3(prefix_hits, index * prefix_depth + i));
            channel::math::store_vec3(
                interaction_normals, base + i, channel::math::load_vec3(prefix_normals, index * prefix_depth + i));
        }
        primitive_sequence[base + prefix_depth] = edge_id[index];
        channel::math::store_vec3(interaction_positions, base + prefix_depth, point);
        // A diffraction edge has an axis and two face normals, not one surface normal.
        channel::math::store_vec3(interaction_normals, base + prefix_depth, make_float3(NAN, NAN, NAN));
        for (int64_t j = 0; j < suffix_depth; ++j) {
            const int64_t slot = depth - 1 - j;
            primitive_sequence[base + slot] = suffix_prims[index * suffix_depth + j];
            channel::math::store_vec3(
                interaction_positions, base + slot, channel::math::load_vec3(suffix_hits, index * suffix_depth + j));
            channel::math::store_vec3(
                interaction_normals, base + slot, channel::math::load_vec3(suffix_normals, index * suffix_depth + j));
        }
        const float prefix = prefix_depth > 0
                                 ? prefix_length[index]
                                 : channel::math::length(channel::math::sub(point, channel::math::load_vec3(source, index)));
        const float suffix = suffix_depth > 0
                                 ? suffix_length[index]
                                 : channel::math::length(channel::math::sub(point, channel::math::load_vec3(receiver, index)));
        const bool segments = (prefix_depth > 0 || prefix >= min_segment_length) &&
                              (suffix_depth > 0 || suffix >= min_segment_length);
        const bool valid = prefix_valid[index] && suffix_valid[index] && segments;
        valid_out[index] = valid;
        const float total = valid ? prefix + suffix : NAN;
        path_length[index] = total;
        delay[index] = valid ? total / kSpeedOfLightF : NAN;
    }
}

__global__ void coupled_active_mask_kernel(
    int64_t count,
    const bool *__restrict__ lhs,
    const bool *__restrict__ rhs,
    bool *__restrict__ out) {
    for (int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < count;
         index += static_cast<int64_t>(blockDim.x) * gridDim.x) {
        out[index] = lhs[index] && rhs[index];
    }
}

__global__ void coupled_active_either_mask_kernel(
    int64_t count,
    const bool *__restrict__ lhs,
    const bool *__restrict__ rhs,
    const bool *__restrict__ other,
    bool *__restrict__ out) {
    for (int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < count;
         index += static_cast<int64_t>(blockDim.x) * gridDim.x) {
        out[index] = lhs[index] && (rhs[index] || other[index]);
    }
}

// Double-diffraction (cid 7) discovery. Eliminating Q1 with RayD's closed-form
// line solve leaves a convex one-dimensional path-length objective on edge 2.
// A signed derivative bracket plus fixed bisection replaces the unverified
// sixteen-step alternating projection.
constexpr int kDoubleDiffractionBisectionIterations = 32;
constexpr float kDoubleDiffractionPositionUlps = 8.0f;

struct ReducedDdState {
    float parameter1;
    float3 q1;
    float3 q2;
    float derivative;
    bool regular;
};

__device__ __forceinline__ ReducedDdState reduced_dd_state(
    float parameter2,
    float3 tx,
    float3 rx,
    float3 origin1,
    float3 dir1,
    float3 origin2,
    float3 dir2) {
    // Solve and differentiate in edge-1 local coordinates so a short middle
    // leg does not lose direction through an absolute-coordinate round trip.
    const float3 source_from_origin1 = channel::math::sub(tx, origin1);
    const float3 edge2_origin_from_origin1 = channel::math::sub(origin2, origin1);
    const float3 q2_from_origin1 = channel::math::add(
        edge2_origin_from_origin1, channel::math::scale(dir2, parameter2));
    const float3 q2 = channel::math::add(origin2, channel::math::scale(dir2, parameter2));
    const float parameter1 = utd::first_order_diffraction_parameter(
        to_utd(source_from_origin1),
        to_utd(q2_from_origin1),
        to_utd(make_float3(0.0f, 0.0f, 0.0f)),
        to_utd(dir1));
    const float3 q1_from_origin1 = channel::math::scale(dir1, parameter1);
    const float3 q1 = channel::math::add(origin1, q1_from_origin1);
    const float3 incoming = channel::math::sub(q1_from_origin1, source_from_origin1);
    const float3 middle = channel::math::sub(q2_from_origin1, q1_from_origin1);
    const float3 outgoing = channel::math::add(
        channel::math::sub(origin2, rx), channel::math::scale(dir2, parameter2));
    const float incoming_length = channel::math::length(incoming);
    const float middle_length = channel::math::length(middle);
    const float outgoing_length = channel::math::length(outgoing);
    const bool regular =
        isfinite(parameter1) &&
        isfinite(q1.x) && isfinite(q1.y) && isfinite(q1.z) &&
        isfinite(q2.x) && isfinite(q2.y) && isfinite(q2.z) &&
        isfinite(incoming_length) && incoming_length > kGeometryEpsilon &&
        isfinite(middle_length) && middle_length > kGeometryEpsilon &&
        isfinite(outgoing_length) && outgoing_length > kGeometryEpsilon;
    const float derivative = regular
        ? channel::math::dot(
              dir2,
              channel::math::add(
                  channel::math::scale(middle, 1.0f / middle_length),
                  channel::math::scale(outgoing, 1.0f / outgoing_length)))
        : NAN;
    return {parameter1, q1, q2, derivative, regular && isfinite(derivative)};
}

// A candidate and its reverse (endpoints and edges exchanged) are solved as the same problem:
// the edge with the smaller id is always the first edge of the solve, so the reduced bisection
// runs on the same edge in both directions and their published points agree exactly.
__global__ void coupled_dd_prepare_kernel(
    int64_t count,
    const int *__restrict__ edge1_id,
    const int *__restrict__ edge2_id,
    const float *__restrict__ source_in,
    const float *__restrict__ receiver_in,
    const float *__restrict__ edge1_pos_in,
    const float *__restrict__ edge1_dir_in,
    const float *__restrict__ edge1_t_min_in,
    const float *__restrict__ edge1_t_max_in,
    const float *__restrict__ edge2_pos_in,
    const float *__restrict__ edge2_dir_in,
    const float *__restrict__ edge2_t_min_in,
    const float *__restrict__ edge2_t_max_in,
    bool *__restrict__ active,
    float *__restrict__ edge1_point_out,
    float *__restrict__ edge2_point_out,
    int *__restrict__ certificate_rejections) {
    for (int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < count;
         index += static_cast<int64_t>(blockDim.x) * gridDim.x) {
        const bool flip = edge1_id[index] > edge2_id[index];
        const float *source = flip ? receiver_in : source_in;
        const float *receiver = flip ? source_in : receiver_in;
        const float *edge1_pos = flip ? edge2_pos_in : edge1_pos_in;
        const float *edge1_dir = flip ? edge2_dir_in : edge1_dir_in;
        const float *edge1_t_min = flip ? edge2_t_min_in : edge1_t_min_in;
        const float *edge1_t_max = flip ? edge2_t_max_in : edge1_t_max_in;
        const float *edge2_pos = flip ? edge1_pos_in : edge2_pos_in;
        const float *edge2_dir = flip ? edge1_dir_in : edge2_dir_in;
        const float *edge2_t_min = flip ? edge1_t_min_in : edge2_t_min_in;
        const float *edge2_t_max = flip ? edge1_t_max_in : edge2_t_max_in;
        float *edge1_point = flip ? edge2_point_out : edge1_point_out;
        float *edge2_point = flip ? edge1_point_out : edge2_point_out;
        const float3 tx = channel::math::load_vec3(source, index);
        const float3 rx = channel::math::load_vec3(receiver, index);
        const float3 dir1 = channel::math::normalize_or_zero(channel::math::load_vec3(edge1_dir, index), kGeometryEpsilon);
        const float3 dir2 = channel::math::normalize_or_zero(channel::math::load_vec3(edge2_dir, index), kGeometryEpsilon);
        const float t_min1 = edge1_t_min[index];
        const float t_max1 = edge1_t_max[index];
        const float t_min2 = edge2_t_min[index];
        const float t_max2 = edge2_t_max[index];
        const float length1 = t_max1 - t_min1;
        const float length2 = t_max2 - t_min2;
        const float3 origin1 = channel::math::add(channel::math::load_vec3(edge1_pos, index), channel::math::scale(dir1, t_min1));
        const float3 origin2 = channel::math::add(channel::math::load_vec3(edge2_pos, index), channel::math::scale(dir2, t_min2));
        const bool finite_inputs =
            isfinite(tx.x) && isfinite(tx.y) && isfinite(tx.z) &&
            isfinite(rx.x) && isfinite(rx.y) && isfinite(rx.z) &&
            isfinite(origin1.x) && isfinite(origin1.y) && isfinite(origin1.z) &&
            isfinite(origin2.x) && isfinite(origin2.y) && isfinite(origin2.z) &&
            channel::math::length(dir1) > 0.0f && channel::math::length(dir2) > 0.0f &&
            isfinite(length1) && length1 > kGeometryEpsilon &&
            isfinite(length2) && length2 > kGeometryEpsilon;
        if (!finite_inputs) {
            active[index] = false;
            channel::math::store_vec3(edge1_point, index, make_float3(NAN, NAN, NAN));
            channel::math::store_vec3(edge2_point, index, make_float3(NAN, NAN, NAN));
            continue;
        }

        // Collinear edge pairs lie on one physical line; they are duplicate
        // representations of the same edge and cannot form a two-edge cascade.
        // Same-line test uses the shared kGeometryEpsilon: parallel directions
        // AND zero perpendicular offset of origin2 from edge 1's line.
        const bool parallel = channel::math::length(channel::math::cross(dir1, dir2)) < kGeometryEpsilon;
        const float3 delta = channel::math::sub(origin2, origin1);
        const float3 perpendicular = channel::math::sub(delta, channel::math::scale(dir1, channel::math::dot(delta, dir1)));
        const bool same_line = parallel && channel::math::length(perpendicular) < kGeometryEpsilon;
        if (same_line) {
            active[index] = false;
            channel::math::store_vec3(edge1_point, index, make_float3(NAN, NAN, NAN));
            channel::math::store_vec3(edge2_point, index, make_float3(NAN, NAN, NAN));
            continue;
        }

        // The reduced path-length objective h(v) is convex. Its envelope
        // derivative is monotone, so a strict sign change over the finite-edge
        // interior proves one ordinary stationary pair. Endpoint equality is
        // vertex physics and is deliberately excluded.
        float lower = kGeometryEpsilon;
        float upper = length2 - kGeometryEpsilon;
        const ReducedDdState lower_state = reduced_dd_state(
            lower, tx, rx, origin1, dir1, origin2, dir2);
        const ReducedDdState upper_state = reduced_dd_state(
            upper, tx, rx, origin1, dir1, origin2, dir2);
        bool bracketed =
            upper > lower && lower_state.regular && upper_state.regular &&
            lower_state.derivative < 0.0f && upper_state.derivative > 0.0f;
        for (int iteration = 0;
             iteration < kDoubleDiffractionBisectionIterations && bracketed;
             ++iteration) {
            const float midpoint = lower + 0.5f * (upper - lower);
            if (midpoint == lower || midpoint == upper) {
                break;
            }
            const ReducedDdState midpoint_state = reduced_dd_state(
                midpoint, tx, rx, origin1, dir1, origin2, dir2);
            if (!midpoint_state.regular) {
                bracketed = false;
            } else if (midpoint_state.derivative > 0.0f) {
                upper = midpoint;
            } else if (midpoint_state.derivative < 0.0f) {
                lower = midpoint;
            } else {
                const float left = nextafterf(midpoint, lower);
                const float right = nextafterf(midpoint, upper);
                const ReducedDdState left_state = reduced_dd_state(
                    left, tx, rx, origin1, dir1, origin2, dir2);
                const ReducedDdState right_state = reduced_dd_state(
                    right, tx, rx, origin1, dir1, origin2, dir2);
                bracketed =
                    left < midpoint && midpoint < right &&
                    left_state.regular && right_state.regular &&
                    left_state.derivative < 0.0f && right_state.derivative > 0.0f;
                if (bracketed) {
                    lower = left;
                    upper = right;
                }
                break;
            }
        }

        const ReducedDdState final_lower_state = reduced_dd_state(
            lower, tx, rx, origin1, dir1, origin2, dir2);
        const ReducedDdState final_upper_state = reduced_dd_state(
            upper, tx, rx, origin1, dir1, origin2, dir2);
        const bool conservative_bracket =
            bracketed &&
            lower > kGeometryEpsilon && upper < length2 - kGeometryEpsilon &&
            lower < upper &&
            final_lower_state.regular && final_upper_state.regular &&
            final_lower_state.derivative < 0.0f &&
            final_upper_state.derivative > 0.0f &&
            final_lower_state.parameter1 > kGeometryEpsilon &&
            final_lower_state.parameter1 < length1 - kGeometryEpsilon &&
            final_upper_state.parameter1 > kGeometryEpsilon &&
            final_upper_state.parameter1 < length1 - kGeometryEpsilon;
        const float parameter2 = lower + 0.5f * (upper - lower);
        const ReducedDdState solution_state = reduced_dd_state(
            parameter2, tx, rx, origin1, dir1, origin2, dir2);
        const float parameter1 = solution_state.parameter1;
        const float3 q1 = solution_state.q1;
        const float3 q2 = solution_state.q2;

        const bool inside1 = isfinite(parameter1) && parameter1 > kGeometryEpsilon &&
                             parameter1 < length1 - kGeometryEpsilon;
        const bool inside2 = isfinite(parameter2) && parameter2 > kGeometryEpsilon &&
                             parameter2 < length2 - kGeometryEpsilon;
        const bool finite_points =
            isfinite(q1.x) && isfinite(q1.y) && isfinite(q1.z) &&
            isfinite(q2.x) && isfinite(q2.y) && isfinite(q2.z);
        // Each sequential UTD operator shares RayD's ordinary-leg domain;
        // kGeometryEpsilon remains the finite-edge interior tolerance.
        const float incoming_length = channel::math::length(channel::math::sub(q1, tx));
        const float middle_length = channel::math::length(channel::math::sub(q2, q1));
        const float outgoing_length = channel::math::length(channel::math::sub(rx, q2));
        const bool legs_in_utd_domain =
            isfinite(incoming_length) && incoming_length > utd::UTD_MIN_DISTANCE &&
            isfinite(middle_length) && middle_length > utd::UTD_MIN_DISTANCE &&
            isfinite(outgoing_length) && outgoing_length > utd::UTD_MIN_DISTANCE;
        const float local_scale = fmaxf(
            fmaxf(length1, length2),
            fmaxf(incoming_length, fmaxf(middle_length, outgoing_length)));
        const float position_tolerance =
            kDoubleDiffractionPositionUlps * FLT_EPSILON * local_scale;
        const float edge1_bracket_span = channel::math::length(
            channel::math::sub(final_upper_state.q1, final_lower_state.q1));
        const float edge2_bracket_span = upper - lower;
        const bool position_resolved =
            isfinite(position_tolerance) && position_tolerance > 0.0f &&
            isfinite(edge1_bracket_span) &&
            edge1_bracket_span <= position_tolerance &&
            isfinite(edge2_bracket_span) &&
            edge2_bracket_span <= position_tolerance;
        const bool physical_domain =
            conservative_bracket && inside1 && inside2 && finite_points &&
            solution_state.regular && legs_in_utd_domain;
        if (physical_domain && !position_resolved) {
            // The float32 bisection bracketed a root but could not resolve it
            // to the position tolerance: the candidate has no certified
            // stationary point, so it is dropped and counted.
            active[index] = false;
            channel::math::store_vec3(
                edge1_point, index, make_float3(NAN, NAN, NAN));
            channel::math::store_vec3(
                edge2_point, index, make_float3(NAN, NAN, NAN));
            atomicAdd(certificate_rejections, 1);
            continue;
        }
        const bool valid = physical_domain && position_resolved;
        active[index] = valid;
        channel::math::store_vec3(edge1_point, index, valid ? q1 : make_float3(NAN, NAN, NAN));
        channel::math::store_vec3(edge2_point, index, valid ? q2 : make_float3(NAN, NAN, NAN));
    }
}

__global__ void coupled_dd_finalize_kernel(
    int64_t count,
    const bool *__restrict__ prefix_active,
    const int *__restrict__ edge1_id,
    const int *__restrict__ edge2_id,
    const float *__restrict__ edge1_point,
    const float *__restrict__ edge2_point,
    const float *__restrict__ source,
    const float *__restrict__ receiver,
    int *__restrict__ interaction_type_sequence,
    int *__restrict__ primitive_sequence,
    int *__restrict__ edge_sequence,
    float *__restrict__ interaction_positions,
    float *__restrict__ interaction_normals,
    float *__restrict__ path_length,
    float *__restrict__ delay,
    bool *__restrict__ valid_out) {
    for (int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < count;
         index += static_cast<int64_t>(blockDim.x) * gridDim.x) {
        const int64_t sequence_base = index * 2;
        // Two diffraction events; edge ids for BOTH edges live in edge_sequence
        // (slot 0 = e1, slot 1 = e2). primitive_sequence is fully -1: a
        // double-diffraction row touches no face.
        interaction_type_sequence[sequence_base + 0] = 2;
        interaction_type_sequence[sequence_base + 1] = 2;
        primitive_sequence[sequence_base + 0] = -1;
        primitive_sequence[sequence_base + 1] = -1;
        edge_sequence[sequence_base + 0] = edge1_id[index];
        edge_sequence[sequence_base + 1] = edge2_id[index];

        const float3 q1 = channel::math::load_vec3(edge1_point, index);
        const float3 q2 = channel::math::load_vec3(edge2_point, index);
        channel::math::store_vec3(interaction_positions, sequence_base + 0, q1);
        channel::math::store_vec3(interaction_positions, sequence_base + 1, q2);
        // A diffraction edge has an axis and two face normals, not one surface
        // normal. Mirror the coupled R/D finalize: both diffraction slots keep
        // the generic normal slot explicitly unavailable.
        channel::math::store_vec3(interaction_normals, sequence_base + 0, make_float3(NAN, NAN, NAN));
        channel::math::store_vec3(interaction_normals, sequence_base + 1, make_float3(NAN, NAN, NAN));

        const bool valid = prefix_active[index];
        valid_out[index] = valid;
        const float3 tx = channel::math::load_vec3(source, index);
        const float3 rx = channel::math::load_vec3(receiver, index);
        const float total = channel::math::length(channel::math::sub(q1, tx)) + channel::math::length(channel::math::sub(q2, q1)) +
                            channel::math::length(channel::math::sub(rx, q2));
        path_length[index] = valid ? total : NAN;
        delay[index] = valid ? total / kSpeedOfLightF : NAN;
    }
}

}  // namespace

pybind11::dict channel_coupled_chain_finalize_cuda(
    at::Tensor prefix_valid,
    at::Tensor suffix_valid,
    at::Tensor prefix_length,
    at::Tensor suffix_length,
    at::Tensor prefix_prims,
    at::Tensor suffix_prims,
    at::Tensor prefix_hits,
    at::Tensor suffix_hits,
    at::Tensor prefix_normals,
    at::Tensor suffix_normals,
    at::Tensor source,
    at::Tensor receiver,
    at::Tensor edge_point,
    at::Tensor edge_id,
    double min_segment_length) {
    const int64_t count = source.size(0);
    const int64_t prefix_depth = prefix_prims.size(1);
    const int64_t suffix_depth = suffix_prims.size(1);
    const int64_t depth = prefix_depth + suffix_depth + 1;
    TORCH_CHECK(prefix_valid.scalar_type() == at::kBool && suffix_valid.scalar_type() == at::kBool &&
                    prefix_valid.numel() == count && suffix_valid.numel() == count,
                "chain leg validity must be one bool per row");
    TORCH_CHECK(prefix_depth == 0 || prefix_length.numel() == count, "prefix path length must match rows");
    TORCH_CHECK(suffix_depth == 0 || suffix_length.numel() == count, "suffix path length must match rows");
    auto int_options = edge_id.options();
    auto float_options = source.options();
    pybind11::dict out;
    at::Tensor primitive_sequence = at::empty({count, depth}, int_options);
    at::Tensor interaction_positions = at::empty({count, depth, 3}, float_options);
    at::Tensor interaction_normals = at::empty({count, depth, 3}, float_options);
    at::Tensor path_length = at::empty({count}, float_options);
    at::Tensor delay = at::empty({count}, float_options);
    at::Tensor valid = at::empty({count}, prefix_valid.options());
    if (count > 0) {
        const at::Tensor prefix_valid_c = prefix_valid.contiguous();
        const at::Tensor suffix_valid_c = suffix_valid.contiguous();
        coupled_chain_finalize_kernel<<<::channel::launch_blocks(count, kCoupledTopologyBlockSize),
                                        kCoupledTopologyBlockSize, 0,
                                        at::cuda::getCurrentCUDAStream(source.get_device()).stream()>>>(
            count, prefix_depth, suffix_depth, prefix_valid_c.data_ptr<bool>(), suffix_valid_c.data_ptr<bool>(),
            prefix_depth > 0 ? prefix_length.data_ptr<float>() : nullptr,
            suffix_depth > 0 ? suffix_length.data_ptr<float>() : nullptr,
            prefix_depth > 0 ? prefix_prims.data_ptr<int>() : nullptr,
            suffix_depth > 0 ? suffix_prims.data_ptr<int>() : nullptr,
            prefix_depth > 0 ? prefix_hits.data_ptr<float>() : nullptr,
            suffix_depth > 0 ? suffix_hits.data_ptr<float>() : nullptr,
            prefix_depth > 0 ? prefix_normals.data_ptr<float>() : nullptr,
            suffix_depth > 0 ? suffix_normals.data_ptr<float>() : nullptr, source.data_ptr<float>(),
            receiver.data_ptr<float>(), edge_point.data_ptr<float>(), edge_id.data_ptr<int>(),
            static_cast<float>(min_segment_length), primitive_sequence.data_ptr<int>(),
            interaction_positions.data_ptr<float>(), interaction_normals.data_ptr<float>(),
            path_length.data_ptr<float>(), delay.data_ptr<float>(), valid.data_ptr<bool>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    out["valid"] = valid;
    out["primitive_sequence"] = primitive_sequence;
    out["interaction_positions"] = interaction_positions;
    out["interaction_normals"] = interaction_normals;
    out["path_length_m"] = path_length;
    out["delay_s"] = delay;
    return out;
}

std::vector<at::Tensor> channel_coupled_dd_prepare_cuda(
    at::Tensor edge1_id,
    at::Tensor edge2_id,
    at::Tensor source,
    at::Tensor receiver,
    at::Tensor edge1_pos,
    at::Tensor edge1_dir,
    at::Tensor edge1_t_min,
    at::Tensor edge1_t_max,
    at::Tensor edge2_pos,
    at::Tensor edge2_dir,
    at::Tensor edge2_t_min,
    at::Tensor edge2_t_max) {
    using channel::check_flat_tensor;
    using channel::check_vec3_table;
    check_vec3_table(source, "source");
    check_vec3_table(receiver, "receiver");
    check_vec3_table(edge1_pos, "edge1_pos");
    check_vec3_table(edge1_dir, "edge1_dir");
    check_flat_tensor(edge1_t_min, "edge1_t_min", at::kFloat);
    check_flat_tensor(edge1_t_max, "edge1_t_max", at::kFloat);
    check_vec3_table(edge2_pos, "edge2_pos");
    check_vec3_table(edge2_dir, "edge2_dir");
    check_flat_tensor(edge2_t_min, "edge2_t_min", at::kFloat);
    check_flat_tensor(edge2_t_max, "edge2_t_max", at::kFloat);
    check_flat_tensor(edge1_id, "edge1_id", at::kInt);
    check_flat_tensor(edge2_id, "edge2_id", at::kInt);
    const int64_t count = source.size(0);
    TORCH_CHECK(edge1_id.size(0) == count && edge2_id.size(0) == count, "edge ids must match source rows");
    for (const auto &tensor : {receiver, edge1_pos, edge1_dir, edge2_pos, edge2_dir})
        TORCH_CHECK(tensor.size(0) == count,
                    "coupled double-diffraction vector tables must have matching rows");
    for (const auto &tensor : {edge1_t_min, edge1_t_max, edge2_t_min, edge2_t_max})
        TORCH_CHECK(tensor.size(0) == count,
                    "coupled double-diffraction edge bounds must match source rows");

    auto active = at::empty({count}, source.options().dtype(at::kBool));
    auto edge1_point = at::empty_like(source);
    auto edge2_point = at::empty_like(source);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(source.get_device()).stream();
    auto certificate_rejections = channel::empty_zero_cuda(
        {1}, source.options().dtype(at::kInt), stream);
    if (count > 0) {
        coupled_dd_prepare_kernel<<<::channel::launch_blocks(count, kCoupledTopologyBlockSize),
                                    kCoupledTopologyBlockSize, 0, stream>>>(
            count,
            edge1_id.data_ptr<int>(),
            edge2_id.data_ptr<int>(),
            source.data_ptr<float>(),
            receiver.data_ptr<float>(),
            edge1_pos.data_ptr<float>(),
            edge1_dir.data_ptr<float>(),
            edge1_t_min.data_ptr<float>(),
            edge1_t_max.data_ptr<float>(),
            edge2_pos.data_ptr<float>(),
            edge2_dir.data_ptr<float>(),
            edge2_t_min.data_ptr<float>(),
            edge2_t_max.data_ptr<float>(),
            active.data_ptr<bool>(),
            edge1_point.data_ptr<float>(),
            edge2_point.data_ptr<float>(),
            certificate_rejections.data_ptr<int>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return {active, edge1_point, edge2_point, certificate_rejections};
}

pybind11::dict channel_coupled_dd_finalize_cuda(
    at::Tensor prefix_active,
    at::Tensor edge1_id,
    at::Tensor edge2_id,
    at::Tensor edge1_point,
    at::Tensor edge2_point,
    at::Tensor source,
    at::Tensor receiver) {
    using channel::check_flat_tensor;
    using channel::check_vec3_table;
    check_flat_tensor(prefix_active, "prefix_active", at::kBool);
    check_flat_tensor(edge1_id, "edge1_id", at::kInt);
    check_flat_tensor(edge2_id, "edge2_id", at::kInt);
    check_vec3_table(edge1_point, "edge1_point");
    check_vec3_table(edge2_point, "edge2_point");
    check_vec3_table(source, "source");
    check_vec3_table(receiver, "receiver");
    const int64_t count = prefix_active.size(0);
    TORCH_CHECK(edge1_id.size(0) == count && edge2_id.size(0) == count,
                "coupled double-diffraction edge ids must match valid");
    for (const auto &tensor : {edge1_point, edge2_point, source, receiver})
        TORCH_CHECK(tensor.size(0) == count,
                    "coupled double-diffraction vector rows must match valid");

    auto int_options = edge1_id.options().dtype(at::kInt);
    auto interaction_type_sequence = at::empty({count, 2}, int_options);
    auto primitive_sequence = at::empty({count, 2}, int_options);
    auto edge_sequence = at::empty({count, 2}, int_options);
    auto interaction_positions = at::empty({count, 2, 3}, receiver.options());
    auto interaction_normals = at::empty_like(interaction_positions);
    auto path_length = at::empty({count}, receiver.options());
    auto delay = at::empty_like(path_length);
    auto valid = at::empty_like(prefix_active);
    if (count > 0) {
        cudaStream_t stream = at::cuda::getCurrentCUDAStream(prefix_active.get_device()).stream();
        coupled_dd_finalize_kernel<<<::channel::launch_blocks(count, kCoupledTopologyBlockSize),
                                     kCoupledTopologyBlockSize, 0, stream>>>(
            count,
            prefix_active.data_ptr<bool>(),
            edge1_id.data_ptr<int>(),
            edge2_id.data_ptr<int>(),
            edge1_point.data_ptr<float>(),
            edge2_point.data_ptr<float>(),
            source.data_ptr<float>(),
            receiver.data_ptr<float>(),
            interaction_type_sequence.data_ptr<int>(),
            primitive_sequence.data_ptr<int>(),
            edge_sequence.data_ptr<int>(),
            interaction_positions.data_ptr<float>(),
            interaction_normals.data_ptr<float>(),
            path_length.data_ptr<float>(),
            delay.data_ptr<float>(),
            valid.data_ptr<bool>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

    pybind11::dict out;
    out["valid"] = valid;
    out["interaction_type_sequence"] = interaction_type_sequence;
    out["primitive_sequence"] = primitive_sequence;
    out["edge_sequence"] = edge_sequence;
    out["edge1_id"] = edge1_id;
    out["edge2_id"] = edge2_id;
    out["interaction_positions"] = interaction_positions;
    out["interaction_normals"] = interaction_normals;
    out["edge1_position"] = edge1_point;
    out["edge2_position"] = edge2_point;
    out["path_length_m"] = path_length;
    out["delay_s"] = delay;
    return out;
}

at::Tensor channel_coupled_active_mask_cuda(at::Tensor lhs, at::Tensor rhs) {
    using channel::check_flat_tensor;
    check_flat_tensor(lhs, "lhs", at::kBool);
    check_flat_tensor(rhs, "rhs", at::kBool);
    TORCH_CHECK(rhs.size(0) == lhs.size(0), "rhs must match lhs");
    auto out = at::empty_like(lhs);
    const int64_t count = lhs.size(0);
    if (count > 0) {
        cudaStream_t stream = at::cuda::getCurrentCUDAStream(lhs.get_device()).stream();
        coupled_active_mask_kernel<<<::channel::launch_blocks(count, kCoupledTopologyBlockSize),
                                     kCoupledTopologyBlockSize, 0, stream>>>(
            count,
            lhs.data_ptr<bool>(),
            rhs.data_ptr<bool>(),
            out.data_ptr<bool>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return out;
}

// lhs && (rhs || other): a row stays active when either of two traces of one segment is visible.
at::Tensor channel_coupled_active_either_mask_cuda(at::Tensor lhs, at::Tensor rhs, at::Tensor other) {
    using channel::check_flat_tensor;
    check_flat_tensor(lhs, "lhs", at::kBool);
    check_flat_tensor(rhs, "rhs", at::kBool);
    check_flat_tensor(other, "other", at::kBool);
    TORCH_CHECK(rhs.size(0) == lhs.size(0) && other.size(0) == lhs.size(0), "masks must match lhs");
    auto out = at::empty_like(lhs);
    const int64_t count = lhs.size(0);
    if (count > 0) {
        cudaStream_t stream = at::cuda::getCurrentCUDAStream(lhs.get_device()).stream();
        coupled_active_either_mask_kernel<<<
            ::channel::launch_blocks(count, kCoupledTopologyBlockSize),
            kCoupledTopologyBlockSize, 0, stream>>>(
            count, lhs.data_ptr<bool>(), rhs.data_ptr<bool>(), other.data_ptr<bool>(), out.data_ptr<bool>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return out;
}


