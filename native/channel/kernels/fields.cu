// Copyright Xingyu Chen.
// Implements field transport CUDA operations.

// ==== Section: Core field transport ====
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/util/complex.h>
#include "torch_cuda.h"

#include "math.cuh"
#include "ordered_sum.cuh"
#include <rayd/field_transport.cuh>
#include "layered_reflection.cuh"
#include "../tensor_checks.h"

#include <algorithm>
#include <vector>

#define launch_blocks field_transport_launch_blocks
#define kBlockSize kFieldTransportBlockSize

namespace {

constexpr int kBlockSize = 256;
namespace field = rayd::shared::diffraction;
namespace transport = rayd::shared::field_transport;
namespace layered = channel::layered_reflection;



// Up to this many carrier frequencies evaluated by one field-transport launch. Each (row,
// frequency) lane runs the single-frequency arithmetic unchanged; outputs are written at
// row * out_stride + column_offset + f, and outputs a caller does not request are null.
constexpr int kFrequencyColumnLimit = 64;

struct FrequencyColumns {
    float values[kFrequencyColumnLimit];
    int64_t count;
    int64_t out_stride;
    int64_t column_offset;
};

FrequencyColumns single_frequency(double frequency_hz) {
    FrequencyColumns columns = {};
    columns.values[0] = static_cast<float>(frequency_hz);
    columns.count = 1;
    columns.out_stride = 1;
    columns.column_offset = 0;
    return columns;
}

__device__ __forceinline__ c10::complex<float> to_complex(field::Complex value) {
    return c10::complex<float>(value.re, value.im);
}

__device__ __forceinline__ field::Complex from_complex(c10::complex<float> value) {
    return field::cplx(value.real(), value.imag());
}

// One transport lane (row, frequency) before it is stored: the polarized field, the receiver
// projection, the source-excited projection, the path length and the arrival direction.
struct TransportLane {
    field::Complex3 value;
    field::Complex scalar;
    field::Complex received;
    float length;
    field::float3a direction;
};

// Output columns of a transport launch. Every pointer after path_field is null when the caller
// only asks for the field vector and the source-excited projection.
struct TransportOutputs {
    c10::complex<float>* field_vector;
    c10::complex<float>* coefficient;
    c10::complex<float>* path_field;
    float* path_gain;
    float* path_length;
    float* delay;
    float* direction;
};

__device__ __forceinline__ void store_transport_lane(
    const TransportOutputs& out, int64_t index, const TransportLane& lane) {
    const int64_t base = index * 3;
    out.field_vector[base] = to_complex(lane.value.x);
    out.field_vector[base + 1] = to_complex(lane.value.y);
    out.field_vector[base + 2] = to_complex(lane.value.z);
    out.path_field[index] = to_complex(lane.received);
    if (out.coefficient == nullptr) {
        return;
    }
    out.coefficient[index] = to_complex(lane.scalar);
    out.path_gain[index] = field::cplx_abs_sqr(lane.received);
    out.path_length[index] = lane.length;
    out.delay[index] = lane.length / transport::kSpeedOfLight;
    out.direction[base] = lane.direction.x;
    out.direction[base + 1] = lane.direction.y;
    out.direction[base + 2] = lane.direction.z;
}

__device__ __forceinline__ TransportLane free_space_lane(
    field::float3a source_value, field::float3a target_value, float tx_power,
    field::float3a tx_pol, field::float3a rx_pol, float frequency_hz) {
    const field::float3a offset = field::f3_sub(target_value, source_value);
    const float distance = field::safe_length(offset);
    const field::float3a direction = field::safe_normalize(
        offset, field::make_f3(0.0f, 0.0f, 1.0f));
    const float wave_number =
        2.0f * field::UTD_PI * frequency_hz / transport::kSpeedOfLight;
    const float source_amplitude = sqrtf(fmaxf(tx_power, 0.0f));
    TransportLane lane;
    lane.value = transport::free_space_complex3(
        source_value, target_value, wave_number, tx_pol);
    lane.scalar = transport::project_receiver(lane.value, direction, rx_pol);
    lane.received = field::cplx_mul_real(lane.scalar, source_amplitude);
    lane.length = distance;
    lane.direction = direction;
    return lane;
}

__global__ void free_space_kernel(
    int64_t count,
    const float* source,
    const float* target,
    const float* tx_power,
    const float* tx_polarization,
    const float* rx_polarization,
    FrequencyColumns columns,
    TransportOutputs out) {
    for (int64_t lane = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         lane < count * columns.count;
         lane += static_cast<int64_t>(blockDim.x) * gridDim.x) {
        const int64_t index = lane / columns.count;
        const int64_t column = lane - index * columns.count;
        store_transport_lane(
            out, index * columns.out_stride + columns.column_offset + column,
            free_space_lane(
                channel::math::load_field_vec3(source, index),
                channel::math::load_field_vec3(target, index), tx_power[index],
                channel::math::load_field_vec3(tx_polarization, index),
                channel::math::load_field_vec3(rx_polarization, index), columns.values[column]));
    }
}

__global__ void project_complex3_kernel(
    int64_t count,
    const c10::complex<float>* field_vector,
    const float* direction,
    const float* rx_polarization,
    c10::complex<float>* coefficient,
    float* path_gain) {
    for (int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < count;
         index += static_cast<int64_t>(blockDim.x) * gridDim.x) {
        const int64_t base = index * 3;
        const field::Complex3 value = {
            from_complex(field_vector[base]),
            from_complex(field_vector[base + 1]),
            from_complex(field_vector[base + 2]),
        };
        const field::Complex scalar = transport::project_receiver(
            value, channel::math::load_field_vec3(direction, index), channel::math::load_field_vec3(rx_polarization, index));
        coefficient[index] = to_complex(scalar);
        path_gain[index] = field::cplx_abs_sqr(scalar);
    }
}

__device__ __forceinline__ field::float3a load_sequence3(
    const float* values, int64_t index, int64_t bounce, int64_t depth) {
    const int64_t base = (index * depth + bounce) * 3;
    return field::make_f3(values[base], values[base + 1], values[base + 2]);
}

// Every output of one reflection lane, computed before any is stored.
struct ReflectionRow {
    field::Complex3 value;
    field::Complex scalar;
    field::Complex received;
    float path_gain;
    float length;
    float delay;
    field::float3a direction;
};

// One reflection lane. It is not inlined, so its floating-point code (and therefore its rounding)
// is the same in every kernel that evaluates reflection rows. ``positions``/``normals`` point at
// the row's first bounce. Slab parameters are read at ``slot_base + bounce`` of (N, D) gathered
// tables, or, when ``face_ids`` is set, at the bounce's face of per-face tables; ``material`` is
// null when no layer store is consulted.
__device__ __noinline__ void reflection_row(
    ReflectionRow& row, field::float3a source_value, field::float3a target_value, float tx_power,
    field::float3a tx_pol, field::float3a rx_pol, const float* positions, const float* normals,
    const float* eps_r, const float* sigma_e, const float* mu_r, const float* gain,
    const float* thickness, const int* material, const int* face_ids, int64_t face_count,
    int64_t slot_base, int64_t depth, const layered::LayerTables& layers, float frequency_hz) {
    field::float3a previous = source_value;
    field::float3a first_hit = load_sequence3(positions, 0, 0, 1);
    field::float3a incident = field::safe_normalize(
        field::f3_sub(first_hit, previous), field::make_f3(0.0f, 0.0f, 1.0f));
    // unnormalized transverse projection of the transmit polarization
    // (short-dipole sin(theta) weight); the Jones s/p bases stay orthonormal.
    field::float3a tx_axis = field::project_to_wedge_plane(tx_pol, incident);
    field::Complex3 value = field::cplx_scale_real(
        tx_axis, field::cplx(1.0f, 0.0f));
    float total_length = 0.0f;
    field::float3a outgoing = incident;
    for (int64_t bounce = 0; bounce < depth; ++bounce) {
        const field::float3a hit = load_sequence3(positions, 0, bounce, 1);
        incident = field::safe_normalize(
            field::f3_sub(hit, previous), outgoing);
        total_length += field::safe_length(field::f3_sub(hit, previous));
        int64_t slot = slot_base + bounce;
        if (face_ids != nullptr) {
            slot = face_ids[bounce];
            slot = slot < 0 ? slot + face_count : slot;
        }
        value = layered::reflect_layered(
            value,
            incident,
            load_sequence3(normals, 0, bounce, 1),
            eps_r[slot],
            sigma_e[slot],
            mu_r[slot],
            gain[slot],
            thickness[slot],
            frequency_hz,
            material == nullptr ? -1 : material[slot],
            layers,
            outgoing);
        previous = hit;
    }
    const field::float3a final_offset = field::f3_sub(target_value, previous);
    const field::float3a final_direction = field::safe_normalize(
        final_offset, outgoing);
    total_length += field::safe_length(final_offset);
    const float wave_number =
        2.0f * field::UTD_PI * frequency_hz / transport::kSpeedOfLight;
    const float amplitude = 1.0f /
                            (2.0f * wave_number *
                             fmaxf(total_length, field::UTD_EPS));
    const field::Complex propagation = field::cplx_mul_real(
        field::cplx_exp_phase(
            transport::precise_neg_kd(wave_number, total_length)),
        amplitude);
    row.value = field::c3_scale(value, propagation);
    row.scalar = transport::project_receiver(row.value, final_direction, rx_pol);
    row.received = field::cplx_mul_real(row.scalar, sqrtf(fmaxf(tx_power, 0.0f)));
    row.path_gain = field::cplx_abs_sqr(row.received);
    row.length = total_length;
    row.delay = total_length / transport::kSpeedOfLight;
    row.direction = final_direction;
}

__global__ void reflection_sequence_kernel(
    int64_t count,
    int64_t depth,
    const float* source,
    const float* target,
    const float* interaction_positions,
    const float* interaction_normals,
    const float* tx_power,
    const float* tx_polarization,
    const float* rx_polarization,
    const float* eps_r,
    const float* sigma_e,
    const float* mu_r,
    const float* gain,
    const float* thickness,
    const int* bounce_material,
    layered::LayerTables layers,
    FrequencyColumns columns,
    TransportOutputs out) {
    for (int64_t lane = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         lane < count * columns.count;
         lane += static_cast<int64_t>(blockDim.x) * gridDim.x) {
        const int64_t index = lane / columns.count;
        const int64_t column = lane - index * columns.count;
        const int64_t target_index = index * columns.out_stride + columns.column_offset + column;
        ReflectionRow row;
        reflection_row(
            row, channel::math::load_field_vec3(source, index),
            channel::math::load_field_vec3(target, index), tx_power[index],
            channel::math::load_field_vec3(tx_polarization, index),
            channel::math::load_field_vec3(rx_polarization, index),
            interaction_positions + index * depth * 3, interaction_normals + index * depth * 3,
            eps_r, sigma_e, mu_r, gain, thickness, bounce_material, nullptr, 0, index * depth,
            depth, layers, columns.values[column]);
        const int64_t base = target_index * 3;
        out.field_vector[base] = to_complex(row.value.x);
        out.field_vector[base + 1] = to_complex(row.value.y);
        out.field_vector[base + 2] = to_complex(row.value.z);
        out.path_field[target_index] = to_complex(row.received);
        if (out.coefficient == nullptr) {
            continue;
        }
        out.coefficient[target_index] = to_complex(row.scalar);
        out.path_gain[target_index] = row.path_gain;
        out.path_length[target_index] = row.length;
        out.delay[target_index] = row.delay;
        out.direction[base] = row.direction.x;
        out.direction[base + 1] = row.direction.y;
        out.direction[base + 2] = row.direction.z;
    }
}

__device__ __forceinline__ field::float3a reflect_point(
    field::float3a point,
    field::float3a plane_point,
    field::float3a normal) {
    const field::float3a n = field::safe_normalize(
        normal, field::make_f3(0.0f, 0.0f, 1.0f));
    return field::f3_sub(
        point,
        field::f3_mul(n, 2.0f * field::f3_dot(field::f3_sub(point, plane_point), n)));
}

__global__ void coupled_rd_field_kernel(
    int64_t count,
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
    const int* reflection_material,
    layered::LayerTables layers,
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
    const float* edge_line_min,
    const float* edge_line_max,
    float frequency_hz,
    bool reverse,
    c10::complex<float>* field_vector,
    c10::complex<float>* coefficient,
    c10::complex<float>* path_field,
    float* path_gain,
    float* direction_out) {
    for (int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < count;
         index += static_cast<int64_t>(blockDim.x) * gridDim.x) {
        const field::float3a src = channel::math::load_field_vec3(source, index);
        const field::float3a dst = channel::math::load_field_vec3(target, index);
        const field::float3a hit = channel::math::load_field_vec3(reflection_position, index);
        const field::float3a normal = channel::math::load_field_vec3(reflection_normal, index);
        const field::float3a edge = channel::math::load_field_vec3(edge_position, index);
        const field::float3a edge_axis = field::safe_normalize(
            channel::math::load_field_vec3(edge_direction, index), field::make_f3(0.0f, 0.0f, 1.0f));
        const float wave_number =
            2.0f * field::UTD_PI * frequency_hz / transport::kSpeedOfLight;

        const field::float3a diffraction_source =
            reverse ? src : reflect_point(src, hit, normal);
        const field::float3a diffraction_target =
            reverse ? reflect_point(dst, hit, normal) : dst;
        const field::float3a incident_direction = field::safe_normalize(
            field::f3_sub(edge, diffraction_source),
            field::make_f3(1.0f, 0.0f, 0.0f));
        const field::float3a outgoing_direction = field::safe_normalize(
            field::f3_sub(diffraction_target, edge),
            field::make_f3(1.0f, 0.0f, 0.0f));
        const field::Basis3 input_edge_basis = field::diffraction_edge_basis(
            field::f3_sub(edge, diffraction_source), edge_axis, false);
        const field::Basis3 output_edge_basis = field::diffraction_edge_basis(
            field::f3_sub(diffraction_target, edge), edge_axis, true);

        field::Complex3 incident_field;
        if (reverse) {
            incident_field = transport::free_space_complex3(
                src,
                edge,
                wave_number,
                channel::math::load_field_vec3(tx_polarization, index));
        } else {
            const field::float3a source_to_hit = field::safe_normalize(
                field::f3_sub(hit, src), incident_direction);
            // unnormalized transverse projection of the transmit polarization.
            const field::float3a tx_axis = field::project_to_wedge_plane(
                channel::math::load_field_vec3(tx_polarization, index), source_to_hit);
            field::float3a reflected_direction;
            field::Complex3 reflected = layered::reflect_layered(
                field::cplx_scale_real(tx_axis, field::cplx(1.0f, 0.0f)),
                source_to_hit,
                normal,
                reflection_eps_r[index],
                reflection_sigma_e[index],
                reflection_mu_r[index],
                reflection_gain[index],
                reflection_thickness[index],
                frequency_hz,
                reflection_material[index],
                layers,
                reflected_direction);
            const float unfolded_distance = field::safe_length(
                field::f3_sub(edge, diffraction_source));
            const float amplitude = 1.0f /
                                    (2.0f * wave_number *
                                     fmaxf(unfolded_distance, field::UTD_EPS));
            const field::Complex propagation = field::cplx_mul_real(
                field::cplx_exp_phase(transport::precise_neg_kd(
                    wave_number, unfolded_distance)),
                amplitude);
            incident_field = field::c3_scale(reflected, propagation);
        }

        field::PairInputs pair{};
        pair.edgePos = edge;
        pair.edgeDir = edge_axis;
        pair.n0 = channel::math::load_field_vec3(edge_n0, index);
        pair.nn = channel::math::load_field_vec3(edge_n1, index);
        pair.wedgeN = exterior_angle[index] / field::UTD_PI;
        // Real segment bounds let the stationary solver truncate the coupled leg
        // and apply its corner correction around the supplied edge point.
        pair.edgeLineMin = edge_line_min[index];
        pair.edgeLineMax = edge_line_max[index];
        pair.sourcePos = diffraction_source;
        pair.incidentBasis = input_edge_basis;
        pair.incidentJones = field::jones_from_vector(
            incident_field, input_edge_basis);
        pair.incidentDerivativeJones = field::jones_zero();
        pair.face0Material.present = 1.0f;
        pair.face1Material.present = 1.0f;
        float pair_face0_cos, pair_face1_cos;
        field::pair_face_cosines(pair, diffraction_target, pair_face0_cos, pair_face1_cos);
        pair.face0Operator = transport::slab_face_operator(
            pair_face0_cos,
            wedge_eps_r0[index],
            wedge_sigma_e0[index],
            wedge_mu_r0[index],
            wedge_gain0[index],
            wedge_thickness0[index],
            frequency_hz,
            pair.n0,
            incident_direction,
            outgoing_direction,
            input_edge_basis,
            output_edge_basis);
        pair.face1Operator = transport::slab_face_operator(
            pair_face1_cos,
            wedge_eps_r1[index],
            wedge_sigma_e1[index],
            wedge_mu_r1[index],
            wedge_gain1[index],
            wedge_thickness1[index],
            frequency_hz,
            pair.nn,
            incident_direction,
            outgoing_direction,
            input_edge_basis,
            output_edge_basis);
        // run the stationary-path continuity machinery (edge re-anchoring,
        // monotone even truncation, corner_mend_gamma, boundary-distance blend)
        // on the coupled diffraction leg. stationaryExternalIncident tells the
        // header to re-extrapolate the frozen EXTERNAL incident (the coupled
        // image-source spherical wave stored in incidentJones/incidentBasis) to
        // the re-anchored stationary point instead of using the direct source.
        // The slab face operators stay frozen at this Keller point (omega < 0
        // keeps the stored operators) per the coupled contract.
        pair.selectStationaryPoint = 1.0f;
        pair.stationaryExternalIncident = 1.0f;
        field::MaterialParams material{};
        material.omega = -1.0f;
        field::Complex3 value = field::compute_pair_contribution(
            pair, diffraction_target, wave_number, material).vectorField;

        field::float3a final_direction = outgoing_direction;
        if (reverse) {
            value = layered::reflect_layered(
                value,
                field::safe_normalize(field::f3_sub(hit, edge), outgoing_direction),
                normal,
                reflection_eps_r[index],
                reflection_sigma_e[index],
                reflection_mu_r[index],
                reflection_gain[index],
                reflection_thickness[index],
                frequency_hz,
                reflection_material[index],
                layers,
                final_direction);
        }
        const int64_t base = index * 3;
        field_vector[base] = to_complex(value.x);
        field_vector[base + 1] = to_complex(value.y);
        field_vector[base + 2] = to_complex(value.z);
        const field::Complex scalar = transport::project_receiver(
            value, final_direction, channel::math::load_field_vec3(rx_polarization, index));
        coefficient[index] = to_complex(scalar);
        const field::Complex received = field::cplx_mul_real(
            scalar, sqrtf(fmaxf(tx_power[index], 0.0f)));
        path_field[index] = to_complex(received);
        path_gain[index] = field::cplx_abs_sqr(received);
        direction_out[base] = final_direction.x;
        direction_out[base + 1] = final_direction.y;
        direction_out[base + 2] = final_direction.z;
    }
}

// coupled double diffraction: coupled double diffraction (TX -> e1 -> e2 -> RX), component id 7.
//
// Two sequential wedge operators in ONE launch. Both legs run the stationary
// path so each inherits the full continuity machinery (edge re-anchoring,
// monotone even truncation, corner-mend gamma, boundary blend) - the lesson
// is honored from day one. Q1/Q2 are the frozen discovery Fermat seeds; the
// per-leg edge bounds are frozen (detached) offsets from the passed Keller
// point, exactly like the coupled R->D leg.
//
// Leg 1 (e1): direct spherical incident from tx (stationaryExternalIncident=0),
// observation at the frozen Q2. The single-edge stationary point on e1 between
// tx and Q2 reproduces Q1 (the two-edge Fermat point), so the re-anchor is a
// no-op at the discovered geometry. Output: the diffracted vector field at Q2.
// Leg 2 (e2): sourcePos = frozen Q1, incidentJones = leg-1 output projected on
// e2's incident basis at the frozen Q2, stationaryExternalIncident=1 so the
// header re-extrapolates that frozen wave from Q2 to the re-anchored Q2*
// (sPrimeFrozen = |edgePos - sourcePos| = |Q1 - Q2| is captured internally),
// observation at rx. The re-extrapolation is EXACT at Q2* == Q2 and
// second-order in the re-anchor displacement (the coupled-path approximation approximation class).
// The leg-2 coefficient evaluation is a SINGLE call site (compute_pair_
// contribution) so the transition-coefficient isolation can swap it for the two-variable transition
// function without touching the surrounding transport.
__global__ void coupled_dd_field_kernel(
    int64_t count,
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
    float frequency_hz,
    c10::complex<float>* field_vector,
    c10::complex<float>* coefficient,
    c10::complex<float>* path_field,
    float* path_gain,
    float* direction_out) {
    for (int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < count;
         index += static_cast<int64_t>(blockDim.x) * gridDim.x) {
        const field::float3a src = channel::math::load_field_vec3(source, index);
        const field::float3a dst = channel::math::load_field_vec3(target, index);
        const field::float3a q1 = channel::math::load_field_vec3(edge1_position, index);
        const field::float3a q2 = channel::math::load_field_vec3(edge2_position, index);
        const field::float3a e1_axis = field::safe_normalize(
            channel::math::load_field_vec3(edge1_direction, index), field::make_f3(0.0f, 0.0f, 1.0f));
        const field::float3a e2_axis = field::safe_normalize(
            channel::math::load_field_vec3(edge2_direction, index), field::make_f3(0.0f, 0.0f, 1.0f));
        const float wave_number =
            2.0f * field::UTD_PI * frequency_hz / transport::kSpeedOfLight;

        // --- Leg 1: direct-source diffraction at e1, observed at frozen Q2 ---
        const field::float3a incident_direction1 = field::safe_normalize(
            field::f3_sub(q1, src), field::make_f3(1.0f, 0.0f, 0.0f));
        const field::float3a outgoing_direction1 = field::safe_normalize(
            field::f3_sub(q2, q1), field::make_f3(1.0f, 0.0f, 0.0f));
        const field::Basis3 input_edge_basis1 = field::diffraction_edge_basis(
            field::f3_sub(q1, src), e1_axis, false);
        const field::Basis3 output_edge_basis1 = field::diffraction_edge_basis(
            field::f3_sub(q2, q1), e1_axis, true);

        field::PairInputs pair1{};
        pair1.edgePos = q1;
        pair1.edgeDir = e1_axis;
        pair1.n0 = channel::math::load_field_vec3(edge1_n0, index);
        pair1.nn = channel::math::load_field_vec3(edge1_n1, index);
        pair1.wedgeN = edge1_exterior[index] / field::UTD_PI;
        pair1.edgeLineMin = edge1_line_min[index];
        pair1.edgeLineMax = edge1_line_max[index];
        pair1.sourcePos = src;
        // Direct-source path ignores incidentBasis/Jones; set them consistently.
        pair1.incidentBasis = input_edge_basis1;
        pair1.incidentJones = field::jones_zero();
        pair1.incidentDerivativeJones = field::jones_zero();
        pair1.face0Material.present = 1.0f;
        pair1.face1Material.present = 1.0f;
        float pair1_face0_cos, pair1_face1_cos;
        field::pair_face_cosines(pair1, q2, pair1_face0_cos, pair1_face1_cos);
        pair1.face0Operator = transport::slab_face_operator(
            pair1_face0_cos,
            wedge1_eps_r0[index],
            wedge1_sigma_e0[index],
            wedge1_mu_r0[index],
            wedge1_gain0[index],
            wedge1_thickness0[index],
            frequency_hz,
            pair1.n0,
            incident_direction1,
            outgoing_direction1,
            input_edge_basis1,
            output_edge_basis1);
        pair1.face1Operator = transport::slab_face_operator(
            pair1_face1_cos,
            wedge1_eps_r1[index],
            wedge1_sigma_e1[index],
            wedge1_mu_r1[index],
            wedge1_gain1[index],
            wedge1_thickness1[index],
            frequency_hz,
            pair1.nn,
            incident_direction1,
            outgoing_direction1,
            input_edge_basis1,
            output_edge_basis1);
        pair1.selectStationaryPoint = 1.0f;
        pair1.stationaryExternalIncident = 0.0f;
        const field::float3a tx_pol = channel::math::load_field_vec3(tx_polarization, index);
        field::MaterialParams material1{};
        // omega < 0 keeps the stored (frozen-at-Keller) slab face operators; the
        // direct incident spherical wave still uses the transmitter polarization.
        material1.omega = -1.0f;
        material1.txPolX = tx_pol.x;
        material1.txPolY = tx_pol.y;
        material1.txPolZ = tx_pol.z;
        const field::Complex3 leg1_field = field::compute_pair_contribution(
            pair1, q2, wave_number, material1).vectorField;

        // --- Leg 2: external-incident diffraction at e2, observed at rx ---
        const field::float3a incident_direction2 = field::safe_normalize(
            field::f3_sub(q2, q1), field::make_f3(1.0f, 0.0f, 0.0f));
        const field::float3a outgoing_direction2 = field::safe_normalize(
            field::f3_sub(dst, q2), field::make_f3(1.0f, 0.0f, 0.0f));
        const field::Basis3 input_edge_basis2 = field::diffraction_edge_basis(
            field::f3_sub(q2, q1), e2_axis, false);
        const field::Basis3 output_edge_basis2 = field::diffraction_edge_basis(
            field::f3_sub(dst, q2), e2_axis, true);

        field::PairInputs pair2{};
        pair2.edgePos = q2;
        pair2.edgeDir = e2_axis;
        pair2.n0 = channel::math::load_field_vec3(edge2_n0, index);
        pair2.nn = channel::math::load_field_vec3(edge2_n1, index);
        pair2.wedgeN = edge2_exterior[index] / field::UTD_PI;
        pair2.edgeLineMin = edge2_line_min[index];
        pair2.edgeLineMax = edge2_line_max[index];
        pair2.sourcePos = q1;
        pair2.incidentBasis = input_edge_basis2;
        // Leg-1 diffracted field projected onto e2's incident basis at frozen Q2.
        pair2.incidentJones = field::jones_from_vector(
            leg1_field, input_edge_basis2);
        pair2.incidentDerivativeJones = field::jones_zero();
        pair2.face0Material.present = 1.0f;
        pair2.face1Material.present = 1.0f;
        float pair2_face0_cos, pair2_face1_cos;
        field::pair_face_cosines(pair2, dst, pair2_face0_cos, pair2_face1_cos);
        pair2.face0Operator = transport::slab_face_operator(
            pair2_face0_cos,
            wedge2_eps_r0[index],
            wedge2_sigma_e0[index],
            wedge2_mu_r0[index],
            wedge2_gain0[index],
            wedge2_thickness0[index],
            frequency_hz,
            pair2.n0,
            incident_direction2,
            outgoing_direction2,
            input_edge_basis2,
            output_edge_basis2);
        pair2.face1Operator = transport::slab_face_operator(
            pair2_face1_cos,
            wedge2_eps_r1[index],
            wedge2_sigma_e1[index],
            wedge2_mu_r1[index],
            wedge2_gain1[index],
            wedge2_thickness1[index],
            frequency_hz,
            pair2.nn,
            incident_direction2,
            outgoing_direction2,
            input_edge_basis2,
            output_edge_basis2);
        pair2.selectStationaryPoint = 1.0f;
        pair2.stationaryExternalIncident = 1.0f;
        field::MaterialParams material2{};
        material2.omega = -1.0f;
        // Single leg-2 coefficient call site (the transition-coefficient isolation swaps this for the
        // two-variable transition function). sPrimeFrozen = |Q1 - Q2| is
        // recaptured inside compute_pair_vector_contribution before the re-anchor.
        const field::Complex3 value = field::compute_pair_contribution(
            pair2, dst, wave_number, material2).vectorField;

        const field::float3a final_direction = outgoing_direction2;
        const int64_t base = index * 3;
        field_vector[base] = to_complex(value.x);
        field_vector[base + 1] = to_complex(value.y);
        field_vector[base + 2] = to_complex(value.z);
        const field::Complex scalar = transport::project_receiver(
            value, final_direction, channel::math::load_field_vec3(rx_polarization, index));
        coefficient[index] = to_complex(scalar);
        const field::Complex received = field::cplx_mul_real(
            scalar, sqrtf(fmaxf(tx_power[index], 0.0f)));
        path_field[index] = to_complex(received);
        path_gain[index] = field::cplx_abs_sqr(received);
        direction_out[base] = final_direction.x;
        direction_out[base + 1] = final_direction.y;
        direction_out[base + 2] = final_direction.z;
    }
}

int launch_blocks(int64_t count) {
    return static_cast<int>((count + kBlockSize - 1) / kBlockSize);
}

}  // namespace

namespace {

void check_free_space_inputs(
    const at::Tensor& source, const at::Tensor& target, const at::Tensor& tx_power,
    const at::Tensor& tx_polarization, const at::Tensor& rx_polarization) {
    using channel::check_flat_tensor;
    using channel::check_vec3_table;
    check_vec3_table(source, "source");
    check_vec3_table(target, "target");
    check_flat_tensor(tx_power, "tx_power", at::kFloat);
    check_vec3_table(tx_polarization, "tx_polarization");
    check_vec3_table(rx_polarization, "rx_polarization");
    const int64_t count = source.size(0);
    TORCH_CHECK(target.size(0) == count, "target must match source rows");
    TORCH_CHECK(tx_power.size(0) == count, "tx_power must match source rows");
    TORCH_CHECK(tx_polarization.size(0) == count, "tx_polarization must match source rows");
    TORCH_CHECK(rx_polarization.size(0) == count, "rx_polarization must match source rows");
}

std::vector<FrequencyColumns> frequency_column_chunks(const std::vector<double>& frequencies) {
    TORCH_CHECK(!frequencies.empty(), "frequency columns must not be empty");
    std::vector<FrequencyColumns> chunks;
    const int64_t total = static_cast<int64_t>(frequencies.size());
    for (int64_t start = 0; start < total; start += kFrequencyColumnLimit) {
        FrequencyColumns columns = {};
        columns.count = std::min<int64_t>(kFrequencyColumnLimit, total - start);
        columns.out_stride = total;
        columns.column_offset = start;
        for (int64_t column = 0; column < columns.count; ++column) {
            const double frequency = frequencies[static_cast<size_t>(start + column)];
            TORCH_CHECK(frequency > 0.0, "frequency_hz must be positive");
            columns.values[column] = static_cast<float>(frequency);
        }
        chunks.push_back(columns);
    }
    return chunks;
}

void launch_free_space(
    const at::Tensor& source, const at::Tensor& target, const at::Tensor& tx_power,
    const at::Tensor& tx_polarization, const at::Tensor& rx_polarization,
    const FrequencyColumns& columns, c10::complex<float>* field_vector,
    c10::complex<float>* coefficient, c10::complex<float>* path_field, float* path_gain,
    float* path_length, float* delay, float* direction) {
    const int64_t count = source.size(0);
    if (count == 0) {
        return;
    }
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(source.get_device()).stream();
    free_space_kernel<<<launch_blocks(count * columns.count), kBlockSize, 0, stream>>>(
        count,
        source.data_ptr<float>(),
        target.data_ptr<float>(),
        tx_power.data_ptr<float>(),
        tx_polarization.data_ptr<float>(),
        rx_polarization.data_ptr<float>(),
        columns,
        TransportOutputs{
            field_vector, coefficient, path_field, path_gain, path_length, delay, direction});
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace

pybind11::dict channel_field_free_space(
    at::Tensor source,
    at::Tensor target,
    at::Tensor tx_power,
    at::Tensor tx_polarization,
    at::Tensor rx_polarization,
    double frequency_hz) {
    check_free_space_inputs(source, target, tx_power, tx_polarization, rx_polarization);
    TORCH_CHECK(frequency_hz > 0.0, "frequency_hz must be positive");
    const int64_t count = source.size(0);
    auto complex_options = source.options().dtype(at::kComplexFloat);
    auto field_vector = at::empty({count, 3}, complex_options);
    auto coefficient = at::empty({count}, complex_options);
    auto path_field = at::empty({count}, complex_options);
    auto path_gain = at::empty({count}, source.options());
    auto path_length = at::empty_like(path_gain);
    auto delay = at::empty_like(path_gain);
    auto direction = at::empty_like(source);
    launch_free_space(
        source, target, tx_power, tx_polarization, rx_polarization, single_frequency(frequency_hz),
        field_vector.data_ptr<c10::complex<float>>(), coefficient.data_ptr<c10::complex<float>>(),
        path_field.data_ptr<c10::complex<float>>(), path_gain.data_ptr<float>(),
        path_length.data_ptr<float>(), delay.data_ptr<float>(), direction.data_ptr<float>());
    pybind11::dict out;
    out["field_vector"] = field_vector;
    out["coefficient"] = coefficient;
    out["path_field"] = path_field;
    out["path_gain"] = path_gain;
    out["path_length_m"] = path_length;
    out["delay_s"] = delay;
    out["direction"] = direction;
    return out;
}

pybind11::dict channel_field_free_space_columns(
    at::Tensor source,
    at::Tensor target,
    at::Tensor tx_power,
    at::Tensor tx_polarization,
    at::Tensor rx_polarization,
    std::vector<double> frequencies_hz) {
    check_free_space_inputs(source, target, tx_power, tx_polarization, rx_polarization);
    const auto chunks = frequency_column_chunks(frequencies_hz);
    const int64_t count = source.size(0);
    const int64_t columns = static_cast<int64_t>(frequencies_hz.size());
    auto complex_options = source.options().dtype(at::kComplexFloat);
    auto field_vector = at::empty({count, columns, 3}, complex_options);
    auto path_field = at::empty({count, columns}, complex_options);
    for (const auto& chunk : chunks) {
        launch_free_space(
            source, target, tx_power, tx_polarization, rx_polarization, chunk,
            field_vector.data_ptr<c10::complex<float>>(), nullptr,
            path_field.data_ptr<c10::complex<float>>(), nullptr, nullptr, nullptr, nullptr);
    }
    pybind11::dict out;
    out["field_vector"] = field_vector;
    out["path_field"] = path_field;
    return out;
}

pybind11::dict channel_field_project_complex3(
    at::Tensor field_vector,
    at::Tensor direction,
    at::Tensor rx_polarization) {
    using channel::check_tensor;
    using channel::check_vec3_table;
    check_tensor(field_vector, "field_vector", at::kComplexFloat, 2);
    TORCH_CHECK(field_vector.size(1) == 3, "field_vector must have shape (N, 3)");
    check_vec3_table(direction, "direction");
    check_vec3_table(rx_polarization, "rx_polarization");
    const int64_t count = field_vector.size(0);
    TORCH_CHECK(direction.size(0) == count, "direction must match field_vector rows");
    TORCH_CHECK(rx_polarization.size(0) == count,
                "rx_polarization must match field_vector rows");
    auto coefficient = at::empty({count}, field_vector.options());
    auto path_gain = at::empty({count}, direction.options());
    if (count > 0) {
        cudaStream_t stream = at::cuda::getCurrentCUDAStream(direction.get_device()).stream();
        project_complex3_kernel<<<launch_blocks(count), kBlockSize, 0, stream>>>(
            count,
            field_vector.data_ptr<c10::complex<float>>(),
            direction.data_ptr<float>(),
            rx_polarization.data_ptr<float>(),
            coefficient.data_ptr<c10::complex<float>>(),
            path_gain.data_ptr<float>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    pybind11::dict out;
    out["coefficient"] = coefficient;
    out["path_gain"] = path_gain;
    return out;
}

namespace {

void check_reflection_sequence_inputs(
    const at::Tensor& source, const at::Tensor& target, const at::Tensor& interaction_positions,
    const at::Tensor& interaction_normals, const at::Tensor& tx_power,
    const at::Tensor& tx_polarization, const at::Tensor& rx_polarization, const at::Tensor& eps_r,
    const at::Tensor& sigma_e, const at::Tensor& mu_r, const at::Tensor& gain,
    const at::Tensor& thickness) {
    using channel::check_flat_tensor;
    using channel::check_tensor;
    using channel::check_vec3_table;
    check_vec3_table(source, "source");
    check_vec3_table(target, "target");
    check_tensor(interaction_positions, "interaction_positions", at::kFloat, 3);
    check_tensor(interaction_normals, "interaction_normals", at::kFloat, 3);
    check_flat_tensor(tx_power, "tx_power", at::kFloat);
    check_vec3_table(tx_polarization, "tx_polarization");
    check_vec3_table(rx_polarization, "rx_polarization");
    check_tensor(eps_r, "eps_r", at::kFloat, 2);
    check_tensor(sigma_e, "sigma_e", at::kFloat, 2);
    check_tensor(mu_r, "mu_r", at::kFloat, 2);
    check_tensor(gain, "gain", at::kFloat, 2);
    check_tensor(thickness, "thickness", at::kFloat, 2);
    const int64_t count = source.size(0);
    const int64_t depth = interaction_positions.size(1);
    TORCH_CHECK(depth > 0 && interaction_positions.size(2) == 3,
                "interaction_positions must have shape (N, D, 3) with D > 0");
    TORCH_CHECK(interaction_positions.size(0) == count,
                "interaction_positions must match source rows");
    TORCH_CHECK(interaction_normals.sizes() == interaction_positions.sizes(),
                "interaction_normals must match interaction_positions");
    for (const auto& tensor : {eps_r, sigma_e, mu_r, gain, thickness})
        TORCH_CHECK(tensor.size(0) == count && tensor.size(1) == depth,
                    "reflection material tensors must have shape (N, D)");
    TORCH_CHECK(target.size(0) == count && tx_power.size(0) == count &&
                    tx_polarization.size(0) == count && rx_polarization.size(0) == count,
                "reflection endpoint tensors must match source rows");
}

// Layer store view of a reflection request: per-bounce material ids (N, D) plus the compiled CSR layer
// tables. Material ids outside the store fall back to the scalar slab view in the kernel.
layered::LayerTables reflection_layer_tables(
    const at::Tensor& bounce_material,
    const at::Tensor& layer_offset,
    const at::Tensor& layer_count,
    const at::Tensor& layer_thickness_m,
    const at::Tensor& layer_eps_r,
    const at::Tensor& layer_sigma_e,
    const at::Tensor& layer_mu_r,
    int64_t count,
    int64_t depth) {
    using channel::check_tensor;
    check_tensor(bounce_material, "bounce_material", at::kInt, 2);
    TORCH_CHECK(bounce_material.size(0) == count && bounce_material.size(1) == depth,
                "bounce_material must have shape (N, D)");
    check_tensor(layer_offset, "layer_offset", at::kInt, 1);
    check_tensor(layer_count, "layer_count", at::kInt, 1);
    TORCH_CHECK(layer_count.size(0) == layer_offset.size(0), "layer_count must match layer_offset");
    for (const auto& tensor : {layer_thickness_m, layer_eps_r, layer_sigma_e, layer_mu_r})
        check_tensor(tensor, "layer table", at::kFloat, 1);
    TORCH_CHECK(layer_eps_r.size(0) == layer_thickness_m.size(0) && layer_sigma_e.size(0) == layer_thickness_m.size(0) &&
                    layer_mu_r.size(0) == layer_thickness_m.size(0),
                "layer tables must share one length");
    layered::LayerTables tables;
    tables.offset = layer_offset.data_ptr<int>();
    tables.count = layer_count.data_ptr<int>();
    tables.thickness_m = layer_thickness_m.data_ptr<float>();
    tables.eps_r = layer_eps_r.data_ptr<float>();
    tables.sigma_e = layer_sigma_e.data_ptr<float>();
    tables.mu_r = layer_mu_r.data_ptr<float>();
    tables.material_count = layer_count.size(0);
    return tables;
}

void launch_reflection_sequence(
    const at::Tensor& source, const at::Tensor& target, const at::Tensor& interaction_positions,
    const at::Tensor& interaction_normals, const at::Tensor& tx_power,
    const at::Tensor& tx_polarization, const at::Tensor& rx_polarization, const at::Tensor& eps_r,
    const at::Tensor& sigma_e, const at::Tensor& mu_r, const at::Tensor& gain,
    const at::Tensor& thickness, const at::Tensor& bounce_material, const layered::LayerTables& layers,
    const FrequencyColumns& columns, c10::complex<float>* field_vector, c10::complex<float>* coefficient,
    c10::complex<float>* path_field, float* path_gain, float* path_length, float* delay,
    float* direction) {
    const int64_t count = source.size(0);
    if (count == 0) {
        return;
    }
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(source.get_device()).stream();
    reflection_sequence_kernel<<<launch_blocks(count * columns.count), kBlockSize, 0, stream>>>(
        count,
        interaction_positions.size(1),
        source.data_ptr<float>(),
        target.data_ptr<float>(),
        interaction_positions.data_ptr<float>(),
        interaction_normals.data_ptr<float>(),
        tx_power.data_ptr<float>(),
        tx_polarization.data_ptr<float>(),
        rx_polarization.data_ptr<float>(),
        eps_r.data_ptr<float>(),
        sigma_e.data_ptr<float>(),
        mu_r.data_ptr<float>(),
        gain.data_ptr<float>(),
        thickness.data_ptr<float>(),
        bounce_material.data_ptr<int>(),
        layers,
        columns,
        TransportOutputs{
            field_vector, coefficient, path_field, path_gain, path_length, delay, direction});
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace

pybind11::dict channel_field_reflection_sequence(
    at::Tensor source,
    at::Tensor target,
    at::Tensor interaction_positions,
    at::Tensor interaction_normals,
    at::Tensor tx_power,
    at::Tensor tx_polarization,
    at::Tensor rx_polarization,
    at::Tensor eps_r,
    at::Tensor sigma_e,
    at::Tensor mu_r,
    at::Tensor gain,
    at::Tensor thickness,
    at::Tensor bounce_material,
    at::Tensor layer_offset,
    at::Tensor layer_count,
    at::Tensor layer_thickness_m,
    at::Tensor layer_eps_r,
    at::Tensor layer_sigma_e,
    at::Tensor layer_mu_r,
    double frequency_hz) {
    check_reflection_sequence_inputs(
        source, target, interaction_positions, interaction_normals, tx_power, tx_polarization,
        rx_polarization, eps_r, sigma_e, mu_r, gain, thickness);
    const layered::LayerTables layers = reflection_layer_tables(
        bounce_material, layer_offset, layer_count, layer_thickness_m, layer_eps_r, layer_sigma_e,
        layer_mu_r, source.size(0), interaction_positions.size(1));
    TORCH_CHECK(frequency_hz > 0.0, "frequency_hz must be positive");
    const int64_t count = source.size(0);
    auto complex_options = source.options().dtype(at::kComplexFloat);
    auto field_vector = at::empty({count, 3}, complex_options);
    auto coefficient = at::empty({count}, complex_options);
    auto path_field = at::empty({count}, complex_options);
    auto path_gain = at::empty({count}, source.options());
    auto path_length = at::empty_like(path_gain);
    auto delay = at::empty_like(path_gain);
    auto direction = at::empty_like(source);
    launch_reflection_sequence(
        source, target, interaction_positions, interaction_normals, tx_power, tx_polarization,
        rx_polarization, eps_r, sigma_e, mu_r, gain, thickness, bounce_material, layers,
        single_frequency(frequency_hz),
        field_vector.data_ptr<c10::complex<float>>(), coefficient.data_ptr<c10::complex<float>>(),
        path_field.data_ptr<c10::complex<float>>(), path_gain.data_ptr<float>(),
        path_length.data_ptr<float>(), delay.data_ptr<float>(), direction.data_ptr<float>());
    pybind11::dict out;
    out["field_vector"] = field_vector;
    out["coefficient"] = coefficient;
    out["path_field"] = path_field;
    out["path_gain"] = path_gain;
    out["path_length_m"] = path_length;
    out["delay_s"] = delay;
    out["direction"] = direction;
    return out;
}

pybind11::dict channel_field_reflection_sequence_columns(
    at::Tensor source,
    at::Tensor target,
    at::Tensor interaction_positions,
    at::Tensor interaction_normals,
    at::Tensor tx_power,
    at::Tensor tx_polarization,
    at::Tensor rx_polarization,
    at::Tensor eps_r,
    at::Tensor sigma_e,
    at::Tensor mu_r,
    at::Tensor gain,
    at::Tensor thickness,
    at::Tensor bounce_material,
    at::Tensor layer_offset,
    at::Tensor layer_count,
    at::Tensor layer_thickness_m,
    at::Tensor layer_eps_r,
    at::Tensor layer_sigma_e,
    at::Tensor layer_mu_r,
    std::vector<double> frequencies_hz) {
    check_reflection_sequence_inputs(
        source, target, interaction_positions, interaction_normals, tx_power, tx_polarization,
        rx_polarization, eps_r, sigma_e, mu_r, gain, thickness);
    const layered::LayerTables layers = reflection_layer_tables(
        bounce_material, layer_offset, layer_count, layer_thickness_m, layer_eps_r, layer_sigma_e,
        layer_mu_r, source.size(0), interaction_positions.size(1));
    const auto chunks = frequency_column_chunks(frequencies_hz);
    const int64_t count = source.size(0);
    const int64_t columns = static_cast<int64_t>(frequencies_hz.size());
    auto complex_options = source.options().dtype(at::kComplexFloat);
    auto field_vector = at::empty({count, columns, 3}, complex_options);
    auto path_field = at::empty({count, columns}, complex_options);
    for (const auto& chunk : chunks) {
        launch_reflection_sequence(
            source, target, interaction_positions, interaction_normals, tx_power, tx_polarization,
            rx_polarization, eps_r, sigma_e, mu_r, gain, thickness, bounce_material, layers, chunk,
            field_vector.data_ptr<c10::complex<float>>(), nullptr,
            path_field.data_ptr<c10::complex<float>>(), nullptr, nullptr, nullptr, nullptr);
    }
    pybind11::dict out;
    out["field_vector"] = field_vector;
    out["path_field"] = path_field;
    return out;
}

pybind11::dict channel_field_coupled_rd(
    at::Tensor source,
    at::Tensor target,
    at::Tensor reflection_position,
    at::Tensor reflection_normal,
    at::Tensor edge_position,
    at::Tensor edge_direction,
    at::Tensor edge_n0,
    at::Tensor edge_n1,
    at::Tensor exterior_angle,
    at::Tensor tx_power,
    at::Tensor tx_polarization,
    at::Tensor rx_polarization,
    at::Tensor reflection_eps_r,
    at::Tensor reflection_sigma_e,
    at::Tensor reflection_mu_r,
    at::Tensor reflection_gain,
    at::Tensor reflection_thickness,
    at::Tensor reflection_material,
    at::Tensor layer_offset,
    at::Tensor layer_count,
    at::Tensor layer_thickness_m,
    at::Tensor layer_eps_r,
    at::Tensor layer_sigma_e,
    at::Tensor layer_mu_r,
    at::Tensor wedge_eps_r0,
    at::Tensor wedge_sigma_e0,
    at::Tensor wedge_mu_r0,
    at::Tensor wedge_gain0,
    at::Tensor wedge_thickness0,
    at::Tensor wedge_eps_r1,
    at::Tensor wedge_sigma_e1,
    at::Tensor wedge_mu_r1,
    at::Tensor wedge_gain1,
    at::Tensor wedge_thickness1,
    at::Tensor edge_line_min,
    at::Tensor edge_line_max,
    double frequency_hz,
    bool reverse) {
    using channel::check_flat_tensor;
    using channel::check_vec3_table;
    for (const auto& named : std::vector<std::pair<at::Tensor, const char*>>{
             {source, "source"},
             {target, "target"},
             {reflection_position, "reflection_position"},
             {reflection_normal, "reflection_normal"},
             {edge_position, "edge_position"},
             {edge_direction, "edge_direction"},
             {edge_n0, "edge_n0"},
             {edge_n1, "edge_n1"},
             {tx_polarization, "tx_polarization"},
             {rx_polarization, "rx_polarization"}})
        check_vec3_table(named.first, named.second);
    for (const auto& named : std::vector<std::pair<at::Tensor, const char*>>{
             {exterior_angle, "exterior_angle"},
             {tx_power, "tx_power"},
             {reflection_eps_r, "reflection_eps_r"},
             {reflection_sigma_e, "reflection_sigma_e"},
             {reflection_mu_r, "reflection_mu_r"},
             {reflection_gain, "reflection_gain"},
             {reflection_thickness, "reflection_thickness"},
             {wedge_eps_r0, "wedge_eps_r0"},
             {wedge_sigma_e0, "wedge_sigma_e0"},
             {wedge_mu_r0, "wedge_mu_r0"},
             {wedge_gain0, "wedge_gain0"},
             {wedge_thickness0, "wedge_thickness0"},
             {wedge_eps_r1, "wedge_eps_r1"},
             {wedge_sigma_e1, "wedge_sigma_e1"},
             {wedge_mu_r1, "wedge_mu_r1"},
             {wedge_gain1, "wedge_gain1"},
             {wedge_thickness1, "wedge_thickness1"},
             {edge_line_min, "edge_line_min"},
             {edge_line_max, "edge_line_max"}})
        check_flat_tensor(named.first, named.second, at::kFloat);
    const int64_t count = source.size(0);
    const layered::LayerTables layers = reflection_layer_tables(
        reflection_material, layer_offset, layer_count, layer_thickness_m, layer_eps_r, layer_sigma_e,
        layer_mu_r, count, 1);
    for (const auto& tensor : {target,
                               reflection_position,
                               reflection_normal,
                               edge_position,
                               edge_direction,
                               edge_n0,
                               edge_n1,
                               tx_polarization,
                               rx_polarization})
        TORCH_CHECK(tensor.size(0) == count, "coupled field vector rows must match source");
    for (const auto& tensor : {exterior_angle,
                               tx_power,
                               reflection_eps_r,
                               reflection_sigma_e,
                               reflection_mu_r,
                               reflection_gain,
                               reflection_thickness,
                               wedge_eps_r0,
                               wedge_sigma_e0,
                               wedge_mu_r0,
                               wedge_gain0,
                               wedge_thickness0,
                               wedge_eps_r1,
                               wedge_sigma_e1,
                               wedge_mu_r1,
                               wedge_gain1,
                               wedge_thickness1,
                               edge_line_min,
                               edge_line_max})
        TORCH_CHECK(tensor.size(0) == count, "coupled field scalar rows must match source");
    TORCH_CHECK(frequency_hz > 0.0, "frequency_hz must be positive");

    auto complex_options = source.options().dtype(at::kComplexFloat);
    auto field_vector = at::empty({count, 3}, complex_options);
    auto coefficient = at::empty({count}, complex_options);
    auto path_field = at::empty({count}, complex_options);
    auto path_gain = at::empty({count}, source.options());
    auto direction = at::empty_like(source);
    if (count > 0) {
        cudaStream_t stream = at::cuda::getCurrentCUDAStream(source.get_device()).stream();
        coupled_rd_field_kernel<<<launch_blocks(count), kBlockSize, 0, stream>>>(
            count,
            source.data_ptr<float>(),
            target.data_ptr<float>(),
            reflection_position.data_ptr<float>(),
            reflection_normal.data_ptr<float>(),
            edge_position.data_ptr<float>(),
            edge_direction.data_ptr<float>(),
            edge_n0.data_ptr<float>(),
            edge_n1.data_ptr<float>(),
            exterior_angle.data_ptr<float>(),
            tx_power.data_ptr<float>(),
            tx_polarization.data_ptr<float>(),
            rx_polarization.data_ptr<float>(),
            reflection_eps_r.data_ptr<float>(),
            reflection_sigma_e.data_ptr<float>(),
            reflection_mu_r.data_ptr<float>(),
            reflection_gain.data_ptr<float>(),
            reflection_thickness.data_ptr<float>(),
            reflection_material.data_ptr<int>(),
            layers,
            wedge_eps_r0.data_ptr<float>(),
            wedge_sigma_e0.data_ptr<float>(),
            wedge_mu_r0.data_ptr<float>(),
            wedge_gain0.data_ptr<float>(),
            wedge_thickness0.data_ptr<float>(),
            wedge_eps_r1.data_ptr<float>(),
            wedge_sigma_e1.data_ptr<float>(),
            wedge_mu_r1.data_ptr<float>(),
            wedge_gain1.data_ptr<float>(),
            wedge_thickness1.data_ptr<float>(),
            edge_line_min.data_ptr<float>(),
            edge_line_max.data_ptr<float>(),
            static_cast<float>(frequency_hz),
            reverse,
            field_vector.data_ptr<c10::complex<float>>(),
            coefficient.data_ptr<c10::complex<float>>(),
            path_field.data_ptr<c10::complex<float>>(),
            path_gain.data_ptr<float>(),
            direction.data_ptr<float>());
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

// coupled double diffraction: coupled double diffraction field (component id 7). Outputs are
// identical in shape to channel_field_coupled_rd (no path_length/delay: the geometry
// stage owns those for the DD row, matching the coupled contract).
pybind11::dict channel_field_coupled_dd(
    at::Tensor source,
    at::Tensor target,
    at::Tensor edge1_position,
    at::Tensor edge1_direction,
    at::Tensor edge1_n0,
    at::Tensor edge1_n1,
    at::Tensor edge1_exterior,
    at::Tensor edge2_position,
    at::Tensor edge2_direction,
    at::Tensor edge2_n0,
    at::Tensor edge2_n1,
    at::Tensor edge2_exterior,
    at::Tensor tx_power,
    at::Tensor tx_polarization,
    at::Tensor rx_polarization,
    at::Tensor wedge1_eps_r0,
    at::Tensor wedge1_sigma_e0,
    at::Tensor wedge1_mu_r0,
    at::Tensor wedge1_gain0,
    at::Tensor wedge1_thickness0,
    at::Tensor wedge1_eps_r1,
    at::Tensor wedge1_sigma_e1,
    at::Tensor wedge1_mu_r1,
    at::Tensor wedge1_gain1,
    at::Tensor wedge1_thickness1,
    at::Tensor wedge2_eps_r0,
    at::Tensor wedge2_sigma_e0,
    at::Tensor wedge2_mu_r0,
    at::Tensor wedge2_gain0,
    at::Tensor wedge2_thickness0,
    at::Tensor wedge2_eps_r1,
    at::Tensor wedge2_sigma_e1,
    at::Tensor wedge2_mu_r1,
    at::Tensor wedge2_gain1,
    at::Tensor wedge2_thickness1,
    at::Tensor edge1_line_min,
    at::Tensor edge1_line_max,
    at::Tensor edge2_line_min,
    at::Tensor edge2_line_max,
    double frequency_hz) {
    using channel::check_flat_tensor;
    using channel::check_vec3_table;
    for (const auto& named : std::vector<std::pair<at::Tensor, const char*>>{
             {source, "source"},
             {target, "target"},
             {edge1_position, "edge1_position"},
             {edge1_direction, "edge1_direction"},
             {edge1_n0, "edge1_n0"},
             {edge1_n1, "edge1_n1"},
             {edge2_position, "edge2_position"},
             {edge2_direction, "edge2_direction"},
             {edge2_n0, "edge2_n0"},
             {edge2_n1, "edge2_n1"},
             {tx_polarization, "tx_polarization"},
             {rx_polarization, "rx_polarization"}})
        check_vec3_table(named.first, named.second);
    for (const auto& named : std::vector<std::pair<at::Tensor, const char*>>{
             {edge1_exterior, "edge1_exterior"},
             {edge2_exterior, "edge2_exterior"},
             {tx_power, "tx_power"},
             {wedge1_eps_r0, "wedge1_eps_r0"},
             {wedge1_sigma_e0, "wedge1_sigma_e0"},
             {wedge1_mu_r0, "wedge1_mu_r0"},
             {wedge1_gain0, "wedge1_gain0"},
             {wedge1_thickness0, "wedge1_thickness0"},
             {wedge1_eps_r1, "wedge1_eps_r1"},
             {wedge1_sigma_e1, "wedge1_sigma_e1"},
             {wedge1_mu_r1, "wedge1_mu_r1"},
             {wedge1_gain1, "wedge1_gain1"},
             {wedge1_thickness1, "wedge1_thickness1"},
             {wedge2_eps_r0, "wedge2_eps_r0"},
             {wedge2_sigma_e0, "wedge2_sigma_e0"},
             {wedge2_mu_r0, "wedge2_mu_r0"},
             {wedge2_gain0, "wedge2_gain0"},
             {wedge2_thickness0, "wedge2_thickness0"},
             {wedge2_eps_r1, "wedge2_eps_r1"},
             {wedge2_sigma_e1, "wedge2_sigma_e1"},
             {wedge2_mu_r1, "wedge2_mu_r1"},
             {wedge2_gain1, "wedge2_gain1"},
             {wedge2_thickness1, "wedge2_thickness1"},
             {edge1_line_min, "edge1_line_min"},
             {edge1_line_max, "edge1_line_max"},
             {edge2_line_min, "edge2_line_min"},
             {edge2_line_max, "edge2_line_max"}})
        check_flat_tensor(named.first, named.second, at::kFloat);
    const int64_t count = source.size(0);
    for (const auto& tensor : {target,
                               edge1_position,
                               edge1_direction,
                               edge1_n0,
                               edge1_n1,
                               edge2_position,
                               edge2_direction,
                               edge2_n0,
                               edge2_n1,
                               tx_polarization,
                               rx_polarization})
        TORCH_CHECK(tensor.size(0) == count, "coupled dd field vector rows must match source");
    for (const auto& tensor : {edge1_exterior,
                               edge2_exterior,
                               tx_power,
                               wedge1_eps_r0,
                               wedge1_sigma_e0,
                               wedge1_mu_r0,
                               wedge1_gain0,
                               wedge1_thickness0,
                               wedge1_eps_r1,
                               wedge1_sigma_e1,
                               wedge1_mu_r1,
                               wedge1_gain1,
                               wedge1_thickness1,
                               wedge2_eps_r0,
                               wedge2_sigma_e0,
                               wedge2_mu_r0,
                               wedge2_gain0,
                               wedge2_thickness0,
                               wedge2_eps_r1,
                               wedge2_sigma_e1,
                               wedge2_mu_r1,
                               wedge2_gain1,
                               wedge2_thickness1,
                               edge1_line_min,
                               edge1_line_max,
                               edge2_line_min,
                               edge2_line_max})
        TORCH_CHECK(tensor.size(0) == count, "coupled dd field scalar rows must match source");
    TORCH_CHECK(frequency_hz > 0.0, "frequency_hz must be positive");

    auto complex_options = source.options().dtype(at::kComplexFloat);
    auto field_vector = at::empty({count, 3}, complex_options);
    auto coefficient = at::empty({count}, complex_options);
    auto path_field = at::empty({count}, complex_options);
    auto path_gain = at::empty({count}, source.options());
    auto direction = at::empty_like(source);
    if (count > 0) {
        cudaStream_t stream = at::cuda::getCurrentCUDAStream(source.get_device()).stream();
        coupled_dd_field_kernel<<<launch_blocks(count), kBlockSize, 0, stream>>>(
            count,
            source.data_ptr<float>(),
            target.data_ptr<float>(),
            edge1_position.data_ptr<float>(),
            edge1_direction.data_ptr<float>(),
            edge1_n0.data_ptr<float>(),
            edge1_n1.data_ptr<float>(),
            edge1_exterior.data_ptr<float>(),
            edge2_position.data_ptr<float>(),
            edge2_direction.data_ptr<float>(),
            edge2_n0.data_ptr<float>(),
            edge2_n1.data_ptr<float>(),
            edge2_exterior.data_ptr<float>(),
            tx_power.data_ptr<float>(),
            tx_polarization.data_ptr<float>(),
            rx_polarization.data_ptr<float>(),
            wedge1_eps_r0.data_ptr<float>(),
            wedge1_sigma_e0.data_ptr<float>(),
            wedge1_mu_r0.data_ptr<float>(),
            wedge1_gain0.data_ptr<float>(),
            wedge1_thickness0.data_ptr<float>(),
            wedge1_eps_r1.data_ptr<float>(),
            wedge1_sigma_e1.data_ptr<float>(),
            wedge1_mu_r1.data_ptr<float>(),
            wedge1_gain1.data_ptr<float>(),
            wedge1_thickness1.data_ptr<float>(),
            wedge2_eps_r0.data_ptr<float>(),
            wedge2_sigma_e0.data_ptr<float>(),
            wedge2_mu_r0.data_ptr<float>(),
            wedge2_gain0.data_ptr<float>(),
            wedge2_thickness0.data_ptr<float>(),
            wedge2_eps_r1.data_ptr<float>(),
            wedge2_sigma_e1.data_ptr<float>(),
            wedge2_mu_r1.data_ptr<float>(),
            wedge2_gain1.data_ptr<float>(),
            wedge2_thickness1.data_ptr<float>(),
            edge1_line_min.data_ptr<float>(),
            edge1_line_max.data_ptr<float>(),
            edge2_line_min.data_ptr<float>(),
            edge2_line_max.data_ptr<float>(),
            static_cast<float>(frequency_hz),
            field_vector.data_ptr<c10::complex<float>>(),
            coefficient.data_ptr<c10::complex<float>>(),
            path_field.data_ptr<c10::complex<float>>(),
            path_gain.data_ptr<float>(),
            direction.data_ptr<float>());
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

#undef launch_blocks
#undef kBlockSize

// ==== Section: Free-space transport ====
#include "field_ad.cuh"

namespace {

// ---------------------------------------------------------------------------
// Free space (frequency is the only differentiable input in AD).
// ---------------------------------------------------------------------------

template <typename T>
__global__ void free_space_fwd_kernel(
    int64_t count,
    const T* source,
    const T* target,
    const T* tx_power,
    const T* tx_polarization,
    const T* rx_polarization,
    T frequency_hz,
    c10::complex<T>* field_vector,
    c10::complex<T>* coefficient,
    c10::complex<T>* path_field,
    T* path_gain,
    T* path_length,
    T* delay,
    T* direction_out) {
    for (int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < count;
         index += static_cast<int64_t>(blockDim.x) * gridDim.x) {
        const ad::FreeSpaceEval<T> eval = ad::free_space_eval<T>(
            channel::math::load_rayd_vec3(source, index),
            channel::math::load_rayd_vec3(target, index),
            channel::math::load_rayd_vec3(tx_polarization, index),
            channel::math::load_rayd_vec3(rx_polarization, index),
            tx_power[index],
            frequency_hz);
        const int64_t base = index * 3;
        field_vector[base] = eval.carrier * eval.tx_axis.x;
        field_vector[base + 1] = eval.carrier * eval.tx_axis.y;
        field_vector[base + 2] = eval.carrier * eval.tx_axis.z;
        const c10::complex<T> scalar = eval.carrier * eval.projection;
        coefficient[index] = scalar;
        const c10::complex<T> received = scalar * eval.amplitude_scale;
        path_field[index] = received;
        path_gain[index] = received.real() * received.real() +
                           received.imag() * received.imag();
        path_length[index] = eval.distance;
        delay[index] = eval.distance / T(ad::kSpeedOfLight);
        direction_out[base] = eval.direction.x;
        direction_out[base + 1] = eval.direction.y;
        direction_out[base + 2] = eval.direction.z;
    }
}

template <typename T>
__global__ void free_space_backward_kernel(
    int64_t count,
    const T* source,
    const T* target,
    const T* tx_power,
    const T* tx_polarization,
    const T* rx_polarization,
    T frequency_hz,
    const c10::complex<T>* grad_field_vector,
    const c10::complex<T>* grad_coefficient,
    const c10::complex<T>* grad_path_field,
    const T* grad_path_gain,
    const T* grad_path_length,
    const T* grad_delay,
    const T* grad_direction,
    T* grad_frequency,
    T* grad_source,
    T* grad_target) {
    for (int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < count;
         index += static_cast<int64_t>(blockDim.x) * gridDim.x) {
        const ad::FreeSpaceEval<T> eval = ad::free_space_eval<T>(
            channel::math::load_rayd_vec3(source, index),
            channel::math::load_rayd_vec3(target, index),
            channel::math::load_rayd_vec3(tx_polarization, index),
            channel::math::load_rayd_vec3(rx_polarization, index),
            tx_power[index],
            frequency_hz);
        const int64_t base = index * 3;
        // Fold the field-output cotangents onto the carrier P; the geometry
        // enters through the carrier distance, the tx/rx bases and the raw
        // straight length (path_length_m / delay_s).
        const c10::complex<T> path_field_value =
            eval.carrier * eval.projection * eval.amplitude_scale;
        c10::complex<T> g_scalar(T(0), T(0));
        if (grad_coefficient != nullptr)
            g_scalar += grad_coefficient[index];
        if (grad_path_field != nullptr)
            g_scalar += grad_path_field[index] * eval.amplitude_scale;
        if (grad_path_gain != nullptr)
            g_scalar += path_field_value *
                        (T(2) * grad_path_gain[index] * eval.amplitude_scale);
        c10::complex<T> g_carrier = g_scalar * eval.projection;
        if (grad_field_vector != nullptr) {
            g_carrier += grad_field_vector[base] * eval.tx_axis.x;
            g_carrier += grad_field_vector[base + 1] * eval.tx_axis.y;
            g_carrier += grad_field_vector[base + 2] * eval.tx_axis.z;
        }
        if (grad_frequency != nullptr) {
            const T g_freq = g_carrier.real() * eval.carrier_dfreq.real() +
                             g_carrier.imag() * eval.carrier_dfreq.imag();
            grad_frequency[index] = g_freq;
        }
        if (grad_source == nullptr && grad_target == nullptr)
            continue;
        // Real-pair cotangents of the tx/rx bases: coefficient = P *
        // <tx_axis, rx_axis> and field_vector = P * tx_axis.
        const T g_projection = g_scalar.real() * eval.carrier.real() +
                               g_scalar.imag() * eval.carrier.imag();
        ad::Vec3<T> g_tx_axis = {T(0), T(0), T(0)};
        ad::Vec3<T> g_rx_axis = {T(0), T(0), T(0)};
        g_tx_axis = vmath::add(g_tx_axis, vmath::scale(eval.rx_axis, g_projection));
        g_rx_axis = vmath::add(g_rx_axis, vmath::scale(eval.tx_axis, g_projection));
        if (grad_field_vector != nullptr) {
            g_tx_axis.x += grad_field_vector[base].real() * eval.carrier.real() +
                           grad_field_vector[base].imag() * eval.carrier.imag();
            g_tx_axis.y +=
                grad_field_vector[base + 1].real() * eval.carrier.real() +
                grad_field_vector[base + 1].imag() * eval.carrier.imag();
            g_tx_axis.z +=
                grad_field_vector[base + 2].real() * eval.carrier.real() +
                grad_field_vector[base + 2].imag() * eval.carrier.imag();
        }
        T g_distance = g_carrier.real() * eval.carrier_ddist.real() +
                       g_carrier.imag() * eval.carrier_ddist.imag();
        if (grad_path_length != nullptr)
            g_distance += grad_path_length[index];
        if (grad_delay != nullptr)
            g_distance += grad_delay[index] / T(ad::kSpeedOfLight);
        const ad::Vec3<T> offset =
            vmath::subtract(
                channel::math::load_rayd_vec3(target, index),
                channel::math::load_rayd_vec3(source, index));
        // The published arrival direction is exactly eval.direction, so an
        // external cotangent on it seeds the same accumulator the two
        // transverse projections below feed. Seeding here rather than after
        // them keeps one adjoint chain: direction -> offset -> endpoints.
        ad::Vec3<T> g_direction = grad_direction != nullptr
                                      ? channel::math::load_rayd_vec3(
                                            grad_direction, index)
                                      : ad::Vec3<T>{T(0), T(0), T(0)};
        ad::adj_v3_transverse_project(
            eval.direction,
            channel::math::load_rayd_vec3(tx_polarization, index), g_tx_axis,
            g_direction);
        ad::adj_v3_transverse_project(
            eval.direction,
            channel::math::load_rayd_vec3(rx_polarization, index), g_rx_axis,
            g_direction);
        ad::Vec3<T> g_offset = {T(0), T(0), T(0)};
        ad::Vec3<T> g_alternate = {T(0), T(0), T(0)};
        ad::adj_v3_safe_normalize(
            offset, ad::Vec3<T>{T(0), T(0), T(1)}, g_direction, g_offset,
            g_alternate);
        ad::adj_v3_length(offset, g_distance, g_offset);
        if (grad_target != nullptr) {
            grad_target[base] = g_offset.x;
            grad_target[base + 1] = g_offset.y;
            grad_target[base + 2] = g_offset.z;
        }
        if (grad_source != nullptr) {
            grad_source[base] = -g_offset.x;
            grad_source[base + 1] = -g_offset.y;
            grad_source[base + 2] = -g_offset.z;
        }
    }
}

template <typename T>
__global__ void free_space_jvp_kernel(
    int64_t count,
    const T* source,
    const T* target,
    const T* tx_power,
    const T* tx_polarization,
    const T* rx_polarization,
    T frequency_hz,
    T tangent_frequency,
    const T* tangent_source,
    const T* tangent_target,
    c10::complex<T>* t_field_vector,
    c10::complex<T>* t_coefficient,
    c10::complex<T>* t_path_field,
    T* t_path_gain,
    T* t_path_length,
    T* t_delay,
    T* t_direction) {
    for (int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < count;
         index += static_cast<int64_t>(blockDim.x) * gridDim.x) {
        const ad::FreeSpaceEval<T> eval = ad::free_space_eval<T>(
            channel::math::load_rayd_vec3(source, index),
            channel::math::load_rayd_vec3(target, index),
            channel::math::load_rayd_vec3(tx_polarization, index),
            channel::math::load_rayd_vec3(rx_polarization, index),
            tx_power[index],
            frequency_hz);
        const ad::Vec3<T> zero3 = {T(0), T(0), T(0)};
        const ad::Vec3<T> d_source =
            tangent_source != nullptr
                ? channel::math::load_rayd_vec3(tangent_source, index)
                : zero3;
        const ad::Vec3<T> d_target =
            tangent_target != nullptr
                ? channel::math::load_rayd_vec3(tangent_target, index)
                : zero3;
        const ad::DualV3<T> offset = {
            vmath::subtract(
                channel::math::load_rayd_vec3(target, index),
                channel::math::load_rayd_vec3(source, index)),
            vmath::subtract(d_target, d_source)};
        T d_distance = T(0);
        (void)ad::dual_v3_length(offset, d_distance);
        const ad::DualV3<T> direction = ad::dual_v3_safe_normalize(
            offset, ad::dv3_const(ad::Vec3<T>{T(0), T(0), T(1)}));
        const ad::DualV3<T> tx_axis = ad::dual_v3_transverse_project(
            direction,
            channel::math::load_rayd_vec3(tx_polarization, index));
        const ad::DualV3<T> rx_axis = ad::dual_v3_transverse_project(
            direction,
            channel::math::load_rayd_vec3(rx_polarization, index));
        const c10::complex<T> d_carrier =
            eval.carrier_dfreq * tangent_frequency +
            eval.carrier_ddist * d_distance;
        const int64_t base = index * 3;
        t_field_vector[base] =
            d_carrier * eval.tx_axis.x + eval.carrier * tx_axis.d.x;
        t_field_vector[base + 1] =
            d_carrier * eval.tx_axis.y + eval.carrier * tx_axis.d.y;
        t_field_vector[base + 2] =
            d_carrier * eval.tx_axis.z + eval.carrier * tx_axis.d.z;
        const T d_projection = vmath::dot(tx_axis.d, eval.rx_axis) +
                               vmath::dot(eval.tx_axis, rx_axis.d);
        const c10::complex<T> d_scalar =
            d_carrier * eval.projection + eval.carrier * d_projection;
        t_coefficient[index] = d_scalar;
        const c10::complex<T> d_path_field = d_scalar * eval.amplitude_scale;
        t_path_field[index] = d_path_field;
        const c10::complex<T> path_field_value =
            eval.carrier * eval.projection * eval.amplitude_scale;
        t_path_gain[index] =
            T(2) * (path_field_value.real() * d_path_field.real() +
                    path_field_value.imag() * d_path_field.imag());
        t_path_length[index] = d_distance;
        t_delay[index] = d_distance / T(ad::kSpeedOfLight);
        // direction.d is the dual already built above for the two transverse
        // projections; publishing it is three stores and no extra math.
        t_direction[base] = direction.d.x;
        t_direction[base + 1] = direction.d.y;
        t_direction[base + 2] = direction.d.z;
    }
}

void check_free_space_primal(
    const at::Tensor& source,
    const at::Tensor& target,
    const at::Tensor& tx_power,
    const at::Tensor& tx_polarization,
    const at::Tensor& rx_polarization,
    double frequency_hz,
    c10::ScalarType real_dtype) {
    using channel::check_tensor;
    for (const auto& named : std::vector<std::pair<at::Tensor, const char*>>{
             {source, "source"},
             {target, "target"},
             {tx_polarization, "tx_polarization"},
             {rx_polarization, "rx_polarization"}}) {
        check_tensor(named.first, named.second, real_dtype, 2);
        TORCH_CHECK(
            named.first.size(1) == 3, named.second, " must have shape (N, 3)");
    }
    check_tensor(tx_power, "tx_power", real_dtype, 1);
    const int64_t count = source.size(0);
    TORCH_CHECK(
        target.size(0) == count && tx_power.size(0) == count &&
            tx_polarization.size(0) == count && rx_polarization.size(0) == count,
        "free-space field tensors must have matching rows");
    TORCH_CHECK(frequency_hz > 0.0, "frequency_hz must be positive");
}

}  // namespace

pybind11::dict channel_field_free_space_fwd64(
    at::Tensor source,
    at::Tensor target,
    at::Tensor tx_power,
    at::Tensor tx_polarization,
    at::Tensor rx_polarization,
    double frequency_hz) {
    check_free_space_primal(
        source, target, tx_power, tx_polarization, rx_polarization,
        frequency_hz, at::kDouble);
    const int64_t count = source.size(0);
    auto complex_options = source.options().dtype(at::kComplexDouble);
    auto field_vector = at::empty({count, 3}, complex_options);
    auto coefficient = at::empty({count}, complex_options);
    auto path_field = at::empty({count}, complex_options);
    auto path_gain = at::empty({count}, source.options());
    auto path_length = at::empty_like(path_gain);
    auto delay = at::empty_like(path_gain);
    auto direction = at::empty_like(source);
    if (count > 0) {
        cudaStream_t stream =
            at::cuda::getCurrentCUDAStream(source.get_device()).stream();
        free_space_fwd_kernel<double><<<launch_blocks(count), kBlockSize, 0, stream>>>(
            count,
            source.data_ptr<double>(),
            target.data_ptr<double>(),
            tx_power.data_ptr<double>(),
            tx_polarization.data_ptr<double>(),
            rx_polarization.data_ptr<double>(),
            frequency_hz,
            field_vector.data_ptr<c10::complex<double>>(),
            coefficient.data_ptr<c10::complex<double>>(),
            path_field.data_ptr<c10::complex<double>>(),
            path_gain.data_ptr<double>(),
            path_length.data_ptr<double>(),
            delay.data_ptr<double>(),
            direction.data_ptr<double>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    pybind11::dict out;
    out["field_vector"] = field_vector;
    out["coefficient"] = coefficient;
    out["path_field"] = path_field;
    out["path_gain"] = path_gain;
    out["path_length_m"] = path_length;
    out["delay_s"] = delay;
    out["direction"] = direction;
    return out;
}

pybind11::dict channel_field_free_space_backward(
    at::Tensor source,
    at::Tensor target,
    at::Tensor tx_power,
    at::Tensor tx_polarization,
    at::Tensor rx_polarization,
    double frequency_hz,
    pybind11::object grad_field_vector,
    pybind11::object grad_coefficient,
    pybind11::object grad_path_field,
    pybind11::object grad_path_gain,
    pybind11::object grad_path_length,
    pybind11::object grad_delay,
    pybind11::object grad_direction,
    bool need_grad_frequency,
    bool need_grad_geometry) {
    const c10::ScalarType real_dtype = source.scalar_type();
    TORCH_CHECK(
        real_dtype == at::kFloat || real_dtype == at::kDouble,
        "field_free_space_backward supports float32 and float64");
    const c10::ScalarType complex_dtype =
        real_dtype == at::kFloat ? at::kComplexFloat : at::kComplexDouble;
    check_free_space_primal(
        source, target, tx_power, tx_polarization, rx_polarization,
        frequency_hz, real_dtype);
    const int64_t count = source.size(0);
    at::Tensor gfv_storage;
    at::Tensor gc_storage;
    at::Tensor gpf_storage;
    at::Tensor gpg_storage;
    at::Tensor gpl_storage;
    at::Tensor gd_storage;
    at::Tensor gdir_storage;
    const at::Tensor* gfv = optional_grad(
        std::move(grad_field_vector), gfv_storage, "grad_field_vector",
        complex_dtype, {count, 3}, source);
    const at::Tensor* gc = optional_grad(
        std::move(grad_coefficient), gc_storage, "grad_coefficient",
        complex_dtype, {count}, source);
    const at::Tensor* gpf = optional_grad(
        std::move(grad_path_field), gpf_storage, "grad_path_field",
        complex_dtype, {count}, source);
    const at::Tensor* gpg = optional_grad(
        std::move(grad_path_gain), gpg_storage, "grad_path_gain",
        real_dtype, {count}, source);
    const at::Tensor* gpl = optional_grad(
        std::move(grad_path_length), gpl_storage, "grad_path_length",
        real_dtype, {count}, source);
    const at::Tensor* gd = optional_grad(
        std::move(grad_delay), gd_storage, "grad_delay",
        real_dtype, {count}, source);
    const at::Tensor* gdir = optional_grad(
        std::move(grad_direction), gdir_storage, "grad_direction",
        real_dtype, {count, 3}, source);

    pybind11::dict out;
    if (!need_grad_frequency && !need_grad_geometry) {
        out["grad_frequency"] = pybind11::none();
        out["grad_source"] = pybind11::none();
        out["grad_target"] = pybind11::none();
        return out;
    }
    at::Tensor grad_frequency = need_grad_frequency
                                    ? zero_filled({1}, source.options())
                                    : at::Tensor();
    // Per-row frequency cotangents, summed below in a fixed order.
    at::Tensor row_frequency = need_grad_frequency
                                   ? zero_filled({count}, source.options())
                                   : at::Tensor();
    at::Tensor grad_source = need_grad_geometry
                                 ? zero_filled({count, 3}, source.options())
                                 : at::Tensor();
    at::Tensor grad_target = need_grad_geometry
                                 ? zero_filled({count, 3}, source.options())
                                 : at::Tensor();
    const bool any_grad = gfv != nullptr || gc != nullptr || gpf != nullptr ||
                          gpg != nullptr || gpl != nullptr || gd != nullptr ||
                          gdir != nullptr;
    if (count > 0 && any_grad) {
        cudaStream_t stream =
            at::cuda::getCurrentCUDAStream(source.get_device()).stream();
        if (real_dtype == at::kFloat) {
            free_space_backward_kernel<float>
                <<<launch_blocks(count), kBlockSize, 0, stream>>>(
                    count,
                    source.data_ptr<float>(),
                    target.data_ptr<float>(),
                    tx_power.data_ptr<float>(),
                    tx_polarization.data_ptr<float>(),
                    rx_polarization.data_ptr<float>(),
                    static_cast<float>(frequency_hz),
                    gfv ? gfv->data_ptr<c10::complex<float>>() : nullptr,
                    gc ? gc->data_ptr<c10::complex<float>>() : nullptr,
                    gpf ? gpf->data_ptr<c10::complex<float>>() : nullptr,
                    grad_ptr<float>(gpg),
                    grad_ptr<float>(gpl),
                    grad_ptr<float>(gd),
                    grad_ptr<float>(gdir),
                    need_grad_frequency ? row_frequency.data_ptr<float>()
                                        : nullptr,
                    need_grad_geometry ? grad_source.data_ptr<float>() : nullptr,
                    need_grad_geometry ? grad_target.data_ptr<float>() : nullptr);
        } else {
            free_space_backward_kernel<double>
                <<<launch_blocks(count), kBlockSize, 0, stream>>>(
                    count,
                    source.data_ptr<double>(),
                    target.data_ptr<double>(),
                    tx_power.data_ptr<double>(),
                    tx_polarization.data_ptr<double>(),
                    rx_polarization.data_ptr<double>(),
                    frequency_hz,
                    gfv ? gfv->data_ptr<c10::complex<double>>() : nullptr,
                    gc ? gc->data_ptr<c10::complex<double>>() : nullptr,
                    gpf ? gpf->data_ptr<c10::complex<double>>() : nullptr,
                    grad_ptr<double>(gpg),
                    grad_ptr<double>(gpl),
                    grad_ptr<double>(gd),
                    grad_ptr<double>(gdir),
                    need_grad_frequency ? row_frequency.data_ptr<double>()
                                        : nullptr,
                    need_grad_geometry ? grad_source.data_ptr<double>() : nullptr,
                    need_grad_geometry ? grad_target.data_ptr<double>()
                                       : nullptr);
        }
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        if (need_grad_frequency) {
            if (real_dtype == at::kFloat)
                channel::ordered_sum::add_sum<float>(
                    row_frequency.data_ptr<float>(), count, grad_frequency.data_ptr<float>(),
                    source.options(), stream);
            else
                channel::ordered_sum::add_sum<double>(
                    row_frequency.data_ptr<double>(), count, grad_frequency.data_ptr<double>(),
                    source.options(), stream);
        }
    }
    out["grad_frequency"] = need_grad_frequency
                                ? pybind11::cast(grad_frequency)
                                : pybind11::object(pybind11::none());
    out["grad_source"] = need_grad_geometry
                             ? pybind11::cast(grad_source)
                             : pybind11::object(pybind11::none());
    out["grad_target"] = need_grad_geometry
                             ? pybind11::cast(grad_target)
                             : pybind11::object(pybind11::none());
    return out;
}

pybind11::dict channel_field_free_space_jvp(
    at::Tensor source,
    at::Tensor target,
    at::Tensor tx_power,
    at::Tensor tx_polarization,
    at::Tensor rx_polarization,
    double frequency_hz,
    double tangent_frequency,
    pybind11::object tangent_source,
    pybind11::object tangent_target) {
    const c10::ScalarType real_dtype = source.scalar_type();
    TORCH_CHECK(
        real_dtype == at::kFloat || real_dtype == at::kDouble,
        "field_free_space_jvp supports float32 and float64");
    const c10::ScalarType complex_dtype =
        real_dtype == at::kFloat ? at::kComplexFloat : at::kComplexDouble;
    check_free_space_primal(
        source, target, tx_power, tx_polarization, rx_polarization,
        frequency_hz, real_dtype);
    const int64_t count = source.size(0);
    at::Tensor ts_storage;
    at::Tensor tt_storage;
    const at::Tensor* t_source = optional_grad(
        std::move(tangent_source), ts_storage, "tangent_source",
        real_dtype, {count, 3}, source);
    const at::Tensor* t_target = optional_grad(
        std::move(tangent_target), tt_storage, "tangent_target",
        real_dtype, {count, 3}, source);
    auto complex_options = source.options().dtype(complex_dtype);
    auto t_field_vector = at::empty({count, 3}, complex_options);
    auto t_coefficient = at::empty({count}, complex_options);
    auto t_path_field = at::empty({count}, complex_options);
    auto t_path_gain = at::empty({count}, source.options());
    auto t_path_length = at::empty({count}, source.options());
    auto t_delay = at::empty({count}, source.options());
    auto t_direction = at::empty({count, 3}, source.options());
    if (count > 0) {
        cudaStream_t stream =
            at::cuda::getCurrentCUDAStream(source.get_device()).stream();
        if (real_dtype == at::kFloat) {
            free_space_jvp_kernel<float>
                <<<launch_blocks(count), kBlockSize, 0, stream>>>(
                    count,
                    source.data_ptr<float>(),
                    target.data_ptr<float>(),
                    tx_power.data_ptr<float>(),
                    tx_polarization.data_ptr<float>(),
                    rx_polarization.data_ptr<float>(),
                    static_cast<float>(frequency_hz),
                    static_cast<float>(tangent_frequency),
                    grad_ptr<float>(t_source),
                    grad_ptr<float>(t_target),
                    t_field_vector.data_ptr<c10::complex<float>>(),
                    t_coefficient.data_ptr<c10::complex<float>>(),
                    t_path_field.data_ptr<c10::complex<float>>(),
                    t_path_gain.data_ptr<float>(),
                    t_path_length.data_ptr<float>(),
                    t_delay.data_ptr<float>(),
                    t_direction.data_ptr<float>());
        } else {
            free_space_jvp_kernel<double>
                <<<launch_blocks(count), kBlockSize, 0, stream>>>(
                    count,
                    source.data_ptr<double>(),
                    target.data_ptr<double>(),
                    tx_power.data_ptr<double>(),
                    tx_polarization.data_ptr<double>(),
                    rx_polarization.data_ptr<double>(),
                    frequency_hz,
                    tangent_frequency,
                    grad_ptr<double>(t_source),
                    grad_ptr<double>(t_target),
                    t_field_vector.data_ptr<c10::complex<double>>(),
                    t_coefficient.data_ptr<c10::complex<double>>(),
                    t_path_field.data_ptr<c10::complex<double>>(),
                    t_path_gain.data_ptr<double>(),
                    t_path_length.data_ptr<double>(),
                    t_delay.data_ptr<double>(),
                    t_direction.data_ptr<double>());
        }
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    pybind11::dict out;
    out["field_vector"] = t_field_vector;
    out["coefficient"] = t_coefficient;
    out["path_field"] = t_path_field;
    out["path_gain"] = t_path_gain;
    out["path_length_m"] = t_path_length;
    out["delay_s"] = t_delay;
    out["direction"] = t_direction;
    return out;
}

// ==== Section: Reflection transport ====
#include "field_ad.cuh"

namespace {

// ---------------------------------------------------------------------------
// Reflection sequence (per-bounce eps_r / sigma_e / gain / thickness and
// frequency are differentiable).
// ---------------------------------------------------------------------------

// Fixed per-path chain state recomputed from the primal inputs; shared by the
// backward and jvp kernels so both differentiate the identical forward chain.
struct ReflectionChain {
    transport::ReflectFrame frames[kMaxAdDepth];
    field::Complex3 value_in[kMaxAdDepth];  // field entering each bounce
    field::Complex e_s[kMaxAdDepth];
    field::Complex e_p[kMaxAdDepth];
    field::Complex r_te[kMaxAdDepth];
    field::Complex r_tm[kMaxAdDepth];
    field::Complex3 value_chain;  // pre-propagation
    field::float3a rx_axis;
    field::Complex propagation;
    field::Complex propagation_dfreq;
    field::Complex propagation_dlength;
    float total_length;
    float amplitude_scale;
};

__device__ void reflection_chain_eval(
    int64_t index,
    int64_t depth,
    const float* source,
    const float* target,
    const float* interaction_positions,
    const float* interaction_normals,
    const float* tx_power,
    const float* tx_polarization,
    const float* rx_polarization,
    const float* eps_r,
    const float* sigma_e,
    const float* mu_r,
    const float* gain,
    const float* thickness,
    const int* bounce_material,
    const layered::LayerTables& layers,
    float frequency_hz,
    ReflectionChain& chain) {
    field::float3a previous = load3f(source, index);
    const field::float3a first_hit = load_sequence3f(
        interaction_positions, index, 0, depth);
    field::float3a incident = field::safe_normalize(
        field::f3_sub(first_hit, previous), field::make_f3(0.0f, 0.0f, 1.0f));
    // unnormalized transverse projection of the transmit polarization.
    const field::float3a tx_axis = field::project_to_wedge_plane(
        load3f(tx_polarization, index), incident);
    field::Complex3 value = field::cplx_scale_real(tx_axis, field::cplx(1.0f, 0.0f));
    float total_length = 0.0f;
    field::float3a outgoing = incident;
    for (int64_t bounce = 0; bounce < depth; ++bounce) {
        const field::float3a hit = load_sequence3f(
            interaction_positions, index, bounce, depth);
        incident = field::safe_normalize(field::f3_sub(hit, previous), outgoing);
        total_length += field::safe_length(field::f3_sub(hit, previous));
        const int64_t scalar = index * depth + bounce;
        const transport::ReflectFrame frame = transport::reflect_frame(
            incident, load_sequence3f(interaction_normals, index, bounce, depth));
        field::Complex r_te;
        field::Complex r_tm;
        layered::face_fresnel(
            frame.cos_theta,
            eps_r[scalar],
            sigma_e[scalar],
            mu_r[scalar],
            gain[scalar],
            thickness[scalar],
            frequency_hz,
            bounce_material[scalar],
            layers,
            r_te,
            r_tm);
        const field::Complex e_s = transport::complex3_dot_real(value, frame.s_axis);
        const field::Complex e_p = transport::complex3_dot_real(value, frame.p_in);
        chain.frames[bounce] = frame;
        chain.value_in[bounce] = value;
        chain.e_s[bounce] = e_s;
        chain.e_p[bounce] = e_p;
        chain.r_te[bounce] = r_te;
        chain.r_tm[bounce] = r_tm;
        value = field::c3_add(
            field::cplx_scale_real(frame.s_axis, field::cplx_mul(r_te, e_s)),
            field::cplx_scale_real(frame.p_out, field::cplx_mul(r_tm, e_p)));
        outgoing = frame.reflected_direction;
        previous = hit;
    }
    const field::float3a target_value = load3f(target, index);
    const field::float3a final_offset = field::f3_sub(target_value, previous);
    const field::float3a final_direction = field::safe_normalize(
        final_offset, outgoing);
    total_length += field::safe_length(final_offset);
    const float wave_number =
        2.0f * field::UTD_PI * frequency_hz / transport::kSpeedOfLight;
    const float amplitude = 1.0f /
                            (2.0f * wave_number *
                             fmaxf(total_length, field::UTD_EPS));
    const field::Complex propagation = field::cplx_mul_real(
        field::cplx_exp_phase(
            transport::precise_neg_kd(wave_number, total_length)),
        amplitude);
    chain.value_chain = value;
    // receiver scalar = p_rx . E via the unnormalized transverse of p_rx.
    chain.rx_axis = field::project_to_wedge_plane(
        load3f(rx_polarization, index), final_direction);
    chain.propagation = propagation;
    // dP/df = P * (-1/k - j*L) * (2*pi/c); the phase term uses the raw length
    // (fmod has unit slope) and the amplitude term has no length dependence.
    const field::Complex dlog = field::cplx(-1.0f / wave_number, -total_length);
    chain.propagation_dfreq = field::cplx_mul_real(
        field::cplx_mul(propagation, dlog),
        2.0f * field::UTD_PI / transport::kSpeedOfLight);
    // dP/dL = P * (-[L >= EPS]/L_clamped - j*k): phase over the raw length,
    // amplitude over the clamped length (clamp_min subgradient convention).
    const float length_gate = total_length >= field::UTD_EPS
                                  ? 1.0f / fmaxf(total_length, field::UTD_EPS)
                                  : 0.0f;
    chain.propagation_dlength = field::cplx_mul(
        propagation, field::cplx(-length_gate, -wave_number));
    chain.total_length = total_length;
    chain.amplitude_scale = sqrtf(fmaxf(tx_power[index], 0.0f));
}

__global__ void reflection_sequence_backward_kernel(
    int64_t count,
    int64_t depth,
    const float* source,
    const float* target,
    const float* interaction_positions,
    const float* interaction_normals,
    const float* tx_power,
    const float* tx_polarization,
    const float* rx_polarization,
    const float* eps_r,
    const float* sigma_e,
    const float* mu_r,
    const float* gain,
    const float* thickness,
    const int* bounce_material,
    layered::LayerTables layers,
    float frequency_hz,
    const c10::complex<float>* grad_field_vector,
    const c10::complex<float>* grad_coefficient,
    const c10::complex<float>* grad_path_field,
    const float* grad_path_gain,
    const float* grad_path_length,
    const float* grad_delay,
    const float* grad_direction,
    float* grad_eps_r,
    float* grad_sigma_e,
    float* grad_gain,
    float* grad_thickness,
    float* grad_frequency,
    float* grad_source,
    float* grad_target,
    float* grad_positions,
    float* grad_normals) {
    const bool need_geometry = grad_source != nullptr;
    for (int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < count;
         index += static_cast<int64_t>(blockDim.x) * gridDim.x) {
        ReflectionChain chain;
        reflection_chain_eval(
            index, depth, source, target, interaction_positions,
            interaction_normals, tx_power, tx_polarization, rx_polarization,
            eps_r, sigma_e, mu_r, gain, thickness, bounce_material, layers,
            frequency_hz, chain);

        const field::Complex3 value_final = field::c3_scale(
            chain.value_chain, chain.propagation);
        const field::Complex scalar = transport::complex3_dot_real(
            value_final, chain.rx_axis);
        const field::Complex path_field_value = field::cplx_mul_real(
            scalar, chain.amplitude_scale);
        field::Complex g_scalar = field::cplx_zero();
        field::Complex3 g_value = fold_output_cotangents(
            grad_field_vector, grad_coefficient, grad_path_field, grad_path_gain,
            index, chain.rx_axis, path_field_value, chain.amplitude_scale,
            g_scalar);

        // Split value_final = P * value_chain.
        field::Complex g_propagation = field::cplx_zero();
        field::Complex3 g_chain = field::c3_zero();
        field::adj_cplx_mul(
            chain.value_chain.x, chain.propagation, g_value.x,
            g_chain.x, g_propagation);
        field::adj_cplx_mul(
            chain.value_chain.y, chain.propagation, g_value.y,
            g_chain.y, g_propagation);
        field::adj_cplx_mul(
            chain.value_chain.z, chain.propagation, g_value.z,
            g_chain.z, g_propagation);
        float g_freq = adj_dot(g_propagation, chain.propagation_dfreq);

        // Geometry adjoint state (geometry AD): the total path length
        // cotangent, the cotangent flowing onto the `previous` endpoint of
        // the segment being processed, the alternate-branch cotangent on the
        // bounce's outgoing direction, and the final receive-segment terms.
        const field::float3a e_z = field::make_f3(0.0f, 0.0f, 1.0f);
        float g_length = 0.0f;
        field::float3a g_carry = field::f3_zero();
        field::float3a g_outgoing = field::f3_zero();
        field::float3a g_hit_first = field::f3_zero();
        if (need_geometry) {
            g_length = adj_dot(g_propagation, chain.propagation_dlength);
            if (grad_path_length != nullptr)
                g_length += grad_path_length[index];
            if (grad_delay != nullptr)
                g_length += grad_delay[index] / transport::kSpeedOfLight;
            // rx_axis cotangent from the folded scalar (coefficient =
            // <value_final, rx_axis> with a real axis).
            field::float3a g_rx_axis = field::make_f3(
                field::cplx_adj_dot(g_scalar, value_final.x),
                field::cplx_adj_dot(g_scalar, value_final.y),
                field::cplx_adj_dot(g_scalar, value_final.z));
            const field::float3a previous_last = load_sequence3f(
                interaction_positions, index, depth - 1, depth);
            const field::float3a final_offset = field::f3_sub(
                load3f(target, index), previous_last);
            const field::float3a outgoing_last =
                chain.frames[depth - 1].reflected_direction;
            const field::float3a final_direction = field::safe_normalize(
                final_offset, outgoing_last);
            // The published direction is exactly this final_direction, so an
            // external cotangent seeds the same accumulator the rx transverse
            // projection feeds. adj_safe_normalize then splits it over
            // final_offset and the alternate branch (outgoing_last), which is
            // what carries a direction cotangent back into the bounce chain.
            field::float3a g_final_direction =
                grad_direction != nullptr ? load3f(grad_direction, index)
                                          : field::f3_zero();
            field::float3a g_pol_dump = field::f3_zero();
            ad::adj_transverse_project(
                final_direction, load3f(rx_polarization, index), g_rx_axis,
                g_final_direction, g_pol_dump);
            field::float3a g_final_offset = field::f3_zero();
            field::adj_safe_normalize(
                final_offset, outgoing_last, g_final_direction,
                g_final_offset, g_outgoing);
            ad::adj_safe_length(final_offset, g_length, g_final_offset);
            const int64_t base = index * 3;
            grad_target[base] = g_final_offset.x;
            grad_target[base + 1] = g_final_offset.y;
            grad_target[base + 2] = g_final_offset.z;
            g_carry = field::f3_neg(g_final_offset);
        }

        for (int64_t bounce = depth - 1; bounce >= 0; --bounce) {
            const transport::ReflectFrame& frame = chain.frames[bounce];
            // value_out = s_axis*(r_te*e_s) + p_out*(r_tm*e_p): recover the
            // complex coefficient cotangents and (for geometry) the real
            // basis cotangents in one adjoint step each.
            field::float3a g_s_axis = field::f3_zero();
            field::float3a g_p_in = field::f3_zero();
            field::float3a g_p_out = field::f3_zero();
            field::Complex gs = field::cplx_zero();
            field::Complex gp = field::cplx_zero();
            field::adj_cplx_scale_real(
                frame.s_axis,
                field::cplx_mul(chain.r_te[bounce], chain.e_s[bounce]),
                g_chain, g_s_axis, gs);
            field::adj_cplx_scale_real(
                frame.p_out,
                field::cplx_mul(chain.r_tm[bounce], chain.e_p[bounce]),
                g_chain, g_p_out, gp);
            field::Complex g_r_te = field::cplx_zero();
            field::Complex g_r_tm = field::cplx_zero();
            field::Complex g_e_s = field::cplx_zero();
            field::Complex g_e_p = field::cplx_zero();
            field::adj_cplx_mul(
                chain.r_te[bounce], chain.e_s[bounce], gs, g_r_te, g_e_s);
            field::adj_cplx_mul(
                chain.r_tm[bounce], chain.e_p[bounce], gp, g_r_tm, g_e_p);
            // e_s = <value_in, s_axis>, e_p = <value_in, p_in>.
            field::Complex3 g_value_in = field::c3_zero();
            field::adj_cplx_dot_real(
                chain.value_in[bounce], frame.s_axis, g_e_s, g_value_in,
                g_s_axis);
            field::adj_cplx_dot_real(
                chain.value_in[bounce], frame.p_in, g_e_p, g_value_in, g_p_in);
            g_chain = g_value_in;

            // Convert (g_r_te, g_r_tm) to input gradients with one dual
            // slab_fresnel evaluation per requested basis direction.
            const int64_t scalar_index = index * depth + bounce;
            const float b_eps = eps_r[scalar_index];
            const float b_sigma = sigma_e[scalar_index];
            const float b_mu = mu_r[scalar_index];
            const float b_gain = gain[scalar_index];
            const float b_thickness = thickness[scalar_index];
            const int b_material = bounce_material[scalar_index];
            DualC r_te_dual;
            DualC r_tm_dual;
            if (grad_eps_r != nullptr) {
                layered::face_fresnel_dual(
                    frame.cos_theta, b_eps, b_sigma, b_mu, b_gain, b_thickness,
                    frequency_hz, b_material, layers, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                    r_te_dual, r_tm_dual);
                grad_eps_r[scalar_index] =
                    adj_dot(g_r_te, r_te_dual.d) + adj_dot(g_r_tm, r_tm_dual.d);
            }
            if (grad_sigma_e != nullptr) {
                layered::face_fresnel_dual(
                    frame.cos_theta, b_eps, b_sigma, b_mu, b_gain, b_thickness,
                    frequency_hz, b_material, layers, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f,
                    r_te_dual, r_tm_dual);
                grad_sigma_e[scalar_index] =
                    adj_dot(g_r_te, r_te_dual.d) + adj_dot(g_r_tm, r_tm_dual.d);
            }
            if (grad_gain != nullptr) {
                layered::face_fresnel_dual(
                    frame.cos_theta, b_eps, b_sigma, b_mu, b_gain, b_thickness,
                    frequency_hz, b_material, layers, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f,
                    r_te_dual, r_tm_dual);
                grad_gain[scalar_index] =
                    adj_dot(g_r_te, r_te_dual.d) + adj_dot(g_r_tm, r_tm_dual.d);
            }
            if (grad_thickness != nullptr) {
                layered::face_fresnel_dual(
                    frame.cos_theta, b_eps, b_sigma, b_mu, b_gain, b_thickness,
                    frequency_hz, b_material, layers, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f,
                    r_te_dual, r_tm_dual);
                grad_thickness[scalar_index] =
                    adj_dot(g_r_te, r_te_dual.d) + adj_dot(g_r_tm, r_tm_dual.d);
            }
            if (grad_frequency != nullptr) {
                layered::face_fresnel_dual(
                    frame.cos_theta, b_eps, b_sigma, b_mu, b_gain, b_thickness,
                    frequency_hz, b_material, layers, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f,
                    r_te_dual, r_tm_dual);
                g_freq +=
                    adj_dot(g_r_te, r_te_dual.d) + adj_dot(g_r_tm, r_tm_dual.d);
            }
            if (!need_geometry)
                continue;

            // Geometry enters the Fresnel only through cos_theta.
            layered::face_fresnel_dual(
                frame.cos_theta, b_eps, b_sigma, b_mu, b_gain, b_thickness,
                frequency_hz, b_material, layers, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                r_te_dual, r_tm_dual);
            const float g_cos_theta =
                adj_dot(g_r_te, r_te_dual.d) + adj_dot(g_r_tm, r_tm_dual.d);

            // Replay the segment geometry of this bounce.
            const field::float3a previous =
                bounce > 0 ? load_sequence3f(
                                 interaction_positions, index, bounce - 1, depth)
                           : load3f(source, index);
            const field::float3a hit = load_sequence3f(
                interaction_positions, index, bounce, depth);
            const field::float3a segment = field::f3_sub(hit, previous);
            const field::float3a outgoing_previous =
                bounce > 0 ? chain.frames[bounce - 1].reflected_direction
                           : field::safe_normalize(
                                 field::f3_sub(
                                     load_sequence3f(
                                         interaction_positions, index, 0, depth),
                                     load3f(source, index)),
                                 e_z);
            const field::float3a incident_argument = field::safe_normalize(
                segment, outgoing_previous);
            const field::float3a raw_normal = load_sequence3f(
                interaction_normals, index, bounce, depth);

            // Frame outputs -> incident direction + raw normal. The frozen
            // normal flip contributes only its sign; g_outgoing carries the
            // downstream alternate-branch cotangent of frame.reflected_direction.
            field::float3a g_incident = field::f3_zero();
            field::float3a g_normal_raw = field::f3_zero();
            ad::adj_reflect_frame(
                incident_argument, raw_normal, g_s_axis, g_p_in, g_p_out,
                g_outgoing, g_cos_theta, g_incident, g_normal_raw);
            const int64_t normal_base = scalar_index * 3;
            grad_normals[normal_base] = g_normal_raw.x;
            grad_normals[normal_base + 1] = g_normal_raw.y;
            grad_normals[normal_base + 2] = g_normal_raw.z;

            // incident = safe_normalize(segment, outgoing_previous); the
            // segment also carries the path-length cotangent.
            field::float3a g_segment = field::f3_zero();
            field::float3a g_outgoing_previous = field::f3_zero();
            field::adj_safe_normalize(
                segment, outgoing_previous, g_incident, g_segment,
                g_outgoing_previous);
            ad::adj_safe_length(segment, g_length, g_segment);
            field::float3a g_hit = field::f3_add(g_carry, g_segment);
            g_carry = field::f3_neg(g_segment);
            g_outgoing = g_outgoing_previous;
            if (bounce > 0) {
                const int64_t hit_base = scalar_index * 3;
                grad_positions[hit_base] = g_hit.x;
                grad_positions[hit_base + 1] = g_hit.y;
                grad_positions[hit_base + 2] = g_hit.z;
            } else {
                g_hit_first = g_hit;
            }
        }
        if (grad_frequency != nullptr)
            grad_frequency[index] = g_freq;
        if (!need_geometry)
            continue;

        // Launch segment: value_0 = tx_axis * (1 + 0j) with tx_axis built on
        // the pre-loop incident direction (alternate e_z); the bounce-0
        // normalize falls back onto this same direction, so g_outgoing now
        // holds that branch's cotangent.
        const field::float3a source_value = load3f(source, index);
        const field::float3a first_hit = load_sequence3f(
            interaction_positions, index, 0, depth);
        const field::float3a segment_pre = field::f3_sub(first_hit, source_value);
        const field::float3a incident_pre = field::safe_normalize(segment_pre, e_z);
        field::float3a g_tx_axis = field::make_f3(
            g_chain.x.re, g_chain.y.re, g_chain.z.re);
        field::float3a g_incident_pre = g_outgoing;
        field::float3a g_pol_dump = field::f3_zero();
        ad::adj_transverse_project(
            incident_pre, load3f(tx_polarization, index), g_tx_axis,
            g_incident_pre, g_pol_dump);
        field::float3a g_segment_pre = field::f3_zero();
        field::float3a g_ez_dump = field::f3_zero();
        field::adj_safe_normalize(
            segment_pre, e_z, g_incident_pre, g_segment_pre, g_ez_dump);
        g_hit_first = field::f3_add(g_hit_first, g_segment_pre);
        const field::float3a g_source_total = field::f3_sub(
            g_carry, g_segment_pre);
        const int64_t base = index * 3;
        const int64_t first_base = index * depth * 3;
        grad_positions[first_base] = g_hit_first.x;
        grad_positions[first_base + 1] = g_hit_first.y;
        grad_positions[first_base + 2] = g_hit_first.z;
        grad_source[base] = g_source_total.x;
        grad_source[base + 1] = g_source_total.y;
        grad_source[base + 2] = g_source_total.z;
    }
}

__global__ void reflection_sequence_jvp_kernel(
    int64_t count,
    int64_t depth,
    const float* source,
    const float* target,
    const float* interaction_positions,
    const float* interaction_normals,
    const float* tx_power,
    const float* tx_polarization,
    const float* rx_polarization,
    const float* eps_r,
    const float* sigma_e,
    const float* mu_r,
    const float* gain,
    const float* thickness,
    const int* bounce_material,
    layered::LayerTables layers,
    float frequency_hz,
    const float* tangent_eps_r,
    const float* tangent_sigma_e,
    const float* tangent_gain,
    const float* tangent_thickness,
    float tangent_frequency,
    const float* tangent_source,
    const float* tangent_target,
    const float* tangent_positions,
    const float* tangent_normals,
    c10::complex<float>* t_field_vector,
    c10::complex<float>* t_coefficient,
    c10::complex<float>* t_path_field,
    float* t_path_gain,
    float* t_path_length,
    float* t_delay,
    float* t_direction) {
    for (int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < count;
         index += static_cast<int64_t>(blockDim.x) * gridDim.x) {
        // Full forward-mode dual sweep mirroring reflection_sequence_kernel
        // (and reflection_chain_eval) step by step: the dual vector helpers
        // replay the same utd formulas, so material/frequency-only seeds
        // reduce exactly to the AD tangent chain while geometry seeds move
        // the frames, the incidence cosines and the path length.
        const ad::DualF3 e_z = ad::df3_const(field::make_f3(0.0f, 0.0f, 1.0f));
        ad::DualF3 previous = load_dual3f(source, tangent_source, index);
        const ad::DualF3 first_hit = load_dual_sequence3f(
            interaction_positions, tangent_positions, index, 0, depth);
        const ad::DualF3 incident_pre = ad::dual_safe_normalize(
            ad::df3_sub(first_hit, previous), e_z);
        const ad::DualF3 tx_axis = ad::dual_transverse_project(
            incident_pre, ad::df3_const(load3f(tx_polarization, index)));
        field::Complex3 value = field::cplx_scale_real(
            tx_axis.v, field::cplx(1.0f, 0.0f));
        field::Complex3 d_value = field::cplx_scale_real(
            tx_axis.d, field::cplx(1.0f, 0.0f));
        ad::DualF total_length = {0.0f, 0.0f};
        ad::DualF3 outgoing = incident_pre;
        for (int64_t bounce = 0; bounce < depth; ++bounce) {
            const ad::DualF3 hit = load_dual_sequence3f(
                interaction_positions, tangent_positions, index, bounce, depth);
            const ad::DualF3 segment = ad::df3_sub(hit, previous);
            const ad::DualF3 incident = ad::dual_safe_normalize(
                segment, outgoing);
            const ad::DualF segment_length = ad::dual_safe_length(segment);
            total_length.v += segment_length.v;
            total_length.d += segment_length.d;
            const ad::DualF3 raw_normal = load_dual_sequence3f(
                interaction_normals, tangent_normals, index, bounce, depth);
            const ad::DualReflectFrame frame = ad::dual_reflect_frame(
                incident, raw_normal);
            const int64_t scalar_index = index * depth + bounce;
            DualC r_te_dual;
            DualC r_tm_dual;
            layered::face_fresnel_dual(
                frame.cos_theta.v,
                eps_r[scalar_index],
                sigma_e[scalar_index],
                mu_r[scalar_index],
                gain[scalar_index],
                thickness[scalar_index],
                frequency_hz,
                bounce_material[scalar_index],
                layers,
                frame.cos_theta.d,
                tangent_eps_r != nullptr ? tangent_eps_r[scalar_index] : 0.0f,
                tangent_sigma_e != nullptr ? tangent_sigma_e[scalar_index] : 0.0f,
                tangent_gain != nullptr ? tangent_gain[scalar_index] : 0.0f,
                tangent_thickness != nullptr ? tangent_thickness[scalar_index]
                                             : 0.0f,
                tangent_frequency,
                r_te_dual,
                r_tm_dual);
            const field::Complex e_s = transport::complex3_dot_real(
                value, frame.s_axis.v);
            const field::Complex e_p = transport::complex3_dot_real(
                value, frame.p_in.v);
            const field::Complex d_e_s = field::cplx_add(
                transport::complex3_dot_real(d_value, frame.s_axis.v),
                transport::complex3_dot_real(value, frame.s_axis.d));
            const field::Complex d_e_p = field::cplx_add(
                transport::complex3_dot_real(d_value, frame.p_in.v),
                transport::complex3_dot_real(value, frame.p_in.d));
            const field::Complex w_te = field::cplx_mul(r_te_dual.v, e_s);
            const field::Complex w_tm = field::cplx_mul(r_tm_dual.v, e_p);
            const field::Complex d_w_te = field::cplx_add(
                field::cplx_mul(r_te_dual.d, e_s),
                field::cplx_mul(r_te_dual.v, d_e_s));
            const field::Complex d_w_tm = field::cplx_add(
                field::cplx_mul(r_tm_dual.d, e_p),
                field::cplx_mul(r_tm_dual.v, d_e_p));
            value = field::c3_add(
                field::cplx_scale_real(frame.s_axis.v, w_te),
                field::cplx_scale_real(frame.p_out.v, w_tm));
            d_value = field::c3_add(
                field::c3_add(
                    field::cplx_scale_real(frame.s_axis.d, w_te),
                    field::cplx_scale_real(frame.s_axis.v, d_w_te)),
                field::c3_add(
                    field::cplx_scale_real(frame.p_out.d, w_tm),
                    field::cplx_scale_real(frame.p_out.v, d_w_tm)));
            outgoing = frame.reflected_direction;
            previous = hit;
        }
        const ad::DualF3 target_dual = load_dual3f(target, tangent_target, index);
        const ad::DualF3 final_offset = ad::df3_sub(target_dual, previous);
        const ad::DualF3 final_direction = ad::dual_safe_normalize(
            final_offset, outgoing);
        const ad::DualF final_length = ad::dual_safe_length(final_offset);
        total_length.v += final_length.v;
        total_length.d += final_length.d;
        const float wave_number =
            2.0f * field::UTD_PI * frequency_hz / transport::kSpeedOfLight;
        const float amplitude = 1.0f /
                                (2.0f * wave_number *
                                 fmaxf(total_length.v, field::UTD_EPS));
        const field::Complex propagation = field::cplx_mul_real(
            field::cplx_exp_phase(
                transport::precise_neg_kd(wave_number, total_length.v)),
            amplitude);
        const field::Complex dlog_freq = field::cplx(
            -1.0f / wave_number, -total_length.v);
        const field::Complex propagation_dfreq = field::cplx_mul_real(
            field::cplx_mul(propagation, dlog_freq),
            2.0f * field::UTD_PI / transport::kSpeedOfLight);
        const float length_gate =
            total_length.v >= field::UTD_EPS
                ? 1.0f / fmaxf(total_length.v, field::UTD_EPS)
                : 0.0f;
        const field::Complex propagation_dlength = field::cplx_mul(
            propagation, field::cplx(-length_gate, -wave_number));
        const field::Complex d_propagation = field::cplx_add(
            field::cplx_mul_real(propagation_dfreq, tangent_frequency),
            field::cplx_mul_real(propagation_dlength, total_length.d));
        const ad::DualF3 rx_axis = ad::dual_transverse_project(
            final_direction, ad::df3_const(load3f(rx_polarization, index)));
        const field::Complex3 value_final = field::c3_scale(value, propagation);
        const field::Complex3 d_final = field::c3_add(
            field::c3_scale(d_value, propagation),
            field::c3_scale(value, d_propagation));
        write_output_tangents(
            index, value_final, d_final, rx_axis.v, rx_axis.d,
            sqrtf(fmaxf(tx_power[index], 0.0f)), total_length.d,
            t_field_vector, t_coefficient, t_path_field, t_path_gain,
            t_path_length, t_delay);
        // final_direction is the published direction output; its dual is
        // already built above for the rx transverse projection.
        const int64_t direction_base = index * 3;
        t_direction[direction_base] = final_direction.d.x;
        t_direction[direction_base + 1] = final_direction.d.y;
        t_direction[direction_base + 2] = final_direction.d.z;
    }
}

int64_t check_reflection_primal(
    const at::Tensor& source,
    const at::Tensor& target,
    const at::Tensor& interaction_positions,
    const at::Tensor& interaction_normals,
    const at::Tensor& tx_power,
    const at::Tensor& tx_polarization,
    const at::Tensor& rx_polarization,
    const at::Tensor& eps_r,
    const at::Tensor& sigma_e,
    const at::Tensor& mu_r,
    const at::Tensor& gain,
    const at::Tensor& thickness,
    double frequency_hz) {
    using channel::check_flat_tensor;
    using channel::check_tensor;
    using channel::check_vec3_table;
    check_vec3_table(source, "source");
    check_vec3_table(target, "target");
    check_tensor(interaction_positions, "interaction_positions", at::kFloat, 3);
    check_tensor(interaction_normals, "interaction_normals", at::kFloat, 3);
    check_flat_tensor(tx_power, "tx_power", at::kFloat);
    check_vec3_table(tx_polarization, "tx_polarization");
    check_vec3_table(rx_polarization, "rx_polarization");
    const int64_t count = source.size(0);
    const int64_t depth = interaction_positions.size(1);
    TORCH_CHECK(
        depth > 0 && depth <= kMaxAdDepth && interaction_positions.size(2) == 3,
        "interaction_positions must have shape (N, D, 3) with 0 < D <= ",
        kMaxAdDepth);
    TORCH_CHECK(
        interaction_positions.size(0) == count &&
            interaction_normals.sizes() == interaction_positions.sizes(),
        "interaction tensors must match source rows");
    for (const auto& tensor : {eps_r, sigma_e, mu_r, gain, thickness})
        TORCH_CHECK(
            tensor.scalar_type() == at::kFloat && tensor.is_contiguous() &&
                tensor.dim() == 2 && tensor.size(0) == count &&
                tensor.size(1) == depth,
            "reflection material tensors must be contiguous f32 (N, D)");
    TORCH_CHECK(
        target.size(0) == count && tx_power.size(0) == count &&
            tx_polarization.size(0) == count && rx_polarization.size(0) == count,
        "reflection endpoint tensors must match source rows");
    TORCH_CHECK(frequency_hz > 0.0, "frequency_hz must be positive");
    return depth;
}

}  // namespace

pybind11::dict channel_field_reflection_sequence_backward(
    at::Tensor source,
    at::Tensor target,
    at::Tensor interaction_positions,
    at::Tensor interaction_normals,
    at::Tensor tx_power,
    at::Tensor tx_polarization,
    at::Tensor rx_polarization,
    at::Tensor eps_r,
    at::Tensor sigma_e,
    at::Tensor mu_r,
    at::Tensor gain,
    at::Tensor thickness,
    at::Tensor bounce_material,
    at::Tensor layer_offset,
    at::Tensor layer_count,
    at::Tensor layer_thickness_m,
    at::Tensor layer_eps_r,
    at::Tensor layer_sigma_e,
    at::Tensor layer_mu_r,
    double frequency_hz,
    pybind11::object grad_field_vector,
    pybind11::object grad_coefficient,
    pybind11::object grad_path_field,
    pybind11::object grad_path_gain,
    pybind11::object grad_path_length,
    pybind11::object grad_delay,
    pybind11::object grad_direction,
    bool need_grad_eps_r,
    bool need_grad_sigma_e,
    bool need_grad_gain,
    bool need_grad_thickness,
    bool need_grad_frequency,
    bool need_grad_geometry) {
    const int64_t depth = check_reflection_primal(
        source, target, interaction_positions, interaction_normals, tx_power,
        tx_polarization, rx_polarization, eps_r, sigma_e, mu_r, gain, thickness,
        frequency_hz);
    const layered::LayerTables layers = reflection_layer_tables(
        bounce_material, layer_offset, layer_count, layer_thickness_m, layer_eps_r, layer_sigma_e,
        layer_mu_r, source.size(0), depth);
    const int64_t count = source.size(0);
    at::Tensor gfv_storage;
    at::Tensor gc_storage;
    at::Tensor gpf_storage;
    at::Tensor gpg_storage;
    at::Tensor gpl_storage;
    at::Tensor gd_storage;
    at::Tensor gdir_storage;
    const at::Tensor* gfv = optional_grad(
        std::move(grad_field_vector), gfv_storage, "grad_field_vector",
        at::kComplexFloat, {count, 3}, source);
    const at::Tensor* gc = optional_grad(
        std::move(grad_coefficient), gc_storage, "grad_coefficient",
        at::kComplexFloat, {count}, source);
    const at::Tensor* gpf = optional_grad(
        std::move(grad_path_field), gpf_storage, "grad_path_field",
        at::kComplexFloat, {count}, source);
    const at::Tensor* gpg = optional_grad(
        std::move(grad_path_gain), gpg_storage, "grad_path_gain",
        at::kFloat, {count}, source);
    const at::Tensor* gpl = optional_grad(
        std::move(grad_path_length), gpl_storage, "grad_path_length",
        at::kFloat, {count}, source);
    const at::Tensor* gd = optional_grad(
        std::move(grad_delay), gd_storage, "grad_delay",
        at::kFloat, {count}, source);
    const at::Tensor* gdir = optional_grad(
        std::move(grad_direction), gdir_storage, "grad_direction",
        at::kFloat, {count, 3}, source);

    auto material_grad = [&](bool needed) {
        return needed ? zero_filled({count, depth}, source.options()) : at::Tensor();
    };
    at::Tensor grad_eps = material_grad(need_grad_eps_r);
    at::Tensor grad_sigma = material_grad(need_grad_sigma_e);
    at::Tensor grad_gain_out = material_grad(need_grad_gain);
    at::Tensor grad_thickness_out = material_grad(need_grad_thickness);
    at::Tensor grad_frequency = need_grad_frequency
                                    ? zero_filled({1}, source.options())
                                    : at::Tensor();
    // Per-row frequency cotangents, summed below in a fixed order.
    at::Tensor row_frequency = need_grad_frequency
                                   ? zero_filled({count}, source.options())
                                   : at::Tensor();
    at::Tensor grad_source;
    at::Tensor grad_target;
    at::Tensor grad_positions;
    at::Tensor grad_normals;
    if (need_grad_geometry) {
        grad_source = zero_filled({count, 3}, source.options());
        grad_target = zero_filled({count, 3}, source.options());
        grad_positions = zero_filled({count, depth, 3}, source.options());
        grad_normals = zero_filled({count, depth, 3}, source.options());
    }
    const bool any_grad_in = gfv != nullptr || gc != nullptr || gpf != nullptr ||
                             gpg != nullptr || gpl != nullptr || gd != nullptr ||
                             gdir != nullptr;
    const bool any_grad_out = need_grad_eps_r || need_grad_sigma_e ||
                              need_grad_gain || need_grad_thickness ||
                              need_grad_frequency || need_grad_geometry;
    if (count > 0 && any_grad_in && any_grad_out) {
        cudaStream_t stream =
            at::cuda::getCurrentCUDAStream(source.get_device()).stream();
        reflection_sequence_backward_kernel
            <<<launch_blocks(count), kBlockSize, 0, stream>>>(
                count,
                depth,
                source.data_ptr<float>(),
                target.data_ptr<float>(),
                interaction_positions.data_ptr<float>(),
                interaction_normals.data_ptr<float>(),
                tx_power.data_ptr<float>(),
                tx_polarization.data_ptr<float>(),
                rx_polarization.data_ptr<float>(),
                eps_r.data_ptr<float>(),
                sigma_e.data_ptr<float>(),
                mu_r.data_ptr<float>(),
                gain.data_ptr<float>(),
                thickness.data_ptr<float>(),
                bounce_material.data_ptr<int>(),
                layers,
                static_cast<float>(frequency_hz),
                gfv ? gfv->data_ptr<c10::complex<float>>() : nullptr,
                gc ? gc->data_ptr<c10::complex<float>>() : nullptr,
                gpf ? gpf->data_ptr<c10::complex<float>>() : nullptr,
                grad_ptr<float>(gpg),
                grad_ptr<float>(gpl),
                grad_ptr<float>(gd),
                grad_ptr<float>(gdir),
                need_grad_eps_r ? grad_eps.data_ptr<float>() : nullptr,
                need_grad_sigma_e ? grad_sigma.data_ptr<float>() : nullptr,
                need_grad_gain ? grad_gain_out.data_ptr<float>() : nullptr,
                need_grad_thickness ? grad_thickness_out.data_ptr<float>()
                                    : nullptr,
                need_grad_frequency ? row_frequency.data_ptr<float>() : nullptr,
                need_grad_geometry ? grad_source.data_ptr<float>() : nullptr,
                need_grad_geometry ? grad_target.data_ptr<float>() : nullptr,
                need_grad_geometry ? grad_positions.data_ptr<float>() : nullptr,
                need_grad_geometry ? grad_normals.data_ptr<float>() : nullptr);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        if (need_grad_frequency)
            channel::ordered_sum::add_sum<float>(
                row_frequency.data_ptr<float>(), count, grad_frequency.data_ptr<float>(),
                source.options(), stream);
    }
    pybind11::dict out;
    out["grad_eps_r"] = need_grad_eps_r ? pybind11::cast(grad_eps)
                                        : pybind11::object(pybind11::none());
    out["grad_sigma_e"] = need_grad_sigma_e ? pybind11::cast(grad_sigma)
                                            : pybind11::object(pybind11::none());
    out["grad_gain"] = need_grad_gain ? pybind11::cast(grad_gain_out)
                                      : pybind11::object(pybind11::none());
    out["grad_thickness"] = need_grad_thickness
                                ? pybind11::cast(grad_thickness_out)
                                : pybind11::object(pybind11::none());
    out["grad_frequency"] = need_grad_frequency
                                ? pybind11::cast(grad_frequency)
                                : pybind11::object(pybind11::none());
    out["grad_source"] = need_grad_geometry
                             ? pybind11::cast(grad_source)
                             : pybind11::object(pybind11::none());
    out["grad_target"] = need_grad_geometry
                             ? pybind11::cast(grad_target)
                             : pybind11::object(pybind11::none());
    out["grad_interaction_positions"] =
        need_grad_geometry ? pybind11::cast(grad_positions)
                           : pybind11::object(pybind11::none());
    out["grad_interaction_normals"] =
        need_grad_geometry ? pybind11::cast(grad_normals)
                           : pybind11::object(pybind11::none());
    return out;
}

pybind11::dict channel_field_reflection_sequence_jvp(
    at::Tensor source,
    at::Tensor target,
    at::Tensor interaction_positions,
    at::Tensor interaction_normals,
    at::Tensor tx_power,
    at::Tensor tx_polarization,
    at::Tensor rx_polarization,
    at::Tensor eps_r,
    at::Tensor sigma_e,
    at::Tensor mu_r,
    at::Tensor gain,
    at::Tensor thickness,
    at::Tensor bounce_material,
    at::Tensor layer_offset,
    at::Tensor layer_count,
    at::Tensor layer_thickness_m,
    at::Tensor layer_eps_r,
    at::Tensor layer_sigma_e,
    at::Tensor layer_mu_r,
    double frequency_hz,
    pybind11::object tangent_eps_r,
    pybind11::object tangent_sigma_e,
    pybind11::object tangent_gain,
    pybind11::object tangent_thickness,
    double tangent_frequency,
    pybind11::object tangent_source,
    pybind11::object tangent_target,
    pybind11::object tangent_interaction_positions,
    pybind11::object tangent_interaction_normals) {
    const int64_t depth = check_reflection_primal(
        source, target, interaction_positions, interaction_normals, tx_power,
        tx_polarization, rx_polarization, eps_r, sigma_e, mu_r, gain, thickness,
        frequency_hz);
    const layered::LayerTables layers = reflection_layer_tables(
        bounce_material, layer_offset, layer_count, layer_thickness_m, layer_eps_r, layer_sigma_e,
        layer_mu_r, source.size(0), depth);
    const int64_t count = source.size(0);
    at::Tensor te_storage;
    at::Tensor ts_storage;
    at::Tensor tg_storage;
    at::Tensor tt_storage;
    at::Tensor tsrc_storage;
    at::Tensor ttgt_storage;
    at::Tensor tpos_storage;
    at::Tensor tnrm_storage;
    const at::Tensor* t_eps = optional_grad(
        std::move(tangent_eps_r), te_storage, "tangent_eps_r",
        at::kFloat, {count, depth}, source);
    const at::Tensor* t_sigma = optional_grad(
        std::move(tangent_sigma_e), ts_storage, "tangent_sigma_e",
        at::kFloat, {count, depth}, source);
    const at::Tensor* t_gain = optional_grad(
        std::move(tangent_gain), tg_storage, "tangent_gain",
        at::kFloat, {count, depth}, source);
    const at::Tensor* t_thickness = optional_grad(
        std::move(tangent_thickness), tt_storage, "tangent_thickness",
        at::kFloat, {count, depth}, source);
    const at::Tensor* t_source = optional_grad(
        std::move(tangent_source), tsrc_storage, "tangent_source",
        at::kFloat, {count, 3}, source);
    const at::Tensor* t_target = optional_grad(
        std::move(tangent_target), ttgt_storage, "tangent_target",
        at::kFloat, {count, 3}, source);
    const at::Tensor* t_positions = optional_grad(
        std::move(tangent_interaction_positions), tpos_storage,
        "tangent_interaction_positions", at::kFloat, {count, depth, 3}, source);
    const at::Tensor* t_normals = optional_grad(
        std::move(tangent_interaction_normals), tnrm_storage,
        "tangent_interaction_normals", at::kFloat, {count, depth, 3}, source);

    auto complex_options = source.options().dtype(at::kComplexFloat);
    auto t_field_vector = at::empty({count, 3}, complex_options);
    auto t_coefficient = at::empty({count}, complex_options);
    auto t_path_field = at::empty({count}, complex_options);
    auto t_path_gain = at::empty({count}, source.options());
    auto t_path_length = at::empty({count}, source.options());
    auto t_delay = at::empty({count}, source.options());
    auto t_direction = at::empty({count, 3}, source.options());
    if (count > 0) {
        cudaStream_t stream =
            at::cuda::getCurrentCUDAStream(source.get_device()).stream();
        reflection_sequence_jvp_kernel
            <<<launch_blocks(count), kBlockSize, 0, stream>>>(
                count,
                depth,
                source.data_ptr<float>(),
                target.data_ptr<float>(),
                interaction_positions.data_ptr<float>(),
                interaction_normals.data_ptr<float>(),
                tx_power.data_ptr<float>(),
                tx_polarization.data_ptr<float>(),
                rx_polarization.data_ptr<float>(),
                eps_r.data_ptr<float>(),
                sigma_e.data_ptr<float>(),
                mu_r.data_ptr<float>(),
                gain.data_ptr<float>(),
                thickness.data_ptr<float>(),
                bounce_material.data_ptr<int>(),
                layers,
                static_cast<float>(frequency_hz),
                grad_ptr<float>(t_eps),
                grad_ptr<float>(t_sigma),
                grad_ptr<float>(t_gain),
                grad_ptr<float>(t_thickness),
                static_cast<float>(tangent_frequency),
                grad_ptr<float>(t_source),
                grad_ptr<float>(t_target),
                grad_ptr<float>(t_positions),
                grad_ptr<float>(t_normals),
                t_field_vector.data_ptr<c10::complex<float>>(),
                t_coefficient.data_ptr<c10::complex<float>>(),
                t_path_field.data_ptr<c10::complex<float>>(),
                t_path_gain.data_ptr<float>(),
                t_path_length.data_ptr<float>(),
                t_delay.data_ptr<float>(),
                t_direction.data_ptr<float>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    pybind11::dict out;
    out["field_vector"] = t_field_vector;
    out["coefficient"] = t_coefficient;
    out["path_field"] = t_path_field;
    out["path_gain"] = t_path_gain;
    out["path_length_m"] = t_path_length;
    out["delay_s"] = t_delay;
    out["direction"] = t_direction;
    return out;
}

// ==== Section: Rough-reflection scaling ====
// rough-surface scattering: native rough-surface coherent attenuation C_r and its
// application onto the reflection field outputs.
//
// C_r = prod_b att_b with att_b = exp(-2*(k0*cos_b*sigma_b)^2) on rough
// bounces (else 1), cos_b = |dot(seg_dir_b, n_b)|, seg_dir_b the unit
// direction of the incoming segment (pos_b - prev_b, prev_0 = source). The
// factor is real, so the four reflection outputs scale by C_r (path_gain by
// C_r^2). Rows flagged ``replaced`` (a realization phase screen replaces the
// delta specular) are zeroed. One forward launch scales all four outputs; the
// backward/jvp companions differentiate frequency and the hit geometry
// (positions, normals, source), matching the input set the previous Torch
// implementation reached under the fixed-topology contract.
//
// Elementwise over rows; per-row bounce loop (depth <= 5). No float atomics
// except the scalar frequency-gradient reduction (same convention as the
// other field backward kernels).

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/util/complex.h>
#include "torch_cuda.h"

#include "../tensor_checks.h"

#define launch_blocks rough_scale_launch_blocks
#define kBlockSize kRoughScaleBlockSize
#define zero_filled rough_scale_zero_filled

namespace {

constexpr int kBlockSize = 128;
constexpr int kMaxDepth = 5;
constexpr double kPi = 3.14159265358979323846;
constexpr double kC0 = 299792458.0;

int launch_blocks(int64_t count) {
    return static_cast<int>((count + kBlockSize - 1) / kBlockSize);
}

at::Tensor zero_filled(at::IntArrayRef sizes, const at::TensorOptions& options) {
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

const at::Tensor* optional_arg(
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

using cfloat = c10::complex<float>;

__device__ __forceinline__ cfloat cscale(cfloat value, float s) {
    return cfloat(value.real() * s, value.imag() * s);
}

// Rough-scale inputs of a request whose per-bounce inputs were gathered into (N, D) tables.
struct GatheredRoughBounces {
    const float* positions;
    const float* normals;
    const float* source;
    const float* sigma_b;
    const bool* rough_b;
    int64_t row;
    int depth;

    __device__ __forceinline__ const float* position(int b) const {
        return positions + (row * depth + b) * 3;
    }
    __device__ __forceinline__ const float* normal(int b) const {
        return normals + (row * depth + b) * 3;
    }
    __device__ __forceinline__ const float* origin() const { return source + row * 3; }
    __device__ __forceinline__ float sigma(int b) const { return sigma_b[row * depth + b]; }
    __device__ __forceinline__ bool rough(int b) const { return rough_b[row * depth + b]; }
};

// Per-bounce recomputation shared by forward, backward and jvp. Fills the
// bounce arrays and returns the product factor (before the replaced mask).
template <typename Bounces>
__device__ __forceinline__ float rough_bounce_factor(
    int depth, float k0, const Bounces& bounces,
    float seg_dir[kMaxDepth][3],
    float normal[kMaxDepth][3],
    float sign[kMaxDepth],
    float cos_b[kMaxDepth],
    float inv_len[kMaxDepth],
    bool rough[kMaxDepth],
    float sigma[kMaxDepth]) {
    float factor = 1.0f;
    for (int b = 0; b < depth; ++b) {
        const float* hit = bounces.position(b);
        const float* previous = b == 0 ? bounces.origin() : bounces.position(b - 1);
        const float sx = hit[0] - previous[0];
        const float sy = hit[1] - previous[1];
        const float sz = hit[2] - previous[2];
        const float len = sqrtf(sx * sx + sy * sy + sz * sz);
        // Match Torch's ``seg / norm.clamp_min(1e-9)`` (division, not a
        // reciprocal multiply) so the coherent factor is float-faithful.
        const float denom = fmaxf(len, 1.0e-9f);
        const float dx = sx / denom, dy = sy / denom, dz = sz / denom;
        const float inv = 1.0f / denom;
        const float* face_normal = bounces.normal(b);
        const float nx = face_normal[0];
        const float ny = face_normal[1];
        const float nz = face_normal[2];
        // Torch evaluates (seg_dir*normal).sum(-1) as three rounded products
        // then a left-associated sum; keep the products in registers so the
        // compiler cannot fuse them into an fma with different rounding.
        const float p0 = __fmul_rn(dx, nx);
        const float p1 = __fmul_rn(dy, ny);
        const float p2 = __fmul_rn(dz, nz);
        const float dot = __fadd_rn(__fadd_rn(p0, p1), p2);
        const float cb = fabsf(dot);
        seg_dir[b][0] = dx; seg_dir[b][1] = dy; seg_dir[b][2] = dz;
        normal[b][0] = nx; normal[b][1] = ny; normal[b][2] = nz;
        sign[b] = dot > 0.0f ? 1.0f : (dot < 0.0f ? -1.0f : 0.0f);
        cos_b[b] = cb;
        inv_len[b] = inv;
        const bool r = bounces.rough(b);
        rough[b] = r;
        const float s = bounces.sigma(b);
        sigma[b] = s;
        if (r) {
            // Match the Torch association exp(-2 * (k0*cos*sigma).square):
            // square first, then scale by -2.
            const float u = k0 * cb * s;
            factor *= expf(-2.0f * (u * u));
        }
    }
    return factor;
}

__device__ __forceinline__ float rough_bounces(
    int64_t row, int depth, float k0,
    const float* __restrict__ positions,
    const float* __restrict__ normals,
    const float* __restrict__ source,
    const float* __restrict__ sigma_b,
    const bool* __restrict__ rough_b,
    float seg_dir[kMaxDepth][3],
    float normal[kMaxDepth][3],
    float sign[kMaxDepth],
    float cos_b[kMaxDepth],
    float inv_len[kMaxDepth],
    bool rough[kMaxDepth],
    float sigma[kMaxDepth]) {
    const GatheredRoughBounces bounces = {positions, normals, source, sigma_b, rough_b, row, depth};
    return rough_bounce_factor(
        depth, k0, bounces, seg_dir, normal, sign, cos_b, inv_len, rough, sigma);
}

__global__ void rough_scale_forward_kernel(
    int64_t count, int depth, float k0,
    const cfloat* __restrict__ field_vector,
    const cfloat* __restrict__ coefficient,
    const cfloat* __restrict__ path_field,
    const float* __restrict__ path_gain,
    const float* __restrict__ positions,
    const float* __restrict__ normals,
    const float* __restrict__ source,
    const float* __restrict__ sigma_b,
    const bool* __restrict__ rough_b,
    const bool* __restrict__ replaced,
    cfloat* __restrict__ out_field_vector,
    cfloat* __restrict__ out_coefficient,
    cfloat* __restrict__ out_path_field,
    float* __restrict__ out_path_gain,
    float* __restrict__ out_factor) {
    for (int64_t row = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         row < count; row += static_cast<int64_t>(blockDim.x) * gridDim.x) {
        float seg_dir[kMaxDepth][3], normal[kMaxDepth][3];
        float sign[kMaxDepth], cos_b[kMaxDepth], inv_len[kMaxDepth], sigma[kMaxDepth];
        bool rough[kMaxDepth];
        float factor = rough_bounces(
            row, depth, k0, positions, normals, source, sigma_b, rough_b,
            seg_dir, normal, sign, cos_b, inv_len, rough, sigma);
        if (replaced[row]) factor = 0.0f;
        out_factor[row] = factor;
        for (int c = 0; c < 3; ++c)
            out_field_vector[row * 3 + c] = cscale(field_vector[row * 3 + c], factor);
        out_coefficient[row] = cscale(coefficient[row], factor);
        out_path_field[row] = cscale(path_field[row], factor);
        out_path_gain[row] = path_gain[row] * factor * factor;
    }
}

__global__ void rough_scale_backward_kernel(
    int64_t count, int depth, float k0, float dk0_df,
    const cfloat* __restrict__ field_vector,
    const cfloat* __restrict__ coefficient,
    const cfloat* __restrict__ path_field,
    const float* __restrict__ path_gain,
    const float* __restrict__ positions,
    const float* __restrict__ normals,
    const float* __restrict__ source,
    const float* __restrict__ sigma_b,
    const bool* __restrict__ rough_b,
    const bool* __restrict__ replaced,
    const cfloat* __restrict__ grad_field_vector,
    const cfloat* __restrict__ grad_coefficient,
    const cfloat* __restrict__ grad_path_field,
    const float* __restrict__ grad_path_gain,
    cfloat* __restrict__ out_grad_field_vector,
    cfloat* __restrict__ out_grad_coefficient,
    cfloat* __restrict__ out_grad_path_field,
    float* __restrict__ out_grad_path_gain,
    float* __restrict__ out_grad_positions,
    float* __restrict__ out_grad_normals,
    float* __restrict__ out_grad_source,
    float* __restrict__ out_grad_frequency,
    bool need_field, bool need_geometry, bool need_frequency) {
    for (int64_t row = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         row < count; row += static_cast<int64_t>(blockDim.x) * gridDim.x) {
        float seg_dir[kMaxDepth][3], normal[kMaxDepth][3];
        float sign[kMaxDepth], cos_b[kMaxDepth], inv_len[kMaxDepth], sigma[kMaxDepth];
        bool rough[kMaxDepth];
        float factor = rough_bounces(
            row, depth, k0, positions, normals, source, sigma_b, rough_b,
            seg_dir, normal, sign, cos_b, inv_len, rough, sigma);
        const bool rep = replaced[row];
        if (rep) factor = 0.0f;

        // grad_factor = sum_outputs Re(conj(cotangent) * primal); path_gain
        // (out = pg*factor^2) adds cotangent * 2 * factor * pg.
        float grad_factor = 0.0f;
        for (int c = 0; c < 3; ++c) {
            const cfloat in = field_vector[row * 3 + c];
            const cfloat g = grad_field_vector != nullptr ? grad_field_vector[row * 3 + c]
                                                          : cfloat(0.0f, 0.0f);
            grad_factor += g.real() * in.real() + g.imag() * in.imag();
            if (need_field)
                out_grad_field_vector[row * 3 + c] = cscale(g, factor);
        }
        {
            const cfloat in = coefficient[row];
            const cfloat g = grad_coefficient != nullptr ? grad_coefficient[row]
                                                         : cfloat(0.0f, 0.0f);
            grad_factor += g.real() * in.real() + g.imag() * in.imag();
            if (need_field) out_grad_coefficient[row] = cscale(g, factor);
        }
        {
            const cfloat in = path_field[row];
            const cfloat g = grad_path_field != nullptr ? grad_path_field[row]
                                                        : cfloat(0.0f, 0.0f);
            grad_factor += g.real() * in.real() + g.imag() * in.imag();
            if (need_field) out_grad_path_field[row] = cscale(g, factor);
        }
        {
            const float in = path_gain[row];
            const float g = grad_path_gain != nullptr ? grad_path_gain[row] : 0.0f;
            grad_factor += g * 2.0f * factor * in;
            if (need_field) out_grad_path_gain[row] = g * factor * factor;
        }

        if (need_geometry) {
            float gpos[kMaxDepth][3];
            for (int b = 0; b < depth; ++b) {
                gpos[b][0] = gpos[b][1] = gpos[b][2] = 0.0f;
            }
            float gsrc[3] = {0.0f, 0.0f, 0.0f};
            for (int b = 0; b < depth; ++b) {
                float gn0 = 0.0f, gn1 = 0.0f, gn2 = 0.0f;
                if (!rep && rough[b]) {
                    // A_b = grad_factor * d(factor)/d(cos_b)
                    // = grad_factor * factor * (-4 k0^2 sigma_b^2 cos_b).
                    const float A = grad_factor * factor *
                        (-4.0f * k0 * k0 * sigma[b] * sigma[b] * cos_b[b]);
                    const float s = sign[b];
                    // grad_normal_b = A * sign * seg_dir_b.
                    gn0 = A * s * seg_dir[b][0];
                    gn1 = A * s * seg_dir[b][1];
                    gn2 = A * s * seg_dir[b][2];
                    // d(cos)/d(seg) = sign*(n - dot*seg_dir)*inv (len>clamp) or
                    // sign*n*inv when the norm was clamped.
                    const float dot = cos_b[b] * s;  // signed dot
                    float dsx, dsy, dsz;
                    if (inv_len[b] < (1.0f / 1.0e-9f)) {
                        dsx = s * (normal[b][0] - dot * seg_dir[b][0]) * inv_len[b];
                        dsy = s * (normal[b][1] - dot * seg_dir[b][1]) * inv_len[b];
                        dsz = s * (normal[b][2] - dot * seg_dir[b][2]) * inv_len[b];
                    } else {
                        dsx = s * normal[b][0] * inv_len[b];
                        dsy = s * normal[b][1] * inv_len[b];
                        dsz = s * normal[b][2] * inv_len[b];
                    }
                    const float gsx = A * dsx, gsy = A * dsy, gsz = A * dsz;
                    gpos[b][0] += gsx; gpos[b][1] += gsy; gpos[b][2] += gsz;
                    if (b == 0) {
                        gsrc[0] -= gsx; gsrc[1] -= gsy; gsrc[2] -= gsz;
                    } else {
                        gpos[b - 1][0] -= gsx; gpos[b - 1][1] -= gsy; gpos[b - 1][2] -= gsz;
                    }
                }
                const int64_t nb = (row * depth + b) * 3;
                out_grad_normals[nb + 0] = gn0;
                out_grad_normals[nb + 1] = gn1;
                out_grad_normals[nb + 2] = gn2;
            }
            for (int b = 0; b < depth; ++b) {
                const int64_t pb = (row * depth + b) * 3;
                out_grad_positions[pb + 0] = gpos[b][0];
                out_grad_positions[pb + 1] = gpos[b][1];
                out_grad_positions[pb + 2] = gpos[b][2];
            }
            out_grad_source[row * 3 + 0] = gsrc[0];
            out_grad_source[row * 3 + 1] = gsrc[1];
            out_grad_source[row * 3 + 2] = gsrc[2];
        }

        if (need_frequency && !rep) {
            float df = 0.0f;
            for (int b = 0; b < depth; ++b) {
                if (rough[b]) {
                    // d(factor)/d(k0) contribution of bounce b, times dk0/df.
                    df += factor *
                        (-4.0f * k0 * cos_b[b] * cos_b[b] * sigma[b] * sigma[b]) * dk0_df;
                }
            }
            out_grad_frequency[row] = grad_factor * df;
        }
    }
}

__global__ void rough_scale_jvp_kernel(
    int64_t count, int depth, float k0, float dk0_df,
    const cfloat* __restrict__ field_vector,
    const cfloat* __restrict__ coefficient,
    const cfloat* __restrict__ path_field,
    const float* __restrict__ path_gain,
    const float* __restrict__ positions,
    const float* __restrict__ normals,
    const float* __restrict__ source,
    const float* __restrict__ sigma_b,
    const bool* __restrict__ rough_b,
    const bool* __restrict__ replaced,
    const cfloat* __restrict__ t_field_vector,
    const cfloat* __restrict__ t_coefficient,
    const cfloat* __restrict__ t_path_field,
    const float* __restrict__ t_path_gain,
    const float* __restrict__ t_positions,
    const float* __restrict__ t_normals,
    const float* __restrict__ t_source,
    float t_frequency,
    cfloat* __restrict__ out_t_field_vector,
    cfloat* __restrict__ out_t_coefficient,
    cfloat* __restrict__ out_t_path_field,
    float* __restrict__ out_t_path_gain) {
    for (int64_t row = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         row < count; row += static_cast<int64_t>(blockDim.x) * gridDim.x) {
        float seg_dir[kMaxDepth][3], normal[kMaxDepth][3];
        float sign[kMaxDepth], cos_b[kMaxDepth], inv_len[kMaxDepth], sigma[kMaxDepth];
        bool rough[kMaxDepth];
        float factor = rough_bounces(
            row, depth, k0, positions, normals, source, sigma_b, rough_b,
            seg_dir, normal, sign, cos_b, inv_len, rough, sigma);
        const bool rep = replaced[row];
        float dfactor = 0.0f;
        if (!rep) {
            for (int b = 0; b < depth; ++b) {
                if (!rough[b]) continue;
                const float s = sign[b];
                const float dot = cos_b[b] * s;
                // d(cos_b) along geometry tangents.
                float tsegx = 0.0f, tsegy = 0.0f, tsegz = 0.0f;
                const int64_t pb = (row * depth + b) * 3;
                if (t_positions != nullptr) {
                    tsegx = t_positions[pb + 0];
                    tsegy = t_positions[pb + 1];
                    tsegz = t_positions[pb + 2];
                }
                if (b == 0) {
                    if (t_source != nullptr) {
                        tsegx -= t_source[row * 3 + 0];
                        tsegy -= t_source[row * 3 + 1];
                        tsegz -= t_source[row * 3 + 2];
                    }
                } else if (t_positions != nullptr) {
                    const int64_t prev = (row * depth + b - 1) * 3;
                    tsegx -= t_positions[prev + 0];
                    tsegy -= t_positions[prev + 1];
                    tsegz -= t_positions[prev + 2];
                }
                float dcos = 0.0f;
                if (inv_len[b] < (1.0f / 1.0e-9f)) {
                    dcos = s * (
                        (normal[b][0] - dot * seg_dir[b][0]) * tsegx +
                        (normal[b][1] - dot * seg_dir[b][1]) * tsegy +
                        (normal[b][2] - dot * seg_dir[b][2]) * tsegz) * inv_len[b];
                } else {
                    dcos = s * (normal[b][0] * tsegx + normal[b][1] * tsegy +
                                normal[b][2] * tsegz) * inv_len[b];
                }
                if (t_normals != nullptr) {
                    dcos += s * (seg_dir[b][0] * t_normals[pb + 0] +
                                 seg_dir[b][1] * t_normals[pb + 1] +
                                 seg_dir[b][2] * t_normals[pb + 2]);
                }
                // d(factor)/d(cos_b) * dcos.
                dfactor += factor *
                    (-4.0f * k0 * k0 * sigma[b] * sigma[b] * cos_b[b]) * dcos;
                // d(factor)/d(f) * t_frequency.
                dfactor += factor *
                    (-4.0f * k0 * cos_b[b] * cos_b[b] * sigma[b] * sigma[b]) *
                    dk0_df * t_frequency;
            }
        } else {
            factor = 0.0f;
        }
        for (int c = 0; c < 3; ++c) {
            const cfloat in = field_vector[row * 3 + c];
            cfloat t = cscale(in, dfactor);
            if (t_field_vector != nullptr)
                t += cscale(t_field_vector[row * 3 + c], factor);
            out_t_field_vector[row * 3 + c] = t;
        }
        {
            const cfloat in = coefficient[row];
            cfloat t = cscale(in, dfactor);
            if (t_coefficient != nullptr) t += cscale(t_coefficient[row], factor);
            out_t_coefficient[row] = t;
        }
        {
            const cfloat in = path_field[row];
            cfloat t = cscale(in, dfactor);
            if (t_path_field != nullptr) t += cscale(t_path_field[row], factor);
            out_t_path_field[row] = t;
        }
        {
            const float in = path_gain[row];
            float t = in * 2.0f * factor * dfactor;
            if (t_path_gain != nullptr) t += t_path_gain[row] * factor * factor;
            out_t_path_gain[row] = t;
        }
    }
}

void check_inputs(
    const at::Tensor& field_vector,
    const at::Tensor& coefficient,
    const at::Tensor& path_field,
    const at::Tensor& path_gain,
    const at::Tensor& positions,
    const at::Tensor& normals,
    const at::Tensor& source,
    const at::Tensor& sigma_b,
    const at::Tensor& rough_b,
    const at::Tensor& replaced,
    int64_t& count,
    int& depth) {
    using channel::check_tensor;
    check_tensor(field_vector, "field_vector", at::kComplexFloat, 2);
    TORCH_CHECK(field_vector.size(1) == 3, "field_vector must have shape (R, 3)");
    count = field_vector.size(0);
    check_tensor(coefficient, "coefficient", at::kComplexFloat, 1);
    check_tensor(path_field, "path_field", at::kComplexFloat, 1);
    check_tensor(path_gain, "path_gain", at::kFloat, 1);
    check_tensor(positions, "positions", at::kFloat, 3);
    check_tensor(normals, "normals", at::kFloat, 3);
    check_tensor(source, "source", at::kFloat, 2);
    check_tensor(sigma_b, "sigma_b", at::kFloat, 2);
    check_tensor(rough_b, "rough_b", at::kBool, 2);
    check_tensor(replaced, "replaced", at::kBool, 1);
    depth = static_cast<int>(positions.size(1));
    TORCH_CHECK(depth >= 1 && depth <= kMaxDepth, "depth must be in [1, 5]");
    TORCH_CHECK(
        coefficient.size(0) == count && path_field.size(0) == count &&
            path_gain.size(0) == count && positions.size(0) == count &&
            normals.size(0) == count && source.size(0) == count &&
            sigma_b.size(0) == count && rough_b.size(0) == count &&
            replaced.size(0) == count,
        "rough-scale row counts must match field_vector");
    TORCH_CHECK(
        positions.size(2) == 3 && normals.size(1) == depth &&
            normals.size(2) == 3 && source.size(1) == 3 &&
            sigma_b.size(1) == depth && rough_b.size(1) == depth,
        "rough-scale per-bounce shapes are inconsistent");
    for (const auto& t : {coefficient, path_field, path_gain, positions, normals,
                          source, sigma_b, rough_b, replaced}) {
        TORCH_CHECK(t.get_device() == field_vector.get_device(),
                    "rough-scale tensors must share device");
    }
}

}  // namespace

pybind11::dict channel_field_rough_reflection_scale(
    at::Tensor field_vector,
    at::Tensor coefficient,
    at::Tensor path_field,
    at::Tensor path_gain,
    at::Tensor positions,
    at::Tensor normals,
    at::Tensor source,
    at::Tensor sigma_b,
    at::Tensor rough_b,
    at::Tensor replaced,
    double frequency_hz) {
    int64_t count = 0;
    int depth = 0;
    check_inputs(field_vector, coefficient, path_field, path_gain, positions,
                 normals, source, sigma_b, rough_b, replaced, count, depth);
    const float k0 = static_cast<float>(2.0 * kPi * frequency_hz / kC0);
    auto out_field_vector = at::empty_like(field_vector);
    auto out_coefficient = at::empty_like(coefficient);
    auto out_path_field = at::empty_like(path_field);
    auto out_path_gain = at::empty_like(path_gain);
    auto out_factor = at::empty({count}, path_gain.options());
    if (count > 0) {
        cudaStream_t stream =
            at::cuda::getCurrentCUDAStream(field_vector.get_device()).stream();
        rough_scale_forward_kernel<<<launch_blocks(count), kBlockSize, 0, stream>>>(
            count, depth, k0,
            field_vector.data_ptr<cfloat>(), coefficient.data_ptr<cfloat>(),
            path_field.data_ptr<cfloat>(), path_gain.data_ptr<float>(),
            positions.data_ptr<float>(), normals.data_ptr<float>(),
            source.data_ptr<float>(), sigma_b.data_ptr<float>(),
            rough_b.data_ptr<bool>(), replaced.data_ptr<bool>(),
            out_field_vector.data_ptr<cfloat>(), out_coefficient.data_ptr<cfloat>(),
            out_path_field.data_ptr<cfloat>(), out_path_gain.data_ptr<float>(),
            out_factor.data_ptr<float>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    pybind11::dict out;
    out["field_vector"] = out_field_vector;
    out["coefficient"] = out_coefficient;
    out["path_field"] = out_path_field;
    out["path_gain"] = out_path_gain;
    out["factor"] = out_factor;
    return out;
}

pybind11::dict channel_field_rough_reflection_scale_backward(
    at::Tensor field_vector,
    at::Tensor coefficient,
    at::Tensor path_field,
    at::Tensor path_gain,
    at::Tensor positions,
    at::Tensor normals,
    at::Tensor source,
    at::Tensor sigma_b,
    at::Tensor rough_b,
    at::Tensor replaced,
    double frequency_hz,
    pybind11::object grad_field_vector,
    pybind11::object grad_coefficient,
    pybind11::object grad_path_field,
    pybind11::object grad_path_gain,
    bool need_field,
    bool need_geometry,
    bool need_frequency) {
    int64_t count = 0;
    int depth = 0;
    check_inputs(field_vector, coefficient, path_field, path_gain, positions,
                 normals, source, sigma_b, rough_b, replaced, count, depth);
    const float k0 = static_cast<float>(2.0 * kPi * frequency_hz / kC0);
    const float dk0_df = static_cast<float>(2.0 * kPi / kC0);
    at::Tensor storage[4];
    const at::Tensor* g_fv = optional_arg(
        std::move(grad_field_vector), storage[0], "grad_field_vector",
        at::kComplexFloat, {count, 3}, field_vector);
    const at::Tensor* g_coef = optional_arg(
        std::move(grad_coefficient), storage[1], "grad_coefficient",
        at::kComplexFloat, {count}, field_vector);
    const at::Tensor* g_pf = optional_arg(
        std::move(grad_path_field), storage[2], "grad_path_field",
        at::kComplexFloat, {count}, field_vector);
    const at::Tensor* g_pg = optional_arg(
        std::move(grad_path_gain), storage[3], "grad_path_gain",
        at::kFloat, {count}, field_vector);

    at::Tensor grad_field_vector_out, grad_coefficient_out, grad_path_field_out,
        grad_path_gain_out, grad_positions, grad_normals, grad_source,
        grad_frequency;
    if (need_field) {
        grad_field_vector_out = at::empty_like(field_vector);
        grad_coefficient_out = at::empty_like(coefficient);
        grad_path_field_out = at::empty_like(path_field);
        grad_path_gain_out = at::empty_like(path_gain);
    }
    if (need_geometry) {
        grad_positions = zero_filled({count, depth, 3}, positions.options());
        grad_normals = zero_filled({count, depth, 3}, normals.options());
        grad_source = zero_filled({count, 3}, source.options());
    }
    at::Tensor row_frequency;
    if (need_frequency) {
        grad_frequency = zero_filled({1}, path_gain.options());
        // Per-row frequency cotangents, summed below in a fixed order.
        row_frequency = zero_filled({count}, path_gain.options());
    }
    const bool any_grad =
        g_fv != nullptr || g_coef != nullptr || g_pf != nullptr || g_pg != nullptr;
    if (count > 0 && any_grad) {
        cudaStream_t stream =
            at::cuda::getCurrentCUDAStream(field_vector.get_device()).stream();
        rough_scale_backward_kernel<<<launch_blocks(count), kBlockSize, 0, stream>>>(
            count, depth, k0, dk0_df,
            field_vector.data_ptr<cfloat>(), coefficient.data_ptr<cfloat>(),
            path_field.data_ptr<cfloat>(), path_gain.data_ptr<float>(),
            positions.data_ptr<float>(), normals.data_ptr<float>(),
            source.data_ptr<float>(), sigma_b.data_ptr<float>(),
            rough_b.data_ptr<bool>(), replaced.data_ptr<bool>(),
            opt_ptr<cfloat>(g_fv), opt_ptr<cfloat>(g_coef), opt_ptr<cfloat>(g_pf),
            opt_ptr<float>(g_pg),
            need_field ? grad_field_vector_out.data_ptr<cfloat>() : nullptr,
            need_field ? grad_coefficient_out.data_ptr<cfloat>() : nullptr,
            need_field ? grad_path_field_out.data_ptr<cfloat>() : nullptr,
            need_field ? grad_path_gain_out.data_ptr<float>() : nullptr,
            need_geometry ? grad_positions.data_ptr<float>() : nullptr,
            need_geometry ? grad_normals.data_ptr<float>() : nullptr,
            need_geometry ? grad_source.data_ptr<float>() : nullptr,
            need_frequency ? row_frequency.data_ptr<float>() : nullptr,
            need_field, need_geometry, need_frequency);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        if (need_frequency)
            channel::ordered_sum::add_sum<float>(
                row_frequency.data_ptr<float>(), count, grad_frequency.data_ptr<float>(),
                path_gain.options(), stream);
    }
    pybind11::dict out;
    out["grad_field_vector"] =
        need_field ? pybind11::cast(grad_field_vector_out) : pybind11::object(pybind11::none());
    out["grad_coefficient"] =
        need_field ? pybind11::cast(grad_coefficient_out) : pybind11::object(pybind11::none());
    out["grad_path_field"] =
        need_field ? pybind11::cast(grad_path_field_out) : pybind11::object(pybind11::none());
    out["grad_path_gain"] =
        need_field ? pybind11::cast(grad_path_gain_out) : pybind11::object(pybind11::none());
    out["grad_positions"] =
        need_geometry ? pybind11::cast(grad_positions) : pybind11::object(pybind11::none());
    out["grad_normals"] =
        need_geometry ? pybind11::cast(grad_normals) : pybind11::object(pybind11::none());
    out["grad_source"] =
        need_geometry ? pybind11::cast(grad_source) : pybind11::object(pybind11::none());
    out["grad_frequency"] =
        need_frequency ? pybind11::cast(grad_frequency) : pybind11::object(pybind11::none());
    return out;
}

pybind11::dict channel_field_rough_reflection_scale_jvp(
    at::Tensor field_vector,
    at::Tensor coefficient,
    at::Tensor path_field,
    at::Tensor path_gain,
    at::Tensor positions,
    at::Tensor normals,
    at::Tensor source,
    at::Tensor sigma_b,
    at::Tensor rough_b,
    at::Tensor replaced,
    double frequency_hz,
    pybind11::object tangent_field_vector,
    pybind11::object tangent_coefficient,
    pybind11::object tangent_path_field,
    pybind11::object tangent_path_gain,
    pybind11::object tangent_positions,
    pybind11::object tangent_normals,
    pybind11::object tangent_source,
    double tangent_frequency) {
    int64_t count = 0;
    int depth = 0;
    check_inputs(field_vector, coefficient, path_field, path_gain, positions,
                 normals, source, sigma_b, rough_b, replaced, count, depth);
    const float k0 = static_cast<float>(2.0 * kPi * frequency_hz / kC0);
    const float dk0_df = static_cast<float>(2.0 * kPi / kC0);
    at::Tensor storage[7];
    const at::Tensor* t_fv = optional_arg(
        std::move(tangent_field_vector), storage[0], "tangent_field_vector",
        at::kComplexFloat, {count, 3}, field_vector);
    const at::Tensor* t_coef = optional_arg(
        std::move(tangent_coefficient), storage[1], "tangent_coefficient",
        at::kComplexFloat, {count}, field_vector);
    const at::Tensor* t_pf = optional_arg(
        std::move(tangent_path_field), storage[2], "tangent_path_field",
        at::kComplexFloat, {count}, field_vector);
    const at::Tensor* t_pg = optional_arg(
        std::move(tangent_path_gain), storage[3], "tangent_path_gain",
        at::kFloat, {count}, field_vector);
    const at::Tensor* t_pos = optional_arg(
        std::move(tangent_positions), storage[4], "tangent_positions",
        at::kFloat, {count, depth, 3}, field_vector);
    const at::Tensor* t_nrm = optional_arg(
        std::move(tangent_normals), storage[5], "tangent_normals",
        at::kFloat, {count, depth, 3}, field_vector);
    const at::Tensor* t_src = optional_arg(
        std::move(tangent_source), storage[6], "tangent_source",
        at::kFloat, {count, 3}, field_vector);

    auto out_field_vector = at::empty_like(field_vector);
    auto out_coefficient = at::empty_like(coefficient);
    auto out_path_field = at::empty_like(path_field);
    auto out_path_gain = at::empty_like(path_gain);
    if (count > 0) {
        cudaStream_t stream =
            at::cuda::getCurrentCUDAStream(field_vector.get_device()).stream();
        rough_scale_jvp_kernel<<<launch_blocks(count), kBlockSize, 0, stream>>>(
            count, depth, k0, dk0_df,
            field_vector.data_ptr<cfloat>(), coefficient.data_ptr<cfloat>(),
            path_field.data_ptr<cfloat>(), path_gain.data_ptr<float>(),
            positions.data_ptr<float>(), normals.data_ptr<float>(),
            source.data_ptr<float>(), sigma_b.data_ptr<float>(),
            rough_b.data_ptr<bool>(), replaced.data_ptr<bool>(),
            opt_ptr<cfloat>(t_fv), opt_ptr<cfloat>(t_coef), opt_ptr<cfloat>(t_pf),
            opt_ptr<float>(t_pg), opt_ptr<float>(t_pos), opt_ptr<float>(t_nrm),
            opt_ptr<float>(t_src), static_cast<float>(tangent_frequency),
            out_field_vector.data_ptr<cfloat>(), out_coefficient.data_ptr<cfloat>(),
            out_path_field.data_ptr<cfloat>(), out_path_gain.data_ptr<float>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    pybind11::dict out;
    out["tangent_field_vector"] = out_field_vector;
    out["tangent_coefficient"] = out_coefficient;
    out["tangent_path_field"] = out_path_field;
    out["tangent_path_gain"] = out_path_gain;
    return out;
}

#undef launch_blocks
#undef kBlockSize
#undef zero_filled

// ==== Section: Source amplitude ====
// source excitation: source-amplitude application onto a transported complex3 field.
//
// The field transport kernels publish two families on one launch: the
// unit-excitation pair (``field_vector``, ``coefficient``) and the excited
// pair (``path_field = coefficient * sqrt(tx_power)``, ``path_gain``). There
// is no excited complex3 vector, so the public consumer complex3 response had
// no power-carrying quantity to publish. This owner supplies exactly that
// missing output:
//
// path_field_vector = field_vector * sqrt(max(tx_power, 0))
//
// with the identical ``sqrtf(fmaxf(tx_power, 0))`` amplitude expression the
// transport kernels use, so ``<path_field_vector, rx_axis>`` and
// ``path_field`` are the same quantity. They are not required to be
// bit-identical: this owner scales and the caller then projects, while the
// transport kernel projects and then scales, so the two differ by float
// rounding order. The map is linear in the field vector and the
// amplitude is real, so the VJP and JVP are the same scale.
//
// ``tx_power`` is a frozen primal here exactly as it is in every field
// transport companion: no gradient or tangent is produced for it, and the
// Python wrappers reject a request for one.
//
// Elementwise over rows, one launch, no reduction and no atomics.

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/util/complex.h>
#include "torch_cuda.h"

#include "../tensor_checks.h"

#define launch_blocks source_amplitude_launch_blocks
#define kBlockSize kSourceAmplitudeBlockSize

namespace {

constexpr int kBlockSize = 128;

int launch_blocks(int64_t count) {
    return static_cast<int>((count + kBlockSize - 1) / kBlockSize);
}

using cfloat = c10::complex<float>;

__global__ void source_amplitude_scale_kernel(
    int64_t count,
    const cfloat* __restrict__ field_vector,
    const float* __restrict__ tx_power,
    cfloat* __restrict__ out_field_vector) {
    for (int64_t row = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         row < count; row += static_cast<int64_t>(blockDim.x) * gridDim.x) {
        const float amplitude = sqrtf(fmaxf(tx_power[row], 0.0f));
        for (int c = 0; c < 3; ++c) {
            const cfloat value = field_vector[row * 3 + c];
            out_field_vector[row * 3 + c] =
                cfloat(value.real() * amplitude, value.imag() * amplitude);
        }
    }
}

void check_power(const at::Tensor& tx_power, int64_t count, const at::Tensor& reference) {
    channel::check_tensor(tx_power, "tx_power", at::kFloat, 1);
    TORCH_CHECK(
        tx_power.size(0) == count,
        "tx_power must have one row per complex3 field row");
    TORCH_CHECK(
        tx_power.get_device() == reference.get_device(),
        "tx_power must share the field device");
}

int64_t check_field(const at::Tensor& field_vector, const char* name) {
    channel::check_tensor(field_vector, name, at::kComplexFloat, 2);
    TORCH_CHECK(field_vector.size(1) == 3, name, " must have shape (N, 3)");
    return field_vector.size(0);
}

at::Tensor scaled(const at::Tensor& field_vector, const at::Tensor& tx_power) {
    const int64_t count = field_vector.size(0);
    auto out = at::empty_like(field_vector);
    if (count > 0) {
        cudaStream_t stream =
            at::cuda::getCurrentCUDAStream(field_vector.get_device()).stream();
        source_amplitude_scale_kernel<<<launch_blocks(count), kBlockSize, 0, stream>>>(
            count,
            field_vector.data_ptr<cfloat>(),
            tx_power.data_ptr<float>(),
            out.data_ptr<cfloat>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return out;
}

}  // namespace

pybind11::dict channel_field_source_amplitude_scale(
    at::Tensor field_vector,
    at::Tensor tx_power) {
    // Autograd hands cotangents and tangents in as strided views; the scale is
    // elementwise, so a canonical contiguous view is the whole staging cost.
    field_vector = field_vector.contiguous();
    tx_power = tx_power.contiguous();
    const int64_t count = check_field(field_vector, "field_vector");
    check_power(tx_power, count, field_vector);
    pybind11::dict out;
    out["path_field_vector"] = scaled(field_vector, tx_power);
    return out;
}

pybind11::dict channel_field_source_amplitude_scale_backward(
    at::Tensor tx_power,
    at::Tensor grad_path_field_vector) {
    grad_path_field_vector = grad_path_field_vector.contiguous();
    tx_power = tx_power.contiguous();
    const int64_t count = check_field(grad_path_field_vector, "grad_path_field_vector");
    check_power(tx_power, count, grad_path_field_vector);
    pybind11::dict out;
    out["grad_field_vector"] = scaled(grad_path_field_vector, tx_power);
    return out;
}

pybind11::dict channel_field_source_amplitude_scale_jvp(
    at::Tensor tx_power,
    at::Tensor tangent_field_vector) {
    tangent_field_vector = tangent_field_vector.contiguous();
    tx_power = tx_power.contiguous();
    const int64_t count = check_field(tangent_field_vector, "tangent_field_vector");
    check_power(tx_power, count, tangent_field_vector);
    pybind11::dict out;
    out["tangent_path_field_vector"] = scaled(tangent_field_vector, tx_power);
    return out;
}

#undef launch_blocks
#undef kBlockSize

// ==== Section: Canonical-row field stage ====
// One launch evaluates every field-stage row whose operator lives in this translation unit:
// line of sight, specular reflection of every depth (finite slab or layer stack, with the
// rough-surface coherent factor) and the receiver projection of order-1 diffraction. Each
// thread reads its row's endpoints, interaction tables and per-face materials directly from
// the canonical tables, runs the same per-row arithmetic as the per-operator kernels above,
// and writes the seven field-stage columns in place. Rows of any other kind keep their input
// values for the operators that run after this launch.

#define launch_blocks canonical_field_launch_blocks
#define kBlockSize kCanonicalFieldBlockSize

namespace {

constexpr int kBlockSize = 256;

int launch_blocks(int64_t count) {
    return static_cast<int>((count + kBlockSize - 1) / kBlockSize);
}

constexpr int kLineOfSightComponent = 0;
constexpr int kReflectionComponent = 1;
constexpr int kDiffractionComponent = 2;

struct CanonicalRows {
    int64_t count;
    int64_t width;
    const int64_t* order;
    const int* component_id;
    const int* depth;
    const int* tx_id;
    const int* rx_id;
    const int* primitive_sequence;
    const float* tx_positions;
    const float* tx_power;
    const float* tx_polarization;
    int64_t tx_count;
    const float* rx_positions;
    const float* rx_polarization;
    int64_t rx_count;
    const float* interaction_positions;
    const float* interaction_normals;
    const cfloat* field_xyz;
    const cfloat* coefficient;
    const cfloat* path_field;
    const float* path_gain;
    const float* path_length;
    const float* delay;
    const float* direction;
    bool evaluate_los;
    float frequency_hz;
};

struct FaceTables {
    const float* eps_r;
    const float* sigma_e;
    const float* mu_r;
    const float* gain;
    const float* thickness;
    const int* material;
    int64_t count;
    layered::LayerTables layers;
    bool multilayer;
    const float* rough_sigma;
    const bool* rough;
    const bool* replaced;
    float rough_k0;
};

// Index semantics of a Torch gather: a negative index counts from the end. Returns -1 when the
// index is outside the table.
__device__ __forceinline__ int64_t table_index(int64_t index, int64_t count) {
    const int64_t wrapped = index < 0 ? index + count : index;
    return wrapped >= 0 && wrapped < count ? wrapped : -1;
}

__device__ __forceinline__ int64_t bounce_face(
    const CanonicalRows& rows, const FaceTables& faces, int64_t row, int64_t bounce) {
    return table_index(rows.primitive_sequence[row * rows.width + bounce], faces.count);
}

struct FaceRoughBounces {
    const CanonicalRows& rows;
    const FaceTables& faces;
    int64_t row;
    const float* source;

    __device__ __forceinline__ const float* position(int b) const {
        return rows.interaction_positions + (row * rows.width + b) * 3;
    }
    __device__ __forceinline__ const float* normal(int b) const {
        return rows.interaction_normals + (row * rows.width + b) * 3;
    }
    __device__ __forceinline__ const float* origin() const { return source; }
    __device__ __forceinline__ float sigma(int b) const {
        return faces.rough == nullptr ? 0.0f : faces.rough_sigma[bounce_face(rows, faces, row, b)];
    }
    __device__ __forceinline__ bool rough(int b) const {
        return faces.rough != nullptr && faces.rough[bounce_face(rows, faces, row, b)];
    }
};

// Reflection row through the shared reflection lane, with slab parameters read from the per-face
// tables. ``factor`` is the rough-surface coherent factor applied as the rough-scale kernel applies
// it; a smooth row passes 1, which scales exactly.
__device__ __forceinline__ void store_reflection_row(
    const CanonicalRows& rows, const FaceTables& faces, const TransportOutputs& out, int64_t index,
    int64_t tx, int64_t rx, int64_t depth, float factor) {
    ReflectionRow row;
    reflection_row(
        row, channel::math::load_field_vec3(rows.tx_positions, tx),
        channel::math::load_field_vec3(rows.rx_positions, rx), rows.tx_power[tx],
        channel::math::load_field_vec3(rows.tx_polarization, tx),
        channel::math::load_field_vec3(rows.rx_polarization, rx),
        rows.interaction_positions + index * rows.width * 3,
        rows.interaction_normals + index * rows.width * 3, faces.eps_r, faces.sigma_e, faces.mu_r,
        faces.gain, faces.thickness, faces.multilayer ? faces.material : nullptr,
        rows.primitive_sequence + index * rows.width, faces.count, 0, depth, faces.layers,
        rows.frequency_hz);
    const int64_t base = index * 3;
    out.field_vector[base] = cscale(to_complex(row.value.x), factor);
    out.field_vector[base + 1] = cscale(to_complex(row.value.y), factor);
    out.field_vector[base + 2] = cscale(to_complex(row.value.z), factor);
    out.path_field[index] = cscale(to_complex(row.received), factor);
    out.coefficient[index] = cscale(to_complex(row.scalar), factor);
    out.path_gain[index] = row.path_gain * factor * factor;
    out.path_length[index] = row.length;
    out.delay[index] = row.delay;
    out.direction[base] = row.direction.x;
    out.direction[base + 1] = row.direction.y;
    out.direction[base + 2] = row.direction.z;
}

__device__ __forceinline__ void store_input_row(
    const CanonicalRows& rows, const TransportOutputs& out, int64_t row) {
    for (int c = 0; c < 3; ++c) {
        out.field_vector[row * 3 + c] = rows.field_xyz[row * 3 + c];
        out.direction[row * 3 + c] = rows.direction[row * 3 + c];
    }
    out.coefficient[row] = rows.coefficient[row];
    out.path_field[row] = rows.path_field[row];
    out.path_gain[row] = rows.path_gain[row];
    out.path_length[row] = rows.path_length[row];
    out.delay[row] = rows.delay[row];
}

// A row whose endpoint or face index is outside its table publishes inert zeros.
__device__ __forceinline__ void store_inert_row(const TransportOutputs& out, int64_t row) {
    for (int c = 0; c < 3; ++c) {
        out.field_vector[row * 3 + c] = cfloat(0.0f, 0.0f);
        out.direction[row * 3 + c] = 0.0f;
    }
    out.coefficient[row] = cfloat(0.0f, 0.0f);
    out.path_field[row] = cfloat(0.0f, 0.0f);
    out.path_gain[row] = 0.0f;
    out.path_length[row] = 0.0f;
    out.delay[row] = 0.0f;
}

// Receiver projection of a diffraction row whose source-excited UTD field was exported by
// discovery: arrival direction from the edge point, projection, and the unit-source columns.
__device__ __forceinline__ void store_diffraction_row(
    const CanonicalRows& rows, const TransportOutputs& out, int64_t row, int64_t tx, int64_t rx) {
    const float* target = rows.rx_positions + rx * 3;
    const float* edge_point = rows.interaction_positions + row * rows.width * 3;
    const float x = target[0] - edge_point[0];
    const float y = target[1] - edge_point[1];
    const float z = target[2] - edge_point[2];
    const float length = fmaxf(sqrtf(x * x + y * y + z * z), 1.0e-6f);
    const field::float3a arrival = field::make_f3(x / length, y / length, z / length);
    const field::Complex3 value = {
        from_complex(rows.field_xyz[row * 3]),
        from_complex(rows.field_xyz[row * 3 + 1]),
        from_complex(rows.field_xyz[row * 3 + 2]),
    };
    const field::Complex scalar = transport::project_receiver(
        value, arrival, channel::math::load_field_vec3(rows.rx_polarization, rx));
    // Torch semantics of ``power.clamp_min(1e-30).sqrt()`` and a complex-by-real division.
    const float power = rows.tx_power[tx];
    const cfloat amplitude(sqrtf(isnan(power) ? power : fmaxf(power, 1.0e-30f)), 0.0f);
    for (int c = 0; c < 3; ++c) {
        out.field_vector[row * 3 + c] = rows.field_xyz[row * 3 + c] / amplitude;
    }
    out.direction[row * 3] = arrival.x;
    out.direction[row * 3 + 1] = arrival.y;
    out.direction[row * 3 + 2] = arrival.z;
    out.path_field[row] = to_complex(scalar);
    out.coefficient[row] = to_complex(scalar) / amplitude;
    out.path_gain[row] = field::cplx_abs_sqr(scalar);
    out.path_length[row] = rows.path_length[row];
    out.delay[row] = rows.delay[row];
}

__global__ void canonical_field_kernel(CanonicalRows rows, FaceTables faces, TransportOutputs out) {
    for (int64_t lane = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         lane < rows.count;
         lane += static_cast<int64_t>(blockDim.x) * gridDim.x) {
        const int64_t row = rows.order == nullptr ? lane : rows.order[lane];
        const int component = rows.component_id[row];
        const int64_t depth = rows.depth[row];
        const bool los = component == kLineOfSightComponent && rows.evaluate_los;
        const bool reflection =
            component == kReflectionComponent && depth >= 1 && depth <= rows.width;
        const bool diffraction = component == kDiffractionComponent && rows.width >= 1;
        if (!los && !reflection && !diffraction) {
            store_input_row(rows, out, row);
            continue;
        }
        const int64_t tx = table_index(rows.tx_id[row], rows.tx_count);
        const int64_t rx = table_index(rows.rx_id[row], rows.rx_count);
        bool inert = tx < 0 || rx < 0;
        for (int64_t bounce = 0; reflection && !inert && bounce < depth; ++bounce) {
            inert = bounce_face(rows, faces, row, bounce) < 0;
        }
        if (inert) {
            store_inert_row(out, row);
            continue;
        }
        if (diffraction) {
            store_diffraction_row(rows, out, row, tx, rx);
            continue;
        }
        const field::float3a source = channel::math::load_field_vec3(rows.tx_positions, tx);
        const field::float3a target = channel::math::load_field_vec3(rows.rx_positions, rx);
        const field::float3a tx_pol = channel::math::load_field_vec3(rows.tx_polarization, tx);
        const field::float3a rx_pol = channel::math::load_field_vec3(rows.rx_polarization, rx);
        if (los) {
            store_transport_lane(
                out, row,
                free_space_lane(source, target, rows.tx_power[tx], tx_pol, rx_pol, rows.frequency_hz));
            continue;
        }
        float factor = 1.0f;
        if (faces.rough != nullptr || faces.replaced != nullptr) {
            float seg_dir[kMaxDepth][3], normal[kMaxDepth][3];
            float sign[kMaxDepth], cos_b[kMaxDepth], inv_len[kMaxDepth], sigma[kMaxDepth];
            bool rough[kMaxDepth];
            const FaceRoughBounces rough_bounces_of_row = {
                rows, faces, row, rows.tx_positions + tx * 3};
            factor = rough_bounce_factor(
                static_cast<int>(depth), faces.rough_k0, rough_bounces_of_row, seg_dir, normal,
                sign, cos_b, inv_len, rough, sigma);
            if (depth == 1 && faces.replaced != nullptr &&
                faces.replaced[bounce_face(rows, faces, row, 0)]) {
                factor = 0.0f;
            }
        }
        store_reflection_row(rows, faces, out, row, tx, rx, depth, factor);
    }
}

}  // namespace

pybind11::dict channel_field_canonical_rows(
    at::Tensor order,
    at::Tensor component_id,
    at::Tensor depth,
    at::Tensor tx_id,
    at::Tensor rx_id,
    at::Tensor primitive_sequence,
    at::Tensor tx_positions,
    at::Tensor tx_power,
    at::Tensor tx_polarization,
    at::Tensor rx_positions,
    at::Tensor rx_polarization,
    at::Tensor interaction_positions,
    at::Tensor interaction_normals,
    at::Tensor field_xyz,
    at::Tensor coefficient,
    at::Tensor path_field,
    at::Tensor path_gain,
    at::Tensor path_length,
    at::Tensor delay,
    at::Tensor direction,
    at::Tensor face_eps_r,
    at::Tensor face_sigma_e,
    at::Tensor face_mu_r,
    at::Tensor face_gain,
    at::Tensor face_thickness,
    at::Tensor face_material,
    at::Tensor layer_offset,
    at::Tensor layer_count,
    at::Tensor layer_thickness_m,
    at::Tensor layer_eps_r,
    at::Tensor layer_sigma_e,
    at::Tensor layer_mu_r,
    bool multilayer,
    at::Tensor face_rough_sigma,
    at::Tensor face_rough,
    at::Tensor face_replaced,
    bool rough,
    bool replace_specular,
    bool evaluate_los,
    double frequency_hz) {
    using channel::check_flat_tensor;
    using channel::check_tensor;
    using channel::check_vec3_table;
    const int64_t count = component_id.size(0);
    check_flat_tensor(order, "order", at::kLong);
    TORCH_CHECK(order.size(0) == 0 || order.size(0) == count,
                "order must be empty or list every row");
    for (const auto& named : std::vector<std::pair<at::Tensor, const char*>>{
             {component_id, "component_id"}, {depth, "depth"}, {tx_id, "tx_id"}, {rx_id, "rx_id"}}) {
        check_flat_tensor(named.first, named.second, at::kInt);
        TORCH_CHECK(named.first.size(0) == count, named.second, " must list every row");
    }
    check_tensor(primitive_sequence, "primitive_sequence", at::kInt, 2);
    const int64_t width = primitive_sequence.size(1);
    TORCH_CHECK(primitive_sequence.size(0) == count, "primitive_sequence must list every row");
    for (const auto& named : std::vector<std::pair<at::Tensor, const char*>>{
             {interaction_positions, "interaction_positions"},
             {interaction_normals, "interaction_normals"}}) {
        check_tensor(named.first, named.second, at::kFloat, 3);
        TORCH_CHECK(named.first.size(0) == count && named.first.size(1) == width &&
                        named.first.size(2) == 3,
                    named.second, " must have shape (N, W, 3)");
    }
    check_vec3_table(tx_positions, "tx_positions");
    check_vec3_table(tx_polarization, "tx_polarization");
    check_flat_tensor(tx_power, "tx_power", at::kFloat);
    TORCH_CHECK(tx_power.size(0) == tx_positions.size(0) &&
                    tx_polarization.size(0) == tx_positions.size(0),
                "transmitter tables must share one length");
    check_vec3_table(rx_positions, "rx_positions");
    check_vec3_table(rx_polarization, "rx_polarization");
    TORCH_CHECK(rx_polarization.size(0) == rx_positions.size(0),
                "receiver tables must share one length");
    check_tensor(field_xyz, "field_xyz", at::kComplexFloat, 2);
    TORCH_CHECK(field_xyz.size(0) == count && field_xyz.size(1) == 3,
                "field_xyz must have shape (N, 3)");
    check_vec3_table(direction, "direction");
    TORCH_CHECK(direction.size(0) == count, "direction must list every row");
    for (const auto& named : std::vector<std::pair<at::Tensor, const char*>>{
             {coefficient, "coefficient"}, {path_field, "path_field"}}) {
        check_flat_tensor(named.first, named.second, at::kComplexFloat);
        TORCH_CHECK(named.first.size(0) == count, named.second, " must list every row");
    }
    for (const auto& named : std::vector<std::pair<at::Tensor, const char*>>{
             {path_gain, "path_gain"}, {path_length, "path_length"}, {delay, "delay"}}) {
        check_flat_tensor(named.first, named.second, at::kFloat);
        TORCH_CHECK(named.first.size(0) == count, named.second, " must list every row");
    }
    const int64_t face_count = face_eps_r.size(0);
    for (const auto& named : std::vector<std::pair<at::Tensor, const char*>>{
             {face_eps_r, "face_eps_r"}, {face_sigma_e, "face_sigma_e"}, {face_mu_r, "face_mu_r"},
             {face_gain, "face_gain"}, {face_thickness, "face_thickness"}}) {
        check_flat_tensor(named.first, named.second, at::kFloat);
        TORCH_CHECK(named.first.size(0) == face_count, named.second, " must list every face");
    }
    check_flat_tensor(face_material, "face_material", at::kInt);
    TORCH_CHECK(face_material.size(0) == face_count, "face_material must list every face");
    TORCH_CHECK(frequency_hz > 0.0, "frequency_hz must be positive");

    FaceTables faces = {};
    faces.eps_r = face_eps_r.data_ptr<float>();
    faces.sigma_e = face_sigma_e.data_ptr<float>();
    faces.mu_r = face_mu_r.data_ptr<float>();
    faces.gain = face_gain.data_ptr<float>();
    faces.thickness = face_thickness.data_ptr<float>();
    faces.material = face_material.data_ptr<int>();
    faces.count = face_count;
    faces.multilayer = multilayer;
    if (multilayer) {
        faces.layers = reflection_layer_tables(
            face_material.reshape({face_count, 1}), layer_offset, layer_count, layer_thickness_m,
            layer_eps_r, layer_sigma_e, layer_mu_r, face_count, 1);
    }
    if (rough || replace_specular) {
        TORCH_CHECK(width <= kMaxDepth, "rough reflection supports at most ", kMaxDepth, " bounces");
        faces.rough_k0 = static_cast<float>(2.0 * kPi * frequency_hz / kC0);
    }
    if (rough) {
        check_flat_tensor(face_rough_sigma, "face_rough_sigma", at::kFloat);
        check_flat_tensor(face_rough, "face_rough", at::kBool);
        TORCH_CHECK(face_rough_sigma.size(0) == face_count && face_rough.size(0) == face_count,
                    "rough face tables must list every face");
        faces.rough_sigma = face_rough_sigma.data_ptr<float>();
        faces.rough = face_rough.data_ptr<bool>();
    }
    if (replace_specular) {
        check_flat_tensor(face_replaced, "face_replaced", at::kBool);
        TORCH_CHECK(face_replaced.size(0) == face_count, "face_replaced must list every face");
        faces.replaced = face_replaced.data_ptr<bool>();
    }

    CanonicalRows rows = {};
    rows.count = count;
    rows.width = width;
    rows.order = order.size(0) == 0 ? nullptr : order.data_ptr<int64_t>();
    rows.component_id = component_id.data_ptr<int>();
    rows.depth = depth.data_ptr<int>();
    rows.tx_id = tx_id.data_ptr<int>();
    rows.rx_id = rx_id.data_ptr<int>();
    rows.primitive_sequence = primitive_sequence.data_ptr<int>();
    rows.tx_positions = tx_positions.data_ptr<float>();
    rows.tx_power = tx_power.data_ptr<float>();
    rows.tx_polarization = tx_polarization.data_ptr<float>();
    rows.tx_count = tx_positions.size(0);
    rows.rx_positions = rx_positions.data_ptr<float>();
    rows.rx_polarization = rx_polarization.data_ptr<float>();
    rows.rx_count = rx_positions.size(0);
    rows.interaction_positions = interaction_positions.data_ptr<float>();
    rows.interaction_normals = interaction_normals.data_ptr<float>();
    rows.field_xyz = field_xyz.data_ptr<cfloat>();
    rows.coefficient = coefficient.data_ptr<cfloat>();
    rows.path_field = path_field.data_ptr<cfloat>();
    rows.path_gain = path_gain.data_ptr<float>();
    rows.path_length = path_length.data_ptr<float>();
    rows.delay = delay.data_ptr<float>();
    rows.direction = direction.data_ptr<float>();
    rows.evaluate_los = evaluate_los;
    rows.frequency_hz = static_cast<float>(frequency_hz);

    auto out_field_xyz = at::empty_like(field_xyz);
    auto out_coefficient = at::empty_like(coefficient);
    auto out_path_field = at::empty_like(path_field);
    auto out_path_gain = at::empty_like(path_gain);
    auto out_path_length = at::empty_like(path_length);
    auto out_delay = at::empty_like(delay);
    auto out_direction = at::empty_like(direction);
    if (count > 0) {
        cudaStream_t stream = at::cuda::getCurrentCUDAStream(field_xyz.get_device()).stream();
        canonical_field_kernel<<<launch_blocks(count), kBlockSize, 0, stream>>>(
            rows, faces,
            TransportOutputs{
                out_field_xyz.data_ptr<cfloat>(), out_coefficient.data_ptr<cfloat>(),
                out_path_field.data_ptr<cfloat>(), out_path_gain.data_ptr<float>(),
                out_path_length.data_ptr<float>(), out_delay.data_ptr<float>(),
                out_direction.data_ptr<float>()});
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    pybind11::dict result;
    result["field_xyz"] = out_field_xyz;
    result["coefficient"] = out_coefficient;
    result["path_field"] = out_path_field;
    result["path_gain"] = out_path_gain;
    result["path_length"] = out_path_length;
    result["delay"] = out_delay;
    result["direction"] = out_direction;
    return result;
}

#undef launch_blocks
#undef kBlockSize
