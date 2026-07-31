// Copyright Xingyu Chen.
// Implements field transport CUDA operations.

// ==== Section: Core field transport ====
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/util/complex.h>
#include "torch_cuda.h"

#include "math.cuh"
#include <rayd/field_transport.cuh>
#include "../tensor_checks.h"

#include <vector>

#define launch_blocks field_transport_launch_blocks
#define kBlockSize kFieldTransportBlockSize

namespace {

constexpr int kBlockSize = 256;
namespace field = rayd::shared::diffraction;
namespace transport = rayd::shared::field_transport;



__device__ __forceinline__ c10::complex<float> to_complex(field::Complex value) {
    return c10::complex<float>(value.re, value.im);
}

__device__ __forceinline__ field::Complex from_complex(c10::complex<float> value) {
    return field::cplx(value.real(), value.imag());
}

__global__ void free_space_kernel(
    int64_t count,
    const float* source,
    const float* target,
    const float* tx_power,
    const float* tx_polarization,
    const float* rx_polarization,
    float frequency_hz,
    c10::complex<float>* field_vector,
    c10::complex<float>* coefficient,
    c10::complex<float>* path_field,
    float* path_gain,
    float* path_length,
    float* delay,
    float* direction_out) {
    for (int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < count;
         index += static_cast<int64_t>(blockDim.x) * gridDim.x) {
        const field::float3a source_value = channel::math::load_field_vec3(source, index);
        const field::float3a target_value = channel::math::load_field_vec3(target, index);
        const field::float3a offset = field::f3_sub(target_value, source_value);
        const float distance = field::safe_length(offset);
        const field::float3a direction = field::safe_normalize(
            offset, field::make_f3(0.0f, 0.0f, 1.0f));
        const field::float3a tx_pol = channel::math::load_field_vec3(tx_polarization, index);
        const float wave_number =
            2.0f * field::UTD_PI * frequency_hz / transport::kSpeedOfLight;
        const float source_amplitude = sqrtf(fmaxf(tx_power[index], 0.0f));
        const field::Complex3 value = transport::free_space_complex3(
            source_value, target_value, wave_number, tx_pol);
        const int64_t base = index * 3;
        field_vector[base] = to_complex(value.x);
        field_vector[base + 1] = to_complex(value.y);
        field_vector[base + 2] = to_complex(value.z);
        const field::Complex scalar = transport::project_receiver(
            value, direction, channel::math::load_field_vec3(rx_polarization, index));
        coefficient[index] = to_complex(scalar);
        const field::Complex received = field::cplx_mul_real(scalar, source_amplitude);
        path_field[index] = to_complex(received);
        path_gain[index] = field::cplx_abs_sqr(received);
        path_length[index] = distance;
        delay[index] = distance / transport::kSpeedOfLight;
        direction_out[base] = direction.x;
        direction_out[base + 1] = direction.y;
        direction_out[base + 2] = direction.z;
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
    float frequency_hz,
    c10::complex<float>* field_vector,
    c10::complex<float>* coefficient,
    c10::complex<float>* path_field,
    float* path_gain,
    float* path_length,
    float* delay,
    float* direction_out) {
    for (int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < count;
         index += static_cast<int64_t>(blockDim.x) * gridDim.x) {
        field::float3a previous = channel::math::load_field_vec3(source, index);
        field::float3a first_hit = load_sequence3(
            interaction_positions, index, 0, depth);
        field::float3a incident = field::safe_normalize(
            field::f3_sub(first_hit, previous), field::make_f3(0.0f, 0.0f, 1.0f));
        // unnormalized transverse projection of the transmit polarization
        // (short-dipole sin(theta) weight); the Jones s/p bases stay orthonormal.
        field::float3a tx_axis = field::project_to_wedge_plane(
            channel::math::load_field_vec3(tx_polarization, index), incident);
        field::Complex3 value = field::cplx_scale_real(
            tx_axis, field::cplx(1.0f, 0.0f));
        float total_length = 0.0f;
        field::float3a outgoing = incident;
        for (int64_t bounce = 0; bounce < depth; ++bounce) {
            const field::float3a hit = load_sequence3(
                interaction_positions, index, bounce, depth);
            incident = field::safe_normalize(
                field::f3_sub(hit, previous), outgoing);
            total_length += field::safe_length(field::f3_sub(hit, previous));
            const int64_t scalar = index * depth + bounce;
            value = transport::reflect_complex3(
                value,
                incident,
                load_sequence3(interaction_normals, index, bounce, depth),
                eps_r[scalar],
                sigma_e[scalar],
                mu_r[scalar],
                gain[scalar],
                thickness[scalar],
                frequency_hz,
                outgoing);
            previous = hit;
        }
        const field::float3a target_value = channel::math::load_field_vec3(target, index);
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
        value = field::c3_scale(value, propagation);
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
        path_length[index] = total_length;
        delay[index] = total_length / transport::kSpeedOfLight;
        direction_out[base] = final_direction.x;
        direction_out[base + 1] = final_direction.y;
        direction_out[base + 2] = final_direction.z;
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
            field::Complex3 reflected = transport::reflect_complex3(
                field::cplx_scale_real(tx_axis, field::cplx(1.0f, 0.0f)),
                source_to_hit,
                normal,
                reflection_eps_r[index],
                reflection_sigma_e[index],
                reflection_mu_r[index],
                reflection_gain[index],
                reflection_thickness[index],
                frequency_hz,
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
        pair.face0Operator = transport::slab_face_operator(
            fabsf(field::f3_dot(pair.n0, incident_direction)),
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
            fabsf(field::f3_dot(pair.nn, incident_direction)),
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
            value = transport::reflect_complex3(
                value,
                field::safe_normalize(field::f3_sub(hit, edge), outgoing_direction),
                normal,
                reflection_eps_r[index],
                reflection_sigma_e[index],
                reflection_mu_r[index],
                reflection_gain[index],
                reflection_thickness[index],
                frequency_hz,
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
        pair1.face0Operator = transport::slab_face_operator(
            fabsf(field::f3_dot(pair1.n0, incident_direction1)),
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
            fabsf(field::f3_dot(pair1.nn, incident_direction1)),
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
        pair2.face0Operator = transport::slab_face_operator(
            fabsf(field::f3_dot(pair2.n0, incident_direction2)),
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
            fabsf(field::f3_dot(pair2.nn, incident_direction2)),
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

pybind11::dict channel_field_free_space(
    at::Tensor source,
    at::Tensor target,
    at::Tensor tx_power,
    at::Tensor tx_polarization,
    at::Tensor rx_polarization,
    double frequency_hz) {
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
    TORCH_CHECK(frequency_hz > 0.0, "frequency_hz must be positive");

    auto complex_options = source.options().dtype(at::kComplexFloat);
    auto field_vector = at::empty({count, 3}, complex_options);
    auto coefficient = at::empty({count}, complex_options);
    auto path_field = at::empty({count}, complex_options);
    auto path_gain = at::empty({count}, source.options());
    auto path_length = at::empty_like(path_gain);
    auto delay = at::empty_like(path_gain);
    auto direction = at::empty_like(source);
    if (count > 0) {
        cudaStream_t stream = at::cuda::getCurrentCUDAStream(source.get_device()).stream();
        free_space_kernel<<<launch_blocks(count), kBlockSize, 0, stream>>>(
            count,
            source.data_ptr<float>(),
            target.data_ptr<float>(),
            tx_power.data_ptr<float>(),
            tx_polarization.data_ptr<float>(),
            rx_polarization.data_ptr<float>(),
            static_cast<float>(frequency_hz),
            field_vector.data_ptr<c10::complex<float>>(),
            coefficient.data_ptr<c10::complex<float>>(),
            path_field.data_ptr<c10::complex<float>>(),
            path_gain.data_ptr<float>(),
            path_length.data_ptr<float>(),
            delay.data_ptr<float>(),
            direction.data_ptr<float>());
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
    TORCH_CHECK(frequency_hz > 0.0, "frequency_hz must be positive");

    auto complex_options = source.options().dtype(at::kComplexFloat);
    auto field_vector = at::empty({count, 3}, complex_options);
    auto coefficient = at::empty({count}, complex_options);
    auto path_field = at::empty({count}, complex_options);
    auto path_gain = at::empty({count}, source.options());
    auto path_length = at::empty_like(path_gain);
    auto delay = at::empty_like(path_gain);
    auto direction = at::empty_like(source);
    if (count > 0) {
        cudaStream_t stream = at::cuda::getCurrentCUDAStream(source.get_device()).stream();
        reflection_sequence_kernel<<<launch_blocks(count), kBlockSize, 0, stream>>>(
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
            static_cast<float>(frequency_hz),
            field_vector.data_ptr<c10::complex<float>>(),
            coefficient.data_ptr<c10::complex<float>>(),
            path_field.data_ptr<c10::complex<float>>(),
            path_gain.data_ptr<float>(),
            path_length.data_ptr<float>(),
            delay.data_ptr<float>(),
            direction.data_ptr<float>());
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
            atomicAdd(grad_frequency, g_freq);
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
                    need_grad_frequency ? grad_frequency.data_ptr<float>()
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
                    need_grad_frequency ? grad_frequency.data_ptr<double>()
                                        : nullptr,
                    need_grad_geometry ? grad_source.data_ptr<double>() : nullptr,
                    need_grad_geometry ? grad_target.data_ptr<double>()
                                       : nullptr);
        }
        C10_CUDA_KERNEL_LAUNCH_CHECK();
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
        transport::slab_fresnel(
            frame.cos_theta,
            eps_r[scalar],
            sigma_e[scalar],
            mu_r[scalar],
            gain[scalar],
            thickness[scalar],
            frequency_hz,
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
            eps_r, sigma_e, mu_r, gain, thickness, frequency_hz, chain);

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
            DualC r_te_dual;
            DualC r_tm_dual;
            if (grad_eps_r != nullptr) {
                ad::slab_fresnel_dual(
                    frame.cos_theta, b_eps, b_sigma, b_mu, b_gain, b_thickness,
                    frequency_hz, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                    r_te_dual, r_tm_dual);
                grad_eps_r[scalar_index] =
                    adj_dot(g_r_te, r_te_dual.d) + adj_dot(g_r_tm, r_tm_dual.d);
            }
            if (grad_sigma_e != nullptr) {
                ad::slab_fresnel_dual(
                    frame.cos_theta, b_eps, b_sigma, b_mu, b_gain, b_thickness,
                    frequency_hz, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f,
                    r_te_dual, r_tm_dual);
                grad_sigma_e[scalar_index] =
                    adj_dot(g_r_te, r_te_dual.d) + adj_dot(g_r_tm, r_tm_dual.d);
            }
            if (grad_gain != nullptr) {
                ad::slab_fresnel_dual(
                    frame.cos_theta, b_eps, b_sigma, b_mu, b_gain, b_thickness,
                    frequency_hz, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f,
                    r_te_dual, r_tm_dual);
                grad_gain[scalar_index] =
                    adj_dot(g_r_te, r_te_dual.d) + adj_dot(g_r_tm, r_tm_dual.d);
            }
            if (grad_thickness != nullptr) {
                ad::slab_fresnel_dual(
                    frame.cos_theta, b_eps, b_sigma, b_mu, b_gain, b_thickness,
                    frequency_hz, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f,
                    r_te_dual, r_tm_dual);
                grad_thickness[scalar_index] =
                    adj_dot(g_r_te, r_te_dual.d) + adj_dot(g_r_tm, r_tm_dual.d);
            }
            if (grad_frequency != nullptr) {
                ad::slab_fresnel_dual(
                    frame.cos_theta, b_eps, b_sigma, b_mu, b_gain, b_thickness,
                    frequency_hz, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f,
                    r_te_dual, r_tm_dual);
                g_freq +=
                    adj_dot(g_r_te, r_te_dual.d) + adj_dot(g_r_tm, r_tm_dual.d);
            }
            if (!need_geometry)
                continue;

            // Geometry enters the Fresnel only through cos_theta.
            ad::slab_fresnel_dual(
                frame.cos_theta, b_eps, b_sigma, b_mu, b_gain, b_thickness,
                frequency_hz, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
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
            atomicAdd(grad_frequency, g_freq);
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
            ad::slab_fresnel_dual(
                frame.cos_theta.v,
                eps_r[scalar_index],
                sigma_e[scalar_index],
                mu_r[scalar_index],
                gain[scalar_index],
                thickness[scalar_index],
                frequency_hz,
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
                need_grad_frequency ? grad_frequency.data_ptr<float>() : nullptr,
                need_grad_geometry ? grad_source.data_ptr<float>() : nullptr,
                need_grad_geometry ? grad_target.data_ptr<float>() : nullptr,
                need_grad_geometry ? grad_positions.data_ptr<float>() : nullptr,
                need_grad_geometry ? grad_normals.data_ptr<float>() : nullptr);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
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

// Per-bounce recomputation shared by forward, backward and jvp. Fills the
// bounce arrays and returns the product factor (before the replaced mask).
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
    float factor = 1.0f;
    for (int b = 0; b < depth; ++b) {
        const int64_t pb = (row * depth + b) * 3;
        float sx, sy, sz;
        if (b == 0) {
            sx = positions[pb + 0] - source[row * 3 + 0];
            sy = positions[pb + 1] - source[row * 3 + 1];
            sz = positions[pb + 2] - source[row * 3 + 2];
        } else {
            const int64_t prev = (row * depth + b - 1) * 3;
            sx = positions[pb + 0] - positions[prev + 0];
            sy = positions[pb + 1] - positions[prev + 1];
            sz = positions[pb + 2] - positions[prev + 2];
        }
        const float len = sqrtf(sx * sx + sy * sy + sz * sz);
        // Match Torch's ``seg / norm.clamp_min(1e-9)`` (division, not a
        // reciprocal multiply) so the coherent factor is float-faithful.
        const float denom = fmaxf(len, 1.0e-9f);
        const float dx = sx / denom, dy = sy / denom, dz = sz / denom;
        const float inv = 1.0f / denom;
        const float nx = normals[pb + 0];
        const float ny = normals[pb + 1];
        const float nz = normals[pb + 2];
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
        const bool r = rough_b[row * depth + b];
        rough[b] = r;
        const float s = sigma_b[row * depth + b];
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
            atomicAdd(out_grad_frequency, grad_factor * df);
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
    if (need_frequency) {
        grad_frequency = zero_filled({1}, path_gain.options());
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
            need_frequency ? grad_frequency.data_ptr<float>() : nullptr,
            need_field, need_geometry, need_frequency);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
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
