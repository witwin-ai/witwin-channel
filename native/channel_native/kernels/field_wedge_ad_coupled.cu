#include "field_wedge_ad_common.cuh"

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
// Coupled reflection-diffraction dual row (components 3/4). Mirrors
// coupled_rd_field_kernel step by step on RayD duals.
//
// Truncation-factor policy (G4): the primal now evaluates the coupled leg on
// the stationary path (selectStationaryPoint = 1, stationaryExternalIncident =
// 1) with real edge-segment bounds, so the truncation lives INSIDE the pair
// coefficient (monotone even T_mono + corner_mend_gamma + boundary blend). The
// dual mirrors this by calling compute_pair_vector_contribution directly, and
// those derivatives flow in lockstep with the primal (there is no longer a
// pseudo-infinite factor to freeze). The edge geometry itself stays frozen
// (ADR-011: coupled rows carry no mesh-vertex gradient).
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
    float edge_line_min;
    float edge_line_max;
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
        const DualV3 axis = field::project_to_wedge_plane(
            field::dual_const3(in.tx_pol), direction);
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
        const DualV3 tx_axis = field::project_to_wedge_plane(
            field::dual_const3(in.tx_pol), source_to_hit);
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
    // G4: real edge-segment bounds (frozen constants; the coupled leg does not
    // differentiate the edge geometry per ADR-011) so the stationary machinery
    // truncates and corner-mends. Replaces the former +-1e5 infinite edge.
    pair.edgeLineMin = Dual(in.edge_line_min);
    pair.edgeLineMax = Dual(in.edge_line_max);
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
    // G4: run the stationary-path machinery (re-anchor + monotone even
    // truncation + corner_mend_gamma + boundary-distance blend + external
    // incidence re-extrapolation) on the coupled leg, matching the primal.
    pair.selectStationaryPoint = 1.0f;
    pair.stationaryExternalIncident = 1.0f;
    field::MaterialParamsT<Dual> material{};
    material.omega = Dual(-1.0f);

    // Evaluate through the shared RayD header template, identical in structure
    // to the primal's compute_pair_contribution and to the order-1 diffraction
    // dual (field_wedge_ad_diffraction.cu). The header owns edge re-anchoring,
    // truncation and validity (it returns zero for a blocked/short geometry),
    // so the former manual pre-check and frozen-finite-factor inline are gone;
    // the T_mono / gamma / B derivatives now flow through the dual in lockstep
    // with the primal.
    DualC3 value = field::compute_pair_vector_contribution(
        pair, diffraction_target, wave_number, material);

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
    const float* edge_line_min,
    const float* edge_line_max,
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
    in.edge_line_min = edge_line_min[index];
    in.edge_line_max = edge_line_max[index];
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
        const float* wedge_thickness1, const float* edge_line_min,            \
        const float* edge_line_max, float frequency_hz, bool reverse

#define COUPLED_ROW_ARGS(index)                                               \
    index, source, target, reflection_position, reflection_normal,            \
        edge_position, edge_direction, edge_n0, edge_n1, exterior_angle,      \
        tx_power, tx_polarization, rx_polarization, reflection_eps_r,         \
        reflection_sigma_e, reflection_mu_r, reflection_gain,                 \
        reflection_thickness, wedge_eps_r0, wedge_sigma_e0, wedge_mu_r0,      \
        wedge_gain0, wedge_thickness0, wedge_eps_r1, wedge_sigma_e1,          \
        wedge_mu_r1, wedge_gain1, wedge_thickness1, edge_line_min,            \
        edge_line_max, frequency_hz, reverse

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
// ADR-013 D4: coupled double-diffraction dual row (component 7). Mirrors
// coupled_dd_field_kernel (field_transport.cu) step by step on RayD duals.
//
// Both legs run the stationary path so the truncation / corner-mend / boundary
// blend derivatives flow through the dual in lockstep with the primal (the
// dual calls compute_pair_vector_contribution directly, exactly like the
// order-1 diffraction dual and the coupled R->D dual after ADR-012 G4-3). The
// gradient contract (ADR-013 D4): tx (source) and rx (target) flow through the
// live per-leg re-anchoring; Q1/Q2, the edge axes/normals/exterior, the edge
// bounds and the polarizations are frozen seeds (detached), so they are loaded
// as dual constants and carry no gradient. The wedge face materials and the
// frequency are differentiable. This file is a PRECISE-math translation unit
// (only field_wedge_ad_diffraction.cu is fast-math), matching the primal.
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
    const DualV3 incident_direction1 = field::safe_normalize(
        field::f3_sub(q1, src), field::v3_const<Dual>(1.f, 0.f, 0.f));
    const DualV3 outgoing_direction1 = field::safe_normalize(
        field::f3_sub(q2, q1), field::v3_const<Dual>(1.f, 0.f, 0.f));
    const field::Basis3T<Dual> input_edge_basis1 = field::diffraction_edge_basis(
        field::f3_sub(q1, src), e1_axis, false);
    const field::Basis3T<Dual> output_edge_basis1 = field::diffraction_edge_basis(
        field::f3_sub(q2, q1), e1_axis, true);

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
    pair1.face0Operator = slab_face_operator_dual(
        field::fabsf(field::f3_dot(pair1.n0, incident_direction1)),
        {in.w1a_eps, seeds.w1a_eps},
        {in.w1a_sigma, seeds.w1a_sigma},
        in.w1a_mu,
        {in.w1a_gain, seeds.w1a_gain},
        {in.w1a_thickness, seeds.w1a_thickness},
        frequency,
        pair1.n0,
        incident_direction1,
        outgoing_direction1,
        input_edge_basis1,
        output_edge_basis1);
    pair1.face1Operator = slab_face_operator_dual(
        field::fabsf(field::f3_dot(pair1.nn, incident_direction1)),
        {in.w1b_eps, seeds.w1b_eps},
        {in.w1b_sigma, seeds.w1b_sigma},
        in.w1b_mu,
        {in.w1b_gain, seeds.w1b_gain},
        {in.w1b_thickness, seeds.w1b_thickness},
        frequency,
        pair1.nn,
        incident_direction1,
        outgoing_direction1,
        input_edge_basis1,
        output_edge_basis1);
    pair1.selectStationaryPoint = 1.0f;
    pair1.stationaryExternalIncident = 0.0f;
    field::MaterialParamsT<Dual> material1{};
    // omega < 0 keeps the stored (frozen-at-Keller) slab face operators; the
    // direct incident spherical wave uses the (frozen) transmitter
    // polarization, exactly like coupled_dd_field_kernel.
    material1.omega = Dual(-1.0f);
    material1.txPolX = Dual(in.tx_pol.x);
    material1.txPolY = Dual(in.tx_pol.y);
    material1.txPolZ = Dual(in.tx_pol.z);
    const DualC3 leg1_field = field::compute_pair_vector_contribution(
        pair1, q2, wave_number, material1);

    // --- Leg 2: external-incident diffraction at e2, observed at rx ---
    const DualV3 incident_direction2 = field::safe_normalize(
        field::f3_sub(q2, q1), field::v3_const<Dual>(1.f, 0.f, 0.f));
    const DualV3 outgoing_direction2 = field::safe_normalize(
        field::f3_sub(dst, q2), field::v3_const<Dual>(1.f, 0.f, 0.f));
    const field::Basis3T<Dual> input_edge_basis2 = field::diffraction_edge_basis(
        field::f3_sub(q2, q1), e2_axis, false);
    const field::Basis3T<Dual> output_edge_basis2 = field::diffraction_edge_basis(
        field::f3_sub(dst, q2), e2_axis, true);

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
    pair2.face0Operator = slab_face_operator_dual(
        field::fabsf(field::f3_dot(pair2.n0, incident_direction2)),
        {in.w2a_eps, seeds.w2a_eps},
        {in.w2a_sigma, seeds.w2a_sigma},
        in.w2a_mu,
        {in.w2a_gain, seeds.w2a_gain},
        {in.w2a_thickness, seeds.w2a_thickness},
        frequency,
        pair2.n0,
        incident_direction2,
        outgoing_direction2,
        input_edge_basis2,
        output_edge_basis2);
    pair2.face1Operator = slab_face_operator_dual(
        field::fabsf(field::f3_dot(pair2.nn, incident_direction2)),
        {in.w2b_eps, seeds.w2b_eps},
        {in.w2b_sigma, seeds.w2b_sigma},
        in.w2b_mu,
        {in.w2b_gain, seeds.w2b_gain},
        {in.w2b_thickness, seeds.w2b_thickness},
        frequency,
        pair2.nn,
        incident_direction2,
        outgoing_direction2,
        input_edge_basis2,
        output_edge_basis2);
    pair2.selectStationaryPoint = 1.0f;
    pair2.stationaryExternalIncident = 1.0f;
    field::MaterialParamsT<Dual> material2{};
    material2.omega = Dual(-1.0f);
    const DualC3 value = field::compute_pair_vector_contribution(
        pair2, dst, wave_number, material2);

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
            atomicAdd(grad_frequency, coupled_contract(
                index, grad_field_vector, grad_coefficient, grad_path_field,
                grad_path_gain, t));
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
        edge_line_min.data_ptr<float>(), edge_line_max.data_ptr<float>(),     \
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
        at::Tensor wedge_thickness1, at::Tensor edge_line_min,                \
        at::Tensor edge_line_max, double frequency_hz, bool reverse

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
         {wedge_thickness1, "wedge_thickness1"},                              \
         {edge_line_min, "edge_line_min"},                                    \
         {edge_line_max, "edge_line_max"}},                                   \
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

// ---------------------------------------------------------------------------
// ADR-013 D4: coupled double-diffraction AD host companions. Twins of the RD
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

pybind11::dict cn_field_coupled_dd_backward(
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
    if (need_grad_frequency) {
        grad_frequency = zero_scalar(options);
        grad_frequency_ptr = &grad_frequency;
    }
    const bool any_grad = g_field != nullptr || g_coefficient != nullptr ||
                          g_path_field != nullptr || g_path_gain != nullptr;
    if (count > 0 && any_grad) {
        cudaStream_t stream = at::cuda::getCurrentCUDAStream(source.get_device()).stream();
        coupled_dd_backward_kernel<<<launch_blocks(count), kBlockSize, 0, stream>>>(
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
            opt_mut_ptr<float>(grad_frequency_ptr));
        C10_CUDA_KERNEL_LAUNCH_CHECK();
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

pybind11::dict cn_field_coupled_dd_jvp(
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
        coupled_dd_jvp_kernel<<<launch_blocks(count), kBlockSize, 0, stream>>>(
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
