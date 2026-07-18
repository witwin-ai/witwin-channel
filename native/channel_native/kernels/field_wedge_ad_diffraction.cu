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

// Scalar/vector seeding shims shared with the diffraction-map companions
// (field_transport_ad.cuh).
using ad::seeded;
using ad::seeded3;
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
    // ISB boundary taper (ADR-017), D member. Plain (non-differentiated) config
    // scalar carried into pair.isbTaperWidthScale so the shared UTD header
    // notches the incident-boundary odd part. 0 reproduces the hard GO step.
    // Taper + AD is refused upstream (deterministic/path pipelines), so this is
    // always 0 on the live AD path; it is threaded for lockstep completeness of
    // the guarded path only.
    float isb_taper_width_scale;
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
    // ISB boundary taper (ADR-017), D member. isbTaperWidthScale is a plain
    // float in PairInputsT (a config scalar like selectStationaryPoint), so it
    // is assigned directly and carries no tangent; the header derives w_F / s2
    // internally, so the kernel must not precompute them. 0 = hard GO step
    // (bit-identical to the pre-ADR-017 twin).
    pair.isbTaperWidthScale = in.isb_taper_width_scale;
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
    const bool* edge_boundary,
    float isb_taper_width) {
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
    in.isb_taper_width_scale = isb_taper_width;
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
        const bool* edge_boundary, float isb_taper_width

#define WEDGE_ROW_ARGS(index)                                                 \
    index, source, target, edge_position, edge_direction, edge_t_min,         \
        edge_t_max, edge_n0, edge_n1, exterior_angle, face0_valid,            \
        face0_eps_r, face0_sigma_e, face0_mu_r, face0_gain, face1_valid,      \
        face1_eps_r, face1_sigma_e, face1_mu_r, face1_gain, tx_power,         \
        frequency_hz, vertex_v0, vertex_v1, vertex_opp0, vertex_opp1,         \
        edge_boundary, isb_taper_width

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
        opt_ptr<float>(vertex_args.opp1), opt_ptr<bool>(vertex_args.boundary), \
        static_cast<float>(isb_boundary_taper_width)

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
    pybind11::object edge_boundary,
    double isb_boundary_taper_width) {
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
    bool need_grad_vertices,
    double isb_boundary_taper_width) {
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
    pybind11::object tangent_vertex_opp1,
    double isb_boundary_taper_width) {
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
