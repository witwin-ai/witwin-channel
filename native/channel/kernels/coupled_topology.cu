#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime_api.h>
#include <rayd/shared/utd/utd_math.h>
#include "torch_cuda_minimal.h"

#include "../tensor_checks.h"

#include <cmath>
#include <vector>

namespace {

constexpr int kBlockSize = 256;
constexpr float kGeometryEpsilon = 1.0e-6f;
constexpr float kSpeedOfLight = 299792458.0f;
namespace utd = rayd::shared::utd;

__device__ __forceinline__ float3 load3(const float *values, int64_t index) {
    const int64_t base = index * 3;
    return make_float3(values[base], values[base + 1], values[base + 2]);
}

__device__ __forceinline__ void store3(float *values, int64_t index, float3 value) {
    const int64_t base = index * 3;
    values[base] = value.x;
    values[base + 1] = value.y;
    values[base + 2] = value.z;
}

__device__ __forceinline__ float dot3(float3 a, float3 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

__device__ __forceinline__ float3 add3(float3 a, float3 b) {
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

__device__ __forceinline__ float3 sub3(float3 a, float3 b) {
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

__device__ __forceinline__ float3 mul3(float3 value, float scale) {
    return make_float3(value.x * scale, value.y * scale, value.z * scale);
}

__device__ __forceinline__ float3 cross3(float3 a, float3 b) {
    return make_float3(
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x);
}

__device__ __forceinline__ float length3(float3 value) {
    return sqrtf(dot3(value, value));
}

__device__ __forceinline__ float3 normalize3(float3 value) {
    const float length = length3(value);
    return length > kGeometryEpsilon ? mul3(value, 1.0f / length) : make_float3(0.0f, 0.0f, 0.0f);
}

__device__ __forceinline__ utd::float3a to_utd(float3 value) {
    return utd::make_f3(value.x, value.y, value.z);
}

__global__ void coupled_rd_prepare_kernel(
    int64_t count,
    const float *__restrict__ source,
    const float *__restrict__ receiver,
    const float *__restrict__ plane_point,
    const float *__restrict__ plane_normal,
    const float *__restrict__ edge_pos,
    const float *__restrict__ edge_dir,
    const float *__restrict__ edge_t_min,
    const float *__restrict__ edge_t_max,
    bool *__restrict__ active,
    float *__restrict__ edge_point,
    float *__restrict__ virtual_source,
    float *__restrict__ predicted_reflection_point) {
    for (int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < count;
         index += static_cast<int64_t>(blockDim.x) * gridDim.x) {
        const float3 src = load3(source, index);
        const float3 rx = load3(receiver, index);
        const float3 p0 = load3(plane_point, index);
        const float3 normal = normalize3(load3(plane_normal, index));
        const float3 direction = normalize3(load3(edge_dir, index));
        const float t_min = edge_t_min[index];
        const float t_max = edge_t_max[index];
        const float edge_length = t_max - t_min;
        const bool finite_inputs =
            isfinite(src.x) && isfinite(src.y) && isfinite(src.z) &&
            isfinite(rx.x) && isfinite(rx.y) && isfinite(rx.z) &&
            isfinite(p0.x) && isfinite(p0.y) && isfinite(p0.z) &&
            length3(normal) > 0.0f && length3(direction) > 0.0f &&
            isfinite(edge_length) && edge_length > kGeometryEpsilon;
        if (!finite_inputs) {
            active[index] = false;
            store3(edge_point, index, make_float3(NAN, NAN, NAN));
            store3(virtual_source, index, make_float3(NAN, NAN, NAN));
            store3(predicted_reflection_point, index, make_float3(NAN, NAN, NAN));
            continue;
        }

        const float signed_distance = dot3(sub3(src, p0), normal);
        const float3 image = sub3(src, mul3(normal, 2.0f * signed_distance));
        const float3 edge_origin = add3(load3(edge_pos, index), mul3(direction, t_min));
        const float parameter = utd::first_order_diffraction_parameter(
            to_utd(image), to_utd(rx), to_utd(edge_origin), to_utd(direction));
        const bool inside_edge = isfinite(parameter) && parameter > kGeometryEpsilon &&
                                 parameter < edge_length - kGeometryEpsilon;
        const float3 diffraction_point = add3(edge_origin, mul3(direction, parameter));

        const float3 image_to_edge = sub3(diffraction_point, image);
        const float plane_denominator = dot3(image_to_edge, normal);
        const float plane_parameter =
            fabsf(plane_denominator) > kGeometryEpsilon
                ? dot3(sub3(p0, image), normal) / plane_denominator
                : NAN;
        const bool reflection_between = isfinite(plane_parameter) &&
                                        plane_parameter > kGeometryEpsilon &&
                                        plane_parameter < 1.0f - kGeometryEpsilon;
        const float3 reflection_point = add3(image, mul3(image_to_edge, plane_parameter));
        const bool valid = inside_edge && reflection_between;
        active[index] = valid;
        store3(edge_point, index, valid ? diffraction_point : make_float3(NAN, NAN, NAN));
        store3(virtual_source, index, image);
        store3(predicted_reflection_point, index, valid ? reflection_point : make_float3(NAN, NAN, NAN));
    }
}

__global__ void coupled_rd_finalize_kernel(
    int64_t count,
    const bool *__restrict__ prefix_active,
    const bool *__restrict__ suffix_visible,
    const float *__restrict__ epc_path_length,
    const int *__restrict__ resolved_face,
    const int *__restrict__ edge_id,
    const float *__restrict__ reflection_point,
    const float *__restrict__ reflection_normal,
    const float *__restrict__ edge_point,
    const float *__restrict__ edge_direction,
    const float *__restrict__ receiver,
    bool reverse,
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
        const int reflection_slot = reverse ? 1 : 0;
        const int diffraction_slot = reverse ? 0 : 1;
        const int64_t sequence_base = index * 2;
        interaction_type_sequence[sequence_base + reflection_slot] = 1;
        interaction_type_sequence[sequence_base + diffraction_slot] = 2;
        primitive_sequence[sequence_base] = -1;
        primitive_sequence[sequence_base + 1] = -1;
        primitive_sequence[sequence_base + reflection_slot] = resolved_face[index];
        edge_sequence[sequence_base] = -1;
        edge_sequence[sequence_base + 1] = -1;
        edge_sequence[sequence_base + diffraction_slot] = edge_id[index];

        const float3 hit = load3(reflection_point, index);
        const float3 normal = load3(reflection_normal, index);
        const float3 edge = load3(edge_point, index);
        const float3 direction = normalize3(load3(edge_direction, index));
        store3(interaction_positions, sequence_base + reflection_slot, hit);
        store3(interaction_positions, sequence_base + diffraction_slot, edge);
        store3(interaction_normals, sequence_base + reflection_slot, normal);
        // A diffraction edge has an axis and two face normals, not one surface
        // normal. Keep the generic normal slot explicitly unavailable.
        store3(interaction_normals, sequence_base + diffraction_slot, make_float3(NAN, NAN, NAN));

        const bool valid = prefix_active[index] && suffix_visible[index];
        valid_out[index] = valid;
        const float suffix = length3(sub3(load3(receiver, index), edge));
        const float total = valid ? epc_path_length[index] + suffix : NAN;
        path_length[index] = total;
        delay[index] = valid ? total / kSpeedOfLight : NAN;
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

// Double-diffraction (cid 7) discovery: alternating-projection Fermat solve of
// the two-edge Keller point pair, mirroring the eps/inside semantics of
// coupled_rd_prepare_kernel. No CPU/Torch recomputation and no heuristic clamping:
// each iteration takes the raw closed-form single-edge projection and the row
// is validated (or dropped) purely by the strict-inside test on both segments.
constexpr int kDoubleDiffractionIterations = 16;

__global__ void coupled_dd_prepare_kernel(
    int64_t count,
    const float *__restrict__ source,
    const float *__restrict__ receiver,
    const float *__restrict__ edge1_pos,
    const float *__restrict__ edge1_dir,
    const float *__restrict__ edge1_t_min,
    const float *__restrict__ edge1_t_max,
    const float *__restrict__ edge2_pos,
    const float *__restrict__ edge2_dir,
    const float *__restrict__ edge2_t_min,
    const float *__restrict__ edge2_t_max,
    bool *__restrict__ active,
    float *__restrict__ edge1_point,
    float *__restrict__ edge2_point) {
    for (int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < count;
         index += static_cast<int64_t>(blockDim.x) * gridDim.x) {
        const float3 tx = load3(source, index);
        const float3 rx = load3(receiver, index);
        const float3 dir1 = normalize3(load3(edge1_dir, index));
        const float3 dir2 = normalize3(load3(edge2_dir, index));
        const float t_min1 = edge1_t_min[index];
        const float t_max1 = edge1_t_max[index];
        const float t_min2 = edge2_t_min[index];
        const float t_max2 = edge2_t_max[index];
        const float length1 = t_max1 - t_min1;
        const float length2 = t_max2 - t_min2;
        const float3 origin1 = add3(load3(edge1_pos, index), mul3(dir1, t_min1));
        const float3 origin2 = add3(load3(edge2_pos, index), mul3(dir2, t_min2));
        const bool finite_inputs =
            isfinite(tx.x) && isfinite(tx.y) && isfinite(tx.z) &&
            isfinite(rx.x) && isfinite(rx.y) && isfinite(rx.z) &&
            length3(dir1) > 0.0f && length3(dir2) > 0.0f &&
            isfinite(length1) && length1 > kGeometryEpsilon &&
            isfinite(length2) && length2 > kGeometryEpsilon;
        if (!finite_inputs) {
            active[index] = false;
            store3(edge1_point, index, make_float3(NAN, NAN, NAN));
            store3(edge2_point, index, make_float3(NAN, NAN, NAN));
            continue;
        }

        // Collinear edge pairs lie on one physical line; they are duplicate
        // representations of the same edge and cannot form a two-edge cascade.
        // Same-line test uses the shared kGeometryEpsilon: parallel directions
        // AND zero perpendicular offset of origin2 from edge 1's line.
        const bool parallel = length3(cross3(dir1, dir2)) < kGeometryEpsilon;
        const float3 delta = sub3(origin2, origin1);
        const float3 perpendicular = sub3(delta, mul3(dir1, dot3(delta, dir1)));
        const bool same_line = parallel && length3(perpendicular) < kGeometryEpsilon;
        if (same_line) {
            active[index] = false;
            store3(edge1_point, index, make_float3(NAN, NAN, NAN));
            store3(edge2_point, index, make_float3(NAN, NAN, NAN));
            continue;
        }

        // Alternating projection: seed Q2 at edge 2's midpoint, then for a
        // fixed number of deterministic float32 iterations project Q1 onto
        // edge 1 for the (tx, Q2) diffraction and Q2 onto edge 2 for the
        // (Q1, rx) diffraction. Deterministic order, identical every launch.
        float3 q1 = make_float3(NAN, NAN, NAN);
        float3 q2 = add3(origin2, mul3(dir2, 0.5f * length2));
        float parameter1 = NAN;
        float parameter2 = NAN;
        for (int iteration = 0; iteration < kDoubleDiffractionIterations; ++iteration) {
            parameter1 = utd::first_order_diffraction_parameter(
                to_utd(tx), to_utd(q2), to_utd(origin1), to_utd(dir1));
            q1 = add3(origin1, mul3(dir1, parameter1));
            parameter2 = utd::first_order_diffraction_parameter(
                to_utd(q1), to_utd(rx), to_utd(origin2), to_utd(dir2));
            q2 = add3(origin2, mul3(dir2, parameter2));
        }

        const bool inside1 = isfinite(parameter1) && parameter1 > kGeometryEpsilon &&
                             parameter1 < length1 - kGeometryEpsilon;
        const bool inside2 = isfinite(parameter2) && parameter2 > kGeometryEpsilon &&
                             parameter2 < length2 - kGeometryEpsilon;
        const bool finite_points =
            isfinite(q1.x) && isfinite(q1.y) && isfinite(q1.z) &&
            isfinite(q2.x) && isfinite(q2.y) && isfinite(q2.z);
        // All three segment directions (tx->Q1, Q1->Q2, Q2->rx) must be
        // well-defined; a degenerate zero-length leg has no diffraction cone.
        const bool segments_defined =
            length3(sub3(q1, tx)) > kGeometryEpsilon &&
            length3(sub3(q2, q1)) > kGeometryEpsilon &&
            length3(sub3(rx, q2)) > kGeometryEpsilon;
        const bool valid = inside1 && inside2 && finite_points && segments_defined;
        active[index] = valid;
        store3(edge1_point, index, valid ? q1 : make_float3(NAN, NAN, NAN));
        store3(edge2_point, index, valid ? q2 : make_float3(NAN, NAN, NAN));
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

        const float3 q1 = load3(edge1_point, index);
        const float3 q2 = load3(edge2_point, index);
        store3(interaction_positions, sequence_base + 0, q1);
        store3(interaction_positions, sequence_base + 1, q2);
        // A diffraction edge has an axis and two face normals, not one surface
        // normal. Mirror the coupled R/D finalize: both diffraction slots keep
        // the generic normal slot explicitly unavailable.
        store3(interaction_normals, sequence_base + 0, make_float3(NAN, NAN, NAN));
        store3(interaction_normals, sequence_base + 1, make_float3(NAN, NAN, NAN));

        const bool valid = prefix_active[index];
        valid_out[index] = valid;
        const float3 tx = load3(source, index);
        const float3 rx = load3(receiver, index);
        const float total = length3(sub3(q1, tx)) + length3(sub3(q2, q1)) +
                            length3(sub3(rx, q2));
        path_length[index] = valid ? total : NAN;
        delay[index] = valid ? total / kSpeedOfLight : NAN;
    }
}

int launch_blocks(int64_t count) {
    return static_cast<int>((count + kBlockSize - 1) / kBlockSize);
}

}  // namespace

std::vector<at::Tensor> channel_coupled_rd_prepare_cuda(
    at::Tensor source,
    at::Tensor receiver,
    at::Tensor plane_point,
    at::Tensor plane_normal,
    at::Tensor edge_pos,
    at::Tensor edge_dir,
    at::Tensor edge_t_min,
    at::Tensor edge_t_max) {
    using channel::check_flat_tensor;
    using channel::check_vec3_table;
    check_vec3_table(source, "source");
    check_vec3_table(receiver, "receiver");
    check_vec3_table(plane_point, "plane_point");
    check_vec3_table(plane_normal, "plane_normal");
    check_vec3_table(edge_pos, "edge_pos");
    check_vec3_table(edge_dir, "edge_dir");
    check_flat_tensor(edge_t_min, "edge_t_min", at::kFloat);
    check_flat_tensor(edge_t_max, "edge_t_max", at::kFloat);
    const int64_t count = source.size(0);
    for (const auto &tensor : {receiver, plane_point, plane_normal, edge_pos, edge_dir})
        TORCH_CHECK(tensor.size(0) == count, "coupled geometry vector tables must have matching rows");
    TORCH_CHECK(edge_t_min.size(0) == count && edge_t_max.size(0) == count,
                "coupled geometry edge bounds must match source rows");

    auto active = at::empty({count}, source.options().dtype(at::kBool));
    auto edge_point = at::empty_like(source);
    auto virtual_source = at::empty_like(source);
    auto predicted_reflection_point = at::empty_like(source);
    if (count > 0) {
        cudaStream_t stream = at::cuda::getCurrentCUDAStream(source.get_device()).stream();
        coupled_rd_prepare_kernel<<<launch_blocks(count), kBlockSize, 0, stream>>>(
            count,
            source.data_ptr<float>(),
            receiver.data_ptr<float>(),
            plane_point.data_ptr<float>(),
            plane_normal.data_ptr<float>(),
            edge_pos.data_ptr<float>(),
            edge_dir.data_ptr<float>(),
            edge_t_min.data_ptr<float>(),
            edge_t_max.data_ptr<float>(),
            active.data_ptr<bool>(),
            edge_point.data_ptr<float>(),
            virtual_source.data_ptr<float>(),
            predicted_reflection_point.data_ptr<float>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return {active, edge_point, virtual_source, predicted_reflection_point};
}

pybind11::dict channel_coupled_rd_finalize_cuda(
    at::Tensor prefix_active,
    at::Tensor suffix_visible,
    at::Tensor epc_path_length,
    at::Tensor resolved_face,
    at::Tensor edge_id,
    at::Tensor reflection_point,
    at::Tensor reflection_normal,
    at::Tensor edge_point,
    at::Tensor edge_direction,
    at::Tensor receiver,
    bool reverse) {
    using channel::check_flat_tensor;
    using channel::check_vec3_table;
    check_flat_tensor(prefix_active, "prefix_active", at::kBool);
    check_flat_tensor(suffix_visible, "suffix_visible", at::kBool);
    check_flat_tensor(epc_path_length, "epc_path_length", at::kFloat);
    check_flat_tensor(resolved_face, "resolved_face", at::kInt);
    check_flat_tensor(edge_id, "edge_id", at::kInt);
    check_vec3_table(reflection_point, "reflection_point");
    check_vec3_table(reflection_normal, "reflection_normal");
    check_vec3_table(edge_point, "edge_point");
    check_vec3_table(edge_direction, "edge_direction");
    check_vec3_table(receiver, "receiver");
    const int64_t count = prefix_active.size(0);
    TORCH_CHECK(suffix_visible.size(0) == count, "suffix_visible must match prefix_active");
    TORCH_CHECK(epc_path_length.size(0) == count && resolved_face.size(0) == count && edge_id.size(0) == count,
                "coupled geometry scalar rows must match valid");
    for (const auto &tensor : {reflection_point, reflection_normal, edge_point, edge_direction, receiver})
        TORCH_CHECK(tensor.size(0) == count, "coupled geometry vector rows must match valid");

    auto int_options = edge_id.options().dtype(at::kInt);
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
        coupled_rd_finalize_kernel<<<launch_blocks(count), kBlockSize, 0, stream>>>(
            count,
            prefix_active.data_ptr<bool>(),
            suffix_visible.data_ptr<bool>(),
            epc_path_length.data_ptr<float>(),
            resolved_face.data_ptr<int>(),
            edge_id.data_ptr<int>(),
            reflection_point.data_ptr<float>(),
            reflection_normal.data_ptr<float>(),
            edge_point.data_ptr<float>(),
            edge_direction.data_ptr<float>(),
            receiver.data_ptr<float>(),
            reverse,
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
    out["face_id"] = resolved_face;
    out["edge_id"] = edge_id;
    out["interaction_positions"] = interaction_positions;
    out["interaction_normals"] = interaction_normals;
    out["reflection_position"] = reflection_point;
    out["reflection_normal"] = reflection_normal;
    out["edge_position"] = edge_point;
    out["edge_direction"] = edge_direction;
    out["path_length_m"] = path_length;
    out["delay_s"] = delay;
    return out;
}

std::vector<at::Tensor> channel_coupled_dd_prepare_cuda(
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
    const int64_t count = source.size(0);
    for (const auto &tensor : {receiver, edge1_pos, edge1_dir, edge2_pos, edge2_dir})
        TORCH_CHECK(tensor.size(0) == count,
                    "coupled double-diffraction vector tables must have matching rows");
    for (const auto &tensor : {edge1_t_min, edge1_t_max, edge2_t_min, edge2_t_max})
        TORCH_CHECK(tensor.size(0) == count,
                    "coupled double-diffraction edge bounds must match source rows");

    auto active = at::empty({count}, source.options().dtype(at::kBool));
    auto edge1_point = at::empty_like(source);
    auto edge2_point = at::empty_like(source);
    if (count > 0) {
        cudaStream_t stream = at::cuda::getCurrentCUDAStream(source.get_device()).stream();
        coupled_dd_prepare_kernel<<<launch_blocks(count), kBlockSize, 0, stream>>>(
            count,
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
            edge2_point.data_ptr<float>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return {active, edge1_point, edge2_point};
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
        coupled_dd_finalize_kernel<<<launch_blocks(count), kBlockSize, 0, stream>>>(
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
        coupled_active_mask_kernel<<<launch_blocks(count), kBlockSize, 0, stream>>>(
            count,
            lhs.data_ptr<bool>(),
            rhs.data_ptr<bool>(),
            out.data_ptr<bool>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return out;
}
