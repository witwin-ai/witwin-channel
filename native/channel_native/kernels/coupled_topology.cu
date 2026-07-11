#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime_api.h>
#include <rayd/shared/utd/utd_math.h>
#include <torch/extension.h>

#include "../tensor_checks.h"

#include <cmath>
#include <vector>

namespace {

constexpr int kBlockSize = 256;
constexpr float kGeometryEpsilon = 1.0e-6f;
constexpr float kSpeedOfLight = 299792458.0f;
namespace utd = witwin::channel::native_ext;

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

int launch_blocks(int64_t count) {
    return static_cast<int>((count + kBlockSize - 1) / kBlockSize);
}

}  // namespace

std::vector<at::Tensor> cn_coupled_rd_prepare_cuda(
    at::Tensor source,
    at::Tensor receiver,
    at::Tensor plane_point,
    at::Tensor plane_normal,
    at::Tensor edge_pos,
    at::Tensor edge_dir,
    at::Tensor edge_t_min,
    at::Tensor edge_t_max) {
    using channel_native::check_flat_tensor;
    using channel_native::check_vec3_table;
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

pybind11::dict cn_coupled_rd_finalize_cuda(
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
    using channel_native::check_flat_tensor;
    using channel_native::check_vec3_table;
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

at::Tensor cn_coupled_active_mask_cuda(at::Tensor lhs, at::Tensor rhs) {
    using channel_native::check_flat_tensor;
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
