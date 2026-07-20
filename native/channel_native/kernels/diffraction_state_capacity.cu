#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cub/device/device_scan.cuh>
#include <cuda_runtime_api.h>

#include "../tensor_checks.h"

#include <algorithm>
#include <cstdint>
#include <vector>

namespace {

constexpr int kDiffractionBlockSize = 256;
using channel_native::check_tensor;

struct DiffractionStateInputView {
    const int *edge_index;
    int64_t edge_index_stride;
    const float *edge_position;
    int64_t edge_position_row_stride;
    int64_t edge_position_column_stride;
    const float *edge_direction;
    int64_t edge_direction_row_stride;
    int64_t edge_direction_column_stride;
    const float *edge_t_min;
    int64_t edge_t_min_stride;
    const float *edge_t_max;
    int64_t edge_t_max_stride;
    const float *n0;
    int64_t n0_row_stride;
    int64_t n0_column_stride;
    const float *n1;
    int64_t n1_row_stride;
    int64_t n1_column_stride;
    const int *prim0;
    int64_t prim0_stride;
    const int *prim1;
    int64_t prim1_stride;
    const float *exterior_angle;
    int64_t exterior_angle_stride;
    const float *source;
    int64_t source_row_stride;
    int64_t source_column_stride;
    const float *source_power;
    int64_t source_power_stride;
};

struct DiffractionStateOutputView {
    int *edge_index;
    float *edge_position;
    float *edge_direction;
    float *edge_t_min;
    float *edge_t_max;
    float *n0;
    float *n1;
    int *prim0;
    int *prim1;
    float *exterior_angle;
    float *source;
    float *source_power;
    bool *valid;
};

__device__ __forceinline__ void copy_strided_vec3(
    const float *input,
    int64_t input_row_stride,
    int64_t input_column_stride,
    int64_t input_row,
    float *output,
    int64_t output_row) {
    const int64_t input_base = input_row * input_row_stride;
    const int64_t output_base = output_row * 3;
    output[output_base + 0] = input[input_base + 0 * input_column_stride];
    output[output_base + 1] = input[input_base + 1 * input_column_stride];
    output[output_base + 2] = input[input_base + 2 * input_column_stride];
}

__global__ void diffraction_state_selection_flags_kernel(
    const bool *__restrict__ active,
    int *__restrict__ flags,
    int64_t state_count) {
    const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (int64_t state = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         state < state_count;
         state += stride) {
        flags[state] = active[state] ? 1 : 0;
    }
}

__global__ void diffraction_state_capacity_init_kernel(
    DiffractionStateOutputView output,
    int64_t capacity) {
    const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (int64_t row = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         row < capacity;
         row += stride) {
        output.edge_index[row] = -1;
        output.edge_t_min[row] = 0.0f;
        output.edge_t_max[row] = 0.0f;
        output.prim0[row] = -1;
        output.prim1[row] = -1;
        output.exterior_angle[row] = 0.0f;
        output.source_power[row] = 0.0f;
        output.valid[row] = false;
        const int64_t base = row * 3;
        for (int component = 0; component < 3; ++component) {
            output.edge_position[base + component] = 0.0f;
            output.edge_direction[base + component] = 0.0f;
            output.n0[base + component] = 0.0f;
            output.n1[base + component] = 0.0f;
            output.source[base + component] = 0.0f;
        }
    }
}

__global__ void diffraction_state_selection_status_init_kernel(
    int *__restrict__ actual_count,
    bool *__restrict__ overflow) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        actual_count[0] = 0;
        overflow[0] = false;
    }
}

__global__ void diffraction_state_selection_status_kernel(
    const int *__restrict__ flags,
    const int *__restrict__ offsets,
    int64_t state_count,
    int64_t capacity,
    int *__restrict__ actual_count,
    bool *__restrict__ overflow) {
    if (blockIdx.x != 0 || threadIdx.x != 0) {
        return;
    }
    const int last = static_cast<int>(state_count - 1);
    const int selected_count = offsets[last] + flags[last];
    const bool did_overflow = selected_count > capacity;
    actual_count[0] = did_overflow ? 0 : selected_count;
    overflow[0] = did_overflow;
}

__global__ void diffraction_state_capacity_gather_kernel(
    const bool *__restrict__ active,
    const int *__restrict__ offsets,
    const bool *__restrict__ overflow,
    DiffractionStateInputView input,
    DiffractionStateOutputView output,
    int64_t state_count) {
    if (overflow[0]) {
        return;
    }
    const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (int64_t state = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         state < state_count;
         state += stride) {
        if (!active[state]) {
            continue;
        }
        const int64_t row = offsets[state];
        output.edge_index[row] = input.edge_index[state * input.edge_index_stride];
        copy_strided_vec3(
            input.edge_position,
            input.edge_position_row_stride,
            input.edge_position_column_stride,
            state,
            output.edge_position,
            row);
        copy_strided_vec3(
            input.edge_direction,
            input.edge_direction_row_stride,
            input.edge_direction_column_stride,
            state,
            output.edge_direction,
            row);
        output.edge_t_min[row] = input.edge_t_min[state * input.edge_t_min_stride];
        output.edge_t_max[row] = input.edge_t_max[state * input.edge_t_max_stride];
        copy_strided_vec3(
            input.n0,
            input.n0_row_stride,
            input.n0_column_stride,
            state,
            output.n0,
            row);
        copy_strided_vec3(
            input.n1,
            input.n1_row_stride,
            input.n1_column_stride,
            state,
            output.n1,
            row);
        output.prim0[row] = input.prim0[state * input.prim0_stride];
        output.prim1[row] = input.prim1[state * input.prim1_stride];
        output.exterior_angle[row] =
            input.exterior_angle[state * input.exterior_angle_stride];
        copy_strided_vec3(
            input.source,
            input.source_row_stride,
            input.source_column_stride,
            state,
            output.source,
            row);
        output.source_power[row] =
            input.source_power[state * input.source_power_stride];
        output.valid[row] = true;
    }
}

__global__ void diffraction_state_selection_overflow_kernel(
    const bool *__restrict__ overflow) {
    if (blockIdx.x == 0 && threadIdx.x == 0 && overflow[0]) {
        asm volatile("trap;");
    }
}
std::vector<at::Tensor> diffraction_state_capacity_select_cuda_impl(
    at::Tensor active,
    at::Tensor edge_index,
    at::Tensor edge_position,
    at::Tensor edge_direction,
    at::Tensor edge_t_min,
    at::Tensor edge_t_max,
    at::Tensor n0,
    at::Tensor n1,
    at::Tensor prim0,
    at::Tensor prim1,
    at::Tensor exterior_angle,
    at::Tensor source,
    at::Tensor source_power,
    int64_t state_capacity) {
    check_tensor(active, "active", at::kBool, 1);
    const auto check_state_tensor = [&active](
                                      const at::Tensor& tensor,
                                      const char *name,
                                      c10::ScalarType dtype,
                                      int64_t rank) {
        TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
        TORCH_CHECK(tensor.scalar_type() == dtype, name, " has the wrong dtype");
        TORCH_CHECK(tensor.dim() == rank, name, " has the wrong rank");
        TORCH_CHECK(
            tensor.get_device() == active.get_device(),
            name,
            " must share active device");
    };
    check_state_tensor(edge_index, "edge_index", at::kInt, 1);
    check_state_tensor(edge_position, "edge_position", at::kFloat, 2);
    check_state_tensor(edge_direction, "edge_direction", at::kFloat, 2);
    check_state_tensor(edge_t_min, "edge_t_min", at::kFloat, 1);
    check_state_tensor(edge_t_max, "edge_t_max", at::kFloat, 1);
    check_state_tensor(n0, "n0", at::kFloat, 2);
    check_state_tensor(n1, "n1", at::kFloat, 2);
    check_state_tensor(prim0, "prim0", at::kInt, 1);
    check_state_tensor(prim1, "prim1", at::kInt, 1);
    check_state_tensor(exterior_angle, "exterior_angle", at::kFloat, 1);
    check_state_tensor(source, "source", at::kFloat, 2);
    check_state_tensor(source_power, "source_power", at::kFloat, 1);
    TORCH_CHECK(state_capacity >= 0, "state_capacity must be non-negative");

    const int64_t state_count = active.size(0);
    TORCH_CHECK(
        state_count <= 4'194'304,
        "diffraction state capacity exceeds 4194304");
    for (const auto& named_tensor : std::initializer_list<std::pair<const char *, const at::Tensor *>>{
             {"edge_index", &edge_index},
             {"edge_position", &edge_position},
             {"edge_direction", &edge_direction},
             {"edge_t_min", &edge_t_min},
             {"edge_t_max", &edge_t_max},
             {"n0", &n0},
             {"n1", &n1},
             {"prim0", &prim0},
             {"prim1", &prim1},
             {"exterior_angle", &exterior_angle},
             {"source", &source},
             {"source_power", &source_power}}) {
        TORCH_CHECK(
            named_tensor.second->size(0) == state_count,
            named_tensor.first,
            " must share active row capacity");
    }
    for (const auto& named_tensor : std::initializer_list<std::pair<const char *, const at::Tensor *>>{
             {"edge_position", &edge_position},
             {"edge_direction", &edge_direction},
             {"n0", &n0},
             {"n1", &n1},
             {"source", &source}}) {
        TORCH_CHECK(
            named_tensor.second->size(1) == 3,
            named_tensor.first,
            " must have shape (N, 3)");
    }

    const int64_t capacity = std::min(state_capacity, state_count);
    auto int_options = active.options().dtype(at::kInt);
    auto float_options = active.options().dtype(at::kFloat);
    auto state_edge_index = at::empty({capacity}, int_options);
    auto state_edge_position = at::empty({capacity, 3}, float_options);
    auto state_edge_direction = at::empty({capacity, 3}, float_options);
    auto state_edge_t_min = at::empty({capacity}, float_options);
    auto state_edge_t_max = at::empty({capacity}, float_options);
    auto state_n0 = at::empty({capacity, 3}, float_options);
    auto state_n1 = at::empty({capacity, 3}, float_options);
    auto state_prim0 = at::empty({capacity}, int_options);
    auto state_prim1 = at::empty({capacity}, int_options);
    auto state_exterior_angle = at::empty({capacity}, float_options);
    auto state_source = at::empty({capacity, 3}, float_options);
    auto state_source_power = at::empty({capacity}, float_options);
    auto valid = at::empty({capacity}, active.options());
    auto actual_count = at::empty({1}, int_options);
    auto overflow = at::empty({1}, active.options());

    const DiffractionStateInputView input_view{
        edge_index.data_ptr<int>(), edge_index.stride(0),
        edge_position.data_ptr<float>(), edge_position.stride(0), edge_position.stride(1),
        edge_direction.data_ptr<float>(), edge_direction.stride(0), edge_direction.stride(1),
        edge_t_min.data_ptr<float>(), edge_t_min.stride(0),
        edge_t_max.data_ptr<float>(), edge_t_max.stride(0),
        n0.data_ptr<float>(), n0.stride(0), n0.stride(1),
        n1.data_ptr<float>(), n1.stride(0), n1.stride(1),
        prim0.data_ptr<int>(), prim0.stride(0),
        prim1.data_ptr<int>(), prim1.stride(0),
        exterior_angle.data_ptr<float>(), exterior_angle.stride(0),
        source.data_ptr<float>(), source.stride(0), source.stride(1),
        source_power.data_ptr<float>(), source_power.stride(0)};
    const DiffractionStateOutputView output_view{
        state_edge_index.data_ptr<int>(),
        state_edge_position.data_ptr<float>(),
        state_edge_direction.data_ptr<float>(),
        state_edge_t_min.data_ptr<float>(),
        state_edge_t_max.data_ptr<float>(),
        state_n0.data_ptr<float>(),
        state_n1.data_ptr<float>(),
        state_prim0.data_ptr<int>(),
        state_prim1.data_ptr<int>(),
        state_exterior_angle.data_ptr<float>(),
        state_source.data_ptr<float>(),
        state_source_power.data_ptr<float>(),
        valid.data_ptr<bool>()};

    cudaStream_t stream = at::cuda::getCurrentCUDAStream(active.get_device()).stream();
    diffraction_state_selection_status_init_kernel<<<1, 1, 0, stream>>>(
        actual_count.data_ptr<int>(), overflow.data_ptr<bool>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    if (capacity > 0) {
        const int capacity_blocks = static_cast<int>(
            (capacity + kDiffractionBlockSize - 1) / kDiffractionBlockSize);
        diffraction_state_capacity_init_kernel<<<
            capacity_blocks, kDiffractionBlockSize, 0, stream>>>(
            output_view, capacity);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

    if (state_count > 0) {
        auto flags = at::empty({state_count}, int_options);
        auto offsets = at::empty({state_count}, int_options);
        const int state_blocks = static_cast<int>(
            (state_count + kDiffractionBlockSize - 1) / kDiffractionBlockSize);
        diffraction_state_selection_flags_kernel<<<
            state_blocks, kDiffractionBlockSize, 0, stream>>>(
            active.data_ptr<bool>(), flags.data_ptr<int>(), state_count);
        C10_CUDA_KERNEL_LAUNCH_CHECK();

        size_t scratch_bytes = 0;
        C10_CUDA_CHECK(cub::DeviceScan::ExclusiveSum(
            nullptr,
            scratch_bytes,
            flags.data_ptr<int>(),
            offsets.data_ptr<int>(),
            static_cast<int>(state_count),
            stream));
        auto scratch = at::empty(
            {static_cast<int64_t>(scratch_bytes)},
            active.options().dtype(at::kByte));
        C10_CUDA_CHECK(cub::DeviceScan::ExclusiveSum(
            scratch.data_ptr<uint8_t>(),
            scratch_bytes,
            flags.data_ptr<int>(),
            offsets.data_ptr<int>(),
            static_cast<int>(state_count),
            stream));

        diffraction_state_selection_status_kernel<<<1, 1, 0, stream>>>(
            flags.data_ptr<int>(),
            offsets.data_ptr<int>(),
            state_count,
            capacity,
            actual_count.data_ptr<int>(),
            overflow.data_ptr<bool>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        diffraction_state_capacity_gather_kernel<<<
            state_blocks, kDiffractionBlockSize, 0, stream>>>(
            active.data_ptr<bool>(),
            offsets.data_ptr<int>(),
            overflow.data_ptr<bool>(),
            input_view,
            output_view,
            state_count);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        diffraction_state_selection_overflow_kernel<<<1, 1, 0, stream>>>(
            overflow.data_ptr<bool>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

    return {
        state_edge_index,
        state_edge_position,
        state_edge_direction,
        state_edge_t_min,
        state_edge_t_max,
        state_n0,
        state_n1,
        state_prim0,
        state_prim1,
        state_exterior_angle,
        state_source,
        state_source_power,
        valid,
        actual_count,
        overflow};
}


}  // namespace

std::vector<at::Tensor> cn_deterministic_diffraction_state_capacity_select_cuda(
    at::Tensor active,
    at::Tensor edge_index,
    at::Tensor edge_position,
    at::Tensor edge_direction,
    at::Tensor edge_t_min,
    at::Tensor edge_t_max,
    at::Tensor n0,
    at::Tensor n1,
    at::Tensor prim0,
    at::Tensor prim1,
    at::Tensor exterior_angle,
    at::Tensor source,
    at::Tensor source_power,
    int64_t state_capacity) {
    return diffraction_state_capacity_select_cuda_impl(
        active,
        edge_index,
        edge_position,
        edge_direction,
        edge_t_min,
        edge_t_max,
        n0,
        n1,
        prim0,
        prim1,
        exterior_angle,
        source,
        source_power,
        state_capacity);
}
