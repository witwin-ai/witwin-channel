#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/util/complex.h>
#include <cuda_runtime_api.h>
#include <torch/extension.h>

#include "../tensor_checks.h"
#include "capacity_failure_state.h"
#include "evaluated_paths_continuous_gather_ad.cuh"

#include <cstdint>
#include <limits>
#include <optional>

namespace {

constexpr int kGatherAdBlockSize = 256;
using cfloat = c10::complex<float>;
using channel_native::check_tensor;
using channel_native::evaluated_paths_ad::AllocatedContinuous;
using channel_native::evaluated_paths_ad::ContinuousOutputs;
using channel_native::evaluated_paths_ad::allocate_continuous;

template <typename T>
struct OptionalView {
    const T *data;
    int64_t stride0;
    int64_t stride1;
    int64_t stride2;
    bool conjugated;
};

struct ContinuousViews {
    OptionalView<float> path_length_m;
    OptionalView<float> delay_s;
    OptionalView<float> field_direction;
    OptionalView<float> interaction_position;
    OptionalView<float> interaction_normal;
    OptionalView<float> interaction_positions;
    OptionalView<float> interaction_normals;
    OptionalView<float> path_gain;
    OptionalView<cfloat> path_field;
    OptionalView<cfloat> field_xyz;
    OptionalView<cfloat> coefficient;
};

__device__ __forceinline__ float apply_conjugation(float value, bool) {
    return value;
}

__device__ __forceinline__ cfloat apply_conjugation(
    cfloat value,
    bool conjugated) {
    return conjugated ? cfloat(value.real(), -value.imag()) : value;
}

template <typename T>
__device__ __forceinline__ T read_scalar(
    const OptionalView<T>& view,
    int64_t row) {
    return view.data == nullptr
        ? T(0)
        : apply_conjugation(view.data[row * view.stride0], view.conjugated);
}

template <typename T>
__device__ __forceinline__ T read_vector(
    const OptionalView<T>& view,
    int64_t row,
    int64_t component) {
    return view.data == nullptr
        ? T(0)
        : apply_conjugation(
              view.data[row * view.stride0 + component * view.stride1],
              view.conjugated);
}

template <typename T>
__device__ __forceinline__ T read_sequence_vector(
    const OptionalView<T>& view,
    int64_t row,
    int64_t slot,
    int64_t component) {
    return view.data == nullptr
        ? T(0)
        : apply_conjugation(
              view.data[
                  row * view.stride0 + slot * view.stride1 +
                  component * view.stride2],
              view.conjugated);
}

__global__ void canonical_gather_ad_init_kernel(
    ContinuousOutputs output,
    int64_t rows,
    int64_t sequence_width) {
    const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (int64_t row = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         row < rows;
         row += stride) {
        output.path_length_m[row] = 0.0f;
        output.delay_s[row] = 0.0f;
        output.path_gain[row] = 0.0f;
        output.path_field[row] = cfloat(0.0f, 0.0f);
        output.coefficient[row] = cfloat(0.0f, 0.0f);
        const int64_t vec = row * 3;
        for (int component = 0; component < 3; ++component) {
            output.field_direction[vec + component] = 0.0f;
            output.interaction_position[vec + component] = 0.0f;
            output.interaction_normal[vec + component] = 0.0f;
            output.field_xyz[vec + component] = cfloat(0.0f, 0.0f);
        }
        const int64_t sequence_vec = row * sequence_width * 3;
        for (int64_t item = 0; item < sequence_width * 3; ++item) {
            output.interaction_positions[sequence_vec + item] = 0.0f;
            output.interaction_normals[sequence_vec + item] = 0.0f;
        }
    }
}

__global__ void canonical_gather_ad_selection_kernel(
    int *__restrict__ failure_state,
    const bool *__restrict__ valid,
    const int64_t *__restrict__ selected_row_index,
    unsigned int *__restrict__ seen_source,
    int64_t row_capacity,
    int64_t candidate_count) {
    if (failure_state[0] != 0) {
        return;
    }
    const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (int64_t row = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         row < row_capacity;
         row += stride) {
        if (!valid[row]) {
            continue;
        }
        if (row > 0 && !valid[row - 1]) {
            atomicOr(failure_state, channel_native::capacity::kPairContractError);
            continue;
        }
        const int64_t source = selected_row_index[row];
        if (source < 0 || source >= candidate_count) {
            atomicOr(failure_state, channel_native::capacity::kPairContractError);
            continue;
        }
        const int64_t word = source >> 5;
        const unsigned int mask = 1u << static_cast<unsigned int>(source & 31);
        const unsigned int previous = atomicOr(seen_source + word, mask);
        if ((previous & mask) != 0) {
            atomicOr(failure_state, channel_native::capacity::kPairContractError);
        }
    }
}

__global__ void canonical_gather_backward_kernel(
    const int *__restrict__ failure_state,
    const bool *__restrict__ valid,
    const int64_t *__restrict__ selected_row_index,
    ContinuousViews grad_output,
    ContinuousOutputs grad_input,
    int64_t row_capacity,
    int64_t sequence_width) {
    if (failure_state[0] != 0) {
        return;
    }
    const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (int64_t destination =
             static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         destination < row_capacity;
         destination += stride) {
        if (!valid[destination]) {
            continue;
        }
        const int64_t source = selected_row_index[destination];
        grad_input.path_length_m[source] =
            read_scalar(grad_output.path_length_m, destination);
        grad_input.delay_s[source] = read_scalar(grad_output.delay_s, destination);
        grad_input.path_gain[source] = read_scalar(grad_output.path_gain, destination);
        grad_input.path_field[source] = read_scalar(grad_output.path_field, destination);
        grad_input.coefficient[source] = read_scalar(grad_output.coefficient, destination);
        const int64_t source_vec = source * 3;
        for (int component = 0; component < 3; ++component) {
            grad_input.field_direction[source_vec + component] =
                read_vector(grad_output.field_direction, destination, component);
            grad_input.interaction_position[source_vec + component] =
                read_vector(grad_output.interaction_position, destination, component);
            grad_input.interaction_normal[source_vec + component] =
                read_vector(grad_output.interaction_normal, destination, component);
            grad_input.field_xyz[source_vec + component] =
                read_vector(grad_output.field_xyz, destination, component);
        }
        const int64_t source_sequence = source * sequence_width * 3;
        for (int64_t slot = 0; slot < sequence_width; ++slot) {
            for (int component = 0; component < 3; ++component) {
                const int64_t item = source_sequence + slot * 3 + component;
                grad_input.interaction_positions[item] = read_sequence_vector(
                    grad_output.interaction_positions,
                    destination,
                    slot,
                    component);
                grad_input.interaction_normals[item] = read_sequence_vector(
                    grad_output.interaction_normals,
                    destination,
                    slot,
                    component);
            }
        }
    }
}

__global__ void canonical_gather_jvp_kernel(
    const int *__restrict__ failure_state,
    const bool *__restrict__ valid,
    const int64_t *__restrict__ selected_row_index,
    ContinuousViews tangent_input,
    ContinuousOutputs tangent_output,
    int64_t row_capacity,
    int64_t sequence_width) {
    if (failure_state[0] != 0) {
        return;
    }
    const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (int64_t destination =
             static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         destination < row_capacity;
         destination += stride) {
        if (!valid[destination]) {
            continue;
        }
        const int64_t source = selected_row_index[destination];
        tangent_output.path_length_m[destination] =
            read_scalar(tangent_input.path_length_m, source);
        tangent_output.delay_s[destination] = read_scalar(tangent_input.delay_s, source);
        tangent_output.path_gain[destination] = read_scalar(tangent_input.path_gain, source);
        tangent_output.path_field[destination] =
            read_scalar(tangent_input.path_field, source);
        tangent_output.coefficient[destination] =
            read_scalar(tangent_input.coefficient, source);
        const int64_t destination_vec = destination * 3;
        for (int component = 0; component < 3; ++component) {
            tangent_output.field_direction[destination_vec + component] =
                read_vector(tangent_input.field_direction, source, component);
            tangent_output.interaction_position[destination_vec + component] =
                read_vector(tangent_input.interaction_position, source, component);
            tangent_output.interaction_normal[destination_vec + component] =
                read_vector(tangent_input.interaction_normal, source, component);
            tangent_output.field_xyz[destination_vec + component] =
                read_vector(tangent_input.field_xyz, source, component);
        }
        const int64_t destination_sequence = destination * sequence_width * 3;
        for (int64_t slot = 0; slot < sequence_width; ++slot) {
            for (int component = 0; component < 3; ++component) {
                const int64_t item = destination_sequence + slot * 3 + component;
                tangent_output.interaction_positions[item] = read_sequence_vector(
                    tangent_input.interaction_positions, source, slot, component);
                tangent_output.interaction_normals[item] = read_sequence_vector(
                    tangent_input.interaction_normals, source, slot, component);
            }
        }
    }
}

int launch_blocks(int64_t count) {
    return static_cast<int>((count + kGatherAdBlockSize - 1) / kGatherAdBlockSize);
}

void check_selection(
    const at::Tensor& failure_state,
    const at::Tensor& valid,
    const at::Tensor& selected_row_index,
    int64_t candidate_count) {
    check_tensor(valid, "valid", at::kBool, 1);
    check_tensor(selected_row_index, "selected_row_index", at::kLong, 1);
    TORCH_CHECK(selected_row_index.sizes() == valid.sizes(), "selected_row_index must match valid");
    TORCH_CHECK(selected_row_index.get_device() == valid.get_device(), "selected_row_index must share valid device");
    channel_native::capacity::validate_failure_state(failure_state, valid);
    TORCH_CHECK(candidate_count >= 0, "candidate_count must be non-negative");
    TORCH_CHECK(candidate_count <= std::numeric_limits<int>::max(), "candidate_count exceeds int32 status capacity");
}

void launch_selection_status(
    at::Tensor failure_state,
    const at::Tensor& valid,
    const at::Tensor& selected_row_index,
    int64_t candidate_count,
    cudaStream_t stream) {
    const int64_t row_capacity = valid.size(0);
    const int64_t seen_words = (candidate_count + 31) / 32;
    auto seen_source = at::empty({seen_words}, valid.options().dtype(at::kInt));
    if (seen_words > 0) {
        C10_CUDA_CHECK(cudaMemsetAsync(
            seen_source.data_ptr<int>(), 0, seen_words * sizeof(int), stream));
    }
    if (row_capacity > 0) {
        canonical_gather_ad_selection_kernel<<<
            launch_blocks(row_capacity), kGatherAdBlockSize, 0, stream>>>(
            failure_state.data_ptr<int>(), valid.data_ptr<bool>(),
            selected_row_index.data_ptr<int64_t>(),
            reinterpret_cast<unsigned int *>(seen_source.data_ptr<int>()),
            row_capacity, candidate_count);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
}

}  // namespace

pybind11::dict cn_evaluated_paths_canonical_capacity_gather_backward(
    at::Tensor failure_state,
    at::Tensor valid,
    at::Tensor selected_row_index,
    std::optional<at::Tensor> grad_path_length_m,
    std::optional<at::Tensor> grad_delay_s,
    std::optional<at::Tensor> grad_field_direction,
    std::optional<at::Tensor> grad_interaction_position,
    std::optional<at::Tensor> grad_interaction_normal,
    std::optional<at::Tensor> grad_interaction_positions,
    std::optional<at::Tensor> grad_interaction_normals,
    std::optional<at::Tensor> grad_path_gain,
    std::optional<at::Tensor> grad_path_field,
    std::optional<at::Tensor> grad_field_xyz,
    std::optional<at::Tensor> grad_coefficient,
    int64_t candidate_count,
    int64_t sequence_width) {
    check_selection(failure_state, valid, selected_row_index, candidate_count);
    TORCH_CHECK(sequence_width >= 0, "sequence_width must be non-negative");
    const int device = valid.get_device();
    const int64_t row_capacity = valid.size(0);
    auto views = channel_native::evaluated_paths_ad::make_continuous_views<
        ContinuousViews, OptionalView<float>, OptionalView<cfloat>>(
        grad_path_length_m, grad_delay_s, grad_field_direction,
        grad_interaction_position, grad_interaction_normal,
        grad_interaction_positions, grad_interaction_normals, grad_path_gain,
        grad_path_field, grad_field_xyz, grad_coefficient, row_capacity,
        sequence_width, device, true);
    auto outputs = allocate_continuous(valid, candidate_count, sequence_width);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(device).stream();
    if (candidate_count > 0) {
        canonical_gather_ad_init_kernel<<<
            launch_blocks(candidate_count), kGatherAdBlockSize, 0, stream>>>(
            outputs.view(), candidate_count, sequence_width);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    launch_selection_status(
        failure_state, valid, selected_row_index, candidate_count, stream);
    if (row_capacity > 0) {
        canonical_gather_backward_kernel<<<
            launch_blocks(row_capacity), kGatherAdBlockSize, 0, stream>>>(
            failure_state.data_ptr<int>(), valid.data_ptr<bool>(),
            selected_row_index.data_ptr<int64_t>(), views, outputs.view(),
            row_capacity, sequence_width);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return outputs.dict();
}

pybind11::dict cn_evaluated_paths_canonical_capacity_gather_jvp(
    at::Tensor failure_state,
    at::Tensor valid,
    at::Tensor selected_row_index,
    std::optional<at::Tensor> tangent_path_length_m,
    std::optional<at::Tensor> tangent_delay_s,
    std::optional<at::Tensor> tangent_field_direction,
    std::optional<at::Tensor> tangent_interaction_position,
    std::optional<at::Tensor> tangent_interaction_normal,
    std::optional<at::Tensor> tangent_interaction_positions,
    std::optional<at::Tensor> tangent_interaction_normals,
    std::optional<at::Tensor> tangent_path_gain,
    std::optional<at::Tensor> tangent_path_field,
    std::optional<at::Tensor> tangent_field_xyz,
    std::optional<at::Tensor> tangent_coefficient,
    int64_t candidate_count,
    int64_t sequence_width) {
    check_selection(failure_state, valid, selected_row_index, candidate_count);
    TORCH_CHECK(sequence_width >= 0, "sequence_width must be non-negative");
    const int device = valid.get_device();
    const int64_t row_capacity = valid.size(0);
    auto views = channel_native::evaluated_paths_ad::make_continuous_views<
        ContinuousViews, OptionalView<float>, OptionalView<cfloat>>(
        tangent_path_length_m, tangent_delay_s, tangent_field_direction,
        tangent_interaction_position, tangent_interaction_normal,
        tangent_interaction_positions, tangent_interaction_normals,
        tangent_path_gain, tangent_path_field, tangent_field_xyz,
        tangent_coefficient, candidate_count, sequence_width, device, true);
    auto outputs = allocate_continuous(valid, row_capacity, sequence_width);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(device).stream();
    if (row_capacity > 0) {
        canonical_gather_ad_init_kernel<<<
            launch_blocks(row_capacity), kGatherAdBlockSize, 0, stream>>>(
            outputs.view(), row_capacity, sequence_width);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    launch_selection_status(
        failure_state, valid, selected_row_index, candidate_count, stream);
    if (row_capacity > 0) {
        canonical_gather_jvp_kernel<<<
            launch_blocks(row_capacity), kGatherAdBlockSize, 0, stream>>>(
            failure_state.data_ptr<int>(), valid.data_ptr<bool>(),
            selected_row_index.data_ptr<int64_t>(), views, outputs.view(),
            row_capacity, sequence_width);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return outputs.dict();
}
