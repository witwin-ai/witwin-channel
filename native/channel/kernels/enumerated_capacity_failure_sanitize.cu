#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/util/complex.h>
#include <cuda_runtime_api.h>
#include "torch_cuda_minimal.h"

#include "../tensor_checks.h"
#include "capacity_failure_state.h"
#include "evaluated_paths_payload_plumbing.h"

#include <cstdint>

namespace {

constexpr int kBlockSize = 256;
using cfloat = c10::complex<float>;

__global__ void enumerated_capacity_failure_sanitize_kernel(
    const int *failure_state,
    const bool *input_valid,
    channel::evaluated_paths::PayloadInputView input,
    int64_t *selected_row_index,
    bool *output_valid,
    channel::evaluated_paths::PayloadOutputView output,
    int64_t rows,
    int64_t sequence_width) {
    const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (int64_t row = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         row < rows;
         row += stride) {
        selected_row_index[row] = -1;
        output_valid[row] = false;
        channel::evaluated_paths::initialize_row(output, row, sequence_width);
        if (failure_state[0] != 0 || !input_valid[row]) {
            continue;
        }
        selected_row_index[row] = row;
        output_valid[row] = true;
        channel::evaluated_paths::copy_row(
            input, output, row, row, sequence_width);
    }
}

int launch_blocks(int64_t count) {
    return static_cast<int>((count + kBlockSize - 1) / kBlockSize);
}

__global__ void enumerated_capacity_failure_vector_sanitize_kernel(
    const int *failure_state,
    const cfloat *input,
    cfloat *output,
    int64_t dim0,
    int64_t dim1,
    int64_t stride0,
    int64_t stride1,
    int64_t stride2) {
    const int64_t count = dim0 * dim1 * 3;
    const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < count;
         index += stride) {
        const int64_t component = index % 3;
        const int64_t column = (index / 3) % dim1;
        const int64_t row = index / (dim1 * 3);
        const int64_t source = row * stride0 + column * stride1 + component * stride2;
        output[index] = failure_state[0] == 0
            ? input[source]
            : cfloat(0.0f, 0.0f);
    }
}

}  // namespace

pybind11::dict channel_enumerated_capacity_failure_sanitize(
    at::Tensor failure_state,
    at::Tensor valid,
    at::Tensor tx_id,
    at::Tensor rx_id,
    at::Tensor depth,
    at::Tensor component_id,
    at::Tensor primitive_id,
    at::Tensor edge_id,
    at::Tensor material_id,
    at::Tensor primitive_sequence,
    at::Tensor material_sequence,
    at::Tensor interaction_type,
    at::Tensor path_length_m,
    at::Tensor delay_s,
    at::Tensor field_direction,
    at::Tensor interaction_position,
    at::Tensor interaction_normal,
    at::Tensor interaction_positions,
    at::Tensor interaction_normals,
    at::Tensor path_gain,
    at::Tensor path_field,
    at::Tensor field_xyz,
    at::Tensor coefficient) {
    channel::check_tensor(valid, "valid", at::kBool, 1);
    channel::capacity::validate_failure_state(failure_state, valid);
    const int64_t rows = valid.size(0);
    const int device = valid.get_device();
    const channel::evaluated_paths::PayloadTensors payload{
        tx_id, rx_id, depth, component_id, primitive_id, edge_id, material_id,
        primitive_sequence, material_sequence, interaction_type, path_length_m,
        delay_s, field_direction, interaction_position, interaction_normal,
        interaction_positions, interaction_normals, path_gain, path_field,
        field_xyz, coefficient};
    const int64_t sequence_width =
        channel::evaluated_paths::validate_payload(payload, rows, device);
    const c10::cuda::CUDAGuard device_guard(valid.device());
    auto selected_row_index = at::empty({rows}, valid.options().dtype(at::kLong));
    auto output_valid = at::empty({rows}, valid.options().dtype(at::kBool));
    auto output = channel::evaluated_paths::allocate_payload(
        valid, rows, sequence_width);
    if (rows > 0) {
        const auto input = channel::evaluated_paths::input_view(payload);
        const auto output_view = channel::evaluated_paths::output_view(output);
        cudaStream_t stream = at::cuda::getCurrentCUDAStream(device).stream();
        enumerated_capacity_failure_sanitize_kernel<<<
            launch_blocks(rows), kBlockSize, 0, stream>>>(
            failure_state.data_ptr<int>(), valid.data_ptr<bool>(), input,
            selected_row_index.data_ptr<int64_t>(), output_valid.data_ptr<bool>(),
            output_view, rows, sequence_width);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    pybind11::dict result;
    result["selected_row_index"] = selected_row_index;
    result["valid"] = output_valid;
    output.append_to(result);
    return result;
}

at::Tensor channel_enumerated_capacity_failure_vector_sanitize(
    at::Tensor failure_state,
    at::Tensor values) {
    TORCH_CHECK(values.is_cuda(), "diffraction_vector_field must be a CUDA tensor");
    TORCH_CHECK(
        values.scalar_type() == at::kComplexFloat,
        "diffraction_vector_field must use complex64");
    TORCH_CHECK(values.dim() == 3, "diffraction_vector_field must have rank 3");
    TORCH_CHECK(values.size(2) == 3, "diffraction_vector_field must end in vec3");
    channel::capacity::validate_failure_state(failure_state, values);
    const c10::cuda::CUDAGuard device_guard(values.device());
    auto output = at::empty(values.sizes(), values.options());
    const int64_t count = values.numel();
    if (count > 0) {
        cudaStream_t stream =
            at::cuda::getCurrentCUDAStream(values.get_device()).stream();
        enumerated_capacity_failure_vector_sanitize_kernel<<<
            launch_blocks(count), kBlockSize, 0, stream>>>(
            failure_state.data_ptr<int>(), values.data_ptr<cfloat>(),
            output.data_ptr<cfloat>(), values.size(0), values.size(1),
            values.stride(0), values.stride(1), values.stride(2));
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return output;
}
