// ADR-044 consolidated CUDA translation unit.
// Physical co-location only: ABI, launches, synchronization, and numerical order are unchanged.

// ---- Consolidated from capacity_failure_state.cu ----
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime_api.h>

#include "../tensor_checks.h"
#include "capacity_failure_state.h"

void channel::capacity::validate_failure_state(
    const at::Tensor& failure_state,
    const at::Tensor& reference) {
    channel::check_tensor(
        failure_state, "failure_state", at::kInt, 1);
    TORCH_CHECK(
        failure_state.size(0) == 1,
        "failure_state must have shape (1,)");
    TORCH_CHECK(
        failure_state.get_device() == reference.get_device(),
        "failure_state must share the input device");
}

at::Tensor channel_capacity_failure_state_create(at::Tensor reference) {
    TORCH_CHECK(reference.is_cuda(), "reference must be a CUDA tensor");
    auto failure_state = at::empty({1}, reference.options().dtype(at::kInt));
    cudaStream_t stream =
        at::cuda::getCurrentCUDAStream(reference.get_device()).stream();
    C10_CUDA_CHECK(cudaMemsetAsync(
        failure_state.data_ptr<int>(), 0, sizeof(int), stream));
    return failure_state;
}

// ---- Consolidated from capacity_failure_terminal.cu ----
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime_api.h>

#include "../tensor_checks.h"

namespace {

__global__ void capacity_failure_terminal_check_kernel(
    const int *__restrict__ failure_state) {
    if (failure_state[0] != 0) {
        asm volatile("trap;");
    }
}

}  // namespace

void channel_capacity_failure_terminal_check(at::Tensor failure_state) {
    channel::check_tensor(
        failure_state, "failure_state", at::kInt, 1);
    TORCH_CHECK(
        failure_state.size(0) == 1,
        "failure_state must have shape (1,)");

    cudaStream_t stream =
        at::cuda::getCurrentCUDAStream(failure_state.get_device()).stream();
    capacity_failure_terminal_check_kernel<<<1, 1, 0, stream>>>(
        failure_state.data_ptr<int>());
    // A post-launch status query could observe this deliberately tiny trap
    // before the caller's next normal synchronization boundary.
}

// ---- Consolidated from enumerated_capacity_failure_sanitize.cu ----
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

// ---- Consolidated from mc_capacity_failure_component_maps_sanitize.cu ----
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime_api.h>
#include "torch_cuda_minimal.h"

#include "../tensor_checks.h"
#include "capacity_failure_state.h"

#include <array>
#include <cstdint>
#include <optional>

#define kBlockSize kMcSanitizeBlockSize

namespace {

constexpr int kBlockSize = 256;
constexpr int kMapCount = 5;

struct MapInputView {
    const float* data;
    int64_t stride0;
    int64_t stride1;
    int64_t stride2;
};

struct MapInputViews {
    MapInputView maps[kMapCount];
};

struct MapOutputViews {
    float* maps[kMapCount];
};

__global__ void capacity_failure_component_maps_sanitize_kernel(
    const int* failure_state,
    MapInputViews input,
    MapOutputViews output,
    int64_t dim1,
    int64_t dim2,
    int64_t count) {
    const int64_t grid_stride =
        static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (int64_t index =
             static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < count;
         index += grid_stride) {
        const int64_t axis2 = index % dim2;
        const int64_t axis1 = (index / dim2) % dim1;
        const int64_t axis0 = index / (dim1 * dim2);
        if (failure_state[0] != 0) {
            for (int map = 0; map < kMapCount; ++map) {
                output.maps[map][index] = 0.0f;
            }
            continue;
        }
        for (int map = 0; map < kMapCount; ++map) {
            const MapInputView view = input.maps[map];
            const int64_t source =
                axis0 * view.stride0 + axis1 * view.stride1 +
                axis2 * view.stride2;
            output.maps[map][index] =
                view.data != nullptr ? view.data[source] : 0.0f;
        }
    }
}

void validate_map(
    const at::Tensor& value,
    const char* name,
    const at::Tensor& reference,
    bool require_contiguous) {
    TORCH_CHECK(value.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(value.scalar_type() == at::kFloat, name, " must use float32");
    TORCH_CHECK(value.dim() == 3, name, " must have rank 3");
    TORCH_CHECK(value.sizes() == reference.sizes(), name, " must match los shape");
    TORCH_CHECK(value.device() == reference.device(), name, " must share the los device");
    if (require_contiguous) {
        TORCH_CHECK(value.is_contiguous(), name, " must be contiguous");
    }
}

MapInputView input_view(const std::optional<at::Tensor>& value) {
    if (!value.has_value()) {
        return {nullptr, 0, 0, 0};
    }
    return {
        value->data_ptr<float>(),
        value->stride(0),
        value->stride(1),
        value->stride(2),
    };
}

pybind11::dict sanitize_maps(
    const at::Tensor& failure_state,
    const at::Tensor& reference,
    const std::array<std::optional<at::Tensor>, kMapCount>& values,
    bool require_all,
    bool require_contiguous) {
    channel::check_tensor(reference, "reference", at::kFloat, 3);
    channel::capacity::validate_failure_state(failure_state, reference);
    constexpr std::array<const char*, kMapCount> names{
        "los", "reflection", "diffraction", "transmission", "scattering"};
    for (int map = 0; map < kMapCount; ++map) {
        TORCH_CHECK(
            !require_all || values[map].has_value(),
            names[map],
            " must be present");
        if (values[map].has_value()) {
            validate_map(
                *values[map], names[map], reference, require_contiguous);
        }
    }

    const c10::cuda::CUDAGuard device_guard(reference.device());
    std::array<at::Tensor, kMapCount> outputs;
    MapInputViews input{};
    MapOutputViews output{};
    for (int map = 0; map < kMapCount; ++map) {
        outputs[map] = at::empty(reference.sizes(), reference.options());
        input.maps[map] = input_view(values[map]);
        output.maps[map] = outputs[map].data_ptr<float>();
    }
    const int64_t count = reference.numel();
    if (count > 0) {
        const int blocks = static_cast<int>((count + kBlockSize - 1) / kBlockSize);
        cudaStream_t stream =
            at::cuda::getCurrentCUDAStream(reference.get_device()).stream();
        capacity_failure_component_maps_sanitize_kernel<<<
            blocks, kBlockSize, 0, stream>>>(
            failure_state.data_ptr<int>(), input, output,
            reference.size(1), reference.size(2), count);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

    pybind11::dict result;
    for (int map = 0; map < kMapCount; ++map) {
        result[names[map]] = outputs[map];
    }
    return result;
}

}  // namespace

pybind11::dict channel_mc_capacity_failure_component_maps_sanitize(
    at::Tensor failure_state,
    at::Tensor los,
    at::Tensor reflection,
    at::Tensor diffraction,
    at::Tensor transmission,
    at::Tensor scattering) {
    const std::array<std::optional<at::Tensor>, kMapCount> values{
        los, reflection, diffraction, transmission, scattering};
    return sanitize_maps(failure_state, los, values, true, true);
}

pybind11::dict channel_mc_capacity_failure_component_maps_sanitize_backward(
    at::Tensor failure_state,
    at::Tensor reference,
    std::optional<at::Tensor> grad_los,
    std::optional<at::Tensor> grad_reflection,
    std::optional<at::Tensor> grad_diffraction,
    std::optional<at::Tensor> grad_transmission,
    std::optional<at::Tensor> grad_scattering) {
    const std::array<std::optional<at::Tensor>, kMapCount> values{
        grad_los, grad_reflection, grad_diffraction,
        grad_transmission, grad_scattering};
    return sanitize_maps(failure_state, reference, values, false, false);
}

pybind11::dict channel_mc_capacity_failure_component_maps_sanitize_jvp(
    at::Tensor failure_state,
    at::Tensor reference,
    std::optional<at::Tensor> tangent_los,
    std::optional<at::Tensor> tangent_reflection,
    std::optional<at::Tensor> tangent_diffraction,
    std::optional<at::Tensor> tangent_transmission,
    std::optional<at::Tensor> tangent_scattering) {
    const std::array<std::optional<at::Tensor>, kMapCount> values{
        tangent_los, tangent_reflection, tangent_diffraction,
        tangent_transmission, tangent_scattering};
    return sanitize_maps(failure_state, reference, values, false, false);
}

#undef kBlockSize

