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
