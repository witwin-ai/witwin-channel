#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime_api.h>
#include "torch_cuda_minimal.h"
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/scan.h>

#include "../tensor_checks.h"

#include <cstdint>
#include <limits>
#include <optional>

namespace {

constexpr int kBlockSize = 256;
constexpr int kLoSComponentId = 0;

enum ContractError : int {
    kIndexBounds = 1 << 0,
    kPairOrder = 1 << 1,
    kDepth = 1 << 2,
    kComponent = 1 << 3,
    kStableId = 1 << 4,
};

struct OptionalVector {
    const float* data;
    int64_t stride0;
    int64_t stride1;
    bool present;
};

struct OptionalScalar {
    const float* data;
    int64_t stride0;
    bool present;
};

__global__ void validate_fixed_los_kernel(
    const int* __restrict__ source_index,
    const int* __restrict__ sink_index,
    const int64_t* __restrict__ source_id,
    const int64_t* __restrict__ sink_id,
    const int* __restrict__ depth,
    const int* __restrict__ component_id,
    const int64_t* __restrict__ source_stable_ids,
    const int64_t* __restrict__ sink_stable_ids,
    int64_t row_count,
    int64_t source_count,
    int64_t sink_count,
    int64_t* __restrict__ pair_index,
    int* __restrict__ contract_error) {
    const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (int64_t row =
             static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         row < row_count;
         row += stride) {
        const int source = source_index[row];
        const int sink = sink_index[row];
        const bool in_bounds =
            source >= 0 && static_cast<int64_t>(source) < source_count &&
            sink >= 0 && static_cast<int64_t>(sink) < sink_count;
        int error = 0;
        int64_t pair = -1;
        if (!in_bounds) {
            error |= kIndexBounds;
        } else {
            pair = static_cast<int64_t>(sink) * source_count + source;
            if (source_id[row] != source_stable_ids[source] ||
                sink_id[row] != sink_stable_ids[sink]) {
                error |= kStableId;
            }
            if (row > 0) {
                const int previous_source = source_index[row - 1];
                const int previous_sink = sink_index[row - 1];
                const bool previous_in_bounds =
                    previous_source >= 0 &&
                    static_cast<int64_t>(previous_source) < source_count &&
                    previous_sink >= 0 &&
                    static_cast<int64_t>(previous_sink) < sink_count;
                if (previous_in_bounds) {
                    const int64_t previous_pair =
                        static_cast<int64_t>(previous_sink) * source_count +
                        previous_source;
                    if (pair < previous_pair) {
                        error |= kPairOrder;
                    }
                }
            }
        }
        pair_index[row] = pair;
        if (depth[row] != 0) {
            error |= kDepth;
        }
        if (component_id[row] != kLoSComponentId) {
            error |= kComponent;
        }
        if (error != 0) {
            atomicOr(contract_error, error);
        }
    }
}

__global__ void gather_fixed_los_kernel(
    const int* __restrict__ source_index,
    const int* __restrict__ sink_index,
    const int64_t* __restrict__ pair_index,
    const float* __restrict__ source_positions,
    const float* __restrict__ sink_positions,
    const float* __restrict__ source_powers,
    const float* __restrict__ source_polarizations,
    const float* __restrict__ sink_polarizations,
    int64_t row_count,
    float* __restrict__ source,
    float* __restrict__ target,
    float* __restrict__ tx_power,
    float* __restrict__ tx_polarization,
    float* __restrict__ rx_polarization,
    int64_t* __restrict__ pair_offsets) {
    const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (int64_t row =
             static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         row < row_count;
         row += stride) {
        const int source_row = source_index[row];
        const int sink_row = sink_index[row];
        const int64_t output_base = row * 3;
        const int64_t source_base = static_cast<int64_t>(source_row) * 3;
        const int64_t sink_base = static_cast<int64_t>(sink_row) * 3;
        for (int component = 0; component < 3; ++component) {
            source[output_base + component] =
                source_positions[source_base + component];
            target[output_base + component] =
                sink_positions[sink_base + component];
            tx_polarization[output_base + component] =
                source_polarizations[source_base + component];
            rx_polarization[output_base + component] =
                sink_polarizations[sink_base + component];
        }
        tx_power[row] = source_powers[source_row];
        atomicAdd(
            reinterpret_cast<unsigned long long*>(
                pair_offsets + pair_index[row] + 1),
            1ULL);
    }
}

__device__ __forceinline__ float read_vector(
    const OptionalVector& view,
    int64_t row,
    int component) {
    return view.present
        ? view.data[row * view.stride0 + component * view.stride1]
        : 0.0f;
}

__device__ __forceinline__ float read_scalar(
    const OptionalScalar& view,
    int64_t row) {
    return view.present ? view.data[row * view.stride0] : 0.0f;
}

__global__ void fixed_los_backward_kernel(
    const int* __restrict__ source_index,
    const int* __restrict__ sink_index,
    OptionalVector grad_source,
    OptionalVector grad_target,
    OptionalScalar grad_tx_power,
    OptionalVector grad_tx_polarization,
    OptionalVector grad_rx_polarization,
    int64_t row_count,
    float* __restrict__ grad_source_positions,
    float* __restrict__ grad_sink_positions,
    float* __restrict__ grad_source_powers,
    float* __restrict__ grad_source_polarizations,
    float* __restrict__ grad_sink_polarizations) {
    const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (int64_t row =
             static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         row < row_count;
         row += stride) {
        const int source_row = source_index[row];
        const int sink_row = sink_index[row];
        const int64_t source_base = static_cast<int64_t>(source_row) * 3;
        const int64_t sink_base = static_cast<int64_t>(sink_row) * 3;
        for (int component = 0; component < 3; ++component) {
            if (grad_source.present) {
                atomicAdd(
                    grad_source_positions + source_base + component,
                    read_vector(grad_source, row, component));
            }
            if (grad_target.present) {
                atomicAdd(
                    grad_sink_positions + sink_base + component,
                    read_vector(grad_target, row, component));
            }
            if (grad_tx_polarization.present) {
                atomicAdd(
                    grad_source_polarizations + source_base + component,
                    read_vector(grad_tx_polarization, row, component));
            }
            if (grad_rx_polarization.present) {
                atomicAdd(
                    grad_sink_polarizations + sink_base + component,
                    read_vector(grad_rx_polarization, row, component));
            }
        }
        if (grad_tx_power.present) {
            atomicAdd(
                grad_source_powers + source_row,
                read_scalar(grad_tx_power, row));
        }
    }
}

__global__ void fixed_los_jvp_kernel(
    const int* __restrict__ source_index,
    const int* __restrict__ sink_index,
    OptionalVector tangent_source_positions,
    OptionalVector tangent_sink_positions,
    OptionalScalar tangent_source_powers,
    OptionalVector tangent_source_polarizations,
    OptionalVector tangent_sink_polarizations,
    int64_t row_count,
    float* __restrict__ tangent_source,
    float* __restrict__ tangent_target,
    float* __restrict__ tangent_tx_power,
    float* __restrict__ tangent_tx_polarization,
    float* __restrict__ tangent_rx_polarization) {
    const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (int64_t row =
             static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         row < row_count;
         row += stride) {
        const int source_row = source_index[row];
        const int sink_row = sink_index[row];
        const int64_t output_base = row * 3;
        for (int component = 0; component < 3; ++component) {
            tangent_source[output_base + component] = read_vector(
                tangent_source_positions, source_row, component);
            tangent_target[output_base + component] = read_vector(
                tangent_sink_positions, sink_row, component);
            tangent_tx_polarization[output_base + component] = read_vector(
                tangent_source_polarizations, source_row, component);
            tangent_rx_polarization[output_base + component] = read_vector(
                tangent_sink_polarizations, sink_row, component);
        }
        tangent_tx_power[row] =
            read_scalar(tangent_source_powers, source_row);
    }
}

int launch_blocks(int64_t count) {
    return static_cast<int>((count + kBlockSize - 1) / kBlockSize);
}

void check_row_tensor(
    const at::Tensor& tensor,
    const char* name,
    c10::ScalarType dtype,
    int64_t rows,
    int device) {
    channel::check_tensor(tensor, name, dtype, 1);
    TORCH_CHECK(tensor.size(0) == rows, name, " must share topology rows");
    TORCH_CHECK(tensor.get_device() == device, name, " must share topology device");
}

void check_endpoint_vec3(
    const at::Tensor& tensor,
    const char* name,
    int64_t rows,
    int device) {
    channel::check_vec3_table(tensor, name);
    TORCH_CHECK(tensor.size(0) == rows, name, " has the wrong endpoint count");
    TORCH_CHECK(tensor.get_device() == device, name, " must share topology device");
}

void check_endpoint_vector(
    const at::Tensor& tensor,
    const char* name,
    c10::ScalarType dtype,
    int64_t rows,
    int device) {
    channel::check_tensor(tensor, name, dtype, 1);
    TORCH_CHECK(tensor.size(0) == rows, name, " has the wrong endpoint count");
    TORCH_CHECK(tensor.get_device() == device, name, " must share topology device");
}

OptionalVector optional_vector(
    const std::optional<at::Tensor>& tensor,
    const char* name,
    int64_t rows,
    int device) {
    if (!tensor.has_value()) {
        return {nullptr, 0, 0, false};
    }
    TORCH_CHECK(tensor->is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(tensor->scalar_type() == at::kFloat, name, " must use float32");
    TORCH_CHECK(
        tensor->dim() == 2 && tensor->size(0) == rows && tensor->size(1) == 3,
        name,
        " must have shape (K, 3)");
    TORCH_CHECK(tensor->get_device() == device, name, " must share topology device");
    return {
        tensor->data_ptr<float>(),
        tensor->stride(0),
        tensor->stride(1),
        true};
}

OptionalScalar optional_scalar(
    const std::optional<at::Tensor>& tensor,
    const char* name,
    int64_t rows,
    int device) {
    if (!tensor.has_value()) {
        return {nullptr, 0, false};
    }
    TORCH_CHECK(tensor->is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(tensor->scalar_type() == at::kFloat, name, " must use float32");
    TORCH_CHECK(
        tensor->dim() == 1 && tensor->size(0) == rows,
        name,
        " must have shape (K,)");
    TORCH_CHECK(tensor->get_device() == device, name, " must share topology device");
    return {tensor->data_ptr<float>(), tensor->stride(0), true};
}

void check_selection(
    const at::Tensor& source_index,
    const at::Tensor& sink_index) {
    channel::check_tensor(source_index, "source_index", at::kInt, 1);
    check_row_tensor(
        sink_index,
        "sink_index",
        at::kInt,
        source_index.size(0),
        source_index.get_device());
}

}  // namespace

pybind11::dict channel_consumer_fixed_los_gather(
    at::Tensor source_index,
    at::Tensor sink_index,
    at::Tensor source_id,
    at::Tensor sink_id,
    at::Tensor depth,
    at::Tensor component_id,
    at::Tensor source_positions,
    at::Tensor sink_positions,
    at::Tensor source_powers,
    at::Tensor source_polarizations,
    at::Tensor sink_polarizations,
    at::Tensor source_stable_ids,
    at::Tensor sink_stable_ids) {
    check_selection(source_index, sink_index);
    const int64_t row_count = source_index.size(0);
    const int device = source_index.get_device();
    check_row_tensor(source_id, "source_id", at::kLong, row_count, device);
    check_row_tensor(sink_id, "sink_id", at::kLong, row_count, device);
    check_row_tensor(depth, "depth", at::kInt, row_count, device);
    check_row_tensor(component_id, "component_id", at::kInt, row_count, device);
    channel::check_vec3_table(source_positions, "source_positions");
    channel::check_vec3_table(sink_positions, "sink_positions");
    const int64_t source_count = source_positions.size(0);
    const int64_t sink_count = sink_positions.size(0);
    check_endpoint_vec3(
        source_polarizations,
        "source_polarizations",
        source_count,
        device);
    check_endpoint_vec3(
        sink_polarizations,
        "sink_polarizations",
        sink_count,
        device);
    check_endpoint_vector(
        source_powers,
        "source_powers",
        at::kFloat,
        source_count,
        device);
    check_endpoint_vector(
        source_stable_ids,
        "source_stable_ids",
        at::kLong,
        source_count,
        device);
    check_endpoint_vector(
        sink_stable_ids,
        "sink_stable_ids",
        at::kLong,
        sink_count,
        device);
    TORCH_CHECK(
        source_positions.get_device() == device &&
            sink_positions.get_device() == device,
        "endpoint positions must share topology device");
    TORCH_CHECK(
        source_count == 0 ||
            sink_count <= std::numeric_limits<int64_t>::max() / source_count,
        "source/sink pair count overflows int64");
    const int64_t pair_count = source_count * sink_count;

    auto float_options = source_positions.options();
    auto long_options = source_index.options().dtype(at::kLong);
    cudaStream_t stream =
        at::cuda::getCurrentCUDAStream(device).stream();
    auto source = at::empty({row_count, 3}, float_options);
    auto target = at::empty({row_count, 3}, float_options);
    auto tx_power = at::empty({row_count}, float_options);
    auto tx_polarization = at::empty({row_count, 3}, float_options);
    auto rx_polarization = at::empty({row_count, 3}, float_options);
    auto pair_index = at::empty({row_count}, long_options);
    auto pair_offsets = channel::empty_zero_cuda(
        {pair_count + 1}, long_options, stream);

    if (row_count > 0) {
        auto contract_error = channel::empty_zero_cuda(
            {1}, source_index.options().dtype(at::kInt), stream);
        validate_fixed_los_kernel<<<
            launch_blocks(row_count), kBlockSize, 0, stream>>>(
            source_index.data_ptr<int>(),
            sink_index.data_ptr<int>(),
            source_id.data_ptr<int64_t>(),
            sink_id.data_ptr<int64_t>(),
            depth.data_ptr<int>(),
            component_id.data_ptr<int>(),
            source_stable_ids.data_ptr<int64_t>(),
            sink_stable_ids.data_ptr<int64_t>(),
            row_count,
            source_count,
            sink_count,
            pair_index.data_ptr<int64_t>(),
            contract_error.data_ptr<int>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        int host_error = 0;
        C10_CUDA_CHECK(cudaMemcpyAsync(
            &host_error,
            contract_error.data_ptr<int>(),
            sizeof(int),
            cudaMemcpyDeviceToHost,
            stream));
        C10_CUDA_CHECK(cudaStreamSynchronize(stream));
        TORCH_CHECK(
            host_error == 0,
            "fixed LoS topology validation failed (error bitmask ",
            host_error,
            ")");

        gather_fixed_los_kernel<<<
            launch_blocks(row_count), kBlockSize, 0, stream>>>(
            source_index.data_ptr<int>(),
            sink_index.data_ptr<int>(),
            pair_index.data_ptr<int64_t>(),
            source_positions.data_ptr<float>(),
            sink_positions.data_ptr<float>(),
            source_powers.data_ptr<float>(),
            source_polarizations.data_ptr<float>(),
            sink_polarizations.data_ptr<float>(),
            row_count,
            source.data_ptr<float>(),
            target.data_ptr<float>(),
            tx_power.data_ptr<float>(),
            tx_polarization.data_ptr<float>(),
            rx_polarization.data_ptr<float>(),
            pair_offsets.data_ptr<int64_t>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        thrust::inclusive_scan(
            thrust::cuda::par.on(stream),
            thrust::device_pointer_cast(pair_offsets.data_ptr<int64_t>()),
            thrust::device_pointer_cast(
                pair_offsets.data_ptr<int64_t>() + pair_count + 1),
            thrust::device_pointer_cast(pair_offsets.data_ptr<int64_t>()));
    }

    pybind11::dict result;
    result["source"] = source;
    result["target"] = target;
    result["tx_power"] = tx_power;
    result["tx_polarization"] = tx_polarization;
    result["rx_polarization"] = rx_polarization;
    result["pair_index"] = pair_index;
    result["pair_offsets"] = pair_offsets;
    return result;
}

pybind11::dict channel_consumer_fixed_los_gather_backward(
    at::Tensor source_index,
    at::Tensor sink_index,
    std::optional<at::Tensor> grad_source,
    std::optional<at::Tensor> grad_target,
    std::optional<at::Tensor> grad_tx_power,
    std::optional<at::Tensor> grad_tx_polarization,
    std::optional<at::Tensor> grad_rx_polarization,
    int64_t source_count,
    int64_t sink_count) {
    check_selection(source_index, sink_index);
    TORCH_CHECK(source_count >= 0, "source_count must be non-negative");
    TORCH_CHECK(sink_count >= 0, "sink_count must be non-negative");
    const int64_t row_count = source_index.size(0);
    const int device = source_index.get_device();
    cudaStream_t stream =
        at::cuda::getCurrentCUDAStream(device).stream();
    const OptionalVector source_view =
        optional_vector(grad_source, "grad_source", row_count, device);
    const OptionalVector target_view =
        optional_vector(grad_target, "grad_target", row_count, device);
    const OptionalScalar power_view =
        optional_scalar(grad_tx_power, "grad_tx_power", row_count, device);
    const OptionalVector tx_pol_view = optional_vector(
        grad_tx_polarization,
        "grad_tx_polarization",
        row_count,
        device);
    const OptionalVector rx_pol_view = optional_vector(
        grad_rx_polarization,
        "grad_rx_polarization",
        row_count,
        device);
    auto float_options = source_index.options().dtype(at::kFloat);
    auto grad_source_positions = channel::empty_zero_cuda(
        {source_count, 3}, float_options, stream);
    auto grad_sink_positions = channel::empty_zero_cuda(
        {sink_count, 3}, float_options, stream);
    auto grad_source_powers = channel::empty_zero_cuda(
        {source_count}, float_options, stream);
    auto grad_source_polarizations =
        channel::empty_zero_cuda({source_count, 3}, float_options, stream);
    auto grad_sink_polarizations =
        channel::empty_zero_cuda({sink_count, 3}, float_options, stream);
    if (row_count > 0) {
        fixed_los_backward_kernel<<<
            launch_blocks(row_count), kBlockSize, 0, stream>>>(
            source_index.data_ptr<int>(),
            sink_index.data_ptr<int>(),
            source_view,
            target_view,
            power_view,
            tx_pol_view,
            rx_pol_view,
            row_count,
            grad_source_positions.data_ptr<float>(),
            grad_sink_positions.data_ptr<float>(),
            grad_source_powers.data_ptr<float>(),
            grad_source_polarizations.data_ptr<float>(),
            grad_sink_polarizations.data_ptr<float>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    pybind11::dict result;
    result["source_positions"] = grad_source_positions;
    result["sink_positions"] = grad_sink_positions;
    result["source_powers"] = grad_source_powers;
    result["source_polarizations"] = grad_source_polarizations;
    result["sink_polarizations"] = grad_sink_polarizations;
    return result;
}

pybind11::dict channel_consumer_fixed_los_gather_jvp(
    at::Tensor source_index,
    at::Tensor sink_index,
    std::optional<at::Tensor> tangent_source_positions,
    std::optional<at::Tensor> tangent_sink_positions,
    std::optional<at::Tensor> tangent_source_powers,
    std::optional<at::Tensor> tangent_source_polarizations,
    std::optional<at::Tensor> tangent_sink_polarizations,
    int64_t source_count,
    int64_t sink_count) {
    check_selection(source_index, sink_index);
    TORCH_CHECK(source_count >= 0, "source_count must be non-negative");
    TORCH_CHECK(sink_count >= 0, "sink_count must be non-negative");
    const int64_t row_count = source_index.size(0);
    const int device = source_index.get_device();
    cudaStream_t stream =
        at::cuda::getCurrentCUDAStream(device).stream();
    const OptionalVector source_view = optional_vector(
        tangent_source_positions,
        "tangent_source_positions",
        source_count,
        device);
    const OptionalVector target_view = optional_vector(
        tangent_sink_positions,
        "tangent_sink_positions",
        sink_count,
        device);
    const OptionalScalar power_view = optional_scalar(
        tangent_source_powers,
        "tangent_source_powers",
        source_count,
        device);
    const OptionalVector tx_pol_view = optional_vector(
        tangent_source_polarizations,
        "tangent_source_polarizations",
        source_count,
        device);
    const OptionalVector rx_pol_view = optional_vector(
        tangent_sink_polarizations,
        "tangent_sink_polarizations",
        sink_count,
        device);
    auto float_options = source_index.options().dtype(at::kFloat);
    auto tangent_source = channel::empty_zero_cuda(
        {row_count, 3}, float_options, stream);
    auto tangent_target = channel::empty_zero_cuda(
        {row_count, 3}, float_options, stream);
    auto tangent_tx_power = channel::empty_zero_cuda(
        {row_count}, float_options, stream);
    auto tangent_tx_polarization = channel::empty_zero_cuda(
        {row_count, 3}, float_options, stream);
    auto tangent_rx_polarization = channel::empty_zero_cuda(
        {row_count, 3}, float_options, stream);
    if (row_count > 0) {
        fixed_los_jvp_kernel<<<
            launch_blocks(row_count), kBlockSize, 0, stream>>>(
            source_index.data_ptr<int>(),
            sink_index.data_ptr<int>(),
            source_view,
            target_view,
            power_view,
            tx_pol_view,
            rx_pol_view,
            row_count,
            tangent_source.data_ptr<float>(),
            tangent_target.data_ptr<float>(),
            tangent_tx_power.data_ptr<float>(),
            tangent_tx_polarization.data_ptr<float>(),
            tangent_rx_polarization.data_ptr<float>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    pybind11::dict result;
    result["source"] = tangent_source;
    result["target"] = tangent_target;
    result["tx_power"] = tangent_tx_power;
    result["tx_polarization"] = tangent_tx_polarization;
    result["rx_polarization"] = tangent_rx_polarization;
    return result;
}
