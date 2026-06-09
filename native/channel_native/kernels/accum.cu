#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime_api.h>

#include <tuple>

namespace {

constexpr int kFinalizeBlockSize = 256;
constexpr int kComponentMapBlockSize = 256;

void check_component_map(const at::Tensor &tensor, const char *name) {
    TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(tensor.scalar_type() == at::kFloat, name, " must be float32");
    TORCH_CHECK(tensor.dim() == 3, name, " must have shape (tx, y, x)");
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

void check_source_map(const at::Tensor &tensor, const char *name) {
    TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(tensor.scalar_type() == at::kFloat, name, " must be float32");
    TORCH_CHECK(tensor.dim() == 2, name, " must have shape (y, x)");
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

__global__ void store_component_map_kernel(
    float *__restrict__ maps,
    const float *__restrict__ source,
    const float *__restrict__ scale_values,
    int64_t tx_index,
    int64_t scale_index,
    int use_scale,
    int64_t element_count) {
    const float scale = use_scale ? scale_values[scale_index] : 1.0f;
    const int64_t base = tx_index * element_count;
    const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         idx < element_count;
         idx += stride) {
        maps[base + idx] = source[idx] * scale;
    }
}

__global__ void finalize_component_maps_kernel(
    const float *__restrict__ los,
    const float *__restrict__ reflection,
    const float *__restrict__ diffraction,
    float *__restrict__ path_gain,
    float *__restrict__ los_power,
    float *__restrict__ reflection_power,
    float *__restrict__ diffraction_power,
    int64_t element_count) {
    __shared__ float los_sum[kFinalizeBlockSize];
    __shared__ float reflection_sum[kFinalizeBlockSize];
    __shared__ float diffraction_sum[kFinalizeBlockSize];

    const int tid = threadIdx.x;
    const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    float local_los = 0.0f;
    float local_reflection = 0.0f;
    float local_diffraction = 0.0f;

    for (int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + tid;
         idx < element_count;
         idx += stride) {
        const float los_value = los[idx];
        const float reflection_value = reflection[idx];
        const float diffraction_value = diffraction[idx];
        path_gain[idx] = los_value + reflection_value + diffraction_value;
        local_los += los_value;
        local_reflection += reflection_value;
        local_diffraction += diffraction_value;
    }

    los_sum[tid] = local_los;
    reflection_sum[tid] = local_reflection;
    diffraction_sum[tid] = local_diffraction;
    __syncthreads();

    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (tid < offset) {
            los_sum[tid] += los_sum[tid + offset];
            reflection_sum[tid] += reflection_sum[tid + offset];
            diffraction_sum[tid] += diffraction_sum[tid + offset];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(los_power, los_sum[0]);
        atomicAdd(reflection_power, reflection_sum[0]);
        atomicAdd(diffraction_power, diffraction_sum[0]);
    }
}

}  // namespace

at::Tensor cn_mc_component_map_buffer_cuda(
    at::Tensor reference,
    int64_t tx_count,
    int64_t dim0,
    int64_t dim1) {
    TORCH_CHECK(reference.is_cuda(), "reference must be a CUDA tensor");
    TORCH_CHECK(reference.scalar_type() == at::kFloat, "reference must be float32");
    TORCH_CHECK(tx_count >= 0, "tx_count must be non-negative");
    TORCH_CHECK(dim0 >= 0, "dim0 must be non-negative");
    TORCH_CHECK(dim1 >= 0, "dim1 must be non-negative");
    return at::zeros({tx_count, dim0, dim1}, reference.options());
}

at::Tensor cn_mc_store_component_map_cuda(
    at::Tensor maps,
    at::Tensor source,
    int64_t tx_index) {
    check_component_map(maps, "maps");
    check_source_map(source, "source");
    TORCH_CHECK(tx_index >= 0 && tx_index < maps.size(0), "tx_index is out of bounds");
    TORCH_CHECK(source.size(0) == maps.size(1), "source dim0 must match maps");
    TORCH_CHECK(source.size(1) == maps.size(2), "source dim1 must match maps");

    const int64_t element_count = source.numel();
    if (element_count > 0) {
        cudaStream_t stream = at::cuda::getCurrentCUDAStream(maps.get_device()).stream();
        const int block_count = static_cast<int>(
            (element_count + kComponentMapBlockSize - 1) / kComponentMapBlockSize);
        store_component_map_kernel<<<block_count, kComponentMapBlockSize, 0, stream>>>(
            maps.data_ptr<float>(),
            source.data_ptr<float>(),
            nullptr,
            tx_index,
            0,
            0,
            element_count);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return maps;
}

at::Tensor cn_mc_store_scaled_component_map_cuda(
    at::Tensor maps,
    at::Tensor source,
    at::Tensor scale_values,
    int64_t tx_index,
    int64_t scale_index) {
    check_component_map(maps, "maps");
    check_source_map(source, "source");
    TORCH_CHECK(scale_values.is_cuda(), "scale_values must be a CUDA tensor");
    TORCH_CHECK(scale_values.scalar_type() == at::kFloat, "scale_values must be float32");
    TORCH_CHECK(scale_values.dim() == 1, "scale_values must have shape (N,)");
    TORCH_CHECK(scale_values.is_contiguous(), "scale_values must be contiguous");
    TORCH_CHECK(tx_index >= 0 && tx_index < maps.size(0), "tx_index is out of bounds");
    TORCH_CHECK(scale_index >= 0 && scale_index < scale_values.size(0), "scale_index is out of bounds");
    TORCH_CHECK(source.size(0) == maps.size(1), "source dim0 must match maps");
    TORCH_CHECK(source.size(1) == maps.size(2), "source dim1 must match maps");

    const int64_t element_count = source.numel();
    if (element_count > 0) {
        cudaStream_t stream = at::cuda::getCurrentCUDAStream(maps.get_device()).stream();
        const int block_count = static_cast<int>(
            (element_count + kComponentMapBlockSize - 1) / kComponentMapBlockSize);
        store_component_map_kernel<<<block_count, kComponentMapBlockSize, 0, stream>>>(
            maps.data_ptr<float>(),
            source.data_ptr<float>(),
            scale_values.data_ptr<float>(),
            tx_index,
            scale_index,
            1,
            element_count);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return maps;
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> cn_mc_finalize_component_maps_cuda(
    at::Tensor los,
    at::Tensor reflection,
    at::Tensor diffraction) {
    check_component_map(los, "los");
    check_component_map(reflection, "reflection");
    check_component_map(diffraction, "diffraction");
    TORCH_CHECK(reflection.sizes() == los.sizes(), "reflection must match los shape");
    TORCH_CHECK(diffraction.sizes() == los.sizes(), "diffraction must match los shape");

    auto path_gain = at::empty({los.size(0), los.size(1) * los.size(2)}, los.options());
    auto los_power = at::empty({}, los.options());
    auto reflection_power = at::empty({}, los.options());
    auto diffraction_power = at::empty({}, los.options());

    cudaStream_t stream = at::cuda::getCurrentCUDAStream(los.get_device()).stream();
    C10_CUDA_CHECK(cudaMemsetAsync(los_power.data_ptr<float>(), 0, sizeof(float), stream));
    C10_CUDA_CHECK(cudaMemsetAsync(reflection_power.data_ptr<float>(), 0, sizeof(float), stream));
    C10_CUDA_CHECK(cudaMemsetAsync(diffraction_power.data_ptr<float>(), 0, sizeof(float), stream));

    const int64_t element_count = los.numel();
    if (element_count > 0) {
        const int block_count = static_cast<int>(
            (element_count + kFinalizeBlockSize - 1) / kFinalizeBlockSize);
        finalize_component_maps_kernel<<<block_count, kFinalizeBlockSize, 0, stream>>>(
            los.data_ptr<float>(),
            reflection.data_ptr<float>(),
            diffraction.data_ptr<float>(),
            path_gain.data_ptr<float>(),
            los_power.data_ptr<float>(),
            reflection_power.data_ptr<float>(),
            diffraction_power.data_ptr<float>(),
            element_count);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

    return {path_gain, los_power, reflection_power, diffraction_power};
}
