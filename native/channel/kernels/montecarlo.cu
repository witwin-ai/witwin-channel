// Copyright Xingyu Chen.
// Implements shared Monte Carlo CUDA operations.

// ==== Section: Monte Carlo accumulation ====
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime_api.h>
#include "math.cuh"

#include <tuple>

#define check_component_map check_shared_component_map

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
    const float *__restrict__ transmission,
    const float *__restrict__ scattering,
    float *__restrict__ path_gain,
    float *__restrict__ los_power,
    float *__restrict__ reflection_power,
    float *__restrict__ diffraction_power,
    float *__restrict__ transmission_power,
    float *__restrict__ scattering_power,
    int64_t element_count) {
    __shared__ float los_sum[kFinalizeBlockSize];
    __shared__ float reflection_sum[kFinalizeBlockSize];
    __shared__ float diffraction_sum[kFinalizeBlockSize];
    __shared__ float transmission_sum[kFinalizeBlockSize];
    __shared__ float scattering_sum[kFinalizeBlockSize];

    const int tid = threadIdx.x;
    const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    float local_los = 0.0f;
    float local_reflection = 0.0f;
    float local_diffraction = 0.0f;
    float local_transmission = 0.0f;
    float local_scattering = 0.0f;

    for (int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + tid;
         idx < element_count;
         idx += stride) {
        const float los_value = los[idx];
        const float reflection_value = reflection[idx];
        const float diffraction_value = diffraction[idx];
        const float transmission_value = transmission[idx];
        const float scattering_value = scattering[idx];
        path_gain[idx] = los_value + reflection_value + diffraction_value +
            transmission_value + scattering_value;
        local_los += los_value;
        local_reflection += reflection_value;
        local_diffraction += diffraction_value;
        local_transmission += transmission_value;
        local_scattering += scattering_value;
    }

    los_sum[tid] = local_los;
    reflection_sum[tid] = local_reflection;
    diffraction_sum[tid] = local_diffraction;
    transmission_sum[tid] = local_transmission;
    scattering_sum[tid] = local_scattering;
    __syncthreads();

    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (tid < offset) {
            los_sum[tid] += los_sum[tid + offset];
            reflection_sum[tid] += reflection_sum[tid + offset];
            diffraction_sum[tid] += diffraction_sum[tid + offset];
            transmission_sum[tid] += transmission_sum[tid + offset];
            scattering_sum[tid] += scattering_sum[tid + offset];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(los_power, los_sum[0]);
        atomicAdd(reflection_power, reflection_sum[0]);
        atomicAdd(diffraction_power, diffraction_sum[0]);
        atomicAdd(transmission_power, transmission_sum[0]);
        atomicAdd(scattering_power, scattering_sum[0]);
    }
}

}  // namespace

at::Tensor channel_mc_component_map_buffer_cuda(
    at::Tensor reference,
    int64_t tx_count,
    int64_t dim0,
    int64_t dim1) {
    TORCH_CHECK(reference.is_cuda(), "reference must be a CUDA tensor");
    TORCH_CHECK(reference.scalar_type() == at::kFloat, "reference must be float32");
    TORCH_CHECK(tx_count >= 0, "tx_count must be non-negative");
    TORCH_CHECK(dim0 >= 0, "dim0 must be non-negative");
    TORCH_CHECK(dim1 >= 0, "dim1 must be non-negative");
    auto maps = at::empty({tx_count, dim0, dim1}, reference.options());
    if (maps.numel() > 0) {
        cudaStream_t stream = at::cuda::getCurrentCUDAStream(reference.get_device()).stream();
        C10_CUDA_CHECK(cudaMemsetAsync(maps.data_ptr<float>(), 0, maps.numel() * sizeof(float), stream));
    }
    return maps;
}

at::Tensor channel_mc_store_component_map_cuda(
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

at::Tensor channel_mc_store_scaled_component_map_cuda(
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

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor> channel_mc_finalize_component_maps_cuda(
    at::Tensor los,
    at::Tensor reflection,
    at::Tensor diffraction,
    at::Tensor transmission,
    at::Tensor scattering) {
    check_component_map(los, "los");
    check_component_map(reflection, "reflection");
    check_component_map(diffraction, "diffraction");
    check_component_map(transmission, "transmission");
    check_component_map(scattering, "scattering");
    TORCH_CHECK(reflection.sizes() == los.sizes(), "reflection must match los shape");
    TORCH_CHECK(diffraction.sizes() == los.sizes(), "diffraction must match los shape");
    TORCH_CHECK(transmission.sizes() == los.sizes(), "transmission must match los shape");
    TORCH_CHECK(scattering.sizes() == los.sizes(), "scattering must match los shape");

    auto path_gain = at::empty({los.size(0), los.size(1) * los.size(2)}, los.options());
    auto los_power = at::empty({}, los.options());
    auto reflection_power = at::empty({}, los.options());
    auto diffraction_power = at::empty({}, los.options());
    auto transmission_power = at::empty({}, los.options());
    auto scattering_power = at::empty({}, los.options());

    cudaStream_t stream = at::cuda::getCurrentCUDAStream(los.get_device()).stream();
    C10_CUDA_CHECK(cudaMemsetAsync(los_power.data_ptr<float>(), 0, sizeof(float), stream));
    C10_CUDA_CHECK(cudaMemsetAsync(reflection_power.data_ptr<float>(), 0, sizeof(float), stream));
    C10_CUDA_CHECK(cudaMemsetAsync(diffraction_power.data_ptr<float>(), 0, sizeof(float), stream));
    C10_CUDA_CHECK(cudaMemsetAsync(transmission_power.data_ptr<float>(), 0, sizeof(float), stream));
    C10_CUDA_CHECK(cudaMemsetAsync(scattering_power.data_ptr<float>(), 0, sizeof(float), stream));

    const int64_t element_count = los.numel();
    if (element_count > 0) {
        const int block_count = static_cast<int>(
            (element_count + kFinalizeBlockSize - 1) / kFinalizeBlockSize);
        finalize_component_maps_kernel<<<block_count, kFinalizeBlockSize, 0, stream>>>(
            los.data_ptr<float>(),
            reflection.data_ptr<float>(),
            diffraction.data_ptr<float>(),
            transmission.data_ptr<float>(),
            scattering.data_ptr<float>(),
            path_gain.data_ptr<float>(),
            los_power.data_ptr<float>(),
            reflection_power.data_ptr<float>(),
            diffraction_power.data_ptr<float>(),
            transmission_power.data_ptr<float>(),
            scattering_power.data_ptr<float>(),
            element_count);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

    return {path_gain, los_power, reflection_power, diffraction_power, transmission_power, scattering_power};
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor> channel_bdpt_finalize_component_maps_cuda(
    at::Tensor los,
    at::Tensor reflection,
    at::Tensor diffraction,
    at::Tensor transmission,
    at::Tensor scattering) {
    auto [path_gain, los_power, reflection_power, diffraction_power,
          transmission_power, scattering_power] =
        channel_mc_finalize_component_maps_cuda(
            los, reflection, diffraction, transmission, scattering);
    return {
        path_gain.view(los.sizes()),
        los_power,
        reflection_power,
        diffraction_power,
        transmission_power,
        scattering_power};
}

#undef check_component_map

// ==== Section: BDPT accumulation ====
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime_api.h>
#include "math.cuh"
#include "torch_cuda.h"

#include <algorithm>

namespace {

void check_path_gain(const at::Tensor& tensor) {
    TORCH_CHECK(tensor.is_cuda(), "path_gain must be a CUDA tensor");
    TORCH_CHECK(tensor.scalar_type() == at::kFloat, "path_gain must be float32");
    TORCH_CHECK(tensor.dim() == 2, "path_gain must have rank 2");
    TORCH_CHECK(tensor.is_contiguous(), "path_gain must be contiguous");
}

__global__ void pack_int2_kernel(
    int64_t count,
    const int* x,
    const int* y,
    int* out) {
    int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= count) {
        return;
    }
    out[index * 2 + 0] = x[index];
    out[index * 2 + 1] = y[index];
}

}  // namespace

at::Tensor channel_core_pack_int2_cuda(at::Tensor x, at::Tensor y) {
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(y.is_cuda(), "y must be a CUDA tensor");
    TORCH_CHECK(x.scalar_type() == at::kInt, "x must be int32");
    TORCH_CHECK(y.scalar_type() == at::kInt, "y must be int32");
    TORCH_CHECK(x.dim() == 1, "x must have rank 1");
    TORCH_CHECK(y.dim() == 1, "y must have rank 1");
    TORCH_CHECK(x.sizes() == y.sizes(), "x and y must have the same shape");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    TORCH_CHECK(y.is_contiguous(), "y must be contiguous");

    const int64_t count = x.numel();
    auto out = at::empty({count, 2}, x.options().dtype(at::kInt));
    if (count > 0) {
        cudaStream_t stream = at::cuda::getCurrentCUDAStream(x.get_device()).stream();
        constexpr int kBlockSize = 256;
        int blocks = static_cast<int>((count + kBlockSize - 1) / kBlockSize);
        pack_int2_kernel<<<blocks, kBlockSize, 0, stream>>>(
            count,
            x.data_ptr<int>(),
            y.data_ptr<int>(),
            out.data_ptr<int>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return out;
}

at::Tensor channel_bdpt_zero_matrix_cuda(at::Tensor reference, int64_t rows, int64_t cols) {
    TORCH_CHECK(reference.is_cuda(), "reference must be a CUDA tensor");
    TORCH_CHECK(reference.scalar_type() == at::kFloat, "reference must be float32");
    TORCH_CHECK(rows >= 0 && cols >= 0, "rows and cols must be non-negative");
    auto out = at::empty({rows, cols}, reference.options().dtype(at::kFloat));
    if (out.numel() > 0) {
        cudaStream_t stream = at::cuda::getCurrentCUDAStream(reference.get_device()).stream();
        C10_CUDA_CHECK(cudaMemsetAsync(out.data_ptr<float>(), 0, out.numel() * sizeof(float), stream));
    }
    return out;
}

__global__ void bdpt_finalize_point_components_kernel(
    int64_t count,
    const float* los,
    const float* reflection,
    const float* diffraction,
    const float* transmission,
    const float* scattering,
    float* path_gain,
    float* los_power,
    float* reflection_power,
    float* diffraction_power,
    float* transmission_power,
    float* scattering_power) {
    int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= count) {
        return;
    }
    float los_value = los[index];
    float reflection_value = reflection[index];
    float diffraction_value = diffraction[index];
    float transmission_value = transmission[index];
    float scattering_value = scattering[index];
    path_gain[index] = los_value + reflection_value + diffraction_value +
        transmission_value + scattering_value;
    atomicAdd(los_power, los_value);
    atomicAdd(reflection_power, reflection_value);
    atomicAdd(diffraction_power, diffraction_value);
    atomicAdd(transmission_power, transmission_value);
    atomicAdd(scattering_power, scattering_value);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor> channel_bdpt_finalize_point_components_cuda(
    at::Tensor los,
    at::Tensor reflection,
    at::Tensor diffraction,
    at::Tensor transmission,
    at::Tensor scattering) {
    check_path_gain(los);
    check_path_gain(reflection);
    check_path_gain(diffraction);
    check_path_gain(transmission);
    check_path_gain(scattering);
    TORCH_CHECK(
        reflection.sizes() == los.sizes() && diffraction.sizes() == los.sizes() &&
            transmission.sizes() == los.sizes() && scattering.sizes() == los.sizes(),
        "point component matrices must share shape");
    TORCH_CHECK(reflection.get_device() == los.get_device(), "reflection must share los device");
    TORCH_CHECK(diffraction.get_device() == los.get_device(), "diffraction must share los device");
    TORCH_CHECK(transmission.get_device() == los.get_device(), "transmission must share los device");
    TORCH_CHECK(scattering.get_device() == los.get_device(), "scattering must share los device");
    auto float_options = los.options().dtype(at::kFloat);
    auto path_gain = at::empty_like(los);
    auto los_power = at::empty({}, float_options);
    auto reflection_power = at::empty({}, float_options);
    auto diffraction_power = at::empty({}, float_options);
    auto transmission_power = at::empty({}, float_options);
    auto scattering_power = at::empty({}, float_options);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(los.get_device()).stream();
    C10_CUDA_CHECK(cudaMemsetAsync(los_power.data_ptr<float>(), 0, sizeof(float), stream));
    C10_CUDA_CHECK(cudaMemsetAsync(reflection_power.data_ptr<float>(), 0, sizeof(float), stream));
    C10_CUDA_CHECK(cudaMemsetAsync(diffraction_power.data_ptr<float>(), 0, sizeof(float), stream));
    C10_CUDA_CHECK(cudaMemsetAsync(transmission_power.data_ptr<float>(), 0, sizeof(float), stream));
    C10_CUDA_CHECK(cudaMemsetAsync(scattering_power.data_ptr<float>(), 0, sizeof(float), stream));
    int64_t count = los.numel();
    if (count > 0) {
        constexpr int threads = 256;
        int blocks = static_cast<int>((count + threads - 1) / threads);
        bdpt_finalize_point_components_kernel<<<blocks, threads, 0, stream>>>(
            count,
            los.data_ptr<float>(),
            reflection.data_ptr<float>(),
            diffraction.data_ptr<float>(),
            transmission.data_ptr<float>(),
            scattering.data_ptr<float>(),
            path_gain.data_ptr<float>(),
            los_power.data_ptr<float>(),
            reflection_power.data_ptr<float>(),
            diffraction_power.data_ptr<float>(),
            transmission_power.data_ptr<float>(),
            scattering_power.data_ptr<float>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return {path_gain, los_power, reflection_power, diffraction_power, transmission_power, scattering_power};
}

// ==== Section: Monte Carlo sampling ====
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime_api.h>
#include "math.cuh"

namespace {

constexpr int kSampleBlockSize = 256;
constexpr double kPi = 3.141592653589793238462643383279502884;
constexpr double kGoldenRatio = 1.618033988749894848204586834365638118;

__global__ void sample_directions_kernel(float *__restrict__ directions, int64_t count) {
    const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         idx < count;
         idx += stride) {
        const double index = static_cast<double>(idx);
        const double azimuth = index / kGoldenRatio;
        const double azimuth_u = azimuth - floor(azimuth);
        const double elevation_v = count == 1 ? 0.0 : index / static_cast<double>(count - 1);
        const double phi = 2.0 * kPi * azimuth_u;
        const double z = 1.0 - 2.0 * elevation_v;
        const double radial = sqrt(fmax(1.0 - z * z, 0.0));

        float *out = directions + idx * 3;
        out[0] = static_cast<float>(radial * cos(phi));
        out[1] = static_cast<float>(radial * sin(phi));
        out[2] = static_cast<float>(z);
    }
}

}  // namespace

at::Tensor channel_mc_sample_directions_cuda(int64_t count, at::Tensor reference) {
    TORCH_CHECK(count >= 0, "count must be non-negative");
    TORCH_CHECK(reference.is_cuda(), "reference must be a CUDA tensor");
    TORCH_CHECK(reference.scalar_type() == at::kFloat, "reference must be float32");

    auto directions = at::empty({count, 3}, reference.options());
    if (count > 0) {
        cudaStream_t stream = at::cuda::getCurrentCUDAStream(reference.get_device()).stream();
        const int block_count = static_cast<int>((count + kSampleBlockSize - 1) / kSampleBlockSize);
        sample_directions_kernel<<<block_count, kSampleBlockSize, 0, stream>>>(
            directions.data_ptr<float>(),
            count);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return directions;
}

// ==== Section: Monte Carlo material sampling ====
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime_api.h>
#include "math.cuh"

#include "../tensor_checks.h"

#include <tuple>
#include <vector>

namespace {

constexpr int kMaterialBlockSize = 256;

using channel::check_tensor;

__global__ void face_material_tensors_kernel(
    const float *__restrict__ material_eps_r,
    const float *__restrict__ material_sigma_e,
    const float *__restrict__ material_mu_r,
    const int *__restrict__ face_material_id,
    float *__restrict__ face_eps_r,
    float *__restrict__ face_sigma_e,
    float *__restrict__ face_mu_r,
    float *__restrict__ face_gain,
    bool *__restrict__ face_valid,
    int64_t face_count) {
    const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (int64_t face = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         face < face_count;
         face += stride) {
        const int material_id = face_material_id[face];
        face_eps_r[face] = material_eps_r[material_id];
        face_sigma_e[face] = material_sigma_e[material_id];
        face_mu_r[face] = material_mu_r[material_id];
        face_gain[face] = 1.0f;
        face_valid[face] = true;
    }
}

using FaceMaterialTensors =
    std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>;

FaceMaterialTensors face_material_tensors_cuda_impl(
    at::Tensor material_eps_r,
    at::Tensor material_sigma_e,
    at::Tensor material_mu_r,
    at::Tensor face_material_id) {
    check_tensor(material_eps_r, "material_eps_r", at::kFloat, 1);
    check_tensor(material_sigma_e, "material_sigma_e", at::kFloat, 1);
    check_tensor(material_mu_r, "material_mu_r", at::kFloat, 1);
    check_tensor(face_material_id, "face_material_id", at::kInt, 1);
    TORCH_CHECK(material_sigma_e.size(0) == material_eps_r.size(0), "material_sigma_e must match material_eps_r");
    TORCH_CHECK(material_mu_r.size(0) == material_eps_r.size(0), "material_mu_r must match material_eps_r");

    const int64_t face_count = face_material_id.size(0);
    auto face_eps_r = at::empty({face_count}, material_eps_r.options());
    auto face_sigma_e = at::empty({face_count}, material_eps_r.options());
    auto face_mu_r = at::empty({face_count}, material_eps_r.options());
    auto face_gain = at::empty({face_count}, material_eps_r.options());
    auto face_valid = at::empty({face_count}, face_material_id.options().dtype(at::kBool));

    if (face_count > 0) {
        cudaStream_t stream = at::cuda::getCurrentCUDAStream(material_eps_r.get_device()).stream();
        const int block_count = static_cast<int>((face_count + kMaterialBlockSize - 1) / kMaterialBlockSize);
        face_material_tensors_kernel<<<block_count, kMaterialBlockSize, 0, stream>>>(
            material_eps_r.data_ptr<float>(),
            material_sigma_e.data_ptr<float>(),
            material_mu_r.data_ptr<float>(),
            face_material_id.data_ptr<int>(),
            face_eps_r.data_ptr<float>(),
            face_sigma_e.data_ptr<float>(),
            face_mu_r.data_ptr<float>(),
            face_gain.data_ptr<float>(),
            face_valid.data_ptr<bool>(),
            face_count);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

    return {face_eps_r, face_sigma_e, face_mu_r, face_gain, face_valid};
}

}  // namespace

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor> channel_mc_face_material_tensors_cuda(
    at::Tensor material_eps_r,
    at::Tensor material_sigma_e,
    at::Tensor material_mu_r,
    at::Tensor face_material_id) {
    return face_material_tensors_cuda_impl(
        material_eps_r,
        material_sigma_e,
        material_mu_r,
        face_material_id);
}

// ==== Section: Monte Carlo scattering receiver pairs ====
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <rayd/scattering_table.cuh>
#include "math.cuh"

#include "../tensor_checks.h"

#include <algorithm>
#include <tuple>

namespace {

constexpr int kPairBlockSize = 256;
namespace table = rayd::shared::scattering;

// Per-sample inputs of the area-sampled Kirchhoff estimator: rough-surface point, mean-plane
// normal flipped towards the transmitter, local frame, incident direction in that frame,
// incident weight cos_i / r1^2 (0 when the sample is not lit), spawn epsilon and face material.
struct ScatterSamples {
    const float* points;
    const float* normals;
    const float* frame_t1;
    const float* frame_t2;
    const float* wi_local;
    const float* incident;
    const float* epsilon;
    const int* material;
    int64_t count;
};

struct ScatterTables {
    const int* material_slot;
    int64_t material_count;
    const int64_t* table_offset;
    const int* table_dims;
    const float* f_te;
    const float* f_tm;
};

struct PairGeometry {
    bool active;
    float wo_world[3];
    float wo_local[3];
    float weight;
    int slot;
};

using channel::math::load_vec3;
using channel::math::dot;

// Outgoing geometry of sample n towards receiver rx: active when the sample is lit, carries a
// table, and the receiver lies on the lit side; weight = cos_o * incident / r2^2.
__device__ __forceinline__ PairGeometry pair_geometry(
    const ScatterSamples& samples, const ScatterTables& tables, int64_t n, float3 rx) {
    PairGeometry pair{};
    pair.active = false;
    const int material = samples.material[n];
    pair.slot = material >= 0 && material < tables.material_count ? tables.material_slot[material] : -1;
    const float incident = samples.incident[n];
    if (pair.slot < 0 || !(incident > 0.0f)) return pair;
    const float3 p = load_vec3(samples.points, n);
    const float3 delta = make_float3(rx.x - p.x, rx.y - p.y, rx.z - p.z);
    const float r2 = fmaxf(sqrtf(dot(delta, delta)), 1.0e-6f);
    const float3 wo = make_float3(delta.x / r2, delta.y / r2, delta.z / r2);
    const float3 normal = load_vec3(samples.normals, n);
    const float cos_o = dot(wo, normal);
    if (!(cos_o > 0.0f)) return pair;
    pair.active = true;
    pair.wo_world[0] = wo.x;
    pair.wo_world[1] = wo.y;
    pair.wo_world[2] = wo.z;
    pair.wo_local[0] = dot(wo, load_vec3(samples.frame_t1, n));
    pair.wo_local[1] = dot(wo, load_vec3(samples.frame_t2, n));
    pair.wo_local[2] = cos_o;
    pair.weight = incident * cos_o / (r2 * r2);
    return pair;
}

__device__ __forceinline__ void slot_table(
    const ScatterTables& tables, int slot, const float* f_te, const float* f_tm,
    const float*& te, const float*& tm, int dims[4]) {
    const int64_t offset = tables.table_offset[slot];
    te = f_te + offset;
    tm = f_tm + offset;
    for (int k = 0; k < 4; ++k) dims[k] = tables.table_dims[4 * slot + k];
}

// Visibility legs of every (receiver j of the chunk, sample n) pair, entry j * N + n: from the
// sample point moved epsilon along the outgoing ray to the receiver.
__global__ void scattering_pair_legs_kernel(
    ScatterSamples samples, ScatterTables tables, const float* receivers, int64_t receiver_count,
    float* start, float* end, bool* active) {
    const int64_t total = receiver_count * samples.count;
    const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (int64_t entry = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; entry < total;
         entry += stride) {
        const int64_t j = entry / samples.count;
        const int64_t n = entry - j * samples.count;
        const float3 rx = load_vec3(receivers, j);
        const PairGeometry pair = pair_geometry(samples, tables, n, rx);
        const float3 p = load_vec3(samples.points, n);
        const float eps = samples.epsilon[n];
        start[3 * entry] = p.x + pair.wo_world[0] * eps;
        start[3 * entry + 1] = p.y + pair.wo_world[1] * eps;
        start[3 * entry + 2] = p.z + pair.wo_world[2] * eps;
        end[3 * entry] = rx.x;
        end[3 * entry + 1] = rx.y;
        end[3 * entry + 2] = rx.z;
        active[entry] = pair.active;
    }
}

// One block per receiver of the chunk: the lanes add their samples' deposits
// 0.5 * (f_te + f_tm) * weight in ascending sample order, then a fixed tree combines the lanes,
// so the sum is reproducible. f_te / f_tm are the (possibly tangent) table values.
__global__ void scattering_pair_sum_kernel(
    ScatterSamples samples, ScatterTables tables, const float* receivers, const bool* visible,
    const float* f_te, const float* f_tm, float* out, unsigned long long* deposited) {
    __shared__ float partial[kPairBlockSize];
    __shared__ unsigned int counts[kPairBlockSize];
    const int64_t j = blockIdx.x;
    const float3 rx = load_vec3(receivers, j);
    float sum = 0.0f;
    unsigned int count = 0;
    for (int64_t n = threadIdx.x; n < samples.count; n += blockDim.x) {
        if (!visible[j * samples.count + n]) continue;
        const PairGeometry pair = pair_geometry(samples, tables, n, rx);
        if (!pair.active) continue;
        const float* te;
        const float* tm;
        int dims[4];
        slot_table(tables, pair.slot, f_te, f_tm, te, tm, dims);
        float out_te, out_tm;
        table::eval_te_tm(te, tm, dims[0], dims[1], dims[2], dims[3], samples.wi_local + 3 * n,
                          pair.wo_local, out_te, out_tm);
        const float deposit = 0.5f * (out_te + out_tm) * pair.weight;
        sum += deposit;
        count += deposit > 0.0f ? 1u : 0u;
    }
    partial[threadIdx.x] = sum;
    counts[threadIdx.x] = count;
    __syncthreads();
    for (int width = blockDim.x / 2; width > 0; width /= 2) {
        if (static_cast<int>(threadIdx.x) < width) {
            partial[threadIdx.x] += partial[threadIdx.x + width];
            counts[threadIdx.x] += counts[threadIdx.x + width];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        out[j] = partial[0];
        if (deposited != nullptr) atomicAdd(deposited, static_cast<unsigned long long>(counts[0]));
    }
}

// VJP of the pair sum with respect to the flat table values: every visible pair spreads
// grad[j] * 0.5 * weight over its 16 interpolation corners in both channels.
__global__ void scattering_pair_sum_backward_kernel(
    ScatterSamples samples, ScatterTables tables, const float* receivers, int64_t receiver_count,
    const bool* visible, const float* grad, float* grad_te, float* grad_tm) {
    const int64_t total = receiver_count * samples.count;
    const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (int64_t entry = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; entry < total;
         entry += stride) {
        if (!visible[entry]) continue;
        const int64_t j = entry / samples.count;
        const int64_t n = entry - j * samples.count;
        const float g = grad[j];
        if (g == 0.0f) continue;
        const PairGeometry pair = pair_geometry(samples, tables, n, load_vec3(receivers, j));
        if (!pair.active) continue;
        const float* te;
        const float* tm;
        int dims[4];
        slot_table(tables, pair.slot, tables.f_te, tables.f_tm, te, tm, dims);
        table::TableEvalGrad eval;
        table::eval_te_tm_grad(te, tm, dims[0], dims[1], dims[2], dims[3], samples.wi_local + 3 * n,
                               pair.wo_local, eval);
        if (!eval.active) continue;
        const int64_t offset = tables.table_offset[pair.slot];
        const float scale = g * 0.5f * pair.weight;
        for (int k = 0; k < 16; ++k) {
            if (eval.cw[k] == 0.0f) continue;
            atomicAdd(grad_te + offset + eval.idx[k], scale * eval.cw[k]);
            atomicAdd(grad_tm + offset + eval.idx[k], scale * eval.cw[k]);
        }
    }
}

ScatterSamples scatter_samples(
    const at::Tensor& points, const at::Tensor& normals, const at::Tensor& frame_t1,
    const at::Tensor& frame_t2, const at::Tensor& wi_local, const at::Tensor& incident,
    const at::Tensor& epsilon, const at::Tensor& material) {
    return {points.data_ptr<float>(), normals.data_ptr<float>(), frame_t1.data_ptr<float>(),
            frame_t2.data_ptr<float>(), wi_local.data_ptr<float>(), incident.data_ptr<float>(),
            epsilon.data_ptr<float>(), material.data_ptr<int>(), points.size(0)};
}

ScatterTables scatter_tables(
    const at::Tensor& material_slot, const at::Tensor& table_offset, const at::Tensor& table_dims,
    const at::Tensor& f_te, const at::Tensor& f_tm) {
    return {material_slot.data_ptr<int>(), material_slot.numel(), table_offset.data_ptr<int64_t>(),
            table_dims.data_ptr<int>(), f_te.data_ptr<float>(), f_tm.data_ptr<float>()};
}

int pair_blocks(int64_t count) {
    return static_cast<int>(std::min<int64_t>((count + kPairBlockSize - 1) / kPairBlockSize, 65535));
}

}  // namespace

std::tuple<at::Tensor, at::Tensor, at::Tensor> channel_mc_scattering_pair_legs_cuda(
    at::Tensor points, at::Tensor normals, at::Tensor frame_t1, at::Tensor frame_t2,
    at::Tensor wi_local, at::Tensor incident, at::Tensor epsilon, at::Tensor material,
    at::Tensor material_slot, at::Tensor table_offset, at::Tensor table_dims, at::Tensor f_te,
    at::Tensor f_tm, at::Tensor receivers) {
    const int64_t total = receivers.size(0) * points.size(0);
    auto start = at::empty({total, 3}, points.options());
    auto end = at::empty({total, 3}, points.options());
    auto active = at::empty({total}, points.options().dtype(at::kBool));
    if (total > 0) {
        auto stream = at::cuda::getCurrentCUDAStream(points.get_device()).stream();
        scattering_pair_legs_kernel<<<pair_blocks(total), kPairBlockSize, 0, stream>>>(
            scatter_samples(points, normals, frame_t1, frame_t2, wi_local, incident, epsilon, material),
            scatter_tables(material_slot, table_offset, table_dims, f_te, f_tm),
            receivers.data_ptr<float>(), receivers.size(0), start.data_ptr<float>(),
            end.data_ptr<float>(), active.data_ptr<bool>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return {start, end, active};
}

at::Tensor channel_mc_scattering_pair_sum_cuda(
    at::Tensor points, at::Tensor normals, at::Tensor frame_t1, at::Tensor frame_t2,
    at::Tensor wi_local, at::Tensor incident, at::Tensor epsilon, at::Tensor material,
    at::Tensor material_slot, at::Tensor table_offset, at::Tensor table_dims, at::Tensor f_te,
    at::Tensor f_tm, at::Tensor receivers, at::Tensor visible, at::Tensor value_te,
    at::Tensor value_tm, at::Tensor deposited) {
    auto out = at::empty({receivers.size(0)}, points.options());
    if (receivers.size(0) == 0) return out;
    auto stream = at::cuda::getCurrentCUDAStream(points.get_device()).stream();
    scattering_pair_sum_kernel<<<static_cast<unsigned int>(receivers.size(0)), kPairBlockSize, 0, stream>>>(
        scatter_samples(points, normals, frame_t1, frame_t2, wi_local, incident, epsilon, material),
        scatter_tables(material_slot, table_offset, table_dims, f_te, f_tm),
        receivers.data_ptr<float>(), visible.data_ptr<bool>(), value_te.data_ptr<float>(),
        value_tm.data_ptr<float>(), out.data_ptr<float>(),
        deposited.numel() > 0 ? reinterpret_cast<unsigned long long*>(deposited.data_ptr<int64_t>())
                              : nullptr);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

std::tuple<at::Tensor, at::Tensor> channel_mc_scattering_pair_sum_backward_cuda(
    at::Tensor points, at::Tensor normals, at::Tensor frame_t1, at::Tensor frame_t2,
    at::Tensor wi_local, at::Tensor incident, at::Tensor epsilon, at::Tensor material,
    at::Tensor material_slot, at::Tensor table_offset, at::Tensor table_dims, at::Tensor f_te,
    at::Tensor f_tm, at::Tensor receivers, at::Tensor visible, at::Tensor grad) {
    auto grad_te = at::empty_like(f_te);
    auto grad_tm = at::empty_like(f_tm);
    auto stream = at::cuda::getCurrentCUDAStream(points.get_device()).stream();
    for (const auto* tensor : {&grad_te, &grad_tm})
        if (tensor->numel() > 0)
            C10_CUDA_CHECK(cudaMemsetAsync(tensor->data_ptr(), 0,
                                           static_cast<size_t>(tensor->numel()) * sizeof(float), stream));
    const int64_t total = receivers.size(0) * points.size(0);
    if (total > 0) {
        scattering_pair_sum_backward_kernel<<<pair_blocks(total), kPairBlockSize, 0, stream>>>(
            scatter_samples(points, normals, frame_t1, frame_t2, wi_local, incident, epsilon, material),
            scatter_tables(material_slot, table_offset, table_dims, f_te, f_tm),
            receivers.data_ptr<float>(), receivers.size(0), visible.data_ptr<bool>(),
            grad.data_ptr<float>(), grad_te.data_ptr<float>(), grad_tm.data_ptr<float>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return {grad_te, grad_tm};
}
