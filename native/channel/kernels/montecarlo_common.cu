// Copyright Xingyu Chen.
// Implements montecarlo common CUDA operations.

// ---- Consolidated from accum.cu ----
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

at::Tensor channel_bdpt_component_map_buffer_cuda(
    at::Tensor reference,
    int64_t tx_count,
    int64_t dim0,
    int64_t dim1) {
    return channel_mc_component_map_buffer_cuda(
        reference, tx_count, dim0, dim1);
}

at::Tensor channel_bdpt_store_component_map_cuda(
    at::Tensor maps,
    at::Tensor source,
    int64_t tx_index) {
    return channel_mc_store_component_map_cuda(maps, source, tx_index);
}

at::Tensor channel_bdpt_store_scaled_component_map_cuda(
    at::Tensor maps,
    at::Tensor source,
    at::Tensor scale_values,
    int64_t tx_index,
    int64_t scale_index) {
    return channel_mc_store_scaled_component_map_cuda(
        maps, source, scale_values, tx_index, scale_index);
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

// ---- Consolidated from bdpt_accum.cu ----
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime_api.h>
#include "math.cuh"
#include "torch_cuda_minimal.h"

#include <algorithm>

#define check_component_map check_bdpt_point_component_map

namespace {

void check_path_gain(const at::Tensor& tensor) {
    TORCH_CHECK(tensor.is_cuda(), "path_gain must be a CUDA tensor");
    TORCH_CHECK(tensor.scalar_type() == at::kFloat, "path_gain must be float32");
    TORCH_CHECK(tensor.dim() == 2, "path_gain must have rank 2");
    TORCH_CHECK(tensor.is_contiguous(), "path_gain must be contiguous");
}

void check_component_map(const at::Tensor& tensor, const char* name) {
    TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(tensor.scalar_type() == at::kFloat, name, " must be float32");
    TORCH_CHECK(tensor.dim() == 3, name, " must have rank 3");
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
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

__device__ int opposite_vertex(const int* face, int shared0, int shared1) {
    if (face[0] != shared0 && face[0] != shared1) {
        return face[0];
    }
    if (face[1] != shared0 && face[1] != shared1) {
        return face[1];
    }
    return face[2];
}



__global__ void diffraction_edge_count_kernel(
    int64_t count,
    const float* vertices,
    const int* faces,
    const float* face_normals,
    const int* edge_v0,
    const int* edge_v1,
    const int* face0,
    const int* face1,
    int vertical_only,
    float vertical_ratio_threshold,
    int boundary_half_plane,
    float plane_tol,
    int* selected_count) {
    constexpr float kEpsilon = 1.0e-6f;
    constexpr float kNormalCosTol = 1.0f - 1.0e-5f;
    constexpr float kPi = 3.14159265358979323846f;
    int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= count) {
        return;
    }

    const int v0 = edge_v0[index];
    const int v1 = edge_v1[index];
    const int f0 = face0[index];
    const int f1 = face1[index];
    const bool valid0 = f0 >= 0;
    const bool valid1 = f1 >= 0;
    const bool boundary = valid0 && !valid1;
    const bool interior = valid0 && valid1;
    if ((!interior && !boundary) || (boundary && !boundary_half_plane)) {
        return;
    }

    const float vx = vertices[static_cast<int64_t>(v1) * 3 + 0] - vertices[static_cast<int64_t>(v0) * 3 + 0];
    const float vy = vertices[static_cast<int64_t>(v1) * 3 + 1] - vertices[static_cast<int64_t>(v0) * 3 + 1];
    const float vz = vertices[static_cast<int64_t>(v1) * 3 + 2] - vertices[static_cast<int64_t>(v0) * 3 + 2];
    const float length = sqrtf(vx * vx + vy * vy + vz * vz);
    if (length <= kEpsilon) {
        return;
    }

    const int safe0 = max(f0, 0);
    const int safe1 = max(f1, 0);
    const float3 n0 = channel::math::load_normalized_min_length(face_normals, safe0, kEpsilon);
    const float3 n1 = channel::math::load_normalized_min_length(face_normals, safe1, kEpsilon);
    const float normal_dot = n0.x * n1.x + n0.y * n1.y + n0.z * n1.z;
    bool coplanar = false;
    if (interior && fabsf(normal_dot) >= kNormalCosTol) {
        const int* face_a = faces + static_cast<int64_t>(safe0) * 3;
        const int* face_b = faces + static_cast<int64_t>(safe1) * 3;
        const int opposite_a = opposite_vertex(face_a, v0, v1);
        const int opposite_b = opposite_vertex(face_b, v0, v1);
        const float px = vertices[static_cast<int64_t>(v0) * 3 + 0];
        const float py = vertices[static_cast<int64_t>(v0) * 3 + 1];
        const float pz = vertices[static_cast<int64_t>(v0) * 3 + 2];
        const float ax = vertices[static_cast<int64_t>(opposite_a) * 3 + 0] - px;
        const float ay = vertices[static_cast<int64_t>(opposite_a) * 3 + 1] - py;
        const float az = vertices[static_cast<int64_t>(opposite_a) * 3 + 2] - pz;
        const float bx = vertices[static_cast<int64_t>(opposite_b) * 3 + 0] - px;
        const float by = vertices[static_cast<int64_t>(opposite_b) * 3 + 1] - py;
        const float bz = vertices[static_cast<int64_t>(opposite_b) * 3 + 2] - pz;
        const float dist_a = fabsf(ax * n0.x + ay * n0.y + az * n0.z);
        const float dist_b = fabsf(bx * n0.x + by * n0.y + bz * n0.z);
        coplanar = dist_a <= plane_tol && dist_b <= plane_tol;
    }

    const float clamped = fminf(fmaxf(-normal_dot, -1.0f), 1.0f);
    const float interior_angle = acosf(clamped);
    const float exterior_angle = interior ? (2.0f * kPi - interior_angle) : (boundary_half_plane ? 2.0f * kPi : 0.0f);
    const float wedge_n = exterior_angle / kPi;
    if (coplanar || wedge_n <= 1.0f + kEpsilon) {
        return;
    }
    if (vertical_only && (fabsf(vz) / length) <= vertical_ratio_threshold) {
        return;
    }
    atomicAdd(selected_count, 1);
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

int64_t channel_core_diffraction_edge_count_cuda(
    at::Tensor vertices,
    at::Tensor faces,
    at::Tensor face_normals,
    at::Tensor edge_v0,
    at::Tensor edge_v1,
    at::Tensor face0,
    at::Tensor face1,
    bool vertical_only,
    double vertical_ratio,
    bool boundary_half_plane,
    double plane_tol) {
    TORCH_CHECK(vertices.is_cuda() && faces.is_cuda() && face_normals.is_cuda(), "geometry tensors must be CUDA tensors");
    TORCH_CHECK(edge_v0.is_cuda() && edge_v1.is_cuda() && face0.is_cuda() && face1.is_cuda(), "edge tensors must be CUDA tensors");
    TORCH_CHECK(vertices.scalar_type() == at::kFloat, "vertices must be float32");
    TORCH_CHECK(face_normals.scalar_type() == at::kFloat, "face_normals must be float32");
    TORCH_CHECK(faces.scalar_type() == at::kInt, "faces must be int32");
    TORCH_CHECK(edge_v0.scalar_type() == at::kInt && edge_v1.scalar_type() == at::kInt, "edge endpoints must be int32");
    TORCH_CHECK(face0.scalar_type() == at::kInt && face1.scalar_type() == at::kInt, "edge faces must be int32");
    TORCH_CHECK(vertices.dim() == 2 && vertices.size(1) == 3, "vertices must have shape (N, 3)");
    TORCH_CHECK(faces.dim() == 2 && faces.size(1) == 3, "faces must have shape (F, 3)");
    TORCH_CHECK(face_normals.dim() == 2 && face_normals.size(1) == 3, "face_normals must have shape (F, 3)");
    TORCH_CHECK(edge_v0.dim() == 1 && edge_v1.sizes() == edge_v0.sizes(), "edge endpoint tensors must share shape");
    TORCH_CHECK(face0.dim() == 1 && face0.sizes() == edge_v0.sizes() && face1.sizes() == edge_v0.sizes(), "edge face tensors must share shape");
    TORCH_CHECK(vertices.is_contiguous() && faces.is_contiguous() && face_normals.is_contiguous(), "geometry tensors must be contiguous");
    TORCH_CHECK(edge_v0.is_contiguous() && edge_v1.is_contiguous() && face0.is_contiguous() && face1.is_contiguous(), "edge tensors must be contiguous");

    const int64_t count = edge_v0.numel();
    auto selected = at::empty({1}, edge_v0.options().dtype(at::kInt));
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(vertices.get_device()).stream();
    C10_CUDA_CHECK(cudaMemsetAsync(selected.data_ptr<int>(), 0, sizeof(int), stream));
    if (count > 0) {
        constexpr int kBlockSize = 256;
        int blocks = static_cast<int>((count + kBlockSize - 1) / kBlockSize);
        diffraction_edge_count_kernel<<<blocks, kBlockSize, 0, stream>>>(
            count,
            vertices.data_ptr<float>(),
            faces.data_ptr<int>(),
            face_normals.data_ptr<float>(),
            edge_v0.data_ptr<int>(),
            edge_v1.data_ptr<int>(),
            face0.data_ptr<int>(),
            face1.data_ptr<int>(),
            vertical_only ? 1 : 0,
            static_cast<float>(vertical_ratio),
            boundary_half_plane ? 1 : 0,
            static_cast<float>(plane_tol),
            selected.data_ptr<int>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    int selected_count = 0;
    C10_CUDA_CHECK(cudaMemcpyAsync(
        &selected_count,
        selected.data_ptr<int>(),
        sizeof(int),
        cudaMemcpyDeviceToHost,
        stream));
    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
    return static_cast<int64_t>(selected_count);
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

__global__ void bdpt_sum_kernel(int64_t count, const float* values, bool include_los, float* los) {
    int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (!include_los || index >= count) {
        return;
    }
    atomicAdd(los, values[index]);
}

__global__ void bdpt_store_point_component_column_kernel(
    int64_t tx_count,
    int64_t rx_count,
    int64_t rx_index,
    const float* source,
    float* target) {
    int64_t tx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (tx >= tx_count) {
        return;
    }
    target[tx * rx_count + rx_index] = source[tx];
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

std::tuple<at::Tensor, at::Tensor, at::Tensor> channel_bdpt_point_component_power_cuda(
    at::Tensor path_gain,
    bool include_los) {
    check_path_gain(path_gain);
    auto scalar_options = path_gain.options().dtype(at::kFloat);
    auto los = at::empty({}, scalar_options);
    auto reflection = at::empty({}, scalar_options);
    auto diffraction = at::empty({}, scalar_options);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(path_gain.get_device()).stream();
    C10_CUDA_CHECK(cudaMemsetAsync(los.data_ptr<float>(), 0, sizeof(float), stream));
    C10_CUDA_CHECK(cudaMemsetAsync(reflection.data_ptr<float>(), 0, sizeof(float), stream));
    C10_CUDA_CHECK(cudaMemsetAsync(diffraction.data_ptr<float>(), 0, sizeof(float), stream));
    int64_t count = path_gain.numel();
    if (count > 0 && include_los) {
        constexpr int threads = 256;
        int blocks = static_cast<int>((count + threads - 1) / threads);
        bdpt_sum_kernel<<<blocks, threads, 0, stream>>>(
            count,
            path_gain.data_ptr<float>(),
            include_los,
            los.data_ptr<float>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return {los, reflection, diffraction};
}

at::Tensor channel_bdpt_store_point_component_column_cuda(
    at::Tensor target,
    at::Tensor source,
    int64_t rx_index) {
    check_path_gain(target);
    check_component_map(source, "source");
    TORCH_CHECK(source.size(0) == target.size(0), "source transmitter count must match target");
    TORCH_CHECK(source.size(1) == 1 && source.size(2) == 1, "source must have shape (tx, 1, 1)");
    TORCH_CHECK(source.get_device() == target.get_device(), "source must share target device");
    TORCH_CHECK(rx_index >= 0 && rx_index < target.size(1), "rx_index is out of range");
    int64_t tx_count = target.size(0);
    if (tx_count > 0) {
        constexpr int threads = 256;
        int blocks = static_cast<int>((tx_count + threads - 1) / threads);
        cudaStream_t stream = at::cuda::getCurrentCUDAStream(target.get_device()).stream();
        bdpt_store_point_component_column_kernel<<<blocks, threads, 0, stream>>>(
            tx_count,
            target.size(1),
            rx_index,
            source.data_ptr<float>(),
            target.data_ptr<float>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return target;
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

#undef check_component_map

// ---- Consolidated from sampling.cu ----
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

// ---- Consolidated from material.cu ----
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

at::Tensor copy_float_vector_to_cuda(const std::vector<float> &values) {
    auto out = at::empty(
        {static_cast<int64_t>(values.size())},
        at::TensorOptions().device(at::kCUDA).dtype(at::kFloat));
    if (!values.empty()) {
        cudaStream_t stream = at::cuda::getCurrentCUDAStream(out.get_device()).stream();
        C10_CUDA_CHECK(cudaMemcpyAsync(
            out.data_ptr<float>(),
            values.data(),
            values.size() * sizeof(float),
            cudaMemcpyHostToDevice,
            stream));
    }
    return out;
}

at::Tensor copy_int_vector_to_cuda(const std::vector<int> &values) {
    auto out = at::empty(
        {static_cast<int64_t>(values.size())},
        at::TensorOptions().device(at::kCUDA).dtype(at::kInt));
    if (!values.empty()) {
        cudaStream_t stream = at::cuda::getCurrentCUDAStream(out.get_device()).stream();
        C10_CUDA_CHECK(cudaMemcpyAsync(
            out.data_ptr<int>(),
            values.data(),
            values.size() * sizeof(int),
            cudaMemcpyHostToDevice,
            stream));
    }
    return out;
}

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

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor> channel_bdpt_face_material_tensors_cuda(
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

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
channel_bdpt_face_material_tensors_from_host_cuda(
    const std::vector<float> &material_eps_r,
    const std::vector<float> &material_sigma_e,
    const std::vector<float> &material_mu_r,
    const std::vector<int> &face_material_id) {
    TORCH_CHECK(!material_eps_r.empty(), "material_eps_r must not be empty");
    TORCH_CHECK(
        material_sigma_e.size() == material_eps_r.size(),
        "material_sigma_e must match material_eps_r");
    TORCH_CHECK(material_mu_r.size() == material_eps_r.size(), "material_mu_r must match material_eps_r");
    for (int material_id : face_material_id) {
        TORCH_CHECK(material_id >= 0, "face_material_id entries must be non-negative");
        TORCH_CHECK(
            static_cast<size_t>(material_id) < material_eps_r.size(),
            "face_material_id entry is out of range");
    }

    return channel_bdpt_face_material_tensors_cuda(
        copy_float_vector_to_cuda(material_eps_r),
        copy_float_vector_to_cuda(material_sigma_e),
        copy_float_vector_to_cuda(material_mu_r),
        copy_int_vector_to_cuda(face_material_id));
}

// ---- Consolidated from scattering.cu ----
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include "torch_cuda_minimal.h"

#include "../tensor_checks.h"

namespace {

constexpr int kBlockSize = 256;
constexpr float kTwoPi = 6.2831853071795864769f;
constexpr float kC0 = 299792458.0f;

__global__ void scattering_event_kernel(
    int64_t count, const float* cos_theta, const int* material_id,
    const float* cap_r_te, const float* cap_r_tm,
    const float* cap_t_te, const float* cap_t_tm,
    const float* rough_sigma, const int* scatter_model, int64_t material_count,
    float frequency, float probability_floor,
    float* p_scatter, float* p_transmit, float* coherent, bool* rough) {
    const float k0 = kTwoPi * frequency / kC0;
    for (int64_t row = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         row < count; row += static_cast<int64_t>(blockDim.x) * gridDim.x) {
        const int mat=material_id[row];
        if (mat<0 || mat>=material_count || scatter_model[mat]!=1) {
            p_scatter[row]=p_transmit[row]=0.0f; coherent[row]=1.0f; rough[row]=false; continue;
        }
        const float c=expf(-2.0f*(k0*fabsf(cos_theta[row])*rough_sigma[mat])*(k0*fabsf(cos_theta[row])*rough_sigma[mat]));
        const float rbar=0.5f*(cap_r_te[row]+cap_r_tm[row]);
        const float tbar=0.5f*(cap_t_te[row]+cap_t_tm[row]);
        const float rcoh=rbar*c*c, rdiff=fmaxf(0.0f,rbar-rcoh);
        const float total=fmaxf(rcoh+rdiff+tbar,1e-12f);
        float pr=rcoh/total, ps=rdiff/total, pt=tbar/total;
        if (rcoh>0.0f) pr=fmaxf(pr,probability_floor);
        if (rdiff>0.0f) ps=fmaxf(ps,probability_floor);
        if (tbar>0.0f) pt=fmaxf(pt,probability_floor);
        const float norm=fmaxf(pr+ps+pt,1e-12f);
        p_scatter[row]=ps/norm; p_transmit[row]=pt/norm; coherent[row]=c; rough[row]=true;
    }
}

int blocks(int64_t n) { return static_cast<int>((n+kBlockSize-1)/kBlockSize); }

} // namespace

pybind11::dict channel_scattering_event_probabilities(at::Tensor cos_theta,at::Tensor material_id,at::Tensor cap_r_te,at::Tensor cap_r_tm,at::Tensor cap_t_te,at::Tensor cap_t_tm,at::Tensor rough_sigma,at::Tensor scatter_model,double frequency,double probability_floor){
    channel::check_flat_tensor(cos_theta,"cos_theta",at::kFloat);channel::check_flat_tensor(material_id,"material_id",at::kInt);channel::check_flat_tensor(cap_r_te,"cap_R_te",at::kFloat);channel::check_flat_tensor(cap_r_tm,"cap_R_tm",at::kFloat);channel::check_flat_tensor(cap_t_te,"cap_T_te",at::kFloat);channel::check_flat_tensor(cap_t_tm,"cap_T_tm",at::kFloat);channel::check_flat_tensor(rough_sigma,"rough_sigma_h_m",at::kFloat);channel::check_flat_tensor(scatter_model,"scatter_model_id",at::kInt);
    const int64_t count=cos_theta.size(0); TORCH_CHECK(material_id.size(0)==count&&cap_r_te.size(0)==count&&cap_r_tm.size(0)==count&&cap_t_te.size(0)==count&&cap_t_tm.size(0)==count,"per-ray scattering arrays must match cos_theta"); TORCH_CHECK(scatter_model.size(0)==rough_sigma.size(0),"material arrays must match");
    for(const auto&t:{material_id,cap_r_te,cap_r_tm,cap_t_te,cap_t_tm,rough_sigma,scatter_model})TORCH_CHECK(t.get_device()==cos_theta.get_device(),"scattering tensors must share device");
    auto ps=at::empty_like(cos_theta),pt=at::empty_like(cos_theta),cr=at::empty_like(cos_theta);auto rb=at::empty({count},cos_theta.options().dtype(at::kBool));if(count>0){auto s=at::cuda::getCurrentCUDAStream(cos_theta.get_device()).stream();scattering_event_kernel<<<blocks(count),kBlockSize,0,s>>>(count,cos_theta.data_ptr<float>(),material_id.data_ptr<int>(),cap_r_te.data_ptr<float>(),cap_r_tm.data_ptr<float>(),cap_t_te.data_ptr<float>(),cap_t_tm.data_ptr<float>(),rough_sigma.data_ptr<float>(),scatter_model.data_ptr<int>(),rough_sigma.size(0),static_cast<float>(frequency),static_cast<float>(probability_floor),ps.data_ptr<float>(),pt.data_ptr<float>(),cr.data_ptr<float>(),rb.data_ptr<bool>());C10_CUDA_KERNEL_LAUNCH_CHECK();}pybind11::dict out;out["p_scatter"]=ps;out["p_transmit"]=pt;out["r_coh_amplitude"]=cr;out["rough"]=rb;return out;
}
