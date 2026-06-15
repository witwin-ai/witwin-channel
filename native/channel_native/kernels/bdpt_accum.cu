#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime_api.h>
#include <torch/extension.h>

#include <algorithm>

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

__device__ float3 normalized3(const float* values, int index, float eps) {
    float3 v{
        values[static_cast<int64_t>(index) * 3 + 0],
        values[static_cast<int64_t>(index) * 3 + 1],
        values[static_cast<int64_t>(index) * 3 + 2],
    };
    float length = sqrtf(v.x * v.x + v.y * v.y + v.z * v.z);
    length = fmaxf(length, eps);
    return {v.x / length, v.y / length, v.z / length};
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
    const float3 n0 = normalized3(face_normals, safe0, kEpsilon);
    const float3 n1 = normalized3(face_normals, safe1, kEpsilon);
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

at::Tensor cn_core_pack_int2_cuda(at::Tensor x, at::Tensor y) {
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

int64_t cn_core_diffraction_edge_count_cuda(
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

at::Tensor cn_bdpt_zero_matrix_cuda(at::Tensor reference, int64_t rows, int64_t cols) {
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
    float* path_gain,
    float* los_power,
    float* reflection_power,
    float* diffraction_power) {
    int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= count) {
        return;
    }
    float los_value = los[index];
    float reflection_value = reflection[index];
    float diffraction_value = diffraction[index];
    path_gain[index] = los_value + reflection_value + diffraction_value;
    atomicAdd(los_power, los_value);
    atomicAdd(reflection_power, reflection_value);
    atomicAdd(diffraction_power, diffraction_value);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> cn_bdpt_point_component_power_cuda(
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

at::Tensor cn_bdpt_store_point_component_column_cuda(
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

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> cn_bdpt_finalize_point_components_cuda(
    at::Tensor los,
    at::Tensor reflection,
    at::Tensor diffraction) {
    check_path_gain(los);
    check_path_gain(reflection);
    check_path_gain(diffraction);
    TORCH_CHECK(reflection.sizes() == los.sizes() && diffraction.sizes() == los.sizes(), "point component matrices must share shape");
    TORCH_CHECK(reflection.get_device() == los.get_device(), "reflection must share los device");
    TORCH_CHECK(diffraction.get_device() == los.get_device(), "diffraction must share los device");
    auto float_options = los.options().dtype(at::kFloat);
    auto path_gain = at::empty_like(los);
    auto los_power = at::empty({}, float_options);
    auto reflection_power = at::empty({}, float_options);
    auto diffraction_power = at::empty({}, float_options);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(los.get_device()).stream();
    C10_CUDA_CHECK(cudaMemsetAsync(los_power.data_ptr<float>(), 0, sizeof(float), stream));
    C10_CUDA_CHECK(cudaMemsetAsync(reflection_power.data_ptr<float>(), 0, sizeof(float), stream));
    C10_CUDA_CHECK(cudaMemsetAsync(diffraction_power.data_ptr<float>(), 0, sizeof(float), stream));
    int64_t count = los.numel();
    if (count > 0) {
        constexpr int threads = 256;
        int blocks = static_cast<int>((count + threads - 1) / threads);
        bdpt_finalize_point_components_kernel<<<blocks, threads, 0, stream>>>(
            count,
            los.data_ptr<float>(),
            reflection.data_ptr<float>(),
            diffraction.data_ptr<float>(),
            path_gain.data_ptr<float>(),
            los_power.data_ptr<float>(),
            reflection_power.data_ptr<float>(),
            diffraction_power.data_ptr<float>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return {path_gain, los_power, reflection_power, diffraction_power};
}
