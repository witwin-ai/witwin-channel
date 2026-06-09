#include <raydn/common/camera.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>

#include <cmath>
#include <stdexcept>
#include <string>

namespace raydn {

namespace {

void cuda_check(cudaError_t result, const char *expr) {
    if (result == cudaSuccess)
        return;
    throw std::runtime_error(
        std::string("CUDA error in ") + expr + ": " + cudaGetErrorString(result));
}

__global__ void camera_sample_to_world_kernel(
    int64_t count,
    const float *__restrict__ sample,
    int64_t sample_stride0,
    int64_t sample_stride1,
    float *__restrict__ world,
    float tan_x,
    float tan_y,
    float depth) {
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= count)
        return;
    const float u = sample[idx * sample_stride0 + 0 * sample_stride1];
    const float v = sample[idx * sample_stride0 + 1 * sample_stride1];
    world[idx * 3 + 0] = (u * 2.0f - 1.0f) * tan_x * depth;
    world[idx * 3 + 1] = (1.0f - v * 2.0f) * tan_y * depth;
    world[idx * 3 + 2] = depth;
}

__global__ void camera_sample_to_world_backward_kernel(
    int64_t count,
    const float *__restrict__ grad_world,
    int64_t grad_world_stride0,
    int64_t grad_world_stride1,
    float *__restrict__ grad_sample,
    float tan_x,
    float tan_y,
    float depth) {
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= count)
        return;
    grad_sample[idx * 2 + 0] =
        grad_world[idx * grad_world_stride0 + 0 * grad_world_stride1] * (2.0f * tan_x * depth);
    grad_sample[idx * 2 + 1] =
        grad_world[idx * grad_world_stride0 + 1 * grad_world_stride1] * (-2.0f * tan_y * depth);
}

__global__ void camera_world_to_sample_kernel(
    int64_t count,
    const float *__restrict__ point,
    int64_t point_stride0,
    int64_t point_stride1,
    float *__restrict__ sample,
    float tan_x,
    float tan_y) {
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= count)
        return;
    const float x = point[idx * point_stride0 + 0 * point_stride1];
    const float y = point[idx * point_stride0 + 1 * point_stride1];
    const float z = fmaxf(point[idx * point_stride0 + 2 * point_stride1], 1.0e-12f);
    sample[idx * 2 + 0] = x / (z * tan_x) * 0.5f + 0.5f;
    sample[idx * 2 + 1] = 0.5f - y / (z * tan_y) * 0.5f;
}

__global__ void camera_world_to_sample_backward_kernel(
    int64_t count,
    const float *__restrict__ point,
    int64_t point_stride0,
    int64_t point_stride1,
    const float *__restrict__ grad_sample,
    int64_t grad_sample_stride0,
    int64_t grad_sample_stride1,
    float *__restrict__ grad_point,
    float tan_x,
    float tan_y) {
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= count)
        return;
    const float x = point[idx * point_stride0 + 0 * point_stride1];
    const float y = point[idx * point_stride0 + 1 * point_stride1];
    const float z_raw = point[idx * point_stride0 + 2 * point_stride1];
    const float z = fmaxf(z_raw, 1.0e-12f);
    const float gu = grad_sample[idx * grad_sample_stride0 + 0 * grad_sample_stride1];
    const float gv = grad_sample[idx * grad_sample_stride0 + 1 * grad_sample_stride1];
    const float inv_z = 1.0f / z;
    const float inv_z2 = inv_z * inv_z;
    grad_point[idx * 3 + 0] = gu * (0.5f * inv_z / tan_x);
    grad_point[idx * 3 + 1] = gv * (-0.5f * inv_z / tan_y);
    grad_point[idx * 3 + 2] = z_raw > 1.0e-12f
        ? gu * (-0.5f * x * inv_z2 / tan_x) + gv * (0.5f * y * inv_z2 / tan_y)
        : 0.0f;
}

__global__ void camera_sample_ray_kernel(
    int64_t count,
    const float *__restrict__ sample,
    int64_t sample_stride0,
    int64_t sample_stride1,
    float *__restrict__ origin,
    float *__restrict__ direction,
    float tan_x,
    float tan_y) {
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= count)
        return;
    const float u = sample[idx * sample_stride0 + 0 * sample_stride1];
    const float v = sample[idx * sample_stride0 + 1 * sample_stride1];
    const float x = (u * 2.0f - 1.0f) * tan_x;
    const float y = (1.0f - v * 2.0f) * tan_y;
    const float z = 1.0f;
    const float inv_norm = rsqrtf(x * x + y * y + z * z);
    origin[idx * 3 + 0] = 0.0f;
    origin[idx * 3 + 1] = 0.0f;
    origin[idx * 3 + 2] = 0.0f;
    direction[idx * 3 + 0] = x * inv_norm;
    direction[idx * 3 + 1] = y * inv_norm;
    direction[idx * 3 + 2] = z * inv_norm;
}

__global__ void camera_sample_ray_backward_kernel(
    int64_t count,
    const float *__restrict__ sample,
    int64_t sample_stride0,
    int64_t sample_stride1,
    const float *__restrict__ grad_direction,
    int64_t grad_direction_stride0,
    int64_t grad_direction_stride1,
    float *__restrict__ grad_sample,
    float tan_x,
    float tan_y) {
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= count)
        return;
    if (grad_direction == nullptr) {
        grad_sample[idx * 2 + 0] = 0.0f;
        grad_sample[idx * 2 + 1] = 0.0f;
        return;
    }
    const float u = sample[idx * sample_stride0 + 0 * sample_stride1];
    const float v = sample[idx * sample_stride0 + 1 * sample_stride1];
    const float x = (u * 2.0f - 1.0f) * tan_x;
    const float y = (1.0f - v * 2.0f) * tan_y;
    const float z = 1.0f;
    const float inv_norm = rsqrtf(x * x + y * y + z * z);
    const float dx = x * inv_norm;
    const float dy = y * inv_norm;
    const float dz = z * inv_norm;
    const float gx = grad_direction[idx * grad_direction_stride0 + 0 * grad_direction_stride1];
    const float gy = grad_direction[idx * grad_direction_stride0 + 1 * grad_direction_stride1];
    const float gz = grad_direction[idx * grad_direction_stride0 + 2 * grad_direction_stride1];
    const float dot = gx * dx + gy * dy + gz * dz;
    const float grad_x = (gx - dx * dot) * inv_norm;
    const float grad_y = (gy - dy * dot) * inv_norm;
    grad_sample[idx * 2 + 0] = grad_x * (2.0f * tan_x);
    grad_sample[idx * 2 + 1] = grad_y * (-2.0f * tan_y);
}

template <typename Launch>
void launch_1d(int64_t count, Launch &&launch) {
    if (count == 0)
        return;
    constexpr int threads = 256;
    const int blocks = static_cast<int>((count + threads - 1) / threads);
    launch(blocks, threads);
}

} // namespace

at::Tensor camera_sample_to_world_cuda(
    const at::Tensor &sample,
    double tan_x,
    double tan_y,
    double depth) {
    c10::cuda::CUDAGuard guard(sample.device());
    at::Tensor world = at::empty({sample.size(0), 3}, sample.options());
    const int64_t count = sample.size(0);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(sample.get_device());
    launch_1d(count, [&](int blocks, int threads) {
        camera_sample_to_world_kernel<<<blocks, threads, 0, stream>>>(
            count,
            sample.data_ptr<float>(),
            sample.stride(0),
            sample.stride(1),
            world.data_ptr<float>(),
            static_cast<float>(tan_x),
            static_cast<float>(tan_y),
            static_cast<float>(depth));
    });
    cuda_check(cudaGetLastError(), "camera_sample_to_world_kernel");
    return world;
}

at::Tensor camera_sample_to_world_backward_cuda(
    const at::Tensor &grad_world,
    int64_t sample_count,
    double tan_x,
    double tan_y,
    double depth) {
    c10::cuda::CUDAGuard guard(grad_world.device());
    at::Tensor grad_sample = at::empty({sample_count, 2}, grad_world.options());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(grad_world.get_device());
    launch_1d(sample_count, [&](int blocks, int threads) {
        camera_sample_to_world_backward_kernel<<<blocks, threads, 0, stream>>>(
            sample_count,
            grad_world.data_ptr<float>(),
            grad_world.stride(0),
            grad_world.stride(1),
            grad_sample.data_ptr<float>(),
            static_cast<float>(tan_x),
            static_cast<float>(tan_y),
            static_cast<float>(depth));
    });
    cuda_check(cudaGetLastError(), "camera_sample_to_world_backward_kernel");
    return grad_sample;
}

at::Tensor camera_world_to_sample_cuda(
    const at::Tensor &point,
    double tan_x,
    double tan_y) {
    c10::cuda::CUDAGuard guard(point.device());
    at::Tensor sample = at::empty({point.size(0), 2}, point.options());
    const int64_t count = point.size(0);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(point.get_device());
    launch_1d(count, [&](int blocks, int threads) {
        camera_world_to_sample_kernel<<<blocks, threads, 0, stream>>>(
            count,
            point.data_ptr<float>(),
            point.stride(0),
            point.stride(1),
            sample.data_ptr<float>(),
            static_cast<float>(tan_x),
            static_cast<float>(tan_y));
    });
    cuda_check(cudaGetLastError(), "camera_world_to_sample_kernel");
    return sample;
}

at::Tensor camera_world_to_sample_backward_cuda(
    const at::Tensor &point,
    const at::Tensor &grad_sample,
    double tan_x,
    double tan_y) {
    c10::cuda::CUDAGuard guard(point.device());
    at::Tensor grad_point = at::empty({point.size(0), 3}, point.options());
    const int64_t count = point.size(0);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(point.get_device());
    launch_1d(count, [&](int blocks, int threads) {
        camera_world_to_sample_backward_kernel<<<blocks, threads, 0, stream>>>(
            count,
            point.data_ptr<float>(),
            point.stride(0),
            point.stride(1),
            grad_sample.data_ptr<float>(),
            grad_sample.stride(0),
            grad_sample.stride(1),
            grad_point.data_ptr<float>(),
            static_cast<float>(tan_x),
            static_cast<float>(tan_y));
    });
    cuda_check(cudaGetLastError(), "camera_world_to_sample_backward_kernel");
    return grad_point;
}

std::tuple<at::Tensor, at::Tensor> camera_sample_ray_cuda(
    const at::Tensor &sample,
    double tan_x,
    double tan_y) {
    c10::cuda::CUDAGuard guard(sample.device());
    at::Tensor origin = at::empty({sample.size(0), 3}, sample.options());
    at::Tensor direction = at::empty({sample.size(0), 3}, sample.options());
    const int64_t count = sample.size(0);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(sample.get_device());
    launch_1d(count, [&](int blocks, int threads) {
        camera_sample_ray_kernel<<<blocks, threads, 0, stream>>>(
            count,
            sample.data_ptr<float>(),
            sample.stride(0),
            sample.stride(1),
            origin.data_ptr<float>(),
            direction.data_ptr<float>(),
            static_cast<float>(tan_x),
            static_cast<float>(tan_y));
    });
    cuda_check(cudaGetLastError(), "camera_sample_ray_kernel");
    return {origin, direction};
}

at::Tensor camera_sample_ray_backward_cuda(
    const at::Tensor &sample,
    const at::Tensor *grad_direction,
    double tan_x,
    double tan_y) {
    c10::cuda::CUDAGuard guard(sample.device());
    at::Tensor grad_sample = at::empty({sample.size(0), 2}, sample.options());
    const int64_t count = sample.size(0);
    const float *grad_direction_ptr =
        grad_direction == nullptr || grad_direction->numel() == 0 ? nullptr : grad_direction->data_ptr<float>();
    const int64_t grad_direction_stride0 =
        grad_direction_ptr == nullptr ? 0 : grad_direction->stride(0);
    const int64_t grad_direction_stride1 =
        grad_direction_ptr == nullptr ? 0 : grad_direction->stride(1);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(sample.get_device());
    launch_1d(count, [&](int blocks, int threads) {
        camera_sample_ray_backward_kernel<<<blocks, threads, 0, stream>>>(
            count,
            sample.data_ptr<float>(),
            sample.stride(0),
            sample.stride(1),
            grad_direction_ptr,
            grad_direction_stride0,
            grad_direction_stride1,
            grad_sample.data_ptr<float>(),
            static_cast<float>(tan_x),
            static_cast<float>(tan_y));
    });
    cuda_check(cudaGetLastError(), "camera_sample_ray_backward_kernel");
    return grad_sample;
}

} // namespace raydn
