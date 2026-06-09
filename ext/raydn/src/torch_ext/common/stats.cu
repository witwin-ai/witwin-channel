#include <raydn/common/stats.h>

#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>

namespace raydn {

namespace {

__global__ void reflection_trace_stats_kernel(
    const bool *__restrict__ valid,
    const float *__restrict__ t,
    int64_t ray_count,
    int64_t max_bounces,
    long long *__restrict__ counts,
    double *__restrict__ checksum) {
    const int ray_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (ray_idx >= ray_count)
        return;
    int valid_for_ray = 0;
    double local_sum = 0.0;
    for (int64_t bounce = 0; bounce < max_bounces; ++bounce) {
        const int64_t slot = static_cast<int64_t>(ray_idx) * max_bounces + bounce;
        if (valid[slot]) {
            ++valid_for_ray;
            local_sum += static_cast<double>(t[slot]);
        }
    }
    if (valid_for_ray != 0) {
        atomicAdd(reinterpret_cast<unsigned long long *>(counts), static_cast<unsigned long long>(valid_for_ray));
        atomicAdd(checksum, local_sum);
    }
    if (valid_for_ray == max_bounces) {
        atomicAdd(
            reinterpret_cast<unsigned long long *>(counts + 1),
            static_cast<unsigned long long>(1));
    }
}

__global__ void diffraction_path_stats_kernel(
    const int *__restrict__ count,
    const bool *__restrict__ valid,
    const float *__restrict__ delay,
    int64_t capacity,
    long long *__restrict__ valid_count,
    double *__restrict__ checksum) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        valid_count[0] = static_cast<long long>(count[0]);
    }
    const int path_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (path_idx >= capacity)
        return;
    if (valid[path_idx]) {
        atomicAdd(checksum, static_cast<double>(delay[path_idx]));
    }
}

__global__ void default_dfr_material_kernel(
    int64_t count,
    float *__restrict__ eta_r,
    float *__restrict__ sigma,
    float *__restrict__ mu_r,
    float *__restrict__ gain,
    bool *__restrict__ valid) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count)
        return;
    eta_r[idx] = 1.f;
    sigma[idx] = 0.f;
    mu_r[idx] = 1.f;
    gain[idx] = 1.f;
    valid[idx] = true;
}

__global__ void intersection_valid_from_t_kernel(
    const float *__restrict__ t,
    int64_t count,
    int64_t t_stride,
    bool *__restrict__ out_valid) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count)
        return;
    out_valid[idx] = isfinite(t[static_cast<int64_t>(idx) * t_stride]);
}

__global__ void intersection_valid_from_shape_kernel(
    const int *__restrict__ shape_id,
    int64_t count,
    int64_t shape_stride,
    bool *__restrict__ out_valid) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count)
        return;
    out_valid[idx] = shape_id[static_cast<int64_t>(idx) * shape_stride] >= 0;
}

void zero_tensor_async(const at::Tensor &tensor, cudaStream_t stream) {
    if (tensor.defined() && tensor.numel() > 0) {
        cudaMemsetAsync(tensor.data_ptr(), 0, static_cast<size_t>(tensor.nbytes()), stream);
    }
}

} // namespace

std::tuple<at::Tensor, at::Tensor> reflection_trace_stats_cuda(
    const at::Tensor &valid,
    const at::Tensor &t) {
    const int64_t ray_count = valid.size(0);
    const int64_t max_bounces = valid.size(1);
    at::Tensor counts = at::empty({2}, valid.options().dtype(at::kLong));
    at::Tensor checksum = at::empty({1}, t.options().dtype(at::kDouble));
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(valid.get_device()).stream();
    zero_tensor_async(counts, stream);
    zero_tensor_async(checksum, stream);
    const int threads = 128;
    const int blocks = static_cast<int>((ray_count + threads - 1) / threads);
    reflection_trace_stats_kernel<<<blocks, threads, 0, stream>>>(
        valid.data_ptr<bool>(),
        t.data_ptr<float>(),
        ray_count,
        max_bounces,
        counts.data_ptr<long long>(),
        checksum.data_ptr<double>());
    return std::make_tuple(counts, checksum);
}

std::tuple<at::Tensor, at::Tensor> diffraction_path_stats_cuda(
    const at::Tensor &count,
    const at::Tensor &valid,
    const at::Tensor &delay) {
    const int64_t capacity = valid.size(0);
    at::Tensor valid_count = at::empty({1}, valid.options().dtype(at::kLong));
    at::Tensor checksum = at::empty({1}, delay.options().dtype(at::kDouble));
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(valid.get_device()).stream();
    zero_tensor_async(valid_count, stream);
    zero_tensor_async(checksum, stream);
    const int threads = 128;
    const int blocks = static_cast<int>((capacity + threads - 1) / threads);
    diffraction_path_stats_kernel<<<blocks, threads, 0, stream>>>(
        count.data_ptr<int>(),
        valid.data_ptr<bool>(),
        delay.data_ptr<float>(),
        capacity,
        valid_count.data_ptr<long long>(),
        checksum.data_ptr<double>());
    return std::make_tuple(valid_count, checksum);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor> default_dfr_material_cuda(
    int64_t count,
    const at::Tensor &like) {
    at::Tensor eta_r = at::empty({count}, like.options());
    at::Tensor sigma = at::empty({count}, like.options());
    at::Tensor mu_r = at::empty({count}, like.options());
    at::Tensor gain = at::empty({count}, like.options());
    at::Tensor valid = at::empty({count}, like.options().dtype(at::kBool));
    if (count > 0) {
        cudaStream_t stream = at::cuda::getCurrentCUDAStream(like.get_device()).stream();
        const int threads = 128;
        const int blocks = static_cast<int>((count + threads - 1) / threads);
        default_dfr_material_kernel<<<blocks, threads, 0, stream>>>(
            count,
            eta_r.data_ptr<float>(),
            sigma.data_ptr<float>(),
            mu_r.data_ptr<float>(),
            gain.data_ptr<float>(),
            valid.data_ptr<bool>());
    }
    return std::make_tuple(eta_r, sigma, mu_r, gain, valid);
}

at::Tensor intersection_valid_cuda(
    const at::Tensor &t,
    const at::Tensor &shape_id) {
    const int64_t count = t.numel();
    at::Tensor valid = at::empty({count}, t.options().dtype(at::kBool));
    if (count == 0) {
        return valid;
    }

    cudaStream_t stream = at::cuda::getCurrentCUDAStream(t.get_device()).stream();
    const int threads = 128;
    const int blocks = static_cast<int>((count + threads - 1) / threads);
    if (shape_id.defined() && shape_id.numel() == count) {
        intersection_valid_from_shape_kernel<<<blocks, threads, 0, stream>>>(
            shape_id.data_ptr<int>(),
            count,
            shape_id.stride(0),
            valid.data_ptr<bool>());
    } else {
        intersection_valid_from_t_kernel<<<blocks, threads, 0, stream>>>(
            t.data_ptr<float>(),
            count,
            t.stride(0),
            valid.data_ptr<bool>());
    }
    return valid;
}

} // namespace raydn
