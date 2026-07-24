#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime_api.h>
#include "torch_cuda_minimal.h"

namespace {

constexpr double kPi = 3.141592653589793238462643383279502884;

void check_reference(const at::Tensor& tensor) {
    TORCH_CHECK(tensor.is_cuda(), "reference must be a CUDA tensor");
    TORCH_CHECK(tensor.scalar_type() == at::kFloat, "reference must be float32");
    TORCH_CHECK(tensor.dim() == 2, "reference must have rank 2");
    TORCH_CHECK(tensor.is_contiguous(), "reference must be contiguous");
}

__device__ unsigned long long splitmix64(unsigned long long x) {
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}

__device__ float uniform01_from_u64(unsigned long long value) {
    constexpr double scale = 1.0 / 9007199254740992.0;
    return static_cast<float>(((value >> 11) & 0x1fffffffffffffULL) * scale);
}

__global__ void bdpt_launch_state_kernel(
    int64_t total,
    int64_t samples,
    int64_t sample_streams,
    unsigned long long seed,
    int* tx_id,
    int* sample_id,
    int* stream_id,
    int64_t* light_seed) {
    int64_t linear = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (linear >= total) {
        return;
    }
    int64_t stream = linear % sample_streams;
    int64_t sample = (linear / sample_streams) % samples;
    int64_t tx = linear / (sample_streams * samples);
    tx_id[linear] = static_cast<int>(tx);
    sample_id[linear] = static_cast<int>(sample);
    stream_id[linear] = static_cast<int>(stream);

    unsigned long long base = seed
        ^ (static_cast<unsigned long long>(tx) * 0xd1b54a32d192ed03ULL)
        ^ (static_cast<unsigned long long>(sample) * 0x94d049bb133111ebULL)
        ^ (static_cast<unsigned long long>(stream) * 0xbf58476d1ce4e5b9ULL);
    light_seed[linear] = static_cast<int64_t>(splitmix64(base ^ 0x1111111111111111ULL) & 0x7fffffffffffffffULL);
}

__global__ void bdpt_sample_directions_kernel(
    int64_t count,
    unsigned long long seed,
    float* directions) {
    int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= count) {
        return;
    }
    unsigned long long base = seed ^ (static_cast<unsigned long long>(index) * 0xd1b54a32d192ed03ULL);
    float u0 = uniform01_from_u64(splitmix64(base ^ 0x57c5d1f5f8c0a9b3ULL));
    float u1 = uniform01_from_u64(splitmix64(base ^ 0xa24baed4963ee407ULL));
    float z = 1.0f - 2.0f * u0;
    float phi = static_cast<float>(2.0 * kPi) * u1;
    float radial = sqrtf(fmaxf(1.0f - z * z, 0.0f));
    float* out = directions + index * 3;
    out[0] = radial * cosf(phi);
    out[1] = radial * sinf(phi);
    out[2] = z;
}

}  // namespace

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>
channel_bdpt_launch_state_cuda(
    at::Tensor reference,
    int64_t tx_count,
    int64_t samples,
    int64_t sample_streams,
    int64_t seed) {
    check_reference(reference);
    TORCH_CHECK(tx_count >= 0, "tx_count must be non-negative");
    TORCH_CHECK(samples > 0, "samples must be positive");
    TORCH_CHECK(sample_streams > 0, "sample_streams must be positive");
    TORCH_CHECK(seed >= 0, "seed must be non-negative");
    int64_t total = tx_count * samples * sample_streams;
    auto int_options = reference.options().dtype(at::kInt);
    auto long_options = reference.options().dtype(at::kLong);
    auto tx_id = at::empty({total}, int_options);
    auto sample_id = at::empty({total}, int_options);
    auto stream_id = at::empty({total}, int_options);
    auto light_seed = at::empty({total}, long_options);
    if (total > 0) {
        constexpr int threads = 256;
        int blocks = static_cast<int>((total + threads - 1) / threads);
        cudaStream_t stream = at::cuda::getCurrentCUDAStream(reference.get_device()).stream();
        bdpt_launch_state_kernel<<<blocks, threads, 0, stream>>>(
            total,
            samples,
            sample_streams,
            static_cast<unsigned long long>(seed),
            tx_id.data_ptr<int>(),
            sample_id.data_ptr<int>(),
            stream_id.data_ptr<int>(),
            light_seed.data_ptr<int64_t>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return {tx_id, sample_id, stream_id, light_seed};
}

at::Tensor channel_bdpt_sample_directions_cuda(int64_t count, at::Tensor reference, int64_t seed) {
    check_reference(reference);
    TORCH_CHECK(count >= 0, "count must be non-negative");
    TORCH_CHECK(seed >= 0, "seed must be non-negative");
    auto directions = at::empty({count, 3}, reference.options().dtype(at::kFloat));
    if (count > 0) {
        constexpr int threads = 256;
        int blocks = static_cast<int>((count + threads - 1) / threads);
        cudaStream_t stream = at::cuda::getCurrentCUDAStream(reference.get_device()).stream();
        bdpt_sample_directions_kernel<<<blocks, threads, 0, stream>>>(
            count,
            static_cast<unsigned long long>(seed),
            directions.data_ptr<float>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return directions;
}
