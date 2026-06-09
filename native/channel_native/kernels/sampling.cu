#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime_api.h>

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

at::Tensor cn_mc_sample_directions_cuda(int64_t count, at::Tensor reference) {
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
