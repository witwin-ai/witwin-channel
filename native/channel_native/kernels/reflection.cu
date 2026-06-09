#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime_api.h>

#include <tuple>

namespace {

constexpr int kReflectionBlockSize = 256;

void check_tensor(
    const at::Tensor &tensor,
    const char *name,
    c10::ScalarType dtype,
    int64_t dimensions) {
    TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(tensor.scalar_type() == dtype, name, " has the wrong dtype");
    TORCH_CHECK(tensor.dim() == dimensions, name, " has the wrong rank");
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

__global__ void reflection_launch_inputs_kernel(
    const float *__restrict__ tx_positions,
    float *__restrict__ ray_o,
    bool *__restrict__ active,
    float *__restrict__ tx_pol,
    int64_t tx_index,
    int64_t sample_count) {
    const float tx_x = tx_positions[tx_index * 3 + 0];
    const float tx_y = tx_positions[tx_index * 3 + 1];
    const float tx_z = tx_positions[tx_index * 3 + 2];
    const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (int64_t sample = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         sample < sample_count;
         sample += stride) {
        float *origin = ray_o + sample * 3;
        origin[0] = tx_x;
        origin[1] = tx_y;
        origin[2] = tx_z;

        float *pol = tx_pol + sample * 3;
        pol[0] = 1.0f;
        pol[1] = 0.0f;
        pol[2] = 0.0f;

        active[sample] = true;
    }
}

}  // namespace

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> cn_mc_reflection_launch_inputs_cuda(
    at::Tensor tx_positions,
    int64_t tx_index,
    int64_t sample_count) {
    check_tensor(tx_positions, "tx_positions", at::kFloat, 2);
    TORCH_CHECK(tx_positions.size(1) == 3, "tx_positions must have shape (N, 3)");
    TORCH_CHECK(tx_index >= 0 && tx_index < tx_positions.size(0), "tx_index is out of range");
    TORCH_CHECK(sample_count >= 0, "sample_count must be non-negative");

    auto ray_o = at::empty({sample_count, 3}, tx_positions.options());
    auto ray_tmax = at::empty({0}, tx_positions.options());
    auto active = at::empty({sample_count}, tx_positions.options().dtype(at::kBool));
    auto tx_pol = at::empty({sample_count, 3}, tx_positions.options());
    if (sample_count > 0) {
        cudaStream_t stream = at::cuda::getCurrentCUDAStream(tx_positions.get_device()).stream();
        const int block_count = static_cast<int>((sample_count + kReflectionBlockSize - 1) / kReflectionBlockSize);
        reflection_launch_inputs_kernel<<<block_count, kReflectionBlockSize, 0, stream>>>(
            tx_positions.data_ptr<float>(),
            ray_o.data_ptr<float>(),
            active.data_ptr<bool>(),
            tx_pol.data_ptr<float>(),
            tx_index,
            sample_count);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return {ray_o, ray_tmax, active, tx_pol};
}
