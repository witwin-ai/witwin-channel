#include "bdpt_connect_common.cuh"

namespace {

__global__ void bdpt_mis_weights_kernel(
    int64_t count,
    const float* pdf,
    const float* strategy_pdf_sum,
    int mode_id,
    float beta,
    float* weights) {
    int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= count) {
        return;
    }
    float value = pdf[index];
    float sum = strategy_pdf_sum[0];
    if (value <= 0.0f || sum <= 0.0f) {
        weights[index] = 0.0f;
        return;
    }
    if (mode_id == 0) {
        weights[index] = 1.0f;
    } else if (mode_id == 1) {
        weights[index] = value / fmaxf(sum, 1.17549435e-38f);
    } else {
        weights[index] = powf(value, beta) / fmaxf(sum, 1.17549435e-38f);
    }
}

}  // namespace

at::Tensor channel_bdpt_mis_weights_cuda(
    at::Tensor pdf,
    at::Tensor strategy_pdf_sum,
    int64_t mode_id,
    double beta) {
    check_float_cuda(pdf, "pdf", 1);
    check_float_cuda(strategy_pdf_sum, "strategy_pdf_sum", 0);
    TORCH_CHECK(mode_id >= 0 && mode_id <= 2, "mode_id must be 0, 1, or 2");
    TORCH_CHECK(beta > 0.0, "beta must be positive");
    auto weights = at::empty_like(pdf);
    int64_t count = pdf.numel();
    if (count > 0) {
        constexpr int threads = 256;
        int blocks = static_cast<int>((count + threads - 1) / threads);
        cudaStream_t stream = at::cuda::getCurrentCUDAStream(pdf.get_device()).stream();
        bdpt_mis_weights_kernel<<<blocks, threads, 0, stream>>>(
            count,
            pdf.data_ptr<float>(),
            strategy_pdf_sum.data_ptr<float>(),
            static_cast<int>(mode_id),
            static_cast<float>(beta),
            weights.data_ptr<float>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return weights;
}
