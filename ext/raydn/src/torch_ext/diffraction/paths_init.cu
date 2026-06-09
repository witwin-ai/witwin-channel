#include <raydn/diffraction/paths_init.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>

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

__global__ void init_dfr_path_outputs_kernel(
    int64_t capacity,
    int *__restrict__ out_count,
    uint8_t *__restrict__ out_valid,
    int *__restrict__ out_tx_id,
    int *__restrict__ out_rx_id,
    int *__restrict__ out_order,
    int *__restrict__ out_edge0,
    int *__restrict__ out_edge1,
    int *__restrict__ out_edge2,
    float *__restrict__ out_delay,
    float *__restrict__ out_field_x_re,
    float *__restrict__ out_field_x_im,
    float *__restrict__ out_field_y_re,
    float *__restrict__ out_field_y_im,
    float *__restrict__ out_field_z_re,
    float *__restrict__ out_field_z_im,
    float *__restrict__ out_p0,
    float *__restrict__ out_p1,
    float *__restrict__ out_p2) {
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx == 0) {
        out_count[0] = 0;
    }
    if (idx >= capacity) {
        return;
    }
    out_valid[idx] = 0u;
    out_tx_id[idx] = -1;
    out_rx_id[idx] = -1;
    out_order[idx] = 0;
    out_edge0[idx] = -1;
    out_edge1[idx] = -1;
    out_edge2[idx] = -1;
    out_delay[idx] = 0.0f;
    out_field_x_re[idx] = 0.0f;
    out_field_x_im[idx] = 0.0f;
    out_field_y_re[idx] = 0.0f;
    out_field_y_im[idx] = 0.0f;
    out_field_z_re[idx] = 0.0f;
    out_field_z_im[idx] = 0.0f;
    const int64_t vec = idx * 3;
    out_p0[vec + 0] = 0.0f;
    out_p0[vec + 1] = 0.0f;
    out_p0[vec + 2] = 0.0f;
    out_p1[vec + 0] = 0.0f;
    out_p1[vec + 1] = 0.0f;
    out_p1[vec + 2] = 0.0f;
    out_p2[vec + 0] = 0.0f;
    out_p2[vec + 1] = 0.0f;
    out_p2[vec + 2] = 0.0f;
}

} // namespace

void init_dfr_path_outputs_cuda(
    int64_t capacity,
    at::Tensor &out_count,
    at::Tensor &out_valid,
    at::Tensor &out_tx_id,
    at::Tensor &out_rx_id,
    at::Tensor &out_order,
    at::Tensor &out_edge0,
    at::Tensor &out_edge1,
    at::Tensor &out_edge2,
    at::Tensor &out_delay,
    at::Tensor &out_field_x_re,
    at::Tensor &out_field_x_im,
    at::Tensor &out_field_y_re,
    at::Tensor &out_field_y_im,
    at::Tensor &out_field_z_re,
    at::Tensor &out_field_z_im,
    at::Tensor &out_p0,
    at::Tensor &out_p1,
    at::Tensor &out_p2) {
    c10::cuda::CUDAGuard guard(out_count.device());
    const int64_t launch_count = capacity > 0 ? capacity : 1;
    constexpr int threads = 256;
    const int blocks = static_cast<int>((launch_count + threads - 1) / threads);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(out_count.get_device());
    init_dfr_path_outputs_kernel<<<blocks, threads, 0, stream>>>(
        capacity,
        out_count.data_ptr<int>(),
        reinterpret_cast<uint8_t *>(out_valid.data_ptr<bool>()),
        out_tx_id.data_ptr<int>(),
        out_rx_id.data_ptr<int>(),
        out_order.data_ptr<int>(),
        out_edge0.data_ptr<int>(),
        out_edge1.data_ptr<int>(),
        out_edge2.data_ptr<int>(),
        out_delay.data_ptr<float>(),
        out_field_x_re.data_ptr<float>(),
        out_field_x_im.data_ptr<float>(),
        out_field_y_re.data_ptr<float>(),
        out_field_y_im.data_ptr<float>(),
        out_field_z_re.data_ptr<float>(),
        out_field_z_im.data_ptr<float>(),
        out_p0.data_ptr<float>(),
        out_p1.data_ptr<float>(),
        out_p2.data_ptr<float>());
    cuda_check(cudaGetLastError(), "init_dfr_path_outputs_kernel");
}

} // namespace raydn
