#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <torch/extension.h>

namespace {

constexpr int kBlockSize = 256;
constexpr int kComponentCount = 3;

void check_flat_tensor(const at::Tensor &tensor, const char *name, c10::ScalarType dtype) {
    TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(tensor.scalar_type() == dtype, name, " has the wrong dtype");
    TORCH_CHECK(tensor.dim() == 1, name, " must have shape (path_count,)");
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

__global__ void deterministic_accumulate_paths_kernel(
    const int *__restrict__ tx_id,
    const int *__restrict__ rx_id,
    const int *__restrict__ component_id,
    const float *__restrict__ path_gain,
    const float *__restrict__ field_real,
    const float *__restrict__ field_imag,
    float *__restrict__ component_power,
    float *__restrict__ component_field_real,
    float *__restrict__ component_field_imag,
    int64_t path_count,
    int64_t num_tx,
    int64_t num_rx) {
    const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    const int64_t cell_count = num_tx * num_rx;
    for (int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         idx < path_count;
         idx += stride) {
        const int cid = component_id[idx];
        const int tx = tx_id[idx];
        const int rx = rx_id[idx];
        if (cid < 0 || cid >= kComponentCount || tx < 0 || rx < 0 || tx >= num_tx || rx >= num_rx) {
            continue;
        }
        const int64_t cell = static_cast<int64_t>(tx) * num_rx + rx;
        const int64_t out = static_cast<int64_t>(cid) * cell_count + cell;
        atomicAdd(component_power + out, path_gain[idx]);
        atomicAdd(component_field_real + out, field_real[idx]);
        atomicAdd(component_field_imag + out, field_imag[idx]);
    }
}

__global__ void deterministic_finalize_accumulation_kernel(
    float *__restrict__ component_power,
    const float *__restrict__ component_field_real,
    const float *__restrict__ component_field_imag,
    float *__restrict__ power_total,
    float *__restrict__ field_total_real,
    float *__restrict__ field_total_imag,
    int64_t cell_count,
    int coherent) {
    const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (int64_t cell = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         cell < cell_count;
         cell += stride) {
        float real_sum = 0.0f;
        float imag_sum = 0.0f;
        float power_sum = 0.0f;
        for (int cid = 0; cid < kComponentCount; ++cid) {
            const int64_t out = static_cast<int64_t>(cid) * cell_count + cell;
            const float real = component_field_real[out];
            const float imag = component_field_imag[out];
            if (coherent) {
                const float coherent_power = real * real + imag * imag;
                component_power[out] = coherent_power;
                power_sum += coherent_power;
            } else {
                power_sum += component_power[out];
            }
            real_sum += real;
            imag_sum += imag;
        }
        if (coherent) {
            field_total_real[cell] = real_sum;
            field_total_imag[cell] = imag_sum;
            power_total[cell] = real_sum * real_sum + imag_sum * imag_sum;
        } else {
            power_total[cell] = power_sum;
            field_total_real[cell] = sqrtf(fmaxf(power_sum, 0.0f));
            field_total_imag[cell] = 0.0f;
        }
    }
}

}  // namespace

pybind11::dict cn_deterministic_accumulate_flat(
    at::Tensor tx_id,
    at::Tensor rx_id,
    at::Tensor component_id,
    at::Tensor path_gain,
    at::Tensor field_real,
    at::Tensor field_imag,
    int64_t num_tx,
    int64_t num_rx,
    bool coherent) {
    check_flat_tensor(tx_id, "tx_id", at::kInt);
    check_flat_tensor(rx_id, "rx_id", at::kInt);
    check_flat_tensor(component_id, "component_id", at::kInt);
    check_flat_tensor(path_gain, "path_gain", at::kFloat);
    check_flat_tensor(field_real, "field_real", at::kFloat);
    check_flat_tensor(field_imag, "field_imag", at::kFloat);
    TORCH_CHECK(rx_id.sizes() == tx_id.sizes(), "rx_id must match tx_id");
    TORCH_CHECK(component_id.sizes() == tx_id.sizes(), "component_id must match tx_id");
    TORCH_CHECK(path_gain.sizes() == tx_id.sizes(), "path_gain must match tx_id");
    TORCH_CHECK(field_real.sizes() == tx_id.sizes(), "field_real must match tx_id");
    TORCH_CHECK(field_imag.sizes() == tx_id.sizes(), "field_imag must match tx_id");
    TORCH_CHECK(num_tx >= 0, "num_tx must be non-negative");
    TORCH_CHECK(num_rx >= 0, "num_rx must be non-negative");

    auto fopts = path_gain.options();
    at::Tensor component_power = at::zeros({kComponentCount, num_tx, num_rx}, fopts);
    at::Tensor component_field_real = at::zeros({kComponentCount, num_tx, num_rx}, fopts);
    at::Tensor component_field_imag = at::zeros({kComponentCount, num_tx, num_rx}, fopts);
    at::Tensor power_total = at::empty({num_tx, num_rx}, fopts);
    at::Tensor field_total_real = at::empty({num_tx, num_rx}, fopts);
    at::Tensor field_total_imag = at::empty({num_tx, num_rx}, fopts);

    const int64_t path_count = tx_id.numel();
    const int64_t cell_count = num_tx * num_rx;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(path_gain.get_device()).stream();
    if (path_count > 0) {
        const int block_count = static_cast<int>((path_count + kBlockSize - 1) / kBlockSize);
        deterministic_accumulate_paths_kernel<<<block_count, kBlockSize, 0, stream>>>(
            tx_id.data_ptr<int>(),
            rx_id.data_ptr<int>(),
            component_id.data_ptr<int>(),
            path_gain.data_ptr<float>(),
            field_real.data_ptr<float>(),
            field_imag.data_ptr<float>(),
            component_power.data_ptr<float>(),
            component_field_real.data_ptr<float>(),
            component_field_imag.data_ptr<float>(),
            path_count,
            num_tx,
            num_rx);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    if (cell_count > 0) {
        const int block_count = static_cast<int>((cell_count + kBlockSize - 1) / kBlockSize);
        deterministic_finalize_accumulation_kernel<<<block_count, kBlockSize, 0, stream>>>(
            component_power.data_ptr<float>(),
            component_field_real.data_ptr<float>(),
            component_field_imag.data_ptr<float>(),
            power_total.data_ptr<float>(),
            field_total_real.data_ptr<float>(),
            field_total_imag.data_ptr<float>(),
            cell_count,
            coherent ? 1 : 0);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

    pybind11::dict out;
    out["power_total"] = power_total;
    out["field_total_real"] = field_total_real;
    out["field_total_imag"] = field_total_imag;
    out["component_power"] = component_power;
    out["component_field_real"] = component_field_real;
    out["component_field_imag"] = component_field_imag;
    return out;
}
