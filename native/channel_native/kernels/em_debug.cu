#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <torch/extension.h>

#include "../em/layer_stack.cuh"
#include "../tensor_checks.h"

// Debug/parity surface for the shared em/ layer-stack core: evaluates the
// full stack r/t (both polarizations) plus power R/T per input angle. This is
// the oracle-parity op the CPU complex128 golden tests compare against.

namespace {

constexpr int kBlockSize = 256;
namespace em = channel_native::em;
namespace utd = witwin::channel::native_ext;

__global__ void em_layer_stack_eval_kernel(
    int64_t count,
    const float* cos_theta,
    const int* material_id,
    const int* layer_offset,
    const int* layer_count,
    const float* layer_thickness_m,
    const float* layer_eps_r,
    const float* layer_sigma_e,
    const float* layer_mu_r,
    int64_t material_count,
    float frequency_hz,
    float* r_te_real,
    float* r_te_imag,
    float* r_tm_real,
    float* r_tm_imag,
    float* t_te_real,
    float* t_te_imag,
    float* t_tm_real,
    float* t_tm_imag,
    float* cap_r_te,
    float* cap_r_tm,
    float* cap_t_te,
    float* cap_t_tm) {
    for (int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < count;
         index += static_cast<int64_t>(blockDim.x) * gridDim.x) {
        const int material = material_id[index];
        if (material < 0 || static_cast<int64_t>(material) >= material_count) {
            r_te_real[index] = 0.0f;
            r_te_imag[index] = 0.0f;
            r_tm_real[index] = 0.0f;
            r_tm_imag[index] = 0.0f;
            t_te_real[index] = 0.0f;
            t_te_imag[index] = 0.0f;
            t_tm_real[index] = 0.0f;
            t_tm_imag[index] = 0.0f;
            cap_r_te[index] = 0.0f;
            cap_r_tm[index] = 0.0f;
            cap_t_te[index] = 0.0f;
            cap_t_tm[index] = 0.0f;
            continue;
        }
        em::LayerView layers{
            layer_offset,
            layer_count,
            layer_thickness_m,
            layer_eps_r,
            layer_sigma_e,
            layer_mu_r,
            material,
        };
        const em::StackRT te = em::stack_rt(
            cos_theta[index], layers, frequency_hz, em::kPolTE);
        const em::StackRT tm = em::stack_rt(
            cos_theta[index], layers, frequency_hz, em::kPolTM);
        r_te_real[index] = te.r.re;
        r_te_imag[index] = te.r.im;
        r_tm_real[index] = tm.r.re;
        r_tm_imag[index] = tm.r.im;
        t_te_real[index] = te.t.re;
        t_te_imag[index] = te.t.im;
        t_tm_real[index] = tm.t.re;
        t_tm_imag[index] = tm.t.im;
        cap_r_te[index] = te.cap_r;
        cap_r_tm[index] = tm.cap_r;
        cap_t_te[index] = te.cap_t;
        cap_t_tm[index] = tm.cap_t;
    }
}

int launch_blocks(int64_t count) {
    return static_cast<int>((count + kBlockSize - 1) / kBlockSize);
}

}  // namespace

pybind11::dict cn_em_layer_stack_eval(
    at::Tensor cos_theta,
    at::Tensor material_id,
    at::Tensor layer_offset,
    at::Tensor layer_count,
    at::Tensor layer_thickness_m,
    at::Tensor layer_eps_r,
    at::Tensor layer_sigma_e,
    at::Tensor layer_mu_r,
    double frequency_hz) {
    using channel_native::check_flat_tensor;
    check_flat_tensor(cos_theta, "cos_theta", at::kFloat);
    check_flat_tensor(material_id, "material_id", at::kInt);
    check_flat_tensor(layer_offset, "layer_offset", at::kInt);
    check_flat_tensor(layer_count, "layer_count", at::kInt);
    check_flat_tensor(layer_thickness_m, "layer_thickness_m", at::kFloat);
    check_flat_tensor(layer_eps_r, "layer_eps_r", at::kFloat);
    check_flat_tensor(layer_sigma_e, "layer_sigma_e", at::kFloat);
    check_flat_tensor(layer_mu_r, "layer_mu_r", at::kFloat);
    const int64_t count = cos_theta.size(0);
    const int64_t material_count = layer_offset.size(0);
    const int64_t layer_total = layer_thickness_m.size(0);
    TORCH_CHECK(material_id.size(0) == count, "material_id must match cos_theta rows");
    TORCH_CHECK(layer_count.size(0) == material_count,
                "layer_count must match layer_offset rows");
    for (const auto& tensor : {layer_eps_r, layer_sigma_e, layer_mu_r})
        TORCH_CHECK(tensor.size(0) == layer_total,
                    "layer parameter tensors must match layer_thickness_m rows");
    for (const auto& tensor : {material_id, layer_offset, layer_count,
                               layer_thickness_m, layer_eps_r, layer_sigma_e,
                               layer_mu_r})
        TORCH_CHECK(tensor.get_device() == cos_theta.get_device(),
                    "em_layer_stack_eval tensors must share one CUDA device");
    TORCH_CHECK(frequency_hz > 0.0, "frequency_hz must be positive");

    auto options = cos_theta.options();
    static constexpr const char* kFields[12] = {
        "r_te_real", "r_te_imag", "r_tm_real", "r_tm_imag",
        "t_te_real", "t_te_imag", "t_tm_real", "t_tm_imag",
        "cap_R_te", "cap_R_tm", "cap_T_te", "cap_T_tm",
    };
    at::Tensor outputs[12];
    for (int field = 0; field < 12; ++field)
        outputs[field] = at::empty({count}, options);
    if (count > 0) {
        cudaStream_t stream =
            at::cuda::getCurrentCUDAStream(cos_theta.get_device()).stream();
        em_layer_stack_eval_kernel<<<launch_blocks(count), kBlockSize, 0, stream>>>(
            count,
            cos_theta.data_ptr<float>(),
            material_id.data_ptr<int>(),
            layer_offset.data_ptr<int>(),
            layer_count.data_ptr<int>(),
            layer_thickness_m.data_ptr<float>(),
            layer_eps_r.data_ptr<float>(),
            layer_sigma_e.data_ptr<float>(),
            layer_mu_r.data_ptr<float>(),
            material_count,
            static_cast<float>(frequency_hz),
            outputs[0].data_ptr<float>(),
            outputs[1].data_ptr<float>(),
            outputs[2].data_ptr<float>(),
            outputs[3].data_ptr<float>(),
            outputs[4].data_ptr<float>(),
            outputs[5].data_ptr<float>(),
            outputs[6].data_ptr<float>(),
            outputs[7].data_ptr<float>(),
            outputs[8].data_ptr<float>(),
            outputs[9].data_ptr<float>(),
            outputs[10].data_ptr<float>(),
            outputs[11].data_ptr<float>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    pybind11::dict out;
    for (int field = 0; field < 12; ++field)
        out[kFields[field]] = outputs[field];
    return out;
}
