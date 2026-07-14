#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <torch/extension.h>

#include "../em/layer_stack.cuh"
#include "../field_transport_ad.cuh"
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

// ---------------------------------------------------------------------------
// Backward / JVP companions of em_layer_stack_eval_kernel (plan 07 AD-3, MC
// transmission radiomap). Differentiable inputs: cos_theta (per row), the CSR
// layer thickness / eps_r / sigma_e, and the carrier frequency. layer_mu_r,
// the material ids and the CSR topology stay fixed. Derivatives come from
// stack_rt_dual, which mirrors em::stack_rt clamp for clamp; invalid material
// rows produced zero outputs in the forward and carry zero derivatives here.
// ---------------------------------------------------------------------------

namespace ad = channel_native::field_transport_ad;

struct StackZeroSeed {
    __device__ ad::LayerSeed operator()(int) const { return {0.0f, 0.0f, 0.0f}; }
};

struct StackBasisSeed {
    int slot;
    int param;  // 0 thickness, 1 eps, 2 sigma
    __device__ ad::LayerSeed operator()(int query) const {
        ad::LayerSeed seed{0.0f, 0.0f, 0.0f};
        if (query == slot) {
            if (param == 0)
                seed.d_thickness = 1.0f;
            else if (param == 1)
                seed.d_eps = 1.0f;
            else
                seed.d_sigma = 1.0f;
        }
        return seed;
    }
};

struct StackTangentSeed {
    const float* t_thickness;
    const float* t_eps;
    const float* t_sigma;
    __device__ ad::LayerSeed operator()(int query) const {
        return {
            t_thickness != nullptr ? t_thickness[query] : 0.0f,
            t_eps != nullptr ? t_eps[query] : 0.0f,
            t_sigma != nullptr ? t_sigma[query] : 0.0f};
    }
};

// Kernel-argument bundles for the twelve output arrays (passed by value so
// the pointers live in kernel parameter space, not host memory).
struct StackGradPtrs {
    const float* p[12];
};

struct StackTangentPtrs {
    float* p[12];
};

// Fold the twelve per-output tangents of one (te, tm) dual evaluation against
// the row's cotangents (output order matches cn_em_layer_stack_eval).
__device__ __forceinline__ float stack_adj_combine(
    const ad::DualStackRT& te,
    const ad::DualStackRT& tm,
    const float g[12]) {
    return g[0] * te.r.d.re + g[1] * te.r.d.im +
           g[2] * tm.r.d.re + g[3] * tm.r.d.im +
           g[4] * te.t.d.re + g[5] * te.t.d.im +
           g[6] * tm.t.d.re + g[7] * tm.t.d.im +
           g[8] * te.cap_r.d + g[9] * tm.cap_r.d +
           g[10] * te.cap_t.d + g[11] * tm.cap_t.d;
}

__global__ void em_layer_stack_backward_kernel(
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
    StackGradPtrs grad_outputs,  // 12 nullable cotangent arrays
    float* grad_cos_theta,
    float* grad_layer_thickness,
    float* grad_layer_eps_r,
    float* grad_layer_sigma_e,
    float* grad_frequency) {
    for (int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < count;
         index += static_cast<int64_t>(blockDim.x) * gridDim.x) {
        const int material = material_id[index];
        if (material < 0 || static_cast<int64_t>(material) >= material_count)
            continue;
        float g[12];
        bool any = false;
        for (int field = 0; field < 12; ++field) {
            g[field] = grad_outputs.p[field] != nullptr
                           ? grad_outputs.p[field][index]
                           : 0.0f;
            any = any || g[field] != 0.0f;
        }
        if (!any)
            continue;
        em::LayerView layers{
            layer_offset,
            layer_count,
            layer_thickness_m,
            layer_eps_r,
            layer_sigma_e,
            layer_mu_r,
            material,
        };
        const float ct = cos_theta[index];
        const StackZeroSeed zero_seed;
        if (grad_frequency != nullptr) {
            const ad::DualStackRT te = ad::stack_rt_dual(
                ct, layers, frequency_hz, 0.0f, 1.0f, em::kPolTE, zero_seed);
            const ad::DualStackRT tm = ad::stack_rt_dual(
                ct, layers, frequency_hz, 0.0f, 1.0f, em::kPolTM, zero_seed);
            atomicAdd(grad_frequency, stack_adj_combine(te, tm, g));
        }
        if (grad_cos_theta != nullptr) {
            const ad::DualStackRT te = ad::stack_rt_dual(
                ct, layers, frequency_hz, 1.0f, 0.0f, em::kPolTE, zero_seed);
            const ad::DualStackRT tm = ad::stack_rt_dual(
                ct, layers, frequency_hz, 1.0f, 0.0f, em::kPolTM, zero_seed);
            grad_cos_theta[index] = stack_adj_combine(te, tm, g);
        }
        if (grad_layer_thickness != nullptr || grad_layer_eps_r != nullptr ||
            grad_layer_sigma_e != nullptr) {
            const int first = layer_offset[material];
            const int layers_in_material = layer_count[material];
            for (int layer = 0; layer < layers_in_material; ++layer) {
                const int slot = first + layer;
                for (int param = 0; param < 3; ++param) {
                    float* destination = param == 0 ? grad_layer_thickness
                                         : param == 1 ? grad_layer_eps_r
                                                      : grad_layer_sigma_e;
                    if (destination == nullptr)
                        continue;
                    const StackBasisSeed seed{slot, param};
                    const ad::DualStackRT te = ad::stack_rt_dual(
                        ct, layers, frequency_hz, 0.0f, 0.0f, em::kPolTE, seed);
                    const ad::DualStackRT tm = ad::stack_rt_dual(
                        ct, layers, frequency_hz, 0.0f, 0.0f, em::kPolTM, seed);
                    atomicAdd(destination + slot, stack_adj_combine(te, tm, g));
                }
            }
        }
    }
}

__global__ void em_layer_stack_jvp_kernel(
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
    const float* tangent_cos_theta,
    const float* tangent_layer_thickness,
    const float* tangent_layer_eps_r,
    const float* tangent_layer_sigma_e,
    float tangent_frequency,
    StackTangentPtrs output_tangents) {  // 12 tangent arrays
    const StackTangentSeed seed{
        tangent_layer_thickness, tangent_layer_eps_r, tangent_layer_sigma_e};
    for (int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < count;
         index += static_cast<int64_t>(blockDim.x) * gridDim.x) {
        const int material = material_id[index];
        if (material < 0 || static_cast<int64_t>(material) >= material_count) {
            for (int field = 0; field < 12; ++field)
                output_tangents.p[field][index] = 0.0f;
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
        const float d_cos =
            tangent_cos_theta != nullptr ? tangent_cos_theta[index] : 0.0f;
        const ad::DualStackRT te = ad::stack_rt_dual(
            cos_theta[index], layers, frequency_hz, d_cos, tangent_frequency,
            em::kPolTE, seed);
        const ad::DualStackRT tm = ad::stack_rt_dual(
            cos_theta[index], layers, frequency_hz, d_cos, tangent_frequency,
            em::kPolTM, seed);
        output_tangents.p[0][index] = te.r.d.re;
        output_tangents.p[1][index] = te.r.d.im;
        output_tangents.p[2][index] = tm.r.d.re;
        output_tangents.p[3][index] = tm.r.d.im;
        output_tangents.p[4][index] = te.t.d.re;
        output_tangents.p[5][index] = te.t.d.im;
        output_tangents.p[6][index] = tm.t.d.re;
        output_tangents.p[7][index] = tm.t.d.im;
        output_tangents.p[8][index] = te.cap_r.d;
        output_tangents.p[9][index] = tm.cap_r.d;
        output_tangents.p[10][index] = te.cap_t.d;
        output_tangents.p[11][index] = tm.cap_t.d;
    }
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

namespace {

constexpr const char* kStackFields[12] = {
    "r_te_real", "r_te_imag", "r_tm_real", "r_tm_imag",
    "t_te_real", "t_te_imag", "t_tm_real", "t_tm_imag",
    "cap_R_te", "cap_R_tm", "cap_T_te", "cap_T_tm",
};

at::Tensor stack_zero_filled(at::IntArrayRef sizes, const at::TensorOptions& options) {
    auto tensor = at::empty(sizes, options);
    if (tensor.numel() > 0) {
        cudaStream_t stream =
            at::cuda::getCurrentCUDAStream(tensor.get_device()).stream();
        C10_CUDA_CHECK(cudaMemsetAsync(
            tensor.data_ptr(),
            0,
            static_cast<size_t>(tensor.numel()) * tensor.element_size(),
            stream));
    }
    return tensor;
}

void check_stack_primal(
    const at::Tensor& cos_theta,
    const at::Tensor& material_id,
    const at::Tensor& layer_offset,
    const at::Tensor& layer_count,
    const at::Tensor& layer_thickness_m,
    const at::Tensor& layer_eps_r,
    const at::Tensor& layer_sigma_e,
    const at::Tensor& layer_mu_r,
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
    TORCH_CHECK(
        material_id.size(0) == cos_theta.size(0),
        "material_id must match cos_theta rows");
    TORCH_CHECK(
        layer_count.size(0) == layer_offset.size(0),
        "layer_count must match layer_offset rows");
    const int64_t layer_total = layer_thickness_m.size(0);
    for (const auto& tensor : {layer_eps_r, layer_sigma_e, layer_mu_r})
        TORCH_CHECK(tensor.size(0) == layer_total,
                    "layer parameter tensors must match layer_thickness_m rows");
    TORCH_CHECK(frequency_hz > 0.0, "frequency_hz must be positive");
}

const float* stack_optional_grad(
    pybind11::object value,
    at::Tensor& storage,
    const char* name,
    int64_t rows,
    const at::Tensor& reference) {
    if (value.is_none())
        return nullptr;
    storage = value.cast<at::Tensor>();
    TORCH_CHECK(storage.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(storage.scalar_type() == at::kFloat, name, " must be float32");
    TORCH_CHECK(storage.dim() == 1 && storage.size(0) == rows,
                name, " must have one value per row");
    TORCH_CHECK(storage.is_contiguous(), name, " must be contiguous");
    TORCH_CHECK(
        storage.get_device() == reference.get_device(),
        name, " must share the primal device");
    return storage.data_ptr<float>();
}

}  // namespace

pybind11::dict cn_em_layer_stack_backward(
    at::Tensor cos_theta,
    at::Tensor material_id,
    at::Tensor layer_offset,
    at::Tensor layer_count,
    at::Tensor layer_thickness_m,
    at::Tensor layer_eps_r,
    at::Tensor layer_sigma_e,
    at::Tensor layer_mu_r,
    double frequency_hz,
    pybind11::sequence grad_outputs,
    bool need_cos_theta,
    bool need_layers,
    bool need_frequency) {
    check_stack_primal(
        cos_theta, material_id, layer_offset, layer_count, layer_thickness_m,
        layer_eps_r, layer_sigma_e, layer_mu_r, frequency_hz);
    TORCH_CHECK(
        grad_outputs.size() == 12,
        "grad_outputs must carry the twelve stack output cotangents");
    const int64_t count = cos_theta.size(0);
    const int64_t material_count = layer_offset.size(0);
    const int64_t layer_total = layer_thickness_m.size(0);

    StackGradPtrs grads{};
    at::Tensor storages[12];
    for (int field = 0; field < 12; ++field) {
        grads.p[field] = stack_optional_grad(
            grad_outputs[field], storages[field], kStackFields[field], count,
            cos_theta);
    }

    auto options = cos_theta.options();
    auto grad_cos_theta = stack_zero_filled({count}, options);
    auto grad_layer_thickness = stack_zero_filled({layer_total}, options);
    auto grad_layer_eps_r = stack_zero_filled({layer_total}, options);
    auto grad_layer_sigma_e = stack_zero_filled({layer_total}, options);
    auto grad_frequency = stack_zero_filled({1}, options);
    if (count > 0 && (need_cos_theta || need_layers || need_frequency)) {
        cudaStream_t stream =
            at::cuda::getCurrentCUDAStream(cos_theta.get_device()).stream();
        em_layer_stack_backward_kernel<<<launch_blocks(count), kBlockSize, 0, stream>>>(
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
            grads,
            need_cos_theta ? grad_cos_theta.data_ptr<float>() : nullptr,
            need_layers ? grad_layer_thickness.data_ptr<float>() : nullptr,
            need_layers ? grad_layer_eps_r.data_ptr<float>() : nullptr,
            need_layers ? grad_layer_sigma_e.data_ptr<float>() : nullptr,
            need_frequency ? grad_frequency.data_ptr<float>() : nullptr);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    pybind11::dict out;
    out["grad_cos_theta"] = grad_cos_theta;
    out["grad_layer_thickness_m"] = grad_layer_thickness;
    out["grad_layer_eps_r"] = grad_layer_eps_r;
    out["grad_layer_sigma_e"] = grad_layer_sigma_e;
    out["grad_frequency"] = grad_frequency;
    return out;
}

pybind11::dict cn_em_layer_stack_jvp(
    at::Tensor cos_theta,
    at::Tensor material_id,
    at::Tensor layer_offset,
    at::Tensor layer_count,
    at::Tensor layer_thickness_m,
    at::Tensor layer_eps_r,
    at::Tensor layer_sigma_e,
    at::Tensor layer_mu_r,
    double frequency_hz,
    pybind11::object tangent_cos_theta,
    pybind11::object tangent_layer_thickness,
    pybind11::object tangent_layer_eps_r,
    pybind11::object tangent_layer_sigma_e,
    double tangent_frequency) {
    check_stack_primal(
        cos_theta, material_id, layer_offset, layer_count, layer_thickness_m,
        layer_eps_r, layer_sigma_e, layer_mu_r, frequency_hz);
    const int64_t count = cos_theta.size(0);
    const int64_t material_count = layer_offset.size(0);
    const int64_t layer_total = layer_thickness_m.size(0);

    at::Tensor cos_storage;
    at::Tensor thickness_storage;
    at::Tensor eps_storage;
    at::Tensor sigma_storage;
    const float* t_cos = stack_optional_grad(
        tangent_cos_theta, cos_storage, "tangent_cos_theta", count, cos_theta);
    const float* t_thickness = stack_optional_grad(
        tangent_layer_thickness, thickness_storage, "tangent_layer_thickness_m",
        layer_total, cos_theta);
    const float* t_eps = stack_optional_grad(
        tangent_layer_eps_r, eps_storage, "tangent_layer_eps_r", layer_total,
        cos_theta);
    const float* t_sigma = stack_optional_grad(
        tangent_layer_sigma_e, sigma_storage, "tangent_layer_sigma_e",
        layer_total, cos_theta);

    auto options = cos_theta.options();
    at::Tensor outputs[12];
    StackTangentPtrs tangents{};
    for (int field = 0; field < 12; ++field) {
        outputs[field] = at::empty({count}, options);
        tangents.p[field] = outputs[field].data_ptr<float>();
    }
    if (count > 0) {
        cudaStream_t stream =
            at::cuda::getCurrentCUDAStream(cos_theta.get_device()).stream();
        em_layer_stack_jvp_kernel<<<launch_blocks(count), kBlockSize, 0, stream>>>(
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
            t_cos,
            t_thickness,
            t_eps,
            t_sigma,
            static_cast<float>(tangent_frequency),
            tangents);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    pybind11::dict out;
    for (int field = 0; field < 12; ++field)
        out[kStackFields[field]] = outputs[field];
    return out;
}
