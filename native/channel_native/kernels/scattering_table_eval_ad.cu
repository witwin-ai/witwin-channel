// ADR-015 Part A: native JVP/VJP companions of the resident Kirchhoff BSDF
// table lookup (kernels/scattering.cu::scattering_eval_kernel, facade
// scattering_table_eval). The forward is untouched; these kernels recompute
// every forward interpolation intermediate through the shared
// scattering_table.cuh::eval_te_tm_grad helper (this TU compiles with
// --fmad=false so the in-helper primal recompute rounds exactly like the
// forward) and differentiate the live inputs wi, wo, f_te, f_tm.
//
// Backward: one thread per row (grid-stride loop). The direction gradients
// (grad_wi, grad_wo) are direct stores; the 16 interpolation corners scatter
// into the zero-initialised table gradient buffers with atomicAdd (same
// run-to-run nondeterministic accumulation policy as the ADR-014 ensemble and
// the transmission-layer backward).
//
// JVP: elementwise, tangent-forward, no atomics; a missing tangent is a zero
// tangent. Both companions share the quadrilinear table derivative helper in
// scattering_table.cuh with the forward, so no table math is duplicated here.

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <torch/extension.h>

#include "../tensor_checks.h"
#include "scattering_table.cuh"

namespace {

constexpr int kBlockSize = 256;
namespace st = channel_native::scattering_tables;

int launch_blocks(int64_t count) {
    return static_cast<int>((count + kBlockSize - 1) / kBlockSize);
}

at::Tensor zero_filled(at::IntArrayRef sizes, const at::TensorOptions& options) {
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

// Optional CUDA-tensor argument with the same dtype/shape/device contract as
// the ensemble companion; None -> nullptr (a zero cotangent/tangent).
const at::Tensor* optional_arg(
    pybind11::object value,
    at::Tensor& storage,
    const char* name,
    c10::ScalarType dtype,
    at::IntArrayRef sizes,
    const at::Tensor& reference) {
    if (value.is_none())
        return nullptr;
    storage = value.cast<at::Tensor>().contiguous();
    TORCH_CHECK(storage.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(storage.scalar_type() == dtype, name, " has the wrong dtype");
    TORCH_CHECK(storage.sizes() == sizes, name, " has the wrong shape");
    TORCH_CHECK(
        storage.get_device() == reference.get_device(),
        name, " must share the primal device");
    return &storage;
}

template <typename T>
const T* opt_ptr(const at::Tensor* tensor) {
    return tensor == nullptr ? nullptr : tensor->data_ptr<T>();
}

__global__ void table_eval_backward_kernel(
    int64_t count, int nti, int npi, int nto, int npo,
    const float* __restrict__ wi,
    const float* __restrict__ wo,
    const float* __restrict__ fte,
    const float* __restrict__ ftm,
    const float* __restrict__ grad_out_f_te,
    const float* __restrict__ grad_out_f_tm,
    float* __restrict__ out_grad_wi,
    float* __restrict__ out_grad_wo,
    float* __restrict__ out_grad_fte,
    float* __restrict__ out_grad_ftm,
    bool need_grad_dirs, bool need_grad_tables) {
    for (int64_t row = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         row < count; row += static_cast<int64_t>(blockDim.x) * gridDim.x) {
        st::TableEvalGrad g;
        st::eval_te_tm_grad(
            fte, ftm, nti, npi, nto, npo, wi + row * 3, wo + row * 3, g);

        const float gte = grad_out_f_te != nullptr ? grad_out_f_te[row] : 0.0f;
        const float gtm = grad_out_f_tm != nullptr ? grad_out_f_tm[row] : 0.0f;

        if (need_grad_dirs) {
            // Below the horizon g.active is false and all partials are zero,
            // so the direct stores are exactly zero there.
#pragma unroll
            for (int i = 0; i < 3; ++i) {
                out_grad_wi[row * 3 + i] = gte * g.dte_dwi[i] + gtm * g.dtm_dwi[i];
                out_grad_wo[row * 3 + i] = gte * g.dte_dwo[i] + gtm * g.dtm_dwo[i];
            }
        }
        if (need_grad_tables && g.active) {
            for (int k = 0; k < 16; ++k) {
                atomicAdd(&out_grad_fte[g.idx[k]], gte * g.cw[k]);
                atomicAdd(&out_grad_ftm[g.idx[k]], gtm * g.cw[k]);
            }
        }
    }
}

__global__ void table_eval_jvp_kernel(
    int64_t count, int nti, int npi, int nto, int npo,
    const float* __restrict__ wi,
    const float* __restrict__ wo,
    const float* __restrict__ fte,
    const float* __restrict__ ftm,
    const float* __restrict__ t_wi,
    const float* __restrict__ t_wo,
    const float* __restrict__ t_fte,
    const float* __restrict__ t_ftm,
    float* __restrict__ out_tangent_f_te,
    float* __restrict__ out_tangent_f_tm) {
    for (int64_t row = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         row < count; row += static_cast<int64_t>(blockDim.x) * gridDim.x) {
        st::TableEvalGrad g;
        st::eval_te_tm_grad(
            fte, ftm, nti, npi, nto, npo, wi + row * 3, wo + row * 3, g);

        float tte = 0.0f, ttm = 0.0f;
        if (g.active) {
            // Direction-coordinate chain (missing tangent = zero tangent).
            if (t_wi != nullptr) {
#pragma unroll
                for (int i = 0; i < 3; ++i) {
                    const float twi = t_wi[row * 3 + i];
                    tte += g.dte_dwi[i] * twi;
                    ttm += g.dtm_dwi[i] * twi;
                }
            }
            if (t_wo != nullptr) {
#pragma unroll
                for (int i = 0; i < 3; ++i) {
                    const float two = t_wo[row * 3 + i];
                    tte += g.dte_dwo[i] * two;
                    ttm += g.dtm_dwo[i] * two;
                }
            }
            // Table-value chain over the 16 interpolation corners.
            if (t_fte != nullptr) {
                for (int k = 0; k < 16; ++k)
                    tte += g.cw[k] * t_fte[g.idx[k]];
            }
            if (t_ftm != nullptr) {
                for (int k = 0; k < 16; ++k)
                    ttm += g.cw[k] * t_ftm[g.idx[k]];
            }
        }
        out_tangent_f_te[row] = tte;
        out_tangent_f_tm[row] = ttm;
    }
}

// Validate the four primal tensors of the table eval, returning row/table dims.
void check_table_eval_inputs(
    const at::Tensor& wi, const at::Tensor& wo,
    const at::Tensor& f_te, const at::Tensor& f_tm,
    int64_t& count, int& nti, int& npi, int& nto, int& npo) {
    using channel_native::check_tensor;
    using channel_native::check_vec3_table;
    check_vec3_table(wi, "wi");
    check_vec3_table(wo, "wo");
    count = wi.size(0);
    TORCH_CHECK(wo.size(0) == count, "wi and wo must have matching rows");
    check_tensor(f_te, "f_te", at::kFloat, 4);
    check_tensor(f_tm, "f_tm", at::kFloat, 4);
    TORCH_CHECK(f_te.sizes() == f_tm.sizes(), "f_te and f_tm must share shape");
    TORCH_CHECK(
        wo.get_device() == wi.get_device() &&
            f_te.get_device() == wi.get_device() &&
            f_tm.get_device() == wi.get_device(),
        "table eval tensors must share device");
    nti = static_cast<int>(f_te.size(0));
    npi = static_cast<int>(f_te.size(1));
    nto = static_cast<int>(f_te.size(2));
    npo = static_cast<int>(f_te.size(3));
}

}  // namespace

pybind11::dict cn_scattering_table_eval_backward(
    at::Tensor wi,
    at::Tensor wo,
    at::Tensor f_te,
    at::Tensor f_tm,
    pybind11::object grad_out_f_te,
    pybind11::object grad_out_f_tm,
    bool need_grad_dirs,
    bool need_grad_tables) {
    int64_t count = 0;
    int nti = 0, npi = 0, nto = 0, npo = 0;
    check_table_eval_inputs(wi, wo, f_te, f_tm, count, nti, npi, nto, npo);

    at::Tensor storage[2];
    const at::Tensor* g_te = optional_arg(
        std::move(grad_out_f_te), storage[0], "grad_out_f_te", at::kFloat, {count}, wi);
    const at::Tensor* g_tm = optional_arg(
        std::move(grad_out_f_tm), storage[1], "grad_out_f_tm", at::kFloat, {count}, wi);

    at::Tensor grad_wi, grad_wo, grad_fte, grad_ftm;
    if (need_grad_dirs) {
        grad_wi = at::empty({count, 3}, wi.options());
        grad_wo = at::empty({count, 3}, wo.options());
    }
    if (need_grad_tables) {
        grad_fte = zero_filled(f_te.sizes(), f_te.options());
        grad_ftm = zero_filled(f_tm.sizes(), f_tm.options());
    }

    const bool any_grad = g_te != nullptr || g_tm != nullptr;
    if (count > 0 && any_grad && (need_grad_dirs || need_grad_tables)) {
        cudaStream_t stream =
            at::cuda::getCurrentCUDAStream(wi.get_device()).stream();
        table_eval_backward_kernel<<<launch_blocks(count), kBlockSize, 0, stream>>>(
            count, nti, npi, nto, npo,
            wi.data_ptr<float>(), wo.data_ptr<float>(),
            f_te.data_ptr<float>(), f_tm.data_ptr<float>(),
            opt_ptr<float>(g_te), opt_ptr<float>(g_tm),
            need_grad_dirs ? grad_wi.data_ptr<float>() : nullptr,
            need_grad_dirs ? grad_wo.data_ptr<float>() : nullptr,
            need_grad_tables ? grad_fte.data_ptr<float>() : nullptr,
            need_grad_tables ? grad_ftm.data_ptr<float>() : nullptr,
            need_grad_dirs, need_grad_tables);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    } else if (need_grad_dirs && count > 0 && !any_grad) {
        // No cotangent -> the direct stores are exactly zero.
        grad_wi.zero_();
        grad_wo.zero_();
    }

    pybind11::dict out;
    out["grad_wi"] =
        need_grad_dirs ? pybind11::cast(grad_wi) : pybind11::object(pybind11::none());
    out["grad_wo"] =
        need_grad_dirs ? pybind11::cast(grad_wo) : pybind11::object(pybind11::none());
    out["grad_f_te"] =
        need_grad_tables ? pybind11::cast(grad_fte) : pybind11::object(pybind11::none());
    out["grad_f_tm"] =
        need_grad_tables ? pybind11::cast(grad_ftm) : pybind11::object(pybind11::none());
    return out;
}

pybind11::dict cn_scattering_table_eval_jvp(
    at::Tensor wi,
    at::Tensor wo,
    at::Tensor f_te,
    at::Tensor f_tm,
    pybind11::object t_wi,
    pybind11::object t_wo,
    pybind11::object t_f_te,
    pybind11::object t_f_tm) {
    int64_t count = 0;
    int nti = 0, npi = 0, nto = 0, npo = 0;
    check_table_eval_inputs(wi, wo, f_te, f_tm, count, nti, npi, nto, npo);

    at::Tensor storage[4];
    const at::Tensor* tw_wi = optional_arg(
        std::move(t_wi), storage[0], "t_wi", at::kFloat, {count, 3}, wi);
    const at::Tensor* tw_wo = optional_arg(
        std::move(t_wo), storage[1], "t_wo", at::kFloat, {count, 3}, wi);
    const at::Tensor* tw_fte = optional_arg(
        std::move(t_f_te), storage[2], "t_f_te", at::kFloat, f_te.sizes(), wi);
    const at::Tensor* tw_ftm = optional_arg(
        std::move(t_f_tm), storage[3], "t_f_tm", at::kFloat, f_tm.sizes(), wi);

    auto tangent_f_te = at::empty({count}, f_te.options());
    auto tangent_f_tm = at::empty({count}, f_tm.options());
    if (count > 0) {
        cudaStream_t stream =
            at::cuda::getCurrentCUDAStream(wi.get_device()).stream();
        table_eval_jvp_kernel<<<launch_blocks(count), kBlockSize, 0, stream>>>(
            count, nti, npi, nto, npo,
            wi.data_ptr<float>(), wo.data_ptr<float>(),
            f_te.data_ptr<float>(), f_tm.data_ptr<float>(),
            opt_ptr<float>(tw_wi), opt_ptr<float>(tw_wo),
            opt_ptr<float>(tw_fte), opt_ptr<float>(tw_ftm),
            tangent_f_te.data_ptr<float>(), tangent_f_tm.data_ptr<float>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    pybind11::dict out;
    out["tangent_f_te"] = tangent_f_te;
    out["tangent_f_tm"] = tangent_f_tm;
    return out;
}
