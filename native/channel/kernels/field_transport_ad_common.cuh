// Copyright Xingyu Chen.
// Shares field transport ad common CUDA helpers.

#pragma once

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/util/complex.h>
#include "torch_cuda_minimal.h"

#include <rayd/torch/rf/field_transport_ad.cuh>
#include "../tensor_checks.h"

// Backward / JVP companion kernels for the field transport forwards
// (plan 07 AD-1 materials/frequency, AD-2 geometry). Fixed-topology contract:
// the discrete winner (face sequence, validity, normal flips, polarizations,
// tx_power, material ids) is constant; the differentiable inputs are
// eps_r / sigma_e / gain / thickness (per bounce or CSR layer), the carrier
// frequency, and the continuous hit geometry (source, target,
// interaction_positions, interaction_normals) behind need_grad_geometry.
// path_length_m / delay_s are differentiable outputs of the geometry alone
// (their material/frequency cotangent is exactly zero).

namespace {

constexpr int kBlockSize = 256;
constexpr int kMaxAdDepth = 8;
namespace field = rayd::shared::utd;
namespace em = rayd::shared::rf::em;
namespace transport = rayd::shared::rf::field_transport;
namespace ad = rayd::torch::rf::field_transport_ad;

using ad::DualC;
using ad::adj_dot;
using ad::fold_output_cotangents;
using ad::write_output_tangents;

__device__ __forceinline__ field::float3a load3f(const float* values, int64_t index) {
    const int64_t base = index * 3;
    return field::make_f3(values[base], values[base + 1], values[base + 2]);
}

__device__ __forceinline__ field::float3a load_sequence3f(
    const float* values, int64_t index, int64_t bounce, int64_t depth) {
    const int64_t base = (index * depth + bounce) * 3;
    return field::make_f3(values[base], values[base + 1], values[base + 2]);
}

__device__ __forceinline__ ad::DualF3 load_dual3f(
    const float* values, const float* tangents, int64_t index) {
    return {
        load3f(values, index),
        tangents != nullptr ? load3f(tangents, index) : field::f3_zero()};
}

__device__ __forceinline__ ad::DualF3 load_dual_sequence3f(
    const float* values,
    const float* tangents,
    int64_t index,
    int64_t bounce,
    int64_t depth) {
    return {
        load_sequence3f(values, index, bounce, depth),
        tangents != nullptr ? load_sequence3f(tangents, index, bounce, depth)
                            : field::f3_zero()};
}

__device__ __forceinline__ field::Complex complex_of(c10::complex<float> value) {
    return field::cplx(value.real(), value.imag());
}

__device__ __forceinline__ c10::complex<float> to_c10(field::Complex value) {
    return c10::complex<float>(value.re, value.im);
}

// ---------------------------------------------------------------------------
// Host entries.
// ---------------------------------------------------------------------------

int launch_blocks(int64_t count) {
    return static_cast<int>((count + kBlockSize - 1) / kBlockSize);
}

// Gradient accumulators must start at zero; allocate raw and memset on the
// current stream (same pattern as los.cu) instead of ATen zero-fill.
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

const at::Tensor* optional_grad(
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
const T* grad_ptr(const at::Tensor* tensor) {
    return tensor == nullptr ? nullptr : tensor->data_ptr<T>();
}

}  // namespace
