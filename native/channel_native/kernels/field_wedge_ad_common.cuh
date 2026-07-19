#pragma once

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/util/complex.h>
#include <torch/extension.h>

#include <rayd/shared/rf/field_transport.cuh>
#include <rayd/torch/rf/field_transport_ad.cuh>
#include "../tensor_checks.h"

#include <array>
#include <vector>

namespace {

constexpr int kBlockSize = 128;
namespace field = rayd::shared::utd;
namespace transport = rayd::shared::rf::field_transport;
namespace ad = rayd::torch::rf::field_transport_ad;

using Dual = field::Dual;
using DualV3 = field::Vec3T<Dual>;
using DualCx = field::ComplexT<Dual>;
using DualC3 = field::Complex3T<Dual>;

__device__ __forceinline__ field::float3a load3f(const float* values, int64_t index) {
    const int64_t base = index * 3;
    return field::make_f3(values[base], values[base + 1], values[base + 2]);
}

__device__ __forceinline__ c10::complex<float> to_c10(field::Complex value) {
    return c10::complex<float>(value.re, value.im);
}

__device__ __forceinline__ field::Complex from_c10(c10::complex<float> value) {
    return field::cplx(value.real(), value.imag());
}

int launch_blocks(int64_t count) {
    return static_cast<int>((count + kBlockSize - 1) / kBlockSize);
}

at::Tensor zero_scalar(const at::TensorOptions& options) {
    auto tensor = at::empty({1}, options);
    cudaStream_t stream =
        at::cuda::getCurrentCUDAStream(tensor.get_device()).stream();
    C10_CUDA_CHECK(cudaMemsetAsync(
        tensor.data_ptr(), 0, tensor.element_size(), stream));
    return tensor;
}

const at::Tensor* optional_tensor_arg(
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

template <typename T>
T* opt_mut_ptr(at::Tensor* tensor) {
    return tensor == nullptr ? nullptr : tensor->data_ptr<T>();
}

}  // namespace
