// Copyright Xingyu Chen.
// Shares bdpt connect common CUDA helpers.

#pragma once

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime_api.h>
#include "torch_cuda_minimal.h"

#include <rayd/shared/rf/field_transport.cuh>

#include <algorithm>
#include <cmath>
#include <tuple>
#include <utility>
#include <vector>

namespace {

namespace utd = rayd::shared::utd;
namespace transport = rayd::shared::rf::field_transport;

constexpr float kLightSpeedMPerS = 299792458.0f;
constexpr float kPi = 3.14159265358979323846f;

// BDPT component_mask bits (component classification): 1=los, 2=reflection,
// 4=diffraction, 8=transmission, 16=scattering.
constexpr int kMaskReflection = 2;
constexpr int kMaskDiffraction = 4;
constexpr int kMaskTransmission = 8;
constexpr int kMaskScattering = 16;
// Connection-sample component ids follow core/path_topology.py:
// 0=los, 1=reflection, 2=diffraction, 5=transmission, 6=scattering.
constexpr int kComponentLos = 0;
constexpr int kComponentReflection = 1;
constexpr int kComponentDiffraction = 2;
constexpr int kComponentTransmission = 5;
constexpr int kComponentScattering = 6;

// Collapse a per-path component mask to the EXCLUSIVE path_class with the
// contract priority scattering > diffraction > transmission > reflection >
// los, so mixed paths are never double counted across component buckets.
__device__ int bdpt_component_from_mask(int mask) {
    if (mask & kMaskScattering) {
        return kComponentScattering;
    }
    if (mask & kMaskDiffraction) {
        return kComponentDiffraction;
    }
    if (mask & kMaskTransmission) {
        return kComponentTransmission;
    }
    if (mask & kMaskReflection) {
        return kComponentReflection;
    }
    return kComponentLos;
}

__device__ bool bdpt_component_accumulable(int component) {
    return component == kComponentLos || component == kComponentReflection ||
        component == kComponentDiffraction || component == kComponentTransmission ||
        component == kComponentScattering;
}

void check_float_cuda(const at::Tensor& tensor, const char* name, int64_t dim) {
    TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(tensor.scalar_type() == at::kFloat, name, " must be float32");
    TORCH_CHECK(tensor.dim() == dim, name, " has wrong rank");
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

void check_int_cuda(const at::Tensor& tensor, const char* name, int64_t dim) {
    TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(tensor.scalar_type() == at::kInt, name, " must be int32");
    TORCH_CHECK(tensor.dim() == dim, name, " has wrong rank");
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

void check_bool_cuda(const at::Tensor& tensor, const char* name, int64_t dim) {
    TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(tensor.scalar_type() == at::kBool, name, " must be bool");
    TORCH_CHECK(tensor.dim() == dim, name, " has wrong rank");
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

void check_vec3_cuda(const at::Tensor& tensor, const char* name) {
    check_float_cuda(tensor, name, 2);
    TORCH_CHECK(tensor.size(1) == 3, name, " must have shape (N, 3)");
}

void check_same_device(const at::Tensor& tensor, const at::Tensor& reference, const char* name) {
    TORCH_CHECK(tensor.get_device() == reference.get_device(), name, " must share device");
}

void check_mis_args(int64_t mode_id, int64_t strategy_count) {
    TORCH_CHECK(mode_id >= 0 && mode_id <= 2, "mode_id must be 0, 1, or 2");
    TORCH_CHECK(strategy_count > 0, "strategy_count must be positive");
}

__device__ float bdpt_connection_mis_weight_from_sums(
    float pdf,
    float balance_pdf_sum,
    float power_pdf_sum,
    int mode_id,
    float beta) {
    if (pdf <= 0.0f) {
        return 0.0f;
    }
    if (mode_id == 0) {
        return 1.0f;
    }
    if (mode_id == 1) {
        return pdf / fmaxf(balance_pdf_sum, 1.17549435e-38f);
    }
    return powf(pdf, beta) / fmaxf(power_pdf_sum, 1.17549435e-38f);
}

__device__ float bdpt_single_strategy_mis_weight(float pdf, int mode_id, float beta) {
    return bdpt_connection_mis_weight_from_sums(pdf, pdf, powf(pdf, beta), mode_id, beta);
}

__device__ float bdpt_free_space_gain(float tx_power, float distance, float frequency_hz) {
    const float wavelength = kLightSpeedMPerS / fmaxf(frequency_hz, 1.0f);
    const float denom = 4.0f * kPi * fmaxf(distance, 1.0e-6f) / wavelength;
    return tx_power / fmaxf(denom * denom, 1.0e-30f);
}



std::tuple<
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor>
allocate_connection_samples(const at::Tensor& reference, int64_t count) {
    auto int_options = reference.options().dtype(at::kInt);
    auto float_options = reference.options().dtype(at::kFloat);
    auto bool_options = reference.options().dtype(at::kBool);
    return {
        at::empty({count, 4}, int_options),
        at::empty({count}, float_options),
        at::empty({count}, float_options),
        at::empty({count}, float_options),
        at::empty({count}, int_options),
        at::empty({count}, bool_options),
        at::empty({count}, int_options),
        at::empty({count}, int_options),
        at::empty({count}, int_options),
        at::empty({count}, int_options),
        at::empty({count}, int_options),
        at::empty({count}, float_options),
    };
}

void zero_double_tensor(at::Tensor tensor) {
    if (tensor.numel() == 0) {
        return;
    }
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(tensor.get_device()).stream();
    C10_CUDA_CHECK(cudaMemsetAsync(tensor.data_ptr<double>(), 0, tensor.numel() * sizeof(double), stream));
}

void zero_int_tensor(at::Tensor tensor) {
    if (tensor.numel() == 0) {
        return;
    }
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(tensor.get_device()).stream();
    C10_CUDA_CHECK(cudaMemsetAsync(tensor.data_ptr<int>(), 0, tensor.numel() * sizeof(int), stream));
}

void zero_float_tensor(at::Tensor tensor) {
    if (tensor.numel() == 0) {
        return;
    }
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(tensor.get_device()).stream();
    C10_CUDA_CHECK(cudaMemsetAsync(tensor.data_ptr<float>(), 0, tensor.numel() * sizeof(float), stream));
}

}  // namespace
