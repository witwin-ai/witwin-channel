// Copyright Xingyu Chen.
// Includes the minimal Torch and CUDA declarations used by native kernels.

#pragma once

// CUDA translation units need ATen and PyTorch's pybind tensor caster, not the
// full C++ frontend pulled in by torch/extension.h.
#include <ATen/ATen.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime_api.h>
#include <torch/csrc/utils/pybind.h>

namespace channel {

inline at::Tensor empty_zero_cuda(
    at::IntArrayRef sizes,
    const at::TensorOptions& options,
    cudaStream_t stream) {
    auto tensor = at::empty(sizes, options);
    if (tensor.numel() > 0) {
        C10_CUDA_CHECK(cudaMemsetAsync(
            tensor.data_ptr(),
            0,
            static_cast<size_t>(tensor.numel()) * tensor.element_size(),
            stream));
    }
    return tensor;
}

}  // namespace channel
