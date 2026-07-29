// Copyright Xingyu Chen.
// Declares tensor checks native contracts.

#pragma once

#include "kernels/torch_cuda_minimal.h"

namespace channel {

inline void check_tensor(
    const at::Tensor& tensor,
    const char* name,
    c10::ScalarType dtype,
    int64_t rank) {
    TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(tensor.scalar_type() == dtype, name, " has the wrong dtype");
    TORCH_CHECK(tensor.dim() == rank, name, " has the wrong rank");
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

inline void check_vec3_table(const at::Tensor& tensor, const char* name) {
    check_tensor(tensor, name, at::kFloat, 2);
    TORCH_CHECK(tensor.size(1) == 3, name, " must have shape (N, 3)");
}

inline void check_flat_tensor(
    const at::Tensor& tensor,
    const char* name,
    c10::ScalarType dtype) {
    check_tensor(tensor, name, dtype, 1);
}

}  // namespace channel
