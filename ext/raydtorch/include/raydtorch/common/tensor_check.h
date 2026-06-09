#pragma once

#include <ATen/ATen.h>
#include <string_view>

namespace raydtorch {

void require_cuda(const at::Tensor &tensor, std::string_view name);
void require_contiguous(const at::Tensor &tensor, std::string_view name);
void require_dtype(const at::Tensor &tensor, at::ScalarType dtype, std::string_view name);
void require_rank(const at::Tensor &tensor, int64_t rank, std::string_view name);
void require_last_dim(const at::Tensor &tensor, int64_t last_dim, std::string_view name);

void require_vec3f(const at::Tensor &tensor, std::string_view name);
void require_vec2f(const at::Tensor &tensor, std::string_view name);
void require_vec3i(const at::Tensor &tensor, std::string_view name);
void require_scalar_f(const at::Tensor &tensor, std::string_view name);
void require_mask(const at::Tensor &tensor, std::string_view name);

} // namespace raydtorch
