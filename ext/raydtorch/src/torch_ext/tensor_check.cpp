#include <raydtorch/common/tensor_check.h>

#include <stdexcept>
#include <string>

namespace raydtorch {

namespace {
std::string message(std::string_view name, std::string_view detail) {
    return std::string(name) + " " + std::string(detail);
}
} // namespace

void require_cuda(const at::Tensor &tensor, std::string_view name) {
    if (!tensor.is_cuda())
        throw std::runtime_error(message(name, "must be CUDA."));
}

void require_contiguous(const at::Tensor &tensor, std::string_view name) {
    if (!tensor.is_contiguous())
        throw std::runtime_error(message(name, "must be contiguous."));
}

void require_dtype(const at::Tensor &tensor, at::ScalarType dtype, std::string_view name) {
    if (tensor.scalar_type() != dtype)
        throw std::runtime_error(message(name, "has the wrong dtype."));
}

void require_rank(const at::Tensor &tensor, int64_t rank, std::string_view name) {
    if (tensor.dim() != rank)
        throw std::runtime_error(message(name, "has the wrong rank."));
}

void require_last_dim(const at::Tensor &tensor, int64_t last_dim, std::string_view name) {
    if (tensor.dim() == 0 || tensor.size(tensor.dim() - 1) != last_dim)
        throw std::runtime_error(message(name, "has the wrong last dimension."));
}

void require_vec3f(const at::Tensor &tensor, std::string_view name) {
    require_cuda(tensor, name);
    require_contiguous(tensor, name);
    require_dtype(tensor, at::kFloat, name);
    require_rank(tensor, 2, name);
    require_last_dim(tensor, 3, name);
}

void require_vec2f(const at::Tensor &tensor, std::string_view name) {
    require_cuda(tensor, name);
    require_contiguous(tensor, name);
    require_dtype(tensor, at::kFloat, name);
    require_rank(tensor, 2, name);
    require_last_dim(tensor, 2, name);
}

void require_vec3i(const at::Tensor &tensor, std::string_view name) {
    require_cuda(tensor, name);
    require_contiguous(tensor, name);
    require_dtype(tensor, at::kInt, name);
    require_rank(tensor, 2, name);
    require_last_dim(tensor, 3, name);
}

void require_scalar_f(const at::Tensor &tensor, std::string_view name) {
    require_cuda(tensor, name);
    require_contiguous(tensor, name);
    require_dtype(tensor, at::kFloat, name);
    require_rank(tensor, 1, name);
}

void require_mask(const at::Tensor &tensor, std::string_view name) {
    require_cuda(tensor, name);
    require_contiguous(tensor, name);
    require_dtype(tensor, at::kBool, name);
    require_rank(tensor, 1, name);
}

} // namespace raydtorch
