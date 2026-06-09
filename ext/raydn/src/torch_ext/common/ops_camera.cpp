#include <raydn/common/camera.h>
#include <raydn/common/tensor_check.h>

#include <torch/extension.h>

#include <stdexcept>
#include <string>

namespace raydn {

namespace {

void require_sample(const at::Tensor &sample, const char *name) {
    require_cuda(sample, name);
    require_dtype(sample, at::kFloat, name);
    require_rank(sample, 2, name);
    require_last_dim(sample, 2, name);
}

void require_point(const at::Tensor &point, const char *name) {
    require_cuda(point, name);
    require_dtype(point, at::kFloat, name);
    require_rank(point, 2, name);
    require_last_dim(point, 3, name);
}

void require_grad_or_empty(const at::Tensor &grad, int64_t rows, int64_t cols, const char *name) {
    require_cuda(grad, name);
    require_dtype(grad, at::kFloat, name);
    if (grad.numel() == 0)
        return;
    require_rank(grad, 2, name);
    if (grad.size(0) != rows || grad.size(1) != cols)
        throw std::runtime_error(std::string(name) + " has the wrong shape.");
}

void require_optional_grad(
    const c10::optional<at::Tensor> &grad,
    int64_t rows,
    int64_t cols,
    const char *name) {
    if (!grad.has_value())
        return;
    require_grad_or_empty(*grad, rows, cols, name);
}

} // namespace

at::Tensor camera_sample_to_world_op(
    at::Tensor sample,
    double tan_x,
    double tan_y,
    double depth) {
    require_sample(sample, "sample");
    return camera_sample_to_world_cuda(sample, tan_x, tan_y, depth);
}

at::Tensor camera_sample_to_world_backward_op(
    at::Tensor grad_world,
    int64_t sample_count,
    double tan_x,
    double tan_y,
    double depth) {
    if (sample_count < 0)
        throw std::runtime_error("sample_count must be non-negative.");
    require_grad_or_empty(grad_world, sample_count, 3, "grad_world");
    if (grad_world.numel() == 0)
        return at::empty({sample_count, 2}, grad_world.options());
    return camera_sample_to_world_backward_cuda(grad_world, sample_count, tan_x, tan_y, depth);
}

at::Tensor camera_world_to_sample_op(
    at::Tensor point,
    double tan_x,
    double tan_y) {
    require_point(point, "point");
    return camera_world_to_sample_cuda(point, tan_x, tan_y);
}

at::Tensor camera_world_to_sample_backward_op(
    at::Tensor point,
    at::Tensor grad_sample,
    double tan_x,
    double tan_y) {
    require_point(point, "point");
    require_grad_or_empty(grad_sample, point.size(0), 2, "grad_sample");
    if (grad_sample.numel() == 0)
        return at::empty({point.size(0), 3}, point.options());
    return camera_world_to_sample_backward_cuda(point, grad_sample, tan_x, tan_y);
}

py::tuple camera_sample_ray_op(
    at::Tensor sample,
    double tan_x,
    double tan_y) {
    require_sample(sample, "sample");
    auto [origin, direction] = camera_sample_ray_cuda(sample, tan_x, tan_y);
    return py::make_tuple(origin, direction);
}

at::Tensor camera_sample_ray_backward_op(
    at::Tensor sample,
    c10::optional<at::Tensor> grad_direction,
    double tan_x,
    double tan_y) {
    require_sample(sample, "sample");
    require_optional_grad(grad_direction, sample.size(0), 3, "grad_direction");
    return camera_sample_ray_backward_cuda(
        sample,
        grad_direction.has_value() ? &(*grad_direction) : nullptr,
        tan_x,
        tan_y);
}

} // namespace raydn
