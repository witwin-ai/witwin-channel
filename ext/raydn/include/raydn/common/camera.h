#pragma once

#include <ATen/ATen.h>

#include <tuple>

namespace raydn {

at::Tensor camera_sample_to_world_cuda(
    const at::Tensor &sample,
    double tan_x,
    double tan_y,
    double depth);

at::Tensor camera_sample_to_world_backward_cuda(
    const at::Tensor &grad_world,
    int64_t sample_count,
    double tan_x,
    double tan_y,
    double depth);

at::Tensor camera_world_to_sample_cuda(
    const at::Tensor &point,
    double tan_x,
    double tan_y);

at::Tensor camera_world_to_sample_backward_cuda(
    const at::Tensor &point,
    const at::Tensor &grad_sample,
    double tan_x,
    double tan_y);

std::tuple<at::Tensor, at::Tensor> camera_sample_ray_cuda(
    const at::Tensor &sample,
    double tan_x,
    double tan_y);

at::Tensor camera_sample_ray_backward_cuda(
    const at::Tensor &sample,
    const at::Tensor *grad_direction,
    double tan_x,
    double tan_y);

} // namespace raydn
