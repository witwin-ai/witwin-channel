#pragma once

#include <ATen/ATen.h>

#include <tuple>

namespace raydn {

std::tuple<at::Tensor, at::Tensor> reflection_trace_stats_cuda(
    const at::Tensor &valid,
    const at::Tensor &t);

std::tuple<at::Tensor, at::Tensor> diffraction_path_stats_cuda(
    const at::Tensor &count,
    const at::Tensor &valid,
    const at::Tensor &delay);

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor> default_dfr_material_cuda(
    int64_t count,
    const at::Tensor &like);

at::Tensor intersection_valid_cuda(
    const at::Tensor &t,
    const at::Tensor &shape_id);

} // namespace raydn
