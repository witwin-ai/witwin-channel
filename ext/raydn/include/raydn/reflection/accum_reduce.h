#pragma once

#include <ATen/ATen.h>

namespace raydn {

void reduce_refl_accum_staged_cuda(
    int64_t sample_count,
    const at::Tensor &stage_cell,
    const at::Tensor &stage_value,
    at::Tensor &out_power,
    at::Tensor &out_field_x_re,
    at::Tensor &out_field_x_im,
    at::Tensor &out_field_y_re,
    at::Tensor &out_field_y_im,
    at::Tensor &out_field_z_re,
    at::Tensor &out_field_z_im,
    at::Tensor &out_reflection_count);

} // namespace raydn
