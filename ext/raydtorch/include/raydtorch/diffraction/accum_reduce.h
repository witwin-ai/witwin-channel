#pragma once

#include <ATen/ATen.h>

#include <cstdint>

namespace raydtorch {

void reduce_dfr_accum_staged_cuda(
    int64_t sample_count,
    const at::Tensor &stage_cell,
    const at::Tensor &stage_value,
    at::Tensor &out_power,
    at::Tensor &out_field_x_re,
    at::Tensor &out_direct_count,
    at::Tensor &out_keller_count,
    at::Tensor &out_edge_uses);

void reduce_dfr_coherent_accum_staged_cuda(
    int64_t sample_count,
    int64_t cell_count,
    const at::Tensor &stage_key,
    const at::Tensor &stage_value,
    at::Tensor &out_direct_field_x_re,
    at::Tensor &out_direct_field_x_im,
    at::Tensor &out_direct_field_y_re,
    at::Tensor &out_direct_field_y_im,
    at::Tensor &out_direct_field_z_re,
    at::Tensor &out_direct_field_z_im,
    at::Tensor &out_multi_field_x_re,
    at::Tensor &out_multi_field_x_im,
    at::Tensor &out_multi_field_y_re,
    at::Tensor &out_multi_field_y_im,
    at::Tensor &out_multi_field_z_re,
    at::Tensor &out_multi_field_z_im,
    at::Tensor &out_direct_count,
    at::Tensor &out_multi_count);

} // namespace raydtorch
