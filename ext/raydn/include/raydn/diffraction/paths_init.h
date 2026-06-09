#pragma once

#include <ATen/ATen.h>

#include <cstdint>

namespace raydn {

void init_dfr_path_outputs_cuda(
    int64_t capacity,
    at::Tensor &out_count,
    at::Tensor &out_valid,
    at::Tensor &out_tx_id,
    at::Tensor &out_rx_id,
    at::Tensor &out_order,
    at::Tensor &out_edge0,
    at::Tensor &out_edge1,
    at::Tensor &out_edge2,
    at::Tensor &out_delay,
    at::Tensor &out_field_x_re,
    at::Tensor &out_field_x_im,
    at::Tensor &out_field_y_re,
    at::Tensor &out_field_y_im,
    at::Tensor &out_field_z_re,
    at::Tensor &out_field_z_im,
    at::Tensor &out_p0,
    at::Tensor &out_p1,
    at::Tensor &out_p2);

} // namespace raydn
