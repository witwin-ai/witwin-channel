#pragma once

#include <ATen/ATen.h>

namespace raydn {

void diffraction_discover_edges_cuda(
    const at::Tensor &tx_pos,
    const at::Tensor &ray_dir,
    const at::Tensor &prim_index,
    const at::Tensor &hit_p,
    const at::Tensor &hit_n,
    const at::Tensor &hit_geo_n,
    const at::Tensor &triangle_edge_count,
    const at::Tensor &triangle_edge_indices,
    const at::Tensor &edge_pos,
    const at::Tensor &edge_dir,
    const at::Tensor &edge_n0,
    const at::Tensor &edge_nn,
    const at::Tensor &edge_line_min,
    const at::Tensor &edge_line_max,
    const at::Tensor &edge_adjacent_face1,
    at::Tensor &out_seen_edge_mask);

} // namespace raydn
