#pragma once

#include <common/primitives.h>

namespace witwin::channel::native_ext {

void build_receiver_tiles(
    int plane_axis,
    float plane_position,
    const float* coord_0,
    const float* coord_1,
    int n_coord_0,
    int n_coord_1,
    int tile_size_0,
    int tile_size_1,
    int* out_tile_i0,
    int* out_tile_i1,
    int* out_tile_extent_0,
    int* out_tile_extent_1,
    float* out_coord_0_min,
    float* out_coord_0_max,
    float* out_coord_1_min,
    float* out_coord_1_max,
    float* out_aabb_min_x,
    float* out_aabb_min_y,
    float* out_aabb_min_z,
    float* out_aabb_max_x,
    float* out_aabb_max_y,
    float* out_aabb_max_z
);

} // namespace witwin::channel::native_ext
