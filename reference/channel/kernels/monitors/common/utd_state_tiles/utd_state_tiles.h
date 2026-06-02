#pragma once

#include <cstdint>

namespace witwin::channel::native_ext {

void utd_state_tile_plan_count(
    float support_eps,
    const float* coeff0_0,
    const float* coeff0_1,
    const float* bias0,
    const float* coeff1_0,
    const float* coeff1_1,
    const float* bias1,
    const int* finite_mask,
    const float* tile_coord_0_min,
    const float* tile_coord_0_max,
    const float* tile_coord_1_min,
    const float* tile_coord_1_max,
    int n_states,
    int n_tiles,
    std::uint32_t* out_state_tile_counts
);

std::uint32_t utd_state_tile_plan_scan(
    const std::uint32_t* state_tile_counts,
    int n_states,
    std::uint32_t* out_state_offsets
);

void utd_state_tile_plan_write(
    float support_eps,
    const float* coeff0_0,
    const float* coeff0_1,
    const float* bias0,
    const float* coeff1_0,
    const float* coeff1_1,
    const float* bias1,
    const int* finite_mask,
    const float* tile_coord_0_min,
    const float* tile_coord_0_max,
    const float* tile_coord_1_min,
    const float* tile_coord_1_max,
    int n_states,
    int n_tiles,
    const std::uint32_t* state_offsets,
    std::uint32_t* out_task_state_idx,
    std::uint32_t* out_task_tile_idx
);

} // namespace witwin::channel::native_ext
