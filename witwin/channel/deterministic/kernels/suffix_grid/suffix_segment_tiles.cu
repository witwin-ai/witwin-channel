#include <suffix_grid/suffix_grid.h>

#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/scan.h>

#include <common/cuda_check.h>
#include <common/grid_ops.h>

namespace witwin::channel::native_ext {

namespace {

using common::ceil_div_int;
using common::throw_cuda;

constexpr float SGT_EPS = 1.0e-8f;

template <bool WriteTasks>
__global__ void suffix_segment_tile_plan_kernel(
    float bound_0_min,
    float bound_0_max,
    float bound_1_min,
    float bound_1_max,
    float cell_size_0,
    float cell_size_1,
    int n_coord_0,
    int n_coord_1,
    int tile_shape_0,
    int tile_shape_1,
    int n_tiles_0,
    int max_steps,
    const float* __restrict__ origin_0,
    const float* __restrict__ origin_1,
    const float* __restrict__ dir_0,
    const float* __restrict__ dir_1,
    const float* __restrict__ blocker_dist,
    const int* __restrict__ active,
    int n_segments,
    const std::uint32_t* __restrict__ segment_offsets,
    std::uint32_t* __restrict__ out_segment_tile_counts,
    std::uint32_t* __restrict__ out_task_segment_idx,
    std::uint32_t* __restrict__ out_task_tile_idx,
    float* __restrict__ out_entry_coord_0,
    float* __restrict__ out_entry_coord_1,
    float* __restrict__ out_entry_t,
    float* __restrict__ out_entry_t_max_0,
    float* __restrict__ out_entry_t_max_1
) {
    int tid = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (tid >= n_segments) {
        return;
    }

    const float ray_origin_0 = origin_0[tid];
    const float ray_origin_1 = origin_1[tid];
    const float ray_dir_0 = dir_0[tid];
    const float ray_dir_1 = dir_1[tid];
    const float ray_blocker = blocker_dist[tid];
    const bool start_in_bounds = (
        ray_origin_0 >= bound_0_min &&
        ray_origin_0 < bound_0_max &&
        ray_origin_1 >= bound_1_min &&
        ray_origin_1 < bound_1_max
    );
    const bool valid = (
        active[tid] != 0 &&
        isfinite(ray_origin_0) &&
        isfinite(ray_origin_1) &&
        isfinite(ray_dir_0) &&
        isfinite(ray_dir_1) &&
        isfinite(ray_blocker) &&
        ray_blocker > 0.0f &&
        start_in_bounds
    );

    if (!valid) {
        if constexpr (!WriteTasks) {
            out_segment_tile_counts[tid] = 0u;
        }
        return;
    }

    const float abs_dir_0 = fmaxf(fabsf(ray_dir_0), SGT_EPS);
    const float abs_dir_1 = fmaxf(fabsf(ray_dir_1), SGT_EPS);
    const float dt_0 = fabsf(cell_size_0 / abs_dir_0);
    const float dt_1 = fabsf(cell_size_1 / abs_dir_1);
    const float step_0 = ray_dir_0 > 0.0f ? cell_size_0 : -cell_size_0;
    const float step_1 = ray_dir_1 > 0.0f ? cell_size_1 : -cell_size_1;
    const float next_0 = ray_dir_0 > 0.0f
        ? (floorf((ray_origin_0 - bound_0_min) / cell_size_0) + 1.0f) * cell_size_0 + bound_0_min
        : floorf((ray_origin_0 - bound_0_min) / cell_size_0) * cell_size_0 + bound_0_min;
    const float next_1 = ray_dir_1 > 0.0f
        ? (floorf((ray_origin_1 - bound_1_min) / cell_size_1) + 1.0f) * cell_size_1 + bound_1_min
        : floorf((ray_origin_1 - bound_1_min) / cell_size_1) * cell_size_1 + bound_1_min;

    float current_coord_0 = ray_origin_0;
    float current_coord_1 = ray_origin_1;
    float current_t = 0.0f;
    float current_t_max_0 = fabsf((next_0 - current_coord_0) / abs_dir_0);
    float current_t_max_1 = fabsf((next_1 - current_coord_1) / abs_dir_1);
    int previous_tile_idx = -1;
    std::uint32_t emitted = 0u;
    std::uint32_t write_offset = WriteTasks ? segment_offsets[tid] : 0u;

    for (int step = 0; step < max_steps; ++step) {
        const bool in_bounds = (
            current_coord_0 >= bound_0_min &&
            current_coord_0 < bound_0_max &&
            current_coord_1 >= bound_1_min &&
            current_coord_1 < bound_1_max &&
            current_t < ray_blocker
        );
        if (!in_bounds) {
            break;
        }

        const int cell_idx_0 = common::clamp_index(
            static_cast<int>(floorf((current_coord_0 - bound_0_min) / cell_size_0)),
            n_coord_0 - 1
        );
        const int cell_idx_1 = common::clamp_index(
            static_cast<int>(floorf((current_coord_1 - bound_1_min) / cell_size_1)),
            n_coord_1 - 1
        );
        const int tile_idx_0 = cell_idx_0 / tile_shape_0;
        const int tile_idx_1 = cell_idx_1 / tile_shape_1;
        const int current_tile_idx = tile_idx_1 * n_tiles_0 + tile_idx_0;

        if (current_tile_idx != previous_tile_idx) {
            if constexpr (WriteTasks) {
                out_task_segment_idx[write_offset] = static_cast<std::uint32_t>(tid);
                out_task_tile_idx[write_offset] = static_cast<std::uint32_t>(current_tile_idx);
                out_entry_coord_0[write_offset] = current_coord_0;
                out_entry_coord_1[write_offset] = current_coord_1;
                out_entry_t[write_offset] = current_t;
                out_entry_t_max_0[write_offset] = current_t_max_0;
                out_entry_t_max_1[write_offset] = current_t_max_1;
                ++write_offset;
            }
            ++emitted;
            previous_tile_idx = current_tile_idx;
        }

        const bool move_0 = current_t_max_0 < current_t_max_1;
        if (move_0) {
            current_t = current_t_max_0;
            current_coord_0 += step_0;
            current_t_max_0 += dt_0;
        } else {
            current_t = current_t_max_1;
            current_coord_1 += step_1;
            current_t_max_1 += dt_1;
        }
    }

    if constexpr (!WriteTasks) {
        out_segment_tile_counts[tid] = emitted;
    }
}

} // namespace

void suffix_segment_tile_plan_count(
    float bound_0_min,
    float bound_0_max,
    float bound_1_min,
    float bound_1_max,
    float cell_size_0,
    float cell_size_1,
    int n_coord_0,
    int n_coord_1,
    int tile_shape_0,
    int tile_shape_1,
    int max_steps,
    const float* origin_0,
    const float* origin_1,
    const float* dir_0,
    const float* dir_1,
    const float* blocker_dist,
    const int* active,
    int n_segments,
    std::uint32_t* out_segment_tile_counts
) {
    if (n_segments <= 0) {
        return;
    }
    const int n_tiles_0 = ceil_div_int(n_coord_0, tile_shape_0);
    constexpr int threads = 128;
    const int blocks = ceil_div_int(n_segments, threads);
    suffix_segment_tile_plan_kernel<false><<<blocks, threads>>>(
        bound_0_min,
        bound_0_max,
        bound_1_min,
        bound_1_max,
        cell_size_0,
        cell_size_1,
        n_coord_0,
        n_coord_1,
        tile_shape_0,
        tile_shape_1,
        n_tiles_0,
        max_steps,
        origin_0,
        origin_1,
        dir_0,
        dir_1,
        blocker_dist,
        active,
        n_segments,
        nullptr,
        out_segment_tile_counts,
        nullptr,
        nullptr,
        nullptr,
        nullptr,
        nullptr,
        nullptr,
        nullptr
    );
    throw_cuda(cudaGetLastError(), "suffix_segment_tile_plan_count kernel launch");
}

std::uint32_t suffix_segment_tile_plan_scan(
    const std::uint32_t* segment_tile_counts,
    int n_segments,
    std::uint32_t* out_segment_offsets
) {
    if (n_segments <= 0) {
        return 0u;
    }

    thrust::exclusive_scan(
        thrust::device,
        thrust::device_pointer_cast(segment_tile_counts),
        thrust::device_pointer_cast(segment_tile_counts) + n_segments,
        thrust::device_pointer_cast(out_segment_offsets)
    );
    throw_cuda(cudaGetLastError(), "suffix_segment_tile_plan_scan thrust::exclusive_scan");

    std::uint32_t last_count = 0u;
    std::uint32_t last_offset = 0u;
    throw_cuda(
        cudaMemcpy(
            &last_count,
            segment_tile_counts + (n_segments - 1),
            sizeof(std::uint32_t),
            cudaMemcpyDeviceToHost
        ),
        "suffix_segment_tile_plan_scan copy last_count"
    );
    throw_cuda(
        cudaMemcpy(
            &last_offset,
            out_segment_offsets + (n_segments - 1),
            sizeof(std::uint32_t),
            cudaMemcpyDeviceToHost
        ),
        "suffix_segment_tile_plan_scan copy last_offset"
    );
    return last_offset + last_count;
}

void suffix_segment_tile_plan_write(
    float bound_0_min,
    float bound_0_max,
    float bound_1_min,
    float bound_1_max,
    float cell_size_0,
    float cell_size_1,
    int n_coord_0,
    int n_coord_1,
    int tile_shape_0,
    int tile_shape_1,
    int max_steps,
    const float* origin_0,
    const float* origin_1,
    const float* dir_0,
    const float* dir_1,
    const float* blocker_dist,
    const int* active,
    int n_segments,
    const std::uint32_t* segment_offsets,
    std::uint32_t* out_task_segment_idx,
    std::uint32_t* out_task_tile_idx,
    float* out_entry_coord_0,
    float* out_entry_coord_1,
    float* out_entry_t,
    float* out_entry_t_max_0,
    float* out_entry_t_max_1
) {
    if (n_segments <= 0) {
        return;
    }
    const int n_tiles_0 = ceil_div_int(n_coord_0, tile_shape_0);
    constexpr int threads = 128;
    const int blocks = ceil_div_int(n_segments, threads);
    suffix_segment_tile_plan_kernel<true><<<blocks, threads>>>(
        bound_0_min,
        bound_0_max,
        bound_1_min,
        bound_1_max,
        cell_size_0,
        cell_size_1,
        n_coord_0,
        n_coord_1,
        tile_shape_0,
        tile_shape_1,
        n_tiles_0,
        max_steps,
        origin_0,
        origin_1,
        dir_0,
        dir_1,
        blocker_dist,
        active,
        n_segments,
        segment_offsets,
        nullptr,
        out_task_segment_idx,
        out_task_tile_idx,
        out_entry_coord_0,
        out_entry_coord_1,
        out_entry_t,
        out_entry_t_max_0,
        out_entry_t_max_1
    );
    throw_cuda(cudaGetLastError(), "suffix_segment_tile_plan_write kernel launch");
}

} // namespace witwin::channel::native_ext
