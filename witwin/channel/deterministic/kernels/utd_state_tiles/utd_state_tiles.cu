#include <utd_state_tiles/utd_state_tiles.h>

#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/scan.h>

#include <common/cuda_check.h>
#include <common/primitives.h>

namespace witwin::channel::native_ext {

namespace {

using common::ceil_div_int;
using common::throw_cuda;

template <bool WriteTasks>
__global__ void utd_state_tile_plan_kernel(
    float support_eps,
    const float* __restrict__ coeff0_0,
    const float* __restrict__ coeff0_1,
    const float* __restrict__ bias0,
    const float* __restrict__ coeff1_0,
    const float* __restrict__ coeff1_1,
    const float* __restrict__ bias1,
    const int* __restrict__ finite_mask,
    const float* __restrict__ tile_coord_0_min,
    const float* __restrict__ tile_coord_0_max,
    const float* __restrict__ tile_coord_1_min,
    const float* __restrict__ tile_coord_1_max,
    int n_states,
    int n_tiles,
    const std::uint32_t* __restrict__ state_offsets,
    std::uint32_t* __restrict__ out_state_tile_counts,
    std::uint32_t* __restrict__ out_task_state_idx,
    std::uint32_t* __restrict__ out_task_tile_idx
) {
    const int tid = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (tid >= n_states) {
        return;
    }

    const float c00 = coeff0_0[tid];
    const float c01 = coeff0_1[tid];
    const float b0 = bias0[tid];
    const float c10 = coeff1_0[tid];
    const float c11 = coeff1_1[tid];
    const float b1 = bias1[tid];
    const bool finite = finite_mask[tid] != 0;

    std::uint32_t supported_tiles = 0u;
    if (finite) {
        for (int tile = 0; tile < n_tiles; ++tile) {
            const float coord0_face0 = c00 >= 0.0f ? tile_coord_0_max[tile] : tile_coord_0_min[tile];
            const float coord1_face0 = c01 >= 0.0f ? tile_coord_1_max[tile] : tile_coord_1_min[tile];
            const float coord0_face1 = c10 >= 0.0f ? tile_coord_0_max[tile] : tile_coord_0_min[tile];
            const float coord1_face1 = c11 >= 0.0f ? tile_coord_1_max[tile] : tile_coord_1_min[tile];
            const float max0 = c00 * coord0_face0 + c01 * coord1_face0 + b0;
            const float max1 = c10 * coord0_face1 + c11 * coord1_face1 + b1;
            if ((max0 >= -support_eps) || (max1 >= -support_eps)) {
                ++supported_tiles;
            }
        }
    }

    const bool emit_all_tiles = (!finite) || (supported_tiles == 0u);
    if constexpr (!WriteTasks) {
        out_state_tile_counts[tid] = emit_all_tiles ? static_cast<std::uint32_t>(n_tiles) : supported_tiles;
        return;
    }

    std::uint32_t write_offset = state_offsets[tid];
    for (int tile = 0; tile < n_tiles; ++tile) {
        bool keep = emit_all_tiles;
        if (!keep) {
            const float coord0_face0 = c00 >= 0.0f ? tile_coord_0_max[tile] : tile_coord_0_min[tile];
            const float coord1_face0 = c01 >= 0.0f ? tile_coord_1_max[tile] : tile_coord_1_min[tile];
            const float coord0_face1 = c10 >= 0.0f ? tile_coord_0_max[tile] : tile_coord_0_min[tile];
            const float coord1_face1 = c11 >= 0.0f ? tile_coord_1_max[tile] : tile_coord_1_min[tile];
            const float max0 = c00 * coord0_face0 + c01 * coord1_face0 + b0;
            const float max1 = c10 * coord0_face1 + c11 * coord1_face1 + b1;
            keep = (max0 >= -support_eps) || (max1 >= -support_eps);
        }
        if (!keep) {
            continue;
        }
        out_task_state_idx[write_offset] = static_cast<std::uint32_t>(tid);
        out_task_tile_idx[write_offset] = static_cast<std::uint32_t>(tile);
        ++write_offset;
    }
}

} // namespace

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
) {
    if (n_states <= 0 || n_tiles <= 0) {
        return;
    }
    constexpr int threads = 128;
    const int blocks = ceil_div_int(n_states, threads);
    utd_state_tile_plan_kernel<false><<<blocks, threads>>>(
        support_eps,
        coeff0_0,
        coeff0_1,
        bias0,
        coeff1_0,
        coeff1_1,
        bias1,
        finite_mask,
        tile_coord_0_min,
        tile_coord_0_max,
        tile_coord_1_min,
        tile_coord_1_max,
        n_states,
        n_tiles,
        nullptr,
        out_state_tile_counts,
        nullptr,
        nullptr
    );
    throw_cuda(cudaGetLastError(), "utd_state_tile_plan_count kernel launch");
}

std::uint32_t utd_state_tile_plan_scan(
    const std::uint32_t* state_tile_counts,
    int n_states,
    std::uint32_t* out_state_offsets
) {
    if (n_states <= 0) {
        return 0u;
    }

    thrust::exclusive_scan(
        thrust::device,
        thrust::device_pointer_cast(state_tile_counts),
        thrust::device_pointer_cast(state_tile_counts) + n_states,
        thrust::device_pointer_cast(out_state_offsets)
    );
    throw_cuda(cudaGetLastError(), "utd_state_tile_plan_scan thrust::exclusive_scan");

    std::uint32_t last_count = 0u;
    std::uint32_t last_offset = 0u;
    throw_cuda(
        cudaMemcpy(
            &last_count,
            state_tile_counts + (n_states - 1),
            sizeof(std::uint32_t),
            cudaMemcpyDeviceToHost
        ),
        "utd_state_tile_plan_scan copy last_count"
    );
    throw_cuda(
        cudaMemcpy(
            &last_offset,
            out_state_offsets + (n_states - 1),
            sizeof(std::uint32_t),
            cudaMemcpyDeviceToHost
        ),
        "utd_state_tile_plan_scan copy last_offset"
    );
    return last_offset + last_count;
}

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
) {
    if (n_states <= 0 || n_tiles <= 0) {
        return;
    }
    constexpr int threads = 128;
    const int blocks = ceil_div_int(n_states, threads);
    utd_state_tile_plan_kernel<true><<<blocks, threads>>>(
        support_eps,
        coeff0_0,
        coeff0_1,
        bias0,
        coeff1_0,
        coeff1_1,
        bias1,
        finite_mask,
        tile_coord_0_min,
        tile_coord_0_max,
        tile_coord_1_min,
        tile_coord_1_max,
        n_states,
        n_tiles,
        state_offsets,
        nullptr,
        out_task_state_idx,
        out_task_tile_idx
    );
    throw_cuda(cudaGetLastError(), "utd_state_tile_plan_write kernel launch");
}

} // namespace witwin::channel::native_ext
