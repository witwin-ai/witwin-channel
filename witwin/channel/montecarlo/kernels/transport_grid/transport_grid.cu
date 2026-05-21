#include <cuda_runtime.h>

#include <common/cuda_check.h>
#include <transport_grid/transport_grid.h>

#include <cmath>

namespace witwin::channel::native_ext {
namespace {

using common::throw_cuda;

constexpr int BLOCK_SIZE = 256;

__device__ __forceinline__ bool axis_support(
    float coord,
    float coord_min,
    float cell_size,
    int n_cells,
    int &base_idx,
    int &next_idx,
    float &base_weight,
    float &next_weight
) {
    float scaled = (coord - coord_min) / cell_size - 0.5f;
    float base_float = floorf(scaled);
    base_idx = static_cast<int>(base_float);
    next_idx = base_idx + 1;
    float frac = scaled - base_float;
    base_weight = 1.0f - frac;
    next_weight = frac;
    return true;
}

__device__ __forceinline__ void accumulate_weighted_neighbors(
    int x0,
    int x1,
    int y0,
    int y1,
    float wx0,
    float wx1,
    float wy0,
    float wy1,
    float value,
    int n_coord_0,
    int n_coord_1,
    float *out_grid
) {
    const int xs[2] = {x0, x1};
    const int ys[2] = {y0, y1};
    const float wx[2] = {wx0, wx1};
    const float wy[2] = {wy0, wy1};
    for (int yi = 0; yi < 2; ++yi) {
        if (ys[yi] < 0 || ys[yi] >= n_coord_1) {
            continue;
        }
        for (int xi = 0; xi < 2; ++xi) {
            if (xs[xi] < 0 || xs[xi] >= n_coord_0) {
                continue;
            }
            float weight = wx[xi] * wy[yi];
            if (weight == 0.0f) {
                continue;
            }
            atomicAdd(&out_grid[ys[yi] * n_coord_0 + xs[xi]], value * weight);
        }
    }
}

__global__ void transport_grid_forward_kernel(
    const float *coord_0,
    const float *coord_1,
    const float *power,
    const int *active_mask,
    float *out_grid,
    int n_samples,
    float coord_0_min,
    float coord_1_min,
    float cell_size_0,
    float cell_size_1,
    int n_coord_0,
    int n_coord_1
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_samples || active_mask[tid] == 0) {
        return;
    }

    int x0 = 0;
    int x1 = 0;
    int y0 = 0;
    int y1 = 0;
    float wx0 = 0.0f;
    float wx1 = 0.0f;
    float wy0 = 0.0f;
    float wy1 = 0.0f;
    axis_support(coord_0[tid], coord_0_min, cell_size_0, n_coord_0, x0, x1, wx0, wx1);
    axis_support(coord_1[tid], coord_1_min, cell_size_1, n_coord_1, y0, y1, wy0, wy1);
    accumulate_weighted_neighbors(
        x0, x1, y0, y1, wx0, wx1, wy0, wy1,
        power[tid],
        n_coord_0,
        n_coord_1,
        out_grid
    );
}

__global__ void transport_grid_jvp_kernel(
    const float *coord_0,
    const float *coord_1,
    const float *power,
    const int *active_mask,
    const float *t_coord_0,
    const float *t_coord_1,
    const float *t_power,
    float *out_grid,
    int n_samples,
    float coord_0_min,
    float coord_1_min,
    float cell_size_0,
    float cell_size_1,
    int n_coord_0,
    int n_coord_1
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_samples || active_mask[tid] == 0) {
        return;
    }

    int x0 = 0;
    int x1 = 0;
    int y0 = 0;
    int y1 = 0;
    float wx0 = 0.0f;
    float wx1 = 0.0f;
    float wy0 = 0.0f;
    float wy1 = 0.0f;
    axis_support(coord_0[tid], coord_0_min, cell_size_0, n_coord_0, x0, x1, wx0, wx1);
    axis_support(coord_1[tid], coord_1_min, cell_size_1, n_coord_1, y0, y1, wy0, wy1);

    const float inv_cell_0 = 1.0f / cell_size_0;
    const float inv_cell_1 = 1.0f / cell_size_1;
    const float d_wx0 = -inv_cell_0 * t_coord_0[tid];
    const float d_wx1 = inv_cell_0 * t_coord_0[tid];
    const float d_wy0 = -inv_cell_1 * t_coord_1[tid];
    const float d_wy1 = inv_cell_1 * t_coord_1[tid];

    const int xs[2] = {x0, x1};
    const int ys[2] = {y0, y1};
    const float wx[2] = {wx0, wx1};
    const float wy[2] = {wy0, wy1};
    const float d_wx[2] = {d_wx0, d_wx1};
    const float d_wy[2] = {d_wy0, d_wy1};
    for (int yi = 0; yi < 2; ++yi) {
        if (ys[yi] < 0 || ys[yi] >= n_coord_1) {
            continue;
        }
        for (int xi = 0; xi < 2; ++xi) {
            if (xs[xi] < 0 || xs[xi] >= n_coord_0) {
                continue;
            }
            float weight = wx[xi] * wy[yi];
            float d_weight = d_wx[xi] * wy[yi] + wx[xi] * d_wy[yi];
            float jvp_value = t_power[tid] * weight + power[tid] * d_weight;
            if (jvp_value == 0.0f) {
                continue;
            }
            atomicAdd(&out_grid[ys[yi] * n_coord_0 + xs[xi]], jvp_value);
        }
    }
}

__global__ void transport_grid_backward_kernel(
    const float *coord_0,
    const float *coord_1,
    const float *power,
    const int *active_mask,
    const float *upstream_grid,
    float *grad_coord_0,
    float *grad_coord_1,
    float *grad_power,
    int n_samples,
    float coord_0_min,
    float coord_1_min,
    float cell_size_0,
    float cell_size_1,
    int n_coord_0,
    int n_coord_1
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_samples || active_mask[tid] == 0) {
        return;
    }

    int x0 = 0;
    int x1 = 0;
    int y0 = 0;
    int y1 = 0;
    float wx0 = 0.0f;
    float wx1 = 0.0f;
    float wy0 = 0.0f;
    float wy1 = 0.0f;
    axis_support(coord_0[tid], coord_0_min, cell_size_0, n_coord_0, x0, x1, wx0, wx1);
    axis_support(coord_1[tid], coord_1_min, cell_size_1, n_coord_1, y0, y1, wy0, wy1);

    const float inv_cell_0 = 1.0f / cell_size_0;
    const float inv_cell_1 = 1.0f / cell_size_1;

    float local_grad_coord_0 = 0.0f;
    float local_grad_coord_1 = 0.0f;
    float local_grad_power = 0.0f;

    const int xs[2] = {x0, x1};
    const int ys[2] = {y0, y1};
    const float wx[2] = {wx0, wx1};
    const float wy[2] = {wy0, wy1};
    const float d_wx[2] = {-inv_cell_0, inv_cell_0};
    const float d_wy[2] = {-inv_cell_1, inv_cell_1};
    for (int yi = 0; yi < 2; ++yi) {
        if (ys[yi] < 0 || ys[yi] >= n_coord_1) {
            continue;
        }
        for (int xi = 0; xi < 2; ++xi) {
            if (xs[xi] < 0 || xs[xi] >= n_coord_0) {
                continue;
            }
            float upstream = upstream_grid[ys[yi] * n_coord_0 + xs[xi]];
            float weight = wx[xi] * wy[yi];
            local_grad_power += upstream * weight;
            local_grad_coord_0 += upstream * power[tid] * d_wx[xi] * wy[yi];
            local_grad_coord_1 += upstream * power[tid] * wx[xi] * d_wy[yi];
        }
    }

    grad_coord_0[tid] = local_grad_coord_0;
    grad_coord_1[tid] = local_grad_coord_1;
    grad_power[tid] = local_grad_power;
}

} // namespace

void monte_carlo_transport_grid_forward(
    const float *coord_0,
    const float *coord_1,
    const float *power,
    const int *active_mask,
    float *out_grid,
    int n_samples,
    float coord_0_min,
    float coord_1_min,
    float cell_size_0,
    float cell_size_1,
    int n_coord_0,
    int n_coord_1
) {
    if (n_samples <= 0) {
        return;
    }
    int grid_size = (n_samples + BLOCK_SIZE - 1) / BLOCK_SIZE;
    transport_grid_forward_kernel<<<grid_size, BLOCK_SIZE>>>(
        coord_0,
        coord_1,
        power,
        active_mask,
        out_grid,
        n_samples,
        coord_0_min,
        coord_1_min,
        cell_size_0,
        cell_size_1,
        n_coord_0,
        n_coord_1
    );
    throw_cuda(cudaGetLastError(), "transport_grid_forward_kernel launch");
}

void monte_carlo_transport_grid_jvp(
    const float *coord_0,
    const float *coord_1,
    const float *power,
    const int *active_mask,
    const float *t_coord_0,
    const float *t_coord_1,
    const float *t_power,
    float *out_grid,
    int n_samples,
    float coord_0_min,
    float coord_1_min,
    float cell_size_0,
    float cell_size_1,
    int n_coord_0,
    int n_coord_1
) {
    if (n_samples <= 0) {
        return;
    }
    int grid_size = (n_samples + BLOCK_SIZE - 1) / BLOCK_SIZE;
    transport_grid_jvp_kernel<<<grid_size, BLOCK_SIZE>>>(
        coord_0,
        coord_1,
        power,
        active_mask,
        t_coord_0,
        t_coord_1,
        t_power,
        out_grid,
        n_samples,
        coord_0_min,
        coord_1_min,
        cell_size_0,
        cell_size_1,
        n_coord_0,
        n_coord_1
    );
    throw_cuda(cudaGetLastError(), "transport_grid_jvp_kernel launch");
}

void monte_carlo_transport_grid_backward(
    const float *coord_0,
    const float *coord_1,
    const float *power,
    const int *active_mask,
    const float *upstream_grid,
    float *grad_coord_0,
    float *grad_coord_1,
    float *grad_power,
    int n_samples,
    float coord_0_min,
    float coord_1_min,
    float cell_size_0,
    float cell_size_1,
    int n_coord_0,
    int n_coord_1
) {
    if (n_samples <= 0) {
        return;
    }
    int grid_size = (n_samples + BLOCK_SIZE - 1) / BLOCK_SIZE;
    transport_grid_backward_kernel<<<grid_size, BLOCK_SIZE>>>(
        coord_0,
        coord_1,
        power,
        active_mask,
        upstream_grid,
        grad_coord_0,
        grad_coord_1,
        grad_power,
        n_samples,
        coord_0_min,
        coord_1_min,
        cell_size_0,
        cell_size_1,
        n_coord_0,
        n_coord_1
    );
    throw_cuda(cudaGetLastError(), "transport_grid_backward_kernel launch");
}

} // namespace witwin::channel::native_ext
