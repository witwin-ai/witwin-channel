#include <cuda_runtime.h>

#include <common/cuda_check.h>
#include <common/primitives.h>
#include <transport_vertex/transport_vertex.h>

#include <cmath>

namespace witwin::channel::native_ext {
namespace {

using common::ceil_div_int;
using common::throw_cuda;

constexpr int kBlockSize = 256;

__device__ __forceinline__ void axis_support(
    float coord,
    float coord_min,
    float cell_size,
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
}

__global__ void monte_carlo_transport_vertex_jvp_into_kernel(
    const float *coord_0,
    const float *coord_1,
    const float *power,
    const int *active_mask,
    const int *vertex_indices,
    const float *coord_0_coeff_x,
    const float *coord_0_coeff_y,
    const float *coord_0_coeff_z,
    const float *coord_1_coeff_x,
    const float *coord_1_coeff_y,
    const float *coord_1_coeff_z,
    int vertex_slot_count,
    const float *vertex_tangent_x,
    const float *vertex_tangent_y,
    const float *vertex_tangent_z,
    float *out_grid,
    int n_samples,
    float coord_0_min,
    float coord_1_min,
    float cell_size_0,
    float cell_size_1,
    int n_coord_0,
    int n_coord_1
) {
    int sample_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (sample_idx >= n_samples || active_mask[sample_idx] == 0) {
        return;
    }

    float t_coord_0 = 0.0f;
    float t_coord_1 = 0.0f;
    for (int slot = 0; slot < vertex_slot_count; ++slot) {
        int flat_idx = slot * n_samples + sample_idx;
        int global_idx = vertex_indices[flat_idx];
        if (global_idx < 0) {
            continue;
        }
        float tangent_x = vertex_tangent_x[global_idx];
        float tangent_y = vertex_tangent_y[global_idx];
        float tangent_z = vertex_tangent_z[global_idx];
        t_coord_0 += coord_0_coeff_x[flat_idx] * tangent_x
            + coord_0_coeff_y[flat_idx] * tangent_y
            + coord_0_coeff_z[flat_idx] * tangent_z;
        t_coord_1 += coord_1_coeff_x[flat_idx] * tangent_x
            + coord_1_coeff_y[flat_idx] * tangent_y
            + coord_1_coeff_z[flat_idx] * tangent_z;
    }

    if (t_coord_0 == 0.0f && t_coord_1 == 0.0f) {
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
    axis_support(coord_0[sample_idx], coord_0_min, cell_size_0, x0, x1, wx0, wx1);
    axis_support(coord_1[sample_idx], coord_1_min, cell_size_1, y0, y1, wy0, wy1);

    const float inv_cell_0 = 1.0f / cell_size_0;
    const float inv_cell_1 = 1.0f / cell_size_1;
    const float d_wx[2] = {-inv_cell_0 * t_coord_0, inv_cell_0 * t_coord_0};
    const float d_wy[2] = {-inv_cell_1 * t_coord_1, inv_cell_1 * t_coord_1};
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
            float d_weight = d_wx[xi] * wy[yi] + wx[xi] * d_wy[yi];
            float jvp_value = power[sample_idx] * d_weight;
            if (jvp_value == 0.0f) {
                continue;
            }
            atomicAdd(&out_grid[ys[yi] * n_coord_0 + xs[xi]], jvp_value);
        }
    }
}

__global__ void monte_carlo_transport_vertex_vjp_into_kernel(
    const float *coord_0,
    const float *coord_1,
    const float *power,
    const int *active_mask,
    const int *vertex_indices,
    const float *coord_0_coeff_x,
    const float *coord_0_coeff_y,
    const float *coord_0_coeff_z,
    const float *coord_1_coeff_x,
    const float *coord_1_coeff_y,
    const float *coord_1_coeff_z,
    int vertex_slot_count,
    const float *upstream_grid,
    float *out_vertex_grad_x,
    float *out_vertex_grad_y,
    float *out_vertex_grad_z,
    int n_samples,
    float coord_0_min,
    float coord_1_min,
    float cell_size_0,
    float cell_size_1,
    int n_coord_0,
    int n_coord_1
) {
    int sample_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (sample_idx >= n_samples || active_mask[sample_idx] == 0) {
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
    axis_support(coord_0[sample_idx], coord_0_min, cell_size_0, x0, x1, wx0, wx1);
    axis_support(coord_1[sample_idx], coord_1_min, cell_size_1, y0, y1, wy0, wy1);

    const float inv_cell_0 = 1.0f / cell_size_0;
    const float inv_cell_1 = 1.0f / cell_size_1;
    const float d_wx[2] = {-inv_cell_0, inv_cell_0};
    const float d_wy[2] = {-inv_cell_1, inv_cell_1};
    const int xs[2] = {x0, x1};
    const int ys[2] = {y0, y1};
    const float wx[2] = {wx0, wx1};
    const float wy[2] = {wy0, wy1};

    float local_grad_coord_0 = 0.0f;
    float local_grad_coord_1 = 0.0f;
    for (int yi = 0; yi < 2; ++yi) {
        if (ys[yi] < 0 || ys[yi] >= n_coord_1) {
            continue;
        }
        for (int xi = 0; xi < 2; ++xi) {
            if (xs[xi] < 0 || xs[xi] >= n_coord_0) {
                continue;
            }
            float upstream = upstream_grid[ys[yi] * n_coord_0 + xs[xi]];
            if (upstream == 0.0f) {
                continue;
            }
            local_grad_coord_0 += upstream * power[sample_idx] * d_wx[xi] * wy[yi];
            local_grad_coord_1 += upstream * power[sample_idx] * wx[xi] * d_wy[yi];
        }
    }

    if (local_grad_coord_0 == 0.0f && local_grad_coord_1 == 0.0f) {
        return;
    }

    for (int slot = 0; slot < vertex_slot_count; ++slot) {
        int flat_idx = slot * n_samples + sample_idx;
        int global_idx = vertex_indices[flat_idx];
        if (global_idx < 0) {
            continue;
        }
        atomicAdd(
            out_vertex_grad_x + global_idx,
            local_grad_coord_0 * coord_0_coeff_x[flat_idx]
                + local_grad_coord_1 * coord_1_coeff_x[flat_idx]
        );
        atomicAdd(
            out_vertex_grad_y + global_idx,
            local_grad_coord_0 * coord_0_coeff_y[flat_idx]
                + local_grad_coord_1 * coord_1_coeff_y[flat_idx]
        );
        atomicAdd(
            out_vertex_grad_z + global_idx,
            local_grad_coord_0 * coord_0_coeff_z[flat_idx]
                + local_grad_coord_1 * coord_1_coeff_z[flat_idx]
        );
    }
}

} // namespace

void monte_carlo_transport_vertex_jvp_into(
    const float *coord_0,
    const float *coord_1,
    const float *power,
    const int *active_mask,
    const int *vertex_indices,
    const float *coord_0_coeff_x,
    const float *coord_0_coeff_y,
    const float *coord_0_coeff_z,
    const float *coord_1_coeff_x,
    const float *coord_1_coeff_y,
    const float *coord_1_coeff_z,
    int vertex_slot_count,
    const float *vertex_tangent_x,
    const float *vertex_tangent_y,
    const float *vertex_tangent_z,
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
    int grid_size = ceil_div_int(n_samples, kBlockSize);
    monte_carlo_transport_vertex_jvp_into_kernel<<<grid_size, kBlockSize>>>(
        coord_0,
        coord_1,
        power,
        active_mask,
        vertex_indices,
        coord_0_coeff_x,
        coord_0_coeff_y,
        coord_0_coeff_z,
        coord_1_coeff_x,
        coord_1_coeff_y,
        coord_1_coeff_z,
        vertex_slot_count,
        vertex_tangent_x,
        vertex_tangent_y,
        vertex_tangent_z,
        out_grid,
        n_samples,
        coord_0_min,
        coord_1_min,
        cell_size_0,
        cell_size_1,
        n_coord_0,
        n_coord_1
    );
    throw_cuda(cudaGetLastError(), "monte_carlo_transport_vertex_jvp_into_kernel launch");
}

void monte_carlo_transport_vertex_vjp_into(
    const float *coord_0,
    const float *coord_1,
    const float *power,
    const int *active_mask,
    const int *vertex_indices,
    const float *coord_0_coeff_x,
    const float *coord_0_coeff_y,
    const float *coord_0_coeff_z,
    const float *coord_1_coeff_x,
    const float *coord_1_coeff_y,
    const float *coord_1_coeff_z,
    int vertex_slot_count,
    const float *upstream_grid,
    float *out_vertex_grad_x,
    float *out_vertex_grad_y,
    float *out_vertex_grad_z,
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
    int grid_size = ceil_div_int(n_samples, kBlockSize);
    monte_carlo_transport_vertex_vjp_into_kernel<<<grid_size, kBlockSize>>>(
        coord_0,
        coord_1,
        power,
        active_mask,
        vertex_indices,
        coord_0_coeff_x,
        coord_0_coeff_y,
        coord_0_coeff_z,
        coord_1_coeff_x,
        coord_1_coeff_y,
        coord_1_coeff_z,
        vertex_slot_count,
        upstream_grid,
        out_vertex_grad_x,
        out_vertex_grad_y,
        out_vertex_grad_z,
        n_samples,
        coord_0_min,
        coord_1_min,
        cell_size_0,
        cell_size_1,
        n_coord_0,
        n_coord_1
    );
    throw_cuda(cudaGetLastError(), "monte_carlo_transport_vertex_vjp_into_kernel launch");
}

} // namespace witwin::channel::native_ext
