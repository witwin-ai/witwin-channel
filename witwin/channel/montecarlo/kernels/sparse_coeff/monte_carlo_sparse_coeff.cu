#include <cuda_runtime.h>

#include <common/cuda_check.h>
#include <common/primitives.h>
#include <sparse_coeff/monte_carlo_sparse_coeff.h>

namespace witwin::channel::native_ext {
namespace {

using common::ceil_div_int;
using common::throw_cuda;

constexpr int kBlockSize = 256;

__global__ void monte_carlo_sparse_coeff_jvp_into_kernel(
    const unsigned int* cell_idx,
    const float* tx_coeff_x,
    const float* tx_coeff_y,
    const float* tx_coeff_z,
    const int* vertex_indices,
    const float* vertex_coeff_x,
    const float* vertex_coeff_y,
    const float* vertex_coeff_z,
    int vertex_slot_count,
    const int* material_indices,
    const float* material_coeff_eps,
    const float* material_coeff_sigma,
    int material_slot_count,
    const float* tx_tangent_x,
    const float* tx_tangent_y,
    const float* tx_tangent_z,
    const float* vertex_tangent_x,
    const float* vertex_tangent_y,
    const float* vertex_tangent_z,
    const float* material_tangent_eps,
    const float* material_tangent_sigma,
    float* out_component,
    int n_samples
) {
    int sample_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (sample_idx >= n_samples) {
        return;
    }

    float tangent = tx_coeff_x[sample_idx] * tx_tangent_x[sample_idx]
        + tx_coeff_y[sample_idx] * tx_tangent_y[sample_idx]
        + tx_coeff_z[sample_idx] * tx_tangent_z[sample_idx];

    for (int slot = 0; slot < vertex_slot_count; ++slot) {
        int flat_idx = slot * n_samples + sample_idx;
        int global_idx = vertex_indices[flat_idx];
        if (global_idx < 0) {
            continue;
        }
        tangent += vertex_coeff_x[flat_idx] * vertex_tangent_x[global_idx];
        tangent += vertex_coeff_y[flat_idx] * vertex_tangent_y[global_idx];
        tangent += vertex_coeff_z[flat_idx] * vertex_tangent_z[global_idx];
    }

    for (int slot = 0; slot < material_slot_count; ++slot) {
        int flat_idx = slot * n_samples + sample_idx;
        int global_idx = material_indices[flat_idx];
        if (global_idx < 0) {
            continue;
        }
        tangent += material_coeff_eps[flat_idx] * material_tangent_eps[global_idx];
        tangent += material_coeff_sigma[flat_idx] * material_tangent_sigma[global_idx];
    }

    if (tangent != 0.0f) {
        atomicAdd(out_component + cell_idx[sample_idx], tangent);
    }
}

__global__ void monte_carlo_sparse_coeff_vjp_into_kernel(
    const unsigned int* cell_idx,
    const float* tx_coeff_x,
    const float* tx_coeff_y,
    const float* tx_coeff_z,
    const int* vertex_indices,
    const float* vertex_coeff_x,
    const float* vertex_coeff_y,
    const float* vertex_coeff_z,
    int vertex_slot_count,
    const int* material_indices,
    const float* material_coeff_eps,
    const float* material_coeff_sigma,
    int material_slot_count,
    const float* upstream_component,
    float* out_tx_grad_x,
    float* out_tx_grad_y,
    float* out_tx_grad_z,
    float* out_vertex_grad_x,
    float* out_vertex_grad_y,
    float* out_vertex_grad_z,
    float* out_material_grad_eps,
    float* out_material_grad_sigma,
    int n_samples
) {
    int sample_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (sample_idx >= n_samples) {
        return;
    }

    float upstream = upstream_component[cell_idx[sample_idx]];
    if (upstream == 0.0f) {
        return;
    }

    out_tx_grad_x[sample_idx] += tx_coeff_x[sample_idx] * upstream;
    out_tx_grad_y[sample_idx] += tx_coeff_y[sample_idx] * upstream;
    out_tx_grad_z[sample_idx] += tx_coeff_z[sample_idx] * upstream;

    for (int slot = 0; slot < vertex_slot_count; ++slot) {
        int flat_idx = slot * n_samples + sample_idx;
        int global_idx = vertex_indices[flat_idx];
        if (global_idx < 0) {
            continue;
        }
        atomicAdd(out_vertex_grad_x + global_idx, vertex_coeff_x[flat_idx] * upstream);
        atomicAdd(out_vertex_grad_y + global_idx, vertex_coeff_y[flat_idx] * upstream);
        atomicAdd(out_vertex_grad_z + global_idx, vertex_coeff_z[flat_idx] * upstream);
    }

    for (int slot = 0; slot < material_slot_count; ++slot) {
        int flat_idx = slot * n_samples + sample_idx;
        int global_idx = material_indices[flat_idx];
        if (global_idx < 0) {
            continue;
        }
        atomicAdd(out_material_grad_eps + global_idx, material_coeff_eps[flat_idx] * upstream);
        atomicAdd(out_material_grad_sigma + global_idx, material_coeff_sigma[flat_idx] * upstream);
    }
}

} // namespace

void monte_carlo_sparse_coeff_jvp_into(
    const unsigned int* cell_idx,
    const float* tx_coeff_x,
    const float* tx_coeff_y,
    const float* tx_coeff_z,
    const int* vertex_indices,
    const float* vertex_coeff_x,
    const float* vertex_coeff_y,
    const float* vertex_coeff_z,
    int vertex_slot_count,
    const int* material_indices,
    const float* material_coeff_eps,
    const float* material_coeff_sigma,
    int material_slot_count,
    const float* tx_tangent_x,
    const float* tx_tangent_y,
    const float* tx_tangent_z,
    const float* vertex_tangent_x,
    const float* vertex_tangent_y,
    const float* vertex_tangent_z,
    const float* material_tangent_eps,
    const float* material_tangent_sigma,
    float* out_component,
    int n_samples
) {
    if (n_samples <= 0) {
        return;
    }
    int grid_size = ceil_div_int(n_samples, kBlockSize);
    monte_carlo_sparse_coeff_jvp_into_kernel<<<grid_size, kBlockSize>>>(
        cell_idx,
        tx_coeff_x,
        tx_coeff_y,
        tx_coeff_z,
        vertex_indices,
        vertex_coeff_x,
        vertex_coeff_y,
        vertex_coeff_z,
        vertex_slot_count,
        material_indices,
        material_coeff_eps,
        material_coeff_sigma,
        material_slot_count,
        tx_tangent_x,
        tx_tangent_y,
        tx_tangent_z,
        vertex_tangent_x,
        vertex_tangent_y,
        vertex_tangent_z,
        material_tangent_eps,
        material_tangent_sigma,
        out_component,
        n_samples
    );
    throw_cuda(cudaGetLastError(), "monte_carlo_sparse_coeff_jvp_into_kernel launch");
}

void monte_carlo_sparse_coeff_vjp_into(
    const unsigned int* cell_idx,
    const float* tx_coeff_x,
    const float* tx_coeff_y,
    const float* tx_coeff_z,
    const int* vertex_indices,
    const float* vertex_coeff_x,
    const float* vertex_coeff_y,
    const float* vertex_coeff_z,
    int vertex_slot_count,
    const int* material_indices,
    const float* material_coeff_eps,
    const float* material_coeff_sigma,
    int material_slot_count,
    const float* upstream_component,
    float* out_tx_grad_x,
    float* out_tx_grad_y,
    float* out_tx_grad_z,
    float* out_vertex_grad_x,
    float* out_vertex_grad_y,
    float* out_vertex_grad_z,
    float* out_material_grad_eps,
    float* out_material_grad_sigma,
    int n_samples
) {
    if (n_samples <= 0) {
        return;
    }
    int grid_size = ceil_div_int(n_samples, kBlockSize);
    monte_carlo_sparse_coeff_vjp_into_kernel<<<grid_size, kBlockSize>>>(
        cell_idx,
        tx_coeff_x,
        tx_coeff_y,
        tx_coeff_z,
        vertex_indices,
        vertex_coeff_x,
        vertex_coeff_y,
        vertex_coeff_z,
        vertex_slot_count,
        material_indices,
        material_coeff_eps,
        material_coeff_sigma,
        material_slot_count,
        upstream_component,
        out_tx_grad_x,
        out_tx_grad_y,
        out_tx_grad_z,
        out_vertex_grad_x,
        out_vertex_grad_y,
        out_vertex_grad_z,
        out_material_grad_eps,
        out_material_grad_sigma,
        n_samples
    );
    throw_cuda(cudaGetLastError(), "monte_carlo_sparse_coeff_vjp_into_kernel launch");
}

} // namespace witwin::channel::native_ext
