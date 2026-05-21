#include <cuda_runtime.h>

#include <cmath>

#include <common/cuda_check.h>
#include <reflection/reflection_prefix_compact.h>

namespace witwin::channel::native_ext {
namespace {

using common::throw_cuda;

__device__ int canonical_prim(
    int prim,
    const int* canonical_prim_table,
    int canonical_table_size
) {
    if (canonical_prim_table == nullptr || prim < 0 || prim >= canonical_table_size) {
        return prim;
    }
    int mapped = canonical_prim_table[prim];
    return mapped >= 0 ? mapped : prim;
}

__device__ long long quantized_coord(float value, double inv_tol) {
    return llrint(static_cast<double>(value) * inv_tol);
}

__device__ bool active_for_depth(
    int idx,
    int depth,
    const int* bounce_count,
    const int* discovery_count
) {
    return bounce_count[idx] >= depth && discovery_count[idx] > 0;
}

__device__ bool same_prefix_key(
    int lhs,
    int rhs,
    int max_bounces,
    int depth,
    const int* global_prim_ids,
    const float* image_source_x,
    const float* image_source_y,
    const float* image_source_z,
    const int* canonical_prim_table,
    int canonical_table_size,
    double inv_tol
) {
    int lhs_base = lhs * max_bounces;
    int rhs_base = rhs * max_bounces;
    for (int slot = 0; slot < depth; ++slot) {
        int lhs_prim = canonical_prim(
            global_prim_ids[lhs_base + slot],
            canonical_prim_table,
            canonical_table_size
        );
        int rhs_prim = canonical_prim(
            global_prim_ids[rhs_base + slot],
            canonical_prim_table,
            canonical_table_size
        );
        if (lhs_prim != rhs_prim) {
            return false;
        }
    }

    int lhs_image_slot = lhs_base + depth - 1;
    int rhs_image_slot = rhs_base + depth - 1;
    return quantized_coord(image_source_x[lhs_image_slot], inv_tol)
            == quantized_coord(image_source_x[rhs_image_slot], inv_tol)
        && quantized_coord(image_source_y[lhs_image_slot], inv_tol)
            == quantized_coord(image_source_y[rhs_image_slot], inv_tol)
        && quantized_coord(image_source_z[lhs_image_slot], inv_tol)
            == quantized_coord(image_source_z[rhs_image_slot], inv_tol);
}

__global__ void mark_prefix_representatives_kernel(
    int ray_count,
    int max_bounces,
    int depth,
    double inv_tol,
    const int* bounce_count,
    const int* discovery_count,
    const int* representative_ray_index,
    const int* global_prim_ids,
    const float* image_source_x,
    const float* image_source_y,
    const float* image_source_z,
    const int* canonical_prim_table,
    int canonical_table_size,
    int* rep_flags,
    int* rep_first_seen,
    int* rep_discovery_count,
    int* out_count
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= ray_count) {
        return;
    }

    rep_flags[idx] = 0;
    rep_first_seen[idx] = 0x7fffffff;
    rep_discovery_count[idx] = 0;

    if (!active_for_depth(idx, depth, bounce_count, discovery_count)) {
        return;
    }

    int best_idx = idx;
    int best_first_seen = representative_ray_index[idx];
    int discovery_sum = 0;
    for (int other = 0; other < ray_count; ++other) {
        if (!active_for_depth(other, depth, bounce_count, discovery_count)) {
            continue;
        }
        if (!same_prefix_key(
                idx,
                other,
                max_bounces,
                depth,
                global_prim_ids,
                image_source_x,
                image_source_y,
                image_source_z,
                canonical_prim_table,
                canonical_table_size,
                inv_tol
            )) {
            continue;
        }
        discovery_sum += discovery_count[other];
        int other_first_seen = representative_ray_index[other];
        if (other_first_seen < best_first_seen ||
            (other_first_seen == best_first_seen && other < best_idx)) {
            best_first_seen = other_first_seen;
            best_idx = other;
        }
    }

    if (idx == best_idx) {
        rep_flags[idx] = 1;
        rep_first_seen[idx] = best_first_seen;
        rep_discovery_count[idx] = discovery_sum;
        atomicAdd(out_count, 1);
    }
}

__global__ void write_ordered_prefix_representatives_kernel(
    int ray_count,
    const int* rep_flags,
    const int* rep_first_seen,
    const int* rep_discovery_count,
    int* out_representative_chain_idx,
    int* out_discovery_count
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= ray_count || rep_flags[idx] == 0) {
        return;
    }

    int rank = 0;
    int first_seen = rep_first_seen[idx];
    for (int other = 0; other < ray_count; ++other) {
        if (rep_flags[other] == 0) {
            continue;
        }
        int other_first_seen = rep_first_seen[other];
        if (other_first_seen < first_seen ||
            (other_first_seen == first_seen && other < idx)) {
            ++rank;
        }
    }

    out_representative_chain_idx[rank] = idx;
    out_discovery_count[rank] = rep_discovery_count[idx];
}

} // namespace

void reflection_prefix_compact_representatives(
    const int* bounce_count,
    const int* discovery_count,
    const int* representative_ray_index,
    const int* global_prim_ids,
    const float* image_source_x,
    const float* image_source_y,
    const float* image_source_z,
    const int* canonical_prim_table,
    int canonical_table_size,
    int ray_count,
    int max_bounces,
    int depth,
    double image_source_tolerance,
    int* out_count,
    int* out_representative_chain_idx,
    int* out_discovery_count
) {
    if (ray_count <= 0 || max_bounces <= 0 || depth <= 0) {
        return;
    }

    throw_cuda(cudaMemset(out_count, 0, sizeof(int)), "reflection_prefix_compact count memset");
    throw_cuda(
        cudaMemset(
            out_representative_chain_idx,
            0xFF,
            static_cast<size_t>(ray_count) * sizeof(int)
        ),
        "reflection_prefix_compact representative memset"
    );
    throw_cuda(
        cudaMemset(
            out_discovery_count,
            0,
            static_cast<size_t>(ray_count) * sizeof(int)
        ),
        "reflection_prefix_compact discovery memset"
    );

    int* rep_flags = nullptr;
    int* rep_first_seen = nullptr;
    int* rep_discovery_count = nullptr;
    size_t bytes = static_cast<size_t>(ray_count) * sizeof(int);
    throw_cuda(cudaMalloc(&rep_flags, bytes), "reflection_prefix_compact malloc rep_flags");
    throw_cuda(cudaMalloc(&rep_first_seen, bytes), "reflection_prefix_compact malloc first_seen");
    throw_cuda(cudaMalloc(&rep_discovery_count, bytes), "reflection_prefix_compact malloc discovery");

    constexpr int BLOCK = 256;
    int grid = (ray_count + BLOCK - 1) / BLOCK;
    double inv_tol = 1.0 / fmax(image_source_tolerance, 1e-12);
    mark_prefix_representatives_kernel<<<grid, BLOCK>>>(
        ray_count,
        max_bounces,
        depth,
        inv_tol,
        bounce_count,
        discovery_count,
        representative_ray_index,
        global_prim_ids,
        image_source_x,
        image_source_y,
        image_source_z,
        canonical_prim_table,
        canonical_table_size,
        rep_flags,
        rep_first_seen,
        rep_discovery_count,
        out_count
    );
    throw_cuda(cudaGetLastError(), "mark_prefix_representatives_kernel launch");

    write_ordered_prefix_representatives_kernel<<<grid, BLOCK>>>(
        ray_count,
        rep_flags,
        rep_first_seen,
        rep_discovery_count,
        out_representative_chain_idx,
        out_discovery_count
    );
    throw_cuda(cudaGetLastError(), "write_ordered_prefix_representatives_kernel launch");

    cudaFree(rep_flags);
    cudaFree(rep_first_seen);
    cudaFree(rep_discovery_count);
}

} // namespace witwin::channel::native_ext
