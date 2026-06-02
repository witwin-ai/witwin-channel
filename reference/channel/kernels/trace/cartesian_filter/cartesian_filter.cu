#include <cuda_runtime.h>
#include <thrust/copy.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/sort.h>
#include <thrust/tuple.h>

#include <common/cuda_check.h>
#include <trace/cartesian_filter/cartesian_filter.h>

namespace witwin::channel::native_ext {
namespace {

using common::throw_cuda;

// =========================================================================
// Bruteforce Cartesian filter kernel
//
// One thread per (prev, edge) pair. Checks:
//   1. edge_idx is not the same as prev_edge_idx
//   2. edge_idx is not in the prev state's edge history
//   3. incident power exceeds threshold
// If all pass, atomically appends to output.
// =========================================================================
__global__ void cartesian_filter_bruteforce_kernel(
    const int*   __restrict__ prev_edge_idx,
    const int*   __restrict__ prev_edge_history,   // [history_size * n_prev], slot-major
    int                       history_size,
    const float* __restrict__ prev_power,
    int n_prev,
    int n_edges,
    float min_power,
    int* __restrict__ out_prev_idx,
    int* __restrict__ out_edge_idx,
    int* __restrict__ out_count)
{
    int pair_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (pair_id >= n_prev * n_edges) return;

    int prev_idx = pair_id / n_edges;
    int edge_idx = pair_id % n_edges;

    // --- Check 1: power threshold ---
    if (prev_power[prev_idx] <= min_power) return;

    // --- Check 2: distinct from current edge ---
    if (edge_idx == prev_edge_idx[prev_idx]) return;

    // --- Check 3: distinct from edge history ---
    for (int s = 0; s < history_size; ++s) {
        int hist_edge = prev_edge_history[s * n_prev + prev_idx];
        if (hist_edge >= 0 && edge_idx == hist_edge) return;
    }

    // All checks passed â€?append to output
    int slot = atomicAdd(out_count, 1);
    out_prev_idx[slot] = prev_idx;
    out_edge_idx[slot] = edge_idx;
}

__global__ void build_pair_sort_keys_kernel(
    const int* __restrict__ prev_idx,
    const int* __restrict__ edge_idx,
    unsigned long long* __restrict__ keys,
    int count,
    int n_edges)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= count) return;
    keys[tid] =
        static_cast<unsigned long long>(static_cast<unsigned int>(prev_idx[tid])) * static_cast<unsigned long long>(n_edges)
        + static_cast<unsigned long long>(static_cast<unsigned int>(edge_idx[tid]));
}

__global__ void unique_sorted_cartesian_pairs_kernel(
    const int* __restrict__ sorted_prev_idx,
    const int* __restrict__ sorted_edge_idx,
    int count,
    int* __restrict__ out_prev_idx,
    int* __restrict__ out_edge_idx,
    int* __restrict__ out_count)
{
    if (blockIdx.x != 0 || threadIdx.x != 0)
        return;

    if (count <= 0) {
        *out_count = 0;
        return;
    }

    int out = 0;
    int last_prev = sorted_prev_idx[0];
    int last_edge = sorted_edge_idx[0];
    out_prev_idx[out] = last_prev;
    out_edge_idx[out] = last_edge;
    ++out;

    for (int i = 1; i < count; ++i) {
        int cur_prev = sorted_prev_idx[i];
        int cur_edge = sorted_edge_idx[i];
        if (cur_prev != last_prev || cur_edge != last_edge) {
            out_prev_idx[out] = cur_prev;
            out_edge_idx[out] = cur_edge;
            last_prev = cur_prev;
            last_edge = cur_edge;
            ++out;
        }
    }

    *out_count = out;
}

struct active_pair_mask {
    __host__ __device__ bool operator()(int value) const {
        return value != 0;
    }
};

} // anonymous namespace

// =========================================================================
// Host launchers
// =========================================================================

void cartesian_filter_bruteforce(
    const int*   prev_edge_idx,
    const int*   prev_edge_history,
    int          history_size,
    const float* prev_power,
    int n_prev,
    int n_edges,
    float min_power,
    int* out_prev_idx,
    int* out_edge_idx,
    int* out_count)
{
    int total_pairs = n_prev * n_edges;
    if (total_pairs <= 0) return;
    if (history_size > CF_MAX_HISTORY_SLOTS)
        throw std::runtime_error("history_size exceeds CF_MAX_HISTORY_SLOTS");

    constexpr int BLOCK = 256;
    int grid = (total_pairs + BLOCK - 1) / BLOCK;

    cartesian_filter_bruteforce_kernel<<<grid, BLOCK>>>(
        prev_edge_idx, prev_edge_history, history_size,
        prev_power, n_prev, n_edges, min_power,
        out_prev_idx, out_edge_idx, out_count);

    throw_cuda(cudaGetLastError(), "cartesian_filter_bruteforce_kernel launch");
}

int read_compacted_count(const int* d_count) {
    int h_count = 0;
    throw_cuda(cudaMemcpy(&h_count, d_count, sizeof(int), cudaMemcpyDeviceToHost),
               "read_compacted_count memcpy");
    return h_count;
}

void sort_cartesian_pairs(
    int* out_prev_idx,
    int* out_edge_idx,
    int count,
    int n_edges)
{
    if (count <= 1) return;

    unsigned long long* d_keys = nullptr;
    throw_cuda(cudaMalloc(&d_keys, static_cast<size_t>(count) * sizeof(unsigned long long)), "malloc pair sort keys");

    try {
        constexpr int BLOCK = 256;
        int grid = (count + BLOCK - 1) / BLOCK;
        build_pair_sort_keys_kernel<<<grid, BLOCK>>>(
            out_prev_idx,
            out_edge_idx,
            d_keys,
            count,
            n_edges
        );
        throw_cuda(cudaGetLastError(), "build_pair_sort_keys_kernel launch");

        auto key_begin = thrust::device_pointer_cast(d_keys);
        auto prev_begin = thrust::device_pointer_cast(out_prev_idx);
        auto edge_begin = thrust::device_pointer_cast(out_edge_idx);
        auto value_begin = thrust::make_zip_iterator(thrust::make_tuple(prev_begin, edge_begin));
        thrust::sort_by_key(thrust::device, key_begin, key_begin + count, value_begin);
    } catch (...) {
        cudaFree(d_keys);
        throw;
    }

    cudaFree(d_keys);
}

void unique_sorted_cartesian_pairs(
    const int* sorted_prev_idx,
    const int* sorted_edge_idx,
    int count,
    int* out_prev_idx,
    int* out_edge_idx,
    int* out_count)
{
    unique_sorted_cartesian_pairs_kernel<<<1, 1>>>(
        sorted_prev_idx,
        sorted_edge_idx,
        count,
        out_prev_idx,
        out_edge_idx,
        out_count
    );
    throw_cuda(cudaGetLastError(), "unique_sorted_cartesian_pairs_kernel launch");
}

void compact_index_pairs(
    const int* lhs_idx,
    const int* rhs_idx,
    const int* active_mask,
    int count,
    int* out_lhs_idx,
    int* out_rhs_idx,
    int* out_count)
{
    if (count <= 0)
        return;

    auto lhs_begin = thrust::device_pointer_cast(lhs_idx);
    auto rhs_begin = thrust::device_pointer_cast(rhs_idx);
    auto mask_begin = thrust::device_pointer_cast(active_mask);
    auto out_lhs_begin = thrust::device_pointer_cast(out_lhs_idx);
    auto out_rhs_begin = thrust::device_pointer_cast(out_rhs_idx);
    auto in_begin = thrust::make_zip_iterator(thrust::make_tuple(lhs_begin, rhs_begin));
    auto out_begin = thrust::make_zip_iterator(thrust::make_tuple(out_lhs_begin, out_rhs_begin));
    auto out_end = thrust::copy_if(
        thrust::device,
        in_begin,
        in_begin + count,
        mask_begin,
        out_begin,
        active_pair_mask{}
    );
    int compacted = static_cast<int>(out_end - out_begin);
    throw_cuda(
        cudaMemcpy(out_count, &compacted, sizeof(int), cudaMemcpyHostToDevice),
        "compact_index_pairs count memcpy"
    );
}

} // namespace witwin::channel::native_ext
