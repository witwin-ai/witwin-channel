#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>

#include <common/cuda_check.h>
#include <trace/pruning_sort/pruning_sort.h>

namespace witwin::channel::native_ext {
namespace {

using common::throw_cuda;

struct PruningLexicographicLess {
    const float* power = nullptr;
    const int* order = nullptr;
    const int* prefix_depth = nullptr;
    const int* inter_depth = nullptr;
    const int* suffix_depth = nullptr;
    const int* edge_idx = nullptr;
    const int* path_edge_history[PS_MAX_HISTORY_SLOTS] = {};
    const int* path_reflection_history[PS_MAX_HISTORY_SLOTS] = {};
    int history_size = 0;

    __host__ __device__ bool operator()(int lhs, int rhs) const {
        float lhs_power = power[lhs];
        float rhs_power = power[rhs];
        if (lhs_power > rhs_power) {
            return true;
        }
        if (lhs_power < rhs_power) {
            return false;
        }

        const int lhs_order = order[lhs];
        const int rhs_order = order[rhs];
        if (lhs_order != rhs_order) {
            return lhs_order < rhs_order;
        }

        const int lhs_prefix = prefix_depth[lhs];
        const int rhs_prefix = prefix_depth[rhs];
        if (lhs_prefix != rhs_prefix) {
            return lhs_prefix < rhs_prefix;
        }

        const int lhs_inter = inter_depth[lhs];
        const int rhs_inter = inter_depth[rhs];
        if (lhs_inter != rhs_inter) {
            return lhs_inter < rhs_inter;
        }

        const int lhs_suffix = suffix_depth[lhs];
        const int rhs_suffix = suffix_depth[rhs];
        if (lhs_suffix != rhs_suffix) {
            return lhs_suffix < rhs_suffix;
        }

        for (int slot = 0; slot < history_size; ++slot) {
            const int lhs_hist = path_edge_history[slot][lhs];
            const int rhs_hist = path_edge_history[slot][rhs];
            if (lhs_hist != rhs_hist) {
                return lhs_hist < rhs_hist;
            }
        }

        for (int slot = 0; slot < history_size; ++slot) {
            const int lhs_hist = path_reflection_history[slot][lhs];
            const int rhs_hist = path_reflection_history[slot][rhs];
            if (lhs_hist != rhs_hist) {
                return lhs_hist < rhs_hist;
            }
        }

        const int lhs_edge = edge_idx[lhs];
        const int rhs_edge = edge_idx[rhs];
        if (lhs_edge != rhs_edge) {
            return lhs_edge < rhs_edge;
        }

        return lhs < rhs;
    }
};

} // anonymous namespace

int prune_state_arrays_by_budget(
    const float*    power,
    const int*      order,
    const int*      prefix_depth,
    const int*      inter_depth,
    const int*      suffix_depth,
    const int*      edge_idx,
    const int* const* path_edge_history,
    const int* const* path_reflection_history,
    int             history_size,
    int             n_states,
    int             budget,
    int*            out_indices)
{
    if (n_states <= 0 || budget <= 0) {
        return 0;
    }
    if (history_size < 0 || history_size > PS_MAX_HISTORY_SLOTS) {
        throw std::runtime_error("history_size exceeds PS_MAX_HISTORY_SLOTS");
    }

    const int keep = (n_states < budget) ? n_states : budget;
    int* d_indices = nullptr;
    throw_cuda(cudaMalloc(&d_indices, static_cast<size_t>(n_states) * sizeof(int)), "malloc d_indices");

    try {
        PruningLexicographicLess cmp;
        cmp.power = power;
        cmp.order = order;
        cmp.prefix_depth = prefix_depth;
        cmp.inter_depth = inter_depth;
        cmp.suffix_depth = suffix_depth;
        cmp.edge_idx = edge_idx;
        cmp.history_size = history_size;
        for (int slot = 0; slot < history_size; ++slot) {
            cmp.path_edge_history[slot] = path_edge_history[slot];
            cmp.path_reflection_history[slot] = path_reflection_history[slot];
        }

        auto begin = thrust::device_pointer_cast(d_indices);
        auto end = begin + n_states;

        thrust::sequence(thrust::device, begin, end);
        thrust::sort(thrust::device, begin, end, cmp);
        thrust::sort(thrust::device, begin, begin + keep);

        throw_cuda(
            cudaMemcpy(
                out_indices,
                d_indices,
                static_cast<size_t>(keep) * sizeof(int),
                cudaMemcpyDeviceToDevice
            ),
            "memcpy out_indices"
        );
        throw_cuda(cudaDeviceSynchronize(), "prune_state_arrays_by_budget sync");
    } catch (...) {
        cudaFree(d_indices);
        throw;
    }

    cudaFree(d_indices);
    return keep;
}

void prune_state_arrays_by_budget_pair(
    const float*    power,
    const int*      order,
    const int*      prefix_depth,
    const int*      inter_depth,
    const int*      suffix_depth,
    const int*      edge_idx,
    const int* const* path_edge_history,
    const int* const* path_reflection_history,
    int             history_size,
    int             n_states,
    int             higher_budget,
    int             inserted_budget,
    int*            out_higher_indices,
    int*            out_inserted_indices)
{
    if (n_states <= 0 || (higher_budget <= 0 && inserted_budget <= 0)) {
        return;
    }
    if (history_size < 0 || history_size > PS_MAX_HISTORY_SLOTS) {
        throw std::runtime_error("history_size exceeds PS_MAX_HISTORY_SLOTS");
    }

    const int higher_keep = (higher_budget <= 0) ? 0 : ((n_states < higher_budget) ? n_states : higher_budget);
    const int inserted_keep = (inserted_budget <= 0) ? 0 : ((n_states < inserted_budget) ? n_states : inserted_budget);
    const int max_keep = higher_keep > inserted_keep ? higher_keep : inserted_keep;
    if (max_keep <= 0) {
        return;
    }

    int* d_indices = nullptr;
    int* d_ranked = nullptr;
    throw_cuda(cudaMalloc(&d_indices, static_cast<size_t>(n_states) * sizeof(int)), "malloc d_indices");
    throw_cuda(cudaMalloc(&d_ranked, static_cast<size_t>(max_keep) * sizeof(int)), "malloc d_ranked");

    try {
        PruningLexicographicLess cmp;
        cmp.power = power;
        cmp.order = order;
        cmp.prefix_depth = prefix_depth;
        cmp.inter_depth = inter_depth;
        cmp.suffix_depth = suffix_depth;
        cmp.edge_idx = edge_idx;
        cmp.history_size = history_size;
        for (int slot = 0; slot < history_size; ++slot) {
            cmp.path_edge_history[slot] = path_edge_history[slot];
            cmp.path_reflection_history[slot] = path_reflection_history[slot];
        }

        auto begin = thrust::device_pointer_cast(d_indices);
        auto ranked_begin = thrust::device_pointer_cast(d_ranked);
        auto end = begin + n_states;

        thrust::sequence(thrust::device, begin, end);
        thrust::sort(thrust::device, begin, end, cmp);
        throw_cuda(
            cudaMemcpy(
                d_ranked,
                d_indices,
                static_cast<size_t>(max_keep) * sizeof(int),
                cudaMemcpyDeviceToDevice
            ),
            "memcpy ranked indices"
        );

        if (higher_keep > 0) {
            throw_cuda(
                cudaMemcpy(
                    out_higher_indices,
                    d_ranked,
                    static_cast<size_t>(higher_keep) * sizeof(int),
                    cudaMemcpyDeviceToDevice
                ),
                "memcpy higher indices"
            );
            thrust::sort(
                thrust::device,
                thrust::device_pointer_cast(out_higher_indices),
                thrust::device_pointer_cast(out_higher_indices) + higher_keep
            );
        }

        if (inserted_keep > 0) {
            throw_cuda(
                cudaMemcpy(
                    out_inserted_indices,
                    d_ranked,
                    static_cast<size_t>(inserted_keep) * sizeof(int),
                    cudaMemcpyDeviceToDevice
                ),
                "memcpy inserted indices"
            );
            thrust::sort(
                thrust::device,
                thrust::device_pointer_cast(out_inserted_indices),
                thrust::device_pointer_cast(out_inserted_indices) + inserted_keep
            );
        }

        throw_cuda(cudaDeviceSynchronize(), "prune_state_arrays_by_budget_pair sync");
    } catch (...) {
        cudaFree(d_indices);
        cudaFree(d_ranked);
        throw;
    }

    cudaFree(d_indices);
    cudaFree(d_ranked);
}

} // namespace witwin::channel::native_ext
