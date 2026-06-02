#pragma once

#include <cstdint>

namespace witwin::channel::native_ext {

// Maximum number of diffraction history slots that participate in the
// pruning tie-break tuple.
constexpr int PS_MAX_HISTORY_SLOTS = 8;

// Sort states by the exact pruning tuple used by the Python reference:
//   (-power, order, prefix_depth, intermediate_depth, suffix_depth,
//    path_edge_idx_0..N, path_reflection_depth_0..N, edge_idx, original_idx)
//
// The output contains the kept indices in ascending original order,
// matching ``torch.sort(ranked[:budget]).values`` in the reference path.
int prune_state_arrays_by_budget(
    const float*    power,                      // [n_states]
    const int*      order,                      // [n_states]
    const int*      prefix_depth,               // [n_states]
    const int*      inter_depth,                // [n_states]
    const int*      suffix_depth,               // [n_states]
    const int*      edge_idx,                   // [n_states]
    const int* const* path_edge_history,        // [history_size][n_states]
    const int* const* path_reflection_history,  // [history_size][n_states]
    int             history_size,
    int             n_states,
    int             budget,
    int*            out_indices                 // [budget]
);

void prune_state_arrays_by_budget_pair(
    const float*    power,                      // [n_states]
    const int*      order,                      // [n_states]
    const int*      prefix_depth,               // [n_states]
    const int*      inter_depth,                // [n_states]
    const int*      suffix_depth,               // [n_states]
    const int*      edge_idx,                   // [n_states]
    const int* const* path_edge_history,        // [history_size][n_states]
    const int* const* path_reflection_history,  // [history_size][n_states]
    int             history_size,
    int             n_states,
    int             higher_budget,
    int             inserted_budget,
    int*            out_higher_indices,         // [higher_budget]
    int*            out_inserted_indices        // [inserted_budget]
);

} // namespace witwin::channel::native_ext
