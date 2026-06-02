#pragma once

#include <cstdint>

namespace witwin::channel::native_ext {

// =========================================================================
// Fused Cartesian Filter for Higher-Order Diffraction
//
// Replaces the Python chunk loop that does:
//   cartesian product → dr.compress(distinct_edge)
//   → dr.compress(visible) → dr.compress(power > threshold)
// with a single kernel that evaluates all conditions per (prev, edge)
// pair and atomically appends valid pairs to a compacted output.
//
// Result: 1 GPU→CPU sync (reading out_count) instead of 5+ syncs from
// repeated dr.compress() + dr.width() calls.
// =========================================================================

// Maximum number of edge history slots checked for "distinct edge"
constexpr int CF_MAX_HISTORY_SLOTS = 8;

// -------------------------------------------------------------------------
// Bruteforce Cartesian filter: tests all (n_prev × n_edges) pairs.
//
// Validity checks fused into one kernel:
//   1. Distinct edge: edge_idx != prev_edge_idx (and not in history)
//   2. Power threshold: incident field power > min_power
//
// Visibility is NOT included here (requires BVH). The caller should
// run visibility as a separate DrJit pass on the compacted output if
// needed, or pass a pre-computed edge-pair visibility mask.
// -------------------------------------------------------------------------
void cartesian_filter_bruteforce(
    // Previous state edge indices [n_prev]
    const int*      prev_edge_idx,
    // Previous state edge history [CF_MAX_HISTORY_SLOTS * n_prev], slot-major
    // (set unused slots to -1)
    const int*      prev_edge_history,
    int             history_size,       // actual number of history slots (0..CF_MAX_HISTORY_SLOTS)

    // Previous state incident power [n_prev] (precomputed on Python side)
    const float*    prev_power,

    // Edge data [n_edges]
    // (only edge indices needed for distinct-edge check; positions used later)

    int n_prev,
    int n_edges,
    float min_power,                    // power threshold (e.g. 1e-20)

    // Output: compacted valid pairs (pre-allocated to n_prev * n_edges max)
    int*            out_prev_idx,       // [capacity]
    int*            out_edge_idx,       // [capacity]
    int*            out_count           // [1] — atomic counter, must be zero-initialized
);

// -------------------------------------------------------------------------
// Read back the compacted count from device to host.
// -------------------------------------------------------------------------
int read_compacted_count(const int* d_count);

// Restore the canonical `(prev_idx, edge_idx)` lexicographic order used
// by the DrJit reference after atomic compaction.
void sort_cartesian_pairs(
    int* out_prev_idx,   // [count]
    int* out_edge_idx,   // [count]
    int  count,
    int  n_edges
);

// Remove duplicate `(prev_idx, edge_idx)` pairs from already sorted buffers.
// Output order matches the sorted input order.
void unique_sorted_cartesian_pairs(
    const int* sorted_prev_idx,   // [count]
    const int* sorted_edge_idx,   // [count]
    int  count,
    int* out_prev_idx,            // [count] max capacity
    int* out_edge_idx,            // [count] max capacity
    int* out_count                // [1]
);

// Compact already-built `(lhs_idx, rhs_idx)` arrays using an active mask.
void compact_index_pairs(
    const int* lhs_idx,
    const int* rhs_idx,
    const int* active_mask,
    int count,
    int* out_lhs_idx,
    int* out_rhs_idx,
    int* out_count
);

} // namespace witwin::channel::native_ext
