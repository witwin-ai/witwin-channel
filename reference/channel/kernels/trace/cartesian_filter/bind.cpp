#include "drjit_common.h"
#include <trace/cartesian_filter/bind.h>

#include <trace/cartesian_filter/cartesian_filter.h>

void register_cartesian_filter_bindings(nb::module_ &m) {
    m.def(
        "cartesian_filter_bruteforce_raw",
        [](
            std::uintptr_t prev_edge_idx,
            std::uintptr_t prev_edge_history,
            int history_size,
            std::uintptr_t prev_power,
            int n_prev,
            int n_edges,
            float min_power,
            std::uintptr_t out_prev_idx,
            std::uintptr_t out_edge_idx,
            std::uintptr_t out_count
        ) {
            witwin::channel::native_ext::cartesian_filter_bruteforce(
                ptr<const int>(prev_edge_idx),
                ptr<const int>(prev_edge_history),
                history_size,
                ptr<const float>(prev_power),
                n_prev,
                n_edges,
                min_power,
                ptr_mut<int>(out_prev_idx),
                ptr_mut<int>(out_edge_idx),
                ptr_mut<int>(out_count)
            );
        },
        "Fused Cartesian filter: distinct-edge + power threshold + atomic compaction."
    );
    m.def(
        "read_compacted_count",
        [](std::uintptr_t d_count) {
            return witwin::channel::native_ext::read_compacted_count(ptr<const int>(d_count));
        },
        "Read compacted pair count from device to host."
    );
    m.def(
        "sort_cartesian_pairs_raw",
        [](
            std::uintptr_t out_prev_idx,
            std::uintptr_t out_edge_idx,
            int count,
            int n_edges
        ) {
            witwin::channel::native_ext::sort_cartesian_pairs(
                ptr_mut<int>(out_prev_idx),
                ptr_mut<int>(out_edge_idx),
                count,
                n_edges
            );
        },
        "Restore canonical pair order after atomic Cartesian compaction."
    );
    m.def(
        "cartesian_filter_bruteforce_arrays",
        [](
            Int32 prev_edge_idx,
            Int32 prev_edge_history,
            int history_size,
            Float prev_power,
            int n_prev,
            int n_edges,
            float min_power
        ) {
            if (n_prev <= 0 || n_edges <= 0) {
                return nb::make_tuple(drjit::zeros<Int32>(0), drjit::zeros<Int32>(0));
            }

            int capacity = n_prev * n_edges;
            Int32 out_prev = drjit::zeros<Int32>(capacity);
            Int32 out_edge = drjit::zeros<Int32>(capacity);
            Int32 out_count = drjit::zeros<Int32>(1);

            drjit::eval(prev_edge_idx, prev_edge_history, prev_power, out_prev, out_edge, out_count);

            witwin::channel::native_ext::cartesian_filter_bruteforce(
                drjit_data_ptr(prev_edge_idx),
                history_size > 0 ? drjit_data_ptr(prev_edge_history) : nullptr,
                history_size,
                drjit_data_ptr(prev_power),
                n_prev,
                n_edges,
                min_power,
                drjit_data_ptr_mut(out_prev),
                drjit_data_ptr_mut(out_edge),
                drjit_data_ptr_mut(out_count)
            );

            int count = witwin::channel::native_ext::read_compacted_count(drjit_data_ptr(out_count));
            if (count <= 0) {
                return nb::make_tuple(drjit::zeros<Int32>(0), drjit::zeros<Int32>(0));
            }

            witwin::channel::native_ext::sort_cartesian_pairs(
                drjit_data_ptr_mut(out_prev),
                drjit_data_ptr_mut(out_edge),
                count,
                n_edges
            );

            UInt32 idx = drjit::arange<UInt32>(count);
            return nb::make_tuple(
                drjit::gather<Int32>(out_prev, idx),
                drjit::gather<Int32>(out_edge, idx)
            );
        },
        "Fused Cartesian filter with zero-copy Dr.Jit array bindings."
    );
    m.def(
        "unique_sorted_cartesian_pairs_raw",
        [](
            std::uintptr_t sorted_prev_idx,
            std::uintptr_t sorted_edge_idx,
            int count,
            std::uintptr_t out_prev_idx,
            std::uintptr_t out_edge_idx,
            std::uintptr_t out_count
        ) {
            witwin::channel::native_ext::unique_sorted_cartesian_pairs(
                ptr<const int>(sorted_prev_idx),
                ptr<const int>(sorted_edge_idx),
                count,
                ptr_mut<int>(out_prev_idx),
                ptr_mut<int>(out_edge_idx),
                ptr_mut<int>(out_count)
            );
        },
        "Remove duplicate sorted Cartesian pairs on the GPU."
    );
    m.def(
        "deduplicate_cartesian_pairs_arrays",
        [](Int32 prev_idx, Int32 edge_idx, int n_edges) {
            int count = (int) prev_idx.size();
            if (count <= 0) {
                return nb::make_tuple(drjit::zeros<Int32>(0), drjit::zeros<Int32>(0));
            }

            UInt32 idx = drjit::arange<UInt32>(count);
            Int32 sorted_prev = drjit::gather<Int32>(prev_idx, idx);
            Int32 sorted_edge = drjit::gather<Int32>(edge_idx, idx);
            Int32 out_prev = drjit::zeros<Int32>(count);
            Int32 out_edge = drjit::zeros<Int32>(count);
            Int32 out_count = drjit::zeros<Int32>(1);

            drjit::eval(sorted_prev, sorted_edge, out_prev, out_edge, out_count);

            witwin::channel::native_ext::sort_cartesian_pairs(
                drjit_data_ptr_mut(sorted_prev),
                drjit_data_ptr_mut(sorted_edge),
                count,
                n_edges
            );
            witwin::channel::native_ext::unique_sorted_cartesian_pairs(
                drjit_data_ptr(sorted_prev),
                drjit_data_ptr(sorted_edge),
                count,
                drjit_data_ptr_mut(out_prev),
                drjit_data_ptr_mut(out_edge),
                drjit_data_ptr_mut(out_count)
            );

            int unique_count = witwin::channel::native_ext::read_compacted_count(drjit_data_ptr(out_count));
            if (unique_count <= 0) {
                return nb::make_tuple(drjit::zeros<Int32>(0), drjit::zeros<Int32>(0));
            }

            UInt32 unique_idx = drjit::arange<UInt32>(unique_count);
            return nb::make_tuple(
                drjit::gather<Int32>(out_prev, unique_idx),
                drjit::gather<Int32>(out_edge, unique_idx)
            );
        },
        "Sort and deduplicate Cartesian pairs with zero-copy Dr.Jit array bindings."
    );
    m.def(
        "compact_index_pairs_raw",
        [](
            std::uintptr_t lhs_idx,
            std::uintptr_t rhs_idx,
            std::uintptr_t active_mask,
            int count,
            std::uintptr_t out_lhs_idx,
            std::uintptr_t out_rhs_idx,
            std::uintptr_t out_count
        ) {
            witwin::channel::native_ext::compact_index_pairs(
                ptr<const int>(lhs_idx),
                ptr<const int>(rhs_idx),
                ptr<const int>(active_mask),
                count,
                ptr_mut<int>(out_lhs_idx),
                ptr_mut<int>(out_rhs_idx),
                ptr_mut<int>(out_count)
            );
        },
        "Compact explicit pair-index arrays with an active mask."
    );
    m.def(
        "compact_index_pairs_arrays",
        [](Int32 lhs_idx, Int32 rhs_idx, Int32 active_mask) {
            int count = (int) lhs_idx.size();
            if (count <= 0) {
                return nb::make_tuple(drjit::zeros<Int32>(0), drjit::zeros<Int32>(0));
            }
            Int32 out_lhs = drjit::zeros<Int32>(count);
            Int32 out_rhs = drjit::zeros<Int32>(count);
            Int32 out_count = drjit::zeros<Int32>(1);
            drjit::eval(lhs_idx, rhs_idx, active_mask, out_lhs, out_rhs, out_count);
            witwin::channel::native_ext::compact_index_pairs(
                drjit_data_ptr(lhs_idx),
                drjit_data_ptr(rhs_idx),
                drjit_data_ptr(active_mask),
                count,
                drjit_data_ptr_mut(out_lhs),
                drjit_data_ptr_mut(out_rhs),
                drjit_data_ptr_mut(out_count)
            );
            int compacted = witwin::channel::native_ext::read_compacted_count(drjit_data_ptr(out_count));
            if (compacted <= 0) {
                return nb::make_tuple(drjit::zeros<Int32>(0), drjit::zeros<Int32>(0));
            }
            UInt32 keep = drjit::arange<UInt32>(compacted);
            return nb::make_tuple(
                drjit::gather<Int32>(out_lhs, keep),
                drjit::gather<Int32>(out_rhs, keep)
            );
        },
        "Compact explicit pair-index arrays with zero-copy Dr.Jit bindings."
    );
    m.attr("CF_MAX_HISTORY_SLOTS") = witwin::channel::native_ext::CF_MAX_HISTORY_SLOTS;
}
