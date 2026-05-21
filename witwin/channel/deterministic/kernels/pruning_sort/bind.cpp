#include "drjit_common.h"
#include <pruning_sort/bind.h>

#include <pruning_sort/pruning_sort.h>

void register_pruning_sort_bindings(nb::module_ &m) {
    m.def(
        "prune_state_arrays_by_budget_arrays",
        [](
            Float power,
            Int32 order,
            Int32 prefix_depth,
            Int32 inter_depth,
            Int32 suffix_depth,
            Int32 edge_idx,
            nb::list path_edge_history,
            nb::list path_reflection_history,
            int history_size,
            int n_states,
            int budget
        ) {
            auto edge_history_ptrs = array_pointer_list(
                path_edge_history,
                "prune_state_arrays_by_budget_arrays(path_edge_history)"
            );
            auto reflection_history_ptrs = array_pointer_list(
                path_reflection_history,
                "prune_state_arrays_by_budget_arrays(path_reflection_history)"
            );
            if (edge_history_ptrs.size() != static_cast<size_t>(history_size)
                || reflection_history_ptrs.size() != static_cast<size_t>(history_size)) {
                throw std::runtime_error(
                    "prune_state_arrays_by_budget_arrays: history pointer length mismatch"
                );
            }

            std::vector<const int*> edge_history_raw;
            std::vector<const int*> reflection_history_raw;
            edge_history_raw.reserve(edge_history_ptrs.size());
            reflection_history_raw.reserve(reflection_history_ptrs.size());
            for (size_t i = 0; i < edge_history_ptrs.size(); ++i) {
                edge_history_raw.push_back(ptr<const int>(edge_history_ptrs[i]));
                reflection_history_raw.push_back(ptr<const int>(reflection_history_ptrs[i]));
            }

            DiffInt32 out_indices = drjit::zeros<DiffInt32>(static_cast<size_t>(budget));
            drjit::eval(
                power,
                order,
                prefix_depth,
                inter_depth,
                suffix_depth,
                edge_idx,
                out_indices
            );
            int kept = witwin::channel::native_ext::prune_state_arrays_by_budget(
                drjit_data_ptr(power),
                drjit_data_ptr(order),
                drjit_data_ptr(prefix_depth),
                drjit_data_ptr(inter_depth),
                drjit_data_ptr(suffix_depth),
                drjit_data_ptr(edge_idx),
                edge_history_raw.data(),
                reflection_history_raw.data(),
                history_size,
                n_states,
                budget,
                drjit_data_ptr_mut(out_indices)
            );
            return nb::make_tuple(kept, out_indices);
        },
        "Sort states by the exact pruning tuple and return top-budget indices."
    );

    m.def(
        "prune_state_arrays_by_budget_pair_arrays",
        [](
            Float power,
            Int32 order,
            Int32 prefix_depth,
            Int32 inter_depth,
            Int32 suffix_depth,
            Int32 edge_idx,
            nb::list path_edge_history,
            nb::list path_reflection_history,
            int history_size,
            int n_states,
            int higher_budget,
            int inserted_budget
        ) {
            auto edge_history_ptrs = array_pointer_list(
                path_edge_history,
                "prune_state_arrays_by_budget_pair_arrays(path_edge_history)"
            );
            auto reflection_history_ptrs = array_pointer_list(
                path_reflection_history,
                "prune_state_arrays_by_budget_pair_arrays(path_reflection_history)"
            );
            if (edge_history_ptrs.size() != static_cast<size_t>(history_size)
                || reflection_history_ptrs.size() != static_cast<size_t>(history_size)) {
                throw std::runtime_error(
                    "prune_state_arrays_by_budget_pair_arrays: history pointer length mismatch"
                );
            }

            std::vector<const int*> edge_history_raw;
            std::vector<const int*> reflection_history_raw;
            edge_history_raw.reserve(edge_history_ptrs.size());
            reflection_history_raw.reserve(reflection_history_ptrs.size());
            for (size_t i = 0; i < edge_history_ptrs.size(); ++i) {
                edge_history_raw.push_back(ptr<const int>(edge_history_ptrs[i]));
                reflection_history_raw.push_back(ptr<const int>(reflection_history_ptrs[i]));
            }

            DiffInt32 out_higher_indices = drjit::zeros<DiffInt32>(
                static_cast<size_t>(higher_budget > 0 ? higher_budget : 0)
            );
            DiffInt32 out_inserted_indices = drjit::zeros<DiffInt32>(
                static_cast<size_t>(inserted_budget > 0 ? inserted_budget : 0)
            );
            drjit::eval(
                power,
                order,
                prefix_depth,
                inter_depth,
                suffix_depth,
                edge_idx,
                out_higher_indices,
                out_inserted_indices
            );
            witwin::channel::native_ext::prune_state_arrays_by_budget_pair(
                drjit_data_ptr(power),
                drjit_data_ptr(order),
                drjit_data_ptr(prefix_depth),
                drjit_data_ptr(inter_depth),
                drjit_data_ptr(suffix_depth),
                drjit_data_ptr(edge_idx),
                edge_history_raw.data(),
                reflection_history_raw.data(),
                history_size,
                n_states,
                higher_budget,
                inserted_budget,
                drjit_data_ptr_mut(out_higher_indices),
                drjit_data_ptr_mut(out_inserted_indices)
            );
            return nb::make_tuple(out_higher_indices, out_inserted_indices);
        },
        "Sort states once and return paired higher-order/inserted top-budget indices."
    );
}
