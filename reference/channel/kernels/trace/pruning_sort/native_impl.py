"""
Native C++/CUDA implementation of the pruning sort.

The native path sorts state indices directly on the GPU with the full
lexicographic pruning tuple, then gathers the kept states.
"""

from __future__ import annotations

import drjit as dr

import witwin as wt
from witwin.channel._native import _extension
from witwin.channel.kernels.trace.pruning_sort.drjit_impl import (
    _state_pruning_metric,
)
from witwin.channel.trace.diffraction.constants import _state_history_size


def _pruning_report(n_states: int, budget_name: str, requested_budget: int | None) -> dict:
    return {
        "budget_name": budget_name,
        "requested_budget": None if requested_budget is None else int(requested_budget),
        "pruning_metric": "incident_power",
        "input_states": n_states,
        "kept_states": n_states,
        "dropped_states": 0,
        "applied": False,
    }


def _prepare_native_pruning_inputs(state_arrays: dict):
    from witwin.channel.trace.diffraction.state.arrays import _materialize_state_history

    history_size = _state_history_size(state_arrays)
    n_states = 0 if state_arrays is None else int(state_arrays["n_states"])
    power = _state_pruning_metric(state_arrays)
    order = wt.Int32(state_arrays["order"])
    prefix_d = wt.Int32(state_arrays["prefix_reflection_depth"])
    inter_d = wt.Int32(state_arrays["intermediate_reflection_depth"])
    suffix_d = wt.Int32(state_arrays["suffix_reflection_depth"])
    edge_idx = wt.Int32(state_arrays["edge_idx"])
    materialized_edge_history, materialized_reflection_history = _materialize_state_history(
        state_arrays
    )
    path_edge_history = [
        wt.Int32(materialized_edge_history[slot])
        for slot in range(min(history_size, len(materialized_edge_history)))
    ]
    path_reflection_history = [
        wt.Int32(materialized_reflection_history[slot])
        for slot in range(min(history_size, len(materialized_reflection_history)))
    ]
    while len(path_edge_history) < history_size:
        path_edge_history.append(dr.full(wt.Int32, -1, n_states))
    while len(path_reflection_history) < history_size:
        path_reflection_history.append(dr.zeros(wt.Int32, n_states))
    return (
        history_size,
        n_states,
        power,
        order,
        prefix_d,
        inter_d,
        suffix_d,
        edge_idx,
        path_edge_history,
        path_reflection_history,
    )


def prune_state_arrays_by_budget(
    state_arrays: dict,
    max_states: int | None,
    budget_name: str = "default",
) -> tuple[dict, dict]:
    """
    Native CUDA path for budget-based state pruning.

    Same signature as ``drjit_impl.prune_state_arrays_by_budget``.
    """
    from witwin.channel.trace.diffraction.state.arrays import (
        _empty_state_arrays,
        _take_state_arrays,
    )

    ext = _extension()
    history_size = _state_history_size(state_arrays)
    n_states = 0 if state_arrays is None else int(state_arrays["n_states"])
    report = _pruning_report(n_states, budget_name, max_states)

    if state_arrays is None or n_states == 0:
        return _empty_state_arrays(history_size=history_size), report
    if max_states is None or int(max_states) < 0 or n_states <= int(max_states):
        return state_arrays, report
    if int(max_states) == 0:
        report["kept_states"] = 0
        report["dropped_states"] = n_states
        report["applied"] = True
        return _empty_state_arrays(history_size=history_size), report

    budget = int(max_states)
    (
        history_size,
        n_states,
        power,
        order,
        prefix_d,
        inter_d,
        suffix_d,
        edge_idx,
        path_edge_history,
        path_reflection_history,
    ) = _prepare_native_pruning_inputs(state_arrays)

    dr.eval(
        power,
        order,
        prefix_d,
        inter_d,
        suffix_d,
        edge_idx,
        *path_edge_history,
        *path_reflection_history,
    )

    kept, out_indices = ext.prune_state_arrays_by_budget_arrays(
        power,
        order,
        prefix_d,
        inter_d,
        suffix_d,
        edge_idx,
        path_edge_history,
        path_reflection_history,
        history_size,
        n_states,
        budget,
    )

    keep_idx = wt.UInt32(dr.gather(wt.Int32, out_indices, dr.arange(wt.UInt32, kept)))
    pruned = _take_state_arrays(state_arrays, keep_idx)
    report["kept_states"] = int(pruned["n_states"])
    report["dropped_states"] = n_states - int(pruned["n_states"])
    report["applied"] = True
    return pruned, report


def prune_state_arrays_by_budget_pair(
    state_arrays: dict,
    higher_budget: int | None,
    inserted_budget: int | None,
    *,
    higher_budget_name: str = "higher_budget",
    inserted_budget_name: str = "inserted_budget",
):
    from witwin.channel.trace.diffraction.state.arrays import (
        _empty_state_arrays,
        _take_state_arrays,
    )

    ext = _extension()
    history_size = _state_history_size(state_arrays)
    n_states = 0 if state_arrays is None else int(state_arrays["n_states"])
    higher_report = _pruning_report(n_states, higher_budget_name, higher_budget)
    inserted_report = _pruning_report(n_states, inserted_budget_name, inserted_budget)
    higher_report["paired_pre_expansion_sort"] = False
    inserted_report["paired_pre_expansion_sort"] = False

    if state_arrays is None or n_states == 0:
        empty = _empty_state_arrays(history_size=history_size)
        return empty, higher_report, empty, inserted_report

    def _should_skip(budget):
        return budget is None or int(budget) < 0 or n_states <= int(budget)

    if _should_skip(higher_budget) and _should_skip(inserted_budget):
        return state_arrays, higher_report, state_arrays, inserted_report

    if higher_budget is not None and int(higher_budget) == 0 and inserted_budget is not None and int(inserted_budget) == 0:
        empty = _empty_state_arrays(history_size=history_size)
        higher_report["kept_states"] = 0
        higher_report["dropped_states"] = n_states
        higher_report["applied"] = True
        higher_report["paired_pre_expansion_sort"] = True
        inserted_report["kept_states"] = 0
        inserted_report["dropped_states"] = n_states
        inserted_report["applied"] = True
        inserted_report["paired_pre_expansion_sort"] = True
        return empty, higher_report, empty, inserted_report

    if higher_budget == inserted_budget:
        pruned, report = prune_state_arrays_by_budget(
            state_arrays,
            higher_budget,
            higher_budget_name,
        )
        report["paired_pre_expansion_sort"] = True
        paired_report = dict(report)
        paired_report["budget_name"] = inserted_budget_name
        return pruned, report, pruned, paired_report

    (
        history_size,
        n_states,
        power,
        order,
        prefix_d,
        inter_d,
        suffix_d,
        edge_idx,
        path_edge_history,
        path_reflection_history,
    ) = _prepare_native_pruning_inputs(state_arrays)

    higher_keep = 0 if higher_budget is None or int(higher_budget) <= 0 else min(n_states, int(higher_budget))
    inserted_keep = 0 if inserted_budget is None or int(inserted_budget) <= 0 else min(n_states, int(inserted_budget))
    dr.eval(
        power,
        order,
        prefix_d,
        inter_d,
        suffix_d,
        edge_idx,
        *path_edge_history,
        *path_reflection_history,
    )

    out_higher, out_inserted = ext.prune_state_arrays_by_budget_pair_arrays(
        power,
        order,
        prefix_d,
        inter_d,
        suffix_d,
        edge_idx,
        path_edge_history,
        path_reflection_history,
        history_size,
        n_states,
        higher_keep,
        inserted_keep,
    )

    def _take_budget(keep, out_indices, report):
        if keep == 0:
            report["kept_states"] = 0
            report["dropped_states"] = n_states
            report["applied"] = True
            report["paired_pre_expansion_sort"] = True
            return _empty_state_arrays(history_size=history_size), report
        keep_idx = wt.UInt32(dr.gather(wt.Int32, out_indices, dr.arange(wt.UInt32, keep)))
        pruned = _take_state_arrays(state_arrays, keep_idx)
        report["kept_states"] = int(pruned["n_states"])
        report["dropped_states"] = n_states - int(pruned["n_states"])
        report["applied"] = True
        report["paired_pre_expansion_sort"] = True
        return pruned, report

    higher_state, higher_report = (
        (state_arrays, higher_report)
        if _should_skip(higher_budget)
        else _take_budget(higher_keep, out_higher, higher_report)
    )
    inserted_state, inserted_report = (
        (state_arrays, inserted_report)
        if _should_skip(inserted_budget)
        else _take_budget(inserted_keep, out_inserted, inserted_report)
    )
    return higher_state, higher_report, inserted_state, inserted_report
