"""
Pure-DrJit/Torch reference implementation of pruning sort.

Interface matches pruning_sort/pruning_sort.h.

The C++ kernel uses CUB DeviceRadixSort with a 64-bit composite key.
This DrJit path uses the existing torch_lexsort round-trip as baseline.
"""

from __future__ import annotations

import torch
import drjit as dr
from witwin.channel.deterministic import types as wt

from witwin.channel.core.numerics.tensors import to_torch_view
from witwin.channel.core.numerics.arrays import complex_abs_sqr
from witwin.channel.deterministic.diffraction.state import Geo


def _torch_lexsort(keys):
    if len(keys) == 0:
        return torch.zeros(0, dtype=torch.int64)
    order = torch.arange(keys[0].shape[0], device=keys[0].device, dtype=torch.int64)
    for key in reversed(keys):
        order = order.index_select(0, torch.argsort(key.index_select(0, order), stable=True))
    return order


def _state_pruning_metric(state_arrays):
    if state_arrays is None or state_arrays["n_states"] == 0:
        return dr.zeros(wt.Float, 0)
    return (
        complex_abs_sqr(state_arrays["incident_jones_u"])
        + complex_abs_sqr(state_arrays["incident_jones_v"])
        + complex_abs_sqr(state_arrays["incident_derivative_jones_u"])
        + complex_abs_sqr(state_arrays["incident_derivative_jones_v"])
    )


def _prepare_pruning_sort_inputs(state_arrays):
    from witwin.channel.deterministic.diffraction.state import (
        _materialize_state_history,
        _torch_state_key,
    )

    history_size = Geo.history_size(state_arrays)
    n_states = 0 if state_arrays is None else int(state_arrays["n_states"])
    metric = to_torch_view(_state_pruning_metric(state_arrays), detach=True)
    order = _torch_state_key(state_arrays, "order")
    prefix_depth = _torch_state_key(state_arrays, "prefix_reflection_depth")
    intermediate_depth = _torch_state_key(state_arrays, "intermediate_reflection_depth")
    suffix_depth = _torch_state_key(state_arrays, "suffix_reflection_depth")
    edge_idx = _torch_state_key(state_arrays, "edge_idx")
    path_edge_history, path_reflection_history = _materialize_state_history(state_arrays)
    edge_history = [
        to_torch_view(path_edge_history[slot], detach=True).to(dtype=torch.int32)
        for slot in range(min(history_size, len(path_edge_history)))
    ]
    reflection_history = [
        to_torch_view(path_reflection_history[slot], detach=True).to(dtype=torch.int32)
        for slot in range(min(history_size, len(path_reflection_history)))
    ]
    while len(edge_history) < history_size:
        edge_history.append(torch.full_like(edge_idx, -1))
    while len(reflection_history) < history_size:
        reflection_history.append(torch.zeros_like(edge_idx))

    sort_keys = [
        -metric,
        order,
        prefix_depth,
        intermediate_depth,
        suffix_depth,
        *edge_history,
        *reflection_history,
        edge_idx,
        torch.arange(n_states, device=metric.device, dtype=torch.int32),
    ]
    return _torch_lexsort(sort_keys), history_size, n_states


def prune_state_arrays_by_budget(
    state_arrays: dict,
    max_states: int | None,
    budget_name: str = "default",
) -> tuple[dict, dict]:
    """
    Prune states to a budget using lexicographic power-based ranking.

    Sort keys (highest to lowest priority):
      1. Incident power (descending)
      2. Diffraction order (ascending)
      3. Prefix / intermediate / suffix reflection depths (ascending)
      4. Edge history + edge index (tiebreaker)

    Parameters
    ----------
    state_arrays : dict
        Full SoA state dictionary.
    max_states : int or None
        Maximum states to keep. None = no pruning.
    budget_name : str
        Label for the pruning report.

    Returns
    -------
    pruned : dict
        State arrays truncated to at most ``max_states``.
    report : dict
        Pruning statistics (input_states, kept_states, etc.).
    """
    from witwin.channel.deterministic.diffraction.state import (
        _empty_state_arrays,
        _take_state_arrays,
    )

    history_size = Geo.history_size(state_arrays)
    n_states = 0 if state_arrays is None else int(state_arrays["n_states"])
    report = {
        "budget_name": budget_name,
        "requested_budget": None if max_states is None else int(max_states),
        "pruning_metric": "incident_power",
        "input_states": n_states,
        "kept_states": n_states,
        "dropped_states": 0,
        "applied": False,
    }
    if state_arrays is None or n_states == 0:
        return _empty_state_arrays(history_size=history_size), report
    if max_states is None or int(max_states) < 0 or n_states <= int(max_states):
        return state_arrays, report
    if int(max_states) == 0:
        report["kept_states"] = 0
        report["dropped_states"] = n_states
        report["applied"] = True
        return _empty_state_arrays(history_size=history_size), report

    ranked, _, _ = _prepare_pruning_sort_inputs(state_arrays)
    keep_indices = torch.sort(ranked[:int(max_states)]).values.to(dtype=torch.int32)
    keep_idx = wt.UInt32(keep_indices)
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
    from witwin.channel.deterministic.diffraction.state import (
        _empty_state_arrays,
        _take_state_arrays,
    )

    history_size = Geo.history_size(state_arrays)
    n_states = 0 if state_arrays is None else int(state_arrays["n_states"])

    def _report(name, requested):
        return {
            "budget_name": name,
            "requested_budget": None if requested is None else int(requested),
            "pruning_metric": "incident_power",
            "input_states": n_states,
            "kept_states": n_states,
            "dropped_states": 0,
            "applied": False,
            "paired_pre_expansion_sort": False,
        }

    higher_report = _report(higher_budget_name, higher_budget)
    inserted_report = _report(inserted_budget_name, inserted_budget)
    if state_arrays is None or n_states == 0:
        empty = _empty_state_arrays(history_size=history_size)
        return empty, higher_report, empty, inserted_report

    def _should_skip(budget):
        return budget is None or int(budget) < 0 or n_states <= int(budget)

    if _should_skip(higher_budget) and _should_skip(inserted_budget):
        return state_arrays, higher_report, state_arrays, inserted_report

    ranked, _, _ = _prepare_pruning_sort_inputs(state_arrays)

    def _take_budget(budget, report):
        if budget is None or int(budget) < 0 or n_states <= int(budget):
            return state_arrays, report
        if int(budget) == 0:
            report["kept_states"] = 0
            report["dropped_states"] = n_states
            report["applied"] = True
            report["paired_pre_expansion_sort"] = True
            return _empty_state_arrays(history_size=history_size), report
        keep_indices = torch.sort(ranked[: int(budget)]).values.to(dtype=torch.int32)
        pruned = _take_state_arrays(state_arrays, wt.UInt32(keep_indices))
        report["kept_states"] = int(pruned["n_states"])
        report["dropped_states"] = n_states - int(pruned["n_states"])
        report["applied"] = True
        report["paired_pre_expansion_sort"] = True
        return pruned, report

    higher_state, higher_report = _take_budget(higher_budget, higher_report)
    inserted_state, inserted_report = _take_budget(inserted_budget, inserted_report)
    return higher_state, higher_report, inserted_state, inserted_report

