"""Budget-based pruning helpers for diffraction state arrays."""

from __future__ import annotations

from witwin.channel.kernels.trace.pruning_sort import (  # noqa: F401
    _state_pruning_metric,
    prune_state_arrays_by_budget as _prune_state_arrays_by_budget,
)

_PRE_EXPANSION_SOURCE_DIVISOR = 4


def _reduced_budget(base_budget: int | None, *, divisor: int) -> int | None:
    if base_budget is None:
        return None
    budget = max(0, int(base_budget))
    if budget == 0:
        return 0
    divisor = max(1, int(divisor))
    return max(1, (budget + divisor - 1) // divisor)


def _bounded_source_budget(
    total_state_budget_per_order: int | None,
    *,
    inserted_state_budget_per_order: int | None = None,
) -> int | None:
    budgets = []
    reduced_total_budget = _reduced_budget(
        total_state_budget_per_order,
        divisor=_PRE_EXPANSION_SOURCE_DIVISOR,
    )
    if reduced_total_budget is not None:
        budgets.append(int(reduced_total_budget))
    if inserted_state_budget_per_order is not None:
        budgets.append(max(0, int(inserted_state_budget_per_order)))
    if not budgets:
        return None
    return min(budgets)


def resolve_pre_expansion_pruning_policy(
    *,
    solver_mode: str,
    memory_profile: str,
    total_state_budget_per_order: int | None,
    inserted_state_budget_per_order: int | None,
) -> dict[str, object]:
    """Resolve Phase 5 source-side pruning before expensive expansion stages."""

    bounded_mode = str(solver_mode) == "fast_approximate" or str(memory_profile) == "memory_safe"
    if not bounded_mode:
        return {
            "enabled": False,
            "policy": "disabled",
            "higher_order_source_budget": None,
            "inserted_source_budget": None,
            "source_budget_divisor": int(_PRE_EXPANSION_SOURCE_DIVISOR),
            "reason": (
                "Accuracy mode with the default memory profile does not apply automatic "
                "pre-expansion pruning."
            ),
        }

    higher_order_source_budget = _bounded_source_budget(
        total_state_budget_per_order,
    )
    inserted_source_budget = _bounded_source_budget(
        total_state_budget_per_order,
        inserted_state_budget_per_order=inserted_state_budget_per_order,
    )

    if str(memory_profile) == "memory_safe":
        policy = "memory_safe_topk_power"
        reason = (
            "Memory-safe mode prunes weak source states before higher-order and inserted-reflection "
            "expansion to reduce candidate growth."
        )
    else:
        policy = "fast_approximate_topk_power"
        reason = (
            "Fast approximate mode prunes weak source states before higher-order and inserted-reflection "
            "expansion to keep Cartesian growth bounded."
        )

    return {
        "enabled": bool(
            higher_order_source_budget is not None or inserted_source_budget is not None
        ),
        "policy": policy,
        "higher_order_source_budget": higher_order_source_budget,
        "inserted_source_budget": inserted_source_budget,
        "source_budget_divisor": int(_PRE_EXPANSION_SOURCE_DIVISOR),
        "reason": reason,
    }


def prune_state_arrays_for_pre_expansion(
    state_arrays: dict,
    max_states: int | None,
    *,
    budget_name: str,
    policy: str,
) -> tuple[dict, dict]:
    """Apply a Phase 5 source-side pruning budget and annotate the report."""

    pruned, report = _prune_state_arrays_by_budget(
        state_arrays,
        max_states,
        budget_name,
    )
    report = dict(report)
    report["stage"] = "pre_expansion"
    report["policy"] = str(policy)
    return pruned, report


__all__ = [
    "_prune_state_arrays_by_budget",
    "_state_pruning_metric",
    "prune_state_arrays_for_pre_expansion",
    "resolve_pre_expansion_pruning_policy",
]
