"""Shared profiling helpers for diffraction state memory and builder reports."""

from __future__ import annotations

from typing import Any, Mapping


PACKED_CORE_FLOATS = 72
PACKED_ALIGNMENT_FLOATS = 4
FLOAT32_BYTES = 4
LINEAGE_PARENT_LINK_BYTES_PER_STATE = 12
COLD_METADATA_BYTES_PER_STATE = 28


def packed_state_stride(history_size: int) -> int:
    del history_size
    raw = PACKED_CORE_FLOATS
    return (raw + (PACKED_ALIGNMENT_FLOATS - 1)) & ~(PACKED_ALIGNMENT_FLOATS - 1)


def packed_state_layout_metrics(
    history_size: int,
    *,
    cold_metadata_retained: bool = True,
) -> dict[str, int | bool | str]:
    resolved_history_size = max(0, int(history_size))
    stride_floats = packed_state_stride(resolved_history_size)
    hot_bytes_per_state = int(stride_floats * FLOAT32_BYTES)
    lineage_bytes_per_state = (
        int(LINEAGE_PARENT_LINK_BYTES_PER_STATE) if cold_metadata_retained else 0
    )
    cold_bytes_per_state = (
        int(COLD_METADATA_BYTES_PER_STATE) if cold_metadata_retained else 0
    )
    total_bytes_per_state = hot_bytes_per_state + lineage_bytes_per_state + cold_bytes_per_state
    return {
        "history_size": resolved_history_size,
        "lineage_history_slots": resolved_history_size,
        "lineage_mode": "parent_link",
        "cold_metadata_retained": bool(cold_metadata_retained),
        "packed_core_floats": PACKED_CORE_FLOATS,
        "packed_alignment_floats": PACKED_ALIGNMENT_FLOATS,
        "packed_stride_floats": stride_floats,
        "packed_state_stride_floats": stride_floats,
        "packed_stride_bytes": hot_bytes_per_state,
        "packed_state_stride_bytes": hot_bytes_per_state,
        "hot_bytes_per_state": hot_bytes_per_state,
        "lineage_bytes_per_state": lineage_bytes_per_state,
        "cold_metadata_bytes_per_state": cold_bytes_per_state,
        "bytes_per_state": total_bytes_per_state,
        "bytes_per_state_estimate": total_bytes_per_state,
    }


def _builder_max_pairs_per_chunk(*builders: Mapping[str, Any] | None) -> int:
    peak = 0
    for builder in builders:
        if not isinstance(builder, Mapping):
            continue
        peak = max(peak, int(builder.get("max_candidate_pairs_per_chunk", 0)))
    return int(peak)


def summarize_state_memory_profile(
    path_budget_report: Mapping[str, Any] | None,
    *,
    fallback_history_size: int = 0,
) -> dict[str, Any]:
    report = {} if path_budget_report is None else dict(path_budget_report)
    history_size = int(report.get("history_size", fallback_history_size))
    layout = packed_state_layout_metrics(
        history_size,
        cold_metadata_retained=bool(report.get("cold_metadata_retained", True)),
    )
    per_order = list(report.get("per_order", []))

    peak_total_states_before_prune = int(report.get("peak_total_states_before_prune", 0))
    peak_total_states_after_prune = int(report.get("peak_total_states_after_prune", 0))
    final_total_states = int(report.get("final_total_states", 0))
    pre_expansion_policy = dict(report.get("pre_expansion_policy", {}) or {})

    estimated_peak_state_bytes_before_prune = int(
        peak_total_states_before_prune * layout["bytes_per_state"]
    )
    estimated_peak_state_bytes_after_prune = int(
        peak_total_states_after_prune * layout["bytes_per_state"]
    )

    per_order_summary = []
    max_cartesian_pairs_per_chunk = 0
    peak_higher_order_source_before_pre_prune = 0
    peak_higher_order_source_after_pre_prune = 0
    peak_inserted_source_before_pre_prune = 0
    peak_inserted_source_after_pre_prune = 0
    for item in per_order:
        if not isinstance(item, Mapping):
            continue
        reflection_prefix_builder = item.get("reflection_prefix_builder")
        higher_order_builder = item.get("higher_order_builder")
        order_peak_pairs = _builder_max_pairs_per_chunk(
            reflection_prefix_builder,
            higher_order_builder,
        )
        max_cartesian_pairs_per_chunk = max(max_cartesian_pairs_per_chunk, order_peak_pairs)
        peak_higher_order_source_before_pre_prune = max(
            peak_higher_order_source_before_pre_prune,
            int(item.get("higher_order_source_states_before_pre_prune", 0)),
        )
        peak_higher_order_source_after_pre_prune = max(
            peak_higher_order_source_after_pre_prune,
            int(item.get("higher_order_source_states_after_pre_prune", 0)),
        )
        peak_inserted_source_before_pre_prune = max(
            peak_inserted_source_before_pre_prune,
            int(item.get("inserted_source_states_before_pre_prune", 0)),
        )
        peak_inserted_source_after_pre_prune = max(
            peak_inserted_source_after_pre_prune,
            int(item.get("inserted_source_states_after_pre_prune", 0)),
        )
        per_order_summary.append(
            {
                "order": int(item.get("order", 0)),
                "higher_order_source_states_before_pre_prune": int(
                    item.get("higher_order_source_states_before_pre_prune", 0)
                ),
                "higher_order_source_states_after_pre_prune": int(
                    item.get("higher_order_source_states_after_pre_prune", 0)
                ),
                "inserted_source_states_before_pre_prune": int(
                    item.get("inserted_source_states_before_pre_prune", 0)
                ),
                "inserted_source_states_after_pre_prune": int(
                    item.get("inserted_source_states_after_pre_prune", 0)
                ),
                "total_states_before_prune": int(item.get("total_states_before_prune", 0)),
                "total_states_after_prune": int(item.get("total_states_after_prune", 0)),
                "inserted_states_before_prune": int(item.get("inserted_states_before_prune", 0)),
                "inserted_states_after_prune": int(item.get("inserted_states_after_prune", 0)),
                "max_cartesian_pairs_per_chunk": int(order_peak_pairs),
            }
        )

    return {
        **layout,
        "peak_total_states_before_prune": peak_total_states_before_prune,
        "peak_total_states_after_prune": peak_total_states_after_prune,
        "pre_expansion_policy": pre_expansion_policy,
        "peak_higher_order_source_states_before_pre_prune": int(
            peak_higher_order_source_before_pre_prune
        ),
        "peak_higher_order_source_states_after_pre_prune": int(
            peak_higher_order_source_after_pre_prune
        ),
        "peak_inserted_source_states_before_pre_prune": int(
            peak_inserted_source_before_pre_prune
        ),
        "peak_inserted_source_states_after_pre_prune": int(
            peak_inserted_source_after_pre_prune
        ),
        "final_total_states": final_total_states,
        "estimated_peak_state_bytes_before_prune": estimated_peak_state_bytes_before_prune,
        "estimated_peak_state_megabytes_before_prune": float(
            estimated_peak_state_bytes_before_prune / (1024.0 * 1024.0)
        ),
        "estimated_peak_state_bytes_after_prune": estimated_peak_state_bytes_after_prune,
        "estimated_peak_state_megabytes_after_prune": float(
            estimated_peak_state_bytes_after_prune / (1024.0 * 1024.0)
        ),
        "max_cartesian_pairs_per_chunk": int(max_cartesian_pairs_per_chunk),
        "per_order": tuple(per_order_summary),
    }


__all__ = [
    "FLOAT32_BYTES",
    "PACKED_ALIGNMENT_FLOATS",
    "PACKED_CORE_FLOATS",
    "packed_state_layout_metrics",
    "packed_state_stride",
    "summarize_state_memory_profile",
]
