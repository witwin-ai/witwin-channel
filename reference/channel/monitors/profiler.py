"""Profiling helpers for monitor-specific tracing."""

from __future__ import annotations

import drjit as dr
import torch

from ..trace.diffraction.profiling import summarize_state_memory_profile


def capture_cuda_memory_report(*, release_reclaimable_caches: bool = False) -> dict[str, object]:
    if not torch.cuda.is_available():
        return {"available": False}
    if release_reclaimable_caches:
        if hasattr(dr, "flush_malloc_cache"):
            dr.flush_malloc_cache()
        torch.cuda.empty_cache()
    device = torch.cuda.current_device()
    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    return {
        "available": True,
        "device_index": int(device),
        "device_name": torch.cuda.get_device_name(device),
        "free_bytes": int(free_bytes),
        "total_bytes": int(total_bytes),
        "allocated_bytes": int(torch.cuda.memory_allocated(device)),
        "reserved_bytes": int(torch.cuda.memory_reserved(device)),
        "max_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "max_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
    }


def build_state_guardrail_profile(
    solver_controls,
    *,
    path_budget_report,
    fallback_history_size: int,
    final_total_states: int,
) -> dict[str, object]:
    state_memory_profile = summarize_state_memory_profile(
        path_budget_report,
        fallback_history_size=fallback_history_size,
    )
    if not state_memory_profile["final_total_states"]:
        state_memory_profile["final_total_states"] = int(final_total_states)
    peak_pre = int(state_memory_profile["peak_total_states_before_prune"])

    risk_level = "low"
    if peak_pre >= 4096:
        risk_level = "high"
    elif peak_pre >= 1024:
        risk_level = "medium"

    return {
        "applied": bool(solver_controls["changes"]),
        "changes": tuple(solver_controls["changes"]),
        "profiling": {
            **state_memory_profile,
            "risk_level": risk_level,
        },
    }


__all__ = [
    "build_state_guardrail_profile",
    "capture_cuda_memory_report",
]
