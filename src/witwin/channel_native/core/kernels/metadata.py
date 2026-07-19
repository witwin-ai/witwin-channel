from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import torch


@dataclass
class AdLaunchLedger:
    """Per-solve accounting of the plan 07 AD companion kernels.

    ``launches`` counts the native backward/jvp companion launches one full
    reverse pass (vjp) or forward-dual pass (jvp) performs for this solve:
    one per registered differentiable Function. ``tape_bytes`` sums the
    tensors the reverse pass retains via ``save_for_backward``; forward mode
    retains nothing past the solve, so jvp reports zero tape. One ledger
    shape for every solver (montecarlo.basic, deterministic, path).
    """

    launches: int = 0
    tape_bytes: int = 0

    def add(self, *saved: object) -> None:
        self.launches += 1
        for tensor in saved:
            if isinstance(tensor, torch.Tensor):
                self.tape_bytes += tensor.numel() * tensor.element_size()


ACCUMULATION_STRATEGIES = frozenset(
    {
        "none",
        "atomic_add",
        "cell_reduce",
        "compact_atomic_add",
        "sorted_segment_reduce",
        "shared_memory_private_reduce",
        "hybrid_tile_reduce",
    }
)

AD_STATUSES = frozenset({"none", "primal", "vjp", "jvp"})

REQUIRED_METADATA_FIELDS = (
    "primitive",
    "launch_count",
    "forward_launch_count",
    "backward_launch_count",
    "jvp_launch_count",
    "intermediate_bytes",
    "tape_bytes",
    "fused_stages",
    "accumulation_strategy",
    "scheduling_strategy",
    "registers_per_thread",
    "shared_memory_bytes",
    "occupancy_estimate",
    "spill_bytes",
    "rayd_native",
    "ad_status",
)


def make_metadata(
    *,
    primitive: str,
    forward_launch_count: int = 0,
    backward_launch_count: int = 0,
    jvp_launch_count: int = 0,
    intermediate_bytes: int = 0,
    tape_bytes: int = 0,
    fused_stages: int = 0,
    accumulation_strategy: str = "none",
    scheduling_strategy: str = "none",
    registers_per_thread: int = 0,
    shared_memory_bytes: int = 0,
    occupancy_estimate: float = 0.0,
    spill_bytes: int = 0,
    rayd_native: bool = False,
    ad_status: str = "none",
    forward_time_ms: float = 0.0,
    peak_memory_bytes: int = 0,
) -> dict[str, bool | float | int | str]:
    metadata: dict[str, bool | float | int | str] = {
        "primitive": primitive,
        "launch_count": forward_launch_count + backward_launch_count + jvp_launch_count,
        "forward_launch_count": forward_launch_count,
        "backward_launch_count": backward_launch_count,
        "jvp_launch_count": jvp_launch_count,
        "intermediate_bytes": intermediate_bytes,
        "tape_bytes": tape_bytes,
        "fused_stages": fused_stages,
        "accumulation_strategy": accumulation_strategy,
        "scheduling_strategy": scheduling_strategy,
        "registers_per_thread": registers_per_thread,
        "shared_memory_bytes": shared_memory_bytes,
        "occupancy_estimate": occupancy_estimate,
        "spill_bytes": spill_bytes,
        "rayd_native": rayd_native,
        "ad_status": ad_status,
        # Wall-clock (CUDA-synchronized) solve duration and the amount the
        # solve raised the process CUDA high-water mark. A jvp solve carries
        # its dual pass inside this forward time; a vjp solve cannot observe
        # its future backward, so reverse-pass time/memory budgets are pinned
        # by the tests/ad overhead gates instead of a metadata field.
        "forward_time_ms": float(forward_time_ms),
        "peak_memory_bytes": int(peak_memory_bytes),
    }
    validate_metadata(metadata)
    return metadata


def noop_metadata(
    *, accumulation_strategy: str = "none"
) -> dict[str, bool | float | int | str]:
    return make_metadata(
        primitive="noop_metadata",
        accumulation_strategy=accumulation_strategy,
        scheduling_strategy="none",
        ad_status="none",
    )


def validate_metadata(metadata: Mapping[str, object]) -> None:
    for field in REQUIRED_METADATA_FIELDS:
        if field not in metadata:
            raise ValueError(f"metadata missing required field: {field}")

    if not isinstance(metadata["primitive"], str) or not metadata["primitive"]:
        raise ValueError("metadata primitive must be a non-empty string")

    for field in (
        "launch_count",
        "forward_launch_count",
        "backward_launch_count",
        "jvp_launch_count",
        "intermediate_bytes",
        "tape_bytes",
        "fused_stages",
        "registers_per_thread",
        "shared_memory_bytes",
        "spill_bytes",
    ):
        value = metadata[field]
        if not isinstance(value, int) or value < 0:
            raise ValueError(f"metadata {field} must be a non-negative integer")

    expected_launch_count = (
        metadata["forward_launch_count"]
        + metadata["backward_launch_count"]
        + metadata["jvp_launch_count"]
    )
    if metadata["launch_count"] != expected_launch_count:
        raise ValueError("metadata launch_count must equal forward+backward+jvp counts")

    if metadata["accumulation_strategy"] not in ACCUMULATION_STRATEGIES:
        raise ValueError("metadata accumulation_strategy is not recognized")

    if metadata["ad_status"] not in AD_STATUSES:
        raise ValueError("metadata ad_status is not recognized")

    if not isinstance(metadata["scheduling_strategy"], str):
        raise ValueError("metadata scheduling_strategy must be a string")

    occupancy = metadata["occupancy_estimate"]
    if not isinstance(occupancy, int | float) or occupancy < 0.0:
        raise ValueError("metadata occupancy_estimate must be non-negative")

    if not isinstance(metadata["rayd_native"], bool):
        raise ValueError("metadata rayd_native must be a boolean")

    for field in ("forward_time_ms", "peak_memory_bytes"):
        value = metadata.get(field, 0)
        if not isinstance(value, int | float) or value < 0:
            raise ValueError(f"metadata {field} must be non-negative")
