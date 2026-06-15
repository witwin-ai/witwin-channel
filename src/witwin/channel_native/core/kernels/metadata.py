from __future__ import annotations

from collections.abc import Mapping


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
    "raydn_native",
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
    raydn_native: bool = False,
    ad_status: str = "none",
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
        "raydn_native": raydn_native,
        "ad_status": ad_status,
    }
    validate_metadata(metadata)
    return metadata


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

    if not isinstance(metadata["raydn_native"], bool):
        raise ValueError("metadata raydn_native must be a boolean")
