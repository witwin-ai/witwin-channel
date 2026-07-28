"""Shared native companions for exact-row compact autograd."""

from __future__ import annotations

from collections.abc import Callable

import torch

from witwin.channel.runtime import required_symbol as _required_native_op


COMPACT_CONTINUOUS_FIELDS = (
    "path_length_m",
    "delay_s",
    "field_direction",
    "interaction_position",
    "interaction_normal",
    "interaction_positions",
    "interaction_normals",
    "path_gain",
    "path_field",
    "field_xyz",
    "coefficient",
)


def _compact_autograd_companion(
    native_op: Callable[..., object],
    operation: str,
    valid: torch.Tensor,
    selected_row_index: torch.Tensor,
    continuous_values: tuple[torch.Tensor | None, ...],
    *,
    candidate_count: int,
    sequence_width: int,
) -> dict[str, torch.Tensor]:
    if len(continuous_values) != len(COMPACT_CONTINUOUS_FIELDS):
        raise ValueError("compact autograd companion requires all continuous fields")
    raw = native_op(
        valid,
        selected_row_index,
        *continuous_values,
        int(candidate_count),
        int(sequence_width),
    )
    if not isinstance(raw, dict) or set(raw) != set(COMPACT_CONTINUOUS_FIELDS):
        raise TypeError(f"native compact {operation} returned bad fields")
    return raw


def evaluated_paths_compact_finalize_backward(
    valid: torch.Tensor,
    selected_row_index: torch.Tensor,
    *gradients: torch.Tensor | None,
    candidate_count: int,
    sequence_width: int,
) -> dict[str, torch.Tensor]:
    """Scatter compact continuous cotangents to candidate rows."""

    return _compact_autograd_companion(
        _required_native_op("evaluated_paths_compact_finalize_backward"),
        "backward",
        valid,
        selected_row_index,
        gradients,
        candidate_count=candidate_count,
        sequence_width=sequence_width,
    )


def evaluated_paths_compact_finalize_jvp(
    valid: torch.Tensor,
    selected_row_index: torch.Tensor,
    *tangents: torch.Tensor | None,
    candidate_count: int,
    sequence_width: int,
) -> dict[str, torch.Tensor]:
    """Gather candidate continuous tangents into exact compact rows."""

    return _compact_autograd_companion(
        _required_native_op("evaluated_paths_compact_finalize_jvp"),
        "JVP",
        valid,
        selected_row_index,
        tangents,
        candidate_count=candidate_count,
        sequence_width=sequence_width,
    )


__all__ = [
    "COMPACT_CONTINUOUS_FIELDS",
    "evaluated_paths_compact_finalize_backward",
    "evaluated_paths_compact_finalize_jvp",
]
