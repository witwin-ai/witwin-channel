"""Enumerated line-of-sight topology discovery."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from witwin.channel_native.propagation.geometry.visibility import (
    VisibilityQuery,
    run_visibility_query,
)
from witwin.channel_native.propagation.topology.discovery.los import (
    prepare_los_candidates,
)
from witwin.channel_native.propagation.topology.export import _ensure_topology_fields
from witwin.channel_native.propagation.topology.kernels import blocks as topology_blocks
from witwin.channel_native.propagation.topology.kernels import (
    construction as topology_construction,
)

if TYPE_CHECKING:
    from witwin.channel_native.scene.models import Scene


def _los_topology(
    scene: Scene,
    compiled: object,
    tx_positions: torch.Tensor,
    tx_power: torch.Tensor,
    rx_positions: torch.Tensor,
    tx_polarizations: torch.Tensor,
    *,
    frequency_hz: float,
    sequence_width: int,
) -> tuple[dict[str, torch.Tensor], int, int, int]:
    # R5: the per-transmitter polarization (threaded from the caller) drives the
    # LoS dipole sin^2 pattern in path_los_export.
    exported = topology_blocks.path_los_export(
        tx_positions,
        tx_power,
        rx_positions,
        tx_polarizations,
        frequency_hz=frequency_hz,
    )
    launch_count = 1
    plan = prepare_los_candidates(
        tx_id=exported["tx_id"],
        rx_id=exported["rx_id"],
        sequence_width=sequence_width,
    )
    visible = None
    if bool(scene.structures) and plan.candidate_count > 0:
        visibility_inputs = topology_blocks.path_los_visibility_inputs(
            tx_positions,
            rx_positions,
            plan.tx_id.to(dtype=torch.int32).contiguous(),
            plan.rx_id.to(dtype=torch.int32).contiguous(),
        )
        visible = run_visibility_query(
            VisibilityQuery(
                raydn=compiled.raydn,
                start=visibility_inputs["start"],
                end=visibility_inputs["end"],
                active=visibility_inputs["active"],
            )
        ).visible
        launch_count += 1
    los_block = _ensure_topology_fields(
        topology_construction.deterministic_los_topology_block(
            plan.tx_id.to(dtype=torch.int32).contiguous(),
            plan.rx_id.to(dtype=torch.int32).contiguous(),
            exported["path_length_m"].to(dtype=torch.float32).contiguous(),
            exported["delay_s"].to(dtype=torch.float32).contiguous(),
            exported["path_gain"].to(dtype=torch.float32).contiguous(),
            visible,
            frequency_hz=frequency_hz,
            sequence_width=plan.sequence_width,
        )
    )
    visibility_rejection_count = 0
    if visible is not None:
        visibility_rejection_count += plan.candidate_count - int(
            los_block["valid"].numel()
        )
    return (
        los_block,
        launch_count,
        plan.candidate_count,
        visibility_rejection_count,
    )
