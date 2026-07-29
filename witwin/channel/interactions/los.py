# Copyright Xingyu Chen.
# Line-of-sight: discrete candidate planning and enumerated orchestration.

"""Line-of-sight: discrete candidate planning and enumerated orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from witwin.channel.propagation.geometry import (
    los_clearance_factor,
    occluder_boxes,
)
from witwin.channel.propagation.geometry import (
    VisibilityQuery,
    run_visibility_query,
)
from witwin.channel.propagation.topology import _ensure_topology_fields
from witwin.channel.kernels import topology as topology_kernels

if TYPE_CHECKING:
    from witwin.channel.scene.endpoints import SolverScene as Scene


@dataclass(frozen=True, slots=True)
class LosCandidatePlan:
    tx_id: torch.Tensor
    rx_id: torch.Tensor
    sequence_width: int
    candidate_count: int


def prepare_los_candidates(
    *, tx_id: torch.Tensor, rx_id: torch.Tensor, sequence_width: int,
) -> LosCandidatePlan:
    return LosCandidatePlan(
        tx_id=tx_id,
        rx_id=rx_id,
        sequence_width=int(sequence_width),
        candidate_count=int(tx_id.numel()),
    )


def _los_topology(
    scene: Scene, compiled: object, tx_positions: torch.Tensor, tx_power: torch.Tensor,
    rx_positions: torch.Tensor, tx_polarizations: torch.Tensor, *, frequency_hz: float,
    sequence_width: int, isb_boundary_taper: bool = False, isb_boundary_taper_width: float = 0.5,
) -> tuple[dict[str, torch.Tensor], int, int, int]:
    # the per-transmitter polarization (threaded from the caller) drives the
    # LoS dipole sin^2 pattern in path_los_export.
    exported = topology_kernels.path_los_export(
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
        # ISB boundary taper (the boundary taper), LoS member. When on, the hard RayD
        # occlusion gate is replaced by the C1 membership predicate tau > 0: LoS
        # rows within one taper margin of the shadow boundary survive and carry
        # the clearance factor (re-derived and applied in the field stage). The
        # off path (the default) is untouched and stays the exact RayD gate.
        taper_boxes = (
            occluder_boxes(compiled) if isb_boundary_taper else None
        )
        if taper_boxes is not None:
            box_min, box_max = taper_boxes
            tau = los_clearance_factor(
                tx_positions[plan.tx_id.to(dtype=torch.int64)].contiguous(),
                rx_positions[plan.rx_id.to(dtype=torch.int64)].contiguous(),
                box_min,
                box_max,
                frequency_hz=frequency_hz,
                width=isb_boundary_taper_width,
            )
            visible = tau > 0.0
            launch_count += 1
        else:
            visibility_inputs = topology_kernels.path_los_visibility_inputs(
                tx_positions,
                rx_positions,
                plan.tx_id.to(dtype=torch.int32).contiguous(),
                plan.rx_id.to(dtype=torch.int32).contiguous(),
            )
            visible = run_visibility_query(
                VisibilityQuery(
                    rayd=compiled.rayd,
                    start=visibility_inputs["start"],
                    end=visibility_inputs["end"],
                    active=visibility_inputs["active"],
                )
            ).visible
            launch_count += 1
    los_block = _ensure_topology_fields(
        topology_kernels.deterministic_los_topology_block(
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