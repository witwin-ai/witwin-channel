"""Structural contracts for propagation configuration and evaluated rows."""

from __future__ import annotations

from typing import Protocol

import torch


class TopologyConfig(Protocol):
    max_depth: int
    components: frozenset[str] | set[str] | tuple[str, ...] | list[str]
    max_paths: int | None
    max_paths_scope: str


TopologyConfig.__module__ = "witwin.channel_native.core.path_topology"


class EvaluatedRowsSource(Protocol):
    """Structural source contract for the legacy mixed row table."""

    valid: torch.Tensor
    tx_id: torch.Tensor
    rx_id: torch.Tensor
    depth: torch.Tensor
    component_id: torch.Tensor
    primitive_id: torch.Tensor
    edge_id: torch.Tensor
    material_id: torch.Tensor
    primitive_sequence: torch.Tensor
    material_sequence: torch.Tensor
    interaction_type: torch.Tensor
    path_length_m: torch.Tensor
    delay_s: torch.Tensor
    field_direction: torch.Tensor
    interaction_position: torch.Tensor
    interaction_normal: torch.Tensor
    interaction_positions: torch.Tensor
    interaction_normals: torch.Tensor
    path_gain: torch.Tensor
    path_field: torch.Tensor
    field_xyz: torch.Tensor
    coefficient: torch.Tensor
    launch_count: int
    visibility_rejection_count: int
    selected_edge_count: int
    candidate_count: int
    guardrail_count: int
    ad_companion_launches: int
    ad_tape_bytes: int
    diffraction_vector_field: torch.Tensor | None
