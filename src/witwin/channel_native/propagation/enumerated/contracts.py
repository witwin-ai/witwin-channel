from __future__ import annotations

from typing import Protocol

import torch


class TopologyConfig(Protocol):
    components: frozenset[str] | set[str] | tuple[str, ...] | list[str]
    max_depth: int
    scattering_samples_per_m2: float
    scattering_power_threshold: float
    scattering_max_paths_per_pair: int


class TopologyBatch(Protocol):
    valid: torch.Tensor
    tx_id: torch.Tensor
    rx_id: torch.Tensor
    depth: torch.Tensor
    component_id: torch.Tensor
    primitive_id: torch.Tensor
    edge_id: torch.Tensor
    path_length_m: torch.Tensor
    delay_s: torch.Tensor
    path_gain: torch.Tensor
    path_field: torch.Tensor
    field_xyz: torch.Tensor
    coefficient: torch.Tensor
    field_direction: torch.Tensor
    interaction_position: torch.Tensor
    interaction_normal: torch.Tensor
    material_id: torch.Tensor
    primitive_sequence: torch.Tensor
    material_sequence: torch.Tensor
    interaction_type: torch.Tensor
    interaction_positions: torch.Tensor
    interaction_normals: torch.Tensor
    launch_count: int
    candidate_count: int
    guardrail_count: int
