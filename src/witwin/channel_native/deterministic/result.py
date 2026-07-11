from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass(frozen=True, slots=True)
class PathTable:
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
    interaction_position: torch.Tensor
    interaction_normal: torch.Tensor
    material_id: torch.Tensor
    primitive_sequence: torch.Tensor
    material_sequence: torch.Tensor
    interaction_positions: torch.Tensor
    interaction_normals: torch.Tensor
    field_real: torch.Tensor
    field_imag: torch.Tensor
    coefficient: torch.Tensor
    field_xyz: torch.Tensor
    field_direction: torch.Tensor
    phase_rad: torch.Tensor
    interaction_count: torch.Tensor


@dataclass(frozen=True, slots=True)
class Result:
    path_gain: torch.Tensor
    field: torch.Tensor
    component_power: dict[str, torch.Tensor]
    component_fields: dict[str, torch.Tensor]
    paths: PathTable | None
    metadata: dict[str, Any]
    diagnostics: dict[str, Any] | None = None
