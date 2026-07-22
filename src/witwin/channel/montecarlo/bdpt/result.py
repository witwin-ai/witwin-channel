from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass(frozen=True, slots=True)
class BDPTPathSamples:
    topology: torch.Tensor
    contribution: torch.Tensor
    pdf: torch.Tensor
    mis_weight: torch.Tensor
    component_id: torch.Tensor
    valid: torch.Tensor
    tx_id: torch.Tensor
    rx_id: torch.Tensor
    grid_linear_id: torch.Tensor
    light_depth: torch.Tensor
    sensor_depth: torch.Tensor
    path_length_m: torch.Tensor


@dataclass(frozen=True, slots=True)
class Result:
    path_gain: torch.Tensor
    component_power: dict[str, torch.Tensor]
    metadata: dict[str, Any]
    diagnostics: dict[str, Any] | None = None
    component_maps: dict[str, torch.Tensor] | None = None
    variance: torch.Tensor | None = None
    path_samples: BDPTPathSamples | None = None
