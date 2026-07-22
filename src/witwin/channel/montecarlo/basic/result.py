from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass(frozen=True, slots=True)
class Result:
    path_gain: torch.Tensor
    component_power: dict[str, torch.Tensor]
    metadata: dict[str, Any]
    diagnostics: dict[str, Any] | None = None
    component_maps: dict[str, torch.Tensor] | None = None
