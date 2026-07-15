"""Continuous fixed-winner path geometry contract."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .topology import _RowIdentity, _require_tensor


@dataclass(frozen=True, slots=True, eq=False)
class PathGeometry:
    """Continuous geometry aligned to an existing ``PathTopology`` row table."""

    row_identity: _RowIdentity
    path_length_m: torch.Tensor
    delay_s: torch.Tensor
    field_direction: torch.Tensor
    interaction_position: torch.Tensor
    interaction_normal: torch.Tensor
    interaction_positions: torch.Tensor
    interaction_normals: torch.Tensor

    def __post_init__(self) -> None:
        if not isinstance(self.row_identity, _RowIdentity):
            raise TypeError("row_identity must come from PathTopology")
        rows = self.row_identity.row_count
        width = self.row_identity.sequence_width
        device = self.row_identity.device
        for name in ("path_length_m", "delay_s"):
            _require_tensor(
                name,
                getattr(self, name),
                dtype=torch.float32,
                shape=(rows,),
                device=device,
            )
        for name in (
            "field_direction",
            "interaction_position",
            "interaction_normal",
        ):
            _require_tensor(
                name,
                getattr(self, name),
                dtype=torch.float32,
                shape=(rows, 3),
                device=device,
            )
        for name in ("interaction_positions", "interaction_normals"):
            _require_tensor(
                name,
                getattr(self, name),
                dtype=torch.float32,
                shape=(rows, width, 3),
                device=device,
            )

    @property
    def row_count(self) -> int:
        return self.row_identity.row_count

    @property
    def device(self) -> torch.device:
        return self.row_identity.device


__all__ = ["PathGeometry"]
