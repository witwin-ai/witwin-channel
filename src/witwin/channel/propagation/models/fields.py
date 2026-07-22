"""Row-aligned RF field contract."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .topology import _RowIdentity, _require_tensor


@dataclass(frozen=True, slots=True, eq=False)
class PathFields:
    """RF fields aligned to an existing ``PathTopology`` row table."""

    row_identity: _RowIdentity
    path_gain: torch.Tensor
    path_field: torch.Tensor
    field_xyz: torch.Tensor
    coefficient: torch.Tensor

    def __post_init__(self) -> None:
        if not isinstance(self.row_identity, _RowIdentity):
            raise TypeError("row_identity must come from PathTopology")
        rows = self.row_identity.row_count
        device = self.row_identity.device
        _require_tensor(
            "path_gain",
            self.path_gain,
            dtype=torch.float32,
            shape=(rows,),
            device=device,
        )
        for name in ("path_field", "coefficient"):
            _require_tensor(
                name,
                getattr(self, name),
                dtype=torch.complex64,
                shape=(rows,),
                device=device,
            )
        _require_tensor(
            "field_xyz",
            self.field_xyz,
            dtype=torch.complex64,
            shape=(rows, 3),
            device=device,
        )

    @property
    def row_count(self) -> int:
        return self.row_identity.row_count

    @property
    def device(self) -> torch.device:
        return self.row_identity.device


__all__ = ["PathFields"]
