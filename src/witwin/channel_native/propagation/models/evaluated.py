"""Composition contract for row-aligned topology, geometry, and fields."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .fields import PathFields
from .geometry import PathGeometry
from .topology import PathTopology, _RowIdentity


@dataclass(frozen=True, slots=True, eq=False)
class EvaluatedPaths:
    """Internal propagation result with exact shared path-row identity."""

    topology: PathTopology
    geometry: PathGeometry
    fields: PathFields

    def __post_init__(self) -> None:
        if not isinstance(self.topology, PathTopology):
            raise TypeError("topology must be a PathTopology")
        if not isinstance(self.geometry, PathGeometry):
            raise TypeError("geometry must be a PathGeometry")
        if not isinstance(self.fields, PathFields):
            raise TypeError("fields must be PathFields")
        identity = self.topology.row_identity
        if self.geometry.row_identity is not identity:
            raise ValueError("geometry must share topology row_identity")
        if self.fields.row_identity is not identity:
            raise ValueError("fields must share topology row_identity")

    @property
    def row_identity(self) -> _RowIdentity:
        return self.topology.row_identity

    @property
    def row_count(self) -> int:
        return self.topology.row_count

    @property
    def device(self) -> torch.device:
        return self.topology.device


__all__ = ["EvaluatedPaths"]
