"""Discrete path identity and winner contract."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch


def _require_tensor(
    name: str,
    value: object,
    *,
    dtype: torch.dtype,
    shape: tuple[int, ...] | None = None,
    ndim: int | None = None,
    device: torch.device | None = None,
) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if value.dtype != dtype:
        raise ValueError(f"{name} must use {dtype}, got {value.dtype}")
    if shape is not None and tuple(value.shape) != shape:
        raise ValueError(f"{name} must have shape {shape}, got {tuple(value.shape)}")
    if ndim is not None and value.ndim != ndim:
        raise ValueError(f"{name} must have rank {ndim}, got {value.ndim}")
    if device is not None and value.device != device:
        raise ValueError(f"{name} must be on {device}, got {value.device}")
    return value


@dataclass(frozen=True, slots=True, eq=False)
class _RowIdentity:
    """Opaque row token shared by all views of one path table."""

    row_count: int
    sequence_width: int
    device: torch.device


@dataclass(frozen=True, slots=True, eq=False)
class PathTopology:
    """Discrete path rows without continuous geometry, fields, or runtime state."""

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
    _row_identity: _RowIdentity = field(init=False, repr=False)

    def __post_init__(self) -> None:
        valid = _require_tensor("valid", self.valid, dtype=torch.bool, ndim=1)
        row_count = int(valid.shape[0])
        device = valid.device
        vector_shape = (row_count,)
        for name in (
            "tx_id",
            "rx_id",
            "depth",
            "component_id",
            "primitive_id",
            "edge_id",
            "material_id",
        ):
            _require_tensor(
                name,
                getattr(self, name),
                dtype=torch.int32,
                shape=vector_shape,
                device=device,
            )

        primitive_sequence = _require_tensor(
            "primitive_sequence",
            self.primitive_sequence,
            dtype=torch.int32,
            ndim=2,
            device=device,
        )
        sequence_width = int(primitive_sequence.shape[1])
        sequence_shape = (row_count, sequence_width)
        if tuple(primitive_sequence.shape) != sequence_shape:
            raise ValueError(
                "primitive_sequence must have shape "
                f"{sequence_shape}, got {tuple(primitive_sequence.shape)}"
            )
        for name in ("material_sequence", "interaction_type"):
            _require_tensor(
                name,
                getattr(self, name),
                dtype=torch.int32,
                shape=sequence_shape,
                device=device,
            )

        object.__setattr__(
            self,
            "_row_identity",
            _RowIdentity(
                row_count=row_count,
                sequence_width=sequence_width,
                device=device,
            ),
        )

    @property
    def row_identity(self) -> _RowIdentity:
        """Opaque identity token; downstream contracts must reuse this object."""

        return self._row_identity

    @property
    def row_count(self) -> int:
        return self._row_identity.row_count

    @property
    def sequence_width(self) -> int:
        return self._row_identity.sequence_width

    @property
    def device(self) -> torch.device:
        return self._row_identity.device


__all__ = ["PathTopology"]
