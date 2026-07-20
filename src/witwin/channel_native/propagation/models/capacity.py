"""Device-resident path-capacity layout contract."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from witwin.channel_native.runtime.capacity import (
    CapacityFailureState,
    require_capacity_failure_state,
)

from .evaluated import EvaluatedPaths


def _require_host_count(name: str, value: object) -> int:
    if type(value) is not int:
        raise TypeError(f"{name} must be an int")
    if value < 0:
        raise ValueError(f"{name} must be non-negative")
    return value


def _require_cuda_tensor(
    name: str,
    value: object,
    *,
    dtype: torch.dtype,
    shape: tuple[int, ...],
    device: torch.device | None = None,
) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if not value.is_cuda:
        raise ValueError(f"{name} must be a CUDA tensor")
    if value.dtype != dtype:
        raise ValueError(f"{name} must use {dtype}, got {value.dtype}")
    if tuple(value.shape) != shape:
        raise ValueError(f"{name} must have shape {shape}, got {tuple(value.shape)}")
    if not value.is_contiguous():
        raise ValueError(f"{name} must be contiguous")
    if device is not None and value.device != device:
        raise ValueError(f"{name} must be on {device}, got {value.device}")
    return value


@dataclass(frozen=True, slots=True, eq=False)
class CapacityPathLayout:
    """Host-known row capacity with device-resident validity and counts.

    Construction validates tensor metadata only. Native producers own the
    numerical relationship between ``valid``, ``num_paths``, and ``overflow``;
    this contract never reads or recomputes their device values.
    """

    pair_count: int
    path_capacity_per_pair: int
    failure_state: CapacityFailureState
    valid: torch.Tensor
    num_paths: torch.Tensor
    overflow: torch.Tensor

    def __post_init__(self) -> None:
        pair_count = _require_host_count("pair_count", self.pair_count)
        capacity = _require_host_count(
            "path_capacity_per_pair", self.path_capacity_per_pair
        )
        row_capacity = pair_count * capacity
        valid = _require_cuda_tensor(
            "valid",
            self.valid,
            dtype=torch.bool,
            shape=(row_capacity,),
        )
        require_capacity_failure_state(self.failure_state, device=valid.device)
        _require_cuda_tensor(
            "num_paths",
            self.num_paths,
            dtype=torch.int32,
            shape=(pair_count,),
            device=valid.device,
        )
        _require_cuda_tensor(
            "overflow",
            self.overflow,
            dtype=torch.bool,
            shape=(1,),
            device=valid.device,
        )

    @property
    def row_capacity(self) -> int:
        """Return the host-known flat row capacity."""

        return self.pair_count * self.path_capacity_per_pair

    @property
    def device(self) -> torch.device:
        """Return the shared CUDA device without reading tensor values."""

        return self.valid.device


@dataclass(frozen=True, slots=True, eq=False)
class CapacityPathSelection:
    """Stable source-row selection paired with a fixed-capacity layout."""

    selected_row_index: torch.Tensor
    layout: CapacityPathLayout

    def __post_init__(self) -> None:
        _require_cuda_tensor(
            "selected_row_index",
            self.selected_row_index,
            dtype=torch.int64,
            shape=(self.layout.row_capacity,),
            device=self.layout.device,
        )

    @property
    def pair_count(self) -> int:
        return self.layout.pair_count

    @property
    def path_capacity_per_pair(self) -> int:
        return self.layout.path_capacity_per_pair

    @property
    def row_capacity(self) -> int:
        return self.layout.row_capacity

    @property
    def valid(self) -> torch.Tensor:
        return self.layout.valid

    @property
    def num_paths(self) -> torch.Tensor:
        return self.layout.num_paths

    @property
    def overflow(self) -> torch.Tensor:
        return self.layout.overflow

    @property
    def device(self) -> torch.device:
        return self.layout.device


@dataclass(frozen=True, slots=True, eq=False)
class CapacityEvaluatedPaths:
    """Packed evaluated rows and their shared capacity selection contract."""

    selection: CapacityPathSelection
    evaluated: EvaluatedPaths

    def __post_init__(self) -> None:
        if not isinstance(self.selection, CapacityPathSelection):
            raise TypeError("selection must be a CapacityPathSelection")
        if not isinstance(self.evaluated, EvaluatedPaths):
            raise TypeError("evaluated must be EvaluatedPaths")
        if self.evaluated.row_count != self.selection.row_capacity:
            raise ValueError("evaluated rows must match selection capacity")
        if self.evaluated.topology.valid is not self.selection.valid:
            raise ValueError("evaluated topology must share selection validity")


__all__ = [
    "CapacityEvaluatedPaths",
    "CapacityPathLayout",
    "CapacityPathSelection",
]
