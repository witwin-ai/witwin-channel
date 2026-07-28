"""The native field-state ABI contracts.

Two frozen dataclasses and nothing else: the world-Cartesian
:class:`Complex3State` and the transverse-basis :class:`JonesState` that every
native field kernel reads and writes. They live at the package root because
:mod:`witwin.channel` exports them and the ``public_init_internal`` boundary
forbids the root ``__init__`` from importing ``runtime``, ``propagation``, or a
``kernels`` package to reach them.

This module used to be called ``field_state`` and also held the scene-derived
transmitter/receiver polarization tensors. Those were never an ABI contract -
they read a logical scene and build endpoint tensors - so they moved to
:mod:`witwin.channel.scene.endpoints`, which owns endpoint geometry. What is
left here depends on ``torch`` alone, so importing the public root no longer
drags the scene package in behind it.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


def _validate_rows(name: str, tensor: torch.Tensor, width: int) -> torch.Tensor:
    if tensor.ndim != 2 or tensor.shape[1] != width:
        raise ValueError(f"{name} must have shape (N, {width})")
    if not tensor.is_cuda:
        raise ValueError(f"{name} must be a CUDA tensor")
    return tensor.contiguous()


@dataclass(frozen=True, slots=True)
class Complex3State:
    """Canonical native field ABI: world-Cartesian complex E and ray direction."""

    field: torch.Tensor
    direction: torch.Tensor

    def __post_init__(self) -> None:
        field = _validate_rows("field", self.field, 3)
        direction = _validate_rows("direction", self.direction, 3)
        if field.dtype != torch.complex64:
            raise TypeError("field must have dtype complex64")
        if direction.dtype != torch.float32:
            raise TypeError("direction must have dtype float32")
        if field.shape[0] != direction.shape[0] or field.device != direction.device:
            raise ValueError("field and direction must share rows and device")
        object.__setattr__(self, "field", field)
        object.__setattr__(self, "direction", direction)


@dataclass(frozen=True, slots=True)
class JonesState:
    """Two-component complex field in an explicit transverse world basis."""

    value: torch.Tensor
    basis: torch.Tensor
    direction: torch.Tensor

    def __post_init__(self) -> None:
        value = _validate_rows("value", self.value, 2)
        direction = _validate_rows("direction", self.direction, 3)
        basis = self.basis
        if basis.ndim != 3 or tuple(basis.shape[1:]) != (2, 3):
            raise ValueError("basis must have shape (N, 2, 3)")
        if value.dtype != torch.complex64:
            raise TypeError("value must have dtype complex64")
        if basis.dtype != torch.float32 or direction.dtype != torch.float32:
            raise TypeError("basis and direction must have dtype float32")
        if not basis.is_cuda:
            raise ValueError("basis must be a CUDA tensor")
        if not (value.shape[0] == basis.shape[0] == direction.shape[0]):
            raise ValueError("Jones state tensors must share rows")
        if not (value.device == basis.device == direction.device):
            raise ValueError("Jones state tensors must share device")
        object.__setattr__(self, "value", value.contiguous())
        object.__setattr__(self, "basis", basis.contiguous())
        object.__setattr__(self, "direction", direction)
