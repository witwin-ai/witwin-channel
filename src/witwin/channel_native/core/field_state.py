from __future__ import annotations

from dataclasses import dataclass

import torch

from witwin.channel_native.core.objects import ReceiverGrid, ReceiverPoint


PHASE_CONVENTION = {
    "phasor": "exp(-j*k*d)",
    "time_dependence": "exp(+j*2*pi*f*t)",
    "free_space_amplitude": "sqrt(tx_power)*wavelength/(4*pi*distance)",
    "polarization": "world_cartesian_complex3_then_receiver_projection",
}


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


def transmitter_polarizations(scene: object, *, device: torch.device) -> torch.Tensor:
    values = [tx.polarization for tx in scene.transmitters]
    if not values:
        return torch.empty((0, 3), device=device, dtype=torch.float32)
    return torch.stack(values).to(device=device, dtype=torch.float32).contiguous()


def receiver_polarizations(
    scene: object,
    *,
    device: torch.device,
    grid: ReceiverGrid | None = None,
) -> torch.Tensor:
    if grid is not None:
        return (
            grid.polarization.to(device=device, dtype=torch.float32)
            .expand(grid.shape[0] * grid.shape[1], 3)
            .contiguous()
        )
    values: list[torch.Tensor] = []
    for receiver in scene.receivers:
        if isinstance(receiver, ReceiverGrid):
            values.extend([receiver.polarization] * (receiver.shape[0] * receiver.shape[1]))
        elif isinstance(receiver, ReceiverPoint):
            values.append(receiver.polarization)
        else:
            raise TypeError(f"receiver type is not accepted: {type(receiver)!r}")
    if not values:
        return torch.empty((0, 3), device=device, dtype=torch.float32)
    return torch.stack(values).to(device=device, dtype=torch.float32).contiguous()
