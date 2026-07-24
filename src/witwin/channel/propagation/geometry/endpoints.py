from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from witwin.channel.scene.endpoints import ReceiverGrid
from witwin.channel.scene.tensors import (
    receiver_positions as _native_receiver_positions,
    transmitter_positions as _native_transmitter_positions,
)

if TYPE_CHECKING:
    from witwin.channel.scene.endpoints import SolverScene as Scene


@dataclass(frozen=True, slots=True)
class ReceiverLayout:
    """Maps flat receiver ids to the deterministic public result layout."""

    kind: str
    receiver_count: int
    grid_shape: tuple[int, int] | None = None

    def apply(self, values: torch.Tensor) -> torch.Tensor:
        if self.kind == "grid":
            if self.grid_shape is None:
                raise ValueError("grid layout requires grid_shape")
            rows, cols = self.grid_shape
            return (
                values.reshape(values.shape[0], rows, cols).transpose(1, 2).contiguous()
            )
        if self.kind == "point":
            return values.contiguous()
        raise ValueError(f"receiver layout kind is not accepted: {self.kind}")


def transmitter_tensors(
    scene: Scene, *, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    return _native_transmitter_positions(scene, device=device)


def receiver_positions_and_layout(
    scene: Scene, *, device: torch.device
) -> tuple[torch.Tensor, ReceiverLayout]:
    if not scene.receivers:
        return torch.empty((0, 3), device=device, dtype=torch.float32), ReceiverLayout(
            "point", 0
        )

    reference, _power = transmitter_tensors(scene, device=device)
    positions = _native_receiver_positions(scene, device=device, reference=reference)
    if len(scene.receivers) == 1 and isinstance(scene.receivers[0], ReceiverGrid):
        grid = scene.receivers[0]
        return positions, ReceiverLayout("grid", int(positions.shape[0]), grid.shape)

    return positions, ReceiverLayout("point", int(positions.shape[0]))


def apply_receiver_layout(values: torch.Tensor, layout: ReceiverLayout) -> torch.Tensor:
    return layout.apply(values)
