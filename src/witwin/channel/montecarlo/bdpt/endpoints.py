"""Transmitter/receiver endpoint tensor packing for the BDPT solver."""

from __future__ import annotations

import torch

from witwin.channel.scene.endpoints import ReceiverGrid, ReceiverPoint
from witwin.channel.montecarlo.bdpt.kernels.maps import (
    bdpt_host_vec3_tensor,
    bdpt_receiver_grid_points,
    bdpt_transmitter_tensors,
)
from witwin.channel.scene.endpoints import SolverScene as Scene
from witwin.channel.scene.endpoints import (
    vector3_tuple as _vector3_tuple,
)


def transmitter_tensors(scene: Scene) -> tuple[torch.Tensor, torch.Tensor]:
    flat_positions = tuple(
        component
        for transmitter in scene.transmitters
        for component in _vector3_tuple(transmitter.position)
    )
    # Read the host value for the native pack; a live power_w leaf is detached
    # here (its gradient is reattached under ad by the pipeline's _live_tx_power,
    # ADR-022 tx_power threading) so this stays a plain host read.
    powers = tuple(
        float(transmitter.power_w.detach())
        if isinstance(transmitter.power_w, torch.Tensor)
        else float(transmitter.power_w)
        for transmitter in scene.transmitters
    )
    exported = bdpt_transmitter_tensors(flat_positions, powers)
    return exported["positions"], exported["power"]


def receiver_positions(
    scene: Scene,
    *,
    reference: torch.Tensor,
    grid: ReceiverGrid | None = None,
) -> torch.Tensor:
    if grid is not None:
        return bdpt_receiver_grid_points(
            reference,
            origin=_vector3_tuple(grid.origin),
            x_axis=_vector3_tuple(grid.x_axis),
            y_axis=_vector3_tuple(grid.y_axis),
            shape=grid.shape,
            spacing=grid.spacing,
        )
    if len(scene.receivers) == 1 and isinstance(scene.receivers[0], ReceiverGrid):
        return receiver_positions(scene, reference=reference, grid=scene.receivers[0])

    flat_positions: list[float] = []
    for receiver in scene.receivers:
        if isinstance(receiver, ReceiverPoint):
            flat_positions.extend(_vector3_tuple(receiver.position))
        elif isinstance(receiver, ReceiverGrid):
            raise ValueError("BDPT supports either one ReceiverGrid or point receivers, not mixed receiver grids")
        else:
            raise TypeError(f"receiver type is not accepted: {type(receiver)!r}")
    return bdpt_host_vec3_tensor(tuple(flat_positions))


