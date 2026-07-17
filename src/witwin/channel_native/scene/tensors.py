from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from witwin.channel_native.scene.models import ReceiverGrid, ReceiverPoint
from witwin.channel_native.core.receiver_geometry import vector3_tuple
from witwin.channel_native.propagation.topology.kernels import (
    primitives as topology_primitives,
)
from witwin.channel_native.runtime.native_buffers import (
    mc_receiver_grid_points,
    mc_transmitter_tensors,
)

if TYPE_CHECKING:
    from witwin.channel_native.scene.models import Scene


LIGHT_SPEED_M_PER_S = 299_792_458.0


def _frequency_scalar(scene: Scene) -> float:
    """Detached scalar carrier for non-differentiable consumers.

    Topology discovery and metadata never differentiate with respect to
    frequency (fixed-topology contract), so detach before float() to keep AD
    solves with a requires_grad tensor frequency warning-free. The field
    evaluation seam must NOT use this helper: it forwards the live tensor so
    the frequency stays on the autograd graph.
    """

    frequency = scene.frequency
    if isinstance(frequency, torch.Tensor):
        return float(frequency.detach())
    return float(frequency)


def receiver_grid_points(grid: ReceiverGrid, *, reference: torch.Tensor) -> torch.Tensor:
    return mc_receiver_grid_points(
        reference,
        origin=vector3_tuple(grid.origin),
        x_axis=vector3_tuple(grid.x_axis),
        y_axis=vector3_tuple(grid.y_axis),
        shape=grid.shape,
        spacing=grid.spacing,
    )


def host_vec3_tensor(flat_positions: tuple[float, ...]) -> torch.Tensor:
    powers = tuple(1.0 for _ in range(len(flat_positions) // 3))
    return mc_transmitter_tensors(flat_positions, powers)["positions"]


def receiver_positions(
    scene: object,
    *,
    device: torch.device,
    reference: torch.Tensor | None = None,
) -> torch.Tensor:
    del device
    if (
        len(scene.receivers) == 1
        and isinstance(scene.receivers[0], ReceiverGrid)
        and reference is not None
    ):
        return receiver_grid_points(scene.receivers[0], reference=reference)
    blocks: list[torch.Tensor] = []
    point_positions: list[float] = []
    grid_reference = reference

    def flush_points() -> None:
        nonlocal point_positions, grid_reference
        if not point_positions:
            return
        block = host_vec3_tensor(tuple(point_positions))
        blocks.append(block)
        if grid_reference is None:
            grid_reference = block
        point_positions = []

    for receiver in scene.receivers:
        if isinstance(receiver, ReceiverPoint):
            point_positions.extend(vector3_tuple(receiver.position))
        elif isinstance(receiver, ReceiverGrid):
            flush_points()
            if grid_reference is None:
                grid_reference = host_vec3_tensor(())
            blocks.append(receiver_grid_points(receiver, reference=grid_reference))
        else:
            raise TypeError(f"receiver type is not accepted: {type(receiver)!r}")
    flush_points()
    if not blocks:
        return host_vec3_tensor(())
    if len(blocks) == 1:
        return blocks[0]
    return topology_primitives.path_concat_vec3(blocks)


def transmitter_positions(
    scene: object, *, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    del device
    if not scene.transmitters:
        exported = mc_transmitter_tensors((), ())
        return exported["positions"], exported["power"]
    flat_positions = tuple(
        component
        for transmitter in scene.transmitters
        for component in vector3_tuple(transmitter.position)
    )
    powers = tuple(float(transmitter.power_w) for transmitter in scene.transmitters)
    exported = mc_transmitter_tensors(flat_positions, powers)
    return exported["positions"], exported["power"]


def transmitter_polarizations(
    scene: object, *, device: torch.device
) -> torch.Tensor:
    """Per-transmitter polarization unit vectors as a (N, 3) CUDA tensor.

    Row order matches :func:`transmitter_positions`. The transmitter model
    already normalizes and orients ``.polarization`` in ``__post_init__``, so
    this is a straight device upload of the fixed physical vectors (frozen
    winners of AD; the dipole sin^2 pattern they induce is differentiated
    through the endpoint geometry, not through the polarization itself).
    """

    del device
    if not scene.transmitters:
        return host_vec3_tensor(())
    flat = tuple(
        component
        for transmitter in scene.transmitters
        for component in vector3_tuple(transmitter.polarization)
    )
    return host_vec3_tensor(flat)


# Keep the long-standing compatibility and pickle paths stable while this
# module becomes the canonical implementation owner.
_frequency_scalar.__module__ = "witwin.channel_native.core.scene_tensors"
receiver_grid_points.__module__ = "witwin.channel_native.core.scene_tensors"
host_vec3_tensor.__module__ = "witwin.channel_native.core.scene_tensors"
receiver_positions.__module__ = "witwin.channel_native.core.scene_tensors"
transmitter_positions.__module__ = "witwin.channel_native.core.scene_tensors"
transmitter_polarizations.__module__ = "witwin.channel_native.core.scene_tensors"
