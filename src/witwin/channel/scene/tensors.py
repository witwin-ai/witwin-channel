from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from witwin.channel.scene.endpoints import ReceiverGrid, ReceiverPoint
from witwin.channel.scene.receiver_geometry import vector3_tuple
from witwin.channel.propagation.topology.kernels import (
    primitives as topology_primitives,
)
from witwin.channel.runtime.native_buffers import (
    mc_receiver_grid_points,
    mc_transmitter_tensors,
)

if TYPE_CHECKING:
    from witwin.channel.scene.endpoints import SolverScene as Scene


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
    if (
        len(scene.receivers) == 1
        and isinstance(scene.receivers[0], ReceiverGrid)
        and reference is not None
    ):
        return receiver_grid_points(scene.receivers[0], reference=reference)
    blocks: list[torch.Tensor] = []
    grid_reference = reference

    for receiver in scene.receivers:
        if isinstance(receiver, ReceiverPoint):
            block = receiver.position.reshape(1, 3).to(device=device)
            blocks.append(block)
            if grid_reference is None:
                grid_reference = block
        elif isinstance(receiver, ReceiverGrid):
            if grid_reference is None:
                grid_reference = host_vec3_tensor(())
            blocks.append(receiver_grid_points(receiver, reference=grid_reference))
        else:
            raise TypeError(f"receiver type is not accepted: {type(receiver)!r}")
    if not blocks:
        return host_vec3_tensor(())
    if len(blocks) == 1:
        return blocks[0]
    return topology_primitives.path_concat_vec3(blocks)


def transmitter_positions(
    scene: object, *, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    if not scene.transmitters:
        exported = mc_transmitter_tensors((), ())
        return exported["positions"], exported["power"]
    positions = torch.stack(
        tuple(transmitter.position for transmitter in scene.transmitters)
    ).to(device=device, dtype=torch.float32)
    powers = torch.stack(
        tuple(
            power.reshape(())
            if isinstance((power := transmitter.power_w), torch.Tensor)
            else positions.new_tensor(float(power))
            for transmitter in scene.transmitters
        )
    ).to(device=device, dtype=torch.float32)
    return positions, powers


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

    if not scene.transmitters:
        return host_vec3_tensor(())
    return torch.stack(
        tuple(transmitter.polarization for transmitter in scene.transmitters)
    ).to(
        device=device
    )


# Keep the long-standing compatibility and pickle paths stable while this
# module becomes the canonical implementation owner.
