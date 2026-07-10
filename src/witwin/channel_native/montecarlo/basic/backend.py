from __future__ import annotations

import torch

from witwin.channel_native import ReceiverGrid, ReceiverPoint, Scene
from witwin.channel_native.core.kernels.ops import (
    mc_los_visibility_inputs,
    mc_receiver_grid_points,
    mc_transmitter_tensors,
    mc_zero_matrix,
    path_concat_vec3,
    path_los_export,
    raydn_visibility_forward,
)


_LIGHT_SPEED_M_PER_S = 299_792_458.0


def _vector3_tuple(value: torch.Tensor) -> tuple[float, float, float]:
    return (float(value[0]), float(value[1]), float(value[2]))


def receiver_grid_points(grid: ReceiverGrid, *, reference: torch.Tensor) -> torch.Tensor:
    return mc_receiver_grid_points(
        reference,
        origin=_vector3_tuple(grid.origin),
        x_axis=_vector3_tuple(grid.x_axis),
        y_axis=_vector3_tuple(grid.y_axis),
        shape=grid.shape,
        spacing=grid.spacing,
    )


def _host_vec3_tensor(flat_positions: tuple[float, ...]) -> torch.Tensor:
    powers = tuple(1.0 for _ in range(len(flat_positions) // 3))
    return mc_transmitter_tensors(flat_positions, powers)["positions"]


def receiver_positions(
    scene: Scene,
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
    point_positions: list[float] = []
    grid_reference = reference

    def flush_points() -> None:
        nonlocal point_positions, grid_reference
        if not point_positions:
            return
        block = _host_vec3_tensor(tuple(point_positions))
        blocks.append(block)
        if grid_reference is None:
            grid_reference = block
        point_positions = []

    for receiver in scene.receivers:
        if isinstance(receiver, ReceiverPoint):
            point_positions.extend(_vector3_tuple(receiver.position))
        elif isinstance(receiver, ReceiverGrid):
            flush_points()
            if grid_reference is None:
                grid_reference = _host_vec3_tensor(())
            blocks.append(receiver_grid_points(receiver, reference=grid_reference))
        else:
            raise TypeError(f"receiver type is not accepted: {type(receiver)!r}")
    flush_points()
    if not blocks:
        return _host_vec3_tensor(())
    if len(blocks) == 1:
        return blocks[0]
    return path_concat_vec3(blocks)


def transmitter_positions(scene: Scene, *, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    del device
    if not scene.transmitters:
        exported = mc_transmitter_tensors((), ())
        return exported["positions"], exported["power"]
    flat_positions = tuple(
        component
        for tx in scene.transmitters
        for component in _vector3_tuple(tx.position)
    )
    powers = tuple(float(tx.power_w) for tx in scene.transmitters)
    exported = mc_transmitter_tensors(flat_positions, powers)
    return exported["positions"], exported["power"]


def los_path_gain(scene: Scene, *, device: torch.device) -> torch.Tensor:
    tx_pos, tx_power = transmitter_positions(scene, device=device)
    rx_pos = receiver_positions(scene, device=device, reference=tx_pos)
    if tx_pos.shape[0] == 0 or rx_pos.shape[0] == 0:
        return mc_zero_matrix(tx_pos, rows=tx_pos.shape[0], cols=rx_pos.shape[0])

    exported = path_los_export(
        tx_pos,
        tx_power,
        rx_pos,
        frequency_hz=float(scene.frequency),
    )
    return exported["path_gain_matrix"]


def apply_point_los_visibility(scene: Scene, raydn: object, los: torch.Tensor, *, device: torch.device) -> torch.Tensor:
    """Zero occluded (tx, rx) entries of a point-receiver LoS matrix."""

    if not scene.structures or los.numel() == 0:
        return los
    if not raydn.available:
        raise RuntimeError("LoS visibility requires RayDN native scene capability")
    handle = raydn.require_handle()
    tx_pos, _ = transmitter_positions(scene, device=device)
    rx_pos = receiver_positions(scene, device=device, reference=tx_pos)
    masks: list[torch.Tensor] = []
    for tx_index in range(int(tx_pos.shape[0])):
        inputs = mc_los_visibility_inputs(tx_pos, tx_index=tx_index, rx_count=int(rx_pos.shape[0]))
        masks.append(raydn_visibility_forward(handle, inputs["start"], rx_pos, inputs["active"])[0])
    return los * torch.stack(masks, dim=0).to(dtype=los.dtype)
