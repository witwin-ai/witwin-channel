from __future__ import annotations

import torch

from witwin.channel_native import ReceiverGrid, ReceiverPoint, Scene
from witwin.channel_native.core.kernels.ops import (
    mc_los_path_gain_backward,
    mc_los_path_gain_jvp,
    mc_receiver_grid_points,
    mc_transmitter_tensors,
    path_los_export,
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
    positions = []
    for receiver in scene.receivers:
        if isinstance(receiver, ReceiverPoint):
            positions.append(receiver.position)
        elif isinstance(receiver, ReceiverGrid):
            positions.extend(receiver.points())
        else:
            raise TypeError(f"unsupported receiver type: {type(receiver)!r}")
    return torch.stack(positions, dim=0).to(device=device, dtype=torch.float32).contiguous()


def transmitter_positions(scene: Scene, *, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    del device
    if not scene.transmitters:
        return (
            torch.empty((0, 3), device="cuda", dtype=torch.float32),
            torch.empty((0,), device="cuda", dtype=torch.float32),
        )
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
        return torch.zeros((tx_pos.shape[0], rx_pos.shape[0]), device=device, dtype=torch.float32)

    exported = path_los_export(
        tx_pos,
        tx_power,
        rx_pos,
        frequency_hz=float(scene.frequency),
    )
    return exported["path_gain_matrix"]


class _LosPathGainAutograd(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        tx_pos: torch.Tensor,
        tx_power: torch.Tensor,
        rx_pos: torch.Tensor,
        frequency_hz: float,
    ) -> torch.Tensor:
        ctx.frequency_hz = float(frequency_hz)
        ctx.save_for_backward(tx_pos, tx_power, rx_pos)
        ctx.save_for_forward(tx_pos, tx_power, rx_pos)
        exported = path_los_export(
            tx_pos,
            tx_power,
            rx_pos,
            frequency_hz=float(frequency_hz),
        )
        return exported["path_gain_matrix"]

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx,
        grad_output: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, None]:
        tx_pos, tx_power, rx_pos = ctx.saved_tensors
        grad_tx, grad_power, grad_rx = mc_los_path_gain_backward(
            tx_pos,
            tx_power,
            rx_pos,
            grad_output,
            frequency_hz=ctx.frequency_hz,
        )
        return grad_tx, grad_power, grad_rx, None

    @staticmethod
    def jvp(
        ctx: torch.autograd.function.FunctionCtx,
        tx_tangent: torch.Tensor | None,
        power_tangent: torch.Tensor | None,
        rx_tangent: torch.Tensor | None,
        frequency_tangent: object,
    ) -> torch.Tensor:
        del frequency_tangent
        tx_pos, tx_power, rx_pos = ctx.saved_tensors
        return mc_los_path_gain_jvp(
            tx_pos,
            tx_power,
            rx_pos,
            tx_pos if tx_tangent is None else tx_tangent,
            tx_power if power_tangent is None else power_tangent,
            rx_pos if rx_tangent is None else rx_tangent,
            tx_tangent is not None,
            power_tangent is not None,
            rx_tangent is not None,
            frequency_hz=ctx.frequency_hz,
        )


def los_path_gain_autograd(scene: Scene, *, device: torch.device) -> torch.Tensor:
    tx_pos, tx_power = transmitter_positions(scene, device=device)
    rx_pos = receiver_positions(scene, device=device, reference=tx_pos)
    if tx_pos.shape[0] == 0 or rx_pos.shape[0] == 0:
        return torch.zeros((tx_pos.shape[0], rx_pos.shape[0]), device=device, dtype=torch.float32)

    return _LosPathGainAutograd.apply(tx_pos, tx_power, rx_pos, float(scene.frequency))
