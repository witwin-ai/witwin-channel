from __future__ import annotations

import math

import torch

from witwin.channel_native import ReceiverGrid, ReceiverPoint, Scene


_LIGHT_SPEED_M_PER_S = 299_792_458.0


def receiver_positions(scene: Scene, *, device: torch.device) -> torch.Tensor:
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
    positions = [tx.position for tx in scene.transmitters]
    powers = [tx.power_w for tx in scene.transmitters]
    if not positions:
        return (
            torch.empty((0, 3), device=device, dtype=torch.float32),
            torch.empty((0,), device=device, dtype=torch.float32),
        )
    return (
        torch.stack(positions, dim=0).to(device=device, dtype=torch.float32).contiguous(),
        torch.tensor(powers, device=device, dtype=torch.float32),
    )


def los_path_gain(scene: Scene, *, device: torch.device) -> torch.Tensor:
    tx_pos, tx_power = transmitter_positions(scene, device=device)
    rx_pos = receiver_positions(scene, device=device)
    if tx_pos.shape[0] == 0 or rx_pos.shape[0] == 0:
        return torch.zeros((tx_pos.shape[0], rx_pos.shape[0]), device=device, dtype=torch.float32)

    distance = torch.linalg.vector_norm(tx_pos[:, None, :] - rx_pos[None, :, :], dim=-1)
    distance = torch.clamp(distance, min=1.0e-6)
    wavelength = _LIGHT_SPEED_M_PER_S / scene.frequency
    free_space_loss = (4.0 * math.pi * distance / wavelength) ** 2
    return tx_power[:, None] / free_space_loss
