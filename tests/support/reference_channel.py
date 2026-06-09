from __future__ import annotations

import torch

from witwin.channel_native import ReceiverGrid, ReceiverPoint, Scene


_LIGHT_SPEED_M_PER_S = 299_792_458.0


def _receiver_positions(scene: Scene, *, device: torch.device) -> torch.Tensor:
    blocks = []
    for receiver in scene.receivers:
        if isinstance(receiver, ReceiverPoint):
            blocks.append(receiver.position)
        elif isinstance(receiver, ReceiverGrid):
            blocks.append(receiver.points())
        else:
            raise TypeError(f"unsupported receiver type: {type(receiver).__name__}")
    return torch.vstack(
        [
            block.reshape(-1, 3).to(device=device, dtype=torch.float32, non_blocking=True)
            for block in blocks
        ]
    )


def los_path_gain_reference(scene: Scene, *, device: torch.device) -> torch.Tensor:
    tx_positions = torch.stack([tx.position for tx in scene.transmitters], dim=0).to(
        device=device, dtype=torch.float32, non_blocking=True
    )
    tx_power = torch.tensor(
        [tx.power_w for tx in scene.transmitters],
        device=device,
        dtype=torch.float32,
    )
    rx_positions = _receiver_positions(scene, device=device)
    distances = torch.linalg.vector_norm(
        tx_positions[:, None, :] - rx_positions[None, :, :],
        dim=-1,
    ).clamp_min(torch.finfo(torch.float32).eps)
    wavelength = _LIGHT_SPEED_M_PER_S / float(scene.frequency)
    free_space_gain = (wavelength / (4.0 * torch.pi * distances)).square()
    return tx_power[:, None] * free_space_gain
