from __future__ import annotations

import torch

from witwin.channel_native import Scene
from witwin.channel_native.core.kernels.ops import (
    mc_los_visibility_inputs,
    mc_zero_matrix,
    path_los_export,
    raydn_visibility_forward,
)
from witwin.channel_native.core.scene_tensors import (
    LIGHT_SPEED_M_PER_S as _LIGHT_SPEED_M_PER_S,
    receiver_grid_points,
    receiver_positions,
    transmitter_positions,
)

__all__ = [
    "_LIGHT_SPEED_M_PER_S",
    "receiver_grid_points",
    "receiver_positions",
    "transmitter_positions",
]


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
