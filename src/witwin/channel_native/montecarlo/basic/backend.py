from __future__ import annotations

import torch

from typing import TYPE_CHECKING
from witwin.channel_native.core.ad_geometry import (
    receiver_positions_ad,
    transmitter_positions_ad,
)
from witwin.channel_native.montecarlo.basic.kernels.maps import (
    mc_los_path_gain_ad,
    mc_los_visibility_inputs,
    mc_zero_matrix,
)
from witwin.channel_native.scene.tensors import (
    LIGHT_SPEED_M_PER_S as _LIGHT_SPEED_M_PER_S,
    receiver_grid_points,
    receiver_positions,
    transmitter_positions,
)

from witwin.channel_native.propagation.geometry.kernels import bridge as geometry_bridge
from witwin.channel_native.propagation.topology.kernels.blocks import path_los_export

if TYPE_CHECKING:
    from witwin.channel_native.scene.models import Scene

__all__ = [
    "_LIGHT_SPEED_M_PER_S",
    "receiver_grid_points",
    "receiver_positions",
    "transmitter_positions",
]


def los_path_gain(
    scene: Scene,
    *,
    device: torch.device,
    ad: bool = False,
    ledger: object | None = None,
) -> torch.Tensor:
    tx_pos, tx_power = transmitter_positions(scene, device=device)
    rx_pos = receiver_positions(scene, device=device, reference=tx_pos)
    if tx_pos.shape[0] == 0 or rx_pos.shape[0] == 0:
        return mc_zero_matrix(tx_pos, rows=tx_pos.shape[0], cols=rx_pos.shape[0])

    if ad:
        # Plan 07 AD-3: swap the host-float endpoint tensors for the live
        # scene leaves (same float32 values) and route through the LoS AD
        # Function so tx/rx position and frequency gradients survive. Grid
        # receiver points stay native: a grid exposes no position leaf.
        tx_live = transmitter_positions_ad(scene, tx_pos, device=device)
        rx_live = receiver_positions_ad(scene, rx_pos, device=device)
        if ledger is not None:
            ledger.add(tx_live, tx_power, rx_live)
        return mc_los_path_gain_ad(
            tx_live, tx_power, rx_live, frequency=scene.frequency
        )
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
        masks.append(
            geometry_bridge.raydn_visibility_forward(
                handle, inputs["start"], rx_pos, inputs["active"]
            )[0]
        )
    return los * torch.stack(masks, dim=0).to(dtype=los.dtype)
