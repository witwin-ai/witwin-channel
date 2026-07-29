# Copyright Xingyu Chen.
# Implements reference channel.

from __future__ import annotations

import torch

from witwin.core import AntennaState, ReceiverGrid, Scene


_LIGHT_SPEED_M_PER_S = 299_792_458.0


def _receiver_blocks(
    scene: Scene, *, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    positions = []
    polarizations = []
    receivers = tuple(endpoint for endpoint in scene.endpoints if endpoint.role == "rx")
    for receiver in receivers:
        if isinstance(receiver, ReceiverGrid):
            points = receiver.points().reshape(-1, 3)
        elif isinstance(receiver, AntennaState):
            points = receiver.position.reshape(-1, 3)
        else:
            raise TypeError(f"unsupported receiver type: {type(receiver).__name__}")
        points = points.to(device=device, dtype=torch.float32, non_blocking=True)
        polarization = receiver.polarization
        if polarization is None:
            polarization = receiver.position.new_tensor((0.0, 0.0, 1.0))
        polarization = polarization.to(device=device, dtype=torch.float32).reshape(3)
        positions.append(points)
        polarizations.append(polarization.expand(points.shape[0], 3))
    return torch.vstack(positions), torch.vstack(polarizations)


def los_path_gain_reference(
    scene: Scene,
    *,
    device: torch.device,
    reference_frequency_hz: float,
) -> torch.Tensor:
    transmitters = tuple(
        endpoint for endpoint in scene.endpoints if endpoint.role == "tx"
    )
    tx_positions = torch.stack([tx.position for tx in transmitters], dim=0).to(
        device=device, dtype=torch.float32, non_blocking=True
    )
    tx_polarizations = torch.stack(
        [
            (
                tx.polarization
                if tx.polarization is not None
                else tx.position.new_tensor((0.0, 0.0, 1.0))
            )
            for tx in transmitters
        ],
        dim=0,
    ).to(device=device, dtype=torch.float32)
    tx_power = torch.tensor(
        [1.0 if tx.power_w is None else tx.power_w for tx in transmitters],
        device=device,
        dtype=torch.float32,
    )
    rx_positions, rx_polarizations = _receiver_blocks(scene, device=device)
    delta = rx_positions[None, :, :] - tx_positions[:, None, :]
    distances = torch.linalg.vector_norm(delta, dim=-1).clamp_min(
        torch.finfo(torch.float32).eps
    )
    khat = delta / distances[..., None]
    wavelength = _LIGHT_SPEED_M_PER_S / float(reference_frequency_hz)
    free_space_gain = (wavelength / (4.0 * torch.pi * distances)).square()
    # the LoS field carries the short-dipole
    # sin(theta) pattern of the transmit polarization (unnormalized transverse
    # projection p_tx - (p_tx.khat) khat), and the exported LoS scalar projects
    # it onto the receiver polarization. path_gain = |scalar|^2 therefore folds
    # in both dipole factors, |p_rx.p_tx - (p_tx.khat)(p_rx.khat)|^2; for the
    # default z-hat polarizations this is sin^2(theta_tx) * sin^2(theta_rx) and
    # equals 1 only when the link lies in a plane orthogonal to z-hat.
    p_tx = tx_polarizations[:, None, :]
    p_rx = rx_polarizations[None, :, :]
    dipole = (
        (p_tx * p_rx).sum(dim=-1)
        - (p_tx * khat).sum(dim=-1) * (p_rx * khat).sum(dim=-1)
    ).square()
    return tx_power[:, None] * free_space_gain * dipole