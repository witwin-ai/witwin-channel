from __future__ import annotations

import math
from collections.abc import Sequence

import torch

from witwin.core import AntennaPattern as _CoreAntennaPattern


_C0 = 299_792_458.0


def _vector3(name: str, value: torch.Tensor) -> torch.Tensor:
    if value.shape != (3,):
        raise ValueError(f"{name} must have shape (3,)")
    value = value.to(dtype=torch.float32).contiguous()
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} must be finite")
    return value


def orientation_matrix(orientation: torch.Tensor) -> torch.Tensor:
    """Return the local-to-world yaw/pitch/roll rotation matrix.

    The orientation vector is ``(yaw, pitch, roll)`` in radians and uses the
    intrinsic Z-Y-X convention.
    """

    if orientation.shape == (4,):
        from witwin.core.math import quat_to_rotation_matrix

        return quat_to_rotation_matrix(orientation).to(dtype=torch.float32)
    yaw, pitch, roll = _vector3("orientation", orientation).unbind()
    cy, sy = torch.cos(yaw), torch.sin(yaw)
    cp, sp = torch.cos(pitch), torch.sin(pitch)
    cr, sr = torch.cos(roll), torch.sin(roll)
    return torch.stack(
        (
            torch.stack((cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr)),
            torch.stack((sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr)),
            torch.stack((-sp, cp * sr, cp * cr)),
        )
    ).to(dtype=torch.float32)


def pattern_field_response(
    pattern: _CoreAntennaPattern,
    local_direction: torch.Tensor,
) -> torch.Tensor:
    """Evaluate one canonical Core antenna pattern in the endpoint-local frame."""

    if not isinstance(pattern, _CoreAntennaPattern):
        raise TypeError("pattern must be a witwin.core.AntennaPattern")
    if local_direction.shape[-1] != 3:
        raise ValueError("local_direction must have a vec3 tail")
    direction = local_direction.to(dtype=torch.float32)
    direction = direction / torch.linalg.vector_norm(
        direction, dim=-1, keepdim=True
    ).clamp_min(torch.finfo(torch.float32).tiny)
    if pattern.kind == "custom":
        assert pattern.custom is not None
        response = pattern.custom(direction)
        if response.shape != direction.shape[:-1]:
            raise ValueError("custom pattern response must match direction batch shape")
        return response.to(device=direction.device, dtype=torch.complex64)
    if pattern.kind == "vertical":
        response = torch.sqrt(
            torch.clamp(1.0 - direction[..., 2].square(), min=0.0)
        )
    elif pattern.kind == "horizontal":
        response = torch.sqrt(
            torch.clamp(1.0 - direction[..., 0].square(), min=0.0)
        )
    else:
        response = torch.ones(direction.shape[:-1], device=direction.device)
    return response.to(dtype=torch.complex64)


def steering_vector(
    array: object,
    direction: torch.Tensor,
    *,
    frequency_hz: float,
    orientation: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return ``exp(+j k r·u)`` under the package's ``exp(-j k d)`` convention."""

    if isinstance(frequency_hz, torch.Tensor):
        # Synthetic-array steering is evaluated at the primal frequency; its
        # frequency derivative is exactly zero for single-element centre
        # arrays and detached otherwise (plan 07 AD-1 fixed-array contract).
        frequency_hz = float(frequency_hz.detach())
    if frequency_hz <= 0.0:
        raise ValueError("frequency_hz must be positive")
    if direction.shape[-1] != 3:
        raise ValueError("direction must have a vec3 tail")
    device = direction.device
    positions = array.positions.to(device=device)
    if orientation is not None:
        rotation = orientation_matrix(orientation).to(device=device)
        positions = positions @ rotation.T
    unit = direction.to(dtype=torch.float32)
    unit = unit / torch.linalg.vector_norm(unit, dim=-1, keepdim=True).clamp_min(
        torch.finfo(torch.float32).tiny
    )
    phase = (2.0 * math.pi * float(frequency_hz) / _C0) * torch.einsum(
        "...c,ac->...a", unit, positions
    )
    return torch.exp(1.0j * phase).to(torch.complex64)


def apply_precoding_combining(
    coefficients: torch.Tensor,
    *,
    tx_weights: torch.Tensor,
    rx_weights: torch.Tensor,
) -> torch.Tensor:
    """Combine ``(..., rx_ant, tx_ant)`` channel coefficients into one stream."""

    if coefficients.ndim < 2:
        raise ValueError("coefficients must have rx_ant and tx_ant tail dimensions")
    if tx_weights.shape != (coefficients.shape[-1],):
        raise ValueError("tx_weights must match tx_ant")
    if rx_weights.shape != (coefficients.shape[-2],):
        raise ValueError("rx_weights must match rx_ant")
    tx = tx_weights.to(device=coefficients.device, dtype=torch.complex64)
    rx = rx_weights.to(device=coefficients.device, dtype=torch.complex64)
    return torch.einsum("...rt,t,r->...", coefficients, tx, rx.conj())


def apply_endpoint_weights(
    coefficients: torch.Tensor,
    *,
    tx_weights: torch.Tensor,
    rx_weights: torch.Tensor,
) -> torch.Tensor:
    """Combine ``(rx, rx_ant, tx, tx_ant, ...)`` endpoint channels.

    The leading endpoint and antenna dimensions match :class:`PathResult`.
    Any trailing signal dimensions (path, time, frequency, or tap) are
    preserved.  Receiver weights follow the usual conjugating convention.
    """

    if coefficients.ndim < 4:
        raise ValueError(
            "coefficients must have (rx, rx_ant, tx, tx_ant, ...) dimensions"
        )
    expected_tx = (coefficients.shape[2], coefficients.shape[3])
    expected_rx = (coefficients.shape[0], coefficients.shape[1])
    if tx_weights.shape != expected_tx:
        raise ValueError(f"tx_weights must have shape {expected_tx}")
    if rx_weights.shape != expected_rx:
        raise ValueError(f"rx_weights must have shape {expected_rx}")
    tx = tx_weights.to(device=coefficients.device, dtype=torch.complex64)
    rx = rx_weights.to(device=coefficients.device, dtype=torch.complex64)
    tail = (1,) * (coefficients.ndim - 4)
    weighted = coefficients * tx.reshape(1, 1, *expected_tx, *tail)
    weighted = weighted * rx.conj().reshape(*expected_rx, 1, 1, *tail)
    return weighted.sum(dim=3).sum(dim=1)


def validate_scalar_endpoint_features(
    transmitters: Sequence[object],
    receivers: Sequence[object],
    *,
    solver: str,
) -> None:
    """Reject endpoint features that a scalar/power solver cannot consume."""

    for endpoint in (*tuple(transmitters), *tuple(receivers)):
        if endpoint.array.num_antennas != 1:
            raise ValueError(f"{solver} does not support antenna arrays")
        if endpoint.pattern.kind != "isotropic":
            raise ValueError(
                f"{solver} does not support directional antenna patterns"
            )
        weights = (
            endpoint.precoding
            if hasattr(endpoint, "precoding")
            else endpoint.combining
        )
        if weights is not None:
            raise ValueError(f"{solver} does not support precoding or combining")


__all__ = [
    "apply_endpoint_weights",
    "apply_precoding_combining",
    "orientation_matrix",
    "pattern_field_response",
    "steering_vector",
    "validate_scalar_endpoint_features",
]
