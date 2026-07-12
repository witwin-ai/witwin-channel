from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import torch


_C0 = 299_792_458.0
_PATTERN_KINDS = frozenset({"isotropic", "vertical", "horizontal", "custom"})


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


@dataclass(frozen=True, slots=True)
class AntennaPattern:
    """Scalar field pattern evaluated in the endpoint's local frame."""

    kind: str = "isotropic"
    custom: Callable[[torch.Tensor], torch.Tensor] | None = None

    def __post_init__(self) -> None:
        if self.kind not in _PATTERN_KINDS:
            raise ValueError(f"pattern kind must be one of {sorted(_PATTERN_KINDS)}")
        if (self.kind == "custom") != (self.custom is not None):
            raise ValueError("custom pattern requires exactly one custom callable")

    def field_response(self, local_direction: torch.Tensor) -> torch.Tensor:
        if local_direction.shape[-1] != 3:
            raise ValueError("local_direction must have a vec3 tail")
        direction = local_direction.to(dtype=torch.float32)
        direction = direction / torch.linalg.vector_norm(
            direction, dim=-1, keepdim=True
        ).clamp_min(torch.finfo(torch.float32).tiny)
        if self.kind == "custom":
            assert self.custom is not None
            response = self.custom(direction)
            if response.shape != direction.shape[:-1]:
                raise ValueError("custom pattern response must match direction batch shape")
            return response.to(device=direction.device, dtype=torch.complex64)
        if self.kind == "vertical":
            response = torch.sqrt(torch.clamp(1.0 - direction[..., 2].square(), min=0.0))
        elif self.kind == "horizontal":
            response = torch.sqrt(torch.clamp(1.0 - direction[..., 0].square(), min=0.0))
        else:
            response = torch.ones(direction.shape[:-1], device=direction.device)
        return response.to(dtype=torch.complex64)


@dataclass(frozen=True, slots=True)
class AntennaArray:
    """Antenna element positions in endpoint-local coordinates, in metres."""

    positions: torch.Tensor

    def __post_init__(self) -> None:
        positions = self.positions
        if positions.ndim != 2 or positions.shape[1] != 3 or positions.shape[0] == 0:
            raise ValueError("array positions must have shape (antenna, 3) with antenna > 0")
        positions = positions.to(dtype=torch.float32).contiguous()
        if not bool(torch.isfinite(positions).all()):
            raise ValueError("array positions must be finite")
        object.__setattr__(self, "positions", positions)

    @property
    def num_antennas(self) -> int:
        return int(self.positions.shape[0])

    @classmethod
    def single(cls) -> AntennaArray:
        return cls(torch.zeros((1, 3), dtype=torch.float32))

    @classmethod
    def ula(
        cls,
        num_antennas: int,
        spacing_m: float,
        *,
        axis: str = "x",
    ) -> AntennaArray:
        if num_antennas <= 0:
            raise ValueError("num_antennas must be positive")
        if spacing_m <= 0.0:
            raise ValueError("spacing_m must be positive")
        axes = {"x": 0, "y": 1, "z": 2}
        if axis not in axes:
            raise ValueError("ULA axis must be 'x', 'y', or 'z'")
        positions = torch.zeros((num_antennas, 3), dtype=torch.float32)
        offset = torch.arange(num_antennas, dtype=torch.float32) - 0.5 * (num_antennas - 1)
        positions[:, axes[axis]] = offset * float(spacing_m)
        return cls(positions)

    @classmethod
    def ura(
        cls,
        rows: int,
        columns: int,
        spacing_m: tuple[float, float],
        *,
        axes: tuple[str, str] = ("x", "y"),
    ) -> AntennaArray:
        if rows <= 0 or columns <= 0:
            raise ValueError("URA rows and columns must be positive")
        if spacing_m[0] <= 0.0 or spacing_m[1] <= 0.0:
            raise ValueError("URA spacing must be positive")
        axis_ids = {"x": 0, "y": 1, "z": 2}
        if axes[0] not in axis_ids or axes[1] not in axis_ids or axes[0] == axes[1]:
            raise ValueError("URA axes must be two different values from 'x', 'y', and 'z'")
        positions = torch.zeros((rows * columns, 3), dtype=torch.float32)
        row_offset = torch.arange(rows, dtype=torch.float32) - 0.5 * (rows - 1)
        column_offset = torch.arange(columns, dtype=torch.float32) - 0.5 * (columns - 1)
        row_grid, column_grid = torch.meshgrid(row_offset, column_offset, indexing="ij")
        positions[:, axis_ids[axes[0]]] = row_grid.reshape(-1) * float(spacing_m[0])
        positions[:, axis_ids[axes[1]]] = column_grid.reshape(-1) * float(spacing_m[1])
        return cls(positions)

    def world_positions(self, origin: torch.Tensor, orientation: torch.Tensor) -> torch.Tensor:
        origin = _vector3("origin", origin)
        rotation = orientation_matrix(orientation).to(device=self.positions.device)
        return origin.to(device=self.positions.device) + self.positions @ rotation.T


def steering_vector(
    array: AntennaArray,
    direction: torch.Tensor,
    *,
    frequency_hz: float,
    orientation: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return ``exp(+j k r·u)`` under the package's ``exp(-j k d)`` convention."""

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
    "AntennaArray",
    "AntennaPattern",
    "apply_endpoint_weights",
    "apply_precoding_combining",
    "orientation_matrix",
    "steering_vector",
    "validate_scalar_endpoint_features",
]
