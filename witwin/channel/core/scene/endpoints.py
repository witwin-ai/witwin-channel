from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from witwin.channel import types as wt

from witwin.channel.core.grid import GridSpec

EndpointPosition = tuple[float, float, float] | wt.Point3f


def _coerce_name(value, *, role: str) -> str:
    name = str(value)
    if not name:
        raise ValueError(f"{role} name must not be empty.")
    return name


def _coerce_vec3(value, *, role: str, field: str) -> tuple[float, float, float] | None:
    if value is None:
        return None
    if len(value) != 3:
        raise ValueError(f"{role} {field} must contain exactly three components.")
    return (float(value[0]), float(value[1]), float(value[2]))


@dataclass(slots=True)
class Transmitter:
    """Scene-owned transmitter endpoint."""

    name: str
    position: EndpointPosition
    polarization: tuple[float, float, float] = (1.0, 0.0, 0.0)
    power: float = 1.0
    orientation: tuple[float, float, float] | None = None
    array: object | None = None

    def __post_init__(self) -> None:
        power = float(self.power)
        if power <= 0.0:
            raise ValueError("Transmitter power must be > 0.")
        self.name = _coerce_name(self.name, role="Transmitter")
        self.polarization = _coerce_vec3(self.polarization, role="Transmitter", field="polarization")
        self.power = power
        self.orientation = _coerce_vec3(self.orientation, role="Transmitter", field="orientation")


@dataclass(slots=True)
class Receiver:
    """Scene-owned point receiver endpoint."""

    name: str
    position: EndpointPosition
    polarization: tuple[float, float, float] | None = None
    orientation: tuple[float, float, float] | None = None
    array: object | None = None

    def __post_init__(self) -> None:
        self.name = _coerce_name(self.name, role="Receiver")
        self.polarization = _coerce_vec3(self.polarization, role="Receiver", field="polarization")
        self.orientation = _coerce_vec3(self.orientation, role="Receiver", field="orientation")


@dataclass(slots=True)
class ReceiverGrid:
    """Scene-owned axis-aligned receiver grid endpoint."""

    name: str
    axis: Literal["x", "y", "z"]
    position: float
    bounds: tuple[tuple[float, float], tuple[float, float]]
    grid_shape: tuple[int, int] | None = None
    cell_size: float | tuple[float, float] | None = None
    polarization: tuple[float, float, float] | None = None
    orientation: tuple[float, float, float] | None = None
    array: object | None = None

    def __post_init__(self) -> None:
        spec = GridSpec(axis=self.axis, position=self.position, bounds=self.bounds,
                        grid_shape=self.grid_shape, cell_size=self.cell_size)
        self.name = _coerce_name(self.name, role="ReceiverGrid")
        self.axis = spec.axis
        self.position = spec.position
        self.bounds = spec.bounds
        self.grid_shape = spec.grid_shape
        self.cell_size = spec.cell_size
        self.polarization = _coerce_vec3(self.polarization, role="ReceiverGrid", field="polarization")
        self.orientation = _coerce_vec3(self.orientation, role="ReceiverGrid", field="orientation")


__all__ = ["EndpointPosition", "Receiver", "ReceiverGrid", "Transmitter"]
