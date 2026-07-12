from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Protocol

import torch

from .antenna import AntennaArray, AntennaPattern, orientation_matrix


class Material(Protocol):
    def parameters(self) -> dict[str, float | int]:
        ...


def _as_vector3(name: str, tensor: torch.Tensor) -> torch.Tensor:
    if tensor.shape != (3,):
        raise ValueError(f"{name} must have shape (3,)")
    return tensor.to(dtype=torch.float32).contiguous()


def _as_polarization(
    value: torch.Tensor | None,
    *,
    pattern: AntennaPattern,
    orientation: torch.Tensor,
) -> torch.Tensor:
    default = [1.0, 0.0, 0.0] if pattern.kind == "horizontal" else [0.0, 0.0, 1.0]
    polarization = (
        torch.tensor(default, dtype=torch.float32)
        if value is None
        else _as_vector3("polarization", value)
    )
    norm = torch.linalg.vector_norm(polarization)
    if not bool(torch.isfinite(norm)) or float(norm) <= 0.0:
        raise ValueError("polarization must be finite and non-zero")
    rotation = orientation_matrix(orientation).to(device=polarization.device)
    return (rotation @ (polarization / norm)).contiguous()


def _as_orientation(value: torch.Tensor | None) -> torch.Tensor:
    return (
        torch.zeros(3, dtype=torch.float32)
        if value is None
        else _as_vector3("orientation", value)
    )


def _as_pattern(value: AntennaPattern | str) -> AntennaPattern:
    return AntennaPattern(value) if isinstance(value, str) else value


def _as_array(value: AntennaArray | None) -> AntennaArray:
    return AntennaArray.single() if value is None else value


def _as_weights(
    name: str, value: torch.Tensor | None, *, num_antennas: int
) -> torch.Tensor | None:
    if value is None:
        return None
    if value.shape != (num_antennas,):
        raise ValueError(f"{name} must have shape ({num_antennas},)")
    value = value.to(dtype=torch.complex64).contiguous()
    if not bool(torch.isfinite(value.real).all() & torch.isfinite(value.imag).all()):
        raise ValueError(f"{name} must be finite")
    return value


@dataclass(frozen=True, slots=True)
class Structure:
    vertices: torch.Tensor
    faces: torch.Tensor
    material: Material
    name: str = ""
    surface_id: int = 0
    metadata: dict[str, object] | None = None

    def __post_init__(self) -> None:
        if self.vertices.ndim != 2 or self.vertices.shape[1] != 3:
            raise ValueError("vertices must have shape (N, 3)")
        if self.faces.ndim != 2 or self.faces.shape[1] != 3:
            raise ValueError("faces must have shape (F, 3)")
        object.__setattr__(self, "vertices", self.vertices.to(dtype=torch.float32).contiguous())
        object.__setattr__(self, "faces", self.faces.to(dtype=torch.int32).contiguous())

    def with_vertices(self, vertices: torch.Tensor) -> Structure:
        return replace(self, vertices=vertices)

    def with_material(self, material: Material) -> Structure:
        return replace(self, material=material)


@dataclass(frozen=True, slots=True)
class Transmitter:
    position: torch.Tensor
    power_w: float = 1.0
    polarization: torch.Tensor | None = None
    orientation: torch.Tensor | None = None
    pattern: AntennaPattern | str = "isotropic"
    array: AntennaArray | None = None
    synthetic_array: bool = True
    precoding: torch.Tensor | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "position", _as_vector3("position", self.position))
        orientation = _as_orientation(self.orientation)
        pattern = _as_pattern(self.pattern)
        array = _as_array(self.array)
        object.__setattr__(self, "orientation", orientation)
        object.__setattr__(self, "pattern", pattern)
        object.__setattr__(self, "array", array)
        object.__setattr__(
            self,
            "polarization",
            _as_polarization(
                self.polarization, pattern=pattern, orientation=orientation
            ),
        )
        object.__setattr__(
            self,
            "precoding",
            _as_weights("precoding", self.precoding, num_antennas=array.num_antennas),
        )


@dataclass(frozen=True, slots=True)
class ReceiverPoint:
    position: torch.Tensor
    polarization: torch.Tensor | None = None
    orientation: torch.Tensor | None = None
    pattern: AntennaPattern | str = "isotropic"
    array: AntennaArray | None = None
    synthetic_array: bool = True
    combining: torch.Tensor | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "position", _as_vector3("position", self.position))
        orientation = _as_orientation(self.orientation)
        pattern = _as_pattern(self.pattern)
        array = _as_array(self.array)
        object.__setattr__(self, "orientation", orientation)
        object.__setattr__(self, "pattern", pattern)
        object.__setattr__(self, "array", array)
        object.__setattr__(
            self,
            "polarization",
            _as_polarization(
                self.polarization, pattern=pattern, orientation=orientation
            ),
        )
        object.__setattr__(
            self,
            "combining",
            _as_weights("combining", self.combining, num_antennas=array.num_antennas),
        )


@dataclass(frozen=True, slots=True)
class ReceiverGrid:
    origin: torch.Tensor
    x_axis: torch.Tensor
    y_axis: torch.Tensor
    shape: tuple[int, int]
    spacing: tuple[float, float]
    polarization: torch.Tensor | None = None
    orientation: torch.Tensor | None = None
    pattern: AntennaPattern | str = "isotropic"
    array: AntennaArray | None = None
    synthetic_array: bool = True
    combining: torch.Tensor | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "origin", _as_vector3("origin", self.origin))
        object.__setattr__(self, "x_axis", _as_vector3("x_axis", self.x_axis))
        object.__setattr__(self, "y_axis", _as_vector3("y_axis", self.y_axis))
        orientation = _as_orientation(self.orientation)
        pattern = _as_pattern(self.pattern)
        array = _as_array(self.array)
        object.__setattr__(self, "orientation", orientation)
        object.__setattr__(self, "pattern", pattern)
        object.__setattr__(self, "array", array)
        object.__setattr__(
            self,
            "polarization",
            _as_polarization(
                self.polarization, pattern=pattern, orientation=orientation
            ),
        )
        object.__setattr__(
            self,
            "combining",
            _as_weights("combining", self.combining, num_antennas=array.num_antennas),
        )
        if len(self.shape) != 2 or self.shape[0] <= 0 or self.shape[1] <= 0:
            raise ValueError("shape must be two positive integers")
        if len(self.spacing) != 2 or self.spacing[0] <= 0.0 or self.spacing[1] <= 0.0:
            raise ValueError("spacing must be two positive values")

    def points(self) -> torch.Tensor:
        rows, cols = self.shape
        origin = (float(self.origin[0]), float(self.origin[1]), float(self.origin[2]))
        x_axis = (float(self.x_axis[0]), float(self.x_axis[1]), float(self.x_axis[2]))
        y_axis = (float(self.y_axis[0]), float(self.y_axis[1]), float(self.y_axis[2]))
        points: list[tuple[float, float, float]] = []
        for i in range(rows):
            for j in range(cols):
                x_weight = i * self.spacing[0]
                y_weight = j * self.spacing[1]
                points.append(
                    (
                        origin[0] + x_axis[0] * x_weight + y_axis[0] * y_weight,
                        origin[1] + x_axis[1] * x_weight + y_axis[1] * y_weight,
                        origin[2] + x_axis[2] * x_weight + y_axis[2] * y_weight,
                    )
                )
        return torch.tensor(points, dtype=torch.float32)
