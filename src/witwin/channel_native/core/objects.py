from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Protocol

import torch


class Material(Protocol):
    def parameters(self) -> dict[str, float | int]:
        ...


def _as_vector3(name: str, tensor: torch.Tensor) -> torch.Tensor:
    if tensor.shape != (3,):
        raise ValueError(f"{name} must have shape (3,)")
    return tensor.to(dtype=torch.float32).contiguous()


@dataclass(frozen=True, slots=True)
class Structure:
    vertices: torch.Tensor
    faces: torch.Tensor
    material: Material
    name: str = ""
    surface_id: int = 0

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

    def __post_init__(self) -> None:
        object.__setattr__(self, "position", _as_vector3("position", self.position))


@dataclass(frozen=True, slots=True)
class ReceiverPoint:
    position: torch.Tensor

    def __post_init__(self) -> None:
        object.__setattr__(self, "position", _as_vector3("position", self.position))


@dataclass(frozen=True, slots=True)
class ReceiverGrid:
    origin: torch.Tensor
    x_axis: torch.Tensor
    y_axis: torch.Tensor
    shape: tuple[int, int]
    spacing: tuple[float, float]

    def __post_init__(self) -> None:
        object.__setattr__(self, "origin", _as_vector3("origin", self.origin))
        object.__setattr__(self, "x_axis", _as_vector3("x_axis", self.x_axis))
        object.__setattr__(self, "y_axis", _as_vector3("y_axis", self.y_axis))
        if len(self.shape) != 2 or self.shape[0] <= 0 or self.shape[1] <= 0:
            raise ValueError("shape must be two positive integers")
        if len(self.spacing) != 2 or self.spacing[0] <= 0.0 or self.spacing[1] <= 0.0:
            raise ValueError("spacing must be two positive values")

    def points(self) -> torch.Tensor:
        rows, cols = self.shape
        points = []
        for i in range(rows):
            for j in range(cols):
                points.append(
                    self.origin
                    + self.x_axis * (i * self.spacing[0])
                    + self.y_axis * (j * self.spacing[1])
                )
        return torch.stack(points, dim=0)
