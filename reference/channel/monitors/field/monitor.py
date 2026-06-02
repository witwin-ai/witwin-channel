from __future__ import annotations

from dataclasses import dataclass, field
import math

from ...utils.plane_axes import normalize_axis, tangential_axes_for_axis
from ..common import (
    coordinate_on_axis,
    normalize_bounds,
    normalize_grid_shape,
    normalize_max_diffractions_override,
    normalize_ray_mode,
    normalize_ray_sampling,
    normalize_resolution,
)


@dataclass(slots=True)
class FieldMonitor:
    name: str
    axis: str = "z"
    position: float = 0.0
    bounds: tuple[tuple[float, float], tuple[float, float]] = ((-8.0, 8.0), (-8.0, 8.0))
    grid_size: tuple[int, int] | None = None
    resolution: float | None = None
    ray_mode: str = "2d"
    ray_sampling: str = "full_sphere"
    max_diffractions: int | None = None
    kind: str = field(init=False, default="field")

    def __post_init__(self):
        resolved_grid_size = normalize_grid_shape(self.grid_size)
        resolved_resolution = normalize_resolution(self.resolution)
        self.name = str(self.name)
        self.axis = normalize_axis(self.axis)
        self.position = float(self.position)
        self.bounds = normalize_bounds(self.bounds)
        self.grid_size = resolved_grid_size
        self.resolution = resolved_resolution
        self.ray_mode = normalize_ray_mode(self.ray_mode)
        self.ray_sampling = normalize_ray_sampling(self.ray_sampling)
        self.max_diffractions = normalize_max_diffractions_override(self.max_diffractions)

    @property
    def tangential_axes(self) -> tuple[str, str]:
        return tangential_axes_for_axis(self.axis)

    @property
    def grid_shape(self) -> tuple[int, int] | None:
        return self.grid_size

    def suggested_ray_mode(self, tx_pos, *, near_plane_ratio: float = 0.25) -> str:
        """Recommend ``'2d'`` near the plane and ``'3d'`` otherwise."""

        resolved_ratio = float(near_plane_ratio)
        if resolved_ratio <= 0.0:
            raise ValueError("near_plane_ratio must be > 0.")
        distance_to_plane = abs(coordinate_on_axis(tx_pos, self.axis) - self.position)
        span_0 = self.bounds[0][1] - self.bounds[0][0]
        span_1 = self.bounds[1][1] - self.bounds[1][0]
        characteristic_span = min(span_0, span_1)
        threshold = resolved_ratio * characteristic_span
        return "2d" if distance_to_plane <= threshold else "3d"

    def with_overrides(self, **overrides) -> "FieldMonitor":
        return FieldMonitor(
            overrides.get("name", self.name),
            axis=overrides.get("axis", self.axis),
            position=overrides.get("position", self.position),
            bounds=overrides.get("bounds", self.bounds),
            grid_size=overrides.get("grid_size", self.grid_size),
            resolution=overrides.get("resolution", self.resolution),
            ray_mode=overrides.get("ray_mode", self.ray_mode),
            ray_sampling=overrides.get("ray_sampling", self.ray_sampling),
            max_diffractions=overrides.get("max_diffractions", self.max_diffractions),
        )

    def resolve_grid_shape(
        self,
        wavelength: float,
        *,
        default_resolution: float | None = None,
    ) -> tuple[int, int]:
        if self.grid_size is not None:
            return self.grid_size

        resolved_resolution = self.resolution
        if resolved_resolution is None:
            resolved_resolution = normalize_resolution(default_resolution)
        if resolved_resolution is None:
            raise ValueError(
                "FieldMonitor requires grid_size or resolution, or Tracer must provide a default resolution."
            )

        cell_size = float(resolved_resolution) * float(wavelength)
        nx = max(1, int(math.ceil((self.bounds[0][1] - self.bounds[0][0]) / cell_size)))
        ny = max(1, int(math.ceil((self.bounds[1][1] - self.bounds[1][0]) / cell_size)))
        return (nx, ny)

    def to_field(self, wavelength: float, *, default_resolution: float | None = None):
        from .field import Field

        return Field(
            bounds=self.bounds,
            size=self.resolve_grid_shape(wavelength, default_resolution=default_resolution),
            axis=self.axis,
            position=self.position,
        )


def resolve_field_monitor(monitor) -> FieldMonitor:
    if isinstance(monitor, FieldMonitor):
        return monitor
    raise TypeError("Channel monitors must be FieldMonitor instances.")


__all__ = [
    "FieldMonitor",
    "resolve_field_monitor",
]
