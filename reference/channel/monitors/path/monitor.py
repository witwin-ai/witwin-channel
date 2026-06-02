from __future__ import annotations

from dataclasses import dataclass, field

import drjit as dr
import witwin as wt

from ..common import (
    normalize_max_diffractions_override,
    normalize_max_num_paths,
    normalize_positions,
    normalize_ray_mode,
)


@dataclass(slots=True)
class PathMonitor:
    name: str
    positions: wt.Point3f
    ray_mode: str = "3d"
    max_num_paths: int | None = None
    max_diffractions: int | None = 1
    return_geometry: bool = False
    kind: str = field(init=False, default="path")

    def __post_init__(self):
        self.name = str(self.name)
        self.positions = normalize_positions(self.positions)
        self.ray_mode = normalize_ray_mode(self.ray_mode)
        self.max_num_paths = normalize_max_num_paths(self.max_num_paths)
        self.max_diffractions = normalize_max_diffractions_override(self.max_diffractions)
        self.return_geometry = bool(self.return_geometry)

    @property
    def num_rx(self) -> int:
        return int(dr.width(self.positions.x))

    def with_overrides(self, **overrides) -> "PathMonitor":
        return PathMonitor(
            overrides.get("name", self.name),
            positions=overrides.get("positions", self.positions),
            ray_mode=overrides.get("ray_mode", self.ray_mode),
            max_num_paths=overrides.get("max_num_paths", self.max_num_paths),
            max_diffractions=overrides.get("max_diffractions", self.max_diffractions),
            return_geometry=overrides.get("return_geometry", self.return_geometry),
        )


def resolve_path_monitor(monitor) -> PathMonitor:
    if isinstance(monitor, PathMonitor):
        return monitor
    raise TypeError("Channel monitors must be PathMonitor instances.")


__all__ = [
    "PathMonitor",
    "resolve_path_monitor",
]
