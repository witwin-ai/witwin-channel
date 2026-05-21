"""Shared result containers and result-level solver controls."""

from .radiomap_result import (
    RadioMapCoordinates,
    RadioMapFieldPayload,
    RadioMapPowerPayload,
    RadioMapResult,
    coordinates_from_grid,
    stack_radiomap_results,
)
from .ray_mode import DEFAULT_RAY_MODE, RAY_MODE_2D, RAY_MODE_3D, RayMode, normalize_ray_mode

__all__ = [
    "DEFAULT_RAY_MODE",
    "RAY_MODE_2D",
    "RAY_MODE_3D",
    "RayMode",
    "RadioMapCoordinates",
    "RadioMapFieldPayload",
    "RadioMapPowerPayload",
    "RadioMapResult",
    "coordinates_from_grid",
    "normalize_ray_mode",
    "stack_radiomap_results",
]
