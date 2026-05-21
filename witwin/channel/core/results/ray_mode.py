"""Shared ray-dimensionality controls for channel solvers."""

from __future__ import annotations

from typing import Literal, cast

RayMode = Literal["2d", "3d"]

RAY_MODE_2D: RayMode = "2d"
RAY_MODE_3D: RayMode = "3d"
DEFAULT_RAY_MODE: RayMode = RAY_MODE_3D
_RAY_MODES = {RAY_MODE_2D, RAY_MODE_3D}


def normalize_ray_mode(ray_mode: str) -> RayMode:
    resolved = str(ray_mode).lower()
    if resolved not in _RAY_MODES:
        raise ValueError("ray_mode must be one of '2d' or '3d'.")
    return cast(RayMode, resolved)


__all__ = [
    "DEFAULT_RAY_MODE",
    "RAY_MODE_2D",
    "RAY_MODE_3D",
    "RayMode",
    "normalize_ray_mode",
]
