from __future__ import annotations

import drjit as dr
import numpy as np
import torch
import witwin as wt

from ..utils.plane_axes import normalize_axis
from ..utils import scalar


def normalize_ray_mode(ray_mode: str) -> str:
    mode = str(ray_mode).lower()
    if mode not in {"2d", "3d"}:
        raise ValueError("ray_mode must be '2d' or '3d'.")
    return mode


def normalize_ray_sampling(ray_sampling: str) -> str:
    sampling = str(ray_sampling).lower()
    if sampling not in {"auto", "full_sphere", "hemisphere"}:
        raise ValueError("ray_sampling must be 'auto', 'full_sphere', or 'hemisphere'.")
    return sampling


def normalize_bounds(bounds) -> tuple[tuple[float, float], tuple[float, float]]:
    if len(bounds) != 2:
        raise ValueError("bounds must contain exactly two tangential axis ranges.")
    normalized = []
    for axis_bounds in bounds:
        if len(axis_bounds) != 2:
            raise ValueError("Each tangential bound must contain exactly two values.")
        lower = float(axis_bounds[0])
        upper = float(axis_bounds[1])
        if upper <= lower:
            raise ValueError("Each tangential bound must satisfy upper > lower.")
        normalized.append((lower, upper))
    return tuple(normalized)


def normalize_grid_shape(grid_size) -> tuple[int, int] | None:
    if grid_size is None:
        return None
    if isinstance(grid_size, int):
        if grid_size <= 0:
            raise ValueError("grid_size must be > 0.")
        return (int(grid_size), int(grid_size))
    if len(grid_size) != 2:
        raise ValueError("grid_size must be an int or a two-value shape.")
    nx = int(grid_size[0])
    ny = int(grid_size[1])
    if nx <= 0 or ny <= 0:
        raise ValueError("grid_size values must be > 0.")
    return (nx, ny)


def normalize_resolution(resolution: float | None) -> float | None:
    if resolution is None:
        return None
    resolved = float(resolution)
    if resolved <= 0.0:
        raise ValueError("resolution must be > 0.")
    return resolved


def normalize_positions(positions):
    if isinstance(positions, wt.Point3f):
        if dr.width(positions.x) == 0:
            raise ValueError("positions must contain at least one receiver point.")
        return positions

    tensor = torch.as_tensor(positions, dtype=torch.float32)
    if tensor.ndim != 2 or tensor.shape[1] != 3:
        raise ValueError("positions must have shape (N, 3).")
    if tensor.shape[0] == 0:
        raise ValueError("positions must contain at least one receiver point.")
    tensor = tensor.contiguous()
    return wt.Point3f(
        wt.Float(tensor[:, 0]),
        wt.Float(tensor[:, 1]),
        wt.Float(tensor[:, 2]),
    )


def normalize_max_num_paths(max_num_paths: int | None) -> int | None:
    if max_num_paths is None:
        return None
    resolved = int(max_num_paths)
    if resolved <= 0:
        raise ValueError("max_num_paths must be > 0.")
    return resolved


def normalize_max_diffractions_override(max_diffractions: int | None) -> int | None:
    if max_diffractions is None:
        return None
    resolved = int(max_diffractions)
    if resolved < 0:
        raise ValueError("max_diffractions must be >= 0 when provided.")
    return resolved


def coordinate_on_axis(value, axis: str) -> float:
    axis_name = normalize_axis(axis)
    if hasattr(value, axis_name):
        return float(scalar(getattr(value, axis_name)))
    axis_index = {"x": 0, "y": 1, "z": 2}[axis_name]
    return float(value[axis_index])


def group_positions_by_z_coordinate(positions) -> list[tuple[float, np.ndarray]]:
    z_values = np.asarray(positions.z, dtype=np.float32)
    if z_values.size == 0:
        return []
    quantized_z = np.rint(z_values * 1e6).astype(np.int64)
    ordered_groups = []
    for group_key in np.unique(quantized_z):
        group_indices = np.nonzero(quantized_z == group_key)[0].astype(np.int64, copy=False)
        ordered_groups.append((float(z_values[group_indices[0]]), group_indices))
    return ordered_groups


__all__ = [
    "coordinate_on_axis",
    "group_positions_by_z_coordinate",
    "normalize_bounds",
    "normalize_max_diffractions_override",
    "normalize_grid_shape",
    "normalize_max_num_paths",
    "normalize_positions",
    "normalize_ray_mode",
    "normalize_ray_sampling",
    "normalize_resolution",
]
