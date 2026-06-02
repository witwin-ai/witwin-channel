from __future__ import annotations

from ..common import normalize_bounds, normalize_grid_shape
from ...utils.plane_axes import normalize_axis
from . import types as rm_types


def normalize_point2(value, *, name: str) -> tuple[float, float]:
    if len(value) != 2:
        raise ValueError(f"{name} must contain exactly two values.")
    x = float(value[0])
    y = float(value[1])
    if x <= 0.0 or y <= 0.0:
        raise ValueError(f"{name} values must be > 0.")
    return (x, y)


def normalize_point3(value, *, name: str) -> tuple[float, float, float]:
    if len(value) != 3:
        raise ValueError(f"{name} must contain exactly three values.")
    return (float(value[0]), float(value[1]), float(value[2]))


def normalize_optional_point2(value, *, name: str) -> tuple[float, float] | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        resolved = float(value)
        if resolved <= 0.0:
            raise ValueError(f"{name} must be > 0.")
        return (resolved, resolved)
    return normalize_point2(value, name=name)


def normalize_quadrature_mode(
    quadrature_mode: str,
    *,
    samples_per_cell: int | None,
) -> tuple[str, int]:
    resolved_mode = str(quadrature_mode).lower()
    if resolved_mode == "center":
        if samples_per_cell is not None and int(samples_per_cell) != 1:
            raise ValueError("samples_per_cell must be 1 when quadrature_mode='center'.")
        return ("center", 1)
    if resolved_mode == "stratified_fixed_n":
        if samples_per_cell is None:
            raise ValueError(
                "samples_per_cell is required when quadrature_mode='stratified_fixed_n'."
            )
        resolved_samples = int(samples_per_cell)
        if resolved_samples <= 0:
            raise ValueError("samples_per_cell must be > 0.")
        return ("stratified_fixed_n", resolved_samples)
    raise ValueError("quadrature_mode must be 'center' or 'stratified_fixed_n'.")


def resolve_surface_fields(
    *,
    axis,
    position,
    bounds,
    center,
    orientation,
    size,
    grid_shape,
    cell_size,
):
    resolved_grid_shape = normalize_grid_shape(grid_shape)
    resolved_cell_size = normalize_optional_point2(cell_size, name="cell_size")
    has_oriented_args = any(value is not None for value in (center, orientation, size))
    if has_oriented_args and not all(value is not None for value in (center, orientation, size)):
        raise ValueError(
            "center, orientation, and size must all be provided for an oriented radio map surface."
        )
    if has_oriented_args:
        surface_mode = rm_types.SurfaceMode.ORIENTED
        resolved_axis = None
        resolved_position = None
        resolved_bounds = None
        resolved_center = normalize_point3(center, name="center")
        resolved_orientation = normalize_point3(orientation, name="orientation")
        resolved_size = normalize_point2(size, name="size")
    else:
        surface_mode = rm_types.SurfaceMode.AXIS_ALIGNED
        resolved_axis = normalize_axis(axis)
        resolved_position = float(position)
        resolved_bounds = normalize_bounds(bounds)
        resolved_center = None
        resolved_orientation = None
        resolved_size = None
    if resolved_grid_shape is not None and resolved_cell_size is not None:
        raise ValueError("Provide either grid_shape or cell_size, not both.")
    return {
        "surface_mode": surface_mode,
        "axis": resolved_axis,
        "position": resolved_position,
        "bounds": resolved_bounds,
        "center": resolved_center,
        "orientation": resolved_orientation,
        "size": resolved_size,
        "grid_shape": resolved_grid_shape,
        "cell_size": resolved_cell_size,
    }


__all__ = [
    "normalize_optional_point2",
    "normalize_quadrature_mode",
    "resolve_surface_fields",
]
