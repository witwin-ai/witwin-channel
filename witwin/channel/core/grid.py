"""Shared axis-aligned receiver grid for radiomap solvers."""

from __future__ import annotations

import math
from dataclasses import dataclass

import drjit as dr
from witwin.channel import types as wt

from .geometry import normalize_axis, tangential_axes_for_axis


_AXIS_BASIS = {
    "x": ((0.0, 1.0, 0.0), (0.0, 0.0, 1.0), (1.0, 0.0, 0.0)),
    "y": ((1.0, 0.0, 0.0), (0.0, 0.0, 1.0), (0.0, 1.0, 0.0)),
    "z": ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
}


def axis_basis(axis: str):
    """Return (basis_u, basis_v, normal) for an axis-aligned plane."""
    return _AXIS_BASIS[normalize_axis(axis)]


def cell_center_coordinates(*, span: tuple[float, float], grid_shape: tuple[int, int]):
    """Build (x_coords, y_coords, grid_x, grid_y) on the centered span."""
    nx, ny = int(grid_shape[0]), int(grid_shape[1])
    span_x, span_y = float(span[0]), float(span[1])
    x_coords = (dr.arange(wt.Float, nx) + 0.5) * (span_x / nx) - 0.5 * span_x
    y_coords = (dr.arange(wt.Float, ny) + 0.5) * (span_y / ny) - 0.5 * span_y
    grid_x = dr.tile(x_coords, ny)
    grid_y = dr.repeat(y_coords, nx)
    dr.eval(x_coords, y_coords, grid_x, grid_y)
    return x_coords, y_coords, grid_x, grid_y


def _resolve_shape_and_cell_size(
    *,
    bounds: tuple[tuple[float, float], tuple[float, float]],
    grid_shape: tuple[int, int] | None,
    cell_size: float | tuple[float, float] | None,
    default_cell_size: float | tuple[float, float] | None,
):
    span = (bounds[0][1] - bounds[0][0], bounds[1][1] - bounds[1][0])
    if grid_shape is not None:
        shape = (int(grid_shape[0]), int(grid_shape[1]))
    else:
        requested = cell_size if cell_size is not None else default_cell_size
        if requested is None:
            raise ValueError(
                "GridSpec requires grid_shape or cell_size, "
                "or solver must provide a default_cell_size."
            )
        cs = requested if isinstance(requested, tuple) else (requested, requested)
        cs = (float(cs[0]), float(cs[1]))
        shape = (max(1, int(math.ceil(span[0] / cs[0]))), max(1, int(math.ceil(span[1] / cs[1]))))
    return shape, (span[0] / shape[0], span[1] / shape[1]), span


@dataclass(slots=True)
class GridSpec:
    """Axis-aligned receiver grid spec shared by all radiomap solvers."""

    axis: str
    position: float
    bounds: tuple[tuple[float, float], tuple[float, float]]
    grid_shape: tuple[int, int] | None = None
    cell_size: float | tuple[float, float] | None = None

    def __post_init__(self) -> None:
        bounds = (
            (float(self.bounds[0][0]), float(self.bounds[0][1])),
            (float(self.bounds[1][0]), float(self.bounds[1][1])),
        )
        if bounds[0][0] >= bounds[0][1] or bounds[1][0] >= bounds[1][1]:
            raise ValueError("bounds must be ordered as ((min_0, max_0), (min_1, max_1)).")
        if (self.grid_shape is None) == (self.cell_size is None):
            raise ValueError("GridSpec requires exactly one of grid_shape or cell_size.")
        self.axis = normalize_axis(self.axis)
        self.position = float(self.position)
        self.bounds = bounds
        if self.grid_shape is not None:
            shape = (int(self.grid_shape[0]), int(self.grid_shape[1]))
            if shape[0] <= 0 or shape[1] <= 0:
                raise ValueError("grid_shape must contain positive integers.")
            self.grid_shape = shape
        if self.cell_size is not None:
            cs = self.cell_size if isinstance(self.cell_size, tuple) else (self.cell_size, self.cell_size)
            cs = (float(cs[0]), float(cs[1]))
            if cs[0] <= 0.0 or cs[1] <= 0.0:
                raise ValueError("cell_size must be > 0.")
            self.cell_size = cs


@dataclass(frozen=True, slots=True)
class GridSample:
    """One quadrature sample on a receiver cell."""

    index: int
    offset_local: tuple[float, float]
    weight: float
    positions: object


@dataclass(frozen=True)
class Grid:
    """Resolved axis-aligned receiver grid with DrJit coordinate arrays."""

    axis: str
    position: float
    bounds: tuple[tuple[float, float], tuple[float, float]]
    grid_shape: tuple[int, int]
    cell_size: tuple[float, float]
    x_coords: object
    y_coords: object
    grid_x: object
    grid_y: object
    cell_centers: object
    surface_mode: str = "axis_aligned"
    sample_sets: tuple[GridSample, ...] = ()

    @property
    def n_cells(self) -> int:
        return int(self.grid_shape[0] * self.grid_shape[1])

    @property
    def tensor_shape(self) -> tuple[int, int]:
        return (int(self.grid_shape[1]), int(self.grid_shape[0]))

    @property
    def tangential_axes(self) -> tuple[str, str]:
        return tangential_axes_for_axis(self.axis)

    @property
    def center(self) -> tuple[float, float, float]:
        mid_0 = 0.5 * (self.bounds[0][0] + self.bounds[0][1])
        mid_1 = 0.5 * (self.bounds[1][0] + self.bounds[1][1])
        if self.axis == "x":
            return (self.position, mid_0, mid_1)
        if self.axis == "y":
            return (mid_0, self.position, mid_1)
        return (mid_0, mid_1, self.position)

    @property
    def size(self) -> tuple[float, float]:
        return (self.bounds[0][1] - self.bounds[0][0], self.bounds[1][1] - self.bounds[1][0])

    def surface_descriptor(self) -> dict[str, object]:
        basis_u, basis_v, normal = axis_basis(self.axis)
        return {
            "surface_mode": str(self.surface_mode),
            "center": tuple(float(v) for v in self.center),
            "orientation": (0.0, 0.0, 0.0),
            "size": tuple(float(v) for v in self.size),
            "basis_u": tuple(float(v) for v in basis_u),
            "basis_v": tuple(float(v) for v in basis_v),
            "normal": tuple(float(v) for v in normal),
            "tangential_axes": self.tangential_axes,
            "axis": self.axis,
            "position": self.position,
            "bounds": tuple(tuple(float(v) for v in pair) for pair in self.bounds),
        }

    @classmethod
    def from_spec(
        cls,
        spec,
        *,
        default_cell_size: float | tuple[float, float] | None = None,
        sample_offsets: tuple[tuple[tuple[float, float], float], ...] | None = None,
        surface_mode: str | None = None,
    ) -> "Grid":
        """Resolve any spec exposing axis/position/bounds/grid_shape/cell_size.

        Optional ``sample_offsets`` is a tuple of ``((offset_x, offset_y), weight)``
        for solvers that need per-cell quadrature samples.
        """
        axis = normalize_axis(spec.axis)
        bounds = tuple(tuple(float(v) for v in pair) for pair in spec.bounds)
        shape, cell_size, span = _resolve_shape_and_cell_size(
            bounds=bounds,
            grid_shape=getattr(spec, "grid_shape", None),
            cell_size=getattr(spec, "cell_size", None),
            default_cell_size=default_cell_size,
        )
        basis_u, basis_v, _ = axis_basis(axis)
        x_base, y_base, gx_base, gy_base = cell_center_coordinates(span=span, grid_shape=shape)
        x_origin = 0.5 * (bounds[0][0] + bounds[0][1])
        y_origin = 0.5 * (bounds[1][0] + bounds[1][1])
        position = float(spec.position)
        center_origin = {
            "x": (position, x_origin, y_origin),
            "y": (x_origin, position, y_origin),
            "z": (x_origin, y_origin, position),
        }[axis]
        cell_centers = wt.Point3f(*(
            wt.Float(center_origin[i]) + basis_u[i] * gx_base + basis_v[i] * gy_base
            for i in range(3)
        ))

        sample_sets = () if sample_offsets is None else tuple(
            GridSample(
                index=i,
                offset_local=(float(ox), float(oy)),
                weight=float(weight),
                positions=wt.Point3f(*(
                    getattr(cell_centers, ("x", "y", "z")[j])
                    + wt.Float(ox) * basis_u[j]
                    + wt.Float(oy) * basis_v[j]
                    for j in range(3)
                )),
            )
            for i, ((ox, oy), weight) in enumerate(sample_offsets)
        )

        return cls(
            axis=axis,
            position=position,
            bounds=bounds,
            grid_shape=(int(shape[0]), int(shape[1])),
            cell_size=(float(cell_size[0]), float(cell_size[1])),
            x_coords=x_base + wt.Float(x_origin),
            y_coords=y_base + wt.Float(y_origin),
            grid_x=gx_base + wt.Float(x_origin),
            grid_y=gy_base + wt.Float(y_origin),
            cell_centers=cell_centers,
            surface_mode=str(surface_mode) if surface_mode is not None else "axis_aligned",
            sample_sets=sample_sets,
        )


__all__ = [
    "Grid",
    "GridSample",
    "GridSpec",
    "axis_basis",
    "cell_center_coordinates",
]
