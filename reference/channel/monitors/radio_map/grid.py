from __future__ import annotations

from dataclasses import dataclass
import math

import drjit as dr
import witwin as wt

from .monitor import RadioMapMonitor
from ...utils.plane_axes import normalize_axis, point_on_axis_aligned_plane


def _rotation_basis(orientation: tuple[float, float, float]):
    alpha, beta, gamma = (float(value) for value in orientation)
    sin_a, cos_a = math.sin(alpha), math.cos(alpha)
    sin_b, cos_b = math.sin(beta), math.cos(beta)
    sin_c, cos_c = math.sin(gamma), math.cos(gamma)

    r_11 = cos_a * cos_b
    r_12 = cos_a * sin_b * sin_c - sin_a * cos_c
    r_13 = cos_a * sin_b * cos_c + sin_a * sin_c

    r_21 = sin_a * cos_b
    r_22 = sin_a * sin_b * sin_c + cos_a * cos_c
    r_23 = sin_a * sin_b * cos_c - cos_a * sin_c

    r_31 = -sin_b
    r_32 = cos_b * sin_c
    r_33 = cos_b * cos_c

    basis_u = (r_11, r_21, r_31)
    basis_v = (r_12, r_22, r_32)
    normal = (r_13, r_23, r_33)
    return basis_u, basis_v, normal


def _quadrature_offsets(
    *,
    quadrature_mode: str,
    samples_per_cell: int,
    cell_size: tuple[float, float],
) -> tuple[tuple[tuple[float, float], float], ...]:
    cell_size_x, cell_size_y = (float(value) for value in cell_size)
    if quadrature_mode == "center":
        return (((0.0, 0.0), 1.0),)

    subdivisions = int(math.ceil(math.sqrt(samples_per_cell)))
    offsets = []
    weight = 1.0 / float(samples_per_cell)
    for sample_idx in range(samples_per_cell):
        row = sample_idx // subdivisions
        col = sample_idx % subdivisions
        offset_x = ((col + 0.5) / subdivisions - 0.5) * cell_size_x
        offset_y = ((row + 0.5) / subdivisions - 0.5) * cell_size_y
        offsets.append(((offset_x, offset_y), weight))
    return tuple(offsets)


def _cell_center_coordinates(
    *,
    span: tuple[float, float],
    grid_shape: tuple[int, int],
) -> tuple[object, object, object, object]:
    nx, ny = (int(value) for value in grid_shape)
    span_x, span_y = (float(value) for value in span)
    cell_size_x = span_x / float(nx)
    cell_size_y = span_y / float(ny)
    x_coords = (dr.arange(wt.Float, nx) + 0.5) * cell_size_x - 0.5 * span_x
    y_coords = (dr.arange(wt.Float, ny) + 0.5) * cell_size_y - 0.5 * span_y
    grid_x = dr.tile(x_coords, ny)
    grid_y = dr.repeat(y_coords, nx)
    dr.eval(x_coords, y_coords, grid_x, grid_y)
    return x_coords, y_coords, grid_x, grid_y


def _oriented_positions(
    *,
    center: tuple[float, float, float],
    basis_u: tuple[float, float, float],
    basis_v: tuple[float, float, float],
    local_x,
    local_y,
):
    return wt.Point3f(
        wt.Float(center[0]) + basis_u[0] * local_x + basis_v[0] * local_y,
        wt.Float(center[1]) + basis_u[1] * local_x + basis_v[1] * local_y,
        wt.Float(center[2]) + basis_u[2] * local_x + basis_v[2] * local_y,
    )


@dataclass(frozen=True)
class RadioMapSampleSet:
    index: int
    offset_local: tuple[float, float]
    weight: float
    positions: object


@dataclass(frozen=True)
class RadioMapGrid:
    surface_mode: str
    axis: str | None
    position: float | None
    bounds: tuple[tuple[float, float], tuple[float, float]] | None
    center: tuple[float, float, float]
    orientation: tuple[float, float, float]
    size: tuple[float, float]
    tangential_axes: tuple[str, str]
    basis_u: tuple[float, float, float]
    basis_v: tuple[float, float, float]
    normal: tuple[float, float, float]
    grid_shape: tuple[int, int]
    tensor_shape: tuple[int, int]
    cell_size: tuple[float, float]
    x_coords: object
    y_coords: object
    grid_x: object
    grid_y: object
    cell_centers: object
    sample_sets: tuple[RadioMapSampleSet, ...]

    @property
    def n_cells(self) -> int:
        return int(self.grid_shape[0] * self.grid_shape[1])

    def surface_descriptor(self) -> dict[str, object]:
        payload = {
            "surface_mode": self.surface_mode,
            "center": tuple(float(value) for value in self.center),
            "orientation": tuple(float(value) for value in self.orientation),
            "size": tuple(float(value) for value in self.size),
            "basis_u": tuple(float(value) for value in self.basis_u),
            "basis_v": tuple(float(value) for value in self.basis_v),
            "normal": tuple(float(value) for value in self.normal),
            "tangential_axes": tuple(self.tangential_axes),
        }
        if self.surface_mode == "axis_aligned":
            payload.update(
                axis=str(self.axis),
                position=float(self.position),
                bounds=tuple(tuple(float(value) for value in pair) for pair in self.bounds),
            )
        return payload

    @classmethod
    def from_monitor(
        cls,
        monitor: RadioMapMonitor,
        *,
        default_cell_size: float | tuple[float, float] | None,
    ) -> "RadioMapGrid":
        grid_shape = monitor.resolve_grid_shape(default_cell_size=default_cell_size)
        cell_size = monitor.resolve_cell_size(default_cell_size=default_cell_size)
        if monitor.surface_mode == "axis_aligned":
            span = monitor.spans
            x_coords_base, y_coords_base, grid_x_base, grid_y_base = _cell_center_coordinates(
                span=span,
                grid_shape=grid_shape,
            )
            bounds = monitor.bounds
            center = {
                "x": (monitor.position, 0.5 * (bounds[0][0] + bounds[0][1]), 0.5 * (bounds[1][0] + bounds[1][1])),
                "y": (0.5 * (bounds[0][0] + bounds[0][1]), monitor.position, 0.5 * (bounds[1][0] + bounds[1][1])),
                "z": (0.5 * (bounds[0][0] + bounds[0][1]), 0.5 * (bounds[1][0] + bounds[1][1]), monitor.position),
            }[monitor.axis]
            orientation = (0.0, 0.0, 0.0)
            if monitor.axis == "x":
                basis_u = (0.0, 1.0, 0.0)
                basis_v = (0.0, 0.0, 1.0)
                normal = (1.0, 0.0, 0.0)
            elif monitor.axis == "y":
                basis_u = (1.0, 0.0, 0.0)
                basis_v = (0.0, 0.0, 1.0)
                normal = (0.0, 1.0, 0.0)
            else:
                basis_u = (1.0, 0.0, 0.0)
                basis_v = (0.0, 1.0, 0.0)
                normal = (0.0, 0.0, 1.0)
            x_origin = 0.5 * (bounds[0][0] + bounds[0][1])
            y_origin = 0.5 * (bounds[1][0] + bounds[1][1])
            x_coords = x_coords_base + wt.Float(x_origin)
            y_coords = y_coords_base + wt.Float(y_origin)
            grid_x = grid_x_base + wt.Float(x_origin)
            grid_y = grid_y_base + wt.Float(y_origin)
            cell_centers = _oriented_positions(
                center=center,
                basis_u=basis_u,
                basis_v=basis_v,
                local_x=grid_x_base,
                local_y=grid_y_base,
            )
        else:
            center = monitor.center
            orientation = monitor.orientation
            basis_u, basis_v, normal = _rotation_basis(orientation)
            x_coords, y_coords, grid_x, grid_y = _cell_center_coordinates(
                span=monitor.size,
                grid_shape=grid_shape,
            )
            bounds = None
            cell_centers = _oriented_positions(
                center=center,
                basis_u=basis_u,
                basis_v=basis_v,
                local_x=grid_x,
                local_y=grid_y,
            )

        sample_sets = []
        for sample_index, (offset_local, weight) in enumerate(
            _quadrature_offsets(
                quadrature_mode=monitor.quadrature_mode,
                samples_per_cell=monitor.samples_per_cell,
                cell_size=cell_size,
            )
        ):
            offset_x, offset_y = offset_local
            if monitor.surface_mode == "axis_aligned":
                sample_positions = _oriented_positions(
                    center=center,
                    basis_u=basis_u,
                    basis_v=basis_v,
                    local_x=grid_x_base + wt.Float(offset_x),
                    local_y=grid_y_base + wt.Float(offset_y),
                )
            else:
                sample_positions = _oriented_positions(
                    center=center,
                    basis_u=basis_u,
                    basis_v=basis_v,
                    local_x=grid_x + wt.Float(offset_x),
                    local_y=grid_y + wt.Float(offset_y),
                )
            sample_sets.append(
                RadioMapSampleSet(
                    index=sample_index,
                    offset_local=(float(offset_x), float(offset_y)),
                    weight=float(weight),
                    positions=sample_positions,
                )
            )

        return cls(
            surface_mode=monitor.surface_mode,
            axis=monitor.axis,
            position=monitor.position,
            bounds=bounds,
            center=tuple(float(value) for value in center),
            orientation=tuple(float(value) for value in orientation),
            size=tuple(float(value) for value in monitor.spans),
            tangential_axes=monitor.tangential_axes,
            basis_u=tuple(float(value) for value in basis_u),
            basis_v=tuple(float(value) for value in basis_v),
            normal=tuple(float(value) for value in normal),
            grid_shape=tuple(int(value) for value in grid_shape),
            tensor_shape=(int(grid_shape[1]), int(grid_shape[0])),
            cell_size=tuple(float(value) for value in cell_size),
            x_coords=x_coords,
            y_coords=y_coords,
            grid_x=grid_x,
            grid_y=grid_y,
            cell_centers=cell_centers,
            sample_sets=tuple(sample_sets),
        )


@dataclass(frozen=True)
class AxisAlignedRadioMapNativeGrid:
    """Field-like cell-centered grid adapter for native radio-map kernels."""

    bounds: tuple[tuple[float, float], tuple[float, float]]
    size: tuple[int, int]
    axis: str
    position: float
    tangential_axes: tuple[str, str]
    cell_size: tuple[float, float]
    x_coords: object
    y_coords: object
    X: object
    Y: object
    n_cells: int
    sample_index: int
    sample_offset_local: tuple[float, float]

    def pos_to_idx(self, coord_0: wt.Float, coord_1: wt.Float) -> wt.UInt32:
        (x_min, _), (y_min, _) = self.bounds
        nx, ny = self.size
        ix = dr.clip(wt.Int32((coord_0 - x_min) / self.cell_size[0]), 0, nx - 1)
        iy = dr.clip(wt.Int32((coord_1 - y_min) / self.cell_size[1]), 0, ny - 1)
        return wt.UInt32(iy * nx + ix)

    def get_coordinates(self) -> dict[str, object]:
        return {
            "axis_x": self.tangential_axes[0],
            "axis_y": self.tangential_axes[1],
            "axis": self.axis,
            "position": self.position,
            "tangential_axes": self.tangential_axes,
            "x_coords": self.x_coords,
            "y_coords": self.y_coords,
            "X": self.X,
            "Y": self.Y,
        }

    def receiver_positions_3d(self, axis: str | None = None, position: float | None = None):
        resolved_axis = self.axis if axis is None else normalize_axis(axis)
        resolved_position = self.position if position is None else float(position)
        if resolved_axis != self.axis:
            raise ValueError(
                f"AxisAlignedRadioMapNativeGrid only supports axis={self.axis!r}, got {resolved_axis!r}."
            )
        return point_on_axis_aligned_plane(
            axis=resolved_axis,
            position=resolved_position,
            tangential_0=self.X,
            tangential_1=self.Y,
        )

    @property
    def receivers(self):
        return self.receiver_positions_3d()

    @property
    def grid_size(self) -> int:
        return self.size[0]

    @classmethod
    def from_grid(
        cls,
        grid: RadioMapGrid,
        *,
        sample_index: int = 0,
    ) -> "AxisAlignedRadioMapNativeGrid":
        if grid.surface_mode != "axis_aligned":
            raise ValueError("Axis-aligned native grid adapter requires an axis-aligned radio-map grid.")
        sample_set = grid.sample_sets[int(sample_index)]
        offset_x, offset_y = sample_set.offset_local
        x_coords = grid.x_coords + wt.Float(offset_x)
        y_coords = grid.y_coords + wt.Float(offset_y)
        X = grid.grid_x + wt.Float(offset_x)
        Y = grid.grid_y + wt.Float(offset_y)
        dr.eval(x_coords, y_coords, X, Y)
        return cls(
            bounds=tuple(tuple(float(value) for value in pair) for pair in grid.bounds),
            size=tuple(int(value) for value in grid.grid_shape),
            axis=str(grid.axis),
            position=float(grid.position),
            tangential_axes=tuple(str(value) for value in grid.tangential_axes),
            cell_size=tuple(float(value) for value in grid.cell_size),
            x_coords=x_coords,
            y_coords=y_coords,
            X=X,
            Y=Y,
            n_cells=int(grid.n_cells),
            sample_index=int(sample_set.index),
            sample_offset_local=tuple(float(value) for value in sample_set.offset_local),
        )


__all__ = [
    "AxisAlignedRadioMapNativeGrid",
    "RadioMapGrid",
    "RadioMapSampleSet",
]
