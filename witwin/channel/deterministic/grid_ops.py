"""Deterministic receiver-grid operations: quadrature samples + native adapter.

The shared :class:`~witwin.channel.core.grid.Grid` /
:class:`~witwin.channel.core.grid.GridSpec` types live in
:mod:`witwin.channel.core.grid`. This module attaches deterministic
quadrature samples to that shared grid and provides the per-sample
:class:`NativeGrid` adapter required by the bundled native kernels.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import drjit as dr
from witwin.channel.deterministic import types as wt

from .config import SolveSpec
from witwin.channel.core.geometry import (
    normalize_axis,
    point_on_axis_aligned_plane,
    tangential_axes_for_axis,
)
from witwin.channel.core.grid import Grid


def _palindromic_row_counts(samples_per_cell: int) -> tuple[int, ...]:
    """Split N samples into symmetric (palindromic) per-row counts.

    Symmetry in both axes keeps the sample mean exactly at the cell center,
    so the midpoint quadrature stays unbiased for any N (not just squares).
    """
    n = int(samples_per_cell)
    target_rows = math.sqrt(n)
    candidates = sorted(range(1, n + 1), key=lambda rows: abs(rows - target_rows))
    for n_rows in candidates:
        base, rem = divmod(n, n_rows)
        if rem % 2 == 1 and n_rows % 2 == 0:
            continue
        counts = [base] * n_rows
        if rem % 2 == 1:
            counts[n_rows // 2] += 1
            rem -= 1
        for offset in range(rem // 2):
            counts[offset] += 1
            counts[n_rows - 1 - offset] += 1
        return tuple(counts)
    return (n,)


def _quadrature_offsets(
    *,
    quadrature_mode: str,
    samples_per_cell: int,
    cell_size: tuple[float, float],
) -> tuple[tuple[tuple[float, float], float], ...]:
    """Return local sample offsets and weights for one receiver cell.

    Samples sit at the centers of a symmetric row partition of the cell and
    carry per-stratum area weights (1 / (n_rows * row_count)), which is the
    exact midpoint rule for the partition and sums to 1.
    """
    cell_size_x, cell_size_y = float(cell_size[0]), float(cell_size[1])
    if quadrature_mode == "center":
        return (((0.0, 0.0), 1.0),)

    row_counts = _palindromic_row_counts(samples_per_cell)
    n_rows = len(row_counts)
    offsets: list[tuple[tuple[float, float], float]] = []
    for row, row_count in enumerate(row_counts):
        offset_y = ((row + 0.5) / n_rows - 0.5) * cell_size_y
        weight = 1.0 / float(n_rows * row_count)
        for col in range(row_count):
            offset_x = ((col + 0.5) / row_count - 0.5) * cell_size_x
            offsets.append(((offset_x, offset_y), weight))
    return tuple(offsets)


def build_grid(
    spec: SolveSpec,
    *,
    default_cell_size: float | tuple[float, float] | None,
) -> Grid:
    """Build a deterministic Grid with quadrature samples derived from ``spec``."""
    cell_size = spec.resolve_cell_size(default_cell_size=default_cell_size)
    sample_offsets = _quadrature_offsets(
        quadrature_mode=spec.quadrature_mode,
        samples_per_cell=spec.samples_per_cell,
        cell_size=cell_size,
    )
    return Grid.from_spec(
        spec,
        default_cell_size=default_cell_size,
        sample_offsets=sample_offsets,
        surface_mode=str(spec.surface_mode),
    )


@dataclass(frozen=True)
class NativeGrid:
    """Adapt one grid sample into the axis-aligned native kernel contract."""

    bounds: tuple[tuple[float, float], tuple[float, float]]
    size: tuple[int, int]
    axis: str
    position: float
    cell_size: tuple[float, float]
    x_coords: object
    y_coords: object
    X: object
    Y: object

    @property
    def tangential_axes(self) -> tuple[str, str]:
        return tangential_axes_for_axis(self.axis)

    @property
    def n_cells(self) -> int:
        return int(self.size[0] * self.size[1])

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
            raise ValueError(f"NativeGrid only supports axis={self.axis!r}, got {resolved_axis!r}.")
        return point_on_axis_aligned_plane(
            axis=resolved_axis,
            position=resolved_position,
            tangential_0=self.X,
            tangential_1=self.Y,
        )

    @classmethod
    def from_grid(
        cls,
        grid: Grid,
        *,
        sample_index: int = 0,
    ) -> "NativeGrid":
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
            cell_size=tuple(float(value) for value in grid.cell_size),
            x_coords=x_coords,
            y_coords=y_coords,
            X=X,
            Y=Y,
        )


__all__ = [
    "NativeGrid",
    "build_grid",
]
