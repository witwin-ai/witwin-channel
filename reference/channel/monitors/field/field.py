from __future__ import annotations

"""Field class for receiver field management."""

import math
import drjit as dr
import witwin as wt

from ...utils.constants import SPEED_OF_LIGHT
from ...utils.plane_axes import (
    normal_axis_for_tangential_axes,
    normalize_axis,
    point_on_axis_aligned_plane,
    tangential_axes_for_axis,
)


def _normalize_tangential_axes(tangential_axes) -> tuple[str, str]:
    if tangential_axes is None:
        return ("x", "y")
    if len(tangential_axes) != 2:
        raise ValueError("tangential_axes must contain exactly two axis labels.")
    normalized = tuple(normalize_axis(axis_name) for axis_name in tangential_axes)
    if normalized[0] == normalized[1]:
        raise ValueError("tangential_axes must use two distinct axes.")
    normal_axis_for_tangential_axes(normalized)
    return normalized


class Field:
    """Axis-aligned 2D receiver sampling grid with legacy boundary-point coordinates.

    Historical note:
        The public/legacy channel grid uses ``linspace``-style sample points on
        the monitor bounds, not cell-centered samples. This matches the original
        ``basic`` scripts and must remain stable because shifting the sampling
        lattice changes phase.

        Reflection/DDA accumulation also preserves the historical index mapping
        based on ``span / n_samples`` bins via :attr:`cell_size`, even though
        the coordinate lattice itself is boundary-aligned. That combination is
        intentionally frozen for compatibility with existing field outputs.
    """

    # Class-level cache for grid coordinates (shared across instances)
    _coord_cache: dict = {}

    def __init__(
        self,
        bounds: tuple,
        size: tuple,
        axis: str = "z",
        position: float = 0.0,
        tangential_axes=None,
    ):
        """
        Initialize a 2D receiver field on an axis-aligned plane.

        Args:
            bounds: ((axis0_min, axis0_max), (axis1_min, axis1_max)) field boundaries
            size: (n_axis0, n_axis1) number of cells in each tangential dimension
            axis: Plane normal axis, one of ``"x"``, ``"y"``, or ``"z"``.
            position: Fixed coordinate value on the normal axis.
            tangential_axes: Optional explicit ordered pair of tangential world
                axes. When omitted, it is derived from ``axis``.
        """
        self.bounds = bounds
        self.size = size
        self.n_cells = size[0] * size[1]
        self.axis = normalize_axis(axis)
        self.position = float(position)
        resolved_tangential_axes = (
            tangential_axes_for_axis(self.axis)
            if tangential_axes is None
            else _normalize_tangential_axes(tangential_axes)
        )
        resolved_normal_axis = normal_axis_for_tangential_axes(resolved_tangential_axes)
        if resolved_normal_axis != self.axis:
            raise ValueError(
                f"tangential_axes {resolved_tangential_axes!r} are incompatible with axis={self.axis!r}."
            )
        self.tangential_axes = resolved_tangential_axes
        self.normal_axis = self.axis

        (x_min, x_max), (y_min, y_max) = bounds
        self.cell_size = ((x_max - x_min) / size[0], (y_max - y_min) / size[1])

    @classmethod
    def from_wavelength(
        cls,
        bounds: tuple,
        wavelength: float,
        resolution: float = 0.125,
        axis: str = "z",
        position: float = 0.0,
        tangential_axes=None,
    ) -> 'Field':
        """
        Create a Field with automatic size based on wavelength.

        Args:
            bounds: ((axis0_min, axis0_max), (axis1_min, axis1_max)) field boundaries
            wavelength: Signal wavelength in meters
            resolution: Cell size as fraction of wavelength (default 0.125 = lambda/8)
            axis: Plane normal axis
            position: Fixed coordinate value on the normal axis
            tangential_axes: Optional explicit ordered pair of tangential world axes

        Returns:
            Field instance with calculated size
        """
        (x_min, x_max), (y_min, y_max) = bounds
        cell_size = resolution * wavelength

        nx = int(math.ceil((x_max - x_min) / cell_size))
        ny = int(math.ceil((y_max - y_min) / cell_size))
        grid_size = max(nx, ny)

        return cls(
            bounds=bounds,
            size=(grid_size, grid_size),
            axis=axis,
            position=position,
            tangential_axes=tangential_axes,
        )

    @classmethod
    def from_frequency(
        cls,
        bounds: tuple,
        frequency: float,
        resolution: float = 0.125,
        axis: str = "z",
        position: float = 0.0,
        tangential_axes=None,
    ) -> 'Field':
        """
        Create a Field with automatic size based on frequency.

        Args:
            bounds: ((axis0_min, axis0_max), (axis1_min, axis1_max)) field boundaries
            frequency: Signal frequency in Hz
            resolution: Cell size as fraction of wavelength (default 0.125 = lambda/8)
            axis: Plane normal axis
            position: Fixed coordinate value on the normal axis
            tangential_axes: Optional explicit ordered pair of tangential world axes

        Returns:
            Field instance with calculated size
        """
        wavelength = SPEED_OF_LIGHT / frequency
        return cls.from_wavelength(
            bounds,
            wavelength,
            resolution,
            axis=axis,
            position=position,
            tangential_axes=tangential_axes,
        )

    def pos_to_idx(self, coord_0: wt.Float, coord_1: wt.Float) -> wt.UInt32:
        """
        Convert tangential positions to flattened field cell indices.

        Args:
            coord_0, coord_1: DrJit Float arrays on the ordered tangential axes

        Returns:
            DrJit UInt32 array of cell indices
        """
        (x_min, _), (y_min, _) = self.bounds
        nx, ny = self.size
        ix = dr.clip(wt.Int32((coord_0 - x_min) / self.cell_size[0]), 0, nx - 1)
        iy = dr.clip(wt.Int32((coord_1 - y_min) / self.cell_size[1]), 0, ny - 1)
        return wt.UInt32(iy * nx + ix)

    def get_coordinates(self) -> dict:
        """
        Get cached boundary-aligned tangential grid coordinates.

        Returns:
            dict with flattened/tiled tangential coordinates and their axis labels
        """
        cache_key = (self.size, self.bounds, self.axis, self.position, self.tangential_axes)
        if cache_key in Field._coord_cache:
            return Field._coord_cache[cache_key]

        (x_min, x_max), (y_min, y_max) = self.bounds
        nx, ny = self.size

        x_step = (x_max - x_min) / (nx - 1) if nx > 1 else 0
        y_step = (y_max - y_min) / (ny - 1) if ny > 1 else 0

        idx_x = dr.arange(wt.Float, nx)
        idx_y = dr.arange(wt.Float, ny)
        x_coords = wt.Float(x_min) + idx_x * wt.Float(x_step)
        y_coords = wt.Float(y_min) + idx_y * wt.Float(y_step)

        X = dr.tile(x_coords, ny)
        Y = dr.repeat(y_coords, nx)

        dr.eval(x_coords, y_coords, X, Y)

        coord_data = {
            'axis_x': self.tangential_axes[0],
            'axis_y': self.tangential_axes[1],
            'axis': self.axis,
            'position': self.position,
            'tangential_axes': self.tangential_axes,
            'x_coords': x_coords,
            'y_coords': y_coords,
            'X': X,
            'Y': Y,
        }

        Field._coord_cache[cache_key] = coord_data
        return coord_data

    def receiver_positions_3d(self, axis: str | None = None, position: float | None = None):
        resolved_axis = self.axis if axis is None else normalize_axis(axis)
        resolved_position = self.position if position is None else float(position)
        if resolved_axis != self.normal_axis:
            raise ValueError(
                f"Field tangential axes {self.tangential_axes!r} are incompatible with axis={resolved_axis!r}."
            )
        coords = self.get_coordinates()
        return point_on_axis_aligned_plane(
            axis=resolved_axis,
            position=resolved_position,
            tangential_0=coords['X'],
            tangential_1=coords['Y'],
        )

    @property
    def receivers(self):
        """Canonical 3D receiver positions on this plane."""
        return self.receiver_positions_3d()

    @property
    def X(self) -> wt.Float:
        """Flattened coordinates on the first tangential axis."""
        return self.get_coordinates()['X']

    @property
    def Y(self) -> wt.Float:
        """Flattened coordinates on the second tangential axis."""
        return self.get_coordinates()['Y']

    @property
    def axis_x(self) -> str:
        return self.tangential_axes[0]

    @property
    def axis_y(self) -> str:
        return self.tangential_axes[1]

    @property
    def grid_size(self) -> int:
        """Legacy square-grid size accessor."""
        return self.size[0]

