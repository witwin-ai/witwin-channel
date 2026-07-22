from __future__ import annotations

from dataclasses import dataclass

import torch

from witwin.channel.scene.models import ReceiverGrid


@dataclass(frozen=True, slots=True)
class AxisAlignedGridSpec:
    grid: ReceiverGrid
    axis: int
    position: float
    coord0_min: float
    coord0_max: float
    coord1_min: float
    coord1_max: float
    resolution0: int
    resolution1: int
    cell_area: float


def vector3_tuple(value: torch.Tensor) -> tuple[float, float, float]:
    return (float(value[0]), float(value[1]), float(value[2]))


def first_receiver_grid(scene: object) -> ReceiverGrid | None:
    for receiver in scene.receivers:
        if isinstance(receiver, ReceiverGrid):
            return receiver
    return None


def component_grid_shape(grid: ReceiverGrid) -> tuple[int, int]:
    return (grid.shape[1], grid.shape[0])


def _axis_index(
    values: tuple[float, float, float], *, name: str
) -> tuple[int, float]:
    nonzero = [idx for idx, value in enumerate(values) if abs(value) > 1.0e-6]
    if len(nonzero) != 1:
        raise ValueError(f"{name} must be axis-aligned")
    index = nonzero[0]
    value = values[index]
    sign = 1.0 if value > 0.0 else -1.0
    if abs(abs(value) - 1.0) > 1.0e-5:
        raise ValueError(f"{name} must be a unit axis vector")
    return index, sign


def axis_aligned_grid_spec(grid: ReceiverGrid) -> AxisAlignedGridSpec:
    rows, cols = grid.shape
    origin = vector3_tuple(grid.origin)
    axis0, sign0 = _axis_index(
        vector3_tuple(grid.x_axis), name="ReceiverGrid.x_axis"
    )
    axis1, sign1 = _axis_index(
        vector3_tuple(grid.y_axis), name="ReceiverGrid.y_axis"
    )
    if axis0 == axis1:
        raise ValueError("ReceiverGrid axes must be orthogonal")
    axis = ({0, 1, 2} - {axis0, axis1}).pop()
    expected = (1, 2) if axis == 0 else (0, 2) if axis == 1 else (0, 1)
    if (axis0, axis1) != expected:
        raise ValueError("ReceiverGrid axes must match RayD grid coordinate order")

    step0 = float(grid.spacing[0]) * sign0
    step1 = float(grid.spacing[1]) * sign1
    first0 = origin[axis0]
    first1 = origin[axis1]
    last0 = first0 + step0 * float(rows - 1)
    last1 = first1 + step1 * float(cols - 1)
    half0 = abs(float(grid.spacing[0])) * 0.5
    half1 = abs(float(grid.spacing[1])) * 0.5
    coord0_min = min(first0, last0) - half0
    coord0_max = max(first0, last0) + half0
    coord1_min = min(first1, last1) - half1
    coord1_max = max(first1, last1) + half1
    return AxisAlignedGridSpec(
        grid=grid,
        axis=axis,
        position=origin[axis],
        coord0_min=coord0_min,
        coord0_max=coord0_max,
        coord1_min=coord1_min,
        coord1_max=coord1_max,
        resolution0=rows,
        resolution1=cols,
        cell_area=abs((coord0_max - coord0_min) * (coord1_max - coord1_min))
        / float(rows * cols),
    )
