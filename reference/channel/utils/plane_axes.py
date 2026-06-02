from __future__ import annotations

from types import MappingProxyType

import drjit as dr
import witwin as wt


_AXES = ("x", "y", "z")

AXIS_TO_TANGENTIAL_AXES = MappingProxyType({
    "z": ("x", "y"),
    "x": ("y", "z"),
    "y": ("x", "z"),
})

_tangential_axes_to_normal_axis = {}
for axis_name, tangential_axes in AXIS_TO_TANGENTIAL_AXES.items():
    _tangential_axes_to_normal_axis[tangential_axes] = axis_name
    _tangential_axes_to_normal_axis[(tangential_axes[1], tangential_axes[0])] = axis_name

TANGENTIAL_AXES_TO_NORMAL_AXIS = MappingProxyType(_tangential_axes_to_normal_axis)


def normalize_axis(axis: str) -> str:
    axis_name = str(axis).lower()
    if axis_name not in _AXES:
        raise ValueError("axis must be one of 'x', 'y', or 'z'.")
    return axis_name


def tangential_axes_for_axis(axis: str) -> tuple[str, str]:
    return AXIS_TO_TANGENTIAL_AXES[normalize_axis(axis)]


def normal_axis_for_tangential_axes(tangential_axes) -> str:
    if len(tangential_axes) != 2:
        raise ValueError("tangential_axes must contain exactly two axis labels.")
    normalized = tuple(normalize_axis(axis_name) for axis_name in tangential_axes)
    normal_axis = TANGENTIAL_AXES_TO_NORMAL_AXIS.get(normalized)
    if normal_axis is None:
        raise ValueError(
            "tangential_axes must select any two distinct axes from 'x', 'y', and 'z'."
        )
    return normal_axis


def point_on_axis_aligned_plane(*, axis: str, position, tangential_0, tangential_1):
    axis_name = normalize_axis(axis)
    width = max(dr.width(tangential_0), dr.width(tangential_1))
    fixed = dr.full(wt.Float, position, width) if width > 1 else wt.Float(position)
    if axis_name == "x":
        return wt.Point3f(fixed, tangential_0, tangential_1)
    if axis_name == "y":
        return wt.Point3f(tangential_0, fixed, tangential_1)
    return wt.Point3f(tangential_0, tangential_1, fixed)


__all__ = [
    "AXIS_TO_TANGENTIAL_AXES",
    "TANGENTIAL_AXES_TO_NORMAL_AXIS",
    "normalize_axis",
    "normal_axis_for_tangential_axes",
    "point_on_axis_aligned_plane",
    "tangential_axes_for_axis",
]
