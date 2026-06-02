from __future__ import annotations

import numpy as np
import drjit as dr


def to_numpy(drjit_array):
    """Convert a DrJit or torch-backed array to numpy."""
    if hasattr(drjit_array, "numpy"):
        return np.array(drjit_array)
    if hasattr(drjit_array, "torch"):
        return drjit_array.torch().cpu().numpy()
    return np.asarray(drjit_array)


def to_numpy_2d(drjit_array, grid_size):
    """Convert a DrJit array to a square numpy grid."""
    return np.array(drjit_array).reshape(grid_size, grid_size)


def to_numpy_complex_2d(drjit_complex, grid_size):
    """Convert a DrJit Complex2f array to a square complex numpy grid."""
    real = np.array(drjit_complex.real).reshape(grid_size, grid_size)
    imag = np.array(drjit_complex.imag).reshape(grid_size, grid_size)
    return real + 1j * imag


def scalar(v):
    """Extract a Python float from a scalar or DrJit array."""
    if hasattr(v, "__len__") and dr.width(v) > 0:
        return float(np.array(v).flat[0])
    return float(v)


def edge_xy(edge):
    """Extract 2D coordinates from an Edge2D."""
    return scalar(edge.p0.x), scalar(edge.p0.y), scalar(edge.p1.x), scalar(edge.p1.y)


def corner_xy(corner):
    """Extract 2D coordinates from a Corner2D."""
    return scalar(corner.position.x), scalar(corner.position.y)


__all__ = [
    "corner_xy",
    "edge_xy",
    "scalar",
    "to_numpy",
    "to_numpy_2d",
    "to_numpy_complex_2d",
]
