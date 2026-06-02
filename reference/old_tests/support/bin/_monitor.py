"""FieldMonitor assertions shared by grad demos."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from witwin.channel import FieldMonitor, to_numpy


DEFAULT_MONITOR_NAME = "grad_grid"


def _coerce_grid_shape(grid_size) -> tuple[int, int]:
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


def scalar_height(value) -> float:
    if hasattr(value, "item"):
        try:
            return float(value.item())
        except (TypeError, ValueError):
            pass
    try:
        return float(value[0])
    except (TypeError, ValueError, IndexError, KeyError):
        return float(value)


def monitor_height(tx_pos, calculation_height: float | None = None) -> float:
    if calculation_height is not None:
        return float(calculation_height)
    if hasattr(tx_pos, "z"):
        return scalar_height(tx_pos.z)
    if len(tx_pos) != 3:
        raise ValueError("tx_pos must provide xyz coordinates when calculation_height is omitted.")
    return scalar_height(tx_pos[2])


def assert_boundary_point_sampling(
    x_coords,
    y_coords,
    *,
    bounds,
    grid_size,
) -> None:
    grid_shape = _coerce_grid_shape(grid_size)
    x_expected = np.linspace(float(bounds[0][0]), float(bounds[0][1]), grid_shape[0], dtype=np.float64)
    y_expected = np.linspace(float(bounds[1][0]), float(bounds[1][1]), grid_shape[1], dtype=np.float64)
    x_actual = np.asarray(to_numpy(x_coords), dtype=np.float64).reshape(grid_shape[0])
    y_actual = np.asarray(to_numpy(y_coords), dtype=np.float64).reshape(grid_shape[1])

    if not np.allclose(x_actual, x_expected, atol=2e-6, rtol=0.0):
        raise AssertionError("Monitor x-coordinates no longer match legacy boundary-point sampling.")
    if not np.allclose(y_actual, y_expected, atol=2e-6, rtol=0.0):
        raise AssertionError("Monitor y-coordinates no longer match legacy boundary-point sampling.")


def assert_plane_monitor_result(result, monitor: FieldMonitor) -> None:
    if isinstance(result, Mapping):
        payload = result[monitor.name]
    else:
        payload = result
    sampling = payload.metadata["receiver_sampling"]

    if sampling["sample_positions"] != "boundary_points":
        raise AssertionError("Monitor sampling drifted away from legacy boundary-point coordinates.")
    if sampling["index_partitioning"] != "span_over_n_bins":
        raise AssertionError("Monitor index partitioning drifted away from the legacy DDA binning.")
    if sampling["axis"] != monitor.axis:
        raise AssertionError("Monitor axis does not match the requested plane.")
    if monitor.grid_shape is not None and tuple(payload.grid_shape) != tuple(monitor.grid_shape):
        raise AssertionError("Trace grid shape does not match the requested monitor grid.")
    if tuple(payload.range_x) != monitor.bounds[0] or tuple(payload.range_y) != monitor.bounds[1]:
        raise AssertionError("Trace bounds do not match the requested monitor bounds.")
    if abs(float(payload.plane_position) - float(monitor.position)) > 1e-6:
        raise AssertionError("Trace plane position does not match the requested monitor height.")
    if payload.ray_mode != monitor.ray_mode:
        raise AssertionError("Trace ray_mode does not match the requested monitor mode.")

    assert_boundary_point_sampling(
        payload.coords.x,
        payload.coords.y,
        bounds=monitor.bounds,
        grid_size=payload.grid_shape,
    )
