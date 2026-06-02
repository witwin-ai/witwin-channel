from __future__ import annotations

from dataclasses import dataclass
from math import ceil

import drjit as dr

import witwin as wt
from witwin.channel._native import _extension, extension_available
from witwin.channel.utils import to_numpy
from witwin.channel.utils.plane_axes import normalize_axis, point_on_axis_aligned_plane


DEFAULT_RECEIVER_TILE_SHAPE = (16, 16)


def _axis_to_int(axis: str) -> int:
    return {"x": 0, "y": 1, "z": 2}[normalize_axis(axis)]


def _normalize_tile_shape(tile_shape) -> tuple[int, int]:
    if tile_shape is None:
        return DEFAULT_RECEIVER_TILE_SHAPE
    if len(tile_shape) != 2:
        raise ValueError("tile_shape must contain exactly two positive integers.")
    tile_0 = int(tile_shape[0])
    tile_1 = int(tile_shape[1])
    if tile_0 <= 0 or tile_1 <= 0:
        raise ValueError("tile_shape entries must be > 0.")
    return tile_0, tile_1


def _resolve_grid_data(grid, grid_data):
    if grid_data is not None:
        return grid_data
    if grid is None:
        raise ValueError("grid_data is required when grid is None.")
    return grid.get_coordinates()


def _resolve_plane_position(grid, grid_data, plane_position):
    if plane_position is not None:
        return float(plane_position)
    if grid is not None:
        return float(grid.position)
    if grid_data is not None and "position" in grid_data:
        return float(grid_data["position"])
    raise ValueError("plane_position is required when it cannot be inferred from grid metadata.")


def _resolve_axis(grid, grid_data):
    if grid is not None:
        return normalize_axis(grid.axis)
    if grid_data is not None and "axis" in grid_data:
        return normalize_axis(grid_data["axis"])
    raise ValueError("Receiver-tile construction requires a monitor-plane axis.")


def _resolve_bounds(grid, x_coords, y_coords):
    if grid is not None:
        return tuple(tuple(float(value) for value in pair) for pair in grid.bounds)
    x_np = to_numpy(x_coords)
    y_np = to_numpy(y_coords)
    return (
        (float(x_np[0]), float(x_np[-1])),
        (float(y_np[0]), float(y_np[-1])),
    )


def _resolve_cell_size(grid, bounds, size):
    if grid is not None:
        return tuple(float(value) for value in grid.cell_size)
    (x_min, x_max), (y_min, y_max) = bounds
    nx, ny = size
    return (
        0.0 if nx <= 0 else float((x_max - x_min) / nx),
        0.0 if ny <= 0 else float((y_max - y_min) / ny),
    )


def _resolve_receiver_positions(*, grid, grid_data, axis: str, plane_position: float, receiver_positions):
    if receiver_positions is not None:
        return receiver_positions
    if grid is not None and hasattr(grid, "receiver_positions_3d"):
        return grid.receiver_positions_3d(position=plane_position)
    if grid_data is not None and "X" in grid_data and "Y" in grid_data:
        return point_on_axis_aligned_plane(
            axis=axis,
            position=plane_position,
            tangential_0=grid_data["X"],
            tangential_1=grid_data["Y"],
        )
    return None


def _native_receiver_tile_builder():
    if not extension_available():
        return None
    ext = _extension()
    if not hasattr(ext, "build_receiver_tiles_arrays"):
        return None
    return ext


def _python_tile_outputs(*, axis: str, plane_position: float, x_coords, y_coords, size, tile_shape):
    n_coord_0, n_coord_1 = size
    tile_size_0, tile_size_1 = tile_shape
    n_tiles_0 = int(ceil(n_coord_0 / tile_size_0)) if n_coord_0 > 0 else 0
    n_tiles_1 = int(ceil(n_coord_1 / tile_size_1)) if n_coord_1 > 0 else 0
    x_np = to_numpy(x_coords)
    y_np = to_numpy(y_coords)
    tile_i0: list[int] = []
    tile_i1: list[int] = []
    tile_extent_0: list[int] = []
    tile_extent_1: list[int] = []
    tile_coord_0_min: list[float] = []
    tile_coord_0_max: list[float] = []
    tile_coord_1_min: list[float] = []
    tile_coord_1_max: list[float] = []
    aabb_min_x: list[float] = []
    aabb_min_y: list[float] = []
    aabb_min_z: list[float] = []
    aabb_max_x: list[float] = []
    aabb_max_y: list[float] = []
    aabb_max_z: list[float] = []

    for tile_y in range(n_tiles_1):
        start_1 = tile_y * tile_size_1
        extent_1 = min(tile_size_1, n_coord_1 - start_1)
        end_1 = start_1 + extent_1 - 1
        coord_1_min = float(y_np[start_1])
        coord_1_max = float(y_np[end_1])
        for tile_x in range(n_tiles_0):
            start_0 = tile_x * tile_size_0
            extent_0 = min(tile_size_0, n_coord_0 - start_0)
            end_0 = start_0 + extent_0 - 1
            coord_0_min = float(x_np[start_0])
            coord_0_max = float(x_np[end_0])
            tile_i0.append(start_0)
            tile_i1.append(start_1)
            tile_extent_0.append(extent_0)
            tile_extent_1.append(extent_1)
            tile_coord_0_min.append(coord_0_min)
            tile_coord_0_max.append(coord_0_max)
            tile_coord_1_min.append(coord_1_min)
            tile_coord_1_max.append(coord_1_max)
            if axis == "x":
                aabb_min_x.append(plane_position)
                aabb_min_y.append(coord_0_min)
                aabb_min_z.append(coord_1_min)
                aabb_max_x.append(plane_position)
                aabb_max_y.append(coord_0_max)
                aabb_max_z.append(coord_1_max)
            elif axis == "y":
                aabb_min_x.append(coord_0_min)
                aabb_min_y.append(plane_position)
                aabb_min_z.append(coord_1_min)
                aabb_max_x.append(coord_0_max)
                aabb_max_y.append(plane_position)
                aabb_max_z.append(coord_1_max)
            else:
                aabb_min_x.append(coord_0_min)
                aabb_min_y.append(coord_1_min)
                aabb_min_z.append(plane_position)
                aabb_max_x.append(coord_0_max)
                aabb_max_y.append(coord_1_max)
                aabb_max_z.append(plane_position)

    outputs = (
        wt.Int32(tile_i0),
        wt.Int32(tile_i1),
        wt.Int32(tile_extent_0),
        wt.Int32(tile_extent_1),
        wt.Float(tile_coord_0_min),
        wt.Float(tile_coord_0_max),
        wt.Float(tile_coord_1_min),
        wt.Float(tile_coord_1_max),
        wt.Float(aabb_min_x),
        wt.Float(aabb_min_y),
        wt.Float(aabb_min_z),
        wt.Float(aabb_max_x),
        wt.Float(aabb_max_y),
        wt.Float(aabb_max_z),
    )
    dr.eval(*outputs)
    return n_tiles_0, n_tiles_1, outputs


@dataclass(frozen=True)
class ReceiverTileDescriptor:
    grid: object | None
    grid_data: dict
    axis: str
    plane_position: float
    bounds: tuple[tuple[float, float], tuple[float, float]]
    cell_size: tuple[float, float]
    size: tuple[int, int]
    tile_shape: tuple[int, int]
    n_tiles: int
    n_tiles_0: int
    n_tiles_1: int
    builder_backend: str
    receiver_positions: object | None
    x_coords: object
    y_coords: object
    tile_i0: object
    tile_i1: object
    tile_extent_0: object
    tile_extent_1: object
    tile_coord_0_min: object
    tile_coord_0_max: object
    tile_coord_1_min: object
    tile_coord_1_max: object
    tile_aabb_min: object
    tile_aabb_max: object


def build_receiver_tiles(
    *,
    grid,
    plane_position=None,
    grid_data=None,
    receiver_positions=None,
    tile_shape=None,
) -> ReceiverTileDescriptor:
    coords = _resolve_grid_data(grid, grid_data)
    axis = _resolve_axis(grid, coords)
    resolved_plane_position = _resolve_plane_position(grid, coords, plane_position)
    resolved_tile_shape = _normalize_tile_shape(tile_shape)
    size = tuple(int(value) for value in (grid.size if grid is not None else (len(coords["x_coords"]), len(coords["y_coords"]))))
    x_coords = coords["x_coords"]
    y_coords = coords["y_coords"]
    native_builder = _native_receiver_tile_builder()

    if native_builder is not None:
        native_outputs = native_builder.build_receiver_tiles_arrays(
            _axis_to_int(axis),
            float(resolved_plane_position),
            x_coords,
            y_coords,
            int(size[0]),
            int(size[1]),
            int(resolved_tile_shape[0]),
            int(resolved_tile_shape[1]),
        )
        n_tiles_0 = int(native_outputs[0])
        n_tiles_1 = int(native_outputs[1])
        outputs = native_outputs[2:]
        builder_backend = "native_cuda"
    else:
        n_tiles_0, n_tiles_1, outputs = _python_tile_outputs(
            axis=axis,
            plane_position=resolved_plane_position,
            x_coords=x_coords,
            y_coords=y_coords,
            size=size,
            tile_shape=resolved_tile_shape,
        )
        builder_backend = "python_fallback"

    (
        tile_i0,
        tile_i1,
        tile_extent_0,
        tile_extent_1,
        tile_coord_0_min,
        tile_coord_0_max,
        tile_coord_1_min,
        tile_coord_1_max,
        tile_aabb_min_x,
        tile_aabb_min_y,
        tile_aabb_min_z,
        tile_aabb_max_x,
        tile_aabb_max_y,
        tile_aabb_max_z,
    ) = outputs

    return ReceiverTileDescriptor(
        grid=grid,
        grid_data=coords,
        axis=axis,
        plane_position=float(resolved_plane_position),
        bounds=_resolve_bounds(grid, x_coords, y_coords),
        cell_size=_resolve_cell_size(grid, _resolve_bounds(grid, x_coords, y_coords), size),
        size=size,
        tile_shape=resolved_tile_shape,
        n_tiles=int(n_tiles_0 * n_tiles_1),
        n_tiles_0=int(n_tiles_0),
        n_tiles_1=int(n_tiles_1),
        builder_backend=builder_backend,
        receiver_positions=_resolve_receiver_positions(
            grid=grid,
            grid_data=coords,
            axis=axis,
            plane_position=resolved_plane_position,
            receiver_positions=receiver_positions,
        ),
        x_coords=x_coords,
        y_coords=y_coords,
        tile_i0=tile_i0,
        tile_i1=tile_i1,
        tile_extent_0=tile_extent_0,
        tile_extent_1=tile_extent_1,
        tile_coord_0_min=tile_coord_0_min,
        tile_coord_0_max=tile_coord_0_max,
        tile_coord_1_min=tile_coord_1_min,
        tile_coord_1_max=tile_coord_1_max,
        tile_aabb_min=wt.Point3f(tile_aabb_min_x, tile_aabb_min_y, tile_aabb_min_z),
        tile_aabb_max=wt.Point3f(tile_aabb_max_x, tile_aabb_max_y, tile_aabb_max_z),
    )


def resolve_receiver_tiles(
    *,
    grid=None,
    plane_position=None,
    grid_data=None,
    receiver_positions=None,
    receiver_tiles: ReceiverTileDescriptor | None = None,
    tile_shape=None,
) -> ReceiverTileDescriptor | None:
    if receiver_tiles is not None:
        return receiver_tiles
    if grid is None:
        return None
    return build_receiver_tiles(
        grid=grid,
        plane_position=plane_position,
        grid_data=grid_data,
        receiver_positions=receiver_positions,
        tile_shape=tile_shape,
    )


def tile_axis_bounds(receiver_tiles):
    n_tiles_0 = int(receiver_tiles.n_tiles_0)
    tile_coord_0_min = to_numpy(receiver_tiles.tile_coord_0_min)
    tile_coord_0_max = to_numpy(receiver_tiles.tile_coord_0_max)
    tile_coord_1_min = to_numpy(receiver_tiles.tile_coord_1_min)
    tile_coord_1_max = to_numpy(receiver_tiles.tile_coord_1_max)
    return (
        tile_coord_0_min[:n_tiles_0],
        tile_coord_0_max[:n_tiles_0],
        tile_coord_1_min[::n_tiles_0],
        tile_coord_1_max[::n_tiles_0],
    )


def tile_cell_bounds(receiver_tiles):
    bound_0_min = float(receiver_tiles.bounds[0][0])
    bound_1_min = float(receiver_tiles.bounds[1][0])
    cell_size_0 = float(receiver_tiles.cell_size[0])
    cell_size_1 = float(receiver_tiles.cell_size[1])
    tile_i0 = to_numpy(receiver_tiles.tile_i0).astype(float, copy=False)
    tile_i1 = to_numpy(receiver_tiles.tile_i1).astype(float, copy=False)
    tile_extent_0 = to_numpy(receiver_tiles.tile_extent_0).astype(float, copy=False)
    tile_extent_1 = to_numpy(receiver_tiles.tile_extent_1).astype(float, copy=False)
    tile_bound_0_min = bound_0_min + tile_i0 * cell_size_0
    tile_bound_0_max = tile_bound_0_min + tile_extent_0 * cell_size_0
    tile_bound_1_min = bound_1_min + tile_i1 * cell_size_1
    tile_bound_1_max = tile_bound_1_min + tile_extent_1 * cell_size_1
    return (
        tile_bound_0_min,
        tile_bound_0_max,
        tile_bound_1_min,
        tile_bound_1_max,
    )


def tile_cell_bounds_arrays(receiver_tiles):
    bound_0_min = float(receiver_tiles.bounds[0][0])
    bound_1_min = float(receiver_tiles.bounds[1][0])
    cell_size_0 = float(receiver_tiles.cell_size[0])
    cell_size_1 = float(receiver_tiles.cell_size[1])
    tile_i0 = wt.Float(receiver_tiles.tile_i0)
    tile_i1 = wt.Float(receiver_tiles.tile_i1)
    tile_extent_0 = wt.Float(receiver_tiles.tile_extent_0)
    tile_extent_1 = wt.Float(receiver_tiles.tile_extent_1)
    tile_bound_0_min = bound_0_min + tile_i0 * cell_size_0
    tile_bound_0_max = tile_bound_0_min + tile_extent_0 * cell_size_0
    tile_bound_1_min = bound_1_min + tile_i1 * cell_size_1
    tile_bound_1_max = tile_bound_1_min + tile_extent_1 * cell_size_1
    dr.eval(
        tile_bound_0_min,
        tile_bound_0_max,
        tile_bound_1_min,
        tile_bound_1_max,
    )
    return (
        tile_bound_0_min,
        tile_bound_0_max,
        tile_bound_1_min,
        tile_bound_1_max,
    )


def tile_receiver_counts(receiver_tiles):
    counts = wt.UInt32(receiver_tiles.tile_extent_0) * wt.UInt32(receiver_tiles.tile_extent_1)
    dr.eval(counts)
    return counts


def tile_receiver_indices(receiver_tiles, tile_idx):
    if not isinstance(tile_idx, wt.UInt32):
        tile_idx = wt.UInt32(tile_idx)
    count = dr.width(tile_idx)
    if count == 0:
        return dr.zeros(wt.UInt32, 0)
    nx = int(receiver_tiles.size[0])
    tile_i0 = dr.gather(type(receiver_tiles.tile_i0), receiver_tiles.tile_i0, tile_idx)
    tile_i1 = dr.gather(type(receiver_tiles.tile_i1), receiver_tiles.tile_i1, tile_idx)
    tile_extent_0 = dr.gather(type(receiver_tiles.tile_extent_0), receiver_tiles.tile_extent_0, tile_idx)
    tile_extent_1 = dr.gather(type(receiver_tiles.tile_extent_1), receiver_tiles.tile_extent_1, tile_idx)
    local_count = tile_extent_0 * tile_extent_1
    if count == 1:
        n_local = int(local_count[0])
        if n_local <= 0:
            return dr.zeros(wt.UInt32, 0)
        local_slot = dr.arange(wt.UInt32, n_local)
        tile_idx = dr.full(wt.UInt32, tile_idx[0], n_local)
        tile_i0 = dr.full(wt.Int32, tile_i0[0], n_local)
        tile_i1 = dr.full(wt.Int32, tile_i1[0], n_local)
        tile_extent_0 = dr.full(wt.Int32, tile_extent_0[0], n_local)
    else:
        raise ValueError("tile_receiver_indices expects a single tile index.")
    row = wt.Int32(local_slot) // tile_extent_0
    col = wt.Int32(local_slot) % tile_extent_0
    receiver_idx = wt.UInt32((tile_i1 + row) * int(nx) + (tile_i0 + col))
    dr.eval(receiver_idx)
    return receiver_idx


def receiver_index_for_tile_slot(receiver_tiles, tile_idx, local_rx_slot):
    if not isinstance(tile_idx, wt.UInt32):
        tile_idx = wt.UInt32(tile_idx)
    if dr.width(tile_idx) == 1 and dr.width(local_rx_slot) != 1:
        tile_idx = dr.full(wt.UInt32, tile_idx[0], dr.width(local_rx_slot))
    tile_i0 = dr.gather(type(receiver_tiles.tile_i0), receiver_tiles.tile_i0, tile_idx)
    tile_i1 = dr.gather(type(receiver_tiles.tile_i1), receiver_tiles.tile_i1, tile_idx)
    tile_extent_0 = dr.gather(type(receiver_tiles.tile_extent_0), receiver_tiles.tile_extent_0, tile_idx)
    row = wt.Int32(local_rx_slot) // tile_extent_0
    col = wt.Int32(local_rx_slot) % tile_extent_0
    receiver_idx = wt.UInt32((tile_i1 + row) * int(receiver_tiles.size[0]) + (tile_i0 + col))
    dr.eval(receiver_idx)
    return receiver_idx


def receiver_indices_per_tile(receiver_tiles) -> tuple[tuple[int, ...], ...]:
    nx, _ = receiver_tiles.size
    tile_i0 = to_numpy(receiver_tiles.tile_i0)
    tile_i1 = to_numpy(receiver_tiles.tile_i1)
    tile_extent_0 = to_numpy(receiver_tiles.tile_extent_0)
    tile_extent_1 = to_numpy(receiver_tiles.tile_extent_1)

    receiver_idx_per_tile: list[tuple[int, ...]] = []
    for i0, i1, extent_0, extent_1 in zip(tile_i0, tile_i1, tile_extent_0, tile_extent_1):
        tile_receiver_idx: list[int] = []
        for row in range(int(extent_1)):
            row_start = int(i1 + row) * int(nx) + int(i0)
            tile_receiver_idx.extend(range(row_start, row_start + int(extent_0)))
        receiver_idx_per_tile.append(tuple(tile_receiver_idx))
    return tuple(receiver_idx_per_tile)


def compact_tile_tasks(family_idx, tile_idx, active_mask):
    if dr.width(family_idx) == 0:
        return dr.zeros(wt.UInt32, 0), dr.zeros(wt.UInt32, 0)
    if extension_available():
        from witwin.channel.kernels.trace.cartesian_filter.native_impl import compact_index_pairs

        return compact_index_pairs(family_idx, tile_idx, active_mask)
    keep = dr.compress(active_mask)
    return (
        dr.gather(type(family_idx), family_idx, keep),
        dr.gather(type(tile_idx), tile_idx, keep),
    )


def deduplicate_tile_tasks(family_idx, tile_idx, *, n_tiles: int):
    if dr.width(family_idx) == 0:
        return dr.zeros(wt.UInt32, 0), dr.zeros(wt.UInt32, 0)
    if extension_available():
        from witwin.channel.kernels.trace.cartesian_filter.native_impl import deduplicate_cartesian_pairs

        return deduplicate_cartesian_pairs(family_idx, tile_idx, n_tiles)
    pairs = sorted(set(zip(to_numpy(family_idx).tolist(), to_numpy(tile_idx).tolist())))
    return (
        wt.UInt32([int(pair[0]) for pair in pairs]),
        wt.UInt32([int(pair[1]) for pair in pairs]),
    )


__all__ = [
    "DEFAULT_RECEIVER_TILE_SHAPE",
    "ReceiverTileDescriptor",
    "build_receiver_tiles",
    "compact_tile_tasks",
    "deduplicate_tile_tasks",
    "receiver_indices_per_tile",
    "receiver_index_for_tile_slot",
    "resolve_receiver_tiles",
    "tile_axis_bounds",
    "tile_cell_bounds",
    "tile_cell_bounds_arrays",
    "tile_receiver_counts",
    "tile_receiver_indices",
]
