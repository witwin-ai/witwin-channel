from __future__ import annotations

from dataclasses import dataclass

import drjit as dr

import witwin as wt
from witwin.channel._native import _extension, extension_available
from witwin.channel.kernels.monitors.common.receiver_tiles.native_impl import compact_tile_tasks, tile_receiver_counts
from witwin.channel.utils.drjit_ops import Concat


_PLANNER_EPS = 1.0e-8
_SEGMENT_TILE_BLOCK_SIZE = 1024


@dataclass(frozen=True)
class SuffixSegmentTilePlan:
    planner_backend: str
    n_segments: int
    n_tiles: int
    tile_task_count: int
    estimated_cell_count: int
    max_segments_per_tile: int
    support_coord_0_min: object
    support_coord_0_max: object
    support_coord_1_min: object
    support_coord_1_max: object
    segment_tile_counts: object
    tile_task_segment_idx: object
    tile_task_tile_idx: object
    tile_task_entry_coord_0: object
    tile_task_entry_coord_1: object
    tile_task_entry_t: object
    tile_task_entry_t_max_0: object
    tile_task_entry_t_max_1: object
    tile_task_counts_per_tile: object


@dataclass(frozen=True)
class SuffixTilePacketPlan:
    packetizer_backend: str
    n_tiles: int
    packet_count: int
    max_packets_per_tile: int
    max_segments_per_packet: int
    packet_counts_per_tile: object


def _tangential_axes(receiver_tiles) -> tuple[str, str]:
    grid_data = getattr(receiver_tiles, "grid_data", None) or {}
    tangential_axes = grid_data.get("tangential_axes")
    if tangential_axes is not None:
        return tuple(str(axis_name) for axis_name in tangential_axes)
    axis_x = grid_data.get("axis_x")
    axis_y = grid_data.get("axis_y")
    if axis_x is not None and axis_y is not None:
        return str(axis_x), str(axis_y)
    if receiver_tiles.axis == "x":
        return "y", "z"
    if receiver_tiles.axis == "y":
        return "x", "z"
    return "x", "y"


def _detach_component(point, axis: str):
    return dr.detach(getattr(point, axis))


def _empty_plan(n_segments: int, n_tiles: int) -> SuffixSegmentTilePlan:
    return SuffixSegmentTilePlan(
        planner_backend="tile_interval_aabb_drjit",
        n_segments=max(0, int(n_segments)),
        n_tiles=max(0, int(n_tiles)),
        tile_task_count=0,
        estimated_cell_count=0,
        max_segments_per_tile=0,
        support_coord_0_min=dr.zeros(wt.Float, max(0, int(n_segments))),
        support_coord_0_max=dr.zeros(wt.Float, max(0, int(n_segments))),
        support_coord_1_min=dr.zeros(wt.Float, max(0, int(n_segments))),
        support_coord_1_max=dr.zeros(wt.Float, max(0, int(n_segments))),
        segment_tile_counts=dr.zeros(wt.UInt32, max(0, int(n_segments))),
        tile_task_segment_idx=dr.zeros(wt.UInt32, 0),
        tile_task_tile_idx=dr.zeros(wt.UInt32, 0),
        tile_task_entry_coord_0=dr.zeros(wt.Float, 0),
        tile_task_entry_coord_1=dr.zeros(wt.Float, 0),
        tile_task_entry_t=dr.zeros(wt.Float, 0),
        tile_task_entry_t_max_0=dr.zeros(wt.Float, 0),
        tile_task_entry_t_max_1=dr.zeros(wt.Float, 0),
        tile_task_counts_per_tile=dr.zeros(wt.UInt32, max(0, int(n_tiles))),
    )


def _native_suffix_segment_planner():
    if not extension_available():
        return None
    ext = _extension()
    if not hasattr(ext, "build_suffix_segment_tile_plan_raw"):
        return None
    return ext


def _finalize_plan(
    *,
    planner_backend: str,
    n_segments: int,
    n_tiles: int,
    receiver_tiles,
    support_coord_0_min,
    support_coord_0_max,
    support_coord_1_min,
    support_coord_1_max,
    segment_tile_counts,
    tile_task_segment_idx,
    tile_task_tile_idx,
    tile_task_entry_coord_0,
    tile_task_entry_coord_1,
    tile_task_entry_t,
    tile_task_entry_t_max_0,
    tile_task_entry_t_max_1,
) -> SuffixSegmentTilePlan:
    tile_task_count = dr.width(tile_task_segment_idx)
    if tile_task_count == 0:
        empty = _empty_plan(n_segments, n_tiles)
        return SuffixSegmentTilePlan(
            planner_backend=planner_backend,
            n_segments=empty.n_segments,
            n_tiles=empty.n_tiles,
            tile_task_count=empty.tile_task_count,
            estimated_cell_count=empty.estimated_cell_count,
            max_segments_per_tile=empty.max_segments_per_tile,
            support_coord_0_min=support_coord_0_min,
            support_coord_0_max=support_coord_0_max,
            support_coord_1_min=support_coord_1_min,
            support_coord_1_max=support_coord_1_max,
            segment_tile_counts=segment_tile_counts,
            tile_task_segment_idx=empty.tile_task_segment_idx,
            tile_task_tile_idx=empty.tile_task_tile_idx,
            tile_task_entry_coord_0=empty.tile_task_entry_coord_0,
            tile_task_entry_coord_1=empty.tile_task_entry_coord_1,
            tile_task_entry_t=empty.tile_task_entry_t,
            tile_task_entry_t_max_0=empty.tile_task_entry_t_max_0,
            tile_task_entry_t_max_1=empty.tile_task_entry_t_max_1,
            tile_task_counts_per_tile=empty.tile_task_counts_per_tile,
        )

    tile_task_counts_per_tile = dr.zeros(wt.UInt32, n_tiles)
    ones = dr.full(wt.UInt32, 1, tile_task_count)
    dr.scatter_reduce(dr.ReduceOp.Add, tile_task_counts_per_tile, ones, tile_task_tile_idx)
    tile_rx_counts = tile_receiver_counts(receiver_tiles)
    estimated_cell_count = int(
        dr.sum(wt.Float(tile_task_counts_per_tile) * wt.Float(tile_rx_counts))[0]
    )
    max_segments_per_tile = int(dr.max(tile_task_counts_per_tile)[0]) if n_tiles > 0 else 0
    dr.eval(
        support_coord_0_min,
        support_coord_0_max,
        support_coord_1_min,
        support_coord_1_max,
        segment_tile_counts,
        tile_task_segment_idx,
        tile_task_tile_idx,
        tile_task_entry_coord_0,
        tile_task_entry_coord_1,
        tile_task_entry_t,
        tile_task_entry_t_max_0,
        tile_task_entry_t_max_1,
        tile_task_counts_per_tile,
    )
    return SuffixSegmentTilePlan(
        planner_backend=planner_backend,
        n_segments=n_segments,
        n_tiles=n_tiles,
        tile_task_count=tile_task_count,
        estimated_cell_count=estimated_cell_count,
        max_segments_per_tile=max_segments_per_tile,
        support_coord_0_min=support_coord_0_min,
        support_coord_0_max=support_coord_0_max,
        support_coord_1_min=support_coord_1_min,
        support_coord_1_max=support_coord_1_max,
        segment_tile_counts=segment_tile_counts,
        tile_task_segment_idx=tile_task_segment_idx,
        tile_task_tile_idx=tile_task_tile_idx,
        tile_task_entry_coord_0=tile_task_entry_coord_0,
        tile_task_entry_coord_1=tile_task_entry_coord_1,
        tile_task_entry_t=tile_task_entry_t,
        tile_task_entry_t_max_0=tile_task_entry_t_max_0,
        tile_task_entry_t_max_1=tile_task_entry_t_max_1,
        tile_task_counts_per_tile=tile_task_counts_per_tile,
    )


def _build_suffix_segment_tile_plan_native(
    *,
    n_segments: int,
    n_tiles: int,
    receiver_tiles,
    origin_0,
    origin_1,
    dir_0,
    dir_1,
    blocker,
    active_mask,
    support_coord_0_min,
    support_coord_0_max,
    support_coord_1_min,
    support_coord_1_max,
):
    native_planner = _native_suffix_segment_planner()
    if native_planner is None:
        return None

    bound_0_min = float(receiver_tiles.bounds[0][0])
    bound_0_max = float(receiver_tiles.bounds[0][1])
    bound_1_min = float(receiver_tiles.bounds[1][0])
    bound_1_max = float(receiver_tiles.bounds[1][1])
    cell_size_0 = float(receiver_tiles.cell_size[0])
    cell_size_1 = float(receiver_tiles.cell_size[1])
    segment_tile_counts, tile_task_segment_idx, tile_task_tile_idx, tile_task_entry_coord_0, tile_task_entry_coord_1, tile_task_entry_t, tile_task_entry_t_max_0, tile_task_entry_t_max_1 = native_planner.build_suffix_segment_tile_plan_raw(
        bound_0_min,
        bound_0_max,
        bound_1_min,
        bound_1_max,
        cell_size_0,
        cell_size_1,
        int(receiver_tiles.size[0]),
        int(receiver_tiles.size[1]),
        int(receiver_tiles.tile_shape[0]),
        int(receiver_tiles.tile_shape[1]),
        origin_0,
        origin_1,
        dir_0,
        dir_1,
        blocker,
        wt.Int32(active_mask),
    )
    return _finalize_plan(
        planner_backend="native_cuda_exact",
        n_segments=n_segments,
        n_tiles=n_tiles,
        receiver_tiles=receiver_tiles,
        support_coord_0_min=support_coord_0_min,
        support_coord_0_max=support_coord_0_max,
        support_coord_1_min=support_coord_1_min,
        support_coord_1_max=support_coord_1_max,
        segment_tile_counts=segment_tile_counts,
        tile_task_segment_idx=tile_task_segment_idx,
        tile_task_tile_idx=tile_task_tile_idx,
        tile_task_entry_coord_0=tile_task_entry_coord_0,
        tile_task_entry_coord_1=tile_task_entry_coord_1,
        tile_task_entry_t=tile_task_entry_t,
        tile_task_entry_t_max_0=tile_task_entry_t_max_0,
        tile_task_entry_t_max_1=tile_task_entry_t_max_1,
    )


def _build_suffix_segment_tile_plan_drjit(
    *,
    n_segments: int,
    n_tiles: int,
    receiver_tiles,
    origin_0,
    origin_1,
    dir_0,
    dir_1,
    blocker,
    active_mask,
    support_coord_0_min,
    support_coord_0_max,
    support_coord_1_min,
    support_coord_1_max,
):
    bound_0_min = float(receiver_tiles.bounds[0][0])
    bound_0_max = float(receiver_tiles.bounds[0][1])
    bound_1_min = float(receiver_tiles.bounds[1][0])
    bound_1_max = float(receiver_tiles.bounds[1][1])
    cell_size_0 = float(receiver_tiles.cell_size[0])
    cell_size_1 = float(receiver_tiles.cell_size[1])
    valid_mask = (
        active_mask
        & dr.isfinite(origin_0)
        & dr.isfinite(origin_1)
        & dr.isfinite(dir_0)
        & dr.isfinite(dir_1)
        & dr.isfinite(blocker)
        & (blocker > 0.0)
        & (origin_0 >= bound_0_min)
        & (origin_0 < bound_0_max)
        & (origin_1 >= bound_1_min)
        & (origin_1 < bound_1_max)
    )

    chunk_segment_idx_parts: list[object] = []
    chunk_tile_idx_parts: list[object] = []
    chunk_entry_coord_0_parts: list[object] = []
    chunk_entry_coord_1_parts: list[object] = []
    chunk_entry_t_parts: list[object] = []
    chunk_entry_t_max_0_parts: list[object] = []
    chunk_entry_t_max_1_parts: list[object] = []

    segment_idx = dr.arange(wt.UInt32, n_segments)
    abs_dir_0 = dr.maximum(dr.abs(dir_0), _PLANNER_EPS)
    abs_dir_1 = dr.maximum(dr.abs(dir_1), _PLANNER_EPS)
    dt_0 = dr.abs(cell_size_0 / abs_dir_0)
    dt_1 = dr.abs(cell_size_1 / abs_dir_1)
    step_0 = dr.select(dir_0 > 0.0, wt.Float(cell_size_0), wt.Float(-cell_size_0))
    step_1 = dr.select(dir_1 > 0.0, wt.Float(cell_size_1), wt.Float(-cell_size_1))
    next_0 = dr.select(
        dir_0 > 0.0,
        (dr.floor((origin_0 - bound_0_min) / cell_size_0) + 1.0) * cell_size_0 + bound_0_min,
        dr.floor((origin_0 - bound_0_min) / cell_size_0) * cell_size_0 + bound_0_min,
    )
    next_1 = dr.select(
        dir_1 > 0.0,
        (dr.floor((origin_1 - bound_1_min) / cell_size_1) + 1.0) * cell_size_1 + bound_1_min,
        dr.floor((origin_1 - bound_1_min) / cell_size_1) * cell_size_1 + bound_1_min,
    )
    current_coord_0 = origin_0
    current_coord_1 = origin_1
    current_t = dr.zeros(wt.Float, n_segments)
    current_t_max_0 = dr.abs((next_0 - current_coord_0) / abs_dir_0)
    current_t_max_1 = dr.abs((next_1 - current_coord_1) / abs_dir_1)
    previous_tile_idx = dr.full(wt.Int32, -1, n_segments)
    max_steps = 2 * (int(receiver_tiles.size[0]) + int(receiver_tiles.size[1]))
    tile_shape_0 = int(receiver_tiles.tile_shape[0])
    tile_shape_1 = int(receiver_tiles.tile_shape[1])

    for _ in range(max_steps):
        in_bounds = (
            (current_coord_0 >= bound_0_min)
            & (current_coord_0 < bound_0_max)
            & (current_coord_1 >= bound_1_min)
            & (current_coord_1 < bound_1_max)
            & (current_t < blocker)
        )
        active_step = valid_mask & in_bounds
        if not bool(dr.any(active_step)):
            break

        cell_idx_0 = wt.UInt32(
            dr.clip(
                dr.floor((current_coord_0 - bound_0_min) / cell_size_0),
                0.0,
                float(receiver_tiles.size[0] - 1),
            )
        )
        cell_idx_1 = wt.UInt32(
            dr.clip(
                dr.floor((current_coord_1 - bound_1_min) / cell_size_1),
                0.0,
                float(receiver_tiles.size[1] - 1),
            )
        )
        tile_idx_0 = cell_idx_0 // tile_shape_0
        tile_idx_1 = cell_idx_1 // tile_shape_1
        current_tile_idx = tile_idx_1 * int(receiver_tiles.n_tiles_0) + tile_idx_0
        emit_mask = active_step & (wt.Int32(current_tile_idx) != previous_tile_idx)
        kept_segment_idx, kept_tile_idx = compact_tile_tasks(segment_idx, current_tile_idx, emit_mask)
        if dr.width(kept_segment_idx) > 0:
            keep_idx = dr.compress(emit_mask)
            chunk_segment_idx_parts.append(kept_segment_idx)
            chunk_tile_idx_parts.append(kept_tile_idx)
            chunk_entry_coord_0_parts.append(dr.gather(type(current_coord_0), current_coord_0, keep_idx))
            chunk_entry_coord_1_parts.append(dr.gather(type(current_coord_1), current_coord_1, keep_idx))
            chunk_entry_t_parts.append(dr.gather(type(current_t), current_t, keep_idx))
            chunk_entry_t_max_0_parts.append(dr.gather(type(current_t_max_0), current_t_max_0, keep_idx))
            chunk_entry_t_max_1_parts.append(dr.gather(type(current_t_max_1), current_t_max_1, keep_idx))
            previous_tile_idx = dr.select(emit_mask, wt.Int32(current_tile_idx), previous_tile_idx)

        move_0 = current_t_max_0 < current_t_max_1
        step_mask_0 = active_step & move_0
        step_mask_1 = active_step & (~move_0)
        current_t = dr.select(
            step_mask_0,
            current_t_max_0,
            dr.select(step_mask_1, current_t_max_1, current_t),
        )
        current_coord_0 = dr.select(step_mask_0, current_coord_0 + step_0, current_coord_0)
        current_coord_1 = dr.select(step_mask_1, current_coord_1 + step_1, current_coord_1)
        current_t_max_0 = dr.select(step_mask_0, current_t_max_0 + dt_0, current_t_max_0)
        current_t_max_1 = dr.select(step_mask_1, current_t_max_1 + dt_1, current_t_max_1)

    tile_task_segment_idx = Concat.uints(chunk_segment_idx_parts)
    tile_task_tile_idx = Concat.uints(chunk_tile_idx_parts)
    segment_tile_counts = dr.zeros(wt.UInt32, n_segments)
    tile_task_count = dr.width(tile_task_segment_idx)
    if tile_task_count > 0:
        dr.scatter_reduce(
            dr.ReduceOp.Add,
            segment_tile_counts,
            dr.full(wt.UInt32, 1, tile_task_count),
            tile_task_segment_idx,
        )

    return _finalize_plan(
        planner_backend="tile_interval_aabb_drjit",
        n_segments=n_segments,
        n_tiles=n_tiles,
        receiver_tiles=receiver_tiles,
        support_coord_0_min=support_coord_0_min,
        support_coord_0_max=support_coord_0_max,
        support_coord_1_min=support_coord_1_min,
        support_coord_1_max=support_coord_1_max,
        segment_tile_counts=segment_tile_counts,
        tile_task_segment_idx=tile_task_segment_idx,
        tile_task_tile_idx=tile_task_tile_idx,
        tile_task_entry_coord_0=Concat.floats(chunk_entry_coord_0_parts),
        tile_task_entry_coord_1=Concat.floats(chunk_entry_coord_1_parts),
        tile_task_entry_t=Concat.floats(chunk_entry_t_parts),
        tile_task_entry_t_max_0=Concat.floats(chunk_entry_t_max_0_parts),
        tile_task_entry_t_max_1=Concat.floats(chunk_entry_t_max_1_parts),
    )


def build_suffix_segment_tile_plan(
    *,
    seg_origin,
    seg_dir,
    blocker_dist,
    active,
    receiver_tiles,
) -> SuffixSegmentTilePlan | None:
    if receiver_tiles is None:
        return None

    n_segments = int(dr.width(blocker_dist))
    n_tiles = int(receiver_tiles.n_tiles)
    if n_segments <= 0 or n_tiles <= 0:
        return _empty_plan(n_segments, n_tiles)

    axis_0, axis_1 = _tangential_axes(receiver_tiles)
    origin_0 = _detach_component(seg_origin, axis_0)
    origin_1 = _detach_component(seg_origin, axis_1)
    dir_0 = _detach_component(seg_dir, axis_0)
    dir_1 = _detach_component(seg_dir, axis_1)
    blocker = dr.detach(blocker_dist)
    active_mask = active != 0

    bound_0_min = float(receiver_tiles.bounds[0][0])
    bound_0_max = float(receiver_tiles.bounds[0][1])
    bound_1_min = float(receiver_tiles.bounds[1][0])
    bound_1_max = float(receiver_tiles.bounds[1][1])
    cell_size_0 = float(receiver_tiles.cell_size[0])
    cell_size_1 = float(receiver_tiles.cell_size[1])
    coord_eps_0 = max(abs(cell_size_0) * 1.0e-6, _PLANNER_EPS)
    coord_eps_1 = max(abs(cell_size_1) * 1.0e-6, _PLANNER_EPS)
    start_in_bounds = (
        (origin_0 >= bound_0_min)
        & (origin_0 < bound_0_max)
        & (origin_1 >= bound_1_min)
        & (origin_1 < bound_1_max)
    )
    valid_mask = (
        active_mask
        & dr.isfinite(origin_0)
        & dr.isfinite(origin_1)
        & dr.isfinite(dir_0)
        & dr.isfinite(dir_1)
        & dr.isfinite(blocker)
        & (blocker > 0.0)
        & start_in_bounds
    )

    end_coord_0 = origin_0 + dir_0 * blocker
    end_coord_1 = origin_1 + dir_1 * blocker
    support_coord_0_min = dr.select(
        valid_mask,
        dr.maximum(wt.Float(bound_0_min), dr.minimum(origin_0, end_coord_0)),
        dr.zeros(wt.Float, n_segments),
    )
    support_coord_0_max = dr.select(
        valid_mask,
        dr.minimum(wt.Float(bound_0_max), dr.maximum(origin_0, end_coord_0)),
        dr.zeros(wt.Float, n_segments),
    )
    support_coord_1_min = dr.select(
        valid_mask,
        dr.maximum(wt.Float(bound_1_min), dr.minimum(origin_1, end_coord_1)),
        dr.zeros(wt.Float, n_segments),
    )
    support_coord_1_max = dr.select(
        valid_mask,
        dr.minimum(wt.Float(bound_1_max), dr.maximum(origin_1, end_coord_1)),
        dr.zeros(wt.Float, n_segments),
    )

    native_plan = _build_suffix_segment_tile_plan_native(
        n_segments=n_segments,
        n_tiles=n_tiles,
        receiver_tiles=receiver_tiles,
        origin_0=origin_0,
        origin_1=origin_1,
        dir_0=dir_0,
        dir_1=dir_1,
        blocker=blocker,
        active_mask=active_mask,
        support_coord_0_min=support_coord_0_min,
        support_coord_0_max=support_coord_0_max,
        support_coord_1_min=support_coord_1_min,
        support_coord_1_max=support_coord_1_max,
    )
    if native_plan is not None:
        return native_plan

    return _build_suffix_segment_tile_plan_drjit(
        n_segments=n_segments,
        n_tiles=n_tiles,
        receiver_tiles=receiver_tiles,
        origin_0=origin_0,
        origin_1=origin_1,
        dir_0=dir_0,
        dir_1=dir_1,
        blocker=blocker,
        active_mask=active_mask,
        support_coord_0_min=support_coord_0_min,
        support_coord_0_max=support_coord_0_max,
        support_coord_1_min=support_coord_1_min,
        support_coord_1_max=support_coord_1_max,
    )


def build_suffix_tile_packet_plan(
    *,
    seg_dir,
    receiver_tiles,
    tile_plan: SuffixSegmentTilePlan | None,
    state_idx=None,
) -> SuffixTilePacketPlan | None:
    del seg_dir, state_idx
    if receiver_tiles is None or tile_plan is None:
        return None
    packet_counts_per_tile = dr.select(tile_plan.tile_task_counts_per_tile > 0, wt.UInt32(1), wt.UInt32(0))
    packet_count = int(dr.sum(packet_counts_per_tile)[0]) if tile_plan.n_tiles > 0 else 0
    max_segments_per_packet = int(dr.max(tile_plan.tile_task_counts_per_tile)[0]) if tile_plan.n_tiles > 0 else 0
    dr.eval(packet_counts_per_tile)
    return SuffixTilePacketPlan(
        packetizer_backend="disabled_single_packet_drjit",
        n_tiles=int(tile_plan.n_tiles),
        packet_count=packet_count,
        max_packets_per_tile=1 if packet_count > 0 else 0,
        max_segments_per_packet=max_segments_per_packet,
        packet_counts_per_tile=packet_counts_per_tile,
    )


__all__ = [
    "SuffixSegmentTilePlan",
    "SuffixTilePacketPlan",
    "build_suffix_segment_tile_plan",
    "build_suffix_tile_packet_plan",
]
