from __future__ import annotations

from dataclasses import dataclass

import drjit as dr

import witwin as wt
from witwin.channel._native import _extension, extension_available
from witwin.channel.kernels.monitors.common.receiver_tiles.native_impl import compact_tile_tasks, tile_receiver_counts
from witwin.channel.utils.drjit_ops import Concat
from witwin.channel.utils.shadow_support import shadow_support_angle_from_cutoff_db


_STATE_TILE_BLOCK_SIZE = 2048
_SUPPORT_EPS = 1.0e-7


@dataclass(frozen=True)
class UTDStateTilePlan:
    planner_backend: str
    n_states: int
    n_tiles: int
    tile_task_count: int
    estimated_pair_count: int
    max_states_per_tile: int
    state_tile_counts: object
    tile_task_state_idx: object
    tile_task_tile_idx: object
    tile_task_counts_per_tile: object


def _detach_point3(point):
    return wt.Point3f(
        dr.detach(point.x),
        dr.detach(point.y),
        dr.detach(point.z),
    )


def _detach_vector3(vec):
    return wt.Vector3f(
        dr.detach(vec.x),
        dr.detach(vec.y),
        dr.detach(vec.z),
    )


def _rotate_about_axis(vec, axis, angle):
    cos_angle = dr.cos(angle)
    sin_angle = dr.sin(angle)
    return wt.Vector3f(
        vec.x * cos_angle + dr.cross(axis, vec).x * sin_angle + axis.x * dr.dot(axis, vec) * (1.0 - cos_angle),
        vec.y * cos_angle + dr.cross(axis, vec).y * sin_angle + axis.y * dr.dot(axis, vec) * (1.0 - cos_angle),
        vec.z * cos_angle + dr.cross(axis, vec).z * sin_angle + axis.z * dr.dot(axis, vec) * (1.0 - cos_angle),
    )


def _axis_coordinate_names(axis: str) -> tuple[str, str, str]:
    if axis == "x":
        return "x", "y", "z"
    if axis == "y":
        return "y", "x", "z"
    return "z", "x", "y"


def _plane_linear_coefficients(normals, edge_pos, *, axis: str, plane_position: float):
    axis_name, coord_0_name, coord_1_name = _axis_coordinate_names(axis)
    coeff_0 = getattr(normals, coord_0_name)
    coeff_1 = getattr(normals, coord_1_name)
    bias = (
        getattr(normals, axis_name) * float(plane_position)
        - (normals.x * edge_pos.x + normals.y * edge_pos.y + normals.z * edge_pos.z)
    )
    return coeff_0, coeff_1, bias


def _empty_plan(n_states: int, n_tiles: int) -> UTDStateTilePlan:
    return UTDStateTilePlan(
        planner_backend="wedge_exterior_halfspace_drjit",
        n_states=max(0, int(n_states)),
        n_tiles=max(0, int(n_tiles)),
        tile_task_count=0,
        estimated_pair_count=0,
        max_states_per_tile=0,
        state_tile_counts=dr.zeros(wt.UInt32, max(0, int(n_states))),
        tile_task_state_idx=dr.zeros(wt.UInt32, 0),
        tile_task_tile_idx=dr.zeros(wt.UInt32, 0),
        tile_task_counts_per_tile=dr.zeros(wt.UInt32, max(0, int(n_tiles))),
    )


def _native_utd_state_tile_planner():
    if not extension_available():
        return None
    ext = _extension()
    if not hasattr(ext, "build_utd_state_tile_plan_raw"):
        return None
    return ext


def _finalize_plan(
    *,
    planner_backend: str,
    n_states: int,
    n_tiles: int,
    receiver_tiles,
    state_tile_counts,
    tile_task_state_idx,
    tile_task_tile_idx,
) -> UTDStateTilePlan:
    tile_task_count = dr.width(tile_task_state_idx)
    if tile_task_count == 0:
        empty = _empty_plan(n_states, n_tiles)
        return UTDStateTilePlan(
            planner_backend=planner_backend,
            n_states=empty.n_states,
            n_tiles=empty.n_tiles,
            tile_task_count=empty.tile_task_count,
            estimated_pair_count=empty.estimated_pair_count,
            max_states_per_tile=empty.max_states_per_tile,
            state_tile_counts=state_tile_counts,
            tile_task_state_idx=empty.tile_task_state_idx,
            tile_task_tile_idx=empty.tile_task_tile_idx,
            tile_task_counts_per_tile=empty.tile_task_counts_per_tile,
        )

    tile_task_counts_per_tile = dr.zeros(wt.UInt32, n_tiles)
    ones = dr.full(wt.UInt32, 1, tile_task_count)
    dr.scatter_reduce(dr.ReduceOp.Add, tile_task_counts_per_tile, ones, tile_task_tile_idx)
    tile_rx_counts = tile_receiver_counts(receiver_tiles)
    estimated_pair_count = int(
        dr.sum(wt.Float(tile_task_counts_per_tile) * wt.Float(tile_rx_counts))[0]
    )
    max_states_per_tile = int(dr.max(tile_task_counts_per_tile)[0]) if n_tiles > 0 else 0
    return UTDStateTilePlan(
        planner_backend=planner_backend,
        n_states=n_states,
        n_tiles=n_tiles,
        tile_task_count=tile_task_count,
        estimated_pair_count=estimated_pair_count,
        max_states_per_tile=max_states_per_tile,
        state_tile_counts=state_tile_counts,
        tile_task_state_idx=tile_task_state_idx,
        tile_task_tile_idx=tile_task_tile_idx,
        tile_task_counts_per_tile=tile_task_counts_per_tile,
    )


def _build_utd_state_tile_plan_native(
    *,
    n_states: int,
    n_tiles: int,
    receiver_tiles,
    support_eps: float,
    coeff0_0,
    coeff0_1,
    bias0,
    coeff1_0,
    coeff1_1,
    bias1,
    finite_mask,
    tile_coord_0_min,
    tile_coord_0_max,
    tile_coord_1_min,
    tile_coord_1_max,
) -> UTDStateTilePlan | None:
    native_planner = _native_utd_state_tile_planner()
    if native_planner is None:
        return None

    state_tile_counts, tile_task_state_idx, tile_task_tile_idx = native_planner.build_utd_state_tile_plan_raw(
        float(support_eps),
        coeff0_0,
        coeff0_1,
        bias0,
        coeff1_0,
        coeff1_1,
        bias1,
        wt.Int32(finite_mask),
        tile_coord_0_min,
        tile_coord_0_max,
        tile_coord_1_min,
        tile_coord_1_max,
    )
    return _finalize_plan(
        planner_backend="native_cuda_halfspace_exact",
        n_states=n_states,
        n_tiles=n_tiles,
        receiver_tiles=receiver_tiles,
        state_tile_counts=state_tile_counts,
        tile_task_state_idx=tile_task_state_idx,
        tile_task_tile_idx=tile_task_tile_idx,
    )


def _build_utd_state_tile_plan_drjit(
    *,
    planner_backend: str,
    n_states: int,
    n_tiles: int,
    receiver_tiles,
    support_eps: float,
    coeff0_0,
    coeff0_1,
    bias0,
    coeff1_0,
    coeff1_1,
    bias1,
    finite_mask,
    tile_coord_0_min,
    tile_coord_0_max,
    tile_coord_1_min,
    tile_coord_1_max,
):
    chunk_task_state_idx: list[object] = []
    chunk_task_tile_idx: list[object] = []

    for state_start in range(0, n_states, _STATE_TILE_BLOCK_SIZE):
        chunk_n_states = min(_STATE_TILE_BLOCK_SIZE, n_states - state_start)
        chunk_state_idx = dr.arange(wt.UInt32, chunk_n_states) + wt.UInt32(state_start)
        block_coeff0_0 = dr.gather(type(coeff0_0), coeff0_0, chunk_state_idx)
        block_coeff0_1 = dr.gather(type(coeff0_1), coeff0_1, chunk_state_idx)
        block_bias0 = dr.gather(type(bias0), bias0, chunk_state_idx)
        block_coeff1_0 = dr.gather(type(coeff1_0), coeff1_0, chunk_state_idx)
        block_coeff1_1 = dr.gather(type(coeff1_1), coeff1_1, chunk_state_idx)
        block_bias1 = dr.gather(type(bias1), bias1, chunk_state_idx)
        block_finite_mask = dr.gather(type(finite_mask), finite_mask, chunk_state_idx)

        pair_count = chunk_n_states * n_tiles
        pair_idx = dr.arange(wt.UInt32, pair_count)
        local_state_slot = pair_idx // n_tiles
        tile_idx = pair_idx % n_tiles
        global_state_idx = dr.gather(type(chunk_state_idx), chunk_state_idx, local_state_slot)

        coeff0_0_pair = dr.gather(type(block_coeff0_0), block_coeff0_0, local_state_slot)
        coeff0_1_pair = dr.gather(type(block_coeff0_1), block_coeff0_1, local_state_slot)
        bias0_pair = dr.gather(type(block_bias0), block_bias0, local_state_slot)
        coeff1_0_pair = dr.gather(type(block_coeff1_0), block_coeff1_0, local_state_slot)
        coeff1_1_pair = dr.gather(type(block_coeff1_1), block_coeff1_1, local_state_slot)
        bias1_pair = dr.gather(type(block_bias1), block_bias1, local_state_slot)
        finite_pair = dr.gather(type(block_finite_mask), block_finite_mask, local_state_slot)

        tile_coord_0_min_pair = dr.gather(type(tile_coord_0_min), tile_coord_0_min, tile_idx)
        tile_coord_0_max_pair = dr.gather(type(tile_coord_0_max), tile_coord_0_max, tile_idx)
        tile_coord_1_min_pair = dr.gather(type(tile_coord_1_min), tile_coord_1_min, tile_idx)
        tile_coord_1_max_pair = dr.gather(type(tile_coord_1_max), tile_coord_1_max, tile_idx)

        coord0_face0 = dr.select(coeff0_0_pair >= 0.0, tile_coord_0_max_pair, tile_coord_0_min_pair)
        coord1_face0 = dr.select(coeff0_1_pair >= 0.0, tile_coord_1_max_pair, tile_coord_1_min_pair)
        coord0_face1 = dr.select(coeff1_0_pair >= 0.0, tile_coord_0_max_pair, tile_coord_0_min_pair)
        coord1_face1 = dr.select(coeff1_1_pair >= 0.0, tile_coord_1_max_pair, tile_coord_1_min_pair)

        max0 = coeff0_0_pair * coord0_face0 + coeff0_1_pair * coord1_face0 + bias0_pair
        max1 = coeff1_0_pair * coord0_face1 + coeff1_1_pair * coord1_face1 + bias1_pair
        keep_mask = (max0 >= -support_eps) | (max1 >= -support_eps) | (~finite_pair)

        kept_counts = dr.zeros(wt.UInt32, chunk_n_states)
        dr.scatter_reduce(
            dr.ReduceOp.Add,
            kept_counts,
            dr.select(keep_mask, wt.UInt32(1), wt.UInt32(0)),
            local_state_slot,
        )
        keep_mask = keep_mask | (dr.gather(type(kept_counts), kept_counts, local_state_slot) == 0)

        kept_state_idx, kept_tile_idx = compact_tile_tasks(global_state_idx, tile_idx, keep_mask)
        if dr.width(kept_state_idx) > 0:
            chunk_task_state_idx.append(kept_state_idx)
            chunk_task_tile_idx.append(kept_tile_idx)

    tile_task_state_idx = Concat.uints(chunk_task_state_idx)
    tile_task_tile_idx = Concat.uints(chunk_task_tile_idx)
    state_tile_counts = dr.zeros(wt.UInt32, n_states)
    if dr.width(tile_task_state_idx) > 0:
        dr.scatter_reduce(
            dr.ReduceOp.Add,
            state_tile_counts,
            dr.full(wt.UInt32, 1, dr.width(tile_task_state_idx)),
            tile_task_state_idx,
        )
    return _finalize_plan(
        planner_backend=planner_backend,
        n_states=n_states,
        n_tiles=n_tiles,
        receiver_tiles=receiver_tiles,
        state_tile_counts=state_tile_counts,
        tile_task_state_idx=tile_task_state_idx,
        tile_task_tile_idx=tile_task_tile_idx,
    )


def build_utd_state_tile_plan(
    *,
    state_arrays,
    receiver_tiles,
    include_shadow_completion_band: bool = False,
    shadow_support_cutoff_db: float | None = None,
) -> UTDStateTilePlan | None:
    if state_arrays is None or receiver_tiles is None:
        return None

    n_states = int(state_arrays.get("n_states", 0))
    n_tiles = int(receiver_tiles.n_tiles)
    if n_states <= 0 or n_tiles <= 0:
        return _empty_plan(n_states, n_tiles)

    edge_pos = _detach_point3(state_arrays["edge_pos"])
    edge_dir = _detach_vector3(state_arrays["edge_dir"])
    n0 = _detach_vector3(state_arrays["n0"])
    nn = _detach_vector3(state_arrays["n_face_n"])
    tile_coord_0_min = dr.detach(receiver_tiles.tile_coord_0_min)
    tile_coord_0_max = dr.detach(receiver_tiles.tile_coord_0_max)
    tile_coord_1_min = dr.detach(receiver_tiles.tile_coord_1_min)
    tile_coord_1_max = dr.detach(receiver_tiles.tile_coord_1_max)

    planner_backend = "wedge_exterior_halfspace_drjit"
    if include_shadow_completion_band:
        wedge_n = dr.detach(state_arrays["wedge_n"])
        shadow_support_angle = shadow_support_angle_from_cutoff_db(
            wedge_n,
            shadow_support_cutoff_db,
        )
        n0 = _rotate_about_axis(n0, edge_dir, -shadow_support_angle)
        nn = _rotate_about_axis(nn, edge_dir, shadow_support_angle)
        planner_backend = (
            "shadow_cutoff_halfspace_drjit"
            if shadow_support_cutoff_db is not None
            else "shadow_band_halfspace_drjit"
        )

    support_eps = max(
        _SUPPORT_EPS,
        abs(float(receiver_tiles.cell_size[0])) * _SUPPORT_EPS,
        abs(float(receiver_tiles.cell_size[1])) * _SUPPORT_EPS,
    )
    coeff0_0, coeff0_1, bias0 = _plane_linear_coefficients(
        n0,
        edge_pos,
        axis=receiver_tiles.axis,
        plane_position=receiver_tiles.plane_position,
    )
    coeff1_0, coeff1_1, bias1 = _plane_linear_coefficients(
        nn,
        edge_pos,
        axis=receiver_tiles.axis,
        plane_position=receiver_tiles.plane_position,
    )
    finite_mask = (
        dr.isfinite(coeff0_0)
        & dr.isfinite(coeff0_1)
        & dr.isfinite(bias0)
        & dr.isfinite(coeff1_0)
        & dr.isfinite(coeff1_1)
        & dr.isfinite(bias1)
    )

    if not include_shadow_completion_band:
        native_plan = _build_utd_state_tile_plan_native(
            n_states=n_states,
            n_tiles=n_tiles,
            receiver_tiles=receiver_tiles,
            support_eps=support_eps,
            coeff0_0=coeff0_0,
            coeff0_1=coeff0_1,
            bias0=bias0,
            coeff1_0=coeff1_0,
            coeff1_1=coeff1_1,
            bias1=bias1,
            finite_mask=finite_mask,
            tile_coord_0_min=tile_coord_0_min,
            tile_coord_0_max=tile_coord_0_max,
            tile_coord_1_min=tile_coord_1_min,
            tile_coord_1_max=tile_coord_1_max,
        )
        if native_plan is not None:
            return native_plan

    return _build_utd_state_tile_plan_drjit(
        planner_backend=planner_backend,
        n_states=n_states,
        n_tiles=n_tiles,
        receiver_tiles=receiver_tiles,
        support_eps=support_eps,
        coeff0_0=coeff0_0,
        coeff0_1=coeff0_1,
        bias0=bias0,
        coeff1_0=coeff1_0,
        coeff1_1=coeff1_1,
        bias1=bias1,
        finite_mask=finite_mask,
        tile_coord_0_min=tile_coord_0_min,
        tile_coord_0_max=tile_coord_0_max,
        tile_coord_1_min=tile_coord_1_min,
        tile_coord_1_max=tile_coord_1_max,
    )


__all__ = [
    "UTDStateTilePlan",
    "build_utd_state_tile_plan",
]
