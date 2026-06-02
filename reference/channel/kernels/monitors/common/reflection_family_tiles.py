from __future__ import annotations

from dataclasses import dataclass

import drjit as dr

import witwin as wt

from .receiver_tiles.native_impl import compact_tile_tasks, tile_receiver_counts
from witwin.channel.utils.drjit_ops import Concat


_PROJECTION_DENOM_EPS = 1.0e-8
_FAMILY_TILE_BLOCK_SIZE = 1024


@dataclass(frozen=True)
class ReflectionFamilyTilePlan:
    planner_backend: str
    n_families: int
    n_tiles: int
    tile_task_count: int
    estimated_pair_count: int
    max_families_per_tile: int
    support_coord_0_min: object
    support_coord_0_max: object
    support_coord_1_min: object
    support_coord_1_max: object
    family_tile_counts: object
    tile_task_family_idx: object
    tile_task_tile_idx: object
    tile_task_counts_per_tile: object


def _detach_point3(point):
    return wt.Point3f(
        dr.detach(point.x),
        dr.detach(point.y),
        dr.detach(point.z),
    )


def _axis_coordinate_names(axis: str) -> tuple[str, str, str]:
    if axis == "x":
        return "x", "y", "z"
    if axis == "y":
        return "y", "x", "z"
    return "z", "x", "y"


def _empty_plan(n_families: int, n_tiles: int) -> ReflectionFamilyTilePlan:
    return ReflectionFamilyTilePlan(
        planner_backend="image_transform_aabb_drjit",
        n_families=max(0, int(n_families)),
        n_tiles=max(0, int(n_tiles)),
        tile_task_count=0,
        estimated_pair_count=0,
        max_families_per_tile=0,
        support_coord_0_min=dr.zeros(wt.Float, max(0, int(n_families))),
        support_coord_0_max=dr.zeros(wt.Float, max(0, int(n_families))),
        support_coord_1_min=dr.zeros(wt.Float, max(0, int(n_families))),
        support_coord_1_max=dr.zeros(wt.Float, max(0, int(n_families))),
        family_tile_counts=dr.zeros(wt.UInt32, max(0, int(n_families))),
        tile_task_family_idx=dr.zeros(wt.UInt32, 0),
        tile_task_tile_idx=dr.zeros(wt.UInt32, 0),
        tile_task_counts_per_tile=dr.zeros(wt.UInt32, max(0, int(n_tiles))),
    )


def _project_group_vertices_to_support(*, paths, scene, receiver_tiles, last_prim_idx):
    n_paths = dr.width(last_prim_idx)
    axis_name, coord_0_name, coord_1_name = _axis_coordinate_names(receiver_tiles.axis)
    image_source = _detach_point3(paths["image_source"])
    image_axis = getattr(image_source, axis_name)
    image_coord_0 = getattr(image_source, coord_0_name)
    image_coord_1 = getattr(image_source, coord_1_name)
    plane_position = float(receiver_tiles.plane_position)
    valid_prim = last_prim_idx >= 0
    coplanar_source_mask = dr.abs(image_axis - plane_position) <= _PROJECTION_DENOM_EPS

    support_coord_0_min = dr.full(wt.Float, float(receiver_tiles.bounds[0][0]), n_paths)
    support_coord_0_max = dr.full(wt.Float, float(receiver_tiles.bounds[0][1]), n_paths)
    support_coord_1_min = dr.full(wt.Float, float(receiver_tiles.bounds[1][0]), n_paths)
    support_coord_1_max = dr.full(wt.Float, float(receiver_tiles.bounds[1][1]), n_paths)

    tri_data = scene.tri_data_gpu
    max_group_size = int(tri_data.get("surface_max_group_size", 0))
    has_group_data = (
        max_group_size > 0
        and "surface_group_size" in tri_data
        and "surface_group_members" in tri_data
    )

    raw_min_0 = dr.full(wt.Float, float("inf"), n_paths)
    raw_max_0 = dr.full(wt.Float, float("-inf"), n_paths)
    raw_min_1 = dr.full(wt.Float, float("inf"), n_paths)
    raw_max_1 = dr.full(wt.Float, float("-inf"), n_paths)
    has_projection = dr.full(wt.Bool, False, n_paths)
    safe_last_prim_idx = dr.maximum(last_prim_idx, wt.Int32(0))
    group_size = (
        wt.Int32(
            dr.gather(
                type(tri_data["surface_group_size"]),
                tri_data["surface_group_size"],
                safe_last_prim_idx,
            )
        )
        if has_group_data
        else dr.zeros(wt.Int32, n_paths)
    )

    for slot in range(max(1, max_group_size)):
        use_group_member = has_group_data and slot < max_group_size
        if use_group_member:
            group_member_active = valid_prim & (group_size > 0) & (group_size > wt.Int32(slot))
            fallback_member_active = valid_prim & (group_size <= 0) & (slot == 0)
            flat_idx = safe_last_prim_idx * max_group_size + slot
            gathered_member_idx = wt.Int32(
                dr.gather(
                    type(tri_data["surface_group_members"]),
                    tri_data["surface_group_members"],
                    flat_idx,
                )
            )
            member_idx = dr.select(group_member_active, gathered_member_idx, last_prim_idx)
            slot_active = group_member_active | fallback_member_active
        else:
            member_idx = last_prim_idx
            slot_active = valid_prim & (slot == 0)

        slot_active = slot_active & (member_idx >= 0)
        safe_member_idx = dr.maximum(member_idx, wt.Int32(0))
        for vertex_name in ("v0", "v1", "v2"):
            vertex = _detach_point3(dr.gather(wt.Point3f, tri_data[vertex_name], safe_member_idx))
            vertex_axis = getattr(vertex, axis_name)
            vertex_coord_0 = getattr(vertex, coord_0_name)
            vertex_coord_1 = getattr(vertex, coord_1_name)
            denom = vertex_axis - image_axis
            denom_safe = dr.select(
                dr.abs(denom) > _PROJECTION_DENOM_EPS,
                denom,
                dr.select(denom >= 0.0, wt.Float(_PROJECTION_DENOM_EPS), wt.Float(-_PROJECTION_DENOM_EPS)),
            )
            t = (plane_position - image_axis) / denom_safe
            projected_coord_0 = image_coord_0 + t * (vertex_coord_0 - image_coord_0)
            projected_coord_1 = image_coord_1 + t * (vertex_coord_1 - image_coord_1)
            valid_projection = (
                slot_active
                & (dr.abs(denom) > _PROJECTION_DENOM_EPS)
                & dr.isfinite(t)
                & dr.isfinite(projected_coord_0)
                & dr.isfinite(projected_coord_1)
            )
            raw_min_0 = dr.select(valid_projection, dr.minimum(raw_min_0, projected_coord_0), raw_min_0)
            raw_max_0 = dr.select(valid_projection, dr.maximum(raw_max_0, projected_coord_0), raw_max_0)
            raw_min_1 = dr.select(valid_projection, dr.minimum(raw_min_1, projected_coord_1), raw_min_1)
            raw_max_1 = dr.select(valid_projection, dr.maximum(raw_max_1, projected_coord_1), raw_max_1)
            has_projection = has_projection | valid_projection

    support_margin_0 = max(abs(float(receiver_tiles.cell_size[0])), _PROJECTION_DENOM_EPS)
    support_margin_1 = max(abs(float(receiver_tiles.cell_size[1])), _PROJECTION_DENOM_EPS)
    use_projected_support = valid_prim & (~coplanar_source_mask) & has_projection
    support_coord_0_min = dr.select(use_projected_support, raw_min_0 - support_margin_0, support_coord_0_min)
    support_coord_0_max = dr.select(use_projected_support, raw_max_0 + support_margin_0, support_coord_0_max)
    support_coord_1_min = dr.select(use_projected_support, raw_min_1 - support_margin_1, support_coord_1_min)
    support_coord_1_max = dr.select(use_projected_support, raw_max_1 + support_margin_1, support_coord_1_max)
    dr.eval(
        support_coord_0_min,
        support_coord_0_max,
        support_coord_1_min,
        support_coord_1_max,
    )
    return (
        support_coord_0_min,
        support_coord_0_max,
        support_coord_1_min,
        support_coord_1_max,
    )


def build_reflection_family_tile_plan(*, paths, scene, receiver_tiles) -> ReflectionFamilyTilePlan | None:
    if paths is None or scene is None or receiver_tiles is None:
        return None
    tri_data = getattr(scene, "tri_data_gpu", None)
    if tri_data is None:
        return None

    chain_depth = int(paths.get("chain_depth", 0))
    n_paths = int(paths.get("n_paths", 0))
    n_tiles = int(receiver_tiles.n_tiles)
    if chain_depth <= 0 or n_paths <= 0 or n_tiles <= 0:
        return _empty_plan(n_paths, n_tiles)

    last_prim_idx = wt.Int32(dr.detach(paths[f"path_prim_idx_{chain_depth - 1}"]))
    (
        support_coord_0_min,
        support_coord_0_max,
        support_coord_1_min,
        support_coord_1_max,
    ) = _project_group_vertices_to_support(
        paths=paths,
        scene=scene,
        receiver_tiles=receiver_tiles,
        last_prim_idx=last_prim_idx,
    )

    chunk_task_family_idx: list[object] = []
    chunk_task_tile_idx: list[object] = []
    tile_coord_0_min = dr.detach(receiver_tiles.tile_coord_0_min)
    tile_coord_0_max = dr.detach(receiver_tiles.tile_coord_0_max)
    tile_coord_1_min = dr.detach(receiver_tiles.tile_coord_1_min)
    tile_coord_1_max = dr.detach(receiver_tiles.tile_coord_1_max)

    for family_start in range(0, n_paths, _FAMILY_TILE_BLOCK_SIZE):
        chunk_n_families = min(_FAMILY_TILE_BLOCK_SIZE, n_paths - family_start)
        chunk_family_idx = dr.arange(wt.UInt32, chunk_n_families) + wt.UInt32(family_start)
        pair_count = chunk_n_families * n_tiles
        pair_idx = dr.arange(wt.UInt32, pair_count)
        local_family_slot = pair_idx // n_tiles
        tile_idx = pair_idx % n_tiles
        global_family_idx = dr.gather(type(chunk_family_idx), chunk_family_idx, local_family_slot)

        family_support_0_min = dr.gather(type(support_coord_0_min), support_coord_0_min, global_family_idx)
        family_support_0_max = dr.gather(type(support_coord_0_max), support_coord_0_max, global_family_idx)
        family_support_1_min = dr.gather(type(support_coord_1_min), support_coord_1_min, global_family_idx)
        family_support_1_max = dr.gather(type(support_coord_1_max), support_coord_1_max, global_family_idx)
        tile_support_0_min = dr.gather(type(tile_coord_0_min), tile_coord_0_min, tile_idx)
        tile_support_0_max = dr.gather(type(tile_coord_0_max), tile_coord_0_max, tile_idx)
        tile_support_1_min = dr.gather(type(tile_coord_1_min), tile_coord_1_min, tile_idx)
        tile_support_1_max = dr.gather(type(tile_coord_1_max), tile_coord_1_max, tile_idx)
        overlap_mask = (
            (family_support_0_max >= tile_support_0_min)
            & (family_support_0_min <= tile_support_0_max)
            & (family_support_1_max >= tile_support_1_min)
            & (family_support_1_min <= tile_support_1_max)
        )
        kept_family_idx, kept_tile_idx = compact_tile_tasks(global_family_idx, tile_idx, overlap_mask)
        if dr.width(kept_family_idx) > 0:
            chunk_task_family_idx.append(kept_family_idx)
            chunk_task_tile_idx.append(kept_tile_idx)

    tile_task_family_idx = Concat.uints(chunk_task_family_idx)
    tile_task_tile_idx = Concat.uints(chunk_task_tile_idx)
    tile_task_count = dr.width(tile_task_family_idx)
    if tile_task_count == 0:
        return _empty_plan(n_paths, n_tiles)

    family_tile_counts = dr.zeros(wt.UInt32, n_paths)
    tile_task_counts_per_tile = dr.zeros(wt.UInt32, n_tiles)
    ones = dr.full(wt.UInt32, 1, tile_task_count)
    dr.scatter_reduce(dr.ReduceOp.Add, family_tile_counts, ones, tile_task_family_idx)
    dr.scatter_reduce(dr.ReduceOp.Add, tile_task_counts_per_tile, ones, tile_task_tile_idx)
    tile_rx_counts = tile_receiver_counts(receiver_tiles)
    estimated_pair_count = int(
        dr.sum(wt.Float(tile_task_counts_per_tile) * wt.Float(tile_rx_counts))[0]
    )
    max_families_per_tile = int(dr.max(tile_task_counts_per_tile)[0]) if n_tiles > 0 else 0
    dr.eval(
        support_coord_0_min,
        support_coord_0_max,
        support_coord_1_min,
        support_coord_1_max,
        family_tile_counts,
        tile_task_family_idx,
        tile_task_tile_idx,
        tile_task_counts_per_tile,
    )
    return ReflectionFamilyTilePlan(
        planner_backend="image_transform_aabb_drjit",
        n_families=n_paths,
        n_tiles=n_tiles,
        tile_task_count=tile_task_count,
        estimated_pair_count=estimated_pair_count,
        max_families_per_tile=max_families_per_tile,
        support_coord_0_min=support_coord_0_min,
        support_coord_0_max=support_coord_0_max,
        support_coord_1_min=support_coord_1_min,
        support_coord_1_max=support_coord_1_max,
        family_tile_counts=family_tile_counts,
        tile_task_family_idx=tile_task_family_idx,
        tile_task_tile_idx=tile_task_tile_idx,
        tile_task_counts_per_tile=tile_task_counts_per_tile,
    )


__all__ = [
    "ReflectionFamilyTilePlan",
    "build_reflection_family_tile_plan",
]
