"""Exact reflection-path descriptor aggregation helpers."""

from __future__ import annotations

from dataclasses import dataclass

import drjit as dr
import witwin as wt

from ..materials import ReflectionSourcePathSet
from ...utils.drjit_ops import ArrayInit, Gather


REFLECTION_PATH_IMAGE_SOURCE_TOL = 1e-5


def _gather_scalar(source, index):
    return dr.gather(type(source), source, index)


@dataclass(frozen=True)
class _RayDReflectionChainView:
    ray_count: int
    max_bounces: int
    bounce_count: wt.UInt32
    discovery_count: wt.UInt32
    representative_ray_index: wt.UInt32
    dedup_keep_mask: wt.Bool | None
    prim_ids: tuple[wt.Int32, ...]
    global_prim_ids: tuple[wt.Int32, ...]
    image_sources: tuple[wt.Point3f, ...]
    plane_points: tuple[wt.Point3f, ...]
    plane_normals: tuple[wt.Vector3f, ...]
    hit_points: tuple[wt.Point3f, ...]
    flat_payload: bool = False
    prim_ids_flat: wt.Int32 | None = None
    global_prim_ids_flat: wt.Int32 | None = None
    image_sources_flat: wt.Point3f | None = None
    plane_points_flat: wt.Point3f | None = None
    plane_normals_flat: wt.Vector3f | None = None
    hit_points_flat: wt.Point3f | None = None

    @classmethod
    def from_payload(cls, chain) -> "_RayDReflectionChainView":
        ray_count = int(chain.ray_count)
        max_bounces = int(chain.max_bounces)
        try:
            representative_ray_index = chain.representative_ray_index
        except AttributeError:
            representative_ray_index = dr.arange(wt.UInt32, ray_count)
        try:
            discovery_count = wt.UInt32(chain.discovery_count)
        except AttributeError:
            discovery_count = wt.UInt32(dr.select(chain.bounce_count > 0, 1, 0))
        try:
            dedup_keep_mask = chain.dedup_keep_mask
        except AttributeError:
            dedup_keep_mask = None
        if hasattr(chain, "bounces"):
            bounce_payloads = tuple(chain.bounce(index) for index in range(max_bounces))
            prim_ids = tuple(wt.Int32(bounce.prim_ids) for bounce in bounce_payloads)
            global_prim_ids = tuple(
                wt.Int32(getattr(bounce, "global_prim_ids", bounce.prim_ids))
                for bounce in bounce_payloads
            )
            image_sources = tuple(bounce.image_sources for bounce in bounce_payloads)
            plane_points = tuple(getattr(bounce, "plane_points", bounce.hit_points) for bounce in bounce_payloads)
            plane_normals = tuple(getattr(bounce, "plane_normals", bounce.geo_normals) for bounce in bounce_payloads)
            hit_points = tuple(bounce.hit_points for bounce in bounce_payloads)
        else:
            plane_points_flat = getattr(chain, "plane_points", None)
            if plane_points_flat is None:
                plane_points_flat = chain.hit_points
            plane_normals_flat = getattr(chain, "plane_normals", None)
            if plane_normals_flat is None:
                plane_normals_flat = chain.geo_normals
            prim_ids = tuple()
            global_prim_ids = tuple()
            image_sources = tuple()
            plane_points = tuple()
            plane_normals = tuple()
            hit_points = tuple()
        return cls(
            ray_count=ray_count,
            max_bounces=max_bounces,
            bounce_count=wt.UInt32(chain.bounce_count),
            discovery_count=discovery_count,
            representative_ray_index=wt.UInt32(representative_ray_index),
            dedup_keep_mask=None if dedup_keep_mask is None else wt.Bool(dedup_keep_mask),
            prim_ids=prim_ids,
            global_prim_ids=global_prim_ids,
            image_sources=image_sources,
            plane_points=plane_points,
            plane_normals=plane_normals,
            hit_points=hit_points,
            flat_payload=not hasattr(chain, "bounces"),
            prim_ids_flat=None if hasattr(chain, "bounces") else wt.Int32(chain.prim_ids),
            global_prim_ids_flat=(
                None
                if hasattr(chain, "bounces")
                else wt.Int32(getattr(chain, "global_prim_ids", chain.prim_ids))
            ),
            image_sources_flat=None if hasattr(chain, "bounces") else chain.image_sources,
            plane_points_flat=None if hasattr(chain, "bounces") else plane_points_flat,
            plane_normals_flat=None if hasattr(chain, "bounces") else plane_normals_flat,
            hit_points_flat=None if hasattr(chain, "bounces") else chain.hit_points,
        )

    def _slot_indices(self, representative_chain_indices: list[int], slot: int) -> wt.UInt32:
        return wt.UInt32(
            [
                _chain_flat_index(int(chain_idx), int(slot), self.max_bounces)
                for chain_idx in representative_chain_indices
            ]
        )

    def global_prim_id_scalar(self, slot: int, chain_idx: int) -> int:
        if not self.flat_payload:
            return int(self.global_prim_ids[int(slot)][int(chain_idx)])
        return int(
            self.global_prim_ids_flat[_chain_flat_index(int(chain_idx), int(slot), self.max_bounces)]
        )

    def quantized_image_source_key(
        self,
        slot: int,
        chain_idx: int,
        *,
        tolerance: float,
    ) -> tuple[int, int, int]:
        if not self.flat_payload:
            return _quantize_image_source_key(
                self.image_sources[int(slot)],
                int(chain_idx),
                tolerance=tolerance,
            )
        return _quantize_image_source_key(
            self.image_sources_flat,
            _chain_flat_index(int(chain_idx), int(slot), self.max_bounces),
            tolerance=tolerance,
        )

    def gather_image_sources(self, slot: int, representative_chain_indices: list[int]):
        if not self.flat_payload:
            return Gather.point3(
                self.image_sources[int(slot)],
                wt.UInt32(representative_chain_indices),
            )
        return Gather.point3(
            self.image_sources_flat,
            self._slot_indices(representative_chain_indices, int(slot)),
        )

    def gather_plane_points(self, slot: int, representative_chain_indices: list[int]):
        if not self.flat_payload:
            return Gather.point3(
                self.plane_points[int(slot)],
                wt.UInt32(representative_chain_indices),
            )
        return Gather.point3(
            self.plane_points_flat,
            self._slot_indices(representative_chain_indices, int(slot)),
        )

    def gather_plane_normals(self, slot: int, representative_chain_indices: list[int]):
        if not self.flat_payload:
            return Gather.vector3(
                self.plane_normals[int(slot)],
                wt.UInt32(representative_chain_indices),
            )
        return Gather.vector3(
            self.plane_normals_flat,
            self._slot_indices(representative_chain_indices, int(slot)),
        )

    def gather_hit_points(self, slot: int, representative_chain_indices: list[int]):
        if not self.flat_payload:
            return Gather.point3(
                self.hit_points[int(slot)],
                wt.UInt32(representative_chain_indices),
            )
        return Gather.point3(
            self.hit_points_flat,
            self._slot_indices(representative_chain_indices, int(slot)),
        )


def _empty_source_path_data(chain_depth):
    resolved_depth = int(chain_depth)
    return ReflectionSourcePathSet(
        image_source=ArrayInit.zeros_point3(0),
        discovery_count=dr.zeros(wt.UInt32, 0),
        chain_depth=resolved_depth,
        n_paths=0,
        path_prim_idx=tuple(dr.zeros(wt.Int32, 0) for _ in range(resolved_depth)),
        path_plane_point=tuple(ArrayInit.zeros_point3(0) for _ in range(resolved_depth)),
        path_plane_normal=tuple(ArrayInit.zeros_vector3(0) for _ in range(resolved_depth)),
        path_hit_point=tuple(ArrayInit.zeros_point3(0) for _ in range(resolved_depth)),
    )


def _quantize_image_source_key(point, index: int, *, tolerance: float) -> tuple[int, int, int]:
    inv_tol = 1.0 / max(float(tolerance), 1e-12)
    return (
        int(round(float(point.x[index]) * inv_tol)),
        int(round(float(point.y[index]) * inv_tol)),
        int(round(float(point.z[index]) * inv_tol)),
    )


def _canonical_prim_index(prim_idx: int, surface_canonical_prims) -> int:
    if surface_canonical_prims is None or prim_idx < 0:
        return int(prim_idx)
    return int(surface_canonical_prims[int(prim_idx)])


def _chain_flat_index(chain_idx: int, slot: int, max_bounces: int) -> int:
    return chain_idx * max_bounces + slot


def _collect_reflection_prefix_paths_from_rayd_chain(
    chain,
    *,
    chain_depth,
    surface_canonical_prims=None,
    image_source_tolerance=REFLECTION_PATH_IMAGE_SOURCE_TOL,
):
    chain_view = _RayDReflectionChainView.from_payload(chain)
    if chain_depth <= 0:
        return tuple()

    ray_count = chain_view.ray_count
    max_bounces = chain_view.max_bounces
    n_depths = int(chain_depth)
    if ray_count <= 0 or max_bounces <= 0:
        return tuple(_empty_source_path_data(depth + 1) for depth in range(n_depths))

    prefix_buckets = [dict() for _ in range(n_depths)]

    for chain_idx in range(ray_count):
        if chain_view.dedup_keep_mask is not None and not bool(chain_view.dedup_keep_mask[chain_idx]):
            continue
        bounce_total = min(int(chain_view.bounce_count[chain_idx]), n_depths, max_bounces)
        if bounce_total <= 0:
            continue

        cluster_count = int(chain_view.discovery_count[chain_idx])
        if cluster_count <= 0:
            continue
        first_seen = int(chain_view.representative_ray_index[chain_idx])

        for depth in range(1, bounce_total + 1):
            chain_key = tuple(
                _canonical_prim_index(
                    chain_view.global_prim_id_scalar(slot, chain_idx),
                    surface_canonical_prims,
                )
                for slot in range(depth)
            )
            image_key = chain_view.quantized_image_source_key(
                depth - 1,
                chain_idx,
                tolerance=image_source_tolerance,
            )
            bucket_key = (chain_key, image_key)
            bucket = prefix_buckets[depth - 1].get(bucket_key)
            if bucket is None:
                prefix_buckets[depth - 1][bucket_key] = {
                    "first_seen": first_seen,
                    "discovery_count": cluster_count,
                    "chain": chain_key,
                    "representative_chain_idx": chain_idx,
                }
                continue

            bucket["discovery_count"] = int(bucket["discovery_count"]) + cluster_count
            if first_seen < int(bucket["first_seen"]):
                bucket["first_seen"] = first_seen
                bucket["representative_chain_idx"] = chain_idx

    source_paths = []
    for depth, bucket_map in enumerate(prefix_buckets, start=1):
        if not bucket_map:
            source_paths.append(_empty_source_path_data(depth))
            continue

        ordered_buckets = sorted(bucket_map.values(), key=lambda item: int(item["first_seen"]))
        representative_chain_indices = [int(bucket["representative_chain_idx"]) for bucket in ordered_buckets]
        discovery_count_values = [int(bucket["discovery_count"]) for bucket in ordered_buckets]

        image_source_dr = chain_view.gather_image_sources(depth - 1, representative_chain_indices)
        discovery_count_dr = wt.UInt32(discovery_count_values)
        prim_idx_slots: list[wt.Int32] = []
        plane_point_slots: list[wt.Point3f] = []
        plane_normal_slots: list[wt.Vector3f] = []
        hit_point_slots: list[wt.Point3f] = []
        for slot in range(depth):
            prim_idx_slots.append(wt.Int32([int(bucket["chain"][slot]) for bucket in ordered_buckets]))
            plane_point_slots.append(
                chain_view.gather_plane_points(slot, representative_chain_indices)
            )
            plane_normal_slots.append(
                chain_view.gather_plane_normals(slot, representative_chain_indices)
            )
            hit_point_slots.append(
                chain_view.gather_hit_points(slot, representative_chain_indices)
            )

        dr.eval(
            image_source_dr,
            discovery_count_dr,
            *prim_idx_slots,
            *plane_point_slots,
            *plane_normal_slots,
            *hit_point_slots,
        )
        source_paths.append(
            ReflectionSourcePathSet(
                image_source=image_source_dr,
                discovery_count=discovery_count_dr,
                chain_depth=int(depth),
                n_paths=len(ordered_buckets),
                path_prim_idx=tuple(prim_idx_slots),
                path_plane_point=tuple(plane_point_slots),
                path_plane_normal=tuple(plane_normal_slots),
                path_hit_point=tuple(hit_point_slots),
            )
        )

    return tuple(source_paths)


def _collect_unique_reflection_paths(
    active,
    image_source,
    chain_prim_history,
    chain_depth,
    chain_plane_point_history=None,
    chain_plane_normal_history=None,
    chain_hit_point_history=None,
    surface_canonical_prims=None,
):
    if chain_depth <= 0 or not dr.any(active):
        return _empty_source_path_data(chain_depth)

    active_idx = dr.compress(active)
    n_active = dr.width(active_idx)
    if n_active == 0:
        return _empty_source_path_data(chain_depth)

    if chain_plane_point_history is None:
        chain_plane_point_history = [ArrayInit.zeros_point3(dr.width(image_source.x)) for _ in range(chain_depth)]
    if chain_plane_normal_history is None:
        chain_plane_normal_history = [ArrayInit.zeros_vector3(dr.width(image_source.x)) for _ in range(chain_depth)]
    if chain_hit_point_history is None:
        chain_hit_point_history = [ArrayInit.zeros_point3(dr.width(image_source.x)) for _ in range(chain_depth)]

    chain_groups: dict[tuple[int, ...], list[tuple[int, int]]] = {}
    for active_pos in range(n_active):
        source_idx = int(active_idx[active_pos])
        chain_key = tuple(
            _canonical_prim_index(int(chain_prim_history[slot][source_idx]), surface_canonical_prims)
            for slot in range(chain_depth)
        )
        chain_groups.setdefault(chain_key, []).append((active_pos, source_idx))

    ordered_buckets: list[dict[str, int | tuple[int, ...]]] = []
    for chain_key, members in sorted(chain_groups.items(), key=lambda item: item[1][0][0]):
        clusters: list[dict[str, int | tuple[int, ...]]] = []
        for active_pos, source_idx in members:
            matched_cluster = None
            for cluster in clusters:
                representative_idx = int(cluster["representative_idx"])
                if (
                    abs(float(image_source.x[source_idx]) - float(image_source.x[representative_idx]))
                    <= REFLECTION_PATH_IMAGE_SOURCE_TOL
                    and abs(float(image_source.y[source_idx]) - float(image_source.y[representative_idx]))
                    <= REFLECTION_PATH_IMAGE_SOURCE_TOL
                    and abs(float(image_source.z[source_idx]) - float(image_source.z[representative_idx]))
                    <= REFLECTION_PATH_IMAGE_SOURCE_TOL
                ):
                    matched_cluster = cluster
                    break
            if matched_cluster is None:
                clusters.append(
                    {
                        "first_seen": active_pos,
                        "representative_idx": source_idx,
                        "discovery_count": 1,
                        "chain": chain_key,
                    }
                )
                continue
            matched_cluster["discovery_count"] = int(matched_cluster["discovery_count"]) + 1
        ordered_buckets.extend(clusters)

    ordered_buckets.sort(key=lambda item: int(item["first_seen"]))
    n_groups = len(ordered_buckets)
    if n_groups == 0:
        return _empty_source_path_data(chain_depth)

    representative_indices = wt.UInt32([int(bucket["representative_idx"]) for bucket in ordered_buckets])
    image_source_dr = Gather.point3(image_source, representative_indices)
    discovery_count_dr = wt.UInt32([int(bucket["discovery_count"]) for bucket in ordered_buckets])
    prim_idx_slots = []
    plane_point_slots = []
    plane_normal_slots = []
    hit_point_slots = []
    for slot in range(chain_depth):
        prim_idx_slots.append(wt.Int32([int(bucket["chain"][slot]) for bucket in ordered_buckets]))
        plane_point_slots.append(Gather.point3(chain_plane_point_history[slot], representative_indices))
        plane_normal_slots.append(Gather.vector3(chain_plane_normal_history[slot], representative_indices))
        hit_point_slots.append(Gather.point3(chain_hit_point_history[slot], representative_indices))

    dr.eval(
        image_source_dr,
        discovery_count_dr,
        *prim_idx_slots,
        *plane_point_slots,
        *plane_normal_slots,
        *hit_point_slots,
    )
    return ReflectionSourcePathSet(
        image_source=image_source_dr,
        discovery_count=discovery_count_dr,
        chain_depth=int(chain_depth),
        n_paths=n_groups,
        path_prim_idx=tuple(prim_idx_slots),
        path_plane_point=tuple(plane_point_slots),
        path_plane_normal=tuple(plane_normal_slots),
        path_hit_point=tuple(hit_point_slots),
    )
