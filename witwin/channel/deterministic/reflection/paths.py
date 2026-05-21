"""Reflection path discovery via RayD + source-path collection / dedup."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import drjit as dr
import rayd
from witwin.channel.deterministic import types as wt

from ..kernels.reflection import native_impl as reflection_epc_native
from witwin.channel.core.runtime import Material, Tx, Wave
from witwin.channel.core.numerics.arrays import (
    broadcast_point,
    gather_point3,
    gather_vector3,
    scalar,
    zeros_point3,
    zeros_vector3,
)
from witwin.channel.core.geometry import reflect_point_across_plane

from .common import (
    PATH_IMAGE_SOURCE_TOL,
    grad_sensitive_workload,
    resolve_sampling_info,
    trace_material_info,
    select_ray_directions,
)
from .detail import SourcePathSet, coerce_trace_detail


# ---------- per-bounce field accumulation (EPC replay) ---------------------

def accumulate_paths_exact(
    *,
    rx,
    tx: Tx,
    scene,
    wave: Wave,
    source_paths_per_bounce,
    reflection_detail,
):
    detail = coerce_trace_detail(reflection_detail)
    return reflection_epc_native.reflection_accumulate_forward(
        rx=rx, tx=tx, scene=scene, wave=wave,
        source_paths_per_bounce=source_paths_per_bounce,
        reflection_detail=detail,
    )


# ---------- RayD path tracing ----------------------------------------------

def trace_paths(
    *,
    tx: Tx,
    scene,
    wave: Wave,
    n_rays,
    max_reflections,
    mode,
    material: Material,
    ray_sampling,
    sampling_axis: str,
    sampling_bounds,
    sampling_plane_position: float,
    tri_data,
):
    del wave
    surface_canonical_prims = tri_data["surface_canonical_prim"]
    if dr.width(surface_canonical_prims) == 0:
        raise RuntimeError("Reflection discovery requires non-empty surface canonical primitive data.")
    reflection_model, reflection_model_source = trace_material_info(
        scene=scene, material=material,
    )
    ray_sampling_info = resolve_sampling_info(
        axis=sampling_axis,
        bounds=sampling_bounds,
        tx=tx,
        mode=mode,
        plane_position=sampling_plane_position,
        ray_sampling=ray_sampling,
    )
    if int(max_reflections) == 1:
        return {
            "source_paths_per_bounce": (
                enumerate_first_bounce_surface_paths(tx=tx, tri_data=tri_data),
            ),
            "ray_sampling_info": ray_sampling_info,
            "reflection_model": reflection_model,
            "reflection_model_source": reflection_model_source,
            "reflection_discovery_backend": "analytic_first_bounce",
        }

    rayd_scene = scene._rayd_scene
    if rayd_scene is None:
        raise RuntimeError("Reflection discovery requires scene._rayd_scene.trace_reflections().")

    ray_dir, ray_sampling_info = select_ray_directions(
        axis=sampling_axis,
        bounds=sampling_bounds,
        tx=tx,
        n_rays=n_rays,
        mode=mode,
        plane_position=sampling_plane_position,
        ray_sampling=ray_sampling,
    )
    ray_origin = wt.Point3f(
        dr.repeat(tx.position.x, n_rays),
        dr.repeat(tx.position.y, n_rays),
        dr.repeat(tx.position.z, n_rays),
    )
    ray_origin_detached = dr.detached_t(wt.Point3f)(
        dr.detach(ray_origin.x), dr.detach(ray_origin.y), dr.detach(ray_origin.z),
    )
    ray_dir_detached = dr.detached_t(wt.Vector3f)(
        dr.detach(ray_dir.x), dr.detach(ray_dir.y), dr.detach(ray_dir.z),
    )
    symbolic_workload = grad_sensitive_workload(tx=tx, scene=scene)
    options = rayd.ReflectionTraceOptions()
    options.deduplicate = not symbolic_workload
    options.canonical_prim_table = surface_canonical_prims
    options.image_source_tolerance = float(PATH_IMAGE_SOURCE_TOL)
    if symbolic_workload:
        chain = rayd_scene.trace_reflections(
            rayd.Ray(ray_origin, ray_dir),
            int(max_reflections),
            options,
            dr.full(wt.Bool, True, n_rays),
            True,
        )
    else:
        with dr.scoped_set_flag(dr.JitFlag.Recording, False):
            chain = rayd_scene.trace_reflections(
                rayd.RayDetached(ray_origin_detached, ray_dir_detached),
                int(max_reflections),
                options,
                dr.full(dr.detached_t(wt.Bool), True, n_rays),
                False,
            )

    analytic_first_bounce = enumerate_first_bounce_surface_paths(
        tx=tx,
        tri_data=tri_data,
    )
    prefix_paths, used_native_prefix_compaction = _collect_prefix_paths_native_if_available(
        chain,
        chain_depth=max_reflections,
        surface_canonical_prims=surface_canonical_prims,
        image_source_tolerance=PATH_IMAGE_SOURCE_TOL,
    )
    if used_native_prefix_compaction:
        source_paths_per_bounce = (analytic_first_bounce, *prefix_paths)
    else:
        source_paths_per_bounce = prefix_paths
    if (
        not used_native_prefix_compaction
        and int(analytic_first_bounce.n_paths) > 0
        and source_paths_per_bounce
    ):
        source_paths_per_bounce = (
            analytic_first_bounce,
            *source_paths_per_bounce[1:],
        )
    return {
        "source_paths_per_bounce": tuple(source_paths_per_bounce),
        "ray_sampling_info": ray_sampling_info,
        "reflection_model": reflection_model,
        "reflection_model_source": reflection_model_source,
        "reflection_discovery_backend": (
            "rayd_trace_native_prefix_compaction"
            if used_native_prefix_compaction
            else "rayd_trace_python_prefix_compaction"
        ),
    }


# ---------- source-path collection (RayD flat chain -> SourcePathSet) --------

def quantize_image_source_key(point, index: int, *, tolerance: float) -> tuple[int, int, int]:
    inv_tol = 1.0 / max(float(tolerance), 1e-12)
    return (
        int(round(float(point.x[index]) * inv_tol)),
        int(round(float(point.y[index]) * inv_tol)),
        int(round(float(point.z[index]) * inv_tol)),
    )


def canonical_prim_index(prim_idx: int, surface_canonical_prims) -> int:
    if surface_canonical_prims is None or prim_idx < 0:
        return int(prim_idx)
    return int(surface_canonical_prims[int(prim_idx)])


def empty_source_path_data(chain_depth):
    resolved_depth = int(chain_depth)
    return SourcePathSet(
        image_source=zeros_point3(0),
        discovery_count=dr.zeros(wt.UInt32, 0),
        chain_depth=resolved_depth,
        n_paths=0,
        path_prim_idx=tuple(dr.zeros(wt.Int32, 0) for _ in range(resolved_depth)),
        path_plane_point=tuple(zeros_point3(0) for _ in range(resolved_depth)),
        path_plane_normal=tuple(zeros_vector3(0) for _ in range(resolved_depth)),
        path_hit_point=tuple(zeros_point3(0) for _ in range(resolved_depth)),
    )


def _surface_root_primitives(tri_data):
    n_triangles = int(tri_data.get("n_triangles", 0)) if tri_data is not None else 0
    if n_triangles <= 0 or "surface_canonical_prim" not in tri_data:
        return dr.zeros(wt.Int32, 0)

    tri_idx = dr.arange(wt.UInt32, n_triangles)
    canonical = wt.Int32(tri_data["surface_canonical_prim"])
    root_mask = canonical == wt.Int32(tri_idx)
    root_counts = dr.select(root_mask, wt.UInt32(1), wt.UInt32(0))
    n_roots = int(scalar(dr.sum(root_counts)))
    if n_roots <= 0:
        return dr.zeros(wt.Int32, 0)

    root_prefix = dr.prefix_reduce(dr.ReduceOp.Add, root_counts, exclusive=False)
    root_slot = dr.select(root_mask, root_prefix - wt.UInt32(1), wt.UInt32(0))
    roots = dr.full(wt.Int32, -1, n_roots)
    dr.scatter(roots, wt.Int32(tri_idx), root_slot, root_mask)
    dr.eval(roots)
    return roots


def enumerate_first_bounce_surface_paths(*, tx: Tx, tri_data) -> SourcePathSet:
    """Build one exact first-bounce image source per planar surface group."""
    prim_idx = _surface_root_primitives(tri_data)
    n_paths = int(dr.width(prim_idx))
    if n_paths <= 0:
        return empty_source_path_data(1)

    safe_prim = wt.UInt32(dr.maximum(prim_idx, wt.Int32(0)))
    plane_point = gather_point3(tri_data["v0"], safe_prim)
    v1 = gather_point3(tri_data["v1"], safe_prim)
    v2 = gather_point3(tri_data["v2"], safe_prim)
    rayd_anchor = (plane_point + v1 + v2) * wt.Float(1.0 / 3.0)
    plane_normal = dr.cross(v1 - plane_point, v2 - plane_point)
    plane_normal = plane_normal / (dr.norm(plane_normal) + wt.Float(1e-12))
    image_source = reflect_point_across_plane(
        broadcast_point(tx.position, n_paths),
        plane_point,
        plane_normal,
    )
    discovery_count = dr.full(wt.UInt32, 1, n_paths)
    dr.eval(image_source, discovery_count, prim_idx, plane_point, plane_normal, rayd_anchor)
    return SourcePathSet(
        image_source=image_source,
        discovery_count=discovery_count,
        chain_depth=1,
        n_paths=n_paths,
        path_prim_idx=(prim_idx,),
        path_plane_point=(plane_point,),
        path_plane_normal=(plane_normal,),
        path_hit_point=(rayd_anchor,),
    )


def _source_path_set_from_native_entry(entry, *, chain_depth: int) -> SourcePathSet:
    if isinstance(entry, SourcePathSet):
        return entry
    if isinstance(entry, Mapping):
        return SourcePathSet.from_payload(entry)

    entry_depth = int(getattr(entry, "chain_depth", chain_depth))
    return SourcePathSet(
        image_source=entry.image_source,
        discovery_count=entry.discovery_count,
        chain_depth=entry_depth,
        n_paths=int(getattr(entry, "n_paths", dr.width(entry.discovery_count))),
        path_prim_idx=tuple(entry.path_prim_idx),
        path_plane_point=tuple(entry.path_plane_point),
        path_plane_normal=tuple(entry.path_plane_normal),
        path_hit_point=tuple(entry.path_hit_point),
    )


def _source_path_sets_from_native_prefix_payload(
    payload,
    *,
    min_prefix_depth: int,
    max_prefix_depth: int,
) -> tuple[SourcePathSet, ...]:
    entries = getattr(payload, "source_paths_per_bounce", payload)
    if isinstance(entries, SourcePathSet) or isinstance(entries, Mapping):
        entries = (entries,)
    if not isinstance(entries, Sequence):
        raise TypeError("Native reflection prefix compaction must return a sequence.")

    result: list[SourcePathSet] = []
    expected_depth = int(min_prefix_depth)
    for entry in entries:
        if expected_depth > int(max_prefix_depth):
            break
        if entry is None:
            result.append(empty_source_path_data(expected_depth))
        else:
            result.append(
                _source_path_set_from_native_entry(
                    entry,
                    chain_depth=expected_depth,
                )
            )
        expected_depth += 1

    while expected_depth <= int(max_prefix_depth):
        result.append(empty_source_path_data(expected_depth))
        expected_depth += 1
    return tuple(result)


def _canonicalize_prim_indices(prim_idx, surface_canonical_prims):
    if surface_canonical_prims is None or dr.width(surface_canonical_prims) == 0:
        return wt.Int32(prim_idx)
    safe_prim = dr.maximum(wt.Int32(prim_idx), wt.Int32(0))
    in_table = safe_prim < wt.Int32(dr.width(surface_canonical_prims))
    mapped = dr.gather(wt.Int32, surface_canonical_prims, wt.UInt32(safe_prim), in_table)
    return dr.select((wt.Int32(prim_idx) >= 0) & in_table & (mapped >= 0), mapped, wt.Int32(prim_idx))


def _flat_chain_slot_indices(rep_idx, *, max_bounces: int, slot: int):
    return wt.UInt32(rep_idx) * wt.UInt32(max_bounces) + wt.UInt32(slot)


def _collect_prefix_paths_channel_native(
    chain,
    *,
    chain_depth,
    surface_canonical_prims,
    image_source_tolerance,
) -> tuple[SourcePathSet, ...]:
    if hasattr(chain, "bounce"):
        raise TypeError("Channel native prefix compaction requires flat RayD ReflectionChain layout.")
    ray_count = int(chain.ray_count)
    max_bounces = int(chain.max_bounces)
    if ray_count <= 0 or max_bounces <= 0:
        return tuple(empty_source_path_data(depth) for depth in range(2, int(chain_depth) + 1))

    source_paths: list[SourcePathSet] = []
    for depth in range(2, int(chain_depth) + 1):
        count_arr, rep_full, discovery_full = reflection_epc_native.reflection_prefix_compact_representatives(
            bounce_count=chain.bounce_count,
            discovery_count=chain.discovery_count,
            representative_ray_index=chain.representative_ray_index,
            global_prim_ids=chain.global_prim_ids,
            image_sources=chain.image_sources,
            canonical_prim_table=(
                dr.zeros(wt.Int32, 0)
                if surface_canonical_prims is None
                else wt.Int32(surface_canonical_prims)
            ),
            ray_count=ray_count,
            max_bounces=max_bounces,
            depth=depth,
            image_source_tolerance=image_source_tolerance,
        )
        n_paths = int(count_arr[0])
        if n_paths <= 0:
            source_paths.append(empty_source_path_data(depth))
            continue

        compact_idx = dr.arange(wt.UInt32, n_paths)
        rep_idx_i32 = wt.Int32(dr.gather(type(rep_full), rep_full, compact_idx))
        rep_idx = wt.UInt32(rep_idx_i32)
        discovery_count = wt.UInt32(dr.gather(type(discovery_full), discovery_full, compact_idx))
        img = gather_point3(
            chain.image_sources,
            _flat_chain_slot_indices(rep_idx, max_bounces=max_bounces, slot=depth - 1),
        )
        image_source = wt.Point3f(img.x, img.y, img.z)

        prim_idx_slots: list[wt.Int32] = []
        plane_point_slots: list[wt.Point3f] = []
        plane_normal_slots: list[wt.Vector3f] = []
        hit_point_slots: list[wt.Point3f] = []
        for slot in range(depth):
            slot_idx = _flat_chain_slot_indices(rep_idx, max_bounces=max_bounces, slot=slot)
            prim_idx = wt.Int32(dr.gather(type(chain.global_prim_ids), chain.global_prim_ids, slot_idx))
            prim_idx_slots.append(_canonicalize_prim_indices(prim_idx, surface_canonical_prims))
            pp = gather_point3(chain.plane_points, slot_idx)
            pn = gather_vector3(chain.plane_normals, slot_idx)
            hp = gather_point3(chain.hit_points, slot_idx)
            plane_point_slots.append(wt.Point3f(pp.x, pp.y, pp.z))
            plane_normal_slots.append(wt.Vector3f(pn.x, pn.y, pn.z))
            hit_point_slots.append(wt.Point3f(hp.x, hp.y, hp.z))

        dr.eval(
            image_source,
            discovery_count,
            *prim_idx_slots,
            *plane_point_slots,
            *plane_normal_slots,
            *hit_point_slots,
        )
        source_paths.append(
            SourcePathSet(
                image_source=image_source,
                discovery_count=discovery_count,
                chain_depth=depth,
                n_paths=n_paths,
                path_prim_idx=tuple(prim_idx_slots),
                path_plane_point=tuple(plane_point_slots),
                path_plane_normal=tuple(plane_normal_slots),
                path_hit_point=tuple(hit_point_slots),
            )
        )
    return tuple(source_paths)


def _collect_prefix_paths_native_if_available(
    chain,
    *,
    chain_depth,
    surface_canonical_prims,
    image_source_tolerance,
) -> tuple[tuple[SourcePathSet, ...], bool]:
    if int(chain_depth) < 2:
        return collect_prefix_paths(
            chain,
            chain_depth=chain_depth,
            surface_canonical_prims=surface_canonical_prims,
            image_source_tolerance=image_source_tolerance,
        ), False

    native = getattr(rayd, "compact_reflection_prefix_paths", None)
    if native is None:
        if not hasattr(chain, "bounce"):
            return _collect_prefix_paths_channel_native(
                chain,
                chain_depth=chain_depth,
                surface_canonical_prims=surface_canonical_prims,
                image_source_tolerance=image_source_tolerance,
            ), True
        return collect_prefix_paths(
            chain,
            chain_depth=chain_depth,
            surface_canonical_prims=surface_canonical_prims,
            image_source_tolerance=image_source_tolerance,
        ), False

    payload = native(
        chain,
        surface_canonical_prims,
        float(image_source_tolerance),
        int(chain_depth),
        2,
    )
    return _source_path_sets_from_native_prefix_payload(
        payload,
        min_prefix_depth=2,
        max_prefix_depth=int(chain_depth),
    ), True


@dataclass(frozen=True)
class _NormalizedChain:
    """Per-bounce view over a RayD reflection chain, layout-normalized.

    ``rayd_scene.trace_reflections()`` returns one of two layouts:

    - Symbolic mode -> ``ReflectionTrace[Detached]``: per-bounce data via
      ``chain.bounce(slot)`` and a real ``dedup_keep_mask``.
    - Recording-disabled mode -> ``ReflectionChain[Detached]``: flat arrays of
      length ``ray_count * max_bounces``, no dedup mask.

    This class slices the flat layout into per-bounce arrays once so the
    bucketing loop below can ignore the difference.
    """

    ray_count: int
    max_bounces: int
    bounce_count: object
    discovery_count: object
    representative_ray_index: object
    dedup_keep_mask: object | None
    global_prim_ids: tuple
    image_sources: tuple
    plane_points: tuple
    plane_normals: tuple
    hit_points: tuple


def _normalize_chain(chain) -> _NormalizedChain:
    ray_count = int(chain.ray_count)
    max_bounces = int(chain.max_bounces)

    if hasattr(chain, "bounce"):
        bounces = tuple(chain.bounce(slot) for slot in range(max_bounces))
        global_prim_ids = tuple(b.global_prim_ids for b in bounces)
        image_sources = tuple(b.image_sources for b in bounces)
        plane_points = tuple(b.plane_points for b in bounces)
        plane_normals = tuple(b.plane_normals for b in bounces)
        hit_points = tuple(b.hit_points for b in bounces)
    else:
        # Flat layout: gather slot-aligned slices once.
        per_bounce = lambda flat, gather: tuple(
            gather(flat, wt.UInt32([ci * max_bounces + slot for ci in range(ray_count)]))
            for slot in range(max_bounces)
        )
        global_prim_ids = per_bounce(
            wt.Int32(chain.global_prim_ids),
            lambda f, idx: dr.gather(wt.Int32, f, idx),
        )
        image_sources = per_bounce(chain.image_sources, gather_point3)
        plane_points = per_bounce(chain.plane_points, gather_point3)
        plane_normals = per_bounce(chain.plane_normals, gather_vector3)
        hit_points = per_bounce(chain.hit_points, gather_point3)

    return _NormalizedChain(
        ray_count=ray_count,
        max_bounces=max_bounces,
        bounce_count=chain.bounce_count,
        discovery_count=chain.discovery_count,
        representative_ray_index=chain.representative_ray_index,
        dedup_keep_mask=getattr(chain, "dedup_keep_mask", None),
        global_prim_ids=global_prim_ids,
        image_sources=image_sources,
        plane_points=plane_points,
        plane_normals=plane_normals,
        hit_points=hit_points,
    )


def collect_prefix_paths(
    chain,
    *,
    chain_depth,
    surface_canonical_prims=None,
    image_source_tolerance=PATH_IMAGE_SOURCE_TOL,
):
    # Python-bound exact prefix bucketing. Keep this out of first-order reflection
    # and replace it with native compaction before increasing multi-bounce budgets.
    if chain_depth <= 0:
        return tuple()
    n_depths = int(chain_depth)

    ch = _normalize_chain(chain)
    if ch.ray_count <= 0 or ch.max_bounces <= 0:
        return tuple(empty_source_path_data(depth + 1) for depth in range(n_depths))

    prefix_buckets: list[dict] = [dict() for _ in range(n_depths)]
    for chain_idx in range(ch.ray_count):
        if ch.dedup_keep_mask is not None and not bool(ch.dedup_keep_mask[chain_idx]):
            continue
        bounce_total = min(int(ch.bounce_count[chain_idx]), n_depths, ch.max_bounces)
        if bounce_total <= 0:
            continue
        cluster_count = int(ch.discovery_count[chain_idx])
        if cluster_count <= 0:
            continue
        first_seen = int(ch.representative_ray_index[chain_idx])

        for depth in range(1, bounce_total + 1):
            chain_key = tuple(
                canonical_prim_index(
                    int(ch.global_prim_ids[slot][chain_idx]),
                    surface_canonical_prims,
                )
                for slot in range(depth)
            )
            image_key = quantize_image_source_key(
                ch.image_sources[depth - 1], chain_idx, tolerance=image_source_tolerance,
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
            source_paths.append(empty_source_path_data(depth))
            continue

        ordered = sorted(bucket_map.values(), key=lambda b: int(b["first_seen"]))
        rep_idx = wt.UInt32([int(b["representative_chain_idx"]) for b in ordered])
        img = gather_point3(ch.image_sources[depth - 1], rep_idx)
        image_source_dr = wt.Point3f(img.x, img.y, img.z)
        discovery_count_dr = wt.UInt32([int(b["discovery_count"]) for b in ordered])

        prim_idx_slots: list[wt.Int32] = []
        plane_point_slots: list[wt.Point3f] = []
        plane_normal_slots: list[wt.Vector3f] = []
        hit_point_slots: list[wt.Point3f] = []
        for slot in range(depth):
            prim_idx_slots.append(wt.Int32([int(b["chain"][slot]) for b in ordered]))
            pp = gather_point3(ch.plane_points[slot], rep_idx)
            plane_point_slots.append(wt.Point3f(pp.x, pp.y, pp.z))
            pn = gather_vector3(ch.plane_normals[slot], rep_idx)
            plane_normal_slots.append(wt.Vector3f(pn.x, pn.y, pn.z))
            hp = gather_point3(ch.hit_points[slot], rep_idx)
            hit_point_slots.append(wt.Point3f(hp.x, hp.y, hp.z))

        dr.eval(
            image_source_dr, discovery_count_dr,
            *prim_idx_slots, *plane_point_slots, *plane_normal_slots, *hit_point_slots,
        )
        source_paths.append(
            SourcePathSet(
                image_source=image_source_dr,
                discovery_count=discovery_count_dr,
                chain_depth=int(depth),
                n_paths=len(ordered),
                path_prim_idx=tuple(prim_idx_slots),
                path_plane_point=tuple(plane_point_slots),
                path_plane_normal=tuple(plane_normal_slots),
                path_hit_point=tuple(hit_point_slots),
            )
        )

    return tuple(source_paths)
