from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from witwin.channel.scene.models import ReceiverPoint
from witwin.channel.propagation.geometry.kernels import (
    autograd as geometry_autograd,
)
from witwin.channel.propagation.geometry.kernels import (
    primitives as geometry_primitives,
)
from witwin.channel.propagation.topology.kernels import (
    construction as topology_construction,
)

if TYPE_CHECKING:
    from witwin.channel.scene.models import Scene


_PLANE_GROUP_QUANTIZATION = 1.0e-4


def _reflect_points(
    points: torch.Tensor, plane_points: torch.Tensor, normals: torch.Tensor
) -> torch.Tensor:
    return geometry_primitives.deterministic_reflect_points(
        points.contiguous(), plane_points.contiguous(), normals.contiguous()
    )


def _coplanar_face_groups(
    tri_a: torch.Tensor,
    normals: torch.Tensor,
    surface_ids: torch.Tensor,
    *,
    quantization: float = _PLANE_GROUP_QUANTIZATION,
) -> dict[str, torch.Tensor | int]:
    if tri_a.ndim != 2 or tri_a.shape[-1] != 3:
        raise ValueError("tri_a must have shape (face_count, 3)")
    if normals.shape != tri_a.shape:
        raise ValueError("normals must match tri_a shape")
    if surface_ids.ndim != 1 or surface_ids.shape[0] != tri_a.shape[0]:
        raise ValueError("surface_ids must have shape (face_count,)")

    return geometry_primitives.deterministic_face_groups(
        tri_a.to(dtype=torch.float32).contiguous(),
        normals.to(dtype=torch.float32).contiguous(),
        surface_ids.to(device=tri_a.device, dtype=torch.long).contiguous(),
        quantization=float(quantization),
    )


def _cached_coplanar_face_groups(
    rayd: object,
    tri_a: torch.Tensor,
    normals: torch.Tensor,
    surface_ids: torch.Tensor,
) -> dict[str, torch.Tensor | int]:
    """Coplanar face groups are geometry-only; cache them per RayD scene so
    the union-find does not rerun for every component of every solve."""

    cache = getattr(rayd, "runtime_cache", None)
    if cache is None:
        return _coplanar_face_groups(tri_a, normals, surface_ids)
    cached = cache.get("deterministic_coplanar_face_groups")
    if cached is not None:
        return cached  # type: ignore[return-value]
    groups = _coplanar_face_groups(tri_a, normals, surface_ids)
    cache["deterministic_coplanar_face_groups"] = groups
    return groups


def _participates_in_ad(value: object) -> bool:
    """True when a leaf carries a reverse-mode graph or a forward-mode tangent."""

    if not isinstance(value, torch.Tensor):
        return False
    if value.requires_grad:
        return True
    return torch.autograd.forward_ad.unpack_dual(value).tangent is not None


def _geometry_participates_in_ad(scene: Scene) -> bool:
    """True when a geometry leaf (mesh vertices, TX/RX position) is on the graph.

    Materials-only AD (plan 07 AD-1) keeps the detached native hit geometry
    and skips the fixed-winner reconstruction entirely, so AD-2 adds no work
    to a solve that does not ask for geometry gradients.
    """

    leaves: list[object] = [tx.position for tx in scene.transmitters]
    leaves += [
        receiver.position
        for receiver in scene.receivers
        if isinstance(receiver, ReceiverPoint)
    ]
    leaves += [structure.vertices for structure in scene.structures]
    return any(_participates_in_ad(leaf) for leaf in leaves)


def _vertices_participate_in_ad(scene: Scene) -> bool:
    """True when a mesh-vertex leaf carries a graph or a forward tangent."""

    return any(
        _participates_in_ad(structure.vertices) for structure in scene.structures
    )


def _opposite_vertex_ids(
    faces: torch.Tensor, v0_ids: torch.Tensor, v1_ids: torch.Tensor
) -> torch.Tensor:
    """Per-row id of the triangle vertex opposite the (v0, v1) edge.

    Frozen-winner integer extraction on detached tables (the same class of
    winner bookkeeping as the validity masks around it).
    """

    shared = (faces == v0_ids[:, None]) | (faces == v1_ids[:, None])
    opposite_slot = (~shared).to(dtype=torch.int64).argmax(dim=1)
    return faces.gather(1, opposite_slot[:, None])[:, 0]


def _reflection_geometry_ad(
    compiled: object,
    vertices: torch.Tensor,
    source: torch.Tensor,
    target: torch.Tensor,
    face_id: torch.Tensor,
    depth: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Differentiable reflection hit geometry from RayD under the frozen winner.

    Re-launches the native EPC discovery (direct-plane mode) on the winner
    face sequence, so the primal hit points and normals ARE the discovery
    values, and RayD's fixed-winner chain companions provide
    d(hits, normals)/d(vertices, source, target). The plane arrays handed to
    RayD are pure gathers of the same anchor/normal tables the discovery
    consumed; RayD chains the plane cotangents to the winner triangle's
    vertices itself, so nothing geometric is re-derived here.
    """

    rayd = compiled.rayd
    records = rayd.edge_records()
    tri_a = topology_construction.deterministic_face_anchor_points(
        records.vertices.contiguous(), records.faces.contiguous()
    )
    normals_table = geometry_primitives.deterministic_normalize_vec3(
        records.face_normals.contiguous(), eps=1.0e-6
    )
    groups = _cached_coplanar_face_groups(
        rayd,
        tri_a,
        normals_table,
        compiled.geometry.face_surface_id.to(
            device=face_id.device, dtype=torch.long
        ).contiguous(),
    )
    epc = geometry_autograd.rayd_reflection_epc_paths_ad(
        rayd.require_resource(),
        vertices,
        source,
        target,
        face_id.to(dtype=torch.int32).contiguous(),
        tri_a[face_id].contiguous(),
        normals_table[face_id].contiguous(),
        groups["surface_group_id"],
        groups["surface_group_size"],
        groups["surface_group_members"],
        depth,
        1,
    )
    if not bool(epc["valid"].all()):
        raise RuntimeError(
            "fixed-winner EPC re-solve no longer reproduces the discovered "
            "reflection paths; the winner topology moved under the current "
            "scene tensors"
        )
    return epc["hit_positions"], epc["normals"]
