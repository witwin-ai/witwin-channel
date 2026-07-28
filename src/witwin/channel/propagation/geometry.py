"""Continuous propagation geometry and native facade ownership.

Merged owner of the endpoint layout, RayD visibility and edge-state facades,
the ISB boundary taper (ADR-017) line-of-sight facades, and the frozen-winner
reevaluation geometry. The ISB facades are only ever invoked when the
DEFAULT-OFF ``isb_boundary_taper`` switch is on, so the off solve never
launches anything here and stays bit-identical.

``occluder_boxes`` builds the per-structure axis-aligned box table once from the
compiled geometry vertices. That is a compile-time structural reduction over the
scene's handful of structures (not a per-(tx, rx) hot path); the per-pair
clearance physics runs entirely in the native kernel.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from witwin.channel.kernels.geometry import (
    _BDPT_INTERSECTION_FIELDS,
)
from witwin.channel.kernels import geometry as geometry_kernels
from witwin.channel.kernels import topology as topology_kernels
from witwin.channel.runtime import (
    required_symbol as _required_native_op,
    validate_cuda_tensor,
)
from witwin.channel.scene.resources import refine_edge_geometry
from witwin.channel.scene.resources import RayDSceneResource
from witwin.channel.scene.endpoints import ReceiverGrid, ReceiverPoint
from witwin.channel.scene.compiler import (
    receiver_positions as _native_receiver_positions,
    transmitter_positions as _native_transmitter_positions,
)

if TYPE_CHECKING:
    from witwin.channel.scene.endpoints import SolverScene as Scene

BDPT_INTERSECTION_FIELDS = _BDPT_INTERSECTION_FIELDS


_RAYD_EDGE_INFO_PLANE_TOL = 1.34e-5


def diffraction_edge_geometry(records: object) -> tuple[torch.Tensor, ...]:
    return geometry_kernels.mc_diffraction_edge_geometry(
        records.vertices,
        records.faces,
        records.face_normals,
        records.edge_v0,
        records.edge_v1,
        records.face0,
        records.face1,
        plane_tol=_RAYD_EDGE_INFO_PLANE_TOL,
    )


def cached_diffraction_edge_geometry(
    rayd: RayDSceneResource,
    *,
    preserve_imported_edges: bool = False,
) -> tuple[torch.Tensor, ...]:
    cache = rayd.runtime_cache
    cache_key = (
        "mc_imported_diffraction_edge_geometry"
        if preserve_imported_edges
        else "mc_diffraction_edge_geometry"
    )
    cached = cache.get(cache_key)
    if cached is not None:
        return cached  # type: ignore[return-value]
    geometry = diffraction_edge_geometry(rayd.edge_records())
    if not preserve_imported_edges:
        geometry = refine_edge_geometry(rayd, geometry)
    cache[cache_key] = geometry
    return geometry


@dataclass(frozen=True, slots=True)
class ReceiverLayout:
    """Maps flat receiver ids to the deterministic public result layout."""

    kind: str
    receiver_count: int
    grid_shape: tuple[int, int] | None = None

    def apply(self, values: torch.Tensor) -> torch.Tensor:
        if self.kind == "grid":
            if self.grid_shape is None:
                raise ValueError("grid layout requires grid_shape")
            rows, cols = self.grid_shape
            return (
                values.reshape(values.shape[0], rows, cols).transpose(1, 2).contiguous()
            )
        if self.kind == "point":
            return values.contiguous()
        raise ValueError(f"receiver layout kind is not accepted: {self.kind}")


def transmitter_tensors(
    scene: Scene, *, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    return _native_transmitter_positions(scene, device=device)


def receiver_positions_and_layout(
    scene: Scene, *, device: torch.device
) -> tuple[torch.Tensor, ReceiverLayout]:
    if not scene.receivers:
        return torch.empty((0, 3), device=device, dtype=torch.float32), ReceiverLayout(
            "point", 0
        )

    reference, _power = transmitter_tensors(scene, device=device)
    positions = _native_receiver_positions(scene, device=device, reference=reference)
    if len(scene.receivers) == 1 and isinstance(scene.receivers[0], ReceiverGrid):
        grid = scene.receivers[0]
        return positions, ReceiverLayout("grid", int(positions.shape[0]), grid.shape)

    return positions, ReceiverLayout("point", int(positions.shape[0]))


def apply_receiver_layout(values: torch.Tensor, layout: ReceiverLayout) -> torch.Tensor:
    return layout.apply(values)


_PLANE_GROUP_QUANTIZATION = 1.0e-4


def _reflect_points(
    points: torch.Tensor, plane_points: torch.Tensor, normals: torch.Tensor
) -> torch.Tensor:
    return geometry_kernels.deterministic_reflect_points(
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

    return geometry_kernels.deterministic_face_groups(
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


def reflection_epc_paths(
    compiled: object,
    vertices: torch.Tensor,
    source: torch.Tensor,
    target: torch.Tensor,
    face_id: torch.Tensor,
    depth: int,
) -> dict[str, torch.Tensor]:
    """Frozen-winner reflection EPC re-solve, published with its validity.

    Re-launches the native EPC discovery (direct-plane mode) on the winner
    face sequence, so the primal hit points and normals ARE the discovery
    values, and RayD's fixed-winner chain companions provide
    d(hits, normals)/d(vertices, source, target). The plane arrays handed to
    RayD are pure gathers of the same anchor/normal tables the discovery
    consumed; RayD chains the plane cotangents to the winner triangle's
    vertices itself, so nothing geometric is re-derived here.

    This is the single implementation of the re-solve. What a caller does with
    ``valid`` is the caller's policy: the enumerated fixed-winner path below
    requires every row to survive, while fixed-topology reevaluation publishes
    the mask per row.
    """

    rayd = compiled.rayd
    records = rayd.edge_records()
    tri_a = topology_kernels.deterministic_face_anchor_points(
        records.vertices.contiguous(), records.faces.contiguous()
    )
    normals_table = geometry_kernels.deterministic_normalize_vec3(
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
    epc = geometry_kernels.rayd_reflection_epc_paths_ad(
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
    return epc


def _reflection_geometry_ad(
    compiled: object,
    vertices: torch.Tensor,
    source: torch.Tensor,
    target: torch.Tensor,
    face_id: torch.Tensor,
    depth: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """All-or-nothing fixed-winner reflection hit geometry.

    The enumerated path evaluates a winner it has just discovered under the
    same scene tensors, so a row that stops reproducing is a contract failure,
    not an answer. Reevaluation at NEW endpoint positions is a different
    operation with a different contract and lives in the consumer.
    """

    epc = reflection_epc_paths(compiled, vertices, source, target, face_id, depth)
    if not bool(epc["valid"].all()):
        raise RuntimeError(
            "fixed-winner EPC re-solve no longer reproduces the discovered "
            "reflection paths; the winner topology moved under the current "
            "scene tensors"
        )
    return epc["hit_positions"], epc["normals"]


# Speed of light (m/s); wavelength = c0 / frequency_hz. Matches the LoS kernel
# and artifacts/isb-taper/common.py (lambda = 0.06 m at 5 GHz).
_C0 = 299792458.0


def occluder_boxes(compiled: object) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Per-structure axis-aligned box table (box_min, box_max) or ``None``.

    Reduces the compiled geometry vertices to one [min, max] box per structure
    that owns at least one face. Structures with no faces are dropped. Returns
    ``None`` when the scene has no boxed occluder so the caller keeps the
    fully-lit fast path.
    """

    geometry = compiled.geometry
    # The box reduction is a per-structure amin/amax scatter that must run on the
    # scene device: the compiled geometry tables may live on the host, so pin
    # faces / structure ids / vertices to the vertex device before the scatter
    # (a CPU index against a CUDA accumulator raises otherwise).
    device = geometry.vertices.device
    faces = geometry.faces.to(device=device, dtype=torch.int64)
    face_structure_id = geometry.face_structure_id.to(device=device, dtype=torch.int64)
    if face_structure_id.numel() == 0 or faces.numel() == 0:
        return None
    vertices = geometry.vertices.to(device=device, dtype=torch.float32)
    structure_count = int(face_structure_id.max().item()) + 1
    corner_vertex = faces.reshape(-1)
    corner_structure = face_structure_id.repeat_interleave(3)
    corner_position = vertices[corner_vertex]
    scatter_index = corner_structure.unsqueeze(1).expand(-1, 3)
    box_min = torch.full(
        (structure_count, 3), float("inf"), device=device, dtype=torch.float32
    )
    box_max = torch.full(
        (structure_count, 3), float("-inf"), device=device, dtype=torch.float32
    )
    box_min.scatter_reduce_(
        0, scatter_index, corner_position, reduce="amin", include_self=True
    )
    box_max.scatter_reduce_(
        0, scatter_index, corner_position, reduce="amax", include_self=True
    )
    populated = torch.isfinite(box_min).all(dim=1)
    box_min = box_min[populated].contiguous()
    box_max = box_max[populated].contiguous()
    if box_min.shape[0] == 0:
        return None
    return box_min, box_max


def los_clearance_factor(
    source: torch.Tensor,
    target: torch.Tensor,
    box_min: torch.Tensor,
    box_max: torch.Tensor,
    *,
    frequency_hz: float,
    width: float,
) -> torch.Tensor:
    """Native per-pair ISB membership factor tau in [0, 1] (ADR-017).

    tau = smoothstep01(0.5 * (c_plane / (width * w_F) + 1)) with c the signed
    clearance of the source->target segment past the nearest occluding box
    silhouette (measured at the occluder), c_plane = c * (d1 + d2) / d1 that
    clearance magnified into the receiver plane by the point-source shadow
    factor, and w_F the grazed-edge Fresnel penumbra. The receiver-plane
    magnification matches the accepted projection's in-plane distance transform
    (artifacts/isb-taper/stage2.py); exact conventions live in the CUDA kernel.
    tau > 0 is the membership predicate; tau < 1 is the amplitude factor.
    """

    validate_cuda_tensor("source", source, dtype=torch.float32, ndim=2, trailing_shape=(3,))
    validate_cuda_tensor("target", target, dtype=torch.float32, ndim=2, trailing_shape=(3,))
    validate_cuda_tensor("box_min", box_min, dtype=torch.float32, ndim=2, trailing_shape=(3,))
    validate_cuda_tensor("box_max", box_max, dtype=torch.float32, ndim=2, trailing_shape=(3,))
    if target.shape != source.shape:
        raise ValueError("target must match source")
    if box_max.shape != box_min.shape:
        raise ValueError("box_max must match box_min")
    if frequency_hz <= 0.0:
        raise ValueError("frequency_hz must be positive")
    if not (0.0 < width <= 4.0):
        raise ValueError("isb_boundary_taper_width must be in (0, 4]")
    wavelength = _C0 / float(frequency_hz)
    tau = _required_native_op("los_silhouette_clearance")(
        source.contiguous(),
        target.contiguous(),
        box_min.contiguous(),
        box_max.contiguous(),
        float(wavelength),
        float(width),
    )
    if not isinstance(tau, torch.Tensor):
        raise TypeError("_channel.los_silhouette_clearance must return a tensor")
    validate_cuda_tensor("tau", tau, dtype=torch.float32, ndim=1)
    return tau


def apply_los_taper(
    field_vector: torch.Tensor,
    coefficient: torch.Tensor,
    path_field: torch.Tensor,
    path_gain: torch.Tensor,
    tau: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Native scale of a LoS field bundle by the per-row factor tau (ADR-017).

    tau multiplies the field amplitude; path_gain (a power) is scaled by tau^2.
    """

    validate_cuda_tensor("tau", tau, dtype=torch.float32, ndim=1)
    out = _required_native_op("los_taper_apply")(
        field_vector.contiguous(),
        coefficient.contiguous(),
        path_field.contiguous(),
        path_gain.contiguous(),
        tau.contiguous(),
    )
    if not isinstance(out, (tuple, list)) or len(out) != 4:
        raise TypeError("_channel.los_taper_apply must return four tensors")
    return {
        "field_vector": out[0],
        "coefficient": out[1],
        "path_field": out[2],
        "path_gain": out[3],
    }


@dataclass(frozen=True, slots=True)
class VisibilityQuery:
    rayd: object
    start: torch.Tensor
    end: torch.Tensor
    active: torch.Tensor | None


@dataclass(frozen=True, slots=True)
class VisibilityResult:
    visible: torch.Tensor


def run_visibility_query(query: VisibilityQuery) -> VisibilityResult:
    return VisibilityResult(
        visible=geometry_kernels.rayd_visibility_forward(
            query.rayd.require_resource(),
            query.start,
            query.end,
            query.active,
        )[0]
    )


def _rayd_visibility_mask(
    rayd: object, start: torch.Tensor, end: torch.Tensor
) -> torch.Tensor:
    if start.shape[0] == 0:
        return torch.empty((0,), device=start.device, dtype=torch.bool)
    return geometry_kernels.rayd_visibility_forward(
        rayd.require_resource(), start.contiguous(), end.contiguous(), None
    )[0]


def _los_visibility_mask(
    rayd: object,
    tx_for_path: torch.Tensor,
    rx_for_path: torch.Tensor,
    *,
    has_structures: bool,
) -> torch.Tensor | None:
    if not has_structures or tx_for_path.shape[0] == 0:
        return None
    if not rayd.available:
        raise RuntimeError("LoS visibility requires RayD native scene capability")
    return _rayd_visibility_mask(rayd, tx_for_path, rx_for_path)
