# Copyright Xingyu Chen.
# The coupled reflection-diffraction interaction concept.

"""The coupled reflection-diffraction interaction concept."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from witwin.channel.propagation.geometry import (
    cached_diffraction_edge_geometry as _cached_diffraction_edge_geometry,
    diffraction_edge_geometry as _diffraction_edge_geometry,
)
from witwin.channel.kernels import geometry as geometry_kernels
from witwin.channel.propagation.geometry import (
    _cached_coplanar_face_groups,
)
from witwin.channel.propagation.topology import (
    _empty_path_block,
    concatenate_path_blocks,
)
from witwin.channel.propagation.topology import _ensure_topology_fields
from witwin.channel.kernels import topology as topology_kernels

if TYPE_CHECKING:
    from witwin.channel.scene.endpoints import SolverScene as Scene


# --------------------------------------------------------------------------
# Discovery: lazy coupled reflection-diffraction candidate planning.
# --------------------------------------------------------------------------

_COUPLED_CANDIDATE_CHUNK_SIZE = 65_536
# cid-7 (D->D) uses a larger chunk so the ~1260-ordered-pair stream collapses to
# a single native launch per receiver block instead of ~8 (coupled double diffraction
# gate). The R->D / D->R stream stays at 65_536 so its cid-3/4 row identity is
# byte-identical to the frozen P1 baseline; only the cid-7 order (not yet frozen)
# depends on this constant, and it is preserved because the linear candidate
# order is chunk-size independent. Peak transient: one 1M-candidate DD chunk
# materializes ~30 float32 fields/candidate (~100 MB), well within budget; the
# streaming block budget (<=1M candidates/block) caps a block's DD stream below
# this size, so a block never splits its DD stream.
_COUPLED_DD_CANDIDATE_CHUNK_SIZE = 1_048_576
_MAX_COUPLED_CANDIDATES = 1_000_000


@dataclass(frozen=True, slots=True)
class CoupledCandidatePlan:
    tx_count: int
    rx_count: int
    representative_faces: torch.Tensor
    selected_edges: torch.Tensor
    edge_count: int
    candidates_per_pair: int
    dd_candidates_per_pair: int
    base_candidate_count: int
    dd_base_candidate_count: int
    theoretical_candidate_count: int
    chunk_size: int
    dd_chunk_size: int


@dataclass(frozen=True, slots=True)
class CoupledCandidateRequest:
    chunk_start: int
    chunk_end: int
    candidate_count: int
    linear: torch.Tensor
    tx_slot: torch.Tensor
    rx_slot: torch.Tensor
    face_id: torch.Tensor
    edge_id: torch.Tensor
    reverse: bool
    component_id: int


@dataclass(frozen=True, slots=True)
class CoupledDdCandidateRequest:
    chunk_start: int
    chunk_end: int
    candidate_count: int
    linear: torch.Tensor
    tx_slot: torch.Tensor
    rx_slot: torch.Tensor
    edge1_id: torch.Tensor
    edge2_id: torch.Tensor
    component_id: int


def prepare_coupled_candidate_plan(
    *, tx_count: int, rx_count: int, representative_faces: torch.Tensor,
    selected_edges: torch.Tensor, candidate_limit: int,
    chunk_size: int = _COUPLED_CANDIDATE_CHUNK_SIZE,
    dd_chunk_size: int = _COUPLED_DD_CANDIDATE_CHUNK_SIZE,
) -> CoupledCandidatePlan:
    tx_count = int(tx_count)
    rx_count = int(rx_count)
    group_count = int(representative_faces.shape[0])
    edge_count = int(selected_edges.shape[0])
    candidates_per_pair = group_count * edge_count
    # cid 7 (double diffraction) enumerates ordered edge pairs e1 != e2 by
    # index; collinear geometric duplicates are the native kernel's job. One
    # direction only: the ordered pair already covers both traversals.
    dd_candidates_per_pair = edge_count * (edge_count - 1)
    base_candidate_count = tx_count * rx_count * candidates_per_pair
    dd_base_candidate_count = tx_count * rx_count * dd_candidates_per_pair
    # The budget must count the whole coupled union that a block evaluates:
    # both R->D / D->R directions (x2) plus the one-direction D->D stream
    # (coupled double diffraction: per-receiver = tx*(2*groups*edges + edges*(edges-1))).
    theoretical_candidate_count = base_candidate_count * 2 + dd_base_candidate_count
    effective_candidate_limit = min(candidate_limit, _MAX_COUPLED_CANDIDATES)
    if theoretical_candidate_count > effective_candidate_limit:
        raise RuntimeError(
            "coupled reflection-diffraction topology requires "
            f"{theoretical_candidate_count} candidates, exceeding "
            f"coupled_candidate_limit={effective_candidate_limit}"
        )
    return CoupledCandidatePlan(
        tx_count=tx_count,
        rx_count=rx_count,
        representative_faces=representative_faces,
        selected_edges=selected_edges,
        edge_count=edge_count,
        candidates_per_pair=candidates_per_pair,
        dd_candidates_per_pair=dd_candidates_per_pair,
        base_candidate_count=base_candidate_count,
        dd_base_candidate_count=dd_base_candidate_count,
        theoretical_candidate_count=theoretical_candidate_count,
        chunk_size=int(chunk_size),
        dd_chunk_size=int(dd_chunk_size),
    )


def iter_coupled_candidate_requests(
    plan: CoupledCandidatePlan, *, device: torch.device,
) -> Iterator[CoupledCandidateRequest]:
    for start in range(0, plan.base_candidate_count, plan.chunk_size):
        end = min(start + plan.chunk_size, plan.base_candidate_count)
        linear = torch.arange(start, end, device=device, dtype=torch.int64)
        pair_slot = torch.div(
            linear,
            plan.candidates_per_pair,
            rounding_mode="floor",
        )
        local_slot = torch.remainder(linear, plan.candidates_per_pair)
        tx_slot = torch.div(pair_slot, plan.rx_count, rounding_mode="floor")
        rx_slot = torch.remainder(pair_slot, plan.rx_count)
        face_slot = torch.div(local_slot, plan.edge_count, rounding_mode="floor")
        edge_slot = torch.remainder(local_slot, plan.edge_count)
        face_id = plan.representative_faces[face_slot]
        edge_id = plan.selected_edges[edge_slot]
        candidate_count = int(linear.shape[0])
        for reverse, component_id in ((False, 3), (True, 4)):
            yield CoupledCandidateRequest(
                chunk_start=start,
                chunk_end=end,
                candidate_count=candidate_count,
                linear=linear,
                tx_slot=tx_slot,
                rx_slot=rx_slot,
                face_id=face_id,
                edge_id=edge_id,
                reverse=reverse,
                component_id=component_id,
            )


def iter_coupled_dd_candidate_requests(
    plan: CoupledCandidatePlan, *, device: torch.device,
) -> Iterator[CoupledDdCandidateRequest]:
    """Stream cid-7 ordered edge-pair candidates (e1 != e2 by index).

 The candidate space is (tx, rx, e1, e2) with e1 != e2, giving
 ``edge_count*(edge_count-1)`` ordered pairs per (tx, rx) pair and one
 direction only. Geometric collinearity (edges on the same physical line) is
 the native kernel's responsibility, not this index-level enumeration.
 """

    stride = plan.edge_count - 1
    for start in range(0, plan.dd_base_candidate_count, plan.dd_chunk_size):
        end = min(start + plan.dd_chunk_size, plan.dd_base_candidate_count)
        linear = torch.arange(start, end, device=device, dtype=torch.int64)
        pair_slot = torch.div(
            linear,
            plan.dd_candidates_per_pair,
            rounding_mode="floor",
        )
        local_slot = torch.remainder(linear, plan.dd_candidates_per_pair)
        tx_slot = torch.div(pair_slot, plan.rx_count, rounding_mode="floor")
        rx_slot = torch.remainder(pair_slot, plan.rx_count)
        first_slot = torch.div(local_slot, stride, rounding_mode="floor")
        remainder_slot = torch.remainder(local_slot, stride)
        # Skip the diagonal (e1 == e2): the second index steps over its own
        # position so the pair is always ordered and distinct by index.
        second_slot = torch.where(
            remainder_slot < first_slot, remainder_slot, remainder_slot + 1
        )
        edge1_id = plan.selected_edges[first_slot]
        edge2_id = plan.selected_edges[second_slot]
        candidate_count = int(linear.shape[0])
        yield CoupledDdCandidateRequest(
            chunk_start=start,
            chunk_end=end,
            candidate_count=candidate_count,
            linear=linear,
            tx_slot=tx_slot,
            rx_slot=rx_slot,
            edge1_id=edge1_id,
            edge2_id=edge2_id,
            component_id=7,
        )


# --------------------------------------------------------------------------
# Geometry: typed coupled reflection-diffraction geometry queries.
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CoupledGeometryQuery:
    rayd_resource: object
    source: torch.Tensor
    receiver: torch.Tensor
    face_id: torch.Tensor
    face_anchor: torch.Tensor
    face_normal: torch.Tensor
    edge_id: torch.Tensor
    edge_position: torch.Tensor
    edge_direction: torch.Tensor
    edge_t_min: torch.Tensor
    edge_t_max: torch.Tensor
    surface_group_id: torch.Tensor
    surface_group_size: torch.Tensor
    surface_group_members: torch.Tensor
    reverse: bool


@dataclass(frozen=True, slots=True)
class CoupledGeometry:
    valid: torch.Tensor
    interaction_type_sequence: torch.Tensor
    primitive_sequence: torch.Tensor
    edge_sequence: torch.Tensor
    face_id: torch.Tensor
    edge_id: torch.Tensor
    interaction_positions: torch.Tensor
    interaction_normals: torch.Tensor
    reflection_position: torch.Tensor
    reflection_normal: torch.Tensor
    edge_position: torch.Tensor
    edge_direction: torch.Tensor
    path_length_m: torch.Tensor
    delay_s: torch.Tensor


@dataclass(frozen=True, slots=True)
class CoupledDdGeometryQuery:
    rayd_resource: object
    source: torch.Tensor
    receiver: torch.Tensor
    edge1_id: torch.Tensor
    edge1_position: torch.Tensor
    edge1_direction: torch.Tensor
    edge1_t_min: torch.Tensor
    edge1_t_max: torch.Tensor
    edge2_id: torch.Tensor
    edge2_position: torch.Tensor
    edge2_direction: torch.Tensor
    edge2_t_min: torch.Tensor
    edge2_t_max: torch.Tensor


@dataclass(frozen=True, slots=True)
class CoupledDdGeometry:
    valid: torch.Tensor
    interaction_type_sequence: torch.Tensor
    primitive_sequence: torch.Tensor
    edge_sequence: torch.Tensor
    edge1_id: torch.Tensor
    edge2_id: torch.Tensor
    interaction_positions: torch.Tensor
    interaction_normals: torch.Tensor
    edge1_position: torch.Tensor
    edge2_position: torch.Tensor
    path_length_m: torch.Tensor
    delay_s: torch.Tensor


def query_coupled_geometry(query: CoupledGeometryQuery) -> CoupledGeometry:
    raw = geometry_kernels.coupled_rd_geometry_forward(
        query.rayd_resource,
        query.source,
        query.receiver,
        query.face_id,
        query.face_anchor,
        query.face_normal,
        query.edge_id,
        query.edge_position,
        query.edge_direction,
        query.edge_t_min,
        query.edge_t_max,
        query.surface_group_id,
        query.surface_group_size,
        query.surface_group_members,
        query.reverse,
    )
    return CoupledGeometry(
        valid=raw["valid"],
        interaction_type_sequence=raw["interaction_type_sequence"],
        primitive_sequence=raw["primitive_sequence"],
        edge_sequence=raw["edge_sequence"],
        face_id=raw["face_id"],
        edge_id=raw["edge_id"],
        interaction_positions=raw["interaction_positions"],
        interaction_normals=raw["interaction_normals"],
        reflection_position=raw["reflection_position"],
        reflection_normal=raw["reflection_normal"],
        edge_position=raw["edge_position"],
        edge_direction=raw["edge_direction"],
        path_length_m=raw["path_length_m"],
        delay_s=raw["delay_s"],
    )


def query_coupled_dd_geometry(query: CoupledDdGeometryQuery) -> CoupledDdGeometry:
    raw = geometry_kernels.coupled_dd_geometry_forward(
        query.rayd_resource,
        query.source,
        query.receiver,
        query.edge1_id,
        query.edge1_position,
        query.edge1_direction,
        query.edge1_t_min,
        query.edge1_t_max,
        query.edge2_id,
        query.edge2_position,
        query.edge2_direction,
        query.edge2_t_min,
        query.edge2_t_max,
    )
    return CoupledDdGeometry(
        valid=raw["valid"],
        interaction_type_sequence=raw["interaction_type_sequence"],
        primitive_sequence=raw["primitive_sequence"],
        edge_sequence=raw["edge_sequence"],
        edge1_id=raw["edge1_id"],
        edge2_id=raw["edge2_id"],
        interaction_positions=raw["interaction_positions"],
        interaction_normals=raw["interaction_normals"],
        edge1_position=raw["edge1_position"],
        edge2_position=raw["edge2_position"],
        path_length_m=raw["path_length_m"],
        delay_s=raw["delay_s"],
    )


# --------------------------------------------------------------------------
# Enumerated orchestration: coupled reflection-diffraction topology discovery.
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _CoupledTopologyContext:
    """Receiver-independent coupled discovery inputs.

 Every field is a function of the scene geometry and materials only, so a
 single context is reused across all receiver blocks of a streamed
 deterministic solve; ``candidates_per_pair`` (= coplanar groups x selected
 edges) sizes the receiver blocks without re-running the geometry setup.
 """

    device: torch.device
    rayd: object
    representative_faces: torch.Tensor
    tri_a: torch.Tensor
    normals: torch.Tensor
    selected_edges: torch.Tensor
    edge_pos: torch.Tensor
    edge_dir: torch.Tensor
    edge_t_min: torch.Tensor
    edge_t_max: torch.Tensor
    face_material_id: torch.Tensor
    surface_group_id: torch.Tensor
    surface_group_size: torch.Tensor
    surface_group_members: torch.Tensor
    candidates_per_pair: int


def _prepare_coupled_topology_context(
    scene: Scene, compiled: object, tx_positions: torch.Tensor, rx_positions: torch.Tensor,
) -> _CoupledTopologyContext | None:
    """Build the receiver-independent coupled context, or ``None`` when empty.

 Returns ``None`` for every case the single-shot discovery would resolve to
 an empty block (no structures, no endpoints, no faces, or zero
 candidates-per-pair) and raises loudly when RayD native capability is
 missing. No candidate budget is evaluated here; the per-block plan owns the
 total-cap guard.
 """

    device = tx_positions.device
    rayd = compiled.rayd
    if not scene.structures or tx_positions.numel() == 0 or rx_positions.numel() == 0:
        return None
    if not rayd.available:
        raise RuntimeError("coupled topology requires RayD native scene capability")

    records = rayd.edge_records()
    faces = records.faces.contiguous()
    if int(faces.shape[0]) == 0:
        return None
    vertices = records.vertices.contiguous()
    normals = geometry_kernels.deterministic_normalize_vec3(
        records.face_normals.contiguous(), eps=1.0e-6
    )
    tri_a = topology_kernels.deterministic_face_anchor_points(vertices, faces)
    groups = _cached_coplanar_face_groups(
        rayd,
        tri_a,
        normals,
        compiled.geometry.face_surface_id.to(
            device=device, dtype=torch.long
        ).contiguous(),
    )
    representative_faces = (
        groups["representative_faces"].to(dtype=torch.int32).contiguous()
    )
    preserve_imported_edges = bool(
        isinstance(scene.metadata.get("mitsuba", {}), dict)
        and scene.metadata.get("mitsuba", {}).get("merge_shapes", False)
    )
    (
        selected,
        edge_pos,
        edge_dir,
        _edge_length,
        edge_t_min,
        edge_t_max,
        _n0,
        _n1,
        _face0,
        _face1,
        _exterior_angle,
    ) = (
        _diffraction_edge_geometry(records)
        if preserve_imported_edges
        else _cached_diffraction_edge_geometry(rayd)
    )
    selected_edges = topology_kernels.mc_selected_edge_indices(selected)
    candidates_per_pair = int(representative_faces.shape[0]) * int(
        selected_edges.shape[0]
    )
    if candidates_per_pair == 0:
        return None

    face_material_id = compiled.assignments.face_material_id.to(
        device=device, dtype=torch.int32
    ).contiguous()
    surface_group_id = groups["surface_group_id"].to(dtype=torch.int32).contiguous()
    surface_group_size = groups["surface_group_size"].to(dtype=torch.int32).contiguous()
    surface_group_members = (
        groups["surface_group_members"].to(dtype=torch.int32).contiguous()
    )
    return _CoupledTopologyContext(
        device=device,
        rayd=rayd,
        representative_faces=representative_faces,
        tri_a=tri_a,
        normals=normals,
        selected_edges=selected_edges,
        edge_pos=edge_pos,
        edge_dir=edge_dir,
        edge_t_min=edge_t_min,
        edge_t_max=edge_t_max,
        face_material_id=face_material_id,
        surface_group_id=surface_group_id,
        surface_group_size=surface_group_size,
        surface_group_members=surface_group_members,
        candidates_per_pair=candidates_per_pair,
    )


def _coupled_topology_rx_block(
    context: _CoupledTopologyContext, tx_positions: torch.Tensor, rx_positions: torch.Tensor, *,
    candidate_limit: int,
) -> tuple[dict[str, torch.Tensor], int, int]:
    """Discover coupled rows for one receiver slice against a prepared context.

 ``rx_id`` is local to ``rx_positions``; the streamed wrapper offsets it back
 to the global receiver index. ``prepare_coupled_candidate_plan`` runs the
 unchanged total-cap guard on this slice's candidate count, so a slice that
 exceeds the budget fails loudly here.
 """

    device = context.device
    plan = prepare_coupled_candidate_plan(
        tx_count=int(tx_positions.shape[0]),
        rx_count=int(rx_positions.shape[0]),
        representative_faces=context.representative_faces,
        selected_edges=context.selected_edges,
        candidate_limit=candidate_limit,
    )
    if plan.candidates_per_pair == 0:
        return _ensure_topology_fields(_empty_path_block(device)), 0, 0

    tri_a = context.tri_a
    normals = context.normals
    edge_pos = context.edge_pos
    edge_dir = context.edge_dir
    edge_t_min = context.edge_t_min
    edge_t_max = context.edge_t_max
    face_material_id = context.face_material_id
    surface_group_id = context.surface_group_id
    surface_group_size = context.surface_group_size
    surface_group_members = context.surface_group_members
    blocks: list[dict[str, torch.Tensor]] = []
    launch_count = 0
    candidate_count = 0
    current_chunk_start = -1
    common_args: tuple[object, ...] = ()
    for request in iter_coupled_candidate_requests(plan, device=device):
        if request.chunk_start != current_chunk_start:
            edge_index = request.edge_id.to(dtype=torch.int64)
            common_args = (
                context.rayd.require_resource(),
                tx_positions[request.tx_slot].contiguous(),
                rx_positions[request.rx_slot].contiguous(),
                request.face_id,
                tri_a[request.face_id.to(dtype=torch.int64)].contiguous(),
                normals[request.face_id.to(dtype=torch.int64)].contiguous(),
                request.edge_id,
                edge_pos[edge_index].contiguous(),
                edge_dir[edge_index].contiguous(),
                edge_t_min[edge_index].contiguous(),
                edge_t_max[edge_index].contiguous(),
                surface_group_id,
                surface_group_size,
                surface_group_members,
            )
            current_chunk_start = request.chunk_start
        exported = query_coupled_geometry(
            CoupledGeometryQuery(*common_args, reverse=request.reverse)
        )
        launch_count += 1
        candidate_count += request.candidate_count
        kept = torch.nonzero(exported.valid, as_tuple=False).reshape(-1)
        kept_count = int(kept.shape[0])
        if kept_count == 0:
            continue
        interaction_type = exported.interaction_type_sequence[kept]
        primitive_sequence = exported.primitive_sequence[kept]
        edge_sequence = exported.edge_sequence[kept]
        object_sequence = (
            torch.where(interaction_type == 2, edge_sequence, primitive_sequence)
            .to(dtype=torch.int32)
            .contiguous()
        )
        resolved_face = exported.face_id[kept]
        resolved_edge = exported.edge_id[kept]
        reflection_material = face_material_id[resolved_face.to(dtype=torch.int64)]
        material_sequence = (
            torch.where(
                interaction_type == 1,
                reflection_material.reshape(-1, 1),
                torch.full_like(interaction_type, -1),
            )
            .to(dtype=torch.int32)
            .contiguous()
        )
        nan = torch.full(
            (kept_count,), float("nan"), device=device, dtype=torch.float32
        )
        blocks.append(
            _ensure_topology_fields(
                {
                    "valid": torch.ones((kept_count,), device=device, dtype=torch.bool),
                    "tx_id": request.tx_slot[kept].to(dtype=torch.int32).contiguous(),
                    "rx_id": request.rx_slot[kept].to(dtype=torch.int32).contiguous(),
                    "depth": torch.full(
                        (kept_count,), 2, device=device, dtype=torch.int32
                    ),
                    "component_id": torch.full(
                        (kept_count,),
                        request.component_id,
                        device=device,
                        dtype=torch.int32,
                    ),
                    "primitive_id": resolved_face.to(dtype=torch.int32),
                    "edge_id": resolved_edge.to(dtype=torch.int32),
                    "path_length_m": exported.path_length_m[kept],
                    "delay_s": exported.delay_s[kept],
                    "path_gain": nan,
                },
                interaction_position=exported.interaction_positions[kept, 0],
                interaction_normal=exported.interaction_normals[kept, 0],
                material_id=reflection_material,
                path_field=torch.complex(nan, nan),
                primitive_sequence=object_sequence,
                material_sequence=material_sequence,
                interaction_positions=exported.interaction_positions[kept],
                interaction_normals=exported.interaction_normals[kept],
            )
        )
    # cid 7 (double diffraction) shares the same plan and receiver block. Its
    # ordered edge-pair stream is emitted after the R->D / D->R rows so the
    # concatenated per-block row order is deterministic. Each row is a two-edge
    # cascade: both edge ids live in the object sequence, no face is touched.
    # Hoist the native handle out of the chunk loop (mirrors the R->D/D->R
    # common_args hoist above): it is a per-solve constant, not per-chunk work.
    dd_rayd_resource = context.rayd.require_resource()
    for dd_request in iter_coupled_dd_candidate_requests(plan, device=device):
        edge1_index = dd_request.edge1_id.to(dtype=torch.int64)
        edge2_index = dd_request.edge2_id.to(dtype=torch.int64)
        dd_exported = query_coupled_dd_geometry(
            CoupledDdGeometryQuery(
                rayd_resource=dd_rayd_resource,
                source=tx_positions[dd_request.tx_slot].contiguous(),
                receiver=rx_positions[dd_request.rx_slot].contiguous(),
                edge1_id=dd_request.edge1_id,
                edge1_position=edge_pos[edge1_index].contiguous(),
                edge1_direction=edge_dir[edge1_index].contiguous(),
                edge1_t_min=edge_t_min[edge1_index].contiguous(),
                edge1_t_max=edge_t_max[edge1_index].contiguous(),
                edge2_id=dd_request.edge2_id,
                edge2_position=edge_pos[edge2_index].contiguous(),
                edge2_direction=edge_dir[edge2_index].contiguous(),
                edge2_t_min=edge_t_min[edge2_index].contiguous(),
                edge2_t_max=edge_t_max[edge2_index].contiguous(),
            )
        )
        launch_count += 1
        candidate_count += dd_request.candidate_count
        dd_kept = torch.nonzero(dd_exported.valid, as_tuple=False).reshape(-1)
        dd_kept_count = int(dd_kept.shape[0])
        if dd_kept_count == 0:
            continue
        # interaction_type is [2, 2], so the object sequence is exactly the two
        # edge ids; the field stage recovers both edges from edge_sequence
        # (slot 0 = e1, slot 1 = e2). material_sequence is fully -1: edges carry
        # wedge materials resolved by the field stage.
        dd_object_sequence = (
            dd_exported.edge_sequence[dd_kept].to(dtype=torch.int32).contiguous()
        )
        dd_material_sequence = torch.full(
            (dd_kept_count, 2), -1, device=device, dtype=torch.int32
        )
        dd_nan = torch.full(
            (dd_kept_count,), float("nan"), device=device, dtype=torch.float32
        )
        blocks.append(
            _ensure_topology_fields(
                {
                    "valid": torch.ones(
                        (dd_kept_count,), device=device, dtype=torch.bool
                    ),
                    "tx_id": dd_request.tx_slot[dd_kept]
                    .to(dtype=torch.int32)
                    .contiguous(),
                    "rx_id": dd_request.rx_slot[dd_kept]
                    .to(dtype=torch.int32)
                    .contiguous(),
                    "depth": torch.full(
                        (dd_kept_count,), 2, device=device, dtype=torch.int32
                    ),
                    "component_id": torch.full(
                        (dd_kept_count,),
                        dd_request.component_id,
                        device=device,
                        dtype=torch.int32,
                    ),
                    "primitive_id": torch.full(
                        (dd_kept_count,), -1, device=device, dtype=torch.int32
                    ),
                    "edge_id": dd_exported.edge1_id[dd_kept].to(dtype=torch.int32),
                    "path_length_m": dd_exported.path_length_m[dd_kept],
                    "delay_s": dd_exported.delay_s[dd_kept],
                    "path_gain": dd_nan,
                },
                interaction_position=dd_exported.interaction_positions[dd_kept, 0],
                interaction_normal=dd_exported.interaction_normals[dd_kept, 0],
                material_id=torch.full(
                    (dd_kept_count,), -1, device=device, dtype=torch.int32
                ),
                path_field=torch.complex(dd_nan, dd_nan),
                primitive_sequence=dd_object_sequence,
                material_sequence=dd_material_sequence,
                interaction_positions=dd_exported.interaction_positions[dd_kept],
                interaction_normals=dd_exported.interaction_normals[dd_kept],
            )
        )
    return (
        _ensure_topology_fields(concatenate_path_blocks(blocks, device=device)),
        launch_count,
        candidate_count,
    )


def _coupled_reflection_diffraction_topology_order2(
    scene: Scene, compiled: object, tx_positions: torch.Tensor, rx_positions: torch.Tensor, *,
    candidate_limit: int,
) -> tuple[dict[str, torch.Tensor], int, int]:
    """Construct bounded 1R+1D and reciprocal 1D+1R geometry.

 This deliberately exports no physical coefficient. the field evaluator applies
 the shared complex/Jones transport to these canonical event sequences.

 Single-shot discovery over the full receiver set: the whole receiver axis is
 one candidate plan, so a scene whose total candidate count exceeds
 ``candidate_limit`` fails loudly in ``prepare_coupled_candidate_plan``. The
 path and Monte Carlo solvers keep this total-cap contract; the deterministic
 grid solver streams over receiver blocks instead
 (:func:`_coupled_reflection_diffraction_topology_rx_streamed`).
 """

    context = _prepare_coupled_topology_context(
        scene, compiled, tx_positions, rx_positions
    )
    if context is None:
        return _ensure_topology_fields(_empty_path_block(tx_positions.device)), 0, 0
    return _coupled_topology_rx_block(
        context, tx_positions, rx_positions, candidate_limit=candidate_limit
    )


def coupled_reflection_diffraction_topology(
    scene: Scene, compiled: object, tx_positions: torch.Tensor, rx_positions: torch.Tensor, *,
    candidate_limit: int, rx_streamed: bool,
) -> tuple[dict[str, torch.Tensor], int, int]:
    """Dispatch coupled discovery: receiver-streamed grid vs single-shot.

 The deterministic grid solver streams over receiver blocks (coupled reflection and diffraction); the
 path and Monte Carlo callers use the single-shot total-cap discovery.
 """

    topology = (
        _coupled_reflection_diffraction_topology_rx_streamed
        if rx_streamed
        else _coupled_reflection_diffraction_topology_order2
    )
    return topology(
        scene, compiled, tx_positions, rx_positions, candidate_limit=candidate_limit
    )


def _coupled_reflection_diffraction_topology_rx_streamed(
    scene: Scene, compiled: object, tx_positions: torch.Tensor, rx_positions: torch.Tensor, *,
    candidate_limit: int,
) -> tuple[dict[str, torch.Tensor], int, int]:
    """Stream coupled discovery over receiver blocks for the grid solver.

 A full 65k-receiver grid needs far more than the 1M candidate budget in one
 plan, so ``candidate_limit`` is treated as a per-block work/safety budget:
 the receiver axis is split into blocks of ``block_rx`` receivers sized so
 each block's candidate count stays under the (min-with-hard-cap) limit. Each
 block runs the same order-2 discovery as the single-shot path, its local
 ``rx_id`` is offset back to the global receiver index, and the compacted
 blocks are concatenated in ascending receiver-block order. That
 concatenation order IS the row identity and is deterministic across runs.

 The shared ``_MAX_COUPLED_CANDIDATES`` guard and the discovery iterator are
 untouched: a block that cannot fit even a single receiver under the budget
 still fails loudly in ``prepare_coupled_candidate_plan``.
 """

    device = tx_positions.device
    context = _prepare_coupled_topology_context(
        scene, compiled, tx_positions, rx_positions
    )
    if context is None:
        return _ensure_topology_fields(_empty_path_block(device)), 0, 0

    tx_count = int(tx_positions.shape[0])
    rx_count = int(rx_positions.shape[0])
    effective_limit = min(int(candidate_limit), _MAX_COUPLED_CANDIDATES)
    # The block budget counts the whole coupled union a receiver evaluates:
    # both R->D / D->R directions (x2 over groups*edges) plus the one-direction
    # D->D ordered edge-pair stream (edges*(edges-1)) (coupled double diffraction).
    edge_count = int(context.selected_edges.shape[0])
    per_receiver_candidates = tx_count * (
        context.candidates_per_pair * 2 + edge_count * (edge_count - 1)
    )
    block_rx = max(1, effective_limit // max(per_receiver_candidates, 1))
    blocks: list[dict[str, torch.Tensor]] = []
    launch_count = 0
    candidate_count = 0
    for rx_start in range(0, rx_count, block_rx):
        rx_end = min(rx_start + block_rx, rx_count)
        rx_slice = rx_positions[rx_start:rx_end].contiguous()
        block, block_launches, block_candidates = _coupled_topology_rx_block(
            context, tx_positions, rx_slice, candidate_limit=candidate_limit
        )
        launch_count += block_launches
        candidate_count += block_candidates
        if int(block["valid"].numel()) == 0:
            continue
        if rx_start > 0:
            block["rx_id"] = block["rx_id"] + rx_start
        blocks.append(block)
    return (
        _ensure_topology_fields(concatenate_path_blocks(blocks, device=device)),
        launch_count,
        candidate_count,
    )