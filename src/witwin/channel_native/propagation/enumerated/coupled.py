"""Enumerated coupled reflection-diffraction topology discovery."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from witwin.channel_native.core.diffraction_geometry import (
    cached_diffraction_edge_geometry as _cached_diffraction_edge_geometry,
    diffraction_edge_geometry as _diffraction_edge_geometry,
)
from witwin.channel_native.propagation.geometry.coupled import (
    CoupledDdGeometryQuery,
    CoupledGeometryQuery,
    query_coupled_dd_geometry,
    query_coupled_geometry,
)
from witwin.channel_native.propagation.geometry.kernels import (
    primitives as geometry_primitives,
)
from witwin.channel_native.propagation.geometry.reevaluate import (
    _cached_coplanar_face_groups,
)
from witwin.channel_native.propagation.topology.concatenate import (
    _empty_path_block,
    concatenate_path_blocks,
)
from witwin.channel_native.propagation.topology.discovery.coupled import (
    _COUPLED_CANDIDATE_CHUNK_SIZE,  # noqa: F401 - compatibility re-export
    _MAX_COUPLED_CANDIDATES,
    iter_coupled_candidate_requests,
    iter_coupled_dd_candidate_requests,
    prepare_coupled_candidate_plan,
)
from witwin.channel_native.propagation.topology.export import _ensure_topology_fields
from witwin.channel_native.propagation.topology.kernels import (
    construction as topology_construction,
)
from witwin.channel_native.propagation.topology.kernels import (
    primitives as topology_primitives,
)

if TYPE_CHECKING:
    from witwin.channel_native.scene.models import Scene


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
    scene: Scene,
    compiled: object,
    tx_positions: torch.Tensor,
    rx_positions: torch.Tensor,
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
    normals = geometry_primitives.deterministic_normalize_vec3(
        records.face_normals.contiguous(), eps=1.0e-6
    )
    tri_a = topology_construction.deterministic_face_anchor_points(vertices, faces)
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
    selected_edges = topology_primitives.mc_selected_edge_indices(selected)
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
    context: _CoupledTopologyContext,
    tx_positions: torch.Tensor,
    rx_positions: torch.Tensor,
    *,
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
    scene: Scene,
    compiled: object,
    tx_positions: torch.Tensor,
    rx_positions: torch.Tensor,
    *,
    candidate_limit: int,
) -> tuple[dict[str, torch.Tensor], int, int]:
    """Construct bounded 1R+1D and reciprocal 1D+1R geometry.

    This phase deliberately exports no physical coefficient. Phase 3 applies
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
    scene: Scene,
    compiled: object,
    tx_positions: torch.Tensor,
    rx_positions: torch.Tensor,
    *,
    candidate_limit: int,
    rx_streamed: bool,
) -> tuple[dict[str, torch.Tensor], int, int]:
    """Dispatch coupled discovery: receiver-streamed grid vs single-shot.

    The deterministic grid solver streams over receiver blocks (ADR-011); the
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
    scene: Scene,
    compiled: object,
    tx_positions: torch.Tensor,
    rx_positions: torch.Tensor,
    *,
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
    # D->D ordered edge-pair stream (edges*(edges-1)) (ADR-013 D1).
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
