"""Enumerated coupled reflection-diffraction topology discovery."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from witwin.channel_native.core.diffraction_geometry import (
    cached_diffraction_edge_geometry as _cached_diffraction_edge_geometry,
    diffraction_edge_geometry as _diffraction_edge_geometry,
)
from witwin.channel_native.propagation.geometry.coupled import (
    CoupledGeometryQuery,
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
    _MAX_COUPLED_CANDIDATES,  # noqa: F401 - compatibility re-export
    iter_coupled_candidate_requests,
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
    from witwin.channel_native.core.scene import Scene


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
    """

    device = tx_positions.device
    raydn = compiled.raydn
    if not scene.structures or tx_positions.numel() == 0 or rx_positions.numel() == 0:
        return _ensure_topology_fields(_empty_path_block(device)), 0, 0
    if not raydn.available:
        raise RuntimeError("coupled topology requires RayDN native scene capability")

    records = raydn.edge_records()
    faces = records.faces.contiguous()
    if int(faces.shape[0]) == 0:
        return _ensure_topology_fields(_empty_path_block(device)), 0, 0
    vertices = records.vertices.contiguous()
    normals = geometry_primitives.deterministic_normalize_vec3(
        records.face_normals.contiguous(), eps=1.0e-6
    )
    tri_a = topology_construction.deterministic_face_anchor_points(vertices, faces)
    groups = _cached_coplanar_face_groups(
        raydn,
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
        else _cached_diffraction_edge_geometry(raydn)
    )
    selected_edges = topology_primitives.mc_selected_edge_indices(selected)
    plan = prepare_coupled_candidate_plan(
        tx_count=int(tx_positions.shape[0]),
        rx_count=int(rx_positions.shape[0]),
        representative_faces=representative_faces,
        selected_edges=selected_edges,
        candidate_limit=candidate_limit,
    )
    if plan.candidates_per_pair == 0:
        return _ensure_topology_fields(_empty_path_block(device)), 0, 0

    face_material_id = compiled.assignments.face_material_id.to(
        device=device, dtype=torch.int32
    ).contiguous()
    blocks: list[dict[str, torch.Tensor]] = []
    launch_count = 0
    candidate_count = 0
    surface_group_id = groups["surface_group_id"].to(dtype=torch.int32).contiguous()
    surface_group_size = groups["surface_group_size"].to(dtype=torch.int32).contiguous()
    surface_group_members = (
        groups["surface_group_members"].to(dtype=torch.int32).contiguous()
    )
    current_chunk_start = -1
    common_args: tuple[object, ...] = ()
    for request in iter_coupled_candidate_requests(plan, device=device):
        if request.chunk_start != current_chunk_start:
            edge_index = request.edge_id.to(dtype=torch.int64)
            common_args = (
                raydn.require_handle(),
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
    return (
        _ensure_topology_fields(concatenate_path_blocks(blocks, device=device)),
        launch_count,
        candidate_count,
    )
