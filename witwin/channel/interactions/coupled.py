# Copyright Xingyu Chen.
# Interaction chains: reflections with one diffraction anywhere, and double diffraction.

"""Interaction chains: reflections with one diffraction anywhere, and double diffraction."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from witwin.channel.interactions.diffraction import (
    _batched_diffraction_states,
    plan_tx_visible_diffraction_states,
    prepare_diffraction_order1_plan,
    selected_diffraction_edges,
)
from witwin.channel.interactions.reflection import (
    ReflectionPlaneGroups,
    _expand_image_tree,
    _image_tree_roots,
    _reflection_geometry_tables,
    reflection_min_segment_length,
    reflection_plane_groups,
)
from witwin.channel.kernels import geometry as geometry_kernels
from witwin.channel.kernels import topology as topology_kernels
from witwin.channel.materials import face_geometry_id_table, face_material_id_table
from witwin.channel.propagation.geometry import _cached_coplanar_face_groups
from witwin.channel.propagation.topology import (
    _empty_path_block,
    _ensure_topology_fields,
    concatenate_path_blocks,
)

if TYPE_CHECKING:
    from witwin.channel.interactions.diffraction import DiffractionEdgeGeometry
    from witwin.channel.interactions.reflection import _ImageTreeFrontier
    from witwin.channel.scene.compiler import CompiledScene
    from witwin.channel.scene.endpoints import SolverScene as Scene
    from witwin.channel.scene.resources import RayDSceneResource


# Component ids of single-diffraction chains: the diffraction last (reflections then the
# diffraction), first (the diffraction then reflections), or at slot s >= 1 with reflections on
# both sides (CHAIN_INTERIOR_COMPONENT_BASE + s). Double diffraction is its own component.
CHAIN_DIFFRACTION_LAST_COMPONENT_ID = 3
CHAIN_DIFFRACTION_FIRST_COMPONENT_ID = 4
DOUBLE_DIFFRACTION_COMPONENT_ID = 7
CHAIN_INTERIOR_COMPONENT_BASE = 8
# Deepest chain the field kernels evaluate (the EPC bounce limit).
CHAIN_MAX_DEPTH = 8

# Join lanes (edge-sharing half pairs) and node-edge beam tests per native launch.
_CHAIN_LANES_PER_LAUNCH = 1 << 23
_NODE_EDGE_TESTS_PER_LAUNCH = 1 << 24
# Image-tree nodes of the receiver halves of one receiver chunk (bounds their edge lists).
_RECEIVER_TREE_NODES = 1 << 22
# Double-diffraction candidates per geometry launch.
_DOUBLE_CANDIDATES_PER_LAUNCH = 1 << 20
# Endpoint-edge visibility states per launch (the native state capacity).
_VISIBILITY_STATES_PER_LAUNCH = 1 << 22
# The finite-edge interior tolerance of the coupled stationary points, metres.
_EDGE_EPSILON = 1.0e-6
# Pruning tolerances, as the reflection candidate tests use them (relative to the scene diagonal).
_BOUNDS_RELATIVE = 1.0e-4
_SEGMENT_PARAMETER_SLACK = 1.0e-3


def chain_depth_limit(max_depth: int) -> int:
    """Deepest chain (interactions) of a solve: ``max_depth`` reflections and the diffraction."""

    return min(int(max_depth) + 1, CHAIN_MAX_DEPTH)


def chain_component_id(prefix_depth: int, suffix_depth: int) -> int:
    """Component id of a chain with ``prefix_depth`` reflections before its diffraction."""

    if prefix_depth == 0:
        return CHAIN_DIFFRACTION_FIRST_COMPONENT_ID
    if suffix_depth == 0:
        return CHAIN_DIFFRACTION_LAST_COMPONENT_ID
    return CHAIN_INTERIOR_COMPONENT_BASE + prefix_depth


# --------------------------------------------------------------------------
# Geometry: typed chain and double-diffraction geometry queries.
# --------------------------------------------------------------------------


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
    # Device int32[1]: candidates dropped for an uncertified stationary point.
    certificate_rejection_count: torch.Tensor


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
        certificate_rejection_count=raw["certificate_rejection_count"],
    )


# --------------------------------------------------------------------------
# Discovery: chain halves, the edge join, and the double-diffraction products.
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _ChainContext:
    """Receiver-independent inputs of one chain discovery."""

    device: torch.device
    rayd: RayDSceneResource
    plane_groups: ReflectionPlaneGroups
    coplanar_groups: dict[str, torch.Tensor]
    face_material_id: torch.Tensor
    edges: tuple[DiffractionEdgeGeometry, torch.Tensor]
    selected_edges: torch.Tensor
    edge_position: torch.Tensor
    edge_direction: torch.Tensor
    edge_t_min: torch.Tensor
    edge_t_max: torch.Tensor
    edge_n0: torch.Tensor
    edge_n1: torch.Tensor
    min_segment_length: float

    @property
    def edge_count(self) -> int:
        return int(self.selected_edges.shape[0])


@dataclass(frozen=True, slots=True)
class _ChainHalf:
    """Per-edge lists (CSR over the selected edges) of one side of a chain.

    An entry is an endpoint index and, for a half with reflections, its image-tree node; the
    node tables hold the node's plane-group sequence from the endpoint and its image.
    """

    edge_offsets: torch.Tensor
    endpoint: torch.Tensor
    node: torch.Tensor
    endpoints: torch.Tensor
    node_sequences: torch.Tensor
    node_images: torch.Tensor

    @property
    def depth(self) -> int:
        return int(self.node_sequences.shape[1])

    def counts(self) -> torch.Tensor:
        return self.edge_offsets[1:] - self.edge_offsets[:-1]


def _prepare_chain_context(
    scene: Scene, compiled: CompiledScene, tx_positions: torch.Tensor, rx_positions: torch.Tensor,
) -> _ChainContext | None:
    """The receiver-independent chain context, or ``None`` when no chain can exist."""

    device = tx_positions.device
    rayd = compiled.rayd
    if not scene.structures or tx_positions.numel() == 0 or rx_positions.numel() == 0:
        return None
    if not rayd.available:
        raise RuntimeError("coupled topology requires RayD native scene capability")
    faces, normals, tri_a = _reflection_geometry_tables(compiled, device)
    if int(faces.shape[0]) == 0:
        return None
    coplanar_groups = _cached_coplanar_face_groups(
        rayd, tri_a, normals, face_geometry_id_table(compiled, "face_surface_id", device=device)
    )
    plane_groups = reflection_plane_groups(compiled, coplanar_groups, tri_a=tri_a, normals=normals)
    plan = prepare_diffraction_order1_plan(
        metadata=scene.metadata, tx_count=int(tx_positions.shape[0]), rx_count=int(rx_positions.shape[0])
    )
    edges = selected_diffraction_edges(rayd, preserve_imported_edges=plan.preserve_imported_edges)
    geometry, selected = edges
    if plane_groups.group_count <= 0 or int(selected.shape[0]) == 0:
        return None
    index = selected.to(dtype=torch.int64)
    return _ChainContext(
        device=device,
        rayd=rayd,
        plane_groups=plane_groups,
        coplanar_groups=coplanar_groups,
        face_material_id=face_material_id_table(compiled, device=device),
        edges=edges,
        selected_edges=selected.to(dtype=torch.int32).contiguous(),
        edge_position=geometry.edge_position[index].contiguous(),
        edge_direction=geometry.edge_direction[index].contiguous(),
        edge_t_min=geometry.line_min[index].contiguous(),
        edge_t_max=geometry.line_max[index].contiguous(),
        edge_n0=geometry.n0[index].contiguous(),
        edge_n1=geometry.n1[index].contiguous(),
        min_segment_length=reflection_min_segment_length(plane_groups),
    )


def _visible_edges(context: _ChainContext, endpoints: torch.Tensor) -> torch.Tensor:
    """``[P, E]`` bool: each endpoint sees some of the four axial samples of each selected edge.

    The same visibility rule selects the order-1 diffraction states, and it is symmetric in the
    two endpoints of a chain, so a chain and its reverse see the same edge sets.
    """

    edge_count = context.edge_count
    per_launch = max(1, _VISIBILITY_STATES_PER_LAUNCH // edge_count)
    power = torch.ones((int(endpoints.shape[0]),), device=context.device, dtype=torch.float32)
    parts: list[torch.Tensor] = []
    for start in range(0, int(endpoints.shape[0]), per_launch):
        chunk = endpoints[start : start + per_launch].contiguous()
        states, _source = _batched_diffraction_states(context.edges, chunk, power[: chunk.shape[0]])
        plan = plan_tx_visible_diffraction_states(context.rayd, states, chunk[0], per_state_source=True)
        parts.append(plan.active.reshape(int(chunk.shape[0]), edge_count))
    return parts[0] if len(parts) == 1 else torch.cat(parts)


def _edge_offsets(edge: torch.Tensor, edge_count: int) -> torch.Tensor:
    counts = torch.bincount(edge.to(dtype=torch.int64), minlength=edge_count)
    offsets = torch.zeros((edge_count + 1,), device=edge.device, dtype=torch.int64)
    torch.cumsum(counts, dim=0, out=offsets[1:])
    return offsets


def _direct_half(edges: torch.Tensor, endpoints: torch.Tensor) -> _ChainHalf:
    """The half without reflections: every ``(edge, endpoint)`` pair marked in ``edges``, edge-major."""

    pairs = torch.nonzero(edges.t(), as_tuple=False)
    edge = pairs[:, 0]
    device = endpoints.device
    return _ChainHalf(
        edge_offsets=_edge_offsets(edge, int(edges.shape[1])),
        endpoint=pairs[:, 1].to(dtype=torch.int32).contiguous(),
        node=torch.full((int(pairs.shape[0]),), -1, device=device, dtype=torch.int32),
        endpoints=endpoints,
        node_sequences=torch.empty((0, 0), device=device, dtype=torch.int32),
        node_images=torch.empty((0, 3), device=device, dtype=torch.float32),
    )


def _reflected_half(
    context: _ChainContext, frontier: _ImageTreeFrontier, endpoints: torch.Tensor,
) -> _ChainHalf:
    """The half whose last reflection is an image-tree node: every (node, edge in its beam)."""

    groups = context.plane_groups
    node_count = int(frontier.size)
    edge_count = context.edge_count
    nodes_per_launch = max(1, _NODE_EDGE_TESTS_PER_LAUNCH // edge_count)
    node_parts: list[torch.Tensor] = []
    edge_parts: list[torch.Tensor] = []
    for start in range(0, node_count, nodes_per_launch):
        node, edge = topology_kernels.chain_node_edges(
            node_sequences=frontier.node_sequences,
            node_images=frontier.node_images,
            plane_point=groups.plane_point,
            plane_normal=groups.plane_normal,
            bounds=groups.bounds,
            occupancy=groups.occupancy,
            edge_position=context.edge_position,
            edge_direction=context.edge_direction,
            edge_t_min=context.edge_t_min,
            edge_t_max=context.edge_t_max,
            node_start=start,
            node_end=min(start + nodes_per_launch, node_count),
            tolerance=_BOUNDS_RELATIVE * groups.scene_scale_m,
        )
        node_parts.append(node)
        edge_parts.append(edge)
    device = context.device
    node = torch.cat(node_parts) if node_parts else torch.empty((0,), device=device, dtype=torch.int32)
    edge = torch.cat(edge_parts) if edge_parts else torch.empty((0,), device=device, dtype=torch.int32)
    order = torch.argsort(edge, stable=True)
    node = node[order].contiguous()
    return _ChainHalf(
        edge_offsets=_edge_offsets(edge, edge_count),
        endpoint=frontier.node_tx[node.to(dtype=torch.int64)].contiguous(),
        node=node,
        endpoints=endpoints,
        node_sequences=frontier.node_sequences.contiguous(),
        node_images=frontier.node_images.contiguous(),
    )


class _HalfSource:
    """Chain halves of one endpoint set, by number of reflections, built on first use."""

    def __init__(self, context: _ChainContext, endpoints: torch.Tensor) -> None:
        self._context = context
        self._endpoints = endpoints.to(dtype=torch.float32).contiguous()
        self._halves: dict[int, _ChainHalf] = {}
        self._frontier = _image_tree_roots(self._endpoints)
        self._frontier_depth = 0
        self.visible: torch.Tensor | None = None

    def visible_edges(self) -> torch.Tensor:
        if self.visible is None:
            self.visible = _visible_edges(self._context, self._endpoints)
        return self.visible

    def half(self, depth: int) -> _ChainHalf:
        if depth not in self._halves:
            if depth == 0:
                # Every edge: the validator's visibility segment decides, so a chain half never
                # depends on a sampled visibility estimate.
                every = torch.ones(
                    (int(self._endpoints.shape[0]), self._context.edge_count),
                    device=self._endpoints.device, dtype=torch.bool,
                )
                self._halves[0] = _direct_half(every, self._endpoints)
            else:
                while self._frontier_depth < depth:
                    self._frontier = _expand_image_tree(
                        self._frontier, self._context.plane_groups, prune=True
                    )
                    self._frontier_depth += 1
                self._halves[depth] = _reflected_half(self._context, self._frontier, self._endpoints)
        return self._halves[depth]


def _lane_offsets(a: _ChainHalf, b: _ChainHalf) -> tuple[torch.Tensor, int]:
    """Per-edge offsets of the joined (a entry, b entry) lanes and their total (one host read)."""

    lanes = a.counts() * b.counts()
    offsets = torch.zeros((int(lanes.shape[0]) + 1,), device=lanes.device, dtype=torch.int64)
    torch.cumsum(lanes, dim=0, out=offsets[1:])
    return offsets, int(offsets[-1].item())


def _iter_chain_candidates(
    context: _ChainContext, a: _ChainHalf, b: _ChainHalf,
) -> Iterator[dict[str, torch.Tensor]]:
    """Chain candidates (transmitter, receiver, edge, half nodes, diffraction point) of one class."""

    offsets, total = _lane_offsets(a, b)
    groups = context.plane_groups
    for start in range(0, total, _CHAIN_LANES_PER_LAUNCH):
        candidates = topology_kernels.chain_candidates(
            lane_offsets=offsets,
            lane_start=start,
            lane_end=min(start + _CHAIN_LANES_PER_LAUNCH, total),
            plane_point=groups.plane_point,
            plane_normal=groups.plane_normal,
            bounds=groups.bounds,
            occupancy=groups.occupancy,
            edge_position=context.edge_position,
            edge_direction=context.edge_direction,
            edge_t_min=context.edge_t_min,
            edge_t_max=context.edge_t_max,
            edge_n0=context.edge_n0,
            edge_n1=context.edge_n1,
            a=(a.edge_offsets, a.endpoint, a.node, a.endpoints, a.node_sequences, a.node_images),
            b=(b.edge_offsets, b.endpoint, b.node, b.endpoints, b.node_sequences, b.node_images),
            box_tolerance=_BOUNDS_RELATIVE * groups.scene_scale_m,
            segment_tolerance=_SEGMENT_PARAMETER_SLACK,
            min_segment_length=0.5 * context.min_segment_length,
            edge_epsilon=_EDGE_EPSILON,
        )
        if int(candidates["tx"].shape[0]) > 0:
            yield candidates


def _half_planes(
    context: _ChainContext, half: _ChainHalf, node: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Representative faces and plane tables of each candidate's half, in endpoint order."""

    count = int(node.shape[0])
    depth = half.depth
    device = context.device
    if depth == 0:
        return (
            torch.empty((count, 0), device=device, dtype=torch.int32),
            torch.empty((count, 0, 3), device=device, dtype=torch.float32),
            torch.empty((count, 0, 3), device=device, dtype=torch.float32),
        )
    groups = context.plane_groups
    sequence = half.node_sequences[node.to(dtype=torch.int64)].to(dtype=torch.int64)
    faces = groups.representative_faces[sequence].to(dtype=torch.int32).contiguous()
    return faces, groups.plane_point[sequence].contiguous(), groups.plane_normal[sequence].contiguous()


def _chain_block(
    context: _ChainContext, exported: dict[str, torch.Tensor], tx: torch.Tensor, rx: torch.Tensor,
    edge_id: torch.Tensor, prefix_depth: int, suffix_depth: int,
) -> dict[str, torch.Tensor] | None:
    kept = torch.nonzero(exported["valid"], as_tuple=False).reshape(-1)
    count = int(kept.shape[0])
    if count == 0:
        return None
    device = context.device
    depth = prefix_depth + suffix_depth + 1
    sequence = exported["primitive_sequence"][kept].contiguous()
    positions = exported["interaction_positions"][kept].contiguous()
    normals = exported["interaction_normals"][kept].contiguous()
    reflection = torch.ones((depth,), device=device, dtype=torch.bool)
    reflection[prefix_depth] = False
    faces = torch.where(reflection, sequence, torch.zeros_like(sequence)).to(dtype=torch.int64)
    material_sequence = torch.where(
        reflection, context.face_material_id[faces], torch.full_like(sequence, -1)
    ).to(dtype=torch.int32).contiguous()
    first_reflection = 0 if prefix_depth > 0 else 1
    primitive_id = sequence[:, first_reflection].contiguous()
    nan = torch.full((count,), float("nan"), device=device, dtype=torch.float32)
    return _ensure_topology_fields(
        {
            "valid": torch.ones((count,), device=device, dtype=torch.bool),
            "tx_id": tx[kept].to(dtype=torch.int32).contiguous(),
            "rx_id": rx[kept].to(dtype=torch.int32).contiguous(),
            "depth": torch.full((count,), depth, device=device, dtype=torch.int32),
            "component_id": torch.full(
                (count,), chain_component_id(prefix_depth, suffix_depth), device=device,
                dtype=torch.int32,
            ),
            "primitive_id": primitive_id,
            "edge_id": edge_id[kept].to(dtype=torch.int32).contiguous(),
            "path_length_m": exported["path_length_m"][kept],
            "delay_s": exported["delay_s"][kept],
            "path_gain": nan,
        },
        interaction_position=positions[:, 0].contiguous(),
        interaction_normal=normals[:, 0].contiguous(),
        material_id=material_sequence[:, first_reflection].contiguous(),
        path_field=torch.complex(nan, nan),
        primitive_sequence=sequence,
        material_sequence=material_sequence,
        interaction_positions=positions,
        interaction_normals=normals,
    )


def _chain_class_blocks(
    context: _ChainContext, a: _ChainHalf, b: _ChainHalf, tx_positions: torch.Tensor,
    rx_positions: torch.Tensor, validator_chunk: int,
) -> tuple[list[dict[str, torch.Tensor]], int, int]:
    """Validated rows of one chain class (``a.depth`` reflections, the diffraction, ``b.depth``)."""

    blocks: list[dict[str, torch.Tensor]] = []
    launches = 0
    candidate_count = 0
    groups = context.coplanar_groups
    resource = context.rayd.require_resource()
    for candidates in _iter_chain_candidates(context, a, b):
        launches += 1
        total = int(candidates["tx"].shape[0])
        candidate_count += total
        for start in range(0, total, validator_chunk):
            end = min(start + validator_chunk, total)
            tx = candidates["tx"][start:end]
            rx = candidates["rx"][start:end]
            edge_id = context.selected_edges[candidates["edge"][start:end].to(dtype=torch.int64)]
            prefix = _half_planes(context, a, candidates["a_node"][start:end])
            suffix = _half_planes(context, b, candidates["b_node"][start:end])
            exported = geometry_kernels.coupled_chain_geometry_forward(
                resource,
                tx_positions[tx.to(dtype=torch.int64)].contiguous(),
                rx_positions[rx.to(dtype=torch.int64)].contiguous(),
                candidates["point"][start:end].contiguous(),
                edge_id.contiguous(),
                *prefix,
                *suffix,
                groups["surface_group_id"].to(dtype=torch.int32).contiguous(),
                groups["surface_group_size"].to(dtype=torch.int32).contiguous(),
                groups["surface_group_members"].to(dtype=torch.int32).contiguous(),
                context.min_segment_length,
            )
            launches += 1
            block = _chain_block(context, exported, tx, rx, edge_id, a.depth, b.depth)
            if block is not None:
                blocks.append(block)
    return blocks, launches, candidate_count


def _endpoint_edge_lists(visible: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """CSR over endpoints of the edges each endpoint sees (ascending edge index)."""

    pairs = torch.nonzero(visible, as_tuple=False)
    counts = torch.bincount(pairs[:, 0], minlength=int(visible.shape[0]))
    offsets = torch.zeros((int(visible.shape[0]) + 1,), device=visible.device, dtype=torch.int64)
    torch.cumsum(counts, dim=0, out=offsets[1:])
    return offsets, pairs[:, 1].to(dtype=torch.int32).contiguous()


def _double_diffraction_blocks(
    context: _ChainContext, tx_source: _HalfSource, rx_source: _HalfSource,
    tx_positions: torch.Tensor, rx_positions: torch.Tensor,
) -> tuple[list[dict[str, torch.Tensor]], int, int]:
    """Double-diffraction rows: the first edge seen from the transmitter, the second from the receiver."""

    tx_offsets, tx_edges = _endpoint_edge_lists(tx_source.visible_edges())
    rx_offsets, rx_edges = _endpoint_edge_lists(rx_source.visible_edges())
    tx_count = int(tx_positions.shape[0])
    rx_count = int(rx_positions.shape[0])
    pair_lanes = (
        (tx_offsets[1:] - tx_offsets[:-1]).reshape(-1, 1) * (rx_offsets[1:] - rx_offsets[:-1]).reshape(1, -1)
    ).reshape(-1)
    pair_offsets = torch.zeros((tx_count * rx_count + 1,), device=context.device, dtype=torch.int64)
    torch.cumsum(pair_lanes, dim=0, out=pair_offsets[1:])
    total = int(pair_offsets[-1].item())
    blocks: list[dict[str, torch.Tensor]] = []
    launches = 0
    resource = context.rayd.require_resource()
    device = context.device
    for start in range(0, total, _DOUBLE_CANDIDATES_PER_LAUNCH):
        expanded = topology_kernels.chain_double_candidates(
            pair_offsets=pair_offsets,
            lane_start=start,
            lane_end=min(start + _DOUBLE_CANDIDATES_PER_LAUNCH, total),
            rx_count=rx_count,
            tx_offsets=tx_offsets,
            tx_edges=tx_edges,
            rx_offsets=rx_offsets,
            rx_edges=rx_edges,
        )
        tx = expanded["tx"].to(dtype=torch.int64)
        rx = expanded["rx"].to(dtype=torch.int64)
        first = expanded["edge1"].to(dtype=torch.int64)
        second = expanded["edge2"].to(dtype=torch.int64)
        exported = query_coupled_dd_geometry(
            CoupledDdGeometryQuery(
                rayd_resource=resource,
                source=tx_positions[tx].contiguous(),
                receiver=rx_positions[rx].contiguous(),
                edge1_id=context.selected_edges[first].contiguous(),
                edge1_position=context.edge_position[first].contiguous(),
                edge1_direction=context.edge_direction[first].contiguous(),
                edge1_t_min=context.edge_t_min[first].contiguous(),
                edge1_t_max=context.edge_t_max[first].contiguous(),
                edge2_id=context.selected_edges[second].contiguous(),
                edge2_position=context.edge_position[second].contiguous(),
                edge2_direction=context.edge_direction[second].contiguous(),
                edge2_t_min=context.edge_t_min[second].contiguous(),
                edge2_t_max=context.edge_t_max[second].contiguous(),
            )
        )
        launches += 2
        kept = torch.nonzero(exported.valid, as_tuple=False).reshape(-1)
        kept_count = int(kept.shape[0])
        if kept_count == 0:
            continue
        # Both events are diffractions: the object sequence holds the two edge ids and no face
        # is touched; the field stage resolves the wedge materials.
        nan = torch.full((kept_count,), float("nan"), device=device, dtype=torch.float32)
        blocks.append(
            _ensure_topology_fields(
                {
                    "valid": torch.ones((kept_count,), device=device, dtype=torch.bool),
                    "tx_id": expanded["tx"][kept].contiguous(),
                    "rx_id": expanded["rx"][kept].contiguous(),
                    "depth": torch.full((kept_count,), 2, device=device, dtype=torch.int32),
                    "component_id": torch.full(
                        (kept_count,), DOUBLE_DIFFRACTION_COMPONENT_ID, device=device, dtype=torch.int32
                    ),
                    "primitive_id": torch.full((kept_count,), -1, device=device, dtype=torch.int32),
                    "edge_id": exported.edge1_id[kept].to(dtype=torch.int32),
                    "path_length_m": exported.path_length_m[kept],
                    "delay_s": exported.delay_s[kept],
                    "path_gain": nan,
                },
                interaction_position=exported.interaction_positions[kept, 0],
                interaction_normal=exported.interaction_normals[kept, 0],
                material_id=torch.full((kept_count,), -1, device=device, dtype=torch.int32),
                path_field=torch.complex(nan, nan),
                primitive_sequence=exported.edge_sequence[kept].to(dtype=torch.int32).contiguous(),
                material_sequence=torch.full((kept_count, 2), -1, device=device, dtype=torch.int32),
                interaction_positions=exported.interaction_positions[kept],
                interaction_normals=exported.interaction_normals[kept],
            )
        )
    return blocks, launches, total


def coupled_reflection_diffraction_topology(
    scene: Scene, compiled: CompiledScene, tx_positions: torch.Tensor, rx_positions: torch.Tensor,
    *, max_depth: int, candidate_limit: int,
) -> tuple[dict[str, torch.Tensor], int, int]:
    """Interaction-chain rows with up to ``max_depth`` reflections, and double-diffraction rows.

    A chain is ``a`` reflections, one diffraction and ``b`` reflections (``1 <= a + b <=
    max_depth``, at most ``CHAIN_MAX_DEPTH`` interactions in all): the reflection paths go up to
    ``max_depth`` reflections, and the chains with as many reflections carry the diffracted field
    that keeps such a path continuous where another object blocks one of its legs. Its candidates
    join a transmitter-side half and a receiver-side half on a shared edge: a half is the endpoint
    itself (any edge) or an image-tree node whose beam may reach the edge (a conservative geometric
    test). Double diffraction takes its first edge from the transmitter-visible and its second from
    the receiver-visible edge sets. The diffraction point is the Keller point between the two half
    images, and the validator back-traces both halves with the reflection EPC (a half without
    reflection is one visibility segment), the receiver half only where the transmitter half is
    valid. ``candidate_limit`` bounds the candidates of one validator launch; it never caps the
    result. Returns the rows, the launch count and the candidate count.
    """

    device = tx_positions.device
    context = _prepare_chain_context(scene, compiled, tx_positions, rx_positions)
    if context is None:
        return _ensure_topology_fields(_empty_path_block(device)), 0, 0
    max_chain_depth = chain_depth_limit(max_depth)
    tx_source = _HalfSource(context, tx_positions)
    blocks: list[dict[str, torch.Tensor]] = []
    launches = 0
    candidates = 0
    # Receivers are taken in chunks whose image trees stay bounded: a receiver's tree holds up
    # to group_count ** (max_chain_depth - 1) nodes at its deepest chain half.
    tree_nodes = max(1, context.plane_groups.group_count) ** max(0, max_chain_depth - 1)
    rx_count = int(rx_positions.shape[0])
    rx_chunk = max(1, _RECEIVER_TREE_NODES // tree_nodes)
    for rx_start in range(0, rx_count, rx_chunk):
        chunk = rx_positions[rx_start : rx_start + rx_chunk].contiguous()
        rx_source = _HalfSource(context, chunk)
        chunk_blocks: list[dict[str, torch.Tensor]] = []
        for reflections in range(1, max_chain_depth):
            for prefix_depth in range(reflections, -1, -1):
                class_blocks, class_launches, class_candidates = _chain_class_blocks(
                    context,
                    tx_source.half(prefix_depth),
                    rx_source.half(reflections - prefix_depth),
                    tx_positions,
                    chunk,
                    int(candidate_limit),
                )
                chunk_blocks.extend(class_blocks)
                launches += class_launches
                candidates += class_candidates
        dd_blocks, dd_launches, dd_candidates = _double_diffraction_blocks(
            context, tx_source, rx_source, tx_positions, chunk
        )
        chunk_blocks.extend(dd_blocks)
        launches += dd_launches
        candidates += dd_candidates
        for block in chunk_blocks:
            if rx_start > 0:
                block["rx_id"] = block["rx_id"] + rx_start
            blocks.append(block)
    return (
        _ensure_topology_fields(concatenate_path_blocks(blocks, device=device)),
        launches,
        candidates,
    )
