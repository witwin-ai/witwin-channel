"""Reflection: discovery planning, EPC geometry, and enumerated orchestration.

One concept, one file. This module holds the reflection discovery limits and
plan iterators, the typed reflection endpoint-connection geometry query, and
the enumerated first-order and multibounce reflection topology owners that
compose them. It calls the native facades in ``witwin.channel.kernels`` and
owns no physics of its own.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Protocol, TYPE_CHECKING

import torch

from witwin.channel.materials import face_material_tensors
from witwin.channel.kernels import geometry as geometry_kernels
from witwin.channel.kernels import topology as topology_kernels
from witwin.channel.kernels.topology import (
    mc_sample_directions,
)
from witwin.channel.propagation.geometry import (
    _cached_coplanar_face_groups,
)
from witwin.channel.propagation.topology import (
    concatenate_path_blocks,
)
from witwin.channel.propagation.topology import _ensure_topology_fields

if TYPE_CHECKING:
    from witwin.channel.scene.endpoints import SolverScene as Scene

_ORDER1_EXHAUSTIVE_GROUP_LIMIT = 4096
_MAX_MULTIBOUNCE_FACE_SEQUENCES = 100_000
_MULTIBOUNCE_SEQUENCE_CHUNK_SIZE = 65_536
_MULTIBOUNCE_PAIR_CHUNK_SIZE = 4_194_304
_MULTIBOUNCE_DISCOVERY_RAYS = 262_144


def _face_sequence_count(
    face_count: int, depth: int, *, adjacent_distinct: bool
) -> int:
    if adjacent_distinct and depth > 1:
        if face_count <= 1:
            return 0
        return int(face_count) * int(face_count - 1) ** int(depth - 1)
    return int(face_count) ** int(depth)


def _face_sequence_chunks(
    face_count: int,
    depth: int,
    *,
    chunk_size: int,
    reference: torch.Tensor,
    face_ids: torch.Tensor | None = None,
    adjacent_distinct: bool = False,
) -> object:
    total = _face_sequence_count(face_count, depth, adjacent_distinct=adjacent_distinct)
    for start in range(0, total, chunk_size):
        end = min(start + chunk_size, total)
        if face_ids is None:
            sequences = topology_kernels.deterministic_face_sequence_chunk(
                reference,
                face_count=face_count,
                depth=depth,
                start=start,
                end=end,
                adjacent_distinct=adjacent_distinct,
            )
        else:
            sequences = topology_kernels.deterministic_mapped_face_sequence_chunk(
                face_ids,
                depth=depth,
                start=start,
                end=end,
                adjacent_distinct=adjacent_distinct,
            )
        if int(sequences.shape[0]) > 0:
            yield sequences


class TraceReflectionGroupChains(Protocol):
    def __call__(
        self,
        tx: torch.Tensor,
        *,
        face_group_id: torch.Tensor,
        max_depth: int,
    ) -> torch.Tensor: ...


class RecordReflectionCandidateCount(Protocol):
    def __call__(self, candidate_count: int) -> None: ...


@dataclass(frozen=True, slots=True)
class ReflectionOrder1Plan:
    exhaustive: bool
    group_count: int
    representative_faces: torch.Tensor
    base_sequences: torch.Tensor | None
    face_group_id: torch.Tensor | None


@dataclass(frozen=True, slots=True)
class ReflectionOrder1EpcRequest:
    tx_index: int
    tx: torch.Tensor
    epc_inputs: dict[str, torch.Tensor]


@dataclass(frozen=True, slots=True)
class ReflectionMultibouncePlan:
    exhaustive: bool
    group_count: int
    representative_faces: torch.Tensor
    face_group_id: torch.Tensor | None
    min_depth: int
    max_depth: int


@dataclass(frozen=True, slots=True)
class ReflectionMultibounceEpcRequest:
    depth: int
    tx_index: int
    tx: torch.Tensor
    epc_inputs: dict[str, torch.Tensor]


def prepare_reflection_order1_plan(
    *,
    group_count: int,
    representative_faces: torch.Tensor,
    face_group_id: torch.Tensor,
) -> ReflectionOrder1Plan:
    exhaustive = group_count <= _ORDER1_EXHAUSTIVE_GROUP_LIMIT
    base_sequences = (
        topology_kernels.deterministic_mapped_face_sequence_chunk(
            representative_faces,
            depth=1,
            start=0,
            end=group_count,
        )
        if exhaustive
        else None
    )
    selected_face_group_id = (
        None if exhaustive else face_group_id.to(dtype=torch.long).contiguous()
    )
    return ReflectionOrder1Plan(
        exhaustive=exhaustive,
        group_count=group_count,
        representative_faces=representative_faces,
        base_sequences=base_sequences,
        face_group_id=selected_face_group_id,
    )


def iter_reflection_order1_epc_requests(
    plan: ReflectionOrder1Plan,
    *,
    tx_positions: torch.Tensor,
    rx_positions: torch.Tensor,
    tri_a: torch.Tensor,
    normals: torch.Tensor,
    trace_group_chains: TraceReflectionGroupChains,
) -> Iterator[ReflectionOrder1EpcRequest]:
    rx_count = int(rx_positions.shape[0])
    if plan.group_count <= 0 or rx_count <= 0:
        return

    for tx_index, tx in enumerate(tx_positions):
        if plan.exhaustive:
            sequences = plan.base_sequences
        else:
            chains = trace_group_chains(
                tx, face_group_id=plan.face_group_id, max_depth=1
            )
            first_groups = torch.unique(chains[chains[:, 0] >= 0][:, 0])
            if int(first_groups.numel()) == 0:
                continue
            sequences = (
                plan.representative_faces[first_groups].reshape(-1, 1).contiguous()
            )
        sequence_count = int(sequences.shape[0])
        if sequence_count <= 0:
            continue
        rx_chunk_size = max(1, _MULTIBOUNCE_PAIR_CHUNK_SIZE // sequence_count)
        for rx_start in range(0, rx_count, rx_chunk_size):
            rx_end = min(rx_start + rx_chunk_size, rx_count)
            epc_inputs = topology_kernels.deterministic_reflection_epc_input_batch(
                tx=tx,
                rx_positions=rx_positions.contiguous(),
                sequences=sequences.contiguous(),
                tri_a=tri_a.contiguous(),
                normals=normals.contiguous(),
                rx_start=rx_start,
                rx_end=rx_end,
            )
            yield ReflectionOrder1EpcRequest(
                tx_index=tx_index,
                tx=tx,
                epc_inputs=epc_inputs,
            )


def prepare_reflection_multibounce_plan(
    *,
    group_count: int,
    representative_faces: torch.Tensor,
    face_group_id: torch.Tensor,
    min_depth: int,
    max_depth: int,
) -> ReflectionMultibouncePlan:
    exhaustive = all(
        _face_sequence_count(group_count, depth, adjacent_distinct=True)
        <= _MAX_MULTIBOUNCE_FACE_SEQUENCES
        for depth in range(min_depth, max_depth + 1)
    )
    selected_face_group_id = (
        None if exhaustive else face_group_id.to(dtype=torch.long).contiguous()
    )
    return ReflectionMultibouncePlan(
        exhaustive=exhaustive,
        group_count=group_count,
        representative_faces=representative_faces,
        face_group_id=selected_face_group_id,
        min_depth=min_depth,
        max_depth=max_depth,
    )


def iter_reflection_multibounce_epc_requests(
    plan: ReflectionMultibouncePlan,
    *,
    tx_positions: torch.Tensor,
    rx_positions: torch.Tensor,
    sequence_reference: torch.Tensor,
    tri_a: torch.Tensor,
    normals: torch.Tensor,
    trace_group_chains: TraceReflectionGroupChains,
    record_candidate_count: RecordReflectionCandidateCount,
) -> Iterator[ReflectionMultibounceEpcRequest]:
    rx_count = int(rx_positions.shape[0])
    if plan.exhaustive:
        for depth in range(plan.min_depth, plan.max_depth + 1):
            candidate_count = _face_sequence_count(
                plan.group_count, depth, adjacent_distinct=True
            )
            record_candidate_count(candidate_count)
            chunk_size = min(_MULTIBOUNCE_SEQUENCE_CHUNK_SIZE, max(candidate_count, 1))
            for sequences in _face_sequence_chunks(
                plan.group_count,
                depth,
                chunk_size=chunk_size,
                reference=sequence_reference,
                face_ids=plan.representative_faces,
                adjacent_distinct=True,
            ):
                sequence_count = int(sequences.shape[0])
                if sequence_count <= 0:
                    continue
                rx_chunk_size = max(1, _MULTIBOUNCE_PAIR_CHUNK_SIZE // sequence_count)
                for rx_start in range(0, rx_count, rx_chunk_size):
                    rx_end = min(rx_start + rx_chunk_size, rx_count)
                    for tx_index, tx in enumerate(tx_positions):
                        epc_inputs = topology_kernels.deterministic_reflection_epc_input_batch(
                            tx=tx,
                            rx_positions=rx_positions.contiguous(),
                            sequences=sequences.contiguous(),
                            tri_a=tri_a.contiguous(),
                            normals=normals.contiguous(),
                            rx_start=rx_start,
                            rx_end=rx_end,
                        )
                        yield ReflectionMultibounceEpcRequest(
                            depth=depth,
                            tx_index=tx_index,
                            tx=tx,
                            epc_inputs=epc_inputs,
                        )
    else:
        for tx_index, tx in enumerate(tx_positions):
            group_chains = trace_group_chains(
                tx,
                face_group_id=plan.face_group_id,
                max_depth=plan.max_depth,
            )
            for depth in range(plan.min_depth, plan.max_depth + 1):
                reached = group_chains[:, depth - 1] >= 0
                if not bool(reached.any()):
                    continue
                unique_chains = torch.unique(group_chains[reached][:, :depth], dim=0)
                record_candidate_count(int(unique_chains.shape[0]))
                sequences_all = plan.representative_faces[unique_chains].contiguous()
                for start in range(
                    0,
                    int(sequences_all.shape[0]),
                    _MULTIBOUNCE_SEQUENCE_CHUNK_SIZE,
                ):
                    sequences = sequences_all[
                        start : start + _MULTIBOUNCE_SEQUENCE_CHUNK_SIZE
                    ].contiguous()
                    rx_chunk_size = max(
                        1, _MULTIBOUNCE_PAIR_CHUNK_SIZE // int(sequences.shape[0])
                    )
                    for rx_start in range(0, rx_count, rx_chunk_size):
                        rx_end = min(rx_start + rx_chunk_size, rx_count)
                        epc_inputs = topology_kernels.deterministic_reflection_epc_input_batch(
                            tx=tx,
                            rx_positions=rx_positions.contiguous(),
                            sequences=sequences.contiguous(),
                            tri_a=tri_a.contiguous(),
                            normals=normals.contiguous(),
                            rx_start=rx_start,
                            rx_end=rx_end,
                        )
                        yield ReflectionMultibounceEpcRequest(
                            depth=depth,
                            tx_index=tx_index,
                            tx=tx,
                            epc_inputs=epc_inputs,
                        )


@dataclass(frozen=True, slots=True)
class ReflectionEpcQuery:
    rayd: object
    source: torch.Tensor
    receiver: torch.Tensor
    active: torch.Tensor | None
    expected_prim_ids: torch.Tensor
    direct_plane_points: torch.Tensor
    direct_plane_normals: torch.Tensor
    surface_group_id: torch.Tensor
    surface_group_size: torch.Tensor
    surface_group_members: torch.Tensor
    max_bounces: int
    visibility_ignore_mode: int


@dataclass(frozen=True, slots=True)
class ReflectionEpcGeometry:
    visible: torch.Tensor
    path_length_m: torch.Tensor
    resolved_prim_ids: torch.Tensor
    surface_group_ids: torch.Tensor
    hit_positions: torch.Tensor
    normals: torch.Tensor


def query_reflection_epc(query: ReflectionEpcQuery) -> ReflectionEpcGeometry:
    raw = geometry_kernels.rayd_reflection_epc_paths_forward(
        query.rayd.require_resource(),
        query.source,
        query.receiver,
        query.active,
        query.expected_prim_ids,
        query.direct_plane_points,
        query.direct_plane_normals,
        query.surface_group_id,
        query.surface_group_size,
        query.surface_group_members,
        query.max_bounces,
        query.visibility_ignore_mode,
    )
    return ReflectionEpcGeometry(
        visible=raw[0],
        path_length_m=raw[1],
        resolved_prim_ids=raw[2],
        surface_group_ids=raw[3],
        hit_positions=raw[4],
        normals=raw[5],
    )


def _reflection_topology_order1(
    scene: Scene,
    compiled: object,
    tx_positions: torch.Tensor,
    tx_power: torch.Tensor,
    rx_positions: torch.Tensor,
    *,
    frequency_hz: float,
) -> tuple[dict[str, torch.Tensor], int]:
    from witwin.channel.kernels import fields as field_kernels

    device = tx_positions.device
    rayd = compiled.rayd
    if not scene.structures or tx_positions.numel() == 0 or rx_positions.numel() == 0:
        return _ensure_topology_fields(
            {
                "valid": torch.empty((0,), device=device, dtype=torch.bool),
                "tx_id": torch.empty((0,), device=device, dtype=torch.int32),
                "rx_id": torch.empty((0,), device=device, dtype=torch.int32),
                "depth": torch.empty((0,), device=device, dtype=torch.int32),
                "component_id": torch.empty((0,), device=device, dtype=torch.int32),
                "primitive_id": torch.empty((0,), device=device, dtype=torch.int32),
                "edge_id": torch.empty((0,), device=device, dtype=torch.int32),
                "path_length_m": torch.empty((0,), device=device, dtype=torch.float32),
                "delay_s": torch.empty((0,), device=device, dtype=torch.float32),
                "path_gain": torch.empty((0,), device=device, dtype=torch.float32),
            }
        ), 0
    if not rayd.available:
        raise RuntimeError(
            "deterministic reflection requires RayD native scene capability"
        )

    records = rayd.edge_records()
    vertices = records.vertices
    faces = records.faces.contiguous()
    normals = geometry_kernels.deterministic_normalize_vec3(
        records.face_normals.contiguous(), eps=1.0e-6
    )
    if faces.shape[0] == 0:
        return _ensure_topology_fields(
            {
                "valid": torch.empty((0,), device=device, dtype=torch.bool),
                "tx_id": torch.empty((0,), device=device, dtype=torch.int32),
                "rx_id": torch.empty((0,), device=device, dtype=torch.int32),
                "depth": torch.empty((0,), device=device, dtype=torch.int32),
                "component_id": torch.empty((0,), device=device, dtype=torch.int32),
                "primitive_id": torch.empty((0,), device=device, dtype=torch.int32),
                "edge_id": torch.empty((0,), device=device, dtype=torch.int32),
                "path_length_m": torch.empty((0,), device=device, dtype=torch.float32),
                "delay_s": torch.empty((0,), device=device, dtype=torch.float32),
                "path_gain": torch.empty((0,), device=device, dtype=torch.float32),
            }
        ), 0

    tri_a = topology_kernels.deterministic_face_anchor_points(
        vertices.contiguous(), faces
    )
    face_eps_r, face_sigma_e, face_mu_r, face_gain, _face_valid = face_material_tensors(
        compiled, device=device
    )
    face_material_id = compiled.assignments.face_material_id.to(
        device=device, dtype=torch.int32
    ).contiguous()
    empty_i32 = torch.empty((0,), device=device, dtype=torch.int32)

    # Enumerate one candidate per coplanar face group so that a wall meshed
    # from several coplanar triangles yields exactly one specular path, and
    # every planar facade (not one representative per structure) is covered.
    # The EPC kernel resolves the actual containing triangle per path.
    groups = _cached_coplanar_face_groups(
        rayd,
        tri_a,
        normals,
        compiled.geometry.face_surface_id.to(
            device=device, dtype=torch.long
        ).contiguous(),
    )
    grouped_export = True
    group_count = int(groups["group_count"])
    representative_faces = groups["representative_faces"].contiguous()
    plan = prepare_reflection_order1_plan(
        group_count=group_count,
        representative_faces=representative_faces,
        face_group_id=groups["face_group_id"],
    )

    blocks: list[dict[str, torch.Tensor]] = []
    launch_count = 0
    rx_count = int(rx_positions.shape[0])
    if group_count <= 0 or rx_count <= 0:
        return _ensure_topology_fields(
            concatenate_path_blocks(blocks, device=device)
        ), launch_count

    def trace_group_chains(
        tx: torch.Tensor, *, face_group_id: torch.Tensor, max_depth: int
    ) -> torch.Tensor:
        nonlocal launch_count
        chains = _discovered_group_chains(
            rayd, tx, face_group_id=face_group_id, max_depth=max_depth
        )
        launch_count += 1
        return chains

    for request in iter_reflection_order1_epc_requests(
        plan,
        tx_positions=tx_positions,
        rx_positions=rx_positions,
        tri_a=tri_a,
        normals=normals,
        trace_group_chains=trace_group_chains,
    ):
        tx_index = request.tx_index
        tx = request.tx
        epc_inputs = request.epc_inputs
        epc = query_reflection_epc(
            ReflectionEpcQuery(
                rayd=rayd,
                source=epc_inputs["tx_batch"],
                receiver=epc_inputs["rx_batch"],
                active=None,
                expected_prim_ids=epc_inputs["sequence_batch"],
                direct_plane_points=epc_inputs["direct_plane_points"],
                direct_plane_normals=epc_inputs["direct_plane_normals"],
                surface_group_id=groups["surface_group_id"],
                surface_group_size=groups["surface_group_size"],
                surface_group_members=groups["surface_group_members"],
                max_bounces=1,
                visibility_ignore_mode=1,
            )
        )
        launch_count += 1
        selected = topology_kernels.deterministic_reflection_order1_compact(
            visible=epc.visible,
            epc_faces=epc.resolved_prim_ids,
            epc_hits=epc.hit_positions,
            epc_normals=epc.normals,
            sequence_batch=epc_inputs["sequence_batch"],
            rx_indices=epc_inputs["rx_indices"],
            tx=tx,
            rx_positions=rx_positions,
            tx_power=tx_power.to(dtype=torch.float32).contiguous(),
            tx_index=tx_index,
            face_eps_r=face_eps_r,
            face_sigma_e=face_sigma_e,
            face_mu_r=face_mu_r,
            face_gain=face_gain,
            face_material_id=face_material_id,
            grouped_export=grouped_export,
        )
        if int(selected["selected_faces"].numel()) == 0:
            continue

        field_result = field_kernels.deterministic_reflection_field(
            tx_position=selected["tx_keep"],
            rx_position=selected["rx_keep"],
            hit_position=selected["selected_points"],
            normal=selected["selected_normals"],
            tx_power=selected["tx_power"],
            eps_r=selected["eps_r"],
            sigma_e=selected["sigma_e"],
            mu_r=selected["mu_r"],
            gain=selected["gain"],
            frequency_hz=frequency_hz,
        )
        path_gain = field_result["path_gain"].to(dtype=torch.float32).contiguous()
        path_field = field_kernels.deterministic_pack_complex(
            field_result["field_real"], field_result["field_imag"]
        )
        path_length = field_result["path_length_m"].to(dtype=torch.float32).contiguous()
        delay = field_result["delay_s"].to(dtype=torch.float32).contiguous()
        blocks.append(
            _ensure_topology_fields(
                topology_kernels.deterministic_topology_base_fields(
                    rx_id=selected["selected_rx_id"],
                    path_length_m=path_length.to(dtype=torch.float32).contiguous(),
                    delay_s=delay,
                    path_gain=path_gain.to(dtype=torch.float32).contiguous(),
                    tx_index=tx_index,
                    component_id=1,
                    depth_source=empty_i32,
                    depth_value=1,
                    primitive_source=selected["selected_faces"],
                    primitive_value=-1,
                    edge_source=empty_i32,
                    edge_value=-1,
                ),
                interaction_position=selected["selected_points"],
                interaction_normal=selected["selected_normals"],
                material_id=selected["material_id"],
                path_field=path_field,
            )
        )
    return _ensure_topology_fields(
        concatenate_path_blocks(blocks, device=device)
    ), launch_count


def _discovered_group_chains(
    rayd: object,
    tx: torch.Tensor,
    *,
    face_group_id: torch.Tensor,
    max_depth: int,
    ray_count: int = _MULTIBOUNCE_DISCOVERY_RAYS,
) -> torch.Tensor:
    """Trace specular chains from the transmitter and map them to plane groups.

    Returns an (N, max_depth) long tensor of plane-group ids per bounce with
    -1 past each ray's last hit. Only chains reachable from the transmitter
    can host a valid specular path, so validating the unique chains found here
    replaces the exhaustive plane-sequence product on large scenes.
    """

    device = face_group_id.device
    ray_o = tx.reshape(1, 3).expand(ray_count, 3).contiguous()
    ray_d = mc_sample_directions(ray_count, tx.reshape(1, 3))
    ray_tmax = torch.empty((0,), device=device, dtype=torch.float32)
    out = geometry_kernels.rayd_trace_reflections_forward(
        rayd.require_resource(),
        ray_o,
        ray_d,
        ray_tmax,
        None,
        int(max_depth),
    )
    prim_chain = out[2].to(dtype=torch.long).reshape(ray_count, int(max_depth))
    chains = torch.full_like(prim_chain, -1)
    hit = prim_chain >= 0
    chains[hit] = face_group_id[prim_chain[hit]]
    return chains


def _reflection_topology_multibounce(
    scene: Scene,
    compiled: object,
    tx_positions: torch.Tensor,
    tx_power: torch.Tensor,
    rx_positions: torch.Tensor,
    *,
    frequency_hz: float,
    min_depth: int,
    max_depth: int,
    max_paths: int | None,
) -> tuple[dict[str, torch.Tensor], int, int]:
    from witwin.channel.kernels import fields as field_kernels

    device = tx_positions.device
    rayd = compiled.rayd
    if not scene.structures or tx_positions.numel() == 0 or rx_positions.numel() == 0:
        return (
            _ensure_topology_fields(
                {
                    "valid": torch.empty((0,), device=device, dtype=torch.bool),
                    "tx_id": torch.empty((0,), device=device, dtype=torch.int32),
                    "rx_id": torch.empty((0,), device=device, dtype=torch.int32),
                    "depth": torch.empty((0,), device=device, dtype=torch.int32),
                    "component_id": torch.empty((0,), device=device, dtype=torch.int32),
                    "primitive_id": torch.empty((0,), device=device, dtype=torch.int32),
                    "edge_id": torch.empty((0,), device=device, dtype=torch.int32),
                    "path_length_m": torch.empty(
                        (0,), device=device, dtype=torch.float32
                    ),
                    "delay_s": torch.empty((0,), device=device, dtype=torch.float32),
                    "path_gain": torch.empty((0,), device=device, dtype=torch.float32),
                }
            ),
            0,
            0,
        )
    if not rayd.available:
        raise RuntimeError(
            "deterministic multibounce reflection requires RayD native scene capability"
        )

    records = rayd.edge_records()
    vertices = records.vertices
    faces = records.faces.contiguous()
    normals = geometry_kernels.deterministic_normalize_vec3(
        records.face_normals.contiguous(), eps=1.0e-6
    )
    face_count = int(faces.shape[0])
    if face_count == 0 or max_depth < min_depth:
        return (
            _ensure_topology_fields(
                {
                    "valid": torch.empty((0,), device=device, dtype=torch.bool),
                    "tx_id": torch.empty((0,), device=device, dtype=torch.int32),
                    "rx_id": torch.empty((0,), device=device, dtype=torch.int32),
                    "depth": torch.empty((0,), device=device, dtype=torch.int32),
                    "component_id": torch.empty((0,), device=device, dtype=torch.int32),
                    "primitive_id": torch.empty((0,), device=device, dtype=torch.int32),
                    "edge_id": torch.empty((0,), device=device, dtype=torch.int32),
                    "path_length_m": torch.empty(
                        (0,), device=device, dtype=torch.float32
                    ),
                    "delay_s": torch.empty((0,), device=device, dtype=torch.float32),
                    "path_gain": torch.empty((0,), device=device, dtype=torch.float32),
                }
            ),
            0,
            0,
        )

    tri_a = topology_kernels.deterministic_face_anchor_points(
        vertices.contiguous(), faces
    )
    face_eps_r, face_sigma_e, face_mu_r, face_gain, _face_valid = face_material_tensors(
        compiled, device=device
    )
    face_material_id = compiled.assignments.face_material_id.to(
        device=device, dtype=torch.int32
    ).contiguous()
    face_group_source = compiled.geometry.face_surface_id.to(
        device=device, dtype=torch.long
    ).contiguous()
    # Coplanar plane groups carry the specular semantics (dedup, adjacency,
    # visibility-ignore scope). When the exhaustive plane-sequence space fits
    # the planning guard, enumerate it exactly; otherwise discover reachable
    # plane chains by tracing rays from the transmitter and validate only
    # those, matching the original discovery-based implementation.
    groups = _cached_coplanar_face_groups(rayd, tri_a, normals, face_group_source)
    group_count = int(groups["group_count"])
    representative_faces = groups["representative_faces"].contiguous()
    surface_group_id = groups["surface_group_id"]
    surface_group_size = groups["surface_group_size"]
    surface_group_members = groups["surface_group_members"]
    plan = prepare_reflection_multibounce_plan(
        group_count=group_count,
        representative_faces=representative_faces,
        face_group_id=groups["face_group_id"],
        min_depth=min_depth,
        max_depth=max_depth,
    )
    tx_power_f32 = tx_power.to(dtype=torch.float32).contiguous()
    blocks: list[dict[str, torch.Tensor]] = []
    launch_count = 0
    theoretical_candidate_count = 0

    def trace_group_chains(
        tx: torch.Tensor, *, face_group_id: torch.Tensor, max_depth: int
    ) -> torch.Tensor:
        nonlocal launch_count
        chains = _discovered_group_chains(
            rayd, tx, face_group_id=face_group_id, max_depth=max_depth
        )
        launch_count += 1
        return chains

    def record_candidate_count(candidate_count: int) -> None:
        nonlocal theoretical_candidate_count
        theoretical_candidate_count += candidate_count

    for request in iter_reflection_multibounce_epc_requests(
        plan,
        tx_positions=tx_positions,
        rx_positions=rx_positions,
        sequence_reference=tx_power_f32,
        tri_a=tri_a,
        normals=normals,
        trace_group_chains=trace_group_chains,
        record_candidate_count=record_candidate_count,
    ):
        depth = request.depth
        tx_index = request.tx_index
        tx = request.tx
        epc_inputs = request.epc_inputs
        epc = query_reflection_epc(
            ReflectionEpcQuery(
                rayd=rayd,
                source=epc_inputs["tx_batch"],
                receiver=epc_inputs["rx_batch"],
                active=None,
                expected_prim_ids=epc_inputs["sequence_batch"],
                direct_plane_points=epc_inputs["direct_plane_points"],
                direct_plane_normals=epc_inputs["direct_plane_normals"],
                surface_group_id=surface_group_id,
                surface_group_size=surface_group_size,
                surface_group_members=surface_group_members,
                max_bounces=int(depth),
                visibility_ignore_mode=1,
            )
        )
        launch_count += 1
        selected = topology_kernels.deterministic_reflection_sequence_compact(
            visible=epc.visible,
            epc_sequences=epc.resolved_prim_ids,
            epc_hits=epc.hit_positions,
            epc_normals=epc.normals,
            rx_indices=epc_inputs["rx_indices"],
            tx=tx,
            rx_positions=rx_positions,
            tx_power=tx_power_f32,
            tx_index=tx_index,
            face_eps_r=face_eps_r,
            face_sigma_e=face_sigma_e,
            face_mu_r=face_mu_r,
            face_gain=face_gain,
            face_material_id=face_material_id,
            max_count=-1,
        )
        count = int(selected["selected_sequences"].shape[0])
        if count == 0:
            continue
        field_result = field_kernels.deterministic_reflection_sequence_field(
            tx_position=selected["selected_tx"],
            rx_position=selected["selected_rx"],
            hit_positions=selected["selected_hits"],
            normals=selected["selected_normals"],
            tx_power=selected["tx_power"],
            eps_r=selected["eps_r"],
            sigma_e=selected["sigma_e"],
            mu_r=selected["mu_r"],
            gain=selected["gain"],
            frequency_hz=frequency_hz,
        )
        path_gain = field_result["path_gain"].to(dtype=torch.float32).contiguous()
        path_field = field_kernels.deterministic_pack_complex(
            field_result["field_real"], field_result["field_imag"]
        )
        path_length = field_result["path_length_m"].to(dtype=torch.float32).contiguous()
        delay = field_result["delay_s"].to(dtype=torch.float32).contiguous()
        empty_i32 = torch.empty((0,), device=device, dtype=torch.int32)
        blocks.append(
            _ensure_topology_fields(
                topology_kernels.deterministic_topology_base_fields(
                    rx_id=selected["selected_rx_id"],
                    path_length_m=path_length,
                    delay_s=delay,
                    path_gain=path_gain,
                    tx_index=tx_index,
                    component_id=1,
                    depth_source=empty_i32,
                    depth_value=depth,
                    primitive_source=selected["first_face"],
                    primitive_value=-1,
                    edge_source=empty_i32,
                    edge_value=-1,
                ),
                interaction_position=selected["first_hit"],
                interaction_normal=selected["first_normal"],
                material_id=selected["material_id"],
                path_field=path_field,
                primitive_sequence=selected["selected_sequences"],
                material_sequence=selected["material_sequence"],
                interaction_positions=selected["selected_hits"],
                interaction_normals=selected["selected_normals"],
            )
        )

    return (
        _ensure_topology_fields(concatenate_path_blocks(blocks, device=device)),
        launch_count,
        theoretical_candidate_count,
    )
