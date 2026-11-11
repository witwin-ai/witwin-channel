# Copyright Xingyu Chen.
# Reflection: discovery planning, EPC geometry, and enumerated orchestration.

"""Reflection: discovery planning, EPC geometry, and enumerated orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Protocol, TYPE_CHECKING

import torch

from witwin.channel.materials import face_geometry_id_table, face_material_id_table
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
# Upper bound of EPC candidates per launch; the memory-derived chunk may be smaller.
_MULTIBOUNCE_PAIR_CHUNK_SIZE = 4_194_304
_MULTIBOUNCE_DISCOVERY_RAYS = 262_144
# Transient EPC working set per candidate and per bounce (inputs, RayD outputs and
# its staging copies), and the share of free device memory one chunk may use.
_EPC_BYTES_PER_CANDIDATE = 192
_EPC_BYTES_PER_BOUNCE = 192
_EPC_CHUNK_MEMORY_LIMIT = 2 << 30
# Discovery traces at most this many transmitters' rays per launch.
_DISCOVERY_TX_CHUNK = 32
_DISCOVERY_DIRECTIONS: dict[tuple[torch.device, int], torch.Tensor] = {}


def _face_sequence_count(face_count: int, depth: int, *, adjacent_distinct: bool) -> int:
    if adjacent_distinct and depth > 1:
        if face_count <= 1:
            return 0
        return int(face_count) * int(face_count - 1) ** int(depth - 1)
    return int(face_count) ** int(depth)


def _face_sequence_chunks(
    face_count: int, depth: int, *, chunk_size: int, reference: torch.Tensor,
    face_ids: torch.Tensor | None = None, adjacent_distinct: bool = False,
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
        self, tx: torch.Tensor, *, face_group_id: torch.Tensor, max_depth: int,
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
class ReflectionMultibouncePlan:
    exhaustive: bool
    group_count: int
    representative_faces: torch.Tensor
    face_group_id: torch.Tensor | None
    min_depth: int
    max_depth: int


@dataclass(frozen=True, slots=True)
class ReflectionEpcRequest:
    """One EPC launch over a contiguous candidate range covering every transmitter."""

    depth: int
    epc_inputs: dict[str, torch.Tensor]


def epc_candidate_chunk(depth: int, device: torch.device) -> int:
    """Candidates per EPC launch, sized from free device memory and the launch cap."""

    if torch.device(device).type != "cuda":
        return _MULTIBOUNCE_PAIR_CHUNK_SIZE
    free_bytes, _total = torch.cuda.mem_get_info(device)
    budget = min(int(free_bytes) // 4, _EPC_CHUNK_MEMORY_LIMIT)
    per_candidate = _EPC_BYTES_PER_CANDIDATE + _EPC_BYTES_PER_BOUNCE * int(depth)
    return max(1, min(_MULTIBOUNCE_PAIR_CHUNK_SIZE, budget // per_candidate))


def _epc_requests(
    depth: int, *, tx_positions: torch.Tensor, rx_positions: torch.Tensor, sequences: torch.Tensor,
    tri_a: torch.Tensor, normals: torch.Tensor, sequence_tx: torch.Tensor | None = None,
) -> Iterator[ReflectionEpcRequest]:
    total = topology_kernels.reflection_epc_candidate_count(
        tx_count=int(tx_positions.shape[0]),
        rx_count=int(rx_positions.shape[0]),
        sequence_count=int(sequences.shape[0]),
        per_transmitter=sequence_tx is not None,
    )
    if total <= 0:
        return
    chunk = epc_candidate_chunk(depth, sequences.device)
    for start in range(0, total, chunk):
        yield ReflectionEpcRequest(
            depth=depth,
            epc_inputs=topology_kernels.deterministic_reflection_epc_input_batch(
                tx_positions=tx_positions.contiguous(),
                rx_positions=rx_positions.contiguous(),
                sequences=sequences.contiguous(),
                tri_a=tri_a.contiguous(),
                normals=normals.contiguous(),
                candidate_start=start,
                candidate_end=min(start + chunk, total),
                sequence_tx=sequence_tx,
            ),
        )


def _unique_tx_chains(chains: torch.Tensor, *, depth: int, group_count: int) -> torch.Tensor:
    """Distinct ``(tx, group_1..group_depth)`` rows of chains reaching ``depth``, sorted.

    ``chains`` is ``(T, N, max_depth)`` plane-group ids with -1 past each ray's last hit.
    """

    tx_count, ray_count, _width = chains.shape
    reached = chains[:, :, depth - 1] >= 0
    radix = max(int(group_count), 1)
    if tx_count * radix**depth < 2**62:
        tx_column = torch.arange(tx_count, device=chains.device, dtype=torch.int64)
        key = tx_column.reshape(-1, 1).expand(tx_count, ray_count)
        for column in range(depth):
            key = key * radix + chains[:, :, column]
        unique = torch.unique(key[reached])
        rows = torch.empty((int(unique.shape[0]), depth + 1), device=chains.device, dtype=torch.int64)
        for column in range(depth, 0, -1):
            rows[:, column] = unique % radix
            unique = unique // radix
        rows[:, 0] = unique
        return rows
    tx_column = torch.arange(tx_count, device=chains.device, dtype=torch.int64)
    tagged = torch.cat(
        (
            tx_column.reshape(-1, 1, 1).expand(tx_count, ray_count, 1),
            chains[:, :, :depth],
        ),
        dim=2,
    )
    return torch.unique(tagged[reached], dim=0)


def _discovered_sequences(
    *, tx_positions: torch.Tensor, face_group_id: torch.Tensor, representative_faces: torch.Tensor,
    group_count: int, min_depth: int, max_depth: int,
    trace_group_chains: TraceReflectionGroupChains,
) -> Iterator[tuple[int, torch.Tensor, torch.Tensor]]:
    """Plane-group chains each transmitter's discovery rays reach, as per-transmitter sequences.

    Yields ``(depth, sequences, sequence_tx)`` per transmitter chunk and depth; one trace
    launch covers every transmitter of the chunk.
    """

    tx_count = int(tx_positions.shape[0])
    for tx_start in range(0, tx_count, _DISCOVERY_TX_CHUNK):
        tx_chunk = tx_positions[tx_start : tx_start + _DISCOVERY_TX_CHUNK].contiguous()
        chunk_count = int(tx_chunk.shape[0])
        chains = trace_group_chains(
            tx_chunk, face_group_id=face_group_id, max_depth=max_depth
        ).reshape(chunk_count, -1, max_depth)
        for depth in range(min_depth, max_depth + 1):
            rows = _unique_tx_chains(chains, depth=depth, group_count=group_count)
            if int(rows.shape[0]) == 0:
                continue
            sequences = representative_faces[rows[:, 1:]].contiguous()
            sequence_tx = (rows[:, 0] + tx_start).to(dtype=torch.int32).contiguous()
            yield depth, sequences, sequence_tx


def prepare_reflection_order1_plan(
    *, group_count: int, representative_faces: torch.Tensor, face_group_id: torch.Tensor,
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
    plan: ReflectionOrder1Plan, *, tx_positions: torch.Tensor, rx_positions: torch.Tensor,
    tri_a: torch.Tensor, normals: torch.Tensor, trace_group_chains: TraceReflectionGroupChains,
) -> Iterator[ReflectionEpcRequest]:
    """First-order EPC requests for every transmitter, in transmitter-batched launches."""

    rx_count = int(rx_positions.shape[0])
    if plan.group_count <= 0 or rx_count <= 0 or int(tx_positions.shape[0]) <= 0:
        return
    if plan.exhaustive:
        assert plan.base_sequences is not None
        yield from _epc_requests(
            1,
            tx_positions=tx_positions,
            rx_positions=rx_positions,
            sequences=plan.base_sequences,
            tri_a=tri_a,
            normals=normals,
        )
        return
    assert plan.face_group_id is not None
    for depth, sequences, sequence_tx in _discovered_sequences(
        tx_positions=tx_positions,
        face_group_id=plan.face_group_id,
        representative_faces=plan.representative_faces,
        group_count=plan.group_count,
        min_depth=1,
        max_depth=1,
        trace_group_chains=trace_group_chains,
    ):
        yield from _epc_requests(
            depth,
            tx_positions=tx_positions,
            rx_positions=rx_positions,
            sequences=sequences,
            tri_a=tri_a,
            normals=normals,
            sequence_tx=sequence_tx,
        )


def prepare_reflection_multibounce_plan(
    *, group_count: int, representative_faces: torch.Tensor, face_group_id: torch.Tensor,
    min_depth: int, max_depth: int,
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
    plan: ReflectionMultibouncePlan, *, tx_positions: torch.Tensor, rx_positions: torch.Tensor,
    sequence_reference: torch.Tensor, tri_a: torch.Tensor, normals: torch.Tensor,
    trace_group_chains: TraceReflectionGroupChains,
    record_candidate_count: RecordReflectionCandidateCount,
) -> Iterator[ReflectionEpcRequest]:
    """Multi-bounce EPC requests for every transmitter, in transmitter-batched launches."""

    if int(rx_positions.shape[0]) <= 0 or int(tx_positions.shape[0]) <= 0:
        return
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
                yield from _epc_requests(
                    depth,
                    tx_positions=tx_positions,
                    rx_positions=rx_positions,
                    sequences=sequences,
                    tri_a=tri_a,
                    normals=normals,
                )
        return
    assert plan.face_group_id is not None
    for depth, sequences, sequence_tx in _discovered_sequences(
        tx_positions=tx_positions,
        face_group_id=plan.face_group_id,
        representative_faces=plan.representative_faces,
        group_count=plan.group_count,
        min_depth=plan.min_depth,
        max_depth=plan.max_depth,
        trace_group_chains=trace_group_chains,
    ):
        record_candidate_count(int(sequences.shape[0]))
        yield from _epc_requests(
            depth,
            tx_positions=tx_positions,
            rx_positions=rx_positions,
            sequences=sequences,
            tri_a=tri_a,
            normals=normals,
            sequence_tx=sequence_tx,
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


def _empty_reflection_block(device: torch.device) -> dict[str, torch.Tensor]:
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
    )


def _discovery_directions(device: torch.device, ray_count: int) -> torch.Tensor:
    """The fixed discovery ray directions, generated once per device and ray count."""

    key = (torch.device(device), int(ray_count))
    directions = _DISCOVERY_DIRECTIONS.get(key)
    if directions is None:
        reference = torch.zeros((1, 3), device=device, dtype=torch.float32)
        directions = mc_sample_directions(int(ray_count), reference)
        _DISCOVERY_DIRECTIONS[key] = directions
    return directions


def _discovered_group_chains(
    rayd: object, tx_positions: torch.Tensor, *, face_group_id: torch.Tensor, max_depth: int,
    ray_count: int = _MULTIBOUNCE_DISCOVERY_RAYS,
) -> torch.Tensor:
    """Trace specular chains from every transmitter and map them to plane groups.

    Returns a ``(T * N, max_depth)`` long tensor of plane-group ids per bounce, transmitter
    major, with -1 past each ray's last hit. Only chains reachable from a transmitter can host
    a valid specular path, so validating the unique chains found here replaces the exhaustive
    plane-sequence product on large scenes. One trace launch covers every transmitter.
    """

    device = face_group_id.device
    tx_positions = tx_positions.reshape(-1, 3)
    tx_count = int(tx_positions.shape[0])
    directions = _discovery_directions(device, ray_count)
    ray_o = tx_positions.repeat_interleave(int(ray_count), dim=0).contiguous()
    ray_d = directions.repeat(tx_count, 1).contiguous() if tx_count > 1 else directions
    ray_tmax = torch.empty((0,), device=device, dtype=torch.float32)
    out = geometry_kernels.rayd_trace_reflections_forward(
        rayd.require_resource(),
        ray_o,
        ray_d,
        ray_tmax,
        None,
        int(max_depth),
    )
    prim_chain = out[2].to(dtype=torch.long).reshape(tx_count * int(ray_count), int(max_depth))
    hit = prim_chain >= 0
    return torch.where(hit, face_group_id[prim_chain.clamp_min(0)], -1)


def _reflection_geometry_tables(compiled: object, device: torch.device) -> tuple[torch.Tensor, ...]:
    records = compiled.rayd.edge_records()
    faces = records.faces.contiguous()
    normals = geometry_kernels.deterministic_normalize_vec3(
        records.face_normals.contiguous(), eps=1.0e-6
    )
    tri_a = topology_kernels.deterministic_face_anchor_points(
        records.vertices.contiguous(), faces
    )
    return faces, normals, tri_a


def _reflection_blocks(
    requests: Iterator[ReflectionEpcRequest], *, rayd: object, groups: dict[str, torch.Tensor],
    face_material_id: torch.Tensor, on_epc: object,
) -> list[dict[str, torch.Tensor]]:
    blocks: list[dict[str, torch.Tensor]] = []
    for request in requests:
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
                max_bounces=int(request.depth),
                visibility_ignore_mode=1,
            )
        )
        on_epc()
        block = topology_kernels.deterministic_reflection_topology_compact(
            visible=epc.visible,
            epc_sequences=epc.resolved_prim_ids,
            epc_hits=epc.hit_positions,
            epc_normals=epc.normals,
            epc_path_length=epc.path_length_m,
            tx_indices=epc_inputs["tx_indices"],
            rx_indices=epc_inputs["rx_indices"],
            face_material_id=face_material_id,
        )
        if int(block["valid"].shape[0]) > 0:
            blocks.append(_ensure_topology_fields(block))
    return blocks


def _reflection_topology_order1(
    scene: Scene, compiled: object, tx_positions: torch.Tensor, tx_power: torch.Tensor,
    rx_positions: torch.Tensor, *, frequency_hz: float,
) -> tuple[dict[str, torch.Tensor], int]:
    """First-order reflection topology for every transmitter and receiver.

    The rows carry EPC geometry and path length only; the field stage evaluates their fields.
    """

    device = tx_positions.device
    rayd = compiled.rayd
    if not scene.structures or tx_positions.numel() == 0 or rx_positions.numel() == 0:
        return _empty_reflection_block(device), 0
    if not rayd.available:
        raise RuntimeError(
            "deterministic reflection requires RayD native scene capability"
        )
    faces, normals, tri_a = _reflection_geometry_tables(compiled, device)
    if faces.shape[0] == 0:
        return _empty_reflection_block(device), 0

    face_material_id = face_material_id_table(compiled, device=device)
    # Enumerate one candidate per coplanar face group so that a wall meshed
    # from several coplanar triangles yields exactly one specular path, and
    # every planar facade (not one representative per structure) is covered.
    # The EPC kernel resolves the actual containing triangle per path.
    groups = _cached_coplanar_face_groups(
        rayd,
        tri_a,
        normals,
        face_geometry_id_table(compiled, "face_surface_id", device=device),
    )
    group_count = int(groups["group_count"])
    if group_count <= 0:
        return _empty_reflection_block(device), 0
    plan = prepare_reflection_order1_plan(
        group_count=group_count,
        representative_faces=groups["representative_faces"].contiguous(),
        face_group_id=groups["face_group_id"],
    )
    launch_count = 0

    def trace_group_chains(
        tx: torch.Tensor, *, face_group_id: torch.Tensor, max_depth: int,
    ) -> torch.Tensor:
        nonlocal launch_count
        chains = _discovered_group_chains(
            rayd, tx, face_group_id=face_group_id, max_depth=max_depth
        )
        launch_count += 1
        return chains

    def count_epc() -> None:
        nonlocal launch_count
        launch_count += 1

    blocks = _reflection_blocks(
        iter_reflection_order1_epc_requests(
            plan,
            tx_positions=tx_positions,
            rx_positions=rx_positions,
            tri_a=tri_a,
            normals=normals,
            trace_group_chains=trace_group_chains,
        ),
        rayd=rayd,
        groups=groups,
        face_material_id=face_material_id,
        on_epc=count_epc,
    )
    return _ensure_topology_fields(
        concatenate_path_blocks(blocks, device=device)
    ), launch_count


def _reflection_topology_multibounce(
    scene: Scene, compiled: object, tx_positions: torch.Tensor, tx_power: torch.Tensor,
    rx_positions: torch.Tensor, *, frequency_hz: float, min_depth: int, max_depth: int,
    max_paths: int | None,
) -> tuple[dict[str, torch.Tensor], int, int]:
    """Multi-bounce reflection topology for every transmitter and receiver.

    The rows carry EPC geometry and path length only; the field stage evaluates their fields.
    """

    device = tx_positions.device
    rayd = compiled.rayd
    if not scene.structures or tx_positions.numel() == 0 or rx_positions.numel() == 0:
        return _empty_reflection_block(device), 0, 0
    if not rayd.available:
        raise RuntimeError(
            "deterministic multibounce reflection requires RayD native scene capability"
        )
    faces, normals, tri_a = _reflection_geometry_tables(compiled, device)
    if int(faces.shape[0]) == 0 or max_depth < min_depth:
        return _empty_reflection_block(device), 0, 0

    face_material_id = face_material_id_table(compiled, device=device)
    # Coplanar plane groups carry the specular semantics (dedup, adjacency,
    # visibility-ignore scope). When the exhaustive plane-sequence space fits
    # the planning guard, enumerate it exactly; otherwise discover reachable
    # plane chains by tracing rays from the transmitters and validate only
    # those.
    groups = _cached_coplanar_face_groups(
        rayd,
        tri_a,
        normals,
        face_geometry_id_table(compiled, "face_surface_id", device=device),
    )
    plan = prepare_reflection_multibounce_plan(
        group_count=int(groups["group_count"]),
        representative_faces=groups["representative_faces"].contiguous(),
        face_group_id=groups["face_group_id"],
        min_depth=min_depth,
        max_depth=max_depth,
    )
    launch_count = 0
    theoretical_candidate_count = 0

    def trace_group_chains(
        tx: torch.Tensor, *, face_group_id: torch.Tensor, max_depth: int,
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

    def count_epc() -> None:
        nonlocal launch_count
        launch_count += 1

    blocks = _reflection_blocks(
        iter_reflection_multibounce_epc_requests(
            plan,
            tx_positions=tx_positions,
            rx_positions=rx_positions,
            sequence_reference=tx_power.to(dtype=torch.float32).contiguous(),
            tri_a=tri_a,
            normals=normals,
            trace_group_chains=trace_group_chains,
            record_candidate_count=record_candidate_count,
        ),
        rayd=rayd,
        groups=groups,
        face_material_id=face_material_id,
        on_epc=count_epc,
    )
    return (
        _ensure_topology_fields(concatenate_path_blocks(blocks, device=device)),
        launch_count,
        theoretical_candidate_count,
    )
