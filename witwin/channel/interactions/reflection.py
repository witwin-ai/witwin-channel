# Copyright Xingyu Chen.
# Reflection: candidate discovery, EPC geometry, and enumerated orchestration.

"""Reflection: candidate discovery, EPC geometry, and enumerated orchestration."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterator, Protocol, TYPE_CHECKING, cast

import torch

from witwin.channel.components import DISCOVERY_MODES
from witwin.channel.materials import face_geometry_id_table, face_material_id_table
from witwin.channel.kernels import geometry as geometry_kernels
from witwin.channel.kernels import topology as topology_kernels
from witwin.channel.kernels.topology import (
    mc_sample_directions,
)
from witwin.channel.propagation.geometry import (
    REFLECTION_MIN_SEGMENT_RELATIVE,
    _cached_coplanar_face_groups,
    reflection_scene_scale_m,
)
from witwin.channel.propagation.topology import (
    concatenate_path_blocks,
)
from witwin.channel.propagation.topology import _ensure_topology_fields

if TYPE_CHECKING:
    from witwin.channel.scene.compiler import CompiledScene
    from witwin.channel.scene.endpoints import SolverScene as Scene
    from witwin.channel.scene.resources import RayDSceneResource

# "auto" builds the complete image tree per depth and hands a depth (and every deeper one) to
# ray discovery only when the tree's cost exceeds the budget below; "exhaustive" is the
# unpruned tree; "rays" is ray discovery at every depth.
# Source names recorded per depth in result metadata.
SOURCE_IMAGE_TREE = "image_tree"
SOURCE_EXHAUSTIVE = "exhaustive"
SOURCE_RAYS = "rays"

# Pair tests (image-tree children or sequence/receiver pairs) one depth of the complete
# source may spend under "auto" before that depth moves to ray discovery.
_COMPLETE_PAIR_BUDGET = 1 << 31
# Pair tests per native launch; keeps every candidate-generation kernel to a few milliseconds.
_PAIR_TESTS_PER_LAUNCH = 1 << 25
# Below this many sequence/receiver pairs every pair goes to the EPC validator directly; above
# it the image-method receiver test selects the pairs first. The test also runs at any size when
# a plane group has more members than _DIRECT_GROUP_SIZE, because the validator's cost for a
# point outside a group grows with the group's member count.
_RECEIVER_TEST_MIN_PAIRS = 1 << 20
_DIRECT_GROUP_SIZE = 64
# Share of free device memory the image-tree frontier of one depth may hold.
_FRONTIER_MEMORY_SHARE = 8
# Pruning tolerances, relative to the scene diagonal. Candidate pruning uses half of the validity
# minimum segment and inflates plane-group bounds by _BOUNDS_RELATIVE so it never rejects a path
# the validator accepts.
_BOUNDS_RELATIVE = 1.0e-4
_TREE_RELATIVE = 1.0e-4
# Slack on the back-trace intersection parameter; ten times the EPC solve's own slack.
_SEGMENT_PARAMETER_SLACK = 1.0e-3
# Upper bound of reflection EPC candidates per launch; the memory-derived chunk may be smaller.
_EPC_MAX_CANDIDATES = 1 << 20
# Upper bound of diffraction (receiver x state) lanes per launch, read by diffraction discovery.
_MULTIBOUNCE_PAIR_CHUNK_SIZE = 4_194_304
_MULTIBOUNCE_DISCOVERY_RAYS = 262_144
# Transient EPC working set per candidate and per bounce (inputs, RayD outputs and
# its staging copies), and the share of free device memory one chunk may use.
_EPC_BYTES_PER_CANDIDATE = 192
_EPC_BYTES_PER_BOUNCE = 192
_EPC_CHUNK_MEMORY_LIMIT = 512 << 20
# Discovery traces at most this many transmitters' rays per launch.
_DISCOVERY_TX_CHUNK = 32
_DISCOVERY_DIRECTIONS: dict[tuple[torch.device, int], torch.Tensor] = {}


class TraceReflectionGroupChains(Protocol):
    def __call__(
        self, tx: torch.Tensor, *, face_group_id: torch.Tensor, max_depth: int,
    ) -> torch.Tensor: ...


class RecordReflectionCandidateCount(Protocol):
    def __call__(self, candidate_count: int, /) -> None: ...


def validate_reflection_discovery(value: object) -> str:
    """The candidate source name, or a ``ValueError`` naming the accepted ones."""

    if not isinstance(value, str) or value not in DISCOVERY_MODES:
        raise ValueError(
            f"discovery must be one of {', '.join(map(repr, DISCOVERY_MODES))}, "
            f"got {value!r}"
        )
    return value


@dataclass(slots=True)
class ReflectionDiscoveryRecord:
    """Which candidate source produced each reflection depth of one solve."""

    sources: dict[int, str] = field(default_factory=dict)

    @property
    def complete(self) -> bool:
        return all(source != SOURCE_RAYS for source in self.sources.values())

    def metadata(self, requested: str) -> dict[str, object]:
        return {
            "requested": requested,
            "sources": {str(depth): self.sources[depth] for depth in sorted(self.sources)},
            "complete": self.complete,
        }


@dataclass(frozen=True, slots=True)
class ReflectionPlaneGroups:
    """Coplanar plane groups with the per-group tables candidate generation reads."""

    group_count: int
    representative_faces: torch.Tensor
    face_group_id: torch.Tensor
    plane_point: torch.Tensor
    plane_normal: torch.Tensor
    bounds: torch.Tensor
    occupancy: torch.Tensor
    scene_scale_m: float
    max_group_size: int = 1


@dataclass(frozen=True, slots=True)
class ReflectionDiscoveryPlan:
    groups: ReflectionPlaneGroups
    min_depth: int
    max_depth: int
    discovery: str


@dataclass(frozen=True, slots=True)
class ReflectionEpcRequest:
    """One EPC launch over a contiguous candidate range covering every transmitter."""

    depth: int
    epc_inputs: dict[str, torch.Tensor]


def reflection_plane_groups(
    compiled: CompiledScene, groups: dict[str, torch.Tensor | int], *, tri_a: torch.Tensor,
    normals: torch.Tensor,
) -> ReflectionPlaneGroups:
    """Plane-group tables of a compiled scene, cached per RayD scene like the groups themselves."""

    rayd = compiled.rayd
    cache = getattr(rayd, "runtime_cache", None)
    cached = cache.get("reflection_plane_groups") if cache is not None else None
    if cached is not None:
        return cast(ReflectionPlaneGroups, cached)
    records = rayd.edge_records()
    representative_faces = cast(torch.Tensor, groups["representative_faces"]).contiguous()
    face_group_id = cast(torch.Tensor, groups["face_group_id"]).to(dtype=torch.int32).contiguous()
    group_count = int(groups["group_count"])
    scale = reflection_scene_scale_m(compiled)
    tables = (
        topology_kernels.reflection_group_tables(
            vertices=records.vertices.detach().to(dtype=torch.float32).contiguous(),
            faces=records.faces.to(dtype=torch.int32).contiguous(),
            face_group_id=face_group_id,
            representative_faces=representative_faces.to(dtype=torch.int64),
            tri_a=tri_a.contiguous(),
            normals=normals.contiguous(),
            tolerance=_BOUNDS_RELATIVE * scale,
        )
        if group_count > 0
        else {
            "plane_point": tri_a.new_empty((0, 3)),
            "plane_normal": tri_a.new_empty((0, 3)),
            "bounds": tri_a.new_empty((0, 6)),
            "occupancy": torch.empty((0, 32), device=tri_a.device, dtype=torch.int32),
        }
    )
    result = ReflectionPlaneGroups(
        group_count=group_count,
        representative_faces=representative_faces,
        face_group_id=cast(torch.Tensor, groups["face_group_id"]).to(dtype=torch.long).contiguous(),
        plane_point=tables["plane_point"],
        plane_normal=tables["plane_normal"],
        bounds=tables["bounds"],
        occupancy=tables["occupancy"],
        scene_scale_m=scale,
        max_group_size=(
            int(cast(torch.Tensor, groups["surface_group_members"]).numel()) // group_count
            if group_count > 0
            else 1
        ),
    )
    if cache is not None:
        cache["reflection_plane_groups"] = result
    return result


def reflection_min_segment_length(groups: ReflectionPlaneGroups) -> float:
    """Shortest valid segment between two interactions, metres: a fixed fraction of the scene diagonal."""

    return REFLECTION_MIN_SEGMENT_RELATIVE * groups.scene_scale_m


def prepare_reflection_discovery_plan(
    groups: ReflectionPlaneGroups, *, min_depth: int, max_depth: int, discovery: str = "auto",
) -> ReflectionDiscoveryPlan:
    return ReflectionDiscoveryPlan(
        groups=groups,
        min_depth=int(min_depth),
        max_depth=int(max_depth),
        discovery=validate_reflection_discovery(discovery),
    )


def epc_candidate_chunk(depth: int, device: torch.device) -> int:
    """Candidates per EPC launch, sized from free device memory and the launch cap."""

    if torch.device(device).type != "cuda":
        return _EPC_MAX_CANDIDATES
    free_bytes, _total = torch.cuda.mem_get_info(device)
    budget = min(int(free_bytes) // 4, _EPC_CHUNK_MEMORY_LIMIT)
    per_candidate = _EPC_BYTES_PER_CANDIDATE + _EPC_BYTES_PER_BOUNCE * int(depth)
    return max(1, min(_EPC_MAX_CANDIDATES, budget // per_candidate))


def _frontier_node_limit(depth: int, device: torch.device) -> int:
    """Image-tree nodes of one depth that fit the frontier's share of free device memory."""

    if torch.device(device).type != "cuda":
        return 1 << 24
    free_bytes, _total = torch.cuda.mem_get_info(device)
    per_node = 4 * (int(depth) + 1) + 12 + 8
    return max(1, int(free_bytes) // _FRONTIER_MEMORY_SHARE // per_node)


def _epc_requests(
    depth: int, *, tx_positions: torch.Tensor, rx_positions: torch.Tensor, sequences: torch.Tensor,
    tri_a: torch.Tensor, normals: torch.Tensor, sequence_tx: torch.Tensor | None = None,
    candidate_sequence: torch.Tensor | None = None, candidate_receiver: torch.Tensor | None = None,
) -> Iterator[ReflectionEpcRequest]:
    if candidate_sequence is not None:
        total = int(candidate_sequence.shape[0])
    else:
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
                candidate_sequence=candidate_sequence,
                candidate_receiver=candidate_receiver,
            ),
        )


def _unique_tx_chains(chains: torch.Tensor, *, depth: int, group_count: int) -> torch.Tensor:
    """Distinct ``(tx, group_1..group_depth)`` rows of chains reaching ``depth``, sorted.

    ``chains`` is ``(T, N, max_depth)`` plane-group ids with -1 past each ray's last hit. A chain
    that repeats a plane group in consecutive bounces is not a specular path and is skipped.
    """

    tx_count, ray_count, _width = chains.shape
    reached = chains[:, :, depth - 1] >= 0
    for column in range(1, depth):
        reached = reached & (chains[:, :, column] != chains[:, :, column - 1])
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


class _RayChains:
    """Plane-group chains traced once per solve from every transmitter, read per depth."""

    def __init__(
        self, tx_positions: torch.Tensor, groups: ReflectionPlaneGroups, max_depth: int,
        trace_group_chains: TraceReflectionGroupChains,
    ) -> None:
        self._chunks: list[tuple[int, torch.Tensor]] = []
        tx_count = int(tx_positions.shape[0])
        for tx_start in range(0, tx_count, _DISCOVERY_TX_CHUNK):
            tx_chunk = tx_positions[tx_start : tx_start + _DISCOVERY_TX_CHUNK].contiguous()
            chunk_count = int(tx_chunk.shape[0])
            chains = trace_group_chains(
                tx_chunk, face_group_id=groups.face_group_id, max_depth=max_depth
            ).reshape(chunk_count, -1, max_depth)
            self._chunks.append((tx_start, chains))
        self._group_count = groups.group_count

    def sequences(self, depth: int) -> tuple[torch.Tensor, torch.Tensor]:
        """``(sequences [S, depth] int32 group ids, sequence_tx [S] int32)`` in lexicographic order."""

        rows = [
            (tx_start, _unique_tx_chains(chains, depth=depth, group_count=self._group_count))
            for tx_start, chains in self._chunks
        ]
        sequences = torch.cat([r[:, 1:] for _start, r in rows]).to(dtype=torch.int32).contiguous()
        sequence_tx = torch.cat([r[:, 0] + start for start, r in rows]).to(dtype=torch.int32)
        return sequences, sequence_tx.contiguous()


@dataclass(slots=True)
class _ImageTreeFrontier:
    node_tx: torch.Tensor
    node_sequences: torch.Tensor
    node_images: torch.Tensor

    @property
    def size(self) -> int:
        return int(self.node_tx.shape[0])


def _image_tree_roots(tx_positions: torch.Tensor) -> _ImageTreeFrontier:
    tx_count = int(tx_positions.shape[0])
    device = tx_positions.device
    return _ImageTreeFrontier(
        node_tx=torch.arange(tx_count, device=device, dtype=torch.int32),
        node_sequences=torch.empty((tx_count, 0), device=device, dtype=torch.int32),
        node_images=tx_positions.to(dtype=torch.float32).contiguous(),
    )


def _expand_image_tree(
    frontier: _ImageTreeFrontier, groups: ReflectionPlaneGroups, *, prune: bool,
) -> _ImageTreeFrontier:
    """One more plane group on every frontier node, in lexicographic order, chunked per launch."""

    nodes_per_launch = max(1, _PAIR_TESTS_PER_LAUNCH // max(groups.group_count, 1))
    if frontier.size == 0:
        width = int(frontier.node_sequences.shape[1]) + 1
        return _ImageTreeFrontier(
            frontier.node_tx, frontier.node_sequences.new_empty((0, width)), frontier.node_images
        )
    parts: list[dict[str, torch.Tensor]] = []
    for start in range(0, frontier.size, nodes_per_launch):
        parts.append(
            topology_kernels.reflection_image_tree_expand(
                node_tx=frontier.node_tx,
                node_sequences=frontier.node_sequences,
                node_images=frontier.node_images,
                plane_point=groups.plane_point,
                plane_normal=groups.plane_normal,
                bounds=groups.bounds,
                occupancy=groups.occupancy,
                node_start=start,
                node_end=min(start + nodes_per_launch, frontier.size),
                tolerance=_TREE_RELATIVE * groups.scene_scale_m,
                prune=prune,
            )
        )
    if len(parts) == 1:
        part = parts[0]
        return _ImageTreeFrontier(part["node_tx"], part["node_sequences"], part["node_images"])
    return _ImageTreeFrontier(
        torch.cat([p["node_tx"] for p in parts]),
        torch.cat([p["node_sequences"] for p in parts]).contiguous(),
        torch.cat([p["node_images"] for p in parts]).contiguous(),
    )


def _candidate_requests(
    depth: int, *, sequences: torch.Tensor, sequence_tx: torch.Tensor,
    groups: ReflectionPlaneGroups, tx_positions: torch.Tensor, rx_positions: torch.Tensor,
    tri_a: torch.Tensor, normals: torch.Tensor,
    record_candidate_count: RecordReflectionCandidateCount | None,
) -> Iterator[ReflectionEpcRequest]:
    """EPC requests for per-transmitter group sequences against every receiver.

    Small products go to the validator whole; larger ones first keep only the
    ``(sequence, receiver)`` pairs the image method cannot rule out.
    """

    sequence_count = int(sequences.shape[0])
    rx_count = int(rx_positions.shape[0])
    if sequence_count == 0 or rx_count == 0:
        return
    face_sequences = groups.representative_faces[sequences.to(dtype=torch.long)].contiguous()
    pairs = sequence_count * rx_count
    if pairs <= _RECEIVER_TEST_MIN_PAIRS and groups.max_group_size <= _DIRECT_GROUP_SIZE:
        if record_candidate_count is not None:
            record_candidate_count(pairs)
        yield from _epc_requests(
            depth,
            tx_positions=tx_positions,
            rx_positions=rx_positions,
            sequences=face_sequences,
            tri_a=tri_a,
            normals=normals,
            sequence_tx=sequence_tx,
        )
        return
    sequences_per_launch = max(1, _PAIR_TESTS_PER_LAUNCH // rx_count)
    min_segment = 0.5 * reflection_min_segment_length(groups)
    box = _BOUNDS_RELATIVE * groups.scene_scale_m
    for start in range(0, sequence_count, sequences_per_launch):
        candidate_sequence, candidate_receiver = topology_kernels.reflection_receiver_candidates(
            tx_positions=tx_positions.to(dtype=torch.float32).contiguous(),
            rx_positions=rx_positions.to(dtype=torch.float32).contiguous(),
            sequences=sequences,
            sequence_tx=sequence_tx,
            plane_point=groups.plane_point,
            plane_normal=groups.plane_normal,
            bounds=groups.bounds,
            occupancy=groups.occupancy,
            sequence_start=start,
            sequence_end=min(start + sequences_per_launch, sequence_count),
            box_tolerance=box,
            segment_tolerance=_SEGMENT_PARAMETER_SLACK,
            min_segment_length=min_segment,
        )
        if record_candidate_count is not None:
            record_candidate_count(int(candidate_sequence.shape[0]))
        yield from _epc_requests(
            depth,
            tx_positions=tx_positions,
            rx_positions=rx_positions,
            sequences=face_sequences,
            tri_a=tri_a,
            normals=normals,
            sequence_tx=sequence_tx,
            candidate_sequence=candidate_sequence,
            candidate_receiver=candidate_receiver,
        )


class _DepthSources:
    """Picks and produces each depth's candidate sequences, depth by depth from depth 1."""

    def __init__(
        self, plan: ReflectionDiscoveryPlan, tx_positions: torch.Tensor, rx_count: int,
        trace_group_chains: TraceReflectionGroupChains,
    ) -> None:
        self._plan = plan
        self._tx_positions = tx_positions
        self._rx_count = rx_count
        self._trace = trace_group_chains
        self._rays: _RayChains | None = None
        self._frontier = _image_tree_roots(tx_positions)
        self.rays_from = 1 if plan.discovery == SOURCE_RAYS else None

    def advance(self, depth: int, *, need_tree: bool) -> str:
        """Grow the tree to ``depth`` (unless rays already own it) and name the depth's source."""

        plan = self._plan
        if self.rays_from is None and need_tree:
            self._grow(depth)
        if self.rays_from is not None:
            return SOURCE_RAYS
        return SOURCE_EXHAUSTIVE if plan.discovery == SOURCE_EXHAUSTIVE else SOURCE_IMAGE_TREE

    def _grow(self, depth: int) -> None:
        plan = self._plan
        auto = plan.discovery == "auto"
        expansion = self._frontier.size * plan.groups.group_count
        limit = _frontier_node_limit(depth, self._tx_positions.device)
        if auto and expansion > _COMPLETE_PAIR_BUDGET:
            self.rays_from = depth
            return
        if plan.discovery == SOURCE_EXHAUSTIVE and expansion > limit:
            raise ValueError(
                f"discovery='exhaustive' needs {expansion} plane-group sequences at reflection "
                f"depth {depth}, more than the {limit} that fit in device memory; "
                "use discovery='auto'"
            )
        self._frontier = _expand_image_tree(
            self._frontier, plan.groups, prune=plan.discovery != SOURCE_EXHAUSTIVE
        )
        size = self._frontier.size
        if auto and (size * self._rx_count > _COMPLETE_PAIR_BUDGET or size > limit):
            self.rays_from = depth

    def sequences(self, depth: int) -> tuple[torch.Tensor, torch.Tensor]:
        if self.rays_from is None:
            return self._frontier.node_sequences, self._frontier.node_tx
        if self._rays is None:
            self._rays = _RayChains(
                self._tx_positions, self._plan.groups, self._plan.max_depth, self._trace
            )
        return self._rays.sequences(depth)


def iter_reflection_epc_requests(
    plan: ReflectionDiscoveryPlan, *, tx_positions: torch.Tensor, rx_positions: torch.Tensor,
    tri_a: torch.Tensor, normals: torch.Tensor, trace_group_chains: TraceReflectionGroupChains,
    record: ReflectionDiscoveryRecord | None = None,
    record_candidate_count: RecordReflectionCandidateCount | None = None,
) -> Iterator[ReflectionEpcRequest]:
    """EPC requests of depths ``min_depth..max_depth`` for every transmitter.

    Each depth's candidate source is chosen from that depth's own cost alone, so the rows of a
    depth do not depend on ``max_depth``. Every complete source yields the same valid rows.
    """

    groups = plan.groups
    tx_count = int(tx_positions.shape[0])
    rx_count = int(rx_positions.shape[0])
    if groups.group_count <= 0 or rx_count <= 0 or tx_count <= 0 or plan.max_depth < plan.min_depth:
        return
    # Depth 1 of a small product pairs every plane group with every endpoint pair directly.
    shared_pairs = tx_count * groups.group_count * rx_count
    shared_first_depth = (
        shared_pairs <= _RECEIVER_TEST_MIN_PAIRS and groups.max_group_size <= _DIRECT_GROUP_SIZE
    )
    sources = _DepthSources(plan, tx_positions, rx_count, trace_group_chains)
    for depth in range(1, plan.max_depth + 1):
        source = sources.advance(
            depth, need_tree=not (depth == 1 and shared_first_depth and plan.max_depth == 1)
        )
        if record is not None:
            record.sources[depth] = source
        if depth < plan.min_depth:
            continue
        if depth == 1 and shared_first_depth and source != SOURCE_RAYS:
            if record_candidate_count is not None:
                record_candidate_count(shared_pairs)
            yield from _epc_requests(
                1,
                tx_positions=tx_positions,
                rx_positions=rx_positions,
                sequences=groups.representative_faces.reshape(-1, 1),
                tri_a=tri_a,
                normals=normals,
            )
            continue
        sequences, sequence_tx = sources.sequences(depth)
        yield from _candidate_requests(
            depth,
            sequences=sequences,
            sequence_tx=sequence_tx,
            groups=groups,
            tx_positions=tx_positions,
            rx_positions=rx_positions,
            tri_a=tri_a,
            normals=normals,
            record_candidate_count=record_candidate_count,
        )


@dataclass(frozen=True, slots=True)
class ReflectionEpcQuery:
    rayd: RayDSceneResource
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
    # Shortest valid segment between two interactions, metres; 0 accepts every segment length.
    min_segment_length: float = 0.0
    # The receiver is an interaction (a scattering vertex on a surface) rather than an endpoint, so
    # the minimum also covers the last segment.
    receiver_is_interaction: bool = False


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
        float(query.min_segment_length),
        bool(query.receiver_is_interaction),
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
    rayd: RayDSceneResource, tx_positions: torch.Tensor, *, face_group_id: torch.Tensor,
    max_depth: int, ray_count: int | None = None,
) -> torch.Tensor:
    """Trace specular chains from every transmitter and map them to plane groups.

    Returns a ``(T * N, max_depth)`` long tensor of plane-group ids per bounce, transmitter
    major, with -1 past each ray's last hit. Only chains reachable from a transmitter can host
    a valid specular path, so validating the unique chains found here replaces the exhaustive
    plane-sequence product on large scenes. One trace launch covers every transmitter.
    """

    device = face_group_id.device
    ray_count = _MULTIBOUNCE_DISCOVERY_RAYS if ray_count is None else int(ray_count)
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


def _reflection_geometry_tables(
    compiled: CompiledScene, device: torch.device,
) -> tuple[torch.Tensor, ...]:
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
    requests: Iterator[ReflectionEpcRequest], *, rayd: RayDSceneResource,
    groups: dict[str, torch.Tensor], face_material_id: torch.Tensor, on_epc: Callable[[], None],
    min_segment_length: float,
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
                min_segment_length=min_segment_length,
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


def _reflection_topology(
    scene: Scene, compiled: CompiledScene, tx_positions: torch.Tensor, rx_positions: torch.Tensor,
    *, min_depth: int, max_depth: int, discovery: str, record: ReflectionDiscoveryRecord | None,
) -> tuple[dict[str, torch.Tensor], int, int]:
    """Reflection topology of depths ``min_depth..max_depth`` for every transmitter and receiver.

    The rows carry EPC geometry and path length only; the field stage evaluates their fields.
    """

    device = tx_positions.device
    rayd = compiled.rayd
    if not scene.structures or tx_positions.numel() == 0 or rx_positions.numel() == 0:
        return _empty_reflection_block(device), 0, 0
    if not rayd.available:
        raise RuntimeError("reflection discovery requires RayD native scene capability")
    faces, normals, tri_a = _reflection_geometry_tables(compiled, device)
    if int(faces.shape[0]) == 0 or max_depth < min_depth:
        return _empty_reflection_block(device), 0, 0

    face_material_id = face_material_id_table(compiled, device=device)
    # One candidate per coplanar plane group: a wall meshed from several coplanar triangles
    # yields one specular path, and the EPC validator resolves the containing triangle.
    groups = _cached_coplanar_face_groups(
        rayd,
        tri_a,
        normals,
        face_geometry_id_table(compiled, "face_surface_id", device=device),
    )
    plane_groups = reflection_plane_groups(compiled, groups, tri_a=tri_a, normals=normals)
    if plane_groups.group_count <= 0:
        return _empty_reflection_block(device), 0, 0
    plan = prepare_reflection_discovery_plan(
        plane_groups, min_depth=min_depth, max_depth=max_depth, discovery=discovery
    )
    launch_count = 0
    candidate_count = 0

    def trace_group_chains(
        tx: torch.Tensor, *, face_group_id: torch.Tensor, max_depth: int,
    ) -> torch.Tensor:
        nonlocal launch_count
        chains = _discovered_group_chains(
            rayd, tx, face_group_id=face_group_id, max_depth=max_depth
        )
        launch_count += 1
        return chains

    def record_candidate_count(count: int) -> None:
        nonlocal candidate_count
        candidate_count += count

    def count_epc() -> None:
        nonlocal launch_count
        launch_count += 1

    blocks = _reflection_blocks(
        iter_reflection_epc_requests(
            plan,
            tx_positions=tx_positions,
            rx_positions=rx_positions,
            tri_a=tri_a,
            normals=normals,
            trace_group_chains=trace_group_chains,
            record=record,
            record_candidate_count=record_candidate_count,
        ),
        rayd=rayd,
        groups=groups,
        face_material_id=face_material_id,
        on_epc=count_epc,
        min_segment_length=reflection_min_segment_length(plane_groups),
    )
    return (
        _ensure_topology_fields(concatenate_path_blocks(blocks, device=device)),
        launch_count,
        candidate_count,
    )
