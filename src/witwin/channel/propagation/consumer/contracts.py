"""Stable solver-neutral propagation consumer contracts.

This module is the single source of truth for the consumer vocabulary. The
accepted component, response, topology, and AD-mode values are declared here as
``Literal`` aliases with matching frozen sets, and :func:`capabilities` returns
the frozen capability record. A consumer can therefore discover what the
contract supports before building a request instead of learning it from a
rejected call.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol, TypeAlias, get_args

import torch

from witwin.channel.constants import (
    NARROWBAND_FREQUENCY_OFFSET_LAW,
    PHASOR,
    TIME_DEPENDENCE,
)


CONTRACT_VERSION = 4


PropagationComponent: TypeAlias = Literal[
    "los", "reflection", "transmission", "diffraction"
]
PropagationResponse: TypeAlias = Literal[
    "scalar_transport", "complex3_transport", "polarimetric_transport"
]
PropagationTopologyMode: TypeAlias = Literal["discover"]
PropagationAdMode: TypeAlias = Literal["none", "jvp", "vjp"]
PropagationWorldMotion: TypeAlias = Literal["frozen_world", "fixed_winner_replay"]

COMPONENTS: frozenset[str] = frozenset(get_args(PropagationComponent))
RESPONSES: frozenset[str] = frozenset(get_args(PropagationResponse))
TOPOLOGY_MODES: frozenset[str] = frozenset(get_args(PropagationTopologyMode))
AD_MODES: frozenset[str] = frozenset(get_args(PropagationAdMode))
WORLD_MOTIONS: frozenset[str] = frozenset(get_args(PropagationWorldMotion))

MAX_DEPTH = 5

# The four witwin.core version domains, in the order a freshness check reports
# them. Geometry is last because it is the one domain a caller can declare
# tolerable: a rigid motion or a deformation renumbers nothing, so a frozen
# winner label still means what it meant, while a topology, material, or
# assignment change respecifies the very labels the frozen rows carry.
WORLD_VERSION_DOMAINS: tuple[str, ...] = (
    "topology_version",
    "material_version",
    "assignment_version",
    "geometry_version",
)

# Native component identifiers a frozen topology may carry into reevaluation.
# The names are the contract vocabulary above; the integers are the discovery
# owner's encoding, which the consumer reads but does not define.
_FIXED_TOPOLOGY_COMPONENT_IDS: tuple[tuple[str, int], ...] = (
    ("los", 0),
    ("reflection", 1),
)


def _require_tensor(
    name: str,
    value: object,
    *,
    dtype: torch.dtype,
    shape: tuple[int, ...] | None = None,
    ndim: int | None = None,
    device: torch.device | None = None,
) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if value.dtype != dtype:
        raise TypeError(f"{name} must use {dtype}, got {value.dtype}")
    if shape is not None and tuple(value.shape) != shape:
        raise ValueError(f"{name} must have shape {shape}, got {tuple(value.shape)}")
    if ndim is not None and value.ndim != ndim:
        raise ValueError(f"{name} must have rank {ndim}, got {value.ndim}")
    if device is not None and value.device != device:
        raise ValueError(f"{name} must be on {device}, got {value.device}")
    return value


class _WorldVersionSource(Protocol):
    """What a freshness check reads. Structural, so it never imports a scene."""

    topology_version: int
    geometry_version: int
    material_version: int
    assignment_version: int


@dataclass(frozen=True, slots=True, eq=False)
class WorldProvenance:
    """Which world a discovered topology belongs to (ADR-040).

    The four integers are the ``witwin.core`` version domains the compiled
    scene was built from. They are content hashes, so equal versions mean equal
    world content, which is exactly the condition under which replaying a
    frozen topology is numerically meaningful. Comparison is by domain, never
    by object identity.

    ``time_s`` is the compiled snapshot instant. It is carried for reporting
    and cross-consumer correlation only: it is never compared and never gates
    a call, because two different instants of a static world are the same
    world.
    """

    topology_version: int
    geometry_version: int
    material_version: int
    assignment_version: int
    time_s: float | torch.Tensor | None = None

    @classmethod
    def of(cls, compiled: _WorldVersionSource) -> WorldProvenance:
        """Read the four version domains a compiled scene recorded."""

        return cls(
            topology_version=int(compiled.topology_version),
            geometry_version=int(compiled.geometry_version),
            material_version=int(compiled.material_version),
            assignment_version=int(compiled.assignment_version),
            time_s=getattr(compiled, "time_s", None),
        )

    def moved_domain(
        self, current: WorldProvenance, *, allow_geometry: bool = False
    ) -> str | None:
        """Name the first version domain that differs, or ``None``.

        Four host integer comparisons. No device work, no allocation, no
        synchronization.
        """

        for name in WORLD_VERSION_DOMAINS:
            if allow_geometry and name == "geometry_version":
                continue
            if getattr(self, name) != getattr(current, name):
                return name
        return None


@dataclass(frozen=True, slots=True, eq=False)
class EndpointBatch:
    """Explicit point endpoints for one propagation request.

    ``polarization_basis`` contains two world-Cartesian endpoint reference
    vectors. The native Jones producer projects and orthonormalizes them for
    each path direction and returns the actual row-aligned transverse bases.
    ``powers_w`` is required for a source batch and must be absent from a sink
    batch.
    """

    stable_ids: torch.Tensor
    positions_m: torch.Tensor
    polarizations: torch.Tensor
    polarization_basis: torch.Tensor | None = None
    powers_w: torch.Tensor | None = None

    def __post_init__(self) -> None:
        positions = _require_tensor(
            "positions_m", self.positions_m, dtype=torch.float32, ndim=2
        )
        if positions.shape[1:] != (3,):
            raise ValueError("positions_m must have shape (N, 3)")
        if positions.device.type != "cuda":
            raise ValueError("endpoint tensors must be CUDA tensors")
        rows = int(positions.shape[0])
        device = positions.device
        _require_tensor(
            "stable_ids",
            self.stable_ids,
            dtype=torch.int64,
            shape=(rows,),
            device=device,
        )
        _require_tensor(
            "polarizations",
            self.polarizations,
            dtype=torch.float32,
            shape=(rows, 3),
            device=device,
        )
        if self.polarization_basis is not None:
            _require_tensor(
                "polarization_basis",
                self.polarization_basis,
                dtype=torch.float32,
                shape=(rows, 2, 3),
                device=device,
            )
        if self.powers_w is not None:
            _require_tensor(
                "powers_w",
                self.powers_w,
                dtype=torch.float32,
                shape=(rows,),
                device=device,
            )
        for name, value in (
            ("stable_ids", self.stable_ids),
            ("positions_m", self.positions_m),
            ("polarizations", self.polarizations),
            ("polarization_basis", self.polarization_basis),
            ("powers_w", self.powers_w),
        ):
            if value is not None and not value.is_contiguous():
                raise ValueError(f"{name} must be contiguous")

    @property
    def count(self) -> int:
        return int(self.positions_m.shape[0])

    @property
    def device(self) -> torch.device:
        return self.positions_m.device


def _require_vocabulary(name: str, value: object, allowed: frozenset[str]) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    if value not in allowed:
        raise NotImplementedError(
            f"unsupported {name} {value!r}; supported values are {sorted(allowed)}"
        )
    return value


def _require_components(value: object) -> frozenset[str]:
    if type(value) is not frozenset or not value:
        raise TypeError("components must be a non-empty frozenset")
    unsupported = value - COMPONENTS
    if unsupported:
        raise NotImplementedError(
            f"unsupported propagation components: {sorted(unsupported)}"
        )
    return value


def _require_endpoints(sources: object, sinks: object) -> None:
    if not isinstance(sources, EndpointBatch) or not isinstance(sinks, EndpointBatch):
        raise TypeError("sources and sinks must be EndpointBatch instances")
    if sources.powers_w is None:
        raise ValueError("sources.powers_w is required")
    if sinks.powers_w is not None:
        raise ValueError("sinks.powers_w must be absent")
    if sources.device != sinks.device:
        raise ValueError("source and sink batches must share one CUDA device")


@dataclass(frozen=True, slots=True)
class PropagationRequest:
    """A discovery request against a compiled scene.

    Structural validity is enforced here. Capability compatibility that depends
    on the compiled scene  -  reference-frequency match, response/component and
    response/AD combinations, polarimetric basis requirements  -  is enforced by
    :func:`witwin.channel.propagation.consumer.evaluate` before any native work.
    """

    sources: EndpointBatch
    sinks: EndpointBatch
    reference_frequency_hz: float | torch.Tensor
    components: frozenset[str]
    max_depth: int
    response: PropagationResponse
    topology_mode: PropagationTopologyMode
    ad_mode: PropagationAdMode
    max_paths: int | None = None

    def __post_init__(self) -> None:
        _require_endpoints(self.sources, self.sinks)
        _require_components(self.components)
        _require_vocabulary("response", self.response, RESPONSES)
        _require_vocabulary("topology_mode", self.topology_mode, TOPOLOGY_MODES)
        _require_vocabulary("ad_mode", self.ad_mode, AD_MODES)
        if type(self.max_depth) is not int or not 0 <= self.max_depth <= MAX_DEPTH:
            raise ValueError(f"max_depth must be an int in [0, {MAX_DEPTH}]")
        if self.max_paths is not None and (
            type(self.max_paths) is not int or self.max_paths <= 0
        ):
            raise ValueError("max_paths must be a positive int when set")


@dataclass(frozen=True, slots=True, eq=False)
class PropagationTopology:
    """Compact discrete rows in stable pair-major order.

    ``provenance`` records which world these rows were discovered against.
    :func:`witwin.channel.propagation.consumer.evaluate` stamps it,
    :func:`prepare_fixed_topology` forwards it verbatim, and
    :func:`witwin.channel.propagation.consumer.reevaluate` refuses a frozen
    replay against a world that moved out from under it. It is ``None`` on a
    hand-built topology, which has no world to be stale against.
    """

    source_index: torch.Tensor
    sink_index: torch.Tensor
    source_id: torch.Tensor
    sink_id: torch.Tensor
    depth: torch.Tensor
    component_id: torch.Tensor
    primitive_id: torch.Tensor
    edge_id: torch.Tensor
    material_id: torch.Tensor
    primitive_sequence: torch.Tensor
    material_sequence: torch.Tensor
    interaction_type: torch.Tensor
    provenance: WorldProvenance | None = None

    def __post_init__(self) -> None:
        if self.provenance is not None and not isinstance(
            self.provenance, WorldProvenance
        ):
            raise TypeError("provenance must be a WorldProvenance or None")
        source_index = _require_tensor(
            "source_index", self.source_index, dtype=torch.int32, ndim=1
        )
        rows = int(source_index.shape[0])
        device = source_index.device
        for name in (
            "sink_index",
            "depth",
            "component_id",
            "primitive_id",
            "edge_id",
            "material_id",
        ):
            _require_tensor(
                name,
                getattr(self, name),
                dtype=torch.int32,
                shape=(rows,),
                device=device,
            )
        for name in ("source_id", "sink_id"):
            _require_tensor(
                name,
                getattr(self, name),
                dtype=torch.int64,
                shape=(rows,),
                device=device,
            )
        primitive_sequence = _require_tensor(
            "primitive_sequence",
            self.primitive_sequence,
            dtype=torch.int32,
            ndim=2,
            device=device,
        )
        sequence_shape = tuple(primitive_sequence.shape)
        if sequence_shape[0] != rows:
            raise ValueError("primitive_sequence must share compact rows")
        for name in ("material_sequence", "interaction_type"):
            _require_tensor(
                name,
                getattr(self, name),
                dtype=torch.int32,
                shape=sequence_shape,
                device=device,
            )

    @property
    def row_count(self) -> int:
        return int(self.source_index.shape[0])

    @property
    def device(self) -> torch.device:
        return self.source_index.device


@dataclass(frozen=True, slots=True, eq=False)
class PropagationGeometry:
    """Continuous SI geometry aligned with one compact topology."""

    path_length_m: torch.Tensor
    delay_s: torch.Tensor
    field_direction: torch.Tensor
    interaction_positions_m: torch.Tensor
    interaction_normals: torch.Tensor

    def __post_init__(self) -> None:
        path_length = _require_tensor(
            "path_length_m", self.path_length_m, dtype=torch.float32, ndim=1
        )
        rows = int(path_length.shape[0])
        device = path_length.device
        _require_tensor(
            "delay_s",
            self.delay_s,
            dtype=torch.float32,
            shape=(rows,),
            device=device,
        )
        _require_tensor(
            "field_direction",
            self.field_direction,
            dtype=torch.float32,
            shape=(rows, 3),
            device=device,
        )
        positions = _require_tensor(
            "interaction_positions_m",
            self.interaction_positions_m,
            dtype=torch.float32,
            ndim=3,
            device=device,
        )
        if positions.shape[0] != rows or positions.shape[2] != 3:
            raise ValueError("interaction_positions_m must have shape (K, depth, 3)")
        _require_tensor(
            "interaction_normals",
            self.interaction_normals,
            dtype=torch.float32,
            shape=tuple(positions.shape),
            device=device,
        )

    @property
    def row_count(self) -> int:
        return int(self.path_length_m.shape[0])

    @property
    def device(self) -> torch.device:
        return self.path_length_m.device


@dataclass(frozen=True, slots=True, eq=False)
class ScalarTransport:
    """Endpoint-projected complex scalar transport at the reference frequency.

    The coefficient carries the declared source amplitude
    ``sqrt(sources.powers_w)`` of the transmitting endpoint, so it is a
    transported field value and not a unit-excitation transfer function
    (ADR-039). Power/gain values are its squared magnitude.
    """

    coefficient: torch.Tensor

    def __post_init__(self) -> None:
        _require_tensor("coefficient", self.coefficient, dtype=torch.complex64, ndim=1)

    @property
    def row_count(self) -> int:
        return int(self.coefficient.shape[0])

    @property
    def device(self) -> torch.device:
        return self.coefficient.device


@dataclass(frozen=True, slots=True, eq=False)
class Complex3Transport:
    """World-Cartesian complex electric field and propagation direction.

    The field carries the declared source amplitude
    ``sqrt(sources.powers_w)`` of the transmitting endpoint, so projecting it
    onto the receive polarization reproduces
    :class:`ScalarTransport.coefficient` (ADR-039).
    """

    field: torch.Tensor
    direction: torch.Tensor

    def __post_init__(self) -> None:
        field = _require_tensor("field", self.field, dtype=torch.complex64, ndim=2)
        if field.shape[1:] != (3,):
            raise ValueError("field must have shape (K, 3)")
        _require_tensor(
            "direction",
            self.direction,
            dtype=torch.float32,
            shape=(int(field.shape[0]), 3),
            device=field.device,
        )

    @property
    def row_count(self) -> int:
        return int(self.field.shape[0])

    @property
    def device(self) -> torch.device:
        return self.field.device


@dataclass(frozen=True, slots=True, eq=False)
class JonesTransport:
    """Complete source-basis to sink-basis complex 2 x 2 operator.

    The operator excludes transmitter power and endpoint antenna-pattern
    factors; those are not part of a linear polarization-basis map. This is
    deliberately unlike :class:`ScalarTransport` and :class:`Complex3Transport`,
    which publish the excited transport: a caller that wants a powered response
    applies ``sqrt(powers_w)`` to the source-basis excitation itself.
    """

    matrix: torch.Tensor
    source_basis: torch.Tensor
    sink_basis: torch.Tensor

    def __post_init__(self) -> None:
        matrix = _require_tensor("matrix", self.matrix, dtype=torch.complex64, ndim=3)
        if matrix.shape[1:] != (2, 2):
            raise ValueError("matrix must have shape (K, 2, 2)")
        rows = int(matrix.shape[0])
        for name in ("source_basis", "sink_basis"):
            _require_tensor(
                name,
                getattr(self, name),
                dtype=torch.float32,
                shape=(rows, 2, 3),
                device=matrix.device,
            )

    @property
    def row_count(self) -> int:
        return int(self.matrix.shape[0])

    @property
    def device(self) -> torch.device:
        return self.matrix.device


PropagationTransport: TypeAlias = ScalarTransport | Complex3Transport | JonesTransport


@dataclass(frozen=True, slots=True, eq=False)
class PropagationPathBatch:
    """Actual compact ``K`` rows with native-produced pair segmentation."""

    pair_count: int
    path_count: int
    pair_index: torch.Tensor
    pair_offsets: torch.Tensor
    topology: PropagationTopology
    geometry: PropagationGeometry
    transport: PropagationTransport

    def __post_init__(self) -> None:
        if type(self.pair_count) is not int or self.pair_count < 0:
            raise ValueError("pair_count must be a non-negative int")
        if type(self.path_count) is not int or self.path_count < 0:
            raise ValueError("path_count must be a non-negative int")
        pair_index = _require_tensor(
            "pair_index",
            self.pair_index,
            dtype=torch.int64,
            shape=(self.path_count,),
        )
        _require_tensor(
            "pair_offsets",
            self.pair_offsets,
            dtype=torch.int64,
            shape=(self.pair_count + 1,),
            device=pair_index.device,
        )
        if self.topology.row_count != self.path_count:
            raise ValueError("topology must have exactly path_count rows")
        if self.geometry.row_count != self.path_count:
            raise ValueError("geometry must have exactly path_count rows")
        if self.transport.row_count != self.path_count:
            raise ValueError("transport must have exactly path_count rows")
        if self.topology.device != pair_index.device:
            raise ValueError("all path tensors must share one device")
        if self.geometry.device != pair_index.device:
            raise ValueError("all path tensors must share one device")
        if self.transport.device != pair_index.device:
            raise ValueError("all path tensors must share one device")


@dataclass(frozen=True, slots=True)
class PropagationConvention:
    contract_version: int = CONTRACT_VERSION
    pair_layout: str = (
        "sink_major_source_minor:"
        "pair_index=sink_index*source_count+source_index"
    )
    # The block-diagonal law a slot-batched reevaluation adds on top of
    # ``pair_layout``, which it does not redefine. Slots never cross-pair: slot
    # ``t`` pairs only the sources and sinks of slot ``t``, so ``pair_count``
    # grows linearly in ``slot_count`` instead of quadratically, and the
    # sink-major layout above is preserved exactly within one slot.
    slot_pair_layout: str = (
        "block_diagonal_slots:"
        "row=slot*frozen_row_count+frozen_row;"
        "source_index=slot*slot_source_count+slot_source_index;"
        "sink_index=slot*slot_sink_count+slot_sink_index;"
        "pair_index=slot*slot_source_count*slot_sink_count"
        "+slot_sink_index*slot_source_count+slot_source_index;"
        "pair_count=slot_count*slot_source_count*slot_sink_count"
    )
    distance_unit: str = "m"
    delay_unit: str = "s"
    phasor: str = PHASOR
    time_dependence: str = TIME_DEPENDENCE
    coefficient_reference: str = "includes_reference_frequency_phase"
    # Stated so a caller can shift off the reference frequency with the correct
    # sign. This contract does not apply it and has no frequency-offset input;
    # `PropagationGeometry.delay_s` is published per row for exactly this use.
    narrowband_frequency_offset_law: str = NARROWBAND_FREQUENCY_OFFSET_LAW
    complex3_basis: str = "world_cartesian"
    jones_mapping: str = "source_transverse_basis_to_sink_transverse_basis"


@dataclass(frozen=True, slots=True)
class PropagationCapabilities:
    contract_version: int
    components: frozenset[str]
    responses: frozenset[str]
    topology_modes: frozenset[str]
    ad_modes: frozenset[str]
    # What a FixedTopologyRequest may declare about world motion between
    # discovery and replay. See FixedTopologyRequest.world_motion.
    world_motions: frozenset[str]
    # The version domains a frozen replay is checked against, in the order a
    # mismatch is reported.
    world_version_domains: tuple[str, ...]
    response_components: tuple[tuple[str, frozenset[str]], ...]
    response_ad_modes: tuple[tuple[str, frozenset[str]], ...]
    component_ad_modes: tuple[tuple[str, frozenset[str]], ...]
    fixed_topology_components: frozenset[str]
    fixed_topology_responses: frozenset[str]
    supports_fixed_topology: bool
    supports_los_jones: bool
    # Components whose fixed-topology rows can stop existing at new endpoint
    # positions and are therefore published with a per-row validity mask
    # instead of failing the whole batch (ADR-037).
    fixed_topology_row_validity_components: frozenset[str]
    # Inputs the composed Jones operator consumes as primal-only constants.
    # The native field companions reject gradients on them by contract, so a
    # differentiable request carrying one of these fails before native work.
    polarimetric_frozen_ad_inputs: tuple[str, ...]
    # Whether a FixedTopologyRequest may declare slot_count > 1 and replay a
    # whole block of world instants in one launch per bucket. The pairing law
    # is PropagationConvention.slot_pair_layout.
    supports_slot_batching: bool
    # A contract bound on slot_count, or None when the only bound is device
    # memory. Slot batching adds no per-slot host observation, no per-slot
    # launch, and no per-slot synchronization, so nothing grows with the slot
    # count except the row and pair tensors themselves.
    max_slot_count: int | None

    def components_for(self, response: str) -> frozenset[str]:
        return dict(self.response_components)[response]

    def ad_modes_for(self, response: str) -> frozenset[str]:
        return dict(self.response_ad_modes)[response]

    def ad_modes_for_component(self, component: str) -> frozenset[str]:
        return dict(self.component_ad_modes)[component]


_CAPABILITIES = PropagationCapabilities(
    contract_version=CONTRACT_VERSION,
    components=COMPONENTS,
    responses=RESPONSES,
    topology_modes=TOPOLOGY_MODES,
    ad_modes=AD_MODES,
    world_motions=WORLD_MOTIONS,
    world_version_domains=WORLD_VERSION_DOMAINS,
    response_components=(
        ("scalar_transport", COMPONENTS),
        ("complex3_transport", COMPONENTS),
        ("polarimetric_transport", frozenset({"los"})),
    ),
    response_ad_modes=(
        ("scalar_transport", AD_MODES),
        ("complex3_transport", AD_MODES),
        ("polarimetric_transport", AD_MODES),
    ),
    component_ad_modes=tuple((component, AD_MODES) for component in sorted(COMPONENTS)),
    fixed_topology_components=frozenset(
        name for name, _ in _FIXED_TOPOLOGY_COMPONENT_IDS
    ),
    fixed_topology_responses=frozenset(
        {"scalar_transport", "complex3_transport", "polarimetric_transport"}
    ),
    supports_fixed_topology=True,
    supports_los_jones=True,
    fixed_topology_row_validity_components=frozenset({"los", "reflection"}),
    polarimetric_frozen_ad_inputs=(
        "tx_power",
        "mu_r",
        "source_basis",
        "sink_basis",
        "endpoint_polarizations",
    ),
    supports_slot_batching=True,
    max_slot_count=None,
)


def capabilities() -> PropagationCapabilities:
    """Return what this consumer contract version supports.

    Call this before building a :class:`PropagationRequest` to check that a
    component, response, and AD-mode combination is available. The record is
    frozen and identical across calls.
    """

    return _CAPABILITIES


@dataclass(frozen=True, slots=True)
class PropagationDiagnostics:
    discovery_launch_count: int
    candidate_count: int
    visibility_rejection_count: int
    compact_count_d2h_copies: int
    compact_count_d2h_bytes: int
    compact_sync_count: int
    validation_d2h_copies: int
    validation_d2h_bytes: int
    validation_sync_count: int


@dataclass(frozen=True, slots=True, eq=False)
class PropagationEvaluation:
    paths: PropagationPathBatch
    convention: PropagationConvention
    capabilities: PropagationCapabilities
    diagnostics: PropagationDiagnostics


@dataclass(frozen=True, slots=True, eq=False)
class FixedTopologyBucket:
    """One host-known ``(component, depth)`` partition of a frozen topology.

    ``rows`` holds ascending indices into the frozen ``K`` rows, so a bucket
    preserves frozen row order. The component name and the depth are host
    integers determined once, at freeze time.
    """

    component: str
    depth: int
    rows: torch.Tensor

    def __post_init__(self) -> None:
        _require_vocabulary("component", self.component, COMPONENTS)
        if type(self.depth) is not int or not 0 <= self.depth <= MAX_DEPTH:
            raise ValueError(f"depth must be an int in [0, {MAX_DEPTH}]")
        _require_tensor("rows", self.rows, dtype=torch.int64, ndim=1)

    @property
    def row_count(self) -> int:
        return int(self.rows.shape[0])


@dataclass(frozen=True, slots=True, eq=False)
class PreparedFixedTopology:
    """A frozen topology partitioned once for repeated reevaluation.

    Reflection field transport takes one uniform interaction depth per native
    launch, so a mixed-depth frozen batch has to be partitioned before it can
    be reevaluated. That partition is a property of the frozen topology alone,
    so it is computed once here and reused by every later call.

    The recorded host-observation counters describe THIS construction, not a
    later :func:`witwin.channel.propagation.consumer.reevaluate` call. They are
    deliberately not folded into per-call diagnostics: preparing the handle
    once per frozen topology is the contract, and preparing it per frame gives
    up the whole point of the capability.
    """

    topology: PropagationTopology
    buckets: tuple[FixedTopologyBucket, ...]
    prepare_d2h_copies: int
    prepare_d2h_bytes: int
    prepare_synchronizations: int

    @property
    def row_count(self) -> int:
        return self.topology.row_count

    @property
    def device(self) -> torch.device:
        return self.topology.device

    @property
    def provenance(self) -> WorldProvenance | None:
        """The world the frozen rows were discovered against, forwarded."""

        return self.topology.provenance


def _fixed_topology_component_name(component_id: int) -> str:
    for name, value in _FIXED_TOPOLOGY_COMPONENT_IDS:
        if value == component_id:
            return name
    raise NotImplementedError(
        f"fixed-topology reevaluation does not support component id "
        f"{component_id}; supported components are "
        f"{sorted(_CAPABILITIES.fixed_topology_components)}"
    )


def _require_bucket_depth(component: str, depth: int) -> None:
    if component == "los" and depth != 0:
        raise ValueError("a los row must have depth 0")
    if component == "reflection" and depth < 1:
        raise ValueError("a reflection row must have depth >= 1")


def _malformed_rows(topology: PropagationTopology, width: int) -> torch.Tensor:
    """Rows whose depth or interaction padding contradicts the contract."""

    depth = topology.depth.to(dtype=torch.int64)
    slots = torch.arange(
        width, device=topology.device, dtype=torch.int64
    ).reshape(1, -1)
    active = slots < depth.reshape(-1, 1)
    sequence = topology.primitive_sequence.to(dtype=torch.int64)
    return (
        (depth < 0)
        | (depth > width)
        | ((sequence < 0) & active).any(dim=1)
        | ((sequence != -1) & ~active).any(dim=1)
    )


def prepare_fixed_topology(
    topology: PropagationTopology,
) -> PreparedFixedTopology:
    """Partition a frozen topology by component and interaction depth.

    This is the one place the consumer looks at a frozen topology on the host.
    It validates the depth/interaction padding of every row, rejects any
    component outside ``capabilities().fixed_topology_components``, and returns
    the ascending ``(component, depth)`` buckets that
    :func:`witwin.channel.propagation.consumer.reevaluate` replays.

    Call it once per frozen topology and reuse the handle. It synchronizes; a
    per-frame call would reintroduce exactly the host observation the fixed
    topology capability exists to avoid.
    """

    if not isinstance(topology, PropagationTopology):
        raise TypeError("topology must be a PropagationTopology")
    if topology.row_count == 0:
        return PreparedFixedTopology(
            topology=topology,
            buckets=(),
            prepare_d2h_copies=0,
            prepare_d2h_bytes=0,
            prepare_synchronizations=0,
        )
    width = int(topology.primitive_sequence.shape[1])
    if bool(_malformed_rows(topology, width).any().item()):
        raise ValueError(
            "frozen topology rows disagree with their interaction sequence "
            "padding; depth must be in [0, sequence width] and unused slots "
            "must hold -1"
        )
    key = topology.component_id.to(dtype=torch.int64) * (width + 1) + (
        topology.depth.to(dtype=torch.int64)
    )
    distinct = torch.unique(key).tolist()
    buckets = []
    for value in distinct:
        component_id, depth = divmod(int(value), width + 1)
        component = _fixed_topology_component_name(component_id)
        _require_bucket_depth(component, depth)
        buckets.append(
            FixedTopologyBucket(
                component=component,
                depth=depth,
                rows=torch.nonzero(key == value, as_tuple=False).reshape(-1),
            )
        )
    return PreparedFixedTopology(
        topology=topology,
        buckets=tuple(buckets),
        prepare_d2h_copies=2 + len(buckets),
        prepare_d2h_bytes=1 + 8 * (len(distinct) + len(buckets)),
        prepare_synchronizations=2 + len(buckets),
    )


def _require_slot_divisible(name: str, count: int, slot_count: int) -> None:
    if count % slot_count:
        raise ValueError(
            f"{name} carries {count} entries, which is not a multiple of "
            f"slot_count={slot_count}; every slot must hold the same number"
        )


def _require_slot_count(slot_count: object) -> int:
    if type(slot_count) is not int or slot_count <= 0:
        raise ValueError("slot_count must be a positive int")
    return slot_count


def replicate_over_slots(
    prepared: PreparedFixedTopology,
    slot_count: int,
    *,
    source_count: int,
    sink_count: int,
) -> PreparedFixedTopology:
    """Tile a frozen topology over ``slot_count`` block-diagonal slots.

    ``source_count`` and ``sink_count`` are the PER-SLOT endpoint counts of the
    stacked batches the replicated topology will be replayed against. They are
    required rather than inferred: an endpoint that publishes no frozen row
    never appears in ``source_index``, so the largest index in a topology is
    not the endpoint count and inferring one would silently mislabel every
    later slot.

    This is pure index arithmetic and bucket re-partitioning: row ``t*K + r``
    names the same frozen row ``r`` shifted into slot ``t``, so the frozen row
    order is preserved inside every slot and the ``(component, depth)`` bucket
    COUNT is unchanged - only the bucket row counts grow. No compaction, no
    physics, no native symbol, no host observation. ``slot_count == 1`` returns
    the handle unchanged, so a single-slot replay is bit-identical to one that
    never asked for slots.

    ``provenance`` is forwarded verbatim, so a replicated topology is checked
    for staleness exactly like the topology it came from (ADR-040).
    """

    if not isinstance(prepared, PreparedFixedTopology):
        raise TypeError("prepared must be a PreparedFixedTopology")
    slot_count = _require_slot_count(slot_count)
    for name, value in (("source_count", source_count), ("sink_count", sink_count)):
        if type(value) is not int or value <= 0:
            raise ValueError(f"{name} must be a positive int")
    if slot_count == 1:
        return prepared
    topology = prepared.topology
    rows = topology.row_count
    device = topology.device
    slot = torch.arange(slot_count, device=device, dtype=torch.int32)
    replicated = PropagationTopology(
        source_index=(
            topology.source_index.repeat(slot_count)
            + slot.mul(source_count).repeat_interleave(rows)
        ),
        sink_index=(
            topology.sink_index.repeat(slot_count)
            + slot.mul(sink_count).repeat_interleave(rows)
        ),
        source_id=topology.source_id.repeat(slot_count),
        sink_id=topology.sink_id.repeat(slot_count),
        depth=topology.depth.repeat(slot_count),
        component_id=topology.component_id.repeat(slot_count),
        primitive_id=topology.primitive_id.repeat(slot_count),
        edge_id=topology.edge_id.repeat(slot_count),
        material_id=topology.material_id.repeat(slot_count),
        primitive_sequence=topology.primitive_sequence.repeat(slot_count, 1),
        material_sequence=topology.material_sequence.repeat(slot_count, 1),
        interaction_type=topology.interaction_type.repeat(slot_count, 1),
        provenance=topology.provenance,
    )
    row_offset = torch.arange(
        slot_count, device=device, dtype=torch.int64
    ).mul(rows).reshape(-1, 1)
    return PreparedFixedTopology(
        topology=replicated,
        buckets=tuple(
            FixedTopologyBucket(
                component=bucket.component,
                depth=bucket.depth,
                rows=(bucket.rows.reshape(1, -1) + row_offset).reshape(-1),
            )
            for bucket in prepared.buckets
        ),
        # Replication observes nothing, so the handle keeps the cost of the one
        # preparation it was derived from rather than claiming a new one.
        prepare_d2h_copies=prepared.prepare_d2h_copies,
        prepare_d2h_bytes=prepared.prepare_d2h_bytes,
        prepare_synchronizations=prepared.prepare_synchronizations,
    )


@dataclass(frozen=True, slots=True)
class FixedTopologyRequest:
    """A reevaluation request against an already-discovered topology.

    ``topology`` is a raw :class:`PropagationTopology` for the zero-interaction
    LoS route, or a :class:`PreparedFixedTopology` for any route that carries
    interactions. Structural validity is enforced here; scene-dependent
    capability checks belong to
    :func:`witwin.channel.propagation.consumer.reevaluate`.

    ``world_motion`` declares what the caller expects of the world between
    discovery and this replay. ``"frozen_world"`` is the safe default and
    accepts only the world the rows were discovered against.
    ``"fixed_winner_replay"`` additionally accepts a moved
    ``geometry_version`` - a translated, rotated, or deformed structure - and
    states that the caller deliberately holds the discrete winner set fixed
    while the geometry moves. It never accepts a moved topology, material, or
    assignment version: those respecify the labels the frozen rows carry.

    ``slot_count`` declares that the frozen rows and the endpoint batches are
    ``slot_count`` block-diagonal slots stacked slot-major, so one call replays
    a whole frame, pulse train, or symbol block in one launch per bucket with
    one validation copy and one synchronization for the whole set. The pairing
    law is :attr:`PropagationConvention.slot_pair_layout`; build the topology
    with :func:`replicate_over_slots`. It requires a
    :class:`PreparedFixedTopology`: the raw zero-interaction route builds its
    pair segmentation inside the native gather over the full source/sink outer
    product and therefore cannot express a block-diagonal layout.
    """

    sources: EndpointBatch
    sinks: EndpointBatch
    reference_frequency_hz: float | torch.Tensor
    topology: PropagationTopology | PreparedFixedTopology
    response: PropagationResponse
    ad_mode: PropagationAdMode
    world_motion: PropagationWorldMotion = "frozen_world"
    slot_count: int = 1

    def __post_init__(self) -> None:
        _require_endpoints(self.sources, self.sinks)
        self._require_slots()
        if not isinstance(
            self.topology, PropagationTopology | PreparedFixedTopology
        ):
            raise TypeError(
                "topology must be a PropagationTopology or a "
                "PreparedFixedTopology"
            )
        response = _require_vocabulary("response", self.response, RESPONSES)
        _require_vocabulary("ad_mode", self.ad_mode, AD_MODES)
        _require_vocabulary("world_motion", self.world_motion, WORLD_MOTIONS)
        if response not in _CAPABILITIES.fixed_topology_responses:
            raise NotImplementedError(
                f"fixed-topology reevaluation does not support {response!r}; "
                f"supported responses are "
                f"{sorted(_CAPABILITIES.fixed_topology_responses)}"
            )

    def _require_slots(self) -> None:
        """Reject a malformed slot declaration before any native work."""

        slot_count = _require_slot_count(self.slot_count)
        if slot_count == 1:
            return
        if isinstance(self.topology, PropagationTopology):
            raise NotImplementedError(
                "slot_count > 1 requires a PreparedFixedTopology; the raw "
                "zero-interaction route builds its pair segmentation in the "
                "native gather over the full source/sink outer product and "
                "cannot express the block-diagonal slot layout. Call "
                "prepare_fixed_topology first."
            )
        _require_slot_divisible("sources", self.sources.count, slot_count)
        _require_slot_divisible("sinks", self.sinks.count, slot_count)
        _require_slot_divisible(
            "topology rows", self.frozen_topology.row_count, slot_count
        )

    @property
    def frozen_topology(self) -> PropagationTopology:
        """The frozen rows, whether or not the request carries a handle."""

        if isinstance(self.topology, PreparedFixedTopology):
            return self.topology.topology
        return self.topology


@dataclass(frozen=True, slots=True, eq=False)
class FixedTopologyEvaluation:
    """A reevaluated batch of frozen rows.

    ``row_valid`` is ``None`` when every published row is valid by
    construction, which is the case for a route whose rows cannot stop
    existing. When it is present it is a CUDA ``bool`` mask over the frozen
    rows in frozen row order and it is the SOLE authority on whether a row's
    payload means anything: a geometrically valid row may legitimately carry a
    zero coefficient, so validity can never be inferred from the payload.

    It covers exactly ``fixed_topology_row_validity_components``. A frozen
    line-of-sight row IS re-tested against the passed scene with the same
    native visibility gate discovery applies, so a sink that moves behind a
    wall publishes ``row_valid=False`` and exact zeros rather than a stale
    full-strength answer. A row keeps its ORIGINAL ``primitive_sequence``
    label when the stationary point slides onto a coplanar twin triangle; the
    numbers are exact, the discrete label is stale. Both are recorded in
    ADR-037.

    **Replay is subtractive (ADR-040).** A frozen row can stop existing and is
    published as ``row_valid=False``; a path that comes into existence at the
    new endpoint or world state is NOT discovered here and is silently absent
    from the batch. The rows that are published are exactly correct, and the
    batch under-reports. There is no birth signal, by design: every candidate
    detector costs either a full discovery or a device reduction plus a host
    read the ADR-032 budget does not have. A caller whose scene can gain paths
    owns the rediscovery cadence; poll
    :func:`witwin.channel.propagation.consumer.rediscovery_required` for a
    changed world and rediscover on a motion-event cadence.
    """

    paths: PropagationPathBatch
    convention: PropagationConvention
    capabilities: PropagationCapabilities
    diagnostics: PropagationDiagnostics
    row_valid: torch.Tensor | None = None

    def __post_init__(self) -> None:
        if self.row_valid is None:
            return
        _require_tensor(
            "row_valid",
            self.row_valid,
            dtype=torch.bool,
            shape=(self.paths.path_count,),
        )


__all__ = [
    "AD_MODES",
    "COMPONENTS",
    "CONTRACT_VERSION",
    "Complex3Transport",
    "EndpointBatch",
    "FixedTopologyBucket",
    "FixedTopologyEvaluation",
    "FixedTopologyRequest",
    "JonesTransport",
    "MAX_DEPTH",
    "PreparedFixedTopology",
    "PropagationAdMode",
    "PropagationCapabilities",
    "PropagationComponent",
    "PropagationConvention",
    "PropagationDiagnostics",
    "PropagationEvaluation",
    "PropagationGeometry",
    "PropagationPathBatch",
    "PropagationRequest",
    "PropagationResponse",
    "PropagationTopology",
    "PropagationTopologyMode",
    "RESPONSES",
    "ScalarTransport",
    "TOPOLOGY_MODES",
    "WORLD_MOTIONS",
    "WORLD_VERSION_DOMAINS",
    "WorldProvenance",
    "capabilities",
    "prepare_fixed_topology",
    "replicate_over_slots",
]
