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
from typing import Literal, TypeAlias, get_args

import torch

from witwin.channel.constants import (
    NARROWBAND_FREQUENCY_OFFSET_LAW,
    PHASOR,
    TIME_DEPENDENCE,
)


CONTRACT_VERSION = 2


PropagationComponent: TypeAlias = Literal[
    "los", "reflection", "transmission", "diffraction"
]
PropagationResponse: TypeAlias = Literal[
    "scalar_transport", "complex3_transport", "polarimetric_transport"
]
PropagationTopologyMode: TypeAlias = Literal["discover"]
PropagationAdMode: TypeAlias = Literal["none", "jvp", "vjp"]

COMPONENTS: frozenset[str] = frozenset(get_args(PropagationComponent))
RESPONSES: frozenset[str] = frozenset(get_args(PropagationResponse))
TOPOLOGY_MODES: frozenset[str] = frozenset(get_args(PropagationTopologyMode))
AD_MODES: frozenset[str] = frozenset(get_args(PropagationAdMode))

MAX_DEPTH = 5

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
    """Compact discrete rows in stable pair-major order."""

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

    def __post_init__(self) -> None:
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
    """Endpoint-projected complex scalar transfer at the reference frequency."""

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
    """World-Cartesian complex electric field and propagation direction."""

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
    factors; those are not part of a linear polarization-basis map.
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
    fixed_topology_row_validity_components=frozenset({"reflection"}),
    polarimetric_frozen_ad_inputs=(
        "tx_power",
        "mu_r",
        "source_basis",
        "sink_basis",
        "endpoint_polarizations",
    ),
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


@dataclass(frozen=True, slots=True)
class FixedTopologyRequest:
    """A reevaluation request against an already-discovered topology.

    ``topology`` is a raw :class:`PropagationTopology` for the zero-interaction
    LoS route, or a :class:`PreparedFixedTopology` for any route that carries
    interactions. Structural validity is enforced here; scene-dependent
    capability checks belong to
    :func:`witwin.channel.propagation.consumer.reevaluate`.
    """

    sources: EndpointBatch
    sinks: EndpointBatch
    reference_frequency_hz: float | torch.Tensor
    topology: PropagationTopology | PreparedFixedTopology
    response: PropagationResponse
    ad_mode: PropagationAdMode

    def __post_init__(self) -> None:
        _require_endpoints(self.sources, self.sinks)
        if not isinstance(
            self.topology, PropagationTopology | PreparedFixedTopology
        ):
            raise TypeError(
                "topology must be a PropagationTopology or a "
                "PreparedFixedTopology"
            )
        response = _require_vocabulary("response", self.response, RESPONSES)
        _require_vocabulary("ad_mode", self.ad_mode, AD_MODES)
        if response not in _CAPABILITIES.fixed_topology_responses:
            raise NotImplementedError(
                f"fixed-topology reevaluation does not support {response!r}; "
                f"supported responses are "
                f"{sorted(_CAPABILITIES.fixed_topology_responses)}"
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
    line-of-sight row is replayed as pure free-space transport and is never
    re-tested for visibility, so a sink that moves behind a wall still reports
    ``True`` and a full-strength field while fresh discovery would drop the
    row. A row also keeps its ORIGINAL ``primitive_sequence`` label when the
    stationary point slides onto a coplanar twin triangle; the numbers are
    exact, the discrete label is stale. Both are recorded in ADR-037.
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
    "capabilities",
    "prepare_fixed_topology",
]
