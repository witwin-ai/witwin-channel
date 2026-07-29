# Copyright Xingyu Chen.
# Stable solver-neutral propagation consumer facade.

"""Stable solver-neutral propagation consumer facade."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
import math
import struct
from typing import TYPE_CHECKING, Literal, Protocol, TypeAlias, get_args

import torch

from witwin.channel.constants import (
    NARROWBAND_FREQUENCY_OFFSET_ERROR_LAW,
    NARROWBAND_FREQUENCY_OFFSET_LAW,
    PHASOR,
    TIME_DEPENDENCE,
)
from witwin.channel.runtime import (
    _ad_first_order_only,
    _ad_geometry_live,
    _ad_native_tangent_or_none,
    _ad_native_tensor,
    disable_functorch,
    require_tensor,
    required_symbol as _required_native_op,
)

if TYPE_CHECKING:
    from witwin.channel.kernels.topology import ExactPairMetadata
    from witwin.channel.propagation.enumerated import (
        EnumeratedEndpointTensors,
    )
    from witwin.channel.propagation.rows import EvaluatedPaths
    from witwin.channel.scene.compiler import CompiledScene
    from witwin.channel.scene.endpoints import SolverScene


# --- Wideband frequency offsets (wideband evaluation) ---------------------------------
#
# The wideband frequency-offset surface of the consumer contract (wideband evaluation).
#
# A fixed-topology request may declare a grid of propagation-frequency offsets and
# receive the same frozen rows evaluated at each absolute frequency. Everything
# that grid needs before it reaches a scene lives here: the float32 launch
# resolution that bounds how fine it may be, its structural validation, and the
# paired-presence law its payload obeys.
#
# It leads the module for one reason worth naming: nothing here depends on any
# contract type, so the dependency runs one way and the vocabulary section below
# stays the single place a reader looks up a field.


# The launch grid the native field bridges actually resolve. Every bridge takes
# a double ``frequency_hz`` and ``static_cast<float>``s it at the launch, so two
# absolute frequencies inside one float32 ULP are the SAME launch and return
# bit-identical coefficients. Published as a law plus a function rather than as
# a constant, because the resolution is a function of the reference frequency
# (8192 Hz at 77 GHz, 64 Hz at 1 GHz).
NATIVE_FREQUENCY_RESOLUTION_LAW = (
    "resolution_hz = ulp_float32(reference_frequency_hz)"
)


def native_frequency_resolution_hz(reference_frequency_hz: float) -> float:
    """Smallest absolute frequency step the native launch grid resolves.

 The value is one float32 unit in the last place at
 ``reference_frequency_hz``. A caller computes the same number the
 wideband refusal uses instead of rederiving it, which is the propagation consumer
 that a declared limit is discoverable rather than learned from a rejection.
 """

    value = abs(float(reference_frequency_hz))
    if not math.isfinite(value) or value == 0.0:
        raise ValueError(
            "reference_frequency_hz must be finite and non-zero to have a "
            "native frequency resolution"
        )
    # Round to the float32 the launch actually receives before reading the
    # binade, so a value that rounds up across a power of two reports the
    # resolution of the launch rather than of the request.
    launched = struct.unpack("<f", struct.pack("<f", value))[0]
    _, exponent = math.frexp(launched)
    # float32 carries a 24-bit significand, so one ULP in that binade is
    # 2**(exponent - 24).
    return math.ldexp(1.0, exponent - 24)


def require_frequency_offsets(value: object) -> tuple[float, ...] | None:
    """Structural validation of a wideband offset grid, before any native work.

 The grid is a HOST DECLARATION in the same class as ``slot_count``: it
 names which absolute frequencies the same frozen rows are evaluated at. It
 is deliberately not a tensor and deliberately not differentiable.
 """

    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        raise TypeError(
            "frequency_offsets_hz must be a tuple of host floats, not a "
            "torch.Tensor: the offset grid is a host declaration of which "
            "absolute frequencies to evaluate, not a differentiable input. A "
            "tangent with respect to one grid point is identical to the "
            "reference_frequency_hz tangent evaluated at that point, so seed "
            "reference_frequency_hz instead"
        )
    if not isinstance(value, tuple):
        raise TypeError("frequency_offsets_hz must be a tuple of floats or None")
    if not value:
        raise ValueError(
            "frequency_offsets_hz must be a non-empty tuple; pass None for a "
            "single-frequency request"
        )
    offsets = []
    for entry in value:
        if isinstance(entry, bool) or not isinstance(entry, (int, float)):
            raise TypeError("frequency_offsets_hz entries must be floats")
        offset = float(entry)
        if not math.isfinite(offset):
            raise ValueError(
                f"frequency_offsets_hz entries must be finite, got {entry!r}"
            )
        offsets.append(offset)
    if len(set(offsets)) != len(offsets):
        raise ValueError(
            "frequency_offsets_hz must not repeat an offset; duplicate entries "
            "produce bit-identical columns and hide a caller bug"
        )
    return tuple(offsets)


def require_wideband_payload(
    name: str,
    payload: object,
    offsets: object,
    reference: torch.Tensor,
) -> None:
    """Enforce the wideband evaluation paired-presence and shape law on one transport.

 The payload and the grid it was evaluated on are both present or both
 absent. An unpaired payload is a column set nobody can label, and an
 unpaired grid is a promise nobody kept; either one is a contract error
 rather than something a reader should have to guess about.

 ``reference`` is the single-frequency tensor the payload is the band of, so
 its shape defines the payload's: the frequency axis is inserted after the
 row axis and every trailing axis is preserved. Taking the shape from the
 reference rather than restating it keeps one description of what a column
 is.
 """

    if payload is None and offsets is None:
        return
    if payload is None or offsets is None:
        raise ValueError(
            f"{name} and frequency_offsets_hz are paired: publish both or neither"
        )
    if not isinstance(offsets, tuple) or not offsets:
        raise TypeError("frequency_offsets_hz must be a non-empty tuple of floats")
    if not isinstance(payload, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if payload.dtype != reference.dtype:
        raise TypeError(f"{name} must use {reference.dtype}, got {payload.dtype}")
    expected = (int(reference.shape[0]), len(offsets), *reference.shape[1:])
    if tuple(payload.shape) != expected:
        raise ValueError(
            f"{name} must have shape {expected}, got {tuple(payload.shape)}"
        )
    if payload.device != reference.device:
        raise ValueError(f"{name} must be on {reference.device}, got {payload.device}")


# The layout a wideband payload adds on top of the row and slot layouts, which
# it does not redefine. Frequency is orthogonal to both: the same rows are
# evaluated at F frequencies, so the axis is appended and nothing is tiled,
# re-paired, or re-segmented.
WIDEBAND_OFFSET_LAYOUT = (
    "frequency_minor:"
    "payload[row, j] = response(row) at"
    " reference_frequency_hz+frequency_offsets_hz[j];"
    "row axis and pair segmentation unchanged;"
    "row_valid stays [K] and broadcasts over j;"
    "geometry published once from the reference evaluation;"
    "slot composition gives [slot_count*frozen_row_count, F]"
)

# Why an offset grid can be refused as unresolvable. Channel publishes the
# resolution and the resulting phase bound; it does not evaluate the bound,
# because that needs max(delay_s), which is a device reduction plus a host read
# the compact output budget does not have. The caller owns that check.
WIDEBAND_FREQUENCY_QUANTIZATION_LAW = (
    "launch_grid=float32;"
    " resolution_hz=ulp_float32(reference_frequency_hz);"
    " abs_phase_error_rad <= pi*resolution_hz*delay_s"
)


# --- AD admission policy ---------------------------------------------------------------
#
# Host preflight validates first-order AD support before any native launch or
# result allocation. The declarations here build the published capability
# record and sit beside the refusals that enforce it.


# Native component identifiers a frozen topology may carry into reevaluation.
# The names are the contract vocabulary; the integers are the discovery
# owner's encoding, which the consumer reads but does not define.
_FIXED_TOPOLOGY_COMPONENT_IDS: tuple[tuple[str, int], ...] = (
    ("los", 0),
    ("reflection", 1),
)

# Publish the material tensors each component reads so structural zero derivatives are explicit.
_COMPONENT_MATERIAL_LEAVES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("diffraction", ("eps_r", "sigma_e", "thickness_m", "gain")),
    ("los", ()),
    ("reflection", ("eps_r", "sigma_e", "thickness_m", "gain")),
    ("transmission", ("layer_eps_r", "layer_sigma_e", "layer_thickness_m")),
)

# Which published geometry tensors carry a derivative, per route (first-order differentiation).
# Discovery re-solves the topology, so its interaction table and arrival
# direction are declared non-differentiable outputs rather than silently
# detached ones; the supported differentiable geometry route is
# prepare_fixed_topology + reevaluate.
_DIFFERENTIABLE_GEOMETRY_OUTPUTS: tuple[tuple[str, frozenset[str]], ...] = (
    ("discovery", frozenset({"path_length_m", "delay_s"})),
    (
        "fixed_topology",
        frozenset(
            {
                "path_length_m",
                "delay_s",
                "interaction_positions_m",
                "field_direction",
            }
        ),
    ),
)

# Inputs every response refuses before any native work, in every AD mode but
# "none". The native field companions reject them by contract, so a request
# that carries one fails at the boundary instead of inside backward.
_PRIMAL_ONLY_AD_INPUTS: tuple[str, ...] = (
    "materials.layer_mu_r",
    "materials.mu_r",
    "sinks.polarization_basis",
    "sinks.polarizations",
    "sources.polarization_basis",
    "sources.polarizations",
    "sources.powers_w",
)

_MATERIAL_AD_LEAVES = (
    "eps_r",
    "sigma_e",
    "thickness_m",
    "gain",
    "layer_eps_r",
    "layer_sigma_e",
    "layer_thickness_m",
)


def has_forward_tangent(value: torch.Tensor) -> bool:
    return torch.autograd.forward_ad.unpack_dual(value).tangent is not None


def carries_ad(value: torch.Tensor | None) -> bool:
    return value is not None and (
        value.requires_grad or has_forward_tangent(value)
    )


def _primal_only_values(
    compiled: CompiledScene, request: object
) -> dict[str, torch.Tensor | None]:
    """The tensors named by ``capabilities.primal_only_ad_inputs``.

 Two of them live on the compiled scene rather than on the request: the
 relative permeabilities reach the Fresnel companions as constants and are
 rejected there, so they belong in the same pre-compute refusal as the
 request-side constants.
 """

    materials = compiled.materials
    return {
        "sources.powers_w": request.sources.powers_w,
        "sources.polarizations": request.sources.polarizations,
        "sinks.polarizations": request.sinks.polarizations,
        "sources.polarization_basis": request.sources.polarization_basis,
        "sinks.polarization_basis": request.sinks.polarization_basis,
        "materials.mu_r": materials.mu_r,
        "materials.layer_mu_r": materials.layer_mu_r,
    }


def require_primal_only_ad_inputs(
    compiled: CompiledScene, request: object
) -> None:
    """Reject before native work any AD input that the field companions declare primal-only."""

    if request.ad_mode == "none":
        return
    values = _primal_only_values(compiled, request)
    for name in capabilities().primal_only_ad_inputs:
        if carries_ad(values[name]):
            raise NotImplementedError(
                f"{name} is primal-only; the native field companion that "
                "consumes it does not differentiate it. "
                "capabilities().primal_only_ad_inputs names every such input"
            )


def _ad_leaf_tensors(
    compiled: CompiledScene, request: object
) -> tuple[tuple[str, torch.Tensor], ...]:
    """Every tensor a caller can seed on this call, named for a refusal.

 Host attribute reads only: no device work, no allocation, and no
 synchronization, so this is free to run on the pre-flight of every call.
 """

    materials = compiled.materials
    candidates: list[tuple[str, object]] = [
        ("sources.positions_m", request.sources.positions_m),
        ("sinks.positions_m", request.sinks.positions_m),
        ("reference_frequency_hz", request.reference_frequency_hz),
    ]
    candidates.extend(_primal_only_values(compiled, request).items())
    candidates.extend(
        (f"materials.{name}", getattr(materials, name, None))
        for name in _MATERIAL_AD_LEAVES
    )
    candidates.extend(
        (f"structures[{index}].vertices", getattr(structure, "vertices", None))
        for index, structure in enumerate(compiled.structures)
    )
    return tuple(
        (name, value)
        for name, value in candidates
        if isinstance(value, torch.Tensor)
    )


def require_first_order_request(
    compiled: CompiledScene, request: object
) -> None:
    """Refuse a forward-over-reverse composition before any numerical work.

 A reverse pass cannot carry a forward tangent through the native
 companions: the gradient comes back with the correct first-order value and
 ``unpack_dual(grad).tangent is None``, so a mixed second derivative reads as
 an exact zero with no error anywhere. That is the worst shape a silent cell
 can take, and it is refused here rather than answered wrongly.

 The symmetric rule ("jvp with a requires_grad input") is deliberately NOT
 enforced: forward-mode liveness's declared convention explicitly supports a dual built on
 a ``requires_grad`` primal, and the field facades run the same Function for
 both modes, so such a request is a legitimate first-order one.
 Reverse-over-reverse is caught instead where it becomes wrong, by
 ``_ad_first_order_only`` inside every backward.
 """

    if request.ad_mode != "vjp":
        return
    for name, value in _ad_leaf_tensors(compiled, request):
        if has_forward_tangent(value):
            raise NotImplementedError(
                f"ad_mode='vjp' with a forward dual on {name} is a "
                "second-order request; Channel is first-order only and a "
                "reverse gradient carries no tangent. "
                "capabilities().supports_higher_order_ad is False"
            )


def ad_ledger(ad_mode: str) -> object | None:
    """One AD ledger per reevaluation, or ``None`` for a primal call.

 The discovery route already builds one inside its field loop and hands it
 up through the execution sidecars. The fixed-topology route built none, so
 the inner loop a per-frame consumer runs reported no AD accounting at all;
 this is the same counter, constructed at the one place that owns the whole
 call. A primal call constructs nothing and pays nothing.
 """

    if ad_mode == "none":
        return None
    from witwin.channel.runtime import AdLaunchLedger

    return AdLaunchLedger()


def tape_bytes(ledger_bytes: int, ad_mode: str) -> int:
    """Reproduce the solver-metadata tape gate rather than the raw counter.

 ``AdLaunchLedger`` sums what every registered companion saved, and forward
 mode retains none of it past the solve. The solver metadata layer applies
 exactly this gate (``deterministic/pipeline.py``), so forwarding the raw
 sidecar number here would report retained tape for a jvp call and
 contradict the ledger's own contract.
 """

    return int(ledger_bytes) if ad_mode == "vjp" else 0


# --- The published consumer vocabulary ------------------------------------
#
# Stable solver-neutral propagation consumer contracts.
#
# This section is the single source of truth for the consumer vocabulary. The
# accepted component, response, topology, and AD-mode values are declared here as
# ``Literal`` aliases with matching frozen sets, and:func:`capabilities` returns
# the frozen capability record. A consumer can therefore discover what the
# contract supports before building a request instead of learning it from a
# rejected call.


CONTRACT_VERSION = 6


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


# The consumer boundary reports a wrong dtype as a TypeError; the internal row
# and capacity contracts report it as a ValueError. That is the only difference
# there ever was between the three copies of this check, so it is the only thing
# bound here and the check itself has one owner.
_require_tensor = partial(require_tensor, dtype_error=TypeError)


class _WorldVersionSource(Protocol):
    """What a freshness check reads. Structural, so it never imports a scene."""

    topology_version: int
    geometry_version: int
    material_version: int
    assignment_version: int


@dataclass(frozen=True, slots=True, eq=False)
class WorldProvenance:
    """Which world a discovered topology belongs to (world-version validation).

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
 on the compiled scene - reference-frequency match, response/component and
 response/AD combinations, polarimetric basis requirements - is enforced by:func:`witwin.channel.propagation.consumer.evaluate` before any native work.
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

 ``provenance`` records which world these rows were discovered against.:func:`witwin.channel.propagation.consumer.evaluate` stamps it:func:`prepare_fixed_topology` forwards it verbatim, and:func:`witwin.channel.propagation.consumer.reevaluate` refuses a frozen
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
 (source excitation). Power/gain values are its squared magnitude.

 ``coefficient_offsets`` is the optional wideband payload (wideband evaluation):
 ``[K, F]`` complex64, where column ``j`` is the SAME row evaluated at
 ``reference_frequency_hz + frequency_offsets_hz[j]``. It is present exactly
 when the request declared ``frequency_offsets_hz``, and the grid it was
 evaluated on is echoed here so a column can never be read against the wrong
 frequency. A ``0.0`` entry produces a column bit-identical to
 ``coefficient``.
 """

    coefficient: torch.Tensor
    coefficient_offsets: torch.Tensor | None = None
    frequency_offsets_hz: tuple[float, ...] | None = None

    def __post_init__(self) -> None:
        _require_tensor("coefficient", self.coefficient, dtype=torch.complex64, ndim=1)
        require_wideband_payload(
            "coefficient_offsets",
            self.coefficient_offsets,
            self.frequency_offsets_hz,
            self.coefficient,
        )

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
 onto the receive polarization reproduces:class:`ScalarTransport.coefficient` (source excitation).

 ``field_offsets`` is the optional wideband payload (wideband evaluation): ``[K, F, 3]``
 complex64 on the same grid law as:attr:`ScalarTransport.coefficient_offsets`. ``direction`` stays ``[K, 3]``
 because it is geometry, and geometry does not depend on frequency.
 """

    field: torch.Tensor
    direction: torch.Tensor
    field_offsets: torch.Tensor | None = None
    frequency_offsets_hz: tuple[float, ...] | None = None

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
        require_wideband_payload(
            "field_offsets", self.field_offsets, self.frequency_offsets_hz, field
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
 deliberately unlike:class:`ScalarTransport` and:class:`Complex3Transport`,
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
    # What that law costs, so "narrowband" is a number rather than an adjective.
    # The wideband route (FixedTopologyRequest.frequency_offsets_hz) removes the
    # spreading and material terms exactly and refuses the dispersion term.
    narrowband_frequency_offset_error_law: str = (
        NARROWBAND_FREQUENCY_OFFSET_ERROR_LAW
    )
    # The layout a wideband payload adds on top of the row layouts above, and
    # the float32 launch grid that bounds how fine an offset grid may be. Both
    # are owned by the wideband module beside this one.
    wideband_offset_layout: str = WIDEBAND_OFFSET_LAYOUT
    wideband_frequency_quantization_law: str = WIDEBAND_FREQUENCY_QUANTIZATION_LAW
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
    # instead of failing the whole batch (fixed-topology replay).
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
    # Whether a FixedTopologyRequest may declare frequency_offsets_hz and
    # receive the same frozen rows evaluated at F absolute frequencies
    # (wideband evaluation). The payload layout is
    # PropagationConvention.wideband_offset_layout.
    supports_wideband_offsets: bool
    # Responses that carry a wideband payload. polarimetric_transport does not:
    # it is line-of-sight only and power-free, and widening it is a separate
    # decision rather than a side effect of this one.
    wideband_responses: frozenset[str]
    # Components a wideband request may replay. This is the fixed-topology set,
    # because the grid is declared on the fixed-topology route only:
    # transmission and diffraction are not freezable components.
    wideband_components: frozenset[str]
    # A compiled dispersive material record is frozen at the primal compile
    # frequency, so evaluating it at an offset would publish the
    # reference-frequency material law under a different frequency label. The
    # request is refused instead, at every AD mode including "none".
    wideband_dispersive_materials: bool
    # A Kirchhoff roughness table and a phase screen are resident resources
    # keyed on a material cache token that hashes the compile frequency, so
    # reusing one at an offset is a frozen approximation of the same class.
    # Refused for the same reason.
    wideband_rough_materials: bool
    # A contract bound on len(frequency_offsets_hz), or None when the only
    # bound is device memory and launch time. Each column is one more launch
    # per bucket; no column adds a host observation or a synchronization.
    max_frequency_offset_count: int | None
    # How the smallest resolvable offset is computed. Matches
    # native_frequency_resolution_hz, which a caller calls to get the number.
    native_frequency_resolution_law: str
    # Which compiled-material tensors each component reads. A leaf outside its
    # component's tuple has an exactly zero derivative, which is the true and
    # complete answer rather than a missing one.
    component_material_leaves: tuple[tuple[str, tuple[str, ...]], ...]
    # Which published PropagationGeometry tensors carry a derivative, per route
    # ("discovery" and "fixed_topology"). A tensor absent from a route's set is
    # a declared non-differentiable output on that route.
    differentiable_geometry_outputs: tuple[tuple[str, frozenset[str]], ...]
    # The components for which PropagationGeometry.field_direction is live on
    # the fixed-topology route. A request that mixes in any other component
    # publishes a fully detached field_direction for the whole result: liveness
    # is one decision per result, never a per-row one.
    direction_differentiable_components: frozenset[str]
    # Inputs refused before any native work on EVERY response, not only the
    # polarimetric one. polarimetric_frozen_ad_inputs stays as the composed
    # operator's own vocabulary and names the same physics.
    primal_only_ad_inputs: tuple[str, ...]
    # Second-order AD is refused everywhere, loudly, before any partial
    # second-order result. There is no route that supports it.
    supports_higher_order_ad: bool
    # Whether PropagationDiagnostics carries ad_companion_launches and
    # ad_tape_bytes.
    ad_accounting: bool

    def components_for(self, response: str) -> frozenset[str]:
        return dict(self.response_components)[response]

    def ad_modes_for(self, response: str) -> frozenset[str]:
        return dict(self.response_ad_modes)[response]

    def ad_modes_for_component(self, component: str) -> frozenset[str]:
        return dict(self.component_ad_modes)[component]

    def material_leaves_for(self, component: str) -> tuple[str, ...]:
        """Which compiled-material tensors ``component`` reads (first-order differentiation)."""

        return dict(self.component_material_leaves)[component]

    def differentiable_geometry_for(self, route: str) -> frozenset[str]:
        """Which published geometry tensors carry a derivative on ``route``."""

        return dict(self.differentiable_geometry_outputs)[route]


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
    # diffraction advertises only the primal: the consumer cannot produce a
    # diffraction row at all (see docs/dev/propagation-ad-capability-matrix.md),
    # so an AD column for it would be fictional. The existing unsupported-AD
    # branch of _preflight_evaluate turns this into a pre-compute refusal.
    component_ad_modes=tuple(
        (component, frozenset({"none"}) if component == "diffraction" else AD_MODES)
        for component in sorted(COMPONENTS)
    ),
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
    supports_wideband_offsets=True,
    wideband_responses=frozenset({"scalar_transport", "complex3_transport"}),
    wideband_components=frozenset(
        name for name, _ in _FIXED_TOPOLOGY_COMPONENT_IDS
    ),
    wideband_dispersive_materials=False,
    wideband_rough_materials=False,
    max_frequency_offset_count=None,
    native_frequency_resolution_law=NATIVE_FREQUENCY_RESOLUTION_LAW,
    component_material_leaves=_COMPONENT_MATERIAL_LEAVES,
    differentiable_geometry_outputs=_DIFFERENTIABLE_GEOMETRY_OUTPUTS,
    direction_differentiable_components=frozenset({"los", "reflection"}),
    primal_only_ad_inputs=_PRIMAL_ONLY_AD_INPUTS,
    supports_higher_order_ad=False,
    ad_accounting=True,
)


def capabilities() -> PropagationCapabilities:
    """Return what this consumer contract version supports.

 Call this before building a:class:`PropagationRequest` to check that a
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
    # Published response columns: len(frequency_offsets_hz) with a wideband
    # grid, and 1 without one, where the single column is the reference
    # evaluation itself. It makes the honest launch law auditable: a wideband
    # call costs (1 + F) * buckets * launches_per_bucket, the leading 1 being
    # the reference column that also produces the shared geometry and
    # row_valid, while the copy and synchronization counts above stay at one
    # whatever F is.
    frequency_column_count: int = 1
    # first-order differentiation AD accounting. One entry per registered differentiable native
    # companion this call launched, and the bytes those companions retained for
    # backward. Forward mode retains nothing past the solve, so a jvp call
    # reports zero tape however many companions it launched.
    ad_companion_launches: int = 0
    ad_tape_bytes: int = 0


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
 later:func:`witwin.channel.propagation.consumer.reevaluate` call. They are
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




@dataclass(frozen=True, slots=True)
class FixedTopologyRequest:
    """A reevaluation request against an already-discovered topology.

 ``topology`` is a raw:class:`PropagationTopology` for the zero-interaction
 LoS route, or a:class:`PreparedFixedTopology` for any route that carries
 interactions. Structural validity is enforced here; scene-dependent
 capability checks belong to:func:`witwin.channel.propagation.consumer.reevaluate`.

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
 law is:attr:`PropagationConvention.slot_pair_layout`; build the topology
 with:func:`replicate_over_slots`. It requires a:class:`PreparedFixedTopology`: the raw zero-interaction route builds its
 pair segmentation inside the native gather over the full source/sink outer
 product and therefore cannot express a block-diagonal layout.

 ``frequency_offsets_hz`` declares that the same frozen rows are wanted at
 the ``F`` absolute frequencies ``reference_frequency_hz + df_j``, in the
 declared order (wideband evaluation). ``None``, the default, is exactly the
 single-frequency behaviour, bit for bit. The grid is a host tuple rather
 than a tensor because it is a declaration in the same class as
 ``slot_count``: it names which frequencies to evaluate, it is structurally
 non-differentiable, and it is float64-exact. It is a PROPAGATION frequency
 grid and nothing else - it never names a subcarrier count, an FFT size, or
 a bandwidth. Scene-dependent limits (dispersive materials, rough materials
 and phase screens, and the native float32 launch resolution) are enforced
 by:func:`witwin.channel.propagation.consumer.reevaluate` before any native
 work; see:func:`native_frequency_resolution_hz`.
 """

    sources: EndpointBatch
    sinks: EndpointBatch
    reference_frequency_hz: float | torch.Tensor
    topology: PropagationTopology | PreparedFixedTopology
    response: PropagationResponse
    ad_mode: PropagationAdMode
    world_motion: PropagationWorldMotion = "frozen_world"
    slot_count: int = 1
    frequency_offsets_hz: tuple[float, ...] | None = None

    def __post_init__(self) -> None:
        _require_endpoints(self.sources, self.sinks)
        self._require_slots()
        object.__setattr__(
            self,
            "frequency_offsets_hz",
            require_frequency_offsets(self.frequency_offsets_hz),
        )
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
        if (
            self.frequency_offsets_hz is not None
            and response not in _CAPABILITIES.wideband_responses
        ):
            raise NotImplementedError(
                f"frequency_offsets_hz is not supported for {response!r}; "
                f"capabilities().wideband_responses is "
                f"{sorted(_CAPABILITIES.wideband_responses)}"
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
 fixed-topology replay.

 **Replay is subtractive (world-version validation).** A frozen row can stop existing and is
 published as ``row_valid=False``; a path that comes into existence at the
 new endpoint or world state is NOT discovered here and is silently absent
 from the batch. The rows that are published are exactly correct, and the
 batch under-reports. There is no birth signal, by design: every candidate
 detector costs either a full discovery or a device reduction plus a host
 read the compact output budget does not have. A caller whose scene can gain paths
 owns the rediscovery cadence; poll:func:`witwin.channel.propagation.consumer.rediscovery_required` for a
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


# --- Preparing a discovered topology for repeated replay ------------------
#
# Preparing a discovered topology for repeated fixed-topology replay.
#
# ``prepare_fixed_topology`` is the one place the consumer looks at frozen rows on
# the host: it validates their depth/interaction padding and partitions them into
# the ascending ``(component, depth)`` buckets a replay launches one at a time.
# ``replicate_over_slots`` then tiles that partition over slot batching block-diagonal
# slots by index arithmetic alone.
#
# Both are vocabulary-level boundary work with no physics in them, and both sit
# beside the vocabulary section rather than inside it, which stays the place a
# reader looks up a TYPE.


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
 component outside ``capabilities.fixed_topology_components``, and returns
 the ascending ``(component, depth)`` buckets that:func:`witwin.channel.propagation.consumer.reevaluate` replays.

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
 for staleness exactly like the topology it came from (world-version validation).
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


# --- Replay: native facades, row gathers, and bucket replay ---------------
#
# Everything the consumer needs to evaluate rows it already owns.
#
# This section is the single owner of the consumer's replay machinery: the native
# ABI facades it dispatches, the structural row gathers that bind frozen rows to
# the current endpoint batches, the source-amplitude and Jones composition
# helpers, and the prepared-topology bucket replay itself. They were six modules
# that only ever called each other, so a reader following one replay had to walk
# six files to see one code path.
#
# Nothing here is a contract: the vocabulary section above stays the single place
# a reader looks up a published type. Nothing here decides admission either; that
# is the AD admission policy section. This is dispatch, structural selection, and
# packing over native owners.
#
# Four things drive the shape of the code.
#
# **The compact finalizer.** Exact valid rows and their pair segmentation come
# from one native owner, through one autograd Function with registered
# forward/backward/JVP companions.
#
# **The frozen line-of-sight gather.** A zero-interaction frozen topology is bound
# to its endpoint batches by a fused native gather with its own AD family. Its
# contract is line-of-sight only, so a frozen topology that carries interactions
# uses the structural gather beside it instead: integer contract validation
# reduced to one device bitmask, ``index_select`` row selection of caller-owned
# endpoint tensors, and the CSR pair segmentation built from integer row
# identity. No geometry, no field, and no material value is computed, transformed,
# or re-derived; every physical quantity is produced later by a native kernel that
# owns it. The validation budget matches the native LoS gather exactly: one
# four-byte device-to-host copy and one synchronization for the whole batch,
# before any native work runs.
#
# **The composed Jones operator.** The native field transport is linear in the
# transmit polarization and linear in the receive polarization:
#
# * ``project_to_wedge_plane(v, e) = v - e*(v.e)`` is linear in ``v``;
# * a Fresnel bounce scales the s and p components by coefficients that depend on
# the incidence frame and the material, never on the field itself;
# * the trailing free-space factor is a complex scalar;
# * ``project_receiver(E, d, p) = E . project_to_wedge_plane(p, d)``.
#
# So the map from a source transverse component to a sink transverse component is
# bilinear, and the four entries of the operator are recovered exactly by
# exciting the SAME native transport twice, once per source basis vector, and
# projecting each response onto both sink basis vectors. Nothing here computes
# physics: the composition chooses excitations, dispatches the native owners, and
# stacks their published results. Both transverse bases are produced by the native
# ``consumer_los_jones`` endpoint-basis owner rather than by a Torch normalize or
# cross product. A reflection row has two different directions - the launch
# direction toward its first interaction and the arrival direction from its last
# interaction - and the basis for each is obtained by handing that leg's two
# endpoints to the native owner, which recomputes the direction with the same
# ``safe_normalize`` the field kernel uses. The bases are structurally
# primal-only: the composition feeds them to the native companions as
# ``tx_polarization`` and ``rx_polarization``, both of which reject gradients by
# contract.
#
# **The prepared-topology replay.** A frozen reflection row is a face sequence,
# not a fixed point in space. At new endpoint positions its stationary point has
# to be resolved again, because the specular point moves and can leave its facet
# or become occluded. The replay runs exactly the owners the discovery path used -
# the RayD fixed-winner EPC re-solve for the geometry, and the native reflection
# field transport for the field - so a reevaluated row is the value discovery
# would have produced at those endpoints.
#
# The native reflection transport takes ONE uniform interaction depth per launch,
# so a mixed-depth frozen batch is replayed one ``(component, depth)`` bucket at a
# time. Those buckets come from ``prepare_fixed_topology`` and are host-known, so
# the per-call path never observes a device count to decide how to launch.
#
# A frozen path can legitimately stop existing. Failing the whole batch would
# force a caller back to full discovery the first time one path dies, which
# defeats the capability, so validity is published per row. An invalid row is NOT
# a failure: it is the correct, complete answer that this frozen path does not
# exist at these endpoints. Capacity, ABI, contract, and device failures remain
# all-or-nothing and still raise before a result exists.
#
# An invalid row is made inert at the input, not patched at the output: its
# transmit polarization is replaced by the zero vector, and the native transport
# carries that exactly through projection, every Fresnel bounce, and the trailing
# free-space scalar, so all four field outputs come out as exact zeros from the
# kernel that owns them. Only the scalar path geometry, which has no such inert
# excitation, is selected against the mask afterwards.


# --- Native compact finalization ------------------------------------------

_TOPOLOGY_FIELDS = (
    "valid",
    "tx_id",
    "rx_id",
    "depth",
    "component_id",
    "primitive_id",
    "edge_id",
    "material_id",
    "primitive_sequence",
    "material_sequence",
    "interaction_type",
)
_CONTINUOUS_FIELDS = (
    "path_length_m",
    "delay_s",
    "field_direction",
    "interaction_position",
    "interaction_normal",
    "interaction_positions",
    "interaction_normals",
    "path_gain",
    "path_field",
    "field_xyz",
    "coefficient",
)
_STRUCTURAL_FIELDS = (
    "selected_row_index",
    "pair_index",
    "pair_offsets",
    "source_id",
    "sink_id",
)
_COMPACT_OUTPUT_FIELDS = (
    *_STRUCTURAL_FIELDS,
    *_TOPOLOGY_FIELDS,
    *_CONTINUOUS_FIELDS,
)
_DISCRETE_OUTPUT_COUNT = len(_STRUCTURAL_FIELDS) + len(_TOPOLOGY_FIELDS)
COMPACT_COUNT_D2H_COPIES = 1
COMPACT_COUNT_D2H_BYTES = 8
COMPACT_COUNT_SYNCHRONIZATIONS = 1


@dataclass(frozen=True, slots=True)
class CompactEvaluatedPaths:
    path_count: int
    pair_index: torch.Tensor
    pair_offsets: torch.Tensor
    source_id: torch.Tensor
    sink_id: torch.Tensor
    evaluated: EvaluatedPaths
    count_d2h_copies: int
    count_d2h_bytes: int
    count_synchronizations: int
    native_launch_count: int


@dataclass(frozen=True, slots=True)
class LoSJonesRows:
    matrix: torch.Tensor
    source_basis: torch.Tensor
    sink_basis: torch.Tensor
    native_launch_count: int


def _candidate_tensors(paths: EvaluatedPaths) -> tuple[torch.Tensor, ...]:
    return (
        *(getattr(paths.topology, name) for name in _TOPOLOGY_FIELDS),
        *(getattr(paths.geometry, name) for name in _CONTINUOUS_FIELDS[:7]),
        *(getattr(paths.fields, name) for name in _CONTINUOUS_FIELDS[7:]),
    )


def _validate_inputs(
    paths: EvaluatedPaths,
    source_stable_ids: torch.Tensor,
    sink_stable_ids: torch.Tensor,
) -> tuple[torch.Tensor, ...]:
    from witwin.channel.propagation.rows import EvaluatedPaths

    if not isinstance(paths, EvaluatedPaths):
        raise TypeError("paths must be EvaluatedPaths")
    tensors = _candidate_tensors(paths)
    device = tensors[0].device
    if device.type != "cuda":
        raise ValueError("compact evaluated-path finalization requires CUDA")
    for name, tensor in zip(
        (*_TOPOLOGY_FIELDS, *_CONTINUOUS_FIELDS), tensors, strict=True
    ):
        if tensor.device != device:
            raise ValueError(f"{name} must share evaluated path device")
        if not tensor.is_contiguous():
            raise ValueError(f"{name} must be contiguous")
    for name, lookup in (
        ("source_stable_ids", source_stable_ids),
        ("sink_stable_ids", sink_stable_ids),
    ):
        if not isinstance(lookup, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")
        if (
            lookup.device != device
            or lookup.dtype != torch.int64
            or lookup.ndim != 1
            or not lookup.is_contiguous()
        ):
            raise ValueError(
                f"{name} must be a contiguous CUDA int64 vector on {device}"
            )
    return tensors


def evaluated_paths_compact_finalize(
    *inputs: torch.Tensor, rows_are_compact: bool
) -> dict[str, object]:
    return _required_native_op("evaluated_paths_compact_finalize")(
        *inputs, rows_are_compact
    )


class _CompactEvaluatedPathsFunction(torch.autograd.Function):
    @staticmethod
    def forward(*inputs):
        raw = evaluated_paths_compact_finalize(
            *inputs, rows_are_compact=False
        )
        expected = {"path_count", *_COMPACT_OUTPUT_FIELDS}
        if not isinstance(raw, dict) or set(raw) != expected:
            raise TypeError("native compact finalizer returned bad fields")
        if type(raw["path_count"]) is not int:
            raise TypeError("native compact finalizer returned a non-int path_count")
        outputs = tuple(raw[name] for name in _COMPACT_OUTPUT_FIELDS)
        if raw["path_count"] != outputs[1].shape[0]:
            raise RuntimeError("native compact finalizer returned inconsistent K")
        return outputs

    @staticmethod
    def setup_context(ctx, inputs, output):
        ctx.set_materialize_grads(False)
        ctx.candidate_count = int(inputs[0].shape[0])
        ctx.sequence_width = int(inputs[8].shape[1])
        saved = tuple(
            torch.autograd.forward_ad.unpack_dual(value).primal
            for value in (output[5], output[0])
        )
        ctx.save_for_backward(*saved)
        ctx.save_for_forward(*saved)
        ctx.mark_non_differentiable(*output[:_DISCRETE_OUTPUT_COUNT])

    @staticmethod
    @_ad_first_order_only
    def backward(ctx, *grad_outputs):
        none_grads = (None,) * 24
        continuous_grads = grad_outputs[_DISCRETE_OUTPUT_COUNT:]
        if all(value is None for value in continuous_grads):
            return none_grads
        if not any(ctx.needs_input_grad[11:22]):
            return none_grads
        from witwin.channel.kernels.topology import (
            evaluated_paths_compact_finalize_backward,
        )

        valid, selected_row_index = ctx.saved_tensors
        raw = evaluated_paths_compact_finalize_backward(
            valid,
            selected_row_index,
            *continuous_grads,
            candidate_count=ctx.candidate_count,
            sequence_width=ctx.sequence_width,
        )
        if not isinstance(raw, dict) or set(raw) != set(_CONTINUOUS_FIELDS):
            raise TypeError("native compact finalizer backward returned bad fields")
        return (
            *(None for _ in range(11)),
            *(
                raw[name] if ctx.needs_input_grad[index] else None
                for index, name in enumerate(_CONTINUOUS_FIELDS, start=11)
            ),
            None,
            None,
        )

    @staticmethod
    def jvp(ctx, *tangents):
        continuous_tangents = tuple(
            _ad_native_tangent_or_none(value) for value in tangents[11:22]
        )
        if all(value is None for value in continuous_tangents):
            return (None,) * len(_COMPACT_OUTPUT_FIELDS)
        from witwin.channel.kernels.topology import (
            evaluated_paths_compact_finalize_jvp,
        )

        valid, selected_row_index = (
            _ad_native_tensor(value) for value in ctx.saved_tensors
        )
        with disable_functorch():
            raw = evaluated_paths_compact_finalize_jvp(
                valid,
                selected_row_index,
                *continuous_tangents,
                candidate_count=ctx.candidate_count,
                sequence_width=ctx.sequence_width,
            )
        if not isinstance(raw, dict) or set(raw) != set(_CONTINUOUS_FIELDS):
            raise TypeError("native compact finalizer JVP returned bad fields")
        return (
            *(None for _ in range(_DISCRETE_OUTPUT_COUNT)),
            *(raw[name] for name in _CONTINUOUS_FIELDS),
        )


def compact_evaluated_paths(
    paths: EvaluatedPaths,
    *,
    source_stable_ids: torch.Tensor,
    sink_stable_ids: torch.Tensor,
    rows_are_compact: bool = False,
) -> CompactEvaluatedPaths:
    """Publish exact valid rows and pair segmentation from the sole native owner."""

    from witwin.channel.propagation.rows import (
        EvaluatedPaths,
        PathFields,
        PathGeometry,
        PathTopology,
    )

    tensors = _validate_inputs(paths, source_stable_ids, sink_stable_ids)
    if rows_are_compact:
        native = evaluated_paths_compact_finalize(
            *tensors,
            source_stable_ids,
            sink_stable_ids,
            rows_are_compact=True,
        )
        expected = {"path_count", *_COMPACT_OUTPUT_FIELDS}
        if not isinstance(native, dict) or set(native) != expected:
            raise TypeError("native exact-row finalizer returned bad fields")
        if native["path_count"] != paths.row_count:
            raise RuntimeError("native exact-row finalizer changed K")
        for name, tensor in zip(
            (*_TOPOLOGY_FIELDS, *_CONTINUOUS_FIELDS), tensors, strict=True
        ):
            alias = native[name]
            if (
                alias.data_ptr() != tensor.data_ptr()
                or alias.shape != tensor.shape
                or alias.stride() != tensor.stride()
            ):
                raise RuntimeError(
                    f"native exact-row finalizer copied payload field {name}"
                )
        raw = native
        evaluated = paths
    else:
        outputs = _CompactEvaluatedPathsFunction.apply(
            *tensors, source_stable_ids, sink_stable_ids
        )
        raw = dict(zip(_COMPACT_OUTPUT_FIELDS, outputs, strict=True))
        topology = PathTopology(
            **{name: raw[name] for name in _TOPOLOGY_FIELDS}
        )
        geometry = PathGeometry(
            row_identity=topology.row_identity,
            **{name: raw[name] for name in _CONTINUOUS_FIELDS[:7]},
        )
        fields = PathFields(
            row_identity=topology.row_identity,
            **{name: raw[name] for name in _CONTINUOUS_FIELDS[7:]},
        )
        evaluated = EvaluatedPaths(
            topology=topology,
            geometry=geometry,
            fields=fields,
        )
    return CompactEvaluatedPaths(
        path_count=int(raw["pair_index"].shape[0]),
        pair_index=raw["pair_index"],
        pair_offsets=raw["pair_offsets"],
        source_id=raw["source_id"],
        sink_id=raw["sink_id"],
        evaluated=evaluated,
        count_d2h_copies=(
            COMPACT_COUNT_D2H_COPIES
            if paths.row_count > 0 and not rows_are_compact
            else 0
        ),
        count_d2h_bytes=(
            COMPACT_COUNT_D2H_BYTES
            if paths.row_count > 0 and not rows_are_compact
            else 0
        ),
        count_synchronizations=(
            COMPACT_COUNT_SYNCHRONIZATIONS
            if paths.row_count > 0 and not rows_are_compact
            else 0
        ),
        native_launch_count=1,
    )


def consumer_los_jones(
    *,
    pair_index: torch.Tensor,
    source_positions: torch.Tensor,
    sink_positions: torch.Tensor,
    source_reference_basis: torch.Tensor,
    sink_reference_basis: torch.Tensor,
    frequency_hz: float,
) -> LoSJonesRows:
    """Evaluate primal-only LoS transport between row-specific transverse bases."""

    tensors = (
        pair_index,
        source_positions,
        sink_positions,
        source_reference_basis,
        sink_reference_basis,
    )
    if any(not isinstance(value, torch.Tensor) for value in tensors):
        raise TypeError("LoS Jones inputs must be torch tensors")
    if any(value.requires_grad for value in tensors):
        raise RuntimeError("LoS Jones transport does not support AD")
    raw = _required_native_op("consumer_los_jones")(
        pair_index,
        source_positions,
        sink_positions,
        source_reference_basis,
        sink_reference_basis,
        float(frequency_hz),
    )
    if not isinstance(raw, dict) or set(raw) != {
        "matrix",
        "source_basis",
        "sink_basis",
    }:
        raise TypeError("native LoS Jones operator returned bad fields")
    return LoSJonesRows(
        matrix=raw["matrix"],
        source_basis=raw["source_basis"],
        sink_basis=raw["sink_basis"],
        native_launch_count=1 if pair_index.shape[0] > 0 else 0,
    )


# --- Frozen line-of-sight gather ------------------------------------------

_ROW_FIELDS = (
    "source",
    "target",
    "tx_power",
    "tx_polarization",
    "rx_polarization",
)
_LOS_GATHER_OUTPUT_FIELDS = (*_ROW_FIELDS, "pair_index", "pair_offsets")
_ENDPOINT_GRAD_FIELDS = (
    "source_positions",
    "sink_positions",
    "source_powers",
    "source_polarizations",
    "sink_polarizations",
)


@dataclass(frozen=True, slots=True)
class FixedLoSRows:
    """Exact frozen LoS rows ready for the RayD-owned free-space field family."""

    source: torch.Tensor
    target: torch.Tensor
    tx_power: torch.Tensor
    tx_polarization: torch.Tensor
    rx_polarization: torch.Tensor
    pair_index: torch.Tensor
    pair_offsets: torch.Tensor
    validation_d2h_copies: int
    validation_d2h_bytes: int
    validation_synchronizations: int

    @property
    def row_count(self) -> int:
        return int(self.pair_index.shape[0])


class _FixedLoSGatherFunction(torch.autograd.Function):
    @staticmethod
    def forward(*inputs):
        raw = _required_native_op("consumer_fixed_los_gather")(*inputs)
        if not isinstance(raw, dict) or set(raw) != set(
            _LOS_GATHER_OUTPUT_FIELDS
        ):
            raise TypeError("native fixed LoS gather returned bad fields")
        return tuple(raw[name] for name in _LOS_GATHER_OUTPUT_FIELDS)

    @staticmethod
    def setup_context(ctx, inputs, output):
        ctx.set_materialize_grads(False)
        source_index, sink_index = (
            torch.autograd.forward_ad.unpack_dual(value).primal
            for value in inputs[:2]
        )
        ctx.source_count = int(inputs[6].shape[0])
        ctx.sink_count = int(inputs[7].shape[0])
        ctx.save_for_backward(source_index, sink_index)
        ctx.save_for_forward(source_index, sink_index)
        ctx.mark_non_differentiable(*output[5:])

    @staticmethod
    @_ad_first_order_only
    def backward(ctx, *grad_outputs):
        endpoint_grads = grad_outputs[:5]
        if all(value is None for value in endpoint_grads):
            return (None,) * 13
        source_index, sink_index = ctx.saved_tensors
        raw = _required_native_op("consumer_fixed_los_gather_backward")(
            source_index,
            sink_index,
            *endpoint_grads,
            ctx.source_count,
            ctx.sink_count,
        )
        if not isinstance(raw, dict) or set(raw) != set(_ENDPOINT_GRAD_FIELDS):
            raise TypeError("native fixed LoS gather backward returned bad fields")
        return (
            *(None for _ in range(6)),
            *(
                raw[name] if ctx.needs_input_grad[index] else None
                for index, name in enumerate(_ENDPOINT_GRAD_FIELDS, start=6)
            ),
            None,
            None,
        )

    @staticmethod
    def jvp(ctx, *tangents):
        endpoint_tangents = tuple(
            _ad_native_tangent_or_none(value) for value in tangents[6:11]
        )
        if all(value is None for value in endpoint_tangents):
            return (None,) * len(_LOS_GATHER_OUTPUT_FIELDS)
        source_index, sink_index = (
            _ad_native_tensor(value) for value in ctx.saved_tensors
        )
        with disable_functorch():
            raw = _required_native_op("consumer_fixed_los_gather_jvp")(
                source_index,
                sink_index,
                *endpoint_tangents,
                ctx.source_count,
                ctx.sink_count,
            )
        if not isinstance(raw, dict) or set(raw) != set(_ROW_FIELDS):
            raise TypeError("native fixed LoS gather jvp returned bad fields")
        return (*(raw[name] for name in _ROW_FIELDS), None, None)


def fixed_los_gather(
    topology: PropagationTopology,
    sources: EndpointBatch,
    sinks: EndpointBatch,
) -> FixedLoSRows:
    """Validate and gather frozen LoS rows without Python/Torch indexing."""

    if not isinstance(topology, PropagationTopology):
        raise TypeError("topology must be a PropagationTopology")
    if not isinstance(sources, EndpointBatch) or not isinstance(sinks, EndpointBatch):
        raise TypeError("sources and sinks must be EndpointBatch instances")
    if sources.powers_w is None:
        raise ValueError("sources.powers_w is required")
    if sinks.powers_w is not None:
        raise ValueError("sinks.powers_w must be absent")
    if topology.device != sources.device or topology.device != sinks.device:
        raise ValueError("topology and endpoint tensors must share one CUDA device")

    values = _FixedLoSGatherFunction.apply(
        topology.source_index,
        topology.sink_index,
        topology.source_id,
        topology.sink_id,
        topology.depth,
        topology.component_id,
        sources.positions_m,
        sinks.positions_m,
        sources.powers_w,
        sources.polarizations,
        sinks.polarizations,
        sources.stable_ids,
        sinks.stable_ids,
    )
    rows = int(topology.source_index.shape[0])
    return FixedLoSRows(
        **dict(zip(_LOS_GATHER_OUTPUT_FIELDS, values, strict=True)),
        validation_d2h_copies=1 if rows else 0,
        validation_d2h_bytes=4 if rows else 0,
        validation_synchronizations=1 if rows else 0,
    )


def fixed_los_geometry_live(rows: FixedLoSRows) -> bool:
    """forward-mode liveness liveness for the raw frozen line-of-sight route.

 A zero-interaction row is a function of its two gathered endpoints alone, so
 this is the complete liveness question for that route. It is answered here,
 once, from the gathered rows, above any frequency-column loop that replays
 them.
 """

    return _ad_geometry_live(rows.source, rows.target)


def require_fixed_los_geometry_live(rows: FixedLoSRows, decided: bool) -> None:
    """Fail loudly if one column disagrees with the hoisted decision.

 The field facade keeps deciding liveness for itself - that is its forward-mode liveness
 contract - and this makes "every column decides the same thing" a checked
 invariant instead of an assumption, before the operator runs.
 """

    if fixed_los_geometry_live(rows) != decided:
        raise RuntimeError(
            "fixed LoS geometry liveness disagrees with the decision taken "
            f"above the frequency-column loop (decided {decided}); every "
            "column must answer the same ADR-038 question"
        )


# --- Structural row selection for a prepared frozen topology ---------------

# Same bit vocabulary the native LoS validator publishes, so a caller reading
# the two error messages does not have to learn two encodings. Depth and
# component (bits 4 and 8) are host-validated once by ``prepare_fixed_topology``
# and therefore never fire here.
_INDEX_BOUNDS = 1
_PAIR_ORDER = 2
_STABLE_ID = 16
# A slot-batched row whose sink does not live in the slot its source does. The
# native validator has no such bit because slot batching is host-side
# structural packing; the number continues the same vocabulary so a caller
# reading a bitmask never has to learn two encodings.
_SLOT_BLOCK = 32

VALIDATION_D2H_COPIES = 1
VALIDATION_D2H_BYTES = 4
VALIDATION_SYNCHRONIZATIONS = 1


@dataclass(frozen=True, slots=True)
class PreparedRows:
    """Frozen rows bound to the current endpoint batches."""

    source: torch.Tensor
    target: torch.Tensor
    tx_power: torch.Tensor
    tx_polarization: torch.Tensor
    rx_polarization: torch.Tensor
    source_row_index: torch.Tensor
    sink_row_index: torch.Tensor
    pair_index: torch.Tensor
    pair_offsets: torch.Tensor
    validation_d2h_copies: int
    validation_d2h_bytes: int
    validation_synchronizations: int

    @property
    def row_count(self) -> int:
        return int(self.pair_index.shape[0])


def _bit(flag: torch.Tensor, bit: int) -> torch.Tensor:
    return flag.to(dtype=torch.int32) * bit


def _order_violation(
    pair_index: torch.Tensor, in_bounds: torch.Tensor
) -> torch.Tensor:
    """True when an in-bounds row breaks non-decreasing pair-major order."""

    if pair_index.shape[0] < 2:
        return torch.zeros((), dtype=torch.bool, device=pair_index.device)
    descending = pair_index[1:] < pair_index[:-1]
    return (descending & in_bounds[1:] & in_bounds[:-1]).any()


def _contract_error(
    topology: PropagationTopology,
    sources: EndpointBatch,
    sinks: EndpointBatch,
    source_row_index: torch.Tensor,
    sink_row_index: torch.Tensor,
    pair_index: torch.Tensor,
    slot_broken: torch.Tensor | None,
) -> torch.Tensor:
    """One device-resident int32 bitmask covering every frozen row."""

    in_bounds = (
        (source_row_index >= 0)
        & (source_row_index < sources.count)
        & (sink_row_index >= 0)
        & (sink_row_index < sinks.count)
    )
    clamped_source = source_row_index.clamp(0, sources.count - 1)
    clamped_sink = sink_row_index.clamp(0, sinks.count - 1)
    identity_broken = (
        (topology.source_id != sources.stable_ids[clamped_source])
        | (topology.sink_id != sinks.stable_ids[clamped_sink])
    ) & in_bounds
    error = (
        _bit((~in_bounds).any(), _INDEX_BOUNDS)
        | _bit(_order_violation(pair_index, in_bounds), _PAIR_ORDER)
        | _bit(identity_broken.any(), _STABLE_ID)
    )
    if slot_broken is None:
        return error
    return error | _bit((slot_broken & in_bounds).any(), _SLOT_BLOCK)


def _slot_pairing(
    source_row_index: torch.Tensor,
    sink_row_index: torch.Tensor,
    slot_sources: int,
    slot_sinks: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Block-diagonal pair index, plus the rows that break the block law.

 A row belongs to the slot its SOURCE lives in. Its sink must live in the
 same slot, which is exactly the statement that the sink's within-slot index
 lands in ``[0, slot_sinks)``; a row that pairs across slots would otherwise
 silently land in another slot's pair segment.
 """

    slot = source_row_index.div(slot_sources, rounding_mode="floor")
    source_slot_index = source_row_index - slot * slot_sources
    sink_slot_index = sink_row_index - slot * slot_sinks
    pair_index = (
        slot * (slot_sources * slot_sinks)
        + sink_slot_index * slot_sources
        + source_slot_index
    )
    broken = (sink_slot_index < 0) | (sink_slot_index >= slot_sinks)
    return pair_index, broken


def _pair_segmentation(
    pair_index: torch.Tensor, pair_count: int
) -> torch.Tensor:
    counts = torch.zeros(
        (pair_count + 1,), dtype=torch.int64, device=pair_index.device
    )
    counts.index_add_(
        0,
        pair_index + 1,
        torch.ones_like(pair_index),
    )
    return counts.cumsum(0)


def prepared_row_gather(
    topology: PropagationTopology,
    sources: EndpointBatch,
    sinks: EndpointBatch,
    *,
    slot_count: int = 1,
) -> PreparedRows:
    """Validate frozen rows and bind them to the current endpoint batches.

 ``slot_count`` selects the pairing law. One slot is the outer product the
 consumer has always published; more than one is the block-diagonal layout
 of:attr:`PropagationConvention.slot_pair_layout`, under which ``pair_count``
 grows linearly rather than quadratically in the slot count. The validation
 budget is unchanged: one four-byte copy and one synchronization for the
 whole batch, whatever the slot count.
 """

    if topology.device != sources.device or topology.device != sinks.device:
        raise ValueError("topology and endpoint tensors must share one CUDA device")
    assert sources.powers_w is not None
    rows = topology.row_count
    source_count = sources.count
    sink_count = sinks.count
    if rows > 0 and (source_count == 0 or sink_count == 0):
        raise ValueError("non-empty frozen topology requires endpoint rows")
    slot_sources = source_count // slot_count
    slot_sinks = sink_count // slot_count
    source_row_index = topology.source_index.to(dtype=torch.int64)
    sink_row_index = topology.sink_index.to(dtype=torch.int64)
    if slot_count == 1:
        pair_index = sink_row_index * source_count + source_row_index
        slot_broken = None
    else:
        # A zero per-slot count only ever accompanies an empty frozen batch,
        # rejected above when it is not; clamp the divisor so an empty batch
        # does not divide by zero on its way to an empty result.
        pair_index, slot_broken = _slot_pairing(
            source_row_index,
            sink_row_index,
            max(slot_sources, 1),
            max(slot_sinks, 1),
        )
    pair_count = slot_count * slot_sources * slot_sinks
    if rows > 0:
        error = int(
            _contract_error(
                topology,
                sources,
                sinks,
                source_row_index,
                sink_row_index,
                pair_index,
                slot_broken,
            ).item()
        )
        if error != 0:
            raise ValueError(
                "frozen topology validation failed against the current "
                f"endpoint batches (error bitmask {error})"
            )
    return PreparedRows(
        source=sources.positions_m.index_select(0, source_row_index).contiguous(),
        target=sinks.positions_m.index_select(0, sink_row_index).contiguous(),
        tx_power=sources.powers_w.index_select(0, source_row_index).contiguous(),
        tx_polarization=(
            sources.polarizations.index_select(0, source_row_index).contiguous()
        ),
        rx_polarization=(
            sinks.polarizations.index_select(0, sink_row_index).contiguous()
        ),
        source_row_index=source_row_index,
        sink_row_index=sink_row_index,
        pair_index=pair_index,
        pair_offsets=_pair_segmentation(pair_index, pair_count),
        validation_d2h_copies=VALIDATION_D2H_COPIES if rows else 0,
        validation_d2h_bytes=VALIDATION_D2H_BYTES if rows else 0,
        validation_synchronizations=VALIDATION_SYNCHRONIZATIONS if rows else 0,
    )


def select_rows(values: torch.Tensor, rows: torch.Tensor) -> torch.Tensor:
    """Row selection that preserves frozen order and contiguity."""

    return values.index_select(0, rows).contiguous()


# --- Source amplitude ------------------------------------------------------


def excited_field(
    field_vector: torch.Tensor, tx_power: torch.Tensor, *, ad_mode: str
) -> torch.Tensor:
    """Return the source-excited complex3 field for a unit-excitation one.

 The field transport kernels carry ``sqrt(tx_power)`` into ``path_field`` and
 ``path_gain`` but leave their complex3 vector at unit excitation, so there
 is no excited vector on that launch. source excitation adds the native owner of
 exactly that quantity; this only chooses between its primal and
 differentiable entry points. No amplitude is computed here.
 """

    from witwin.channel.kernels import fields as field_kernels

    if ad_mode == "none":
        return field_kernels.field_source_amplitude_scale(
            field_vector, tx_power
        )["path_field_vector"]
    return field_kernels.field_source_amplitude_scale_ad(field_vector, tx_power)


# --- Composed source-basis to sink-basis Jones operator --------------------

_FieldOp = Callable[[torch.Tensor], dict[str, torch.Tensor]]


def _primal(value: torch.Tensor) -> torch.Tensor:
    """Detached primal view for an input a native op consumes as a constant."""

    primal = torch.autograd.forward_ad.unpack_dual(value).primal
    return primal.detach().contiguous()


def transverse_basis(
    reference_basis: torch.Tensor,
    leg_origin: torch.Tensor,
    leg_target: torch.Tensor,
    *,
    frequency_hz: float,
) -> torch.Tensor:
    """Row-aligned orthonormal basis transverse to ``leg_target - leg_origin``.

 ``consumer_los_jones`` indexes its endpoint tables through ``pair_index``,
 so handing it per-row tables and the diagonal pair index makes it evaluate
 exactly one leg per row. The same reference basis is supplied for both
 endpoints, which makes its two published bases identical, and the source
 one is returned. This reuses the shipped native endpoint-basis owner
 instead of restating its projection and orthonormalization in Torch.
 """

    rows = int(leg_origin.shape[0])
    pair_index = torch.arange(
        rows, device=leg_origin.device, dtype=torch.int64
    ) * (rows + 1)
    reference = _primal(reference_basis)
    return consumer_los_jones(
        pair_index=pair_index,
        source_positions=_primal(leg_origin),
        sink_positions=_primal(leg_target),
        source_reference_basis=reference,
        sink_reference_basis=reference,
        frequency_hz=frequency_hz,
    ).source_basis


def _project(
    field_vector: torch.Tensor,
    direction: torch.Tensor,
    sink_vector: torch.Tensor,
) -> torch.Tensor:
    from witwin.channel.kernels import fields as field_kernels

    return field_kernels.field_project_complex3_ad(
        field_vector, direction, sink_vector
    )["coefficient"]


def compose_jones(
    excite: _FieldOp,
    *,
    source_basis: torch.Tensor,
    sink_reference_basis: torch.Tensor,
    arrival_origin: torch.Tensor,
    arrival_target: torch.Tensor,
    frequency_hz: float,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    """Build the ``(N, 2, 2)`` operator from two excitations of ``excite``.

 ``matrix[k, i, j]`` is the response of sink basis vector ``i`` to source
 basis vector ``j``, which is the index convention the native LoS Jones
 owner publishes. Returns the operator, the sink basis, and the first
 column's full field result so the caller can publish its geometry without
 re-running the transport.
 """

    columns = tuple(
        excite(source_basis[:, index].contiguous()) for index in (0, 1)
    )
    sink_basis = transverse_basis(
        sink_reference_basis,
        arrival_origin,
        arrival_target,
        frequency_hz=frequency_hz,
    )
    direction = columns[0]["direction"]
    matrix = torch.stack(
        [
            torch.stack(
                [
                    _project(
                        column["field_vector"],
                        direction,
                        sink_basis[:, index].contiguous(),
                    )
                    for column in columns
                ],
                dim=-1,
            )
            for index in (0, 1)
        ],
        dim=-2,
    )
    return matrix, sink_basis, columns[0]


# --- Prepared-topology bucket replay ---------------------------------------

_MATERIAL_FIELDS = ("eps_r", "sigma_e", "mu_r", "gain", "thickness")


@dataclass(frozen=True, slots=True)
class GeometryLiveness:
    """forward-mode liveness liveness, decided once above every loop that reuses it.

 forward-mode liveness the conditional differentiability of ``path_length_m`` and
 ``delay_s`` to be decided where forward duals are still visible, because
 ``Function.apply`` unpacks them before ``setup_context`` runs. A wideband
 request evaluates the same frozen rows at several frequencies, so it drives
 the same field operators repeatedly over identical geometry inputs.
 Deciding liveness inside that loop, or letting the first column decide for
 the rest, is exactly the shape of the defect forward-mode liveness

 So the decision is made ONCE here, above the column loop, from the inputs
 every column shares, and:meth:`require` re-asserts it against the actual
 operator inputs of every bucket of every column. The field facades keep
 deciding for themselves - that is their forward-mode liveness contract and their frozen
 surface - and this record makes it a checked invariant that they all decide
 the same thing, rather than an assumption. A disagreement is a loud failure
 before the operator runs, never a column that silently answers a different
 question than its siblings.

 Two flags rather than one because the two bucket kinds consume different
 geometry: a zero-depth line-of-sight row is a function of the endpoints
 alone, while a reflection row additionally resolves its stationary point
 against the scene vertices, so a differentiable mesh makes the reflection
 geometry live even when the endpoints are primal.
 """

    los: bool
    reflection: bool
    # first-order differentiation: whether the published arrival direction carries a derivative.
    # This is a whole-result decision taken from the request's component set,
    # never a per-row one: a batch that carries a component whose direction
    # seam Channel does not own publishes a fully detached field_direction for
    # every row rather than a live derivative for some rows and a silent zero
    # for the rest.
    direction_components: bool = True

    @classmethod
    def of(
        cls,
        source: torch.Tensor,
        target: torch.Tensor,
        vertices: object,
        *,
        direction_components: bool = True,
    ) -> GeometryLiveness:
        endpoints = _ad_geometry_live(source, target)
        return cls(
            los=endpoints,
            reflection=endpoints or _ad_geometry_live(vertices),
            direction_components=direction_components,
        )

    def at_depth(self, depth: int) -> bool:
        return self.reflection if depth else self.los

    def direction_at_depth(self, depth: int) -> bool:
        """Direction liveness for one bucket, under both decisions.

 A direction derivative is a geometry derivative, so it can only be live
 where the geometry is. The component half is host-known and identical
 for every bucket and every column, which is what makes the whole-result
 rule hold without a second device observation.
 """

        return self.at_depth(depth) and self.direction_components

    def require(self, depth: int, *values: object) -> None:
        """Fail loudly if one column's inputs disagree with the decision."""

        decided = self.at_depth(depth)
        if _ad_geometry_live(*values) != decided:
            raise RuntimeError(
                "fixed-topology geometry liveness disagrees with the decision "
                f"taken above the frequency-column loop (depth {depth}, "
                f"decided {decided}); every column must answer the same ADR-038 "
                "question"
            )


@dataclass(frozen=True, slots=True)
class BucketInputs:
    """Everything one bucket hands to the native transport."""

    depth: int
    source: torch.Tensor
    target: torch.Tensor
    tx_power: torch.Tensor
    tx_polarization: torch.Tensor
    rx_polarization: torch.Tensor
    interaction_positions: torch.Tensor
    interaction_normals: torch.Tensor
    material: tuple[torch.Tensor, ...]
    valid: torch.Tensor

    @property
    def row_count(self) -> int:
        return int(self.source.shape[0])


@dataclass(frozen=True, slots=True)
class FixedRowOutputs:
    """Frozen-order ``K`` row outputs of one reevaluation.

 ``path_field`` and ``path_field_vector`` are the source-excited transport,
 matching what discovery publishes. ``path_field_vector`` is only produced
 for the complex3 response, which is the only reader of it.
 """

    path_field: torch.Tensor
    path_field_vector: torch.Tensor | None
    path_length_m: torch.Tensor
    delay_s: torch.Tensor
    direction: torch.Tensor
    interaction_positions: torch.Tensor
    interaction_normals: torch.Tensor
    matrix: torch.Tensor | None
    source_basis: torch.Tensor | None
    sink_basis: torch.Tensor | None
    row_valid: torch.Tensor | None


def require_smooth_reflection_scene(compiled: CompiledScene) -> None:
    """Reject a scene whose reflection rows need rough-surface attenuation.

 The coherent rough-reflection factor and the realization phase-screen delta
 replacement are owned by the discovery-side field loop and are gated on
 host material state there. Reproducing that gate here would duplicate
 another owner's policy, and silently disagreeing with ``evaluate`` on a
 rough scene is worse than refusing, so this route requires a smooth scene.
 """

    if bool((compiled.materials.scatter_model_id == 1).any()):
        raise NotImplementedError(
            "fixed-topology reflection reevaluation requires a smooth scene; "
            "rough-surface coherent attenuation is owned by the discovery "
            "field loop and is not reproduced here"
        )
    screens = getattr(compiled.assignments, "structure_phase_screens", {})
    if any(
        getattr(screen, "mode", None) == "realization_coherent"
        for screen in screens.values()
    ):
        raise NotImplementedError(
            "fixed-topology reflection reevaluation does not support "
            "realization_coherent phase screens"
        )


def _scene_tables(compiled: CompiledScene) -> dict[str, object]:
    # CompiledScene owns the lazy scene-static cache ; the
    # replay just consumes it. The primal per-frame case stages host-to-device
    # once per compiled scene instead of once per call.
    return compiled.fixed_reevaluation_tables()


def _reflection_inputs(
    compiled: CompiledScene,
    bucket: FixedTopologyBucket,
    tables: dict[str, object],
    prepared: PreparedFixedTopology,
    rows: PreparedRows,
) -> BucketInputs:
    from witwin.channel.propagation.geometry import (
        reflection_epc_paths,
    )

    depth = bucket.depth
    sequence = select_rows(
        prepared.topology.primitive_sequence, bucket.rows
    )[:, :depth].contiguous()
    face_id = sequence.to(dtype=torch.int64)
    source = select_rows(rows.source, bucket.rows)
    target = select_rows(rows.target, bucket.rows)
    epc = reflection_epc_paths(
        compiled, tables["vertices"], source, target, face_id, depth
    )
    valid = epc["valid"]
    material = tables["material"]
    return BucketInputs(
        depth=depth,
        source=source,
        target=target,
        tx_power=select_rows(rows.tx_power, bucket.rows),
        tx_polarization=_inert_where_invalid(
            select_rows(rows.tx_polarization, bucket.rows), valid
        ),
        rx_polarization=select_rows(rows.rx_polarization, bucket.rows),
        interaction_positions=epc["hit_positions"],
        interaction_normals=epc["normals"],
        material=tuple(material[name][face_id].contiguous() for name in _MATERIAL_FIELDS),
        valid=valid,
    )


def _los_inputs(
    compiled: CompiledScene, bucket: FixedTopologyBucket, rows: PreparedRows
) -> BucketInputs:
    source = select_rows(rows.source, bucket.rows)
    target = select_rows(rows.target, bucket.rows)
    empty = source.new_empty((int(source.shape[0]), 0, 3))
    # Re-test visibility with the same native gate discovery applies to LoS
    # candidates, so a sink that moved behind a wall publishes row_valid=False
    # and exact zeros instead of a full-strength free-space answer. A
    # structure-less scene cannot occlude anything and skips the launch.
    if compiled.structures:
        from witwin.channel.propagation.geometry import (
            VisibilityQuery,
            run_visibility_query,
        )

        valid = run_visibility_query(
            VisibilityQuery(
                rayd=compiled.rayd,
                start=source.contiguous(),
                end=target.contiguous(),
                active=None,
            )
        ).visible
    else:
        valid = torch.ones(
            (int(source.shape[0]),), dtype=torch.bool, device=source.device
        )
    return BucketInputs(
        depth=0,
        source=source,
        target=target,
        tx_power=select_rows(rows.tx_power, bucket.rows),
        tx_polarization=_inert_where_invalid(
            select_rows(rows.tx_polarization, bucket.rows), valid
        ),
        rx_polarization=select_rows(rows.rx_polarization, bucket.rows),
        interaction_positions=empty,
        interaction_normals=empty,
        material=(),
        valid=valid,
    )


def _inert_where_invalid(
    values: torch.Tensor, valid: torch.Tensor
) -> torch.Tensor:
    """Select a row's value or the inert constant; never a numerical blend."""

    shape = (-1, *((1,) * (values.ndim - 1)))
    return torch.where(valid.reshape(shape), values, values.new_zeros(()))


def _field_op(
    inputs: BucketInputs,
    *,
    ad_mode: str,
    frequency: float | torch.Tensor,
    frequency_value: float,
    geometry_live: GeometryLiveness | None,
    ledger: object | None = None,
) -> Callable[[torch.Tensor, torch.Tensor], dict[str, torch.Tensor]]:
    from witwin.channel.kernels import fields as field_kernels

    differentiable = ad_mode != "none"

    def run(
        tx_polarization: torch.Tensor, rx_polarization: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        if inputs.depth == 0:
            leading = (inputs.source, inputs.target)
            trailing: tuple[torch.Tensor, ...] = ()
        else:
            leading = (
                inputs.source,
                inputs.target,
                inputs.interaction_positions,
                inputs.interaction_normals,
            )
            trailing = inputs.material
        if geometry_live is not None:
            # The liveness these exact inputs imply must equal the one decided
            # above the frequency-column loop, for every bucket of every column.
            geometry_live.require(inputs.depth, *leading)
        arguments = (
            *leading,
            inputs.tx_power,
            tx_polarization,
            rx_polarization,
            *trailing,
        )
        if differentiable:
            operator = (
                field_kernels.field_free_space_ad
                if inputs.depth == 0
                else field_kernels.field_reflection_sequence_ad
            )
            if ledger is not None:
                ledger.add(*arguments)
            return operator(
                *arguments,
                frequency=frequency,
                frequency_value=frequency_value,
                direction_live=(
                    geometry_live is not None
                    and geometry_live.direction_at_depth(inputs.depth)
                ),
            )
        operator = (
            field_kernels.field_free_space
            if inputs.depth == 0
            else field_kernels.field_reflection_sequence
        )
        return operator(*arguments, frequency_hz=frequency_value)

    return run


def _leg_endpoints(inputs: BucketInputs) -> tuple[torch.Tensor, torch.Tensor]:
    """Where the first leg ends and where the last leg starts.

 A reflection row launches toward its first interaction and arrives from
 its last one, so its transverse bases live in two different planes. Both
 are read off the interaction table the native transport itself consumes;
 neither direction is recomputed here.
 """

    if inputs.depth == 0:
        return inputs.target, inputs.source
    return (
        inputs.interaction_positions[:, 0].contiguous(),
        inputs.interaction_positions[:, inputs.depth - 1].contiguous(),
    )


def _jones_values(
    inputs: BucketInputs,
    run: Callable[[torch.Tensor, torch.Tensor], dict[str, torch.Tensor]],
    *,
    source_reference: torch.Tensor,
    sink_reference: torch.Tensor,
    frequency_value: float,
) -> dict[str, torch.Tensor]:
    launch_target, arrival_origin = _leg_endpoints(inputs)
    source_basis = _inert_where_invalid(
        transverse_basis(
            source_reference,
            inputs.source,
            launch_target,
            frequency_hz=frequency_value,
        ),
        inputs.valid,
    )
    matrix, sink_basis, column = compose_jones(
        lambda polarization: run(polarization, polarization),
        source_basis=source_basis,
        sink_reference_basis=sink_reference,
        arrival_origin=arrival_origin,
        arrival_target=inputs.target,
        frequency_hz=frequency_value,
    )
    return {
        **column,
        "matrix": matrix,
        "source_basis": source_basis,
        "sink_basis": _inert_where_invalid(sink_basis, inputs.valid),
    }


def _bucket_values(
    inputs: BucketInputs,
    run: Callable[[torch.Tensor, torch.Tensor], dict[str, torch.Tensor]],
    *,
    response: str,
    ad_mode: str,
    source_reference: torch.Tensor | None,
    sink_reference: torch.Tensor | None,
    frequency_value: float,
    ledger: object | None = None,
) -> dict[str, torch.Tensor]:
    if response != "polarimetric_transport":
        values = run(inputs.tx_polarization, inputs.rx_polarization)
    else:
        assert source_reference is not None and sink_reference is not None
        values = _jones_values(
            inputs,
            run,
            source_reference=source_reference,
            sink_reference=sink_reference,
            frequency_value=frequency_value,
        )
    if response != "complex3_transport":
        return values
    if ledger is not None and ad_mode != "none":
        ledger.add(values["field_vector"], inputs.tx_power)
    return {
        **values,
        "path_field_vector": excited_field(
            values["field_vector"], inputs.tx_power, ad_mode=ad_mode
        ),
    }


def _pad_interactions(values: torch.Tensor, width: int) -> torch.Tensor:
    depth = int(values.shape[1])
    if depth == width:
        return values
    padding = values.new_zeros((int(values.shape[0]), width - depth, 3))
    return torch.cat((values, padding), dim=1)


def _publish_bucket(
    outputs: dict[str, torch.Tensor],
    values: dict[str, torch.Tensor],
    inputs: BucketInputs,
    bucket: FixedTopologyBucket,
    width: int,
) -> None:
    rows = bucket.rows
    valid = inputs.valid
    outputs["path_field"].index_copy_(0, rows, values["path_field"])
    if outputs["path_field_vector"] is not None:
        outputs["path_field_vector"].index_copy_(
            0, rows, values["path_field_vector"]
        )
    outputs["path_length_m"].index_copy_(
        0, rows, _inert_where_invalid(values["path_length_m"], valid)
    )
    outputs["delay_s"].index_copy_(
        0, rows, _inert_where_invalid(values["delay_s"], valid)
    )
    outputs["direction"].index_copy_(
        0, rows, _inert_where_invalid(values["direction"], valid)
    )
    outputs["interaction_positions"].index_copy_(
        0, rows, _pad_interactions(inputs.interaction_positions, width)
    )
    outputs["interaction_normals"].index_copy_(
        0, rows, _pad_interactions(inputs.interaction_normals, width)
    )
    outputs["row_valid"].index_copy_(0, rows, valid)
    for name in ("matrix", "source_basis", "sink_basis"):
        if outputs[name] is not None:
            outputs[name].index_copy_(0, rows, values[name])


def _allocate(
    rows: PreparedRows, width: int, response: str
) -> dict[str, torch.Tensor | None]:
    count = rows.row_count
    device = rows.source.device
    polarimetric = response == "polarimetric_transport"
    return {
        "path_field": torch.zeros(
            (count,), dtype=torch.complex64, device=device
        ),
        "path_field_vector": (
            torch.zeros((count, 3), dtype=torch.complex64, device=device)
            if response == "complex3_transport"
            else None
        ),
        "path_length_m": torch.zeros((count,), device=device),
        "delay_s": torch.zeros((count,), device=device),
        "direction": torch.zeros((count, 3), device=device),
        "interaction_positions": torch.zeros((count, width, 3), device=device),
        "interaction_normals": torch.zeros((count, width, 3), device=device),
        "row_valid": torch.ones((count,), dtype=torch.bool, device=device),
        "matrix": (
            torch.zeros((count, 2, 2), dtype=torch.complex64, device=device)
            if polarimetric
            else None
        ),
        "source_basis": (
            torch.zeros((count, 2, 3), device=device) if polarimetric else None
        ),
        "sink_basis": (
            torch.zeros((count, 2, 3), device=device) if polarimetric else None
        ),
    }


def evaluate_prepared(
    compiled: CompiledScene,
    prepared: PreparedFixedTopology,
    rows: PreparedRows,
    *,
    response: str,
    ad_mode: str,
    frequency: float | torch.Tensor,
    frequency_value: float,
    source_reference_basis: torch.Tensor | None,
    sink_reference_basis: torch.Tensor | None,
    publish_row_validity: bool,
    geometry_live: GeometryLiveness | None = None,
    ledger: object | None = None,
) -> FixedRowOutputs:
    """Replay every host-known bucket of a prepared frozen topology.

 ``geometry_live`` carries an forward-mode liveness liveness decision the caller took
 before this call. A wideband caller runs this replay once per frequency
 column over identical geometry inputs and passes the SAME record to every
 column, which turns "every column answers the same liveness question" into a
 checked invariant. A single-frequency caller leaves it ``None``.
 """

    width = int(prepared.topology.primitive_sequence.shape[1])
    outputs = _allocate(rows, width, response)
    tables = (
        _scene_tables(compiled)
        if any(bucket.depth > 0 for bucket in prepared.buckets)
        else {}
    )
    for bucket in prepared.buckets:
        inputs = (
            _los_inputs(compiled, bucket, rows)
            if bucket.depth == 0
            else _reflection_inputs(compiled, bucket, tables, prepared, rows)
        )
        values = _bucket_values(
            inputs,
            _field_op(
                inputs,
                ad_mode=ad_mode,
                frequency=frequency,
                frequency_value=frequency_value,
                geometry_live=geometry_live,
                ledger=ledger,
            ),
            response=response,
            ad_mode=ad_mode,
            source_reference=(
                None
                if source_reference_basis is None
                else select_rows(source_reference_basis, bucket.rows)
            ),
            sink_reference=(
                None
                if sink_reference_basis is None
                else select_rows(sink_reference_basis, bucket.rows)
            ),
            frequency_value=frequency_value,
            ledger=ledger,
        )
        _publish_bucket(outputs, values, inputs, bucket, width)
    return FixedRowOutputs(
        path_field=outputs["path_field"],
        path_field_vector=outputs["path_field_vector"],
        path_length_m=outputs["path_length_m"],
        delay_s=outputs["delay_s"],
        direction=outputs["direction"],
        interaction_positions=outputs["interaction_positions"],
        interaction_normals=outputs["interaction_normals"],
        matrix=outputs["matrix"],
        source_basis=outputs["source_basis"],
        sink_basis=outputs["sink_basis"],
        row_valid=outputs["row_valid"] if publish_row_validity else None,
    )


def scene_vertex_table(compiled: CompiledScene) -> object:
    """The scene-static vertex tensor the reflection stationary point uses.

 Exposed so a caller can decide forward-mode liveness liveness before the first
 bucket runs without reaching past this module into the compiled scene's
 lazy table cache. Only a batch that actually carries a reflection bucket
 should ask: the tables are built lazily, and a line-of-sight replay must not
 start paying for them.
 """

    return _scene_tables(compiled)["vertices"]


# --- Orchestration: discovery and fixed-topology reevaluation -------------
#
# Consumer orchestration over the canonical enumerated and compact owners.


# ``_CAPABILITIES`` is the single frozen record built beside the vocabulary
# above; ``capabilities`` returns that same object, so the orchestration
# reads it directly rather than binding a second name to it.
_CONVENTION = PropagationConvention()


@dataclass(frozen=True, slots=True)
class _ConsumerConfig:
    max_depth: int
    components: frozenset[str]
    max_paths: int | None
    ad_mode: str
    max_paths_scope: str = "global"
    max_diffraction_order: int = 1
    coupled_paths: bool = False
    isb_boundary_taper: bool = False
    isb_boundary_taper_width: float = 0.5


@dataclass(frozen=True, slots=True)
class _ConsumerRows:
    evaluated: EvaluatedPaths
    path_count: int
    pair_index: torch.Tensor
    pair_offsets: torch.Tensor
    source_id: torch.Tensor
    sink_id: torch.Tensor
    count_d2h_copies: int
    count_d2h_bytes: int
    count_synchronizations: int


def _preflight_evaluate(
    compiled: object, request: object
) -> tuple[CompiledScene, PropagationRequest]:
    from witwin.channel.scene.compiler import CompiledScene

    if not isinstance(compiled, CompiledScene):
        raise TypeError("evaluate requires a CompiledScene")
    if not isinstance(request, PropagationRequest):
        raise TypeError("request must be a PropagationRequest")
    response_components = _CAPABILITIES.components_for(request.response)
    if not request.components.issubset(response_components):
        raise NotImplementedError(
            f"{request.response} does not support components "
            f"{sorted(request.components - response_components)}"
        )
    if request.ad_mode not in _CAPABILITIES.ad_modes_for(request.response):
        raise NotImplementedError(
            f"{request.response} does not support AD mode {request.ad_mode!r}"
        )
    unsupported_ad = sorted(
        component
        for component in request.components
        if request.ad_mode not in _CAPABILITIES.ad_modes_for_component(component)
    )
    if unsupported_ad:
        raise NotImplementedError(
            f"AD mode {request.ad_mode!r} is unsupported for components "
            f"{unsupported_ad}"
        )
    if request.response == "polarimetric_transport":
        _require_polarimetric_inputs(request)
    require_primal_only_ad_inputs(compiled, request)
    require_first_order_request(compiled, request)
    compiled.require_reference_frequency(request.reference_frequency_hz)
    return compiled, request


def _require_polarimetric_inputs(request: PropagationRequest) -> None:
    """Enforce the polarization-basis contract before any native work.

 The two transverse bases are structurally frozen: they reach the native
 field companions as the transmit and receive polarization, and those
 companions reject gradients on both. The primal-only fused LoS operator
 additionally rejects a differentiable endpoint or a tensor frequency, so
 a caller that wants derivatives asks for an AD mode and gets the composed
 operator instead.
 """

    if (
        request.sources.polarization_basis is None
        or request.sinks.polarization_basis is None
    ):
        raise ValueError(
            "polarimetric_transport requires source and sink "
            "polarization_basis tensors"
        )
    for name, value in (
        ("sources.polarization_basis", request.sources.polarization_basis),
        ("sinks.polarization_basis", request.sinks.polarization_basis),
    ):
        if carries_ad(value):
            raise NotImplementedError(
                f"polarimetric_transport {name} is primal-only; the operator "
                "is published in a frozen world-referenced transverse basis"
            )
    # The capability record declares tx_power and the endpoint polarizations
    # frozen too, and a declaration nobody enforces is the propagation consumer pattern.
    # The composed operator excites the transport with the two basis vectors,
    # so an endpoint polarization never reaches it at all and a gradient on one
    # could only ever come back empty; tx_power reaches a companion that does
    # not differentiate it. Refuse instead of returning a partial derivative.
    for name, value in (
        ("sources.powers_w", request.sources.powers_w),
        ("sources.polarizations", request.sources.polarizations),
        ("sinks.polarizations", request.sinks.polarizations),
    ):
        if carries_ad(value):
            raise NotImplementedError(
                f"polarimetric_transport {name} is primal-only; the operator "
                "is excited by the two transverse basis vectors and carries no "
                "derivative with respect to it"
            )
    if request.ad_mode != "none":
        return
    if isinstance(request.reference_frequency_hz, torch.Tensor):
        raise NotImplementedError(
            "polarimetric_transport requires a scalar compiled frequency"
        )
    if carries_ad(request.sources.positions_m) or carries_ad(
        request.sinks.positions_m
    ):
        raise NotImplementedError(
            "polarimetric_transport with ad_mode='none' is primal-only; "
            "request ad_mode='jvp' or ad_mode='vjp' for a differentiable "
            "operator"
        )


def _solver_scene(
    compiled: CompiledScene, sources: EndpointBatch, sinks: EndpointBatch
) -> tuple[SolverScene, EnumeratedEndpointTensors]:
    """Bind explicit request batches without consulting compiled endpoints."""

    from witwin.channel.propagation.enumerated import (
        EnumeratedEndpointTensors,
    )
    from witwin.channel.scene.endpoints import SolverScene

    assert sources.powers_w is not None
    endpoint_tensors = EnumeratedEndpointTensors(
        tx_positions=sources.positions_m,
        tx_power=sources.powers_w,
        tx_polarizations=sources.polarizations,
        rx_positions=sinks.positions_m,
        rx_polarizations=sinks.polarizations,
        tx_stable_ids=sources.stable_ids,
        rx_stable_ids=sinks.stable_ids,
    )
    return SolverScene(
        compiled=compiled,
        structures=compiled.structures,
        transmitters=(),
        receivers=(),
        frequency=compiled.reference_frequency_hz,
        metadata=compiled.source.metadata,
    ), endpoint_tensors


def _compact(
    evaluated: object,
    metadata: ExactPairMetadata | None,
) -> _ConsumerRows:
    from witwin.channel.propagation.rows import EvaluatedPaths

    if not isinstance(evaluated, EvaluatedPaths):
        raise TypeError("consumer source owner returned non-EvaluatedPaths")
    if metadata is None:
        raise RuntimeError("consumer source owner omitted compact pair metadata")
    if metadata.path_count != evaluated.row_count:
        raise RuntimeError("consumer source owner returned inconsistent exact K")
    if metadata.source_id is None or metadata.sink_id is None:
        raise RuntimeError("consumer source owner omitted endpoint stable IDs")
    return _ConsumerRows(
        evaluated=evaluated,
        path_count=metadata.path_count,
        pair_index=metadata.pair_index,
        pair_offsets=metadata.pair_offsets,
        source_id=metadata.source_id,
        sink_id=metadata.sink_id,
        count_d2h_copies=metadata.count_d2h_copies,
        count_d2h_bytes=metadata.count_d2h_bytes,
        count_synchronizations=metadata.count_synchronizations,
    )


def _fused_los_jones(
    compact: _ConsumerRows,
    *,
    sources: EndpointBatch,
    sinks: EndpointBatch,
    reference_frequency_hz: float | torch.Tensor,
) -> JonesTransport:
    """Primal-only fused operator: one native launch for the whole batch."""

    assert sources.polarization_basis is not None
    assert sinks.polarization_basis is not None
    jones = consumer_los_jones(
        pair_index=compact.pair_index,
        source_positions=sources.positions_m,
        sink_positions=sinks.positions_m,
        source_reference_basis=sources.polarization_basis,
        sink_reference_basis=sinks.polarization_basis,
        frequency_hz=float(reference_frequency_hz),
    )
    return JonesTransport(
        matrix=jones.matrix,
        source_basis=jones.source_basis,
        sink_basis=jones.sink_basis,
    )


def _composed_los_jones(
    compact: _ConsumerRows,
    *,
    sources: EndpointBatch,
    sinks: EndpointBatch,
    frequency: float | torch.Tensor,
    frequency_value: float,
) -> JonesTransport:
    """Differentiable operator composed from the native free-space owner.

 Discovery restricts this response to line-of-sight rows, so every row has
 the same single leg and one excitation pair covers the whole batch. The
 fused primal operator above evaluates the identical native expressions in
 one launch, and both routes are held to bit-identical agreement by test.
 """

    from witwin.channel.kernels import fields as field_kernels

    assert sources.polarization_basis is not None
    assert sinks.polarization_basis is not None
    assert sources.powers_w is not None
    topology = compact.evaluated.topology
    source_rows = topology.tx_id.to(dtype=torch.int64)
    sink_rows = topology.rx_id.to(dtype=torch.int64)
    source = select_rows(sources.positions_m, source_rows)
    target = select_rows(sinks.positions_m, sink_rows)
    power = select_rows(sources.powers_w, source_rows)
    source_basis = transverse_basis(
        select_rows(sources.polarization_basis, source_rows),
        source,
        target,
        frequency_hz=frequency_value,
    )
    matrix, sink_basis, _ = compose_jones(
        lambda polarization: field_kernels.field_free_space_ad(
            source,
            target,
            power,
            polarization,
            polarization,
            frequency=frequency,
            frequency_value=frequency_value,
        ),
        source_basis=source_basis,
        sink_reference_basis=select_rows(sinks.polarization_basis, sink_rows),
        arrival_origin=source,
        arrival_target=target,
        frequency_hz=frequency_value,
    )
    return JonesTransport(
        matrix=matrix, source_basis=source_basis, sink_basis=sink_basis
    )


def _transport(
    response: str,
    compact: _ConsumerRows,
    *,
    sources: EndpointBatch,
    sinks: EndpointBatch,
    reference_frequency_hz: float | torch.Tensor,
    ad_mode: str,
    frequency_value: float,
) -> ScalarTransport | Complex3Transport | JonesTransport:
    fields = compact.evaluated.fields
    geometry = compact.evaluated.geometry
    if response == "scalar_transport":
        return ScalarTransport(coefficient=fields.path_field)
    if response == "complex3_transport":
        assert sources.powers_w is not None
        tx_power = select_rows(
            sources.powers_w,
            compact.evaluated.topology.tx_id.to(dtype=torch.int64),
        )
        return Complex3Transport(
            field=excited_field(fields.field_xyz, tx_power, ad_mode=ad_mode),
            direction=geometry.field_direction,
        )
    if response == "polarimetric_transport":
        if ad_mode == "none":
            return _fused_los_jones(
                compact,
                sources=sources,
                sinks=sinks,
                reference_frequency_hz=reference_frequency_hz,
            )
        return _composed_los_jones(
            compact,
            sources=sources,
            sinks=sinks,
            frequency=reference_frequency_hz,
            frequency_value=frequency_value,
        )
    raise AssertionError("response was not preflighted")


def _path_batch(
    compact: _ConsumerRows,
    *,
    pair_count: int,
    response: str,
    sources: EndpointBatch,
    sinks: EndpointBatch,
    reference_frequency_hz: float | torch.Tensor,
    ad_mode: str,
    frequency_value: float,
    provenance: WorldProvenance,
) -> PropagationPathBatch:
    evaluated = compact.evaluated
    source = evaluated.topology
    continuous = evaluated.geometry
    path_count = compact.path_count
    topology = PropagationTopology(
        source_index=source.tx_id,
        sink_index=source.rx_id,
        source_id=compact.source_id,
        sink_id=compact.sink_id,
        depth=source.depth,
        component_id=source.component_id,
        primitive_id=source.primitive_id,
        edge_id=source.edge_id,
        material_id=source.material_id,
        primitive_sequence=source.primitive_sequence,
        material_sequence=source.material_sequence,
        interaction_type=source.interaction_type,
        provenance=provenance,
    )
    geometry = PropagationGeometry(
        path_length_m=continuous.path_length_m,
        delay_s=continuous.delay_s,
        field_direction=continuous.field_direction,
        interaction_positions_m=continuous.interaction_positions,
        interaction_normals=continuous.interaction_normals,
    )
    transport = _transport(
        response,
        compact,
        sources=sources,
        sinks=sinks,
        reference_frequency_hz=reference_frequency_hz,
        ad_mode=ad_mode,
        frequency_value=frequency_value,
    )
    return PropagationPathBatch(
        pair_count=pair_count,
        path_count=path_count,
        pair_index=compact.pair_index,
        pair_offsets=compact.pair_offsets,
        topology=topology,
        geometry=geometry,
        transport=transport,
    )


def _diagnostics(
    sidecars: object,
    compact: _ConsumerRows,
    ad_mode: str,
) -> PropagationDiagnostics:
    execution = sidecars.execution
    return PropagationDiagnostics(
        discovery_launch_count=int(execution.launch_count),
        candidate_count=int(execution.candidate_count),
        visibility_rejection_count=int(execution.visibility_rejection_count),
        compact_count_d2h_copies=int(compact.count_d2h_copies),
        compact_count_d2h_bytes=int(compact.count_d2h_bytes),
        compact_sync_count=int(compact.count_synchronizations),
        validation_d2h_copies=0,
        validation_d2h_bytes=0,
        validation_sync_count=0,
        ad_companion_launches=int(execution.ad_companion_launches),
        ad_tape_bytes=tape_bytes(execution.ad_tape_bytes, ad_mode),
    )


def evaluate(
    compiled_scene: CompiledScene, request: PropagationRequest
) -> PropagationEvaluation:
    """Discover and evaluate one all-or-nothing compact propagation batch."""

    from witwin.channel.propagation.enumerated import (
        evaluate_enumerated_paths,
        sanitize_enumerated_capacity_transaction,
    )

    compiled, request = _preflight_evaluate(compiled_scene, request)
    scene, endpoint_tensors = _solver_scene(compiled, request.sources, request.sinks)
    config = _ConsumerConfig(
        max_depth=request.max_depth,
        components=request.components,
        max_paths=request.max_paths,
        ad_mode=request.ad_mode,
    )
    evaluated, sidecars = evaluate_enumerated_paths(
        scene,
        config,
        endpoint_tensors=endpoint_tensors,
        defer_capacity_terminal=True,
    )
    evaluated, sidecars = sanitize_enumerated_capacity_transaction(evaluated, sidecars)
    if sidecars.capacity_transaction is not None:
        sidecars.capacity_transaction.terminal_check()
    compact = _compact(evaluated, getattr(sidecars, "compact_metadata", None))
    paths = _path_batch(
        compact,
        pair_count=request.sources.count * request.sinks.count,
        response=request.response,
        sources=request.sources,
        sinks=request.sinks,
        reference_frequency_hz=request.reference_frequency_hz,
        ad_mode=request.ad_mode,
        frequency_value=compiled.materials.frequency_hz,
        provenance=WorldProvenance.of(compiled),
    )
    return PropagationEvaluation(
        paths=paths,
        convention=_CONVENTION,
        capabilities=_CAPABILITIES,
        diagnostics=_diagnostics(sidecars, compact, request.ad_mode),
    )


def _require_current_world(
    compiled: CompiledScene, request: FixedTopologyRequest
) -> None:
    """Refuse a frozen replay against a world that moved (world-version validation).

 Four host integer comparisons against the version domains the compiled
 scene recorded. No device work, no allocation, no synchronization, and no
 compact output budget impact. A frozen topology with no provenance is hand-built
 and has no world to be stale against, so it proceeds.
 """

    provenance = request.frozen_topology.provenance
    if provenance is None:
        return
    moved = provenance.moved_domain(
        WorldProvenance.of(compiled),
        allow_geometry=request.world_motion == "fixed_winner_replay",
    )
    if moved is None:
        return
    remedy = (
        "declare world_motion='fixed_winner_replay' to hold the discrete "
        "winner set fixed while the geometry moves, or rediscover"
        if moved == "geometry_version"
        else "the frozen row labels no longer name the same world; rediscover"
    )
    raise ValueError(
        f"frozen topology is stale: {moved} changed between discovery and "
        f"this reevaluation; {remedy} with evaluate() and "
        f"prepare_fixed_topology()"
    )


def rediscovery_required(
    compiled_scene: CompiledScene,
    topology: PropagationTopology | PreparedFixedTopology,
    *,
    revalidate_source: bool = False,
) -> str | None:
    """Name the version domain that moved under a frozen topology, or ``None``.

 This is the explicit rediscovery signal: poll it per frame and call:func:`evaluate` plus:func:`witwin.channel.propagation.consumer.prepare_fixed_topology` again
 when it fires. The default comparison is four host integers against the
 versions ``compiled_scene`` recorded, so it costs no device work, no
 allocation, and no synchronization. ``"geometry_version"`` is reported
 like any other domain; a caller replaying under
 ``world_motion="fixed_winner_replay"`` deliberately ignores that one.

 ``revalidate_source=True`` additionally recomputes the four domains from
 the live ``witwin.core`` world the compiled scene was built from, which
 catches a scene mutated in place after compilation - the one staleness
 class the recorded versions cannot see, because a compiled scene and the
 rows discovered on it always agree with each other. That recomputation
 walks the world and hashes it, so it is O(scene) host work and belongs on
 a motion-event cadence, never in a per-frame replay loop.

 Returns ``None`` when nothing moved.
 """

    from witwin.channel.scene.compiler import CompiledScene

    if not isinstance(compiled_scene, CompiledScene):
        raise TypeError("rediscovery_required requires a CompiledScene")
    if not isinstance(topology, PropagationTopology | PreparedFixedTopology):
        raise TypeError(
            "topology must be a PropagationTopology or a PreparedFixedTopology"
        )
    current = WorldProvenance.of(compiled_scene)
    provenance = topology.provenance
    if provenance is not None:
        moved = provenance.moved_domain(current)
        if moved is not None:
            return moved
    if not revalidate_source:
        return None
    return current.moved_domain(WorldProvenance.of(compiled_scene.source))


def _require_wideband_dispersive_materials(compiled: CompiledScene) -> None:
    """W1: refuse an offset grid on a scene with a frozen dispersive record.

 ``scene.compile`` evaluates a ``witwin.core`` ``DispersionSpec`` once, at
 the primal frequency, and stores the result as a plain ``eps_r`` plus an
 equivalent ``sigma_e``. Every other frequency dependence in the material
 model - the conductivity loss tangent, the layer electrical thicknesses, the
 whole Airy recursion - is re-derived natively from the frequency the launch
 receives, so it is already exact at an offset. Dispersion is the one term
 that is not, and re-evaluating it here would need either a recompile or a
 second host-side dispersion evaluator, both of which the Channel guardrails
 forbid inside the consumer.

 This fires at EVERY AD mode. The existing gate refuses a frequency GRADIENT
 against a frozen record; the primal at an offset has the identical defect
 and, until an offset grid existed, was unreachable only because the
 compile-frequency mismatch rule forced a recompile.
 """

    dependent = tuple(compiled.materials.frequency_dependent)
    if not dependent:
        return
    raise NotImplementedError(
        "frequency_offsets_hz is not supported on a scene with "
        f"frequency-dependent materials {sorted(dependent)}: their records are "
        "frozen at the primal frequency at compile time, so an offset column "
        "would publish the reference-frequency material law under a different "
        "frequency label. capabilities().wideband_dispersive_materials is "
        "False. Compile one scene per frequency instead, which is the caller's "
        "explicit choice rather than an implicit recompile"
    )


def _require_resolvable_offsets(
    offsets: tuple[float, ...], reference_frequency_hz: float
) -> None:
    """W2: refuse an offset grid the native launch grid cannot resolve.

 Every native field bridge casts the frequency to float32 at the launch, so
 two absolute frequencies inside one float32 ULP are the same launch and
 return bit-identical columns. Publishing them as distinct frequencies would
 be a declaration nobody enforces.
 """

    resolution = native_frequency_resolution_hz(reference_frequency_hz)
    for offset in offsets:
        if offset != 0.0 and abs(offset) < resolution:
            raise ValueError(
                f"frequency_offsets_hz entry {offset!r} Hz is below the native "
                f"frequency resolution {resolution!r} Hz at "
                f"{reference_frequency_hz!r} Hz: the native launch grid is "
                "float32, so this offset evaluates at the reference frequency "
                "and would publish a duplicate column under a different label"
            )
    ordered = sorted(offsets)
    for lower, upper in zip(ordered, ordered[1:]):
        if upper - lower < resolution:
            raise ValueError(
                f"frequency_offsets_hz entries {lower!r} Hz and {upper!r} Hz "
                f"are closer than the native frequency resolution "
                f"{resolution!r} Hz at {reference_frequency_hz!r} Hz: the "
                "native launch grid is float32, so they evaluate at the same "
                "absolute frequency"
            )


def _require_wideband_smooth_scene(compiled: CompiledScene) -> None:
    """W4: refuse an offset grid on a rough or phase-screen scene.

 The Kirchhoff roughness tables and the phase-screen realization resources
 are resident resources keyed on a material cache token that hashes the
 compile frequency (RayD scattering ownership). Reusing a table built at ``f_ref`` at
 ``f_ref + df`` freezes the scattering response the same way a
 ``DispersionSpec`` record freezes the material law, so it is refused for the
 same reason and until a decision that covers resident-table lifetime across
 a band.

 One device read: a single reduced bitmask over the two roughness columns,
 in the preflight, before any native work. It is a refusal guard rather than
 a hot-path transfer and it is not part of the per-call validation budget.
 """

    materials = compiled.materials
    rough = bool(
        (
            (materials.scatter_model_id == 1)
            | (materials.rough_sigma_h_m > 0.0)
        ).any()
    )
    if rough:
        raise NotImplementedError(
            "frequency_offsets_hz is not supported on a scene with rough "
            "materials: the Kirchhoff tables are resident resources keyed on a "
            "material cache token that hashes the compile frequency, so a table "
            "built at the reference frequency is frozen exactly as a dispersive "
            "record is. capabilities().wideband_rough_materials is False"
        )
    screens = getattr(compiled.assignments, "structure_phase_screens", {})
    if screens:
        raise NotImplementedError(
            "frequency_offsets_hz is not supported on a scene carrying phase "
            "screens: their realization resources are keyed on the same "
            "frequency-hashed material cache token as the roughness tables. "
            "capabilities().wideband_rough_materials is False"
        )


def _preflight_wideband(
    compiled: CompiledScene, request: FixedTopologyRequest
) -> None:
    """Scene-dependent wideband refusals, each independent (wideband evaluation).

 Every check is reachable on its own: a dispersive smooth scene with a
 resolvable grid trips only W1, a non-dispersive smooth scene with an
 unresolvable grid trips only W2, and a rough non-dispersive scene with a
 resolvable grid trips only W4. Folding any of them into another would make
 one of the three limits undiscoverable.
 """

    offsets = request.frequency_offsets_hz
    if offsets is None:
        return
    _require_wideband_dispersive_materials(compiled)
    _require_resolvable_offsets(offsets, compiled.materials.frequency_hz)
    _require_wideband_smooth_scene(compiled)


def _preflight_reevaluate(
    compiled: object, request: object
) -> tuple[CompiledScene, FixedTopologyRequest]:
    from witwin.channel.scene.compiler import CompiledScene

    if not isinstance(compiled, CompiledScene):
        raise TypeError("reevaluate requires a CompiledScene")
    if not isinstance(request, FixedTopologyRequest):
        raise TypeError("request must be a FixedTopologyRequest")
    _require_current_world(compiled, request)
    if request.frozen_topology.device != request.sources.device:
        raise ValueError("fixed topology and endpoint batches must share a device")
    compiled.require_reference_frequency(request.reference_frequency_hz)
    if not _CAPABILITIES.supports_fixed_topology:
        raise NotImplementedError(
            "fixed-topology reevaluation is unavailable in this build"
        )
    if (
        isinstance(request.topology, PropagationTopology)
        and request.topology.primitive_sequence.shape[1] != 0
    ):
        raise NotImplementedError(
            "fixed LoS reevaluation requires zero-width interaction sequences; "
            "call prepare_fixed_topology first to reevaluate a topology that "
            "carries interactions"
        )
    if request.response == "polarimetric_transport":
        if (
            request.sources.polarization_basis is None
            or request.sinks.polarization_basis is None
        ):
            raise ValueError(
                "polarimetric_transport requires source and sink "
                "polarization_basis tensors"
            )
        if isinstance(request.topology, PropagationTopology):
            raise NotImplementedError(
                "fixed-topology polarimetric_transport requires a "
                "PreparedFixedTopology; the raw-topology form is the "
                "zero-interaction scalar and complex3 fast path"
            )
    require_primal_only_ad_inputs(compiled, request)
    require_first_order_request(compiled, request)
    _preflight_wideband(compiled, request)
    return compiled, request


def _offset_frequency(
    frequency: float | torch.Tensor, offset: float
) -> float | torch.Tensor:
    """The AD-facing frequency of one wideband column.

 A tensor reference frequency stays a tensor, so the seed a caller placed on
 it reaches every column through the same native companion. ``offset == 0.0``
 is the additive identity in both branches, which is what makes a zero entry
 reproduce the reference column bit for bit.
 """

    return frequency if offset == 0.0 else frequency + offset


def _wideband_columns(
    offsets: tuple[float, ...], column: object
) -> torch.Tensor:
    """Stack per-frequency native outputs into one payload axis.

 Structural packing and nothing else: every value in the stack came out of
 the native owner that computed it at its own absolute frequency, and no
 offset-dependent phase, magnitude, or basis is applied here.
 """

    return torch.stack([column(offset) for offset in offsets], dim=1)


def _column_payload(response: str, outputs: object) -> torch.Tensor:
    """The one tensor a wideband column contributes to the payload axis.

 A column recomputes the geometry natively and discards it: the published
 geometry is the reference column's, because path length, delay, direction,
 and the interaction table are facts about where the path goes and do not
 depend on the frequency it is evaluated at.
 """

    if response == "scalar_transport":
        return outputs.path_field
    assert outputs.path_field_vector is not None
    return outputs.path_field_vector


def _fixed_transport(
    response: str,
    outputs: object,
    *,
    offsets: tuple[float, ...] | None = None,
    payload: torch.Tensor | None = None,
) -> ScalarTransport | Complex3Transport | JonesTransport:
    if response == "scalar_transport":
        return ScalarTransport(
            coefficient=outputs.path_field,
            coefficient_offsets=payload,
            frequency_offsets_hz=offsets,
        )
    if response == "complex3_transport":
        assert outputs.path_field_vector is not None
        return Complex3Transport(
            field=outputs.path_field_vector,
            direction=outputs.direction,
            field_offsets=payload,
            frequency_offsets_hz=offsets,
        )
    assert outputs.matrix is not None
    return JonesTransport(
        matrix=outputs.matrix,
        source_basis=outputs.source_basis,
        sink_basis=outputs.sink_basis,
    )


def _slot_pair_count(request: FixedTopologyRequest) -> int:
    """Pairs published by one call, under the declared slot layout.

 One slot is the full source/sink outer product. More than one is block
 diagonal, so the count is linear in the slot count rather than quadratic.
 """

    slot_count = request.slot_count
    return (
        slot_count
        * (request.sources.count // slot_count)
        * (request.sinks.count // slot_count)
    )


def _reevaluate_prepared(
    compiled: CompiledScene, request: FixedTopologyRequest
) -> FixedTopologyEvaluation:
    """Replay a prepared frozen topology bucket by bucket."""

    prepared = request.topology
    assert isinstance(prepared, PreparedFixedTopology)
    validity = _CAPABILITIES.fixed_topology_row_validity_components
    if any(bucket.component == "reflection" for bucket in prepared.buckets):
        require_smooth_reflection_scene(compiled)
    # The row gather owns the one validation copy and the one synchronization,
    # and it runs ONCE here, above the frequency-column loop below. That is what
    # holds the compact output budget at 1/1 however many columns a wideband request
    # declares.
    rows = prepared_row_gather(
        prepared.topology,
        request.sources,
        request.sinks,
        slot_count=request.slot_count,
    )
    bases = (
        (
            select_rows(request.sources.polarization_basis, rows.source_row_index),
            select_rows(request.sinks.polarization_basis, rows.sink_row_index),
        )
        if request.response == "polarimetric_transport"
        else (None, None)
    )
    # forward-mode liveness: liveness is decided once, here, from the inputs every column
    # shares, and the same record reaches every column. first-order differentiation adds the
    # arrival-direction half of the same decision, taken from the host-known
    # component set of the frozen batch: a batch that carries a component whose
    # direction seam RayD owns publishes a fully detached field_direction for
    # the whole result rather than a partly live one.
    frozen_components = frozenset(
        bucket.component for bucket in prepared.buckets
    )
    geometry_live = GeometryLiveness.of(
        rows.source,
        rows.target,
        scene_vertex_table(compiled)
        if any(bucket.depth > 0 for bucket in prepared.buckets)
        else None,
        direction_components=frozen_components.issubset(
            _CAPABILITIES.direction_differentiable_components
        ),
    )
    ledger = ad_ledger(request.ad_mode)

    def column(offset: float):
        return evaluate_prepared(
            compiled,
            prepared,
            rows,
            response=request.response,
            ad_mode=request.ad_mode,
            frequency=_offset_frequency(request.reference_frequency_hz, offset),
            frequency_value=compiled.materials.frequency_hz + offset,
            source_reference_basis=bases[0],
            sink_reference_basis=bases[1],
            publish_row_validity=any(
                bucket.component in validity for bucket in prepared.buckets
            ),
            geometry_live=geometry_live,
            ledger=ledger,
        )

    outputs = column(0.0)
    offsets = request.frequency_offsets_hz
    payload = (
        None
        if offsets is None
        else _wideband_columns(
            offsets, lambda offset: _column_payload(request.response, column(offset))
        )
    )
    paths = PropagationPathBatch(
        pair_count=_slot_pair_count(request),
        path_count=rows.row_count,
        pair_index=rows.pair_index,
        pair_offsets=rows.pair_offsets,
        topology=prepared.topology,
        geometry=PropagationGeometry(
            path_length_m=outputs.path_length_m,
            delay_s=outputs.delay_s,
            field_direction=outputs.direction,
            interaction_positions_m=outputs.interaction_positions,
            interaction_normals=outputs.interaction_normals,
        ),
        transport=_fixed_transport(
            request.response, outputs, offsets=offsets, payload=payload
        ),
    )
    return FixedTopologyEvaluation(
        paths=paths,
        convention=_CONVENTION,
        capabilities=_CAPABILITIES,
        diagnostics=PropagationDiagnostics(
            discovery_launch_count=0,
            candidate_count=0,
            visibility_rejection_count=0,
            compact_count_d2h_copies=0,
            compact_count_d2h_bytes=0,
            compact_sync_count=0,
            validation_d2h_copies=rows.validation_d2h_copies,
            validation_d2h_bytes=rows.validation_d2h_bytes,
            validation_sync_count=rows.validation_synchronizations,
            frequency_column_count=1 if offsets is None else len(offsets),
            ad_companion_launches=0 if ledger is None else ledger.launches,
            ad_tape_bytes=(
                0
                if ledger is None
                else tape_bytes(ledger.tape_bytes, request.ad_mode)
            ),
        ),
        row_valid=outputs.row_valid,
    )


def reevaluate(
    compiled_scene: CompiledScene, request: FixedTopologyRequest
) -> FixedTopologyEvaluation:
    """Reevaluate frozen rows without topology discovery or compaction."""

    from witwin.channel.kernels import fields as field_kernels

    compiled, request = _preflight_reevaluate(compiled_scene, request)
    if isinstance(request.topology, PreparedFixedTopology):
        return _reevaluate_prepared(compiled, request)
    # One gather for the whole call, above the frequency-column loop: it owns
    # the single validation copy and the single synchronization.
    rows = fixed_los_gather(request.topology, request.sources, request.sinks)
    frequency = request.reference_frequency_hz
    frequency_value = compiled.materials.frequency_hz
    tx_power = rows.tx_power.detach()
    tx_polarization = rows.tx_polarization.detach()
    rx_polarization = rows.rx_polarization.detach()
    # forward-mode liveness: one liveness decision, taken here from the gathered rows, and
    # re-asserted for every column against the inputs that column launches on.
    geometry_live = fixed_los_geometry_live(rows)
    # first-order differentiation: this route carries line-of-sight rows only, so the direction
    # seam is Channel-owned for the whole result and its liveness is exactly
    # the geometry decision above.
    direction_live = geometry_live and frozenset({"los"}).issubset(
        _CAPABILITIES.direction_differentiable_components
    )
    ledger = ad_ledger(request.ad_mode)

    def column(offset: float) -> dict[str, torch.Tensor]:
        if request.ad_mode == "none":
            return field_kernels.field_free_space(
                rows.source,
                rows.target,
                tx_power,
                tx_polarization,
                rx_polarization,
                frequency_hz=frequency_value + offset,
            )
        require_fixed_los_geometry_live(rows, geometry_live)
        assert ledger is not None
        ledger.add(
            rows.source, rows.target, tx_power, tx_polarization, rx_polarization
        )
        return field_kernels.field_free_space_ad(
            rows.source,
            rows.target,
            tx_power,
            tx_polarization,
            rx_polarization,
            frequency=_offset_frequency(frequency, offset),
            frequency_value=frequency_value + offset,
            direction_live=direction_live,
        )

    field_rows = column(0.0)
    row_count = rows.row_count
    empty_interactions = rows.source.new_empty((row_count, 0, 3))
    geometry = PropagationGeometry(
        path_length_m=field_rows["path_length_m"],
        delay_s=field_rows["delay_s"],
        field_direction=field_rows["direction"],
        interaction_positions_m=empty_interactions,
        interaction_normals=empty_interactions,
    )
    offsets = request.frequency_offsets_hz

    def payload_of(values: dict[str, torch.Tensor]) -> torch.Tensor:
        if request.response == "scalar_transport":
            return values["path_field"]
        if ledger is not None:
            ledger.add(values["field_vector"], tx_power)
        return excited_field(
            values["field_vector"], tx_power, ad_mode=request.ad_mode
        )

    payload = (
        None
        if offsets is None
        else _wideband_columns(offsets, lambda offset: payload_of(column(offset)))
    )
    transport = (
        ScalarTransport(
            coefficient=field_rows["path_field"],
            coefficient_offsets=payload,
            frequency_offsets_hz=offsets,
        )
        if request.response == "scalar_transport"
        else Complex3Transport(
            field=payload_of(field_rows),
            direction=field_rows["direction"],
            field_offsets=payload,
            frequency_offsets_hz=offsets,
        )
    )
    paths = PropagationPathBatch(
        pair_count=request.sources.count * request.sinks.count,
        path_count=row_count,
        pair_index=rows.pair_index,
        pair_offsets=rows.pair_offsets,
        topology=request.topology,
        geometry=geometry,
        transport=transport,
    )
    return FixedTopologyEvaluation(
        paths=paths,
        convention=_CONVENTION,
        capabilities=_CAPABILITIES,
        diagnostics=PropagationDiagnostics(
            discovery_launch_count=0,
            candidate_count=0,
            visibility_rejection_count=0,
            compact_count_d2h_copies=0,
            compact_count_d2h_bytes=0,
            compact_sync_count=0,
            validation_d2h_copies=rows.validation_d2h_copies,
            validation_d2h_bytes=rows.validation_d2h_bytes,
            validation_sync_count=rows.validation_synchronizations,
            frequency_column_count=1 if offsets is None else len(offsets),
            ad_companion_launches=0 if ledger is None else ledger.launches,
            ad_tape_bytes=(
                0
                if ledger is None
                else tape_bytes(ledger.tape_bytes, request.ad_mode)
            ),
        ),
    )


# --- Time-varying channel impulse response --------------------------------
#
# Time-varying channel impulse response over one slot-batched replay.
#
# A ``(source, sink)`` pair's ``delay_s`` and transport already ARE its impulse
# response; nothing physical is missing from the consumer contract. What was
# missing was a time axis: a caller who wanted the response at ``T`` instants had
# to run ``T`` reevaluations, which is ``T`` validation copies and ``T``
# synchronizations for a capability whose whole point is to have exactly one.
#
# This section is that axis and nothing else. It replicates the frozen rows over
# ``T`` block-diagonal slots, runs ONE reevaluation, and publishes ``[T, K]``
# views over the storage that replay produced. It owns no physics, adds no
# compaction, allocates no result, and introduces no native symbol. It also does
# not compile scenes: one:class:`~witwin.channel.scene.compiler.CompiledScene`
# covers one structure-geometry epoch, and a world whose structures move is
# ``T`` epochs, which is a motion-event cadence rather than an inner loop.


def _slot_view(values: torch.Tensor, slot_count: int) -> torch.Tensor:
    """Split a slot-major row axis into ``[slot_count, K, ...]``.

 ``view`` rather than ``reshape`` on purpose: the row axis is outermost and
 the replay publishes contiguous storage, so a layout that needed a copy
 here would be a silent regression rather than something to paper over.
 """

    return values.view(slot_count, -1, *values.shape[1:])


@dataclass(frozen=True, slots=True)
class TimeVaryingRequest:
    """Replay one prepared topology across slot-major endpoint batches. Time values label slots without driving computation; native forward AD supplies motion derivatives. Each compiled scene is one structure-geometry epoch, so callers keep labels consistent with that scene."""

    sources: EndpointBatch
    sinks: EndpointBatch
    reference_frequency_hz: float | torch.Tensor
    topology: PreparedFixedTopology
    times_s: torch.Tensor
    response: PropagationResponse
    ad_mode: PropagationAdMode
    world_motion: PropagationWorldMotion = "frozen_world"

    def __post_init__(self) -> None:
        if not isinstance(self.topology, PreparedFixedTopology):
            raise TypeError(
                "topology must be a PreparedFixedTopology; call "
                "prepare_fixed_topology once per frozen topology and reuse it"
            )
        times = self.times_s
        if not isinstance(times, torch.Tensor):
            raise TypeError("times_s must be a torch.Tensor")
        if times.dtype != torch.float64:
            raise TypeError(f"times_s must use torch.float64, got {times.dtype}")
        if times.ndim != 1 or int(times.shape[0]) == 0:
            raise ValueError("times_s must be a non-empty 1-D tensor")

    @property
    def slot_count(self) -> int:
        return int(self.times_s.shape[0])


@dataclass(frozen=True, slots=True, eq=False)
class TimeVaryingTransport:
    """Slot-shaped views over the transport one replay published.

 Exactly the tensors of the requested response are present and the rest are
 ``None``, so a reader cannot pick up a field the response never produced.
 """

    response: PropagationResponse
    coefficient: torch.Tensor | None = None
    field: torch.Tensor | None = None
    direction: torch.Tensor | None = None
    matrix: torch.Tensor | None = None
    source_basis: torch.Tensor | None = None
    sink_basis: torch.Tensor | None = None

    @classmethod
    def from_transport(
        cls, transport: PropagationTransport, slot_count: int
    ) -> TimeVaryingTransport:
        if isinstance(transport, ScalarTransport):
            return cls(
                response="scalar_transport",
                coefficient=_slot_view(transport.coefficient, slot_count),
            )
        if isinstance(transport, Complex3Transport):
            return cls(
                response="complex3_transport",
                field=_slot_view(transport.field, slot_count),
                direction=_slot_view(transport.direction, slot_count),
            )
        return cls(
            response="polarimetric_transport",
            matrix=_slot_view(transport.matrix, slot_count),
            source_basis=_slot_view(transport.source_basis, slot_count),
            sink_basis=_slot_view(transport.sink_basis, slot_count),
        )


@dataclass(frozen=True, slots=True, eq=False)
class TimeVaryingEvaluation:
    """One frozen topology evaluated at ``slot_count`` world instants.

 Every published tensor is slot-major: index ``[t]`` is the complete frozen
 row set at ``times_s[t]``, in frozen row order, so ``delay_s[t]`` and the
 transport at ``[t]`` are the impulse response of that instant and
 ``pair_offsets`` segments it by ``(source, sink)`` pair.

 ``pair_offsets`` and ``pair_count`` are PER SLOT. They are frozen: the same
 rows are replayed at every instant, so every slot carries the same
 segmentation, including the empty segments of pairs that publish no row.

 ``row_valid`` keeps its fixed-topology replay meaning per slot and remains the sole
 authority: a row that stops existing at instant ``t`` publishes exact zeros
 at ``[t]`` and stays alive at the other instants. Replay is still
 subtractive (world-version validation) - a path that comes into existence part way through
 the block is not discovered here and is silently absent from every slot.
 """

    slot_count: int
    row_count: int
    times_s: torch.Tensor
    delay_s: torch.Tensor
    path_length_m: torch.Tensor
    transport: TimeVaryingTransport
    pair_count: int
    pair_offsets: torch.Tensor
    convention: PropagationConvention
    capabilities: PropagationCapabilities
    diagnostics: PropagationDiagnostics
    row_valid: torch.Tensor | None = None

    @classmethod
    def from_evaluation(
        cls,
        evaluation: FixedTopologyEvaluation,
        times_s: torch.Tensor,
        slot_count: int,
    ) -> TimeVaryingEvaluation:
        paths = evaluation.paths
        pair_count = paths.pair_count // slot_count
        return cls(
            slot_count=slot_count,
            row_count=paths.path_count // slot_count,
            times_s=times_s,
            delay_s=_slot_view(paths.geometry.delay_s, slot_count),
            path_length_m=_slot_view(paths.geometry.path_length_m, slot_count),
            transport=TimeVaryingTransport.from_transport(
                paths.transport, slot_count
            ),
            pair_count=pair_count,
            # Every slot repeats the same segmentation, so the leading slot's
            # prefix of the CSR vector IS the per-slot segmentation. Narrowing
            # is a view; rebuilding it would be a second, redundant owner.
            pair_offsets=paths.pair_offsets[: pair_count + 1],
            convention=evaluation.convention,
            capabilities=evaluation.capabilities,
            diagnostics=evaluation.diagnostics,
            row_valid=(
                None
                if evaluation.row_valid is None
                else _slot_view(evaluation.row_valid, slot_count)
            ),
        )


def evaluate_time_varying(
    compiled_scene: CompiledScene, request: TimeVaryingRequest
) -> TimeVaryingEvaluation:
    """Replay one frozen topology across a whole block of world instants.

 One launch per ``(component, depth)`` bucket, one validation copy, and one
 synchronization for the entire block, whatever ``len(times_s)`` is. That is
 the point: a Python loop over instants keeps every individual call inside
 the compact output budget while multiplying the budget of the frame by the number
 of instants.

 The instants must share one compiled scene, which is what makes them one
 frame, pulse train, or symbol block. Structure motion changes the compiled
 scene, so it is a new call with a new scene, not another slot.
 """

    if not isinstance(request, TimeVaryingRequest):
        raise TypeError("request must be a TimeVaryingRequest")
    slot_count = request.slot_count
    _require_slot_divisible("sources", request.sources.count, slot_count)
    _require_slot_divisible("sinks", request.sinks.count, slot_count)
    replicated = replicate_over_slots(
        request.topology,
        slot_count,
        source_count=request.sources.count // slot_count,
        sink_count=request.sinks.count // slot_count,
    )
    evaluation = reevaluate(
        compiled_scene,
        FixedTopologyRequest(
            sources=request.sources,
            sinks=request.sinks,
            reference_frequency_hz=request.reference_frequency_hz,
            topology=replicated,
            response=request.response,
            ad_mode=request.ad_mode,
            world_motion=request.world_motion,
            slot_count=slot_count,
        ),
    )
    return TimeVaryingEvaluation.from_evaluation(
        evaluation, request.times_s, slot_count
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
    "TimeVaryingEvaluation",
    "TimeVaryingRequest",
    "TimeVaryingTransport",
    "WorldProvenance",
    "capabilities",
    "evaluate",
    "evaluate_time_varying",
    "native_frequency_resolution_hz",
    "prepare_fixed_topology",
    "rediscovery_required",
    "reevaluate",
    "replicate_over_slots",
]