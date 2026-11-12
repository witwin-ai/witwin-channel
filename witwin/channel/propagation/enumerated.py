# Copyright Xingyu Chen.
# Enumerated propagation: the shared engine, its typed config protocols, and.

"""Enumerated propagation: the shared engine, its typed config protocols, and."""

from __future__ import annotations

from dataclasses import dataclass, replace
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, ClassVar, Protocol, cast

import torch
from witwin.channel.components import (
    AD_MODES,
    DEFAULT_COMPONENTS,
    DISCOVERY_MODES,
    validate_chain_depth_cap,
    validate_config_fields,
    validate_coupled_candidate_limit,
    validate_coupled_gate,
    validate_isb_boundary_taper,
    validate_max_depth,
    validate_max_paths,
    validate_scatter_chain,
    validate_scattering_quadrature,
    validate_vocabulary,
    validated_components,
)
from witwin.channel.scene.endpoints import require_compiled

from witwin.channel.materials import (
    _require_frequency_ad_constant_materials,
)
from witwin.channel.interactions.coupled import (
    chain_depth_limit,
    coupled_reflection_diffraction_topology,
)
from witwin.channel.interactions.diffraction import (
    _diffraction_topology_order1,
)
from witwin.channel.interactions.los import _los_topology
from witwin.channel.interactions.reflection import (
    ReflectionDiscoveryRecord,
    _reflection_topology,
)
from witwin.channel.interactions.transmission import (
    _transmission_topology,
)
from witwin.channel.propagation.fields import evaluate_path_fields
from witwin.channel.propagation.geometry import (
    receiver_positions_and_layout,
    transmitter_tensors,
)
from witwin.channel.propagation.rows import (
    EvaluatedPaths,
    PathFields,
    PathGeometry,
    PathRowGroups,
    PathTopology,
)
from witwin.channel.propagation.topology import (
    _pad_topology_sequences,
    concatenate_path_blocks,
)
from witwin.channel.propagation.topology import (
    EvaluatedPathSidecars,
    evaluated_paths_from_block,
    evaluated_paths_from_result,
)
from witwin.channel.kernels import topology as topology_kernels
from witwin.channel.runtime import (
    CapacityExecutionCounts,
    CapacityFailureState,
    SolveCapacityTransaction,
    _ad_first_order_only,
    _ad_native_tangent_or_none,
    _ad_native_tensor,
    create_solve_capacity_transaction,
    disable_functorch,
    require_capacity_failure_state,
    required_symbol as _required_native_op,
)
from witwin.channel.scene.compiler import (
    _frequency_scalar,
    transmitter_polarizations_as_stored,
)

if TYPE_CHECKING:
    from witwin.channel.scene.compiler import CompiledScene
    from witwin.channel.scene.endpoints import SolverScene as Scene


__all__: list[str] = []


# --- Typed config protocols ------------------------------------------------
#
# Two structural views of a solver config are read here, and they are
# deliberately different objects with different field sets, because a Protocol
# is exactly its field set. ``TopologyConfig`` is the larger view the enumerated
# scattering stages read (``interactions.scattering`` imports it by name);
# ``EnumeratedPathConfig`` is the four-field view this engine itself reads.
# Merging them would silently widen one of the two contracts.


class TopologyConfig(Protocol):
    @property
    def components(self) -> frozenset[str] | set[str] | tuple[str, ...] | list[str]: ...
    @property
    def max_depth(self) -> int: ...
    @property
    def scattering_samples_per_m2(self) -> float: ...
    @property
    def scattering_power_threshold(self) -> float: ...
    @property
    def scattering_max_paths_per_pair(self) -> int: ...
    # coherent scattering enumerated scatter-chain path class. DEFAULT-OFF: 0 disables
    # chain discovery entirely (the pipeline stays byte-identical). When >= 1 it
    # is the cap on d1 + d2, the combined reflection depth of the two specular
    # legs around the single diffuse vertex; each leg is independently bounded by
    # the native kMaxAdDepth = 8, so the public cap is 2 * 8 = 16.
    @property
    def scattering_chain_max_depth(self) -> int: ...
    # Chain-sample vertex density (samples / m^2). Documented lower density than
    # the single-bounce scattering sampler (scattering_samples_per_m2) because a
    # chain vertex is joined against two specular legs (coherent scattering).
    @property
    def scattering_chain_samples_per_m2(self) -> float: ...
    # Per-(tx, rx) keep-strongest cap on joined chain rows (coherent scattering budget).
    @property
    def scattering_chain_max_rows(self) -> int: ...


# The structural view of a solver config this engine reads. It is declared here,
# beside its only consumer, so a solver config satisfies it without importing a
# solver.
class EnumeratedPathConfig(Protocol):
    @property
    def max_depth(self) -> int: ...
    @property
    def components(self) -> frozenset[str] | set[str] | tuple[str, ...] | list[str]: ...
    @property
    def max_paths(self) -> int | None: ...
    @property
    def max_paths_scope(self) -> str: ...


# --- Shared enumerated options --------------------------------------------


@dataclass(frozen=True, slots=True)
class EnumeratedOptions:
    """The options the enumerated engine reads; the single owner of every shared field.

    ``path.Config`` and ``radiomap.Coherent`` derive from it and add only the fields
    their terminal stage reads; the consumer builds one from its request.
    """

    max_depth: int = 1
    components: frozenset[str] | set[str] | tuple[str, ...] | list[str] = DEFAULT_COMPONENTS
    max_paths: int | None = None
    max_paths_scope: str = "global"
    ad_mode: str = "none"
    # Interaction chains (up to max_depth reflections with one diffraction anywhere) and
    # double diffraction. Opt-in. coupled_candidate_limit bounds the
    # candidates of one validator launch; it never caps the result.
    coupled_paths: bool = False
    coupled_candidate_limit: int = 1_000_000
    # Rough-surface scattering quadrature: a fixed per-area sample density (per-face
    # cap of 4096 samples), the per-(tx, rx) strongest-rows cap, and an absolute
    # path_gain floor. Dropped power is reported in the scattering metadata.
    scattering_samples_per_m2: float = 8.0
    scattering_max_paths_per_pair: int = 4096
    scattering_power_threshold: float = 0.0
    # Scatter-chain rows (TX --C1(d1)--> v_s --C2(d2)--> RX). Off at 0; when >= 1 it
    # caps d1 + d2, the combined specular depth of the two legs around the single
    # diffuse vertex. Each leg is bounded by the native kMaxAdDepth = 8, so the cap
    # is 16. Chain vertices use their own lower sample density and only the
    # strongest scattering_chain_max_rows rows per (tx, rx) survive. Requires the
    # 'scattering' component.
    scattering_chain_max_depth: int = 0
    scattering_chain_samples_per_m2: float = 2.0
    scattering_chain_max_rows: int = 256
    # ISB boundary taper, an off-by-default visual-continuity heuristic: the hard
    # LoS occlusion gate becomes a C1 taper tau(c / (width * w_F)) and the order-1
    # diffraction odd step spreads over the same window. Off is bit-identical to
    # the hard gate. The width scales the Fresnel penumbra w_F of the grazed edge.
    isb_boundary_taper: bool = False
    isb_boundary_taper_width: float = 0.5
    # Reflection candidate source: "auto" (complete image tree, ray discovery only past its
    # cost budget), "exhaustive" (every plane-group sequence) or "rays" (ray discovery). The
    # result metadata's "complete" says whether the paths found are provably all of them.
    discovery: str = "auto"

    # Name used in validation messages, and the accepted max_paths_scope values.
    _option_owner: ClassVar[str] = "enumerated"
    _max_paths_scopes: ClassVar[frozenset[str]] = frozenset({"global", "per_pair"})

    def __post_init__(self) -> None:
        owner = self._option_owner
        validate_config_fields(self)
        validate_max_depth(self.max_depth)
        validate_isb_boundary_taper(self.isb_boundary_taper_width)
        validate_scattering_quadrature(
            samples_per_m2=self.scattering_samples_per_m2,
            max_paths_per_pair=self.scattering_max_paths_per_pair,
            power_threshold=self.scattering_power_threshold,
        )
        components = validated_components(
            self.components, error_message="components must be a non-empty subset of {valid}"
        )
        validate_scatter_chain(
            max_depth=self.scattering_chain_max_depth,
            samples_per_m2=self.scattering_chain_samples_per_m2,
            max_rows=self.scattering_chain_max_rows,
            components=components,
        )
        validate_chain_depth_cap(owner, self.max_depth, components)
        validate_coupled_gate(
            coupled_paths=self.coupled_paths, max_depth=self.max_depth, components=components,
        )
        validate_coupled_candidate_limit(self.coupled_candidate_limit)
        validate_max_paths(self.max_paths)
        validate_vocabulary(
            f"{owner} max_paths_scope", self.max_paths_scope, self._max_paths_scopes
        )
        validate_vocabulary(f"{owner} ad_mode", self.ad_mode, AD_MODES)
        validate_vocabulary(f"{owner} discovery", self.discovery, DISCOVERY_MODES)
        object.__setattr__(self, "components", components)


# --- Native failure sanitizers for complete evaluated enumerated path rows ---

_TOPOLOGY_INPUT_FIELDS = (
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
_SANITIZE_OUTPUT_FIELDS = (
    "selected_row_index",
    *_TOPOLOGY_INPUT_FIELDS,
    *_CONTINUOUS_FIELDS,
)
_SANITIZE_DISCRETE_OUTPUT_COUNT = 1 + len(_TOPOLOGY_INPUT_FIELDS)


def _evaluated_paths_capacity_pack_backward_native(*args: object) -> object:
    return _required_native_op("evaluated_paths_capacity_pack_backward")(*args)


def _evaluated_paths_capacity_pack_jvp_native(*args: object) -> object:
    return _required_native_op("evaluated_paths_capacity_pack_jvp")(*args)


def _candidate_tensors(paths: EvaluatedPaths) -> tuple[torch.Tensor, ...]:
    topology = paths.topology
    geometry = paths.geometry
    fields = paths.fields
    return (
        *(getattr(topology, name) for name in _TOPOLOGY_INPUT_FIELDS),
        geometry.path_length_m,
        geometry.delay_s,
        geometry.field_direction,
        geometry.interaction_position,
        geometry.interaction_normal,
        geometry.interaction_positions,
        geometry.interaction_normals,
        fields.path_gain,
        fields.path_field,
        fields.field_xyz,
        fields.coefficient,
    )


def _validate_candidate(paths: EvaluatedPaths) -> tuple[torch.Tensor, ...]:
    if not isinstance(paths, EvaluatedPaths):
        raise TypeError("paths must be EvaluatedPaths")
    tensors = _candidate_tensors(paths)
    device = tensors[0].device
    if device.type != "cuda":
        raise ValueError("evaluated path capacity packing requires CUDA tensors")
    for name, tensor in zip(
        (*_TOPOLOGY_INPUT_FIELDS, *_CONTINUOUS_FIELDS), tensors, strict=True
    ):
        if tensor.device != device:
            raise ValueError(f"{name} must share evaluated path device")
        if not tensor.is_contiguous():
            raise ValueError(f"{name} must be contiguous")
    return tensors


class _EnumeratedCapacityFailureSanitizeFunction(torch.autograd.Function):
    """Shape-preserving final failure sanitizer with native AD companions."""

    @staticmethod
    def forward(*inputs: Any) -> Any:
        raw = _required_native_op("enumerated_capacity_failure_sanitize")(
            inputs[22], *inputs[:22]
        )
        if not isinstance(raw, dict) or set(raw) != set(_SANITIZE_OUTPUT_FIELDS):
            raise TypeError("native enumerated failure sanitizer returned bad fields")
        return tuple(raw[name] for name in _SANITIZE_OUTPUT_FIELDS)

    @staticmethod
    def setup_context(ctx: Any, inputs: Any, output: Any) -> None:
        ctx.set_materialize_grads(False)
        ctx.candidate_count = int(inputs[0].shape[0])
        ctx.sequence_width = int(inputs[8].shape[1])
        saved = tuple(
            torch.autograd.forward_ad.unpack_dual(value).primal
            for value in (output[1], output[0])
        )
        ctx.save_for_backward(*saved)
        ctx.save_for_forward(*saved)
        ctx.mark_non_differentiable(*output[:_SANITIZE_DISCRETE_OUTPUT_COUNT])

    @staticmethod
    @_ad_first_order_only
    def backward(ctx: Any, *grad_outputs: Any) -> Any:
        none_grads = (None,) * 23
        continuous_grads = grad_outputs[_SANITIZE_DISCRETE_OUTPUT_COUNT:]
        if all(value is None for value in continuous_grads):
            return none_grads
        if not any(ctx.needs_input_grad[11:22]):
            return none_grads
        valid, selected_row_index = ctx.saved_tensors
        raw = _evaluated_paths_capacity_pack_backward_native(
            valid,
            selected_row_index,
            *continuous_grads,
            ctx.candidate_count,
            ctx.sequence_width,
        )
        if not isinstance(raw, dict) or set(raw) != set(_CONTINUOUS_FIELDS):
            raise TypeError(
                "native enumerated failure sanitizer backward returned bad fields"
            )
        return (
            *(None for _ in range(11)),
            *(
                raw[name] if ctx.needs_input_grad[index] else None
                for index, name in enumerate(_CONTINUOUS_FIELDS, start=11)
            ),
            None,
        )

    @staticmethod
    def jvp(ctx: Any, *tangents: Any) -> Any:
        continuous_tangents = tuple(
            _ad_native_tangent_or_none(value) for value in tangents[11:22]
        )
        if all(value is None for value in continuous_tangents):
            return (None,) * len(_SANITIZE_OUTPUT_FIELDS)
        valid, selected_row_index = (
            _ad_native_tensor(value) for value in ctx.saved_tensors
        )
        with disable_functorch():
            raw = _evaluated_paths_capacity_pack_jvp_native(
                valid,
                selected_row_index,
                *continuous_tangents,
                ctx.candidate_count,
                ctx.sequence_width,
            )
        if not isinstance(raw, dict) or set(raw) != set(_CONTINUOUS_FIELDS):
            raise TypeError(
                "native enumerated failure sanitizer JVP returned bad fields"
            )
        return (
            *(None for _ in range(_SANITIZE_DISCRETE_OUTPUT_COUNT)),
            *(raw[name] for name in _CONTINUOUS_FIELDS),
        )


def enumerated_capacity_failure_sanitize(
    paths: EvaluatedPaths, *, failure_state: CapacityFailureState,
) -> EvaluatedPaths:
    """Make every final enumerated row inert after any transaction failure."""

    tensors = _validate_candidate(paths)
    require_capacity_failure_state(failure_state, device=tensors[0].device)
    outputs = _EnumeratedCapacityFailureSanitizeFunction.apply(
        *tensors, failure_state.bits
    )
    raw = dict(zip(_SANITIZE_OUTPUT_FIELDS, outputs, strict=True))
    topology = PathTopology(
        valid=raw["valid"],
        tx_id=raw["tx_id"],
        rx_id=raw["rx_id"],
        depth=raw["depth"],
        component_id=raw["component_id"],
        primitive_id=raw["primitive_id"],
        edge_id=raw["edge_id"],
        material_id=raw["material_id"],
        primitive_sequence=raw["primitive_sequence"],
        material_sequence=raw["material_sequence"],
        interaction_type=raw["interaction_type"],
    )
    geometry = PathGeometry(
        row_identity=topology.row_identity,
        path_length_m=raw["path_length_m"],
        delay_s=raw["delay_s"],
        field_direction=raw["field_direction"],
        interaction_position=raw["interaction_position"],
        interaction_normal=raw["interaction_normal"],
        interaction_positions=raw["interaction_positions"],
        interaction_normals=raw["interaction_normals"],
    )
    fields = PathFields(
        row_identity=topology.row_identity,
        path_gain=raw["path_gain"],
        path_field=raw["path_field"],
        field_xyz=raw["field_xyz"],
        coefficient=raw["coefficient"],
    )
    return EvaluatedPaths(topology=topology, geometry=geometry, fields=fields)


def sanitize_enumerated_capacity_transaction(
    paths: EvaluatedPaths, sidecars: EvaluatedPathSidecars,
) -> tuple[EvaluatedPaths, EvaluatedPathSidecars]:
    """Sanitize all enumerated payloads before outer solver result assembly."""

    if not isinstance(sidecars, EvaluatedPathSidecars):
        raise TypeError("sidecars must be EvaluatedPathSidecars")
    transaction = sidecars.capacity_transaction
    if transaction is None:
        return paths, sidecars
    sanitized = enumerated_capacity_failure_sanitize(
        paths, failure_state=transaction.failure_state
    )
    return sanitized, sidecars


# --- The canonical typed engine for enumerated propagation paths ------------


@dataclass(frozen=True, slots=True)
class EnumeratedEndpointTensors:
    """Explicit batch seam for solver-neutral propagation consumers."""

    tx_positions: torch.Tensor
    tx_power: torch.Tensor
    tx_polarizations: torch.Tensor
    rx_positions: torch.Tensor
    rx_polarizations: torch.Tensor
    tx_stable_ids: torch.Tensor | None = None
    rx_stable_ids: torch.Tensor | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.tx_positions, torch.Tensor):
            raise TypeError("tx_positions must be a torch.Tensor")
        device = self.tx_positions.device
        entries = (
            ("tx_positions", self.tx_positions, 2, (3,)),
            ("tx_power", self.tx_power, 1, ()),
            ("tx_polarizations", self.tx_polarizations, 2, (3,)),
            ("rx_positions", self.rx_positions, 2, (3,)),
            ("rx_polarizations", self.rx_polarizations, 2, (3,)),
        )
        for name, value, ndim, trailing_shape in entries:
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"{name} must be a torch.Tensor")
            if value.device != device or device.type != "cuda":
                raise ValueError("endpoint tensors must share one CUDA device")
            if value.dtype != torch.float32 or value.ndim != ndim:
                raise TypeError(f"{name} must be a rank-{ndim} float32 tensor")
            if trailing_shape and tuple(value.shape[1:]) != trailing_shape:
                raise ValueError(f"{name} has an invalid trailing shape")
        tx_count = int(self.tx_positions.shape[0])
        rx_count = int(self.rx_positions.shape[0])
        if self.tx_power.shape != (tx_count,):
            raise ValueError("tx_power must have shape (num_sources,)")
        if self.tx_polarizations.shape != (tx_count, 3):
            raise ValueError("tx_polarizations must match tx_positions")
        if self.rx_polarizations.shape != (rx_count, 3):
            raise ValueError("rx_polarizations must match rx_positions")
        if (self.tx_stable_ids is None) != (self.rx_stable_ids is None):
            raise ValueError("endpoint stable IDs must be provided together")
        if self.tx_stable_ids is not None:
            assert self.rx_stable_ids is not None
            for name, value, count in (
                ("tx_stable_ids", self.tx_stable_ids, tx_count),
                ("rx_stable_ids", self.rx_stable_ids, rx_count),
            ):
                if (
                    value.device != device
                    or value.dtype != torch.int64
                    or value.ndim != 1
                    or value.shape != (count,)
                    or not value.is_contiguous()
                ):
                    raise ValueError(
                        f"{name} must be contiguous CUDA int64 with shape ({count},)"
                    )


def _path_components(config: EnumeratedPathConfig) -> set[str]:
    components = set(config.components)
    if config.max_depth == 0:
        # Every non-LoS component is a surface interaction that needs at least
        # one bounce. transmission is a wall penetration event and scattering is
        # a single-bounce rough-surface event, so both drop out at depth 0.
        components.discard("reflection")
        components.discard("diffraction")
        components.discard("transmission")
        components.discard("scattering")
    return components


def _resolve_isb_taper(config: EnumeratedPathConfig) -> tuple[bool, float, float]:
    """Resolve the ISB boundary taper flag and per-stage widths.

    DEFAULT-OFF: when the taper is disabled the field/diffraction stages receive
    width 0.0 (every existing call path untouched) and the LoS membership stage
    receives the default 0.5, preserving the pre-taper calls byte-for-byte.
    """

    enabled = bool(getattr(config, "isb_boundary_taper", False))
    configured_width = float(getattr(config, "isb_boundary_taper_width", 0.5))
    field_width = configured_width if enabled else 0.0
    los_width = configured_width if enabled else 0.5
    return enabled, field_width, los_width


def _require_defer_capacity_terminal(value: bool) -> None:
    if type(value) is not bool:
        raise TypeError("defer_capacity_terminal must be a bool")


def _create_transmission_capacity_transaction(
    scene: Scene, config: EnumeratedPathConfig, components: set[str], tx_positions: torch.Tensor,
    rx_positions: torch.Tensor,
) -> SolveCapacityTransaction | None:
    has_work = (
        "transmission" in components
        and config.max_depth >= 1
        and bool(scene.structures)
        and int(tx_positions.shape[0]) > 0
        and int(rx_positions.shape[0]) > 0
    )
    if not has_work:
        return None
    return create_solve_capacity_transaction(tx_positions)


def _capacity_execution_summary(
    execution: CapacityExecutionCounts | None,
) -> tuple[CapacityExecutionCounts | None, int]:
    if execution is None:
        return None, 0
    return execution, execution.candidate_capacity


def _finish_capacity_boundary(
    evaluated: EvaluatedPaths, sidecars: EvaluatedPathSidecars,
    transaction: SolveCapacityTransaction | None, *, defer_terminal: bool,
) -> tuple[EvaluatedPaths, EvaluatedPathSidecars]:
    if transaction is None or defer_terminal:
        return evaluated, sidecars
    evaluated, sidecars = sanitize_enumerated_capacity_transaction(evaluated, sidecars)
    transaction.terminal_check()
    return evaluated, sidecars


def transmission_dropped_rows(sidecars: EvaluatedPathSidecars) -> int:
    """Return the transmission rows dropped for crossing more walls than ``max_depth``.

    Observes the solve's capacity transaction if the compact boundary did not.
    """

    transaction = sidecars.capacity_transaction
    return 0 if transaction is None else transaction.terminal_check()


@dataclass(frozen=True, slots=True)
class BoundEnumeratedEndpoints:
    """Engine endpoint tensors from either scene-owned endpoints or an explicit batch."""

    tx_positions: torch.Tensor
    tx_power: torch.Tensor
    tx_polarizations: torch.Tensor
    rx_positions: torch.Tensor
    # Set only for an explicit batch: scene-owned endpoints keep their own
    # polarization and antenna handling inside the field stage.
    rx_polarizations: torch.Tensor | None
    explicit_tx_polarizations: torch.Tensor | None
    tx_stable_ids: torch.Tensor | None
    rx_stable_ids: torch.Tensor | None

    @property
    def explicit(self) -> bool:
        return self.explicit_tx_polarizations is not None


def bind_enumerated_endpoints(
    scene: Scene, endpoint_tensors: EnumeratedEndpointTensors | None, *, device: torch.device,
) -> BoundEnumeratedEndpoints:
    """The one adapter from a scene's endpoints, or an explicit batch, to engine tensors."""

    if endpoint_tensors is None:
        tx_positions, tx_power = transmitter_tensors(scene, device=device)
        rx_positions, _ = receiver_positions_and_layout(scene, device=device)
        return BoundEnumeratedEndpoints(
            tx_positions=tx_positions,
            tx_power=tx_power,
            tx_polarizations=transmitter_polarizations_as_stored(scene, device=device),
            rx_positions=rx_positions,
            rx_polarizations=None,
            explicit_tx_polarizations=None,
            tx_stable_ids=None,
            rx_stable_ids=None,
        )
    return BoundEnumeratedEndpoints(
        tx_positions=endpoint_tensors.tx_positions,
        tx_power=endpoint_tensors.tx_power,
        tx_polarizations=endpoint_tensors.tx_polarizations,
        rx_positions=endpoint_tensors.rx_positions,
        rx_polarizations=endpoint_tensors.rx_polarizations,
        explicit_tx_polarizations=endpoint_tensors.tx_polarizations,
        tx_stable_ids=endpoint_tensors.tx_stable_ids,
        rx_stable_ids=endpoint_tensors.rx_stable_ids,
    )


def explicit_endpoint_scene(compiled: CompiledScene) -> Scene:
    """A solver scene over ``compiled`` whose endpoints come only from an explicit batch."""

    from witwin.channel.scene.endpoints import SolverScene

    return SolverScene(
        compiled=compiled,
        structures=compiled.structures,
        transmitters=(),
        receivers=(),
        frequency=compiled.reference_frequency_hz,
        metadata=compiled.source.metadata,
    )


def _windowed_receivers(
    rx_positions: torch.Tensor, receiver_window: tuple[int, int] | None,
) -> tuple[torch.Tensor, int]:
    """The receivers whose rows a solve discovers, and the id of the first in the whole set."""

    if receiver_window is None:
        return rx_positions, 0
    start, stop = (int(value) for value in receiver_window)
    if not 0 <= start < stop <= int(rx_positions.shape[0]):
        raise ValueError(
            f"receiver_window {receiver_window} is outside the {int(rx_positions.shape[0])} receivers"
        )
    return rx_positions[start:stop].contiguous(), start


def _offset_receiver_ids(blocks: list[dict[str, torch.Tensor]], start: int) -> None:
    """Give window-local receiver ids their ids in the whole receiver set."""

    if start == 0:
        return
    for block in blocks:
        block["rx_id"] = block["rx_id"] + start


def evaluate_enumerated_paths(
    scene: Scene, config: EnumeratedPathConfig, *, frequency_value: float | None = None,
    defer_capacity_terminal: bool = False,
    endpoint_tensors: EnumeratedEndpointTensors | None = None,
    receiver_window: tuple[int, int] | None = None,
) -> tuple[EvaluatedPaths, EvaluatedPathSidecars]:
    """Discover, select, and evaluate canonical enumerated propagation rows.

    ``defer_capacity_terminal`` is reserved for Path and Deterministic outer
    solvers, which must sanitize scattering-appended rows and enqueue the one
    terminal observer only after their public result/array packing. the enumerated-path oracle callers keep the default and complete the transaction here.
    ``receiver_window`` ``(start, stop)`` discovers rows for those receivers only; the rows keep
    their receiver ids in the whole receiver set, and every row is the one a whole solve gives.
    """

    _require_defer_capacity_terminal(defer_capacity_terminal)
    device = torch.device("cuda")
    endpoints = bind_enumerated_endpoints(scene, endpoint_tensors, device=device)
    tx_positions = endpoints.tx_positions
    tx_power = endpoints.tx_power
    tx_polarizations = endpoints.tx_polarizations
    rx_all = endpoints.rx_positions
    rx_positions, window_start = _windowed_receivers(rx_all, receiver_window)
    rx_polarizations = endpoints.rx_polarizations
    diffraction_tx_polarizations = endpoints.explicit_tx_polarizations
    field_tx_power = (
        tx_power.detach()
        if bool(getattr(config, "_detach_field_tx_power", False))
        else tx_power
    )
    compiled = require_compiled(scene)
    # One host read of a tensor frequency for the whole export: discovery and
    # the field seam below share this detached scalar. Callers
    # that already read it pass it in.
    frequency_hz = (
        _frequency_scalar(scene) if frequency_value is None else float(frequency_value)
    )
    ad_mode = str(getattr(config, "ad_mode", "none"))
    if ad_mode != "none":
        _require_frequency_ad_constant_materials(scene, compiled, ad_mode=ad_mode)
    components = _path_components(config)
    # ISB boundary taper: DEFAULT-OFF. When off, width 0.0 flows to the
    # field/diffraction stages and every existing call path is untouched.
    isb_boundary_taper, isb_boundary_taper_width, isb_los_width = _resolve_isb_taper(
        config
    )
    coupled_paths = bool(getattr(config, "coupled_paths", False))
    sequence_width = max(int(config.max_depth), chain_depth_limit(config.max_depth) if coupled_paths else 0)
    blocks: list[dict[str, torch.Tensor]] = []
    launch_count = 0
    visibility_rejection_count = 0
    candidate_count = 0
    guardrail_count = 0
    capacity_execution: CapacityExecutionCounts | None = None
    capacity_transaction: SolveCapacityTransaction | None = None

    capacity_transaction = _create_transmission_capacity_transaction(
        scene, config, components, tx_positions, rx_positions
    )

    if "los" in components:
        los_block, los_launches, los_candidates, los_visibility_rejections = (
            _los_topology(
                scene,
                compiled,
                tx_positions,
                tx_power,
                rx_positions,
                tx_polarizations,
                frequency_hz=frequency_hz,
                sequence_width=sequence_width,
                isb_boundary_taper=isb_boundary_taper,
                isb_boundary_taper_width=isb_los_width,
            )
        )
        launch_count += los_launches
        candidate_count += los_candidates
        visibility_rejection_count += los_visibility_rejections
        blocks.append(los_block)
    discovery = str(getattr(config, "discovery", "auto"))
    discovery_record: ReflectionDiscoveryRecord | None = None
    if "reflection" in components and config.max_depth >= 1:
        discovery_record = ReflectionDiscoveryRecord()
        block, reflection_launches, reflection_candidates = _reflection_topology(
            scene,
            compiled,
            tx_positions,
            rx_positions,
            min_depth=1,
            max_depth=int(config.max_depth),
            discovery=discovery,
            record=discovery_record,
        )
        launch_count += reflection_launches
        candidate_count += int(reflection_candidates)
        blocks.append(block)
    if "diffraction" in components and config.max_depth >= 1:
        block, diffraction_launches = _diffraction_topology_order1(
            scene,
            compiled,
            tx_positions,
            tx_power,
            rx_positions,
            frequency_hz=frequency_hz,
            isb_boundary_taper_width=isb_boundary_taper_width,
            tx_polarizations=diffraction_tx_polarizations,
        )
        launch_count += diffraction_launches
        candidate_count += int(block["valid"].numel())
        blocks.append(block)
    if "transmission" in components and config.max_depth >= 1:
        block, transmission_launches, transmission_execution = _transmission_topology(
            scene,
            compiled,
            tx_positions,
            rx_positions,
            max_depth=int(config.max_depth),
            ad_mode=ad_mode,
            failure_state=(
                capacity_transaction.failure_state
                if capacity_transaction is not None
                else None
            ),
        )
        launch_count += transmission_launches
        capacity_execution, transmission_candidates = _capacity_execution_summary(
            transmission_execution
        )
        candidate_count += transmission_candidates
        blocks.append(block)
    if (
        coupled_paths
        and config.max_depth >= 2
        and {"reflection", "diffraction"}.issubset(components)
    ):
        block, coupled_launches, coupled_candidates = (
            coupled_reflection_diffraction_topology(
                scene,
                compiled,
                tx_positions,
                rx_positions,
                max_depth=int(config.max_depth),
                candidate_limit=int(
                    getattr(config, "coupled_candidate_limit", 1_000_000)
                ),
            )
        )
        launch_count += coupled_launches
        candidate_count += coupled_candidates
        blocks.append(block)
    _offset_receiver_ids(blocks, window_start)
    rx_positions = rx_all
    if len(blocks) == 1 and components == {"los"} and config.max_paths is None:
        evaluated, sidecars = evaluated_paths_from_result(
            SimpleNamespace(
                **blocks[0],
                launch_count=launch_count,
                visibility_rejection_count=visibility_rejection_count,
                selected_edge_count=0,
                candidate_count=candidate_count,
                guardrail_count=guardrail_count,
            )
        )
        evaluated, execution = evaluate_path_fields(
            scene,
            compiled,
            evaluated,
            sidecars.execution,
            tx_positions,
            field_tx_power,
            rx_positions,
            row_groups=PathRowGroups.single(
                0,
                evaluated.row_count,
                sequence_width=int(evaluated.topology.primitive_sequence.shape[1]),
            ),
            components=components,
            ad_mode=ad_mode,
            frequency_value=frequency_hz,
            isb_boundary_taper_width=isb_boundary_taper_width,
            endpoint_tx_polarizations=endpoints.explicit_tx_polarizations,
            endpoint_rx_polarizations=rx_polarizations,
            explicit_endpoint_geometry=endpoints.explicit,
        )
        sidecars = replace(
            sidecars,
            execution=execution,
            capacity_execution=capacity_execution,
            capacity_transaction=capacity_transaction,
            compact_metadata=topology_kernels.enumerated_exact_pair_metadata(
                evaluated.topology.tx_id,
                evaluated.topology.rx_id,
                pair_count=int(tx_positions.shape[0]) * int(rx_positions.shape[0]),
                num_tx=int(tx_positions.shape[0]),
                num_rx=int(rx_positions.shape[0]),
                source_stable_ids=endpoints.tx_stable_ids,
                sink_stable_ids=endpoints.rx_stable_ids,
            ),
        )
        return _finish_capacity_boundary(
            evaluated,
            sidecars,
            capacity_transaction,
            defer_terminal=defer_capacity_terminal,
        )
    padded_blocks = [
        block
        if "primitive_sequence" in block
        and int(block["primitive_sequence"].shape[1]) == sequence_width
        else _pad_topology_sequences(block, width=sequence_width)
        for block in blocks
    ]
    paths = concatenate_path_blocks(padded_blocks, device=device)
    evaluated, sidecars = evaluated_paths_from_block(
        paths,
        max_paths=config.max_paths,
        max_paths_scope=str(getattr(config, "max_paths_scope", "global")),
        tx_count=int(tx_positions.shape[0]),
        rx_count=int(rx_positions.shape[0]),
        max_depth=sequence_width,
        launch_count=launch_count,
        visibility_rejection_count=visibility_rejection_count,
        candidate_count=candidate_count,
        guardrail_count=guardrail_count,
        source_stable_ids=endpoints.tx_stable_ids,
        sink_stable_ids=endpoints.rx_stable_ids,
        capacity_transaction=capacity_transaction,
    )
    sidecars = replace(
        sidecars,
        execution=replace(
            sidecars.execution,
            selected_edge_ids=paths["edge_id"],
            reflection_discovery=(
                discovery_record.metadata(discovery) if discovery_record is not None else None
            ),
        ),
        capacity_execution=capacity_execution,
        capacity_transaction=capacity_transaction,
    )
    evaluated, execution = evaluate_path_fields(
        scene,
        compiled,
        evaluated,
        sidecars.execution,
        tx_positions,
        field_tx_power,
        rx_positions,
        # evaluated_paths_from_block always groups its rows.
        row_groups=cast(PathRowGroups, sidecars.row_groups),
        components=components,
        ad_mode=ad_mode,
        frequency_value=frequency_hz,
        isb_boundary_taper_width=isb_boundary_taper_width,
        endpoint_tx_polarizations=endpoints.explicit_tx_polarizations,
        endpoint_rx_polarizations=rx_polarizations,
        explicit_endpoint_geometry=endpoints.explicit,
    )
    sidecars = replace(sidecars, execution=execution)
    return _finish_capacity_boundary(
        evaluated,
        sidecars,
        capacity_transaction,
        defer_terminal=defer_capacity_terminal,
    )
