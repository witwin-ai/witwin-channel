"""Enumerated propagation: the shared engine, its typed config protocols, and
its capacity failure sanitizers.

This is one module. The former ``enumerated/`` package split the same owner
across ``contracts.py`` (the scattering-stage config view), ``engine.py`` (the
canonical typed engine) and ``capacity.py`` (the final failure sanitizers);
``engine`` imported ``capacity`` and nothing else imported across the split, so
the three files were one unit already.

``__all__`` stays empty, exactly as the former package ``__init__`` published
it: this module is not a barrel facade, and in particular it publishes no
scattering surface. The former ``capacity.py`` carried its own ``__all__``
listing the five names it defined; that list only governed ``import *`` from a
submodule that no longer exists, and every one of those names is still a module
attribute reached by the same import path minus the ``.capacity`` segment.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from types import SimpleNamespace
from typing import TYPE_CHECKING, Protocol

import torch
from witwin.channel.scene.endpoints import require_compiled

from witwin.channel.materials import (
    _require_frequency_ad_constant_materials,
)
from witwin.channel.interactions.coupled import (
    coupled_reflection_diffraction_topology,
)
from witwin.channel.interactions.diffraction import (
    _diffraction_topology_order1,
)
from witwin.channel.interactions.los import _los_topology
from witwin.channel.interactions.reflection import (
    _reflection_topology_multibounce,
    _reflection_topology_order1,
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
    components: frozenset[str] | set[str] | tuple[str, ...] | list[str]
    max_depth: int
    scattering_samples_per_m2: float
    scattering_power_threshold: float
    scattering_max_paths_per_pair: int
    # ADR-021 D1 enumerated scatter-chain path class. DEFAULT-OFF: 0 disables
    # chain discovery entirely (the pipeline stays byte-identical). When >= 1 it
    # is the cap on d1 + d2, the combined reflection depth of the two specular
    # legs around the single diffuse vertex; each leg is independently bounded by
    # the native kMaxAdDepth = 8, so the public cap is 2 * 8 = 16.
    scattering_chain_max_depth: int
    # Chain-sample vertex density (samples / m^2). Documented lower density than
    # the single-bounce scattering sampler (scattering_samples_per_m2) because a
    # chain vertex is joined against two specular legs (ADR-021 D1).
    scattering_chain_samples_per_m2: float
    # Per-(tx, rx) keep-strongest cap on joined chain rows (ADR-021 D1 budget).
    scattering_chain_max_rows: int


# The structural view of a solver config this engine reads. It is declared here,
# beside its only consumer, so a solver config satisfies it without importing a
# solver.
class EnumeratedPathConfig(Protocol):
    max_depth: int
    components: frozenset[str] | set[str] | tuple[str, ...] | list[str]
    max_paths: int | None
    max_paths_scope: str


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


def _enumerated_capacity_failure_vector_sanitize_native(
    failure_state_bits: torch.Tensor,
    values: torch.Tensor,
) -> torch.Tensor:
    output = _required_native_op("enumerated_capacity_failure_vector_sanitize")(
        failure_state_bits, values
    )
    if not isinstance(output, torch.Tensor):
        raise TypeError("native enumerated vector sanitizer returned a non-tensor")
    return output


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
    def forward(*inputs):
        raw = _required_native_op("enumerated_capacity_failure_sanitize")(
            inputs[22], *inputs[:22]
        )
        if not isinstance(raw, dict) or set(raw) != set(_SANITIZE_OUTPUT_FIELDS):
            raise TypeError("native enumerated failure sanitizer returned bad fields")
        return tuple(raw[name] for name in _SANITIZE_OUTPUT_FIELDS)

    @staticmethod
    def setup_context(ctx, inputs, output):
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
    def backward(ctx, *grad_outputs):
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
    def jvp(ctx, *tangents):
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
    paths: EvaluatedPaths,
    *,
    failure_state: CapacityFailureState,
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


class _EnumeratedCapacityFailureVectorSanitizeFunction(torch.autograd.Function):
    """Failure-aware complex-vector identity with native VJP/JVP copies."""

    @staticmethod
    def forward(failure_state_bits, values):
        return _enumerated_capacity_failure_vector_sanitize_native(
            failure_state_bits, values
        )

    @staticmethod
    def setup_context(ctx, inputs, output):
        del output
        ctx.set_materialize_grads(False)
        failure_state_bits = torch.autograd.forward_ad.unpack_dual(inputs[0]).primal
        ctx.save_for_backward(failure_state_bits)
        ctx.save_for_forward(failure_state_bits)

    @staticmethod
    @_ad_first_order_only
    def backward(ctx, grad_output):
        if grad_output is None or not ctx.needs_input_grad[1]:
            return None, None
        (failure_state_bits,) = ctx.saved_tensors
        grad_values = _enumerated_capacity_failure_vector_sanitize_native(
            failure_state_bits, grad_output
        )
        return None, grad_values

    @staticmethod
    def jvp(ctx, _failure_tangent, values_tangent):
        values_tangent = _ad_native_tangent_or_none(values_tangent)
        if values_tangent is None:
            return None
        (failure_state_bits,) = (
            _ad_native_tensor(value) for value in ctx.saved_tensors
        )
        with disable_functorch():
            return _enumerated_capacity_failure_vector_sanitize_native(
                failure_state_bits, values_tangent
            )


def enumerated_capacity_failure_vector_sanitize(
    values: torch.Tensor,
    *,
    failure_state: CapacityFailureState,
) -> torch.Tensor:
    """Sanitize the deterministic diffraction vector sidecar on failure."""

    if not isinstance(values, torch.Tensor):
        raise TypeError("diffraction_vector_field must be a torch.Tensor")
    if not values.is_cuda or values.dtype != torch.complex64:
        raise ValueError("diffraction_vector_field must be a CUDA complex64 tensor")
    if values.ndim != 3 or values.shape[2] != 3:
        raise ValueError("diffraction_vector_field must have shape (T, R, 3)")
    require_capacity_failure_state(failure_state, device=values.device)
    return _EnumeratedCapacityFailureVectorSanitizeFunction.apply(
        failure_state.bits, values
    )


def sanitize_enumerated_capacity_transaction(
    paths: EvaluatedPaths,
    sidecars: EvaluatedPathSidecars,
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
    vector_field = sidecars.diffraction_vector_field
    if vector_field is not None:
        vector_field = enumerated_capacity_failure_vector_sanitize(
            vector_field, failure_state=transaction.failure_state
        )
    return sanitized, replace(sidecars, diffraction_vector_field=vector_field)


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
    if int(getattr(config, "max_diffraction_order", 1)) == 0:
        components.discard("diffraction")
    return components


def _resolve_isb_taper(config: EnumeratedPathConfig) -> tuple[bool, float, float]:
    """Resolve the ADR-017 ISB boundary taper flag and per-stage widths.

    DEFAULT-OFF: when the taper is disabled the field/diffraction stages receive
    width 0.0 (every existing call path untouched) and the LoS membership stage
    receives the default 0.5, preserving the pre-ADR-017 calls byte-for-byte.
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
    scene: Scene,
    config: EnumeratedPathConfig,
    components: set[str],
    tx_positions: torch.Tensor,
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
    evaluated: EvaluatedPaths,
    sidecars: EvaluatedPathSidecars,
    transaction: SolveCapacityTransaction | None,
    *,
    defer_terminal: bool,
) -> tuple[EvaluatedPaths, EvaluatedPathSidecars]:
    if transaction is None or defer_terminal:
        return evaluated, sidecars
    evaluated, sidecars = sanitize_enumerated_capacity_transaction(evaluated, sidecars)
    transaction.terminal_check()
    return evaluated, replace(sidecars, capacity_transaction=None)


def evaluate_enumerated_paths(
    scene: Scene,
    config: EnumeratedPathConfig,
    *,
    frequency_value: float | None = None,
    coupled_rx_streaming: bool = False,
    defer_capacity_terminal: bool = False,
    endpoint_tensors: EnumeratedEndpointTensors | None = None,
) -> tuple[EvaluatedPaths, EvaluatedPathSidecars]:
    """Discover, select, and evaluate canonical enumerated propagation rows.

    ``coupled_rx_streaming`` streams coupled reflection-diffraction discovery
    over receiver blocks so a full grid solve stays under the per-block
    candidate budget (ADR-011). The deterministic grid solver sets it; the path
    and Monte Carlo callers keep the single-shot total-cap discovery, so their
    coupled behavior is unchanged.

    ``defer_capacity_terminal`` is reserved for Path and Deterministic outer
    solvers, which must sanitize scattering-appended rows and enqueue the one
    terminal observer only after their public result/array packing. ADR-008
    opaque oracle callers keep the default and complete the transaction here.
    """

    _require_defer_capacity_terminal(defer_capacity_terminal)
    device = torch.device("cuda")
    if endpoint_tensors is None:
        tx_positions, tx_power = transmitter_tensors(scene, device=device)
        tx_polarizations = transmitter_polarizations_as_stored(scene, device=device)
        rx_positions, _ = receiver_positions_and_layout(scene, device=device)
        rx_polarizations = None
    else:
        tx_positions = endpoint_tensors.tx_positions
        tx_power = endpoint_tensors.tx_power
        tx_polarizations = endpoint_tensors.tx_polarizations
        rx_positions = endpoint_tensors.rx_positions
        rx_polarizations = endpoint_tensors.rx_polarizations
    field_tx_power = (
        tx_power.detach()
        if bool(getattr(config, "_detach_field_tx_power", False))
        else tx_power
    )
    compiled = require_compiled(scene)
    # One host read of a tensor frequency for the whole export: discovery and
    # the field seam below share this detached scalar (audit M3). Callers
    # that already read it (the solver seams) pass it in.
    frequency_hz = (
        _frequency_scalar(scene) if frequency_value is None else float(frequency_value)
    )
    ad_mode = str(getattr(config, "ad_mode", "none"))
    if ad_mode != "none":
        _require_frequency_ad_constant_materials(scene, compiled, ad_mode=ad_mode)
    components = _path_components(config)
    # ISB boundary taper (ADR-017): DEFAULT-OFF. When off, width 0.0 flows to the
    # field/diffraction stages and every existing call path is untouched.
    isb_boundary_taper, isb_boundary_taper_width, isb_los_width = _resolve_isb_taper(
        config
    )
    coupled_paths = bool(getattr(config, "coupled_paths", False))
    sequence_width = max(int(config.max_depth), 2 if coupled_paths else 0)
    blocks: list[dict[str, torch.Tensor]] = []
    launch_count = 0
    visibility_rejection_count = 0
    candidate_count = 0
    guardrail_count = 0
    diffraction_vector_field = None
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
    if "reflection" in components and config.max_depth >= 1:
        block, reflection_launches = _reflection_topology_order1(
            scene,
            compiled,
            tx_positions,
            tx_power,
            rx_positions,
            frequency_hz=frequency_hz,
        )
        launch_count += reflection_launches
        candidate_count += int(block["valid"].numel())
        blocks.append(block)
    if "reflection" in components and config.max_depth >= 2:
        block, reflection_launches, reflection_candidates = (
            _reflection_topology_multibounce(
                scene,
                compiled,
                tx_positions,
                tx_power,
                rx_positions,
                frequency_hz=frequency_hz,
                min_depth=2,
                max_depth=int(config.max_depth),
                max_paths=config.max_paths,
            )
        )
        launch_count += reflection_launches
        candidate_count += int(reflection_candidates)
        blocks.append(block)
    if "diffraction" in components and config.max_depth >= 1:
        block, diffraction_launches, diffraction_vector_field = (
            _diffraction_topology_order1(
                scene,
                compiled,
                tx_positions,
                tx_power,
                rx_positions,
                frequency_hz=frequency_hz,
                isb_boundary_taper_width=isb_boundary_taper_width,
            )
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
                candidate_limit=int(
                    getattr(config, "coupled_candidate_limit", 1_000_000)
                ),
                rx_streamed=coupled_rx_streaming,
            )
        )
        launch_count += coupled_launches
        candidate_count += coupled_candidates
        blocks.append(block)
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
            components=components,
            ad_mode=ad_mode,
            frequency_value=frequency_hz,
            isb_boundary_taper_width=isb_boundary_taper_width,
            endpoint_tx_polarizations=(
                tx_polarizations if endpoint_tensors is not None else None
            ),
            endpoint_rx_polarizations=rx_polarizations,
            explicit_endpoint_geometry=endpoint_tensors is not None,
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
                source_stable_ids=(
                    endpoint_tensors.tx_stable_ids
                    if endpoint_tensors is not None
                    else None
                ),
                sink_stable_ids=(
                    endpoint_tensors.rx_stable_ids
                    if endpoint_tensors is not None
                    else None
                ),
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
    selected_edge_count = topology_kernels.deterministic_selected_edge_count(
        paths["edge_id"]
    )
    evaluated, sidecars = evaluated_paths_from_block(
        paths,
        max_paths=config.max_paths,
        max_paths_scope=str(getattr(config, "max_paths_scope", "global")),
        tx_count=int(tx_positions.shape[0]),
        rx_count=int(rx_positions.shape[0]),
        max_depth=sequence_width,
        launch_count=launch_count,
        visibility_rejection_count=visibility_rejection_count,
        selected_edge_count=selected_edge_count,
        candidate_count=candidate_count,
        guardrail_count=guardrail_count,
        source_stable_ids=(
            endpoint_tensors.tx_stable_ids if endpoint_tensors is not None else None
        ),
        sink_stable_ids=(
            endpoint_tensors.rx_stable_ids if endpoint_tensors is not None else None
        ),
    )
    sidecars = replace(
        sidecars,
        diffraction_vector_field=diffraction_vector_field,
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
        components=components,
        ad_mode=ad_mode,
        frequency_value=frequency_hz,
        isb_boundary_taper_width=isb_boundary_taper_width,
        endpoint_tx_polarizations=(
            tx_polarizations if endpoint_tensors is not None else None
        ),
        endpoint_rx_polarizations=rx_polarizations,
        explicit_endpoint_geometry=endpoint_tensors is not None,
    )
    sidecars = replace(sidecars, execution=execution)
    return _finish_capacity_boundary(
        evaluated,
        sidecars,
        capacity_transaction,
        defer_terminal=defer_capacity_terminal,
    )
