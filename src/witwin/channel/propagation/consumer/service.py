"""Consumer orchestration over the canonical enumerated and compact owners."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from .contracts import (
    CONTRACT_VERSION,
    Complex3Transport,
    EndpointBatch,
    FixedTopologyEvaluation,
    FixedTopologyRequest,
    JonesTransport,
    PropagationCapabilities,
    PropagationConvention,
    PropagationDiagnostics,
    PropagationEvaluation,
    PropagationGeometry,
    PropagationPathBatch,
    PropagationRequest,
    PropagationTopology,
    ScalarTransport,
)

if TYPE_CHECKING:
    from witwin.channel.propagation.enumerated.engine import (
        EnumeratedEndpointTensors,
    )
    from witwin.channel.propagation.models.evaluated import EvaluatedPaths
    from witwin.channel.propagation.topology.kernels.canonical_compact import (
        ExactPairMetadata,
    )
    from witwin.channel.scene.compiled import CompiledScene
    from witwin.channel.scene.endpoints import SolverScene


_COMPONENTS = frozenset({"los", "reflection", "transmission", "diffraction"})
_RESPONSES = frozenset(
    {"scalar_transport", "complex3_transport", "polarimetric_transport"}
)
_AD_MODES = frozenset({"none", "jvp", "vjp"})
_TOPOLOGY_MODES = frozenset({"discover"})

_CAPABILITIES = PropagationCapabilities(
    contract_version=CONTRACT_VERSION,
    components=_COMPONENTS,
    responses=_RESPONSES,
    topology_modes=_TOPOLOGY_MODES,
    ad_modes=_AD_MODES,
    response_components=(
        ("scalar_transport", _COMPONENTS),
        ("complex3_transport", _COMPONENTS),
        ("polarimetric_transport", frozenset({"los"})),
    ),
    response_ad_modes=(
        ("scalar_transport", _AD_MODES),
        ("complex3_transport", _AD_MODES),
        ("polarimetric_transport", frozenset({"none"})),
    ),
    component_ad_modes=(
        ("los", _AD_MODES),
        ("reflection", _AD_MODES),
        ("transmission", _AD_MODES),
        ("diffraction", _AD_MODES),
    ),
    fixed_topology_components=frozenset({"los"}),
    fixed_topology_responses=frozenset({"scalar_transport", "complex3_transport"}),
    supports_frequency_offsets=False,
    supports_fixed_topology=True,
    supports_los_jones=True,
)
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


def _require_endpoint_roles(sources: EndpointBatch, sinks: EndpointBatch) -> None:
    if sources.powers_w is None:
        raise ValueError("request.sources.powers_w is required")
    if sinks.powers_w is not None:
        raise ValueError("request.sinks.powers_w must be absent")
    if sources.device != sinks.device:
        raise ValueError("source and sink batches must share one CUDA device")


def _require_frequency_offsets(value: torch.Tensor | None) -> None:
    if value is not None:
        raise NotImplementedError(
            "frequency_offsets_hz is unsupported by consumer contract version 1"
        )


def _require_response(value: object) -> str:
    if not isinstance(value, str):
        raise TypeError("response must be a string")
    if value not in _RESPONSES:
        raise NotImplementedError(
            f"unsupported propagation response {value!r}; "
            f"supported responses are {sorted(_RESPONSES)}"
        )
    return value


def _require_ad_mode(value: object) -> str:
    if not isinstance(value, str):
        raise TypeError("ad_mode must be a string")
    if value not in _AD_MODES:
        raise NotImplementedError(
            f"unsupported propagation AD mode {value!r}; "
            f"supported modes are {sorted(_AD_MODES)}"
        )
    return value


def _has_forward_tangent(value: torch.Tensor) -> bool:
    return torch.autograd.forward_ad.unpack_dual(value).tangent is not None


def _require_fixed_los_ad_inputs(request: FixedTopologyRequest) -> None:
    if request.ad_mode == "none":
        return
    fixed_inputs = (
        ("sources.powers_w", request.sources.powers_w),
        ("sources.polarizations", request.sources.polarizations),
        ("sinks.polarizations", request.sinks.polarizations),
    )
    for name, value in fixed_inputs:
        assert value is not None
        if value.requires_grad or _has_forward_tangent(value):
            raise NotImplementedError(
                f"fixed LoS {name} is primal-only; only endpoint positions "
                "and reference_frequency_hz support AD"
            )


def _preflight_evaluate(
    compiled: object, request: object
) -> tuple[CompiledScene, PropagationRequest]:
    from witwin.channel.scene.compiled import CompiledScene

    if not isinstance(compiled, CompiledScene):
        raise TypeError("evaluate requires a CompiledScene")
    if not isinstance(request, PropagationRequest):
        raise TypeError("request must be a PropagationRequest")
    if not isinstance(request.sources, EndpointBatch) or not isinstance(
        request.sinks, EndpointBatch
    ):
        raise TypeError("sources and sinks must be EndpointBatch instances")
    _require_endpoint_roles(request.sources, request.sinks)
    if type(request.components) is not frozenset or not request.components:
        raise TypeError("components must be a non-empty frozenset")
    unsupported = request.components - _COMPONENTS
    if unsupported:
        raise NotImplementedError(
            f"unsupported propagation components: {sorted(unsupported)}"
        )
    if type(request.max_depth) is not int or not 0 <= request.max_depth <= 5:
        raise ValueError("max_depth must be an int in [0, 5]")
    if request.max_paths is not None and (
        type(request.max_paths) is not int or request.max_paths <= 0
    ):
        raise ValueError("max_paths must be a positive int when set")
    if not isinstance(request.topology_mode, str):
        raise TypeError("topology_mode must be a string")
    if request.topology_mode not in _TOPOLOGY_MODES:
        raise NotImplementedError(
            f"unsupported topology_mode {request.topology_mode!r}"
        )
    _require_response(request.response)
    _require_ad_mode(request.ad_mode)
    _require_frequency_offsets(request.frequency_offsets_hz)
    response_components = dict(_CAPABILITIES.response_components)[request.response]
    if not request.components.issubset(response_components):
        raise NotImplementedError(
            f"{request.response} does not support components "
            f"{sorted(request.components - response_components)}"
        )
    response_ad_modes = dict(_CAPABILITIES.response_ad_modes)[request.response]
    if request.ad_mode not in response_ad_modes:
        raise NotImplementedError(
            f"{request.response} does not support AD mode {request.ad_mode!r}"
        )
    component_ad_modes = dict(_CAPABILITIES.component_ad_modes)
    unsupported_ad = sorted(
        component
        for component in request.components
        if request.ad_mode not in component_ad_modes[component]
    )
    if unsupported_ad:
        raise NotImplementedError(
            f"AD mode {request.ad_mode!r} is unsupported for components "
            f"{unsupported_ad}"
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
        if isinstance(request.reference_frequency_hz, torch.Tensor):
            raise NotImplementedError(
                "polarimetric_transport requires a scalar compiled frequency"
            )
        polarimetric_inputs = (
            request.sources.positions_m,
            request.sinks.positions_m,
            request.sources.polarization_basis,
            request.sinks.polarization_basis,
        )
        if any(
            value.requires_grad or _has_forward_tangent(value)
            for value in polarimetric_inputs
            if value is not None
        ):
            raise NotImplementedError(
                "polarimetric_transport is primal-only in contract version 1"
            )
    compiled.require_reference_frequency(request.reference_frequency_hz)
    return compiled, request


def _solver_scene(
    compiled: CompiledScene, sources: EndpointBatch, sinks: EndpointBatch
) -> tuple[SolverScene, EnumeratedEndpointTensors]:
    """Bind explicit request batches without consulting compiled endpoints."""

    from witwin.channel.propagation.enumerated.engine import (
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
    from witwin.channel.propagation.models.evaluated import EvaluatedPaths

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


def _transport(
    response: str,
    compact: _ConsumerRows,
    *,
    sources: EndpointBatch,
    sinks: EndpointBatch,
    reference_frequency_hz: float | torch.Tensor,
) -> ScalarTransport | Complex3Transport | JonesTransport:
    fields = compact.evaluated.fields
    geometry = compact.evaluated.geometry
    if response == "scalar_transport":
        return ScalarTransport(coefficient=fields.coefficient)
    if response == "complex3_transport":
        return Complex3Transport(
            field=fields.field_xyz,
            direction=geometry.field_direction,
        )
    if response == "polarimetric_transport":
        from ._native import consumer_los_jones

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
    raise AssertionError("response was not preflighted")


def _path_batch(
    compact: _ConsumerRows,
    *,
    pair_count: int,
    response: str,
    sources: EndpointBatch,
    sinks: EndpointBatch,
    reference_frequency_hz: float | torch.Tensor,
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
    )
    geometry = PropagationGeometry(
        path_length_m=continuous.path_length_m,
        delay_s=continuous.delay_s,
        field_direction=continuous.field_direction,
        interaction_position_m=continuous.interaction_position,
        interaction_normal=continuous.interaction_normal,
        interaction_positions_m=continuous.interaction_positions,
        interaction_normals=continuous.interaction_normals,
    )
    transport = _transport(
        response,
        compact,
        sources=sources,
        sinks=sinks,
        reference_frequency_hz=reference_frequency_hz,
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
    )


def evaluate(
    compiled_scene: CompiledScene, request: PropagationRequest
) -> PropagationEvaluation:
    """Discover and evaluate one all-or-nothing compact propagation batch."""

    from witwin.channel.propagation.enumerated.engine import (
        evaluate_enumerated_paths,
    )
    from witwin.channel.propagation.enumerated.capacity import (
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
    )
    return PropagationEvaluation(
        paths=paths,
        convention=_CONVENTION,
        capabilities=_CAPABILITIES,
        diagnostics=_diagnostics(sidecars, compact),
    )


def _preflight_reevaluate(
    compiled: object, request: object
) -> tuple[CompiledScene, FixedTopologyRequest]:
    from witwin.channel.scene.compiled import CompiledScene

    if not isinstance(compiled, CompiledScene):
        raise TypeError("reevaluate requires a CompiledScene")
    if not isinstance(request, FixedTopologyRequest):
        raise TypeError("request must be a FixedTopologyRequest")
    if not isinstance(request.sources, EndpointBatch) or not isinstance(
        request.sinks, EndpointBatch
    ):
        raise TypeError("sources and sinks must be EndpointBatch instances")
    if not isinstance(request.topology, PropagationTopology):
        raise TypeError("topology must be a PropagationTopology")
    _require_endpoint_roles(request.sources, request.sinks)
    if request.topology.device != request.sources.device:
        raise ValueError("fixed topology and endpoint batches must share a device")
    _require_frequency_offsets(request.frequency_offsets_hz)
    _require_response(request.response)
    _require_ad_mode(request.ad_mode)
    compiled.require_reference_frequency(request.reference_frequency_hz)
    if not _CAPABILITIES.supports_fixed_topology:
        raise NotImplementedError(
            "fixed-topology reevaluation is unavailable in this build"
        )
    if request.response not in _CAPABILITIES.fixed_topology_responses:
        raise NotImplementedError(
            f"response {request.response!r} has no fixed-topology provider"
        )
    if request.topology.primitive_sequence.shape[1] != 0:
        raise NotImplementedError(
            "fixed LoS reevaluation requires zero-width interaction sequences"
        )
    _require_fixed_los_ad_inputs(request)
    return compiled, request


def reevaluate(
    compiled_scene: CompiledScene, request: FixedTopologyRequest
) -> FixedTopologyEvaluation:
    """Reevaluate frozen rows without topology discovery or compaction."""

    from witwin.channel.propagation.fields.kernels import (
        autograd as field_autograd,
    )
    from witwin.channel.propagation.fields.kernels import (
        functional as field_functional,
    )

    from ._fixed_los import fixed_los_gather

    compiled, request = _preflight_reevaluate(compiled_scene, request)
    rows = fixed_los_gather(request.topology, request.sources, request.sinks)
    frequency = request.reference_frequency_hz
    frequency_value = compiled.materials.frequency_hz
    tx_power = rows.tx_power.detach()
    tx_polarization = rows.tx_polarization.detach()
    rx_polarization = rows.rx_polarization.detach()
    field_rows = (
        field_functional.field_free_space(
            rows.source,
            rows.target,
            tx_power,
            tx_polarization,
            rx_polarization,
            frequency_hz=frequency_value,
        )
        if request.ad_mode == "none"
        else field_autograd.field_free_space_ad(
            rows.source,
            rows.target,
            tx_power,
            tx_polarization,
            rx_polarization,
            frequency=frequency,
            frequency_value=frequency_value,
        )
    )
    row_count = rows.row_count
    empty_interactions = rows.source.new_empty((row_count, 0, 3))
    inert_interaction = rows.source.new_zeros((row_count, 3))
    geometry = PropagationGeometry(
        path_length_m=field_rows["path_length_m"],
        delay_s=field_rows["delay_s"],
        field_direction=field_rows["direction"],
        interaction_position_m=inert_interaction,
        interaction_normal=inert_interaction,
        interaction_positions_m=empty_interactions,
        interaction_normals=empty_interactions,
    )
    transport = (
        ScalarTransport(coefficient=field_rows["coefficient"])
        if request.response == "scalar_transport"
        else Complex3Transport(
            field=field_rows["field_vector"],
            direction=field_rows["direction"],
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
        ),
    )


__all__ = ["evaluate", "reevaluate"]
