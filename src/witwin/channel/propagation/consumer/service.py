"""Consumer orchestration over the canonical enumerated and compact owners."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from .contracts import (
    Complex3Transport,
    EndpointBatch,
    FixedTopologyEvaluation,
    FixedTopologyRequest,
    JonesTransport,
    PreparedFixedTopology,
    PropagationConvention,
    PropagationDiagnostics,
    PropagationEvaluation,
    PropagationGeometry,
    PropagationPathBatch,
    PropagationRequest,
    PropagationTopology,
    ScalarTransport,
    capabilities,
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


_CAPABILITIES = capabilities()
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


def _has_forward_tangent(value: torch.Tensor) -> bool:
    return torch.autograd.forward_ad.unpack_dual(value).tangent is not None


def _carries_ad(value: torch.Tensor | None) -> bool:
    return value is not None and (
        value.requires_grad or _has_forward_tangent(value)
    )


def _require_fixed_los_ad_inputs(request: FixedTopologyRequest) -> None:
    """Reject AD on inputs the native field companions treat as constants.

    Transmit power, endpoint polarizations, and the polarization reference
    vectors are rejected by the native forward/backward/JVP contracts, so a
    differentiable request carrying one of them fails here rather than
    producing a silently incomplete derivative.
    """

    if request.ad_mode == "none":
        return
    fixed_inputs = (
        ("sources.powers_w", request.sources.powers_w),
        ("sources.polarizations", request.sources.polarizations),
        ("sinks.polarizations", request.sinks.polarizations),
        ("sources.polarization_basis", request.sources.polarization_basis),
        ("sinks.polarization_basis", request.sinks.polarization_basis),
    )
    for name, value in fixed_inputs:
        if _carries_ad(value):
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
        if _carries_ad(value):
            raise NotImplementedError(
                f"polarimetric_transport {name} is primal-only; the operator "
                "is published in a frozen world-referenced transverse basis"
            )
    # The capability record declares tx_power and the endpoint polarizations
    # frozen too, and a declaration nobody enforces is the ADR-036 pattern.
    # The composed operator excites the transport with the two basis vectors,
    # so an endpoint polarization never reaches it at all and a gradient on one
    # could only ever come back empty; tx_power reaches a companion that does
    # not differentiate it. Refuse instead of returning a partial derivative.
    for name, value in (
        ("sources.powers_w", request.sources.powers_w),
        ("sources.polarizations", request.sources.polarizations),
        ("sinks.polarizations", request.sinks.polarizations),
    ):
        if _carries_ad(value):
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
    if _carries_ad(request.sources.positions_m) or _carries_ad(
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


def _fused_los_jones(
    compact: _ConsumerRows,
    *,
    sources: EndpointBatch,
    sinks: EndpointBatch,
    reference_frequency_hz: float | torch.Tensor,
) -> JonesTransport:
    """Primal-only fused operator: one native launch for the whole batch."""

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

    from witwin.channel.propagation.fields.kernels import (
        autograd as field_autograd,
    )

    from ._jones import compose_jones, transverse_basis
    from ._rows import select_rows

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
        lambda polarization: field_autograd.field_free_space_ad(
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
        return ScalarTransport(coefficient=fields.coefficient)
    if response == "complex3_transport":
        return Complex3Transport(
            field=fields.field_xyz,
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
        ad_mode=request.ad_mode,
        frequency_value=compiled.materials.frequency_hz,
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
    _require_fixed_los_ad_inputs(request)
    return compiled, request


def _fixed_transport(
    response: str, outputs: object
) -> ScalarTransport | Complex3Transport | JonesTransport:
    if response == "scalar_transport":
        return ScalarTransport(coefficient=outputs.coefficient)
    if response == "complex3_transport":
        return Complex3Transport(
            field=outputs.field_vector, direction=outputs.direction
        )
    assert outputs.matrix is not None
    return JonesTransport(
        matrix=outputs.matrix,
        source_basis=outputs.source_basis,
        sink_basis=outputs.sink_basis,
    )


def _reevaluate_prepared(
    compiled: CompiledScene, request: FixedTopologyRequest
) -> FixedTopologyEvaluation:
    """Replay a prepared frozen topology bucket by bucket."""

    from ._fixed_reflection import (
        evaluate_prepared,
        require_smooth_reflection_scene,
    )
    from ._rows import prepared_row_gather, select_rows

    prepared = request.topology
    assert isinstance(prepared, PreparedFixedTopology)
    validity = _CAPABILITIES.fixed_topology_row_validity_components
    if any(bucket.component == "reflection" for bucket in prepared.buckets):
        require_smooth_reflection_scene(compiled)
    rows = prepared_row_gather(prepared.topology, request.sources, request.sinks)
    bases = (
        (
            select_rows(request.sources.polarization_basis, rows.source_row_index),
            select_rows(request.sinks.polarization_basis, rows.sink_row_index),
        )
        if request.response == "polarimetric_transport"
        else (None, None)
    )
    outputs = evaluate_prepared(
        compiled,
        prepared,
        rows,
        response=request.response,
        ad_mode=request.ad_mode,
        frequency=request.reference_frequency_hz,
        frequency_value=compiled.materials.frequency_hz,
        source_reference_basis=bases[0],
        sink_reference_basis=bases[1],
        publish_row_validity=any(
            bucket.component in validity for bucket in prepared.buckets
        ),
    )
    paths = PropagationPathBatch(
        pair_count=request.sources.count * request.sinks.count,
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
        transport=_fixed_transport(request.response, outputs),
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
        row_valid=outputs.row_valid,
    )


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
    if isinstance(request.topology, PreparedFixedTopology):
        return _reevaluate_prepared(compiled, request)
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
    geometry = PropagationGeometry(
        path_length_m=field_rows["path_length_m"],
        delay_s=field_rows["delay_s"],
        field_direction=field_rows["direction"],
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
