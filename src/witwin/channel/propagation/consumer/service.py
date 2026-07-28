"""Consumer orchestration over the canonical enumerated and compact owners."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from .policy import (
    ad_ledger,
    carries_ad,
    require_first_order_request,
    require_primal_only_ad_inputs,
    tape_bytes,
)
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
    WorldProvenance,
    capabilities,
    native_frequency_resolution_hz,
)

if TYPE_CHECKING:
    from witwin.channel.propagation.enumerated.engine import (
        EnumeratedEndpointTensors,
    )
    from witwin.channel.propagation.rows import EvaluatedPaths
    from witwin.channel.propagation.topology.kernels.canonical_compact import (
        ExactPairMetadata,
    )
    from witwin.channel.scene.compiler import CompiledScene
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

    from .replay import consumer_los_jones

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

    from .replay import compose_jones, select_rows, transverse_basis

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
        return ScalarTransport(coefficient=fields.path_field)
    if response == "complex3_transport":
        from .replay import excited_field, select_rows

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
    """Refuse a frozen replay against a world that moved (ADR-040).

    Four host integer comparisons against the version domains the compiled
    scene recorded. No device work, no allocation, no synchronization, and no
    ADR-032 budget impact. A frozen topology with no provenance is hand-built
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

    This is the explicit rediscovery signal: poll it per frame and call
    :func:`evaluate` plus
    :func:`witwin.channel.propagation.consumer.prepare_fixed_topology` again
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
    compile frequency (ADR-026, Plan-13). Reusing a table built at ``f_ref`` at
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
    """Scene-dependent wideband refusals, each independent (ADR-042).

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

    from .replay import (
        GeometryLiveness,
        evaluate_prepared,
        prepared_row_gather,
        require_smooth_reflection_scene,
        scene_vertex_table,
        select_rows,
    )

    prepared = request.topology
    assert isinstance(prepared, PreparedFixedTopology)
    validity = _CAPABILITIES.fixed_topology_row_validity_components
    if any(bucket.component == "reflection" for bucket in prepared.buckets):
        require_smooth_reflection_scene(compiled)
    # The row gather owns the one validation copy and the one synchronization,
    # and it runs ONCE here, above the frequency-column loop below. That is what
    # holds the ADR-032 budget at 1/1 however many columns a wideband request
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
    # ADR-038: liveness is decided once, here, from the inputs every column
    # shares, and the same record reaches every column. ADR-043 adds the
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

    from witwin.channel.propagation.fields.kernels import (
        autograd as field_autograd,
    )
    from witwin.channel.propagation.fields.kernels import (
        functional as field_functional,
    )

    from .replay import (
        excited_field,
        fixed_los_gather,
        fixed_los_geometry_live,
        require_fixed_los_geometry_live,
    )

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
    # ADR-038: one liveness decision, taken here from the gathered rows, and
    # re-asserted for every column against the inputs that column launches on.
    geometry_live = fixed_los_geometry_live(rows)
    # ADR-043: this route carries line-of-sight rows only, so the direction
    # seam is Channel-owned for the whole result and its liveness is exactly
    # the geometry decision above.
    direction_live = geometry_live and frozenset({"los"}).issubset(
        _CAPABILITIES.direction_differentiable_components
    )
    ledger = ad_ledger(request.ad_mode)

    def column(offset: float) -> dict[str, torch.Tensor]:
        if request.ad_mode == "none":
            return field_functional.field_free_space(
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
        return field_autograd.field_free_space_ad(
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


__all__ = ["evaluate", "reevaluate", "rediscovery_required"]
