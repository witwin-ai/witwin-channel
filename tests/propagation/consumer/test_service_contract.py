from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is required"
)


def _compiled(frequency: float = 77.0e9):
    from witwin.channel.scene.compiled import CompiledScene

    compiled = object.__new__(CompiledScene)
    object.__setattr__(
        compiled,
        "source",
        SimpleNamespace(metadata={}, endpoints=("compiled-endpoint-must-not-run",)),
    )
    object.__setattr__(compiled, "structures", ())
    # mu_r / layer_mu_r are named by capabilities().primal_only_ad_inputs, so
    # the stand-in has to declare them even though this scene has no material
    # store; None is "this scene carries no such tensor", not "unset".
    object.__setattr__(
        compiled,
        "materials",
        SimpleNamespace(frequency_hz=float(frequency), mu_r=None, layer_mu_r=None),
    )
    object.__setattr__(compiled, "reference_frequency_hz", frequency)
    object.__setattr__(compiled, "reference_frequency_revision", None)
    # The four world version domains the ADR-040 freshness check reads, plus
    # the compiled snapshot instant. A stand-in for a CompiledScene has to
    # carry them or `evaluate` cannot stamp the topology it publishes.
    for name in (
        "topology_version",
        "geometry_version",
        "material_version",
        "assignment_version",
    ):
        object.__setattr__(compiled, name, 7)
    object.__setattr__(compiled, "time_s", None)
    return compiled


def _endpoints():
    from witwin.channel.propagation.consumer import EndpointBatch

    sources = EndpointBatch(
        stable_ids=torch.tensor([101, 303], device="cuda", dtype=torch.int64),
        positions_m=torch.tensor(
            [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
            device="cuda",
            dtype=torch.float32,
        ),
        polarizations=torch.tensor(
            [[0.0, 0.0, 1.0], [0.0, 1.0, 0.0]],
            device="cuda",
            dtype=torch.float32,
        ),
        powers_w=torch.ones((2,), device="cuda", dtype=torch.float32),
    )
    sinks = EndpointBatch(
        stable_ids=torch.tensor([707], device="cuda", dtype=torch.int64),
        positions_m=torch.tensor(
            [[10.0, 0.0, 0.0]], device="cuda", dtype=torch.float32
        ),
        polarizations=torch.tensor(
            [[0.0, 0.0, 1.0]], device="cuda", dtype=torch.float32
        ),
    )
    return sources, sinks


def _compact_rows():
    from witwin.channel.propagation.consumer._native import CompactEvaluatedPaths
    from witwin.channel.propagation.models import (
        EvaluatedPaths,
        PathFields,
        PathGeometry,
        PathTopology,
    )

    device = torch.device("cuda")
    rows = 2
    empty_sequence = torch.empty((rows, 0), device=device, dtype=torch.int32)
    topology = PathTopology(
        valid=torch.ones((rows,), device=device, dtype=torch.bool),
        tx_id=torch.tensor([0, 1], device=device, dtype=torch.int32),
        rx_id=torch.zeros((rows,), device=device, dtype=torch.int32),
        depth=torch.zeros((rows,), device=device, dtype=torch.int32),
        component_id=torch.zeros((rows,), device=device, dtype=torch.int32),
        primitive_id=torch.full((rows,), -1, device=device, dtype=torch.int32),
        edge_id=torch.full((rows,), -1, device=device, dtype=torch.int32),
        material_id=torch.full((rows,), -1, device=device, dtype=torch.int32),
        primitive_sequence=empty_sequence,
        material_sequence=empty_sequence,
        interaction_type=empty_sequence,
    )
    geometry = PathGeometry(
        row_identity=topology.row_identity,
        path_length_m=torch.tensor([10.0, 8.0], device=device),
        delay_s=torch.tensor([1.0, 2.0], device=device),
        field_direction=torch.randn((rows, 3), device=device),
        interaction_position=torch.zeros((rows, 3), device=device),
        interaction_normal=torch.zeros((rows, 3), device=device),
        interaction_positions=torch.empty((rows, 0, 3), device=device),
        interaction_normals=torch.empty((rows, 0, 3), device=device),
    )
    fields = PathFields(
        row_identity=topology.row_identity,
        path_gain=torch.ones((rows,), device=device),
        path_field=torch.ones((rows,), device=device, dtype=torch.complex64),
        field_xyz=torch.ones((rows, 3), device=device, dtype=torch.complex64),
        coefficient=torch.ones((rows,), device=device, dtype=torch.complex64),
    )
    return CompactEvaluatedPaths(
        path_count=rows,
        pair_index=torch.tensor([0, 1], device=device, dtype=torch.int64),
        pair_offsets=torch.tensor([0, 1, 2], device=device, dtype=torch.int64),
        source_id=torch.tensor([101, 303], device=device, dtype=torch.int64),
        sink_id=torch.tensor([707, 707], device=device, dtype=torch.int64),
        evaluated=EvaluatedPaths(topology=topology, geometry=geometry, fields=fields),
        count_d2h_copies=1,
        count_d2h_bytes=8,
        count_synchronizations=1,
        native_launch_count=1,
    )


def _fixed_topology():
    from witwin.channel.propagation.consumer import PropagationTopology

    device = torch.device("cuda")
    rows = 2
    empty_sequence = torch.empty((rows, 0), device=device, dtype=torch.int32)
    return PropagationTopology(
        source_index=torch.tensor([0, 1], device=device, dtype=torch.int32),
        sink_index=torch.zeros((rows,), device=device, dtype=torch.int32),
        source_id=torch.tensor([101, 303], device=device, dtype=torch.int64),
        sink_id=torch.tensor([707, 707], device=device, dtype=torch.int64),
        depth=torch.zeros((rows,), device=device, dtype=torch.int32),
        component_id=torch.zeros((rows,), device=device, dtype=torch.int32),
        primitive_id=torch.full((rows,), -1, device=device, dtype=torch.int32),
        edge_id=torch.full((rows,), -1, device=device, dtype=torch.int32),
        material_id=torch.full((rows,), -1, device=device, dtype=torch.int32),
        primitive_sequence=empty_sequence,
        material_sequence=empty_sequence,
        interaction_type=empty_sequence,
    )


def test_evaluate_consumes_request_batch_and_aliases_finalized_rows(
    monkeypatch,
) -> None:
    from witwin.channel.propagation.consumer import (
        PropagationRequest,
        ScalarTransport,
        evaluate,
    )
    from witwin.channel.propagation.consumer import service
    from witwin.channel.propagation.enumerated import capacity
    from witwin.channel.propagation.enumerated import engine

    sources, sinks = _endpoints()
    compact = _compact_rows()
    calls: dict[str, object] = {}

    def fake_engine(scene, config, **kwargs):
        calls["scene"] = scene
        calls["config"] = config
        calls["endpoint_tensors"] = kwargs["endpoint_tensors"]
        calls["defer"] = kwargs["defer_capacity_terminal"]
        return compact.evaluated, SimpleNamespace(
            execution=SimpleNamespace(
                launch_count=3,
                candidate_count=2,
                visibility_rejection_count=0,
                ad_companion_launches=0,
                ad_tape_bytes=0,
            ),
            capacity_transaction=None,
        )

    monkeypatch.setattr(engine, "evaluate_enumerated_paths", fake_engine)
    monkeypatch.setattr(service, "_compact", lambda *args, **kwargs: compact)
    monkeypatch.setattr(
        capacity,
        "sanitize_enumerated_capacity_transaction",
        lambda evaluated, sidecars: (evaluated, sidecars),
    )
    request = PropagationRequest(
        sources=sources,
        sinks=sinks,
        reference_frequency_hz=77.0e9,
        components=frozenset({"los"}),
        max_depth=0,
        response="scalar_transport",
        topology_mode="discover",
        ad_mode="none",
    )

    result = evaluate(_compiled(), request)

    endpoint_tensors = calls["endpoint_tensors"]
    assert endpoint_tensors.tx_positions is sources.positions_m
    assert endpoint_tensors.tx_power is sources.powers_w
    assert endpoint_tensors.rx_positions is sinks.positions_m
    assert calls["scene"].transmitters == ()
    assert calls["scene"].receivers == ()
    assert calls["defer"] is True
    assert isinstance(result.paths.transport, ScalarTransport)
    assert result.paths.pair_index is compact.pair_index
    assert result.paths.pair_offsets is compact.pair_offsets
    assert result.paths.topology.source_index is compact.evaluated.topology.tx_id
    assert (
        result.paths.geometry.path_length_m is compact.evaluated.geometry.path_length_m
    )
    assert result.paths.transport.coefficient is compact.evaluated.fields.path_field
    assert result.diagnostics.discovery_launch_count == 3
    assert result.capabilities.components == frozenset(
        {"los", "reflection", "transmission", "diffraction"}
    )
    assert all(
        "scattering" not in components
        for _, components in result.capabilities.response_components
    )


def test_unsupported_response_fails_at_request_construction(monkeypatch) -> None:
    """An unsupported response is rejected before a request object exists."""

    from witwin.channel.propagation.consumer import PropagationRequest
    from witwin.channel.propagation.enumerated import engine

    sources, sinks = _endpoints()

    def forbidden(*args, **kwargs):
        raise AssertionError("engine must not run")

    monkeypatch.setattr(engine, "evaluate_enumerated_paths", forbidden)

    with pytest.raises(NotImplementedError, match="unsupported response"):
        PropagationRequest(
            sources=sources,
            sinks=sinks,
            reference_frequency_hz=77.0e9,
            components=frozenset({"los"}),
            max_depth=0,
            response="unsupported",
            topology_mode="discover",
            ad_mode="none",
        )


def test_scattering_fails_at_request_construction(monkeypatch) -> None:
    """Scattering is not a v1 consumer component and never reaches the engine."""

    from witwin.channel.propagation.consumer import PropagationRequest
    from witwin.channel.propagation.enumerated import engine

    sources, sinks = _endpoints()

    def forbidden(*args, **kwargs):
        raise AssertionError("engine must not run")

    monkeypatch.setattr(engine, "evaluate_enumerated_paths", forbidden)

    with pytest.raises(
        NotImplementedError,
        match=r"unsupported propagation components: \['scattering'\]",
    ):
        PropagationRequest(
            sources=sources,
            sinks=sinks,
            reference_frequency_hz=77.0e9,
            components=frozenset({"scattering"}),
            max_depth=1,
            response="scalar_transport",
            topology_mode="discover",
            ad_mode="none",
        )


def test_reevaluate_reuses_frozen_topology_without_discovery(
    monkeypatch,
) -> None:
    from witwin.channel.propagation.consumer import (
        FixedTopologyRequest,
        ScalarTransport,
        reevaluate,
    )
    from witwin.channel.propagation.consumer import _fixed_los
    from witwin.channel.propagation.fields.kernels import (
        autograd as field_autograd,
    )
    from witwin.channel.propagation.fields.kernels import (
        functional as field_functional,
    )

    sources, sinks = _endpoints()
    topology = _fixed_topology()
    pair_index = torch.tensor([0, 1], device="cuda", dtype=torch.int64)
    pair_offsets = torch.tensor([0, 1, 2], device="cuda", dtype=torch.int64)
    gathered = SimpleNamespace(
        source=sources.positions_m,
        target=sinks.positions_m.expand(2, 3),
        tx_power=sources.powers_w,
        tx_polarization=sources.polarizations,
        rx_polarization=sinks.polarizations.expand(2, 3),
        pair_index=pair_index,
        pair_offsets=pair_offsets,
        row_count=2,
        validation_d2h_copies=1,
        validation_d2h_bytes=4,
        validation_synchronizations=1,
    )
    coefficient = torch.randn((2,), device="cuda", dtype=torch.complex64)
    path_field = torch.randn((2,), device="cuda", dtype=torch.complex64)
    field_vector = torch.randn((2, 3), device="cuda", dtype=torch.complex64)
    path_length = torch.tensor([10.0, 8.0], device="cuda")
    delay = torch.tensor([1.0, 2.0], device="cuda")
    direction = torch.randn((2, 3), device="cuda")
    calls: dict[str, object] = {}

    def fake_gather(actual_topology, actual_sources, actual_sinks):
        calls["topology"] = actual_topology
        calls["sources"] = actual_sources
        calls["sinks"] = actual_sinks
        return gathered

    def fake_field(*args, **kwargs):
        calls["field_args"] = args
        calls["field_kwargs"] = kwargs
        return {
            "coefficient": coefficient,
            "path_field": path_field,
            "field_vector": field_vector,
            "path_length_m": path_length,
            "delay_s": delay,
            "direction": direction,
        }

    def forbidden(*args, **kwargs):
        raise AssertionError("AD field owner must not run")

    monkeypatch.setattr(_fixed_los, "fixed_los_gather", fake_gather)
    monkeypatch.setattr(field_functional, "field_free_space", fake_field)
    monkeypatch.setattr(field_autograd, "field_free_space_ad", forbidden)
    request = FixedTopologyRequest(
        sources=sources,
        sinks=sinks,
        reference_frequency_hz=77.0e9,
        topology=topology,
        response="scalar_transport",
        ad_mode="none",
    )

    result = reevaluate(_compiled(), request)

    assert calls["topology"] is topology
    assert calls["sources"] is sources
    assert calls["sinks"] is sinks
    assert calls["field_args"][0] is gathered.source
    assert calls["field_kwargs"]["frequency_hz"] == 77.0e9
    assert result.paths.topology is topology
    assert result.paths.pair_index is pair_index
    assert result.paths.pair_offsets is pair_offsets
    assert result.paths.geometry.path_length_m is path_length
    assert isinstance(result.paths.transport, ScalarTransport)
    assert result.paths.transport.coefficient is path_field
    assert result.diagnostics.discovery_launch_count == 0
    assert result.diagnostics.candidate_count == 0
    assert result.diagnostics.validation_d2h_copies == 1
    assert result.diagnostics.validation_d2h_bytes == 4
    assert result.diagnostics.validation_sync_count == 1


def test_unsupported_fixed_response_fails_at_request_construction(
    monkeypatch,
) -> None:
    """A response with no fixed-topology provider is rejected at construction.

    Contract version 2 gives every declared response a fixed-topology
    provider, so ``polarimetric_transport`` is no longer the example: the
    former limit was deliberately lifted by ADR-037. The enforcement point
    itself is unchanged and is exercised here with a response that is outside
    the vocabulary entirely, which is the only way the check can now fire.
    """

    from witwin.channel.propagation.consumer import (
        FixedTopologyRequest,
        capabilities,
    )
    from witwin.channel.propagation.consumer import _fixed_los

    sources, sinks = _endpoints()

    def forbidden(*args, **kwargs):
        raise AssertionError("fixed gather must not run")

    monkeypatch.setattr(_fixed_los, "fixed_los_gather", forbidden)
    assert "polarimetric_transport" in capabilities().fixed_topology_responses

    with pytest.raises(NotImplementedError, match="unsupported response"):
        FixedTopologyRequest(
            sources=sources,
            sinks=sinks,
            reference_frequency_hz=77.0e9,
            topology=_fixed_topology(),
            response="matrix_transport",
            ad_mode="none",
        )


def test_unsupported_fixed_component_fails_at_preparation() -> None:
    """A frozen component with no fixed-topology owner is rejected at freeze.

    ``fixed_topology_components`` used to be advisory: the real gate was the
    zero-width interaction check in the service layer. Preparation is now the
    enforcement point, and it names the capability field so a caller can
    discover the supported set without a failed solve.
    """

    from witwin.channel.propagation.consumer import (
        PropagationTopology,
        prepare_fixed_topology,
    )

    device = torch.device("cuda")
    sequence = torch.tensor([[4]], device=device, dtype=torch.int32)
    diffraction = PropagationTopology(
        source_index=torch.zeros((1,), device=device, dtype=torch.int32),
        sink_index=torch.zeros((1,), device=device, dtype=torch.int32),
        source_id=torch.tensor([101], device=device, dtype=torch.int64),
        sink_id=torch.tensor([707], device=device, dtype=torch.int64),
        depth=torch.ones((1,), device=device, dtype=torch.int32),
        component_id=torch.full((1,), 2, device=device, dtype=torch.int32),
        primitive_id=torch.full((1,), -1, device=device, dtype=torch.int32),
        edge_id=torch.zeros((1,), device=device, dtype=torch.int32),
        material_id=torch.full((1,), -1, device=device, dtype=torch.int32),
        primitive_sequence=sequence,
        material_sequence=sequence,
        interaction_type=sequence,
    )

    with pytest.raises(
        NotImplementedError, match="supported components are .*reflection"
    ):
        prepare_fixed_topology(diffraction)


def test_fixed_primal_only_endpoint_ad_fails_before_gather(monkeypatch) -> None:
    from witwin.channel.propagation.consumer import (
        EndpointBatch,
        FixedTopologyRequest,
        reevaluate,
    )
    from witwin.channel.propagation.consumer import _fixed_los

    sources, sinks = _endpoints()
    sources = EndpointBatch(
        stable_ids=sources.stable_ids,
        positions_m=sources.positions_m,
        polarizations=sources.polarizations,
        powers_w=sources.powers_w.detach().requires_grad_(),
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("fixed gather must not run")

    monkeypatch.setattr(_fixed_los, "fixed_los_gather", forbidden)
    request = FixedTopologyRequest(
        sources=sources,
        sinks=sinks,
        reference_frequency_hz=77.0e9,
        topology=_fixed_topology(),
        response="scalar_transport",
        ad_mode="vjp",
    )

    with pytest.raises(NotImplementedError, match="sources.powers_w is primal-only"):
        reevaluate(_compiled(), request)
