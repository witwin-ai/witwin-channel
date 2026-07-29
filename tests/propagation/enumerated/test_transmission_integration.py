# Copyright Xingyu Chen.
# Tests transmission integration.

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from witwin.channel.interactions import transmission
from witwin.channel.propagation import enumerated as engine
from witwin.channel.propagation.penetration import (
    SegmentPenetrationPolicy,
)
from witwin.channel.propagation.topology import (
    EvaluatedPathSidecars,
    PathExecutionStats,
)
from witwin.channel.runtime import CapacityFailureState
from witwin.channel.scene.endpoints import SolverScene


def _unchecked_failure_state() -> CapacityFailureState:
    state = object.__new__(CapacityFailureState)
    object.__setattr__(state, "bits", torch.zeros(1, dtype=torch.int32))
    return state


def _fixture() -> tuple[object, object, torch.Tensor, torch.Tensor]:
    scene = SimpleNamespace(structures=[object()])
    compiled = SimpleNamespace(
        rayd=SimpleNamespace(available=True),
        enumerated_penetration_scene_diagonal_m=37.5,
        assignments=SimpleNamespace(
            face_material_id=torch.tensor([2, 3], dtype=torch.int64)
        ),
        materials=SimpleNamespace(
            geometry_mode_id=torch.tensor([0, 1, 0, 0], dtype=torch.int64)
        ),
    )
    tx = torch.tensor([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    rx = torch.tensor([[0.0, 4.0, 0.0], [0.0, 5.0, 0.0]])
    return scene, compiled, tx, rx


def test_primal_route_is_one_pair_major_batch_with_shared_state(monkeypatch) -> None:
    scene, compiled, tx, rx = _fixture()
    failure_state = _unchecked_failure_state()
    penetration = object()
    execution = object()
    block = {"valid": torch.tensor([True, False, True, False])}
    calls: list[tuple[object, ...]] = []

    def fake_forward(rayd, origins, targets, active, **kwargs):
        calls.append(("forward", rayd, origins, targets, active, kwargs))
        return penetration

    def fake_pack(actual, face_material_id, geometry_mode_id, **kwargs):
        calls.append(
            (
                "pack",
                actual,
                face_material_id,
                geometry_mode_id,
                kwargs,
            )
        )
        return SimpleNamespace(
            as_block=lambda: block,
            execution=execution,
            valid=block["valid"],
        )

    monkeypatch.setattr(transmission, "rayd_segment_penetration_forward", fake_forward)
    monkeypatch.setattr(
        transmission, "enumerated_transmission_topology_pack", fake_pack
    )
    monkeypatch.setattr(transmission, "_ensure_topology_fields", lambda block: block)

    actual_block, launches, actual_execution = transmission._transmission_topology(
        scene,
        compiled,
        tx,
        rx,
        max_depth=3,
        ad_mode="none",
        failure_state=failure_state,
    )

    assert actual_block["valid"].tolist() == [True, True]
    assert launches == 1
    assert actual_execution is execution
    assert [call[0] for call in calls] == ["forward", "pack"]
    _, rayd, origins, targets, active, kwargs = calls[0]
    assert rayd is compiled.rayd
    assert active is None
    assert origins.tolist() == [
        [1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
    ]
    assert targets.tolist() == [
        [0.0, 4.0, 0.0],
        [0.0, 5.0, 0.0],
        [0.0, 4.0, 0.0],
        [0.0, 5.0, 0.0],
    ]
    assert origins.is_contiguous()
    assert targets.is_contiguous()
    assert kwargs == {
        "input_active_any": True,
        "hit_capacity": 3,
        "policy": SegmentPenetrationPolicy.EnumeratedFullDistance,
        "scene_diagonal": 37.5,
        "failure_state": failure_state,
    }
    _, actual, face_material_id, geometry_mode_id, pack_kwargs = calls[1]
    assert actual is penetration
    assert face_material_id.dtype == torch.int32
    assert geometry_mode_id.dtype == torch.int32
    assert pack_kwargs == {"tx_count": 2, "rx_count": 2}


@pytest.mark.parametrize("ad_mode", ["jvp", "vjp"])
def test_ad_route_uses_native_tape_companions_and_live_geometry(monkeypatch, ad_mode) -> None:
    scene, compiled, tx, rx = _fixture()
    failure_state = _unchecked_failure_state()
    vertices = torch.arange(9, dtype=torch.float32).reshape(3, 3)
    tx_ad = tx + 10.0
    rx_ad = rx + 20.0
    observed: list[tuple[object, ...]] = []

    monkeypatch.setattr(
        transmission.scene_endpoints,
        "scene_vertex_table",
        lambda actual_scene, actual_compiled: vertices,
    )
    monkeypatch.setattr(
        transmission.scene_endpoints,
        "transmitter_positions_ad",
        lambda actual_scene, actual_tx, *, device: tx_ad,
    )
    monkeypatch.setattr(
        transmission.scene_endpoints,
        "receiver_positions_ad",
        lambda actual_scene, actual_rx, *, device: rx_ad,
    )

    def fake_ad(rayd, actual_vertices, origins, targets, active, **kwargs):
        observed.append((rayd, actual_vertices, origins, targets, active, kwargs))
        return object()

    monkeypatch.setattr(transmission, "rayd_segment_penetration_ad", fake_ad)
    monkeypatch.setattr(transmission, "_ensure_topology_fields", lambda block: block)
    monkeypatch.setattr(
        transmission,
        "enumerated_transmission_topology_pack",
        lambda *args, **kwargs: SimpleNamespace(
            as_block=lambda: {"valid": torch.zeros(4, dtype=torch.bool)},
            execution=object(),
            valid=torch.zeros(4, dtype=torch.bool),
        ),
    )

    transmission._transmission_topology(
        scene,
        compiled,
        tx,
        rx,
        max_depth=2,
        ad_mode=ad_mode,
        failure_state=failure_state,
    )

    assert len(observed) == 1
    rayd, actual_vertices, origins, targets, active, kwargs = observed[0]
    assert rayd is compiled.rayd
    assert actual_vertices is vertices
    assert active is None
    assert origins.tolist() == [tx_ad[0].tolist()] * 2 + [tx_ad[1].tolist()] * 2
    assert targets.tolist() == [rx_ad[0].tolist(), rx_ad[1].tolist()] * 2
    assert kwargs["failure_state"] is failure_state
    assert kwargs["policy"] is SegmentPenetrationPolicy.EnumeratedFullDistance


def test_invalid_poison_capacity_tail_is_compacted_before_engine_fields(monkeypatch) -> None:
    scene, compiled, tx, rx = _fixture()
    failure_state = _unchecked_failure_state()
    valid = torch.tensor([True, False, True, False])
    capacity_block = {
        "valid": valid,
        "tx_id": torch.tensor([0, 2_000_000_000, 1, 2_000_000_000]),
        "path_gain": torch.tensor([1.0, float("nan"), 2.0, float("nan")]),
    }
    topology = SimpleNamespace(
        valid=valid,
        execution=object(),
        as_block=lambda: capacity_block,
    )
    monkeypatch.setattr(
        transmission,
        "rayd_segment_penetration_forward",
        lambda *args, **kwargs: object(),
    )
    monkeypatch.setattr(
        transmission,
        "enumerated_transmission_topology_pack",
        lambda *args, **kwargs: topology,
    )
    monkeypatch.setattr(transmission, "_ensure_topology_fields", lambda block: block)

    block, launches, _execution = transmission._transmission_topology(
        scene,
        compiled,
        tx,
        rx,
        max_depth=1,
        ad_mode="none",
        failure_state=failure_state,
    )

    assert launches == 1
    assert block["valid"].tolist() == [True, True]
    assert block["tx_id"].tolist() == [0, 1]
    assert block["path_gain"].tolist() == [1.0, 2.0]


def test_structural_no_work_never_requires_or_observes_failure_state(monkeypatch) -> None:
    _scene, compiled, tx, rx = _fixture()
    scene = SimpleNamespace(structures=[])
    monkeypatch.setattr(transmission, "_ensure_topology_fields", lambda block: block)

    block, launches, execution = transmission._transmission_topology(
        scene,
        compiled,
        tx,
        rx,
        max_depth=2,
        ad_mode="none",
        failure_state=None,
    )

    assert launches == 0
    assert execution is None
    assert block["valid"].numel() == 0


def test_engine_observes_shared_failure_once_after_field_sanitization(monkeypatch) -> None:
    events: list[str] = []
    failure_state = object()
    capacity_execution = SimpleNamespace(candidate_capacity=1)
    evaluated_before = object()
    evaluated_after = object()
    execution_before = PathExecutionStats(
        launch_count=0,
        visibility_rejection_count=0,
        selected_edge_count=0,
        candidate_count=0,
        guardrail_count=0,
        ad_companion_launches=0,
        ad_tape_bytes=0,
    )
    execution_after = PathExecutionStats(
        launch_count=2,
        visibility_rejection_count=0,
        selected_edge_count=0,
        candidate_count=1,
        guardrail_count=0,
        ad_companion_launches=0,
        ad_tape_bytes=0,
    )
    vector_sidecar = torch.zeros((1, 1, 3), dtype=torch.complex64)
    sidecars = EvaluatedPathSidecars(
        execution=execution_before,
        diffraction_vector_field=vector_sidecar,
    )
    tx = torch.tensor([[0.0, 0.0, 0.0]])
    rx = torch.tensor([[1.0, 0.0, 0.0]])
    scene = SolverScene(
        compiled=object(),  # type: ignore[arg-type]
        structures=(object(),),
        transmitters=(object(),),  # type: ignore[arg-type]
        receivers=(),
        frequency=1.0,
        metadata={},
    )
    config = SimpleNamespace(
        components={"transmission"},
        max_depth=1,
        max_diffraction_order=0,
        coupled_paths=False,
        max_paths=None,
        max_paths_scope="global",
        ad_mode="none",
        isb_boundary_taper=False,
        isb_boundary_taper_width=0.5,
    )

    transaction = SimpleNamespace(
        failure_state=failure_state,
        terminal_check=lambda: events.append("terminal"),
    )
    monkeypatch.setattr(
        engine,
        "transmitter_tensors",
        lambda actual_scene, *, device: (tx, torch.ones(1)),
    )
    monkeypatch.setattr(
        engine,
        "transmitter_polarizations_as_stored",
        lambda actual_scene, *, device: torch.zeros((1, 3)),
    )
    monkeypatch.setattr(
        engine,
        "receiver_positions_and_layout",
        lambda actual_scene, *, device: (rx, object()),
    )
    monkeypatch.setattr(
        engine,
        "create_solve_capacity_transaction",
        lambda reference: (events.append("transaction"), transaction)[1],
    )

    def fake_transmission(*args, **kwargs):
        events.append("penetration")
        assert kwargs["failure_state"] is failure_state
        return (
            {
                "valid": torch.ones(1, dtype=torch.bool),
                "edge_id": torch.full((1,), -1, dtype=torch.int32),
                "primitive_sequence": torch.zeros((1, 1), dtype=torch.int32),
            },
            1,
            capacity_execution,
        )

    monkeypatch.setattr(engine, "_transmission_topology", fake_transmission)
    monkeypatch.setattr(
        engine,
        "concatenate_path_blocks",
        lambda blocks, *, device: (events.append("concatenate"), blocks[0])[1],
    )
    monkeypatch.setattr(
        engine.topology_kernels,
        "deterministic_selected_edge_count",
        lambda edge_id: 0,
    )
    monkeypatch.setattr(
        engine,
        "evaluated_paths_from_block",
        lambda *args, **kwargs: (evaluated_before, sidecars),
    )

    def fake_fields(*args, **kwargs):
        events.append("fields")
        return evaluated_after, execution_after

    monkeypatch.setattr(engine, "evaluate_path_fields", fake_fields)

    def fake_sanitize(actual, actual_sidecars):
        events.append("sanitize")
        assert actual is evaluated_after
        assert actual_sidecars.capacity_transaction is transaction
        return actual, actual_sidecars

    monkeypatch.setattr(
        engine,
        "sanitize_enumerated_capacity_transaction",
        fake_sanitize,
    )

    evaluated, actual_sidecars = engine.evaluate_enumerated_paths(
        scene,
        config,
        frequency_value=1.0,
    )

    assert evaluated is evaluated_after
    assert actual_sidecars.execution is execution_after
    assert actual_sidecars.capacity_execution is capacity_execution
    assert actual_sidecars.capacity_transaction is None
    assert events == [
        "transaction",
        "penetration",
        "concatenate",
        "fields",
        "sanitize",
        "terminal",
    ]

    events.clear()
    deferred_evaluated, deferred_sidecars = engine.evaluate_enumerated_paths(
        scene,
        config,
        frequency_value=1.0,
        defer_capacity_terminal=True,
    )
    assert deferred_evaluated is evaluated_after
    assert deferred_sidecars.capacity_transaction is transaction
    assert events == [
        "transaction",
        "penetration",
        "concatenate",
        "fields",
    ]