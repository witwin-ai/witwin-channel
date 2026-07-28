from __future__ import annotations

from types import SimpleNamespace

import torch

from tests.path.test_path_evaluated_paths import _evaluated_paths_fixture
from witwin.channel.propagation import fields as evaluation
from witwin.channel.propagation.rows import (
    EvaluatedPaths,
    PathFields,
    PathGeometry,
    PathTopology,
)


def _patch_field_stages(monkeypatch, events: list[str]) -> None:
    material_tokens: list[object] = []

    class _Ledger:
        launches = 5
        tape_bytes = 7

    def fake_los(*args):
        events.append("los")
        for value, tensor in enumerate(args[8:15], start=1):
            tensor.add_(value)
        # ``launch_count`` precedes the compiled/taper metadata arguments.
        return args[15] + 1

    def fake_reflection(*args):
        events.append("reflection")
        assert args[15] is None
        material = object()
        material_tokens.append(material)
        return material, args[-1] + 1

    def fake_transmission(*args):
        events.append("transmission")
        assert args[13] is material_tokens[-1]
        return args[13], args[-1] + 1

    def fake_diffraction(*args):
        events.append("diffraction")
        assert args[16] is material_tokens[-1]
        return args[16], args[-1] + 1

    def fake_coupled(*args):
        events.append("coupled")
        assert args[15] is material_tokens[-1]
        return args[15], args[-1] + 1

    monkeypatch.setattr(evaluation, "AdLaunchLedger", _Ledger)
    monkeypatch.setattr(evaluation, "_geometry_participates_in_ad", lambda _scene: False)
    monkeypatch.setattr(evaluation, "_vertices_participate_in_ad", lambda _scene: False)
    monkeypatch.setattr(
        evaluation,
        "transmitter_polarizations",
        lambda _scene, *, device: torch.ones((12, 3), device=device),
    )
    monkeypatch.setattr(
        evaluation,
        "receiver_polarizations",
        lambda _scene, *, device: torch.ones((12, 3), device=device),
    )
    monkeypatch.setattr(evaluation, "_evaluate_los_fields", fake_los)
    monkeypatch.setattr(evaluation, "_evaluate_reflection_fields", fake_reflection)
    monkeypatch.setattr(evaluation, "_evaluate_transmission_fields", fake_transmission)
    monkeypatch.setattr(evaluation, "_evaluate_diffraction_fields", fake_diffraction)
    monkeypatch.setattr(evaluation, "_evaluate_coupled_fields", fake_coupled)


def _assert_exact_tensor(observed: torch.Tensor, expected: torch.Tensor) -> None:
    assert observed is expected
    assert observed.untyped_storage()._cdata == expected.untyped_storage()._cdata


def _empty_paths(paths: EvaluatedPaths) -> EvaluatedPaths:
    old_topology = paths.topology
    topology = PathTopology(
        valid=old_topology.valid[:0],
        tx_id=old_topology.tx_id[:0],
        rx_id=old_topology.rx_id[:0],
        depth=old_topology.depth[:0],
        component_id=old_topology.component_id[:0],
        primitive_id=old_topology.primitive_id[:0],
        edge_id=old_topology.edge_id[:0],
        material_id=old_topology.material_id[:0],
        primitive_sequence=old_topology.primitive_sequence[:0],
        material_sequence=old_topology.material_sequence[:0],
        interaction_type=old_topology.interaction_type[:0],
    )
    old_geometry = paths.geometry
    geometry = PathGeometry(
        row_identity=topology.row_identity,
        path_length_m=old_geometry.path_length_m[:0],
        delay_s=old_geometry.delay_s[:0],
        field_direction=old_geometry.field_direction[:0],
        interaction_position=old_geometry.interaction_position[:0],
        interaction_normal=old_geometry.interaction_normal[:0],
        interaction_positions=old_geometry.interaction_positions[:0],
        interaction_normals=old_geometry.interaction_normals[:0],
    )
    old_fields = paths.fields
    fields = PathFields(
        row_identity=topology.row_identity,
        path_gain=old_fields.path_gain[:0],
        path_field=old_fields.path_field[:0],
        field_xyz=old_fields.field_xyz[:0],
        coefficient=old_fields.coefficient[:0],
    )
    return EvaluatedPaths(topology=topology, geometry=geometry, fields=fields)


def test_typed_field_seam_preserves_order_values_and_alias_contracts(monkeypatch):
    paths, sidecars = _evaluated_paths_fixture()
    scene = SimpleNamespace(frequency=3.0e9)
    compiled = object()
    tx_positions = torch.arange(36, dtype=torch.float32).reshape(12, 3)
    tx_power = torch.arange(1, 13, dtype=torch.float32)
    rx_positions = tx_positions + 1.0
    events: list[str] = []
    _patch_field_stages(monkeypatch, events)

    typed_result, execution = evaluation.evaluate_path_fields(
        scene,
        compiled,
        paths,
        sidecars.execution,
        tx_positions,
        tx_power,
        rx_positions,
        components={"los", "reflection", "transmission", "diffraction"},
        ad_mode="vjp",
        frequency_value=3.0e9,
    )
    assert events == ["los", "reflection", "transmission", "diffraction", "coupled"]
    assert execution.launch_count == 96
    assert execution.ad_companion_launches == 101
    assert execution.ad_tape_bytes == 104
    assert typed_result.topology is paths.topology

    for name in (
        "interaction_position",
        "interaction_normal",
        "interaction_positions",
        "interaction_normals",
    ):
        _assert_exact_tensor(
            getattr(typed_result.geometry, name),
            getattr(paths.geometry, name),
        )

    cloned = (
        (typed_result.fields.field_xyz, paths.fields.field_xyz),
        (typed_result.fields.coefficient, paths.fields.coefficient),
        (typed_result.fields.path_field, paths.fields.path_field),
        (typed_result.fields.path_gain, paths.fields.path_gain),
        (typed_result.geometry.path_length_m, paths.geometry.path_length_m),
        (typed_result.geometry.delay_s, paths.geometry.delay_s),
        (typed_result.geometry.field_direction, paths.geometry.field_direction),
    )
    for observed, original in cloned:
        assert observed is not original
        assert observed.untyped_storage()._cdata != original.untyped_storage()._cdata
        assert observed.shape == original.shape
        assert observed.dtype == original.dtype
        assert observed.device == original.device
        assert observed.requires_grad == original.requires_grad


def test_empty_typed_field_seam_preserves_input_identity():
    source, sidecars = _evaluated_paths_fixture()
    paths = _empty_paths(source)

    typed_result, execution = evaluation.evaluate_path_fields(
        object(),
        object(),
        paths,
        sidecars.execution,
        torch.empty((0, 3)),
        torch.empty((0,)),
        torch.empty((0, 3)),
    )

    assert typed_result is paths
    assert execution is sidecars.execution
