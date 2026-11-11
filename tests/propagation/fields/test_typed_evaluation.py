# Copyright Xingyu Chen.
# Tests typed evaluation.

from __future__ import annotations

from types import SimpleNamespace

import torch

from tests.path.test_path_evaluated_paths import _evaluated_paths_fixture
from witwin.channel.propagation import fields as evaluation
from witwin.channel.propagation.rows import (
    EvaluatedPaths,
    PathFields,
    PathGeometry,
    PathRowGroups,
    PathTopology,
)


_OUTPUT_NAMES = (
    "field_xyz",
    "coefficient",
    "path_field",
    "path_gain",
    "path_length",
    "delay",
    "direction",
)


def _outputs(args):
    return next(arg for arg in args if isinstance(arg, evaluation._FieldOutputs))


def _patch_field_stages(monkeypatch, events: list[str]) -> None:
    material_tokens: list[object] = []

    class _Ledger:
        launches = 5
        tape_bytes = 7

    def fake_los(*args):
        events.append("los")
        outputs = _outputs(args)
        rows = torch.tensor([0], dtype=torch.int64)
        for value, name in enumerate(_OUTPUT_NAMES, start=1):
            current = outputs.values[name]
            outputs.write(name, rows, current[rows] + value)
        launch_count = next(arg for arg in args if type(arg) is int)
        return launch_count + 1

    def fake_reflection(*args):
        events.append("reflection")
        material = object()
        material_tokens.append(material)
        return material, args[-1] + 1

    def threaded(name):
        def call(*args):
            events.append(name)
            assert any(arg is material_tokens[-1] for arg in args)
            return material_tokens[-1], args[-1] + 1

        return call

    fake_transmission = threaded("transmission")
    fake_diffraction = threaded("diffraction")
    fake_coupled = threaded("coupled")

    monkeypatch.setattr(evaluation, "AdLaunchLedger", _Ledger)
    monkeypatch.setattr(evaluation, "_geometry_participates_in_ad", lambda _scene: False)
    monkeypatch.setattr(evaluation, "_vertices_participate_in_ad", lambda _scene: False)
    monkeypatch.setattr(
        evaluation,
        "transmitter_polarizations_f32",
        lambda _scene, *, device: torch.ones((12, 3), device=device),
    )
    monkeypatch.setattr(
        evaluation,
        "receiver_polarization_rows",
        lambda _scene, *, device: torch.ones((12, 3), device=device),
    )
    monkeypatch.setattr(evaluation, "_evaluate_los_fields", fake_los)
    monkeypatch.setattr(evaluation, "_evaluate_reflection_fields", fake_reflection)
    monkeypatch.setattr(evaluation, "_evaluate_transmission_fields", fake_transmission)
    monkeypatch.setattr(evaluation, "_evaluate_diffraction_fields", fake_diffraction)
    monkeypatch.setattr(evaluation, "_evaluate_coupled_fields", fake_coupled)


def _groups(paths: EvaluatedPaths) -> PathRowGroups:
    return PathRowGroups.single(
        0, paths.row_count, sequence_width=int(paths.topology.primitive_sequence.shape[1])
    )


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
        row_groups=_groups(paths),
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
    # A partially written output is a clone carrying the written rows; the inputs are untouched.
    for value, (observed, original) in enumerate(cloned, start=1):
        assert observed is not original
        assert observed.untyped_storage()._cdata != original.untyped_storage()._cdata
        assert observed.shape == original.shape
        assert observed.dtype == original.dtype
        assert observed.device == original.device
        assert observed.requires_grad == original.requires_grad
        assert torch.equal(observed[0], original[0] + value)
        assert torch.equal(observed[1:], original[1:])


def test_unwritten_outputs_alias_the_inputs_and_full_writes_take_the_result(monkeypatch):
    paths, sidecars = _evaluated_paths_fixture()
    events: list[str] = []
    _patch_field_stages(monkeypatch, events)
    full = {}

    def fake_los(*args):
        outputs = _outputs(args)
        full["path_gain"] = outputs.values["path_gain"] * 2
        outputs.write("path_gain", None, full["path_gain"])
        return next(arg for arg in args if type(arg) is int)

    monkeypatch.setattr(evaluation, "_evaluate_los_fields", fake_los)
    typed_result, _ = evaluation.evaluate_path_fields(
        SimpleNamespace(frequency=3.0e9),
        object(),
        paths,
        sidecars.execution,
        torch.arange(36, dtype=torch.float32).reshape(12, 3),
        torch.arange(1, 13, dtype=torch.float32),
        torch.arange(36, dtype=torch.float32).reshape(12, 3),
        row_groups=_groups(paths),
        components={"los"},
        ad_mode="vjp",
        frequency_value=3.0e9,
    )
    assert typed_result.fields.path_gain is full["path_gain"]
    assert typed_result.fields.path_field is paths.fields.path_field
    assert typed_result.geometry.delay_s is paths.geometry.delay_s


def _patch_canonical_launch(monkeypatch, calls: list[dict[str, object]]) -> None:
    def fake_canonical_rows(**kwargs):
        calls.append(kwargs)
        return {name: value + 1 for name, value in kwargs["fields"].items()}

    monkeypatch.setattr(evaluation.field_kernels, "field_canonical_rows", fake_canonical_rows)
    monkeypatch.setattr(
        evaluation,
        "_canonical_face_tables",
        lambda _compiled, _groups, _components, _device: "faces",
    )


def test_primal_field_stage_is_one_canonical_launch_over_unsliced_tables(monkeypatch):
    paths, sidecars = _evaluated_paths_fixture()
    events: list[str] = []
    _patch_field_stages(monkeypatch, events)
    calls: list[dict[str, object]] = []
    _patch_canonical_launch(monkeypatch, calls)
    tx_positions = torch.arange(36, dtype=torch.float32).reshape(12, 3)
    rx_positions = tx_positions + 1.0
    typed_result, execution = evaluation.evaluate_path_fields(
        SimpleNamespace(frequency=3.0e9),
        object(),
        paths,
        sidecars.execution,
        tx_positions,
        torch.arange(1, 13, dtype=torch.float32),
        rx_positions,
        row_groups=_groups(paths),
        components={"los", "reflection", "diffraction"},
        frequency_value=3.0e9,
    )
    # Line of sight, reflection and diffraction are one launch; no per-group operator runs.
    assert events == []
    (call,) = calls
    assert execution.launch_count == sidecars.execution.launch_count + 1
    assert call["order"] is None and call["evaluate_los"] is True
    assert call["faces"] == "faces" and call["frequency_hz"] == 3.0e9
    assert torch.equal(call["tx_positions"], tx_positions)
    assert torch.equal(call["rx_positions"], rx_positions)
    assert call["primitive_sequence"] is paths.topology.primitive_sequence
    assert call["fields"]["path_gain"] is paths.fields.path_gain
    assert call["fields"]["delay"] is paths.geometry.delay_s
    assert torch.equal(typed_result.fields.path_gain, paths.fields.path_gain + 1)
    assert torch.equal(typed_result.geometry.field_direction, paths.geometry.field_direction + 1)
    for name in ("interaction_positions", "interaction_normals"):
        _assert_exact_tensor(getattr(typed_result.geometry, name), getattr(paths.geometry, name))


def test_primal_transmission_and_coupled_rows_write_into_the_canonical_columns(monkeypatch):
    paths, sidecars = _evaluated_paths_fixture()
    events: list[str] = []
    _patch_field_stages(monkeypatch, events)
    calls: list[dict[str, object]] = []
    _patch_canonical_launch(monkeypatch, calls)
    written: dict[str, torch.Tensor] = {}

    def fake_transmission(*args):
        events.append("transmission")
        outputs = _outputs(args)
        written["before"] = outputs.values["path_gain"]
        outputs.write("path_gain", torch.tensor([0], dtype=torch.int64), torch.tensor([-5.0]))
        return None, args[-1] + 1

    def fake_coupled(*args):
        events.append("coupled")
        return None, args[-1] + 1

    monkeypatch.setattr(evaluation, "_evaluate_transmission_fields", fake_transmission)
    monkeypatch.setattr(evaluation, "_evaluate_coupled_fields", fake_coupled)
    width = int(paths.topology.primitive_sequence.shape[1])
    counts = [0] * (8 + width + 2)
    counts[5] = paths.row_count
    typed_result, execution = evaluation.evaluate_path_fields(
        SimpleNamespace(frequency=3.0e9),
        object(),
        paths,
        sidecars.execution,
        torch.arange(36, dtype=torch.float32).reshape(12, 3),
        torch.arange(1, 13, dtype=torch.float32),
        torch.arange(36, dtype=torch.float32).reshape(12, 3),
        row_groups=PathRowGroups(order=None, counts=tuple(counts)),
        components={"transmission"},
        frequency_value=3.0e9,
    )
    assert events == ["transmission", "coupled"]
    assert len(calls) == 1
    assert execution.launch_count == sidecars.execution.launch_count + 3
    # The canonical result is owned by the stage, so the row write lands in place.
    assert typed_result.fields.path_gain is written["before"]
    assert float(typed_result.fields.path_gain[0]) == -5.0
    assert torch.equal(typed_result.fields.path_gain[1:], paths.fields.path_gain[1:] + 1)


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
        row_groups=_groups(paths),
    )

    assert typed_result is paths
    assert execution is sidecars.execution