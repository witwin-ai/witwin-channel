from __future__ import annotations

from dataclasses import fields
from types import SimpleNamespace

import torch

from tests.propagation.test_topology_batch_adapter import (
    _GEOMETRY_FIELDS,
    _PATH_FIELDS,
    _TOPOLOGY_FIELDS,
    _assert_exact_tensor,
    _batch,
)
from witwin.channel_native.core.path_topology import TopologyBatch
from witwin.channel_native.propagation.fields import evaluation
from witwin.channel_native.propagation.topology.export import (
    export_evaluated_rows,
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
        return args[-1] + 1

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


def _typed_tensors(paths) -> dict[str, torch.Tensor]:
    return {
        name: getattr(contract, name)
        for contract, names in (
            (paths.topology, _TOPOLOGY_FIELDS),
            (paths.geometry, _GEOMETRY_FIELDS),
            (paths.fields, _PATH_FIELDS),
        )
        for name in names
    }


def test_typed_and_legacy_field_seams_are_exact_and_preserve_aliases(monkeypatch):
    source = _batch()
    scene = SimpleNamespace(frequency=3.0e9)
    compiled = object()
    tx_positions = torch.arange(36, dtype=torch.float32).reshape(12, 3)
    tx_power = torch.arange(1, 13, dtype=torch.float32)
    rx_positions = tx_positions + 1.0
    events: list[str] = []
    _patch_field_stages(monkeypatch, events)

    legacy_result = evaluation._evaluate_shared_fields(
        scene,
        compiled,
        source,
        tx_positions,
        tx_power,
        rx_positions,
        components={"los", "reflection", "transmission", "diffraction"},
        ad_mode="vjp",
        frequency_value=3.0e9,
    )
    assert events == [
        "los",
        "reflection",
        "transmission",
        "diffraction",
        "coupled",
    ]

    events.clear()
    paths, sidecars = export_evaluated_rows(source)
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
    assert events == [
        "los",
        "reflection",
        "transmission",
        "diffraction",
        "coupled",
    ]

    legacy_typed, legacy_sidecars = export_evaluated_rows(legacy_result)
    for name, tensor in _typed_tensors(typed_result).items():
        assert torch.equal(tensor, _typed_tensors(legacy_typed)[name])
    assert execution == legacy_sidecars.execution
    assert execution.launch_count == 16
    assert execution.ad_companion_launches == 21
    assert execution.ad_tape_bytes == 24

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

    changed_tensor_fields = {
        "path_length_m",
        "delay_s",
        "path_gain",
        "path_field",
        "field_xyz",
        "coefficient",
        "field_direction",
    }
    for item in fields(TopologyBatch):
        original = getattr(source, item.name)
        observed = getattr(legacy_result, item.name)
        if isinstance(original, torch.Tensor) and item.name not in changed_tensor_fields:
            _assert_exact_tensor(observed, original)


def test_empty_typed_and_legacy_field_seams_preserve_input_identity():
    source = _batch(rows=0, width=2)
    paths, sidecars = export_evaluated_rows(source)

    typed_result, execution = evaluation.evaluate_path_fields(
        object(),
        object(),
        paths,
        sidecars.execution,
        torch.empty((0, 3)),
        torch.empty((0,)),
        torch.empty((0, 3)),
    )
    legacy_result = evaluation._evaluate_shared_fields(
        object(),
        object(),
        source,
        torch.empty((0, 3)),
        torch.empty((0,)),
        torch.empty((0, 3)),
    )

    assert typed_result is paths
    assert execution is sidecars.execution
    assert legacy_result is source
