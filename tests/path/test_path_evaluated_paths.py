from __future__ import annotations

from dataclasses import fields

import torch

from witwin.channel_native.core.path_topology import TopologyBatch
from witwin.channel_native.path.config import Config
from witwin.channel_native.path.result import (
    InteractionType,
    PathResult,
    from_evaluated_paths,
    from_topology_result,
)
from witwin.channel_native.path import solver as path_solver
from witwin.channel_native.propagation.models.adapters import (
    evaluated_paths_from_topology_batch,
)
from witwin.channel_native.propagation.models.evaluated import EvaluatedPaths


def _mixed_topology_batch() -> TopologyBatch:
    component_id = torch.tensor([0, 1, 2, 5, 3, 4, 6], dtype=torch.int32)
    depth = torch.tensor([0, 1, 1, 1, 2, 2, 1], dtype=torch.int32)
    rows = int(component_id.numel())
    width = 2
    value = torch.arange(1, rows + 1, dtype=torch.float32)
    interaction_type = torch.tensor(
        [
            [InteractionType.NONE, InteractionType.NONE],
            [InteractionType.REFLECTION, InteractionType.NONE],
            [InteractionType.DIFFRACTION, InteractionType.NONE],
            [InteractionType.TRANSMISSION, InteractionType.NONE],
            [InteractionType.REFLECTION, InteractionType.DIFFRACTION],
            [InteractionType.DIFFRACTION, InteractionType.REFLECTION],
            [InteractionType.SCATTERING, InteractionType.NONE],
        ],
        dtype=torch.int32,
    )
    primitive_sequence = torch.tensor(
        [[-1, -1], [11, -1], [-1, -1], [13, -1], [14, 15], [16, 17], [18, -1]],
        dtype=torch.int32,
    )
    material_sequence = torch.arange(rows * width, dtype=torch.int32).reshape(
        rows, width
    )
    interaction_positions = torch.stack(
        (
            torch.stack((value, value + 10.0, value + 20.0), dim=-1),
            torch.stack((value + 1.0, value + 11.0, value + 21.0), dim=-1),
        ),
        dim=1,
    )
    interaction_normals = torch.zeros((rows, width, 3), dtype=torch.float32)
    interaction_normals[..., 2] = 1.0
    interaction_normals[2, 0] = float("nan")
    interaction_normals[4, 1] = float("nan")
    interaction_normals[5, 0] = float("nan")
    field_xyz = torch.complex(
        torch.stack((value, value + 20.0, value + 40.0), dim=-1),
        torch.zeros((rows, 3), dtype=torch.float32),
    )
    coefficient = torch.complex(value, -value)
    return TopologyBatch(
        valid=torch.ones(rows, dtype=torch.bool),
        tx_id=torch.zeros(rows, dtype=torch.int32),
        rx_id=torch.zeros(rows, dtype=torch.int32),
        depth=depth,
        component_id=component_id,
        primitive_id=primitive_sequence[:, 0].clone(),
        edge_id=torch.tensor([-1, -1, 42, -1, 43, 44, -1], dtype=torch.int32),
        path_length_m=value * 3.0,
        delay_s=value * 1.0e-9,
        path_gain=value.square(),
        path_field=coefficient * 2.0,
        field_xyz=field_xyz,
        coefficient=coefficient,
        field_direction=torch.nn.functional.normalize(field_xyz.real, dim=-1),
        interaction_position=interaction_positions[:, 0],
        interaction_normal=interaction_normals[:, 0],
        material_id=material_sequence[:, 0],
        primitive_sequence=primitive_sequence,
        material_sequence=material_sequence,
        interaction_type=interaction_type,
        interaction_positions=interaction_positions,
        interaction_normals=interaction_normals,
        launch_count=91,
        visibility_rejection_count=92,
        selected_edge_count=93,
        candidate_count=94,
        guardrail_count=95,
        diffraction_vector_field=field_xyz * 3.0,
        ad_companion_launches=96,
        ad_tape_bytes=97,
    )


def _pack_evaluated(batch: TopologyBatch) -> PathResult:
    evaluated, _ = evaluated_paths_from_topology_batch(batch)
    return from_evaluated_paths(
        evaluated,
        num_rx=1,
        num_tx=1,
        tx_positions=torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32),
        rx_positions=torch.tensor([[30.0, 20.0, 10.0]], dtype=torch.float32),
        metadata={"fixture": "mixed-components"},
    )


def _assert_path_results_exact(observed: PathResult, expected: PathResult) -> None:
    for item in fields(PathResult):
        observed_value = getattr(observed, item.name)
        expected_value = getattr(expected, item.name)
        if isinstance(expected_value, torch.Tensor):
            torch.testing.assert_close(
                observed_value,
                expected_value,
                rtol=0.0,
                atol=0.0,
                equal_nan=True,
            )
        else:
            assert observed_value == expected_value


def test_canonical_packer_preserves_mixed_component_rows_and_metadata_exactly():
    batch = _mixed_topology_batch()

    result = _pack_evaluated(batch)

    row = (0, 0, 0, 0)
    assert result.num_paths[row] == 7
    torch.testing.assert_close(result.a[row].squeeze(-1), batch.coefficient)
    torch.testing.assert_close(result.tau[row], batch.delay_s)
    torch.testing.assert_close(result.interaction_type[row], batch.interaction_type)
    torch.testing.assert_close(result.material_id[row], batch.material_sequence)
    torch.testing.assert_close(result.position[row], batch.interaction_positions)
    torch.testing.assert_close(result.field_xyz[row], batch.field_xyz)
    torch.testing.assert_close(result.field_direction[row], batch.field_direction)
    assert result.primitive_id[row][2, 0] == batch.edge_id[2]
    assert result.primitive_id[row][4, 0] == batch.primitive_sequence[4, 0]
    assert result.normal[row][2, 0].equal(torch.zeros(3))
    assert result.normal[row][4, 1].equal(torch.zeros(3))
    assert result.normal[row][5, 0].equal(torch.zeros(3))
    assert result.metadata["fixture"] == "mixed-components"
    assert result.metadata["interaction_geometry"] == "canonical_topology"
    assert result.metadata["coefficient_semantics"].startswith("unit_excitation")


def test_legacy_topology_wrapper_is_bit_exact_with_canonical_packer():
    batch = _mixed_topology_batch()
    expected = _pack_evaluated(batch)

    observed = from_topology_result(
        batch,
        num_rx=1,
        num_tx=1,
        tx_positions=torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32),
        rx_positions=torch.tensor([[30.0, 20.0, 10.0]], dtype=torch.float32),
        metadata={"fixture": "mixed-components"},
    )

    _assert_path_results_exact(observed, expected)


def test_solver_adapts_after_scattering_and_passes_only_execution_ad_sidecars(
    monkeypatch,
):
    initial = _mixed_topology_batch()
    appended = _mixed_topology_batch()
    sentinel = object()
    calls: list[str] = []
    captured_metadata: dict[str, object] = {}

    monkeypatch.setattr(
        path_solver, "_validate_runtime", lambda _config: (True, True, True)
    )

    def fake_export(_scene, _config):
        calls.append("export")
        return initial

    def fake_append(_scene, _config, topology):
        assert topology is initial
        calls.append("append")
        return appended, {"path_count": appended.valid.numel()}

    def fake_metadata(**kwargs):
        calls.append("metadata")
        captured_metadata.update(kwargs)
        return {"kernel": {"launch_count": 1}}

    def fake_pack(paths, **kwargs):
        calls.append("pack")
        assert isinstance(paths, EvaluatedPaths)
        assert paths.topology.valid is appended.valid
        assert not hasattr(paths, "diffraction_vector_field")
        assert kwargs["metadata"]["kernel"]["launch_count"] == 1
        return sentinel

    monkeypatch.setattr(path_solver, "export_topology", fake_export)
    monkeypatch.setattr(path_solver, "append_scattering_paths", fake_append)
    monkeypatch.setattr(path_solver, "_metadata", fake_metadata)
    monkeypatch.setattr(path_solver, "from_evaluated_paths", fake_pack)
    monkeypatch.setattr(
        path_solver,
        "_transmitter_tensors",
        lambda _scene: (torch.zeros((1, 3)), torch.ones(1)),
    )
    monkeypatch.setattr(
        path_solver,
        "_receiver_positions",
        lambda _scene, *, reference: torch.zeros_like(reference),
    )

    result = path_solver._solve_base(object(), Config(components={"los", "scattering"}))

    assert result is sentinel
    assert calls == ["export", "append", "metadata", "pack"]
    assert captured_metadata["path_count"] == appended.valid.numel()
    assert captured_metadata["ad_companion_launches"] == 96
    assert captured_metadata["ad_tape_bytes"] == 97
    assert captured_metadata["transmission_path_count"] == 1
    assert captured_metadata["scattering_path_count"] == 1
    assert "launch_count" not in captured_metadata
    assert "diffraction_vector_field" not in captured_metadata
