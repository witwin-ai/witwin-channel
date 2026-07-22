from __future__ import annotations

import torch

from witwin.channel.path import pipeline as path_pipeline
from witwin.channel.path import solver as path_solver
from witwin.channel.path.config import Config
from witwin.channel.path.result import InteractionType, from_evaluated_paths
from witwin.channel.propagation.models.evaluated import EvaluatedPaths
from witwin.channel.propagation.models.fields import PathFields
from witwin.channel.propagation.models.geometry import PathGeometry
from witwin.channel.propagation.models.topology import PathTopology
from witwin.channel.propagation.topology.export import (
    EvaluatedPathSidecars,
    PathExecutionStats,
)


def _evaluated_paths_fixture() -> tuple[EvaluatedPaths, EvaluatedPathSidecars]:
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
    topology = PathTopology(
        valid=torch.ones(rows, dtype=torch.bool),
        tx_id=torch.zeros(rows, dtype=torch.int32),
        rx_id=torch.zeros(rows, dtype=torch.int32),
        depth=depth,
        component_id=component_id,
        primitive_id=primitive_sequence[:, 0].clone(),
        edge_id=torch.tensor([-1, -1, 42, -1, 43, 44, -1], dtype=torch.int32),
        material_id=material_sequence[:, 0],
        primitive_sequence=primitive_sequence,
        material_sequence=material_sequence,
        interaction_type=interaction_type,
    )
    geometry = PathGeometry(
        row_identity=topology.row_identity,
        path_length_m=value * 3.0,
        delay_s=value * 1.0e-9,
        field_direction=torch.nn.functional.normalize(field_xyz.real, dim=-1),
        interaction_position=interaction_positions[:, 0],
        interaction_normal=interaction_normals[:, 0],
        interaction_positions=interaction_positions,
        interaction_normals=interaction_normals,
    )
    fields = PathFields(
        row_identity=topology.row_identity,
        path_gain=value.square(),
        path_field=coefficient * 2.0,
        field_xyz=field_xyz,
        coefficient=coefficient,
    )
    evaluated = EvaluatedPaths(topology=topology, geometry=geometry, fields=fields)
    sidecars = EvaluatedPathSidecars(
        execution=PathExecutionStats(
            launch_count=91,
            visibility_rejection_count=92,
            selected_edge_count=93,
            candidate_count=94,
            guardrail_count=95,
            ad_companion_launches=96,
            ad_tape_bytes=97,
        ),
        diffraction_vector_field=field_xyz * 3.0,
    )
    return evaluated, sidecars


def _pack_evaluated(paths: EvaluatedPaths):
    return from_evaluated_paths(
        paths,
        num_rx=1,
        num_tx=1,
        tx_positions=torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32),
        rx_positions=torch.tensor([[30.0, 20.0, 10.0]], dtype=torch.float32),
        metadata={"fixture": "mixed-components"},
    )


def test_canonical_packer_preserves_component_rows_and_metadata_exactly():
    paths, _ = _evaluated_paths_fixture()
    topology = paths.topology
    geometry = paths.geometry
    fields = paths.fields

    result = _pack_evaluated(paths)

    row = (0, 0, 0, 0)
    assert result.num_paths[row] == 7
    torch.testing.assert_close(result.a[row].squeeze(-1), fields.coefficient)
    torch.testing.assert_close(result.tau[row], geometry.delay_s)
    torch.testing.assert_close(result.interaction_type[row], topology.interaction_type)
    torch.testing.assert_close(result.material_id[row], topology.material_sequence)
    torch.testing.assert_close(result.position[row], geometry.interaction_positions)
    torch.testing.assert_close(result.field_xyz[row], fields.field_xyz)
    torch.testing.assert_close(result.field_direction[row], geometry.field_direction)
    assert result.primitive_id[row][2, 0] == topology.edge_id[2]
    assert result.primitive_id[row][4, 0] == topology.primitive_sequence[4, 0]
    assert result.normal[row][2, 0].equal(torch.zeros(3))
    assert result.normal[row][4, 1].equal(torch.zeros(3))
    assert result.normal[row][5, 0].equal(torch.zeros(3))
    assert result.metadata["fixture"] == "mixed-components"
    assert result.metadata["interaction_geometry"] == "canonical_topology"
    assert result.metadata["coefficient_semantics"].startswith("unit_excitation")


def test_solver_passes_typed_rows_and_only_execution_ad_sidecars(monkeypatch):
    initial, initial_sidecars = _evaluated_paths_fixture()
    appended, appended_sidecars = _evaluated_paths_fixture()
    sentinel = object()
    calls: list[str] = []
    captured_metadata: dict[str, object] = {}

    monkeypatch.setattr(
        path_solver, "_validate_runtime", lambda _config: (True, True, True)
    )

    def fake_engine(_scene, _config, *, defer_capacity_terminal):
        assert defer_capacity_terminal is True
        calls.append("engine")
        return initial, initial_sidecars

    def fake_append(_scene, _config, evaluated, sidecars):
        assert evaluated is initial
        assert sidecars is initial_sidecars
        calls.append("append")
        return appended, appended_sidecars, {"path_count": appended.row_count}

    def fake_metadata(**kwargs):
        calls.append("metadata")
        captured_metadata.update(kwargs)
        return {"kernel": {"launch_count": 1}}

    def fake_sanitize(evaluated, sidecars):
        calls.append("sanitize")
        assert evaluated is appended
        assert sidecars is appended_sidecars
        return evaluated, sidecars

    def fake_compact(evaluated):
        calls.append("compact")
        assert evaluated is appended
        return evaluated

    def fake_pack(paths, **kwargs):
        calls.append("pack")
        assert paths is appended
        assert not hasattr(paths, "diffraction_vector_field")
        assert kwargs["metadata"]["kernel"]["launch_count"] == 1
        return sentinel

    monkeypatch.setattr(path_solver, "evaluate_enumerated_paths", fake_engine)
    monkeypatch.setattr(path_solver, "append_scattering_evaluated_paths", fake_append)
    monkeypatch.setattr(
        path_pipeline, "sanitize_enumerated_capacity_transaction", fake_sanitize
    )
    monkeypatch.setattr(
        path_pipeline,
        "_compact_valid_evaluated_paths_for_legacy_result",
        fake_compact,
    )
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

    assert result.result is sentinel
    assert result.capacity_transaction is None
    assert calls == [
        "engine",
        "append",
        "sanitize",
        "compact",
        "metadata",
        "pack",
    ]
    assert captured_metadata["path_count"] == appended.row_count
    assert captured_metadata["ad_companion_launches"] == 96
    assert captured_metadata["ad_tape_bytes"] == 97
    assert captured_metadata["transmission_path_count"] == 1
    assert captured_metadata["scattering_path_count"] == 1
    assert "launch_count" not in captured_metadata
    assert "diffraction_vector_field" not in captured_metadata
