from __future__ import annotations

import ast
from dataclasses import is_dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from witwin.channel_native.propagation import enumerated
from witwin.channel_native.propagation.enumerated import contracts, scattering
from witwin.channel_native.propagation.models.evaluated import EvaluatedPaths
from witwin.channel_native.propagation.models.fields import PathFields
from witwin.channel_native.propagation.models.geometry import PathGeometry
from witwin.channel_native.propagation.models.topology import PathTopology
from witwin.channel_native.propagation.topology.export import (
    EvaluatedPathSidecars,
    PathExecutionStats,
)


_EXTENDED_ROW_FIELDS = (
    "valid",
    "tx_id",
    "rx_id",
    "depth",
    "component_id",
    "primitive_id",
    "edge_id",
    "path_length_m",
    "delay_s",
    "path_gain",
    "path_field",
    "field_xyz",
    "coefficient",
    "field_direction",
    "interaction_position",
    "interaction_normal",
    "material_id",
    "primitive_sequence",
    "material_sequence",
    "interaction_type",
    "interaction_positions",
    "interaction_normals",
)


def _evaluated_paths(
    *, rows: int = 2, width: int = 2
) -> tuple[EvaluatedPaths, EvaluatedPathSidecars]:
    scalar = torch.arange(1, rows + 1, dtype=torch.float32)
    vector = torch.stack((scalar, scalar + 10.0, scalar + 20.0), dim=-1)
    complex_scalar = torch.complex(scalar, -scalar)
    complex_vector = torch.complex(vector, -vector)
    primitive_sequence = torch.full((rows, width), -1, dtype=torch.int32)
    material_sequence = torch.full((rows, width), -1, dtype=torch.int32)
    interaction_type = torch.zeros((rows, width), dtype=torch.int32)
    interaction_positions = torch.zeros((rows, width, 3), dtype=torch.float32)
    interaction_normals = torch.zeros((rows, width, 3), dtype=torch.float32)
    if width:
        primitive_sequence[:, 0] = torch.arange(rows, dtype=torch.int32) + 30
        material_sequence[:, 0] = torch.arange(rows, dtype=torch.int32) + 40
        interaction_type[:, 0] = 1
        interaction_positions[:, 0] = vector
        interaction_normals[:, 0, 2] = 1.0
    topology = PathTopology(
        valid=torch.ones(rows, dtype=torch.bool),
        tx_id=torch.arange(rows, dtype=torch.int32),
        rx_id=torch.arange(rows, dtype=torch.int32) + 2,
        depth=torch.ones(rows, dtype=torch.int32),
        component_id=torch.ones(rows, dtype=torch.int32),
        primitive_id=torch.arange(rows, dtype=torch.int32) + 30,
        edge_id=torch.full((rows,), -1, dtype=torch.int32),
        material_id=torch.arange(rows, dtype=torch.int32) + 40,
        primitive_sequence=primitive_sequence,
        material_sequence=material_sequence,
        interaction_type=interaction_type,
    )
    geometry = PathGeometry(
        row_identity=topology.row_identity,
        path_length_m=scalar * 3.0,
        delay_s=scalar * 1.0e-9,
        field_direction=vector,
        interaction_position=vector + 30.0,
        interaction_normal=vector + 40.0,
        interaction_positions=interaction_positions,
        interaction_normals=interaction_normals,
    )
    fields = PathFields(
        row_identity=topology.row_identity,
        path_gain=scalar.square(),
        path_field=complex_scalar * 2.0,
        field_xyz=complex_vector,
        coefficient=complex_scalar,
    )
    evaluated = EvaluatedPaths(topology=topology, geometry=geometry, fields=fields)
    sidecars = EvaluatedPathSidecars(
        execution=PathExecutionStats(101, 102, 103, 104, 105, 106, 107),
        diffraction_vector_field=complex_vector * 3.0,
    )
    return evaluated, sidecars


def _scattering_rows() -> dict[str, torch.Tensor]:
    return {
        "tx_id": torch.tensor([9, 8], dtype=torch.int32),
        "rx_id": torch.tensor([7, 6], dtype=torch.int32),
        "primitive_id": torch.tensor([51, 52], dtype=torch.int32),
        "material_id": torch.tensor([61, 62], dtype=torch.int32),
        "position": torch.tensor(
            [[1.5, 2.5, 3.5], [4.5, 5.5, 6.5]], dtype=torch.float32
        ),
        "normal": torch.tensor(
            [[0.0, 0.0, 1.0], [0.0, 1.0, 0.0]], dtype=torch.float32
        ),
        "path_length_m": torch.tensor([11.5, 12.5], dtype=torch.float32),
        "path_gain": torch.tensor([0.25, 0.5], dtype=torch.float32),
        "path_field": torch.tensor([1.0 + 2.0j, 3.0 + 4.0j]),
        "coefficient": torch.tensor([0.25 + 0.5j, 0.75 + 1.0j]),
        "direction": torch.tensor(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=torch.float32
        ),
    }


def _typed_tensors(evaluated: EvaluatedPaths) -> dict[str, torch.Tensor]:
    topology = evaluated.topology
    geometry = evaluated.geometry
    fields = evaluated.fields
    return {
        "valid": topology.valid,
        "tx_id": topology.tx_id,
        "rx_id": topology.rx_id,
        "depth": topology.depth,
        "component_id": topology.component_id,
        "primitive_id": topology.primitive_id,
        "edge_id": topology.edge_id,
        "path_length_m": geometry.path_length_m,
        "delay_s": geometry.delay_s,
        "path_gain": fields.path_gain,
        "path_field": fields.path_field,
        "field_xyz": fields.field_xyz,
        "coefficient": fields.coefficient,
        "field_direction": geometry.field_direction,
        "interaction_position": geometry.interaction_position,
        "interaction_normal": geometry.interaction_normal,
        "material_id": topology.material_id,
        "primitive_sequence": topology.primitive_sequence,
        "material_sequence": topology.material_sequence,
        "interaction_type": topology.interaction_type,
        "interaction_positions": geometry.interaction_positions,
        "interaction_normals": geometry.interaction_normals,
    }


def _install_row_collector(monkeypatch: pytest.MonkeyPatch) -> None:
    def collect(_scene, _config, *, device, info, ad_mode="none"):
        assert device == torch.device("cpu")
        assert ad_mode == "none"
        info.update(
            visibility_launch_count=13,
            path_count=2,
            capped_path_count=1,
            capped_power_fraction=0.125,
        )
        return _scattering_rows(), 11, 7, 1

    monkeypatch.setattr(scattering, "_collect_scattering_rows", collect)


def test_enumerated_package_has_no_scattering_barrel_facade():
    assert enumerated.__all__ == []
    assert not hasattr(enumerated, "append_scattering_paths")
    assert scattering.__all__ == ["append_scattering_evaluated_paths"]


def test_scattering_has_one_typed_concatenation_owner():
    tree = ast.parse(Path(scattering.__file__).read_text(encoding="utf-8"))
    functions = {
        node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)
    }
    owner = functions["_extended_scattering_rows"]
    assert sum(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "torch"
        and node.func.attr == "cat"
        for node in ast.walk(owner)
    ) == 1
    assert "_extend_topology" not in functions
    assert "append_scattering_paths" not in functions


def test_scattering_config_contract_is_minimal_protocol():
    assert contracts.TopologyConfig._is_protocol
    assert not is_dataclass(contracts.TopologyConfig)
    assert set(contracts.TopologyConfig.__annotations__) == {
        "components",
        "max_depth",
        "scattering_samples_per_m2",
        "scattering_power_threshold",
        "scattering_max_paths_per_pair",
    }


def test_typed_scattering_noop_preserves_identity():
    evaluated, sidecars = _evaluated_paths()
    config = SimpleNamespace(components=frozenset(), max_depth=1)

    exported, exported_sidecars, info = scattering.append_scattering_evaluated_paths(
        object(), config, evaluated, sidecars
    )

    assert exported is evaluated
    assert exported_sidecars is sidecars
    assert info == scattering._scattering_info()


@pytest.mark.parametrize(("existing_rows", "width"), ((2, 2), (1, 0), (0, 0)))
def test_typed_scattering_append_preserves_prefix_and_sidecars(
    monkeypatch: pytest.MonkeyPatch,
    existing_rows: int,
    width: int,
):
    evaluated, sidecars = _evaluated_paths(rows=existing_rows, width=width)
    original_identity = evaluated.row_identity
    original_tensors = _typed_tensors(evaluated)
    scene = SimpleNamespace(structures=[object()])
    config = SimpleNamespace(components={"scattering"}, max_depth=1)
    _install_row_collector(monkeypatch)

    typed, typed_sidecars, info = scattering.append_scattering_evaluated_paths(
        scene, config, evaluated, sidecars
    )

    assert typed.row_identity is not original_identity
    assert typed.row_count == existing_rows + 2
    for name, tensor in _typed_tensors(typed).items():
        prefix = original_tensors[name]
        if width == 0 and name in {
            "primitive_sequence",
            "material_sequence",
            "interaction_type",
            "interaction_positions",
            "interaction_normals",
        }:
            assert tensor.shape[1] == 1
        else:
            assert torch.equal(tensor[:existing_rows], prefix), name
        assert tensor.is_contiguous(), name
    assert typed.topology.component_id[-2:].tolist() == [6, 6]
    assert torch.equal(typed.fields.path_gain[-2:], _scattering_rows()["path_gain"])
    assert typed_sidecars.execution.launch_count == 112
    assert typed_sidecars.execution.candidate_count == 111
    assert typed_sidecars.execution.guardrail_count == 106
    assert typed_sidecars.execution.visibility_rejection_count == 102
    assert typed_sidecars.execution.selected_edge_count == 103
    assert typed_sidecars.execution.ad_companion_launches == 106
    assert typed_sidecars.execution.ad_tape_bytes == 107
    assert typed_sidecars.diffraction_vector_field is sidecars.diffraction_vector_field
    assert info["path_count"] == 2


def test_scattering_runtime_cat_order_is_frozen(monkeypatch: pytest.MonkeyPatch):
    evaluated, sidecars = _evaluated_paths()
    scene = SimpleNamespace(structures=[object()])
    config = SimpleNamespace(components={"scattering"}, max_depth=1)
    _install_row_collector(monkeypatch)
    source_names = {id(tensor): name for name, tensor in _typed_tensors(evaluated).items()}
    original_cat = torch.cat
    observed: list[str] = []

    def recording_cat(tensors, *args, **kwargs):
        observed.append(source_names[id(tensors[0])])
        return original_cat(tensors, *args, **kwargs)

    monkeypatch.setattr(scattering.torch, "cat", recording_cat)
    scattering.append_scattering_evaluated_paths(scene, config, evaluated, sidecars)

    assert observed == list(_EXTENDED_ROW_FIELDS)
