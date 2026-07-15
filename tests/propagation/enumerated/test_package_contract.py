from __future__ import annotations

import ast
from collections import defaultdict, deque
from dataclasses import is_dataclass
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch

from ci import check_import_graph
from witwin.channel_native.core.path_topology import TopologyBatch
from witwin.channel_native.propagation import enumerated
from witwin.channel_native.propagation.enumerated import contracts, scattering
from witwin.channel_native.propagation.topology.export import export_evaluated_rows


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
PACKAGE_ROOT = REPOSITORY_ROOT / "src" / "witwin" / "channel_native"
SCATTERING_PATH = PACKAGE_ROOT / "propagation" / "enumerated" / "scattering.py"
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


def _topology_batch(*, rows: int = 2, width: int = 2) -> TopologyBatch:
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
    return TopologyBatch(
        valid=torch.ones(rows, dtype=torch.bool),
        tx_id=torch.arange(rows, dtype=torch.int32),
        rx_id=torch.arange(rows, dtype=torch.int32) + 2,
        depth=torch.ones(rows, dtype=torch.int32),
        component_id=torch.ones(rows, dtype=torch.int32),
        primitive_id=torch.arange(rows, dtype=torch.int32) + 30,
        edge_id=torch.full((rows,), -1, dtype=torch.int32),
        path_length_m=scalar * 3.0,
        delay_s=scalar * 1.0e-9,
        path_gain=scalar.square(),
        path_field=complex_scalar * 2.0,
        field_xyz=complex_vector,
        coefficient=complex_scalar,
        field_direction=vector,
        interaction_position=vector + 30.0,
        interaction_normal=vector + 40.0,
        material_id=torch.arange(rows, dtype=torch.int32) + 40,
        primitive_sequence=primitive_sequence,
        material_sequence=material_sequence,
        interaction_type=interaction_type,
        interaction_positions=interaction_positions,
        interaction_normals=interaction_normals,
        launch_count=101,
        visibility_rejection_count=102,
        selected_edge_count=103,
        candidate_count=104,
        guardrail_count=105,
        diffraction_vector_field=complex_vector * 3.0,
        ad_companion_launches=106,
        ad_tape_bytes=107,
    )


def _scattering_rows() -> dict[str, torch.Tensor]:
    return {
        "tx_id": torch.tensor([9, 8], dtype=torch.int32),
        "rx_id": torch.tensor([7, 6], dtype=torch.int32),
        "primitive_id": torch.tensor([51, 52], dtype=torch.int32),
        "material_id": torch.tensor([61, 62], dtype=torch.int32),
        "position": torch.tensor(
            [[1.5, 2.5, 3.5], [4.5, 5.5, 6.5]], dtype=torch.float32
        ),
        "normal": torch.tensor([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0]], dtype=torch.float32),
        "path_length_m": torch.tensor([11.5, 12.5], dtype=torch.float32),
        "path_gain": torch.tensor([0.25, 0.5], dtype=torch.float32),
        "path_field": torch.tensor([1.0 + 2.0j, 3.0 + 4.0j], dtype=torch.complex64),
        "coefficient": torch.tensor([0.25 + 0.5j, 0.75 + 1.0j], dtype=torch.complex64),
        "direction": torch.tensor(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=torch.float32
        ),
    }


def _legacy_tensors(topology: TopologyBatch) -> dict[str, torch.Tensor]:
    return {name: getattr(topology, name) for name in _EXTENDED_ROW_FIELDS}


def _typed_tensors(evaluated: object) -> dict[str, torch.Tensor]:
    return {
        "valid": evaluated.topology.valid,
        "tx_id": evaluated.topology.tx_id,
        "rx_id": evaluated.topology.rx_id,
        "depth": evaluated.topology.depth,
        "component_id": evaluated.topology.component_id,
        "primitive_id": evaluated.topology.primitive_id,
        "edge_id": evaluated.topology.edge_id,
        "path_length_m": evaluated.geometry.path_length_m,
        "delay_s": evaluated.geometry.delay_s,
        "path_gain": evaluated.fields.path_gain,
        "path_field": evaluated.fields.path_field,
        "field_xyz": evaluated.fields.field_xyz,
        "coefficient": evaluated.fields.coefficient,
        "field_direction": evaluated.geometry.field_direction,
        "interaction_position": evaluated.geometry.interaction_position,
        "interaction_normal": evaluated.geometry.interaction_normal,
        "material_id": evaluated.topology.material_id,
        "primitive_sequence": evaluated.topology.primitive_sequence,
        "material_sequence": evaluated.topology.material_sequence,
        "interaction_type": evaluated.topology.interaction_type,
        "interaction_positions": evaluated.geometry.interaction_positions,
        "interaction_normals": evaluated.geometry.interaction_normals,
    }


def _install_row_collector(monkeypatch: pytest.MonkeyPatch) -> None:
    def collect(_scene, _config, *, device, info):
        assert device == torch.device("cpu")
        info.update(
            visibility_launch_count=13,
            path_count=2,
            capped_path_count=1,
            capped_power_fraction=0.125,
        )
        return _scattering_rows(), 11, 7, 1

    monkeypatch.setattr(scattering, "_collect_scattering_rows", collect)


def test_enumerated_package_reexports_the_scattering_callable_same_object():
    assert enumerated.__all__ == ["append_scattering_paths"]
    assert scattering.__all__ == ["append_scattering_paths"]
    assert enumerated.append_scattering_paths is scattering.append_scattering_paths
    assert scattering.append_scattering_paths.__module__ == scattering.__name__
    assert (
        scattering.append_scattering_evaluated_paths.__module__ == scattering.__name__
    )
    assert not hasattr(enumerated, "append_scattering_evaluated_paths")


@pytest.mark.parametrize(
    "imports",
    (
        (
            "from witwin.channel_native.propagation import enumerated; "
            "from witwin.channel_native.propagation.enumerated import scattering"
        ),
        (
            "from witwin.channel_native.propagation.enumerated import scattering; "
            "from witwin.channel_native.propagation import enumerated"
        ),
        (
            "from witwin.channel_native.propagation.topology.export import "
            "EvaluatedPathSidecars; "
            "from witwin.channel_native.propagation.enumerated import scattering; "
            "from witwin.channel_native.propagation import enumerated"
        ),
    ),
)
def test_enumerated_import_order_preserves_callable_identity(imports: str):
    code = (
        f"{imports}; "
        "assert enumerated.append_scattering_paths is "
        "scattering.append_scattering_paths; "
        "assert scattering.append_scattering_evaluated_paths.__module__ == "
        "scattering.__name__; "
        "assert enumerated.__all__ == ['append_scattering_paths']"
    )
    environment = os.environ.copy()
    source_root = str(REPOSITORY_ROOT / "src")
    environment["PYTHONPATH"] = os.pathsep.join(
        value for value in (source_root, environment.get("PYTHONPATH")) if value
    )

    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPOSITORY_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr


def test_scattering_single_owner_freezes_cat_order_and_field_mapping():
    tree = ast.parse(SCATTERING_PATH.read_text(encoding="utf-8"))
    definitions = {
        node.name: node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    }

    bundle = definitions["_ExtendedScatteringRows"]
    assert isinstance(bundle, ast.ClassDef)
    assert [
        node.target.id
        for node in bundle.body
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name)
    ] == list(_EXTENDED_ROW_FIELDS)

    owner = definitions["_extended_scattering_rows"]
    assert isinstance(owner, ast.FunctionDef)
    constructors = [
        node.value
        for node in ast.walk(owner)
        if isinstance(node, ast.Return)
        and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Name)
        and node.value.func.id == "_ExtendedScatteringRows"
    ]
    assert len(constructors) == 1
    constructor = constructors[0]
    assert [keyword.arg for keyword in constructor.keywords] == list(
        _EXTENDED_ROW_FIELDS
    )
    assert all(
        isinstance(keyword.value, ast.Call)
        and isinstance(keyword.value.func, ast.Name)
        and keyword.value.func.id == "cat"
        for keyword in constructor.keywords
    )

    assert (
        sum(
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "torch"
            and node.func.attr == "cat"
            for node in ast.walk(owner)
        )
        == 1
    )
    for name in ("_extend_topology", "_extend_evaluated_paths"):
        assert not any(
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "torch"
            and node.func.attr == "cat"
            for node in ast.walk(definitions[name])
        )


def test_scattering_legacy_and_typed_entrypoints_share_row_collection_owner():
    tree = ast.parse(SCATTERING_PATH.read_text(encoding="utf-8"))
    functions = {
        node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)
    }

    def calls(function_name: str, callee: str) -> int:
        return sum(
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == callee
            for node in ast.walk(functions[function_name])
        )

    assert calls("append_scattering_paths", "_collect_scattering_rows") == 1
    assert calls("append_scattering_paths", "_extend_topology") == 1
    assert calls("append_scattering_paths", "_extend_evaluated_paths") == 0
    assert calls("append_scattering_evaluated_paths", "_collect_scattering_rows") == 1
    assert calls("append_scattering_evaluated_paths", "_extend_topology") == 0
    assert calls("append_scattering_evaluated_paths", "_extend_evaluated_paths") == 1
    assert calls("_extend_topology", "_extended_scattering_rows") == 1
    assert calls("_extend_evaluated_paths", "_extended_scattering_rows") == 1


def test_scattering_contracts_are_minimal_structural_protocols():
    assert contracts.TopologyBatch._is_protocol
    assert contracts.TopologyConfig._is_protocol
    assert not is_dataclass(contracts.TopologyBatch)
    assert set(contracts.TopologyConfig.__annotations__) == {
        "components",
        "max_depth",
        "scattering_samples_per_m2",
        "scattering_power_threshold",
        "scattering_max_paths_per_pair",
    }
    assert set(contracts.TopologyBatch.__annotations__) == {
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
        "launch_count",
        "candidate_count",
        "guardrail_count",
    }


def test_scattering_has_no_core_path_topology_dependency_or_scc():
    edges = check_import_graph.collect_import_edges(PACKAGE_ROOT)
    package = "witwin.channel_native"
    scattering_module = f"{package}.propagation.enumerated.scattering"
    contracts_module = f"{package}.propagation.enumerated.contracts"
    legacy_module = f"{package}.core.path_topology"

    assert any(
        edge.source == scattering_module and edge.target == contracts_module
        for edge in edges
    )
    assert not any(
        edge.source == scattering_module and edge.target == legacy_module
        for edge in edges
    )

    graph: dict[str, set[str]] = defaultdict(set)
    for edge in edges:
        graph[edge.source].add(edge.target)

    def reachable(source: str, target: str) -> bool:
        queue = deque((source,))
        seen = {source}
        while queue:
            current = queue.popleft()
            if current == target:
                return True
            for child in graph[current]:
                if child.startswith(package) and child not in seen:
                    seen.add(child)
                    queue.append(child)
        return False

    assert not (
        reachable(scattering_module, legacy_module)
        and reachable(legacy_module, scattering_module)
    )


def test_scattering_noop_preserves_topology_row_identity():
    topology = object()
    config = SimpleNamespace(components=frozenset(), max_depth=1)

    exported, info = scattering.append_scattering_paths(object(), config, topology)

    assert exported is topology
    assert info["path_count"] == 0
    assert info["capped_path_count"] == 0


def test_typed_scattering_noop_preserves_evaluated_and_sidecar_identity():
    topology = _topology_batch()
    evaluated, sidecars = export_evaluated_rows(topology)
    config = SimpleNamespace(components=frozenset(), max_depth=1)

    exported, exported_sidecars, info = scattering.append_scattering_evaluated_paths(
        object(), config, evaluated, sidecars
    )

    assert exported is evaluated
    assert exported_sidecars is sidecars
    assert info == scattering._scattering_info()


def test_typed_scattering_empty_collection_is_an_exact_noop(
    monkeypatch: pytest.MonkeyPatch,
):
    topology = _topology_batch()
    evaluated, sidecars = export_evaluated_rows(topology)
    scene = SimpleNamespace(structures=[object()])
    config = SimpleNamespace(components={"scattering"}, max_depth=1)

    monkeypatch.setattr(
        scattering,
        "_collect_scattering_rows",
        lambda *_args, **_kwargs: (None, 0, 0, 0),
    )
    exported, exported_sidecars, info = scattering.append_scattering_evaluated_paths(
        scene, config, evaluated, sidecars
    )

    assert exported is evaluated
    assert exported_sidecars is sidecars
    assert info == scattering._scattering_info()


@pytest.mark.parametrize(("existing_rows", "width"), ((2, 2), (1, 0), (0, 0)))
def test_typed_scattering_bridge_is_bit_exact_with_legacy_append(
    monkeypatch: pytest.MonkeyPatch,
    existing_rows: int,
    width: int,
):
    topology = _topology_batch(rows=existing_rows, width=width)
    evaluated, sidecars = export_evaluated_rows(topology)
    original_identity = evaluated.row_identity
    original_tensors = _typed_tensors(evaluated)
    scene = SimpleNamespace(structures=[object()])
    config = SimpleNamespace(components={"scattering"}, max_depth=1)
    _install_row_collector(monkeypatch)

    legacy, legacy_info = scattering.append_scattering_paths(scene, config, topology)
    typed, typed_sidecars, typed_info = scattering.append_scattering_evaluated_paths(
        scene, config, evaluated, sidecars
    )

    legacy_tensors = _legacy_tensors(legacy)
    typed_tensors = _typed_tensors(typed)
    assert tuple(legacy_tensors) == _EXTENDED_ROW_FIELDS
    assert tuple(typed_tensors) == _EXTENDED_ROW_FIELDS
    for name in _EXTENDED_ROW_FIELDS:
        legacy_tensor = legacy_tensors[name]
        typed_tensor = typed_tensors[name]
        assert torch.equal(typed_tensor, legacy_tensor), name
        assert typed_tensor.dtype == legacy_tensor.dtype
        assert typed_tensor.shape == legacy_tensor.shape
        assert typed_tensor.stride() == legacy_tensor.stride()
        assert typed_tensor.is_contiguous()
        assert legacy_tensor.is_contiguous()
        prefix = original_tensors[name]
        if width == 0 and name in {
            "primitive_sequence",
            "material_sequence",
            "interaction_type",
        }:
            expected = torch.full(
                (existing_rows, 1),
                -1 if name != "interaction_type" else 0,
                dtype=typed_tensor.dtype,
            )
            assert torch.equal(typed_tensor[:existing_rows], expected), name
        elif width == 0 and name in {
            "interaction_positions",
            "interaction_normals",
        }:
            assert torch.equal(
                typed_tensor[:existing_rows],
                torch.zeros((existing_rows, 1, 3), dtype=typed_tensor.dtype),
            ), name
        else:
            assert torch.equal(typed_tensor[:existing_rows], prefix), name
        if prefix.numel() and typed_tensor.numel():
            assert typed_tensor.untyped_storage().data_ptr() != (
                prefix.untyped_storage().data_ptr()
            ), name

    assert torch.equal(typed.topology.tx_id[-2:], _scattering_rows()["tx_id"])
    assert typed.row_identity is not original_identity
    assert typed.geometry.row_identity is typed.topology.row_identity
    assert typed.fields.row_identity is typed.topology.row_identity
    assert legacy_info == typed_info
    assert typed_info == {
        **scattering._scattering_info(),
        "visibility_launch_count": 13,
        "path_count": 2,
        "capped_path_count": 1,
        "capped_power_fraction": 0.125,
    }
    assert typed_sidecars.execution.launch_count == legacy.launch_count == 112
    assert typed_sidecars.execution.candidate_count == legacy.candidate_count == 111
    assert typed_sidecars.execution.guardrail_count == legacy.guardrail_count == 106
    assert typed_sidecars.execution.visibility_rejection_count == 102
    assert typed_sidecars.execution.selected_edge_count == 103
    assert typed_sidecars.execution.ad_companion_launches == 106
    assert typed_sidecars.execution.ad_tape_bytes == 107
    assert typed_sidecars.diffraction_vector_field is topology.diffraction_vector_field


def test_scattering_runtime_cat_order_is_frozen_for_legacy_and_typed_paths(
    monkeypatch: pytest.MonkeyPatch,
):
    topology = _topology_batch()
    evaluated, sidecars = export_evaluated_rows(topology)
    scene = SimpleNamespace(structures=[object()])
    config = SimpleNamespace(components={"scattering"}, max_depth=1)
    _install_row_collector(monkeypatch)
    source_names = {
        id(tensor): name for name, tensor in _legacy_tensors(topology).items()
    }
    original_cat = torch.cat
    observed: list[str] = []

    def recording_cat(tensors, *args, **kwargs):
        observed.append(source_names[id(tensors[0])])
        return original_cat(tensors, *args, **kwargs)

    monkeypatch.setattr(scattering.torch, "cat", recording_cat)
    scattering.append_scattering_paths(scene, config, topology)
    assert observed == list(_EXTENDED_ROW_FIELDS)

    observed.clear()
    scattering.append_scattering_evaluated_paths(scene, config, evaluated, sidecars)
    assert observed == list(_EXTENDED_ROW_FIELDS)
