from __future__ import annotations

import ast
from dataclasses import is_dataclass
import hashlib
import inspect
import os
from pathlib import Path
import pickle
import subprocess
import sys

import pytest

from ci import check_import_graph as graph
from witwin.channel_native.core import path_topology as legacy
from witwin.channel_native.propagation import models
from witwin.channel_native.propagation.fields import evaluation
from witwin.channel_native.propagation.models import adapters, contracts


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = REPOSITORY_ROOT / "src" / "witwin" / "channel_native"

_TOPOLOGY_CONFIG_DIGEST = (
    "a1359261cbd2361e26874a6e951b5877f7a9bcda9642bf1eddbfb5bcec41cbee"
)
_EVALUATED_ROWS_BODY_DIGEST = (
    "af1220e6ab9211f6f6a3a314a6450250e6cdd455b8306e1b247b5e7b254b9aaf"
)
_FIELDS_FUNCTION_DIGESTS = {
    "_rough_reflection_factor": (
        "0261c6cd651790319819a3e1d3a1d14f93a9c4a771bce7933f993f75913a5fa5"
    ),
    "_evaluate_los_fields": (
        "03816f9cd798d1e0d947a97b6dd964b354fc566b965cf540c30b31a893434cdc"
    ),
    "_evaluate_reflection_fields": (
        "c91968647805603ea4070142c2293074ccab1535acddd546cc46e520c8904876"
    ),
    "_evaluate_transmission_fields": (
        "7c1bbc0e12d11b12bc7385c280d952234eb47b90672b11b1972a4e7bb76ad7ff"
    ),
    "_evaluate_diffraction_fields": (
        "d8dfe6e4652bd058dc8ee34e9ba01aa4c1375356254f0e74ecb9b66eb8ce1c24"
    ),
    "_evaluate_coupled_fields": (
        "87d8c609ab9afb6af9efab47d2d91bda20295c084e1dd11546795ea1c2bca4a5"
    ),
    "_evaluate_shared_fields": (
        "010cfc47ae2b566265abb6a777a4564576a09e18360190b01abe68d4a46c543f"
    ),
}

_TOPOLOGY_CONFIG_FIELDS = {
    "max_depth",
    "components",
    "max_paths",
    "max_paths_scope",
}
_EVALUATED_ROWS_FIELDS = {
    "valid",
    "tx_id",
    "rx_id",
    "depth",
    "component_id",
    "primitive_id",
    "edge_id",
    "material_id",
    "primitive_sequence",
    "material_sequence",
    "interaction_type",
    "path_length_m",
    "delay_s",
    "field_direction",
    "interaction_position",
    "interaction_normal",
    "interaction_positions",
    "interaction_normals",
    "path_gain",
    "path_field",
    "field_xyz",
    "coefficient",
    "launch_count",
    "visibility_rejection_count",
    "selected_edge_count",
    "candidate_count",
    "guardrail_count",
    "ad_companion_launches",
    "ad_tape_bytes",
    "diffraction_vector_field",
}


def _definition(module, name: str):
    tree = ast.parse(Path(module.__file__).read_text(encoding="utf-8"))
    return next(
        node
        for node in tree.body
        if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name == name
    )


def _definition_digest(module, name: str) -> str:
    payload = ast.dump(_definition(module, name), include_attributes=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def _body_digest(module, name: str) -> str:
    definition = _definition(module, name)
    payload = "\n".join(
        ast.dump(node, include_attributes=False) for node in definition.body
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def test_structural_protocols_have_exact_fields_and_no_mixed_model_copy():
    assert contracts.TopologyConfig._is_protocol
    assert contracts.EvaluatedRowsSource._is_protocol
    assert not is_dataclass(contracts.TopologyConfig)
    assert not is_dataclass(contracts.EvaluatedRowsSource)
    assert contracts.EvaluatedRowsSource is not legacy.TopologyBatch
    assert set(contracts.TopologyConfig.__annotations__) == _TOPOLOGY_CONFIG_FIELDS
    assert set(contracts.EvaluatedRowsSource.__annotations__) == _EVALUATED_ROWS_FIELDS

    annotations = inspect.get_annotations(
        adapters.evaluated_paths_from_topology_batch, eval_str=True
    )
    assert annotations["source"] is contracts.EvaluatedRowsSource


def test_topology_config_identity_and_legacy_introspection_owner():
    assert models.TopologyConfig is contracts.TopologyConfig
    assert legacy.TopologyConfig is contracts.TopologyConfig
    assert models.EvaluatedRowsSource is contracts.EvaluatedRowsSource
    assert contracts.TopologyConfig.__module__ == legacy.__name__
    assert "TopologyConfig" in models.__all__
    assert "EvaluatedRowsSource" in models.__all__


@pytest.mark.parametrize(
    "imports",
    (
        (
            "from witwin.channel_native.propagation.models import contracts, adapters; "
            "from witwin.channel_native.core import path_topology as legacy; "
            "from witwin.channel_native.propagation.fields import evaluation; "
            "from witwin.channel_native.propagation import models"
        ),
        (
            "from witwin.channel_native.core import path_topology as legacy; "
            "from witwin.channel_native.propagation.fields import evaluation; "
            "from witwin.channel_native.propagation import models; "
            "from witwin.channel_native.propagation.models import contracts, adapters"
        ),
        (
            "from witwin.channel_native.propagation.fields import evaluation; "
            "from witwin.channel_native.propagation.models import adapters, contracts; "
            "from witwin.channel_native.propagation import models; "
            "from witwin.channel_native.core import path_topology as legacy"
        ),
    ),
)
def test_fresh_process_import_orders_preserve_contract_identity(imports: str):
    code = (
        f"{imports}; "
        "assert legacy.TopologyConfig is contracts.TopologyConfig; "
        "assert models.TopologyConfig is contracts.TopologyConfig; "
        "assert models.EvaluatedRowsSource is contracts.EvaluatedRowsSource"
    )
    environment = os.environ.copy()
    source_root = str(REPOSITORY_ROOT / "src")
    environment["PYTHONPATH"] = os.pathsep.join(
        value for value in (source_root, environment.get("PYTHONPATH")) if value
    )
    subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPOSITORY_ROOT,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )


def test_topology_config_class_pickle_replays_through_legacy_owner():
    assert pickle.loads(pickle.dumps(contracts.TopologyConfig)) is (
        contracts.TopologyConfig
    )
    legacy_payload = b"cwitwin.channel_native.core.path_topology\nTopologyConfig\n."
    assert pickle.loads(legacy_payload) is contracts.TopologyConfig


def test_moved_protocol_bodies_and_fields_functions_are_exact():
    assert _definition_digest(contracts, "TopologyConfig") == (_TOPOLOGY_CONFIG_DIGEST)
    assert _body_digest(contracts, "EvaluatedRowsSource") == (
        _EVALUATED_ROWS_BODY_DIGEST
    )
    for name, digest in _FIELDS_FUNCTION_DIGESTS.items():
        assert _definition_digest(evaluation, name) == digest


def test_contract_extraction_does_not_form_a_core_models_scc():
    edges = graph.collect_import_edges(PACKAGE_ROOT)
    adjacency: dict[str, set[str]] = {}
    for edge in edges:
        adjacency.setdefault(edge.source, set()).add(edge.target)

    def reaches(source: str, target: str) -> bool:
        pending = [source]
        seen: set[str] = set()
        while pending:
            current = pending.pop()
            if current == target:
                return True
            if current in seen:
                continue
            seen.add(current)
            pending.extend(adjacency.get(current, ()))
        return False

    core = "witwin.channel_native.core.path_topology"
    contracts_module = "witwin.channel_native.propagation.models.contracts"
    assert reaches(core, contracts_module)
    assert not reaches(contracts_module, core)
