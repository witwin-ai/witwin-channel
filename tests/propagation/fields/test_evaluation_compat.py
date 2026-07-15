from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

import pytest

from ci import check_ops_migration as migration
from tools.refactor_baseline import python_body_hashes
from witwin.channel_native.core import path_topology as legacy
from witwin.channel_native.core import scene_tensors
from witwin.channel_native.propagation.fields import evaluation
from witwin.channel_native.propagation.geometry import reevaluate
from witwin.channel_native.runtime import autograd_contracts


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]

_CONTRACTS = {
    "_rough_reflection_factor": {
        "signature": (
            "(compiled: object, topology: TopologyBatch, rows: torch.Tensor, "
            "depth_value: int, source: torch.Tensor, material: dict[str, "
            "torch.Tensor], positions: torch.Tensor, normals: torch.Tensor, *, "
            "frequency_hz: float | torch.Tensor, scattering_active: bool)"
        ),
        "body_sha256": (
            "bf4ed6fd4c6fbd093aed5dbaffb010a15274ca1f062f01a3324a0341b4e97e48"
        ),
        "normalized_ast_sha256": (
            "0261c6cd651790319819a3e1d3a1d14f93a9c4a771bce7933f993f75913a5fa5"
        ),
    },
    "_evaluate_shared_fields": {
        "signature": (
            "(scene: Scene, compiled: object, topology: TopologyBatch, "
            "tx_positions: torch.Tensor, tx_power: torch.Tensor, rx_positions: "
            "torch.Tensor, *, components: frozenset[str] | set[str]=frozenset(), "
            "ad_mode: str='none', frequency_value: float | None=None)"
        ),
        "body_sha256": (
            "98624e873cdbfc8caa1e575ccb40ce2e3293700653892c265a9083790c3d1438"
        ),
        "normalized_ast_sha256": (
            "0cc5f7790c24b0d23718eb396f9ed934d24ccdde123300760fc21d3180a1b714"
        ),
    },
}


def test_evaluation_helpers_are_same_object_compatibility_exports():
    for name in _CONTRACTS:
        owner = getattr(evaluation, name)

        assert owner.__module__ == evaluation.__name__
        assert getattr(legacy, name) is owner


def test_evaluation_preserves_frozen_function_contracts():
    definitions = {
        item.terminal_name: item
        for item in migration.scan_definitions(REPOSITORY_ROOT)
        if item.qualified_name.startswith(f"{evaluation.__name__}.")
    }

    assert definitions.keys() == _CONTRACTS.keys()
    for name, contract in _CONTRACTS.items():
        definition = definitions[name]
        assert definition.signature == contract["signature"]
        assert definition.body_sha256 == contract["body_sha256"]
        assert definition.normalized_ast_sha256 == contract["normalized_ast_sha256"]


def test_evaluation_preserves_nested_material_tuple_body():
    qualified_name = f"{evaluation.__name__}._evaluate_shared_fields.material_tuple"
    definitions = {
        item["qualified_name"]: item for item in python_body_hashes(REPOSITORY_ROOT)
    }
    definition = definitions[qualified_name]

    assert (
        definition["body_sha256"]
        == "1cd1b88ccfa9f647bacaa020eccd83d4eeb90ec2f7a12ed7c76cbcf1a9e11a87"
    )
    assert (
        definition["normalized_ast_sha256"]
        == "d6f7bc05cbf008ede514ded97373c2ba9357e2c39f58fc87ca0dc743f54efbb6"
    )


def test_evaluation_uses_canonical_dependencies():
    assert evaluation.ops is autograd_contracts
    assert evaluation._frequency_scalar is scene_tensors._frequency_scalar
    assert (
        evaluation._geometry_participates_in_ad
        is reevaluate._geometry_participates_in_ad
    )
    assert evaluation._opposite_vertex_ids is reevaluate._opposite_vertex_ids
    assert evaluation._reflection_geometry_ad is reevaluate._reflection_geometry_ad
    assert (
        evaluation._vertices_participate_in_ad
        is reevaluate._vertices_participate_in_ad
    )


@pytest.mark.parametrize(
    "imports",
    (
        (
            "from witwin.channel_native.core import path_topology as legacy; "
            "from witwin.channel_native.propagation.fields import evaluation"
        ),
        (
            "from witwin.channel_native.propagation.fields import evaluation; "
            "from witwin.channel_native.core import path_topology as legacy"
        ),
    ),
)
def test_evaluation_import_order_preserves_facade_identity(imports: str):
    names = repr(tuple(_CONTRACTS))
    code = (
        f"{imports}; "
        "from witwin.channel_native.runtime import autograd_contracts; "
        f"names={names}; "
        "assert all(getattr(legacy, name) is getattr(evaluation, name) "
        "for name in names); "
        "assert evaluation.ops is autograd_contracts"
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
