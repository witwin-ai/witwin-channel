from __future__ import annotations

import ast
import hashlib
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

_COMPATIBILITY_EXPORTS = frozenset(
    {"_rough_reflection_factor", "_evaluate_shared_fields"}
)
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
    "_evaluate_los_fields": {
        "signature": (
            "(topology: TopologyBatch, source: torch.Tensor, target: torch.Tensor, "
            "source_power: torch.Tensor, tx_pol: torch.Tensor, rx_pol: "
            "torch.Tensor, los_field_op: Callable[..., dict[str, torch.Tensor]], "
            "ledger: AdLaunchLedger | None, field_xyz: torch.Tensor, coefficient: "
            "torch.Tensor, path_field: torch.Tensor, path_gain: torch.Tensor, "
            "path_length: torch.Tensor, delay: torch.Tensor, direction: "
            "torch.Tensor, launch_count: int)"
        ),
        "body_sha256": (
            "b5b86149c055676fd8565a1000f28d33db1cf72641e8c0d374b58950b977e9da"
        ),
        "normalized_ast_sha256": (
            "03816f9cd798d1e0d947a97b6dd964b354fc566b965cf540c30b31a893434cdc"
        ),
    },
    "_evaluate_reflection_fields": {
        "signature": (
            "(compiled: object, topology: TopologyBatch, source: torch.Tensor, "
            "target: torch.Tensor, source_power: torch.Tensor, tx_pol: "
            "torch.Tensor, rx_pol: torch.Tensor, components: frozenset[str] | "
            "set[str], device: torch.device, frequency: float | torch.Tensor, "
            "geometry_ad: bool, vertices: torch.Tensor | None, "
            "reflection_field_op: Callable[..., dict[str, torch.Tensor]], ledger: "
            "AdLaunchLedger | None, material: dict[str, torch.Tensor] | None, "
            "field_xyz: torch.Tensor, coefficient: torch.Tensor, path_field: "
            "torch.Tensor, path_gain: torch.Tensor, path_length: torch.Tensor, "
            "delay: torch.Tensor, direction: torch.Tensor, launch_count: int)"
        ),
        "body_sha256": (
            "1518649e067ad93c37ae872b6610073f906d97d5f433198eaa1dec6e16e7d0db"
        ),
        "normalized_ast_sha256": (
            "c91968647805603ea4070142c2293074ccab1535acddd546cc46e520c8904876"
        ),
    },
    "_evaluate_transmission_fields": {
        "signature": (
            "(compiled: object, topology: TopologyBatch, source: torch.Tensor, "
            "target: torch.Tensor, source_power: torch.Tensor, tx_pol: "
            "torch.Tensor, rx_pol: torch.Tensor, device: torch.device, "
            "geometry_ad: bool, vertices: torch.Tensor | None, "
            "transmission_field_op: Callable[..., dict[str, torch.Tensor]], "
            "ledger: AdLaunchLedger | None, material: dict[str, torch.Tensor] | "
            "None, field_xyz: torch.Tensor, coefficient: torch.Tensor, path_field: "
            "torch.Tensor, path_gain: torch.Tensor, path_length: torch.Tensor, "
            "delay: torch.Tensor, direction: torch.Tensor, launch_count: int)"
        ),
        "body_sha256": (
            "be3ed774bf2b0dcf01d13891d98611f08017a2baf476ca16af52101e3e432251"
        ),
        "normalized_ast_sha256": (
            "7c1bbc0e12d11b12bc7385c280d952234eb47b90672b11b1972a4e7bb76ad7ff"
        ),
    },
    "_evaluate_diffraction_fields": {
        "signature": (
            "(scene: Scene, compiled: object, topology: TopologyBatch, source: "
            "torch.Tensor, target: torch.Tensor, source_power: torch.Tensor, "
            "rx_pol: torch.Tensor, device: torch.device, frequency: float | "
            "torch.Tensor, frequency_value: float | None, ad_enabled: bool, "
            "geometry_ad: bool, vertices: torch.Tensor | None, ledger: "
            "AdLaunchLedger | None, material: dict[str, torch.Tensor] | None, "
            "field_xyz: torch.Tensor, coefficient: torch.Tensor, path_field: "
            "torch.Tensor, path_gain: torch.Tensor, direction: torch.Tensor, "
            "launch_count: int)"
        ),
        "body_sha256": (
            "d89b7367a75646acb25c333698946eb870163edeba6ac0285ea0a459ebafa685"
        ),
        "normalized_ast_sha256": (
            "d8dfe6e4652bd058dc8ee34e9ba01aa4c1375356254f0e74ecb9b66eb8ce1c24"
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
            "832f56e96e3c3f80e8671f33fd6a217ea8a0e03d376ba2df09b743a8217933db"
        ),
        "normalized_ast_sha256": (
            "0de866892f43bc83b1d8fa64044285f88eaa12e07c00598c1022240066f965fb"
        ),
    },
}


def _function_node(name: str) -> ast.FunctionDef:
    source = Path(evaluation.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    return next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    )


def _body_sha256(statements: list[ast.stmt]) -> str:
    dumped = ast.dump(
        ast.Module(body=statements, type_ignores=[]),
        annotate_fields=True,
        include_attributes=False,
    )
    return hashlib.sha256(dumped.encode("utf-8")).hexdigest()


def test_evaluation_helpers_are_same_object_compatibility_exports():
    for name in _COMPATIBILITY_EXPORTS:
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


def test_los_field_extraction_preserves_frozen_prefix_segment():
    helper = _function_node("_evaluate_los_fields")

    assert _body_sha256(helper.body[:-1]) == (
        "31115ef524f94736f69198345a846c03407da5849ed496fe795916864e047a12"
    )
    assert ast.dump(helper.body[-1], include_attributes=False) == (
        "Return(value=Name(id='launch_count', ctx=Load()))"
    )


def test_reflection_field_extraction_preserves_frozen_prefix_segment():
    helper = _function_node("_evaluate_reflection_fields")

    assert _body_sha256(helper.body[:-1]) == (
        "5f297e976b9995d55958b49f84103c1d2627801c2f7fd887bbec587587dd569e"
    )
    assert ast.dump(helper.body[-1], include_attributes=False) == (
        "Return(value=Tuple(elts=[Name(id='material', ctx=Load()), "
        "Name(id='launch_count', ctx=Load())], ctx=Load()))"
    )


def test_transmission_field_extraction_preserves_frozen_prefix_segment():
    helper = _function_node("_evaluate_transmission_fields")

    assert _body_sha256(helper.body[:-1]) == (
        "979b83f4bf93085a5978276b0dcce0e2be8ff961287e165f1410c431a01da8f2"
    )
    assert ast.dump(helper.body[-1], include_attributes=False) == (
        "Return(value=Tuple(elts=[Name(id='material', ctx=Load()), "
        "Name(id='launch_count', ctx=Load())], ctx=Load()))"
    )


def test_diffraction_field_extraction_preserves_frozen_prefix_segment():
    helper = _function_node("_evaluate_diffraction_fields")

    assert _body_sha256(helper.body[:-1]) == (
        "ac8b2a7976ee591ca6c6e6044ebd27983522e1cdd712b4520e0f30dd643d8baa"
    )
    assert ast.dump(helper.body[-1], include_attributes=False) == (
        "Return(value=Tuple(elts=[Name(id='material', ctx=Load()), "
        "Name(id='launch_count', ctx=Load())], ctx=Load()))"
    )


def test_component_field_helpers_do_not_clone_output_buffers():
    for name in (
        "_evaluate_los_fields",
        "_evaluate_reflection_fields",
        "_evaluate_transmission_fields",
        "_evaluate_diffraction_fields",
    ):
        helper = _function_node(name)

        assert not any(
            isinstance(call, ast.Call)
            and isinstance(call.func, ast.Attribute)
            and call.func.attr == "clone"
            for call in ast.walk(helper)
        )


def test_shared_field_orchestrator_preserves_component_order_and_clone_count():
    orchestrator = _function_node("_evaluate_shared_fields")
    clones = sorted(
        (
            call
            for call in ast.walk(orchestrator)
            if isinstance(call, ast.Call)
            and isinstance(call.func, ast.Attribute)
            and call.func.attr == "clone"
        ),
        key=lambda call: call.lineno,
    )
    assert len(clones) == 7
    assert [
        call.func.value.attr
        for call in clones
        if isinstance(call.func.value, ast.Attribute)
    ] == [
        "field_xyz",
        "coefficient",
        "path_field",
        "path_gain",
        "path_length_m",
        "delay_s",
        "field_direction",
    ]

    component_markers = []
    for statement in orchestrator.body:
        if isinstance(statement, ast.Assign):
            target_names = {
                target.id for target in statement.targets if isinstance(target, ast.Name)
            }
            if isinstance(statement.value, ast.Call) and isinstance(
                statement.value.func, ast.Name
            ):
                if statement.value.func.id == "_evaluate_los_fields":
                    component_markers.append("los")
                elif statement.value.func.id == "_evaluate_reflection_fields":
                    component_markers.append("reflection")
                elif statement.value.func.id == "_evaluate_transmission_fields":
                    component_markers.append("transmission")
                elif statement.value.func.id == "_evaluate_diffraction_fields":
                    component_markers.append("diffraction")
            elif target_names == {"material"}:
                component_markers.append("material")
            elif target_names & {"transmission_rows", "diffraction_rows", "coupled_rows"}:
                component_markers.extend(target_names)
        elif (
            isinstance(statement, ast.AnnAssign)
            and isinstance(statement.target, ast.Name)
            and statement.target.id == "material"
        ):
            component_markers.append("material")
        elif (
            isinstance(statement, ast.For)
            and isinstance(statement.target, ast.Name)
            and statement.target.id == "depth_value"
        ):
            component_markers.append("reflection")
        elif isinstance(statement, ast.Return):
            component_markers.append("epilogue")

    assert component_markers == [
        "los",
        "material",
        "reflection",
        "transmission",
        "diffraction",
        "coupled_rows",
        "epilogue",
    ]


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
    names = repr(tuple(_COMPATIBILITY_EXPORTS))
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
