from __future__ import annotations

import ast
from pathlib import Path

from tools.refactor_baseline import python_body_hashes
from witwin.channel.scene import tensors as scene_tensors
from witwin.channel.propagation.fields import evaluation
from witwin.channel.propagation.geometry import reevaluate
from witwin.channel.runtime import autograd_contracts


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]

def _function_node(name: str) -> ast.FunctionDef:
    source = Path(evaluation.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    return next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    )


def test_component_field_helpers_do_not_clone_output_buffers():
    for name in (
        "_evaluate_los_fields",
        "_evaluate_reflection_fields",
        "_evaluate_transmission_fields",
        "_evaluate_diffraction_fields",
        "_evaluate_coupled_fields",
    ):
        helper = _function_node(name)

        assert not any(
            isinstance(call, ast.Call)
            and isinstance(call.func, ast.Attribute)
            and call.func.attr == "clone"
            for call in ast.walk(helper)
        )


def test_transmission_field_helper_forwards_selected_topology_validity_first():
    helper = _function_node("_evaluate_transmission_fields")
    transmission_args = next(
        node.value
        for node in ast.walk(helper)
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id == "transmission_args"
    )
    assert isinstance(transmission_args, ast.Tuple)
    assert ast.unparse(transmission_args.elts[0]) == "topology.valid[rows].contiguous()"


def test_diffraction_field_helper_forwards_selected_topology_validity_first():
    helper = _function_node("_evaluate_diffraction_fields")
    wedge_args = next(
        node.value
        for node in ast.walk(helper)
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id == "wedge_args"
    )
    assert isinstance(wedge_args, ast.Tuple)
    assert (
        ast.unparse(wedge_args.elts[0])
        == "topology.valid[diffraction_rows].contiguous()"
    )


def test_typed_field_orchestrator_preserves_component_order_and_clone_count():
    orchestrator = _function_node("evaluate_path_fields")
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
                elif statement.value.func.id == "_evaluate_coupled_fields":
                    component_markers.append("coupled")
            elif target_names == {"material"}:
                component_markers.append("material")
        elif (
            isinstance(statement, ast.AnnAssign)
            and isinstance(statement.target, ast.Name)
            and statement.target.id == "material"
        ):
            component_markers.append("material")
        elif isinstance(statement, ast.Return):
            component_markers.append("epilogue")

    assert component_markers == [
        "los",
        "material",
        "reflection",
        "transmission",
        "diffraction",
        "coupled",
        "epilogue",
    ]


def test_typed_field_orchestrator_uses_split_domains_and_no_legacy_dependency():
    orchestrator = _function_node("evaluate_path_fields")
    source = Path(evaluation.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported_modules = {
        node.module
        for node in tree.body
        if isinstance(node, ast.ImportFrom) and node.module is not None
    }
    assignments = {
        node.targets[0].id: ast.unparse(node.value)
        for node in orchestrator.body
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
    }

    assert assignments["topology"] == "paths.topology"
    assert assignments["geometry"] == "paths.geometry"
    assert assignments["input_fields"] == "paths.fields"
    assert "witwin.channel.core.path_topology" not in imported_modules


def test_evaluation_preserves_nested_material_tuple_body():
    qualified_name = f"{evaluation.__name__}._evaluate_coupled_fields.material_tuple"
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
