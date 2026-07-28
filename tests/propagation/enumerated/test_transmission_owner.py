from __future__ import annotations

import ast
import inspect
from pathlib import Path

from ci import check_import_graph as graph
from witwin.channel import deterministic as deterministic_module
from witwin.channel.montecarlo import bdpt as bdpt_pipeline
from witwin.channel import path as path_module
from witwin.channel.interactions import transmission
from witwin.channel.propagation.enumerated import engine
from witwin.channel.kernels import geometry as geometry_kernels
from witwin.channel.propagation.penetration import (
    SegmentPenetrationPolicy,
)
from witwin.channel.kernels import topology as topology_pack


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
PACKAGE_ROOT = REPOSITORY_ROOT / "src" / "witwin" / "channel"
_RETIRED_NAMES = (
    "TransmissionClosestHitQuery",
    "query_transmission_closest_hit",
    "iter_transmission_active_rows",
)


def _function(module: object, name: str) -> ast.FunctionDef:
    tree = ast.parse(Path(module.__file__).read_text(encoding="utf-8"))
    return next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    )


def _calls(definition: ast.AST) -> dict[str, list[ast.Call]]:
    result: dict[str, list[ast.Call]] = {}
    for node in ast.walk(definition):
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Name):
            name = node.func.id
        elif isinstance(node.func, ast.Attribute):
            name = node.func.attr
        else:
            continue
        result.setdefault(name, []).append(node)
    return result


def test_transmission_owner_uses_one_stable_native_operation_family() -> None:
    assert transmission.SegmentPenetrationPolicy is SegmentPenetrationPolicy
    assert transmission.rayd_segment_penetration_forward is (
        geometry_kernels.rayd_segment_penetration_forward
    )
    assert transmission.rayd_segment_penetration_ad is (
        geometry_kernels.rayd_segment_penetration_ad
    )
    assert transmission.enumerated_transmission_topology_pack is (
        topology_pack.enumerated_transmission_topology_pack
    )

    signature = inspect.signature(transmission._transmission_topology)
    assert "failure_state" in signature.parameters
    assert "ad_mode" in signature.parameters
    definition = _function(transmission, "_transmission_topology")
    calls = _calls(definition)
    source = ast.unparse(definition)

    assert len(calls["rayd_segment_penetration_forward"]) == 1
    assert len(calls["rayd_segment_penetration_ad"]) == 1
    assert len(calls["enumerated_transmission_topology_pack"]) == 1
    assert "SegmentPenetrationPolicy.EnumeratedFullDistance" in source
    assert "compiled.enumerated_penetration_scene_diagonal_m" in source
    assert not any(
        isinstance(node, (ast.For, ast.While)) for node in ast.walk(definition)
    )
    assert not any(name in source for name in _RETIRED_NAMES)


def test_enumerated_engine_owns_one_transaction_and_terminal_observer() -> None:
    definition = _function(engine, "evaluate_enumerated_paths")
    calls = _calls(definition)
    transaction_factory = _function(engine, "_create_transmission_capacity_transaction")
    finish_boundary = _function(engine, "_finish_capacity_boundary")
    factory_calls = _calls(transaction_factory)
    finish_calls = _calls(finish_boundary)

    assert len(factory_calls["create_solve_capacity_transaction"]) == 1
    assert len(finish_calls["terminal_check"]) == 1
    assert (
        len(calls["_finish_capacity_boundary"])
        == len(calls["evaluate_path_fields"])
        == 2
    )
    assert (
        calls["_create_transmission_capacity_transaction"][0].lineno
        < calls["_transmission_topology"][0].lineno
    )
    for fields, finish in zip(
        sorted(calls["evaluate_path_fields"], key=lambda call: call.lineno),
        sorted(calls["_finish_capacity_boundary"], key=lambda call: call.lineno),
        strict=True,
    ):
        assert fields.lineno < finish.lineno
    assert (
        finish_calls["sanitize_enumerated_capacity_transaction"][0].lineno
        < finish_calls["terminal_check"][0].lineno
    )

    transmission_call = calls["_transmission_topology"][0]
    keywords = {
        keyword.arg: ast.unparse(keyword.value)
        for keyword in transmission_call.keywords
    }
    assert "capacity_transaction.failure_state" in keywords["failure_state"]
    assert keywords["ad_mode"] == "ad_mode"
    assert "capacity_failure_terminal_check" not in ast.unparse(finish_boundary)


def test_path_deterministic_and_adr008_oracle_share_the_engine_entry() -> None:
    assert path_module.evaluate_enumerated_paths is engine.evaluate_enumerated_paths
    assert (
        deterministic_module.evaluate_enumerated_paths
        is engine.evaluate_enumerated_paths
    )
    assert bdpt_pipeline.evaluate_enumerated_paths is engine.evaluate_enumerated_paths

    bdpt_source = Path(bdpt_pipeline.__file__).read_text(encoding="utf-8")
    assert "propagation.enumerated.transmission" not in bdpt_source
    assert "kernels.geometry" not in bdpt_source


def test_retired_depth_march_sources_and_references_are_deleted() -> None:
    assert not (PACKAGE_ROOT / "propagation" / "geometry" / "transmission.py").exists()
    assert not (
        PACKAGE_ROOT / "propagation" / "topology" / "discovery" / "transmission.py"
    ).exists()

    for path in PACKAGE_ROOT.rglob("*.py"):
        source = path.read_text(encoding="utf-8")
        for name in _RETIRED_NAMES:
            assert name not in source, f"{path} retains retired {name}"


def test_activation_adds_no_old_owner_import_edge_or_host_actual_count_read() -> None:
    retired_modules = {
        "witwin.channel.propagation.geometry.transmission",
        "witwin.channel.propagation.topology.discovery.transmission",
    }
    edges = graph.collect_import_edges(PACKAGE_ROOT)
    assert not any(edge.target in retired_modules for edge in edges)

    for module in (engine, transmission):
        source = Path(module.__file__).read_text(encoding="utf-8")
        for forbidden in (
            "device_candidate_count.item",
            "device_candidate_count.tolist",
            "device_guardrail_count.item",
            "device_guardrail_count.tolist",
            "cudaStreamSynchronize",
        ):
            assert forbidden not in source
