from __future__ import annotations

import ast
import json
from pathlib import Path
import tomllib

from ci import check_import_graph


ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = ROOT / "src" / "witwin" / "channel"
CONSUMER_ROOT = PACKAGE_ROOT / "propagation" / "consumer"


def _top_level_imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    imports: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            imports.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imports.add(node.module or "")
    return imports


def test_consumer_contract_is_versioned_and_snapshot_frozen() -> None:
    init = ast.parse(
        (CONSUMER_ROOT / "__init__.py").read_text(encoding="utf-8"),
        filename="consumer/__init__.py",
    )
    exported = next(
        ast.literal_eval(node.value)
        for node in init.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "__all__"
            for target in node.targets
        )
    )
    assert exported == [
        "CONTRACT_VERSION",
        "Complex3Transport",
        "EndpointBatch",
        "FixedTopologyEvaluation",
        "FixedTopologyRequest",
        "JonesTransport",
        "PropagationCapabilities",
        "PropagationConvention",
        "PropagationDiagnostics",
        "PropagationEvaluation",
        "PropagationGeometry",
        "PropagationPathBatch",
        "PropagationRequest",
        "PropagationTopology",
        "ScalarTransport",
        "evaluate",
        "reevaluate",
    ]
    contracts = (CONSUMER_ROOT / "contracts.py").read_text(encoding="utf-8")
    assert "CONTRACT_VERSION = 1" in contracts

    snapshot = json.loads(
        (ROOT / "ci" / "public-api-snapshot.json").read_text(encoding="utf-8")
    )
    modules = {entry["module"]: entry for entry in snapshot["modules"]}
    public = modules["witwin.channel.propagation.consumer"]
    assert [entry["name"] for entry in public["exports"]] == exported


def test_public_consumer_import_does_not_eagerly_load_internal_definitions() -> None:
    service_imports = _top_level_imports(CONSUMER_ROOT / "service.py")
    assert "witwin.channel.propagation.enumerated.engine" not in service_imports
    assert "witwin.channel.propagation.models.evaluated" not in service_imports
    assert "witwin.channel.propagation.models.topology" not in service_imports
    assert "witwin.channel.propagation.models.geometry" not in service_imports
    assert "witwin.channel.propagation.models.fields" not in service_imports
    assert not any(
        module.startswith("witwin.channel.path")
        or module.startswith("witwin.channel.deterministic")
        or module.startswith("witwin.channel.montecarlo")
        for module in service_imports
    )


def test_consumer_has_exact_named_adr008_enumerated_edge() -> None:
    edges = check_import_graph.collect_import_edges(PACKAGE_ROOT)
    consumer_edges = [
        edge
        for edge in edges
        if edge.source.startswith("witwin.channel.propagation.consumer")
        and edge.target == "witwin.channel.propagation.enumerated.engine"
    ]
    assert {(edge.source, edge.imported_name) for edge in consumer_edges} == {
        (
            "witwin.channel.propagation.consumer.service",
            "EnumeratedEndpointTensors",
        ),
        (
            "witwin.channel.propagation.consumer.service",
            "evaluate_enumerated_paths",
        ),
    }


def test_compact_provenance_and_historical_adr_dispositions_are_frozen() -> None:
    provenance = json.loads(
        (
            ROOT / "docs" / "dev" / "audit" / "stage1-phase3-consumer-provenance.json"
        ).read_text(encoding="utf-8")
    )
    compact = provenance["compact_boundary"]
    assert compact["rows"] == "exact K in owning native stable pair-major order"
    assert compact["count_d2h_delta_from_phase2"] == 0
    assert compact["stream_sync_delta_from_phase2"] == 0
    assert compact["python_torch_compaction"] is False
    assert compact["capacity_shaped_public_result"] is False
    assert provenance["historical_experiments"] == {
        "ADR-029": "Superseded; caller-free",
        "ADR-030": "Dormant; caller-free",
        "ADR-031": "Rejected; caller-free",
    }

    consumer_source = "\n".join(
        path.read_text(encoding="utf-8") for path in sorted(CONSUMER_ROOT.glob("*.py"))
    )
    for forbidden in (
        "path_capacity_per_pair",
        "diffraction_state_capacity",
        "Qr",
        "canonical_capacity",
        "diffraction_pair_reduce",
    ):
        assert forbidden not in consumer_source

    expected_status = {
        "adr-029-device-resident-capacity-results.md": "Superseded",
        "adr-030-deterministic-diffraction-pair-reduction.md": "Dormant",
        "adr-031-per-pair-raw-reflection-epc-capacity.md": "Rejected",
    }
    for name, status in expected_status.items():
        header = (ROOT / "docs" / "dev" / "standards" / name).read_text(
            encoding="utf-8"
        )[:500]
        assert status in header


def test_release_matrix_is_truthful_about_channel_abi() -> None:
    with (ROOT / "ci" / "support-matrix.toml").open("rb") as stream:
        runtime = tomllib.load(stream)["runtime"][0]
    assert runtime["python_spec"] == ">=3.11,<3.12"
    assert runtime["torch_spec"] == "==2.10.0"
    assert runtime["cuda_toolkit"] == "12.8.1"
    assert runtime["binary_abi"] == "versioned-libtorch-python-extension"
    assert runtime["libtorch_stable_abi"] is False
    assert runtime["stable_abi_floor_for_eligible_artifacts"] == "2.10"
    assert 87 in runtime["declared_unverified_sm"]

    workflow = (
        ROOT / ".github" / "workflows" / "publish-witwin-channel.yml"
    ).read_text(encoding="utf-8")
    assert "runs-on: windows-2022" in workflow
    assert "CIBW_MANYLINUX_X86_64_IMAGE: manylinux_2_28" in workflow
    assert "--expected-sass 70,75,80,86,87,89,90,100,101,120" in workflow
    assert "--expected-ptx 120" in workflow
    assert "Stable ABI" not in workflow


def test_consumer_governance_is_radar_neutral() -> None:
    capabilities = (CONSUMER_ROOT / "service.py").read_text(encoding="utf-8")
    for forbidden in ("waveform", "rcs", "iq", "adc", "cfar", "detection"):
        assert forbidden not in capabilities.lower()


def test_consumer_v1_has_no_scattering_execution_route() -> None:
    service = (CONSUMER_ROOT / "service.py").read_text(encoding="utf-8")
    assert '"scattering"' not in service
    assert "append_scattering_evaluated_paths" not in service


def test_compact_autograd_native_companions_have_one_topology_owner() -> None:
    symbols = {
        "evaluated_paths_compact_finalize_backward",
        "evaluated_paths_compact_finalize_jvp",
    }
    owners: dict[str, list[Path]] = {symbol: [] for symbol in symbols}
    for path in PACKAGE_ROOT.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "_required_native_op"
                and node.args
                and isinstance(node.args[0], ast.Constant)
                and node.args[0].value in symbols
            ):
                owners[node.args[0].value].append(path)

    facade = (
        PACKAGE_ROOT
        / "propagation"
        / "topology"
        / "kernels"
        / "compact_autograd.py"
    )
    assert owners == {symbol: [facade] for symbol in symbols}
    assert "witwin.channel.propagation.topology.kernels" in _top_level_imports(
        CONSUMER_ROOT / "_native.py"
    )
