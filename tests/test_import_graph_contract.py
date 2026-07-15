from __future__ import annotations

from collections import Counter
import copy
from pathlib import Path

from ci import check_import_graph as graph


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = REPOSITORY_ROOT / "src" / "witwin" / "channel_native"
ALLOWLIST = REPOSITORY_ROOT / "ci" / "import_graph_allowlist.json"


def _synthetic_package(tmp_path: Path, files: dict[str, str]) -> Path:
    package_root = tmp_path / "src" / "witwin" / "channel_native"
    for relative, source in files.items():
        path = package_root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source, encoding="utf-8")
    return package_root


def test_current_import_debt_is_exact_and_allowlisted():
    violations = graph.scan_package(PACKAGE_ROOT)
    allowlist = graph.load_allowlist(ALLOWLIST)

    assert graph.check_allowlist(violations, allowlist) == []
    assert Counter(
        graph._DEBT_GROUP_BY_RULE[violation.rule] for violation in violations
    ) == {
        "existing_boundary": 6,
    }


def test_import_scan_is_stable():
    assert graph.scan_package(PACKAGE_ROOT) == graph.scan_package(PACKAGE_ROOT)


def test_solver_and_ops_boundaries_are_detected(tmp_path: Path):
    package_root = _synthetic_package(
        tmp_path,
        {
            "__init__.py": "",
            "path/__init__.py": "",
            "path/pipeline.py": """
import witwin.channel_native.deterministic.solver
import witwin.channel_native.montecarlo.scattering_events
from witwin.channel_native.core.kernels import ops
from witwin.channel_native.runtime import extension
from witwin.channel_native.runtime import native_extension
""",
            "deterministic/solver.py": "",
            "montecarlo/scattering_events.py": "",
            "core/kernels/ops.py": "",
            "runtime/extension.py": "",
        },
    )

    rules = Counter(violation.rule for violation in graph.scan_package(package_root))

    assert rules == {
        "solver_to_solver": 1,
        "enumerated_pipeline_mc_internal": 1,
        "direct_core_kernels_ops": 1,
        "solver_raw_extension": 2,
    }


def test_propagation_runtime_oracle_and_kernel_boundaries_are_detected(
    tmp_path: Path,
):
    package_root = _synthetic_package(
        tmp_path,
        {
            "__init__.py": "",
            "path/result.py": "",
            "deterministic/solver.py": "",
            "scene/__init__.py": "",
            "scattering/__init__.py": "",
            "propagation/enumerated.py": ("import witwin.channel_native.path.result\n"),
            "propagation/topology.py": "import witwin.channel_native.scene\n",
            "propagation/topology/discovery.py": "",
            "propagation/geometry.py": (
                "import witwin.channel_native.propagation.topology.discovery\n"
            ),
            "propagation/fields.py": "import witwin.channel_native.path.result\n",
            "runtime/state.py": "import witwin.channel_native.scene\n",
            "physics/oracle.py": (
                "import torch\nimport witwin.channel_native.scattering\n"
            ),
            "scene/kernels/private.py": """
import witwin.channel_native.deterministic.solver
import witwin.channel_native.materials.kernels.private
""",
            "materials/kernels/private.py": "",
        },
    )

    rules = Counter(violation.rule for violation in graph.scan_package(package_root))

    assert rules == {
        "enumerated_forbidden_dependency": 1,
        "topology_forbidden_dependency": 1,
        "geometry_forbidden_dependency": 1,
        "fields_forbidden_dependency": 1,
        "runtime_forbidden_dependency": 1,
        "oracle_production_dependency": 2,
        "domain_kernel_solver_dependency": 1,
        "cross_domain_private_kernel": 1,
    }


def test_topology_cannot_import_canonical_or_legacy_compiled_scene(tmp_path: Path):
    package_root = _synthetic_package(
        tmp_path,
        {
            "__init__.py": "",
            "scene/compiled.py": "",
            "core/runtime/compiled_scene.py": "",
            "propagation/topology/canonical.py": (
                "from witwin.channel_native.scene.compiled import CompiledScene\n"
            ),
            "propagation/topology/legacy.py": (
                "from witwin.channel_native.core.runtime.compiled_scene "
                "import CompiledScene\n"
            ),
        },
    )

    rules = Counter(violation.rule for violation in graph.scan_package(package_root))

    assert rules == {"topology_forbidden_dependency": 2}


def test_propagation_legacy_path_topology_dependencies_are_hard_failures(
    tmp_path: Path,
):
    package_root = _synthetic_package(
        tmp_path,
        {
            "__init__.py": "",
            "core/path_topology.py": "",
            "core/legacy_path_topology.py": "",
            "propagation/fields.py": (
                "from witwin.channel_native.core.path_topology import TopologyBatch\n"
            ),
            "propagation/geometry.py": (
                "import witwin.channel_native.core.legacy_path_topology\n"
            ),
        },
    )

    rules = Counter(violation.rule for violation in graph.scan_package(package_root))

    assert rules == {"propagation_legacy_path_topology_dependency": 2}


def test_real_propagation_graph_has_no_legacy_path_topology_dependency():
    assert not any(
        violation.rule == "propagation_legacy_path_topology_dependency"
        for violation in graph.scan_package(PACKAGE_ROOT)
    )


def test_wildcard_and_relative_boundary_escapes_are_hard_failures(tmp_path: Path):
    package_root = _synthetic_package(
        tmp_path,
        {
            "__init__.py": "",
            "runtime/extension.py": "",
            "materials/model.py": "from ..runtime import extension\n",
            "path/pipeline.py": "from ....external import helper\nfrom os import *\n",
        },
    )

    rules = Counter(violation.rule for violation in graph.scan_package(package_root))

    assert rules == {
        "relative_cross_domain": 1,
        "relative_outside_package": 1,
        "wildcard_import": 1,
    }


def test_allowlist_cannot_add_or_relocate_debt():
    violations = graph.scan_package(PACKAGE_ROOT)
    allowlist = graph.load_allowlist(ALLOWLIST)
    changed = copy.deepcopy(allowlist)
    changed["debts"]["solver_to_solver"]["allowed"].append("solver-new")

    issues = graph.check_allowlist(violations, changed)

    assert any("unknown allowed IDs: solver-new" in issue for issue in issues)


def test_allowlist_must_remove_stale_entries_with_resolved_debt():
    violations = graph.scan_package(PACKAGE_ROOT)
    allowlist = graph.load_allowlist(ALLOWLIST)
    resolved = [
        violation
        for violation in violations
        if graph._DEBT_GROUP_BY_RULE[violation.rule] != "existing_boundary"
    ]

    issues = graph.check_allowlist(resolved, allowlist)

    assert any("stale existing_boundary allowance" in issue for issue in issues)


def test_frozen_baseline_universe_cannot_change():
    violations = graph.scan_package(PACKAGE_ROOT)
    allowlist = graph.load_allowlist(ALLOWLIST)
    changed = copy.deepcopy(allowlist)
    changed["debts"]["solver_to_solver"]["baseline"][0]["line"] += 1

    issues = graph.check_allowlist(violations, changed)

    assert any("frozen import debt universe changed" in issue for issue in issues)


def test_cli_passes_with_repository_defaults(capsys):
    assert graph.main(["--repository-root", str(REPOSITORY_ROOT)]) == 0
    assert "import graph contract passed" in capsys.readouterr().out
