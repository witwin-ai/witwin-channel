# Copyright Xingyu Chen.
# Tests import graph contract.

from __future__ import annotations

from collections import Counter
import copy
from pathlib import Path

from ci import check_import_graph as graph


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = REPOSITORY_ROOT / "witwin" / "channel"
ALLOWLIST = REPOSITORY_ROOT / "ci" / "import_graph_allowlist.json"


def _synthetic_package(tmp_path: Path, files: dict[str, str]) -> Path:
    package_root = tmp_path / "witwin" / "channel"
    for relative, source in files.items():
        path = package_root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source, encoding="utf-8")
    return package_root


def test_current_import_debt_is_exact_and_allowlisted():
    violations = graph.scan_package(PACKAGE_ROOT)
    allowlist = graph.load_allowlist(ALLOWLIST)

    assert graph.check_allowlist(violations, allowlist) == []
    # The ``existing_boundary`` group is fully repaid: the last live entry, the
    # relative ``deployment -> runtime.extension`` import, now uses the absolute
    # cross-domain form the rest of the package uses. Only the ADR-008 BDPT
    # enumerated oracle remains, and it is a sanctioned dependency, not debt to
    # be repaid.
    assert Counter(
        graph._DEBT_GROUP_BY_RULE[violation.rule] for violation in violations
    ) == {
        "mc_enumerated_dependency": 1,
    }


def test_import_scan_is_stable():
    assert graph.scan_package(PACKAGE_ROOT) == graph.scan_package(PACKAGE_ROOT)


def test_solver_and_deleted_module_boundaries_are_detected(tmp_path: Path):
    package_root = _synthetic_package(
        tmp_path,
        {
            "__init__.py": "",
            "path/__init__.py": "",
            "path/pipeline.py": """
import witwin.channel.deterministic.solver
import witwin.channel.montecarlo.scattering_events
from witwin.channel.core.kernels import ops
from witwin.channel.runtime import native_extension
import witwin.channel._channel
""",
            "deterministic/solver.py": "",
            "montecarlo/scattering_events.py": "",
            "runtime.py": "",
        },
    )

    rules = Counter(violation.rule for violation in graph.scan_package(package_root))

    assert rules == {
        "solver_to_solver": 1,
        "enumerated_pipeline_mc_internal": 1,
        "deleted_module_dependency": 1,
        "dissolved_module_dependency": 1,
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
            "propagation/enumerated.py": ("import witwin.channel.path.result\n"),
            "propagation/topology.py": "import witwin.channel.scene\n",
            "propagation/topology/discovery.py": "",
            "propagation/geometry.py": (
                "import witwin.channel.propagation.topology.discovery\n"
            ),
            "propagation/fields.py": "import witwin.channel.path.result\n",
            "runtime/state.py": "import witwin.channel.scene\n",
            "scene/kernels/private.py": """
import witwin.channel.deterministic.solver
import witwin.channel.materials.kernels.private
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
                "from witwin.channel.scene.compiled import CompiledScene\n"
            ),
            "propagation/topology/legacy.py": (
                "from witwin.channel.core.runtime.compiled_scene "
                "import CompiledScene\n"
            ),
        },
    )

    rules = Counter(violation.rule for violation in graph.scan_package(package_root))

    assert rules == {
        "topology_forbidden_dependency": 1,
        "dissolved_module_dependency": 1,
    }


def test_topology_and_geometry_cannot_import_scene_handle_helpers(tmp_path: Path):
    package_root = _synthetic_package(
        tmp_path,
        {
            "__init__.py": "",
            "scene/native_handles.py": "",
            "propagation/topology.py": (
                "from witwin.channel.scene.native_handles import helper\n"
            ),
            "propagation/geometry/kernels/helper.py": (
                "from witwin.channel.scene.native_handles import helper\n"
            ),
        },
    )

    rules = Counter(violation.rule for violation in graph.scan_package(package_root))

    assert rules == {
        "topology_forbidden_dependency": 1,
        "geometry_forbidden_dependency": 1,
    }


def test_deleted_modules_are_global_hard_failures(tmp_path: Path):
    package_root = _synthetic_package(
        tmp_path,
        {
            "__init__.py": "",
            "propagation/fields.py": (
                "from witwin.channel.core.path_topology import TopologyBatch\n"
            ),
            "propagation/geometry.py": (
                "from witwin.channel.core.kernels import ops\n"
            ),
        },
    )

    rules = Counter(violation.rule for violation in graph.scan_package(package_root))

    assert rules == {
        "deleted_module_dependency": 2,
        "dissolved_module_dependency": 2,
    }


def test_real_graph_has_no_deleted_module_dependency():
    assert not any(
        violation.rule == "deleted_module_dependency"
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
    """An allowance whose violation disappeared must be deleted, not kept.

    The group is the one that still carries a live allowance, so the check is
    exercised against real data rather than a synthetic package.
    """

    violations = graph.scan_package(PACKAGE_ROOT)
    allowlist = graph.load_allowlist(ALLOWLIST)
    resolved = [
        violation
        for violation in violations
        if graph._DEBT_GROUP_BY_RULE[violation.rule] != "mc_enumerated_dependency"
    ]

    issues = graph.check_allowlist(resolved, allowlist)

    assert any("stale mc_enumerated_dependency allowance" in issue for issue in issues)


def test_frozen_baseline_universe_cannot_change():
    violations = graph.scan_package(PACKAGE_ROOT)
    allowlist = graph.load_allowlist(ALLOWLIST)
    changed = copy.deepcopy(allowlist)
    changed["debts"]["solver_to_solver"]["baseline"][0]["line"] += 1

    issues = graph.check_allowlist(violations, changed)

    assert any("frozen import debt universe changed" in issue for issue in issues)


def test_reexport_map_resolves_facade_symbols_to_defining_modules(tmp_path: Path):
    package_root = _synthetic_package(
        tmp_path,
        {
            "__init__.py": "",
            "propagation/__init__.py": (
                "from .enumerated.engine import evaluate_enumerated_paths\n"
                "from .models import EvaluatedPaths\n"
            ),
            "propagation/enumerated/__init__.py": "",
            "propagation/enumerated/engine.py": "",
            "propagation/models/__init__.py": (
                "from .evaluated import EvaluatedPaths\n"
            ),
            "propagation/models/evaluated.py": "",
            "propagation/topology/__init__.py": (
                "from .kernels.sampling import mc_sample_directions\n"
            ),
            "propagation/topology/kernels/__init__.py": "",
            "propagation/topology/kernels/sampling.py": "",
        },
    )

    reexports = graph.build_reexport_map(package_root)
    package = "witwin.channel"

    assert reexports[(f"{package}.propagation", "evaluate_enumerated_paths")] == (
        f"{package}.propagation.enumerated.engine"
    )
    # Re-export chains resolve recursively through nested package facades.
    assert reexports[(f"{package}.propagation.models", "EvaluatedPaths")] == (
        f"{package}.propagation.models.evaluated"
    )
    assert reexports[(f"{package}.propagation.topology", "mc_sample_directions")] == (
        f"{package}.propagation.topology.kernels.sampling"
    )


def test_reexport_canonicalization_reveals_facade_dependency(tmp_path: Path):
    package_root = _synthetic_package(
        tmp_path,
        {
            "__init__.py": "",
            "propagation/__init__.py": (
                "from .enumerated.engine import evaluate_enumerated_paths\n"
                "from .models import EvaluatedPaths\n"
            ),
            "propagation/enumerated/__init__.py": "",
            "propagation/enumerated/engine.py": "",
            "propagation/models/__init__.py": "",
            "propagation/topology/__init__.py": (
                "from .kernels.sampling import mc_sample_directions\n"
            ),
            "propagation/topology/kernels/__init__.py": "",
            "propagation/topology/kernels/sampling.py": "",
            "montecarlo/__init__.py": "",
            "montecarlo/bdpt/__init__.py": "",
            "montecarlo/bdpt/pipeline.py": (
                "from witwin.channel.propagation import "
                "EvaluatedPaths, evaluate_enumerated_paths\n"
            ),
            "montecarlo/basic/__init__.py": "",
            "montecarlo/basic/kernels/__init__.py": "",
            "montecarlo/basic/kernels/sampling.py": (
                "from witwin.channel.propagation.topology "
                "import mc_sample_directions\n"
            ),
        },
    )

    rules = Counter(violation.rule for violation in graph.scan_package(package_root))

    # The BDPT facade import of ``evaluate_enumerated_paths`` is canonicalized to
    # ``propagation.enumerated.engine`` and fires. The ``EvaluatedPaths``
    # re-export resolves to ``propagation.models`` (no rule), and the basic
    # sampling seam stops at the private ``topology.kernels`` boundary so it does
    # not become a cross-domain private kernel import.
    assert rules == {"mc_enumerated_dependency": 1}


def test_module_only_imports_are_not_canonicalized(tmp_path: Path):
    package_root = _synthetic_package(
        tmp_path,
        {
            "__init__.py": "",
            "propagation/__init__.py": (
                "from .enumerated.engine import evaluate_enumerated_paths\n"
            ),
            "propagation/enumerated/__init__.py": "",
            "propagation/enumerated/engine.py": "",
            "montecarlo/__init__.py": "",
            "montecarlo/bdpt/__init__.py": "",
            "montecarlo/bdpt/pipeline.py": (
                "import witwin.channel.propagation\n"
            ),
        },
    )

    # ``import package`` edges target the package itself and stay as-is, so the
    # module-only dependency does not resolve into the enumerated engine.
    assert not any(
        violation.rule == "mc_enumerated_dependency"
        for violation in graph.scan_package(package_root)
    )


def test_reexport_canonicalization_exposes_real_bdpt_enumerated_edge():
    package = "witwin.channel"
    reexports = graph.build_reexport_map(PACKAGE_ROOT)

    assert reexports[(f"{package}.propagation", "evaluate_enumerated_paths")] == (
        f"{package}.propagation.enumerated"
    )

    # ``collect_import_edges`` keeps the raw facade target that owner and seam
    # tests rely on ...
    raw = graph.collect_import_edges(PACKAGE_ROOT)
    assert any(
        edge.source == f"{package}.montecarlo.bdpt"
        and edge.target == f"{package}.propagation"
        and edge.imported_name == "evaluate_enumerated_paths"
        for edge in raw
    )

    # ... while ``scan_package`` classifies the canonicalized edge.
    violations = graph.scan_package(PACKAGE_ROOT)
    enumerated = [
        violation
        for violation in violations
        if violation.rule == "mc_enumerated_dependency"
    ]
    assert enumerated == [
        graph.Violation(
            "witwin/channel/montecarlo/bdpt.py",
            108,
            0,
            "mc_enumerated_dependency",
            f"{package}.montecarlo.bdpt",
            f"{package}.propagation.enumerated",
        )
    ]


def test_bdpt_enumerated_allowlist_entry_is_exact_and_adr_bound():
    allowlist = graph.load_allowlist(ALLOWLIST)
    group = allowlist["debts"]["mc_enumerated_dependency"]

    assert len(group["baseline"]) == 1
    assert group["allowed"] == ["mc-enum-001"]

    entry = group["baseline"][0]
    assert entry["id"] == "mc-enum-001"
    assert entry["rule"] == "mc_enumerated_dependency"
    # The 2026-07-27 re-baseline re-keyed this entry onto the collapsed
    # ``montecarlo/bdpt.py`` module ahead of that collapse landing, and the
    # concept-axis collapse of ``propagation/enumerated/`` into one module
    # re-keyed ``target`` onto that module. The rule and the ADR binding have
    # never moved; only the two module spellings did, each when its own
    # package became a module.
    assert entry["source"] == "witwin.channel.montecarlo.bdpt"
    assert entry["target"] == "witwin.channel.propagation.enumerated"
    assert "ADR-008" in (entry.get("adr", "") + entry.get("justification", ""))

    assert graph._DEBT_GROUP_BY_RULE["mc_enumerated_dependency"] == (
        "mc_enumerated_dependency"
    )

    violations = graph.scan_package(PACKAGE_ROOT)
    bound = [
        violation
        for violation in violations
        if violation.rule == "mc_enumerated_dependency"
    ]
    assert len(bound) == 1
    assert bound[0].source == "witwin.channel.montecarlo.bdpt"


def test_public_init_forbids_every_internal_kernels_package(tmp_path: Path):
    package_root = _synthetic_package(
        tmp_path,
        {
            "__init__.py": (
                "from witwin.channel.materials.kernels import encoding\n"
                "from witwin.channel.scattering.kernels import lobe\n"
                "from witwin.channel.scene.kernels import handles\n"
                "from witwin.channel.deterministic.kernels import accumulate\n"
                "from witwin.channel.propagation.fields.kernels import transport\n"
            ),
            "materials/kernels/encoding.py": "",
            "scattering/kernels/lobe.py": "",
            "scene/kernels/handles.py": "",
            "deterministic/kernels/accumulate.py": "",
            "propagation/fields/kernels/transport.py": "",
        },
    )

    rules = Counter(violation.rule for violation in graph.scan_package(package_root))

    assert rules == {"public_init_internal": 5}


def test_public_init_real_graph_has_no_internal_targets():
    """The root reaches ``build_info`` through ``deployment``, not a shim.

    Deleting ``core.kernels.extension`` repaid the last ``public_init_internal``
    debt the package root carried.
    """

    violations = graph.scan_package(PACKAGE_ROOT)
    public_init_targets = {
        violation.target
        for violation in violations
        if violation.rule == "public_init_internal"
        and violation.source == "witwin.channel"
    }

    assert public_init_targets == set()


def test_dissolved_namespaces_cannot_be_recreated(tmp_path: Path):
    """``core`` was dissolved into real owners and must not come back.

    The NumPy reference oracle that used to live in ``physics`` now sits in
    ``tests/reference/em_oracle.py``, so its isolation is structural: this
    checker only walks the shipped package and cannot reach it at all.
    """

    package_root = _synthetic_package(
        tmp_path,
        {
            "__init__.py": "",
            "scattering/__init__.py": "",
            "core/field_state.py": "",
            "scattering/lobe.py": (
                "from witwin.channel.core.field_state import Complex3State\n"
            ),
        },
    )

    rules = Counter(violation.rule for violation in graph.scan_package(package_root))

    assert rules == {"dissolved_module_dependency": 1}


def test_cli_passes_with_repository_defaults(capsys):
    assert graph.main(["--repository-root", str(REPOSITORY_ROOT)]) == 0
    assert "import graph contract passed" in capsys.readouterr().out