from __future__ import annotations

import ast
import hashlib
import json
import re
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "docs/dev/audit"
EVIDENCE_PATH = AUDIT / "phase13-boundary-dedup-phase11b-evidence.json"
LEDGER_PATH = AUDIT / "duplication-classification.json"
RELEASE_EVIDENCE_PATH = AUDIT / "phase13-phase11-release-acceptance.json"
STABLE_RECOVERY_EVIDENCE_PATH = (
    AUDIT / "phase13-adr032-stable-recovery-final-report.json"
)
PHASE11_IMPLEMENTATION_BOUNDARY = "d2681c5810d78fdd1132a60a88b568c00581f6e2"
_EXPECTED_PENDING_FINAL_ACCEPTANCE = {
    "clean-checkout nightly tier at the final Channel commit",
    "clean-checkout release tier at the final Channel commit",
    "final wheel contents and SHA-256",
    "final _channel_native PE/DSO audit and SHA-256",
    "final native build fingerprint bound to the accepted build",
    "Phase 12 profiler and performance evidence",
}
_PHASE11B_SOURCE_SHA256 = {
    "src/witwin/channel_native/propagation/fields/kernels/functional.py": (
        "9cc738bb1a5b75e52f7ad7c34ba9f02d4a89492adec3e96702023a965b318ed0"
    ),
    "src/witwin/channel_native/scattering/kernels/functional.py": (
        "92b3764a6553c5f35e4efc4e5bee838b4d4a4ca1186004ef22ea2bd76084da33"
    ),
    "src/witwin/channel_native/scattering/kernels/functional_chain.py": (
        "4e158e006a209f5ec1c34662bdfb6b38bbd50fad27b233b311d328e5f74023dc"
    ),
    "src/witwin/channel_native/scattering/kernels/autograd_chain.py": (
        "8615d85227ea66f69ffb91bc47baa7f8dec9cf9265e354dbc17b5f83f478d95a"
    ),
    "native/channel_native/kernels/bdpt_connect_visibility.cu": (
        "695686f29e181abd7bd8af7971cce09e15f55ebb4cfa863c1b334e0edf061a89"
    ),
    "native/channel_native/kernels/diffraction.cu": (
        "79c08019afae7cd252e5798ededc7767f145b03b54e457177d3d967f829af185"
    ),
    "native/channel_native/kernels/los.cu": (
        "7313fd71274564fa24ca32c935d8074b6f4f75e9968ee87a392563a3f8a45911"
    ),
    "native/channel_native/kernels/reflection.cu": (
        "61d96ef4734c567afe8294e02028786daca7a541cf305547b13f967d3e52d241"
    ),
}

# Plan-15 phase 6 lifted the per-domain kernel facades into the top-level
# `witwin/channel/kernels/` package. The Phase-11b evidence records the paths
# those regions had when it was captured, so the live-file check follows the
# move instead of the evidence being rewritten.
_PHASE15_KERNEL_RELOCATIONS = {
    "witwin/channel/propagation/fields/kernels/functional.py": (
        "witwin/channel/kernels/fields.py"
    ),
    "witwin/channel/scattering/kernels/functional.py": (
        "witwin/channel/kernels/scattering.py"
    ),
    "witwin/channel/scattering/kernels/functional_chain.py": (
        "witwin/channel/kernels/scattering.py"
    ),
    "witwin/channel/scattering/kernels/autograd_chain.py": (
        "witwin/channel/kernels/scattering.py"
    ),
    "native/channel/kernels/bdpt_connect_visibility.cu": (
        "native/channel/kernels/bdpt_connect.cu"
    ),
    "native/channel/kernels/los.cu": (
        "native/channel/kernels/los_consumer.cu"
    ),
}

_LIVE_FILE_SUFFIXES = {".cpp", ".cu", ".cuh", ".h", ".json", ".md", ".py", ".toml", ".yml"}
_LIVE_ROOTS = (
    ROOT / "witwin",
    ROOT / "native",
    ROOT / "ci",
    ROOT / ".github/workflows",
)
_LIVE_TOP_LEVEL = (
    ROOT / "CMakeLists.txt",
    ROOT / "dependencies/rayd.lock.json",
    ROOT / "FEATURE_LIST.md",
    ROOT / "AGENTS.md",
    ROOT / "CLAUDE.md",
)
_TRIANGLE_VERTEX_V2_LINES = {
    (
        "witwin/channel/interactions/scattering.py",
        "v2 = vertices.index_select(0, faces[:, 2])",
    ),
    (
        "witwin/channel/interactions/scattering.py",
        "areas = 0.5 * torch.linalg.cross(v1 - v0, v2 - v0).norm(dim=-1)",
    ),
    (
        "witwin/channel/interactions/scattering.py",
        "+ b2 * v2.index_select(0, chosen)",
    ),
    (
        "native/channel/kernels/diffraction.cu",
        "const int v2 = tri[2];",
    ),
    ("native/channel/kernels/diffraction.cu", "return v2;"),
    (
        "witwin/channel/kernels/geometry.py",
        "companions (the adjoint/tangent of normalize(cross(v1 - v0, v2 - v0))",
    ),
}


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _live_surface_files() -> tuple[Path, ...]:
    nested = (
        path
        for root in _LIVE_ROOTS
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in _LIVE_FILE_SUFFIXES
    )
    return tuple(sorted((*nested, *_LIVE_TOP_LEVEL)))


def _module_functions(path: Path) -> dict[str, ast.FunctionDef]:
    tree = ast.parse(path.read_text(encoding="utf-8-sig"))
    return {
        node.name: node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
    }


def _literal_tuple(path: Path, name: str) -> tuple[str, ...]:
    tree = ast.parse(path.read_text(encoding="utf-8-sig"))
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if any(isinstance(target, ast.Name) and target.id == name for target in node.targets):
            value = ast.literal_eval(node.value)
            assert isinstance(value, tuple)
            return value
    raise AssertionError(f"missing {name} in {path}")


def test_phase11b_duplication_budget_is_met_without_relaxation() -> None:
    evidence = _json(EVIDENCE_PATH)
    ledger = _json(LEDGER_PATH)
    migration = _json(AUDIT / "phase13-migration-delta.json")
    refresh = ledger["phase11b_refresh"]
    current = evidence["duplication"]["current"]

    assert evidence["method"]["comparison"] == "EXACT_TOKEN_MATCH"
    assert evidence["method"]["min_tokens"] == ledger["min_tokens"] == 100
    assert current == {
        "duplicate_lines": refresh["combined_duplicate_lines"],
        "total_lines": refresh["combined_total_lines"],
        "coverage_percent": refresh["coverage_percent"],
        "region_count": refresh["region_count"],
    }
    # Phase 11B is an immutable historical snapshot. Later accepted phases add
    # classified regions to the live ledger, whose completeness is enforced by
    # ci/check_duplication.py rather than by rewriting this evidence record.
    assert refresh["region_count"] == 143
    assert refresh["stale_region_count"] == 0
    assert refresh["unclassified_region_count"] == 0
    assert refresh["budget_relaxed"] is False
    assert refresh["coverage_percent"] < refresh["frozen_coverage_percent"]
    assert refresh["frozen_coverage_percent"] == ledger["baseline"]["coverage_percent"]
    assert evidence["duplication"]["frozen_budget"] == {
        "coverage_percent": 10.211512,
        "relaxed": False,
        "met": True,
        "margin_percentage_points": 0.155099,
    }
    assert migration["phase11b_current"]["duplication_refresh"]["budget_met"] is True
    assert migration["phase11b_current"]["evidence"] == str(
        EVIDENCE_PATH.relative_to(ROOT)
    ).replace("\\", "/")


def test_phase11b_ledger_refresh_and_source_snapshots_are_historical() -> None:
    evidence = _json(EVIDENCE_PATH)
    ledger = _json(LEDGER_PATH)

    stale = set(evidence["ledger_refresh"]["stale_regions_removed"])
    assert len(stale) == 13
    assert stale.isdisjoint(ledger["regions"])
    new = evidence["ledger_refresh"]["new_regions_classified"]
    assert [record["region_id"] for record in new] == ["490234d077127261"]
    assert new[0]["category"] == "fixture_boilerplate"
    assert "490234d077127261" not in ledger["regions"]
    assert evidence["ledger_refresh"]["stale_region_count"] == 0
    assert evidence["ledger_refresh"]["unclassified_region_count"] == 0
    assert evidence["source_sha256"] == _PHASE11B_SOURCE_SHA256
    for relative, digest in evidence["source_sha256"].items():
        live = relative.replace(
            "native/channel_native/", "native/channel/"
        ).replace("src/witwin/channel_native/", "witwin/channel/")
        live = _PHASE15_KERNEL_RELOCATIONS.get(live, live)
        assert (ROOT / live).is_file()
        assert re.fullmatch(r"[0-9a-f]{64}", digest)

    phase10b = _json(AUDIT / "phase13-scattering-phase10b-evidence.json")
    phase11a = _json(AUDIT / "phase13-boundary-dedup-phase11a-evidence.json")
    historical = evidence["historical_records"]
    assert historical["phase10b_binding_manifest_sha256"] == phase10b["activation"][
        "binding_manifest_sha256"
    ]
    assert historical["phase11a_binding_manifest_sha256"] == phase11a[
        "binding_manifest"
    ]["current_sha256"]
    assert historical["binding_manifest_changed_in_phase11b"] is False


def test_phase11b_explicit_signatures_and_tu_local_macro_contract_are_preserved() -> None:
    functional_path = ROOT / "witwin/channel/kernels/scattering.py"
    autograd_path = functional_path
    ensemble = _literal_tuple(functional_path, "_CHAIN_ENSEMBLE_PRIMAL_NAMES")
    realization = _literal_tuple(functional_path, "_CHAIN_REALIZATION_PRIMAL_NAMES")
    functional = _module_functions(functional_path)
    autograd = _module_functions(autograd_path)

    for name in (
        "scattering_chain_ensemble_eval",
        "scattering_chain_ensemble_eval_backward",
        "scattering_chain_ensemble_eval_jvp",
    ):
        function = functional[name]
        assert tuple(arg.arg for arg in function.args.args) == ensemble
        assert function.args.vararg is None
    for name in (
        "scattering_chain_realization_eval",
        "scattering_chain_realization_eval_backward",
        "scattering_chain_realization_eval_jvp",
    ):
        function = functional[name]
        assert tuple(arg.arg for arg in function.args.args) == realization
        assert function.args.vararg is None
    assert tuple(
        arg.arg for arg in autograd["scattering_chain_ensemble_eval_ad"].args.args
    ) == ensemble
    assert tuple(
        arg.arg for arg in autograd["scattering_chain_realization_eval_ad"].args.args
    ) == realization
    assert autograd["scattering_chain_ensemble_eval_ad"].args.vararg is None
    assert autograd["scattering_chain_realization_eval_ad"].args.vararg is None

    macros = {
        "native/channel/kernels/bdpt_connect.cu": (
            "CHANNEL_BDPT_CHECK_CONNECTION_SAMPLE_TENSORS",
            "CHANNEL_BDPT_CHECK_CONNECTION_SAMPLE_ROWS",
            "CHANNEL_BDPT_CONNECTION_OUTPUT_POINTERS",
        ),
        "native/channel/kernels/diffraction.cu": (
            "CHANNEL_DIFFRACTION_CHECK_STATE_PACK_TENSORS",
            "CHANNEL_DIFFRACTION_CHECK_STATE_PACK_POWER",
            "CHANNEL_DIFFRACTION_CHECK_STATE_PACK_SHAPES",
            "CHANNEL_DIFFRACTION_ALLOCATE_STATE_PACK",
            "CHANNEL_DIFFRACTION_STATE_PACK_INPUT_POINTERS",
            "CHANNEL_DIFFRACTION_STATE_PACK_OUTPUT_POINTERS",
            "CHANNEL_DIFFRACTION_STATE_PACK_RESULTS",
        ),
        "native/channel/kernels/los_consumer.cu": (
            "CHANNEL_LOS_CHECK_VISIBILITY_APPLICATION",
            "CHANNEL_LOS_VISIBILITY_LAUNCH_ARGUMENTS",
        ),
        "native/channel/kernels/reflection.cu": (
            "CHANNEL_REFLECTION_PREPARE_LAUNCH_INPUTS",
            "CHANNEL_REFLECTION_LAUNCH_INPUT_PREFIX",
        ),
    }
    for relative, names in macros.items():
        source = (ROOT / relative).read_text(encoding="utf-8-sig")
        for name in names:
            assert source.count(f"#define {name}") == 1
            assert source.count(f"#undef {name}") == 1

    invariants = _json(EVIDENCE_PATH)["invariants"]
    assert all(value is False for value in invariants.values())


def test_phase11_live_surface_has_no_version_suffixed_wip_boundary_name() -> None:
    versioned_boundary = re.compile(
        r"integration[_-]v\d+|rayd\.torch\.integration\.v\d+",
        re.IGNORECASE,
    )
    unexpected_boundary_names: list[str] = []
    observed_triangle_v2: set[tuple[str, str]] = set()
    unexpected_v2: list[str] = []

    for path in _live_surface_files():
        relative = path.relative_to(ROOT).as_posix()
        for number, line in enumerate(
            path.read_text(encoding="utf-8-sig").splitlines(), start=1
        ):
            if versioned_boundary.search(line):
                unexpected_boundary_names.append(f"{relative}:{number}: {line.strip()}")
            if relative.startswith(("witwin/", "native/")) and re.search(
                r"\bv2\b", line, re.IGNORECASE
            ):
                record = (relative, line.strip())
                if record in _TRIANGLE_VERTEX_V2_LINES:
                    observed_triangle_v2.add(record)
                else:
                    unexpected_v2.append(f"{relative}:{number}: {line.strip()}")

    assert unexpected_boundary_names == []
    assert unexpected_v2 == []
    assert observed_triangle_v2 == _TRIANGLE_VERTEX_V2_LINES


def test_phase11_release_record_matches_live_governance_and_is_honest() -> None:
    evidence = _json(RELEASE_EVIDENCE_PATH)
    lock = _json(ROOT / "dependencies/rayd.lock.json")
    inventory = _json(AUDIT / "phase13-current-native-owner-inventory.json")
    migration = _json(AUDIT / "phase13-migration-delta.json")

    boundary = evidence["implementation_boundary"]
    assert boundary["channel_commit"] == PHASE11_IMPLEMENTATION_BOUNDARY
    assert re.fullmatch(r"[0-9a-f]{40}", boundary["rayd_commit"])
    assert re.fullmatch(r"[0-9a-f]{64}", boundary["rayd_lock_sha256"])
    assert boundary["integration_header"].endswith("/integration.h")
    assert re.fullmatch(r"[0-9a-f]{64}", boundary["integration_header_sha256"])
    assert boundary["integration_identity"] == "rayd.torch.integration"
    assert isinstance(boundary["integration_api_version"], int)
    assert isinstance(evidence["release_claim"], bool)
    verified = evidence["verified"]
    assert isinstance(verified["binding_count"], int)
    assert verified["binding_count"] > 0
    assert verified["binding_count"] == sum(verified["owner_counts"].values())
    assert set(verified["owner_counts"]) == {
        "RayD",
        "layered",
        "Channel Native",
    }
    for key in (
        "native_binding_manifest_sha256",
        "contract_coverage_manifest_sha256",
        "agents_claude_sha256",
    ):
        assert re.fullmatch(r"[0-9a-f]{64}", verified[key])
    historical_path = str(RELEASE_EVIDENCE_PATH.relative_to(ROOT)).replace(
        "\\", "/"
    )
    superseding_path = str(
        STABLE_RECOVERY_EVIDENCE_PATH.relative_to(ROOT)
    ).replace("\\", "/")
    for current_record in (
        inventory["phase11_release_governance_closure"],
        migration["phase11_release_governance_closure"],
    ):
        assert current_record["historical_evidence"] == historical_path
        assert current_record["superseding_evidence"] == superseding_path
        assert current_record["pending"] == []
    pending = evidence["pending_final_acceptance"]
    if evidence["release_claim"]:
        assert evidence["status"] == "release accepted"
        assert pending == []
        assert boundary["rayd_commit"] == lock["commit"]
        assert boundary["rayd_lock_sha256"] == _sha256(
            ROOT / "dependencies/rayd.lock.json"
        )
        assert boundary["integration_header"] == lock["integration_abi"]["path"]
        assert (
            boundary["integration_header_sha256"]
            == lock["integration_abi"]["sha256"]
        )
        assert verified["binding_count"] == len(
            _json(ROOT / "ci/native-binding-manifest.json")["symbols"]
        )
        assert verified["owner_counts"] == {
            "RayD": inventory["counts"]["rayd_numerical"],
            "layered": inventory["counts"]["layered"],
            "Channel Native": inventory["counts"]["channel_numerical"],
        }
        assert verified["native_binding_manifest_sha256"] == _sha256(
            ROOT / "ci/native-binding-manifest.json"
        )
        assert verified["contract_coverage_manifest_sha256"] == _sha256(
            ROOT / "ci/contract-coverage-manifest.json"
        )
        assert (ROOT / "AGENTS.md").read_bytes() == (ROOT / "CLAUDE.md").read_bytes()
        assert verified["agents_claude_sha256"] == _sha256(ROOT / "AGENTS.md")
    else:
        assert evidence["status"] == (
            "governance complete; final release evidence pending"
        )
        assert set(pending) == _EXPECTED_PENDING_FINAL_ACCEPTANCE
