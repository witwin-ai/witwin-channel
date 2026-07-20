from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
from collections import Counter
from pathlib import Path

from tools.refactor_baseline import binding_manifest


ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "docs/dev/audit"
RAYD_ROOT = Path(os.environ.get("RAYD_SOURCE_DIR", ROOT.parent.parent / "RayDi"))
RAYD_COMMIT = "768b96e42a95f70c32d55f98a72000085317e288"
INTEGRATION_SHA256 = (
    "0608bfbaf022379bc03442f9baa777ec05cfe3f6ab9b964e2385ec12a7b6c654"
)
TYPED_SHA256 = (
    "ac95c418860d109aeaa96623131592e4df8887992e5fc25ecab71b4ddbf1f55b"
)
SHARED_SHA256 = (
    "38ea9be424640301a88a97bccca9ab4bc599191ecfb0b259881ef6a300c96e38"
)
IDENTITY = (
    "rayd.torch.integration.v2.20260719.rf-transmission-sequence."
    "pure-wedge-diffraction.scattering-table-single-bounce.scattering-chains"
)
CHAIN_SYMBOLS = {
    "scattering_chain_ensemble_eval",
    "scattering_chain_ensemble_eval_backward",
    "scattering_chain_ensemble_eval_jvp",
    "scattering_chain_realization_eval",
    "scattering_chain_realization_eval_backward",
    "scattering_chain_realization_eval_jvp",
}
DELETED = {
    "native/channel_native/kernels/scattering_chain_ensemble.cu",
    "native/channel_native/kernels/scattering_chain_ensemble_ad.cu",
    "native/channel_native/kernels/scattering_chain_realization.cu",
    "native/channel_native/kernels/scattering_chain_realization_ad.cu",
}


def _json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_phase10b_pin_owner_counts_and_manifests_are_atomic() -> None:
    lock = _json(ROOT / "dependencies/rayd.lock.json")
    inventory = _json(AUDIT / "phase13-current-native-owner-inventory.json")
    migration = _json(AUDIT / "phase13-migration-delta.json")
    evidence = _json(AUDIT / "phase13-scattering-phase10b-evidence.json")
    graph = _json(AUDIT / "phase13-shared-rf-dependency-graph.json")

    assert lock["commit"] == RAYD_COMMIT
    assert lock["integration_abi"]["sha256"] == INTEGRATION_SHA256
    assert {
        inventory["phase10b_scattering_chains"]["rayd_commit"],
        migration["phase10b_current"]["rayd_commit"],
        evidence["activation"]["rayd_commit"],
        graph["phase10b_activation"]["rayd_commit"],
    } == {RAYD_COMMIT}
    assert inventory["counts"] == {
        "bindings": 202,
        "rayd_numerical": 43,
        "layered": 2,
        "channel_numerical": 157,
    }
    assert inventory["current_subphase"] == migration["current_subphase"] == "10B"
    assert Counter(row["numerical_owner"] for row in inventory["symbols"]) == {
        "RayD": 43,
        "Channel operation / RayD primitives": 2,
        "Channel Native": 157,
    }
    owners = {row["symbol"]: row["numerical_owner"] for row in inventory["symbols"]}
    assert all(owners[symbol] == "RayD" for symbol in CHAIN_SYMBOLS)

    native_manifest = ROOT / "ci/native-binding-manifest.json"
    coverage_manifest = ROOT / "ci/contract-coverage-manifest.json"
    assert _json(native_manifest) == binding_manifest(ROOT)
    assert len(_json(native_manifest)["symbols"]) == 202
    assert len(_json(coverage_manifest)["native_bindings"]) == 202
    assert _sha256(native_manifest) == evidence["activation"]["binding_manifest_sha256"]
    assert _sha256(coverage_manifest) == evidence["activation"][
        "contract_coverage_manifest_sha256"
    ]


def test_phase10b_rayd_identity_sources_and_direct_contract_are_locked() -> None:
    evidence = _json(AUDIT / "phase13-scattering-phase10b-evidence.json")
    activation = evidence["activation"]
    integration = RAYD_ROOT / activation["integration_header"]
    typed = RAYD_ROOT / activation["typed_header"]
    shared = RAYD_ROOT / activation["shared_table_header"]

    assert _sha256(integration) == INTEGRATION_SHA256
    assert _sha256(typed) == TYPED_SHA256
    assert _sha256(shared) == SHARED_SHA256
    assert IDENTITY in integration.read_text(encoding="utf-8-sig")
    for record in evidence["rayd_sources"]:
        assert _sha256(RAYD_ROOT / record["path"]) == record["sha256"]
    direct = evidence["direct_contract_coverage"]
    assert _sha256(RAYD_ROOT / direct["test"]) == direct["test_sha256"]
    assert direct["ctest_result"] == "4/4 passed"
    assert len(direct["depth8_positive_coverage"]) == 2
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=RAYD_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    assert head == RAYD_COMMIT


def test_phase10b_channel_is_typed_facade_without_duplicate_or_fallback() -> None:
    materials = (ROOT / "native/channel_native/binding/materials.cpp").read_text(
        encoding="utf-8-sig"
    )
    event_source = (ROOT / "native/channel_native/kernels/scattering.cu").read_text(
        encoding="utf-8-sig"
    )
    cmake = (ROOT / "CMakeLists.txt").read_text(encoding="utf-8-sig")

    assert all(not (ROOT / path).exists() for path in DELETED)
    assert all(path not in cmake for path in DELETED)
    assert "<<<" not in materials
    assert "scattering_event_kernel<<<" in event_source
    assert all(materials.count(f"rayd::torch::{symbol}(") == 1 for symbol in CHAIN_SYMBOLS)
    for forbidden in (
        "scattering_chain_ensemble_kernel",
        "scattering_chain_realization_rows_kernel",
        "cudaDeviceSynchronize",
        ".cpu()",
    ):
        assert forbidden not in materials


def test_phase10b_compile_launch_codegen_and_dependency_contracts_are_frozen() -> None:
    evidence = _json(AUDIT / "phase13-scattering-phase10b-evidence.json")
    graph = _json(AUDIT / "phase13-shared-rf-dependency-graph.json")
    rayd_cmake = (RAYD_ROOT / "backends/torch/CMakeLists.txt").read_text(
        encoding="utf-8-sig"
    )
    fmad_blocks = [
        block
        for block in re.findall(
            r"set_source_files_properties\((.*?)\)", rayd_cmake, re.DOTALL
        )
        if "--fmad=false" in block
    ]

    assert len(fmad_blocks) == 1
    for source in (
        "scattering_chain_ensemble.cu",
        "scattering_chain_ensemble_ad.cu",
        "scattering_chain_realization.cu",
        "scattering_chain_realization_ad.cu",
    ):
        assert source in fmad_blocks[0]
    launch = evidence["launch_contract"]
    assert launch["ensemble"] == {
        "primal_launches": 1,
        "backward_launches": 1,
        "jvp_launches": 1,
    }
    assert launch["realization"]["primal_launches"] == 2
    assert launch["realization"]["backward_launches"] == 1
    assert launch["realization"]["jvp_launches"] == 2
    assert launch["zero_row_launches"] == launch["explicit_synchronizations"] == 0
    assert evidence["sm120_codegen"]["exact_match_to_channel_baseline"] is True
    assert all(len(item["sha256"]) == 64 for item in evidence["sm120_codegen"]["families"].values())
    assert all(
        not (
            edge["from"].startswith("RayD:")
            and edge["to"].startswith("native/channel_native/")
        )
        for edge in graph["edges"]
    )


def test_phase10b_ledgers_ad_truth_and_duplication_budget_are_closed() -> None:
    duplication = _json(AUDIT / "duplication-classification.json")
    ledger = _json(AUDIT / "phase13-symbol-delta-ledger.json")
    scattering = _json(AUDIT / "phase13-scattering-bindings.json")
    evidence = _json(AUDIT / "phase13-scattering-phase10b-evidence.json")
    actions = {row["symbol"]: row["status"] for row in ledger["actions"]}
    contracts = {row["symbol"]: row for row in scattering["contracts"]}

    assert all(actions[symbol] == "applied in Phase 10B" for symbol in CHAIN_SYMBOLS)
    assert all(contracts[symbol]["current_numerical_owner"] == "RayD" for symbol in CHAIN_SYMBOLS)
    assert all(contracts[symbol]["rayd_direct_test"] for symbol in CHAIN_SYMBOLS)
    assert scattering["phase10b_activation"]["ad_contract"] == {
        "ensemble_geometry": "JVP-only; VJP fails loudly",
        "realization_geometry": "VJP and JVP supported",
    }
    refresh = duplication["phase10b_refresh"]
    assert refresh["region_count"] == len(duplication["regions"])
    assert refresh["stale_region_count"] == refresh["unclassified_region_count"] == 0
    assert refresh["budget_relaxed"] is False
    assert refresh["coverage_percent"] > duplication["baseline"]["coverage_percent"]
    assert evidence["duplication_refresh"] == {
        "duplicate_lines": refresh["combined_duplicate_lines"],
        "total_lines": refresh["combined_total_lines"],
        "duplicate_coverage_percent": refresh["coverage_percent"],
        "frozen_budget_percent": refresh["frozen_coverage_percent"],
        "budget_relaxed": False,
        "status": (
            "all regions classified; implementation complete; frozen duplication "
            "acceptance remains a Phase 11 blocker"
        ),
    }
    assert (ROOT / "AGENTS.md").read_bytes() == (ROOT / "CLAUDE.md").read_bytes()
