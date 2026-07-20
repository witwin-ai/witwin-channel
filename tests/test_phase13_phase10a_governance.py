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
RAYD_COMMIT = "4577e744adfe8665f7817e3aff5e8e533ec896e7"
INTEGRATION_SHA256 = (
    "9f95ad9e8e3b790d00f8e762a3e6a09252d46afb65bfc3aba7c42325836cb1fb"
)
TYPED_SHA256 = (
    "66d75a20be16057f03cdfb79e3b9dcc85cacec79b555cd73b019259aa510262a"
)
SHARED_SHA256 = (
    "38ea9be424640301a88a97bccca9ab4bc599191ecfb0b259881ef6a300c96e38"
)
INTEGRATION_IDENTITY = (
    "rayd.torch.integration.v2.20260719.rf-transmission-sequence."
    "pure-wedge-diffraction.scattering-table-single-bounce"
)
PHASE10A = {
    "scattering_table_eval",
    "scattering_table_eval_backward",
    "scattering_table_eval_jvp",
    "scattering_table_sample",
    "scattering_table_pdf",
    "scattering_ensemble_eval",
    "scattering_ensemble_eval_backward",
    "scattering_ensemble_eval_jvp",
    "scattering_patch_integral_eval",
    "scattering_patch_integral_eval_backward",
    "scattering_patch_integral_eval_jvp",
}
DELETED = {
    "native/channel_native/kernels/scattering_table_eval_ad.cu",
    "native/channel_native/kernels/scattering_ensemble.cu",
    "native/channel_native/kernels/scattering_ensemble_ad.cu",
    "native/channel_native/kernels/scattering_patch_integral.cu",
    "native/channel_native/kernels/scattering_patch_integral_ad.cu",
    "native/channel_native/kernels/scattering_table.cuh",
}


def _json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _normalized_sha256(path: Path) -> str:
    data = path.read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return hashlib.sha256(data).hexdigest()


def test_phase10a_pin_owner_counts_and_manifests_are_atomic() -> None:
    lock = _json(ROOT / "dependencies/rayd.lock.json")
    inventory = _json(AUDIT / "phase13-current-native-owner-inventory.json")
    migration = _json(AUDIT / "phase13-migration-delta.json")
    evidence = _json(AUDIT / "phase13-scattering-phase10a-evidence.json")
    graph = _json(AUDIT / "phase13-shared-rf-dependency-graph.json")

    assert lock["commit"] == RAYD_COMMIT
    assert lock["integration_abi"]["sha256"] == INTEGRATION_SHA256  # type: ignore[index]
    assert {
        inventory["phase10a_scattering_table_single_bounce"]["rayd_commit"],  # type: ignore[index]
        migration["phase10a_current"]["rayd_commit"],  # type: ignore[index]
        evidence["activation_pin"]["rayd_commit"],  # type: ignore[index]
        graph["phase10a_activation"]["rayd_commit"],  # type: ignore[index]
    } == {RAYD_COMMIT}
    assert inventory["counts"] == {
        "bindings": 202,
        "rayd_numerical": 37,
        "layered": 2,
        "channel_numerical": 163,
    }
    assert inventory["current_subphase"] == migration["current_subphase"] == "10A"
    assert Counter(
        row["numerical_owner"] for row in inventory["symbols"]  # type: ignore[index]
    ) == {
        "RayD": 37,
        "Channel operation / RayD primitives": 2,
        "Channel Native": 163,
    }
    owners = {
        row["symbol"]: row["numerical_owner"]
        for row in inventory["symbols"]  # type: ignore[index]
    }
    assert all(owners[symbol] == "RayD" for symbol in PHASE10A)

    native_manifest = ROOT / "ci/native-binding-manifest.json"
    coverage_manifest = ROOT / "ci/contract-coverage-manifest.json"
    assert _json(native_manifest) == binding_manifest(ROOT)
    assert len(_json(native_manifest)["symbols"]) == 202
    assert len(_json(coverage_manifest)["native_bindings"]) == 202
    assert _sha256(native_manifest) == evidence["activation_pin"][  # type: ignore[index]
        "binding_manifest_sha256"
    ]
    assert _sha256(coverage_manifest) == evidence["activation_pin"][  # type: ignore[index]
        "contract_coverage_manifest_sha256"
    ]


def test_phase10a_rayd_identity_headers_sources_and_direct_test_are_locked() -> None:
    evidence = _json(AUDIT / "phase13-scattering-phase10a-evidence.json")
    pin = evidence["activation_pin"]  # type: ignore[index]
    candidate = evidence["rayd_candidate"]  # type: ignore[index]
    integration = RAYD_ROOT / pin["integration_header"]
    typed = RAYD_ROOT / pin["typed_header"]
    shared = RAYD_ROOT / pin["shared_table_header"]

    assert _normalized_sha256(integration) == INTEGRATION_SHA256
    assert _normalized_sha256(typed) == TYPED_SHA256
    assert _normalized_sha256(shared) == SHARED_SHA256
    assert INTEGRATION_IDENTITY in integration.read_text(encoding="utf-8-sig")
    for relative_path, expected in candidate["numerical_sources"].items():  # type: ignore[union-attr]
        assert _normalized_sha256(RAYD_ROOT / relative_path) == expected
    direct = candidate["direct_contract_test"]  # type: ignore[index]
    assert _normalized_sha256(RAYD_ROOT / direct["path"]) == direct["sha256"]
    assert direct["status"] == "passed; full RayD CTest 3/3"
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=RAYD_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    assert head == RAYD_COMMIT


def test_phase10a_channel_is_typed_facade_without_duplicate_or_fallback() -> None:
    materials = (ROOT / "native/channel_native/binding/materials.cpp").read_text(
        encoding="utf-8-sig"
    )
    event_source = (
        ROOT / "native/channel_native/kernels/scattering.cu"
    ).read_text(encoding="utf-8-sig")
    cmake = (ROOT / "CMakeLists.txt").read_text(encoding="utf-8-sig")

    assert all(not (ROOT / path).exists() for path in DELETED)
    assert all(path not in cmake for path in DELETED)
    assert "<<<" not in materials
    assert "scattering_event_kernel<<<" in event_source
    for forbidden in (
        "scattering_eval_kernel",
        "scattering_pdf_kernel",
        "scattering_sample_kernel",
        "scattering_table.cuh",
        "rayd::torch",
    ):
        assert forbidden not in event_source
    for symbol in PHASE10A:
        assert materials.count(f"rayd::torch::{symbol}(") == 1
    for path in (
        "native/channel_native/kernels/scattering_chain_ensemble.cu",
        "native/channel_native/kernels/scattering_chain_ensemble_ad.cu",
        "native/channel_native/kernels/scattering_chain_realization.cu",
        "native/channel_native/kernels/scattering_chain_realization_ad.cu",
    ):
        assert (ROOT / path).is_file()
    for path in (
        "native/channel_native/kernels/scattering_chain_ensemble.cu",
        "native/channel_native/kernels/scattering_chain_ensemble_ad.cu",
    ):
        assert "<rayd/shared/rf/scattering_table.cuh>" in (
            ROOT / path
        ).read_text(encoding="utf-8-sig")
    assert "scattering_table.cuh" not in (
        ROOT / "native/channel_native/kernels/kirchhoff_table_ad.cu"
    ).read_text(encoding="utf-8-sig")


def test_phase10a_compile_launch_and_dependency_contracts_are_frozen() -> None:
    evidence = _json(AUDIT / "phase13-scattering-phase10a-evidence.json")
    graph = _json(AUDIT / "phase13-shared-rf-dependency-graph.json")
    channel_cmake = (ROOT / "CMakeLists.txt").read_text(encoding="utf-8-sig")
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
        "scattering_table_eval_ad.cu",
        "scattering_ensemble.cu",
        "scattering_ensemble_ad.cu",
        "scattering_patch_integral.cu",
        "scattering_patch_integral_ad.cu",
    ):
        assert source in fmad_blocks[0]
    assert "scattering.cu" not in fmad_blocks[0]
    assert "scattering.cu" not in channel_cmake.split(
        'PROPERTIES COMPILE_OPTIONS "--fmad=false")', 1
    )[0].split("set_source_files_properties(")[-1]

    launches = evidence["launch_contract"]  # type: ignore[index]
    assert launches["active_entry_launch_count"] == {
        symbol: 2 if symbol.endswith("patch_integral_eval") or symbol.endswith("eval_jvp") and "patch_integral" in symbol else 1
        for symbol in PHASE10A
    }
    assert launches["zero_row_launch_count"] == 0
    assert launches["explicit_sync_count"] == 0
    assert launches["persistent_tape"] is False
    assert evidence["codegen_resource_contract"]["normalized_sass_equal"] is True  # type: ignore[index]
    assert evidence["codegen_resource_contract"]["resource_usage_equal"] is True  # type: ignore[index]
    assert all(
        not (
            edge["from"].startswith("RayD:")
            and edge["to"].startswith("native/channel_native/")
        )
        for edge in graph["edges"]  # type: ignore[index]
    )


def test_phase10a_ledgers_duplication_and_guardrails_are_closed() -> None:
    duplication = _json(AUDIT / "duplication-classification.json")
    ledger = _json(AUDIT / "phase13-symbol-delta-ledger.json")
    scattering = _json(AUDIT / "phase13-scattering-bindings.json")
    refresh = duplication["phase10a_refresh"]  # type: ignore[index]

    assert refresh["combined_duplicate_lines"] == 9730
    assert refresh["combined_total_lines"] == 81675
    assert refresh["coverage_percent"] == 11.91307
    assert refresh["frozen_coverage_percent"] == 10.211512
    assert refresh["region_count"] == len(duplication["regions"]) == 169
    assert duplication["baseline"]["coverage_percent"] == 10.211512  # type: ignore[index]
    actions = {
        row["symbol"]: row["status"] for row in ledger["actions"]  # type: ignore[index]
    }
    assert all(actions[symbol] == "applied in Phase 10A" for symbol in PHASE10A)
    current = {
        row["symbol"]: row for row in scattering["contracts"]  # type: ignore[index]
    }
    assert all(current[symbol]["current_numerical_owner"] == "RayD" for symbol in PHASE10A)
    assert all(current[symbol]["rayd_direct_test"] for symbol in PHASE10A)
    assert (ROOT / "AGENTS.md").read_bytes() == (ROOT / "CLAUDE.md").read_bytes()
