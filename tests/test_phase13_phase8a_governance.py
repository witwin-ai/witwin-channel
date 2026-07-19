from __future__ import annotations

import hashlib
import json
import os
import subprocess
from collections import Counter
from pathlib import Path

from tools.refactor_baseline import binding_manifest


ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "docs/dev/audit"
RAYD_ROOT = Path(os.environ.get("RAYD_SOURCE_DIR", ROOT.parent.parent / "RayDi"))
RAYD_COMMIT = "11e72526cdddf669678975c8921a9d44c6504e20"
INTEGRATION_SHA256 = (
    "7a2b68f459e7e981a23735271eff2844fe0483d119cf514d59d2032d11be5aef"
)
INTEGRATION_IDENTITY = (
    "rayd.torch.integration.v2.20260719.rf-transmission-sequence."
    "pure-wedge-diffraction"
)
MANIFEST_SHA256 = (
    "fd42f559bebe3933360312a35fa839b72f94faeef49c88c28e06a83e896994a4"
)
NUMERICAL_REGION_SHA256 = (
    "09b4788ce1c39bb51a1c76f1a6f95269ae65cb8b04a501d174f355bd7bf53f3c"
)
PURE_WEDGE = {
    "field_diffraction_wedge",
    "field_diffraction_wedge_backward",
    "field_diffraction_wedge_jvp",
}


def _json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_phase8a_pin_manifest_and_owner_counts_are_atomic() -> None:
    lock = _json(ROOT / "dependencies/rayd.lock.json")
    inventory = _json(AUDIT / "phase13-current-native-owner-inventory.json")
    migration = _json(AUDIT / "phase13-migration-delta.json")
    evidence = _json(AUDIT / "phase13-diffraction-phase8a-evidence.json")
    graph = _json(AUDIT / "phase13-shared-rf-dependency-graph.json")
    matrix = _json(AUDIT / "phase13-diffraction-family-matrix.json")

    assert lock["commit"] == RAYD_COMMIT
    assert lock["integration_abi"]["sha256"] == INTEGRATION_SHA256  # type: ignore[index]
    assert {
        inventory["phase8a_pure_wedge_diffraction"]["rayd_commit"],  # type: ignore[index]
        migration["phase8a_current"]["rayd_commit"],  # type: ignore[index]
        evidence["activation_pin"]["rayd_commit"],  # type: ignore[index]
        graph["phase8a_activation"]["rayd_commit"],  # type: ignore[index]
        matrix["phase8a_activation"]["rayd_commit"],  # type: ignore[index]
    } == {RAYD_COMMIT}
    assert inventory["counts"] == {
        "bindings": 202,
        "rayd_numerical": 26,
        "layered": 2,
        "channel_numerical": 174,
    }
    assert inventory["current_subphase"] == migration["current_subphase"] == "8A"

    manifest = ROOT / "ci/native-binding-manifest.json"
    assert _sha256(manifest) == MANIFEST_SHA256
    assert _json(manifest) == binding_manifest(ROOT)
    owners = {
        row["symbol"]: row["numerical_owner"]
        for row in inventory["symbols"]  # type: ignore[index]
    }
    assert all(owners[symbol] == "RayD" for symbol in PURE_WEDGE)
    assert Counter(
        row["numerical_owner"] for row in inventory["symbols"]  # type: ignore[index]
    ) == {
        "RayD": 26,
        "Channel operation / RayD primitives": 2,
        "Channel Native": 174,
    }


def test_phase8a_rayd_identity_sources_and_direct_test_are_locked() -> None:
    evidence = _json(AUDIT / "phase13-diffraction-phase8a-evidence.json")
    pin = evidence["activation_pin"]  # type: ignore[index]
    candidate = evidence["rayd_candidate"]  # type: ignore[index]
    header = RAYD_ROOT / pin["integration_header"]
    typed_header = RAYD_ROOT / candidate["typed_header"]["path"]  # type: ignore[index]
    source = RAYD_ROOT / candidate["numerical_source"]["path"]  # type: ignore[index]
    direct_test = RAYD_ROOT / candidate["direct_contract_test"]["path"]  # type: ignore[index]

    assert _sha256(header) == pin["integration_header_sha256"] == INTEGRATION_SHA256
    assert INTEGRATION_IDENTITY in header.read_text(encoding="utf-8-sig")
    assert _sha256(typed_header) == candidate["typed_header"]["sha256"]  # type: ignore[index]
    assert _sha256(source) == candidate["numerical_source"]["sha256"]  # type: ignore[index]
    assert _sha256(direct_test) == candidate["direct_contract_test"]["sha256"]  # type: ignore[index]
    assert candidate["direct_contract_test"]["status"] == "passed; full RayD CTest 2/2"  # type: ignore[index]
    assert evidence["exactness_contract"]["numerical_region_sha256"] == (  # type: ignore[index]
        NUMERICAL_REGION_SHA256
    )
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=RAYD_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    assert head == RAYD_COMMIT


def test_phase8a_channel_is_a_typed_facade_without_numerical_fallback() -> None:
    fields = (ROOT / "native/channel_native/binding/fields.cpp").read_text(
        encoding="utf-8-sig"
    )
    cmake = (ROOT / "CMakeLists.txt").read_text(encoding="utf-8-sig")
    removed = ROOT / "native/channel_native/kernels/field_wedge_ad_diffraction.cu"

    assert not removed.exists()
    assert "field_wedge_ad_diffraction.cu" not in cmake
    assert "CHANNEL_NATIVE_FAST_MATH_WEDGE_TU" not in cmake
    assert "<<<" not in fields
    for symbol in PURE_WEDGE:
        assert fields.count(f"rayd::torch::{symbol}(") == 1

    migration = _json(AUDIT / "phase13-migration-delta.json")["phase8a_current"]
    assert migration["deleted_source_sha256"] == (  # type: ignore[index]
        "68ec3fe180cd900834f0263969ee75d54764ad014e5d22b7c0b57822ea8e975b"
    )
    assert len(migration["approved_phase9_body_hash_deletions"]) == 16  # type: ignore[index]


def test_phase8a_launch_compile_and_dependency_boundaries_are_frozen() -> None:
    evidence = _json(AUDIT / "phase13-diffraction-phase8a-evidence.json")
    graph = _json(AUDIT / "phase13-shared-rf-dependency-graph.json")
    rayd_source = (
        RAYD_ROOT / "backends/torch/src/torch_ext/rf/diffraction_wedge.cu"
    ).read_text(encoding="utf-8-sig")
    rayd_cmake = (RAYD_ROOT / "backends/torch/CMakeLists.txt").read_text(
        encoding="utf-8-sig"
    )

    assert rayd_source.count("<<<") == 3
    assert "cudaDeviceSynchronize" not in rayd_source
    assert "cudaStreamSynchronize" not in rayd_source
    assert evidence["launch_contract"]["active_entry_launch_count"] == {  # type: ignore[index]
        "field_diffraction_wedge": 1,
        "field_diffraction_wedge_backward": 1,
        "field_diffraction_wedge_jvp": 1,
    }
    assert "src/torch_ext/rf/diffraction_wedge.cu" in rayd_cmake
    assert 'COMPILE_OPTIONS "$<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>"' in rayd_cmake
    for path in evidence["compile_contract"]["precise_channel_families"]:  # type: ignore[index]
        assert (ROOT / path).is_file()

    codegen = evidence["codegen_resource_contract"]
    assert codegen["normalized_ptx_equal"] is True  # type: ignore[index]
    assert codegen["normalized_sass_equal"] is True  # type: ignore[index]
    assert codegen["resource_usage_equal"] is True  # type: ignore[index]
    assert codegen["kernels"] == {  # type: ignore[index]
        "diffraction_wedge_forward_kernel": {
            "registers_per_thread": 127,
            "stack_bytes": 0,
            "shared_bytes": 0,
            "local_bytes": 0,
            "constant_bytes": 1136,
            "threads_per_block": 128,
        },
        "diffraction_wedge_backward_kernel": {
            "registers_per_thread": 254,
            "stack_bytes": 672,
            "shared_bytes": 0,
            "local_bytes": 0,
            "constant_bytes": 1240,
            "threads_per_block": 128,
        },
        "diffraction_wedge_jvp_kernel": {
            "registers_per_thread": 254,
            "stack_bytes": 672,
            "shared_bytes": 0,
            "local_bytes": 0,
            "constant_bytes": 1240,
            "threads_per_block": 128,
        },
    }
    packaging = evidence["copy_memory_packaging_contract"]
    assert packaging["host_device_copy_count_delta"] == 0  # type: ignore[index]
    assert packaging["explicit_sync_count_delta"] == 0  # type: ignore[index]
    assert packaging["persistent_tape_bytes_delta"] == 0  # type: ignore[index]
    assert packaging["materialized_intermediate_bytes_delta"] == 0  # type: ignore[index]
    assert packaging["rayd_python_extension_built_or_imported"] is False  # type: ignore[index]

    assert all(
        not (
            edge["from"].startswith("RayD:")
            and edge["to"].startswith("native/channel_native/")
        )
        for edge in graph["edges"]  # type: ignore[index]
    )


def test_phase8a_ledgers_and_guardrails_are_closed() -> None:
    duplication = _json(AUDIT / "duplication-classification.json")
    matrix = _json(AUDIT / "phase13-diffraction-family-matrix.json")
    ledger = _json(AUDIT / "phase13-symbol-delta-ledger.json")
    refresh = duplication["phase8a_refresh"]  # type: ignore[index]

    assert refresh == {
        "combined_duplicate_lines": 10661,
        "combined_total_lines": 84347,
        "coverage_percent": 12.639454,
        "frozen_coverage_percent": 10.211512,
        "note": (
            "Phase 8A removed the Channel pure-wedge numerical owner, pruned "
            "its stale binding/source region, and classified the typed RayD "
            "adapter packing regions without weakening the frozen maintenance budget."
        ),
        "region_count": 179,
        "status": (
            "all current regions classified; frozen coverage budget remains "
            "unchanged and remains a known nightly blocker"
        ),
    }
    assert len(duplication["regions"]) == 179
    assert duplication["baseline"]["coverage_percent"] == 10.211512  # type: ignore[index]
    assert matrix["phase8a_activation"]["current_numerical_owner"] == "RayD"  # type: ignore[index]
    pure = next(
        family
        for family in matrix["families"]  # type: ignore[index]
        if family["family_id"] == "pure-wedge-fixed-winner-field"
    )
    assert pure["phase7_current_owner"] == "Channel Native"
    actions = {
        row["symbol"]: row["status"] for row in ledger["actions"]  # type: ignore[index]
    }
    assert all(actions[symbol] == "applied in Phase 8A" for symbol in PURE_WEDGE)
    assert (ROOT / "AGENTS.md").read_bytes() == (ROOT / "CLAUDE.md").read_bytes()
