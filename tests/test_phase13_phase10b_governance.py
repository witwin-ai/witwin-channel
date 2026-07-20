from __future__ import annotations

import hashlib
import json
import os
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "docs/dev/audit"
RAYD_ROOT = Path(os.environ.get("RAYD_SOURCE_DIR", ROOT.parent.parent / "RayDi"))
PHASE10B_RAYD_COMMIT = "768b96e42a95f70c32d55f98a72000085317e288"
INTEGRATION_SHA256 = (
    "0608bfbaf022379bc03442f9baa777ec05cfe3f6ab9b964e2385ec12a7b6c654"
)
TYPED_SHA256 = (
    "ac95c418860d109aeaa96623131592e4df8887992e5fc25ecab71b4ddbf1f55b"
)
SHARED_SHA256 = (
    "38ea9be424640301a88a97bccca9ab4bc599191ecfb0b259881ef6a300c96e38"
)
PHASE10B_BINDING_MANIFEST_SHA256 = (
    "264bddd77eed701b951bab1bb03185ba8ef53c0e6953c1f1ed3a0a1c12405b71"
)
PHASE10B_COVERAGE_MANIFEST_SHA256 = (
    "a04d12baff4aeb2fdbee09cc40f30dbca9bb7588680c1317423f6e34e856071f"
)
PHASE10B_RAYD_SOURCE_SHA256 = {
    "backends/torch/src/torch_ext/rf/scattering_chain_checks.h": (
        "4f61082059d08112d675613e2e0ff0d8b7489753ffb96aec152aa17ac2409b73"
    ),
    "backends/torch/src/torch_ext/rf/scattering_chain_ad_common.cuh": (
        "2551c33533dc7ea0a0c1680d67e5432587f8c2f77833d5a717fcb2d20597b507"
    ),
    "backends/torch/src/torch_ext/rf/scattering_chain_ensemble.cu": (
        "6293c9238fa5c251d23408493fffd0b88cc557f50de84c90519ec1115ca7d9fd"
    ),
    "backends/torch/src/torch_ext/rf/scattering_chain_ensemble_ad.cu": (
        "a207dbf58b62286b8a58d7f22535900b198f187c7d0bffb2bacce728eaae306e"
    ),
    "backends/torch/src/torch_ext/rf/scattering_chain_realization.cu": (
        "be9601740ad1dce283708446ebc596b5fd5aca1da8f12421cc077d0dac99d424"
    ),
    "backends/torch/src/torch_ext/rf/scattering_chain_realization_ad.cu": (
        "970c579cc9d0c384d28e7aaa8f32200800a1de159de9a0338b2f0bad75f7fa93"
    ),
}
PHASE10B_DIRECT_TEST_SHA256 = (
    "5661129d9662d4f2879aaba284b245dd7f32a61b95c13a77e270d624ff315423"
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


def _lf_text_sha256(path: Path) -> str:
    # Evidence hashes lock Git's LF-normalized UTF-8 text, not checkout EOL bytes.
    text = path.read_bytes().decode("utf-8")
    normalized = text.replace("\r\n", "\n")
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def test_lf_text_sha256_normalizes_eol_without_hiding_content_changes(
    tmp_path: Path,
) -> None:
    lf = tmp_path / "lf.txt"
    crlf = tmp_path / "crlf.txt"
    changed = tmp_path / "changed.txt"
    bare_cr = tmp_path / "bare-cr.txt"
    lf.write_bytes("alpha\nβeta\n".encode("utf-8"))
    crlf.write_bytes("alpha\r\nβeta\r\n".encode("utf-8"))
    changed.write_bytes("alpha\r\nγamma\r\n".encode("utf-8"))
    bare_cr.write_bytes("alpha\rβeta\r".encode("utf-8"))

    assert _lf_text_sha256(lf) == _lf_text_sha256(crlf)
    assert _lf_text_sha256(lf) != _lf_text_sha256(changed)
    assert _lf_text_sha256(lf) != _lf_text_sha256(bare_cr)


def test_phase10b_pin_owner_counts_and_manifests_are_atomic() -> None:
    inventory = _json(AUDIT / "phase13-current-native-owner-inventory.json")
    migration = _json(AUDIT / "phase13-migration-delta.json")
    evidence = _json(AUDIT / "phase13-scattering-phase10b-evidence.json")
    graph = _json(AUDIT / "phase13-shared-rf-dependency-graph.json")

    assert {
        inventory["phase10b_scattering_chains"]["rayd_commit"],
        migration["phase10b_current"]["rayd_commit"],
        evidence["activation"]["rayd_commit"],
        graph["phase10b_activation"]["rayd_commit"],
    } == {PHASE10B_RAYD_COMMIT}
    assert migration["phase10b_current"]["owner_counts"] == {
        "RayD": 43,
        "layered": 2,
        "Channel Native": 157,
    }
    assert evidence["owner_transfer"]["owner_counts"] == {
        "RayD": 43,
        "layered": 2,
        "Channel Native": 157,
    }
    assert inventory["phase10b_scattering_chains"]["binding_count"] == 202
    assert migration["phase10b_current"]["binding_count"] == 202
    assert evidence["activation"]["binding_count"] == 202
    assert Counter(row["numerical_owner"] for row in inventory["symbols"]) == {
        "RayD": 43,
        "Channel operation / RayD primitives": 3,
        "Channel Native": 157,
    }
    owners = {row["symbol"]: row["numerical_owner"] for row in inventory["symbols"]}
    assert all(owners[symbol] == "RayD" for symbol in CHAIN_SYMBOLS)

    assert (
        evidence["activation"]["binding_manifest_sha256"]
        == PHASE10B_BINDING_MANIFEST_SHA256
    )
    assert (
        evidence["activation"]["contract_coverage_manifest_sha256"]
        == PHASE10B_COVERAGE_MANIFEST_SHA256
    )


def test_phase10b_rayd_identity_sources_and_direct_contract_are_recorded() -> None:
    evidence = _json(AUDIT / "phase13-scattering-phase10b-evidence.json")
    activation = evidence["activation"]
    assert activation["integration_header_sha256"] == INTEGRATION_SHA256
    assert activation["integration_header_identity"] == IDENTITY
    assert activation["typed_header_sha256"] == TYPED_SHA256
    assert activation["shared_table_header_sha256"] == SHARED_SHA256
    assert {record["path"]: record["sha256"] for record in evidence["rayd_sources"]} == (
        PHASE10B_RAYD_SOURCE_SHA256
    )
    direct = evidence["direct_contract_coverage"]
    assert direct["test_sha256"] == PHASE10B_DIRECT_TEST_SHA256
    assert direct["ctest_result"] == "4/4 passed"
    assert len(direct["depth8_positive_coverage"]) == 2


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
    assert refresh["region_count"] == 160
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
