# Copyright Xingyu Chen.
# Tests native translation unit architecture.

from __future__ import annotations

import json
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
KERNEL_ROOT = REPOSITORY_ROOT / "native/channel/kernels"
LEDGER_PATH = REPOSITORY_ROOT / (
    "docs/dev/audit/adr-044-native-tu-consolidation.json"
)
ADR_PATH = REPOSITORY_ROOT / (
    "docs/dev/standards/adr-044-native-translation-unit-consolidation.md"
)
BUDGET_PATH = REPOSITORY_ROOT / "ci/maintenance-budgets.json"


def _ledger() -> dict:
    return json.loads(LEDGER_PATH.read_text(encoding="utf-8"))


def test_native_translation_unit_size_policy_is_retired() -> None:
    adr = ADR_PATH.read_text(encoding="utf-8")
    budgets = json.loads(BUDGET_PATH.read_text(encoding="utf-8"))

    assert "There is no maximum line count" in adr
    assert "limits.native_file_lines" not in budgets["limits"]
    assert "native_file_exemptions" not in budgets
    assert "Retired 2026-07-28" in budgets["limits_policy"][
        "native_file_lines"
    ]


def test_cuda_translation_units_match_the_adr044_ledger() -> None:
    ledger = _ledger()
    expected = {group["target"] for group in ledger["groups"]}
    actual = {path.name for path in KERNEL_ROOT.glob("*.cu")}

    assert ledger["baseline_cuda_translation_units"] == 45
    assert ledger["target_cuda_translation_units"] == 15
    assert len(expected) == ledger["target_cuda_translation_units"]
    assert actual == expected


def test_native_headers_are_shared_contracts_with_short_owner_names() -> None:
    expected = {
        "capacity.h",
        "field_ad.cuh",
        "math.cuh",
        "path_compaction.cuh",
        "path_payload.cuh",
        "torch_cuda.h",
    }
    actual = {
        path.name
        for path in KERNEL_ROOT.iterdir()
        if path.suffix in {".h", ".cuh"}
    }

    assert actual == expected
    assert all("common" not in name and "plumbing" not in name for name in actual)

def test_consolidated_units_are_registered_once() -> None:
    ledger = _ledger()
    cmake = (REPOSITORY_ROOT / "CMakeLists.txt").read_text(encoding="utf-8")
    targets = {group["target"] for group in ledger["groups"]}
    retired = {
        source
        for group in ledger["groups"]
        for source in group["sources"]
        if source != group["target"]
    }

    source_block = cmake.split("Python_add_library(", 1)[1].split("\n)", 1)[0]
    for target in targets:
        relative = f"native/channel/kernels/{target}"
        assert sum(line.strip() == relative for line in source_block.splitlines()) == 1
    for source in retired:
        relative = f"native/channel/kernels/{source}"
        assert relative not in cmake


def test_consolidation_preserves_launch_sync_and_trap_multisets() -> None:
    for group in _ledger()["groups"]:
        source = (KERNEL_ROOT / group["target"]).read_text(encoding="utf-8-sig")
        assert source.count("<<<") == group["baseline_launch_sites"]
        assert source.count("cudaStreamSynchronize(") == group[
            "baseline_explicit_sync_sites"
        ]
        assert source.count("__trap(") == group["baseline_device_trap_sites"]


def test_special_compile_modes_remain_narrow() -> None:
    ledger = _ledger()
    cmake = (REPOSITORY_ROOT / "CMakeLists.txt").read_text(encoding="utf-8")
    fmad_false = {
        group["target"]
        for group in ledger["groups"]
        if group["compile_mode"] == "fmad_false"
    }

    assert fmad_false == {
        "kirchhoff.cu",
        "mc_transmission.cu",
    }
    assert cmake.count('PROPERTIES COMPILE_OPTIONS "--fmad=false"') == 2
    assert "--use_fast_math" not in cmake