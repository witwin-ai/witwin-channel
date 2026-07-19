from __future__ import annotations

import json
import os
import re
from collections import Counter
from pathlib import Path

from tools.refactor_baseline import cpp_body_hashes


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
KERNEL_ROOT = REPOSITORY_ROOT / "native/channel_native/kernels"
INVENTORY_PATH = (
    REPOSITORY_ROOT / "docs/dev/audit/phase9-native-owner-inventory.json"
)
RAYD_ROOT = Path(
    os.environ.get("RAYD_SOURCE_DIR", REPOSITORY_ROOT.parent.parent / "RayDi")
)
RAYD_FIELD_TRANSPORT_AD = (
    RAYD_ROOT
    / "backends/torch/include/rayd/torch/rf/field_transport_ad.cuh"
)

TRANSLATION_UNITS = {
    "free_space": KERNEL_ROOT / "field_transport_free_space.cu",
    "reflection": KERNEL_ROOT / "field_transport_reflection.cu",
    "transmission": KERNEL_ROOT / "field_transport_transmission.cu",
}
ABI_BY_OWNER = {
    "free_space": {
        "cn_field_free_space_fwd64",
        "cn_field_free_space_backward",
        "cn_field_free_space_jvp",
    },
    "reflection": {
        "cn_field_reflection_sequence_backward",
        "cn_field_reflection_sequence_jvp",
    },
    "transmission": {
        "cn_field_transmission_sequence_backward",
        "cn_field_transmission_sequence_jvp",
    },
}
COMMON_HELPERS = {
    "load3f",
    "load_sequence3f",
    "load_dual3f",
    "load_dual_sequence3f",
    "complex_of",
    "to_c10",
    "launch_blocks",
    "zero_filled",
    "optional_grad",
    "grad_ptr",
}
RAYD_NUMERICAL_HELPERS = {
    "fold_output_cotangents",
    "write_output_tangents",
}


def _function_names_by_path() -> dict[str, set[str]]:
    names: dict[str, set[str]] = {}
    for entry in cpp_body_hashes(REPOSITORY_ROOT):
        names.setdefault(entry["path"], set()).add(entry["name"])
    return names


def test_field_transport_abi_has_one_semantic_translation_unit_owner() -> None:
    names = _function_names_by_path()
    all_abi = set().union(*ABI_BY_OWNER.values())

    for owner, path in TRANSLATION_UNITS.items():
        relative = path.relative_to(REPOSITORY_ROOT).as_posix()
        assert ABI_BY_OWNER[owner] <= names[relative]
        assert not (all_abi - ABI_BY_OWNER[owner]) & names[relative]


def test_field_transport_split_preserves_launch_and_sync_multisets() -> None:
    inventory = json.loads(INVENTORY_PATH.read_text(encoding="utf-8"))
    source_evidence = next(
        entry
        for entry in inventory["source_evidence"]
        if entry["path"] == "native/channel_native/kernels/field_transport_ad.cu"
    )
    sources = "\n".join(
        path.read_text(encoding="utf-8-sig") for path in TRANSLATION_UNITS.values()
    )
    actual_launches = Counter(
        re.findall(
            r"\b([A-Za-z_]\w*_kernel)(?:\s*<[^;{}]*?>)?\s*<<<",
            sources,
        )
    )
    expected_launches = Counter(
        site["kernel"] for site in source_evidence["kernel_launch_sites"]
    )

    assert actual_launches == expected_launches
    assert sum(actual_launches.values()) == 9
    assert sources.count("cudaStreamSynchronize(") == 0


def test_field_transport_common_helpers_have_one_source() -> None:
    names = _function_names_by_path()
    common = "native/channel_native/kernels/field_transport_ad_common.cuh"

    assert COMMON_HELPERS == names[common]
    for path in TRANSLATION_UNITS.values():
        relative = path.relative_to(REPOSITORY_ROOT).as_posix()
        assert not COMMON_HELPERS & names[relative]
        assert path.read_text(encoding="utf-8-sig").count(
            '#include "field_transport_ad_common.cuh"'
        ) == 1


def test_output_chain_ad_helpers_are_defined_only_in_locked_rayd_header() -> None:
    common_source = (
        KERNEL_ROOT / "field_transport_ad_common.cuh"
    ).read_text(encoding="utf-8-sig")
    rayd_source = RAYD_FIELD_TRANSPORT_AD.read_text(encoding="utf-8-sig")

    for helper in RAYD_NUMERICAL_HELPERS:
        definition = re.compile(
            rf"__device__\s+__forceinline__[^;{{}}]*\b{helper}\s*\("
        )
        assert len(definition.findall(rayd_source)) == 1
        assert not definition.search(common_source)
        assert common_source.count(f"using ad::{helper};") == 1


def test_field_transport_split_is_registered_once_and_below_budget() -> None:
    cmake = (REPOSITORY_ROOT / "CMakeLists.txt").read_text(encoding="utf-8")
    common = "native/channel_native/kernels/field_transport_ad_common.cuh"

    assert "native/channel_native/kernels/field_transport_ad.cu" not in cmake
    assert common not in cmake
    for path in TRANSLATION_UNITS.values():
        relative = path.relative_to(REPOSITORY_ROOT).as_posix()
        assert cmake.count(relative) == 1
        assert len(path.read_text(encoding="utf-8-sig").splitlines()) < 2000
