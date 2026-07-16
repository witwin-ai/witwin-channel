from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

from tools.refactor_baseline import cpp_body_hashes


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
KERNEL_ROOT = REPOSITORY_ROOT / "native/channel_native/kernels"
INVENTORY_PATH = (
    REPOSITORY_ROOT / "docs/dev/audit/phase9-native-owner-inventory.json"
)

TRANSLATION_UNITS = {
    "diffraction": KERNEL_ROOT / "field_wedge_ad_diffraction.cu",
    "coupled": KERNEL_ROOT / "field_wedge_ad_coupled.cu",
    "project": KERNEL_ROOT / "field_wedge_ad_project.cu",
    "prepare": KERNEL_ROOT / "field_wedge_ad_prepare.cu",
}
ABI_BY_OWNER = {
    "diffraction": {
        "cn_field_diffraction_wedge",
        "cn_field_diffraction_wedge_backward",
        "cn_field_diffraction_wedge_jvp",
    },
    "coupled": {
        "cn_field_coupled_rd_backward",
        "cn_field_coupled_rd_jvp",
    },
    "project": {
        "cn_field_project_complex3_backward",
        "cn_field_project_complex3_jvp",
    },
    "prepare": {
        "cn_coupled_rd_prepare_backward",
        "cn_coupled_rd_prepare_jvp",
    },
}
COMMON_HELPERS = {
    "load3f",
    "to_c10",
    "from_c10",
    "launch_blocks",
    "zero_scalar",
    "optional_tensor_arg",
    "opt_ptr",
    "opt_mut_ptr",
}


def _function_names_by_path() -> dict[str, set[str]]:
    names: dict[str, set[str]] = {}
    for entry in cpp_body_hashes(REPOSITORY_ROOT):
        names.setdefault(entry["path"], set()).add(entry["name"])
    return names


def test_field_wedge_abi_has_one_semantic_translation_unit_owner() -> None:
    names = _function_names_by_path()
    all_abi = set().union(*ABI_BY_OWNER.values())

    for owner, path in TRANSLATION_UNITS.items():
        relative = path.relative_to(REPOSITORY_ROOT).as_posix()
        assert ABI_BY_OWNER[owner] <= names[relative]
        assert not (all_abi - ABI_BY_OWNER[owner]) & names[relative]


def test_field_wedge_split_preserves_launch_and_sync_multisets() -> None:
    inventory = json.loads(INVENTORY_PATH.read_text(encoding="utf-8"))
    source_evidence = next(
        entry
        for entry in inventory["source_evidence"]
        if entry["path"] == "native/channel_native/kernels/field_wedge_ad.cu"
    )
    source_by_owner = {
        owner: path.read_text(encoding="utf-8-sig")
        for owner, path in TRANSLATION_UNITS.items()
    }
    sources = "\n".join(source_by_owner.values())
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
    assert sources.count("cudaStreamSynchronize(") == 0
    assert {
        owner: source.count("<<<") for owner, source in source_by_owner.items()
    } == {"diffraction": 3, "coupled": 2, "project": 2, "prepare": 2}


def test_field_wedge_common_and_owner_local_plumbing_are_isolated() -> None:
    names = _function_names_by_path()
    common = "native/channel_native/kernels/field_wedge_ad_common.cuh"

    assert COMMON_HELPERS == names[common]
    for owner, path in TRANSLATION_UNITS.items():
        relative = path.relative_to(REPOSITORY_ROOT).as_posix()
        source = path.read_text(encoding="utf-8-sig")
        assert not COMMON_HELPERS & names[relative]
        assert source.count('#include "field_wedge_ad_common.cuh"') == 1
        if owner != "diffraction":
            assert "#define WEDGE_" not in source
        if owner != "coupled":
            assert "#define COUPLED_" not in source
            assert "check_coupled_primal_rows" not in source


def test_field_wedge_split_is_registered_once_and_below_budget() -> None:
    cmake = (REPOSITORY_ROOT / "CMakeLists.txt").read_text(encoding="utf-8")
    inventory = json.loads(INVENTORY_PATH.read_text(encoding="utf-8"))
    policy = inventory["translation_unit_policy"]
    common = "native/channel_native/kernels/field_wedge_ad_common.cuh"

    assert "native/channel_native/kernels/field_wedge_ad.cu" not in cmake
    assert "native/channel_native/kernels/field_wedge_ad.cu" not in policy[
        "planned_owner_debt"
    ]
    assert common not in cmake
    for path in TRANSLATION_UNITS.values():
        relative = path.relative_to(REPOSITORY_ROOT).as_posix()
        assert cmake.count(relative) == 1
        assert len(path.read_text(encoding="utf-8-sig").splitlines()) < policy[
            "recommended_limit_lines"
        ]
