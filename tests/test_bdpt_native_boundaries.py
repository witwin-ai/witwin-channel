from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

from tools.refactor_baseline import cpp_body_hashes


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
KERNEL_ROOT = REPOSITORY_ROOT / "native/channel/kernels"
INVENTORY_PATH = (
    REPOSITORY_ROOT / "docs/dev/audit/phase9-native-owner-inventory.json"
)

FUNCTIONS_BY_UNIT = {
    "bdpt_connect_mis.cu": {
        "bdpt_mis_weights_kernel",
        "channel_bdpt_mis_weights_cuda",
    },
    "bdpt_connect_samples.cu": {
        "bdpt_endpoint_connection_samples_kernel",
        "channel_bdpt_endpoint_connection_samples_cuda",
    },
    "bdpt_connect_visibility.cu": {
        "bdpt_endpoint_connection_visibility_inputs_kernel",
        "bdpt_filter_connection_samples_kernel",
        "bdpt_compact_connection_samples_kernel",
        "bdpt_copy_connection_samples_kernel",
        "bdpt_count_valid_connection_samples_kernel",
        "channel_bdpt_endpoint_connection_visibility_inputs_cuda",
        "channel_bdpt_filter_connection_samples_cuda",
        "channel_bdpt_count_valid_connection_samples_cuda",
        "channel_bdpt_compact_connection_samples_cuda",
        "channel_bdpt_concat_connection_samples_cuda",
    },
    "bdpt_connect_accumulation.cu": {
        "bdpt_accumulate_connection_samples_double_kernel",
        "bdpt_compact_valid_connection_indices_kernel",
        "bdpt_accumulate_connection_samples_compacted_kernel",
        "bdpt_accumulate_connection_samples_staged_kernel",
        "bdpt_cast_connection_accumulation_kernel",
        "bdpt_connection_variance_accum_double_kernel",
        "bdpt_connection_variance_finalize_double_kernel",
        "channel_bdpt_accumulate_connection_samples_cuda",
        "channel_bdpt_connection_variance_cuda",
        # ADR-019 coherent combine + ADR-022 coherent/power AD companions and
        # their private accumulate helpers, all owned by this accumulate TU.
        "bdpt_accumulate_connection_samples_coherent_kernel",
        "bdpt_finalize_coherent_accumulation_kernel",
        "bdpt_accumulate_power_backward_kernel",
        "bdpt_accumulate_power_jvp_kernel",
        "bdpt_accumulate_coherent_backward_kernel",
        "bdpt_accumulate_coherent_jvp_kernel",
        "channel_bdpt_accumulate_connection_samples_backward_cuda",
        "channel_bdpt_accumulate_connection_samples_jvp_cuda",
        "accumulate_optional",
        "accumulate_ptr",
        "bdpt_component_matrix",
    },
}
COMMON_HELPERS = {
    "bdpt_component_from_mask",
    "bdpt_component_accumulable",
    "check_float_cuda",
    "check_int_cuda",
    "check_bool_cuda",
    "check_vec3_cuda",
    "check_same_device",
    "check_mis_args",
    "bdpt_connection_mis_weight_from_sums",
    "bdpt_single_strategy_mis_weight",
    "bdpt_free_space_gain",
    "bdpt_make_float3",
    "bdpt_add3",
    "bdpt_sub3",
    "bdpt_scale3",
    "bdpt_norm3",
    "bdpt_normalize3",
    "bdpt_vec3_at",
    "allocate_connection_samples",
    "zero_double_tensor",
    "zero_int_tensor",
    "zero_float_tensor",
}
ABI_BY_UNIT = {
    unit: {name for name in functions if name.startswith("channel_")}
    for unit, functions in FUNCTIONS_BY_UNIT.items()
}


def _function_names_by_path() -> dict[str, set[str]]:
    names: dict[str, set[str]] = {}
    for entry in cpp_body_hashes(REPOSITORY_ROOT):
        names.setdefault(entry["path"], set()).add(entry["name"])
    return names


def test_bdpt_functions_have_one_physical_translation_unit_owner() -> None:
    names = _function_names_by_path()
    common = "native/channel/kernels/bdpt_connect_common.cuh"

    assert names[common] == COMMON_HELPERS
    for unit, expected in FUNCTIONS_BY_UNIT.items():
        relative = f"native/channel/kernels/{unit}"
        assert names[relative] == expected

    all_abi = set().union(*ABI_BY_UNIT.values())
    # Phase 4 retires two audited-dead crude diffraction sample ABIs.
    assert len(all_abi) == 11
    for unit, expected in ABI_BY_UNIT.items():
        relative = f"native/channel/kernels/{unit}"
        assert expected == (all_abi & names[relative])


def test_bdpt_common_helpers_keep_internal_linkage() -> None:
    common_path = KERNEL_ROOT / "bdpt_connect_common.cuh"
    common = common_path.read_text(encoding="utf-8-sig")

    assert common.startswith("#pragma once\n")
    assert re.search(r"\bnamespace\s*\{", common)
    assert common.rstrip().endswith("}  // namespace")
    assert not any(name in common for names in ABI_BY_UNIT.values() for name in names)
    for unit in FUNCTIONS_BY_UNIT:
        source = (KERNEL_ROOT / unit).read_text(encoding="utf-8-sig")
        assert source.count('#include "bdpt_connect_common.cuh"') == 1


def test_bdpt_split_preserves_launch_and_sync_multisets() -> None:
    inventory = json.loads(INVENTORY_PATH.read_text(encoding="utf-8"))
    source_evidence = next(
        entry
        for entry in inventory["source_evidence"]
        if entry["path"] == "native/channel_native/kernels/bdpt_connect.cu"
    )
    sources = {
        unit: (KERNEL_ROOT / unit).read_text(encoding="utf-8-sig")
        for unit in FUNCTIONS_BY_UNIT
    }
    combined = "\n".join(sources.values())
    actual_launches = Counter(
        re.findall(
            r"\b([A-Za-z_]\w*_kernel)(?:\s*<[^;{}]*?>)?\s*<<<",
            combined,
        )
    )
    expected_launches = Counter(
        site["kernel"] for site in source_evidence["kernel_launch_sites"]
    )
    expected_launches.subtract(
        {
            "bdpt_diffraction_connection_samples_from_tape_kernel": 1,
            "bdpt_diffraction_point_connection_samples_kernel": 1,
        }
    )
    expected_launches += Counter()

    assert actual_launches == expected_launches
    # Phase 4 removes two audited-dead crude diffraction launches; syncs stay fixed.
    assert sum(actual_launches.values()) == 21
    assert combined.count("cudaStreamSynchronize(") == 2
    assert {
        unit: (
            source.count("<<<"),
            source.count("cudaStreamSynchronize("),
        )
        for unit, source in sources.items()
    } == {
        "bdpt_connect_mis.cu": (1, 0),
        "bdpt_connect_samples.cu": (1, 0),
        "bdpt_connect_visibility.cu": (6, 2),
        "bdpt_connect_accumulation.cu": (13, 0),
    }


def test_bdpt_split_is_registered_once_and_below_budget() -> None:
    cmake = (REPOSITORY_ROOT / "CMakeLists.txt").read_text(encoding="utf-8")
    inventory = json.loads(INVENTORY_PATH.read_text(encoding="utf-8"))
    policy = inventory["translation_unit_policy"]

    assert "native/channel/kernels/bdpt_connect.cu" not in cmake
    assert not (KERNEL_ROOT / "bdpt_connect.cu").exists()
    assert "native/channel/kernels/bdpt_connect_common.cuh" not in cmake
    for unit in FUNCTIONS_BY_UNIT:
        relative = f"native/channel/kernels/{unit}"
        assert cmake.count(relative) == 1
        assert len((KERNEL_ROOT / unit).read_text().splitlines()) < policy[
            "recommended_limit_lines"
        ]
    assert "native/channel/kernels/bdpt_connect.cu" not in policy[
        "planned_owner_debt"
    ]
