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

MOVED_KERNELS = {
    "path_visibility_flags_kernel",
    "path_los_compact_kernel",
    "deterministic_los_topology_compact_kernel",
    "deterministic_reflection_order1_compact_kernel",
    "deterministic_reflection_sequence_compact_kernel",
    "deterministic_diffraction_order1_compact_kernel",
    "path_block_flags_kernel",
    "path_block_compact_kernel",
    "path_diffraction_compact_kernel",
}
MOVED_ABI = {
    "channel_path_filter_los_cuda",
    "channel_deterministic_los_topology_block",
    "channel_deterministic_reflection_order1_compact",
    "channel_deterministic_reflection_sequence_compact",
    "channel_deterministic_diffraction_order1_compact",
    "channel_path_filter_block_cuda",
    "channel_path_diffraction_block_cuda",
}
REMAINING_COMPACTION_ABI = {
    "channel_path_concat_vec3_cuda",
    "channel_path_los_visibility_inputs_cuda",
    "channel_path_finalize_blocks_cuda",
}
EMPTY_FACTORIES = {
    "empty_path_block_from",
    "empty_deterministic_los_topology_block_from",
}
COMMON_HOST_HELPERS = {
    "check_cuda_tensor",
    "check_vec3_table",
    "check_path_block_shapes",
    "launch_blocks",
}
TOPOLOGY_KERNELS = {
    "deterministic_order_init_kernel",
    "deterministic_sort_key_1d_kernel",
    "deterministic_sort_key_sequence_kernel",
    "deterministic_face_group_keys_kernel",
    "deterministic_surface_group_keys_kernel",
    "deterministic_face_group_sort_key_kernel",
    "deterministic_face_group_flags_kernel",
    "deterministic_face_group_assign_kernel",
    "deterministic_face_group_members_kernel",
}
TOPOLOGY_PRIVATE_FUNCTIONS = {
    "block_tensor",
    "block_has_field",
    "check_optional_field_presence",
    "check_topology_concat_schema",
    "copy_tensor_rows",
    "deterministic_gather_rows_kernel",
    "gather_tensor_rows",
}
TOPOLOGY_ABI = {
    "channel_deterministic_concat_topology_blocks",
    "channel_deterministic_gather_topology_block",
    "channel_deterministic_face_groups",
    "channel_deterministic_surface_face_groups",
    "channel_deterministic_sort_order",
}
TOPOLOGY_FUNCTIONS = TOPOLOGY_KERNELS | TOPOLOGY_PRIVATE_FUNCTIONS | TOPOLOGY_ABI


def _function_names_by_path() -> dict[str, set[str]]:
    names: dict[str, set[str]] = {}
    for entry in cpp_body_hashes(REPOSITORY_ROOT):
        names.setdefault(entry["path"], set()).add(entry["name"])
    return names


def test_path_compaction_translation_unit_owns_the_audited_functions() -> None:
    names = _function_names_by_path()
    trace = "native/channel/kernels/path_trace.cu"
    compaction = "native/channel/kernels/path_compaction.cu"
    common = "native/channel/kernels/path_compaction_common.cuh"

    assert MOVED_KERNELS | MOVED_ABI <= names[compaction]
    assert not (MOVED_KERNELS | MOVED_ABI) & names[trace]
    assert REMAINING_COMPACTION_ABI <= names[trace]
    assert not REMAINING_COMPACTION_ABI & names[compaction]
    assert EMPTY_FACTORIES | COMMON_HOST_HELPERS == names[common]
    assert not COMMON_HOST_HELPERS & names[trace]
    assert not COMMON_HOST_HELPERS & names[compaction]

    sources = {
        path: (REPOSITORY_ROOT / path).read_text(encoding="utf-8-sig")
        for path in (trace, compaction, common)
    }
    assert (
        sum(
            source.count("constexpr int kPathBlockSize = 256;")
            for source in sources.values()
        )
        == 1
    )


def test_deterministic_topology_translation_unit_owns_the_audited_functions() -> None:
    names = _function_names_by_path()
    topology = "native/channel/kernels/deterministic_topology.cu"
    trace = "native/channel/kernels/path_trace.cu"
    compaction = "native/channel/kernels/path_compaction.cu"

    assert TOPOLOGY_FUNCTIONS <= names[topology]
    assert not TOPOLOGY_FUNCTIONS & names[trace]
    assert not TOPOLOGY_FUNCTIONS & names[compaction]


def test_path_split_preserves_the_frozen_launch_and_sync_multisets() -> None:
    inventory = json.loads(INVENTORY_PATH.read_text(encoding="utf-8"))
    source_evidence = next(
        entry
        for entry in inventory["source_evidence"]
        if entry["path"] == "native/channel/kernels/path_trace.cu"
    )
    sources = "\n".join(
        (KERNEL_ROOT / name).read_text(encoding="utf-8-sig")
        for name in (
            "deterministic_topology.cu",
            "path_trace.cu",
            "path_compaction.cu",
        )
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
    assert sources.count("cudaStreamSynchronize(") == len(
        source_evidence["explicit_sync_sites"]
    )
    actual_by_unit = {}
    for name in (
        "deterministic_topology.cu",
        "path_trace.cu",
        "path_compaction.cu",
    ):
        source = (KERNEL_ROOT / name).read_text(encoding="utf-8-sig")
        actual_by_unit[name] = (
            source.count("<<<"),
            source.count("cudaStreamSynchronize("),
        )
    assert actual_by_unit == {
        "deterministic_topology.cu": (16, 4),
        "path_trace.cu": (19, 0),
        "path_compaction.cu": (16, 7),
    }


def test_path_split_is_registered_once_and_below_the_recommended_limit() -> None:
    cmake = (REPOSITORY_ROOT / "CMakeLists.txt").read_text(encoding="utf-8")
    inventory = json.loads(INVENTORY_PATH.read_text(encoding="utf-8"))
    policy = inventory["translation_unit_policy"]

    for name in (
        "deterministic_topology.cu",
        "path_trace.cu",
        "path_compaction.cu",
    ):
        relative = f"native/channel/kernels/{name}"
        line_count = len(
            (KERNEL_ROOT / name).read_text(encoding="utf-8-sig").splitlines()
        )
        assert cmake.count(relative) == 1
        assert line_count < policy["recommended_limit_lines"]

    assert "native/channel/kernels/path_trace.cu" not in policy[
        "planned_owner_debt"
    ]
