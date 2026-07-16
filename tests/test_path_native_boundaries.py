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
    "cn_path_filter_los_cuda",
    "cn_deterministic_los_topology_block",
    "cn_deterministic_reflection_order1_compact",
    "cn_deterministic_reflection_sequence_compact",
    "cn_deterministic_diffraction_order1_compact",
    "cn_path_filter_block_cuda",
    "cn_path_diffraction_block_cuda",
}
REMAINING_COMPACTION_ABI = {
    "cn_path_concat_vec3_cuda",
    "cn_path_los_visibility_inputs_cuda",
    "cn_path_finalize_blocks_cuda",
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


def _function_names_by_path() -> dict[str, set[str]]:
    names: dict[str, set[str]] = {}
    for entry in cpp_body_hashes(REPOSITORY_ROOT):
        names.setdefault(entry["path"], set()).add(entry["name"])
    return names


def test_path_compaction_translation_unit_owns_the_audited_functions() -> None:
    names = _function_names_by_path()
    trace = "native/channel_native/kernels/path_trace.cu"
    compaction = "native/channel_native/kernels/path_compaction.cu"
    common = "native/channel_native/kernels/path_compaction_common.cuh"

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


def test_path_split_preserves_the_frozen_launch_and_sync_multisets() -> None:
    inventory = json.loads(INVENTORY_PATH.read_text(encoding="utf-8"))
    source_evidence = next(
        entry
        for entry in inventory["source_evidence"]
        if entry["path"] == "native/channel_native/kernels/path_trace.cu"
    )
    sources = "\n".join(
        (KERNEL_ROOT / name).read_text(encoding="utf-8-sig")
        for name in ("path_trace.cu", "path_compaction.cu")
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


def test_path_split_is_registered_once_and_below_the_hard_limit() -> None:
    cmake = (REPOSITORY_ROOT / "CMakeLists.txt").read_text(encoding="utf-8")
    inventory = json.loads(INVENTORY_PATH.read_text(encoding="utf-8"))
    policy = inventory["translation_unit_policy"]

    for name in ("path_trace.cu", "path_compaction.cu"):
        relative = f"native/channel_native/kernels/{name}"
        line_count = len(
            (KERNEL_ROOT / name).read_text(encoding="utf-8-sig").splitlines()
        )
        assert cmake.count(relative) == 1
        assert line_count <= policy["hard_limit_lines"]

    assert policy["planned_owner_debt"][
        "native/channel_native/kernels/path_trace.cu"
    ] == len(
        (KERNEL_ROOT / "path_trace.cu")
        .read_text(encoding="utf-8-sig")
        .splitlines()
    )
