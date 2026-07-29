from __future__ import annotations

from pathlib import Path

from tools.refactor_baseline import cpp_body_hashes


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
MERGED = "native/channel/kernels/path_topology.cu"
COMMON = "native/channel/kernels/path_compaction_common.cuh"
EXPECTED_ABI = {
    "channel_path_filter_los_cuda",
    "channel_deterministic_los_topology_block",
    "channel_deterministic_reflection_order1_compact",
    "channel_deterministic_reflection_sequence_compact",
    "channel_deterministic_diffraction_order1_compact",
    "channel_path_filter_block_cuda",
    "channel_path_diffraction_block_cuda",
    "channel_path_concat_vec3_cuda",
    "channel_path_los_visibility_inputs_cuda",
    "channel_path_finalize_blocks_cuda",
    "channel_deterministic_concat_topology_blocks",
    "channel_deterministic_gather_topology_block",
    "channel_deterministic_face_groups",
    "channel_deterministic_surface_face_groups",
    "channel_deterministic_sort_order",
}
COMMON_HELPERS = {
    "check_cuda_tensor",
    "check_vec3_table",
    "check_path_block_shapes",
    "launch_blocks",
    "observe_compact_count",
    "empty_path_block_from",
    "empty_deterministic_los_topology_block_from",
    "compact_count_control_metadata_kernel",
}


def _function_names_by_path() -> dict[str, set[str]]:
    names: dict[str, set[str]] = {}
    for entry in cpp_body_hashes(REPOSITORY_ROOT):
        names.setdefault(entry["path"], set()).add(entry["name"])
    return names


def test_path_topology_families_have_one_physical_owner() -> None:
    names = _function_names_by_path()

    assert EXPECTED_ABI <= names[MERGED]
    assert COMMON_HELPERS == names[COMMON]
    assert not COMMON_HELPERS & names[MERGED]


def test_path_topology_consolidation_is_registered_once() -> None:
    cmake = (REPOSITORY_ROOT / "CMakeLists.txt").read_text(encoding="utf-8")

    assert cmake.count(MERGED) == 1
    for retired in (
        "native/channel/kernels/path_trace.cu",
        "native/channel/kernels/path_compaction.cu",
        "native/channel/kernels/deterministic_topology.cu",
    ):
        assert retired not in cmake
    assert COMMON not in cmake