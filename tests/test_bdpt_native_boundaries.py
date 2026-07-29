from __future__ import annotations

from pathlib import Path

from tools.refactor_baseline import cpp_body_hashes


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
MERGED = "native/channel/kernels/bdpt_connect.cu"
COMMON = "native/channel/kernels/bdpt_connect_common.cuh"
EXPECTED_ABI = {
    "channel_bdpt_mis_weights_cuda",
    "channel_bdpt_endpoint_connection_samples_cuda",
    "channel_bdpt_endpoint_connection_samples_backward_cuda",
    "channel_bdpt_endpoint_connection_samples_jvp_cuda",
    "channel_bdpt_endpoint_connection_visibility_inputs_cuda",
    "channel_bdpt_filter_connection_samples_cuda",
    "channel_bdpt_count_valid_connection_samples_cuda",
    "channel_bdpt_compact_connection_samples_cuda",
    "channel_bdpt_concat_connection_samples_cuda",
    "channel_bdpt_accumulate_connection_samples_cuda",
    "channel_bdpt_accumulate_connection_samples_backward_cuda",
    "channel_bdpt_accumulate_connection_samples_jvp_cuda",
    "channel_bdpt_connection_variance_cuda",
}


def _function_names_by_path() -> dict[str, set[str]]:
    names: dict[str, set[str]] = {}
    for entry in cpp_body_hashes(REPOSITORY_ROOT):
        names.setdefault(entry["path"], set()).add(entry["name"])
    return names


def test_bdpt_connection_family_has_one_physical_owner() -> None:
    names = _function_names_by_path()
    source = (REPOSITORY_ROOT / MERGED).read_text(encoding="utf-8-sig")
    common = (REPOSITORY_ROOT / COMMON).read_text(encoding="utf-8-sig")

    assert EXPECTED_ABI <= names[MERGED]
    assert source.count('#include "bdpt_connect_common.cuh"') == 5
    assert common.startswith("#pragma once\n")
    assert "namespace {" in common
    assert common.rstrip().endswith("}  // namespace")


def test_bdpt_connection_consolidation_is_registered_once() -> None:
    cmake = (REPOSITORY_ROOT / "CMakeLists.txt").read_text(encoding="utf-8")

    assert cmake.count(MERGED) == 1
    for retired in (
        "native/channel/kernels/bdpt_connect_mis.cu",
        "native/channel/kernels/bdpt_connect_samples.cu",
        "native/channel/kernels/bdpt_connect_visibility.cu",
        "native/channel/kernels/bdpt_connect_accumulation.cu",
        "native/channel/kernels/bdpt_connect_ad.cu",
    ):
        assert retired not in cmake
    assert COMMON not in cmake