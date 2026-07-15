from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest
import torch

from tools import refactor_runtime_baseline as runtime


@dataclass(frozen=True)
class _NestedPaths:
    valid: torch.Tensor
    primitive_id: torch.Tensor


@dataclass(frozen=True)
class _Result:
    path_gain: torch.Tensor
    paths: _NestedPaths
    metadata: dict[str, object]
    optional: torch.Tensor | None = None


def _result() -> _Result:
    storage = torch.arange(8, dtype=torch.float32)
    return _Result(
        path_gain=storage[1:5].reshape(2, 2),
        paths=_NestedPaths(
            valid=torch.tensor([True, False]),
            primitive_id=storage[:2].view(torch.int32),
        ),
        metadata={
            "kernel": {
                "primitive": "fixture",
                "launch_count": 3,
                "forward_launch_count": 2,
                "backward_launch_count": 1,
                "jvp_launch_count": 0,
                "tape_bytes": 128,
                "peak_memory_bytes": 256,
            },
            "seed": 7,
        },
    )


def test_result_manifest_hashes_every_tensor_and_records_alias_contract():
    result = _result()

    manifest = runtime.result_manifest(result)

    assert manifest["tensor_count"] == 3
    tensors = {row["path"]: row["tensor"] for row in manifest["tensors"]}
    gain = tensors["$.path_gain"]
    primitive = tensors["$.paths.primitive_id"]
    assert gain["shape"] == [2, 2]
    assert gain["dtype"] == "torch.float32"
    assert gain["device"] == "cpu"
    assert gain["stride"] == [2, 1]
    assert gain["storage_offset"] == 1
    assert gain["contiguous"] is True
    assert gain["requires_grad"] is False
    assert gain["alias_group"] == primitive["alias_group"]
    assert {row["path"] for row in manifest["path_identity"]} == {
        "$.paths.valid",
        "$.paths.primitive_id",
    }
    assert manifest["metadata_fingerprint"]
    assert manifest["result_fingerprint"]


def test_tensor_hash_changes_with_value_but_not_with_alias_pointer():
    first = runtime.value_manifest(torch.tensor([1.0, 2.0]))
    same = runtime.value_manifest(torch.tensor([1.0, 2.0]))
    changed = runtime.value_manifest(torch.tensor([1.0, 3.0]))

    assert first["fingerprint"] == same["fingerprint"]
    assert first["fingerprint"] != changed["fingerprint"]


def test_tensor_hash_handles_contiguous_size_one_dimension_with_nonunit_stride():
    cases = (
        torch.arange(2, dtype=torch.float32).as_strided((2, 1), (1, 2)),
        torch.arange(1, dtype=torch.float32).as_strided((1,), (2,)),
    )
    for unusual in cases:
        assert unusual.is_contiguous()
        unusual_manifest = runtime.value_manifest(unusual)
        canonical = torch.tensor(unusual.tolist(), dtype=unusual.dtype)
        canonical_manifest = runtime.value_manifest(canonical)
        assert (
            unusual_manifest["tensors"][0]["tensor"]["sha256"]
            == canonical_manifest["tensors"][0]["tensor"]["sha256"]
        )
        assert unusual_manifest["tensors"][0]["tensor"]["stride"] == list(
            unusual.stride()
        )


def test_launch_ledger_uses_real_aggregate_and_marks_unavailable_detail():
    ledger = runtime.launch_ledger(_result())

    assert ledger["availability"] == "aggregate-only"
    assert ledger["aggregate"] == {
        "primitive": "fixture",
        "launch_count": 3,
        "forward_launch_count": 2,
        "backward_launch_count": 1,
        "jvp_launch_count": 0,
        "tape_bytes": 128,
        "peak_memory_bytes": 256,
    }
    assert ledger["per_launch"] == []
    assert ledger["unavailable_fields"] == [
        "per_launch_name",
        "grid",
        "block",
        "stream",
        "synchronization_points",
    ]


def test_semantic_fingerprint_excludes_only_allowlisted_runtime_metadata():
    first = _result()
    second = _Result(
        path_gain=first.path_gain,
        paths=first.paths,
        metadata={
            **first.metadata,
            "kernel": {
                **first.metadata["kernel"],
                "forward_time_ms": 9.5,
                "peak_memory_bytes": 999,
            },
        },
    )
    third = _Result(
        path_gain=second.path_gain,
        paths=second.paths,
        metadata={**second.metadata, "seed": 8},
    )

    first_manifest = runtime.result_manifest(first)
    second_manifest = runtime.result_manifest(second)
    third_manifest = runtime.result_manifest(third)

    assert (
        first_manifest["semantic_result_fingerprint"]
        == second_manifest["semantic_result_fingerprint"]
    )
    assert (
        first_manifest["semantic_metadata_fingerprint"]
        == second_manifest["semantic_metadata_fingerprint"]
    )
    assert (
        first_manifest["full_result_fingerprint"]
        != second_manifest["full_result_fingerprint"]
    )
    assert {row["path"] for row in second_manifest["volatile_metadata"]} == {
        "$.metadata['kernel']['forward_time_ms']",
        "$.metadata['kernel']['peak_memory_bytes']",
    }
    assert (
        second_manifest["semantic_result_fingerprint"]
        != third_manifest["semantic_result_fingerprint"]
    )


@pytest.mark.parametrize(
    ("processes", "warmup", "repeats", "message"),
    [
        (1, 1, 7, "independent processes"),
        (2, 0, 7, "warmup"),
        (2, 1, 6, "steady repeats"),
    ],
)
def test_measurement_policy_cannot_lower_plan_minimums(
    processes: int, warmup: int, repeats: int, message: str
):
    with pytest.raises(runtime.RuntimeBaselineError, match=message):
        runtime.validate_measurement_policy(processes, warmup, repeats)


def _row(process_index: int, fingerprint: str = "exact") -> dict[str, object]:
    start = float(process_index * 10)
    return {
        "solver": "path",
        "scenario": "empty-los",
        "config": {"max_depth": 0},
        "scene_input_fingerprint": "scene",
        "result": {
            "result_fingerprint": fingerprint,
            "metadata_fingerprint": "metadata",
        },
        "performance": {
            "steady": [
                {"wall_ms": start + value, "cuda_event_ms": start + value / 2}
                for value in range(1, 8)
            ],
            "memory": {
                "peak_allocated_bytes": 100 + process_index,
                "peak_reserved_bytes": 200 + process_index,
                "peak_temporary_allocated_bytes": 50 + process_index,
            },
        },
    }


def test_aggregate_case_requires_exact_results_and_reports_median_p95():
    aggregate = runtime.aggregate_case([_row(0), _row(1)])

    distribution = aggregate["performance_distribution"]
    assert aggregate["exact_across_processes"] is True
    assert aggregate["result_fingerprint"] == "exact"
    assert distribution["process_count"] == 2
    assert distribution["steady_sample_count"] == 14
    assert distribution["wall_median_ms"] == 9.0
    assert distribution["wall_p95_ms"] == pytest.approx(16.35)
    assert distribution["peak_allocated_bytes_max"] == 101

    with pytest.raises(runtime.RuntimeBaselineError, match="result hash differs"):
        runtime.aggregate_case([_row(0), _row(1, fingerprint="changed")])


def test_parse_child_stdout_ignores_non_json_diagnostics():
    payload = runtime._parse_child_stdout(
        'native diagnostic\n{"solver":"path","result":{"result_fingerprint":"a"}}\n'
    )

    assert payload["solver"] == "path"


def test_runtime_report_is_bound_to_sha_and_immutable(tmp_path: Path):
    sha = "c" * 40
    report = {
        "schema": {"name": runtime.SCHEMA_NAME, "version": runtime.SCHEMA_VERSION},
        "git_sha": sha,
        "profile": "reduced",
        "measurement_policy": {},
        "coverage": {},
        "cases": [],
    }

    destination = runtime.write_immutable_report(report, tmp_path)

    assert destination == tmp_path / sha / "reduced.json"
    assert sorted(path.name for path in destination.parent.iterdir()) == [
        "launch-ledger.json",
        "performance.json",
        "reduced.json",
        "solver-results.json",
    ]
    with pytest.raises(runtime.RuntimeBaselineError, match="immutable"):
        runtime.write_immutable_report(report, tmp_path)
