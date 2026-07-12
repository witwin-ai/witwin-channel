from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from benchmarks import harness


def test_versioned_performance_schema_and_sm_matrix_are_committed():
    root = Path(__file__).resolve().parents[2]
    schema = json.loads(
        (root / "benchmarks/schemas/performance-result.schema.json").read_text(
            encoding="utf-8"
        )
    )
    matrix = json.loads(
        (root / "benchmarks/baselines/channel_native_sm_matrix.v1.json").read_text(
            encoding="utf-8"
        )
    )
    sm_schema = json.loads(
        (root / "benchmarks/schemas/sm-support.schema.json").read_text(
            encoding="utf-8"
        )
    )

    assert schema["properties"]["schema"]["properties"]["version"]["const"] == "1.0.0"
    assert [row["sm"] for row in matrix["build_architectures"]] == [75, 80, 86, 89, 120]
    assert sm_schema["properties"]["build_architectures"]["items"]["properties"][
        "status"
    ]["enum"] == ["declared_unverified", "verified"]
    cmake = (root / "CMakeLists.txt").read_text(encoding="utf-8")
    for sm in (75, 80, 86, 89, 120):
        assert f"{sm}-real" in cmake
    assert all(
        row["status"] == "declared_unverified"
        for row in matrix["build_architectures"]
    )
    assert all(row["evidence"] == [] for row in matrix["build_architectures"])


def test_cold_import_smoke_uses_fresh_interpreter():
    measurement = harness.measure_cold_import(timeout_s=60.0)

    assert measurement["ok"], measurement["stderr"]
    assert measurement["scope"] == "source_tree"
    assert measurement["returncode"] == 0
    assert measurement["wall_ms"] > 0.0


def test_tensor_bytes_counts_dataclass_results_without_double_counting_storage():
    from dataclasses import dataclass

    @dataclass
    class Result:
        value: torch.Tensor
        alias: torch.Tensor

    value = torch.zeros(8, dtype=torch.float32)
    assert harness.tensor_bytes(Result(value=value, alias=value[:4])) == 32


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA timing requires CUDA")
def test_cuda_event_and_wall_clock_are_both_recorded():
    result, measurement = harness.benchmark_operation(
        lambda: torch.arange(128, device="cuda").square(), warmup=0, repeats=2
    )

    assert result.shape == (128,)
    assert measurement.first.wall_ms > 0.0
    assert measurement.first.cuda_event_ms is not None
    assert measurement.steady_wall_median_ms > 0.0
    assert measurement.steady_cuda_median_ms is not None
    assert measurement.memory["peak_allocated_bytes"] > 0
