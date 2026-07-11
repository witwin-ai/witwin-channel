import pytest
import torch

from benchmarks.bench_bdpt_basic import run_benchmark
from witwin.channel_native.core.kernels.extension import build_info
from tests.support.bin.benchmark_single_plane_bdpt_native_vs_original import (
    run_benchmark as run_single_plane_native_vs_original,
)


def test_bdpt_basic_performance_gate_smoke():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for BDPT benchmark gate")

    result = run_benchmark(samples=64, grid_size=8)

    assert result["bdpt_seconds"] <= result["mc_basic_seconds"] * 1.25 + 0.05
    assert result["launch_count"] >= 1
    assert result["accumulation_strategy"] in {"atomic", "staged", "compact"}


def test_single_plane_bdpt_native_is_faster_than_original_channel_when_bridge_available():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for native-vs-original BDPT benchmark gate")
    try:
        native_info = build_info()
    except ModuleNotFoundError:
        pytest.skip("_channel_native is required for native-vs-original BDPT benchmark gate")
    if native_info["rayd_integration"] != "source-linked":
        pytest.skip("source-linked RayD is required for native-vs-original BDPT benchmark gate")

    try:
        result = run_single_plane_native_vs_original(
            samples=64,
            grid_size=4,
            warmup_runs=0,
            repeats=1,
            min_speedup=1.25,
            strict_gates=True,
        )
    except RuntimeError as exc:
        if "original Channel single-plane BDPT subprocess failed" not in str(exc):
            raise
        pytest.skip(f"original Channel comparison baseline is unavailable: {exc}")

    assert result["performance"]["native_faster_than_original"] is True
    assert result["performance"]["native_speedup_vs_original"] >= 1.25
    assert result["delta"]["relative_sum_error"] <= 0.75
    assert any(gate["name"] == "relative_sum_error" and gate["passed"] for gate in result["gates"])
