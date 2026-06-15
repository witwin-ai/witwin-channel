import pytest
import torch

from pathlib import Path

from benchmarks.bench_bdpt_munich import run_benchmark
from tests.support.bin import benchmark_munich_bdpt_native_vs_original as munich_parity


def test_reduced_munich_bdpt_nonzero_component_smoke():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for reduced Munich BDPT parity")

    result = run_benchmark(samples=64, grid_size=8, warmup_runs=0, repeats=1, emit_artifacts=False)

    assert result["native_total_sum"] >= 0.0
    assert result["native_seconds"] > 0.0
    assert result["repeats"] == 1
    assert result["component_nonzero_min"] == 1.0
    assert result["all_zero_component_map"] is False


def test_munich_native_vs_original_benchmark_runs_original_channel_subprocess():
    repo = Path(__file__).resolve().parents[3]
    source = (repo / "tests" / "support" / "bin" / "benchmark_munich_bdpt_native_vs_original.py").read_text()

    assert "from benchmarks.bench_bdpt_munich import run_benchmark" not in source
    assert "subprocess.run" in source
    assert "from witwin.channel.core.scene import Mesh, ReceiverGrid, Scene, Transmitter" in source
    assert "from witwin.channel.montecarlo import Config, IntegratorOptions, Tuning, solve" in source
    assert 'integrator="bdpt"' in source
    assert "native_speedup_vs_original_solve" in source


def test_reduced_munich_native_vs_original_strict_parity_gate():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for reduced Munich native/original BDPT parity")

    args = munich_parity._parser().parse_args(
        [
            "--samples",
            "16",
            "--grid-size",
            "4",
            "--max-depth",
            "1",
            "--warmup-runs",
            "1",
            "--original-timeout-seconds",
            "240",
            "--strict-gates",
        ]
    )
    payload = munich_parity.run(args)

    assert all(gate["passed"] for gate in payload["gates"])
    assert payload["performance"]["native_faster_than_original"] is True
