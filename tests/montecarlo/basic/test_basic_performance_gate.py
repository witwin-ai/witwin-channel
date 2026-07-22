import json
from pathlib import Path
import subprocess
import sys

import pytest
import torch


def test_basic_benchmark_outputs_required_json_fields():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for MC basic benchmark")

    completed = subprocess.run(
        [
            sys.executable,
            "benchmarks/bench_mc_basic.py",
            "--scene",
            "small",
            "--samples",
            "256",
            "--json",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(completed.stdout)

    assert payload["scene"] == "small"
    assert payload["samples"] == 256
    assert payload["wall_time_ms"] >= 0.0
    assert payload["launch_count"] >= 0
    assert payload["intermediate_bytes"] >= 0
    assert payload["output_bytes"] > 0
    assert isinstance(payload["rayd_native"], bool)
    assert payload["accumulation_strategy"] == "atomic_add"
    assert "performance_budget_ms" not in payload

    root = Path(__file__).resolve().parents[3]
    budget = json.loads(
        (root / "benchmarks/gates/phase_e_performance.sm120.v1.json").read_text(
            encoding="utf-8"
        )
    )["profiles"]["reduced"]["solver_budgets"]["basic"]
    assert budget["steady_wall_p95_ms"] > 0.0
    assert budget["torch_peak_allocated_bytes"] > 0
    assert budget["output_bytes"] > 0
