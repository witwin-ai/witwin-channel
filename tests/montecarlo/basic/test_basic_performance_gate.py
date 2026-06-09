import json
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
    assert isinstance(payload["raydn_native"], bool)
    assert payload["accumulation_strategy"] == "atomic_add"
    assert payload["performance_budget_ms"] is None
