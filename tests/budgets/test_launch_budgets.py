# Copyright Xingyu Chen.
# Enforce per-solve kernel launch, host sync and copy budgets on small, medium and large scenes.

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from benchmarks.baseline import CASES, run_worker

BUDGETS_FILE = Path(__file__).with_name("launch_budgets.json")
BUDGETS = json.loads(BUDGETS_FILE.read_text(encoding="utf-8"))
METRICS = tuple(BUDGETS["metrics"])


def test_budget_file_names_real_cases_and_metrics():
    assert set(BUDGETS["cases"]) <= set(CASES)
    assert set(BUDGETS.get("peak_alloc_mib", {})) <= set(BUDGETS["cases"])
    for name, bounds in BUDGETS["cases"].items():
        assert set(bounds) == set(METRICS), name
        assert all(isinstance(v, int) and v >= 0 for v in bounds.values()), name


@pytest.mark.skipif(not torch.cuda.is_available(), reason="budget measurement needs CUDA")
@pytest.mark.parametrize("name", sorted(BUDGETS["cases"]))
def test_solve_stays_within_budget(name):
    """One warm solve in a fresh process, counted by the CUDA profiler, against the stored bounds."""
    record = run_worker(CASES[name], counts_only=True, timeout_s=600.0)
    assert "error" not in record, f"{name}: {record['error']}"
    counts = record["counts"]
    bounds = BUDGETS["cases"][name]
    over = {m: (counts[m], bounds[m]) for m in METRICS if counts[m] > bounds[m]}
    peak_bound = BUDGETS.get("peak_alloc_mib", {}).get(name)
    if peak_bound is not None and record["peak_alloc_mib"] > peak_bound:
        over["peak_alloc_mib"] = (round(record["peak_alloc_mib"], 1), peak_bound)
    assert not over, f"{name} over budget (measured, bound): {over}"
