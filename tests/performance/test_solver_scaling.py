from __future__ import annotations

import argparse

import pytest

from benchmarks import bench_solver_scaling as scaling
from witwin.channel.runtime import MemoryBudgetError


@pytest.mark.parametrize("solver", ("basic", "bdpt"))
def test_scaling_mc_operations_use_the_explicit_workspace_budget(solver: str):
    budget = 16 * (1 << 30)
    operation = scaling._operation(
        solver,
        scaling._expanded_scene(1, 1),
        depth=1,
        samples=1_000_000,
        workspace_limit_bytes=budget,
    )
    config = next(
        cell.cell_contents
        for cell in operation.__closure__ or ()
        if hasattr(cell.cell_contents, "workspace_limit_bytes")
    )

    assert config.workspace_limit_bytes == budget


def test_scaling_records_memory_preflight_rejection_and_continues(monkeypatch):
    monkeypatch.setattr(scaling.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(scaling, "_operation", lambda *args, **kwargs: object())

    def reject(*args, **kwargs):
        raise MemoryBudgetError("over budget before launch")

    monkeypatch.setattr(scaling, "benchmark_operation", reject)
    args = argparse.Namespace(
        solvers="basic",
        tx="1",
        rx="1",
        depths="1",
        samples="1000000",
        warmup=0,
        repeats=1,
        gpu_budget_gib=16.0,
    )

    report = scaling.run(args)

    assert report["results"][0]["status"] == "preflight_rejected"
    assert report["results"][0]["timing"] is None
    assert "before launch" in report["results"][0]["preflight_error"]
