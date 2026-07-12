from __future__ import annotations

import argparse

from benchmarks.bench_solver_peak_memory import run


def test_peak_memory_report_includes_100m_preflight_artifact():
    report = run(
        argparse.Namespace(tx=1, rx=1024, depth=3, gpu_budget_gib=16.0)
    )

    assert report["schema"] == {
        "name": "witwin.channel_native.performance",
        "version": "1.0.0",
    }
    by_samples = {row["samples"]: row for row in report["results"]}
    assert by_samples[1_000]["memory_safe"] is True
    assert by_samples[100_000_000]["memory_safe"] is False
    assert "before launch" in by_samples[100_000_000]["preflight_error"]
