from __future__ import annotations

import pytest

from tests.support.bin._sionna_path_solver_benchmark import (
    format_benchmark_summary,
    run_path_solver_benchmark,
    run_path_solver_stress_matrix,
)
from witwin.channel import load_sionna_rt


pytestmark = pytest.mark.gpu


def _require_sionna():
    try:
        load_sionna_rt(prefer_local=True)
    except Exception as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"Local Sionna RT reference is unavailable: {exc}")


def test_path_solver_benchmark_smoke_payload_has_expected_fields():
    _require_sionna()

    payload = run_path_solver_benchmark(
        scenario_name="los_3d",
        tx_count=2,
        rx_count=2,
        warmup=0,
        repeats=1,
    )

    assert payload["benchmark"] == "path_solver_sionna_compare"
    assert payload["scenario"]["name"] == "los_3d"
    assert payload["scenario"]["tx_count"] == 2
    assert payload["scenario"]["rx_count"] == 2
    assert payload["witwin"]["summary"]["signature_counts"] == {"los": 4}
    assert payload["sionna"]["summary"]["signature_counts"] == {"los": 4}
    assert payload["comparison"]["signature_match"]
    assert "witwin_median=" in format_benchmark_summary(payload)


def test_path_solver_stress_matrix_smoke_runs_small_grid():
    _require_sionna()

    payload = run_path_solver_stress_matrix(
        scenario_names=["reflection_3d"],
        tx_counts=[1, 2],
        rx_counts=[1, 2],
        warmup=0,
        repeats=1,
    )

    assert payload["benchmark"] == "path_solver_sionna_stress_matrix"
    assert payload["matrix_config"]["scenario_names"] == ["reflection_3d"]
    assert len(payload["results"]) == 4
    assert all(result["scenario"]["name"] == "reflection_3d" for result in payload["results"])

