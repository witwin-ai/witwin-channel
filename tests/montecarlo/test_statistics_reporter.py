# Copyright Xingyu Chen.
# Tests statistics reporter.

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from benchmarks.statistical_gate import (
    Observation,
    evaluate_thresholds,
    summarize_observations,
)


def test_statistical_summary_reports_required_phase_c_fields():
    summary = summarize_observations(
        [
            Observation(3, 1.0, 2, 2),
            Observation(5, 2.0, 1, 2),
            Observation(7, None, 0, 0, "RuntimeError: failed"),
        ],
        reference=1.5,
    )

    assert summary["mean"] == pytest.approx(1.5)
    assert summary["sample_variance"] == pytest.approx(0.5)
    assert summary["finite_ratio"] == pytest.approx(0.75)
    assert summary["failure_rate"] == pytest.approx(1.0 / 3.0)
    assert summary["relative_bias"] == 0.0
    assert summary["ci95"]["half_width"] > 0.0
    assert summary["ci99"]["half_width"] > summary["ci95"]["half_width"]


def test_thresholds_are_evaluated_without_mutating_the_summary():
    summary = summarize_observations(
        [Observation(seed, value, 1, 1) for seed, value in ((3, 0.9), (5, 1.1))],
        reference=1.0,
    )
    before = json.dumps(summary, sort_keys=True)
    checks = evaluate_thresholds(
        summary,
        {
            "failure_rate_max": 0.0,
            "finite_ratio_min": 1.0,
            "relative_bias_max": 0.01,
            "reference_in_ci99": True,
            "relative_ci95_half_width_max": 2.0,
        },
    )

    assert all(checks.values())
    assert json.dumps(summary, sort_keys=True) == before


def test_nonfinite_scalar_is_not_admitted_to_mean():
    summary = summarize_observations(
        [Observation(3, math.nan, 0, 1), Observation(5, 2.0, 1, 1)]
    )

    assert summary["mean"] == 2.0
    assert summary["success_count"] == 1
    assert summary["finite_ratio"] == 0.5


def test_full_seed_interval_uses_student_t_df15():
    summary = summarize_observations(
        [Observation(seed, float(seed), 1, 1) for seed in range(16)]
    )
    standard_error = summary["standard_error"]

    assert summary["ci95"]["half_width"] == pytest.approx(2.131 * standard_error)
    assert summary["ci99"]["half_width"] == pytest.approx(2.947 * standard_error)


def test_full_gate_has_sixteen_fixed_seeds_and_all_required_cases():
    root = Path(__file__).resolve().parents[2]
    gate = json.loads(
        (root / "benchmarks" / "gates" / "phase_c_statistics.v1.json").read_text(
            encoding="utf-8"
        )
    )

    assert len(gate["full_seeds"]) == 16
    assert len(set(gate["full_seeds"])) == 16
    assert set(gate["cases"]) == {
        "bdpt_wedge_diffraction",
        "mc_basic_rough_scattering",
        "bdpt_mixed_reflection_transmission",
    }
    for case in gate["cases"].values():
        assert case["thresholds"]["failure_rate_max"] == 0.0
        assert case["thresholds"]["finite_ratio_min"] == 1.0


def test_wedge_gate_uses_adr018_deterministic_utd_reference():
    root = Path(__file__).resolve().parents[2]
    gate = json.loads(
        (root / "benchmarks" / "gates" / "phase_c_statistics.v1.json").read_text(
            encoding="utf-8"
        )
    )

    # ADR-018 retired the crude stochastic estimate fossilized as 4.66e-05.
    # Standalone BDPT diffraction now consumes the deterministic UTD oracle.
    assert gate["cases"]["bdpt_wedge_diffraction"]["reference"] == pytest.approx(
        2.1433029573358908e-08, rel=0.0, abs=0.0
    )