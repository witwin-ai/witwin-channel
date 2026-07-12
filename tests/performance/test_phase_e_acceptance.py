from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from benchmarks.bench_phase_e_acceptance import (
    DEFAULT_BUDGET,
    FULL_DEPTHS,
    FULL_ENDPOINT_PAIRS,
    FULL_GRID_SHAPES,
    FULL_MC_SAMPLES,
    PREFLIGHT_MC_SAMPLES,
    evaluate_budget,
    load_budget,
    preflight_rows,
    profile_cases,
    profile_matrix,
    run_profile,
)


ROOT = Path(__file__).resolve().parents[2]
SCHEMA = ROOT / "benchmarks/schemas/phase-e-performance.schema.json"


def test_full_profile_declares_every_required_phase_e_axis() -> None:
    matrix = profile_matrix("full")

    assert matrix["endpoint_pairs"] == [list(pair) for pair in FULL_ENDPOINT_PAIRS]
    assert matrix["grid_shapes"] == [list(shape) for shape in FULL_GRID_SHAPES]
    assert matrix["depths"] == list(FULL_DEPTHS)
    assert matrix["mc_samples"] == list(FULL_MC_SAMPLES)
    assert matrix["preflight_mc_samples"] == [PREFLIGHT_MC_SAMPLES]
    assert matrix["scenarios"] == [
        "analytic",
        "three_cube",
        "terrain",
        "munich_full",
        "sf_full",
    ]

    cases = profile_cases("full")
    assert {case.solver for case in cases} == {
        "path",
        "deterministic",
        "basic",
        "bdpt",
    }
    assert FULL_MC_SAMPLES[-1] in {
        case.samples for case in cases if case.solver in {"basic", "bdpt"}
    }


def test_100m_preflight_is_allocation_free_and_rejected_for_every_depth() -> None:
    before = torch.cuda.memory_allocated() if torch.cuda.is_available() else None
    rows = preflight_rows()
    after = torch.cuda.memory_allocated() if torch.cuda.is_available() else None

    assert len(rows) == 2 * len(FULL_DEPTHS)
    assert {row["solver"] for row in rows} == {"basic", "bdpt"}
    assert {row["depth"] for row in rows} == set(FULL_DEPTHS)
    assert {row["samples"] for row in rows} == {PREFLIGHT_MC_SAMPLES}
    assert all(row["rejected_before_launch"] for row in rows)
    assert all("before launch" in row["error"] for row in rows)
    assert before == after


def test_sm120_budget_is_fixed_and_unknown_sm_is_explicitly_non_gating() -> None:
    budget = load_budget(DEFAULT_BUDGET)

    assert budget["environment"]["sm"] == 120
    assert set(budget["profiles"]) == {"reduced", "full"}
    for profile in budget["profiles"].values():
        assert set(profile["solver_budgets"]) == {
            "path",
            "deterministic",
            "basic",
            "bdpt",
        }
    gate = evaluate_budget(
        [],
        {"runtime": {"device": {"sm": 89}}},
        budget,
    )
    assert gate == {
        "status": "not_gating_environment",
        "eligible": False,
        "passed": None,
        "actual_sm": 89,
        "target_sm": 120,
        "checks": [],
    }


def test_phase_e_schema_is_strict_at_every_acceptance_boundary() -> None:
    schema = json.loads(SCHEMA.read_text(encoding="utf-8"))

    assert schema["additionalProperties"] is False
    for definition in (
        "matrix",
        "case",
        "scene",
        "measurement",
        "timing",
        "memory",
        "torchAllocator",
        "deviceWide",
        "correctness",
        "preflight",
        "gate",
    ):
        assert schema["$defs"][definition]["additionalProperties"] is False


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Phase E gate requires CUDA")
def test_reduced_profile_runs_all_four_solvers_and_validates_schema() -> None:
    jsonschema = pytest.importorskip("jsonschema")
    report = run_profile(
        profile="reduced",
        asset_root=None,
        warmup=0,
        repeats=1,
    )

    assert [row["case"]["solver"] for row in report["measurements"]] == [
        "path",
        "deterministic",
        "basic",
        "bdpt",
    ]
    assert all(row["correctness"]["finite"] for row in report["measurements"])
    assert all(
        row["correctness"]["checksum_abs_sum"] > 0.0
        for row in report["measurements"]
    )
    assert all(
        row["timing"]["pipeline_build_ms"] is None
        and row["timing"]["optix_scene_build_ms"] >= 0.0
        and row["memory"]["optix_build_bytes"] >= 0
        for row in report["measurements"]
    )
    schema = json.loads(SCHEMA.read_text(encoding="utf-8"))
    jsonschema.validate(report, schema)
    if report["gate"]["actual_sm"] == 120:
        assert report["gate"]["eligible"] is True
        assert report["gate"]["passed"] is True
