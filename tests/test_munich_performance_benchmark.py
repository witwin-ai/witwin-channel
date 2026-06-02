from __future__ import annotations

from tests.support.bin import benchmark_munich_performance as bench


def _case(case_id: str, *, workload_key: str, median_ms: float):
    return {
        "case_id": case_id,
        "workload_key": workload_key,
        "profile": {"median_ms": median_ms},
    }


def test_default_args_cover_munich_path_basic_and_bdpt():
    args = bench.build_parser().parse_args([])

    assert bench.parse_cases(args.cases) == bench.DEFAULT_CASES
    assert "path_order2" in bench.DEFAULT_CASES
    assert "mc_basic_order1" in bench.DEFAULT_CASES
    assert "mc_bdpt_order2" in bench.DEFAULT_CASES
    assert args.mc_grid_size == 256
    assert args.mc_samples_per_tx == 1_000_000
    assert args.path_samples == 1_000_000
    assert args.mc_accumulation_backend == "auto"
    assert args.diffraction_accumulate_primal == "auto"


def test_parse_cases_accepts_all_and_rejects_unknown_tokens():
    assert bench.parse_cases("mc_basic_order1, path_order1") == (
        "mc_basic_order1",
        "path_order1",
    )
    assert bench.parse_cases("all") == bench.AVAILABLE_CASES

    try:
        bench.parse_cases("mc_basic_order1,")
    except ValueError as exc:
        assert "empty token" in str(exc)
    else:
        raise AssertionError("parse_cases should reject empty case tokens")

    try:
        bench.parse_cases("deterministic_order1")
    except ValueError as exc:
        assert "unknown Munich performance case" in str(exc)
    else:
        raise AssertionError("parse_cases should reject unsupported cases")


def test_regression_gate_fails_large_slowdown_for_same_setup():
    current = _case("mc_bdpt_order2", workload_key="same", median_ms=260.0)
    baseline = _case("mc_bdpt_order2", workload_key="same", median_ms=100.0)

    gate = bench.compare_case_to_baseline(
        current,
        baseline,
        max_regression_factor=2.0,
    )

    assert gate["status"] == "compared"
    assert gate["passed"] is False
    assert gate["ratio"] == 2.6


def test_regression_gate_passes_same_setup_within_tolerance():
    current = _case("path_order2", workload_key="same", median_ms=175.0)
    baseline = _case("path_order2", workload_key="same", median_ms=100.0)

    gate = bench.compare_case_to_baseline(
        current,
        baseline,
        max_regression_factor=2.0,
    )

    assert gate["passed"] is True
    assert gate["ratio"] == 1.75


def test_regression_gate_refuses_different_setups():
    current = _case("mc_basic_order1", workload_key="grid512", median_ms=210.0)
    baseline = _case("mc_basic_order1", workload_key="grid256", median_ms=100.0)

    gate = bench.compare_case_to_baseline(
        current,
        baseline,
        max_regression_factor=2.0,
    )

    assert gate["status"] == "setup_mismatch"
    assert gate["passed"] is False


def test_attach_baseline_gates_marks_strict_missing_baseline_as_failure():
    cases = [_case("path_order1", workload_key="same", median_ms=10.0)]

    gates = bench.attach_baseline_gates(
        cases,
        None,
        max_regression_factor=2.0,
        strict=True,
    )

    assert gates["status"] == "not_configured"
    assert gates["passed"] is False
    assert gates["failed_cases"] == ["path_order1"]
    assert cases[0]["gate"]["status"] == "no_baseline"
