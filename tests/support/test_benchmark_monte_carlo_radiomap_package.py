from __future__ import annotations

from types import SimpleNamespace

import drjit as dr
import pytest

from tests.support.bin import benchmark_monte_carlo_radiomap_package as bench


def test_benchmark_runtime_report_is_in_tree():
    report = bench.benchmark_environment_report()

    assert "python" in report
    assert "platform" in report
    assert "drjit" in report


def test_parser_exposes_rayd_diffraction_modes_and_gates():
    args = bench._parser().parse_args([
        "--mode",
        "basic-rayd-diffraction",
        "--strict-gates",
        "--min-speedup",
        "1.5",
    ])

    assert args.mode == "basic-rayd-diffraction"
    assert args.strict_gates is True
    assert args.min_speedup == 1.5

    choices = bench._parser()._option_string_actions["--mode"].choices
    assert "basic-rayd-diffraction" in choices
    assert "bdpt-rayd-diffraction" in choices
    assert "path-rayd-diffraction" in choices


def test_diffraction_benchmark_configs_select_requested_backends():
    drjit_config = bench._monte_carlo_diffraction_config(
        integrator="basic",
        samples_per_tx=64,
        seed=3,
        max_diffractions=1,
        accumulate_primal="drjit",
        reflection_coupled=False,
    )
    rayd_config = bench._monte_carlo_diffraction_config(
        integrator="bdpt",
        samples_per_tx=64,
        seed=3,
        max_diffractions=1,
        accumulate_primal="rayd_optix",
        reflection_coupled=True,
    )
    path_config = bench._path_diffraction_config(
        samples_per_tx=64,
        max_diffractions=1,
        accumulate_primal="rayd_optix",
    )

    assert drjit_config.tuning.diffraction_execution.accumulate_primal == "drjit"
    assert rayd_config.tuning.diffraction_execution.accumulate_primal == "rayd_optix"
    assert rayd_config.tuning.enable_bdpt_reflection_coupled_diffraction is True
    assert path_config.tuning.diffraction_execution.accumulate_primal == "rayd_optix"


def test_gate_helpers_require_speedup_and_path_count_parity():
    passed = bench._diffraction_speedup_gate(
        baseline={"median_ms": 20.0},
        candidate={"median_ms": 5.0},
        min_speedup=2.0,
    )
    failed = bench._diffraction_speedup_gate(
        baseline={"median_ms": 20.0},
        candidate={"median_ms": 15.0},
        min_speedup=2.0,
    )
    path_gate = bench._path_count_gate(
        baseline={"path_count": 3},
        candidate={"path_count": 3},
    )

    assert passed["passed"] is True
    assert failed["passed"] is False
    assert path_gate["passed"] is True


def test_path_count_metric_accepts_drjit_result_arrays():
    result = SimpleNamespace(num_paths=dr.scalar.Array3u(1, 2, 3))

    assert bench._path_count_metric(result) == 6


def test_path_rayd_diffraction_rejects_non_wall_scene():
    args = bench._parser().parse_args([
        "--mode",
        "path-rayd-diffraction",
        "--scene",
        "three_cubes",
    ])

    with pytest.raises(ValueError, match="supports --scene wall only"):
        bench._path_rayd_diffraction_benchmark(args)
