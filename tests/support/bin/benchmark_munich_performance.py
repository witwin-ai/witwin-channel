"""Unified Munich performance regression benchmark for path and Monte Carlo solvers."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from tests.support.bin import benchmark_path_solver_munich_vs_sionna as path_bench
from tests.support.bin import validate_path_solver_munich as munich_base


DEFAULT_CASES = (
    "path_order1",
    "path_order2",
    "mc_basic_order1",
    "mc_bdpt_order1",
    "mc_bdpt_order2",
)
AVAILABLE_CASES = DEFAULT_CASES + (
    "path_order0",
    "path_order3",
    "mc_basic_order0",
    "mc_bdpt_order0",
    "mc_bdpt_order3",
)
DEFAULT_OUTPUT_JSON = (
    munich_base.CHANNEL_ROOT
    / "docs"
    / "dev"
    / "optimization"
    / "munich_solver_performance.json"
)

DEFAULT_TX_POS = (8.5, 21.0, 27.0)
DEFAULT_BOUNDS = ((-120.0, 120.0), (-120.0, 140.0))
DEFAULT_PLANE_Z = 1.5


def parse_cases(text: str) -> tuple[str, ...]:
    if str(text).strip() == "all":
        return AVAILABLE_CASES
    cases: list[str] = []
    for token in str(text).split(","):
        case = token.strip()
        if not case:
            raise ValueError("empty token in --cases")
        if case not in AVAILABLE_CASES:
            raise ValueError(
                f"unknown Munich performance case {case!r}; "
                f"available cases are {', '.join(AVAILABLE_CASES)}"
            )
        cases.append(case)
    if not cases:
        raise ValueError("--cases must contain at least one case")
    return tuple(cases)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sionna-source-root", type=Path, default=munich_base.DEFAULT_SIONNA_SOURCE_ROOT)
    parser.add_argument("--munich-xml", type=Path, default=munich_base.DEFAULT_MUNICH_XML)
    parser.add_argument("--frequency-hz", type=float, default=2.4e9)
    parser.add_argument("--cases", type=str, default=",".join(DEFAULT_CASES))
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--path-samples", type=int, default=1_000_000)
    parser.add_argument("--path-max-bounces", type=int, default=1)
    parser.add_argument("--path-max-num-paths", type=int, default=256)
    parser.add_argument("--path-diffraction-state-budget", type=int, default=4096)
    parser.add_argument("--path-inserted-reflection-state-budget", type=int, default=2048)
    parser.add_argument("--path-accumulate-primal", choices=("auto", "drjit", "rayd_optix"), default="auto")
    parser.add_argument("--path-enable-rd-diffraction", action="store_true", default=False)
    parser.add_argument("--mc-grid-size", type=int, default=256)
    parser.add_argument("--mc-samples-per-tx", type=int, default=1_000_000)
    parser.add_argument("--mc-max-bounces", type=int, default=5)
    parser.add_argument("--mc-seed", type=int, default=11)
    parser.add_argument(
        "--mc-accumulation-backend",
        choices=("auto", "native_monte_carlo", "rayd_reflection_accumulation"),
        default="auto",
    )
    parser.add_argument("--mc-diffraction-state-budget", type=int, default=None)
    parser.add_argument("--mc-inserted-reflection-state-budget", type=int, default=None)
    parser.add_argument("--mc-disable-bdpt-coupled-suffix", action="store_true", default=False)
    parser.add_argument("--mc-shadow-boundary-mode", choices=("none", "utd_power_smoothing"), default="none")
    parser.add_argument("--diffraction-accumulate-primal", choices=("auto", "drjit", "rayd_optix"), default="auto")
    parser.add_argument("--baseline-json", type=Path, default=None)
    parser.add_argument("--max-regression-factor", type=float, default=2.0)
    parser.add_argument("--strict-gates", action="store_true", default=False)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--json", action="store_true", default=False)
    return parser


def _jsonable(value: Any) -> Any:
    return munich_base._jsonable(value)


def _gpu_info() -> dict[str, Any]:
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,memory.used,driver_version",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=3.0,
        ).strip()
    except Exception as exc:
        return {"available": False, "error": str(exc)}
    fields = [field.strip() for field in output.split(",", 3)]
    if len(fields) != 4:
        return {"available": True, "raw": output}
    return {
        "available": True,
        "name": fields[0],
        "memory_total_mib": int(fields[1]),
        "memory_used_mib": int(fields[2]),
        "driver_version": fields[3],
    }


def _timed(label: str, operation, sync_result, *, warmup: int, repeats: int) -> dict[str, Any]:
    import drjit as dr
    import numpy as np

    dr.sync_thread()
    if hasattr(dr, "flush_malloc_cache"):
        dr.flush_malloc_cache()
    for _ in range(max(0, int(warmup))):
        sync_result(operation())

    samples_ms: list[float] = []
    result = None
    for _ in range(max(1, int(repeats))):
        start = time.perf_counter()
        result = operation()
        sync_result(result)
        samples_ms.append((time.perf_counter() - start) * 1000.0)

    return {
        "label": str(label),
        "samples_ms": samples_ms,
        "median_ms": float(np.median(samples_ms)),
        "mean_ms": float(np.mean(samples_ms)),
        "min_ms": float(np.min(samples_ms)),
        "max_ms": float(np.max(samples_ms)),
        "result": result,
    }


def _sync_monte_carlo_result(result) -> None:
    import drjit as dr

    dr.eval(result.path_gain)
    dr.sync_thread()


def _radio_map_stats(value) -> dict[str, Any]:
    import numpy as np

    array = np.asarray(value, dtype=np.float64)
    flat = array.reshape(-1)
    if flat.size == 0:
        return {"shape": list(array.shape), "finite": True, "sum": 0.0, "nonzero": 0}
    return {
        "shape": list(array.shape),
        "finite": bool(np.isfinite(flat).all()),
        "sum": float(np.nansum(flat, dtype=np.float64)),
        "nonzero": int(np.count_nonzero(flat > 0.0)),
        "min": float(np.nanmin(flat)),
        "max": float(np.nanmax(flat)),
    }


def _case_order(case_id: str) -> int:
    return int(case_id.rsplit("order", 1)[1])


def _case_solver(case_id: str) -> str:
    if case_id.startswith("path_"):
        return "path"
    if case_id.startswith("mc_basic_"):
        return "mc_basic"
    if case_id.startswith("mc_bdpt_"):
        return "mc_bdpt"
    raise ValueError(f"unsupported case id {case_id!r}")


def _workload_key(payload: Mapping[str, Any]) -> str:
    text = json.dumps(_jsonable(payload), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _path_case_setup(args: argparse.Namespace, *, case_id: str) -> dict[str, Any]:
    order = _case_order(case_id)
    return {
        "scene": "munich",
        "solver": "path",
        "case_id": case_id,
        "munich_xml": str(Path(args.munich_xml)),
        "sionna_source_root": str(Path(args.sionna_source_root)),
        "frequency_hz": float(args.frequency_hz),
        "num_samples": int(args.path_samples),
        "max_bounces": int(args.path_max_bounces),
        "max_diffraction_order": int(order),
        "max_num_paths": int(args.path_max_num_paths),
        "diffraction_state_budget": int(args.path_diffraction_state_budget),
        "inserted_reflection_state_budget": int(args.path_inserted_reflection_state_budget),
        "accumulate_primal": str(args.path_accumulate_primal),
        "enable_rd_diffraction": bool(args.path_enable_rd_diffraction),
        "tx_positions": munich_base.DEFAULT_TX_POSITIONS,
        "rx_positions": munich_base.DEFAULT_RX_POSITIONS,
    }


def _mc_case_setup(args: argparse.Namespace, *, case_id: str) -> dict[str, Any]:
    order = _case_order(case_id)
    solver = _case_solver(case_id)
    return {
        "scene": "munich",
        "solver": solver,
        "case_id": case_id,
        "munich_xml": str(Path(args.munich_xml)),
        "sionna_source_root": str(Path(args.sionna_source_root)),
        "frequency_hz": float(args.frequency_hz),
        "grid_size": int(args.mc_grid_size),
        "samples_per_tx": int(args.mc_samples_per_tx),
        "max_bounces": int(args.mc_max_bounces),
        "max_diffraction_order": int(order),
        "seed": int(args.mc_seed),
        "accumulation_backend": str(args.mc_accumulation_backend),
        "diffraction_accumulate_primal": str(args.diffraction_accumulate_primal),
        "diffraction_state_budget": args.mc_diffraction_state_budget,
        "inserted_reflection_state_budget": args.mc_inserted_reflection_state_budget,
        "bdpt_coupled_suffix": not bool(args.mc_disable_bdpt_coupled_suffix),
        "shadow_boundary_mode": str(args.mc_shadow_boundary_mode),
        "tx_pos": tuple(float(v) for v in DEFAULT_TX_POS),
        "bounds": DEFAULT_BOUNDS,
        "plane_z": float(DEFAULT_PLANE_Z),
    }


def _path_args_for_case(args: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(
        max_bounces=int(args.path_max_bounces),
        max_num_paths=int(args.path_max_num_paths),
        diffraction_state_budget=int(args.path_diffraction_state_budget),
        inserted_reflection_state_budget=int(args.path_inserted_reflection_state_budget),
        enable_rd_diffraction=bool(args.path_enable_rd_diffraction),
        accumulate_primal=str(args.path_accumulate_primal),
        warmup=int(args.warmup),
        repeats=int(args.repeats),
    )


def _run_path_case(args: argparse.Namespace, scene, *, case_id: str) -> dict[str, Any]:
    setup = _path_case_setup(args, case_id=case_id)
    order = _case_order(case_id)
    case = {
        "case_id": case_id,
        "solver": "path",
        "setup": setup,
        "workload_key": _workload_key(setup),
    }
    try:
        result = path_bench._run_witwin_case(
            _path_args_for_case(args),
            scene,
            samples=int(args.path_samples),
            order=int(order),
        )
    except Exception as exc:
        result = {
            "ok": False,
            "error": f"{type(exc).__name__}: {exc}",
            "profile": None,
            "stats": None,
            "metadata": None,
        }
    return {**case, **result}


def _build_monte_carlo_scene(args: argparse.Namespace):
    import witwin.channel as wc

    scene = wc.Scene.load_mitsuba(
        Path(args.munich_xml),
        source_root=Path(args.sionna_source_root),
        frequency=float(args.frequency_hz),
        merge_shapes=True,
        device="cuda",
    )
    scene.add(wc.Transmitter("tx", tuple(float(v) for v in DEFAULT_TX_POS)))
    scene.add(
        wc.ReceiverGrid(
            "rm",
            axis="z",
            position=float(DEFAULT_PLANE_Z),
            bounds=DEFAULT_BOUNDS,
            grid_shape=(int(args.mc_grid_size), int(args.mc_grid_size)),
        )
    )
    return scene


def _edge_policy_for_order(order: int):
    if int(order) <= 0:
        return None
    import witwin.channel as wc

    return wc.EdgePolicy(
        edge_selection_mode="all_edges",
        edge_diffraction=True,
        boundary_edge_policy="half_plane",
    )


def _monte_carlo_config(args: argparse.Namespace, *, integrator: str, order: int):
    import witwin.channel as wc

    return wc.montecarlo.Config(
        num_samples=int(args.mc_samples_per_tx),
        max_bounces=int(args.mc_max_bounces),
        max_diffraction_order=int(order),
        edge_policy=_edge_policy_for_order(int(order)),
        tuning=wc.montecarlo.Tuning(
            enable_rd_diffraction=bool(int(order) > 0),
            enable_bdpt_reflection_coupled_diffraction=not bool(args.mc_disable_bdpt_coupled_suffix),
            shadow_boundary_mode=str(args.mc_shadow_boundary_mode),
            shadow_boundary_backend="auto",
            shadow_boundary_max_candidate_factor=128.0,
            diffraction_state_budget=args.mc_diffraction_state_budget,
            inserted_reflection_state_budget=args.mc_inserted_reflection_state_budget,
            diffraction_execution={"accumulate_primal": str(args.diffraction_accumulate_primal)},
        ),
        integrator_options=wc.montecarlo.IntegratorOptions(
            integrator=str(integrator),
            samples_per_tx=int(args.mc_samples_per_tx),
            seed=int(args.mc_seed),
            accumulation_backend=str(args.mc_accumulation_backend),
            ad=False,
        ),
    )


def _run_monte_carlo_case(args: argparse.Namespace, scene, *, case_id: str) -> dict[str, Any]:
    import witwin.channel as wc

    setup = _mc_case_setup(args, case_id=case_id)
    solver = _case_solver(case_id)
    integrator = "basic" if solver == "mc_basic" else "bdpt"
    order = _case_order(case_id)
    case = {
        "case_id": case_id,
        "solver": solver,
        "setup": setup,
        "workload_key": _workload_key(setup),
    }
    try:
        config = _monte_carlo_config(args, integrator=integrator, order=order)
        profile = _timed(
            f"{case_id}_witwin",
            lambda: wc.montecarlo.solve(
                scene=scene,
                transmitter="tx",
                receiver="rm",
                config=config,
            ),
            _sync_monte_carlo_result,
            warmup=int(args.warmup),
            repeats=int(args.repeats),
        )
        result = profile.pop("result")
        return {
            **case,
            "ok": True,
            "error": None,
            "profile": profile,
            "stats": _radio_map_stats(result.path_gain),
            "metadata": {
                "solver": result.metadata.get("solver", {}),
                "monte_carlo": result.metadata.get("monte_carlo", {}),
                "accumulation_backend": result.metadata.get("accumulation_backend", {}),
                "timing": result.metadata.get("timing", {}),
            },
        }
    except Exception as exc:
        return {
            **case,
            "ok": False,
            "error": f"{type(exc).__name__}: {exc}",
            "profile": None,
            "stats": None,
            "metadata": None,
        }


def _load_baseline(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _baseline_case_map(baseline: Mapping[str, Any] | None) -> dict[str, Mapping[str, Any]]:
    if baseline is None:
        return {}
    cases = baseline.get("cases", ())
    if not isinstance(cases, list):
        return {}
    return {
        str(case.get("case_id")): case
        for case in cases
        if isinstance(case, Mapping) and "case_id" in case
    }


def compare_case_to_baseline(
    current: Mapping[str, Any],
    baseline: Mapping[str, Any] | None,
    *,
    max_regression_factor: float,
) -> dict[str, Any]:
    case_id = str(current.get("case_id"))
    if baseline is None:
        return {
            "case_id": case_id,
            "status": "no_baseline",
            "passed": None,
            "reason": "No baseline JSON was provided.",
        }
    if str(current.get("workload_key")) != str(baseline.get("workload_key")):
        return {
            "case_id": case_id,
            "status": "setup_mismatch",
            "passed": False,
            "current_workload_key": current.get("workload_key"),
            "baseline_workload_key": baseline.get("workload_key"),
            "reason": "Current and baseline workload keys differ; timings are not comparable.",
        }
    current_profile = current.get("profile")
    baseline_profile = baseline.get("profile")
    if not isinstance(current_profile, Mapping) or not isinstance(baseline_profile, Mapping):
        return {
            "case_id": case_id,
            "status": "missing_profile",
            "passed": False,
            "reason": "Current or baseline case is missing a timing profile.",
        }
    baseline_ms = float(baseline_profile.get("median_ms", 0.0))
    current_ms = float(current_profile.get("median_ms", 0.0))
    if baseline_ms <= 0.0 or current_ms <= 0.0:
        return {
            "case_id": case_id,
            "status": "invalid_timing",
            "passed": False,
            "current_median_ms": current_ms,
            "baseline_median_ms": baseline_ms,
            "reason": "Median timings must be positive.",
        }
    ratio = current_ms / baseline_ms
    return {
        "case_id": case_id,
        "status": "compared",
        "passed": bool(ratio <= float(max_regression_factor)),
        "current_median_ms": current_ms,
        "baseline_median_ms": baseline_ms,
        "ratio": ratio,
        "max_regression_factor": float(max_regression_factor),
    }


def attach_baseline_gates(
    cases: list[dict[str, Any]],
    baseline: Mapping[str, Any] | None,
    *,
    max_regression_factor: float,
    strict: bool,
) -> dict[str, Any]:
    baseline_cases = _baseline_case_map(baseline)
    gate_results = []
    for case in cases:
        baseline_case = baseline_cases.get(str(case["case_id"]))
        if baseline is not None and baseline_case is None:
            gate = {
                "case_id": str(case["case_id"]),
                "status": "missing_baseline_case",
                "passed": False,
                "reason": "Baseline JSON does not contain this case.",
            }
        else:
            gate = compare_case_to_baseline(
                case,
                baseline_case,
                max_regression_factor=float(max_regression_factor),
            )
        case["gate"] = gate
        gate_results.append(gate)

    failures = [
        gate
        for gate in gate_results
        if gate.get("passed") is False or (strict and gate.get("passed") is None)
    ]
    if baseline is None:
        status = "not_configured"
    elif failures:
        status = "failed"
    else:
        status = "passed"
    return {
        "status": status,
        "strict": bool(strict),
        "passed": bool(not failures),
        "max_regression_factor": float(max_regression_factor),
        "failed_cases": [str(gate.get("case_id")) for gate in failures],
        "results": gate_results,
    }


def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    munich_base._ensure_import_paths(Path(args.sionna_source_root))
    import drjit as dr

    selected_cases = parse_cases(str(args.cases))
    path_cases = [case for case in selected_cases if _case_solver(case) == "path"]
    mc_cases = [case for case in selected_cases if _case_solver(case) != "path"]

    path_scene = None
    if path_cases:
        path_scene = munich_base._build_witwin_scene(
            munich_xml=Path(args.munich_xml),
            sionna_source_root=Path(args.sionna_source_root),
            frequency_hz=float(args.frequency_hz),
            tx_positions=munich_base.DEFAULT_TX_POSITIONS,
            rx_positions=munich_base.DEFAULT_RX_POSITIONS,
        )
    mc_scene = _build_monte_carlo_scene(args) if mc_cases else None

    cases: list[dict[str, Any]] = []
    for case_id in selected_cases:
        if _case_solver(case_id) == "path":
            cases.append(_run_path_case(args, path_scene, case_id=case_id))
        else:
            cases.append(_run_monte_carlo_case(args, mc_scene, case_id=case_id))

    baseline = _load_baseline(args.baseline_json)
    gates = attach_baseline_gates(
        cases,
        baseline,
        max_regression_factor=float(args.max_regression_factor),
        strict=bool(args.strict_gates),
    )
    return {
        "scenario": {
            "scene": "munich",
            "munich_xml": str(Path(args.munich_xml)),
            "sionna_source_root": str(Path(args.sionna_source_root)),
            "frequency_hz": float(args.frequency_hz),
            "cases": selected_cases,
            "warmup": int(args.warmup),
            "repeats": int(args.repeats),
        },
        "environment": {
            "gpu": _gpu_info(),
            "drjit_version": dr.__version__,
        },
        "cases": cases,
        "gates": gates,
        "notes": [
            "The baseline gate compares only identical workload keys, so grid size, sample count, solver, order, and backend changes do not produce false timing regressions.",
            "Use --strict-gates with --baseline-json in automation; without a baseline the benchmark reports timings but does not fail.",
            "Deterministic Munich performance gates are intentionally left for a later workflow.",
        ],
    }


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = run_benchmark(args)
    text = json.dumps(_jsonable(result), indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    if args.json or args.output is None:
        print(text)
    gates = result.get("gates", {})
    return 1 if bool(gates.get("strict")) and not bool(gates.get("passed")) else 0


if __name__ == "__main__":
    raise SystemExit(main())
