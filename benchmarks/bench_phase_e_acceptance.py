# Copyright Xingyu Chen.
# Benchmarks performance acceptance behavior.

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, is_dataclass
import json
import math
from pathlib import Path
import sys
import time
from typing import Any

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.meta_path = [
    finder
    for finder in sys.meta_path
    if "_witwin_channel_editable" not in type(finder).__module__
]
sys.path.insert(0, str(REPO_ROOT.parent / "core-radar-architecture-stage1"))
sys.path.insert(0, str(REPO_ROOT))

from benchmarks.harness import (  # noqa: E402
    benchmark_operation,
    environment_record,
    tensor_bytes,
)
from tests.support.native_ext import inject_native_paths  # noqa: E402
from witwin.channel.runtime import (  # noqa: E402
    MemoryBudgetError,
    MemoryEstimate,
    enforce_memory_budget,
    estimate_monte_carlo_memory,
)


inject_native_paths()


SCHEMA_NAME = "witwin.channel.phase-e-performance"
SCHEMA_VERSION = "1.0.0"
SOLVERS = ("path", "deterministic", "basic", "bdpt")
FULL_ENDPOINT_PAIRS = ((1, 1), (8, 1_000), (16, 1_000))
FULL_GRID_SHAPES = ((128, 128), (512, 512))
FULL_DEPTHS = (0, 1, 3, 5)
FULL_MC_SAMPLES = (1_000, 1_000_000, 10_000_000)
PREFLIGHT_MC_SAMPLES = 100_000_000
DEFAULT_BUDGET = (
    REPO_ROOT / "benchmarks/gates/phase_e_performance.sm120.v1.json"
)


@dataclass(frozen=True, slots=True)
class CaseSpec:
    case_id: str
    scenario: str
    solver: str
    depth: int
    samples: int | None
    tx_count: int | None = None
    receiver_count: int | None = None
    grid_shape: tuple[int, int] | None = None

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        if self.grid_shape is not None:
            payload["grid_shape"] = list(self.grid_shape)
        return payload


def profile_matrix(profile: str) -> dict[str, Any]:
    if profile == "reduced":
        return {
            "endpoint_pairs": [[1, 1]],
            "grid_shapes": [[1, 1]],
            "depths": [0],
            "mc_samples": [256],
            "preflight_mc_samples": [PREFLIGHT_MC_SAMPLES],
            "scenarios": ["three_cube"],
        }
    if profile != "full":
        raise ValueError("profile must be 'reduced' or 'full'")
    return {
        "endpoint_pairs": [list(pair) for pair in FULL_ENDPOINT_PAIRS],
        "grid_shapes": [list(shape) for shape in FULL_GRID_SHAPES],
        "depths": list(FULL_DEPTHS),
        "mc_samples": list(FULL_MC_SAMPLES),
        "preflight_mc_samples": [PREFLIGHT_MC_SAMPLES],
        "scenarios": [
            "analytic",
            "three_cube",
            "terrain",
            "munich_full",
            "sf_full",
        ],
    }


def profile_cases(profile: str) -> tuple[CaseSpec, ...]:
    if profile == "reduced":
        return tuple(
            CaseSpec(
                case_id=f"reduced-three-cube-{solver}",
                scenario="three_cube",
                solver=solver,
                depth=0,
                samples=256 if solver in {"basic", "bdpt"} else None,
                tx_count=1,
                receiver_count=1,
            )
            for solver in SOLVERS
        )
    if profile != "full":
        raise ValueError("profile must be 'reduced' or 'full'")

    cases: list[CaseSpec] = []
    scenario_rows = (
        ("analytic", SOLVERS, 0, None),
        ("three_cube", SOLVERS, 3, None),
        ("terrain", ("deterministic", "basic", "bdpt"), 1, (128, 128)),
        ("munich_full", ("basic", "bdpt"), 1, (128, 128)),
        ("sf_full", ("basic", "bdpt"), 1, (512, 512)),
    )
    for scenario, solvers, depth, grid_shape in scenario_rows:
        for solver in solvers:
            cases.append(
                CaseSpec(
                    case_id=f"scene-{scenario}-{solver}",
                    scenario=scenario,
                    solver=solver,
                    depth=depth,
                    samples=1_000 if solver in {"basic", "bdpt"} else None,
                    receiver_count=1 if grid_shape is None else None,
                    grid_shape=grid_shape,
                )
            )
    for solver in ("path", "deterministic"):
        for tx_count, receiver_count in FULL_ENDPOINT_PAIRS[1:]:
            cases.append(
                CaseSpec(
                    case_id=f"pairs-{solver}-{tx_count}x{receiver_count}",
                    scenario="three_cube",
                    solver=solver,
                    depth=1,
                    samples=None,
                    tx_count=tx_count,
                    receiver_count=receiver_count,
                )
            )
    for depth in FULL_DEPTHS:
        cases.append(
            CaseSpec(
                case_id=f"depth-path-d{depth}",
                scenario="analytic",
                solver="path",
                depth=depth,
                samples=None,
                tx_count=1,
                receiver_count=1,
            )
        )
    for solver in SOLVERS:
        cases.append(
            CaseSpec(
                case_id=f"grid-{solver}-512x512",
                scenario="terrain",
                solver=solver,
                depth=0,
                samples=1_000 if solver in {"basic", "bdpt"} else None,
                grid_shape=(512, 512),
            )
        )
    sample_depth_rows = (
        (1_000, 0),
        (1_000_000, 1),
        (10_000_000, 3),
        (1_000, 5),
    )
    for solver in ("basic", "bdpt"):
        for samples, depth in sample_depth_rows:
            cases.append(
                CaseSpec(
                    case_id=f"mc-{solver}-s{samples}-d{depth}",
                    scenario="analytic",
                    solver=solver,
                    depth=depth,
                    samples=samples,
                    tx_count=1,
                    receiver_count=1,
                )
            )
    return tuple(cases)


def _solver_operation(scene: Any, spec: CaseSpec):
    components = {"los"} if spec.depth == 0 else {"los", "reflection"}
    reference_frequency_hz = scene.metadata["reference_frequency_hz"]
    if spec.solver == "path":
        from witwin.channel.path import Config, solve

        config = Config(max_depth=spec.depth, components=components)
        return lambda: solve(
            scene,
            config,
            reference_frequency_hz=reference_frequency_hz,
        )
    if spec.solver == "deterministic":
        from witwin.channel.deterministic import Config, solve

        config = Config(max_depth=spec.depth, components=components)
        return lambda: solve(
            scene,
            config,
            reference_frequency_hz=reference_frequency_hz,
        )
    if spec.solver == "basic":
        from witwin.channel.montecarlo.basic import Config, solve

        config = Config(
            samples=int(spec.samples or 1_000),
            max_depth=spec.depth,
            components=components,
            workspace_limit_bytes=15 << 30,
        )
        return lambda: solve(
            scene,
            config,
            reference_frequency_hz=reference_frequency_hz,
        )
    if spec.solver == "bdpt":
        from witwin.channel.montecarlo.bdpt import Config, solve

        config = Config(
            samples=int(spec.samples or 1_000),
            max_depth=spec.depth,
            components=components,
            workspace_limit_bytes=15 << 30,
        )
        return lambda: solve(
            scene,
            config,
            reference_frequency_hz=reference_frequency_hz,
        )
    raise ValueError(f"unknown solver: {spec.solver}")


def _primary_tensor(result: Any) -> torch.Tensor:
    path_amplitude = getattr(result, "a", None)
    if isinstance(path_amplitude, torch.Tensor):
        return path_amplitude
    coefficient = getattr(result, "coefficient", None)
    if isinstance(coefficient, torch.Tensor):
        return coefficient
    path_gain = getattr(result, "path_gain", None)
    if isinstance(path_gain, torch.Tensor):
        return path_gain
    raise TypeError("solver result exposes neither coefficient nor path_gain")


def _correctness_record(result: Any) -> dict[str, Any]:
    value = _primary_tensor(result)
    finite_mask = (
        torch.isfinite(value.real) & torch.isfinite(value.imag)
        if value.is_complex()
        else torch.isfinite(value)
    )
    finite = bool(finite_mask.all().item())
    checksum = float(value.abs().to(torch.float64).sum().item())
    return {
        "finite": finite,
        "checksum_abs_sum": checksum,
        "observable_numel": int(value.numel()),
    }


def _device_memory() -> dict[str, int]:
    free_bytes, total_bytes = torch.cuda.mem_get_info()
    return {"free_bytes": int(free_bytes), "total_bytes": int(total_bytes)}


def _torch_memory_snapshot() -> dict[str, int]:
    return {
        "allocated_bytes": int(torch.cuda.memory_allocated()),
        "reserved_bytes": int(torch.cuda.memory_reserved()),
    }


def _record_bundle(bundle: Any) -> dict[str, Any]:
    record = bundle.record
    if is_dataclass(record):
        return asdict(record)
    if isinstance(record, dict):
        return dict(record)
    raise TypeError("ScenarioBundle.record must be a dataclass or mapping")


def _load_case_scene(spec: CaseSpec, asset_root: Path | None):
    from benchmarks.phase_e_scenarios import build_scenario

    kwargs: dict[str, Any] = {}
    if spec.scenario in {"munich_full", "sf_full"}:
        kwargs["asset_root"] = asset_root
    if spec.tx_count is not None:
        kwargs["tx_count"] = spec.tx_count
    if spec.receiver_count is not None:
        kwargs["receiver_count"] = spec.receiver_count
    if spec.grid_shape is not None:
        kwargs["grid_shape"] = spec.grid_shape
    return build_scenario(spec.scenario, **kwargs)


def run_case(
    spec: CaseSpec, *, asset_root: Path | None, warmup: int, repeats: int,
) -> dict[str, Any]:
    scene_load_started = time.perf_counter()
    bundle = _load_case_scene(spec, asset_root)
    scene_load_ms = (time.perf_counter() - scene_load_started) * 1_000.0
    scene = bundle.scene
    reference_frequency_hz = scene.metadata["reference_frequency_hz"]
    from witwin.channel.scene import compile as compile_scene

    torch.cuda.synchronize()
    scene_torch_before = _torch_memory_snapshot()
    scene_device_before = _device_memory()
    optix_started = time.perf_counter()
    compile_scene(
        scene,
        reference_frequency_hz=reference_frequency_hz,
    )
    torch.cuda.synchronize()
    optix_scene_build_ms = (time.perf_counter() - optix_started) * 1_000.0
    optix_device_after = _device_memory()
    torch.cuda.reset_peak_memory_stats()
    scene_compile_started = time.perf_counter()
    compile_scene(
        scene,
        reference_frequency_hz=reference_frequency_hz,
    )
    torch.cuda.synchronize()
    scene_compile_ms = (time.perf_counter() - scene_compile_started) * 1_000.0
    scene_torch_after = _torch_memory_snapshot()
    scene_device_after = _device_memory()
    scene_torch_peak = {
        "allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "reserved_bytes": int(torch.cuda.max_memory_reserved()),
    }

    operation = _solver_operation(scene, spec)
    solve_device_before = _device_memory()
    result, measurement = benchmark_operation(
        operation,
        warmup=warmup,
        repeats=repeats,
    )
    solve_device_after = _device_memory()
    solve_torch_after = _torch_memory_snapshot()
    output_bytes = int(tensor_bytes(result))
    correctness = _correctness_record(result)

    return {
        "kind": "benchmark",
        "case": spec.as_dict(),
        "scene": _record_bundle(bundle),
        "timing": {
            "scene_load_ms": float(scene_load_ms),
            "optix_scene_build_ms": float(optix_scene_build_ms),
            "scene_compile_ms": float(scene_compile_ms),
            "pipeline_build_ms": None,
            "pipeline_build_status": (
                "unavailable_without_solver_native_instrumentation"
            ),
            "first": asdict(measurement.first),
            "steady": [asdict(row) for row in measurement.steady],
            "steady_wall_median_ms": measurement.steady_wall_median_ms,
            "steady_wall_p95_ms": measurement.steady_wall_p95_ms,
            "steady_cuda_median_ms": measurement.steady_cuda_median_ms,
            "steady_cuda_p95_ms": measurement.steady_cuda_p95_ms,
        },
        "memory": {
            "output_bytes": output_bytes,
            "torch_allocator": {
                "scene_before": scene_torch_before,
                "scene_after": scene_torch_after,
                "scene_peak": scene_torch_peak,
                "solve_after": solve_torch_after,
                "solve_peak_allocated_bytes": measurement.memory[
                    "peak_allocated_bytes"
                ],
                "solve_peak_reserved_bytes": measurement.memory[
                    "peak_reserved_bytes"
                ],
                "solve_peak_temporary_allocated_bytes": (
                    measurement.memory["peak_temporary_allocated_bytes"]
                ),
                "solve_persistent_growth_excluding_output_bytes": measurement.memory[
                    "persistent_growth_excluding_output_bytes"
                ],
            },
            "device_wide": {
                "scene_before": scene_device_before,
                "scene_after": scene_device_after,
                "solve_before": solve_device_before,
                "solve_after": solve_device_after,
                "peak_bytes": None,
                "peak_status": "unavailable_without_device_memory_sampling",
            },
            "optix_build_bytes": max(
                0,
                scene_device_before["free_bytes"]
                - optix_device_after["free_bytes"],
            ),
            "optix_build_status": (
                "device_wide_delta_for_raydn_optix_scene_build"
            ),
        },
        "correctness": correctness,
    }


def preflight_rows(
    *, budget_bytes: int = 16 << 30, headroom_bytes: int = 1 << 30,
) -> list[dict[str, Any]]:
    rows = []
    for solver in ("basic", "bdpt"):
        for depth in FULL_DEPTHS:
            tx_count = 16
            if solver == "basic":
                estimate = estimate_monte_carlo_memory(
                    samples=PREFLIGHT_MC_SAMPLES,
                    transmitters=tx_count,
                    receivers=1_000,
                    depth=depth,
                )
            else:
                from witwin.channel.montecarlo.bdpt import Config
                from witwin.channel.montecarlo.bdpt import (
                    _estimate_workspace_bytes,
                )

                components = {"los"} if depth == 0 else {"los", "reflection"}
                config = Config(
                    samples=PREFLIGHT_MC_SAMPLES,
                    max_depth=depth,
                    components=components,
                )
                estimate = MemoryEstimate(
                    temporary_bytes=_estimate_workspace_bytes(
                        config,
                        tx_count=tx_count,
                        grid_cells=0,
                        rx_count=1_000,
                    )
                )
            error = None
            rejected = False
            try:
                enforce_memory_budget(
                    estimate,
                    budget_bytes=budget_bytes,
                    headroom_bytes=headroom_bytes,
                    workload=f"Phase E {solver} 100M-sample preflight",
                )
            except MemoryBudgetError as exc:
                rejected = True
                error = str(exc)
            rows.append(
                {
                    "kind": "preflight",
                    "solver": solver,
                    "samples": PREFLIGHT_MC_SAMPLES,
                    "depth": depth,
                    "tx": tx_count,
                    "rx": 1_000,
                    "estimate": estimate.as_dict(),
                    "budget_bytes": budget_bytes,
                    "headroom_bytes": headroom_bytes,
                    "rejected_before_launch": rejected,
                    "error": error,
                }
            )
    return rows


def load_budget(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _device_sm(environment: dict[str, Any]) -> int | None:
    runtime = environment.get("runtime")
    if not isinstance(runtime, dict):
        return None
    device = runtime.get("device")
    if not isinstance(device, dict):
        return None
    sm = device.get("sm")
    return int(sm) if isinstance(sm, int | float) else None


def evaluate_budget(
    measurements: list[dict[str, Any]], environment: dict[str, Any], budget: dict[str, Any],
) -> dict[str, Any]:
    actual_sm = _device_sm(environment)
    target_sm = int(budget["environment"]["sm"])
    if actual_sm != target_sm:
        return {
            "status": "not_gating_environment",
            "eligible": False,
            "passed": None,
            "actual_sm": actual_sm,
            "target_sm": target_sm,
            "checks": [],
        }

    budget_profile = (
        "reduced"
        if measurements
        and all(
            row["case"]["case_id"].startswith("reduced-")
            for row in measurements
        )
        else "full"
    )
    by_solver = budget["profiles"][budget_profile]["solver_budgets"]
    checks: list[dict[str, Any]] = []
    for row in measurements:
        solver = row["case"]["solver"]
        limits = by_solver[solver]
        actuals = {
            "first_wall_ms": row["timing"]["first"]["wall_ms"],
            "steady_wall_median_ms": row["timing"]["steady_wall_median_ms"],
            "steady_wall_p95_ms": row["timing"]["steady_wall_p95_ms"],
            "steady_cuda_median_ms": row["timing"]["steady_cuda_median_ms"],
            "steady_cuda_p95_ms": row["timing"]["steady_cuda_p95_ms"],
            "torch_peak_allocated_bytes": row["memory"]["torch_allocator"][
                "solve_peak_allocated_bytes"
            ],
            "torch_peak_temporary_bytes": row["memory"]["torch_allocator"][
                "solve_peak_temporary_allocated_bytes"
            ],
            "torch_persistent_growth_bytes": row["memory"]["torch_allocator"][
                "solve_persistent_growth_excluding_output_bytes"
            ],
            "optix_build_bytes": row["memory"]["optix_build_bytes"],
            "output_bytes": row["memory"]["output_bytes"],
        }
        for metric, actual in actuals.items():
            limit = limits[metric]
            passed = actual is not None and math.isfinite(actual) and actual <= limit
            checks.append(
                {
                    "case_id": row["case"]["case_id"],
                    "metric": metric,
                    "actual": actual,
                    "limit": limit,
                    "passed": passed,
                }
            )
        checks.append(
            {
                "case_id": row["case"]["case_id"],
                "metric": "correctness_finite",
                "actual": row["correctness"]["finite"],
                "limit": True,
                "passed": row["correctness"]["finite"],
            }
        )
    passed = bool(checks) and all(row["passed"] for row in checks)
    return {
        "status": "passed" if passed else "failed",
        "eligible": True,
        "passed": passed,
        "actual_sm": actual_sm,
        "target_sm": target_sm,
        "checks": checks,
    }


def run_profile(
    *, profile: str, asset_root: Path | None, warmup: int, repeats: int,
    budget_path: Path = DEFAULT_BUDGET,
) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("Phase E benchmark acceptance requires CUDA")
    measurements = [
        run_case(
            spec,
            asset_root=asset_root,
            warmup=warmup,
            repeats=repeats,
        )
        for spec in profile_cases(profile)
    ]
    environment = environment_record()
    budget = load_budget(budget_path)
    preflight = preflight_rows()
    return {
        "schema": {"name": SCHEMA_NAME, "version": SCHEMA_VERSION},
        "profile": profile,
        "matrix": profile_matrix(profile),
        "environment": environment,
        "budget": {
            "path": str(budget_path),
            "name": budget["name"],
            "version": budget["version"],
        },
        "measurements": measurements,
        "preflight": preflight,
        "gate": evaluate_budget(measurements, environment, budget),
    }


def _write_report(report: dict[str, Any], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("reduced", "full"), default="reduced")
    parser.add_argument("--asset-root", type=Path)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--budget", type=Path, default=DEFAULT_BUDGET)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/phase-e-acceptance.v1.json"),
    )
    parser.add_argument("--matrix-only", action="store_true")
    parser.add_argument("--fail-on-gate", action="store_true")
    args = parser.parse_args(argv)
    if args.warmup < 0 or args.repeats <= 0:
        parser.error("warmup must be non-negative and repeats must be positive")
    if args.matrix_only:
        print(json.dumps(profile_matrix(args.profile), indent=2, sort_keys=True))
        return 0
    report = run_profile(
        profile=args.profile,
        asset_root=args.asset_root,
        warmup=args.warmup,
        repeats=args.repeats,
        budget_path=args.budget,
    )
    _write_report(report, args.output)
    print(json.dumps(report, sort_keys=True))
    gate = report["gate"]
    if args.fail_on_gate and gate["eligible"] and not gate["passed"]:
        return 2
    if not all(row["rejected_before_launch"] for row in report["preflight"]):
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())