"""Benchmark deterministic reflection F-weight transition overhead."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys
import time

import drjit as dr
import numpy as np


def _repo_root() -> Path:
    path = Path.cwd().resolve()
    if (path / "witwin").exists():
        return path
    return next(parent for parent in path.parents if (parent / "witwin").exists())


ROOT = _repo_root()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from examples.deterministic_radiomap_three_cubes import ThreeCubeExperiment  # noqa: E402


def _config_for(base_config, mode: str, *, boundary_radius_wavelengths: float):
    tuning = replace(
        base_config.tuning,
        reflection_transition_mode=mode,
        reflection_f_weight_boundary_radius_wavelengths=float(boundary_radius_wavelengths),
        reflection_f_weight_max_edges_per_slot=1,
    )
    return replace(base_config, tuning=tuning)


def _run_once(experiment: ThreeCubeExperiment, mode: str, *, boundary_radius_wavelengths: float):
    result = experiment._solve(
        config=_config_for(
            experiment.forward_config,
            mode,
            boundary_radius_wavelengths=boundary_radius_wavelengths,
        )
    ).squeeze_tx(0)
    path_gain_sum = float(np.asarray(result.path_gain, dtype=np.float64).sum())
    dr.sync_thread()
    return path_gain_sum, result.metadata.get("runtime_backends", {}).get("reflection_transition", {})


def _timed(
    experiment: ThreeCubeExperiment,
    mode: str,
    repeats: int,
    *,
    boundary_radius_wavelengths: float,
) -> dict[str, object]:
    _run_once(experiment, mode, boundary_radius_wavelengths=boundary_radius_wavelengths)
    durations = []
    path_gain_sum = 0.0
    metadata = {}
    for _ in range(int(repeats)):
        start = time.perf_counter()
        path_gain_sum, metadata = _run_once(
            experiment,
            mode,
            boundary_radius_wavelengths=boundary_radius_wavelengths,
        )
        durations.append(time.perf_counter() - start)
    return {
        "mode": mode,
        "seconds": durations,
        "best_seconds": min(durations),
        "path_gain_sum": path_gain_sum,
        "reflection_transition": metadata,
    }


def run_benchmark(args) -> dict[str, object]:
    experiment = ThreeCubeExperiment(
        grid_shape=(int(args.grid_size), int(args.grid_size)),
        forward_num_samples=int(args.num_samples),
        gradient_num_samples=int(args.num_samples),
        max_bounces=int(args.max_bounces),
        max_diffraction_order=0,
        shadow_boundary_correction=False,
        seed=7,
    )
    hard = _timed(
        experiment,
        "hard",
        int(args.repeats),
        boundary_radius_wavelengths=float(args.boundary_radius_wavelengths),
    )
    native = _timed(
        experiment,
        "f_weight_native",
        int(args.repeats),
        boundary_radius_wavelengths=float(args.boundary_radius_wavelengths),
    )
    hard_best = float(hard["best_seconds"])
    native_best = float(native["best_seconds"])
    overhead_ratio = native_best / hard_best if hard_best > 0.0 else float("inf")
    native_backend = str(native.get("reflection_transition", {}).get("resolved_backend", ""))
    native_backend_ok = native_backend == "native_cuda_f_weight"
    return {
        "workload": {
            "scene": "deterministic_three_cubes",
            "grid_size": int(args.grid_size),
            "num_samples": int(args.num_samples),
            "max_bounces": int(args.max_bounces),
            "max_diffraction_order": 0,
            "boundary_radius_wavelengths": float(args.boundary_radius_wavelengths),
        },
        "gate": {
            "max_overhead_ratio": float(args.max_overhead_ratio),
            "native_backend_ok": native_backend_ok,
            "passed": overhead_ratio <= float(args.max_overhead_ratio) and native_backend_ok,
        },
        "results": {
            "hard": hard,
            "f_weight_native": native,
            "overhead_ratio": overhead_ratio,
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid-size", type=int, default=12)
    parser.add_argument("--num-samples", type=int, default=32)
    parser.add_argument("--max-bounces", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--boundary-radius-wavelengths", type=float, default=0.01)
    parser.add_argument("--max-overhead-ratio", type=float, default=1.50)
    parser.add_argument("--strict-gate", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    metrics = run_benchmark(args)
    if args.json:
        print(json.dumps(metrics, indent=2, sort_keys=True))
    else:
        ratio = metrics["results"]["overhead_ratio"]
        print(
            "reflection f-weight native overhead: "
            f"{ratio:.3f}x (gate <= {args.max_overhead_ratio:.3f}x)"
        )
    if args.strict_gate and not metrics["gate"]["passed"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
