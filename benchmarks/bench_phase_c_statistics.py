from __future__ import annotations

# ruff: noqa: E402 -- source/build paths must be injected before project imports

import argparse
import json
from pathlib import Path
import sys
from typing import Callable

import torch

_ROOT = Path(__file__).resolve().parents[1]
for path in (_ROOT, _ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from tests.support.native_ext import inject_native_paths

inject_native_paths()

from benchmarks.statistical_gate import (
    Observation,
    evaluate_thresholds,
    summarize_observations,
)
from tests.montecarlo.basic import test_basic_scattering as basic_scattering
from tests.montecarlo.bdpt import test_transmission as bdpt_transmission
from tests.support.scenes import wedge_diffraction_scene
from witwin.channel_native.core.materials import PerfectConductor
from witwin.channel_native.montecarlo.basic import Config as BasicConfig
from witwin.channel_native.montecarlo.basic import solve as solve_basic
from witwin.channel_native.montecarlo.bdpt import Config as BDPTConfig
from witwin.channel_native.montecarlo.bdpt import solve as solve_bdpt


_DEFAULT_GATE = _ROOT / "benchmarks" / "gates" / "phase_c_statistics.v1.json"


def _observe(
    seed: int, operation: Callable[[], tuple[torch.Tensor, float]]
) -> Observation:
    try:
        tensor, value = operation()
        finite = torch.isfinite(tensor)
        return Observation(seed, value, int(finite.sum().item()), int(finite.numel()))
    except (
        Exception
    ) as exc:  # gate reports failures instead of losing the remaining seeds
        return Observation(seed, None, 0, 0, f"{type(exc).__name__}: {exc}")


def _run_case(
    name: str, seeds: tuple[int, ...], samples: int
) -> tuple[list[Observation], float | None]:
    if name == "bdpt_wedge_diffraction":
        scene = wedge_diffraction_scene()
        return [
            _observe(
                seed,
                lambda seed=seed: (
                    (
                        result := solve_bdpt(
                            scene,
                            BDPTConfig(
                                samples=samples,
                                seed=seed,
                                components={"diffraction"},
                                receiver_strategy="point_sphere",
                            ),
                        )
                    ).path_gain,
                    float(result.component_power["diffraction"]),
                ),
            )
            for seed in seeds
        ], None
    if name == "mc_basic_rough_scattering":
        scene = basic_scattering._grid_scene(
            [
                basic_scattering._wall(
                    basic_scattering._material(basic_scattering._roughness())
                )
            ]
        )
        reference = basic_scattering._quadrature_reference_unpolarized()
        return [
            _observe(
                seed,
                lambda seed=seed: (
                    (
                        result := solve_basic(
                            scene,
                            BasicConfig(
                                samples=samples, seed=seed, components={"scattering"}
                            ),
                        )
                    ).path_gain,
                    float(result.component_maps["scattering"][0, 0, 0]),
                ),
            )
            for seed in seeds
        ], reference
    if name == "bdpt_mixed_reflection_transmission":
        scene = bdpt_transmission._point_scene(
            [
                bdpt_transmission._wall(
                    bdpt_transmission._lossy(), x=2.5, surface_id=1
                ),
                bdpt_transmission._wall(PerfectConductor(), x=6.0, surface_id=2),
            ]
        )
        return [
            _observe(
                seed,
                lambda seed=seed: (
                    (
                        result := solve_bdpt(
                            scene,
                            BDPTConfig(
                                samples=samples,
                                seed=seed,
                                max_depth=3,
                                components={"reflection", "transmission"},
                            ),
                        )
                    ).path_gain,
                    float(result.component_power["transmission"]),
                ),
            )
            for seed in seeds
        ], None
    raise ValueError(f"unknown statistical gate case: {name}")


def run(*, mode: str, gate_path: Path) -> dict[str, object]:
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    full_seeds = tuple(int(seed) for seed in gate["full_seeds"])
    seeds = full_seeds if mode == "full" else full_seeds[:4]
    cases = {}
    passed = True
    for name, case in gate["cases"].items():
        observations, derived_reference = _run_case(name, seeds, int(case["samples"]))
        reference = case.get("reference", derived_reference)
        summary = summarize_observations(observations, reference=reference)
        checks = evaluate_thresholds(summary, case["thresholds"])
        case_passed = all(checks.values())
        cases[name] = {"summary": summary, "checks": checks, "passed": case_passed}
        passed &= case_passed
    return {
        "schema": {"name": "witwin.channel_native.statistics", "version": "1.0.0"},
        "mode": mode,
        "seeds": list(seeds),
        "cases": cases,
        "passed": passed,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("reduced", "full"), default="full")
    parser.add_argument("--gate", type=Path, default=_DEFAULT_GATE)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    report = run(mode=args.mode, gate_path=args.gate)
    payload = json.dumps(report, indent=2, sort_keys=True, allow_nan=False)
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(payload + "\n", encoding="utf-8")
    print(payload)
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
