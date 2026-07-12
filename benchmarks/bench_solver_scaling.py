from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT))

from benchmarks.harness import (  # noqa: E402
    benchmark_operation,
    tensor_bytes,
    versioned_report,
    write_report,
)
from tests.support.native_ext import inject_native_paths  # noqa: E402

inject_native_paths()

from tests.support.scenes import same_side_wall_reflection_scene  # noqa: E402
from witwin.channel_native import ReceiverPoint, Scene, Transmitter  # noqa: E402
from witwin.channel_native.core.memory_budget import (  # noqa: E402
    estimate_monte_carlo_memory,
)


def _ints(value: str) -> tuple[int, ...]:
    return tuple(dict.fromkeys(int(item) for item in value.split(",") if item))


def _expanded_scene(tx_count: int, rx_count: int) -> Scene:
    base = same_side_wall_reflection_scene()
    return Scene(
        structures=base.structures,
        transmitters=[
            Transmitter(position=torch.tensor([0.0, -1.0 + 0.05 * i, 0.5]))
            for i in range(tx_count)
        ],
        receivers=[
            ReceiverPoint(position=torch.tensor([0.0, 1.0 + 0.05 * i, 0.5]))
            for i in range(rx_count)
        ],
        frequency=base.frequency,
    )


def _operation(solver: str, scene: Scene, *, depth: int, samples: int):
    components = {"los"} if depth == 0 else {"los", "reflection"}
    if solver == "path":
        from witwin.channel_native.path import Config, solve

        config = Config(max_depth=depth, components=components)
        return lambda: solve(scene, config)
    if solver == "deterministic":
        from witwin.channel_native.deterministic import Config, solve

        config = Config(max_depth=depth, components=components)
        return lambda: solve(scene, config)
    if solver == "basic":
        from witwin.channel_native.montecarlo.basic import Config, solve

        config = Config(samples=samples, max_depth=depth, components=components)
        return lambda: solve(scene, config)
    if solver == "bdpt":
        from witwin.channel_native.montecarlo.bdpt import Config, solve

        config = Config(samples=samples, max_depth=depth, components=components)
        return lambda: solve(scene, config)
    raise ValueError(f"unknown solver: {solver}")


def run(args: argparse.Namespace) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("solver scaling benchmark requires CUDA")
    rows = []
    for solver in args.solvers.split(","):
        for tx_count in _ints(args.tx):
            for rx_count in _ints(args.rx):
                scene = _expanded_scene(tx_count, rx_count)
                for depth in _ints(args.depths):
                    sample_axis = _ints(args.samples) if solver in {"basic", "bdpt"} else (1,)
                    for samples in sample_axis:
                        operation = _operation(
                            solver, scene, depth=depth, samples=samples
                        )
                        result, measurement = benchmark_operation(
                            operation, warmup=args.warmup, repeats=args.repeats
                        )
                        output_bytes = tensor_bytes(result)
                        estimate = (
                            estimate_monte_carlo_memory(
                                samples=samples,
                                transmitters=tx_count,
                                receivers=rx_count,
                                depth=depth,
                            )
                            if solver in {"basic", "bdpt"}
                            else None
                        )
                        rows.append(
                            {
                                "solver": solver,
                                "tx": tx_count,
                                "rx": rx_count,
                                "depth": depth,
                                "samples": samples,
                                "timing": measurement.as_dict(),
                                "output_bytes": int(output_bytes),
                                "estimated_scale_memory": (
                                    estimate.as_dict() if estimate is not None else None
                                ),
                            }
                        )
    return versioned_report(
        benchmark="solver_scaling",
        scenario={
            "scene": "same_side_wall",
            "solvers": args.solvers.split(","),
            "axes": {
                "tx": _ints(args.tx),
                "rx": _ints(args.rx),
                "depth": _ints(args.depths),
                "samples": _ints(args.samples),
            },
        },
        results=rows,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--solvers", default="path,deterministic,basic,bdpt")
    parser.add_argument("--tx", default="1,4")
    parser.add_argument("--rx", default="1,64")
    parser.add_argument("--depths", default="1,3,5")
    parser.add_argument("--samples", default="1000,1000000")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output", type=Path, default=Path("artifacts/solver_scaling.v1.json"))
    args = parser.parse_args()
    report = run(args)
    write_report(report, args.output)
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
