# Copyright Xingyu Chen.
# Benchmarks bdpt basic.

from __future__ import annotations

import argparse
import json
import time

import torch

from witwin.channel.montecarlo import basic
from witwin.channel.montecarlo import bdpt
from witwin.core import AntennaState, ReceiverGrid, Scene
from witwin.core.identity import new_antenna_id


REFERENCE_FREQUENCY_HZ = 3.0e9


def _grid_scene(grid_size: int) -> Scene:
    return Scene(
        structures=[],
        endpoints=[
            AntennaState(
                new_antenna_id(),
                "tx",
                torch.tensor([0.0, 0.0, 1.0]),
                power_w=1.0,
            ),
            ReceiverGrid(
                new_antenna_id(),
                origin=torch.tensor([5.0, -1.0, 0.0]),
                x_axis=torch.tensor([0.0, 1.0, 0.0]),
                y_axis=torch.tensor([0.0, 0.0, 1.0]),
                shape=(grid_size, grid_size),
                spacing=(2.0 / max(1, grid_size - 1), 2.0 / max(1, grid_size - 1)),
            )
        ],
    )


def _elapsed_seconds(fn) -> float:
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start = time.perf_counter()
    fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.perf_counter() - start


def run_benchmark(*, samples: int = 4096, grid_size: int = 32) -> dict[str, float | int | str]:
    if not torch.cuda.is_available():
        raise RuntimeError("bench_bdpt_basic requires CUDA")

    scene = _grid_scene(grid_size)
    basic_config = basic.Config(samples=samples, seed=11, components={"los"})
    bdpt_config = bdpt.Config(samples=samples, seed=11, components={"los"})
    basic.solve(
        scene,
        basic_config,
        reference_frequency_hz=REFERENCE_FREQUENCY_HZ,
    )
    bdpt_result = bdpt.solve(
        scene,
        bdpt_config,
        reference_frequency_hz=REFERENCE_FREQUENCY_HZ,
    )

    basic_seconds = _elapsed_seconds(
        lambda: basic.solve(
            scene,
            basic_config,
            reference_frequency_hz=REFERENCE_FREQUENCY_HZ,
        )
    )
    bdpt_seconds = _elapsed_seconds(
        lambda: bdpt.solve(
            scene,
            bdpt_config,
            reference_frequency_hz=REFERENCE_FREQUENCY_HZ,
        )
    )
    return {
        "bdpt_seconds": bdpt_seconds,
        "mc_basic_seconds": basic_seconds,
        "launch_count": int(bdpt_result.metadata["launch_count"]),
        "accumulation_strategy": str(bdpt_result.metadata["accumulation_strategy"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=4096)
    parser.add_argument("--grid-size", type=int, default=32)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    result = run_benchmark(samples=args.samples, grid_size=args.grid_size)
    if args.json:
        print(json.dumps(result, sort_keys=True))
    else:
        print(result)


if __name__ == "__main__":
    main()