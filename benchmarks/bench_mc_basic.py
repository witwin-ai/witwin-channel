# Copyright Xingyu Chen.
# Benchmarks mc basic.

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time
from typing import Any

import torch

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.meta_path = [
    finder
    for finder in sys.meta_path
    if "_witwin_channel_editable" not in type(finder).__module__
]
sys.path.insert(0, str(_REPO_ROOT.parent / "core-radar-architecture-stage1"))
sys.path.insert(0, str(_REPO_ROOT))

from tests.support.native_ext import inject_native_paths  # noqa: E402

inject_native_paths()

from tests.support.scenes import empty_space_los_scene  # noqa: E402
from witwin.channel.montecarlo.basic import Config, solve  # noqa: E402


def _output_bytes(result: Any) -> int:
    total = result.path_gain.numel() * result.path_gain.element_size()
    for tensor in result.component_power.values():
        total += tensor.numel() * tensor.element_size()
    return int(total)


def run_benchmark(*, scene_name: str, samples: int) -> dict[str, Any]:
    if scene_name != "small":
        raise ValueError("only the 'small' benchmark scene is currently maintained")
    if not torch.cuda.is_available():
        raise RuntimeError("MC basic benchmark requires CUDA")

    scene = empty_space_los_scene()
    config = Config(samples=samples, seed=0)
    torch.cuda.synchronize()
    started = time.perf_counter()
    result = solve(scene, config, reference_frequency_hz=3.5e9)
    torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    kernel = result.metadata["kernel"]
    return {
        "benchmark": "mc_basic",
        "scene": scene_name,
        "samples": samples,
        "wall_time_ms": elapsed_ms,
        "launch_count": kernel["launch_count"],
        "intermediate_bytes": kernel["intermediate_bytes"],
        "output_bytes": _output_bytes(result),
        "rayd_native": kernel["rayd_native"],
        "accumulation_strategy": kernel["accumulation_strategy"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", default="small")
    parser.add_argument("--samples", type=int, default=4096)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    payload = run_benchmark(scene_name=args.scene, samples=args.samples)
    if args.json:
        print(json.dumps(payload, sort_keys=True))
    else:
        for key, value in payload.items():
            print(f"{key}: {value}")


if __name__ == "__main__":
    main()