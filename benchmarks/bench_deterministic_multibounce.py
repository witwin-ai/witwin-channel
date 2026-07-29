# Copyright Xingyu Chen.
# Benchmarks deterministic multibounce.

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time
from typing import Any

import torch

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))

# Local source and test helpers must resolve from this checkout before importing them.
from tests.deterministic.test_reflection_multibounce import two_wall_multibounce_scene  # noqa: E402
from witwin.channel.deterministic import Config, solve  # noqa: E402


def run_benchmark() -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("deterministic multibounce benchmark requires CUDA")
    scene = two_wall_multibounce_scene()
    torch.cuda.synchronize()
    started = time.perf_counter()
    result = solve(scene, Config(components={"reflection"}, max_depth=2, coherent=True, export_paths=True))
    torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    depth_two = 0 if result.paths is None else int((result.paths.depth == 2).sum().detach().cpu().item())
    return {
        "benchmark": "deterministic_multibounce",
        "scene": "two_wall",
        "wall_time_ms": elapsed_ms,
        "launch_count": result.metadata["kernel"]["launch_count"],
        "path_count": result.metadata["counts"]["path_count"],
        "depth_two_path_count": depth_two,
        "raydn_native": result.metadata["kernel"]["raydn_native"],
        "accumulation_strategy": result.metadata["accumulation_strategy"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    payload = run_benchmark()
    if args.json:
        print(json.dumps(payload, sort_keys=True))
    else:
        for key, value in payload.items():
            print(f"{key}: {value}")


if __name__ == "__main__":
    main()