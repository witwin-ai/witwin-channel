from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time
from typing import Any

import torch

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))
sys.path.insert(0, str(_REPO_ROOT))

from tests.deterministic.test_component_layout import _grid_scene
from witwin.channel_native.deterministic import Config, solve


def _output_bytes(result: Any) -> int:
    tensors = [result.path_gain, result.field]
    tensors.extend(result.component_power.values())
    tensors.extend(result.component_fields.values())
    return int(sum(t.numel() * t.element_size() for t in tensors))


def run_benchmark() -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("deterministic radiomap benchmark requires CUDA")
    scene = _grid_scene()
    torch.cuda.synchronize()
    started = time.perf_counter()
    result = solve(scene, Config(max_depth=0, components={"los"}))
    torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    return {
        "benchmark": "deterministic_radiomap",
        "scene": "grid_los",
        "wall_time_ms": elapsed_ms,
        "launch_count": result.metadata["kernel"]["launch_count"],
        "path_count": result.metadata["counts"]["path_count"],
        "output_bytes": _output_bytes(result),
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
