# Copyright Xingyu Chen.
# Tests benchmark diffraction selector.

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import statistics
import sys
import time

import torch


_REPO_ROOT = pathlib.Path(
    os.environ.get(
        "WITWIN_CHANNEL_BENCHMARK_SOURCE_ROOT",
        pathlib.Path(__file__).resolve().parents[3],
    )
).resolve()
sys.path.insert(0, str(_REPO_ROOT))

from tests.support.scenes import wedge_diffraction_scene  # noqa: E402
from witwin.channel.path import Config, solve  # noqa: E402
from witwin.channel.runtime import build_info  # noqa: E402


def _result_hash(result: object) -> str:
    digest = hashlib.sha256()
    for name in ("valid", "a", "tau", "primitive_id"):
        tensor = getattr(result, name)
        digest.update(tensor.detach().contiguous().cpu().numpy().tobytes())
    return digest.hexdigest()


def _percentile95(values: list[float]) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(0.95 * (len(ordered) - 1)))]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True)
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--solves-per-repeat", type=int, default=1)
    parser.add_argument("--profile-one", action="store_true")
    args = parser.parse_args()
    if args.solves_per_repeat < 1:
        parser.error("--solves-per-repeat must be at least 1")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    native_info = build_info()
    if not native_info["uses_rayd_native"]:
        raise RuntimeError("the native RayD extension is required")

    scene = wedge_diffraction_scene()
    config = Config(components={"diffraction"})

    first_start = time.perf_counter()
    result = solve(scene, config)
    torch.cuda.synchronize()
    first_wall_ms = (time.perf_counter() - first_start) * 1000.0

    for _ in range(max(0, args.warmup_runs)):
        for _ in range(args.solves_per_repeat):
            result = solve(scene, config)
        torch.cuda.synchronize()

    if args.profile_one:
        torch.cuda.cudart().cudaProfilerStart()
        result = solve(scene, config)
        torch.cuda.synchronize()
        torch.cuda.cudart().cudaProfilerStop()

    wall_ms: list[float] = []
    cuda_ms: list[float] = []
    torch.cuda.reset_peak_memory_stats()
    for _ in range(max(1, args.repeats)):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        wall_start = time.perf_counter()
        start.record()
        for _ in range(args.solves_per_repeat):
            result = solve(scene, config)
        end.record()
        torch.cuda.synchronize()
        wall_ms.append(
            (time.perf_counter() - wall_start) * 1000.0 / args.solves_per_repeat
        )
        cuda_ms.append(float(start.elapsed_time(end)) / args.solves_per_repeat)

    print(
        json.dumps(
            {
                "schema_version": 1,
                "label": args.label,
                "build_fingerprint": native_info["build_fingerprint"],
                "channel_commit": native_info["channel_git_sha"],
                "rayd_commit": native_info["rayd_commit"],
                "cuda_architectures": native_info["cuda_architectures"],
                "first_wall_ms": first_wall_ms,
                "solves_per_repeat": args.solves_per_repeat,
                "steady_cuda_ms": cuda_ms,
                "steady_wall_ms": wall_ms,
                "cuda_median_ms": statistics.median(cuda_ms),
                "cuda_p95_ms": _percentile95(cuda_ms),
                "wall_median_ms": statistics.median(wall_ms),
                "wall_p95_ms": _percentile95(wall_ms),
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                "result_hash": _result_hash(result),
                "path_capacity": int(result.valid.numel()),
                "valid_count": int(result.valid.sum().item()),
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()