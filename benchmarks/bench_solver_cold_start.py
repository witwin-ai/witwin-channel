from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]


def _child(solver: str) -> dict[str, Any]:
    phase_start = time.perf_counter()
    import torch

    sys.path.insert(0, str(REPO_ROOT / "src"))
    sys.path.insert(0, str(REPO_ROOT))
    from tests.support.native_ext import inject_native_paths

    inject_native_paths()
    import_started = time.perf_counter()
    import witwin.channel_native  # noqa: F401

    import_ms = (time.perf_counter() - import_started) * 1000.0
    from tests.support.scenes import same_side_wall_reflection_scene

    scene_started = time.perf_counter()
    scene = same_side_wall_reflection_scene()
    scene_load_ms = (time.perf_counter() - scene_started) * 1000.0
    torch.cuda.synchronize()
    optix_free_before, device_total = torch.cuda.mem_get_info()
    optix_started = time.perf_counter()
    scene.raydn_scene()
    torch.cuda.synchronize()
    optix_scene_build_ms = (time.perf_counter() - optix_started) * 1000.0
    optix_free_after, _ = torch.cuda.mem_get_info()
    compile_started = time.perf_counter()
    scene.compile()
    torch.cuda.synchronize()
    scene_compile_ms = (time.perf_counter() - compile_started) * 1000.0

    if solver == "path":
        from witwin.channel_native.path import Config, solve as solve_fn

        config = Config(max_depth=1, components={"los", "reflection"})
    elif solver == "deterministic":
        from witwin.channel_native.deterministic import Config, solve as solve_fn

        config = Config(max_depth=1, components={"los", "reflection"})
    elif solver == "basic":
        from witwin.channel_native.montecarlo.basic import Config, solve as solve_fn

        config = Config(
            samples=256, max_depth=1, components={"los", "reflection"}
        )
    elif solver == "bdpt":
        from witwin.channel_native.montecarlo.bdpt import Config, solve as solve_fn

        config = Config(
            samples=256, max_depth=1, components={"los", "reflection"}
        )
    else:
        raise ValueError(f"unknown solver: {solver}")

    def operation():
        return solve_fn(scene, config)

    torch.cuda.reset_peak_memory_stats()
    solve_started = time.perf_counter()
    operation()
    torch.cuda.synchronize()
    first_solve_ms = (time.perf_counter() - solve_started) * 1000.0
    return {
        "solver": solver,
        "torch_and_bootstrap_ms": (import_started - phase_start) * 1000.0,
        "channel_native_import_ms": import_ms,
        "scene_load_ms": scene_load_ms,
        "optix_scene_build_ms": optix_scene_build_ms,
        "optix_scene_build_device_delta_bytes": max(
            0, int(optix_free_before - optix_free_after)
        ),
        "device_total_bytes": int(device_total),
        "scene_compile_ms": scene_compile_ms,
        "pipeline_build_ms": None,
        "pipeline_build_note": "no public solver prepare boundary; included in first_solve_ms",
        "first_solve_ms": first_solve_ms,
        "peak_gpu_allocated_bytes": int(torch.cuda.max_memory_allocated()),
    }


def _parent(solvers: tuple[str, ...], repeats: int) -> dict[str, Any]:
    sys.path.insert(0, str(REPO_ROOT / "src"))
    sys.path.insert(0, str(REPO_ROOT))
    from benchmarks.harness import versioned_report
    from tests.support.native_ext import inject_native_paths

    inject_native_paths()

    rows = []
    for solver in solvers:
        for repeat in range(repeats):
            command = [sys.executable, str(Path(__file__).resolve()), "--child", solver]
            started = time.perf_counter()
            completed = subprocess.run(
                command,
                cwd=REPO_ROOT,
                capture_output=True,
                text=True,
                check=False,
            )
            process_wall_ms = (time.perf_counter() - started) * 1000.0
            if completed.returncode != 0:
                raise RuntimeError(
                    f"cold-start child failed for {solver}: {completed.stderr.strip()}"
                )
            row = json.loads(completed.stdout)
            row.update({"repeat": repeat, "process_wall_ms": process_wall_ms})
            rows.append(row)
    return versioned_report(
        benchmark="solver_cold_start",
        scenario={"solvers": solvers, "repeats": repeats},
        results=rows,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--child", choices=("path", "deterministic", "basic", "bdpt"))
    parser.add_argument("--solvers", default="path,deterministic,basic,bdpt")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path, default=Path("artifacts/solver_cold_start.v1.json"))
    args = parser.parse_args()
    if args.child:
        print(json.dumps(_child(args.child), sort_keys=True))
        return 0
    report = _parent(tuple(args.solvers.split(",")), args.repeats)
    from benchmarks.harness import write_report

    write_report(report, args.output)
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
