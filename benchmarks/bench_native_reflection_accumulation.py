# Copyright Xingyu Chen.
# Benchmarks Stochastic reflection-only radiomap accumulation on the San Francisco scene.

from __future__ import annotations

import argparse
import json
import pathlib
import statistics
import sys
from typing import Any


_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))

# The repository root must resolve before importing the shared benchmark inputs.
from benchmarks.reference_inputs import require_city_xml  # noqa: E402

TX = (468.0, 106.0, 70.0)
BOUNDS = ((-520.0, 720.0), (-480.0, 470.0))
GRID = (256, 256)
PLANE_Z = 1.5
FREQUENCY = 3.5e9


def _write_json(path: pathlib.Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _median(values: list[float]) -> float:
    return float(statistics.median(values)) if values else float("nan")


def _is_oom(error: RuntimeError) -> bool:
    text = str(error).lower()
    return "out of memory" in text or "cuda error: out of memory" in text


def _build_scene(scene_xml: pathlib.Path, grid: tuple[int, int]) -> Any:
    import torch
    from witwin.channel.importers import load_mitsuba
    from witwin.core import AntennaState, ReceiverGrid, Scene
    from witwin.core.identity import new_antenna_id

    dx = (BOUNDS[0][1] - BOUNDS[0][0]) / grid[0]
    dy = (BOUNDS[1][1] - BOUNDS[1][0]) / grid[1]
    receiver_grid = ReceiverGrid(
        new_antenna_id(),
        origin=torch.tensor([BOUNDS[0][0] + 0.5 * dx, BOUNDS[1][0] + 0.5 * dy, PLANE_Z]),
        x_axis=torch.tensor([1.0, 0.0, 0.0]),
        y_axis=torch.tensor([0.0, 1.0, 0.0]),
        shape=grid,
        spacing=(dx, dy),
    )
    imported = load_mitsuba(scene_xml, frequency_hz=FREQUENCY, merge_shapes=True)
    return Scene(
        structures=list(imported.structures),
        endpoints=[AntennaState(new_antenna_id(), "tx", torch.tensor(TX)), receiver_grid],
        # The solvers keep a merged Mitsuba import's edge table as imported.
        metadata={"mitsuba": {"merge_shapes": True, "source": str(scene_xml)}},
    )


def _run_once(scene: Any, *, samples: int, max_depth: int) -> dict[str, Any]:
    import torch
    from witwin.channel.radiomap import Stochastic, solve

    config = Stochastic(
        samples=int(samples), max_depth=int(max_depth), seed=0, components={"reflection"},
    )
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    result = solve(scene, config, reference_frequency_hz=FREQUENCY)
    end.record()
    torch.cuda.synchronize()
    path_gain = result.path_gain.detach()
    kernel = dict(result.metadata.get("kernel", {}))
    return {
        "solve_ms": float(start.elapsed_time(end)),
        "shape": list(path_gain.shape),
        "path_gain_sum": float(path_gain.double().sum().item()),
        "nonzero": int(torch.count_nonzero(path_gain).item()),
        "reflection_power": float(result.component_power["reflection"].double().sum().item()),
        "path_count": int(result.metadata.get("path_count", -1)),
        "launch_count": kernel.get("launch_count"),
        "kernel": kernel,
    }


def run_native_reflection_benchmark(
    *, samples: list[int], max_depths: list[int], repeats: int, scene_xml: pathlib.Path,
    grid: tuple[int, int] = GRID,
) -> dict[str, Any]:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("native reflection benchmark requires CUDA")

    scene = _build_scene(scene_xml, grid)
    rows: list[dict[str, Any]] = []
    for sample_count in samples:
        for max_depth in max_depths:
            summaries: list[dict[str, Any]] = []
            error: str | None = None
            peak_allocated_bytes = 0
            for _ in range(int(repeats)):
                try:
                    torch.cuda.empty_cache()
                    torch.cuda.reset_peak_memory_stats()
                    summaries.append(
                        _run_once(scene, samples=int(sample_count), max_depth=int(max_depth))
                    )
                    peak_allocated_bytes = max(
                        peak_allocated_bytes, int(torch.cuda.max_memory_allocated())
                    )
                except RuntimeError as exc:
                    if _is_oom(exc):
                        torch.cuda.empty_cache()
                    error = str(exc)
                    break
            row: dict[str, Any] = {
                "backend": "native",
                "samples": int(sample_count),
                "max_depth": int(max_depth),
            }
            if error is not None:
                row["error"] = error
            else:
                last = summaries[-1]
                times = [summary["solve_ms"] for summary in summaries]
                row.update(
                    {
                        "solve_times_ms": times,
                        "solve_median_ms": _median(times),
                        "shape": last["shape"],
                        "path_gain_sum": [summary["path_gain_sum"] for summary in summaries],
                        "nonzero": [summary["nonzero"] for summary in summaries],
                        "reflection_power": [summary["reflection_power"] for summary in summaries],
                        "path_count": [summary["path_count"] for summary in summaries],
                        "launch_count": last["launch_count"],
                        "kernel": last["kernel"],
                        "peak_allocated_bytes": int(peak_allocated_bytes),
                    }
                )
            rows.append(row)
    return {"benchmark": "native_reflection_accumulation", "results": rows}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Time reflection-only radiomap.Stochastic solves on the San Francisco scene."
    )
    parser.add_argument("--samples", type=int, nargs="+", default=[1_000_000, 10_000_000, 100_000_000])
    parser.add_argument("--max-depths", type=int, nargs="+", default=[1, 3, 5])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--grid", type=int, nargs=2, default=list(GRID), metavar=("ROWS", "COLS"))
    parser.add_argument(
        "--scene-xml", type=pathlib.Path, default=None,
        help="San Francisco XML; default: the Sionna layout under $WITWIN_CHANNEL_SCENE_ROOT",
    )
    parser.add_argument(
        "--json",
        type=pathlib.Path,
        default=pathlib.Path("artifacts/native_reflection_accumulation_benchmark.json"),
    )
    args = parser.parse_args()
    scene_xml = require_city_xml("san_francisco", args.scene_xml)
    grid = (int(args.grid[0]), int(args.grid[1]))

    payload = run_native_reflection_benchmark(
        samples=args.samples,
        max_depths=args.max_depths,
        repeats=args.repeats,
        scene_xml=scene_xml,
        grid=grid,
    )
    payload["config"] = {
        "scene_xml": str(scene_xml),
        "tx": list(TX),
        "bounds": [list(BOUNDS[0]), list(BOUNDS[1])],
        "grid": list(grid),
        "plane_z": PLANE_Z,
        "frequency": FREQUENCY,
        "components": ["reflection"],
    }
    _write_json(args.json, payload)
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
