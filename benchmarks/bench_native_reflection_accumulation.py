from __future__ import annotations

import argparse
import json
import math
import pathlib
import statistics
import sys
from typing import Any


_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
_SIONNA_SOURCE_ROOT = pathlib.Path(
    "E:/Code/witwin-platform/channel/reference/sionna-rt-reference-2.0.1/src"
)
_SIONNA_XML = _SIONNA_SOURCE_ROOT / "sionna/rt/scenes/san_francisco/san_francisco.xml"

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


def _native_imports() -> dict[str, Any]:
    sys.path.insert(0, str(_REPO_ROOT / "src"))
    sys.path.insert(0, str(_REPO_ROOT))
    import torch
    from witwin.channel import Transmitter
    from witwin.core import ReceiverGrid, Scene
    from witwin.channel.kernels.montecarlo import (
        mc_component_map_buffer,
        mc_finalize_component_maps,
        mc_reflection_launch_inputs,
        mc_store_scaled_component_map,
    )
    from witwin.channel.core.material_runtime import face_material_tensors
    from witwin.channel.montecarlo.basic.backend import _LIGHT_SPEED_M_PER_S, transmitter_positions
    from witwin.channel.montecarlo.basic.raydn_components import _sample_directions, grid_spec

    return {
        "torch": torch,
        "ReceiverGrid": ReceiverGrid,
        "Scene": Scene,
        "Transmitter": Transmitter,
        "mc_component_map_buffer": mc_component_map_buffer,
        "mc_finalize_component_maps": mc_finalize_component_maps,
        "mc_reflection_launch_inputs": mc_reflection_launch_inputs,
        "mc_store_scaled_component_map": mc_store_scaled_component_map,
        "face_material_tensors": face_material_tensors,
        "_LIGHT_SPEED_M_PER_S": _LIGHT_SPEED_M_PER_S,
        "transmitter_positions": transmitter_positions,
        "_sample_directions": _sample_directions,
        "grid_spec": grid_spec,
    }


def _build_scene(imports: dict[str, Any]) -> Any:
    torch = imports["torch"]
    ReceiverGrid = imports["ReceiverGrid"]
    Scene = imports["Scene"]
    Transmitter = imports["Transmitter"]
    dx = (BOUNDS[0][1] - BOUNDS[0][0]) / GRID[0]
    dy = (BOUNDS[1][1] - BOUNDS[1][0]) / GRID[1]
    grid = ReceiverGrid(
        origin=torch.tensor([BOUNDS[0][0] + 0.5 * dx, BOUNDS[1][0] + 0.5 * dy, PLANE_Z]),
        x_axis=torch.tensor([1.0, 0.0, 0.0]),
        y_axis=torch.tensor([0.0, 1.0, 0.0]),
        shape=GRID,
        spacing=(dx, dy),
    )
    scene = Scene.load_mitsuba(str(_SIONNA_XML), source_root=_SIONNA_SOURCE_ROOT)
    return type(scene)(
        structures=scene.structures,
        transmitters=[Transmitter(position=torch.tensor(TX))],
        receivers=[grid],
        frequency=FREQUENCY,
        metadata=scene.metadata,
    ), grid


def _time_event(torch: Any, func: Any) -> tuple[float, Any]:
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    value = func()
    end.record()
    torch.cuda.synchronize()
    return float(start.elapsed_time(end)), value


def _run_once(imports: dict[str, Any], *, samples: int, max_depth: int, strategy: str) -> dict[str, Any]:
    torch = imports["torch"]
    mc_component_map_buffer = imports["mc_component_map_buffer"]
    mc_finalize_component_maps = imports["mc_finalize_component_maps"]
    mc_reflection_launch_inputs = imports["mc_reflection_launch_inputs"]
    mc_store_scaled_component_map = imports["mc_store_scaled_component_map"]
    face_material_tensors = imports["face_material_tensors"]
    transmitter_positions = imports["transmitter_positions"]
    _LIGHT_SPEED_M_PER_S = imports["_LIGHT_SPEED_M_PER_S"]
    _sample_directions = imports["_sample_directions"]
    grid_spec = imports["grid_spec"]

    scene, grid = _build_scene(imports)
    compiled = scene.compile()
    if not compiled.raydn.available:
        raise RuntimeError("RayDN native scene is unavailable")

    spec = grid_spec(grid)
    handle = compiled.raydn.require_handle()
    device = torch.device("cuda")
    material_tensors = face_material_tensors(compiled, device=device)
    material_eta_r, material_sigma, material_mu_r, material_gain, material_valid = material_tensors
    tx_pos, tx_power = transmitter_positions(scene, device=device)
    tx_index = 0
    wavelength = _LIGHT_SPEED_M_PER_S / float(scene.frequency)
    solid_angle_per_ray = float(4.0 * math.pi / max(1, int(samples)))
    dim0 = GRID[1]
    dim1 = GRID[0]
    zero = torch.zeros((1, dim0, dim1), device=device, dtype=torch.float32)
    strategy_id = {
        "auto": 0,
        "atomic": 1,
        "staged": 2,
        "compact": 3,
        "streaming_planar": 4,
    }[strategy]

    stage_times: dict[str, float] = {}
    if strategy == "streaming_planar":
        ray_o = tx_pos[tx_index : tx_index + 1].contiguous()
        ray_d = torch.empty((0, 3), device=device, dtype=torch.float32)
        ray_tmax = torch.empty((0,), device=device, dtype=torch.float32)
        active = torch.empty((0,), device=device, dtype=torch.bool)
        tx_batch = ray_o
        tx_pol = torch.tensor([[1.0, 0.0, 0.0]], device=device, dtype=torch.float32)
        stage_times["sample_directions"] = 0.0
        stage_times["launch_inputs"] = 0.0
    else:
        stage_times["sample_directions"], ray_d = _time_event(
            torch,
            lambda: _sample_directions(samples, reference=tx_pos),
        )
        stage_times["launch_inputs"], launch_inputs = _time_event(
            torch,
            lambda: mc_reflection_launch_inputs(tx_pos, tx_index=tx_index, sample_count=samples),
        )
        ray_o = launch_inputs["ray_o"]
        ray_tmax = launch_inputs["ray_tmax"]
        active = launch_inputs["active"]
        tx_batch = ray_o
        tx_pol = launch_inputs["tx_pol"]

    def reflection_forward() -> Any:
        return torch.ops.raydn.reflection_accumulation_forward(
            handle,
            ray_o,
            ray_d,
            ray_tmax,
            active,
            tx_batch,
            tx_pol,
            material_eta_r,
            material_sigma,
            material_mu_r,
            material_gain,
            material_valid,
            int(max_depth),
            int(spec.axis),
            float(spec.position),
            float(spec.coord0_min),
            float(spec.coord0_max),
            float(spec.coord1_min),
            float(spec.coord1_max),
            int(spec.resolution0),
            int(spec.resolution1),
            float(wavelength),
            solid_angle_per_ray,
            False,
            False,
            0,
            1,
            strategy_id,
            262_144,
            64,
            int(samples) if strategy == "streaming_planar" else 0,
            True,
        )

    stage_times["reflection_accumulation_forward"], out = _time_event(torch, reflection_forward)
    maps = mc_component_map_buffer(tx_pos, tx_count=tx_pos.shape[0], dim0=dim0, dim1=dim1)
    stage_times["store"], _ = _time_event(
        torch,
        lambda: mc_store_scaled_component_map(
            maps,
            out[0].contiguous(),
            tx_power,
            tx_index=tx_index,
            scale_index=tx_index,
        ),
    )
    stage_times["finalize"], finalized = _time_event(
        torch,
        lambda: mc_finalize_component_maps(zero, maps, zero),
    )
    path_gain = finalized["path_gain"].detach()
    torch.cuda.synchronize()
    return {
        "stage_times_ms": stage_times,
        "shape": list(path_gain.shape),
        "path_gain_sum": float(path_gain.sum().item()),
        "nonzero": int(torch.count_nonzero(path_gain).item()),
        "reflection_count": int(out[7].detach().cpu().item()),
    }


def run_native_reflection_benchmark(
    *,
    samples: list[int],
    max_depths: list[int],
    strategy: str,
    repeats: int,
) -> dict[str, Any]:
    imports = _native_imports()
    torch = imports["torch"]
    if not torch.cuda.is_available():
        raise RuntimeError("native reflection benchmark requires CUDA")

    rows: list[dict[str, Any]] = []
    for sample_count in samples:
        for max_depth in max_depths:
            times_by_stage: dict[str, list[float]] = {}
            summaries: list[dict[str, Any]] = []
            error: str | None = None
            peak_allocated_bytes = 0
            for _ in range(int(repeats)):
                try:
                    torch.cuda.empty_cache()
                    torch.cuda.reset_peak_memory_stats()
                    result = _run_once(
                        imports,
                        samples=int(sample_count),
                        max_depth=int(max_depth),
                        strategy=strategy,
                    )
                    peak_allocated_bytes = max(peak_allocated_bytes, int(torch.cuda.max_memory_allocated()))
                    for stage, elapsed_ms in result["stage_times_ms"].items():
                        times_by_stage.setdefault(stage, []).append(float(elapsed_ms))
                    summaries.append(result)
                except RuntimeError as exc:
                    if _is_oom(exc) and torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    error = str(exc)
                    break
            row: dict[str, Any] = {
                "backend": "native",
                "samples": int(sample_count),
                "max_depth": int(max_depth),
                "strategy": strategy,
            }
            if error is not None:
                row["error"] = error
            else:
                last = summaries[-1]
                row.update(
                    {
                        "stage_times_ms": times_by_stage,
                        "stage_median_ms": {
                            stage: _median(values) for stage, values in times_by_stage.items()
                        },
                        "total_median_ms": _median(
                            [sum(values) for values in zip(*times_by_stage.values(), strict=True)]
                        ),
                        "shape": last["shape"],
                        "path_gain_sum": [summary["path_gain_sum"] for summary in summaries],
                        "nonzero": [summary["nonzero"] for summary in summaries],
                        "reflection_count": [summary["reflection_count"] for summary in summaries],
                        "peak_allocated_bytes": int(peak_allocated_bytes),
                    }
                )
            rows.append(row)
    return {"benchmark": "native_reflection_accumulation", "results": rows}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, nargs="+", default=[1_000_000, 10_000_000, 100_000_000])
    parser.add_argument("--max-depths", type=int, nargs="+", default=[1, 3, 5])
    parser.add_argument(
        "--strategy",
        choices=("auto", "atomic", "staged", "compact", "streaming_planar"),
        default="auto",
    )
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--json",
        type=pathlib.Path,
        default=pathlib.Path("artifacts/native_reflection_accumulation_benchmark.json"),
    )
    args = parser.parse_args()

    payload = run_native_reflection_benchmark(
        samples=args.samples,
        max_depths=args.max_depths,
        strategy=args.strategy,
        repeats=args.repeats,
    )
    payload["config"] = {
        "source_root": str(_SIONNA_SOURCE_ROOT),
        "scene_xml": str(_SIONNA_XML),
        "tx": list(TX),
        "bounds": [list(BOUNDS[0]), list(BOUNDS[1])],
        "grid": list(GRID),
        "plane_z": PLANE_Z,
        "frequency": FREQUENCY,
    }
    _write_json(args.json, payload)
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
