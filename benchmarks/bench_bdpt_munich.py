from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from witwin.channel_native import ReceiverGrid, Scene, Structure, Transmitter
from witwin.channel_native.core.materials import Dielectric, PerfectConductor
from witwin.channel_native.montecarlo.bdpt import Config, solve


def _reduced_scene(grid_size: int) -> Scene:
    wall = Structure(
        vertices=torch.tensor(
            [
                [20.0, -70.0, 0.0],
                [20.0, 90.0, 0.0],
                [20.0, -70.0, 45.0],
                [20.0, 90.0, 45.0],
            ]
        ),
        faces=torch.tensor([[0, 1, 2], [1, 3, 2]]),
        material=Dielectric(eps_r=5.0, sigma_e=0.02),
        name="reduced-munich-wall",
        surface_id=101,
    )
    wedge_a = Structure(
        vertices=torch.tensor(
            [
                [-35.0, -20.0, 0.0],
                [-35.0, -20.0, 35.0],
                [-35.0, 55.0, 0.0],
            ]
        ),
        faces=torch.tensor([[0, 1, 2]]),
        material=PerfectConductor(),
        name="reduced-munich-wedge-a",
        surface_id=102,
    )
    wedge_b = Structure(
        vertices=torch.tensor(
            [
                [-35.0, -20.0, 0.0],
                [-35.0, -20.0, 35.0],
                [40.0, -20.0, 0.0],
            ]
        ),
        faces=torch.tensor([[0, 2, 1]]),
        material=PerfectConductor(),
        name="reduced-munich-wedge-b",
        surface_id=103,
    )
    return Scene(
        structures=[wall, wedge_a, wedge_b],
        transmitters=[Transmitter(position=torch.tensor([8.5, 21.0, 27.0]), power_w=1.0)],
        receivers=[
            ReceiverGrid(
                origin=torch.tensor([-120.0, -120.0, 1.5]),
                x_axis=torch.tensor([1.0, 0.0, 0.0]),
                y_axis=torch.tensor([0.0, 1.0, 0.0]),
                shape=(grid_size, grid_size),
                spacing=(240.0 / max(1, grid_size - 1), 260.0 / max(1, grid_size - 1)),
            )
        ],
        frequency=2.4e9,
    )


def run_benchmark(
    *,
    samples: int = 4096,
    grid_size: int = 32,
    warmup_runs: int = 1,
    repeats: int = 3,
    emit_artifacts: bool = True,
    artifact_dir: str | Path = "artifacts/bdpt_munich",
) -> dict[str, float | bool | int | str]:
    if not torch.cuda.is_available():
        raise RuntimeError("bench_bdpt_munich requires CUDA")

    scene = _reduced_scene(grid_size)
    config = Config(
        samples=samples,
        seed=11,
        components={"los", "reflection", "diffraction"},
    )
    for _ in range(max(0, int(warmup_runs))):
        solve(scene, config)
        torch.cuda.synchronize()

    timings: list[float] = []
    result = None
    for _ in range(max(1, int(repeats))):
        torch.cuda.synchronize()
        start = time.perf_counter()
        result = solve(scene, config)
        torch.cuda.synchronize()
        timings.append(time.perf_counter() - start)
    assert result is not None
    sorted_timings = sorted(timings)
    median_seconds = sorted_timings[len(sorted_timings) // 2]
    p95_seconds = sorted_timings[min(len(sorted_timings) - 1, int(0.95 * (len(sorted_timings) - 1)))]
    component_power = {name: float(value.detach().cpu().item()) for name, value in result.component_power.items()}
    native_total_sum = sum(component_power.values())
    enabled_components = ("los", "reflection", "diffraction")
    component_nonzero_score = {
        name: 1.0 if component_power.get(name, 0.0) > 0.0 else 0.0
        for name in enabled_components
    }
    all_zero_component_map = any(component_nonzero_score[name] == 0.0 for name in enabled_components)
    empty_components = [name for name in enabled_components if component_power.get(name, 0.0) <= 0.0]
    if empty_components:
        raise RuntimeError(
            "reduced Munich BDPT produced empty components: "
            f"{empty_components}; component_power={component_power}"
        )
    if emit_artifacts:
        path = Path(artifact_dir)
        path.mkdir(parents=True, exist_ok=True)
        (path / "metadata.json").write_text(json.dumps(result.metadata, sort_keys=True, indent=2), encoding="utf-8")
        (path / "component_power.json").write_text(
            json.dumps(component_power, sort_keys=True, indent=2),
            encoding="utf-8",
        )
    return {
        "samples": samples,
        "grid_size": grid_size,
        "native_total_sum": native_total_sum,
        "native_seconds": median_seconds,
        "native_p95_seconds": p95_seconds,
        "native_min_seconds": min(timings),
        "native_max_seconds": max(timings),
        "warmup_runs": int(warmup_runs),
        "repeats": max(1, int(repeats)),
        "component_nonzero_min": min(component_nonzero_score.values()),
        "all_zero_component_map": all_zero_component_map,
        "artifact_dir": str(artifact_dir),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=4096)
    parser.add_argument("--grid-size", type=int, default=32)
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--artifact-dir", default="artifacts/bdpt_munich")
    parser.add_argument("--no-artifacts", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    result = run_benchmark(
        samples=args.samples,
        grid_size=args.grid_size,
        warmup_runs=args.warmup_runs,
        repeats=args.repeats,
        emit_artifacts=not args.no_artifacts,
        artifact_dir=args.artifact_dir,
    )
    if args.json:
        print(json.dumps(result, sort_keys=True))
    else:
        print(result)


if __name__ == "__main__":
    main()
