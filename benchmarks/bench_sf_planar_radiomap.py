# Copyright Xingyu Chen.
# Benchmarks sf planar radiomap.

from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import statistics
import sys
import time
from typing import Any

import numpy as np


_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))

# The repository root must resolve before importing the shared benchmark inputs.
from benchmarks.reference_inputs import require_city_xml  # noqa: E402

TX = (468.0, 106.0, 70.0)
BOUNDS = ((-520.0, 720.0), (-480.0, 470.0))
GRID = (256, 256)
PLANE_Z = 1.5
FREQUENCY = 3.5e9
MAX_DEPTH = 1
COMPONENTS = ("los", "reflection")


def _median(values: list[float]) -> float:
    return float(statistics.median(values)) if values else float("nan")


def _summarize_path_gain(path_gain: Any) -> tuple[list[int], float, int]:
    array = np.asarray(path_gain, dtype=np.float64)
    return list(array.shape), float(array.sum()), int(np.count_nonzero(array))


def _write_json(path: pathlib.Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _require_sionna_source_root(sionna_source_root: pathlib.Path | None) -> pathlib.Path:
    if sionna_source_root is None:
        raise RuntimeError("the sionna backend requires --sionna-source-root (a Sionna RT source tree)")
    root = pathlib.Path(sionna_source_root).expanduser().resolve()
    if not (root / "sionna" / "rt").is_dir():
        raise FileNotFoundError(f"--sionna-source-root has no sionna/rt package: {root}")
    return root


def _sionna_scene(scene_xml: pathlib.Path, sionna_source_root: pathlib.Path) -> Any:
    sys.path.insert(0, str(sionna_source_root))
    from sionna.rt import PlanarArray, Transmitter, load_scene

    scene = load_scene(str(scene_xml))
    scene.frequency = FREQUENCY
    scene.tx_array = PlanarArray(
        num_rows=1,
        num_cols=1,
        vertical_spacing=0.5,
        horizontal_spacing=0.5,
        pattern="iso",
        polarization="V",
    )
    scene.add(Transmitter(name="tx", position=list(TX), orientation=[0.0, 0.0, 0.0]))
    return scene


def run_sionna_planar_benchmark(
    *, samples: int, repeats: int, components: set[str], scene_xml: pathlib.Path,
    sionna_source_root: pathlib.Path | None,
) -> dict[str, Any]:
    root = _require_sionna_source_root(sionna_source_root)
    sys.path.insert(0, str(root))
    import drjit as dr
    from sionna.rt import RadioMapSolver

    scene = _sionna_scene(scene_xml, root)
    solver = RadioMapSolver()
    size = (
        BOUNDS[0][1] - BOUNDS[0][0],
        BOUNDS[1][1] - BOUNDS[1][0],
    )
    cell_size = (size[0] / GRID[0], size[1] / GRID[1])
    center = (
        0.5 * (BOUNDS[0][0] + BOUNDS[0][1]),
        0.5 * (BOUNDS[1][0] + BOUNDS[1][1]),
        PLANE_Z,
    )
    kwargs = {
        "center": center,
        "orientation": (0.0, 0.0, 0.0),
        "size": size,
        "cell_size": cell_size,
        "samples_per_tx": int(samples),
        "max_depth": MAX_DEPTH,
        "los": "los" in components,
        "specular_reflection": "reflection" in components,
        "diffuse_reflection": False,
        "refraction": False,
        "diffraction": False,
        "edge_diffraction": False,
        "seed": 0,
    }

    times_ms: list[float] = []
    sums: list[float] = []
    nonzero: list[int] = []
    shape: list[int] | None = None
    for _ in range(int(repeats)):
        start = time.perf_counter()
        rm = solver(scene, **kwargs)
        dr.eval(rm.path_gain)
        dr.sync_thread()
        path_gain = np.asarray(rm.path_gain, dtype=np.float64)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        current_shape, current_sum, current_nonzero = _summarize_path_gain(path_gain)
        times_ms.append(float(elapsed_ms))
        sums.append(current_sum)
        nonzero.append(current_nonzero)
        shape = current_shape

    return {
        "backend": "sionna",
        "samples": int(samples),
        "times_ms": times_ms,
        "median_ms": _median(times_ms),
        "shape": shape or [],
        "path_gain_sum": sums,
        "nonzero": nonzero,
    }


def _run_sionna_in_child(
    *, samples: int, repeats: int, json_path: pathlib.Path, components: set[str],
    scene_xml: pathlib.Path, sionna_source_root: pathlib.Path | None,
) -> dict[str, Any]:
    root = _require_sionna_source_root(sionna_source_root)
    child_json = json_path.with_name(f"{json_path.stem}.sionna_child{json_path.suffix}")
    if child_json.exists():
        child_json.unlink()
    command = [
        sys.executable,
        str(pathlib.Path(__file__).resolve()),
        "--backend",
        "sionna",
        "--samples",
        str(int(samples)),
        "--repeats",
        str(int(repeats)),
        "--components",
        *sorted(components),
        "--scene-xml",
        str(scene_xml),
        "--sionna-source-root",
        str(root),
        "--json",
        str(child_json),
        "--_sionna-child",
    ]
    completed = subprocess.run(command, cwd=str(_REPO_ROOT), capture_output=True, text=True, check=False)
    if not child_json.exists():
        raise RuntimeError(
            "Sionna child benchmark did not write JSON. "
            f"returncode={completed.returncode}\nstdout={completed.stdout}\nstderr={completed.stderr}"
        )
    payload = json.loads(child_json.read_text(encoding="utf-8"))
    for result in payload.get("results", []):
        if result.get("backend") == "sionna":
            return result
    raise RuntimeError(f"Sionna child benchmark JSON did not contain a Sionna result: {child_json}")


def _native_scene(scene_xml: pathlib.Path) -> Any:
    import torch
    from witwin.channel.importers import load_mitsuba
    from witwin.core import AntennaState, ReceiverGrid, Scene
    from witwin.core.identity import new_antenna_id

    dx = (BOUNDS[0][1] - BOUNDS[0][0]) / GRID[0]
    dy = (BOUNDS[1][1] - BOUNDS[1][0]) / GRID[1]
    grid = ReceiverGrid(
        new_antenna_id(),
        origin=torch.tensor([BOUNDS[0][0] + 0.5 * dx, BOUNDS[1][0] + 0.5 * dy, PLANE_Z]),
        x_axis=torch.tensor([1.0, 0.0, 0.0]),
        y_axis=torch.tensor([0.0, 1.0, 0.0]),
        shape=GRID,
        spacing=(dx, dy),
    )
    imported = load_mitsuba(scene_xml, frequency_hz=FREQUENCY, merge_shapes=True)
    return Scene(
        structures=list(imported.structures),
        endpoints=[AntennaState(new_antenna_id(), "tx", torch.tensor(TX)), grid],
        # The solvers keep a merged Mitsuba import's edge table as imported.
        metadata={"mitsuba": {"merge_shapes": True, "source": str(scene_xml)}},
    )


def run_native_planar_benchmark(
    *, samples: int, repeats: int, components: set[str], scene_xml: pathlib.Path,
) -> dict[str, Any]:
    import torch
    from witwin.channel.radiomap import Stochastic, solve

    if not torch.cuda.is_available():
        raise RuntimeError("native benchmark requires CUDA")

    scene = _native_scene(scene_xml)
    config = Stochastic(
        samples=int(samples),
        max_depth=MAX_DEPTH,
        seed=0,
        components=components,
    )

    times_ms: list[float] = []
    sums: list[float] = []
    nonzero: list[int] = []
    component_power: dict[str, list[float]] = {}
    shape: list[int] | None = None
    peak_allocated_bytes = 0
    for _ in range(int(repeats)):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        result = solve(scene, config, reference_frequency_hz=FREQUENCY)
        end.record()
        torch.cuda.synchronize()
        elapsed_ms = float(start.elapsed_time(end))
        peak_allocated_bytes = max(peak_allocated_bytes, int(torch.cuda.max_memory_allocated()))
        path_gain = result.path_gain.detach()
        torch.cuda.synchronize()
        current_shape, current_sum, current_nonzero = _summarize_path_gain(path_gain.cpu().numpy())
        for key, value in result.component_power.items():
            component_power.setdefault(key, []).append(float(value.detach().cpu().item()))
        times_ms.append(elapsed_ms)
        sums.append(current_sum)
        nonzero.append(current_nonzero)
        shape = current_shape

    return {
        "backend": "native",
        "samples": int(samples),
        "times_ms": times_ms,
        "median_ms": _median(times_ms),
        "shape": shape or [],
        "path_gain_sum": sums,
        "nonzero": nonzero,
        "component_power": component_power,
        "peak_allocated_bytes": int(peak_allocated_bytes),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=100_000_000)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--backend", choices=("sionna", "native", "both"), default="both")
    parser.add_argument(
        "--components",
        nargs="+",
        choices=("los", "reflection"),
        default=list(COMPONENTS),
    )
    parser.add_argument(
        "--scene-xml", type=pathlib.Path, default=None,
        help="San Francisco XML; default: the Sionna layout under $WITWIN_CHANNEL_SCENE_ROOT",
    )
    parser.add_argument(
        "--sionna-source-root", type=pathlib.Path, default=None,
        help="Sionna RT source tree, required by the sionna backend",
    )
    parser.add_argument("--json", type=pathlib.Path, default=pathlib.Path("artifacts/sf_planar_radiomap_benchmark.json"))
    parser.add_argument("--_sionna-child", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    components = set(args.components)
    scene_xml = require_city_xml("san_francisco", args.scene_xml)

    results = []
    if args.backend in {"sionna", "both"}:
        if args._sionna_child:
            results.append(
                run_sionna_planar_benchmark(
                    samples=args.samples,
                    repeats=args.repeats,
                    components=components,
                    scene_xml=scene_xml,
                    sionna_source_root=args.sionna_source_root,
                )
            )
        else:
            results.append(
                _run_sionna_in_child(
                    samples=args.samples,
                    repeats=args.repeats,
                    json_path=args.json,
                    components=components,
                    scene_xml=scene_xml,
                    sionna_source_root=args.sionna_source_root,
                )
            )
    if args.backend in {"native", "both"}:
        results.append(
            run_native_planar_benchmark(
                samples=args.samples,
                repeats=args.repeats,
                components=components,
                scene_xml=scene_xml,
            )
        )
    payload = {
        "benchmark": "sf_planar_radiomap",
        "config": {
            "sionna_source_root": None if args.sionna_source_root is None else str(args.sionna_source_root),
            "scene_xml": str(scene_xml),
            "tx": list(TX),
            "bounds": [list(BOUNDS[0]), list(BOUNDS[1])],
            "grid": list(GRID),
            "plane_z": PLANE_Z,
            "frequency": FREQUENCY,
            "max_depth": MAX_DEPTH,
            "components": sorted(components),
        },
        "results": results,
    }
    _write_json(args.json, payload)
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()