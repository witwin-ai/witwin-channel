"""Benchmark witwin MC-basic Munich radio maps against Sionna RT 2.0.1."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any


CHANNEL_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SIONNA_SOURCE_ROOT = CHANNEL_ROOT / "reference" / "sionna-rt-reference-2.0.1" / "src"
DEFAULT_MUNICH_XML = (
    DEFAULT_SIONNA_SOURCE_ROOT / "sionna" / "rt" / "scenes" / "munich" / "munich.xml"
)
DEFAULT_OUTPUT_JSON = (
    CHANNEL_ROOT / "docs" / "dev" / "optimization" / "mc_basic_munich_vs_sionna.json"
)

DEFAULT_TX_POS = (8.5, 21.0, 27.0)
DEFAULT_BOUNDS = ((-120.0, 120.0), (-120.0, 140.0))
DEFAULT_PLANE_Z = 1.5


def parse_seeds(text: str) -> tuple[int, ...]:
    seeds: list[int] = []
    for token in str(text).split(","):
        stripped = token.strip()
        if not stripped:
            raise ValueError("empty seed token in --seeds")
        seeds.append(int(stripped))
    if not seeds:
        raise ValueError("--seeds must contain at least one integer")
    return tuple(seeds)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sionna-source-root", type=Path, default=DEFAULT_SIONNA_SOURCE_ROOT)
    parser.add_argument("--munich-xml", type=Path, default=DEFAULT_MUNICH_XML)
    parser.add_argument("--grid-size", type=int, default=256)
    parser.add_argument("--samples-per-tx", type=int, default=1_000_000)
    parser.add_argument("--max-depth", type=int, default=5)
    parser.add_argument("--frequency-hz", type=float, default=2.4e9)
    parser.add_argument("--tx-pos", type=float, nargs=3, default=DEFAULT_TX_POS)
    parser.add_argument("--plane-z", type=float, default=DEFAULT_PLANE_Z)
    parser.add_argument("--bounds", type=float, nargs=4, default=(-120.0, 120.0, -120.0, 140.0))
    parser.add_argument("--seeds", type=str, default="11,17,23")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--shadow-boundary-mode", choices=("none", "utd_power_smoothing"), default="none")
    parser.add_argument(
        "--witwin-accumulation-backend",
        choices=("auto", "native_monte_carlo", "rayd_reflection_accumulation"),
        default="rayd_reflection_accumulation",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--json", action="store_true", default=False)
    return parser


def _jsonable(value: Any) -> Any:
    try:
        import numpy as np
    except Exception:
        np = None
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if np is not None and isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _bounds_from_args(args: argparse.Namespace) -> tuple[tuple[float, float], tuple[float, float]]:
    xmin, xmax, ymin, ymax = (float(v) for v in args.bounds)
    return ((xmin, xmax), (ymin, ymax))


def _gpu_info() -> dict[str, Any]:
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,memory.used,driver_version",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=3.0,
        ).strip()
    except Exception as exc:
        return {"available": False, "error": str(exc)}
    fields = [field.strip() for field in output.split(",", 3)]
    if len(fields) != 4:
        return {"available": True, "raw": output}
    return {
        "available": True,
        "name": fields[0],
        "memory_total_mib": int(fields[1]),
        "memory_used_mib": int(fields[2]),
        "driver_version": fields[3],
    }


def _ensure_import_paths(*, sionna_source_root: Path) -> None:
    for path in (CHANNEL_ROOT, Path(sionna_source_root).resolve()):
        text = str(path)
        if text not in sys.path:
            sys.path.insert(0, text)


def _sync_witwin(result) -> None:
    import drjit as dr

    dr.eval(result.path_gain)
    dr.sync_thread()


def _sync_sionna(result) -> None:
    import drjit as dr

    dr.eval(result.path_gain)
    dr.sync_thread()


def _witwin_edge_policy():
    from witwin.channel.core.scene import EdgePolicy

    return EdgePolicy(
        edge_selection_mode="all_edges",
        edge_diffraction=True,
        boundary_edge_policy="half_plane",
    )


def _result_stats(value) -> dict[str, Any]:
    import numpy as np

    array = np.asarray(value, dtype=np.float64)
    flat = array.reshape(-1)
    if flat.size == 0:
        return {"shape": list(array.shape), "finite": True, "sum": 0.0, "nonzero": 0}
    return {
        "shape": list(array.shape),
        "finite": bool(np.isfinite(flat).all()),
        "sum": float(np.nansum(flat, dtype=np.float64)),
        "nonzero": int(np.count_nonzero(flat > 0.0)),
        "min": float(np.nanmin(flat)),
        "max": float(np.nanmax(flat)),
    }


def _timed(label: str, operation, sync_result, *, warmup: int, repeats: int) -> dict[str, Any]:
    import drjit as dr
    import numpy as np

    dr.sync_thread()
    if hasattr(dr, "flush_malloc_cache"):
        dr.flush_malloc_cache()
    for _ in range(max(0, int(warmup))):
        sync_result(operation())

    samples_ms: list[float] = []
    result = None
    for _ in range(max(1, int(repeats))):
        start = time.perf_counter()
        result = operation()
        sync_result(result)
        samples_ms.append((time.perf_counter() - start) * 1000.0)

    return {
        "label": label,
        "samples_ms": samples_ms,
        "median_ms": float(np.median(samples_ms)),
        "mean_ms": float(np.mean(samples_ms)),
        "min_ms": float(np.min(samples_ms)),
        "max_ms": float(np.max(samples_ms)),
        "result": result,
    }


def _build_witwin_context(
    *,
    munich_xml: Path,
    sionna_source_root: Path,
    frequency_hz: float,
    tx_pos: tuple[float, float, float],
    bounds: tuple[tuple[float, float], tuple[float, float]],
    plane_z: float,
    grid_size: int,
    samples_per_tx: int,
    max_depth: int,
    seed: int,
    shadow_boundary_mode: str,
    accumulation_backend: str,
):
    from witwin.channel.core.scene import ReceiverGrid, Scene, Transmitter
    from witwin.channel.montecarlo import Config, IntegratorOptions, Tuning

    start = time.perf_counter()
    scene = Scene.load_mitsuba(
        munich_xml,
        device="cuda",
        merge_shapes=True,
        frequency=float(frequency_hz),
        source_root=sionna_source_root,
    )
    scene.add(Transmitter("tx", tx_pos))
    scene.add(
        ReceiverGrid(
            "rm",
            axis="z",
            position=float(plane_z),
            bounds=bounds,
            grid_shape=(int(grid_size), int(grid_size)),
        )
    )
    config = Config(
        num_samples=int(samples_per_tx),
        max_bounces=int(max_depth),
        max_diffraction_order=1,
        edge_policy=_witwin_edge_policy(),
        tuning=Tuning(
            enable_rd_diffraction=True,
            shadow_boundary_mode=str(shadow_boundary_mode),
            shadow_boundary_backend="auto",
            shadow_boundary_max_candidate_factor=128.0,
        ),
        integrator_options=IntegratorOptions(
            integrator="basic",
            samples_per_tx=int(samples_per_tx),
            accumulation_backend=str(accumulation_backend),
            seed=int(seed),
        ),
    )
    info = {
        "load_seconds": time.perf_counter() - start,
        "structures": len(scene.structures),
        "triangles": None if scene.tri_data is None else int(scene.tri_data["n_triangles"]),
        "diffraction_edges": scene.diffraction_edge_count(edge_policy=config.edge_policy),
        "edge_policy": {
            "edge_selection_mode": str(config.edge_policy.edge_selection_mode),
            "edge_diffraction": bool(config.edge_policy.edge_diffraction),
            "boundary_edge_policy": str(config.edge_policy.boundary_edge_policy),
        },
        "metadata": scene.metadata.get("mitsuba"),
        "accumulation_backend": str(accumulation_backend),
    }
    return scene, config, info


def _build_sionna_context(
    *,
    munich_xml: Path,
    sionna_source_root: Path,
    frequency_hz: float,
    tx_pos: tuple[float, float, float],
    bounds: tuple[tuple[float, float], tuple[float, float]],
    plane_z: float,
    grid_size: int,
    samples_per_tx: int,
    max_depth: int,
    seed: int,
):
    from witwin.channel.core import Grid
    from witwin.channel.core.scene import ReceiverGrid
    from witwin.channel.core.scene.sionna_adaptor import SionnaAdaptor

    rt = SionnaAdaptor.load_rt(source_root=sionna_source_root, prefer_local=True)
    import mitsuba as mi

    start = time.perf_counter()
    scene = rt.load_scene(str(munich_xml), merge_shapes=True)
    scene.frequency = float(frequency_hz)
    scene.tx_array = rt.PlanarArray(num_rows=1, num_cols=1, pattern="iso", polarization="V")
    scene.add(rt.Transmitter("tx", position=mi.Point3f(*tx_pos), power_dbm=0.0))

    grid = ReceiverGrid(
        "rm",
        axis="z",
        position=float(plane_z),
        bounds=bounds,
        grid_shape=(int(grid_size), int(grid_size)),
    )
    resolved = Grid.from_spec(grid)
    solver = rt.RadioMapSolver()
    kwargs = {
        "center": mi.Point3f(*resolved.center),
        "orientation": mi.Point3f(0.0, 0.0, 0.0),
        "size": mi.Point2f(*resolved.size),
        "cell_size": mi.Point2f(*resolved.cell_size),
        "samples_per_tx": int(samples_per_tx),
        "max_depth": int(max_depth),
        "los": True,
        "specular_reflection": True,
        "diffuse_reflection": False,
        "refraction": False,
        "diffraction": True,
        "edge_diffraction": True,
        "diffraction_lit_region": True,
        "seed": int(seed),
    }
    info = {
        "load_seconds": time.perf_counter() - start,
        "loop_mode": solver.loop_mode,
        "center": tuple(float(v) for v in resolved.center),
        "size": tuple(float(v) for v in resolved.size),
        "cell_size": tuple(float(v) for v in resolved.cell_size),
    }
    return scene, solver, kwargs, info


def _summarize_seed(
    *,
    args: argparse.Namespace,
    seed: int,
    bounds: tuple[tuple[float, float], tuple[float, float]],
) -> dict[str, Any]:
    import sionna
    from witwin.channel.montecarlo import solve

    tx_pos = tuple(float(v) for v in args.tx_pos)
    witwin_scene, witwin_config, witwin_scene_info = _build_witwin_context(
        munich_xml=Path(args.munich_xml),
        sionna_source_root=Path(args.sionna_source_root),
        frequency_hz=float(args.frequency_hz),
        tx_pos=tx_pos,
        bounds=bounds,
        plane_z=float(args.plane_z),
        grid_size=int(args.grid_size),
        samples_per_tx=int(args.samples_per_tx),
        max_depth=int(args.max_depth),
        seed=int(seed),
        shadow_boundary_mode=str(args.shadow_boundary_mode),
        accumulation_backend=str(args.witwin_accumulation_backend),
    )
    sionna_scene, sionna_solver, sionna_kwargs, sionna_scene_info = _build_sionna_context(
        munich_xml=Path(args.munich_xml),
        sionna_source_root=Path(args.sionna_source_root),
        frequency_hz=float(args.frequency_hz),
        tx_pos=tx_pos,
        bounds=bounds,
        plane_z=float(args.plane_z),
        grid_size=int(args.grid_size),
        samples_per_tx=int(args.samples_per_tx),
        max_depth=int(args.max_depth),
        seed=int(seed),
    )

    witwin_profile = _timed(
        "witwin_mc_basic",
        lambda: solve(scene=witwin_scene, transmitter="tx", receiver="rm", config=witwin_config),
        _sync_witwin,
        warmup=int(args.warmup),
        repeats=int(args.repeats),
    )
    sionna_profile = _timed(
        "sionna_radio_map_solver",
        lambda: sionna_solver(sionna_scene, **sionna_kwargs),
        _sync_sionna,
        warmup=int(args.warmup),
        repeats=int(args.repeats),
    )
    witwin_result = witwin_profile.pop("result")
    sionna_result = sionna_profile.pop("result")
    return {
        "seed": int(seed),
        "scene_load": {"witwin": witwin_scene_info, "sionna": sionna_scene_info},
        "profiles": {"witwin": witwin_profile, "sionna": sionna_profile},
        "result_stats": {
            "witwin": _result_stats(witwin_result.path_gain),
            "sionna": _result_stats(sionna_result.path_gain),
        },
        "speed": _speed_summary(witwin_profile, sionna_profile),
        "sionna_file": sionna.__file__,
    }


def _speed_summary(witwin_profile: Mapping[str, Any], sionna_profile: Mapping[str, Any]) -> dict[str, float | None]:
    witwin_ms = float(witwin_profile["median_ms"])
    sionna_ms = float(sionna_profile["median_ms"])
    return {
        "witwin_over_sionna_median": None if sionna_ms <= 0.0 else witwin_ms / sionna_ms,
        "sionna_over_witwin_median": None if witwin_ms <= 0.0 else sionna_ms / witwin_ms,
    }


def _aggregate_seed_speeds(seed_results: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    import numpy as np

    ratios = [
        float(result["speed"]["witwin_over_sionna_median"])
        for result in seed_results
        if result["speed"]["witwin_over_sionna_median"] is not None
    ]
    if not ratios:
        return {"witwin_over_sionna_median_by_seed": [], "median_witwin_over_sionna": None}
    return {
        "witwin_over_sionna_median_by_seed": ratios,
        "median_witwin_over_sionna": float(np.median(ratios)),
        "max_witwin_over_sionna": float(np.max(ratios)),
        "min_witwin_over_sionna": float(np.min(ratios)),
    }


def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    _ensure_import_paths(sionna_source_root=Path(args.sionna_source_root))
    import drjit as dr
    import mitsuba as mi

    seeds = parse_seeds(str(args.seeds))
    bounds = _bounds_from_args(args)
    seed_results = [
        _summarize_seed(args=args, seed=seed, bounds=bounds)
        for seed in seeds
    ]
    return {
        "scenario": {
            "scene": "munich",
            "munich_xml": str(Path(args.munich_xml)),
            "grid_size": int(args.grid_size),
            "samples_per_tx": int(args.samples_per_tx),
            "max_depth": int(args.max_depth),
            "frequency_hz": float(args.frequency_hz),
            "tx_pos": tuple(float(v) for v in args.tx_pos),
            "plane_z": float(args.plane_z),
            "bounds": bounds,
            "seeds": seeds,
            "shadow_boundary_mode": str(args.shadow_boundary_mode),
            "witwin_accumulation_backend": str(args.witwin_accumulation_backend),
        },
        "environment": {
            "gpu": _gpu_info(),
            "sionna_source_root": str(Path(args.sionna_source_root)),
            "drjit_version": dr.__version__,
            "mitsuba_version": getattr(mi, "__version__", None),
            "mitsuba_variant": mi.variant(),
        },
        "seed_results": seed_results,
        "aggregate": _aggregate_seed_speeds(seed_results),
        "caveats": [
            "This script measures warmed solver wallclock and result synchronization; scene load is reported separately.",
            "Use Nsight Systems separately for OptiX launch-count gates.",
            "Use a Sionna 2.0.1 dependency-pinned environment before publishing external benchmark claims.",
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    result = run_benchmark(args)
    text = json.dumps(_jsonable(result), indent=2, sort_keys=True)
    if args.output is not None:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(text + "\n", encoding="utf-8")
    if args.json or args.output is None:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
