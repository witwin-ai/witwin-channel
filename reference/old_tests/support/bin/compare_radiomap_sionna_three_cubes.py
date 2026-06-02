"""Benchmark three-cube radio maps: witwin Monte Carlo vs Sionna RT."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean, median
from typing import Any, Callable, Mapping
if os.environ.get("WITWIN_CHANNEL_MAIN_SHOW", "0") != "1":
    os.environ.setdefault("MPLBACKEND", "Agg")

import drjit as dr
import matplotlib.pyplot as plt
import numpy as np
import torch

import witwin as wt

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
tests_dir = REPO_ROOT / "tests"
tests_init = tests_dir / "__init__.py"
existing_tests_module = sys.modules.get("tests")
existing_tests_path = getattr(existing_tests_module, "__file__", "")
if not existing_tests_path or not str(existing_tests_path).startswith(str(REPO_ROOT)):
    tests_spec = importlib.util.spec_from_file_location(
        "tests",
        tests_init,
        submodule_search_locations=[str(tests_dir)],
    )
    if tests_spec is None or tests_spec.loader is None:
        raise RuntimeError(f"Unable to load local tests package from {tests_init}.")
    tests_module = importlib.util.module_from_spec(tests_spec)
    sys.modules["tests"] = tests_module
    tests_spec.loader.exec_module(tests_module)
    sys.modules.pop("tests.main", None)

try:
    from ._benchmark_runtime import benchmark_environment_report
except ImportError:
    THIS_DIR = Path(__file__).resolve().parent
    if str(THIS_DIR) not in sys.path:
        sys.path.insert(0, str(THIS_DIR))
    from _benchmark_runtime import benchmark_environment_report
from tests.main.plot_multipath_components import CUBE1_BASE_CENTER
from tests.main.plot_radiomap_sionna_three_cubes import (
    DEFAULT_BOUNDS,
    DEFAULT_DB_MAX,
    DEFAULT_DB_MIN,
    DEFAULT_PLANE_Z,
    DEFAULT_TX_POS,
    DEFAULT_WITWIN_EDGE_SELECTION_MODE,
    _absolute_db_map,
    _build_comparison_scene,
    _correlation,
    _decorate_axis,
    _extent,
    _prepare_sionna_scene,
    _run_sionna,
    _to_float_grid,
)
from witwin.channel import RadioMapMonitor, Tracer
DEFAULT_GRID_SIZE = 256
DEFAULT_SAMPLES_PER_TX = 250_000
DEFAULT_WARMUP = 1
DEFAULT_REPEATS = 3
DEFAULT_SEED = 7
DEFAULT_OUTPUT_PREFIX = (
    Path(__file__).resolve().parents[2]
    / "output"
    / "radiomap_three_cubes_monte_carlo_vs_sionna"
)


@dataclass(frozen=True)
class TimingSummary:
    samples_ms: tuple[float, ...]
    median_ms: float
    mean_ms: float
    min_ms: float
    max_ms: float
    warmup: int
    repeats: int


@dataclass(frozen=True)
class ComparisonSummary:
    grid_size: int
    bounds: tuple[tuple[float, float], tuple[float, float]]
    plane_z: float
    tx_pos: tuple[float, float, float]
    samples_per_tx: int
    db_min: float
    db_max: float
    warmup: int
    repeats: int
    witwin_backend: str
    witwin_sampling_mode: str
    witwin_reflection_source_paths: int
    witwin_diffraction_total_states: int
    witwin_diffraction_kept_states: int
    total_db_corr: float
    diff_db_corr: float
    total_delta_db_mae: float
    diff_delta_db_mae: float
    witwin_no_diff_timing: TimingSummary
    witwin_with_diff_timing: TimingSummary
    sionna_no_diff_timing: TimingSummary
    sionna_with_diff_timing: TimingSummary
    no_diff_speedup_vs_sionna: float
    with_diff_speedup_vs_sionna: float
    benchmark_isolation_mode: str
    sionna_source: str
    environment: Mapping[str, Any]


def _sync_gpu() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    if hasattr(dr, "sync_thread"):
        dr.sync_thread()


def flush_gpu_caches() -> None:
    _sync_gpu()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    if hasattr(dr, "flush_malloc_cache"):
        dr.flush_malloc_cache()
    _sync_gpu()


def _timing_summary(*, samples_ms: list[float], warmup: int, repeats: int) -> TimingSummary:
    return TimingSummary(
        samples_ms=tuple(float(value) for value in samples_ms),
        median_ms=float(median(samples_ms)),
        mean_ms=float(mean(samples_ms)),
        min_ms=float(min(samples_ms)),
        max_ms=float(max(samples_ms)),
        warmup=int(warmup),
        repeats=int(repeats),
    )


def _timed_repeat(
    fn: Callable[[], Any],
    *,
    sync_result: Callable[[Any], None],
    warmup: int,
    repeats: int,
) -> tuple[Any, TimingSummary]:
    last_value = None
    for _ in range(int(warmup)):
        last_value = fn()
        sync_result(last_value)

    samples_ms: list[float] = []
    for _ in range(int(repeats)):
        _sync_gpu()
        t0 = time.perf_counter()
        last_value = fn()
        sync_result(last_value)
        samples_ms.append((time.perf_counter() - t0) * 1000.0)

    if not samples_ms:
        raise ValueError("repeats must be greater than zero.")
    return last_value, _timing_summary(
        samples_ms=samples_ms,
        warmup=warmup,
        repeats=repeats,
    )


def _sync_witwin_result(result) -> None:
    del result
    _sync_gpu()


def _sync_sionna_result(result) -> None:
    _ = float(result.path_gain[0, 0, 0].numpy())
    _sync_gpu()


def _make_witwin_tracer(*, scene, samples_per_tx: int, max_diffractions: int) -> Tracer:
    return Tracer(
        frequency=1.0e9,
        scene=scene,
        reflection_n_rays=int(samples_per_tx),
        reflection_max_bounces=3,
        reflection_coef=0.8,
        enable_rd_diffraction=bool(int(max_diffractions) > 0),
        max_diffractions=int(max_diffractions),
    )


def _make_witwin_monitor(
    *,
    plane_z: float,
    bounds,
    grid_size: int,
    samples_per_tx: int,
    max_diffractions: int,
    seed: int,
) -> RadioMapMonitor:
    return RadioMapMonitor(
        "radio_map_three_cubes_mc_compare",
        axis="z",
        position=float(plane_z),
        bounds=bounds,
        grid_shape=(int(grid_size), int(grid_size)),
        metric="path_gain",
        combine_mode="incoherent",
        receiver_model="matched_isotropic",
        accumulation_backend="auto",
        ray_mode="3d",
        max_diffractions=int(max_diffractions),
        sampling_mode="monte_carlo",
        samples_per_tx=int(samples_per_tx),
        seed=int(seed),
    )


def _run_witwin_monte_carlo(
    *,
    scene,
    tx_pos,
    plane_z: float,
    bounds,
    grid_size: int,
    samples_per_tx: int,
    max_diffractions: int,
    seed: int,
    warmup: int,
    repeats: int,
):
    flush_gpu_caches()
    tracer = _make_witwin_tracer(
        scene=scene,
        samples_per_tx=samples_per_tx,
        max_diffractions=max_diffractions,
    )
    monitor = _make_witwin_monitor(
        plane_z=plane_z,
        bounds=bounds,
        grid_size=grid_size,
        samples_per_tx=samples_per_tx,
        max_diffractions=max_diffractions,
        seed=seed,
    )

    def _trace_once():
        result = tracer.trace(wt.Point3f(*tx_pos), monitor=monitor, verbose=False)
        return result.monitor(monitor.name) if hasattr(result, "monitor") else result

    payload, timing = _timed_repeat(
        _trace_once,
        sync_result=_sync_witwin_result,
        warmup=warmup,
        repeats=repeats,
    )
    return payload, timing


def _run_sionna_benchmark(
    *,
    scene,
    tx_pos,
    plane_z: float,
    bounds,
    grid_size: int,
    samples_per_tx: int,
    diffraction: bool,
    warmup: int,
    repeats: int,
):
    flush_gpu_caches()
    conversion, rt, sionna_scene = _prepare_sionna_scene(scene=scene, tx_pos=tx_pos)

    def _trace_once():
        result, _elapsed = _run_sionna(
            rt=rt,
            scene=sionna_scene,
            plane_z=plane_z,
            bounds=bounds,
            grid_size=grid_size,
            samples_per_tx=samples_per_tx,
            diffraction=diffraction,
            edge_diffraction=True,
        )
        return result

    result, timing = _timed_repeat(
        _trace_once,
        sync_result=_sync_sionna_result,
        warmup=warmup,
        repeats=repeats,
    )
    return result, timing, str(conversion.source)


def _save_child_array(*, output_path: Path, path_gain: np.ndarray) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, path_gain=np.asarray(path_gain, dtype=np.float32))
    return output_path


def _run_child_benchmark(
    *,
    framework: str,
    diffraction: bool,
    grid_size: int,
    bounds,
    plane_z: float,
    tx_pos,
    samples_per_tx: int,
    seed: int,
    warmup: int,
    repeats: int,
    array_output: Path,
):
    scene = _build_comparison_scene(
        float(CUBE1_BASE_CENTER[0]),
        edge_selection_mode=DEFAULT_WITWIN_EDGE_SELECTION_MODE,
    )
    mode = "with_diff" if diffraction else "no_diff"
    if framework == "witwin":
        payload, timing = _run_witwin_monte_carlo(
            scene=scene,
            tx_pos=tx_pos,
            plane_z=plane_z,
            bounds=bounds,
            grid_size=grid_size,
            samples_per_tx=samples_per_tx,
            max_diffractions=1 if diffraction else 0,
            seed=seed,
            warmup=warmup,
            repeats=repeats,
        )
        _save_child_array(
            output_path=array_output,
            path_gain=_to_float_grid(payload.path_gain),
        )
        mc_meta = dict(payload.metadata.get("monte_carlo", {}))
        state_pool = dict(mc_meta.get("diffraction_state_pool", {}))
        return {
            "framework": "witwin",
            "mode": str(mode),
            "array_path": str(array_output),
            "timing": asdict(timing),
            "witwin_backend": str(payload.metadata.get("accumulation_backend", {}).get("resolved", "")),
            "witwin_sampling_mode": str(payload.metadata.get("sampling_mode", "")),
            "witwin_reflection_source_paths": int(mc_meta.get("reflection_source_paths", 0)),
            "witwin_diffraction_total_states": int(state_pool.get("total", 0)),
            "witwin_diffraction_kept_states": int(state_pool.get("kept", 0)),
        }
    if framework == "sionna":
        result, timing, sionna_source = _run_sionna_benchmark(
            scene=scene,
            tx_pos=tx_pos,
            plane_z=plane_z,
            bounds=bounds,
            grid_size=grid_size,
            samples_per_tx=samples_per_tx,
            diffraction=diffraction,
            warmup=warmup,
            repeats=repeats,
        )
        _save_child_array(
            output_path=array_output,
            path_gain=_to_float_grid(np.asarray(result.path_gain)[0]),
        )
        return {
            "framework": "sionna",
            "mode": str(mode),
            "array_path": str(array_output),
            "timing": asdict(timing),
            "sionna_source": str(sionna_source),
        }
    raise ValueError(f"Unsupported framework: {framework!r}")


def _run_child_subprocess(
    *,
    framework: str,
    diffraction: bool,
    grid_size: int,
    bounds,
    plane_z: float,
    tx_pos,
    samples_per_tx: int,
    seed: int,
    warmup: int,
    repeats: int,
    array_output: Path,
) -> dict[str, Any]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--child-benchmark",
        "--framework",
        str(framework),
        "--grid-size",
        str(int(grid_size)),
        "--samples-per-tx",
        str(int(samples_per_tx)),
        "--warmup",
        str(int(warmup)),
        "--repeats",
        str(int(repeats)),
        "--seed",
        str(int(seed)),
        "--plane-z",
        str(float(plane_z)),
        "--tx-x",
        str(float(tx_pos[0])),
        "--tx-y",
        str(float(tx_pos[1])),
        "--tx-z",
        str(float(tx_pos[2])),
        "--xmin",
        str(float(bounds[0][0])),
        "--xmax",
        str(float(bounds[0][1])),
        "--ymin",
        str(float(bounds[1][0])),
        "--ymax",
        str(float(bounds[1][1])),
        "--child-array-output",
        str(array_output),
    ]
    if diffraction:
        command.append("--child-diffraction")
    completed = subprocess.run(
        command,
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            completed.stderr.strip()
            or completed.stdout.strip()
            or f"child benchmark failed with exit code {completed.returncode}"
        )
    stdout = completed.stdout.strip()
    if not stdout:
        raise RuntimeError("child benchmark produced no stdout")
    return json.loads(stdout.splitlines()[-1])


def build_comparison(
    *,
    grid_size: int,
    bounds,
    plane_z: float,
    tx_pos,
    samples_per_tx: int,
    db_min: float,
    db_max: float,
    seed: int,
    warmup: int,
    repeats: int,
    isolate_processes: bool = True,
):
    if isolate_processes:
        with tempfile.TemporaryDirectory(prefix="radiomap_compare_") as scratch_dir:
            scratch = Path(scratch_dir)
            witwin_no_diff_child = _run_child_subprocess(
                framework="witwin",
                diffraction=False,
                grid_size=grid_size,
                bounds=bounds,
                plane_z=plane_z,
                tx_pos=tx_pos,
                samples_per_tx=samples_per_tx,
                seed=seed,
                warmup=warmup,
                repeats=repeats,
                array_output=scratch / "witwin_no_diff.npz",
            )
            witwin_with_diff_child = _run_child_subprocess(
                framework="witwin",
                diffraction=True,
                grid_size=grid_size,
                bounds=bounds,
                plane_z=plane_z,
                tx_pos=tx_pos,
                samples_per_tx=samples_per_tx,
                seed=seed,
                warmup=warmup,
                repeats=repeats,
                array_output=scratch / "witwin_with_diff.npz",
            )
            sionna_no_diff_child = _run_child_subprocess(
                framework="sionna",
                diffraction=False,
                grid_size=grid_size,
                bounds=bounds,
                plane_z=plane_z,
                tx_pos=tx_pos,
                samples_per_tx=samples_per_tx,
                seed=seed,
                warmup=warmup,
                repeats=repeats,
                array_output=scratch / "sionna_no_diff.npz",
            )
            sionna_with_diff_child = _run_child_subprocess(
                framework="sionna",
                diffraction=True,
                grid_size=grid_size,
                bounds=bounds,
                plane_z=plane_z,
                tx_pos=tx_pos,
                samples_per_tx=samples_per_tx,
                seed=seed,
                warmup=warmup,
                repeats=repeats,
                array_output=scratch / "sionna_with_diff.npz",
            )

            witwin_total_no_diff = _to_float_grid(np.load(witwin_no_diff_child["array_path"])["path_gain"])
            witwin_total = _to_float_grid(np.load(witwin_with_diff_child["array_path"])["path_gain"])
            sionna_total_no_diff = _to_float_grid(np.load(sionna_no_diff_child["array_path"])["path_gain"])
            sionna_total = _to_float_grid(np.load(sionna_with_diff_child["array_path"])["path_gain"])

            witwin_no_diff_timing = TimingSummary(**witwin_no_diff_child["timing"])
            witwin_with_diff_timing = TimingSummary(**witwin_with_diff_child["timing"])
            sionna_no_diff_timing = TimingSummary(**sionna_no_diff_child["timing"])
            sionna_with_diff_timing = TimingSummary(**sionna_with_diff_child["timing"])
            sionna_source = str(sionna_with_diff_child.get("sionna_source", sionna_no_diff_child.get("sionna_source", "")))
            witwin_meta = witwin_with_diff_child
    else:
        scene = _build_comparison_scene(
            float(CUBE1_BASE_CENTER[0]),
            edge_selection_mode=DEFAULT_WITWIN_EDGE_SELECTION_MODE,
        )
        witwin_no_diff, witwin_no_diff_timing = _run_witwin_monte_carlo(
            scene=scene,
            tx_pos=tx_pos,
            plane_z=plane_z,
            bounds=bounds,
            grid_size=grid_size,
            samples_per_tx=samples_per_tx,
            max_diffractions=0,
            seed=seed,
            warmup=warmup,
            repeats=repeats,
        )
        witwin_with_diff, witwin_with_diff_timing = _run_witwin_monte_carlo(
            scene=scene,
            tx_pos=tx_pos,
            plane_z=plane_z,
            bounds=bounds,
            grid_size=grid_size,
            samples_per_tx=samples_per_tx,
            max_diffractions=1,
            seed=seed,
            warmup=warmup,
            repeats=repeats,
        )
        sionna_no_diff, sionna_no_diff_timing, sionna_source = _run_sionna_benchmark(
            scene=scene,
            tx_pos=tx_pos,
            plane_z=plane_z,
            bounds=bounds,
            grid_size=grid_size,
            samples_per_tx=samples_per_tx,
            diffraction=False,
            warmup=warmup,
            repeats=repeats,
        )
        sionna_with_diff, sionna_with_diff_timing, _ = _run_sionna_benchmark(
            scene=scene,
            tx_pos=tx_pos,
            plane_z=plane_z,
            bounds=bounds,
            grid_size=grid_size,
            samples_per_tx=samples_per_tx,
            diffraction=True,
            warmup=warmup,
            repeats=repeats,
        )
        witwin_total = _to_float_grid(witwin_with_diff.path_gain)
        witwin_total_no_diff = _to_float_grid(witwin_no_diff.path_gain)
        sionna_total = _to_float_grid(np.asarray(sionna_with_diff.path_gain)[0])
        sionna_total_no_diff = _to_float_grid(np.asarray(sionna_no_diff.path_gain)[0])
        mc_meta = dict(witwin_with_diff.metadata.get("monte_carlo", {}))
        state_pool = dict(mc_meta.get("diffraction_state_pool", {}))
        witwin_meta = {
            "witwin_backend": str(
                witwin_with_diff.metadata.get("accumulation_backend", {}).get("resolved", "")
            ),
            "witwin_sampling_mode": str(witwin_with_diff.metadata.get("sampling_mode", "")),
            "witwin_reflection_source_paths": int(mc_meta.get("reflection_source_paths", 0)),
            "witwin_diffraction_total_states": int(state_pool.get("total", 0)),
            "witwin_diffraction_kept_states": int(state_pool.get("kept", 0)),
        }

    witwin_diff_increment = np.maximum(witwin_total - witwin_total_no_diff, 0.0)
    sionna_diff_increment = np.maximum(sionna_total - sionna_total_no_diff, 0.0)
    witwin_total_db = _absolute_db_map(witwin_total, floor_db=db_min)
    sionna_total_db = _absolute_db_map(sionna_total, floor_db=db_min)
    witwin_diff_db = _absolute_db_map(witwin_diff_increment, floor_db=db_min)
    sionna_diff_db = _absolute_db_map(sionna_diff_increment, floor_db=db_min)

    total_delta_db = witwin_total_db - sionna_total_db
    diff_delta_db = witwin_diff_db - sionna_diff_db

    summary = ComparisonSummary(
        grid_size=int(grid_size),
        bounds=tuple((float(a), float(b)) for (a, b) in bounds),
        plane_z=float(plane_z),
        tx_pos=tuple(float(value) for value in tx_pos),
        samples_per_tx=int(samples_per_tx),
        db_min=float(db_min),
        db_max=float(db_max),
        warmup=int(warmup),
        repeats=int(repeats),
        witwin_backend=str(witwin_meta["witwin_backend"]),
        witwin_sampling_mode=str(witwin_meta["witwin_sampling_mode"]),
        witwin_reflection_source_paths=int(witwin_meta["witwin_reflection_source_paths"]),
        witwin_diffraction_total_states=int(witwin_meta["witwin_diffraction_total_states"]),
        witwin_diffraction_kept_states=int(witwin_meta["witwin_diffraction_kept_states"]),
        total_db_corr=_correlation(witwin_total_db, sionna_total_db),
        diff_db_corr=_correlation(witwin_diff_db, sionna_diff_db),
        total_delta_db_mae=float(np.mean(np.abs(total_delta_db))),
        diff_delta_db_mae=float(np.mean(np.abs(diff_delta_db))),
        witwin_no_diff_timing=witwin_no_diff_timing,
        witwin_with_diff_timing=witwin_with_diff_timing,
        sionna_no_diff_timing=sionna_no_diff_timing,
        sionna_with_diff_timing=sionna_with_diff_timing,
        no_diff_speedup_vs_sionna=(
            float(sionna_no_diff_timing.median_ms / witwin_no_diff_timing.median_ms)
            if witwin_no_diff_timing.median_ms > 0.0
            else float("nan")
        ),
        with_diff_speedup_vs_sionna=(
            float(sionna_with_diff_timing.median_ms / witwin_with_diff_timing.median_ms)
            if witwin_with_diff_timing.median_ms > 0.0
            else float("nan")
        ),
        benchmark_isolation_mode=(
            "subprocess_per_framework_mode"
            if isolate_processes
            else "single_process_shared_runtime"
        ),
        sionna_source=str(sionna_source),
        environment=benchmark_environment_report(),
    )

    return {
        "extent": _extent(bounds),
        "summary": summary,
        "witwin": {
            "total": witwin_total,
            "total_db": witwin_total_db,
            "diff_increment": witwin_diff_increment,
            "diff_db": witwin_diff_db,
        },
        "sionna": {
            "total": sionna_total,
            "total_db": sionna_total_db,
            "diff_increment": sionna_diff_increment,
            "diff_db": sionna_diff_db,
        },
        "delta": {
            "total_db": total_delta_db,
            "diff_db": diff_delta_db,
        },
    }


def save_figure(comparison, *, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary: ComparisonSummary = comparison["summary"]
    extent = comparison["extent"]
    bounds = summary.bounds
    tx_pos = summary.tx_pos
    db_min = float(summary.db_min)
    db_max = float(summary.db_max)
    total_delta_vmax = max(6.0, float(np.percentile(np.abs(comparison["delta"]["total_db"]), 99.0)))
    diff_delta_vmax = max(6.0, float(np.percentile(np.abs(comparison["delta"]["diff_db"]), 99.0)))

    fig, axes = plt.subplots(2, 4, figsize=(18.5, 9.0), constrained_layout=True)
    heat_panels = (
        (axes[0, 0], comparison["witwin"]["total_db"], f"Witwin MC Total (dB)", "viridis", db_min, db_max),
        (axes[0, 1], comparison["sionna"]["total_db"], f"Sionna Total (dB)", "viridis", db_min, db_max),
        (axes[0, 2], comparison["delta"]["total_db"], "Total Delta (dB, Witwin - Sionna)", "coolwarm", -total_delta_vmax, total_delta_vmax),
        (axes[1, 0], comparison["witwin"]["diff_db"], "Witwin MC Diff Increment (dB)", "magma", db_min, db_max),
        (axes[1, 1], comparison["sionna"]["diff_db"], "Sionna Diff Increment (dB)", "magma", db_min, db_max),
        (axes[1, 2], comparison["delta"]["diff_db"], "Diff Increment Delta (dB, Witwin - Sionna)", "coolwarm", -diff_delta_vmax, diff_delta_vmax),
    )
    for ax, values, title, cmap, vmin, vmax in heat_panels:
        image = ax.imshow(
            values,
            origin="lower",
            extent=extent,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            interpolation="nearest",
        )
        _decorate_axis(ax, bounds=bounds, cube1_x=float(CUBE1_BASE_CENTER[0]), tx_pos=tx_pos)
        ax.set_title(title, fontsize=10)
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.03)

    runtime_ax = axes[0, 3]
    categories = ("No Diff", "With Diff")
    x = np.arange(len(categories), dtype=np.float32)
    width = 0.34
    witwin_ms = (
        summary.witwin_no_diff_timing.median_ms,
        summary.witwin_with_diff_timing.median_ms,
    )
    sionna_ms = (
        summary.sionna_no_diff_timing.median_ms,
        summary.sionna_with_diff_timing.median_ms,
    )
    runtime_ax.bar(x - width / 2.0, witwin_ms, width=width, label="Witwin MC", color="#2a9d8f")
    runtime_ax.bar(x + width / 2.0, sionna_ms, width=width, label="Sionna RT", color="#e76f51")
    runtime_ax.set_xticks(x, categories)
    runtime_ax.set_ylabel("Median runtime (ms)")
    runtime_ax.set_title("Hot Runtime Benchmark", fontsize=10)
    runtime_ax.legend(loc="upper left")
    runtime_ax.grid(True, axis="y", alpha=0.25)
    for xpos, value in zip(x - width / 2.0, witwin_ms):
        runtime_ax.text(float(xpos), float(value), f"{value:.0f}", ha="center", va="bottom", fontsize=9)
    for xpos, value in zip(x + width / 2.0, sionna_ms):
        runtime_ax.text(float(xpos), float(value), f"{value:.0f}", ha="center", va="bottom", fontsize=9)
    runtime_ax.text(
        0.02,
        0.98,
        (
            f"Sionna/Witwin speedup\n"
            f"No diff: {summary.no_diff_speedup_vs_sionna:.2f}x\n"
            f"With diff: {summary.with_diff_speedup_vs_sionna:.2f}x"
        ),
        transform=runtime_ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        bbox={"facecolor": "white", "edgecolor": "0.7", "alpha": 0.9},
    )

    text_ax = axes[1, 3]
    text_ax.axis("off")
    text_ax.text(
        0.0,
        1.0,
        "\n".join(
            (
                "Setup",
                f"grid={summary.grid_size}x{summary.grid_size}",
                f"plane z={summary.plane_z:.1f}, tx=({summary.tx_pos[0]:.1f}, {summary.tx_pos[1]:.1f}, {summary.tx_pos[2]:.1f})",
                f"samples_per_tx={summary.samples_per_tx:,}",
                f"warmup={summary.warmup}, repeats={summary.repeats}",
                f"isolation={summary.benchmark_isolation_mode}",
                "",
                "Witwin MC",
                f"backend={summary.witwin_backend}",
                f"sampling_mode={summary.witwin_sampling_mode}",
                f"reflection source paths={summary.witwin_reflection_source_paths}",
                f"diffraction states kept/total={summary.witwin_diffraction_kept_states}/{summary.witwin_diffraction_total_states}",
                "",
                "Agreement",
                f"total dB corr={summary.total_db_corr:.4f}",
                f"diff dB corr={summary.diff_db_corr:.4f}",
                f"total dB MAE={summary.total_delta_db_mae:.2f}",
                f"diff dB MAE={summary.diff_delta_db_mae:.2f}",
                "",
                "Source",
                f"Sionna scene={summary.sionna_source}",
            )
        ),
        ha="left",
        va="top",
        fontsize=9,
        family="monospace",
    )

    fig.suptitle(
        "Three-cube radio map: Witwin Monte Carlo vs Sionna RT",
        fontsize=14,
    )
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def save_arrays(comparison, *, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        witwin_total=comparison["witwin"]["total"],
        witwin_diff_increment=comparison["witwin"]["diff_increment"],
        sionna_total=comparison["sionna"]["total"],
        sionna_diff_increment=comparison["sionna"]["diff_increment"],
        witwin_total_db=comparison["witwin"]["total_db"],
        witwin_diff_db=comparison["witwin"]["diff_db"],
        sionna_total_db=comparison["sionna"]["total_db"],
        sionna_diff_db=comparison["sionna"]["diff_db"],
        total_delta_db=comparison["delta"]["total_db"],
        diff_delta_db=comparison["delta"]["diff_db"],
    )
    return output_path


def save_json(comparison, *, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(asdict(comparison["summary"]), indent=2) + "\n",
        encoding="utf-8",
    )
    return output_path


def save_outputs(
    output_prefix: Path,
    *,
    grid_size: int,
    bounds,
    plane_z: float,
    tx_pos,
    samples_per_tx: int,
    db_min: float,
    db_max: float,
    seed: int,
    warmup: int,
    repeats: int,
    isolate_processes: bool = True,
) -> tuple[Path, Path, Path]:
    comparison = build_comparison(
        grid_size=grid_size,
        bounds=bounds,
        plane_z=plane_z,
        tx_pos=tx_pos,
        samples_per_tx=samples_per_tx,
        db_min=db_min,
        db_max=db_max,
        seed=seed,
        warmup=warmup,
        repeats=repeats,
        isolate_processes=isolate_processes,
    )
    figure_path = save_figure(comparison, output_path=output_prefix.with_suffix(".png"))
    arrays_path = save_arrays(comparison, output_path=output_prefix.with_suffix(".npz"))
    json_path = save_json(comparison, output_path=output_prefix.with_suffix(".json"))
    return figure_path, arrays_path, json_path


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid-size", type=int, default=DEFAULT_GRID_SIZE)
    parser.add_argument("--samples-per-tx", type=int, default=DEFAULT_SAMPLES_PER_TX)
    parser.add_argument("--warmup", type=int, default=DEFAULT_WARMUP)
    parser.add_argument("--repeats", type=int, default=DEFAULT_REPEATS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--plane-z", type=float, default=DEFAULT_PLANE_Z)
    parser.add_argument("--db-min", type=float, default=DEFAULT_DB_MIN)
    parser.add_argument("--db-max", type=float, default=DEFAULT_DB_MAX)
    parser.add_argument("--tx-x", type=float, default=DEFAULT_TX_POS[0])
    parser.add_argument("--tx-y", type=float, default=DEFAULT_TX_POS[1])
    parser.add_argument("--tx-z", type=float, default=4.0)
    parser.add_argument("--xmin", type=float, default=DEFAULT_BOUNDS[0][0])
    parser.add_argument("--xmax", type=float, default=DEFAULT_BOUNDS[0][1])
    parser.add_argument("--ymin", type=float, default=DEFAULT_BOUNDS[1][0])
    parser.add_argument("--ymax", type=float, default=DEFAULT_BOUNDS[1][1])
    parser.add_argument("--output-prefix", type=Path, default=DEFAULT_OUTPUT_PREFIX)
    parser.add_argument("--same-process", action="store_true")
    parser.add_argument("--child-benchmark", action="store_true")
    parser.add_argument("--framework", type=str, choices=("witwin", "sionna"))
    parser.add_argument("--child-array-output", type=Path)
    parser.add_argument("--child-diffraction", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.child_benchmark:
        if args.framework is None or args.child_array_output is None:
            raise SystemExit("--child-benchmark requires --framework and --child-array-output")
        child_result = _run_child_benchmark(
            framework=str(args.framework),
            diffraction=bool(args.child_diffraction),
            grid_size=int(args.grid_size),
            bounds=((float(args.xmin), float(args.xmax)), (float(args.ymin), float(args.ymax))),
            plane_z=float(args.plane_z),
            tx_pos=(float(args.tx_x), float(args.tx_y), float(args.tx_z)),
            samples_per_tx=int(args.samples_per_tx),
            seed=int(args.seed),
            warmup=int(args.warmup),
            repeats=int(args.repeats),
            array_output=args.child_array_output,
        )
        print(json.dumps(child_result, separators=(",", ":")))
        return
    figure_path, arrays_path, json_path = save_outputs(
        args.output_prefix,
        grid_size=int(args.grid_size),
        bounds=((float(args.xmin), float(args.xmax)), (float(args.ymin), float(args.ymax))),
        plane_z=float(args.plane_z),
        tx_pos=(float(args.tx_x), float(args.tx_y), float(args.tx_z)),
        samples_per_tx=int(args.samples_per_tx),
        db_min=float(args.db_min),
        db_max=float(args.db_max),
        seed=int(args.seed),
        warmup=int(args.warmup),
        repeats=int(args.repeats),
        isolate_processes=not bool(args.same_process),
    )
    print(
        json.dumps(
            {
                "figure": str(figure_path),
                "arrays": str(arrays_path),
                "json": str(json_path),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
