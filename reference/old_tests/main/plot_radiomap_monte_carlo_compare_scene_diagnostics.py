"""Unified three-cube Monte Carlo forward and gradient diagnostics."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from time import perf_counter

import drjit as dr
if os.environ.get("WITWIN_CHANNEL_MAIN_SHOW", "0") != "1":
    os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import numpy as np
import witwin as wt

from tests._scene_helpers import box_drjit_geometry, build_scene as build_test_scene
from tests.main.plot_multipath_components import (
    CUBE1_BASE_CENTER,
    CUBE2_CENTER,
    CUBE3_CENTER,
    CUBE_SIZE,
    MULTIPATH_SCENE_MATERIAL,
    gradient_db_magnitude,
)
from tests.main.plot_radiomap_gradients_three_cubes import (
    _as_grid,
    _format_seconds,
    _scalar_from_drjit,
    _time_call,
    parameter_config,
)
from tests.main.plot_radiomap_sionna_three_cubes import (
    DEFAULT_BOUNDS,
    DEFAULT_DB_MAX,
    DEFAULT_DB_MIN,
    DEFAULT_PLANE_Z,
    DEFAULT_TX_POS,
    DEFAULT_WITWIN_EDGE_SELECTION_MODE,
    _absolute_db_map,
    _correlation,
    _decorate_axis,
    _output_dir,
    _prepare_sionna_scene,
    _run_sionna,
    _to_float_grid,
)
from tests.support.bin.compare_radiomap_sionna_three_cubes import (
    build_comparison,
)
from witwin.channel import RadioMapMonitor, Tracer
DEFAULT_GRID_SIZE = 256
DEFAULT_SAMPLES_PER_TX = 250_000
DEFAULT_FD_STEP = 1.0e-3
DEFAULT_WARMUP = 1
DEFAULT_REPEATS = 3
DEFAULT_SEED = 7
DEFAULT_OUTPUT_PREFIX = (
    _output_dir() / "radiomap_three_cubes_monte_carlo_compare_scene_diagnostics"
)
_GRAD_FLAGS = dr.ADFlag.Default | dr.ADFlag.AllowNoGrad


def _is_python_scalar(value) -> bool:
    return isinstance(value, (int, float, np.floating))


@lru_cache(maxsize=16)
def _build_compare_scene_cached(cube1_x: float):
    cube1_center = wt.Point3f(cube1_x, CUBE1_BASE_CENTER[1], CUBE1_BASE_CENTER[2])
    cube1 = box_drjit_geometry(center=cube1_center, size=CUBE_SIZE, rotation=None).to_mesh()
    cube2 = box_drjit_geometry(center=CUBE2_CENTER, size=CUBE_SIZE, rotation=None).to_mesh()
    cube3 = box_drjit_geometry(center=CUBE3_CENTER, size=CUBE_SIZE, rotation=None).to_mesh()
    return build_test_scene(
        cube1,
        cube2,
        cube3,
        material=MULTIPATH_SCENE_MATERIAL,
        edge_selection_mode=str(DEFAULT_WITWIN_EDGE_SELECTION_MODE),
    )


def _build_compare_scene_for_cube1_x(cube1_x):
    if _is_python_scalar(cube1_x):
        return _build_compare_scene_cached(float(cube1_x))
    cube1_center = wt.Point3f(cube1_x, CUBE1_BASE_CENTER[1], CUBE1_BASE_CENTER[2])
    cube1 = box_drjit_geometry(center=cube1_center, size=CUBE_SIZE, rotation=None).to_mesh()
    cube2 = box_drjit_geometry(center=CUBE2_CENTER, size=CUBE_SIZE, rotation=None).to_mesh()
    cube3 = box_drjit_geometry(center=CUBE3_CENTER, size=CUBE_SIZE, rotation=None).to_mesh()
    return build_test_scene(
        cube1,
        cube2,
        cube3,
        material=MULTIPATH_SCENE_MATERIAL,
        edge_selection_mode=str(DEFAULT_WITWIN_EDGE_SELECTION_MODE),
    )


@dataclass(frozen=True)
class ForwardConsistencySummary:
    total_max_abs_diff: float
    total_mean_abs_diff: float
    total_sum_abs_diff: float
    diff_increment_max_abs_diff: float
    diff_increment_mean_abs_diff: float
    diff_increment_sum_abs_diff: float
    noad_path_counts: dict
    ad_path_counts: dict
    noad_monte_carlo: dict
    ad_monte_carlo: dict


@dataclass(frozen=True)
class CompareSceneGradientSummary:
    grid_size: int
    bounds: tuple[tuple[float, float], tuple[float, float]]
    plane_z: float
    tx_pos: tuple[float, float, float]
    samples_per_tx: int
    reflection_n_rays: int
    reflection_max_bounces: int
    fd_step: float
    warmup: int
    repeats: int
    forward_total_noad_seconds: float
    forward_total_ad_seconds: float
    compare_summary: dict
    one_shot_compare: dict
    forward_consistency: dict
    timings_seconds: dict
    parameters: dict


def _seed_all(seed: int) -> None:
    try:
        dr.seed(int(seed))
    except Exception:
        pass
    try:
        wt.register_sampler_seed(int(seed))
    except Exception:
        pass


def _make_compare_tracer(scene, *, reflection_n_rays: int, max_diffractions: int) -> Tracer:
    return Tracer(
        frequency=1.0e9,
        scene=scene,
        reflection_n_rays=int(reflection_n_rays),
        reflection_max_bounces=3,
        reflection_coef=0.8,
        enable_rd_diffraction=bool(int(max_diffractions) > 0),
        max_diffractions=int(max_diffractions),
    )


def _make_compare_monitor(
    *,
    name: str,
    grid_size: int,
    bounds,
    plane_z: float,
    samples_per_tx: int,
    max_diffractions: int,
    seed: int,
    ad: bool | None,
):
    monitor_kwargs = dict(
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
    if ad is not None:
        monitor_kwargs["ad"] = bool(ad)
    return RadioMapMonitor(name, **monitor_kwargs)


def _trace_compare_scene_payload(
    *,
    cube1_x,
    tx_pos,
    grid_size: int,
    bounds,
    plane_z: float,
    samples_per_tx: int,
    max_diffractions: int,
    seed: int,
    ad: bool | None,
):
    _seed_all(seed)
    scene = _build_compare_scene_for_cube1_x(cube1_x)
    tracer = _make_compare_tracer(
        scene,
        reflection_n_rays=samples_per_tx,
        max_diffractions=max_diffractions,
    )
    monitor = _make_compare_monitor(
        name="three_cubes_compare_scene_mc",
        grid_size=grid_size,
        bounds=bounds,
        plane_z=plane_z,
        samples_per_tx=samples_per_tx,
        max_diffractions=max_diffractions,
        seed=seed,
        ad=ad,
    )
    trace_output = tracer.trace(tx_pos, monitor=monitor, verbose=False)
    result = trace_output.monitor(monitor.name) if hasattr(trace_output, "monitor") else trace_output
    return {
        "coords": {
            "grid_x": result.coords.grid_x,
            "grid_y": result.coords.grid_y,
        },
        "metrics": {
            "path_gain": result.path_gain,
        },
        "metadata": result.metadata,
    }


def _trace_total_map(
    *,
    cube1_x,
    tx_pos,
    grid_size: int,
    bounds,
    plane_z: float,
    samples_per_tx: int,
    max_diffractions: int,
    seed: int,
    ad: bool | None,
):
    payload = _trace_compare_scene_payload(
        cube1_x=cube1_x,
        tx_pos=tx_pos,
        grid_size=grid_size,
        bounds=bounds,
        plane_z=plane_z,
        samples_per_tx=samples_per_tx,
        max_diffractions=max_diffractions,
        seed=seed,
        ad=ad,
    )
    return (
        np.asarray(payload["coords"]["grid_x"], dtype=np.float64),
        np.asarray(payload["coords"]["grid_y"], dtype=np.float64),
        np.asarray(payload["metrics"]["path_gain"], dtype=np.float64),
        payload,
    )


def _trace_diff_increment(
    *,
    cube1_x,
    tx_pos,
    grid_size: int,
    bounds,
    plane_z: float,
    samples_per_tx: int,
    seed: int,
    ad: bool | None,
):
    _, _, no_diff, no_diff_payload = _trace_total_map(
        cube1_x=cube1_x,
        tx_pos=tx_pos,
        grid_size=grid_size,
        bounds=bounds,
        plane_z=plane_z,
        samples_per_tx=samples_per_tx,
        max_diffractions=0,
        seed=seed,
        ad=ad,
    )
    grid_x, grid_y, with_diff, with_diff_payload = _trace_total_map(
        cube1_x=cube1_x,
        tx_pos=tx_pos,
        grid_size=grid_size,
        bounds=bounds,
        plane_z=plane_z,
        samples_per_tx=samples_per_tx,
        max_diffractions=1,
        seed=seed,
        ad=ad,
    )
    return {
        "grid_x": grid_x,
        "grid_y": grid_y,
        "total": with_diff,
        "total_no_diff": no_diff,
        "diff_increment": np.maximum(with_diff - no_diff, 0.0),
        "metadata": with_diff_payload["metadata"],
        "metadata_no_diff": no_diff_payload["metadata"],
    }


def _trace_sionna_snapshot(
    *,
    tx_pos,
    grid_size: int,
    bounds,
    plane_z: float,
    samples_per_tx: int,
):
    scene = _build_compare_scene_cached(float(CUBE1_BASE_CENTER[0]))
    conversion, rt, sionna_scene = _prepare_sionna_scene(scene=scene, tx_pos=tx_pos)
    del conversion
    no_diff, _ = _run_sionna(
        rt=rt,
        scene=sionna_scene,
        plane_z=plane_z,
        bounds=bounds,
        grid_size=grid_size,
        samples_per_tx=samples_per_tx,
        diffraction=False,
        edge_diffraction=True,
    )
    with_diff, _ = _run_sionna(
        rt=rt,
        scene=sionna_scene,
        plane_z=plane_z,
        bounds=bounds,
        grid_size=grid_size,
        samples_per_tx=samples_per_tx,
        diffraction=True,
        edge_diffraction=True,
    )
    total = _to_float_grid(np.asarray(with_diff.path_gain)[0])
    total_no_diff = _to_float_grid(np.asarray(no_diff.path_gain)[0])
    diff_increment = np.maximum(total - total_no_diff, 0.0)
    return {
        "total": total,
        "total_no_diff": total_no_diff,
        "diff_increment": diff_increment,
        "total_db": _absolute_db_map(total, floor_db=DEFAULT_DB_MIN),
        "diff_db": _absolute_db_map(diff_increment, floor_db=DEFAULT_DB_MIN),
    }


def _pair_stats(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    diff = a - b
    return {
        "max_abs_diff": float(np.max(np.abs(diff))),
        "mean_abs_diff": float(np.mean(np.abs(diff))),
        "sum_abs_diff": float(np.sum(np.abs(diff))),
    }


def _ad_gradient_total(
    parameter: str,
    *,
    tx_pos,
    grid_size: int,
    bounds,
    plane_z: float,
    samples_per_tx: int,
    seed: int,
):
    config = parameter_config(parameter, tx_pos=tx_pos)
    if parameter == "cube1_x":
        cube1_x = wt.Float(config["cube1_x"])
        dr.enable_grad(cube1_x)
        payload = _trace_compare_scene_payload(
            cube1_x=cube1_x,
            tx_pos=wt.Point3f(*config["tx_pos"]),
            grid_size=grid_size,
            bounds=bounds,
            plane_z=plane_z,
            samples_per_tx=samples_per_tx,
            max_diffractions=1,
            seed=seed,
            ad=True,
        )
        parameter_value = cube1_x
    else:
        tx_x = wt.Float(config["tx_pos"][0])
        dr.enable_grad(tx_x)
        payload = _trace_compare_scene_payload(
            cube1_x=config["cube1_x"],
            tx_pos=wt.Point3f(tx_x, config["tx_pos"][1], config["tx_pos"][2]),
            grid_size=grid_size,
            bounds=bounds,
            plane_z=plane_z,
            samples_per_tx=samples_per_tx,
            max_diffractions=1,
            seed=seed,
            ad=True,
        )
        parameter_value = tx_x
    dr.set_grad(parameter_value, 1.0)
    grad = dr.forward_to(payload["metrics"]["path_gain"], flags=_GRAD_FLAGS)
    return (
        payload["coords"]["grid_x"],
        payload["coords"]["grid_y"],
        np.asarray(grad, dtype=np.float64),
        payload,
    )


def _fd_gradient_total(
    parameter: str,
    *,
    tx_pos,
    grid_size: int,
    bounds,
    plane_z: float,
    samples_per_tx: int,
    seed: int,
    fd_step: float,
):
    config = parameter_config(parameter, tx_pos=tx_pos)
    plus_cfg, minus_cfg = config["perturb"](fd_step)
    _, _, plus_path_gain, _ = _trace_total_map(
        cube1_x=plus_cfg["cube1_x"],
        tx_pos=wt.Point3f(*plus_cfg["tx_pos"]),
        grid_size=grid_size,
        bounds=bounds,
        plane_z=plane_z,
        samples_per_tx=samples_per_tx,
        max_diffractions=1,
        seed=seed,
        ad=False,
    )
    _, _, minus_path_gain, _ = _trace_total_map(
        cube1_x=minus_cfg["cube1_x"],
        tx_pos=wt.Point3f(*minus_cfg["tx_pos"]),
        grid_size=grid_size,
        bounds=bounds,
        plane_z=plane_z,
        samples_per_tx=samples_per_tx,
        max_diffractions=1,
        seed=seed,
        ad=False,
    )
    return np.asarray(
        (plus_path_gain - minus_path_gain) / (2.0 * float(fd_step)),
        dtype=np.float64,
    )


def _vjp_scalar_loss_with_backward_timing(
    parameter: str,
    *,
    tx_pos,
    grid_size: int,
    bounds,
    plane_z: float,
    samples_per_tx: int,
    seed: int,
):
    config = parameter_config(parameter, tx_pos=tx_pos)
    weights_np = np.asarray(
        np.cos(4.0 * np.pi * np.linspace(-1.0, 1.0, int(grid_size), dtype=np.float32))[:, None]
        * np.cos(4.0 * np.pi * np.linspace(-1.0, 1.0, int(grid_size), dtype=np.float32))[None, :],
        dtype=np.float32,
    )
    weights_np = weights_np - weights_np.mean(dtype=np.float64)
    scale = max(float(np.max(np.abs(weights_np))), 1.0)
    weights_np = (weights_np / scale).astype(np.float32)
    if parameter == "cube1_x":
        cube1_x = wt.Float(config["cube1_x"])
        dr.enable_grad(cube1_x)
        payload = _trace_compare_scene_payload(
            cube1_x=cube1_x,
            tx_pos=wt.Point3f(*config["tx_pos"]),
            grid_size=grid_size,
            bounds=bounds,
            plane_z=plane_z,
            samples_per_tx=samples_per_tx,
            max_diffractions=1,
            seed=seed,
            ad=True,
        )
        loss = dr.sum(payload["metrics"]["path_gain"] * type(payload["metrics"]["path_gain"])(weights_np))
        dr.eval(loss)
        dr.sync_thread()
        backward_start = perf_counter()
        dr.backward(loss, flags=_GRAD_FLAGS)
        grad = dr.grad(cube1_x)
        dr.eval(grad)
        dr.sync_thread()
        return _scalar_from_drjit(grad), float(perf_counter() - backward_start)

    tx_x = wt.Float(config["tx_pos"][0])
    dr.enable_grad(tx_x)
    payload = _trace_compare_scene_payload(
        cube1_x=config["cube1_x"],
        tx_pos=wt.Point3f(tx_x, config["tx_pos"][1], config["tx_pos"][2]),
        grid_size=grid_size,
        bounds=bounds,
        plane_z=plane_z,
        samples_per_tx=samples_per_tx,
        max_diffractions=1,
        seed=seed,
        ad=True,
    )
    loss = dr.sum(payload["metrics"]["path_gain"] * type(payload["metrics"]["path_gain"])(weights_np))
    dr.eval(loss)
    dr.sync_thread()
    backward_start = perf_counter()
    dr.backward(loss, flags=_GRAD_FLAGS)
    grad = dr.grad(tx_x)
    dr.eval(grad)
    dr.sync_thread()
    return _scalar_from_drjit(grad), float(perf_counter() - backward_start)


def build_diagnostics(
    *,
    grid_size: int,
    bounds,
    plane_z: float,
    tx_pos,
    samples_per_tx: int,
    fd_step: float,
    seed: int,
    warmup: int,
    repeats: int,
):
    comparison = build_comparison(
        grid_size=int(grid_size),
        bounds=bounds,
        plane_z=float(plane_z),
        tx_pos=tuple(float(value) for value in tx_pos),
        samples_per_tx=int(samples_per_tx),
        db_min=float(DEFAULT_DB_MIN),
        db_max=float(DEFAULT_DB_MAX),
        seed=int(seed),
        warmup=int(warmup),
        repeats=int(repeats),
        isolate_processes=True,
    )

    noad_forward, noad_seconds = _time_call(
        _trace_diff_increment,
        cube1_x=float(CUBE1_BASE_CENTER[0]),
        tx_pos=wt.Point3f(*tx_pos),
        grid_size=grid_size,
        bounds=bounds,
        plane_z=plane_z,
        samples_per_tx=samples_per_tx,
        seed=seed,
        ad=False,
    )
    ad_forward, ad_seconds = _time_call(
        _trace_diff_increment,
        cube1_x=float(CUBE1_BASE_CENTER[0]),
        tx_pos=wt.Point3f(*tx_pos),
        grid_size=grid_size,
        bounds=bounds,
        plane_z=plane_z,
        samples_per_tx=samples_per_tx,
        seed=seed,
        ad=True,
    )
    sionna_snapshot = _trace_sionna_snapshot(
        tx_pos=tx_pos,
        grid_size=grid_size,
        bounds=bounds,
        plane_z=plane_z,
        samples_per_tx=samples_per_tx,
    )

    total_pair = _pair_stats(noad_forward["total"], ad_forward["total"])
    diff_pair = _pair_stats(noad_forward["diff_increment"], ad_forward["diff_increment"])
    forward_consistency = ForwardConsistencySummary(
        total_max_abs_diff=total_pair["max_abs_diff"],
        total_mean_abs_diff=total_pair["mean_abs_diff"],
        total_sum_abs_diff=total_pair["sum_abs_diff"],
        diff_increment_max_abs_diff=diff_pair["max_abs_diff"],
        diff_increment_mean_abs_diff=diff_pair["mean_abs_diff"],
        diff_increment_sum_abs_diff=diff_pair["sum_abs_diff"],
        noad_path_counts=dict(noad_forward["metadata"].get("path_counts", {})),
        ad_path_counts=dict(ad_forward["metadata"].get("path_counts", {})),
        noad_monte_carlo=dict(noad_forward["metadata"].get("monte_carlo", {})),
        ad_monte_carlo=dict(ad_forward["metadata"].get("monte_carlo", {})),
    )

    noad_total_db = _absolute_db_map(noad_forward["total"], floor_db=DEFAULT_DB_MIN)
    ad_total_db = _absolute_db_map(ad_forward["total"], floor_db=DEFAULT_DB_MIN)
    noad_diff_db = _absolute_db_map(noad_forward["diff_increment"], floor_db=DEFAULT_DB_MIN)
    ad_diff_db = _absolute_db_map(ad_forward["diff_increment"], floor_db=DEFAULT_DB_MIN)
    compare_total_delta_db = noad_total_db - sionna_snapshot["total_db"]
    compare_diff_delta_db = noad_diff_db - sionna_snapshot["diff_db"]
    one_shot_compare = {
        "total_db_corr": float(_correlation(noad_total_db, sionna_snapshot["total_db"])),
        "diff_db_corr": float(_correlation(noad_diff_db, sionna_snapshot["diff_db"])),
        "total_delta_db_mae": float(np.mean(np.abs(compare_total_delta_db))),
        "diff_delta_db_mae": float(np.mean(np.abs(compare_diff_delta_db))),
    }

    parameter_results = {}
    timings_seconds = {}
    for parameter in ("tx_x", "cube1_x"):
        (_, _, ad_gradient, _), ad_grad_seconds = _time_call(
            _ad_gradient_total,
            parameter,
            tx_pos=tx_pos,
            grid_size=grid_size,
            bounds=bounds,
            plane_z=plane_z,
            samples_per_tx=samples_per_tx,
            seed=seed,
        )
        fd_gradient, fd_grad_seconds = _time_call(
            _fd_gradient_total,
            parameter,
            tx_pos=tx_pos,
            grid_size=grid_size,
            bounds=bounds,
            plane_z=plane_z,
            samples_per_tx=samples_per_tx,
            seed=seed,
            fd_step=fd_step,
        )
        ad_grid = _as_grid(ad_gradient, grid_size)
        fd_grid = _as_grid(fd_gradient, grid_size)
        ad_vis = gradient_db_magnitude(ad_grid)
        fd_vis = gradient_db_magnitude(fd_grid)
        diff_vis = ad_vis - fd_vis
        (scalar_vjp, backward_only_seconds), vjp_seconds = _time_call(
            _vjp_scalar_loss_with_backward_timing,
            parameter,
            tx_pos=tx_pos,
            grid_size=grid_size,
            bounds=bounds,
            plane_z=plane_z,
            samples_per_tx=samples_per_tx,
            seed=seed,
        )
        parameter_results[parameter] = {
            "ad": ad_grid,
            "fd": fd_grid,
            "ad_vis": ad_vis,
            "fd_vis": fd_vis,
            "diff_vis": diff_vis,
            "ad_abs_sum": float(np.sum(np.abs(ad_grid))),
            "fd_abs_sum": float(np.sum(np.abs(fd_grid))),
            "ad_fd_max_abs_diff": float(np.max(np.abs(ad_grid - fd_grid))),
            "ad_fd_mean_abs_diff": float(np.mean(np.abs(ad_grid - fd_grid))),
            "ad_fd_corr": float(np.corrcoef(ad_grid.ravel(), fd_grid.ravel())[0, 1]),
            "timings_seconds": {
                "ad": float(ad_grad_seconds),
                "fd": float(fd_grad_seconds),
                "vjp": float(vjp_seconds),
                "backward_only": float(backward_only_seconds),
                "scalar_vjp": float(scalar_vjp),
            },
        }
        timings_seconds[parameter] = dict(parameter_results[parameter]["timings_seconds"])

    summary = CompareSceneGradientSummary(
        grid_size=int(grid_size),
        bounds=tuple((float(a), float(b)) for (a, b) in bounds),
        plane_z=float(plane_z),
        tx_pos=tuple(float(value) for value in tx_pos),
        samples_per_tx=int(samples_per_tx),
        reflection_n_rays=int(samples_per_tx),
        reflection_max_bounces=3,
        fd_step=float(fd_step),
        warmup=int(warmup),
        repeats=int(repeats),
        forward_total_noad_seconds=float(noad_seconds),
        forward_total_ad_seconds=float(ad_seconds),
        compare_summary=asdict(comparison["summary"]),
        one_shot_compare=one_shot_compare,
        forward_consistency=asdict(forward_consistency),
        timings_seconds=timings_seconds,
        parameters={
            parameter: {
                key: value
                for key, value in parameter_results[parameter].items()
                if key not in {"ad", "fd", "ad_vis", "fd_vis", "diff_vis"}
            }
            for parameter in parameter_results
        },
    )

    return {
        "comparison": comparison,
        "compare_scene": {
            "extent": comparison["extent"],
            "witwin": {
                "total": noad_forward["total"],
                "diff_increment": noad_forward["diff_increment"],
                "total_db": noad_total_db,
                "diff_db": noad_diff_db,
            },
            "sionna": sionna_snapshot,
            "delta": {
                "total_db": compare_total_delta_db,
                "diff_db": compare_diff_delta_db,
            },
        },
        "summary": summary,
        "forward_noad": {
            "total": noad_forward["total"],
            "diff_increment": noad_forward["diff_increment"],
            "total_db": noad_total_db,
            "diff_db": noad_diff_db,
        },
        "forward_ad": {
            "total": ad_forward["total"],
            "diff_increment": ad_forward["diff_increment"],
            "total_db": ad_total_db,
            "diff_db": ad_diff_db,
        },
        "parameter_results": parameter_results,
        "cube1_x": float(CUBE1_BASE_CENTER[0]),
    }


def save_compare_scene_figure(diagnostics, *, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary = diagnostics["summary"]
    compare_summary = summary.compare_summary
    compare_scene = diagnostics["compare_scene"]
    extent = compare_scene["extent"]
    bounds = summary.bounds
    tx_pos = summary.tx_pos
    cube1_x = diagnostics["cube1_x"]
    total_delta_vmax = max(6.0, float(np.percentile(np.abs(compare_scene["delta"]["total_db"]), 99.0)))
    diff_delta_vmax = max(6.0, float(np.percentile(np.abs(compare_scene["delta"]["diff_db"]), 99.0)))

    fig, axes = plt.subplots(2, 4, figsize=(18.5, 9.0), constrained_layout=True)
    heat_panels = (
        (axes[0, 0], compare_scene["witwin"]["total_db"], "Witwin MC Total (one-shot, dB)", "viridis", DEFAULT_DB_MIN, DEFAULT_DB_MAX),
        (axes[0, 1], compare_scene["sionna"]["total_db"], "Sionna Total (one-shot, dB)", "viridis", DEFAULT_DB_MIN, DEFAULT_DB_MAX),
        (axes[0, 2], compare_scene["delta"]["total_db"], "Total Delta (dB, Witwin - Sionna)", "coolwarm", -total_delta_vmax, total_delta_vmax),
        (axes[1, 0], compare_scene["witwin"]["diff_db"], "Witwin MC Diff Increment (one-shot, dB)", "magma", DEFAULT_DB_MIN, DEFAULT_DB_MAX),
        (axes[1, 1], compare_scene["sionna"]["diff_db"], "Sionna Diff Increment (one-shot, dB)", "magma", DEFAULT_DB_MIN, DEFAULT_DB_MAX),
        (axes[1, 2], compare_scene["delta"]["diff_db"], "Diff Increment Delta (dB, Witwin - Sionna)", "coolwarm", -diff_delta_vmax, diff_delta_vmax),
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
        _decorate_axis(ax, bounds=bounds, cube1_x=cube1_x, tx_pos=tx_pos)
        ax.set_title(title, fontsize=10)
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.03)

    runtime_ax = axes[0, 3]
    categories = ("No Diff", "With Diff")
    x = np.arange(len(categories), dtype=np.float32)
    width = 0.34
    witwin_ms = (
        compare_summary["witwin_no_diff_timing"]["median_ms"],
        compare_summary["witwin_with_diff_timing"]["median_ms"],
    )
    sionna_ms = (
        compare_summary["sionna_no_diff_timing"]["median_ms"],
        compare_summary["sionna_with_diff_timing"]["median_ms"],
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
            "Sionna/Witwin speedup\n"
            f"No diff: {compare_summary['no_diff_speedup_vs_sionna']:.2f}x\n"
            f"With diff: {compare_summary['with_diff_speedup_vs_sionna']:.2f}x"
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
                f"reflection_max_bounces={summary.reflection_max_bounces}",
                f"edge_selection_mode={DEFAULT_WITWIN_EDGE_SELECTION_MODE}",
                "",
                "Witwin MC",
                f"backend={compare_summary['witwin_backend']}",
                f"sampling_mode={compare_summary['witwin_sampling_mode']}",
                "",
                "One-shot agreement",
                f"total dB corr={summary.one_shot_compare['total_db_corr']:.4f}",
                f"diff dB corr={summary.one_shot_compare['diff_db_corr']:.4f}",
                f"total dB MAE={summary.one_shot_compare['total_delta_db_mae']:.2f}",
                f"diff dB MAE={summary.one_shot_compare['diff_delta_db_mae']:.2f}",
                "",
                "Benchmark source",
                f"isolation={compare_summary['benchmark_isolation_mode']}",
                f"Sionna scene={compare_summary['sionna_source']}",
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


def save_forward_consistency_figure(diagnostics, *, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    comparison = diagnostics["comparison"]
    extent = comparison["extent"]
    bounds = diagnostics["summary"].bounds
    tx_pos = diagnostics["summary"].tx_pos
    cube1_x = diagnostics["cube1_x"]
    total_delta = diagnostics["forward_ad"]["total_db"] - diagnostics["forward_noad"]["total_db"]
    diff_delta = diagnostics["forward_ad"]["diff_db"] - diagnostics["forward_noad"]["diff_db"]
    total_delta_vmax = max(3.0, float(np.percentile(np.abs(total_delta), 99.0)))
    diff_delta_vmax = max(3.0, float(np.percentile(np.abs(diff_delta), 99.0)))

    fig, axes = plt.subplots(2, 3, figsize=(14.5, 8.6), constrained_layout=True)
    panels = (
        (axes[0, 0], diagnostics["forward_noad"]["total_db"], "Witwin MC Total (ad=False, dB)", "viridis", DEFAULT_DB_MIN, DEFAULT_DB_MAX),
        (axes[0, 1], diagnostics["forward_ad"]["total_db"], "Witwin MC Total (ad=True, dB)", "viridis", DEFAULT_DB_MIN, DEFAULT_DB_MAX),
        (axes[0, 2], total_delta, "Total Delta (dB, ad=True - ad=False)", "coolwarm", -total_delta_vmax, total_delta_vmax),
        (axes[1, 0], diagnostics["forward_noad"]["diff_db"], "Witwin MC Diff Increment (ad=False, dB)", "magma", DEFAULT_DB_MIN, DEFAULT_DB_MAX),
        (axes[1, 1], diagnostics["forward_ad"]["diff_db"], "Witwin MC Diff Increment (ad=True, dB)", "magma", DEFAULT_DB_MIN, DEFAULT_DB_MAX),
        (axes[1, 2], diff_delta, "Diff Increment Delta (dB, ad=True - ad=False)", "coolwarm", -diff_delta_vmax, diff_delta_vmax),
    )
    for ax, values, title, cmap, vmin, vmax in panels:
        image = ax.imshow(
            values,
            origin="lower",
            extent=extent,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            interpolation="nearest",
        )
        _decorate_axis(ax, bounds=bounds, cube1_x=cube1_x, tx_pos=tx_pos)
        ax.set_title(title, fontsize=10)
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.03)

    consistency = diagnostics["summary"].forward_consistency
    fig.suptitle(
        "Three-cube Monte Carlo forward consistency under compare-scene parameters\n"
        f"grid={diagnostics['summary'].grid_size}, samples_per_tx={diagnostics['summary'].samples_per_tx}, "
        f"reflection_n_rays={diagnostics['summary'].reflection_n_rays}, reflection_max_bounces=3, "
        f"total max|ad-noad|={consistency['total_max_abs_diff']:.3e}, "
        f"diff max|ad-noad|={consistency['diff_increment_max_abs_diff']:.3e}",
        fontsize=13,
    )
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def save_gradient_figure(diagnostics, *, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    comparison = diagnostics["comparison"]
    extent = comparison["extent"]
    bounds = diagnostics["summary"].bounds
    tx_pos = diagnostics["summary"].tx_pos
    cube1_x = diagnostics["cube1_x"]
    total_db = diagnostics["forward_noad"]["total_db"]

    fig, axes = plt.subplots(2, 4, figsize=(16.0, 8.8), constrained_layout=True)
    for row, parameter in enumerate(("tx_x", "cube1_x")):
        row_results = diagnostics["parameter_results"][parameter]
        row_timings = row_results["timings_seconds"]
        diff_vmax = max(float(np.percentile(np.abs(row_results["diff_vis"]), 99.0)), 1.0)
        panels = (
            (
                axes[row, 0],
                total_db,
                (
                    f"MC Path Gain (ad=False, dB), {parameter}\n"
                    f"fwd noad={_format_seconds(diagnostics['summary'].forward_total_noad_seconds)} | "
                    f"fwd ad={_format_seconds(diagnostics['summary'].forward_total_ad_seconds)}"
                ),
                "viridis",
                float(np.nanpercentile(total_db, 1.0)),
                float(np.nanpercentile(total_db, 99.0)),
            ),
            (
                axes[row, 1],
                row_results["ad_vis"],
                f"AD |d path_gain / d {parameter}| (dB)\nAD={_format_seconds(row_timings['ad'])}",
                "magma",
                float(np.nanpercentile(row_results["ad_vis"], 5.0)),
                float(np.nanpercentile(row_results["ad_vis"], 99.0)),
            ),
            (
                axes[row, 2],
                row_results["fd_vis"],
                f"FD |d path_gain / d {parameter}| (dB)\nFD={_format_seconds(row_timings['fd'])}",
                "magma",
                float(np.nanpercentile(row_results["fd_vis"], 5.0)),
                float(np.nanpercentile(row_results["fd_vis"], 99.0)),
            ),
            (
                axes[row, 3],
                row_results["diff_vis"],
                (
                    f"AD-FD Gradient Delta (dB), {parameter}\n"
                    f"VJP={_format_seconds(row_timings['vjp'])} | "
                    f"BWD={_format_seconds(row_timings['backward_only'])}"
                ),
                "coolwarm",
                -diff_vmax,
                diff_vmax,
            ),
        )
        for ax, image, title, cmap, vmin, vmax in panels:
            im = ax.imshow(
                image,
                origin="lower",
                extent=extent,
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                interpolation="nearest",
            )
            _decorate_axis(ax, bounds=bounds, cube1_x=cube1_x, tx_pos=tx_pos)
            ax.set_title(title, fontsize=10)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)

    fig.suptitle(
        "Three-cube Monte Carlo gradients under compare-scene parameters\n"
        f"reflection_n_rays={diagnostics['summary'].reflection_n_rays}, "
        f"reflection_max_bounces={diagnostics['summary'].reflection_max_bounces}, "
        f"samples_per_tx={diagnostics['summary'].samples_per_tx}, "
        f"max_diffractions=1",
        fontsize=13,
    )
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def save_compare_scene_arrays(diagnostics, *, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    compare_scene = diagnostics["compare_scene"]
    np.savez_compressed(
        output_path,
        witwin_total=compare_scene["witwin"]["total"],
        witwin_diff_increment=compare_scene["witwin"]["diff_increment"],
        sionna_total=compare_scene["sionna"]["total"],
        sionna_diff_increment=compare_scene["sionna"]["diff_increment"],
        witwin_total_db=compare_scene["witwin"]["total_db"],
        witwin_diff_db=compare_scene["witwin"]["diff_db"],
        sionna_total_db=compare_scene["sionna"]["total_db"],
        sionna_diff_db=compare_scene["sionna"]["diff_db"],
        total_delta_db=compare_scene["delta"]["total_db"],
        diff_delta_db=compare_scene["delta"]["diff_db"],
    )
    return output_path


def save_diagnostic_arrays(diagnostics, *, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        forward_noad_total=diagnostics["forward_noad"]["total"],
        forward_noad_diff_increment=diagnostics["forward_noad"]["diff_increment"],
        forward_ad_total=diagnostics["forward_ad"]["total"],
        forward_ad_diff_increment=diagnostics["forward_ad"]["diff_increment"],
        total_delta_db=diagnostics["forward_ad"]["total_db"] - diagnostics["forward_noad"]["total_db"],
        diff_increment_delta_db=diagnostics["forward_ad"]["diff_db"] - diagnostics["forward_noad"]["diff_db"],
        tx_x_ad=diagnostics["parameter_results"]["tx_x"]["ad"],
        tx_x_fd=diagnostics["parameter_results"]["tx_x"]["fd"],
        cube1_x_ad=diagnostics["parameter_results"]["cube1_x"]["ad"],
        cube1_x_fd=diagnostics["parameter_results"]["cube1_x"]["fd"],
    )
    return output_path


def save_summary_json(diagnostics, *, output_path: Path, paths: dict[str, str]) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary = asdict(diagnostics["summary"])
    summary["paths"] = paths
    output_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return output_path


def save_compare_scene_diagnostics(
    output_prefix: Path,
    *,
    grid_size: int = DEFAULT_GRID_SIZE,
    samples_per_tx: int = DEFAULT_SAMPLES_PER_TX,
    fd_step: float = DEFAULT_FD_STEP,
    plane_z: float = DEFAULT_PLANE_Z,
    tx_pos: tuple[float, float, float] = DEFAULT_TX_POS,
    bounds=DEFAULT_BOUNDS,
    seed: int = DEFAULT_SEED,
    warmup: int = DEFAULT_WARMUP,
    repeats: int = DEFAULT_REPEATS,
):
    diagnostics = build_diagnostics(
        grid_size=int(grid_size),
        bounds=bounds,
        plane_z=float(plane_z),
        tx_pos=tuple(float(value) for value in tx_pos),
        samples_per_tx=int(samples_per_tx),
        fd_step=float(fd_step),
        seed=int(seed),
        warmup=int(warmup),
        repeats=int(repeats),
    )

    compare_figure_path = output_prefix.with_name(output_prefix.name + "_compare.png")
    compare_arrays_path = output_prefix.with_name(output_prefix.name + "_compare.npz")
    forward_consistency_path = output_prefix.with_name(output_prefix.name + "_forward_consistency.png")
    gradient_path = output_prefix.with_name(output_prefix.name + "_gradients.png")
    arrays_path = output_prefix.with_suffix(".npz")
    json_path = output_prefix.with_suffix(".json")

    save_compare_scene_figure(diagnostics, output_path=compare_figure_path)
    save_compare_scene_arrays(diagnostics, output_path=compare_arrays_path)
    save_forward_consistency_figure(diagnostics, output_path=forward_consistency_path)
    save_gradient_figure(diagnostics, output_path=gradient_path)
    save_diagnostic_arrays(diagnostics, output_path=arrays_path)
    save_summary_json(
        diagnostics,
        output_path=json_path,
        paths={
            "compare_figure": str(compare_figure_path),
            "compare_arrays": str(compare_arrays_path),
            "forward_consistency_figure": str(forward_consistency_path),
            "gradient_figure": str(gradient_path),
            "diagnostic_arrays": str(arrays_path),
        },
    )
    return compare_figure_path, forward_consistency_path, gradient_path, arrays_path, json_path


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid-size", type=int, default=DEFAULT_GRID_SIZE)
    parser.add_argument("--samples-per-tx", type=int, default=DEFAULT_SAMPLES_PER_TX)
    parser.add_argument("--fd-step", type=float, default=DEFAULT_FD_STEP)
    parser.add_argument("--warmup", type=int, default=DEFAULT_WARMUP)
    parser.add_argument("--repeats", type=int, default=DEFAULT_REPEATS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--plane-z", type=float, default=DEFAULT_PLANE_Z)
    parser.add_argument("--tx-x", type=float, default=DEFAULT_TX_POS[0])
    parser.add_argument("--tx-y", type=float, default=DEFAULT_TX_POS[1])
    parser.add_argument("--tx-z", type=float, default=DEFAULT_TX_POS[2])
    parser.add_argument("--xmin", type=float, default=DEFAULT_BOUNDS[0][0])
    parser.add_argument("--xmax", type=float, default=DEFAULT_BOUNDS[0][1])
    parser.add_argument("--ymin", type=float, default=DEFAULT_BOUNDS[1][0])
    parser.add_argument("--ymax", type=float, default=DEFAULT_BOUNDS[1][1])
    parser.add_argument("--output-prefix", type=Path, default=DEFAULT_OUTPUT_PREFIX)
    return parser.parse_args()


def main():
    args = parse_args()
    bounds = ((float(args.xmin), float(args.xmax)), (float(args.ymin), float(args.ymax)))
    tx_pos = (float(args.tx_x), float(args.tx_y), float(args.tx_z))
    outputs = save_compare_scene_diagnostics(
        args.output_prefix,
        grid_size=int(args.grid_size),
        samples_per_tx=int(args.samples_per_tx),
        fd_step=float(args.fd_step),
        plane_z=float(args.plane_z),
        tx_pos=tx_pos,
        bounds=bounds,
        seed=int(args.seed),
        warmup=int(args.warmup),
        repeats=int(args.repeats),
    )
    print(
        json.dumps(
            {
                "compare_figure": str(outputs[0]),
                "forward_consistency_figure": str(outputs[1]),
                "gradient_figure": str(outputs[2]),
                "arrays": str(outputs[3]),
                "json": str(outputs[4]),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()


__all__ = [
    "DEFAULT_OUTPUT_PREFIX",
    "build_diagnostics",
    "save_compare_scene_diagnostics",
    "save_forward_consistency_figure",
    "save_gradient_figure",
]
