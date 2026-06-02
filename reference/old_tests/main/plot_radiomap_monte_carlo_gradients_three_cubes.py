"""Three-cube Monte Carlo radio-map forward-gradient benchmark helper."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
import os
from pathlib import Path
from time import perf_counter

import drjit as dr
import numpy as np
import witwin as wt
if os.environ.get("WITWIN_CHANNEL_MAIN_SHOW", "0") != "1":
    os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt

from tests.main.plot_multipath_components import (
    CUBE1_BASE_CENTER,
    build_scene_for_cube1_x,
    gradient_db_magnitude,
)
from tests.main.plot_radiomap_gradients_three_cubes import (
    _as_grid,
    _format_seconds,
    _loss_weight_probe,
    _path_gain_db,
    _scalar_from_drjit,
    _time_call,
    parameter_config,
)
from tests.main.plot_radiomap_sionna_three_cubes import (
    DEFAULT_BOUNDS,
    DEFAULT_PLANE_Z,
    DEFAULT_TX_POS,
    _decorate_axis,
    _output_dir,
)
from witwin.channel import RadioMapMonitor, Tracer
DEFAULT_GRID_SIZE = 128
DEFAULT_REFLECTION_N_RAYS = 512
DEFAULT_SAMPLES_PER_TX = 128
DEFAULT_FD_STEP = 1.0e-3
DEFAULT_COMBINE_MODE = "incoherent"
DEFAULT_RECEIVER_MODEL = "matched_isotropic"
DEFAULT_SHADOW_BOUNDARY_MODE = "none"
DEFAULT_ACCUMULATION_BACKEND = "auto"
DEFAULT_MAX_DIFFRACTIONS = 1
DEFAULT_OUTPUT_PREFIX = _output_dir() / "radiomap_monte_carlo_three_cubes_gradients"
DEFAULT_TRACE_SEED = 7
_GRAD_FLAGS = dr.ADFlag.Default | dr.ADFlag.AllowNoGrad


@dataclass(frozen=True)
class MonteCarloGradientSummary:
    grid_size: int
    bounds: tuple[tuple[float, float], tuple[float, float]]
    plane_z: float
    tx_pos: tuple[float, float, float]
    reflection_n_rays: int
    samples_per_tx: int
    fd_step: float
    combine_mode: str
    receiver_model: str
    shadow_boundary_mode: str
    accumulation_backend_requested: str
    accumulation_backend_resolved: str
    monte_carlo_ad_mode: bool
    monte_carlo_ad_backend: str
    monte_carlo_tape_layout_version: str
    max_diffractions: int
    path_counts: dict
    runtime_backends: dict
    parameter_backends: dict
    timings_seconds: dict


def _make_monitor(
    *,
    grid_size: int,
    bounds,
    plane_z: float,
    combine_mode: str,
    receiver_model: str,
    shadow_boundary_mode: str,
    accumulation_backend: str,
    max_diffractions: int,
    samples_per_tx: int,
):
    return RadioMapMonitor(
        "three_cubes_gradient_rm_mc",
        axis="z",
        position=float(plane_z),
        bounds=bounds,
        grid_shape=(int(grid_size), int(grid_size)),
        metric="path_gain",
        combine_mode=str(combine_mode),
        receiver_model=str(receiver_model),
        accumulation_backend=str(accumulation_backend),
        sampling_mode="monte_carlo",
        ad=True,
        samples_per_tx=int(samples_per_tx),
        ray_mode="3d",
        quadrature_mode="center",
        max_diffractions=int(max_diffractions),
        shadow_boundary_mode=str(shadow_boundary_mode),
        seed=int(DEFAULT_TRACE_SEED),
    )


def _make_tracer(scene, *, reflection_n_rays: int, max_diffractions: int):
    return Tracer(
        frequency=1.0e9,
        scene=scene,
        reflection_n_rays=int(reflection_n_rays),
        reflection_max_bounces=1,
        reflection_coef=0.8,
        enable_rd_diffraction=True,
        max_diffractions=int(max_diffractions),
    )


def _trace_path_gain_payload(
    *,
    cube1_x,
    tx_pos,
    grid_size: int,
    reflection_n_rays: int,
    samples_per_tx: int,
    bounds,
    plane_z: float,
    combine_mode: str,
    receiver_model: str,
    shadow_boundary_mode: str,
    accumulation_backend: str,
    max_diffractions: int,
):
    try:
        dr.seed(int(DEFAULT_TRACE_SEED))
    except Exception:
        pass
    try:
        wt.register_sampler_seed(int(DEFAULT_TRACE_SEED))
    except Exception:
        pass
    scene = build_scene_for_cube1_x(cube1_x)
    tracer = _make_tracer(
        scene,
        reflection_n_rays=reflection_n_rays,
        max_diffractions=max_diffractions,
    )
    monitor = _make_monitor(
        grid_size=grid_size,
        bounds=bounds,
        plane_z=plane_z,
        combine_mode=combine_mode,
        receiver_model=receiver_model,
        shadow_boundary_mode=shadow_boundary_mode,
        accumulation_backend=accumulation_backend,
        max_diffractions=max_diffractions,
        samples_per_tx=samples_per_tx,
    )
    trace_output = tracer.trace(tx_pos, monitor=monitor, verbose=False)
    result = trace_output.monitor(monitor.name) if hasattr(trace_output, "monitor") else trace_output
    return {
        "coords": {
            "grid_x": result.coords.grid_x,
            "grid_y": result.coords.grid_y,
            "x": result.coords.x,
            "y": result.coords.y,
        },
        "metrics": {
            "path_gain": result.path_gain,
            "los": result.incoherent["los"],
            "reflection": result.incoherent["reflection"],
            "diffraction": result.incoherent["diffraction"],
        },
        "metadata": result.metadata,
    }


def _trace_path_gain(
    *,
    cube1_x,
    tx_pos,
    grid_size: int,
    reflection_n_rays: int,
    samples_per_tx: int,
    bounds,
    plane_z: float,
    combine_mode: str,
    receiver_model: str,
    shadow_boundary_mode: str,
    accumulation_backend: str,
    max_diffractions: int,
):
    payload = _trace_path_gain_payload(
        cube1_x=cube1_x,
        tx_pos=tx_pos,
        grid_size=grid_size,
        reflection_n_rays=reflection_n_rays,
        samples_per_tx=samples_per_tx,
        bounds=bounds,
        plane_z=plane_z,
        combine_mode=combine_mode,
        receiver_model=receiver_model,
        shadow_boundary_mode=shadow_boundary_mode,
        accumulation_backend=accumulation_backend,
        max_diffractions=max_diffractions,
    )
    return (
        np.asarray(payload["coords"]["grid_x"], dtype=np.float64),
        np.asarray(payload["coords"]["grid_y"], dtype=np.float64),
        np.asarray(payload["metrics"]["path_gain"], dtype=np.float64),
        payload,
    )


def ad_gradient_path_gain(
    parameter: str,
    *,
    tx_pos,
    grid_size: int,
    reflection_n_rays: int,
    samples_per_tx: int,
    bounds,
    plane_z: float,
    combine_mode: str,
    receiver_model: str,
    shadow_boundary_mode: str,
    accumulation_backend: str,
    max_diffractions: int,
):
    config = parameter_config(parameter, tx_pos=tx_pos)
    def _build_payload():
        if parameter == "cube1_x":
            cube1_x = wt.Float(config["cube1_x"])
            dr.enable_grad(cube1_x)
            payload = _trace_path_gain_payload(
                cube1_x=cube1_x,
                tx_pos=wt.Point3f(*config["tx_pos"]),
                grid_size=grid_size,
                reflection_n_rays=reflection_n_rays,
                samples_per_tx=samples_per_tx,
                bounds=bounds,
                plane_z=plane_z,
                combine_mode=combine_mode,
                receiver_model=receiver_model,
                shadow_boundary_mode=shadow_boundary_mode,
                accumulation_backend=accumulation_backend,
                max_diffractions=max_diffractions,
            )
            return payload, cube1_x
        tx_x = wt.Float(config["tx_pos"][0])
        dr.enable_grad(tx_x)
        payload = _trace_path_gain_payload(
            cube1_x=config["cube1_x"],
            tx_pos=wt.Point3f(tx_x, config["tx_pos"][1], config["tx_pos"][2]),
            grid_size=grid_size,
            reflection_n_rays=reflection_n_rays,
            samples_per_tx=samples_per_tx,
            bounds=bounds,
            plane_z=plane_z,
            combine_mode=combine_mode,
            receiver_model=receiver_model,
            shadow_boundary_mode=shadow_boundary_mode,
            accumulation_backend=accumulation_backend,
            max_diffractions=max_diffractions,
        )
        return payload, tx_x

    payload, parameter_value = _build_payload()
    dr.set_grad(parameter_value, 1.0)
    path_gain_grad = dr.forward_to(
        payload["metrics"]["path_gain"],
        flags=_GRAD_FLAGS,
    )
    component_payload, parameter_value = _build_payload()
    dr.set_grad(parameter_value, 1.0)
    los_grad, reflection_grad, diffraction_grad = dr.forward_to(
        component_payload["metrics"]["los"],
        component_payload["metrics"]["reflection"],
        component_payload["metrics"]["diffraction"],
        flags=_GRAD_FLAGS,
    )
    return (
        payload["coords"]["grid_x"],
        payload["coords"]["grid_y"],
        {
            "path_gain": np.asarray(path_gain_grad, dtype=np.float64),
            "los": np.asarray(los_grad, dtype=np.float64),
            "reflection": np.asarray(reflection_grad, dtype=np.float64),
            "diffraction": np.asarray(diffraction_grad, dtype=np.float64),
        },
        payload,
    )


def fd_gradient_path_gain(
    parameter: str,
    *,
    tx_pos,
    grid_size: int,
    reflection_n_rays: int,
    samples_per_tx: int,
    fd_step: float,
    bounds,
    plane_z: float,
    combine_mode: str,
    receiver_model: str,
    shadow_boundary_mode: str,
    accumulation_backend: str,
    max_diffractions: int,
):
    config = parameter_config(parameter, tx_pos=tx_pos)
    plus_cfg, minus_cfg = config["perturb"](fd_step)
    _, _, _, plus_payload = _trace_path_gain(
        cube1_x=plus_cfg["cube1_x"],
        tx_pos=wt.Point3f(*plus_cfg["tx_pos"]),
        grid_size=grid_size,
        reflection_n_rays=reflection_n_rays,
        samples_per_tx=samples_per_tx,
        bounds=bounds,
        plane_z=plane_z,
        combine_mode=combine_mode,
        receiver_model=receiver_model,
        shadow_boundary_mode=shadow_boundary_mode,
        accumulation_backend=accumulation_backend,
        max_diffractions=max_diffractions,
    )
    _, _, _, minus_payload = _trace_path_gain(
        cube1_x=minus_cfg["cube1_x"],
        tx_pos=wt.Point3f(*minus_cfg["tx_pos"]),
        grid_size=grid_size,
        reflection_n_rays=reflection_n_rays,
        samples_per_tx=samples_per_tx,
        bounds=bounds,
        plane_z=plane_z,
        combine_mode=combine_mode,
        receiver_model=receiver_model,
        shadow_boundary_mode=shadow_boundary_mode,
        accumulation_backend=accumulation_backend,
        max_diffractions=max_diffractions,
    )
    return {
        component: np.asarray(
            (
                np.asarray(plus_payload["metrics"][component], dtype=np.float64)
                - np.asarray(minus_payload["metrics"][component], dtype=np.float64)
            )
            / (2.0 * float(fd_step)),
            dtype=np.float64,
        )
        for component in ("path_gain", "los", "reflection", "diffraction")
    }


def jvp_scalar_loss(
    parameter: str,
    *,
    tx_pos,
    grid_size: int,
    reflection_n_rays: int,
    samples_per_tx: int,
    bounds,
    plane_z: float,
    combine_mode: str,
    receiver_model: str,
    shadow_boundary_mode: str,
    accumulation_backend: str,
    max_diffractions: int,
):
    config = parameter_config(parameter, tx_pos=tx_pos)
    weights = None
    if parameter == "cube1_x":
        cube1_x = wt.Float(config["cube1_x"])
        dr.enable_grad(cube1_x)
        payload = _trace_path_gain_payload(
            cube1_x=cube1_x,
            tx_pos=wt.Point3f(*config["tx_pos"]),
            grid_size=grid_size,
            reflection_n_rays=reflection_n_rays,
            samples_per_tx=samples_per_tx,
            bounds=bounds,
            plane_z=plane_z,
            combine_mode=combine_mode,
            receiver_model=receiver_model,
            shadow_boundary_mode=shadow_boundary_mode,
            accumulation_backend=accumulation_backend,
            max_diffractions=max_diffractions,
        )
        weights = type(payload["metrics"]["path_gain"])(_loss_weight_probe(grid_size))
        loss = dr.sum(payload["metrics"]["path_gain"] * weights)
        dr.set_grad(cube1_x, 1.0)
        dr.forward_to(loss, flags=_GRAD_FLAGS)
        return _scalar_from_drjit(dr.grad(loss))

    tx_x = wt.Float(config["tx_pos"][0])
    dr.enable_grad(tx_x)
    payload = _trace_path_gain_payload(
        cube1_x=config["cube1_x"],
        tx_pos=wt.Point3f(tx_x, config["tx_pos"][1], config["tx_pos"][2]),
        grid_size=grid_size,
        reflection_n_rays=reflection_n_rays,
        samples_per_tx=samples_per_tx,
        bounds=bounds,
        plane_z=plane_z,
        combine_mode=combine_mode,
        receiver_model=receiver_model,
        shadow_boundary_mode=shadow_boundary_mode,
        accumulation_backend=accumulation_backend,
        max_diffractions=max_diffractions,
    )
    weights = type(payload["metrics"]["path_gain"])(_loss_weight_probe(grid_size))
    loss = dr.sum(payload["metrics"]["path_gain"] * weights)
    dr.set_grad(tx_x, 1.0)
    dr.forward_to(loss, flags=_GRAD_FLAGS)
    return _scalar_from_drjit(dr.grad(loss))


def vjp_scalar_loss(
    parameter: str,
    *,
    tx_pos,
    grid_size: int,
    reflection_n_rays: int,
    samples_per_tx: int,
    bounds,
    plane_z: float,
    combine_mode: str,
    receiver_model: str,
    shadow_boundary_mode: str,
    accumulation_backend: str,
    max_diffractions: int,
):
    gradient, _ = vjp_scalar_loss_with_backward_timing(
        parameter,
        tx_pos=tx_pos,
        grid_size=grid_size,
        reflection_n_rays=reflection_n_rays,
        samples_per_tx=samples_per_tx,
        bounds=bounds,
        plane_z=plane_z,
        combine_mode=combine_mode,
        receiver_model=receiver_model,
        shadow_boundary_mode=shadow_boundary_mode,
        accumulation_backend=accumulation_backend,
        max_diffractions=max_diffractions,
    )
    return gradient


def vjp_scalar_loss_with_backward_timing(
    parameter: str,
    *,
    tx_pos,
    grid_size: int,
    reflection_n_rays: int,
    samples_per_tx: int,
    bounds,
    plane_z: float,
    combine_mode: str,
    receiver_model: str,
    shadow_boundary_mode: str,
    accumulation_backend: str,
    max_diffractions: int,
):
    config = parameter_config(parameter, tx_pos=tx_pos)
    if parameter == "cube1_x":
        cube1_x = wt.Float(config["cube1_x"])
        dr.enable_grad(cube1_x)
        payload = _trace_path_gain_payload(
            cube1_x=cube1_x,
            tx_pos=wt.Point3f(*config["tx_pos"]),
            grid_size=grid_size,
            reflection_n_rays=reflection_n_rays,
            samples_per_tx=samples_per_tx,
            bounds=bounds,
            plane_z=plane_z,
            combine_mode=combine_mode,
            receiver_model=receiver_model,
            shadow_boundary_mode=shadow_boundary_mode,
            accumulation_backend=accumulation_backend,
            max_diffractions=max_diffractions,
        )
        loss = dr.sum(
            payload["metrics"]["path_gain"]
            * type(payload["metrics"]["path_gain"])(_loss_weight_probe(grid_size))
        )
        dr.eval(loss)
        dr.sync_thread()
        backward_start = perf_counter()
        dr.backward(loss, flags=_GRAD_FLAGS)
        cube1_x_grad = dr.grad(cube1_x)
        dr.eval(cube1_x_grad)
        dr.sync_thread()
        return _scalar_from_drjit(cube1_x_grad), float(perf_counter() - backward_start)

    tx_x = wt.Float(config["tx_pos"][0])
    dr.enable_grad(tx_x)
    payload = _trace_path_gain_payload(
        cube1_x=config["cube1_x"],
        tx_pos=wt.Point3f(tx_x, config["tx_pos"][1], config["tx_pos"][2]),
        grid_size=grid_size,
        reflection_n_rays=reflection_n_rays,
        samples_per_tx=samples_per_tx,
        bounds=bounds,
        plane_z=plane_z,
        combine_mode=combine_mode,
        receiver_model=receiver_model,
        shadow_boundary_mode=shadow_boundary_mode,
        accumulation_backend=accumulation_backend,
        max_diffractions=max_diffractions,
    )
    loss = dr.sum(
        payload["metrics"]["path_gain"]
        * type(payload["metrics"]["path_gain"])(_loss_weight_probe(grid_size))
    )
    dr.eval(loss)
    dr.sync_thread()
    backward_start = perf_counter()
    dr.backward(loss, flags=_GRAD_FLAGS)
    tx_x_grad = dr.grad(tx_x)
    dr.eval(tx_x_grad)
    dr.sync_thread()
    return _scalar_from_drjit(tx_x_grad), float(perf_counter() - backward_start)


def fd_scalar_loss(
    parameter: str,
    *,
    tx_pos,
    grid_size: int,
    reflection_n_rays: int,
    samples_per_tx: int,
    fd_step: float,
    bounds,
    plane_z: float,
    combine_mode: str,
    receiver_model: str,
    shadow_boundary_mode: str,
    accumulation_backend: str,
    max_diffractions: int,
):
    weights = np.asarray(_loss_weight_probe(grid_size), dtype=np.float64)
    config = parameter_config(parameter, tx_pos=tx_pos)
    plus_cfg, minus_cfg = config["perturb"](fd_step)
    _, _, plus_path_gain, _ = _trace_path_gain(
        cube1_x=plus_cfg["cube1_x"],
        tx_pos=wt.Point3f(*plus_cfg["tx_pos"]),
        grid_size=grid_size,
        reflection_n_rays=reflection_n_rays,
        samples_per_tx=samples_per_tx,
        bounds=bounds,
        plane_z=plane_z,
        combine_mode=combine_mode,
        receiver_model=receiver_model,
        shadow_boundary_mode=shadow_boundary_mode,
        accumulation_backend=accumulation_backend,
        max_diffractions=max_diffractions,
    )
    _, _, minus_path_gain, _ = _trace_path_gain(
        cube1_x=minus_cfg["cube1_x"],
        tx_pos=wt.Point3f(*minus_cfg["tx_pos"]),
        grid_size=grid_size,
        reflection_n_rays=reflection_n_rays,
        samples_per_tx=samples_per_tx,
        bounds=bounds,
        plane_z=plane_z,
        combine_mode=combine_mode,
        receiver_model=receiver_model,
        shadow_boundary_mode=shadow_boundary_mode,
        accumulation_backend=accumulation_backend,
        max_diffractions=max_diffractions,
    )
    plus_loss = float(np.sum(np.asarray(plus_path_gain, dtype=np.float64) * weights))
    minus_loss = float(np.sum(np.asarray(minus_path_gain, dtype=np.float64) * weights))
    return (plus_loss - minus_loss) / (2.0 * float(fd_step))


def build_gradient_benchmark(
    *,
    grid_size: int,
    reflection_n_rays: int,
    samples_per_tx: int,
    fd_step: float,
    bounds,
    plane_z: float,
    tx_pos,
    combine_mode: str,
    receiver_model: str,
    shadow_boundary_mode: str,
    accumulation_backend: str,
    max_diffractions: int,
):
    (_, _, total_path_gain, total_payload), total_forward_seconds = _time_call(
        _trace_path_gain,
        cube1_x=CUBE1_BASE_CENTER[0],
        tx_pos=wt.Point3f(*tx_pos),
        grid_size=grid_size,
        reflection_n_rays=reflection_n_rays,
        samples_per_tx=samples_per_tx,
        bounds=bounds,
        plane_z=plane_z,
        combine_mode=combine_mode,
        receiver_model=receiver_model,
        shadow_boundary_mode=shadow_boundary_mode,
        accumulation_backend=accumulation_backend,
        max_diffractions=max_diffractions,
    )
    total_db = _path_gain_db(_as_grid(total_path_gain, grid_size))
    parameter_results = {}
    parameter_backends = {}
    timings_seconds = {"forward_total": float(total_forward_seconds)}

    for parameter in ("tx_x", "cube1_x"):
        (_, _, ad_gradient_payload, ad_payload), ad_seconds = _time_call(
            ad_gradient_path_gain,
            parameter,
            tx_pos=tx_pos,
            grid_size=grid_size,
            reflection_n_rays=reflection_n_rays,
            samples_per_tx=samples_per_tx,
            bounds=bounds,
            plane_z=plane_z,
            combine_mode=combine_mode,
            receiver_model=receiver_model,
            shadow_boundary_mode=shadow_boundary_mode,
            accumulation_backend=accumulation_backend,
            max_diffractions=max_diffractions,
        )
        fd_gradient_payload, fd_seconds = _time_call(
            fd_gradient_path_gain,
            parameter,
            tx_pos=tx_pos,
            grid_size=grid_size,
            reflection_n_rays=reflection_n_rays,
            samples_per_tx=samples_per_tx,
            fd_step=fd_step,
            bounds=bounds,
            plane_z=plane_z,
            combine_mode=combine_mode,
            receiver_model=receiver_model,
            shadow_boundary_mode=shadow_boundary_mode,
            accumulation_backend=accumulation_backend,
            max_diffractions=max_diffractions,
        )
        ad_grid = _as_grid(ad_gradient_payload["path_gain"], grid_size)
        fd_grid = _as_grid(fd_gradient_payload["path_gain"], grid_size)
        scalar_probe = _loss_weight_probe(grid_size).astype(np.float64)
        scalar_jvp_start = perf_counter()
        scalar_jvp = float(np.sum(ad_grid * scalar_probe))
        jvp_seconds = float(perf_counter() - scalar_jvp_start)
        (scalar_vjp, backward_only_seconds), vjp_seconds = _time_call(
            vjp_scalar_loss_with_backward_timing,
            parameter,
            tx_pos=tx_pos,
            grid_size=grid_size,
            reflection_n_rays=reflection_n_rays,
            samples_per_tx=samples_per_tx,
            bounds=bounds,
            plane_z=plane_z,
            combine_mode=combine_mode,
            receiver_model=receiver_model,
            shadow_boundary_mode=shadow_boundary_mode,
            accumulation_backend=accumulation_backend,
            max_diffractions=max_diffractions,
        )
        scalar_fd_start = perf_counter()
        scalar_fd = float(np.sum(fd_grid * scalar_probe))
        scalar_fd_seconds = float(perf_counter() - scalar_fd_start)
        ad_vis = gradient_db_magnitude(ad_grid)
        fd_vis = gradient_db_magnitude(fd_grid)
        diff_vis = ad_vis - fd_vis
        parameter_results[parameter] = {
            "ad": ad_grid,
            "fd": fd_grid,
            "ad_vis": ad_vis,
            "fd_vis": fd_vis,
            "diff_vis": diff_vis,
            "component_grids": {
                component: {
                    "ad": _as_grid(ad_gradient_payload[component], grid_size),
                    "fd": _as_grid(fd_gradient_payload[component], grid_size),
                }
                for component in ("los", "reflection", "diffraction")
            },
            "component_grad_abs_sums": {
                "los": float(np.sum(np.abs(ad_gradient_payload["los"]))),
                "reflection": float(np.sum(np.abs(ad_gradient_payload["reflection"]))),
                "diffraction": float(np.sum(np.abs(ad_gradient_payload["diffraction"]))),
            },
            "scalar_jvp": float(scalar_jvp),
            "scalar_vjp": float(scalar_vjp),
            "scalar_fd": float(scalar_fd),
            "timings_seconds": {
                "ad": float(ad_seconds),
                "fd": float(fd_seconds),
                "jvp": float(jvp_seconds),
                "vjp": float(vjp_seconds),
                "backward_only": float(backward_only_seconds),
                "scalar_fd": float(scalar_fd_seconds),
            },
        }
        parameter_backends[parameter] = dict(ad_payload["metadata"]["accumulation_backend"])
        timings_seconds[parameter] = dict(parameter_results[parameter]["timings_seconds"])

    summary = MonteCarloGradientSummary(
        grid_size=int(grid_size),
        bounds=tuple((float(a), float(b)) for (a, b) in bounds),
        plane_z=float(plane_z),
        tx_pos=tuple(float(value) for value in tx_pos),
        reflection_n_rays=int(reflection_n_rays),
        samples_per_tx=int(samples_per_tx),
        fd_step=float(fd_step),
        combine_mode=str(combine_mode),
        receiver_model=str(receiver_model),
        shadow_boundary_mode=str(shadow_boundary_mode),
        accumulation_backend_requested=str(accumulation_backend),
        accumulation_backend_resolved=str(
            total_payload["metadata"]["accumulation_backend"]["resolved"]
        ),
        monte_carlo_ad_mode=bool(total_payload["metadata"]["monte_carlo"]["ad_mode"]),
        monte_carlo_ad_backend=str(total_payload["metadata"]["monte_carlo"]["ad_backend"]),
        monte_carlo_tape_layout_version=str(
            total_payload["metadata"]["monte_carlo"]["tape_layout_version"]
        ),
        max_diffractions=int(max_diffractions),
        path_counts=dict(total_payload["metadata"].get("path_counts", {})),
        runtime_backends=dict(total_payload["metadata"].get("runtime_backends", {})),
        parameter_backends=parameter_backends,
        timings_seconds=timings_seconds,
    )
    return {
        "summary": summary,
        "grid_x": _as_grid(total_payload["coords"]["grid_x"], grid_size),
        "grid_y": _as_grid(total_payload["coords"]["grid_y"], grid_size),
        "total_db": total_db,
        "parameter_results": parameter_results,
        "cube1_x": float(CUBE1_BASE_CENTER[0]),
    }


def save_figure(gradient_benchmark, *, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 4, figsize=(16.0, 8.8), constrained_layout=True)
    extent = (
        float(gradient_benchmark["summary"].bounds[0][0]),
        float(gradient_benchmark["summary"].bounds[0][1]),
        float(gradient_benchmark["summary"].bounds[1][0]),
        float(gradient_benchmark["summary"].bounds[1][1]),
    )
    tx_pos = gradient_benchmark["summary"].tx_pos
    cube1_x = gradient_benchmark["cube1_x"]

    for row, parameter in enumerate(("tx_x", "cube1_x")):
        row_results = gradient_benchmark["parameter_results"][parameter]
        row_timings = row_results["timings_seconds"]
        diff_vmax = max(float(np.percentile(np.abs(row_results["diff_vis"]), 99.0)), 1.0)
        panels = (
            (
                axes[row, 0],
                gradient_benchmark["total_db"],
                (
                    f"MC Path Gain (dB), {parameter}\n"
                    f"fwd={_format_seconds(gradient_benchmark['summary'].timings_seconds['forward_total'])} | "
                    f"AD={_format_seconds(row_timings['ad'])} | "
                    f"FD={_format_seconds(row_timings['fd'])}"
                ),
                "viridis",
                float(np.nanpercentile(gradient_benchmark["total_db"], 1.0)),
                float(np.nanpercentile(gradient_benchmark["total_db"], 99.0)),
            ),
            (
                axes[row, 1],
                row_results["ad_vis"],
                f"AD |d path_gain / d {parameter}| (dB)\nprobe-JVP={_format_seconds(row_timings['jvp'])}",
                "magma",
                float(np.nanpercentile(row_results["ad_vis"], 5.0)),
                float(np.nanpercentile(row_results["ad_vis"], 99.0)),
            ),
            (
                axes[row, 2],
                row_results["fd_vis"],
                (
                    f"FD |d path_gain / d {parameter}| (dB)\n"
                    f"probe-FD={_format_seconds(row_timings['scalar_fd'])}"
                ),
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
            _decorate_axis(
                ax,
                bounds=gradient_benchmark["summary"].bounds,
                cube1_x=cube1_x,
                tx_pos=tx_pos,
            )
            ax.set_title(title, fontsize=10)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)

    fig.suptitle(
        (
            "Three-Cube Monte Carlo Radiomap Single-Solver AD Gradients\n"
            f"grid={gradient_benchmark['summary'].grid_size}x{gradient_benchmark['summary'].grid_size}, "
            f"xy slice z={gradient_benchmark['summary'].plane_z:.1f}, "
            f"tx=({tx_pos[0]:.1f}, {tx_pos[1]:.1f}, {tx_pos[2]:.1f})"
        ),
        fontsize=13,
    )
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def save_component_figure(gradient_benchmark, *, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 9, figsize=(28.0, 8.8), constrained_layout=True)
    extent = (
        float(gradient_benchmark["summary"].bounds[0][0]),
        float(gradient_benchmark["summary"].bounds[0][1]),
        float(gradient_benchmark["summary"].bounds[1][0]),
        float(gradient_benchmark["summary"].bounds[1][1]),
    )
    tx_pos = gradient_benchmark["summary"].tx_pos
    cube1_x = gradient_benchmark["cube1_x"]

    for row, parameter in enumerate(("tx_x", "cube1_x")):
        row_results = gradient_benchmark["parameter_results"][parameter]
        for component_index, component in enumerate(("los", "reflection", "diffraction")):
            ad_grid = row_results["component_grids"][component]["ad"]
            fd_grid = row_results["component_grids"][component]["fd"]
            diff_grid = gradient_db_magnitude(ad_grid) - gradient_db_magnitude(fd_grid)
            diff_vmax = max(float(np.nanpercentile(np.abs(diff_grid), 99.0)), 1.0)
            col_base = 3 * component_index
            panels = (
                (
                    axes[row, col_base + 0],
                    gradient_db_magnitude(ad_grid),
                    f"{component} AD, {parameter}",
                    "magma",
                    float(np.nanpercentile(gradient_db_magnitude(ad_grid), 5.0)),
                    float(np.nanpercentile(gradient_db_magnitude(ad_grid), 99.0)),
                ),
                (
                    axes[row, col_base + 1],
                    gradient_db_magnitude(fd_grid),
                    f"{component} FD, {parameter}",
                    "magma",
                    float(np.nanpercentile(gradient_db_magnitude(fd_grid), 5.0)),
                    float(np.nanpercentile(gradient_db_magnitude(fd_grid), 99.0)),
                ),
                (
                    axes[row, col_base + 2],
                    diff_grid,
                    f"{component} AD-FD, {parameter}",
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
                _decorate_axis(
                    ax,
                    bounds=gradient_benchmark["summary"].bounds,
                    cube1_x=cube1_x,
                    tx_pos=tx_pos,
                )
                ax.set_title(title, fontsize=9)
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)

    fig.suptitle(
        (
            "Three-Cube Monte Carlo Component Gradients\n"
            f"grid={gradient_benchmark['summary'].grid_size}x{gradient_benchmark['summary'].grid_size}, "
            f"samples_per_tx={gradient_benchmark['summary'].samples_per_tx}, "
            f"reflection_rays={gradient_benchmark['summary'].reflection_n_rays}"
        ),
        fontsize=13,
    )
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def save_arrays(gradient_benchmark, *, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        total_db=gradient_benchmark["total_db"],
        tx_x_ad=gradient_benchmark["parameter_results"]["tx_x"]["ad"],
        tx_x_fd=gradient_benchmark["parameter_results"]["tx_x"]["fd"],
        tx_x_diff_vis=gradient_benchmark["parameter_results"]["tx_x"]["diff_vis"],
        tx_x_los_ad=gradient_benchmark["parameter_results"]["tx_x"]["component_grids"]["los"]["ad"],
        tx_x_los_fd=gradient_benchmark["parameter_results"]["tx_x"]["component_grids"]["los"]["fd"],
        tx_x_reflection_ad=gradient_benchmark["parameter_results"]["tx_x"]["component_grids"]["reflection"]["ad"],
        tx_x_reflection_fd=gradient_benchmark["parameter_results"]["tx_x"]["component_grids"]["reflection"]["fd"],
        tx_x_diffraction_ad=gradient_benchmark["parameter_results"]["tx_x"]["component_grids"]["diffraction"]["ad"],
        tx_x_diffraction_fd=gradient_benchmark["parameter_results"]["tx_x"]["component_grids"]["diffraction"]["fd"],
        cube1_x_ad=gradient_benchmark["parameter_results"]["cube1_x"]["ad"],
        cube1_x_fd=gradient_benchmark["parameter_results"]["cube1_x"]["fd"],
        cube1_x_diff_vis=gradient_benchmark["parameter_results"]["cube1_x"]["diff_vis"],
        cube1_x_los_ad=gradient_benchmark["parameter_results"]["cube1_x"]["component_grids"]["los"]["ad"],
        cube1_x_los_fd=gradient_benchmark["parameter_results"]["cube1_x"]["component_grids"]["los"]["fd"],
        cube1_x_reflection_ad=gradient_benchmark["parameter_results"]["cube1_x"]["component_grids"]["reflection"]["ad"],
        cube1_x_reflection_fd=gradient_benchmark["parameter_results"]["cube1_x"]["component_grids"]["reflection"]["fd"],
        cube1_x_diffraction_ad=gradient_benchmark["parameter_results"]["cube1_x"]["component_grids"]["diffraction"]["ad"],
        cube1_x_diffraction_fd=gradient_benchmark["parameter_results"]["cube1_x"]["component_grids"]["diffraction"]["fd"],
        grid_x=gradient_benchmark["grid_x"],
        grid_y=gradient_benchmark["grid_y"],
    )
    return output_path


def save_json(gradient_benchmark, *, output_path: Path, component_figure_path: Path | None = None) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary = asdict(gradient_benchmark["summary"])
    if component_figure_path is not None:
        summary["component_figure"] = str(component_figure_path)
    summary["parameters"] = {}
    for parameter in ("tx_x", "cube1_x"):
        ad = gradient_benchmark["parameter_results"][parameter]["ad"]
        fd = gradient_benchmark["parameter_results"][parameter]["fd"]
        diff = ad - fd
        corr = np.corrcoef(ad.ravel(), fd.ravel())[0, 1]
        summary["parameters"][parameter] = {
            "ad_abs_sum": float(np.sum(np.abs(ad))),
            "fd_abs_sum": float(np.sum(np.abs(fd))),
            "los_ad_abs_sum": float(
                gradient_benchmark["parameter_results"][parameter]["component_grad_abs_sums"]["los"]
            ),
            "reflection_ad_abs_sum": float(
                gradient_benchmark["parameter_results"][parameter]["component_grad_abs_sums"]["reflection"]
            ),
            "diffraction_ad_abs_sum": float(
                gradient_benchmark["parameter_results"][parameter]["component_grad_abs_sums"]["diffraction"]
            ),
            "ad_fd_corr": float(corr),
            "ad_fd_max_abs_diff": float(np.max(np.abs(diff))),
            "ad_fd_mean_abs_diff": float(np.mean(np.abs(diff))),
            "scalar_jvp": float(gradient_benchmark["parameter_results"][parameter]["scalar_jvp"]),
            "scalar_vjp": float(gradient_benchmark["parameter_results"][parameter]["scalar_vjp"]),
            "scalar_fd": float(gradient_benchmark["parameter_results"][parameter]["scalar_fd"]),
            "backward_only_seconds": float(
                gradient_benchmark["parameter_results"][parameter]["timings_seconds"]["backward_only"]
            ),
            "vjp_total_seconds": float(
                gradient_benchmark["parameter_results"][parameter]["timings_seconds"]["vjp"]
            ),
            "scalar_jvp_fd_abs_diff": float(
                abs(
                    gradient_benchmark["parameter_results"][parameter]["scalar_jvp"]
                    - gradient_benchmark["parameter_results"][parameter]["scalar_fd"]
                )
            ),
            "scalar_vjp_fd_abs_diff": float(
                abs(
                    gradient_benchmark["parameter_results"][parameter]["scalar_vjp"]
                    - gradient_benchmark["parameter_results"][parameter]["scalar_fd"]
                )
            ),
            "scalar_vjp_jvp_abs_diff": float(
                abs(
                    gradient_benchmark["parameter_results"][parameter]["scalar_vjp"]
                    - gradient_benchmark["parameter_results"][parameter]["scalar_jvp"]
                )
            ),
        }
    output_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return output_path


def save_radiomap_monte_carlo_gradients_three_cubes(
    output_prefix: Path,
    *,
    grid_size: int = DEFAULT_GRID_SIZE,
    reflection_n_rays: int = DEFAULT_REFLECTION_N_RAYS,
    samples_per_tx: int = DEFAULT_SAMPLES_PER_TX,
    fd_step: float = DEFAULT_FD_STEP,
    bounds=DEFAULT_BOUNDS,
    plane_z: float = DEFAULT_PLANE_Z,
    tx_pos: tuple[float, float, float] = DEFAULT_TX_POS,
    combine_mode: str = DEFAULT_COMBINE_MODE,
    receiver_model: str = DEFAULT_RECEIVER_MODEL,
    shadow_boundary_mode: str = DEFAULT_SHADOW_BOUNDARY_MODE,
    accumulation_backend: str = DEFAULT_ACCUMULATION_BACKEND,
    max_diffractions: int = DEFAULT_MAX_DIFFRACTIONS,
) -> tuple[Path, Path, Path]:
    gradient_benchmark = build_gradient_benchmark(
        grid_size=int(grid_size),
        reflection_n_rays=int(reflection_n_rays),
        samples_per_tx=int(samples_per_tx),
        fd_step=float(fd_step),
        bounds=bounds,
        plane_z=float(plane_z),
        tx_pos=tuple(float(value) for value in tx_pos),
        combine_mode=str(combine_mode),
        receiver_model=str(receiver_model),
        shadow_boundary_mode=str(shadow_boundary_mode),
        accumulation_backend=str(accumulation_backend),
        max_diffractions=int(max_diffractions),
    )
    figure_path = save_figure(gradient_benchmark, output_path=output_prefix.with_suffix(".png"))
    component_figure_path = save_component_figure(
        gradient_benchmark,
        output_path=output_prefix.with_name(output_prefix.stem + "_components").with_suffix(".png"),
    )
    arrays_path = save_arrays(gradient_benchmark, output_path=output_prefix.with_suffix(".npz"))
    json_path = save_json(
        gradient_benchmark,
        output_path=output_prefix.with_suffix(".json"),
        component_figure_path=component_figure_path,
    )
    return figure_path, arrays_path, json_path


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid-size", type=int, default=DEFAULT_GRID_SIZE)
    parser.add_argument("--reflection-rays", type=int, default=DEFAULT_REFLECTION_N_RAYS)
    parser.add_argument("--samples-per-tx", type=int, default=DEFAULT_SAMPLES_PER_TX)
    parser.add_argument("--fd-step", type=float, default=DEFAULT_FD_STEP)
    parser.add_argument("--plane-z", type=float, default=DEFAULT_PLANE_Z)
    parser.add_argument("--tx-x", type=float, default=float(DEFAULT_TX_POS[0]))
    parser.add_argument("--tx-y", type=float, default=float(DEFAULT_TX_POS[1]))
    parser.add_argument("--tx-z", type=float, default=float(DEFAULT_TX_POS[2]))
    parser.add_argument("--combine-mode", type=str, default=DEFAULT_COMBINE_MODE)
    parser.add_argument("--receiver-model", type=str, default=DEFAULT_RECEIVER_MODEL)
    parser.add_argument("--shadow-boundary-mode", type=str, default=DEFAULT_SHADOW_BOUNDARY_MODE)
    parser.add_argument("--accumulation-backend", type=str, default=DEFAULT_ACCUMULATION_BACKEND)
    parser.add_argument("--max-diffractions", type=int, default=DEFAULT_MAX_DIFFRACTIONS)
    parser.add_argument("--xmin", type=float, default=float(DEFAULT_BOUNDS[0][0]))
    parser.add_argument("--xmax", type=float, default=float(DEFAULT_BOUNDS[0][1]))
    parser.add_argument("--ymin", type=float, default=float(DEFAULT_BOUNDS[1][0]))
    parser.add_argument("--ymax", type=float, default=float(DEFAULT_BOUNDS[1][1]))
    parser.add_argument("--output-prefix", type=Path, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    bounds = (
        (float(args.xmin), float(args.xmax)),
        (float(args.ymin), float(args.ymax)),
    )
    tx_pos = (float(args.tx_x), float(args.tx_y), float(args.tx_z))
    output_prefix = args.output_prefix if args.output_prefix is not None else DEFAULT_OUTPUT_PREFIX
    figure_path, arrays_path, json_path = save_radiomap_monte_carlo_gradients_three_cubes(
        output_prefix,
        grid_size=int(args.grid_size),
        reflection_n_rays=int(args.reflection_rays),
        samples_per_tx=int(args.samples_per_tx),
        fd_step=float(args.fd_step),
        bounds=bounds,
        plane_z=float(args.plane_z),
        tx_pos=tx_pos,
        combine_mode=str(args.combine_mode),
        receiver_model=str(args.receiver_model),
        shadow_boundary_mode=str(args.shadow_boundary_mode),
        accumulation_backend=str(args.accumulation_backend),
        max_diffractions=int(args.max_diffractions),
    )
    print(
        json.dumps(
            {
                "figure": str(figure_path),
                "component_figure": str(
                    output_prefix.with_name(output_prefix.stem + "_components").with_suffix(".png")
                ),
                "arrays": str(arrays_path),
                "json": str(json_path),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
