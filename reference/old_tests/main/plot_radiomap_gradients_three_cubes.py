"""Pure witwin three-cube radio-map gradient benchmark helper."""

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
from tests.main.plot_radiomap_sionna_three_cubes import (
    DEFAULT_BOUNDS,
    DEFAULT_GRID_SIZE,
    DEFAULT_N_RAYS,
    DEFAULT_PLANE_Z,
    DEFAULT_TX_POS,
    _decorate_axis,
    _output_dir,
)
from witwin.channel import RadioMapMonitor, Tracer
DEFAULT_FD_STEP = 1.0e-3
DEFAULT_COMBINE_MODE = "coherent"
DEFAULT_RECEIVER_MODEL = "matched_isotropic"
DEFAULT_SHADOW_BOUNDARY_MODE = "matched_isb_completion"
DEFAULT_ACCUMULATION_BACKEND = "cell_accumulation"
DEFAULT_MAX_DIFFRACTIONS = 2
DEFAULT_OUTPUT_PREFIX = _output_dir() / "radiomap_three_cubes_gradients"
DEFAULT_TRACE_SEED = 7
_GRAD_FLAGS = dr.ADFlag.Default | dr.ADFlag.AllowNoGrad
_COMPONENTS = ("los", "reflection", "diffraction")
_DIFFRACTION_BREAKDOWN_METRICS = (
    "raw_diffraction",
    "matched_isb_completion_only",
    "folded_diffraction",
)
_SCALAR_DIAGNOSTIC_METRICS = (
    "path_gain",
    *_DIFFRACTION_BREAKDOWN_METRICS,
)
_ALL_AD_METRICS = (
    "path_gain",
    *_COMPONENTS,
    *_DIFFRACTION_BREAKDOWN_METRICS,
)


@dataclass(frozen=True)
class GradientSummary:
    grid_size: int
    bounds: tuple[tuple[float, float], tuple[float, float]]
    plane_z: float
    tx_pos: tuple[float, float, float]
    n_rays: int
    fd_step: float
    combine_mode: str
    receiver_model: str
    shadow_boundary_mode: str
    accumulation_backend_requested: str
    accumulation_backend_resolved: str
    gradient_accumulation_backend: str
    max_diffractions: int
    path_counts: dict
    diffraction_diagnostics: dict
    runtime_backends: dict
    forward_backend_comparison: dict
    parameter_backends: dict
    timings_seconds: dict


def _time_call(func, /, *args, **kwargs):
    start = perf_counter()
    result = func(*args, **kwargs)
    return result, float(perf_counter() - start)


def _format_seconds(seconds: float) -> str:
    return f"{float(seconds):.2f}s"


class _temporary_env:
    def __init__(self, key: str, value: str):
        self._key = str(key)
        self._value = str(value)
        self._previous = None
        self._had_previous = False

    def __enter__(self):
        self._had_previous = self._key in os.environ
        self._previous = os.environ.get(self._key)
        os.environ[self._key] = self._value
        return self

    def __exit__(self, exc_type, exc, tb):
        if self._had_previous:
            os.environ[self._key] = self._previous
        else:
            os.environ.pop(self._key, None)
        return False


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
):
    return RadioMapMonitor(
        "three_cubes_gradient_rm",
        axis="z",
        position=float(plane_z),
        bounds=bounds,
        grid_shape=(int(grid_size), int(grid_size)),
        metric="path_gain",
        combine_mode=str(combine_mode),
        receiver_model=str(receiver_model),
        accumulation_backend=str(accumulation_backend),
        ray_mode="3d",
        quadrature_mode="center",
        max_diffractions=int(max_diffractions),
        shadow_boundary_mode=str(shadow_boundary_mode),
    )


def _make_tracer(scene, *, n_rays: int, max_diffractions: int):
    return Tracer(
        frequency=1.0e9,
        scene=scene,
        reflection_n_rays=int(n_rays),
        reflection_max_bounces=3,
        reflection_coef=0.8,
        enable_rd_diffraction=True,
        max_diffractions=int(max_diffractions),
    )


def _component_metrics(result, *, combine_mode: str) -> dict[str, object]:
    metrics = (
        getattr(result, "coherent_power", None)
        if str(combine_mode) == "coherent"
        else getattr(result, "incoherent", None)
    )
    if metrics is None:
        raise ValueError(f"Missing component metrics for combine_mode={combine_mode!r}")
    missing = [component for component in _COMPONENTS if component not in metrics]
    if missing:
        raise KeyError(f"Missing component metrics: {missing}")
    return {component: metrics[component] for component in _COMPONENTS}


def _zero_like_metric(value):
    return dr.zeros(type(value), dr.width(value))


def _diffraction_breakdown_metrics(result, *, combine_mode: str) -> dict[str, object]:
    metrics = (
        getattr(result, "coherent_power", None)
        if str(combine_mode) == "coherent"
        else getattr(result, "incoherent", None)
    )
    if metrics is None:
        raise ValueError(f"Missing diffraction diagnostics for combine_mode={combine_mode!r}")
    folded_diffraction = metrics["diffraction"]
    raw_diffraction = metrics.get("raw_diffraction", folded_diffraction)
    completion_only = metrics.get(
        "matched_isb_completion_only",
        metrics.get("matched_isb_completion"),
    )
    if completion_only is None:
        completion_only = _zero_like_metric(raw_diffraction)
    return {
        "raw_diffraction": raw_diffraction,
        "matched_isb_completion_only": completion_only,
        "folded_diffraction": metrics.get("folded_diffraction", folded_diffraction),
    }


def _trace_path_gain_payload(
    *,
    cube1_x,
    tx_pos,
    grid_size: int,
    n_rays: int,
    bounds,
    plane_z: float,
    combine_mode: str,
    receiver_model: str,
    shadow_boundary_mode: str,
    accumulation_backend: str,
    max_diffractions: int,
    include_components: bool = False,
):
    requested_accumulation_backend = str(accumulation_backend)
    grad_sensitive = False
    try:
        grad_sensitive = bool(dr.grad_enabled(cube1_x))
    except Exception:
        grad_sensitive = False
    if not grad_sensitive:
        for axis in ("x", "y", "z"):
            try:
                if bool(dr.grad_enabled(getattr(tx_pos, axis))):
                    grad_sensitive = True
                    break
            except Exception:
                continue
    if grad_sensitive and requested_accumulation_backend == "cell_accumulation":
        requested_accumulation_backend = "baseline"

    try:
        dr.seed(int(DEFAULT_TRACE_SEED))
    except Exception:
        pass
    try:
        wt.register_sampler_seed(int(DEFAULT_TRACE_SEED))
    except Exception:
        pass
    scene = build_scene_for_cube1_x(cube1_x)
    tracer = _make_tracer(scene, n_rays=n_rays, max_diffractions=max_diffractions)
    monitor = _make_monitor(
        grid_size=grid_size,
        bounds=bounds,
        plane_z=plane_z,
        combine_mode=combine_mode,
        receiver_model=receiver_model,
        shadow_boundary_mode=shadow_boundary_mode,
        accumulation_backend=requested_accumulation_backend,
        max_diffractions=max_diffractions,
    )
    trace_output = tracer.trace(tx_pos, monitor=monitor, verbose=False)
    result = trace_output.monitor(monitor.name) if hasattr(trace_output, "monitor") else trace_output
    metrics = {"path_gain": result.path_gain}
    if include_components:
        metrics.update(_component_metrics(result, combine_mode=combine_mode))
        metrics.update(_diffraction_breakdown_metrics(result, combine_mode=combine_mode))
    return {
        "coords": {
            "grid_x": result.coords.grid_x,
            "grid_y": result.coords.grid_y,
            "x": result.coords.x,
            "y": result.coords.y,
        },
        "metrics": metrics,
        "metadata": result.metadata,
    }


def _trace_path_gain(
    *,
    cube1_x,
    tx_pos,
    grid_size: int,
    n_rays: int,
    bounds,
    plane_z: float,
    combine_mode: str,
    receiver_model: str,
    shadow_boundary_mode: str,
    accumulation_backend: str,
    max_diffractions: int,
    include_components: bool = False,
):
    payload = _trace_path_gain_payload(
        cube1_x=cube1_x,
        tx_pos=tx_pos,
        grid_size=grid_size,
        n_rays=n_rays,
        bounds=bounds,
        plane_z=plane_z,
        combine_mode=combine_mode,
        receiver_model=receiver_model,
        shadow_boundary_mode=shadow_boundary_mode,
        accumulation_backend=accumulation_backend,
        max_diffractions=max_diffractions,
        include_components=include_components,
    )
    return (
        np.asarray(payload["coords"]["grid_x"], dtype=np.float64),
        np.asarray(payload["coords"]["grid_y"], dtype=np.float64),
        np.asarray(payload["metrics"]["path_gain"], dtype=np.float64),
        payload,
    )


def parameter_config(parameter: str, *, tx_pos):
    tx_pos = tuple(float(value) for value in tx_pos)
    if parameter == "tx_x":
        return {
            "label": "tx_x",
            "cube1_x": CUBE1_BASE_CENTER[0],
            "tx_pos": tx_pos,
            "perturb": lambda step: (
                {"cube1_x": CUBE1_BASE_CENTER[0], "tx_pos": (tx_pos[0] + step, tx_pos[1], tx_pos[2])},
                {"cube1_x": CUBE1_BASE_CENTER[0], "tx_pos": (tx_pos[0] - step, tx_pos[1], tx_pos[2])},
            ),
        }
    if parameter == "cube1_x":
        return {
            "label": "cube1_x",
            "cube1_x": CUBE1_BASE_CENTER[0],
            "tx_pos": tx_pos,
            "perturb": lambda step: (
                {"cube1_x": CUBE1_BASE_CENTER[0] + step, "tx_pos": tx_pos},
                {"cube1_x": CUBE1_BASE_CENTER[0] - step, "tx_pos": tx_pos},
            ),
        }
    raise ValueError(f"Unsupported parameter: {parameter}")


def ad_gradient_path_gain(
    parameter: str,
    *,
    tx_pos,
    grid_size: int,
    n_rays: int,
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
        parameter_value = wt.Float(config["cube1_x"])
        dr.enable_grad(parameter_value)
        payload = _trace_path_gain_payload(
            cube1_x=parameter_value,
            tx_pos=wt.Point3f(*config["tx_pos"]),
            grid_size=grid_size,
            n_rays=n_rays,
            bounds=bounds,
            plane_z=plane_z,
            combine_mode=combine_mode,
            receiver_model=receiver_model,
            shadow_boundary_mode=shadow_boundary_mode,
            accumulation_backend=accumulation_backend,
            max_diffractions=max_diffractions,
            include_components=True,
        )
    else:
        parameter_value = wt.Float(config["tx_pos"][0])
        dr.enable_grad(parameter_value)
        payload = _trace_path_gain_payload(
            cube1_x=config["cube1_x"],
            tx_pos=wt.Point3f(parameter_value, config["tx_pos"][1], config["tx_pos"][2]),
            grid_size=grid_size,
            n_rays=n_rays,
            bounds=bounds,
            plane_z=plane_z,
            combine_mode=combine_mode,
            receiver_model=receiver_model,
            shadow_boundary_mode=shadow_boundary_mode,
            accumulation_backend=accumulation_backend,
            max_diffractions=max_diffractions,
            include_components=True,
        )
    metric_names = tuple(name for name in _ALL_AD_METRICS if name in payload["metrics"])
    dr.set_grad(parameter_value, 1.0)
    grads = dr.forward_to(*(payload["metrics"][name] for name in metric_names), flags=_GRAD_FLAGS)
    if not isinstance(grads, tuple):
        grads = (grads,)
    gradient_payload = {
        name: np.asarray(grad, dtype=np.float64)
        for name, grad in zip(metric_names, grads, strict=True)
    }
    return (
        payload["coords"]["grid_x"],
        payload["coords"]["grid_y"],
        gradient_payload,
        payload,
    )


def fd_gradient_path_gain(
    parameter: str,
    *,
    tx_pos,
    grid_size: int,
    n_rays: int,
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
    plus_payload = _trace_path_gain_payload(
        cube1_x=plus_cfg["cube1_x"],
        tx_pos=wt.Point3f(*plus_cfg["tx_pos"]),
        grid_size=grid_size,
        n_rays=n_rays,
        bounds=bounds,
        plane_z=plane_z,
        combine_mode=combine_mode,
        receiver_model=receiver_model,
        shadow_boundary_mode=shadow_boundary_mode,
        accumulation_backend=accumulation_backend,
        max_diffractions=max_diffractions,
        include_components=True,
    )
    minus_payload = _trace_path_gain_payload(
        cube1_x=minus_cfg["cube1_x"],
        tx_pos=wt.Point3f(*minus_cfg["tx_pos"]),
        grid_size=grid_size,
        n_rays=n_rays,
        bounds=bounds,
        plane_z=plane_z,
        combine_mode=combine_mode,
        receiver_model=receiver_model,
        shadow_boundary_mode=shadow_boundary_mode,
        accumulation_backend=accumulation_backend,
        max_diffractions=max_diffractions,
        include_components=True,
    )
    metric_names = tuple(name for name in _ALL_AD_METRICS if name in plus_payload["metrics"])
    gradients = {
        name: np.asarray(
            (
                np.asarray(plus_payload["metrics"][name], dtype=np.float64)
                - np.asarray(minus_payload["metrics"][name], dtype=np.float64)
            )
            / (2.0 * float(fd_step)),
            dtype=np.float64,
        )
        for name in metric_names
    }
    return gradients, plus_payload, minus_payload


def _as_grid(array, grid_size: int):
    return np.asarray(array, dtype=np.float64).reshape(grid_size, grid_size)


def _path_gain_db(path_gain):
    return 10.0 * np.log10(np.maximum(np.asarray(path_gain, dtype=np.float64), 1.0e-20))


def _loss_weight_probe(grid_size: int) -> np.ndarray:
    # Use a zero-mean spatial probe so scalar JVP/VJP-vs-FD checks are driven
    # by local gradient shape rather than a tiny map-wide bias accumulated over
    # every cell in the reduction.
    coord = np.linspace(-1.0, 1.0, int(grid_size), dtype=np.float32)
    grid_y, grid_x = np.meshgrid(coord, coord, indexing="ij")
    probe = np.cos(4.0 * np.pi * grid_x) * np.cos(4.0 * np.pi * grid_y)
    probe = probe - probe.mean(dtype=np.float64)
    scale = max(float(np.max(np.abs(probe))), 1.0)
    return (probe / scale).astype(np.float32)


def _loss_weights(path_gain, grid_size: int):
    probe = _loss_weight_probe(grid_size)
    return type(path_gain)(probe)


def _scalar_from_drjit(value) -> float:
    array = np.asarray(dr.detach(value), dtype=np.float64).reshape(-1)
    return float(array[0]) if array.size > 0 else 0.0


def _safe_correlation(lhs, rhs) -> float:
    lhs_flat = np.asarray(lhs, dtype=np.float64).ravel()
    rhs_flat = np.asarray(rhs, dtype=np.float64).ravel()
    if lhs_flat.size == 0 or rhs_flat.size == 0:
        return 0.0
    lhs_std = float(np.std(lhs_flat))
    rhs_std = float(np.std(rhs_flat))
    if lhs_std == 0.0 or rhs_std == 0.0:
        return 1.0 if np.allclose(lhs_flat, rhs_flat) else 0.0
    corr = float(np.corrcoef(lhs_flat, rhs_flat)[0, 1])
    return corr if np.isfinite(corr) else 0.0


def _build_metric_payload(
    parameter: str,
    *,
    tx_pos,
    grid_size: int,
    n_rays: int,
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
        parameter_value = wt.Float(config["cube1_x"])
        dr.enable_grad(parameter_value)
        payload = _trace_path_gain_payload(
            cube1_x=parameter_value,
            tx_pos=wt.Point3f(*config["tx_pos"]),
            grid_size=grid_size,
            n_rays=n_rays,
            bounds=bounds,
            plane_z=plane_z,
            combine_mode=combine_mode,
            receiver_model=receiver_model,
            shadow_boundary_mode=shadow_boundary_mode,
            accumulation_backend=accumulation_backend,
            max_diffractions=max_diffractions,
            include_components=True,
        )
        return payload, parameter_value
    parameter_value = wt.Float(config["tx_pos"][0])
    dr.enable_grad(parameter_value)
    payload = _trace_path_gain_payload(
        cube1_x=config["cube1_x"],
        tx_pos=wt.Point3f(parameter_value, config["tx_pos"][1], config["tx_pos"][2]),
        grid_size=grid_size,
        n_rays=n_rays,
        bounds=bounds,
        plane_z=plane_z,
        combine_mode=combine_mode,
        receiver_model=receiver_model,
        shadow_boundary_mode=shadow_boundary_mode,
        accumulation_backend=accumulation_backend,
        max_diffractions=max_diffractions,
        include_components=True,
    )
    return payload, parameter_value


def vjp_scalar_losses(
    parameter: str,
    *,
    metric_names,
    tx_pos,
    grid_size: int,
    n_rays: int,
    bounds,
    plane_z: float,
    combine_mode: str,
    receiver_model: str,
    shadow_boundary_mode: str,
    accumulation_backend: str,
    max_diffractions: int,
):
    payload, parameter_value = _build_metric_payload(
        parameter,
        tx_pos=tx_pos,
        grid_size=grid_size,
        n_rays=n_rays,
        bounds=bounds,
        plane_z=plane_z,
        combine_mode=combine_mode,
        receiver_model=receiver_model,
        shadow_boundary_mode=shadow_boundary_mode,
        accumulation_backend=accumulation_backend,
        max_diffractions=max_diffractions,
    )
    losses = {
        name: dr.sum(payload["metrics"][name] * _loss_weights(payload["metrics"][name], grid_size))
        for name in metric_names
    }
    gradients = {}
    for name, loss in losses.items():
        dr.clear_grad(parameter_value)
        dr.backward(loss, flags=_GRAD_FLAGS)
        gradients[name] = _scalar_from_drjit(dr.grad(parameter_value))
    return gradients, payload


def _grid_metric_summary(ad, fd) -> dict[str, float]:
    ad_array = np.asarray(ad, dtype=np.float64)
    fd_array = np.asarray(fd, dtype=np.float64)
    diff = ad_array - fd_array
    return {
        "ad_abs_sum": float(np.sum(np.abs(ad_array))),
        "fd_abs_sum": float(np.sum(np.abs(fd_array))),
        "ad_fd_corr": float(_safe_correlation(ad_array, fd_array)),
        "ad_fd_max_abs_diff": float(np.max(np.abs(diff))),
        "ad_fd_mean_abs_diff": float(np.mean(np.abs(diff))),
    }


def _forward_backend_comparison(
    *,
    cube1_x,
    tx_pos,
    grid_size: int,
    n_rays: int,
    bounds,
    plane_z: float,
    combine_mode: str,
    receiver_model: str,
    shadow_boundary_mode: str,
    accumulation_backend: str,
    max_diffractions: int,
):
    with _temporary_env("WITWIN_RADIOMAP_DIFF2_FORWARD_FAST_PATH", "0"):
        baseline_payload = _trace_path_gain_payload(
            cube1_x=cube1_x,
            tx_pos=tx_pos,
            grid_size=grid_size,
            n_rays=n_rays,
            bounds=bounds,
            plane_z=plane_z,
            combine_mode=combine_mode,
            receiver_model=receiver_model,
            shadow_boundary_mode=shadow_boundary_mode,
            accumulation_backend="baseline",
            max_diffractions=max_diffractions,
            include_components=True,
        )
    with _temporary_env("WITWIN_RADIOMAP_DIFF2_FORWARD_FAST_PATH", "1"):
        requested_payload = _trace_path_gain_payload(
            cube1_x=cube1_x,
            tx_pos=tx_pos,
            grid_size=grid_size,
            n_rays=n_rays,
            bounds=bounds,
            plane_z=plane_z,
            combine_mode=combine_mode,
            receiver_model=receiver_model,
            shadow_boundary_mode=shadow_boundary_mode,
            accumulation_backend=accumulation_backend,
            max_diffractions=max_diffractions,
            include_components=True,
        )
    metric_names = (
        "path_gain",
        "los",
        "reflection",
        "diffraction",
        "raw_diffraction",
        "matched_isb_completion_only",
        "folded_diffraction",
    )
    return {
        "baseline_backend": dict(baseline_payload["metadata"]["accumulation_backend"]),
        "requested_backend": dict(requested_payload["metadata"]["accumulation_backend"]),
        "metrics": {
            name: _grid_metric_summary(
                np.asarray(requested_payload["metrics"][name], dtype=np.float64),
                np.asarray(baseline_payload["metrics"][name], dtype=np.float64),
            )
            for name in metric_names
            if name in requested_payload["metrics"] and name in baseline_payload["metrics"]
        },
    }


def jvp_scalar_loss(
    parameter: str,
    *,
    tx_pos,
    grid_size: int,
    n_rays: int,
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
            n_rays=n_rays,
            bounds=bounds,
            plane_z=plane_z,
            combine_mode=combine_mode,
            receiver_model=receiver_model,
            shadow_boundary_mode=shadow_boundary_mode,
            accumulation_backend=accumulation_backend,
            max_diffractions=max_diffractions,
        )
        weights = _loss_weights(payload["metrics"]["path_gain"], grid_size)
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
        n_rays=n_rays,
        bounds=bounds,
        plane_z=plane_z,
        combine_mode=combine_mode,
        receiver_model=receiver_model,
        shadow_boundary_mode=shadow_boundary_mode,
        accumulation_backend=accumulation_backend,
        max_diffractions=max_diffractions,
    )
    weights = _loss_weights(payload["metrics"]["path_gain"], grid_size)
    loss = dr.sum(payload["metrics"]["path_gain"] * weights)
    dr.set_grad(tx_x, 1.0)
    dr.forward_to(loss, flags=_GRAD_FLAGS)
    return _scalar_from_drjit(dr.grad(loss))


def vjp_scalar_loss(
    parameter: str,
    *,
    tx_pos,
    grid_size: int,
    n_rays: int,
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
            n_rays=n_rays,
            bounds=bounds,
            plane_z=plane_z,
            combine_mode=combine_mode,
            receiver_model=receiver_model,
            shadow_boundary_mode=shadow_boundary_mode,
            accumulation_backend=accumulation_backend,
            max_diffractions=max_diffractions,
        )
        weights = _loss_weights(payload["metrics"]["path_gain"], grid_size)
        loss = dr.sum(payload["metrics"]["path_gain"] * weights)
        dr.backward(loss, flags=_GRAD_FLAGS)
        return _scalar_from_drjit(dr.grad(cube1_x))

    tx_x = wt.Float(config["tx_pos"][0])
    dr.enable_grad(tx_x)
    payload = _trace_path_gain_payload(
        cube1_x=config["cube1_x"],
        tx_pos=wt.Point3f(tx_x, config["tx_pos"][1], config["tx_pos"][2]),
        grid_size=grid_size,
        n_rays=n_rays,
        bounds=bounds,
        plane_z=plane_z,
        combine_mode=combine_mode,
        receiver_model=receiver_model,
        shadow_boundary_mode=shadow_boundary_mode,
        accumulation_backend=accumulation_backend,
        max_diffractions=max_diffractions,
    )
    weights = _loss_weights(payload["metrics"]["path_gain"], grid_size)
    loss = dr.sum(payload["metrics"]["path_gain"] * weights)
    dr.backward(loss, flags=_GRAD_FLAGS)
    return _scalar_from_drjit(dr.grad(tx_x))


def fd_scalar_loss(
    parameter: str,
    *,
    tx_pos,
    grid_size: int,
    n_rays: int,
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
        n_rays=n_rays,
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
        n_rays=n_rays,
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
    n_rays: int,
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
    gradient_accumulation_backend = "baseline"
    (_, _, total_path_gain, total_payload), total_forward_seconds = _time_call(
        _trace_path_gain,
        cube1_x=CUBE1_BASE_CENTER[0],
        tx_pos=wt.Point3f(*tx_pos),
        grid_size=grid_size,
        n_rays=n_rays,
        bounds=bounds,
        plane_z=plane_z,
        combine_mode=combine_mode,
        receiver_model=receiver_model,
        shadow_boundary_mode=shadow_boundary_mode,
        accumulation_backend=accumulation_backend,
        max_diffractions=max_diffractions,
    )
    forward_backend_comparison, forward_backend_seconds = _time_call(
        _forward_backend_comparison,
        cube1_x=CUBE1_BASE_CENTER[0],
        tx_pos=wt.Point3f(*tx_pos),
        grid_size=grid_size,
        n_rays=n_rays,
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
    timings_seconds = {
        "forward_total": float(total_forward_seconds),
        "forward_backend_comparison": float(forward_backend_seconds),
    }

    for parameter in ("tx_x", "cube1_x"):
        (_, _, ad_gradient_payload, ad_payload), ad_seconds = _time_call(
            ad_gradient_path_gain,
            parameter,
            tx_pos=tx_pos,
            grid_size=grid_size,
            n_rays=n_rays,
            bounds=bounds,
            plane_z=plane_z,
            combine_mode=combine_mode,
            receiver_model=receiver_model,
            shadow_boundary_mode=shadow_boundary_mode,
            accumulation_backend=gradient_accumulation_backend,
            max_diffractions=max_diffractions,
        )
        (fd_gradient_payload, plus_payload, minus_payload), fd_seconds = _time_call(
            fd_gradient_path_gain,
            parameter,
            tx_pos=tx_pos,
            grid_size=grid_size,
            n_rays=n_rays,
            fd_step=fd_step,
            bounds=bounds,
            plane_z=plane_z,
            combine_mode=combine_mode,
            receiver_model=receiver_model,
            shadow_boundary_mode=shadow_boundary_mode,
            accumulation_backend=gradient_accumulation_backend,
            max_diffractions=max_diffractions,
        )
        ad_grid = _as_grid(ad_gradient_payload["path_gain"], grid_size)
        fd_grid = _as_grid(fd_gradient_payload["path_gain"], grid_size)
        scalar_probe = _loss_weight_probe(grid_size).astype(np.float64)
        scalar_jvp_start = perf_counter()
        scalar_jvp = {
            metric_name: float(
                np.sum(_as_grid(ad_gradient_payload[metric_name], grid_size) * scalar_probe)
            )
            for metric_name in _SCALAR_DIAGNOSTIC_METRICS
            if metric_name in ad_gradient_payload
        }
        jvp_seconds = float(perf_counter() - scalar_jvp_start)
        (scalar_vjp, _vjp_payload), vjp_seconds = _time_call(
            vjp_scalar_losses,
            parameter,
            metric_names=tuple(
                metric_name
                for metric_name in _SCALAR_DIAGNOSTIC_METRICS
                if metric_name in ad_payload["metrics"]
            ),
            tx_pos=tx_pos,
            grid_size=grid_size,
            n_rays=n_rays,
            bounds=bounds,
            plane_z=plane_z,
            combine_mode=combine_mode,
            receiver_model=receiver_model,
            shadow_boundary_mode=shadow_boundary_mode,
            accumulation_backend=gradient_accumulation_backend,
            max_diffractions=max_diffractions,
        )
        scalar_fd_start = perf_counter()
        scalar_fd = {
            metric_name: float(
                np.sum(_as_grid(fd_gradient_payload[metric_name], grid_size) * scalar_probe)
            )
            for metric_name in _SCALAR_DIAGNOSTIC_METRICS
            if metric_name in fd_gradient_payload
        }
        scalar_fd_seconds = float(perf_counter() - scalar_fd_start)
        ad_vis = gradient_db_magnitude(ad_grid)
        fd_vis = gradient_db_magnitude(fd_grid)
        diff_vis = ad_vis - fd_vis
        diagnostic_grids = {
            metric_name: {
                "ad": _as_grid(ad_gradient_payload[metric_name], grid_size),
                "fd": _as_grid(fd_gradient_payload[metric_name], grid_size),
            }
            for metric_name in _DIFFRACTION_BREAKDOWN_METRICS
            if metric_name in ad_gradient_payload and metric_name in fd_gradient_payload
        }
        parameter_results[parameter] = {
            "ad": ad_grid,
            "fd": fd_grid,
            "ad_vis": ad_vis,
            "fd_vis": fd_vis,
            "diff_vis": diff_vis,
            "ad_backend": dict(ad_payload["metadata"]["accumulation_backend"]),
            "fd_backend": dict(plus_payload["metadata"]["accumulation_backend"]),
            "plus_backend": dict(plus_payload["metadata"]["accumulation_backend"]),
            "minus_backend": dict(minus_payload["metadata"]["accumulation_backend"]),
            "plus_path_counts": dict(plus_payload["metadata"].get("path_counts", {})),
            "minus_path_counts": dict(minus_payload["metadata"].get("path_counts", {})),
            "plus_runtime_backends": dict(plus_payload["metadata"].get("runtime_backends", {})),
            "minus_runtime_backends": dict(minus_payload["metadata"].get("runtime_backends", {})),
            "plus_diffraction_diagnostics": dict(
                plus_payload["metadata"].get("diffraction_diagnostics", {})
            ),
            "minus_diffraction_diagnostics": dict(
                minus_payload["metadata"].get("diffraction_diagnostics", {})
            ),
            "component_grids": {
                component: {
                    "ad": _as_grid(ad_gradient_payload[component], grid_size),
                    "fd": _as_grid(fd_gradient_payload[component], grid_size),
                }
                for component in _COMPONENTS
            },
            "diagnostic_grids": diagnostic_grids,
            "component_grad_abs_sums": {
                component: float(np.sum(np.abs(ad_gradient_payload[component])))
                for component in _COMPONENTS
            },
            "scalar_metrics": {
                metric_name: {
                    "jvp": float(scalar_jvp[metric_name]),
                    "vjp": float(scalar_vjp[metric_name]),
                    "fd": float(scalar_fd[metric_name]),
                }
                for metric_name in scalar_jvp
            },
            "scalar_jvp": float(scalar_jvp["path_gain"]),
            "scalar_vjp": float(scalar_vjp["path_gain"]),
            "scalar_fd": float(scalar_fd["path_gain"]),
            "timings_seconds": {
                "ad": float(ad_seconds),
                "fd": float(fd_seconds),
                "jvp": float(jvp_seconds),
                "vjp": float(vjp_seconds),
                "scalar_fd": float(scalar_fd_seconds),
            },
        }
        parameter_backends[parameter] = {
            "ad_backend": dict(ad_payload["metadata"]["accumulation_backend"]),
            "fd_backend": dict(plus_payload["metadata"]["accumulation_backend"]),
            "plus_backend": dict(plus_payload["metadata"]["accumulation_backend"]),
            "minus_backend": dict(minus_payload["metadata"]["accumulation_backend"]),
        }
        timings_seconds[parameter] = dict(parameter_results[parameter]["timings_seconds"])

    summary = GradientSummary(
        grid_size=int(grid_size),
        bounds=tuple((float(a), float(b)) for (a, b) in bounds),
        plane_z=float(plane_z),
        tx_pos=tuple(float(value) for value in tx_pos),
        n_rays=int(n_rays),
        fd_step=float(fd_step),
        combine_mode=str(combine_mode),
        receiver_model=str(receiver_model),
        shadow_boundary_mode=str(shadow_boundary_mode),
        accumulation_backend_requested=str(accumulation_backend),
        accumulation_backend_resolved=str(
            total_payload["metadata"]["accumulation_backend"]["resolved"]
        ),
        gradient_accumulation_backend=str(gradient_accumulation_backend),
        max_diffractions=int(max_diffractions),
        path_counts=dict(total_payload["metadata"].get("path_counts", {})),
        diffraction_diagnostics=dict(total_payload["metadata"].get("diffraction_diagnostics", {})),
        runtime_backends=dict(total_payload["metadata"].get("runtime_backends", {})),
        forward_backend_comparison=forward_backend_comparison,
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
                    f"Path Gain (dB), {parameter}\n"
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
                f"AD-FD Gradient Delta (dB), {parameter}\nVJP={_format_seconds(row_timings['vjp'])}",
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
            "Pure Witwin Three-Cube Radiomap Matched-ISB Off-Plane Gradients\n"
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
        row_timings = row_results["timings_seconds"]
        for component_index, component in enumerate(_COMPONENTS):
            ad_grid = row_results["component_grids"][component]["ad"]
            fd_grid = row_results["component_grids"][component]["fd"]
            ad_vis = gradient_db_magnitude(ad_grid)
            fd_vis = gradient_db_magnitude(fd_grid)
            diff_grid = ad_vis - fd_vis
            diff_vmax = max(float(np.nanpercentile(np.abs(diff_grid), 99.0)), 1.0)
            col_base = 3 * component_index
            panels = (
                (
                    axes[row, col_base + 0],
                    ad_vis,
                    (
                        f"{component} AD, {parameter}\n"
                        f"AD={_format_seconds(row_timings['ad'])}"
                    ),
                    "magma",
                    float(np.nanpercentile(ad_vis, 5.0)),
                    float(np.nanpercentile(ad_vis, 99.0)),
                ),
                (
                    axes[row, col_base + 1],
                    fd_vis,
                    (
                        f"{component} FD, {parameter}\n"
                        f"FD={_format_seconds(row_timings['fd'])}"
                    ),
                    "magma",
                    float(np.nanpercentile(fd_vis, 5.0)),
                    float(np.nanpercentile(fd_vis, 99.0)),
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
            "Pure Witwin Three-Cube Radiomap Component Gradients\n"
            f"grid={gradient_benchmark['summary'].grid_size}x{gradient_benchmark['summary'].grid_size}, "
            f"xy slice z={gradient_benchmark['summary'].plane_z:.1f}, "
            f"tx=({tx_pos[0]:.1f}, {tx_pos[1]:.1f}, {tx_pos[2]:.1f})"
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
        tx_x_reflection_ad=gradient_benchmark["parameter_results"]["tx_x"]["component_grids"][
            "reflection"
        ]["ad"],
        tx_x_reflection_fd=gradient_benchmark["parameter_results"]["tx_x"]["component_grids"][
            "reflection"
        ]["fd"],
        tx_x_diffraction_ad=gradient_benchmark["parameter_results"]["tx_x"]["component_grids"][
            "diffraction"
        ]["ad"],
        tx_x_diffraction_fd=gradient_benchmark["parameter_results"]["tx_x"]["component_grids"][
            "diffraction"
        ]["fd"],
        tx_x_raw_diffraction_ad=gradient_benchmark["parameter_results"]["tx_x"]["diagnostic_grids"][
            "raw_diffraction"
        ]["ad"],
        tx_x_raw_diffraction_fd=gradient_benchmark["parameter_results"]["tx_x"]["diagnostic_grids"][
            "raw_diffraction"
        ]["fd"],
        tx_x_matched_isb_completion_only_ad=gradient_benchmark["parameter_results"]["tx_x"][
            "diagnostic_grids"
        ]["matched_isb_completion_only"]["ad"],
        tx_x_matched_isb_completion_only_fd=gradient_benchmark["parameter_results"]["tx_x"][
            "diagnostic_grids"
        ]["matched_isb_completion_only"]["fd"],
        tx_x_folded_diffraction_ad=gradient_benchmark["parameter_results"]["tx_x"]["diagnostic_grids"][
            "folded_diffraction"
        ]["ad"],
        tx_x_folded_diffraction_fd=gradient_benchmark["parameter_results"]["tx_x"]["diagnostic_grids"][
            "folded_diffraction"
        ]["fd"],
        cube1_x_ad=gradient_benchmark["parameter_results"]["cube1_x"]["ad"],
        cube1_x_fd=gradient_benchmark["parameter_results"]["cube1_x"]["fd"],
        cube1_x_diff_vis=gradient_benchmark["parameter_results"]["cube1_x"]["diff_vis"],
        cube1_x_los_ad=gradient_benchmark["parameter_results"]["cube1_x"]["component_grids"]["los"][
            "ad"
        ],
        cube1_x_los_fd=gradient_benchmark["parameter_results"]["cube1_x"]["component_grids"]["los"][
            "fd"
        ],
        cube1_x_reflection_ad=gradient_benchmark["parameter_results"]["cube1_x"]["component_grids"][
            "reflection"
        ]["ad"],
        cube1_x_reflection_fd=gradient_benchmark["parameter_results"]["cube1_x"]["component_grids"][
            "reflection"
        ]["fd"],
        cube1_x_diffraction_ad=gradient_benchmark["parameter_results"]["cube1_x"]["component_grids"][
            "diffraction"
        ]["ad"],
        cube1_x_diffraction_fd=gradient_benchmark["parameter_results"]["cube1_x"]["component_grids"][
            "diffraction"
        ]["fd"],
        cube1_x_raw_diffraction_ad=gradient_benchmark["parameter_results"]["cube1_x"][
            "diagnostic_grids"
        ]["raw_diffraction"]["ad"],
        cube1_x_raw_diffraction_fd=gradient_benchmark["parameter_results"]["cube1_x"][
            "diagnostic_grids"
        ]["raw_diffraction"]["fd"],
        cube1_x_matched_isb_completion_only_ad=gradient_benchmark["parameter_results"]["cube1_x"][
            "diagnostic_grids"
        ]["matched_isb_completion_only"]["ad"],
        cube1_x_matched_isb_completion_only_fd=gradient_benchmark["parameter_results"]["cube1_x"][
            "diagnostic_grids"
        ]["matched_isb_completion_only"]["fd"],
        cube1_x_folded_diffraction_ad=gradient_benchmark["parameter_results"]["cube1_x"][
            "diagnostic_grids"
        ]["folded_diffraction"]["ad"],
        cube1_x_folded_diffraction_fd=gradient_benchmark["parameter_results"]["cube1_x"][
            "diagnostic_grids"
        ]["folded_diffraction"]["fd"],
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
        row = gradient_benchmark["parameter_results"][parameter]
        path_gain_summary = _grid_metric_summary(row["ad"], row["fd"])
        scalar_metrics = {}
        for metric_name, values in row["scalar_metrics"].items():
            scalar_metrics[metric_name] = {
                "jvp": float(values["jvp"]),
                "vjp": float(values["vjp"]),
                "fd": float(values["fd"]),
                "jvp_fd_abs_diff": float(abs(values["jvp"] - values["fd"])),
                "vjp_fd_abs_diff": float(abs(values["vjp"] - values["fd"])),
                "vjp_jvp_abs_diff": float(abs(values["vjp"] - values["jvp"])),
            }
        summary["parameters"][parameter] = {
            **path_gain_summary,
            "ad_backend": dict(row["ad_backend"]),
            "fd_backend": dict(row["fd_backend"]),
            "plus_path_counts": dict(row["plus_path_counts"]),
            "minus_path_counts": dict(row["minus_path_counts"]),
            "plus_runtime_backends": dict(row["plus_runtime_backends"]),
            "minus_runtime_backends": dict(row["minus_runtime_backends"]),
            "plus_diffraction_diagnostics": dict(row["plus_diffraction_diagnostics"]),
            "minus_diffraction_diagnostics": dict(row["minus_diffraction_diagnostics"]),
            "los_ad_abs_sum": float(row["component_grad_abs_sums"]["los"]),
            "reflection_ad_abs_sum": float(row["component_grad_abs_sums"]["reflection"]),
            "diffraction_ad_abs_sum": float(row["component_grad_abs_sums"]["diffraction"]),
            "scalar_metrics": scalar_metrics,
            "scalar_jvp": float(row["scalar_metrics"]["path_gain"]["jvp"]),
            "scalar_vjp": float(row["scalar_metrics"]["path_gain"]["vjp"]),
            "scalar_fd": float(row["scalar_metrics"]["path_gain"]["fd"]),
            "scalar_jvp_fd_abs_diff": float(
                scalar_metrics["path_gain"]["jvp_fd_abs_diff"]
            ),
            "scalar_vjp_fd_abs_diff": float(
                scalar_metrics["path_gain"]["vjp_fd_abs_diff"]
            ),
            "scalar_vjp_jvp_abs_diff": float(
                scalar_metrics["path_gain"]["vjp_jvp_abs_diff"]
            ),
            "components": {
                component: _grid_metric_summary(
                    row["component_grids"][component]["ad"],
                    row["component_grids"][component]["fd"],
                )
                for component in _COMPONENTS
            },
            "diagnostic_metrics": {
                metric_name: {
                    **_grid_metric_summary(
                        row["diagnostic_grids"][metric_name]["ad"],
                        row["diagnostic_grids"][metric_name]["fd"],
                    ),
                    **scalar_metrics[metric_name],
                }
                for metric_name in _DIFFRACTION_BREAKDOWN_METRICS
            },
        }
    output_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return output_path


def save_radiomap_gradients_three_cubes(
    output_prefix: Path,
    *,
    grid_size: int = DEFAULT_GRID_SIZE,
    n_rays: int = DEFAULT_N_RAYS,
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
        n_rays=int(n_rays),
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
    parser.add_argument("--n-rays", type=int, default=DEFAULT_N_RAYS)
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
    figure_path, arrays_path, json_path = save_radiomap_gradients_three_cubes(
        output_prefix,
        grid_size=int(args.grid_size),
        n_rays=int(args.n_rays),
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


__all__ = [
    "DEFAULT_OUTPUT_PREFIX",
    "build_gradient_benchmark",
    "save_arrays",
    "save_component_figure",
    "save_figure",
    "save_json",
    "save_radiomap_gradients_three_cubes",
]
