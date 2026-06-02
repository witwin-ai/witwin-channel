"""Visualize multipath total-field and per-cell AD/FD gradient fields."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path

import drjit as dr
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

import witwin as wt
from tests._scene_helpers import box_drjit_geometry, build_scene as build_test_scene
from witwin.channel import DEFAULT_VARIANT, Material, FieldMonitor, Tracer
from witwin.channel import native_extension_available
import witwin.channel.trace.reflection.api as reflection_api

CUBE1_BASE_CENTER = (-2.5, -3.0, 1.5)
CUBE2_CENTER = (2.0, 0.5, 1.5)
CUBE3_CENTER = (-0.5, 3.5, 1.5)
CUBE_SIZE = 2.0
TX_POS = (0.0, -5.0, 1.5)
TRACE_BOUNDS = ((-6.0, 6.0), (-6.0, 6.0))
MULTIPATH_RELATIVE_PERMITTIVITY = 1.0e4
MULTIPATH_SCENE_MATERIAL = Material(eps_r=MULTIPATH_RELATIVE_PERMITTIVITY, sigma_e=0.0)
_GRAD_TRACE_CONFIG = {
    "trace": {
        "reflection_field_backend": "native",
        "diffraction_execution": {
            "suffix_backend": "native",
            "suffix_dda": "symbolic",
        }
    }
}


@contextmanager
def _diagnostic_epc_override():
    """Force EPC so the multipath forward model matches the diagnostic reference."""

    original_policy = reflection_api._reflection_epc_policy

    def _forced_policy(
        *,
        reflection_detail,
        grid_axis: str,
        has_mesh_data: bool,
        scene,
        tx_pos,
        source_paths_per_bounce=None,
        prefer_epc=True,
        **kwargs,
    ):
        policy = original_policy(
            reflection_detail=reflection_detail,
            grid_axis=grid_axis,
            has_mesh_data=has_mesh_data,
            scene=scene,
            tx_pos=tx_pos,
            source_paths_per_bounce=source_paths_per_bounce,
            prefer_epc=prefer_epc,
            **kwargs,
        )
        if not policy.get("epc_eligible", False):
            return policy
        forced = dict(policy)
        forced["use_epc"] = True
        forced["policy"] = "diagnostic_forced_epc"
        forced["discovery_gradients_preserved"] = False
        return forced

    reflection_api._reflection_epc_policy = _forced_policy
    try:
        yield
    finally:
        reflection_api._reflection_epc_policy = original_policy


class _KeepAliveArray:
    """Keep the trace context alive until the Dr.Jit array is materialized."""

    def __init__(self, value, *keepalive):
        self._value = value
        self._keepalive = keepalive

    def __array__(self, dtype=None):
        array = np.asarray(self._value)
        if dtype is not None:
            array = array.astype(dtype, copy=False)
        return array


def cube_specs(cube1_x: float):
    return (
        {"center": (cube1_x, CUBE1_BASE_CENTER[1]), "size": CUBE_SIZE},
        {"center": (CUBE2_CENTER[0], CUBE2_CENTER[1]), "size": CUBE_SIZE},
        {"center": (CUBE3_CENTER[0], CUBE3_CENTER[1]), "size": CUBE_SIZE},
    )


def _is_python_scalar(value) -> bool:
    return isinstance(value, (int, float, np.floating))


@lru_cache(maxsize=16)
def _build_scene_for_cube1_x_cached(cube1_x: float):
    cube1_center = wt.Point3f(cube1_x, CUBE1_BASE_CENTER[1], CUBE1_BASE_CENTER[2])
    cube1 = box_drjit_geometry(center=cube1_center, size=CUBE_SIZE, rotation=None).to_mesh()
    cube2 = box_drjit_geometry(center=CUBE2_CENTER, size=CUBE_SIZE, rotation=None).to_mesh()
    cube3 = box_drjit_geometry(center=CUBE3_CENTER, size=CUBE_SIZE, rotation=None).to_mesh()
    return build_test_scene(cube1, cube2, cube3, material=MULTIPATH_SCENE_MATERIAL)


def build_scene_for_cube1_x(cube1_x):
    if _is_python_scalar(cube1_x):
        return _build_scene_for_cube1_x_cached(float(cube1_x))
    cube1_center = wt.Point3f(cube1_x, CUBE1_BASE_CENTER[1], CUBE1_BASE_CENTER[2])
    cube1 = box_drjit_geometry(center=cube1_center, size=CUBE_SIZE, rotation=None).to_mesh()
    cube2 = box_drjit_geometry(center=CUBE2_CENTER, size=CUBE_SIZE, rotation=None).to_mesh()
    cube3 = box_drjit_geometry(center=CUBE3_CENTER, size=CUBE_SIZE, rotation=None).to_mesh()
    return build_test_scene(cube1, cube2, cube3, material=MULTIPATH_SCENE_MATERIAL)


def make_monitor(grid_size: int):
    return FieldMonitor(
        "multipath_grad_viz",
        axis="z",
        position=TX_POS[2],
        bounds=TRACE_BOUNDS,
        grid_size=grid_size,
    )


def make_tracer(scene, n_rays: int):
    return Tracer(
        frequency=1e9,
        scene=scene,
        config=_GRAD_TRACE_CONFIG,
        reflection_n_rays=n_rays,
        reflection_max_bounces=3,
        reflection_coef=0.8,
        reflection_relative_permittivity=MULTIPATH_RELATIVE_PERMITTIVITY,
        enable_rd_diffraction=True,
        max_diffractions=2,
    )


def _coerce_tx_pos(tx_pos):
    if hasattr(tx_pos, "x") and hasattr(tx_pos, "y") and hasattr(tx_pos, "z"):
        return tx_pos
    return wt.Point3f(*tx_pos)


def _require_native_multipath_prerequisites(*, scene, tracer, monitor, tx_pos):
    del scene, monitor, tx_pos
    if not native_extension_available():
        raise RuntimeError("Multipath main requires the bundled native CUDA extension.")

    trace_config = tracer.config.trace
    if trace_config.reflection_field_backend != "native":
        raise RuntimeError(
            "Multipath main requires trace.reflection_field_backend='native'; "
            f"got {trace_config.reflection_field_backend!r}."
        )
    if trace_config.diffraction_execution.suffix_backend != "native":
        raise RuntimeError(
            "Multipath main requires trace.diffraction_execution.suffix_backend='native'; "
            f"got {trace_config.diffraction_execution.suffix_backend!r}."
        )


def _require_native_multipath_backends(metadata):
    reflection_backend = dict(metadata.get("reflection_backend", {}))
    diffraction_backend = dict(metadata.get("diffraction_accumulation_backend", {}))
    suffix_backend = dict(metadata.get("reflection_suffix_backend", {}))

    if (
        reflection_backend.get("resolved_backend") != "native"
        or reflection_backend.get("implementation") not in {"native_cuda_custom_op", "epc"}
    ):
        raise RuntimeError(
            "Multipath main requires the native reflection backend; "
            f"got reflection_backend={reflection_backend!r}."
        )
    if diffraction_backend.get("implementation") != "native_cuda_custom_op":
        raise RuntimeError(
            "Multipath main requires native diffraction accumulation; "
            f"got diffraction_accumulation_backend={diffraction_backend!r}."
        )
    if (
        suffix_backend.get("resolved_backend") != "native"
        or suffix_backend.get("implementation") != "native_cuda_custom_op"
    ):
        raise RuntimeError(
            "Multipath main requires the native reflected-suffix backend; "
            f"got reflection_suffix_backend={suffix_backend!r}."
        )


def build_trace_payload(
    *,
    cube1_x,
    tx_pos,
    grid_size: int,
    n_rays: int,
    scene=None,
    monitor=None,
    tracer=None,
):
    scene = build_scene_for_cube1_x(cube1_x) if scene is None else scene
    monitor = make_monitor(grid_size) if monitor is None else monitor
    tracer = make_tracer(scene, n_rays) if tracer is None else tracer
    tx_pos = _coerce_tx_pos(tx_pos)
    _require_native_multipath_prerequisites(
        scene=scene,
        tracer=tracer,
        monitor=monitor,
        tx_pos=tx_pos,
    )
    with _diagnostic_epc_override():
        result = tracer.trace(tx_pos, monitor=monitor, verbose=False, return_diffraction_audit=False)
    _require_native_multipath_backends(result.primary.metadata)
    return {
        "scene": scene,
        "monitor": monitor,
        "tracer": tracer,
        "result": result,
        "grid_x": result.primary.coords.grid_x,
        "grid_y": result.primary.coords.grid_y,
        "total": result.primary.field.total,
    }


def _coerce_trace_payload(
    *,
    payload,
    cube1_x,
    tx_pos,
    grid_size: int,
    n_rays: int,
):
    if payload is not None:
        return payload
    return build_trace_payload(
        cube1_x=cube1_x,
        tx_pos=tx_pos,
        grid_size=grid_size,
        n_rays=n_rays,
    )


def trace_total_field(*, cube1_x, tx_pos, grid_size: int, n_rays: int, payload=None):
    payload = _coerce_trace_payload(
        payload=payload,
        cube1_x=cube1_x,
        tx_pos=tx_pos,
        grid_size=grid_size,
        n_rays=n_rays,
    )
    return payload["grid_x"], payload["grid_y"], payload["total"]


def trace_total_power(*, cube1_x, tx_pos, grid_size: int, n_rays: int, payload=None):
    payload = _coerce_trace_payload(
        payload=payload,
        cube1_x=cube1_x,
        tx_pos=tx_pos,
        grid_size=grid_size,
        n_rays=n_rays,
    )
    grid_x = payload["grid_x"]
    grid_y = payload["grid_y"]
    total = payload["total"]
    power = total.real * total.real + total.imag * total.imag
    return grid_x, grid_y, power


_trace_total_payload = build_trace_payload


def parameter_config(parameter: str):
    if parameter == "tx_x":
        return {
            "label": "tx_x",
            "cube1_x": CUBE1_BASE_CENTER[0],
            "tx_pos": TX_POS,
            "tangent": lambda: ("tx_x", wt.Float(0.0), wt.Float(-5.0)),
            "perturb": lambda step: (
                {"cube1_x": CUBE1_BASE_CENTER[0], "tx_pos": (TX_POS[0] + step, TX_POS[1], TX_POS[2])},
                {"cube1_x": CUBE1_BASE_CENTER[0], "tx_pos": (TX_POS[0] - step, TX_POS[1], TX_POS[2])},
            ),
        }
    if parameter == "tx_y":
        return {
            "label": "tx_y",
            "cube1_x": CUBE1_BASE_CENTER[0],
            "tx_pos": TX_POS,
            "tangent": lambda: ("tx_y", wt.Float(0.0), wt.Float(-5.0)),
            "perturb": lambda step: (
                {"cube1_x": CUBE1_BASE_CENTER[0], "tx_pos": (TX_POS[0], TX_POS[1] + step, TX_POS[2])},
                {"cube1_x": CUBE1_BASE_CENTER[0], "tx_pos": (TX_POS[0], TX_POS[1] - step, TX_POS[2])},
            ),
        }
    if parameter == "cube1_x":
        return {
            "label": "cube1_x",
            "cube1_x": CUBE1_BASE_CENTER[0],
            "tx_pos": TX_POS,
            "tangent": lambda: ("cube1_x", wt.Float(CUBE1_BASE_CENTER[0]), None),
            "perturb": lambda step: (
                {"cube1_x": CUBE1_BASE_CENTER[0] + step, "tx_pos": TX_POS},
                {"cube1_x": CUBE1_BASE_CENTER[0] - step, "tx_pos": TX_POS},
            ),
        }
    raise ValueError(f"Unsupported parameter: {parameter}")


def _forward_power_gradient(total):
    """Derive d|E|^2/dp from complex-field tangents via chain rule."""
    FLAGS = dr.ADFlag.Default | dr.ADFlag.AllowNoGrad
    dr.forward_to((total.real, total.imag), flags=FLAGS)
    grad_real = dr.grad(total.real)
    grad_imag = dr.grad(total.imag)
    return 2.0 * (total.real * grad_real + total.imag * grad_imag)


def ad_gradient_field(parameter: str, grid_size: int, n_rays: int):
    config = parameter_config(parameter)
    tangent_name, variable_a, variable_b = config["tangent"]()
    if tangent_name == "cube1_x":
        cube1_x = variable_a
        dr.enable_grad(cube1_x)
        payload = _trace_total_payload(
            cube1_x=cube1_x,
            tx_pos=config["tx_pos"],
            grid_size=grid_size,
            n_rays=n_rays,
        )
        dr.set_grad(cube1_x, 1.0)
        gradient = _KeepAliveArray(_forward_power_gradient(payload["total"]), payload)
        return payload["grid_x"], payload["grid_y"], gradient

    tx_x = variable_a
    tx_y = variable_b
    dr.enable_grad(tx_x, tx_y)
    payload = _trace_total_payload(
        cube1_x=config["cube1_x"],
        tx_pos=(tx_x, tx_y, TX_POS[2]),
        grid_size=grid_size,
        n_rays=n_rays,
    )
    if tangent_name == "tx_x":
        dr.set_grad(tx_x, 1.0)
        dr.set_grad(tx_y, 0.0)
    else:
        dr.set_grad(tx_x, 0.0)
        dr.set_grad(tx_y, 1.0)
    gradient = _KeepAliveArray(_forward_power_gradient(payload["total"]), payload)
    return payload["grid_x"], payload["grid_y"], gradient


def fd_gradient_field(parameter: str, grid_size: int, n_rays: int, step: float):
    config = parameter_config(parameter)
    plus_cfg, minus_cfg = config["perturb"](step)
    _, _, power_p = trace_total_power(
        cube1_x=plus_cfg["cube1_x"],
        tx_pos=plus_cfg["tx_pos"],
        grid_size=grid_size,
        n_rays=n_rays,
    )
    _, _, power_m = trace_total_power(
        cube1_x=minus_cfg["cube1_x"],
        tx_pos=minus_cfg["tx_pos"],
        grid_size=grid_size,
        n_rays=n_rays,
    )
    return (power_p - power_m) / (2.0 * step)


def as_grid(array, grid_size: int):
    return np.asarray(array, dtype=np.float64).reshape(grid_size, grid_size)


def to_db_magnitude(array, *, floor_db: float = -160.0):
    db = 10.0 * np.log10(np.abs(array) + 1e-30)
    return np.maximum(db, floor_db)


def gradient_db_magnitude(array):
    safe_magnitude = np.where(np.isfinite(array), np.maximum(np.abs(array), 1e-20), np.nan)
    return 20.0 * np.log10(safe_magnitude)


def panel_stats_text(array):
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return "mean=nan med=nan std=nan"
    return (
        f"mean={float(np.mean(finite)):.2f} "
        f"med={float(np.median(finite)):.2f} "
        f"std={float(np.std(finite)):.2f}"
    )


def decorate_axis(ax, specs, tx_xy, title):
    for spec in specs:
        cx, cy = spec["center"]
        size = spec["size"]
        ax.add_patch(
            Rectangle((cx - size / 2.0, cy - size / 2.0), size, size, fill=False, edgecolor="black", linewidth=1.0)
        )
    ax.plot([tx_xy[0]], [tx_xy[1]], marker="*", markersize=8, color="gold", markeredgecolor="black")
    ax.set_xlim(TRACE_BOUNDS[0][0], TRACE_BOUNDS[0][1])
    ax.set_ylim(TRACE_BOUNDS[1][0], TRACE_BOUNDS[1][1])
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("x")
    ax.set_ylabel("y")


def plot_single_parameter(
    parameter: str,
    grid_size: int,
    n_rays: int,
    fd_step: float,
    output: Path,
    gradient_scale: str,
):
    config = parameter_config(parameter)
    _, _, total_power = trace_total_power(
        cube1_x=config["cube1_x"],
        tx_pos=config["tx_pos"],
        grid_size=grid_size,
        n_rays=n_rays,
    )
    total_np = as_grid(total_power, grid_size)
    total_db = 10.0 * np.log10(total_np + 1e-20)
    _, _, ad_grad = ad_gradient_field(parameter, grid_size, n_rays)
    ad_np = as_grid(ad_grad, grid_size)
    # Materialize AD before launching the FD pass. Otherwise the later trace can
    # perturb the lazy Dr.Jit buffer backing `ad_grad`, which changes the image.
    fd_grad = fd_gradient_field(parameter, grid_size, n_rays, fd_step)
    fd_np = as_grid(fd_grad, grid_size)
    diff_np = ad_np - fd_np

    extent = (
        float(TRACE_BOUNDS[0][0]),
        float(TRACE_BOUNDS[0][1]),
        float(TRACE_BOUNDS[1][0]),
        float(TRACE_BOUNDS[1][1]),
    )
    if gradient_scale == "db":
        ad_vis = gradient_db_magnitude(ad_np)
        fd_vis = gradient_db_magnitude(fd_np)
        diff_vis = ad_vis - fd_vis

        grad_vmax = max(
            np.percentile(ad_vis, 99.5),
            np.percentile(fd_vis, 99.5),
        )
        grad_vmin = grad_vmax - 60.0
        diff_vmax = max(np.percentile(np.abs(diff_vis), 99.5), 3.0)
        diff_vmin = -diff_vmax
        grad_cmap = "magma"
        diff_cmap = "RdBu_r"
        ad_title = f"AD |d|E|^2/d{config['label']}| (dB)\n{panel_stats_text(ad_vis)}"
        fd_title = f"FD |d|E|^2/d{config['label']}| (dB)\n{panel_stats_text(fd_vis)}"
        diff_title = f"AD - FD on gradient magnitude (dB)\n{panel_stats_text(diff_vis)}"
    else:
        ad_vis = ad_np
        fd_vis = fd_np
        diff_vis = diff_np
        grad_vmax = max(
            np.percentile(np.abs(ad_np), 99.5),
            np.percentile(np.abs(fd_np), 99.5),
            1e-12,
        )
        grad_vmin = -grad_vmax
        diff_vmax = max(np.percentile(np.abs(diff_np), 99.5), 1e-12)
        diff_vmin = -diff_vmax
        grad_cmap = "coolwarm"
        diff_cmap = "coolwarm"
        ad_title = f"AD d|E|^2/d{config['label']}\n{panel_stats_text(ad_vis)}"
        fd_title = f"FD d|E|^2/d{config['label']}\n{panel_stats_text(fd_vis)}"
        diff_title = f"AD - FD\n{panel_stats_text(diff_vis)}"

    fig, axes = plt.subplots(1, 4, figsize=(16, 4.6), constrained_layout=True)
    specs = cube_specs(config["cube1_x"])
    tx_xy = (config["tx_pos"][0], config["tx_pos"][1])

    im_total = axes[0].imshow(total_db, origin="lower", extent=extent, cmap="viridis", interpolation="nearest")
    decorate_axis(axes[0], specs, tx_xy, f"total power (dB)\n{panel_stats_text(total_db)}")

    im_ad = axes[1].imshow(
        ad_vis,
        origin="lower",
        extent=extent,
        cmap=grad_cmap,
        vmin=grad_vmin,
        vmax=grad_vmax,
        interpolation="nearest",
    )
    decorate_axis(axes[1], specs, tx_xy, ad_title)

    im_fd = axes[2].imshow(
        fd_vis,
        origin="lower",
        extent=extent,
        cmap=grad_cmap,
        vmin=grad_vmin,
        vmax=grad_vmax,
        interpolation="nearest",
    )
    decorate_axis(axes[2], specs, tx_xy, fd_title)

    im_diff = axes[3].imshow(
        diff_vis,
        origin="lower",
        extent=extent,
        cmap=diff_cmap,
        vmin=diff_vmin,
        vmax=diff_vmax,
        interpolation="nearest",
    )
    decorate_axis(axes[3], specs, tx_xy, diff_title)

    fig.colorbar(im_total, ax=axes[0], shrink=0.82)
    fig.colorbar(im_ad, ax=axes[1], shrink=0.82)
    fig.colorbar(im_fd, ax=axes[2], shrink=0.82)
    fig.colorbar(im_diff, ax=axes[3], shrink=0.82)

    fig.suptitle(
        f"Multipath Total Field And Gradient Fields For {config['label']}\n"
        f"grid={grid_size}, n_rays={n_rays}, fd_step={fd_step}, gradient_scale={gradient_scale}",
        fontsize=14,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)

    print(f"Saved: {output.resolve()}")
    print(
        f"{config['label']}: sum(AD)={float(ad_np.sum()):.6e}, "
        f"sum(FD)={float(fd_np.sum()):.6e}, "
        f"max|AD-FD|={float(np.max(np.abs(diff_np))):.6e}"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parameter", choices=("tx_x", "tx_y", "cube1_x"), default="tx_x")
    parser.add_argument("--grid-size", type=int, default=64)
    parser.add_argument("--n-rays", type=int, default=640)
    parser.add_argument("--fd-step", type=float, default=1e-3)
    parser.add_argument("--gradient-scale", choices=("linear", "db"), default="db")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("tests/output/grad_multipath_total_field_and_gradients.png"),
    )
    args = parser.parse_args()
    plot_single_parameter(
        args.parameter,
        args.grid_size,
        args.n_rays,
        args.fd_step,
        args.output,
        args.gradient_scale,
    )


if __name__ == "__main__":
    main()

