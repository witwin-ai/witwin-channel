"""Helpers for RadioMapMonitor visual tests on the multipath scene."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

import witwin as wt
from tests._scene_helpers import box_drjit_geometry, mesh_structure
from witwin.channel import Material, RadioMapMonitor, Scene, Tracer
from .plot_multipath_components import (
    CUBE1_BASE_CENTER,
    CUBE2_CENTER,
    CUBE3_CENTER,
    CUBE_SIZE,
    TRACE_BOUNDS,
    TX_POS,
    build_scene_for_cube1_x,
    cube_specs,
)


def make_tracer(scene, n_rays: int) -> Tracer:
    return Tracer(
        frequency=1e9,
        scene=scene,
        reflection_n_rays=n_rays,
        reflection_max_bounces=3,
        reflection_coef=0.8,
        enable_rd_diffraction=True,
        max_diffractions=2,
    )


def make_monitor(
    grid_size: int,
    *,
    metric: str = "path_gain",
    combine_mode: str = "incoherent",
    quadrature_mode: str = "center",
    tx_power: float = 1.0,
    noise_power: float | None = None,
):
    return RadioMapMonitor(
        "radiomap_main",
        axis="z",
        position=TX_POS[2],
        bounds=TRACE_BOUNDS,
        grid_shape=(int(grid_size), int(grid_size)),
        metric=metric,
        combine_mode=combine_mode,
        quadrature_mode=quadrature_mode,
        tx_power=tx_power,
        noise_power=noise_power,
    )


def _unit_material() -> Material:
    return Material()


def _eps_material(eps_r: float) -> Material:
    return Material(eps_r=float(eps_r), sigma_e=0.0)


@lru_cache(maxsize=16)
def _build_scene_for_cube1_x_with_right_cube_eps_cached(
    cube1_x: float,
    right_cube_eps_r: float,
):
    cube1_center = wt.Point3f(cube1_x, CUBE1_BASE_CENTER[1], CUBE1_BASE_CENTER[2])
    cube1 = box_drjit_geometry(center=cube1_center, size=CUBE_SIZE, rotation=None)
    cube2 = box_drjit_geometry(center=CUBE2_CENTER, size=CUBE_SIZE, rotation=None)
    cube3 = box_drjit_geometry(center=CUBE3_CENTER, size=CUBE_SIZE, rotation=None)
    return Scene(
        structures=[
            mesh_structure(cube1, name="cube_left", material=_unit_material()),
            mesh_structure(cube2, name="cube_right", material=_eps_material(right_cube_eps_r)),
            mesh_structure(cube3, name="cube_top", material=_unit_material()),
        ]
    )


def build_radiomap_scene(
    cube1_x: float,
    *,
    right_cube_eps_r: float | None = None,
):
    if right_cube_eps_r is None:
        return build_scene_for_cube1_x(cube1_x)
    return _build_scene_for_cube1_x_with_right_cube_eps_cached(
        float(cube1_x),
        float(right_cube_eps_r),
    )


def trace_radio_map(
    *,
    cube1_x: float,
    tx_pos,
    grid_size: int,
    n_rays: int,
    metric: str = "path_gain",
    combine_mode: str = "incoherent",
    quadrature_mode: str = "center",
    tx_power: float = 1.0,
    noise_power: float | None = None,
    right_cube_eps_r: float | None = None,
    scene=None,
    tracer=None,
):
    scene = (
        build_radiomap_scene(cube1_x, right_cube_eps_r=right_cube_eps_r)
        if scene is None
        else scene
    )
    tracer = make_tracer(scene, n_rays) if tracer is None else tracer
    monitor = make_monitor(
        grid_size,
        metric=metric,
        combine_mode=combine_mode,
        quadrature_mode=quadrature_mode,
        tx_power=tx_power,
        noise_power=noise_power,
    )
    payload = tracer.trace(wt.Point3f(*tx_pos), monitor=monitor, verbose=False).monitor(monitor.name)
    return {
        "scene": scene,
        "tracer": tracer,
        "monitor": monitor,
        "payload": payload,
    }


def trace_radio_map_many(
    *,
    cube1_x: float,
    tx_positions,
    tx_labels,
    grid_size: int,
    n_rays: int,
    metric: str = "sinr",
    combine_mode: str = "incoherent",
    quadrature_mode: str = "center",
    tx_power: float = 1.0,
    noise_power: float | None = None,
    right_cube_eps_r: float | None = None,
):
    scene = build_radiomap_scene(cube1_x, right_cube_eps_r=right_cube_eps_r)
    tracer = make_tracer(scene, n_rays)
    monitor = make_monitor(
        grid_size,
        metric=metric,
        combine_mode=combine_mode,
        quadrature_mode=quadrature_mode,
        tx_power=tx_power,
        noise_power=noise_power,
    )
    requests = [
        {"tx_pos": wt.Point3f(*tx_pos), "tx_label": str(label)}
        for tx_pos, label in zip(tx_positions, tx_labels)
    ]
    results = tracer.trace_many(requests, monitor=monitor, verbose=False)
    return {
        "scene": scene,
        "tracer": tracer,
        "monitor": monitor,
        "payloads": [result.monitor(monitor.name) for result in results],
    }


def as_numpy_grid(value) -> np.ndarray:
    return np.asarray(value, dtype=np.float32)


def db_map(values, *, eps: float = 1.0e-20) -> np.ndarray:
    return 10.0 * np.log10(np.maximum(values, eps))


def component_db_map(values, *, dynamic_range_db: float = 60.0) -> np.ndarray:
    values_np = as_numpy_grid(values)
    peak = float(np.max(values_np)) if values_np.size > 0 else 0.0
    if peak <= 0.0:
        floor = 10.0 ** (-float(dynamic_range_db) / 10.0)
        return db_map(np.full_like(values_np, floor, dtype=np.float32), eps=floor)
    floor = peak * (10.0 ** (-float(dynamic_range_db) / 10.0))
    return db_map(np.maximum(values_np, floor), eps=floor)


def decorate_axis(ax, *, title: str, cube1_x: float, tx_xy):
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    for spec in cube_specs(cube1_x):
        cx, cy = spec["center"]
        size = spec["size"]
        ax.add_patch(
            Rectangle(
                (cx - 0.5 * size, cy - 0.5 * size),
                size,
                size,
                facecolor="none",
                edgecolor="white",
                linewidth=0.8,
                alpha=0.7,
            )
        )
    ax.scatter([tx_xy[0]], [tx_xy[1]], marker="*", s=60, c="white", edgecolors="black", linewidths=0.6)


def save_radiomap_main_figure(
    output_path: Path,
    *,
    grid_size: int = 56,
    n_rays: int = 384,
) -> Path:
    cube1_x = CUBE1_BASE_CENTER[0]
    single_tx = TX_POS
    left_tx = (-2.0, -4.0, 1.5)
    right_tx = (2.0, -4.0, 1.5)

    incoherent = trace_radio_map(
        cube1_x=cube1_x,
        tx_pos=single_tx,
        grid_size=grid_size,
        n_rays=n_rays,
        metric="path_gain",
        combine_mode="incoherent",
    )["payload"]
    coherent = trace_radio_map(
        cube1_x=cube1_x,
        tx_pos=single_tx,
        grid_size=grid_size,
        n_rays=n_rays,
        metric="path_gain",
        combine_mode="coherent",
    )["payload"]
    multi_tx = trace_radio_map_many(
        cube1_x=cube1_x,
        tx_positions=(left_tx, right_tx),
        tx_labels=("left", "right"),
        grid_size=grid_size,
        n_rays=n_rays,
        metric="sinr",
        combine_mode="incoherent",
        tx_power=2.0,
        noise_power=1.0e-9,
    )["payloads"]
    left_payload = multi_tx[0]

    incoherent_db = db_map(as_numpy_grid(incoherent.path_gain))
    coherent_db = db_map(as_numpy_grid(coherent.path_gain))
    combine_delta = coherent_db - incoherent_db
    sinr_db = db_map(as_numpy_grid(left_payload.sinr), eps=1.0e-12)
    association = np.asarray(left_payload.tx_association(), dtype=np.int32)
    sampled = left_payload.sample_metric_positions(
        96,
        tx_association="left",
        seed=7,
        jitter=True,
    ).detach().cpu().numpy()

    extent = (
        float(TRACE_BOUNDS[0][0]),
        float(TRACE_BOUNDS[0][1]),
        float(TRACE_BOUNDS[1][0]),
        float(TRACE_BOUNDS[1][1]),
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 3, figsize=(14.5, 8.5), constrained_layout=True)

    im0 = axes[0, 0].imshow(incoherent_db, origin="lower", extent=extent, cmap="viridis")
    decorate_axis(axes[0, 0], title="Path Gain (Incoherent, dB)", cube1_x=cube1_x, tx_xy=single_tx[:2])
    fig.colorbar(im0, ax=axes[0, 0], fraction=0.046, pad=0.03)

    im1 = axes[0, 1].imshow(coherent_db, origin="lower", extent=extent, cmap="viridis")
    decorate_axis(axes[0, 1], title="Path Gain (Coherent, dB)", cube1_x=cube1_x, tx_xy=single_tx[:2])
    fig.colorbar(im1, ax=axes[0, 1], fraction=0.046, pad=0.03)

    delta_vmax = max(3.0, float(np.percentile(np.abs(combine_delta), 99.0)))
    im2 = axes[0, 2].imshow(
        combine_delta,
        origin="lower",
        extent=extent,
        cmap="coolwarm",
        vmin=-delta_vmax,
        vmax=delta_vmax,
    )
    decorate_axis(axes[0, 2], title="Coherent - Incoherent (dB)", cube1_x=cube1_x, tx_xy=single_tx[:2])
    fig.colorbar(im2, ax=axes[0, 2], fraction=0.046, pad=0.03)

    im3 = axes[1, 0].imshow(sinr_db, origin="lower", extent=extent, cmap="magma")
    decorate_axis(axes[1, 0], title="Left-TX SINR (dB)", cube1_x=cube1_x, tx_xy=left_tx[:2])
    fig.colorbar(im3, ax=axes[1, 0], fraction=0.046, pad=0.03)

    im4 = axes[1, 1].imshow(association, origin="lower", extent=extent, cmap="tab10", vmin=0, vmax=1)
    decorate_axis(axes[1, 1], title="TX Association", cube1_x=cube1_x, tx_xy=left_tx[:2])
    axes[1, 1].scatter(sampled[:, 0], sampled[:, 1], s=8, c="white", alpha=0.55, linewidths=0.0)
    fig.colorbar(im4, ax=axes[1, 1], fraction=0.046, pad=0.03)

    diff_linear = as_numpy_grid(coherent.path_gain) - as_numpy_grid(incoherent.path_gain)
    diff_vmax = max(float(np.percentile(np.abs(diff_linear), 99.0)), 1.0e-6)
    im5 = axes[1, 2].imshow(
        diff_linear,
        origin="lower",
        extent=extent,
        cmap="coolwarm",
        vmin=-diff_vmax,
        vmax=diff_vmax,
    )
    decorate_axis(axes[1, 2], title="Coherent - Incoherent (Linear)", cube1_x=cube1_x, tx_xy=single_tx[:2])
    fig.colorbar(im5, ax=axes[1, 2], fraction=0.046, pad=0.03)

    fig.suptitle(
        f"RadioMap multipath diagnostics ({grid_size}x{grid_size}, rays={n_rays})",
        fontsize=13,
    )
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def save_radiomap_component_split_figure(
    output_path: Path,
    *,
    grid_size: int = 56,
    n_rays: int = 384,
    right_cube_eps_r: float | None = None,
) -> Path:
    cube1_x = CUBE1_BASE_CENTER[0]
    single_tx = TX_POS
    payload = trace_radio_map(
        cube1_x=cube1_x,
        tx_pos=single_tx,
        grid_size=grid_size,
        n_rays=n_rays,
        metric="path_gain",
        combine_mode="incoherent",
        right_cube_eps_r=right_cube_eps_r,
    )["payload"]

    total = as_numpy_grid(payload.path_gain)
    los = as_numpy_grid(payload.incoherent["los"])
    reflection = as_numpy_grid(payload.incoherent["reflection"])
    diffraction = as_numpy_grid(payload.incoherent["diffraction"])
    reflection_ratio_db = 10.0 * np.log10(np.maximum(reflection, 1.0e-20) / np.maximum(total, 1.0e-20))
    diffraction_ratio_db = 10.0 * np.log10(np.maximum(diffraction, 1.0e-20) / np.maximum(total, 1.0e-20))

    extent = (
        float(TRACE_BOUNDS[0][0]),
        float(TRACE_BOUNDS[0][1]),
        float(TRACE_BOUNDS[1][0]),
        float(TRACE_BOUNDS[1][1]),
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 3, figsize=(14.5, 8.5), constrained_layout=True)

    panels = (
        ("Total Path Gain (dB)", component_db_map(total), "viridis", None),
        ("LoS Power (dB, local scale)", component_db_map(los), "viridis", None),
        ("Reflection Power (dB, local scale)", component_db_map(reflection), "magma", None),
        ("Diffraction Power (dB, local scale)", component_db_map(diffraction), "magma", None),
        ("Reflection / Total (dB)", reflection_ratio_db, "coolwarm", (-35.0, 0.0)),
        ("Diffraction / Total (dB)", diffraction_ratio_db, "coolwarm", (-35.0, 0.0)),
    )

    for ax, (title, image, cmap, limits) in zip(axes.reshape(-1), panels):
        kwargs = {"origin": "lower", "extent": extent, "cmap": cmap}
        if limits is not None:
            kwargs["vmin"] = float(limits[0])
            kwargs["vmax"] = float(limits[1])
        im = ax.imshow(image, **kwargs)
        decorate_axis(ax, title=title, cube1_x=cube1_x, tx_xy=single_tx[:2])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)

    title_suffix = (
        ""
        if right_cube_eps_r is None
        else f", right cube eps_r={float(right_cube_eps_r):.0e}"
    )
    fig.suptitle(
        f"RadioMap split multipath components ({grid_size}x{grid_size}, rays={n_rays}{title_suffix})",
        fontsize=13,
    )
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


__all__ = [
    "as_numpy_grid",
    "component_db_map",
    "db_map",
    "make_monitor",
    "make_tracer",
    "save_radiomap_component_split_figure",
    "save_radiomap_main_figure",
    "trace_radio_map",
    "trace_radio_map_many",
]
