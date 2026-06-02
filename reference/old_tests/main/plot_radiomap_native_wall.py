"""Visualize native coherent radio-map parity on a wall-driven multipath scene."""

from __future__ import annotations

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

import witwin as wt
from tests._scene_helpers import box_geometry, build_scene
from witwin.channel import RadioMapMonitor, Tracer
from witwin.channel.monitors.radio_map.deterministic.trace import trace_radio_map_monitor
from .plot_radiomap_components import db_map
def _build_wall_scene():
    return build_scene(box_geometry(center=(0.0, 0.0, 1.5), size=(0.25, 6.0, 3.0)))


def _build_wall_tracer() -> Tracer:
    return Tracer(
        frequency=1.0e9,
        scene=_build_wall_scene(),
        reflection_n_rays=4096,
        reflection_max_bounces=1,
        max_diffractions=1,
    )


def _build_wall_monitor(grid_size: int) -> RadioMapMonitor:
    return RadioMapMonitor(
        "radiomap_native_wall",
        axis="z",
        position=1.5,
        bounds=((-4.0, 4.0), (-6.0, 6.0)),
        grid_shape=(int(grid_size), int(grid_size)),
        combine_mode="coherent",
        receiver_model="projected_polarized",
        metric="path_gain",
        ray_mode="3d",
    )


def _reshape_flat(values, *, grid_shape: tuple[int, int]) -> np.ndarray:
    nx, ny = (int(grid_shape[0]), int(grid_shape[1]))
    return np.asarray(values, dtype=np.float32).reshape((ny, nx))


def _decorate_axis(ax, *, title: str, bounds, tx_xy):
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.add_patch(
        Rectangle(
            (-0.125, -3.0),
            0.25,
            6.0,
            facecolor="none",
            edgecolor="white",
            linewidth=0.9,
            alpha=0.8,
        )
    )
    ax.scatter([tx_xy[0]], [tx_xy[1]], marker="*", s=60, c="white", edgecolors="black", linewidths=0.6)
    ax.set_xlim(bounds[0])
    ax.set_ylim(bounds[1])


def save_radiomap_native_wall_figure(
    output_path: Path,
    *,
    grid_size: int = 32,
) -> Path:
    tracer = _build_wall_tracer()
    monitor = _build_wall_monitor(grid_size)
    tx_pos = wt.Point3f(-3.0, -5.0, 1.5)
    result = tracer.trace(tx_pos, monitor=monitor, verbose=False)
    native_payload = result.monitor(monitor.name)
    solver_controls = tracer._resolve_monitor_solver_controls(
        monitor,
        execution_intent="radio_map_coherent",
    )
    baseline_payload = trace_radio_map_monitor(
        tx_pos,
        monitor,
        tracer.scene,
        tracer._resolved_trace_config,
        solver_controls,
        radio_map_accumulation_backend="baseline",
    )

    baseline_path_gain = _reshape_flat(
        baseline_payload["metrics"]["path_gain"],
        grid_shape=monitor.grid_shape,
    )
    native_path_gain = np.asarray(native_payload.path_gain, dtype=np.float32)
    reflection_power = np.asarray(native_payload.coherent_power["reflection"], dtype=np.float32)
    diffraction_power = np.asarray(native_payload.coherent_power["diffraction"], dtype=np.float32)
    total_diff = native_path_gain - baseline_path_gain
    extent = (
        float(monitor.bounds[0][0]),
        float(monitor.bounds[0][1]),
        float(monitor.bounds[1][0]),
        float(monitor.bounds[1][1]),
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 8.0), constrained_layout=True)

    im0 = axes[0, 0].imshow(db_map(baseline_path_gain), origin="lower", extent=extent, cmap="viridis")
    _decorate_axis(axes[0, 0], title="Baseline Path Gain (dB)", bounds=monitor.bounds, tx_xy=(-3.0, -5.0))
    fig.colorbar(im0, ax=axes[0, 0], fraction=0.046, pad=0.03)

    im1 = axes[0, 1].imshow(db_map(native_path_gain), origin="lower", extent=extent, cmap="viridis")
    _decorate_axis(axes[0, 1], title="Native Coherent Path Gain (dB)", bounds=monitor.bounds, tx_xy=(-3.0, -5.0))
    fig.colorbar(im1, ax=axes[0, 1], fraction=0.046, pad=0.03)

    diff_vmax = max(float(np.percentile(np.abs(total_diff), 99.0)), 1.0e-7)
    im2 = axes[0, 2].imshow(
        total_diff,
        origin="lower",
        extent=extent,
        cmap="coolwarm",
        vmin=-diff_vmax,
        vmax=diff_vmax,
    )
    _decorate_axis(axes[0, 2], title="Native - Baseline (Linear)", bounds=monitor.bounds, tx_xy=(-3.0, -5.0))
    fig.colorbar(im2, ax=axes[0, 2], fraction=0.046, pad=0.03)

    im3 = axes[1, 0].imshow(db_map(reflection_power), origin="lower", extent=extent, cmap="magma")
    _decorate_axis(axes[1, 0], title="Native Reflection Power (dB)", bounds=monitor.bounds, tx_xy=(-3.0, -5.0))
    fig.colorbar(im3, ax=axes[1, 0], fraction=0.046, pad=0.03)

    im4 = axes[1, 1].imshow(db_map(diffraction_power), origin="lower", extent=extent, cmap="magma")
    _decorate_axis(axes[1, 1], title="Native Diffraction Power (dB)", bounds=monitor.bounds, tx_xy=(-3.0, -5.0))
    fig.colorbar(im4, ax=axes[1, 1], fraction=0.046, pad=0.03)

    axes[1, 2].axis("off")
    axes[1, 2].text(
        0.0,
        1.0,
        "\n".join(
            [
                "Native Coherent Radio Map",
                f"grid={grid_size}x{grid_size}",
                f"max_abs_diff={float(np.max(np.abs(total_diff))):.3e}",
                f"backend={native_payload.metadata['accumulation_backend']['resolved']}",
                f"reflection={native_payload.metadata['runtime_backends']['reflection'].get('implementation', '')}",
                f"diffraction={native_payload.metadata['runtime_backends']['diffraction'].get('implementation', '')}",
                f"suffix={native_payload.metadata['runtime_backends']['suffix'].get('implementation', '')}",
            ]
        ),
        va="top",
        ha="left",
        fontsize=10,
        family="monospace",
    )

    fig.suptitle("RadioMap native coherent wall parity", fontsize=13)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


__all__ = ["save_radiomap_native_wall_figure"]
