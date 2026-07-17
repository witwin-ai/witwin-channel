from __future__ import annotations

from dataclasses import asdict
import os
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from benchmarks.fullwave_validation.metrics import (  # noqa: E402
    analyze_boundaries,
    compare_magnitudes,
    resample_regular,
)
from benchmarks.fullwave_validation.models import FieldMap  # noqa: E402


OUTPUT_DIR = Path(
    os.environ.get(
        "WITWIN_FULLWAVE_OUTPUT_DIR",
        ROOT / "artifacts/fullwave/single-cube-metal-z042",
    )
).resolve()
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
DETERMINISTIC_PATH = OUTPUT_DIR / "visual-deterministic-metal-centered-5ghz-256.npz"
FULLWAVE_PATH = OUTPUT_DIR / "visual-maxwell-metal-centered-5ghz-256.npz"
EMPTY_DETERMINISTIC_PATH = OUTPUT_DIR / "visual-deterministic-empty-5ghz-256.npz"
EMPTY_FULLWAVE_PATH = OUTPUT_DIR / "visual-maxwell-empty-5ghz-256.npz"
OUTPUT_PATH = (
    OUTPUT_DIR
    / "single-cube-metal-centered-tx-z042-5ghz-256-empty-calibrated-6k.png"
)


def db_relative(values: np.ndarray, reference: float) -> np.ndarray:
    floor = max(reference * 1.0e-8, 1.0e-30)
    return np.clip(20.0 * np.log10(np.maximum(np.abs(values), floor) / reference), -60.0, 0.0)


def boundary_mask(component: np.ndarray) -> np.ndarray:
    magnitude = np.abs(component)
    return magnitude > float(magnitude.max()) * 1.0e-4


def strongest_boundary_profile(
    candidate: FieldMap,
    reference: FieldMap,
    component: str,
    *,
    valid_mask: np.ndarray,
    half_width: int = 7,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    mask = boundary_mask(candidate.components[component])
    cand_db = db_relative(candidate.field, float(np.abs(reference.field).max()))
    ref_db = db_relative(reference.field, float(np.abs(reference.field).max()))
    x_edges = mask[:, 1:] != mask[:, :-1]
    y_edges = mask[1:, :] != mask[:-1, :]
    x_edges &= valid_mask[:, 1:] & valid_mask[:, :-1]
    y_edges &= valid_mask[1:, :] & valid_mask[:-1, :]

    x_jump = np.where(x_edges, np.abs(cand_db[:, 1:] - cand_db[:, :-1]), -np.inf)
    y_jump = np.where(y_edges, np.abs(cand_db[1:, :] - cand_db[:-1, :]), -np.inf)
    if float(np.max(x_jump)) >= float(np.max(y_jump)):
        row, left = np.unravel_index(int(np.argmax(x_jump)), x_jump.shape)
        center = left + 1
        lo, hi = max(0, center - half_width), min(candidate.x.size, center + half_width + 1)
        coord = candidate.x[lo:hi] - 0.5 * (candidate.x[left] + candidate.x[left + 1])
        return coord, cand_db[row, lo:hi], ref_db[row, lo:hi], f"y={candidate.y[row]:+.2f} m"

    lower, col = np.unravel_index(int(np.argmax(y_jump)), y_jump.shape)
    center = lower + 1
    lo, hi = max(0, center - half_width), min(candidate.y.size, center + half_width + 1)
    coord = candidate.y[lo:hi] - 0.5 * (candidate.y[lower] + candidate.y[lower + 1])
    return coord, cand_db[lo:hi, col], ref_db[lo:hi, col], f"x={candidate.x[col]:+.2f} m"


def main() -> None:
    deterministic = FieldMap.load(DETERMINISTIC_PATH)
    fullwave = FieldMap.load(FULLWAVE_PATH)
    empty_deterministic = FieldMap.load(EMPTY_DETERMINISTIC_PATH)
    empty_fullwave = FieldMap.load(EMPTY_FULLWAVE_PATH)
    aligned = resample_regular(fullwave, deterministic.x, deterministic.y)
    xx, yy = np.meshgrid(deterministic.x, deterministic.y)
    valid_samples = ~((np.abs(xx) < 0.1) & (np.abs(yy) < 0.1))
    baseline_metrics = compare_magnitudes(
        empty_deterministic,
        empty_fullwave,
        valid_mask=valid_samples,
    )
    scale = baseline_metrics.amplitude_scale
    magnitude_metrics = compare_magnitudes(
        deterministic,
        fullwave,
        valid_mask=valid_samples,
        amplitude_scale=scale,
    )
    boundary_metrics = analyze_boundaries(
        deterministic,
        fullwave,
        valid_mask=valid_samples,
    )
    calibrated = scale * deterministic.field
    calibrated_map = FieldMap(
        x=deterministic.x,
        y=deterministic.y,
        field=calibrated,
        components={name: scale * values for name, values in deterministic.components.items()},
        metadata=deterministic.metadata,
    )
    peak = float(np.abs(aligned.field).max())

    extent = (
        float(deterministic.x[0]),
        float(deterministic.x[-1]),
        float(deterministic.y[0]),
        float(deterministic.y[-1]),
    )
    det_db = db_relative(calibrated, peak)
    full_db = db_relative(aligned.field, peak)
    residual_db = db_relative(np.abs(calibrated) - np.abs(aligned.field), peak)
    det_db[~valid_samples] = np.nan
    residual_db[~valid_samples] = np.nan

    plt.rcParams.update(
        {
            "font.family": ["Microsoft YaHei", "DejaVu Sans"],
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
            "figure.titlesize": 16,
            "axes.unicode_minus": False,
        }
    )
    figure = plt.figure(figsize=(19.2, 10.8), constrained_layout=True)
    grid = figure.add_gridspec(2, 3, height_ratios=(1.05, 0.95))

    field_axes = [figure.add_subplot(grid[0, index]) for index in range(3)]
    deterministic_cmap = plt.get_cmap("viridis").copy()
    deterministic_cmap.set_bad("0.72")
    residual_cmap = plt.get_cmap("magma").copy()
    residual_cmap.set_bad("0.72")
    images = []
    for axis, values, title, cmap in zip(
        field_axes,
        (det_db, full_db, residual_db),
        (
            f"Deterministic channel |h| (empty-space calibrated ×{scale:.2f})",
            "witwin-maxwell FDTD |Ez|",
            "Magnitude residual ||s·h| − |Ez||",
        ),
        (deterministic_cmap, "viridis", residual_cmap),
        strict=True,
    ):
        image = axis.imshow(
            values,
            origin="lower",
            extent=extent,
            vmin=-60.0,
            vmax=0.0,
            cmap=cmap,
            interpolation="nearest",
            aspect="equal",
        )
        images.append(image)
        axis.add_patch(
            plt.Rectangle((-0.1, -0.1), 0.2, 0.2, fill=False, color="white", linewidth=1.5)
        )
        axis.scatter([-0.2], [-0.5], marker="*", s=75, c="white", edgecolors="black", linewidths=0.5)
        axis.set_title(title)
        axis.set_xlabel("x (m)")
        axis.set_ylabel("y (m)")
        axis.grid(False)

    figure.colorbar(images[0], ax=field_axes[:2], location="right", shrink=0.88, label="Normalized magnitude (dB)")
    figure.colorbar(images[2], ax=field_axes[2], location="right", shrink=0.88, label="Magnitude residual / FDTD peak (dB)")

    edge_axis = figure.add_subplot(grid[1, 0])
    edge_image = edge_axis.imshow(
        det_db,
        origin="lower",
        extent=extent,
        vmin=-60.0,
        vmax=0.0,
        cmap="Greys",
        interpolation="nearest",
        aspect="equal",
    )
    isb_mask = boundary_mask(deterministic.components["los"])
    rsb_mask = boundary_mask(deterministic.components["reflection"])
    edge_axis.contour(deterministic.x, deterministic.y, isb_mask, levels=[0.5], colors=["#00c8ff"], linewidths=1.7)
    edge_axis.contour(deterministic.x, deterministic.y, rsb_mask, levels=[0.5], colors=["#ff4fa3"], linewidths=1.7)
    edge_axis.add_patch(plt.Rectangle((-0.1, -0.1), 0.2, 0.2, fill=False, color="white", linewidth=1.5))
    edge_axis.scatter([-0.2], [-0.5], marker="*", s=75, c="white", edgecolors="black", linewidths=0.5)
    edge_axis.plot([], [], color="#00c8ff", linewidth=2, label="ISB support edge")
    edge_axis.plot([], [], color="#ff4fa3", linewidth=2, label="RSB support edge")
    edge_axis.set_title("Detected deterministic support boundaries")
    edge_axis.set_xlabel("x (m)")
    edge_axis.set_ylabel("y (m)")
    edge_axis.legend(loc="lower right", framealpha=0.9)
    figure.colorbar(edge_image, ax=edge_axis, location="right", shrink=0.83, label="Calibrated deterministic |h| (dB)")

    bar_axis = figure.add_subplot(grid[1, 1])
    categories = ["ISB p95", "ISB max", "RSB p95", "RSB max"]
    det_values = [
        boundary_metrics["ISB"].deterministic_jump_db_p95,
        boundary_metrics["ISB"].deterministic_jump_db_max,
        boundary_metrics["RSB"].deterministic_jump_db_p95,
        boundary_metrics["RSB"].deterministic_jump_db_max,
    ]
    fdtd_values = [
        boundary_metrics["ISB"].fullwave_jump_db_p95,
        boundary_metrics["ISB"].fullwave_jump_db_max,
        boundary_metrics["RSB"].fullwave_jump_db_p95,
        boundary_metrics["RSB"].fullwave_jump_db_max,
    ]
    positions = np.arange(len(categories))
    width = 0.36
    det_bars = bar_axis.bar(positions - width / 2, det_values, width, label="Deterministic", color="#3366cc")
    fdtd_bars = bar_axis.bar(positions + width / 2, fdtd_values, width, label="FDTD", color="#dc3912")
    bar_axis.bar_label(det_bars, fmt="%.2f", padding=3)
    bar_axis.bar_label(fdtd_bars, fmt="%.2f", padding=3)
    bar_axis.set_xticks(positions, categories)
    bar_axis.set_ylabel("Adjacent-cell magnitude jump (dB)")
    bar_axis.set_yscale("symlog", linthresh=10.0, linscale=1.0)
    bar_axis.set_ylim(0.0, max(det_values + fdtd_values) * 1.35)
    bar_axis.set_title(
        "ISB / RSB jump statistics\n"
        f"p95 excess: ISB {boundary_metrics['ISB'].p95_excess_jump_db:+.2f} dB  |  "
        f"RSB {boundary_metrics['RSB'].p95_excess_jump_db:+.2f} dB"
    )
    bar_axis.grid(axis="y", alpha=0.25)
    bar_axis.legend(loc="upper left")
    profile_axis = figure.add_subplot(grid[1, 2])
    colors = {"ISB": "#00a6d6", "RSB": "#d62976"}
    for kind, component in (("ISB", "los"), ("RSB", "reflection")):
        offset, det_profile, fdtd_profile, slice_label = strongest_boundary_profile(
            calibrated_map,
            aligned,
            component,
            valid_mask=valid_samples,
        )
        profile_axis.plot(
            offset,
            det_profile,
            color=colors[kind],
            linewidth=2.0,
            label=f"{kind} Deterministic ({slice_label})",
        )
        profile_axis.plot(
            offset,
            fdtd_profile,
            color=colors[kind],
            linewidth=2.0,
            linestyle="--",
            label=f"{kind} FDTD ({slice_label})",
        )
    profile_axis.axvline(0.0, color="black", linewidth=1.0, alpha=0.55)
    profile_axis.set_title("Local profiles through strongest deterministic edge")
    profile_axis.set_xlabel("Offset from support edge (m)")
    profile_axis.set_ylabel("Normalized |Ez| (dB)")
    profile_axis.set_ylim(-60.0, 2.0)
    profile_axis.grid(alpha=0.25)
    profile_axis.legend(loc="best")

    figure.suptitle(
        "Single centered PEC cube (side 0.2 m), Tx z=0.42 m, no transmission, 5 GHz, 256×256 samples\n"
        "Empty-space calibrated envelope comparison (fixed scale; source units differ) — "
        f"scale={scale:.2f}×, post-calibration energy ratio="
        f"{magnitude_metrics.calibrated_reference_to_candidate_energy_ratio:.3f}×, "
        f"envelope NMSE={magnitude_metrics.calibrated_magnitude_nmse:.4f}, "
        f"correlation={magnitude_metrics.magnitude_correlation:.3f}, "
        f"RMSE={magnitude_metrics.magnitude_rmse_db:.2f} dB",
        fontweight="normal",
    )
    figure.savefig(OUTPUT_PATH, dpi=320, facecolor="white")
    plt.close(figure)
    print(OUTPUT_PATH)
    print({"empty_baseline": asdict(baseline_metrics)})
    print(asdict(magnitude_metrics))
    print({key: asdict(value) for key, value in boundary_metrics.items()})


if __name__ == "__main__":
    main()
