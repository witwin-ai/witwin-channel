from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Sequence


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from benchmarks.fullwave_validation.models import FieldMap  # noqa: E402
from benchmarks.fullwave_validation.scenarios import (  # noqa: E402
    load_case,
    observation_valid_mask,
)


DEFAULT_OUTPUT_DIR = ROOT / "artifacts/fullwave/three-cube-metal"
DEFAULT_DETERMINISTIC_PATH = DEFAULT_OUTPUT_DIR / "deterministic.npz"
DEFAULT_PLOT_PATH = DEFAULT_OUTPUT_DIR / "three-cube-deterministic-components.png"
_COMPONENTS = ("los", "reflection", "diffraction")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot the deterministic three-cube PEC field and components."
    )
    parser.add_argument(
        "--deterministic",
        type=Path,
        default=DEFAULT_DETERMINISTIC_PATH,
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_PLOT_PATH)
    return parser


def _validate_current_case(field_map: FieldMap) -> None:
    spec = load_case("three_cube", "metal")
    expected = {
        "case_id": spec.case_id,
        "case_fingerprint": spec.fingerprint,
        "frequency_hz": spec.frequency_hz,
    }
    for key, value in expected.items():
        actual = field_map.metadata.get(key)
        if actual != value:
            raise ValueError(
                f"deterministic reference mismatch for {key}: "
                f"expected {value!r}, got {actual!r}"
            )
    missing = sorted(set(_COMPONENTS) - field_map.components.keys())
    if missing:
        raise ValueError(f"deterministic reference is missing components: {missing}")


def _db_relative(values: np.ndarray, peak: float) -> np.ndarray:
    floor = max(peak * 1.0e-3, 1.0e-30)
    return np.clip(
        20.0 * np.log10(np.maximum(np.abs(values), floor) / peak),
        -60.0,
        0.0,
    )


def _image_extent(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float, float]:
    dx = float(np.median(np.diff(x)))
    dy = float(np.median(np.diff(y)))
    return (
        float(x[0] - 0.5 * dx),
        float(x[-1] + 0.5 * dx),
        float(y[0] - 0.5 * dy),
        float(y[-1] + 0.5 * dy),
    )


def _add_scene_overlay(axis, spec) -> None:
    half_size = spec.cube_size_m / 2.0
    for center_x, center_y, _center_z in spec.cube_centers:
        axis.add_patch(
            plt.Rectangle(
                (center_x - half_size, center_y - half_size),
                spec.cube_size_m,
                spec.cube_size_m,
                fill=False,
                edgecolor="black",
                linewidth=1.0,
            )
        )
    axis.scatter(
        [spec.tx_position[0]],
        [spec.tx_position[1]],
        marker="*",
        s=70,
        c="gold",
        edgecolors="black",
        linewidths=0.8,
        zorder=5,
    )


def plot_three_cube_deterministic(
    deterministic: FieldMap,
    output: str | Path,
) -> Path:
    _validate_current_case(deterministic)
    spec = load_case("three_cube", "metal")
    valid = observation_valid_mask(spec, deterministic.x, deterministic.y)
    peak = float(np.max(np.abs(deterministic.field[valid])))
    if not np.isfinite(peak) or peak <= 0.0:
        raise ValueError("deterministic field must have finite non-zero valid energy")

    values = (
        deterministic.field,
        deterministic.components["los"],
        deterministic.components["reflection"],
        deterministic.components["diffraction"],
    )
    titles = ("Total |Ez|", "LoS |Ez|", "Reflection |Ez|", "Diffraction |Ez|")
    extent = _image_extent(deterministic.x, deterministic.y)
    colormap = plt.get_cmap("viridis").copy()
    colormap.set_bad("0.72")

    figure, axes = plt.subplots(2, 2, figsize=(12.0, 10.4), constrained_layout=True)
    images = []
    for axis, field, title in zip(axes.ravel(), values, titles, strict=True):
        field_db = _db_relative(field, peak)
        field_db[~valid] = np.nan
        image = axis.imshow(
            field_db,
            origin="lower",
            extent=extent,
            vmin=-60.0,
            vmax=0.0,
            cmap=colormap,
            interpolation="nearest",
            aspect="equal",
        )
        images.append(image)
        _add_scene_overlay(axis, spec)
        axis.set_title(title)
        axis.set_xlabel("x (m)")
        axis.set_ylabel("y (m)")
        axis.grid(False)

    figure.colorbar(
        images[0],
        ax=axes.ravel().tolist(),
        location="right",
        shrink=0.88,
        label="Magnitude relative to total-field peak (dB)",
    )
    figure.suptitle(
        "Three-cube deterministic field — 0.1× geometric layout from original "
        "channel; 5 GHz PEC benchmark\n"
        "Current deterministic solver (coupled R↔D unavailable; no full-wave "
        "reference loaded)",
        fontweight="normal",
    )
    output_path = Path(output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=220, facecolor="white")
    plt.close(figure)
    return output_path


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output = plot_three_cube_deterministic(
        FieldMap.load(args.deterministic),
        args.output,
    )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
