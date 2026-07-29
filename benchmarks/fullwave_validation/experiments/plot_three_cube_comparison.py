# Copyright Xingyu Chen.
# Renders the three-cube full-wave and deterministic comparison.

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
from matplotlib import colormaps
from PIL import Image, ImageDraw, ImageFont

from benchmarks.fullwave_validation.metrics import resample_regular
from benchmarks.fullwave_validation.models import FieldMap
from benchmarks.fullwave_validation.scenarios import load_case, observation_valid_mask


ROOT = Path(__file__).resolve().parents[3]
OUTPUT_DIR = Path(
    os.environ.get(
        "WITWIN_FULLWAVE_OUTPUT_DIR",
        ROOT / "artifacts/fullwave/three-cube-metal-320",
    )
).resolve()
_SCALE = 2


def _font(size: int):
    try:
        return ImageFont.truetype("arial.ttf", size)
    except OSError:
        return ImageFont.load_default()


def _db(values: np.ndarray) -> np.ndarray:
    return 20.0 * np.log10(np.maximum(np.abs(values), 1.0e-30))


def _image(
    values: np.ndarray, valid: np.ndarray, lower: float, upper: float, colormap: str,
) -> Image.Image:
    lut = colormaps.get_cmap(colormap)
    normalized = np.clip((values - lower) / (upper - lower), 0.0, 1.0)
    rgb = (lut(normalized)[..., :3] * 255.0).astype(np.uint8)
    rgb[~valid] = (180, 180, 180)
    image = Image.fromarray(rgb[::-1])
    return image.resize(
        (image.width * _SCALE, image.height * _SCALE), Image.Resampling.NEAREST
    )


def _overlay(image: Image.Image, x: np.ndarray, y: np.ndarray, label: str) -> Image.Image:
    spec = load_case("three_cube_320", "metal")
    draw = ImageDraw.Draw(image)

    def pixel(x_value: float, y_value: float) -> tuple[float, float]:
        horizontal = (x_value - x[0]) / (x[-1] - x[0]) * (image.width - 1)
        vertical = (1.0 - (y_value - y[0]) / (y[-1] - y[0])) * (image.height - 1)
        return horizontal, vertical

    for center_x, center_y, _ in spec.cube_centers:
        half = 0.5 * spec.cube_size_m
        points = [
            pixel(center_x - half, center_y - half),
            pixel(center_x + half, center_y - half),
            pixel(center_x + half, center_y + half),
            pixel(center_x - half, center_y + half),
            pixel(center_x - half, center_y - half),
        ]
        draw.line(points, fill=(255, 70, 70), width=2)
    tx_x, tx_y = pixel(spec.tx_position[0], spec.tx_position[1])
    draw.ellipse(
        [tx_x - 4, tx_y - 4, tx_x + 4, tx_y + 4],
        outline=(255, 255, 255),
        width=2,
    )
    draw.text((8, 6), label, fill="white", font=_font(18))
    return image


def plot_three_cube_comparison(output_dir: str | Path = OUTPUT_DIR) -> Path:
    base = Path(output_dir).resolve()
    report = json.loads((base / "three_cube_320_comparison.json").read_text())
    scale = float(report["s_empty"])
    fullwave = FieldMap.load(base / "visual-maxwell-metal-three-cube-5ghz-320.npz")
    coupled_off = FieldMap.load(base / "three_cube_320_coupled_off.npz")
    coupled_on = FieldMap.load(base / "three_cube_320_coupled_on.npz")
    spec = load_case("three_cube_320", "metal")
    valid = observation_valid_mask(spec, coupled_off.x, coupled_off.y)
    reference = resample_regular(fullwave, coupled_off.x, coupled_off.y).field
    reference_db = _db(reference)
    upper = float(reference_db[valid].max())
    lower = upper - 60.0

    panels = [
        _overlay(
            _image(reference_db, valid, lower, upper, "inferno"),
            coupled_off.x,
            coupled_off.y,
            "witwin-maxwell FDTD |Ez| dB",
        ),
        _overlay(
            _image(_db(scale * coupled_off.field), valid, lower, upper, "inferno"),
            coupled_off.x,
            coupled_off.y,
            "deterministic coupled OFF",
        ),
        _overlay(
            _image(_db(scale * coupled_on.field), valid, lower, upper, "inferno"),
            coupled_on.x,
            coupled_on.y,
            "deterministic coupled ON",
        ),
        _overlay(
            _image(_db(scale * coupled_off.field) - reference_db, valid, -15.0, 15.0, "coolwarm"),
            coupled_off.x,
            coupled_off.y,
            "gap OFF (det - truth, +/-15 dB)",
        ),
        _overlay(
            _image(_db(scale * coupled_on.field) - reference_db, valid, -15.0, 15.0, "coolwarm"),
            coupled_on.x,
            coupled_on.y,
            "gap ON (det - truth, +/-15 dB)",
        ),
    ]

    padding = 10
    panel_width, panel_height = panels[0].size
    sheet = Image.new(
        "RGB",
        (3 * panel_width + 4 * padding, 2 * panel_height + 3 * padding + 30),
        "white",
    )
    for index, panel in enumerate(panels):
        row, column = divmod(index, 3)
        sheet.paste(
            panel,
            (padding + column * (panel_width + padding), padding + row * (panel_height + padding)),
        )
    draw = ImageDraw.Draw(sheet)
    draw.text(
        (padding, 2 * panel_height + 2 * padding + 4),
        f"three_cube_320 metal 5 GHz; empty-field scale={scale:.4f}",
        fill="black",
        font=_font(14),
    )
    output = base / "three_cube_320_fullwave_vs_deterministic.png"
    sheet.save(output)
    return output


if __name__ == "__main__":
    print(plot_three_cube_comparison())