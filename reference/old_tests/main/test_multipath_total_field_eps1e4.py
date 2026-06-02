"""Visualize the multipath total-field map with eps_r=1e4."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
if os.environ.get("WITWIN_CHANNEL_MAIN_SHOW", "0") != "1":
    os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt

from tests.main.plot_multipath_components import (
    TRACE_BOUNDS,
    TX_POS,
    build_trace_payload,
    cube_specs,
    decorate_axis,
)
from witwin.channel import to_numpy
pytestmark = pytest.mark.gpu

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output"
OUTPUT_PATH = OUTPUT_DIR / "multipath_total_field_eps1e4.png"

CUBE1_X = -2.5
GRID_SIZE = 256
N_RAYS = 1_280


def _power_db_grid(field_component, *, grid_size: int) -> np.ndarray:
    real = np.asarray(to_numpy(field_component.real), dtype=np.float64)
    imag = np.asarray(to_numpy(field_component.imag), dtype=np.float64)
    power = real * real + imag * imag
    return 10.0 * np.log10(power.reshape(grid_size, grid_size) + 1e-20)


def _trace_total_field_db() -> tuple[np.ndarray, dict]:
    payload = build_trace_payload(
        cube1_x=CUBE1_X,
        tx_pos=TX_POS,
        grid_size=GRID_SIZE,
        n_rays=N_RAYS,
    )
    result = payload["result"].primary
    return _power_db_grid(result.field.total, grid_size=GRID_SIZE), dict(result.metadata["reflection_backend"])


def test_multipath_total_field_eps1e4_visual():
    total_db, reflection_backend = _trace_total_field_db()
    specs = cube_specs(CUBE1_X)
    tx_xy = (TX_POS[0], TX_POS[1])
    extent = (
        float(TRACE_BOUNDS[0][0]),
        float(TRACE_BOUNDS[0][1]),
        float(TRACE_BOUNDS[1][0]),
        float(TRACE_BOUNDS[1][1]),
    )

    fig, ax = plt.subplots(figsize=(7.2, 7.0), constrained_layout=True)
    try:
        image = ax.imshow(
            total_db,
            origin="lower",
            extent=extent,
            cmap="jet",
            vmin=-60.0,
            vmax=-20.0,
            interpolation="nearest",
        )
        decorate_axis(ax, specs, tx_xy, "Total Field (dB)\neps_r=1e4")
        fig.colorbar(image, ax=ax, shrink=0.88, label="Power [dB]")
        fig.suptitle(
            "Multipath Total Field\n"
            f"grid={GRID_SIZE}, n_rays={N_RAYS}, reflection={reflection_backend['backend']}",
            fontsize=14,
        )
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        fig.savefig(OUTPUT_PATH, dpi=180)
    finally:
        plt.close(fig)

    assert OUTPUT_PATH.exists()
    assert OUTPUT_PATH.stat().st_size > 0
