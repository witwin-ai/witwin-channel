"""Visualize forward multipath total, reflection, and diffraction fields."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
if os.environ.get("WITWIN_CHANNEL_MAIN_SHOW", "0") != "1":
    os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt

import witwin as wt
from tests._scene_helpers import box_drjit_geometry, build_scene as build_test_scene
from tests.main.plot_multipath_components import (
    TRACE_BOUNDS,
    TX_POS,
    cube_specs,
    decorate_axis,
)
from witwin.channel import FieldMonitor, Material, Tracer, to_numpy
pytestmark = pytest.mark.gpu

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output"
OUTPUT_PATH = OUTPUT_DIR / "multipath_forward_polarization.png"

CUBE1_X = -2.5
EPS_R = 100.0
GRID_SIZE = 256
N_RAYS = 1_280
FREQUENCY = 1e9
TX_POLARIZATIONS = (
    ("Tx=Ex", (1.0, 0.0, 0.0)),
    ("Tx=Ey", (0.0, 1.0, 0.0)),
)
TRACE_CONFIG = {
    "trace": {
        "reflection_field_backend": "native",
        "diffraction_execution": {
            "suffix_backend": "native",
            "suffix_dda": "symbolic",
        }
    }
}


def _power_grid(field_component, *, grid_size: int) -> np.ndarray:
    real = np.asarray(to_numpy(field_component.real), dtype=np.float64)
    imag = np.asarray(to_numpy(field_component.imag), dtype=np.float64)
    return (real * real + imag * imag).reshape(grid_size, grid_size)


def _power_db_grid(power: np.ndarray, *, floor_db: float = -120.0) -> np.ndarray:
    return np.maximum(10.0 * np.log10(np.asarray(power, dtype=np.float64) + 1e-20), floor_db)


def _build_scene():
    cube1 = box_drjit_geometry(center=wt.Point3f(CUBE1_X, -3.0, 1.5), size=2.0, rotation=None).to_mesh()
    cube2 = box_drjit_geometry(center=wt.Point3f(2.0, 0.5, 1.5), size=2.0, rotation=None).to_mesh()
    cube3 = box_drjit_geometry(center=wt.Point3f(-0.5, 3.5, 1.5), size=2.0, rotation=None).to_mesh()
    return build_test_scene(
        cube1,
        cube2,
        cube3,
        material=Material(eps_r=EPS_R, sigma_e=0.0),
    )


def _trace_payload(*, tx_polarization) -> dict:
    scene = _build_scene()
    discover_monitor = FieldMonitor(
        "multipath_forward_polarization_discover",
        axis="z",
        position=TX_POS[2],
        bounds=TRACE_BOUNDS,
        grid_size=GRID_SIZE,
    )
    replay_monitor = FieldMonitor(
        "multipath_forward_polarization_replay",
        axis="z",
        position=TX_POS[2],
        bounds=TRACE_BOUNDS,
        grid_size=GRID_SIZE,
    )
    tracer = Tracer(
        frequency=FREQUENCY,
        scene=scene,
        config=TRACE_CONFIG,
        reflection_n_rays=N_RAYS,
        reflection_max_bounces=3,
        reflection_coef=0.8,
        enable_rd_diffraction=True,
        max_diffractions=2,
        tx_polarization=tx_polarization,
    )
    result = tracer.trace(
        wt.Point3f(*TX_POS),
        monitor=[discover_monitor, replay_monitor],
        verbose=False,
        return_diffraction_audit=False,
    )
    replay_result = result["multipath_forward_polarization_replay"]
    return {
        "scene": scene,
        "result": replay_result,
        "reflection_backend": dict(replay_result.metadata["reflection_backend"]),
        "scalar_total_power": _power_grid(replay_result.field.total, grid_size=GRID_SIZE),
        "scalar_reflection_power": _power_grid(replay_result.field.reflection, grid_size=GRID_SIZE),
        "scalar_diffraction_power": _power_grid(replay_result.field.diffraction, grid_size=GRID_SIZE),
        "jones_x_total_power": _power_grid(replay_result.jones.total["x"], grid_size=GRID_SIZE),
        "jones_x_reflection_power": _power_grid(replay_result.jones.reflection["x"], grid_size=GRID_SIZE),
        "jones_x_diffraction_power": _power_grid(replay_result.jones.diffraction["x"], grid_size=GRID_SIZE),
        "jones_y_total_power": _power_grid(replay_result.jones.total["y"], grid_size=GRID_SIZE),
        "jones_y_reflection_power": _power_grid(replay_result.jones.reflection["y"], grid_size=GRID_SIZE),
        "jones_y_diffraction_power": _power_grid(replay_result.jones.diffraction["y"], grid_size=GRID_SIZE),
    }


def test_multipath_forward_field_polarization_visual():
    base_payloads = [(_label, _trace_payload(tx_polarization=tx_pol)) for _label, tx_pol in TX_POLARIZATIONS]
    ex_payload = base_payloads[0][1]
    ey_payload = base_payloads[1][1]
    payloads = [
        base_payloads[0],
        (
            "Tx=Ex + Tx=Ey",
            {
                "reflection_backend": ex_payload["reflection_backend"],
                "scalar_total_power": ex_payload["scalar_total_power"] + ey_payload["scalar_total_power"],
                "scalar_reflection_power": ex_payload["scalar_reflection_power"] + ey_payload["scalar_reflection_power"],
                "scalar_diffraction_power": ex_payload["scalar_diffraction_power"] + ey_payload["scalar_diffraction_power"],
                "jones_x_total_power": ex_payload["jones_x_total_power"] + ey_payload["jones_x_total_power"],
                "jones_x_reflection_power": ex_payload["jones_x_reflection_power"] + ey_payload["jones_x_reflection_power"],
                "jones_x_diffraction_power": ex_payload["jones_x_diffraction_power"] + ey_payload["jones_x_diffraction_power"],
                "jones_y_total_power": ex_payload["jones_y_total_power"] + ey_payload["jones_y_total_power"],
                "jones_y_reflection_power": ex_payload["jones_y_reflection_power"] + ey_payload["jones_y_reflection_power"],
                "jones_y_diffraction_power": ex_payload["jones_y_diffraction_power"] + ey_payload["jones_y_diffraction_power"],
            },
        )
    ]
    specs = cube_specs(CUBE1_X)
    tx_xy = (TX_POS[0], TX_POS[1])
    extent = (
        float(TRACE_BOUNDS[0][0]),
        float(TRACE_BOUNDS[0][1]),
        float(TRACE_BOUNDS[1][0]),
        float(TRACE_BOUNDS[1][1]),
    )
    reflection_backend = ex_payload["reflection_backend"]
    reflection_backend_note = (
        "reflection="
        f"{reflection_backend['backend']} / {reflection_backend['implementation']} "
        f"(fresh trace, epc={reflection_backend['use_epc']})"
    )

    fig, axes = plt.subplots(len(payloads), 9, figsize=(23.0, 7.6), constrained_layout=True, squeeze=False)
    try:
        colorbar_handle = None
        for row_idx, (row_label, payload) in enumerate(payloads):
            panel_defs = (
                ("Scalar Total", "total", _power_db_grid(payload["scalar_total_power"])),
                ("Scalar Reflection", "reflection", _power_db_grid(payload["scalar_reflection_power"])),
                ("Scalar Diffraction", "diffraction", _power_db_grid(payload["scalar_diffraction_power"])),
                ("Jones X Total", "total", _power_db_grid(payload["jones_x_total_power"])),
                ("Jones X Reflection", "reflection", _power_db_grid(payload["jones_x_reflection_power"])),
                ("Jones X Diffraction", "diffraction", _power_db_grid(payload["jones_x_diffraction_power"])),
                ("Jones Y Total", "total", _power_db_grid(payload["jones_y_total_power"])),
                ("Jones Y Reflection", "reflection", _power_db_grid(payload["jones_y_reflection_power"])),
                ("Jones Y Diffraction", "diffraction", _power_db_grid(payload["jones_y_diffraction_power"])),
            )
            for col_idx, (title, family, values) in enumerate(panel_defs):
                ax = axes[row_idx, col_idx]
                image = ax.imshow(
                    values,
                    origin="lower",
                    extent=extent,
                    cmap="jet",
                    vmin=-60.0,
                    vmax=-20.0,
                    interpolation="nearest",
                )
                decorate_axis(ax, specs, tx_xy, f"{row_label}\n{title}")
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_xlabel("")
                ax.set_ylabel("")
                colorbar_handle = image

        if colorbar_handle is not None:
            fig.colorbar(colorbar_handle, ax=axes, shrink=0.86, label="Power [dB]")

        fig.suptitle(
            "Multipath Forward Total/Reflection/Diffraction Polarization View\n"
            f"scene_material=eps_r={EPS_R:.0e} sigma_e=0.0, grid={GRID_SIZE}, n_rays={N_RAYS}\n"
            f"{reflection_backend_note}",
            fontsize=14,
        )
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        fig.savefig(OUTPUT_PATH, dpi=180)
    finally:
        plt.close(fig)

    assert OUTPUT_PATH.exists()
    assert OUTPUT_PATH.stat().st_size > 0
