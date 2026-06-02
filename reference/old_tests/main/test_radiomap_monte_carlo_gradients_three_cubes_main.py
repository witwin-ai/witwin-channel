"""Standalone three-cube Monte Carlo radio-map gradient test."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest
if os.environ.get("WITWIN_CHANNEL_MAIN_SHOW", "0") != "1":
    os.environ.setdefault("MPLBACKEND", "Agg")

from tests.main.plot_radiomap_monte_carlo_gradients_three_cubes import (
    save_radiomap_monte_carlo_gradients_three_cubes,
)


pytestmark = pytest.mark.gpu

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output"
OUTPUT_PREFIX = OUTPUT_DIR / "radiomap_monte_carlo_three_cubes_gradients"


def test_radiomap_monte_carlo_gradients_three_cubes_main():
    figure_path, arrays_path, json_path = save_radiomap_monte_carlo_gradients_three_cubes(
        OUTPUT_PREFIX,
        grid_size=96,
        reflection_n_rays=256,
        samples_per_tx=96,
        fd_step=1.0e-3,
    )

    assert figure_path.exists()
    assert figure_path.stat().st_size > 0
    assert arrays_path.exists()
    assert arrays_path.stat().st_size > 0
    assert json_path.exists()
    assert json_path.stat().st_size > 0

    arrays = np.load(arrays_path)
    for key in (
        "total_db",
        "tx_x_ad",
        "tx_x_fd",
        "cube1_x_ad",
        "cube1_x_fd",
    ):
        assert key in arrays
        assert np.isfinite(arrays[key]).all()
        assert arrays[key].shape == (96, 96)

    summary = json.loads(json_path.read_text(encoding="utf-8"))
    assert int(summary["grid_size"]) == 96
    assert int(summary["reflection_n_rays"]) == 256
    assert int(summary["samples_per_tx"]) == 96
    assert float(summary["tx_pos"][2]) == 4.0
    assert float(summary["tx_pos"][2]) > float(summary["plane_z"])
    assert summary["combine_mode"] == "incoherent"
    assert summary["receiver_model"] == "matched_isotropic"
    assert summary["shadow_boundary_mode"] == "none"
    assert float(summary["timings_seconds"]["forward_total"]) > 0.0
    assert summary["accumulation_backend_requested"] == "auto"
    assert summary["accumulation_backend_resolved"] == "native_monte_carlo"
    assert summary["monte_carlo_ad_mode"] is True
    assert summary["monte_carlo_ad_backend"] == "outer_custom_op_native_sparse_coeff_cuda"
    assert summary["monte_carlo_tape_layout_version"] == "single_solver_native_sparse_coeff_tape_v3"
    assert set(summary["parameters"].keys()) == {"tx_x", "cube1_x"}
    assert float(summary["parameters"]["tx_x"]["ad_abs_sum"]) > 0.0
    assert float(summary["parameters"]["cube1_x"]["ad_abs_sum"]) > 0.0
    assert float(summary["parameters"]["tx_x"]["fd_abs_sum"]) > 0.0
    assert float(summary["parameters"]["cube1_x"]["fd_abs_sum"]) > 0.0
    assert np.isfinite(float(summary["parameters"]["tx_x"]["reflection_ad_abs_sum"]))
    assert np.isfinite(float(summary["parameters"]["cube1_x"]["reflection_ad_abs_sum"]))
    assert np.isfinite(float(summary["parameters"]["tx_x"]["diffraction_ad_abs_sum"]))
    assert np.isfinite(float(summary["parameters"]["cube1_x"]["diffraction_ad_abs_sum"]))
    assert np.isfinite(float(summary["parameters"]["tx_x"]["ad_fd_mean_abs_diff"]))
    assert np.isfinite(float(summary["parameters"]["cube1_x"]["ad_fd_mean_abs_diff"]))
    assert np.isfinite(float(summary["parameters"]["tx_x"]["ad_fd_max_abs_diff"]))
    assert np.isfinite(float(summary["parameters"]["cube1_x"]["ad_fd_max_abs_diff"]))
    assert np.isfinite(float(summary["parameters"]["tx_x"]["scalar_jvp_fd_abs_diff"]))
    assert np.isfinite(float(summary["parameters"]["cube1_x"]["scalar_jvp_fd_abs_diff"]))
    assert np.isfinite(float(summary["parameters"]["tx_x"]["scalar_vjp_fd_abs_diff"]))
    assert np.isfinite(float(summary["parameters"]["cube1_x"]["scalar_vjp_fd_abs_diff"]))
    assert float(summary["parameters"]["tx_x"]["scalar_vjp_jvp_abs_diff"]) < 1.0e-5
    assert float(summary["parameters"]["cube1_x"]["scalar_vjp_jvp_abs_diff"]) < 1.0e-5
    assert float(summary["parameters"]["tx_x"]["backward_only_seconds"]) > 0.0
    assert float(summary["parameters"]["cube1_x"]["backward_only_seconds"]) > 0.0
    assert float(summary["parameters"]["tx_x"]["vjp_total_seconds"]) >= float(
        summary["parameters"]["tx_x"]["backward_only_seconds"]
    )
    assert float(summary["parameters"]["cube1_x"]["vjp_total_seconds"]) >= float(
        summary["parameters"]["cube1_x"]["backward_only_seconds"]
    )
