"""Standalone pure witwin three-cube radio-map gradient test."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest
if os.environ.get("WITWIN_CHANNEL_MAIN_SHOW", "0") != "1":
    os.environ.setdefault("MPLBACKEND", "Agg")

from tests.main.plot_radiomap_gradients_three_cubes import (
    save_radiomap_gradients_three_cubes,
)


pytestmark = pytest.mark.gpu

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output"
OUTPUT_PREFIX = OUTPUT_DIR / "radiomap_three_cubes_gradients"


def test_radiomap_gradients_three_cubes_main():
    figure_path, arrays_path, json_path = save_radiomap_gradients_three_cubes(
        OUTPUT_PREFIX,
        grid_size=256,
        n_rays=512,
        fd_step=1.0e-3,
    )
    component_figure_path = OUTPUT_PREFIX.with_name(OUTPUT_PREFIX.stem + "_components").with_suffix(".png")

    assert figure_path.exists()
    assert figure_path.stat().st_size > 0
    assert component_figure_path.exists()
    assert component_figure_path.stat().st_size > 0
    assert arrays_path.exists()
    assert arrays_path.stat().st_size > 0
    assert json_path.exists()
    assert json_path.stat().st_size > 0

    arrays = np.load(arrays_path)
    for key in (
        "total_db",
        "tx_x_ad",
        "tx_x_fd",
        "tx_x_los_ad",
        "tx_x_reflection_ad",
        "tx_x_diffraction_ad",
        "tx_x_raw_diffraction_ad",
        "tx_x_matched_isb_completion_only_ad",
        "tx_x_folded_diffraction_ad",
        "cube1_x_ad",
        "cube1_x_fd",
        "cube1_x_los_ad",
        "cube1_x_reflection_ad",
        "cube1_x_diffraction_ad",
        "cube1_x_raw_diffraction_ad",
        "cube1_x_matched_isb_completion_only_ad",
        "cube1_x_folded_diffraction_ad",
    ):
        assert key in arrays
        assert np.isfinite(arrays[key]).all()
        assert arrays[key].shape == (256, 256)

    summary = json.loads(json_path.read_text(encoding="utf-8"))
    assert int(summary["grid_size"]) == 256
    assert float(summary["tx_pos"][2]) == 4.0
    assert float(summary["tx_pos"][2]) > float(summary["plane_z"])
    assert summary["combine_mode"] == "coherent"
    assert summary["receiver_model"] == "matched_isotropic"
    assert summary["shadow_boundary_mode"] == "matched_isb_completion"
    assert summary["component_figure"] == str(component_figure_path)
    assert float(summary["timings_seconds"]["forward_total"]) > 0.0
    assert summary["accumulation_backend_requested"] == "cell_accumulation"
    assert summary["accumulation_backend_resolved"] == "cell_accumulation"
    assert summary["gradient_accumulation_backend"] == "baseline"
    assert "runtime_backends" in summary
    assert "diffraction" in summary["runtime_backends"]
    assert "forward_fast_path" in summary["runtime_backends"]["diffraction"]
    assert "planner_strategy" in summary["runtime_backends"]["diffraction"]
    assert "planner_skip_reason" in summary["runtime_backends"]["diffraction"]
    assert "pair_chunk_budget" in summary["runtime_backends"]["diffraction"]
    assert "peak_pair_count_estimate" in summary["runtime_backends"]["diffraction"]
    assert "diffraction_diagnostics" in summary
    assert int(summary["diffraction_diagnostics"]["prepared_state_count"]) > 0
    assert "forward_backend_comparison" in summary
    assert summary["forward_backend_comparison"]["baseline_backend"]["resolved"] == "baseline"
    assert summary["forward_backend_comparison"]["requested_backend"]["resolved"] == "cell_accumulation"
    assert "path_gain" in summary["forward_backend_comparison"]["metrics"]
    assert "los" in summary["forward_backend_comparison"]["metrics"]
    assert "reflection" in summary["forward_backend_comparison"]["metrics"]
    assert "raw_diffraction" in summary["forward_backend_comparison"]["metrics"]
    assert "matched_isb_completion_only" in summary["forward_backend_comparison"]["metrics"]
    assert "folded_diffraction" in summary["forward_backend_comparison"]["metrics"]
    for metric_name in (
        "path_gain",
        "los",
        "reflection",
        "raw_diffraction",
        "matched_isb_completion_only",
        "folded_diffraction",
    ):
        metric_summary = summary["forward_backend_comparison"]["metrics"][metric_name]
        assert np.isfinite(float(metric_summary["ad_fd_mean_abs_diff"]))
        assert np.isfinite(float(metric_summary["ad_fd_max_abs_diff"]))
    assert set(summary["parameters"].keys()) == {"tx_x", "cube1_x"}
    assert float(summary["parameters"]["tx_x"]["ad_abs_sum"]) > 0.0
    assert float(summary["parameters"]["cube1_x"]["ad_abs_sum"]) > 0.0
    assert float(summary["parameters"]["tx_x"]["fd_abs_sum"]) > 0.0
    assert float(summary["parameters"]["cube1_x"]["fd_abs_sum"]) > 0.0
    assert np.isfinite(float(summary["parameters"]["tx_x"]["ad_fd_mean_abs_diff"]))
    assert np.isfinite(float(summary["parameters"]["cube1_x"]["ad_fd_mean_abs_diff"]))
    assert np.isfinite(float(summary["parameters"]["tx_x"]["ad_fd_max_abs_diff"]))
    assert np.isfinite(float(summary["parameters"]["cube1_x"]["ad_fd_max_abs_diff"]))
    assert np.isfinite(float(summary["parameters"]["tx_x"]["scalar_jvp_fd_abs_diff"]))
    assert np.isfinite(float(summary["parameters"]["cube1_x"]["scalar_jvp_fd_abs_diff"]))
    assert np.isfinite(float(summary["parameters"]["tx_x"]["scalar_vjp_fd_abs_diff"]))
    assert np.isfinite(float(summary["parameters"]["cube1_x"]["scalar_vjp_fd_abs_diff"]))
    assert float(summary["parameters"]["tx_x"]["scalar_vjp_jvp_abs_diff"]) < 1.0e-5
    assert float(summary["parameters"]["cube1_x"]["scalar_vjp_jvp_abs_diff"]) < 5.0e-5
    assert summary["parameters"]["tx_x"]["ad_backend"]["resolved"] == "baseline"
    assert summary["parameters"]["cube1_x"]["fd_backend"]["resolved"] == "baseline"
    assert "plus_path_counts" in summary["parameters"]["tx_x"]
    assert "minus_path_counts" in summary["parameters"]["tx_x"]
    assert "plus_runtime_backends" in summary["parameters"]["tx_x"]
    assert "minus_runtime_backends" in summary["parameters"]["tx_x"]
    assert "plus_diffraction_diagnostics" in summary["parameters"]["tx_x"]
    assert "minus_diffraction_diagnostics" in summary["parameters"]["tx_x"]
    assert "scalar_metrics" in summary["parameters"]["tx_x"]
    assert set(summary["parameters"]["tx_x"]["scalar_metrics"].keys()) == {
        "path_gain",
        "raw_diffraction",
        "matched_isb_completion_only",
        "folded_diffraction",
    }
    assert set(summary["parameters"]["tx_x"]["components"].keys()) == {
        "los",
        "reflection",
        "diffraction",
    }
    assert set(summary["parameters"]["cube1_x"]["components"].keys()) == {
        "los",
        "reflection",
        "diffraction",
    }
    assert set(summary["parameters"]["tx_x"]["diagnostic_metrics"].keys()) == {
        "raw_diffraction",
        "matched_isb_completion_only",
        "folded_diffraction",
    }
    assert np.isfinite(
        float(
            summary["parameters"]["tx_x"]["diagnostic_metrics"]["raw_diffraction"][
                "ad_fd_mean_abs_diff"
            ]
        )
    )
