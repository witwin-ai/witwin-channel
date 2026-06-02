"""Standalone three-cube witwin-vs-Sionna radio-map visual test."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
if os.environ.get("WITWIN_CHANNEL_MAIN_SHOW", "0") != "1":
    os.environ.setdefault("MPLBACKEND", "Agg")

from tests.main.plot_radiomap_sionna_three_cubes import (
    save_radiomap_sionna_three_cubes_comparison,
)


pytestmark = pytest.mark.gpu

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output"
OUTPUT_PREFIX = OUTPUT_DIR / "radiomap_three_cubes_sionna_compare_matched_isb_completion"


def test_radiomap_sionna_three_cubes_main():
    figure_path, arrays_path, json_path = save_radiomap_sionna_three_cubes_comparison(
        OUTPUT_PREFIX,
        # Keep the standalone script defaults high-resolution, but make the
        # pytest regression lighter so second-order diffraction does not push
        # the matched-ISB coherent replay into tens of millions of pairs.
        grid_size=256,
        n_rays=384,
        samples_per_tx=1000_000,
    )

    assert figure_path.exists()
    assert figure_path.stat().st_size > 0
    assert arrays_path.exists()
    assert arrays_path.stat().st_size > 0
    assert json_path.exists()
    assert json_path.stat().st_size > 0

    summary = json.loads(json_path.read_text(encoding="utf-8"))
    assert summary["witwin_edge_selection_mode"] == "all_edges"
    assert summary["witwin_shadow_support_cutoff_db"] == 25.0
    assert summary["sionna_edge_diffraction"] is True
