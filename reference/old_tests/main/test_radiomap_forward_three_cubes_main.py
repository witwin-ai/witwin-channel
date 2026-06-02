"""Standalone pure witwin three-cube radio-map forward test."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
if os.environ.get("WITWIN_CHANNEL_MAIN_SHOW", "0") != "1":
    os.environ.setdefault("MPLBACKEND", "Agg")

from tests.main.plot_radiomap_forward_three_cubes import (
    save_radiomap_forward_three_cubes,
)


pytestmark = pytest.mark.gpu

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output"
OUTPUT_PREFIX = OUTPUT_DIR / "radiomap_three_cubes_witwin_forward_matched_isb_completion"


def test_radiomap_forward_three_cubes_main():
    figure_path, arrays_path, json_path = save_radiomap_forward_three_cubes(
        OUTPUT_PREFIX,
        grid_size=256,
        n_rays=384,
    )

    assert figure_path.exists()
    assert figure_path.stat().st_size > 0
    assert arrays_path.exists()
    assert arrays_path.stat().st_size > 0
    assert json_path.exists()
    assert json_path.stat().st_size > 0

    summary = json.loads(json_path.read_text(encoding="utf-8"))
    assert summary["witwin_profile"] == "matched_isb_completion"
    assert summary["witwin_combine_mode"] == "coherent"
    assert summary["witwin_receiver_model"] == "matched_isotropic"
    assert summary["witwin_shadow_boundary_mode"] == "matched_isb_completion"
    assert summary["witwin_edge_selection_mode"] == "all_edges"
