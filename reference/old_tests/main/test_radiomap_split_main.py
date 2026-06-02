"""Standalone split-component radio-map visual test."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
if os.environ.get("WITWIN_CHANNEL_MAIN_SHOW", "0") != "1":
    os.environ.setdefault("MPLBACKEND", "Agg")

from tests.main.plot_radiomap_components import save_radiomap_component_split_figure
pytestmark = pytest.mark.gpu

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output"
OUTPUT_PATH = OUTPUT_DIR / "radiomap_multipath_split_right_cube_eps1e4.png"


def test_radiomap_multipath_split_main():
    output_path = save_radiomap_component_split_figure(
        OUTPUT_PATH,
        grid_size=56,
        n_rays=384,
        right_cube_eps_r=1.0e4,
    )

    assert output_path.exists()
    assert output_path.stat().st_size > 0
