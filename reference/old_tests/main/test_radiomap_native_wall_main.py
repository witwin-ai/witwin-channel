"""Standalone native coherent radio-map wall visual test."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from witwin.channel import native_extension_available
if os.environ.get("WITWIN_CHANNEL_MAIN_SHOW", "0") != "1":
    os.environ.setdefault("MPLBACKEND", "Agg")

from tests.main.plot_radiomap_native_wall import save_radiomap_native_wall_figure
pytestmark = pytest.mark.gpu

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output"
OUTPUT_PATH = OUTPUT_DIR / "radiomap_native_coherent_wall.png"


def test_radiomap_native_coherent_wall_main():
    if not native_extension_available():
        pytest.skip("Native coherent wall figure requires the bundled native extension.")

    output_path = save_radiomap_native_wall_figure(
        OUTPUT_PATH,
        grid_size=32,
    )

    assert output_path.exists()
    assert output_path.stat().st_size > 0

