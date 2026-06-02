"""Shared path helpers for channel bin scripts and main visual tests."""

from __future__ import annotations

import os
import sys
from pathlib import Path

if os.environ.get("WITWIN_CHANNEL_BIN_SHOW", "0") != "1":
    os.environ.setdefault("MPLBACKEND", "Agg")

BIN_DIR = Path(__file__).resolve().parent
SUPPORT_DIR = BIN_DIR.parent
TESTS_DIR = SUPPORT_DIR.parent
REPO_ROOT = TESTS_DIR.parent

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

OUTPUT_DIR = TESTS_DIR / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FIGURES_DIR = OUTPUT_DIR


def maybe_show():
    import matplotlib.pyplot as plt

    if os.environ.get("WITWIN_CHANNEL_BIN_SHOW", "0") == "1":
        plt.show()
    else:
        plt.close("all")
