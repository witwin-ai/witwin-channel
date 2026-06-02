"""Shared case definitions for multipath scaling stress tests."""

from __future__ import annotations

DEFAULT_BASELINE = {"grid_size": 256, "n_rays": 10000, "motif_repeats": 1}

DEFAULT_SWEEPS = {
    "resolution": [
        {"grid_size": 64, "n_rays": 10000, "motif_repeats": 1},
        {"grid_size": 128, "n_rays": 10000, "motif_repeats": 1},
        {"grid_size": 192, "n_rays": 10000, "motif_repeats": 1},
        {"grid_size": 256, "n_rays": 10000, "motif_repeats": 1},
        {"grid_size": 384, "n_rays": 10000, "motif_repeats": 1},
        {"grid_size": 512, "n_rays": 10000, "motif_repeats": 1},
    ],
    "rays": [
        {"grid_size": 256, "n_rays": 2500, "motif_repeats": 1},
        {"grid_size": 256, "n_rays": 5000, "motif_repeats": 1},
        {"grid_size": 256, "n_rays": 10000, "motif_repeats": 1},
        {"grid_size": 256, "n_rays": 20000, "motif_repeats": 1},
        {"grid_size": 256, "n_rays": 40000, "motif_repeats": 1},
    ],
    "triangles": [
        {"grid_size": 256, "n_rays": 10000, "motif_repeats": 1},
        {"grid_size": 256, "n_rays": 10000, "motif_repeats": 4},
        {"grid_size": 256, "n_rays": 10000, "motif_repeats": 9},
        {"grid_size": 256, "n_rays": 10000, "motif_repeats": 16},
        {"grid_size": 256, "n_rays": 10000, "motif_repeats": 25},
    ],
    "interactions": [
        {"grid_size": 256, "n_rays": 10000, "motif_repeats": 1},
        {"grid_size": 512, "n_rays": 10000, "motif_repeats": 1},
        {"grid_size": 256, "n_rays": 40000, "motif_repeats": 1},
        {"grid_size": 256, "n_rays": 10000, "motif_repeats": 16},
        {"grid_size": 512, "n_rays": 40000, "motif_repeats": 1},
        {"grid_size": 512, "n_rays": 10000, "motif_repeats": 16},
        {"grid_size": 256, "n_rays": 40000, "motif_repeats": 16},
        {"grid_size": 512, "n_rays": 40000, "motif_repeats": 16},
    ],
}

