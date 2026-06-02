"""Standalone coherent rotated-cube radio-map visual test."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import witwin as wt
if os.environ.get("WITWIN_CHANNEL_MAIN_SHOW", "0") != "1":
    os.environ.setdefault("MPLBACKEND", "Agg")

from tests.main.plot_radiomap_rotated_cube_coherent_total_diffraction import (
    PLANE_Z,
    TX_POS,
    _build_monitor,
    _build_scene,
    _build_tracer,
    _subset_raw_collection_by_edge,
    save_radiomap_rotated_cube_coherent_total_diffraction_figure,
)
from witwin.channel.monitors.radio_map.samples import (
    _baseline_matched_isotropic_diffraction_power,
    _trace_diffraction_raw_collections,
)


pytestmark = pytest.mark.gpu

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output"
OUTPUT_PATH = OUTPUT_DIR / "radiomap_rotated_cube_coherent_total_diffraction.png"
METADATA_PATH = OUTPUT_DIR / "radiomap_rotated_cube_coherent_total_diffraction.json"


def _replay_edge_power(*, edge_idx: int, points_xy: list[tuple[float, float]]):
    scene = _build_scene(edge_selection_mode="vertical_only")
    tracer = _build_tracer(scene, reflection_n_rays=1024)
    monitor = _build_monitor(grid_size=64, ray_mode="3d")
    config = tracer._resolved_trace_config
    solver_controls = tracer._resolve_monitor_solver_controls(
        monitor,
        execution_intent="radio_map_coherent",
    )
    rx_pos = wt.Point3f(
        [point[0] for point in points_xy],
        [point[1] for point in points_xy],
        [PLANE_Z for _ in points_xy],
    )
    diffraction_raw_collections, _, _ = _trace_diffraction_raw_collections(
        sample_positions=rx_pos,
        tx_pos=wt.Point3f(*TX_POS),
        scene=scene,
        config=config,
        solver_controls=solver_controls,
        monitor=monitor,
        reflection_detail=None,
        persistent_diffraction_state_cache=None,
        local_diffraction_state_cache={},
        diffraction_state_cache_key_fn=None,
        state_layout="reduced_v2",
    )
    wedge_raw = [
        subset
        for raw in diffraction_raw_collections
        if (subset := _subset_raw_collection_by_edge(raw, edge_idx=edge_idx)) is not None
    ]
    power, _, _, _, _ = _baseline_matched_isotropic_diffraction_power(
        diffraction_raw_collections=wedge_raw,
        scene=scene,
        config=config,
        n_rx=len(points_xy),
    )
    return [float(value) for value in power]


def test_radiomap_rotated_cube_coherent_total_diffraction_main():
    output_path, metadata_path = save_radiomap_rotated_cube_coherent_total_diffraction_figure(
        OUTPUT_PATH,
        metadata_path=METADATA_PATH,
        grid_size=512,
        reflection_n_rays=2048,
    )

    assert output_path.exists()
    assert output_path.stat().st_size > 0
    assert metadata_path.exists()
    assert metadata_path.stat().st_size > 0


def test_coplanar_wedge_shadow_fan_stays_nonzero_outside_cube():
    power = _replay_edge_power(
        edge_idx=0,
        points_xy=[(-5.1, -1.5), (3.0, 3.0), (0.0, 0.0)],
    )

    assert float(power[0]) > 1.0e-10
    assert float(power[1]) < 1.0e-3 * float(power[0])
    assert float(power[2]) == 0.0


def test_coplanar_wedge_four_shadow_boundary_completion_is_smooth_and_decays():
    x_line = _replay_edge_power(
        edge_idx=3,
        points_xy=[
            (5.0, 1.55),
            (5.0, 1.57),
            (5.0, 1.60),
            (5.0, -5.0),
        ],
    )
    y_line = _replay_edge_power(
        edge_idx=3,
        points_xy=[
            (-2.55, -5.0),
            (-2.50, -5.0),
            (-2.475, -5.0),
            (-2.45, -5.0),
            (-2.425, -5.0),
        ],
    )

    assert x_line[0] > 1.0e-10
    assert x_line[1] > 1.0e-10
    assert x_line[2] > 1.0e-10
    assert max(x_line[:3]) / min(x_line[:3]) < 1.02
    assert x_line[3] < 1.0e-3 * x_line[0]

    assert y_line[3] > 0.8 * max(y_line[2], y_line[4])
    assert y_line[4] > y_line[1]
