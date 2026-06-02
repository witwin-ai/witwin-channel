from __future__ import annotations

import numpy as np
import pytest
import torch

from tests._scene_helpers import box_geometry, build_scene
from tests.support.bin._path_monitor_phase6 import (
    PHASE6_BENCHMARK_CASE_IDS,
    PHASE6_GATE_IDS,
    evaluate_geometry_toggle_correctness,
    evaluate_mixed_monitor_correctness,
    evaluate_warm_cache_trace_many_correctness,
    resolve_path_monitor_diffraction_depth_report,
)
from witwin.channel import FieldMonitor, PathMonitor, Tracer
from witwin.channel.validation import build_single_wedge_case


def _resolve_path_payload(result, name: str):
    if isinstance(result, dict):
        return result[name]
    return result


def test_phase6_benchmark_matrix_lists_required_cases_and_gates():
    assert set(PHASE6_BENCHMARK_CASE_IDS) >= {
        "default_first_order_path_export",
        "explicit_multi_order_path_export",
        "geometry_off_path_export",
        "geometry_on_path_export",
        "mixed_field_path_trace",
        "warm_cache_trace_many",
    }
    assert set(PHASE6_GATE_IDS) >= {
        "first_order_vs_multi_order_bounded",
        "geometry_off_vs_on_bounded",
        "mixed_monitor_vs_separate_bounded",
        "warm_cache_trace_many_reuse",
    }


@pytest.mark.gpu
def test_phase6_depth_report_tracks_default_and_inherited_monitor_depths():
    tracer = Tracer(
        frequency=1e9,
        scene=build_scene(),
        reflection_n_rays=0,
        reflection_max_bounces=0,
        max_diffractions=3,
    )

    default_report = resolve_path_monitor_diffraction_depth_report(
        tracer.config.trace,
        requested_max_diffractions=1,
    )
    inherited_report = resolve_path_monitor_diffraction_depth_report(
        tracer.config.trace,
        requested_max_diffractions=None,
    )

    assert default_report["monitor_requested_max_diffractions"] == 1
    assert default_report["effective_max_diffractions"] == 1
    assert inherited_report["monitor_requested_max_diffractions"] is None
    assert inherited_report["effective_max_diffractions"] == 3


@pytest.mark.gpu
def test_phase6_geometry_toggle_correctness_gate_matches_trace_outputs():
    scene = build_scene(
        box_geometry(center=(0.0, 0.0, 1.5), size=(0.25, 6.0, 3.0)),
    )
    tracer = Tracer(
        frequency=1e9,
        scene=scene,
        reflection_n_rays=8192,
        reflection_max_bounces=1,
        max_diffractions=0,
    )
    tx = (-3.0, -5.0, 1.5)
    rx_positions = torch.tensor(
        [
            [-3.0, 5.0, 1.5],
            [-2.0, 4.0, 1.5],
        ],
        dtype=torch.float32,
    )

    no_geometry = _resolve_path_payload(
        tracer.trace(
            tx,
            monitor=PathMonitor(
                "rx",
                positions=rx_positions,
                max_diffractions=0,
                return_geometry=False,
            ),
            verbose=False,
        ),
        "rx",
    )
    geometry = _resolve_path_payload(
        tracer.trace(
            tx,
            monitor=PathMonitor(
                "rx",
                positions=rx_positions,
                max_diffractions=0,
                return_geometry=True,
            ),
            verbose=False,
        ),
        "rx",
    )

    check = evaluate_geometry_toggle_correctness(no_geometry, geometry)

    assert check["passed"], check["errors"]


@pytest.mark.gpu
def test_phase6_mixed_monitor_correctness_gate_matches_standalone_results():
    scene = build_scene(
        box_geometry(center=(0.0, 0.0, 1.5), size=(1.0, 1.0, 3.0)),
    )
    tracer = Tracer(
        frequency=1e9,
        scene=scene,
        reflection_n_rays=0,
        reflection_max_bounces=0,
        max_diffractions=0,
    )
    tx = (0.0, -4.0, 1.5)
    field_name = "field_xy"
    path_name = "rx"
    field_monitor = FieldMonitor(
        field_name,
        axis="z",
        position=1.5,
        bounds=((-2.0, 2.0), (-1.0, 3.0)),
        grid_size=(8, 8),
    )
    path_monitor = PathMonitor(
        path_name,
        positions=torch.tensor(
            [
                [-1.0, 2.0, 1.5],
                [0.0, 2.5, 1.5],
                [1.0, 3.0, 1.5],
            ],
            dtype=torch.float32,
        ),
        max_diffractions=0,
    )

    field_only = tracer.trace(tx, monitor=field_monitor, verbose=False)
    path_only = tracer.trace(tx, monitor=path_monitor, verbose=False)
    mixed = tracer.trace(tx, monitor=(field_monitor, path_monitor), verbose=False)

    check = evaluate_mixed_monitor_correctness(
        field_only,
        path_only,
        mixed,
        field_name=field_name,
        path_name=path_name,
    )

    assert check["passed"], check["errors"]


@pytest.mark.gpu
def test_phase6_warm_cache_trace_many_correctness_reports_hits_and_overrides():
    case = build_single_wedge_case()
    tracer = Tracer(
        frequency=1e9,
        scene=case.scene,
        reflection_n_rays=0,
        reflection_max_bounces=0,
        max_diffractions=2,
    )
    base_monitor = PathMonitor(
        "rx",
        positions=torch.tensor(
            [
                [-1.0, 2.0, case.calculation_height],
                [0.5, 2.5, case.calculation_height],
            ],
            dtype=torch.float32,
        ),
    )
    requests = [
        {
            "tx_pos": torch.tensor(case.tx_pos, dtype=torch.float32),
            "monitor_overrides": {
                "rx": {
                    "positions": torch.tensor(
                        [
                            [-1.5, 2.1, case.calculation_height],
                            [0.0, 2.6, case.calculation_height],
                        ],
                        dtype=torch.float32,
                    ),
                }
            },
        },
        {
            "tx_pos": torch.tensor((1.0, -6.0, case.calculation_height), dtype=torch.float32),
            "monitor_overrides": {
                "rx": {
                    "positions": torch.tensor(
                        [
                            [-0.75, 2.0, case.calculation_height],
                            [1.25, 2.8, case.calculation_height],
                        ],
                        dtype=torch.float32,
                    ),
                }
            },
        },
    ]

    tracer.trace_many(requests, monitor=base_monitor, verbose=False)
    results = tracer.trace_many(requests, monitor=base_monitor, verbose=False)
    check = evaluate_warm_cache_trace_many_correctness(
        results=results,
        requests=requests,
        path_name="rx",
    )

    assert check["passed"], check["errors"]
    assert check["persistent_hit_results"] == len(results)
    for result, request in zip(results, requests, strict=True):
        np.testing.assert_allclose(
            np.asarray(_resolve_path_payload(result, "rx").rx_positions, dtype=np.float32),
            np.asarray(request["monitor_overrides"]["rx"]["positions"], dtype=np.float32),
            rtol=0.0,
            atol=0.0,
        )
