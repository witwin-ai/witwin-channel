"""Regression tests for generic non-vertical diffraction edges."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import witwin as wt

import drjit as dr

from tests._scene_helpers import build_scene
from witwin.channel import FieldMonitor, Tracer
def _build_slanted_wedge_scene(edge_selection_mode: str):
    points = [
        (0.0, 0.0, 0.0),
        (2.0, 0.0, 1.0),
        (0.0, 2.0, 0.0),
        (0.0, 0.0, 2.0),
    ]
    vertices = wt.Point3f(
        wt.Float(*[point[0] for point in points]),
        wt.Float(*[point[1] for point in points]),
        wt.Float(*[point[2] for point in points]),
    )
    faces = wt.Vector3u(
        wt.UInt32(0, 1),
        wt.UInt32(1, 0),
        wt.UInt32(2, 3),
    )
    return build_scene((vertices, faces), edge_selection_mode=edge_selection_mode)


def _field_power(field):
    return float(dr.sum(field.real * field.real + field.imag * field.imag)[0])


def test_all_edges_mode_exposes_nonvertical_shared_edge():
    vertical_scene = _build_slanted_wedge_scene("vertical_only")
    generic_scene = _build_slanted_wedge_scene("all_edges")

    assert vertical_scene.get_edge_data(0.5)["edge_data"] is None
    generic_edge_data = generic_scene.get_edge_data(0.5)["edge_data"]
    assert generic_edge_data is not None
    assert generic_edge_data["n_edges"] == 1
    assert generic_scene.edge_selection_summary["selection_mode"] == "all_edges"
    assert generic_scene.edge_selection_summary["included_edges"] == 1
    assert generic_scene.edge_selection_summary["total_candidate_edges"] == 5

    edge_dir = generic_edge_data["edge_dir"]
    assert float(dr.max(dr.abs(edge_dir.x))[0]) > 1e-6
    assert float(dr.max(dr.abs(edge_dir.z))[0]) > 1e-6


def test_all_edges_mode_produces_diffraction_for_slanted_wedge():
    tx = wt.Point3f(-3.0, -2.0, 0.5)
    monitor = FieldMonitor(
        "slanted_wedge_plane",
        axis="z",
        position=0.5,
        bounds=((-1.0, 3.0), (-2.0, 2.0)),
        grid_size=12,
    )

    vertical_result = Tracer(
        frequency=1e9,
        scene=_build_slanted_wedge_scene("vertical_only"),
        reflection_n_rays=0,
        reflection_max_bounces=0,
        max_diffractions=1,
    ).trace(tx, monitor=monitor, verbose=False, return_diffraction_audit=True)
    generic_result = Tracer(
        frequency=1e9,
        scene=_build_slanted_wedge_scene("all_edges"),
        reflection_n_rays=0,
        reflection_max_bounces=0,
        max_diffractions=1,
    ).trace(tx, monitor=monitor, verbose=False, return_diffraction_audit=True)

    assert _field_power(vertical_result.primary.field.diffraction) == 0.0
    assert _field_power(generic_result.primary.field.diffraction) > 1e-8
    assert generic_result.primary.metadata["edge_selection_mode"] == "all_edges"

    audit = generic_result.primary.diffraction_detail["state_audit"]
    assert audit["n_states"] > 0
    assert set(audit["path_sequence"]) == {"S -> D"}
    assert np.max(np.abs(np.asarray(audit["edge_dir"].x))) > 1e-6
    assert np.max(np.abs(np.asarray(audit["edge_dir"].z))) > 1e-6


def test_tracer_accepts_declarative_all_edges_scene():
    scene = _build_slanted_wedge_scene("all_edges")
    tracer = Tracer(
        frequency=1e9,
        scene=scene,
        reflection_n_rays=0,
        reflection_max_bounces=0,
        max_diffractions=1,
    )
    monitor = FieldMonitor(
        "all_edges_plane",
        axis="z",
        position=0.5,
        bounds=((-1.0, 3.0), (-2.0, 2.0)),
        grid_size=10,
    )

    result = tracer.trace(
        tx_pos=wt.Point3f(-3.0, -2.0, 0.5),
        monitor=monitor,
        verbose=False,
    )

    assert result.primary.metadata["edge_selection_mode"] == "all_edges"
    assert _field_power(result.primary.field.diffraction) > 1e-8


