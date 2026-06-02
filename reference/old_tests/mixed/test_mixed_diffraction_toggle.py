"""Smoke test for reflection-coupled diffraction toggling."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import witwin as wt

import drjit as dr

from tests._scene_helpers import box_geometry, build_scene as build_test_scene
from witwin.channel import FieldMonitor, Tracer
def build_scene():
    cube1 = box_geometry(center=(-2.0, -2.0, 1.5), size=2.0)
    cube2 = box_geometry(center=(2.0, 1.5, 1.5), size=2.0)
    return build_test_scene(cube1, cube2)


def field_power(field):
    return float(dr.sum(field.real * field.real + field.imag * field.imag)[0])


def test_reflection_coupled_diffraction_toggle():
    scene = build_scene()

    tracer_off = Tracer(
        frequency=1e9,
        scene=scene,
        reflection_n_rays=256,
        reflection_max_bounces=1,
        enable_rd_diffraction=False,
        max_diffractions=2,
    )
    tracer_on = Tracer(
        frequency=1e9,
        scene=scene,
        reflection_n_rays=256,
        reflection_max_bounces=1,
        enable_rd_diffraction=True,
        max_diffractions=2,
    )

    tx = wt.Point3f(0.0, -4.0, 1.5)
    monitor = FieldMonitor(
        "toggle_plane",
        axis="z",
        position=1.5,
        bounds=((-4.0, 4.0), (-4.0, 4.0)),
        grid_size=20,
    )

    res_off = tracer_off.trace(tx, monitor=monitor, verbose=False)
    res_on = tracer_on.trace(tx, monitor=monitor, verbose=False)

    assert field_power(res_off.primary.field.diffraction_mixed) < 1e-20
    assert field_power(res_on.primary.field.diffraction_mixed) > 1e-12
    assert res_off.primary.metadata["edge_selection_mode"] == "vertical_only"
    assert res_off.primary.metadata["path_families"]["R^n -> D"]["status"] == "absent"
    assert res_on.primary.metadata["reflection_suffix_enabled"] is True
    assert res_on.primary.metadata["path_families"]["... -> D -> R^n"]["status"] == "approximate"


