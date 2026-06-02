"""Regression test for diffraction-order control."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import witwin as wt

import drjit as dr

from tests._scene_helpers import box_geometry, build_scene as build_test_scene
from witwin.channel import FieldMonitor, Tracer
def build_scene():
    cube1 = box_geometry(center=(-1.8, -1.2, 1.5), size=2.0)
    cube2 = box_geometry(center=(1.8, 1.2, 1.5), size=2.0)
    return build_test_scene(cube1, cube2)


def diffraction_power(result):
    a_dif = result.primary.field.diffraction
    return float(dr.sum(a_dif.real * a_dif.real + a_dif.imag * a_dif.imag)[0])


def test_second_order_diffraction_changes_field():
    scene = build_scene()

    tracer_order1 = Tracer(
        frequency=1e9,
        scene=scene,
        reflection_n_rays=0,
        reflection_max_bounces=0,
        enable_rd_diffraction=False,
        max_diffractions=1,
    )
    tracer_order2 = Tracer(
        frequency=1e9,
        scene=scene,
        reflection_n_rays=0,
        reflection_max_bounces=0,
        enable_rd_diffraction=False,
        max_diffractions=2,
    )

    tx = wt.Point3f(0.0, -4.0, 1.5)
    monitor = FieldMonitor(
        "diffraction_plane",
        axis="z",
        position=1.5,
        bounds=((-4.0, 4.0), (-4.0, 4.0)),
        grid_size=24,
    )

    res1 = tracer_order1.trace(tx, monitor=monitor, verbose=False)
    res2 = tracer_order2.trace(tx, monitor=monitor, verbose=False)

    p1 = diffraction_power(res1)
    p2 = diffraction_power(res2)
    assert abs(p2 - p1) > 1e-8, f"Expected order-2 diffraction to change the field, got p1={p1:.6e}, p2={p2:.6e}"


