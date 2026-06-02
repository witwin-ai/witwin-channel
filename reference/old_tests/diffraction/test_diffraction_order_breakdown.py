"""Regression tests for order-separated diffraction breakdown helpers."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import witwin as wt

import drjit as dr

from tests._scene_helpers import box_geometry, build_scene as build_test_scene
from witwin.channel import Field, native_extension_available
from witwin.channel.trace.diffraction import compute_diffraction_field, compute_diffraction_order_breakdown
def build_scene():
    cube1 = box_geometry(center=(-1.8, -1.2, 1.5), size=2.0)
    cube2 = box_geometry(center=(1.8, 1.2, 1.5), size=2.0)
    return build_test_scene(cube1, cube2)


def field_power(field):
    return float(dr.sum(field.real * field.real + field.imag * field.imag)[0])


def _expected_execution_validation():
    if native_extension_available():
        return "native_default_dispatch", "native CUDA"
    return "validated_drjit", "strict Dr.Jit"


def test_order_breakdown_reconstructs_total_diffraction():
    scene = build_scene()
    field = Field(bounds=((-4, 4), (-4, 4)), size=(16, 16))
    coords = field.get_coordinates()
    wavelength = 299792458.0 / 1e9
    k = 2.0 * dr.pi / wavelength
    tx = wt.Point3f(0.0, -4.0, 1.5)

    dif_real, dif_imag, _, dif_components = compute_diffraction_field(
        coords["X"],
        coords["Y"],
        1.5,
        tx,
        scene,
        wavelength,
        k,
        reflection_detail=None,
        max_diffractions=2,
        reflection_n_rays=0,
        reflection_max_bounces=0,
        grid=field,
        grid_data=coords,
        return_components=True,
        return_per_edge=False,
    )
    total_field = wt.Complex2f(dif_real, dif_imag)
    expected_tier, expected_note_fragment = _expected_execution_validation()
    assert dif_components["solver_metadata"]["execution_validation_tier"] == expected_tier
    assert expected_note_fragment in dif_components["solver_metadata"]["execution_validation_note"]

    breakdown = compute_diffraction_order_breakdown(
        coords["X"],
        coords["Y"],
        1.5,
        tx,
        scene,
        wavelength,
        k,
        reflection_detail=None,
        max_diffractions=2,
        reflection_n_rays=0,
        reflection_max_bounces=0,
        grid=field,
        grid_data=coords,
        split_by_edge=True,
    )
    assert breakdown["solver_metadata"]["execution_validation_tier"] == expected_tier
    assert expected_note_fragment in breakdown["solver_metadata"]["execution_validation_note"]

    order_total = wt.Complex2f(0.0, 0.0)
    for order_field in breakdown["order_fields"]:
        order_total = order_total + order_field

    diff_total = wt.Complex2f(total_field.real - order_total.real, total_field.imag - order_total.imag)
    assert field_power(diff_total) < 1e-10

    for order_idx, order_field in enumerate(breakdown["order_fields"]):
        edge_sum = wt.Complex2f(0.0, 0.0)
        for edge_field in breakdown["order_edge_fields"][order_idx]:
            edge_sum = edge_sum + edge_field
        diff_edge = wt.Complex2f(order_field.real - edge_sum.real, order_field.imag - edge_sum.imag)
        assert field_power(diff_edge) < 1e-10

        first_edge_sum = wt.Complex2f(0.0, 0.0)
        for first_edge_field in breakdown["order_first_edge_fields"][order_idx]:
            first_edge_sum = first_edge_sum + first_edge_field
        diff_first_edge = wt.Complex2f(
            order_field.real - first_edge_sum.real,
            order_field.imag - first_edge_sum.imag,
        )
        assert field_power(diff_first_edge) < 1e-10


