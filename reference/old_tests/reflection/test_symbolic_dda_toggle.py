"""Regression tests for the symbolic reflection DDA path."""

import math
import sys
from pathlib import Path

import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import drjit as dr
import witwin as wt

from tests._scene_helpers import box_geometry, build_scene as build_test_scene
from witwin.channel import DiffractionExecutionConfig, Field, compute_diffraction_field
from witwin.channel.trace import compute_reflection_field
FREQUENCY = 1e9
WAVELENGTH = 299792458.0 / FREQUENCY
WAVENUMBER = 2.0 * dr.pi / WAVELENGTH
TX_POLARIZATION = (0.0, 1.0, 0.0)


def build_scene():
    cube1 = box_geometry(center=(-2.0, -2.0, 1.5), size=2.0)
    cube2 = box_geometry(center=(2.0, 1.5, 1.5), size=2.0)
    return build_test_scene(cube1, cube2)


def field_power(field):
    return float(dr.sum(field.real * field.real + field.imag * field.imag)[0])


def assert_field_finite(field):
    power = field_power(field)
    assert math.isfinite(power)
    assert power >= 0.0


def run_case(*, suffix_dda: str = "symbolic"):
    scene = build_scene()
    field = Field(bounds=((-6.0, 6.0), (-6.0, 6.0)), size=(16, 16))
    coords = field.get_coordinates()
    tx = wt.Point3f(0.0, -5.0, 1.5)
    execution = DiffractionExecutionConfig(
        accumulate_primal="drjit",
        accumulate_jvp="drjit_replay",
        accumulate_backward="drjit_replay",
        suffix_dda=suffix_dda,
    )

    a_ref, _, reflection_detail = compute_reflection_field(
        grid=field,
        rx_z=1.5,
        tx_pos=tx,
        scene=scene,
        wavelength=WAVELENGTH,
        k=WAVENUMBER,
        n_rays=64,
        max_reflections=2,
        mode="2d",
        reflection_coef=0.82,
        tx_polarization=TX_POLARIZATION,
        return_per_bounce=False,
        grid_data=coords,
    )

    _, _, _, dif_components = compute_diffraction_field(
        coords["X"],
        coords["Y"],
        1.5,
        tx,
        scene,
        WAVELENGTH,
        WAVENUMBER,
        reflection_detail=reflection_detail,
        max_diffractions=1,
        reflection_n_rays=64,
        reflection_max_bounces=2,
        reflection_coef=0.82,
        reflection_mode="2d",
        grid=field,
        grid_data=coords,
        return_components=True,
        return_per_edge=False,
        tx_polarization=TX_POLARIZATION,
        execution=execution,
    )

    return {
        "reflection": a_ref,
        "diffraction_direct": dif_components["a_direct"],
        "diffraction_multi": dif_components["a_multi"],
    }


def test_symbolic_dda_produces_finite_reflection_and_diffraction_fields():
    result = run_case()
    assert_field_finite(result["reflection"])
    assert_field_finite(result["diffraction_direct"])
    assert_field_finite(result["diffraction_multi"])


def test_evaluated_dda_mode_is_rejected_by_config():
    with pytest.raises(ValueError, match="suffix_dda must be one of symbolic"):
        DiffractionExecutionConfig(suffix_dda="evaluated")
