"""Regression checks for the rebuilt material diffraction operator module."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import drjit as dr
import numpy as np
import pytest
import witwin as wt

from tests._scene_helpers import box_drjit_geometry, build_scene
from witwin.channel import (
    ChannelConfig,
    DiffractionExecutionConfig,
    Material,
    FieldMonitor,
    TraceConfig,
    Tracer,
)
from witwin.channel.trace.diffraction.operator import assemble_material_diffraction_operators
from witwin.channel.trace.diffraction.utd import _diffraction_beta_groups_3d
def _complex_abs_max(value):
    return max(float(dr.max(dr.abs(value.real))[0]), float(dr.max(dr.abs(value.imag))[0]))


def _manual_operator(free_term, face0_term, face1_term, face0_operator, face1_operator):
    return {
        "m00": free_term + face0_term * face0_operator["m00"] + face1_term * face1_operator["m00"],
        "m01": face0_term * face0_operator["m01"] + face1_term * face1_operator["m01"],
        "m10": face0_term * face0_operator["m10"] + face1_term * face1_operator["m10"],
        "m11": free_term + face0_term * face0_operator["m11"] + face1_term * face1_operator["m11"],
    }


def test_material_diffraction_operator_matches_canonical_grouped_terms():
    phi = wt.Float(1.1)
    phi_prime = wt.Float(0.7)
    wedge_n = wt.Float(1.5)
    k = wt.Float(2.0 * dr.pi / (299792458.0 / 1e9))
    s = wt.Float(2.4)
    s_prime = wt.Float(1.9)
    sin_beta0 = wt.Float(0.8)
    face0_operator = {
        "m00": wt.Complex2f(0.6, 0.1),
        "m01": wt.Complex2f(0.2, -0.3),
        "m10": wt.Complex2f(-0.1, 0.25),
        "m11": wt.Complex2f(-0.4, 0.05),
    }
    face1_operator = {
        "m00": wt.Complex2f(-0.2, 0.15),
        "m01": wt.Complex2f(0.05, 0.12),
        "m10": wt.Complex2f(-0.08, -0.11),
        "m11": wt.Complex2f(0.45, -0.07),
    }

    operators = assemble_material_diffraction_operators(
        phi=phi,
        phi_prime=phi_prime,
        wedge_n=wedge_n,
        k=k,
        s=s,
        s_prime=s_prime,
        face0_operator=face0_operator,
        face1_operator=face1_operator,
        sin_beta0=sin_beta0,
    )

    factor, dif_group, dif_group_1, dif_group_2, _, _, _ = _diffraction_beta_groups_3d(
        phi,
        phi_prime,
        wedge_n,
        k,
        s,
        s_prime,
        sin_beta0,
        wt.Complex2f(0.0, 0.0),
        wt.Complex2f(1.0, 0.0),
    )
    terms = operators["terms"]
    free_term = terms["direct"]
    face0_term = terms["face0"]
    face1_term = terms["face1"]
    slope_factor = wt.Complex2f(0.0, -1.0) * dr.rcp(k)

    manual_field = _manual_operator(free_term, face0_term, face1_term, face0_operator, face1_operator)
    manual_slope = _manual_operator(
        slope_factor * terms["direct_dphi_prime"],
        slope_factor * terms["face0_dphi_prime"],
        slope_factor * terms["face1_dphi_prime"],
        face0_operator,
        face1_operator,
    )
    manual_field_dphi = _manual_operator(
        terms["direct_dphi"],
        terms["face0_dphi"],
        terms["face1_dphi"],
        face0_operator,
        face1_operator,
    )
    manual_slope_dphi = _manual_operator(
        slope_factor * terms["direct_d2phi_phi_prime"],
        slope_factor * terms["face0_d2phi_phi_prime"],
        slope_factor * terms["face1_d2phi_phi_prime"],
        face0_operator,
        face1_operator,
    )

    for key in ("m00", "m01", "m10", "m11"):
        assert _complex_abs_max(operators["field"][key] - manual_field[key]) < 1e-6
        assert _complex_abs_max(operators["slope"][key] - manual_slope[key]) < 1e-6
        assert _complex_abs_max(operators["field_dphi"][key] - manual_field_dphi[key]) < 1e-6
        assert _complex_abs_max(operators["slope_dphi"][key] - manual_slope_dphi[key]) < 1e-6


def test_material_diffraction_operator_negates_direct_term_without_swapping_faces():
    phi = wt.Float(1.1)
    phi_prime = wt.Float(0.7)
    wedge_n = wt.Float(1.5)
    k = wt.Float(2.0 * dr.pi / (299792458.0 / 1e9))
    s = wt.Float(2.4)
    s_prime = wt.Float(1.9)
    sin_beta0 = wt.Float(0.8)
    face0_operator = {
        "m00": wt.Complex2f(0.6, 0.1),
        "m01": wt.Complex2f(0.2, -0.3),
        "m10": wt.Complex2f(-0.1, 0.25),
        "m11": wt.Complex2f(-0.4, 0.05),
    }
    face1_operator = {
        "m00": wt.Complex2f(-0.2, 0.15),
        "m01": wt.Complex2f(0.05, 0.12),
        "m10": wt.Complex2f(-0.08, -0.11),
        "m11": wt.Complex2f(0.45, -0.07),
    }

    operators = assemble_material_diffraction_operators(
        phi=phi,
        phi_prime=phi_prime,
        wedge_n=wedge_n,
        k=k,
        s=s,
        s_prime=s_prime,
        face0_operator=face0_operator,
        face1_operator=face1_operator,
        sin_beta0=sin_beta0,
    )

    factor, direct_group, _, _, _, _, _ = _diffraction_beta_groups_3d(
        phi,
        phi_prime,
        wedge_n,
        k,
        s,
        s_prime,
        sin_beta0,
        wt.Complex2f(0.0, 0.0),
        wt.Complex2f(0.0, 0.0),
    )
    _, _, _, _, face0_group, _, _ = _diffraction_beta_groups_3d(
        phi,
        phi_prime,
        wedge_n,
        k,
        s,
        s_prime,
        sin_beta0,
        wt.Complex2f(1.0, 0.0),
        wt.Complex2f(0.0, 0.0),
    )
    _, _, _, _, face1_group, _, _ = _diffraction_beta_groups_3d(
        phi,
        phi_prime,
        wedge_n,
        k,
        s,
        s_prime,
        sin_beta0,
        wt.Complex2f(0.0, 0.0),
        wt.Complex2f(1.0, 0.0),
    )
    manual_field = _manual_operator(
        -factor * direct_group,
        factor * face0_group,
        factor * face1_group,
        face0_operator,
        face1_operator,
    )

    for key in ("m00", "m01", "m10", "m11"):
        assert _complex_abs_max(operators["field"][key] - manual_field[key]) < 1e-6


@pytest.mark.gpu
def test_material_diffraction_operator_keeps_isb_cross_section_nearly_continuous():
    grid_size = 512
    frequency = 1e9
    tx_pos = wt.Point3f(-5.0, 5.0, 1.5)
    scene = build_scene(
        box_drjit_geometry(
            center=wt.Point3f(0.0, 0.0, 2.0),
            size=4.0,
            rotation=wt.Float(float(np.deg2rad(-5.0))),
        ),
        material=Material(eps_r=1.0e4),
    )
    monitor = FieldMonitor(
        "isb_operator_regression",
        axis="z",
        position=1.5,
        bounds=((-8.0, 8.0), (-8.0, 8.0)),
        grid_size=grid_size,
    )
    scene.add_monitor(monitor)
    tracer = Tracer(
        frequency=frequency,
        scene=scene,
        config=ChannelConfig(
            trace=TraceConfig(
                diffraction_execution=DiffractionExecutionConfig(suffix_dda="symbolic"),
            )
        ),
        reflection_n_rays=20_000,
        reflection_max_bounces=1,
        reflection_coef=1.0,
        max_diffractions=2,
        tx_polarization=(1.0, 0.0, 0.0),
        use_scene_materials_for_diffraction=True,
    )
    result = tracer.trace(tx_pos=tx_pos)
    row = 128

    def _row_magnitude(field):
        re = np.array(field.real.numpy()).reshape(grid_size, grid_size)[row]
        im = np.array(field.imag.numpy()).reshape(grid_size, grid_size)[row]
        return np.sqrt(re**2 + im**2)

    los_line = _row_magnitude(result.primary.field.los)
    diffraction_line = _row_magnitude(result.primary.field.diffraction)
    total_line = _row_magnitude(result.primary.field.total)

    # The high-epsilon rotated box scene produces an ISB near x ~= -1.27 on
    # y = -4. In the finite-wedge-only model, truncation weakens the shadow-side
    # completion relative to the old infinite-wedge expectation, but the total
    # field should still stay well bounded across the transition.
    lit_idx = 215
    shd_idx = 216
    relative_jump = abs(total_line[lit_idx] - total_line[shd_idx]) / max(total_line[lit_idx], total_line[shd_idx])

    assert los_line[lit_idx] > 0.0
    assert los_line[shd_idx] == 0.0
    assert diffraction_line[shd_idx] > diffraction_line[lit_idx]
    assert relative_jump < 0.10
