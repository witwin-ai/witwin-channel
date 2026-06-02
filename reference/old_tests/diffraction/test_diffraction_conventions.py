"""Regression tests for diffraction angle and wedge-ordering conventions."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import math

import witwin as wt

import drjit as dr

from witwin.channel.trace.diffraction import (
    _build_tx_first_order_state_arrays,
    _compute_edge_angles,
    _compute_incident_edge_geometry,
)
from witwin.channel.trace.diffraction.utd import _beta_term_state, diffraction_coefficient_2d
from witwin.channel.utils import scalar
from witwin.channel.validation import build_double_wedge_case
def _max_abs(value):
    return scalar(dr.max(dr.abs(value)))


def test_wedge_face_order_matches_exterior_rotation():
    case = build_double_wedge_case()
    edge_data = case.scene.get_edge_data(case.calculation_height)["edge_data"]

    assert edge_data is not None
    assert edge_data["n_edges"] > 0

    to_hat = dr.normalize(dr.cross(edge_data["n0"], edge_data["edge_dir"]))
    tn_hat = dr.normalize(dr.cross(edge_data["n_face_n"], edge_data["edge_dir"]))
    rotation = dr.atan2(dr.dot(dr.cross(to_hat, tn_hat), edge_data["edge_dir"]), dr.dot(to_hat, tn_hat))
    rotation = dr.select(rotation < 0.0, rotation + 2.0 * dr.pi, rotation)
    expected = (edge_data["wedge_n"] - 1.0) * dr.pi

    assert _max_abs(rotation - expected) < 1e-6
    assert scalar(dr.min(rotation)) > 0.0
    assert scalar(dr.max(rotation)) <= math.pi + 1e-6


def test_incident_phi_prime_helper_matches_general_angle_helper():
    case = build_double_wedge_case()
    edge_data = case.scene.get_edge_data(case.calculation_height)["edge_data"]
    wavelength = 299792458.0 / 1e9
    k = 2.0 * math.pi / wavelength
    tx = wt.Point3f(*case.tx_pos)

    first_order = _build_tx_first_order_state_arrays(
        tx, edge_data, wavelength, k, history_size=2, scene=case.scene
    )
    assert first_order["n_states"] > 0

    target_pos = first_order["edge_pos"] + 0.5 * dr.normalize(dr.cross(first_order["n0"], first_order["edge_dir"]))
    _, phi_prime_general, _, s_prime_general = _compute_edge_angles(
        first_order["source_pos"],
        first_order["edge_pos"],
        first_order["edge_dir"],
        first_order["n0"],
        target_pos,
    )
    phi_prime_incident, s_prime_incident = _compute_incident_edge_geometry(
        first_order["source_pos"],
        first_order["edge_pos"],
        first_order["edge_dir"],
        first_order["n0"],
    )

    assert _max_abs(phi_prime_general - phi_prime_incident) < 1e-6
    assert _max_abs(s_prime_general - s_prime_incident) < 1e-6

    wedge_limit = first_order["wedge_n"] * dr.pi
    assert scalar(dr.min(phi_prime_incident)) >= -1e-6
    assert scalar(dr.max(phi_prime_incident - wedge_limit)) <= 1e-6


def test_diffraction_coefficient_is_reciprocal_under_source_receiver_swap():
    phi = wt.Float(1.1)
    phi_prime = wt.Float(2.2)
    wedge_n = wt.Float(1.5)
    k = wt.Float(20.0)
    s = wt.Float(3.0)
    s_prime = wt.Float(4.0)
    R0 = wt.Complex2f(-0.3, 0.2)
    Rn = wt.Complex2f(-0.4, -0.1)

    forward = diffraction_coefficient_2d(phi, phi_prime, wedge_n, k, s, s_prime, R0=R0, Rn=Rn)
    reverse = diffraction_coefficient_2d(phi_prime, phi, wedge_n, k, s_prime, s, R0=R0, Rn=Rn)

    assert scalar(dr.abs(forward - reverse)) < 1e-12


def test_direct_beta_term_keeps_shadow_boundary_limit_finite():
    wedge_n = wt.Float(2.0)
    k = wt.Float(2.0 * math.pi / 0.3)
    s = wt.Float(3.0)
    s_prime = wt.Float(5.0)
    kL = k * s * s_prime * dr.rcp(s + s_prime)
    expected_mag = scalar(wedge_n * dr.sqrt(2.0 * dr.pi * kL))

    for delta in (1.0e-6, 1.0e-7, 1.0e-8):
        lit_term, _, _ = _beta_term_state(
            wt.Float(math.pi - delta),
            wedge_n,
            kL,
            -1.0,
            "minus",
        )
        shadow_term, _, _ = _beta_term_state(
            wt.Float(math.pi + delta),
            wedge_n,
            kL,
            -1.0,
            "minus",
        )
        lit_mag = scalar(dr.abs(lit_term))
        shadow_mag = scalar(dr.abs(shadow_term))

        assert math.isfinite(lit_mag)
        assert math.isfinite(shadow_mag)
        assert abs(lit_mag - expected_mag) / expected_mag < 0.1
        assert abs(shadow_mag - expected_mag) / expected_mag < 0.1
