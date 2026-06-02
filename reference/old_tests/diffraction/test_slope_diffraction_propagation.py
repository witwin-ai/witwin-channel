"""Regression tests for slope-assisted higher-order diffraction propagation."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import witwin as wt

import drjit as dr

from tests._scene_helpers import box_geometry, build_scene as build_test_scene
from witwin.channel.monitors.field.field import Field
from witwin.channel.kernels.trace.utd import utd_accumulate_forward as _accumulate_edge_states_to_receivers
from witwin.channel.trace.diffraction import (
    _build_higher_order_state_arrays,
    _slope_derivative_safe_mask,
    _build_tx_first_order_state_arrays,
    _make_state_arrays,
)


def build_scene():
    cube1 = box_geometry(center=(-1.8, -1.2, 1.5), size=2.0)
    cube2 = box_geometry(center=(1.8, 1.2, 1.5), size=2.0)
    return build_test_scene(cube1, cube2)


def field_power(field):
    return float(dr.sum(field.real * field.real + field.imag * field.imag)[0])


def test_second_order_states_carry_slope_channel():
    scene = build_scene()
    edge_data = scene.get_edge_data(1.5)["edge_data"]
    wavelength = 299792458.0 / 1e9
    k = 2.0 * dr.pi / wavelength
    tx = wt.Point3f(0.0, -4.0, 1.5)

    first_order = _build_tx_first_order_state_arrays(tx, edge_data, wavelength, k)
    second_order = _build_higher_order_state_arrays(
        first_order,
        edge_data,
        k,
        candidate_backend="bruteforce",
    )

    assert second_order["n_states"] > 0
    slope_power = field_power(second_order["incident_normal_derivative"])
    assert slope_power > 1e-12, f"Expected non-zero incident slope for second-order states, got {slope_power:.6e}"


def test_second_order_receiver_field_uses_slope_channel():
    scene = build_scene()
    edge_data = scene.get_edge_data(1.5)["edge_data"]
    wavelength = 299792458.0 / 1e9
    k = 2.0 * dr.pi / wavelength
    tx = wt.Point3f(0.0, -4.0, 1.5)

    first_order = _build_tx_first_order_state_arrays(tx, edge_data, wavelength, k)
    second_order = _build_higher_order_state_arrays(
        first_order,
        edge_data,
        k,
        candidate_backend="bruteforce",
    )

    field = Field(bounds=((-4, 4), (-4, 4)), size=(12, 12))
    coords = field.get_coordinates()
    rx_pos = wt.Point3f(coords["X"], coords["Y"], wt.Float(1.5))

    with_slope_direct, with_slope_multi, _, _, _ = _accumulate_edge_states_to_receivers(
        second_order, rx_pos, k, edge_data["n_edges"], return_per_edge=False
    )
    with_slope = wt.Complex2f(
        with_slope_direct.real + with_slope_multi.real,
        with_slope_direct.imag + with_slope_multi.imag,
    )

    no_slope_states = _make_state_arrays(
        edge_idx=second_order["edge_idx"],
        edge_pos=second_order["edge_pos"],
        edge_dir=second_order["edge_dir"],
        n0=second_order["n0"],
        nn=second_order["n_face_n"],
        wedge_n=second_order["wedge_n"],
        adjacent_face0=second_order["adjacent_face0"],
        adjacent_face1=second_order["adjacent_face1"],
        source_pos=second_order["source_pos"],
        edge_line_min=second_order["edge_line_min"],
        edge_line_max=second_order["edge_line_max"],
        incident_field=second_order["incident_field"],
        incident_normal_derivative=wt.Complex2f(
            dr.zeros(wt.Float, second_order["n_states"]),
            dr.zeros(wt.Float, second_order["n_states"]),
        ),
        incident_vector={
            "x": second_order["incident_vector_x"],
            "y": second_order["incident_vector_y"],
            "z": second_order["incident_vector_z"],
        },
        incident_normal_derivative_vector={
            "x": wt.Complex2f(dr.zeros(wt.Float, second_order["n_states"]), dr.zeros(wt.Float, second_order["n_states"])),
            "y": wt.Complex2f(dr.zeros(wt.Float, second_order["n_states"]), dr.zeros(wt.Float, second_order["n_states"])),
            "z": wt.Complex2f(dr.zeros(wt.Float, second_order["n_states"]), dr.zeros(wt.Float, second_order["n_states"])),
        },
        prefix_reflection_depth=second_order["prefix_reflection_depth"],
        intermediate_reflection_depth=second_order["intermediate_reflection_depth"],
        suffix_reflection_depth=second_order["suffix_reflection_depth"],
        order=second_order["order"],
        retain_cold_metadata=False,
    )
    without_slope_direct, without_slope_multi, _, _, _ = _accumulate_edge_states_to_receivers(
        no_slope_states, rx_pos, k, edge_data["n_edges"], return_per_edge=False
    )
    without_slope = wt.Complex2f(
        without_slope_direct.real + without_slope_multi.real,
        without_slope_direct.imag + without_slope_multi.imag,
    )

    diff = wt.Complex2f(with_slope.real - without_slope.real, with_slope.imag - without_slope.imag)
    diff_power = field_power(diff)
    assert diff_power > 1e-12, f"Expected slope channel to change second-order receiver field, got {diff_power:.6e}"


def test_slope_derivative_masks_canonical_pole():
    wedge_n = wt.Float(1.5)
    phi = wt.Float(5.0 * dr.pi / 4.0)
    phi_prime = wt.Float(dr.pi / 4.0)
    safe = _slope_derivative_safe_mask(phi, phi_prime, wedge_n, wt.Float(1e-4))
    assert not bool(safe[0]), "Expected slope derivative mask to reject canonical cotangent pole"


