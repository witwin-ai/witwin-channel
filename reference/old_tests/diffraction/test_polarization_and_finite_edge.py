"""Regression tests for polarization transport and finite-edge policy."""

import sys
from pathlib import Path
from types import SimpleNamespace
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import witwin as wt

import drjit as dr
import numpy as np
import pytest

from tests._scene_helpers import box_geometry, build_scene as build_test_scene
from witwin.channel import Field, FieldMonitor, Tracer
from witwin.channel.monitors.radio_map import backend as radio_map_helpers
from witwin.channel.utils.polarization import (
    apply_jones_operator,
    complex_dot_real,
    diffraction_edge_basis,
    jones_from_vector,
    path_basis,
    reflect_field_vector,
    reflection_jones_operator,
    scalarize_xy_jones,
    transport_diffraction_vector,
    vector_from_jones,
    vector_from_scalar_and_real_direction,
)
from witwin.channel.trace.diffraction import compute_diffraction_field
from witwin.channel.trace.diffraction.builders import _build_tx_first_order_state_arrays
from witwin.channel.trace.diffraction.field import _edge_state_field_to_targets
from witwin.channel.trace import compute_reflection_field
def _complex_abs_max(value):
    return max(float(dr.max(dr.abs(value.real))[0]), float(dr.max(dr.abs(value.imag))[0]))


def _field_power(field):
    return float(dr.sum(field.real * field.real + field.imag * field.imag)[0])


def _vector_power(vec):
    return float(dr.sum(
        vec["x"].real * vec["x"].real + vec["x"].imag * vec["x"].imag
        + vec["y"].real * vec["y"].real + vec["y"].imag * vec["y"].imag
        + vec["z"].real * vec["z"].real + vec["z"].imag * vec["z"].imag
    )[0])


def _scalar_field_magnitude(field):
    magnitude = dr.sqrt(field.real * field.real + field.imag * field.imag)
    return float(magnitude[0])


def _build_endpoint_clamped_scene():
    points = [
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 2.0),
        (2.0, 0.0, 1.0),
        (0.0, 2.0, 0.0),
    ]
    vertices = wt.Point3f(
        wt.Float(*[point[0] for point in points]),
        wt.Float(*[point[1] for point in points]),
        wt.Float(*[point[2] for point in points]),
    )
    faces = wt.Vector3u(
        wt.UInt32(0, 0),
        wt.UInt32(1, 3),
        wt.UInt32(2, 1),
    )
    return build_test_scene(
        (vertices, faces),
        edge_selection_mode="all_edges",
    )


def _make_manual_edge_state(*, line_min=None, line_max=None):
    source_pos = wt.Point3f(-1.5, 0.2, -1.5)
    edge_pos = wt.Point3f(0.0, 0.0, 0.0)
    edge_dir = wt.Vector3f(0.0, 0.0, 1.0)
    n0 = wt.Vector3f(1.0, 0.0, 0.0)
    nn = wt.Vector3f(0.0, 1.0, 0.0)
    incident_basis = path_basis(edge_pos - source_pos, preferred=edge_dir)
    zero = wt.Complex2f(0.0, 0.0)
    edge_state = {
        "source_pos": source_pos,
        "edge_pos": edge_pos,
        "edge_dir": edge_dir,
        "n0": n0,
        "n_face_n": nn,
        "wedge_n": wt.Float(1.5),
        "adjacent_face0": wt.Int32(0),
        "adjacent_face1": wt.Int32(1),
        "incident_field": zero,
        "incident_normal_derivative": zero,
        "incident_jones_u": wt.Complex2f(1.0, 0.0),
        "incident_jones_v": wt.Complex2f(0.0, 0.0),
        "incident_derivative_jones_u": zero,
        "incident_derivative_jones_v": zero,
        "incident_basis_u": incident_basis["u"],
        "incident_basis_v": incident_basis["v"],
        "incident_basis_k": incident_basis["k"],
        "face0_operator_m00": wt.Complex2f(1.0, 0.0),
        "face0_operator_m01": zero,
        "face0_operator_m10": zero,
        "face0_operator_m11": wt.Complex2f(-1.0, 0.0),
        "face1_operator_m00": zero,
        "face1_operator_m01": zero,
        "face1_operator_m10": zero,
        "face1_operator_m11": zero,
        "r_face0": zero,
        "r_face_n": zero,
        "incident_vector_x": zero,
        "incident_vector_y": zero,
        "incident_vector_z": zero,
        "incident_normal_derivative_vector_x": zero,
        "incident_normal_derivative_vector_y": zero,
        "incident_normal_derivative_vector_z": zero,
    }
    if line_min is not None:
        edge_state["edge_line_min"] = wt.Float(line_min)
    if line_max is not None:
        edge_state["edge_line_max"] = wt.Float(line_max)
    return edge_state


def test_reflection_transport_preserves_s_polarization_subspace():
    incident_dir = wt.Vector3f(1.0, -1.0, 0.0)
    incident_dir = incident_dir / dr.norm(incident_dir)
    normal = wt.Vector3f(0.0, 1.0, 0.0)
    s_hat = wt.Vector3f(0.0, 0.0, 1.0)

    field_vec = vector_from_scalar_and_real_direction(wt.Complex2f(1.0, 0.0), s_hat)
    reflected = reflect_field_vector(
        field_vec,
        incident_dir,
        normal,
        eta_r=5.0,
        sigma=0.0,
        omega=2.0 * np.pi * 1e9,
        gain=1.0,
    )

    assert _complex_abs_max(complex_dot_real(reflected, s_hat)) > 1e-6
    assert _complex_abs_max(reflected["x"]) < 1e-6
    assert _complex_abs_max(reflected["y"]) < 1e-6


def test_reflection_jones_operator_matches_vector_transport():
    incident_dir = wt.Vector3f(1.0, -1.0, 0.25)
    incident_dir = incident_dir / dr.norm(incident_dir)
    normal = wt.Vector3f(0.0, 1.0, 0.0)
    reflected_dir = incident_dir - 2.0 * dr.dot(incident_dir, normal) * normal
    basis_in = path_basis(incident_dir, preferred=wt.Vector3f(1.0, 0.0, 0.0))
    basis_out = path_basis(reflected_dir, preferred=normal)

    input_vector = vector_from_scalar_and_real_direction(wt.Complex2f(1.0, 0.0), basis_in["u"])
    direct_reflected = reflect_field_vector(
        input_vector,
        incident_dir,
        normal,
        eta_r=5.0,
        sigma=0.0,
        omega=2.0 * np.pi * 1e9,
        gain=1.0,
    )
    direct_jones = jones_from_vector(direct_reflected, basis_out)

    operator = reflection_jones_operator(
        incident_dir,
        normal,
        eta_r=5.0,
        sigma=0.0,
        omega=2.0 * np.pi * 1e9,
        gain=1.0,
        incoming_basis=basis_in,
        outgoing_basis=basis_out,
    )
    operator_jones = apply_jones_operator(
        {"u": wt.Complex2f(1.0, 0.0), "v": wt.Complex2f(0.0, 0.0)},
        operator,
    )
    operator_vector = vector_from_jones(operator_jones, basis_out)

    assert _complex_abs_max(direct_jones["u"] - operator_jones["u"]) < 1e-6
    assert _complex_abs_max(direct_jones["v"] - operator_jones["v"]) < 1e-6
    assert _complex_abs_max(direct_reflected["x"] - operator_vector["x"]) < 1e-6
    assert _complex_abs_max(direct_reflected["y"] - operator_vector["y"]) < 1e-6
    assert _complex_abs_max(direct_reflected["z"] - operator_vector["z"]) < 1e-6


def test_diffraction_transport_rotates_phi_component_between_edge_local_bases():
    incident_vec = vector_from_scalar_and_real_direction(wt.Complex2f(1.0, 0.0), wt.Vector3f(0.0, 1.0, 0.0))
    zero_vec = vector_from_scalar_and_real_direction(wt.Complex2f(0.0, 0.0), wt.Vector3f(0.0, 1.0, 0.0))

    transported = transport_diffraction_vector(
        incident_vector=incident_vec,
        incident_derivative_vector=zero_vec,
        source_pos=wt.Point3f(2.0, 0.0, 0.0),
        edge_pos=wt.Point3f(0.0, 0.0, 0.0),
        edge_dir=wt.Vector3f(0.0, 0.0, 1.0),
        target_pos=wt.Point3f(0.0, 2.0, 0.0),
        direct_gain=wt.Complex2f(1.0, 0.0),
        derivative_gain=wt.Complex2f(0.0, 0.0),
    )

    assert _complex_abs_max(transported["x"]) > 1e-6
    assert _complex_abs_max(transported["y"]) < 1e-6
    assert _complex_abs_max(transported["z"]) < 1e-6
    assert float(transported["x"].real[0]) < 0.0


def test_diffraction_edge_basis_uses_ray_only_transverse_gauge():
    ray_dir = wt.Vector3f(0.35, -0.4, 0.85)
    edge_dir_a = wt.Vector3f(0.0, 0.0, 1.0)
    edge_dir_b = wt.Vector3f(1.0, 0.2, 0.1)

    basis_a = diffraction_edge_basis(ray_dir, edge_dir_a, outgoing=True)
    basis_b = diffraction_edge_basis(ray_dir, edge_dir_b, outgoing=True)

    for axis in ("x", "y", "z"):
        assert float(dr.max(dr.abs(getattr(basis_a["u"], axis) - getattr(basis_b["u"], axis)))[0]) < 1e-6
        assert float(dr.max(dr.abs(getattr(basis_a["v"], axis) - getattr(basis_b["v"], axis)))[0]) < 1e-6
        assert float(dr.max(dr.abs(getattr(basis_a["k"], axis) - getattr(basis_b["k"], axis)))[0]) < 1e-6


def test_first_order_diffraction_state_stores_canonical_jones_transport():
    scene = _build_endpoint_clamped_scene()
    edge_data = scene.get_edge_data(1.0, include_projection=False)["edge_data"]
    wavelength = 299792458.0 / 1e9
    k = 2.0 * np.pi / wavelength
    states = _build_tx_first_order_state_arrays(
        wt.Point3f(2.0, -1.0, 1.0),
        edge_data,
        wavelength,
        k,
        history_size=1,
        scene=scene,
        material_detail={
            "relative_permittivity": 5.0,
            "conductivity": 0.0,
            "gain": 1.0,
        },
        reflection_coef=1.0,
        tx_polarization=(1.0, 1.0, 0.0),
    )

    assert states["n_states"] > 0
    reconstructed = vector_from_jones(
        {
            "u": states["incident_jones_u"],
            "v": states["incident_jones_v"],
        },
        {
            "u": states["incident_basis_u"],
            "v": states["incident_basis_v"],
            "k": states["incident_basis_k"],
        },
    )
    assert _complex_abs_max(reconstructed["x"] - states["incident_vector_x"]) < 1e-6
    assert _complex_abs_max(reconstructed["y"] - states["incident_vector_y"]) < 1e-6
    assert _complex_abs_max(reconstructed["z"] - states["incident_vector_z"]) < 1e-6
    assert _complex_abs_max(states["face0_operator_m00"]) > 1e-6
    assert _complex_abs_max(states["face1_operator_m00"]) > 1e-6


def test_diffraction_face_operator_is_not_collapsed_back_to_scalar_average():
    source_pos = wt.Point3f(-1.5, 0.2, 0.0)
    edge_pos = wt.Point3f(0.0, 0.0, 0.0)
    edge_dir = wt.Vector3f(0.0, 0.0, 1.0)
    n0 = wt.Vector3f(1.0, 0.0, 0.0)
    nn = wt.Vector3f(0.0, 1.0, 0.0)
    target_pos = wt.Point3f(0.3, 1.7, 0.4)
    incident_basis = path_basis(edge_pos - source_pos, preferred=edge_dir)
    zero = wt.Complex2f(0.0, 0.0)
    edge_state = {
        "source_pos": source_pos,
        "edge_pos": edge_pos,
        "edge_dir": edge_dir,
        "n0": n0,
        "n_face_n": nn,
        "wedge_n": wt.Float(1.5),
        "edge_line_min": wt.Float(-20.0),
        "edge_line_max": wt.Float(20.0),
        "adjacent_face0": wt.Int32(0),
        "adjacent_face1": wt.Int32(1),
        "incident_field": zero,
        "incident_normal_derivative": zero,
        "incident_jones_u": wt.Complex2f(1.0, 0.0),
        "incident_jones_v": wt.Complex2f(0.0, 0.0),
        "incident_derivative_jones_u": zero,
        "incident_derivative_jones_v": zero,
        "incident_basis_u": incident_basis["u"],
        "incident_basis_v": incident_basis["v"],
        "incident_basis_k": incident_basis["k"],
        "face0_operator_m00": wt.Complex2f(1.0, 0.0),
        "face0_operator_m01": zero,
        "face0_operator_m10": zero,
        "face0_operator_m11": wt.Complex2f(-1.0, 0.0),
        "face1_operator_m00": zero,
        "face1_operator_m01": zero,
        "face1_operator_m10": zero,
        "face1_operator_m11": zero,
        "r_face0": zero,
        "r_face_n": zero,
        "incident_vector_x": zero,
        "incident_vector_y": zero,
        "incident_vector_z": zero,
        "incident_normal_derivative_vector_x": zero,
        "incident_normal_derivative_vector_y": zero,
        "incident_normal_derivative_vector_z": zero,
    }

    field, vector_field = _edge_state_field_to_targets(
        edge_state,
        target_pos,
        2.0 * np.pi / (299792458.0 / 1e9),
        return_vector=True,
        wavelength=299792458.0 / 1e9,
        material_detail=None,
    )
    outgoing_edge_basis = diffraction_edge_basis(target_pos - edge_pos, edge_dir, outgoing=True)
    outgoing_jones = jones_from_vector(vector_field, outgoing_edge_basis)

    assert _complex_abs_max(outgoing_jones["u"]) > 1e-6 or _complex_abs_max(outgoing_jones["v"]) > 1e-6
    assert _complex_abs_max(field - outgoing_jones["u"]) < 1e-6


def test_diffraction_normal_derivative_scalar_is_derived_from_vector_truth():
    source_pos = wt.Point3f(-1.5, 0.2, 0.0)
    edge_pos = wt.Point3f(0.0, 0.0, 0.0)
    edge_dir = wt.Vector3f(0.0, 0.0, 1.0)
    n0 = wt.Vector3f(1.0, 0.0, 0.0)
    nn = wt.Vector3f(0.0, 1.0, 0.0)
    target_pos = wt.Point3f(0.3, 1.7, 0.4)
    incident_basis = path_basis(edge_pos - source_pos, preferred=edge_dir)
    zero = wt.Complex2f(0.0, 0.0)
    edge_state = {
        "source_pos": source_pos,
        "edge_pos": edge_pos,
        "edge_dir": edge_dir,
        "n0": n0,
        "n_face_n": nn,
        "wedge_n": wt.Float(1.5),
        "edge_line_min": wt.Float(-20.0),
        "edge_line_max": wt.Float(20.0),
        "adjacent_face0": wt.Int32(0),
        "adjacent_face1": wt.Int32(1),
        "incident_field": zero,
        "incident_normal_derivative": zero,
        "incident_jones_u": zero,
        "incident_jones_v": zero,
        "incident_derivative_jones_u": wt.Complex2f(1.0, 0.0),
        "incident_derivative_jones_v": wt.Complex2f(0.25, -0.1),
        "incident_basis_u": incident_basis["u"],
        "incident_basis_v": incident_basis["v"],
        "incident_basis_k": incident_basis["k"],
        "face0_operator_m00": wt.Complex2f(1.0, 0.0),
        "face0_operator_m01": wt.Complex2f(0.15, 0.05),
        "face0_operator_m10": wt.Complex2f(-0.2, 0.1),
        "face0_operator_m11": wt.Complex2f(-1.0, 0.0),
        "face1_operator_m00": zero,
        "face1_operator_m01": zero,
        "face1_operator_m10": zero,
        "face1_operator_m11": zero,
        "r_face0": zero,
        "r_face_n": zero,
        "incident_vector_x": zero,
        "incident_vector_y": zero,
        "incident_vector_z": zero,
        "incident_normal_derivative_vector_x": zero,
        "incident_normal_derivative_vector_y": zero,
        "incident_normal_derivative_vector_z": zero,
    }

    field, normal_derivative, vector_field, normal_derivative_vector = _edge_state_field_to_targets(
        edge_state,
        target_pos,
        2.0 * np.pi / (299792458.0 / 1e9),
        return_normal_derivative=True,
        return_vector=True,
        wavelength=299792458.0 / 1e9,
        material_detail=None,
    )
    outgoing_edge_basis = diffraction_edge_basis(target_pos - edge_pos, edge_dir, outgoing=True)
    outgoing_jones = jones_from_vector(vector_field, outgoing_edge_basis)
    outgoing_derivative_jones = jones_from_vector(normal_derivative_vector, outgoing_edge_basis)

    assert _complex_abs_max(field - outgoing_jones["u"]) < 1e-6
    assert _complex_abs_max(normal_derivative - outgoing_derivative_jones["u"]) < 1e-6


def test_tracer_exposes_jones_fields_for_los_only_case():
    far_cube = box_geometry(center=(500.0, 500.0, 1.5), size=1.0)
    scene = build_test_scene(far_cube)
    tracer = Tracer(
        frequency=1e9,
        scene=scene,
        reflection_n_rays=0,
        reflection_max_bounces=0,
        max_diffractions=1,
        tx_polarization=(0.0, 1.0, 0.0),
    )
    monitor = FieldMonitor(
        "los_plane",
        axis="z",
        position=1.5,
        bounds=((-1.0, 1.0), (-1.0, 1.0)),
        grid_size=9,
    )

    result = tracer.trace(
        tx_pos=wt.Point3f(-2.0, 0.0, 1.5),
        monitor=monitor,
        verbose=False,
    )

    payload = result.primary
    assert payload.metadata["polarization_transport"]["enabled"] is True
    assert payload.metadata["polarization_transport"]["tx_polarization"] == (0.0, 1.0, 0.0)
    assert payload.metadata["polarization_transport"]["rx_polarization"] == (0.0, 1.0, 0.0)
    assert payload.metadata["transport_basis"] == "path_transverse"
    assert payload.metadata["result_jones_basis"] == "global_xy"
    assert payload.metadata["scalar_projection_rule"] == "global_xy_default_receiver_projection_from_jones"
    center_idx = (payload.grid_shape[0] * payload.grid_shape[1]) // 2
    assert abs(float(payload.vector.total["x"].real[center_idx] - payload.jones.total["x"].real[center_idx])) < 1e-6
    assert abs(float(payload.vector.total["y"].real[center_idx] - payload.jones.total["y"].real[center_idx])) < 1e-6
    assert abs(float(payload.jones.total["x"].real[center_idx])) < 1e-6
    assert abs(float(payload.jones.total["x"].imag[center_idx])) < 1e-6
    assert abs(float(payload.jones.total["y"].real[center_idx] - payload.field.total.real[center_idx])) < 1e-6
    assert abs(float(payload.jones.total["y"].imag[center_idx] - payload.field.total.imag[center_idx])) < 5e-6


def test_tracer_rx_polarization_controls_los_scalar_projection():
    far_cube = box_geometry(center=(500.0, 500.0, 1.5), size=1.0)
    scene = build_test_scene(far_cube)
    tracer = Tracer(
        frequency=1e9,
        scene=scene,
        reflection_n_rays=0,
        reflection_max_bounces=0,
        max_diffractions=1,
        tx_polarization=(0.0, 1.0, 0.0),
        rx_polarization=(1.0, 0.0, 0.0),
    )
    monitor = FieldMonitor(
        "los_plane_rx",
        axis="z",
        position=1.5,
        bounds=((-1.0, 1.0), (-1.0, 1.0)),
        grid_size=9,
    )

    result = tracer.trace(
        tx_pos=wt.Point3f(-2.0, 0.0, 1.5),
        monitor=monitor,
        verbose=False,
    )

    payload = result.primary
    projected = scalarize_xy_jones(payload.jones.total, (1.0, 0.0, 0.0))
    center_idx = (payload.grid_shape[0] * payload.grid_shape[1]) // 2
    assert payload.metadata["polarization_transport"]["rx_polarization"] == (1.0, 0.0, 0.0)
    assert abs(float(projected.real[center_idx] - payload.field.total.real[center_idx])) < 1e-6
    assert abs(float(projected.imag[center_idx] - payload.field.total.imag[center_idx])) < 1e-6


def test_diffraction_suffix_reflection_updates_mixed_jones_field():
    scene = build_test_scene(
        box_geometry(center=(-2.0, -2.0, 1.5), size=2.0),
        box_geometry(center=(2.0, 1.5, 1.5), size=2.0),
    )
    field = Field(bounds=((-4.0, 4.0), (-4.0, 4.0)), size=(20, 20))
    coords = field.get_coordinates()
    wavelength = 299792458.0 / 1e9
    k = 2.0 * np.pi / wavelength
    tx = wt.Point3f(0.0, -4.0, 1.5)
    tx_polarization = (0.0, 1.0, 0.0)

    _, _, reflection_detail = compute_reflection_field(
        grid=field,
        rx_z=1.5,
        tx_pos=tx,
        scene=scene,
        wavelength=wavelength,
        k=k,
        n_rays=256,
        max_reflections=1,
        reflection_coef=0.7,
        grid_data=coords,
        tx_polarization=tx_polarization,
    )

    shared_kwargs = dict(
        X=coords["X"],
        Y=coords["Y"],
        rx_z=1.5,
        tx_pos=tx,
        scene=scene,
        wavelength=wavelength,
        k=k,
        reflection_detail=reflection_detail,
        max_diffractions=1,
        reflection_n_rays=256,
        reflection_coef=0.7,
        reflection_mode="2d",
        return_components=True,
        return_per_edge=False,
        tx_polarization=tx_polarization,
    )
    _, _, _, without_suffix = compute_diffraction_field(
        reflection_max_bounces=0,
        grid=None,
        grid_data=None,
        **shared_kwargs,
    )
    _, _, _, with_suffix = compute_diffraction_field(
        reflection_max_bounces=1,
        grid=field,
        grid_data=coords,
        **shared_kwargs,
    )

    scalar_delta = wt.Complex2f(
        with_suffix["a_multi"].real - without_suffix["a_multi"].real,
        with_suffix["a_multi"].imag - without_suffix["a_multi"].imag,
    )
    vector_delta = {
        "x": with_suffix["polarization_multi"]["x"] - without_suffix["polarization_multi"]["x"],
        "y": with_suffix["polarization_multi"]["y"] - without_suffix["polarization_multi"]["y"],
        "z": with_suffix["polarization_multi"]["z"] - without_suffix["polarization_multi"]["z"],
    }

    assert _field_power(scalar_delta) > 1e-12
    assert _vector_power(vector_delta) > 1e-12
    for axis in ("x", "y", "z"):
        assert not bool(dr.any(
            dr.isnan(with_suffix["polarization_multi"][axis].real)
            | dr.isnan(with_suffix["polarization_multi"][axis].imag)
        ))


def test_finite_wedge_keeps_generic_endpoint_anchors():
    scene = _build_endpoint_clamped_scene()
    edge_data = scene.get_edge_data(2.0)["edge_data"]

    assert edge_data is not None
    assert edge_data["n_edges"] == 1
    assert scene.edge_selection_summary.get("excluded_endpoint_anchors", 0) == 0


def test_finite_wedge_state_arrays_keep_edge_bounds():
    scene = _build_endpoint_clamped_scene()
    edge_data = scene.get_edge_data(1.0, include_projection=False)["edge_data"]

    assert edge_data is not None
    assert "line_min" in edge_data
    assert "line_max" in edge_data

    wavelength = 299792458.0 / 1e9
    k = 2.0 * np.pi / wavelength
    states = _build_tx_first_order_state_arrays(
        wt.Point3f(2.0, -1.0, 1.0),
        edge_data,
        wavelength,
        k,
        history_size=1,
        scene=scene,
        material_detail={
            "relative_permittivity": 5.0,
            "conductivity": 0.0,
            "gain": 1.0,
        },
        reflection_coef=1.0,
        tx_polarization=(1.0, 0.0, 0.0),
    )

    assert "edge_line_min" in states
    assert "edge_line_max" in states
    assert float(dr.min(states["edge_line_min"])[0]) <= 0.0
    assert float(dr.max(states["edge_line_max"])[0]) >= 0.0


def test_finite_wedge_field_smoothly_truncates_beyond_segment_endpoints():
    wavelength = 299792458.0 / 1e9
    k = 2.0 * np.pi / wavelength
    long_state = _make_manual_edge_state(line_min=-20.0, line_max=20.0)
    finite_state = _make_manual_edge_state(line_min=-3.0, line_max=3.0)

    inside_target = wt.Point3f(0.3, 1.7, 0.0)
    outside_target = wt.Point3f(0.3, 1.7, 8.0)

    long_inside = _scalar_field_magnitude(
        _edge_state_field_to_targets(
            long_state,
            inside_target,
            k,
            wavelength=wavelength,
            material_detail=None,
        )
    )
    finite_inside = _scalar_field_magnitude(
        _edge_state_field_to_targets(
            finite_state,
            inside_target,
            k,
            wavelength=wavelength,
            material_detail=None,
        )
    )
    long_outside = _scalar_field_magnitude(
        _edge_state_field_to_targets(
            long_state,
            outside_target,
            k,
            wavelength=wavelength,
            material_detail=None,
        )
    )
    finite_outside = _scalar_field_magnitude(
        _edge_state_field_to_targets(
            finite_state,
            outside_target,
            k,
            wavelength=wavelength,
            material_detail=None,
        )
    )

    assert finite_inside > 0.9 * long_inside
    assert finite_outside < 0.8 * long_outside
    assert finite_outside > 0.5 * long_outside


def test_finite_wedge_field_requires_explicit_segment_bounds():
    wavelength = 299792458.0 / 1e9
    k = 2.0 * np.pi / wavelength
    state = _make_manual_edge_state()

    with pytest.raises(RuntimeError, match="edge_line_min and edge_line_max"):
        _edge_state_field_to_targets(
            state,
            wt.Point3f(0.3, 1.7, 0.0),
            k,
            wavelength=wavelength,
            material_detail=None,
        )


def test_tracer_metadata_reports_finite_edge_mode():
    scene = _build_endpoint_clamped_scene()
    tracer = Tracer(
        frequency=1e9,
        scene=scene,
        reflection_n_rays=0,
        reflection_max_bounces=0,
        max_diffractions=1,
    )
    monitor = FieldMonitor(
        "finite_edge_plane",
        axis="z",
        position=2.0,
        bounds=((-1.0, 3.0), (-2.0, 2.0)),
        grid_size=8,
    )

    result = tracer.trace(
        tx_pos=wt.Point3f(-3.0, -1.0, 2.0),
        monitor=monitor,
        verbose=False,
    )

    treatment = result.primary.metadata["finite_edge_treatment"]
    assert treatment["mode"] == "finite_wedge"
    assert treatment["bounds_required"] is True
    assert "finite-wedge edge bounds" in treatment["notes"]


def test_finite_wedge_radiomap_backend_allows_native_coherent(monkeypatch):
    monkeypatch.setattr(radio_map_helpers, "native_extension_available", lambda: True)
    monitor = SimpleNamespace(combine_mode="coherent", receiver_model="projected_polarized")
    grid = SimpleNamespace(surface_mode="axis_aligned")
    config = SimpleNamespace(
        reflection_field_backend="native",
        diffraction_execution=SimpleNamespace(suffix_backend="native"),
        use_scene_materials_for_reflection=False,
        use_scene_materials_for_diffraction=False,
    )
    scene = SimpleNamespace(vertices=None, tri_data_gpu=None)
    tx_pos = wt.Point3f(0.0, 0.0, 0.0)

    resolved = radio_map_helpers._resolve_radio_map_accumulation_backend(
        requested_backend="auto",
        monitor=monitor,
        grid=grid,
        config=config,
        tx_pos=tx_pos,
        scene=scene,
    )

    assert resolved == "native_coherent"
    assert radio_map_helpers._resolve_radio_map_accumulation_backend(
        requested_backend="native_coherent",
        monitor=monitor,
        grid=grid,
        config=config,
        tx_pos=tx_pos,
        scene=scene,
    ) == "native_coherent"


