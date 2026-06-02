"""Regression tests for boundary-consistent diffraction validity handling."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import witwin as wt

import drjit as dr

from tests._scene_helpers import build_scene
from witwin.channel import FieldMonitor, Tracer
from witwin.channel.utils.polarization import path_basis, jones_from_vector, vector_from_jones
from witwin.channel.trace.diffraction import _edge_state_field_to_targets, _wedge_exterior_region_mask
def _field_power(field):
    return float(dr.sum(field.real * field.real + field.imag * field.imag)[0])


def test_wedge_exterior_region_mask_uses_face_half_spaces():
    edge_dir = wt.Vector3f(0.0, 0.0, 1.0)
    n0 = wt.Vector3f(1.0, 0.0, 0.0)
    nn = wt.Vector3f(0.0, 1.0, 0.0)

    assert bool(_wedge_exterior_region_mask(wt.Vector3f(1.0, 1.0, 0.0), edge_dir, n0, nn)[0])
    assert bool(_wedge_exterior_region_mask(wt.Vector3f(-1.0, 0.0, 0.0), edge_dir, n0, nn)[0])
    assert not bool(_wedge_exterior_region_mask(wt.Vector3f(-1.0, -1.0, 0.0), edge_dir, n0, nn)[0])


def _canonicalize_test_edge_state(state):
    """Add jones+basis+material keys to a hand-built edge state dict for test use."""
    source_pos = state["source_pos"]
    edge_pos = state["edge_pos"]
    edge_dir = state["edge_dir"]
    basis = path_basis(edge_pos - source_pos, preferred=edge_dir)
    incident_field = state["incident_field"]
    incident_nd = state["incident_normal_derivative"]
    zero = wt.Complex2f(wt.Float(0.0), wt.Float(0.0))
    one = wt.Complex2f(wt.Float(1.0), wt.Float(0.0))
    # Place the field in the transverse plane (v component) for nonzero UTD diffraction.
    incident_vector = vector_from_jones({"u": zero, "v": incident_field}, basis)
    jones = jones_from_vector(incident_vector, basis)
    nd_vector = vector_from_jones({"u": incident_nd, "v": zero}, basis)
    nd_jones = jones_from_vector(nd_vector, basis)
    state.update({
        "edge_line_min": state.get("edge_line_min", wt.Float(-10.0)),
        "edge_line_max": state.get("edge_line_max", wt.Float(10.0)),
        "incident_jones_u": jones["u"],
        "incident_jones_v": jones["v"],
        "incident_derivative_jones_u": nd_jones["u"],
        "incident_derivative_jones_v": nd_jones["v"],
        "incident_basis_u": basis["u"],
        "incident_basis_v": basis["v"],
        "incident_basis_k": basis["k"],
        "incident_vector_x": incident_vector["x"],
        "incident_vector_y": incident_vector["y"],
        "incident_vector_z": incident_vector["z"],
        "incident_normal_derivative_vector_x": nd_vector["x"],
        "incident_normal_derivative_vector_y": nd_vector["y"],
        "incident_normal_derivative_vector_z": nd_vector["z"],
        "r_face0": zero,
        "r_face_n": zero,
        "adjacent_face0": wt.Int32(0),
        "adjacent_face1": wt.Int32(1),
        "face0_operator_m00": one,
        "face0_operator_m01": zero,
        "face0_operator_m10": zero,
        "face0_operator_m11": one,
        "face1_operator_m00": one,
        "face1_operator_m01": zero,
        "face1_operator_m10": zero,
        "face1_operator_m11": one,
    })
    return state


def test_boundary_consistent_field_rejects_interior_targets_without_smoothing():
    edge_state = _canonicalize_test_edge_state({
        "source_pos": wt.Point3f(2.0, 0.5, 0.0),
        "edge_pos": wt.Point3f(0.0, 0.0, 0.0),
        "edge_dir": wt.Vector3f(0.0, 0.0, 1.0),
        "n0": wt.Vector3f(1.0, 0.0, 0.0),
        "n_face_n": wt.Vector3f(0.0, 1.0, 0.0),
        "wedge_n": wt.Float(1.5),
        "incident_field": wt.Complex2f(1.0, 0.0),
        "incident_normal_derivative": wt.Complex2f(0.0, 0.0),
    })

    interior_target = wt.Point3f(-1.0, -1.0, 0.0)
    boundary_target = wt.Point3f(-2.0, 0.1, 0.0)
    exterior_target = wt.Point3f(-2.0, 1.0, 0.0)

    interior_field = _edge_state_field_to_targets(edge_state, interior_target, 2.0 * dr.pi, wavelength=1.0)
    boundary_field = _edge_state_field_to_targets(edge_state, boundary_target, 2.0 * dr.pi, wavelength=1.0)
    exterior_field = _edge_state_field_to_targets(edge_state, exterior_target, 2.0 * dr.pi, wavelength=1.0)

    assert _field_power(interior_field) == 0.0
    assert _field_power(boundary_field) > 1e-8
    assert _field_power(exterior_field) > 1e-8


def test_tracer_metadata_reports_geometric_shadow_boundary_treatment():
    points = [
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 2.0),
        (2.0, 0.0, 0.0),
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
    scene = build_scene((vertices, faces))
    tracer = Tracer(
        frequency=1e9,
        scene=scene,
        reflection_n_rays=0,
        reflection_max_bounces=0,
        max_diffractions=1,
    )
    monitor = FieldMonitor(
        "shadow_plane",
        axis="z",
        position=1.0,
        bounds=((-3.0, 3.0), (-3.0, 3.0)),
        grid_size=8,
    )

    result = tracer.trace(
        tx_pos=wt.Point3f(3.0, 0.5, 1.0),
        monitor=monitor,
        verbose=False,
    )

    treatment = result.primary.metadata["shadow_boundary_treatment"]
    assert treatment["mode"] == "geometric_half_space_classification"
    assert "phi/phi_prime angular clipping" in treatment["notes"]


