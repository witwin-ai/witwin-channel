"""Phase 4 regression coverage for tracer-level axis-aligned field monitors."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import drjit as dr
import numpy as np
import pytest
import witwin as wt

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tests._scene_helpers import box_geometry, build_scene
from witwin.channel import DrJitMesh, FieldMonitor, Tracer, compute_los_field
from witwin.channel.utils.polarization import (
    project_real_polarization_to_ray,
    scalarize_tangential_jones,
    tangential_jones,
    vector_eval,
    vector_from_scalar_and_real_direction,
)
from witwin.channel.utils import to_point3f, to_vector3u
from witwin.channel.validation import build_mixed_prefix_suffix_case, build_single_wedge_case


FREQUENCY = 1e9
WAVELENGTH = 299792458.0 / FREQUENCY
WAVENUMBER = 2.0 * math.pi / WAVELENGTH
FAR_CUBE_CENTER = (500.0, 500.0, 1.5)
TX_POS = (0.0, -4.0, 1.5)
PHASE4A_BOUNDS = ((-2.5, 2.5), (-2.0, 2.0))


def _far_cube_scene():
    return build_scene(box_geometry(center=FAR_CUBE_CENTER, size=1.0))


def _expected_los_scalar(scene, field, tx_pos, tx_polarization, rx_polarization):
    rx_positions = field.receivers
    tx_point = wt.Point3f(*tx_pos)
    a_los = compute_los_field(scene, rx_positions, tx_point, WAVELENGTH, WAVENUMBER)
    los_ray_dir = rx_positions - tx_point
    los_ray_dir = los_ray_dir / (dr.norm(los_ray_dir) + 1e-12)
    los_pol_dir = project_real_polarization_to_ray(tx_polarization, los_ray_dir)
    polarization_los = vector_eval(vector_from_scalar_and_real_direction(a_los, los_pol_dir))
    return scalarize_tangential_jones(
        tangential_jones(polarization_los, axis=field.axis),
        rx_polarization,
        axis=field.axis,
    )


def _project_jones_field(jones_field, *, axis: str, rx_polarization):
    return scalarize_tangential_jones(jones_field, rx_polarization, axis=axis)


@pytest.mark.gpu
@pytest.mark.parametrize(
    ("axis", "position", "tx_polarization", "expected_axes", "expected_basis"),
    [
        ("x", 1.5, (0.0, 1.0, 0.0), ("y", "z"), "monitor_tangential"),
        ("y", 1.5, (1.0, 0.0, 0.0), ("x", "z"), "monitor_tangential"),
        ("z", 1.5, (1.0, 0.0, 0.0), ("x", "y"), "global_xy"),
    ],
)
def test_tracer_los_supports_axis_aligned_monitors_when_diffraction_is_disabled(
    axis,
    position,
    tx_polarization,
    expected_axes,
    expected_basis,
):
    scene = _far_cube_scene()
    monitor = FieldMonitor(
        f"plane_{axis}",
        axis=axis,
        position=position,
        bounds=PHASE4A_BOUNDS,
        grid_size=(6, 5),
    )
    tracer = Tracer(
        frequency=FREQUENCY,
        scene=scene,
        reflection_n_rays=0,
        reflection_max_bounces=0,
        max_diffractions=0,
        tx_polarization=tx_polarization,
    )

    payload = tracer.trace(wt.Point3f(*TX_POS), monitor=monitor, verbose=False)
    field = monitor.to_field(WAVELENGTH)
    expected = _expected_los_scalar(scene, field, TX_POS, tx_polarization, tx_polarization)

    np.testing.assert_allclose(
        np.asarray(payload.field.los.real, dtype=np.float64),
        np.asarray(expected.real, dtype=np.float64),
        rtol=1e-6,
        atol=1e-8,
    )
    np.testing.assert_allclose(
        np.asarray(payload.field.los.imag, dtype=np.float64),
        np.asarray(expected.imag, dtype=np.float64),
        rtol=1e-6,
        atol=1e-8,
    )
    np.testing.assert_allclose(
        np.asarray(payload.field.total.real, dtype=np.float64),
        np.asarray(expected.real, dtype=np.float64),
        rtol=1e-6,
        atol=1e-8,
    )
    np.testing.assert_allclose(
        np.asarray(payload.field.total.imag, dtype=np.float64),
        np.asarray(expected.imag, dtype=np.float64),
        rtol=1e-6,
        atol=1e-8,
    )
    np.testing.assert_allclose(np.asarray(payload.field.reflection.real, dtype=np.float64), 0.0, atol=1e-8)
    np.testing.assert_allclose(np.asarray(payload.field.diffraction.real, dtype=np.float64), 0.0, atol=1e-8)
    assert tuple(payload.tangential_axes) == expected_axes
    assert tuple(payload.metadata["result_jones_axes"]) == expected_axes
    assert payload.metadata["result_jones_basis"] == expected_basis
    assert tuple(payload.jones.total.keys()) == expected_axes
    assert payload.metadata["receiver_sampling"]["backend"] == (
        "dda_planar_grid" if axis == "z" else "axis_aligned_plane_monitor"
    )
    assert payload.metadata["diffraction_skipped"] is True
    assert payload.metadata["diffraction_skip_reason"] == "max_diffractions_disabled"


@pytest.mark.gpu
def test_tracer_supports_mixed_z_and_x_monitors_when_diffraction_is_disabled():
    scene = _far_cube_scene()
    monitors = [
        FieldMonitor(
            "xy_plane",
            axis="z",
            position=1.5,
            bounds=PHASE4A_BOUNDS,
            grid_size=(5, 4),
        ),
        FieldMonitor(
            "yz_plane",
            axis="x",
            position=1.5,
            bounds=PHASE4A_BOUNDS,
            grid_size=(5, 4),
        ),
    ]
    tracer = Tracer(
        frequency=FREQUENCY,
        scene=scene,
        reflection_n_rays=0,
        reflection_max_bounces=0,
        max_diffractions=0,
        tx_polarization=(0.0, 1.0, 0.0),
    )

    result = tracer.trace(wt.Point3f(*TX_POS), monitor=monitors, verbose=False)

    assert set(result.keys()) == {"xy_plane", "yz_plane"}
    assert result["xy_plane"].name == "xy_plane"
    assert result["xy_plane"].axis == "z"
    assert result["yz_plane"].axis == "x"
    assert tuple(result["yz_plane"].tangential_axes) == ("y", "z")
    assert tuple(result["yz_plane"].metadata["result_jones_axes"]) == ("y", "z")
    assert result["yz_plane"].metadata["diffraction_skipped"] is True


def _rotate_mesh_y(mesh_geometry, degrees: float) -> DrJitMesh:
    vertices, faces = mesh_geometry.to_mesh()
    transform = wt.Transform4f().rotate(wt.Vector3f(0.0, 1.0, 0.0), wt.Float(degrees))
    return DrJitMesh(transform @ to_point3f(vertices), to_vector3u(faces))


def _rotate_point_y_90(point):
    x, y, z = point
    return (z, y, -x)


def _rotate_mesh_z(mesh_geometry, degrees: float) -> DrJitMesh:
    vertices, faces = mesh_geometry.to_mesh()
    transform = wt.Transform4f().rotate(wt.Vector3f(0.0, 0.0, 1.0), wt.Float(degrees))
    return DrJitMesh(transform @ to_point3f(vertices), to_vector3u(faces))


def _rotate_point_z_90(point):
    x, y, z = point
    return (-y, x, z)


def _scalar_field_magnitude_matrix(field_component, grid_shape):
    dr.eval(field_component.real, field_component.imag)
    magnitude = (
        np.asarray(field_component.real, dtype=np.float64) ** 2
        + np.asarray(field_component.imag, dtype=np.float64) ** 2
    )
    return np.sqrt(magnitude).reshape(grid_shape[1], grid_shape[0])


def _field_power(field_component):
    return float(dr.sum(field_component.real * field_component.real + field_component.imag * field_component.imag)[0])


@pytest.mark.gpu
def test_tracer_x_normal_matches_rotated_z_reference_for_los_plus_reflection():
    ceiling_center = (0.0, 0.0, 6.0)
    ceiling_size = (8.0, 10.0, 0.25)
    base_scene = build_scene(box_geometry(center=ceiling_center, size=ceiling_size))
    base_monitor = FieldMonitor(
        "xy_ref",
        axis="z",
        position=1.5,
        bounds=((-6.0, 6.0), (-6.0, 6.0)),
        grid_size=(16, 16),
        ray_mode="3d",
    )
    base_tracer = Tracer(
        frequency=FREQUENCY,
        scene=base_scene,
        reflection_n_rays=16384,
        reflection_max_bounces=1,
        reflection_coef=0.82,
        max_diffractions=0,
        tx_polarization=(0.0, 1.0, 0.0),
    )
    base_result = base_tracer.trace(wt.Point3f(-1.0, -5.0, 4.5), monitor=base_monitor, verbose=False)

    rotated_scene = build_scene(_rotate_mesh_y(box_geometry(center=ceiling_center, size=ceiling_size), 90.0))
    rotated_monitor = FieldMonitor(
        "yz_ref",
        axis="x",
        position=1.5,
        bounds=((-6.0, 6.0), (-6.0, 6.0)),
        grid_size=(16, 16),
        ray_mode="3d",
    )
    rotated_tracer = Tracer(
        frequency=FREQUENCY,
        scene=rotated_scene,
        reflection_n_rays=65536,
        reflection_max_bounces=1,
        reflection_coef=0.82,
        max_diffractions=0,
        tx_polarization=(0.0, 1.0, 0.0),
    )
    rotated_result = rotated_tracer.trace(
        wt.Point3f(*_rotate_point_y_90((-1.0, -5.0, 4.5))),
        monitor=rotated_monitor,
        verbose=False,
    )

    base_magnitude = _scalar_field_magnitude_matrix(base_result.field.total, base_result.grid_shape)
    rotated_magnitude = _scalar_field_magnitude_matrix(rotated_result.field.total, rotated_result.grid_shape)
    base_db = 20.0 * np.log10(np.maximum(base_magnitude[:, ::-1].T, 1e-20))
    rotated_db = 20.0 * np.log10(np.maximum(rotated_magnitude, 1e-20))
    mask = (base_magnitude[:, ::-1].T > base_magnitude.max() * 1e-1) | (
        rotated_magnitude > rotated_magnitude.max() * 1e-1
    )
    rms_db = float(np.sqrt(np.mean((rotated_db[mask] - base_db[mask]) ** 2)))

    assert rotated_result.metadata["result_jones_basis"] == "monitor_tangential"
    assert rotated_result.metadata["diffraction_skipped"] is True
    assert rms_db < 0.75


@pytest.mark.gpu
@pytest.mark.parametrize(
    ("axis", "position", "bounds", "tx_polarization", "expected_axes"),
    [
        ("x", 0.0, ((-6.0, 6.0), (0.5, 2.5)), (0.0, 0.0, 1.0), ("y", "z")),
        ("y", 1.0, ((-6.0, 6.0), (0.5, 2.5)), (1.0, 0.0, 0.0), ("x", "z")),
    ],
)
def test_tracer_supports_axis_aligned_diffraction_on_x_and_y_monitors(
    axis,
    position,
    bounds,
    tx_polarization,
    expected_axes,
):
    case = build_single_wedge_case()
    monitor = FieldMonitor(
        f"{axis}_diffraction_plane",
        axis=axis,
        position=position,
        bounds=bounds,
        grid_size=(12, 10),
    )
    tracer = Tracer(
        frequency=FREQUENCY,
        scene=case.scene,
        reflection_n_rays=0,
        reflection_max_bounces=0,
        max_diffractions=1,
        tx_polarization=tx_polarization,
    )

    payload = tracer.trace(
        wt.Point3f(*case.tx_pos),
        monitor=monitor,
        verbose=False,
        return_diffraction_audit=True,
    )

    assert _field_power(payload.field.diffraction) > 1e-10
    assert payload.metadata["result_jones_basis"] == "monitor_tangential"
    assert tuple(payload.metadata["result_jones_axes"]) == expected_axes
    assert tuple(payload.jones.diffraction.keys()) == expected_axes
    assert payload.diffraction_detail["state_audit"]["n_states"] > 0


@pytest.mark.gpu
def test_tracer_supports_mixed_z_and_x_monitors_with_diffraction_enabled():
    case = build_mixed_prefix_suffix_case()
    monitors = [
        FieldMonitor(
            "xy_full",
            axis="z",
            position=case.calculation_height,
            bounds=((-6.0, 6.0), (-6.0, 6.0)),
            grid_size=(12, 12),
            ray_mode="3d",
        ),
        FieldMonitor(
            "yz_full",
            axis="x",
            position=0.0,
            bounds=((-6.0, 6.0), (0.5, 2.5)),
            grid_size=(12, 10),
            ray_mode="3d",
        ),
    ]
    tracer = Tracer(
        frequency=FREQUENCY,
        scene=case.scene,
        reflection_n_rays=1024,
        reflection_max_bounces=1,
        max_diffractions=1,
        enable_rd_diffraction=True,
        tx_polarization=(1.0, 0.0, 1.0),
    )

    result = tracer.trace(
        wt.Point3f(*case.tx_pos),
        monitor=monitors,
        verbose=False,
        return_diffraction_audit=True,
    )

    xy_payload = result["xy_full"]
    yz_payload = result["yz_full"]
    assert _field_power(xy_payload.field.diffraction) > 1e-10
    assert _field_power(yz_payload.field.diffraction) > 1e-10
    assert _field_power(yz_payload.field.reflection) > 1e-10
    assert tuple(xy_payload.metadata["result_jones_axes"]) == ("x", "y")
    assert tuple(yz_payload.metadata["result_jones_axes"]) == ("y", "z")
    assert yz_payload.diffraction_detail["state_audit"]["n_states"] > 0


@pytest.mark.gpu
def test_tracer_scalar_components_match_projected_jones_fields_for_mixed_multipath_scene():
    case = build_mixed_prefix_suffix_case()
    monitor = FieldMonitor(
        "xy_scalarization_regression",
        axis="z",
        position=case.calculation_height,
        bounds=(case.range_x, case.range_y),
        grid_size=(24, 24),
        ray_mode="3d",
    )
    tracer = Tracer(
        frequency=FREQUENCY,
        scene=case.scene,
        reflection_n_rays=1024,
        reflection_max_bounces=2,
        reflection_coef=0.8,
        max_diffractions=2,
        enable_rd_diffraction=True,
        tx_polarization=(1.0, 0.0, 0.0),
    )

    payload = tracer.trace(
        wt.Point3f(*case.tx_pos),
        monitor=monitor,
        verbose=False,
        return_diffraction_audit=True,
    )
    rx_polarization = payload.metadata["polarization_transport"]["rx_polarization"]

    for component_name in (
        "reflection",
        "diffraction_direct",
        "diffraction_mixed",
        "diffraction",
        "total",
    ):
        expected = _project_jones_field(
            getattr(payload.jones, component_name),
            axis=payload.axis,
            rx_polarization=rx_polarization,
        )
        actual = getattr(payload.field, component_name)
        np.testing.assert_allclose(
            np.asarray(actual.real, dtype=np.float64),
            np.asarray(expected.real, dtype=np.float64),
            rtol=1e-6,
            atol=1e-8,
            err_msg=f"{component_name} real part must match projected Jones field",
        )
        np.testing.assert_allclose(
            np.asarray(actual.imag, dtype=np.float64),
            np.asarray(expected.imag, dtype=np.float64),
            rtol=1e-6,
            atol=1e-8,
            err_msg=f"{component_name} imaginary part must match projected Jones field",
        )


@pytest.mark.gpu
def test_tracer_x_normal_matches_rotated_y_reference_for_diffraction():
    case = build_single_wedge_case()
    base_scene = case.scene
    rotated_scene = build_scene(
        *[
            _rotate_mesh_z(structure.geometry, 90.0)
            for structure in base_scene.structures
        ],
        edge_selection_mode=base_scene.edge_selection_mode,
        boundary_edge_policy=base_scene.boundary_edge_policy,
    )
    base_monitor = FieldMonitor(
        "xz_diff",
        axis="y",
        position=1.0,
        bounds=((-6.0, 6.0), (0.5, 2.5)),
        grid_size=(18, 10),
    )
    rotated_monitor = FieldMonitor(
        "yz_diff",
        axis="x",
        position=-1.0,
        bounds=((-6.0, 6.0), (0.5, 2.5)),
        grid_size=(18, 10),
    )
    base_tracer = Tracer(
        frequency=FREQUENCY,
        scene=base_scene,
        reflection_n_rays=0,
        reflection_max_bounces=0,
        max_diffractions=1,
        tx_polarization=(0.0, 0.0, 1.0),
    )
    rotated_tracer = Tracer(
        frequency=FREQUENCY,
        scene=rotated_scene,
        reflection_n_rays=0,
        reflection_max_bounces=0,
        max_diffractions=1,
        tx_polarization=(0.0, 0.0, 1.0),
    )

    base_result = base_tracer.trace(
        wt.Point3f(*case.tx_pos),
        monitor=base_monitor,
        verbose=False,
    )
    rotated_result = rotated_tracer.trace(
        wt.Point3f(*_rotate_point_z_90(case.tx_pos)),
        monitor=rotated_monitor,
        verbose=False,
    )

    base_magnitude = _scalar_field_magnitude_matrix(base_result.field.diffraction, base_result.grid_shape)
    rotated_magnitude = _scalar_field_magnitude_matrix(rotated_result.field.diffraction, rotated_result.grid_shape)
    base_db = 20.0 * np.log10(np.maximum(base_magnitude, 1e-20))
    rotated_db = 20.0 * np.log10(np.maximum(rotated_magnitude, 1e-20))
    mask = (base_magnitude > base_magnitude.max() * 1e-2) | (
        rotated_magnitude > rotated_magnitude.max() * 1e-2
    )
    rms_db = float(np.sqrt(np.mean((rotated_db[mask] - base_db[mask]) ** 2)))

    assert rotated_result.metadata["result_jones_basis"] == "monitor_tangential"
    assert rms_db < 0.75
