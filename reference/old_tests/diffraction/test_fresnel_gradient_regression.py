"""Regression tests for Fresnel-driven diffraction gradients."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import witwin as wt

import drjit as dr

from tests._scene_helpers import box_geometry, build_scene as build_test_scene
from witwin.channel import Field
from witwin.channel.trace.diffraction import compute_diffraction_field
from witwin.channel.trace import compute_reflection_field
from witwin.channel.utils.material import scalar_fresnel_reflection
from witwin.channel.utils.polarization import (
    complex_dot_real,
    polarization_consistent_scalar_reflection,
    project_real_polarization_to_ray,
    reflect_field_vector,
    vector_from_scalar_and_real_direction,
)


FREQUENCY = 1e9
WAVELENGTH = 299792458.0 / FREQUENCY
WAVENUMBER = 2.0 * np.pi / WAVELENGTH
OMEGA = wt.Float(2.0 * np.pi * FREQUENCY)
TX_POLARIZATION = (1.0, 0.0, 0.0)
FLAGS = dr.ADFlag.Default | dr.ADFlag.AllowNoGrad
GRAD_DIFFRACTION_EXECUTION = {
    "suffix_dda": "symbolic",
}
DIRECT_DIFFRACTION_TEST_EPS = 1e-2


def build_scene():
    cube1 = box_geometry(center=(-2.8, -2.4, 1.5), size=2.0)
    cube2 = box_geometry(center=(2.6, 0.3, 1.5), size=2.0)
    cube3 = box_geometry(center=(-0.4, 2.9, 1.5), size=1.8)
    return build_test_scene(cube1, cube2, cube3)


def _override_scene_runtime_material(scene, eps_r, sigma_e=0.0):
    n_triangles = int(scene.tri_data_gpu["n_triangles"])
    if isinstance(eps_r, wt.Float) and int(dr.width(eps_r)) == 1:
        eps_r_values = dr.repeat(eps_r, n_triangles)
    else:
        eps_r_values = dr.full(wt.Float, float(eps_r), n_triangles)
    sigma_e_values = dr.full(wt.Float, float(sigma_e), n_triangles)
    specified = dr.full(wt.Bool, True, n_triangles)
    structure_idx = dr.full(wt.Int32, 0, n_triangles)
    scene.tri_data_gpu["material_eps_r"] = eps_r_values
    scene.tri_data_gpu["material_sigma_e"] = sigma_e_values
    scene.tri_data_gpu["material_specified"] = specified
    scene.tri_data_gpu["material_structure_idx"] = structure_idx
    scene.tri_data_gpu["material_has_specified_materials"] = True
    scene.tri_data_gpu["material_n_specified_triangles"] = n_triangles
    scene.tri_data_gpu["material_n_default_material_triangles"] = 0
    scene._triangle_material_data = {
        "eps_r": eps_r_values,
        "sigma_e": sigma_e_values,
        "specified": specified,
        "structure_idx": structure_idx,
        "has_specified_materials": True,
        "n_specified_triangles": n_triangles,
        "n_default_material_triangles": 0,
    }


def _direct_diffraction_field(eps_r):
    scene = build_scene()
    _override_scene_runtime_material(scene, eps_r)
    field = Field(bounds=((-6.5, 6.5), (-6.5, 6.5)), size=(24, 24))
    coords = field.get_coordinates()
    tx = wt.Point3f(0.0, -5.2, 1.5)
    dif_real, dif_imag, _ = compute_diffraction_field(
        coords["X"],
        coords["Y"],
        1.5,
        tx,
        scene,
        WAVELENGTH,
        WAVENUMBER,
        reflection_detail=None,
        max_diffractions=1,
        reflection_n_rays=0,
        reflection_max_bounces=0,
        reflection_coef=0.82,
        reflection_mode="2d",
        grid=field,
        grid_data=coords,
        return_components=False,
        return_per_edge=False,
        diffraction_material=None,
        use_scene_materials=True,
        tx_polarization=TX_POLARIZATION,
        execution=GRAD_DIFFRACTION_EXECUTION,
    )
    return dif_real, dif_imag


def fresnel_real_value(cos_theta: float) -> float:
    coeff = scalar_fresnel_reflection(
        cos_theta=wt.Float(cos_theta),
        eta_r=wt.Float(5.0),
        sigma=wt.Float(0.0),
        omega=OMEGA,
        gain=wt.Float(0.82),
    )
    return float(coeff.real[0])


def test_polarization_consistent_scalar_reflection_matches_vector_projection():
    incident_dir = wt.Vector3f(0.0, 0.0, -1.0)
    normal = wt.Vector3f(0.0, 0.0, 1.0)
    coeff = polarization_consistent_scalar_reflection(
        incident_dir=incident_dir,
        normal=normal,
        eta_r=wt.Float(2.0),
        sigma=wt.Float(0.0),
        omega=OMEGA,
        gain=wt.Float(1.0),
        polarization=TX_POLARIZATION,
    )

    incident_basis = project_real_polarization_to_ray(TX_POLARIZATION, incident_dir)
    incident_vector = vector_from_scalar_and_real_direction(wt.Complex2f(1.0, 0.0), incident_basis)
    reflected_vector = reflect_field_vector(
        incident_vector,
        incident_dir,
        normal,
        eta_r=wt.Float(2.0),
        sigma=wt.Float(0.0),
        omega=OMEGA,
        gain=wt.Float(1.0),
    )
    reflected_dir = incident_dir - 2.0 * dr.dot(incident_dir, normal) * normal
    reflected_basis = project_real_polarization_to_ray(TX_POLARIZATION, reflected_dir)
    projected = complex_dot_real(reflected_vector, reflected_basis)

    old_coeff = scalar_fresnel_reflection(
        cos_theta=wt.Float(1.0),
        eta_r=wt.Float(2.0),
        sigma=wt.Float(0.0),
        omega=OMEGA,
        gain=wt.Float(1.0),
    )

    assert abs(float(coeff.real[0] - projected.real[0])) < 1e-6
    assert abs(float(coeff.imag[0] - projected.imag[0])) < 1e-6
    assert abs(float(old_coeff.real[0])) < 1e-3
    assert abs(float(coeff.real[0])) > 1e-2


def test_lossless_scalar_fresnel_gradient_matches_fd():
    cos_values = np.array(
        [0.28734788, 0.33500963, 0.49483863, 0.62469506, 0.98588660, 0.98994946],
        dtype=np.float64,
    )
    cos_theta = wt.Float(cos_values.tolist())
    dr.enable_grad(cos_theta)
    dr.set_grad(cos_theta, wt.Float([1.0] * len(cos_values)))

    coeff = scalar_fresnel_reflection(
        cos_theta=cos_theta,
        eta_r=wt.Float(5.0),
        sigma=wt.Float(0.0),
        omega=OMEGA,
        gain=wt.Float(0.82),
    )
    dr.forward_to(coeff.real, flags=FLAGS)
    ad = np.array(dr.grad(coeff.real).numpy(), dtype=np.float64)

    eps = 1e-4
    fd = np.array(
        [
            (fresnel_real_value(value + eps) - fresnel_real_value(value - eps)) / (2.0 * eps)
            for value in cos_values
        ],
        dtype=np.float64,
    )
    rel_err = np.abs(ad - fd) / np.maximum(np.abs(fd), 1e-8)

    assert np.isfinite(ad).all(), f"AD gradient contains non-finite entries: {ad}"
    assert np.max(rel_err) < 1e-3, f"Lossless Fresnel AD/FD mismatch: ad={ad}, fd={fd}, rel={rel_err}"


def test_diffraction_gradient_stays_finite_with_reflection_detail():
    scene = build_scene()
    diffraction_materials = (
        None,
        {
            "relative_permittivity": 5.0,
            "conductivity": 0.0,
            "gain": 0.82,
        },
    )

    for diffraction_material in diffraction_materials:
        field = Field(bounds=((-6.5, 6.5), (-6.5, 6.5)), size=(16, 16))
        coords = field.get_coordinates()
        tx_x = wt.Float(0.0)
        dr.enable_grad(tx_x)
        dr.set_grad(tx_x, 1.0)
        tx = wt.Point3f(tx_x, -5.2, 1.5)

        _, _, reflection_detail = compute_reflection_field(
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
        dif_real, _, _ = compute_diffraction_field(
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
            return_components=False,
            return_per_edge=False,
            diffraction_material=diffraction_material,
            tx_polarization=TX_POLARIZATION,
            execution=GRAD_DIFFRACTION_EXECUTION,
        )
        dr.forward_to(dif_real, flags=FLAGS)
        grad = np.array(dr.grad(dif_real).numpy(), dtype=np.float64)

        assert np.isfinite(grad).all(), (
            "Diffraction gradient contains non-finite entries for "
            f"diffraction_material={diffraction_material}: {grad}"
        )
        assert np.linalg.norm(grad) > 1e-6, (
            "Diffraction gradient unexpectedly vanished for "
            f"diffraction_material={diffraction_material}"
        )


def test_direct_diffraction_material_response_changes_and_matches_fd():
    low_real, low_imag = _direct_diffraction_field(1.1)
    high_real, high_imag = _direct_diffraction_field(4.0)
    low_field = np.asarray(low_real.numpy(), dtype=np.float64) + 1j * np.asarray(low_imag.numpy(), dtype=np.float64)
    high_field = np.asarray(high_real.numpy(), dtype=np.float64) + 1j * np.asarray(high_imag.numpy(), dtype=np.float64)
    relative_field_change = np.linalg.norm(high_field - low_field) / np.maximum(np.linalg.norm(low_field), 1e-12)
    assert relative_field_change > 0.05, (
        "Direct diffraction field should respond measurably to scene material changes. "
        f"Observed relative change: {relative_field_change}"
    )

    eps_r = wt.Float(2.0)
    dr.enable_grad(eps_r)
    dr.set_grad(eps_r, 1.0)
    dif_real, dif_imag = _direct_diffraction_field(eps_r)
    dr.forward_to(dif_real, dif_imag, flags=FLAGS)
    ad = np.sqrt(
        np.asarray(dr.grad(dif_real).numpy(), dtype=np.float64) ** 2
        + np.asarray(dr.grad(dif_imag).numpy(), dtype=np.float64) ** 2
    )

    dif_real_p, dif_imag_p = _direct_diffraction_field(2.0 + DIRECT_DIFFRACTION_TEST_EPS)
    dif_real_m, dif_imag_m = _direct_diffraction_field(2.0 - DIRECT_DIFFRACTION_TEST_EPS)
    fd = np.sqrt(
        (
            (
                np.asarray(dif_real_p.numpy(), dtype=np.float64)
                - np.asarray(dif_real_m.numpy(), dtype=np.float64)
            )
            / (2.0 * DIRECT_DIFFRACTION_TEST_EPS)
        )
        ** 2
        + (
            (
                np.asarray(dif_imag_p.numpy(), dtype=np.float64)
                - np.asarray(dif_imag_m.numpy(), dtype=np.float64)
            )
            / (2.0 * DIRECT_DIFFRACTION_TEST_EPS)
        )
        ** 2
    )
    relative_gradient_error = np.linalg.norm(ad - fd) / np.maximum(np.linalg.norm(fd), 1e-12)
    assert np.sum(ad) > 1e-6
    assert relative_gradient_error < 1e-2, (
        "Direct diffraction material AD gradient should match finite differences. "
        f"Observed relative error: {relative_gradient_error}"
    )


