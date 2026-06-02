"""Regression tests for Fresnel/material-dependent reflection response."""

import math
import sys
from pathlib import Path
TEST_FILE = Path(__file__).resolve()
CHANNEL_ROOT = TEST_FILE.parents[2]
CORE_ROOT = CHANNEL_ROOT.parent / "core"
sys.path.insert(0, str(CORE_ROOT))
sys.path.insert(0, str(CHANNEL_ROOT))

import witwin as wt

import drjit as dr
import numpy as np

from tests._scene_helpers import box_geometry, build_scene as build_test_scene
from witwin.channel import (
    Field,
    Material,
    FieldMonitor,
    Scene,
    Structure,
    Tracer,
    compute_diffraction_field,
    native_extension_available,
)
from witwin.channel.utils.polarization import scalarize_xy_jones
from witwin.channel.trace import compute_reflection_field
FREQUENCY = 1e9
WAVELENGTH = 299792458.0 / FREQUENCY
WAVENUMBER = 2.0 * dr.pi / WAVELENGTH
TX_POLARIZATION = (1.0, 0.0, 0.0)


def build_scene():
    cube1 = box_geometry(center=(-2.0, -2.0, 1.5), size=2.0)
    cube2 = box_geometry(center=(2.0, 1.5, 1.5), size=2.0)
    return build_test_scene(cube1, cube2)


def build_scene_with_materials(left_material, right_material):
    cube1 = box_geometry(center=(-2.0, -2.0, 1.5), size=2.0)
    cube2 = box_geometry(center=(2.0, 1.5, 1.5), size=2.0)
    return Scene(
        structures=[
            Structure(geometry=cube1, material=left_material, name="left"),
            Structure(geometry=cube2, material=right_material, name="right"),
        ],
        device="cuda",
    )


def field_power(field):
    return float(dr.sum(field.real * field.real + field.imag * field.imag)[0])


def field_delta_power(lhs, rhs):
    delta = wt.Complex2f(lhs.real - rhs.real, lhs.imag - rhs.imag)
    return field_power(delta)


def expected_execution_validation():
    if native_extension_available():
        return "native_strict", "native CUDA"
    return "validated_drjit", "strict Dr.Jit"


def build_tracer(
    scene,
    reflection_relative_permittivity,
    enable_rd=True,
    reflection_material=None,
    diffraction_material=None,
    use_scene_materials_for_reflection=True,
    use_scene_materials_for_diffraction=True,
):
    return Tracer(
        frequency=FREQUENCY,
        scene=scene,
        reflection_n_rays=256,
        reflection_max_bounces=1,
        reflection_coef=1.0,
        reflection_relative_permittivity=reflection_relative_permittivity,
        reflection_conductivity=0.0,
        reflection_material=reflection_material,
        diffraction_material=diffraction_material,
        use_scene_materials_for_reflection=use_scene_materials_for_reflection,
        use_scene_materials_for_diffraction=use_scene_materials_for_diffraction,
        enable_rd_diffraction=enable_rd,
        max_diffractions=2,
    )


def trace_kwargs():
    return {
        "monitor": FieldMonitor(
            "material_plane",
            axis="z",
            position=1.5,
            bounds=((-4.0, 4.0), (-4.0, 4.0)),
            grid_size=20,
        ),
        "verbose": False,
    }


def test_matched_material_suppresses_reflection_and_rd():
    scene = build_scene()
    tracer_reflective = build_tracer(
        scene,
        reflection_relative_permittivity=5.0,
        enable_rd=True,
        reflection_material={
            "relative_permittivity": 5.0,
            "conductivity": 0.0,
            "gain": 1.0,
        },
    )
    tracer_reflective_off = build_tracer(
        scene,
        reflection_relative_permittivity=5.0,
        enable_rd=False,
        reflection_material={
            "relative_permittivity": 5.0,
            "conductivity": 0.0,
            "gain": 1.0,
        },
    )
    tracer_matched = build_tracer(
        scene,
        reflection_relative_permittivity=1.0,
        enable_rd=True,
        reflection_material={
            "relative_permittivity": 1.0,
            "conductivity": 0.0,
            "gain": 1.0,
        },
    )
    tracer_matched_off = build_tracer(
        scene,
        reflection_relative_permittivity=1.0,
        enable_rd=False,
        reflection_material={
            "relative_permittivity": 1.0,
            "conductivity": 0.0,
            "gain": 1.0,
        },
    )

    tx = wt.Point3f(0.0, -4.0, 1.5)
    kwargs = trace_kwargs()

    res_reflective = tracer_reflective.trace(tx, **kwargs)
    res_reflective_off = tracer_reflective_off.trace(tx, **kwargs)
    res_matched = tracer_matched.trace(tx, **kwargs)
    res_matched_off = tracer_matched_off.trace(tx, **kwargs)

    ref_power = field_power(res_reflective.primary.field.reflection)
    matched_power = field_power(res_matched.primary.field.reflection)
    assert ref_power > 1e-12
    assert matched_power < 1e-16, f"Expected near-zero matched-medium reflection, got {matched_power:.6e}"

    rd_reflective_delta = wt.Complex2f(
        res_reflective.primary.field.diffraction.real - res_reflective_off.primary.field.diffraction.real,
        res_reflective.primary.field.diffraction.imag - res_reflective_off.primary.field.diffraction.imag,
    )
    rd_matched_delta = wt.Complex2f(
        res_matched.primary.field.diffraction.real - res_matched_off.primary.field.diffraction.real,
        res_matched.primary.field.diffraction.imag - res_matched_off.primary.field.diffraction.imag,
    )
    rd_ref_power = field_power(rd_reflective_delta)
    rd_matched_power = field_power(rd_matched_delta)
    assert rd_ref_power > 1e-12
    assert rd_matched_power < 1e-16, (
        f"Expected near-zero matched-medium RD increment, got {rd_matched_power:.6e}"
    )


def test_tracer_default_reflection_uses_default_material_reference_mode():
    scene = build_scene()
    tracer = Tracer(
        frequency=FREQUENCY,
        scene=scene,
        reflection_n_rays=256,
        reflection_max_bounces=1,
        reflection_coef=1.0,
        reflection_relative_permittivity=5.0,
        reflection_conductivity=0.0,
        max_diffractions=1,
    )

    tx = wt.Point3f(0.0, -4.0, 1.5)
    kwargs = trace_kwargs()
    result = tracer.trace(tx, **kwargs)

    monitor = kwargs["monitor"]
    field = monitor.to_field(WAVELENGTH, default_resolution=tracer.resolution_wavelength)
    coords = field.get_coordinates()
    a_ref_scalar, _, reflection_detail = compute_reflection_field(
        grid=field,
        rx_z=monitor.position,
        tx_pos=tx,
        scene=scene,
        wavelength=WAVELENGTH,
        k=WAVENUMBER,
        n_rays=256,
        max_reflections=1,
        reflection_coef=1.0,
        reflection_material=None,
        grid_data=coords,
        tx_polarization=TX_POLARIZATION,
    )
    delta = wt.Complex2f(
        result.primary.field.reflection.real - a_ref_scalar.real,
        result.primary.field.reflection.imag - a_ref_scalar.imag,
    )

    assert reflection_detail["reflection_model"] == "materialized"
    assert field_power(delta) < 1e-3
    assert result.primary.metadata["polarization_transport"]["reflection_transport"].startswith(
        "TE/TM Fresnel transport"
    )


def test_tracer_default_direct_diffraction_uses_default_material_reference_mode():
    scene = build_scene()
    tracer = Tracer(
        frequency=FREQUENCY,
        scene=scene,
        reflection_n_rays=256,
        reflection_max_bounces=1,
        reflection_coef=1.0,
        reflection_relative_permittivity=5.0,
        reflection_conductivity=0.0,
        max_diffractions=1,
    )

    tx = wt.Point3f(0.0, -4.0, 1.5)
    kwargs = trace_kwargs()
    result = tracer.trace(tx, **kwargs)

    monitor = kwargs["monitor"]
    field = monitor.to_field(WAVELENGTH, default_resolution=tracer.resolution_wavelength)
    coords = field.get_coordinates()
    _, _, _, components = compute_diffraction_field(
        coords["X"],
        coords["Y"],
        monitor.position,
        tx,
        scene,
        WAVELENGTH,
        WAVENUMBER,
        reflection_detail=None,
        max_diffractions=1,
        reflection_n_rays=0,
        reflection_max_bounces=0,
        reflection_mode="2d",
        grid=field,
        grid_data=coords,
        return_components=True,
        return_per_edge=False,
        diffraction_material=None,
        tx_polarization=TX_POLARIZATION,
    )

    delta = wt.Complex2f(
        result.primary.field.diffraction_direct.real - components["a_direct"].real,
        result.primary.field.diffraction_direct.imag - components["a_direct"].imag,
    )
    assert components["solver_metadata"]["diffraction_face_material_source"] == "default"
    assert result.primary.metadata["material_sources"]["diffraction"] in {"default", "scene"}
    assert field_power(delta) < 1e-4


def test_diffraction_material_changes_direct_diffraction_response():
    scene = build_scene()
    tracer_reflective = build_tracer(
        scene,
        reflection_relative_permittivity=5.0,
        enable_rd=False,
        diffraction_material={
            "relative_permittivity": 5.0,
            "conductivity": 0.0,
            "gain": 1.0,
        },
    )
    tracer_matched = build_tracer(
        scene,
        reflection_relative_permittivity=1.0,
        enable_rd=False,
        diffraction_material={
            "relative_permittivity": 1.0,
            "conductivity": 0.0,
            "gain": 1.0,
        },
    )

    tx = wt.Point3f(0.0, -4.0, 1.5)
    kwargs = trace_kwargs()

    res_reflective = tracer_reflective.trace(tx, **kwargs)
    res_matched = tracer_matched.trace(tx, **kwargs)

    reflective_power = field_power(res_reflective.primary.field.diffraction)
    matched_power = field_power(res_matched.primary.field.diffraction)
    delta = wt.Complex2f(
        res_reflective.primary.field.diffraction.real - res_matched.primary.field.diffraction.real,
        res_reflective.primary.field.diffraction.imag - res_matched.primary.field.diffraction.imag,
    )
    delta_power = field_power(delta)

    assert abs(reflective_power - matched_power) > 1e-6
    assert delta_power > 1e-6, (
        f"Expected material-aware diffraction change, got delta power {delta_power:.6e}"
    )


def test_runtime_triangle_material_table_tracks_structure_materials():
    scene = build_scene_with_materials(
        Material(eps_r=2.5, sigma_e=0.05),
        Material(eps_r=7.0, sigma_e=0.2),
    )
    tri_data = scene.tri_data_gpu

    structure_idx = np.asarray(tri_data["material_structure_idx"], dtype=np.int32)
    eps_r = np.asarray(tri_data["material_eps_r"], dtype=np.float32)
    sigma_e = np.asarray(tri_data["material_sigma_e"], dtype=np.float32)
    specified = np.asarray(tri_data["material_specified"], dtype=bool)

    assert bool(tri_data["material_has_specified_materials"])
    assert int(tri_data["material_n_specified_triangles"]) == int(specified.sum())
    assert int(tri_data["material_n_default_material_triangles"]) == int((~specified).sum())
    assert set(structure_idx[specified].tolist()) == {0, 1}
    assert set(float(value) for value in eps_r[structure_idx == 0]) == {2.5}
    assert set(float(value) for value in eps_r[structure_idx == 1]) == {7.0}
    assert np.allclose(sigma_e[structure_idx == 0], 0.05)
    assert np.allclose(sigma_e[structure_idx == 1], 0.2)


def test_scene_material_values_change_reflection_response():
    default_scene = build_scene()
    scene_with_materials = build_scene_with_materials(Material(eps_r=2.0), Material(eps_r=8.0))

    tx = wt.Point3f(0.0, -4.0, 1.5)
    kwargs = trace_kwargs()

    default_result = build_tracer(default_scene, reflection_relative_permittivity=5.0, enable_rd=False).trace(tx, **kwargs)
    scene_default = build_tracer(
        scene_with_materials,
        reflection_relative_permittivity=5.0,
        enable_rd=False,
    ).trace(tx, **kwargs)

    assert scene_default.primary.metadata["material_sources"]["reflection"] == "scene"
    assert scene_default.primary.metadata["reflection_model_source"] == "scene"
    assert default_result.primary.metadata["material_sources"]["reflection"] == "scene"
    assert default_result.primary.metadata["reflection_model_source"] == "scene"
    assert field_delta_power(scene_default.primary.field.reflection, default_result.primary.field.reflection) > 1e-6


def test_scene_material_opt_in_drives_reflection_and_override_wins():
    scene_low = build_scene_with_materials(Material(eps_r=2.0), Material(eps_r=2.0))
    scene_high = build_scene_with_materials(Material(eps_r=8.0), Material(eps_r=8.0))
    scene_mixed = build_scene_with_materials(Material(eps_r=2.0), Material(eps_r=8.0))

    tx = wt.Point3f(0.0, -4.0, 1.5)
    kwargs = trace_kwargs()

    low = build_tracer(
        scene_low,
        reflection_relative_permittivity=5.0,
        enable_rd=False,
        use_scene_materials_for_reflection=True,
    ).trace(tx, **kwargs)
    high = build_tracer(
        scene_high,
        reflection_relative_permittivity=5.0,
        enable_rd=False,
        use_scene_materials_for_reflection=True,
    ).trace(tx, **kwargs)
    mixed = build_tracer(
        scene_mixed,
        reflection_relative_permittivity=5.0,
        enable_rd=False,
        use_scene_materials_for_reflection=True,
    ).trace(tx, **kwargs)
    mixed_override = build_tracer(
        scene_mixed,
        reflection_relative_permittivity=5.0,
        enable_rd=False,
        use_scene_materials_for_reflection=True,
        reflection_material={
            "relative_permittivity": 8.0,
            "conductivity": 0.0,
            "gain": 1.0,
        },
    ).trace(tx, **kwargs)

    assert mixed.primary.metadata["material_sources"]["reflection"] == "scene"
    assert mixed.primary.metadata["reflection_model_source"] == "scene"
    assert field_delta_power(mixed.primary.field.reflection, low.primary.field.reflection) > 1e-8
    assert field_delta_power(mixed.primary.field.reflection, high.primary.field.reflection) > 1e-8
    assert field_delta_power(
        mixed_override.primary.field.reflection,
        high.primary.field.reflection,
    ) < 1e-12


def test_scene_material_values_change_diffraction_response():
    default_scene = build_scene()
    scene_with_materials = build_scene_with_materials(Material(eps_r=2.0), Material(eps_r=8.0))

    tx = wt.Point3f(0.0, -4.0, 1.5)
    kwargs = trace_kwargs()

    default_result = build_tracer(default_scene, reflection_relative_permittivity=5.0, enable_rd=False).trace(tx, **kwargs)
    scene_default = build_tracer(
        scene_with_materials,
        reflection_relative_permittivity=5.0,
        enable_rd=False,
    ).trace(tx, **kwargs)

    assert scene_default.primary.metadata["material_sources"]["diffraction"] == "scene"
    assert scene_default.primary.metadata["diffraction_face_material_source"] == "scene"
    assert default_result.primary.metadata["material_sources"]["diffraction"] == "scene"
    assert field_delta_power(
        scene_default.primary.field.diffraction_direct,
        default_result.primary.field.diffraction_direct,
    ) > 1e-6


def test_scene_material_opt_in_drives_diffraction_and_override_wins():
    scene_low = build_scene_with_materials(Material(eps_r=2.0), Material(eps_r=2.0))
    scene_high = build_scene_with_materials(Material(eps_r=8.0), Material(eps_r=8.0))
    scene_mixed = build_scene_with_materials(Material(eps_r=2.0), Material(eps_r=8.0))

    tx = wt.Point3f(0.0, -4.0, 1.5)
    kwargs = trace_kwargs()

    low = build_tracer(
        scene_low,
        reflection_relative_permittivity=5.0,
        enable_rd=False,
        use_scene_materials_for_diffraction=True,
    ).trace(tx, **kwargs)
    high = build_tracer(
        scene_high,
        reflection_relative_permittivity=5.0,
        enable_rd=False,
        use_scene_materials_for_diffraction=True,
    ).trace(tx, **kwargs)
    mixed = build_tracer(
        scene_mixed,
        reflection_relative_permittivity=5.0,
        enable_rd=False,
        use_scene_materials_for_diffraction=True,
    ).trace(tx, **kwargs)
    mixed_override = build_tracer(
        scene_mixed,
        reflection_relative_permittivity=5.0,
        enable_rd=False,
        use_scene_materials_for_diffraction=True,
        diffraction_material={
            "relative_permittivity": 8.0,
            "conductivity": 0.0,
            "gain": 1.0,
        },
    ).trace(tx, **kwargs)

    assert mixed.primary.metadata["material_sources"]["diffraction"] == "scene"
    assert mixed.primary.metadata["diffraction_face_material_source"] == "scene"
    assert field_delta_power(mixed.primary.field.diffraction_direct, low.primary.field.diffraction_direct) > 1e-8
    assert field_delta_power(mixed.primary.field.diffraction_direct, high.primary.field.diffraction_direct) > 1e-8
    assert field_delta_power(
        mixed_override.primary.field.diffraction_direct,
        high.primary.field.diffraction_direct,
    ) < 1e-12


def test_material_produces_finite_nonzero_reflection_gradient():
    eps_r = wt.Float(4.0)
    dr.enable_grad(eps_r)
    scene = build_scene_with_materials(
        Material(eps_r=eps_r, sigma_e=0.0),
        Material(eps_r=2.0),
    )
    tracer = build_tracer(
        scene,
        reflection_relative_permittivity=5.0,
        enable_rd=False,
        use_scene_materials_for_reflection=True,
    )

    result = tracer.trace(wt.Point3f(0.0, -4.0, 1.5), **trace_kwargs())
    objective = dr.sum(
        result.primary.field.reflection.real * result.primary.field.reflection.real
        + result.primary.field.reflection.imag * result.primary.field.reflection.imag
    )
    dr.backward(objective)

    grad = float(dr.grad(eps_r)[0])
    assert math.isfinite(grad)
    assert abs(grad) > 1e-12


def test_material_aware_reflection_field_matches_projected_jones():
    scene = build_scene()
    tracer = Tracer(
        frequency=FREQUENCY,
        scene=scene,
        reflection_n_rays=256,
        reflection_max_bounces=1,
        reflection_coef=1.0,
        reflection_material={
            "relative_permittivity": 5.0,
            "conductivity": 0.0,
            "gain": 1.0,
        },
        max_diffractions=1,
        tx_polarization=(1.0, 1.0, 0.0),
        rx_polarization=(0.0, 1.0, 0.0),
    )

    result = tracer.trace(wt.Point3f(0.0, -4.0, 1.5), **trace_kwargs())
    projected = scalarize_xy_jones(result.primary.jones.reflection, (0.0, 1.0, 0.0))
    delta = wt.Complex2f(
        projected.real - result.primary.field.reflection.real,
        projected.imag - result.primary.field.reflection.imag,
    )

    assert field_power(result.primary.field.reflection) > 1e-10
    assert field_power(delta) < 1e-12
    expected_tier, expected_note_fragment = expected_execution_validation()
    assert result.primary.metadata["execution_validation_tier"] == expected_tier
    assert expected_note_fragment in result.primary.metadata["execution_validation_note"]


def test_material_aware_diffraction_field_matches_projected_jones():
    scene = build_scene()
    tracer = Tracer(
        frequency=FREQUENCY,
        scene=scene,
        reflection_n_rays=0,
        reflection_max_bounces=0,
        reflection_coef=1.0,
        diffraction_material={
            "relative_permittivity": 5.0,
            "conductivity": 0.0,
            "gain": 1.0,
        },
        max_diffractions=1,
        tx_polarization=(1.0, 1.0, 0.0),
        rx_polarization=(0.0, 1.0, 0.0),
    )

    result = tracer.trace(wt.Point3f(0.0, -4.0, 1.5), **trace_kwargs())
    projected = scalarize_xy_jones(result.primary.jones.diffraction, (0.0, 1.0, 0.0))
    delta = wt.Complex2f(
        projected.real - result.primary.field.diffraction.real,
        projected.imag - result.primary.field.diffraction.imag,
    )

    assert field_power(result.primary.field.diffraction) > 1e-10
    assert field_power(delta) < 1e-12
    expected_tier, expected_note_fragment = expected_execution_validation()
    assert result.primary.metadata["execution_validation_tier"] == expected_tier
    assert expected_note_fragment in result.primary.metadata["execution_validation_note"]


def test_material_aware_default_scalar_fields_use_implicit_tx_copolar_projection():
    tx_polarization = (1.0, 1.0, 0.0)
    scene = build_scene_with_materials(Material(eps_r=5.0), Material(eps_r=5.0))
    default_tracer = Tracer(
        frequency=FREQUENCY,
        scene=scene,
        reflection_n_rays=256,
        reflection_max_bounces=1,
        reflection_coef=1.0,
        use_scene_materials_for_reflection=True,
        use_scene_materials_for_diffraction=True,
        max_diffractions=2,
        tx_polarization=tx_polarization,
    )
    explicit_tracer = Tracer(
        frequency=FREQUENCY,
        scene=scene,
        reflection_n_rays=256,
        reflection_max_bounces=1,
        reflection_coef=1.0,
        use_scene_materials_for_reflection=True,
        use_scene_materials_for_diffraction=True,
        max_diffractions=2,
        tx_polarization=tx_polarization,
        rx_polarization=tx_polarization,
    )

    tx = wt.Point3f(0.0, -4.0, 1.5)
    default_result = default_tracer.trace(tx, **trace_kwargs())
    explicit_result = explicit_tracer.trace(tx, **trace_kwargs())
    metadata = default_result.primary.metadata
    polarization_transport = metadata["polarization_transport"]

    for name in ("los", "reflection", "diffraction", "total"):
        default_field = getattr(default_result.primary.field, name)
        explicit_field = getattr(explicit_result.primary.field, name)
        projected = scalarize_xy_jones(getattr(default_result.primary.jones, name), tx_polarization)
        explicit_delta = wt.Complex2f(
            default_field.real - explicit_field.real,
            default_field.imag - explicit_field.imag,
        )
        projected_delta = wt.Complex2f(
            default_field.real - projected.real,
            default_field.imag - projected.imag,
        )
        assert field_power(explicit_delta) < 1e-12
        assert field_power(projected_delta) < 1e-12

    assert polarization_transport["rx_polarization"] == tx_polarization
    assert polarization_transport["rx_polarization_source"] == "default_from_tx_polarization"
    assert polarization_transport["reflection_scalarization"] == "default_receiver_projection_from_jones"
    assert (
        polarization_transport["diffraction_face_scalarization"]
        == "default_receiver_projection_from_jones_face_operator"
    )
    assert metadata["scalar_projection_rule"] == "global_xy_default_receiver_projection_from_jones"


def test_material_aware_default_reflection_does_not_collapse_at_high_permittivity():
    tx_polarization = (1.0, 1.0, 0.0)
    kwargs = trace_kwargs()
    tx = wt.Point3f(0.0, -4.0, 1.5)

    low_scene = build_scene_with_materials(Material(eps_r=5.0), Material(eps_r=5.0))
    high_scene = build_scene_with_materials(Material(eps_r=1.0e4), Material(eps_r=1.0e4))

    low = Tracer(
        frequency=FREQUENCY,
        scene=low_scene,
        reflection_n_rays=256,
        reflection_max_bounces=1,
        reflection_coef=1.0,
        use_scene_materials_for_reflection=True,
        max_diffractions=1,
        tx_polarization=tx_polarization,
    ).trace(tx, **kwargs)
    high = Tracer(
        frequency=FREQUENCY,
        scene=high_scene,
        reflection_n_rays=256,
        reflection_max_bounces=1,
        reflection_coef=1.0,
        use_scene_materials_for_reflection=True,
        max_diffractions=1,
        tx_polarization=tx_polarization,
    ).trace(tx, **kwargs)

    low_power = field_power(low.primary.field.reflection)
    high_power = field_power(high.primary.field.reflection)

    assert low.primary.metadata["polarization_transport"]["reflection_scalarization"] == (
        "default_receiver_projection_from_jones"
    )
    assert high.primary.metadata["polarization_transport"]["reflection_scalarization"] == (
        "default_receiver_projection_from_jones"
    )
    assert low_power > 1e-8
    assert high_power > low_power * 2.0


