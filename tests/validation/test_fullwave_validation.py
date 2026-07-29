# Copyright Xingyu Chen.
# Tests fullwave validation.

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from benchmarks.fullwave_validation.backends import (
    build_tidy3d_simulation,
    extract_tidy3d_field_map,
)
from benchmarks.fullwave_validation.metrics import (
    analyze_boundaries,
    compare_fields,
    compare_magnitudes,
    comparison_report,
    resample_regular,
)
from benchmarks.fullwave_validation.models import FieldMap
from benchmarks.fullwave_validation.scenarios import (
    MANIFEST_PATH,
    build_channel_scene,
    load_case,
    load_manifest,
    observation_valid_mask,
)
from witwin.core import PhysicalMaterial, ReceiverGrid


def _map(field, *, components=None, backend="test") -> FieldMap:
    values = np.asarray(field, dtype=np.complex128)
    return FieldMap(
        x=np.arange(values.shape[1], dtype=np.float64),
        y=np.arange(values.shape[0], dtype=np.float64),
        field=values,
        components=components or {},
        metadata={
            "backend": backend,
            "case_id": "synthetic",
            "case_fingerprint": "a" * 64,
            "frequency_hz": 1.0e9,
        },
    )


def test_manifest_declares_original_layout_provenance_and_six_cases():
    manifest = load_manifest()
    single_metal = load_case("single_cube", "metal")

    assert MANIFEST_PATH.is_file()
    assert manifest["schema"] == {
        "name": "witwin.channel.fullwave-validation-scenarios",
        "version": "1.0.0",
    }
    assert (
        manifest["electromagnetic_scaling"][
            "geometry_scale_from_original_channel_examples"
        ]
        == 0.1
    )
    cases = {
        load_case(scenario, material).case_id
        for scenario in ("single_cube", "three_cube", "three_cube_320")
        for material in ("metal", "dielectric")
    }
    assert cases == {
        "single_cube-metal",
        "single_cube-dielectric",
        "three_cube-metal",
        "three_cube-dielectric",
        "three_cube_320-metal",
        "three_cube_320-dielectric",
    }
    assert single_metal.tx_position == (-0.2, -0.5, 0.42)
    assert single_metal.receiver_shape == (256, 256)
    assert single_metal.fullwave_dl_m == pytest.approx(0.00625)


def test_three_cube_320_shares_physical_scene_with_three_cube():
    base = load_case("three_cube", "metal")
    aligned = load_case("three_cube_320", "metal")

    assert aligned.cube_centers == base.cube_centers
    assert aligned.tx_position == base.tx_position
    assert aligned.plane_z == base.plane_z
    assert aligned.cube_size_m == base.cube_size_m
    assert aligned.max_depth == base.max_depth
    assert aligned.analysis_bounds_xy == base.analysis_bounds_xy
    assert aligned.receiver_shape == (320, 320)
    assert aligned.fullwave_dl_m == base.fullwave_dl_m
    assert aligned.fingerprint != base.fingerprint


@pytest.mark.parametrize("scenario", ["single_cube", "three_cube_320"])
def test_receiver_cells_are_yee_phase_locked(scenario):
    """Receiver cell centers must coincide with Maxwell Ez Yee nodes.

 Ez nodes sit at integer x/y offsets and half-integer z offsets from the
 domain origin on a uniform Yee grid; jump metrics must never come from a
 half-cell-interpolated field.
 """
    spec = load_case(scenario, "metal")
    dl = spec.fullwave_dl_m

    for axis_values, (domain_lo, domain_hi) in zip(
        (spec.x, spec.y), spec.domain_bounds_xyz[:2], strict=True
    ):
        offsets = (axis_values - domain_lo) / dl
        assert np.max(np.abs(offsets - np.round(offsets))) < 1.0e-9
        span_cells = (domain_hi - domain_lo) / dl
        assert abs(span_cells - round(span_cells)) < 1.0e-9
        assert axis_values[1] - axis_values[0] == pytest.approx(dl)

    z_lo, z_hi = spec.domain_bounds_xyz[2]
    plane_offset = (spec.plane_z - z_lo) / dl
    assert abs(plane_offset - np.floor(plane_offset) - 0.5) < 1.0e-9
    z_span_cells = (z_hi - z_lo) / dl
    assert abs(z_span_cells - round(z_span_cells)) < 1.0e-9


def test_legacy_three_cube_256_is_not_yee_phase_locked():
    """The legacy 256-case pitch (7.8125 mm) is documented as NOT grid-
 coincident; full-wave jump statistics must use three_cube_320 instead."""
    spec = load_case("three_cube", "metal")
    assert spec.x[1] - spec.x[0] != pytest.approx(spec.fullwave_dl_m)


@pytest.mark.parametrize(
    "scenario,cube_count",
    [("single_cube", 1), ("three_cube", 3), ("three_cube_320", 3)],
)
@pytest.mark.parametrize("material", ["metal", "dielectric"])
def test_channel_scene_matches_case_geometry_and_material(
    scenario, cube_count, material
):
    spec = load_case(scenario, material)
    scene = build_channel_scene(spec)

    assert len(scene.structures) == cube_count
    assert scene.metadata["fullwave_validation_fingerprint"] == spec.fingerprint
    receivers = tuple(
        endpoint for endpoint in scene.endpoints if endpoint.role == "rx"
    )
    assert isinstance(receivers[0], ReceiverGrid)
    assert receivers[0].shape == (spec.x.size, spec.y.size)
    assert all(
        isinstance(structure.material, PhysicalMaterial)
        for structure in scene.structures
    )
    if material == "metal":
        assert all(
            structure.material.conductor_model == "perfect"
            for structure in scene.structures
        )
    else:
        assert all(structure.material.layers for structure in scene.structures)


def test_tidy3d_scene_uses_true_pec_and_dielectric_volumes():
    pytest.importorskip("tidy3d")
    metal = build_tidy3d_simulation(load_case("single_cube", "metal"))
    dielectric = build_tidy3d_simulation(load_case("single_cube", "dielectric"))

    assert type(metal.structures[0].medium).__name__ == "PECMedium"
    assert dielectric.structures[0].medium.permittivity == 4.0
    assert dielectric.structures[0].medium.conductivity == 1.0e-8
    assert metal.size == (2.0e6, 2.0e6, 9.0e5)
    assert metal.boundary_spec.x.minus.num_layers == metal.boundary_spec.x.plus.num_layers == 12
    assert metal.monitors[0].name == "field_xy"


def test_fullwave_analysis_source_and_geometry_are_outside_pml():
    for scenario in ("single_cube", "three_cube", "three_cube_320"):
        spec = load_case(scenario, "metal")
        pml = spec.fullwave_pml_layers * spec.fullwave_dl_m
        interior = tuple((lo + pml, hi - pml) for lo, hi in spec.domain_bounds_xyz)

        for axis, bounds in enumerate(spec.analysis_bounds_xy):
            assert bounds[0] >= interior[axis][0]
            assert bounds[1] <= interior[axis][1]
        assert all(lo < value < hi for value, (lo, hi) in zip(spec.tx_position, interior, strict=True))


def test_three_cube_pec_observation_mask_is_union_and_plane_aware():
    spec = load_case("three_cube", "metal")
    x = np.linspace(-0.5, 0.5, 81)
    y = np.linspace(-0.6, 0.6, 97)
    actual = observation_valid_mask(spec, x, y)

    expected = np.ones((y.size, x.size), dtype=bool)
    half_size = spec.cube_size_m / 2.0
    for center_x, center_y, _center_z in spec.cube_centers:
        expected &= ~(
            (np.abs(x[None, :] - center_x) <= half_size)
            & (np.abs(y[:, None] - center_y) <= half_size)
        )
    np.testing.assert_array_equal(actual, expected)

    top_surface = replace(spec, plane_z=spec.cube_centers[0][2] + half_size)
    np.testing.assert_array_equal(
        observation_valid_mask(top_surface, x, y),
        expected,
    )
    above_cubes = replace(spec, plane_z=spec.cube_centers[0][2] + half_size + 1.0e-4)
    assert observation_valid_mask(above_cubes, x, y).all()


def test_tidy3d_extraction_returns_meter_coordinates_and_yx_layout():
    xr = pytest.importorskip("xarray")
    spec = load_case("single_cube", "metal")
    ez = xr.DataArray(
        np.arange(6, dtype=np.float64).reshape(2, 3, 1, 1),
        dims=("x", "y", "z", "f"),
        coords={
            "x": [-2.0e5, 2.0e5],
            "y": [-3.0e5, 0.0, 3.0e5],
            "z": [spec.plane_z * 1.0e6],
            "f": [spec.frequency_hz],
        },
    )

    reference = extract_tidy3d_field_map(
        {"field_xy": type("Monitor", (), {"Ez": ez})()}, spec
    )

    np.testing.assert_allclose(reference.x, [-0.2, 0.2])
    np.testing.assert_allclose(reference.y, [-0.3, 0.0, 0.3])
    np.testing.assert_array_equal(reference.field, np.arange(6).reshape(2, 3).T)


def test_field_map_npz_round_trip(tmp_path):
    original = _map(
        np.arange(12).reshape(3, 4) * (1.0 + 2.0j),
        components={"los": np.ones((3, 4), dtype=np.complex128)},
    )
    path = original.save(tmp_path / "reference.npz")
    restored = FieldMap.load(path)

    np.testing.assert_array_equal(restored.field, original.field)
    np.testing.assert_array_equal(
        restored.components["los"], original.components["los"]
    )
    assert restored.metadata == original.metadata


def test_three_cube_deterministic_plot_draws_scene_and_writes_png(tmp_path):
    pytest.importorskip("matplotlib")
    from benchmarks.fullwave_validation.experiments.plot_three_cube_deterministic import (
        _add_scene_overlay,
        plot_three_cube_deterministic,
    )
    import matplotlib.pyplot as plt

    spec = load_case("three_cube", "metal")
    x = np.linspace(-1.0, 1.0, 21)
    y = np.linspace(-1.0, 1.0, 19)
    yy, xx = np.meshgrid(y, x, indexing="ij")
    base = (1.0 + 0.2 * xx - 0.1 * yy) * np.exp(1j * (0.3 * xx + 0.4 * yy))
    field_map = FieldMap(
        x=x,
        y=y,
        field=base,
        components={
            "los": 0.6 * base,
            "reflection": 0.3j * base,
            "diffraction": 0.1 * base,
            "coupled": 0.05j * base,
        },
        metadata={
            "backend": "deterministic",
            "case_id": spec.case_id,
            "case_fingerprint": spec.fingerprint,
            "frequency_hz": spec.frequency_hz,
        },
    )

    output = plot_three_cube_deterministic(field_map, tmp_path / "three-cube.png")

    assert output.is_file()
    assert output.stat().st_size > 0
    figure, axis = plt.subplots()
    _add_scene_overlay(axis, spec)
    assert len(axis.patches) == 3
    assert len(axis.collections) == 1
    plt.close(figure)


def test_complex_calibration_removes_source_amplitude_and_phase_offset():
    y, x = np.mgrid[0:5, 0:6]
    candidate = _map(np.exp(1j * (0.2 * x + 0.3 * y)))
    reference = _map((2.0j) * candidate.field, backend="tidy3d")

    metrics = compare_fields(candidate, reference)

    assert metrics.raw_nmse > 1.0
    assert metrics.complex_scale_real == pytest.approx(0.0, abs=1.0e-12)
    assert metrics.complex_scale_imag == pytest.approx(2.0, abs=1.0e-12)
    assert metrics.calibrated_nmse == pytest.approx(0.0, abs=1.0e-24)
    assert metrics.magnitude_correlation == pytest.approx(1.0)


def test_magnitude_calibration_is_not_suppressed_by_incoherent_phase():
    candidate = _map(np.ones((2, 4), dtype=np.complex128))
    reference = _map(
        4.0 * np.tile([1.0, -1.0, 1.0, -1.0], (2, 1)),
        backend="fullwave",
    )

    complex_metrics = compare_fields(candidate, reference)
    magnitude_metrics = compare_magnitudes(candidate, reference)

    assert complex_metrics.complex_scale_real == pytest.approx(0.0, abs=1.0e-12)
    assert magnitude_metrics.amplitude_scale == pytest.approx(4.0)
    assert magnitude_metrics.reference_to_candidate_energy_ratio == pytest.approx(16.0)
    assert magnitude_metrics.calibrated_reference_to_candidate_energy_ratio == pytest.approx(1.0)
    assert magnitude_metrics.calibrated_magnitude_nmse == pytest.approx(0.0)
    assert magnitude_metrics.complex_coherence == pytest.approx(0.0)

    fixed = compare_magnitudes(candidate, reference, amplitude_scale=2.0)
    assert fixed.amplitude_scale == pytest.approx(2.0)
    assert fixed.calibrated_reference_to_candidate_energy_ratio == pytest.approx(4.0)


def test_comparison_mask_excludes_invalid_conductor_interior():
    candidate = _map(np.ones((4, 4), dtype=np.complex128))
    reference_values = np.ones((4, 4), dtype=np.complex128)
    reference_values[1:3, 1:3] = 0.0
    reference = _map(reference_values, backend="fullwave")
    valid = np.ones((4, 4), dtype=bool)
    valid[1:3, 1:3] = False

    masked = compare_fields(candidate, reference, valid_mask=valid)

    assert masked.calibrated_nmse == pytest.approx(0.0)


def test_comparison_mask_rejects_wrong_shape():
    field = _map(np.ones((3, 4), dtype=np.complex128))

    with pytest.raises(ValueError, match="valid_mask must have shape"):
        compare_fields(field, field, valid_mask=np.ones((2, 2), dtype=bool))


def test_regular_grid_resampling_is_complex_and_coordinate_aware():
    source_x = np.linspace(0.0, 3.0, 4)
    source_y = np.linspace(0.0, 2.0, 3)
    y, x = np.meshgrid(source_y, source_x, indexing="ij")
    source = FieldMap(source_x, source_y, x + 2j * y)

    target = resample_regular(
        source, np.linspace(0.5, 2.5, 5), np.linspace(0.5, 1.5, 3)
    )
    yy, xx = np.meshgrid(target.y, target.x, indexing="ij")

    np.testing.assert_allclose(target.field, xx + 2j * yy)


def test_isb_and_rsb_report_deterministic_excess_jump():
    field = np.ones((6, 8), dtype=np.complex128)
    field[:, 4:] = 0.1
    smooth = np.tile(np.linspace(1.0, 0.7, 8), (6, 1)).astype(np.complex128)
    los = np.zeros_like(field)
    los[:, :4] = 1.0
    reflection = np.zeros_like(field)
    reflection[:, 4:] = 1.0
    deterministic = _map(field, components={"los": los, "reflection": reflection})
    fullwave = _map(smooth, backend="tidy3d")

    boundaries = analyze_boundaries(deterministic, fullwave)

    assert boundaries["ISB"].edge_count == 6
    assert boundaries["RSB"].edge_count == 6
    assert boundaries["ISB"].p95_excess_jump_db > 15.0
    assert boundaries["RSB"].p95_excess_jump_db > 15.0


def test_comparison_rejects_mismatched_case_fingerprints():
    first = _map(np.ones((3, 3)))
    second_metadata = dict(first.metadata)
    second_metadata["case_fingerprint"] = "b" * 64
    second = FieldMap(first.x, first.y, first.field, metadata=second_metadata)

    with pytest.raises(ValueError, match="case_fingerprint"):
        comparison_report(first, second)