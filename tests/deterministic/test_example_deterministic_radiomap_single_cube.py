"""Smoke test for the single-cube deterministic radiomap example."""

from __future__ import annotations

import numpy as np
import pytest

import witwin.channel as wt
import examples.deterministic_radiomap_single_cube as single_cube
from examples.deterministic_radiomap_single_cube import SingleCubeExperiment
from witwin.channel.core.runtime import TraceContext
from witwin.channel.deterministic.config import resolve_trace_config
from witwin.channel.deterministic.trace import reflection


def test_deterministic_single_cube_forward_components_are_finite():
    experiment = SingleCubeExperiment(
        grid_shape=(16, 16),
        forward_num_samples=64,
        max_bounces=2,
        max_diffraction_order=1,
        shadow_boundary_correction=False,
    )

    snapshot = experiment.forward()

    assert snapshot.path_gain.shape == (16, 16)
    assert np.isfinite(snapshot.path_gain).all()
    assert float(snapshot.path_gain.max()) > 0.0
    for name in ("los", "reflection", "diffraction"):
        component = snapshot.components[name]
        assert component.shape == snapshot.path_gain.shape
        assert np.isfinite(component).all()

    assert experiment.forward_config.tuning.enable_rd_diffraction is True
    assert experiment.forward_config.max_diffraction_order == 1


def test_deterministic_single_cube_y_line_profiles_report_component_mix():
    experiment = SingleCubeExperiment(
        grid_shape=(16, 16),
        forward_num_samples=64,
        max_bounces=2,
        max_diffraction_order=1,
        shadow_boundary_correction=False,
    )
    snapshot = experiment.forward()

    profiles = single_cube.line_power_profiles(snapshot, y_values=(-4.0, 4.0))
    summary = single_cube.summarize_line_power_profiles(profiles)

    assert set(profiles) == {-4.0, 4.0}
    assert set(summary) == {"y=-4", "y=4"}
    for profile in profiles.values():
        assert profile["x"].shape == (16,)
        assert profile["path_gain"].shape == (16,)
        assert np.isfinite(profile["path_gain"]).all()
        component_sum = np.asarray(profile["component_sum"], dtype=np.float64)
        assert float(component_sum.max()) > 0.0
        fraction_sum = np.zeros_like(component_sum)
        for name in ("los", "reflection", "diffraction"):
            component = profile["components"][name]
            fraction = profile["component_fraction"][name]
            assert component.shape == (16,)
            assert np.isfinite(component).all()
            assert np.isfinite(fraction).all()
            assert np.all(fraction >= 0.0)
            fraction_sum = fraction_sum + fraction
        np.testing.assert_allclose(
            fraction_sum,
            np.where(component_sum > 0.0, 1.0, 0.0),
            rtol=1.0e-6,
            atol=1.0e-6,
        )


@pytest.mark.gpu
def test_deterministic_single_cube_reflection_discovery_includes_left_face():
    experiment = SingleCubeExperiment(
        grid_shape=(16, 16),
        forward_num_samples=192,
        max_bounces=2,
        max_diffraction_order=1,
        shadow_boundary_correction=False,
    )
    resolved = resolve_trace_config(
        frequency=single_cube.DEFAULT_FREQUENCY_HZ,
        config=experiment.forward_config,
    )
    runtime = TraceContext.from_config(
        tx_pos=wt.Point3f(*experiment.tx_pos),
        config=resolved,
    )

    detail = reflection.discover_paths(
        tx=runtime.tx,
        scene=experiment.scene,
        wave=runtime.wave,
        n_rays=192,
        max_reflections=2,
        mode="3d",
        material=runtime.reflection,
        ray_sampling="full_sphere",
        sampling_axis="z",
        sampling_plane_position=experiment.plane_z,
        sampling_bounds=experiment.bounds,
    )

    first_bounce = detail.source_paths_per_bounce[0]
    images = np.column_stack(
        [
            np.asarray(first_bounce.image_source.x, dtype=np.float64),
            np.asarray(first_bounce.image_source.y, dtype=np.float64),
            np.asarray(first_bounce.image_source.z, dtype=np.float64),
        ]
    )
    normals = np.column_stack(
        [
            np.asarray(first_bounce.path_plane_normal[0].x, dtype=np.float64),
            np.asarray(first_bounce.path_plane_normal[0].y, dtype=np.float64),
            np.asarray(first_bounce.path_plane_normal[0].z, dtype=np.float64),
        ]
    )

    assert int(first_bounce.n_paths) == 6
    assert np.any(np.all(np.isclose(images, [0.0, -5.0, 4.0]), axis=1))
    assert np.any(np.all(np.isclose(normals, [-1.0, 0.0, 0.0]), axis=1))


@pytest.mark.gpu
def test_deterministic_single_cube_left_face_reflection_projection_is_nonempty():
    experiment = SingleCubeExperiment(
        bounds=((-1.82, -1.78), (3.0, 3.2)),
        grid_shape=(2, 2),
        forward_num_samples=192,
        max_bounces=2,
        max_diffraction_order=1,
        shadow_boundary_correction=False,
    )

    snapshot = experiment.forward()

    reflection_power = np.asarray(snapshot.components["reflection"], dtype=np.float64)
    assert np.all(reflection_power > 0.0)


@pytest.mark.gpu
@pytest.mark.parametrize("boundary_x", [4.0, -0.5])
def test_deterministic_single_cube_raw_utd_cancels_los_shadow_boundary(boundary_x):
    half_span = 0.004
    experiment = SingleCubeExperiment(
        bounds=((boundary_x - half_span, boundary_x + half_span), (4.0 - 0.001, 4.0 + 0.001)),
        grid_shape=(2, 1),
        forward_num_samples=64,
        max_bounces=0,
        max_diffraction_order=1,
        shadow_boundary_correction=False,
    )

    result = experiment.solve()
    coherent = result.field.vector_coherent

    def component_vectors(name: str) -> np.ndarray:
        return np.stack(
            [
                np.asarray(coherent[name][axis], dtype=np.complex64)
                for axis in ("x", "y", "z")
            ],
            axis=1,
        )

    los_vectors = component_vectors("los")
    total_vectors = component_vectors("total")
    los_step = float(np.linalg.norm(los_vectors[1] - los_vectors[0]))
    total_step = float(np.linalg.norm(total_vectors[1] - total_vectors[0]))

    assert los_step > 1.0e-3
    assert total_step < 0.25 * los_step


@pytest.mark.gpu
@pytest.mark.parametrize("boundary_x", [-0.25, 3.25])
def test_deterministic_single_cube_raw_utd_cancels_reflection_shadow_boundary(boundary_x):
    half_span = 0.004
    experiment = SingleCubeExperiment(
        bounds=((boundary_x - half_span, boundary_x + half_span), (-4.0 - 0.001, -4.0 + 0.001)),
        grid_shape=(2, 1),
        forward_num_samples=96,
        max_bounces=2,
        max_diffraction_order=1,
        shadow_boundary_correction=False,
    )

    result = experiment.solve()
    coherent = result.field.vector_coherent

    def component_vectors(name: str) -> np.ndarray:
        return np.stack(
            [
                np.asarray(coherent[name][axis], dtype=np.complex64)
                for axis in ("x", "y", "z")
            ],
            axis=1,
        )

    reflection_vectors = component_vectors("reflection")
    total_vectors = component_vectors("total")
    reflection_step = float(np.linalg.norm(reflection_vectors[1] - reflection_vectors[0]))
    total_step = float(np.linalg.norm(total_vectors[1] - total_vectors[0]))

    assert reflection_step > 1.0e-3
    assert total_step < 0.25 * reflection_step


@pytest.mark.gpu
def test_deterministic_single_cube_raw_utd_finite_endpoint_is_not_hard_cutoff():
    experiment = SingleCubeExperiment(
        bounds=((-0.09, 0.09), (7.035, 7.045)),
        grid_shape=(2, 1),
        forward_num_samples=64,
        max_bounces=0,
        max_diffraction_order=1,
        shadow_boundary_correction=False,
    )

    result = experiment.solve()
    coherent = result.field.vector_coherent
    diffraction_vectors = np.stack(
        [
            np.asarray(coherent["diffraction"][axis], dtype=np.complex64)
            for axis in ("x", "y", "z")
        ],
        axis=1,
    )
    diffraction_magnitudes = np.linalg.norm(diffraction_vectors, axis=1)
    max_magnitude = float(np.max(diffraction_magnitudes))
    min_magnitude = float(np.min(diffraction_magnitudes))
    diffraction_step = float(
        np.linalg.norm(diffraction_vectors[1] - diffraction_vectors[0])
    )

    assert max_magnitude > 1.0e-4
    assert min_magnitude > 1.0e-5
    assert diffraction_step < 0.99 * max_magnitude


@pytest.mark.gpu
def test_deterministic_single_cube_raw_utd_shadow_wedge_projection_is_continuous():
    experiment = SingleCubeExperiment(
        bounds=((-0.01, 0.01), (2.996, 3.004)),
        grid_shape=(1, 2),
        forward_num_samples=64,
        max_bounces=0,
        max_diffraction_order=1,
        shadow_boundary_correction=False,
    )

    result = experiment.solve()
    coherent = result.field.vector_coherent
    diffraction_vectors = np.stack(
        [
            np.asarray(coherent["diffraction"][axis], dtype=np.complex64)
            for axis in ("x", "y", "z")
        ],
        axis=1,
    )
    diffraction_magnitudes = np.linalg.norm(diffraction_vectors, axis=1)
    max_magnitude = float(np.max(diffraction_magnitudes))
    diffraction_step = float(
        np.linalg.norm(diffraction_vectors[1] - diffraction_vectors[0])
    )

    assert max_magnitude > 1.0e-4
    assert diffraction_step < 0.5 * max_magnitude


@pytest.mark.gpu
def test_deterministic_single_cube_raw_utd_top_edge_unpaired_isb_is_continuous():
    experiment = SingleCubeExperiment(
        bounds=((-0.004, 0.004), (3.96825, 3.96925)),
        grid_shape=(2, 1),
        forward_num_samples=64,
        max_bounces=0,
        max_diffraction_order=1,
        shadow_boundary_correction=False,
    )

    result = experiment.solve()
    coherent = result.field.vector_coherent
    diffraction_vectors = np.stack(
        [
            np.asarray(coherent["diffraction"][axis], dtype=np.complex64)
            for axis in ("x", "y", "z")
        ],
        axis=1,
    )
    diffraction_magnitudes = np.linalg.norm(diffraction_vectors, axis=1)
    max_magnitude = float(np.max(diffraction_magnitudes))
    diffraction_step = float(
        np.linalg.norm(diffraction_vectors[1] - diffraction_vectors[0])
    )

    assert max_magnitude > 1.0e-4
    assert diffraction_step < 0.1 * max_magnitude

