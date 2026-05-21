"""Integration coverage for deterministic radiomap field solve support."""

from __future__ import annotations

import drjit as dr
import numpy as np
import pytest
import witwin.channel as wt

import witwin.channel.deterministic as drm
from examples.field_solver_multipath_main import ThreeCubeFieldExperiment, smoke_profile
from examples.field_solver_three_cubes_3d import smoke_profile as smoke_profile_3d
from witwin.channel.core.scene import EdgePolicy, Mesh as ChannelMesh
from witwin.channel.core.scene import Scene as ChannelScene
from witwin.channel.core.geometry.mesh_buffers import to_point3f, to_vector3u
from witwin.core import Box, Material, Structure
from witwin.channel.deterministic import Config, FieldSpec, Tuning, solve_field
from witwin.channel.deterministic.diffraction.state import Geo

pytestmark = pytest.mark.gpu

FREQUENCY = 1.0e9


def _far_scene() -> ChannelScene:
    return ChannelScene(
        structures=[
            Structure(
                name="far_wall",
                geometry=Box(
                    position=(20.0, 0.0, 1.5),
                    size=(0.25, 2.0, 2.0),
                    device="cuda",
                ),
                material=Material(eps_r=4.0, sigma_e=0.0),
            )
        ],
        device="cuda",
    )


def _cube_scene(*, size: float = 1.0, center=(0.0, 0.0, 1.5)) -> ChannelScene:
    return ChannelScene(
        structures=[
            Structure(
                name="cube",
                geometry=Box(position=center, size=(size, size, size), device="cuda"),
                material=Material(eps_r=4.0, sigma_e=0.0),
            )
        ],
        device="cuda",
    )


def _moving_cube_scene(cube1_x) -> ChannelScene:
    vertices, faces = Box(position=(0.0, 0.0, 0.0), size=(2.0, 2.0, 2.0), device="cuda").to_mesh()
    base_vertices = to_point3f(vertices)
    return ChannelScene(
        structures=[
            Structure(
                name="cube",
                geometry=ChannelMesh(
                    vertices=wt.Point3f(
                        base_vertices.x + cube1_x,
                        base_vertices.y,
                        base_vertices.z + 1.5,
                    ),
                    faces=to_vector3u(faces),
                ),
                material=Material(eps_r=4.0, sigma_e=0.0),
            )
        ],
        device="cuda",
    )


def _complex_abs_sum(value) -> float:
    return float(
        np.sum(np.abs(np.asarray(value.real, dtype=np.float64)))
        + np.sum(np.abs(np.asarray(value.imag, dtype=np.float64)))
    )


def test_package_root_exports_field_api_names():
    assert drm.__all__ == [
        "Config",
        "FieldResult",
        "FieldSpec",
        "NativeExtension",
        "RadioMapResult",
        "Tuning",
        "native_extension_available",
        "solve",
        "solve_field",
    ]
    assert hasattr(drm, "Config")
    assert hasattr(drm, "FieldResult")
    assert hasattr(drm, "FieldSpec")
    assert hasattr(drm, "RadioMapResult")
    assert not hasattr(drm, "GridSpec")
    assert not hasattr(drm, "Result")
    assert hasattr(drm, "solve")
    assert hasattr(drm, "solve_field")
    assert hasattr(drm, "native_extension_available")
    assert not hasattr(drm, "Solver")


def test_field_spec_validation_and_resolution_behavior():
    with pytest.raises(ValueError, match="axis"):
        FieldSpec(axis="bad", position=0.0, bounds=((-1, 1), (-1, 1)), grid_shape=(2, 2))
    with pytest.raises(ValueError, match="bounds"):
        FieldSpec(axis="z", position=0.0, bounds=((1, -1), (-1, 1)), grid_shape=(2, 2))
    with pytest.raises(ValueError, match="grid_shape"):
        FieldSpec(axis="z", position=0.0, bounds=((-1, 1), (-1, 1)), grid_shape=(0, 2))
    with pytest.raises(ValueError, match="resolution"):
        FieldSpec(axis="z", position=0.0, bounds=((-1, 1), (-1, 1)), resolution=0.0)

    explicit = FieldSpec(axis="Z", position=1.5, bounds=((-1, 1), (-2, 2)), grid_shape=(3, 5))
    assert explicit.axis == "z"
    assert explicit.tangential_axes == ("x", "y")
    assert explicit.resolve_grid_shape(0.5) == (3, 5)

    auto = FieldSpec(axis="x", position=0.0, bounds=((-1, 1), (-1, 2)), resolution=0.5)
    assert auto.tangential_axes == ("y", "z")
    assert auto.resolve_grid_shape(1.0) == (4, 6)


def test_field_grid_boundary_coordinates_and_span_over_n_bins():
    field = FieldSpec(
        axis="z",
        position=1.5,
        bounds=((0.0, 2.0), (10.0, 14.0)),
        grid_shape=(3, 2),
    ).to_field(1.0)
    coords = field.get_coordinates()

    np.testing.assert_allclose(np.asarray(coords["x_coords"]), [0.0, 1.0, 2.0])
    np.testing.assert_allclose(np.asarray(coords["y_coords"]), [10.0, 14.0])
    np.testing.assert_allclose(np.asarray(coords["X"]), [0.0, 1.0, 2.0, 0.0, 1.0, 2.0])
    np.testing.assert_allclose(np.asarray(coords["Y"]), [10.0, 10.0, 10.0, 14.0, 14.0, 14.0])
    assert field.cell_size == pytest.approx((2.0 / 3.0, 4.0 / 2.0))

    idx = field.pos_to_idx(wt.Float([0.0, 0.7, 2.0]), wt.Float([10.0, 12.1, 14.0]))
    np.testing.assert_array_equal(np.asarray(idx, dtype=np.uint32), [0, 4, 5])


def test_sionna_first_order_diffraction_point_selection():
    edge_origin = wt.Point3f(1.0, 0.0, 0.5)
    edge_dir = wt.Vector3f(0.0, 0.0, 1.0)

    coplanar = Geo.first_order_diffraction_parameter(
        wt.Point3f(0.0, -5.0, 1.5),
        wt.Point3f(0.0, 5.0, 1.5),
        edge_origin,
        edge_dir,
    )
    elevated = Geo.first_order_diffraction_parameter(
        wt.Point3f(0.0, -5.0, 3.0),
        wt.Point3f(0.0, 5.0, 1.0),
        edge_origin,
        edge_dir,
    )

    assert float(coplanar[0]) == pytest.approx(1.0, rel=1e-6, abs=1e-6)
    assert float(elevated[0]) == pytest.approx(1.5, rel=1e-6, abs=1e-6)


@pytest.mark.parametrize(
    ("axis", "position", "bounds"),
    [
        ("z", 1.5, ((-1.0, 1.0), (-1.0, 1.0))),
        ("x", 1.0, ((-1.0, 1.0), (0.5, 2.5))),
        ("y", 1.0, ((-1.0, 1.0), (0.5, 2.5))),
    ],
)
def test_los_only_solves_supported_axis_aligned_planes(axis, position, bounds):
    result = solve_field(
        scene=_far_scene(),
        frequency=FREQUENCY,
        tx_pos=wt.Point3f(0.0, -5.0, 1.5),
        field=FieldSpec(axis=axis, position=position, bounds=bounds, grid_shape=(4, 4)),
        config=Config(
            num_samples=8,
            max_bounces=0,
            max_diffraction_order=0,
            edge_policy=EdgePolicy(edge_selection_mode="all_edges"),
        ),
    )

    np.testing.assert_allclose(np.asarray(result.field.total.real), np.asarray(result.field.los.real))
    np.testing.assert_allclose(np.asarray(result.field.total.imag), np.asarray(result.field.los.imag))
    assert _complex_abs_sum(result.field.reflection) == pytest.approx(0.0)
    assert _complex_abs_sum(result.field.diffraction) == pytest.approx(0.0)
    assert np.all(np.isfinite(np.asarray(result.power["total"], dtype=np.float64)))
    assert not hasattr(result, "timing")
    assert "performance_timing" not in result.metadata
    assert "runtime_backends" not in result.metadata
    assert result.coords.tangential_axes == FieldSpec(
        axis=axis,
        position=position,
        bounds=bounds,
        grid_shape=(4, 4),
    ).tangential_axes


def test_reflection_and_diffraction_smoke_on_channel_scene():
    result = solve_field(
        scene=_cube_scene(size=1.0),
        frequency=FREQUENCY,
        tx_pos=wt.Point3f(0.0, -3.0, 1.5),
        field=FieldSpec(
            axis="z",
            position=1.5,
            bounds=((-2.0, 2.0), (-2.0, 2.0)),
            grid_shape=(3, 3),
            ray_mode="2d",
        ),
        config=Config(
            num_samples=16,
            max_bounces=1,
            max_diffraction_order=1,
            edge_policy=EdgePolicy(edge_selection_mode="all_edges"),
            tuning=Tuning(
                enable_rd_diffraction=True,
                diffraction_state_budget=64,
                inserted_reflection_state_budget=32,
            ),
        ),
        return_diffraction_audit=True,
    )

    assert result.grid_shape == (3, 3)
    assert result.diffraction_detail is not None
    assert result.metadata["diffraction"]["diffraction_skipped"] is False
    assert result.metadata["diffraction"]["edge_diffraction"] is True
    assert result.metadata["diffraction"]["boundary_edge_policy"] == "half_plane"
    assert np.all(np.isfinite(np.asarray(result.power["total"], dtype=np.float64)))


def test_ad_flow_for_transmitter_position():
    tx_x = wt.Float(0.0)
    dr.enable_grad(tx_x)
    result = solve_field(
        scene=_far_scene(),
        frequency=FREQUENCY,
        tx_pos=wt.Point3f(tx_x, -5.0, 1.5),
        field=FieldSpec(axis="z", position=1.5, bounds=((-1.0, 1.0), (-1.0, 1.0)), grid_shape=(4, 4)),
        config=Config(
            num_samples=8,
            max_bounces=0,
            max_diffraction_order=0,
            edge_policy=EdgePolicy(edge_selection_mode="all_edges"),
        ),
    )

    dr.set_grad(tx_x, 1.0)
    jvp = np.asarray(
        dr.forward_to(result.power["total"], flags=dr.ADFlag.Default | dr.ADFlag.AllowNoGrad),
        dtype=np.float64,
    )

    assert np.all(np.isfinite(jvp))
    assert float(np.sum(np.abs(jvp))) > 0.0


def test_ad_flow_for_moving_cube_geometry_with_direct_diffraction():
    cube1_x = wt.Float(-0.5)
    dr.enable_grad(cube1_x)
    result = solve_field(
        scene=_moving_cube_scene(cube1_x),
        frequency=FREQUENCY,
        tx_pos=wt.Point3f(0.0, -4.0, 1.5),
        field=FieldSpec(axis="z", position=1.5, bounds=((-3.0, 3.0), (-3.0, 3.0)), grid_shape=(4, 4)),
        config=Config(
            num_samples=64,
            max_bounces=1,
            max_diffraction_order=1,
            edge_policy=EdgePolicy(edge_selection_mode="all_edges"),
            tuning=Tuning(
                enable_rd_diffraction=False,
                diffraction_state_budget=64,
                inserted_reflection_state_budget=32,
            ),
        ),
    )

    dr.set_grad(cube1_x, 1.0)
    jvp = np.asarray(
        dr.forward_to(result.field.total.real, flags=dr.ADFlag.Default | dr.ADFlag.AllowNoGrad),
        dtype=np.float64,
    )

    assert np.all(np.isfinite(jvp))
    assert float(np.sum(np.abs(jvp))) > 0.0


def test_field_solver_multipath_example_smoke_profile_runs():
    profile = smoke_profile()

    assert profile["forward_shape"] == (16, 16)
    assert profile["tx_x_ad_l1"] > 0.0
    assert profile["cube1_eps_ad_l1"] > 0.0
    assert profile["cube1_eps_diffraction_ad_l1"] > 0.0


def test_field_solver_3d_three_cubes_example_smoke_profile_runs():
    profile = smoke_profile_3d()

    assert profile["forward_shape"] == (16, 16)
    assert profile["tx_x_ad_l1"] > 0.0


def test_field_solver_total_gradient_with_rd_diffraction_smoke():
    experiment = ThreeCubeFieldExperiment(
        grid_shape=(16, 16),
        num_samples=64,
        max_diffraction_order=2,
    )

    gradient = experiment.gradient(
        "tx_x",
        grid_shape=(16, 16),
        num_samples=64,
        max_diffraction_order=2,
        enable_rd_diffraction=True,
    )

    assert gradient.ad.shape == (16, 16)
    assert gradient.fd.shape == (16, 16)
    assert gradient.delta.shape == (16, 16)
    assert np.all(np.isfinite(gradient.ad))
    assert np.all(np.isfinite(gradient.fd))
    assert float(np.sum(np.abs(gradient.ad))) > 0.0

