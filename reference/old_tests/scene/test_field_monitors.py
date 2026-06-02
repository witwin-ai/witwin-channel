from __future__ import annotations

import numpy as np
import pytest
import witwin as wt
from tests._scene_helpers import box_geometry, build_scene
from witwin.channel import Field, MonitorResult, FieldMonitor, Tracer
TEST_WAVELENGTH = 299792458.0 / 1e9


def _build_scene(*, monitors=None):
    cube = box_geometry(center=(0.0, 0.0, 1.5), size=2.0)
    return build_scene(cube, monitors=monitors)


@pytest.mark.gpu
def test_scene_accepts_field_monitors_and_resolves_them():
    plane = FieldMonitor(
        "field_xy",
        axis="z",
        position=1.5,
        bounds=((-2.0, 2.0), (-3.0, 3.0)),
        grid_size=(8, 6),
    )
    scene = _build_scene(monitors=[plane])

    assert scene.monitors == [plane]
    assert scene.resolved_monitors() == [plane]
    assert scene.clone().monitors == [plane]


@pytest.mark.gpu
def test_tracer_trace_uses_scene_monitor_when_trace_monitor_is_omitted():
    plane = FieldMonitor(
        "field_xy",
        axis="z",
        position=1.5,
        bounds=((-2.0, 2.0), (-3.0, 3.0)),
        grid_size=(8, 6),
    )
    scene = _build_scene(monitors=[plane])
    tracer = Tracer(
        frequency=1e9,
        scene=scene,
        reflection_n_rays=0,
        reflection_max_bounces=0,
        max_diffractions=1,
    )

    payload = tracer.trace(wt.Point3f(0.0, -4.0, 1.5), verbose=False)

    assert payload.grid_shape == (8, 6)
    assert np.asarray(payload.coords.x).shape == (8,)
    assert np.asarray(payload.coords.y).shape == (6,)
    assert payload.coords.axis_x == "x"
    assert payload.coords.axis_y == "y"
    assert payload.tangential_axes == ("x", "y")
    assert payload.metadata["receiver_sampling"]["monitor_name"] == "field_xy"
    assert tuple(payload.metadata["receiver_sampling"]["tangential_axes"]) == ("x", "y")
    assert payload.metadata["receiver_sampling"]["sample_positions"] == "boundary_points"
    assert payload.metadata["receiver_sampling"]["index_partitioning"] == "span_over_n_bins"
    assert payload.metadata["receiver_sampling"]["future_3d_plane_switch_preserved"] is True
    assert np.asarray(payload.field.total.real).shape == (48,)


@pytest.mark.gpu
def test_trace_monitor_overrides_scene_monitors():
    scene_monitor = FieldMonitor(
        "scene_plane",
        axis="z",
        position=1.5,
        bounds=((-2.0, 2.0), (-2.0, 2.0)),
        grid_size=(8, 8),
    )
    trace_monitor = FieldMonitor(
        "trace_field",
        axis="z",
        position=2.0,
        bounds=((-1.0, 1.0), (-3.0, 3.0)),
        grid_size=(5, 7),
        ray_mode="3d",
    )
    scene = _build_scene(monitors=[scene_monitor])
    tracer = Tracer(
        frequency=1e9,
        scene=scene,
        reflection_n_rays=0,
        reflection_max_bounces=0,
        max_diffractions=1,
    )

    result = tracer.trace(
        wt.Point3f(0.0, -4.0, 1.5),
        monitor=trace_monitor,
        verbose=False,
    )
    payload = result

    assert payload.ray_mode == "3d"
    assert payload.range_x == (-1.0, 1.0)
    assert payload.range_y == (-3.0, 3.0)
    assert payload.grid_shape == (5, 7)
    assert payload.metadata["receiver_sampling"]["resolution_source"] == "tracer_default"


@pytest.mark.gpu
def test_monitor_auto_grid_uses_tracer_default_resolution():
    plane = FieldMonitor(
        "field_xy",
        axis="z",
        position=1.5,
        bounds=((-2.0, 2.0), (-4.0, 4.0)),
        resolution=None,
    )
    scene = _build_scene(monitors=[plane])
    tracer = Tracer(
        frequency=1e9,
        scene=scene,
        reflection_n_rays=0,
        reflection_max_bounces=0,
        max_diffractions=1,
        resolution_wavelength=0.25,
    )

    result = tracer.trace(wt.Point3f(0.0, -4.0, 1.5), verbose=False)
    expected_grid_shape = plane.resolve_grid_shape(
        tracer.wavelength,
        default_resolution=tracer.resolution_wavelength,
    )

    assert result.grid_shape == expected_grid_shape
    assert result.grid_shape[0] != result.grid_shape[1]
    assert result.metadata["receiver_sampling"]["resolution_wavelength"] == 0.25
    assert result.metadata["receiver_sampling"]["resolution_source"] == "tracer_default"


@pytest.mark.gpu
def test_trace_requires_monitor_when_scene_has_none():
    scene = _build_scene()
    tracer = Tracer(
        frequency=1e9,
        scene=scene,
        reflection_n_rays=0,
        reflection_max_bounces=0,
        max_diffractions=1,
    )

    with pytest.raises(ValueError, match="requires monitor"):
        tracer.trace(wt.Point3f(0.0, -4.0, 1.5), verbose=False)


@pytest.mark.parametrize(
    ("axis", "expected_tangential_axes"),
    [
        ("x", ("y", "z")),
        ("y", ("x", "z")),
        ("z", ("x", "y")),
    ],
)
def test_field_monitor_to_field_supports_all_axis_aligned_planes(axis, expected_tangential_axes):
    monitor = FieldMonitor(
        f"plane_{axis}",
        axis=axis,
        position=5.0,
        bounds=((-3.0, 3.0), (-2.0, 4.0)),
        grid_size=(7, 9),
    )

    field = monitor.to_field(TEST_WAVELENGTH)
    coords = field.get_coordinates()

    assert monitor.tangential_axes == expected_tangential_axes
    assert field.axis == axis
    assert field.position == 5.0
    assert field.tangential_axes == expected_tangential_axes
    assert field.normal_axis == axis
    assert coords["axis_x"] == expected_tangential_axes[0]
    assert coords["axis_y"] == expected_tangential_axes[1]
    assert coords["axis"] == axis
    assert coords["position"] == 5.0
    assert np.asarray(coords["x_coords"]).shape == (7,)
    assert np.asarray(coords["y_coords"]).shape == (9,)


@pytest.mark.parametrize(
    ("axis", "tangential_axes", "position"),
    [
        ("x", ("y", "z"), 5.0),
        ("y", ("x", "z"), -2.5),
        ("z", ("x", "y"), 1.5),
    ],
)
def test_field_receiver_positions_3d_match_axis_aligned_planes(axis, tangential_axes, position):
    field = Field(
        bounds=((-1.0, 1.0), (-2.0, 2.0)),
        size=(3, 4),
        axis=axis,
        position=position,
    )
    coords = field.get_coordinates()
    rx_positions = field.receivers

    assert field.tangential_axes == tangential_axes

    coord_0 = np.asarray(coords["X"], dtype=np.float64)
    coord_1 = np.asarray(coords["Y"], dtype=np.float64)
    rx_x = np.asarray(rx_positions.x, dtype=np.float64)
    rx_y = np.asarray(rx_positions.y, dtype=np.float64)
    rx_z = np.asarray(rx_positions.z, dtype=np.float64)

    if axis == "x":
        assert np.allclose(rx_x, position)
        assert np.allclose(rx_y, coord_0)
        assert np.allclose(rx_z, coord_1)
    elif axis == "y":
        assert np.allclose(rx_x, coord_0)
        assert np.allclose(rx_y, position)
        assert np.allclose(rx_z, coord_1)
    else:
        assert np.allclose(rx_x, coord_0)
        assert np.allclose(rx_y, coord_1)
        assert np.allclose(rx_z, position)


def test_monitor_result_preserves_tangential_axis_labels():
    zero = np.zeros(6, dtype=np.float32)
    payload = {
        "name": "yz_plane",
        "kind": "field",
        "axis": "x",
        "plane_position": 5.0,
        "ray_mode": "3d",
        "bounds": ((-3.0, 3.0), (-2.0, 4.0)),
        "grid_shape": (2, 3),
        "coords": {
            "grid_x": zero,
            "grid_y": zero,
            "x": np.asarray([-3.0, 3.0], dtype=np.float32),
            "y": np.asarray([-2.0, 1.0, 4.0], dtype=np.float32),
            "axis_x": "y",
            "axis_y": "z",
        },
        "field": {
            "los": zero,
            "reflection": zero,
            "diffraction_direct": zero,
            "diffraction_mixed": zero,
            "diffraction": zero,
            "total": zero,
        },
        "vector": {
            "los": {"x": 0.0, "y": 0.0, "z": 0.0},
            "reflection": {"x": 0.0, "y": 0.0, "z": 0.0},
            "diffraction_direct": {"x": 0.0, "y": 0.0, "z": 0.0},
            "diffraction_mixed": {"x": 0.0, "y": 0.0, "z": 0.0},
            "diffraction": {"x": 0.0, "y": 0.0, "z": 0.0},
            "total": {"x": 0.0, "y": 0.0, "z": 0.0},
        },
        "jones": {
            "los": {"x": 0.0, "y": 0.0},
            "reflection": {"x": 0.0, "y": 0.0},
            "diffraction_direct": {"x": 0.0, "y": 0.0},
            "diffraction_mixed": {"x": 0.0, "y": 0.0},
            "diffraction": {"x": 0.0, "y": 0.0},
            "total": {"x": 0.0, "y": 0.0},
        },
        "metadata": {
            "receiver_sampling": {
                "axis": "x",
                "tangential_axes": ("y", "z"),
            },
        },
        "tx_pos": (0.0, -1.0, 2.0),
    }

    result = MonitorResult.from_payload(payload)

    assert result.axis == "x"
    assert result.plane_position == 5.0
    assert result.coords.axis_x == "y"
    assert result.coords.axis_y == "z"
    assert result.tangential_axes == ("y", "z")
    assert result.range_x == (-3.0, 3.0)
    assert result.range_y == (-2.0, 4.0)
