"""Shared helpers for deterministic reflection tracing.

Bundles material/Fresnel response, grad-sensitivity probes, ray-direction sampling,
and monitor-plane / Jones-projection conventions.
"""

import drjit as dr
from witwin.channel.deterministic import types as wt

from witwin.channel.core.runtime import (
    Material,
    Rx,
    Tx,
    point_grad_enabled,
    scene_geometry_grad_enabled,
    scene_has_material_table,
    scene_material_grad_enabled,
)
from witwin.channel.core.numerics.arrays import complex_zero, scalar
from witwin.channel.core.geometry.raygen import (
    generate_circle_directions,
    generate_hemisphere_directions,
    generate_sphere_directions,
)


AUTO_HEMISPHERE_NEAR_PLANE_RATIO = 0.25
PATH_IMAGE_SOURCE_TOL = 1e-5


# ---------- grid / monitor frame -------------------------------------------

def resolve_plane(grid, rx_z):
    axis = getattr(grid, "axis", "z")
    if axis == "z":
        return axis, float(rx_z)
    return axis, float(getattr(grid, "position", rx_z))


def axis_coordinate(point, axis: str) -> float:
    return float(scalar(point[axis] if isinstance(point, dict) else getattr(point, axis)))


def monitor_facing_normal(axis: str, plane_position: float, tx: Tx):
    sign = 1.0 if (plane_position - axis_coordinate(tx.position, axis)) >= 0.0 else -1.0
    if axis == "x":
        return wt.Vector3f(sign, 0.0, 0.0)
    if axis == "y":
        return wt.Vector3f(0.0, sign, 0.0)
    return wt.Vector3f(0.0, 0.0, sign)


# ---------- gradient sensitivity -------------------------------------------

def grad_sensitive_workload(*, tx: Tx, scene) -> bool:
    return (
        point_grad_enabled(tx.position)
        or scene_geometry_grad_enabled(scene)
        or scene_material_grad_enabled(scene)
    )


# ---------- ray-direction sampling -----------------------------------------

def resolve_sampling_info(
    *, axis, bounds, tx: Tx, mode, plane_position, ray_sampling
):
    plane_distance = abs(plane_position - axis_coordinate(tx.position, axis))
    if mode == "2d":
        return {
            "requested_ray_sampling": "circle",
            "selected_ray_sampling": "circle_2d",
            "monitor_plane_distance_to_tx": plane_distance,
            "near_plane_sampling_threshold": 0.0,
        }

    requested_sampling = str(ray_sampling).lower()
    if requested_sampling not in {"auto", "full_sphere", "hemisphere"}:
        raise ValueError("ray_sampling must be one of 'auto', 'full_sphere', or 'hemisphere'.")

    span = min(float(bounds[0][1] - bounds[0][0]), float(bounds[1][1] - bounds[1][0]))
    near_plane_threshold = AUTO_HEMISPHERE_NEAR_PLANE_RATIO * span
    selected_sampling = requested_sampling
    if requested_sampling == "auto":
        selected_sampling = "full_sphere" if plane_distance <= near_plane_threshold else "hemisphere"

    distribution = "full_sphere" if selected_sampling == "full_sphere" else "hemisphere_facing_monitor"
    return {
        "requested_ray_sampling": requested_sampling,
        "selected_ray_sampling": distribution,
        "monitor_plane_distance_to_tx": plane_distance,
        "near_plane_sampling_threshold": near_plane_threshold,
    }


def select_ray_directions(
    *, axis, bounds, tx: Tx, n_rays, mode, plane_position, ray_sampling
):
    sampling_info = resolve_sampling_info(
        axis=axis, bounds=bounds, tx=tx, mode=mode,
        plane_position=plane_position, ray_sampling=ray_sampling,
    )
    selected = sampling_info["selected_ray_sampling"]
    if selected == "circle_2d":
        return generate_circle_directions(n_rays), sampling_info
    if selected == "full_sphere":
        return generate_sphere_directions(n_rays), sampling_info
    return generate_hemisphere_directions(
        n_rays, monitor_facing_normal(axis, plane_position, tx),
    ), sampling_info


def trace_material_info(*, scene, material: Material):
    del material
    model_source = "scene" if scene_has_material_table(scene) else "default"
    return "materialized", model_source


# ---------- complex-field helpers ------------------------------------------

def sum_complex_fields(fields, *, n_rx: int):
    field_list = tuple(fields)
    if not field_list:
        return complex_zero(n_rx)
    total = field_list[0]
    for field in field_list[1:]:
        total = total + field
    return total


def sampling_frame_from_rx(rx: Rx):
    rx_positions = rx.positions
    coord_min = (
        float(scalar(dr.min(rx_positions.x))),
        float(scalar(dr.min(rx_positions.y))),
        float(scalar(dr.min(rx_positions.z))),
    )
    coord_max = (
        float(scalar(dr.max(rx_positions.x))),
        float(scalar(dr.max(rx_positions.y))),
        float(scalar(dr.max(rx_positions.z))),
    )
    spans = [upper - lower for lower, upper in zip(coord_min, coord_max)]
    axis_idx = int(min(range(3), key=spans.__getitem__))
    axis = ("x", "y", "z")[axis_idx]
    tangential_indices = tuple(index for index in range(3) if index != axis_idx)
    bounds = tuple((coord_min[t_idx], coord_max[t_idx]) for t_idx in tangential_indices)
    plane_position = float((coord_min[axis_idx] + coord_max[axis_idx]) * 0.5)
    return axis, plane_position, bounds
