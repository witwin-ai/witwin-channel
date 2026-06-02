"""Reflection tracing using ray tracing + image method"""

import drjit as dr
import rayd
import witwin as wt

from ...utils.constants import EPS, RAY_ORIGIN_BIAS
from ...utils.drjit_ops import ArrayInit
from ...utils.polarization import (
    effective_rx_polarization,
    project_real_polarization_to_ray,
    reflect_field_vector,
    scalarize_tangential_jones,
    tangential_jones,
    vector_scale,
    vector_select,
    vector_zero,
)
from ...utils import scalar
from ...utils.raygen import (
    generate_circle_directions,
    generate_hemisphere_directions,
    generate_sphere_directions,
)
from ..materials import (
    bounce_reflection_weight as _bounce_reflection_weight,
    build_reflection_trace_detail,
    coerce_reflection_trace_detail,
    material_source_label,
    normalized_override_material as _normalized_reflection_material,
    reflection_material_omega as _reflection_material_omega,
    reflection_model_label,
    resolve_surface_material,
)
from ...kernels.trace.reflection import reflection_accumulate_forward as accumulate_reflection_paths_to_receivers
from ...kernels.trace.reflection import drjit_impl as reflection_epc_drjit
try:
    from ...kernels.trace.reflection import native_impl as reflection_epc_native
except Exception:  # pragma: no cover - native extension may be unavailable
    reflection_epc_native = None
from ...kernels.monitors.common.receiver_tiles import resolve_receiver_tiles
from ...monitors.field import grid_reflection as reflection_grid_drjit
from ...kernels.monitors.field.reflection_grid import native_impl as reflection_grid_native
from ..._native import native_extension_available
from ...utils.geometry import reflect_point_across_plane
from .paths import (
    REFLECTION_PATH_IMAGE_SOURCE_TOL,
    _collect_reflection_prefix_paths_from_rayd_chain,
    _collect_unique_reflection_paths,
    _empty_source_path_data,
)


_DDA_DEGENERATE_DT_THRESHOLD = 1e6
_AUTO_HEMISPHERE_NEAR_PLANE_RATIO = 0.25


def _mask_count(mask) -> int:
    return int(scalar(dr.sum(dr.select(mask, wt.UInt32(1), wt.UInt32(0)))))


def _masked_max(value, mask) -> float:
    return float(scalar(dr.max(dr.select(mask, value, wt.Float(0.0)))))


def _resolve_reflection_plane(grid, rx_z):
    try:
        axis = grid.axis
    except AttributeError:
        axis = "z"
    if axis == "z":
        return axis, float(rx_z)
    try:
        position = grid.position
    except AttributeError:
        position = rx_z
    return axis, float(position)


def _axis_coordinate(point, axis: str) -> float:
    return float(scalar(point[axis] if isinstance(point, dict) else getattr(point, axis)))


def _monitor_characteristic_span(bounds) -> float:
    span_0 = float(bounds[0][1] - bounds[0][0])
    span_1 = float(bounds[1][1] - bounds[1][0])
    return min(span_0, span_1)


def _point_grad_enabled(point) -> bool:
    if point is None:
        return False
    try:
        return any(bool(dr.grad_enabled(component)) for component in (point.x, point.y, point.z))
    except Exception:
        return False


def _scene_geometry_grad_enabled(scene) -> bool:
    if scene is None:
        return False

    vertices = scene.vertices
    if _point_grad_enabled(vertices):
        return True

    tri_data = scene.tri_data_gpu
    if isinstance(tri_data, dict):
        for key in ("v0", "v1", "v2"):
            value = tri_data.get(key)
            if value is not None and _point_grad_enabled(value):
                return True
    return False


def _scene_material_grad_enabled(scene) -> bool:
    if scene is None:
        return False
    tri_data = scene.tri_data_gpu
    if isinstance(tri_data, dict):
        for key in ("material_eps_r", "material_sigma_e"):
            value = tri_data.get(key)
            if value is None:
                continue
            try:
                if bool(dr.grad_enabled(value)):
                    return True
            except Exception:
                continue
    return False


def _reflection_grad_sensitive_workload(*, tx_pos, scene) -> bool:
    return (
        _point_grad_enabled(tx_pos)
        or _scene_geometry_grad_enabled(scene)
        or _scene_material_grad_enabled(scene)
    )


def _reflection_epc_policy(
    *,
    reflection_detail,
    grid_axis: str,
    has_mesh_data: bool,
    scene,
    tx_pos,
    source_paths_per_bounce=None,
    prefer_epc: bool = True,
):
    tx_grad_enabled = _point_grad_enabled(tx_pos)
    scene_geometry_grad_enabled = _scene_geometry_grad_enabled(scene)
    scene_material_grad_enabled = _scene_material_grad_enabled(scene)
    grad_sensitive_workload = _reflection_grad_sensitive_workload(
        tx_pos=tx_pos,
        scene=scene,
    )
    if reflection_detail is not None:
        return {
            "use_epc": True,
            "policy": "provided_reflection_detail_epc",
            "reused_discovery": True,
            "discovery_gradients_preserved": False,
            "tx_grad_enabled": tx_grad_enabled,
            "scene_geometry_grad_enabled": scene_geometry_grad_enabled,
            "scene_material_grad_enabled": scene_material_grad_enabled,
            "epc_eligible": True,
        }

    eligible = grid_axis == "z" and has_mesh_data
    if source_paths_per_bounce is not None:
        eligible = eligible and any(
            int(paths.n_paths) > 0 for paths in source_paths_per_bounce if paths is not None
        )

    if not eligible:
        return {
            "use_epc": False,
            "policy": "fresh_trace_requested_backend",
            "reused_discovery": False,
            "discovery_gradients_preserved": True,
            "tx_grad_enabled": tx_grad_enabled,
            "scene_geometry_grad_enabled": scene_geometry_grad_enabled,
            "scene_material_grad_enabled": scene_material_grad_enabled,
            "epc_eligible": False,
        }

    if grad_sensitive_workload:
        return {
            "use_epc": False,
            "policy": "fresh_trace_geometry_grad_preserving_requested_backend",
            "reused_discovery": False,
            "discovery_gradients_preserved": True,
            "tx_grad_enabled": tx_grad_enabled,
            "scene_geometry_grad_enabled": scene_geometry_grad_enabled,
            "scene_material_grad_enabled": scene_material_grad_enabled,
            "epc_eligible": True,
        }

    if not prefer_epc:
        return {
            "use_epc": False,
            "policy": "fresh_trace_native_requested_backend",
            "reused_discovery": False,
            "discovery_gradients_preserved": True,
            "tx_grad_enabled": False,
            "scene_geometry_grad_enabled": False,
            "scene_material_grad_enabled": False,
            "epc_eligible": True,
        }

    return {
        "use_epc": True,
        "policy": "fresh_trace_epc",
        "reused_discovery": False,
        "discovery_gradients_preserved": False,
        "tx_grad_enabled": False,
        "scene_geometry_grad_enabled": False,
        "scene_material_grad_enabled": False,
        "epc_eligible": True,
    }


def _monitor_facing_normal(axis: str, plane_position: float, tx_pos):
    delta = plane_position - _axis_coordinate(tx_pos, axis)
    sign = 1.0 if delta >= 0.0 else -1.0
    if axis == "x":
        return wt.Vector3f(sign, 0.0, 0.0)
    if axis == "y":
        return wt.Vector3f(0.0, sign, 0.0)
    return wt.Vector3f(0.0, 0.0, sign)


def _resolve_reflection_sampling_metadata(*, axis, bounds, tx_pos, mode, plane_position, ray_sampling):
    plane_distance = abs(plane_position - _axis_coordinate(tx_pos, axis))
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

    near_plane_threshold = _AUTO_HEMISPHERE_NEAR_PLANE_RATIO * _monitor_characteristic_span(bounds)
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


def _select_reflection_ray_directions(*, axis, bounds, tx_pos, n_rays, mode, plane_position, ray_sampling):
    metadata = _resolve_reflection_sampling_metadata(
        axis=axis,
        bounds=bounds,
        tx_pos=tx_pos,
        mode=mode,
        plane_position=plane_position,
        ray_sampling=ray_sampling,
    )
    if metadata["selected_ray_sampling"] == "circle_2d":
        return generate_circle_directions(n_rays), metadata
    if metadata["selected_ray_sampling"] == "full_sphere":
        return generate_sphere_directions(n_rays), metadata
    return generate_hemisphere_directions(
        n_rays,
        _monitor_facing_normal(axis, plane_position, tx_pos),
    ), metadata


def _reflection_jones_metadata(
    axis: str,
    *,
    explicit_receiver_projection: bool,
    tangential_axes: tuple[str, str],
) -> dict[str, object]:
    if axis == "z":
        return {
            "scalar_projection_rule": (
                "global_xy_receiver_projection_from_jones"
                if explicit_receiver_projection
                else "global_xy_default_receiver_projection_from_jones"
            ),
            "reflection_scalarization": (
                "explicit_receiver_projection_from_jones"
                if explicit_receiver_projection
                else "default_receiver_projection_from_jones"
            ),
            "result_jones_basis": "global_xy",
            "result_jones_axes": ("x", "y"),
        }
    return {
        "scalar_projection_rule": (
            "explicit_monitor_tangential_receiver_projection_from_jones"
            if explicit_receiver_projection
            else "default_monitor_tangential_receiver_projection_from_jones"
        ),
        "reflection_scalarization": (
            "explicit_monitor_tangential_receiver_projection_from_jones"
            if explicit_receiver_projection
            else "default_monitor_tangential_receiver_projection_from_jones"
        ),
        "result_jones_basis": "monitor_tangential",
        "result_jones_axes": tangential_axes,
    }


def _reflection_trace_material_metadata(
    *,
    scene,
    reflection_material,
    reflection_coef,
    reflection_relative_permittivity,
    reflection_conductivity,
    use_scene_materials,
):
    normalized_reflection_material = _normalized_reflection_material(
        reflection_material,
        reflection_coef=reflection_coef,
        eta_r=reflection_relative_permittivity,
        sigma=reflection_conductivity,
    )
    return (
        normalized_reflection_material,
        reflection_model_label(
            scene,
            normalized_reflection_material,
            use_scene_materials=use_scene_materials,
        ),
        material_source_label(
            scene,
            normalized_reflection_material,
            use_scene_materials=use_scene_materials,
        ),
    )


def _reflection_detail_material_override(*, reflection_model_source: str, reflection_material):
    return None if str(reflection_model_source) == "scene" else reflection_material


def _sum_complex_fields(fields, *, n_rx: int):
    field_list = tuple(fields)
    if len(field_list) == 0:
        return ArrayInit.complex_zero(n_rx)
    if len(field_list) == 1:
        return field_list[0]
    if len(field_list) == 2:
        return field_list[0] + field_list[1]
    if len(field_list) == 3:
        return field_list[0] + field_list[1] + field_list[2]
    total = field_list[0] + field_list[1] + field_list[2]
    for field in field_list[3:]:
        total = total + field
    return total


def _complex_from_accumulators(real, imag, count):
    mask = count > 0
    safe_count = dr.select(mask, count, wt.Float(1.0))
    return wt.Complex2f(
        dr.select(mask, real / safe_count, wt.Float(0.0)),
        dr.select(mask, imag / safe_count, wt.Float(0.0)),
    )


def _accumulate_reflection_paths_exact(
    *,
    rx_pos,
    scene,
    wavelength,
    k,
    source_paths_per_bounce,
    reflection_detail,
    tx_polarization,
    receiver_tiles=None,
):
    detail = coerce_reflection_trace_detail(reflection_detail)
    if receiver_tiles is not None and receiver_tiles.receiver_positions is not None:
        rx_pos = receiver_tiles.receiver_positions
    requires_ad = False
    if reflection_epc_native is not None:
        requires_ad = any(
            reflection_epc_native._reflection_paths_require_ad(paths=paths, rx_pos=rx_pos)
            for paths in source_paths_per_bounce
            if paths is not None
        )
    use_native_replay = (
        reflection_epc_native is not None
        and native_extension_available()
        and not bool(detail.use_scene_materials)
        and not requires_ad
    )
    accumulator = (
        reflection_epc_native.reflection_accumulate_forward
        if use_native_replay
        else reflection_epc_drjit.reflection_accumulate_forward
    )
    accumulator_kwargs = {
        "rx_pos": rx_pos,
        "scene": scene,
        "wavelength": wavelength,
        "k": k,
        "source_paths_per_bounce": source_paths_per_bounce,
        "reflection_detail": detail,
        "tx_polarization": tx_polarization,
    }
    if use_native_replay:
        accumulator_kwargs["receiver_tiles"] = receiver_tiles
    return accumulator(**accumulator_kwargs)


def _reflection_rayd_scene_handle(scene, tri_data):
    if scene is None or tri_data is None:
        return None
    if "surface_canonical_prim" not in tri_data:
        return None
    surface_canonical_prims = tri_data["surface_canonical_prim"]
    if dr.width(surface_canonical_prims) == 0:
        return None

    rayd_scene = scene._rayd_scene
    if rayd_scene is None:
        return None
    try:
        rayd_scene.trace_reflections
    except AttributeError:
        return None
    return rayd_scene


def _trace_reflection_paths_rayd(
    *,
    tx_pos,
    scene,
    wavelength,
    n_rays,
    max_reflections,
    mode,
    reflection_coef,
    ray_sampling,
    tx_polarization,
    reflection_relative_permittivity,
    reflection_conductivity,
    reflection_material,
    use_scene_materials,
    sampling_axis: str,
    sampling_bounds,
    sampling_plane_position: float,
    tri_data,
    ray_dir,
    ray_sampling_metadata,
    on_segment=None,
):
    del wavelength, tx_polarization, on_segment

    rayd_scene = _reflection_rayd_scene_handle(scene, tri_data)
    if rayd_scene is None:
        raise RuntimeError("Reflection discovery requires scene._rayd_scene.trace_reflections().")

    surface_canonical_prims = tri_data["surface_canonical_prim"]
    normalized_reflection_material, reflection_model, reflection_model_source = (
        _reflection_trace_material_metadata(
            scene=scene,
            reflection_material=reflection_material,
            reflection_coef=reflection_coef,
            reflection_relative_permittivity=reflection_relative_permittivity,
            reflection_conductivity=reflection_conductivity,
            use_scene_materials=use_scene_materials,
        )
    )
    ray_origin = wt.Point3f(
        dr.repeat(tx_pos.x, n_rays),
        dr.repeat(tx_pos.y, n_rays),
        dr.repeat(tx_pos.z, n_rays),
    )
    point3_detached = dr.detached_t(wt.Point3f)
    vector3_detached = dr.detached_t(wt.Vector3f)
    ray_origin_detached = point3_detached(
        dr.detach(ray_origin.x),
        dr.detach(ray_origin.y),
        dr.detach(ray_origin.z),
    )
    ray_dir_detached = vector3_detached(
        dr.detach(ray_dir.x),
        dr.detach(ray_dir.y),
        dr.detach(ray_dir.z),
    )
    symbolic_workload = _reflection_grad_sensitive_workload(
        tx_pos=tx_pos,
        scene=scene,
    )
    options = rayd.ReflectionTraceOptions()
    options.deduplicate = not symbolic_workload
    options.canonical_prim_table = surface_canonical_prims
    options.image_source_tolerance = float(REFLECTION_PATH_IMAGE_SOURCE_TOL)
    if symbolic_workload:
        chain = rayd_scene.trace_reflections(
            rayd.Ray(ray_origin, ray_dir),
            int(max_reflections),
            options,
            dr.full(wt.Bool, True, n_rays),
            True,
        )
    else:
        with dr.scoped_set_flag(dr.JitFlag.Recording, False):
            chain = rayd_scene.trace_reflections(
                rayd.RayDetached(ray_origin_detached, ray_dir_detached),
                int(max_reflections),
                options,
                dr.full(dr.detached_t(wt.Bool), True, n_rays),
                False,
            )
    source_paths_per_bounce = _collect_reflection_prefix_paths_from_rayd_chain(
        chain,
        chain_depth=max_reflections,
        surface_canonical_prims=surface_canonical_prims,
        image_source_tolerance=REFLECTION_PATH_IMAGE_SOURCE_TOL,
    )
    return {
        "source_paths_per_bounce": tuple(source_paths_per_bounce),
        "segment_stats": tuple(),
        "ray_sampling_metadata": ray_sampling_metadata,
        "reflection_model": reflection_model,
        "reflection_model_source": reflection_model_source,
        "normalized_reflection_material": normalized_reflection_material,
    }


def _trace_reflection_paths_legacy(
    *,
    tx_pos,
    scene,
    wavelength,
    n_rays,
    max_reflections,
    mode,
    reflection_coef,
    ray_sampling,
    tx_polarization,
    reflection_relative_permittivity,
    reflection_conductivity,
    reflection_material,
    use_scene_materials,
    sampling_axis: str,
    sampling_bounds,
    sampling_plane_position: float,
    tri_data,
    ray_dir,
    ray_sampling_metadata,
    on_segment=None,
):
    has_mesh_data = tri_data is not None
    if has_mesh_data:
        tri_v0 = tri_data["v0"]
        tri_v1 = tri_data["v1"]
        tri_v2 = tri_data["v2"]
        tri_surface_data = {
            "group_id": tri_data["surface_group_id"],
            "canonical_prim": tri_data["surface_canonical_prim"],
            "group_size": tri_data["surface_group_size"],
            "group_members": tri_data["surface_group_members"],
            "max_group_size": int(tri_data["surface_max_group_size"]),
        }
        surface_canonical_prims = tri_data["surface_canonical_prim"]
    else:
        tri_v0 = tri_v1 = tri_v2 = None
        tri_surface_data = None
        surface_canonical_prims = None

    source_paths_per_bounce = [_empty_source_path_data(chain_depth=bounce + 1) for bounce in range(max_reflections)]
    segment_stats = []

    ray_origin = wt.Point3f(
        dr.repeat(tx_pos.x, n_rays),
        dr.repeat(tx_pos.y, n_rays),
        dr.repeat(tx_pos.z, n_rays),
    )
    active = dr.full(wt.Bool, True, n_rays)
    weight = wt.Complex2f(dr.ones(wt.Float, n_rays), dr.zeros(wt.Float, n_rays))
    tx_pol_dir = project_real_polarization_to_ray(tx_polarization, ray_dir)
    polarization_vec = {
        "x": wt.Complex2f(tx_pol_dir.x, dr.zeros(wt.Float, n_rays)),
        "y": wt.Complex2f(tx_pol_dir.y, dr.zeros(wt.Float, n_rays)),
        "z": wt.Complex2f(tx_pol_dir.z, dr.zeros(wt.Float, n_rays)),
    }
    cumulative_image_source = wt.Point3f(
        dr.repeat(tx_pos.x, n_rays),
        dr.repeat(tx_pos.y, n_rays),
        dr.repeat(tx_pos.z, n_rays),
    )
    chain_prim_history = [dr.full(wt.Int32, -1, n_rays) for _ in range(max_reflections)]
    chain_plane_point_history = [
        wt.Point3f(
            dr.zeros(wt.Float, n_rays),
            dr.zeros(wt.Float, n_rays),
            dr.zeros(wt.Float, n_rays),
        )
        for _ in range(max_reflections)
    ]
    chain_plane_normal_history = [
        wt.Vector3f(
            dr.zeros(wt.Float, n_rays),
            dr.zeros(wt.Float, n_rays),
            dr.zeros(wt.Float, n_rays),
        )
        for _ in range(max_reflections)
    ]
    chain_hit_point_history = [
        wt.Point3f(
            dr.zeros(wt.Float, n_rays),
            dr.zeros(wt.Float, n_rays),
            dr.zeros(wt.Float, n_rays),
        )
        for _ in range(max_reflections)
    ]
    prev_refl_p = wt.Point3f(
        dr.zeros(wt.Float, n_rays),
        dr.zeros(wt.Float, n_rays),
        dr.zeros(wt.Float, n_rays),
    )
    prev_refl_n = wt.Vector3f(
        dr.zeros(wt.Float, n_rays),
        dr.zeros(wt.Float, n_rays),
        dr.zeros(wt.Float, n_rays),
    )
    prev_tx = wt.Point3f(
        dr.repeat(tx_pos.x, n_rays),
        dr.repeat(tx_pos.y, n_rays),
        dr.repeat(tx_pos.z, n_rays),
    )
    prev_weight = wt.Complex2f(dr.ones(wt.Float, n_rays), dr.zeros(wt.Float, n_rays))
    prev_polarization = vector_zero(n_rays)
    prev_prim_idx = dr.zeros(wt.UInt32, n_rays)
    normalized_reflection_material, reflection_model, reflection_model_source = (
        _reflection_trace_material_metadata(
            scene=scene,
            reflection_material=reflection_material,
            reflection_coef=reflection_coef,
            reflection_relative_permittivity=reflection_relative_permittivity,
            reflection_conductivity=reflection_conductivity,
            use_scene_materials=use_scene_materials,
    )
    )
    material_omega = _reflection_material_omega(wavelength)
    grad_sensitive_workload = _reflection_grad_sensitive_workload(
        tx_pos=tx_pos,
        scene=scene,
    )

    for bounce in range(max_reflections + 1):
        if not dr.any(active):
            break

        ray = rayd.Ray(ray_origin, ray_dir)
        if grad_sensitive_workload:
            si = scene.ray_intersect(ray, active=active)
            hit = si.is_valid() & active
            blocker_dist = dr.select(hit, si.t, wt.Float(1e10))
        else:
            with dr.suspend_grad():
                si = scene.ray_intersect(ray, active=active)
                hit = si.is_valid() & active
                blocker_dist = dr.select(hit, si.t, wt.Float(1e10))
        prim_idx_i32 = wt.Int32(si.prim_index)

        if grad_sensitive_workload:
            geom_n = dr.select(
                dr.norm(si.geo_n) > wt.Float(EPS),
                si.geo_n,
                si.n,
            )
            geom_n = geom_n / (dr.norm(geom_n) + EPS)
            si_p = si.p
            si_n = dr.select(
                dr.dot(ray_dir, geom_n) > 0.0,
                -geom_n,
                geom_n,
            )
        else:
            si_p = si.p
            si_n = si.geo_n

        if bounce == 0:
            active = hit
            if not dr.any(active):
                break

        if bounce > 0 and on_segment is not None:
            stats = on_segment(
                bounce_index=bounce - 1,
                active=active,
                hit=hit,
                blocker_dist=blocker_dist,
                ray_origin=ray_origin,
                ray_dir=ray_dir,
                prev_refl_p=prev_refl_p,
                prev_refl_n=prev_refl_n,
                prev_tx=prev_tx,
                prev_weight=prev_weight,
                prev_polarization=prev_polarization,
                prev_prim_idx=prev_prim_idx,
                has_mesh_data=has_mesh_data,
                tri_v0=tri_v0,
                tri_v1=tri_v1,
                tri_v2=tri_v2,
                tri_surface_data=tri_surface_data,
            )
            if stats is not None:
                segment_stats.append(stats)

        if not dr.any(hit):
            break

        material_inputs = resolve_surface_material(
            scene=scene,
            prim_idx=si.prim_index,
            override_material=normalized_reflection_material,
            reflection_coef=reflection_coef,
            default_eta_r=reflection_relative_permittivity,
            default_sigma=reflection_conductivity,
            valid_mask=hit,
            use_scene_materials=use_scene_materials,
        )
        safe_ray_dir = dr.select(hit, ray_dir, wt.Vector3f(1.0, 0.0, 0.0))
        safe_si_n = dr.select(hit, si_n, wt.Vector3f(0.0, 0.0, 1.0))
        bounce_weight = _bounce_reflection_weight(
            incident_dir=safe_ray_dir,
            normal=safe_si_n,
            wavelength=wavelength,
            reflection_coef=reflection_coef,
            material_inputs=material_inputs,
            tx_polarization=tx_polarization,
        )
        weight = dr.select(hit, weight * bounce_weight, weight)
        reflected_polarization = reflect_field_vector(
            polarization_vec,
            safe_ray_dir,
            safe_si_n,
            eta_r=material_inputs["eta_r"],
            sigma=material_inputs["sigma"],
            omega=material_omega,
            gain=material_inputs["gain"],
        )
        polarization_vec = vector_select(hit, reflected_polarization, polarization_vec)
        active = hit

        if has_mesh_data and bounce < max_reflections:
            reflected_image_source = reflect_point_across_plane(cumulative_image_source, si_p, si_n)
            cumulative_image_source = dr.select(hit, reflected_image_source, cumulative_image_source)
            chain_prim_history[bounce] = dr.select(hit, prim_idx_i32, chain_prim_history[bounce])
            chain_plane_point_history[bounce] = dr.select(hit, si_p, chain_plane_point_history[bounce])
            chain_plane_normal_history[bounce] = dr.select(hit, si_n, chain_plane_normal_history[bounce])
            chain_hit_point_history[bounce] = dr.select(hit, si_p, chain_hit_point_history[bounce])

            source_paths_per_bounce[bounce] = _collect_unique_reflection_paths(
                active=active,
                image_source=cumulative_image_source,
                chain_prim_history=chain_prim_history,
                chain_plane_point_history=chain_plane_point_history,
                chain_plane_normal_history=chain_plane_normal_history,
                chain_hit_point_history=chain_hit_point_history,
                chain_depth=bounce + 1,
                surface_canonical_prims=surface_canonical_prims,
            )

        prev_refl_p = dr.select(hit, si_p, prev_refl_p)
        prev_refl_n = dr.select(hit, si_n, prev_refl_n)
        prev_tx = dr.select(hit, ray_origin, prev_tx)
        prev_weight = dr.select(hit, weight, prev_weight)
        prev_polarization = vector_select(hit, polarization_vec, prev_polarization)
        prev_prim_idx = dr.select(hit, wt.UInt32(prim_idx_i32), prev_prim_idx)

        dot_dn = dr.dot(ray_dir, si_n)
        ray_dir = ray_dir - 2.0 * dot_dn * si_n
        ray_origin = si_p + ray_dir * RAY_ORIGIN_BIAS

    return {
        "source_paths_per_bounce": tuple(source_paths_per_bounce),
        "segment_stats": tuple(segment_stats),
        "ray_sampling_metadata": ray_sampling_metadata,
        "reflection_model": reflection_model,
        "reflection_model_source": reflection_model_source,
        "normalized_reflection_material": normalized_reflection_material,
    }


def _trace_reflection_paths(
    *,
    tx_pos,
    scene,
    wavelength,
    n_rays,
    max_reflections,
    mode,
    reflection_coef,
    ray_sampling,
    tx_polarization,
    reflection_relative_permittivity,
    reflection_conductivity,
    reflection_material,
    use_scene_materials,
    sampling_axis: str,
    sampling_bounds,
    sampling_plane_position: float,
    tri_data,
    on_segment=None,
):
    ray_dir, ray_sampling_metadata = _select_reflection_ray_directions(
        axis=sampling_axis,
        bounds=sampling_bounds,
        tx_pos=tx_pos,
        n_rays=n_rays,
        mode=mode,
        plane_position=sampling_plane_position,
        ray_sampling=ray_sampling,
    )
    use_rayd = _reflection_rayd_scene_handle(scene, tri_data) is not None

    if on_segment is None and use_rayd:
        return _trace_reflection_paths_rayd(
            tx_pos=tx_pos,
            scene=scene,
            wavelength=wavelength,
            n_rays=n_rays,
            max_reflections=max_reflections,
            mode=mode,
            reflection_coef=reflection_coef,
            ray_sampling=ray_sampling,
            tx_polarization=tx_polarization,
            reflection_relative_permittivity=reflection_relative_permittivity,
            reflection_conductivity=reflection_conductivity,
            reflection_material=reflection_material,
            use_scene_materials=use_scene_materials,
            sampling_axis=sampling_axis,
            sampling_bounds=sampling_bounds,
            sampling_plane_position=sampling_plane_position,
            tri_data=tri_data,
            ray_dir=ray_dir,
            ray_sampling_metadata=ray_sampling_metadata,
            on_segment=on_segment,
        )

    trace_data = _trace_reflection_paths_legacy(
        tx_pos=tx_pos,
        scene=scene,
        wavelength=wavelength,
        n_rays=n_rays,
        max_reflections=max_reflections,
        mode=mode,
        reflection_coef=reflection_coef,
        ray_sampling=ray_sampling,
        tx_polarization=tx_polarization,
        reflection_relative_permittivity=reflection_relative_permittivity,
        reflection_conductivity=reflection_conductivity,
        reflection_material=reflection_material,
        use_scene_materials=use_scene_materials,
        sampling_axis=sampling_axis,
        sampling_bounds=sampling_bounds,
        sampling_plane_position=sampling_plane_position,
        tri_data=tri_data,
        ray_dir=ray_dir,
        ray_sampling_metadata=ray_sampling_metadata,
        on_segment=on_segment,
    )
    if use_rayd:
        rayd_trace_data = _trace_reflection_paths_rayd(
            tx_pos=tx_pos,
            scene=scene,
            wavelength=wavelength,
            n_rays=n_rays,
            max_reflections=max_reflections,
            mode=mode,
            reflection_coef=reflection_coef,
            ray_sampling=ray_sampling,
            tx_polarization=tx_polarization,
            reflection_relative_permittivity=reflection_relative_permittivity,
            reflection_conductivity=reflection_conductivity,
            reflection_material=reflection_material,
            use_scene_materials=use_scene_materials,
            sampling_axis=sampling_axis,
            sampling_bounds=sampling_bounds,
            sampling_plane_position=sampling_plane_position,
            tri_data=tri_data,
            ray_dir=ray_dir,
            ray_sampling_metadata=ray_sampling_metadata,
            on_segment=on_segment,
        )
        trace_data["source_paths_per_bounce"] = rayd_trace_data["source_paths_per_bounce"]
    return trace_data


def discover_reflection_paths(
    *,
    tx_pos,
    scene,
    wavelength,
    k,
    n_rays=1000,
    max_reflections=2,
    mode="2d",
    reflection_coef=0.7,
    ray_sampling="full_sphere",
    min_ray_contribution_threshold=0.0,
    tx_polarization=(1.0, 0.0, 0.0),
    reflection_relative_permittivity=5.0,
    reflection_conductivity=0.0,
    reflection_material=None,
    use_scene_materials=False,
    sampling_axis: str = "z",
    sampling_plane_position: float | None = None,
    sampling_bounds=((-1.0, 1.0), (-1.0, 1.0)),
):
    del k, min_ray_contribution_threshold
    if scene is None or n_rays <= 0 or max_reflections <= 0:
        return build_reflection_trace_detail(
            reflection_model="materialized",
            reflection_model_source="default",
            reflection_gain=float(reflection_coef),
            reflection_material=None,
            use_scene_materials=False,
            source_paths_per_bounce=(),
            discovery_sampling={
                "plane_axis": str(sampling_axis),
                "plane_position": float(
                    _axis_coordinate(tx_pos, sampling_axis)
                    if sampling_plane_position is None
                    else sampling_plane_position
                ),
                "ray_mode": str(mode),
                "requested_ray_sampling": "circle" if mode == "2d" else str(ray_sampling),
                "selected_ray_sampling": "circle_2d" if mode == "2d" else str(ray_sampling),
            },
        )

    resolved_plane_position = (
        _axis_coordinate(tx_pos, sampling_axis)
        if sampling_plane_position is None
        else float(sampling_plane_position)
    )
    trace_data = _trace_reflection_paths(
        tx_pos=tx_pos,
        scene=scene,
        wavelength=wavelength,
        n_rays=n_rays,
        max_reflections=max_reflections,
        mode=mode,
        reflection_coef=reflection_coef,
        ray_sampling=ray_sampling,
        tx_polarization=tx_polarization,
        reflection_relative_permittivity=reflection_relative_permittivity,
        reflection_conductivity=reflection_conductivity,
        reflection_material=reflection_material,
        use_scene_materials=use_scene_materials,
        sampling_axis=str(sampling_axis),
        sampling_bounds=sampling_bounds,
        sampling_plane_position=resolved_plane_position,
        tri_data=None if scene is None else scene.tri_data_gpu,
        on_segment=None,
    )
    ray_sampling_metadata = trace_data["ray_sampling_metadata"]
    return build_reflection_trace_detail(
        reflection_model=trace_data["reflection_model"],
        reflection_model_source=trace_data["reflection_model_source"],
        reflection_gain=float(reflection_coef),
        reflection_material=_reflection_detail_material_override(
            reflection_model_source=trace_data["reflection_model_source"],
            reflection_material=trace_data["normalized_reflection_material"],
        ),
        use_scene_materials=bool(use_scene_materials),
        source_paths_per_bounce=trace_data["source_paths_per_bounce"],
        discovery_sampling={
            "plane_axis": str(sampling_axis),
            "plane_position": float(resolved_plane_position),
            "ray_mode": str(mode),
            "requested_ray_sampling": ray_sampling_metadata["requested_ray_sampling"],
            "selected_ray_sampling": ray_sampling_metadata["selected_ray_sampling"],
            "monitor_plane_distance_to_tx": float(ray_sampling_metadata["monitor_plane_distance_to_tx"]),
            "near_plane_sampling_threshold": float(ray_sampling_metadata["near_plane_sampling_threshold"]),
        },
    )


def _assemble_reflection_outputs(
    *,
    n_rx: int,
    grid_axis: str,
    plane_position: float,
    tx_pos,
    mode,
    min_ray_contribution_threshold: float,
    ray_sampling_metadata,
    polarization_per_bounce,
    source_paths_per_bounce,
    reflection_model: str,
    reflection_model_source: str,
    reflection_gain: float,
    reflection_material,
    use_scene_materials: bool,
    active_rx_polarization,
    rx_polarization,
    tangential_axes: tuple[str, str],
    dda_backend: str,
    requested_backend: str,
    resolved_backend: str,
    implementation: str,
    native_ad_mode: str | None,
    dda_stats_per_bounce,
    policy_metadata=None,
    return_per_bounce: bool = False,
    include_field_payload: bool = True,
    polarization_total=None,
):
    jones_metadata = _reflection_jones_metadata(
        grid_axis,
        explicit_receiver_projection=rx_polarization is not None,
        tangential_axes=tangential_axes,
    )
    a_ref_list = []
    if return_per_bounce:
        a_ref_list = [
            scalarize_tangential_jones(
                tangential_jones(polarization_field, axis=grid_axis),
                active_rx_polarization,
                axis=grid_axis,
            )
            for polarization_field in polarization_per_bounce
        ]

    if polarization_total is None:
        polarization_total = {
            "x": _sum_complex_fields(
                (polarization_field["x"] for polarization_field in polarization_per_bounce),
                n_rx=n_rx,
            ),
            "y": _sum_complex_fields(
                (polarization_field["y"] for polarization_field in polarization_per_bounce),
                n_rx=n_rx,
            ),
            "z": _sum_complex_fields(
                (polarization_field["z"] for polarization_field in polarization_per_bounce),
                n_rx=n_rx,
            ),
        }
    a_ref_total = scalarize_tangential_jones(
        tangential_jones(polarization_total, axis=grid_axis),
        active_rx_polarization,
        axis=grid_axis,
    )

    dda_stats = {
        "backend": dda_backend,
        "requested_backend": str(requested_backend),
        "resolved_backend": str(resolved_backend),
        "implementation": str(implementation),
        "native_ad_mode": native_ad_mode,
        "plane_axis": grid_axis,
        "plane_position": float(plane_position),
        "ray_mode": str(mode),
        "requested_ray_sampling": ray_sampling_metadata["requested_ray_sampling"],
        "selected_ray_sampling": ray_sampling_metadata["selected_ray_sampling"],
        "monitor_plane_distance_to_tx": float(ray_sampling_metadata["monitor_plane_distance_to_tx"]),
        "near_plane_sampling_threshold": float(ray_sampling_metadata["near_plane_sampling_threshold"]),
        "projected_direction_norm_threshold": float(min_ray_contribution_threshold),
        "diagnostic_dt_threshold": float(_DDA_DEGENERATE_DT_THRESHOLD),
        "per_bounce": tuple(dda_stats_per_bounce),
    }
    if policy_metadata:
        dda_stats.update(policy_metadata)

    detail_payload = {
        "tx_pos": tx_pos,
        "rx_polarization": active_rx_polarization,
        "rx_polarization_source": "explicit" if rx_polarization is not None else "default_from_tx_polarization",
        "scalar_projection_rule": jones_metadata["scalar_projection_rule"],
        "reflection_scalarization": jones_metadata["reflection_scalarization"],
        "transport_basis": "path_transverse",
        "result_jones_basis": jones_metadata["result_jones_basis"],
        "result_jones_axes": jones_metadata["result_jones_axes"],
        "dda_stats": dda_stats,
    }
    if include_field_payload:
        detail_payload["polarization_field_total"] = polarization_total
        if len(polarization_per_bounce) > 0:
            detail_payload["polarization_field_per_bounce"] = tuple(polarization_per_bounce)
        detail_payload["jones_field_total"] = tangential_jones(polarization_total, axis=grid_axis)

    rd_detail = build_reflection_trace_detail(
        reflection_model=reflection_model,
        reflection_model_source=reflection_model_source,
        reflection_gain=float(reflection_gain),
        reflection_material=_reflection_detail_material_override(
            reflection_model_source=reflection_model_source,
            reflection_material=reflection_material,
        ),
        use_scene_materials=bool(use_scene_materials),
        source_paths_per_bounce=source_paths_per_bounce,
        **detail_payload,
    )
    return a_ref_total, a_ref_list, rd_detail


def _compute_reflection_field_impl(grid, rx_z, tx_pos, scene, wavelength, k,
                                    n_rays, max_reflections, mode, reflection_coef,
                                    ray_sampling,
                                    min_ray_contribution_threshold,
                                    return_per_bounce, validate_paths,
                                    tri_data, grid_data=None,
                                    receiver_tiles=None,
                                    reflection_field_backend="native",
                                    tx_polarization=(1.0, 0.0, 0.0),
                                    rx_polarization=None,
                                    reflection_relative_permittivity=5.0,
                                    reflection_conductivity=0.0,
                                    reflection_material=None,
                                    use_scene_materials=False,
                                    include_field_payload=True,
                                    prefer_epc=True):
    """
    Reflection field computation using DrJit with GPU-parallel DDA traversal.
    Uses dr.while_loop for GPU-native loop execution.

    Args:
        grid: Field object with bounds, size, pos_to_idx method
        rx_z: Receiver Z coordinate (scalar float)
        tx_pos: Transmitter position - wt.Point3f (gradient-preserving)
        scene: Scene object
        wavelength: Wavelength in meters
        k: Wave number (2*pi/lambda)
        n_rays: Number of rays to emit
        max_reflections: Max reflection bounces
        mode: '2d' (circle) or '3d' ray directions
        reflection_coef: Amplitude loss per bounce
        ray_sampling: 3D emission distribution ('auto', 'full_sphere', or
            'hemisphere'). Ignored in '2d' mode.
        return_per_bounce: Return per-bounce contributions
        validate_paths: Enable path validation (secondary check)
        tri_data: Preloaded triangle data dict with v0x,v0y,v0z,v1x,...
        grid_data: Optional preloaded grid coordinates dict with x_coords, y_coords
        scene: Scene object for reflection-prefix path collection

    Returns:
        a_ref_total: Total reflection field (DrJit Complex2f)
        a_ref_list: List of per-bounce fields
        detail: Reflection detail dict with source_paths_per_bounce
    """
    n_rx = grid.n_cells
    (x_min, x_max), (y_min, y_max) = grid.bounds
    cell_size_x, cell_size_y = grid.cell_size
    nx, ny = grid.size
    max_steps = 2 * (nx + ny)
    grid_axis, plane_position = _resolve_reflection_plane(grid, rx_z)
    receiver_tiles = resolve_receiver_tiles(
        grid=grid,
        plane_position=plane_position,
        grid_data=grid_data,
        receiver_tiles=receiver_tiles,
    )

    # Use preloaded grid coordinates if available, otherwise compute
    if receiver_tiles is not None:
        x_coords_dr = receiver_tiles.x_coords
        y_coords_dr = receiver_tiles.y_coords
    elif grid_data is not None:
        x_coords_dr = grid_data['x_coords']
        y_coords_dr = grid_data['y_coords']
    else:
        # Pure DrJit linspace
        x_step = (x_max - x_min) / (nx - 1) if nx > 1 else 0
        y_step = (y_max - y_min) / (ny - 1) if ny > 1 else 0
        idx_x = dr.arange(wt.Float, nx)
        idx_y = dr.arange(wt.Float, ny)
        x_coords_dr = wt.Float(x_min) + idx_x * wt.Float(x_step)
        y_coords_dr = wt.Float(y_min) + idx_y * wt.Float(y_step)

    has_mesh_data = tri_data is not None
    requested_backend = str(reflection_field_backend)
    use_native_grid = requested_backend == "native"
    epc_policy = _reflection_epc_policy(
        reflection_detail=None,
        grid_axis=grid_axis,
        has_mesh_data=has_mesh_data,
        scene=scene,
        tx_pos=tx_pos,
        prefer_epc=prefer_epc,
    )

    active_rx_polarization = effective_rx_polarization(rx_polarization, tx_polarization)
    if epc_policy["use_epc"]:
        trace_data = _trace_reflection_paths(
            tx_pos=tx_pos,
            scene=scene,
            wavelength=wavelength,
            n_rays=n_rays,
            max_reflections=max_reflections,
            mode=mode,
            reflection_coef=reflection_coef,
            ray_sampling=ray_sampling,
            tx_polarization=tx_polarization,
            reflection_relative_permittivity=reflection_relative_permittivity,
            reflection_conductivity=reflection_conductivity,
            reflection_material=reflection_material,
            use_scene_materials=use_scene_materials,
            sampling_axis=grid_axis,
            sampling_bounds=grid.bounds,
            sampling_plane_position=plane_position,
            tri_data=tri_data,
            on_segment=None,
        )
        source_paths_per_bounce = list(trace_data["source_paths_per_bounce"])
        ray_sampling_metadata = trace_data["ray_sampling_metadata"]
        reflection_model = trace_data["reflection_model"]
        reflection_model_source = trace_data["reflection_model_source"]
        normalized_reflection_material = trace_data["normalized_reflection_material"]
        discovery_sampling = {
            "backend": "dda_planar_grid" if grid_axis == "z" else "ray_plane_scatter",
            "plane_axis": str(grid_axis),
            "plane_position": float(plane_position),
            "ray_mode": str(mode),
            "requested_ray_sampling": ray_sampling_metadata["requested_ray_sampling"],
            "selected_ray_sampling": ray_sampling_metadata["selected_ray_sampling"],
            "monitor_plane_distance_to_tx": float(
                ray_sampling_metadata["monitor_plane_distance_to_tx"]
            ),
            "near_plane_sampling_threshold": float(
                ray_sampling_metadata["near_plane_sampling_threshold"]
            ),
            "projected_direction_norm_threshold": float(min_ray_contribution_threshold),
            "diagnostic_dt_threshold": float(_DDA_DEGENERATE_DT_THRESHOLD),
            "per_bounce": tuple(trace_data["segment_stats"]),
        }
        reflection_chain_detail = build_reflection_trace_detail(
            reflection_model=reflection_model,
            reflection_model_source=reflection_model_source,
            reflection_gain=float(reflection_coef),
            reflection_material=_reflection_detail_material_override(
                reflection_model_source=reflection_model_source,
                reflection_material=normalized_reflection_material,
            ),
            use_scene_materials=bool(use_scene_materials),
            source_paths_per_bounce=source_paths_per_bounce,
            discovery_sampling=discovery_sampling,
        )
        polarization_per_bounce = _accumulate_reflection_paths_exact(
            rx_pos=(
                receiver_tiles.receiver_positions
                if receiver_tiles is not None and receiver_tiles.receiver_positions is not None
                else grid.receiver_positions_3d(position=plane_position)
            ),
            scene=scene,
            wavelength=wavelength,
            k=k,
            source_paths_per_bounce=source_paths_per_bounce,
            reflection_detail=reflection_chain_detail,
            tx_polarization=tx_polarization,
            receiver_tiles=receiver_tiles,
        )
        a_ref_total, a_ref_list, rd_detail = _assemble_reflection_outputs(
            n_rx=n_rx,
            grid_axis=grid_axis,
            plane_position=plane_position,
            tx_pos=tx_pos,
            mode=mode,
            min_ray_contribution_threshold=min_ray_contribution_threshold,
            ray_sampling_metadata=ray_sampling_metadata,
            polarization_per_bounce=polarization_per_bounce,
            source_paths_per_bounce=tuple(source_paths_per_bounce),
            reflection_model=reflection_model,
            reflection_model_source=reflection_model_source,
            reflection_gain=float(reflection_coef),
            reflection_material=normalized_reflection_material,
            use_scene_materials=bool(use_scene_materials),
            active_rx_polarization=active_rx_polarization,
            rx_polarization=rx_polarization,
            tangential_axes=getattr(grid, "tangential_axes", ("x", "y")),
            dda_backend="epc",
            requested_backend=requested_backend,
            resolved_backend=requested_backend,
            implementation="epc",
            native_ad_mode=None,
            dda_stats_per_bounce=tuple(),
            policy_metadata=epc_policy,
        )
        rd_detail["discovery_sampling"] = dict(discovery_sampling)
        rd_detail["reflection_sampling"] = dict(discovery_sampling)
        if return_per_bounce:
            return a_ref_total, a_ref_list, rd_detail
        return a_ref_total, [], rd_detail

    if use_native_grid and not native_extension_available():
        raise RuntimeError(
            "reflection_field_backend='native' requires the witwin.channel native extension."
        )

    # Initialize result buffers (always use list for dr.while_loop compatibility)
    results_real = [dr.zeros(wt.Float, n_rx) for _ in range(max_reflections)]
    results_imag = [dr.zeros(wt.Float, n_rx) for _ in range(max_reflections)]
    results_count = [dr.zeros(wt.Float, n_rx) for _ in range(max_reflections)]
    results_pol_real_x = [dr.zeros(wt.Float, n_rx) for _ in range(max_reflections)]
    results_pol_imag_x = [dr.zeros(wt.Float, n_rx) for _ in range(max_reflections)]
    results_pol_real_y = [dr.zeros(wt.Float, n_rx) for _ in range(max_reflections)]
    results_pol_imag_y = [dr.zeros(wt.Float, n_rx) for _ in range(max_reflections)]
    results_pol_real_z = [dr.zeros(wt.Float, n_rx) for _ in range(max_reflections)]
    results_pol_imag_z = [dr.zeros(wt.Float, n_rx) for _ in range(max_reflections)]

    def _segment_accumulator(
        *,
        bounce_index,
        active,
        hit,
        blocker_dist,
        ray_origin,
        ray_dir,
        prev_refl_p,
        prev_refl_n,
        prev_tx,
        prev_weight,
        prev_polarization,
        prev_prim_idx,
        has_mesh_data,
        tri_v0,
        tri_v1,
        tri_v2,
        tri_surface_data,
    ):
        if grid_axis == "z":
            projected_direction_norm = dr.sqrt(
                dr.maximum(
                    ray_dir.x * ray_dir.x + ray_dir.y * ray_dir.y,
                    wt.Float(0.0),
                )
            )
            dt_x_diag = dr.abs(wt.Float(cell_size_x) / dr.maximum(dr.abs(ray_dir.x), wt.Float(EPS)))
            dt_y_diag = dr.abs(wt.Float(cell_size_y) / dr.maximum(dr.abs(ray_dir.y), wt.Float(EPS)))
            skipped_low_projection = active & (projected_direction_norm < wt.Float(min_ray_contribution_threshold))
            dda_active = active & ~skipped_low_projection
            degenerate_dt_both = active & (
                (dt_x_diag > wt.Float(_DDA_DEGENERATE_DT_THRESHOLD))
                & (dt_y_diag > wt.Float(_DDA_DEGENERATE_DT_THRESHOLD))
            )
            stats = {
                "bounce_index": bounce_index + 1,
                "backend": "dda_planar_grid",
                "segment_rays": _mask_count(active),
                "hit_rays": _mask_count(hit),
                "dda_candidate_rays": _mask_count(dda_active),
                "skipped_low_projection_rays": _mask_count(skipped_low_projection),
                "degenerate_projected_rays": _mask_count(degenerate_dt_both),
                "max_dt_x": _masked_max(dt_x_diag, active),
                "max_dt_y": _masked_max(dt_y_diag, active),
            }
            if dr.any(dda_active):
                if use_native_grid:
                    native_outputs = reflection_grid_native.accumulate_reflection_grid(
                        grid=grid,
                        plane_position=plane_position,
                        grid_data={"x_coords": x_coords_dr, "y_coords": y_coords_dr},
                        receiver_tiles=receiver_tiles,
                        ray_origin=ray_origin,
                        ray_dir=ray_dir,
                        active=dda_active,
                        blocker_dist=blocker_dist,
                        prev_refl_p=prev_refl_p,
                        prev_refl_n=prev_refl_n,
                        prev_tx=prev_tx,
                        prev_weight=prev_weight,
                        prev_polarization=prev_polarization,
                        prev_prim_idx=prev_prim_idx,
                        wavelength=wavelength,
                        k=k,
                        validate_paths=validate_paths,
                        tri_data=tri_data,
                    )
                    results_real[bounce_index] = results_real[bounce_index] + native_outputs[0]
                    results_imag[bounce_index] = results_imag[bounce_index] + native_outputs[1]
                    results_count[bounce_index] = results_count[bounce_index] + native_outputs[2]
                    results_pol_real_x[bounce_index] = results_pol_real_x[bounce_index] + native_outputs[3]
                    results_pol_imag_x[bounce_index] = results_pol_imag_x[bounce_index] + native_outputs[4]
                    results_pol_real_y[bounce_index] = results_pol_real_y[bounce_index] + native_outputs[5]
                    results_pol_imag_y[bounce_index] = results_pol_imag_y[bounce_index] + native_outputs[6]
                    results_pol_real_z[bounce_index] = results_pol_real_z[bounce_index] + native_outputs[7]
                    results_pol_imag_z[bounce_index] = results_pol_imag_z[bounce_index] + native_outputs[8]
                else:
                    reflection_grid_drjit.run_dda_traversal(
                        grid=grid,
                        ray_origin=ray_origin,
                        ray_dir=ray_dir,
                        active=dda_active,
                        blocker_dist=blocker_dist,
                        prev_refl_p=prev_refl_p,
                        prev_refl_n=prev_refl_n,
                        prev_tx=prev_tx,
                        prev_weight=prev_weight,
                        prev_polarization=prev_polarization,
                        prev_prim_idx=prev_prim_idx,
                        x_min=x_min,
                        x_max=x_max,
                        y_min=y_min,
                        y_max=y_max,
                        cell_size_x=cell_size_x,
                        cell_size_y=cell_size_y,
                        nx=nx,
                        max_steps=max_steps,
                        x_coords_dr=x_coords_dr,
                        y_coords_dr=y_coords_dr,
                        rx_z=plane_position,
                        wavelength=wavelength,
                        k=k,
                        validate_paths=validate_paths,
                        has_mesh_data=has_mesh_data,
                        tri_v0=tri_v0,
                        tri_v1=tri_v1,
                        tri_v2=tri_v2,
                        tri_surface_data=tri_surface_data,
                        result_real=results_real,
                        result_imag=results_imag,
                        result_count=results_count,
                        result_pol_real_x=results_pol_real_x,
                        result_pol_imag_x=results_pol_imag_x,
                        result_pol_real_y=results_pol_real_y,
                        result_pol_imag_y=results_pol_imag_y,
                        result_pol_real_z=results_pol_real_z,
                        result_pol_imag_z=results_pol_imag_z,
                        bounce_idx=bounce_index,
                    )
            return stats

        intersections = reflection_grid_drjit.prepare_plane_intersections(
            grid=grid,
            ray_origin=ray_origin,
            ray_dir=ray_dir,
            active=active,
            blocker_dist=blocker_dist,
            plane_position=plane_position,
        )
        stats = {
            "bounce_index": bounce_index + 1,
            "backend": "ray_plane_scatter",
            "segment_rays": _mask_count(active),
            "hit_rays": _mask_count(hit),
            "plane_facing_candidate_rays": _mask_count(intersections["candidate"]),
            "scatter_candidate_rays": _mask_count(intersections["valid"]),
            "parallel_plane_rays": _mask_count(intersections["parallel"]),
            "pointing_away_rays": _mask_count(intersections["points_away"]),
        }
        use_native_plane_scatter = (
            use_native_grid and not reflection_grid_drjit.chunked_scatter_override_active()
        )
        if use_native_plane_scatter:
            native_outputs = reflection_grid_native.accumulate_reflection_grid(
                grid=grid,
                plane_position=plane_position,
                grid_data={"x_coords": x_coords_dr, "y_coords": y_coords_dr},
                receiver_tiles=receiver_tiles,
                ray_origin=ray_origin,
                ray_dir=ray_dir,
                active=active,
                blocker_dist=blocker_dist,
                prev_refl_p=prev_refl_p,
                prev_refl_n=prev_refl_n,
                prev_tx=prev_tx,
                prev_weight=prev_weight,
                prev_polarization=prev_polarization,
                prev_prim_idx=prev_prim_idx,
                wavelength=wavelength,
                k=k,
                validate_paths=validate_paths,
                tri_data=tri_data,
            )
            results_real[bounce_index] = results_real[bounce_index] + native_outputs[0]
            results_imag[bounce_index] = results_imag[bounce_index] + native_outputs[1]
            results_count[bounce_index] = results_count[bounce_index] + native_outputs[2]
            results_pol_real_x[bounce_index] = results_pol_real_x[bounce_index] + native_outputs[3]
            results_pol_imag_x[bounce_index] = results_pol_imag_x[bounce_index] + native_outputs[4]
            results_pol_real_y[bounce_index] = results_pol_real_y[bounce_index] + native_outputs[5]
            results_pol_imag_y[bounce_index] = results_pol_imag_y[bounce_index] + native_outputs[6]
            results_pol_real_z[bounce_index] = results_pol_real_z[bounce_index] + native_outputs[7]
            results_pol_imag_z[bounce_index] = results_pol_imag_z[bounce_index] + native_outputs[8]
            scatter_stats = {
                "compressed_candidate_rays": stats["scatter_candidate_rays"],
                "scatter_chunks_used": 1,
                "chunked_scatter": False,
                "scatter_chunk_size": stats["scatter_candidate_rays"],
            }
        else:
            scatter_stats = reflection_grid_drjit.intersect_and_scatter(
                grid=grid,
                plane_position=plane_position,
                intersections=intersections,
                prev_refl_p=prev_refl_p,
                prev_refl_n=prev_refl_n,
                prev_tx=prev_tx,
                prev_weight=prev_weight,
                prev_polarization=prev_polarization,
                prev_prim_idx=prev_prim_idx,
                nx=nx,
                x_coords_dr=x_coords_dr,
                y_coords_dr=y_coords_dr,
                wavelength=wavelength,
                k=k,
                validate_paths=validate_paths,
                has_mesh_data=has_mesh_data,
                tri_v0=tri_v0,
                tri_v1=tri_v1,
                tri_v2=tri_v2,
                tri_surface_data=tri_surface_data,
                result_real=results_real,
                result_imag=results_imag,
                result_count=results_count,
                result_pol_real_x=results_pol_real_x,
                result_pol_imag_x=results_pol_imag_x,
                result_pol_real_y=results_pol_real_y,
                result_pol_imag_y=results_pol_imag_y,
                result_pol_real_z=results_pol_real_z,
                result_pol_imag_z=results_pol_imag_z,
                bounce_idx=bounce_index,
            )
        stats.update(scatter_stats)
        return stats

    trace_data = _trace_reflection_paths(
        tx_pos=tx_pos,
        scene=scene,
        wavelength=wavelength,
        n_rays=n_rays,
        max_reflections=max_reflections,
        mode=mode,
        reflection_coef=reflection_coef,
        ray_sampling=ray_sampling,
        tx_polarization=tx_polarization,
        reflection_relative_permittivity=reflection_relative_permittivity,
        reflection_conductivity=reflection_conductivity,
        reflection_material=reflection_material,
        use_scene_materials=use_scene_materials,
        sampling_axis=grid_axis,
        sampling_bounds=grid.bounds,
        sampling_plane_position=plane_position,
        tri_data=tri_data,
        on_segment=_segment_accumulator,
    )
    source_paths_per_bounce = list(trace_data["source_paths_per_bounce"])
    dda_stats_per_bounce = list(trace_data["segment_stats"])
    ray_sampling_metadata = trace_data["ray_sampling_metadata"]
    reflection_model = trace_data["reflection_model"]
    reflection_model_source = trace_data["reflection_model_source"]
    normalized_reflection_material = trace_data["normalized_reflection_material"]

    # Compute per-bounce receiver averages lazily so Dr.Jit can keep
    # subsequent scalarization and reduction work fused on symbolic traces.
    polarization_per_bounce = []
    polarization_total = None
    epc_policy = _reflection_epc_policy(
        reflection_detail=None,
        grid_axis=grid_axis,
        has_mesh_data=has_mesh_data,
        scene=scene,
        tx_pos=tx_pos,
        source_paths_per_bounce=source_paths_per_bounce,
        prefer_epc=prefer_epc,
    )
    use_epc_accumulation = epc_policy["use_epc"]
    resolved_backend = "native" if use_native_grid else "drjit"
    implementation = (
        "epc"
        if use_epc_accumulation
        else ("native_cuda_custom_op" if use_native_grid else "drjit_reference")
    )
    native_ad_mode = "drjit_custom_op_forward_backward" if use_native_grid and not use_epc_accumulation else None
    if use_epc_accumulation:
        rx_pos = grid.receiver_positions_3d(position=plane_position)
        reflection_chain_detail = build_reflection_trace_detail(
            reflection_model=reflection_model,
            reflection_model_source=reflection_model_source,
            reflection_gain=float(reflection_coef),
            reflection_material=_reflection_detail_material_override(
                reflection_model_source=reflection_model_source,
                reflection_material=normalized_reflection_material,
            ),
            use_scene_materials=bool(use_scene_materials),
            source_paths_per_bounce=source_paths_per_bounce,
        )
        polarization_per_bounce = _accumulate_reflection_paths_exact(
            rx_pos=rx_pos,
            scene=scene,
            wavelength=wavelength,
            k=k,
            source_paths_per_bounce=source_paths_per_bounce,
            reflection_detail=reflection_chain_detail,
            tx_polarization=tx_polarization,
            receiver_tiles=receiver_tiles,
        )
        polarization_total = {
            "x": _sum_complex_fields(
                (polarization_field["x"] for polarization_field in polarization_per_bounce),
                n_rx=n_rx,
            ),
            "y": _sum_complex_fields(
                (polarization_field["y"] for polarization_field in polarization_per_bounce),
                n_rx=n_rx,
            ),
            "z": _sum_complex_fields(
                (polarization_field["z"] for polarization_field in polarization_per_bounce),
                n_rx=n_rx,
            ),
        }
        if not return_per_bounce:
            polarization_per_bounce = []
    else:
        polarization_total = {
            "x": _sum_complex_fields(
                (
                    _complex_from_accumulators(
                        results_pol_real_x[i],
                        results_pol_imag_x[i],
                        results_count[i],
                    )
                    for i in range(max_reflections)
                ),
                n_rx=n_rx,
            ),
            "y": _sum_complex_fields(
                (
                    _complex_from_accumulators(
                        results_pol_real_y[i],
                        results_pol_imag_y[i],
                        results_count[i],
                    )
                    for i in range(max_reflections)
                ),
                n_rx=n_rx,
            ),
            "z": _sum_complex_fields(
                (
                    _complex_from_accumulators(
                        results_pol_real_z[i],
                        results_pol_imag_z[i],
                        results_count[i],
                    )
                    for i in range(max_reflections)
                ),
                n_rx=n_rx,
            ),
        }
        if return_per_bounce:
            for i in range(max_reflections):
                polarization_per_bounce.append({
                    "x": _complex_from_accumulators(
                        results_pol_real_x[i],
                        results_pol_imag_x[i],
                        results_count[i],
                    ),
                    "y": _complex_from_accumulators(
                        results_pol_real_y[i],
                        results_pol_imag_y[i],
                        results_count[i],
                    ),
                    "z": _complex_from_accumulators(
                        results_pol_real_z[i],
                        results_pol_imag_z[i],
                        results_count[i],
                    ),
                })

    a_ref_total, a_ref_list, rd_detail = _assemble_reflection_outputs(
        n_rx=n_rx,
        grid_axis=grid_axis,
        plane_position=plane_position,
        tx_pos=tx_pos,
        mode=mode,
        min_ray_contribution_threshold=min_ray_contribution_threshold,
        ray_sampling_metadata=ray_sampling_metadata,
        polarization_per_bounce=polarization_per_bounce,
        source_paths_per_bounce=tuple(source_paths_per_bounce),
        reflection_model=reflection_model,
        reflection_model_source=reflection_model_source,
        reflection_gain=float(reflection_coef),
        reflection_material=normalized_reflection_material,
        use_scene_materials=bool(use_scene_materials),
        active_rx_polarization=active_rx_polarization,
        rx_polarization=rx_polarization,
        tangential_axes=getattr(grid, "tangential_axes", ("x", "y")),
        dda_backend=(
            "dda_planar_grid"
            if grid_axis == "z"
            else "ray_plane_scatter"
        ),
        requested_backend=requested_backend,
        resolved_backend=resolved_backend,
        implementation=implementation,
        native_ad_mode=native_ad_mode,
        dda_stats_per_bounce=tuple(dda_stats_per_bounce),
        policy_metadata=epc_policy,
        return_per_bounce=return_per_bounce,
        include_field_payload=include_field_payload,
        polarization_total=polarization_total,
    )
    if return_per_bounce:
        return a_ref_total, a_ref_list, rd_detail
    return a_ref_total, [], rd_detail


def compute_reflection_field(grid, rx_z, tx_pos, scene, wavelength, k,
                             n_rays=1000, max_reflections=2,
                             mode='2d', reflection_coef=0.7,
                             ray_sampling='full_sphere',
                             min_ray_contribution_threshold=0.0,
                             reflection_field_backend="native",
                             tx_polarization=(1.0, 0.0, 0.0),
                             rx_polarization=None,
                             reflection_relative_permittivity=5.0,
                             reflection_conductivity=0.0,
                             reflection_material=None,
                             use_scene_materials=False,
                             return_per_bounce=False,
                             validate_paths=True,
                             allow_empty_scene=True,
                             grid_data=None,
                             reflection_detail=None,
                             include_field_payload=True,
                             prefer_epc=True):
    """
    Compute reflection field using Monte Carlo + Image Method.

    Args:
        grid: Field object with bounds, size, pos_to_idx method
        rx_z: Receiver Z coordinate (scalar float)
        tx_pos: Transmitter position - wt.Point3f (gradient-preserving)
        scene: Scene object
        wavelength: Wavelength in meters
        k: Wave number (2*pi/lambda)
        n_rays: Number of rays to emit (default 1000)
        max_reflections: Max reflection bounces (default 2)
        mode: '2d' (circle) or '3d' ray directions
        reflection_coef: Amplitude loss per bounce (default 0.7)
        ray_sampling: 3D emission distribution. 'auto' selects hemisphere
            sampling when the transmitter is far from the monitor plane and
            full-sphere sampling when it is close. Ignored in '2d' mode.
        min_ray_contribution_threshold: Minimum XY projected ray-direction norm
            required before a reflected 3D ray segment enters the DDA
            receiver traversal. ``0.0`` preserves the historical behavior.
        tx_polarization: Transmit polarization direction used for vector-field transport
        reflection_material: Optional explicit material dict for Fresnel-aware
            reflection coefficients with keys relative_permittivity,
            conductivity, and gain. If omitted, reflections use the default
            dielectric material or per-structure scene materials.
        use_scene_materials: When True, per-structure `Scene` materials drive
            Fresnel reflection on triangles that carry specified material
            values. This is enabled by default.
        return_per_bounce: Return per-bounce contributions
        reflection_relative_permittivity: Fallback relative permittivity used
            only when reflection_material is provided without that key
        reflection_conductivity: Fallback conductivity used only when
            reflection_material is provided without that key
        validate_paths: Enable path validation (default True)
        allow_empty_scene: Return zeros when scene is None (default True)
        grid_data: Preloaded grid coordinates dict
        reflection_detail: Optional pre-discovered reflection path payload
            from ``discover_reflection_paths(...)`` or a prior
            ``compute_reflection_field(...)`` call. When provided, receiver
            evaluation runs EPC on those paths directly instead of re-running the
            Monte Carlo discovery stage.
        include_field_payload: When False, the returned detail payload omits
            receiver-wide polarization/Jones arrays and keeps only the core
            EPC/material metadata needed by downstream builders and
            benchmark bookkeeping.
        prefer_epc: When True, fresh no-grad z-plane traces may
            reuse exact receiver EPC instead of the requested runtime
            backend. Set False to force the native or Dr.Jit accumulation path
            even for EPC-eligible fresh traces.

    Returns:
        a_ref_total: Total reflection field (DrJit Complex2f)
        a_ref_list: List of per-bounce fields (if return_per_bounce=True)
        detail: Reflection detail dict with source_paths_per_bounce
    """
    grid_axis, plane_position = _resolve_reflection_plane(grid, rx_z)
    receiver_tiles = resolve_receiver_tiles(
        grid=grid,
        plane_position=plane_position,
        grid_data=grid_data,
    )
    ray_sampling_metadata = _resolve_reflection_sampling_metadata(
        axis=grid_axis,
        bounds=grid.bounds,
        tx_pos=tx_pos,
        mode=mode,
        plane_position=plane_position,
        ray_sampling=ray_sampling,
    )

    if reflection_detail is not None:
        if scene is None:
            raise ValueError("reflection_detail EPC requires a non-empty scene.")
        detail = coerce_reflection_trace_detail(reflection_detail)
        epc_policy = _reflection_epc_policy(
            reflection_detail=detail,
            grid_axis=grid_axis,
            has_mesh_data=scene.tri_data_gpu is not None,
            scene=scene,
            tx_pos=tx_pos,
        )
        n_rx = grid.n_cells
        active_rx_polarization = effective_rx_polarization(rx_polarization, tx_polarization)
        source_paths_per_bounce = list(detail.source_paths_per_bounce[:max_reflections])
        while len(source_paths_per_bounce) < max_reflections:
            source_paths_per_bounce.append(_empty_source_path_data(chain_depth=len(source_paths_per_bounce) + 1))
        polarization_per_bounce = _accumulate_reflection_paths_exact(
            rx_pos=(
                receiver_tiles.receiver_positions
                if receiver_tiles is not None and receiver_tiles.receiver_positions is not None
                else grid.receiver_positions_3d(position=plane_position)
            ),
            scene=scene,
            wavelength=wavelength,
            k=k,
            source_paths_per_bounce=source_paths_per_bounce,
            reflection_detail=reflection_detail,
            tx_polarization=tx_polarization,
            receiver_tiles=receiver_tiles,
        )
        a_ref_total, a_ref_list, rd_detail = _assemble_reflection_outputs(
            n_rx=n_rx,
            grid_axis=grid_axis,
            plane_position=plane_position,
            tx_pos=tx_pos,
            mode=mode,
            min_ray_contribution_threshold=min_ray_contribution_threshold,
            ray_sampling_metadata=ray_sampling_metadata,
            polarization_per_bounce=polarization_per_bounce,
            source_paths_per_bounce=tuple(source_paths_per_bounce),
            reflection_model=detail.reflection_model,
            reflection_model_source=detail.reflection_model_source,
            reflection_gain=detail.reflection_gain,
            reflection_material=detail.reflection_material,
            use_scene_materials=detail.use_scene_materials,
            active_rx_polarization=active_rx_polarization,
            rx_polarization=rx_polarization,
            tangential_axes=getattr(grid, "tangential_axes", ("x", "y")),
            dda_backend="epc",
            requested_backend=str(reflection_field_backend),
            resolved_backend=str(reflection_field_backend),
            implementation="epc",
            native_ad_mode=None,
            dda_stats_per_bounce=(),
            policy_metadata=epc_policy,
            return_per_bounce=return_per_bounce,
            include_field_payload=include_field_payload,
        )
        if return_per_bounce:
            return a_ref_total, a_ref_list, rd_detail
        return a_ref_total, [], rd_detail

    if scene is None or n_rays <= 0 or max_reflections <= 0:
        if not allow_empty_scene:
            raise ValueError("scene is None; set allow_empty_scene=True to return zeros.")
        n_rx = grid.n_cells
        jones_metadata = _reflection_jones_metadata(
            grid_axis,
            explicit_receiver_projection=rx_polarization is not None,
            tangential_axes=getattr(grid, "tangential_axes", ("x", "y")),
        )
        zero_real = dr.zeros(wt.Float, n_rx)
        zero_imag = dr.zeros(wt.Float, n_rx)
        zero_field = wt.Complex2f(zero_real, zero_imag)
        zero_detail_payload = {
            "rx_polarization": effective_rx_polarization(rx_polarization, tx_polarization),
            "rx_polarization_source": "explicit" if rx_polarization is not None else "default_from_tx_polarization",
            "scalar_projection_rule": jones_metadata["scalar_projection_rule"],
            "reflection_scalarization": jones_metadata["reflection_scalarization"],
            "transport_basis": "path_transverse",
            "result_jones_basis": jones_metadata["result_jones_basis"],
            "result_jones_axes": jones_metadata["result_jones_axes"],
            "dda_stats": {
                "backend": "dda_planar_grid" if grid_axis == "z" else "ray_plane_scatter",
                "requested_backend": str(reflection_field_backend),
                "resolved_backend": str(reflection_field_backend),
                "implementation": "empty_scene",
                "native_ad_mode": None,
                "plane_axis": grid_axis,
                "plane_position": float(plane_position),
                "ray_mode": str(mode),
                "requested_ray_sampling": ray_sampling_metadata["requested_ray_sampling"],
                "selected_ray_sampling": ray_sampling_metadata["selected_ray_sampling"],
                "monitor_plane_distance_to_tx": float(ray_sampling_metadata["monitor_plane_distance_to_tx"]),
                "near_plane_sampling_threshold": float(ray_sampling_metadata["near_plane_sampling_threshold"]),
                "projected_direction_norm_threshold": float(min_ray_contribution_threshold),
                "diagnostic_dt_threshold": float(_DDA_DEGENERATE_DT_THRESHOLD),
                "per_bounce": (),
            },
        }
        if include_field_payload:
            zero_vector = vector_zero(n_rx)
            zero_detail_payload.update(
                polarization_field_total=zero_vector,
                polarization_field_per_bounce=tuple(vector_zero(n_rx) for _ in range(max_reflections))
                if return_per_bounce
                else (),
                jones_field_total=tangential_jones(zero_vector, axis=grid_axis),
            )
        zero_detail = build_reflection_trace_detail(
            reflection_model="materialized",
            reflection_model_source="default",
            reflection_gain=float(reflection_coef),
            reflection_material=None,
            use_scene_materials=False,
            source_paths_per_bounce=(),
            **zero_detail_payload,
        )
        if return_per_bounce:
            return zero_field, [zero_field] * max_reflections, zero_detail
        return zero_field, [], zero_detail

    return _compute_reflection_field_impl(
        grid=grid,
        rx_z=rx_z,
        tx_pos=tx_pos,
        scene=scene,
        wavelength=wavelength,
        k=k,
        n_rays=n_rays,
        max_reflections=max_reflections,
        mode=mode,
        reflection_coef=reflection_coef,
        ray_sampling=ray_sampling,
        min_ray_contribution_threshold=min_ray_contribution_threshold,
        reflection_field_backend=reflection_field_backend,
        tx_polarization=tx_polarization,
        rx_polarization=rx_polarization,
        reflection_relative_permittivity=reflection_relative_permittivity,
        reflection_conductivity=reflection_conductivity,
        reflection_material=reflection_material,
        return_per_bounce=return_per_bounce,
        validate_paths=validate_paths,
        tri_data=scene.tri_data_gpu,
        grid_data=grid_data,
        receiver_tiles=receiver_tiles,
        use_scene_materials=use_scene_materials,
        include_field_payload=include_field_payload,
        prefer_epc=prefer_epc,
    )
