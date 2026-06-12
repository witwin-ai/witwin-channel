"""Public reflection phase entry for the deterministic radiomap solver."""

import drjit as dr

from witwin.channel.deterministic import types as wt

from witwin.channel.core.runtime import Material, Rx, Tx, Wave
from witwin.channel.core.physics import polarization
from ..reflection import accumulation as accum, common, detail, paths
from ..reflection.paths import empty_source_path_data


def discover(*, ctx):
    """Discover reflection path topology once per solve.

    Discovery is detached and rx-independent (it depends on tx, scene, and the
    sampling frame only), so the returned detail is shared across quadrature
    sample sets and with mixed reflection-diffraction state preparation.
    """
    runtime = ctx.runtime
    spec = ctx.spec
    config = ctx.config
    effective = ctx.solver_controls["effective"]
    if (
        ctx.scene is None
        or effective["reflection_n_rays"] <= 0
        or effective["reflection_max_bounces"] <= 0
    ):
        return None
    ray_sampling = getattr(
        spec,
        "ray_sampling",
        "full_sphere" if spec.ray_mode == "3d" else "circle",
    )
    if ctx.sample_grid is not None:
        sampling_axis = ctx.sample_grid.axis
        sampling_plane_position = ctx.sample_grid.position
        sampling_bounds = ctx.sample_grid.bounds
    else:
        sampling_axis, sampling_plane_position, sampling_bounds = (
            common.sampling_frame_from_rx(runtime.rx)
        )
    return discover_paths(
        tx=runtime.tx,
        scene=ctx.scene,
        wave=runtime.wave,
        n_rays=effective["reflection_n_rays"],
        max_reflections=effective["reflection_max_bounces"],
        mode=spec.ray_mode,
        material=runtime.reflection,
        ray_sampling=ray_sampling,
        sampling_axis=sampling_axis,
        sampling_plane_position=sampling_plane_position,
        sampling_bounds=sampling_bounds,
        reflection_transition_mode=config.reflection_transition_mode,
        reflection_f_weight_boundary_radius_wavelengths=(
            config.reflection_f_weight_boundary_radius_wavelengths
        ),
        reflection_f_weight_max_edges_per_slot=config.reflection_f_weight_max_edges_per_slot,
        reflection_secondary_visibility_mode=config.reflection_secondary_visibility_mode,
    )


def trace(*, ctx, reflection_detail):
    scene = ctx.scene
    runtime = ctx.runtime
    sample_grid = ctx.sample_grid
    spec = ctx.spec
    n_rx = ctx.n_rx
    grad_preserving = ctx.grad_preserving
    ray_sampling = getattr(
        spec,
        "ray_sampling",
        "full_sphere" if spec.ray_mode == "3d" else "circle",
    )
    effective = ctx.solver_controls["effective"]
    if reflection_detail is None:
        reflection_detail = discover(ctx=ctx)
    if grad_preserving:
        _, _, reflection_detail, vector = compute_field(
            grid=sample_grid,
            rx_z=sample_grid.position,
            tx=runtime.tx,
            scene=scene,
            wave=runtime.wave,
            n_rays=effective["reflection_n_rays"],
            max_reflections=effective["reflection_max_bounces"],
            mode=spec.ray_mode,
            ray_sampling="full_sphere" if spec.ray_mode == "3d" else "circle",
            material=runtime.reflection,
            rx=runtime.rx,
            return_per_bounce=False,
            reflection_detail=reflection_detail,
        )
        return {
            "field": {"vector": vector},
            "detail": reflection_detail,
        }

    vector_coherent = accum.accumulate_vector_field(
        rx=runtime.rx,
        tx=runtime.tx,
        scene=scene,
        wave=runtime.wave,
        reflection_detail=reflection_detail,
    )
    if (
        getattr(spec, "kind", None) == "field"
        and reflection_detail is not None
        and effective["reflection_n_rays"] > 0
        and effective["reflection_max_bounces"] > 0
    ):
        setattr(reflection_detail, "epc_stats", {
            "backend": "epc",
            "implementation": "epc",
            "plane_axis": sample_grid.axis,
            "plane_position": sample_grid.position,
            "ray_mode": spec.ray_mode,
            "requested_ray_sampling": ray_sampling,
            "use_epc": True,
            "policy": "solve_field_epc",
        })
    return {
        "field": {"vector": vector_coherent},
        "detail": reflection_detail,
    }


def discover_paths(
    *,
    tx: Tx,
    scene,
    wave: Wave,
    n_rays=1000,
    max_reflections=2,
    mode="2d",
    material: Material | None = None,
    ray_sampling="full_sphere",
    min_ray_contribution_threshold=0.0,
    sampling_axis: str = "z",
    sampling_plane_position: float | None = None,
    sampling_bounds=((-1.0, 1.0), (-1.0, 1.0)),
    reflection_transition_mode: str = "hard",
    reflection_f_weight_boundary_radius_wavelengths: float = 2.0,
    reflection_f_weight_max_edges_per_slot: int = 1,
    reflection_secondary_visibility_mode: str = "hard",
):
    del min_ray_contribution_threshold
    if material is None:
        material = Material(reflection_coef=1.0)
    if scene is None or n_rays <= 0 or max_reflections <= 0:
        return detail.build_trace_detail(
            reflection_model="materialized",
            reflection_model_source="default",
            reflection_gain=material.gain_scalar,
            source_paths_per_bounce=(),
            reflection_transition_mode=reflection_transition_mode,
            reflection_f_weight_boundary_radius_wavelengths=reflection_f_weight_boundary_radius_wavelengths,
            reflection_f_weight_max_edges_per_slot=reflection_f_weight_max_edges_per_slot,
            reflection_secondary_visibility_mode=reflection_secondary_visibility_mode,
        )

    resolved_plane_position = (
        common.axis_coordinate(tx.position, sampling_axis)
        if sampling_plane_position is None
        else float(sampling_plane_position)
    )
    trace_data = paths.trace_paths(
        tx=tx,
        scene=scene,
        wave=wave,
        n_rays=n_rays,
        max_reflections=max_reflections,
        mode=mode,
        material=material,
        ray_sampling=ray_sampling,
        sampling_axis=str(sampling_axis),
        sampling_bounds=sampling_bounds,
        sampling_plane_position=resolved_plane_position,
        tri_data=None if scene is None else scene._triangle_runtime(),
    )
    return detail.build_trace_detail(
        reflection_model=trace_data["reflection_model"],
        reflection_model_source=trace_data["reflection_model_source"],
        reflection_gain=material.gain_scalar,
        source_paths_per_bounce=trace_data["source_paths_per_bounce"],
        reflection_transition_mode=reflection_transition_mode,
        reflection_f_weight_boundary_radius_wavelengths=reflection_f_weight_boundary_radius_wavelengths,
        reflection_f_weight_max_edges_per_slot=reflection_f_weight_max_edges_per_slot,
        reflection_secondary_visibility_mode=reflection_secondary_visibility_mode,
    )


def compute_field(
    grid,
    rx_z,
    tx: Tx,
    scene,
    wave: Wave,
    n_rays=1000,
    max_reflections=2,
    mode="2d",
    material: Material | None = None,
    ray_sampling="full_sphere",
    rx: Rx | None = None,
    return_per_bounce=False,
    allow_empty_scene=True,
    reflection_detail=None,
):
    if material is None:
        material = Material(reflection_coef=1.0)
    grid_axis, plane_position = common.resolve_plane(grid, rx_z)
    rx_context = Rx(
        positions=grid.receiver_positions_3d(position=plane_position),
        polarization=None if rx is None else rx.polarization,
    )

    if reflection_detail is not None:
        if scene is None:
            raise ValueError("reflection_detail EPC requires a non-empty scene.")
        td = detail.coerce_trace_detail(reflection_detail)
        n_rx = grid.n_cells
        active_rx_polarization = rx_context.effective_polarization(tx)
        source_paths_per_bounce = list(
            td.source_paths_per_bounce[:max_reflections]
        )
        while len(source_paths_per_bounce) < max_reflections:
            source_paths_per_bounce.append(
                empty_source_path_data(chain_depth=len(source_paths_per_bounce) + 1)
            )
        polarization_per_bounce = paths.accumulate_paths_exact(
            rx=rx_context,
            tx=tx,
            scene=scene,
            wave=wave,
            source_paths_per_bounce=source_paths_per_bounce,
            reflection_detail=reflection_detail,
        )
        a_ref_total, a_ref_list, rd_detail, polarization_total = accum.assemble_outputs(
            n_rx=n_rx,
            grid_axis=grid_axis,
            polarization_per_bounce=polarization_per_bounce,
            source_paths_per_bounce=tuple(source_paths_per_bounce),
            reflection_model=td.reflection_model,
            reflection_model_source=td.reflection_model_source,
            reflection_gain=td.reflection_gain,
            active_rx_polarization=active_rx_polarization,
            return_per_bounce=return_per_bounce,
            reflection_transition_mode=td.reflection_transition_mode,
            reflection_f_weight_boundary_radius_wavelengths=(
                td.reflection_f_weight_boundary_radius_wavelengths
            ),
            reflection_f_weight_max_edges_per_slot=td.reflection_f_weight_max_edges_per_slot,
            reflection_secondary_visibility_mode=td.reflection_secondary_visibility_mode,
        )
        if return_per_bounce:
            return a_ref_total, a_ref_list, rd_detail, polarization_total
        return a_ref_total, [], rd_detail, polarization_total

    if scene is None or n_rays <= 0 or max_reflections <= 0:
        if not allow_empty_scene:
            raise ValueError("scene is None; set allow_empty_scene=True to return zeros.")
        n_rx = grid.n_cells
        zero_real = dr.zeros(wt.Float, n_rx)
        zero_imag = dr.zeros(wt.Float, n_rx)
        zero_field = wt.Complex2f(zero_real, zero_imag)
        zero_vector = polarization.vector_zero(n_rx)
        zero_detail = detail.build_trace_detail(
            reflection_model="materialized",
            reflection_model_source="default",
            reflection_gain=material.gain_scalar,
            source_paths_per_bounce=(),
        )
        if return_per_bounce:
            return zero_field, [zero_field] * max_reflections, zero_detail, zero_vector
        return zero_field, [], zero_detail, zero_vector

    return accum.compute_field_impl(
        grid=grid,
        rx_z=rx_z,
        tx=tx,
        scene=scene,
        wave=wave,
        n_rays=n_rays,
        max_reflections=max_reflections,
        mode=mode,
        material=material,
        ray_sampling=ray_sampling,
        rx=rx_context,
        return_per_bounce=return_per_bounce,
        tri_data=scene._triangle_runtime(),
    )



__all__ = ["compute_field", "discover", "discover_paths", "trace"]
