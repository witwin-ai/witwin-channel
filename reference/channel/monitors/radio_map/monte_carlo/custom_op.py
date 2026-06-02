from __future__ import annotations

import math
from typing import Callable, Mapping

import drjit as dr
import witwin as wt

from ...orchestration import ResolvedTraceConfig
from .. import diagnostics as rm_diag
from ..grid import RadioMapGrid
from ..monitor import RadioMapMonitor
from ..payload import RadioMapPayload
from ....scene import Scene
from ....trace.materials import reflection_material_omega
from ....utils.drjit_ops import EvalSync
from . import ad_support as mc_ad
from . import common as mc_common
from .native import launch_sparse_coeff_jvp_into, launch_sparse_coeff_vjp_into


MC_COMPONENT_NAMES = ("los", "reflection", "diffraction")


def _component_tuple(component_power: Mapping[str, object]):
    return tuple(component_power[name] for name in MC_COMPONENT_NAMES)


def empty_material_inputs():
    return dr.zeros(wt.Float, 0), dr.zeros(wt.Float, 0)


def _detached_point3(point):
    return wt.Point3f(
        dr.detach(point.x),
        dr.detach(point.y),
        dr.detach(point.z),
    )


def scene_material_arrays(scene: Scene, config: ResolvedTraceConfig):
    if not (
        bool(config.use_scene_materials_for_reflection)
        or bool(config.use_scene_materials_for_diffraction)
    ):
        return empty_material_inputs()
    tri_data = getattr(scene, "tri_data_gpu", None)
    if not isinstance(tri_data, dict):
        return empty_material_inputs()
    eps_r = tri_data.get("material_eps_r")
    sigma_e = tri_data.get("material_sigma_e")
    if eps_r is None or sigma_e is None:
        return empty_material_inputs()
    return eps_r, sigma_e


def _apply_scene_material_arrays(scene: Scene, *, eps_r, sigma_e) -> None:
    tri_data = getattr(scene, "tri_data_gpu", None)
    if not isinstance(tri_data, dict):
        return
    n_triangles = int(tri_data.get("n_triangles", 0))
    if n_triangles <= 0:
        return
    if int(dr.width(eps_r)) != n_triangles or int(dr.width(sigma_e)) != n_triangles:
        return
    tri_data["material_eps_r"] = eps_r
    tri_data["material_sigma_e"] = sigma_e
    scene._triangle_material_data = {
        "eps_r": eps_r,
        "sigma_e": sigma_e,
        "specified": tri_data["material_specified"],
        "structure_idx": tri_data["material_structure_idx"],
        "has_specified_materials": bool(tri_data["material_has_specified_materials"]),
        "n_specified_triangles": int(tri_data["material_n_specified_triangles"]),
        "n_default_material_triangles": int(tri_data["material_n_default_material_triangles"]),
    }


def _clone_scene_with_runtime_state(
    scene: Scene,
    *,
    vertices,
    material_eps_r,
    material_sigma_e,
    use_scene_materials: bool,
):
    shadow_scene = scene.clone()
    shadow_scene.update_vertices(vertices, recompute_edges=True)
    if use_scene_materials:
        _apply_scene_material_arrays(
            shadow_scene,
            eps_r=material_eps_r,
            sigma_e=material_sigma_e,
        )
    return shadow_scene


def detached_workload(
    scene: Scene,
    tx_pos,
    config: ResolvedTraceConfig,
):
    detached_tx_pos = _detached_point3(tx_pos)
    detached_vertices = _detached_point3(scene.vertices)
    material_eps_r, material_sigma_e = scene_material_arrays(scene, config)
    detached_material_eps_r = dr.detach(material_eps_r)
    detached_material_sigma_e = dr.detach(material_sigma_e)
    detached_scene = _clone_scene_with_runtime_state(
        scene,
        vertices=detached_vertices,
        material_eps_r=detached_material_eps_r,
        material_sigma_e=detached_material_sigma_e,
        use_scene_materials=bool(
            config.use_scene_materials_for_reflection
            or config.use_scene_materials_for_diffraction
        ),
    )
    return {
        "tx_pos": detached_tx_pos,
        "scene": detached_scene,
        "material_eps_r": detached_material_eps_r,
        "material_sigma_e": detached_material_sigma_e,
    }


def _radio_map_payload_from_components(
    *,
    monitor: RadioMapMonitor,
    grid: RadioMapGrid,
    component_power: Mapping[str, object],
    metadata,
    tx_pos,
    noise_power: float,
    timing,
):
    n_rx = int(grid.n_cells)
    weighted_diagnostics = rm_diag._empty_radio_map_diagnostics(n_rx)
    for component in MC_COMPONENT_NAMES:
        weighted_diagnostics["incoherent"][component] = component_power[component]
    rm_diag._finalize_radio_map_component_totals(weighted_diagnostics)
    path_gain = weighted_diagnostics["incoherent"]["total"]
    rss = path_gain * float(monitor.tx_power)
    sinr = rm_diag._single_tx_sinr(rss, noise_power=float(noise_power))
    return RadioMapPayload(
        monitor=monitor,
        grid=grid,
        weighted_diagnostics=weighted_diagnostics,
        metadata=metadata,
        path_gain=path_gain,
        rss=rss,
        sinr=sinr,
        tx_pos=tx_pos,
        noise_power=float(noise_power),
        sample_payload_positions=(),
        timing=timing,
    )


def _scalar_grad_or_zero(value):
    return dr.zeros(wt.Float, 1) if value is None else wt.Float(value)


def _point_grad_or_zero(x_grad, y_grad, z_grad, width: int):
    zero = dr.zeros(wt.Float, width)
    return wt.Point3f(
        zero if x_grad is None else wt.Float(x_grad),
        zero if y_grad is None else wt.Float(y_grad),
        zero if z_grad is None else wt.Float(z_grad),
    )


def _material_grad_or_zero(eps_grad, sigma_grad, width: int):
    zero = dr.zeros(wt.Float, width)
    return {
        "eps_r": zero if eps_grad is None else wt.Float(eps_grad),
        "sigma_e": zero if sigma_grad is None else wt.Float(sigma_grad),
    }


def trace_with_custom_op(
    *,
    tx_pos,
    monitor: RadioMapMonitor,
    scene: Scene,
    config: ResolvedTraceConfig,
    solver_controls: Mapping[str, object],
    reflection_detail: Mapping[str, object] | None,
    radio_map_accumulation_backend: str,
    return_timing: bool,
    grad_sensitive_workload: bool,
    trace_primal_fn: Callable[..., object],
):
    detached = (
        detached_workload(scene, tx_pos, config)
        if grad_sensitive_workload
        else {"tx_pos": tx_pos, "scene": scene}
    )
    primal_state = trace_primal_fn(
        detached["tx_pos"],
        monitor,
        detached["scene"],
        config,
        solver_controls,
        reflection_detail=reflection_detail,
        radio_map_accumulation_backend=radio_map_accumulation_backend,
        return_timing=return_timing,
        resolved_ad_mode=True,
        ad_backend="outer_custom_op_native_sparse_coeff_cuda",
        loop_mode="symbolic",
        collect_ad_tapes=True,
    )
    primal_components = {
        component: dr.detach(value)
        for component, value in primal_state.component_power.items()
    }
    ray_sampling_metadata = mc_common._sionna_full_sphere_sampling_metadata(
        axis=str(primal_state.grid.axis),
        plane_position=float(primal_state.grid.position),
        tx_pos=detached["tx_pos"],
    )
    solid_angle_per_ray = mc_common._solid_angle_per_ray(
        ray_sampling_metadata,
        int(monitor.samples_per_tx),
    )
    cell_area = float(primal_state.grid.cell_size[0] * primal_state.grid.cell_size[1])
    material_omega = reflection_material_omega(config.wavelength)
    material_eps_r, material_sigma_e = scene_material_arrays(scene, config)
    if not (
        bool(config.use_scene_materials_for_reflection)
        or bool(config.use_scene_materials_for_diffraction)
    ):
        material_eps_r, material_sigma_e = empty_material_inputs()
    los_tape = (
        primal_state.los_tape
        if primal_state.los_tape is not None
        else mc_ad.empty_los_tape()
    )
    reflection_tape = (
        primal_state.reflection_tape
        if primal_state.reflection_tape is not None
        else mc_ad.empty_reflection_tape(int(solver_controls["effective"]["reflection_max_bounces"]))
    )
    diffraction_tape = (
        primal_state.diffraction_tape
        if primal_state.diffraction_tape is not None
        else mc_ad.empty_diffraction_tape()
    )
    total_length_weight = float(primal_state.diffraction_total_length_weight)
    diffraction_path_gain_scale = wt.Float(
        (float(config.wavelength) / (4.0 * math.pi)) ** 2 / float(cell_area)
    )

    n_vertices = int(dr.width(scene.vertices.x))
    n_materials = int(dr.width(material_eps_r))
    n_cells = int(primal_state.grid.n_cells)
    coeff_cache: dict[str, mc_ad.SparseCoeffBuffers] = {}
    transport_cache: dict[str, dict[str, object]] = {}

    def _ensure_coeff_buffers():
        if "los" in coeff_cache:
            return (
                coeff_cache["los"],
                coeff_cache["reflection"],
                coeff_cache["diffraction"],
            )

        coeff_cache["los"] = mc_ad.extract_los_sparse_coefficients(
            tape=los_tape,
            tx_pos=detached["tx_pos"],
            grid=primal_state.grid,
            config=config,
            solid_angle_per_ray=float(solid_angle_per_ray),
            cell_area=float(cell_area),
        )
        coeff_cache["reflection"] = mc_ad.extract_reflection_sparse_coefficients(
            tape=reflection_tape,
            scene=detached["scene"],
            tx_pos=detached["tx_pos"],
            grid=primal_state.grid,
            config=config,
            solid_angle_per_ray=float(solid_angle_per_ray),
            cell_area=float(cell_area),
            material_omega=material_omega,
        )
        coeff_cache["diffraction"] = mc_ad.extract_diffraction_sparse_coefficients(
            tape=diffraction_tape,
            scene=detached["scene"],
            tx_pos=detached["tx_pos"],
            grid=primal_state.grid,
            config=config,
            diffraction_path_gain_scale=diffraction_path_gain_scale,
            total_length_weight=float(total_length_weight),
        )
        return (
            coeff_cache["los"],
            coeff_cache["reflection"],
            coeff_cache["diffraction"],
        )

    def _ensure_tx_transport_basis_maps():
        if "los" in transport_cache:
            return transport_cache["los"], transport_cache["reflection"]
        transport_cache["los"] = mc_ad.los_tx_transport_basis_maps(
            tape=los_tape,
            scene=detached["scene"],
            tx_pos=detached["tx_pos"],
            grid=primal_state.grid,
            config=config,
            solid_angle_per_ray=float(solid_angle_per_ray),
            cell_area=float(cell_area),
        )
        transport_cache["reflection"] = mc_ad.reflection_tx_transport_basis_maps(
            tape=reflection_tape,
            scene=detached["scene"],
            tx_pos=detached["tx_pos"],
            grid=primal_state.grid,
            config=config,
            solid_angle_per_ray=float(solid_angle_per_ray),
            cell_area=float(cell_area),
            material_omega=material_omega,
        )
        return transport_cache["los"], transport_cache["reflection"]

    class _SingleSolverAD(dr.CustomOp):
        def eval(self, tx_x, tx_y, tx_z, vertices_x, vertices_y, vertices_z, material_eps_r, material_sigma_e):
            self.tx_x = tx_x
            self.tx_y = tx_y
            self.tx_z = tx_z
            self.vertices_x = vertices_x
            self.vertices_y = vertices_y
            self.vertices_z = vertices_z
            self.material_eps_r = material_eps_r
            self.material_sigma_e = material_sigma_e
            return _component_tuple(primal_components)

        def forward(self):
            los_coeff_buffers, reflection_coeff_buffers, diffraction_coeff_buffers = (
                _ensure_coeff_buffers()
            )
            tx_tangent = wt.Point3f(
                _scalar_grad_or_zero(self.grad_in("tx_x")),
                _scalar_grad_or_zero(self.grad_in("tx_y")),
                _scalar_grad_or_zero(self.grad_in("tx_z")),
            )
            zero_tx_tangent = wt.Point3f(
                dr.zeros(wt.Float, 1),
                dr.zeros(wt.Float, 1),
                dr.zeros(wt.Float, 1),
            )
            los_transport_basis, reflection_transport_basis = _ensure_tx_transport_basis_maps()
            vertex_tangent = _point_grad_or_zero(
                self.grad_in("vertices_x"),
                self.grad_in("vertices_y"),
                self.grad_in("vertices_z"),
                n_vertices,
            )
            material_tangent = _material_grad_or_zero(
                self.grad_in("material_eps_r"),
                self.grad_in("material_sigma_e"),
                n_materials,
            )
            los_grad = launch_sparse_coeff_jvp_into(
                buffers=los_coeff_buffers,
                tx_tangent=zero_tx_tangent,
                vertex_tangent=vertex_tangent,
                material_tangent=material_tangent,
                out_size=n_cells,
            )
            los_grad = (
                los_grad
                + los_transport_basis["x"] * tx_tangent.x
                + los_transport_basis["y"] * tx_tangent.y
                + los_transport_basis["z"] * tx_tangent.z
            )
            reflection_grad = launch_sparse_coeff_jvp_into(
                buffers=reflection_coeff_buffers,
                tx_tangent=zero_tx_tangent,
                vertex_tangent=vertex_tangent,
                material_tangent=material_tangent,
                out_size=n_cells,
            )
            reflection_grad = (
                reflection_grad
                + reflection_transport_basis["x"] * tx_tangent.x
                + reflection_transport_basis["y"] * tx_tangent.y
                + reflection_transport_basis["z"] * tx_tangent.z
            )
            diffraction_grad = launch_sparse_coeff_jvp_into(
                buffers=diffraction_coeff_buffers,
                tx_tangent=tx_tangent,
                vertex_tangent=vertex_tangent,
                material_tangent=material_tangent,
                out_size=n_cells,
            )
            self.set_grad_out((los_grad, reflection_grad, diffraction_grad))

        def backward(self):
            los_coeff_buffers, reflection_coeff_buffers, diffraction_coeff_buffers = (
                _ensure_coeff_buffers()
            )
            upstream_los, upstream_reflection, upstream_diffraction = self.grad_out()
            upstream_los = dr.zeros(wt.Float, n_cells) if upstream_los is None else wt.Float(upstream_los)
            upstream_reflection = (
                dr.zeros(wt.Float, n_cells)
                if upstream_reflection is None
                else wt.Float(upstream_reflection)
            )
            upstream_diffraction = (
                dr.zeros(wt.Float, n_cells)
                if upstream_diffraction is None
                else wt.Float(upstream_diffraction)
            )
            los_transport_basis, reflection_transport_basis = _ensure_tx_transport_basis_maps()
            tx_grad_x = dr.zeros(wt.Float, 1)
            tx_grad_y = dr.zeros(wt.Float, 1)
            tx_grad_z = dr.zeros(wt.Float, 1)
            vertex_grad_x = dr.zeros(wt.Float, n_vertices)
            vertex_grad_y = dr.zeros(wt.Float, n_vertices)
            vertex_grad_z = dr.zeros(wt.Float, n_vertices)
            material_grad_eps = dr.zeros(wt.Float, n_materials)
            material_grad_sigma = dr.zeros(wt.Float, n_materials)

            for buffers, upstream, include_tx_from_sparse in (
                (los_coeff_buffers, upstream_los, False),
                (reflection_coeff_buffers, upstream_reflection, False),
                (diffraction_coeff_buffers, upstream_diffraction, True),
            ):
                (
                    tx_lane_grad_x,
                    tx_lane_grad_y,
                    tx_lane_grad_z,
                    vertex_component_grad_x,
                    vertex_component_grad_y,
                    vertex_component_grad_z,
                    material_component_grad_eps,
                    material_component_grad_sigma,
                ) = launch_sparse_coeff_vjp_into(
                    buffers=buffers,
                    upstream_component=upstream,
                    n_vertices=n_vertices,
                    n_materials=n_materials,
                )
                if include_tx_from_sparse:
                    tx_grad_x = tx_grad_x + dr.sum(tx_lane_grad_x)
                    tx_grad_y = tx_grad_y + dr.sum(tx_lane_grad_y)
                    tx_grad_z = tx_grad_z + dr.sum(tx_lane_grad_z)
                vertex_grad_x = vertex_grad_x + vertex_component_grad_x
                vertex_grad_y = vertex_grad_y + vertex_component_grad_y
                vertex_grad_z = vertex_grad_z + vertex_component_grad_z
                material_grad_eps = material_grad_eps + material_component_grad_eps
                material_grad_sigma = material_grad_sigma + material_component_grad_sigma

            tx_grad_x = tx_grad_x + dr.sum(upstream_los * los_transport_basis["x"])
            tx_grad_y = tx_grad_y + dr.sum(upstream_los * los_transport_basis["y"])
            tx_grad_z = tx_grad_z + dr.sum(upstream_los * los_transport_basis["z"])
            tx_grad_x = tx_grad_x + dr.sum(upstream_reflection * reflection_transport_basis["x"])
            tx_grad_y = tx_grad_y + dr.sum(upstream_reflection * reflection_transport_basis["y"])
            tx_grad_z = tx_grad_z + dr.sum(upstream_reflection * reflection_transport_basis["z"])
            self.set_grad_in("tx_x", tx_grad_x)
            self.set_grad_in("tx_y", tx_grad_y)
            self.set_grad_in("tx_z", tx_grad_z)
            self.set_grad_in("vertices_x", vertex_grad_x)
            self.set_grad_in("vertices_y", vertex_grad_y)
            self.set_grad_in("vertices_z", vertex_grad_z)
            self.set_grad_in("material_eps_r", material_grad_eps)
            self.set_grad_in("material_sigma_e", material_grad_sigma)

        def name(self):
            return "RadioMapMonteCarloNativeSparseCoeffAD"

    los_power, reflection_power, diffraction_power = dr.custom(
        _SingleSolverAD,
        tx_x=tx_pos.x,
        tx_y=tx_pos.y,
        tx_z=tx_pos.z,
        vertices_x=scene.vertices.x,
        vertices_y=scene.vertices.y,
        vertices_z=scene.vertices.z,
        material_eps_r=material_eps_r,
        material_sigma_e=material_sigma_e,
    )
    payload = _radio_map_payload_from_components(
        monitor=monitor,
        grid=primal_state.grid,
        component_power={
            "los": los_power,
            "reflection": reflection_power,
            "diffraction": diffraction_power,
        },
        metadata=primal_state.metadata,
        tx_pos=tx_pos,
        noise_power=primal_state.noise_power,
        timing=primal_state.timing,
    )
    return payload, primal_state.reflection_detail


__all__ = [
    "MC_COMPONENT_NAMES",
    "detached_workload",
    "empty_material_inputs",
    "scene_material_arrays",
    "trace_with_custom_op",
]
