from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Mapping

import drjit as dr
from witwin.channel.core.scene import Scene
from witwin.channel.montecarlo import types as wt
from witwin.channel.core.grid import Grid
from ..config import Config, ResolvedTraceConfig
from witwin.channel.core.results import RadioMapResult
from ..sampler import Sampler
from ..trace import ad_support as mc_ad
from ..trace.diffraction import DiffractionTape
from ..trace.diffraction_ad import DiffractionAD
from ..trace.los import LosAD, LosTape
from ..trace.reflection import ReflectionAD, ReflectionTape
from ..kernels.sparse_coeff import SparseCoeffKernel
from ..kernels.transport_vertex import TransportVertexKernel
from witwin.channel.core.numerics.arrays import scalar
from witwin.channel.core.physics.wave_math import material_angular_frequency
from witwin.channel.core.runtime import (
    point_grad_enabled,
    scene_geometry_grad_enabled,
    scene_material_grad_enabled,
)


def grad_sensitive(
    config: ResolvedTraceConfig,
    *,
    tx_pos,
    scene: Scene,
) -> bool:
    del config
    if point_grad_enabled(tx_pos):
        return True
    if scene_geometry_grad_enabled(scene):
        return True
    if scene_material_grad_enabled(scene):
        return True
    return False


# ---------------------------------------------------------------------------
# AD solver context
# ---------------------------------------------------------------------------


@dataclass
class ADContext:
    """Store shared state for the Monte Carlo radiomap AD custom op."""
    primal_components: dict
    n_cells: int
    n_vertices: int
    n_materials: int
    diff_gain_scale: object
    total_length_weight: float
    detached: dict
    grid: Grid
    config: ResolvedTraceConfig
    los_tape: object
    reflection_tape: object
    diffraction_tape: object
    material_omega: object
    solid_angle_per_ray: float
    cell_area: float
    coeff_cache: dict = field(default_factory=dict)
    transport_cache: dict = field(default_factory=dict)

    @staticmethod
    def ensure_coeff_buffers(ctx: ADContext):
        """Lazily compute and cache sparse coefficient buffers for all components."""
        if "los" in ctx.coeff_cache:
            return (
                ctx.coeff_cache["los"],
                ctx.coeff_cache["reflection"],
                ctx.coeff_cache["diffraction"],
            )
        ctx.coeff_cache["los"] = LosAD.sparse_coeffs(
            tape=ctx.los_tape,
            tx_pos=ctx.detached["tx_pos"],
            grid=ctx.grid,
            config=ctx.config,
            solid_angle_per_ray=ctx.solid_angle_per_ray,
            cell_area=ctx.cell_area,
        )
        ctx.coeff_cache["reflection"] = ReflectionAD.sparse_coeffs(
            tape=ctx.reflection_tape,
            scene=ctx.detached["scene"],
            tx_pos=ctx.detached["tx_pos"],
            grid=ctx.grid,
            config=ctx.config,
            solid_angle_per_ray=ctx.solid_angle_per_ray,
            cell_area=ctx.cell_area,
            material_omega=ctx.material_omega,
        )
        ctx.coeff_cache["diffraction"] = DiffractionAD.sparse_coeffs(
            tape=ctx.diffraction_tape,
            scene=ctx.detached["scene"],
            tx_pos=ctx.detached["tx_pos"],
            grid=ctx.grid,
            config=ctx.config,
            diff_gain_scale=ctx.diff_gain_scale,
            total_length_weight=ctx.total_length_weight,
        )
        return (
            ctx.coeff_cache["los"],
            ctx.coeff_cache["reflection"],
            ctx.coeff_cache["diffraction"],
        )

    @staticmethod
    def ensure_tx_transport(ctx: ADContext):
        """Lazily compute and cache TX transport basis maps."""
        if "los" in ctx.transport_cache:
            return ctx.transport_cache["los"], ctx.transport_cache["reflection"]
        ctx.transport_cache["los"] = LosAD.transport_maps(
            tape=ctx.los_tape,
            scene=ctx.detached["scene"],
            tx_pos=ctx.detached["tx_pos"],
            grid=ctx.grid,
            config=ctx.config,
            solid_angle_per_ray=ctx.solid_angle_per_ray,
            cell_area=ctx.cell_area,
        )
        ctx.transport_cache["reflection"] = ReflectionAD.tx_basis_maps(
            tape=ctx.reflection_tape,
            scene=ctx.detached["scene"],
            tx_pos=ctx.detached["tx_pos"],
            grid=ctx.grid,
            config=ctx.config,
            solid_angle_per_ray=ctx.solid_angle_per_ray,
            cell_area=ctx.cell_area,
            material_omega=ctx.material_omega,
        )
        return ctx.transport_cache["los"], ctx.transport_cache["reflection"]

    @staticmethod
    def ensure_transport_coeff_buffers(ctx: ADContext):
        """Lazily compute and cache reflection/diffraction transport sparse coefficients."""
        if "reflection_transport" in ctx.coeff_cache:
            return (
                ctx.coeff_cache["reflection_transport"],
                ctx.coeff_cache["diffraction_transport"],
            )
        ctx.coeff_cache["reflection_transport"] = ReflectionAD.transport_vertex_coeffs(
            tape=ctx.reflection_tape,
            scene=ctx.detached["scene"],
            tx_pos=ctx.detached["tx_pos"],
            grid=ctx.grid,
            config=ctx.config,
            solid_angle_per_ray=ctx.solid_angle_per_ray,
            cell_area=ctx.cell_area,
            material_omega=ctx.material_omega,
        )
        ctx.coeff_cache["diffraction_transport"] = DiffractionAD.transport_vertex_coeffs(
            tape=ctx.diffraction_tape,
            scene=ctx.detached["scene"],
            tx_pos=ctx.detached["tx_pos"],
            grid=ctx.grid,
            config=ctx.config,
            diff_gain_scale=ctx.diff_gain_scale,
            total_length_weight=ctx.total_length_weight,
        )
        return (
            ctx.coeff_cache["reflection_transport"],
            ctx.coeff_cache["diffraction_transport"],
        )

    @staticmethod
    def make_solver_ad(ctx: ADContext):
        """Create a DrJit CustomOp that captures the solver AD context."""

        class _SingleSolverAD(dr.CustomOp):
            def eval(self, tx_x, tx_y, tx_z, vertices_x, vertices_y, vertices_z, material_eps_r, material_mu_r, material_sigma_e):
                self.tx_x = tx_x
                self.tx_y = tx_y
                self.tx_z = tx_z
                self.vertices_x = vertices_x
                self.vertices_y = vertices_y
                self.vertices_z = vertices_z
                self.material_eps_r = material_eps_r
                self.material_mu_r = material_mu_r
                self.material_sigma_e = material_sigma_e
                return BasicIntegratorAD.component_tuple(ctx.primal_components)

            def forward(self):
                los_buf, refl_buf, diff_buf = ADContext.ensure_coeff_buffers(ctx)
                refl_transport_buf, diff_transport_buf = ADContext.ensure_transport_coeff_buffers(ctx)
                tx_tangent = wt.Point3f(
                    BasicIntegratorAD.scalar_grad_or_zero(self.grad_in("tx_x")),
                    BasicIntegratorAD.scalar_grad_or_zero(self.grad_in("tx_y")),
                    BasicIntegratorAD.scalar_grad_or_zero(self.grad_in("tx_z")),
                )
                los_transport, refl_transport = ADContext.ensure_tx_transport(ctx)
                vertex_tangent = BasicIntegratorAD.point_grad_or_zero(
                    self.grad_in("vertices_x"),
                    self.grad_in("vertices_y"),
                    self.grad_in("vertices_z"),
                    ctx.n_vertices,
                )
                material_tangent = BasicIntegratorAD.material_grad_or_zero(
                    self.grad_in("material_eps_r"),
                    self.grad_in("material_mu_r"),
                    self.grad_in("material_sigma_e"),
                    ctx.n_materials,
                )
                los_grad = SparseCoeffKernel.launch_jvp_into(
                    buffers=los_buf,
                    tx_tangent=tx_tangent,
                    vertex_tangent=vertex_tangent,
                    material_tangent=material_tangent,
                    out_size=ctx.n_cells,
                )
                los_grad = (
                    los_grad
                    + los_transport["x"] * tx_tangent.x
                    + los_transport["y"] * tx_tangent.y
                    + los_transport["z"] * tx_tangent.z
                )
                reflection_grad = SparseCoeffKernel.launch_jvp_into(
                    buffers=refl_buf,
                    tx_tangent=tx_tangent,
                    vertex_tangent=vertex_tangent,
                    material_tangent=material_tangent,
                    out_size=ctx.n_cells,
                )
                reflection_grad = (
                    reflection_grad
                    + TransportVertexKernel.launch_jvp_into(
                        buffers=refl_transport_buf,
                        vertex_tangent=vertex_tangent,
                        out_size=ctx.n_cells,
                        bounds=ctx.grid.bounds,
                        cell_size=ctx.grid.cell_size,
                        grid_shape=ctx.grid.grid_shape,
                    )
                    + refl_transport["x"] * tx_tangent.x
                    + refl_transport["y"] * tx_tangent.y
                    + refl_transport["z"] * tx_tangent.z
                )
                diffraction_grad = SparseCoeffKernel.launch_jvp_into(
                    buffers=diff_buf,
                    tx_tangent=tx_tangent,
                    vertex_tangent=vertex_tangent,
                    material_tangent=material_tangent,
                    out_size=ctx.n_cells,
                )
                diffraction_grad = diffraction_grad + TransportVertexKernel.launch_jvp_into(
                    buffers=diff_transport_buf,
                    vertex_tangent=vertex_tangent,
                    out_size=ctx.n_cells,
                    bounds=ctx.grid.bounds,
                    cell_size=ctx.grid.cell_size,
                    grid_shape=ctx.grid.grid_shape,
                )
                self.set_grad_out((los_grad, reflection_grad, diffraction_grad))

            def backward(self):
                los_buf, refl_buf, diff_buf = ADContext.ensure_coeff_buffers(ctx)
                refl_transport_buf, diff_transport_buf = ADContext.ensure_transport_coeff_buffers(ctx)
                upstream_los, upstream_refl, upstream_diff = self.grad_out()
                upstream_los = dr.zeros(wt.Float, ctx.n_cells) if upstream_los is None else wt.Float(upstream_los)
                upstream_refl = dr.zeros(wt.Float, ctx.n_cells) if upstream_refl is None else wt.Float(upstream_refl)
                upstream_diff = dr.zeros(wt.Float, ctx.n_cells) if upstream_diff is None else wt.Float(upstream_diff)
                los_transport, refl_transport = ADContext.ensure_tx_transport(ctx)
                tx_grad_x = dr.zeros(wt.Float, 1)
                tx_grad_y = dr.zeros(wt.Float, 1)
                tx_grad_z = dr.zeros(wt.Float, 1)
                vertex_grad_x = dr.zeros(wt.Float, ctx.n_vertices)
                vertex_grad_y = dr.zeros(wt.Float, ctx.n_vertices)
                vertex_grad_z = dr.zeros(wt.Float, ctx.n_vertices)
                material_grad_eps = dr.zeros(wt.Float, ctx.n_materials)
                material_grad_mu = dr.zeros(wt.Float, ctx.n_materials)
                material_grad_sigma = dr.zeros(wt.Float, ctx.n_materials)

                for buffers, upstream, include_tx in (
                    (los_buf, upstream_los, True),
                    (refl_buf, upstream_refl, True),
                    (diff_buf, upstream_diff, True),
                ):
                    (
                        tx_gx,
                        tx_gy,
                        tx_gz,
                        vtx_gx,
                        vtx_gy,
                        vtx_gz,
                        mat_ge,
                        mat_gs,
                    ) = SparseCoeffKernel.launch_vjp_into(
                        buffers=buffers,
                        upstream_component=upstream,
                        n_vertices=ctx.n_vertices,
                        n_materials=ctx.n_materials,
                    )
                    if include_tx:
                        tx_grad_x = tx_grad_x + dr.sum(tx_gx)
                        tx_grad_y = tx_grad_y + dr.sum(tx_gy)
                        tx_grad_z = tx_grad_z + dr.sum(tx_gz)
                    vertex_grad_x = vertex_grad_x + vtx_gx
                    vertex_grad_y = vertex_grad_y + vtx_gy
                    vertex_grad_z = vertex_grad_z + vtx_gz
                    material_grad_eps = material_grad_eps + mat_ge
                    material_grad_sigma = material_grad_sigma + mat_gs

                refl_vtx = TransportVertexKernel.launch_vjp_into(
                    buffers=refl_transport_buf,
                    upstream_component=upstream_refl,
                    n_vertices=ctx.n_vertices,
                    bounds=ctx.grid.bounds,
                    cell_size=ctx.grid.cell_size,
                    grid_shape=ctx.grid.grid_shape,
                )
                vertex_grad_x = vertex_grad_x + refl_vtx.x
                vertex_grad_y = vertex_grad_y + refl_vtx.y
                vertex_grad_z = vertex_grad_z + refl_vtx.z
                diff_vtx = TransportVertexKernel.launch_vjp_into(
                    buffers=diff_transport_buf,
                    upstream_component=upstream_diff,
                    n_vertices=ctx.n_vertices,
                    bounds=ctx.grid.bounds,
                    cell_size=ctx.grid.cell_size,
                    grid_shape=ctx.grid.grid_shape,
                )
                vertex_grad_x = vertex_grad_x + diff_vtx.x
                vertex_grad_y = vertex_grad_y + diff_vtx.y
                vertex_grad_z = vertex_grad_z + diff_vtx.z

                tx_grad_x = tx_grad_x + dr.sum(upstream_los * los_transport["x"])
                tx_grad_y = tx_grad_y + dr.sum(upstream_los * los_transport["y"])
                tx_grad_z = tx_grad_z + dr.sum(upstream_los * los_transport["z"])
                tx_grad_x = tx_grad_x + dr.sum(upstream_refl * refl_transport["x"])
                tx_grad_y = tx_grad_y + dr.sum(upstream_refl * refl_transport["y"])
                tx_grad_z = tx_grad_z + dr.sum(upstream_refl * refl_transport["z"])
                self.set_grad_in("tx_x", tx_grad_x)
                self.set_grad_in("tx_y", tx_grad_y)
                self.set_grad_in("tx_z", tx_grad_z)
                self.set_grad_in("vertices_x", vertex_grad_x)
                self.set_grad_in("vertices_y", vertex_grad_y)
                self.set_grad_in("vertices_z", vertex_grad_z)
                self.set_grad_in("material_eps_r", material_grad_eps)
                self.set_grad_in("material_mu_r", material_grad_mu)
                self.set_grad_in("material_sigma_e", material_grad_sigma)

            def name(self):
                return "RadioMapMonteCarloNativeSparseAndTransportCoeffAD"

        return _SingleSolverAD



class BasicIntegratorAD:
    """Custom-op AD wrapper for the basic Monte Carlo integrator."""

    COMPONENT_NAMES = ("los", "reflection", "diffraction")

    @staticmethod
    def component_tuple(component_power: Mapping[str, object]):
        return tuple(component_power[name] for name in BasicIntegratorAD.COMPONENT_NAMES)

    @staticmethod
    def empty_material_inputs():
        return dr.zeros(wt.Float, 0), dr.zeros(wt.Float, 0), dr.zeros(wt.Float, 0)

    @staticmethod
    def detached_point3(point):
        return wt.Point3f(
            dr.detach(point.x),
            dr.detach(point.y),
            dr.detach(point.z),
        )

    @staticmethod
    def scene_materials(scene: Scene, config: ResolvedTraceConfig):
        del config
        tri_data = scene._triangle_runtime()
        if not isinstance(tri_data, dict):
            return BasicIntegratorAD.empty_material_inputs()
        eps_r = tri_data.get("material_eps_r")
        mu_r = tri_data.get("material_mu_r")
        sigma_e = tri_data.get("material_sigma_e")
        if eps_r is None or mu_r is None or sigma_e is None:
            return BasicIntegratorAD.empty_material_inputs()
        return eps_r, mu_r, sigma_e

    @staticmethod
    def apply_scene_materials(scene: Scene, *, eps_r, mu_r, sigma_e) -> None:
        tri_data = scene._triangle_runtime()
        if not isinstance(tri_data, dict):
            return
        n_triangles = int(tri_data.get("n_triangles", 0))
        if n_triangles <= 0:
            return
        if int(dr.width(eps_r)) != n_triangles or int(dr.width(mu_r)) != n_triangles or int(dr.width(sigma_e)) != n_triangles:
            return
        scene._set_triangle_material_runtime(
            eps_r=eps_r,
            mu_r=mu_r,
            sigma_e=sigma_e,
            specified=tri_data.get("material_specified"),
            structure_idx=tri_data.get("material_structure_idx"),
        )

    @staticmethod
    def clone_scene(
        scene: Scene,
        *,
        vertices,
        material_eps_r,
        material_mu_r,
        material_sigma_e,
    ):
        shadow_scene = scene.clone()
        shadow_scene.update_vertices(vertices, recompute_edges=True)
        BasicIntegratorAD.apply_scene_materials(
            shadow_scene,
            eps_r=material_eps_r,
            mu_r=material_mu_r,
            sigma_e=material_sigma_e,
        )
        return shadow_scene

    @staticmethod
    def detached_workload(
        scene: Scene,
        tx_pos,
        config: ResolvedTraceConfig,
        ):
        detached_tx_pos = BasicIntegratorAD.detached_point3(tx_pos)
        detached_vertices = BasicIntegratorAD.detached_point3(scene._merged_vertices())
        material_eps_r, material_mu_r, material_sigma_e = BasicIntegratorAD.scene_materials(scene, config)
        detached_material_eps_r = dr.detach(material_eps_r)
        detached_material_mu_r = dr.detach(material_mu_r)
        detached_material_sigma_e = dr.detach(material_sigma_e)
        detached_scene = BasicIntegratorAD.clone_scene(
            scene,
            vertices=detached_vertices,
            material_eps_r=detached_material_eps_r,
            material_mu_r=detached_material_mu_r,
            material_sigma_e=detached_material_sigma_e,
        )
        return {
            "tx_pos": detached_tx_pos,
            "scene": detached_scene,
            "material_eps_r": detached_material_eps_r,
            "material_mu_r": detached_material_mu_r,
            "material_sigma_e": detached_material_sigma_e,
        }

    @staticmethod
    def result_from_components(
        *,
        grid: Grid,
        tx_power: float,
        component_power: Mapping[str, object],
        metadata,
        tx_pos,
        noise_power: float,
        timing,
        shadow_boundary_mode: str,
        filtering=None,
    ) -> RadioMapResult:
        from .basic import (
            EXTRA_COMPONENT_KEYS,
            _empty_radio_map,
            _single_tx_sinr,
            build_result,
            finalize_weighted_diagnostics,
        )

        n_rx = int(grid.n_cells)
        weighted_diagnostics = _empty_radio_map(n_rx)
        for component in BasicIntegratorAD.COMPONENT_NAMES:
            weighted_diagnostics["incoherent"][component] = component_power[component]
        for component in EXTRA_COMPONENT_KEYS:
            if component in component_power:
                weighted_diagnostics["incoherent"][component] = component_power[component]
        finalize_weighted_diagnostics(
            weighted_diagnostics,
            shadow_boundary_mode=shadow_boundary_mode,
            grid=grid,
            filtering=filtering,
        )
        path_gain = weighted_diagnostics["incoherent"]["total"]
        rss = path_gain * float(tx_power)
        return build_result(
            grid=grid,
            tx_power=tx_power,
            weighted_diagnostics=weighted_diagnostics,
            metadata=metadata,
            path_gain=path_gain,
            rss=rss,
            sinr=_single_tx_sinr(rss, noise_power=float(noise_power)),
            tx_pos=tx_pos,
            noise_power=float(noise_power),
            timing=timing,
        )

    @staticmethod
    def scalar_grad_or_zero(value):
        return dr.zeros(wt.Float, 1) if value is None else wt.Float(value)

    @staticmethod
    def point_grad_or_zero(x_grad, y_grad, z_grad, width: int):
        zero = dr.zeros(wt.Float, width)
        return wt.Point3f(
            zero if x_grad is None else wt.Float(x_grad),
            zero if y_grad is None else wt.Float(y_grad),
            zero if z_grad is None else wt.Float(z_grad),
        )

    @staticmethod
    def material_grad_or_zero(eps_grad, mu_grad, sigma_grad, width: int):
        zero = dr.zeros(wt.Float, width)
        return {
            "eps_r": zero if eps_grad is None else wt.Float(eps_grad),
            "mu_r": zero if mu_grad is None else wt.Float(mu_grad),
            "sigma_e": zero if sigma_grad is None else wt.Float(sigma_grad),
        }

    @staticmethod
    def integrate(
        *,
        tx_pos,
        grid_spec,
        mc_config: Config,
        scene: Scene,
        config: ResolvedTraceConfig,
        solver_controls: Mapping[str, object],
        accumulation_backend: str,
        return_timing: bool,
        grad_sensitive_workload: bool,
        trace_primal_fn: Callable[..., object],
        tx_power: float = 1.0,
        noise_power: float | None = None,
    ):
        detached = (
            BasicIntegratorAD.detached_workload(scene, tx_pos, config)
            if grad_sensitive_workload
            else {"tx_pos": tx_pos, "scene": scene}
        )
        primal_state = trace_primal_fn(
            detached["tx_pos"],
            grid_spec,
            mc_config,
            detached["scene"],
            config,
            solver_controls,
            accumulation_backend=accumulation_backend,
            return_timing=return_timing,
            resolved_ad_mode=True,
            ad_backend="outer_custom_op_native_sparse_and_transport_coeff_cuda",
            loop_mode="symbolic",
            tx_power=float(tx_power),
            noise_power=noise_power,
            collect_ad_tapes=True,
            apply_result_filtering=False,
        )
        primal_components = {
            component: dr.detach(value)
            for component, value in primal_state.component_power.items()
        }
        samples_per_tx = int(mc_config.integrator_options.samples_per_tx)
        ray_sampling_metadata = Sampler.metadata(
            axis=str(primal_state.grid.axis),
            plane_position=float(primal_state.grid.position),
            tx_pos=detached["tx_pos"],
        )
        solid_angle_per_ray = Sampler.solid_angle(
            ray_sampling_metadata,
            samples_per_tx,
        )
        cell_area = float(primal_state.grid.cell_size[0] * primal_state.grid.cell_size[1])
        material_omega = material_angular_frequency(config.wavelength)
        material_eps_r, material_mu_r, material_sigma_e = BasicIntegratorAD.scene_materials(scene, config)
        los_tape = (
            primal_state.los_tape
            if primal_state.los_tape is not None
            else LosTape.empty()
        )
        reflection_tape = (
            primal_state.reflection_tape
            if primal_state.reflection_tape is not None
            else ReflectionTape.empty(int(solver_controls["effective"]["reflection_max_bounces"]))
        )
        diffraction_tape = (
            primal_state.diffraction_tape
            if primal_state.diffraction_tape is not None
            else DiffractionTape.empty()
        )
        total_length_weight = float(primal_state.diff_length_weight)
        diff_gain_scale = wt.Float(
            (float(config.wavelength) / (4.0 * math.pi)) ** 2 / float(cell_area)
        )
        scene_vertices = scene._merged_vertices()

        ctx = ADContext(
            primal_components=primal_components,
            n_cells=int(primal_state.grid.n_cells),
            n_vertices=int(dr.width(scene_vertices.x)),
            n_materials=int(dr.width(material_eps_r)),
            diff_gain_scale=diff_gain_scale,
            total_length_weight=float(total_length_weight),
            detached=detached,
            grid=primal_state.grid,
            config=config,
            los_tape=los_tape,
            reflection_tape=reflection_tape,
            diffraction_tape=diffraction_tape,
            material_omega=material_omega,
            solid_angle_per_ray=float(solid_angle_per_ray),
            cell_area=float(cell_area),
        )
        solver_cls = ADContext.make_solver_ad(ctx)
        los_power, reflection_power, diffraction_power = dr.custom(
            solver_cls,
            tx_x=tx_pos.x,
            tx_y=tx_pos.y,
            tx_z=tx_pos.z,
            vertices_x=scene_vertices.x,
            vertices_y=scene_vertices.y,
            vertices_z=scene_vertices.z,
            material_eps_r=material_eps_r,
            material_mu_r=material_mu_r,
            material_sigma_e=material_sigma_e,
        )
        from .basic import EXTRA_COMPONENT_KEYS
        result = BasicIntegratorAD.result_from_components(
            grid=primal_state.grid,
            tx_power=float(tx_power),
            component_power={
                "los": los_power,
                "reflection": reflection_power,
                "diffraction": diffraction_power,
                **{key: primal_components[key] for key in EXTRA_COMPONENT_KEYS},
            },
            metadata=primal_state.metadata,
            tx_pos=tx_pos,
            noise_power=primal_state.noise_power,
            timing=primal_state.timing,
            shadow_boundary_mode=mc_config.tuning.shadow_boundary_mode,
            filtering=mc_config.filtering,
        )
        return result


__all__ = [
    "BasicIntegratorAD",
]
