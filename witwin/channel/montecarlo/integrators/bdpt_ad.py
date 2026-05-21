from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Mapping

import drjit as dr

from witwin.channel.core.scene import Scene
from witwin.channel.core.numerics import arrays
from witwin.channel.core import geometry
from witwin.channel.core.physics import polarization, wave_math
from witwin.channel.montecarlo import types as wt

from .. import grid_ops
from ..config import Config, ResolvedTraceConfig
from witwin.channel.core.grid import Grid
from ..kernels.sparse_coeff import SparseCoeffKernel
from ..kernels.transport_vertex import TransportVertexKernel
from ..trace import ad_support as mc_ad
from ..trace.diffraction import FaceMaterial, OrientedEdgeView
from ..trace.diffraction_ad import DiffractionAD
from ..trace.diffraction import DiffractionEdgeSampler
from ..trace.diffraction_utd import UTD
from ..trace.los import LosAD
from ..trace.reflection import ReflectionAD
from ..sampler import Sampler
from .basic_ad import BasicIntegratorAD
from .bdpt_diffraction import BDPTDiffractionMIS, BDPTDiffractionTape, BDPTDiffractionTapeStore


class BDPTDiffractionAD:
    """Sparse coefficient replay for fixed-topology BDPT diffraction samples."""

    @staticmethod
    def _finite(value):
        return dr.select(dr.isfinite(value), value, wt.Float(0.0))

    @staticmethod
    def _edge_geometry(
        *,
        edge_index,
        edge_fraction,
        source_pos,
        scene: Scene,
        config: ResolvedTraceConfig,
    ) -> dict:
        support = DiffractionAD.edge_support_arrays(scene)
        edge_data = scene.gather_edge_subset(
            edge_index,
            valid_mask=edge_index >= wt.Int32(0),
        )
        safe_edge_idx = wt.UInt32(dr.select(edge_index >= wt.Int32(0), edge_index, wt.Int32(0)))
        edge_v0_idx = dr.gather(wt.Int32, support["edge_v0"], safe_edge_idx)
        edge_v1_idx = dr.gather(wt.Int32, support["edge_v1"], safe_edge_idx)
        face0_third_idx = dr.gather(wt.Int32, support["face0_third"], safe_edge_idx)
        face1_third_idx = dr.gather(wt.Int32, support["face1_third"], safe_edge_idx)
        face0_prim_idx = dr.gather(wt.Int32, support["face0_prim"], safe_edge_idx)
        face1_prim_idx = dr.gather(wt.Int32, support["face1_prim"], safe_edge_idx)

        edge_v0 = mc_ad.SceneQuery.vertex_point(scene, edge_v0_idx)
        edge_v1 = mc_ad.SceneQuery.vertex_point(scene, edge_v1_idx)
        face0_third = mc_ad.SceneQuery.vertex_point(scene, face0_third_idx)
        face1_third = mc_ad.SceneQuery.vertex_point(scene, face1_third_idx)
        dr.enable_grad(
            edge_v0.x,
            edge_v0.y,
            edge_v0.z,
            edge_v1.x,
            edge_v1.y,
            edge_v1.z,
            face0_third.x,
            face0_third.y,
            face0_third.z,
            face1_third.x,
            face1_third.y,
            face1_third.z,
        )

        edge_vec = edge_v1 - edge_v0
        edge_length = dr.norm(edge_vec) + wt.Float(1.0e-6)
        edge_dir = edge_vec / edge_length
        edge_pos = wt.Point3f(
            wt.Float(0.5) * (edge_v0.x + edge_v1.x),
            wt.Float(0.5) * (edge_v0.y + edge_v1.y),
            wt.Float(0.5) * (edge_v0.z + edge_v1.z),
        )
        edge_length_nominal = dr.maximum(edge_data["length"], wt.Float(1.0e-6))
        edge_length_scale = edge_length / edge_length_nominal
        line_min = edge_data["line_min"] * edge_length_scale
        line_max = edge_data["line_max"] * edge_length_scale
        line_length = dr.maximum(line_max - line_min, wt.Float(0.0))
        diff_point = edge_pos + edge_dir * (line_min + line_length * edge_fraction)

        face0_valid = face0_third_idx >= wt.Int32(0)
        face1_valid = face1_third_idx >= wt.Int32(0)
        face0_normal, face1_normal, wedge_n = DiffractionAD.recompute_face_geometry(
            edge_data=edge_data,
            edge_v0=edge_v0,
            edge_v1=edge_v1,
            face0_third=face0_third,
            face1_third=face1_third,
            face0_valid=face0_valid,
            face1_valid=face1_valid,
        )

        face0_material = mc_ad.SceneQuery.material(
            face0_prim_idx,
            scene=scene,
            gain=1.0,
        )
        face1_material = mc_ad.SceneQuery.material(
            face1_prim_idx,
            scene=scene,
            gain=1.0,
        )

        incident_dir = diff_point - source_pos
        flip = dr.dot(incident_dir, face0_normal) > wt.Float(0.0)
        oriented = OrientedEdgeView(
            edge_dir=dr.select(flip, -edge_dir, edge_dir),
            n0=dr.select(flip, face1_normal, face0_normal),
            nn=dr.select(flip, face0_normal, face1_normal),
            face0_material=FaceMaterial(
                eta_r=dr.select(flip, face1_material.eta_r, face0_material.eta_r),
                sigma=dr.select(flip, face1_material.sigma, face0_material.sigma),
                gain=dr.select(flip, face1_material.gain, face0_material.gain),
                use_fresnel=dr.select(flip, face1_material.use_fresnel, face0_material.use_fresnel),
                mu_r=dr.select(flip, face1_material.mu_r, face0_material.mu_r),
            ),
            face1_material=FaceMaterial(
                eta_r=dr.select(flip, face0_material.eta_r, face1_material.eta_r),
                sigma=dr.select(flip, face0_material.sigma, face1_material.sigma),
                gain=dr.select(flip, face0_material.gain, face1_material.gain),
                use_fresnel=dr.select(flip, face0_material.use_fresnel, face1_material.use_fresnel),
                mu_r=dr.select(flip, face0_material.mu_r, face1_material.mu_r),
            ),
        )
        return {
            "edge_index": edge_index,
            "edge_pos": edge_pos,
            "oriented": oriented,
            "wedge_n": wedge_n,
            "diff_point": diff_point,
            "vertex_indices": (
                edge_v0_idx,
                edge_v1_idx,
                face0_third_idx,
                face1_third_idx,
            ),
            "vertex_vars": (edge_v0, edge_v1, face0_third, face1_third),
            "materials": (face0_material, face1_material),
        }

    @staticmethod
    def _edge_power(*, event, source_pos, target_pos, config: ResolvedTraceConfig):
        field_power, _, _ = UTD.edge_diffraction_power(
            source_pos=source_pos,
            oriented=event["oriented"],
            wedge_n=event["wedge_n"],
            sampled_edge_pos=event["diff_point"],
            target_pos=target_pos,
            k=config.k,
            wavelength=config.wavelength,
        )
        return BDPTDiffractionAD._finite(field_power)

    @staticmethod
    def _suffix_geometry(*, tape: BDPTDiffractionTape, final_event, target_pos, scene: Scene,
                         config: ResolvedTraceConfig, material_omega):
        tri_data = scene._triangle_runtime()
        width = int(dr.width(tape.strategy))
        zero_point = wt.Point3f(
            dr.zeros(wt.Float, width),
            dr.zeros(wt.Float, width),
            dr.zeros(wt.Float, width),
        )
        zero_vec = wt.Vector3f(
            dr.zeros(wt.Float, width),
            dr.zeros(wt.Float, width),
            dr.zeros(wt.Float, width),
        )
        if tri_data is None:
            return {
                "reflection_point": zero_point,
                "reflection_power": dr.zeros(wt.Float, width),
                "suffix_fspl": dr.zeros(wt.Float, width),
                "vertex_indices": (),
                "vertex_vars": (),
                "materials": (),
            }

        faces = scene._merged_faces()
        face_x = wt.Int32(faces.x)
        face_y = wt.Int32(faces.y)
        face_z = wt.Int32(faces.z)
        valid_prim = tape.suffix_prim_idx >= wt.Int32(0)
        safe_prim_idx = wt.UInt32(dr.select(valid_prim, tape.suffix_prim_idx, wt.Int32(0)))
        v0_idx = dr.select(valid_prim, dr.gather(wt.Int32, face_x, safe_prim_idx), wt.Int32(-1))
        v1_idx = dr.select(valid_prim, dr.gather(wt.Int32, face_y, safe_prim_idx), wt.Int32(-1))
        v2_idx = dr.select(valid_prim, dr.gather(wt.Int32, face_z, safe_prim_idx), wt.Int32(-1))
        v0 = mc_ad.SceneQuery.vertex_point(scene, v0_idx)
        v1 = mc_ad.SceneQuery.vertex_point(scene, v1_idx)
        v2 = mc_ad.SceneQuery.vertex_point(scene, v2_idx)
        dr.enable_grad(v0.x, v0.y, v0.z, v1.x, v1.y, v1.z, v2.x, v2.y, v2.z)
        geom_n = dr.cross(v1 - v0, v2 - v0)
        geom_n = geom_n / (dr.norm(geom_n) + wt.Float(1.0e-6))
        image_source = geometry.reflect_point_across_plane(final_event["diff_point"], v0, geom_n)
        segment = target_pos - image_source
        denom = dr.dot(segment, geom_n)
        t_hit = dr.dot(v0 - image_source, geom_n) / (denom + wt.Float(1.0e-6))
        reflection_point = image_source + t_hit * segment
        incoming = reflection_point - final_event["diff_point"]
        outgoing = target_pos - reflection_point
        incoming_hat = incoming / (dr.norm(incoming) + wt.Float(1.0e-6))
        outgoing_dist = dr.norm(outgoing)
        oriented_normal = dr.select(
            dr.dot(incoming_hat, geom_n) > wt.Float(0.0),
            -geom_n,
            geom_n,
        )
        material_inputs = mc_ad.SceneQuery.material(tape.suffix_prim_idx, scene=scene, gain=1.0)
        cos_theta = dr.clip(
            dr.abs(dr.dot(-incoming_hat, oriented_normal)),
            wt.Float(1.0e-6),
            wt.Float(1.0),
        )
        eta = wave_math.complex_relative_permittivity(
            material_inputs.eta_r, material_inputs.sigma, material_omega,
        )
        r_te, r_tm = wave_math.fresnel_reflection(cos_theta, eta, mu_r=material_inputs.mu_r)
        fresnel_power = wt.Float(0.5) * (
            arrays.complex_abs_sqr(r_te) + arrays.complex_abs_sqr(r_tm)
        )
        reflection_power = dr.square(material_inputs.gain) * fresnel_power
        suffix_fspl = dr.square(
            wt.Float(config.wavelength / (4.0 * math.pi))
            / dr.maximum(outgoing_dist, wt.Float(1.0e-6))
        )
        return {
            "reflection_point": dr.select(valid_prim, reflection_point, zero_point),
            "reflection_power": dr.select(valid_prim, reflection_power, wt.Float(0.0)),
            "suffix_fspl": dr.select(valid_prim, suffix_fspl, wt.Float(0.0)),
            "vertex_indices": (v0_idx, v1_idx, v2_idx),
            "vertex_vars": (v0, v1, v2),
            "materials": (material_inputs,),
        }

    @staticmethod
    def sparse_coeffs(
        *,
        tape: BDPTDiffractionTape,
        scene: Scene,
        tx_pos,
        grid: Grid,
        config: ResolvedTraceConfig,
        material_omega,
        solid_angle_per_ray: float,
    ):
        width = int(dr.width(tape.strategy))
        if width <= 0:
            return mc_ad.SparseCoeffBuffers.empty()

        local_tx = mc_ad.SceneQuery.tx_lanes(tx_pos, width)
        dr.enable_grad(local_tx.x, local_tx.y, local_tx.z)
        direct_source = tape.prefix_reflection_depth <= wt.Int32(0)
        safe_prefix_ray_dir = wt.Vector3f(
            dr.select(direct_source, wt.Float(1.0), tape.prefix_initial_ray_dir.x),
            dr.select(direct_source, wt.Float(0.0), tape.prefix_initial_ray_dir.y),
            dr.select(direct_source, wt.Float(0.0), tape.prefix_initial_ray_dir.z),
        )
        prefix_bounced = ReflectionAD.replay_bounces(
            ray_dir=safe_prefix_ray_dir,
            ray_origin=local_tx,
            cumulative_image_source=local_tx,
            scene=scene,
            config=config,
            depth_array=tape.prefix_reflection_depth,
            prim_by_bounce=tape.prefix_prim_by_bounce,
            material_omega=material_omega,
            track_vertices=True,
            track_materials=True,
        )
        source_pos = wt.Point3f(
            dr.select(direct_source, local_tx.x, prefix_bounced["cumulative_image_source"].x),
            dr.select(direct_source, local_tx.y, prefix_bounced["cumulative_image_source"].y),
            dr.select(direct_source, local_tx.z, prefix_bounced["cumulative_image_source"].z),
        )
        prefix_power = (
            polarization.vector_power(prefix_bounced["polarization_vec"])
            * wt.Float(float(solid_angle_per_ray))
        )
        detached_prefix_power = dr.detach(prefix_power)
        prefix_power_scale = dr.select(
            dr.abs(detached_prefix_power) > wt.Float(1.0e-30),
            dr.detach(tape.source_power) / detached_prefix_power,
            wt.Float(0.0),
        )
        source_power = dr.select(
            direct_source,
            wt.Float(1.0),
            BDPTDiffractionAD._finite(prefix_power * prefix_power_scale),
        )

        events = []
        target_events = []
        source_for_step = []
        throughput_for_step = []
        source_cursor = source_pos
        throughput = source_power
        event_gain_scale = wt.Float((float(config.wavelength) / (4.0 * math.pi)) ** 2)
        for step in range(BDPTDiffractionTapeStore.MAX_DEPTH):
            event = BDPTDiffractionAD._edge_geometry(
                edge_index=tape.edge_index_by_step[step],
                edge_fraction=tape.edge_fraction_by_step[step],
                source_pos=source_cursor,
                scene=scene,
                config=config,
            )
            events.append(event)
            source_for_step.append(source_cursor)
            throughput_for_step.append(throughput)
            if step + 1 < BDPTDiffractionTapeStore.MAX_DEPTH:
                next_event = BDPTDiffractionAD._edge_geometry(
                    edge_index=tape.edge_index_by_step[step + 1],
                    edge_fraction=tape.edge_fraction_by_step[step + 1],
                    source_pos=event["diff_point"],
                    scene=scene,
                    config=config,
                )
                target_events.append(next_event)
                intermediate_active = tape.order > wt.Int32(step + 1)
                field_power = BDPTDiffractionAD._edge_power(
                    event=event,
                    source_pos=source_cursor,
                    target_pos=next_event["diff_point"],
                    config=config,
                )
                throughput = dr.select(
                    intermediate_active,
                    BDPTDiffractionAD._finite(throughput * field_power * event_gain_scale),
                    throughput,
                )
                source_cursor = dr.select(
                    intermediate_active,
                    event["diff_point"],
                    source_cursor,
                )

        cell_target = dr.gather(wt.Point3f, grid.cell_centers, tape.cell_idx)
        contribution = dr.zeros(wt.Float, width)
        plane_normal = Sampler.axis_unit_normal(str(grid.axis))
        suffix_geometries = []
        for step, event in enumerate(events):
            final_mask = tape.order == wt.Int32(step + 1)
            source_step = source_for_step[step]
            throughput_step = throughput_for_step[step]

            direct_power = BDPTDiffractionAD._edge_power(
                event=event,
                source_pos=source_step,
                target_pos=cell_target,
                config=config,
            )
            direct_contribution = throughput_step * direct_power * tape.scalar_weight
            contribution = contribution + dr.select(
                final_mask & (tape.strategy == wt.Int32(BDPTDiffractionTapeStore.DIRECT_ID)),
                direct_contribution,
                wt.Float(0.0),
            )

            oriented = event["oriented"]
            ko = UTD.sample_keller_cone(
                oriented.edge_dir,
                oriented.n0,
                oriented.nn,
                tape.keller_sample,
                event["diff_point"] - source_step,
                lit_region=True,
            )
            ray_origin = event["diff_point"] + ko * wt.Float(1.0e-4)
            plane_hit = grid_ops.plane_hit(
                ray_origin=ray_origin,
                ray_dir=ko,
                blocker_dist=dr.full(wt.Float, 1.0e10, width),
                grid=grid,
                active=dr.full(wt.Bool, True, width),
            )
            keller_power = BDPTDiffractionAD._edge_power(
                event=event,
                source_pos=source_step,
                target_pos=plane_hit.target_pos,
                config=config,
            )
            exterior_angle = BDPTDiffractionMIS.exterior_angle(oriented.n0, oriented.nn)
            iw = UTD.integration_weight(
                edge_origin=event["edge_pos"],
                edge_dir=oriented.edge_dir,
                n0=oriented.n0,
                source_pos=source_step,
                diff_point=event["diff_point"],
                k_world=ko,
                target_pos=plane_hit.target_pos,
                plane_normal=plane_normal,
            )
            keller_contribution = (
                throughput_step
                * keller_power
                * BDPTDiffractionAD._finite(iw)
                * BDPTDiffractionAD._finite(exterior_angle)
                * tape.scalar_weight
            )
            contribution = contribution + dr.select(
                final_mask & (tape.strategy == wt.Int32(BDPTDiffractionTapeStore.KELLER_ID)),
                keller_contribution,
                wt.Float(0.0),
            )

            suffix = BDPTDiffractionAD._suffix_geometry(
                tape=tape,
                final_event=event,
                target_pos=cell_target,
                scene=scene,
                config=config,
                material_omega=material_omega,
            )
            suffix_geometries.append(suffix)
            suffix_power = BDPTDiffractionAD._edge_power(
                event=event,
                source_pos=source_step,
                target_pos=suffix["reflection_point"],
                config=config,
            )
            suffix_contribution = (
                throughput_step
                * suffix_power
                * BDPTDiffractionAD._finite(suffix["reflection_power"])
                * BDPTDiffractionAD._finite(suffix["suffix_fspl"])
                * tape.scalar_weight
            )
            contribution = contribution + dr.select(
                final_mask
                & (tape.strategy == wt.Int32(BDPTDiffractionTapeStore.SUFFIX_REFLECTION_ID)),
                suffix_contribution,
                wt.Float(0.0),
            )

        contribution = BDPTDiffractionAD._finite(contribution)
        dr.backward(dr.sum(contribution))

        vertex_indices = []
        vertex_coeff_x = []
        vertex_coeff_y = []
        vertex_coeff_z = []
        material_indices = []
        material_coeff_eps = []
        material_coeff_sigma = []
        vertex_indices.extend(prefix_bounced["vertex_index_slots"])
        vertex_coeff_x.extend(dr.grad(vertex_var.x) for vertex_var in prefix_bounced["vertex_vars"])
        vertex_coeff_y.extend(dr.grad(vertex_var.y) for vertex_var in prefix_bounced["vertex_vars"])
        vertex_coeff_z.extend(dr.grad(vertex_var.z) for vertex_var in prefix_bounced["vertex_vars"])
        for material_idx, material_vars in zip(
            prefix_bounced["material_index_slots"],
            prefix_bounced["material_grad_sources"],
        ):
            material_indices.append(material_idx)
            eta, sigma = material_vars
            material_coeff_eps.append(dr.grad(eta))
            material_coeff_sigma.append(dr.grad(sigma))
        for event in tuple(events) + tuple(target_events):
            vertex_indices.extend(event["vertex_indices"])
            vertex_coeff_x.extend(dr.grad(vertex_var.x) for vertex_var in event["vertex_vars"])
            vertex_coeff_y.extend(dr.grad(vertex_var.y) for vertex_var in event["vertex_vars"])
            vertex_coeff_z.extend(dr.grad(vertex_var.z) for vertex_var in event["vertex_vars"])
            for material_support in event["materials"]:
                material_indices.append(material_support.material_idx)
                material_coeff_eps.append(dr.grad(material_support.eta_r))
                material_coeff_sigma.append(dr.grad(material_support.sigma))

        for suffix in suffix_geometries:
            vertex_indices.extend(suffix["vertex_indices"])
            vertex_coeff_x.extend(dr.grad(vertex_var.x) for vertex_var in suffix["vertex_vars"])
            vertex_coeff_y.extend(dr.grad(vertex_var.y) for vertex_var in suffix["vertex_vars"])
            vertex_coeff_z.extend(dr.grad(vertex_var.z) for vertex_var in suffix["vertex_vars"])
            for material_support in suffix["materials"]:
                material_indices.append(material_support.material_idx)
                material_coeff_eps.append(dr.grad(material_support.eta_r))
                material_coeff_sigma.append(dr.grad(material_support.sigma))

        return mc_ad.SparseCoeffBuffers(
            cell_idx=tape.cell_idx,
            tx_coeff_x=dr.grad(local_tx.x),
            tx_coeff_y=dr.grad(local_tx.y),
            tx_coeff_z=dr.grad(local_tx.z),
            vertex_indices=mc_ad.GridScatter.flatten_slots(wt.Int32, vertex_indices),
            vertex_coeff_x=mc_ad.GridScatter.flatten_slots(wt.Float, vertex_coeff_x),
            vertex_coeff_y=mc_ad.GridScatter.flatten_slots(wt.Float, vertex_coeff_y),
            vertex_coeff_z=mc_ad.GridScatter.flatten_slots(wt.Float, vertex_coeff_z),
            vertex_slot_count=len(vertex_indices),
            material_indices=mc_ad.GridScatter.flatten_slots(wt.Int32, material_indices),
            material_coeff_eps=mc_ad.GridScatter.flatten_slots(wt.Float, material_coeff_eps),
            material_coeff_sigma=mc_ad.GridScatter.flatten_slots(wt.Float, material_coeff_sigma),
            material_slot_count=len(material_indices),
        )


@dataclass
class BDPTADContext:
    primal_components: dict
    n_cells: int
    n_vertices: int
    n_materials: int
    detached: dict
    grid: Grid
    config: ResolvedTraceConfig
    los_tape: object
    reflection_tape: object
    diffraction_tape: BDPTDiffractionTape
    material_omega: object
    solid_angle_per_ray: float
    cell_area: float
    coeff_cache: dict = field(default_factory=dict)
    transport_cache: dict = field(default_factory=dict)

    @staticmethod
    def ensure_coeff_buffers(ctx: "BDPTADContext"):
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
        ctx.coeff_cache["diffraction"] = BDPTDiffractionAD.sparse_coeffs(
            tape=ctx.diffraction_tape,
            scene=ctx.detached["scene"],
            tx_pos=ctx.detached["tx_pos"],
            grid=ctx.grid,
            config=ctx.config,
            material_omega=ctx.material_omega,
            solid_angle_per_ray=ctx.solid_angle_per_ray,
        )
        return (
            ctx.coeff_cache["los"],
            ctx.coeff_cache["reflection"],
            ctx.coeff_cache["diffraction"],
        )

    @staticmethod
    def ensure_tx_transport(ctx: "BDPTADContext"):
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
    def ensure_reflection_transport_coeff_buffers(ctx: "BDPTADContext"):
        if "reflection_transport" in ctx.coeff_cache:
            return ctx.coeff_cache["reflection_transport"]
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
        return ctx.coeff_cache["reflection_transport"]

    @staticmethod
    def make_solver_ad(ctx: "BDPTADContext"):
        class _BDPTSolverAD(dr.CustomOp):
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
                los_buf, refl_buf, diff_buf = BDPTADContext.ensure_coeff_buffers(ctx)
                refl_transport_buf = BDPTADContext.ensure_reflection_transport_coeff_buffers(ctx)
                tx_tangent = wt.Point3f(
                    BasicIntegratorAD.scalar_grad_or_zero(self.grad_in("tx_x")),
                    BasicIntegratorAD.scalar_grad_or_zero(self.grad_in("tx_y")),
                    BasicIntegratorAD.scalar_grad_or_zero(self.grad_in("tx_z")),
                )
                los_transport, refl_transport = BDPTADContext.ensure_tx_transport(ctx)
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
                self.set_grad_out((los_grad, reflection_grad, diffraction_grad))

            def backward(self):
                los_buf, refl_buf, diff_buf = BDPTADContext.ensure_coeff_buffers(ctx)
                refl_transport_buf = BDPTADContext.ensure_reflection_transport_coeff_buffers(ctx)
                upstream_los, upstream_refl, upstream_diff = self.grad_out()
                upstream_los = dr.zeros(wt.Float, ctx.n_cells) if upstream_los is None else wt.Float(upstream_los)
                upstream_refl = dr.zeros(wt.Float, ctx.n_cells) if upstream_refl is None else wt.Float(upstream_refl)
                upstream_diff = dr.zeros(wt.Float, ctx.n_cells) if upstream_diff is None else wt.Float(upstream_diff)
                los_transport, refl_transport = BDPTADContext.ensure_tx_transport(ctx)
                tx_grad_x = dr.zeros(wt.Float, 1)
                tx_grad_y = dr.zeros(wt.Float, 1)
                tx_grad_z = dr.zeros(wt.Float, 1)
                vertex_grad_x = dr.zeros(wt.Float, ctx.n_vertices)
                vertex_grad_y = dr.zeros(wt.Float, ctx.n_vertices)
                vertex_grad_z = dr.zeros(wt.Float, ctx.n_vertices)
                material_grad_eps = dr.zeros(wt.Float, ctx.n_materials)
                material_grad_mu = dr.zeros(wt.Float, ctx.n_materials)
                material_grad_sigma = dr.zeros(wt.Float, ctx.n_materials)

                for buffers, upstream in (
                    (los_buf, upstream_los),
                    (refl_buf, upstream_refl),
                    (diff_buf, upstream_diff),
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
                return "BDPTRadioMapMonteCarloPathwiseSparseCoeffAD"

        return _BDPTSolverAD


class BDPTIntegratorAD:
    """Custom-op AD wrapper for the BDPT Monte Carlo integrator."""

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
        if not SparseCoeffKernel.available():
            raise RuntimeError(
                "IntegratorOptions(integrator='bdpt', ad=True) requires the Monte Carlo "
                "native AD kernels. Rebuild the witwin.channel.montecarlo native extension."
            )
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
            ad_backend="outer_custom_op_bdpt_pathwise_sparse_coeff_cuda",
            tx_power=float(tx_power),
            noise_power=noise_power,
            collect_ad_tapes=True,
            return_primal_state=True,
            apply_result_filtering=False,
        )
        primal_components = {
            component: dr.detach(value)
            for component, value in primal_state.component_power.items()
        }
        material_eps_r, material_mu_r, material_sigma_e = BasicIntegratorAD.scene_materials(scene, config)
        scene_vertices = scene._merged_vertices()
        ctx = BDPTADContext(
            primal_components=primal_components,
            n_cells=int(primal_state.grid.n_cells),
            n_vertices=int(dr.width(scene_vertices.x)),
            n_materials=int(dr.width(material_eps_r)),
            detached=detached,
            grid=primal_state.grid,
            config=config,
            los_tape=primal_state.los_tape,
            reflection_tape=primal_state.reflection_tape,
            diffraction_tape=(
                primal_state.diffraction_tape
                if primal_state.diffraction_tape is not None
                else BDPTDiffractionTape.empty()
            ),
            material_omega=wave_math.material_angular_frequency(config.wavelength),
            solid_angle_per_ray=float(
                primal_state.metadata["ray_sampling"]["solid_angle_per_ray_sr"]
            ),
            cell_area=float(primal_state.grid.cell_size[0] * primal_state.grid.cell_size[1]),
        )
        solver_cls = BDPTADContext.make_solver_ad(ctx)
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
    "BDPTDiffractionAD",
    "BDPTIntegratorAD",
]
