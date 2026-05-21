"""Diffraction AD support: sparse coefficients, vertex transport JVP/VJP."""
from __future__ import annotations

import math

import drjit as dr
from witwin.channel.core.scene import Scene
from witwin.channel.montecarlo import types as wt
from witwin.channel.core.grid import Grid
from .. import grid_ops
from ..config import ResolvedTraceConfig
from ..sampler import Sampler
from ..kernels.transport_vertex import TransportVertexKernel
from .ad_support import SparseCoeffBuffers, GridScatter, SceneQuery, TransportCoeffBuilder
from .diffraction import DiffractionEdgeSampler, DiffractionTape, FaceMaterial, OrientedEdgeView
from .diffraction_utd import UTD, DiffractionSupportOverride


class DiffractionAD:
    """Static namespace for diffraction AD: sparse coefficients, vertex transport JVP/VJP."""

    @staticmethod
    def recompute_face_geometry(*, edge_data, edge_v0, edge_v1, face0_third, face1_third, face0_valid, face1_valid):
        edge_vec = edge_v1 - edge_v0
        raw_face0 = dr.normalize(dr.cross(edge_vec, face0_third - edge_v0))
        raw_face1 = dr.normalize(dr.cross(edge_vec, face1_third - edge_v0))
        face0_normal = dr.select(
            face0_valid,
            dr.select(dr.dot(raw_face0, edge_data["n0"]) >= 0.0, raw_face0, -raw_face0),
            edge_data["n0"],
        )
        face1_normal = dr.select(
            face1_valid,
            dr.select(dr.dot(raw_face1, edge_data["n_face_n"]) >= 0.0, raw_face1, -raw_face1),
            edge_data["n_face_n"],
        )
        wedge_interior = dr.safe_acos(
            dr.clip(
                -dr.dot(face0_normal, face1_normal),
                wt.Float(-1.0),
                wt.Float(1.0),
            )
        )
        wedge_n = dr.select(
            face0_valid & face1_valid,
            (wt.Float(2.0 * math.pi) - wedge_interior) / wt.Float(math.pi),
            edge_data["wedge_n"],
        )
        return face0_normal, face1_normal, wedge_n

    @staticmethod
    def edge_geometry(*, tape: DiffractionTape, scene: Scene, local_tx, grid: Grid,
                      config: ResolvedTraceConfig, width: int, enable_vertex_grad: bool = False,
                      enable_face_grad: bool = False) -> dict:
        support = DiffractionAD.edge_support_arrays(scene)
        edge_data = scene.gather_edge_subset(tape.edge_index, valid_mask=tape.edge_index >= 0)
        safe_edge_idx = wt.UInt32(dr.select(tape.edge_index >= 0, tape.edge_index, wt.Int32(0)))
        edge_v0_idx = dr.gather(wt.Int32, support["edge_v0"], safe_edge_idx)
        edge_v1_idx = dr.gather(wt.Int32, support["edge_v1"], safe_edge_idx)
        face0_third_idx = dr.gather(wt.Int32, support["face0_third"], safe_edge_idx)
        face1_third_idx = dr.gather(wt.Int32, support["face1_third"], safe_edge_idx)
        face0_prim_idx = dr.gather(wt.Int32, support["face0_prim"], safe_edge_idx)
        face1_prim_idx = dr.gather(wt.Int32, support["face1_prim"], safe_edge_idx)

        edge_v0 = SceneQuery.vertex_point(scene, edge_v0_idx)
        edge_v1 = SceneQuery.vertex_point(scene, edge_v1_idx)
        face0_third = SceneQuery.vertex_point(scene, face0_third_idx)
        face1_third = SceneQuery.vertex_point(scene, face1_third_idx)
        if enable_vertex_grad:
            dr.enable_grad(edge_v0.x, edge_v0.y, edge_v0.z, edge_v1.x, edge_v1.y, edge_v1.z)
        if enable_face_grad:
            dr.enable_grad(
                face0_third.x, face0_third.y, face0_third.z,
                face1_third.x, face1_third.y, face1_third.z,
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
        diff_point = edge_pos + edge_dir * (line_min + line_length * tape.edge_fraction)
        incident_dir = diff_point - local_tx
        face0_valid = face0_third_idx >= 0
        face1_valid = face1_third_idx >= 0
        face0_normal, face1_normal, wedge_n = DiffractionAD.recompute_face_geometry(
            edge_data=edge_data,
            edge_v0=edge_v0,
            edge_v1=edge_v1,
            face0_third=face0_third,
            face1_third=face1_third,
            face0_valid=face0_valid,
            face1_valid=face1_valid,
        )
        exterior_angle = wedge_n * wt.Float(math.pi)

        face0_material = SceneQuery.material(face0_prim_idx, scene=scene, gain=1.0)
        face1_material = SceneQuery.material(face1_prim_idx, scene=scene, gain=1.0)

        flip = dr.dot(incident_dir, face0_normal) > 0.0
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

        face_sum = oriented.n0 + oriented.nn
        face_sum_norm = dr.norm(face_sum)
        offset_normal = dr.select(
            face_sum_norm > wt.Float(1.0e-6), face_sum / face_sum_norm, wt.Vector3f(0.0, 0.0, 0.0),
        )
        ko = UTD.sample_keller_cone(
            oriented.edge_dir, oriented.n0, oriented.nn, tape.cone_sample, incident_dir, lit_region=True,
        )
        ray_origin = Sampler.spawn_offset_ray_origin(diff_point, ko, offset_normal)
        plane_hit = grid_ops.plane_hit(
            ray_origin=ray_origin, ray_dir=ko,
            blocker_dist=dr.full(wt.Float, 1.0e10, width),
            grid=grid, active=dr.full(wt.Bool, True, width),
        )
        integration_weight = UTD.integration_weight(
            edge_origin=edge_pos, edge_dir=oriented.edge_dir, n0=oriented.n0,
            source_pos=local_tx, diff_point=diff_point, k_world=ko,
            target_pos=plane_hit.target_pos,
            plane_normal=Sampler.axis_unit_normal(str(grid.axis)),
        )
        field_power, _, _ = UTD.edge_diffraction_power(
            source_pos=local_tx, oriented=oriented, wedge_n=wedge_n,
            sampled_edge_pos=diff_point, target_pos=plane_hit.target_pos,
            k=config.k, wavelength=config.wavelength,
            support_override=DiffractionSupportOverride(
                field_valid=tape.field_valid, pole_safe=tape.pole_safe,
                dif_n_p=tape.dif_n_p, dif_n_m=tape.dif_n_m,
                sum_n_p=tape.sum_n_p, sum_n_m=tape.sum_n_m,
            ),
        )
        return {
            "plane_hit": plane_hit,
            "field_power": field_power,
            "integration_weight": integration_weight,
            "exterior_angle": exterior_angle,
            "edge_v0": edge_v0, "edge_v1": edge_v1,
            "face0_third": face0_third, "face1_third": face1_third,
            "edge_v0_idx": edge_v0_idx, "edge_v1_idx": edge_v1_idx,
            "face0_third_idx": face0_third_idx, "face1_third_idx": face1_third_idx,
            "face0_material": face0_material, "face1_material": face1_material,
        }

    @staticmethod
    def transport_geometry_state(*, tape: DiffractionTape, scene: Scene, tx_pos, grid: Grid,
                                 config: ResolvedTraceConfig, diff_gain_scale, total_length_weight: float):
        width = int(dr.width(tape.cell_idx))
        if width <= 0:
            return None

        local_tx = SceneQuery.tx_lanes(tx_pos, width)
        geo = DiffractionAD.edge_geometry(
            tape=tape, scene=scene, local_tx=local_tx, grid=grid,
            config=config, width=width, enable_vertex_grad=True, enable_face_grad=True,
        )
        contribution = dr.detach(
            geo["field_power"]
            * diff_gain_scale
            * geo["integration_weight"]
            * wt.Float(total_length_weight)
            * geo["exterior_angle"]
        )
        return {
            "plane_hit": geo["plane_hit"],
            "contribution": contribution,
            "edge_v0_idx": geo["edge_v0_idx"],
            "edge_v1_idx": geo["edge_v1_idx"],
            "face0_third_idx": geo["face0_third_idx"],
            "face1_third_idx": geo["face1_third_idx"],
            "edge_v0": geo["edge_v0"],
            "edge_v1": geo["edge_v1"],
            "face0_third": geo["face0_third"],
            "face1_third": geo["face1_third"],
        }

    @staticmethod
    def vertex_transport_state(*, tape: DiffractionTape, scene: Scene, tx_pos, grid: Grid,
                               config: ResolvedTraceConfig, diff_gain_scale, total_length_weight: float):
        state = DiffractionAD.transport_geometry_state(
            tape=tape,
            scene=scene,
            tx_pos=tx_pos,
            grid=grid,
            config=config,
            diff_gain_scale=diff_gain_scale,
            total_length_weight=total_length_weight,
        )
        if state is None:
            return None
        width = int(dr.width(state["contribution"]))
        state["transport_map"] = GridScatter.tent_splat(
            grid=grid,
            coord_0=state["plane_hit"].coord_0,
            coord_1=state["plane_hit"].coord_1,
            power=state["contribution"],
            active=dr.full(wt.Bool, True, width),
        )
        return state

    @staticmethod
    def transport_vertex_coeffs(*, tape: DiffractionTape, scene: Scene, tx_pos, grid: Grid,
                                config: ResolvedTraceConfig, diff_gain_scale, total_length_weight: float):
        return TransportCoeffBuilder.build_vertex_buffers(
            grid=grid,
            state_factory=lambda: DiffractionAD.transport_geometry_state(
                tape=tape,
                scene=scene,
                tx_pos=tx_pos,
                grid=grid,
                config=config,
                diff_gain_scale=diff_gain_scale,
                total_length_weight=total_length_weight,
            ),
            vertex_index_getter=lambda state: (
                state["edge_v0_idx"],
                state["edge_v1_idx"],
                state["face0_third_idx"],
                state["face1_third_idx"],
            ),
            vertex_var_getter=lambda state: (
                state["edge_v0"],
                state["edge_v1"],
                state["face0_third"],
                state["face1_third"],
            ),
        )

    @staticmethod
    def edge_support_arrays(scene: Scene):
        mesh_version = int(getattr(scene, "_mesh_version", 0))
        cached_version = getattr(scene, "_mc_rm_diff_edge_support_version", None)
        cached_payload = getattr(scene, "_mc_rm_diff_edge_support", None)
        if cached_version == mesh_version and cached_payload is not None:
            return cached_payload

        selection = getattr(scene, "_wedge_selection", None)
        tri_data = scene._triangle_runtime()
        faces = scene._merged_faces()
        if selection is None or tri_data is None or faces is None or int(selection.size()) <= 0:
            empty = {
                "edge_v0": dr.zeros(wt.Int32, 0),
                "edge_v1": dr.zeros(wt.Int32, 0),
                "face0_third": dr.zeros(wt.Int32, 0),
                "face1_third": dr.zeros(wt.Int32, 0),
                "face0_prim": dr.zeros(wt.Int32, 0),
                "face1_prim": dr.zeros(wt.Int32, 0),
            }
            scene._mc_rm_diff_edge_support_version = mesh_version
            scene._mc_rm_diff_edge_support = empty
            return empty

        selected_idx = selection.selected_idx
        geometry = selection.geometry
        face_x = wt.Int32(faces.x)
        face_y = wt.Int32(faces.y)
        face_z = wt.Int32(faces.z)
        edge_v0 = dr.gather(wt.Int32, wt.Int32(geometry.v0), selected_idx)
        edge_v1 = dr.gather(wt.Int32, wt.Int32(geometry.v1), selected_idx)
        face0_prim = dr.gather(wt.Int32, wt.Int32(geometry.face0), selected_idx)
        face1_prim = dr.gather(wt.Int32, wt.Int32(geometry.face1), selected_idx)

        def third_vertex_idx(face_idx, edge_a, edge_b):
            valid = face_idx >= 0
            safe_face = wt.UInt32(dr.select(valid, face_idx, wt.Int32(0)))
            fv0 = dr.gather(wt.Int32, face_x, safe_face)
            fv1 = dr.gather(wt.Int32, face_y, safe_face)
            fv2 = dr.gather(wt.Int32, face_z, safe_face)
            fv0_is_edge = (fv0 == edge_a) | (fv0 == edge_b)
            fv1_is_edge = (fv1 == edge_a) | (fv1 == edge_b)
            third = dr.select(
                ~fv0_is_edge,
                fv0,
                dr.select(~fv1_is_edge, fv1, fv2),
            )
            return dr.select(valid, third, wt.Int32(-1))

        payload = {
            "edge_v0": edge_v0,
            "edge_v1": edge_v1,
            "face0_third": third_vertex_idx(face0_prim, edge_v0, edge_v1),
            "face1_third": third_vertex_idx(face1_prim, edge_v0, edge_v1),
            "face0_prim": face0_prim,
            "face1_prim": face1_prim,
        }
        scene._mc_rm_diff_edge_support_version = mesh_version
        scene._mc_rm_diff_edge_support = payload
        return payload

    @staticmethod
    def sparse_coeffs(*, tape: DiffractionTape, scene: Scene, tx_pos, grid: Grid, config: ResolvedTraceConfig,
                      diff_gain_scale, total_length_weight: float):
        width = int(dr.width(tape.cell_idx))
        if width <= 0:
            return SparseCoeffBuffers.empty()

        local_tx = SceneQuery.tx_lanes(tx_pos, width)
        dr.enable_grad(local_tx.x, local_tx.y, local_tx.z)
        geo = DiffractionAD.edge_geometry(
            tape=tape, scene=scene, local_tx=local_tx, grid=grid,
            config=config, width=width, enable_vertex_grad=True, enable_face_grad=True,
        )
        contribution = (
            geo["field_power"]
            * diff_gain_scale
            * geo["integration_weight"]
            * wt.Float(total_length_weight)
            * geo["exterior_angle"]
        )
        dr.backward(dr.sum(contribution))

        material_index_slots = []
        material_grad_sources = []
        for material_support in (geo["face0_material"], geo["face1_material"]):
            material_index_slots.append(material_support.material_idx)
            material_grad_sources.append((material_support.eta_r, material_support.sigma))

        vertex_vars = (
            geo["edge_v0"],
            geo["edge_v1"],
            geo["face0_third"],
            geo["face1_third"],
        )
        vertex_indices = (
            geo["edge_v0_idx"],
            geo["edge_v1_idx"],
            geo["face0_third_idx"],
            geo["face1_third_idx"],
        )
        return SparseCoeffBuffers(
            cell_idx=tape.cell_idx,
            tx_coeff_x=dr.grad(local_tx.x),
            tx_coeff_y=dr.grad(local_tx.y),
            tx_coeff_z=dr.grad(local_tx.z),
            vertex_indices=GridScatter.flatten_slots(wt.Int32, list(vertex_indices)),
            vertex_coeff_x=GridScatter.flatten_slots(
                wt.Float, [dr.grad(vertex_var.x) for vertex_var in vertex_vars]
            ),
            vertex_coeff_y=GridScatter.flatten_slots(
                wt.Float, [dr.grad(vertex_var.y) for vertex_var in vertex_vars]
            ),
            vertex_coeff_z=GridScatter.flatten_slots(
                wt.Float, [dr.grad(vertex_var.z) for vertex_var in vertex_vars]
            ),
            vertex_slot_count=len(vertex_vars),
            material_indices=GridScatter.flatten_slots(wt.Int32, material_index_slots),
            material_coeff_eps=GridScatter.flatten_slots(
                wt.Float, [dr.grad(local_eta) for local_eta, _ in material_grad_sources]
            ),
            material_coeff_sigma=GridScatter.flatten_slots(
                wt.Float, [dr.grad(local_sigma) for _, local_sigma in material_grad_sources]
            ),
            material_slot_count=len(material_grad_sources),
        )

    @staticmethod
    def vertex_transport_jvp(*, tape: DiffractionTape, scene: Scene, tx_pos, grid: Grid,
                             config: ResolvedTraceConfig, diff_gain_scale, total_length_weight: float,
                             vertex_tangent):
        buffers = DiffractionAD.transport_vertex_coeffs(
            tape=tape,
            scene=scene,
            tx_pos=tx_pos,
            grid=grid,
            config=config,
            diff_gain_scale=diff_gain_scale,
            total_length_weight=total_length_weight,
        )
        return TransportVertexKernel.launch_jvp_into(
            buffers=buffers,
            vertex_tangent=vertex_tangent,
            out_size=int(grid.n_cells),
            bounds=grid.bounds,
            cell_size=grid.cell_size,
            grid_shape=grid.grid_shape,
        )

    @staticmethod
    def vertex_transport_vjp(*, tape: DiffractionTape, scene: Scene, tx_pos, grid: Grid,
                             config: ResolvedTraceConfig, diff_gain_scale, total_length_weight: float,
                             upstream_component, n_vertices: int):
        buffers = DiffractionAD.transport_vertex_coeffs(
            tape=tape,
            scene=scene,
            tx_pos=tx_pos,
            grid=grid,
            config=config,
            diff_gain_scale=diff_gain_scale,
            total_length_weight=total_length_weight,
        )
        zero = dr.zeros(wt.Float, int(n_vertices))
        vertex_grad = TransportVertexKernel.launch_vjp_into(
            buffers=buffers,
            upstream_component=upstream_component,
            n_vertices=n_vertices,
            bounds=grid.bounds,
            cell_size=grid.cell_size,
            grid_shape=grid.grid_shape,
        )
        if int(dr.width(vertex_grad.x)) <= 0:
            return wt.Point3f(zero, zero, zero)
        return vertex_grad

__all__ = ["DiffractionAD"]
