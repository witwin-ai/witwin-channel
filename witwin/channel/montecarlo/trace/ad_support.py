from __future__ import annotations

from dataclasses import dataclass

import drjit as dr
from witwin.channel.core.scene import Scene
from witwin.channel.montecarlo import types as wt
from witwin.channel.core.grid import Grid
from .. import grid_ops
from ..kernels.transport_grid import TransportGridKernel
from witwin.channel.core.numerics.arrays import scalar
from witwin.channel.core.numerics.constants import RAY_ORIGIN_BIAS
from witwin.channel.core.numerics import arrays
from witwin.channel.core import geometry
MC_TX_TRANSPORT_FD_STEP = 1.0e-3


@dataclass(slots=True)
class MaterialAD:
    """Per-primitive material with gradient-tracked eps_r / sigma for AD replay."""
    eta_r: object
    mu_r: object
    sigma: object
    gain: object
    use_fresnel: object
    material_idx: object


@dataclass(slots=True)
class SparseCoeffBuffers:
    cell_idx: object
    tx_coeff_x: object
    tx_coeff_y: object
    tx_coeff_z: object
    vertex_indices: object
    vertex_coeff_x: object
    vertex_coeff_y: object
    vertex_coeff_z: object
    vertex_slot_count: int
    material_indices: object
    material_coeff_eps: object
    material_coeff_sigma: object
    material_slot_count: int

    @staticmethod
    def empty():
        zero_float = dr.zeros(wt.Float, 0)
        zero_int = dr.zeros(wt.Int32, 0)
        zero_uint = dr.zeros(wt.UInt32, 0)
        return SparseCoeffBuffers(
            cell_idx=zero_uint,
            tx_coeff_x=zero_float,
            tx_coeff_y=zero_float,
            tx_coeff_z=zero_float,
            vertex_indices=zero_int,
            vertex_coeff_x=zero_float,
            vertex_coeff_y=zero_float,
            vertex_coeff_z=zero_float,
            vertex_slot_count=0,
            material_indices=zero_int,
            material_coeff_eps=zero_float,
            material_coeff_sigma=zero_float,
            material_slot_count=0,
        )


@dataclass(slots=True)
class TransportVertexCoeffBuffers:
    coord_0: object
    coord_1: object
    power: object
    active_mask: object
    vertex_indices: object
    coord_0_coeff_x: object
    coord_0_coeff_y: object
    coord_0_coeff_z: object
    coord_1_coeff_x: object
    coord_1_coeff_y: object
    coord_1_coeff_z: object
    vertex_slot_count: int

    @staticmethod
    def empty():
        zero_float = dr.zeros(wt.Float, 0)
        zero_int = dr.zeros(wt.Int32, 0)
        return TransportVertexCoeffBuffers(
            coord_0=zero_float,
            coord_1=zero_float,
            power=zero_float,
            active_mask=zero_int,
            vertex_indices=zero_int,
            coord_0_coeff_x=zero_float,
            coord_0_coeff_y=zero_float,
            coord_0_coeff_z=zero_float,
            coord_1_coeff_x=zero_float,
            coord_1_coeff_y=zero_float,
            coord_1_coeff_z=zero_float,
            vertex_slot_count=0,
        )

class ADContext:
    """Static namespace for Monte Carlo radiomap AD support helpers."""

    @staticmethod
    def tent_axis_support(coord, coord_min: float, cell_size: float, n_cells: int):
        scaled = (coord - wt.Float(coord_min)) / wt.Float(cell_size) - wt.Float(0.5)
        base_float = dr.floor(scaled)
        frac = scaled - base_float
        base = wt.Int32(base_float)
        next_idx = base + wt.Int32(1)
        w_base = wt.Float(1.0) - frac
        w_next = frac
        base_valid = (base >= 0) & (base < wt.Int32(int(n_cells)))
        next_valid = (next_idx >= 0) & (next_idx < wt.Int32(int(n_cells)))
        return {
            "base_idx": base,
            "next_idx": next_idx,
            "base_weight": w_base,
            "next_weight": w_next,
            "base_valid": base_valid,
            "next_valid": next_valid,
        }

    @staticmethod
    def finalize_tapes(
        *,
        collect_ad_tapes: bool,
        max_bounces: int,
        path_tape_store,
        diffraction_tape_store,
        los_tape=None,
    ) -> tuple:
        """Construct AD tapes from path and diffraction tape stores."""
        if not collect_ad_tapes:
            return None, None, None
        # Local imports break the path-layer import cycle: each tape lives
        # alongside its producer (diffraction.py / los.py / reflection.py),
        # all of which depend on ad_support.
        from .diffraction import DiffractionTape
        from .los import LosTape
        from .reflection import ReflectionTape

        if path_tape_store is None:
            los_tape = LosTape.empty() if los_tape is None else los_tape
            reflection_tape = ReflectionTape.empty(max_bounces)
        else:
            tape_payload = path_tape_store.finalize()
            if los_tape is None:
                los_tape = LosTape.from_payload(tape_payload["los"])
            reflection_tape = ReflectionTape.from_payload(tape_payload["reflection"])
        if diffraction_tape_store is None:
            diffraction_tape = DiffractionTape.empty()
        else:
            diffraction_tape = diffraction_tape_store.finalize()
        return los_tape, reflection_tape, diffraction_tape


class GridScatter:
    """Namespace for grid scatter operations."""

    @staticmethod
    def tent_splat(*, grid: Grid, coord_0, coord_1, power, active):
        return TransportGridKernel.tent_splat(
            coord_0=coord_0,
            coord_1=coord_1,
            power=power,
            active=active,
            bounds=grid.bounds,
            cell_size=grid.cell_size,
            grid_shape=grid.grid_shape,
        )

    @staticmethod
    def power(*, grid: Grid, coord_0, coord_1, power, active):
        out = dr.zeros(wt.Float, int(grid.n_cells))
        if int(dr.width(power)) <= 0:
            return out
        cell_idx = grid_ops.cell_index(
            grid=grid,
            coord_0=coord_0,
            coord_1=coord_1,
        )
        dr.scatter_reduce(
            dr.ReduceOp.Add,
            out,
            dr.select(active, power, wt.Float(0.0)),
            cell_idx,
            active,
        )
        return out

    @staticmethod
    def flatten_slots(dtype, slot_arrays):
        if len(slot_arrays) <= 0:
            return dr.zeros(dtype, 0)
        return arrays.concat_arrays(dtype, slot_arrays)


class TransportCoeffBuilder:
    """Build sparse transport coefficients for native JVP/VJP application."""

    @staticmethod
    def tent_support(*, grid: Grid, coord_0, coord_1, active):
        n_coord_0 = int(grid.grid_shape[0])
        n_coord_1 = int(grid.grid_shape[1])
        support_0 = ADContext.tent_axis_support(
            coord_0,
            float(grid.bounds[0][0]),
            float(grid.cell_size[0]),
            n_coord_0,
        )
        support_1 = ADContext.tent_axis_support(
            coord_1,
            float(grid.bounds[1][0]),
            float(grid.cell_size[1]),
            n_coord_1,
        )
        inv_cell_0 = wt.Float(1.0 / float(grid.cell_size[0]))
        inv_cell_1 = wt.Float(1.0 / float(grid.cell_size[1]))
        axis_0_slots = (
            (
                support_0["base_idx"],
                support_0["base_weight"],
                support_0["base_valid"],
                -inv_cell_0,
            ),
            (
                support_0["next_idx"],
                support_0["next_weight"],
                support_0["next_valid"],
                inv_cell_0,
            ),
        )
        axis_1_slots = (
            (
                support_1["base_idx"],
                support_1["base_weight"],
                support_1["base_valid"],
                -inv_cell_1,
            ),
            (
                support_1["next_idx"],
                support_1["next_weight"],
                support_1["next_valid"],
                inv_cell_1,
            ),
        )
        cell_idx_slots = []
        active_slots = []
        dweight_coord_0_slots = []
        dweight_coord_1_slots = []
        for idx_1, weight_1, valid_1, dweight_1 in axis_1_slots:
            for idx_0, weight_0, valid_0, dweight_0 in axis_0_slots:
                slot_active = active & valid_0 & valid_1
                weight = weight_0 * weight_1
                slot_active = slot_active & (weight != wt.Float(0.0))
                cell_idx_slots.append(
                    wt.UInt32(
                        dr.select(
                            slot_active,
                            idx_1 * wt.Int32(n_coord_0) + idx_0,
                            wt.Int32(0),
                        )
                    )
                )
                active_slots.append(slot_active)
                dweight_coord_0_slots.append(dweight_0 * weight_1)
                dweight_coord_1_slots.append(weight_0 * dweight_1)
        return {
            "cell_idx_slots": tuple(cell_idx_slots),
            "active_slots": tuple(active_slots),
            "dweight_coord_0_slots": tuple(dweight_coord_0_slots),
            "dweight_coord_1_slots": tuple(dweight_coord_1_slots),
        }

    @staticmethod
    def clear_point_grads(points) -> None:
        for point in points:
            width = int(dr.width(point.x))
            if width <= 0:
                continue
            zero = dr.zeros(wt.Float, width)
            dr.set_grad(point.x, zero)
            dr.set_grad(point.y, zero)
            dr.set_grad(point.z, zero)

    @staticmethod
    def clear_float_grads(arrays) -> None:
        for array in arrays:
            width = int(dr.width(array))
            if width <= 0:
                continue
            dr.set_grad(array, dr.zeros(wt.Float, width))

    @staticmethod
    def expand_vertex_coeffs(*, support, power, dcoord_0, dcoord_1):
        coeff_slots = []
        for slot_active, dweight_coord_0, dweight_coord_1 in zip(
            support["active_slots"],
            support["dweight_coord_0_slots"],
            support["dweight_coord_1_slots"],
            strict=True,
        ):
            coeff = power * (dweight_coord_0 * dcoord_0 + dweight_coord_1 * dcoord_1)
            coeff_slots.append(dr.select(slot_active, coeff, wt.Float(0.0)))
        return arrays.concat_floats(coeff_slots)

    @staticmethod
    def build_vertex_buffers(
        *,
        grid: Grid,
        state_factory,
        vertex_index_getter,
        vertex_var_getter,
    ) -> TransportVertexCoeffBuffers:
        base_state = state_factory()
        if base_state is None:
            return TransportVertexCoeffBuffers.empty()
        n_samples = int(dr.width(base_state["contribution"]))
        base_vertex_indices = tuple(vertex_index_getter(base_state))
        if n_samples <= 0 or len(base_vertex_indices) <= 0:
            return TransportVertexCoeffBuffers.empty()

        coord_0_coeff_x = []
        coord_0_coeff_y = []
        coord_0_coeff_z = []
        coord_1_coeff_x = []
        coord_1_coeff_y = []
        coord_1_coeff_z = []
        vertex_indices = []

        for slot_idx, vertex_idx in enumerate(base_vertex_indices):
            vertex_indices.append(wt.Int32(vertex_idx))
            axis_coeff = {}
            for axis_name in ("x", "y", "z"):
                axis_state = state_factory()
                if axis_state is None:
                    axis_coeff[axis_name] = (
                        dr.zeros(wt.Float, n_samples),
                        dr.zeros(wt.Float, n_samples),
                    )
                    continue
                vertex_var = tuple(vertex_var_getter(axis_state))[slot_idx]
                width = int(dr.width(vertex_var.x))
                zero = dr.zeros(wt.Float, width)
                dr.set_grad(vertex_var.x, zero)
                dr.set_grad(vertex_var.y, zero)
                dr.set_grad(vertex_var.z, zero)
                dr.set_grad(getattr(vertex_var, axis_name), dr.full(wt.Float, 1.0, width))
                dcoord_0, dcoord_1 = dr.forward_to(
                    axis_state["plane_hit"].coord_0,
                    axis_state["plane_hit"].coord_1,
                )
                axis_coeff[axis_name] = (wt.Float(dcoord_0), wt.Float(dcoord_1))
            coord_0_coeff_x.append(axis_coeff["x"][0])
            coord_0_coeff_y.append(axis_coeff["y"][0])
            coord_0_coeff_z.append(axis_coeff["z"][0])
            coord_1_coeff_x.append(axis_coeff["x"][1])
            coord_1_coeff_y.append(axis_coeff["y"][1])
            coord_1_coeff_z.append(axis_coeff["z"][1])

        return TransportVertexCoeffBuffers(
            coord_0=base_state["plane_hit"].coord_0,
            coord_1=base_state["plane_hit"].coord_1,
            power=base_state["contribution"],
            active_mask=wt.Int32(dr.select(base_state["plane_hit"].valid, wt.Int32(1), wt.Int32(0))),
            vertex_indices=GridScatter.flatten_slots(wt.Int32, vertex_indices),
            coord_0_coeff_x=GridScatter.flatten_slots(wt.Float, coord_0_coeff_x),
            coord_0_coeff_y=GridScatter.flatten_slots(wt.Float, coord_0_coeff_y),
            coord_0_coeff_z=GridScatter.flatten_slots(wt.Float, coord_0_coeff_z),
            coord_1_coeff_x=GridScatter.flatten_slots(wt.Float, coord_1_coeff_x),
            coord_1_coeff_y=GridScatter.flatten_slots(wt.Float, coord_1_coeff_y),
            coord_1_coeff_z=GridScatter.flatten_slots(wt.Float, coord_1_coeff_z),
            vertex_slot_count=len(base_vertex_indices),
        )


class SceneQuery:
    """Namespace for AD replay / scene query helpers."""

    @staticmethod
    def tx_lanes(tx_pos, width: int):
        return wt.Point3f(
            dr.full(wt.Float, float(scalar(dr.detach(tx_pos.x))), width),
            dr.full(wt.Float, float(scalar(dr.detach(tx_pos.y))), width),
            dr.full(wt.Float, float(scalar(dr.detach(tx_pos.z))), width),
        )

    @staticmethod
    def vertex_point(scene: Scene, indices):
        vertices = scene._merged_vertices()
        safe_idx = wt.UInt32(dr.select(indices >= 0, indices, wt.Int32(0)))
        return wt.Point3f(
            dr.gather(wt.Float, dr.detach(vertices.x), safe_idx),
            dr.gather(wt.Float, dr.detach(vertices.y), safe_idx),
            dr.gather(wt.Float, dr.detach(vertices.z), safe_idx),
        )

    @staticmethod
    def blocker_dist(scene: Scene, *, ray_origin, ray_dir, prim_idx):
        width = int(dr.width(prim_idx))
        if width <= 0:
            return dr.zeros(wt.Float, 0)
        faces = scene._merged_faces()
        face_x = wt.Int32(faces.x)
        face_y = wt.Int32(faces.y)
        face_z = wt.Int32(faces.z)
        active = prim_idx >= 0
        safe_prim_idx = wt.UInt32(dr.select(active, prim_idx, wt.Int32(0)))
        v_idx0 = dr.gather(wt.Int32, face_x, safe_prim_idx)
        v_idx1 = dr.gather(wt.Int32, face_y, safe_prim_idx)
        v_idx2 = dr.gather(wt.Int32, face_z, safe_prim_idx)
        local_v0 = SceneQuery.vertex_point(scene, v_idx0)
        local_v1 = SceneQuery.vertex_point(scene, v_idx1)
        local_v2 = SceneQuery.vertex_point(scene, v_idx2)
        face_normal = dr.normalize(dr.cross(local_v1 - local_v0, local_v2 - local_v0))
        denominator = dr.dot(ray_dir, face_normal)
        safe_denominator = denominator + dr.mulsign(wt.Float(1.0e-6), denominator)
        hit_t = dr.dot(local_v0 - ray_origin, face_normal) / safe_denominator
        hit_p = ray_origin + ray_dir * hit_t
        tri_data = scene._triangle_runtime()
        if (
            isinstance(tri_data, dict)
            and "surface_group_size" in tri_data
            and "surface_group_members" in tri_data
            and int(tri_data.get("surface_max_group_size", 0)) > 0
        ):
            tri_v0 = SceneQuery.vertex_point(scene, face_x)
            tri_v1 = SceneQuery.vertex_point(scene, face_y)
            tri_v2 = SceneQuery.vertex_point(scene, face_z)
            triangle_hit = geometry.surface_contains_point(
                hit_p,
                prim_idx,
                tri_v0,
                tri_v1,
                tri_v2,
                {
                    "group_size": tri_data["surface_group_size"],
                    "group_members": tri_data["surface_group_members"],
                    "max_group_size": int(tri_data["surface_max_group_size"]),
                },
            )
        else:
            triangle_hit = geometry.point_in_triangle_3d(hit_p, local_v0, local_v1, local_v2)
        valid = (
            active
            & (dr.abs(denominator) > wt.Float(1.0e-6))
            & (hit_t > wt.Float(RAY_ORIGIN_BIAS))
            & triangle_hit
        )
        return dr.select(valid, hit_t, wt.Float(1.0e10))

    @staticmethod
    def material(prim_idx, *, scene: Scene, gain: float) -> MaterialAD:
        width = int(dr.width(prim_idx))
        tri_data = scene._triangle_runtime()
        if not isinstance(tri_data, dict):
            raise RuntimeError(
                "Monte Carlo AD material resolution requires a scene material table. "
                "Attach witwin.core.Material to every scene structure."
            )

        safe_idx = wt.UInt32(dr.select(prim_idx >= 0, prim_idx, wt.Int32(0)))
        specified = dr.gather(wt.Bool, tri_data["material_specified"], safe_idx) & (prim_idx >= 0)
        eta_r = dr.gather(wt.Float, dr.detach(tri_data["material_eps_r"]), safe_idx)
        mu_r = dr.gather(wt.Float, dr.detach(tri_data["material_mu_r"]), safe_idx)
        sigma = dr.gather(wt.Float, dr.detach(tri_data["material_sigma_e"]), safe_idx)
        dr.enable_grad(eta_r, mu_r, sigma)
        return MaterialAD(
            eta_r=eta_r,
            mu_r=mu_r,
            sigma=sigma,
            gain=dr.full(wt.Float, float(gain), width),
            use_fresnel=dr.full(wt.Bool, True, width),
            material_idx=dr.select(specified, wt.Int32(prim_idx), wt.Int32(-1)),
        )


__all__ = [
    "ADContext",
    "MaterialAD",
    "SparseCoeffBuffers",
    "TransportVertexCoeffBuffers",
    "MC_TX_TRANSPORT_FD_STEP",
    "GridScatter",
    "TransportCoeffBuilder",
    "SceneQuery",
]
