from __future__ import annotations

from dataclasses import dataclass
import math

import drjit as dr
import witwin as wt

from ..grid import RadioMapGrid
from .. import diagnostics as rm_diag
from ....scene import Scene
from ....trace.diffraction.geometry import _point_source_field
from ....utils import scalar
from ....utils.constants import RAY_ORIGIN_BIAS
from ....utils.drjit_ops import ArrayInit, Concat
from ....utils.geometry import (
    point_in_triangle_3d,
    reflect_point_across_plane,
    surface_contains_point,
)
from ....utils.polarization import reflect_field_vector, vector_scale, vector_select
from ...orchestration import ResolvedTraceConfig
from . import common as mc_common
from . import diffraction as mc_diff

_MC_TX_TRANSPORT_FD_STEP = 1.0e-3


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


@dataclass(slots=True)
class LosTape:
    ray_dir: object
    cell_idx: object
    transport_ray_dir: object
    transport_blocker_prim_idx: object


@dataclass(slots=True)
class ReflectionTape:
    initial_ray_dir: object
    blocker_dist: object
    cell_idx: object
    depth: object
    prim_index_by_bounce: tuple[object, ...]
    transport_initial_ray_dir: object
    transport_depth: object
    transport_blocker_prim_idx: object
    transport_prim_index_by_bounce: tuple[object, ...]


@dataclass(slots=True)
class DiffractionTape:
    edge_index: object
    edge_fraction: object
    cone_sample: object
    cell_idx: object
    field_valid: object
    pole_safe: object
    dif_n_p: object
    dif_n_m: object
    sum_n_p: object
    sum_n_m: object


def empty_sparse_coeff_buffers():
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


def empty_los_tape():
    return LosTape(
        ray_dir=ArrayInit.empty_vector3(),
        cell_idx=dr.zeros(wt.UInt32, 0),
        transport_ray_dir=ArrayInit.empty_vector3(),
        transport_blocker_prim_idx=dr.zeros(wt.Int32, 0),
    )


def empty_reflection_tape(max_bounces: int):
    zero_int = dr.zeros(wt.Int32, 0)
    return ReflectionTape(
        initial_ray_dir=ArrayInit.empty_vector3(),
        blocker_dist=dr.zeros(wt.Float, 0),
        cell_idx=dr.zeros(wt.UInt32, 0),
        depth=zero_int,
        prim_index_by_bounce=tuple(dr.zeros(wt.Int32, 0) for _ in range(max(0, int(max_bounces)))),
        transport_initial_ray_dir=ArrayInit.empty_vector3(),
        transport_depth=zero_int,
        transport_blocker_prim_idx=zero_int,
        transport_prim_index_by_bounce=tuple(
            dr.zeros(wt.Int32, 0) for _ in range(max(0, int(max_bounces)))
        ),
    )


def empty_diffraction_tape():
    zero_float = dr.zeros(wt.Float, 0)
    return DiffractionTape(
        edge_index=dr.zeros(wt.Int32, 0),
        edge_fraction=zero_float,
        cone_sample=zero_float,
        cell_idx=dr.zeros(wt.UInt32, 0),
        field_valid=dr.zeros(wt.Bool, 0),
        pole_safe=dr.zeros(wt.Bool, 0),
        dif_n_p=zero_float,
        dif_n_m=zero_float,
        sum_n_p=zero_float,
        sum_n_m=zero_float,
    )


def flatten_slot_arrays(dtype, slot_arrays):
    if len(slot_arrays) <= 0:
        return dr.zeros(dtype, 0)
    return Concat.arrays(dtype, slot_arrays)


def _transport_repeat_tangent(value, width: int):
    return dr.repeat(wt.Float(value), width)


def _tent_axis_support(coord, coord_min: float, cell_size: float, n_cells: int):
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


def _tent_splat_to_grid(*, grid: RadioMapGrid, coord_0, coord_1, power, active):
    out = dr.zeros(wt.Float, int(grid.n_cells))
    if int(dr.width(power)) <= 0:
        return out
    nx, ny = int(grid.grid_shape[0]), int(grid.grid_shape[1])
    support_x = _tent_axis_support(coord_0, float(grid.bounds[0][0]), float(grid.cell_size[0]), nx)
    support_y = _tent_axis_support(coord_1, float(grid.bounds[1][0]), float(grid.cell_size[1]), ny)

    for x_idx, x_weight, x_valid in (
        (support_x["base_idx"], support_x["base_weight"], support_x["base_valid"]),
        (support_x["next_idx"], support_x["next_weight"], support_x["next_valid"]),
    ):
        for y_idx, y_weight, y_valid in (
            (support_y["base_idx"], support_y["base_weight"], support_y["base_valid"]),
            (support_y["next_idx"], support_y["next_weight"], support_y["next_valid"]),
        ):
            lane_active = active & x_valid & y_valid
            weight = x_weight * y_weight
            lane_active = lane_active & (weight != wt.Float(0.0))
            cell_idx = wt.UInt32(y_idx * wt.Int32(nx) + x_idx)
            dr.scatter_reduce(
                dr.ReduceOp.Add,
                out,
                power * weight,
                cell_idx,
                lane_active,
            )
    return out


def _scatter_power_to_grid(*, grid: RadioMapGrid, coord_0, coord_1, power, active):
    out = dr.zeros(wt.Float, int(grid.n_cells))
    if int(dr.width(power)) <= 0:
        return out
    cell_idx = mc_common._axis_aligned_cell_index(
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


def local_tx_lanes(tx_pos, width: int):
    return wt.Point3f(
        dr.full(wt.Float, float(scalar(dr.detach(tx_pos.x))), width),
        dr.full(wt.Float, float(scalar(dr.detach(tx_pos.y))), width),
        dr.full(wt.Float, float(scalar(dr.detach(tx_pos.z))), width),
    )


def local_vertex_point(scene: Scene, indices):
    safe_idx = wt.UInt32(dr.select(indices >= 0, indices, wt.Int32(0)))
    return wt.Point3f(
        dr.gather(wt.Float, dr.detach(scene.vertices.x), safe_idx),
        dr.gather(wt.Float, dr.detach(scene.vertices.y), safe_idx),
        dr.gather(wt.Float, dr.detach(scene.vertices.z), safe_idx),
    )


def fixed_primitive_blocker_distance(scene: Scene, *, ray_origin, ray_dir, prim_idx):
    width = int(dr.width(prim_idx))
    if width <= 0:
        return dr.zeros(wt.Float, 0)
    face_x = wt.Int32(scene.faces.x)
    face_y = wt.Int32(scene.faces.y)
    face_z = wt.Int32(scene.faces.z)
    active = prim_idx >= 0
    safe_prim_idx = wt.UInt32(dr.select(active, prim_idx, wt.Int32(0)))
    v_idx0 = dr.gather(wt.Int32, face_x, safe_prim_idx)
    v_idx1 = dr.gather(wt.Int32, face_y, safe_prim_idx)
    v_idx2 = dr.gather(wt.Int32, face_z, safe_prim_idx)
    local_v0 = local_vertex_point(scene, v_idx0)
    local_v1 = local_vertex_point(scene, v_idx1)
    local_v2 = local_vertex_point(scene, v_idx2)
    face_normal = dr.normalize(dr.cross(local_v1 - local_v0, local_v2 - local_v0))
    denominator = dr.dot(ray_dir, face_normal)
    safe_denominator = denominator + dr.mulsign(wt.Float(1.0e-6), denominator)
    hit_t = dr.dot(local_v0 - ray_origin, face_normal) / safe_denominator
    hit_p = ray_origin + ray_dir * hit_t
    tri_data = getattr(scene, "tri_data_gpu", None)
    if (
        isinstance(tri_data, dict)
        and "surface_group_size" in tri_data
        and "surface_group_members" in tri_data
        and int(tri_data.get("surface_max_group_size", 0)) > 0
    ):
        tri_v0 = local_vertex_point(scene, face_x)
        tri_v1 = local_vertex_point(scene, face_y)
        tri_v2 = local_vertex_point(scene, face_z)
        triangle_hit = surface_contains_point(
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
        triangle_hit = point_in_triangle_3d(hit_p, local_v0, local_v1, local_v2)
    valid = (
        active
        & (dr.abs(denominator) > wt.Float(1.0e-6))
        & (hit_t > wt.Float(RAY_ORIGIN_BIAS))
        & triangle_hit
    )
    return dr.select(valid, hit_t, wt.Float(1.0e10))


def local_material_support(
    prim_idx,
    *,
    scene: Scene,
    override_material,
    use_scene_materials: bool,
    default_eta_r: float,
    default_sigma: float,
    gain: float,
):
    width = int(dr.width(prim_idx))
    if override_material is not None:
        return {
            "eta_r": dr.full(wt.Float, float(override_material["relative_permittivity"]), width),
            "sigma": dr.full(wt.Float, float(override_material["conductivity"]), width),
            "gain": dr.full(wt.Float, float(override_material["gain"]), width),
            "use_fresnel": dr.full(wt.Bool, True, width),
            "material_idx": dr.full(wt.Int32, -1, width),
            "local_eta": None,
            "local_sigma": None,
        }

    tri_data = getattr(scene, "tri_data_gpu", None)
    if not use_scene_materials or not isinstance(tri_data, dict):
        return {
            "eta_r": dr.full(wt.Float, float(default_eta_r), width),
            "sigma": dr.full(wt.Float, float(default_sigma), width),
            "gain": dr.full(wt.Float, float(gain), width),
            "use_fresnel": dr.full(wt.Bool, True, width),
            "material_idx": dr.full(wt.Int32, -1, width),
            "local_eta": None,
            "local_sigma": None,
        }

    safe_idx = wt.UInt32(dr.select(prim_idx >= 0, prim_idx, wt.Int32(0)))
    specified = dr.gather(wt.Bool, tri_data["material_specified"], safe_idx) & (prim_idx >= 0)
    local_eta = dr.gather(wt.Float, dr.detach(tri_data["material_eps_r"]), safe_idx)
    local_sigma = dr.gather(wt.Float, dr.detach(tri_data["material_sigma_e"]), safe_idx)
    dr.enable_grad(local_eta, local_sigma)
    return {
        "eta_r": dr.select(specified, local_eta, wt.Float(default_eta_r)),
        "sigma": dr.select(specified, local_sigma, wt.Float(default_sigma)),
        "gain": dr.full(wt.Float, float(gain), width),
        "use_fresnel": dr.full(wt.Bool, True, width),
        "material_idx": dr.select(specified, wt.Int32(prim_idx), wt.Int32(-1)),
        "local_eta": local_eta,
        "local_sigma": local_sigma,
    }


def diffraction_edge_support_arrays(scene: Scene):
    edge_v0 = []
    edge_v1 = []
    face0_third = []
    face1_third = []
    face0_prim = []
    face1_prim = []
    face_x = scene.faces.x
    face_y = scene.faces.y
    face_z = scene.faces.z

    def _third_vertex(face_idx: int, edge_pair: tuple[int, int]) -> int:
        vertices = (
            int(face_x[face_idx]),
            int(face_y[face_idx]),
            int(face_z[face_idx]),
        )
        for vertex_idx in vertices:
            if vertex_idx not in edge_pair:
                return int(vertex_idx)
        return -1

    for edge in scene.vertical_edges:
        v0_idx, v1_idx = (int(edge.vertex_indices[0]), int(edge.vertex_indices[1]))
        edge_pair = (v0_idx, v1_idx)
        adjacent_faces = tuple(int(face) for face in (edge.adjacent_faces or ()))
        face0_idx = adjacent_faces[0] if len(adjacent_faces) > 0 else -1
        face1_idx = adjacent_faces[1] if len(adjacent_faces) > 1 else -1
        edge_v0.append(v0_idx)
        edge_v1.append(v1_idx)
        face0_prim.append(face0_idx)
        face1_prim.append(face1_idx)
        face0_third.append(_third_vertex(face0_idx, edge_pair) if face0_idx >= 0 else -1)
        face1_third.append(_third_vertex(face1_idx, edge_pair) if face1_idx >= 0 else -1)

    return {
        "edge_v0": wt.Int32(*edge_v0) if edge_v0 else dr.zeros(wt.Int32, 0),
        "edge_v1": wt.Int32(*edge_v1) if edge_v1 else dr.zeros(wt.Int32, 0),
        "face0_third": wt.Int32(*face0_third) if face0_third else dr.zeros(wt.Int32, 0),
        "face1_third": wt.Int32(*face1_third) if face1_third else dr.zeros(wt.Int32, 0),
        "face0_prim": wt.Int32(*face0_prim) if face0_prim else dr.zeros(wt.Int32, 0),
        "face1_prim": wt.Int32(*face1_prim) if face1_prim else dr.zeros(wt.Int32, 0),
    }


def extract_los_sparse_coefficients(
    *,
    tape: LosTape,
    tx_pos,
    grid: RadioMapGrid,
    config: ResolvedTraceConfig,
    solid_angle_per_ray: float,
    cell_area: float,
):
    width = int(dr.width(tape.cell_idx))
    if width <= 0:
        return empty_sparse_coeff_buffers()

    local_tx = local_tx_lanes(tx_pos, width)
    dr.enable_grad(local_tx.x, local_tx.y, local_tx.z)
    plane_hit = mc_common._plane_hit_from_segment(
        ray_origin=local_tx,
        ray_dir=tape.ray_dir,
        blocker_dist=dr.full(wt.Float, 1.0e10, width),
        grid=grid,
        active=dr.full(wt.Bool, True, width),
    )
    field = _point_source_field(
        local_tx,
        wt.Complex2f(1.0, 0.0),
        plane_hit["target_pos"],
        config.wavelength,
        config.k,
    )
    field_vector = vector_scale(mc_common._monte_carlo_source_field_vector(tape.ray_dir), field)
    unfolded_distance = dr.norm(plane_hit["target_pos"] - local_tx)
    contribution = (
        rm_diag._vector_power(field_vector)
        * wt.Float(solid_angle_per_ray / cell_area)
        * unfolded_distance
        * unfolded_distance
        / dr.maximum(plane_hit["cos_theta"], wt.Float(1.0e-6))
    )
    dr.backward(dr.sum(contribution))
    return SparseCoeffBuffers(
        cell_idx=tape.cell_idx,
        tx_coeff_x=dr.grad(local_tx.x),
        tx_coeff_y=dr.grad(local_tx.y),
        tx_coeff_z=dr.grad(local_tx.z),
        vertex_indices=dr.zeros(wt.Int32, 0),
        vertex_coeff_x=dr.zeros(wt.Float, 0),
        vertex_coeff_y=dr.zeros(wt.Float, 0),
        vertex_coeff_z=dr.zeros(wt.Float, 0),
        vertex_slot_count=0,
        material_indices=dr.zeros(wt.Int32, 0),
        material_coeff_eps=dr.zeros(wt.Float, 0),
        material_coeff_sigma=dr.zeros(wt.Float, 0),
        material_slot_count=0,
    )


def los_tx_transport_jvp(
    *,
    tape: LosTape,
    tx_pos,
    grid: RadioMapGrid,
    config: ResolvedTraceConfig,
    solid_angle_per_ray: float,
    cell_area: float,
    tx_tangent,
):
    width = int(dr.width(tape.cell_idx))
    if width <= 0:
        return dr.zeros(wt.Float, int(grid.n_cells))
    local_tx = local_tx_lanes(tx_pos, width)
    dr.enable_grad(local_tx.x, local_tx.y, local_tx.z)
    plane_hit = mc_common._plane_hit_from_segment(
        ray_origin=local_tx,
        ray_dir=tape.ray_dir,
        blocker_dist=dr.full(wt.Float, 1.0e10, width),
        grid=grid,
        active=dr.full(wt.Bool, True, width),
    )
    field = _point_source_field(
        local_tx,
        wt.Complex2f(1.0, 0.0),
        plane_hit["target_pos"],
        config.wavelength,
        config.k,
    )
    field_vector = vector_scale(mc_common._monte_carlo_source_field_vector(tape.ray_dir), field)
    unfolded_distance = dr.norm(plane_hit["target_pos"] - local_tx)
    contribution = dr.detach(
        rm_diag._vector_power(field_vector)
        * wt.Float(solid_angle_per_ray / cell_area)
        * unfolded_distance
        * unfolded_distance
        / dr.maximum(plane_hit["cos_theta"], wt.Float(1.0e-6))
    )
    transport_map = _tent_splat_to_grid(
        grid=grid,
        coord_0=plane_hit["coord_0"],
        coord_1=plane_hit["coord_1"],
        power=contribution,
        active=dr.full(wt.Bool, True, width),
    )
    dr.set_grad(local_tx.x, _transport_repeat_tangent(tx_tangent.x, width))
    dr.set_grad(local_tx.y, _transport_repeat_tangent(tx_tangent.y, width))
    dr.set_grad(local_tx.z, _transport_repeat_tangent(tx_tangent.z, width))
    return wt.Float(dr.forward_to(transport_map))


def los_tx_transport_vjp(
    *,
    tape: LosTape,
    tx_pos,
    grid: RadioMapGrid,
    config: ResolvedTraceConfig,
    solid_angle_per_ray: float,
    cell_area: float,
    upstream_component,
):
    width = int(dr.width(tape.cell_idx))
    if width <= 0:
        zero = dr.zeros(wt.Float, 1)
        return wt.Point3f(zero, zero, zero)
    local_tx = local_tx_lanes(tx_pos, width)
    dr.enable_grad(local_tx.x, local_tx.y, local_tx.z)
    plane_hit = mc_common._plane_hit_from_segment(
        ray_origin=local_tx,
        ray_dir=tape.ray_dir,
        blocker_dist=dr.full(wt.Float, 1.0e10, width),
        grid=grid,
        active=dr.full(wt.Bool, True, width),
    )
    field = _point_source_field(
        local_tx,
        wt.Complex2f(1.0, 0.0),
        plane_hit["target_pos"],
        config.wavelength,
        config.k,
    )
    field_vector = vector_scale(mc_common._monte_carlo_source_field_vector(tape.ray_dir), field)
    unfolded_distance = dr.norm(plane_hit["target_pos"] - local_tx)
    contribution = dr.detach(
        rm_diag._vector_power(field_vector)
        * wt.Float(solid_angle_per_ray / cell_area)
        * unfolded_distance
        * unfolded_distance
        / dr.maximum(plane_hit["cos_theta"], wt.Float(1.0e-6))
    )
    transport_map = _tent_splat_to_grid(
        grid=grid,
        coord_0=plane_hit["coord_0"],
        coord_1=plane_hit["coord_1"],
        power=contribution,
        active=dr.full(wt.Bool, True, width),
    )
    loss = dr.sum(transport_map * wt.Float(upstream_component))
    dr.backward(loss)
    return wt.Point3f(
        dr.sum(dr.grad(local_tx.x)),
        dr.sum(dr.grad(local_tx.y)),
        dr.sum(dr.grad(local_tx.z)),
    )


def los_tx_transport_basis_maps(
    *,
    tape: LosTape,
    scene: Scene,
    tx_pos,
    grid: RadioMapGrid,
    config: ResolvedTraceConfig,
    solid_angle_per_ray: float,
    cell_area: float,
    transport_step: float = _MC_TX_TRANSPORT_FD_STEP,
):
    width = int(dr.width(tape.transport_blocker_prim_idx))
    zero_map = dr.zeros(wt.Float, int(grid.n_cells))
    if width <= 0:
        return {"x": zero_map, "y": zero_map, "z": zero_map}
    local_tx = local_tx_lanes(tx_pos, width)
    ray_dir = tape.transport_ray_dir
    blocker_prim_idx = tape.transport_blocker_prim_idx
    step_scalar = float(transport_step)
    step = wt.Float(step_scalar)

    def _replay(tx_lanes):
        blocker_dist = fixed_primitive_blocker_distance(
            scene,
            ray_origin=tx_lanes,
            ray_dir=ray_dir,
            prim_idx=blocker_prim_idx,
        )
        plane_hit = mc_common._plane_hit_from_segment(
            ray_origin=tx_lanes,
            ray_dir=ray_dir,
            blocker_dist=blocker_dist,
            grid=grid,
            active=dr.full(wt.Bool, True, width),
        )
        field = _point_source_field(
            tx_lanes,
            wt.Complex2f(1.0, 0.0),
            plane_hit["target_pos"],
            config.wavelength,
            config.k,
        )
        field_vector = vector_scale(mc_common._monte_carlo_source_field_vector(ray_dir), field)
        unfolded_distance = dr.norm(plane_hit["target_pos"] - tx_lanes)
        power = dr.detach(
            rm_diag._vector_power(field_vector)
            * wt.Float(solid_angle_per_ray / cell_area)
            * unfolded_distance
            * unfolded_distance
            / dr.maximum(plane_hit["cos_theta"], wt.Float(1.0e-6))
        )
        return plane_hit, power

    def _map_for_shift(dx: float, dy: float, dz: float):
        shifted_tx = wt.Point3f(local_tx.x + wt.Float(dx), local_tx.y + wt.Float(dy), local_tx.z + wt.Float(dz))
        shifted_plane_hit, shifted_power = _replay(shifted_tx)
        return _scatter_power_to_grid(
            grid=grid,
            coord_0=shifted_plane_hit["coord_0"],
            coord_1=shifted_plane_hit["coord_1"],
            power=shifted_power,
            active=shifted_plane_hit["valid"],
        )

    return {
        "x": (_map_for_shift(step_scalar, 0.0, 0.0) - _map_for_shift(-step_scalar, 0.0, 0.0)) / (wt.Float(2.0) * step),
        "y": (_map_for_shift(0.0, step_scalar, 0.0) - _map_for_shift(0.0, -step_scalar, 0.0)) / (wt.Float(2.0) * step),
        "z": (_map_for_shift(0.0, 0.0, step_scalar) - _map_for_shift(0.0, 0.0, -step_scalar)) / (wt.Float(2.0) * step),
    }


def extract_reflection_sparse_coefficients(
    *,
    tape: ReflectionTape,
    scene: Scene,
    tx_pos,
    grid: RadioMapGrid,
    config: ResolvedTraceConfig,
    solid_angle_per_ray: float,
    cell_area: float,
    material_omega,
):
    width = int(dr.width(tape.cell_idx))
    max_bounces = len(tape.prim_index_by_bounce)
    if width <= 0 or max_bounces <= 0:
        return empty_sparse_coeff_buffers()

    local_tx = local_tx_lanes(tx_pos, width)
    dr.enable_grad(local_tx.x, local_tx.y, local_tx.z)
    ray_dir = tape.initial_ray_dir
    ray_origin = local_tx
    cumulative_image_source = local_tx
    polarization_vec = mc_common._monte_carlo_source_field_vector(ray_dir)

    vertex_index_slots = []
    vertex_vars = []
    material_index_slots = []
    material_grad_sources = []
    face_x = wt.Int32(scene.faces.x)
    face_y = wt.Int32(scene.faces.y)
    face_z = wt.Int32(scene.faces.z)

    for bounce_slot in range(max_bounces):
        active_bounce = tape.depth > wt.Int32(bounce_slot)
        prim_idx = tape.prim_index_by_bounce[bounce_slot]
        safe_prim_idx = wt.UInt32(dr.select(active_bounce, prim_idx, wt.Int32(0)))

        v_idx0 = dr.select(active_bounce, dr.gather(wt.Int32, face_x, safe_prim_idx), wt.Int32(-1))
        v_idx1 = dr.select(active_bounce, dr.gather(wt.Int32, face_y, safe_prim_idx), wt.Int32(-1))
        v_idx2 = dr.select(active_bounce, dr.gather(wt.Int32, face_z, safe_prim_idx), wt.Int32(-1))
        local_v0 = local_vertex_point(scene, v_idx0)
        local_v1 = local_vertex_point(scene, v_idx1)
        local_v2 = local_vertex_point(scene, v_idx2)
        dr.enable_grad(
            local_v0.x,
            local_v0.y,
            local_v0.z,
            local_v1.x,
            local_v1.y,
            local_v1.z,
            local_v2.x,
            local_v2.y,
            local_v2.z,
        )
        vertex_index_slots.extend((v_idx0, v_idx1, v_idx2))
        vertex_vars.extend((local_v0, local_v1, local_v2))

        face_normal = dr.normalize(dr.cross(local_v1 - local_v0, local_v2 - local_v0))
        oriented_normal = dr.select(dr.dot(ray_dir, face_normal) > 0.0, -face_normal, face_normal)
        safe_denominator = dr.dot(ray_dir, face_normal) + dr.mulsign(
            wt.Float(1.0e-6),
            dr.dot(ray_dir, face_normal),
        )
        hit_t = dr.dot(local_v0 - ray_origin, face_normal) / safe_denominator
        hit_p = ray_origin + ray_dir * hit_t

        material_inputs = local_material_support(
            prim_idx,
            scene=scene,
            override_material=config.reflection_material,
            use_scene_materials=bool(config.use_scene_materials_for_reflection),
            default_eta_r=float(config.reflection_relative_permittivity),
            default_sigma=float(config.reflection_conductivity),
            gain=float(config.reflection_coef),
        )
        if material_inputs["local_eta"] is not None:
            material_index_slots.append(material_inputs["material_idx"])
            material_grad_sources.append((material_inputs["local_eta"], material_inputs["local_sigma"]))

        reflected_polarization = reflect_field_vector(
            polarization_vec,
            ray_dir,
            oriented_normal,
            eta_r=material_inputs["eta_r"],
            sigma=material_inputs["sigma"],
            omega=material_omega,
            gain=material_inputs["gain"],
        )
        reflected_dir = ray_dir - wt.Float(2.0) * dr.dot(ray_dir, oriented_normal) * oriented_normal
        cumulative_image_source = dr.select(
            active_bounce,
            reflect_point_across_plane(cumulative_image_source, hit_p, oriented_normal),
            cumulative_image_source,
        )
        ray_origin = dr.select(active_bounce, hit_p + reflected_dir * wt.Float(1.0e-4), ray_origin)
        ray_dir = dr.select(active_bounce, reflected_dir, ray_dir)
        polarization_vec = vector_select(active_bounce, reflected_polarization, polarization_vec)

    plane_hit = mc_common._plane_hit_from_segment(
        ray_origin=ray_origin,
        ray_dir=ray_dir,
        blocker_dist=tape.blocker_dist,
        grid=grid,
        active=dr.full(wt.Bool, True, width),
    )
    unfolded_distance = dr.norm(plane_hit["target_pos"] - cumulative_image_source)
    field = _point_source_field(
        cumulative_image_source,
        wt.Complex2f(1.0, 0.0),
        plane_hit["target_pos"],
        config.wavelength,
        config.k,
    )
    field_vector = vector_scale(polarization_vec, field)
    contribution = (
        rm_diag._vector_power(field_vector)
        * wt.Float(solid_angle_per_ray / cell_area)
        * unfolded_distance
        * unfolded_distance
        / dr.maximum(plane_hit["cos_theta"], wt.Float(1.0e-6))
    )
    dr.backward(dr.sum(contribution))

    return SparseCoeffBuffers(
        cell_idx=tape.cell_idx,
        tx_coeff_x=dr.grad(local_tx.x),
        tx_coeff_y=dr.grad(local_tx.y),
        tx_coeff_z=dr.grad(local_tx.z),
        vertex_indices=flatten_slot_arrays(wt.Int32, vertex_index_slots),
        vertex_coeff_x=flatten_slot_arrays(wt.Float, [dr.grad(vertex_var.x) for vertex_var in vertex_vars]),
        vertex_coeff_y=flatten_slot_arrays(wt.Float, [dr.grad(vertex_var.y) for vertex_var in vertex_vars]),
        vertex_coeff_z=flatten_slot_arrays(wt.Float, [dr.grad(vertex_var.z) for vertex_var in vertex_vars]),
        vertex_slot_count=len(vertex_vars),
        material_indices=flatten_slot_arrays(wt.Int32, material_index_slots),
        material_coeff_eps=flatten_slot_arrays(wt.Float, [dr.grad(local_eta) for local_eta, _ in material_grad_sources]),
        material_coeff_sigma=flatten_slot_arrays(wt.Float, [dr.grad(local_sigma) for _, local_sigma in material_grad_sources]),
        material_slot_count=len(material_grad_sources),
    )


def reflection_tx_transport_jvp(
    *,
    tape: ReflectionTape,
    scene: Scene,
    tx_pos,
    grid: RadioMapGrid,
    config: ResolvedTraceConfig,
    solid_angle_per_ray: float,
    cell_area: float,
    material_omega,
    tx_tangent,
):
    width = int(dr.width(tape.cell_idx))
    max_bounces = len(tape.prim_index_by_bounce)
    if width <= 0 or max_bounces <= 0:
        return dr.zeros(wt.Float, int(grid.n_cells))

    local_tx = local_tx_lanes(tx_pos, width)
    dr.enable_grad(local_tx.x, local_tx.y, local_tx.z)
    ray_dir = tape.initial_ray_dir
    ray_origin = local_tx
    cumulative_image_source = local_tx
    polarization_vec = mc_common._monte_carlo_source_field_vector(ray_dir)
    face_x = wt.Int32(scene.faces.x)
    face_y = wt.Int32(scene.faces.y)
    face_z = wt.Int32(scene.faces.z)

    for bounce_slot in range(max_bounces):
        active_bounce = tape.depth > wt.Int32(bounce_slot)
        prim_idx = tape.prim_index_by_bounce[bounce_slot]
        safe_prim_idx = wt.UInt32(dr.select(active_bounce, prim_idx, wt.Int32(0)))
        v_idx0 = dr.select(active_bounce, dr.gather(wt.Int32, face_x, safe_prim_idx), wt.Int32(-1))
        v_idx1 = dr.select(active_bounce, dr.gather(wt.Int32, face_y, safe_prim_idx), wt.Int32(-1))
        v_idx2 = dr.select(active_bounce, dr.gather(wt.Int32, face_z, safe_prim_idx), wt.Int32(-1))
        local_v0 = local_vertex_point(scene, v_idx0)
        local_v1 = local_vertex_point(scene, v_idx1)
        local_v2 = local_vertex_point(scene, v_idx2)
        face_normal = dr.normalize(dr.cross(local_v1 - local_v0, local_v2 - local_v0))
        oriented_normal = dr.select(dr.dot(ray_dir, face_normal) > 0.0, -face_normal, face_normal)
        safe_denominator = dr.dot(ray_dir, face_normal) + dr.mulsign(
            wt.Float(1.0e-6),
            dr.dot(ray_dir, face_normal),
        )
        hit_t = dr.dot(local_v0 - ray_origin, face_normal) / safe_denominator
        hit_p = ray_origin + ray_dir * hit_t
        material_inputs = local_material_support(
            prim_idx,
            scene=scene,
            override_material=config.reflection_material,
            use_scene_materials=bool(config.use_scene_materials_for_reflection),
            default_eta_r=float(config.reflection_relative_permittivity),
            default_sigma=float(config.reflection_conductivity),
            gain=float(config.reflection_coef),
        )
        reflected_polarization = reflect_field_vector(
            polarization_vec,
            ray_dir,
            oriented_normal,
            eta_r=material_inputs["eta_r"],
            sigma=material_inputs["sigma"],
            omega=material_omega,
            gain=material_inputs["gain"],
        )
        reflected_dir = ray_dir - wt.Float(2.0) * dr.dot(ray_dir, oriented_normal) * oriented_normal
        cumulative_image_source = dr.select(
            active_bounce,
            reflect_point_across_plane(cumulative_image_source, hit_p, oriented_normal),
            cumulative_image_source,
        )
        ray_origin = dr.select(active_bounce, hit_p + reflected_dir * wt.Float(1.0e-4), ray_origin)
        ray_dir = dr.select(active_bounce, reflected_dir, ray_dir)
        polarization_vec = vector_select(active_bounce, reflected_polarization, polarization_vec)

    plane_hit = mc_common._plane_hit_from_segment(
        ray_origin=ray_origin,
        ray_dir=ray_dir,
        blocker_dist=tape.blocker_dist,
        grid=grid,
        active=dr.full(wt.Bool, True, width),
    )
    unfolded_distance = dr.norm(plane_hit["target_pos"] - cumulative_image_source)
    field = _point_source_field(
        cumulative_image_source,
        wt.Complex2f(1.0, 0.0),
        plane_hit["target_pos"],
        config.wavelength,
        config.k,
    )
    field_vector = vector_scale(polarization_vec, field)
    contribution = dr.detach(
        rm_diag._vector_power(field_vector)
        * wt.Float(solid_angle_per_ray / cell_area)
        * unfolded_distance
        * unfolded_distance
        / dr.maximum(plane_hit["cos_theta"], wt.Float(1.0e-6))
    )
    transport_map = _tent_splat_to_grid(
        grid=grid,
        coord_0=plane_hit["coord_0"],
        coord_1=plane_hit["coord_1"],
        power=contribution,
        active=dr.full(wt.Bool, True, width),
    )
    dr.set_grad(local_tx.x, _transport_repeat_tangent(tx_tangent.x, width))
    dr.set_grad(local_tx.y, _transport_repeat_tangent(tx_tangent.y, width))
    dr.set_grad(local_tx.z, _transport_repeat_tangent(tx_tangent.z, width))
    return wt.Float(dr.forward_to(transport_map))


def reflection_tx_transport_vjp(
    *,
    tape: ReflectionTape,
    scene: Scene,
    tx_pos,
    grid: RadioMapGrid,
    config: ResolvedTraceConfig,
    solid_angle_per_ray: float,
    cell_area: float,
    material_omega,
    upstream_component,
):
    width = int(dr.width(tape.cell_idx))
    max_bounces = len(tape.prim_index_by_bounce)
    if width <= 0 or max_bounces <= 0:
        zero = dr.zeros(wt.Float, 1)
        return wt.Point3f(zero, zero, zero)

    local_tx = local_tx_lanes(tx_pos, width)
    dr.enable_grad(local_tx.x, local_tx.y, local_tx.z)
    ray_dir = tape.initial_ray_dir
    ray_origin = local_tx
    cumulative_image_source = local_tx
    polarization_vec = mc_common._monte_carlo_source_field_vector(ray_dir)
    face_x = wt.Int32(scene.faces.x)
    face_y = wt.Int32(scene.faces.y)
    face_z = wt.Int32(scene.faces.z)

    for bounce_slot in range(max_bounces):
        active_bounce = tape.depth > wt.Int32(bounce_slot)
        prim_idx = tape.prim_index_by_bounce[bounce_slot]
        safe_prim_idx = wt.UInt32(dr.select(active_bounce, prim_idx, wt.Int32(0)))
        v_idx0 = dr.select(active_bounce, dr.gather(wt.Int32, face_x, safe_prim_idx), wt.Int32(-1))
        v_idx1 = dr.select(active_bounce, dr.gather(wt.Int32, face_y, safe_prim_idx), wt.Int32(-1))
        v_idx2 = dr.select(active_bounce, dr.gather(wt.Int32, face_z, safe_prim_idx), wt.Int32(-1))
        local_v0 = local_vertex_point(scene, v_idx0)
        local_v1 = local_vertex_point(scene, v_idx1)
        local_v2 = local_vertex_point(scene, v_idx2)
        face_normal = dr.normalize(dr.cross(local_v1 - local_v0, local_v2 - local_v0))
        oriented_normal = dr.select(dr.dot(ray_dir, face_normal) > 0.0, -face_normal, face_normal)
        safe_denominator = dr.dot(ray_dir, face_normal) + dr.mulsign(
            wt.Float(1.0e-6),
            dr.dot(ray_dir, face_normal),
        )
        hit_t = dr.dot(local_v0 - ray_origin, face_normal) / safe_denominator
        hit_p = ray_origin + ray_dir * hit_t
        material_inputs = local_material_support(
            prim_idx,
            scene=scene,
            override_material=config.reflection_material,
            use_scene_materials=bool(config.use_scene_materials_for_reflection),
            default_eta_r=float(config.reflection_relative_permittivity),
            default_sigma=float(config.reflection_conductivity),
            gain=float(config.reflection_coef),
        )
        reflected_polarization = reflect_field_vector(
            polarization_vec,
            ray_dir,
            oriented_normal,
            eta_r=material_inputs["eta_r"],
            sigma=material_inputs["sigma"],
            omega=material_omega,
            gain=material_inputs["gain"],
        )
        reflected_dir = ray_dir - wt.Float(2.0) * dr.dot(ray_dir, oriented_normal) * oriented_normal
        cumulative_image_source = dr.select(
            active_bounce,
            reflect_point_across_plane(cumulative_image_source, hit_p, oriented_normal),
            cumulative_image_source,
        )
        ray_origin = dr.select(active_bounce, hit_p + reflected_dir * wt.Float(1.0e-4), ray_origin)
        ray_dir = dr.select(active_bounce, reflected_dir, ray_dir)
        polarization_vec = vector_select(active_bounce, reflected_polarization, polarization_vec)

    plane_hit = mc_common._plane_hit_from_segment(
        ray_origin=ray_origin,
        ray_dir=ray_dir,
        blocker_dist=tape.blocker_dist,
        grid=grid,
        active=dr.full(wt.Bool, True, width),
    )
    unfolded_distance = dr.norm(plane_hit["target_pos"] - cumulative_image_source)
    field = _point_source_field(
        cumulative_image_source,
        wt.Complex2f(1.0, 0.0),
        plane_hit["target_pos"],
        config.wavelength,
        config.k,
    )
    field_vector = vector_scale(polarization_vec, field)
    contribution = dr.detach(
        rm_diag._vector_power(field_vector)
        * wt.Float(solid_angle_per_ray / cell_area)
        * unfolded_distance
        * unfolded_distance
        / dr.maximum(plane_hit["cos_theta"], wt.Float(1.0e-6))
    )
    transport_map = _tent_splat_to_grid(
        grid=grid,
        coord_0=plane_hit["coord_0"],
        coord_1=plane_hit["coord_1"],
        power=contribution,
        active=dr.full(wt.Bool, True, width),
    )
    loss = dr.sum(transport_map * wt.Float(upstream_component))
    dr.backward(loss)
    return wt.Point3f(
        dr.sum(dr.grad(local_tx.x)),
        dr.sum(dr.grad(local_tx.y)),
        dr.sum(dr.grad(local_tx.z)),
    )


def reflection_tx_transport_basis_maps(
    *,
    tape: ReflectionTape,
    scene: Scene,
    tx_pos,
    grid: RadioMapGrid,
    config: ResolvedTraceConfig,
    solid_angle_per_ray: float,
    cell_area: float,
    material_omega,
    transport_step: float = _MC_TX_TRANSPORT_FD_STEP,
):
    width = int(dr.width(tape.transport_blocker_prim_idx))
    max_bounces = len(tape.transport_prim_index_by_bounce)
    zero_map = dr.zeros(wt.Float, int(grid.n_cells))
    if width <= 0 or max_bounces <= 0:
        return {"x": zero_map, "y": zero_map, "z": zero_map}

    face_x = wt.Int32(scene.faces.x)
    face_y = wt.Int32(scene.faces.y)
    face_z = wt.Int32(scene.faces.z)
    base_tx = local_tx_lanes(tx_pos, width)

    def _replay(tx_lanes):
        ray_dir = tape.transport_initial_ray_dir
        ray_origin = tx_lanes
        cumulative_image_source = tx_lanes
        polarization_vec = mc_common._monte_carlo_source_field_vector(ray_dir)

        for bounce_slot in range(max_bounces):
            active_bounce = tape.transport_depth > wt.Int32(bounce_slot)
            prim_idx = tape.transport_prim_index_by_bounce[bounce_slot]
            safe_prim_idx = wt.UInt32(dr.select(active_bounce, prim_idx, wt.Int32(0)))
            v_idx0 = dr.select(active_bounce, dr.gather(wt.Int32, face_x, safe_prim_idx), wt.Int32(-1))
            v_idx1 = dr.select(active_bounce, dr.gather(wt.Int32, face_y, safe_prim_idx), wt.Int32(-1))
            v_idx2 = dr.select(active_bounce, dr.gather(wt.Int32, face_z, safe_prim_idx), wt.Int32(-1))
            local_v0 = local_vertex_point(scene, v_idx0)
            local_v1 = local_vertex_point(scene, v_idx1)
            local_v2 = local_vertex_point(scene, v_idx2)
            face_normal = dr.normalize(dr.cross(local_v1 - local_v0, local_v2 - local_v0))
            oriented_normal = dr.select(dr.dot(ray_dir, face_normal) > 0.0, -face_normal, face_normal)
            safe_denominator = dr.dot(ray_dir, face_normal) + dr.mulsign(
                wt.Float(1.0e-6),
                dr.dot(ray_dir, face_normal),
            )
            hit_t = dr.dot(local_v0 - ray_origin, face_normal) / safe_denominator
            hit_p = ray_origin + ray_dir * hit_t
            material_inputs = local_material_support(
                prim_idx,
                scene=scene,
                override_material=config.reflection_material,
                use_scene_materials=bool(config.use_scene_materials_for_reflection),
                default_eta_r=float(config.reflection_relative_permittivity),
                default_sigma=float(config.reflection_conductivity),
                gain=float(config.reflection_coef),
            )
            reflected_polarization = reflect_field_vector(
                polarization_vec,
                ray_dir,
                oriented_normal,
                eta_r=material_inputs["eta_r"],
                sigma=material_inputs["sigma"],
                omega=material_omega,
                gain=material_inputs["gain"],
            )
            reflected_dir = ray_dir - wt.Float(2.0) * dr.dot(ray_dir, oriented_normal) * oriented_normal
            cumulative_image_source = dr.select(
                active_bounce,
                reflect_point_across_plane(cumulative_image_source, hit_p, oriented_normal),
                cumulative_image_source,
            )
            ray_origin = dr.select(active_bounce, hit_p + reflected_dir * wt.Float(1.0e-4), ray_origin)
            ray_dir = dr.select(active_bounce, reflected_dir, ray_dir)
            polarization_vec = vector_select(active_bounce, reflected_polarization, polarization_vec)

        blocker_dist = fixed_primitive_blocker_distance(
            scene,
            ray_origin=ray_origin,
            ray_dir=ray_dir,
            prim_idx=tape.transport_blocker_prim_idx,
        )
        plane_hit = mc_common._plane_hit_from_segment(
            ray_origin=ray_origin,
            ray_dir=ray_dir,
            blocker_dist=blocker_dist,
            grid=grid,
            active=dr.full(wt.Bool, True, width),
        )
        unfolded_distance = dr.norm(plane_hit["target_pos"] - cumulative_image_source)
        field = _point_source_field(
            cumulative_image_source,
            wt.Complex2f(1.0, 0.0),
            plane_hit["target_pos"],
            config.wavelength,
            config.k,
        )
        field_vector = vector_scale(polarization_vec, field)
        power = dr.detach(
            rm_diag._vector_power(field_vector)
            * wt.Float(solid_angle_per_ray / cell_area)
            * unfolded_distance
            * unfolded_distance
            / dr.maximum(plane_hit["cos_theta"], wt.Float(1.0e-6))
        )
        return plane_hit, power

    step_scalar = float(transport_step)
    step = wt.Float(step_scalar)

    def _map_for_shift(dx: float, dy: float, dz: float):
        shifted_tx = wt.Point3f(base_tx.x + wt.Float(dx), base_tx.y + wt.Float(dy), base_tx.z + wt.Float(dz))
        shifted_plane_hit, shifted_power = _replay(shifted_tx)
        return _scatter_power_to_grid(
            grid=grid,
            coord_0=shifted_plane_hit["coord_0"],
            coord_1=shifted_plane_hit["coord_1"],
            power=shifted_power,
            active=shifted_plane_hit["valid"],
        )

    return {
        "x": (_map_for_shift(step_scalar, 0.0, 0.0) - _map_for_shift(-step_scalar, 0.0, 0.0)) / (wt.Float(2.0) * step),
        "y": (_map_for_shift(0.0, step_scalar, 0.0) - _map_for_shift(0.0, -step_scalar, 0.0)) / (wt.Float(2.0) * step),
        "z": (_map_for_shift(0.0, 0.0, step_scalar) - _map_for_shift(0.0, 0.0, -step_scalar)) / (wt.Float(2.0) * step),
    }


def extract_diffraction_sparse_coefficients(
    *,
    tape: DiffractionTape,
    scene: Scene,
    tx_pos,
    grid: RadioMapGrid,
    config: ResolvedTraceConfig,
    diffraction_path_gain_scale,
    total_length_weight: float,
):
    width = int(dr.width(tape.cell_idx))
    if width <= 0:
        return empty_sparse_coeff_buffers()

    local_tx = local_tx_lanes(tx_pos, width)
    dr.enable_grad(local_tx.x, local_tx.y, local_tx.z)
    support = diffraction_edge_support_arrays(scene)
    edge_data = mc_diff._gather_diffraction_edge_subset(
        scene,
        tape.edge_index,
        valid_mask=tape.edge_index >= 0,
    )
    safe_edge_idx = wt.UInt32(dr.select(tape.edge_index >= 0, tape.edge_index, wt.Int32(0)))

    edge_v0_idx = dr.gather(wt.Int32, support["edge_v0"], safe_edge_idx)
    edge_v1_idx = dr.gather(wt.Int32, support["edge_v1"], safe_edge_idx)
    face0_prim_idx = dr.gather(wt.Int32, support["face0_prim"], safe_edge_idx)
    face1_prim_idx = dr.gather(wt.Int32, support["face1_prim"], safe_edge_idx)

    edge_v0 = local_vertex_point(scene, edge_v0_idx)
    edge_v1 = local_vertex_point(scene, edge_v1_idx)
    dr.enable_grad(
        edge_v0.x,
        edge_v0.y,
        edge_v0.z,
        edge_v1.x,
        edge_v1.y,
        edge_v1.z,
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
    face0_normal = edge_data["n0"]
    face1_normal = edge_data["n_face_n"]
    wedge_n = edge_data["wedge_n"]
    exterior_angle = wedge_n * wt.Float(math.pi)

    face0_material = local_material_support(
        face0_prim_idx,
        scene=scene,
        override_material=config.diffraction_material,
        use_scene_materials=bool(config.use_scene_materials_for_diffraction),
        default_eta_r=5.0,
        default_sigma=0.0,
        gain=float(config.reflection_coef),
    )
    face1_material = local_material_support(
        face1_prim_idx,
        scene=scene,
        override_material=config.diffraction_material,
        use_scene_materials=bool(config.use_scene_materials_for_diffraction),
        default_eta_r=5.0,
        default_sigma=0.0,
        gain=float(config.reflection_coef),
    )

    flip = dr.dot(incident_dir, face0_normal) > 0.0
    oriented_edge_dir = dr.select(flip, -edge_dir, edge_dir)
    oriented_n0 = dr.select(flip, face1_normal, face0_normal)
    oriented_nn = dr.select(flip, face0_normal, face1_normal)
    oriented_face0_eta_r = dr.select(flip, face1_material["eta_r"], face0_material["eta_r"])
    oriented_face0_sigma = dr.select(flip, face1_material["sigma"], face0_material["sigma"])
    oriented_face0_gain = dr.select(flip, face1_material["gain"], face0_material["gain"])
    oriented_face0_use_fresnel = dr.select(flip, face1_material["use_fresnel"], face0_material["use_fresnel"])
    oriented_face1_eta_r = dr.select(flip, face0_material["eta_r"], face1_material["eta_r"])
    oriented_face1_sigma = dr.select(flip, face0_material["sigma"], face1_material["sigma"])
    oriented_face1_gain = dr.select(flip, face0_material["gain"], face1_material["gain"])
    oriented_face1_use_fresnel = dr.select(flip, face0_material["use_fresnel"], face1_material["use_fresnel"])

    face_sum = oriented_n0 + oriented_nn
    face_sum_norm = dr.norm(face_sum)
    offset_normal = dr.select(
        face_sum_norm > wt.Float(1.0e-6),
        face_sum / face_sum_norm,
        wt.Vector3f(0.0, 0.0, 0.0),
    )
    ko = mc_diff._sample_keller_cone(
        oriented_edge_dir,
        oriented_n0,
        oriented_nn,
        tape.cone_sample,
        incident_dir,
        lit_region=True,
    )
    ray_origin = mc_common._spawn_offset_ray_origin(diff_point, ko, offset_normal)
    plane_hit = mc_common._plane_hit_from_segment(
        ray_origin=ray_origin,
        ray_dir=ko,
        blocker_dist=dr.full(wt.Float, 1.0e10, width),
        grid=grid,
        active=dr.full(wt.Bool, True, width),
    )
    integration_weight = mc_diff._diffraction_integration_weight(
        edge_origin=edge_pos,
        edge_dir=oriented_edge_dir,
        n0=oriented_n0,
        source_pos=local_tx,
        diff_point=diff_point,
        k_world=ko,
        target_pos=plane_hit["target_pos"],
        plane_normal=mc_common._axis_unit_normal(str(grid.axis)),
    )
    field_power = mc_diff.diff_field._sampled_edge_diffraction_power_to_targets_mc(
        source_pos=local_tx,
        edge_dir=oriented_edge_dir,
        n0=oriented_n0,
        nn=oriented_nn,
        wedge_n=wedge_n,
        face0_eta_r=oriented_face0_eta_r,
        face0_sigma=oriented_face0_sigma,
        face0_gain=oriented_face0_gain,
        face0_use_fresnel=oriented_face0_use_fresnel,
        face1_eta_r=oriented_face1_eta_r,
        face1_sigma=oriented_face1_sigma,
        face1_gain=oriented_face1_gain,
        face1_use_fresnel=oriented_face1_use_fresnel,
        sampled_edge_pos=diff_point,
        target_pos=plane_hit["target_pos"],
        k=config.k,
        wavelength=config.wavelength,
        support_override={
            "field_valid": tape.field_valid,
            "pole_safe": tape.pole_safe,
            "dif_n_p": tape.dif_n_p,
            "dif_n_m": tape.dif_n_m,
            "sum_n_p": tape.sum_n_p,
            "sum_n_m": tape.sum_n_m,
        },
    )
    contribution = (
        field_power
        * diffraction_path_gain_scale
        * integration_weight
        * wt.Float(total_length_weight)
        * exterior_angle
    )
    dr.backward(dr.sum(contribution))

    material_index_slots = []
    material_grad_sources = []
    for material_support in (face0_material, face1_material):
        if material_support["local_eta"] is None:
            continue
        material_index_slots.append(material_support["material_idx"])
        material_grad_sources.append((material_support["local_eta"], material_support["local_sigma"]))

    vertex_vars = (edge_v0, edge_v1)
    vertex_indices = (edge_v0_idx, edge_v1_idx)
    return SparseCoeffBuffers(
        cell_idx=tape.cell_idx,
        tx_coeff_x=dr.grad(local_tx.x),
        tx_coeff_y=dr.grad(local_tx.y),
        tx_coeff_z=dr.grad(local_tx.z),
        vertex_indices=flatten_slot_arrays(wt.Int32, list(vertex_indices)),
        vertex_coeff_x=flatten_slot_arrays(wt.Float, [dr.grad(vertex_var.x) for vertex_var in vertex_vars]),
        vertex_coeff_y=flatten_slot_arrays(wt.Float, [dr.grad(vertex_var.y) for vertex_var in vertex_vars]),
        vertex_coeff_z=flatten_slot_arrays(wt.Float, [dr.grad(vertex_var.z) for vertex_var in vertex_vars]),
        vertex_slot_count=len(vertex_vars),
        material_indices=flatten_slot_arrays(wt.Int32, material_index_slots),
        material_coeff_eps=flatten_slot_arrays(wt.Float, [dr.grad(local_eta) for local_eta, _ in material_grad_sources]),
        material_coeff_sigma=flatten_slot_arrays(wt.Float, [dr.grad(local_sigma) for _, local_sigma in material_grad_sources]),
        material_slot_count=len(material_grad_sources),
    )


__all__ = [
    "DiffractionTape",
    "LosTape",
    "ReflectionTape",
    "SparseCoeffBuffers",
    "diffraction_edge_support_arrays",
    "empty_diffraction_tape",
    "empty_los_tape",
    "empty_reflection_tape",
    "empty_sparse_coeff_buffers",
    "extract_diffraction_sparse_coefficients",
    "extract_los_sparse_coefficients",
    "extract_reflection_sparse_coefficients",
]
