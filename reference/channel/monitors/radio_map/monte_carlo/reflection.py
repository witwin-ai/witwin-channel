from __future__ import annotations

import math

import drjit as dr
import rayd
import witwin as wt

from .. import diagnostics as rm_diag
from ....trace.diffraction.geometry import _point_source_field, reflect_point_across_plane
from ....trace.materials import resolve_surface_material
from ....utils import scalar
from ....utils.constants import EPS, RAY_ORIGIN_BIAS
from ....utils.drjit_ops import Broadcast, broadcast_point
from ....utils.polarization import (
    reflect_field_vector,
    vector_scale,
    vector_select,
)
from . import common as mc_common
from . import diffraction as mc_diff


class PathTapeStore:
    def __init__(self, *, samples_per_tx: int, max_bounces: int) -> None:
        self.samples_per_tx = max(0, int(samples_per_tx))
        self.max_bounces = max(0, int(max_bounces))
        self.los_capacity = self.samples_per_tx
        self.reflection_capacity = self.samples_per_tx * self.max_bounces
        self.los_count = wt.UInt32(0)
        self.reflection_count = wt.UInt32(0)
        self.los_transport_count = wt.UInt32(0)
        self.reflection_transport_count = wt.UInt32(0)

        los_zero = dr.zeros(wt.Float, self.los_capacity)
        reflection_zero = dr.zeros(wt.Float, self.reflection_capacity)
        self.los_ray_dir = wt.Vector3f(los_zero, los_zero, los_zero)
        self.los_cell_idx = dr.zeros(wt.UInt32, self.los_capacity)
        self.los_transport_ray_dir = wt.Vector3f(los_zero, los_zero, los_zero)
        self.los_transport_blocker_prim_idx = dr.full(wt.Int32, -1, self.los_capacity)
        self.reflection_initial_ray_dir = wt.Vector3f(
            reflection_zero,
            reflection_zero,
            reflection_zero,
        )
        self.reflection_blocker_dist = dr.zeros(wt.Float, self.reflection_capacity)
        self.reflection_cell_idx = dr.zeros(wt.UInt32, self.reflection_capacity)
        self.reflection_depth = dr.zeros(wt.Int32, self.reflection_capacity)
        self.reflection_prim_by_bounce = tuple(
            dr.full(wt.Int32, -1, self.reflection_capacity)
            for _ in range(self.max_bounces)
        )
        self.reflection_transport_initial_ray_dir = wt.Vector3f(
            reflection_zero,
            reflection_zero,
            reflection_zero,
        )
        self.reflection_transport_depth = dr.zeros(wt.Int32, self.reflection_capacity)
        self.reflection_transport_blocker_prim_idx = dr.full(
            wt.Int32,
            -1,
            self.reflection_capacity,
        )
        self.reflection_transport_prim_by_bounce = tuple(
            dr.full(wt.Int32, -1, self.reflection_capacity)
            for _ in range(self.max_bounces)
        )

    def store_los(self, *, initial_ray_dir, cell_idx, active) -> None:
        if self.los_capacity <= 0:
            return
        slot = dr.scatter_inc(self.los_count, wt.UInt32(0), active)
        store_mask = active & (slot < wt.UInt32(self.los_capacity))
        dr.scatter(self.los_ray_dir, initial_ray_dir, slot, store_mask)
        dr.scatter(self.los_cell_idx, cell_idx, slot, store_mask)

    def store_los_transport(self, *, initial_ray_dir, blocker_prim_idx, active) -> None:
        if self.los_capacity <= 0:
            return
        slot = dr.scatter_inc(self.los_transport_count, wt.UInt32(0), active)
        store_mask = active & (slot < wt.UInt32(self.los_capacity))
        dr.scatter(self.los_transport_ray_dir, initial_ray_dir, slot, store_mask)
        dr.scatter(self.los_transport_blocker_prim_idx, blocker_prim_idx, slot, store_mask)

    def store_reflection(
        self,
        *,
        initial_ray_dir,
        blocker_dist,
        cell_idx,
        depth,
        prim_history,
        active,
    ) -> None:
        if self.reflection_capacity <= 0:
            return
        slot = dr.scatter_inc(self.reflection_count, wt.UInt32(0), active)
        store_mask = active & (slot < wt.UInt32(self.reflection_capacity))
        width = int(dr.width(cell_idx))
        depth_value = dr.full(wt.Int32, 0, width) + wt.Int32(depth)
        dr.scatter(self.reflection_initial_ray_dir, initial_ray_dir, slot, store_mask)
        dr.scatter(self.reflection_blocker_dist, blocker_dist, slot, store_mask)
        dr.scatter(self.reflection_cell_idx, cell_idx, slot, store_mask)
        dr.scatter(self.reflection_depth, depth_value, slot, store_mask)
        for bounce_slot, prim_index in enumerate(prim_history):
            dr.scatter(
                self.reflection_prim_by_bounce[bounce_slot],
                prim_index,
                slot,
                store_mask,
            )

    def store_reflection_transport(
        self,
        *,
        initial_ray_dir,
        depth,
        blocker_prim_idx,
        prim_history,
        active,
    ) -> None:
        if self.reflection_capacity <= 0:
            return
        slot = dr.scatter_inc(self.reflection_transport_count, wt.UInt32(0), active)
        store_mask = active & (slot < wt.UInt32(self.reflection_capacity))
        width = int(dr.width(blocker_prim_idx))
        depth_value = dr.full(wt.Int32, 0, width) + wt.Int32(depth)
        dr.scatter(self.reflection_transport_initial_ray_dir, initial_ray_dir, slot, store_mask)
        dr.scatter(self.reflection_transport_depth, depth_value, slot, store_mask)
        dr.scatter(self.reflection_transport_blocker_prim_idx, blocker_prim_idx, slot, store_mask)
        for bounce_slot, prim_index in enumerate(prim_history):
            dr.scatter(
                self.reflection_transport_prim_by_bounce[bounce_slot],
                prim_index,
                slot,
                store_mask,
            )

    def finalize(self):
        zero_float = dr.zeros(wt.Float, 0)
        zero_int = dr.zeros(wt.Int32, 0)
        zero_uint = dr.zeros(wt.UInt32, 0)
        los_count = int(scalar(self.los_count))
        reflection_count = int(scalar(self.reflection_count))
        los_transport_count = int(scalar(self.los_transport_count))
        reflection_transport_count = int(scalar(self.reflection_transport_count))
        los_idx = dr.arange(wt.UInt32, los_count)
        reflection_idx = dr.arange(wt.UInt32, reflection_count)
        los_transport_idx = dr.arange(wt.UInt32, los_transport_count)
        reflection_transport_idx = dr.arange(wt.UInt32, reflection_transport_count)
        return {
            "los": {
                "ray_dir": (
                    dr.gather(wt.Vector3f, self.los_ray_dir, los_idx)
                    if los_count > 0
                    else wt.Vector3f(zero_float, zero_float, zero_float)
                ),
                "cell_idx": (
                    dr.gather(wt.UInt32, self.los_cell_idx, los_idx)
                    if los_count > 0
                    else zero_uint
                ),
                "transport_ray_dir": (
                    dr.gather(wt.Vector3f, self.los_transport_ray_dir, los_transport_idx)
                    if los_transport_count > 0
                    else wt.Vector3f(zero_float, zero_float, zero_float)
                ),
                "transport_blocker_prim_idx": (
                    dr.gather(wt.Int32, self.los_transport_blocker_prim_idx, los_transport_idx)
                    if los_transport_count > 0
                    else zero_int
                ),
            },
            "reflection": {
                "initial_ray_dir": (
                    dr.gather(wt.Vector3f, self.reflection_initial_ray_dir, reflection_idx)
                    if reflection_count > 0
                    else wt.Vector3f(zero_float, zero_float, zero_float)
                ),
                "blocker_dist": (
                    dr.gather(wt.Float, self.reflection_blocker_dist, reflection_idx)
                    if reflection_count > 0
                    else zero_float
                ),
                "cell_idx": (
                    dr.gather(wt.UInt32, self.reflection_cell_idx, reflection_idx)
                    if reflection_count > 0
                    else zero_uint
                ),
                "depth": (
                    dr.gather(wt.Int32, self.reflection_depth, reflection_idx)
                    if reflection_count > 0
                    else zero_int
                ),
                "prim_index_by_bounce": tuple(
                    dr.gather(wt.Int32, prim_index, reflection_idx)
                    if reflection_count > 0
                    else zero_int
                    for prim_index in self.reflection_prim_by_bounce
                ),
                "transport_initial_ray_dir": (
                    dr.gather(
                        wt.Vector3f,
                        self.reflection_transport_initial_ray_dir,
                        reflection_transport_idx,
                    )
                    if reflection_transport_count > 0
                    else wt.Vector3f(zero_float, zero_float, zero_float)
                ),
                "transport_depth": (
                    dr.gather(wt.Int32, self.reflection_transport_depth, reflection_transport_idx)
                    if reflection_transport_count > 0
                    else zero_int
                ),
                "transport_blocker_prim_idx": (
                    dr.gather(
                        wt.Int32,
                        self.reflection_transport_blocker_prim_idx,
                        reflection_transport_idx,
                    )
                    if reflection_transport_count > 0
                    else zero_int
                ),
                "transport_prim_index_by_bounce": tuple(
                    dr.gather(wt.Int32, prim_index, reflection_transport_idx)
                    if reflection_transport_count > 0
                    else zero_int
                    for prim_index in self.reflection_transport_prim_by_bounce
                ),
            },
        }


@dr.syntax
def trace_reflection(
    *,
    scene,
    grid,
    tx_pos,
    ray_index,
    ray_dir,
    config,
    solid_angle_per_ray: float,
    cell_area: float,
    max_bounces: int,
    seed: int,
    rr_depth: int | None,
    rr_prob: float,
    stop_threshold_linear: float,
    material_omega,
    weighted_diagnostics,
    collect_diffraction_wedges: bool,
    direct_tx_diffraction_state_store=None,
    path_tape_store: PathTapeStore | None = None,
    loop_mode: str = "symbolic",
):
    # Trace reflected TX-emitted rays with the symbolic Dr.Jit loop used by the Monte Carlo radiomap path.
    current_batch_size = int(dr.width(ray_dir.x))
    initial_ray_dir = ray_dir
    ray_origin = broadcast_point(tx_pos, current_batch_size)
    cumulative_image_source = broadcast_point(tx_pos, current_batch_size)
    polarization_vec = mc_common._monte_carlo_source_field_vector(ray_dir)
    ray_path_length = rm_diag._zero_float(current_batch_size)
    active = dr.full(wt.Bool, True, current_batch_size)
    depth = wt.UInt32(0)
    los_hits = wt.UInt32(0)
    reflection_hits = wt.UInt32(0)
    max_bounces_u32 = wt.UInt32(int(max_bounces))
    prim_history = [dr.full(wt.Int32, -1, current_batch_size) for _ in range(int(max_bounces))]

    while dr.hint(
        active & (depth <= max_bounces_u32),
        mode=str(loop_mode),
        max_iterations=max(1, int(max_bounces) + 1),
        label="radio_map_mc_reflection",
        exclude=[
            scene,
            grid,
            tx_pos,
            config,
            material_omega,
            weighted_diagnostics,
            direct_tx_diffraction_state_store,
            path_tape_store,
        ],
    ):
        ray = rayd.Ray(ray_origin, ray_dir)
        si = scene.ray_intersect(ray, active=active)
        hit = si.is_valid() & active
        if dr.hint(collect_diffraction_wedges, mode="scalar"):
            direct_hit = hit & (depth == wt.UInt32(0))
            direct_tx_diffraction_state_store.store_from_hit_data(
                tx_pos=tx_pos,
                ray_directions=ray_dir,
                prim_index=si.prim_index,
                hit_p=si.p,
                hit_n=si.n,
                hit_geo_n=si.geo_n,
                hit=direct_hit,
                scene=scene,
                config=config,
            )
        blocker_dist = dr.select(hit, si.t, wt.Float(1.0e10))
        blocker_prim_idx = dr.select(hit, si.prim_index, wt.Int32(-1))
        if dr.hint(path_tape_store is not None, mode="scalar"):
            path_tape_store.store_los_transport(
                initial_ray_dir=initial_ray_dir,
                blocker_prim_idx=blocker_prim_idx,
                active=active & (depth == wt.UInt32(0)),
            )
            path_tape_store.store_reflection_transport(
                initial_ray_dir=initial_ray_dir,
                depth=depth,
                blocker_prim_idx=blocker_prim_idx,
                prim_history=prim_history,
                active=active & (depth > wt.UInt32(0)),
            )
        plane_hit = mc_common._plane_hit_from_segment(
            ray_origin=ray_origin,
            ray_dir=ray_dir,
            blocker_dist=blocker_dist,
            grid=grid,
            active=active,
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
        contribution = dr.select(
            plane_hit["valid"],
            rm_diag._vector_power(field_vector)
            * wt.Float(solid_angle_per_ray / cell_area)
            * unfolded_distance
            * unfolded_distance
            / dr.maximum(plane_hit["cos_theta"], wt.Float(EPS)),
            wt.Float(0.0),
        )

        los_mask = plane_hit["valid"] & (depth == wt.UInt32(0))
        reflection_mask = plane_hit["valid"] & (depth > wt.UInt32(0))
        cell_idx = mc_common._axis_aligned_cell_index(
            grid=grid,
            coord_0=plane_hit["coord_0"],
            coord_1=plane_hit["coord_1"],
        )
        if dr.hint(path_tape_store is not None, mode="scalar"):
            path_tape_store.store_los(
                initial_ray_dir=initial_ray_dir,
                cell_idx=cell_idx,
                active=los_mask,
            )
            path_tape_store.store_reflection(
                initial_ray_dir=initial_ray_dir,
                blocker_dist=blocker_dist,
                cell_idx=cell_idx,
                depth=depth,
                prim_history=prim_history,
                active=reflection_mask,
            )
        mc_common._scatter_component(
            grid=grid,
            weighted_diagnostics=weighted_diagnostics,
            component="los",
            coord_0=plane_hit["coord_0"],
            coord_1=plane_hit["coord_1"],
            power=contribution,
            active=los_mask,
        )
        mc_common._scatter_component(
            grid=grid,
            weighted_diagnostics=weighted_diagnostics,
            component="reflection",
            coord_0=plane_hit["coord_0"],
            coord_1=plane_hit["coord_1"],
            power=contribution,
            active=reflection_mask,
        )

        continue_hit = hit & (depth < max_bounces_u32)
        geometric_normal = dr.select(
            dr.norm(si.geo_n) > wt.Float(EPS),
            si.geo_n,
            si.n,
        )
        oriented_normal = dr.select(
            dr.dot(ray_dir, geometric_normal) > 0.0,
            -geometric_normal,
            geometric_normal,
        )
        material_inputs = resolve_surface_material(
            scene=scene,
            prim_idx=si.prim_index,
            override_material=config.reflection_material,
            reflection_coef=config.reflection_coef,
            default_eta_r=config.reflection_relative_permittivity,
            default_sigma=config.reflection_conductivity,
            valid_mask=continue_hit,
            use_scene_materials=config.use_scene_materials_for_reflection,
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
        polarization_vec = vector_select(continue_hit, reflected_polarization, polarization_vec)
        cumulative_image_source = dr.select(
            continue_hit,
            reflect_point_across_plane(cumulative_image_source, si.p, oriented_normal),
            cumulative_image_source,
        )
        reflected_dir = ray_dir - wt.Float(2.0) * dr.dot(ray_dir, oriented_normal) * oriented_normal
        ray_origin = dr.select(continue_hit, si.p + reflected_dir * RAY_ORIGIN_BIAS, ray_origin)
        ray_dir = dr.select(continue_hit, reflected_dir, ray_dir)
        ray_path_length = dr.select(continue_hit, ray_path_length + si.t, ray_path_length)
        for bounce_slot in range(int(max_bounces)):
            bounce_mask = continue_hit & (depth == wt.UInt32(bounce_slot))
            prim_history[bounce_slot] = dr.select(
                bounce_mask,
                si.prim_index,
                prim_history[bounce_slot],
            )
        active = continue_hit

        next_depth = depth + wt.UInt32(1)
        if dr.hint(rr_depth is not None and rr_prob < 1.0, mode="scalar"):
            gain_no_spread = rm_diag._vector_power(polarization_vec)
            continue_prob = dr.minimum(gain_no_spread, wt.Float(rr_prob))
            continue_prob = dr.maximum(
                continue_prob,
                wt.Float(mc_common._MC_MIN_RR_PROBABILITY),
            )
            rr_inactive = dr.full(wt.Bool, next_depth < int(rr_depth), current_batch_size)
            rr_continue = mc_common._hash_uniform_uint32(
                ray_index,
                stream=wt.UInt32(500) + next_depth,
                seed=int(seed),
            ) < continue_prob
            survive = rr_inactive | rr_continue
            survival_scale = dr.select(
                rr_inactive,
                wt.Float(1.0),
                dr.select(survive, dr.rsqrt(continue_prob), wt.Float(0.0)),
            )
            polarization_vec = vector_scale(polarization_vec, survival_scale)
            active = active & survive

        if dr.hint(stop_threshold_linear > 0.0, mode="scalar"):
            gain_no_spread = rm_diag._vector_power(polarization_vec)
            fspl = wt.Float(config.wavelength / (4.0 * math.pi)) / dr.maximum(
                ray_path_length,
                wt.Float(EPS),
            )
            active = active & (gain_no_spread * fspl * fspl > wt.Float(stop_threshold_linear))

        depth += wt.UInt32(1)

    return los_hits, reflection_hits, direct_tx_diffraction_state_store


__all__ = [
    "PathTapeStore",
    "trace_reflection",
]
