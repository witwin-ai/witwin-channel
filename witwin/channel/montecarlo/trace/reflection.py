from __future__ import annotations

import math
from dataclasses import dataclass

import drjit as dr
import rayd.drjit as rayd
from witwin.channel.core.scene import Scene
from witwin.channel.montecarlo import types as wt
from witwin.channel.core.grid import Grid
from .. import grid_ops
from ..config import ResolvedTraceConfig
from ..kernels.transport_vertex import TransportVertexKernel
from ..sampler import Sampler
from witwin.channel.core.numerics import arrays
from witwin.channel.core import geometry
from witwin.channel.core.physics import polarization
from witwin.channel.core.physics.materials import resolve_surface_material
from witwin.channel.core.numerics.arrays import scalar
from witwin.channel.core.numerics.constants import EPS, RAY_ORIGIN_BIAS
from .ad_support import (
    SparseCoeffBuffers,
    MC_TX_TRANSPORT_FD_STEP,
    GridScatter,
    SceneQuery,
    TransportCoeffBuilder,
)


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

    @staticmethod
    def empty(max_bounces: int) -> "ReflectionTape":
        zero_int = dr.zeros(wt.Int32, 0)
        bounce_count = max(0, int(max_bounces))
        return ReflectionTape(
            initial_ray_dir=arrays.empty_vector3(),
            blocker_dist=dr.zeros(wt.Float, 0),
            cell_idx=dr.zeros(wt.UInt32, 0),
            depth=zero_int,
            prim_index_by_bounce=tuple(dr.zeros(wt.Int32, 0) for _ in range(bounce_count)),
            transport_initial_ray_dir=arrays.empty_vector3(),
            transport_depth=zero_int,
            transport_blocker_prim_idx=zero_int,
            transport_prim_index_by_bounce=tuple(dr.zeros(wt.Int32, 0) for _ in range(bounce_count)),
        )

    @classmethod
    def from_payload(cls, payload: dict) -> "ReflectionTape":
        return cls(
            initial_ray_dir=payload["initial_ray_dir"],
            blocker_dist=payload["blocker_dist"],
            cell_idx=payload["cell_idx"],
            depth=payload["depth"],
            prim_index_by_bounce=payload["prim_index_by_bounce"],
            transport_initial_ray_dir=payload["transport_initial_ray_dir"],
            transport_depth=payload["transport_depth"],
            transport_blocker_prim_idx=payload["transport_blocker_prim_idx"],
            transport_prim_index_by_bounce=payload["transport_prim_index_by_bounce"],
        )

# Path-tracing thresholds local to the Monte Carlo reflection path.
MC_MIN_RR_PROBABILITY = 1.0e-8
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


class Reflection:
    # Evaluate the receiver-plane contribution for a single ray segment.
    @staticmethod
    def eval_plane_contribution(
        *,
        ray_origin,
        ray_dir,
        blocker_dist,
        grid,
        active,
        cumulative_image_source,
        polarization_vec,
        config,
        solid_angle_per_ray,
        cell_area,
    ):
        plane_hit = grid_ops.plane_hit(
            ray_origin=ray_origin,
            ray_dir=ray_dir,
            blocker_dist=blocker_dist,
            grid=grid,
            active=active,
        )
        unfolded_distance = dr.norm(plane_hit.target_pos - cumulative_image_source)
        field = grid_ops.point_source(
            cumulative_image_source,
            wt.Complex2f(1.0, 0.0),
            plane_hit.target_pos,
            config.wavelength,
            config.k,
        )
        field_vector = polarization.vector_scale(polarization_vec, field)
        contribution = dr.select(
            plane_hit.valid,
            polarization.vector_power(field_vector)
            * wt.Float(solid_angle_per_ray / cell_area)
            * unfolded_distance
            * unfolded_distance
            / dr.maximum(plane_hit.cos_theta, wt.Float(EPS)),
            wt.Float(0.0),
        )
        return plane_hit, contribution

    # Classify LOS/reflection hits, store tapes, and scatter power to the grid.
    @staticmethod
    def scatter_and_store_tapes(
        *,
        plane_hit,
        contribution,
        depth,
        grid,
        active,
        weighted_diagnostics,
        path_tape_store,
        initial_ray_dir,
        blocker_dist,
        blocker_prim_idx,
        prim_history,
        contribution_store,
    ):
        reflection_mask = plane_hit.valid & (depth > wt.UInt32(0))
        cell_idx = grid_ops.cell_index(
            grid=grid,
            coord_0=plane_hit.coord_0,
            coord_1=plane_hit.coord_1,
        )
        if dr.hint(path_tape_store is not None, mode="scalar"):
            path_tape_store.store_reflection_transport(
                initial_ray_dir=initial_ray_dir,
                depth=depth,
                blocker_prim_idx=blocker_prim_idx,
                prim_history=prim_history,
                active=active & (depth > wt.UInt32(0)),
            )
            path_tape_store.store_reflection(
                initial_ray_dir=initial_ray_dir,
                blocker_dist=blocker_dist,
                cell_idx=cell_idx,
                depth=depth,
                prim_history=prim_history,
                active=reflection_mask,
            )
        contribution_store.store(
            coord_0=plane_hit.coord_0,
            coord_1=plane_hit.coord_1,
            component_power={
                "reflection": dr.select(reflection_mask, contribution, wt.Float(0.0)),
            },
            active=reflection_mask,
        )

    # Compute reflected ray, update polarization, image source, and prim history.
    @staticmethod
    def reflect_and_update(
        *,
        hit,
        depth,
        max_bounces_u32,
        si,
        ray_dir,
        ray_origin,
        cumulative_image_source,
        polarization_vec,
        ray_path_length,
        prim_history,
        scene,
        config,
        material_omega,
        max_bounces,
    ):
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
            prim_idx=si.global_prim_id,
            default_gain=1.0,
            valid_mask=continue_hit,
        )
        reflected_polarization = polarization.reflect_field_vector(
            polarization_vec,
            ray_dir,
            oriented_normal,
            eta_r=material_inputs.eta_r,
            sigma=material_inputs.sigma,
            omega=material_omega,
            gain=material_inputs.gain,
            mu_r=material_inputs.mu_r,
        )
        new_polarization = polarization.vector_select(continue_hit, reflected_polarization, polarization_vec)
        new_image_source = dr.select(
            continue_hit,
            geometry.reflect_point_across_plane(cumulative_image_source, si.p, oriented_normal),
            cumulative_image_source,
        )
        reflected_dir = ray_dir - wt.Float(2.0) * dr.dot(ray_dir, oriented_normal) * oriented_normal
        new_ray_origin = dr.select(continue_hit, si.p + reflected_dir * RAY_ORIGIN_BIAS, ray_origin)
        new_ray_dir = dr.select(continue_hit, reflected_dir, ray_dir)
        new_path_length = dr.select(continue_hit, ray_path_length + si.t, ray_path_length)
        new_prim_history = list(prim_history)
        for bounce_slot in range(int(max_bounces)):
            bounce_mask = continue_hit & (depth == wt.UInt32(bounce_slot))
            new_prim_history[bounce_slot] = dr.select(
                bounce_mask,
                wt.Int32(si.global_prim_id),
                prim_history[bounce_slot],
            )
        return {
            "ray_origin": new_ray_origin,
            "ray_dir": new_ray_dir,
            "cumulative_image_source": new_image_source,
            "polarization_vec": new_polarization,
            "ray_path_length": new_path_length,
            "prim_history": new_prim_history,
            "active": continue_hit,
        }

    # Apply Russian roulette survival test and early stop threshold.
    @staticmethod
    def apply_rr_and_threshold(
        *,
        active,
        depth,
        rr_depth,
        rr_prob,
        ray_index,
        seed,
        polarization_vec,
        ray_path_length,
        config,
        stop_threshold_linear,
        current_batch_size,
    ):
        next_depth = depth + wt.UInt32(1)
        if dr.hint(rr_depth is not None and rr_prob < 1.0, mode="scalar"):
            gain_no_spread = polarization.vector_power(polarization_vec)
            continue_prob = dr.minimum(gain_no_spread, wt.Float(rr_prob))
            continue_prob = dr.maximum(
                continue_prob,
                wt.Float(MC_MIN_RR_PROBABILITY),
            )
            rr_inactive = dr.full(wt.Bool, next_depth < int(rr_depth), current_batch_size)
            rr_continue = Sampler.hash_uniform(
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
            polarization_vec = polarization.vector_scale(polarization_vec, survival_scale)
            active = active & survive

        if dr.hint(stop_threshold_linear > 0.0, mode="scalar"):
            gain_no_spread = polarization.vector_power(polarization_vec)
            fspl = wt.Float(config.wavelength / (4.0 * math.pi)) / dr.maximum(
                ray_path_length,
                wt.Float(EPS),
            )
            active = active & (gain_no_spread * fspl * fspl > wt.Float(stop_threshold_linear))

        return active, next_depth, polarization_vec

    @staticmethod
    @dr.syntax
    def trace(
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
        collect_wedges: bool,
        collect_wedge_prefixes: bool = False,
        diff_state_store=None,
        path_tape_store: PathTapeStore | None = None,
        loop_mode: str = "symbolic",
        contribution_store=None,
    ):
        # Trace reflected TX-emitted rays with the symbolic Dr.Jit loop used by the Monte Carlo radiomap path.
        current_batch_size = int(dr.width(ray_dir.x))
        initial_ray_dir = ray_dir
        ray_origin = arrays.broadcast(tx_pos, current_batch_size)
        cumulative_image_source = arrays.broadcast(tx_pos, current_batch_size)
        polarization_vec = Sampler.source_field(ray_dir)
        ray_path_length = dr.zeros(wt.Float, current_batch_size)
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
                diff_state_store,
                path_tape_store,
            ],
        ):
            ray = rayd.RayAD(ray_origin, ray_dir)
            si = scene.ray_intersect(ray, active=active, flags=rayd.RayFlags.All)
            hit = si.is_valid() & active
            if dr.hint(collect_wedges, mode="scalar"):
                direct_hit = hit & (depth == wt.UInt32(0))
                prefix_hit = (hit & (depth > wt.UInt32(0))) if collect_wedge_prefixes else direct_hit
                prefix_depth = dr.full(wt.Int32, 0, current_batch_size) + wt.Int32(depth)
                prefix_slot = (
                    ray_index * wt.UInt32(int(max_bounces) + 1) + wt.UInt32(depth)
                    if collect_wedge_prefixes
                    else None
                )
                prefix_source_power = polarization.vector_power(polarization_vec)
                if dr.hint(collect_wedge_prefixes, mode="scalar"):
                    prefix_source_power = prefix_source_power * wt.Float(float(solid_angle_per_ray))
                diff_state_store.store(
                    ray_directions=ray_dir,
                    prim_index=si.global_prim_id,
                    hit_p=si.p,
                    hit_n=si.n,
                    hit_geo_n=si.geo_n,
                    source_pos=cumulative_image_source,
                    source_power=prefix_source_power,
                    prefix_reflection_depth=prefix_depth,
                    initial_ray_dir=initial_ray_dir,
                    prim_history=prim_history,
                    slot_index=prefix_slot,
                    active=prefix_hit,
                )
            blocker_dist = dr.select(hit, si.t, wt.Float(1.0e10))
            blocker_prim_idx = dr.select(hit, wt.Int32(si.global_prim_id), wt.Int32(-1))

            plane_hit, contribution = Reflection.eval_plane_contribution(
                ray_origin=ray_origin, ray_dir=ray_dir,
                blocker_dist=blocker_dist, grid=grid, active=active,
                cumulative_image_source=cumulative_image_source,
                polarization_vec=polarization_vec, config=config,
                solid_angle_per_ray=solid_angle_per_ray, cell_area=cell_area,
            )
            Reflection.scatter_and_store_tapes(
                plane_hit=plane_hit, contribution=contribution,
                depth=depth, grid=grid, active=active,
                weighted_diagnostics=weighted_diagnostics,
                path_tape_store=path_tape_store,
                initial_ray_dir=initial_ray_dir, blocker_dist=blocker_dist,
                blocker_prim_idx=blocker_prim_idx, prim_history=prim_history,
                contribution_store=contribution_store,
            )

            refl = Reflection.reflect_and_update(
                hit=hit, depth=depth, max_bounces_u32=max_bounces_u32, si=si,
                ray_dir=ray_dir, ray_origin=ray_origin,
                cumulative_image_source=cumulative_image_source,
                polarization_vec=polarization_vec,
                ray_path_length=ray_path_length, prim_history=prim_history,
                scene=scene, config=config,
                material_omega=material_omega, max_bounces=max_bounces,
            )
            ray_origin = refl["ray_origin"]
            ray_dir = refl["ray_dir"]
            cumulative_image_source = refl["cumulative_image_source"]
            polarization_vec = refl["polarization_vec"]
            ray_path_length = refl["ray_path_length"]
            prim_history = refl["prim_history"]
            active = refl["active"]

            active, depth, polarization_vec = Reflection.apply_rr_and_threshold(
                active=active, depth=depth,
                rr_depth=rr_depth, rr_prob=rr_prob,
                ray_index=ray_index, seed=seed,
                polarization_vec=polarization_vec,
                ray_path_length=ray_path_length, config=config,
                stop_threshold_linear=stop_threshold_linear,
                current_batch_size=current_batch_size,
            )

        return los_hits, reflection_hits, diff_state_store


class ReflectionAD:
    # Replay reflection bounces from a recorded tape, returning final ray state
    @staticmethod
    def replay_bounces(
        *,
        ray_dir,
        ray_origin,
        cumulative_image_source,
        scene: Scene,
        config: ResolvedTraceConfig,
        depth_array,
        prim_by_bounce: tuple,
        material_omega,
        track_vertices: bool = False,
        track_materials: bool = False,
    ) -> dict:
        max_bounces = len(prim_by_bounce)
        faces = scene._merged_faces()
        face_x = wt.Int32(faces.x)
        face_y = wt.Int32(faces.y)
        face_z = wt.Int32(faces.z)
        polarization_vec = Sampler.source_field(ray_dir)
        vertex_index_slots = []
        vertex_vars = []
        material_index_slots = []
        material_grad_sources = []

        for bounce_slot in range(max_bounces):
            active_bounce = depth_array > wt.Int32(bounce_slot)
            prim_idx = prim_by_bounce[bounce_slot]
            safe_prim_idx = wt.UInt32(dr.select(active_bounce, prim_idx, wt.Int32(0)))
            v_idx0 = dr.select(active_bounce, dr.gather(wt.Int32, face_x, safe_prim_idx), wt.Int32(-1))
            v_idx1 = dr.select(active_bounce, dr.gather(wt.Int32, face_y, safe_prim_idx), wt.Int32(-1))
            v_idx2 = dr.select(active_bounce, dr.gather(wt.Int32, face_z, safe_prim_idx), wt.Int32(-1))
            local_v0 = SceneQuery.vertex_point(scene, v_idx0)
            local_v1 = SceneQuery.vertex_point(scene, v_idx1)
            local_v2 = SceneQuery.vertex_point(scene, v_idx2)
            if track_vertices:
                dr.enable_grad(
                    local_v0.x, local_v0.y, local_v0.z,
                    local_v1.x, local_v1.y, local_v1.z,
                    local_v2.x, local_v2.y, local_v2.z,
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
            material_inputs = SceneQuery.material(
                prim_idx,
                scene=scene,
                gain=1.0,
            )
            if track_materials:
                material_index_slots.append(material_inputs.material_idx)
                material_grad_sources.append((material_inputs.eta_r, material_inputs.sigma))
            reflected_polarization = polarization.reflect_field_vector(
                polarization_vec,
                ray_dir,
                oriented_normal,
                eta_r=material_inputs.eta_r,
                sigma=material_inputs.sigma,
                omega=material_omega,
                gain=material_inputs.gain,
                mu_r=material_inputs.mu_r,
            )
            reflected_dir = ray_dir - wt.Float(2.0) * dr.dot(ray_dir, oriented_normal) * oriented_normal
            cumulative_image_source = dr.select(
                active_bounce,
            geometry.reflect_point_across_plane(cumulative_image_source, hit_p, oriented_normal),
                cumulative_image_source,
            )
            ray_origin = dr.select(active_bounce, hit_p + reflected_dir * wt.Float(1.0e-4), ray_origin)
            ray_dir = dr.select(active_bounce, reflected_dir, ray_dir)
            polarization_vec = polarization.vector_select(active_bounce, reflected_polarization, polarization_vec)

        return {
            "ray_origin": ray_origin,
            "ray_dir": ray_dir,
            "cumulative_image_source": cumulative_image_source,
            "polarization_vec": polarization_vec,
            "vertex_index_slots": tuple(vertex_index_slots),
            "vertex_vars": tuple(vertex_vars),
            "material_index_slots": material_index_slots,
            "material_grad_sources": material_grad_sources,
        }

    # Compute plane_hit, field, and power contribution after reflection bounces
    @staticmethod
    def post_bounce(
        *,
        ray_origin,
        ray_dir,
        cumulative_image_source,
        polarization_vec,
        blocker_dist,
        grid: Grid,
        config: ResolvedTraceConfig,
        solid_angle_scale: float,
        width: int,
        detach: bool = True,
    ) -> tuple:
        plane_hit = grid_ops.plane_hit(
            ray_origin=ray_origin,
            ray_dir=ray_dir,
            blocker_dist=blocker_dist,
            grid=grid,
            active=dr.full(wt.Bool, True, width),
        )
        unfolded_dist = dr.norm(plane_hit.target_pos - cumulative_image_source)
        field = grid_ops.point_source(
            cumulative_image_source,
            wt.Complex2f(1.0, 0.0),
            plane_hit.target_pos,
            config.wavelength,
            config.k,
        )
        field_vector = polarization.vector_scale(polarization_vec, field)
        power = (
            polarization.vector_power(field_vector)
            * wt.Float(solid_angle_scale)
            * unfolded_dist
            * unfolded_dist
            / dr.maximum(plane_hit.cos_theta, wt.Float(1.0e-6))
        )
        if detach:
            power = dr.detach(power)
        return plane_hit, power

    @staticmethod
    def sparse_coeffs(
        *,
        tape: ReflectionTape,
        scene: Scene,
        tx_pos,
        grid: Grid,
        config: ResolvedTraceConfig,
        solid_angle_per_ray: float,
        cell_area: float,
        material_omega,
    ):
        width = int(dr.width(tape.cell_idx))
        max_bounces = len(tape.prim_index_by_bounce)
        if width <= 0 or max_bounces <= 0:
            return SparseCoeffBuffers.empty()

        local_tx = SceneQuery.tx_lanes(tx_pos, width)
        dr.enable_grad(local_tx.x, local_tx.y, local_tx.z)
        bounced = ReflectionAD.replay_bounces(
            ray_dir=tape.initial_ray_dir,
            ray_origin=local_tx,
            cumulative_image_source=local_tx,
            scene=scene,
            config=config,
            depth_array=tape.depth,
            prim_by_bounce=tape.prim_index_by_bounce,
            material_omega=material_omega,
            track_vertices=True,
            track_materials=True,
        )
        _, contribution = ReflectionAD.post_bounce(
            ray_origin=bounced["ray_origin"],
            ray_dir=bounced["ray_dir"],
            cumulative_image_source=bounced["cumulative_image_source"],
            polarization_vec=bounced["polarization_vec"],
            blocker_dist=tape.blocker_dist,
            grid=grid,
            config=config,
            solid_angle_scale=solid_angle_per_ray / cell_area,
            width=width,
            detach=False,
        )
        dr.backward(dr.sum(contribution))

        return SparseCoeffBuffers(
            cell_idx=tape.cell_idx,
            tx_coeff_x=dr.grad(local_tx.x),
            tx_coeff_y=dr.grad(local_tx.y),
            tx_coeff_z=dr.grad(local_tx.z),
            vertex_indices=GridScatter.flatten_slots(wt.Int32, list(bounced["vertex_index_slots"])),
            vertex_coeff_x=GridScatter.flatten_slots(wt.Float, [dr.grad(v.x) for v in bounced["vertex_vars"]]),
            vertex_coeff_y=GridScatter.flatten_slots(wt.Float, [dr.grad(v.y) for v in bounced["vertex_vars"]]),
            vertex_coeff_z=GridScatter.flatten_slots(wt.Float, [dr.grad(v.z) for v in bounced["vertex_vars"]]),
            vertex_slot_count=len(bounced["vertex_vars"]),
            material_indices=GridScatter.flatten_slots(wt.Int32, bounced["material_index_slots"]),
            material_coeff_eps=GridScatter.flatten_slots(wt.Float, [dr.grad(e) for e, _ in bounced["material_grad_sources"]]),
            material_coeff_sigma=GridScatter.flatten_slots(wt.Float, [dr.grad(s) for _, s in bounced["material_grad_sources"]]),
            material_slot_count=len(bounced["material_grad_sources"]),
        )

    @staticmethod
    def transport_geometry_state(
        *,
        tape: ReflectionTape,
        scene: Scene,
        tx_pos,
        grid: Grid,
        config: ResolvedTraceConfig,
        solid_angle_per_ray: float,
        cell_area: float,
        material_omega,
    ):
        width = int(dr.width(tape.transport_blocker_prim_idx))
        max_bounces = len(tape.transport_prim_index_by_bounce)
        if width <= 0 or max_bounces <= 0:
            return None

        local_tx = SceneQuery.tx_lanes(tx_pos, width)
        bounced = ReflectionAD.replay_bounces(
            ray_dir=tape.transport_initial_ray_dir,
            ray_origin=local_tx,
            cumulative_image_source=local_tx,
            scene=scene,
            config=config,
            depth_array=tape.transport_depth,
            prim_by_bounce=tape.transport_prim_index_by_bounce,
            material_omega=material_omega,
            track_vertices=True,
        )
        blocker_dist = SceneQuery.blocker_dist(
            scene,
            ray_origin=bounced["ray_origin"],
            ray_dir=bounced["ray_dir"],
            prim_idx=tape.transport_blocker_prim_idx,
        )
        plane_hit, contribution = ReflectionAD.post_bounce(
            ray_origin=bounced["ray_origin"],
            ray_dir=bounced["ray_dir"],
            cumulative_image_source=bounced["cumulative_image_source"],
            polarization_vec=bounced["polarization_vec"],
            blocker_dist=blocker_dist,
            grid=grid,
            config=config,
            solid_angle_scale=solid_angle_per_ray / cell_area,
            width=width,
        )
        return {
            "plane_hit": plane_hit,
            "contribution": contribution,
            "vertex_index_slots": bounced["vertex_index_slots"],
            "vertex_vars": bounced["vertex_vars"],
        }

    @staticmethod
    def vertex_state(
        *,
        tape: ReflectionTape,
        scene: Scene,
        tx_pos,
        grid: Grid,
        config: ResolvedTraceConfig,
        solid_angle_per_ray: float,
        cell_area: float,
        material_omega,
    ):
        state = ReflectionAD.transport_geometry_state(
            tape=tape,
            scene=scene,
            tx_pos=tx_pos,
            grid=grid,
            config=config,
            solid_angle_per_ray=solid_angle_per_ray,
            cell_area=cell_area,
            material_omega=material_omega,
        )
        if state is None:
            return None
        transport_map = GridScatter.tent_splat(
            grid=grid,
            coord_0=state["plane_hit"].coord_0,
            coord_1=state["plane_hit"].coord_1,
            power=state["contribution"],
            active=state["plane_hit"].valid,
        )
        state["transport_map"] = transport_map
        return state

    @staticmethod
    def transport_vertex_coeffs(
        *,
        tape: ReflectionTape,
        scene: Scene,
        tx_pos,
        grid: Grid,
        config: ResolvedTraceConfig,
        solid_angle_per_ray: float,
        cell_area: float,
        material_omega,
    ):
        return TransportCoeffBuilder.build_vertex_buffers(
            grid=grid,
            state_factory=lambda: ReflectionAD.transport_geometry_state(
                tape=tape,
                scene=scene,
                tx_pos=tx_pos,
                grid=grid,
                config=config,
                solid_angle_per_ray=solid_angle_per_ray,
                cell_area=cell_area,
                material_omega=material_omega,
            ),
            vertex_index_getter=lambda state: state["vertex_index_slots"],
            vertex_var_getter=lambda state: state["vertex_vars"],
        )

    @staticmethod
    def vertex_jvp(
        *,
        tape: ReflectionTape,
        scene: Scene,
        tx_pos,
        grid: Grid,
        config: ResolvedTraceConfig,
        solid_angle_per_ray: float,
        cell_area: float,
        material_omega,
        vertex_tangent,
    ):
        buffers = ReflectionAD.transport_vertex_coeffs(
            tape=tape,
            scene=scene,
            tx_pos=tx_pos,
            grid=grid,
            config=config,
            solid_angle_per_ray=solid_angle_per_ray,
            cell_area=cell_area,
            material_omega=material_omega,
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
    def vertex_vjp(
        *,
        tape: ReflectionTape,
        scene: Scene,
        tx_pos,
        grid: Grid,
        config: ResolvedTraceConfig,
        solid_angle_per_ray: float,
        cell_area: float,
        material_omega,
        upstream_component,
        n_vertices: int,
    ):
        zero = dr.zeros(wt.Float, int(n_vertices))
        buffers = ReflectionAD.transport_vertex_coeffs(
            tape=tape,
            scene=scene,
            tx_pos=tx_pos,
            grid=grid,
            config=config,
            solid_angle_per_ray=solid_angle_per_ray,
            cell_area=cell_area,
            material_omega=material_omega,
        )
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

    @staticmethod
    def tx_basis_maps(
        *,
        tape: ReflectionTape,
        scene: Scene,
        tx_pos,
        grid: Grid,
        config: ResolvedTraceConfig,
        solid_angle_per_ray: float,
        cell_area: float,
        material_omega,
        transport_step: float = MC_TX_TRANSPORT_FD_STEP,
    ):
        width = int(dr.width(tape.transport_blocker_prim_idx))
        max_bounces = len(tape.transport_prim_index_by_bounce)
        zero_map = dr.zeros(wt.Float, int(grid.n_cells))
        if width <= 0 or max_bounces <= 0:
            return {"x": zero_map, "y": zero_map, "z": zero_map}

        base_tx = SceneQuery.tx_lanes(tx_pos, width)

        def replay(tx_lanes):
            bounced = ReflectionAD.replay_bounces(
                ray_dir=tape.transport_initial_ray_dir,
                ray_origin=tx_lanes,
                cumulative_image_source=tx_lanes,
                scene=scene,
                config=config,
                depth_array=tape.transport_depth,
                prim_by_bounce=tape.transport_prim_index_by_bounce,
                material_omega=material_omega,
            )
            blocker_dist = SceneQuery.blocker_dist(
                scene,
                ray_origin=bounced["ray_origin"],
                ray_dir=bounced["ray_dir"],
                prim_idx=tape.transport_blocker_prim_idx,
            )
            return ReflectionAD.post_bounce(
                ray_origin=bounced["ray_origin"],
                ray_dir=bounced["ray_dir"],
                cumulative_image_source=bounced["cumulative_image_source"],
                polarization_vec=bounced["polarization_vec"],
                blocker_dist=blocker_dist,
                grid=grid,
                config=config,
                solid_angle_scale=solid_angle_per_ray / cell_area,
                width=width,
            )

        step_scalar = float(transport_step)
        step = wt.Float(step_scalar)

        def map_for_shift(dx: float, dy: float, dz: float):
            shifted_tx = wt.Point3f(base_tx.x + wt.Float(dx), base_tx.y + wt.Float(dy), base_tx.z + wt.Float(dz))
            shifted_plane_hit, shifted_power = replay(shifted_tx)
            return GridScatter.power(
                grid=grid,
                coord_0=shifted_plane_hit.coord_0,
                coord_1=shifted_plane_hit.coord_1,
                power=shifted_power,
                active=shifted_plane_hit.valid,
            )

        return {
            "x": (map_for_shift(step_scalar, 0.0, 0.0) - map_for_shift(-step_scalar, 0.0, 0.0)) / (wt.Float(2.0) * step),
            "y": (map_for_shift(0.0, step_scalar, 0.0) - map_for_shift(0.0, -step_scalar, 0.0)) / (wt.Float(2.0) * step),
            "z": (map_for_shift(0.0, 0.0, step_scalar) - map_for_shift(0.0, 0.0, -step_scalar)) / (wt.Float(2.0) * step),
        }


__all__ = [
    "PathTapeStore",
    "Reflection",
    "ReflectionAD",
    "ReflectionTape",
]
