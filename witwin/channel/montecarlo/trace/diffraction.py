"""Diffraction tracing, state, tape store, and edge sampling runtime."""
from __future__ import annotations

import math
from dataclasses import dataclass

import drjit as dr
from witwin.channel.core.scene import Scene
from witwin.channel.montecarlo import types as wt
from .. import grid_ops
from ..kernels.diffraction_builder import DiffractionBuilderKernel
from ..sampler import Sampler
from ..config import ResolvedTraceConfig
from witwin.channel.core.numerics import arrays
from witwin.channel.core.numerics.arrays import scalar
from witwin.channel.core.numerics.constants import EPS
from witwin.channel.core.physics.materials import FaceMaterial, resolve_surface_material
from .diffraction_utd import UTD


# Diffraction-edge offset used by the Monte Carlo edge sampler to push the
# sample point away from the wedge before transport.
MC_DIFFRACTION_OFFSET = 5.0e-2


@dataclass(slots=True)
class OrientedEdgeView:
    """Edge frame oriented relative to an incident ray (face0 = near face)."""
    edge_dir: object
    n0: object
    nn: object
    face0_material: FaceMaterial
    face1_material: FaceMaterial


@dataclass(slots=True)
class SampledDiffPoint:
    """Per-sample diffraction-point state produced by Diffraction.sample_diff_point."""
    sample_active: object
    batch_states: object
    edge_fraction: object
    diff_point: object
    diff_point_offset: object
    ko: object
    ray_origin: object
    oriented: OrientedEdgeView
    keller_sample: object


@dataclass(slots=True)
class VisibilityResult:
    """Source / target visibility outcomes for a diffraction sample."""
    plane_hit: object
    source_visible: object
    visible_target: object


# ============================================================================
# Tape store (records diffraction samples for AD replay)
# ============================================================================

@dataclass(slots=True)
class DiffractionTape:
    """Recorded diffraction samples for AD replay."""
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

    @staticmethod
    def empty() -> DiffractionTape:
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


class DiffractionTapeStore:
    def __init__(self, *, capacity: int) -> None:
        self.capacity = max(0, int(capacity))
        self.next_slot = wt.UInt32(0)
        self.edge_index = dr.zeros(wt.Int32, self.capacity)
        self.edge_fraction = dr.zeros(wt.Float, self.capacity)
        self.cone_sample = dr.zeros(wt.Float, self.capacity)
        self.cell_idx = dr.zeros(wt.UInt32, self.capacity)
        self.field_valid = dr.zeros(wt.Bool, self.capacity)
        self.pole_safe = dr.zeros(wt.Bool, self.capacity)
        self.dif_n_p = dr.zeros(wt.Float, self.capacity)
        self.dif_n_m = dr.zeros(wt.Float, self.capacity)
        self.sum_n_p = dr.zeros(wt.Float, self.capacity)
        self.sum_n_m = dr.zeros(wt.Float, self.capacity)

    def store(self, *, edge_index, edge_fraction, cone_sample, cell_idx, field_valid, pole_safe,
              dif_n_p, dif_n_m, sum_n_p, sum_n_m, active) -> None:
        if self.capacity <= 0:
            return
        slot = dr.scatter_inc(self.next_slot, wt.UInt32(0), active)
        store_mask = active & (slot < wt.UInt32(self.capacity))
        dr.scatter(self.edge_index, edge_index, slot, store_mask)
        dr.scatter(self.edge_fraction, edge_fraction, slot, store_mask)
        dr.scatter(self.cone_sample, cone_sample, slot, store_mask)
        dr.scatter(self.cell_idx, cell_idx, slot, store_mask)
        dr.scatter(self.field_valid, field_valid, slot, store_mask)
        dr.scatter(self.pole_safe, pole_safe, slot, store_mask)
        dr.scatter(self.dif_n_p, dif_n_p, slot, store_mask)
        dr.scatter(self.dif_n_m, dif_n_m, slot, store_mask)
        dr.scatter(self.sum_n_p, sum_n_p, slot, store_mask)
        dr.scatter(self.sum_n_m, sum_n_m, slot, store_mask)

    def finalize(self) -> DiffractionTape:
        zero_float = dr.zeros(wt.Float, 0)
        zero_int = dr.zeros(wt.Int32, 0)
        zero_uint = dr.zeros(wt.UInt32, 0)
        count = int(scalar(self.next_slot))
        indices = dr.arange(wt.UInt32, count)
        gather = lambda dtype, values, empty: dr.gather(dtype, values, indices) if count > 0 else empty
        return DiffractionTape(
            edge_index=gather(wt.Int32, self.edge_index, zero_int),
            edge_fraction=gather(wt.Float, self.edge_fraction, zero_float),
            cone_sample=gather(wt.Float, self.cone_sample, zero_float),
            cell_idx=gather(wt.UInt32, self.cell_idx, zero_uint),
            field_valid=gather(wt.Bool, self.field_valid, dr.zeros(wt.Bool, 0)),
            pole_safe=gather(wt.Bool, self.pole_safe, dr.zeros(wt.Bool, 0)),
            dif_n_p=gather(wt.Float, self.dif_n_p, zero_float),
            dif_n_m=gather(wt.Float, self.dif_n_m, zero_float),
            sum_n_p=gather(wt.Float, self.sum_n_p, zero_float),
            sum_n_m=gather(wt.Float, self.sum_n_m, zero_float),
        )


class DiffractionHitStore:
    """Store direct-hit wedges candidates before native edge discovery."""

    def __init__(self, *, capacity: int, max_bounces: int = 0) -> None:
        self.capacity = max(0, int(capacity))
        self.max_bounces = max(0, int(max_bounces))
        self.count = wt.UInt32(0)
        zero_float = dr.zeros(wt.Float, self.capacity)
        self.ray_directions = wt.Vector3f(zero_float, zero_float, zero_float)
        self.prefix_initial_ray_dir = wt.Vector3f(zero_float, zero_float, zero_float)
        self.prim_index = dr.full(wt.Int32, -1, self.capacity)
        self.hit_p = wt.Point3f(zero_float, zero_float, zero_float)
        self.hit_n = wt.Vector3f(zero_float, zero_float, zero_float)
        self.hit_geo_n = wt.Vector3f(zero_float, zero_float, zero_float)
        self.source_pos = wt.Point3f(zero_float, zero_float, zero_float)
        self.source_power = dr.zeros(wt.Float, self.capacity)
        self.prefix_reflection_depth = dr.zeros(wt.Int32, self.capacity)
        self.prefix_prim_by_bounce = tuple(
            dr.full(wt.Int32, -1, self.capacity)
            for _ in range(self.max_bounces)
        )
        self.valid = dr.zeros(wt.Bool, self.capacity)

    def store(self, *, ray_directions, prim_index, hit_p, hit_n, hit_geo_n, active,
              source_pos, source_power, prefix_reflection_depth,
              initial_ray_dir, prim_history, slot_index) -> None:
        if self.capacity <= 0:
            return
        if slot_index is None:
            slot = dr.scatter_inc(self.count, wt.UInt32(0), active)
        else:
            slot = wt.UInt32(slot_index)
            dr.scatter_inc(self.count, wt.UInt32(0), active)
        store_mask = active & (slot < wt.UInt32(self.capacity))
        width = int(dr.width(ray_directions.x))
        dr.scatter(self.ray_directions, ray_directions, slot, store_mask)
        dr.scatter(self.prefix_initial_ray_dir, initial_ray_dir, slot, store_mask)
        dr.scatter(self.prim_index, prim_index, slot, store_mask)
        dr.scatter(self.hit_p, hit_p, slot, store_mask)
        dr.scatter(self.hit_n, hit_n, slot, store_mask)
        dr.scatter(self.hit_geo_n, hit_geo_n, slot, store_mask)
        dr.scatter(self.source_pos, source_pos, slot, store_mask)
        dr.scatter(self.source_power, source_power, slot, store_mask)
        dr.scatter(self.prefix_reflection_depth, prefix_reflection_depth, slot, store_mask)
        for bounce_slot, prim_idx in enumerate(tuple(prim_history)[: self.max_bounces]):
            dr.scatter(self.prefix_prim_by_bounce[bounce_slot], prim_idx, slot, store_mask)
        dr.scatter(self.valid, dr.full(wt.Bool, True, width), slot, store_mask)

    def finalize(self):
        if self.capacity > 0 and dr.any(self.valid):
            indices = dr.compress(self.valid)
            count = int(dr.width(indices))
        else:
            count = 0
            indices = dr.zeros(wt.UInt32, 0)
        zero_float = dr.zeros(wt.Float, 0)
        zero_int = dr.zeros(wt.Int32, 0)
        return {
            "ray_directions": (
                dr.gather(wt.Vector3f, self.ray_directions, indices)
                if count > 0
                else wt.Vector3f(zero_float, zero_float, zero_float)
            ),
            "prefix_initial_ray_dir": (
                dr.gather(wt.Vector3f, self.prefix_initial_ray_dir, indices)
                if count > 0
                else wt.Vector3f(zero_float, zero_float, zero_float)
            ),
            "prim_index": dr.gather(wt.Int32, self.prim_index, indices) if count > 0 else zero_int,
            "hit_p": dr.gather(wt.Point3f, self.hit_p, indices) if count > 0 else wt.Point3f(zero_float, zero_float, zero_float),
            "hit_n": dr.gather(wt.Vector3f, self.hit_n, indices) if count > 0 else wt.Vector3f(zero_float, zero_float, zero_float),
            "hit_geo_n": (
                dr.gather(wt.Vector3f, self.hit_geo_n, indices)
                if count > 0
                else wt.Vector3f(zero_float, zero_float, zero_float)
            ),
            "source_pos": (
                dr.gather(wt.Point3f, self.source_pos, indices)
                if count > 0
                else wt.Point3f(zero_float, zero_float, zero_float)
            ),
            "source_power": (
                dr.gather(wt.Float, self.source_power, indices)
                if count > 0
                else zero_float
            ),
            "prefix_reflection_depth": (
                dr.gather(wt.Int32, self.prefix_reflection_depth, indices)
                if count > 0
                else zero_int
            ),
            "prefix_prim_by_bounce": tuple(
                dr.gather(wt.Int32, prim_idx, indices) if count > 0 else zero_int
                for prim_idx in self.prefix_prim_by_bounce
            ),
            "count": count,
        }

class Diffraction:
    """Static namespace for diffraction runtime orchestration and Monte Carlo tracing."""

    @staticmethod
    def sample_diff_point(
        *,
        sample_lane,
        batch_start,
        total_samples_u32,
        diffraction_states,
        state_slot_all,
        edge_fraction_all,
        keller_sample_all,
    ):
        sample_index = sample_lane + batch_start
        sample_active = sample_index < total_samples_u32
        safe_sample_index = wt.UInt32(dr.select(sample_active, sample_index, wt.UInt32(0)))
        state_slot = dr.gather(wt.UInt32, state_slot_all, safe_sample_index)
        batch_states = diffraction_states.gather(state_slot)
        edge_hat = batch_states.edge_dir / (dr.norm(batch_states.edge_dir) + wt.Float(EPS))
        line_length = dr.maximum(
            batch_states.edge_line_max - batch_states.edge_line_min,
            wt.Float(0.0),
        )
        edge_fraction = dr.gather(wt.Float, edge_fraction_all, safe_sample_index)
        ell_sample = batch_states.edge_line_min + line_length * edge_fraction
        diff_point = batch_states.edge_pos + edge_hat * ell_sample
        incident_dir = diff_point - batch_states.source_pos
        oriented = batch_states.orient_face_view(incident_dir)
        face_sum = oriented.n0 + oriented.nn
        face_sum_norm = dr.norm(face_sum)
        offset_normal = dr.select(
            face_sum_norm > wt.Float(EPS),
            face_sum / face_sum_norm,
            wt.Vector3f(0.0, 0.0, 0.0),
        )
        diff_point_offset = diff_point + wt.Float(MC_DIFFRACTION_OFFSET) * offset_normal
        keller_sample = dr.gather(wt.Float, keller_sample_all, safe_sample_index)
        ko = UTD.sample_keller_cone(
            oriented.edge_dir, oriented.n0, oriented.nn,
            keller_sample, incident_dir, lit_region=True,
        )
        ray_origin = Sampler.spawn_offset_ray_origin(diff_point, ko, offset_normal)
        return SampledDiffPoint(
            sample_active=sample_active,
            batch_states=batch_states,
            edge_fraction=edge_fraction,
            diff_point=diff_point,
            diff_point_offset=diff_point_offset,
            ko=ko,
            ray_origin=ray_origin,
            oriented=oriented,
            keller_sample=keller_sample,
        )

    @staticmethod
    def check_visibility(*, diff_point, diff_point_offset, source_pos, ko, ray_origin,
                         diffraction_batch_size, grid, sample_active, scene) -> VisibilityResult:
        plane_hit = grid_ops.plane_hit(
            ray_origin=ray_origin,
            ray_dir=ko,
            blocker_dist=dr.full(wt.Float, 1.0e10, int(diffraction_batch_size)),
            grid=grid,
            active=sample_active,
        )
        visible_source = scene.segment_pair_visible(
            source_pos,
            diff_point,
            diff_point_offset,
            active=sample_active,
        )
        safe_target_diff_point = dr.select(plane_hit.valid, diff_point, plane_hit.target_pos)
        safe_target_diff_point_offset = dr.select(
            plane_hit.valid,
            diff_point_offset,
            plane_hit.target_pos,
        )
        visible_target = scene.segment_pair_visible(
            plane_hit.target_pos,
            safe_target_diff_point,
            safe_target_diff_point_offset,
            active=plane_hit.valid,
        )
        return VisibilityResult(
            plane_hit=plane_hit,
            source_visible=visible_source,
            visible_target=visible_target,
        )

    @staticmethod
    # MC diffraction batch tracing loop (symbolic DrJit loop)
    @dr.syntax
    def trace_batches(*, scene, grid, diffraction_states, sampler, diffraction_batch_size: int,
                      diffraction_batch_count: int, samples_per_tx: int, seed: int, k, wavelength,
                      plane_normal, diff_gain_scale, weighted_diagnostics,
                      diffraction_tape_store: DiffractionTapeStore | None = None,
                      loop_mode: str = "symbolic",
                      contribution_store=None):
        sample_lane = dr.arange(wt.UInt32, int(diffraction_batch_size))
        batch_start = wt.UInt32(0)
        batch_stride = wt.UInt32(int(diffraction_batch_size))
        total_samples_u32 = wt.UInt32(int(samples_per_tx))
        total_length_weight = wt.Float(sampler.total_length_scalar / float(max(1, samples_per_tx)))
        all_sample_index = dr.arange(wt.UInt32, int(samples_per_tx))
        state_slot_all = sampler.sample_slots(all_sample_index, seed=int(seed))
        edge_fraction_all = Sampler.hash_uniform(all_sample_index, stream=602, seed=int(seed))
        keller_sample_all = Sampler.hash_uniform(all_sample_index, stream=603, seed=int(seed))

        while dr.hint(
            batch_start < total_samples_u32,
            mode=str(loop_mode),
            max_iterations=max(1, int(diffraction_batch_count)),
            label="radio_map_mc_diffraction",
            exclude=[
                scene,
                grid,
                diffraction_states,
                sampler,
                plane_normal,
                weighted_diagnostics,
                diffraction_tape_store,
            ],
        ):
            sp = Diffraction.sample_diff_point(
                sample_lane=sample_lane,
                batch_start=batch_start,
                total_samples_u32=total_samples_u32,
                diffraction_states=diffraction_states,
                state_slot_all=state_slot_all,
                edge_fraction_all=edge_fraction_all,
                keller_sample_all=keller_sample_all,
            )
            vis = Diffraction.check_visibility(
                diff_point=sp.diff_point,
                diff_point_offset=sp.diff_point_offset,
                source_pos=sp.batch_states.source_pos,
                ko=sp.ko,
                ray_origin=sp.ray_origin,
                diffraction_batch_size=diffraction_batch_size,
                grid=grid,
                sample_active=sp.sample_active,
                scene=scene,
            )
            ev = UTD.eval_diff_contribution(
                oriented=sp.oriented,
                batch_states=sp.batch_states,
                diff_point=sp.diff_point,
                ko=sp.ko,
                plane_hit=vis.plane_hit,
                source_visible=vis.source_visible,
                visible_target=vis.visible_target,
                sample_active=sp.sample_active,
                grid=grid,
                k=k,
                wavelength=wavelength,
                diff_gain_scale=diff_gain_scale,
                total_length_weight=total_length_weight,
                plane_normal=plane_normal,
            )
            if dr.hint(diffraction_tape_store is not None, mode="scalar"):
                diffraction_tape_store.store(
                    edge_index=sp.batch_states.edge_index,
                    edge_fraction=sp.edge_fraction,
                    cone_sample=sp.keller_sample,
                    cell_idx=ev.cell_idx,
                    field_valid=ev.field_support.field_valid,
                    pole_safe=ev.field_support.pole_safe,
                    dif_n_p=ev.field_support.dif_n_p,
                    dif_n_m=ev.field_support.dif_n_m,
                    sum_n_p=ev.field_support.sum_n_p,
                    sum_n_m=ev.field_support.sum_n_m,
                    active=ev.contribution_active,
                )
            contribution_store.store(
                coord_0=vis.plane_hit.coord_0,
                coord_1=vis.plane_hit.coord_1,
                component_power={
                    "diffraction": dr.select(
                        ev.contribution_active, ev.contribution, wt.Float(0.0),
                    ),
                    "diffraction_incident_transition_power": dr.select(
                        ev.contribution_active,
                        ev.contribution * ev.field_support.incident_transition_weight,
                        wt.Float(0.0),
                    ),
                    "diffraction_reflection_transition_power": dr.select(
                        ev.contribution_active,
                        ev.contribution * ev.field_support.reflection_transition_weight,
                        wt.Float(0.0),
                    ),
                },
                active=ev.contribution_active,
            )
            batch_start += batch_stride

        return wt.UInt32(0)


class DiffractionEdgeSampler:
    """Sample and discover diffraction edges for runtime setup."""

    def __init__(
        self,
        *,
        cdf,
        sample_weight,
        n_states: int,
        total_length_scalar: float,
        total_sampling_weight_scalar: float,
    ) -> None:
        self.cdf = cdf
        self.sample_weight = sample_weight
        self.n_states = int(n_states)
        self.total_length_scalar = float(total_length_scalar)
        self.total_sampling_weight_scalar = float(total_sampling_weight_scalar)

    @classmethod
    def from_line_length(cls, line_length) -> DiffractionEdgeSampler | None:
        n_states = int(dr.width(line_length))
        if n_states <= 0:
            return None
        total_length = dr.sum(line_length)
        total_length_scalar = float(scalar(total_length))
        if total_length_scalar <= 0.0:
            return None
        return cls(
            cdf=dr.cumsum(line_length),
            sample_weight=line_length,
            n_states=n_states,
            total_length_scalar=total_length_scalar,
            total_sampling_weight_scalar=total_length_scalar,
        )

    @classmethod
    def from_sample_weight(
        cls,
        *,
        line_length,
        sample_weight,
    ) -> DiffractionEdgeSampler | None:
        n_states = int(dr.width(line_length))
        if n_states <= 0 or int(dr.width(sample_weight)) != n_states:
            return None
        total_length = dr.sum(line_length)
        total_length_scalar = float(scalar(total_length))
        if total_length_scalar <= 0.0:
            return None
        positive_length = line_length > wt.Float(0.0)
        safe_sample_weight = dr.select(
            positive_length,
            dr.maximum(sample_weight, wt.Float(1.0e-12)),
            wt.Float(0.0),
        )
        total_sampling_weight = dr.sum(safe_sample_weight)
        total_sampling_weight_scalar = float(scalar(total_sampling_weight))
        if total_sampling_weight_scalar <= 0.0:
            return None
        return cls(
            cdf=dr.cumsum(safe_sample_weight),
            sample_weight=safe_sample_weight,
            n_states=n_states,
            total_length_scalar=total_length_scalar,
            total_sampling_weight_scalar=total_sampling_weight_scalar,
        )

    def sample_slots(self, sample_index, *, seed: int):
        if self.n_states <= 0 or int(dr.width(sample_index)) <= 0:
            return dr.zeros(wt.UInt32, 0)
        return DiffractionBuilderKernel.sample_slots(
            sample_index=sample_index,
            cdf=self.cdf,
            n_states=self.n_states,
            total_length_scalar=self.total_sampling_weight_scalar,
            seed=int(seed),
        )

    def sample_slots_from_uniform(self, sample_u):
        if self.n_states <= 0 or int(dr.width(sample_u)) <= 0:
            return dr.zeros(wt.UInt32, 0)
        scaled_u = (
            dr.clip(wt.Float(sample_u), wt.Float(0.0), wt.Float(1.0))
            * wt.Float(self.total_sampling_weight_scalar)
        )
        return wt.UInt32(
            dr.binary_search(
                0,
                self.n_states - 1,
                lambda index: dr.gather(wt.Float, self.cdf, index) < scaled_u,
            )
        )

    def edge_measure_weight(self, state_slot, line_length):
        if self.n_states <= 0 or int(dr.width(state_slot)) <= 0:
            return dr.zeros(wt.Float, 0)
        selected_weight = dr.maximum(
            dr.gather(wt.Float, self.sample_weight, state_slot),
            wt.Float(1.0e-12),
        )
        return wt.Float(self.total_sampling_weight_scalar) * line_length / selected_weight

    @staticmethod
    def surface_tangent_from_hit(ray_dir, surface_normal):
        tangent = ray_dir - dr.dot(ray_dir, surface_normal) * surface_normal
        tangent_norm = dr.norm(tangent)
        fallback_x = dr.cross(surface_normal, wt.Vector3f(1.0, 0.0, 0.0))
        fallback_y = dr.cross(surface_normal, wt.Vector3f(0.0, 1.0, 0.0))
        fallback = dr.select(dr.norm(fallback_x) > wt.Float(EPS), fallback_x, fallback_y)
        tangent = dr.select(tangent_norm > wt.Float(EPS), tangent, fallback)
        return tangent / (dr.norm(tangent) + wt.Float(EPS))

    @staticmethod
    def silhouette_viewpoint(hit_p, shading_normal, geometric_normal, ray_dir):
        geometric_normal = dr.select(dr.norm(geometric_normal) > wt.Float(EPS), geometric_normal, shading_normal)
        surface_normal = dr.select(dr.dot(ray_dir, geometric_normal) > 0.0, -geometric_normal, geometric_normal)
        tangent = DiffractionEdgeSampler.surface_tangent_from_hit(ray_dir, surface_normal)
        theta = wt.Float(0.5 * math.pi - 0.05)
        d = dr.cos(theta) * surface_normal + dr.sin(theta) * tangent
        return hit_p + wt.Float(0.1) * d

    @staticmethod
    def closest_point_on_edge(*, query_point, edge_pos, edge_dir, line_min, line_max):
        edge_hat = edge_dir / (dr.norm(edge_dir) + wt.Float(EPS))
        ell = dr.clip(dr.dot(query_point - edge_pos, edge_hat), line_min, line_max)
        return edge_pos + edge_hat * ell, ell

    @staticmethod
    def unique_edge_indices(edge_idx, *, n_edges: int):
        if n_edges <= 0 or int(dr.width(edge_idx)) <= 0:
            return dr.zeros(wt.UInt32, 0)
        active = edge_idx < wt.UInt32(int(n_edges))
        if not dr.any(active):
            return dr.zeros(wt.UInt32, 0)
        safe_edge_idx = dr.select(active, edge_idx, wt.UInt32(0))
        seen = dr.zeros(wt.UInt32, int(n_edges))
        previous = dr.scatter_inc(seen, safe_edge_idx, active)
        unique_lane = dr.compress(active & (previous == 0))
        if int(dr.width(unique_lane)) <= 0:
            return dr.zeros(wt.UInt32, 0)
        return dr.gather(wt.UInt32, safe_edge_idx, unique_lane)

    @classmethod
    def best_edge_indices_from_hit_data(cls, *, tx_pos, ray_directions, prim_index, hit_p, hit_n, hit_geo_n, hit,
                                        scene: Scene):
        n_rays = int(dr.width(ray_directions.x))
        tri_data = scene._triangle_runtime()
        edge_runtime = scene._selected_edge_runtime()
        edge_views = scene._selected_edge_views()
        if tri_data is None or edge_runtime is None or n_rays <= 0 or len(edge_views) <= 0:
            return dr.full(wt.Int32, -1, max(0, n_rays))
        max_surface_edge_count = int(tri_data.get("surface_max_edge_count", 0))
        if max_surface_edge_count <= 0:
            return dr.full(wt.Int32, -1, n_rays)
        prim_index_i32 = wt.Int32(prim_index)
        valid_prim = (prim_index_i32 >= 0) & (prim_index_i32 < wt.Int32(int(tri_data.get("n_triangles", 0))))
        safe_prim_idx = wt.UInt32(dr.select(valid_prim, prim_index_i32, wt.Int32(0)))
        triangle_edge_count = dr.select(
            valid_prim,
            dr.gather(wt.UInt32, tri_data["surface_edge_size"], safe_prim_idx),
            wt.UInt32(0),
        )
        triangle_edge_indices = dr.full(wt.Int32, -1, n_rays * max_surface_edge_count)
        ray_idx = dr.arange(wt.UInt32, n_rays)
        row_base = ray_idx * wt.UInt32(max_surface_edge_count)
        for slot in range(max_surface_edge_count):
            flat_idx = safe_prim_idx * wt.UInt32(max_surface_edge_count) + wt.UInt32(slot)
            slot_values = dr.select(
                valid_prim,
                dr.gather(wt.Int32, tri_data["surface_edge_indices"], flat_idx),
                wt.Int32(-1),
            )
            dr.scatter(
                triangle_edge_indices,
                slot_values,
                row_base + wt.UInt32(slot),
            )
        return DiffractionBuilderKernel.best_edge_indices(
            tx_pos=tx_pos,
            ray_directions=ray_directions,
            hit_p=hit_p,
            hit_n=hit_n,
            hit_geo_n=hit_geo_n,
            hit=hit,
            triangle_edge_count=triangle_edge_count,
            triangle_edge_indices=triangle_edge_indices,
            max_triangle_edge_slots=max_surface_edge_count,
            edge_runtime=edge_runtime,
        )

    @classmethod
    def discover_from_hit_data(cls, *, tx_pos, ray_directions, prim_index, hit_p, hit_n, hit_geo_n, hit,
                               scene: Scene):
        tri_data = scene._triangle_runtime()
        edge_runtime = scene._selected_edge_runtime()
        max_surface_edge_count = 0 if tri_data is None else int(tri_data.get("surface_max_edge_count", 0))
        if tri_data is None or edge_runtime is None or max_surface_edge_count <= 0 or not dr.any(hit):
            return dr.zeros(wt.UInt32, 0)
        discovered_lane = dr.compress(hit)
        n_hits = int(dr.width(discovered_lane))
        if n_hits <= 0:
            return dr.zeros(wt.UInt32, 0)
        compact_ray_dir = dr.gather(wt.Vector3f, ray_directions, discovered_lane)
        compact_prim_index = dr.gather(wt.Int32, wt.Int32(prim_index), discovered_lane)
        compact_hit_p = dr.gather(wt.Point3f, hit_p, discovered_lane)
        compact_hit_n = dr.gather(wt.Vector3f, hit_n, discovered_lane)
        compact_hit_geo_n = dr.gather(wt.Vector3f, hit_geo_n, discovered_lane)
        return DiffractionBuilderKernel.discover_edge_indices_from_hits(
            tx_pos=tx_pos,
            ray_directions=compact_ray_dir,
            prim_index=compact_prim_index,
            hit_p=compact_hit_p,
            hit_n=compact_hit_n,
            hit_geo_n=compact_hit_geo_n,
            n_hits=n_hits,
            triangle_edge_count=tri_data["surface_edge_size"],
            triangle_edge_indices=tri_data["surface_edge_indices"],
            max_triangle_edge_slots=max_surface_edge_count,
            n_triangles=int(tri_data.get("n_triangles", 0)),
            edge_runtime=edge_runtime,
        )

    @classmethod
    def discover_from_hits(cls, *, tx_pos, ray_directions, si, hit, scene: Scene):
        return cls.discover_from_hit_data(
            tx_pos=tx_pos,
            ray_directions=ray_directions,
            prim_index=si.global_prim_id,
            hit_p=si.p,
            hit_n=si.n,
            hit_geo_n=si.geo_n,
            hit=hit,
            scene=scene,
        )


class DiffractionStates:
    def __init__(self, *, edge_index, edge_pos, edge_dir, n0, nn, wedge_n, edge_line_min, edge_line_max,
                 source_pos, adjacent_face0, adjacent_face1,
                 face0_material: FaceMaterial, face1_material: FaceMaterial,
                 source_power=None, prefix_reflection_depth=None,
                 prefix_initial_ray_dir=None, prefix_prim_by_bounce=None,
                 stored_count: int | None = None, capacity: int | None = None, seen=None,
                 next_state_slot=None) -> None:
        (
            self.edge_index, self.edge_pos, self.edge_dir, self.n0, self.n_face_n,
            self.wedge_n, self.edge_line_min, self.edge_line_max, self.source_pos,
            self.adjacent_face0, self.adjacent_face1,
        ) = (
            edge_index, edge_pos, edge_dir, n0, nn, wedge_n, edge_line_min, edge_line_max,
            source_pos, adjacent_face0, adjacent_face1,
        )
        self.face0_material = self._coerce_face_material(face0_material)
        self.face1_material = self._coerce_face_material(face1_material)
        width = int(dr.width(edge_index))
        self.source_power = (
            dr.ones(wt.Float, width)
            if source_power is None
            else source_power
        )
        self.prefix_reflection_depth = (
            dr.zeros(wt.Int32, width)
            if prefix_reflection_depth is None
            else prefix_reflection_depth
        )
        self.prefix_initial_ray_dir = (
            wt.Vector3f(
                dr.zeros(wt.Float, width),
                dr.zeros(wt.Float, width),
                dr.zeros(wt.Float, width),
            )
            if prefix_initial_ray_dir is None
            else prefix_initial_ray_dir
        )
        self.prefix_prim_by_bounce = tuple(prefix_prim_by_bounce or ())
        self.stored_count = None if stored_count is None else int(stored_count)
        self.capacity = None if capacity is None else int(capacity)
        self.seen = seen
        self.next_state_slot = wt.UInt32(0) if next_state_slot is None else next_state_slot

    @staticmethod
    def _coerce_face_material(material) -> FaceMaterial:
        if isinstance(material, FaceMaterial):
            return material
        mu_r = material.get("mu_r")
        if mu_r is None:
            mu_r = dr.full(wt.Float, 1.0, int(dr.width(wt.Float(material["eta_r"]))))
        return FaceMaterial(
            eta_r=material["eta_r"],
            sigma=material["sigma"],
            gain=material["gain"],
            use_fresnel=material["use_fresnel"],
            mu_r=mu_r,
        )

    @classmethod
    def empty(cls, capacity: int, *, track_seen: bool = False) -> DiffractionStates:
        zero_material = lambda: FaceMaterial(
            eta_r=dr.zeros(wt.Float, capacity),
            sigma=dr.zeros(wt.Float, capacity),
            gain=dr.zeros(wt.Float, capacity),
            use_fresnel=dr.zeros(wt.Bool, capacity),
            mu_r=dr.ones(wt.Float, capacity),
        )
        return cls(
            edge_index=dr.full(wt.Int32, -1, capacity),
            edge_pos=dr.zeros(wt.Point3f, capacity),
            edge_dir=dr.zeros(wt.Vector3f, capacity),
            n0=dr.zeros(wt.Vector3f, capacity),
            nn=dr.zeros(wt.Vector3f, capacity),
            wedge_n=dr.zeros(wt.Float, capacity),
            edge_line_min=dr.zeros(wt.Float, capacity),
            edge_line_max=dr.zeros(wt.Float, capacity),
            source_pos=dr.zeros(wt.Point3f, capacity),
            adjacent_face0=dr.full(wt.Int32, -1, capacity),
            adjacent_face1=dr.full(wt.Int32, -1, capacity),
            face0_material=zero_material(),
            face1_material=zero_material(),
            source_power=dr.zeros(wt.Float, capacity),
            prefix_reflection_depth=dr.zeros(wt.Int32, capacity),
            prefix_initial_ray_dir=dr.zeros(wt.Vector3f, capacity),
            prefix_prim_by_bounce=(),
            stored_count=0 if track_seen else None,
            capacity=capacity if track_seen else None,
            seen=dr.zeros(wt.UInt32, capacity) if track_seen else None,
            next_state_slot=wt.UInt32(0),
        )

    @classmethod
    def from_edge_indices(cls, *, tx_pos, edge_idx, scene: Scene,
                          config: ResolvedTraceConfig) -> DiffractionStates | None:
        n_states = int(dr.width(edge_idx))
        if n_states <= 0:
            return None

        edge_runtime = scene._selected_edge_runtime()
        if edge_runtime is None:
            return None
        edge_data = DiffractionBuilderKernel.build_state_arrays(
            edge_idx=wt.UInt32(edge_idx),
            tx_pos=tx_pos,
            edge_runtime=edge_runtime,
        )
        if edge_data is None:
            return None
        tri_data = scene._triangle_runtime()

        def resolve_face_material(face_idx) -> FaceMaterial:
            valid_face = face_idx >= 0
            material_table_available = (
                tri_data is not None
                and bool(tri_data.get("material_has_specified_materials", False))
            )
            if not material_table_available:
                raise RuntimeError(
                    "Monte Carlo diffraction material resolution requires a scene material table. "
                    "Attach witwin.core.Material to every scene structure."
                )
            safe_face = wt.UInt32(dr.select(valid_face, face_idx, wt.Int32(0)))
            return FaceMaterial(
                eta_r=dr.gather(wt.Float, tri_data["material_eps_r"], safe_face),
                sigma=dr.gather(wt.Float, tri_data["material_sigma_e"], safe_face),
                gain=dr.full(wt.Float, 1.0, n_states),
                use_fresnel=valid_face,
                mu_r=dr.gather(wt.Float, tri_data["material_mu_r"], safe_face),
            )

        face0_material = resolve_face_material(edge_data["adjacent_face0"])
        face1_material = resolve_face_material(edge_data["adjacent_face1"])
        return cls(
            edge_index=edge_data["edge_index"],
            edge_pos=edge_data["edge_pos"],
            edge_dir=edge_data["edge_dir"],
            n0=edge_data["n0"],
            nn=edge_data["n_face_n"],
            wedge_n=edge_data["wedge_n"],
            edge_line_min=edge_data["line_min"],
            edge_line_max=edge_data["line_max"],
            source_pos=edge_data["source_pos"],
            adjacent_face0=edge_data["adjacent_face0"],
            adjacent_face1=edge_data["adjacent_face1"],
            face0_material=face0_material,
            face1_material=face1_material,
        )

    @classmethod
    def from_edge_indices_with_sources(cls, *, edge_idx, source_pos, source_power,
                                       prefix_reflection_depth, prefix_initial_ray_dir=None,
                                       prefix_prim_by_bounce=None, scene: Scene,
                                       config: ResolvedTraceConfig) -> DiffractionStates | None:
        n_states = int(dr.width(edge_idx))
        if n_states <= 0:
            return None
        edge_runtime = scene._selected_edge_runtime()
        if edge_runtime is None:
            return None
        base = cls.from_edge_indices(
            tx_pos=source_pos,
            edge_idx=edge_idx,
            scene=scene,
            config=config,
        )
        if base is None:
            return None
        base.source_power = source_power
        base.prefix_reflection_depth = prefix_reflection_depth
        if prefix_initial_ray_dir is not None:
            base.prefix_initial_ray_dir = prefix_initial_ray_dir
        if prefix_prim_by_bounce is not None:
            base.prefix_prim_by_bounce = tuple(prefix_prim_by_bounce)
        return base

    @classmethod
    def concat(cls, states) -> DiffractionStates | None:
        non_empty = [
            state for state in states
            if state is not None and int(dr.width(state.edge_index)) > 0
        ]
        if len(non_empty) == 0:
            return None
        if len(non_empty) == 1:
            return non_empty[0]
        max_prefix_bounces = max(len(state.prefix_prim_by_bounce) for state in non_empty)

        def prefix_prim(state, bounce_slot):
            if bounce_slot < len(state.prefix_prim_by_bounce):
                return state.prefix_prim_by_bounce[bounce_slot]
            return dr.full(wt.Int32, -1, int(dr.width(state.edge_index)))

        return cls(
            edge_index=arrays.concat_ints([state.edge_index for state in non_empty]),
            edge_pos=arrays.concat_points([state.edge_pos for state in non_empty]),
            edge_dir=arrays.concat_vectors([state.edge_dir for state in non_empty]),
            n0=arrays.concat_vectors([state.n0 for state in non_empty]),
            nn=arrays.concat_vectors([state.n_face_n for state in non_empty]),
            wedge_n=arrays.concat_floats([state.wedge_n for state in non_empty]),
            edge_line_min=arrays.concat_floats([state.edge_line_min for state in non_empty]),
            edge_line_max=arrays.concat_floats([state.edge_line_max for state in non_empty]),
            source_pos=arrays.concat_points([state.source_pos for state in non_empty]),
            adjacent_face0=arrays.concat_ints([state.adjacent_face0 for state in non_empty]),
            adjacent_face1=arrays.concat_ints([state.adjacent_face1 for state in non_empty]),
            face0_material=FaceMaterial(
                eta_r=arrays.concat_floats([state.face0_material.eta_r for state in non_empty]),
                sigma=arrays.concat_floats([state.face0_material.sigma for state in non_empty]),
                gain=arrays.concat_floats([state.face0_material.gain for state in non_empty]),
                use_fresnel=arrays.concat_arrays(wt.Bool, [state.face0_material.use_fresnel for state in non_empty]),
                mu_r=arrays.concat_floats([state.face0_material.mu_r for state in non_empty]),
            ),
            face1_material=FaceMaterial(
                eta_r=arrays.concat_floats([state.face1_material.eta_r for state in non_empty]),
                sigma=arrays.concat_floats([state.face1_material.sigma for state in non_empty]),
                gain=arrays.concat_floats([state.face1_material.gain for state in non_empty]),
                use_fresnel=arrays.concat_arrays(wt.Bool, [state.face1_material.use_fresnel for state in non_empty]),
                mu_r=arrays.concat_floats([state.face1_material.mu_r for state in non_empty]),
            ),
            source_power=arrays.concat_floats([state.source_power for state in non_empty]),
            prefix_reflection_depth=arrays.concat_ints([state.prefix_reflection_depth for state in non_empty]),
            prefix_initial_ray_dir=arrays.concat_vectors([state.prefix_initial_ray_dir for state in non_empty]),
            prefix_prim_by_bounce=tuple(
                arrays.concat_ints([prefix_prim(state, bounce_slot) for state in non_empty])
                for bounce_slot in range(max_prefix_bounces)
            ),
        )

    def gather(self, indices) -> DiffractionStates:
        gather = lambda dtype, values: dr.gather(dtype, values, indices)
        gather_face = lambda m: FaceMaterial(
            eta_r=gather(wt.Float, m.eta_r),
            sigma=gather(wt.Float, m.sigma),
            gain=gather(wt.Float, m.gain),
            use_fresnel=gather(wt.Bool, m.use_fresnel),
            mu_r=gather(wt.Float, m.mu_r),
        )
        return DiffractionStates(
            edge_index=gather(wt.Int32, self.edge_index),
            edge_pos=gather(wt.Point3f, self.edge_pos),
            edge_dir=gather(wt.Vector3f, self.edge_dir),
            n0=gather(wt.Vector3f, self.n0),
            nn=gather(wt.Vector3f, self.n_face_n),
            wedge_n=gather(wt.Float, self.wedge_n),
            edge_line_min=gather(wt.Float, self.edge_line_min),
            edge_line_max=gather(wt.Float, self.edge_line_max),
            source_pos=gather(wt.Point3f, self.source_pos),
            adjacent_face0=gather(wt.Int32, self.adjacent_face0),
            adjacent_face1=gather(wt.Int32, self.adjacent_face1),
            face0_material=gather_face(self.face0_material),
            face1_material=gather_face(self.face1_material),
            source_power=gather(wt.Float, self.source_power),
            prefix_reflection_depth=gather(wt.Int32, self.prefix_reflection_depth),
            prefix_initial_ray_dir=gather(wt.Vector3f, self.prefix_initial_ray_dir),
            prefix_prim_by_bounce=tuple(
                gather(wt.Int32, prim_idx) for prim_idx in self.prefix_prim_by_bounce
            ),
        )

    def _to_rayd_state_table(self, scene: Scene):
        return scene._make_rayd_dfr_states(self)

    @staticmethod
    def _face_materials(*, scene, adjacent_face0, adjacent_face1) -> tuple[FaceMaterial, FaceMaterial]:
        if scene is None:
            raise RuntimeError(
                "Monte Carlo edge material resolution requires a scene material table. "
                "Attach witwin.core.Material to every scene structure."
            )
        face0 = resolve_surface_material(
            scene=scene, prim_idx=adjacent_face0, default_gain=1.0,
            valid_mask=adjacent_face0 >= 0,
        )
        face1 = resolve_surface_material(
            scene=scene, prim_idx=adjacent_face1, default_gain=1.0,
            valid_mask=adjacent_face1 >= 0,
        )
        return (face0, face1)

    def orient_face_view(self, incident_dir) -> OrientedEdgeView:
        flip = dr.dot(incident_dir, self.n0) > 0.0
        f0, f1 = self.face0_material, self.face1_material
        return OrientedEdgeView(
            edge_dir=dr.select(flip, -self.edge_dir, self.edge_dir),
            n0=dr.select(flip, self.n_face_n, self.n0),
            nn=dr.select(flip, self.n0, self.n_face_n),
            face0_material=FaceMaterial(
                eta_r=dr.select(flip, f1.eta_r, f0.eta_r),
                sigma=dr.select(flip, f1.sigma, f0.sigma),
                gain=dr.select(flip, f1.gain, f0.gain),
                use_fresnel=dr.select(flip, f1.use_fresnel, f0.use_fresnel),
                mu_r=dr.select(flip, f1.mu_r, f0.mu_r),
            ),
            face1_material=FaceMaterial(
                eta_r=dr.select(flip, f0.eta_r, f1.eta_r),
                sigma=dr.select(flip, f0.sigma, f1.sigma),
                gain=dr.select(flip, f0.gain, f1.gain),
                use_fresnel=dr.select(flip, f0.use_fresnel, f1.use_fresnel),
                mu_r=dr.select(flip, f0.mu_r, f1.mu_r),
            ),
        )

    def store_from_hit_data(self, *, tx_pos, ray_directions, prim_index, hit_p, hit_n, hit_geo_n, hit,
                            scene: Scene, config: ResolvedTraceConfig) -> None:
        if self.capacity is None or self.capacity <= 0 or self.seen is None:
            return

        best_edge_idx = DiffractionEdgeSampler.best_edge_indices_from_hit_data(
            tx_pos=tx_pos,
            ray_directions=ray_directions,
            prim_index=prim_index,
            hit_p=hit_p,
            hit_n=hit_n,
            hit_geo_n=hit_geo_n,
            hit=hit,
            scene=scene,
        )
        store_active = hit & (best_edge_idx >= 0)
        safe_edge_idx = wt.UInt32(dr.select(store_active, best_edge_idx, wt.Int32(0)))
        previous = dr.scatter_inc(self.seen, safe_edge_idx, store_active)
        store_unique = store_active & (previous == 0)

        state_slot = dr.scatter_inc(self.next_state_slot, wt.UInt32(0), store_unique)
        store_unique &= state_slot < wt.UInt32(self.capacity)

        edge_data = scene.gather_edge_subset(best_edge_idx, valid_mask=store_unique)
        n_rays = int(dr.width(ray_directions.x))
        source_pos = arrays.broadcast(tx_pos, n_rays)
        face0_material, face1_material = DiffractionStates._face_materials(
            scene=scene,
            adjacent_face0=edge_data["adjacent_face0"],
            adjacent_face1=edge_data["adjacent_face1"],
        )

        dr.scatter(self.edge_pos, edge_data["pos"], state_slot, store_unique)
        dr.scatter(self.edge_index, wt.Int32(best_edge_idx), state_slot, store_unique)
        dr.scatter(self.edge_dir, edge_data["edge_dir"], state_slot, store_unique)
        dr.scatter(self.n0, edge_data["n0"], state_slot, store_unique)
        dr.scatter(self.n_face_n, edge_data["n_face_n"], state_slot, store_unique)
        dr.scatter(self.wedge_n, edge_data["wedge_n"], state_slot, store_unique)
        dr.scatter(self.edge_line_min, edge_data["line_min"], state_slot, store_unique)
        dr.scatter(self.edge_line_max, edge_data["line_max"], state_slot, store_unique)
        dr.scatter(self.source_pos, source_pos, state_slot, store_unique)
        dr.scatter(self.adjacent_face0, edge_data["adjacent_face0"], state_slot, store_unique)
        dr.scatter(self.adjacent_face1, edge_data["adjacent_face1"], state_slot, store_unique)
        dr.scatter(self.face0_material.eta_r, face0_material.eta_r, state_slot, store_unique)
        dr.scatter(self.face0_material.sigma, face0_material.sigma, state_slot, store_unique)
        dr.scatter(self.face0_material.gain, face0_material.gain, state_slot, store_unique)
        dr.scatter(self.face0_material.use_fresnel, face0_material.use_fresnel, state_slot, store_unique)
        dr.scatter(self.face0_material.mu_r, face0_material.mu_r, state_slot, store_unique)
        dr.scatter(self.face1_material.eta_r, face1_material.eta_r, state_slot, store_unique)
        dr.scatter(self.face1_material.sigma, face1_material.sigma, state_slot, store_unique)
        dr.scatter(self.face1_material.gain, face1_material.gain, state_slot, store_unique)
        dr.scatter(self.face1_material.use_fresnel, face1_material.use_fresnel, state_slot, store_unique)
        dr.scatter(self.face1_material.mu_r, face1_material.mu_r, state_slot, store_unique)
__all__ = [
    "Diffraction",
    "DiffractionEdgeSampler",
    "DiffractionHitStore",
    "DiffractionStates",
    "DiffractionTape",
    "DiffractionTapeStore",
    "FaceMaterial",
    "OrientedEdgeView",
    "SampledDiffPoint",
    "VisibilityResult",
]
