"""BDPT wedge-diffraction MIS strategies and result container."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping

import drjit as dr

from witwin.channel.core.scene import Scene
from witwin.channel.core.numerics import arrays
from witwin.channel.core import geometry
from witwin.channel.core.physics import wave_math
from witwin.channel.core.numerics.arrays import scalar
from witwin.channel.montecarlo import types as wt

from ..config import ResolvedTraceConfig
from witwin.channel.core.physics.materials import resolve_surface_material
from .. import grid_ops
from ..grid_ops import GridContributionStore
from ..trace.diffraction import DiffractionStates, FaceMaterial, MC_DIFFRACTION_OFFSET, OrientedEdgeView
from ..trace.diffraction import DiffractionEdgeSampler
from ..trace.diffraction_utd import UTD
from ..sampler import Sampler


@dataclass(slots=True)
class BDPTDiffractionTape:
    strategy: object
    order: object
    cell_idx: object
    coord_0: object
    coord_1: object
    scalar_weight: object
    keller_sample: object
    suffix_prim_idx: object
    source_pos: object
    source_power: object
    prefix_reflection_depth: object
    prefix_initial_ray_dir: object
    prefix_prim_by_bounce: tuple[object, ...]
    edge_index_by_step: tuple[object, object, object]
    edge_fraction_by_step: tuple[object, object, object]

    @staticmethod
    def empty() -> "BDPTDiffractionTape":
        zero_float = dr.zeros(wt.Float, 0)
        zero_int = dr.zeros(wt.Int32, 0)
        zero_uint = dr.zeros(wt.UInt32, 0)
        zero_point = arrays.empty_point3()
        return BDPTDiffractionTape(
            strategy=zero_int,
            order=zero_int,
            cell_idx=zero_uint,
            coord_0=zero_float,
            coord_1=zero_float,
            scalar_weight=zero_float,
            keller_sample=zero_float,
            suffix_prim_idx=zero_int,
            source_pos=zero_point,
            source_power=zero_float,
            prefix_reflection_depth=zero_int,
            prefix_initial_ray_dir=arrays.empty_vector3(),
            prefix_prim_by_bounce=(),
            edge_index_by_step=(zero_int, zero_int, zero_int),
            edge_fraction_by_step=(zero_float, zero_float, zero_float),
        )


@dataclass(frozen=True, slots=True)
class BDPTDiffractionResult:
    """Diffraction MIS summary."""

    path_count: object
    state_count: int
    prefix_state_count: int
    edge_indices: object | None
    total_edge_length: float
    strategy_counts: Mapping[str, int]
    strategy_samples: Mapping[str, int]
    order_counts: Mapping[int, Mapping[str, int]]
    order_samples: Mapping[int, Mapping[str, int]]
    tape: BDPTDiffractionTape | None = None

    @classmethod
    def zero(
        cls,
        *,
        state_count: int = 0,
        prefix_state_count: int = 0,
        order_counts: Mapping[int, Mapping[str, int]] | None = None,
        order_samples: Mapping[int, Mapping[str, int]] | None = None,
    ) -> "BDPTDiffractionResult":
        zero_counts = BDPTDiffractionMIS._zero_strategy_counts()
        return cls(
            path_count=wt.UInt32(0),
            state_count=int(state_count),
            prefix_state_count=int(prefix_state_count),
            edge_indices=dr.zeros(wt.UInt32, 0),
            total_edge_length=0.0,
            strategy_counts=zero_counts,
            strategy_samples=dict(zero_counts),
            order_counts=dict(order_counts or {}),
            order_samples=dict(order_samples or {}),
            tape=None,
        )


class BDPTDiffractionTapeStore:
    """Fixed-width accepted-contribution tape for BDPT diffraction AD replay."""

    DIRECT_ID = 0
    KELLER_ID = 1
    SUFFIX_REFLECTION_ID = 2
    MAX_DEPTH = 3

    def __init__(self, *, capacity: int, max_prefix_bounces: int = 0) -> None:
        self.capacity = max(0, int(capacity))
        self.max_prefix_bounces = max(0, int(max_prefix_bounces))
        self.next_slot = wt.UInt32(0)
        zero_float = dr.zeros(wt.Float, self.capacity)
        zero_int = dr.zeros(wt.Int32, self.capacity)
        self.strategy = dr.full(wt.Int32, -1, self.capacity)
        self.order = dr.zeros(wt.Int32, self.capacity)
        self.cell_idx = dr.zeros(wt.UInt32, self.capacity)
        self.coord_0 = zero_float
        self.coord_1 = zero_float
        self.scalar_weight = zero_float
        self.keller_sample = zero_float
        self.suffix_prim_idx = dr.full(wt.Int32, -1, self.capacity)
        self.source_pos = wt.Point3f(zero_float, zero_float, zero_float)
        self.source_power = zero_float
        self.prefix_reflection_depth = zero_int
        self.prefix_initial_ray_dir = wt.Vector3f(zero_float, zero_float, zero_float)
        self.prefix_prim_by_bounce = tuple(
            dr.full(wt.Int32, -1, self.capacity)
            for _ in range(self.max_prefix_bounces)
        )
        self.edge_index_by_step = tuple(
            dr.full(wt.Int32, -1, self.capacity)
            for _ in range(self.MAX_DEPTH)
        )
        self.edge_fraction_by_step = tuple(
            dr.zeros(wt.Float, self.capacity)
            for _ in range(self.MAX_DEPTH)
        )

    @staticmethod
    def _pad_tuple(values, *, fill, length: int = MAX_DEPTH):
        result = list(values)
        while len(result) < int(length):
            result.append(fill)
        return tuple(result[: int(length)])

    def store(
        self,
        *,
        strategy_id: int,
        order: int,
        cell_idx,
        coord_0,
        coord_1,
        scalar_weight,
        keller_sample,
        suffix_prim_idx,
        source_pos,
        source_power,
        prefix_reflection_depth,
        prefix_initial_ray_dir,
        prefix_prim_by_bounce,
        edge_indices,
        edge_fractions,
        active,
    ) -> None:
        if self.capacity <= 0:
            return
        slot = dr.scatter_inc(self.next_slot, wt.UInt32(0), active)
        store_mask = active & (slot < wt.UInt32(self.capacity))
        width = int(dr.width(coord_0))
        zero_i = dr.full(wt.Int32, -1, width)
        zero_f = dr.zeros(wt.Float, width)
        padded_edge_indices = self._pad_tuple(edge_indices, fill=zero_i)
        padded_edge_fractions = self._pad_tuple(edge_fractions, fill=zero_f)
        dr.scatter(self.strategy, dr.full(wt.Int32, int(strategy_id), width), slot, store_mask)
        dr.scatter(self.order, dr.full(wt.Int32, int(order), width), slot, store_mask)
        dr.scatter(self.cell_idx, wt.UInt32(cell_idx), slot, store_mask)
        dr.scatter(self.coord_0, coord_0, slot, store_mask)
        dr.scatter(self.coord_1, coord_1, slot, store_mask)
        dr.scatter(self.scalar_weight, scalar_weight, slot, store_mask)
        dr.scatter(self.keller_sample, keller_sample, slot, store_mask)
        dr.scatter(self.suffix_prim_idx, suffix_prim_idx, slot, store_mask)
        dr.scatter(self.source_pos, source_pos, slot, store_mask)
        dr.scatter(self.source_power, source_power, slot, store_mask)
        dr.scatter(self.prefix_reflection_depth, prefix_reflection_depth, slot, store_mask)
        dr.scatter(self.prefix_initial_ray_dir, prefix_initial_ray_dir, slot, store_mask)
        for bounce_slot, prim_idx in enumerate(tuple(prefix_prim_by_bounce)[: self.max_prefix_bounces]):
            dr.scatter(self.prefix_prim_by_bounce[bounce_slot], prim_idx, slot, store_mask)
        for step in range(self.MAX_DEPTH):
            dr.scatter(
                self.edge_index_by_step[step],
                padded_edge_indices[step],
                slot,
                store_mask,
            )
            dr.scatter(
                self.edge_fraction_by_step[step],
                padded_edge_fractions[step],
                slot,
                store_mask,
            )

    def finalize(self) -> BDPTDiffractionTape:
        count = int(scalar(self.next_slot))
        if count <= 0:
            return BDPTDiffractionTape.empty()
        indices = dr.arange(wt.UInt32, count)
        return BDPTDiffractionTape(
            strategy=dr.gather(wt.Int32, self.strategy, indices),
            order=dr.gather(wt.Int32, self.order, indices),
            cell_idx=dr.gather(wt.UInt32, self.cell_idx, indices),
            coord_0=dr.gather(wt.Float, self.coord_0, indices),
            coord_1=dr.gather(wt.Float, self.coord_1, indices),
            scalar_weight=dr.gather(wt.Float, self.scalar_weight, indices),
            keller_sample=dr.gather(wt.Float, self.keller_sample, indices),
            suffix_prim_idx=dr.gather(wt.Int32, self.suffix_prim_idx, indices),
            source_pos=dr.gather(wt.Point3f, self.source_pos, indices),
            source_power=dr.gather(wt.Float, self.source_power, indices),
            prefix_reflection_depth=dr.gather(wt.Int32, self.prefix_reflection_depth, indices),
            prefix_initial_ray_dir=dr.gather(wt.Vector3f, self.prefix_initial_ray_dir, indices),
            prefix_prim_by_bounce=tuple(
                dr.gather(wt.Int32, prim_idx, indices)
                for prim_idx in self.prefix_prim_by_bounce
            ),
            edge_index_by_step=tuple(
                dr.gather(wt.Int32, edge_index, indices)
                for edge_index in self.edge_index_by_step
            ),
            edge_fraction_by_step=tuple(
                dr.gather(wt.Float, edge_fraction, indices)
                for edge_fraction in self.edge_fraction_by_step
            ),
        )


class BDPTDiffractionEdgeUseStore:
    """Unique first-order edge set accepted by BDPT diffraction sampling."""

    def __init__(self, *, n_edges: int) -> None:
        self.n_edges = max(0, int(n_edges))
        self.seen = dr.zeros(wt.UInt32, self.n_edges)

    def store(self, *, edge_index, active) -> None:
        if self.n_edges <= 0 or int(dr.width(edge_index)) <= 0:
            return
        edge_index_i32 = wt.Int32(edge_index)
        valid = (
            active
            & (edge_index_i32 >= wt.Int32(0))
            & (edge_index_i32 < wt.Int32(self.n_edges))
        )
        safe_edge_index = wt.UInt32(dr.select(valid, edge_index_i32, wt.Int32(0)))
        dr.scatter_inc(self.seen, safe_edge_index, valid)

    def finalize(self):
        if self.n_edges <= 0:
            return dr.zeros(wt.UInt32, 0)
        lanes = dr.compress(self.seen > wt.UInt32(0))
        if int(dr.width(lanes)) <= 0:
            return dr.zeros(wt.UInt32, 0)
        return wt.UInt32(lanes)


class BDPTDiffractionMIS:
    """Wedge-diffraction balance-MIS strategies for BDPT path connections."""

    DIRECT_STRATEGY = "direct_wedge_connection"
    KELLER_STRATEGY = "keller_cone_plane_hit"
    SUFFIX_REFLECTION_STRATEGY = "specular_suffix_connection"
    MAX_SUPPORTED_DIFFRACTION_DEPTH = 3
    HASH_SEQUENCE = "hash"
    SOBOL_SEQUENCE = "sobol"
    FIRST_ORDER_IMPORTANCE_MIX = 0.75
    FIRST_ORDER_IMPORTANCE_MAX = 16.0

    @staticmethod
    def _zero_strategy_counts() -> dict[str, int]:
        return {
            BDPTDiffractionMIS.DIRECT_STRATEGY: 0,
            BDPTDiffractionMIS.KELLER_STRATEGY: 0,
            BDPTDiffractionMIS.SUFFIX_REFLECTION_STRATEGY: 0,
        }

    @staticmethod
    def sample_uniform(
        sample_index,
        *,
        stream: int,
        dimension: int,
        seed: int,
        sample_sequence: str,
    ):
        resolved_sequence = str(sample_sequence)
        if resolved_sequence == BDPTDiffractionMIS.SOBOL_SEQUENCE:
            return Sampler.sobol_uniform(
                sample_index,
                dimension=int(dimension),
                seed=int(seed),
            )
        if resolved_sequence == BDPTDiffractionMIS.HASH_SEQUENCE:
            return Sampler.hash_uniform(
                sample_index,
                stream=int(stream),
                seed=int(seed),
            )
        raise ValueError(f"Unsupported BDPT diffraction sample sequence: {resolved_sequence!r}.")

    @staticmethod
    def first_order_importance_sampler(
        *,
        states,
        line_length,
        grid,
        fallback_sampler: DiffractionEdgeSampler,
    ) -> DiffractionEdgeSampler:
        n_states = int(dr.width(line_length))
        if n_states <= 0:
            return fallback_sampler
        edge_hat = states.edge_dir / (dr.norm(states.edge_dir) + wt.Float(1.0e-6))
        edge_mid = states.edge_pos + edge_hat * (
            wt.Float(0.5) * (states.edge_line_min + states.edge_line_max)
        )
        grid_center = arrays.broadcast_point(wt.Point3f(*grid.center), n_states)
        to_grid = grid_center - edge_mid
        dist2 = dr.maximum(dr.squared_norm(to_grid), wt.Float(1.0e-6))
        target_dir = to_grid * dr.rsqrt(dist2)
        plane_normal = arrays.broadcast_vector(
            Sampler.axis_unit_normal(str(grid.axis)),
            n_states,
        )
        projected_receiver = dr.maximum(
            dr.abs(dr.dot(target_dir, plane_normal)),
            wt.Float(5.0e-2),
        )
        receiver_area = float(grid.size[0] * grid.size[1])
        receiver_proxy = wt.Float(receiver_area) * projected_receiver / dist2
        source_dist2 = dr.maximum(
            dr.squared_norm(edge_mid - states.source_pos),
            wt.Float(1.0e-6),
        )
        source_proxy = dr.rsqrt(source_dist2)
        power_proxy = dr.maximum(states.source_power, wt.Float(1.0e-6))
        raw_proxy = receiver_proxy * source_proxy * power_proxy
        proxy_sum = dr.sum(raw_proxy * line_length)
        proxy_mean = float(scalar(proxy_sum)) / max(
            float(fallback_sampler.total_length_scalar),
            1.0e-12,
        )
        if not math.isfinite(proxy_mean) or proxy_mean <= 0.0:
            return fallback_sampler
        normalized_proxy = dr.minimum(
            raw_proxy / wt.Float(proxy_mean),
            wt.Float(BDPTDiffractionMIS.FIRST_ORDER_IMPORTANCE_MAX),
        )
        mixture = float(BDPTDiffractionMIS.FIRST_ORDER_IMPORTANCE_MIX)
        sample_weight = line_length * (
            wt.Float(1.0 - mixture) + wt.Float(mixture) * normalized_proxy
        )
        return (
            DiffractionEdgeSampler.from_sample_weight(
                line_length=line_length,
                sample_weight=sample_weight,
            )
            or fallback_sampler
        )

    @staticmethod
    def prepare_direct_states(*, tx_pos, scene: Scene, config: ResolvedTraceConfig):
        edge_runtime = scene._selected_edge_runtime()
        if edge_runtime is None:
            return None
        n_edges = int(edge_runtime.get("n_edges", 0))
        if n_edges <= 0:
            return None
        edge_idx = dr.arange(wt.UInt32, n_edges)
        return DiffractionStates.from_edge_indices(
            tx_pos=tx_pos,
            edge_idx=edge_idx,
            scene=scene,
            config=config,
        )

    @staticmethod
    def prepare_prefix_states(*, prefix_store, scene: Scene, config: ResolvedTraceConfig):
        if prefix_store is None:
            return None
        store_valid = getattr(prefix_store, "valid", None)
        if store_valid is not None:
            if not dr.any(store_valid):
                return None
            lane = dr.compress(store_valid)
            count = int(dr.width(lane))
        else:
            count = int(scalar(getattr(prefix_store, "count", wt.UInt32(0))))
            lane = dr.arange(wt.UInt32, count)
        if count <= 0:
            return None
        ray_directions = dr.gather(wt.Vector3f, prefix_store.ray_directions, lane)
        prim_index = dr.gather(wt.Int32, prefix_store.prim_index, lane)
        hit_p = dr.gather(wt.Point3f, prefix_store.hit_p, lane)
        hit_n = dr.gather(wt.Vector3f, prefix_store.hit_n, lane)
        hit_geo_n = dr.gather(wt.Vector3f, prefix_store.hit_geo_n, lane)
        source_pos = dr.gather(wt.Point3f, prefix_store.source_pos, lane)
        source_power = dr.gather(wt.Float, prefix_store.source_power, lane)
        prefix_depth = dr.gather(wt.Int32, prefix_store.prefix_reflection_depth, lane)
        prefix_initial_ray_dir = dr.gather(wt.Vector3f, prefix_store.prefix_initial_ray_dir, lane)
        prefix_prim_by_bounce = tuple(
            dr.gather(wt.Int32, prim_idx, lane)
            for prim_idx in prefix_store.prefix_prim_by_bounce
        )
        active = dr.full(wt.Bool, True, count)
        edge_idx = DiffractionEdgeSampler.best_edge_indices_from_hit_data(
            tx_pos=source_pos,
            ray_directions=ray_directions,
            prim_index=prim_index,
            hit_p=hit_p,
            hit_n=hit_n,
            hit_geo_n=hit_geo_n,
            hit=active,
            scene=scene,
        )
        valid = edge_idx >= wt.Int32(0)
        if not dr.any(valid):
            return None
        valid_lane = dr.compress(valid)
        compact_edge_idx = wt.UInt32(dr.gather(wt.Int32, edge_idx, valid_lane))
        return DiffractionStates.from_edge_indices_with_sources(
            edge_idx=compact_edge_idx,
            source_pos=dr.gather(wt.Point3f, source_pos, valid_lane),
            source_power=dr.gather(wt.Float, source_power, valid_lane),
            prefix_reflection_depth=dr.gather(wt.Int32, prefix_depth, valid_lane),
            prefix_initial_ray_dir=dr.gather(wt.Vector3f, prefix_initial_ray_dir, valid_lane),
            prefix_prim_by_bounce=tuple(
                dr.gather(wt.Int32, prim_idx, valid_lane)
                for prim_idx in prefix_prim_by_bounce
            ),
            scene=scene,
            config=config,
        )

    @staticmethod
    def prepare_states(*, tx_pos, scene: Scene, config: ResolvedTraceConfig, prefix_store=None):
        direct_states = BDPTDiffractionMIS.prepare_direct_states(
            tx_pos=tx_pos,
            scene=scene,
            config=config,
        )
        prefix_states = BDPTDiffractionMIS.prepare_prefix_states(
            prefix_store=prefix_store,
            scene=scene,
            config=config,
        )
        initial_states = DiffractionStates.concat((direct_states, prefix_states))
        return {
            "initial": initial_states,
            "recursive": direct_states,
            "prefix_state_count": 0 if prefix_states is None else int(dr.width(prefix_states.edge_index)),
        }

    @staticmethod
    def allocate_samples_for_order(
        samples: int,
        *,
        include_suffix_reflection: bool = True,
    ) -> dict[str, int]:
        total = max(0, int(samples))
        if total <= 0:
            return BDPTDiffractionMIS._zero_strategy_counts()
        if not include_suffix_reflection:
            direct_samples = (total + 1) // 2
            keller_samples = total - direct_samples
            return {
                BDPTDiffractionMIS.DIRECT_STRATEGY: int(direct_samples),
                BDPTDiffractionMIS.KELLER_STRATEGY: int(keller_samples),
                BDPTDiffractionMIS.SUFFIX_REFLECTION_STRATEGY: 0,
            }
        direct_samples = (total + 2) // 3
        keller_samples = (total + 1) // 3
        suffix_samples = total - direct_samples - keller_samples
        return {
            BDPTDiffractionMIS.DIRECT_STRATEGY: int(direct_samples),
            BDPTDiffractionMIS.KELLER_STRATEGY: int(keller_samples),
            BDPTDiffractionMIS.SUFFIX_REFLECTION_STRATEGY: int(suffix_samples),
        }

    @staticmethod
    def allocate_samples(
        samples_per_tx: int,
        max_depth: int = 1,
        *,
        include_suffix_reflection: bool = True,
    ) -> dict[int, dict[str, int]]:
        total = max(0, int(samples_per_tx))
        depth = max(1, min(int(max_depth), BDPTDiffractionMIS.MAX_SUPPORTED_DIFFRACTION_DEPTH))
        if total <= 0:
            return {
                order: BDPTDiffractionMIS._zero_strategy_counts()
                for order in range(1, depth + 1)
            }
        base = total // depth
        remainder = total - base * depth
        return {
            order: BDPTDiffractionMIS.allocate_samples_for_order(
                base + (1 if order <= remainder else 0),
                include_suffix_reflection=include_suffix_reflection,
            )
            for order in range(1, depth + 1)
        }

    @staticmethod
    def exterior_angle(n0, nn):
        wedge_interior = dr.safe_acos(
            dr.clip(
                -dr.dot(
                    n0 / (dr.norm(n0) + wt.Float(1.0e-6)),
                    nn / (dr.norm(nn) + wt.Float(1.0e-6)),
                ),
                wt.Float(-1.0),
                wt.Float(1.0),
            )
        )
        return wt.Float(2.0 * math.pi) - wedge_interior

    @staticmethod
    def keller_mis_weight(
        *,
        direct_samples: int,
        keller_samples: int,
        total_edge_length: float,
        n_cells: int,
        cell_area: float,
        exterior_angle,
        integration_weight,
    ):
        if int(direct_samples) <= 0:
            return wt.Float(1.0)
        safe_total_length = wt.Float(max(float(total_edge_length), 1.0e-12))
        safe_exterior = dr.maximum(exterior_angle, wt.Float(1.0e-6))
        safe_iw = dr.maximum(integration_weight, wt.Float(1.0e-9))
        keller_pdf = wt.Float(1.0) / (safe_total_length * safe_exterior * safe_iw)
        direct_area_pdf = wt.Float(1.0) / (
            safe_total_length
            * wt.Float(max(1, int(n_cells)))
            * wt.Float(max(float(cell_area), 1.0e-12))
        )
        keller_score = wt.Float(max(0, int(keller_samples))) * keller_pdf
        direct_score = wt.Float(max(0, int(direct_samples))) * direct_area_pdf
        return keller_score / (keller_score + direct_score + wt.Float(1.0e-12))

    @staticmethod
    def direct_mis_weight(
        *,
        direct_samples: int,
        keller_samples: int,
        total_edge_length: float,
        n_cells: int,
        cell_area: float,
        exterior_angle,
        integration_weight,
    ):
        if int(keller_samples) <= 0:
            return wt.Float(1.0)
        keller_weight = BDPTDiffractionMIS.keller_mis_weight(
            direct_samples=int(direct_samples),
            keller_samples=int(keller_samples),
            total_edge_length=float(total_edge_length),
            n_cells=int(n_cells),
            cell_area=float(cell_area),
            exterior_angle=exterior_angle,
            integration_weight=integration_weight,
        )
        return dr.clip(
            wt.Float(1.0) - keller_weight,
            wt.Float(0.0),
            wt.Float(1.0),
        )

    @staticmethod
    def sample_edge_point(
        *,
        sample_lane,
        batch_size: int,
        samples: int,
        states,
        sampler: DiffractionEdgeSampler,
        seed: int,
        sample_sequence: str,
        state_dimension: int,
        state_stream: int,
        edge_fraction_dimension: int,
        edge_fraction_stream: int,
        batch_idx: int = 0,
        batch_start=None,
    ):
        if batch_start is None:
            batch_start = wt.UInt32(int(batch_idx) * int(batch_size))
        else:
            batch_start = wt.UInt32(batch_start)
        sample_index = sample_lane + batch_start
        sample_active = sample_index < wt.UInt32(int(samples))
        safe_sample_index = wt.UInt32(dr.select(sample_active, sample_index, wt.UInt32(0)))
        state_sample = BDPTDiffractionMIS.sample_uniform(
            safe_sample_index,
            stream=int(state_stream),
            dimension=int(state_dimension),
            seed=int(seed),
            sample_sequence=str(sample_sequence),
        )
        state_slot = sampler.sample_slots_from_uniform(state_sample)
        batch_states = states.gather(state_slot)
        edge_hat = batch_states.edge_dir / (dr.norm(batch_states.edge_dir) + wt.Float(1.0e-6))
        line_length = dr.maximum(
            batch_states.edge_line_max - batch_states.edge_line_min,
            wt.Float(0.0),
        )
        edge_measure_weight = sampler.edge_measure_weight(state_slot, line_length)
        edge_fraction = BDPTDiffractionMIS.sample_uniform(
            safe_sample_index,
            dimension=int(edge_fraction_dimension),
            stream=int(edge_fraction_stream),
            seed=int(seed),
            sample_sequence=str(sample_sequence),
        )
        ell = batch_states.edge_line_min + line_length * edge_fraction
        diff_point = batch_states.edge_pos + edge_hat * ell
        face_sum = batch_states.n0 + batch_states.n_face_n
        face_sum_norm = dr.norm(face_sum)
        offset_normal = dr.select(
            face_sum_norm > wt.Float(1.0e-6),
            face_sum / face_sum_norm,
            wt.Vector3f(0.0, 0.0, 0.0),
        )
        diff_point_offset = diff_point + wt.Float(MC_DIFFRACTION_OFFSET) * offset_normal
        return {
            "sample_index": safe_sample_index,
            "sample_active": sample_active,
            "batch_states": batch_states,
            "edge_measure_weight": edge_measure_weight,
            "edge_fraction": edge_fraction,
            "diff_point": diff_point,
            "diff_point_offset": diff_point_offset,
            "offset_normal": offset_normal,
        }

    @staticmethod
    def orient_sample(*, batch_states, diff_point, source_pos) -> OrientedEdgeView:
        incident_dir = diff_point - source_pos
        return batch_states.orient_face_view(incident_dir)

    @staticmethod
    def _segment_pair_visible(scene: Scene, a, b, b_offset):
        return scene.segment_pair_visible(a, b, b_offset)

    @staticmethod
    def _sample_target_cell(*, sample_index, stream: int, dimension: int, seed: int,
                            sample_sequence: str, grid):
        n_cells = int(grid.n_cells)
        cell_u = BDPTDiffractionMIS.sample_uniform(
            sample_index,
            stream=int(stream),
            dimension=int(dimension),
            seed=int(seed),
            sample_sequence=str(sample_sequence),
        )
        cell_idx = wt.UInt32(
            dr.minimum(
                wt.Float(max(0, n_cells - 1)),
                dr.floor(cell_u * wt.Float(max(1, n_cells))),
            )
        )
        return {
            "cell_idx": cell_idx,
            "target_pos": dr.gather(wt.Point3f, grid.cell_centers, cell_idx),
            "coord_0": dr.gather(wt.Float, grid.grid_x, cell_idx),
            "coord_1": dr.gather(wt.Float, grid.grid_y, cell_idx),
        }

    @staticmethod
    def _store_diffraction_contribution(
        *,
        contribution_store: GridContributionStore,
        coord_0,
        coord_1,
        contribution,
        field_support,
        active,
    ):
        contribution_store.store(
            coord_0=coord_0,
            coord_1=coord_1,
            component_power={
                "diffraction": contribution,
                "diffraction_incident_transition_power": (
                    contribution * field_support.incident_transition_weight
                ),
                "diffraction_reflection_transition_power": (
                    contribution * field_support.reflection_transition_weight
                ),
            },
            active=active,
        )

    @staticmethod
    def _utd_power(*, source_pos, orient: OrientedEdgeView, batch_states, diff_point, target_pos,
                   config: ResolvedTraceConfig):
        return UTD.edge_diffraction_power(
            source_pos=source_pos,
            oriented=orient,
            wedge_n=batch_states.wedge_n,
            sampled_edge_pos=diff_point,
            target_pos=target_pos,
            k=config.k,
            wavelength=config.wavelength,
        )

    @staticmethod
    @dr.syntax
    def trace_direct_batches(
        *,
        scene: Scene,
        grid,
        states,
        sampler: DiffractionEdgeSampler,
        batch_size: int,
        batch_count: int,
        direct_samples: int,
        keller_samples: int,
        seed: int,
        sample_sequence: str,
        config: ResolvedTraceConfig,
        weighted_diagnostics: dict,
        cell_area: float,
        contribution_store: GridContributionStore,
        loop_mode: str,
        tape_store: BDPTDiffractionTapeStore | None = None,
        edge_use_store: BDPTDiffractionEdgeUseStore | None = None,
    ):
        sample_lane = dr.arange(wt.UInt32, int(batch_size))
        batch_start = wt.UInt32(0)
        batch_stride = wt.UInt32(int(batch_size))
        total_samples_u32 = wt.UInt32(int(direct_samples))
        start_slot = wt.UInt32(contribution_store.next_slot)
        n_cells = int(grid.n_cells)
        direct_gain_scale = wt.Float((float(config.wavelength) / (4.0 * math.pi)) ** 2)
        plane_normal = Sampler.axis_unit_normal(str(grid.axis))

        while dr.hint(
            batch_start < total_samples_u32,
            mode=str(loop_mode),
            max_iterations=max(1, int(batch_count)),
            label="bdpt_direct_diffraction",
            exclude=[scene, grid, states, sampler, config, weighted_diagnostics],
        ):
            sp = BDPTDiffractionMIS.sample_edge_point(
                sample_lane=sample_lane,
                batch_start=batch_start,
                batch_size=batch_size,
                samples=direct_samples,
                states=states,
                sampler=sampler,
                seed=int(seed),
                sample_sequence=str(sample_sequence),
                state_dimension=0,
                state_stream=601,
                edge_fraction_dimension=1,
                edge_fraction_stream=702,
            )

            cell = BDPTDiffractionMIS._sample_target_cell(
                sample_index=sp["sample_index"],
                stream=703,
                dimension=2,
                seed=int(seed),
                sample_sequence=str(sample_sequence),
                grid=grid,
            )
            orient = BDPTDiffractionMIS.orient_sample(
                batch_states=sp["batch_states"],
                diff_point=sp["diff_point"],
                source_pos=sp["batch_states"].source_pos,
            )
            source_visible = BDPTDiffractionMIS._segment_pair_visible(
                scene, sp["batch_states"].source_pos, sp["diff_point"], sp["diff_point_offset"]
            )
            target_visible = BDPTDiffractionMIS._segment_pair_visible(
                scene, cell["target_pos"], sp["diff_point"], sp["diff_point_offset"]
            )
            field_power, field_valid, field_support = BDPTDiffractionMIS._utd_power(
                source_pos=sp["batch_states"].source_pos,
                orient=orient,
                batch_states=sp["batch_states"],
                diff_point=sp["diff_point"],
                target_pos=cell["target_pos"],
                config=config,
            )
            target_dir = cell["target_pos"] - sp["diff_point"]
            target_dir = target_dir / (dr.norm(target_dir) + wt.Float(1.0e-6))
            exterior_angle = BDPTDiffractionMIS.exterior_angle(orient.n0, orient.nn)
            integration_weight = UTD.integration_weight(
                edge_origin=sp["batch_states"].edge_pos,
                edge_dir=orient.edge_dir,
                n0=orient.n0,
                source_pos=sp["batch_states"].source_pos,
                diff_point=sp["diff_point"],
                k_world=target_dir,
                target_pos=cell["target_pos"],
                plane_normal=plane_normal,
            )
            mis_weight = BDPTDiffractionMIS.direct_mis_weight(
                direct_samples=int(direct_samples),
                keller_samples=int(keller_samples),
                total_edge_length=float(sampler.total_length_scalar),
                n_cells=n_cells,
                cell_area=float(cell_area),
                exterior_angle=exterior_angle,
                integration_weight=integration_weight,
            )
            contribution_active = sp["sample_active"] & source_visible & target_visible & field_valid
            sample_weight = sp["edge_measure_weight"] * wt.Float(
                float(max(1, n_cells)) / float(max(1, direct_samples))
            )
            contribution = dr.select(
                contribution_active,
                (
                    sp["batch_states"].source_power
                    * field_power
                    * direct_gain_scale
                    * sample_weight
                    * mis_weight
                ),
                wt.Float(0.0),
            )
            BDPTDiffractionMIS._store_diffraction_contribution(
                contribution_store=contribution_store,
                coord_0=cell["coord_0"],
                coord_1=cell["coord_1"],
                contribution=contribution,
                field_support=field_support,
                active=contribution_active,
            )
            if dr.hint(edge_use_store is not None, mode="scalar"):
                edge_use_store.store(
                    edge_index=sp["batch_states"].edge_index,
                    active=(
                        contribution_active
                        & (sp["batch_states"].prefix_reflection_depth <= wt.Int32(0))
                    ),
                )
            if dr.hint(tape_store is not None, mode="scalar"):
                tape_store.store(
                    strategy_id=BDPTDiffractionTapeStore.DIRECT_ID,
                    order=1,
                    cell_idx=cell["cell_idx"],
                    coord_0=cell["coord_0"],
                    coord_1=cell["coord_1"],
                    scalar_weight=direct_gain_scale * sample_weight * mis_weight,
                    keller_sample=dr.zeros(wt.Float, int(batch_size)),
                    suffix_prim_idx=dr.full(wt.Int32, -1, int(batch_size)),
                    source_pos=sp["batch_states"].source_pos,
                    source_power=sp["batch_states"].source_power,
                    prefix_reflection_depth=sp["batch_states"].prefix_reflection_depth,
                    prefix_initial_ray_dir=sp["batch_states"].prefix_initial_ray_dir,
                    prefix_prim_by_bounce=sp["batch_states"].prefix_prim_by_bounce,
                    edge_indices=(sp["batch_states"].edge_index,),
                    edge_fractions=(sp["edge_fraction"],),
                    active=contribution_active,
                )
            batch_start += batch_stride

        return wt.UInt32(contribution_store.next_slot - start_slot)

    @staticmethod
    @dr.syntax
    def trace_keller_batches(
        *,
        scene: Scene,
        grid,
        states,
        sampler: DiffractionEdgeSampler,
        batch_size: int,
        batch_count: int,
        direct_samples: int,
        keller_samples: int,
        seed: int,
        sample_sequence: str,
        config: ResolvedTraceConfig,
        diff_gain_scale,
        weighted_diagnostics: dict,
        cell_area: float,
        contribution_store: GridContributionStore,
        loop_mode: str,
        tape_store: BDPTDiffractionTapeStore | None = None,
        edge_use_store: BDPTDiffractionEdgeUseStore | None = None,
    ):
        sample_lane = dr.arange(wt.UInt32, int(batch_size))
        batch_start = wt.UInt32(0)
        batch_stride = wt.UInt32(int(batch_size))
        total_samples_u32 = wt.UInt32(int(keller_samples))
        start_slot = wt.UInt32(contribution_store.next_slot)
        n_cells = int(grid.n_cells)
        plane_normal = Sampler.axis_unit_normal(str(grid.axis))

        while dr.hint(
            batch_start < total_samples_u32,
            mode=str(loop_mode),
            max_iterations=max(1, int(batch_count)),
            label="bdpt_keller_diffraction",
            exclude=[scene, grid, states, sampler, config, weighted_diagnostics],
        ):
            sp = BDPTDiffractionMIS.sample_edge_point(
                sample_lane=sample_lane,
                batch_start=batch_start,
                batch_size=batch_size,
                samples=keller_samples,
                states=states,
                sampler=sampler,
                seed=int(seed) + 17,
                sample_sequence=str(sample_sequence),
                state_dimension=0,
                state_stream=601,
                edge_fraction_dimension=1,
                edge_fraction_stream=712,
            )
            orient = BDPTDiffractionMIS.orient_sample(
                batch_states=sp["batch_states"],
                diff_point=sp["diff_point"],
                source_pos=sp["batch_states"].source_pos,
            )
            keller_sample = BDPTDiffractionMIS.sample_uniform(
                sp["sample_index"],
                stream=713,
                dimension=2,
                seed=int(seed),
                sample_sequence=str(sample_sequence),
            )
            ko = UTD.sample_keller_cone(
                orient.edge_dir,
                orient.n0,
                orient.nn,
                keller_sample,
                sp["diff_point"] - sp["batch_states"].source_pos,
                lit_region=True,
            )
            ray_origin = Sampler.spawn_offset_ray_origin(
                sp["diff_point"],
                ko,
                sp["offset_normal"],
            )
            plane_hit = grid_ops.plane_hit(
                ray_origin=ray_origin,
                ray_dir=ko,
                blocker_dist=dr.full(wt.Float, 1.0e10, int(batch_size)),
                grid=grid,
                active=sp["sample_active"],
            )
            source_visible = BDPTDiffractionMIS._segment_pair_visible(
                scene, sp["batch_states"].source_pos, sp["diff_point"], sp["diff_point_offset"]
            )
            safe_diff = dr.select(plane_hit.valid, sp["diff_point"], plane_hit.target_pos)
            safe_diff_offset = dr.select(
                plane_hit.valid, sp["diff_point_offset"], plane_hit.target_pos
            )
            visible_target = plane_hit.valid & BDPTDiffractionMIS._segment_pair_visible(
                scene, plane_hit.target_pos, safe_diff, safe_diff_offset
            )
            ev = UTD.eval_diff_contribution(
                oriented=orient,
                batch_states=sp["batch_states"],
                diff_point=sp["diff_point"],
                ko=ko,
                plane_hit=plane_hit,
                source_visible=source_visible,
                visible_target=visible_target,
                sample_active=sp["sample_active"],
                grid=grid,
                k=config.k,
                wavelength=config.wavelength,
                diff_gain_scale=diff_gain_scale,
                total_length_weight=(
                    sp["edge_measure_weight"] / wt.Float(float(max(1, int(keller_samples))))
                ),
                plane_normal=plane_normal,
            )
            mis_weight = BDPTDiffractionMIS.keller_mis_weight(
                direct_samples=int(direct_samples),
                keller_samples=int(keller_samples),
                total_edge_length=float(sampler.total_length_scalar),
                n_cells=n_cells,
                cell_area=float(cell_area),
                exterior_angle=ev.exterior_angle,
                integration_weight=ev.integration_weight,
            )
            contribution = dr.select(
                ev.contribution_active,
                sp["batch_states"].source_power * ev.contribution * mis_weight,
                wt.Float(0.0),
            )
            BDPTDiffractionMIS._store_diffraction_contribution(
                contribution_store=contribution_store,
                coord_0=plane_hit.coord_0,
                coord_1=plane_hit.coord_1,
                contribution=contribution,
                field_support=ev.field_support,
                active=ev.contribution_active,
            )
            if dr.hint(edge_use_store is not None, mode="scalar"):
                edge_use_store.store(
                    edge_index=sp["batch_states"].edge_index,
                    active=(
                        ev.contribution_active
                        & (sp["batch_states"].prefix_reflection_depth <= wt.Int32(0))
                    ),
                )
            if dr.hint(tape_store is not None, mode="scalar"):
                tape_store.store(
                    strategy_id=BDPTDiffractionTapeStore.KELLER_ID,
                    order=1,
                    cell_idx=ev.cell_idx,
                    coord_0=plane_hit.coord_0,
                    coord_1=plane_hit.coord_1,
                    scalar_weight=(
                        diff_gain_scale
                        * (sp["edge_measure_weight"] / wt.Float(float(max(1, int(keller_samples)))))
                        * mis_weight
                    ),
                    keller_sample=keller_sample,
                    suffix_prim_idx=dr.full(wt.Int32, -1, int(batch_size)),
                    source_pos=sp["batch_states"].source_pos,
                    source_power=sp["batch_states"].source_power,
                    prefix_reflection_depth=sp["batch_states"].prefix_reflection_depth,
                    prefix_initial_ray_dir=sp["batch_states"].prefix_initial_ray_dir,
                    prefix_prim_by_bounce=sp["batch_states"].prefix_prim_by_bounce,
                    edge_indices=(sp["batch_states"].edge_index,),
                    edge_fractions=(sp["edge_fraction"],),
                    active=ev.contribution_active,
                )
            batch_start += batch_stride

        return wt.UInt32(contribution_store.next_slot - start_slot)

    @staticmethod
    def sample_chain_prefix(
        *,
        scene: Scene,
        initial_states,
        recursive_states,
        initial_sampler: DiffractionEdgeSampler,
        recursive_sampler: DiffractionEdgeSampler,
        sample_lane,
        batch_size: int,
        samples: int,
        order: int,
        seed: int,
        stream_base: int,
        sample_sequence: str,
        config: ResolvedTraceConfig,
        batch_idx: int = 0,
        batch_start=None,
    ):
        event_gain_scale = wt.Float((float(config.wavelength) / (4.0 * math.pi)) ** 2)
        sps = []
        sps.append(
            BDPTDiffractionMIS.sample_edge_point(
                sample_lane=sample_lane,
                batch_idx=batch_idx,
                batch_start=batch_start,
                batch_size=batch_size,
                samples=samples,
                states=initial_states,
                sampler=initial_sampler,
                seed=int(seed),
                sample_sequence=str(sample_sequence),
                state_dimension=0,
                state_stream=601,
                edge_fraction_dimension=1,
                edge_fraction_stream=int(stream_base),
            )
        )
        for step in range(1, int(order)):
            sps.append(
                BDPTDiffractionMIS.sample_edge_point(
                    sample_lane=sample_lane,
                    batch_idx=batch_idx,
                    batch_start=batch_start,
                    batch_size=batch_size,
                    samples=samples,
                    states=recursive_states,
                    sampler=recursive_sampler,
                    seed=int(seed) + 31 * step,
                    sample_sequence=str(sample_sequence),
                    state_dimension=2 * int(step),
                    state_stream=601,
                    edge_fraction_dimension=2 * int(step) + 1,
                    edge_fraction_stream=int(stream_base) + step,
                )
            )

        source_pos = sps[0]["batch_states"].source_pos
        throughput = sps[0]["batch_states"].source_power
        active = sps[0]["sample_active"]
        for step in range(0, int(order) - 1):
            current = sps[step]
            nxt = sps[step + 1]
            orient = BDPTDiffractionMIS.orient_sample(
                batch_states=current["batch_states"],
                diff_point=current["diff_point"],
                source_pos=source_pos,
            )
            source_visible = BDPTDiffractionMIS._segment_pair_visible(
                scene, source_pos, current["diff_point"], current["diff_point_offset"]
            )
            next_visible = BDPTDiffractionMIS._segment_pair_visible(
                scene, nxt["diff_point"], current["diff_point"], current["diff_point_offset"]
            )
            field_power, field_valid, _ = BDPTDiffractionMIS._utd_power(
                source_pos=source_pos,
                orient=orient,
                batch_states=current["batch_states"],
                diff_point=current["diff_point"],
                target_pos=nxt["diff_point"],
                config=config,
            )
            distinct_edge = current["batch_states"].edge_index != nxt["batch_states"].edge_index
            segment_active = (
                active
                & current["sample_active"]
                & nxt["sample_active"]
                & distinct_edge
                & source_visible
                & next_visible
                & field_valid
            )
            throughput = dr.select(
                segment_active,
                throughput * field_power * event_gain_scale,
                wt.Float(0.0),
            )
            active = segment_active
            source_pos = current["diff_point"]

        final_sp = sps[-1]
        final_orient = BDPTDiffractionMIS.orient_sample(
            batch_states=final_sp["batch_states"],
            diff_point=final_sp["diff_point"],
            source_pos=source_pos,
        )
        return {
            "active": active,
            "throughput": throughput,
            "source_pos": source_pos,
            "final_sp": final_sp,
            "final_orient": final_orient,
            "sps": tuple(sps),
        }

    @staticmethod
    @dr.syntax
    def trace_chain_direct_batches(
        *,
        scene: Scene,
        grid,
        initial_states,
        recursive_states,
        initial_sampler: DiffractionEdgeSampler,
        recursive_sampler: DiffractionEdgeSampler,
        batch_size: int,
        batch_count: int,
        direct_samples: int,
        keller_samples: int,
        order: int,
        seed: int,
        sample_sequence: str,
        config: ResolvedTraceConfig,
        weighted_diagnostics: dict,
        cell_area: float,
        contribution_store: GridContributionStore,
        loop_mode: str,
        tape_store: BDPTDiffractionTapeStore | None = None,
        edge_use_store: BDPTDiffractionEdgeUseStore | None = None,
    ):
        sample_lane = dr.arange(wt.UInt32, int(batch_size))
        batch_start = wt.UInt32(0)
        batch_stride = wt.UInt32(int(batch_size))
        total_samples_u32 = wt.UInt32(int(direct_samples))
        start_slot = wt.UInt32(contribution_store.next_slot)
        n_cells = int(grid.n_cells)
        length_product = float(initial_sampler.total_length_scalar) * (
            float(recursive_sampler.total_length_scalar) ** max(0, int(order) - 1)
        )
        total_weight = wt.Float(
            length_product * float(max(1, n_cells)) / float(max(1, direct_samples))
        )
        direct_gain_scale = wt.Float((float(config.wavelength) / (4.0 * math.pi)) ** 2)
        plane_normal = Sampler.axis_unit_normal(str(grid.axis))

        while dr.hint(
            batch_start < total_samples_u32,
            mode=str(loop_mode),
            max_iterations=max(1, int(batch_count)),
            label="bdpt_chain_direct_diffraction",
            exclude=[
                scene,
                grid,
                initial_states,
                recursive_states,
                initial_sampler,
                recursive_sampler,
                config,
                weighted_diagnostics,
            ],
        ):
            chain = BDPTDiffractionMIS.sample_chain_prefix(
                scene=scene,
                initial_states=initial_states,
                recursive_states=recursive_states,
                initial_sampler=initial_sampler,
                recursive_sampler=recursive_sampler,
                sample_lane=sample_lane,
                batch_start=batch_start,
                batch_size=batch_size,
                samples=direct_samples,
                order=int(order),
                seed=int(seed) + 101 * int(order),
                stream_base=800 + 16 * int(order),
                sample_sequence=str(sample_sequence),
                config=config,
            )
            sp = chain["final_sp"]
            orient = chain["final_orient"]
            cell = BDPTDiffractionMIS._sample_target_cell(
                sample_index=sp["sample_index"],
                stream=803 + int(order),
                dimension=2 * int(order),
                seed=int(seed),
                sample_sequence=str(sample_sequence),
                grid=grid,
            )
            source_visible = BDPTDiffractionMIS._segment_pair_visible(
                scene, chain["source_pos"], sp["diff_point"], sp["diff_point_offset"]
            )
            target_visible = BDPTDiffractionMIS._segment_pair_visible(
                scene, cell["target_pos"], sp["diff_point"], sp["diff_point_offset"]
            )
            field_power, field_valid, field_support = BDPTDiffractionMIS._utd_power(
                source_pos=chain["source_pos"],
                orient=orient,
                batch_states=sp["batch_states"],
                diff_point=sp["diff_point"],
                target_pos=cell["target_pos"],
                config=config,
            )
            target_dir = cell["target_pos"] - sp["diff_point"]
            target_dir = target_dir / (dr.norm(target_dir) + wt.Float(1.0e-6))
            exterior_angle = BDPTDiffractionMIS.exterior_angle(orient.n0, orient.nn)
            integration_weight = UTD.integration_weight(
                edge_origin=sp["batch_states"].edge_pos,
                edge_dir=orient.edge_dir,
                n0=orient.n0,
                source_pos=chain["source_pos"],
                diff_point=sp["diff_point"],
                k_world=target_dir,
                target_pos=cell["target_pos"],
                plane_normal=plane_normal,
            )
            mis_weight = BDPTDiffractionMIS.direct_mis_weight(
                direct_samples=int(direct_samples),
                keller_samples=int(keller_samples),
                total_edge_length=length_product,
                n_cells=n_cells,
                cell_area=float(cell_area),
                exterior_angle=exterior_angle,
                integration_weight=integration_weight,
            )
            contribution_active = (
                chain["active"] & source_visible & target_visible & field_valid
            )
            contribution = dr.select(
                contribution_active,
                (
                    chain["throughput"]
                    * field_power
                    * direct_gain_scale
                    * total_weight
                    * mis_weight
                ),
                wt.Float(0.0),
            )
            BDPTDiffractionMIS._store_diffraction_contribution(
                contribution_store=contribution_store,
                coord_0=cell["coord_0"],
                coord_1=cell["coord_1"],
                contribution=contribution,
                field_support=field_support,
                active=contribution_active,
            )
            first_sp = chain["sps"][0]
            if dr.hint(edge_use_store is not None, mode="scalar"):
                edge_use_store.store(
                    edge_index=first_sp["batch_states"].edge_index,
                    active=(
                        contribution_active
                        & (first_sp["batch_states"].prefix_reflection_depth <= wt.Int32(0))
                    ),
                )
            if dr.hint(tape_store is not None, mode="scalar"):
                tape_store.store(
                    strategy_id=BDPTDiffractionTapeStore.DIRECT_ID,
                    order=int(order),
                    cell_idx=cell["cell_idx"],
                    coord_0=cell["coord_0"],
                    coord_1=cell["coord_1"],
                    scalar_weight=direct_gain_scale * total_weight * mis_weight,
                    keller_sample=dr.zeros(wt.Float, int(batch_size)),
                    suffix_prim_idx=dr.full(wt.Int32, -1, int(batch_size)),
                    source_pos=first_sp["batch_states"].source_pos,
                    source_power=first_sp["batch_states"].source_power,
                    prefix_reflection_depth=first_sp["batch_states"].prefix_reflection_depth,
                    prefix_initial_ray_dir=first_sp["batch_states"].prefix_initial_ray_dir,
                    prefix_prim_by_bounce=first_sp["batch_states"].prefix_prim_by_bounce,
                    edge_indices=tuple(
                        step_sp["batch_states"].edge_index
                        for step_sp in chain["sps"]
                    ),
                    edge_fractions=tuple(
                        step_sp["edge_fraction"]
                        for step_sp in chain["sps"]
                    ),
                    active=contribution_active,
                )
            batch_start += batch_stride

        return wt.UInt32(contribution_store.next_slot - start_slot)

    @staticmethod
    @dr.syntax
    def trace_chain_keller_batches(
        *,
        scene: Scene,
        grid,
        initial_states,
        recursive_states,
        initial_sampler: DiffractionEdgeSampler,
        recursive_sampler: DiffractionEdgeSampler,
        batch_size: int,
        batch_count: int,
        direct_samples: int,
        keller_samples: int,
        order: int,
        seed: int,
        sample_sequence: str,
        config: ResolvedTraceConfig,
        diff_gain_scale,
        weighted_diagnostics: dict,
        cell_area: float,
        contribution_store: GridContributionStore,
        loop_mode: str,
        tape_store: BDPTDiffractionTapeStore | None = None,
        edge_use_store: BDPTDiffractionEdgeUseStore | None = None,
    ):
        sample_lane = dr.arange(wt.UInt32, int(batch_size))
        batch_start = wt.UInt32(0)
        batch_stride = wt.UInt32(int(batch_size))
        total_samples_u32 = wt.UInt32(int(keller_samples))
        start_slot = wt.UInt32(contribution_store.next_slot)
        n_cells = int(grid.n_cells)
        plane_normal = Sampler.axis_unit_normal(str(grid.axis))
        length_product = float(initial_sampler.total_length_scalar) * (
            float(recursive_sampler.total_length_scalar) ** max(0, int(order) - 1)
        )
        total_length_weight = wt.Float(length_product / float(max(1, int(keller_samples))))

        while dr.hint(
            batch_start < total_samples_u32,
            mode=str(loop_mode),
            max_iterations=max(1, int(batch_count)),
            label="bdpt_chain_keller_diffraction",
            exclude=[
                scene,
                grid,
                initial_states,
                recursive_states,
                initial_sampler,
                recursive_sampler,
                config,
                weighted_diagnostics,
            ],
        ):
            chain = BDPTDiffractionMIS.sample_chain_prefix(
                scene=scene,
                initial_states=initial_states,
                recursive_states=recursive_states,
                initial_sampler=initial_sampler,
                recursive_sampler=recursive_sampler,
                sample_lane=sample_lane,
                batch_start=batch_start,
                batch_size=batch_size,
                samples=keller_samples,
                order=int(order),
                seed=int(seed) + 139 * int(order),
                stream_base=900 + 16 * int(order),
                sample_sequence=str(sample_sequence),
                config=config,
            )
            sp = chain["final_sp"]
            orient = chain["final_orient"]
            keller_sample = BDPTDiffractionMIS.sample_uniform(
                sp["sample_index"],
                stream=903 + int(order),
                dimension=2 * int(order),
                seed=int(seed),
                sample_sequence=str(sample_sequence),
            )
            ko = UTD.sample_keller_cone(
                orient.edge_dir,
                orient.n0,
                orient.nn,
                keller_sample,
                sp["diff_point"] - chain["source_pos"],
                lit_region=True,
            )
            ray_origin = Sampler.spawn_offset_ray_origin(
                sp["diff_point"],
                ko,
                sp["offset_normal"],
            )
            plane_hit = grid_ops.plane_hit(
                ray_origin=ray_origin,
                ray_dir=ko,
                blocker_dist=dr.full(wt.Float, 1.0e10, int(batch_size)),
                grid=grid,
                active=chain["active"],
            )
            source_visible = BDPTDiffractionMIS._segment_pair_visible(
                scene, chain["source_pos"], sp["diff_point"], sp["diff_point_offset"]
            )
            safe_diff = dr.select(plane_hit.valid, sp["diff_point"], plane_hit.target_pos)
            safe_diff_offset = dr.select(
                plane_hit.valid, sp["diff_point_offset"], plane_hit.target_pos
            )
            visible_target = plane_hit.valid & BDPTDiffractionMIS._segment_pair_visible(
                scene, plane_hit.target_pos, safe_diff, safe_diff_offset
            )
            field_power, field_valid, field_support = BDPTDiffractionMIS._utd_power(
                source_pos=chain["source_pos"],
                orient=orient,
                batch_states=sp["batch_states"],
                diff_point=sp["diff_point"],
                target_pos=plane_hit.target_pos,
                config=config,
            )
            exterior_angle = BDPTDiffractionMIS.exterior_angle(orient.n0, orient.nn)
            iw = UTD.integration_weight(
                edge_origin=sp["batch_states"].edge_pos,
                edge_dir=orient.edge_dir,
                n0=orient.n0,
                source_pos=chain["source_pos"],
                diff_point=sp["diff_point"],
                k_world=ko,
                target_pos=plane_hit.target_pos,
                plane_normal=plane_normal,
            )
            mis_weight = BDPTDiffractionMIS.keller_mis_weight(
                direct_samples=int(direct_samples),
                keller_samples=int(keller_samples),
                total_edge_length=length_product,
                n_cells=n_cells,
                cell_area=float(cell_area),
                exterior_angle=exterior_angle,
                integration_weight=iw,
            )
            contribution_active = (
                chain["active"]
                & source_visible
                & visible_target
                & field_valid
            )
            contribution = dr.select(
                contribution_active,
                chain["throughput"]
                * field_power
                * diff_gain_scale
                * iw
                * total_length_weight
                * exterior_angle
                * mis_weight,
                wt.Float(0.0),
            )
            BDPTDiffractionMIS._store_diffraction_contribution(
                contribution_store=contribution_store,
                coord_0=plane_hit.coord_0,
                coord_1=plane_hit.coord_1,
                contribution=contribution,
                field_support=field_support,
                active=contribution_active,
            )
            first_sp = chain["sps"][0]
            if dr.hint(edge_use_store is not None, mode="scalar"):
                edge_use_store.store(
                    edge_index=first_sp["batch_states"].edge_index,
                    active=(
                        contribution_active
                        & (first_sp["batch_states"].prefix_reflection_depth <= wt.Int32(0))
                    ),
                )
            if dr.hint(tape_store is not None, mode="scalar"):
                tape_store.store(
                    strategy_id=BDPTDiffractionTapeStore.KELLER_ID,
                    order=int(order),
                    cell_idx=grid_ops.cell_index(
                        grid=grid,
                        coord_0=plane_hit.coord_0,
                        coord_1=plane_hit.coord_1,
                    ),
                    coord_0=plane_hit.coord_0,
                    coord_1=plane_hit.coord_1,
                    scalar_weight=diff_gain_scale * total_length_weight * mis_weight,
                    keller_sample=keller_sample,
                    suffix_prim_idx=dr.full(wt.Int32, -1, int(batch_size)),
                    source_pos=first_sp["batch_states"].source_pos,
                    source_power=first_sp["batch_states"].source_power,
                    prefix_reflection_depth=first_sp["batch_states"].prefix_reflection_depth,
                    prefix_initial_ray_dir=first_sp["batch_states"].prefix_initial_ray_dir,
                    prefix_prim_by_bounce=first_sp["batch_states"].prefix_prim_by_bounce,
                    edge_indices=tuple(
                        step_sp["batch_states"].edge_index
                        for step_sp in chain["sps"]
                    ),
                    edge_fractions=tuple(
                        step_sp["edge_fraction"]
                        for step_sp in chain["sps"]
                    ),
                    active=contribution_active,
                )
            batch_start += batch_stride

        return wt.UInt32(contribution_store.next_slot - start_slot)

    @staticmethod
    @dr.syntax
    def trace_suffix_reflection_batches(
        *,
        scene: Scene,
        grid,
        initial_states,
        recursive_states,
        initial_sampler: DiffractionEdgeSampler,
        recursive_sampler: DiffractionEdgeSampler,
        batch_size: int,
        batch_count: int,
        suffix_samples: int,
        order: int,
        seed: int,
        sample_sequence: str,
        config: ResolvedTraceConfig,
        weighted_diagnostics: dict,
        contribution_store: GridContributionStore,
        loop_mode: str,
        tape_store: BDPTDiffractionTapeStore | None = None,
        edge_use_store: BDPTDiffractionEdgeUseStore | None = None,
    ):
        tri_data = scene._triangle_runtime()
        n_triangles = 0 if tri_data is None else int(tri_data.get("n_triangles", 0))
        if dr.hint(n_triangles <= 0, mode="scalar"):
            return wt.UInt32(0)

        sample_lane = dr.arange(wt.UInt32, int(batch_size))
        batch_start = wt.UInt32(0)
        batch_stride = wt.UInt32(int(batch_size))
        total_samples_u32 = wt.UInt32(int(suffix_samples))
        start_slot = wt.UInt32(contribution_store.next_slot)
        n_cells = int(grid.n_cells)
        length_product = float(initial_sampler.total_length_scalar) * (
            float(recursive_sampler.total_length_scalar) ** max(0, int(order) - 1)
        )
        total_weight = wt.Float(
            length_product
            * float(max(1, n_cells))
            * float(max(1, n_triangles))
            / float(max(1, suffix_samples))
        )
        material_omega = wave_math.material_angular_frequency(config.wavelength)
        direct_gain_scale = wt.Float((float(config.wavelength) / (4.0 * math.pi)) ** 2)

        while dr.hint(
            batch_start < total_samples_u32,
            mode=str(loop_mode),
            max_iterations=max(1, int(batch_count)),
            label="bdpt_suffix_reflection_diffraction",
            exclude=[
                scene,
                grid,
                initial_states,
                recursive_states,
                initial_sampler,
                recursive_sampler,
                config,
                tri_data,
                weighted_diagnostics,
            ],
        ):
            chain = BDPTDiffractionMIS.sample_chain_prefix(
                scene=scene,
                initial_states=initial_states,
                recursive_states=recursive_states,
                initial_sampler=initial_sampler,
                recursive_sampler=recursive_sampler,
                sample_lane=sample_lane,
                batch_start=batch_start,
                batch_size=batch_size,
                samples=suffix_samples,
                order=int(order),
                seed=int(seed) + 173 * int(order),
                stream_base=1000 + 16 * int(order),
                sample_sequence=str(sample_sequence),
                config=config,
            )
            sp = chain["final_sp"]
            orient = chain["final_orient"]
            cell = BDPTDiffractionMIS._sample_target_cell(
                sample_index=sp["sample_index"],
                stream=1003 + int(order),
                dimension=2 * int(order),
                seed=int(seed),
                sample_sequence=str(sample_sequence),
                grid=grid,
            )
            prim_u = BDPTDiffractionMIS.sample_uniform(
                sp["sample_index"],
                stream=1004 + int(order),
                dimension=2 * int(order) + 1,
                seed=int(seed),
                sample_sequence=str(sample_sequence),
            )
            prim_idx = wt.Int32(
                dr.minimum(
                    wt.Float(max(0, n_triangles - 1)),
                    dr.floor(prim_u * wt.Float(max(1, n_triangles))),
                )
            )
            safe_prim_idx = wt.UInt32(dr.maximum(prim_idx, wt.Int32(0)))
            v0 = dr.gather(wt.Point3f, tri_data["v0"], safe_prim_idx)
            v1 = dr.gather(wt.Point3f, tri_data["v1"], safe_prim_idx)
            v2 = dr.gather(wt.Point3f, tri_data["v2"], safe_prim_idx)
            geom_n = dr.cross(v1 - v0, v2 - v0)
            geom_n = geom_n / (dr.norm(geom_n) + wt.Float(1.0e-6))
            target_pos = cell["target_pos"]
            image_source = geometry.reflect_point_across_plane(sp["diff_point"], v0, geom_n)
            reflected_valid, reflection_point, reflection_normal, resolved_prim_idx = (
                scene.triangle_surface_intersection(image_source, target_pos, prim_idx)
            )
            source_visible = BDPTDiffractionMIS._segment_pair_visible(
                scene, chain["source_pos"], sp["diff_point"], sp["diff_point_offset"]
            )
            edge_to_reflection_visible = scene.segment_visible(
                sp["diff_point"],
                reflection_point,
                ignore_prim_idx=resolved_prim_idx,
                max_ignored_hits=2,
            )
            reflection_to_target_visible = scene.segment_visible(
                reflection_point,
                target_pos,
                ignore_prim_idx=resolved_prim_idx,
                max_ignored_hits=2,
            )
            field_power, field_valid, field_support = BDPTDiffractionMIS._utd_power(
                source_pos=chain["source_pos"],
                orient=orient,
                batch_states=sp["batch_states"],
                diff_point=sp["diff_point"],
                target_pos=reflection_point,
                config=config,
            )
            incoming = reflection_point - sp["diff_point"]
            outgoing = target_pos - reflection_point
            incoming_hat = incoming / (dr.norm(incoming) + wt.Float(1.0e-6))
            outgoing_dist = dr.norm(outgoing)
            oriented_normal = dr.select(
                dr.dot(incoming_hat, reflection_normal) > wt.Float(0.0),
                -reflection_normal,
                reflection_normal,
            )
            reflected_hat = incoming_hat - wt.Float(2.0) * dr.dot(incoming_hat, oriented_normal) * oriented_normal
            outgoing_hat = outgoing / (outgoing_dist + wt.Float(1.0e-6))
            specular_match = dr.dot(reflected_hat, outgoing_hat) > wt.Float(1.0 - 1.0e-3)
            material_inputs = resolve_surface_material(
                scene=scene,
                prim_idx=resolved_prim_idx,
                default_gain=1.0,
                valid_mask=reflected_valid,
            )
            cos_theta = dr.clip(
                dr.abs(dr.dot(-incoming_hat, oriented_normal)),
                wt.Float(1.0e-6),
                wt.Float(1.0),
            )
            eta = wave_math.complex_relative_permittivity(
                material_inputs["eta_r"],
                material_inputs["sigma"],
                material_omega,
            )
            r_te, r_tm = wave_math.fresnel_reflection(cos_theta, eta, mu_r=material_inputs["mu_r"])
            fresnel_power = wt.Float(0.5) * (
                arrays.complex_abs_sqr(r_te) + arrays.complex_abs_sqr(r_tm)
            )
            reflection_power = dr.square(material_inputs["gain"]) * fresnel_power
            suffix_fspl = dr.square(
                wt.Float(config.wavelength / (4.0 * math.pi))
                / dr.maximum(outgoing_dist, wt.Float(1.0e-6))
            )
            contribution_active = (
                chain["active"]
                & reflected_valid
                & specular_match
                & source_visible
                & edge_to_reflection_visible
                & reflection_to_target_visible
                & field_valid
            )
            contribution = dr.select(
                contribution_active,
                chain["throughput"]
                * field_power
                * direct_gain_scale
                * reflection_power
                * suffix_fspl
                * total_weight,
                wt.Float(0.0),
            )
            BDPTDiffractionMIS._store_diffraction_contribution(
                contribution_store=contribution_store,
                coord_0=cell["coord_0"],
                coord_1=cell["coord_1"],
                contribution=contribution,
                field_support=field_support,
                active=contribution_active,
            )
            first_sp = chain["sps"][0]
            if dr.hint(edge_use_store is not None, mode="scalar"):
                edge_use_store.store(
                    edge_index=first_sp["batch_states"].edge_index,
                    active=(
                        contribution_active
                        & (first_sp["batch_states"].prefix_reflection_depth <= wt.Int32(0))
                    ),
                )
            if dr.hint(tape_store is not None, mode="scalar"):
                tape_store.store(
                    strategy_id=BDPTDiffractionTapeStore.SUFFIX_REFLECTION_ID,
                    order=int(order),
                    cell_idx=cell["cell_idx"],
                    coord_0=cell["coord_0"],
                    coord_1=cell["coord_1"],
                    scalar_weight=direct_gain_scale * total_weight,
                    keller_sample=dr.zeros(wt.Float, int(batch_size)),
                    suffix_prim_idx=prim_idx,
                    source_pos=first_sp["batch_states"].source_pos,
                    source_power=first_sp["batch_states"].source_power,
                    prefix_reflection_depth=first_sp["batch_states"].prefix_reflection_depth,
                    prefix_initial_ray_dir=first_sp["batch_states"].prefix_initial_ray_dir,
                    prefix_prim_by_bounce=first_sp["batch_states"].prefix_prim_by_bounce,
                    edge_indices=tuple(
                        step_sp["batch_states"].edge_index
                        for step_sp in chain["sps"]
                    ),
                    edge_fractions=tuple(
                        step_sp["edge_fraction"]
                        for step_sp in chain["sps"]
                    ),
                    active=contribution_active,
                )
            batch_start += batch_stride

        return wt.UInt32(contribution_store.next_slot - start_slot)

    @staticmethod
    def trace(
        *,
        scene: Scene,
        grid,
        tx_pos,
        config: ResolvedTraceConfig,
        samples_per_tx: int,
        seed: int,
        diff_gain_scale,
        cell_area: float,
        weighted_diagnostics: dict,
        loop_mode: str,
        max_depth: int = 1,
        sample_sequence: str = SOBOL_SEQUENCE,
        prefix_store=None,
        collect_ad_tapes: bool = False,
    ) -> BDPTDiffractionResult:
        depth = max(1, min(int(max_depth), BDPTDiffractionMIS.MAX_SUPPORTED_DIFFRACTION_DEPTH))
        include_reflection_coupled = bool(config.enable_bdpt_reflection_coupled_diffraction)
        strategy_samples = BDPTDiffractionMIS.allocate_samples(
            samples_per_tx,
            depth,
            include_suffix_reflection=include_reflection_coupled,
        )
        zero_counts = BDPTDiffractionMIS._zero_strategy_counts()
        order_counts = {
            order: BDPTDiffractionMIS._zero_strategy_counts()
            for order in range(1, depth + 1)
        }
        state_sets = BDPTDiffractionMIS.prepare_states(
            tx_pos=tx_pos,
            scene=scene,
            config=config,
            prefix_store=prefix_store if include_reflection_coupled else None,
        )
        initial_states = state_sets["initial"]
        recursive_states = state_sets["recursive"]
        prefix_state_count = int(state_sets["prefix_state_count"])
        if initial_states is None or recursive_states is None:
            return BDPTDiffractionResult.zero(
                order_counts=order_counts,
                order_samples=strategy_samples,
            )
        initial_line_length = dr.maximum(
            initial_states.edge_line_max - initial_states.edge_line_min,
            wt.Float(0.0),
        )
        recursive_line_length = dr.maximum(
            recursive_states.edge_line_max - recursive_states.edge_line_min,
            wt.Float(0.0),
        )
        initial_sampler = DiffractionEdgeSampler.from_line_length(initial_line_length)
        recursive_sampler = DiffractionEdgeSampler.from_line_length(recursive_line_length)
        if initial_sampler is None or recursive_sampler is None:
            return BDPTDiffractionResult.zero(
                state_count=int(dr.width(initial_states.edge_index)),
                prefix_state_count=prefix_state_count,
                order_counts=order_counts,
                order_samples=strategy_samples,
            )
        first_order_sampler = BDPTDiffractionMIS.first_order_importance_sampler(
            states=initial_states,
            line_length=initial_line_length,
            grid=grid,
            fallback_sampler=initial_sampler,
        )
        edge_runtime = scene._selected_edge_runtime()
        edge_use_store = BDPTDiffractionEdgeUseStore(
            n_edges=0 if edge_runtime is None else int(edge_runtime.get("n_edges", 0))
        )
        total_direct_count = wt.UInt32(0)
        total_keller_count = wt.UInt32(0)
        total_suffix_count = wt.UInt32(0)
        order_count_values = {}
        contribution_store = GridContributionStore(
            capacity=max(0, int(samples_per_tx)),
            grid=grid,
            weighted_diagnostics=weighted_diagnostics,
        )
        tape_store = (
            BDPTDiffractionTapeStore(
                capacity=max(0, int(samples_per_tx)),
                max_prefix_bounces=int(config.reflection_max_bounces),
            )
            if collect_ad_tapes
            else None
        )
        for order in range(1, depth + 1):
            order_samples = strategy_samples[order]
            direct_samples = int(order_samples[BDPTDiffractionMIS.DIRECT_STRATEGY])
            keller_samples = int(order_samples[BDPTDiffractionMIS.KELLER_STRATEGY])
            suffix_samples = int(order_samples[BDPTDiffractionMIS.SUFFIX_REFLECTION_STRATEGY])
            direct_count = wt.UInt32(0)
            keller_count = wt.UInt32(0)
            suffix_count = wt.UInt32(0)
            if direct_samples > 0:
                direct_batch_size = min(max(1, direct_samples), 65536)
                direct_batch_count = int(math.ceil(direct_samples / direct_batch_size))
                if order == 1:
                    direct_count = BDPTDiffractionMIS.trace_direct_batches(
                        scene=scene,
                        grid=grid,
                        states=initial_states,
                        sampler=first_order_sampler,
                        batch_size=direct_batch_size,
                        batch_count=direct_batch_count,
                        direct_samples=direct_samples,
                        keller_samples=keller_samples,
                        seed=int(seed),
                        sample_sequence=str(sample_sequence),
                        config=config,
                        weighted_diagnostics=weighted_diagnostics,
                        cell_area=float(cell_area),
                        contribution_store=contribution_store,
                        loop_mode=str(loop_mode),
                        tape_store=tape_store,
                        edge_use_store=edge_use_store,
                    )
                else:
                    direct_count = BDPTDiffractionMIS.trace_chain_direct_batches(
                        scene=scene,
                        grid=grid,
                        initial_states=initial_states,
                        recursive_states=recursive_states,
                        initial_sampler=initial_sampler,
                        recursive_sampler=recursive_sampler,
                        batch_size=direct_batch_size,
                        batch_count=direct_batch_count,
                        direct_samples=direct_samples,
                        keller_samples=keller_samples,
                        order=order,
                        seed=int(seed),
                        sample_sequence=str(sample_sequence),
                        config=config,
                        weighted_diagnostics=weighted_diagnostics,
                        cell_area=float(cell_area),
                        contribution_store=contribution_store,
                        loop_mode=str(loop_mode),
                        tape_store=tape_store,
                        edge_use_store=edge_use_store,
                    )
            if keller_samples > 0:
                keller_batch_size = min(max(1, keller_samples), 65536)
                keller_batch_count = int(math.ceil(keller_samples / keller_batch_size))
                if order == 1:
                    keller_count = BDPTDiffractionMIS.trace_keller_batches(
                        scene=scene,
                        grid=grid,
                        states=initial_states,
                        sampler=first_order_sampler,
                        batch_size=keller_batch_size,
                        batch_count=keller_batch_count,
                        direct_samples=direct_samples,
                        keller_samples=keller_samples,
                        seed=int(seed),
                        sample_sequence=str(sample_sequence),
                        config=config,
                        diff_gain_scale=diff_gain_scale,
                        weighted_diagnostics=weighted_diagnostics,
                        cell_area=float(cell_area),
                        contribution_store=contribution_store,
                        loop_mode=str(loop_mode),
                        tape_store=tape_store,
                        edge_use_store=edge_use_store,
                    )
                else:
                    keller_count = BDPTDiffractionMIS.trace_chain_keller_batches(
                        scene=scene,
                        grid=grid,
                        initial_states=initial_states,
                        recursive_states=recursive_states,
                        initial_sampler=initial_sampler,
                        recursive_sampler=recursive_sampler,
                        batch_size=keller_batch_size,
                        batch_count=keller_batch_count,
                        direct_samples=direct_samples,
                        keller_samples=keller_samples,
                        order=order,
                        seed=int(seed),
                        sample_sequence=str(sample_sequence),
                        config=config,
                        diff_gain_scale=diff_gain_scale,
                        weighted_diagnostics=weighted_diagnostics,
                        cell_area=float(cell_area),
                        contribution_store=contribution_store,
                        loop_mode=str(loop_mode),
                        tape_store=tape_store,
                        edge_use_store=edge_use_store,
                    )
            if include_reflection_coupled and suffix_samples > 0:
                suffix_batch_size = min(max(1, suffix_samples), 65536)
                suffix_count = BDPTDiffractionMIS.trace_suffix_reflection_batches(
                    scene=scene,
                    grid=grid,
                    initial_states=initial_states,
                    recursive_states=recursive_states,
                    initial_sampler=initial_sampler,
                    recursive_sampler=recursive_sampler,
                    batch_size=suffix_batch_size,
                    batch_count=int(math.ceil(suffix_samples / suffix_batch_size)),
                    suffix_samples=suffix_samples,
                    order=order,
                    seed=int(seed),
                    sample_sequence=str(sample_sequence),
                    config=config,
                    weighted_diagnostics=weighted_diagnostics,
                    contribution_store=contribution_store,
                    loop_mode=str(loop_mode),
                    tape_store=tape_store,
                    edge_use_store=edge_use_store,
                )
            order_count_values[order] = {
                BDPTDiffractionMIS.DIRECT_STRATEGY: direct_count,
                BDPTDiffractionMIS.KELLER_STRATEGY: keller_count,
                BDPTDiffractionMIS.SUFFIX_REFLECTION_STRATEGY: suffix_count,
            }
            total_direct_count += direct_count
            total_keller_count += keller_count
            total_suffix_count += suffix_count

        contribution_store.scatter_into(
            grid=grid,
            weighted_diagnostics=weighted_diagnostics,
        )
        count = total_direct_count + total_keller_count + total_suffix_count
        count_values = [
            order_count_values[order][strategy]
            for order in sorted(order_count_values)
            for strategy in zero_counts
        ] + [total_direct_count, total_keller_count, total_suffix_count, count]
        if count_values:
            dr.eval(*count_values)
        order_counts = {
            order: {
                BDPTDiffractionMIS.DIRECT_STRATEGY: int(
                    scalar(values[BDPTDiffractionMIS.DIRECT_STRATEGY])
                ),
                BDPTDiffractionMIS.KELLER_STRATEGY: int(
                    scalar(values[BDPTDiffractionMIS.KELLER_STRATEGY])
                ),
                BDPTDiffractionMIS.SUFFIX_REFLECTION_STRATEGY: int(
                    scalar(values[BDPTDiffractionMIS.SUFFIX_REFLECTION_STRATEGY])
                ),
            }
            for order, values in order_count_values.items()
        }
        flat_samples = {
            strategy: sum(int(samples[strategy]) for samples in strategy_samples.values())
            for strategy in zero_counts
        }
        edge_indices = edge_use_store.finalize()
        return BDPTDiffractionResult(
            path_count=count,
            state_count=int(dr.width(initial_states.edge_index)),
            prefix_state_count=prefix_state_count,
            edge_indices=edge_indices,
            total_edge_length=float(initial_sampler.total_length_scalar),
            strategy_counts={
                BDPTDiffractionMIS.DIRECT_STRATEGY: int(scalar(total_direct_count)),
                BDPTDiffractionMIS.KELLER_STRATEGY: int(scalar(total_keller_count)),
                BDPTDiffractionMIS.SUFFIX_REFLECTION_STRATEGY: int(scalar(total_suffix_count)),
            },
            strategy_samples=flat_samples,
            order_counts=order_counts,
            order_samples=strategy_samples,
            tape=None if tape_store is None else tape_store.finalize(),
        )


__all__ = ["BDPTDiffractionMIS", "BDPTDiffractionResult", "BDPTDiffractionTape"]
