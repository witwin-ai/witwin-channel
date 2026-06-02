from __future__ import annotations

from collections.abc import Iterator, Mapping
import math
import time
from typing import Callable

import drjit as dr
import witwin as wt

from ...orchestration import ResolvedTraceConfig
from ....scene import Scene
from ....trace.diffraction import geometry as diff_geometry
from ....utils import scalar
from ....utils.constants import EPS
from ....utils import drjit_ops as dj_ops  # Broadcast, EvalSync
from . import common as mc_common
from . import field as diff_field

def _sample_keller_cone(edge_dir, n0, nn, sample, ki, *, lit_region: bool):
    edge_hat = edge_dir / (dr.norm(edge_dir) + wt.Float(EPS))
    n0_hat = n0 / (dr.norm(n0) + wt.Float(EPS))
    nn_hat = nn / (dr.norm(nn) + wt.Float(EPS))
    ki_hat = ki / (dr.norm(ki) + wt.Float(EPS))
    t0 = dr.normalize(dr.cross(n0_hat, edge_hat))
    e_fwd = dr.select(dr.dot(edge_hat, ki_hat) > 0.0, edge_hat, -edge_hat)
    ki_local = wt.Vector3f(
        dr.dot(ki_hat, t0),
        dr.dot(ki_hat, n0_hat),
        dr.dot(ki_hat, e_fwd),
    )
    sin_beta0 = dr.sqrt(
        dr.maximum(wt.Float(0.0), wt.Float(1.0) - ki_local.z * ki_local.z)
    )
    beta0 = dr.atan2(sin_beta0, ki_local.z)
    phi_i = dr.atan2(ki_local.y, ki_local.x)
    phi_i = dr.select(phi_i < 0.0, phi_i + wt.Float(2.0 * math.pi), phi_i)
    wedge_interior = dr.safe_acos(
        dr.clip(-dr.dot(n0_hat, nn_hat), wt.Float(-1.0), wt.Float(1.0))
    )
    exterior_angle = wt.Float(2.0 * math.pi) - wedge_interior
    phi = (
        sample * exterior_angle
        if lit_region
        else phi_i + sample * dr.maximum(exterior_angle - phi_i, wt.Float(0.0))
    )
    sin_beta, cos_beta = dr.sincos(beta0)
    sin_phi, cos_phi = dr.sincos(phi)
    return (
        sin_phi * sin_beta * n0_hat
        + cos_phi * sin_beta * t0
        + cos_beta * e_fwd
    )


def _diffraction_integration_weight(
    *,
    edge_origin,
    edge_dir,
    n0,
    source_pos,
    diff_point,
    k_world,
    target_pos,
    plane_normal,
):
    width = int(dr.width(target_pos.x))
    edge_origin_b = dj_ops.broadcast_point(edge_origin, width)
    edge_hat = dj_ops.broadcast_vector(edge_dir / (dr.norm(edge_dir) + wt.Float(EPS)), width)
    n0_b = dj_ops.broadcast_vector(n0 / (dr.norm(n0) + wt.Float(EPS)), width)
    plane_normal_b = dj_ops.broadcast_vector(plane_normal, width)
    source_pos_b = dj_ops.broadcast_point(source_pos, width)
    incident_dir = diff_point - source_pos_b
    e_fwd = dr.select(dr.dot(edge_hat, incident_dir) > 0.0, edge_hat, -edge_hat)
    t0 = dr.normalize(dr.cross(n0_b, edge_hat))
    k_local_x = dr.dot(k_world, t0)
    k_local_y = dr.dot(k_world, n0_b)
    phi = dr.atan2(k_local_y, k_local_x)
    ell = dr.dot(diff_point - edge_origin_b, edge_hat)
    v = source_pos_b - edge_origin_b
    w = dr.dot(v, edge_hat)
    source_proj = edge_origin_b + w * edge_hat
    perp_offset = source_pos_b - source_proj
    perp_norm = dr.norm(perp_offset)
    u = ell - w
    radial_distance = dr.norm(ell * edge_hat - v)
    nrm = radial_distance + wt.Float(EPS)
    sin_phi, cos_phi = dr.sincos(phi)
    tangential_dir = cos_phi * t0 + sin_phi * n0_b
    angular_dir = -sin_phi * t0 + cos_phi * n0_b
    d_world = (perp_norm / nrm) * tangential_dir + (u / nrm) * e_fwd
    dd_dphi = (perp_norm / nrm) * angular_dir
    safe_radial_distance = dr.maximum(radial_distance, wt.Float(EPS))
    dd_dell = e_fwd / nrm - d_world * (u / safe_radial_distance) / nrm
    numerator = dr.dot(
        plane_normal_b,
        target_pos - edge_origin_b - ell * edge_hat,
    )
    denominator = dr.dot(plane_normal_b, d_world)
    safe_denominator = denominator + wt.Float(EPS)
    numerator_dell = -dr.dot(plane_normal_b, edge_hat)
    denominator_dell = dr.dot(plane_normal_b, dd_dell)
    denominator_dphi = dr.dot(plane_normal_b, dd_dphi)
    travel = numerator / safe_denominator
    dtravel_dell = (
        numerator_dell * safe_denominator - numerator * denominator_dell
    ) / (safe_denominator * safe_denominator)
    dtravel_dphi = (
        -numerator * denominator_dphi
    ) / (safe_denominator * safe_denominator)
    ds_dell = edge_hat + dtravel_dell * d_world + travel * dd_dell
    ds_dphi = dtravel_dphi * d_world + travel * dd_dphi
    return dr.norm(dr.cross(ds_dphi, ds_dell))


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

    @classmethod
    def for_samples(cls, samples_per_tx: int) -> DiffractionTapeStore:
        return cls(capacity=max(0, int(samples_per_tx)))

    def store(
        self,
        *,
        edge_index,
        edge_fraction,
        cone_sample,
        cell_idx,
        field_valid,
        pole_safe,
        dif_n_p,
        dif_n_m,
        sum_n_p,
        sum_n_m,
        active,
    ) -> None:
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

    def finalize(self):
        zero_float = dr.zeros(wt.Float, 0)
        zero_int = dr.zeros(wt.Int32, 0)
        zero_uint = dr.zeros(wt.UInt32, 0)
        count = int(scalar(self.next_slot))
        indices = dr.arange(wt.UInt32, count)
        return {
            "edge_index": (
                dr.gather(wt.Int32, self.edge_index, indices)
                if count > 0
                else zero_int
            ),
            "edge_fraction": (
                dr.gather(wt.Float, self.edge_fraction, indices)
                if count > 0
                else zero_float
            ),
            "cone_sample": (
                dr.gather(wt.Float, self.cone_sample, indices)
                if count > 0
                else zero_float
            ),
            "cell_idx": (
                dr.gather(wt.UInt32, self.cell_idx, indices)
                if count > 0
                else zero_uint
            ),
            "field_valid": (
                dr.gather(wt.Bool, self.field_valid, indices)
                if count > 0
                else dr.zeros(wt.Bool, 0)
            ),
            "pole_safe": (
                dr.gather(wt.Bool, self.pole_safe, indices)
                if count > 0
                else dr.zeros(wt.Bool, 0)
            ),
            "dif_n_p": (
                dr.gather(wt.Float, self.dif_n_p, indices)
                if count > 0
                else zero_float
            ),
            "dif_n_m": (
                dr.gather(wt.Float, self.dif_n_m, indices)
                if count > 0
                else zero_float
            ),
            "sum_n_p": (
                dr.gather(wt.Float, self.sum_n_p, indices)
                if count > 0
                else zero_float
            ),
            "sum_n_m": (
                dr.gather(wt.Float, self.sum_n_m, indices)
                if count > 0
                else zero_float
            ),
        }


class LengthProportionalStateSampler:
    def __init__(self, *, cdf, n_states: int, total_length_scalar: float) -> None:
        self.cdf = cdf
        self.n_states = int(n_states)
        self.total_length_scalar = float(total_length_scalar)

    @classmethod
    def from_line_length(cls, line_length) -> LengthProportionalStateSampler | None:
        n_states = int(dr.width(line_length))
        if n_states <= 0:
            return None
        total_length = dr.sum(line_length)
        total_length_scalar = float(scalar(total_length))
        if total_length_scalar <= 0.0:
            return None
        return cls(
            cdf=dr.cumsum(line_length),
            n_states=n_states,
            total_length_scalar=total_length_scalar,
        )

    def sample_slots(self, sample_index, *, seed: int):
        if self.n_states <= 0 or int(dr.width(sample_index)) <= 0:
            return dr.zeros(wt.UInt32, 0)
        sample_u = mc_common._hash_uniform_uint32(
            sample_index,
            stream=601,
            seed=seed,
        ) * wt.Float(self.total_length_scalar)
        return wt.UInt32(
            dr.binary_search(
                0,
                self.n_states - 1,
                lambda index: dr.gather(wt.Float, self.cdf, index) < sample_u,
            )
        )

    def total_length_weight(self, *, samples_per_tx: int) -> wt.Float:
        return wt.Float(self.total_length_scalar / float(max(1, samples_per_tx)))


def _gather_diffraction_edge_subset(scene: Scene, edge_idx, *, valid_mask=None):
    width = int(dr.width(edge_idx))
    if scene._diffraction_edge_gpu is None:
        return {
            "pos": wt.Point3f(0.0, 0.0, 0.0),
            "edge_dir": wt.Vector3f(0.0, 0.0, 1.0),
            "n0": wt.Vector3f(0.0, 0.0, 1.0),
            "n_face_n": wt.Vector3f(0.0, 0.0, -1.0),
            "wedge_n": wt.Float(1.5),
            "length": wt.Float(0.0),
            "line_min": wt.Float(0.0),
            "line_max": wt.Float(0.0),
            "adjacent_face0": wt.Int32(-1),
            "adjacent_face1": wt.Int32(-1),
            "valid": dr.zeros(wt.Bool, width),
        }

    edge_idx_i32 = wt.Int32(edge_idx)
    if valid_mask is None:
        valid_mask = edge_idx_i32 >= 0
    safe_idx = wt.UInt32(dr.select(valid_mask, edge_idx_i32, wt.Int32(0)))
    edge_gpu = scene._diffraction_edge_gpu
    return {
        "pos": dr.gather(wt.Point3f, edge_gpu["pos"], safe_idx),
        "edge_dir": dr.gather(wt.Vector3f, edge_gpu["edge_dir"], safe_idx),
        "n0": dr.gather(wt.Vector3f, edge_gpu["n0"], safe_idx),
        "n_face_n": dr.gather(wt.Vector3f, edge_gpu["n_face_n"], safe_idx),
        "wedge_n": dr.gather(wt.Float, edge_gpu["wedge_n"], safe_idx),
        "length": dr.gather(wt.Float, edge_gpu["length"], safe_idx),
        "line_min": dr.gather(wt.Float, edge_gpu["line_min"], safe_idx),
        "line_max": dr.gather(wt.Float, edge_gpu["line_max"], safe_idx),
        "adjacent_face0": dr.gather(wt.Int32, edge_gpu["adjacent_face0"], safe_idx),
        "adjacent_face1": dr.gather(wt.Int32, edge_gpu["adjacent_face1"], safe_idx),
        "valid": wt.Bool(valid_mask),
    }


class DirectTxDiffractionStates(Mapping[str, object]):
    FIELD_NAMES = (
        "edge_index",
        "edge_pos",
        "edge_dir",
        "n0",
        "n_face_n",
        "wedge_n",
        "edge_line_min",
        "edge_line_max",
        "source_pos",
        "adjacent_face0",
        "adjacent_face1",
        "face0_eta_r",
        "face0_sigma",
        "face0_gain",
        "face0_use_fresnel",
        "face1_eta_r",
        "face1_sigma",
        "face1_gain",
        "face1_use_fresnel",
    )

    def __init__(
        self,
        *,
        edge_index,
        edge_pos,
        edge_dir,
        n0,
        nn,
        wedge_n,
        edge_line_min,
        edge_line_max,
        source_pos,
        adjacent_face0,
        adjacent_face1,
        face0_material,
        face1_material,
        stored_count: int | None = None,
    ) -> None:
        self.edge_index = edge_index
        self.edge_pos = edge_pos
        self.edge_dir = edge_dir
        self.n0 = n0
        self.n_face_n = nn
        self.wedge_n = wedge_n
        self.edge_line_min = edge_line_min
        self.edge_line_max = edge_line_max
        self.source_pos = source_pos
        self.adjacent_face0 = adjacent_face0
        self.adjacent_face1 = adjacent_face1
        self.face0_eta_r = face0_material["eta_r"]
        self.face0_sigma = face0_material["sigma"]
        self.face0_gain = face0_material["gain"]
        self.face0_use_fresnel = face0_material["use_fresnel"]
        self.face1_eta_r = face1_material["eta_r"]
        self.face1_sigma = face1_material["sigma"]
        self.face1_gain = face1_material["gain"]
        self.face1_use_fresnel = face1_material["use_fresnel"]
        self.stored_count = None if stored_count is None else int(stored_count)

    def __getitem__(self, key: str):
        if key == "stored_count":
            if self.stored_count is None:
                raise KeyError(key)
            return self.stored_count
        if key not in self.FIELD_NAMES:
            raise KeyError(key)
        return getattr(self, key)

    def __iter__(self) -> Iterator[str]:
        yield from self.FIELD_NAMES
        if self.stored_count is not None:
            yield "stored_count"

    def __len__(self) -> int:
        return len(self.FIELD_NAMES) + (1 if self.stored_count is not None else 0)

    def get(self, key: str, default=None):
        if key == "stored_count":
            return self.stored_count if self.stored_count is not None else default
        if key not in self.FIELD_NAMES:
            return default
        return getattr(self, key)

    @classmethod
    def empty(cls, capacity: int) -> DirectTxDiffractionStates:
        zero_material = {
            "eta_r": dr.zeros(wt.Float, capacity),
            "sigma": dr.zeros(wt.Float, capacity),
            "gain": dr.zeros(wt.Float, capacity),
            "use_fresnel": dr.zeros(wt.Bool, capacity),
        }
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
            face0_material=zero_material,
            face1_material=zero_material,
        )

    @classmethod
    def from_mapping(cls, state_arrays: Mapping[str, object] | None) -> DirectTxDiffractionStates | None:
        if state_arrays is None:
            return None
        face0_material = {
            "eta_r": state_arrays["face0_eta_r"],
            "sigma": state_arrays["face0_sigma"],
            "gain": state_arrays["face0_gain"],
            "use_fresnel": state_arrays["face0_use_fresnel"],
        }
        face1_material = {
            "eta_r": state_arrays["face1_eta_r"],
            "sigma": state_arrays["face1_sigma"],
            "gain": state_arrays["face1_gain"],
            "use_fresnel": state_arrays["face1_use_fresnel"],
        }
        return cls(
            edge_index=state_arrays["edge_index"],
            edge_pos=state_arrays["edge_pos"],
            edge_dir=state_arrays["edge_dir"],
            n0=state_arrays["n0"],
            nn=state_arrays["n_face_n"],
            wedge_n=state_arrays["wedge_n"],
            edge_line_min=state_arrays["edge_line_min"],
            edge_line_max=state_arrays["edge_line_max"],
            source_pos=state_arrays["source_pos"],
            adjacent_face0=state_arrays["adjacent_face0"],
            adjacent_face1=state_arrays["adjacent_face1"],
            face0_material=face0_material,
            face1_material=face1_material,
            stored_count=state_arrays.get("stored_count"),
        )

    @classmethod
    def from_edge_indices(
        cls,
        *,
        tx_pos,
        edge_idx,
        scene: Scene,
        config: ResolvedTraceConfig,
    ) -> DirectTxDiffractionStates | None:
        n_states = int(dr.width(edge_idx))
        if n_states <= 0:
            return None

        edge_data = _gather_diffraction_edge_subset(
            scene,
            wt.Int32(edge_idx),
            valid_mask=dr.full(wt.Bool, True, n_states),
        )
        source_pos = wt.Point3f(
            dr.repeat(tx_pos.x, n_states),
            dr.repeat(tx_pos.y, n_states),
            dr.repeat(tx_pos.z, n_states),
        )
        edge_state = {
            "edge_pos": edge_data["pos"],
            "edge_dir": edge_data["edge_dir"],
            "n0": edge_data["n0"],
            "n_face_n": edge_data["n_face_n"],
            "source_pos": source_pos,
            "adjacent_face0": edge_data["adjacent_face0"],
            "adjacent_face1": edge_data["adjacent_face1"],
        }
        face0_material, face1_material = diff_geometry._edge_face_material_inputs(
            edge_state,
            n_states,
            config.diffraction_material,
            scene=scene,
            reflection_coef=config.reflection_coef,
            use_scene_materials=config.use_scene_materials_for_diffraction,
        )
        return cls(
            edge_index=wt.Int32(edge_idx),
            edge_pos=edge_data["pos"],
            edge_dir=edge_data["edge_dir"],
            n0=edge_data["n0"],
            nn=edge_data["n_face_n"],
            wedge_n=edge_data["wedge_n"],
            edge_line_min=edge_data["line_min"],
            edge_line_max=edge_data["line_max"],
            source_pos=source_pos,
            adjacent_face0=edge_data["adjacent_face0"],
            adjacent_face1=edge_data["adjacent_face1"],
            face0_material=face0_material,
            face1_material=face1_material,
        )

    def set_stored_count(self, count: int) -> None:
        self.stored_count = int(count)

    def width(self) -> int:
        if self.stored_count is not None:
            return int(self.stored_count)
        return int(dr.width(self.edge_pos.x))

    def line_lengths(self):
        return dr.maximum(self.edge_line_max - self.edge_line_min, wt.Float(0.0))

    def gather(self, indices) -> DirectTxDiffractionStates:
        face0_material = {
            "eta_r": dr.gather(wt.Float, self.face0_eta_r, indices),
            "sigma": dr.gather(wt.Float, self.face0_sigma, indices),
            "gain": dr.gather(wt.Float, self.face0_gain, indices),
            "use_fresnel": dr.gather(wt.Bool, self.face0_use_fresnel, indices),
        }
        face1_material = {
            "eta_r": dr.gather(wt.Float, self.face1_eta_r, indices),
            "sigma": dr.gather(wt.Float, self.face1_sigma, indices),
            "gain": dr.gather(wt.Float, self.face1_gain, indices),
            "use_fresnel": dr.gather(wt.Bool, self.face1_use_fresnel, indices),
        }
        return DirectTxDiffractionStates(
            edge_index=dr.gather(wt.Int32, self.edge_index, indices),
            edge_pos=dr.gather(wt.Point3f, self.edge_pos, indices),
            edge_dir=dr.gather(wt.Vector3f, self.edge_dir, indices),
            n0=dr.gather(wt.Vector3f, self.n0, indices),
            nn=dr.gather(wt.Vector3f, self.n_face_n, indices),
            wedge_n=dr.gather(wt.Float, self.wedge_n, indices),
            edge_line_min=dr.gather(wt.Float, self.edge_line_min, indices),
            edge_line_max=dr.gather(wt.Float, self.edge_line_max, indices),
            source_pos=dr.gather(wt.Point3f, self.source_pos, indices),
            adjacent_face0=dr.gather(wt.Int32, self.adjacent_face0, indices),
            adjacent_face1=dr.gather(wt.Int32, self.adjacent_face1, indices),
            face0_material=face0_material,
            face1_material=face1_material,
        )

    def orient_face_view(self, incident_dir):
        flip = dr.dot(incident_dir, self.n0) > 0.0
        return (
            dr.select(flip, -self.edge_dir, self.edge_dir),
            dr.select(flip, self.n_face_n, self.n0),
            dr.select(flip, self.n0, self.n_face_n),
            dr.select(flip, self.face1_eta_r, self.face0_eta_r),
            dr.select(flip, self.face1_sigma, self.face0_sigma),
            dr.select(flip, self.face1_gain, self.face0_gain),
            dr.select(flip, self.face1_use_fresnel, self.face0_use_fresnel),
            dr.select(flip, self.face0_eta_r, self.face1_eta_r),
            dr.select(flip, self.face0_sigma, self.face1_sigma),
            dr.select(flip, self.face0_gain, self.face1_gain),
            dr.select(flip, self.face0_use_fresnel, self.face1_use_fresnel),
        )


class DirectTxDiffractionStateStore:
    def __init__(self, *, capacity: int, seen, next_state_slot, state_arrays) -> None:
        self.capacity = int(capacity)
        self.seen = seen
        self.next_state_slot = next_state_slot
        self.state_arrays = state_arrays

    @classmethod
    def for_scene(cls, scene: Scene) -> DirectTxDiffractionStateStore:
        capacity = max(0, len(scene.vertical_edges))
        return cls(
            capacity=capacity,
            seen=dr.zeros(wt.UInt32, capacity),
            next_state_slot=wt.UInt32(0),
            state_arrays=DirectTxDiffractionStates.empty(capacity),
        )

    def count(self) -> int:
        return int(scalar(self.next_state_slot))

    def store_from_hit_data(
        self,
        *,
        tx_pos,
        ray_directions,
        prim_index,
        hit_p,
        hit_n,
        hit_geo_n,
        hit,
        scene: Scene,
        config: ResolvedTraceConfig,
    ) -> None:
        if self.capacity <= 0:
            return

        best_edge_idx = DirectTxWedgeDiscovery.best_edge_indices_from_hit_data(
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

        edge_data = _gather_diffraction_edge_subset(scene, best_edge_idx, valid_mask=store_unique)
        n_rays = int(dr.width(ray_directions.x))
        source_pos = dj_ops.broadcast_point(tx_pos, n_rays)
        edge_state = {
            "edge_pos": edge_data["pos"],
            "edge_dir": edge_data["edge_dir"],
            "n0": edge_data["n0"],
            "n_face_n": edge_data["n_face_n"],
            "source_pos": source_pos,
            "adjacent_face0": edge_data["adjacent_face0"],
            "adjacent_face1": edge_data["adjacent_face1"],
        }
        face0_material, face1_material = diff_geometry._edge_face_material_inputs(
            edge_state,
            n_rays,
            config.diffraction_material,
            scene=scene,
            reflection_coef=config.reflection_coef,
            use_scene_materials=config.use_scene_materials_for_diffraction,
        )

        dr.scatter(self.state_arrays.edge_pos, edge_data["pos"], state_slot, store_unique)
        dr.scatter(self.state_arrays.edge_index, wt.Int32(best_edge_idx), state_slot, store_unique)
        dr.scatter(self.state_arrays.edge_dir, edge_data["edge_dir"], state_slot, store_unique)
        dr.scatter(self.state_arrays.n0, edge_data["n0"], state_slot, store_unique)
        dr.scatter(self.state_arrays.n_face_n, edge_data["n_face_n"], state_slot, store_unique)
        dr.scatter(self.state_arrays.wedge_n, edge_data["wedge_n"], state_slot, store_unique)
        dr.scatter(self.state_arrays.edge_line_min, edge_data["line_min"], state_slot, store_unique)
        dr.scatter(self.state_arrays.edge_line_max, edge_data["line_max"], state_slot, store_unique)
        dr.scatter(self.state_arrays.source_pos, source_pos, state_slot, store_unique)
        dr.scatter(self.state_arrays.adjacent_face0, edge_data["adjacent_face0"], state_slot, store_unique)
        dr.scatter(self.state_arrays.adjacent_face1, edge_data["adjacent_face1"], state_slot, store_unique)
        dr.scatter(self.state_arrays.face0_eta_r, face0_material["eta_r"], state_slot, store_unique)
        dr.scatter(self.state_arrays.face0_sigma, face0_material["sigma"], state_slot, store_unique)
        dr.scatter(self.state_arrays.face0_gain, face0_material["gain"], state_slot, store_unique)
        dr.scatter(
            self.state_arrays.face0_use_fresnel,
            face0_material["use_fresnel"],
            state_slot,
            store_unique,
        )
        dr.scatter(self.state_arrays.face1_eta_r, face1_material["eta_r"], state_slot, store_unique)
        dr.scatter(self.state_arrays.face1_sigma, face1_material["sigma"], state_slot, store_unique)
        dr.scatter(self.state_arrays.face1_gain, face1_material["gain"], state_slot, store_unique)
        dr.scatter(
            self.state_arrays.face1_use_fresnel,
            face1_material["use_fresnel"],
            state_slot,
            store_unique,
        )


def _closest_point_on_edge_segment(*, query_point, edge_pos, edge_dir, line_min, line_max):
    edge_hat = edge_dir / (dr.norm(edge_dir) + wt.Float(EPS))
    ell = dr.clip(dr.dot(query_point - edge_pos, edge_hat), line_min, line_max)
    return edge_pos + edge_hat * ell, ell


def _surface_tangent_from_hit(ray_dir, surface_normal):
    tangent = ray_dir - dr.dot(ray_dir, surface_normal) * surface_normal
    tangent_norm = dr.norm(tangent)
    fallback_x = dr.cross(surface_normal, wt.Vector3f(1.0, 0.0, 0.0))
    fallback_y = dr.cross(surface_normal, wt.Vector3f(0.0, 1.0, 0.0))
    fallback = dr.select(dr.norm(fallback_x) > wt.Float(EPS), fallback_x, fallback_y)
    tangent = dr.select(tangent_norm > wt.Float(EPS), tangent, fallback)
    return tangent / (dr.norm(tangent) + wt.Float(EPS))


def _silhouette_viewpoint_from_hit_data(hit_p, shading_normal, geometric_normal, ray_dir):
    geometric_normal = dr.select(
        dr.norm(geometric_normal) > wt.Float(EPS),
        geometric_normal,
        shading_normal,
    )
    surface_normal = dr.select(
        dr.dot(ray_dir, geometric_normal) > 0.0,
        -geometric_normal,
        geometric_normal,
    )
    tangent = _surface_tangent_from_hit(ray_dir, surface_normal)
    theta = wt.Float(0.5 * math.pi - 0.05)
    d = dr.cos(theta) * surface_normal + dr.sin(theta) * tangent
    return hit_p + wt.Float(0.1) * d


class DirectTxWedgeDiscovery:
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
    def best_edge_indices_from_hit_data(
        cls,
        *,
        tx_pos,
        ray_directions,
        prim_index,
        hit_p,
        hit_n,
        hit_geo_n,
        hit,
        scene: Scene,
    ):
        n_rays = int(dr.width(ray_directions.x))
        if (
            scene.tri_data_gpu is None
            or scene._diffraction_edge_gpu is None
            or n_rays <= 0
            or len(scene.vertical_edges) <= 0
        ):
            return dr.full(wt.Int32, -1, max(0, n_rays))

        triangle_edges = scene.get_triangle_surface_edge_candidates(prim_index)
        candidate_slots = triangle_edges.get("slots", ())
        if len(candidate_slots) <= 0:
            return dr.full(wt.Int32, -1, n_rays)

        viewpoint = _silhouette_viewpoint_from_hit_data(
            hit_p,
            hit_n,
            hit_geo_n,
            ray_directions,
        )
        tx_pos_b = dj_ops.broadcast_point(tx_pos, n_rays)
        best_edge_idx = dr.full(wt.Int32, -1, n_rays)
        best_distance = dr.full(wt.Float, 1.0e30, n_rays)

        for slot_index, slot_edge_idx in enumerate(candidate_slots):
            slot_active = hit & (triangle_edges["count"] > wt.UInt32(slot_index)) & (slot_edge_idx >= 0)
            edge_data = _gather_diffraction_edge_subset(scene, slot_edge_idx, valid_mask=slot_active)
            edge_point, _ = _closest_point_on_edge_segment(
                query_point=viewpoint,
                edge_pos=edge_data["pos"],
                edge_dir=edge_data["edge_dir"],
                line_min=edge_data["line_min"],
                line_max=edge_data["line_max"],
            )
            flip = dr.dot(ray_directions, edge_data["n0"]) > 0.0
            oriented_n0 = dr.select(flip, edge_data["n_face_n"], edge_data["n0"])
            oriented_nn = dr.select(flip, edge_data["n0"], edge_data["n_face_n"])
            source_exterior = diff_geometry._wedge_exterior_region_mask(
                tx_pos_b - edge_point,
                edge_data["edge_dir"],
                oriented_n0,
                oriented_nn,
            )
            view_vec = viewpoint - edge_point
            face0_front = dr.dot(view_vec, edge_data["n0"]) > wt.Float(EPS)
            face1_front = dr.dot(view_vec, edge_data["n_face_n"]) > wt.Float(EPS)
            is_boundary_edge = edge_data["adjacent_face1"] < 0
            silhouette_edge = is_boundary_edge | (face0_front != face1_front)
            candidate_distance = dr.squared_norm(viewpoint - edge_point)
            better = (
                slot_active
                & silhouette_edge
                & source_exterior
                & (candidate_distance < best_distance)
            )
            best_distance = dr.select(better, candidate_distance, best_distance)
            best_edge_idx = dr.select(better, wt.Int32(slot_edge_idx), best_edge_idx)
        return best_edge_idx

    @classmethod
    def discover_from_hit_data(
        cls,
        *,
        tx_pos,
        ray_directions,
        prim_index,
        hit_p,
        hit_n,
        hit_geo_n,
        hit,
        scene: Scene,
    ):
        best_edge_idx = cls.best_edge_indices_from_hit_data(
            tx_pos=tx_pos,
            ray_directions=ray_directions,
            prim_index=prim_index,
            hit_p=hit_p,
            hit_n=hit_n,
            hit_geo_n=hit_geo_n,
            hit=hit,
            scene=scene,
        )
        discovered = hit & (best_edge_idx >= 0)
        if not dr.any(discovered):
            return dr.zeros(wt.UInt32, 0)

        safe_best_edge_idx = wt.UInt32(dr.select(discovered, best_edge_idx, wt.Int32(0)))
        discovered_lane = dr.compress(discovered)
        if int(dr.width(discovered_lane)) <= 0:
            return dr.zeros(wt.UInt32, 0)
        return cls.unique_edge_indices(
            dr.gather(wt.UInt32, safe_best_edge_idx, discovered_lane),
            n_edges=len(scene.vertical_edges),
        )

    @classmethod
    def discover_from_hits(cls, *, tx_pos, ray_directions, si, hit, scene: Scene):
        return cls.discover_from_hit_data(
            tx_pos=tx_pos,
            ray_directions=ray_directions,
            prim_index=si.prim_index,
            hit_p=si.p,
            hit_n=si.n,
            hit_geo_n=si.geo_n,
            hit=hit,
            scene=scene,
        )


@dr.syntax
def _trace_diffraction_batches_symbolic(
    *,
    scene,
    grid,
    diffraction_states,
    sampler,
    diffraction_batch_size: int,
    diffraction_batch_count: int,
    samples_per_tx: int,
    seed: int,
    k,
    wavelength,
    plane_normal,
    diffraction_path_gain_scale,
    weighted_diagnostics,
    diffraction_tape_store: DiffractionTapeStore | None = None,
    loop_mode: str = "symbolic",
):
    sample_lane = dr.arange(wt.UInt32, int(diffraction_batch_size))
    batch_start = wt.UInt32(0)
    batch_stride = wt.UInt32(int(diffraction_batch_size))
    total_samples_u32 = wt.UInt32(int(samples_per_tx))
    total_length_weight = sampler.total_length_weight(samples_per_tx=samples_per_tx)

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
        sample_index = sample_lane + batch_start
        sample_active = sample_index < total_samples_u32
        state_slot = sampler.sample_slots(
            sample_index,
            seed=int(seed),
        )
        batch_states = diffraction_states.gather(state_slot)
        edge_hat = batch_states["edge_dir"] / (dr.norm(batch_states["edge_dir"]) + wt.Float(EPS))
        line_min = batch_states["edge_line_min"]
        line_max = batch_states["edge_line_max"]
        line_length = dr.maximum(line_max - line_min, wt.Float(0.0))
        edge_fraction = mc_common._hash_uniform_uint32(
            sample_index,
            stream=602,
            seed=int(seed),
        )
        ell_sample = line_min + line_length * edge_fraction
        diff_point = batch_states["edge_pos"] + edge_hat * ell_sample
        incident_dir = diff_point - batch_states["source_pos"]
        (
            oriented_edge_dir,
            oriented_n0,
            oriented_nn,
            oriented_face0_eta_r,
            oriented_face0_sigma,
            oriented_face0_gain,
            oriented_face0_use_fresnel,
            oriented_face1_eta_r,
            oriented_face1_sigma,
            oriented_face1_gain,
            oriented_face1_use_fresnel,
        ) = batch_states.orient_face_view(incident_dir)
        face_sum = oriented_n0 + oriented_nn
        face_sum_norm = dr.norm(face_sum)
        offset_normal = dr.select(
            face_sum_norm > wt.Float(EPS),
            face_sum / face_sum_norm,
            wt.Vector3f(0.0, 0.0, 0.0),
        )
        diff_point_offset = diff_point + wt.Float(mc_common._MC_DIFFRACTION_OFFSET) * offset_normal
        keller_sample = mc_common._hash_uniform_uint32(sample_index, stream=603, seed=int(seed))
        ko = _sample_keller_cone(
            oriented_edge_dir,
            oriented_n0,
            oriented_nn,
            keller_sample,
            incident_dir,
            lit_region=True,
        )
        ray_origin = mc_common._spawn_offset_ray_origin(
            diff_point,
            ko,
            offset_normal,
        )
        plane_hit = mc_common._plane_hit_from_segment(
            ray_origin=ray_origin,
            ray_dir=ko,
            blocker_dist=dr.full(wt.Float, 1.0e10, int(diffraction_batch_size)),
            grid=grid,
            active=sample_active,
        )
        visible_source = diff_geometry._segment_visibility_mask(
            batch_states["source_pos"],
            diff_point,
            scene,
        )
        visible_source_offset = diff_geometry._segment_visibility_mask(
            batch_states["source_pos"],
            diff_point_offset,
            scene,
        )
        safe_target_diff_point = dr.select(plane_hit["valid"], diff_point, plane_hit["target_pos"])
        safe_target_diff_point_offset = dr.select(
            plane_hit["valid"],
            diff_point_offset,
            plane_hit["target_pos"],
        )
        visible_target_base = diff_geometry._segment_visibility_mask(
            plane_hit["target_pos"],
            safe_target_diff_point,
            scene,
        )
        visible_target_offset = diff_geometry._segment_visibility_mask(
            plane_hit["target_pos"],
            safe_target_diff_point_offset,
            scene,
        )
        source_visible = visible_source & visible_source_offset
        visible_target = plane_hit["valid"] & visible_target_base & visible_target_offset
        wedge_interior = dr.safe_acos(
            dr.clip(
                -dr.dot(
                    oriented_n0 / (dr.norm(oriented_n0) + wt.Float(EPS)),
                    oriented_nn / (dr.norm(oriented_nn) + wt.Float(EPS)),
                ),
                wt.Float(-1.0),
                wt.Float(1.0),
            )
        )
        exterior_angle = wt.Float(2.0 * math.pi) - wedge_interior
        integration_weight = _diffraction_integration_weight(
            edge_origin=batch_states["edge_pos"],
            edge_dir=oriented_edge_dir,
            n0=oriented_n0,
            source_pos=batch_states["source_pos"],
            diff_point=diff_point,
            k_world=ko,
            target_pos=plane_hit["target_pos"],
            plane_normal=plane_normal,
        )
        field_power, field_valid, field_support = diff_field._sampled_edge_diffraction_power_to_targets_mc(
            source_pos=batch_states["source_pos"],
            edge_dir=oriented_edge_dir,
            n0=oriented_n0,
            nn=oriented_nn,
            wedge_n=batch_states["wedge_n"],
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
            k=k,
            wavelength=wavelength,
            return_valid=True,
            return_support=True,
        )
        contribution_active = sample_active & source_visible & visible_target & field_valid
        contribution = dr.select(
            contribution_active,
            field_power
            * diffraction_path_gain_scale
            * integration_weight
            * total_length_weight
            * exterior_angle,
            wt.Float(0.0),
        )
        cell_idx = mc_common._axis_aligned_cell_index(
            grid=grid,
            coord_0=plane_hit["coord_0"],
            coord_1=plane_hit["coord_1"],
        )
        if dr.hint(diffraction_tape_store is not None, mode="scalar"):
            diffraction_tape_store.store(
                edge_index=batch_states.edge_index,
                edge_fraction=edge_fraction,
                cone_sample=keller_sample,
                cell_idx=cell_idx,
                field_valid=field_support["field_valid"],
                pole_safe=field_support["pole_safe"],
                dif_n_p=field_support["dif_n_p"],
                dif_n_m=field_support["dif_n_m"],
                sum_n_p=field_support["sum_n_p"],
                sum_n_m=field_support["sum_n_m"],
                active=contribution_active,
            )
        mc_common._scatter_component(
            grid=grid,
            weighted_diagnostics=weighted_diagnostics,
            component="diffraction",
            coord_0=plane_hit["coord_0"],
            coord_1=plane_hit["coord_1"],
            power=contribution,
            active=contribution_active,
        )
        batch_start += batch_stride

    return wt.UInt32(0)


def _direct_tx_diffraction_runtime_backend(*, implementation: str, wedge_discovery_backend: str):
    return {
        "implementation": str(implementation),
        "cell_scatter_backend": "drjit_scatter_reduce",
        "wedge_discovery_backend": str(wedge_discovery_backend),
        "state_sampler": "discovered_wedge_length_proportional_then_uniform_edge_position_then_keller_cone",
        "point_evaluation_backend": "sampled_edge_diffraction_field_to_plane_hits",
        "source_field_contract": "sionna_iso_v_implicit_basis",
    }


def _prepare_direct_tx_diffraction_states(
    *,
    tx_pos,
    discovered_edge_idx,
    scene: Scene,
    config: ResolvedTraceConfig,
    solver_controls,
    persistent_diffraction_state_cache: dict[tuple[object, ...], object] | None,
    local_diffraction_state_cache: dict[tuple[object, ...], object] | None,
    diffraction_state_cache_key_fn: Callable[[float, object | None], tuple[object, ...]] | None,
    return_timing: bool,
    state_builder=None,
):
    del persistent_diffraction_state_cache
    del local_diffraction_state_cache
    del diffraction_state_cache_key_fn
    if state_builder is None:
        state_builder = DirectTxDiffractionStates.from_edge_indices
    effective = solver_controls["effective"]
    runtime_backend = _direct_tx_diffraction_runtime_backend(
        implementation="depth0_reused_first_hit_wedge_discovery_plus_keller_cone_symbolic_loop_scatter_reduce",
        wedge_discovery_backend="depth0_reused_triangle_surface_edge_candidates_nearest_silhouette_projection",
    )
    if int(effective["max_diffractions"]) <= 0:
        return {
            "state_arrays": None,
            "runtime_reuse": {
                "cache_mode": "disabled",
                "state_preparation_hits": 0,
                "state_preparation_misses": 0,
                "state_layout": "direct_tx_first_order_only",
            },
            "state_pool": {
                "total": 0,
                "kept": 0,
                "threshold_pruned": 0,
                "roulette_pruned": 0,
            },
            "runtime_backend": runtime_backend,
            "preparation_seconds": 0.0,
        }
    cache_mode = "disabled"
    cache_hits = 0
    cache_misses = 0
    preparation_seconds = 0.0
    t0 = None
    if return_timing:
        dj_ops.sync_thread()
        t0 = time.perf_counter()
    state_arrays = state_builder(
        tx_pos=tx_pos,
        edge_idx=discovered_edge_idx,
        scene=scene,
        config=config,
    )
    if return_timing and t0 is not None:
        dj_ops.sync_thread()
        preparation_seconds = time.perf_counter() - t0

    n_states = 0 if state_arrays is None else state_arrays.width()
    return {
        "state_arrays": state_arrays,
        "runtime_reuse": {
            "cache_mode": cache_mode,
            "state_preparation_hits": cache_hits,
            "state_preparation_misses": cache_misses,
            "state_layout": "depth0_reused_discovered_wedges",
        },
        "state_pool": {
            "total": n_states,
            "kept": n_states,
            "threshold_pruned": 0,
            "roulette_pruned": 0,
        },
        "runtime_backend": runtime_backend,
        "preparation_seconds": preparation_seconds,
    }
