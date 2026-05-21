"""Current TX-emitted Monte Carlo transport integrator."""

from __future__ import annotations

import dataclasses
import math
import time
from dataclasses import dataclass
from typing import Mapping, TYPE_CHECKING

if TYPE_CHECKING:
    from ..trace.diffraction import DiffractionTape
    from ..trace.los import LosTape
    from ..trace.reflection import ReflectionTape

import drjit as dr
import torch
from witwin.channel.core.scene import Scene
from ..sampler import Sampler
from ..trace import ad_support as mc_ad
from ..trace.diffraction import (
    Diffraction,
    DiffractionHitStore,
    DiffractionStates,
    DiffractionEdgeSampler,
    DiffractionTapeStore,
)
from ..trace.postprocessing import ShadowBoundary
from ..trace.los import LoS
from ..trace import reflection as mc_reflection
from .. import types as wt
from witwin.channel._native.montecarlo import NativeExtension
from ..config import Config, ResolvedTraceConfig
from witwin.channel.core.grid import Grid, GridSpec
from ..grid_ops import GridContributionStore
from ..kernels.diffraction_builder import DiffractionBuilderKernel
from ..kernels.sparse_coeff import SparseCoeffKernel
from ..filtering import apply_power_filtering
from ..types import Integrator
from witwin.channel.core.numerics.arrays import scalar
from witwin.channel.core.numerics import arrays
from witwin.channel.core.physics import polarization
from witwin.channel.core.numerics.arrays import complex_zero
from witwin.channel.core.runtime import (
    point_grad_enabled,
    scene_geometry_grad_enabled,
    scene_material_grad_enabled,
)
from witwin.channel.core.physics.wave_math import material_angular_frequency
from witwin.channel.core.numerics.tensors import (
    to_complex_array,
    to_float_tensor,
    to_int_tensor,
    to_mapping_proxy,
)
from witwin.channel.core.results import (
    RadioMapPowerPayload,
    RadioMapResult,
    coordinates_from_grid,
)
from . import basic_ad as mc_custom
from .metadata import COMPONENT_NAMES, build_metadata
_MC_MIN_BATCH_MEMORY_BYTES = 16 * 1024 * 1024
_MC_RAY_BATCH_BUDGET_RATIO = 0.5
_MC_DIFFRACTION_BATCH_BUDGET_RATIO = 0.5
_MC_ESTIMATED_RAY_WORKING_SET_BYTES = 768
_MC_ESTIMATED_DIFFRACTION_SAMPLE_BYTES = 3072
_RAYD_REFLECTION_ACCUMULATION_BACKEND = "rayd_reflection_accumulation"

EXTRA_COMPONENT_KEYS = (
    "diffraction_incident_transition_power",
    "diffraction_reflection_transition_power",
    "continued_incident_power",
    "incident_shadow_boundary_weight",
    "reflection_shadow_boundary_weight",
    "incident_transition_response_real",
    "incident_transition_response_imag",
    "reflection_transition_response_real",
    "reflection_transition_response_imag",
)


def _empty_radio_map(n_rx: int) -> dict:
    """Zero-initialized per-cell accumulators for coherent / incoherent / coherent_power."""
    keys = ("los", "reflection", "diffraction", "total")
    incoherent_keys = keys + ("raw_total", "shadow_boundary_correction", *EXTRA_COMPONENT_KEYS)
    zero = lambda: dr.zeros(wt.Float, n_rx)
    return {
        "coherent": {k: complex_zero(n_rx) for k in keys},
        "incoherent": {k: zero() for k in incoherent_keys},
        "coherent_power": {k: zero() for k in keys},
    }


def _finalize_component_totals(
    weighted_diagnostics: dict,
    *,
    shadow_boundary_mode: str = "utd_power_smoothing",
) -> dict:
    coh = weighted_diagnostics["coherent"]
    weighted_diagnostics["coherent"]["total"] = wt.Complex2f(
        coh["los"].real + coh["reflection"].real + coh["diffraction"].real,
        coh["los"].imag + coh["reflection"].imag + coh["diffraction"].imag,
    )
    inc = weighted_diagnostics["incoherent"]
    raw_total = inc["los"] + inc["reflection"] + inc["diffraction"]
    inc["raw_total"] = raw_total
    if str(shadow_boundary_mode) == "utd_power_smoothing":
        incident_transition_excess = inc["diffraction_incident_transition_power"]
        reflection_transition_excess = inc["diffraction_reflection_transition_power"]
        incident_weight = inc["incident_shadow_boundary_weight"]
        reflection_weight = inc["reflection_shadow_boundary_weight"]
        incident_response_real = inc["incident_transition_response_real"]
        incident_response_imag = inc["incident_transition_response_imag"]
        reflection_response_real = inc["reflection_transition_response_real"]
        reflection_response_imag = inc["reflection_transition_response_imag"]
        incident_side = dr.select(
            inc["los"] > wt.Float(0.0),
            wt.Float(1.0),
            wt.Float(-1.0),
        )
        incident_coeff_abs2 = wt.Float(0.25) * (
            dr.square(wt.Float(1.0) + incident_side * incident_response_real)
            + dr.square(incident_response_imag)
        )
        incident_target_power = inc["continued_incident_power"] * incident_coeff_abs2
        incident_correction = incident_weight * (
            incident_target_power - inc["los"] - incident_transition_excess
        )

        reflection_side = dr.select(
            inc["reflection"] > wt.Float(0.0),
            wt.Float(1.0),
            wt.Float(-1.0),
        )
        reflection_coeff_abs2 = wt.Float(0.25) * (
            dr.square(
                wt.Float(1.0)
                + reflection_side * reflection_response_real
            )
            + dr.square(reflection_response_imag)
        )
        reflection_continued_power = dr.maximum(
            inc["reflection"],
            reflection_transition_excess,
        )
        reflection_target_power = reflection_continued_power * reflection_coeff_abs2
        reflection_correction = reflection_weight * (
            reflection_target_power - inc["reflection"] - reflection_transition_excess
        )
        correction = incident_correction + reflection_correction
        positive_support = raw_total > wt.Float(1.0e-14)
        correction = dr.select(
            (correction > wt.Float(0.0)) & ~positive_support,
            wt.Float(0.0),
            correction,
        )
        correction = dr.maximum(correction, -raw_total)
    elif str(shadow_boundary_mode) == "none":
        correction = dr.zeros(wt.Float, int(dr.width(raw_total)))
    else:
        raise ValueError(
            "shadow_boundary_mode must be 'utd_power_smoothing' or 'none'."
        )
    inc["shadow_boundary_correction"] = correction
    inc["total"] = raw_total + correction
    return weighted_diagnostics


def finalize_weighted_diagnostics(
    weighted_diagnostics: dict,
    *,
    shadow_boundary_mode: str = "utd_power_smoothing",
    grid=None,
    filtering=None,
) -> dict:
    """Apply optional power filtering, then finalize derived map totals."""
    apply_power_filtering(
        weighted_diagnostics,
        filtering=filtering,
        grid=grid,
    )
    return _finalize_component_totals(
        weighted_diagnostics,
        shadow_boundary_mode=shadow_boundary_mode,
    )


def _single_tx_sinr(rss: wt.Float, *, noise_power: float) -> wt.Float:
    n_rx = int(dr.width(rss))
    if noise_power > 0.0:
        return rss / float(noise_power)
    inf = dr.full(wt.Float, float("inf"), n_rx)
    return dr.select(rss > 0.0, inf, dr.zeros(wt.Float, n_rx))


def _capture_free_bytes(*, release_reclaimable_caches: bool = False) -> int:
    """Return CUDA free bytes after optional cache reclaim."""
    if release_reclaimable_caches:
        if hasattr(dr, "flush_malloc_cache"):
            dr.flush_malloc_cache()
        torch.cuda.empty_cache()
    device = torch.cuda.current_device()
    free_bytes, _ = torch.cuda.mem_get_info(device)
    return int(free_bytes)


def _align_batch_size(target: int, *, upper_bound: int) -> int:
    if upper_bound <= 0:
        return 0
    resolved_target = max(1, int(target))
    return int(min(resolved_target, int(upper_bound)))


def _phase_batch_size(
    samples_per_tx: int,
    free_bytes: int,
    budget_ratio: float,
    estimated_bytes_per_sample: int,
) -> tuple[int, str]:
    """Pick a batch size from the current CUDA free-memory budget."""
    budget_bytes = max(_MC_MIN_BATCH_MEMORY_BYTES, int(budget_ratio * free_bytes))
    target = max(1, budget_bytes // estimated_bytes_per_sample)
    if target >= samples_per_tx:
        return samples_per_tx, "full_batch_with_memory_guardrail"
    return int(target), "cuda_free_memory_guardrail"


def _stop_threshold_linear(stop_threshold: float) -> float:
    if stop_threshold <= 0.0:
        return 0.0
    return float(math.pow(10.0, stop_threshold / 10.0))


def _resolve_batch_plan(
    *,
    samples_per_tx: int,
    diffraction_state_count: int,
    free_bytes: int,
) -> wt.BatchPlan:
    """Resolve reflection and diffraction batch sizes from the current CUDA budget."""
    ray_batch_size, ray_reason = _phase_batch_size(
        samples_per_tx, free_bytes,
        _MC_RAY_BATCH_BUDGET_RATIO, _MC_ESTIMATED_RAY_WORKING_SET_BYTES,
    )
    ray_batch_count = int(math.ceil(samples_per_tx / ray_batch_size))

    if diffraction_state_count > 0:
        diff_batch_size, diff_reason = _phase_batch_size(
            samples_per_tx, free_bytes,
            _MC_DIFFRACTION_BATCH_BUDGET_RATIO, _MC_ESTIMATED_DIFFRACTION_SAMPLE_BYTES,
        )
        diff_batch_count = int(math.ceil(samples_per_tx / diff_batch_size))
    else:
        diff_batch_size, diff_batch_count, diff_reason = 0, 0, "disabled"

    return wt.BatchPlan(
        ray_batch_size=ray_batch_size,
        ray_batch_count=ray_batch_count,
        ray_policy=ray_reason,
        diffraction_batch_size=diff_batch_size,
        diffraction_batch_count=diff_batch_count,
        diffraction_policy=diff_reason,
        free_cuda_bytes=int(free_bytes),
        scatter_safe_batch_cap=samples_per_tx,
    )


def build_result(
    *,
    grid: Grid,
    tx_power: float,
    weighted_diagnostics: dict,
    metadata: dict,
    path_gain,
    rss,
    sinr,
    tx_pos,
    noise_power: float,
    timing: wt.TraceTiming | None,
    detach_tensors: bool = False,
) -> RadioMapResult:
    """Construct the public Result payload from solver-side primitives."""
    grid_shape = (int(grid.grid_shape[0]), int(grid.grid_shape[1]))
    cell_tensor_shape = (grid_shape[1], grid_shape[0])
    tx_tensor_shape = (1, *cell_tensor_shape)
    timing_payload = None if timing is None else dataclasses.asdict(timing)
    result_metadata = dict(metadata)
    if timing_payload is not None:
        result_metadata["timing"] = dict(timing_payload)
    coherent_payload = to_mapping_proxy({
        str(name): to_complex_array(value, shape=cell_tensor_shape)
        for name, value in dict(weighted_diagnostics["coherent"]).items()
    })
    incoherent_payload = to_mapping_proxy({
        str(name): to_float_tensor(value, shape=tx_tensor_shape, detach=detach_tensors)
        for name, value in dict(weighted_diagnostics["incoherent"]).items()
    })
    coherent_power_payload = to_mapping_proxy({
        str(name): to_float_tensor(value, shape=tx_tensor_shape, detach=detach_tensors)
        for name, value in dict(weighted_diagnostics["coherent_power"]).items()
    })
    best_tx_index = to_int_tensor(torch.zeros(cell_tensor_shape, dtype=torch.int32), shape=cell_tensor_shape)
    result_metadata["transmitter_count"] = 1
    return RadioMapResult(
        name="montecarlo",
        kind="radio_map",
        metric="path_gain",
        solver="montecarlo",
        combine_mode="incoherent",
        receiver_model="matched_isotropic",
        grid_shape=grid_shape,
        cell_size=(float(grid.cell_size[0]), float(grid.cell_size[1])),
        surface=to_mapping_proxy(grid.surface_descriptor()),
        coords=coordinates_from_grid(grid),
        path_gain=to_float_tensor(path_gain, shape=tx_tensor_shape, detach=detach_tensors),
        rss=to_float_tensor(rss, shape=tx_tensor_shape, detach=detach_tensors),
        sinr=to_float_tensor(sinr, shape=tx_tensor_shape, detach=detach_tensors),
        best_tx_index=best_tx_index,
        tx_association_map=best_tx_index,
        coherent=coherent_payload,
        incoherent=incoherent_payload,
        coherent_power=coherent_power_payload,
        metadata=to_mapping_proxy(result_metadata),
        tx_pos=(scalar(tx_pos.x), scalar(tx_pos.y), scalar(tx_pos.z)),
        tx_power=float(tx_power),
        noise_power=float(noise_power),
        field=None,
        power=RadioMapPowerPayload(
            incoherent=incoherent_payload,
            coherent_power=coherent_power_payload,
            coherent=coherent_payload,
        ),
        timing=None if timing_payload is None else to_mapping_proxy(timing_payload),
    )


@dataclass(slots=True)
class ForwardPassState:
    """Forward-pass state bundle before result assembly or AD replay."""

    grid: Grid
    component_power: dict[str, object]
    weighted_diagnostics: dict[str, object]
    metadata: dict[str, object]
    noise_power: float
    timing: wt.TraceTiming | None
    diffraction_edge_indices: object | None
    los_tape: LosTape | None
    reflection_tape: ReflectionTape | None
    diffraction_tape: DiffractionTape | None
    diff_length_weight: float


@dataclass(slots=True)
class IntegratorSetting:
    grid: Grid
    resolved_accumulation_backend: str
    samples_per_tx: int
    reflection_n_rays: int
    seed: int
    tx_power: float
    rr_depth: int | None
    rr_prob: float
    stop_threshold: float
    stop_threshold_linear: float
    weighted_diagnostics: dict[str, object]
    cell_area: float
    diff_gain_scale: object
    material_omega: object
    timing: wt.TraceTiming | None
    ray_sampling_metadata: Mapping[str, object]
    solid_angle_per_ray: float
    reflection_solid_angle_per_ray: float
    reflection_runtime_backend: dict[str, object]
    collect_wedges: bool
    batch_plan: wt.BatchPlan
    reflection_batch_plan: wt.BatchPlan
    loop_mode: str
    config: ResolvedTraceConfig


class Basic(Integrator):
    """Own sampling and path-family composition for the current public solver."""

    mode = "basic"

    @staticmethod
    def resolve_noise_power(scene: Scene, noise_power: float | None) -> float:
        if noise_power is not None:
            return float(noise_power)
        for key in ("noise_power", "thermal_noise_power"):
            value = getattr(scene, key, None)
            if value is not None:
                return float(value)
        return 0.0

    @staticmethod
    def resolve_accumulation(requested_backend: str) -> str:
        backend = str(requested_backend).lower()
        if backend not in {"auto", "native_monte_carlo", _RAYD_REFLECTION_ACCUMULATION_BACKEND}:
            raise ValueError(
                "accumulation_backend must be 'auto', 'native_monte_carlo', "
                "or 'rayd_reflection_accumulation'."
            )
        if not NativeExtension.native_extension_available():
            raise RuntimeError(
                "sampling_mode='monte_carlo' requires the bundled native extension."
            )
        if backend == _RAYD_REFLECTION_ACCUMULATION_BACKEND:
            return _RAYD_REFLECTION_ACCUMULATION_BACKEND
        return "native_monte_carlo"

    @staticmethod
    def _rayd_reflection_wedge_store(
        *,
        events,
        tx_pos,
        max_bounces: int,
    ) -> DiffractionHitStore:
        capacity = int(getattr(events, "capacity", 0))
        count = int(scalar(wt.Int32(events.count)))
        if count > capacity:
            raise RuntimeError(
                "RayD reflection wedge event capacity was exceeded; increase the "
                "explicit wedge capacity before using rayd_reflection_accumulation."
            )
        if count <= 0:
            return DiffractionHitStore(capacity=0, max_bounces=0)

        indices = dr.arange(wt.UInt32, count)
        directions_all = wt.Vector3f(events.directions)
        hit_points_all = wt.Point3f(events.hit_points)
        normals_all = wt.Vector3f(events.normals)
        prim_all = wt.Int32(events.prim_id)
        depth_all = wt.Int32(events.bounce_depth)

        directions = dr.gather(wt.Vector3f, directions_all, indices)
        hit_points = dr.gather(wt.Point3f, hit_points_all, indices)
        normals = dr.gather(wt.Vector3f, normals_all, indices)
        prim_id = dr.gather(wt.Int32, prim_all, indices)
        depth = dr.gather(wt.Int32, depth_all, indices)
        active = (prim_id >= wt.Int32(0)) & (depth == wt.Int32(0))

        store = DiffractionHitStore(capacity=count, max_bounces=0)
        store.store(
            ray_directions=directions,
            prim_index=prim_id,
            hit_p=hit_points,
            hit_n=normals,
            hit_geo_n=normals,
            source_pos=arrays.broadcast_point(tx_pos, count),
            source_power=dr.full(wt.Float, 1.0, count),
            prefix_reflection_depth=depth,
            initial_ray_dir=directions,
            prim_history=(),
            slot_index=None,
            active=active,
        )
        return store

    @staticmethod
    def _rayd_reflection_field_vector(result):
        missing = [
            name
            for name in ("reflection_field_x", "reflection_field_y", "reflection_field_z")
            if not hasattr(result, name)
        ]
        if missing:
            raise RuntimeError(
                "RayD reflection accumulation must return complex reflection field "
                f"components; missing {', '.join(missing)}."
            )
        return {
            "x": wt.Complex2f(
                wt.Float(result.reflection_field_x.real),
                wt.Float(result.reflection_field_x.imag),
            ),
            "y": wt.Complex2f(
                wt.Float(result.reflection_field_y.real),
                wt.Float(result.reflection_field_y.imag),
            ),
            "z": wt.Complex2f(
                wt.Float(result.reflection_field_z.real),
                wt.Float(result.reflection_field_z.imag),
            ),
        }

    @staticmethod
    def _run_rayd_reflection_accumulation(
        *,
        scene,
        grid,
        tx_pos,
        config,
        samples_per_tx: int,
        seed: int,
        solid_angle_per_ray: float,
        cell_area: float,
        effective: dict,
        rr_depth,
        rr_prob: float,
        stop_threshold_linear: float,
        weighted_diagnostics: dict,
        collect_wedges: bool,
        collect_ad_tapes: bool,
        collect_wedge_prefixes: bool,
    ) -> wt.ReflectionPhaseResult:
        if collect_wedge_prefixes:
            raise RuntimeError(
                "accumulation_backend='rayd_reflection_accumulation' does not support prefix wedge events yet."
            )
        scene_grad_enabled = (
            hasattr(scene, "_triangle_runtime")
            and (scene_geometry_grad_enabled(scene) or scene_material_grad_enabled(scene))
        )
        if collect_ad_tapes or point_grad_enabled(tx_pos) or scene_grad_enabled:
            raise RuntimeError(
                "accumulation_backend='rayd_reflection_accumulation' does not support AD; "
                "use the existing AD tape path explicitly."
            )
        trace = getattr(scene, "trace_reflections_accumulating", None)
        if trace is None:
            raise RuntimeError(
                "accumulation_backend='rayd_reflection_accumulation' requires "
                "Scene.trace_reflections_accumulating."
            )

        max_bounces = int(effective["reflection_max_bounces"])
        path_counts = wt.PathCounts()
        if max_bounces <= 0:
            diff_state_store = (
                DiffractionHitStore(capacity=0, max_bounces=0)
                if collect_wedges
                else None
            )
            return wt.ReflectionPhaseResult(
                path_counts=path_counts,
                path_tape_store=None,
                diff_state_store=diff_state_store,
            )

        rr_depth_native = 0 if rr_depth is None else int(rr_depth)
        if rr_depth_native < 0:
            raise RuntimeError("rayd_reflection_accumulation requires rr_depth >= 0.")
        if not (0.0 < float(rr_prob) <= 1.0):
            raise RuntimeError("rayd_reflection_accumulation requires rr_prob in (0, 1].")
        if float(stop_threshold_linear) < 0.0:
            raise RuntimeError("rayd_reflection_accumulation requires stop_threshold >= 0.")

        ray_index = dr.arange(wt.UInt32, int(samples_per_tx))
        ray_dir = Sampler.directions(samples_per_tx, ray_index=ray_index)
        ray_origin = arrays.broadcast_point(tx_pos, int(samples_per_tx))
        wedge_capacity = int(samples_per_tx) if collect_wedges else 0
        result = trace(
            ray_origin=ray_origin,
            ray_dir=ray_dir,
            tx_pos=tx_pos,
            grid=grid,
            config=config,
            max_bounces=max_bounces,
            seed=int(seed),
            rr_depth=rr_depth_native,
            rr_prob=float(rr_prob),
            stop_threshold_linear=float(stop_threshold_linear),
            solid_angle_per_ray=float(solid_angle_per_ray),
            cell_area=float(cell_area),
            collect_wedges=bool(collect_wedges),
            wedge_capacity=wedge_capacity,
            tx_polarization=getattr(config, "tx_polarization", (1.0, 0.0, 0.0)),
            active=dr.full(wt.Bool, True, int(samples_per_tx)),
        )

        reflection_power = wt.Float(result.reflection_power)
        if int(dr.width(reflection_power)) != int(grid.n_cells):
            raise RuntimeError(
                "RayD reflection accumulation returned a grid width that does not match the receiver grid."
            )
        weighted_diagnostics["incoherent"]["reflection"] = (
            weighted_diagnostics["incoherent"]["reflection"] + reflection_power
        )
        field_vec = Basic._rayd_reflection_field_vector(result)
        rx_polarization = polarization.effective_rx_polarization(
            getattr(config, "rx_polarization", None),
            getattr(config, "tx_polarization", (1.0, 0.0, 0.0)),
        )
        coherent_reflection = polarization.scalarize_vector_to_tangential_polarization(
            field_vec,
            rx_polarization,
            axis=str(grid.axis),
        )
        weighted_diagnostics["coherent"]["reflection"] = (
            weighted_diagnostics["coherent"]["reflection"] + coherent_reflection
        )
        weighted_diagnostics["coherent_power"]["reflection"] = (
            weighted_diagnostics["coherent_power"]["reflection"]
            + (
                coherent_reflection.real * coherent_reflection.real
                + coherent_reflection.imag * coherent_reflection.imag
            )
        )
        path_counts.reflection += wt.UInt32(result.reflection_count)
        diff_state_store = (
            Basic._rayd_reflection_wedge_store(
                events=result.wedge_events,
                tx_pos=tx_pos,
                max_bounces=max_bounces,
            )
            if collect_wedges
            else None
        )
        return wt.ReflectionPhaseResult(
            path_counts=path_counts,
            path_tape_store=None,
            diff_state_store=diff_state_store,
        )

    # Run reflection trace in batches, collecting LOS and reflection hits.
    @staticmethod
    def run_reflection(
        *,
        scene,
        grid,
        tx_pos,
        config,
        samples_per_tx: int,
        seed: int,
        ray_batch_size: int,
        solid_angle_per_ray: float,
        cell_area: float,
        effective: dict,
        rr_depth,
        rr_prob: float,
        stop_threshold_linear: float,
        material_omega,
        weighted_diagnostics: dict,
        collect_wedges: bool,
        collect_ad_tapes: bool,
        collect_wedge_prefixes: bool = False,
        loop_mode: str,
        resolved_accumulation_backend: str,
    ) -> wt.ReflectionPhaseResult:
        del ray_batch_size
        if str(resolved_accumulation_backend) == _RAYD_REFLECTION_ACCUMULATION_BACKEND:
            return Basic._run_rayd_reflection_accumulation(
                scene=scene,
                grid=grid,
                tx_pos=tx_pos,
                config=config,
                samples_per_tx=samples_per_tx,
                seed=seed,
                solid_angle_per_ray=solid_angle_per_ray,
                cell_area=cell_area,
                effective=effective,
                rr_depth=rr_depth,
                rr_prob=rr_prob,
                stop_threshold_linear=stop_threshold_linear,
                weighted_diagnostics=weighted_diagnostics,
                collect_wedges=collect_wedges,
                collect_ad_tapes=collect_ad_tapes,
                collect_wedge_prefixes=collect_wedge_prefixes,
            )
        # Trace emitted rays in one symbolic batch; Dr.Jit, not Python, owns the loop body.
        path_counts = wt.PathCounts()
        path_tape_store = (
            mc_reflection.PathTapeStore(
                samples_per_tx=samples_per_tx,
                max_bounces=int(effective["reflection_max_bounces"]),
            )
            if collect_ad_tapes
            else None
        )
        diff_state_store = None
        if collect_wedges:
            wedge_capacity = samples_per_tx
            if collect_wedge_prefixes:
                wedge_capacity *= int(effective["reflection_max_bounces"]) + 1
            diff_state_store = DiffractionHitStore(
                capacity=wedge_capacity,
                max_bounces=(
                    int(effective["reflection_max_bounces"])
                    if collect_wedge_prefixes
                    else 0
                ),
            )
        current_batch_size = int(samples_per_tx)
        ray_index = dr.arange(wt.UInt32, current_batch_size)
        ray_dir = Sampler.directions(
            samples_per_tx,
            ray_index=ray_index,
        )
        contribution_store = GridContributionStore(
            capacity=current_batch_size * (int(effective["reflection_max_bounces"]) + 1),
            grid=grid,
            weighted_diagnostics=weighted_diagnostics,
        )
        batch_los_hits, batch_reflection_hits, diff_state_store = (
            mc_reflection.Reflection.trace(
                scene=scene,
                grid=grid,
                tx_pos=tx_pos,
                ray_index=ray_index,
                ray_dir=ray_dir,
                config=config,
                solid_angle_per_ray=float(solid_angle_per_ray),
                cell_area=float(cell_area),
                max_bounces=int(effective["reflection_max_bounces"]),
                seed=seed,
                rr_depth=rr_depth,
                rr_prob=rr_prob,
                stop_threshold_linear=stop_threshold_linear,
                material_omega=material_omega,
                weighted_diagnostics=weighted_diagnostics,
                collect_wedges=collect_wedges,
                collect_wedge_prefixes=collect_wedge_prefixes,
                diff_state_store=diff_state_store,
                path_tape_store=path_tape_store,
                loop_mode=loop_mode,
                contribution_store=contribution_store,
            )
        )
        contribution_store.scatter_into(
            grid=grid,
            weighted_diagnostics=weighted_diagnostics,
        )
        path_counts.los += batch_los_hits
        path_counts.reflection += batch_reflection_hits
        return wt.ReflectionPhaseResult(
            path_counts=path_counts,
            path_tape_store=path_tape_store,
            diff_state_store=diff_state_store,
        )

    # Prepare diffraction wedge states and runtime backend from the state store.
    @staticmethod
    def prepare_wedges(
        diff_state_store,
        tx_pos,
        scene,
        config,
        loop_mode: str,
        resolved_accumulation_backend: str,
    ) -> tuple[object | None, object | None, dict[str, int], dict[str, object]]:
        # Materialize diffraction states and backend metadata from the reflection handoff store.
        stored_state_count = (
            0
            if diff_state_store is None
            else int(scalar(getattr(diff_state_store, "count", wt.UInt32(0))))
        )
        diffraction_states = None
        diffraction_edge_indices = None
        if diff_state_store is not None and stored_state_count > 0:
            tri_data = scene._triangle_runtime()
            edge_runtime = scene._selected_edge_runtime()
            max_surface_edge_count = 0 if tri_data is None else int(tri_data.get("surface_max_edge_count", 0))
            if tri_data is None or edge_runtime is None or max_surface_edge_count <= 0:
                diffraction_edge_indices = dr.zeros(wt.UInt32, 0)
            else:
                diffraction_edge_indices = DiffractionBuilderKernel.discover_edge_indices_from_hits(
                    tx_pos=tx_pos,
                    ray_directions=diff_state_store.ray_directions,
                    prim_index=diff_state_store.prim_index,
                    hit_p=diff_state_store.hit_p,
                    hit_n=diff_state_store.hit_n,
                    hit_geo_n=diff_state_store.hit_geo_n,
                    n_hits=stored_state_count,
                    triangle_edge_count=tri_data["surface_edge_size"],
                    triangle_edge_indices=tri_data["surface_edge_indices"],
                    max_triangle_edge_slots=max_surface_edge_count,
                    n_triangles=int(tri_data.get("n_triangles", 0)),
                    edge_runtime=edge_runtime,
                )
            diffraction_states = DiffractionStates.from_edge_indices(
                tx_pos=tx_pos,
                edge_idx=diffraction_edge_indices,
                scene=scene,
                config=config,
            )
            if diffraction_states is not None:
                diffraction_states.stored_count = int(dr.width(diffraction_states.edge_index))
        state_pool = {
            "total": 0 if diffraction_states is None else int(dr.width(diffraction_states.edge_index)),
            "kept": 0 if diffraction_states is None else int(dr.width(diffraction_states.edge_index)),
            "threshold_pruned": 0,
            "roulette_pruned": 0,
        }
        diffraction_runtime_backend = {
            "implementation": "depth0_direct_hit_store_plus_native_wedge_builder_plus_keller_cone_symbolic_loop_drjit_scatter_reduce",
            "cell_scatter_backend": (
                "drjit_scatter_reduce"
                if str(resolved_accumulation_backend) in {
                    "native_monte_carlo",
                    _RAYD_REFLECTION_ACCUMULATION_BACKEND,
                }
                else "disabled"
            ),
            "wedge_discovery_backend": "depth0_direct_hit_store_triangle_surface_edge_candidates_native_best_edge_builder",
            "state_sampler": "discovered_wedge_length_proportional_then_uniform_edge_position_then_keller_cone",
            "point_evaluation_backend": "sampled_edge_diffraction_field_to_plane_hits",
            "source_field_contract": "sionna_iso_v_implicit_basis",
            "loop_mode": str(loop_mode),
        }
        return diffraction_states, diffraction_edge_indices, state_pool, diffraction_runtime_backend

    # Run the diffraction phase: state preparation, sampling, and batched trace.
    @staticmethod
    def run_diffraction(
        *,
        collect_wedges: bool,
        diff_state_store,
        tx_pos,
        samples_per_tx: int,
        seed: int,
        config,
        scene,
        grid,
        diff_gain_scale,
        weighted_diagnostics: dict,
        collect_ad_tapes: bool,
        loop_mode: str,
        return_timing: bool,
        timing: wt.TraceTiming | None,
        path_counts: wt.PathCounts,
        resolved_accumulation_backend: str,
    ) -> wt.DiffractionPhaseResult:
        # Trace diffraction samples for the stored wedge states and report phase metadata.
        runtime_reuse = {
            "cache_mode": "disabled",
            "state_preparation_hits": 0,
            "state_preparation_misses": 0,
            "state_layout": "depth0_in_loop_state_store",
        }
        state_pool = {
            "total": 0,
            "kept": 0,
            "threshold_pruned": 0,
            "roulette_pruned": 0,
        }
        diffraction_runtime_backend = {
            "implementation": "disabled",
            "cell_scatter_backend": "disabled",
            "wedge_discovery_backend": "disabled",
            "state_sampler": "discovered_wedge_length_proportional_then_uniform_edge_position_then_keller_cone",
            "point_evaluation_backend": "sampled_edge_diffraction_field_to_plane_hits",
            "source_field_contract": "sionna_iso_v_implicit_basis",
        }
        diffraction_states = None
        diffraction_edge_indices = None
        diffraction_tape_store = None
        diff_length_weight = 0.0

        if collect_wedges:
            diffraction_states, diffraction_edge_indices, state_pool, diffraction_runtime_backend = (
                Basic.prepare_wedges(
                    diff_state_store,
                    tx_pos,
                    scene,
                    config,
                    loop_mode,
                    resolved_accumulation_backend,
                )
            )

        free_bytes = _capture_free_bytes(
            release_reclaimable_caches=collect_wedges,
        )
        n_diffraction_states = (
            0
            if diffraction_states is None
            else int(
                diffraction_states.stored_count
                if diffraction_states.stored_count is not None
                else dr.width(diffraction_states.edge_pos.x)
            )
        )
        batch_plan = _resolve_batch_plan(
            samples_per_tx=samples_per_tx,
            diffraction_state_count=n_diffraction_states,
            free_bytes=free_bytes,
        )

        if n_diffraction_states > 0:
            sampler = DiffractionEdgeSampler.from_line_length(
                dr.maximum(
                    diffraction_states.edge_line_max - diffraction_states.edge_line_min,
                    wt.Float(0.0),
                )
            )
            if sampler is not None:
                diff_length_weight = (
                    sampler.total_length_scalar / float(max(1, samples_per_tx))
                )
                diffraction_tape_store = (
                    DiffractionTapeStore(capacity=max(0, int(samples_per_tx)))
                    if collect_ad_tapes
                    else None
                )
                contribution_store = GridContributionStore(
                    capacity=max(0, int(samples_per_tx)),
                    grid=grid,
                    weighted_diagnostics=weighted_diagnostics,
                )
                plane_normal = Sampler.axis_unit_normal(str(grid.axis))
                diffraction_batch_size = max(1, int(batch_plan.diffraction_batch_size))
                if return_timing and timing is not None:
                    arrays.sync_thread()
                    t0 = time.perf_counter()
                path_counts.diffraction += Diffraction.trace_batches(
                    scene=scene,
                    grid=grid,
                    diffraction_states=diffraction_states,
                    sampler=sampler,
                    diffraction_batch_size=diffraction_batch_size,
                    diffraction_batch_count=int(batch_plan.diffraction_batch_count),
                    samples_per_tx=samples_per_tx,
                    seed=seed,
                    k=config.k,
                    wavelength=config.wavelength,
                    plane_normal=plane_normal,
                    diff_gain_scale=diff_gain_scale,
                    weighted_diagnostics=weighted_diagnostics,
                    diffraction_tape_store=diffraction_tape_store,
                    loop_mode=loop_mode,
                    contribution_store=contribution_store,
                )
                contribution_store.scatter_into(
                    grid=grid,
                    weighted_diagnostics=weighted_diagnostics,
                )
                if return_timing and timing is not None:
                    arrays.sync_thread()
                    timing.diffraction_seconds += time.perf_counter() - t0

        return wt.DiffractionPhaseResult(
            runtime_reuse=runtime_reuse,
            state_pool=state_pool,
            runtime_backend=diffraction_runtime_backend,
            edge_indices=diffraction_edge_indices,
            diff_length_weight=diff_length_weight,
            diffraction_tape_store=diffraction_tape_store,
            batch_plan=batch_plan,
        )

    # Resolve config scalars, grid, diagnostics, timing, and batch plan.
    @staticmethod
    def prepare_config(
        *,
        tx_pos,
        grid_spec,
        mc_config,
        scene,
        config,
        tx_power: float,
        accumulation_backend: str,
        return_timing: bool,
        loop_mode: str,
        effective: dict,
    ) -> IntegratorSetting:
        grid = Grid.from_spec(grid_spec, default_cell_size=config.cell_size)
        resolved_accumulation_backend = Basic.resolve_accumulation(accumulation_backend)

        integrator_options = mc_config.integrator_options
        samples_per_tx = int(integrator_options.samples_per_tx)
        reflection_n_rays = int(effective["reflection_n_rays"])
        seed = int(integrator_options.seed)
        tx_power = float(tx_power)
        rr_depth = None if integrator_options.rr_depth is None else int(integrator_options.rr_depth)
        if integrator_options.rr_prob is not None:
            rr_prob = float(integrator_options.rr_prob)
        else:
            rr_prob = 1.0 if rr_depth is None else 0.5
        stop_threshold = float(integrator_options.stop_threshold or 0.0)
        stop_threshold_linear = _stop_threshold_linear(stop_threshold)

        n_rx = int(grid.n_cells)
        weighted_diagnostics = _empty_radio_map(n_rx)
        cell_area = float(grid.cell_size[0] * grid.cell_size[1])
        diff_gain_scale = wt.Float(
            (float(config.wavelength) / (4.0 * math.pi)) ** 2 / cell_area
        )
        material_omega = material_angular_frequency(config.wavelength)
        timing = wt.TraceTiming() if return_timing else None

        ray_sampling_metadata = Sampler.metadata(
            axis=str(grid.axis),
            plane_position=float(grid.position),
            tx_pos=tx_pos,
        )
        solid_angle_per_ray = Sampler.solid_angle(ray_sampling_metadata, samples_per_tx)
        reflection_solid_angle_per_ray = Sampler.solid_angle(
            ray_sampling_metadata,
            reflection_n_rays,
        )
        reflection_runtime_backend = {
            "implementation": "cell_center_los_plus_tx_emitted_specular_reflection_single_symbolic_batch_drjit_scatter_reduce",
            "cell_scatter_backend": "drjit_scatter_reduce",
            "point_evaluation_backend": "direct_image_source_field",
            "ray_distribution": str(ray_sampling_metadata.get("selected_ray_sampling", "")),
            "source_field_contract": "sionna_iso_v_implicit_basis",
            "reflection_model": "materialized",
            "material_source": "scene",
            "loop_mode": str(loop_mode),
            "ray_budget": int(reflection_n_rays),
            "solid_angle_per_ray_sr": float(reflection_solid_angle_per_ray),
        }
        if resolved_accumulation_backend == _RAYD_REFLECTION_ACCUMULATION_BACKEND:
            reflection_runtime_backend = {
                **reflection_runtime_backend,
                "implementation": "rayd_trace_reflections_accumulating_complex_polarized_non_ad",
                "cell_scatter_backend": "rayd_optix_atomic_add",
                "point_evaluation_backend": "rayd_complex_image_source_field",
                "reflection_model": "rayd_complex_polarized_fresnel",
                "ad_contract": "explicit_non_ad_backend_raises_on_ad_inputs",
            }
        collect_wedges = int(effective["max_diffractions"]) > 0
        free_bytes = _capture_free_bytes(release_reclaimable_caches=collect_wedges)
        reflection_batch_plan = wt.BatchPlan(
            ray_batch_size=reflection_n_rays,
            ray_batch_count=1,
            ray_policy="single_symbolic_batch",
            diffraction_batch_size=0,
            diffraction_batch_count=0,
            diffraction_policy="disabled",
            free_cuda_bytes=int(free_bytes),
            scatter_safe_batch_cap=reflection_n_rays,
        )
        batch_plan = _resolve_batch_plan(
            samples_per_tx=samples_per_tx,
            diffraction_state_count=0,
            free_bytes=free_bytes,
        )
        return IntegratorSetting(
            grid=grid,
            resolved_accumulation_backend=resolved_accumulation_backend,
            samples_per_tx=samples_per_tx,
            reflection_n_rays=reflection_n_rays,
            seed=seed,
            tx_power=tx_power,
            rr_depth=rr_depth,
            rr_prob=rr_prob,
            stop_threshold=stop_threshold,
            stop_threshold_linear=stop_threshold_linear,
            weighted_diagnostics=weighted_diagnostics,
            cell_area=cell_area,
            diff_gain_scale=diff_gain_scale,
            material_omega=material_omega,
            timing=timing,
            ray_sampling_metadata=ray_sampling_metadata,
            solid_angle_per_ray=solid_angle_per_ray,
            reflection_solid_angle_per_ray=reflection_solid_angle_per_ray,
            reflection_runtime_backend=reflection_runtime_backend,
            collect_wedges=collect_wedges,
            batch_plan=batch_plan,
            reflection_batch_plan=reflection_batch_plan,
            loop_mode=str(loop_mode),
            config=config,
        )

    @staticmethod
    def primal(
        tx_pos,
        grid_spec: GridSpec,
        mc_config: Config,
        scene: Scene,
        config: ResolvedTraceConfig,
        solver_controls: Mapping[str, object],
        *,
        accumulation_backend: str = "auto",
        return_timing: bool = False,
        resolved_ad_mode: bool,
        ad_backend: str,
        loop_mode: str,
        tx_power: float = 1.0,
        noise_power: float | None = None,
        collect_ad_tapes: bool = False,
        apply_result_filtering: bool = True,
    ) -> ForwardPassState:
        # Run the full primal solve and package the phase outputs for result/AD assembly.
        effective = solver_controls["effective"]
        if int(effective["max_diffractions"]) > 1:
            raise RuntimeError(
                "sampling_mode='monte_carlo' currently supports only direct first-order diffraction."
            )

        setup = Basic.prepare_config(
            tx_pos=tx_pos, grid_spec=grid_spec, mc_config=mc_config, scene=scene,
            config=config, tx_power=float(tx_power), accumulation_backend=accumulation_backend,
            return_timing=return_timing, loop_mode=loop_mode, effective=effective,
        )
        grid, timing = setup.grid, setup.timing
        weighted_diagnostics = setup.weighted_diagnostics

        total_start = None
        if return_timing:
            arrays.sync_thread()
            total_start = time.perf_counter()

        # Phase 1: deterministic per-cell LOS
        if return_timing and timing is not None:
            arrays.sync_thread()
            t0 = time.perf_counter()
        los_result = LoS.trace(
            scene=scene,
            grid=grid,
            tx_pos=tx_pos,
            config=config,
            collect_ad_tapes=collect_ad_tapes,
        )
        weighted_diagnostics["incoherent"]["los"] = (
            weighted_diagnostics["incoherent"]["los"] + los_result.power
        )
        if return_timing and timing is not None:
            arrays.sync_thread()
            timing.los_seconds += time.perf_counter() - t0

        # Phase 2: reflection batches (multi-bounce specular reflection)
        refl_result = Basic.run_reflection(
            scene=scene, grid=grid, tx_pos=tx_pos, config=config,
            samples_per_tx=setup.reflection_n_rays, seed=setup.seed,
            ray_batch_size=int(setup.reflection_batch_plan.ray_batch_size),
            solid_angle_per_ray=float(setup.reflection_solid_angle_per_ray),
            cell_area=float(setup.cell_area), effective=effective,
            rr_depth=setup.rr_depth, rr_prob=setup.rr_prob,
            stop_threshold_linear=setup.stop_threshold_linear,
            material_omega=setup.material_omega,
            weighted_diagnostics=weighted_diagnostics,
            collect_wedges=setup.collect_wedges,
            collect_ad_tapes=collect_ad_tapes, loop_mode=loop_mode,
            resolved_accumulation_backend=setup.resolved_accumulation_backend,
        )
        refl_result.path_counts.los += los_result.path_count

        # Phase 3: diffraction phase
        diff_result = Basic.run_diffraction(
            collect_wedges=setup.collect_wedges,
            diff_state_store=refl_result.diff_state_store,
            tx_pos=tx_pos,
            samples_per_tx=setup.samples_per_tx, seed=setup.seed,
            config=config, scene=scene, grid=grid,
            diff_gain_scale=setup.diff_gain_scale,
            weighted_diagnostics=weighted_diagnostics,
            collect_ad_tapes=collect_ad_tapes, loop_mode=loop_mode,
            return_timing=return_timing, timing=timing,
            path_counts=refl_result.path_counts,
            resolved_accumulation_backend=setup.resolved_accumulation_backend,
        )

        # Phase 4: metadata assembly
        if (
            str(mc_config.tuning.shadow_boundary_mode) == "utd_power_smoothing"
            and int(effective["max_diffractions"]) > 0
        ):
            ShadowBoundary.accumulate_into_diagnostics(
                weighted_diagnostics=weighted_diagnostics,
                scene=scene,
                tx_pos=tx_pos,
                grid=grid,
                config=config,
                edge_indices=(
                    diff_result.edge_indices
                    if diff_result.edge_indices is not None
                    else dr.zeros(wt.UInt32, 0)
                ),
                ad_enabled=resolved_ad_mode,
            )
        finalize_weighted_diagnostics(
            weighted_diagnostics,
            shadow_boundary_mode=mc_config.tuning.shadow_boundary_mode,
            grid=grid,
            filtering=(
                mc_config.filtering
                if apply_result_filtering
                else None
            ),
        )
        noise_power = Basic.resolve_noise_power(scene, noise_power)
        metadata = build_metadata(
            grid=setup.grid,
            scene=scene,
            batch_plan=diff_result.batch_plan,
            ray_sampling_metadata=setup.ray_sampling_metadata,
            runtime_reuse=diff_result.runtime_reuse,
            solver_controls=solver_controls,
            resolved_ad_mode=resolved_ad_mode,
            ad_backend=ad_backend,
            mc_config=mc_config,
            samples_per_tx=setup.samples_per_tx,
            reflection_n_rays=setup.reflection_n_rays,
            reflection_batch_plan=setup.reflection_batch_plan,
            seed=setup.seed,
            accepted_hit_counts=refl_result.path_counts,
            weighted_diagnostics=weighted_diagnostics,
            resolved_accumulation_backend=setup.resolved_accumulation_backend,
            reflection_runtime_backend=setup.reflection_runtime_backend,
            diffraction_runtime_backend=diff_result.runtime_backend,
            state_pool=diff_result.state_pool,
            tx_power=setup.tx_power,
            rr_depth=setup.rr_depth,
            rr_prob=setup.rr_prob,
            stop_threshold=setup.stop_threshold,
            solid_angle_per_ray=setup.solid_angle_per_ray,
            reflection_solid_angle_per_ray=setup.reflection_solid_angle_per_ray,
            loop_mode=setup.loop_mode,
            integrator="basic",
            noise_power=noise_power,
            scalar_fn=scalar,
        )

        if return_timing and timing is not None and total_start is not None:
            arrays.sync_thread()
            timing.total_seconds = time.perf_counter() - total_start

        # Phase 5: AD tape finalization
        los_tape, reflection_tape, diffraction_tape = mc_ad.ADContext.finalize_tapes(
            collect_ad_tapes=collect_ad_tapes,
            max_bounces=int(effective["reflection_max_bounces"]),
            path_tape_store=refl_result.path_tape_store,
            diffraction_tape_store=diff_result.diffraction_tape_store,
            los_tape=los_result.tape,
        )

        return ForwardPassState(
            grid=grid,
            component_power={
                key: weighted_diagnostics["incoherent"][key]
                for key in (*mc_custom.BasicIntegratorAD.COMPONENT_NAMES, *EXTRA_COMPONENT_KEYS)
            },
            weighted_diagnostics=weighted_diagnostics,
            metadata=metadata,
            noise_power=float(noise_power),
            timing=timing,
            diffraction_edge_indices=diff_result.edge_indices,
            los_tape=los_tape,
            reflection_tape=reflection_tape,
            diffraction_tape=diffraction_tape,
            diff_length_weight=float(diff_result.diff_length_weight),
        )

    def integrate(
        self,
        tx_pos,
        grid_spec: GridSpec,
        mc_config: Config,
        scene: Scene,
        config: ResolvedTraceConfig,
        solver_controls: Mapping[str, object],
        *,
        accumulation_backend: str = "auto",
        return_timing: bool = False,
        tx_power: float = 1.0,
        noise_power: float | None = None,
    ):
        del self
        grad_sensitive_workload = mc_custom.grad_sensitive(
            config,
            tx_pos=tx_pos,
            scene=scene,
        )
        resolved_ad_mode = (
            bool(grad_sensitive_workload)
            if mc_config.integrator_options.ad is None
            else bool(mc_config.integrator_options.ad)
        )

        tx_power = float(tx_power)

        if resolved_ad_mode:
            if str(accumulation_backend).lower() == _RAYD_REFLECTION_ACCUMULATION_BACKEND:
                raise RuntimeError(
                    "accumulation_backend='rayd_reflection_accumulation' does not support AD. "
                    "Use accumulation_backend='native_monte_carlo' explicitly for the Monte Carlo AD tape path."
                )
            if not SparseCoeffKernel.available():
                raise RuntimeError(
                    "IntegratorOptions(ad=True) requires the Monte Carlo "
                    "native AD kernels. Rebuild the witwin.channel.montecarlo native extension."
                )
            return mc_custom.BasicIntegratorAD.integrate(
                tx_pos=tx_pos,
                grid_spec=grid_spec,
                mc_config=mc_config,
                scene=scene,
                config=config,
                solver_controls=solver_controls,
                accumulation_backend=accumulation_backend,
                return_timing=return_timing,
                grad_sensitive_workload=grad_sensitive_workload,
                trace_primal_fn=Basic.primal,
                tx_power=tx_power,
                noise_power=noise_power,
            )
        if grad_sensitive_workload:
            detached = mc_custom.BasicIntegratorAD.detached_workload(scene, tx_pos, config)
            tx_pos = detached["tx_pos"]
            scene = detached["scene"]
        primal_state = Basic.primal(
            tx_pos,
            grid_spec,
            mc_config,
            scene,
            config,
            solver_controls,
            accumulation_backend=accumulation_backend,
            return_timing=return_timing,
            resolved_ad_mode=False,
            ad_backend="disabled",
            loop_mode="symbolic",
            tx_power=tx_power,
            noise_power=noise_power,
            collect_ad_tapes=False,
        )
        path_gain = primal_state.weighted_diagnostics["incoherent"]["total"]
        rss = path_gain * tx_power
        return build_result(
            grid=primal_state.grid,
            tx_power=tx_power,
            weighted_diagnostics=primal_state.weighted_diagnostics,
            metadata=primal_state.metadata,
            path_gain=path_gain,
            rss=rss,
            sinr=_single_tx_sinr(rss, noise_power=float(primal_state.noise_power)),
            tx_pos=tx_pos,
            noise_power=float(primal_state.noise_power),
            timing=primal_state.timing,
            detach_tensors=True,
        )


__all__ = [
    "Basic",
    "ForwardPassState",
    "IntegratorSetting",
    "build_result",
    "finalize_weighted_diagnostics",
]
