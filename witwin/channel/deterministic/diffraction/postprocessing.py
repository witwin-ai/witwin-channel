"""Matched-ISB shadow-boundary correction: edge transition statistics and accumulation."""

from __future__ import annotations

import drjit as dr
from witwin.channel.deterministic import types as wt
from witwin.channel.core.kernels.shadow_boundary import ShadowBoundaryKernel

from ..kernels.radio_map_accumulate.native_impl import (
    matched_isb_completion,
    shadow_boundary_incident_statistics,
)
from witwin.channel.core.runtime import Rx, Tx, Wave
from witwin.channel.core.numerics.constants import EPS
from witwin.channel.core.numerics.arrays import (
    broadcast,
    complex_abs_sqr,
    complex_zero,
    gather,
)
from witwin.channel.core.physics.polarization import project_real_polarization_to_ray, vector_eval, vector_power, vector_zero
from witwin.channel.core.geometry.diffraction import wedge_exterior_mask
from witwin.channel.core.physics.shadow_boundary_policy import ShadowBoundaryBackendPolicy
from witwin.channel.core.physics.wave_math import unit_phase_neg_kd
from .forward import ForwardEval, f_utd
from .state import Geo


_MAX_DENSE_SHADOW_BOUNDARY_EDGE_RX_PAIRS = 8_000_000
_DET_SHADOW_BOUNDARY_POLICY = ShadowBoundaryBackendPolicy(
    small_backend="dense_native",
    pair_threshold=_MAX_DENSE_SHADOW_BOUNDARY_EDGE_RX_PAIRS,
    too_large_message=(
        "Deterministic matched-ISB shadow-boundary correction would use the "
        "dense all-edge x all-receiver path "
        "({n_pairs} pairs). Munich-scale correction requires the candidate-pruned backend."
    ),
    no_native_message=(
        "Deterministic matched-ISB shadow-boundary correction requires the "
        "native_candidate candidate-pruned backend for this workload, but the "
        "Monte Carlo native shadow-boundary kernel is unavailable. Rebuild "
        "the package with `pip install . --no-deps`, request "
        "shadow_boundary_backend='dense_native' for small reference grids, or "
        "disable shadow_boundary_correction for this run."
    ),
    ad_unsupported_message=(
        "shadow_boundary_backend='native_candidate' is forward-only and does not "
        "support AD. Use shadow_boundary_backend='dense_native' for small AD "
        "reference grids or disable shadow_boundary_correction."
    ),
)


class _CandidateShadowBoundaryStates:
    def __init__(
        self,
        *,
        edge_pos,
        edge_dir,
        n0,
        n_face_n,
        wedge_n,
        edge_line_min,
        edge_line_max,
        source_pos,
    ) -> None:
        self.edge_pos = edge_pos
        self.edge_dir = edge_dir
        self.n0 = n0
        self.n_face_n = n_face_n
        self.wedge_n = wedge_n
        self.edge_line_min = edge_line_min
        self.edge_line_max = edge_line_max
        self.source_pos = source_pos


def _array_grad_enabled(value) -> bool:
    if value is None:
        return False
    try:
        if dr.grad_enabled(value):
            return True
    except Exception:
        pass
    for axis in ("x", "y", "z", "real", "imag"):
        try:
            component = getattr(value, axis, None)
        except Exception:
            continue
        if component is None:
            continue
        try:
            if dr.grad_enabled(component):
                return True
        except Exception:
            continue
    return False


def resolve_shadow_boundary_statistics_backend(
    *,
    n_edges: int,
    n_rx: int,
    requested_backend: str = "auto",
    native_candidate_available: bool | None = None,
    ad_enabled: bool = False,
) -> str:
    return _DET_SHADOW_BOUNDARY_POLICY.resolve(
        requested=str(requested_backend),
        n_pairs=int(n_edges) * int(n_rx),
        ad_enabled=bool(ad_enabled),
        native_available=native_candidate_available,
    )


def validate_dense_shadow_boundary_workload(*, n_edges: int, n_rx: int) -> None:
    _DET_SHADOW_BOUNDARY_POLICY.validate_small_workload(int(n_edges) * int(n_rx))


def trace_shadow_boundary_correction(*, spec, grid, scene, runtime, components):
    if not bool(spec.shadow_boundary_correction):
        return None
    correction_runtime = runtime.with_rx(grid.cell_centers)
    return accumulate_matched_isb_shadow_boundary_correction(
        rx=correction_runtime.rx,
        grid=grid,
        scene=scene,
        spec=spec,
        tx=correction_runtime.tx,
        wave=correction_runtime.wave,
        los_vector_coherent=components["vector_coherent"]["los"],
        raw_transition_vector={
            axis: components["vector_coherent"]["los"][axis]
            + components["vector_coherent"]["diffraction"][axis]
            for axis in ("x", "y", "z")
        },
        reflection_vector_coherent=components["vector_coherent"]["reflection"],
    )


def _vector_add(lhs, rhs):
    return {axis: lhs[axis] + rhs[axis] for axis in ("x", "y", "z")}


def _vector_scale_complex(vector, scale):
    return {axis: vector[axis] * scale for axis in ("x", "y", "z")}


def _matched_rsb_completion_vector(
    *,
    reflection_vector_coherent,
    reflection_weight,
    reflection_response,
):
    hard_visibility = dr.select(
        vector_power(reflection_vector_coherent) > wt.Float(1.0e-14),
        wt.Float(1.0),
        wt.Float(0.0),
    )
    side_sign = dr.select(hard_visibility > wt.Float(0.0), wt.Float(1.0), wt.Float(-1.0))
    smooth_coeff = wt.Complex2f(
        wt.Float(0.5) * (wt.Float(1.0) + side_sign * reflection_response.real),
        wt.Float(0.5) * side_sign * reflection_response.imag,
    )
    delta_coeff = wt.Complex2f(
        reflection_weight * (smooth_coeff.real - hard_visibility),
        reflection_weight * smooth_coeff.imag,
    )
    return _vector_scale_complex(reflection_vector_coherent, delta_coeff)


def _utd_transition_weight(x):
    transition_mag = dr.sqrt(complex_abs_sqr(f_utd(x)))
    return dr.maximum(
        wt.Float(0.0),
        wt.Float(1.0) - dr.minimum(transition_mag, wt.Float(1.0)),
    )


def _shadow_boundary_transition_support_mask(batch_states, batch_rx):
    width = dr.width(batch_rx.x)
    support = dr.full(wt.Bool, True, width)
    nn = batch_states.get("n_face_n")
    if nn is not None:
        source_exterior = wedge_exterior_mask(
            batch_states["source_pos"] - batch_states["edge_pos"],
            batch_states["edge_dir"],
            batch_states["n0"],
            nn,
        )
        target_exterior = wedge_exterior_mask(
            batch_rx - batch_states["edge_pos"],
            batch_states["edge_dir"],
            batch_states["n0"],
            nn,
        )
        support = support & source_exterior & target_exterior
    source_visible = batch_states.get("source_visible")
    if source_visible is not None:
        source_visible_b = (
            source_visible
            if dr.width(source_visible) == width
            else dr.repeat(source_visible, width)
        )
        support = support & source_visible_b
    return support


def _shadow_boundary_transition_responses(batch_states, batch_rx, *, wave: Wave):
    width = dr.width(batch_rx.x)
    edge_dir = batch_states["edge_dir"]
    source_to_edge = batch_states["edge_pos"] - batch_states["source_pos"]
    source_to_edge_proj = source_to_edge - dr.dot(source_to_edge, edge_dir) * edge_dir
    s_prime_proj = dr.norm(source_to_edge_proj) + EPS
    to_hat = dr.normalize(dr.cross(batch_states["n0"], edge_dir))
    ki_proj = source_to_edge_proj / s_prime_proj
    phi_prime = dr.pi - dr.safe_acos(dr.clip(-dr.dot(ki_proj, to_hat), -1.0, 1.0))
    phi_prime = phi_prime * (-dr.sign(-dr.dot(ki_proj, batch_states["n0"])))
    phi_prime = phi_prime + dr.pi

    edge_to_target = batch_rx - batch_states["edge_pos"]
    edge_to_target_proj = edge_to_target - dr.dot(edge_to_target, edge_dir) * edge_dir
    s_proj = dr.norm(edge_to_target_proj) + EPS
    ko_proj = edge_to_target_proj / s_proj
    phi = dr.pi - dr.safe_acos(dr.clip(dr.dot(ko_proj, to_hat), -1.0, 1.0))
    phi = phi * (-dr.sign(dr.dot(ko_proj, batch_states["n0"])))
    phi = phi + dr.pi

    s = dr.norm(edge_to_target) + EPS
    s_prime = dr.norm(source_to_edge) + EPS
    wedge_n = batch_states["wedge_n"]
    kL = wave.k * s * s_prime * dr.rcp(s + s_prime)
    inc_a0, inc_a1 = ForwardEval.a_pm(phi - phi_prime, wedge_n)
    incident_transition = f_utd(kL * dr.minimum(inc_a0, inc_a1))
    incident_weight = _utd_transition_weight(kL * dr.minimum(inc_a0, inc_a1))
    ref_a0, ref_a1 = ForwardEval.a_pm(phi + phi_prime, wedge_n)
    reflection_transition = f_utd(kL * dr.minimum(ref_a0, ref_a1))
    reflection_weight = _utd_transition_weight(kL * dr.minimum(ref_a0, ref_a1))
    Geo.state_line_bounds(
        batch_states,
        context="_shadow_boundary_transition_responses",
    )
    finite_wedge_factor = ForwardEval.truncation_factor(
        batch_states,
        {"edge_hat": edge_dir, "s_prime_proj": s_prime_proj, "s_proj": s_proj},
        batch_rx,
        wave.k,
        width=width,
    )
    finite_wedge_scale = dr.minimum(
        dr.sqrt(complex_abs_sqr(finite_wedge_factor)),
        wt.Float(1.0),
    )
    incident_transition = finite_wedge_factor * incident_transition
    incident_weight = incident_weight * finite_wedge_scale
    reflection_transition = finite_wedge_factor * reflection_transition
    reflection_weight = reflection_weight * finite_wedge_scale
    support_mask = _shadow_boundary_transition_support_mask(batch_states, batch_rx)
    zero = complex_zero(width)
    incident_transition = dr.select(support_mask, incident_transition, zero)
    incident_weight = dr.select(support_mask, incident_weight, wt.Float(0.0))
    reflection_transition = dr.select(support_mask, reflection_transition, zero)
    reflection_weight = dr.select(support_mask, reflection_weight, wt.Float(0.0))
    return incident_transition, reflection_transition, incident_weight, reflection_weight


def _accumulate_shadow_boundary_incident_statistics(*, rx: Rx, scene, tx: Tx, wave: Wave):
    rx_pos = rx.positions
    n_rx = int(dr.width(rx_pos.x))
    edge_runtime = scene._selected_edge_runtime()
    n_edges = 0 if edge_runtime is None else int(edge_runtime.get("n_edges", 0))
    zero_float = dr.zeros(wt.Float, n_rx)
    if n_rx <= 0 or n_edges <= 0:
        return {
            "n_edges": int(n_edges),
            "sum_incident_weight": zero_float,
            "max_incident_weight": zero_float,
            "weighted_incident_response_real": zero_float,
            "weighted_incident_response_imag": zero_float,
            "sum_reflection_weight": zero_float,
            "max_reflection_weight": zero_float,
            "weighted_reflection_response_real": zero_float,
            "weighted_reflection_response_imag": zero_float,
        }

    explicit_line_min, explicit_line_max = Geo.data_line_bounds(
        edge_runtime,
        context="_accumulate_shadow_boundary_incident_statistics",
    )
    adjacent_face0 = edge_runtime.get("adjacent_face0")
    adjacent_face1 = edge_runtime.get("adjacent_face1")
    adjacent_surface_group0 = edge_runtime.get("adjacent_surface_group0")
    adjacent_surface_group1 = edge_runtime.get("adjacent_surface_group1")
    ignore_prim_idx = None
    ignore_surface_group_idx = None
    if adjacent_surface_group0 is not None or adjacent_surface_group1 is not None:
        ignore_surface_group_idx = tuple(
            v for v in (adjacent_surface_group0, adjacent_surface_group1) if v is not None
        )
    if adjacent_face0 is not None or adjacent_face1 is not None:
        ignore_prim_idx = tuple(
            v for v in (adjacent_face0, adjacent_face1) if v is not None
        )
    source_visible = scene.segment_visible(
        tx.position,
        edge_runtime["pos"],
        ignore_prim_idx=ignore_prim_idx,
        ignore_surface_group_idx=ignore_surface_group_idx,
    )
    stats = shadow_boundary_incident_statistics(
        tx_pos=tx.position,
        rx_pos=rx_pos,
        edge_pos=edge_runtime["pos"],
        edge_dir=edge_runtime["edge_dir"],
        n0=edge_runtime["n0"],
        n_face_n=edge_runtime["n_face_n"],
        wedge_n=edge_runtime["wedge_n"],
        edge_line_min=explicit_line_min,
        edge_line_max=explicit_line_max,
        source_visible=source_visible,
        k=wave.k_scalar,
    )
    stats["n_edges"] = int(n_edges)
    stats["_metadata"] = {
        "backend": "dense_native",
        "n_edges": int(n_edges),
        "n_rx": int(n_rx),
        "full_pair_count": int(n_edges) * int(n_rx),
        "candidate_pairs": int(n_edges) * int(n_rx),
        "candidate_ratio": 1.0,
        "weight_aggregation": "max_weight_weighted_response_average",
        "incident_support": "all_receiver_cells_source_visible_edges",
        "transition_terms": "incident_and_reflection_shadow_boundaries",
        "statistics_implementation": "drjit_reference_selected_stationary",
    }
    return stats


def _source_visibility_for_edges(*, edge_runtime, scene, tx: Tx):
    adjacent_face0 = edge_runtime.get("adjacent_face0")
    adjacent_face1 = edge_runtime.get("adjacent_face1")
    adjacent_surface_group0 = edge_runtime.get("adjacent_surface_group0")
    adjacent_surface_group1 = edge_runtime.get("adjacent_surface_group1")
    ignore_prim_idx = None
    ignore_surface_group_idx = None
    if adjacent_surface_group0 is not None or adjacent_surface_group1 is not None:
        ignore_surface_group_idx = tuple(
            v for v in (adjacent_surface_group0, adjacent_surface_group1) if v is not None
        )
    if adjacent_face0 is not None or adjacent_face1 is not None:
        ignore_prim_idx = tuple(
            v for v in (adjacent_face0, adjacent_face1) if v is not None
        )
    return scene.segment_visible(
        tx.position,
        edge_runtime["pos"],
        ignore_prim_idx=ignore_prim_idx,
        ignore_surface_group_idx=ignore_surface_group_idx,
    )


def _zero_shadow_boundary_statistics(*, n_rx: int, n_edges: int, backend: str):
    zero_float = dr.zeros(wt.Float, n_rx)
    return {
        "n_edges": int(n_edges),
        "sum_incident_weight": zero_float,
        "max_incident_weight": zero_float,
        "weighted_incident_response_real": zero_float,
        "weighted_incident_response_imag": zero_float,
        "sum_reflection_weight": zero_float,
        "max_reflection_weight": zero_float,
        "weighted_reflection_response_real": zero_float,
        "weighted_reflection_response_imag": zero_float,
        "_metadata": {
            "backend": str(backend),
            "n_edges": int(n_edges),
            "n_rx": int(n_rx),
            "full_pair_count": int(n_edges) * int(n_rx),
            "source_visible_edges": 0,
            "candidate_pairs": 0,
            "candidate_ratio": 0.0,
        },
    }


def _gather_or_default_int(edge_runtime, key: str, slots, *, width: int):
    values = edge_runtime.get(key)
    if values is None:
        return dr.full(wt.Int32, -1, width)
    return dr.gather(wt.Int32, values, slots)


def _candidate_pruned_shadow_boundary_incident_statistics(
    *,
    rx: Rx,
    grid,
    edge_runtime,
    source_visible,
    tx: Tx,
    wave: Wave,
    spec,
):
    rx_pos = rx.positions
    n_rx = int(dr.width(rx_pos.x))
    n_edges = int(edge_runtime.get("n_edges", 0))
    if n_rx <= 0 or n_edges <= 0:
        return _zero_shadow_boundary_statistics(
            n_rx=n_rx,
            n_edges=n_edges,
            backend="native_candidate",
        )
    dr.eval(source_visible)
    visible_slots = dr.compress(source_visible)
    source_visible_edges = int(dr.width(visible_slots))
    if source_visible_edges <= 0:
        return _zero_shadow_boundary_statistics(
            n_rx=n_rx,
            n_edges=n_edges,
            backend="native_candidate",
        )

    edge_line_min, edge_line_max = Geo.data_line_bounds(
        edge_runtime,
        context="_candidate_pruned_shadow_boundary_incident_statistics",
    )
    states = _CandidateShadowBoundaryStates(
        edge_pos=gather(edge_runtime["pos"], visible_slots),
        edge_dir=gather(edge_runtime["edge_dir"], visible_slots),
        n0=gather(edge_runtime["n0"], visible_slots),
        n_face_n=gather(edge_runtime["n_face_n"], visible_slots),
        wedge_n=dr.gather(wt.Float, edge_runtime["wedge_n"], visible_slots),
        edge_line_min=dr.gather(wt.Float, edge_line_min, visible_slots),
        edge_line_max=dr.gather(wt.Float, edge_line_max, visible_slots),
        source_pos=broadcast(tx.position, source_visible_edges),
    )
    direct_los_visible = dr.full(wt.UInt32, 1, n_rx)
    direct_blocker_group = dr.full(wt.Int32, -1, n_rx)
    adjacent_group0 = _gather_or_default_int(
        edge_runtime,
        "adjacent_surface_group0",
        visible_slots,
        width=source_visible_edges,
    )
    adjacent_group1 = _gather_or_default_int(
        edge_runtime,
        "adjacent_surface_group1",
        visible_slots,
        width=source_visible_edges,
    )
    weights = ShadowBoundaryKernel.accumulate(
        states=states,
        grid=grid,
        k=float(wave.k_scalar),
        wavelength=float(wave.wavelength_scalar),
        tile_shape=tuple(int(v) for v in spec.shadow_boundary_tile_shape),
        band_width_wavelengths=float(spec.shadow_boundary_band_width_wavelengths),
        max_candidate_factor=float(spec.shadow_boundary_max_candidate_factor),
        direct_los_visible=direct_los_visible,
        direct_blocker_group=direct_blocker_group,
        edge_adjacent_group0=adjacent_group0,
        edge_adjacent_group1=adjacent_group1,
    )
    incident_weight = wt.Float(weights["incident_shadow_boundary_weight"])
    incident_response_real = wt.Float(weights["incident_transition_response_real"])
    incident_response_imag = wt.Float(weights["incident_transition_response_imag"])
    reflection_weight = wt.Float(weights["reflection_shadow_boundary_weight"])
    reflection_response_real = wt.Float(weights["reflection_transition_response_real"])
    reflection_response_imag = wt.Float(weights["reflection_transition_response_imag"])
    metadata = dict(weights.get("_metadata", {}))
    metadata.update(
        {
            "backend": "native_candidate",
            "kernel": "shadow_boundary_candidate_accumulate",
            "n_edges": int(n_edges),
            "n_rx": int(n_rx),
            "source_total_edges": int(n_edges),
            "source_visible_edges": int(source_visible_edges),
            "full_pair_count": int(n_edges) * int(n_rx),
            "candidate_state_count": int(source_visible_edges),
            "dense_pair_limit": int(_MAX_DENSE_SHADOW_BOUNDARY_EDGE_RX_PAIRS),
            "incident_support": "all_receiver_cells_source_visible_edges",
            "weight_aggregation": "max_weight_weighted_response_average",
            "transition_terms": "incident_and_reflection_shadow_boundaries",
        }
    )
    return {
        "n_edges": int(n_edges),
        "sum_incident_weight": incident_weight,
        "max_incident_weight": incident_weight,
        "weighted_incident_response_real": incident_weight * incident_response_real,
        "weighted_incident_response_imag": incident_weight * incident_response_imag,
        "sum_reflection_weight": reflection_weight,
        "max_reflection_weight": reflection_weight,
        "weighted_reflection_response_real": reflection_weight * reflection_response_real,
        "weighted_reflection_response_imag": reflection_weight * reflection_response_imag,
        "_metadata": metadata,
    }


def _accumulate_shadow_boundary_statistics(
    *,
    rx: Rx,
    grid,
    scene,
    spec,
    tx: Tx,
    wave: Wave,
    hard_visibility,
):
    rx_pos = rx.positions
    n_rx = int(dr.width(rx_pos.x))
    edge_runtime = scene._selected_edge_runtime()
    n_edges = 0 if edge_runtime is None else int(edge_runtime.get("n_edges", 0))
    if n_rx <= 0 or n_edges <= 0:
        return _zero_shadow_boundary_statistics(
            n_rx=n_rx,
            n_edges=n_edges,
            backend="none",
        )
    source_visible = _source_visibility_for_edges(
        edge_runtime=edge_runtime,
        scene=scene,
        tx=tx,
    )
    ad_enabled = any(
        _array_grad_enabled(value)
        for value in (rx_pos, tx.position, edge_runtime.get("pos"), hard_visibility)
    )
    backend = resolve_shadow_boundary_statistics_backend(
        n_edges=n_edges,
        n_rx=n_rx,
        requested_backend=str(spec.shadow_boundary_backend),
        ad_enabled=ad_enabled,
    )
    if backend == "dense_native":
        stats = _accumulate_shadow_boundary_incident_statistics(
            rx=rx,
            scene=scene,
            tx=tx,
            wave=wave,
        )
        stats["_metadata"].update(
            {
                "requested_backend": str(spec.shadow_boundary_backend),
                "resolved_backend": backend,
                "dense_pair_limit": int(_MAX_DENSE_SHADOW_BOUNDARY_EDGE_RX_PAIRS),
            }
        )
        return stats
    stats = _candidate_pruned_shadow_boundary_incident_statistics(
        rx=rx,
        grid=grid,
        edge_runtime=edge_runtime,
        source_visible=source_visible,
        tx=tx,
        wave=wave,
        spec=spec,
    )
    stats["_metadata"].update(
        {
            "requested_backend": str(spec.shadow_boundary_backend),
            "resolved_backend": backend,
        }
    )
    return stats


def accumulate_matched_isb_shadow_boundary_correction(
    *,
    rx: Rx,
    grid,
    scene,
    spec,
    tx: Tx,
    wave: Wave,
    los_vector_coherent,
    raw_transition_vector,
    reflection_vector_coherent=None,
):
    rx_pos = rx.positions
    n_rx = int(dr.width(rx_pos.x))
    zero_float = dr.zeros(wt.Float, n_rx)
    zero_complex = complex_zero(n_rx)
    zero_vector = vector_zero(n_rx)
    empty_payload = {
        "coherent": zero_complex,
        "vector_coherent": zero_vector,
        "power": zero_float,
        "metadata": {
            "backend": "none",
            "resolved_backend": "none",
            "n_edges": int(0 if scene is None else (scene._selected_edge_runtime() or {}).get("n_edges", 0)),
            "n_rx": int(n_rx),
        },
    }
    if n_rx <= 0 or scene is None:
        dr.eval(
            zero_complex.real, zero_complex.imag,
            zero_vector["x"].real, zero_vector["x"].imag,
            zero_vector["y"].real, zero_vector["y"].imag,
            zero_vector["z"].real, zero_vector["z"].imag,
            zero_float,
        )
        return empty_payload
    if los_vector_coherent is None or raw_transition_vector is None:
        raise RuntimeError(
            "matched-ISB shadow-boundary correction requires both "
            "los_vector_coherent and raw_transition_vector."
        )

    edge_runtime = scene._selected_edge_runtime()
    n_edges = 0 if edge_runtime is None else int(edge_runtime.get("n_edges", 0))
    if n_edges <= 0:
        dr.eval(
            zero_complex.real, zero_complex.imag,
            zero_vector["x"].real, zero_vector["x"].imag,
            zero_vector["y"].real, zero_vector["y"].imag,
            zero_vector["z"].real, zero_vector["z"].imag,
            zero_float,
        )
        return empty_payload
    ray_dir = rx_pos - tx.position
    distance = dr.norm(ray_dir) + EPS
    continued_direct = (
        wave.wavelength / (wt.Float(4.0) * dr.pi * distance)
    ) * unit_phase_neg_kd(wave.k, distance)
    tx_pol_dir = project_real_polarization_to_ray(tx.polarization, ray_dir)
    active_rx_polarization = rx.effective_polarization(tx)
    hard_visibility = dr.select(
        vector_power(los_vector_coherent) > wt.Float(1.0e-14),
        wt.Float(1.0),
        wt.Float(0.0),
    )
    shadow_mask = hard_visibility <= wt.Float(0.0)
    interior_mask = scene.point_inside_closed_mesh(
        rx_pos, robust=True, active=shadow_mask,
    )
    stats = _accumulate_shadow_boundary_statistics(
        rx=rx,
        grid=grid,
        scene=scene,
        spec=spec,
        tx=tx,
        wave=wave,
        hard_visibility=hard_visibility,
    )
    scene_sum_incident_weight = stats["sum_incident_weight"]
    scene_max_incident_weight = stats["max_incident_weight"]
    scene_weighted_real = stats["weighted_incident_response_real"]
    scene_weighted_imag = stats["weighted_incident_response_imag"]
    scene_sum_reflection_weight = stats["sum_reflection_weight"]
    scene_max_reflection_weight = stats["max_reflection_weight"]
    scene_weighted_reflection_real = stats["weighted_reflection_response_real"]
    scene_weighted_reflection_imag = stats["weighted_reflection_response_imag"]

    safe_sum_weight = dr.maximum(scene_sum_incident_weight, wt.Float(1.0e-6))
    scene_incident_response = wt.Complex2f(
        dr.select(
            scene_sum_incident_weight > wt.Float(1.0e-6),
            scene_weighted_real / safe_sum_weight,
            wt.Float(1.0),
        ),
        dr.select(
            scene_sum_incident_weight > wt.Float(1.0e-6),
            scene_weighted_imag / safe_sum_weight,
            wt.Float(0.0),
        ),
    )
    aggregate_incident_weight = dr.minimum(
        dr.maximum(scene_max_incident_weight, wt.Float(0.0)),
        wt.Float(1.0),
    )
    if reflection_vector_coherent is not None:
        safe_reflection_sum_weight = dr.maximum(scene_sum_reflection_weight, wt.Float(1.0e-6))
        scene_reflection_response = wt.Complex2f(
            dr.select(
                scene_sum_reflection_weight > wt.Float(1.0e-6),
                scene_weighted_reflection_real / safe_reflection_sum_weight,
                wt.Float(1.0),
            ),
            dr.select(
                scene_sum_reflection_weight > wt.Float(1.0e-6),
                scene_weighted_reflection_imag / safe_reflection_sum_weight,
                wt.Float(0.0),
            ),
        )
        aggregate_reflection_weight = dr.minimum(
            dr.maximum(scene_max_reflection_weight, wt.Float(0.0)),
            wt.Float(1.0),
        )
    else:
        scene_reflection_response = wt.Complex2f(wt.Float(1.0), wt.Float(0.0))
        aggregate_reflection_weight = wt.Float(0.0)

    rx_pol_dir = project_real_polarization_to_ray(active_rx_polarization, ray_dir)
    correction_payload = matched_isb_completion(
        continued_direct=continued_direct,
        tx_basis=tx_pol_dir,
        rx_basis=rx_pol_dir,
        hard_visibility=hard_visibility,
        interior_mask=interior_mask,
        incident_weight=aggregate_incident_weight,
        incident_response=scene_incident_response,
        raw_transition_vector=raw_transition_vector,
    )
    isb_vector = correction_payload["vector_coherent"]
    rsb_vector = _matched_rsb_completion_vector(
        reflection_vector_coherent=reflection_vector_coherent or zero_vector,
        reflection_weight=aggregate_reflection_weight,
        reflection_response=scene_reflection_response,
    )
    correction_vector = _vector_add(isb_vector, rsb_vector)
    return {
        "coherent": correction_payload["coherent"],
        "vector_coherent": vector_eval(correction_vector),
        "power": vector_power(correction_vector),
        "metadata": stats.get("_metadata", {}),
    }


__all__ = [
    "accumulate_matched_isb_shadow_boundary_correction",
    "resolve_shadow_boundary_statistics_backend",
    "trace_shadow_boundary_correction",
    "validate_dense_shadow_boundary_workload",
]
