"""Trace metadata assembly for Monte Carlo radiomap integrators."""

from __future__ import annotations

from typing import Mapping

import drjit as dr

from witwin.channel.montecarlo import types as wt
from ..config import Config
from ..filtering import filtering_metadata
from witwin.channel.core.grid import Grid
from ..trace import ad_support as mc_ad


COMPONENT_NAMES = ("los", "reflection", "diffraction")


def build_metadata(
    *,
    grid: Grid,
    scene=None,
    batch_plan: wt.BatchPlan,
    ray_sampling_metadata: Mapping[str, object],
    runtime_reuse: Mapping[str, object],
    solver_controls: Mapping[str, object],
    resolved_ad_mode: bool,
    ad_backend: str,
    mc_config: Config,
    samples_per_tx: int,
    reflection_n_rays: int | None = None,
    reflection_batch_plan: wt.BatchPlan | None = None,
    seed: int,
    accepted_hit_counts: wt.PathCounts,
    weighted_diagnostics: dict[str, object],
    resolved_accumulation_backend: str,
    reflection_runtime_backend: Mapping[str, object],
    diffraction_runtime_backend: Mapping[str, object],
    state_pool: Mapping[str, int],
    tx_power: float,
    rr_depth: int | None,
    rr_prob: float,
    stop_threshold: float,
    solid_angle_per_ray: float,
    reflection_solid_angle_per_ray: float | None = None,
    loop_mode: str,
    integrator: str,
    noise_power: float,
    scalar_fn,
) -> dict[str, object]:
    """Assemble the trace metadata dict from intermediate solver state."""
    counts = {
        "los": int(scalar_fn(accepted_hit_counts.los)),
        "reflection": int(scalar_fn(accepted_hit_counts.reflection)),
        "diffraction": int(scalar_fn(accepted_hit_counts.diffraction)),
    }
    inc = weighted_diagnostics["incoherent"]
    reflection_ray_count = (
        int(samples_per_tx)
        if reflection_n_rays is None
        else int(reflection_n_rays)
    )
    ray_batch_plan = batch_plan if reflection_batch_plan is None else reflection_batch_plan
    ray_solid_angle = (
        float(solid_angle_per_ray)
        if reflection_solid_angle_per_ray is None
        else float(reflection_solid_angle_per_ray)
    )
    tuning = mc_config.tuning
    integrator_options = mc_config.integrator_options
    shadow_boundary_mode = str(tuning.shadow_boundary_mode)
    shadow_runtime = dict(weighted_diagnostics.get("shadow_boundary_runtime") or {})
    runtime_tile_shape = shadow_runtime.get(
        "tile_shape",
        tuning.shadow_boundary_tile_shape,
    )
    try:
        shadow_tile_shape = [int(runtime_tile_shape[0]), int(runtime_tile_shape[1])]
    except (TypeError, IndexError, ValueError):
        shadow_tile_shape = [int(v) for v in tuning.shadow_boundary_tile_shape]
    for component in COMPONENT_NAMES:
        if counts[component] <= 0:
            counts[component] = int(scalar_fn(dr.count(inc[component] > wt.Float(0.0))))
    path_counts = {**counts, "total": int(sum(counts.values()))}
    cell_accumulation_mode = (
        "rayd_optix_atomic_add"
        if str(resolved_accumulation_backend) == "rayd_reflection_accumulation"
        else "drjit_scatter_reduce"
    )

    edge_policy = getattr(scene, "_edge_policy", None)
    return {
        "sampling_mode": "monte_carlo",
        "receiver_sampling": {
            "sampling_mode": "monte_carlo",
            "surface_mode": "axis_aligned",
            "axis": grid.axis,
            "position": grid.position,
            "bounds": grid.bounds,
            "center": grid.center,
            "orientation": (0.0, 0.0, 0.0),
            "size": grid.size,
            "tangential_axes": grid.tangential_axes,
            "grid_shape": grid.grid_shape,
            "cell_size": grid.cell_size,
            "cell_centered": True,
            "quadrature_mode": None,
            "samples_per_cell": None,
            "sample_offsets_local": (),
            "samples_per_tx": samples_per_tx,
            "reflection_n_rays": reflection_ray_count,
            "seed": seed,
            "strategy": "cell_center_los_plus_tx_emitted_non_los",
        },
        "ray_sampling": {
            "samples_per_tx": reflection_ray_count,
            "diffraction_samples_per_tx": samples_per_tx,
            "seed": seed,
            "batch_size": int(ray_batch_plan.ray_batch_size),
            "batch_count": int(ray_batch_plan.ray_batch_count),
            "batch_policy": str(ray_batch_plan.ray_policy),
            "diffraction_batch_size": int(batch_plan.diffraction_batch_size),
            "diffraction_batch_count": int(batch_plan.diffraction_batch_count),
            "diffraction_batch_policy": str(batch_plan.diffraction_policy),
            "scatter_safe_batch_cap": int(ray_batch_plan.scatter_safe_batch_cap),
            "distribution": str(ray_sampling_metadata.get("selected_ray_sampling", "")),
            "sequence": str(ray_sampling_metadata.get("sampling_sequence", "")),
            "solid_angle_per_ray_sr": ray_solid_angle,
        },
        "metric_contract": {
            "path_gain": (
                "cell_center_los_plus_tx_emitted_non_los_monte_carlo_matched_isotropic_"
                "incoherent_power_with_signed_shadow_boundary_correction"
                if shadow_boundary_mode == "utd_power_smoothing"
                else "cell_center_los_plus_tx_emitted_non_los_monte_carlo_matched_isotropic_incoherent_power"
            ),
            "incoherent_total": (
                "total=raw_total+shadow_boundary_correction; raw_total=los+reflection+"
                "diffraction; shadow_boundary_correction is signed and clamped so total "
                "remains non-negative"
                if shadow_boundary_mode == "utd_power_smoothing"
                else "total=raw_total=los+reflection+diffraction"
            ),
            "raw_component_powers": (
                "incoherent['los'], incoherent['reflection'], and "
                "incoherent['diffraction'] remain non-negative raw powers"
            ),
            "rss": "tx_power_times_path_gain",
            "sinr": "rss_over_noise_plus_other_tx_rss",
        },
        "metric": "path_gain",
        "integrator": str(integrator),
        "combine_mode": "incoherent",
        "receiver_model": "matched_isotropic",
        "tx_power": tx_power,
        "noise_power": float(noise_power),
        "path_counts": path_counts,
        "scene": {
            "edge_selection_mode": getattr(edge_policy, "edge_selection_mode", "vertical_only"),
            "edge_diffraction": getattr(
                edge_policy,
                "edge_diffraction",
                getattr(edge_policy, "boundary_edge_policy", "exclude") == "half_plane",
            ),
            "boundary_edge_policy": getattr(edge_policy, "boundary_edge_policy", "exclude"),
            "n_diffraction_edges": int(getattr(scene, "n_diffraction_edges", 0)),
        },
        "runtime_reuse": {
            "diffraction_state_prep_cache": {
                "mode": str(runtime_reuse["cache_mode"]),
                "hits": int(runtime_reuse["state_preparation_hits"]),
                "misses": int(runtime_reuse["state_preparation_misses"]),
                "state_layout": str(runtime_reuse["state_layout"]),
            },
        },
        "accumulation_backend": {
            "requested": str(integrator_options.accumulation_backend),
            "resolved": str(resolved_accumulation_backend),
            "cell_accumulation_mode": cell_accumulation_mode,
        },
        "solver_mode": solver_controls,
        "execution_intent": dict(solver_controls["execution_intent"]),
        "runtime_backends": {
            "reflection": dict(reflection_runtime_backend),
            "diffraction": dict(diffraction_runtime_backend),
            "suffix": {
                "requested_backend": "disabled_for_monte_carlo",
                "resolved_backend": "disabled_for_monte_carlo",
                "implementation": "disabled",
            },
        },
        "monte_carlo": {
            "ad_mode": bool(resolved_ad_mode),
            "ad_backend": str(ad_backend),
            "primal_loop_mode": str(loop_mode),
            "tx_accumulation_transport_mode": (
                "fixed_tape_full_replay_central_difference"
                if resolved_ad_mode else "disabled"
            ),
            "tx_accumulation_transport_step": (
                float(mc_ad.MC_TX_TRANSPORT_FD_STEP) if resolved_ad_mode else 0.0
            ),
            "tape_layout_version": (
                "single_solver_native_sparse_coeff_tape_v3"
                if resolved_ad_mode else "disabled"
            ),
            "samples_per_tx": samples_per_tx,
            "reflection_n_rays": reflection_ray_count,
            "seed": seed,
            "shadow_boundary_mode": shadow_boundary_mode,
            "shadow_boundary_correction": {
                "enabled": shadow_boundary_mode == "utd_power_smoothing",
                "backend": shadow_runtime.get(
                    "backend",
                    "disabled"
                    if shadow_boundary_mode == "none"
                    else str(tuning.shadow_boundary_backend),
                ),
                "requested_backend": str(tuning.shadow_boundary_backend),
                "candidate_tiles": int(shadow_runtime.get("candidate_tiles", 0)),
                "candidate_pairs": int(shadow_runtime.get("candidate_pairs", 0)),
                "candidate_ratio": float(shadow_runtime.get("candidate_ratio", 0.0)),
                "source_total_edges": int(shadow_runtime.get("source_total_edges", 0)),
                "source_visible_edges": int(shadow_runtime.get("source_visible_edges", 0)),
                "source_visibility_samples_per_edge": int(
                    shadow_runtime.get("source_visibility_samples_per_edge", 0)
                ),
                "tile_shape": shadow_tile_shape,
                "band_width_wavelengths": float(
                    shadow_runtime.get(
                        "band_width_wavelengths",
                        tuning.shadow_boundary_band_width_wavelengths,
                    )
                ),
                "weight_aggregation": str(
                    shadow_runtime.get(
                        "weight_aggregation",
                        "max_weight_weighted_response_average",
                    )
                ),
                "incident_support": str(
                    shadow_runtime.get(
                        "incident_support",
                        "direct_los_or_matching_first_blocker_surface_group",
                    )
                ),
                "domain": "power",
                "phase_reference": "none",
                "formula": (
                    "per-cell shadow weights use max over source-visible finite-edge "
                    "transition weights; transition responses use weighted averages. "
                    "Final incident/reflection corrections are enabled only in cells "
                    "with sampled diffraction transition power for that boundary family. "
                    "incident_weight*(continued_incident_power*|0.5*(1+/-"
                    "incident_transition_response)|^2-los-"
                    "diffraction_incident_transition_power) plus "
                    "reflection_weight*(max(reflection,"
                    "diffraction_reflection_transition_power)*|0.5*(1+/-"
                    "reflection_transition_response)|^2-"
                    "reflection-diffraction_reflection_transition_power), clamped "
                    "below by -raw_total"
                ),
            },
            "filtering": filtering_metadata(mc_config.filtering),
            "rr_depth": rr_depth,
            "rr_prob": float(rr_prob),
            "stop_threshold_db": stop_threshold,
            "accepted_hit_counts": dict(counts),
            "state_pool": dict(state_pool),
            "los_strategy": "cell_center_visibility",
        },
    }


__all__ = ["COMPONENT_NAMES", "build_metadata"]
