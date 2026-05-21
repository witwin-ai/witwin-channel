"""Bidirectional path-tracing radiomap integrator."""

from __future__ import annotations

import dataclasses
import time
from dataclasses import dataclass
from typing import Mapping

from witwin.channel.core.scene import Scene
from witwin.channel.core.numerics import arrays
from witwin.channel.core.numerics.arrays import scalar

from ..config import Config, ResolvedTraceConfig
from witwin.channel.core.grid import GridSpec
from ..trace import ad_support as mc_ad
from ..trace.postprocessing import ShadowBoundary
from ..trace.los import LoS
from ..types import Integrator
from . import basic_ad as mc_custom
from .basic import (
    Basic,
    EXTRA_COMPONENT_KEYS,
    ForwardPassState,
    _single_tx_sinr,
    build_result,
    finalize_weighted_diagnostics,
)
from .metadata import build_metadata
from .bdpt_diffraction import BDPTDiffractionMIS, BDPTDiffractionResult, BDPTDiffractionTape


@dataclass(frozen=True, slots=True)
class BDPT(Integrator):
    """Bidirectional radiomap integrator, staged around exact specular constraints."""

    mode: str = "bdpt"

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
        resolved_ad_mode: bool | None = None,
        ad_backend: str = "disabled",
        collect_ad_tapes: bool = False,
        return_primal_state: bool = False,
        apply_result_filtering: bool = True,
    ):
        del self
        grad_sensitive_workload = mc_custom.grad_sensitive(
            config,
            tx_pos=tx_pos,
            scene=scene,
        )
        if resolved_ad_mode is None:
            resolved_ad_mode = (
                bool(grad_sensitive_workload)
                if mc_config.integrator_options.ad is None
                else bool(mc_config.integrator_options.ad)
            )
        resolved_ad_mode = bool(resolved_ad_mode)
        if resolved_ad_mode and not return_primal_state:
            from .bdpt_ad import BDPTIntegratorAD

            return BDPTIntegratorAD.integrate(
                tx_pos=tx_pos,
                grid_spec=grid_spec,
                mc_config=mc_config,
                scene=scene,
                config=config,
                solver_controls=solver_controls,
                accumulation_backend=accumulation_backend,
                return_timing=return_timing,
                grad_sensitive_workload=grad_sensitive_workload,
                trace_primal_fn=BDPT().integrate,
                tx_power=float(tx_power),
                noise_power=noise_power,
            )
        if grad_sensitive_workload and not resolved_ad_mode:
            detached = mc_custom.BasicIntegratorAD.detached_workload(scene, tx_pos, config)
            tx_pos = detached["tx_pos"]
            scene = detached["scene"]

        effective = solver_controls["effective"]
        setup = Basic.prepare_config(
            tx_pos=tx_pos,
            grid_spec=grid_spec,
            mc_config=mc_config,
            scene=scene,
            config=config,
            tx_power=float(tx_power),
            accumulation_backend=accumulation_backend,
            return_timing=return_timing,
            loop_mode="symbolic",
            effective=effective,
        )
        grid = setup.grid
        weighted_diagnostics = setup.weighted_diagnostics
        timing = setup.timing
        total_start = None
        if return_timing:
            arrays.sync_thread()
            total_start = time.perf_counter()

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

        reflection_coupled_diffraction = bool(config.enable_bdpt_reflection_coupled_diffraction)
        max_diffraction_depth = int(effective["max_diffractions"])
        refl_result = Basic.run_reflection(
            scene=scene,
            grid=grid,
            tx_pos=tx_pos,
            config=config,
            samples_per_tx=setup.reflection_n_rays,
            seed=setup.seed,
            ray_batch_size=int(setup.reflection_batch_plan.ray_batch_size),
            solid_angle_per_ray=float(setup.reflection_solid_angle_per_ray),
            cell_area=float(setup.cell_area),
            effective=effective,
            rr_depth=setup.rr_depth,
            rr_prob=setup.rr_prob,
            stop_threshold_linear=setup.stop_threshold_linear,
            material_omega=setup.material_omega,
            weighted_diagnostics=weighted_diagnostics,
            collect_wedges=(
                max_diffraction_depth > 0
                and reflection_coupled_diffraction
            ),
            collect_ad_tapes=collect_ad_tapes,
            collect_wedge_prefixes=reflection_coupled_diffraction,
            loop_mode=setup.loop_mode,
            resolved_accumulation_backend=setup.resolved_accumulation_backend,
        )
        refl_result.path_counts.los += los_result.path_count

        if max_diffraction_depth > 0:
            if return_timing and timing is not None:
                arrays.sync_thread()
                t0 = time.perf_counter()
            diff_result = BDPTDiffractionMIS.trace(
                scene=scene,
                grid=grid,
                tx_pos=tx_pos,
                config=config,
                samples_per_tx=setup.samples_per_tx,
                seed=setup.seed,
                diff_gain_scale=setup.diff_gain_scale,
                cell_area=float(setup.cell_area),
                weighted_diagnostics=weighted_diagnostics,
                loop_mode=setup.loop_mode,
                max_depth=max_diffraction_depth,
                sample_sequence=str(mc_config.integrator_options.bdpt_diffraction_sampling),
                prefix_store=(
                    refl_result.diff_state_store
                    if reflection_coupled_diffraction
                    else None
                ),
                collect_ad_tapes=collect_ad_tapes,
            )
            refl_result.path_counts.diffraction += diff_result.path_count
            if return_timing and timing is not None:
                arrays.sync_thread()
                timing.diffraction_seconds += time.perf_counter() - t0
        else:
            diff_result = BDPTDiffractionResult.zero()

        if (
            str(mc_config.tuning.shadow_boundary_mode) == "utd_power_smoothing"
            and max_diffraction_depth > 0
        ):
            ShadowBoundary.accumulate_into_diagnostics(
                weighted_diagnostics=weighted_diagnostics,
                scene=scene,
                tx_pos=tx_pos,
                grid=grid,
                config=config,
                edge_indices=diff_result.edge_indices,
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
            grid=grid,
            scene=scene,
            batch_plan=setup.batch_plan,
            ray_sampling_metadata=setup.ray_sampling_metadata,
            runtime_reuse={
                "cache_mode": "disabled",
                "state_preparation_hits": 0,
                "state_preparation_misses": 0,
                "state_layout": (
                    "bdpt_selected_wedge_direct_and_reflection_prefix_connection"
                    if reflection_coupled_diffraction
                    else "bdpt_selected_wedge_direct_and_recursive_connection"
                ),
            },
            solver_controls=solver_controls,
            resolved_ad_mode=resolved_ad_mode,
            ad_backend=ad_backend if resolved_ad_mode else "disabled",
            mc_config=mc_config,
            samples_per_tx=setup.samples_per_tx,
            reflection_n_rays=setup.reflection_n_rays,
            reflection_batch_plan=setup.reflection_batch_plan,
            seed=setup.seed,
            accepted_hit_counts=refl_result.path_counts,
            weighted_diagnostics=weighted_diagnostics,
            resolved_accumulation_backend=setup.resolved_accumulation_backend,
            reflection_runtime_backend={
                **setup.reflection_runtime_backend,
                "implementation": "bdpt_forward_specular_reflection_only",
            },
            diffraction_runtime_backend={
                "implementation": "bdpt_wedge_balance_mis_depth_limited_v2",
                "cell_scatter_backend": "drjit_scatter_reduce",
                "wedge_discovery_backend": (
                    "selected_scene_wedges_plus_forward_specular_prefix_wedges"
                    if reflection_coupled_diffraction
                    else "selected_scene_wedges_only"
                ),
                "state_sampler": (
                    "bdpt_light_subpath_edge_length_chain_plus_receiver_and_specular_suffix_connections"
                    if reflection_coupled_diffraction
                    else "bdpt_light_subpath_edge_length_chain_plus_receiver_connections"
                ),
                "point_evaluation_backend": (
                    "sampled_edge_chain_diffraction_field_to_cell_center_or_plane_hit"
                ),
                "source_field_contract": "sionna_iso_v_implicit_basis",
                "mis_heuristic": "balance",
                "sample_sequence": str(mc_config.integrator_options.bdpt_diffraction_sampling),
                "first_order_edge_sampler": "length_receiver_solid_angle_source_power_mixture",
            },
            state_pool={
                "total": int(diff_result.state_count),
                "kept": int(diff_result.state_count),
                "reflection_prefix": int(diff_result.prefix_state_count),
                "threshold_pruned": 0,
                "roulette_pruned": 0,
            },
            tx_power=setup.tx_power,
            rr_depth=setup.rr_depth,
            rr_prob=setup.rr_prob,
            stop_threshold=setup.stop_threshold,
            solid_angle_per_ray=setup.solid_angle_per_ray,
            reflection_solid_angle_per_ray=setup.reflection_solid_angle_per_ray,
            loop_mode=setup.loop_mode,
            integrator="bdpt",
            noise_power=noise_power,
            scalar_fn=scalar,
        )
        metadata["bdpt"] = {
            "version": "v4_depth_limited_diffraction_materialized_balance_mis",
            "ad_mode": bool(resolved_ad_mode),
            "tape_layout_version": (
                "bdpt_diffraction_fixed_width_v1"
                if collect_ad_tapes
                else "disabled"
            ),
            "los_strategy": "cell_center_visibility",
            "reflection_policy": "forward_sampled_specular_only",
            "max_diffraction_depth_supported": BDPTDiffractionMIS.MAX_SUPPORTED_DIFFRACTION_DEPTH,
            "max_diffraction_depth_active": max_diffraction_depth,
            "mis_policy": {
                "heuristic": "balance",
                "scope": "depth_limited_wedge_diffraction",
                "sample_sequence": str(mc_config.integrator_options.bdpt_diffraction_sampling),
                "first_order_edge_sampler": {
                    "proposal": "length_receiver_solid_angle_source_power_mixture",
                    "baseline_mixture": float(1.0 - BDPTDiffractionMIS.FIRST_ORDER_IMPORTANCE_MIX),
                    "importance_mixture": float(BDPTDiffractionMIS.FIRST_ORDER_IMPORTANCE_MIX),
                    "pdf_correction": "edge_measure_weight",
                    "applies_to_orders": [1],
                },
                "active_order_range": (
                    [1, max_diffraction_depth]
                    if max_diffraction_depth > 0
                    else []
                ),
                "active_strategy_count": sum(
                    1
                    for sample_count in diff_result.strategy_samples.values()
                    if int(sample_count) > 0
                ) if max_diffraction_depth > 0 else 0,
                "sample_allocation": (
                    "split_even_by_diffraction_order_then_direct_cell_keller_cone_and_suffix_reflection"
                    if reflection_coupled_diffraction
                    else "split_even_by_diffraction_order_then_direct_cell_and_keller_cone"
                ),
                "pdf_measure": "discrete_receiver_cells_plus_continuous_plane_hits",
                "reflection_coupled_diffraction": reflection_coupled_diffraction,
                "reflection_prefix_measure": (
                    "stable_ray_depth_slots_with_solid_angle_source_weight"
                    if reflection_coupled_diffraction
                    else "disabled"
                ),
                "diffraction_material_gain": "applied_to_direct_and_face_utd_terms",
            },
            "path_families": {
                "S -> D": (
                    "inactive"
                    if max_diffraction_depth <= 0
                    else "active"
                ),
                "S -> D -> ... -> D": (
                    "active"
                    if max_diffraction_depth > 1
                    else "inactive"
                ),
                "R^n -> D": (
                    "disabled"
                    if not reflection_coupled_diffraction
                    else "active"
                    if int(diff_result.prefix_state_count) > 0
                    else "sampled_no_states"
                ),
                "R^n -> D -> ... -> D": (
                    "disabled"
                    if not reflection_coupled_diffraction
                    else "active"
                    if int(diff_result.prefix_state_count) > 0
                    and max_diffraction_depth > 1
                    else "inactive"
                ),
                "... -> D -> R^n": (
                    "active_one_bounce_suffix"
                    if reflection_coupled_diffraction
                    else "disabled"
                ),
                "D -> R": "active" if reflection_coupled_diffraction else "disabled",
                "D -> R -> D": "not_active",
            },
            "order_breakdown": {
                int(order): {
                    "samples": {
                        BDPTDiffractionMIS.DIRECT_STRATEGY: int(
                            diff_result.order_samples[order][BDPTDiffractionMIS.DIRECT_STRATEGY]
                        ),
                        BDPTDiffractionMIS.KELLER_STRATEGY: int(
                            diff_result.order_samples[order][BDPTDiffractionMIS.KELLER_STRATEGY]
                        ),
                        BDPTDiffractionMIS.SUFFIX_REFLECTION_STRATEGY: int(
                            diff_result.order_samples[order][
                                BDPTDiffractionMIS.SUFFIX_REFLECTION_STRATEGY
                            ]
                        ),
                    },
                    "accepted": {
                        BDPTDiffractionMIS.DIRECT_STRATEGY: int(
                            diff_result.order_counts[order][BDPTDiffractionMIS.DIRECT_STRATEGY]
                        ),
                        BDPTDiffractionMIS.KELLER_STRATEGY: int(
                            diff_result.order_counts[order][BDPTDiffractionMIS.KELLER_STRATEGY]
                        ),
                        BDPTDiffractionMIS.SUFFIX_REFLECTION_STRATEGY: int(
                            diff_result.order_counts[order][
                                BDPTDiffractionMIS.SUFFIX_REFLECTION_STRATEGY
                            ]
                        ),
                    },
                }
                for order in diff_result.order_samples
            },
            "strategies": {
                BDPTDiffractionMIS.DIRECT_STRATEGY: {
                    "samples": int(
                        diff_result.strategy_samples[BDPTDiffractionMIS.DIRECT_STRATEGY]
                    ),
                    "accepted": int(
                        diff_result.strategy_counts[BDPTDiffractionMIS.DIRECT_STRATEGY]
                    ),
                    "pdf": "uniform_cell_times_edge_length_density",
                    "gain_scale": "lambda_over_4pi_squared",
                    "mis_weight": "balance_against_keller_cone_area_density",
                    "total_edge_length": float(diff_result.total_edge_length),
                },
                BDPTDiffractionMIS.KELLER_STRATEGY: {
                    "samples": int(
                        diff_result.strategy_samples[BDPTDiffractionMIS.KELLER_STRATEGY]
                    ),
                    "accepted": int(
                        diff_result.strategy_counts[BDPTDiffractionMIS.KELLER_STRATEGY]
                    ),
                    "pdf": "edge_length_density_times_uniform_keller_cone_angle",
                    "mis_weight": "balance_against_direct_cell_area_density",
                    "total_edge_length": float(diff_result.total_edge_length),
                },
                BDPTDiffractionMIS.SUFFIX_REFLECTION_STRATEGY: {
                    "samples": int(
                        diff_result.strategy_samples[
                            BDPTDiffractionMIS.SUFFIX_REFLECTION_STRATEGY
                        ]
                    ),
                    "accepted": int(
                        diff_result.strategy_counts[
                            BDPTDiffractionMIS.SUFFIX_REFLECTION_STRATEGY
                        ]
                    ),
                    "pdf": "edge_chain_density_times_uniform_cell_times_uniform_reflection_primitive",
                    "reflection_constraint": "single_specular_image_connection",
                    "mis_weight": "separate_delta_family_no_balance_with_direct_or_keller",
                    "total_edge_length": float(diff_result.total_edge_length),
                },
            },
        }
        metadata["receiver_sampling"]["strategy"] = "bdpt_cell_center_connections"
        metadata["metric_contract"]["path_gain"] = (
            "cell_center_los_plus_sampled_reflection_and_bdpt_direct_diffraction_"
            "matched_isotropic_incoherent_power_with_signed_shadow_boundary_correction"
            if mc_config.tuning.shadow_boundary_mode == "utd_power_smoothing"
            else "cell_center_los_plus_sampled_reflection_and_bdpt_direct_diffraction_matched_isotropic_incoherent_power"
        )

        if return_timing and timing is not None and total_start is not None:
            arrays.sync_thread()
            timing.total_seconds = time.perf_counter() - total_start

        if return_primal_state:
            if collect_ad_tapes:
                los_tape, reflection_tape, _ = mc_ad.ADContext.finalize_tapes(
                    collect_ad_tapes=True,
                    max_bounces=int(effective["reflection_max_bounces"]),
                    path_tape_store=refl_result.path_tape_store,
                    diffraction_tape_store=None,
                    los_tape=los_result.tape,
                )
                diffraction_tape = (
                    diff_result.tape
                    if diff_result.tape is not None
                    else BDPTDiffractionTape.empty()
                )
            else:
                los_tape = None
                reflection_tape = None
                diffraction_tape = None
            return ForwardPassState(
                grid=grid,
                component_power={
                    key: weighted_diagnostics["incoherent"][key]
                    for key in ("los", "reflection", "diffraction", *EXTRA_COMPONENT_KEYS)
                },
                weighted_diagnostics=weighted_diagnostics,
                metadata=metadata,
                noise_power=float(noise_power),
                timing=timing,
                diffraction_edge_indices=diff_result.edge_indices,
                los_tape=los_tape,
                reflection_tape=reflection_tape,
                diffraction_tape=diffraction_tape,
                diff_length_weight=float(diff_result.total_edge_length),
            )

        path_gain = weighted_diagnostics["incoherent"]["total"]
        rss = path_gain * float(tx_power)
        return build_result(
            grid=grid,
            tx_power=float(tx_power),
            weighted_diagnostics=weighted_diagnostics,
            metadata=metadata,
            path_gain=path_gain,
            rss=rss,
            sinr=_single_tx_sinr(rss, noise_power=float(noise_power)),
            tx_pos=tx_pos,
            noise_power=float(noise_power),
            timing=None if timing is None else dataclasses.replace(timing),
            detach_tensors=True,
        )


__all__ = [
    "BDPT",
    "BDPTDiffractionMIS",
    "BDPTDiffractionResult",
]
