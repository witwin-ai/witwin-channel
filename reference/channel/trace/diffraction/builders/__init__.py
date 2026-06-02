"""State construction, budgeting, and metadata for diffraction solving."""

import time

import drjit as dr
import witwin as wt

from ...materials import coerce_reflection_material_context, coerce_reflection_trace_detail
from ....utils.drjit_ops import EvalSync, complex_abs_sqr
from ..constants import (
    APPROX_MODE_DIRECT_FIRST_ORDER,
    APPROX_MODE_RECURSIVE_DIFFRACTION,
    APPROX_MODE_SAMPLED_INSERTED_REFLECTION,
    APPROX_MODE_SAMPLED_INSERTED_REFLECTION_CHAIN,
    APPROX_MODE_SAMPLED_REFLECTION_PREFIX,
    APPROX_MODE_SAMPLED_REFLECTION_PREFIX_CHAIN,
    SOURCE_TYPE_DIRECT_TX,
    SOURCE_TYPE_REFLECTION_PREFIX,
    _approximation_mode_label,
    _ownership_label,
    _source_type_label,
)
from ..geometry import (
    _edge_face_material_inputs,
    _edge_face_reflection_operators,
)
from ..state import (
    _build_global_to_local_index,
    _empty_state_arrays,
    _finalize_state_lineage,
    path_export_state_layout,
    reduce_state_arrays_for_path_export,
)
from ..state.pruning import (
    prune_state_arrays_for_pre_expansion,
    resolve_pre_expansion_pruning_policy,
)
from ..profiling import summarize_state_memory_profile
from witwin.channel.kernels.trace.packed_state import concat_state_arrays
from witwin.channel.kernels.trace.pruning_sort import (
    prune_state_arrays_by_budget,
    prune_state_arrays_by_budget_pair,
)


def _vector_field_power(field_vec):
    return (
        complex_abs_sqr(field_vec["x"])
        + complex_abs_sqr(field_vec["y"])
        + complex_abs_sqr(field_vec["z"])
    )


def _empty_builder_timing_report():
    return {
        "edge_cache_seconds": 0.0,
        "global_index_seconds": 0.0,
        "candidate_backend_resolution_seconds": 0.0,
        "tx_first_order_seconds": 0.0,
        "reflection_prefix_first_order_seconds": 0.0,
        "concat_first_order_seconds": 0.0,
        "higher_order_seconds": 0.0,
        "inserted_reflection_seconds": 0.0,
        "pre_expansion_prune_seconds": 0.0,
        "inserted_prune_seconds": 0.0,
        "concat_higher_order_seconds": 0.0,
        "total_prune_seconds": 0.0,
        "total_seconds": 0.0,
    }


def _timing_stage_start(timing_report):
    if timing_report is None:
        return None
    EvalSync.sync()
    return time.perf_counter()


def _timing_stage_finish(timing_report, key: str, start_time, *values):
    if timing_report is None or start_time is None:
        return
    if len(values) > 0:
        EvalSync.barrier(*values)
    timing_report[key] += time.perf_counter() - start_time


def _state_edge_face_material_response(
    *,
    edge_pos,
    edge_dir,
    n0,
    nn,
    source_pos,
    adjacent_face0,
    adjacent_face1,
    wavelength,
    scene,
    material_detail,
    reflection_coef,
    use_scene_materials,
    tx_polarization,
):
    edge_state = {
        "edge_pos": edge_pos,
        "edge_dir": edge_dir,
        "n0": n0,
        "n_face_n": nn,
        "source_pos": source_pos,
        "adjacent_face0": adjacent_face0,
        "adjacent_face1": adjacent_face1,
    }
    width = dr.width(edge_pos.x)
    face0_material, face1_material = _edge_face_material_inputs(
        edge_state,
        width,
        material_detail,
        scene=scene,
        reflection_coef=reflection_coef,
        use_scene_materials=use_scene_materials,
    )
    face0_operator, face1_operator = _edge_face_reflection_operators(
        edge_state,
        width,
        material_detail,
        wavelength,
        scene=scene,
        reflection_coef=reflection_coef,
        use_scene_materials=use_scene_materials,
        tx_polarization=tx_polarization,
    )
    return face0_operator, face1_operator, face0_material, face1_material


# ---------------------------------------------------------------------------
# Imports from sub-modules (re-exported so that ``from .builders import X``
# continues to work everywhere).
# ---------------------------------------------------------------------------

from .tx import _build_tx_first_order_state_arrays  # noqa: E402
from .prefix import _build_reflection_first_order_state_arrays  # noqa: E402
from .higher import (  # noqa: E402
    _build_higher_order_state_arrays,
    _build_inserted_reflection_state_arrays,
    _resolve_higher_order_candidate_backend,
)


def _count_point_sources(reflection_detail):
    count = 1
    if reflection_detail is None:
        return count
    for paths in coerce_reflection_trace_detail(reflection_detail).source_paths_per_bounce:
        if paths is not None:
            count += int(paths.n_paths)
    return count


def _count_reflection_prefix_clusters(reflection_detail):
    count = 0
    if reflection_detail is None:
        return count
    for paths in coerce_reflection_trace_detail(reflection_detail).source_paths_per_bounce:
        if paths is not None:
            count += int(paths.n_paths)
    return count


def _build_solver_metadata(
    scene,
    max_diffractions,
    reflection_detail,
    reflection_n_rays,
    reflection_max_bounces,
    reflected_suffix_enabled,
    inserted_reflection_enabled,
    max_inserted_reflections_per_path,
    total_state_budget_per_order,
    inserted_state_budget_per_order,
    path_budget_report,
):
    state_memory_profile = summarize_state_memory_profile(
        path_budget_report,
        fallback_history_size=max(1, int(max_diffractions)),
    )
    prefix_clusters = _count_reflection_prefix_clusters(reflection_detail)
    prefix_enabled = prefix_clusters > 0
    alternating_chain_enabled = (
        bool(inserted_reflection_enabled)
        and int(max_inserted_reflections_per_path) > 1
        and int(max_diffractions) > 2
    )
    edge_selection_mode = getattr(scene, "edge_selection_mode", "vertical_only")
    vertical_ratio = None if scene is None else float(scene.vertical_ratio)
    boundary_edge_policy = getattr(scene, "boundary_edge_policy", "exclude")
    runtime_state = getattr(scene, "_runtime", scene)
    edge_selection_summary = dict(getattr(runtime_state, "edge_selection_summary", {}))
    higher_order_candidate_backend = "not_used"
    if path_budget_report is not None:
        higher_order_candidate_backend = path_budget_report.get(
            "higher_order_candidate_backend",
            higher_order_candidate_backend,
        )

    return {
        "edge_selection_mode": edge_selection_mode,
        "boundary_edge_policy": boundary_edge_policy,
        "finite_edge_treatment": {
            "mode": "finite_wedge",
            "bounds_required": True,
            "notes": (
                "All diffraction states use explicit finite edge_line_min/edge_line_max bounds "
                "and the shared finite-segment truncation model."
            ),
        },
        "shadow_boundary_treatment": {
            "mode": "geometric_half_space_classification",
            "notes": (
                "Diffraction source and target validity are classified with exact wedge-face half-space tests "
                "in the plane perpendicular to the edge, rather than hard phi/phi_prime angular clipping."
            ),
        },
        "edge_selection_summary": edge_selection_summary,
        "vertical_ratio": vertical_ratio,
        "max_diffractions": int(max_diffractions),
        "reflection_prefix_path_count": int(prefix_clusters),
        "reflection_suffix_enabled": bool(reflected_suffix_enabled),
        "reflection_suffix_budget": {
            "n_rays": int(reflection_n_rays),
            "max_bounces": int(reflection_max_bounces),
        },
        "path_budget_policy": {
            "enabled": bool(
                total_state_budget_per_order is not None
                or inserted_state_budget_per_order is not None
            ),
            "pruning_metric": "incident_power",
            "pre_expansion_policy": (
                None if path_budget_report is None else path_budget_report.get("pre_expansion_policy")
            ),
            "total_state_budget_per_order": (
                None if total_state_budget_per_order is None else int(total_state_budget_per_order)
            ),
            "inserted_state_budget_per_order": (
                None if inserted_state_budget_per_order is None else int(inserted_state_budget_per_order)
            ),
            "report": path_budget_report,
        },
        "state_memory_profile": state_memory_profile,
        "higher_order_candidate_builder": {
            "mode": higher_order_candidate_backend,
            "notes": (
                "Higher-order edge candidates use RayD nearest-edge BVH queries restricted to the selected "
                "diffraction-wedge set."
                if higher_order_candidate_backend == "rayd_edge_bvh"
                else (
                    "Higher-order edge candidates use Cartesian prev-state x selected-edge expansion in explicit "
                    "test mode."
                    if higher_order_candidate_backend == "bruteforce"
                    else "Higher-order edge candidate construction is not used when max_diffractions <= 1."
                )
            ),
        },
        "mixed_chain_budget": {
            "max_inserted_reflections_per_path": int(max_inserted_reflections_per_path),
        },
        "field_component_ownership": {
            "a_ref": {
                "owns": "Pure reflection-only paths with no diffraction events.",
            },
            "a_direct": {
                "owns": "Diffraction states with zero prefix/intermediate/suffix reflection depth.",
            },
            "a_multi": {
                "owns": "Diffraction states with at least one prefix, inserted, or suffix reflection event.",
            },
            "a_dif": {
                "composition": "a_direct + a_multi",
            },
            "a_tot": {
                "composition": "a_los + a_ref + a_dif",
            },
        },
        "audit_value_meanings": {
            "source_type_code": {
                SOURCE_TYPE_DIRECT_TX: _source_type_label(SOURCE_TYPE_DIRECT_TX),
                SOURCE_TYPE_REFLECTION_PREFIX: _source_type_label(SOURCE_TYPE_REFLECTION_PREFIX),
            },
            "ownership_code": {
                0: _ownership_label(0),
                1: _ownership_label(1),
            },
            "approximation_mode_code": {
                APPROX_MODE_DIRECT_FIRST_ORDER: _approximation_mode_label(APPROX_MODE_DIRECT_FIRST_ORDER),
                APPROX_MODE_RECURSIVE_DIFFRACTION: _approximation_mode_label(APPROX_MODE_RECURSIVE_DIFFRACTION),
                APPROX_MODE_SAMPLED_REFLECTION_PREFIX: _approximation_mode_label(APPROX_MODE_SAMPLED_REFLECTION_PREFIX),
                APPROX_MODE_SAMPLED_REFLECTION_PREFIX_CHAIN: _approximation_mode_label(APPROX_MODE_SAMPLED_REFLECTION_PREFIX_CHAIN),
                APPROX_MODE_SAMPLED_INSERTED_REFLECTION: _approximation_mode_label(
                    APPROX_MODE_SAMPLED_INSERTED_REFLECTION
                ),
                APPROX_MODE_SAMPLED_INSERTED_REFLECTION_CHAIN: _approximation_mode_label(
                    APPROX_MODE_SAMPLED_INSERTED_REFLECTION_CHAIN
                ),
            },
        },
        "path_families": {
            "S -> D": {
                "status": "exact",
                "notes": "Direct transmitter to diffraction edge using the currently selected edge-selection mode.",
            },
            "R^n -> D": {
                "status": "approximate" if prefix_enabled else "absent",
                "notes": "Reflection prefixes use sampled path-faithful image-source chains discovered by the reflection tracer.",
            },
            "S -> D -> ... -> D": {
                "status": "approximate" if int(max_diffractions) > 1 else "absent",
                "notes": "Higher-order diffraction is propagated recursively through edge states.",
            },
            "R^n -> D -> ... -> D": {
                "status": "approximate" if prefix_enabled and int(max_diffractions) > 1 else "absent",
                "notes": "Mixed reflection-prefix and higher-order diffraction uses sampled reflection chains plus recursive diffraction propagation.",
            },
            "... -> D -> R^n": {
                "status": "approximate" if reflected_suffix_enabled else "absent",
                "notes": "Suffix reflections are sampled from the last diffraction state with Monte Carlo reflection rays.",
            },
            "D -> R -> D": {
                "status": "approximate" if inserted_reflection_enabled else "absent",
                "notes": "One inserted reflection between diffraction events is sampled from diffraction states and re-injected as a new edge state.",
            },
            "Arbitrary alternating mixed chains": {
                "status": "approximate" if alternating_chain_enabled else "absent",
                "notes": (
                    "Alternating D/R chains are expanded recursively with at most one sampled reflection between "
                    "consecutive diffraction events, bounded by max_diffractions and max_inserted_reflections_per_path."
                ),
            },
        },
    }


def _prepare_diffraction_state_arrays(
    tx_pos,
    rx_z,
    scene,
    wavelength,
    k,
    reflection_detail,
    material_detail,
    reflection_n_rays,
    reflection_max_bounces,
    reflection_coef,
    reflection_mode,
    max_diffractions,
    total_state_budget_per_order=None,
    inserted_state_budget_per_order=None,
    max_inserted_reflections_per_path=None,
    retain_cold_metadata=True,
    use_scene_materials=False,
    tx_polarization=(1.0, 0.0, 0.0),
    solver_mode="accuracy",
    memory_profile="default",
    state_layout="full",
    preserve_higher_order_candidate_topology: bool = False,
    collect_timing: bool = False,
):
    timing_report = _empty_builder_timing_report() if collect_timing else None
    total_start = _timing_stage_start(timing_report)
    max_order = max(1, int(max_diffractions))
    if max_inserted_reflections_per_path is None:
        max_inserted_reflections_per_path = max(0, max_order - 1)
    else:
        max_inserted_reflections_per_path = max(0, int(max_inserted_reflections_per_path))
    t0 = _timing_stage_start(timing_report)
    edge_cache = scene.get_edge_data(rx_z, include_projection=False)
    _timing_stage_finish(timing_report, "edge_cache_seconds", t0, edge_cache)
    edge_data = edge_cache.get("edge_data")
    if edge_data is None:
        empty_states = _empty_state_arrays(history_size=max_order)
        if str(state_layout) == "path_export_reduced":
            empty_states = reduce_state_arrays_for_path_export(empty_states)
        if timing_report is not None and total_start is not None:
            timing_report["total_seconds"] = time.perf_counter() - total_start
        return edge_cache, None, empty_states, {
            "per_order": [],
            "timing": timing_report,
            "state_layout": str(state_layout),
        }

    t0 = _timing_stage_start(timing_report)
    global_to_local_idx = _build_global_to_local_index(scene, edge_data)
    _timing_stage_finish(timing_report, "global_index_seconds", t0, global_to_local_idx)
    t0 = _timing_stage_start(timing_report)
    higher_order_candidate_backend = (
        _resolve_higher_order_candidate_backend(
            scene,
            edge_data,
            global_to_local_idx,
            candidate_backend="auto",
        )
        if max_order > 1
        else "not_used"
    )
    _timing_stage_finish(
        timing_report,
        "candidate_backend_resolution_seconds",
        t0,
    )
    t0 = _timing_stage_start(timing_report)
    tx_first_order = _build_tx_first_order_state_arrays(
        tx_pos,
        edge_data,
        wavelength,
        k,
        history_size=max_order,
        retain_cold_metadata=retain_cold_metadata,
        scene=scene,
        material_detail=material_detail,
        reflection_coef=reflection_coef,
        use_scene_materials=use_scene_materials,
        tx_polarization=tx_polarization,
    )
    _timing_stage_finish(timing_report, "tx_first_order_seconds", t0, tx_first_order)
    reflection_prefix_builder_stats = {}
    t0 = _timing_stage_start(timing_report)
    reflection_first_order = _build_reflection_first_order_state_arrays(
        reflection_detail,
        scene,
        edge_data,
        wavelength,
        k,
        global_to_local_idx,
        history_size=max_order,
        retain_cold_metadata=retain_cold_metadata,
        material_detail=material_detail,
        reflection_coef=reflection_coef,
        use_scene_materials=use_scene_materials,
        tx_polarization=tx_polarization,
        stats=reflection_prefix_builder_stats,
    )
    _timing_stage_finish(
        timing_report,
        "reflection_prefix_first_order_seconds",
        t0,
        reflection_first_order,
    )
    t0 = _timing_stage_start(timing_report)
    first_order_states = concat_state_arrays([tx_first_order, reflection_first_order])
    lineage_store = None
    next_state_id = 0
    if retain_cold_metadata:
        first_order_states, lineage_store, next_state_id = _finalize_state_lineage(
            first_order_states,
            lineage_store=lineage_store,
            next_state_id=next_state_id,
        )
    _timing_stage_finish(timing_report, "concat_first_order_seconds", t0, first_order_states)

    all_state_arrays = [first_order_states]
    prev_states = first_order_states
    budget_report = {
        "higher_order_candidate_backend": higher_order_candidate_backend,
        "per_order": [
            {
                "order": 1,
                "direct_states": int(tx_first_order["n_states"]),
                "prefix_states": int(reflection_first_order["n_states"]),
                "reflection_prefix_builder": reflection_prefix_builder_stats,
                "inserted_states_before_prune": 0,
                "inserted_states_after_prune": 0,
                "inserted_budget_applied": False,
                "total_states_before_prune": int(first_order_states["n_states"]),
                "total_states_after_prune": int(first_order_states["n_states"]),
                "total_budget_applied": False,
            }
        ],
    }
    inserted_reflection_enabled = (
        scene is not None
        and reflection_n_rays > 0
        and reflection_max_bounces > 0
        and max_order > 1
        and max_inserted_reflections_per_path > 0
    )
    reflection_context = coerce_reflection_material_context(
        reflection_detail,
        default_gain=reflection_coef,
    )
    pre_expansion_policy = resolve_pre_expansion_pruning_policy(
        solver_mode=str(solver_mode),
        memory_profile=str(memory_profile),
        total_state_budget_per_order=total_state_budget_per_order,
        inserted_state_budget_per_order=inserted_state_budget_per_order,
    )
    budget_report["pre_expansion_policy"] = dict(pre_expansion_policy)
    for order_idx in range(2, max_order + 1):
        higher_order_builder_stats = {}
        inserted_reflection_builder_stats = None
        higher_source_states = prev_states
        inserted_source_states = prev_states
        higher_source_budget = {
            "budget_name": "pre_expansion_higher_order_source_budget",
            "requested_budget": None,
            "pruning_metric": "incident_power",
            "input_states": int(prev_states["n_states"]),
            "kept_states": int(prev_states["n_states"]),
            "dropped_states": 0,
            "applied": False,
            "stage": "pre_expansion",
            "policy": str(pre_expansion_policy["policy"]),
            "shared_pre_expansion_result": False,
        }
        inserted_source_budget = {
            "budget_name": "pre_expansion_inserted_source_budget",
            "requested_budget": None,
            "pruning_metric": "incident_power",
            "input_states": int(prev_states["n_states"]),
            "kept_states": int(prev_states["n_states"]),
            "dropped_states": 0,
            "applied": False,
            "stage": "pre_expansion",
            "policy": str(pre_expansion_policy["policy"]),
            "shared_pre_expansion_result": False,
        }
        if pre_expansion_policy["enabled"]:
            higher_budget_value = pre_expansion_policy["higher_order_source_budget"]
            inserted_budget_value = (
                pre_expansion_policy["inserted_source_budget"]
                if inserted_reflection_enabled
                else None
            )
            if inserted_reflection_enabled:
                t0 = _timing_stage_start(timing_report)
                (
                    higher_source_states,
                    higher_source_budget,
                    inserted_source_states,
                    inserted_source_budget,
                ) = prune_state_arrays_by_budget_pair(
                    prev_states,
                    higher_budget_value,
                    inserted_budget_value,
                    higher_budget_name="pre_expansion_higher_order_source_budget",
                    inserted_budget_name="pre_expansion_inserted_source_budget",
                )
                _timing_stage_finish(
                    timing_report,
                    "pre_expansion_prune_seconds",
                    t0,
                    higher_source_states,
                    inserted_source_states,
                )
                shared_pre_expansion_result = higher_budget_value == inserted_budget_value
                higher_source_budget["stage"] = "pre_expansion"
                higher_source_budget["policy"] = str(pre_expansion_policy["policy"])
                higher_source_budget["shared_pre_expansion_result"] = bool(
                    shared_pre_expansion_result
                )
                inserted_source_budget["stage"] = "pre_expansion"
                inserted_source_budget["policy"] = str(pre_expansion_policy["policy"])
                inserted_source_budget["shared_pre_expansion_result"] = bool(
                    shared_pre_expansion_result
                )
            else:
                t0 = _timing_stage_start(timing_report)
                higher_source_states, higher_source_budget = prune_state_arrays_for_pre_expansion(
                    prev_states,
                    higher_budget_value,
                    budget_name="pre_expansion_higher_order_source_budget",
                    policy=str(pre_expansion_policy["policy"]),
                )
                _timing_stage_finish(
                    timing_report,
                    "pre_expansion_prune_seconds",
                    t0,
                    higher_source_states,
                )
        t0 = _timing_stage_start(timing_report)
        direct_states = _build_higher_order_state_arrays(
            higher_source_states,
            edge_data,
            k,
            scene=scene,
            wavelength=wavelength,
            material_detail=material_detail,
            reflection_coef=reflection_coef,
            use_scene_materials=use_scene_materials,
            tx_polarization=tx_polarization,
            global_to_local_idx=global_to_local_idx,
            candidate_backend=higher_order_candidate_backend,
            retain_cold_metadata=retain_cold_metadata,
            preserve_candidate_topology=bool(preserve_higher_order_candidate_topology),
            stats=higher_order_builder_stats,
        )
        _timing_stage_finish(timing_report, "higher_order_seconds", t0, direct_states)
        inserted_states = _empty_state_arrays(history_size=max_order)
        if inserted_reflection_enabled:
            inserted_reflection_builder_stats = {}
            if pre_expansion_policy["enabled"] and not bool(
                inserted_source_budget.get("shared_pre_expansion_result", False)
                or inserted_source_budget.get("paired_pre_expansion_sort", False)
            ):
                t0 = _timing_stage_start(timing_report)
                inserted_source_states, inserted_source_budget = prune_state_arrays_for_pre_expansion(
                    prev_states,
                    pre_expansion_policy["inserted_source_budget"],
                    budget_name="pre_expansion_inserted_source_budget",
                    policy=str(pre_expansion_policy["policy"]),
                )
                _timing_stage_finish(
                    timing_report,
                    "pre_expansion_prune_seconds",
                    t0,
                    inserted_source_states,
                )
            t0 = _timing_stage_start(timing_report)
            inserted_states = _build_inserted_reflection_state_arrays(
                inserted_source_states,
                scene,
                edge_data,
                global_to_local_idx,
                wavelength,
                k,
                n_rays=reflection_n_rays,
                reflection_gain=reflection_context.reflection_gain,
                reflection_material_override=(
                    reflection_context.reflection_material
                    if reflection_detail is not None
                    else material_detail
                ),
                reflection_use_scene_materials=(
                    reflection_context.use_scene_materials
                    if reflection_detail is not None
                    else False
                ),
                material_detail=material_detail,
                use_scene_materials=use_scene_materials,
                reflection_mode=reflection_mode,
                max_inserted_reflections_per_path=max_inserted_reflections_per_path,
                tx_polarization=tx_polarization,
                retain_cold_metadata=retain_cold_metadata,
                stats=inserted_reflection_builder_stats,
            )
            _timing_stage_finish(
                timing_report,
                "inserted_reflection_seconds",
                t0,
                inserted_states,
            )
        t0 = _timing_stage_start(timing_report)
        inserted_states, inserted_budget = prune_state_arrays_by_budget(
            inserted_states,
            inserted_state_budget_per_order,
            "inserted_state_budget_per_order",
        )
        _timing_stage_finish(timing_report, "inserted_prune_seconds", t0, inserted_states)
        t0 = _timing_stage_start(timing_report)
        combined_states = concat_state_arrays([direct_states, inserted_states])
        _timing_stage_finish(timing_report, "concat_higher_order_seconds", t0, combined_states)
        t0 = _timing_stage_start(timing_report)
        combined_states, total_budget = prune_state_arrays_by_budget(
            combined_states,
            total_state_budget_per_order,
            "total_state_budget_per_order",
        )
        if retain_cold_metadata:
            combined_states, lineage_store, next_state_id = _finalize_state_lineage(
                combined_states,
                lineage_store=lineage_store,
                next_state_id=next_state_id,
            )
        _timing_stage_finish(timing_report, "total_prune_seconds", t0, combined_states)
        prev_states = combined_states
        budget_report["per_order"].append({
            "order": int(order_idx),
            "higher_order_source_states_before_pre_prune": int(higher_source_budget["input_states"]),
            "higher_order_source_states_after_pre_prune": int(higher_source_budget["kept_states"]),
            "higher_order_source_pre_prune_applied": bool(higher_source_budget["applied"]),
            "higher_order_source_pre_prune_shared": bool(
                higher_source_budget.get("shared_pre_expansion_result", False)
            ),
            "inserted_source_states_before_pre_prune": int(inserted_source_budget["input_states"]),
            "inserted_source_states_after_pre_prune": int(inserted_source_budget["kept_states"]),
            "inserted_source_pre_prune_applied": bool(inserted_source_budget["applied"]),
            "inserted_source_pre_prune_shared": bool(
                inserted_source_budget.get("shared_pre_expansion_result", False)
            ),
            "direct_states": int(direct_states["n_states"]),
            "higher_order_builder": higher_order_builder_stats,
            "inserted_reflection_builder": inserted_reflection_builder_stats,
            "inserted_states_before_prune": inserted_budget["input_states"],
            "inserted_states_after_prune": inserted_budget["kept_states"],
            "inserted_budget_applied": bool(inserted_budget["applied"]),
            "total_states_before_prune": total_budget["input_states"],
            "total_states_after_prune": total_budget["kept_states"],
            "total_budget_applied": bool(total_budget["applied"]),
        })
        if prev_states["n_states"] == 0:
            break
        all_state_arrays.append(prev_states)

    peak_pre = 0
    peak_post = 0
    for item in budget_report["per_order"]:
        peak_pre = max(peak_pre, int(item["total_states_before_prune"]))
        peak_post = max(peak_post, int(item["total_states_after_prune"]))
    budget_report["peak_total_states_before_prune"] = int(peak_pre)
    budget_report["peak_total_states_after_prune"] = int(peak_post)
    budget_report["final_total_states"] = int(sum(state_arrays["n_states"] for state_arrays in all_state_arrays))
    budget_report["history_size"] = int(max_order)
    budget_report["cold_metadata_retained"] = bool(retain_cold_metadata)
    budget_report["preserve_higher_order_candidate_topology"] = bool(
        preserve_higher_order_candidate_topology
    )
    final_state_arrays = concat_state_arrays(all_state_arrays)
    if str(state_layout) == "path_export_reduced":
        final_state_arrays = reduce_state_arrays_for_path_export(final_state_arrays)
    budget_report["state_layout"] = (
        path_export_state_layout(final_state_arrays) or str(state_layout)
    )
    if timing_report is not None and total_start is not None:
        EvalSync.barrier(final_state_arrays)
        timing_report["total_seconds"] = time.perf_counter() - total_start
    budget_report["timing"] = timing_report

    return edge_cache, edge_data, final_state_arrays, budget_report


__all__ = [
    "_build_higher_order_state_arrays",
    "_build_inserted_reflection_state_arrays",
    "_build_reflection_first_order_state_arrays",
    "_build_solver_metadata",
    "_build_tx_first_order_state_arrays",
    "_count_point_sources",
    "_count_reflection_prefix_clusters",
    "_prepare_diffraction_state_arrays",
]
