from __future__ import annotations

from dataclasses import dataclass

import drjit as dr
import witwin as wt

from ....kernels.monitors.common.reflection_family_tiles import build_reflection_family_tile_plan
from ....kernels.monitors.common.receiver_tiles import resolve_receiver_tiles
from ....kernels.monitors.common.receiver_tiles.native_impl import tile_receiver_counts
from ....kernels.monitors.common.utd_state_tiles import build_utd_state_tile_plan


_RADIO_MAP_DIFFRACTION_TILED_MAX_PAIR_RATIO = 0.50
_RADIO_MAP_DIFFRACTION_TILED_MAX_TASK_COUNT = 256
_RADIO_MAP_REFLECTION_TILED_MAX_PAIR_RATIO = 0.95
_RADIO_MAP_REFLECTION_TILED_MAX_TASK_COUNT = 4096


@dataclass(frozen=True)
class RadioMapDiffractionSchedulerDecision:
    receiver_tiles: object | None
    tile_plan: object | None
    state_scheduler: str
    planner_strategy: str
    planner_backend: str | None
    planner_skip_reason: str | None
    selected_reason: str
    tile_task_count: int
    estimated_pair_count: int
    full_pair_count: int
    estimated_pair_ratio: float
    pair_chunk_budget: int | None
    cartesian_peak_pair_count: int
    tiled_peak_pair_count: int
    peak_pair_count_estimate: int
    estimated_launch_count: int


@dataclass(frozen=True)
class RadioMapReflectionSchedulerDecision:
    receiver_tiles: object | None
    tile_plan: object | None
    path_scheduler: str
    planner_backend: str | None
    selected_reason: str
    tile_task_count: int
    estimated_pair_count: int
    full_pair_count: int
    estimated_pair_ratio: float


def resolve_radio_map_receiver_tiles(*, grid=None, receiver_positions):
    if grid is None:
        return resolve_receiver_tiles(
            grid=None,
            receiver_positions=receiver_positions,
        )
    return resolve_receiver_tiles(
        grid=grid,
        plane_position=grid.position,
        grid_data=grid.get_coordinates(),
        receiver_positions=receiver_positions,
    )


def _radio_map_cartesian_launch_count(
    *,
    n_states: int,
    receiver_count: int,
    pair_chunk_budget: int | None,
) -> int:
    if n_states <= 0 or receiver_count <= 0:
        return 0
    if pair_chunk_budget is None or int(pair_chunk_budget) <= 0:
        return 1
    chunk_states = max(1, int(pair_chunk_budget) // int(receiver_count))
    return int((int(n_states) + chunk_states - 1) // chunk_states)


def _use_radio_map_diffraction_tiled_scheduler(
    tile_plan,
    *,
    full_pair_count: int,
    cartesian_launch_count: int,
    tiled_launch_count: int,
) -> bool:
    if tile_plan is None:
        return False
    tile_task_count = int(tile_plan.tile_task_count)
    estimated_pair_count = int(tile_plan.estimated_pair_count)
    if tile_task_count <= 0 or estimated_pair_count >= int(full_pair_count):
        return False
    # RadioMap scalar-power/coherent accumulation still pays more fixed work per
    # tile task than the FieldMonitor path. Only use tiling when the plan
    # removes a substantial fraction of the dense cartesian pair workload.
    if estimated_pair_count > int(full_pair_count * _RADIO_MAP_DIFFRACTION_TILED_MAX_PAIR_RATIO):
        return False
    if tile_task_count > int(_RADIO_MAP_DIFFRACTION_TILED_MAX_TASK_COUNT):
        return False
    if (
        cartesian_launch_count > 0
        and tiled_launch_count > 0
        and tiled_launch_count >= cartesian_launch_count
    ):
        return False
    return True


def _radio_map_diffraction_planner_precheck(
    *,
    receiver_tiles,
    n_states: int,
    receiver_count: int,
    pair_chunk_budget: int | None,
    full_pair_count: int,
) -> str | None:
    if receiver_tiles is None:
        return "receiver_tiling_unavailable"
    if int(receiver_tiles.n_tiles) <= 1:
        return "receiver_tiling_single_tile"
    if n_states <= 0 or receiver_count <= 0 or full_pair_count <= 0:
        return "empty_cartesian_workload"
    tile_rx_counts = tile_receiver_counts(receiver_tiles)
    if tile_rx_counts is None or int(dr.width(tile_rx_counts)) <= 0:
        return "receiver_tiling_empty"
    nonzero_tile_idx = dr.compress(tile_rx_counts > wt.UInt32(0))
    nonzero_tile_count = int(dr.width(nonzero_tile_idx))
    if nonzero_tile_count <= 1:
        return "receiver_tiling_single_nonempty_tile"
    positive_tile_rx_counts = dr.gather(wt.UInt32, tile_rx_counts, nonzero_tile_idx)
    min_tile_rx = int(dr.min(positive_tile_rx_counts)[0])
    max_tile_rx = int(dr.max(positive_tile_rx_counts)[0])
    best_case_pair_count = int(n_states * min_tile_rx)
    best_case_pair_ratio = (
        1.0 if full_pair_count <= 0 else float(best_case_pair_count) / float(full_pair_count)
    )
    cartesian_launch_count = _radio_map_cartesian_launch_count(
        n_states=n_states,
        receiver_count=receiver_count,
        pair_chunk_budget=pair_chunk_budget,
    )
    if pair_chunk_budget is None or int(pair_chunk_budget) <= 0:
        best_case_tiled_launch_count = 1
    else:
        max_chunk_states_per_tile = max(1, int(pair_chunk_budget) // max(1, max_tile_rx))
        best_case_tiled_launch_count = int(
            (int(n_states) + max_chunk_states_per_tile - 1) // max_chunk_states_per_tile
        )
    if (
        best_case_pair_ratio > float(_RADIO_MAP_DIFFRACTION_TILED_MAX_PAIR_RATIO)
        and cartesian_launch_count > 0
        and best_case_tiled_launch_count >= cartesian_launch_count
    ):
        return "precheck_cartesian_dominates"
    return None


def _radio_map_cartesian_peak_pair_count(
    *,
    n_states: int,
    receiver_count: int,
    pair_chunk_budget: int | None,
) -> int:
    if n_states <= 0 or receiver_count <= 0:
        return 0
    if pair_chunk_budget is None or int(pair_chunk_budget) <= 0:
        return int(n_states * receiver_count)
    chunk_states = max(1, int(pair_chunk_budget) // int(receiver_count))
    return int(min(int(n_states), chunk_states) * int(receiver_count))


def _radio_map_tiled_peak_pair_stats(
    tile_plan,
    receiver_tiles,
    *,
    pair_chunk_budget: int | None,
) -> tuple[int, int]:
    if tile_plan is None or receiver_tiles is None:
        return 0, 0
    if pair_chunk_budget is None or int(pair_chunk_budget) <= 0:
        return int(tile_plan.estimated_pair_count), int(tile_plan.tile_task_count)
    tile_task_counts_per_tile = tile_plan.tile_task_counts_per_tile
    if tile_task_counts_per_tile is None or int(dr.width(tile_task_counts_per_tile)) <= 0:
        return 0, 0
    tile_rx_counts = tile_receiver_counts(receiver_tiles)
    safe_tile_rx_counts = dr.maximum(tile_rx_counts, wt.UInt32(1))
    chunk_states_per_tile = dr.maximum(
        wt.UInt32(1),
        wt.UInt32(int(pair_chunk_budget)) // safe_tile_rx_counts,
    )
    peak_pairs_per_tile = dr.minimum(tile_task_counts_per_tile, chunk_states_per_tile) * tile_rx_counts
    launch_count_per_tile = (
        tile_task_counts_per_tile + chunk_states_per_tile - wt.UInt32(1)
    ) // chunk_states_per_tile
    tiled_peak_pair_count = int(dr.max(peak_pairs_per_tile)[0])
    estimated_launch_count = int(dr.sum(launch_count_per_tile)[0])
    return tiled_peak_pair_count, estimated_launch_count


def _use_radio_map_reflection_tiled_scheduler(tile_plan, *, full_pair_count: int) -> bool:
    if tile_plan is None:
        return False
    tile_task_count = int(tile_plan.tile_task_count)
    estimated_pair_count = int(tile_plan.estimated_pair_count)
    if tile_task_count <= 0 or estimated_pair_count >= int(full_pair_count):
        return False
    if estimated_pair_count > int(full_pair_count * _RADIO_MAP_REFLECTION_TILED_MAX_PAIR_RATIO):
        return False
    if tile_task_count > int(_RADIO_MAP_REFLECTION_TILED_MAX_TASK_COUNT):
        return False
    return True


def select_radio_map_diffraction_receiver_tiles(
    *,
    state_arrays,
    receiver_tiles,
    receiver_count: int,
    shadow_support_cutoff_db: float | None = None,
    pair_chunk_budget: int | None = None,
) -> RadioMapDiffractionSchedulerDecision:
    n_states = 0 if state_arrays is None else int(state_arrays["n_states"])
    full_pair_count = int(n_states * int(receiver_count))
    pair_chunk_budget_value = (
        None if pair_chunk_budget is None else max(1, int(pair_chunk_budget))
    )
    planner_skip_reason = _radio_map_diffraction_planner_precheck(
        receiver_tiles=receiver_tiles,
        n_states=n_states,
        receiver_count=int(receiver_count),
        pair_chunk_budget=pair_chunk_budget_value,
        full_pair_count=full_pair_count,
    )
    tile_plan = None
    if (
        planner_skip_reason is None
        and receiver_tiles is not None
        and int(receiver_tiles.n_tiles) > 1
        and n_states > 0
        and int(receiver_count) > 0
    ):
        tile_plan = build_utd_state_tile_plan(
            state_arrays=state_arrays,
            receiver_tiles=receiver_tiles,
            include_shadow_completion_band=True,
            shadow_support_cutoff_db=shadow_support_cutoff_db,
        )
    use_tiled = _use_radio_map_diffraction_tiled_scheduler(
        tile_plan,
        full_pair_count=full_pair_count,
        cartesian_launch_count=_radio_map_cartesian_launch_count(
            n_states=n_states,
            receiver_count=int(receiver_count),
            pair_chunk_budget=pair_chunk_budget_value,
        ),
        tiled_launch_count=_radio_map_tiled_peak_pair_stats(
            tile_plan,
            receiver_tiles,
            pair_chunk_budget=pair_chunk_budget_value,
        )[1],
    )
    tile_task_count = 0 if tile_plan is None else int(tile_plan.tile_task_count)
    estimated_pair_count = full_pair_count if tile_plan is None else int(tile_plan.estimated_pair_count)
    estimated_pair_ratio = (
        1.0 if full_pair_count <= 0 else float(estimated_pair_count) / float(full_pair_count)
    )
    cartesian_peak_pair_count = _radio_map_cartesian_peak_pair_count(
        n_states=n_states,
        receiver_count=int(receiver_count),
        pair_chunk_budget=pair_chunk_budget_value,
    )
    cartesian_launch_count = _radio_map_cartesian_launch_count(
        n_states=n_states,
        receiver_count=int(receiver_count),
        pair_chunk_budget=pair_chunk_budget_value,
    )
    tiled_peak_pair_count, estimated_launch_count = _radio_map_tiled_peak_pair_stats(
        tile_plan,
        receiver_tiles,
        pair_chunk_budget=pair_chunk_budget_value,
    )
    if planner_skip_reason is not None:
        selected_reason = planner_skip_reason
    elif tile_plan is None:
        selected_reason = "no_tile_plan"
    elif tile_task_count <= 0:
        selected_reason = "empty_tile_plan"
    elif estimated_pair_count >= int(full_pair_count):
        selected_reason = "tile_plan_does_not_reduce_pairs"
    elif estimated_pair_count > int(full_pair_count * _RADIO_MAP_DIFFRACTION_TILED_MAX_PAIR_RATIO):
        selected_reason = "pair_reduction_below_radiomap_threshold"
    elif tile_task_count > int(_RADIO_MAP_DIFFRACTION_TILED_MAX_TASK_COUNT):
        selected_reason = "tile_task_count_above_radiomap_threshold"
    elif (
        cartesian_launch_count > 0
        and estimated_launch_count > 0
        and estimated_launch_count >= cartesian_launch_count
    ):
        selected_reason = "tiled_launch_count_not_better_than_cartesian"
    elif use_tiled:
        selected_reason = "receiver_tiling_reduces_pair_work_enough"
    else:
        selected_reason = "cartesian_chunked_preferred"
    peak_pair_count_estimate = (
        int(tiled_peak_pair_count) if use_tiled else int(cartesian_peak_pair_count)
    )
    return RadioMapDiffractionSchedulerDecision(
        receiver_tiles=receiver_tiles if use_tiled else None,
        tile_plan=tile_plan if use_tiled else None,
        state_scheduler="receiver_tiled" if use_tiled else "cartesian_chunked",
        planner_strategy="receiver_tiled" if use_tiled else "cartesian_chunked",
        planner_backend=None if tile_plan is None else str(tile_plan.planner_backend),
        planner_skip_reason=planner_skip_reason,
        selected_reason=selected_reason,
        tile_task_count=tile_task_count,
        estimated_pair_count=estimated_pair_count,
        full_pair_count=full_pair_count,
        estimated_pair_ratio=estimated_pair_ratio,
        pair_chunk_budget=pair_chunk_budget_value,
        cartesian_peak_pair_count=cartesian_peak_pair_count,
        tiled_peak_pair_count=tiled_peak_pair_count,
        peak_pair_count_estimate=peak_pair_count_estimate,
        estimated_launch_count=(
            estimated_launch_count if use_tiled else int(cartesian_launch_count)
        ),
    )


def select_radio_map_reflection_family_tiles(
    *,
    paths,
    scene,
    receiver_tiles,
    receiver_count: int,
) -> RadioMapReflectionSchedulerDecision:
    n_paths = 0 if paths is None else int(paths.n_paths)
    full_pair_count = int(n_paths * int(receiver_count))
    tile_plan = None
    if (
        receiver_tiles is not None
        and int(receiver_tiles.n_tiles) > 1
        and n_paths > 0
        and int(receiver_count) > 0
    ):
        tile_plan = build_reflection_family_tile_plan(
            paths=paths,
            scene=scene,
            receiver_tiles=receiver_tiles,
        )
    use_tiled = _use_radio_map_reflection_tiled_scheduler(
        tile_plan,
        full_pair_count=full_pair_count,
    )
    tile_task_count = 0 if tile_plan is None else int(tile_plan.tile_task_count)
    estimated_pair_count = full_pair_count if tile_plan is None else int(tile_plan.estimated_pair_count)
    estimated_pair_ratio = (
        1.0 if full_pair_count <= 0 else float(estimated_pair_count) / float(full_pair_count)
    )
    if tile_plan is None:
        selected_reason = "no_tile_plan"
    elif tile_task_count <= 0:
        selected_reason = "empty_tile_plan"
    elif estimated_pair_count >= int(full_pair_count):
        selected_reason = "tile_plan_does_not_reduce_pairs"
    elif estimated_pair_count > int(full_pair_count * _RADIO_MAP_REFLECTION_TILED_MAX_PAIR_RATIO):
        selected_reason = "pair_reduction_below_radiomap_threshold"
    elif tile_task_count > int(_RADIO_MAP_REFLECTION_TILED_MAX_TASK_COUNT):
        selected_reason = "tile_task_count_above_radiomap_threshold"
    elif use_tiled:
        selected_reason = "receiver_tiling_reduces_pair_work_enough"
    else:
        selected_reason = "cartesian_chunked_preferred"
    return RadioMapReflectionSchedulerDecision(
        receiver_tiles=receiver_tiles if use_tiled else None,
        tile_plan=tile_plan if use_tiled else None,
        path_scheduler="receiver_tiled" if use_tiled else "cartesian_chunked",
        planner_backend=None if tile_plan is None else str(tile_plan.planner_backend),
        selected_reason=selected_reason,
        tile_task_count=tile_task_count,
        estimated_pair_count=estimated_pair_count,
        full_pair_count=full_pair_count,
        estimated_pair_ratio=estimated_pair_ratio,
    )


__all__ = [
    "RadioMapDiffractionSchedulerDecision",
    "RadioMapReflectionSchedulerDecision",
    "resolve_radio_map_receiver_tiles",
    "select_radio_map_diffraction_receiver_tiles",
    "select_radio_map_reflection_family_tiles",
]
