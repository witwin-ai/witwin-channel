"""Plot matched-isotropic RadioMapMonitor total, diffraction, and per-wedge maps."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
import time
if os.environ.get("WITWIN_CHANNEL_MAIN_SHOW", "0") != "1":
    os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import numpy as np
import witwin as wt

if not hasattr(wt, "Point3f"):
    import witwin.channel.backend as wt

from tests._scene_helpers import box_geometry, build_scene
from witwin.channel import (
    DiffractionExecutionConfig,
    Material,
    RadioMapMonitor,
    Tracer,
    draw_scene,
    scene_to_sionna_scene,
)
from witwin.channel.kernels.trace.packed_state import subset_state_arrays
from witwin.channel.monitors.path.collectors import collect_los_paths, collect_reflection_paths
from witwin.channel.monitors.radio_map.grid import RadioMapGrid
from witwin.channel.monitors.radio_map.backend import (
    _resolve_radio_map_accumulation_backend,
)
from witwin.channel.monitors.radio_map.diagnostics import (
    MATCHED_ISB_COMPLETION_BOUNDARY_HALF_FIELD,
    _accumulate_complex_by_rx,
    _add_complex_vector,
    _baseline_los_power,
    _scale_complex_vector,
    _vector_power,
    _zero_float,
)
from witwin.channel.monitors.radio_map.metadata import (
    _radio_map_diffraction_state_layout,
)
from witwin.channel.monitors.radio_map.deterministic.cell_accumulation import (
    accumulate_matched_isb_shadow_completion,
)
from witwin.channel.monitors.radio_map.samples import (
    _baseline_matched_isotropic_diffraction_power,
    _baseline_matched_isotropic_reflection_power,
    _trace_diffraction_raw_collections,
)


OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output"
DEFAULT_OUTPUT_PATH = OUTPUT_DIR / "radiomap_rotated_cube_coherent_total_diffraction.png"
DEFAULT_METADATA_PATH = OUTPUT_DIR / "radiomap_rotated_cube_coherent_total_diffraction.json"

ROTATED_CUBE_CENTER = (0.0, 0.0, 2.0)
ROTATED_CUBE_SIZE = 4.0
ROTATED_CUBE_ROTATION_RAD = float(np.deg2rad(-5.0))
ROTATED_CUBE_EPS_R = 1.0e4

TX_POS = (-5.0, 5.0, 1.5)
PLANE_Z = 1.5
BOUNDS = ((-8.0, 8.0), (-8.0, 8.0))

DEFAULT_GRID_SIZE = 512
DEFAULT_REFLECTION_N_RAYS = 8192
DEFAULT_DB_MIN = -90.0
DEFAULT_DB_MAX = -40.0
DEFAULT_DIFFRACTION_DB_MIN = -120.0
DEFAULT_DIFFRACTION_DB_MAX = -60.0
DEFAULT_DPI = 240
DEFAULT_RAY_MODE = "3d"
DEFAULT_ELEVATED_TX_Z = 10.0
DEFAULT_SIONNA_SAMPLES_PER_TX = 10_000_000


@dataclass(frozen=True)
class _RowMaps:
    label: str
    tx_pos: tuple[float, float, float]
    edges_2d: object
    edge_selection_mode: str
    runtime_edge_count: int
    raw_diffraction_power: np.ndarray
    raw_total_power: np.ndarray
    matched_isb_total_power: np.ndarray
    matched_isb_completion_power: np.ndarray
    wedge_powers: dict[int, np.ndarray]
    active_edge_indices: tuple[int, ...]
    runtime_metadata: dict[str, object]


def _build_scene(*, edge_selection_mode: str = "vertical_only"):
    return build_scene(
        box_geometry(
            center=ROTATED_CUBE_CENTER,
            size=ROTATED_CUBE_SIZE,
            rotation=ROTATED_CUBE_ROTATION_RAD,
        ),
        material=Material(eps_r=ROTATED_CUBE_EPS_R),
        edge_selection_mode=str(edge_selection_mode),
    )


def _build_tracer(scene, *, reflection_n_rays: int) -> Tracer:
    strict_drjit = DiffractionExecutionConfig.strict_drjit().to_dict()
    return Tracer(
        frequency=1.0e9,
        scene=scene,
        config={
            "trace": {
                "reflection_n_rays": int(reflection_n_rays),
                "reflection_max_bounces": 1,
                "reflection_coef": 1.0,
                "enable_rd_diffraction": True,
                "max_diffractions": 1,
                "use_scene_materials_for_reflection": True,
                "use_scene_materials_for_diffraction": True,
                "reflection_field_backend": "drjit",
                "diffraction_execution": strict_drjit,
                "tx_polarization": (1.0, 0.0, 0.0),
            }
        },
    )


def _build_monitor(*, grid_size: int, ray_mode: str) -> RadioMapMonitor:
    return RadioMapMonitor(
        "radiomap_rotated_cube_coherent",
        axis="z",
        position=PLANE_Z,
        bounds=BOUNDS,
        grid_shape=(int(grid_size), int(grid_size)),
        metric="path_gain",
        combine_mode="coherent",
        receiver_model="matched_isotropic",
        quadrature_mode="center",
        ray_mode=str(ray_mode),
        max_diffractions=1,
        accumulation_backend="baseline",
        shadow_boundary_mode="none",
    )


def _to_numpy_grid(values, *, tensor_shape: tuple[int, int]) -> np.ndarray:
    if hasattr(values, "detach") and hasattr(values, "cpu"):
        values_np = values.detach().cpu().numpy()
    else:
        values_np = np.asarray(values)
    values_np = np.asarray(values_np, dtype=np.float32)
    if values_np.shape != tensor_shape:
        values_np = values_np.reshape(tensor_shape)
    return values_np


def _db_map(values, *, floor_db: float) -> np.ndarray:
    floor = 10.0 ** (float(floor_db) / 10.0)
    values_np = np.asarray(values, dtype=np.float32)
    return 10.0 * np.log10(np.maximum(values_np, floor))


def _active_diffraction_edge_indices(diffraction_raw_collections) -> tuple[int, ...]:
    active_edge_indices: set[int] = set()
    for raw in diffraction_raw_collections:
        state_arrays = raw.get("state_arrays")
        if state_arrays is None or int(state_arrays["n_states"]) <= 0:
            continue
        active_edge_indices.update(
            int(edge_idx)
            for edge_idx in np.asarray(state_arrays["edge_idx"], dtype=np.int32).reshape(-1)
        )
    return tuple(sorted(active_edge_indices))


def _subset_raw_collection_by_edge(raw: dict[str, object], *, edge_idx: int) -> dict[str, object] | None:
    state_arrays = raw.get("state_arrays")
    if state_arrays is None or int(state_arrays["n_states"]) <= 0:
        return None
    subset = subset_state_arrays(
        state_arrays,
        state_arrays["edge_idx"] == wt.UInt32(int(edge_idx)),
    )
    if int(subset["n_states"]) <= 0:
        return None
    return {
        "state_arrays": subset,
        "rx_positions": raw["rx_positions"],
        "radio_map_receiver_index_map": raw["radio_map_receiver_index_map"],
    }


def _replay_wedge_diffraction_power(
    *,
    diffraction_raw_collections,
    edge_idx: int,
    scene,
    config,
    n_rx: int,
    los_reference_vector,
    reflection_reference_vector,
):
    wedge_raw_collections = []
    for raw in diffraction_raw_collections:
        subset = _subset_raw_collection_by_edge(raw, edge_idx=edge_idx)
        if subset is not None:
            wedge_raw_collections.append(subset)
    if len(wedge_raw_collections) == 0:
        return _zero_float(n_rx)
    wedge_power, _, _, _, _ = _baseline_matched_isotropic_diffraction_power(
        diffraction_raw_collections=wedge_raw_collections,
        scene=scene,
        config=config,
        n_rx=n_rx,
        los_reference_vector=los_reference_vector,
        reflection_reference_vector=reflection_reference_vector,
    )
    return wedge_power


def _build_row_maps(
    scene,
    *,
    label: str,
    tx_pos: tuple[float, float, float],
    grid_size: int,
    reflection_n_rays: int,
    ray_mode: str,
) -> _RowMaps:
    tracer = _build_tracer(scene, reflection_n_rays=reflection_n_rays)
    monitor = _build_monitor(grid_size=grid_size, ray_mode=ray_mode)
    tx_point = wt.Point3f(*tx_pos)
    config = tracer._resolved_trace_config
    grid = RadioMapGrid.from_monitor(monitor, default_cell_size=config.cell_size)
    resolved_accumulation_backend = _resolve_radio_map_accumulation_backend(
        requested_backend=monitor.accumulation_backend,
        monitor=monitor,
        grid=grid,
        config=config,
        tx_pos=tx_point,
        scene=scene,
    )
    if resolved_accumulation_backend != "baseline":
        raise RuntimeError(
            "Per-wedge matched-isotropic replay currently expects the baseline radio-map backend."
        )
    solver_controls = tracer._resolve_monitor_solver_controls(
        monitor,
        execution_intent="radio_map_coherent",
    )
    state_layout = _radio_map_diffraction_state_layout(resolved_accumulation_backend)
    n_rx = int(grid.n_cells)
    edge_info = scene.get_edge_data(PLANE_Z)

    wedge_power_by_edge: dict[int, object] = {}
    active_edge_indices: set[int] = set()
    reflection_detail = None
    state_preparation_hits = 0
    state_preparation_misses = 0
    los_vector_total = None
    reflection_vector_total = None
    diffraction_vector_total = None

    for sample_set in grid.sample_sets:
        sample_weight = float(sample_set.weight)
        los_raw = collect_los_paths(
            scene=scene,
            rx_positions=sample_set.positions,
            tx_pos=tx_point,
            wavelength=config.wavelength,
            k=config.k,
            tx_polarization=config.tx_polarization,
            rx_polarization=config.rx_polarization,
        )
        los_coherent = _accumulate_complex_by_rx(los_raw, n_rx=n_rx)
        _, los_field_vector = _baseline_los_power(
            monitor=monitor,
            sample_positions=sample_set.positions,
            tx_pos=tx_point,
            config=config,
            los_coherent=los_coherent,
        )

        _, reflection_detail = collect_reflection_paths(
            scene=scene,
            rx_positions=sample_set.positions,
            tx_pos=tx_point,
            wavelength=config.wavelength,
            k=config.k,
            n_rays=solver_controls["effective"]["reflection_n_rays"],
            max_reflections=solver_controls["effective"]["reflection_max_bounces"],
            mode=monitor.ray_mode,
            tx_polarization=config.tx_polarization,
            rx_polarization=config.rx_polarization,
            reflection_coef=config.reflection_coef,
            min_ray_contribution_threshold=config.min_ray_contribution_threshold,
            reflection_relative_permittivity=config.reflection_relative_permittivity,
            reflection_conductivity=config.reflection_conductivity,
            reflection_material=config.reflection_material,
            use_scene_materials=config.use_scene_materials_for_reflection,
            return_geometry=False,
            reflection_detail=reflection_detail,
        )
        _, reflection_vector_coherent, _ = _baseline_matched_isotropic_reflection_power(
            sample_positions=sample_set.positions,
            scene=scene,
            config=config,
            reflection_detail=reflection_detail,
        )

        diffraction_raw_collections, _, runtime_reuse = _trace_diffraction_raw_collections(
            sample_positions=sample_set.positions,
            tx_pos=tx_point,
            scene=scene,
            config=config,
            solver_controls=solver_controls,
            monitor=monitor,
            reflection_detail=reflection_detail,
            persistent_diffraction_state_cache=None,
            local_diffraction_state_cache={},
            diffraction_state_cache_key_fn=None,
            state_layout=state_layout,
        )
        state_preparation_hits += int(runtime_reuse["state_preparation_hits"])
        state_preparation_misses += int(runtime_reuse["state_preparation_misses"])

        _, _, _, diffraction_vector_coherent, _ = (
            _baseline_matched_isotropic_diffraction_power(
                diffraction_raw_collections=diffraction_raw_collections,
                scene=scene,
                config=config,
                n_rx=n_rx,
                los_reference_vector=los_field_vector,
                reflection_reference_vector=reflection_vector_coherent,
            )
        )
        los_vector_total = _add_complex_vector(
            los_vector_total,
            _scale_complex_vector(los_field_vector, sample_weight),
        )
        reflection_vector_total = _add_complex_vector(
            reflection_vector_total,
            _scale_complex_vector(reflection_vector_coherent, sample_weight),
        )
        diffraction_vector_total = _add_complex_vector(
            diffraction_vector_total,
            _scale_complex_vector(diffraction_vector_coherent, sample_weight),
        )

        sample_active_edges = _active_diffraction_edge_indices(diffraction_raw_collections)
        active_edge_indices.update(sample_active_edges)
        for edge_idx in sample_active_edges:
            wedge_power_sample = _replay_wedge_diffraction_power(
                diffraction_raw_collections=diffraction_raw_collections,
                edge_idx=edge_idx,
                scene=scene,
                config=config,
                n_rx=n_rx,
                los_reference_vector=los_field_vector,
                reflection_reference_vector=reflection_vector_coherent,
            )
            accumulated = wedge_power_by_edge.get(edge_idx)
            if accumulated is None:
                accumulated = _zero_float(n_rx)
            wedge_power_by_edge[edge_idx] = accumulated + wedge_power_sample * sample_weight

    tensor_shape = grid.tensor_shape
    raw_total_vector = _add_complex_vector(
        _add_complex_vector(los_vector_total, reflection_vector_total),
        diffraction_vector_total,
    )
    raw_total_power = _zero_float(n_rx) if raw_total_vector is None else _vector_power(raw_total_vector)
    raw_diffraction_power = (
        _zero_float(n_rx) if diffraction_vector_total is None else _vector_power(diffraction_vector_total)
    )
    matched_isb_completion_payload = accumulate_matched_isb_shadow_completion(
        rx_pos=grid.cell_centers,
        scene=scene,
        tx_pos=tx_point,
        wavelength=config.wavelength,
        k=config.k,
        tx_polarization=config.tx_polarization,
        rx_polarization=config.rx_polarization,
        los_vector_coherent=los_vector_total,
        raw_transition_vector=_add_complex_vector(los_vector_total, diffraction_vector_total),
    )
    matched_isb_total_vector = _add_complex_vector(
        raw_total_vector,
        matched_isb_completion_payload["vector_coherent"],
    )
    matched_isb_total_power = (
        _zero_float(n_rx)
        if matched_isb_total_vector is None
        else _vector_power(matched_isb_total_vector)
    )
    matched_isb_completion_weight = _to_numpy_grid(
        matched_isb_completion_payload["incident_weight"],
        tensor_shape=tensor_shape,
    )
    matched_isb_transition_magnitude = _to_numpy_grid(
        matched_isb_completion_payload["transition_magnitude"],
        tensor_shape=tensor_shape,
    )
    return _RowMaps(
        label=label,
        tx_pos=tuple(float(value) for value in tx_pos),
        edges_2d=edge_info["edges_2d"],
        edge_selection_mode=str(scene.edge_selection_mode),
        runtime_edge_count=int(edge_info["edge_data"]["n_edges"]),
        raw_total_power=_to_numpy_grid(
            raw_total_power,
            tensor_shape=tensor_shape,
        ),
        raw_diffraction_power=_to_numpy_grid(
            raw_diffraction_power,
            tensor_shape=tensor_shape,
        ),
        matched_isb_total_power=_to_numpy_grid(
            matched_isb_total_power,
            tensor_shape=tensor_shape,
        ),
        matched_isb_completion_power=_to_numpy_grid(
            matched_isb_completion_payload["power"],
            tensor_shape=tensor_shape,
        ),
        wedge_powers={
            int(edge_idx): _to_numpy_grid(power, tensor_shape=tensor_shape)
            for edge_idx, power in wedge_power_by_edge.items()
        },
        active_edge_indices=tuple(sorted(active_edge_indices)),
        runtime_metadata={
            "grid_shape": [int(grid.grid_shape[0]), int(grid.grid_shape[1])],
            "tensor_shape": [int(grid.tensor_shape[0]), int(grid.tensor_shape[1])],
            "resolved_accumulation_backend": str(resolved_accumulation_backend),
            "state_layout": str(state_layout),
            "state_preparation_hits": int(state_preparation_hits),
            "state_preparation_misses": int(state_preparation_misses),
            "raw_shadow_boundary_mode": str(monitor.shadow_boundary_mode),
            "matched_isb_shadow_boundary_mode": "matched_isb_completion",
            "matched_isb_metric_contract": (
                "squared_norm_of_matched_isotropic_vector_coherent_sum_plus_isb_visibility_"
                "completion_weighted_by_fixed_cell_quadrature"
            ),
            "matched_isb_completion_model": (
                "matched_isotropic_isb_scene_edge_complex_transition_residual_completion"
            ),
            "matched_isb_visibility_model": (
                "scene_edge_incident_transition_weighted_average_with_incident_gated_direct_mode_"
                "residual_matching"
            ),
            "matched_isb_boundary_half_field_limit": float(
                MATCHED_ISB_COMPLETION_BOUNDARY_HALF_FIELD
            ),
            "matched_isb_completion_weight_max": float(np.max(matched_isb_completion_weight)),
            "matched_isb_transition_magnitude_max": float(
                np.max(matched_isb_transition_magnitude)
            ),
            "edge_selection_summary": dict(getattr(scene, "edge_selection_summary", {}) or {}),
            "vertical_ratio": float(scene.vertical_ratio),
        },
    )


def _decorate_axis(
    ax,
    *,
    title: str | None,
    edges,
    tx_pos: tuple[float, float, float],
    show_xlabel: bool,
    show_ylabel: bool,
):
    if title is not None:
        ax.set_title(title, fontsize=12)
    ax.set_xlabel("x (m)" if show_xlabel else "")
    ax.set_ylabel("y (m)" if show_ylabel else "")
    draw_scene(ax, edges, tx_pos, BOUNDS[0], BOUNDS[1])


def _build_sionna_power_map(
    scene,
    *,
    tx_pos: tuple[float, float, float],
    grid_size: int,
    samples_per_tx: int,
    los: bool,
    specular_reflection: bool,
    diffraction: bool,
) -> tuple[np.ndarray, dict[str, object]]:
    import importlib

    conversion = scene_to_sionna_scene(scene, prefer_local=True)
    rt = conversion.rt
    mi = importlib.import_module("mitsuba")
    sionna_scene = conversion.scene
    sionna_scene.frequency = 1.0e9
    sionna_scene.tx_array = rt.PlanarArray(
        num_rows=1,
        num_cols=1,
        pattern="iso",
        polarization="V",
    )
    sionna_scene.add(
        rt.Transmitter(
            "tx",
            position=mi.Point3f(*tx_pos),
            power_dbm=0.0,
        )
    )
    solver = rt.RadioMapSolver()
    span_x = float(BOUNDS[0][1] - BOUNDS[0][0])
    span_y = float(BOUNDS[1][1] - BOUNDS[1][0])
    t0 = time.perf_counter()
    result = solver(
        sionna_scene,
        center=mi.Point3f(
            0.5 * (float(BOUNDS[0][0]) + float(BOUNDS[0][1])),
            0.5 * (float(BOUNDS[1][0]) + float(BOUNDS[1][1])),
            float(PLANE_Z),
        ),
        orientation=mi.Point3f(0.0, 0.0, 0.0),
        size=mi.Point2f(span_x, span_y),
        cell_size=mi.Point2f(span_x / float(grid_size), span_y / float(grid_size)),
        samples_per_tx=int(samples_per_tx),
        max_depth=1,
        los=bool(los),
        specular_reflection=bool(specular_reflection),
        diffraction=bool(diffraction),
        edge_diffraction=False,
        refraction=False,
        seed=7,
    )
    elapsed = time.perf_counter() - t0
    return (
        _to_numpy_grid(
            np.asarray(result.path_gain, dtype=np.float32)[0],
            tensor_shape=(int(grid_size), int(grid_size)),
        ),
        {
            "source": str(conversion.source),
            "runtime_seconds": float(elapsed),
            "samples_per_tx": int(samples_per_tx),
            "max_depth": 1,
            "los": bool(los),
            "diffraction": bool(diffraction),
            "specular_reflection": bool(specular_reflection),
        },
    )


def save_radiomap_rotated_cube_coherent_total_diffraction_figure(
    output_path: Path,
    *,
    metadata_path: Path | None = None,
    grid_size: int = DEFAULT_GRID_SIZE,
    reflection_n_rays: int = DEFAULT_REFLECTION_N_RAYS,
    floor_db: float = DEFAULT_DB_MIN,
    vmax_db: float = DEFAULT_DB_MAX,
    diffraction_floor_db: float = DEFAULT_DIFFRACTION_DB_MIN,
    diffraction_vmax_db: float = DEFAULT_DIFFRACTION_DB_MAX,
    dpi: int = DEFAULT_DPI,
    ray_mode: str = DEFAULT_RAY_MODE,
    elevated_tx_z: float = DEFAULT_ELEVATED_TX_Z,
    sionna_samples_per_tx: int = DEFAULT_SIONNA_SAMPLES_PER_TX,
) -> tuple[Path, Path]:
    elevated_tx_pos = (TX_POS[0], TX_POS[1], float(elevated_tx_z))
    row_specs = (
        ("Coplanar", TX_POS, "vertical_only"),
        ("Elevated", elevated_tx_pos, "all_edges"),
    )
    row_scenes = {
        label: _build_scene(edge_selection_mode=edge_selection_mode)
        for label, _, edge_selection_mode in row_specs
    }
    rows = [
        _build_row_maps(
            row_scenes[label],
            label=label,
            tx_pos=tx_pos,
            grid_size=grid_size,
            reflection_n_rays=reflection_n_rays,
            ray_mode=ray_mode,
        )
        for label, tx_pos, edge_selection_mode in row_specs
    ]
    sionna_total_power, sionna_total_runtime = _build_sionna_power_map(
        row_scenes["Elevated"],
        tx_pos=elevated_tx_pos,
        grid_size=grid_size,
        samples_per_tx=sionna_samples_per_tx,
        los=True,
        specular_reflection=True,
        diffraction=True,
    )
    sionna_diffraction_power, sionna_diff_runtime = _build_sionna_power_map(
        row_scenes["Elevated"],
        tx_pos=elevated_tx_pos,
        grid_size=grid_size,
        samples_per_tx=sionna_samples_per_tx,
        los=False,
        specular_reflection=False,
        diffraction=True,
    )
    extent = (
        float(BOUNDS[0][0]),
        float(BOUNDS[0][1]),
        float(BOUNDS[1][0]),
        float(BOUNDS[1][1]),
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    resolved_metadata_path = metadata_path if metadata_path is not None else output_path.with_suffix(".json")
    resolved_metadata_path.parent.mkdir(parents=True, exist_ok=True)

    n_cols = max(max(4 + int(row.runtime_edge_count) for row in rows), 4)
    n_rows = len(rows) + 1
    fig_width = max(12.0, 2.5 * float(n_cols))
    fig_height = max(10.8, 3.5 * float(n_rows))
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(fig_width, fig_height),
        constrained_layout=True,
        squeeze=False,
    )
    total_mappable = None
    diffraction_mappable = None
    visible_total_axes = []
    visible_diffraction_axes = []
    row_metadata = []

    for row_index, row in enumerate(rows):
        panel_specs = [
            ("Raw Total", row.raw_total_power, float(floor_db), float(vmax_db)),
            ("Matched ISB Total", row.matched_isb_total_power, float(floor_db), float(vmax_db)),
            ("Raw Diffraction", row.raw_diffraction_power, float(diffraction_floor_db), float(diffraction_vmax_db)),
            (
                "ISB Completion",
                row.matched_isb_completion_power,
                float(diffraction_floor_db),
                float(diffraction_vmax_db),
            ),
        ]
        for wedge_index, edge_idx in enumerate(range(row.runtime_edge_count), start=1):
            panel_specs.append(
                (
                    f"Wedge {wedge_index}",
                    row.wedge_powers.get(
                        edge_idx,
                        np.zeros_like(row.raw_diffraction_power, dtype=np.float32),
                    ),
                    float(diffraction_floor_db),
                    float(diffraction_vmax_db),
                )
            )

        raw_total_db_map = _db_map(row.raw_total_power, floor_db=floor_db)
        matched_isb_total_db_map = _db_map(row.matched_isb_total_power, floor_db=floor_db)
        raw_diffraction_db_map = _db_map(row.raw_diffraction_power, floor_db=diffraction_floor_db)
        matched_isb_completion_db_map = _db_map(
            row.matched_isb_completion_power,
            floor_db=diffraction_floor_db,
        )
        row_metadata.append(
            {
                "label": row.label,
                "tx_pos": [float(value) for value in row.tx_pos],
                "edge_selection_mode": str(row.edge_selection_mode),
                "runtime_edge_count": int(row.runtime_edge_count),
                "active_runtime_edge_indices": [int(value) for value in row.active_edge_indices],
                "source": "witwin",
                "raw_total_max_db": float(np.max(raw_total_db_map)),
                "raw_total_min_db": float(np.min(raw_total_db_map)),
                "matched_isb_total_max_db": float(np.max(matched_isb_total_db_map)),
                "matched_isb_total_min_db": float(np.min(matched_isb_total_db_map)),
                "raw_diffraction_max_db": float(np.max(raw_diffraction_db_map)),
                "raw_diffraction_min_db": float(np.min(raw_diffraction_db_map)),
                "matched_isb_completion_max_db": float(np.max(matched_isb_completion_db_map)),
                "matched_isb_completion_min_db": float(np.min(matched_isb_completion_db_map)),
                "runtime": dict(row.runtime_metadata),
                "wedge_panels": [
                    {
                        "panel_index": int(4 + wedge_index),
                        "label": f"Wedge {wedge_index + 1}",
                        "runtime_edge_idx": int(edge_idx),
                    }
                    for wedge_index, edge_idx in enumerate(range(row.runtime_edge_count))
                ],
            }
        )

        for col_index, (title, power_map, panel_min_db, panel_max_db) in enumerate(panel_specs):
            ax = axes[row_index, col_index]
            image = _db_map(
                power_map,
                floor_db=(floor_db if col_index < 2 else diffraction_floor_db),
            )
            im = ax.imshow(
                image,
                origin="lower",
                extent=extent,
                cmap="inferno",
                vmin=panel_min_db,
                vmax=panel_max_db,
            )
            _decorate_axis(
                ax,
                title=title,
                edges=row.edges_2d,
                tx_pos=row.tx_pos,
                show_xlabel=(row_index == len(rows) - 1),
                show_ylabel=(col_index == 0),
            )
            if col_index == 0:
                ax.text(
                    -0.34,
                    0.5,
                    (
                        f"{row.label}\n"
                        f"TX z={row.tx_pos[2]:.1f} m\n"
                        f"{row.edge_selection_mode}"
                    ),
                    transform=ax.transAxes,
                    rotation=90,
                    va="center",
                    ha="center",
                    fontsize=11,
                )
                total_mappable = im
            if col_index < 2:
                total_mappable = im
                visible_total_axes.append(ax)
            else:
                diffraction_mappable = im
                visible_diffraction_axes.append(ax)

        for col_index in range(len(panel_specs), n_cols):
            axes[row_index, col_index].set_axis_off()

    sionna_row_index = len(rows)
    sionna_ax = axes[sionna_row_index, 0]
    sionna_total_db_map = _db_map(sionna_total_power, floor_db=floor_db)
    sionna_im = sionna_ax.imshow(
        sionna_total_db_map,
        origin="lower",
        extent=extent,
        cmap="inferno",
        vmin=float(floor_db),
        vmax=float(vmax_db),
    )
    _decorate_axis(
        sionna_ax,
        title="Sionna Total",
        edges=rows[-1].edges_2d,
        tx_pos=elevated_tx_pos,
        show_xlabel=True,
        show_ylabel=True,
    )
    sionna_ax.text(
        -0.34,
        0.5,
        (
            "Sionna\n"
            f"TX z={elevated_tx_pos[2]:.1f} m\n"
            f"{sionna_total_runtime['source']}"
        ),
        transform=sionna_ax.transAxes,
        rotation=90,
        va="center",
        ha="center",
        fontsize=11,
    )
    total_mappable = sionna_im
    visible_total_axes.append(sionna_ax)
    axes[sionna_row_index, 1].set_axis_off()
    sionna_diff_ax = axes[sionna_row_index, 2]
    sionna_diff_db_map = _db_map(
        sionna_diffraction_power,
        floor_db=diffraction_floor_db,
    )
    sionna_diff_im = sionna_diff_ax.imshow(
        sionna_diff_db_map,
        origin="lower",
        extent=extent,
        cmap="inferno",
        vmin=float(diffraction_floor_db),
        vmax=float(diffraction_vmax_db),
    )
    _decorate_axis(
        sionna_diff_ax,
        title="Sionna Diffraction",
        edges=rows[-1].edges_2d,
        tx_pos=elevated_tx_pos,
        show_xlabel=True,
        show_ylabel=False,
    )
    diffraction_mappable = sionna_diff_im
    visible_diffraction_axes.append(sionna_diff_ax)
    row_metadata.append(
        {
            "label": "Sionna",
            "tx_pos": [float(value) for value in elevated_tx_pos],
            "edge_selection_mode": str(rows[-1].edge_selection_mode),
            "runtime_edge_count": 0,
            "source": "sionna",
            "raw_total_max_db": float(np.max(sionna_total_db_map)),
            "raw_total_min_db": float(np.min(sionna_total_db_map)),
            "raw_diffraction_max_db": float(np.max(sionna_diff_db_map)),
            "raw_diffraction_min_db": float(np.min(sionna_diff_db_map)),
            "runtime": {
                "total": dict(sionna_total_runtime),
                "diffraction_only": dict(sionna_diff_runtime),
            },
            "wedge_panels": [],
        }
    )
    axes[sionna_row_index, 3].set_axis_off()
    for col_index in range(4, n_cols):
        axes[sionna_row_index, col_index].set_axis_off()

    if total_mappable is not None and len(visible_total_axes) > 0:
        fig.colorbar(
            total_mappable,
            ax=visible_total_axes,
            fraction=0.03,
            pad=0.02,
        )
    if diffraction_mappable is not None:
        fig.colorbar(
            diffraction_mappable,
            ax=visible_diffraction_axes,
            fraction=0.03,
            pad=0.02,
        )

    fig.suptitle(
        (
            "Rotated Cube RadioMap Components\n"
            "raw coherent total vs matched_isb_completion total + raw diffraction/wedge replay + "
            f"ray_mode={ray_mode} + max_diffractions=1 + elevated Sionna raw-total reference"
        ),
        fontsize=14,
    )
    fig.savefig(output_path, dpi=int(dpi))
    plt.close(fig)

    metadata = {
        "image": str(output_path),
        "grid_shape": [int(grid_size), int(grid_size)],
        "plane_z": float(PLANE_Z),
        "bounds": [[float(a), float(b)] for (a, b) in BOUNDS],
        "rotation_rad": float(ROTATED_CUBE_ROTATION_RAD),
        "material_eps_r": float(ROTATED_CUBE_EPS_R),
        "reflection_n_rays": int(reflection_n_rays),
        "dpi": int(dpi),
        "ray_mode": str(ray_mode),
        "finite_edge_mode": "finite_wedge",
        "combine_mode": "coherent",
        "receiver_model": "matched_isotropic",
        "shadow_boundary_mode": {
            "raw_total": "none",
            "matched_isb_total": "matched_isb_completion",
            "raw_diffraction_and_wedges": "none",
        },
        "accumulation_backend": "baseline",
        "max_diffractions": 1,
        "path_gain_contract": {
            "raw_total": "squared_norm_of_matched_isotropic_vector_coherent_sum_weighted_by_fixed_cell_quadrature",
            "matched_isb_total": (
                "squared_norm_of_matched_isotropic_vector_coherent_sum_plus_isb_visibility_"
                "completion_weighted_by_fixed_cell_quadrature"
            ),
            "raw_diffraction_and_wedges": "matched_isotropic_raw_component_replay",
        },
        "panel_layout": [
            "Raw Total",
            "Matched ISB Total",
            "Raw Diffraction",
            "ISB Completion",
            "Wedge 1..N",
        ],
        "matched_isb_completion_parameters": {
            "shadow_boundary_half_field_limit": float(
                MATCHED_ISB_COMPLETION_BOUNDARY_HALF_FIELD
            ),
        },
        "sionna_samples_per_tx": int(sionna_samples_per_tx),
        "sionna_source": str(sionna_total_runtime["source"]),
        "total_display_db_range": [float(floor_db), float(vmax_db)],
        "diffraction_display_db_range": [float(diffraction_floor_db), float(diffraction_vmax_db)],
        "figure_max_runtime_edge_count": int(max(row.runtime_edge_count for row in rows)),
        "rows": row_metadata,
    }
    resolved_metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return output_path, resolved_metadata_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA_PATH)
    parser.add_argument("--grid-size", type=int, default=DEFAULT_GRID_SIZE)
    parser.add_argument("--n-rays", type=int, default=DEFAULT_REFLECTION_N_RAYS)
    parser.add_argument("--db-min", type=float, default=DEFAULT_DB_MIN)
    parser.add_argument("--db-max", type=float, default=DEFAULT_DB_MAX)
    parser.add_argument("--diffraction-db-min", type=float, default=DEFAULT_DIFFRACTION_DB_MIN)
    parser.add_argument("--diffraction-db-max", type=float, default=DEFAULT_DIFFRACTION_DB_MAX)
    parser.add_argument("--dpi", type=int, default=DEFAULT_DPI)
    parser.add_argument("--ray-mode", type=str, default=DEFAULT_RAY_MODE)
    parser.add_argument("--elevated-tx-z", type=float, default=DEFAULT_ELEVATED_TX_Z)
    parser.add_argument("--sionna-samples-per-tx", type=int, default=DEFAULT_SIONNA_SAMPLES_PER_TX)
    args = parser.parse_args()
    save_radiomap_rotated_cube_coherent_total_diffraction_figure(
        args.output,
        metadata_path=args.metadata,
        grid_size=args.grid_size,
        reflection_n_rays=args.n_rays,
        floor_db=args.db_min,
        vmax_db=args.db_max,
        diffraction_floor_db=args.diffraction_db_min,
        diffraction_vmax_db=args.diffraction_db_max,
        dpi=args.dpi,
        ray_mode=args.ray_mode,
        elevated_tx_z=args.elevated_tx_z,
        sionna_samples_per_tx=args.sionna_samples_per_tx,
    )


if __name__ == "__main__":
    main()


__all__ = ["save_radiomap_rotated_cube_coherent_total_diffraction_figure"]
