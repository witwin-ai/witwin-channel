"""Plot radiomap per-component power-gradient contributions for the three-cube scene."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path

import drjit as dr
import numpy as np
import witwin as wt

if os.environ.get("WITWIN_CHANNEL_MAIN_SHOW", "0") != "1":
    os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt

from samples.save_multipath_main_component_gradient_figure import (
    COMPONENT_LABELS,
    PRIMITIVE_COMPONENTS,
)
from tests.main.plot_multipath_components import CUBE1_BASE_CENTER, build_scene_for_cube1_x
from tests.main.plot_radiomap_gradients_three_cubes import (
    DEFAULT_ACCUMULATION_BACKEND,
    DEFAULT_COMBINE_MODE,
    DEFAULT_FD_STEP,
    DEFAULT_MAX_DIFFRACTIONS,
    DEFAULT_RECEIVER_MODEL,
    DEFAULT_SHADOW_BOUNDARY_MODE,
    DEFAULT_TRACE_SEED,
    _GRAD_FLAGS,
    _make_monitor,
    _make_tracer,
    _safe_correlation,
    parameter_config,
)
from tests.main.plot_radiomap_sionna_three_cubes import (
    DEFAULT_BOUNDS,
    DEFAULT_GRID_SIZE,
    DEFAULT_N_RAYS,
    DEFAULT_PLANE_Z,
    DEFAULT_TX_POS,
    _decorate_axis,
    _output_dir,
)
from witwin.channel.config import ReflectionSuffixConfig
from witwin.channel.kernels.trace.packed_state import subset_state_arrays
from witwin.channel.monitors.radio_map.deterministic.coherent import (
    accumulate_radio_map_diffraction_coherent,
    accumulate_radio_map_los_coherent,
    accumulate_radio_map_reflection_coherent,
)
from witwin.channel.monitors.radio_map.deterministic.samples import (
    _discover_radio_map_reflection_detail,
)
from witwin.channel.monitors.radio_map.grid import (
    AxisAlignedRadioMapNativeGrid,
    RadioMapGrid,
)
from witwin.channel.monitors.radio_map.diagnostics import _add_complex_vector
from witwin.channel.trace.cache import radio_map_execution_intent
from witwin.channel.trace.diffraction import _prepare_diffraction_state_arrays
from witwin.channel.trace.diffraction.constants import (
    APPROX_MODE_DIRECT_FIRST_ORDER,
    APPROX_MODE_RECURSIVE_DIFFRACTION,
    APPROX_MODE_SAMPLED_INSERTED_REFLECTION,
    APPROX_MODE_SAMPLED_INSERTED_REFLECTION_CHAIN,
    APPROX_MODE_SAMPLED_REFLECTION_PREFIX,
    APPROX_MODE_SAMPLED_REFLECTION_PREFIX_CHAIN,
    OWNERSHIP_DIRECT_DIFFRACTION,
    OWNERSHIP_MIXED_DIFFRACTION,
    _ownership_code_from_depths,
)
from witwin.channel.trace.diffraction.suffix import trace_reflected_suffix_from_edge_states
from witwin.channel.utils.polarization import (
    effective_rx_polarization,
    project_real_polarization_to_ray,
    vector_from_scalar_and_real_direction,
    vector_zero,
)


DEFAULT_OUTPUT_PREFIX = _output_dir() / "radiomap_cube1_x_power_components"
_PARAMETERS = ("tx_x", "cube1_x")
_DERIVED_COMPONENTS = ("a_dif_direct", "a_dif_mixed", "a_dif", "a_tot")
_DISPLAY_COMPONENTS = tuple(
    component_name
    for component_name in (PRIMITIVE_COMPONENTS + _DERIVED_COMPONENTS)
    if component_name not in ("a_los", "a_ref")
)
_DIFF_DB_LIMIT = 50.0


@dataclass(frozen=True)
class RadiomapPowerComponentSummary:
    parameter: str
    grid_size: int
    bounds: tuple[tuple[float, float], tuple[float, float]]
    plane_z: float
    tx_pos: tuple[float, float, float]
    n_rays: int
    fd_step: float
    combine_mode: str
    receiver_model: str
    shadow_boundary_mode: str
    accumulation_backend_requested: str
    max_diffractions: int
    timings_seconds: dict[str, float]


def _diffraction_anchor_coordinate(axis: str, tx_pos, position: float) -> float:
    return float(position) if str(axis) == "z" else float(tx_pos.z)


def _time_call(func, /, *args, **kwargs):
    import time

    started = time.perf_counter()
    value = func(*args, **kwargs)
    return value, float(time.perf_counter() - started)


def _component_label(component_name: str) -> str:
    if component_name == "a_tot":
        return "Power Total"
    return COMPONENT_LABELS[component_name]


def _is_object_parameter(parameter: str) -> bool:
    return str(parameter).startswith("cube")


def _selected_center(parameter: str) -> tuple[float, float] | None:
    if not _is_object_parameter(parameter):
        return None
    return (float(CUBE1_BASE_CENTER[0]), float(CUBE1_BASE_CENTER[1]))


def _grad_db_grid(grad_mag: np.ndarray) -> np.ndarray:
    safe_grad = np.where(np.isfinite(grad_mag), np.maximum(grad_mag, 1.0e-20), np.nan)
    return 20.0 * np.log10(safe_grad)


def _auto_limits_many(
    data_list: list[np.ndarray],
    *,
    span: float,
    floor: float = -120.0,
) -> tuple[float, float]:
    stacked = np.concatenate([data.ravel() for data in data_list], axis=0)
    finite = stacked[np.isfinite(stacked)]
    if finite.size == 0:
        return floor, floor + span
    vmax = float(np.percentile(finite, 99.0))
    return max(floor, vmax - span), vmax


def _symmetric_limits_many(
    data_list: list[np.ndarray],
    *,
    percentile: float = 99.0,
    minimum: float = 1.0,
) -> tuple[float, float]:
    stacked = np.concatenate([data.ravel() for data in data_list], axis=0)
    finite = stacked[np.isfinite(stacked)]
    if finite.size == 0:
        return -minimum, minimum
    vmax = float(np.percentile(np.abs(finite), percentile))
    vmax = max(vmax, minimum)
    return -vmax, vmax


def _relative_l2_error(lhs: np.ndarray, rhs: np.ndarray) -> float:
    finite_mask = np.isfinite(lhs) & np.isfinite(rhs)
    if not np.any(finite_mask):
        return float("nan")
    lhs_vals = lhs[finite_mask]
    rhs_vals = rhs[finite_mask]
    denom = np.linalg.norm(rhs_vals.ravel())
    if denom <= 1.0e-20:
        return float("nan")
    return float(np.linalg.norm((lhs_vals - rhs_vals).ravel()) / denom)


def _panel_stats_text(data: np.ndarray) -> str:
    finite = data[np.isfinite(data)]
    if finite.size == 0:
        return "mean=nan med=nan std=nan"
    return (
        f"mean={float(np.mean(finite)):.2f} "
        f"med={float(np.median(finite)):.2f} "
        f"std={float(np.std(finite)):.2f}"
    )


def _zero_vector_component(n_rx: int):
    return vector_zero(int(n_rx))


def _add_derived_vector_fields(fields):
    derived = dict(fields)
    derived["a_dif_direct"] = _add_complex_vector(
        derived["a_dif_direct_first_order"],
        derived["a_dif_direct_recursive"],
    )
    derived["a_dif_mixed"] = _add_complex_vector(
        derived["a_dif_mixed_state_accum"],
        derived["a_dif_mixed_reflected_suffix"],
    )
    derived["a_dif"] = _add_complex_vector(
        derived["a_dif_direct"],
        derived["a_dif_mixed"],
    )
    derived["a_tot"] = _add_complex_vector(
        _add_complex_vector(derived["a_los"], derived["a_ref"]),
        derived["a_dif"],
    )
    return derived


def _vector_field_to_numpy(field_vector, grid_size: int) -> dict[str, object]:
    axis_maps = {}
    power = np.zeros((grid_size, grid_size), dtype=np.float64)
    for axis in ("x", "y", "z"):
        real = np.asarray(field_vector[axis].real, dtype=np.float64).reshape(grid_size, grid_size)
        imag = np.asarray(field_vector[axis].imag, dtype=np.float64).reshape(grid_size, grid_size)
        axis_maps[axis] = {
            "real": real,
            "imag": imag,
            "mag": np.sqrt(real * real + imag * imag),
        }
        power = power + real * real + imag * imag
    axis_maps["power"] = power
    return axis_maps


def _vector_power_contribution_maps(
    total_field: dict[str, object],
    gradient_maps: dict[str, dict[str, object]],
) -> dict[str, dict[str, np.ndarray]]:
    power_maps = {}
    for component_name, grad_map in gradient_maps.items():
        signed = np.zeros_like(total_field["power"])
        for axis in ("x", "y", "z"):
            signed = signed + 2.0 * (
                total_field[axis]["real"] * grad_map[axis]["real"]
                + total_field[axis]["imag"] * grad_map[axis]["imag"]
            )
        power_maps[component_name] = {
            "signed": signed,
            "mag": np.abs(signed),
        }
    return power_maps


def _fd_total_power_map(
    plus_maps: dict[str, dict[str, object]],
    minus_maps: dict[str, dict[str, object]],
    *,
    fd_step: float,
) -> np.ndarray:
    return (plus_maps["a_tot"]["power"] - minus_maps["a_tot"]["power"]) / (2.0 * float(fd_step))


def _seed_trace():
    try:
        dr.seed(int(DEFAULT_TRACE_SEED))
    except Exception:
        pass
    try:
        wt.register_sampler_seed(int(DEFAULT_TRACE_SEED))
    except Exception:
        pass


def _build_radiomap_context(
    *,
    cube1_x,
    tx_pos,
    grid_size: int,
    n_rays: int,
    bounds,
    plane_z: float,
    combine_mode: str,
    receiver_model: str,
    shadow_boundary_mode: str,
    accumulation_backend: str,
    max_diffractions: int,
):
    _seed_trace()
    scene = build_scene_for_cube1_x(cube1_x)
    tracer = _make_tracer(scene, n_rays=n_rays, max_diffractions=max_diffractions)
    monitor = _make_monitor(
        grid_size=grid_size,
        bounds=bounds,
        plane_z=plane_z,
        combine_mode=combine_mode,
        receiver_model=receiver_model,
        shadow_boundary_mode=shadow_boundary_mode,
        accumulation_backend=accumulation_backend,
        max_diffractions=max_diffractions,
    )
    grid = RadioMapGrid.from_monitor(
        monitor,
        default_cell_size=tracer._resolved_trace_config.cell_size,
    )
    sample_set = grid.sample_sets[0]
    sample_grid = AxisAlignedRadioMapNativeGrid.from_grid(
        grid,
        sample_index=sample_set.index,
    )
    sample_positions = sample_grid.receivers
    solver_controls = tracer._resolve_monitor_solver_controls(
        monitor,
        execution_intent=radio_map_execution_intent(monitor),
    )
    effective = solver_controls["effective"]
    config = tracer._resolved_trace_config
    tx_point = wt.Point3f(tx_pos.x, tx_pos.y, tx_pos.z)
    reflection_detail = _discover_radio_map_reflection_detail(
        sample_grid=sample_grid,
        tx_pos=tx_point,
        scene=scene,
        config=config,
        solver_controls=solver_controls,
        monitor=monitor,
        reflection_detail=None,
    )
    anchor_coordinate = _diffraction_anchor_coordinate(
        sample_grid.axis,
        tx_point,
        sample_grid.position,
    )
    reflection_n_rays = effective["reflection_n_rays"] if config.enable_rd_diffraction else 0
    reflection_max_bounces = (
        effective["reflection_max_bounces"] if config.enable_rd_diffraction else 0
    )
    prepared_state_group = None
    if effective["max_diffractions"] > 0:
        prepared_state_group = _prepare_diffraction_state_arrays(
            tx_point,
            anchor_coordinate,
            scene,
            config.wavelength,
            config.k,
            reflection_detail if config.enable_rd_diffraction else None,
            config.diffraction_material,
            reflection_n_rays,
            reflection_max_bounces,
            config.reflection_coef,
            monitor.ray_mode,
            effective["max_diffractions"],
            total_state_budget_per_order=effective["diffraction_state_budget"],
            inserted_state_budget_per_order=effective["inserted_reflection_state_budget"],
            max_inserted_reflections_per_path=effective["max_inserted_reflections_per_path"],
            retain_cold_metadata=True,
            use_scene_materials=config.use_scene_materials_for_diffraction,
            tx_polarization=config.tx_polarization,
            solver_mode=solver_controls["selected"],
            memory_profile=effective["memory_profile"],
            state_layout="full",
        )
    _edge_cache, edge_data, state_arrays, _path_budget_report = prepared_state_group or (
        None,
        None,
        None,
        None,
    )
    suffix = ReflectionSuffixConfig(
        n_rays=reflection_n_rays,
        max_bounces=reflection_max_bounces,
        coef=config.reflection_coef,
        mode=monitor.ray_mode,
        detail=reflection_detail if config.enable_rd_diffraction else None,
        grid=sample_grid,
        grid_data=sample_grid.get_coordinates(),
    )
    return {
        "scene": scene,
        "monitor": monitor,
        "tracer": tracer,
        "grid": grid,
        "sample_grid": sample_grid,
        "sample_positions": sample_positions,
        "solver_controls": solver_controls,
        "effective": effective,
        "config": config,
        "tx_point": tx_point,
        "reflection_detail": reflection_detail,
        "edge_data": edge_data,
        "state_arrays": state_arrays,
        "suffix": suffix,
    }


def _accumulate_subset_vector(
    subset_states,
    *,
    context,
):
    n_rx = int(context["sample_grid"].n_cells)
    if subset_states is None or int(subset_states["n_states"]) <= 0:
        return _zero_vector_component(n_rx)
    _scalar_total, vector_total, _scheduler, _timing = accumulate_radio_map_diffraction_coherent(
        state_arrays=subset_states,
        edge_data=context["edge_data"],
        sample_grid=context["sample_grid"],
        rx_pos=context["sample_positions"],
        scene=context["scene"],
        wavelength=context["config"].wavelength,
        k=context["config"].k,
        material_detail=context["config"].diffraction_material,
        suffix=ReflectionSuffixConfig(),
        tx_polarization=context["config"].tx_polarization,
        rx_polarization=effective_rx_polarization(
            context["config"].rx_polarization,
            context["config"].tx_polarization,
        ),
        execution=context["config"].diffraction_execution,
        return_timing=False,
        return_vector=True,
        receiver_axis=str(context["sample_grid"].axis),
    )
    if vector_total is None:
        return _zero_vector_component(n_rx)
    return vector_total


def _trace_suffix_vector(*, context):
    n_rx = int(context["sample_grid"].n_cells)
    if context["state_arrays"] is None or int(context["state_arrays"]["n_states"]) <= 0:
        return _zero_vector_component(n_rx)
    _suffix_scalar, suffix_vector = trace_reflected_suffix_from_edge_states(
        state_arrays=context["state_arrays"],
        suffix=context["suffix"],
        scene=context["scene"],
        wavelength=context["config"].wavelength,
        k=context["config"].k,
        tx_polarization=context["config"].tx_polarization,
        execution=context["config"].diffraction_execution,
        receiver_tiles=None,
    )
    return suffix_vector


def _build_component_vector_fields(*, context):
    n_rx = int(context["sample_grid"].n_cells)
    sample_positions = context["sample_positions"]
    tx_point = context["tx_point"]
    config = context["config"]

    los_coherent = accumulate_radio_map_los_coherent(
        scene=context["scene"],
        rx_pos=sample_positions,
        tx_pos=tx_point,
        wavelength=config.wavelength,
        k=config.k,
    )
    ray_dir = sample_positions - tx_point
    tx_pol_dir = project_real_polarization_to_ray(config.tx_polarization, ray_dir)
    a_los = vector_from_scalar_and_real_direction(los_coherent, tx_pol_dir)

    _reflection_coherent, reflection_vector, _reflection_detail, _reflection_seconds = (
        accumulate_radio_map_reflection_coherent(
            sample_grid=context["sample_grid"],
            tx_pos=tx_point,
            scene=context["scene"],
            wavelength=config.wavelength,
            k=config.k,
            reflection_n_rays=context["effective"]["reflection_n_rays"],
            reflection_max_bounces=context["effective"]["reflection_max_bounces"],
            ray_mode=context["monitor"].ray_mode,
            reflection_coef=config.reflection_coef,
            min_ray_contribution_threshold=config.min_ray_contribution_threshold,
            reflection_field_backend=config.reflection_field_backend,
            tx_polarization=config.tx_polarization,
            rx_polarization=config.rx_polarization,
            reflection_relative_permittivity=config.reflection_relative_permittivity,
            reflection_conductivity=config.reflection_conductivity,
            reflection_material=config.reflection_material,
            use_scene_materials=config.use_scene_materials_for_reflection,
            reflection_detail=context["reflection_detail"],
            return_timing=False,
            return_vector=True,
        )
    )
    if reflection_vector is None:
        reflection_vector = _zero_vector_component(n_rx)

    zero = _zero_vector_component(n_rx)
    state_arrays = context["state_arrays"]
    if state_arrays is None or int(state_arrays["n_states"]) <= 0 or context["edge_data"] is None:
        primitive_fields = {
            "a_los": a_los,
            "a_ref": reflection_vector,
            "a_dif_direct_first_order": zero,
            "a_dif_direct_recursive": zero,
            "a_dif_mixed_state_accum": zero,
            "a_dif_mixed_prefix_first_order": zero,
            "a_dif_mixed_prefix_chain": zero,
            "a_dif_mixed_inserted_first_order": zero,
            "a_dif_mixed_inserted_chain": zero,
            "a_dif_mixed_reflected_suffix": zero,
        }
        return _add_derived_vector_fields(primitive_fields)

    ownership = _ownership_code_from_depths(
        state_arrays["prefix_reflection_depth"],
        state_arrays["intermediate_reflection_depth"],
        state_arrays["suffix_reflection_depth"],
    )
    direct_mask = ownership == wt.UInt32(OWNERSHIP_DIRECT_DIFFRACTION)
    mixed_mask = ownership == wt.UInt32(OWNERSHIP_MIXED_DIFFRACTION)
    mixed_states = subset_state_arrays(state_arrays, mixed_mask)

    primitive_fields = {
        "a_los": a_los,
        "a_ref": reflection_vector,
        "a_dif_direct_first_order": _accumulate_subset_vector(
            subset_state_arrays(
                state_arrays,
                direct_mask
                & (
                    state_arrays["approximation_mode_code"]
                    == wt.UInt32(APPROX_MODE_DIRECT_FIRST_ORDER)
                ),
            ),
            context=context,
        ),
        "a_dif_direct_recursive": _accumulate_subset_vector(
            subset_state_arrays(
                state_arrays,
                direct_mask
                & (
                    state_arrays["approximation_mode_code"]
                    == wt.UInt32(APPROX_MODE_RECURSIVE_DIFFRACTION)
                ),
            ),
            context=context,
        ),
        "a_dif_mixed_state_accum": _accumulate_subset_vector(
            mixed_states,
            context=context,
        ),
    }

    mixed_modes = (
        (APPROX_MODE_SAMPLED_REFLECTION_PREFIX, "a_dif_mixed_prefix_first_order"),
        (APPROX_MODE_SAMPLED_REFLECTION_PREFIX_CHAIN, "a_dif_mixed_prefix_chain"),
        (APPROX_MODE_SAMPLED_INSERTED_REFLECTION, "a_dif_mixed_inserted_first_order"),
        (APPROX_MODE_SAMPLED_INSERTED_REFLECTION_CHAIN, "a_dif_mixed_inserted_chain"),
    )
    for mode_code, component_name in mixed_modes:
        subset_mask = mixed_mask & (
            state_arrays["approximation_mode_code"] == wt.UInt32(mode_code)
        )
        primitive_fields[component_name] = _accumulate_subset_vector(
            subset_state_arrays(state_arrays, subset_mask),
            context=context,
        )

    primitive_fields["a_dif_mixed_reflected_suffix"] = _trace_suffix_vector(context=context)
    return _add_derived_vector_fields(primitive_fields)


def _component_maps_from_fields(
    component_fields,
    component_names: tuple[str, ...],
    *,
    grid_size: int,
):
    return {
        component_name: _vector_field_to_numpy(component_fields[component_name], grid_size)
        for component_name in component_names
    }


def _ad_component_maps(
    parameter: str,
    *,
    tx_pos,
    grid_size: int,
    n_rays: int,
    fd_step: float,
    bounds,
    plane_z: float,
    combine_mode: str,
    receiver_model: str,
    shadow_boundary_mode: str,
    accumulation_backend: str,
    max_diffractions: int,
):
    del fd_step
    config = parameter_config(parameter, tx_pos=tx_pos)
    if parameter == "cube1_x":
        parameter_value = wt.Float(config["cube1_x"])
        dr.enable_grad(parameter_value)
        cube1_x = parameter_value
        tx_point = wt.Point3f(*config["tx_pos"])
    else:
        parameter_value = wt.Float(config["tx_pos"][0])
        dr.enable_grad(parameter_value)
        cube1_x = config["cube1_x"]
        tx_point = wt.Point3f(parameter_value, config["tx_pos"][1], config["tx_pos"][2])

    context = _build_radiomap_context(
        cube1_x=cube1_x,
        tx_pos=tx_point,
        grid_size=grid_size,
        n_rays=n_rays,
        bounds=bounds,
        plane_z=plane_z,
        combine_mode=combine_mode,
        receiver_model=receiver_model,
        shadow_boundary_mode=shadow_boundary_mode,
        accumulation_backend=accumulation_backend,
        max_diffractions=max_diffractions,
    )
    primitive_fields = _build_component_vector_fields(context=context)
    primitive_names = tuple(PRIMITIVE_COMPONENTS)
    dr.set_grad(parameter_value, 1.0)
    grad_targets = tuple(
        primitive_fields[component_name][axis]
        for component_name in primitive_names
        for axis in ("x", "y", "z")
    )
    grads = dr.forward_to(*grad_targets, flags=_GRAD_FLAGS)
    if not isinstance(grads, tuple):
        grads = (grads,)

    grad_fields = {}
    grad_index = 0
    for component_name in primitive_names:
        grad_fields[component_name] = {
            axis: grads[grad_index + axis_index]
            for axis_index, axis in enumerate(("x", "y", "z"))
        }
        grad_index += 3

    all_forward_fields = _add_derived_vector_fields(
        {name: primitive_fields[name] for name in primitive_names}
    )
    all_grad_fields = _add_derived_vector_fields(grad_fields)
    forward_maps = _component_maps_from_fields(
        all_forward_fields,
        _DISPLAY_COMPONENTS,
        grid_size=grid_size,
    )
    ad_maps = _component_maps_from_fields(
        all_grad_fields,
        _DISPLAY_COMPONENTS,
        grid_size=grid_size,
    )
    metadata = {
        "prepared_state_count": (
            0 if context["state_arrays"] is None else int(context["state_arrays"]["n_states"])
        ),
        "reflection_path_count": int(
            0
            if context["reflection_detail"] is None
            else int(context["reflection_detail"].get("n_valid", 0))
        ),
    }
    return forward_maps, ad_maps, metadata


def _fd_component_maps(
    parameter: str,
    *,
    tx_pos,
    grid_size: int,
    n_rays: int,
    fd_step: float,
    bounds,
    plane_z: float,
    combine_mode: str,
    receiver_model: str,
    shadow_boundary_mode: str,
    accumulation_backend: str,
    max_diffractions: int,
):
    config = parameter_config(parameter, tx_pos=tx_pos)
    plus_cfg, minus_cfg = config["perturb"](fd_step)
    plus_context = _build_radiomap_context(
        cube1_x=plus_cfg["cube1_x"],
        tx_pos=wt.Point3f(*plus_cfg["tx_pos"]),
        grid_size=grid_size,
        n_rays=n_rays,
        bounds=bounds,
        plane_z=plane_z,
        combine_mode=combine_mode,
        receiver_model=receiver_model,
        shadow_boundary_mode=shadow_boundary_mode,
        accumulation_backend=accumulation_backend,
        max_diffractions=max_diffractions,
    )
    minus_context = _build_radiomap_context(
        cube1_x=minus_cfg["cube1_x"],
        tx_pos=wt.Point3f(*minus_cfg["tx_pos"]),
        grid_size=grid_size,
        n_rays=n_rays,
        bounds=bounds,
        plane_z=plane_z,
        combine_mode=combine_mode,
        receiver_model=receiver_model,
        shadow_boundary_mode=shadow_boundary_mode,
        accumulation_backend=accumulation_backend,
        max_diffractions=max_diffractions,
    )
    plus_fields = _build_component_vector_fields(context=plus_context)
    minus_fields = _build_component_vector_fields(context=minus_context)
    plus_maps = _component_maps_from_fields(plus_fields, _DISPLAY_COMPONENTS, grid_size=grid_size)
    minus_maps = _component_maps_from_fields(minus_fields, _DISPLAY_COMPONENTS, grid_size=grid_size)

    fd_maps = {}
    for component_name in _DISPLAY_COMPONENTS:
        fd_maps[component_name] = {}
        for axis in ("x", "y", "z"):
            grad_real = (
                plus_maps[component_name][axis]["real"] - minus_maps[component_name][axis]["real"]
            ) / (2.0 * float(fd_step))
            grad_imag = (
                plus_maps[component_name][axis]["imag"] - minus_maps[component_name][axis]["imag"]
            ) / (2.0 * float(fd_step))
            fd_maps[component_name][axis] = {
                "real": grad_real,
                "imag": grad_imag,
                "mag": np.sqrt(grad_real * grad_real + grad_imag * grad_imag),
            }
        fd_maps[component_name]["power"] = (
            np.zeros((grid_size, grid_size), dtype=np.float64)
            if component_name != "a_tot"
            else _fd_total_power_map(plus_maps, minus_maps, fd_step=fd_step)
        )
    return fd_maps


def build_power_component_benchmark(
    *,
    parameter: str,
    grid_size: int,
    n_rays: int,
    fd_step: float,
    bounds,
    plane_z: float,
    tx_pos,
    combine_mode: str,
    receiver_model: str,
    shadow_boundary_mode: str,
    accumulation_backend: str,
    max_diffractions: int,
):
    (forward_maps, ad_vector_maps, metadata), ad_seconds = _time_call(
        _ad_component_maps,
        parameter,
        tx_pos=tx_pos,
        grid_size=grid_size,
        n_rays=n_rays,
        fd_step=fd_step,
        bounds=bounds,
        plane_z=plane_z,
        combine_mode=combine_mode,
        receiver_model=receiver_model,
        shadow_boundary_mode=shadow_boundary_mode,
        accumulation_backend=accumulation_backend,
        max_diffractions=max_diffractions,
    )
    fd_vector_maps, fd_seconds = _time_call(
        _fd_component_maps,
        parameter,
        tx_pos=tx_pos,
        grid_size=grid_size,
        n_rays=n_rays,
        fd_step=fd_step,
        bounds=bounds,
        plane_z=plane_z,
        combine_mode=combine_mode,
        receiver_model=receiver_model,
        shadow_boundary_mode=shadow_boundary_mode,
        accumulation_backend=accumulation_backend,
        max_diffractions=max_diffractions,
    )
    ad_power_maps = _vector_power_contribution_maps(forward_maps["a_tot"], ad_vector_maps)
    fd_power_maps = _vector_power_contribution_maps(forward_maps["a_tot"], fd_vector_maps)
    summary = RadiomapPowerComponentSummary(
        parameter=str(parameter),
        grid_size=int(grid_size),
        bounds=(
            (float(bounds[0][0]), float(bounds[0][1])),
            (float(bounds[1][0]), float(bounds[1][1])),
        ),
        plane_z=float(plane_z),
        tx_pos=tuple(float(value) for value in tx_pos),
        n_rays=int(n_rays),
        fd_step=float(fd_step),
        combine_mode=str(combine_mode),
        receiver_model=str(receiver_model),
        shadow_boundary_mode=str(shadow_boundary_mode),
        accumulation_backend_requested=str(accumulation_backend),
        max_diffractions=int(max_diffractions),
        timings_seconds={
            "ad": float(ad_seconds),
            "fd": float(fd_seconds),
        },
    )
    return {
        "summary": summary,
        "forward": forward_maps,
        "ad": ad_power_maps,
        "fd": fd_power_maps,
        "metadata": metadata,
    }


def save_figure(benchmark, *, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    grad_db_maps = []
    diff_db_maps = []
    for component_name in _DISPLAY_COMPONENTS:
        grad_db_maps.append(_grad_db_grid(benchmark["ad"][component_name]["mag"]))
        grad_db_maps.append(_grad_db_grid(benchmark["fd"][component_name]["mag"]))
        diff_db_maps.append(
            _grad_db_grid(benchmark["ad"][component_name]["mag"])
            - _grad_db_grid(benchmark["fd"][component_name]["mag"])
        )
    grad_vmin, grad_vmax = _auto_limits_many(grad_db_maps, span=55.0)
    diff_vmin, diff_vmax = -_DIFF_DB_LIMIT, _DIFF_DB_LIMIT

    n_cols = len(_DISPLAY_COMPONENTS)
    fig_width = max(24.0, 2.9 * n_cols)
    fig, axes = plt.subplots(3, n_cols, figsize=(fig_width, 9.2), constrained_layout=True)
    row_specs = (
        ("AD", benchmark["ad"], None, grad_vmin, grad_vmax, "magma"),
        ("FD", benchmark["fd"], None, grad_vmin, grad_vmax, "magma"),
        ("AD - FD", benchmark["ad"], benchmark["fd"], diff_vmin, diff_vmax, "RdBu_r"),
    )

    grad_handle = None
    diff_handle = None
    selected_center = _selected_center(benchmark["summary"].parameter)
    for row_idx, (row_label, maps, diff_against, row_vmin, row_vmax, cmap) in enumerate(row_specs):
        for col_idx, component_name in enumerate(_DISPLAY_COMPONENTS):
            panel_data = _grad_db_grid(maps[component_name]["mag"])
            if diff_against is not None:
                panel_data = panel_data - _grad_db_grid(diff_against[component_name]["mag"])
            ax = axes[row_idx, col_idx]
            image = ax.imshow(
                panel_data,
                origin="lower",
                extent=benchmark["summary"].bounds[0] + benchmark["summary"].bounds[1],
                cmap=cmap,
                vmin=row_vmin,
                vmax=row_vmax,
                interpolation="nearest",
            )
            _decorate_axis(
                ax,
                bounds=benchmark["summary"].bounds,
                cube1_x=CUBE1_BASE_CENTER[0],
                tx_pos=benchmark["summary"].tx_pos,
            )
            ax.set_xlabel("")
            ax.set_ylabel("")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.tick_params(
                axis="both",
                which="both",
                bottom=False,
                top=False,
                left=False,
                right=False,
                labelbottom=False,
                labelleft=False,
            )
            if selected_center is not None:
                ax.scatter(
                    [selected_center[0]],
                    [selected_center[1]],
                    c="cyan",
                    s=55,
                    marker="s",
                    edgecolors="black",
                    linewidths=1.0,
                    zorder=9,
                )
            if row_idx == 0:
                rel_l2 = _relative_l2_error(
                    benchmark["ad"][component_name]["signed"],
                    benchmark["fd"][component_name]["signed"],
                )
                ax.set_title(
                    f"{_component_label(component_name)}\n"
                    f"rel-L2={rel_l2:.2e}\n"
                    f"{_panel_stats_text(panel_data)}",
                    fontsize=10,
                )
            if col_idx == 0:
                ax.set_ylabel(row_label, fontsize=11)
            if row_idx < 2:
                grad_handle = image
            else:
                diff_handle = image

    if grad_handle is not None:
        fig.colorbar(
            grad_handle,
            ax=axes[:2, :],
            shrink=0.86,
            label="Power-gradient contribution magnitude [dB]",
        )
    if diff_handle is not None:
        fig.colorbar(
            diff_handle,
            ax=axes[2, :],
            shrink=0.86,
            label="Signed difference [dB] on power-gradient contribution magnitude",
        )

    fig.suptitle(
        (
            "Radiomap Three-Cube Per-Component Power-Gradient Contribution Comparison\n"
            f"parameter={benchmark['summary'].parameter}, "
            f"grid={benchmark['summary'].grid_size}, rays={benchmark['summary'].n_rays}, "
            f"fd_step={benchmark['summary'].fd_step}, "
            f"shadow={benchmark['summary'].shadow_boundary_mode}"
        ),
        fontsize=14,
    )
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def save_json(benchmark, *, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary = asdict(benchmark["summary"])
    summary["components"] = {}
    for component_name in _DISPLAY_COMPONENTS:
        ad_signed = benchmark["ad"][component_name]["signed"]
        fd_signed = benchmark["fd"][component_name]["signed"]
        diff_grid = ad_signed - fd_signed
        summary["components"][component_name] = {
            "label": _component_label(component_name),
            "ad_abs_sum": float(np.sum(np.abs(ad_signed))),
            "fd_abs_sum": float(np.sum(np.abs(fd_signed))),
            "ad_fd_corr": float(_safe_correlation(ad_signed, fd_signed)),
            "ad_fd_rel_l2": float(_relative_l2_error(ad_signed, fd_signed)),
            "ad_fd_mean_abs_diff": float(np.mean(np.abs(diff_grid))),
            "ad_fd_max_abs_diff": float(np.max(np.abs(diff_grid))),
        }
    summary["metadata"] = dict(benchmark["metadata"])
    output_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return output_path


def default_output_path(parameter: str) -> Path:
    return _output_dir() / f"radiomap_{parameter}_power_components.png"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parameter", choices=_PARAMETERS, default="cube1_x")
    parser.add_argument("--grid-size", type=int, default=DEFAULT_GRID_SIZE)
    parser.add_argument("--n-rays", type=int, default=DEFAULT_N_RAYS)
    parser.add_argument("--fd-step", type=float, default=DEFAULT_FD_STEP)
    parser.add_argument("--plane-z", type=float, default=DEFAULT_PLANE_Z)
    parser.add_argument("--tx-x", type=float, default=float(DEFAULT_TX_POS[0]))
    parser.add_argument("--tx-y", type=float, default=float(DEFAULT_TX_POS[1]))
    parser.add_argument("--tx-z", type=float, default=float(DEFAULT_TX_POS[2]))
    parser.add_argument("--combine-mode", type=str, default=DEFAULT_COMBINE_MODE)
    parser.add_argument("--receiver-model", type=str, default=DEFAULT_RECEIVER_MODEL)
    parser.add_argument("--shadow-boundary-mode", type=str, default=DEFAULT_SHADOW_BOUNDARY_MODE)
    parser.add_argument("--accumulation-backend", type=str, default=DEFAULT_ACCUMULATION_BACKEND)
    parser.add_argument("--max-diffractions", type=int, default=DEFAULT_MAX_DIFFRACTIONS)
    parser.add_argument("--xmin", type=float, default=float(DEFAULT_BOUNDS[0][0]))
    parser.add_argument("--xmax", type=float, default=float(DEFAULT_BOUNDS[0][1]))
    parser.add_argument("--ymin", type=float, default=float(DEFAULT_BOUNDS[1][0]))
    parser.add_argument("--ymax", type=float, default=float(DEFAULT_BOUNDS[1][1]))
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    bounds = (
        (float(args.xmin), float(args.xmax)),
        (float(args.ymin), float(args.ymax)),
    )
    tx_pos = (float(args.tx_x), float(args.tx_y), float(args.tx_z))
    benchmark = build_power_component_benchmark(
        parameter=str(args.parameter),
        grid_size=int(args.grid_size),
        n_rays=int(args.n_rays),
        fd_step=float(args.fd_step),
        bounds=bounds,
        plane_z=float(args.plane_z),
        tx_pos=tx_pos,
        combine_mode=str(args.combine_mode),
        receiver_model=str(args.receiver_model),
        shadow_boundary_mode=str(args.shadow_boundary_mode),
        accumulation_backend=str(args.accumulation_backend),
        max_diffractions=int(args.max_diffractions),
    )
    output_path = args.output if args.output is not None else default_output_path(str(args.parameter))
    figure_path = save_figure(benchmark, output_path=output_path)
    json_path = save_json(benchmark, output_path=figure_path.with_suffix(".json"))
    print(
        json.dumps(
            {
                "figure": str(figure_path),
                "json": str(json_path),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
