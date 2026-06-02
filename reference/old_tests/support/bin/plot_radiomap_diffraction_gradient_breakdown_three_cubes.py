"""Break down deterministic three-cube diffraction AD/FD gradients by internal subcomponent."""

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

from tests.main.plot_multipath_components import (
    CUBE1_BASE_CENTER,
    build_scene_for_cube1_x,
    gradient_db_magnitude,
)
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
    _time_call,
    parameter_config,
)
from tests.main.plot_radiomap_sionna_three_cubes import (
    DEFAULT_BOUNDS,
    DEFAULT_GRID_SIZE,
    DEFAULT_N_RAYS,
    DEFAULT_OUTPUT_PREFIX,
    DEFAULT_PLANE_Z,
    DEFAULT_TX_POS,
    _decorate_axis,
    _output_dir,
)
from witwin.channel.kernels.monitors.field.radio_map_accumulate import (
    radiomap_vector_power,
)
from witwin.channel.monitors.radio_map.deterministic.cell_accumulation import (
    accumulate_diffraction_scalar_power,
)
from witwin.channel.monitors.radio_map.deterministic.coherent import (
    accumulate_radio_map_los_coherent,
    accumulate_radio_map_reflection_coherent,
)
from witwin.channel.monitors.radio_map.deterministic.samples import (
    _discover_radio_map_reflection_detail,
    _trace_diffraction_raw_collections,
)
from witwin.channel.monitors.radio_map.diagnostics import (
    _accumulate_complex_by_rx,
    _add_complex,
    _scatter_float,
    _vector_power_symbolic,
)
from witwin.channel.monitors.radio_map.grid import (
    AxisAlignedRadioMapNativeGrid,
    RadioMapGrid,
)
from witwin.channel.trace.cache import radio_map_execution_intent
from witwin.channel.utils.drjit_ops import complex_abs_sqr, complex_zero
from witwin.channel.utils.polarization import (
    project_real_polarization_to_ray,
    vector_from_scalar_and_real_direction,
)


DEFAULT_OUTPUT_PREFIX = (
    _output_dir() / "radiomap_three_cubes_diffraction_gradient_breakdown"
)
_PARAMETERS = ("tx_x", "cube1_x")
_FIGURE_METRICS = (
    ("raw_projected_abs", "Raw Projected"),
    ("replay_projected_abs", "Replay Projected"),
    ("replay_power", "Per-Path Power"),
    ("vector_x_power", "Vector X"),
    ("vector_y_power", "Vector Y"),
    ("vector_z_power", "Vector Z"),
    ("vector_power_symbolic", "Vector Total"),
)
_SUMMARY_ONLY_METRICS = (
    ("vector_power_custom", "Vector Total (Kernel)"),
)
_METRIC_ROW_SPLITS = (
    _FIGURE_METRICS[:4],
    _FIGURE_METRICS[4:],
)


@dataclass(frozen=True)
class DiffractionBreakdownSummary:
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
    parameters: tuple[str, ...]
    timings_seconds: dict[str, dict[str, float]]


def _trace_diffraction_breakdown_payload(
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
    coords, field_components, replay_power, metadata = _trace_diffraction_breakdown_fields(
        cube1_x=cube1_x,
        tx_pos=tx_pos,
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
    metrics = {
        "raw_projected_abs": complex_abs_sqr(field_components["raw_projected"]),
        "replay_projected_abs": complex_abs_sqr(field_components["replay_projected"]),
        "replay_power": replay_power,
        "vector_x_power": complex_abs_sqr(field_components["vector_x"]),
        "vector_y_power": complex_abs_sqr(field_components["vector_y"]),
        "vector_z_power": complex_abs_sqr(field_components["vector_z"]),
        "vector_power_custom": radiomap_vector_power(
            {
                "x": field_components["vector_x"],
                "y": field_components["vector_y"],
                "z": field_components["vector_z"],
            }
        ),
        "vector_power_symbolic": _vector_power_symbolic(
            {
                "x": field_components["vector_x"],
                "y": field_components["vector_y"],
                "z": field_components["vector_z"],
            }
        ),
    }
    return coords, metrics, metadata


def _trace_diffraction_breakdown_fields(
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
    del accumulation_backend
    try:
        dr.seed(int(DEFAULT_TRACE_SEED))
    except Exception:
        pass
    try:
        wt.register_sampler_seed(int(DEFAULT_TRACE_SEED))
    except Exception:
        pass

    scene = build_scene_for_cube1_x(cube1_x)
    tracer = _make_tracer(scene, n_rays=n_rays, max_diffractions=max_diffractions)
    monitor = _make_monitor(
        grid_size=grid_size,
        bounds=bounds,
        plane_z=plane_z,
        combine_mode=combine_mode,
        receiver_model=receiver_model,
        shadow_boundary_mode=shadow_boundary_mode,
        accumulation_backend="baseline",
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
    config = tracer._resolved_trace_config
    tx_point = wt.Point3f(tx_pos.x, tx_pos.y, tx_pos.z)

    los_coherent = accumulate_radio_map_los_coherent(
        scene=scene,
        rx_pos=sample_positions,
        tx_pos=tx_point,
        wavelength=config.wavelength,
        k=config.k,
    )
    ray_dir = sample_positions - tx_point
    tx_pol_dir = project_real_polarization_to_ray(config.tx_polarization, ray_dir)
    los_field_vector = vector_from_scalar_and_real_direction(
        los_coherent,
        tx_pol_dir,
    )

    reflection_detail = _discover_radio_map_reflection_detail(
        sample_grid=sample_grid,
        tx_pos=tx_point,
        scene=scene,
        config=config,
        solver_controls=solver_controls,
        monitor=monitor,
        reflection_detail=None,
    )
    (
        _reflection_coherent,
        reflection_vector_coherent,
        reflection_detail,
        _reflection_seconds,
    ) = accumulate_radio_map_reflection_coherent(
        sample_grid=sample_grid,
        tx_pos=tx_point,
        scene=scene,
        wavelength=config.wavelength,
        k=config.k,
        reflection_n_rays=solver_controls["effective"]["reflection_n_rays"],
        reflection_max_bounces=solver_controls["effective"]["reflection_max_bounces"],
        ray_mode=monitor.ray_mode,
        reflection_coef=config.reflection_coef,
        min_ray_contribution_threshold=config.min_ray_contribution_threshold,
        reflection_field_backend=config.reflection_field_backend,
        tx_polarization=config.tx_polarization,
        rx_polarization=config.rx_polarization,
        reflection_relative_permittivity=config.reflection_relative_permittivity,
        reflection_conductivity=config.reflection_conductivity,
        reflection_material=config.reflection_material,
        use_scene_materials=config.use_scene_materials_for_reflection,
        reflection_detail=reflection_detail,
        return_timing=False,
        return_vector=True,
    )
    diffraction_raw_collections, _, _ = _trace_diffraction_raw_collections(
        sample_positions=sample_positions,
        tx_pos=tx_point,
        scene=scene,
        config=config,
        solver_controls=solver_controls,
        monitor=monitor,
        reflection_detail=reflection_detail,
        persistent_diffraction_state_cache=None,
        local_diffraction_state_cache={},
        diffraction_state_cache_key_fn=None,
        state_layout="full",
    )

    n_rx = int(sample_grid.n_cells)
    raw_projected = complex_zero(n_rx)
    replay_projected = complex_zero(n_rx)
    replay_power = dr.zeros(wt.Float, n_rx)
    replay_vector = {axis: complex_zero(n_rx) for axis in ("x", "y", "z")}

    for raw in diffraction_raw_collections:
        raw_projected = _add_complex(
            raw_projected,
            _accumulate_complex_by_rx(raw, n_rx=n_rx),
        )
        receiver_index_map = raw.get("radio_map_receiver_index_map")
        local_positions = raw.get("rx_positions")
        state_arrays = raw.get("state_arrays")
        if receiver_index_map is None or local_positions is None or state_arrays is None:
            continue
        local_los_reference_vector = {
            axis: dr.gather(
                wt.Complex2f,
                los_field_vector[axis],
                receiver_index_map,
            )
            for axis in ("x", "y", "z")
        }
        local_reflection_reference_vector = {
            axis: dr.gather(
                wt.Complex2f,
                reflection_vector_coherent[axis],
                receiver_index_map,
            )
            for axis in ("x", "y", "z")
        }
        payload = accumulate_diffraction_scalar_power(
            state_arrays=state_arrays,
            rx_pos=local_positions,
            scene=scene,
            wavelength=config.wavelength,
            k=config.k,
            material_detail=config.diffraction_material,
            tx_polarization=config.tx_polarization,
            rx_polarization=config.rx_polarization,
            receiver_model="matched_isotropic",
            return_vector_coherent=True,
            incident_reference_vector=local_los_reference_vector,
            reflection_reference_vector=local_reflection_reference_vector,
            shadow_support_cutoff_db=getattr(
                monitor,
                "shadow_support_cutoff_db",
                None,
            ),
        )
        dr.scatter_reduce(
            dr.ReduceOp.Add,
            replay_projected.real,
            payload["coherent"].real,
            receiver_index_map,
        )
        dr.scatter_reduce(
            dr.ReduceOp.Add,
            replay_projected.imag,
            payload["coherent"].imag,
            receiver_index_map,
        )
        _scatter_float(
            replay_power,
            payload["power"],
            receiver_index_map,
        )
        for axis in ("x", "y", "z"):
            dr.scatter_reduce(
                dr.ReduceOp.Add,
                replay_vector[axis].real,
                payload["vector_coherent"][axis].real,
                receiver_index_map,
            )
            dr.scatter_reduce(
                dr.ReduceOp.Add,
                replay_vector[axis].imag,
                payload["vector_coherent"][axis].imag,
                receiver_index_map,
            )

    coords = {
        "grid_x": np.asarray(grid.grid_x, dtype=np.float64).reshape(grid_size, grid_size),
        "grid_y": np.asarray(grid.grid_y, dtype=np.float64).reshape(grid_size, grid_size),
    }
    metadata = {
        "raw_collection_count": int(len(diffraction_raw_collections)),
        "raw_path_count": int(
            sum(int(dr.width(raw["rx_index"])) for raw in diffraction_raw_collections)
        ),
    }
    return (
        coords,
        {
            "raw_projected": raw_projected,
            "replay_projected": replay_projected,
            "vector_x": replay_vector["x"],
            "vector_y": replay_vector["y"],
            "vector_z": replay_vector["z"],
        },
        replay_power,
        metadata,
    )


def _ad_diffraction_metrics(
    parameter: str,
    *,
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
    coords, metrics, metadata = _trace_diffraction_breakdown_payload(
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
    metric_names = tuple(metrics.keys())
    dr.set_grad(parameter_value, 1.0)
    grads = dr.forward_to(
        *(metrics[name] for name in metric_names),
        flags=_GRAD_FLAGS,
    )
    if not isinstance(grads, tuple):
        grads = (grads,)
    return (
        coords,
        {
            name: np.asarray(grad, dtype=np.float64).reshape(grid_size, grid_size)
            for name, grad in zip(metric_names, grads, strict=True)
        },
        metadata,
    )


def _fd_diffraction_metrics(
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
    _, plus_metrics, _ = _trace_diffraction_breakdown_payload(
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
    _, minus_metrics, _ = _trace_diffraction_breakdown_payload(
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
    return {
        name: (
            np.asarray(plus_metrics[name], dtype=np.float64).reshape(grid_size, grid_size)
            - np.asarray(minus_metrics[name], dtype=np.float64).reshape(grid_size, grid_size)
        )
        / (2.0 * float(fd_step))
        for name in plus_metrics
    }


def build_diffraction_gradient_breakdown(
    *,
    parameters: tuple[str, ...],
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
    rows = {}
    timings_seconds = {}
    grid_x = None
    grid_y = None
    metadata = {}
    for parameter in parameters:
        (coords, ad_metrics, parameter_metadata), ad_seconds = _time_call(
            _ad_diffraction_metrics,
            parameter,
            tx_pos=tx_pos,
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
        fd_metrics, fd_seconds = _time_call(
            _fd_diffraction_metrics,
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
        if grid_x is None:
            grid_x = coords["grid_x"]
            grid_y = coords["grid_y"]
        rows[parameter] = {
            "ad": ad_metrics,
            "fd": fd_metrics,
        }
        metadata[parameter] = dict(parameter_metadata)
        timings_seconds[parameter] = {
            "ad": float(ad_seconds),
            "fd": float(fd_seconds),
        }

    summary = DiffractionBreakdownSummary(
        grid_size=int(grid_size),
        bounds=tuple((float(a), float(b)) for (a, b) in bounds),
        plane_z=float(plane_z),
        tx_pos=tuple(float(value) for value in tx_pos),
        n_rays=int(n_rays),
        fd_step=float(fd_step),
        combine_mode=str(combine_mode),
        receiver_model=str(receiver_model),
        shadow_boundary_mode=str(shadow_boundary_mode),
        accumulation_backend_requested=str(accumulation_backend),
        max_diffractions=int(max_diffractions),
        parameters=tuple(parameters),
        timings_seconds=timings_seconds,
    )
    return {
        "summary": summary,
        "grid_x": grid_x,
        "grid_y": grid_y,
        "rows": rows,
        "metadata": metadata,
    }


def save_figure(benchmark, *, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    parameters = tuple(benchmark["summary"].parameters)
    max_metrics_per_row = max(len(split) for split in _METRIC_ROW_SPLITS)
    n_rows = len(parameters) * len(_METRIC_ROW_SPLITS)
    n_cols = 3 * max_metrics_per_row
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(3.4 * n_cols, 3.2 * n_rows),
        constrained_layout=False,
        squeeze=False,
    )
    fig.subplots_adjust(
        left=0.012,
        right=0.995,
        bottom=0.03,
        top=0.84,
        wspace=0.10,
        hspace=0.18,
    )
    extent = (
        float(benchmark["summary"].bounds[0][0]),
        float(benchmark["summary"].bounds[0][1]),
        float(benchmark["summary"].bounds[1][0]),
        float(benchmark["summary"].bounds[1][1]),
    )
    tx_pos = benchmark["summary"].tx_pos
    cube1_x = CUBE1_BASE_CENTER[0]

    for row_index, parameter in enumerate(parameters):
        row = benchmark["rows"][parameter]
        timings = benchmark["summary"].timings_seconds[parameter]
        for split_index, metrics in enumerate(_METRIC_ROW_SPLITS):
            panel_row = row_index * len(_METRIC_ROW_SPLITS) + split_index
            for unused_col in range(3 * len(metrics), n_cols):
                axes[panel_row, unused_col].axis("off")
            for metric_index, (metric_name, metric_label) in enumerate(metrics):
                ad_grid = row["ad"][metric_name]
                fd_grid = row["fd"][metric_name]
                ad_vis = gradient_db_magnitude(ad_grid)
                fd_vis = gradient_db_magnitude(fd_grid)
                diff_vis = ad_vis - fd_vis
                diff_vmax = max(float(np.nanpercentile(np.abs(diff_vis), 99.0)), 1.0)
                panels = (
                    (
                        axes[panel_row, 3 * metric_index + 0],
                        ad_vis,
                        f"{metric_label} AD, {parameter}\nAD={timings['ad']:.2f}s",
                        "magma",
                        float(np.nanpercentile(ad_vis, 5.0)),
                        float(np.nanpercentile(ad_vis, 99.0)),
                    ),
                    (
                        axes[panel_row, 3 * metric_index + 1],
                        fd_vis,
                        f"{metric_label} FD, {parameter}\nFD={timings['fd']:.2f}s",
                        "magma",
                        float(np.nanpercentile(fd_vis, 5.0)),
                        float(np.nanpercentile(fd_vis, 99.0)),
                    ),
                    (
                        axes[panel_row, 3 * metric_index + 2],
                        diff_vis,
                        f"{metric_label} AD-FD, {parameter}",
                        "coolwarm",
                        -diff_vmax,
                        diff_vmax,
                    ),
                )
                for ax, image, title, cmap, vmin, vmax in panels:
                    ax.imshow(
                        image,
                        origin="lower",
                        extent=extent,
                        cmap=cmap,
                        vmin=vmin,
                        vmax=vmax,
                        interpolation="nearest",
                    )
                    _decorate_axis(
                        ax,
                        bounds=benchmark["summary"].bounds,
                        cube1_x=cube1_x,
                        tx_pos=tx_pos,
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
                    ax.set_title(title, fontsize=13, pad=8.0)

    fig.suptitle(
        (
            "Three-Cube Radiomap Diffraction Gradient Breakdown\n"
            f"grid={benchmark['summary'].grid_size}x{benchmark['summary'].grid_size}, "
            f"rays={benchmark['summary'].n_rays}, "
            f"z={benchmark['summary'].plane_z:.1f}, "
            f"shadow={benchmark['summary'].shadow_boundary_mode}"
        ),
        fontsize=18,
        y=0.975,
    )
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def save_json(benchmark, *, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary = asdict(benchmark["summary"])
    summary["parameters_summary"] = {}
    all_metrics = _FIGURE_METRICS + _SUMMARY_ONLY_METRICS
    for parameter in benchmark["summary"].parameters:
        row = benchmark["rows"][parameter]
        parameter_summary = {}
        for metric_name, metric_label in all_metrics:
            ad_grid = row["ad"][metric_name]
            fd_grid = row["fd"][metric_name]
            diff_grid = ad_grid - fd_grid
            parameter_summary[metric_name] = {
                "label": metric_label,
                "ad_abs_sum": float(np.sum(np.abs(ad_grid))),
                "fd_abs_sum": float(np.sum(np.abs(fd_grid))),
                "ad_fd_corr": float(_safe_correlation(ad_grid, fd_grid)),
                "ad_fd_mean_abs_diff": float(np.mean(np.abs(diff_grid))),
                "ad_fd_max_abs_diff": float(np.max(np.abs(diff_grid))),
            }
        summary["parameters_summary"][parameter] = {
            "metrics": parameter_summary,
            "metadata": benchmark["metadata"][parameter],
        }
    output_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return output_path


def save_breakdown(
    output_prefix: Path,
    *,
    parameters: tuple[str, ...],
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
    benchmark = build_diffraction_gradient_breakdown(
        parameters=parameters,
        grid_size=grid_size,
        n_rays=n_rays,
        fd_step=fd_step,
        bounds=bounds,
        plane_z=plane_z,
        tx_pos=tx_pos,
        combine_mode=combine_mode,
        receiver_model=receiver_model,
        shadow_boundary_mode=shadow_boundary_mode,
        accumulation_backend=accumulation_backend,
        max_diffractions=max_diffractions,
    )
    figure_path = save_figure(
        benchmark,
        output_path=output_prefix.with_suffix(".png"),
    )
    json_path = save_json(
        benchmark,
        output_path=output_prefix.with_suffix(".json"),
    )
    return figure_path, json_path


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--parameter",
        choices=("tx_x", "cube1_x", "both"),
        default="both",
    )
    parser.add_argument("--grid-size", type=int, default=DEFAULT_GRID_SIZE)
    parser.add_argument("--n-rays", type=int, default=DEFAULT_N_RAYS)
    parser.add_argument("--fd-step", type=float, default=DEFAULT_FD_STEP)
    parser.add_argument("--plane-z", type=float, default=DEFAULT_PLANE_Z)
    parser.add_argument("--tx-x", type=float, default=float(DEFAULT_TX_POS[0]))
    parser.add_argument("--tx-y", type=float, default=float(DEFAULT_TX_POS[1]))
    parser.add_argument("--tx-z", type=float, default=float(DEFAULT_TX_POS[2]))
    parser.add_argument("--combine-mode", type=str, default=DEFAULT_COMBINE_MODE)
    parser.add_argument("--receiver-model", type=str, default=DEFAULT_RECEIVER_MODEL)
    parser.add_argument(
        "--shadow-boundary-mode",
        type=str,
        default=DEFAULT_SHADOW_BOUNDARY_MODE,
    )
    parser.add_argument(
        "--accumulation-backend",
        type=str,
        default=DEFAULT_ACCUMULATION_BACKEND,
    )
    parser.add_argument("--max-diffractions", type=int, default=DEFAULT_MAX_DIFFRACTIONS)
    parser.add_argument("--xmin", type=float, default=float(DEFAULT_BOUNDS[0][0]))
    parser.add_argument("--xmax", type=float, default=float(DEFAULT_BOUNDS[0][1]))
    parser.add_argument("--ymin", type=float, default=float(DEFAULT_BOUNDS[1][0]))
    parser.add_argument("--ymax", type=float, default=float(DEFAULT_BOUNDS[1][1]))
    parser.add_argument("--output-prefix", type=Path, default=DEFAULT_OUTPUT_PREFIX)
    return parser.parse_args()


def main():
    args = parse_args()
    parameters = _PARAMETERS if args.parameter == "both" else (str(args.parameter),)
    bounds = (
        (float(args.xmin), float(args.xmax)),
        (float(args.ymin), float(args.ymax)),
    )
    tx_pos = (float(args.tx_x), float(args.tx_y), float(args.tx_z))
    figure_path, json_path = save_breakdown(
        args.output_prefix,
        parameters=parameters,
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
