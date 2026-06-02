"""Plot radiomap diffraction-only field components with forward and AD/FD comparisons."""

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

from tests.main.plot_multipath_components import CUBE1_BASE_CENTER
from tests.main.plot_radiomap_gradients_three_cubes import (
    DEFAULT_ACCUMULATION_BACKEND,
    DEFAULT_COMBINE_MODE,
    DEFAULT_FD_STEP,
    DEFAULT_MAX_DIFFRACTIONS,
    DEFAULT_RECEIVER_MODEL,
    DEFAULT_SHADOW_BOUNDARY_MODE,
    _GRAD_FLAGS,
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
from tests.support.bin.plot_radiomap_diffraction_gradient_breakdown_three_cubes import (
    _trace_diffraction_breakdown_fields,
)


DEFAULT_OUTPUT_PREFIX = (
    _output_dir() / "radiomap_three_cubes_gradients_diffraction_field_components"
)
_PARAMETERS = ("tx_x", "cube1_x")
_FIELD_COMPONENTS = (
    ("raw_projected", "Raw Projected"),
    ("replay_projected", "Replay Projected"),
    ("vector_x", "Vector X"),
    ("vector_y", "Vector Y"),
    ("vector_z", "Vector Z"),
)


@dataclass(frozen=True)
class DiffractionFieldComponentSummary:
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


def _time_call(func, /, *args, **kwargs):
    import time

    started = time.perf_counter()
    value = func(*args, **kwargs)
    return value, float(time.perf_counter() - started)


def _complex_field_to_numpy(field, grid_size: int) -> dict[str, np.ndarray]:
    real = np.asarray(field.real, dtype=np.float64).reshape(grid_size, grid_size)
    imag = np.asarray(field.imag, dtype=np.float64).reshape(grid_size, grid_size)
    mag = np.sqrt(real * real + imag * imag)
    return {
        "real": real,
        "imag": imag,
        "mag": mag,
        "power": mag * mag,
    }


def _complex_fd_map(plus, minus, *, grid_size: int, fd_step: float) -> dict[str, np.ndarray]:
    grad_real = (
        np.asarray(plus.real, dtype=np.float64).reshape(grid_size, grid_size)
        - np.asarray(minus.real, dtype=np.float64).reshape(grid_size, grid_size)
    ) / (2.0 * float(fd_step))
    grad_imag = (
        np.asarray(plus.imag, dtype=np.float64).reshape(grid_size, grid_size)
        - np.asarray(minus.imag, dtype=np.float64).reshape(grid_size, grid_size)
    ) / (2.0 * float(fd_step))
    grad_mag = np.sqrt(grad_real * grad_real + grad_imag * grad_imag)
    return {
        "real": grad_real,
        "imag": grad_imag,
        "mag": grad_mag,
    }


def _db_from_magnitude(values: np.ndarray, *, floor: float = 1.0e-20) -> np.ndarray:
    safe = np.where(np.isfinite(values), np.maximum(values, floor), np.nan)
    return 20.0 * np.log10(safe)


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
    minimum: float = 3.0,
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


def _ad_diffraction_field_components(
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

    coords, field_components, _replay_power, metadata = _trace_diffraction_breakdown_fields(
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
    component_names = tuple(field_components.keys())
    forward_maps = {
        name: _complex_field_to_numpy(field_components[name], grid_size)
        for name in component_names
    }
    dr.set_grad(parameter_value, 1.0)
    grads = dr.forward_to(
        *(field_components[name] for name in component_names),
        flags=_GRAD_FLAGS,
    )
    if not isinstance(grads, tuple):
        grads = (grads,)
    ad_maps = {
        name: _complex_field_to_numpy(grad, grid_size)
        for name, grad in zip(component_names, grads, strict=True)
    }
    return coords, forward_maps, ad_maps, metadata


def _fd_diffraction_field_components(
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
    _, plus_fields, _plus_power, _plus_metadata = _trace_diffraction_breakdown_fields(
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
    _, minus_fields, _minus_power, _minus_metadata = _trace_diffraction_breakdown_fields(
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
        name: _complex_fd_map(
            plus_fields[name],
            minus_fields[name],
            grid_size=grid_size,
            fd_step=fd_step,
        )
        for name in plus_fields
    }


def build_diffraction_field_component_benchmark(
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
    metadata = {}
    timings_seconds = {}
    grid_x = None
    grid_y = None
    for parameter in parameters:
        (coords, forward_maps, ad_maps, parameter_metadata), ad_seconds = _time_call(
            _ad_diffraction_field_components,
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
        fd_maps, fd_seconds = _time_call(
            _fd_diffraction_field_components,
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
            "forward": forward_maps,
            "ad": ad_maps,
            "fd": fd_maps,
        }
        metadata[parameter] = dict(parameter_metadata)
        timings_seconds[parameter] = {
            "ad": float(ad_seconds),
            "fd": float(fd_seconds),
        }

    summary = DiffractionFieldComponentSummary(
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
    component_names = tuple(name for name, _label in _FIELD_COMPONENTS)
    n_rows = 4 * len(parameters)
    n_cols = len(component_names)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(3.4 * n_cols, 3.0 * n_rows),
        constrained_layout=False,
        squeeze=False,
    )
    fig.subplots_adjust(
        left=0.02,
        right=0.985,
        bottom=0.03,
        top=0.92,
        wspace=0.06,
        hspace=0.18,
    )

    field_db_maps = []
    grad_db_maps = []
    diff_db_maps = []
    for parameter in parameters:
        row = benchmark["rows"][parameter]
        for component_name in component_names:
            field_db_maps.append(_db_from_magnitude(row["forward"][component_name]["mag"]))
            grad_db_maps.append(_db_from_magnitude(row["ad"][component_name]["mag"]))
            grad_db_maps.append(_db_from_magnitude(row["fd"][component_name]["mag"]))
            diff_db_maps.append(
                _db_from_magnitude(row["ad"][component_name]["mag"])
                - _db_from_magnitude(row["fd"][component_name]["mag"])
            )
    field_vmin, field_vmax = _auto_limits_many(field_db_maps, span=60.0)
    grad_vmin, grad_vmax = _auto_limits_many(grad_db_maps, span=55.0)
    diff_vmin, diff_vmax = _symmetric_limits_many(diff_db_maps)

    field_axes = []
    grad_axes = []
    diff_axes = []
    field_handle = None
    grad_handle = None
    diff_handle = None

    for parameter_index, parameter in enumerate(parameters):
        row = benchmark["rows"][parameter]
        timings = benchmark["summary"].timings_seconds[parameter]
        row_base = 4 * parameter_index
        row_specs = (
            ("Forward", row["forward"], None, field_vmin, field_vmax, "viridis"),
            (
                f"AD\n{timings['ad']:.2f}s",
                row["ad"],
                None,
                grad_vmin,
                grad_vmax,
                "magma",
            ),
            (
                f"FD\n{timings['fd']:.2f}s",
                row["fd"],
                None,
                grad_vmin,
                grad_vmax,
                "magma",
            ),
            ("AD-FD", row["ad"], row["fd"], diff_vmin, diff_vmax, "RdBu_r"),
        )
        for local_row, (row_label, maps, diff_against, vmin, vmax, cmap) in enumerate(row_specs):
            for col_idx, (component_name, component_label) in enumerate(_FIELD_COMPONENTS):
                ax = axes[row_base + local_row, col_idx]
                panel_data = _db_from_magnitude(maps[component_name]["mag"])
                if diff_against is not None:
                    panel_data = panel_data - _db_from_magnitude(diff_against[component_name]["mag"])
                image = ax.imshow(
                    panel_data,
                    origin="lower",
                    extent=benchmark["summary"].bounds[0] + benchmark["summary"].bounds[1],
                    cmap=cmap,
                    vmin=vmin,
                    vmax=vmax,
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
                if local_row == 0:
                    rel_l2 = _relative_l2_error(
                        row["ad"][component_name]["mag"],
                        row["fd"][component_name]["mag"],
                    )
                    corr = _safe_correlation(
                        row["ad"][component_name]["mag"],
                        row["fd"][component_name]["mag"],
                    )
                    ax.set_title(
                        f"{component_label}\ncorr={corr:.3f} rel-L2={rel_l2:.2e}",
                        fontsize=11,
                        pad=7.0,
                    )
                if col_idx == 0:
                    ax.set_ylabel(f"{parameter}\n{row_label}", fontsize=11)
                if local_row == 0:
                    field_axes.append(ax)
                    field_handle = image
                elif local_row in (1, 2):
                    grad_axes.append(ax)
                    grad_handle = image
                else:
                    diff_axes.append(ax)
                    diff_handle = image

    if field_handle is not None and field_axes:
        fig.colorbar(
            field_handle,
            ax=field_axes,
            shrink=0.84,
            label="Forward diffraction field magnitude [dB]",
        )
    if grad_handle is not None and grad_axes:
        fig.colorbar(
            grad_handle,
            ax=grad_axes,
            shrink=0.84,
            label="Complex-field gradient magnitude [dB]",
        )
    if diff_handle is not None and diff_axes:
        fig.colorbar(
            diff_handle,
            ax=diff_axes,
            shrink=0.84,
            label="AD minus FD on gradient magnitude [dB]",
        )

    fig.suptitle(
        (
            "Three-Cube Radiomap Diffraction Field Components\n"
            f"grid={benchmark['summary'].grid_size}x{benchmark['summary'].grid_size}, "
            f"rays={benchmark['summary'].n_rays}, "
            f"z={benchmark['summary'].plane_z:.1f}, "
            f"shadow={benchmark['summary'].shadow_boundary_mode}"
        ),
        fontsize=16,
        y=0.985,
    )
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def save_json(benchmark, *, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary = asdict(benchmark["summary"])
    summary["parameters_summary"] = {}
    for parameter in benchmark["summary"].parameters:
        row = benchmark["rows"][parameter]
        parameter_summary = {}
        for component_name, component_label in _FIELD_COMPONENTS:
            forward_mag = row["forward"][component_name]["mag"]
            ad_mag = row["ad"][component_name]["mag"]
            fd_mag = row["fd"][component_name]["mag"]
            parameter_summary[component_name] = {
                "label": component_label,
                "forward_peak_db": float(np.nanmax(_db_from_magnitude(forward_mag))),
                "forward_abs_sum": float(np.nansum(np.abs(forward_mag))),
                "ad_abs_sum": float(np.nansum(np.abs(ad_mag))),
                "fd_abs_sum": float(np.nansum(np.abs(fd_mag))),
                "ad_fd_corr": float(_safe_correlation(ad_mag, fd_mag)),
                "ad_fd_rel_l2": float(_relative_l2_error(ad_mag, fd_mag)),
                "ad_fd_mean_abs_diff": float(np.nanmean(np.abs(ad_mag - fd_mag))),
                "ad_fd_max_abs_diff": float(np.nanmax(np.abs(ad_mag - fd_mag))),
            }
        summary["parameters_summary"][parameter] = {
            "components": parameter_summary,
            "metadata": benchmark["metadata"][parameter],
        }
    output_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return output_path


def save_component_benchmark(
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
    benchmark = build_diffraction_field_component_benchmark(
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
    figure_path = save_figure(benchmark, output_path=output_prefix.with_suffix(".png"))
    json_path = save_json(benchmark, output_path=output_prefix.with_suffix(".json"))
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
    figure_path, json_path = save_component_benchmark(
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
