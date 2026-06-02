"""Save AD/FD gradient maps for each direct-recursive pair in the main scene."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import drjit as dr
import matplotlib.pyplot as plt
import numpy as np
import witwin as wt

from samples.save_multipath_main_component_gradient_figure import (
    CALC_HEIGHT,
    FLAGS,
    K,
    MAX_DIFFRACTIONS,
    MAX_REFLECTIONS,
    REFLECTION_COEF,
    TRACE_BOUNDS,
    TX_POLARIZATION,
    WAVELENGTH,
    _accumulate_subset_field,
    _auto_limits_many,
    _build_parameter_state,
    _component_maps_from_ad_result,
    _grad_db_grid,
    _panel_stats_text,
    _parse_parameter,
    _plot_panel,
    _relative_l2_error,
)
from witwin.channel import Field, to_numpy
from witwin.channel.trace import compute_reflection_field
from witwin.channel.trace.diffraction import (
    APPROX_MODE_RECURSIVE_DIFFRACTION,
    OWNERSHIP_DIRECT_DIFFRACTION,
    _ownership_code_from_depths,
    _prepare_diffraction_state_arrays,
)
from witwin.channel.kernels.trace.packed_state import subset_state_arrays


def _zero_field(width: int) -> wt.Complex2f:
    return wt.Complex2f(dr.zeros(wt.Float, width), dr.zeros(wt.Float, width))


def _direct_recursive_pair_specs(state_arrays) -> list[dict]:
    if state_arrays["n_states"] == 0:
        return []
    ownership = _ownership_code_from_depths(
        state_arrays["prefix_reflection_depth"],
        state_arrays["intermediate_reflection_depth"],
        state_arrays["suffix_reflection_depth"],
    )
    recursive_mask = (
        (ownership == wt.UInt32(OWNERSHIP_DIRECT_DIFFRACTION))
        & (state_arrays["approximation_mode_code"] == wt.UInt32(APPROX_MODE_RECURSIVE_DIFFRACTION))
    )
    mask_np = np.asarray(to_numpy(recursive_mask), dtype=bool)
    if not np.any(mask_np):
        return []

    path_edge_0 = np.asarray(to_numpy(state_arrays["path_edge_idx_0"]), dtype=np.int32)
    edge_idx = np.asarray(to_numpy(state_arrays["edge_idx"]), dtype=np.int32)
    specs = []
    for row_idx, state_idx in enumerate(np.flatnonzero(mask_np)):
        prev_edge = int(path_edge_0[state_idx])
        curr_edge = int(edge_idx[state_idx])
        specs.append(
            {
                "component_name": f"a_dif_direct_recursive_pair_{row_idx:02d}_{prev_edge}_{curr_edge}",
                "label": f"{prev_edge}->{curr_edge}",
                "prev_edge": prev_edge,
                "edge_idx": curr_edge,
            }
        )
    return specs


def _pair_mask(state_arrays, spec: dict):
    ownership = _ownership_code_from_depths(
        state_arrays["prefix_reflection_depth"],
        state_arrays["intermediate_reflection_depth"],
        state_arrays["suffix_reflection_depth"],
    )
    return (
        (ownership == wt.UInt32(OWNERSHIP_DIRECT_DIFFRACTION))
        & (state_arrays["approximation_mode_code"] == wt.UInt32(APPROX_MODE_RECURSIVE_DIFFRACTION))
        & (state_arrays["path_edge_idx_0"] == wt.Int32(spec["prev_edge"]))
        & (state_arrays["edge_idx"] == wt.UInt32(spec["edge_idx"]))
    )


def _build_direct_recursive_pair_fields(
    scene,
    tx_point: wt.Point3f,
    grid_size: int,
    n_rays: int,
    *,
    pair_specs: list[dict] | None = None,
):
    field = Field(bounds=TRACE_BOUNDS, size=(grid_size, grid_size))
    coords = field.get_coordinates()
    rx_pos = wt.Point3f(coords["X"], coords["Y"], wt.Float(CALC_HEIGHT))
    n_rx = field.n_cells

    _, _, reflection_detail = compute_reflection_field(
        grid=field,
        rx_z=CALC_HEIGHT,
        tx_pos=tx_point,
        scene=scene,
        wavelength=WAVELENGTH,
        k=K,
        n_rays=n_rays,
        max_reflections=MAX_REFLECTIONS,
        mode="2d",
        reflection_coef=REFLECTION_COEF,
        tx_polarization=TX_POLARIZATION,
        return_per_bounce=False,
        grid_data=coords,
    )
    edge_cache, edge_data, state_arrays, _ = _prepare_diffraction_state_arrays(
        tx_point,
        CALC_HEIGHT,
        scene,
        WAVELENGTH,
        K,
        reflection_detail,
        None,
        n_rays,
        MAX_REFLECTIONS,
        REFLECTION_COEF,
        "2d",
        MAX_DIFFRACTIONS,
        tx_polarization=TX_POLARIZATION,
    )
    draw_cache = scene.get_edge_data(CALC_HEIGHT)
    if pair_specs is None:
        pair_specs = _direct_recursive_pair_specs(state_arrays)

    zero = _zero_field(n_rx)
    if edge_data is None or state_arrays["n_states"] == 0:
        return {
            "fields": {spec["component_name"]: zero for spec in pair_specs},
            "pair_specs": pair_specs,
            "draw_cache": draw_cache,
        }

    fields = {}
    for spec in pair_specs:
        subset = subset_state_arrays(state_arrays, _pair_mask(state_arrays, spec))
        fields[spec["component_name"]] = _accumulate_subset_field(
            subset,
            rx_pos,
            edge_data["n_edges"],
            scene,
            ownership_bucket="direct",
        )
    return {
        "fields": fields,
        "pair_specs": pair_specs,
        "draw_cache": draw_cache,
    }


def _compute_ad_pair_maps(parameter: dict, grid_size: int, n_rays: int):
    scene, tx_point, centers, selected_center = _build_parameter_state(parameter, enable_grad=True)
    payload = _build_direct_recursive_pair_fields(scene, tx_point, grid_size, n_rays)
    component_names = tuple(spec["component_name"] for spec in payload["pair_specs"])
    if component_names:
        dr.forward_to(tuple(payload["fields"][name] for name in component_names), flags=FLAGS)
    maps = _component_maps_from_ad_result(payload["fields"], component_names, grid_size)
    return payload, maps, selected_center


def _compute_fd_pair_maps(
    parameter: dict,
    grid_size: int,
    n_rays: int,
    fd_step: float,
    pair_specs: list[dict],
):
    scene_plus, tx_plus, _, _ = _build_parameter_state(parameter, delta=fd_step, enable_grad=False)
    scene_minus, tx_minus, _, _ = _build_parameter_state(parameter, delta=-fd_step, enable_grad=False)
    payload_plus = _build_direct_recursive_pair_fields(
        scene_plus,
        tx_plus,
        grid_size,
        n_rays,
        pair_specs=pair_specs,
    )
    payload_minus = _build_direct_recursive_pair_fields(
        scene_minus,
        tx_minus,
        grid_size,
        n_rays,
        pair_specs=pair_specs,
    )

    maps = {}
    for spec in pair_specs:
        name = spec["component_name"]
        plus = payload_plus["fields"][name]
        minus = payload_minus["fields"][name]
        grad_real = (to_numpy(plus.real) - to_numpy(minus.real)) / (2.0 * fd_step)
        grad_imag = (to_numpy(plus.imag) - to_numpy(minus.imag)) / (2.0 * fd_step)
        grad_real = grad_real.reshape(grid_size, grid_size)
        grad_imag = grad_imag.reshape(grid_size, grid_size)
        maps[name] = {
            "real": grad_real,
            "imag": grad_imag,
            "mag": np.sqrt(grad_real * grad_real + grad_imag * grad_imag),
        }
    return maps


def _sorted_pair_specs(
    pair_specs: list[dict],
    fd_maps: dict[str, dict[str, np.ndarray]],
    ad_maps: dict[str, dict[str, np.ndarray]],
    max_pairs: int | None,
) -> list[dict]:
    enriched = []
    for spec in pair_specs:
        name = spec["component_name"]
        fd_norm = float(np.linalg.norm(fd_maps[name]["mag"].ravel()))
        rel_l2 = _relative_l2_error(ad_maps[name]["mag"], fd_maps[name]["mag"])
        enriched.append({**spec, "fd_norm": fd_norm, "rel_l2": rel_l2})
    enriched.sort(key=lambda item: item["fd_norm"], reverse=True)
    if max_pairs is not None:
        return enriched[: max(0, int(max_pairs))]
    return enriched


def make_figure(
    *,
    parameter_name: str,
    output_path: Path,
    grid_size: int,
    n_rays: int,
    fd_step: float,
    max_pairs: int | None,
):
    parameter = _parse_parameter(parameter_name)
    ad_payload, ad_maps, selected_center = _compute_ad_pair_maps(parameter, grid_size, n_rays)
    pair_specs = ad_payload["pair_specs"]
    fd_maps = _compute_fd_pair_maps(parameter, grid_size, n_rays, fd_step, pair_specs)
    pair_specs = _sorted_pair_specs(pair_specs, fd_maps, ad_maps, max_pairs)

    gradient_db_maps = []
    diff_db_maps = []
    for spec in pair_specs:
        name = spec["component_name"]
        ad_db = _grad_db_grid(ad_maps[name]["mag"])
        fd_db = _grad_db_grid(fd_maps[name]["mag"])
        gradient_db_maps.extend((ad_db, fd_db))
        diff_db_maps.append(ad_db - fd_db)
    if not gradient_db_maps:
        raise RuntimeError("No direct-recursive pair components were available to plot.")

    grad_vmin, grad_vmax = _auto_limits_many(gradient_db_maps, span=55.0)
    diff_abs = np.concatenate([np.abs(data).ravel() for data in diff_db_maps], axis=0)
    diff_abs = diff_abs[np.isfinite(diff_abs)]
    diff_vmax = max(float(np.percentile(diff_abs, 99.0)), 3.0) if diff_abs.size else 3.0
    diff_vmin = -diff_vmax

    edges = ad_payload["draw_cache"]["edges_2d"]
    corners = ad_payload["draw_cache"]["corners_2d"]

    fig_height = max(8.0, 2.9 * len(pair_specs))
    fig, axes = plt.subplots(len(pair_specs), 3, figsize=(11.0, fig_height), constrained_layout=True, squeeze=False)
    handles = [None, None, None]

    for row_idx, spec in enumerate(pair_specs):
        name = spec["component_name"]
        ad_db = _grad_db_grid(ad_maps[name]["mag"])
        fd_db = _grad_db_grid(fd_maps[name]["mag"])
        diff_db = ad_db - fd_db
        rel_l2 = spec["rel_l2"]

        ad_title = f"AD\nrel-L2={rel_l2:.2e}" if row_idx == 0 else ""
        fd_title = "FD" if row_idx == 0 else ""
        diff_title = f"AD - FD\n{_panel_stats_text(diff_db)}" if row_idx == 0 else ""

        handles[0] = _plot_panel(
            axes[row_idx, 0],
            ad_db,
            ad_title,
            edges,
            corners,
            selected_center,
            grad_vmin,
            grad_vmax,
            "magma",
        )
        handles[1] = _plot_panel(
            axes[row_idx, 1],
            fd_db,
            fd_title,
            edges,
            corners,
            selected_center,
            grad_vmin,
            grad_vmax,
            "magma",
        )
        handles[2] = _plot_panel(
            axes[row_idx, 2],
            diff_db,
            diff_title,
            edges,
            corners,
            selected_center,
            diff_vmin,
            diff_vmax,
            "RdBu_r",
        )

        axes[row_idx, 0].set_ylabel(
            f"Rec {spec['label']}\n|FD|={spec['fd_norm']:.2e}",
            fontsize=9,
        )
        if row_idx == len(pair_specs) - 1:
            xticks = np.linspace(TRACE_BOUNDS[0][0], TRACE_BOUNDS[0][1], 5)
            for col_idx in range(3):
                axes[row_idx, col_idx].set_xticks(xticks)
                axes[row_idx, col_idx].set_xticklabels([f"{value:.0f}" for value in xticks])

    fig.colorbar(handles[0], ax=axes[:, :2], shrink=0.92, label="Complex-field gradient magnitude [dB]")
    fig.colorbar(handles[2], ax=axes[:, 2], shrink=0.92, label="Signed difference [dB] on gradient magnitude")
    fig.suptitle(
        "Direct Recursive Diffraction Pair Breakdown\n"
        f"parameter={parameter['label']}, grid={grid_size}, n_rays={n_rays}, fd_step={fd_step}, "
        f"pairs={len(pair_specs)}",
        fontsize=14,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)

    print(f"Saved figure to {output_path}")
    for spec in pair_specs:
        print(
            f"  {spec['label']}: fd_norm={spec['fd_norm']:.6e}, rel-L2={spec['rel_l2']:.6e}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parameter", default="cube1_x", choices=("tx_x", "cube1_x"))
    parser.add_argument("--grid-size", type=int, default=64)
    parser.add_argument("--n-rays", type=int, default=640)
    parser.add_argument("--fd-step", type=float, default=1e-3)
    parser.add_argument("--max-pairs", type=int, default=None)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("tests/output/multipath_cube1_x_direct_recursive_pairs.png"),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    make_figure(
        parameter_name=args.parameter,
        output_path=args.output,
        grid_size=args.grid_size,
        n_rays=args.n_rays,
        fd_step=args.fd_step,
        max_pairs=args.max_pairs,
    )


if __name__ == "__main__":
    main()
