"""Save AD/FD gradient maps for grouped Mix Insert 1 components in the main scene."""

from __future__ import annotations

import argparse
from collections import OrderedDict
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
    APPROX_MODE_SAMPLED_INSERTED_REFLECTION,
    OWNERSHIP_MIXED_DIFFRACTION,
    _ownership_code_from_depths,
    _prepare_diffraction_state_arrays,
)
from witwin.channel.kernels.trace.packed_state import subset_state_arrays
from witwin.channel.trace.diffraction.state import _build_state_audit


def _zero_field(width: int) -> wt.Complex2f:
    return wt.Complex2f(dr.zeros(wt.Float, width), dr.zeros(wt.Float, width))


def _mixed_inserted_base_mask(state_arrays):
    ownership = _ownership_code_from_depths(
        state_arrays["prefix_reflection_depth"],
        state_arrays["intermediate_reflection_depth"],
        state_arrays["suffix_reflection_depth"],
    )
    return (
        (ownership == wt.UInt32(OWNERSHIP_MIXED_DIFFRACTION))
        & (state_arrays["approximation_mode_code"] == wt.UInt32(APPROX_MODE_SAMPLED_INSERTED_REFLECTION))
    )


def _edge_global_idx_array(state_arrays, edge_data):
    if edge_data is None or edge_data.get("global_idx") is None:
        return wt.Int32(state_arrays["edge_idx"])
    return dr.gather(wt.Int32, edge_data["global_idx"], state_arrays["edge_idx"])


def _mixed_inserted_group_specs(state_arrays, edge_data) -> list[dict]:
    if edge_data is None or state_arrays["n_states"] == 0:
        return []

    base_mask = _mixed_inserted_base_mask(state_arrays)
    if not bool(dr.any(base_mask)):
        return []

    subset = subset_state_arrays(state_arrays, base_mask)
    audit = _build_state_audit(subset, edge_data)
    specs_by_key: OrderedDict[tuple, dict] = OrderedDict()
    history_size = int(audit["history_size"])
    for state_idx in range(subset["n_states"]):
        path_reflection_depths = tuple(
            int(audit[f"path_reflection_depth_{slot}"][state_idx])
            for slot in range(history_size)
        )
        key = (
            int(audit["edge_global_idx"][state_idx]),
            int(audit["order"][state_idx]),
            int(audit["prefix_reflection_depth"][state_idx]),
            int(audit["intermediate_reflection_depth"][state_idx]),
            int(audit["suffix_reflection_depth"][state_idx]),
            path_reflection_depths,
        )
        if key not in specs_by_key:
            path_sequence = str(audit["path_sequence"][state_idx])
            specs_by_key[key] = {
                "edge_global_idx": key[0],
                "order": key[1],
                "prefix_reflection_depth": key[2],
                "intermediate_reflection_depth": key[3],
                "suffix_reflection_depth": key[4],
                "path_reflection_depths": path_reflection_depths,
                "path_sequence": path_sequence,
                "count": 0,
            }
        specs_by_key[key]["count"] += 1

    specs = []
    for row_idx, (_, spec) in enumerate(specs_by_key.items()):
        compact_sequence = spec["path_sequence"].replace(" -> ", "-")
        specs.append(
            {
                **spec,
                "component_name": (
                    f"a_dif_mixed_inserted_group_{row_idx:02d}_"
                    f"edge_{spec['edge_global_idx']}"
                ),
                "label": f"e{spec['edge_global_idx']} | {compact_sequence}",
            }
        )
    return specs


def _group_mask(state_arrays, edge_data, spec: dict):
    mask = _mixed_inserted_base_mask(state_arrays)
    edge_global_idx = _edge_global_idx_array(state_arrays, edge_data)
    mask &= edge_global_idx == wt.Int32(spec["edge_global_idx"])
    mask &= state_arrays["order"] == wt.UInt32(spec["order"])
    mask &= state_arrays["prefix_reflection_depth"] == wt.UInt32(spec["prefix_reflection_depth"])
    mask &= state_arrays["intermediate_reflection_depth"] == wt.UInt32(spec["intermediate_reflection_depth"])
    mask &= state_arrays["suffix_reflection_depth"] == wt.UInt32(spec["suffix_reflection_depth"])
    for slot, reflection_depth in enumerate(spec["path_reflection_depths"]):
        mask &= state_arrays[f"path_reflection_depth_{slot}"] == wt.UInt32(reflection_depth)
    return mask


def _build_mixed_inserted_group_fields(
    scene,
    tx_point: wt.Point3f,
    grid_size: int,
    n_rays: int,
    *,
    group_specs: list[dict] | None = None,
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
    _, edge_data, state_arrays, _ = _prepare_diffraction_state_arrays(
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
    if group_specs is None:
        group_specs = _mixed_inserted_group_specs(state_arrays, edge_data)

    zero = _zero_field(n_rx)
    if edge_data is None or state_arrays["n_states"] == 0:
        return {
            "fields": {spec["component_name"]: zero for spec in group_specs},
            "group_specs": group_specs,
            "draw_cache": draw_cache,
        }

    fields = {}
    for spec in group_specs:
        subset = subset_state_arrays(state_arrays, _group_mask(state_arrays, edge_data, spec))
        fields[spec["component_name"]] = _accumulate_subset_field(
            subset,
            rx_pos,
            edge_data["n_edges"],
            scene,
            ownership_bucket="mixed",
        )
    return {
        "fields": fields,
        "group_specs": group_specs,
        "draw_cache": draw_cache,
    }


def _compute_ad_group_maps(parameter: dict, grid_size: int, n_rays: int):
    scene, tx_point, centers, selected_center = _build_parameter_state(parameter, enable_grad=True)
    payload = _build_mixed_inserted_group_fields(scene, tx_point, grid_size, n_rays)
    component_names = tuple(spec["component_name"] for spec in payload["group_specs"])
    if component_names:
        dr.forward_to(tuple(payload["fields"][name] for name in component_names), flags=FLAGS)
    maps = _component_maps_from_ad_result(payload["fields"], component_names, grid_size)
    return payload, maps, selected_center


def _compute_fd_group_maps(
    parameter: dict,
    grid_size: int,
    n_rays: int,
    fd_step: float,
    group_specs: list[dict],
):
    scene_plus, tx_plus, _, _ = _build_parameter_state(parameter, delta=fd_step, enable_grad=False)
    scene_minus, tx_minus, _, _ = _build_parameter_state(parameter, delta=-fd_step, enable_grad=False)
    payload_plus = _build_mixed_inserted_group_fields(
        scene_plus,
        tx_plus,
        grid_size,
        n_rays,
        group_specs=group_specs,
    )
    payload_minus = _build_mixed_inserted_group_fields(
        scene_minus,
        tx_minus,
        grid_size,
        n_rays,
        group_specs=group_specs,
    )

    maps = {}
    for spec in group_specs:
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


def _sorted_group_specs(
    group_specs: list[dict],
    fd_maps: dict[str, dict[str, np.ndarray]],
    ad_maps: dict[str, dict[str, np.ndarray]],
    max_groups: int | None,
) -> list[dict]:
    enriched = []
    for spec in group_specs:
        name = spec["component_name"]
        fd_norm = float(np.linalg.norm(fd_maps[name]["mag"].ravel()))
        rel_l2 = _relative_l2_error(ad_maps[name]["mag"], fd_maps[name]["mag"])
        enriched.append({**spec, "fd_norm": fd_norm, "rel_l2": rel_l2})
    enriched.sort(key=lambda item: item["fd_norm"], reverse=True)
    if max_groups is not None:
        return enriched[: max(0, int(max_groups))]
    return enriched


def make_figure(
    *,
    parameter_name: str,
    output_path: Path,
    grid_size: int,
    n_rays: int,
    fd_step: float,
    max_groups: int | None,
):
    parameter = _parse_parameter(parameter_name)
    ad_payload, ad_maps, selected_center = _compute_ad_group_maps(parameter, grid_size, n_rays)
    group_specs = ad_payload["group_specs"]
    fd_maps = _compute_fd_group_maps(parameter, grid_size, n_rays, fd_step, group_specs)
    group_specs = _sorted_group_specs(group_specs, fd_maps, ad_maps, max_groups)

    gradient_db_maps = []
    diff_db_maps = []
    for spec in group_specs:
        name = spec["component_name"]
        ad_db = _grad_db_grid(ad_maps[name]["mag"])
        fd_db = _grad_db_grid(fd_maps[name]["mag"])
        gradient_db_maps.extend((ad_db, fd_db))
        diff_db_maps.append(ad_db - fd_db)
    if not gradient_db_maps:
        raise RuntimeError("No Mix Insert 1 grouped components were available to plot.")

    grad_vmin, grad_vmax = _auto_limits_many(gradient_db_maps, span=55.0)
    diff_abs = np.concatenate([np.abs(data).ravel() for data in diff_db_maps], axis=0)
    diff_abs = diff_abs[np.isfinite(diff_abs)]
    diff_vmax = max(float(np.percentile(diff_abs, 99.0)), 3.0) if diff_abs.size else 3.0
    diff_vmin = -diff_vmax

    edges = ad_payload["draw_cache"]["edges_2d"]
    corners = ad_payload["draw_cache"]["corners_2d"]

    fig_height = max(8.0, 2.9 * len(group_specs))
    fig, axes = plt.subplots(len(group_specs), 3, figsize=(11.5, fig_height), constrained_layout=True, squeeze=False)
    handles = [None, None, None]

    for row_idx, spec in enumerate(group_specs):
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
            f"{spec['label']}\nN={spec['count']} |FD|={spec['fd_norm']:.2e}",
            fontsize=8.5,
        )
        if row_idx == len(group_specs) - 1:
            xticks = np.linspace(TRACE_BOUNDS[0][0], TRACE_BOUNDS[0][1], 5)
            for col_idx in range(3):
                axes[row_idx, col_idx].set_xticks(xticks)
                axes[row_idx, col_idx].set_xticklabels([f"{value:.0f}" for value in xticks])

    fig.colorbar(handles[0], ax=axes[:, :2], shrink=0.92, label="Complex-field gradient magnitude [dB]")
    fig.colorbar(handles[2], ax=axes[:, 2], shrink=0.92, label="Signed difference [dB] on gradient magnitude")
    fig.suptitle(
        "Mix Insert 1 Breakdown\n"
        f"parameter={parameter['label']}, grid={grid_size}, n_rays={n_rays}, fd_step={fd_step}, "
        f"groups={len(group_specs)}",
        fontsize=14,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)

    print(f"Saved figure to {output_path}")
    for spec in group_specs:
        print(
            f"  {spec['label']}: count={spec['count']}, "
            f"fd_norm={spec['fd_norm']:.6e}, rel-L2={spec['rel_l2']:.6e}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parameter", default="cube1_x", choices=("tx_x", "cube1_x"))
    parser.add_argument("--grid-size", type=int, default=64)
    parser.add_argument("--n-rays", type=int, default=640)
    parser.add_argument("--fd-step", type=float, default=1e-3)
    parser.add_argument("--max-groups", type=int, default=12)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("tests/output/multipath_cube1_x_mixed_inserted_breakdown.png"),
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
        max_groups=args.max_groups,
    )


if __name__ == "__main__":
    main()
