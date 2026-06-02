"""Pure witwin forward three-cube radio-map benchmark helper."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

from tests.main.plot_multipath_components import CUBE1_BASE_CENTER
from tests.main.plot_radiomap_sionna_three_cubes import (
    DEFAULT_BOUNDS,
    DEFAULT_DB_MAX,
    DEFAULT_DB_MIN,
    DEFAULT_DIFF_RATIO_RANGE_DB,
    DEFAULT_GRID_SIZE,
    DEFAULT_N_RAYS,
    DEFAULT_PLANE_Z,
    DEFAULT_TX_POS,
    DEFAULT_WITWIN_COMBINE_MODE,
    DEFAULT_WITWIN_EDGE_SELECTION_MODE,
    DEFAULT_WITWIN_PROFILE,
    DEFAULT_WITWIN_RECEIVER_MODEL,
    DEFAULT_WITWIN_SHADOW_BOUNDARY_MODE,
    DEFAULT_WITWIN_SHADOW_SUPPORT_CUTOFF_DB,
    _absolute_db_map,
    _build_comparison_scene,
    _decorate_axis,
    _extent,
    _output_dir,
    _ratio_db_map,
    _resolve_witwin_profile,
    _run_witwin,
    _to_float_grid,
    _witwin_diff_component_label,
    _witwin_diff_component_map,
)


@dataclass(frozen=True)
class ForwardSummary:
    grid_size: int
    bounds: tuple[tuple[float, float], tuple[float, float]]
    plane_z: float
    tx_pos: tuple[float, float, float]
    n_rays: int
    db_min: float
    db_max: float
    witwin_no_diff_seconds: float
    witwin_with_diff_seconds: float
    witwin_profile: str
    witwin_profile_label: str
    witwin_combine_mode: str
    witwin_receiver_model: str
    witwin_shadow_boundary_mode: str
    witwin_shadow_support_cutoff_db: float | None
    witwin_edge_selection_mode: str
    witwin_metric_contract: str
    witwin_path_counts: dict
    witwin_runtime_backends: dict


def build_forward_benchmark(
    *,
    grid_size: int,
    bounds,
    plane_z: float,
    tx_pos,
    n_rays: int,
    db_min: float,
    db_max: float,
    diff_ratio_range_db: float,
    witwin_profile: str,
    witwin_combine_mode: str,
    witwin_receiver_model: str,
    witwin_shadow_boundary_mode: str,
    witwin_shadow_support_cutoff_db: float | None,
):
    cube1_x = float(CUBE1_BASE_CENTER[0])
    scene = _build_comparison_scene(
        cube1_x,
        edge_selection_mode=DEFAULT_WITWIN_EDGE_SELECTION_MODE,
    )
    (
        resolved_witwin_profile,
        resolved_witwin_profile_label,
        resolved_witwin_combine_mode,
        resolved_witwin_receiver_model,
        resolved_witwin_shadow_boundary_mode,
    ) = _resolve_witwin_profile(
        profile=witwin_profile,
        combine_mode=witwin_combine_mode,
        receiver_model=witwin_receiver_model,
        shadow_boundary_mode=witwin_shadow_boundary_mode,
    )

    witwin_with_diff, witwin_with_diff_seconds = _run_witwin(
        scene=scene,
        tx_pos=tx_pos,
        plane_z=plane_z,
        bounds=bounds,
        grid_size=grid_size,
        n_rays=n_rays,
        max_diffractions=2,
        combine_mode=resolved_witwin_combine_mode,
        receiver_model=resolved_witwin_receiver_model,
        shadow_boundary_mode=resolved_witwin_shadow_boundary_mode,
        shadow_support_cutoff_db=witwin_shadow_support_cutoff_db,
    )
    witwin_no_diff, witwin_no_diff_seconds = _run_witwin(
        scene=scene,
        tx_pos=tx_pos,
        plane_z=plane_z,
        bounds=bounds,
        grid_size=grid_size,
        n_rays=n_rays,
        max_diffractions=0,
        combine_mode=resolved_witwin_combine_mode,
        receiver_model=resolved_witwin_receiver_model,
        shadow_boundary_mode=resolved_witwin_shadow_boundary_mode,
        shadow_support_cutoff_db=witwin_shadow_support_cutoff_db,
    )

    witwin_total = _to_float_grid(witwin_with_diff.path_gain)
    witwin_total_no_diff = _to_float_grid(witwin_no_diff.path_gain)
    witwin_diff_component = _witwin_diff_component_map(witwin_with_diff)
    witwin_diff_increment = np.maximum(witwin_total - witwin_total_no_diff, 0.0)

    witwin_total_db = _absolute_db_map(witwin_total, floor_db=db_min)
    witwin_total_no_diff_db = _absolute_db_map(witwin_total_no_diff, floor_db=db_min)
    witwin_diff_db = _absolute_db_map(witwin_diff_component, floor_db=db_min)
    witwin_diff_increment_db = _absolute_db_map(witwin_diff_increment, floor_db=db_min)
    witwin_diff_ratio_db = _ratio_db_map(
        witwin_diff_component,
        witwin_total,
        dynamic_range_db=diff_ratio_range_db,
    )

    summary = ForwardSummary(
        grid_size=int(grid_size),
        bounds=tuple((float(a), float(b)) for (a, b) in bounds),
        plane_z=float(plane_z),
        tx_pos=tuple(float(value) for value in tx_pos),
        n_rays=int(n_rays),
        db_min=float(db_min),
        db_max=float(db_max),
        witwin_no_diff_seconds=float(witwin_no_diff_seconds),
        witwin_with_diff_seconds=float(witwin_with_diff_seconds),
        witwin_profile=resolved_witwin_profile,
        witwin_profile_label=resolved_witwin_profile_label,
        witwin_combine_mode=resolved_witwin_combine_mode,
        witwin_receiver_model=resolved_witwin_receiver_model,
        witwin_shadow_boundary_mode=resolved_witwin_shadow_boundary_mode,
        witwin_shadow_support_cutoff_db=(
            None
            if witwin_shadow_support_cutoff_db is None
            else float(witwin_shadow_support_cutoff_db)
        ),
        witwin_edge_selection_mode=str(DEFAULT_WITWIN_EDGE_SELECTION_MODE),
        witwin_metric_contract=str(
            witwin_with_diff.metadata.get("metric_contract", {}).get("path_gain", "")
        ),
        witwin_path_counts=dict(witwin_with_diff.metadata.get("path_counts", {})),
        witwin_runtime_backends=dict(witwin_with_diff.metadata.get("runtime_backends", {})),
    )

    return {
        "cube1_x": cube1_x,
        "extent": _extent(bounds),
        "summary": summary,
        "witwin": {
            "total": witwin_total,
            "total_no_diff": witwin_total_no_diff,
            "diff_component": witwin_diff_component,
            "diff_component_label": _witwin_diff_component_label(witwin_with_diff),
            "diff_increment": witwin_diff_increment,
            "total_db": witwin_total_db,
            "total_no_diff_db": witwin_total_no_diff_db,
            "diff_db": witwin_diff_db,
            "diff_increment_db": witwin_diff_increment_db,
            "diff_ratio_db": witwin_diff_ratio_db,
            "grid_x": _to_float_grid(witwin_with_diff.coords.grid_x),
            "grid_y": _to_float_grid(witwin_with_diff.coords.grid_y),
        },
    }


def save_figure(forward_benchmark, *, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 9.0), constrained_layout=True)
    extent = forward_benchmark["extent"]
    bounds = forward_benchmark["summary"].bounds
    tx_pos = forward_benchmark["summary"].tx_pos
    cube1_x = forward_benchmark["cube1_x"]
    db_min = float(forward_benchmark["summary"].db_min)
    db_max = float(forward_benchmark["summary"].db_max)
    witwin_with_diff_seconds = float(forward_benchmark["summary"].witwin_with_diff_seconds)
    witwin_no_diff_seconds = float(forward_benchmark["summary"].witwin_no_diff_seconds)
    diff_label = str(forward_benchmark["witwin"]["diff_component_label"])

    panels = (
        (
            axes[0, 0],
            forward_benchmark["witwin"]["total_db"],
            f"Witwin Total (dB, {witwin_with_diff_seconds:.3f}s)",
            "viridis",
            db_min,
            db_max,
        ),
        (
            axes[0, 1],
            forward_benchmark["witwin"]["total_no_diff_db"],
            f"Witwin Total, No Diff (dB, {witwin_no_diff_seconds:.3f}s)",
            "viridis",
            db_min,
            db_max,
        ),
        (
            axes[1, 0],
            forward_benchmark["witwin"]["diff_db"],
            f"Witwin {diff_label} (dB, {witwin_with_diff_seconds:.3f}s)",
            "magma",
            db_min,
            db_max,
        ),
        (
            axes[1, 1],
            forward_benchmark["witwin"]["diff_ratio_db"],
            f"{diff_label} / Total (dB)",
            "cividis",
            -30.0,
            0.0,
        ),
    )

    for ax, image, title, cmap, vmin, vmax in panels:
        im = ax.imshow(
            image,
            origin="lower",
            extent=extent,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            interpolation="nearest",
        )
        _decorate_axis(ax, bounds=bounds, cube1_x=cube1_x, tx_pos=tx_pos)
        ax.set_title(title, fontsize=10)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)

    fig.suptitle(
        "Pure Witwin Three-Cube Radiomap Forward Benchmark",
        fontsize=13,
    )
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def save_arrays(forward_benchmark, *, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        witwin_total=forward_benchmark["witwin"]["total"],
        witwin_total_no_diff=forward_benchmark["witwin"]["total_no_diff"],
        witwin_diff_component=forward_benchmark["witwin"]["diff_component"],
        witwin_diff_increment=forward_benchmark["witwin"]["diff_increment"],
        witwin_total_db=forward_benchmark["witwin"]["total_db"],
        witwin_total_no_diff_db=forward_benchmark["witwin"]["total_no_diff_db"],
        witwin_diff_db=forward_benchmark["witwin"]["diff_db"],
        witwin_diff_increment_db=forward_benchmark["witwin"]["diff_increment_db"],
        witwin_diff_ratio_db=forward_benchmark["witwin"]["diff_ratio_db"],
        grid_x=forward_benchmark["witwin"]["grid_x"],
        grid_y=forward_benchmark["witwin"]["grid_y"],
    )
    return output_path


def save_json(forward_benchmark, *, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary = asdict(forward_benchmark["summary"])
    summary["paths"] = {
        "witwin_total_shape": list(forward_benchmark["witwin"]["total"].shape),
        "witwin_total_no_diff_shape": list(forward_benchmark["witwin"]["total_no_diff"].shape),
    }
    output_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return output_path


DEFAULT_OUTPUT_PREFIX = _output_dir() / "radiomap_three_cubes_witwin_forward_matched_isb_completion"


def save_radiomap_forward_three_cubes(
    output_prefix: Path,
    *,
    grid_size: int = DEFAULT_GRID_SIZE,
    n_rays: int = DEFAULT_N_RAYS,
    db_min: float = DEFAULT_DB_MIN,
    db_max: float = DEFAULT_DB_MAX,
    plane_z: float = DEFAULT_PLANE_Z,
    tx_pos: tuple[float, float, float] = DEFAULT_TX_POS,
    bounds=DEFAULT_BOUNDS,
    witwin_profile: str = DEFAULT_WITWIN_PROFILE,
    witwin_combine_mode: str = DEFAULT_WITWIN_COMBINE_MODE,
    witwin_receiver_model: str = DEFAULT_WITWIN_RECEIVER_MODEL,
    witwin_shadow_boundary_mode: str = DEFAULT_WITWIN_SHADOW_BOUNDARY_MODE,
    witwin_shadow_support_cutoff_db: float | None = DEFAULT_WITWIN_SHADOW_SUPPORT_CUTOFF_DB,
) -> tuple[Path, Path, Path]:
    forward_benchmark = build_forward_benchmark(
        grid_size=int(grid_size),
        bounds=bounds,
        plane_z=float(plane_z),
        tx_pos=tuple(float(value) for value in tx_pos),
        n_rays=int(n_rays),
        db_min=float(db_min),
        db_max=float(db_max),
        diff_ratio_range_db=DEFAULT_DIFF_RATIO_RANGE_DB,
        witwin_profile=str(witwin_profile),
        witwin_combine_mode=str(witwin_combine_mode),
        witwin_receiver_model=str(witwin_receiver_model),
        witwin_shadow_boundary_mode=str(witwin_shadow_boundary_mode),
        witwin_shadow_support_cutoff_db=witwin_shadow_support_cutoff_db,
    )
    figure_path = save_figure(forward_benchmark, output_path=output_prefix.with_suffix(".png"))
    arrays_path = save_arrays(forward_benchmark, output_path=output_prefix.with_suffix(".npz"))
    json_path = save_json(forward_benchmark, output_path=output_prefix.with_suffix(".json"))
    return figure_path, arrays_path, json_path


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid-size", type=int, default=DEFAULT_GRID_SIZE)
    parser.add_argument("--n-rays", type=int, default=DEFAULT_N_RAYS)
    parser.add_argument("--db-min", type=float, default=DEFAULT_DB_MIN)
    parser.add_argument("--db-max", type=float, default=DEFAULT_DB_MAX)
    parser.add_argument("--plane-z", type=float, default=DEFAULT_PLANE_Z)
    parser.add_argument("--tx-x", type=float, default=float(DEFAULT_TX_POS[0]))
    parser.add_argument("--tx-y", type=float, default=float(DEFAULT_TX_POS[1]))
    parser.add_argument("--tx-z", type=float, default=float(DEFAULT_TX_POS[2]))
    parser.add_argument(
        "--witwin-profile",
        choices=("matched_isb_completion", "legacy", "projected_coherent", "smooth", "custom"),
        default=DEFAULT_WITWIN_PROFILE,
    )
    parser.add_argument("--witwin-combine-mode", type=str, default=DEFAULT_WITWIN_COMBINE_MODE)
    parser.add_argument("--witwin-receiver-model", type=str, default=DEFAULT_WITWIN_RECEIVER_MODEL)
    parser.add_argument(
        "--witwin-shadow-boundary-mode",
        type=str,
        default=DEFAULT_WITWIN_SHADOW_BOUNDARY_MODE,
    )
    parser.add_argument(
        "--witwin-shadow-support-cutoff-db",
        type=float,
        default=DEFAULT_WITWIN_SHADOW_SUPPORT_CUTOFF_DB,
    )
    parser.add_argument("--xmin", type=float, default=float(DEFAULT_BOUNDS[0][0]))
    parser.add_argument("--xmax", type=float, default=float(DEFAULT_BOUNDS[0][1]))
    parser.add_argument("--ymin", type=float, default=float(DEFAULT_BOUNDS[1][0]))
    parser.add_argument("--ymax", type=float, default=float(DEFAULT_BOUNDS[1][1]))
    parser.add_argument("--output-prefix", type=Path, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    bounds = (
        (float(args.xmin), float(args.xmax)),
        (float(args.ymin), float(args.ymax)),
    )
    tx_pos = (float(args.tx_x), float(args.tx_y), float(args.tx_z))
    output_prefix = args.output_prefix if args.output_prefix is not None else DEFAULT_OUTPUT_PREFIX
    figure_path, arrays_path, json_path = save_radiomap_forward_three_cubes(
        output_prefix,
        grid_size=int(args.grid_size),
        n_rays=int(args.n_rays),
        db_min=float(args.db_min),
        db_max=float(args.db_max),
        plane_z=float(args.plane_z),
        tx_pos=tx_pos,
        bounds=bounds,
        witwin_profile=str(args.witwin_profile),
        witwin_combine_mode=str(args.witwin_combine_mode),
        witwin_receiver_model=str(args.witwin_receiver_model),
        witwin_shadow_boundary_mode=str(args.witwin_shadow_boundary_mode),
        witwin_shadow_support_cutoff_db=args.witwin_shadow_support_cutoff_db,
    )
    print(
        json.dumps(
            {
                "figure": str(figure_path),
                "arrays": str(arrays_path),
                "json": str(json_path),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()


__all__ = [
    "DEFAULT_OUTPUT_PREFIX",
    "build_forward_benchmark",
    "save_arrays",
    "save_figure",
    "save_json",
    "save_radiomap_forward_three_cubes",
]
