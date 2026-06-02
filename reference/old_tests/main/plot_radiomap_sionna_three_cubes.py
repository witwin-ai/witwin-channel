"""Compare three-cube radio maps between witwin matched-ISB completion and Sionna RT."""

from __future__ import annotations

import argparse
import importlib
import json
from dataclasses import asdict, dataclass
from functools import lru_cache
import os
from pathlib import Path
import time
if os.environ.get("WITWIN_CHANNEL_MAIN_SHOW", "0") != "1":
    os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

import witwin as wt
from tests._scene_helpers import box_drjit_geometry, build_scene as build_test_scene
from tests.main.plot_multipath_components import (
    CUBE1_BASE_CENTER,
    CUBE2_CENTER,
    CUBE3_CENTER,
    CUBE_SIZE,
    MULTIPATH_SCENE_MATERIAL,
    TX_POS,
    cube_specs,
)
from witwin.channel import RadioMapMonitor, Tracer, scene_to_sionna_scene
DEFAULT_BOUNDS = ((-10.0, 10.0), (-10.0, 10.0))
DEFAULT_GRID_SIZE = 256
DEFAULT_PLANE_Z = 1.0
DEFAULT_TX_POS = (TX_POS[0], TX_POS[1], 4.0)
DEFAULT_N_RAYS = 4096
DEFAULT_SAMPLES_PER_TX = 1_000_000
DEFAULT_DB_MIN = -90.0
DEFAULT_DB_MAX = -40.0
DEFAULT_DIFF_RATIO_RANGE_DB = 30.0
DEFAULT_WITWIN_PROFILE = "matched_isb_completion"
DEFAULT_WITWIN_COMBINE_MODE = "coherent"
DEFAULT_WITWIN_RECEIVER_MODEL = "matched_isotropic"
DEFAULT_WITWIN_SHADOW_BOUNDARY_MODE = "matched_isb_completion"
DEFAULT_WITWIN_SHADOW_SUPPORT_CUTOFF_DB = 25.0
DEFAULT_WITWIN_EDGE_SELECTION_MODE = "all_edges"
DEFAULT_SIONNA_EDGE_DIFFRACTION = True

mi = importlib.import_module("mitsuba")

WITWIN_PROFILES = {
    "matched_isb_completion": {
        "combine_mode": "coherent",
        "receiver_model": "matched_isotropic",
        "shadow_boundary_mode": "matched_isb_completion",
        "label": "Matched ISB completion",
    },
    "legacy": {
        "combine_mode": "incoherent",
        "receiver_model": "matched_isotropic",
        "shadow_boundary_mode": "none",
        "label": "Legacy",
    },
    "projected_coherent": {
        "combine_mode": "coherent",
        "receiver_model": "projected_polarized",
        "shadow_boundary_mode": "none",
        "label": "Projected coherent",
    },
    "smooth": {
        "combine_mode": "coherent",
        "receiver_model": "projected_polarized",
        "shadow_boundary_mode": "projected_isb_completion",
        "label": "Projected ISB completion",
    },
    "custom": {
        "combine_mode": None,
        "receiver_model": None,
        "shadow_boundary_mode": None,
        "label": "Custom",
    },
}


@dataclass(frozen=True)
class ComparisonSummary:
    grid_size: int
    bounds: tuple[tuple[float, float], tuple[float, float]]
    plane_z: float
    tx_pos: tuple[float, float, float]
    n_rays: int
    samples_per_tx: int
    db_min: float
    db_max: float
    witwin_no_diff_seconds: float
    witwin_with_diff_seconds: float
    sionna_no_diff_seconds: float
    sionna_with_diff_seconds: float
    witwin_profile: str
    witwin_profile_label: str
    witwin_combine_mode: str
    witwin_receiver_model: str
    witwin_shadow_boundary_mode: str
    witwin_shadow_support_cutoff_db: float | None
    witwin_edge_selection_mode: str
    witwin_metric_contract: str
    witwin_path_counts: dict
    witwin_diff_component_increment_db_corr: float
    witwin_sionna_total_db_corr: float
    witwin_sionna_diff_db_corr: float
    witwin_diff_ratio_p95: float
    sionna_diff_ratio_p95: float
    sionna_edge_diffraction: bool
    witwin_runtime_backends: dict
    sionna_source: str


def _output_dir() -> Path:
    return Path(__file__).resolve().parent.parent / "output"


def _to_float_grid(value) -> np.ndarray:
    return np.asarray(value, dtype=np.float32)


def _absolute_db_map(values, *, floor_db: float) -> np.ndarray:
    values_np = _to_float_grid(values)
    floor = 10.0 ** (float(floor_db) / 10.0)
    return 10.0 * np.log10(np.maximum(values_np, floor))


def _ratio_db_map(numerator, denominator, *, dynamic_range_db: float) -> np.ndarray:
    numerator_np = _to_float_grid(numerator)
    denominator_np = _to_float_grid(denominator)
    floor_ratio = 10.0 ** (-float(dynamic_range_db) / 10.0)
    ratio = numerator_np / np.maximum(denominator_np, 1.0e-20)
    return 10.0 * np.log10(np.maximum(ratio, floor_ratio))


def _correlation(a, b) -> float:
    a_np = _to_float_grid(a).ravel()
    b_np = _to_float_grid(b).ravel()
    if a_np.size == 0 or b_np.size == 0:
        return float("nan")
    a_std = float(np.std(a_np))
    b_std = float(np.std(b_np))
    if a_std <= 0.0 or b_std <= 0.0:
        return float("nan")
    return float(np.corrcoef(a_np, b_np)[0, 1])


def _decorate_axis(ax, *, bounds, cube1_x: float, tx_pos):
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_xlim(float(bounds[0][0]), float(bounds[0][1]))
    ax.set_ylim(float(bounds[1][0]), float(bounds[1][1]))
    for spec in cube_specs(cube1_x):
        cx, cy = spec["center"]
        size = spec["size"]
        ax.add_patch(
            Rectangle(
                (cx - 0.5 * size, cy - 0.5 * size),
                size,
                size,
                facecolor="none",
                edgecolor="white",
                linewidth=0.9,
                alpha=0.8,
            )
        )
    ax.scatter(
        [float(tx_pos[0])],
        [float(tx_pos[1])],
        marker="*",
        s=70,
        c="white",
        edgecolors="black",
        linewidths=0.6,
    )


def _extent(bounds) -> tuple[float, float, float, float]:
    return (
        float(bounds[0][0]),
        float(bounds[0][1]),
        float(bounds[1][0]),
        float(bounds[1][1]),
    )


@lru_cache(maxsize=8)
def _build_comparison_scene(
    cube1_x: float,
    *,
    edge_selection_mode: str,
):
    cube1_center = wt.Point3f(cube1_x, CUBE1_BASE_CENTER[1], CUBE1_BASE_CENTER[2])
    cube1 = box_drjit_geometry(center=cube1_center, size=CUBE_SIZE, rotation=None).to_mesh()
    cube2 = box_drjit_geometry(center=CUBE2_CENTER, size=CUBE_SIZE, rotation=None).to_mesh()
    cube3 = box_drjit_geometry(center=CUBE3_CENTER, size=CUBE_SIZE, rotation=None).to_mesh()
    return build_test_scene(
        cube1,
        cube2,
        cube3,
        material=MULTIPATH_SCENE_MATERIAL,
        edge_selection_mode=str(edge_selection_mode),
    )


def _run_witwin(
    *,
    scene,
    tx_pos,
    plane_z: float,
    bounds,
    grid_size: int,
    n_rays: int,
    max_diffractions: int,
    combine_mode: str,
    receiver_model: str,
    shadow_boundary_mode: str,
    shadow_support_cutoff_db: float | None,
):
    monitor = RadioMapMonitor(
        "compare_rm",
        axis="z",
        position=float(plane_z),
        bounds=bounds,
        grid_shape=(int(grid_size), int(grid_size)),
        metric="path_gain",
        combine_mode=str(combine_mode),
        quadrature_mode="center",
        receiver_model=str(receiver_model),
        ray_mode="3d",
        max_diffractions=int(max_diffractions),
        shadow_boundary_mode=str(shadow_boundary_mode),
        shadow_support_cutoff_db=shadow_support_cutoff_db,
    )
    tracer = Tracer(
        frequency=1.0e9,
        scene=scene,
        reflection_n_rays=int(n_rays),
        reflection_max_bounces=3,
        reflection_coef=0.8,
        enable_rd_diffraction=True,
        max_diffractions=int(max_diffractions),
    )
    t0 = time.perf_counter()
    trace_output = tracer.trace(wt.Point3f(*tx_pos), monitor=monitor, verbose=False)
    result = trace_output.monitor(monitor.name) if hasattr(trace_output, "monitor") else trace_output
    elapsed = time.perf_counter() - t0
    return result, float(elapsed)


def _prepare_sionna_scene(*, scene, tx_pos):
    conversion = scene_to_sionna_scene(scene, prefer_local=True)
    rt = conversion.rt
    sionna_scene = conversion.scene
    sionna_scene.frequency = 1.0e9
    sionna_scene.tx_array = rt.PlanarArray(
        num_rows=1,
        num_cols=1,
        pattern="iso",
        polarization="V",
    )
    sionna_scene.add(rt.Transmitter("tx", position=mi.Point3f(*tx_pos), power_dbm=0.0))
    return conversion, rt, sionna_scene


def _normalize_witwin_contract(
    *,
    combine_mode: str,
    receiver_model: str,
    shadow_boundary_mode: str,
) -> tuple[str, str, str]:
    resolved_combine_mode = str(combine_mode).lower()
    resolved_receiver_model = str(receiver_model).lower()
    resolved_shadow_boundary_mode = str(shadow_boundary_mode).lower()
    if resolved_combine_mode not in {"incoherent", "coherent"}:
        raise ValueError("witwin_combine_mode must be 'incoherent' or 'coherent'.")
    if resolved_receiver_model not in {"matched_isotropic", "projected_polarized"}:
        raise ValueError(
            "witwin_receiver_model must be 'matched_isotropic' or 'projected_polarized'."
        )
    if resolved_shadow_boundary_mode not in {
        "none",
        "utd_cross_term_surrogate",
        "projected_isb_completion",
        "matched_isb_completion",
    }:
        raise ValueError(
            "witwin_shadow_boundary_mode must be 'none', "
            "'utd_cross_term_surrogate', 'projected_isb_completion', "
            "or 'matched_isb_completion'."
        )
    return (
        resolved_combine_mode,
        resolved_receiver_model,
        resolved_shadow_boundary_mode,
    )


def _resolve_witwin_profile(
    *,
    profile: str,
    combine_mode: str,
    receiver_model: str,
    shadow_boundary_mode: str,
) -> tuple[str, str, str, str, str]:
    resolved_profile = str(profile).lower()
    if resolved_profile not in WITWIN_PROFILES:
        raise ValueError(
            "witwin_profile must be one of "
            f"{', '.join(sorted(WITWIN_PROFILES.keys()))}."
        )
    profile_spec = WITWIN_PROFILES[resolved_profile]
    if resolved_profile == "custom":
        (
            resolved_combine_mode,
            resolved_receiver_model,
            resolved_shadow_boundary_mode,
        ) = _normalize_witwin_contract(
            combine_mode=combine_mode,
            receiver_model=receiver_model,
            shadow_boundary_mode=shadow_boundary_mode,
        )
    else:
        resolved_combine_mode = str(profile_spec["combine_mode"])
        resolved_receiver_model = str(profile_spec["receiver_model"])
        resolved_shadow_boundary_mode = str(profile_spec["shadow_boundary_mode"])
    return (
        resolved_profile,
        str(profile_spec["label"]),
        resolved_combine_mode,
        resolved_receiver_model,
        resolved_shadow_boundary_mode,
    )


def _witwin_diff_component_map(payload) -> np.ndarray:
    combine_mode = str(payload.combine_mode)
    if combine_mode == "coherent":
        return _to_float_grid(payload.coherent_power["diffraction"])
    return _to_float_grid(payload.incoherent["diffraction"])


def _witwin_diff_component_label(payload) -> str:
    combine_mode = str(payload.combine_mode)
    if combine_mode == "coherent":
        return "Diffraction Coherent Power"
    return "Diffraction Power"


def _run_sionna(
    *,
    rt,
    scene,
    plane_z: float,
    bounds,
    grid_size: int,
    samples_per_tx: int,
    diffraction: bool,
    edge_diffraction: bool,
    specular_reflection: bool = True,
    max_depth: int = 3,
):
    solver = rt.RadioMapSolver()
    span_x = float(bounds[0][1] - bounds[0][0])
    span_y = float(bounds[1][1] - bounds[1][0])
    t0 = time.perf_counter()
    result = solver(
        scene,
        center=mi.Point3f(
            0.5 * (float(bounds[0][0]) + float(bounds[0][1])),
            0.5 * (float(bounds[1][0]) + float(bounds[1][1])),
            float(plane_z),
        ),
        orientation=mi.Point3f(0.0, 0.0, 0.0),
        size=mi.Point2f(span_x, span_y),
        cell_size=mi.Point2f(span_x / float(grid_size), span_y / float(grid_size)),
        samples_per_tx=int(samples_per_tx),
        max_depth=int(max_depth),
        los=True,
        specular_reflection=bool(specular_reflection),
        diffraction=bool(diffraction),
        edge_diffraction=bool(edge_diffraction),
        refraction=False,
        seed=7,
    )
    elapsed = time.perf_counter() - t0
    return result, float(elapsed)


def build_comparison(
    *,
    grid_size: int,
    bounds,
    plane_z: float,
    tx_pos,
    n_rays: int,
    samples_per_tx: int,
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

    conversion, rt, sionna_scene = _prepare_sionna_scene(scene=scene, tx_pos=tx_pos)
    sionna_no_diff, sionna_no_diff_seconds = _run_sionna(
        rt=rt,
        scene=sionna_scene,
        plane_z=plane_z,
        bounds=bounds,
        grid_size=grid_size,
        samples_per_tx=samples_per_tx,
        diffraction=False,
        edge_diffraction=DEFAULT_SIONNA_EDGE_DIFFRACTION,
    )
    sionna_with_diff, sionna_with_diff_seconds = _run_sionna(
        rt=rt,
        scene=sionna_scene,
        plane_z=plane_z,
        bounds=bounds,
        grid_size=grid_size,
        samples_per_tx=samples_per_tx,
        diffraction=True,
        edge_diffraction=DEFAULT_SIONNA_EDGE_DIFFRACTION,
    )

    witwin_total = _to_float_grid(witwin_with_diff.path_gain)
    witwin_total_no_diff = _to_float_grid(witwin_no_diff.path_gain)
    witwin_diff_component = _witwin_diff_component_map(witwin_with_diff)
    witwin_diff_increment = np.maximum(witwin_total - witwin_total_no_diff, 0.0)

    sionna_total = _to_float_grid(np.asarray(sionna_with_diff.path_gain)[0])
    sionna_total_no_diff = _to_float_grid(np.asarray(sionna_no_diff.path_gain)[0])
    sionna_diff_increment = np.maximum(sionna_total - sionna_total_no_diff, 0.0)

    witwin_total_db = _absolute_db_map(witwin_total, floor_db=db_min)
    sionna_total_db = _absolute_db_map(sionna_total, floor_db=db_min)
    witwin_diff_db = _absolute_db_map(witwin_diff_component, floor_db=db_min)
    sionna_diff_db = _absolute_db_map(sionna_diff_increment, floor_db=db_min)
    witwin_diff_ratio_db = _ratio_db_map(
        witwin_diff_component,
        witwin_total,
        dynamic_range_db=diff_ratio_range_db,
    )
    sionna_diff_ratio_db = _ratio_db_map(
        sionna_diff_increment,
        sionna_total,
        dynamic_range_db=diff_ratio_range_db,
    )
    total_delta_db = witwin_total_db - sionna_total_db
    diff_delta_db = witwin_diff_db - sionna_diff_db

    summary = ComparisonSummary(
        grid_size=int(grid_size),
        bounds=tuple((float(a), float(b)) for (a, b) in bounds),
        plane_z=float(plane_z),
        tx_pos=tuple(float(value) for value in tx_pos),
        n_rays=int(n_rays),
        samples_per_tx=int(samples_per_tx),
        db_min=float(db_min),
        db_max=float(db_max),
        witwin_no_diff_seconds=float(witwin_no_diff_seconds),
        witwin_with_diff_seconds=float(witwin_with_diff_seconds),
        sionna_no_diff_seconds=float(sionna_no_diff_seconds),
        sionna_with_diff_seconds=float(sionna_with_diff_seconds),
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
        witwin_diff_component_increment_db_corr=_correlation(
            _absolute_db_map(witwin_diff_component, floor_db=db_min),
            _absolute_db_map(witwin_diff_increment, floor_db=db_min),
        ),
        witwin_sionna_total_db_corr=_correlation(witwin_total_db, sionna_total_db),
        witwin_sionna_diff_db_corr=_correlation(witwin_diff_db, sionna_diff_db),
        witwin_diff_ratio_p95=float(
            np.percentile(witwin_diff_component / np.maximum(witwin_total, 1.0e-20), 95.0)
        ),
        sionna_diff_ratio_p95=float(
            np.percentile(sionna_diff_increment / np.maximum(sionna_total, 1.0e-20), 95.0)
        ),
        sionna_edge_diffraction=bool(DEFAULT_SIONNA_EDGE_DIFFRACTION),
        witwin_runtime_backends=dict(witwin_with_diff.metadata.get("runtime_backends", {})),
        sionna_source=str(conversion.source),
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
            "diff_db": witwin_diff_db,
            "diff_ratio_db": witwin_diff_ratio_db,
            "grid_x": _to_float_grid(witwin_with_diff.coords.grid_x),
            "grid_y": _to_float_grid(witwin_with_diff.coords.grid_y),
        },
        "sionna": {
            "total": sionna_total,
            "total_no_diff": sionna_total_no_diff,
            "diff_increment": sionna_diff_increment,
            "total_db": sionna_total_db,
            "diff_db": sionna_diff_db,
            "diff_ratio_db": sionna_diff_ratio_db,
        },
        "delta": {
            "total_db": total_delta_db,
            "diff_db": diff_delta_db,
        },
    }


def save_figure(comparison, *, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 4, figsize=(18.0, 9.0), constrained_layout=True)
    extent = comparison["extent"]
    bounds = comparison["summary"].bounds
    tx_pos = comparison["summary"].tx_pos
    cube1_x = comparison["cube1_x"]
    witwin_with_diff_seconds = comparison["summary"].witwin_with_diff_seconds
    sionna_with_diff_seconds = comparison["summary"].sionna_with_diff_seconds
    witwin_no_diff_seconds = comparison["summary"].witwin_no_diff_seconds
    sionna_no_diff_seconds = comparison["summary"].sionna_no_diff_seconds
    witwin_diff_component_label = str(comparison["witwin"]["diff_component_label"])
    dynamic_vmax = max(
        6.0,
        float(np.percentile(np.abs(comparison["delta"]["total_db"]), 99.0)),
        float(np.percentile(np.abs(comparison["delta"]["diff_db"]), 99.0)),
    )
    db_min = float(comparison["summary"].db_min)
    db_max = float(comparison["summary"].db_max)

    panels = (
        (axes[0, 0], comparison["witwin"]["total_db"], f"Witwin Total (dB, {witwin_with_diff_seconds:.3f}s)", "viridis", db_min, db_max),
        (axes[0, 1], comparison["sionna"]["total_db"], f"Sionna Total (dB, {sionna_with_diff_seconds:.3f}s)", "viridis", db_min, db_max),
        (axes[0, 2], comparison["delta"]["total_db"], f"Total Delta (dB, Witwin {witwin_with_diff_seconds:.3f}s - Sionna {sionna_with_diff_seconds:.3f}s)", "coolwarm", -dynamic_vmax, dynamic_vmax),
        (axes[0, 3], comparison["sionna"]["diff_ratio_db"], f"Sionna Diff / Total (dB, {sionna_with_diff_seconds:.3f}s)", "cividis", -30.0, 0.0),
        (axes[1, 0], comparison["witwin"]["diff_db"], f"Witwin {witwin_diff_component_label} (dB, {witwin_with_diff_seconds:.3f}s)", "magma", db_min, db_max),
        (axes[1, 1], comparison["sionna"]["diff_db"], f"Sionna Diff Increment (dB, {sionna_with_diff_seconds:.3f}s)", "magma", db_min, db_max),
        (axes[1, 2], comparison["delta"]["diff_db"], f"Diff Delta (dB, Witwin {witwin_with_diff_seconds:.3f}s - Sionna {sionna_with_diff_seconds:.3f}s)", "coolwarm", -dynamic_vmax, dynamic_vmax),
        (axes[1, 3], comparison["witwin"]["diff_ratio_db"], f"Witwin Diff / Total (dB, {witwin_with_diff_seconds:.3f}s)", "cividis", -30.0, 0.0),
    )

    for ax, values, title, cmap, vmin, vmax in panels:
        image = ax.imshow(values, origin="lower", extent=extent, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(title, fontsize=10)
        _decorate_axis(ax, bounds=bounds, cube1_x=cube1_x, tx_pos=tx_pos)
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.03)

    fig.suptitle(
        "Three-cube off-plane radio map: witwin vs Sionna RT\n"
        f"tx=({tx_pos[0]:.1f}, {tx_pos[1]:.1f}, {tx_pos[2]:.1f}), plane z={comparison['summary'].plane_z:.1f}, "
        f"grid={comparison['summary'].grid_size}, samples_per_tx={comparison['summary'].samples_per_tx}, "
        f"dB=[{db_min:.0f}, {db_max:.0f}], "
        f"profile={comparison['summary'].witwin_profile_label}, "
        f"witwin={comparison['summary'].witwin_combine_mode}/{comparison['summary'].witwin_receiver_model}, "
        f"shadow={comparison['summary'].witwin_shadow_boundary_mode}, "
        f"shadow_cutoff_db={comparison['summary'].witwin_shadow_support_cutoff_db}, "
        f"wedges={comparison['summary'].witwin_edge_selection_mode}, finite_wedge, "
        f"sionna_edge_diffraction={comparison['summary'].sionna_edge_diffraction}, "
        f"no-diff times: witwin {witwin_no_diff_seconds:.3f}s / sionna {sionna_no_diff_seconds:.3f}s",
        fontsize=13,
    )
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def save_arrays(comparison, *, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        witwin_total=comparison["witwin"]["total"],
        witwin_total_no_diff=comparison["witwin"]["total_no_diff"],
        witwin_diff_component=comparison["witwin"]["diff_component"],
        witwin_diff_increment=comparison["witwin"]["diff_increment"],
        sionna_total=comparison["sionna"]["total"],
        sionna_total_no_diff=comparison["sionna"]["total_no_diff"],
        sionna_diff_increment=comparison["sionna"]["diff_increment"],
        witwin_total_db=comparison["witwin"]["total_db"],
        witwin_diff_db=comparison["witwin"]["diff_db"],
        sionna_total_db=comparison["sionna"]["total_db"],
        sionna_diff_db=comparison["sionna"]["diff_db"],
        total_delta_db=comparison["delta"]["total_db"],
        diff_delta_db=comparison["delta"]["diff_db"],
    )
    return output_path


def save_json(comparison, *, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary = asdict(comparison["summary"])
    summary["witwin_runtime_backends"] = comparison["summary"].witwin_runtime_backends
    summary["paths"] = {
        "witwin_total_shape": list(comparison["witwin"]["total"].shape),
        "sionna_total_shape": list(comparison["sionna"]["total"].shape),
    }
    output_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return output_path


DEFAULT_OUTPUT_PREFIX = _output_dir() / "radiomap_three_cubes_sionna_compare_matched_isb_completion"


def _save_comparison_outputs(comparison, *, output_prefix: Path) -> tuple[Path, Path, Path]:
    figure_path = save_figure(comparison, output_path=output_prefix.with_suffix(".png"))
    arrays_path = save_arrays(comparison, output_path=output_prefix.with_suffix(".npz"))
    json_path = save_json(comparison, output_path=output_prefix.with_suffix(".json"))
    return figure_path, arrays_path, json_path


def save_radiomap_sionna_three_cubes_comparison(
    output_prefix: Path,
    *,
    grid_size: int = DEFAULT_GRID_SIZE,
    n_rays: int = DEFAULT_N_RAYS,
    samples_per_tx: int = DEFAULT_SAMPLES_PER_TX,
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
    comparison = build_comparison(
        grid_size=int(grid_size),
        bounds=bounds,
        plane_z=float(plane_z),
        tx_pos=tuple(float(value) for value in tx_pos),
        n_rays=int(n_rays),
        samples_per_tx=int(samples_per_tx),
        db_min=float(db_min),
        db_max=float(db_max),
        diff_ratio_range_db=DEFAULT_DIFF_RATIO_RANGE_DB,
        witwin_profile=str(witwin_profile),
        witwin_combine_mode=str(witwin_combine_mode),
        witwin_receiver_model=str(witwin_receiver_model),
        witwin_shadow_boundary_mode=str(witwin_shadow_boundary_mode),
        witwin_shadow_support_cutoff_db=witwin_shadow_support_cutoff_db,
    )
    return _save_comparison_outputs(comparison, output_prefix=output_prefix)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid-size", type=int, default=DEFAULT_GRID_SIZE)
    parser.add_argument("--n-rays", type=int, default=DEFAULT_N_RAYS)
    parser.add_argument("--samples-per-tx", type=int, default=DEFAULT_SAMPLES_PER_TX)
    parser.add_argument("--db-min", type=float, default=DEFAULT_DB_MIN)
    parser.add_argument("--db-max", type=float, default=DEFAULT_DB_MAX)
    parser.add_argument(
        "--witwin-profile",
        type=str,
        default=DEFAULT_WITWIN_PROFILE,
        choices=tuple(WITWIN_PROFILES.keys()),
    )
    parser.add_argument("--plane-z", type=float, default=DEFAULT_PLANE_Z)
    parser.add_argument("--tx-x", type=float, default=DEFAULT_TX_POS[0])
    parser.add_argument("--tx-y", type=float, default=DEFAULT_TX_POS[1])
    parser.add_argument("--tx-z", type=float, default=DEFAULT_TX_POS[2])
    parser.add_argument("--xmin", type=float, default=DEFAULT_BOUNDS[0][0])
    parser.add_argument("--xmax", type=float, default=DEFAULT_BOUNDS[0][1])
    parser.add_argument("--ymin", type=float, default=DEFAULT_BOUNDS[1][0])
    parser.add_argument("--ymax", type=float, default=DEFAULT_BOUNDS[1][1])
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
    parser.add_argument("--output-prefix", type=Path, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    bounds = ((float(args.xmin), float(args.xmax)), (float(args.ymin), float(args.ymax)))
    tx_pos = (float(args.tx_x), float(args.tx_y), float(args.tx_z))
    output_prefix = args.output_prefix if args.output_prefix is not None else DEFAULT_OUTPUT_PREFIX

    comparison = build_comparison(
        grid_size=int(args.grid_size),
        bounds=bounds,
        plane_z=float(args.plane_z),
        tx_pos=tx_pos,
        n_rays=int(args.n_rays),
        samples_per_tx=int(args.samples_per_tx),
        db_min=float(args.db_min),
        db_max=float(args.db_max),
        diff_ratio_range_db=DEFAULT_DIFF_RATIO_RANGE_DB,
        witwin_profile=str(args.witwin_profile),
        witwin_combine_mode=str(args.witwin_combine_mode),
        witwin_receiver_model=str(args.witwin_receiver_model),
        witwin_shadow_boundary_mode=str(args.witwin_shadow_boundary_mode),
        witwin_shadow_support_cutoff_db=args.witwin_shadow_support_cutoff_db,
    )
    figure_path, arrays_path, json_path = _save_comparison_outputs(
        comparison,
        output_prefix=output_prefix,
    )
    print(json.dumps({
        "figure": str(figure_path),
        "arrays": str(arrays_path),
        "json": str(json_path),
        "summary": asdict(comparison["summary"]),
    }, indent=2))


if __name__ == "__main__":
    main()


__all__ = [
    "DEFAULT_OUTPUT_PREFIX",
    "build_comparison",
    "save_arrays",
    "save_figure",
    "save_json",
    "save_radiomap_sionna_three_cubes_comparison",
]
