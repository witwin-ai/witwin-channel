"""Profile the per-call breakdown when caches are warm."""

from __future__ import annotations

import sys
import time
from pathlib import Path
from collections import defaultdict

import drjit as dr
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
CORE_ROOT = ROOT.parent / "core"
for root in (CORE_ROOT, ROOT):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

import witwin as wt

from tests._scene_helpers import box_geometry, build_scene
from tests.support.bin._path_monitor_phase6 import (
    format_path_monitor_diffraction_depth_report,
    resolve_path_monitor_diffraction_depth_report,
)
from witwin.channel import PathMonitor, Tracer
from witwin.channel.monitors.path.trace import (
    trace_path_monitor, _receiver_groups, _gather_positions, _remap_raw_rx_index,
)
from witwin.channel.trace.tracer import resolve_solver_controls
from witwin.channel.trace.diffraction.builders import _prepare_diffraction_state_arrays
from witwin.channel.monitors.path.collectors import (
    collect_los_paths, collect_reflection_paths, collect_diffraction_state_paths,
)
from witwin.channel.monitors.path.result import PathResult
from witwin.channel.utils import scalar


def _sync():
    dr.sync_thread()
    torch.cuda.synchronize()


def _build_scene():
    return build_scene(
        box_geometry(center=(-3.0, -3.0, 1.5), size=2.0),
        box_geometry(center=(3.0, -3.0, 1.5), size=2.0),
        box_geometry(center=(-3.0, 3.0, 1.5), size=2.0),
        box_geometry(center=(3.0, 3.0, 1.5), size=2.0),
    )


def main():
    scene = _build_scene()
    n_rx = 5
    n_frames = 30

    tx_bk = wt.Point3f(0.0, -6.0, 1.5)

    base_rx = np.array([
        [1.0, 2.0, 1.5], [-1.0, 2.0, 1.5], [0.0, 3.0, 1.5],
        [1.5, 3.5, 1.5], [-1.5, 3.5, 1.5],
    ])[:n_rx]

    rng = np.random.RandomState(42)
    rx_per_frame = []
    current_rx = base_rx.copy()
    for _ in range(n_frames):
        current_rx = current_rx + rng.randn(n_rx, 3) * 0.05
        current_rx[:, 2] = 1.5
        rx_per_frame.append(current_rx.copy())

    tracer = Tracer(
        frequency=1e9, scene=scene,
        reflection_n_rays=1000, reflection_max_bounces=3, max_diffractions=2,
    )
    monitor = PathMonitor(
        "rx",
        positions=torch.tensor(base_rx, dtype=torch.float32),
        max_diffractions=tracer.config.trace.max_diffractions,
    )
    depth_report = resolve_path_monitor_diffraction_depth_report(
        tracer.config.trace,
        requested_max_diffractions=monitor.max_diffractions,
    )
    solver_controls = resolve_solver_controls(
        tracer.config.trace,
        execution_intent="path_export",
        max_diffractions_override=monitor.max_diffractions,
    )
    config = tracer._resolved_trace_config
    effective = solver_controls["effective"]

    # Build caches
    rx0_bk = wt.Point3f(wt.Float(base_rx[:, 0].tolist()), wt.Float(base_rx[:, 1].tolist()), wt.Float(base_rx[:, 2].tolist()))
    _, ref_detail = trace_path_monitor(tx_bk, monitor, scene, config, solver_controls, verbose=False)
    _sync()

    groups = _receiver_groups(rx0_bk)
    mixed_ref = ref_detail if config.enable_rd_diffraction else None
    ref_n_rays = effective["reflection_n_rays"] if config.enable_rd_diffraction else 0
    ref_max = effective["reflection_max_bounces"] if config.enable_rd_diffraction else 0

    diff_groups = []
    for receiver_z, group_indices in groups:
        edge_cache, edge_data, state_arrays, _ = _prepare_diffraction_state_arrays(
            tx_bk, receiver_z, scene, config.wavelength, config.k,
            mixed_ref, config.diffraction_material, ref_n_rays, ref_max,
            config.reflection_coef, "exhaustive", effective["max_diffractions"],
            total_state_budget_per_order=effective["diffraction_state_budget"],
            inserted_state_budget_per_order=effective["inserted_reflection_state_budget"],
            max_inserted_reflections_per_path=effective["max_inserted_reflections_per_path"],
            use_scene_materials=config.use_scene_materials_for_diffraction,
            tx_polarization=config.tx_polarization,
        )
        _sync()
        diff_groups.append((receiver_z, group_indices, edge_cache, edge_data, state_arrays))

    # Warmup cached path
    for _ in range(3):
        rx_bk = wt.Point3f(wt.Float(base_rx[:, 0].tolist()), wt.Float(base_rx[:, 1].tolist()), wt.Float(base_rx[:, 2].tolist()))
        collect_los_paths(scene=scene, rx_positions=rx_bk, tx_pos=tx_bk,
                          wavelength=config.wavelength, k=config.k,
                          tx_polarization=config.tx_polarization, rx_polarization=config.rx_polarization)
        collect_reflection_paths(scene=scene, rx_positions=rx_bk, tx_pos=tx_bk,
                                 wavelength=config.wavelength, k=config.k,
                                 n_rays=effective["reflection_n_rays"],
                                 max_reflections=effective["reflection_max_bounces"],
                                 mode="exhaustive", tx_polarization=config.tx_polarization,
                                 rx_polarization=config.rx_polarization,
                                 reflection_coef=config.reflection_coef,
                                 min_ray_contribution_threshold=config.min_ray_contribution_threshold,
                                 reflection_relative_permittivity=config.reflection_relative_permittivity,
                                 reflection_conductivity=config.reflection_conductivity,
                                 reflection_material=config.reflection_material,
                                 use_scene_materials=config.use_scene_materials_for_reflection,
                                 return_geometry=False, reflection_detail=ref_detail)
        for receiver_z, group_indices, edge_cache, edge_data, state_arrays in diff_groups:
            group_rx = _gather_positions(rx_bk, group_indices)
            collect_diffraction_state_paths(
                state_arrays=state_arrays,
                edge_data=edge_data if edge_data is not None else edge_cache.get("edge_data"),
                scene=scene, rx_positions=group_rx, tx_pos=tx_bk,
                wavelength=config.wavelength, k=config.k,
                tx_polarization=config.tx_polarization, rx_polarization=config.rx_polarization,
                material_detail=config.diffraction_material, return_geometry=False,
            )
        _sync()

    # Profile each sub-stage
    stage_times = defaultdict(list)
    total_times = []

    for frame_idx in range(n_frames):
        rx_np = rx_per_frame[frame_idx]
        rx_bk = wt.Point3f(wt.Float(rx_np[:, 0].tolist()), wt.Float(rx_np[:, 1].tolist()), wt.Float(rx_np[:, 2].tolist()))

        overall_t0 = time.perf_counter()

        _sync()
        t0 = time.perf_counter()
        los_raw = collect_los_paths(
            scene=scene, rx_positions=rx_bk, tx_pos=tx_bk,
            wavelength=config.wavelength, k=config.k,
            tx_polarization=config.tx_polarization, rx_polarization=config.rx_polarization,
        )
        _sync()
        stage_times["los"].append((time.perf_counter() - t0) * 1000.0)

        _sync()
        t0 = time.perf_counter()
        refl_raw, _ = collect_reflection_paths(
            scene=scene, rx_positions=rx_bk, tx_pos=tx_bk,
            wavelength=config.wavelength, k=config.k,
            n_rays=effective["reflection_n_rays"],
            max_reflections=effective["reflection_max_bounces"],
            mode="exhaustive", tx_polarization=config.tx_polarization,
            rx_polarization=config.rx_polarization,
            reflection_coef=config.reflection_coef,
            min_ray_contribution_threshold=config.min_ray_contribution_threshold,
            reflection_relative_permittivity=config.reflection_relative_permittivity,
            reflection_conductivity=config.reflection_conductivity,
            reflection_material=config.reflection_material,
            use_scene_materials=config.use_scene_materials_for_reflection,
            return_geometry=False, reflection_detail=ref_detail,
        )
        _sync()
        stage_times["refl_replay"].append((time.perf_counter() - t0) * 1000.0)

        diff_raws = []
        _sync()
        t0 = time.perf_counter()
        for receiver_z, group_indices, edge_cache, edge_data, state_arrays in diff_groups:
            group_rx = _gather_positions(rx_bk, group_indices)
            raw = collect_diffraction_state_paths(
                state_arrays=state_arrays,
                edge_data=edge_data if edge_data is not None else edge_cache.get("edge_data"),
                scene=scene, rx_positions=group_rx, tx_pos=tx_bk,
                wavelength=config.wavelength, k=config.k,
                tx_polarization=config.tx_polarization, rx_polarization=config.rx_polarization,
                material_detail=config.diffraction_material, return_geometry=False,
            )
            _remap_raw_rx_index(raw, group_indices)
            diff_raws.append(raw)
        _sync()
        stage_times["diff_eval"].append((time.perf_counter() - t0) * 1000.0)

        _sync()
        t0 = time.perf_counter()
        pr = PathResult.from_raw_collections(
            name="rx", num_rx=n_rx, max_num_paths=None,
            tx_pos=(scalar(tx_bk.x), scalar(tx_bk.y), scalar(tx_bk.z)),
            rx_positions=rx_bk, frequency=float(config.frequency),
            wavelength=float(config.wavelength),
            raw_collections=[los_raw, refl_raw, *diff_raws],
            return_geometry=False, metadata={},
        )
        _sync()
        stage_times["assembly"].append((time.perf_counter() - t0) * 1000.0)

        total_times.append((time.perf_counter() - overall_t0) * 1000.0)

    print("=" * 70)
    print(f"Per-call breakdown (cached, 1 TX, {n_rx} RX, {n_frames} frames)")
    print(f"Path diffraction depth: {format_path_monitor_diffraction_depth_report(depth_report)}")
    print("=" * 70)
    avg_total = np.mean(total_times)
    print(f"Total per call: {avg_total:.1f} ms  (std: {np.std(total_times):.1f})")
    accounted = 0.0
    for key in ["los", "refl_replay", "diff_eval", "assembly"]:
        vals = stage_times[key]
        avg = np.mean(vals)
        pct = avg / avg_total * 100
        accounted += avg
        print(f"  {key:>15s}: {avg:6.1f} ms  ({pct:5.1f}%)")
    remainder = avg_total - accounted
    if abs(remainder) > 0.05:
        print(f"  {'overhead':>15s}: {remainder:6.1f} ms  ({remainder/avg_total*100:5.1f}%)")
    print("=" * 70)


if __name__ == "__main__":
    main()
