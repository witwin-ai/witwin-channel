"""Profile PathMonitor for multi-TX, few-RX, many-frame scenario."""

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


def profile_multi_frame():
    scene = _build_scene()

    n_tx = 5
    n_rx = 5
    n_frames = 50

    tx_positions = [
        torch.tensor([x, y, 1.5], dtype=torch.float32)
        for x, y in [(-5, 0), (5, 0), (0, -5), (0, 5), (0, 0)]
    ][:n_tx]

    base_rx = np.array([
        [1.0, 2.0, 1.5],
        [-1.0, 2.0, 1.5],
        [0.0, 3.0, 1.5],
        [1.5, 3.5, 1.5],
        [-1.5, 3.5, 1.5],
    ])[:n_rx]

    rng = np.random.RandomState(42)
    rx_per_frame = []
    current_rx = base_rx.copy()
    for _ in range(n_frames):
        current_rx = current_rx + rng.randn(n_rx, 3) * 0.05
        current_rx[:, 2] = 1.5
        rx_per_frame.append(torch.tensor(current_rx.copy(), dtype=torch.float32))

    tracer = Tracer(
        frequency=1e9,
        scene=scene,
        reflection_n_rays=1000,
        reflection_max_bounces=3,
        max_diffractions=2,
    )
    path_max_diffractions = tracer.config.trace.max_diffractions
    depth_report = resolve_path_monitor_diffraction_depth_report(
        tracer.config.trace,
        requested_max_diffractions=path_max_diffractions,
    )

    print("=" * 70)
    print(f"Scenario: {n_tx} TX x {n_rx} RX x {n_frames} frames")
    print(f"  Scene: 4 boxes, RX moves each frame (z fixed at 1.5)")
    print(f"  Path diffraction depth: {format_path_monitor_diffraction_depth_report(depth_report)}")
    print("=" * 70)

    # Warmup
    for _ in range(2):
        for tx in tx_positions:
            monitor = PathMonitor(
                "rx",
                positions=rx_per_frame[0],
                max_diffractions=path_max_diffractions,
            )
            tracer.trace(tx, monitor=monitor, verbose=False)
            _sync()

    # ========================================
    # A: Naive
    # ========================================
    print(f"\n--- A: Naive (full trace every frame) ---")
    per_call_times = []
    total_t0 = time.perf_counter()
    for frame_idx in range(n_frames):
        rx_pos = rx_per_frame[frame_idx]
        for tx in tx_positions:
            monitor = PathMonitor(
                "rx",
                positions=rx_pos,
                max_diffractions=path_max_diffractions,
            )
            t0 = time.perf_counter()
            result = tracer.trace(tx, monitor=monitor, verbose=False, return_timing=True)
            _sync()
            per_call_times.append((time.perf_counter() - t0) * 1000.0)
    total_naive = (time.perf_counter() - total_t0) * 1000.0

    # Per-stage from last trace
    meta = result.paths("rx").metadata
    timing = meta.get("timing", {})

    print(f"  Per trace call: {np.mean(per_call_times):.1f} ms  (std: {np.std(per_call_times):.1f})")
    print(f"  Per frame ({n_tx} TX): {total_naive / n_frames:.1f} ms")
    print(f"  FPS: {1000.0 / (total_naive / n_frames):.1f}")
    for key, val in timing.items():
        print(f"    {key}: {val*1000:.1f} ms")

    # ========================================
    # B: Cache reflection_detail per TX (use internal mechanism)
    # ========================================
    print(f"\n--- B: Cache reflection_detail per TX ---")

    # Import internals for manual caching
    from witwin.channel.monitors.path.trace import trace_path_monitor
    from witwin.channel.trace.tracer import resolve_solver_controls

    solver_controls = resolve_solver_controls(
        tracer.config.trace,
        execution_intent="path_export",
        max_diffractions_override=path_max_diffractions,
    )
    config = tracer._resolved_trace_config

    # Build reflection cache: one discovery per TX
    cache_t0 = time.perf_counter()
    ref_caches = {}
    for tx_idx, tx_torch in enumerate(tx_positions):
        tx_bk = wt.Point3f(float(tx_torch[0]), float(tx_torch[1]), float(tx_torch[2]))
        monitor = PathMonitor(
            "rx",
            positions=rx_per_frame[0],
            max_diffractions=path_max_diffractions,
        )
        _, ref_detail = trace_path_monitor(
            tx_bk, monitor, scene, config, solver_controls,
            reflection_detail=None, verbose=False,
        )
        _sync()
        ref_caches[tx_idx] = ref_detail
    cache_build = (time.perf_counter() - cache_t0) * 1000.0
    print(f"  Cache build ({n_tx} TX): {cache_build:.0f} ms")

    # Now trace with cached reflection_detail
    per_call_cached = []
    total_t0 = time.perf_counter()
    for frame_idx in range(n_frames):
        rx_pos = rx_per_frame[frame_idx]
        for tx_idx, tx_torch in enumerate(tx_positions):
            tx_bk = wt.Point3f(float(tx_torch[0]), float(tx_torch[1]), float(tx_torch[2]))
            monitor = PathMonitor(
                "rx",
                positions=rx_pos,
                max_diffractions=path_max_diffractions,
            )
            t0 = time.perf_counter()
            path_result, _ = trace_path_monitor(
                tx_bk, monitor, scene, config, solver_controls,
                reflection_detail=ref_caches[tx_idx],
                verbose=False, return_timing=True,
            )
            _sync()
            per_call_cached.append((time.perf_counter() - t0) * 1000.0)
    total_cached = (time.perf_counter() - total_t0) * 1000.0

    meta_b = path_result.metadata
    timing_b = meta_b.get("timing", {})

    print(f"  Per trace call: {np.mean(per_call_cached):.1f} ms  (std: {np.std(per_call_cached):.1f})")
    print(f"  Per frame ({n_tx} TX): {total_cached / n_frames:.1f} ms")
    print(f"  FPS: {1000.0 / (total_cached / n_frames):.1f}")
    print(f"  Speedup vs naive: {total_naive / total_cached:.2f}x")
    for key, val in timing_b.items():
        print(f"    {key}: {val*1000:.1f} ms")

    # ========================================
    # C: Cache reflection + diffraction state_arrays per TX
    # ========================================
    print(f"\n--- C: Cache reflection + diffraction states per TX ---")

    from witwin.channel.monitors.path.trace import _receiver_groups, _gather_positions
    from witwin.channel.trace.diffraction.builders import (
        _build_solver_metadata,
        _prepare_diffraction_state_arrays,
    )
    from witwin.channel.monitors.path.collectors import (
        collect_los_paths,
        collect_reflection_paths,
        collect_diffraction_state_paths,
    )
    from witwin.channel.monitors.path.result import PathResult
    from witwin.channel.monitors.path.trace import _remap_raw_rx_index
    from witwin.channel.utils import scalar
    from witwin.channel.utils.polarization import effective_rx_polarization

    effective = solver_controls["effective"]

    # Build diffraction state cache per TX
    diff_cache_t0 = time.perf_counter()
    diff_caches = {}  # tx_idx -> list of (receiver_z, group_indices, edge_cache, edge_data, state_arrays)
    for tx_idx, tx_torch in enumerate(tx_positions):
        tx_bk = wt.Point3f(float(tx_torch[0]), float(tx_torch[1]), float(tx_torch[2]))
        rx_bk = wt.Point3f(
            wt.Float(rx_per_frame[0][:, 0].tolist()),
            wt.Float(rx_per_frame[0][:, 1].tolist()),
            wt.Float(rx_per_frame[0][:, 2].tolist()),
        )
        groups = _receiver_groups(rx_bk)
        mixed_ref = ref_caches[tx_idx] if config.enable_rd_diffraction else None
        ref_n_rays = effective["reflection_n_rays"] if config.enable_rd_diffraction else 0
        ref_max = effective["reflection_max_bounces"] if config.enable_rd_diffraction else 0

        per_group = []
        for receiver_z, group_indices in groups:
            edge_cache, edge_data, state_arrays, _ = _prepare_diffraction_state_arrays(
                tx_bk, receiver_z, scene,
                config.wavelength, config.k,
                mixed_ref, config.diffraction_material,
                ref_n_rays, ref_max, config.reflection_coef,
                "exhaustive", effective["max_diffractions"],
                total_state_budget_per_order=effective["diffraction_state_budget"],
                inserted_state_budget_per_order=effective["inserted_reflection_state_budget"],
                max_inserted_reflections_per_path=effective["max_inserted_reflections_per_path"],
                use_scene_materials=config.use_scene_materials_for_diffraction,
                tx_polarization=config.tx_polarization,
            )
            _sync()
            per_group.append((receiver_z, group_indices, edge_cache, edge_data, state_arrays))
        diff_caches[tx_idx] = per_group
    diff_cache_build = (time.perf_counter() - diff_cache_t0) * 1000.0
    print(f"  Diffraction cache build ({n_tx} TX): {diff_cache_build:.0f} ms")

    # Trace with both caches
    per_call_full_cache = []
    total_t0 = time.perf_counter()
    for frame_idx in range(n_frames):
        rx_pos = rx_per_frame[frame_idx]
        rx_bk = wt.Point3f(
            wt.Float(rx_pos[:, 0].tolist()),
            wt.Float(rx_pos[:, 1].tolist()),
            wt.Float(rx_pos[:, 2].tolist()),
        )
        for tx_idx, tx_torch in enumerate(tx_positions):
            tx_bk = wt.Point3f(float(tx_torch[0]), float(tx_torch[1]), float(tx_torch[2]))
            t0 = time.perf_counter()

            # LoS
            los_raw = collect_los_paths(
                scene=scene, rx_positions=rx_bk, tx_pos=tx_bk,
                wavelength=config.wavelength, k=config.k,
                tx_polarization=config.tx_polarization,
                rx_polarization=config.rx_polarization,
            )

            # Reflection with cached detail
            reflection_raw, _ = collect_reflection_paths(
                scene=scene, rx_positions=rx_bk, tx_pos=tx_bk,
                wavelength=config.wavelength, k=config.k,
                n_rays=effective["reflection_n_rays"],
                max_reflections=effective["reflection_max_bounces"],
                mode="exhaustive",
                tx_polarization=config.tx_polarization,
                rx_polarization=config.rx_polarization,
                reflection_coef=config.reflection_coef,
                min_ray_contribution_threshold=config.min_ray_contribution_threshold,
                reflection_relative_permittivity=config.reflection_relative_permittivity,
                reflection_conductivity=config.reflection_conductivity,
                reflection_material=config.reflection_material,
                use_scene_materials=config.use_scene_materials_for_reflection,
                return_geometry=False,
                reflection_detail=ref_caches[tx_idx],
            )

            # Diffraction with cached states
            diff_raws = []
            for receiver_z, group_indices, edge_cache, edge_data, state_arrays in diff_caches[tx_idx]:
                group_rx = _gather_positions(rx_bk, group_indices)
                raw = collect_diffraction_state_paths(
                    state_arrays=state_arrays,
                    edge_data=edge_data if edge_data is not None else edge_cache.get("edge_data"),
                    scene=scene, rx_positions=group_rx, tx_pos=tx_bk,
                    wavelength=config.wavelength, k=config.k,
                    tx_polarization=config.tx_polarization,
                    rx_polarization=config.rx_polarization,
                    material_detail=config.diffraction_material,
                    return_geometry=False,
                )
                _remap_raw_rx_index(raw, group_indices)
                diff_raws.append(raw)

            # Assemble
            n_rx_actual = int(dr.width(rx_bk.x))
            pr = PathResult.from_raw_collections(
                name="rx", num_rx=n_rx_actual, max_num_paths=None,
                tx_pos=(scalar(tx_bk.x), scalar(tx_bk.y), scalar(tx_bk.z)),
                rx_positions=rx_bk, frequency=float(config.frequency),
                wavelength=float(config.wavelength),
                raw_collections=[los_raw, reflection_raw, *diff_raws],
                return_geometry=False, metadata={},
            )
            _sync()
            per_call_full_cache.append((time.perf_counter() - t0) * 1000.0)
    total_full_cache = (time.perf_counter() - total_t0) * 1000.0

    print(f"  Per trace call: {np.mean(per_call_full_cache):.1f} ms  (std: {np.std(per_call_full_cache):.1f})")
    print(f"  Per frame ({n_tx} TX): {total_full_cache / n_frames:.1f} ms")
    print(f"  FPS: {1000.0 / (total_full_cache / n_frames):.1f}")
    print(f"  Speedup vs naive: {total_naive / total_full_cache:.2f}x")

    # ========================================
    # Summary
    # ========================================
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"  Config: {n_tx} TX, {n_rx} RX, {n_frames} frames, 4 boxes")
    print(f"  {'Naive':30s}: {total_naive/n_frames:6.1f} ms/frame  ({1000/(total_naive/n_frames):5.1f} FPS)")
    print(f"  {'Cache refl only':30s}: {total_cached/n_frames:6.1f} ms/frame  ({1000/(total_cached/n_frames):5.1f} FPS)  {total_naive/total_cached:.1f}x")
    print(f"  {'Cache refl + diff states':30s}: {total_full_cache/n_frames:6.1f} ms/frame  ({1000/(total_full_cache/n_frames):5.1f} FPS)  {total_naive/total_full_cache:.1f}x")
    overhead = cache_build + diff_cache_build
    amortized = (total_full_cache + overhead) / n_frames
    print(f"  Amortized (incl {overhead:.0f}ms cache): {amortized:.1f} ms/frame over {n_frames} frames")
    print(f"{'='*70}")


if __name__ == "__main__":
    profile_multi_frame()
