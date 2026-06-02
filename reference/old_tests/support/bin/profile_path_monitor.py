"""Profile PathMonitor pipeline to identify bottleneck stages."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import drjit as dr
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
CORE_ROOT = ROOT.parent / "core"
for root in (CORE_ROOT, ROOT):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from tests._scene_helpers import box_geometry, build_scene
from tests.support.bin._path_monitor_phase6 import (
    format_path_monitor_diffraction_depth_report,
    resolve_path_monitor_diffraction_depth_report,
)
from witwin.channel import PathMonitor, Tracer
import witwin as wt


def _sync():
    dr.sync_thread()
    torch.cuda.synchronize()


def _build_scene(n_boxes: int = 4):
    """Build a scene with multiple boxes to generate enough paths."""
    boxes = []
    offsets = [
        (-3.0, -3.0), (3.0, -3.0), (-3.0, 3.0), (3.0, 3.0),
        (0.0, -5.0), (0.0, 5.0), (-5.0, 0.0), (5.0, 0.0),
    ]
    for i in range(min(n_boxes, len(offsets))):
        ox, oy = offsets[i]
        boxes.append(box_geometry(center=(ox, oy, 1.5), size=2.0))
    return build_scene(*boxes)


def profile_path_monitor(n_rx: int, n_boxes: int, n_rays: int, max_bounces: int, max_diff: int):
    scene = _build_scene(n_boxes)

    # Generate receiver grid
    rx_x = np.linspace(-4.0, 4.0, int(np.sqrt(n_rx)))
    rx_y = np.linspace(-4.0, 4.0, int(np.sqrt(n_rx)))
    rx_grid = np.array(np.meshgrid(rx_x, rx_y)).T.reshape(-1, 2)
    n_rx_actual = rx_grid.shape[0]
    rx_positions = torch.tensor(
        np.column_stack([rx_grid, np.full(n_rx_actual, 1.5)]),
        dtype=torch.float32,
    )

    tx = torch.tensor([0.0, -6.0, 1.5], dtype=torch.float32)
    tracer = Tracer(
        frequency=1e9,
        scene=scene,
        reflection_n_rays=n_rays,
        reflection_max_bounces=max_bounces,
        max_diffractions=max_diff,
    )
    monitor = PathMonitor(
        "rx",
        positions=rx_positions,
        max_diffractions=max_diff,
    )
    depth_report = resolve_path_monitor_diffraction_depth_report(
        tracer.config.trace,
        requested_max_diffractions=monitor.max_diffractions,
    )

    # Warmup
    for _ in range(2):
        result = tracer.trace(tx, monitor=monitor, verbose=False, return_timing=True)
        _sync()

    # Timed runs
    n_runs = 5
    timings_list = []
    total_times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        result = tracer.trace(tx, monitor=monitor, verbose=False, return_timing=True)
        _sync()
        total_times.append((time.perf_counter() - t0) * 1000.0)

        meta = result.paths("rx").metadata
        if "timing" in meta:
            timings_list.append(dict(meta["timing"]))

    paths = result.paths("rx")
    n_valid = int(np.asarray(paths.valid, dtype=np.bool_).sum())

    print(f"\n{'='*60}")
    print(f"Config: {n_rx_actual} RX, {n_boxes} boxes, {n_rays} rays, "
          f"{max_bounces} bounces, {max_diff} diffractions")
    print(f"Path diffraction depth: {format_path_monitor_diffraction_depth_report(depth_report)}")
    print(f"Paths found: {n_valid} valid out of {paths.max_num_paths} max × {n_rx_actual} RX")
    print(f"  LoS: {meta.get('path_counts', {}).get('los', '?')}")
    print(f"  Reflection: {meta.get('path_counts', {}).get('reflection', '?')}")
    print(f"  Diffraction: {meta.get('path_counts', {}).get('diffraction', '?')}")
    print(f"-" * 60)

    avg_total = np.mean(total_times)
    print(f"Total trace time: {avg_total:.1f} ms  (std: {np.std(total_times):.1f} ms)")

    if timings_list:
        for key in timings_list[0]:
            vals = [t[key] * 1000.0 for t in timings_list]
            pct = np.mean(vals) / avg_total * 100
            print(f"  {key:>15s}: {np.mean(vals):8.1f} ms  ({pct:5.1f}%)  std={np.std(vals):.1f}")

    assembly_time = avg_total - sum(
        np.mean([t[k] * 1000.0 for t in timings_list])
        for k in timings_list[0]
    ) if timings_list else 0
    if timings_list:
        pct = assembly_time / avg_total * 100
        print(f"  {'assembly':>15s}: {assembly_time:8.1f} ms  ({pct:5.1f}%)  (result build + overhead)")
    print(f"{'='*60}")
    return avg_total


if __name__ == "__main__":
    print("PathMonitor Pipeline Profiling")
    print("=" * 60)

    configs = [
        # (n_rx, n_boxes, n_rays, max_bounces, max_diffractions)
        (16,  2,   500,  2, 0),   # Small, reflection only
        (100, 4,  1000,  3, 0),   # Medium, reflection only
        (100, 4,     0,  0, 2),   # Medium, diffraction only
        (100, 4,  1000,  3, 2),   # Medium, full
        (400, 6,  2000,  3, 2),   # Large, full
    ]

    for cfg in configs:
        try:
            profile_path_monitor(*cfg)
        except Exception as e:
            print(f"\nConfig {cfg} failed: {e}")
