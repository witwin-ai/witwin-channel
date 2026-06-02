"""Detailed sub-stage profiling for PathMonitor collectors."""

from __future__ import annotations

import sys
import time
from pathlib import Path
from contextlib import contextmanager
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
from witwin.channel.monitors.path import collectors as coll_mod
from witwin.channel.monitors.path import trace as tp_mod
from witwin.channel.trace.reflection import api as ref_api
from witwin.channel.trace.diffraction import builders as diff_builders


def _sync():
    dr.sync_thread()
    torch.cuda.synchronize()


_timings = defaultdict(list)


@contextmanager
def _timed(label: str):
    _sync()
    t0 = time.perf_counter()
    yield
    _sync()
    _timings[label].append((time.perf_counter() - t0) * 1000.0)


# Monkey-patch discover_reflection_paths
_orig_discover = ref_api.discover_reflection_paths


def _patched_discover(*args, **kwargs):
    with _timed("ref.discovery"):
        return _orig_discover(*args, **kwargs)


# Monkey-patch epc_reflection_chain_to_target
from witwin.channel.trace.reflection import epc as epc_mod
_orig_epc = epc_mod.epc_reflection_chain_to_target


def _patched_epc(*args, **kwargs):
    with _timed("ref.replay"):
        return _orig_epc(*args, **kwargs)


# Monkey-patch _prepare_diffraction_state_arrays
_orig_prepare_diff = diff_builders._prepare_diffraction_state_arrays


def _patched_prepare_diff(*args, **kwargs):
    with _timed("diff.state_prep"):
        return _orig_prepare_diff(*args, **kwargs)


# Monkey-patch _edge_state_field_to_targets
from witwin.channel.trace.diffraction import field as field_mod
_orig_field_eval = field_mod._edge_state_field_to_targets


def _patched_field_eval(*args, **kwargs):
    with _timed("diff.utd_eval"):
        return _orig_field_eval(*args, **kwargs)


# Monkey-patch gather_state_arrays
from witwin.channel.kernels import packed_state as ps_mod
_orig_gather = ps_mod.gather_state_arrays


def _patched_gather(*args, **kwargs):
    with _timed("diff.gather_states"):
        return _orig_gather(*args, **kwargs)


# Monkey-patch _segment_visibility_mask
from witwin.channel.trace.diffraction import geometry as geom_mod
_orig_vis = geom_mod._segment_visibility_mask


def _patched_vis(*args, **kwargs):
    with _timed("diff.visibility"):
        return _orig_vis(*args, **kwargs)


# Monkey-patch PathResult.from_raw_collections
from witwin.channel.monitors.path.result import PathResult
_orig_from_raw = PathResult.from_raw_collections


@classmethod
def _patched_from_raw(cls, **kwargs):
    with _timed("assembly.from_raw"):
        return _orig_from_raw(**kwargs)


def _apply_patches():
    coll_mod.discover_reflection_paths = _patched_discover
coll_mod.epc_reflection_chain_to_target = _patched_epc
    tp_mod._prepare_diffraction_state_arrays = _patched_prepare_diff
    field_mod._edge_state_field_to_targets = _patched_field_eval
    coll_mod._edge_state_field_to_targets = _patched_field_eval
    coll_mod.gather_state_arrays = _patched_gather
    geom_mod._segment_visibility_mask = _patched_vis
    coll_mod._segment_visibility_mask = _patched_vis
    PathResult.from_raw_collections = _patched_from_raw


def _build_scene(n_boxes: int = 4):
    boxes = []
    offsets = [
        (-3.0, -3.0), (3.0, -3.0), (-3.0, 3.0), (3.0, 3.0),
        (0.0, -5.0), (0.0, 5.0), (-5.0, 0.0), (5.0, 0.0),
    ]
    for i in range(min(n_boxes, len(offsets))):
        ox, oy = offsets[i]
        boxes.append(box_geometry(center=(ox, oy, 1.5), size=2.0))
    return build_scene(*boxes)


def profile_detailed(n_rx: int, n_boxes: int, n_rays: int, max_bounces: int, max_diff: int):
    global _timings
    scene = _build_scene(n_boxes)
    rx_x = np.linspace(-4.0, 4.0, int(np.sqrt(n_rx)))
    rx_y = np.linspace(-4.0, 4.0, int(np.sqrt(n_rx)))
    rx_grid = np.array(np.meshgrid(rx_x, rx_y)).T.reshape(-1, 2)
    n_rx_actual = rx_grid.shape[0]
    rx_positions = torch.tensor(
        np.column_stack([rx_grid, np.full(n_rx_actual, 1.5)]),
        dtype=torch.float32,
    )

    tx = torch.tensor([0.0, -6.0, 1.5], dtype=torch.float32)
    monitor = PathMonitor(
        "rx",
        positions=rx_positions,
        max_diffractions=max_diff,
    )
    tracer = Tracer(
        frequency=1e9,
        scene=scene,
        reflection_n_rays=n_rays,
        reflection_max_bounces=max_bounces,
        max_diffractions=max_diff,
    )
    depth_report = resolve_path_monitor_diffraction_depth_report(
        tracer.config.trace,
        requested_max_diffractions=monitor.max_diffractions,
    )

    # Warmup
    for _ in range(2):
        _timings.clear()
        result = tracer.trace(tx, monitor=monitor, verbose=False)
        _sync()

    # Timed runs
    n_runs = 5
    total_times = []
    all_sub_timings = defaultdict(list)
    for _ in range(n_runs):
        _timings.clear()
        t0 = time.perf_counter()
        result = tracer.trace(tx, monitor=monitor, verbose=False)
        _sync()
        total_times.append((time.perf_counter() - t0) * 1000.0)

        for key, vals in _timings.items():
            all_sub_timings[key].append(sum(vals))

    paths = result.paths("rx")
    meta = paths.metadata

    print(f"\n{'='*70}")
    print(f"Config: {n_rx_actual} RX, {n_boxes} boxes, {n_rays} rays, "
          f"{max_bounces} bounces, {max_diff} diffractions")
    print(f"Path diffraction depth: {format_path_monitor_diffraction_depth_report(depth_report)}")
    print(f"Paths: LoS={meta.get('path_counts', {}).get('los', '?')}, "
          f"Refl={meta.get('path_counts', {}).get('reflection', '?')}, "
          f"Diff={meta.get('path_counts', {}).get('diffraction', '?')}")
    print(f"-" * 70)

    avg_total = np.mean(total_times)
    print(f"Total: {avg_total:.1f} ms  (std: {np.std(total_times):.1f})")
    print()

    ordered_keys = [
        "ref.discovery", "ref.replay",
        "diff.state_prep", "diff.visibility", "diff.gather_states", "diff.utd_eval",
        "assembly.from_raw",
    ]
    accounted = 0.0
    for key in ordered_keys:
        if key in all_sub_timings:
            vals = all_sub_timings[key]
            avg = np.mean(vals)
            pct = avg / avg_total * 100
            accounted += avg
            print(f"  {key:>25s}: {avg:7.1f} ms  ({pct:5.1f}%)")

    remainder = avg_total - accounted
    if remainder > 0.1:
        pct = remainder / avg_total * 100
        print(f"  {'other/overhead':>25s}: {remainder:7.1f} ms  ({pct:5.1f}%)")

    print(f"{'='*70}")


if __name__ == "__main__":
    _apply_patches()
    print("PathMonitor Detailed Sub-Stage Profiling")
    print("=" * 70)

    configs = [
        (100, 4, 1000, 3, 0),    # Reflection only
        (100, 4,    0, 0, 2),    # Diffraction only
        (100, 4, 1000, 3, 2),    # Full
        (400, 6, 2000, 3, 2),    # Large full
    ]

    for cfg in configs:
        try:
            profile_detailed(*cfg)
        except Exception as e:
            import traceback
            print(f"\nConfig {cfg} failed: {e}")
            traceback.print_exc()
