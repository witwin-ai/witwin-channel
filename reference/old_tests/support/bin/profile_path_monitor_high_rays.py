"""Profile PathMonitor with very high ray counts to find scaling bottlenecks."""

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

# Monkey-patch sub-stages for detailed timing
_timings = defaultdict(list)

def _sync():
    dr.sync_thread()
    torch.cuda.synchronize()

def _timed_start():
    _sync()
    return time.perf_counter()

def _timed_end(label, t0):
    _sync()
    _timings[label].append((time.perf_counter() - t0) * 1000.0)

# Patch reflection internals
from witwin.channel.trace.reflection import api as ref_api
_orig_trace_refl = ref_api._trace_reflection_paths
def _patched_trace_refl(*a, **kw):
    t0 = _timed_start()
    r = _orig_trace_refl(*a, **kw)
    _timed_end("refl._trace_rays", t0)
    return r
ref_api._trace_reflection_paths = _patched_trace_refl

# Patch _trace_reflection_paths separately (sub-stage of discovery)
# Already patched above via _patched_trace_refl

# Patch EPC
from witwin.channel.trace.reflection import epc as epc_mod
from witwin.channel.monitors.path import collectors as coll_mod
_orig_epc = epc_mod.epc_reflection_chain_to_target
def _patched_epc(*a, **kw):
    t0 = _timed_start()
    r = _orig_epc(*a, **kw)
    _timed_end("refl.replay", t0)
    return r
coll_mod.epc_reflection_chain_to_target = _patched_epc

# Patch diffraction
from witwin.channel.trace.diffraction import builders as diff_builders
from witwin.channel.monitors.path import trace as tp_mod
_orig_prep = diff_builders._prepare_diffraction_state_arrays
def _patched_prep(*a, **kw):
    t0 = _timed_start()
    r = _orig_prep(*a, **kw)
    _timed_end("diff.state_prep", t0)
    return r
tp_mod._prepare_diffraction_state_arrays = _patched_prep

from witwin.channel.trace.diffraction import field as field_mod
_orig_field = field_mod._edge_state_field_to_targets
def _patched_field(*a, **kw):
    t0 = _timed_start()
    r = _orig_field(*a, **kw)
    _timed_end("diff.utd_eval", t0)
    return r
coll_mod._edge_state_field_to_targets = _patched_field

from witwin.channel.kernels import packed_state as ps_mod
_orig_gather = ps_mod.gather_state_arrays
def _patched_gather(*a, **kw):
    t0 = _timed_start()
    r = _orig_gather(*a, **kw)
    _timed_end("diff.gather_states", t0)
    return r
coll_mod.gather_state_arrays = _patched_gather

from witwin.channel.trace.diffraction import geometry as geom_mod
_orig_vis = geom_mod._segment_visibility_mask
def _patched_vis(*a, **kw):
    t0 = _timed_start()
    r = _orig_vis(*a, **kw)
    _timed_end("diff.visibility", t0)
    return r
coll_mod._segment_visibility_mask = _patched_vis

# Patch discovery wrapper
_orig_discover = ref_api.discover_reflection_paths
def _patched_discover(*a, **kw):
    t0 = _timed_start()
    r = _orig_discover(*a, **kw)
    _timed_end("refl.discovery_total", t0)
    return r
coll_mod.discover_reflection_paths = _patched_discover

# Patch assembly
from witwin.channel.monitors.path.result import PathResult
_orig_from_raw = PathResult.from_raw_collections.__func__
@classmethod
def _patched_from_raw(cls, **kwargs):
    t0 = _timed_start()
    r = _orig_from_raw(cls, **kwargs)
    _timed_end("assembly", t0)
    return r
PathResult.from_raw_collections = _patched_from_raw

from tests._scene_helpers import box_geometry, build_scene
from tests.support.bin._path_monitor_phase6 import (
    format_path_monitor_diffraction_depth_report,
    resolve_path_monitor_diffraction_depth_report,
)
from witwin.channel import PathMonitor, Tracer


def _build_scene(n_boxes=4):
    boxes = []
    offsets = [
        (-3.0, -3.0), (3.0, -3.0), (-3.0, 3.0), (3.0, 3.0),
        (0.0, -5.0), (0.0, 5.0), (-5.0, 0.0), (5.0, 0.0),
    ]
    for i in range(min(n_boxes, len(offsets))):
        ox, oy = offsets[i]
        boxes.append(box_geometry(center=(ox, oy, 1.5), size=2.0))
    return build_scene(*boxes)


def profile_high_rays(n_rays, n_rx=5, n_boxes=4, max_bounces=3, max_diff=2):
    global _timings
    scene = _build_scene(n_boxes)

    rx = np.array([[1,2,1.5],[-1,2,1.5],[0,3,1.5],[1.5,3.5,1.5],[-1.5,3.5,1.5]])[:n_rx]
    rx_pos = torch.tensor(rx, dtype=torch.float32)
    tx = torch.tensor([0.0, -6.0, 1.5], dtype=torch.float32)

    tracer = Tracer(
        frequency=1e9, scene=scene,
        reflection_n_rays=n_rays,
        reflection_max_bounces=max_bounces,
        max_diffractions=max_diff,
    )
    depth_report = resolve_path_monitor_diffraction_depth_report(
        tracer.config.trace,
        requested_max_diffractions=max_diff,
    )

    # Warmup
    _timings.clear()
    for _ in range(2):
        _timings.clear()
        monitor = PathMonitor(
            "rx",
            positions=rx_pos,
            max_diffractions=max_diff,
        )
        tracer.trace(tx, monitor=monitor, verbose=False)
        _sync()

    # Timed runs
    n_runs = 3
    total_times = []
    all_timings = defaultdict(list)

    for _ in range(n_runs):
        _timings.clear()
        monitor = PathMonitor(
            "rx",
            positions=rx_pos,
            max_diffractions=max_diff,
        )
        t0 = time.perf_counter()
        result = tracer.trace(tx, monitor=monitor, verbose=False)
        _sync()
        total_times.append((time.perf_counter() - t0) * 1000.0)
        for k, v in _timings.items():
            all_timings[k].append(sum(v))

    meta = result.paths("rx").metadata
    pc = meta.get("path_counts", {})

    print(f"\n{'='*70}")
    print(f"n_rays={n_rays:,}, {n_rx} RX, {n_boxes} boxes, {max_bounces} bounces, {max_diff} diff")
    print(f"Path diffraction depth: {format_path_monitor_diffraction_depth_report(depth_report)}")
    print(f"Paths: LoS={pc.get('los','?')}, Refl={pc.get('reflection','?')}, Diff={pc.get('diffraction','?')}")
    print(f"-" * 70)

    avg_total = np.mean(total_times)
    print(f"Total: {avg_total:.1f} ms")

    ordered = [
        "refl.discovery_total", "refl._trace_rays", "refl.replay",
        "diff.state_prep", "diff.visibility", "diff.gather_states", "diff.utd_eval",
        "assembly",
    ]
    accounted = 0.0
    for key in ordered:
        if key in all_timings:
            avg = np.mean(all_timings[key])
            pct = avg / avg_total * 100
            # Don't double-count sub-stages of discovery_total
            if key not in ("refl._trace_rays",):
                accounted += avg
            prefix = "  " if key in ("refl._trace_rays", "refl._classify") else ""
            print(f"  {prefix}{key:>25s}: {avg:8.1f} ms  ({pct:5.1f}%)")

    remainder = avg_total - accounted
    if abs(remainder) > 0.1:
        print(f"  {'other':>25s}: {remainder:8.1f} ms  ({remainder/avg_total*100:5.1f}%)")
    print(f"{'='*70}")


if __name__ == "__main__":
    print("High Ray Count Scaling Profile")
    print("=" * 70)

    for n_rays in [1_000, 10_000, 100_000, 500_000, 1_000_000]:
        try:
            profile_high_rays(n_rays)
        except Exception as e:
            import traceback
            print(f"\nn_rays={n_rays} failed: {e}")
            traceback.print_exc()
