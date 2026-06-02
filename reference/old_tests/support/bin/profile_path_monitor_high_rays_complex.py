"""Profile PathMonitor with high ray counts in a COMPLEX scene."""

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

# Patches
from witwin.channel.trace.reflection import api as ref_api
from witwin.channel.trace.reflection import epc as epc_mod
from witwin.channel.monitors.path import collectors as coll_mod
from witwin.channel.monitors.path import trace as tp_mod
from witwin.channel.trace.diffraction import builders as diff_builders
from witwin.channel.trace.diffraction import field as field_mod
from witwin.channel.trace.diffraction import geometry as geom_mod
from witwin.channel.kernels import packed_state as ps_mod
from witwin.channel.monitors.path.result import PathResult

_orig_trace_refl = ref_api._trace_reflection_paths
def _p1(*a, **kw):
    t0 = _timed_start(); r = _orig_trace_refl(*a, **kw); _timed_end("refl.trace_rays", t0); return r
ref_api._trace_reflection_paths = _p1

_orig_discover = ref_api.discover_reflection_paths
def _p2(*a, **kw):
    t0 = _timed_start(); r = _orig_discover(*a, **kw); _timed_end("refl.discovery", t0); return r
coll_mod.discover_reflection_paths = _p2

_orig_epc = epc_mod.epc_reflection_chain_to_target
def _p3(*a, **kw):
    t0 = _timed_start(); r = _orig_epc(*a, **kw); _timed_end("refl.replay", t0); return r
coll_mod.epc_reflection_chain_to_target = _p3

_orig_prep = diff_builders._prepare_diffraction_state_arrays
def _p4(*a, **kw):
    t0 = _timed_start(); r = _orig_prep(*a, **kw); _timed_end("diff.state_prep", t0); return r
tp_mod._prepare_diffraction_state_arrays = _p4

_orig_field = field_mod._edge_state_field_to_targets
def _p5(*a, **kw):
    t0 = _timed_start(); r = _orig_field(*a, **kw); _timed_end("diff.utd_eval", t0); return r
coll_mod._edge_state_field_to_targets = _p5

_orig_gather = ps_mod.gather_state_arrays
def _p6(*a, **kw):
    t0 = _timed_start(); r = _orig_gather(*a, **kw); _timed_end("diff.gather", t0); return r
coll_mod.gather_state_arrays = _p6

_orig_vis = geom_mod._segment_visibility_mask
def _p7(*a, **kw):
    t0 = _timed_start(); r = _orig_vis(*a, **kw); _timed_end("diff.vis", t0); return r
coll_mod._segment_visibility_mask = _p7

_orig_from_raw = PathResult.from_raw_collections.__func__
@classmethod
def _p8(cls, **kw):
    t0 = _timed_start(); r = _orig_from_raw(cls, **kw); _timed_end("assembly", t0); return r
PathResult.from_raw_collections = _p8


from tests._scene_helpers import box_geometry, build_scene
from tests.support.bin._path_monitor_phase6 import (
    format_path_monitor_diffraction_depth_report,
    resolve_path_monitor_diffraction_depth_report,
)
from witwin.channel import PathMonitor, Tracer


def _build_complex_scene(n_buildings=16):
    """Urban-like scene é–?buildings along a street corridor."""
    rng = np.random.RandomState(123)
    boxes = []
    # Place buildings along two sides of a street (y axis is street direction)
    for i in range(n_buildings):
        side = -1 if i % 2 == 0 else 1
        along = (i // 2) * 5.0 - (n_buildings // 4) * 5.0
        h = rng.uniform(2.0, 6.0)
        w = rng.uniform(1.5, 3.0)
        d = rng.uniform(2.0, 4.0)
        cx = side * (3.0 + w / 2 + rng.uniform(0, 1.0))
        cy = along + rng.uniform(-0.5, 0.5)
        boxes.append(box_geometry(center=(cx, cy, h / 2), size=(w, d, h)))
    # Add a ground plane
    boxes.append(box_geometry(center=(0, 0, -0.1), size=(30, 60, 0.2)))
    return build_scene(*boxes)


def profile(n_rays, n_rx=5, n_buildings=16, max_bounces=3, max_diff=1):
    global _timings
    scene = _build_complex_scene(n_buildings)

    # TX at one end of street, RX at the other
    rx = np.array([[0,12,1.5],[-1,10,1.5],[1,10,1.5],[0,14,1.5],[0.5,11,1.5]])[:n_rx]
    rx_pos = torch.tensor(rx, dtype=torch.float32)
    tx = torch.tensor([0.0, -12.0, 3.0], dtype=torch.float32)

    tracer = Tracer(
        frequency=1e9, scene=scene,
        reflection_n_rays=n_rays, reflection_max_bounces=max_bounces,
        max_diffractions=max_diff,
    )
    depth_report = resolve_path_monitor_diffraction_depth_report(
        tracer.config.trace,
        requested_max_diffractions=max_diff,
    )

    # Warmup
    for _ in range(2):
        _timings.clear()
        tracer.trace(
            tx,
            monitor=PathMonitor(
                "rx",
                positions=rx_pos,
                max_diffractions=max_diff,
            ),
            verbose=False,
        )
        _sync()

    n_runs = 3
    total_times = []
    all_t = defaultdict(list)

    for _ in range(n_runs):
        _timings.clear()
        t0 = time.perf_counter()
        result = tracer.trace(
            tx,
            monitor=PathMonitor(
                "rx",
                positions=rx_pos,
                max_diffractions=max_diff,
            ),
            verbose=False,
        )
        _sync()
        total_times.append((time.perf_counter() - t0) * 1000.0)
        for k, v in _timings.items():
            all_t[k].append(sum(v))

    meta = result.paths("rx").metadata
    pc = meta.get("path_counts", {})

    print(f"\n{'='*70}")
    print(f"n_rays={n_rays:>9,} | {n_buildings} buildings | {n_rx} RX | {max_bounces}B {max_diff}D")
    print(f"Path diffraction depth: {format_path_monitor_diffraction_depth_report(depth_report)}")
    print(f"Paths: LoS={pc.get('los','?')}, Refl={pc.get('reflection','?')}, Diff={pc.get('diffraction','?')}")
    print(f"-" * 70)

    avg = np.mean(total_times)
    print(f"Total: {avg:.1f} ms")

    keys = ["refl.discovery", "refl.trace_rays", "refl.replay",
            "diff.state_prep", "diff.vis", "diff.gather", "diff.utd_eval",
            "assembly"]
    accounted = 0.0
    for key in keys:
        if key not in all_t:
            continue
        v = np.mean(all_t[key])
        pct = v / avg * 100
        indent = "    " if key == "refl.trace_rays" else ""
        if key != "refl.trace_rays":
            accounted += v
        print(f"  {indent}{key:>22s}: {v:8.1f} ms  ({pct:5.1f}%)")

    rem = avg - accounted
    if abs(rem) > 0.1:
        print(f"  {'other':>22s}: {rem:8.1f} ms  ({rem/avg*100:5.1f}%)")
    print(f"{'='*70}")


if __name__ == "__main__":
    print("High Ray Count Scaling é–?Complex Urban Scene")
    print("=" * 70)

    # Reflection-only first to isolate reflection scaling
    print("\n>>> REFLECTION ONLY (max_diff=0)")
    for n_rays in [1_000, 10_000, 100_000, 500_000, 1_000_000]:
        try:
            profile(n_rays, max_diff=0)
        except Exception as e:
            print(f"\nn_rays={n_rays} failed: {e}")

    print("\n\n>>> FULL (reflection + 1 diffraction order)")
    for n_rays in [1_000, 10_000, 100_000, 500_000]:
        try:
            profile(n_rays, max_diff=1)
        except Exception as e:
            print(f"\nn_rays={n_rays} failed: {e}")
