"""Profile standalone path reflection discovery scheduling on Munich.

This script separates RayD native reflection tracing from Witwin-side prefix
collection so multi-bounce optimization work can target the actual bottleneck.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

CHANNEL_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SIONNA_SOURCE_ROOT = CHANNEL_ROOT / "reference" / "sionna-rt-reference-2.0.1" / "src"
DEFAULT_MUNICH_XML = DEFAULT_SIONNA_SOURCE_ROOT / "sionna" / "rt" / "scenes" / "munich" / "munich.xml"
DEFAULT_TX_POSITIONS = ((8.5, 21.0, 27.0), (45.0, 15.0, 22.0))
DEFAULT_RX_POSITIONS = ((45.0, 90.0, 1.5), (30.0, 55.0, 1.5), (60.0, 20.0, 1.5))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sionna-source-root", type=Path, default=DEFAULT_SIONNA_SOURCE_ROOT)
    parser.add_argument("--munich-xml", type=Path, default=DEFAULT_MUNICH_XML)
    parser.add_argument("--frequency-hz", type=float, default=3.5e9)
    parser.add_argument("--num-samples", type=int, default=32768)
    parser.add_argument("--max-bounces", type=int, default=2)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--json", action="store_true", default=False)
    return parser


def _ensure_import_paths(sionna_source_root: Path) -> None:
    for path in (CHANNEL_ROOT, Path(sionna_source_root).resolve()):
        text = str(path)
        if text not in sys.path:
            sys.path.insert(0, text)


def _jsonable(value: Any) -> Any:
    try:
        import numpy as np
    except Exception:
        np = None
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if np is not None and isinstance(value, (np.floating, np.integer, np.bool_)):
        return value.item()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _time_stage(stats: dict[str, float], label: str, fn: Callable[[], Any]):
    import drjit as dr

    start = time.perf_counter()
    out = fn()
    dr.sync_thread()
    stats[label] = stats.get(label, 0.0) + time.perf_counter() - start
    return out


def _build_scene(args: argparse.Namespace):
    import witwin.channel as wc

    scene = wc.Scene.load_mitsuba(
        Path(args.munich_xml),
        source_root=Path(args.sionna_source_root),
        frequency=float(args.frequency_hz),
        merge_shapes=True,
        device="cuda",
    )
    for index, position in enumerate(DEFAULT_TX_POSITIONS):
        scene.add(wc.Transmitter(f"tx{index}", position))
    for index, position in enumerate(DEFAULT_RX_POSITIONS):
        scene.add(wc.Receiver(f"rx{index}", position))
    return scene


def _rx_positions():
    from witwin.channel.deterministic import types as wt

    return wt.Point3f(
        wt.Float([position[0] for position in DEFAULT_RX_POSITIONS]),
        wt.Float([position[1] for position in DEFAULT_RX_POSITIONS]),
        wt.Float([position[2] for position in DEFAULT_RX_POSITIONS]),
    )


def _profile_once(*, scene, n_rays: int, max_bounces: int, frequency_hz: float) -> dict[str, Any]:
    import drjit as dr
    import rayd
    from witwin.channel.core.runtime import Material, Tx, Wave
    from witwin.channel.deterministic import types as wt
    from witwin.channel.deterministic.reflection import common
    from witwin.channel.deterministic.reflection import paths as reflection_paths
    from witwin.channel.deterministic.reflection.paths import (
        PATH_IMAGE_SOURCE_TOL,
        collect_prefix_paths,
        enumerate_first_bounce_surface_paths,
    )
    from witwin.channel.deterministic.trace.path_export import _sampling_frame_from_positions

    stats: dict[str, float] = {}
    rx_positions = _rx_positions()
    sampling_axis, sampling_plane_position, sampling_bounds = _sampling_frame_from_positions(rx_positions)
    tri_data = scene._triangle_runtime()
    canonical = tri_data["surface_canonical_prim"]

    path_counts = []
    for tx_position in DEFAULT_TX_POSITIONS:
        tx = Tx(position=tx_position)
        ray_dir, sampling_info = _time_stage(
            stats,
            "select_ray_directions",
            lambda: common.select_ray_directions(
                axis=sampling_axis,
                bounds=sampling_bounds,
                tx=tx,
                n_rays=int(n_rays),
                mode="3d",
                plane_position=sampling_plane_position,
                ray_sampling="full_sphere",
            ),
        )
        ray_origin = wt.Point3f(
            dr.repeat(tx.position.x, int(n_rays)),
            dr.repeat(tx.position.y, int(n_rays)),
            dr.repeat(tx.position.z, int(n_rays)),
        )
        ray_origin_detached = dr.detached_t(wt.Point3f)(
            dr.detach(ray_origin.x),
            dr.detach(ray_origin.y),
            dr.detach(ray_origin.z),
        )
        ray_dir_detached = dr.detached_t(wt.Vector3f)(
            dr.detach(ray_dir.x),
            dr.detach(ray_dir.y),
            dr.detach(ray_dir.z),
        )
        options = rayd.ReflectionTraceOptions()
        options.deduplicate = True
        options.canonical_prim_table = canonical
        options.image_source_tolerance = float(PATH_IMAGE_SOURCE_TOL)

        chain = _time_stage(
            stats,
            "rayd.trace_reflections",
            lambda: scene._rayd_scene.trace_reflections(
                rayd.RayDetached(ray_origin_detached, ray_dir_detached),
                int(max_bounces),
                options,
                dr.full(dr.detached_t(wt.Bool), True, int(n_rays)),
                False,
            ),
        )
        source_paths = _time_stage(
            stats,
            "collect_prefix_paths",
            lambda: collect_prefix_paths(
                chain,
                chain_depth=int(max_bounces),
                surface_canonical_prims=canonical,
                image_source_tolerance=PATH_IMAGE_SOURCE_TOL,
            ),
        )
        analytic_first = _time_stage(
            stats,
            "enumerate_first_bounce_surface_paths",
            lambda: enumerate_first_bounce_surface_paths(tx=tx, tri_data=tri_data),
        )
        current_trace = _time_stage(
            stats,
            "trace_paths_current",
            lambda: reflection_paths.trace_paths(
                tx=tx,
                scene=scene,
                wave=Wave.from_frequency(float(frequency_hz)),
                n_rays=int(n_rays),
                max_reflections=int(max_bounces),
                mode="3d",
                material=Material(reflection_coef=1.0),
                ray_sampling="full_sphere",
                sampling_axis=sampling_axis,
                sampling_bounds=sampling_bounds,
                sampling_plane_position=sampling_plane_position,
                tri_data=tri_data,
            ),
        )
        path_counts.append({
            "sampled_per_depth": [int(paths.n_paths) for paths in source_paths],
            "trace_paths_per_depth": [
                int(paths.n_paths)
                for paths in current_trace["source_paths_per_bounce"]
            ],
            "analytic_first_bounce": int(analytic_first.n_paths),
            "sampling": dict(sampling_info),
        })

    return {
        "timing_seconds": stats,
        "path_counts": path_counts,
    }


def run_profile(args: argparse.Namespace) -> dict[str, Any]:
    _ensure_import_paths(Path(args.sionna_source_root))
    scene = _build_scene(args)

    for _ in range(max(0, int(args.warmup))):
        _profile_once(
            scene=scene,
            n_rays=int(args.num_samples),
            max_bounces=int(args.max_bounces),
            frequency_hz=float(args.frequency_hz),
        )

    samples = [
        _profile_once(
            scene=scene,
            n_rays=int(args.num_samples),
            max_bounces=int(args.max_bounces),
            frequency_hz=float(args.frequency_hz),
        )
        for _ in range(max(1, int(args.repeats)))
    ]
    return {
        "scenario": {
            "scene": "munich",
            "frequency_hz": float(args.frequency_hz),
            "num_samples": int(args.num_samples),
            "max_bounces": int(args.max_bounces),
            "num_tx": len(DEFAULT_TX_POSITIONS),
            "num_rx": len(DEFAULT_RX_POSITIONS),
        },
        "samples": samples,
    }


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = run_profile(args)
    if args.json:
        print(json.dumps(_jsonable(report), indent=2, sort_keys=True))
    else:
        for index, sample in enumerate(report["samples"]):
            print(f"sample {index}: {sample['timing_seconds']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
