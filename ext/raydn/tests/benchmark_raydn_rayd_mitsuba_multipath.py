from __future__ import annotations

import argparse
import csv
import gc
import importlib
import json
import math
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Callable

import torch

import raydn as rt

try:
    from .benchmark_raydn_rayd_mitsuba_stress import (
        RAYDI_ROOT,
        _cleanup_drjit,
        _cleanup_torch,
        _load_rayd,
        _try_import_mitsuba,
    )
except ImportError:  # pragma: no cover - supports direct script execution.
    from benchmark_raydn_rayd_mitsuba_stress import (  # type: ignore
        RAYDI_ROOT,
        _cleanup_drjit,
        _cleanup_torch,
        _load_rayd,
        _try_import_mitsuba,
    )


DEFAULT_SIONNA_ROOT = Path(r"E:\Code\witwin-platform\channel\reference\sionna-rt-reference-2.0.1")
SPEED_OF_LIGHT = 299_792_458.0
WORKLOADS = ("reflection_trace", "diffraction_export")

PRESETS: dict[str, dict[str, Any]] = {
    "smoke": {
        "ray_counts": [4_096],
        "state_counts": [4_096],
        "max_bounces": [2],
        "repeats": 2,
        "warmup": 1,
    },
    "standard": {
        "ray_counts": [65_536, 1_048_576],
        "state_counts": [65_536, 1_048_576],
        "max_bounces": [2, 4],
        "repeats": 5,
        "warmup": 2,
    },
    "large": {
        "ray_counts": [65_536, 1_048_576, 10_485_760],
        "state_counts": [65_536, 1_048_576, 10_485_760],
        "max_bounces": [2, 4],
        "repeats": 5,
        "warmup": 2,
    },
}


def _parse_int_list(values: list[str] | None, default: list[int]) -> list[int]:
    if not values:
        return default
    out: list[int] = []
    for value in values:
        for part in value.split(","):
            part = part.strip().replace("_", "")
            if part:
                out.append(int(part))
    if not out:
        raise ValueError("integer list must not be empty")
    return out


def _format_count(value: int) -> str:
    if value >= 1_000_000_000:
        return f"{value / 1_000_000_000:.3g}B"
    if value >= 1_000_000:
        return f"{value / 1_000_000:.3g}M"
    if value >= 1_000:
        return f"{value / 1_000:.3g}K"
    return str(value)


def _summarize_samples(samples_ms: list[float]) -> dict[str, Any]:
    ordered = sorted(samples_ms)
    return {
        "samples_ms": samples_ms,
        "min_ms": min(samples_ms),
        "avg_ms": statistics.fmean(samples_ms),
        "p50_ms": statistics.median(samples_ms),
        "p95_ms": ordered[max(0, math.ceil(0.95 * len(ordered)) - 1)],
    }


def _measure(
    fn: Callable[[], Any],
    materialize: Callable[[Any], None],
    sync: Callable[[], None],
    repeats: int,
    warmup: int,
) -> dict[str, Any]:
    for _ in range(warmup):
        value = fn()
        materialize(value)
        sync()

    samples_ms: list[float] = []
    last_value: Any = None
    for _ in range(repeats):
        sync()
        start = time.perf_counter()
        last_value = fn()
        materialize(last_value)
        sync()
        samples_ms.append((time.perf_counter() - start) * 1000.0)
    return {**_summarize_samples(samples_ms), "last_value": last_value}


def _time_build(fn: Callable[[], Any], sync: Callable[[], None]) -> tuple[Any, float]:
    start = time.perf_counter()
    value = fn()
    sync()
    return value, (time.perf_counter() - start) * 1000.0


def _reflection_trace_output_bytes(ray_count: int, max_bounces: int) -> int:
    slot_count = ray_count * max_bounces
    return ray_count * 4 + slot_count * (4 + 4 + 4)


def _dfr_path_output_bytes(capacity: int) -> int:
    per_path = 1 + 6 * 4 + 4 + 3 * 2 * 4 + 3 * 3 * 4
    return 4 + capacity * per_path


def _base_metric(
    *,
    backend: str,
    workload: str,
    scene: str,
    input_count: int,
    valid_path_count: int,
    estimated_output_bytes: int,
    timing: dict[str, Any],
    build_ms: float = 0.0,
    **extra: Any,
) -> dict[str, Any]:
    result = {
        "backend": backend,
        "workload": workload,
        "scene": scene,
        "input_count": input_count,
        "valid_path_count": valid_path_count,
        "estimated_output_bytes": estimated_output_bytes,
        "estimated_output_mib": estimated_output_bytes / (1024.0 * 1024.0),
        "build_ms": build_ms,
        "timing": timing,
    }
    result.update(extra)
    return result


def _torch_parallel_reflector_scene() -> tuple[rt.Scene, float]:
    vertices = torch.tensor(
        [
            [-1.0, -1.0, 0.0],
            [1.0, -1.0, 0.0],
            [1.0, 1.0, 0.0],
            [-1.0, 1.0, 0.0],
            [-1.0, -1.0, 1.0],
            [1.0, -1.0, 1.0],
            [1.0, 1.0, 1.0],
            [-1.0, 1.0, 1.0],
        ],
        device="cuda",
        dtype=torch.float32,
    )
    faces = torch.tensor(
        [[0, 1, 2], [0, 2, 3], [4, 5, 6], [4, 6, 7]],
        device="cuda",
        dtype=torch.int32,
    )
    scene = rt.Scene()
    scene.add_mesh(rt.Mesh(vertices, faces))
    _, build_ms = _time_build(scene.build, torch.cuda.synchronize)
    return scene, build_ms


def _torch_reflection_ray(ray_count: int) -> rt.Ray:
    grid_side = int(math.ceil(math.sqrt(ray_count)))
    idx = torch.arange(ray_count, device="cuda", dtype=torch.int64)
    ix = idx % grid_side
    iy = idx // grid_side
    denom = max(1, grid_side - 1)
    x = -0.9 + 1.8 * ix.to(torch.float32) / float(denom)
    y = -0.9 + 1.8 * iy.to(torch.float32) / float(denom)
    origins = torch.stack((x, y, torch.full_like(x, 0.5)), dim=1).contiguous()
    directions = torch.stack(
        (
            torch.zeros_like(x),
            torch.zeros_like(x),
            torch.ones_like(x),
        ),
        dim=1,
    ).contiguous()
    return rt.Ray(origins, directions)


def run_raydn_reflection_trace(args: argparse.Namespace, ray_count: int, max_bounces: int) -> dict[str, Any]:
    scene, build_ms = _torch_parallel_reflector_scene()
    ray = _torch_reflection_ray(ray_count)

    def call_kernel():
        chain = scene.trace_reflections(ray, max_bounces=max_bounces)
        return chain.valid, chain.t, chain.prim_ids

    measured = _measure(call_kernel, lambda _value: None, torch.cuda.synchronize, args.repeats, args.warmup)
    valid, t, _prim_ids = measured.pop("last_value")
    counts, checksum = torch.ops.raydn.reflection_trace_stats(valid.contiguous(), t.contiguous())
    slot_count = int(counts[0].item())
    full_depth_count = int(counts[1].item())
    result = _base_metric(
        backend="raydn",
        workload="reflection_trace",
        scene="parallel_reflectors",
        input_count=ray_count,
        valid_path_count=slot_count,
        estimated_output_bytes=_reflection_trace_output_bytes(ray_count, max_bounces),
        timing=measured,
        build_ms=build_ms,
        ray_count=ray_count,
        max_bounces=max_bounces,
        valid_full_depth_path_count=full_depth_count,
        valid_bounce_slot_count=slot_count,
        path_length_checksum=float(checksum[0].item()),
    )
    _cleanup_torch()
    return result


def _torch_dfr_scene() -> tuple[rt.Scene, float]:
    vertices = torch.tensor(
        [[-1.0, -1.0, 10.0], [1.0, -1.0, 10.0], [-1.0, 1.0, 10.0]],
        device="cuda",
        dtype=torch.float32,
    )
    faces = torch.tensor([[0, 1, 2]], device="cuda", dtype=torch.int32)
    scene = rt.Scene()
    scene.add_mesh(rt.Mesh(vertices, faces))
    _, build_ms = _time_build(scene.build, torch.cuda.synchronize)
    return scene, build_ms


def _torch_dfr_states(state_count: int) -> rt.DfrStates:
    device = torch.device("cuda")
    edge_pos = torch.zeros((state_count, 3), device=device, dtype=torch.float32)
    edge_dir = torch.zeros_like(edge_pos)
    edge_dir[:, 0] = 1.0
    n0 = torch.zeros_like(edge_pos)
    n0[:, 1] = 1.0
    n1 = torch.zeros_like(edge_pos)
    n1[:, 1] = -1.0
    src = torch.zeros_like(edge_pos)
    src[:, 2] = 1.0
    wi = torch.zeros_like(edge_pos)
    wi[:, 2] = -1.0
    d0 = torch.zeros_like(edge_pos)
    d0[:, 2] = -1.0
    return rt.DfrStates(
        edge_index=torch.arange(state_count, device=device, dtype=torch.int32),
        edge_pos=edge_pos,
        edge_dir=edge_dir,
        edge_t_min=torch.full((state_count,), -0.5, device=device, dtype=torch.float32),
        edge_t_max=torch.full((state_count,), 0.5, device=device, dtype=torch.float32),
        n0=n0,
        n1=n1,
        prim0=torch.full((state_count,), -1, device=device, dtype=torch.int32),
        prim1=torch.full((state_count,), -1, device=device, dtype=torch.int32),
        exterior_angle=torch.full((state_count,), 1.5 * math.pi, device=device, dtype=torch.float32),
        src=src,
        src_power=torch.full((state_count,), 2.0, device=device, dtype=torch.float32),
        wi=wi,
        d0=d0,
        count=state_count,
    )


def _torch_dfr_material() -> rt.DfrMaterial:
    device = torch.device("cuda")
    return rt.DfrMaterial(
        eta_r=torch.tensor([4.0], device=device, dtype=torch.float32),
        sigma=torch.tensor([0.0], device=device, dtype=torch.float32),
        mu_r=torch.tensor([1.0], device=device, dtype=torch.float32),
        gain=torch.tensor([1.0], device=device, dtype=torch.float32),
        valid=torch.tensor([True], device=device, dtype=torch.bool),
    )


def run_raydn_diffraction_export(args: argparse.Namespace, state_count: int) -> dict[str, Any]:
    scene, build_ms = _torch_dfr_scene()
    states = _torch_dfr_states(state_count)
    material = _torch_dfr_material()
    tx = torch.tensor([[0.0, 0.0, 1.0]], device="cuda", dtype=torch.float32)
    rx = torch.tensor([[0.0, 0.0, -1.0]], device="cuda", dtype=torch.float32)

    def call_kernel():
        paths = scene.trace_dfr_paths(
            tx_positions=tx,
            rx_positions=rx,
            states=states,
            material=material,
            max_paths=state_count,
            wavelength=0.125,
        )
        return (
            paths.count,
            paths.valid,
            paths.rx_id,
            paths.edge0,
            paths.delay,
            paths.field_x_re,
            paths.field_x_im,
            paths.p0,
            paths.p1,
            paths.p2,
        )

    measured = _measure(call_kernel, lambda _value: None, torch.cuda.synchronize, args.repeats, args.warmup)
    count, valid, _rx_id, _edge0, delay, _field_x_re, _field_x_im, _p0, _p1, _p2 = measured.pop("last_value")
    valid_count_tensor, checksum = torch.ops.raydn.diffraction_path_stats(
        count.contiguous(),
        valid.contiguous(),
        delay.contiguous(),
    )
    valid_count = int(valid_count_tensor[0].item())
    result = _base_metric(
        backend="raydn",
        workload="diffraction_export",
        scene="synthetic_single_edge_state",
        input_count=state_count,
        valid_path_count=valid_count,
        estimated_output_bytes=_dfr_path_output_bytes(state_count),
        timing=measured,
        build_ms=build_ms,
        state_count=state_count,
        path_capacity=state_count,
        path_length_checksum=float(checksum[0].item()),
    )
    _cleanup_torch()
    return result


def _rayd_parallel_reflector_scene(rayd: Any, cuda: Any, dr: Any) -> tuple[Any, float]:
    vertices = cuda.Array3f(
        [-1.0, 1.0, 1.0, -1.0, -1.0, 1.0, 1.0, -1.0],
        [-1.0, -1.0, 1.0, 1.0, -1.0, -1.0, 1.0, 1.0],
        [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0],
    )
    faces = cuda.Array3i([0, 0, 4, 4], [1, 2, 5, 6], [2, 3, 6, 7])
    scene = rayd.Scene()
    scene.add_mesh(rayd.Mesh(vertices, faces))
    _, build_ms = _time_build(scene.build, dr.sync_thread)
    return scene, build_ms


def _rayd_reflection_ray(rayd: Any, cuda: Any, dr: Any, ray_count: int) -> Any:
    grid_side = int(math.ceil(math.sqrt(ray_count)))
    idx = dr.arange(cuda.UInt, ray_count)
    ix = idx % grid_side
    iy = idx // grid_side
    denom = max(1, grid_side - 1)
    x = -0.9 + 1.8 * cuda.Float(ix) / denom
    y = -0.9 + 1.8 * cuda.Float(iy) / denom
    origin = cuda.Array3f(x, y, dr.full(cuda.Float, 0.5, ray_count))
    direction = cuda.Array3f(
        dr.zeros(cuda.Float, ray_count),
        dr.zeros(cuda.Float, ray_count),
        dr.full(cuda.Float, 1.0, ray_count),
    )
    return rayd.Ray(origin, direction)


def run_rayd_reflection_trace(
    args: argparse.Namespace,
    rayd: Any,
    cuda: Any,
    dr: Any,
    ray_count: int,
    max_bounces: int,
) -> dict[str, Any]:
    scene, build_ms = _rayd_parallel_reflector_scene(rayd, cuda, dr)
    ray = _rayd_reflection_ray(rayd, cuda, dr, ray_count)
    options = rayd.ReflectionTraceOptions()
    options.export_mode = rayd.REFLECTION_EXPORT_MINIMAL
    options.return_trailing = False

    def call_kernel():
        if hasattr(rayd, "native_launch_audit_clear"):
            rayd.native_launch_audit_clear()
        return scene.trace_reflections(ray, max_bounces, options, True, False)

    def materialize(result: Any) -> None:
        dr.eval(result.bounce_count, result.t, result.prim_ids, result.global_prim_ids)

    measured = _measure(call_kernel, materialize, dr.sync_thread, args.repeats, args.warmup)
    trace = measured.pop("last_value")
    valid = trace.is_valid()
    slot_count = int(dr.sum(dr.select(valid, 1.0, 0.0))[0])
    full_depth_count = int(dr.sum(dr.select(trace.bounce_count == max_bounces, 1.0, 0.0))[0])
    checksum = float(dr.sum(dr.select(valid, trace.t, 0.0))[0])
    audit = rayd.native_launch_audit() if hasattr(rayd, "native_launch_audit") else {}
    return _base_metric(
        backend="rayd_path",
        workload="reflection_trace",
        scene="parallel_reflectors",
        input_count=ray_count,
        valid_path_count=slot_count,
        estimated_output_bytes=_reflection_trace_output_bytes(ray_count, max_bounces),
        timing=measured,
        build_ms=build_ms,
        ray_count=ray_count,
        max_bounces=max_bounces,
        valid_full_depth_path_count=full_depth_count,
        valid_bounce_slot_count=slot_count,
        path_length_checksum=checksum,
        native_audit=audit,
    )


def _rayd_dfr_states(cuda: Any, dr: Any, state_count: int) -> Any:
    import rayd as rd

    states = rd.DfrStates()
    states.count = state_count
    states.edge_index = dr.arange(cuda.Int, state_count)
    states.edge_pos = cuda.Array3f(
        dr.zeros(cuda.Float, state_count),
        dr.zeros(cuda.Float, state_count),
        dr.zeros(cuda.Float, state_count),
    )
    states.edge_dir = cuda.Array3f(
        dr.full(cuda.Float, 1.0, state_count),
        dr.zeros(cuda.Float, state_count),
        dr.zeros(cuda.Float, state_count),
    )
    states.edge_t_min = dr.full(cuda.Float, -0.5, state_count)
    states.edge_t_max = dr.full(cuda.Float, 0.5, state_count)
    states.n0 = cuda.Array3f(
        dr.zeros(cuda.Float, state_count),
        dr.full(cuda.Float, 1.0, state_count),
        dr.zeros(cuda.Float, state_count),
    )
    states.n1 = cuda.Array3f(
        dr.zeros(cuda.Float, state_count),
        dr.full(cuda.Float, -1.0, state_count),
        dr.zeros(cuda.Float, state_count),
    )
    states.prim0 = dr.full(cuda.Int, -1, state_count)
    states.prim1 = dr.full(cuda.Int, -1, state_count)
    states.exterior_angle = dr.full(cuda.Float, 1.5 * math.pi, state_count)
    states.src = cuda.Array3f(
        dr.zeros(cuda.Float, state_count),
        dr.zeros(cuda.Float, state_count),
        dr.full(cuda.Float, 1.0, state_count),
    )
    states.src_power = dr.full(cuda.Float, 2.0, state_count)
    states.wi = cuda.Array3f(
        dr.zeros(cuda.Float, state_count),
        dr.zeros(cuda.Float, state_count),
        dr.full(cuda.Float, -1.0, state_count),
    )
    states.d0 = cuda.Array3f(
        dr.zeros(cuda.Float, state_count),
        dr.zeros(cuda.Float, state_count),
        dr.full(cuda.Float, -1.0, state_count),
    )
    states.prefix_depth = dr.full(cuda.Int, 0, state_count)
    return states


def run_rayd_diffraction_export(
    args: argparse.Namespace,
    rayd: Any,
    cuda: Any,
    dr: Any,
    state_count: int,
) -> dict[str, Any]:
    scene = rayd.Scene()
    vertices = cuda.Array3f([-1.0, 1.0, -1.0], [-1.0, -1.0, 1.0], [10.0, 10.0, 10.0])
    scene.add_mesh(rayd.Mesh(vertices, cuda.Array3i([0], [1], [2])))
    _, build_ms = _time_build(scene.build, dr.sync_thread)
    states = _rayd_dfr_states(cuda, dr, state_count)

    material = rayd.DfrMaterial()
    material.eta_r = cuda.Float([4.0])
    material.sigma = cuda.Float([0.0])
    material.mu_r = cuda.Float([1.0])
    material.gain = cuda.Float([1.0])
    material.valid = cuda.Bool([True])

    options = rayd.DfrPathOptions()
    options.wavelength = 0.125
    options.k = 50.26548245743669
    options.seed = args.seed
    options.max_order = 1
    options.max_paths = state_count
    options.max_rx = 1
    options.strategy_mask = rayd.RAYD_DFR_DIRECT
    options.sample_count = 1
    options.return_geom = 1
    options.receiver_model = rayd.RAYD_DFR_MATCHED_ISO

    tx = cuda.Array3f([0.0], [0.0], [1.0])
    rx = cuda.Array3f([0.0], [0.0], [-1.0])
    active = cuda.Bool([True])

    def call_kernel():
        if hasattr(rayd, "native_launch_audit_clear"):
            rayd.native_launch_audit_clear()
        return scene.trace_dfr_paths(tx, rx, states, material, options, active)

    def materialize(result: Any) -> None:
        dr.eval(
            result.count,
            result.valid,
            result.rx_id,
            result.edge0,
            result.delay,
            result.field_x.real,
            result.field_x.imag,
            result.p0,
            result.p1,
            result.p2,
        )

    measured = _measure(call_kernel, materialize, dr.sync_thread, args.repeats, args.warmup)
    paths = measured.pop("last_value")
    valid_count = int(paths.count[0])
    checksum = float(dr.sum(dr.select(paths.valid, paths.delay, 0.0))[0])
    audit = rayd.native_launch_audit() if hasattr(rayd, "native_launch_audit") else {}
    return _base_metric(
        backend="rayd_path",
        workload="diffraction_export",
        scene="synthetic_single_edge_state",
        input_count=state_count,
        valid_path_count=valid_count,
        estimated_output_bytes=_dfr_path_output_bytes(int(paths.capacity)),
        timing=measured,
        build_ms=build_ms,
        state_count=state_count,
        path_capacity=int(paths.capacity),
        path_length_checksum=checksum,
        native_audit=audit,
    )


def _mitsuba_parallel_reflector_scene(mi: Any, dr: Any) -> tuple[Any, float]:
    transform = mi.ScalarTransform4f

    def build():
        return mi.load_dict(
            {
                "type": "scene",
                "lower": {
                    "type": "rectangle",
                    "to_world": transform.translate([0.0, 0.0, 0.0]),
                },
                "upper": {
                    "type": "rectangle",
                    "to_world": transform.translate([0.0, 0.0, 1.0]),
                },
            }
        )

    return _time_build(build, dr.sync_thread)


def _mitsuba_reflection_ray(mi: Any, dr: Any, ray_count: int) -> Any:
    grid_side = int(math.ceil(math.sqrt(ray_count)))
    idx = dr.arange(mi.UInt, ray_count)
    ix = idx % grid_side
    iy = idx // grid_side
    denom = max(1, grid_side - 1)
    x = -0.9 + 1.8 * mi.Float(ix) / denom
    y = -0.9 + 1.8 * mi.Float(iy) / denom
    origin = mi.Point3f(x, y, dr.full(mi.Float, 0.5, ray_count))
    direction = mi.Vector3f(
        dr.zeros(mi.Float, ray_count),
        dr.zeros(mi.Float, ray_count),
        dr.full(mi.Float, 1.0, ray_count),
    )
    return mi.Ray3f(origin, direction)


def run_mitsuba_reflection_trace(
    args: argparse.Namespace,
    mi: Any,
    dr: Any,
    ray_count: int,
    max_bounces: int,
) -> dict[str, Any]:
    scene, build_ms = _mitsuba_parallel_reflector_scene(mi, dr)
    ray = _mitsuba_reflection_ray(mi, dr, ray_count)

    def call_kernel():
        active = dr.full(mi.Bool, True, ray_count)
        origin = ray.o
        direction = ray.d
        ts = []
        prim_ids = []
        for _ in range(max_bounces):
            bounce_ray = mi.Ray3f(origin, direction)
            if args.mitsuba_ray_api == "preliminary":
                pi = scene.ray_intersect_preliminary(bounce_ray, coherent=True, active=active)
                hit = active & pi.is_valid()
                si = pi.compute_surface_interaction(bounce_ray, mi.RayFlags.Minimal, hit)
                hit_t = pi.t
                prim_index = pi.prim_index
                hit_point = origin + direction * pi.t
                normal = si.n
            else:
                si = scene.ray_intersect(
                    bounce_ray,
                    ray_flags=mi.RayFlags.Minimal,
                    coherent=True,
                    active=active,
                )
                hit = active & si.is_valid()
                hit_t = si.t
                prim_index = si.prim_index
                hit_point = si.p
                normal = si.n
            ts.append(dr.select(hit, hit_t, 0.0))
            prim_ids.append(dr.select(hit, prim_index, mi.UInt(0xFFFFFFFF)))
            direction = direction - 2.0 * dr.dot(direction, normal) * normal
            origin = hit_point + direction * 1e-4
            active &= hit
        valid_count = dr.sum(dr.select(active, 1.0, 0.0))
        slot_count = dr.sum(dr.select(dr.concat(ts) > 0.0, 1.0, 0.0))
        checksum = dr.sum(dr.concat(ts))
        return ts, prim_ids, valid_count, slot_count, checksum

    def materialize(value: Any) -> None:
        ts, prim_ids, valid_count, slot_count, checksum = value
        dr.eval(*ts, *prim_ids, valid_count, slot_count, checksum)

    measured = _measure(call_kernel, materialize, dr.sync_thread, args.repeats, args.warmup)
    _ts, _prim_ids, valid_count, slot_count, checksum = measured.pop("last_value")
    return _base_metric(
        backend="mitsuba_path",
        workload="reflection_trace",
        scene="parallel_reflectors",
        input_count=ray_count,
        valid_path_count=int(slot_count[0]),
        estimated_output_bytes=_reflection_trace_output_bytes(ray_count, max_bounces),
        timing=measured,
        build_ms=build_ms,
        ray_count=ray_count,
        max_bounces=max_bounces,
        mitsuba_ray_api=args.mitsuba_ray_api,
        valid_full_depth_path_count=int(valid_count[0]),
        valid_bounce_slot_count=int(slot_count[0]),
        path_length_checksum=float(checksum[0]),
    )


def _prepare_sionna_imports(sionna_root: Path) -> None:
    src = sionna_root / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))


def _mitsuba_scene_from_sionna(sionna_root: Path, workload: str) -> Any:
    _prepare_sionna_imports(sionna_root)
    import sionna.rt as sionna_rt
    from sionna.rt import load_scene

    scene_file = sionna_rt.scene.simple_reflector if workload in ("los", "reflection") else sionna_rt.scene.simple_wedge
    return load_scene(scene_file, merge_shapes=False).mi_scene


def _mi_repeat_point(mi: Any, pos: tuple[float, float, float], count: int) -> Any:
    return mi.Point3f([pos[0]] * count, [pos[1]] * count, [pos[2]] * count)


def _mi_segment_visible(mi: Any, dr: Any, scene: Any, start: Any, end: Any, eps: float = 1e-4) -> Any:
    delta = end - start
    dist = dr.norm(delta)
    direction = delta / dist
    ray = mi.Ray3f(start + direction * eps, direction)
    ray.maxt = dr.maximum(dist - 2.0 * eps, 0.0)
    active = dist > (2.0 * eps)
    return ~scene.ray_test(ray, active=active)


def run_mitsuba_diffraction_export(args: argparse.Namespace, mi: Any, dr: Any, state_count: int) -> dict[str, Any]:
    scene, build_ms = _time_build(
        lambda: _mitsuba_scene_from_sionna(args.sionna_root, "diffraction"),
        dr.sync_thread,
    )
    tx = _mi_repeat_point(mi, (0.0, 0.0, 1.0), state_count)
    rx = _mi_repeat_point(mi, (0.0, 0.0, -1.0), state_count)
    edge_pos = _mi_repeat_point(mi, (0.0, 0.0, 0.0), state_count)

    def call_kernel():
        visible0 = _mi_segment_visible(mi, dr, scene, tx, edge_pos)
        visible1 = _mi_segment_visible(mi, dr, scene, edge_pos, rx)
        valid = visible0 & visible1
        delay = (dr.norm(edge_pos - tx) + dr.norm(rx - edge_pos)) / SPEED_OF_LIGHT
        checksum = dr.sum(dr.select(valid, delay, 0.0))
        count = dr.sum(dr.select(valid, 1.0, 0.0))
        return valid, delay, edge_pos, count, checksum

    def materialize(value: Any) -> None:
        dr.eval(*value)

    measured = _measure(call_kernel, materialize, dr.sync_thread, args.repeats, args.warmup)
    _valid, _delay, _edge_pos, count, checksum = measured.pop("last_value")
    valid_count = int(count[0])
    return _base_metric(
        backend="mitsuba_path",
        workload="diffraction_export",
        scene="simple_wedge",
        input_count=state_count,
        valid_path_count=valid_count,
        estimated_output_bytes=_dfr_path_output_bytes(state_count),
        timing=measured,
        build_ms=build_ms,
        state_count=state_count,
        path_capacity=state_count,
        path_length_checksum=float(checksum[0]),
    )


def _run_backend_case(
    args: argparse.Namespace,
    backend: str,
    workload: str,
    *,
    ray_count: int | None,
    max_bounces: int | None,
    state_count: int | None,
    rayd_bundle: tuple[Any, Any, Any] | None,
    mitsuba_bundle: tuple[Any, Any] | None,
) -> dict[str, Any]:
    try:
        if backend == "raydn":
            if workload == "reflection_trace":
                assert ray_count is not None and max_bounces is not None
                return run_raydn_reflection_trace(args, ray_count, max_bounces)
            assert state_count is not None
            return run_raydn_diffraction_export(args, state_count)
        if backend == "rayd_path":
            if rayd_bundle is None:
                raise RuntimeError("RayD backend is not loaded.")
            rayd, cuda, dr = rayd_bundle
            if workload == "reflection_trace":
                assert ray_count is not None and max_bounces is not None
                return run_rayd_reflection_trace(args, rayd, cuda, dr, ray_count, max_bounces)
            assert state_count is not None
            return run_rayd_diffraction_export(args, rayd, cuda, dr, state_count)
        if backend == "mitsuba_path":
            if mitsuba_bundle is None:
                raise RuntimeError("Mitsuba backend is not loaded.")
            mi, dr = mitsuba_bundle
            if workload == "reflection_trace":
                assert ray_count is not None and max_bounces is not None
                return run_mitsuba_reflection_trace(args, mi, dr, ray_count, max_bounces)
            assert state_count is not None
            return run_mitsuba_diffraction_export(args, mi, dr, state_count)
    except Exception as exc:
        if args.fail_fast:
            raise
        return {
            "backend": backend,
            "workload": workload,
            "error": f"{type(exc).__name__}: {exc}",
        }
    raise ValueError(f"unsupported backend/workload pair: {backend}/{workload}")


def _case_label(case: dict[str, Any]) -> str:
    if case["workload"] == "reflection_trace":
        return f"{_format_count(case['ray_count'])} rays\n{case['max_bounces']} bounces"
    return f"{_format_count(case['state_count'])} states"


def _rows(results: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for case in results["cases"]:
        cfg = case["config"]
        for backend, result in case["backends"].items():
            if "error" in result:
                rows.append(
                    {
                        "backend": backend,
                        "workload": cfg["workload"],
                        "case_label": _case_label(cfg),
                        "error": result["error"],
                    }
                )
                continue
            timing = result["timing"]
            rows.append(
                {
                    "backend": backend,
                    "workload": cfg["workload"],
                    "case_label": _case_label(cfg),
                    "ray_count": cfg.get("ray_count"),
                    "max_bounces": cfg.get("max_bounces"),
                    "state_count": cfg.get("state_count"),
                    "input_count": result.get("input_count"),
                    "valid_path_count": result.get("valid_path_count"),
                    "build_ms": result.get("build_ms", 0.0),
                    "avg_ms": timing["avg_ms"],
                    "min_ms": timing["min_ms"],
                    "p50_ms": timing["p50_ms"],
                    "p95_ms": timing["p95_ms"],
                    "path_length_checksum": result.get("path_length_checksum"),
                    "estimated_output_mib": result.get("estimated_output_mib"),
                    "error": "",
                }
            )
    return rows


def _write_csv(path: Path, results: dict[str, Any]) -> None:
    rows = _rows(results)
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "backend",
        "workload",
        "case_label",
        "ray_count",
        "max_bounces",
        "state_count",
        "input_count",
        "valid_path_count",
        "build_ms",
        "avg_ms",
        "min_ms",
        "p50_ms",
        "p95_ms",
        "path_length_checksum",
        "estimated_output_mib",
        "error",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _plot_results(results: dict[str, Any], output_dir: Path) -> list[str]:
    import matplotlib.pyplot as plt

    rows = [row for row in _rows(results) if not row.get("error")]
    if not rows:
        return []
    output_dir.mkdir(parents=True, exist_ok=True)
    backend_order = ["raydn", "rayd_path", "mitsuba_path"]
    colors = {"raydn": "#2563eb", "rayd_path": "#16a34a", "mitsuba_path": "#c2410c"}
    backend_titles = {
        "raydn": "RayDN",
        "rayd_path": "RayD path",
        "mitsuba_path": "Mitsuba path",
    }
    workloads = [workload for workload in WORKLOADS if any(row["workload"] == workload for row in rows)]
    fig, axes = plt.subplots(
        len(workloads),
        1,
        figsize=(max(8.5, 1.2 * max(1, len(rows))), 4.8 * len(workloads)),
        squeeze=False,
    )
    titles = {
        "reflection_trace": "Reflection trace: parallel reflectors, public reduced path fields",
        "diffraction_export": "Diffraction path export: synthetic single-edge states",
    }
    for workload_index, workload in enumerate(workloads):
        ax = axes[workload_index][0]
        workload_rows = [row for row in rows if row["workload"] == workload]
        cases = []
        for row in workload_rows:
            label = row["case_label"]
            if label not in cases:
                cases.append(label)
        backends = [backend for backend in backend_order if any(row["backend"] == backend for row in workload_rows)]
        width = 0.78 / max(1, len(backends))
        for backend_index, backend in enumerate(backends):
            values = []
            for label in cases:
                match = next(
                    (row for row in workload_rows if row["backend"] == backend and row["case_label"] == label),
                    None,
                )
                values.append(float(match["avg_ms"]) if match else float("nan"))
            offsets = [
                case_index + (backend_index - (len(backends) - 1) / 2) * width
                for case_index in range(len(cases))
            ]
            ax.bar(offsets, values, width=width, label=backend_titles.get(backend, backend), color=colors.get(backend))
        ax.set_xticks(range(len(cases)))
        ax.set_xticklabels(cases, rotation=25, ha="right")
        ax.set_ylabel("Average time (ms)")
        ax.set_title(titles.get(workload, workload))
        ax.grid(True, axis="y", alpha=0.25)
        ax.legend()
    fig.suptitle("RayDN / RayD / Mitsuba multipath benchmark")
    fig.tight_layout()
    path = output_dir / "time_ms_multipath.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return [str(path)]


def _default_output_dir(preset: str) -> Path:
    return Path("artifacts") / "benchmarks" / "multipath" / preset


def _build_cases(args: argparse.Namespace) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    if "reflection_trace" in args.workloads:
        for ray_count in args.ray_counts:
            for max_bounces in args.max_bounces_values:
                cases.append(
                    {
                        "workload": "reflection_trace",
                        "ray_count": ray_count,
                        "max_bounces": max_bounces,
                    }
                )
    if "diffraction_export" in args.workloads:
        for state_count in args.state_counts:
            cases.append(
                {
                    "workload": "diffraction_export",
                    "state_count": state_count,
                }
            )
    return cases


def main() -> None:
    parser = argparse.ArgumentParser(
        description="RayD latest-style multipath benchmark with RayDN, RayD, and Mitsuba path backends."
    )
    parser.add_argument("--preset", choices=sorted(PRESETS), default="standard")
    parser.add_argument("--workloads", nargs="+", choices=WORKLOADS, default=list(WORKLOADS))
    parser.add_argument(
        "--backends",
        nargs="+",
        choices=["raydn", "rayd_path", "mitsuba_path"],
        default=["raydn", "rayd_path", "mitsuba_path"],
    )
    parser.add_argument("--ray-count", action="append", default=None)
    parser.add_argument("--state-count", action="append", default=None)
    parser.add_argument("--max-bounces", action="append", default=None)
    parser.add_argument("--repeats", type=int, default=None)
    parser.add_argument("--warmup", type=int, default=None)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--rayd-source", choices=("package", "local"), default="package")
    parser.add_argument("--rayd-root", type=Path, default=RAYDI_ROOT)
    parser.add_argument("--mitsuba-variant", default="cuda_ad_rgb")
    parser.add_argument("--mitsuba-ray-api", choices=("preliminary", "surface"), default="preliminary")
    parser.add_argument("--sionna-root", type=Path, default=DEFAULT_SIONNA_ROOT)
    parser.add_argument("--require-mitsuba", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--json-output", type=Path, default=None)
    parser.add_argument("--csv-output", type=Path, default=None)
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()

    preset = PRESETS[args.preset]
    args.ray_counts = _parse_int_list(args.ray_count, preset["ray_counts"])
    args.state_counts = _parse_int_list(args.state_count, preset["state_counts"])
    args.max_bounces_values = _parse_int_list(args.max_bounces, preset["max_bounces"])
    args.repeats = int(args.repeats if args.repeats is not None else preset["repeats"])
    args.warmup = int(args.warmup if args.warmup is not None else preset["warmup"])
    output_dir = args.output_dir or _default_output_dir(args.preset)
    json_output = args.json_output or (output_dir / "multipath.json")
    csv_output = args.csv_output or (output_dir / "multipath.csv")

    if "raydn" in args.backends and not torch.cuda.is_available():
        raise SystemExit("RayDN backend requires CUDA torch.")

    rayd_bundle: tuple[Any, Any, Any] | None = None
    if "rayd_path" in args.backends:
        rayd_bundle = _load_rayd(args.rayd_source, args.rayd_root)

    mitsuba_bundle: tuple[Any, Any] | None = None
    if "mitsuba_path" in args.backends:
        dr = rayd_bundle[2] if rayd_bundle is not None else importlib.import_module("drjit")
        mi = _try_import_mitsuba(args.mitsuba_variant)
        if mi is None:
            if args.require_mitsuba:
                raise RuntimeError("Mitsuba is not installed in the current environment.")
        else:
            mitsuba_bundle = (mi, dr)

    cases: list[dict[str, Any]] = []
    for case_cfg in _build_cases(args):
        backends: dict[str, Any] = {}
        for backend in args.backends:
            if backend == "mitsuba_path" and mitsuba_bundle is None:
                backends[backend] = {"backend": backend, "workload": case_cfg["workload"], "error": "Mitsuba is not installed."}
                continue
            result = _run_backend_case(
                args,
                backend,
                case_cfg["workload"],
                ray_count=case_cfg.get("ray_count"),
                max_bounces=case_cfg.get("max_bounces"),
                state_count=case_cfg.get("state_count"),
                rayd_bundle=rayd_bundle,
                mitsuba_bundle=mitsuba_bundle,
            )
            backends[backend] = result
            gc.collect()
        cases.append({"config": case_cfg, "backends": backends})

    if rayd_bundle is not None:
        _cleanup_drjit(rayd_bundle[2])
    elif mitsuba_bundle is not None:
        _cleanup_drjit(mitsuba_bundle[1])

    results = {
        "benchmark": "raydn_rayd_mitsuba_multipath",
        "suite_config": {
            "preset": args.preset,
            "workloads": args.workloads,
            "backends": args.backends,
            "ray_counts": args.ray_counts,
            "state_counts": args.state_counts,
            "max_bounces": args.max_bounces_values,
            "repeats": args.repeats,
            "warmup": args.warmup,
            "rayd_source": args.rayd_source,
            "rayd_root": str(args.rayd_root) if args.rayd_source == "local" else None,
            "mitsuba_variant": args.mitsuba_variant if "mitsuba_path" in args.backends else None,
            "mitsuba_ray_api": args.mitsuba_ray_api,
            "sionna_root": str(args.sionna_root),
            "plots": "Grouped bar charts show absolute average time in ms; no throughput plots are emitted.",
        },
        "cases": cases,
    }

    json_output.parent.mkdir(parents=True, exist_ok=True)
    json_output.write_text(json.dumps(results, indent=2, sort_keys=True), encoding="utf-8")
    _write_csv(csv_output, results)
    plot_outputs: list[str] = []
    if not args.no_plots:
        plot_outputs = _plot_results(results, output_dir)
    results["outputs"] = {"json": str(json_output), "csv": str(csv_output), "plots": plot_outputs}
    json_output.write_text(json.dumps(results, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
