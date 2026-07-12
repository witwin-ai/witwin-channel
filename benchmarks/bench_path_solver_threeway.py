from __future__ import annotations

import argparse
import cmath
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import time
import tracemalloc
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHANNEL_ROOT = REPO_ROOT.parent / "channel"
DEFAULT_SIONNA_ROOT = (
    DEFAULT_CHANNEL_ROOT / "reference" / "sionna-rt-reference-2.0.1" / "src"
)
DEFAULT_SCENE = "san_francisco"
DEFAULT_TX = ((0.0, 0.0, 180.0),)
DEFAULT_RX = ((250.0, 0.0, 180.0), (-250.0, 120.0, 180.0))
SCHEMA_NAME = "witwin.channel_native.path_solver_threeway"
SCHEMA_VERSION = "1.0.0"
COMPONENT_NAMES = {0: "los", 1: "reflection", 2: "diffraction"}


def _parse_points(text: str) -> tuple[tuple[float, float, float], ...]:
    points = []
    for item in text.split(";"):
        item = item.strip()
        if not item:
            continue
        values = tuple(float(v.strip()) for v in item.split(","))
        if len(values) != 3:
            raise ValueError(f"point must have three comma-separated values: {item!r}")
        points.append(values)
    if not points:
        raise ValueError("at least one point is required")
    return tuple(points)  # type: ignore[return-value]


def _parse_ints(text: str) -> tuple[int, ...]:
    values = tuple(dict.fromkeys(int(item.strip()) for item in text.split(",") if item.strip()))
    if not values:
        raise ValueError("at least one integer is required")
    return values


def _parse_floats(text: str) -> tuple[float, ...]:
    values = tuple(float(item.strip()) for item in text.split(",") if item.strip())
    if not values:
        raise ValueError("at least one float is required")
    return values


def _jsonable(value: Any) -> Any:
    try:
        import numpy as np
    except Exception:
        np = None
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if np is not None and isinstance(value, (np.floating, np.integer, np.bool_)):
        return value.item()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _median(values: list[float]) -> float | None:
    return float(statistics.median(values)) if values else None


def _mean(values: list[float]) -> float | None:
    return float(statistics.mean(values)) if values else None


def _scene_xml(args: argparse.Namespace) -> Path:
    return Path(args.sionna_source_root) / "sionna" / "rt" / "scenes" / args.scene / f"{args.scene}.xml"


def _time_repeated(
    operation, sync, *, warmup: int, repeats: int
) -> tuple[Any, float, list[float], dict[str, int]]:
    tracemalloc.start()
    tracemalloc.reset_peak()
    started = time.perf_counter()
    result = operation()
    sync(result)
    cold_ms = (time.perf_counter() - started) * 1000.0
    for _ in range(max(0, int(warmup))):
        result = operation()
        sync(result)
    times = []
    for _ in range(max(1, int(repeats))):
        started = time.perf_counter()
        result = operation()
        sync(result)
        times.append((time.perf_counter() - started) * 1000.0)
    current_bytes, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return result, cold_ms, times, {
        "host_traced_current_bytes": int(current_bytes),
        "host_traced_peak_bytes": int(peak_bytes),
    }


def _git_commit() -> str | None:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    return completed.stdout.strip() or None if completed.returncode == 0 else None


def _common_environment() -> dict[str, Any]:
    return {
        "commit_sha": _git_commit(),
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
    }


def _empty_component_stats(num_rx: int, num_tx: int) -> dict[str, Any]:
    return {
        "total": 0,
        "per_pair_counts": [[0 for _ in range(num_tx)] for _ in range(num_rx)],
        "finite_delay": True,
        "min_delay_s": None,
        "max_delay_s": None,
        "delays_by_pair": {
            f"{rx},{tx}": [] for rx in range(num_rx) for tx in range(num_tx)
        },
        "records_by_pair": {
            f"{rx},{tx}": [] for rx in range(num_rx) for tx in range(num_tx)
        },
    }


def _complex_pair(value: complex) -> list[float]:
    return [float(value.real), float(value.imag)]


def _complex_from_pair(value: list[float] | tuple[float, float]) -> complex:
    return complex(float(value[0]), float(value[1]))


def _component_signal_views(
    component_stats: dict[str, Any], frequency_offsets_hz: tuple[float, ...]
) -> None:
    for bucket in component_stats.values():
        bucket["cir_by_pair"] = {}
        bucket["cfr_by_pair"] = {}
        for pair, records in bucket["records_by_pair"].items():
            bucket["cir_by_pair"][pair] = [
                {
                    "delay_s": record["delay_s"],
                    "coefficient": record["coefficient"],
                }
                for record in records
            ]
            response = []
            for offset_hz in frequency_offsets_hz:
                value = sum(
                    _complex_from_pair(record["coefficient"])
                    * cmath.exp(-2.0j * math.pi * float(offset_hz) * record["delay_s"])
                    for record in records
                )
                response.append(_complex_pair(value))
            bucket["cfr_by_pair"][pair] = response


def _stats_from_labeled_paths(
    *,
    tau,
    valid,
    labels,
    num_rx: int,
    num_tx: int,
    records_by_index: dict[tuple[int, int, int], dict[str, Any]] | None = None,
    frequency_offsets_hz: tuple[float, ...] = (0.0,),
) -> dict[str, Any]:
    import numpy as np

    components = {
        "los": _empty_component_stats(num_rx, num_tx),
        "reflection": _empty_component_stats(num_rx, num_tx),
        "diffraction": _empty_component_stats(num_rx, num_tx),
    }
    for rx in range(num_rx):
        for tx in range(num_tx):
            for path_idx in range(tau.shape[2]):
                if not bool(valid[rx, tx, path_idx]):
                    continue
                component = str(labels[rx, tx, path_idx])
                if component not in components:
                    continue
                delay = float(tau[rx, tx, path_idx])
                bucket = components[component]
                bucket["total"] += 1
                bucket["per_pair_counts"][rx][tx] += 1
                bucket["delays_by_pair"][f"{rx},{tx}"].append(delay)
                record = (
                    records_by_index.get((rx, tx, path_idx))
                    if records_by_index is not None
                    else None
                )
                if record is not None:
                    bucket["records_by_pair"][f"{rx},{tx}"].append(record)
    for bucket in components.values():
        all_delays = [
            delay
            for delays in bucket["delays_by_pair"].values()
            for delay in delays
        ]
        all_delays.sort()
        for key in bucket["delays_by_pair"]:
            bucket["delays_by_pair"][key].sort()
        bucket["finite_delay"] = bool(np.isfinite(all_delays).all()) if all_delays else True
        bucket["min_delay_s"] = float(min(all_delays)) if all_delays else None
        bucket["max_delay_s"] = float(max(all_delays)) if all_delays else None
        for key in bucket["records_by_pair"]:
            bucket["records_by_pair"][key].sort(key=lambda item: item["delay_s"])
    _component_signal_views(components, frequency_offsets_hz)
    return components


def _native_case_stats(
    result,
    *,
    num_rx: int,
    num_tx: int,
    frequency_offsets_hz: tuple[float, ...],
) -> dict[str, Any]:
    import numpy as np

    tau = result.tau[:, 0, :, 0, :].detach().cpu().numpy().astype(np.float64)
    valid = result.valid[:, 0, :, 0, :].detach().cpu().numpy().astype(bool)
    labels = np.empty(tau.shape, dtype=object)
    labels[:] = ""
    coefficient = result.a[:, 0, :, 0, :, 0].detach().cpu().numpy()
    theta_t = result.theta_t[:, 0, :, 0, :].detach().cpu().numpy()
    phi_t = result.phi_t[:, 0, :, 0, :].detach().cpu().numpy()
    theta_r = result.theta_r[:, 0, :, 0, :].detach().cpu().numpy()
    phi_r = result.phi_r[:, 0, :, 0, :].detach().cpu().numpy()
    interaction_type = result.interaction_type[:, 0, :, 0, :, :].detach().cpu().numpy()
    interaction_position = result.position[:, 0, :, 0, :, :, :].detach().cpu().numpy()
    records_by_index: dict[tuple[int, int, int], dict[str, Any]] = {}
    for rx in range(num_rx):
        for tx in range(num_tx):
            for slot in range(tau.shape[2]):
                if not valid[rx, tx, slot]:
                    continue
                depth_types = [
                    int(value)
                    for value in interaction_type[rx, tx, slot]
                    if int(value) > 0
                ]
                labels[rx, tx, slot] = (
                    "diffraction"
                    if 2 in depth_types
                    else "reflection"
                    if 1 in depth_types
                    else "los"
                )
                depth_positions = [
                    [float(v) for v in interaction_position[rx, tx, slot, depth_idx]]
                    for depth_idx in range(len(depth_types))
                ]
                value = complex(coefficient[rx, tx, slot])
                records_by_index[(rx, tx, slot)] = {
                    "delay_s": float(tau[rx, tx, slot]),
                    "coefficient": _complex_pair(value),
                    "angles_rad": {
                        "theta_t": float(theta_t[rx, tx, slot]),
                        "phi_t": float(phi_t[rx, tx, slot]),
                        "theta_r": float(theta_r[rx, tx, slot]),
                        "phi_r": float(phi_r[rx, tx, slot]),
                    },
                    "interaction_types": depth_types,
                    "interaction_positions_m": depth_positions,
                }
    return _stats_from_labeled_paths(
        tau=tau,
        valid=valid,
        labels=labels,
        num_rx=num_rx,
        num_tx=num_tx,
        records_by_index=records_by_index,
        frequency_offsets_hz=frequency_offsets_hz,
    )


def _run_native(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    sys.path.insert(0, str(REPO_ROOT / "src"))
    sys.path.insert(0, str(REPO_ROOT))
    from witwin.channel_native import ReceiverPoint, Scene, Transmitter, build_info
    from witwin.channel_native.core.edge_policy import EdgePolicy
    from witwin.channel_native.path import Config, solve

    tx_points = _parse_points(args.tx)
    rx_points = _parse_points(args.rx)
    xml = _scene_xml(args)
    started = time.perf_counter()
    base_scene = Scene.load_mitsuba(
        xml,
        source_root=Path(args.sionna_source_root),
        merge_shapes=True,
        frequency=float(args.frequency_hz),
        edge_selection_mode="all_edges",
        boundary_edge_policy="half_plane",
    )
    load_ms = (time.perf_counter() - started) * 1000.0
    scene = Scene(
        structures=base_scene.structures,
        transmitters=[
            Transmitter(position=torch.tensor(point, dtype=torch.float32), power_w=1.0)
            for point in tx_points
        ],
        receivers=[
            ReceiverPoint(position=torch.tensor(point, dtype=torch.float32))
            for point in rx_points
        ],
        frequency=base_scene.frequency,
        metadata=base_scene.metadata,
    )
    policy = EdgePolicy(edge_selection_mode="all_edges", boundary_edge_policy="half_plane")
    scene_stats = {
        "structures": len(scene.structures),
        "triangles": sum(int(s.faces.shape[0]) for s in scene.structures),
        "diffraction_edges": scene.diffraction_edge_count(policy),
    }

    configs = {
        "los": Config(components={"los"}),
        "reflection": Config(components={"reflection"}),
        "diffraction": Config(components={"diffraction"}),
        "all": Config(components={"los", "reflection", "diffraction"}),
    }

    def sync(result) -> None:
        for tensor in (
            result.valid,
            result.a,
            result.tau,
            result.interaction_type,
        ):
            _ = tensor.numel()
        torch.cuda.synchronize()

    cases = {}
    frequency_offsets_hz = _parse_floats(args.cfr_offsets_hz)
    for name, config in configs.items():
        torch.cuda.reset_peak_memory_stats()
        allocated_before = int(torch.cuda.memory_allocated())
        result, cold_ms, times, memory = _time_repeated(
            lambda config=config: solve(scene, config),
            sync,
            warmup=args.warmup,
            repeats=args.repeats,
        )
        memory.update(
            {
                "gpu_allocated_before_bytes": allocated_before,
                "gpu_peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
                "gpu_peak_delta_bytes": max(
                    0, int(torch.cuda.max_memory_allocated()) - allocated_before
                ),
            }
        )
        component_stats = _native_case_stats(
            result,
            num_rx=len(rx_points),
            num_tx=len(tx_points),
            frequency_offsets_hz=frequency_offsets_hz,
        )
        cases[name] = {
            "cold_ms": cold_ms,
            "steady_times_ms": times,
            "steady_median_ms": _median(times),
            "steady_mean_ms": _mean(times),
            "memory": memory,
            "path_count": int(result.valid.sum().item()),
            "component_counts": {k: v["total"] for k, v in component_stats.items()},
            "component_stats": component_stats,
            "metadata": result.metadata,
        }
    return {
        "provider": "channel_native",
        "build_info": build_info(),
        "scene": scene_stats,
        "load_ms": load_ms,
        "cases": cases,
        "environment": {
            **_common_environment(),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
    }


def _sionna_labels(paths, *, force_los: bool = False):
    import numpy as np

    tau = np.asarray(paths.tau, dtype=np.float64)
    valid = np.asarray(paths.valid, dtype=bool)
    labels = np.empty(valid.shape, dtype=object)
    labels[:] = ""
    if force_los or not hasattr(paths, "interactions"):
        labels[valid] = "los"
        return tau, valid, labels
    interactions = np.asarray(paths.interactions)
    if interactions.ndim == 4:
        per_path = np.moveaxis(interactions, 0, -1)
    else:
        per_path = interactions[..., None]
    diff = np.any(per_path == 8, axis=-1)
    refl = np.any(per_path == 1, axis=-1)
    labels[valid & diff] = "diffraction"
    labels[valid & ~diff & refl] = "reflection"
    labels[valid & ~diff & ~refl] = "los"
    return tau, valid, labels


def _dense_path_records(
    *,
    tau,
    valid,
    coefficient,
    angles: dict[str, Any],
    interaction_types,
    interaction_positions,
) -> dict[tuple[int, int, int], dict[str, Any]]:
    import numpy as np

    coeff = np.asarray(coefficient)
    while coeff.ndim > valid.ndim:
        coeff = coeff[..., 0]
    normalized_angles = {}
    for name, values in angles.items():
        array = np.asarray(values)
        while array.ndim > valid.ndim:
            array = array[..., 0]
        normalized_angles[name] = array
    records = {}
    for rx in range(valid.shape[0]):
        for tx in range(valid.shape[1]):
            for path_idx in range(valid.shape[2]):
                if not bool(valid[rx, tx, path_idx]):
                    continue
                types = [
                    int(value)
                    for value in interaction_types[rx, tx, path_idx]
                    if int(value) > 0
                ]
                positions = []
                if interaction_positions is not None:
                    positions = [
                        [float(v) for v in interaction_positions[rx, tx, path_idx, depth]]
                        for depth in range(min(len(types), interaction_positions.shape[-2]))
                    ]
                records[(rx, tx, path_idx)] = {
                    "delay_s": float(tau[rx, tx, path_idx]),
                    "coefficient": _complex_pair(complex(coeff[rx, tx, path_idx])),
                    "angles_rad": {
                        name: float(values[rx, tx, path_idx])
                        for name, values in normalized_angles.items()
                    },
                    "interaction_types": types,
                    "interaction_positions_m": positions,
                }
    return records


def _run_sionna(args: argparse.Namespace) -> dict[str, Any]:
    import inspect

    if not hasattr(inspect, "getargspec"):
        inspect.getargspec = inspect.getfullargspec  # type: ignore[attr-defined]
    sys.path.insert(0, str(Path(args.sionna_source_root)))
    import drjit as dr
    import mitsuba as mi
    import numpy as np
    import sionna.rt as rt

    tx_points = _parse_points(args.tx)
    rx_points = _parse_points(args.rx)
    xml = _scene_xml(args)
    started = time.perf_counter()
    scene = rt.load_scene(str(xml), merge_shapes=True)
    scene.frequency = float(args.frequency_hz)
    scene.tx_array = rt.PlanarArray(num_rows=1, num_cols=1, pattern="iso", polarization="V")
    scene.rx_array = rt.PlanarArray(num_rows=1, num_cols=1, pattern="iso", polarization="V")
    for index, point in enumerate(tx_points):
        scene.add(rt.Transmitter(f"tx{index}", position=mi.Point3f(*point)))
    for index, point in enumerate(rx_points):
        scene.add(rt.Receiver(f"rx{index}", position=mi.Point3f(*point)))
    load_ms = (time.perf_counter() - started) * 1000.0
    solver = rt.PathSolver()
    configs = {
        "los": dict(
            max_depth=0,
            los=True,
            specular_reflection=False,
            diffraction=False,
            edge_diffraction=False,
            samples_per_src=1,
        ),
        "reflection": dict(
            max_depth=1,
            los=False,
            specular_reflection=True,
            diffraction=False,
            edge_diffraction=False,
            samples_per_src=int(args.samples),
        ),
        "diffraction": dict(
            max_depth=1,
            los=False,
            specular_reflection=False,
            diffraction=True,
            edge_diffraction=True,
            samples_per_src=int(args.samples),
        ),
        "all": dict(
            max_depth=1,
            los=True,
            specular_reflection=True,
            diffraction=True,
            edge_diffraction=True,
            samples_per_src=int(args.samples),
        ),
    }

    def run(config):
        return solver(
            scene,
            synthetic_array=True,
            diffuse_reflection=False,
            refraction=False,
            seed=int(args.seed),
            **config,
        )

    def sync(result) -> None:
        dr.eval(result)
        dr.sync_thread()

    cases = {}
    frequency_offsets_hz = _parse_floats(args.cfr_offsets_hz)
    for name, config in configs.items():
        result, cold_ms, times, memory = _time_repeated(
            lambda config=config: run(config),
            sync,
            warmup=args.warmup,
            repeats=args.repeats,
        )
        tau, valid, labels = _sionna_labels(result, force_los=(name == "los"))
        interactions = np.asarray(getattr(result, "interactions", np.zeros((0, *valid.shape))))
        interaction_types = (
            np.moveaxis(interactions, 0, -1)
            if interactions.ndim == valid.ndim + 1
            else interactions[..., None]
        )
        interaction_types = np.where(interaction_types == 8, 2, interaction_types)
        vertices = getattr(result, "vertices", None)
        interaction_positions = None
        if vertices is not None:
            vertices = np.asarray(vertices)
            if vertices.ndim == valid.ndim + 2 and vertices.shape[-1] == 3:
                interaction_positions = np.moveaxis(vertices, 0, -2)
        records = _dense_path_records(
            tau=tau,
            valid=valid,
            coefficient=np.asarray(result.a),
            angles={
                key: np.asarray(getattr(result, key))
                for key in ("theta_t", "phi_t", "theta_r", "phi_r")
            },
            interaction_types=interaction_types,
            interaction_positions=interaction_positions,
        )
        component_stats = _stats_from_labeled_paths(
            tau=tau,
            valid=valid,
            labels=labels,
            num_rx=len(rx_points),
            num_tx=len(tx_points),
            records_by_index=records,
            frequency_offsets_hz=frequency_offsets_hz,
        )
        cases[name] = {
            "cold_ms": cold_ms,
            "steady_times_ms": times,
            "steady_median_ms": _median(times),
            "steady_mean_ms": _mean(times),
            "memory": memory,
            "path_shape": list(valid.shape),
            "valid_paths": int(np.count_nonzero(valid)),
            "component_counts": {k: v["total"] for k, v in component_stats.items()},
            "component_stats": component_stats,
        }
    return {
        "provider": "sionna",
        "load_ms": load_ms,
        "cases": cases,
        "environment": {
            **_common_environment(),
            "sionna_file": getattr(sys.modules.get("sionna"), "__file__", None),
            "drjit_version": dr.__version__,
        },
    }


def _original_case_config(args: argparse.Namespace, name: str):
    import witwin.channel as wc

    if name == "los":
        return wc.path.Config(
            num_samples=1,
            max_bounces=0,
            max_diffraction_order=0,
            max_num_paths=int(args.max_num_paths),
            return_geometry=True,
        )
    if name == "reflection":
        return wc.path.Config(
            num_samples=int(args.samples),
            max_bounces=1,
            max_diffraction_order=0,
            max_num_paths=int(args.max_num_paths),
            return_geometry=True,
            edge_policy=wc.EdgePolicy(edge_selection_mode="all_edges"),
        )
    tuning = wc.path.Tuning(
        diffraction_state_budget=int(args.diffraction_state_budget),
        inserted_reflection_state_budget=int(args.inserted_reflection_state_budget),
    )
    edge_policy = wc.EdgePolicy(
        edge_selection_mode="all_edges",
        edge_diffraction=True,
        boundary_edge_policy="half_plane",
    )
    return wc.path.Config(
        num_samples=int(args.samples),
        max_bounces=0 if name == "diffraction" else 1,
        max_diffraction_order=1,
        max_num_paths=int(args.max_num_paths),
        return_geometry=True,
        edge_policy=edge_policy,
        tuning=tuning,
    )


def _original_labels(result):
    import numpy as np

    tau = np.asarray(result.tau, dtype=np.float64)[:, 0, :, 0, :]
    valid = np.asarray(result.valid, dtype=bool)[:, 0, :, 0, :]
    types = np.asarray(result.types)[:, 0, :, 0, :, :]
    diff = np.any(types == 2, axis=-1)
    refl = np.any(types == 1, axis=-1)
    labels = np.empty(valid.shape, dtype=object)
    labels[:] = ""
    labels[valid & diff] = "diffraction"
    labels[valid & ~diff & refl] = "reflection"
    labels[valid & ~diff & ~refl] = "los"
    return tau, valid, labels


def _run_original(args: argparse.Namespace) -> dict[str, Any]:
    import inspect

    if not hasattr(inspect, "getargspec"):
        inspect.getargspec = inspect.getfullargspec  # type: ignore[attr-defined]
    os.environ["WITWIN_DETERMINISTIC_NATIVE_PROBE"] = "1"
    sys.path.insert(0, str(Path(args.channel_root)))
    sys.path.insert(0, str(Path(args.sionna_source_root)))
    import drjit as dr
    import numpy as np
    import witwin.channel as wc

    tx_points = _parse_points(args.tx)
    rx_points = _parse_points(args.rx)
    xml = _scene_xml(args)
    started = time.perf_counter()
    scene = wc.Scene.load_mitsuba(
        xml,
        source_root=Path(args.sionna_source_root),
        frequency=float(args.frequency_hz),
        merge_shapes=True,
        device="cuda",
    )
    for index, point in enumerate(tx_points):
        scene.add(wc.Transmitter(f"tx{index}", point))
    for index, point in enumerate(rx_points):
        scene.add(wc.Receiver(f"rx{index}", point))
    load_ms = (time.perf_counter() - started) * 1000.0

    def sync(result) -> None:
        dr.eval(result.a, result.tau, result.valid, result.num_paths, result.types)
        dr.sync_thread()

    cases = {}
    frequency_offsets_hz = _parse_floats(args.cfr_offsets_hz)
    for name in ("los", "reflection", "diffraction", "all"):
        config = _original_case_config(args, name)
        result, cold_ms, times, memory = _time_repeated(
            lambda config=config: wc.path.solve(
                scene=scene,
                transmitter=[f"tx{i}" for i in range(len(tx_points))],
                receiver=[f"rx{i}" for i in range(len(rx_points))],
                config=config,
            ),
            sync,
            warmup=args.warmup,
            repeats=args.repeats,
        )
        tau, valid, labels = _original_labels(result)
        interaction_types = np.asarray(result.types)[:, 0, :, 0, :, :]
        vertices = getattr(result, "vertices", None)
        interaction_positions = None
        if vertices is not None:
            vertices = np.asarray(vertices)
            if vertices.ndim == 7:
                interaction_positions = vertices[:, 0, :, 0, :, :, :]
        records = _dense_path_records(
            tau=tau,
            valid=valid,
            coefficient=np.asarray(result.a)[:, 0, :, 0, ...],
            angles={
                key: np.asarray(getattr(result, key))[:, 0, :, 0, :]
                for key in ("theta_t", "phi_t", "theta_r", "phi_r")
            },
            interaction_types=interaction_types,
            interaction_positions=interaction_positions,
        )
        component_stats = _stats_from_labeled_paths(
            tau=tau,
            valid=valid,
            labels=labels,
            num_rx=len(rx_points),
            num_tx=len(tx_points),
            records_by_index=records,
            frequency_offsets_hz=frequency_offsets_hz,
        )
        cases[name] = {
            "cold_ms": cold_ms,
            "steady_times_ms": times,
            "steady_median_ms": _median(times),
            "steady_mean_ms": _mean(times),
            "memory": memory,
            "path_shape": list(valid.shape),
            "valid_paths": int(np.count_nonzero(valid)),
            "component_counts": {k: v["total"] for k, v in component_stats.items()},
            "component_stats": component_stats,
            "metadata": {
                "path_counts": result.metadata.get("path_counts", {}),
                "runtime_backends": result.metadata.get("runtime_backends", {}),
                "timing": result.metadata.get("timing", {}),
            },
        }
    return {
        "provider": "original_channel",
        "load_ms": load_ms,
        "cases": cases,
        "environment": {
            **_common_environment(),
            "drjit_version": dr.__version__,
        },
    }


def _nearest_deltas(reference: list[float], candidate: list[float]) -> list[float]:
    if not reference or not candidate:
        return [float("inf") for _ in reference]
    sorted_candidate = sorted(float(v) for v in candidate)
    deltas = []
    for value in reference:
        value = float(value)
        lo = 0
        hi = len(sorted_candidate)
        while lo < hi:
            mid = (lo + hi) // 2
            if sorted_candidate[mid] < value:
                lo = mid + 1
            else:
                hi = mid
        options = []
        if lo < len(sorted_candidate):
            options.append(abs(sorted_candidate[lo] - value))
        if lo > 0:
            options.append(abs(sorted_candidate[lo - 1] - value))
        deltas.append(min(options) if options else float("inf"))
    return deltas


def _wrap_delta(value: float) -> float:
    return abs((float(value) + math.pi) % (2.0 * math.pi) - math.pi)


def _match_records(
    reference: list[dict[str, Any]],
    candidate: list[dict[str, Any]],
    tau_tol_s: float,
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    available = set(range(len(candidate)))
    matches = []
    for reference_record in sorted(reference, key=lambda item: item["delay_s"]):
        if not available:
            break
        index = min(
            available,
            key=lambda item: abs(candidate[item]["delay_s"] - reference_record["delay_s"]),
        )
        if abs(candidate[index]["delay_s"] - reference_record["delay_s"]) <= tau_tol_s:
            matches.append((reference_record, candidate[index]))
            available.remove(index)
    return matches


def _matched_path_metrics(
    matches: list[tuple[dict[str, Any], dict[str, Any]]]
) -> dict[str, Any]:
    magnitude_errors_db = []
    phase_errors_rad = []
    angle_errors_rad = []
    geometry_errors_m = []
    interaction_types_equal = 0
    finite = True
    for reference, candidate in matches:
        ref_coeff = _complex_from_pair(reference["coefficient"])
        candidate_coeff = _complex_from_pair(candidate["coefficient"])
        values = (ref_coeff.real, ref_coeff.imag, candidate_coeff.real, candidate_coeff.imag)
        finite = finite and all(math.isfinite(value) for value in values)
        if abs(ref_coeff) > 0.0 and abs(candidate_coeff) > 0.0:
            magnitude_errors_db.append(abs(20.0 * math.log10(abs(candidate_coeff) / abs(ref_coeff))))
            phase_errors_rad.append(_wrap_delta(cmath.phase(candidate_coeff) - cmath.phase(ref_coeff)))
        ref_angles = reference.get("angles_rad", {})
        candidate_angles = candidate.get("angles_rad", {})
        if set(ref_angles) == set(candidate_angles):
            angle_errors_rad.extend(
                _wrap_delta(candidate_angles[name] - ref_angles[name])
                for name in ref_angles
            )
        ref_types = reference.get("interaction_types", [])
        candidate_types = candidate.get("interaction_types", [])
        interaction_types_equal += int(ref_types == candidate_types)
        ref_positions = reference.get("interaction_positions_m", [])
        candidate_positions = candidate.get("interaction_positions_m", [])
        if len(ref_positions) == len(candidate_positions):
            for ref_position, candidate_position in zip(ref_positions, candidate_positions):
                geometry_errors_m.append(
                    math.sqrt(
                        sum((float(a) - float(b)) ** 2 for a, b in zip(ref_position, candidate_position))
                    )
                )
    return {
        "matched_paths": len(matches),
        "finite": finite,
        "median_magnitude_error_db": _median(magnitude_errors_db),
        "max_wrapped_phase_error_rad": max(phase_errors_rad) if phase_errors_rad else None,
        "max_angle_error_rad": max(angle_errors_rad) if angle_errors_rad else None,
        "interaction_types_equal": interaction_types_equal,
        "max_interaction_geometry_error_m": max(geometry_errors_m) if geometry_errors_m else None,
    }


def _cfr_metrics(native_stats: dict[str, Any], reference_stats: dict[str, Any]) -> dict[str, Any]:
    magnitude_errors_db = []
    phase_errors_rad = []
    absolute_errors = []
    finite = True
    count = 0
    for pair, reference_values in reference_stats["cfr_by_pair"].items():
        native_values = native_stats["cfr_by_pair"].get(pair, [])
        for reference_pair, native_pair in zip(reference_values, native_values):
            reference = _complex_from_pair(reference_pair)
            native = _complex_from_pair(native_pair)
            finite = finite and all(
                math.isfinite(value)
                for value in (reference.real, reference.imag, native.real, native.imag)
            )
            absolute_errors.append(abs(native - reference))
            if abs(reference) > 0.0 and abs(native) > 0.0:
                magnitude_errors_db.append(abs(20.0 * math.log10(abs(native) / abs(reference))))
                phase_errors_rad.append(_wrap_delta(cmath.phase(native) - cmath.phase(reference)))
            count += 1
    return {
        "samples": count,
        "finite": finite,
        "max_absolute_error": max(absolute_errors) if absolute_errors else None,
        "median_magnitude_error_db": _median(magnitude_errors_db),
        "max_wrapped_phase_error_rad": max(phase_errors_rad) if phase_errors_rad else None,
    }


def _component_delay_comparison(
    native: dict[str, Any],
    other: dict[str, Any],
    *,
    component: str,
    case: str,
    tau_tol_s: float,
    exact_counts: bool,
    magnitude_tol_db: float = 0.25,
    phase_tol_rad: float = 1.0e-3,
    angle_tol_rad: float = 1.0e-3,
    geometry_tol_m: float = 1.0e-3,
) -> dict[str, Any]:
    native_stats = native["cases"][case]["component_stats"][component]
    other_stats = other["cases"][case]["component_stats"][component]
    pair_reports = {}
    all_deltas = []
    count_mismatch_pairs = []
    covered = 0
    reference_total = 0
    record_matches = []
    for pair, reference_delays in other_stats["delays_by_pair"].items():
        native_delays = native_stats["delays_by_pair"].get(pair, [])
        reference_total += len(reference_delays)
        if len(reference_delays) != len(native_delays):
            count_mismatch_pairs.append(pair)
        if exact_counts and len(reference_delays) == len(native_delays):
            deltas = [abs(a - b) for a, b in zip(sorted(reference_delays), sorted(native_delays))]
        else:
            deltas = _nearest_deltas(reference_delays, native_delays)
        covered += sum(1 for delta in deltas if delta <= tau_tol_s)
        all_deltas.extend(deltas)
        record_matches.extend(
            _match_records(
                other_stats["records_by_pair"].get(pair, []),
                native_stats["records_by_pair"].get(pair, []),
                tau_tol_s,
            )
        )
        pair_reports[pair] = {
            "native_count": len(native_delays),
            "reference_count": len(reference_delays),
            "max_abs_delay_delta_s": max(deltas) if deltas else 0.0,
            "within_tolerance": sum(1 for delta in deltas if delta <= tau_tol_s),
        }
    finite_deltas = [d for d in all_deltas if d != float("inf")]
    path_metrics = _matched_path_metrics(record_matches)
    cfr_metrics = _cfr_metrics(native_stats, other_stats)
    coefficient_passed = (
        reference_total == 0
        or (
            path_metrics["matched_paths"] == reference_total
            and path_metrics["median_magnitude_error_db"] is not None
            and path_metrics["median_magnitude_error_db"] < magnitude_tol_db
            and path_metrics["max_wrapped_phase_error_rad"] is not None
            and path_metrics["max_wrapped_phase_error_rad"] <= phase_tol_rad
        )
    )
    angle_passed = (
        reference_total == 0
        or (
            path_metrics["max_angle_error_rad"] is not None
            and path_metrics["max_angle_error_rad"] <= angle_tol_rad
        )
    )
    interaction_passed = (
        reference_total == 0
        or (
            path_metrics["interaction_types_equal"] == reference_total
            and (
                path_metrics["max_interaction_geometry_error_m"] is None
                or path_metrics["max_interaction_geometry_error_m"] <= geometry_tol_m
            )
        )
    )
    delay_passed = (not exact_counts or not count_mismatch_pairs) and covered == reference_total
    return {
        "component": component,
        "case": case,
        "reference_provider": other["provider"],
        "native_total": native_stats["total"],
        "reference_total": other_stats["total"],
        "counts_equal": not count_mismatch_pairs,
        "count_mismatch_pairs": count_mismatch_pairs,
        "reference_delays_covered": covered,
        "reference_delay_count": reference_total,
        "coverage_ratio": (covered / reference_total) if reference_total else 1.0,
        "max_abs_delay_delta_s": max(all_deltas) if all_deltas else 0.0,
        "median_abs_delay_delta_s": (
            float(statistics.median(finite_deltas)) if finite_deltas else None
        ),
        "tau_tol_s": tau_tol_s,
        "comparison_mode": "exact_count" if exact_counts else "coverage",
        "path_metrics": path_metrics,
        "signal_views": {
            "cir": path_metrics,
            "cfr": cfr_metrics,
        },
        "gates": {
            "delay": delay_passed,
            "coefficient": coefficient_passed,
            "angles": angle_passed,
            "interactions": interaction_passed,
            "finite": path_metrics["finite"],
        },
        "passed": delay_passed and coefficient_passed and angle_passed and interaction_passed and path_metrics["finite"],
        "pairs": pair_reports,
    }


def _strip_delays(payload: Any) -> Any:
    if isinstance(payload, dict):
        return {
            key: ([] if key == "delays_by_pair" else _strip_delays(value))
            for key, value in payload.items()
        }
    if isinstance(payload, list):
        return [_strip_delays(value) for value in payload]
    return payload


def _speed_summary(providers: dict[str, dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    native = providers.get("channel_native")
    if native is None:
        return summary
    for case in ("los", "reflection", "diffraction", "all"):
        native_ms = native["cases"][case]["steady_median_ms"]
        row = {"channel_native_ms": native_ms}
        for provider_name in ("original_channel", "sionna"):
            provider = providers.get(provider_name)
            if provider is None:
                continue
            other_ms = provider["cases"][case]["steady_median_ms"]
            row[f"{provider_name}_ms"] = other_ms
            row[f"native_speedup_vs_{provider_name}"] = (
                None if not native_ms or native_ms <= 0 else other_ms / native_ms
            )
        summary[case] = row
    return summary


def _run_provider_subprocess(
    args: argparse.Namespace, provider: str, *, seed: int | None = None
) -> dict[str, Any]:
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--provider",
        provider,
        "--scene",
        args.scene,
        "--sionna-source-root",
        str(Path(args.sionna_source_root)),
        "--channel-root",
        str(Path(args.channel_root)),
        "--frequency-hz",
        str(args.frequency_hz),
        "--tx",
        args.tx,
        "--rx",
        args.rx,
        "--samples",
        str(args.samples),
        "--max-num-paths",
        str(args.max_num_paths),
        "--diffraction-state-budget",
        str(args.diffraction_state_budget),
        "--inserted-reflection-state-budget",
        str(args.inserted_reflection_state_budget),
        "--warmup",
        str(args.warmup),
        "--repeats",
        str(args.repeats),
        "--seed",
        str(args.seed if seed is None else seed),
        f"--cfr-offsets-hz={args.cfr_offsets_hz}",
    ]
    env = dict(os.environ)
    completed = subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        return {
            "provider": provider,
            "ok": False,
            "returncode": completed.returncode,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
        }
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        return {
            "provider": provider,
            "ok": False,
            "error": f"JSONDecodeError: {exc}",
            "stdout": completed.stdout,
            "stderr": completed.stderr,
        }
    payload["ok"] = True
    if completed.stderr.strip():
        payload["stderr"] = completed.stderr.strip()
    return payload


def _confidence_interval(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"count": 0, "mean": None, "lower_95": None, "upper_95": None}
    mean = float(statistics.mean(values))
    if len(values) == 1:
        return {"count": 1, "mean": mean, "lower_95": mean, "upper_95": mean}
    half_width = 1.96 * statistics.stdev(values) / math.sqrt(len(values))
    return {
        "count": len(values),
        "mean": mean,
        "lower_95": max(0.0, mean - half_width),
        "upper_95": min(1.0, mean + half_width),
    }


def _diffraction_seed_summary(
    seed_payloads: list[tuple[int, list[dict[str, Any]]]],
    args: argparse.Namespace,
) -> dict[str, Any]:
    by_provider: dict[str, list[dict[str, Any]]] = {}
    for seed, payloads in seed_payloads:
        providers = {
            payload.get("provider", "unknown"): payload
            for payload in payloads
            if payload.get("ok")
        }
        native = providers.get("channel_native")
        if native is None:
            continue
        for provider_name in ("original_channel", "sionna"):
            provider = providers.get(provider_name)
            if provider is None:
                continue
            report = _component_delay_comparison(
                native,
                provider,
                component="diffraction",
                case="diffraction",
                tau_tol_s=float(args.tau_tol_s),
                exact_counts=False,
                magnitude_tol_db=float(args.magnitude_tol_db),
                phase_tol_rad=float(args.phase_tol_rad),
                angle_tol_rad=float(args.angle_tol_rad),
                geometry_tol_m=float(args.geometry_tol_m),
            )
            by_provider.setdefault(provider_name, []).append(
                {"seed": seed, "coverage_ratio": report["coverage_ratio"], "passed": report["passed"]}
            )
    return {
        provider: {
            "seeds": rows,
            "coverage_confidence_interval": _confidence_interval(
                [float(row["coverage_ratio"]) for row in rows]
            ),
        }
        for provider, rows in by_provider.items()
    }


def _run_all(args: argparse.Namespace) -> dict[str, Any]:
    primary_seed = int(args.seed)
    seeds = tuple(dict.fromkeys((primary_seed, *_parse_ints(args.diffraction_seeds))))
    provider_payloads = [
        _run_provider_subprocess(args, provider, seed=primary_seed)
        for provider in ("native", "original", "sionna")
    ]
    providers = {
        payload.get("provider", payload.get("provider_arg", "unknown")): payload
        for payload in provider_payloads
        if payload.get("ok")
    }
    native = providers.get("channel_native")
    correctness: dict[str, Any] = {}
    if native is not None:
        for provider_name in ("original_channel", "sionna"):
            provider = providers.get(provider_name)
            if provider is None:
                continue
            correctness[f"los_native_vs_{provider_name}"] = _component_delay_comparison(
                native,
                provider,
                component="los",
                case="los",
                tau_tol_s=float(args.tau_tol_s),
                exact_counts=True,
                magnitude_tol_db=float(args.magnitude_tol_db),
                phase_tol_rad=float(args.phase_tol_rad),
                angle_tol_rad=float(args.angle_tol_rad),
                geometry_tol_m=float(args.geometry_tol_m),
            )
            correctness[f"reflection_native_vs_{provider_name}"] = _component_delay_comparison(
                native,
                provider,
                component="reflection",
                case="reflection",
                tau_tol_s=float(args.tau_tol_s),
                exact_counts=True,
                magnitude_tol_db=float(args.magnitude_tol_db),
                phase_tol_rad=float(args.phase_tol_rad),
                angle_tol_rad=float(args.angle_tol_rad),
                geometry_tol_m=float(args.geometry_tol_m),
            )
            correctness[f"diffraction_reference_covered_by_native_{provider_name}"] = (
                _component_delay_comparison(
                    native,
                    provider,
                    component="diffraction",
                    case="diffraction",
                    tau_tol_s=float(args.tau_tol_s),
                    exact_counts=False,
                    magnitude_tol_db=float(args.magnitude_tol_db),
                    phase_tol_rad=float(args.phase_tol_rad),
                    angle_tol_rad=float(args.angle_tol_rad),
                    geometry_tol_m=float(args.geometry_tol_m),
                )
            )
    seed_payloads = [(primary_seed, provider_payloads)]
    for seed in seeds:
        if seed == primary_seed:
            continue
        seed_payloads.append(
            (
                seed,
                [
                    _run_provider_subprocess(args, provider, seed=seed)
                    for provider in ("native", "original", "sionna")
                ],
            )
        )
    diffraction_confidence = _diffraction_seed_summary(seed_payloads, args)
    scenario = {
        "scene": args.scene,
        "scene_xml": str(_scene_xml(args)),
        "frequency_hz": float(args.frequency_hz),
        "tx": _parse_points(args.tx),
        "rx": _parse_points(args.rx),
        "samples": int(args.samples),
        "max_num_paths": int(args.max_num_paths),
        "warmup": int(args.warmup),
        "repeats": int(args.repeats),
        "tau_tol_s": float(args.tau_tol_s),
        "cfr_offsets_hz": _parse_floats(args.cfr_offsets_hz),
        "diffraction_seeds": seeds,
    }
    all_reports = list(correctness.values())
    gates = {
        "all_primary_comparisons_passed": bool(all_reports) and all(
            report["passed"] for report in all_reports
        ),
        "diffraction_coverage_ci_lower_at_least_0_95": bool(diffraction_confidence)
        and all(
            row["coverage_confidence_interval"]["lower_95"] >= 0.95
            for row in diffraction_confidence.values()
        ),
    }
    gates["passed"] = all(gates.values())
    result = {
        "schema": {"name": SCHEMA_NAME, "version": SCHEMA_VERSION},
        "scenario": scenario,
        "providers": provider_payloads,
        "correctness": correctness,
        "diffraction_multi_seed": diffraction_confidence,
        "gates": gates,
        "speed": _speed_summary(providers),
        "notes": [
            "Original Channel path cases always include LoS; component stats classify paths by interaction type.",
            "Channel Native diffraction exports all first-order edge paths for this scene, while Original Channel and Sionna return capped or selected path sets.",
            "Diffraction correctness is therefore reported as coverage of Original/Sionna delay samples by the Native full export, not equal path counts.",
        ],
    }
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", choices=("native", "original", "sionna"), default=None)
    parser.add_argument("--scene", default=DEFAULT_SCENE)
    parser.add_argument("--sionna-source-root", type=Path, default=DEFAULT_SIONNA_ROOT)
    parser.add_argument("--channel-root", type=Path, default=DEFAULT_CHANNEL_ROOT)
    parser.add_argument("--frequency-hz", type=float, default=3.5e9)
    parser.add_argument(
        "--tx",
        default=";".join(",".join(str(v) for v in p) for p in DEFAULT_TX),
    )
    parser.add_argument(
        "--rx",
        default=";".join(",".join(str(v) for v in p) for p in DEFAULT_RX),
    )
    parser.add_argument("--samples", type=int, default=4096)
    parser.add_argument("--max-num-paths", type=int, default=64)
    parser.add_argument("--diffraction-state-budget", type=int, default=4096)
    parser.add_argument("--inserted-reflection-state-budget", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--diffraction-seeds", default="7,17,29")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--tau-tol-s", type=float, default=1.0e-9)
    parser.add_argument("--magnitude-tol-db", type=float, default=0.25)
    parser.add_argument("--phase-tol-rad", type=float, default=1.0e-3)
    parser.add_argument("--angle-tol-rad", type=float, default=1.0e-3)
    parser.add_argument("--geometry-tol-m", type=float, default=1.0e-3)
    parser.add_argument("--cfr-offsets-hz", default="-1000000,0,1000000")
    parser.add_argument(
        "--reduced-ci",
        action="store_true",
        help="Run one cold/steady sample, one diffraction seed, and fail on a gate miss.",
    )
    parser.add_argument("--fail-on-gate", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.reduced_ci:
        args.samples = min(int(args.samples), 256)
        args.max_num_paths = min(int(args.max_num_paths), 16)
        args.warmup = 0
        args.repeats = 1
        args.diffraction_seeds = str(args.seed)
    if args.provider == "native":
        payload = _run_native(args)
    elif args.provider == "original":
        payload = _run_original(args)
    elif args.provider == "sionna":
        payload = _run_sionna(args)
    else:
        payload = _run_all(args)
    payload.setdefault("schema", {"name": SCHEMA_NAME, "version": SCHEMA_VERSION})
    text = json.dumps(_jsonable(payload), indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    if args.output is None or args.provider is None:
        print(text)
    if args.provider is None and (args.fail_on_gate or args.reduced_ci):
        return 0 if payload.get("gates", {}).get("passed", False) else 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
