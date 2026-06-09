from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
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


def _time_repeated(operation, sync, *, warmup: int, repeats: int) -> tuple[Any, list[float]]:
    result = None
    for _ in range(max(0, int(warmup))):
        result = operation()
        sync(result)
    times = []
    for _ in range(max(1, int(repeats))):
        started = time.perf_counter()
        result = operation()
        sync(result)
        times.append((time.perf_counter() - started) * 1000.0)
    return result, times


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
    }


def _stats_from_labeled_paths(
    *,
    tau,
    valid,
    labels,
    num_rx: int,
    num_tx: int,
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
    return components


def _native_case_stats(result, *, num_rx: int, num_tx: int) -> dict[str, Any]:
    import numpy as np

    tau = np.zeros((num_rx, num_tx, int(result.valid.numel())), dtype=np.float64)
    valid = np.zeros_like(tau, dtype=bool)
    labels = np.empty(tau.shape, dtype=object)
    labels[:] = ""
    tx_id = result.tx_id.detach().cpu().numpy()
    rx_id = result.rx_id.detach().cpu().numpy()
    delay = result.delay_s.detach().cpu().numpy().astype(np.float64)
    component_id = result.component_id.detach().cpu().numpy()
    names = {0: "los", 1: "reflection", 2: "diffraction"}
    offsets = {(rx, tx): 0 for rx in range(num_rx) for tx in range(num_tx)}
    for index in range(delay.shape[0]):
        rx = int(rx_id[index])
        tx = int(tx_id[index])
        slot = offsets[(rx, tx)]
        offsets[(rx, tx)] += 1
        tau[rx, tx, slot] = float(delay[index])
        valid[rx, tx, slot] = True
        labels[rx, tx, slot] = names.get(int(component_id[index]), "")
    return _stats_from_labeled_paths(tau=tau, valid=valid, labels=labels, num_rx=num_rx, num_tx=num_tx)


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
            result.tx_id,
            result.rx_id,
            result.component_id,
            result.delay_s,
            result.path_gain,
        ):
            _ = tensor.numel()
        torch.cuda.synchronize()

    cases = {}
    for name, config in configs.items():
        result, times = _time_repeated(
            lambda config=config: solve(scene, config),
            sync,
            warmup=args.warmup,
            repeats=args.repeats,
        )
        component_stats = _native_case_stats(result, num_rx=len(rx_points), num_tx=len(tx_points))
        cases[name] = {
            "times_ms": times,
            "median_ms": _median(times),
            "mean_ms": _mean(times),
            "path_count": int(result.valid.numel()),
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
            "python": sys.executable,
            "torch": torch.__version__,
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
    for name, config in configs.items():
        result, times = _time_repeated(
            lambda config=config: run(config),
            sync,
            warmup=args.warmup,
            repeats=args.repeats,
        )
        tau, valid, labels = _sionna_labels(result, force_los=(name == "los"))
        component_stats = _stats_from_labeled_paths(
            tau=tau,
            valid=valid,
            labels=labels,
            num_rx=len(rx_points),
            num_tx=len(tx_points),
        )
        cases[name] = {
            "times_ms": times,
            "median_ms": _median(times),
            "mean_ms": _mean(times),
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
            "python": sys.executable,
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
            return_geometry=False,
        )
    if name == "reflection":
        return wc.path.Config(
            num_samples=int(args.samples),
            max_bounces=1,
            max_diffraction_order=0,
            max_num_paths=int(args.max_num_paths),
            return_geometry=False,
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
        return_geometry=False,
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
    for name in ("los", "reflection", "diffraction", "all"):
        config = _original_case_config(args, name)
        result, times = _time_repeated(
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
        component_stats = _stats_from_labeled_paths(
            tau=tau,
            valid=valid,
            labels=labels,
            num_rx=len(rx_points),
            num_tx=len(tx_points),
        )
        cases[name] = {
            "times_ms": times,
            "median_ms": _median(times),
            "mean_ms": _mean(times),
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
            "python": sys.executable,
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


def _component_delay_comparison(
    native: dict[str, Any],
    other: dict[str, Any],
    *,
    component: str,
    case: str,
    tau_tol_s: float,
    exact_counts: bool,
) -> dict[str, Any]:
    native_stats = native["cases"][case]["component_stats"][component]
    other_stats = other["cases"][case]["component_stats"][component]
    pair_reports = {}
    all_deltas = []
    count_mismatch_pairs = []
    covered = 0
    reference_total = 0
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
        pair_reports[pair] = {
            "native_count": len(native_delays),
            "reference_count": len(reference_delays),
            "max_abs_delay_delta_s": max(deltas) if deltas else 0.0,
            "within_tolerance": sum(1 for delta in deltas if delta <= tau_tol_s),
        }
    finite_deltas = [d for d in all_deltas if d != float("inf")]
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
        "passed": (
            (not exact_counts or not count_mismatch_pairs)
            and covered == reference_total
        ),
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
        native_ms = native["cases"][case]["median_ms"]
        row = {"channel_native_ms": native_ms}
        for provider_name in ("original_channel", "sionna"):
            provider = providers.get(provider_name)
            if provider is None:
                continue
            other_ms = provider["cases"][case]["median_ms"]
            row[f"{provider_name}_ms"] = other_ms
            row[f"native_speedup_vs_{provider_name}"] = (
                None if not native_ms or native_ms <= 0 else other_ms / native_ms
            )
        summary[case] = row
    return summary


def _run_provider_subprocess(args: argparse.Namespace, provider: str) -> dict[str, Any]:
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


def _run_all(args: argparse.Namespace) -> dict[str, Any]:
    provider_payloads = [
        _run_provider_subprocess(args, provider)
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
            )
            correctness[f"reflection_native_vs_{provider_name}"] = _component_delay_comparison(
                native,
                provider,
                component="reflection",
                case="reflection",
                tau_tol_s=float(args.tau_tol_s),
                exact_counts=True,
            )
            correctness[f"diffraction_reference_covered_by_native_{provider_name}"] = (
                _component_delay_comparison(
                    native,
                    provider,
                    component="diffraction",
                    case="diffraction",
                    tau_tol_s=float(args.tau_tol_s),
                    exact_counts=False,
                )
            )
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
    }
    result = {
        "scenario": scenario,
        "providers": [_strip_delays(payload) for payload in provider_payloads],
        "correctness": correctness,
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
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--tau-tol-s", type=float, default=1.0e-9)
    parser.add_argument("--output", type=Path, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.provider == "native":
        payload = _run_native(args)
    elif args.provider == "original":
        payload = _run_original(args)
    elif args.provider == "sionna":
        payload = _run_sionna(args)
    else:
        payload = _run_all(args)
    text = json.dumps(_jsonable(payload), indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    if args.output is None or args.provider is None:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
