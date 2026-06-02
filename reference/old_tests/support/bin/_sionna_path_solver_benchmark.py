"""Shared Sionna-vs-Witwin PathSolver benchmark helpers."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from statistics import mean, median
import sys
import time
from pathlib import Path
from typing import Any

try:
    from ._benchmark_runtime import benchmark_environment_report
    from ._multipath_scaling_common import flush_gpu_caches, sync_gpu
except ImportError:
    THIS_DIR = Path(__file__).resolve().parent
    if str(THIS_DIR) not in sys.path:
        sys.path.insert(0, str(THIS_DIR))
    from _benchmark_runtime import benchmark_environment_report
    from _multipath_scaling_common import flush_gpu_caches, sync_gpu

import numpy as np
import torch

from witwin.channel import (
    InteractionType,
    Material,
    Mesh,
    PathMonitor,
    Scene,
    Structure,
    Tracer,
    load_sionna_rt,
    scene_to_sionna_scene,
)
from witwin.channel.validation import build_single_wedge_case


_FREQUENCY_HZ = 28e9
_WITWIN_TYPE_LABELS = {
    int(InteractionType.NONE): "los",
    int(InteractionType.REFLECTION): "reflection",
    int(InteractionType.DIFFRACTION): "diffraction",
    int(InteractionType.TRANSMISSION): "transmission",
    int(InteractionType.SCATTERING): "scattering",
}
_SIONNA_TYPE_LABELS = {
    0: "los",
    1: "reflection",
    2: "scattering",
    4: "transmission",
    8: "diffraction",
}


@dataclass(frozen=True)
class ScenarioDefinition:
    name: str
    description: str
    scene: Scene
    tx_positions: tuple[torch.Tensor, ...]
    rx_positions: torch.Tensor
    reflection_n_rays: int
    reflection_max_bounces: int
    monitor_max_diffractions: int
    sionna_los: bool
    sionna_specular_reflection: bool
    sionna_diffraction: bool
    sionna_samples_per_src: int


def _iso_array(rt):
    return rt.PlanarArray(
        num_rows=1,
        num_cols=1,
        vertical_spacing=0.5,
        horizontal_spacing=0.5,
        pattern="iso",
        polarization="V",
    )


def _interaction_signature(type_codes: np.ndarray, mapping: dict[int, str]) -> str:
    labels = [mapping[int(code)] for code in type_codes if int(code) != 0]
    return "los" if len(labels) == 0 else " -> ".join(labels)


def _summarize_witwin_results(results) -> dict[str, Any]:
    signatures: Counter[str] = Counter()
    total_valid_paths = 0
    for paths in results:
        valid = np.asarray(paths.valid, dtype=np.bool_)
        types = np.asarray(paths.types, dtype=np.int32)
        for rx_index in range(valid.shape[0]):
            for path_index in range(valid.shape[1]):
                if not bool(valid[rx_index, path_index]):
                    continue
                total_valid_paths += 1
                signatures[_interaction_signature(types[rx_index, path_index], _WITWIN_TYPE_LABELS)] += 1
    return {
        "tx_count": int(len(results)),
        "rx_count": int(results[0].num_rx) if len(results) > 0 else 0,
        "total_valid_paths": int(total_valid_paths),
        "signature_counts": dict(sorted(signatures.items())),
    }


def _summarize_sionna_paths(paths) -> dict[str, Any]:
    valid = np.asarray(paths.valid, dtype=np.bool_)
    interactions = np.asarray(paths.interactions, dtype=np.int32)
    total_valid_paths = 0
    signatures: Counter[str] = Counter()
    for rx_index in range(valid.shape[0]):
        for tx_index in range(valid.shape[1]):
            for path_index in range(valid.shape[2]):
                if not bool(valid[rx_index, tx_index, path_index]):
                    continue
                total_valid_paths += 1
                signatures[
                    _interaction_signature(interactions[:, rx_index, tx_index, path_index], _SIONNA_TYPE_LABELS)
                ] += 1
    return {
        "tx_count": int(valid.shape[1]),
        "rx_count": int(valid.shape[0]),
        "total_valid_paths": int(total_valid_paths),
        "signature_counts": dict(sorted(signatures.items())),
    }


def _timed_repeats(fn, *, warmup: int, repeats: int) -> tuple[object, dict[str, Any]]:
    if int(repeats) <= 0:
        raise ValueError("repeats must be > 0.")

    last_value = None
    for _ in range(int(warmup)):
        last_value = fn()
        sync_gpu()

    samples_ms: list[float] = []
    for _ in range(int(repeats)):
        flush_gpu_caches()
        sync_gpu()
        t0 = time.perf_counter()
        last_value = fn()
        sync_gpu()
        samples_ms.append((time.perf_counter() - t0) * 1000.0)

    return last_value, {
        "warmup": int(warmup),
        "repeats": int(repeats),
        "samples_ms": [float(value) for value in samples_ms],
        "median_ms": float(median(samples_ms)),
        "mean_ms": float(mean(samples_ms)),
        "min_ms": float(min(samples_ms)),
        "max_ms": float(max(samples_ms)),
    }


def _los_3d_scenario() -> ScenarioDefinition:
    scene = Scene(structures=[], device="cuda")
    tx_positions = (
        torch.tensor((0.0, 0.0, 1.0), dtype=torch.float32),
        torch.tensor((1.0, -0.5, 2.2), dtype=torch.float32),
        torch.tensor((2.5, 0.5, 0.7), dtype=torch.float32),
        torch.tensor((3.5, -0.5, 1.9), dtype=torch.float32),
    )
    rx_positions = torch.tensor(
        [
            (0.0, 3.0, 1.8),
            (1.0, 3.5, 0.9),
            (2.0, 2.8, 2.5),
            (3.5, 3.2, 1.1),
            (4.2, 2.6, 2.2),
        ],
        dtype=torch.float32,
    )
    return ScenarioDefinition(
        name="los_3d",
        description="Full-3D line-of-sight only multi-TX multi-RX workload.",
        scene=scene,
        tx_positions=tx_positions,
        rx_positions=rx_positions,
        reflection_n_rays=0,
        reflection_max_bounces=0,
        monitor_max_diffractions=0,
        sionna_los=True,
        sionna_specular_reflection=False,
        sionna_diffraction=False,
        sionna_samples_per_src=1,
    )


def _reflection_3d_scenario() -> ScenarioDefinition:
    wall = Mesh(
        vertices=[
            [0.0, -4.0, 0.0],
            [0.0, 4.0, 0.0],
            [0.0, 4.0, 4.0],
            [0.0, -4.0, 4.0],
        ],
        faces=[[0, 1, 2], [0, 2, 3]],
        position=(0.0, 0.0, 0.0),
        recenter=False,
        device="cpu",
    )
    scene = Scene(
        structures=[
            Structure(
                geometry=wall,
                material=Material(name="reflector-3d", eps_r=4.0, sigma_e=0.1),
                name="wall-3d",
            )
        ],
        device="cuda",
    )
    tx_positions = (
        torch.tensor((-3.0, -5.0, 1.4), dtype=torch.float32),
        torch.tensor((-2.0, -5.0, 2.2), dtype=torch.float32),
        torch.tensor((1.5, -4.5, 0.7), dtype=torch.float32),
        torch.tensor((2.8, -4.2, 1.9), dtype=torch.float32),
    )
    rx_positions = torch.tensor(
        [
            (-3.0, 5.0, 0.9),
            (-2.0, 4.0, 1.8),
            (1.5, 3.0, 2.6),
            (2.8, 3.5, 1.1),
            (3.6, 2.6, 2.2),
        ],
        dtype=torch.float32,
    )
    return ScenarioDefinition(
        name="reflection_3d",
        description="Full-3D wall scene with simultaneous LoS and first-order reflection paths.",
        scene=scene,
        tx_positions=tx_positions,
        rx_positions=rx_positions,
        reflection_n_rays=8192,
        reflection_max_bounces=1,
        monitor_max_diffractions=0,
        sionna_los=True,
        sionna_specular_reflection=True,
        sionna_diffraction=False,
        sionna_samples_per_src=50000,
    )


def _diffraction_first_order_scenario() -> ScenarioDefinition:
    case = build_single_wedge_case()
    tx_positions = (
        torch.tensor(case.tx_pos, dtype=torch.float32),
        torch.tensor((1.0, -6.0, case.calculation_height), dtype=torch.float32),
        torch.tensor((-1.0, -6.2, case.calculation_height), dtype=torch.float32),
        torch.tensor((2.0, -5.8, case.calculation_height), dtype=torch.float32),
    )
    rx_positions = torch.tensor(
        [
            (0.0, 3.5, case.calculation_height),
            (1.0, 3.5, case.calculation_height),
            (-1.0, 3.6, case.calculation_height),
            (2.0, 3.2, case.calculation_height),
            (0.5, 4.0, case.calculation_height),
        ],
        dtype=torch.float32,
    )
    return ScenarioDefinition(
        name="diffraction_first_order",
        description="Single-wedge first-order diffraction multi-TX multi-RX workload.",
        scene=case.scene,
        tx_positions=tx_positions,
        rx_positions=rx_positions,
        reflection_n_rays=0,
        reflection_max_bounces=0,
        monitor_max_diffractions=1,
        sionna_los=True,
        sionna_specular_reflection=False,
        sionna_diffraction=True,
        sionna_samples_per_src=50000,
    )


def _mixed_first_order_scenario() -> ScenarioDefinition:
    case = build_single_wedge_case()
    wall = Mesh(
        vertices=[
            [2.0, -4.0, 0.0],
            [2.0, 4.0, 0.0],
            [2.0, 4.0, 3.0],
            [2.0, -4.0, 3.0],
        ],
        faces=[[0, 1, 2], [0, 2, 3]],
        position=(0.0, 0.0, 0.0),
        recenter=False,
        device="cpu",
    )
    scene = Scene(
        structures=[
            *case.scene.structures,
            Structure(
                geometry=wall,
                material=Material(name="reflector-material", eps_r=4.0, sigma_e=0.1),
                name="reflector-wall",
            ),
        ],
        device="cuda",
    )
    tx_positions = (
        torch.tensor(case.tx_pos, dtype=torch.float32),
        torch.tensor((1.0, -6.0, case.calculation_height), dtype=torch.float32),
        torch.tensor((4.0, -3.5, case.calculation_height), dtype=torch.float32),
        torch.tensor((2.6, -4.4, case.calculation_height), dtype=torch.float32),
    )
    rx_positions = torch.tensor(
        [
            (0.0, 3.5, case.calculation_height),
            (1.0, 3.5, case.calculation_height),
            (3.0, 3.0, case.calculation_height),
            (2.4, 2.8, case.calculation_height),
            (-0.5, 3.8, case.calculation_height),
        ],
        dtype=torch.float32,
    )
    return ScenarioDefinition(
        name="mixed_first_order",
        description="Wedge-plus-wall mixed first-order LoS/reflection/diffraction workload.",
        scene=scene,
        tx_positions=tx_positions,
        rx_positions=rx_positions,
        reflection_n_rays=8192,
        reflection_max_bounces=1,
        monitor_max_diffractions=1,
        sionna_los=True,
        sionna_specular_reflection=True,
        sionna_diffraction=True,
        sionna_samples_per_src=50000,
    )


def available_scenarios() -> dict[str, ScenarioDefinition]:
    scenarios = (
        _los_3d_scenario(),
        _reflection_3d_scenario(),
        _diffraction_first_order_scenario(),
        _mixed_first_order_scenario(),
    )
    return {scenario.name: scenario for scenario in scenarios}


def _slice_scenario(
    scenario: ScenarioDefinition,
    *,
    tx_count: int | None = None,
    rx_count: int | None = None,
) -> ScenarioDefinition:
    resolved_tx_count = len(scenario.tx_positions) if tx_count is None else int(tx_count)
    resolved_rx_count = int(scenario.rx_positions.shape[0]) if rx_count is None else int(rx_count)
    if resolved_tx_count <= 0 or resolved_tx_count > len(scenario.tx_positions):
        raise ValueError(
            f"tx_count must be in [1, {len(scenario.tx_positions)}] for scenario '{scenario.name}'."
        )
    if resolved_rx_count <= 0 or resolved_rx_count > int(scenario.rx_positions.shape[0]):
        raise ValueError(
            f"rx_count must be in [1, {int(scenario.rx_positions.shape[0])}] for scenario '{scenario.name}'."
        )
    return ScenarioDefinition(
        name=scenario.name,
        description=scenario.description,
        scene=scenario.scene,
        tx_positions=tuple(scenario.tx_positions[:resolved_tx_count]),
        rx_positions=scenario.rx_positions[:resolved_rx_count].clone(),
        reflection_n_rays=scenario.reflection_n_rays,
        reflection_max_bounces=scenario.reflection_max_bounces,
        monitor_max_diffractions=scenario.monitor_max_diffractions,
        sionna_los=scenario.sionna_los,
        sionna_specular_reflection=scenario.sionna_specular_reflection,
        sionna_diffraction=scenario.sionna_diffraction,
        sionna_samples_per_src=scenario.sionna_samples_per_src,
    )


def _build_witwin_executor(scenario: ScenarioDefinition):
    tracer = Tracer(
        frequency=_FREQUENCY_HZ,
        scene=scenario.scene,
        reflection_n_rays=scenario.reflection_n_rays,
        reflection_max_bounces=scenario.reflection_max_bounces,
        max_diffractions=0,
    )
    monitor = PathMonitor(
        "rx",
        positions=scenario.rx_positions,
        max_diffractions=scenario.monitor_max_diffractions,
        return_geometry=False,
    )

    def _run():
        return tracer.trace_many(scenario.tx_positions, monitor=monitor, verbose=False)

    return _run


def _build_sionna_executor(scenario: ScenarioDefinition):
    import_result = load_sionna_rt(prefer_local=True)
    converted = scene_to_sionna_scene(scenario.scene, prefer_local=True)
    rt = import_result.rt
    sionna_scene = converted.scene
    sionna_scene.frequency = _FREQUENCY_HZ
    array = _iso_array(rt)
    sionna_scene.tx_array = array
    sionna_scene.rx_array = array
    for tx_index, tx_pos in enumerate(scenario.tx_positions):
        sionna_scene.add(rt.Transmitter(name=f"tx-{tx_index}", position=tx_pos.detach().cpu().tolist()))
    for rx_index, rx_pos in enumerate(scenario.rx_positions):
        sionna_scene.add(rt.Receiver(name=f"rx-{rx_index}", position=rx_pos.detach().cpu().tolist()))
    solver = rt.PathSolver()

    def _run():
        return solver(
            scene=sionna_scene,
            max_depth=1,
            synthetic_array=True,
            los=scenario.sionna_los,
            specular_reflection=scenario.sionna_specular_reflection,
            diffuse_reflection=False,
            refraction=False,
            diffraction=scenario.sionna_diffraction,
            edge_diffraction=False,
            samples_per_src=scenario.sionna_samples_per_src,
            seed=7,
        )

    return {
        "source": import_result.source,
        "source_root": None if import_result.source_root is None else str(import_result.source_root),
        "run": _run,
    }


def run_path_solver_benchmark(
    *,
    scenario_name: str,
    tx_count: int | None = None,
    rx_count: int | None = None,
    warmup: int = 1,
    repeats: int = 3,
) -> dict[str, Any]:
    scenarios = available_scenarios()
    if scenario_name not in scenarios:
        raise ValueError(f"Unknown scenario '{scenario_name}'. Valid options: {', '.join(sorted(scenarios))}.")
    scenario = _slice_scenario(scenarios[scenario_name], tx_count=tx_count, rx_count=rx_count)

    witwin_run = _build_witwin_executor(scenario)
    sionna_executor = _build_sionna_executor(scenario)

    witwin_results, witwin_timing = _timed_repeats(witwin_run, warmup=warmup, repeats=repeats)
    sionna_paths, sionna_timing = _timed_repeats(sionna_executor["run"], warmup=warmup, repeats=repeats)

    witwin_summary = _summarize_witwin_results(witwin_results)
    sionna_summary = _summarize_sionna_paths(sionna_paths)
    signature_match = witwin_summary["signature_counts"] == sionna_summary["signature_counts"]

    return {
        "benchmark": "path_solver_sionna_compare",
        "runtime_environment": benchmark_environment_report(),
        "scenario": {
            "name": scenario.name,
            "description": scenario.description,
            "tx_count": int(len(scenario.tx_positions)),
            "rx_count": int(scenario.rx_positions.shape[0]),
            "frequency_hz": float(_FREQUENCY_HZ),
            "reflection_n_rays": int(scenario.reflection_n_rays),
            "reflection_max_bounces": int(scenario.reflection_max_bounces),
            "max_diffractions": int(scenario.monitor_max_diffractions),
            "sionna_samples_per_src": int(scenario.sionna_samples_per_src),
        },
        "witwin": {
            "timing": witwin_timing,
            "summary": witwin_summary,
        },
        "sionna": {
            "timing": sionna_timing,
            "summary": sionna_summary,
            "source": sionna_executor["source"],
            "source_root": sionna_executor["source_root"],
        },
        "comparison": {
            "signature_match": bool(signature_match),
            "path_count_delta": int(
                witwin_summary["total_valid_paths"] - sionna_summary["total_valid_paths"]
            ),
            "median_speedup_vs_sionna": (
                float(sionna_timing["median_ms"] / witwin_timing["median_ms"])
                if float(witwin_timing["median_ms"]) > 0.0
                else None
            ),
            "mean_speedup_vs_sionna": (
                float(sionna_timing["mean_ms"] / witwin_timing["mean_ms"])
                if float(witwin_timing["mean_ms"]) > 0.0
                else None
            ),
        },
    }


def run_path_solver_stress_matrix(
    *,
    scenario_names: list[str],
    tx_counts: list[int],
    rx_counts: list[int],
    warmup: int = 0,
    repeats: int = 1,
) -> dict[str, Any]:
    scenarios = available_scenarios()
    results = []
    for scenario_name in scenario_names:
        if scenario_name not in scenarios:
            raise ValueError(f"Unknown scenario '{scenario_name}'.")
        scenario = scenarios[scenario_name]
        for tx_count in tx_counts:
            for rx_count in rx_counts:
                if tx_count > len(scenario.tx_positions) or rx_count > int(scenario.rx_positions.shape[0]):
                    continue
                results.append(
                    run_path_solver_benchmark(
                        scenario_name=scenario_name,
                        tx_count=int(tx_count),
                        rx_count=int(rx_count),
                        warmup=warmup,
                        repeats=repeats,
                    )
                )

    return {
        "benchmark": "path_solver_sionna_stress_matrix",
        "runtime_environment": benchmark_environment_report(),
        "matrix_config": {
            "scenario_names": list(scenario_names),
            "tx_counts": [int(value) for value in tx_counts],
            "rx_counts": [int(value) for value in rx_counts],
            "warmup": int(warmup),
            "repeats": int(repeats),
        },
        "results": results,
    }


def format_benchmark_summary(payload: dict[str, Any]) -> str:
    scenario = payload["scenario"]
    witwin = payload["witwin"]["timing"]
    sionna = payload["sionna"]["timing"]
    comparison = payload["comparison"]
    return (
        f"{scenario['name']} tx={scenario['tx_count']} rx={scenario['rx_count']} "
        f"witwin_median={witwin['median_ms']:.2f}ms "
        f"sionna_median={sionna['median_ms']:.2f}ms "
        f"speedup={comparison['median_speedup_vs_sionna']:.3f}x "
        f"signature_match={comparison['signature_match']}"
    )


__all__ = [
    "available_scenarios",
    "format_benchmark_summary",
    "run_path_solver_benchmark",
    "run_path_solver_stress_matrix",
]

