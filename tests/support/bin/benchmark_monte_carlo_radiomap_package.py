"""Standalone Monte Carlo radiomap benchmark CLI."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import drjit as dr
import numpy as np
import witwin.channel as wt
from examples.monte_carlo_radiomap_three_cubes import (
    DEFAULT_BOUNDS,
    DEFAULT_FREQUENCY_HZ,
    DEFAULT_PLANE_Z,
    DEFAULT_TX_POS,
    ThreeCubeExperiment,
    _profile_kernel_history,
    _sync_drjit,
    _sync_witwin_result,
    _timed_benchmark,
)
from witwin.channel.core.scene import ReceiverGrid, Scene as ChannelScene, Transmitter
from witwin.core import Box, Material, Structure
from witwin.channel.montecarlo import Config, IntegratorOptions, Tuning, solve
try:
    from ._benchmark_runtime import benchmark_environment_report
except ImportError:
    from _benchmark_runtime import benchmark_environment_report
def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value


def _load_baseline(path: str | None) -> dict[str, Any] | None:
    if not path:
        return None
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _comparison(current: dict[str, Any], baseline: dict[str, Any] | None) -> dict[str, Any] | None:
    if baseline is None:
        return None
    result: dict[str, Any] = {}
    if "median_ms" in current and "median_ms" in baseline:
        baseline_ms = float(baseline["median_ms"])
        current_ms = float(current["median_ms"])
        result["median_ms_delta"] = current_ms - baseline_ms
        result["median_ms_delta_pct"] = 0.0 if baseline_ms == 0.0 else ((current_ms - baseline_ms) / baseline_ms) * 100.0
    if "count" in current and "count" in baseline:
        baseline_count = int(baseline["count"])
        current_count = int(current["count"])
        result["count_delta"] = current_count - baseline_count
        result["count_delta_pct"] = 0.0 if baseline_count == 0 else ((current_count - baseline_count) / baseline_count) * 100.0
    if "peak_memory_mib" in current and "peak_memory_mib" in baseline:
        baseline_peak = baseline["peak_memory_mib"]
        current_peak = current["peak_memory_mib"]
        if baseline_peak is not None and current_peak is not None:
            baseline_peak = float(baseline_peak)
            current_peak = float(current_peak)
            result["peak_memory_mib_delta"] = current_peak - baseline_peak
            result["peak_memory_mib_delta_pct"] = 0.0 if baseline_peak == 0.0 else ((current_peak - baseline_peak) / baseline_peak) * 100.0
    return result


def _kernel_history_metrics(snapshot) -> dict[str, Any]:
    return {
        "label": snapshot.label,
        "count": int(snapshot.count),
        "summary": _jsonable(snapshot.summary),
        "memory_before": _jsonable(snapshot.memory_before),
        "memory_after": _jsonable(snapshot.memory_after),
        "peak_memory_mib": snapshot.process_gpu_peak_mib,
    }


def _wall_structure() -> Structure:
    return Structure(
        name="wall",
        geometry=Box(
            position=(0.0, 0.0, 1.5),
            size=(0.25, 4.0, 3.0),
            device="cuda",
        ),
        material=Material(eps_r=4.0, sigma_e=0.0),
    )


def _wall_scene() -> ChannelScene:
    return ChannelScene(
        structures=[_wall_structure()],
        transmitters=[
            Transmitter("tx", (-2.0, 0.0, 1.5)),
        ],
        frequency=3.5e9,
        device="cuda",
    )


def _wall_forward_operation(
    *,
    grid_shape: tuple[int, int],
    samples_per_tx: int,
    seed: int,
    max_diffractions: int,
    integrator: str,
):
    scene = _wall_scene()
    scene.add(
        ReceiverGrid(
            "rm",
            axis="z",
            position=1.5,
            bounds=((-3.0, 3.0), (-3.0, 3.0)),
            grid_shape=grid_shape,
        )
    )
    config = Config(
        num_samples=max(32, int(samples_per_tx)),
        max_bounces=1,
        max_diffraction_order=int(max_diffractions),
        tuning=Tuning(enable_rd_diffraction=bool(int(max_diffractions) > 0)),
        integrator_options=IntegratorOptions(
            integrator=str(integrator),
            samples_per_tx=int(samples_per_tx),
            seed=int(seed),
            accumulation_backend="auto",
        ),
    )

    def _run():
        return solve(
            scene=scene,
            transmitter="tx",
            receiver="rm",
            config=config,
        )

    return _run


def _three_cube_experiment(*, grid_size: int, samples_per_tx: int, seed: int) -> ThreeCubeExperiment:
    return ThreeCubeExperiment(
        bounds=DEFAULT_BOUNDS,
        grid_shape=(int(grid_size), int(grid_size)),
        plane_z=DEFAULT_PLANE_Z,
        tx_pos=DEFAULT_TX_POS,
        samples_per_tx=int(samples_per_tx),
        seed=int(seed),
    )


def _forward_benchmark(args) -> dict[str, Any]:
    if args.scene == "wall":
        operation = _wall_forward_operation(
            grid_shape=(int(args.grid_size), int(args.grid_size)),
            samples_per_tx=int(args.samples_per_tx),
            seed=int(args.seed),
            max_diffractions=int(args.max_diffractions),
            integrator=str(args.integrator),
        )
    else:
        experiment = _three_cube_experiment(
            grid_size=int(args.grid_size),
            samples_per_tx=int(args.samples_per_tx),
            seed=int(args.seed),
        )
        if str(args.integrator) == "bdpt":
            config = Config(
                num_samples=experiment.forward_config.num_samples,
                max_bounces=experiment.forward_config.max_bounces,
                max_diffraction_order=int(args.max_diffractions),
                tuning=Tuning(enable_rd_diffraction=bool(int(args.max_diffractions) > 0)),
                integrator_options=IntegratorOptions(
                    integrator="bdpt",
                    samples_per_tx=int(args.samples_per_tx),
                    seed=int(args.seed),
                    accumulation_backend="auto",
                    ad=False,
                ),
            )
            operation = lambda: experiment._solve(config=config)
        elif int(args.max_diffractions) <= 0:
            config = Config(
                num_samples=experiment.forward_config.num_samples,
                max_bounces=experiment.forward_config.max_bounces,
                max_diffraction_order=0,
                tuning=Tuning(enable_rd_diffraction=False),
                integrator_options=IntegratorOptions(
                    integrator="basic",
                    samples_per_tx=int(args.samples_per_tx),
                    seed=int(args.seed),
                    accumulation_backend="auto",
                ),
            )

            def operation():
                return experiment._solve(config=config)
        else:
            operation = lambda: experiment._solve(config=experiment.forward_config)

    profile = _timed_benchmark(
        label=f"{args.scene}_forward",
        operation=operation,
        sync_result=_sync_witwin_result,
        warmup=int(args.warmup),
        repeats=int(args.repeats),
    )
    return _jsonable(profile)


def _kernel_history_benchmark(args) -> dict[str, Any]:
    if args.scene != "three_cubes":
        raise ValueError("kernel_history mode currently supports only scene='three_cubes'.")
    experiment = _three_cube_experiment(
        grid_size=int(args.grid_size),
        samples_per_tx=int(args.samples_per_tx),
        seed=int(args.seed),
    )
    if str(args.integrator) == "bdpt":
        config = Config(
            num_samples=experiment.forward_config.num_samples,
            max_bounces=experiment.forward_config.max_bounces,
            max_diffraction_order=int(args.max_diffractions),
            tuning=Tuning(enable_rd_diffraction=bool(int(args.max_diffractions) > 0)),
            integrator_options=IntegratorOptions(
                integrator="bdpt",
                samples_per_tx=int(args.samples_per_tx),
                seed=int(args.seed),
                accumulation_backend="auto",
                ad=False,
            ),
        )
        operation = lambda: experiment._solve(config=config)
    elif int(args.max_diffractions) <= 0:
        config = Config(
            num_samples=experiment.forward_config.num_samples,
            max_bounces=experiment.forward_config.max_bounces,
            max_diffraction_order=0,
            tuning=Tuning(enable_rd_diffraction=False),
            integrator_options=IntegratorOptions(
                integrator="basic",
                samples_per_tx=int(args.samples_per_tx),
                seed=int(args.seed),
                accumulation_backend="auto",
            ),
        )
        operation = lambda: experiment._solve(config=config)
    else:
        operation = lambda: experiment._solve(config=experiment.forward_config)
    warmup = max(int(args.warmup), 2) if str(args.integrator) == "bdpt" else int(args.warmup)
    snapshot = _profile_kernel_history(
        "standalone_monte_carlo_radiomap",
        operation,
        warmup=warmup,
    )
    return {
        "label": snapshot.label,
        "count": int(snapshot.count),
        "summary": _jsonable(snapshot.summary),
        "memory_before": _jsonable(snapshot.memory_before),
        "memory_after": _jsonable(snapshot.memory_after),
        "peak_memory_mib": snapshot.process_gpu_peak_mib,
    }


def _ad_scalar_loss_benchmark(args) -> dict[str, Any]:
    experiment = _three_cube_experiment(
        grid_size=int(args.grid_size),
        samples_per_tx=int(args.samples_per_tx),
        seed=int(args.seed),
    )
    parameter = str(args.parameter)

    def _bdpt_scalar_loss():
        config = Config(
            num_samples=experiment.gradient_config.num_samples,
            max_bounces=experiment.gradient_config.max_bounces,
            max_diffraction_order=int(args.max_diffractions),
            tuning=Tuning(
                enable_rd_diffraction=bool(int(args.max_diffractions) > 0),
                shadow_boundary_mode="none",
            ),
            integrator_options=IntegratorOptions(
                integrator="bdpt",
                samples_per_tx=int(args.samples_per_tx),
                seed=int(args.seed),
                accumulation_backend="auto",
                ad=True,
            ),
        )
        if parameter == "tx_x":
            variable = wt.Float(experiment.tx_pos[0])
            dr.enable_grad(variable)
            result = experiment._solve(tx_x=variable, config=config)
        elif parameter == "cube1_x":
            variable = wt.Float(experiment.base_centers[0][0])
            dr.enable_grad(variable)
            result = experiment._solve(cube1_x=variable, config=config)
        else:
            raise ValueError("parameter must be 'tx_x' or 'cube1_x'.")
        loss = dr.sum(result.path_gain)
        dr.eval(loss)
        dr.sync_thread()
        return variable, loss

    def _forward():
        if str(args.integrator) == "bdpt":
            variable, loss = _bdpt_scalar_loss()
        else:
            variable, loss = experiment.scalar_loss(parameter=parameter)
        return variable, loss

    warmup = max(int(args.warmup), 2) if str(args.integrator) == "bdpt" else int(args.warmup)

    forward_profile = _timed_benchmark(
        label=f"{parameter}_scalar_loss_forward",
        operation=_forward,
        sync_result=lambda result: _sync_drjit(),
        warmup=warmup,
        repeats=int(args.repeats),
    )

    def _backward():
        if str(args.integrator) == "bdpt":
            variable, loss = _bdpt_scalar_loss()
        else:
            variable, loss = experiment.scalar_loss(parameter=parameter)
        dr.backward(loss)
        grad = dr.grad(variable)
        dr.eval(grad)
        _sync_drjit()
        loss_value = float(np.asarray(loss, dtype=np.float64).reshape(-1)[0])
        grad_value = float(np.asarray(grad, dtype=np.float64).reshape(-1)[0])
        return {"loss": loss_value, "grad": grad_value}

    backward_profile = _timed_benchmark(
        label=f"{parameter}_scalar_loss_backward",
        operation=_backward,
        sync_result=lambda result: _sync_drjit(),
        warmup=warmup,
        repeats=int(args.repeats),
    )
    forward_history = _profile_kernel_history(
        f"{parameter}_scalar_loss_forward_kernel_history",
        lambda: _forward(),
        warmup=warmup,
    )

    def _backward_kernel_history_only():
        if str(args.integrator) == "bdpt":
            variable, loss = _bdpt_scalar_loss()
        else:
            variable, loss = experiment.scalar_loss(parameter=parameter)
        _sync_drjit()
        dr.kernel_history_clear()
        dr.backward(loss)
        grad = dr.grad(variable)
        dr.eval(grad)
        _sync_drjit()

    backward_history = _profile_kernel_history(
        f"{parameter}_scalar_loss_backward_kernel_history",
        _backward_kernel_history_only,
        warmup=warmup,
    )

    return {
        "parameter": parameter,
        "forward": _jsonable(forward_profile),
        "backward": _jsonable(backward_profile),
        "kernel_history": {
            "forward": _kernel_history_metrics(forward_history),
            "backward": _kernel_history_metrics(backward_history),
        },
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("forward", "kernel_history", "ad_scalar_loss"), required=True)
    parser.add_argument("--scene", choices=("wall", "three_cubes"), default="three_cubes")
    parser.add_argument("--integrator", choices=("basic", "bdpt"), default="basic")
    parser.add_argument("--grid-size", type=int, default=256)
    parser.add_argument("--samples-per-tx", type=int, default=250_000)
    parser.add_argument("--max-diffractions", type=int, default=1)
    parser.add_argument("--parameter", choices=("tx_x", "cube1_x"), default="tx_x")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--baseline-json", type=str, default=None)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--json", action="store_true", default=False)
    return parser


def main() -> None:
    args = _parser().parse_args()
    baseline = _load_baseline(args.baseline_json)
    if args.mode == "forward":
        metrics = _forward_benchmark(args)
    elif args.mode == "kernel_history":
        metrics = _kernel_history_benchmark(args)
    else:
        metrics = _ad_scalar_loss_benchmark(args)

    result = {
        "mode": str(args.mode),
        "scene": str(args.scene),
        "integrator": str(args.integrator),
        "grid_size": int(args.grid_size),
        "samples_per_tx": int(args.samples_per_tx),
        "max_diffractions": int(args.max_diffractions),
        "parameter": None if args.mode != "ad_scalar_loss" else str(args.parameter),
        "environment": benchmark_environment_report(),
        "metrics": metrics,
    }
    comparison = _comparison(metrics if args.mode != "ad_scalar_loss" else metrics["backward"], baseline)
    if comparison:
        result["comparison"] = comparison

    text = json.dumps(_jsonable(result), indent=2, sort_keys=True)
    if args.output:
        Path(args.output).write_text(text + "\n", encoding="utf-8")
    if args.json or not args.output:
        print(text)


if __name__ == "__main__":
    main()
