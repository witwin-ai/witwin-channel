"""Save a native-vs-Dr.Jit multipath forward/JVP comparison figure."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import drjit as dr
import matplotlib.pyplot as plt
import numpy as np

import witwin as wt
from witwin.channel import DEFAULT_VARIANT, Tracer

try:
    from ._paths import OUTPUT_DIR, maybe_show
except ImportError:
    THIS_DIR = Path(__file__).resolve().parent
    if str(THIS_DIR) not in sys.path:
        sys.path.insert(0, str(THIS_DIR))
    from _paths import OUTPUT_DIR, maybe_show

from tests.main.plot_multipath_components import (
    TRACE_BOUNDS,
    _forward_power_gradient,
    as_grid,
    build_scene_for_cube1_x,
    cube_specs,
    decorate_axis,
    gradient_db_magnitude,
    make_monitor,
    parameter_config,
)

BACKENDS = ("native", "drjit")
PARAMETERS = ("tx_x", "cube1_x")
DEFAULT_OUTPUT = OUTPUT_DIR / "multipath_main_native_drjit_forward_jvp.png"


def _sync_thread() -> None:
    if hasattr(dr, "sync_thread"):
        dr.sync_thread()


def _trace_config(backend: str) -> dict:
    return {
        "trace": {
            "reflection_field_backend": backend,
            "diffraction_execution": {
                "suffix_backend": backend,
                "suffix_dda": "symbolic",
            },
        }
    }


def _make_tracer(scene, *, n_rays: int, backend: str) -> Tracer:
    return Tracer(
        frequency=1e9,
        scene=scene,
        config=_trace_config(backend),
        reflection_n_rays=n_rays,
        reflection_max_bounces=3,
        reflection_coef=0.8,
        enable_rd_diffraction=True,
        max_diffractions=2,
    )


def _grid_extent() -> tuple[float, float, float, float]:
    return (
        float(TRACE_BOUNDS[0][0]),
        float(TRACE_BOUNDS[0][1]),
        float(TRACE_BOUNDS[1][0]),
        float(TRACE_BOUNDS[1][1]),
    )


def _roundtrip_scalar(value) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    array = np.asarray(value)
    return float(array.reshape(-1)[0])


def _solver_backend_metadata(result) -> dict[str, object]:
    metadata = result.primary.metadata or {}
    return {
        "reflection_field_backend": metadata.get("reflection_field_backend"),
        "reflection_suffix_backend": metadata.get("reflection_suffix_backend"),
    }


def _tx_position_with_grad(config: dict) -> tuple[wt.Point3f, tuple[object, ...]]:
    tx_x = wt.Float(config["tx_pos"][0])
    tx_y = wt.Float(config["tx_pos"][1])
    tx_z = wt.Float(config["tx_pos"][2])
    dr.enable_grad(tx_x, tx_y)
    return wt.Point3f(tx_x, tx_y, tx_z), (tx_x, tx_y)


def _build_parameter_trace_inputs(parameter: str, *, enable_grad: bool):
    config = parameter_config(parameter)
    if parameter == "cube1_x":
        cube1_x = wt.Float(config["cube1_x"])
        if enable_grad:
            dr.enable_grad(cube1_x)
            grad_vars = (cube1_x,)
            grad_seed = (1.0,)
        else:
            grad_vars = ()
            grad_seed = ()
        scene = build_scene_for_cube1_x(cube1_x)
        tx_pos = wt.Point3f(*config["tx_pos"])
    else:
        scene = build_scene_for_cube1_x(config["cube1_x"])
        if enable_grad:
            tx_pos, grad_vars = _tx_position_with_grad(config)
            grad_seed = (1.0, 0.0) if parameter == "tx_x" else (0.0, 1.0)
        else:
            tx_pos = wt.Point3f(*config["tx_pos"])
            grad_vars = ()
            grad_seed = ()
    return config, scene, tx_pos, grad_vars, grad_seed


def _run_parameter_backend(
    parameter: str,
    backend: str,
    *,
    grid_size: int,
    n_rays: int,
) -> dict[str, object]:
    enable_grad = backend != "drjit"
    config, scene, tx_pos, grad_vars, grad_seed = _build_parameter_trace_inputs(
        parameter,
        enable_grad=enable_grad,
    )

    monitor = make_monitor(grid_size)
    tracer = _make_tracer(scene, n_rays=n_rays, backend=backend)

    _sync_thread()
    trace_start = time.perf_counter()
    result = tracer.trace(tx_pos, monitor=monitor, verbose=False, return_diffraction_audit=False)
    _sync_thread()
    trace_seconds = time.perf_counter() - trace_start

    total = result.primary.field.total
    total_power = total.real * total.real + total.imag * total.imag
    total_db = 10.0 * np.log10(as_grid(total_power, grid_size) + 1e-20)

    if enable_grad:
        for variable, seed in zip(grad_vars, grad_seed):
            dr.set_grad(variable, seed)

        _sync_thread()
        jvp_start = time.perf_counter()
        power_jvp = _forward_power_gradient(total)
        dr.eval(power_jvp)
        _sync_thread()
        jvp_seconds = time.perf_counter() - jvp_start

        jvp_np = as_grid(power_jvp, grid_size)
        jvp_db = gradient_db_magnitude(jvp_np)
        jvp_abs_max = float(np.max(np.abs(jvp_np)))
        jvp_supported = True
        jvp_note = None
    else:
        jvp_seconds = None
        jvp_db = None
        jvp_abs_max = None
        jvp_supported = False
        jvp_note = "Dr.Jit symbolic suffix baseline is forward-only for AD-sensitive multipath workloads."

    return {
        "parameter": parameter,
        "backend": backend,
        "label": config["label"],
        "specs": cube_specs(config["cube1_x"]),
        "tx_xy": (float(config["tx_pos"][0]), float(config["tx_pos"][1])),
        "extent": _grid_extent(),
        "forward_db": total_db,
        "jvp_db": jvp_db,
        "trace_seconds": trace_seconds,
        "jvp_seconds": jvp_seconds,
        "jvp_abs_max": jvp_abs_max,
        "jvp_supported": jvp_supported,
        "jvp_note": jvp_note,
        "total_power_sum": float(np.sum(np.power(10.0, total_db / 10.0))),
        "backend_metadata": _solver_backend_metadata(result),
        "tx_eval": [float(_roundtrip_scalar(tx_pos.x)), float(_roundtrip_scalar(tx_pos.y)), float(_roundtrip_scalar(tx_pos.z))],
    }


def _warmup(backend: str, parameter: str, *, grid_size: int, n_rays: int, warmup_runs: int) -> None:
    for _ in range(max(0, int(warmup_runs))):
        _run_parameter_backend(parameter, backend, grid_size=grid_size, n_rays=n_rays)


def _collect_cases(*, grid_size: int, n_rays: int, warmup_runs: int) -> list[dict[str, object]]:
    cases: list[dict[str, object]] = []
    for parameter in PARAMETERS:
        for backend in BACKENDS:
            _warmup(backend, parameter, grid_size=grid_size, n_rays=n_rays, warmup_runs=warmup_runs)
            cases.append(_run_parameter_backend(parameter, backend, grid_size=grid_size, n_rays=n_rays))
    return cases


def _jvp_limits(cases: list[dict[str, object]]) -> tuple[float, float]:
    supported = [case for case in cases if case["jvp_supported"]]
    vmax = max(float(np.percentile(case["jvp_db"], 99.5)) for case in supported)
    return vmax - 60.0, vmax


def _save_timings_json(cases: list[dict[str, object]], output: Path) -> Path:
    json_path = output.with_suffix(".json")
    payload = {
        "output_image": str(output.resolve()),
        "grid_size": int(cases[0]["forward_db"].shape[0]),
        "parameters": list(PARAMETERS),
        "backends": list(BACKENDS),
        "cases": [
            {
                "parameter": case["parameter"],
                "backend": case["backend"],
                "trace_seconds": case["trace_seconds"],
                "jvp_seconds": case["jvp_seconds"],
                "jvp_abs_max": case["jvp_abs_max"],
                "jvp_supported": case["jvp_supported"],
                "jvp_note": case["jvp_note"],
                "total_power_sum": case["total_power_sum"],
                "tx_eval": case["tx_eval"],
                "backend_metadata": case["backend_metadata"],
            }
            for case in cases
        ],
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return json_path


def make_figure(*, grid_size: int, n_rays: int, warmup_runs: int, output: Path) -> tuple[Path, Path]:
    cases = _collect_cases(grid_size=grid_size, n_rays=n_rays, warmup_runs=warmup_runs)
    case_map = {(case["parameter"], case["backend"]): case for case in cases}

    fig, axes = plt.subplots(len(PARAMETERS), 4, figsize=(18, 9.6), constrained_layout=True, squeeze=False)

    for row_idx, parameter in enumerate(PARAMETERS):
        row_cases = [case_map[(parameter, backend)] for backend in BACKENDS]
        supported_row_cases = [case for case in row_cases if case["jvp_supported"]]
        jvp_vmin, jvp_vmax = _jvp_limits(row_cases) if supported_row_cases else (None, None)
        forward_artist = None
        jvp_artist = None

        for col_idx, backend in enumerate(BACKENDS):
            case = case_map[(parameter, backend)]

            ax_forward = axes[row_idx][col_idx * 2]
            forward_artist = ax_forward.imshow(
                case["forward_db"],
                origin="lower",
                extent=case["extent"],
                cmap="jet",
                vmin=-60.0,
                vmax=-20.0,
                interpolation="nearest",
            )
            decorate_axis(
                ax_forward,
                case["specs"],
                case["tx_xy"],
                f"{backend} forward power | {case['label']}\ntrace={case['trace_seconds']:.3f}s",
            )

            ax_jvp = axes[row_idx][col_idx * 2 + 1]
            if case["jvp_supported"]:
                jvp_artist = ax_jvp.imshow(
                    case["jvp_db"],
                    origin="lower",
                    extent=case["extent"],
                    cmap="magma",
                    vmin=jvp_vmin,
                    vmax=jvp_vmax,
                    interpolation="nearest",
                )
                decorate_axis(
                    ax_jvp,
                    case["specs"],
                    case["tx_xy"],
                    f"{backend} |JVP d|E|^2/d{case['label']}| (dB)\ntrace={case['trace_seconds']:.3f}s, jvp={case['jvp_seconds']:.3f}s",
                )
            else:
                ax_jvp.set_axis_off()
                ax_jvp.text(
                    0.5,
                    0.5,
                    f"{backend} JVP unavailable\n{case['jvp_note']}",
                    ha="center",
                    va="center",
                    fontsize=11,
                    wrap=True,
                )

        if forward_artist is not None:
            fig.colorbar(forward_artist, ax=axes[row_idx, 0::2].ravel().tolist(), shrink=0.88, label="forward power (dB)")
        if jvp_artist is not None:
            fig.colorbar(jvp_artist, ax=axes[row_idx, 1::2].ravel().tolist(), shrink=0.88, label="|JVP| (dB)")

    fig.suptitle(
        "Multipath Main Native vs Dr.Jit Forward/JVP\n"
        "Dr.Jit suffix baseline is symbolic-only, so AD-sensitive suffix JVP panels are native-only.\n"
        f"scene=test_multipath_main.py, params={', '.join(PARAMETERS)}, grid={grid_size}, n_rays={n_rays}, warmup_runs={warmup_runs}",
        fontsize=14,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)

    json_path = _save_timings_json(cases, output)
    return output, json_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid-size", type=int, default=256)
    parser.add_argument("--n-rays", type=int, default=1280)
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    output_path, json_path = make_figure(
        grid_size=args.grid_size,
        n_rays=args.n_rays,
        warmup_runs=args.warmup_runs,
        output=args.output,
    )
    print(f"Saved image: {output_path.resolve()}")
    print(f"Saved timings: {json_path.resolve()}")
    maybe_show()


if __name__ == "__main__":
    main()
