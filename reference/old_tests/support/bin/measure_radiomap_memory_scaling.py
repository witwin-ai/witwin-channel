"""Measure GPU memory scaling for three-cube radio-map benchmarks."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

if os.environ.get("WITWIN_CHANNEL_MAIN_SHOW", "0") != "1":
    os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import drjit as dr


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

TESTS_DIR = REPO_ROOT / "tests"
TESTS_INIT = TESTS_DIR / "__init__.py"
existing_tests_module = sys.modules.get("tests")
existing_tests_path = getattr(existing_tests_module, "__file__", "")
if not existing_tests_path or not str(existing_tests_path).startswith(str(REPO_ROOT)):
    tests_spec = importlib.util.spec_from_file_location(
        "tests",
        TESTS_INIT,
        submodule_search_locations=[str(TESTS_DIR)],
    )
    if tests_spec is None or tests_spec.loader is None:
        raise RuntimeError(f"Unable to load local tests package from {TESTS_INIT}.")
    tests_module = importlib.util.module_from_spec(tests_spec)
    sys.modules["tests"] = tests_module
    tests_spec.loader.exec_module(tests_module)
    sys.modules.pop("tests.main", None)

try:
    from ._benchmark_runtime import benchmark_environment_report
except ImportError:
    THIS_DIR = Path(__file__).resolve().parent
    if str(THIS_DIR) not in sys.path:
        sys.path.insert(0, str(THIS_DIR))
    from _benchmark_runtime import benchmark_environment_report


DEFAULT_SAMPLES = (50_000, 100_000, 250_000, 500_000, 1_000_000)
DEFAULT_GRID_SIZE = 256
DEFAULT_WARMUP_SAMPLES = 8_192
DEFAULT_CHILD_TIMEOUT_S = 900.0
DEFAULT_OUTPUT_PREFIX = (
    REPO_ROOT / "tests" / "output" / "radiomap_three_cubes_memory_scaling"
)
_MIB = float(1024 ** 2)


@dataclass(frozen=True)
class ChildMeasurement:
    framework: str
    mode: str
    samples_per_tx: int
    elapsed_ms: float
    drjit_device_watermark_mib: float


def _sync_gpu() -> None:
    if hasattr(dr, "sync_thread"):
        dr.sync_thread()


def _clear_drjit_device_statistics() -> None:
    dr.detail.malloc_clear_statistics()


def _drjit_device_watermark_mib() -> float:
    return float(dr.detail.malloc_watermark(dr.detail.AllocType.Device)) / _MIB


def _import_benchmark_helpers():
    from tests.main.plot_multipath_components import CUBE1_BASE_CENTER
    from tests.main.plot_radiomap_sionna_three_cubes import (
        DEFAULT_BOUNDS,
        DEFAULT_PLANE_Z,
        DEFAULT_TX_POS,
        DEFAULT_WITWIN_EDGE_SELECTION_MODE,
        _prepare_sionna_scene,
        _run_sionna,
    )
    from tests.support.bin.compare_radiomap_sionna_three_cubes import (
        _build_comparison_scene,
        _make_witwin_monitor,
        _make_witwin_tracer,
    )
    import witwin as wt

    return {
        "CUBE1_BASE_CENTER": CUBE1_BASE_CENTER,
        "DEFAULT_BOUNDS": DEFAULT_BOUNDS,
        "DEFAULT_PLANE_Z": DEFAULT_PLANE_Z,
        "DEFAULT_TX_POS": DEFAULT_TX_POS,
        "DEFAULT_WITWIN_EDGE_SELECTION_MODE": DEFAULT_WITWIN_EDGE_SELECTION_MODE,
        "_build_comparison_scene": _build_comparison_scene,
        "_make_witwin_monitor": _make_witwin_monitor,
        "_make_witwin_tracer": _make_witwin_tracer,
        "_prepare_sionna_scene": _prepare_sionna_scene,
        "_run_sionna": _run_sionna,
        "wt": wt,
    }


def _child_measurement(
    *,
    framework: str,
    samples_per_tx: int,
    diffraction: bool,
    grid_size: int,
    warmup_samples: int,
) -> ChildMeasurement:
    helpers = _import_benchmark_helpers()
    wt = helpers["wt"]
    bounds = helpers["DEFAULT_BOUNDS"]
    plane_z = float(helpers["DEFAULT_PLANE_Z"])
    tx_pos = helpers["DEFAULT_TX_POS"]
    cube1_x = float(helpers["CUBE1_BASE_CENTER"][0])
    scene = helpers["_build_comparison_scene"](
        cube1_x,
        edge_selection_mode=helpers["DEFAULT_WITWIN_EDGE_SELECTION_MODE"],
    )

    actual_samples = int(samples_per_tx)
    warmup_samples = max(1, min(int(warmup_samples), actual_samples))
    mode = "with_diff" if diffraction else "no_diff"

    if framework == "witwin":
        warmup_tracer = helpers["_make_witwin_tracer"](
            scene=scene,
            samples_per_tx=warmup_samples,
            max_diffractions=1 if diffraction else 0,
        )
        warmup_monitor = helpers["_make_witwin_monitor"](
            plane_z=plane_z,
            bounds=bounds,
            grid_size=grid_size,
            samples_per_tx=warmup_samples,
            max_diffractions=1 if diffraction else 0,
            seed=7,
        )
        trace_output = warmup_tracer.trace(wt.Point3f(*tx_pos), monitor=warmup_monitor, verbose=False)
        _ = trace_output.monitor(warmup_monitor.name) if hasattr(trace_output, "monitor") else trace_output
        _sync_gpu()

        tracer = helpers["_make_witwin_tracer"](
            scene=scene,
            samples_per_tx=actual_samples,
            max_diffractions=1 if diffraction else 0,
        )
        monitor = helpers["_make_witwin_monitor"](
            plane_z=plane_z,
            bounds=bounds,
            grid_size=grid_size,
            samples_per_tx=actual_samples,
            max_diffractions=1 if diffraction else 0,
            seed=7,
        )

        def _run_once() -> None:
            trace_result = tracer.trace(wt.Point3f(*tx_pos), monitor=monitor, verbose=False)
            result = trace_result.monitor(monitor.name) if hasattr(trace_result, "monitor") else trace_result
            del result
            _sync_gpu()

    elif framework == "sionna":
        conversion, rt, sionna_scene = helpers["_prepare_sionna_scene"](scene=scene, tx_pos=tx_pos)
        del conversion
        helpers["_run_sionna"](
            rt=rt,
            scene=sionna_scene,
            plane_z=plane_z,
            bounds=bounds,
            grid_size=grid_size,
            samples_per_tx=warmup_samples,
            diffraction=diffraction,
            edge_diffraction=True,
        )
        _sync_gpu()

        def _run_once() -> None:
            result, _elapsed = helpers["_run_sionna"](
                rt=rt,
                scene=sionna_scene,
                plane_z=plane_z,
                bounds=bounds,
                grid_size=grid_size,
                samples_per_tx=actual_samples,
                diffraction=diffraction,
                edge_diffraction=True,
            )
            _ = float(result.path_gain[0, 0, 0].numpy())
            _sync_gpu()

    else:
        raise ValueError(f"Unsupported framework: {framework}")

    _sync_gpu()
    _clear_drjit_device_statistics()
    t0 = time.perf_counter()
    _run_once()
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    _sync_gpu()
    device_watermark_mib = _drjit_device_watermark_mib()

    return ChildMeasurement(
        framework=str(framework),
        mode=str(mode),
        samples_per_tx=int(actual_samples),
        elapsed_ms=float(elapsed_ms),
        drjit_device_watermark_mib=float(device_watermark_mib),
    )


def _run_child_subprocess(
    *,
    framework: str,
    samples_per_tx: int,
    diffraction: bool,
    grid_size: int,
    warmup_samples: int,
    child_timeout_s: float,
) -> dict[str, object]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--child",
        "--framework",
        str(framework),
        "--samples-per-tx",
        str(int(samples_per_tx)),
        "--grid-size",
        str(int(grid_size)),
        "--warmup-samples",
        str(int(warmup_samples)),
    ]
    if diffraction:
        command.append("--diffraction")
    try:
        completed = subprocess.run(
            command,
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            check=False,
            timeout=max(1.0, float(child_timeout_s)),
        )
    except subprocess.TimeoutExpired:
        return {
            "framework": str(framework),
            "mode": "with_diff" if diffraction else "no_diff",
            "samples_per_tx": int(samples_per_tx),
            "error": f"timeout>{float(child_timeout_s):.1f}s",
        }
    if completed.returncode != 0:
        return {
            "framework": str(framework),
            "mode": "with_diff" if diffraction else "no_diff",
            "samples_per_tx": int(samples_per_tx),
            "error": completed.stderr.strip() or completed.stdout.strip() or f"exit={completed.returncode}",
        }

    stdout = completed.stdout.strip()
    if not stdout:
        return {
            "framework": str(framework),
            "mode": "with_diff" if diffraction else "no_diff",
            "samples_per_tx": int(samples_per_tx),
            "error": "child produced no stdout",
        }
    last_line = stdout.splitlines()[-1]
    return json.loads(last_line)


def _plot_results(results, *, output_path: Path) -> Path:
    modes = ("no_diff", "with_diff")
    frameworks = ("witwin", "sionna")
    colors = {
        "witwin": "#004f5a",
        "sionna": "#c4652d",
    }

    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.5), constrained_layout=True)
    for col, mode in enumerate(modes):
        ax_mem = axes[0, col]
        ax_time = axes[1, col]
        mode_results = [
            item for item in results
            if item.get("mode") == mode and "error" not in item
        ]
        for framework in frameworks:
            series = sorted(
                [item for item in mode_results if item.get("framework") == framework],
                key=lambda item: int(item["samples_per_tx"]),
            )
            if not series:
                continue
            xs = [int(item["samples_per_tx"]) for item in series]
            ys_watermark = [float(item["drjit_device_watermark_mib"]) for item in series]
            ys_time = [float(item["elapsed_ms"]) for item in series]
            color = colors[framework]
            ax_mem.plot(xs, ys_watermark, marker="o", color=color, linewidth=2.0, label=framework)
            ax_time.plot(xs, ys_time, marker="o", color=color, linewidth=2.0, label=framework)

        title = "LoS + Reflection" if mode == "no_diff" else "LoS + Reflection + Diffraction"
        ax_mem.set_title(f"{title} Memory")
        ax_mem.set_xlabel("samples_per_tx")
        ax_mem.set_ylabel("MiB")
        ax_mem.grid(True, alpha=0.25)
        handles, labels = ax_mem.get_legend_handles_labels()
        if handles:
            ax_mem.legend(fontsize=8)

        ax_time.set_title(f"{title} Time")
        ax_time.set_xlabel("samples_per_tx")
        ax_time.set_ylabel("ms")
        ax_time.grid(True, alpha=0.25)
        handles, labels = ax_time.get_legend_handles_labels()
        if handles:
            ax_time.legend(fontsize=8)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def _parent_main(args) -> int:
    results = []
    modes = [False, True] if args.include_no_diff else [True]
    for diffraction in modes:
        for samples_per_tx in args.samples_per_tx:
            for framework in ("witwin", "sionna"):
                result = _run_child_subprocess(
                    framework=framework,
                    samples_per_tx=int(samples_per_tx),
                    diffraction=bool(diffraction),
                    grid_size=int(args.grid_size),
                    warmup_samples=int(args.warmup_samples),
                    child_timeout_s=float(args.child_timeout_s),
                )
                results.append(result)

    output_prefix = Path(args.output_prefix)
    figure_path = _plot_results(results, output_path=output_prefix.with_suffix(".png"))
    json_path = output_prefix.with_suffix(".json")
    payload = {
        "samples_per_tx": [int(value) for value in args.samples_per_tx],
        "grid_size": int(args.grid_size),
        "include_no_diff": bool(args.include_no_diff),
        "warmup_samples": int(args.warmup_samples),
        "child_timeout_s": float(args.child_timeout_s),
        "measurement_method": (
            "separate child process per point; small warmup to compile/cache; "
            "Dr.Jit device allocator watermark via dr.detail.malloc_watermark(Device)"
        ),
        "results": results,
        "environment": benchmark_environment_report(),
        "figure": str(figure_path),
    }
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(
        json.dumps(
            {
                "figure": str(figure_path),
                "json": str(json_path),
            },
            indent=2,
        )
    )
    return 0


def _child_main(args) -> int:
    result = _child_measurement(
        framework=str(args.framework),
        samples_per_tx=int(args.samples_per_tx[0]),
        diffraction=bool(args.diffraction),
        grid_size=int(args.grid_size),
        warmup_samples=int(args.warmup_samples),
    )
    print(json.dumps(asdict(result)))
    return 0


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--child", action="store_true")
    parser.add_argument("--framework", choices=("witwin", "sionna"))
    parser.add_argument("--samples-per-tx", type=int, nargs="+", default=list(DEFAULT_SAMPLES))
    parser.add_argument("--diffraction", action="store_true")
    parser.add_argument("--include-no-diff", action="store_true")
    parser.add_argument("--grid-size", type=int, default=DEFAULT_GRID_SIZE)
    parser.add_argument("--warmup-samples", type=int, default=DEFAULT_WARMUP_SAMPLES)
    parser.add_argument("--child-timeout-s", type=float, default=DEFAULT_CHILD_TIMEOUT_S)
    parser.add_argument("--output-prefix", type=Path, default=DEFAULT_OUTPUT_PREFIX)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.child:
        if args.framework is None:
            raise ValueError("--framework is required in --child mode.")
        if len(args.samples_per_tx) != 1:
            raise ValueError("--child mode expects exactly one --samples-per-tx value.")
        return _child_main(args)
    return _parent_main(args)


if __name__ == "__main__":
    raise SystemExit(main())
