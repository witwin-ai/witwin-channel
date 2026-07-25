from __future__ import annotations

import argparse
import json
import os
import pathlib
import subprocess
import sys
import time
from typing import Any

import numpy as np
import torch

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO_ROOT / "src"))
sys.path.insert(0, str(_REPO_ROOT))

# Local source must resolve from this checkout before importing the benchmark target.
from witwin.core import PhysicalMaterial, Scene, Structure  # noqa: E402
from tests.support.core_world import (  # noqa: E402
    make_receiver_grid,
    make_transmitter,
)
from witwin.channel.montecarlo.bdpt import Config, solve  # noqa: E402


def _default_channel_root() -> pathlib.Path:
    candidates = tuple(parent / "channel" for parent in _REPO_ROOT.parents)
    return next((path for path in candidates if path.is_dir()), candidates[0])


DEFAULT_CHANNEL_ROOT = _default_channel_root()
FREQUENCY_HZ = 3.0e9
TX_POSITION = (0.0, -1.0, 0.5)
GRID_BOUNDS = ((-1.0, 1.0), (0.0, 1.0))
RECEIVER_PLANE_X = 0.0


def _plane_mesh() -> tuple[torch.Tensor, torch.Tensor]:
    vertices = torch.tensor(
        [
            [2.5, -3.0, -1.0],
            [2.5, 3.0, -1.0],
            [2.5, -3.0, 2.0],
            [2.5, 3.0, 2.0],
        ],
        dtype=torch.float32,
    )
    faces = torch.tensor([[0, 1, 2], [1, 3, 2]], dtype=torch.int32)
    return vertices, faces


def _native_scene(grid_size: int) -> Scene:
    vertices, faces = _plane_mesh()
    y_min, y_max = GRID_BOUNDS[0]
    z_min, z_max = GRID_BOUNDS[1]
    spacing_y = (y_max - y_min) / float(grid_size)
    spacing_z = (z_max - z_min) / float(grid_size)
    return Scene(
        structures=[
            Structure(
                vertices=vertices,
                faces=faces,
                material=PhysicalMaterial(eps_r=4.0, sigma_e=0.01),
                name="single-plane-wall",
                surface_id=11,
            )
        ],
        transmitters=[make_transmitter(position=torch.tensor(TX_POSITION, dtype=torch.float32), power_w=1.0)],
        receivers=[
            make_receiver_grid(
                origin=torch.tensor([RECEIVER_PLANE_X, y_min + 0.5 * spacing_y, z_min + 0.5 * spacing_z]),
                x_axis=torch.tensor([0.0, 1.0, 0.0]),
                y_axis=torch.tensor([0.0, 0.0, 1.0]),
                shape=(grid_size, grid_size),
                spacing=(spacing_y, spacing_z),
            )
        ],
        frequency=FREQUENCY_HZ,
    )


def _original_worker_code() -> str:
    return r"""
from __future__ import annotations

import argparse
import inspect
import json
import pathlib
import sys
import time

if not hasattr(inspect, "getargspec"):
    inspect.getargspec = inspect.getfullargspec  # type: ignore[attr-defined]

import drjit as dr
import numpy as np
import torch


def _timing_summary(timings):
    ordered = sorted(float(v) for v in timings)
    return {
        "median_ms": ordered[len(ordered) // 2] * 1000.0,
        "min_ms": min(ordered) * 1000.0,
        "max_ms": max(ordered) * 1000.0,
        "p95_ms": ordered[min(len(ordered) - 1, int(0.95 * (len(ordered) - 1)))] * 1000.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--channel-root", required=True)
    parser.add_argument("--output-npz", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--grid-size", type=int, required=True)
    parser.add_argument("--samples", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--warmup-runs", type=int, required=True)
    parser.add_argument("--repeats", type=int, required=True)
    args = parser.parse_args()

    channel_root = pathlib.Path(args.channel_root)
    sys.path.insert(0, str(channel_root))
    sys.path.insert(0, str(channel_root.parent / "core"))

    from witwin.channel.core.scene import Mesh, ReceiverGrid, Scene, Transmitter
    from witwin.channel.montecarlo import Config, IntegratorOptions, Tuning, solve
    from witwin.core import PhysicalMaterial as Material, Structure

    scene = Scene(
        structures=[
            Structure(
                name="single-plane-wall",
                geometry=Mesh(
                    vertices=torch.tensor(
                        [
                            [2.5, -3.0, -1.0],
                            [2.5, 3.0, -1.0],
                            [2.5, -3.0, 2.0],
                            [2.5, 3.0, 2.0],
                        ],
                        dtype=torch.float32,
                    ),
                    faces=torch.tensor([[0, 1, 2], [1, 3, 2]], dtype=torch.int32),
                ),
                material=Material(eps_r=4.0, sigma_e=0.01),
            ),
        ],
        transmitters=[make_transmitter("tx", (0.0, -1.0, 0.5), power=1.0)],
        receivers=[
            make_receiver_grid(
                "rm",
                axis="x",
                position=0.0,
                bounds=((-1.0, 1.0), (0.0, 1.0)),
                grid_shape=(int(args.grid_size), int(args.grid_size)),
            ),
        ],
        frequency=3.0e9,
        device="cuda",
    )
    config = Config(
        num_samples=int(args.samples),
        max_bounces=1,
        max_diffraction_order=0,
        tuning=Tuning(
            enable_rd_diffraction=False,
            solver_mode="fast_approximate",
            memory_profile="memory_safe",
            shadow_boundary_mode="none",
        ),
        integrator_options=IntegratorOptions(
            samples_per_tx=int(args.samples),
            seed=int(args.seed),
            integrator="bdpt",
            accumulation_backend="rayd_reflection_accumulation",
            ad=False,
        ),
    )
    for _ in range(max(0, int(args.warmup_runs))):
        solve(scene=scene, transmitter="tx", receiver="rm", config=config)
        dr.sync_thread()

    timings = []
    result = None
    for _ in range(max(1, int(args.repeats))):
        start = time.perf_counter()
        result = solve(scene=scene, transmitter="tx", receiver="rm", config=config)
        dr.sync_thread()
        timings.append(time.perf_counter() - start)
    assert result is not None

    arrays = {"path_gain": np.asarray(result.path_gain, dtype=np.float32)}
    for name, values in dict(getattr(result, "components", {}) or {}).items():
        arrays[f"component_{name}"] = np.asarray(values, dtype=np.float32)
    np.savez_compressed(args.output_npz, **arrays)
    metadata = {
        "samples": int(args.samples),
        "seed": int(args.seed),
        "grid_size": int(args.grid_size),
        "warmup_runs": max(0, int(args.warmup_runs)),
        "repeats": max(1, int(args.repeats)),
        "timing": _timing_summary(timings),
        "path_gain_shape": list(arrays["path_gain"].shape),
        "path_gain_sum": float(np.sum(arrays["path_gain"], dtype=np.float64)),
        "path_gain_max": float(np.max(arrays["path_gain"])),
    }
    pathlib.Path(args.output_json).write_text(json.dumps(metadata, sort_keys=True, indent=2), encoding="utf-8")
    print(json.dumps({"available": True, "median_ms": metadata["timing"]["median_ms"]}), flush=True)


if __name__ == "__main__":
    main()
"""


def _timing_summary(timings: list[float]) -> dict[str, float]:
    ordered = sorted(float(value) for value in timings)
    return {
        "median_ms": ordered[len(ordered) // 2] * 1000.0,
        "min_ms": min(ordered) * 1000.0,
        "max_ms": max(ordered) * 1000.0,
        "p95_ms": ordered[min(len(ordered) - 1, int(0.95 * (len(ordered) - 1)))] * 1000.0,
    }


def _run_original(
    *,
    channel_root: pathlib.Path,
    artifact_dir: pathlib.Path,
    samples: int,
    grid_size: int,
    seed: int,
    warmup_runs: int,
    repeats: int,
    timeout_seconds: float,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    if not channel_root.exists():
        raise FileNotFoundError(f"original channel root not found: {channel_root}")
    output_npz = artifact_dir / "original_outputs.npz"
    output_json = artifact_dir / "original_metadata.json"
    command = [
        sys.executable,
        "-c",
        _original_worker_code(),
        "--channel-root",
        str(channel_root),
        "--output-npz",
        str(output_npz),
        "--output-json",
        str(output_json),
        "--grid-size",
        str(int(grid_size)),
        "--samples",
        str(int(samples)),
        "--seed",
        str(int(seed)),
        "--warmup-runs",
        str(int(warmup_runs)),
        "--repeats",
        str(int(repeats)),
    ]
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    env["PYTHONIOENCODING"] = "utf-8"
    try:
        start = time.perf_counter()
        completed = subprocess.run(
            command,
            cwd=str(channel_root),
            env=env,
            text=True,
            capture_output=True,
            check=False,
            timeout=float(timeout_seconds),
        )
        wall_time_ms = (time.perf_counter() - start) * 1000.0
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout if isinstance(exc.stdout, str) else ""
        stderr = exc.stderr if isinstance(exc.stderr, str) else ""
        raise RuntimeError(
            "original Channel single-plane BDPT subprocess timed out: "
            f"timeout={float(timeout_seconds)}s "
            f"stdout_tail={stdout[-1000:]!r} stderr_tail={stderr[-2000:]!r}"
        ) from exc
    if completed.returncode != 0:
        raise RuntimeError(
            "original Channel single-plane BDPT subprocess failed: "
            f"returncode={completed.returncode} stdout_tail={completed.stdout[-1000:]!r} "
            f"stderr_tail={completed.stderr[-4000:]!r}"
        )
    metadata = json.loads(output_json.read_text(encoding="utf-8"))
    metadata["wall_time_ms"] = float(wall_time_ms)
    metadata["stdout_tail"] = completed.stdout[-1000:]
    with np.load(output_npz) as loaded:
        arrays = {name: np.asarray(loaded[name], dtype=np.float32) for name in loaded.files}
    return metadata, arrays


def _as_numpy(values: torch.Tensor) -> np.ndarray:
    array = values.detach().to(dtype=torch.float32).cpu().numpy()
    return np.asarray(array, dtype=np.float32)


def _squeeze_tx(values: np.ndarray) -> np.ndarray:
    if values.ndim == 3 and values.shape[0] == 1:
        return values[0]
    return values


def _finite_nonzero(values: np.ndarray) -> bool:
    finite = np.isfinite(values)
    return bool(np.any(finite & (values > 0.0)))


def _delta_metrics(native: np.ndarray, original: np.ndarray) -> dict[str, float | int]:
    native_db = 10.0 * np.log10(np.clip(native.astype(np.float64), 1.0e-30, None))
    original_db = 10.0 * np.log10(np.clip(original.astype(np.float64), 1.0e-30, None))
    finite = np.isfinite(native_db) & np.isfinite(original_db)
    abs_delta = np.abs(native_db[finite] - original_db[finite])
    native_sum = float(np.sum(native.astype(np.float64)))
    original_sum = float(np.sum(original.astype(np.float64)))
    rel_error = abs(native_sum - original_sum) / max(abs(original_sum), 1.0e-30)
    if abs_delta.size == 0:
        return {
            "finite_count": 0,
            "max_abs_delta_db": float("nan"),
            "median_abs_delta_db": float("nan"),
            "relative_sum_error": rel_error,
        }
    return {
        "finite_count": int(abs_delta.size),
        "max_abs_delta_db": float(np.max(abs_delta)),
        "median_abs_delta_db": float(np.median(abs_delta)),
        "relative_sum_error": rel_error,
    }


def _run_native(
    *,
    samples: int,
    grid_size: int,
    seed: int,
    warmup_runs: int,
    repeats: int,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    scene = _native_scene(grid_size)
    config = Config(
        samples=int(samples),
        seed=int(seed),
        max_depth=1,
        max_diffraction_order=0,
        components={"los", "reflection"},
        diagnostics=True,
    )
    for _ in range(max(0, int(warmup_runs))):
        solve(scene, config)
        torch.cuda.synchronize()

    timings: list[float] = []
    result = None
    for _ in range(max(1, int(repeats))):
        torch.cuda.synchronize()
        start = time.perf_counter()
        result = solve(scene, config)
        torch.cuda.synchronize()
        timings.append(time.perf_counter() - start)
    assert result is not None
    if result.component_maps is None:
        raise RuntimeError("native single-plane BDPT benchmark requires receiver-grid component maps")

    component_los = _as_numpy(result.component_maps["los"])
    component_reflection = _as_numpy(result.component_maps["reflection"])
    arrays = {
        "path_gain": component_los + component_reflection,
        "component_los": component_los,
        "component_reflection": component_reflection,
    }
    metadata = {
        "samples": int(samples),
        "seed": int(seed),
        "grid_size": int(grid_size),
        "warmup_runs": max(0, int(warmup_runs)),
        "repeats": max(1, int(repeats)),
        "timing": _timing_summary(timings),
        "solver_metadata": result.metadata,
        "path_gain_shape": list(arrays["path_gain"].shape),
        "path_gain_sum": float(np.sum(arrays["path_gain"].astype(np.float64))),
        "path_gain_max": float(np.max(arrays["path_gain"])),
    }
    return metadata, arrays


def _gate_payload(name: str, passed: bool, **details: object) -> dict[str, object]:
    return {"name": name, "passed": bool(passed), **details}


def _enforce_gates(gates: list[dict[str, object]]) -> None:
    failed = [gate for gate in gates if not bool(gate["passed"])]
    if failed:
        names = ", ".join(str(gate["name"]) for gate in failed)
        raise RuntimeError(f"single-plane native-vs-original BDPT gates failed: {names}")


def run_benchmark(
    *,
    channel_root: str | pathlib.Path = DEFAULT_CHANNEL_ROOT,
    artifact_dir: str | pathlib.Path = _REPO_ROOT / "artifacts" / "bdpt_single_plane_native_vs_original",
    samples: int = 256,
    grid_size: int = 8,
    seed: int = 7,
    warmup_runs: int = 1,
    repeats: int = 3,
    original_timeout_seconds: float = 180.0,
    min_speedup: float = 1.25,
    max_relative_sum_error: float = 0.75,
    strict_gates: bool = False,
) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("single-plane BDPT native-vs-original benchmark requires CUDA")
    artifact_path = pathlib.Path(artifact_dir).resolve()
    artifact_path.mkdir(parents=True, exist_ok=True)

    original, original_arrays = _run_original(
        channel_root=pathlib.Path(channel_root),
        artifact_dir=artifact_path,
        samples=int(samples),
        grid_size=int(grid_size),
        seed=int(seed),
        warmup_runs=int(warmup_runs),
        repeats=int(repeats),
        timeout_seconds=float(original_timeout_seconds),
    )
    native, native_arrays = _run_native(
        samples=int(samples),
        grid_size=int(grid_size),
        seed=int(seed),
        warmup_runs=int(warmup_runs),
        repeats=int(repeats),
    )

    native_total = _squeeze_tx(native_arrays["path_gain"])
    original_total = _squeeze_tx(original_arrays["path_gain"])
    shape_match = tuple(native_total.shape) == tuple(original_total.shape)
    if shape_match:
        delta = _delta_metrics(native_total, original_total)
    else:
        delta = {
            "finite_count": 0,
            "max_abs_delta_db": float("nan"),
            "median_abs_delta_db": float("nan"),
            "relative_sum_error": float("nan"),
        }

    native_ms = float(native["timing"]["median_ms"])
    original_ms = float(original["timing"]["median_ms"])
    speedup = original_ms / native_ms if native_ms > 0.0 else float("inf")
    gates = [
        _gate_payload("shape_match", shape_match, native_shape=list(native_total.shape), original_shape=list(original_total.shape)),
        _gate_payload("native_nonzero", _finite_nonzero(native_total)),
        _gate_payload("original_nonzero", _finite_nonzero(original_total)),
        _gate_payload(
            "native_faster_than_original",
            native_ms < original_ms,
            native_median_ms=native_ms,
            original_median_ms=original_ms,
        ),
        _gate_payload("min_speedup", speedup >= float(min_speedup), speedup=speedup, min_speedup=float(min_speedup)),
        _gate_payload(
            "relative_sum_error",
            bool(shape_match) and float(delta["relative_sum_error"]) <= float(max_relative_sum_error),
            relative_sum_error=float(delta["relative_sum_error"]),
            max_relative_sum_error=float(max_relative_sum_error),
        ),
    ]
    if strict_gates:
        _enforce_gates(gates)

    payload: dict[str, Any] = {
        "benchmark": "single_plane_bdpt_native_vs_original",
        "samples": int(samples),
        "grid_size": int(grid_size),
        "seed": int(seed),
        "warmup_runs": int(warmup_runs),
        "repeats": max(1, int(repeats)),
        "native": native,
        "original": original,
        "delta": delta,
        "performance": {
            "native_median_ms": native_ms,
            "original_median_ms": original_ms,
            "native_speedup_vs_original": float(speedup),
            "native_faster_than_original": bool(native_ms < original_ms),
        },
        "gates": gates,
        "artifacts": {
            "metadata": str(artifact_path / "metadata.json"),
            "original_outputs": str(artifact_path / "original_outputs.npz"),
            "original_metadata": str(artifact_path / "original_metadata.json"),
        },
    }
    pathlib.Path(payload["artifacts"]["metadata"]).write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return payload


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--channel-root", default=str(DEFAULT_CHANNEL_ROOT))
    parser.add_argument("--artifact-dir", default=str(_REPO_ROOT / "artifacts" / "bdpt_single_plane_native_vs_original"))
    parser.add_argument("--samples", type=int, default=256)
    parser.add_argument("--grid-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--original-timeout-seconds", type=float, default=180.0)
    parser.add_argument("--min-speedup", type=float, default=1.25)
    parser.add_argument("--max-relative-sum-error", type=float, default=0.75)
    parser.add_argument("--strict-gates", action="store_true")
    parser.add_argument("--json", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    payload = run_benchmark(
        channel_root=args.channel_root,
        artifact_dir=args.artifact_dir,
        samples=args.samples,
        grid_size=args.grid_size,
        seed=args.seed,
        warmup_runs=args.warmup_runs,
        repeats=args.repeats,
        original_timeout_seconds=args.original_timeout_seconds,
        min_speedup=args.min_speedup,
        max_relative_sum_error=args.max_relative_sum_error,
        strict_gates=bool(args.strict_gates),
    )
    if args.json:
        print(json.dumps(payload, sort_keys=True))
    else:
        for key, value in payload.items():
            print(f"{key}: {value}")


if __name__ == "__main__":
    main()
