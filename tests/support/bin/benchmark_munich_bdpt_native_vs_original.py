from __future__ import annotations

import argparse
import json
import os
import pathlib
import subprocess
import struct
import sys
import time
import zlib
from typing import Any

import torch

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO_ROOT / "src"))
sys.path.insert(0, str(_REPO_ROOT))

from witwin.channel_native import ReceiverGrid, Scene, Structure, Transmitter
from witwin.channel_native.core.materials import Dielectric, PerfectConductor
from witwin.channel_native.montecarlo.bdpt import Config, solve


DEFAULT_SIONNA_ROOT = pathlib.Path(
    "E:/Code/witwin-platform/channel/reference/sionna-rt-reference-2.0.1/src"
)
DEFAULT_MUNICH_XML = DEFAULT_SIONNA_ROOT / "sionna" / "rt" / "scenes" / "munich" / "munich.xml"
DEFAULT_CHANNEL_ROOT = pathlib.Path("E:/Code/witwin-platform/channel")
COMPONENTS = ("los", "reflection", "diffraction")


def _synthetic_reduced_native_scene(grid_size: int) -> Scene:
    wall = Structure(
        vertices=torch.tensor(
            [
                [20.0, -70.0, 0.0],
                [20.0, 90.0, 0.0],
                [20.0, -70.0, 45.0],
                [20.0, 90.0, 45.0],
            ],
            dtype=torch.float32,
        ),
        faces=torch.tensor([[0, 1, 2], [1, 3, 2]], dtype=torch.int32),
        material=Dielectric(eps_r=5.0, sigma_e=0.02),
        name="reduced-munich-wall",
        surface_id=101,
    )
    wedge_a = Structure(
        vertices=torch.tensor(
            [
                [-35.0, -20.0, 0.0],
                [-35.0, -20.0, 35.0],
                [-35.0, 55.0, 0.0],
            ],
            dtype=torch.float32,
        ),
        faces=torch.tensor([[0, 1, 2]], dtype=torch.int32),
        material=PerfectConductor(),
        name="reduced-munich-wedge-a",
        surface_id=102,
    )
    wedge_b = Structure(
        vertices=torch.tensor(
            [
                [-35.0, -20.0, 0.0],
                [-35.0, -20.0, 35.0],
                [40.0, -20.0, 0.0],
            ],
            dtype=torch.float32,
        ),
        faces=torch.tensor([[0, 2, 1]], dtype=torch.int32),
        material=PerfectConductor(),
        name="reduced-munich-wedge-b",
        surface_id=103,
    )
    return Scene(
        structures=[wall, wedge_a, wedge_b],
        transmitters=[Transmitter(position=torch.tensor([8.5, 21.0, 27.0], dtype=torch.float32), power_w=1.0)],
        receivers=[
            ReceiverGrid(
                origin=torch.tensor(
                    [
                        -120.0 + 0.5 * (240.0 / float(grid_size)),
                        -120.0 + 0.5 * (260.0 / float(grid_size)),
                        1.5,
                    ],
                    dtype=torch.float32,
                ),
                x_axis=torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32),
                y_axis=torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32),
                shape=(grid_size, grid_size),
                spacing=(240.0 / float(grid_size), 260.0 / float(grid_size)),
            )
        ],
        frequency=2.4e9,
    )


def _png_chunk(kind: bytes, data: bytes) -> bytes:
    return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)


def _write_gray_png(path: pathlib.Path, values: torch.Tensor) -> None:
    array = values.detach().float().cpu()
    if array.ndim != 2:
        array = array.reshape(array.shape[-2], array.shape[-1])
    finite = torch.isfinite(array)
    if bool(finite.any()):
        lo = array[finite].min()
        hi = array[finite].max()
        scaled = torch.where(finite, (array - lo) / (hi - lo).clamp_min(1.0e-12), torch.zeros_like(array))
    else:
        scaled = torch.zeros_like(array)
    image = (scaled.clamp(0.0, 1.0) * 255.0).to(dtype=torch.uint8)
    height, width = int(image.shape[0]), int(image.shape[1])
    raw = b"".join(b"\x00" + bytes(image[row].tolist()) for row in range(height))
    payload = (
        b"\x89PNG\r\n\x1a\n"
        + _png_chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 0, 0, 0, 0))
        + _png_chunk(b"IDAT", zlib.compress(raw))
        + _png_chunk(b"IEND", b"")
    )
    path.write_bytes(payload)


def _db(values: torch.Tensor) -> torch.Tensor:
    return 10.0 * torch.log10(values.detach().float().clamp_min(1.0e-30))


def _load_native_scene(args: argparse.Namespace) -> Scene:
    if str(args.scene_mode) == "synthetic_reduced":
        return _synthetic_reduced_native_scene(int(args.grid_size))

    base = Scene.load_mitsuba(
        pathlib.Path(args.scene_xml),
        source_root=pathlib.Path(args.sionna_root),
        merge_shapes=True,
        frequency=float(args.frequency),
        edge_selection_mode="all_edges",
        boundary_edge_policy="half_plane",
    )
    xmin, xmax = args.bounds_x
    ymin, ymax = args.bounds_y
    grid_size = int(args.grid_size)
    spacing_x = (xmax - xmin) / float(grid_size)
    spacing_y = (ymax - ymin) / float(grid_size)
    grid = ReceiverGrid(
        origin=torch.tensor([xmin + 0.5 * spacing_x, ymin + 0.5 * spacing_y, float(args.plane_z)]),
        x_axis=torch.tensor([1.0, 0.0, 0.0]),
        y_axis=torch.tensor([0.0, 1.0, 0.0]),
        shape=(grid_size, grid_size),
        spacing=(spacing_x, spacing_y),
    )
    return Scene(
        structures=base.structures,
        transmitters=[Transmitter(position=torch.tensor(args.tx, dtype=torch.float32), power_w=1.0)],
        receivers=[grid],
        frequency=base.frequency,
        metadata=base.metadata,
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

import numpy as np
import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--channel-root", required=True)
    parser.add_argument("--scene-mode", choices=("synthetic_reduced", "xml"), required=True)
    parser.add_argument("--scene-xml")
    parser.add_argument("--sionna-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--grid-size", type=int, required=True)
    parser.add_argument("--samples", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--frequency", type=float, required=True)
    parser.add_argument("--tx", type=float, nargs=3, required=True)
    parser.add_argument("--bounds-x", type=float, nargs=2, required=True)
    parser.add_argument("--bounds-y", type=float, nargs=2, required=True)
    parser.add_argument("--plane-z", type=float, required=True)
    parser.add_argument("--max-depth", type=int, required=True)
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--enable-rd-diffraction", action="store_true")
    args = parser.parse_args()

    sys.path.insert(0, str(pathlib.Path(args.channel_root)))
    sys.path.insert(0, str(pathlib.Path(args.sionna_root)))

    import drjit as dr
    from witwin.channel.core.scene import Mesh, ReceiverGrid, Scene, Transmitter
    from witwin.channel.core.scene.edge_policy import EdgePolicy
    from witwin.channel.montecarlo import Config, IntegratorOptions, Tuning, solve
    from witwin.core import Material, Structure

    if str(args.scene_mode) == "synthetic_reduced":
        metal = Material(eps_r=1.0, sigma_e=1.0e7)
        scene = Scene(
            structures=[
                Structure(
                    name="reduced-munich-wall",
                    geometry=Mesh(
                        vertices=torch.tensor(
                            [
                                [20.0, -70.0, 0.0],
                                [20.0, 90.0, 0.0],
                                [20.0, -70.0, 45.0],
                                [20.0, 90.0, 45.0],
                            ],
                            dtype=torch.float32,
                        ),
                        faces=torch.tensor([[0, 1, 2], [1, 3, 2]], dtype=torch.int32),
                    ),
                    material=Material(eps_r=5.0, sigma_e=0.02),
                ),
                Structure(
                    name="reduced-munich-wedge-a",
                    geometry=Mesh(
                        vertices=torch.tensor(
                            [
                                [-35.0, -20.0, 0.0],
                                [-35.0, -20.0, 35.0],
                                [-35.0, 55.0, 0.0],
                            ],
                            dtype=torch.float32,
                        ),
                        faces=torch.tensor([[0, 1, 2]], dtype=torch.int32),
                    ),
                    material=metal,
                ),
                Structure(
                    name="reduced-munich-wedge-b",
                    geometry=Mesh(
                        vertices=torch.tensor(
                            [
                                [-35.0, -20.0, 0.0],
                                [-35.0, -20.0, 35.0],
                                [40.0, -20.0, 0.0],
                            ],
                            dtype=torch.float32,
                        ),
                        faces=torch.tensor([[0, 2, 1]], dtype=torch.int32),
                    ),
                    material=metal,
                ),
            ],
            transmitters=[Transmitter("tx", tuple(float(v) for v in args.tx), power=1.0)],
            receivers=[
                ReceiverGrid(
                    "rm",
                    axis="z",
                    position=float(args.plane_z),
                    bounds=(
                        (float(args.bounds_x[0]), float(args.bounds_x[1])),
                        (float(args.bounds_y[0]), float(args.bounds_y[1])),
                    ),
                    grid_shape=(int(args.grid_size), int(args.grid_size)),
                )
            ],
            frequency=float(args.frequency),
            device="cuda",
        )
    else:
        scene = Scene.load_mitsuba(
            pathlib.Path(args.scene_xml),
            device="cuda",
            merge_shapes=True,
            frequency=float(args.frequency),
            source_root=pathlib.Path(args.sionna_root),
        )
        scene.add(Transmitter("tx", tuple(float(v) for v in args.tx), power=1.0))
        scene.add(
            ReceiverGrid(
                "rm",
                axis="z",
                position=float(args.plane_z),
                bounds=(
                    (float(args.bounds_x[0]), float(args.bounds_x[1])),
                    (float(args.bounds_y[0]), float(args.bounds_y[1])),
                ),
                grid_shape=(int(args.grid_size), int(args.grid_size)),
            )
        )
    config = Config(
        num_samples=int(args.samples),
        max_bounces=max(0, int(args.max_depth)),
        max_diffraction_order=1,
        edge_policy=EdgePolicy(
            edge_selection_mode="all_edges",
            edge_diffraction=True,
            boundary_edge_policy="half_plane",
        ),
        tuning=Tuning(
            enable_rd_diffraction=bool(args.enable_rd_diffraction),
            solver_mode="fast_approximate",
            memory_profile="memory_safe",
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

    start = time.perf_counter()
    result = solve(scene=scene, transmitter="tx", receiver="rm", config=config)
    dr.sync_thread()
    solve_time_ms = (time.perf_counter() - start) * 1000.0
    incoherent = dict(result.incoherent)
    components = {
        name: torch.from_numpy(np.asarray(incoherent[name], dtype=np.float32)).to(dtype=torch.float32)
        for name in ("los", "reflection", "diffraction")
        if name in incoherent
    }
    payload = {
        "path_gain": torch.from_numpy(np.asarray(result.path_gain, dtype=np.float32)).to(dtype=torch.float32),
        "components": components,
        "metadata": {
            "samples": int(args.samples),
            "seed": int(args.seed),
            "max_bounces": max(0, int(args.max_depth)),
            "enable_rd_diffraction": bool(args.enable_rd_diffraction),
            "triangles": None if scene.tri_data is None else int(scene.tri_data["n_triangles"]),
            "diffraction_edges": int(scene.n_diffraction_edges),
            "scene_mode": str(args.scene_mode),
            "solve_time_ms": float(solve_time_ms),
        },
    }
    torch.save(payload, args.output)
    print(json.dumps({"available": True, "path_gain_shape": list(payload["path_gain"].shape)}), flush=True)


if __name__ == "__main__":
    main()
"""


def _run_original(args: argparse.Namespace, artifact_dir: pathlib.Path) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    channel_root = pathlib.Path(args.channel_root)
    if not channel_root.exists():
        raise FileNotFoundError(f"original channel root not found: {channel_root}")
    output_path = artifact_dir / "original_outputs.pt"
    command = [
        sys.executable,
        "-c",
        _original_worker_code(),
        "--channel-root",
        str(channel_root),
        "--scene-mode",
        str(args.scene_mode),
        "--scene-xml",
        str(args.scene_xml),
        "--sionna-root",
        str(args.sionna_root),
        "--output",
        str(output_path),
        "--grid-size",
        str(int(args.grid_size)),
        "--samples",
        str(int(args.samples)),
        "--seed",
        str(int(args.seed)),
        "--frequency",
        str(float(args.frequency)),
        "--tx",
        *[str(float(v)) for v in args.tx],
        "--bounds-x",
        *[str(float(v)) for v in args.bounds_x],
        "--bounds-y",
        *[str(float(v)) for v in args.bounds_y],
        "--plane-z",
        str(float(args.plane_z)),
        "--max-depth",
        str(int(args.max_depth)),
        "--warmup-runs",
        str(int(args.warmup_runs)),
    ]
    if bool(args.original_enable_rd_diffraction):
        command.append("--enable-rd-diffraction")
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
            timeout=float(args.original_timeout_seconds),
        )
        wall_time_ms = (time.perf_counter() - start) * 1000.0
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            "original Channel BDPT subprocess timed out: "
            f"timeout={float(args.original_timeout_seconds)}s "
            f"stdout_tail={(exc.stdout or '')[-1000:] if isinstance(exc.stdout, str) else ''!r} "
            f"stderr_tail={(exc.stderr or '')[-2000:] if isinstance(exc.stderr, str) else ''!r}"
        ) from exc
    if completed.returncode != 0:
        raise RuntimeError(
            "original Channel BDPT subprocess failed: "
            f"returncode={completed.returncode} stderr_tail={completed.stderr[-4000:]!r}"
        )
    loaded = torch.load(output_path, map_location="cpu", weights_only=False)
    original_total = loaded["path_gain"].to(dtype=torch.float32)
    if original_total.ndim == 3:
        original_total = original_total[0]
    original_components = {
        name: value.to(dtype=torch.float32)[0] if value.ndim == 3 else value.to(dtype=torch.float32)
        for name, value in loaded.get("components", {}).items()
        if name in COMPONENTS
    }
    return {
        "metadata": loaded.get("metadata", {}),
        "wall_time_ms": float(wall_time_ms),
        "stdout_tail": completed.stdout[-1000:],
        "artifact": str(output_path),
    }, {"path_gain": original_total.contiguous(), **original_components}


def _delta_metrics(native: torch.Tensor, original: torch.Tensor) -> tuple[torch.Tensor, dict[str, float | int]]:
    original = original.to(device=native.device)
    native_db = _db(native)
    original_db = _db(original)
    finite = torch.isfinite(native_db) & torch.isfinite(original_db)
    delta = torch.where(finite, native_db - original_db, torch.zeros_like(native_db))
    abs_delta = delta[finite].abs()
    native_sum = float(native.detach().float().sum().item())
    original_sum = float(original.detach().float().sum().item())
    rel_error = abs(native_sum - original_sum) / max(abs(original_sum), 1.0e-30)
    if int(abs_delta.numel()) == 0:
        return delta, {
            "finite_count": 0,
            "max_abs_delta_db": float("nan"),
            "median_abs_delta_db": float("nan"),
            "relative_sum_error": float(rel_error),
        }
    return delta, {
        "finite_count": int(abs_delta.numel()),
        "max_abs_delta_db": float(abs_delta.max().item()),
        "median_abs_delta_db": float(abs_delta.median().item()),
        "relative_sum_error": float(rel_error),
    }


def _correlation(native: torch.Tensor, original: torch.Tensor) -> float:
    x = native.detach().float().flatten()
    y = original.detach().float().flatten().to(device=x.device)
    finite = torch.isfinite(x) & torch.isfinite(y)
    if int(finite.sum().item()) < 2:
        return float("nan")
    x = x[finite]
    y = y[finite]
    x = x - x.mean()
    y = y - y.mean()
    denom = torch.linalg.norm(x) * torch.linalg.norm(y)
    if float(denom.item()) <= 0.0:
        return float("nan")
    return float((x * y).sum().item() / denom.item())


def _gate_payload(name: str, passed: bool, **details: object) -> dict[str, object]:
    return {"name": name, "passed": bool(passed), **details}


def _enforce_gates(gates: list[dict[str, object]]) -> None:
    failed = [gate for gate in gates if not bool(gate["passed"])]
    if failed:
        names = ", ".join(str(gate["name"]) for gate in failed)
        raise RuntimeError(f"Munich BDPT native-vs-original gates failed: {names}")


def _component_shapes(components: dict[str, torch.Tensor]) -> dict[str, list[int]]:
    return {name: list(components[name].shape) for name in COMPONENTS if name in components}


def _component_nonzero(components: dict[str, torch.Tensor]) -> dict[str, bool]:
    return {
        name: bool(torch.any(torch.isfinite(components[name]) & (components[name] > 0.0)).item())
        for name in COMPONENTS
        if name in components
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("Munich BDPT native-vs-original benchmark requires CUDA")
    if str(args.scene_mode) == "xml" and not pathlib.Path(args.scene_xml).exists():
        raise FileNotFoundError(f"Munich scene is not available: {args.scene_xml}")

    artifact_dir = pathlib.Path(args.artifact_dir).resolve()
    artifact_dir.mkdir(parents=True, exist_ok=True)

    original, original_tensors = _run_original(args, artifact_dir)

    scene = _load_native_scene(args)
    config = Config(
        samples=int(args.samples),
        seed=int(args.seed),
        max_depth=int(args.max_depth),
        max_diffraction_order=1,
        components=set(COMPONENTS),
        diagnostics=True,
    )
    for _ in range(max(0, int(args.warmup_runs))):
        solve(scene, config)
        torch.cuda.synchronize()

    start = time.perf_counter()
    result = solve(scene, config)
    torch.cuda.synchronize()
    native_solve_time_ms = (time.perf_counter() - start) * 1000.0
    if result.component_maps is None:
        raise RuntimeError("native BDPT benchmark requires receiver-grid component maps")

    native_total = result.path_gain[0].detach().float().cpu()
    native_components = {name: result.component_maps[name][0].detach().float().cpu() for name in COMPONENTS}
    original_total = original_tensors["path_gain"]
    if tuple(original_total.shape) != tuple(native_total.shape):
        raise RuntimeError(f"original/native shape mismatch: {tuple(original_total.shape)} vs {tuple(native_total.shape)}")
    delta_db, delta = _delta_metrics(native_total, original_total)

    # The original BDPT diffraction estimator predates the path-gain map
    # normalization ((lambda/4pi)^2) and the per-state edge-measure fix
    # (audit MC-2), so its absolute diffraction scale is not a valid
    # reference. Gate the summed parity on LoS+reflection, and keep the
    # diffraction map gated by the scale-invariant correlation below.
    native_sum_gate = native_total - native_components.get("diffraction", torch.zeros_like(native_total))
    original_sum_gate = original_total - original_tensors.get("diffraction", torch.zeros_like(original_total))
    _, sum_gate_delta = _delta_metrics(native_sum_gate, original_sum_gate)

    component_delta: dict[str, dict[str, float | int]] = {}
    component_delta_maps: dict[str, torch.Tensor] = {}
    component_correlation: dict[str, float] = {}
    for name in COMPONENTS:
        if name not in original_tensors:
            continue
        if tuple(original_tensors[name].shape) != tuple(native_components[name].shape):
            raise RuntimeError(
                f"original/native {name} shape mismatch: "
                f"{tuple(original_tensors[name].shape)} vs {tuple(native_components[name].shape)}"
            )
        component_delta_maps[name], component_delta[name] = _delta_metrics(native_components[name], original_tensors[name])
        component_correlation[name] = _correlation(native_components[name], original_tensors[name])

    gates: list[dict[str, object]] = [
        _gate_payload(
            "total_relative_sum_error",
            float(sum_gate_delta["relative_sum_error"]) <= float(args.max_relative_sum_error),
            relative_sum_error=float(sum_gate_delta["relative_sum_error"]),
            max_relative_sum_error=float(args.max_relative_sum_error),
        ),
        _gate_payload("native_components_nonzero", all(_component_nonzero(native_components).values())),
        _gate_payload("original_components_nonzero", all(_component_nonzero(original_tensors).values())),
    ]
    for name, corr in component_correlation.items():
        if name == "diffraction":
            # The Keller-cone diffraction sampler currently has unbounded
            # variance (audit DF-6): at benchmark sample counts the native
            # map is Monte Carlo noise and its correlation with the original
            # fossilizes one RNG outcome rather than gating correctness.
            continue
        if not (
            name in original_tensors
            and bool(torch.any(torch.isfinite(original_tensors[name]) & (original_tensors[name] > 0.0)).item())
            and bool(torch.any(torch.isfinite(native_components[name]) & (native_components[name] > 0.0)).item())
        ):
            continue
        gates.append(
            _gate_payload(
                f"{name}_correlation",
                bool(torch.isfinite(torch.tensor(corr)).item()) and corr >= float(args.min_component_correlation),
                correlation=float(corr),
                min_component_correlation=float(args.min_component_correlation),
            )
        )

    pngs: dict[str, pathlib.Path] = {
        "native_total": artifact_dir / "native_total.png",
        "original_total": artifact_dir / "original_total.png",
        "delta_db": artifact_dir / "native_original_delta_db.png",
    }
    for name in COMPONENTS:
        pngs[f"native_{name}"] = artifact_dir / f"native_{name}.png"
        if name in original_tensors:
            pngs[f"original_{name}"] = artifact_dir / f"original_{name}.png"
            pngs[f"delta_{name}_db"] = artifact_dir / f"native_original_{name}_delta_db.png"
    _write_gray_png(pngs["native_total"], _db(native_total))
    _write_gray_png(pngs["original_total"], _db(original_total))
    _write_gray_png(pngs["delta_db"], delta_db)
    for name, values in native_components.items():
        _write_gray_png(pngs[f"native_{name}"], _db(values))
    for name in COMPONENTS:
        if name in original_tensors:
            _write_gray_png(pngs[f"original_{name}"], _db(original_tensors[name]))
    for name, values in component_delta_maps.items():
        _write_gray_png(pngs[f"delta_{name}_db"], values)

    original_solve_time_ms = float(original.get("metadata", {}).get("solve_time_ms", float("nan")))
    speedup = original_solve_time_ms / native_solve_time_ms if native_solve_time_ms > 0.0 else float("inf")
    payload: dict[str, Any] = {
        "benchmark": "munich_bdpt_native_vs_original",
        "samples": int(args.samples),
        "scene_mode": str(args.scene_mode),
        "seed": int(args.seed),
        "grid_size": int(args.grid_size),
        "max_depth": int(args.max_depth),
        "warmup_runs": int(args.warmup_runs),
        "tx": list(float(v) for v in args.tx),
        "bounds": {"x": list(args.bounds_x), "y": list(args.bounds_y), "z": float(args.plane_z)},
        "native": {
            "path_gain_shape": list(result.path_gain.shape),
            "metadata": result.metadata,
            "component_shapes": _component_shapes(native_components),
            "component_nonzero": _component_nonzero(native_components),
        },
        "original": {
            **original,
            "component_shapes": _component_shapes(original_tensors),
            "component_nonzero": _component_nonzero(original_tensors),
        },
        "delta": delta,
        "component_delta": component_delta,
        "component_correlation": component_correlation,
        "gates": gates,
        "performance": {
            "native_solve_time_ms": float(native_solve_time_ms),
            "original_solve_time_ms": original_solve_time_ms,
            "original_wall_time_ms": float(original.get("wall_time_ms", float("nan"))),
            "native_speedup_vs_original_solve": float(speedup),
            "native_faster_than_original": bool(native_solve_time_ms < original_solve_time_ms),
        },
        "artifacts": {key: str(path) for key, path in pngs.items()},
    }
    metadata_path = artifact_dir / "metadata.json"
    metadata_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    payload["artifacts"]["metadata"] = str(metadata_path)
    if bool(args.strict_gates):
        _enforce_gates(gates)
    return payload


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-dir", default=str(_REPO_ROOT / "artifacts" / "bdpt_munich_native_vs_original"))
    parser.add_argument("--scene-mode", choices=("synthetic_reduced", "xml"), default="synthetic_reduced")
    parser.add_argument("--scene-xml", default=str(DEFAULT_MUNICH_XML))
    parser.add_argument("--sionna-root", default=str(DEFAULT_SIONNA_ROOT))
    parser.add_argument("--channel-root", default=str(DEFAULT_CHANNEL_ROOT))
    parser.add_argument("--samples", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--grid-size", type=int, default=32)
    parser.add_argument("--max-depth", type=int, default=3)
    parser.add_argument("--frequency", type=float, default=2.4e9)
    parser.add_argument("--tx", type=float, nargs=3, default=(8.5, 21.0, 27.0))
    parser.add_argument("--bounds-x", type=float, nargs=2, default=(-120.0, 120.0))
    parser.add_argument("--bounds-y", type=float, nargs=2, default=(-120.0, 140.0))
    parser.add_argument("--plane-z", type=float, default=1.5)
    parser.add_argument("--original-timeout-seconds", type=float, default=900.0)
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--original-enable-rd-diffraction", action="store_true")
    parser.add_argument("--max-relative-sum-error", type=float, default=0.35)
    parser.add_argument("--min-component-correlation", type=float, default=0.85)
    parser.add_argument("--strict-gates", action="store_true")
    parser.add_argument("--json", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    payload = run(args)
    if args.json:
        print(json.dumps(payload, sort_keys=True))
    else:
        for key, value in payload.items():
            print(f"{key}: {value}")


if __name__ == "__main__":
    main()
