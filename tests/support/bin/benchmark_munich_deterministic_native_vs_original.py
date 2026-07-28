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
sys.path.insert(0, str(_REPO_ROOT))

# Local source must resolve from this checkout before importing the benchmark target.
from witwin.core import Scene  # noqa: E402
from tests.support.core_world import (  # noqa: E402
    make_receiver_grid,
    make_transmitter,
)
from witwin.channel.deterministic import Config, solve  # noqa: E402


DEFAULT_SIONNA_ROOT = pathlib.Path(
    "E:/Code/witwin-platform/channel/reference/sionna-rt-reference-2.0.1/src"
)
DEFAULT_MUNICH_XML = DEFAULT_SIONNA_ROOT / "sionna" / "rt" / "scenes" / "munich" / "munich.xml"
DEFAULT_CHANNEL_ROOT = pathlib.Path("E:/Code/witwin-platform/channel")
COMPONENTS = ("los", "reflection", "diffraction")


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


def _load_scene(args: argparse.Namespace) -> Scene:
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
    grid = make_receiver_grid(
        origin=torch.tensor([xmin + 0.5 * spacing_x, ymin + 0.5 * spacing_y, float(args.plane_z)], dtype=torch.float32),
        x_axis=torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32),
        y_axis=torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32),
        shape=(grid_size, grid_size),
        spacing=(spacing_x, spacing_y),
    )
    return Scene(
        structures=base.structures,
        transmitters=[make_transmitter(position=torch.tensor(args.tx, dtype=torch.float32), power_w=1.0)],
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
    parser.add_argument("--scene-xml", required=True)
    parser.add_argument("--sionna-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--grid-size", type=int, required=True)
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
    from witwin.channel.core.scene import ReceiverGrid, Scene, Transmitter
    from witwin.channel.core.scene.edge_policy import EdgePolicy
    from witwin.channel.deterministic import Config, Tuning, solve

    scene = Scene.load_mitsuba(
        pathlib.Path(args.scene_xml),
        device="cuda",
        merge_shapes=True,
        frequency=float(args.frequency),
        source_root=pathlib.Path(args.sionna_root),
    )
    scene.add(make_transmitter("tx", tuple(float(v) for v in args.tx)))
    scene.add(
        make_receiver_grid(
            "rm",
            axis="z",
            position=float(args.plane_z),
            bounds=((float(args.bounds_x[0]), float(args.bounds_x[1])), (float(args.bounds_y[0]), float(args.bounds_y[1]))),
            grid_shape=(int(args.grid_size), int(args.grid_size)),
        )
    )
    max_bounces = max(0, int(args.max_depth))
    config = Config(
        num_samples=1,
        max_bounces=max_bounces,
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
    )
    for _ in range(max(0, int(args.warmup_runs))):
        solve(scene=scene, transmitter="tx", receiver="rm", config=config)
        dr.sync_thread()

    start = time.perf_counter()
    result = solve(scene=scene, transmitter="tx", receiver="rm", config=config)
    dr.sync_thread()
    solve_time_ms = (time.perf_counter() - start) * 1000.0
    components = {
        name: torch.from_numpy(np.asarray(value, dtype=np.float32)).to(dtype=torch.float32)
        for name, value in dict(result.components).items()
    }
    payload = {
        "path_gain": torch.from_numpy(np.asarray(result.path_gain, dtype=np.float32)).to(dtype=torch.float32),
        "components": components,
        "metadata": {
            "max_bounces": max_bounces,
            "enable_rd_diffraction": bool(args.enable_rd_diffraction),
            "triangles": None if scene.tri_data is None else int(scene.tri_data["n_triangles"]),
            "diffraction_edges": int(scene.n_diffraction_edges),
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
        return {"available": False, "reason": f"channel root not found: {channel_root}"}, {}
    output_path = artifact_dir / "original_outputs.pt"
    command = [
        sys.executable,
        "-c",
        _original_worker_code(),
        "--channel-root",
        str(channel_root),
        "--scene-xml",
        str(args.scene_xml),
        "--sionna-root",
        str(args.sionna_root),
        "--output",
        str(output_path),
        "--grid-size",
        str(int(args.grid_size)),
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
        return {
            "available": False,
            "reason": "original subprocess timed out",
            "timeout_seconds": float(args.original_timeout_seconds),
            "stdout_tail": (exc.stdout or "")[-1000:] if isinstance(exc.stdout, str) else "",
            "stderr_tail": (exc.stderr or "")[-2000:] if isinstance(exc.stderr, str) else "",
        }, {}
    if completed.returncode != 0:
        return {
            "available": False,
            "reason": "original subprocess failed",
            "returncode": int(completed.returncode),
            "stderr_tail": completed.stderr[-2000:],
        }, {}
    loaded = torch.load(output_path, map_location="cpu", weights_only=False)
    original_total = loaded["path_gain"].to(dtype=torch.float32)
    if original_total.ndim == 3:
        original_total = original_total[0]
    original_components = {
        name: value.to(dtype=torch.float32)[0] if value.ndim == 3 else value.to(dtype=torch.float32)
        for name, value in loaded.get("components", {}).items()
        if name in COMPONENTS
    }
    tensors = {"path_gain": original_total.contiguous(), **original_components}
    return {
        "available": True,
        "metadata": loaded.get("metadata", {}),
        "wall_time_ms": float(wall_time_ms),
        "stdout_tail": completed.stdout[-1000:],
        "artifact": str(output_path),
    }, tensors


def _delta_metrics(native_total: torch.Tensor, original_total: torch.Tensor) -> tuple[torch.Tensor, dict[str, float | int]]:
    original_total = original_total.to(device=native_total.device)
    native_db = _db(native_total)
    original_db = _db(original_total)
    finite = torch.isfinite(native_db) & torch.isfinite(original_db)
    delta = torch.where(finite, native_db - original_db, torch.zeros_like(native_db))
    abs_delta = delta[finite].abs()
    if int(abs_delta.numel()) == 0:
        metrics: dict[str, float | int] = {"finite_count": 0, "max_abs_delta_db": float("nan"), "median_abs_delta_db": float("nan")}
    else:
        metrics = {
            "finite_count": int(abs_delta.numel()),
            "max_abs_delta_db": float(abs_delta.max().item()),
            "median_abs_delta_db": float(abs_delta.median().item()),
        }
    return delta, metrics


def _component_shapes(components: dict[str, torch.Tensor]) -> dict[str, list[int]]:
    return {name: list(components[name].shape) for name in COMPONENTS if name in components}


def _component_finite_counts(components: dict[str, torch.Tensor]) -> dict[str, int]:
    return {name: int(torch.isfinite(components[name]).sum().item()) for name in COMPONENTS if name in components}


def _component_delta_metrics(
    native_components: dict[str, torch.Tensor],
    original_components: dict[str, torch.Tensor],
) -> tuple[dict[str, torch.Tensor], dict[str, dict[str, float | int]]]:
    delta_maps: dict[str, torch.Tensor] = {}
    metrics: dict[str, dict[str, float | int]] = {}
    for name in COMPONENTS:
        if name not in native_components or name not in original_components:
            continue
        delta, component_metrics = _delta_metrics(native_components[name], original_components[name])
        delta_maps[name] = delta
        metrics[name] = component_metrics
    return delta_maps, metrics


def run(args: argparse.Namespace) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("Munich deterministic parity requires CUDA")
    if not pathlib.Path(args.scene_xml).exists():
        raise FileNotFoundError(f"Munich scene is not available: {args.scene_xml}")

    artifact_dir = pathlib.Path(args.artifact_dir).resolve()
    artifact_dir.mkdir(parents=True, exist_ok=True)
    original, original_tensors = _run_original(args, artifact_dir)
    if not original.get("available", False):
        raise RuntimeError(f"Original Munich deterministic solver is unavailable: {original}")

    scene = _load_scene(args)
    config = Config(
        max_depth=int(args.max_depth),
        max_diffraction_order=1,
        components=set(COMPONENTS),
        coherent=False,
        return_field=False,
        export_paths=True,
        diagnostics=True,
    )
    for _ in range(max(0, int(args.warmup_runs))):
        solve(scene, config)
        torch.cuda.synchronize()

    start = time.perf_counter()
    result = solve(scene, config)
    torch.cuda.synchronize()
    native_solve_time_ms = (time.perf_counter() - start) * 1000.0

    native_total = result.path_gain[0]
    native_components = {name: result.component_power[name][0] for name in COMPONENTS}
    original_total = original_tensors["path_gain"]
    if tuple(original_total.shape) != tuple(native_total.shape):
        raise RuntimeError(f"original/native shape mismatch: {tuple(original_total.shape)} vs {tuple(native_total.shape)}")
    delta_db, delta_metrics = _delta_metrics(native_total, original_total)
    component_delta_maps, component_delta_metrics = _component_delta_metrics(native_components, original_tensors)

    pngs = {
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
    for name, delta in component_delta_maps.items():
        _write_gray_png(pngs[f"delta_{name}_db"], delta)

    path_count_histogram = {}
    if result.paths is not None:
        for cid, name in enumerate(COMPONENTS):
            path_count_histogram[name] = int((result.paths.component_id == cid).sum().detach().cpu().item())

    payload = {
        "benchmark": "munich_deterministic_native_vs_original",
        "grid_size": int(args.grid_size),
        "max_depth": int(args.max_depth),
        "warmup_runs": int(args.warmup_runs),
        "tx": list(float(v) for v in args.tx),
        "bounds": {"x": list(args.bounds_x), "y": list(args.bounds_y), "z": float(args.plane_z)},
        "native": {
            "path_gain_shape": list(result.path_gain.shape),
            "metadata": result.metadata,
            "path_count_histogram": path_count_histogram,
            "component_shapes": _component_shapes(native_components),
            "component_finite_counts": _component_finite_counts(native_components),
        },
        "original": {
            **original,
            "component_shapes": _component_shapes(original_tensors),
            "component_finite_counts": _component_finite_counts(original_tensors),
        },
        "delta": delta_metrics,
        "component_delta": component_delta_metrics,
        "performance": {
            "native_solve_time_ms": float(native_solve_time_ms),
            "original_solve_time_ms": float(original.get("metadata", {}).get("solve_time_ms", float("nan"))),
            "original_wall_time_ms": float(original.get("wall_time_ms", float("nan"))),
        },
        "artifacts": {key: str(path) for key, path in pngs.items()},
    }
    json_path = artifact_dir / "metadata.json"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    payload["artifacts"]["metadata"] = str(json_path)
    return payload


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-dir", default=str(_REPO_ROOT / "artifacts" / "deterministic_munich"))
    parser.add_argument("--scene-xml", default=str(DEFAULT_MUNICH_XML))
    parser.add_argument("--sionna-root", default=str(DEFAULT_SIONNA_ROOT))
    parser.add_argument("--channel-root", default=str(DEFAULT_CHANNEL_ROOT))
    parser.add_argument("--grid-size", type=int, default=32)
    parser.add_argument("--max-depth", type=int, default=2)
    parser.add_argument("--frequency", type=float, default=2.4e9)
    parser.add_argument("--tx", type=float, nargs=3, default=(8.5, 21.0, 27.0))
    parser.add_argument("--bounds-x", type=float, nargs=2, default=(-120.0, 120.0))
    parser.add_argument("--bounds-y", type=float, nargs=2, default=(-120.0, 140.0))
    parser.add_argument("--plane-z", type=float, default=1.5)
    parser.add_argument("--original-timeout-seconds", type=float, default=600.0)
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--original-enable-rd-diffraction", action="store_true")
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
