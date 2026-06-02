"""Profile the standalone deterministic solver on the bundled Munich scene."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Any


DEFAULT_FREQUENCY_HZ = 3.5e9
DEFAULT_TX_POS = (8.5, 21.0, 27.0)
DEFAULT_PLANE_Z = 1.5
DEFAULT_BOUNDS = ((-120.0, 120.0), (-120.0, 140.0))
DEFAULT_OUTPUT_JSON = (
    Path(__file__).resolve().parents[2] / "output" / "deterministic_munich_profile.json"
)


def _repo_root() -> Path:
    path = Path.cwd().resolve()
    if (path / "witwin").exists():
        return path
    return next(parent for parent in path.parents if (parent / "witwin").exists())


def _default_munich_xml(root: Path) -> Path:
    return root / "sionna-rt-reference-2.0.0" / "src" / "sionna" / "rt" / "scenes" / "munich" / "munich.xml"


def _git_common_repo_root() -> Path | None:
    try:
        output = subprocess.check_output(
            ["git", "rev-parse", "--git-common-dir"],
            text=True,
            cwd=str(_repo_root()),
        ).strip()
    except Exception:
        return None
    common_dir = Path(output)
    if not common_dir.is_absolute():
        common_dir = (_repo_root() / common_dir).resolve()
    if common_dir.name == ".git":
        candidate = common_dir.parent
        if (candidate / "witwin").exists():
            return candidate
    return None


def _resolve_munich_xml(repo_root: Path, explicit: Path | None = None) -> Path:
    if explicit is not None:
        return explicit
    candidates = [_default_munich_xml(repo_root)]
    common_root = _git_common_repo_root()
    if common_root is not None and common_root != repo_root:
        candidates.append(_default_munich_xml(common_root))
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def _sionna_source_root_from_xml(munich_xml: Path) -> Path | None:
    for parent in munich_xml.parents:
        if parent.name == "src" and (parent / "sionna" / "rt").exists():
            return parent
        if parent.name == "src" and str(munich_xml).replace("\\", "/").endswith(
            "src/sionna/rt/scenes/munich/munich.xml"
        ):
            return parent
    return None


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def gpu_memory_mib() -> dict[str, Any]:
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,memory.used",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        ).strip()
    except Exception as exc:
        return {"available": False, "error": str(exc)}
    name, total, used = [part.strip() for part in output.split(",", 2)]
    return {
        "available": True,
        "name": name,
        "total_mib": int(total),
        "used_mib": int(used),
    }


def _value_stats(values) -> dict[str, Any]:
    import numpy as np

    array = np.asarray(values, dtype=np.float64)
    flat = array.reshape(-1)
    if flat.size == 0:
        return {
            "shape": list(array.shape),
            "min": 0.0,
            "max": 0.0,
            "sum": 0.0,
            "nonzero": 0,
            "finite": True,
        }
    return {
        "shape": list(array.shape),
        "min": float(np.nanmin(flat)),
        "max": float(np.nanmax(flat)),
        "sum": float(np.nansum(flat, dtype=np.float64)),
        "nonzero": int(np.count_nonzero(flat > 0.0)),
        "finite": bool(np.isfinite(flat).all()),
    }


def _kernel_history_summary(history: list[dict[str, Any]] | None) -> dict[str, Any] | None:
    if history is None:
        return None
    by_type: dict[str, dict[str, Any]] = {}
    for entry in history:
        type_name = str(entry.get("type", "unknown"))
        bucket = by_type.setdefault(
            type_name,
            {"count": 0, "size_sum": 0, "size_max": 0, "execution_time_ms_sum": 0.0},
        )
        size = int(entry.get("size", 0) or 0)
        bucket["count"] += 1
        bucket["size_sum"] += size
        bucket["size_max"] = max(bucket["size_max"], size)
        bucket["execution_time_ms_sum"] += float(entry.get("execution_time", 0.0) or 0.0)
    return {
        "total_count": len(history),
        "by_type": by_type,
    }


def _used_mib(snapshot: dict[str, Any]) -> int | None:
    if "used_mib" not in snapshot:
        return None
    return int(snapshot["used_mib"])


def build_summary(
    *,
    environment: dict[str, Any],
    scenario: dict[str, Any],
    memory: dict[str, Any],
    timing: dict[str, float],
    result_summary: dict[str, Any],
    kernel_history: list[dict[str, Any]] | None,
) -> dict[str, Any]:
    memory_used = [
        value
        for value in (_used_mib(item) for item in memory.values() if isinstance(item, dict))
        if value is not None
    ]
    first_used = memory_used[0] if memory_used else None
    peak_used = max(memory_used) if memory_used else None
    memory_summary = dict(memory)
    memory_summary["peak_used_mib"] = peak_used
    memory_summary["peak_delta_mib"] = (
        None if first_used is None or peak_used is None else int(peak_used - first_used)
    )
    return _jsonable(
        {
            "environment": environment,
            "scenario": scenario,
            "memory": memory_summary,
            "timing": timing,
            "result": result_summary,
            "kernel_history": _kernel_history_summary(kernel_history),
        }
    )


def validate_summary(
    summary: Mapping[str, Any],
    *,
    assert_peak_used_mib_below: int | None = None,
    assert_finite: bool = False,
) -> None:
    if assert_peak_used_mib_below is not None:
        peak_used_mib = summary.get("memory", {}).get("peak_used_mib")
        if peak_used_mib is None:
            raise AssertionError("peak GPU memory is unavailable in profiler summary.")
        if int(peak_used_mib) > int(assert_peak_used_mib_below):
            raise AssertionError(
                "peak GPU memory exceeded gate: "
                f"{int(peak_used_mib)} MiB > {int(assert_peak_used_mib_below)} MiB."
            )
    if assert_finite:
        path_gain = summary.get("result", {}).get("path_gain", {})
        if not bool(path_gain.get("finite", False)):
            raise AssertionError("path_gain contains non-finite values.")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid-size", type=int, default=512)
    parser.add_argument("--max-diffractions", type=int, default=1)
    parser.add_argument(
        "--shadow-boundary-correction",
        dest="shadow_boundary_correction",
        action="store_true",
    )
    parser.add_argument(
        "--no-shadow-boundary-correction",
        dest="shadow_boundary_correction",
        action="store_false",
    )
    parser.set_defaults(shadow_boundary_correction=False)
    parser.add_argument(
        "--edge-selection-mode",
        choices=("all_edges", "vertical_only"),
        default="all_edges",
    )
    parser.add_argument(
        "--boundary-edge-policy",
        choices=("exclude", "half_plane"),
        default="exclude",
    )
    parser.add_argument("--reflection-n-rays", type=int, default=256)
    parser.add_argument("--reflection-max-bounces", type=int, default=1)
    parser.add_argument("--reflection-coef", type=float, default=0.8)
    parser.add_argument(
        "--shadow-boundary-backend",
        choices=("auto", "dense_native", "native_candidate"),
        default="auto",
    )
    parser.add_argument(
        "--shadow-boundary-tile-shape",
        type=int,
        nargs=2,
        metavar=("NX", "NY"),
        default=(8, 8),
    )
    parser.add_argument(
        "--shadow-boundary-band-width-wavelengths",
        type=float,
        default=3.0,
    )
    parser.add_argument("--shadow-boundary-max-candidate-factor", type=float, default=96.0)
    parser.add_argument(
        "--solver-mode",
        choices=("accuracy", "fast_approximate"),
        default="accuracy",
    )
    parser.add_argument(
        "--memory-profile",
        choices=("default", "memory_safe"),
        default="default",
    )
    parser.add_argument(
        "--diffraction-accumulate-primal",
        choices=("auto", "drjit", "rayd_optix", "rayd_exact_coherent"),
        default="auto",
    )
    parser.add_argument("--frequency-hz", type=float, default=DEFAULT_FREQUENCY_HZ)
    parser.add_argument("--plane-z", type=float, default=DEFAULT_PLANE_Z)
    parser.add_argument("--tx-x", type=float, default=DEFAULT_TX_POS[0])
    parser.add_argument("--tx-y", type=float, default=DEFAULT_TX_POS[1])
    parser.add_argument("--tx-z", type=float, default=DEFAULT_TX_POS[2])
    parser.add_argument("--xmin", type=float, default=DEFAULT_BOUNDS[0][0])
    parser.add_argument("--xmax", type=float, default=DEFAULT_BOUNDS[0][1])
    parser.add_argument("--ymin", type=float, default=DEFAULT_BOUNDS[1][0])
    parser.add_argument("--ymax", type=float, default=DEFAULT_BOUNDS[1][1])
    parser.add_argument("--munich-xml", type=Path, default=None)
    parser.add_argument("--kernel-history", action="store_true")
    parser.add_argument("--assert-peak-used-mib-below", type=int, default=None)
    parser.add_argument("--assert-finite", action="store_true")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--output-json", type=Path, default=None)
    return parser.parse_args(argv)


def run_profile(args: argparse.Namespace) -> dict[str, Any]:
    repo_root = _repo_root()
    repo_root_str = str(repo_root)
    sys.path = [
        path for path in sys.path if str(Path(path or ".").resolve()) != repo_root_str
    ]
    sys.path.insert(0, repo_root_str)

    import drjit as dr
    import numpy as np
    from witwin.channel.core.scene import ReceiverGrid, Scene, Transmitter
    from witwin.channel.core.scene.edge_policy import EdgePolicy
    from witwin.channel.deterministic import Config, Tuning, native_extension_available, solve
    from witwin.channel.deterministic.config import resolve_solver_controls

    munich_xml = _resolve_munich_xml(repo_root, args.munich_xml)
    bounds = ((float(args.xmin), float(args.xmax)), (float(args.ymin), float(args.ymax)))
    grid_shape = (int(args.grid_size), int(args.grid_size))
    tx_pos = (float(args.tx_x), float(args.tx_y), float(args.tx_z))
    total_start = time.perf_counter()
    memory = {"before_scene_load": gpu_memory_mib()}

    scene_start = time.perf_counter()
    scene = Scene.load_mitsuba(
        munich_xml,
        device="cuda",
        merge_shapes=True,
        frequency=float(args.frequency_hz),
        source_root=_sionna_source_root_from_xml(munich_xml),
    )
    scene.add(Transmitter("tx", tx_pos))
    scene.add(
        ReceiverGrid(
            "rm",
            axis="z",
            position=float(args.plane_z),
            bounds=bounds,
            grid_shape=grid_shape,
        )
    )
    dr.sync_thread()
    scene_load_seconds = time.perf_counter() - scene_start
    memory["after_scene_load"] = gpu_memory_mib()

    config = Config(
        num_samples=int(args.reflection_n_rays),
        max_bounces=int(args.reflection_max_bounces),
        max_diffraction_order=int(args.max_diffractions),
        shadow_boundary_correction=bool(args.shadow_boundary_correction),
        edge_policy=EdgePolicy(
            edge_selection_mode=str(args.edge_selection_mode),
            edge_diffraction=True,
            boundary_edge_policy=str(args.boundary_edge_policy),
        ),
        tuning=Tuning(
            shadow_boundary_backend=str(args.shadow_boundary_backend),
            shadow_boundary_tile_shape=tuple(
                int(value) for value in args.shadow_boundary_tile_shape
            ),
            shadow_boundary_band_width_wavelengths=float(
                args.shadow_boundary_band_width_wavelengths
            ),
            shadow_boundary_max_candidate_factor=float(args.shadow_boundary_max_candidate_factor),
            enable_rd_diffraction=int(args.max_diffractions) > 0,
            solver_mode=str(args.solver_mode),
            memory_profile=str(args.memory_profile),
            diffraction_execution={
                "accumulate_primal": str(args.diffraction_accumulate_primal),
            },
        ),
    )

    history = None
    solve_start = time.perf_counter()
    if bool(args.kernel_history):
        with dr.scoped_set_flag(dr.JitFlag.KernelHistory, True):
            dr.kernel_history_clear()
            result = solve(
                scene=scene,
                transmitter="tx",
                receiver="rm",
                config=config,
            )
            dr.sync_thread()
            history = list(dr.kernel_history())
    else:
        result = solve(
            scene=scene,
            transmitter="tx",
            receiver="rm",
            config=config,
        )
        dr.sync_thread()
    solve_seconds = time.perf_counter() - solve_start
    memory["after_solve"] = gpu_memory_mib()

    materialize_start = time.perf_counter()
    path_gain = np.asarray(result.path_gain, dtype=np.float64)
    component_summary = {
        name: _value_stats(values)
        for name, values in dict(result.components).items()
    }
    result_materialization_seconds = time.perf_counter() - materialize_start
    memory["after_result_materialization"] = gpu_memory_mib()

    scenario = {
        "scene_path": munich_xml,
        "frequency_hz": float(args.frequency_hz),
        "tx_pos": tx_pos,
        "bounds": bounds,
        "grid_shape": grid_shape,
        "edge_selection_mode": str(args.edge_selection_mode),
        "boundary_edge_policy": str(config.edge_policy.boundary_edge_policy),
        "triangles": None if scene.tri_data is None else int(scene.tri_data["n_triangles"]),
        "diffraction_edges": int(scene.n_diffraction_edges),
        "num_samples": int(config.num_samples),
        "max_bounces": int(config.max_bounces),
        "shadow_boundary_backend": str(config.tuning.shadow_boundary_backend),
        "shadow_boundary_tile_shape": tuple(
            int(value) for value in config.tuning.shadow_boundary_tile_shape
        ),
        "shadow_boundary_band_width_wavelengths": float(
            config.tuning.shadow_boundary_band_width_wavelengths
        ),
        "shadow_boundary_max_candidate_factor": float(
            config.tuning.shadow_boundary_max_candidate_factor
        ),
        "max_diffraction_order": int(config.max_diffraction_order),
        "shadow_boundary_correction": bool(config.shadow_boundary_correction),
        "solver_mode": str(config.tuning.solver_mode),
        "memory_profile": str(config.tuning.memory_profile),
        "solver_controls": resolve_solver_controls(config, execution_intent="coherent"),
    }
    timing = {
        "scene_load_seconds": float(scene_load_seconds),
        "solve_seconds": float(solve_seconds),
        "result_materialization_seconds": float(result_materialization_seconds),
        "total_seconds": float(time.perf_counter() - total_start),
    }
    environment = {
        "native_extension_available": bool(native_extension_available()),
        "gpu": memory["before_scene_load"],
    }
    result_summary = {
        "path_gain": _value_stats(path_gain),
        "components": component_summary,
        "metadata": getattr(result, "metadata", {}),
    }
    return build_summary(
        environment=environment,
        scenario=scenario,
        memory=memory,
        timing=timing,
        result_summary=result_summary,
        kernel_history=history,
    )


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    summary = run_profile(args)
    validate_summary(
        summary,
        assert_peak_used_mib_below=args.assert_peak_used_mib_below,
        assert_finite=bool(args.assert_finite),
    )
    output_json = args.output_json
    if bool(args.json) and output_json is None:
        output_json = DEFAULT_OUTPUT_JSON
    if output_json is not None:
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        summary["output_json"] = str(output_json)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
