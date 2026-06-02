"""Benchmark standalone path solving on the bundled Munich scene.

The report separates Witwin path-solver scale checks from Sionna comparisons.
Sionna RT 2.0.1 only exposes first-order diffraction, so order-2/3 Witwin
diffraction runs are reported as capability/scale checks rather than speed
ratios against Sionna.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from tests.support.bin import validate_path_solver_munich as base


DEFAULT_OUTPUT_JSON = (
    base.CHANNEL_ROOT
    / "docs"
    / "dev"
    / "optimization"
    / "path_solver_munich_vs_sionna_2026-05-22.json"
)


def _parse_int_csv(text: str, *, name: str) -> tuple[int, ...]:
    values: list[int] = []
    for token in str(text).split(","):
        stripped = token.strip()
        if not stripped:
            raise ValueError(f"empty token in --{name}")
        values.append(int(stripped))
    if not values:
        raise ValueError(f"--{name} must contain at least one integer")
    return tuple(values)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sionna-source-root", type=Path, default=base.DEFAULT_SIONNA_SOURCE_ROOT)
    parser.add_argument("--munich-xml", type=Path, default=base.DEFAULT_MUNICH_XML)
    parser.add_argument("--frequency-hz", type=float, default=3.5e9)
    parser.add_argument("--sample-counts", type=str, default="4096")
    parser.add_argument("--orders", type=str, default="0,1,2,3")
    parser.add_argument("--max-bounces", type=int, default=1)
    parser.add_argument("--max-num-paths", type=int, default=64)
    parser.add_argument("--diffraction-state-budget", type=int, default=4096)
    parser.add_argument("--inserted-reflection-state-budget", type=int, default=2048)
    parser.add_argument("--enable-rd-diffraction", action="store_true", default=False)
    parser.add_argument(
        "--accumulate-primal",
        choices=("auto", "drjit", "rayd_optix"),
        default="auto",
    )
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--skip-sionna", action="store_true", default=False)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--json", action="store_true", default=False)
    return parser


def _jsonable(value: Any) -> Any:
    return base._jsonable(value)


def _gpu_info() -> dict[str, Any]:
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,memory.used,driver_version",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=3.0,
        ).strip()
    except Exception as exc:
        return {"available": False, "error": str(exc)}
    fields = [field.strip() for field in output.split(",", 3)]
    if len(fields) != 4:
        return {"available": True, "raw": output}
    return {
        "available": True,
        "name": fields[0],
        "memory_total_mib": int(fields[1]),
        "memory_used_mib": int(fields[2]),
        "driver_version": fields[3],
    }


def _sync_path_result(result) -> None:
    import drjit as dr

    dr.eval(result.a, result.tau, result.valid, result.num_paths)
    dr.sync_thread()


def _path_stats(result) -> dict[str, Any]:
    import numpy as np

    tau = np.asarray(result.tau, dtype=np.float64)
    valid = np.asarray(result.valid, dtype=bool)
    a = np.asarray(result.a).reshape(valid.shape)
    num_paths = np.asarray(result.num_paths, dtype=np.int64)
    valid_a = a[valid]
    valid_tau = tau[valid]
    return {
        "shape": {
            "a": list(a.shape),
            "tau": list(tau.shape),
            "valid": list(valid.shape),
            "num_paths": list(num_paths.shape),
        },
        "valid_paths": int(np.count_nonzero(valid)),
        "num_paths_sum": int(np.sum(num_paths)),
        "finite_tau": bool(np.isfinite(valid_tau).all()) if valid_tau.size else True,
        "finite_field": bool(np.isfinite(valid_a.real).all() and np.isfinite(valid_a.imag).all())
        if valid_a.size
        else True,
        "per_pair_counts": num_paths.tolist(),
    }


def _witwin_config(args: argparse.Namespace, *, samples: int, order: int):
    import witwin.channel as wc

    tuning = wc.path.Tuning(diffraction_execution={"accumulate_primal": str(args.accumulate_primal)})
    edge_policy = None
    if int(order) > 0:
        tuning = wc.path.Tuning(
            enable_rd_diffraction=bool(args.enable_rd_diffraction),
            diffraction_state_budget=int(args.diffraction_state_budget),
            inserted_reflection_state_budget=int(args.inserted_reflection_state_budget),
            diffraction_execution={"accumulate_primal": str(args.accumulate_primal)},
        )
        edge_policy = wc.EdgePolicy(
            edge_selection_mode="all_edges",
            edge_diffraction=True,
            boundary_edge_policy="half_plane",
        )
    return wc.path.Config(
        num_samples=int(samples),
        max_bounces=int(args.max_bounces),
        max_diffraction_order=int(order),
        max_num_paths=int(args.max_num_paths),
        return_geometry=False,
        edge_policy=edge_policy,
        tuning=tuning,
    )


def _run_witwin_case(
    args: argparse.Namespace,
    scene,
    *,
    samples: int,
    order: int,
) -> dict[str, Any]:
    import witwin.channel as wc

    config = _witwin_config(args, samples=int(samples), order=int(order))
    tx_names = [f"tx{i}" for i in range(len(scene.transmitters))]
    rx_names = [f"rx{i}" for i in range(len(scene.receivers))]
    try:
        profile = base._timed(
            lambda: wc.path.solve(
                scene=scene,
                transmitter=tx_names,
                receiver=rx_names,
                config=config,
            ),
            _sync_path_result,
            warmup=int(args.warmup),
            repeats=int(args.repeats),
        )
        result = profile.pop("result")
        return {
            "ok": True,
            "error": None,
            "profile": profile,
            "stats": _path_stats(result),
            "metadata": {
                "runtime_backends": result.metadata.get("runtime_backends", {}),
                "path_counts": result.metadata.get("path_counts", {}),
                "solver_mode": result.metadata.get("solver_mode", {}),
                "timing": result.metadata.get("timing", {}),
                "diffraction_groups": result.metadata.get("diffraction_groups", ()),
            },
        }
    except Exception as exc:
        return {
            "ok": False,
            "error": f"{type(exc).__name__}: {exc}",
            "profile": None,
            "stats": None,
            "metadata": None,
        }


def _run_sionna_case(
    args: argparse.Namespace,
    scene,
    rt,
    *,
    samples: int,
    order: int,
) -> dict[str, Any]:
    if int(order) > 1:
        return {
            "ok": False,
            "supported": False,
            "error": "Sionna RT 2.0.1 PathSolver supports first-order diffraction only.",
            "profile": None,
            "stats": None,
        }
    reflection = int(args.max_bounces) > 0
    diffraction = int(order) > 0
    max_depth = max(int(args.max_bounces), int(order))
    try:
        profile = base._timed(
            lambda: base._sionna_solve(
                scene,
                rt,
                max_depth=max_depth,
                num_samples=int(samples),
                reflection=reflection,
                diffraction=diffraction,
            ),
            base._sync_sionna_path,
            warmup=int(args.warmup),
            repeats=int(args.repeats),
        )
        result = profile.pop("result")
        return {
            "ok": True,
            "supported": True,
            "error": None,
            "profile": profile,
            "stats": base._path_stats("sionna_path", result, sionna=True),
        }
    except Exception as exc:
        return {
            "ok": False,
            "supported": True,
            "error": f"{type(exc).__name__}: {exc}",
            "profile": None,
            "stats": None,
        }


def _speed_ratio(witwin: Mapping[str, Any], sionna: Mapping[str, Any]) -> dict[str, Any]:
    wt_profile = witwin.get("profile")
    sn_profile = sionna.get("profile")
    if not wt_profile or not sn_profile:
        return {"witwin_over_sionna_median": None, "sionna_over_witwin_median": None}
    wt_ms = float(wt_profile["median_ms"])
    sn_ms = float(sn_profile["median_ms"])
    return {
        "witwin_over_sionna_median": None if sn_ms <= 0.0 else wt_ms / sn_ms,
        "sionna_over_witwin_median": None if wt_ms <= 0.0 else sn_ms / wt_ms,
    }


def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    base._ensure_import_paths(Path(args.sionna_source_root))
    import drjit as dr
    import sionna

    sample_counts = _parse_int_csv(args.sample_counts, name="sample-counts")
    orders = _parse_int_csv(args.orders, name="orders")
    tx_positions = base.DEFAULT_TX_POSITIONS
    rx_positions = base.DEFAULT_RX_POSITIONS
    witwin_scene = base._build_witwin_scene(
        munich_xml=Path(args.munich_xml),
        sionna_source_root=Path(args.sionna_source_root),
        frequency_hz=float(args.frequency_hz),
        tx_positions=tx_positions,
        rx_positions=rx_positions,
    )
    sionna_scene = None
    rt = None
    if not bool(args.skip_sionna):
        sionna_scene, rt = base._build_sionna_scene(
            munich_xml=Path(args.munich_xml),
            sionna_source_root=Path(args.sionna_source_root),
            frequency_hz=float(args.frequency_hz),
            tx_positions=tx_positions,
            rx_positions=rx_positions,
        )

    cases = []
    for samples in sample_counts:
        for order in orders:
            witwin = _run_witwin_case(args, witwin_scene, samples=int(samples), order=int(order))
            if bool(args.skip_sionna):
                sionna_case = {
                    "ok": False,
                    "supported": False,
                    "error": "Sionna run skipped by --skip-sionna.",
                    "profile": None,
                    "stats": None,
                }
            else:
                sionna_case = _run_sionna_case(
                    args,
                    sionna_scene,
                    rt,
                    samples=int(samples),
                    order=int(order),
                )
            cases.append(
                {
                    "samples": int(samples),
                    "max_diffraction_order": int(order),
                    "witwin": witwin,
                    "sionna": sionna_case,
                    "speed": _speed_ratio(witwin, sionna_case),
                }
            )

    return {
        "scenario": {
            "scene": "munich",
            "munich_xml": str(Path(args.munich_xml)),
            "frequency_hz": float(args.frequency_hz),
            "sample_counts": sample_counts,
            "orders": orders,
            "max_bounces": int(args.max_bounces),
            "max_num_paths": int(args.max_num_paths),
            "diffraction_state_budget": int(args.diffraction_state_budget),
            "inserted_reflection_state_budget": int(args.inserted_reflection_state_budget),
            "enable_rd_diffraction": bool(args.enable_rd_diffraction),
            "accumulate_primal": str(args.accumulate_primal),
            "tx_positions": tx_positions,
            "rx_positions": rx_positions,
        },
        "environment": {
            "gpu": _gpu_info(),
            "sionna_file": sionna.__file__,
            "sionna_source_root": str(Path(args.sionna_source_root)),
            "drjit_version": dr.__version__,
        },
        "cases": cases,
        "notes": [
            "Order 0 is LoS plus requested specular reflection depth.",
            "Order 1 is comparable to Sionna first-order edge diffraction, but path counts are not expected to match exactly because candidate policies differ.",
            "Order 2 and 3 are Witwin capability and scale checks; Sionna RT 2.0.1 has no comparable higher-order diffraction path mode.",
        ],
    }


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = run_benchmark(args)
    text = json.dumps(_jsonable(result), indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    if args.json or args.output is None:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
