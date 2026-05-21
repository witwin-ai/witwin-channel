"""Validate the standalone path solver on the bundled Sionna Munich scene.

The runner intentionally separates LoS, reflection, performance, and AD-vs-FD
checks so Munich-scale failures are reported as diagnostics instead of hiding
later checks.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

CHANNEL_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SIONNA_SOURCE_ROOT = CHANNEL_ROOT / "reference" / "sionna-rt-reference-2.0.1" / "src"
DEFAULT_MUNICH_XML = DEFAULT_SIONNA_SOURCE_ROOT / "sionna" / "rt" / "scenes" / "munich" / "munich.xml"
DEFAULT_TX_POSITIONS = ((8.5, 21.0, 27.0), (45.0, 15.0, 22.0))
DEFAULT_RX_POSITIONS = ((45.0, 90.0, 1.5), (30.0, 55.0, 1.5), (60.0, 20.0, 1.5))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sionna-source-root", type=Path, default=DEFAULT_SIONNA_SOURCE_ROOT)
    parser.add_argument("--munich-xml", type=Path, default=DEFAULT_MUNICH_XML)
    parser.add_argument("--frequency-hz", type=float, default=3.5e9)
    parser.add_argument("--num-samples", type=int, default=2048)
    parser.add_argument("--max-num-paths", type=int, default=16)
    parser.add_argument("--reflection-max-bounces", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--fd-step-m", type=float, default=0.1)
    parser.add_argument("--tau-atol-s", type=float, default=1.0e-10)
    parser.add_argument("--ad-rtol", type=float, default=5.0e-2)
    parser.add_argument("--ad-atol", type=float, default=1.0e-12)
    parser.add_argument("--skip-reflection", action="store_true", default=False)
    parser.add_argument("--include-diffraction", action="store_true", default=False)
    parser.add_argument("--skip-ad", action="store_true", default=False)
    parser.add_argument("--no-strict", action="store_true", default=False)
    parser.add_argument("--json", action="store_true", default=False)
    parser.add_argument("--output", type=Path, default=None)
    return parser


def _ensure_import_paths(sionna_source_root: Path) -> None:
    for path in (CHANNEL_ROOT, Path(sionna_source_root).resolve()):
        text = str(path)
        if text not in sys.path:
            sys.path.insert(0, text)


def _jsonable(value: Any) -> Any:
    try:
        import numpy as np
    except Exception:
        np = None
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if np is not None and isinstance(value, (np.floating, np.integer, np.bool_)):
        return value.item()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _sync_witwin_path(result) -> None:
    import drjit as dr

    dr.eval(result.a, result.tau, result.valid, result.num_paths)
    dr.sync_thread()


def _sync_sionna_path(result) -> None:
    import drjit as dr

    dr.eval(result)
    dr.sync_thread()


def _timed(operation: Callable[[], Any], sync: Callable[[Any], None], *, warmup: int, repeats: int) -> dict[str, Any]:
    import drjit as dr
    import numpy as np

    dr.sync_thread()
    for _ in range(max(0, int(warmup))):
        sync(operation())
    samples_ms = []
    result = None
    for _ in range(max(1, int(repeats))):
        start = time.perf_counter()
        result = operation()
        sync(result)
        samples_ms.append((time.perf_counter() - start) * 1000.0)
    return {
        "samples_ms": samples_ms,
        "median_ms": float(np.median(samples_ms)),
        "mean_ms": float(np.mean(samples_ms)),
        "result": result,
    }


def _build_witwin_scene(*, munich_xml: Path, sionna_source_root: Path, frequency_hz: float, tx_positions, rx_positions):
    import witwin.channel as wc

    def _endpoint_position(position):
        if all(hasattr(position, axis) for axis in ("x", "y", "z")):
            return position
        return tuple(float(v) for v in position)

    scene = wc.Scene.load_mitsuba(
        munich_xml,
        source_root=sionna_source_root,
        frequency=float(frequency_hz),
        merge_shapes=True,
        device="cuda",
    )
    for index, position in enumerate(tx_positions):
        scene.add(wc.Transmitter(f"tx{index}", _endpoint_position(position)))
    for index, position in enumerate(rx_positions):
        scene.add(wc.Receiver(f"rx{index}", _endpoint_position(position)))
    return scene


def _build_sionna_scene(*, munich_xml: Path, sionna_source_root: Path, frequency_hz: float, tx_positions, rx_positions):
    from witwin.channel.core.scene.sionna_adaptor import SionnaAdaptor

    rt = SionnaAdaptor.load_rt(source_root=sionna_source_root, prefer_local=True)
    import mitsuba as mi

    scene = rt.load_scene(str(munich_xml), merge_shapes=True)
    scene.frequency = float(frequency_hz)
    scene.tx_array = rt.PlanarArray(num_rows=1, num_cols=1, pattern="iso", polarization="V")
    scene.rx_array = rt.PlanarArray(num_rows=1, num_cols=1, pattern="iso", polarization="V")
    for index, position in enumerate(tx_positions):
        scene.add(rt.Transmitter(f"tx{index}", position=mi.Point3f(*position)))
    for index, position in enumerate(rx_positions):
        scene.add(rt.Receiver(f"rx{index}", position=mi.Point3f(*position)))
    return scene, rt


def _witwin_solve(
    scene,
    *,
    max_bounces: int,
    num_samples: int,
    max_num_paths: int,
    max_diffraction_order: int = 0,
):
    import witwin.channel as wc

    tuning = wc.path.Tuning()
    edge_policy = None
    if int(max_diffraction_order) > 0:
        tuning = wc.path.Tuning(diffraction_state_budget=4096, inserted_reflection_state_budget=2048)
        edge_policy = wc.EdgePolicy(edge_selection_mode="all_edges", edge_diffraction=True, boundary_edge_policy="half_plane")
    return wc.path.solve(
        scene=scene,
        transmitter=[f"tx{i}" for i in range(len(scene.transmitters))],
        receiver=[f"rx{i}" for i in range(len(scene.receivers))],
        config=wc.path.Config(
            num_samples=int(num_samples),
            max_bounces=int(max_bounces),
            max_diffraction_order=int(max_diffraction_order),
            max_num_paths=int(max_num_paths),
            return_geometry=True,
            edge_policy=edge_policy,
            tuning=tuning,
        ),
    )


def _sionna_solve(scene, rt, *, max_depth: int, num_samples: int, reflection: bool, diffraction: bool = False):
    return rt.PathSolver()(
        scene,
        max_depth=int(max_depth),
        synthetic_array=True,
        los=True,
        specular_reflection=bool(reflection),
        diffuse_reflection=False,
        refraction=False,
        diffraction=bool(diffraction),
        edge_diffraction=bool(diffraction),
        samples_per_src=int(num_samples),
        seed=7,
    )


def _witwin_tau_valid(result):
    import numpy as np

    tau = np.asarray(result.tau, dtype=np.float64)[:, 0, :, 0, :]
    valid = np.asarray(result.valid, dtype=bool)[:, 0, :, 0, :]
    return tau, valid


def _sionna_tau_valid(paths):
    import numpy as np

    return np.asarray(paths.tau, dtype=np.float64), np.asarray(paths.valid, dtype=bool)


def _compare_los_tau(witwin_result, sionna_result, *, tau_atol_s: float) -> dict[str, Any]:
    import numpy as np

    wt_tau, wt_valid = _witwin_tau_valid(witwin_result)
    sn_tau, sn_valid = _sionna_tau_valid(sionna_result)
    both = wt_valid[..., 0] & sn_valid[..., 0]
    missing = int(np.count_nonzero(wt_valid[..., 0] != sn_valid[..., 0]))
    if np.any(both):
        delta = np.abs(wt_tau[..., 0][both] - sn_tau[..., 0][both])
        max_delta = float(np.max(delta))
    else:
        max_delta = float("inf")
    passed = missing == 0 and max_delta <= float(tau_atol_s)
    return {
        "passed": bool(passed),
        "missing_los_pairs": missing,
        "max_tau_delta_s": max_delta,
        "witwin_shape": list(wt_tau.shape),
        "sionna_shape": list(sn_tau.shape),
    }


def _compare_tau_sets(witwin_result, sionna_result, *, tau_atol_s: float) -> dict[str, Any]:
    import numpy as np

    wt_tau, wt_valid = _witwin_tau_valid(witwin_result)
    sn_tau, sn_valid = _sionna_tau_valid(sionna_result)
    if wt_valid.shape[:2] != sn_valid.shape[:2]:
        return {
            "passed": False,
            "shape_mismatch": {"witwin": list(wt_valid.shape), "sionna": list(sn_valid.shape)},
            "count_mismatches": None,
            "max_tau_delta_s": float("inf"),
        }

    count_mismatches = 0
    max_delta = 0.0
    for rx_index in range(wt_valid.shape[0]):
        for tx_index in range(wt_valid.shape[1]):
            wt_pair = np.sort(wt_tau[rx_index, tx_index][wt_valid[rx_index, tx_index]])
            sn_pair = np.sort(sn_tau[rx_index, tx_index][sn_valid[rx_index, tx_index]])
            if wt_pair.shape != sn_pair.shape:
                count_mismatches += 1
                continue
            if wt_pair.size:
                max_delta = max(max_delta, float(np.max(np.abs(wt_pair - sn_pair))))
    return {
        "passed": count_mismatches == 0 and max_delta <= float(tau_atol_s),
        "shape_mismatch": None,
        "count_mismatches": count_mismatches,
        "max_tau_delta_s": max_delta,
    }


def _path_stats(label: str, result, *, sionna: bool = False) -> dict[str, Any]:
    import numpy as np

    tau, valid = _sionna_tau_valid(result) if sionna else _witwin_tau_valid(result)
    return {
        "label": label,
        "valid_paths": int(np.count_nonzero(valid)),
        "finite_tau": bool(np.isfinite(tau[valid]).all()) if np.any(valid) else True,
        "per_pair_counts": np.count_nonzero(valid, axis=-1).tolist(),
    }


def _loss_for_tx0_x(*, args: argparse.Namespace, tx0_x: float | object, enable_grad: bool):
    import drjit as dr
    import witwin.channel as wc

    tx_positions = list(DEFAULT_TX_POSITIONS)
    if enable_grad:
        tx_positions[0] = wc.Point3f(tx0_x, tx_positions[0][1], tx_positions[0][2])
    else:
        tx_positions[0] = (float(tx0_x), tx_positions[0][1], tx_positions[0][2])
    scene = _build_witwin_scene(
        munich_xml=Path(args.munich_xml),
        sionna_source_root=Path(args.sionna_source_root),
        frequency_hz=float(args.frequency_hz),
        tx_positions=tuple(tx_positions),
        rx_positions=DEFAULT_RX_POSITIONS,
    )
    result = _witwin_solve(scene, max_bounces=0, num_samples=int(args.num_samples), max_num_paths=1)
    return dr.sum(dr.select(result.valid, result.tau, wc.Float(0.0)))


def _ad_vs_fd(args: argparse.Namespace) -> dict[str, Any]:
    import drjit as dr
    import numpy as np
    import witwin.channel as wc

    flags = dr.ADFlag.Default | dr.ADFlag.AllowNoGrad
    try:
        dr.clear_grad()
    except TypeError:
        pass
    tx0_x = wc.Float(DEFAULT_TX_POSITIONS[0][0])
    dr.enable_grad(tx0_x)
    loss = _loss_for_tx0_x(args=args, tx0_x=tx0_x, enable_grad=True)
    dr.backward(loss, flags=flags)
    ad_grad = float(np.asarray(dr.grad(tx0_x), dtype=np.float64).reshape(-1)[0])

    step = float(args.fd_step_m)
    plus = float(np.asarray(_loss_for_tx0_x(args=args, tx0_x=DEFAULT_TX_POSITIONS[0][0] + step, enable_grad=False), dtype=np.float64))
    minus = float(np.asarray(_loss_for_tx0_x(args=args, tx0_x=DEFAULT_TX_POSITIONS[0][0] - step, enable_grad=False), dtype=np.float64))
    fd_grad = (plus - minus) / (2.0 * step)
    abs_error = abs(ad_grad - fd_grad)
    rel_error = abs_error / max(abs(fd_grad), float(args.ad_atol))
    passed = abs_error <= float(args.ad_atol) or rel_error <= float(args.ad_rtol)
    return {
        "passed": bool(passed),
        "variable": "tx0.position.x",
        "loss": float(np.asarray(loss, dtype=np.float64)),
        "ad_grad": ad_grad,
        "fd_grad": fd_grad,
        "abs_error": abs_error,
        "rel_error": rel_error,
        "fd_step_m": step,
    }


def run_validation(args: argparse.Namespace) -> dict[str, Any]:
    _ensure_import_paths(Path(args.sionna_source_root))
    import drjit as dr
    import sionna

    tx_positions = DEFAULT_TX_POSITIONS
    rx_positions = DEFAULT_RX_POSITIONS
    witwin_scene = _build_witwin_scene(
        munich_xml=Path(args.munich_xml),
        sionna_source_root=Path(args.sionna_source_root),
        frequency_hz=float(args.frequency_hz),
        tx_positions=tx_positions,
        rx_positions=rx_positions,
    )
    sionna_scene, rt = _build_sionna_scene(
        munich_xml=Path(args.munich_xml),
        sionna_source_root=Path(args.sionna_source_root),
        frequency_hz=float(args.frequency_hz),
        tx_positions=tx_positions,
        rx_positions=rx_positions,
    )

    los_wt = _timed(
        lambda: _witwin_solve(witwin_scene, max_bounces=0, num_samples=int(args.num_samples), max_num_paths=int(args.max_num_paths)),
        _sync_witwin_path,
        warmup=int(args.warmup),
        repeats=int(args.repeats),
    )
    los_sn = _timed(
        lambda: _sionna_solve(sionna_scene, rt, max_depth=0, num_samples=int(args.num_samples), reflection=False),
        _sync_sionna_path,
        warmup=int(args.warmup),
        repeats=int(args.repeats),
    )
    los_wt_result = los_wt.pop("result")
    los_sn_result = los_sn.pop("result")
    checks: dict[str, Any] = {
        "los_correctness": _compare_los_tau(los_wt_result, los_sn_result, tau_atol_s=float(args.tau_atol_s)),
        "los_stats": {
            "witwin": _path_stats("witwin_los", los_wt_result),
            "sionna": _path_stats("sionna_los", los_sn_result, sionna=True),
        },
        "los_performance": {
            "witwin": los_wt,
            "sionna": los_sn,
            "witwin_over_sionna_median": los_wt["median_ms"] / los_sn["median_ms"] if los_sn["median_ms"] > 0 else None,
        },
    }

    if not bool(args.skip_reflection):
        checks["reflection_correctness"] = _run_reflection_check(args, witwin_scene, sionna_scene, rt)

    if bool(args.include_diffraction):
        checks["diffraction_correctness"] = _run_diffraction_check(args, witwin_scene, sionna_scene, rt)

    if not bool(args.skip_ad):
        checks["ad_vs_fd"] = _ad_vs_fd(args)
    failures = [name for name, check in checks.items() if isinstance(check, Mapping) and check.get("passed") is False]
    return {
        "scenario": {
            "scene": "munich",
            "munich_xml": str(Path(args.munich_xml)),
            "frequency_hz": float(args.frequency_hz),
            "tx_positions": tx_positions,
            "rx_positions": rx_positions,
            "num_tx": len(tx_positions),
            "num_rx": len(rx_positions),
            "num_samples": int(args.num_samples),
            "max_num_paths": int(args.max_num_paths),
            "reflection_max_bounces": int(args.reflection_max_bounces),
        },
        "environment": {
            "sionna_file": sionna.__file__,
            "sionna_source_root": str(Path(args.sionna_source_root)),
            "drjit_version": dr.__version__,
        },
        "checks": checks,
        "passed": not failures,
        "failures": failures,
    }


def _run_reflection_check(args: argparse.Namespace, witwin_scene, sionna_scene, rt) -> dict[str, Any]:
    reflection_max_bounces = int(args.reflection_max_bounces)
    try:
        wt_profile = _timed(
            lambda: _witwin_solve(
                witwin_scene,
                max_bounces=reflection_max_bounces,
                num_samples=int(args.num_samples),
                max_num_paths=int(args.max_num_paths),
            ),
            _sync_witwin_path,
            warmup=int(args.warmup),
            repeats=int(args.repeats),
        )
        wt_result = wt_profile.pop("result")
        wt_stats = _path_stats("witwin_reflection", wt_result)
        witwin_error = None
    except Exception as exc:
        wt_result = None
        wt_profile = None
        wt_stats = None
        witwin_error = f"{type(exc).__name__}: {exc}"

    try:
        sn_profile = _timed(
            lambda: _sionna_solve(
                sionna_scene,
                rt,
                max_depth=reflection_max_bounces,
                num_samples=int(args.num_samples),
                reflection=True,
            ),
            _sync_sionna_path,
            warmup=int(args.warmup),
            repeats=int(args.repeats),
        )
        sn_result = sn_profile.pop("result")
        sn_stats = _path_stats("sionna_reflection", sn_result, sionna=True)
        sionna_error = None
    except Exception as exc:
        sn_result = None
        sn_profile = None
        sn_stats = None
        sionna_error = f"{type(exc).__name__}: {exc}"

    tau_comparison = None
    if wt_result is not None and sn_result is not None:
        tau_comparison = _compare_tau_sets(wt_result, sn_result, tau_atol_s=float(args.tau_atol_s))
    passed = (
        witwin_error is None
        and sionna_error is None
        and (tau_comparison is None or bool(tau_comparison["passed"]))
    )
    return {
        "passed": bool(passed),
        "max_bounces": reflection_max_bounces,
        "witwin_error": witwin_error,
        "sionna_error": sionna_error,
        "witwin": wt_stats,
        "sionna": sn_stats,
        "tau_comparison": tau_comparison,
        "performance": {
            "witwin": wt_profile,
            "sionna": sn_profile,
            "witwin_over_sionna_median": (
                None
                if wt_profile is None or sn_profile is None or float(sn_profile["median_ms"]) <= 0.0
                else float(wt_profile["median_ms"]) / float(sn_profile["median_ms"])
            ),
        },
    }


def _run_diffraction_check(args: argparse.Namespace, witwin_scene, sionna_scene, rt) -> dict[str, Any]:
    try:
        wt_profile = _timed(
            lambda: _witwin_solve(
                witwin_scene,
                max_bounces=1,
                max_diffraction_order=1,
                num_samples=int(args.num_samples),
                max_num_paths=int(args.max_num_paths),
            ),
            _sync_witwin_path,
            warmup=int(args.warmup),
            repeats=int(args.repeats),
        )
        wt_result = wt_profile.pop("result")
        wt_stats = _path_stats("witwin_diffraction", wt_result)
        witwin_error = None
    except Exception as exc:
        wt_result = None
        wt_profile = None
        wt_stats = None
        witwin_error = f"{type(exc).__name__}: {exc}"

    try:
        sn_profile = _timed(
            lambda: _sionna_solve(
                sionna_scene,
                rt,
                max_depth=1,
                num_samples=int(args.num_samples),
                reflection=True,
                diffraction=True,
            ),
            _sync_sionna_path,
            warmup=int(args.warmup),
            repeats=int(args.repeats),
        )
        sn_result = sn_profile.pop("result")
        sn_stats = _path_stats("sionna_diffraction", sn_result, sionna=True)
        sionna_error = None
    except Exception as exc:
        sn_result = None
        sn_profile = None
        sn_stats = None
        sionna_error = f"{type(exc).__name__}: {exc}"

    tau_comparison = None
    if wt_result is not None and sn_result is not None:
        tau_comparison = _compare_tau_sets(wt_result, sn_result, tau_atol_s=float(args.tau_atol_s))
    passed = (
        witwin_error is None
        and sionna_error is None
        and (tau_comparison is None or bool(tau_comparison["passed"]))
    )
    return {
        "passed": bool(passed),
        "witwin_error": witwin_error,
        "sionna_error": sionna_error,
        "witwin": wt_stats,
        "sionna": sn_stats,
        "tau_comparison": tau_comparison,
        "performance": {
            "witwin": wt_profile,
            "sionna": sn_profile,
            "witwin_over_sionna_median": (
                None
                if wt_profile is None or sn_profile is None or float(sn_profile["median_ms"]) <= 0.0
                else float(wt_profile["median_ms"]) / float(sn_profile["median_ms"])
            ),
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    result = run_validation(args)
    text = json.dumps(_jsonable(result), indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text if args.json else _human_summary(result))
    return 1 if (not args.no_strict and not result["passed"]) else 0


def _human_summary(result: Mapping[str, Any]) -> str:
    lines = [
        "Munich path solver validation",
        f"passed: {result['passed']}",
        f"failures: {', '.join(result['failures']) if result['failures'] else '<none>'}",
    ]
    for name, check in result["checks"].items():
        if isinstance(check, Mapping) and "passed" in check:
            lines.append(f"{name}: passed={check['passed']}")
            if check.get("witwin_error"):
                lines.append(f"  witwin_error={check['witwin_error']}")
            if "ad_grad" in check:
                lines.append(f"  ad_grad={check['ad_grad']} fd_grad={check['fd_grad']} rel_error={check['rel_error']}")
            if "max_tau_delta_s" in check:
                lines.append(f"  max_tau_delta_s={check['max_tau_delta_s']}")
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
