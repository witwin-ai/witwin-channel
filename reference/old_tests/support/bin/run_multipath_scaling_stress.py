"""Run multipath scaling stress sweeps in isolated subprocesses."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

try:
    from ._multipath_scaling_cases import DEFAULT_BASELINE, DEFAULT_SWEEPS
except ImportError:
    THIS_DIR = Path(__file__).resolve().parent
    if str(THIS_DIR) not in sys.path:
        sys.path.insert(0, str(THIS_DIR))
    from _multipath_scaling_cases import DEFAULT_BASELINE, DEFAULT_SWEEPS

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent.parent.parent


def _case_slug(config: dict[str, int]) -> str:
    return (
        f"g{int(config['grid_size']):04d}_"
        f"r{int(config['n_rays']):06d}_"
        f"m{int(config['motif_repeats']):03d}"
    )


def _tail(text: str, *, lines: int = 12) -> str:
    chunks = text.strip().splitlines()
    if not chunks:
        return ""
    return "\n".join(chunks[-lines:])


def _run_child_case(
    *,
    python_executable: str,
    worker_module: str,
    config: dict[str, int],
    output_json: Path,
    warmup_runs: int,
    timeout_seconds: int,
) -> dict[str, Any]:
    command = [
        python_executable,
        "-m",
        worker_module,
        "--grid-size",
        str(int(config["grid_size"])),
        "--n-rays",
        str(int(config["n_rays"])),
        "--motif-repeats",
        str(int(config["motif_repeats"])),
        "--warmup-runs",
        str(int(max(0, warmup_runs))),
        "--output-json",
        str(output_json),
    ]
    started_at = time.strftime("%Y-%m-%d %H:%M:%S")
    start = time.perf_counter()
    try:
        completed = subprocess.run(
            command,
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            timeout=max(1, int(timeout_seconds)),
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        return {
            "status": "timeout",
            "config": dict(config),
            "started_at": started_at,
            "elapsed_seconds": float(time.perf_counter() - start),
            "timeout_seconds": int(timeout_seconds),
            "stdout_tail": _tail(exc.stdout or ""),
            "stderr_tail": _tail(exc.stderr or ""),
            "command": command,
        }

    elapsed = float(time.perf_counter() - start)
    record: dict[str, Any] = {
        "status": "ok" if completed.returncode == 0 and output_json.exists() else "failed",
        "config": dict(config),
        "started_at": started_at,
        "elapsed_seconds": elapsed,
        "returncode": int(completed.returncode),
        "stdout_tail": _tail(completed.stdout),
        "stderr_tail": _tail(completed.stderr),
        "command": command,
        "output_json": str(output_json),
    }
    if completed.returncode == 0 and output_json.exists():
        record["result"] = json.loads(output_json.read_text(encoding="utf-8"))
    return record


def run_stress_sweeps(
    *,
    sweeps: list[str],
    output_dir: Path,
    warmup_runs: int,
    timeout_seconds: int,
    python_executable: str,
    worker_module: str = "tests.support.bin.benchmark_multipath_scaling",
    diff_phase_name: str = "jvp",
    benchmark_name: str = "multipath_scaling_stress",
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    aggregate: dict[str, Any] = {
        "benchmark": benchmark_name,
        "date": time.strftime("%Y-%m-%d"),
        "baseline": dict(DEFAULT_BASELINE),
        "warmup_runs": int(max(0, warmup_runs)),
        "python_executable": python_executable,
        "worker_module": worker_module,
        "diff_phase_name": diff_phase_name,
        "output_dir": str(output_dir),
        "sweeps": {},
    }

    for sweep_name in sweeps:
        configs = DEFAULT_SWEEPS[sweep_name]
        sweep_runs: list[dict[str, Any]] = []
        print(f"[sweep:{sweep_name}] {len(configs)} cases")
        for index, config in enumerate(configs, start=1):
            slug = _case_slug(config)
            output_json = output_dir / f"{sweep_name}_{slug}.json"
            print(
                f"  [{index}/{len(configs)}] grid={config['grid_size']} "
                f"rays={config['n_rays']} motifs={config['motif_repeats']}"
            )
            record = _run_child_case(
                python_executable=python_executable,
                worker_module=worker_module,
                config=config,
                output_json=output_json,
                warmup_runs=warmup_runs,
                timeout_seconds=timeout_seconds,
            )
            record["sweep_name"] = sweep_name
            if record["status"] == "ok":
                result = record["result"]
                trace_peak = (
                    result["phase_metrics"]["trace"]["memory_after"]["drjit_allocator"].get("device_peak")
                )
                diff_peak = (
                    result["phase_metrics"][diff_phase_name]["memory_after"]["drjit_allocator"].get("device_peak")
                )
                print(
                    f"    ok total={result['total_seconds']:.3f}s "
                    f"trace={result['trace_seconds']:.3f}s {diff_phase_name}={result[f'{diff_phase_name}_seconds']:.3f}s "
                    f"peak(trace/{diff_phase_name})={trace_peak}/{diff_peak}"
                )
            else:
                print(
                    f"    {record['status']} returncode={record.get('returncode', 'n/a')} "
                    f"elapsed={record['elapsed_seconds']:.3f}s"
                )
            sweep_runs.append(record)
            aggregate["sweeps"][sweep_name] = {
                "name": sweep_name,
                "runs": sweep_runs,
            }
    return aggregate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sweeps",
        nargs="+",
        default=list(DEFAULT_SWEEPS.keys()),
        choices=tuple(DEFAULT_SWEEPS.keys()),
        help="Subset of scaling sweeps to execute.",
    )
    parser.add_argument(
        "--warmup-runs",
        type=int,
        default=1,
        help="Warmup passes performed inside each isolated worker process before the measured pass.",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=1800,
        help="Per-case timeout for the worker subprocess.",
    )
    parser.add_argument(
        "--python-executable",
        type=str,
        default=sys.executable,
        help="Python interpreter used to spawn isolated worker processes.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "tests" / "output" / f"multipath_scaling_stress_{time.strftime('%Y-%m-%d')}",
        help="Directory used for per-case JSON outputs.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional aggregate JSON path. Defaults to <output-dir>/aggregate.json.",
    )
    args = parser.parse_args()

    payload = run_stress_sweeps(
        sweeps=list(args.sweeps),
        output_dir=args.output_dir,
        warmup_runs=args.warmup_runs,
        timeout_seconds=args.timeout_seconds,
        python_executable=args.python_executable,
        worker_module="tests.support.bin.benchmark_multipath_scaling",
        diff_phase_name="jvp",
        benchmark_name="multipath_scaling_stress",
    )

    output_json = args.output_json or (args.output_dir / "aggregate.json")
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Saved aggregate JSON: {output_json.resolve()}")


if __name__ == "__main__":
    main()
