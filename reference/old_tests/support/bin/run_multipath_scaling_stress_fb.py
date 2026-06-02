"""Run multipath forward/backward scaling stress sweeps in isolated subprocesses."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

try:
    from ._multipath_scaling_cases import DEFAULT_SWEEPS
    from .run_multipath_scaling_stress import REPO_ROOT, run_stress_sweeps
except ImportError:
    THIS_DIR = Path(__file__).resolve().parent
    if str(THIS_DIR) not in sys.path:
        sys.path.insert(0, str(THIS_DIR))
    from _multipath_scaling_cases import DEFAULT_SWEEPS
    from run_multipath_scaling_stress import REPO_ROOT, run_stress_sweeps


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
        default=REPO_ROOT / "tests" / "output" / f"multipath_scaling_stress_fb_{time.strftime('%Y-%m-%d')}",
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
        worker_module="tests.support.bin.benchmark_multipath_scaling_fb",
        diff_phase_name="backward",
        benchmark_name="multipath_scaling_stress_fb",
    )

    output_json = args.output_json or (args.output_dir / "aggregate.json")
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Saved aggregate JSON: {output_json.resolve()}")


if __name__ == "__main__":
    main()

