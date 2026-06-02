"""CLI runner for the fixed ``grad_multipath`` benchmark workload."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
try:
    from ._benchmark_runtime import benchmark_environment_report
    from ._multipath_benchmark import format_benchmark_summary, run_grad_multipath_benchmark
except ImportError:
    THIS_DIR = Path(__file__).resolve().parent
    if str(THIS_DIR) not in sys.path:
        sys.path.insert(0, str(THIS_DIR))
    from _benchmark_runtime import benchmark_environment_report
    from _multipath_benchmark import format_benchmark_summary, run_grad_multipath_benchmark
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", default="manual", help="Label included in the output payload.")
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit the full benchmark payload as JSON instead of a compact summary line.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Pass verbose=True into Tracer.trace().",
    )
    parser.add_argument(
        "--warmup-runs",
        type=int,
        default=0,
        help="Number of warmup passes to run before the measured pass (default: 0).",
    )
    args = parser.parse_args()

    runtime_environment = benchmark_environment_report()
    if not args.json:
        print(
            "Runtime: "
            f"module={runtime_environment.get('channel_module_file', 'n/a')} "
            f"variant={runtime_environment.get('backend_variant', 'n/a')} "
            f"native={runtime_environment.get('native_extension_available', 'n/a')} "
            f"cuda_runtime_version={runtime_environment.get('cuda_runtime_version', 'n/a')}"
        )

    result = run_grad_multipath_benchmark(
        label=args.label,
        verbose=args.verbose,
        warmup_runs=args.warmup_runs,
    )
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
        return
    print(format_benchmark_summary(result))


if __name__ == "__main__":
    main()
