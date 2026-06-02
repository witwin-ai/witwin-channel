"""Compare simultaneous multi-TX/multi-RX path-solver runtime against Sionna RT."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
try:
    from ._sionna_path_solver_benchmark import (
        available_scenarios,
        format_benchmark_summary,
        run_path_solver_benchmark,
    )
except ImportError:
    THIS_DIR = Path(__file__).resolve().parent
    if str(THIS_DIR) not in sys.path:
        sys.path.insert(0, str(THIS_DIR))
    from _sionna_path_solver_benchmark import (
        available_scenarios,
        format_benchmark_summary,
        run_path_solver_benchmark,
    )


def main() -> None:
    scenarios = available_scenarios()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scenario",
        default="mixed_first_order",
        choices=tuple(sorted(scenarios.keys())),
        help="Scenario to benchmark.",
    )
    parser.add_argument("--tx-count", type=int, default=None, help="Optional TX subset size.")
    parser.add_argument("--rx-count", type=int, default=None, help="Optional RX subset size.")
    parser.add_argument("--warmup", type=int, default=1, help="Warmup iterations before timing.")
    parser.add_argument("--repeats", type=int, default=3, help="Measured timing iterations.")
    parser.add_argument("--json", action="store_true", help="Print full JSON payload.")
    args = parser.parse_args()

    payload = run_path_solver_benchmark(
        scenario_name=str(args.scenario),
        tx_count=args.tx_count,
        rx_count=args.rx_count,
        warmup=int(args.warmup),
        repeats=int(args.repeats),
    )

    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
        return
    print(format_benchmark_summary(payload))


if __name__ == "__main__":
    main()

