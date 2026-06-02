"""Run a Sionna-vs-Witwin path-solver stress matrix across multi-TX/multi-RX workloads."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

try:
    from ._sionna_path_solver_benchmark import available_scenarios, run_path_solver_stress_matrix
except ImportError:
    THIS_DIR = Path(__file__).resolve().parent
    if str(THIS_DIR) not in sys.path:
        sys.path.insert(0, str(THIS_DIR))
    from _sionna_path_solver_benchmark import available_scenarios, run_path_solver_stress_matrix


def _csv_ints(text: str) -> list[int]:
    values = []
    for chunk in str(text).split(","):
        stripped = chunk.strip()
        if stripped:
            values.append(int(stripped))
    if not values:
        raise ValueError("Expected at least one integer value.")
    return values


def _csv_strings(text: str) -> list[str]:
    values = [chunk.strip() for chunk in str(text).split(",") if chunk.strip()]
    if not values:
        raise ValueError("Expected at least one scenario name.")
    return values


def main() -> None:
    scenario_names = tuple(sorted(available_scenarios().keys()))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scenarios",
        default="los_3d,reflection_3d,diffraction_first_order,mixed_first_order",
        help=f"Comma-separated scenario names. Valid options: {', '.join(scenario_names)}",
    )
    parser.add_argument(
        "--tx-counts",
        default="1,2,4",
        help="Comma-separated TX counts to evaluate.",
    )
    parser.add_argument(
        "--rx-counts",
        default="1,2,4",
        help="Comma-separated RX counts to evaluate.",
    )
    parser.add_argument("--warmup", type=int, default=0, help="Warmup iterations per matrix cell.")
    parser.add_argument("--repeats", type=int, default=1, help="Measured iterations per matrix cell.")
    parser.add_argument("--json", action="store_true", help="Print full JSON payload.")
    args = parser.parse_args()

    payload = run_path_solver_stress_matrix(
        scenario_names=_csv_strings(args.scenarios),
        tx_counts=_csv_ints(args.tx_counts),
        rx_counts=_csv_ints(args.rx_counts),
        warmup=int(args.warmup),
        repeats=int(args.repeats),
    )

    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
        return

    print(
        f"Stress matrix: scenarios={','.join(payload['matrix_config']['scenario_names'])} "
        f"tx_counts={payload['matrix_config']['tx_counts']} "
        f"rx_counts={payload['matrix_config']['rx_counts']}"
    )
    for result in payload["results"]:
        scenario = result["scenario"]
        comparison = result["comparison"]
        print(
            f"{scenario['name']} tx={scenario['tx_count']} rx={scenario['rx_count']} "
            f"witwin={result['witwin']['timing']['median_ms']:.2f}ms "
            f"sionna={result['sionna']['timing']['median_ms']:.2f}ms "
            f"speedup={comparison['median_speedup_vs_sionna']:.3f}x "
            f"signature_match={comparison['signature_match']}"
        )


if __name__ == "__main__":
    main()

