from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

# Local source and benchmark helpers must resolve from this checkout before importing them.
from benchmarks.harness import versioned_report, write_report  # noqa: E402
from witwin.channel.runtime import (  # noqa: E402
    MemoryBudgetError,
    enforce_memory_budget,
    estimate_monte_carlo_memory,
)


def _artifact(
    *, samples: int, tx: int, rx: int, depth: int, gpu_budget_gib: float
) -> dict[str, Any]:
    estimate = estimate_monte_carlo_memory(
        samples=samples,
        transmitters=tx,
        receivers=rx,
        depth=depth,
    )
    budget = int(gpu_budget_gib * (1 << 30))
    safe = True
    error = None
    try:
        enforce_memory_budget(
            estimate,
            budget_bytes=budget,
            headroom_bytes=1 << 30,
            workload=f"{samples}-sample MC",
        )
    except MemoryBudgetError as exc:
        safe = False
        error = str(exc)
    return {
        "samples": samples,
        "tx": tx,
        "rx": rx,
        "depth": depth,
        "gpu_budget_bytes": budget,
        "headroom_bytes": 1 << 30,
        "estimate": estimate.as_dict(),
        "memory_safe": safe,
        "preflight_error": error,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    rows = [
        _artifact(
            samples=samples,
            tx=args.tx,
            rx=args.rx,
            depth=args.depth,
            gpu_budget_gib=args.gpu_budget_gib,
        )
        for samples in (1_000, 1_000_000, 10_000_000, 100_000_000)
    ]
    return versioned_report(
        benchmark="solver_peak_memory",
        scenario={
            "tx": args.tx,
            "rx": args.rx,
            "depth": args.depth,
            "gpu_budget_gib": args.gpu_budget_gib,
            "note": "100M is a required estimate artifact and is not blindly allocated",
        },
        results=rows,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tx", type=int, default=1)
    parser.add_argument("--rx", type=int, default=1024)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--gpu-budget-gib", type=float, default=16.0)
    parser.add_argument("--output", type=Path, default=Path("artifacts/solver_peak_memory.v1.json"))
    args = parser.parse_args()
    report = run(args)
    write_report(report, args.output)
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
