"""Canonical CLI for Plan 13 Phase 12 evidence generation and replay."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmarks.phase13_phase12_evidence import (  # noqa: E402
    DEFAULT_GATE,
    EvidenceError,
    build_dry_run,
    build_measured_report,
    load_config,
    load_gate,
    read_json,
    replay_measured_report,
    write_report,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate or replay strict Plan 13 Phase 12 evidence."
    )
    parser.add_argument("--config", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--timeout-seconds", type=int, default=7200)
    parser.add_argument("--bootstrap-resamples", type=int, default=100000)
    parser.add_argument("--replay-report", type=Path)
    parser.add_argument("--raw-root", type=Path)
    args = parser.parse_args()
    if args.timeout_seconds <= 0:
        parser.error("--timeout-seconds must be positive")
    if args.bootstrap_resamples != 100000:
        parser.error("--bootstrap-resamples is frozen at exactly 100000")
    replay = args.replay_report is not None or args.raw_root is not None
    if replay:
        if args.replay_report is None or args.raw_root is None or args.config is not None:
            parser.error("replay requires --replay-report and --raw-root, without --config")
        try:
            report = replay_measured_report(
                read_json(args.replay_report), raw_root=args.raw_root
            )
        except (OSError, EvidenceError, ValueError) as exc:
            parser.error(str(exc))
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0
    if args.config is None:
        parser.error("generation requires --config")
    try:
        config = load_config(args.config)
        gate = load_gate(DEFAULT_GATE, measured=not args.dry_run)
        report = (
            build_dry_run(config, gate, gate_path=DEFAULT_GATE)
            if args.dry_run
            else build_measured_report(
                config, gate, gate_path=DEFAULT_GATE,
                timeout_seconds=args.timeout_seconds,
                bootstrap_resamples=args.bootstrap_resamples,
            )
        )
        write_report(report, config.output)
    except (OSError, EvidenceError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
