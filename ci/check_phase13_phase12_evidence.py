# Copyright Xingyu Chen.
# Checks evidence.

"""Checks evidence."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmarks.phase13_phase12_evidence import (  # noqa: E402
    DEFAULT_SCHEMA,
    EvidenceError,
    read_json,
    replay_measured_report,
)


def validate_evidence(
    evidence_path: Path, raw_root: Path, *, repository: Path = ROOT
) -> None:
    try:
        import jsonschema
    except ImportError as exc:
        raise RuntimeError(
            "jsonschema is required for the Phase 12 evidence gate"
        ) from exc
    evidence = read_json(evidence_path)
    schema = read_json(DEFAULT_SCHEMA)
    jsonschema.Draft202012Validator.check_schema(schema)
    validator = jsonschema.Draft202012Validator(
        schema, format_checker=jsonschema.Draft202012Validator.FORMAT_CHECKER
    )
    errors = sorted(validator.iter_errors(evidence), key=lambda row: list(row.path))
    if errors:
        raise ValueError("Phase 12 evidence schema failed: " + errors[0].message)
    replay_measured_report(evidence, raw_root=raw_root, repository=repository)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Replay committed measured Plan 13 Phase 12 evidence."
    )
    parser.add_argument("evidence", type=Path)
    parser.add_argument("--raw-root", type=Path, required=True)
    args = parser.parse_args()
    try:
        validate_evidence(args.evidence, args.raw_root)
    except (OSError, EvidenceError, RuntimeError, ValueError) as exc:
        parser.error(str(exc))
    print(f"Phase 12 evidence replay passed: {args.evidence}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())