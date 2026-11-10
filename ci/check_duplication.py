# Copyright Xingyu Chen.
# Gate exact-token duplicate coverage of the production tree against a tighten-only ceiling.

"""Gate exact-token duplicate coverage of the production tree against a tighten-only ceiling."""

# The report is generated from the source on every run by ``tools/duplication_report.py``; there is
# no per-region ledger to keep in sync. Lower ``COVERAGE_CEILING_PERCENT`` when a cleanup lands; never
# raise it to make a change pass.

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import duplication_report  # noqa: E402

# Combined Python + native duplicate-line coverage for regions of at least MIN_TOKENS tokens.
COVERAGE_CEILING_PERCENT = 8.75


def evaluate(report: dict[str, object], ceiling: float = COVERAGE_CEILING_PERCENT) -> list[str]:
    """Return gate violations; empty means the gate passes."""

    combined = report["combined"]  # type: ignore[index]
    coverage = float(combined["coverage_percent"])  # type: ignore[index]
    if coverage > ceiling:
        return [
            f"duplicate coverage {coverage:.4f}% exceeds the ceiling {ceiling:.4f}%; "
            "run tools/duplication_report.py to list the regions"
        ]
    return []


def _print_current(report: dict[str, object]) -> None:
    combined = report["combined"]  # type: ignore[index]
    corpora = report["corpora"]  # type: ignore[index]
    print(f"exact-token duplicate coverage (>= {report['min_tokens']} tokens)")
    for name in ("python", "native"):
        item = corpora[name]  # type: ignore[index]
        print(
            f"  {name:8} regions={item['region_count']:4} duplicate_lines={item['duplicate_lines']:6} "
            f"total_lines={item['total_lines']:6} coverage={item['coverage_percent']:.4f}%"
        )
    print(
        f"  {'combined':8} regions={combined['region_count']:4} "
        f"duplicate_lines={combined['duplicate_lines']:6} total_lines={combined['total_lines']:6} "
        f"coverage={combined['coverage_percent']:.4f}%"
    )
    print(f"  ceiling coverage={COVERAGE_CEILING_PERCENT:.4f}%")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--repo", type=Path, default=ROOT, help="repository root")
    parser.add_argument(
        "--print-current", action="store_true", help="print current duplication statistics and exit 0"
    )
    args = parser.parse_args(argv)
    report = duplication_report.build_report(args.repo.resolve(), duplication_report.MIN_TOKENS)
    if args.print_current:
        _print_current(report)
        return 0
    violations = evaluate(report)
    for violation in violations:
        print(violation, file=sys.stderr)
    if violations:
        return 1
    print("duplication gate passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
