"""Gate exact-token duplicate regions against the classification ledger.

The gate recomputes the duplicate-region report from the working tree and
enforces three governance invariants for Plan 08 G7:

1. Every duplicate region of at least ``min_tokens`` tokens has a ledger entry
   (no unclassified duplication).
2. The combined duplicate line coverage never rises above the frozen baseline
   (monotonic non-increase).
3. The ledger contains no stale region ids that the current tree no longer
   produces (stale entries must be pruned).

It also validates ledger integrity: every category is one of the allowed
values, and every ``numeric_sensitive_exempt`` region names its lockstep
tests, matching G7's requirement that numeric primal/AD duplicates stay exempt
only with lockstep coverage and owner comments.

Run without arguments to gate; pass ``--print-current`` to print the current
statistics for humans. The gate runs in the nightly tier of
``ci/run_ci_tier.py`` (``nightly.duplication``).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import duplication_report  # noqa: E402

DEFAULT_LEDGER = ROOT / "docs" / "dev" / "audit" / "duplication-classification.json"
ALLOWED_CATEGORIES = frozenset(
    {
        "host_check",
        "indexing_packing",
        "numeric_sensitive_exempt",
        "fixture_boilerplate",
        "other",
    }
)
_COVERAGE_EPSILON = 1e-9


class LedgerError(RuntimeError):
    """Raised when the ledger is structurally invalid."""


def load_ledger(path: Path) -> dict[str, object]:
    ledger = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(ledger, dict):
        raise LedgerError("ledger root must be an object")
    if not isinstance(ledger.get("regions"), dict):
        raise LedgerError("ledger.regions must be an object")
    baseline = ledger.get("baseline")
    if not isinstance(baseline, dict) or "coverage_percent" not in baseline:
        raise LedgerError("ledger.baseline.coverage_percent is required")
    return ledger


def _validate_ledger_entries(ledger: dict[str, object]) -> list[str]:
    problems: list[str] = []
    regions: dict[str, object] = ledger["regions"]  # type: ignore[assignment]
    for region_id, entry in regions.items():
        if not isinstance(entry, dict):
            problems.append(f"ledger entry {region_id} is not an object")
            continue
        category = entry.get("category")
        if category not in ALLOWED_CATEGORIES:
            problems.append(
                f"ledger entry {region_id} has invalid category {category!r}"
            )
        if not entry.get("owner"):
            problems.append(f"ledger entry {region_id} is missing an owner")
        if not entry.get("reason"):
            problems.append(f"ledger entry {region_id} is missing a reason")
        if category == "numeric_sensitive_exempt" and not entry.get(
            "lockstep_tests"
        ):
            problems.append(
                f"numeric_sensitive_exempt region {region_id} is missing lockstep_tests"
            )
    return problems


def evaluate(
    report: dict[str, object], ledger: dict[str, object]
) -> list[str]:
    """Return deterministic gate violations; empty means the gate passes."""

    problems = _validate_ledger_entries(ledger)

    current_ids = duplication_report.region_ids(report)
    ledger_ids = set(ledger["regions"])  # type: ignore[arg-type]

    for region_id in sorted(current_ids - ledger_ids):
        problems.append(f"unclassified region: {region_id}")
    for region_id in sorted(ledger_ids - current_ids):
        problems.append(f"stale ledger entry: {region_id}")

    combined = report["combined"]  # type: ignore[index]
    current_coverage = float(combined["coverage_percent"])  # type: ignore[index]
    baseline_coverage = float(ledger["baseline"]["coverage_percent"])  # type: ignore[index]
    if current_coverage > baseline_coverage + _COVERAGE_EPSILON:
        problems.append(
            "coverage regression: combined duplicate coverage "
            f"{current_coverage:.6f}% exceeds frozen baseline "
            f"{baseline_coverage:.6f}%"
        )
    return problems


def _print_current(report: dict[str, object], ledger: dict[str, object]) -> None:
    from collections import Counter

    combined = report["combined"]  # type: ignore[index]
    corpora = report["corpora"]  # type: ignore[index]
    baseline = float(ledger["baseline"]["coverage_percent"])  # type: ignore[index]
    print("exact-token duplicate coverage (>= "
          f"{report['min_tokens']} tokens)")
    for name in ("python", "native"):
        item = corpora[name]  # type: ignore[index]
        print(
            f"  {name:7} regions={item['region_count']:4} "
            f"duplicate_lines={item['duplicate_lines']:6} "
            f"total_lines={item['total_lines']:6} "
            f"coverage={item['coverage_percent']:.4f}%"
        )
    print(
        f"  {'combined':7} regions={combined['region_count']:4} "
        f"duplicate_lines={combined['duplicate_lines']:6} "
        f"total_lines={combined['total_lines']:6} "
        f"coverage={combined['coverage_percent']:.4f}%"
    )
    print(f"  frozen baseline coverage={baseline:.4f}%")

    categories = Counter(
        str(entry.get("category"))
        for entry in ledger["regions"].values()  # type: ignore[attr-defined]
    )
    print("ledger categories:")
    for category in sorted(categories):
        print(f"  {category:26} {categories[category]}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo",
        type=Path,
        default=ROOT,
        help="repository root (defaults to this repository)",
    )
    parser.add_argument(
        "--ledger",
        type=Path,
        default=DEFAULT_LEDGER,
        help="classification ledger JSON",
    )
    parser.add_argument(
        "--print-current",
        action="store_true",
        help="print current duplication statistics and exit 0",
    )
    args = parser.parse_args(argv)
    repo = args.repo.resolve()

    try:
        ledger = load_ledger(args.ledger)
        min_tokens = int(ledger.get("min_tokens", duplication_report.MIN_TOKENS))
        report = duplication_report.build_report(repo, min_tokens)
    except (LedgerError, OSError, ValueError, json.JSONDecodeError) as error:
        print(f"duplication gate configuration error: {error}", file=sys.stderr)
        return 2

    if args.print_current:
        _print_current(report, ledger)
        return 0

    violations = evaluate(report, ledger)
    for violation in violations:
        print(violation, file=sys.stderr)
    if violations:
        print(
            f"duplication gate failed with {len(violations)} violation(s)",
            file=sys.stderr,
        )
        return 1
    print("duplication gate passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
