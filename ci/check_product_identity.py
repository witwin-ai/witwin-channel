"""Reject predecessor Channel product identities outside frozen history."""

from __future__ import annotations

import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


_FORMER_IDENTITY = re.compile(
    r"channel(?:_|-| )native|\bcn_",
    flags=re.IGNORECASE,
)
_HISTORICAL_PREFIXES = (
    "benchmarks/baselines/",
    "benchmarks/gates/",
    "docs/dev/audit/",
    "docs/dev/baselines/",
    "docs/dev/plans/",
)
_HISTORICAL_FILES = frozenset(
    {
        "docs/dev/replacement/path-threeway-shadow-attempt-2026-07-11.json",
        "docs/dev/standards/adr-033-channel-replacement-product-identity.md",
        "tests/test_native_owner_inventory.py",
        "tests/test_phase10_legacy_dead_binding_audit.py",
    }
)
_ROOT_NAME_FILES = frozenset({"AGENTS.md", "CLAUDE.md"})
_ROOT_NAME_LINE = "`" + "channel" + "_native/` and"


@dataclass(frozen=True, slots=True)
class Finding:
    path: str
    line: int
    text: str


def _is_historical(path: str) -> bool:
    normalized = path.replace("\\", "/")
    return normalized in _HISTORICAL_FILES or normalized.startswith(
        _HISTORICAL_PREFIXES
    )


def scan_text(path: str, source: str) -> list[Finding]:
    normalized = path.replace("\\", "/")
    if _is_historical(normalized):
        return []

    findings: list[Finding] = []
    for line_number, line in enumerate(source.splitlines(), start=1):
        matches = list(_FORMER_IDENTITY.finditer(line))
        if not matches:
            continue
        if normalized in _ROOT_NAME_FILES and line.strip() == _ROOT_NAME_LINE:
            continue
        findings.append(Finding(normalized, line_number, line.strip()))
    return findings


def _tracked_paths(root: Path) -> list[str]:
    completed = subprocess.run(
        ["git", "-C", str(root), "ls-files", "-z"],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if completed.returncode:
        error = completed.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"git ls-files failed: {error}")
    return sorted(
        record.decode("utf-8", errors="surrogateescape")
        for record in completed.stdout.split(b"\0")
        if record
    )


def scan_repository(root: Path) -> list[Finding]:
    findings: list[Finding] = []
    for relative in _tracked_paths(root):
        path = root / relative
        if not path.is_file() or _is_historical(relative):
            continue
        data = path.read_bytes()
        if b"\0" in data:
            continue
        source = data.decode("utf-8", errors="replace")
        findings.extend(scan_text(relative, source))
    return findings


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    try:
        findings = scan_repository(root)
    except (OSError, RuntimeError) as error:
        print(error, file=sys.stderr)
        return 2
    for finding in findings:
        print(f"{finding.path}:{finding.line}: {finding.text}", file=sys.stderr)
    if findings:
        print(
            f"product identity check failed with {len(findings)} predecessor reference(s)",
            file=sys.stderr,
        )
        return 1
    print("product identity check passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
