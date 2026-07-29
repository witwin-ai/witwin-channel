# Copyright Xingyu Chen.
# Scan tracked files for committed credentials and high-entropy secrets.

"""Scan tracked files for committed credentials and high-entropy secrets.

The gate reads the git index (tracked files only), skips known binary assets,
and reports any line that matches a credential pattern or contains a
high-entropy base64 token. Findings can be suppressed with an allowlist file
that pins each false positive by ``(path, rule, fingerprint)`` so an unrelated
match cannot silently reuse an exemption.

The scan is intentionally cheap: a handful of compiled patterns run line by
line over text blobs below a size cap, so the quick CI tier stays fast.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
import sys


DEFAULT_ALLOWLIST_PATH = Path("ci/secret-scan-allowlist.json")
_MAX_SCAN_BYTES = 5 * 1024 * 1024
_ENTROPY_MIN_LENGTH = 40
_ENTROPY_THRESHOLD = 4.5

_SKIP_SUFFIXES = frozenset(
    {
        ".bin",
        ".bmp",
        ".dll",
        ".exe",
        ".gif",
        ".gz",
        ".ico",
        ".jpeg",
        ".jpg",
        ".lib",
        ".npy",
        ".npz",
        ".obj",
        ".pdf",
        ".pkl",
        ".ply",
        ".png",
        ".pt",
        ".pth",
        ".pyd",
        ".so",
        ".webp",
        ".whl",
        ".zip",
    }
)

_PLACEHOLDER_VALUES = frozenset(
    {
        "password",
        "passwd",
        "changeme",
        "example",
        "placeholder",
        "redacted",
        "secret",
        "yourpassword",
    }
)


class GitCommandError(RuntimeError):
    """Raised when the git index cannot be inspected."""


@dataclass(frozen=True, slots=True)
class Pattern:
    name: str
    regex: re.Pattern[str]
    value_group: int


@dataclass(frozen=True, order=True, slots=True)
class Finding:
    path: str
    line: int
    column: int
    rule: str
    fingerprint: str


_PATTERNS = (
    Pattern(
        "private-key-block",
        re.compile(r"-----BEGIN(?: [A-Z0-9]+)* PRIVATE KEY-----"),
        0,
    ),
    Pattern(
        "aws-access-key-id",
        re.compile(r"\b(?:AKIA|ASIA|AGPA|AIDA|AROA|ANPA|ANVA)[0-9A-Z]{16}\b"),
        0,
    ),
    Pattern(
        "github-token",
        re.compile(r"\b(?:ghp|gho|ghu|ghs|ghr)_[0-9A-Za-z]{36,}\b"),
        0,
    ),
    Pattern(
        "github-pat",
        re.compile(r"\bgithub_pat_[0-9A-Za-z_]{22,}\b"),
        0,
    ),
    Pattern(
        "slack-token",
        re.compile(r"\bxox[baprs]-[0-9A-Za-z-]{10,}\b"),
        0,
    ),
    Pattern(
        "google-api-key",
        re.compile(r"\bAIza[0-9A-Za-z_\-]{35}\b"),
        0,
    ),
    Pattern(
        "google-oauth-token",
        re.compile(r"\bya29\.[0-9A-Za-z_\-]{20,}\b"),
        0,
    ),
    Pattern(
        "password-literal",
        re.compile(r"(?i)(?:password|passwd|pwd)\s*[:=]\s*[\"']([^\"'\s]{6,})[\"']"),
        1,
    ),
)

_B64_TOKEN = re.compile(r"[A-Za-z0-9+/]{%d,}={0,2}" % _ENTROPY_MIN_LENGTH)


def _git(root: Path, *args: str) -> bytes:
    command = ["git", "-C", str(root), *args]
    completed = subprocess.run(
        command,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if completed.returncode:
        error = completed.stderr.decode("utf-8", errors="replace").strip()
        raise GitCommandError(f"{' '.join(command)} failed: {error}")
    return completed.stdout


def repository_root(path: Path) -> Path:
    output = _git(path, "rev-parse", "--show-toplevel")
    return Path(output.decode("utf-8", errors="surrogateescape").strip())


def tracked_files(root: Path) -> list[str]:
    output = _git(root, "ls-files", "-z")
    return [
        record.decode("utf-8", errors="surrogateescape")
        for record in output.split(b"\0")
        if record
    ]


def _fingerprint(secret: str) -> str:
    return hashlib.sha256(secret.encode("utf-8", errors="surrogateescape")).hexdigest()


def _shannon_entropy(text: str) -> float:
    counts = Counter(text)
    length = len(text)
    return -sum(
        (count / length) * math.log2(count / length) for count in counts.values()
    )


def _is_placeholder(value: str) -> bool:
    stripped = value.strip()
    folded = stripped.casefold()
    if folded in _PLACEHOLDER_VALUES:
        return True
    if len(set(stripped)) <= 1:
        return True
    if "${" in stripped or "os.environ" in stripped or "getenv" in stripped:
        return True
    return set(folded) <= set("x*.-_")


def _high_entropy_tokens(line: str) -> list[tuple[int, str]]:
    tokens: list[tuple[int, str]] = []
    for match in _B64_TOKEN.finditer(line):
        token = match.group(0)
        core = token.rstrip("=")
        if len(core) < _ENTROPY_MIN_LENGTH:
            continue
        has_lower = any(character.islower() for character in core)
        has_upper = any(character.isupper() for character in core)
        has_digit = any(character.isdigit() for character in core)
        if not (has_lower and has_upper and has_digit):
            continue
        if _shannon_entropy(core) < _ENTROPY_THRESHOLD:
            continue
        tokens.append((match.start(), token))
    return tokens


def _scan_line(path: str, lineno: int, line: str) -> list[Finding]:
    findings: list[Finding] = []
    for pattern in _PATTERNS:
        for match in pattern.regex.finditer(line):
            value = match.group(pattern.value_group)
            if pattern.name == "password-literal" and _is_placeholder(value):
                continue
            findings.append(
                Finding(
                    path,
                    lineno,
                    match.start() + 1,
                    pattern.name,
                    _fingerprint(match.group(0)),
                )
            )
    for column, token in _high_entropy_tokens(line):
        findings.append(
            Finding(
                path,
                lineno,
                column + 1,
                "high-entropy-base64",
                _fingerprint(token),
            )
        )
    return findings


def scan_repository(root: Path) -> list[Finding]:
    findings: list[Finding] = []
    for relative in tracked_files(root):
        path = root / relative
        if path.suffix.casefold() in _SKIP_SUFFIXES:
            continue
        try:
            data = path.read_bytes()
        except OSError:
            continue
        if not data or len(data) > _MAX_SCAN_BYTES or b"\x00" in data:
            continue
        text = data.decode("utf-8", errors="replace")
        for lineno, line in enumerate(text.splitlines(), start=1):
            findings.extend(_scan_line(relative, lineno, line))
    return sorted(findings)


def load_allowlist(path: Path) -> set[tuple[str, str, str]]:
    if not path.is_file():
        return set()
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or data.get("schema_version") != 1:
        raise ValueError("secret-scan allowlist must use schema_version 1")
    allowed = data.get("allowed")
    if not isinstance(allowed, list):
        raise ValueError("secret-scan allowlist must contain an 'allowed' list")
    entries: set[tuple[str, str, str]] = set()
    for entry in allowed:
        if not isinstance(entry, dict):
            raise ValueError("secret-scan allowlist entries must be objects")
        path_value = entry.get("path")
        rule = entry.get("rule")
        fingerprint = entry.get("fingerprint")
        reason = entry.get("reason")
        if not (
            isinstance(path_value, str)
            and path_value
            and isinstance(rule, str)
            and rule
            and isinstance(fingerprint, str)
            and fingerprint
            and isinstance(reason, str)
            and reason
        ):
            raise ValueError(
                "secret-scan allowlist entries need path, rule, fingerprint, reason"
            )
        entries.add((path_value, rule, fingerprint))
    return entries


def filter_findings(
    findings: list[Finding], allowlist: set[tuple[str, str, str]]
) -> list[Finding]:
    return [
        finding
        for finding in findings
        if (finding.path, finding.rule, finding.fingerprint) not in allowlist
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "root",
        nargs="?",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="repository to inspect (defaults to this repository)",
    )
    parser.add_argument(
        "--allowlist",
        type=Path,
        help="allowlist JSON (defaults to ci/secret-scan-allowlist.json under root)",
    )
    args = parser.parse_args(argv)

    try:
        root = repository_root(args.root.resolve())
        allowlist_path = args.allowlist or root / DEFAULT_ALLOWLIST_PATH
        allowlist = load_allowlist(allowlist_path)
        findings = filter_findings(scan_repository(root), allowlist)
    except (GitCommandError, OSError, ValueError, json.JSONDecodeError) as error:
        print(f"secret scan configuration error: {error}", file=sys.stderr)
        return 2

    for finding in findings:
        print(
            f"{finding.rule}: {finding.path}:{finding.line}:{finding.column}: "
            f"fingerprint {finding.fingerprint}",
            file=sys.stderr,
        )
    if findings:
        print(
            f"secret scan found {len(findings)} potential credential(s)",
            file=sys.stderr,
        )
        return 1
    print("secret scan passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())