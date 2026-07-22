from __future__ import annotations

import json
from pathlib import Path
import random
import string
import subprocess

import pytest

from ci import check_secrets as secrets


ROOT = Path(__file__).resolve().parents[1]


def _entropy_token(seed: int = 20260716, length: int = 48) -> str:
    """Build a high-entropy base64 token at runtime.

    Constructing the token dynamically keeps this source file free of any
    literal credential, so the repository-wide scan never trips over the test
    fixtures themselves.
    """

    rng = random.Random(seed)
    alphabet = string.ascii_letters + string.digits
    while True:
        token = "".join(rng.choice(alphabet) for _ in range(length))
        has_diversity = (
            any(character.islower() for character in token)
            and any(character.isupper() for character in token)
            and any(character.isdigit() for character in token)
        )
        if has_diversity and secrets._shannon_entropy(token) >= secrets._ENTROPY_THRESHOLD:
            return token


def _init_repo(root: Path) -> None:
    subprocess.run(["git", "init", "-q", str(root)], check=True)


def _stage(root: Path, relative: str, content: str) -> None:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    subprocess.run(["git", "-C", str(root), "add", relative], check=True)


def test_pattern_families_are_detected() -> None:
    aws = "AKIA" + "1234567890ABCDEF"
    github = "ghp_" + "a" * 36
    slack = "xox" + "b-" + "0123456789abcdef"
    pem = "-----BEGIN " + "RSA PRIVATE KEY-----"
    password_line = "password = " + chr(34) + "S3cr3tValue" + chr(34)

    rules = {
        finding.rule
        for line in (aws, github, slack, pem, password_line)
        for finding in secrets._scan_line("sample.txt", 1, line)
    }

    assert {
        "aws-access-key-id",
        "github-token",
        "slack-token",
        "private-key-block",
        "password-literal",
    } <= rules


def test_high_entropy_token_is_flagged_and_hex_digest_is_ignored() -> None:
    token = _entropy_token()
    hex_digest = "a" * 8 + "b3c9" * 14  # lowercase hex has no upper-case diversity

    entropy_rules = [finding.rule for finding in secrets._scan_line("f", 1, token)]
    digest_rules = [finding.rule for finding in secrets._scan_line("f", 1, hex_digest)]

    assert entropy_rules == ["high-entropy-base64"]
    assert "high-entropy-base64" not in digest_rules


def test_password_placeholders_are_not_flagged() -> None:
    for value in ("changeme", "xxxxxxxx", "${DB_PASSWORD}", "os.environ['PW']"):
        line = "password = " + chr(34) + value + chr(34)
        assert secrets._scan_line("f", 1, line) == []


def test_scan_repository_reads_the_git_index(tmp_path: Path) -> None:
    _init_repo(tmp_path)
    _stage(tmp_path, "clean.py", "value = 1\n")
    token = "ghp_" + "b" * 36
    _stage(tmp_path, "leak.env", "GITHUB_TOKEN=" + token + "\n")
    untracked = tmp_path / "ignored.txt"
    untracked.write_text("AKIA" + "ABCDEF0123456789" + "\n", encoding="utf-8")

    findings = secrets.scan_repository(secrets.repository_root(tmp_path))

    assert [finding.path for finding in findings] == ["leak.env"]
    assert findings[0].rule == "github-token"


def test_allowlist_suppresses_a_pinned_finding(tmp_path: Path) -> None:
    _init_repo(tmp_path)
    token = "ghp_" + "c" * 36
    _stage(tmp_path, "leak.env", "GITHUB_TOKEN=" + token + "\n")
    root = secrets.repository_root(tmp_path)
    findings = secrets.scan_repository(root)
    assert findings

    allowlist = {
        (findings[0].path, findings[0].rule, findings[0].fingerprint)
    }
    assert secrets.filter_findings(findings, allowlist) == []


def test_binary_and_skipped_suffixes_are_not_scanned(tmp_path: Path) -> None:
    _init_repo(tmp_path)
    token = "ghp_" + "d" * 36
    _stage(tmp_path, "asset.png", "GITHUB_TOKEN=" + token + "\n")
    _stage(tmp_path, "blob.txt", "prefix\x00GITHUB_TOKEN=" + token + "\n")

    assert secrets.scan_repository(secrets.repository_root(tmp_path)) == []


def test_load_allowlist_rejects_malformed_documents(tmp_path: Path) -> None:
    path = tmp_path / "allowlist.json"
    path.write_text(json.dumps({"schema_version": 2, "allowed": []}), encoding="utf-8")
    with pytest.raises(ValueError):
        secrets.load_allowlist(path)

    path.write_text(
        json.dumps({"schema_version": 1, "allowed": [{"path": "a"}]}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError):
        secrets.load_allowlist(path)

    assert secrets.load_allowlist(tmp_path / "missing.json") == set()


def test_repository_scan_is_clean_under_the_committed_allowlist() -> None:
    root = secrets.repository_root(ROOT)
    allowlist = secrets.load_allowlist(root / secrets.DEFAULT_ALLOWLIST_PATH)
    assert secrets.filter_findings(secrets.scan_repository(root), allowlist) == []
