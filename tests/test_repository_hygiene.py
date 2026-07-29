# Copyright Xingyu Chen.
# Tests repository hygiene.

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "ci" / "check_repository_hygiene.py"
SPEC = importlib.util.spec_from_file_location("check_repository_hygiene", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
hygiene = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = hygiene
SPEC.loader.exec_module(hygiene)


def _git(repo: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True)


def _track(repo: Path, relative_path: str, content: bytes = b"content\n") -> None:
    path = repo / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    _git(repo, "add", "--", relative_path)


@pytest.fixture
def git_repo(tmp_path: Path) -> Path:
    _git(tmp_path, "init", "--quiet")
    _git(tmp_path, "config", "user.email", "repository-hygiene@example.invalid")
    _git(tmp_path, "config", "user.name", "Repository Hygiene Tests")
    _track(tmp_path, "README.md")
    _git(tmp_path, "commit", "--quiet", "-m", "initial")
    return tmp_path


def test_clean_repository_passes(git_repo: Path):
    assert hygiene.scan_repository(git_repo) == []
    assert hygiene.worktree_changes(git_repo) == []
    assert hygiene.main([str(git_repo)]) == 0


def test_forbidden_tracked_build_and_temporary_artifacts_are_reported(
    git_repo: Path,
):
    _track(git_repo, "build-local/output.txt")
    _track(git_repo, "src/pkg/__pycache__/module.pyc")
    _track(git_repo, "native/generated.dll")

    violations = hygiene.scan_repository(git_repo)

    assert {(violation.kind, violation.path) for violation in violations} == {
        ("forbidden-path", "build-local/output.txt"),
        ("forbidden-path", "native/generated.dll"),
        ("forbidden-path", "src/pkg/__pycache__/module.pyc"),
    }


def test_oversized_gate_reads_the_tracked_blob_from_the_index(git_repo: Path):
    _track(git_repo, "data.bin", b"x" * 32)
    (git_repo / "data.bin").write_bytes(b"x")

    violations = hygiene.scan_repository(git_repo, max_tracked_bytes=16)

    assert [(violation.kind, violation.path) for violation in violations] == [
        ("oversized-file", "data.bin")
    ]


def test_default_cli_rejects_dirty_tree_but_can_scan_during_local_work(
    git_repo: Path,
):
    (git_repo / "generated.txt").write_text("test output\n", encoding="utf-8")

    assert hygiene.main([str(git_repo)]) == 1
    assert hygiene.main([str(git_repo), "--allow-dirty"]) == 0