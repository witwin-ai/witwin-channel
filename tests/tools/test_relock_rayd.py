# Copyright Xingyu Chen.
# Tests the RayD re-lock tool against the committed lock and a synthetic RayD repository.

"""Tests the RayD re-lock tool against the committed lock and a synthetic RayD repository."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from tools.relock_rayd import RelockError, render_lock, replacements

ROOT = Path(__file__).resolve().parents[2]
LOCK_PATH = ROOT / "dependencies" / "rayd.lock.json"


def _lock() -> dict:
    return json.loads(LOCK_PATH.read_text(encoding="utf-8"))


def test_render_reproduces_the_committed_lock_bytes() -> None:
    assert render_lock(_lock()) == LOCK_PATH.read_text(encoding="utf-8")


def test_replacements_map_only_changed_pinned_values() -> None:
    old = _lock()
    new = json.loads(json.dumps(old))
    new["commit"] = "1" * 40
    new["integration_abi"]["headers"][0]["sha256"] = "2" * 64

    assert replacements(old, new) == {
        old["commit"]: "1" * 40,
        old["integration_abi"]["headers"][0]["sha256"]: "2" * 64,
    }
    assert replacements(old, old) == {}


def test_replacements_refuse_a_changed_header_set() -> None:
    old = _lock()
    new = json.loads(json.dumps(old))
    new["integration_abi"]["headers"].pop()

    with pytest.raises(RelockError, match="header set changed"):
        replacements(old, new)


def _git(*arguments: str, cwd: Path) -> str:
    return subprocess.run(
        ("git", *arguments), cwd=cwd, check=True, capture_output=True, text=True
    ).stdout.strip()


@pytest.mark.skipif(shutil.which("git") is None, reason="git is required")
def test_relock_rewrites_lockstep_files_and_is_idempotent(tmp_path: Path) -> None:
    rayd_source = os.environ.get("RAYD_SOURCE_DIR")
    if not rayd_source or not (Path(rayd_source) / ".git").exists():
        pytest.skip("RAYD_SOURCE_DIR does not name a RayD checkout")
    old = _lock()
    channel = tmp_path / "channel"
    (channel / "dependencies").mkdir(parents=True)
    (channel / "tests").mkdir()
    (channel / "docs/dev/standards").mkdir(parents=True)
    (channel / "dependencies/rayd.lock.json").write_text(render_lock(old), encoding="utf-8")
    pinned = f'LOCKED = "{old["commit"]}"\n'
    (channel / "tests/test_pin.py").write_text(pinned, encoding="utf-8")
    (channel / "docs/dev/standards/adr.md").write_text(pinned, encoding="utf-8")
    _git("init", "-q", cwd=channel)
    _git("add", ".", cwd=channel)

    command = (
        sys.executable, str(ROOT / "tools/relock_rayd.py"), "--rayd", rayd_source,
        "--rev", old["commit"], "--root", str(channel),
    )
    first = subprocess.run(command, check=False, capture_output=True, text=True)
    assert first.returncode == 0, first.stderr
    assert "nothing changed" in first.stdout

    parent = _git("rev-parse", f"{old['commit']}^", cwd=Path(rayd_source))
    moved = subprocess.run(
        (*command[:-4], "--rev", parent, "--root", str(channel)),
        check=False, capture_output=True, text=True,
    )
    assert moved.returncode == 0, moved.stderr
    assert json.loads((channel / "dependencies/rayd.lock.json").read_text())["commit"] == parent
    assert parent in (channel / "tests/test_pin.py").read_text(encoding="utf-8")
    assert old["commit"] in (channel / "docs/dev/standards/adr.md").read_text(encoding="utf-8")

    again = subprocess.run(
        (*command[:-4], "--rev", parent, "--root", str(channel), "--check"),
        check=False, capture_output=True, text=True,
    )
    assert again.returncode == 0, again.stdout
    assert "nothing changed" in again.stdout
