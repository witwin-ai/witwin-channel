from __future__ import annotations

import hashlib
from pathlib import Path
import shutil
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]
VALIDATOR = ROOT / "cmake" / "validate_build_identity.cmake"
RAYD_REMOTE = "https://github.com/Asixa/RayD.git"


def _run(*args: str, cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        args,
        cwd=cwd,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )


def _git(*args: str, cwd: Path) -> str:
    git = shutil.which("git")
    assert git is not None
    result = _run(git, *args, cwd=cwd)
    assert result.returncode == 0, result.stdout + result.stderr
    return result.stdout.strip()


def _repository(path: Path, *, remote: str | None = None) -> str:
    path.mkdir()
    _git("init", cwd=path)
    _git("config", "user.name", "Channel Native Test", cwd=path)
    _git("config", "user.email", "test@example.invalid", cwd=path)
    (path / "tracked.txt").write_text("clean\n", encoding="utf-8")
    _git("add", "tracked.txt", cwd=path)
    _git("commit", "-m", "baseline", cwd=path)
    if remote is not None:
        _git("remote", "add", "origin", remote, cwd=path)
    return _git("rev-parse", "HEAD", cwd=path)


def _validate(
    channel: Path,
    rayd: Path,
    *,
    channel_sha: str,
    rayd_sha: str,
    channel_dirty: int = 0,
    rayd_dirty: int = 0,
    release: bool = False,
) -> subprocess.CompletedProcess[str]:
    abi = rayd / "integration.h"
    lock = channel / "rayd.lock.json"
    git = shutil.which("git")
    assert git is not None
    return _run(
        sys.executable,
        "-m",
        "cmake",
        f"-DGIT_EXECUTABLE={git}",
        f"-DCHANNEL_NATIVE_SOURCE_DIR={channel}",
        f"-DCHANNEL_NATIVE_EXPECTED_GIT_SHA={channel_sha}",
        f"-DCHANNEL_NATIVE_EXPECTED_GIT_DIRTY={channel_dirty}",
        f"-DCHANNEL_NATIVE_RAYD_SOURCE_DIR={rayd}",
        f"-DCHANNEL_NATIVE_EXPECTED_RAYD_SHA={rayd_sha}",
        f"-DCHANNEL_NATIVE_EXPECTED_RAYD_DIRTY={rayd_dirty}",
        f"-DCHANNEL_NATIVE_EXPECTED_RAYD_REMOTE={RAYD_REMOTE}",
        f"-DCHANNEL_NATIVE_RAYD_ABI_FILE={abi}",
        "-DCHANNEL_NATIVE_EXPECTED_RAYD_ABI_SHA256="
        + hashlib.sha256(b"abi-v1\n").hexdigest(),
        f"-DCHANNEL_NATIVE_RAYD_LOCK_FILE={lock}",
        "-DCHANNEL_NATIVE_EXPECTED_RAYD_LOCK_SHA256="
        + hashlib.sha256(b"lock-v1\n").hexdigest(),
        f"-DCHANNEL_NATIVE_RELEASE_BUILD={'ON' if release else 'OFF'}",
        "-P",
        str(VALIDATOR),
    )


def _identity_repositories(tmp_path: Path) -> tuple[Path, Path, str, str]:
    channel = tmp_path / "channel"
    rayd = tmp_path / "rayd"
    channel_sha = _repository(channel)
    rayd_sha = _repository(rayd, remote=RAYD_REMOTE)
    (channel / "rayd.lock.json").write_bytes(b"lock-v1\n")
    (rayd / "integration.h").write_bytes(b"abi-v1\n")
    _git("add", "rayd.lock.json", cwd=channel)
    _git("commit", "-m", "add lock", cwd=channel)
    channel_sha = _git("rev-parse", "HEAD", cwd=channel)
    _git("add", "integration.h", cwd=rayd)
    _git("commit", "-m", "add ABI", cwd=rayd)
    rayd_sha = _git("rev-parse", "HEAD", cwd=rayd)
    return channel, rayd, channel_sha, rayd_sha


def test_build_identity_validator_accepts_unchanged_checkouts(tmp_path: Path):
    channel, rayd, channel_sha, rayd_sha = _identity_repositories(tmp_path)

    result = _validate(
        channel,
        rayd,
        channel_sha=channel_sha,
        rayd_sha=rayd_sha,
    )

    assert result.returncode == 0, result.stdout + result.stderr


def test_build_identity_validator_rejects_stale_git_state(tmp_path: Path):
    channel, rayd, channel_sha, rayd_sha = _identity_repositories(tmp_path)
    (channel / "tracked.txt").write_text("dirty\n", encoding="utf-8")

    result = _validate(
        channel,
        rayd,
        channel_sha=channel_sha,
        rayd_sha=rayd_sha,
    )

    output = result.stdout + result.stderr
    assert result.returncode != 0, output
    assert "dirty state changed after configure" in output


def test_release_identity_validator_rejects_dirty_checkout(tmp_path: Path):
    channel, rayd, channel_sha, rayd_sha = _identity_repositories(tmp_path)
    (rayd / "tracked.txt").write_text("dirty\n", encoding="utf-8")

    result = _validate(
        channel,
        rayd,
        channel_sha=channel_sha,
        rayd_sha=rayd_sha,
        rayd_dirty=1,
        release=True,
    )

    output = result.stdout + result.stderr
    assert result.returncode != 0, output
    assert "Release build forbids a dirty RayD checkout" in output


def test_build_identity_validator_rejects_stale_abi_and_lock(tmp_path: Path):
    channel, rayd, channel_sha, rayd_sha = _identity_repositories(tmp_path)
    (rayd / "integration.h").write_bytes(b"abi-v2\n")

    abi_result = _validate(
        channel,
        rayd,
        channel_sha=channel_sha,
        rayd_sha=rayd_sha,
        rayd_dirty=1,
    )

    abi_output = abi_result.stdout + abi_result.stderr
    assert abi_result.returncode != 0, abi_output
    assert "integration ABI changed after configure" in abi_output

    (rayd / "integration.h").write_bytes(b"abi-v1\n")
    (channel / "rayd.lock.json").write_bytes(b"lock-v2\n")
    lock_result = _validate(
        channel,
        rayd,
        channel_sha=channel_sha,
        rayd_sha=rayd_sha,
        channel_dirty=1,
    )

    lock_output = lock_result.stdout + lock_result.stderr
    assert lock_result.returncode != 0, lock_output
    assert "RayD lock changed after configure" in lock_output
