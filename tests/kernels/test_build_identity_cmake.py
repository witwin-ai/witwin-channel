# Copyright Xingyu Chen.
# Tests build identity cmake.

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]
VALIDATOR = ROOT / "cmake" / "validate_build_identity.cmake"
RAYD_REMOTE = "https://github.com/Asixa/RayD"


def _abi_aggregate(content: bytes) -> tuple[str, str]:
    header_sha = hashlib.sha256(content).hexdigest()
    aggregate = hashlib.sha256(f"include/rayd/integration.h\0{header_sha}\n".encode()).hexdigest()
    return header_sha, aggregate


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
    _git("config", "user.name", "Channel Test", cwd=path)
    _git("config", "user.email", "test@example.invalid", cwd=path)
    (path / "tracked.txt").write_text("clean\n", encoding="utf-8")
    _git("add", "tracked.txt", cwd=path)
    _git("commit", "-m", "baseline", cwd=path)
    if remote is not None:
        _git("remote", "add", "origin", remote, cwd=path)
    return _git("rev-parse", "HEAD", cwd=path)


def _validate(
    channel: Path, rayd: Path, *, channel_sha: str, rayd_sha: str, channel_dirty: int = 0,
    rayd_dirty: int = 0, release: bool = False, expected_lock_sha: str | None = None,
) -> subprocess.CompletedProcess[str]:
    abi = rayd / "include/rayd/integration.h"
    lock = channel / "rayd.lock.json"
    git = shutil.which("git")
    assert git is not None
    return _run(
        sys.executable,
        "-m",
        "cmake",
        f"-DGIT_EXECUTABLE={git}",
        f"-DCHANNEL_SOURCE_DIR={channel}",
        f"-DCHANNEL_EXPECTED_GIT_SHA={channel_sha}",
        f"-DCHANNEL_EXPECTED_GIT_DIRTY={channel_dirty}",
        f"-DCHANNEL_RAYD_SOURCE_DIR={rayd}",
        "-DCHANNEL_RAYD_SOURCE_KIND=git-checkout",
        f"-DCHANNEL_EXPECTED_RAYD_SHA={rayd_sha}",
        f"-DCHANNEL_EXPECTED_RAYD_DIRTY={rayd_dirty}",
        f"-DCHANNEL_EXPECTED_RAYD_REMOTE={RAYD_REMOTE}",
        f"-DCHANNEL_RAYD_ABI_FILE={abi}",
        "-DCHANNEL_EXPECTED_RAYD_ABI_SHA256=" + _abi_aggregate(b"abi-v1\n")[1],
        f"-DCHANNEL_RAYD_LOCK_FILE={lock}",
        "-DCHANNEL_EXPECTED_RAYD_LOCK_SHA256="
        + (expected_lock_sha or hashlib.sha256(lock.read_bytes()).hexdigest()),
        f"-DCHANNEL_PYTHON_EXECUTABLE={sys.executable}",
        f"-DCHANNEL_RAYD_RESOLVER={ROOT / 'cmake' / 'resolve_rayd_source.py'}",
        f"-DCHANNEL_RELEASE_BUILD={'ON' if release else 'OFF'}",
        "-P",
        str(VALIDATOR),
    )


def _identity_repositories(tmp_path: Path) -> tuple[Path, Path, str, str]:
    channel = tmp_path / "channel"
    rayd = tmp_path / "rayd"
    channel_sha = _repository(channel)
    rayd_sha = _repository(rayd, remote=RAYD_REMOTE)
    header_path = rayd / "include/rayd/integration.h"
    header_path.parent.mkdir(parents=True)
    header_bytes = b"abi-v1\n"
    header_path.write_bytes(header_bytes)
    (rayd / "torch").mkdir()
    (rayd / "torch/CMakeLists.txt").write_text("# fixture\n", encoding="utf-8")
    (rayd / "src").mkdir()
    (rayd / "src/field_transport_ad.cuh").write_text("// fixture\n", encoding="utf-8")
    (rayd / "src/transmission_device.cuh").write_text("// fixture\n", encoding="utf-8")
    header_sha, aggregate = _abi_aggregate(header_bytes)
    lock = {
        "schema_version": 2,
        "repository_url": RAYD_REMOTE,
        "commit": "0" * 40,
        "integration_abi": {
            "kind": "source-header-set-sha256",
            "entrypoint": "include/rayd/integration.h",
            "headers": [{"path": "include/rayd/integration.h", "sha256": header_sha}],
            "sha256": aggregate,
            "api_version": 8,
            "identity": "rayd.torch.integration",
        },
        "source_bundle": {
            "distribution": "rayd-torch",
            "distribution_version": "0.8.0",
            "metadata_path": "rayd/torch/_source/rayd-source.json",
            "manifest_sha256": "0" * 64,
        },
    }
    (channel / "rayd.lock.json").write_text(
        json.dumps(lock, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _git("add", "rayd.lock.json", cwd=channel)
    _git("commit", "-m", "add lock", cwd=channel)
    channel_sha = _git("rev-parse", "HEAD", cwd=channel)
    _git("add", "include", "torch", "src", cwd=rayd)
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
    (rayd / "include/rayd/integration.h").write_bytes(b"abi-mutated\n")

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

    (rayd / "include/rayd/integration.h").write_bytes(b"abi-v1\n")
    expected_lock_sha = hashlib.sha256((channel / "rayd.lock.json").read_bytes()).hexdigest()
    (channel / "rayd.lock.json").write_bytes(b"lock-v2\n")
    lock_result = _validate(
        channel,
        rayd,
        channel_sha=channel_sha,
        rayd_sha=rayd_sha,
        channel_dirty=1,
        expected_lock_sha=expected_lock_sha,
    )

    lock_output = lock_result.stdout + lock_result.stderr
    assert lock_result.returncode != 0, lock_output
    assert "RayD lock changed after configure" in lock_output
