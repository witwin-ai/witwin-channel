"""Select an explicit development extension for tests and benchmarks.

The extensions live in out-of-tree CMake build directories under
``artifacts/cmake-*``. This helper configures the production loader with the
newest extension matching the running interpreter. It never makes a bare
``_channel`` module globally importable.
"""

from __future__ import annotations

import importlib.machinery
import importlib.util
import os
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_ENABLE_ENV = "WITWIN_CHANNEL_DEVELOPER_OVERRIDE"
_PATH_ENV = "WITWIN_CHANNEL_EXTENSION_PATH"
_FINGERPRINT_ENV = "WITWIN_CHANNEL_EXPECTED_FINGERPRINT"
_FINGERPRINT_FILE = "_channel.build-fingerprint"


def _importable(module_name: str) -> bool:
    try:
        return importlib.util.find_spec(module_name) is not None
    except (ImportError, ValueError):
        return False


def _candidate_extensions() -> list[Path]:
    artifacts = _REPO_ROOT / "artifacts"
    if not artifacts.is_dir():
        return []
    candidates: list[tuple[float, Path]] = []
    for build_dir in artifacts.glob("cmake-*"):
        for suffix in importlib.machinery.EXTENSION_SUFFIXES:
            pyd = build_dir / f"_channel{suffix}"
            if pyd.is_file():
                candidates.append((pyd.stat().st_mtime, pyd.resolve()))
                break
    candidates.sort(key=lambda item: item[0], reverse=True)
    return [extension for _mtime, extension in candidates]


def native_extensions_available() -> bool:
    if _importable("witwin.channel._channel"):
        return True
    configured = os.environ.get(_PATH_ENV)
    fingerprint = os.environ.get(_FINGERPRINT_ENV)
    return (
        os.environ.get(_ENABLE_ENV) == "1"
        and configured is not None
        and Path(configured).is_absolute()
        and Path(configured).is_file()
        and fingerprint is not None
        and len(fingerprint) == 64
        and all(character in "0123456789abcdef" for character in fingerprint)
    )


def inject_native_paths() -> bool:
    """Configure an explicit extension path; return ``True`` on success."""

    if native_extensions_available():
        return True
    if any(
        os.environ.get(name) is not None
        for name in (_ENABLE_ENV, _PATH_ENV, _FINGERPRINT_ENV)
    ):
        return False
    candidates = _candidate_extensions()
    for candidate in candidates:
        sidecar = candidate.with_name(_FINGERPRINT_FILE)
        if not sidecar.is_file():
            continue
        fingerprint = sidecar.read_text(encoding="ascii").strip()
        if len(fingerprint) != 64 or any(
            character not in "0123456789abcdef" for character in fingerprint
        ):
            continue
        os.environ[_ENABLE_ENV] = "1"
        os.environ[_PATH_ENV] = str(candidate)
        os.environ[_FINGERPRINT_ENV] = fingerprint
        return True
    return False


BUILD_GUIDANCE = (
    "A compiled _channel extension with its build-fingerprint sidecar "
    "was not found. Build into artifacts/cmake-<name> (see "
    "docs/dev/plans/00-channel-greenfield-plan.md), or configure the "
    "three WITWIN_CHANNEL developer loader variables explicitly."
)
