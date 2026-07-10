"""Locate the compiled ``_channel_native``/``_raydn`` extensions for tests and benchmarks.

The extensions live in out-of-tree CMake build directories under
``artifacts/cmake-*``. This helper prepends the newest build that matches the
running interpreter to ``sys.path`` so bare ``import _channel_native`` works
without an externally managed PYTHONPATH.
"""

from __future__ import annotations

import importlib.machinery
import importlib.util
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _importable(module_name: str) -> bool:
    try:
        return importlib.util.find_spec(module_name) is not None
    except (ImportError, ValueError):
        return False


def _candidate_build_dirs() -> list[Path]:
    artifacts = _REPO_ROOT / "artifacts"
    if not artifacts.is_dir():
        return []
    candidates: list[tuple[float, Path]] = []
    for build_dir in artifacts.glob("cmake-*"):
        for suffix in importlib.machinery.EXTENSION_SUFFIXES:
            pyd = build_dir / f"_channel_native{suffix}"
            if pyd.is_file():
                candidates.append((pyd.stat().st_mtime, build_dir))
                break
    candidates.sort(key=lambda item: item[0], reverse=True)
    return [build_dir for _mtime, build_dir in candidates]


def native_extensions_available() -> bool:
    return _importable("_channel_native") and _importable("_raydn")


def inject_native_paths() -> bool:
    """Make the native extensions importable; return True on success."""

    if native_extensions_available():
        return True
    for build_dir in _candidate_build_dirs():
        paths = [build_dir, build_dir / "ext" / "raydn"]
        inserted = [str(path) for path in paths if path.is_dir()]
        sys.path[:0] = inserted
        if native_extensions_available():
            return True
        for path in inserted:
            sys.path.remove(path)
    return False


BUILD_GUIDANCE = (
    "The compiled _channel_native/_raydn extensions were not found. "
    "Build them into artifacts/cmake-<name> (see docs/dev/plans/00-channel-native-greenfield-plan.md) "
    "or set PYTHONPATH to a directory containing the built pyds."
)
