from __future__ import annotations

import importlib
import importlib.util
import pathlib
import sys
from types import ModuleType


def _repo_root() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parents[5]


def _candidate_dirs() -> list[pathlib.Path]:
    root = _repo_root()
    dirs = [
        root / "artifacts" / "cmake-channel-native-raydn-witwin2-release",
        root / "artifacts" / "cmake-witwin2-explicit-release",
        root / "artifacts" / "cmake-witwin2-release",
        root / "artifacts" / "cmake-witwin2",
        root / "artifacts",
    ]
    return [path for path in dirs if path.is_dir()]


def _load_from_path(path: pathlib.Path) -> ModuleType | None:
    spec = importlib.util.spec_from_file_location("_raydn", path)
    if spec is None or spec.loader is None:
        return None
    module = importlib.util.module_from_spec(spec)
    sys.modules["_raydn"] = module
    spec.loader.exec_module(module)
    return module


def _artifact_candidates() -> list[pathlib.Path]:
    candidates: list[pathlib.Path] = []
    for directory in _candidate_dirs():
        candidates.extend(sorted(directory.rglob("_raydn*.pyd")))
        candidates.extend(sorted(directory.rglob("_raydn*.so")))
    return candidates


def native_extension() -> ModuleType | None:
    """Load the RayDN native extension without importing the Python raydn package."""

    existing = sys.modules.get("_raydn")
    if isinstance(existing, ModuleType):
        return existing

    try:
        return importlib.import_module("_raydn")
    except ModuleNotFoundError:
        pass
    except ImportError:
        pass

    for candidate in _artifact_candidates():
        try:
            return _load_from_path(candidate)
        except (ImportError, OSError):
            continue
    return None


def require_native_extension() -> ModuleType:
    native = native_extension()
    if native is None:
        raise RuntimeError("RayDN native extension _raydn is not built or loadable")
    return native


def capability_info() -> dict[str, bool | str]:
    native = native_extension()
    return {
        "uses_raydn_native": native is not None,
        "optix_available": native is not None,
        "raydn_extension_loaded": native is not None,
    }
