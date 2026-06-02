"""Python helpers for the bundled native channel extension."""

from __future__ import annotations

import importlib.util
import os
import sys
from functools import lru_cache
from importlib import import_module
from pathlib import Path
from site import getsitepackages, getusersitepackages

_WINDOWS_DLL_DIR_HANDLES: list[object] = []
_MODULE_NAME = "witwin.channel._native._channel_native"


def _installed_native_dirs() -> list[Path]:
    search_roots: list[Path] = []
    try:
        search_roots.extend(Path(path).resolve() for path in getsitepackages())
    except Exception:
        pass
    try:
        search_roots.append(Path(getusersitepackages()).resolve())
    except Exception:
        pass

    native_dirs: list[Path] = []
    seen: set[Path] = set()
    for root in search_roots:
        if root in seen or not root.exists():
            continue
        seen.add(root)
        native_dir = root / "witwin" / "channel" / "_native"
        if native_dir.is_dir():
            native_dirs.append(native_dir)
    return native_dirs


def _configure_windows_dll_paths() -> None:
    if os.name != "nt" or not hasattr(os, "add_dll_directory"):
        return

    added_dirs: set[str] = set()
    path_entries = os.environ.get("PATH", "").split(os.pathsep)
    normalized_path_entries = {entry.lower() for entry in path_entries if entry}

    def _add_dir(target_dir: Path) -> None:
        if not target_dir.is_dir():
            return
        target_str = str(target_dir)
        if target_str in added_dirs:
            return
        # Keep the handle alive for the life of the process. On Windows the
        # search path entry is removed when the handle is closed or collected.
        _WINDOWS_DLL_DIR_HANDLES.append(os.add_dll_directory(target_str))
        if target_str.lower() not in normalized_path_entries:
            path_entries.insert(0, target_str)
            normalized_path_entries.add(target_str.lower())
        added_dirs.add(target_str)

    def _add_package_dir(package: str, relative: str | None = None) -> None:
        spec = importlib.util.find_spec(package)
        if spec is None:
            return
        search_locations = spec.submodule_search_locations
        if not search_locations:
            return
        package_dir = Path(next(iter(search_locations))).resolve()
        target_dir = package_dir if relative is None else (package_dir / relative).resolve()
        _add_dir(target_dir)

    _add_package_dir("drjit")
    _add_package_dir("torch", "lib")

    cuda_root = os.environ.get("CUDA_PATH") or os.environ.get("CUDA_HOME")
    if cuda_root:
        _add_dir(Path(cuda_root).resolve() / "bin")

    conda_prefix = Path(sys.prefix).resolve()
    _add_dir(conda_prefix / "Library" / "bin")
    _add_dir(conda_prefix / "Library" / "usr" / "bin")
    for native_dir in _installed_native_dirs():
        _add_dir(native_dir)
    os.environ["PATH"] = os.pathsep.join(path_entries)


def _extend_package_search_path() -> None:
    package_path = globals().get("__path__")
    if package_path is None:
        return
    for native_dir in _installed_native_dirs():
        native_dir_str = str(native_dir)
        if native_dir_str not in package_path:
            package_path.append(native_dir_str)


@lru_cache(maxsize=1)
def _extension():
    _configure_windows_dll_paths()
    _extend_package_search_path()

    try:
        return import_module(_MODULE_NAME)
    except ImportError as exc:
        raise ImportError(
            "The witwin.channel native extension is unavailable. "
            "Build/install the package from source in the witwin2 environment "
            "to enable the Dr.Jit/CUDA bindings."
        ) from exc


def extension_available() -> bool:
    try:
        _extension()
    except ImportError:
        return False
    return True


def native_extension_available() -> bool:
    return extension_available()


def cuda_runtime_version() -> int:
    return _extension().cuda_runtime_version()


def run_cuda_noop() -> None:
    _extension().run_cuda_noop()


def sample_add_one(value):
    return _extension().sample_add_one(value)


__all__ = [
    "cuda_runtime_version",
    "extension_available",
    "native_extension_available",
    "run_cuda_noop",
    "sample_add_one",
]
