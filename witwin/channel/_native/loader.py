"""Shared loader for bundled native C++/CUDA extensions.

Each solver (``deterministic``, ``montecarlo``) and the shared channel core
native package ship a native extension with their own module name and build
glob. This module owns the loader implementation shared by the centralized
``witwin.channel._native`` extension specs.
"""

from __future__ import annotations

import importlib.machinery
import importlib.util
import os
import subprocess
import sys
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from site import getsitepackages, getusersitepackages
from types import ModuleType


_REPO_ROOT = Path(__file__).resolve().parents[3]


@dataclass(frozen=True)
class NativeExtensionSpec:
    """Static configuration for one bundled native extension."""

    module_name: str
    install_subpath: tuple[str, ...]
    build_glob: str
    binary_glob: str
    probe_env_var: str
    probe_module_name: str
    error_description: str


class NativeExtensionLoader:
    """Locate, load, and probe one bundled native C++/CUDA extension."""

    def __init__(self, spec: NativeExtensionSpec, package_path: list[str]) -> None:
        self._spec = spec
        self._package_path = package_path
        self._dll_handles: list[object] = []
        self._loaded: ModuleType | None = None
        self._probe_cached: bool | None = None

    def installed_native_dirs(self) -> list[Path]:
        search_roots: list[Path] = []
        try:
            search_roots.extend(Path(p).resolve() for p in getsitepackages())
        except Exception:
            pass
        try:
            search_roots.append(Path(getusersitepackages()).resolve())
        except Exception:
            pass

        native_dirs: list[Path] = []
        seen: set[Path] = set()
        build_root = _REPO_ROOT / "build"
        if build_root.is_dir():
            candidates: list[Path] = []
            for native_dir in build_root.glob(f"*/{self._spec.build_glob}"):
                resolved = native_dir.resolve()
                if (
                    resolved.is_dir()
                    and any(resolved.glob(self._spec.binary_glob))
                    and resolved not in seen
                ):
                    candidates.append(resolved)
                    seen.add(resolved)
            candidates.sort(
                key=lambda path: max(
                    c.stat().st_mtime for c in path.glob(self._spec.binary_glob)
                ),
                reverse=True,
            )
            native_dirs.extend(candidates)
        for root in search_roots:
            if root in seen or not root.exists():
                continue
            seen.add(root)
            native_dir = root.joinpath(*self._spec.install_subpath)
            if native_dir.is_dir():
                native_dirs.append(native_dir)
        return native_dirs

    def configure_windows_dll_paths(self) -> None:
        if os.name != "nt" or not hasattr(os, "add_dll_directory"):
            return

        added: set[str] = set()
        path_entries = os.environ.get("PATH", "").split(os.pathsep)
        normalized = {e.lower() for e in path_entries if e}

        def add_dir(target_dir: Path) -> None:
            if not target_dir.is_dir():
                return
            target = str(target_dir)
            if target in added:
                return
            self._dll_handles.append(os.add_dll_directory(target))
            if target.lower() not in normalized:
                path_entries.insert(0, target)
                normalized.add(target.lower())
            added.add(target)

        def add_package_dir(package: str, relative: str | None = None) -> None:
            spec = importlib.util.find_spec(package)
            if spec is None or not spec.submodule_search_locations:
                return
            package_dir = Path(next(iter(spec.submodule_search_locations))).resolve()
            add_dir(package_dir if relative is None else (package_dir / relative).resolve())

        add_package_dir("drjit")
        add_package_dir("torch", "lib")

        cuda_root = os.environ.get("CUDA_PATH") or os.environ.get("CUDA_HOME")
        if cuda_root:
            add_dir(Path(cuda_root).resolve() / "bin")

        conda_prefix = Path(sys.prefix).resolve()
        add_dir(conda_prefix / "Library" / "bin")
        add_dir(conda_prefix / "Library" / "usr" / "bin")
        for native_dir in self.installed_native_dirs():
            add_dir(native_dir)
        os.environ["PATH"] = os.pathsep.join(path_entries)

    def extend_package_search_path(self) -> None:
        for native_dir in reversed(self.installed_native_dirs()):
            native_dir_str = str(native_dir)
            if native_dir_str not in self._package_path:
                self._package_path.insert(0, native_dir_str)

    def load(self) -> ModuleType:
        if self._loaded is not None:
            return self._loaded
        self.configure_windows_dll_paths()
        self.extend_package_search_path()
        existing = sys.modules.get(self._spec.module_name)
        if existing is not None:
            self._loaded = existing
            return self._loaded
        try:
            spec = importlib.machinery.PathFinder.find_spec(
                self._spec.module_name,
                self._package_path,
            )
            if spec is None or spec.loader is None:
                self._loaded = import_module(self._spec.module_name)
            else:
                module = importlib.util.module_from_spec(spec)
                sys.modules[self._spec.module_name] = module
                spec.loader.exec_module(module)
                self._loaded = module
        except ImportError as exc:
            raise ImportError(
                f"The {self._spec.error_description} is unavailable. "
                "Build/install the package from source in the witwin2 environment "
                "to enable the Dr.Jit/CUDA bindings."
            ) from exc
        return self._loaded

    def probe_extension_in_process(self) -> bool:
        try:
            self.load()
        except ImportError:
            return False
        return True

    def _probe_extension_available(self) -> bool:
        if self._probe_cached is not None:
            return self._probe_cached
        env = dict(os.environ)
        env[self._spec.probe_env_var] = "1"
        code = (
            f"from {self._spec.probe_module_name} import NativeExtension; "
            "import sys; "
            "sys.exit(0 if NativeExtension.probe_extension_in_process() else 1)"
        )
        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        try:
            result = subprocess.run(
                [sys.executable, "-c", code],
                env=env,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
                creationflags=creationflags,
            )
        except (OSError, subprocess.SubprocessError):
            self._probe_cached = False
            return False
        self._probe_cached = result.returncode == 0
        return self._probe_cached

    def extension_available(self) -> bool:
        if os.name == "nt" and os.environ.get(self._spec.probe_env_var) != "1":
            return self._probe_extension_available()
        try:
            self.load()
        except ImportError:
            return False
        return True

    def native_extension_available(self) -> bool:
        return self.extension_available()

    def has_functions(self, names: tuple[str, ...]) -> bool:
        if not self.extension_available():
            return False
        ext = self.load()
        return all(hasattr(ext, n) for n in names)

    def require_functions(self, names: tuple[str, ...], *, context: str) -> ModuleType:
        ext = self.load()
        missing = [n for n in names if not hasattr(ext, n)]
        if missing:
            raise RuntimeError(
                f"{context} requires {', '.join(missing)}. "
                f"Rebuild the {self._spec.error_description}."
            )
        return ext


__all__ = ["NativeExtensionLoader", "NativeExtensionSpec"]
