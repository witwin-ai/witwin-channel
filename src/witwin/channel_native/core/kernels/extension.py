from __future__ import annotations

from collections.abc import Mapping
import importlib
import pathlib
import sys

from . import raydn_backend


_DEFAULT_BUILD_INFO = {
    "backend": "channel-native",
    "uses_dr_jit": False,
    "uses_raydn_native": False,
    "uses_path_native": False,
    "cuda_available": False,
    "optix_available": False,
}


def native_extension() -> object | None:
    try:
        return importlib.import_module("_channel_native")
    except ModuleNotFoundError:
        pass

    repo_root = pathlib.Path(__file__).resolve().parents[5]
    artifact_dirs = (
        repo_root / "artifacts" / "cmake-channel-native-raydn-witwin2-release",
        repo_root / "artifacts" / "cmake-witwin2-explicit-release",
    )
    for artifact_dir in artifact_dirs:
        if not artifact_dir.is_dir():
            continue
        artifact_path = str(artifact_dir)
        if artifact_path not in sys.path:
            sys.path.insert(0, artifact_path)
        try:
            return importlib.import_module("_channel_native")
        except ModuleNotFoundError:
            continue
    return None


def build_info() -> dict[str, bool | str]:
    """Return native capability metadata without importing RayDN Python APIs."""

    native = native_extension()
    if native is None:
        return dict(_DEFAULT_BUILD_INFO)

    native_info = native.build_info()
    if not isinstance(native_info, Mapping):
        raise TypeError("_channel_native.build_info() must return a mapping")

    info = dict(_DEFAULT_BUILD_INFO)
    info.update(native_info)
    info.update(raydn_backend.capability_info())
    return info
