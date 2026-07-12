from __future__ import annotations

from collections.abc import Mapping
import importlib

from . import raydn_backend


_DEFAULT_BUILD_INFO = {
    "backend": "channel-native",
    "uses_dr_jit": False,
    "uses_raydn_native": False,
    "rayd_integration": "unavailable",
    "uses_path_native": False,
    "cuda_available": False,
    "optix_available": False,
}


def native_extension() -> object:
    package_module = "witwin.channel_native._channel_native"
    if importlib.util.find_spec(package_module) is not None:
        return importlib.import_module("._channel_native", package="witwin.channel_native")
    return importlib.import_module("_channel_native")


def build_info() -> dict[str, bool | str]:
    """Return native capability metadata without importing RayDN Python APIs."""

    native_info = native_extension().build_info()
    if not isinstance(native_info, Mapping):
        raise TypeError("_channel_native.build_info() must return a mapping")

    info = dict(_DEFAULT_BUILD_INFO)
    info.update(native_info)
    info.update(raydn_backend.capability_info())
    return info
