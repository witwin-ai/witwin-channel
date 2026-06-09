from __future__ import annotations

from collections.abc import Mapping


_DEFAULT_BUILD_INFO = {
    "backend": "channel-native",
    "uses_dr_jit": False,
    "uses_raydn_native": False,
    "cuda_available": False,
    "optix_available": False,
}


def build_info() -> dict[str, bool | str]:
    """Return native capability metadata without importing RayDN Python APIs."""

    try:
        import _channel_native
    except ModuleNotFoundError:
        return dict(_DEFAULT_BUILD_INFO)

    native_info = _channel_native.build_info()
    if not isinstance(native_info, Mapping):
        raise TypeError("_channel_native.build_info() must return a mapping")

    info = dict(_DEFAULT_BUILD_INFO)
    info.update(native_info)
    return info
