from __future__ import annotations

import importlib
from types import ModuleType


def native_extension() -> ModuleType:
    """Load the RayDN native extension without importing the Python raydn package."""

    import sys

    existing = sys.modules.get("_raydn")
    if isinstance(existing, ModuleType):
        return existing
    return importlib.import_module("_raydn")


def require_native_extension() -> ModuleType:
    return native_extension()


def capability_info() -> dict[str, bool | str]:
    native_extension()
    return {
        "uses_raydn_native": True,
        "optix_available": True,
        "raydn_extension_loaded": True,
    }
