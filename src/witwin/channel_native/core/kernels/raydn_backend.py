from __future__ import annotations

def native_extension() -> None:
    """RayD is source-linked into ``_channel_native``; no second module exists."""

    return None


def require_native_extension() -> None:
    return None


def capability_info() -> dict[str, bool | str]:
    return {
        "uses_raydn_native": True,
        "optix_available": True,
        "raydn_extension_loaded": False,
        "rayd_integration": "source-linked",
    }
