"""Temporary compatibility re-export for topology RayD handle consumers.

Remove this bridge in Phase 10 after dummy RayD module handles leave the Python ABI.
"""

from __future__ import annotations

from witwin.channel_native.scene.native_handles import (
    _raydn_module_handle,
    _raydn_scene_handle_id,
)


__all__ = ["_raydn_module_handle", "_raydn_scene_handle_id"]
