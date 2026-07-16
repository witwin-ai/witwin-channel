"""Compatibility import for the canonical RayDN scene lifecycle owner."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from witwin.channel_native.runtime.native_buffers import mc_pack_vec3
from witwin.channel_native.scene.kernels.rayd_scene import (
    RayDNEdgeRecords,
    RayDNScene,
    _empty_tensor,  # noqa: F401 - compatibility re-export
    _mesh_flags,  # noqa: F401 - compatibility re-export
    build_scene_from_structures,
    raydn_scene_create,
    raydn_scene_edge_records,
)

__all__ = [
    "RayDNEdgeRecords",
    "RayDNScene",
    "build_scene_from_structures",
    "dataclass",
    "field",
    "mc_pack_vec3",
    "raydn_scene_create",
    "raydn_scene_edge_records",
    "torch",
]
