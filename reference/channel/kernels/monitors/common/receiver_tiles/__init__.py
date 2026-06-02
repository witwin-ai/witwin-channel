"""Shared receiver-tile infrastructure for dense field monitor paths."""

from .native_impl import (
    DEFAULT_RECEIVER_TILE_SHAPE,
    ReceiverTileDescriptor,
    build_receiver_tiles,
    compact_tile_tasks,
    deduplicate_tile_tasks,
    resolve_receiver_tiles,
)

__all__ = [
    "DEFAULT_RECEIVER_TILE_SHAPE",
    "ReceiverTileDescriptor",
    "build_receiver_tiles",
    "compact_tile_tasks",
    "deduplicate_tile_tasks",
    "resolve_receiver_tiles",
]
