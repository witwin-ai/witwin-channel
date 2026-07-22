"""Compatibility import for the canonical geometry store owner."""

import torch  # noqa: F401 - GeometryStore annotation compatibility

from witwin.channel.scene.stores._validation import require_tensor  # noqa: F401
from witwin.channel.scene.stores.geometry import GeometryStore

__all__ = ["GeometryStore"]
