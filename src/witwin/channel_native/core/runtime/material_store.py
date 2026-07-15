"""Compatibility import for the canonical material store owner."""

import torch  # noqa: F401 - MaterialStore annotation compatibility

from witwin.channel_native.scene.stores._validation import require_tensor  # noqa: F401
from witwin.channel_native.scene.stores.materials import MaterialStore

__all__ = ["MaterialStore"]
