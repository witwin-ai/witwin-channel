"""Compatibility import for the canonical assignment store owner."""

from dataclasses import field  # noqa: F401 - AssignmentStore default compatibility

import torch  # noqa: F401 - AssignmentStore annotation compatibility

from witwin.channel.materials.models import (  # noqa: F401
    PhaseScreen,
)
from witwin.channel.scene.stores._validation import require_tensor  # noqa: F401
from witwin.channel.scene.stores.assignments import AssignmentStore

__all__ = ["AssignmentStore"]
