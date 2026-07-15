"""Compatibility import for the enumerated propagation scattering stage.

The implementation is owned by :mod:`witwin.channel_native.propagation.enumerated`.
This module remains during the architecture migration so existing private imports
fail neither silently nor abruptly.
"""

from witwin.channel_native.propagation.enumerated import append_scattering_paths

__all__ = ["append_scattering_paths"]
