"""Lazy compatibility bridge to the scene material catalog."""

from __future__ import annotations


def itu_material_parameters(name: str, frequency_hz: float) -> tuple[float, float]:
    from witwin.channel_native.scene.loader import (
        itu_material_parameters as _itu_material_parameters,
    )

    return _itu_material_parameters(name, frequency_hz)
