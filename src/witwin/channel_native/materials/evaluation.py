"""Material evaluation contracts shared by solver frontends."""

from __future__ import annotations

from typing import TYPE_CHECKING

from witwin.channel_native.runtime.autograd_contracts import (
    _frequency_participates_in_ad,
)

if TYPE_CHECKING:
    from witwin.channel_native.core.scene import Scene


def _require_frequency_ad_constant_materials(
    scene: Scene, compiled: object, *, ad_mode: str
) -> None:
    """Explicit-failure contract for frequency AD over dispersive materials.

    ``Scene.compile()`` freezes material records at the primal frequency, so
    a frequency gradient through a scene with frequency-dependent material
    laws would silently miss d(material)/d(frequency) (plan 07 section 7:
    never return misleading gradients). Fail before any launch instead.
    """

    dependent = tuple(compiled.materials.frequency_dependent)
    if not dependent or not _frequency_participates_in_ad(scene.frequency):
        return
    raise NotImplementedError(
        f"ad_mode='{ad_mode}' cannot differentiate with respect to frequency "
        "in this scene: material records are frozen at the primal frequency "
        "at compile time, so the gradient would silently miss "
        "d(material)/d(frequency) for the frequency-dependent materials "
        f"{list(dependent)}. Use a constant-material scene for frequency AD, "
        "or drop the frequency requires_grad/tangent for materials-only AD."
    )
