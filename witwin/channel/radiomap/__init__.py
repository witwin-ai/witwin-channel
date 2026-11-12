# Copyright Xingyu Chen.
# Radiomap solve that dispatches on the estimator configuration type.

"""Radiomap solve that dispatches on the estimator configuration type."""

from __future__ import annotations

from typing import TYPE_CHECKING, overload

from witwin.channel.radiomap import _coherent, _stochastic
from witwin.channel.radiomap._coherent import (
    Coherent,
    CoherentRadioMap,
    CoherentRadioMapMetadata,
    PathTable,
)
from witwin.channel.radiomap._stochastic import RadioMap, RadioMapMetadata, Stochastic

if TYPE_CHECKING:
    import torch

    from witwin.core import Scene, SceneSnapshot


@overload
def solve(
    scene: Scene | SceneSnapshot, config: Coherent, *, reference_frequency_hz: float | torch.Tensor,
) -> CoherentRadioMap: ...


@overload
def solve(
    scene: Scene | SceneSnapshot, config: Stochastic, *,
    reference_frequency_hz: float | torch.Tensor,
) -> RadioMap: ...


def solve(
    scene: Scene | SceneSnapshot, config: Coherent | Stochastic, *,
    reference_frequency_hz: float | torch.Tensor,
) -> CoherentRadioMap | RadioMap:
    """Solve a radiomap with the estimator ``config`` selects: Coherent or Stochastic."""

    if isinstance(config, Coherent):
        return _coherent.solve(scene, config, reference_frequency_hz=reference_frequency_hz)
    if isinstance(config, Stochastic):
        return _stochastic.solve(scene, config, reference_frequency_hz=reference_frequency_hz)
    raise TypeError(
        "witwin.channel.radiomap.solve() requires config to be "
        "witwin.channel.radiomap.Coherent or witwin.channel.radiomap.Stochastic, got "
        f"{type(config).__module__}.{type(config).__qualname__}"
    )


__all__ = [
    "Coherent",
    "CoherentRadioMap",
    "CoherentRadioMapMetadata",
    "PathTable",
    "RadioMap",
    "RadioMapMetadata",
    "Stochastic",
    "solve",
]
