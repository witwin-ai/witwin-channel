"""GPU-accelerated differentiable wireless propagation simulation.

The package root exports only what Channel owns: build and runtime reporting,
solver capabilities, and the native field-state contracts. The logical world
model  -  ``Scene``, ``SceneSnapshot``, ``Structure``, ``PhysicalMaterial``,
antenna state, and the identity/version contracts  -  is owned by
:mod:`witwin.core` and must be imported from there. Channel does not re-export
it, so each of those types has exactly one import path.

Solver entry points live in :mod:`witwin.channel.path`,
:mod:`witwin.channel.deterministic`, :mod:`witwin.channel.montecarlo.basic`,
and :mod:`witwin.channel.montecarlo.bdpt`. The solver-neutral propagation
contract lives in :mod:`witwin.channel.propagation.consumer`.
"""

from .capabilities import capabilities
from .deployment import build_info, pipeline_cache_key, runtime_diagnostics
from .abi import Complex3State, JonesState

__all__ = [
    "Complex3State",
    "JonesState",
    "build_info",
    "capabilities",
    "pipeline_cache_key",
    "runtime_diagnostics",
]
