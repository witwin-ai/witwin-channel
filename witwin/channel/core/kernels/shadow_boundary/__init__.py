"""Shared native shadow-boundary candidate-smoothing kernel.

The compiled symbol ``shadow_boundary_candidate_accumulate`` ships inside
``witwin.channel._native._channel_utils_native`` so both
``witwin.channel.deterministic`` and ``witwin.channel.montecarlo`` consume the binary
without either solver hosting the artifact.
"""

from .native_impl import ShadowBoundaryKernel

__all__ = ["ShadowBoundaryKernel"]
