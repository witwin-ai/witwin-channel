"""Which conditional outputs of one field apply carry a derivative.

Two wrapper-level decisions reach the shared field companions as ONE explicit
trailing input and are unpacked onto the autograd context: the ADR-038 geometry
decision behind ``path_length_m`` and ``delay_s``, and the ADR-043 direction
decision behind ``field_direction``. Both are taken where forward duals are
still visible, because ``Function.apply`` unpacks them before ``setup_context``
runs. This module is where they are decided and read back, so the field
Functions agree on what a dead output looks like instead of each spelling it
out.
"""

from __future__ import annotations

import torch

from witwin.channel.runtime.autograd_contracts import _ad_geometry_live

from .functional import _FIELD_AD_DIRECTION_TANGENT_FIELDS


def ad_liveness(direction_live: bool, *geometry: object) -> tuple[bool, bool]:
    """The one liveness record an apply carries, decided by the wrapper.

    A direction derivative is a geometry derivative, so the direction half can
    only be live where the geometry half is. The other half of the direction
    decision is the caller's host-known component set, which is why it arrives
    as a flag rather than being inferred from a tensor.
    """

    geometry_live = _ad_geometry_live(*geometry)
    return (geometry_live, geometry_live and bool(direction_live))


def mark_dead_outputs(ctx, output) -> None:
    """Declare the outputs of one apply that carry no derivative.

    A dead output is marked exactly as it was before the direction seam
    existed, so a caller that does not ask for a live direction sees the same
    object graph it saw at contract version 5.
    """

    dead = []
    if not ctx.geometry_live:
        dead.extend((output[4], output[5]))
    if not ctx.direction_live:
        dead.append(output[6])
    if dead:
        ctx.mark_non_differentiable(*dead)


def direction_cotangent(ctx, grad_direction):
    """The incoming direction cotangent, or ``None`` if it was declared dead.

    Torch does not deliver a cotangent for an output marked non-differentiable,
    so this is belt and braces; it is also the one place that says out loud that
    a dead output's seed never reaches a native companion.
    """

    return grad_direction if ctx.direction_live else None


def direction_tangents(ctx, out: dict[str, torch.Tensor]) -> tuple:
    """Publish one apply's output tangents under the two liveness decisions.

    The native companion always computes the direction tangent - it is the dual
    the transverse projection already needed - so this only decides whether it
    is published. A dead output receives ``None`` rather than a zero tensor,
    because a zero tangent on a declared-dead output is exactly the silent
    answer ADR-043 removes.
    """

    tangents = tuple(out[name] for name in _FIELD_AD_DIRECTION_TANGENT_FIELDS)
    if not ctx.geometry_live:
        return (*tangents[:4], None, None, None)
    if not ctx.direction_live:
        return (*tangents[:6], None)
    return tangents


__all__ = [
    "ad_liveness",
    "direction_cotangent",
    "direction_tangents",
    "mark_dead_outputs",
]
