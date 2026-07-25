"""Composed source-basis to sink-basis complex 2 x 2 transport.

The native field transport is linear in the transmit polarization and linear in
the receive polarization:

* ``project_to_wedge_plane(v, e) = v - e*(v.e)`` is linear in ``v``;
* a Fresnel bounce scales the s and p components by coefficients that depend on
  the incidence frame and the material, never on the field itself;
* the trailing free-space factor is a complex scalar;
* ``project_receiver(E, d, p) = E . project_to_wedge_plane(p, d)``.

So the map from a source transverse component to a sink transverse component is
bilinear, and the four entries of the operator are recovered exactly by
exciting the SAME native transport twice, once per source basis vector, and
projecting each response onto both sink basis vectors. Nothing here computes
physics: this module chooses excitations, dispatches the native owners, and
stacks their published results.

Both transverse bases are produced by the native ``consumer_los_jones``
endpoint-basis owner rather than by a Torch normalize or cross product. A
reflection row has two different directions - the launch direction toward its
first interaction and the arrival direction from its last interaction - and the
basis for each is obtained by handing that leg's two endpoints to the native
owner, which recomputes the direction with the same ``safe_normalize`` the
field kernel uses. The bases are structurally primal-only: the composition
feeds them to the native companions as ``tx_polarization`` and
``rx_polarization``, both of which reject gradients by contract.
"""

from __future__ import annotations

from collections.abc import Callable

import torch

from ._native import consumer_los_jones


_FieldOp = Callable[[torch.Tensor], dict[str, torch.Tensor]]


def _primal(value: torch.Tensor) -> torch.Tensor:
    """Detached primal view for an input a native op consumes as a constant."""

    primal = torch.autograd.forward_ad.unpack_dual(value).primal
    return primal.detach().contiguous()


def transverse_basis(
    reference_basis: torch.Tensor,
    leg_origin: torch.Tensor,
    leg_target: torch.Tensor,
    *,
    frequency_hz: float,
) -> torch.Tensor:
    """Row-aligned orthonormal basis transverse to ``leg_target - leg_origin``.

    ``consumer_los_jones`` indexes its endpoint tables through ``pair_index``,
    so handing it per-row tables and the diagonal pair index makes it evaluate
    exactly one leg per row. The same reference basis is supplied for both
    endpoints, which makes its two published bases identical, and the source
    one is returned. This reuses the shipped native endpoint-basis owner
    instead of restating its projection and orthonormalization in Torch.
    """

    rows = int(leg_origin.shape[0])
    pair_index = torch.arange(
        rows, device=leg_origin.device, dtype=torch.int64
    ) * (rows + 1)
    reference = _primal(reference_basis)
    return consumer_los_jones(
        pair_index=pair_index,
        source_positions=_primal(leg_origin),
        sink_positions=_primal(leg_target),
        source_reference_basis=reference,
        sink_reference_basis=reference,
        frequency_hz=frequency_hz,
    ).source_basis


def _project(
    field_vector: torch.Tensor,
    direction: torch.Tensor,
    sink_vector: torch.Tensor,
) -> torch.Tensor:
    from witwin.channel.propagation.fields.kernels import (
        autograd_projection as field_projection,
    )

    return field_projection.field_project_complex3_ad(
        field_vector, direction, sink_vector
    )["coefficient"]


def compose_jones(
    excite: _FieldOp,
    *,
    source_basis: torch.Tensor,
    sink_reference_basis: torch.Tensor,
    arrival_origin: torch.Tensor,
    arrival_target: torch.Tensor,
    frequency_hz: float,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    """Build the ``(N, 2, 2)`` operator from two excitations of ``excite``.

    ``matrix[k, i, j]`` is the response of sink basis vector ``i`` to source
    basis vector ``j``, which is the index convention the native LoS Jones
    owner publishes. Returns the operator, the sink basis, and the first
    column's full field result so the caller can publish its geometry without
    re-running the transport.
    """

    columns = tuple(
        excite(source_basis[:, index].contiguous()) for index in (0, 1)
    )
    sink_basis = transverse_basis(
        sink_reference_basis,
        arrival_origin,
        arrival_target,
        frequency_hz=frequency_hz,
    )
    direction = columns[0]["direction"]
    matrix = torch.stack(
        [
            torch.stack(
                [
                    _project(
                        column["field_vector"],
                        direction,
                        sink_basis[:, index].contiguous(),
                    )
                    for column in columns
                ],
                dim=-1,
            )
            for index in (0, 1)
        ],
        dim=-2,
    )
    return matrix, sink_basis, columns[0]


__all__ = ["compose_jones", "transverse_basis"]
