"""Source-amplitude dispatch for the consumer's excited complex3 response.

The field transport kernels carry ``sqrt(tx_power)`` into ``path_field`` and
``path_gain`` but leave their complex3 vector at unit excitation, so there is
no excited vector on that launch. ADR-039 adds the native owner of exactly
that quantity; this module only chooses between its primal and differentiable
entry points. No amplitude is computed here.
"""

from __future__ import annotations

import torch


def excited_field(
    field_vector: torch.Tensor, tx_power: torch.Tensor, *, ad_mode: str
) -> torch.Tensor:
    """Return the source-excited complex3 field for a unit-excitation one."""

    from witwin.channel.propagation.fields.kernels import (
        source_amplitude as field_amplitude,
    )

    if ad_mode == "none":
        return field_amplitude.field_source_amplitude_scale(
            field_vector, tx_power
        )["path_field_vector"]
    return field_amplitude.field_source_amplitude_scale_ad(field_vector, tx_power)


__all__ = ["excited_field"]
