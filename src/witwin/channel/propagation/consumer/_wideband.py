"""The wideband frequency-offset surface of the consumer contract (ADR-042).

A fixed-topology request may declare a grid of propagation-frequency offsets and
receive the same frozen rows evaluated at each absolute frequency. Everything
that grid needs before it reaches a scene lives here: the float32 launch
resolution that bounds how fine it may be, its structural validation, and the
paired-presence law its payload obeys.

It is a separate module from ``contracts`` for one reason worth naming: nothing
here depends on any contract type, so the dependency runs one way and the
vocabulary module stays the single place a reader looks up a field.
"""

from __future__ import annotations

import math
import struct

import torch


# The launch grid the native field bridges actually resolve. Every bridge takes
# a double ``frequency_hz`` and ``static_cast<float>``s it at the launch, so two
# absolute frequencies inside one float32 ULP are the SAME launch and return
# bit-identical coefficients. Published as a law plus a function rather than as
# a constant, because the resolution is a function of the reference frequency
# (8192 Hz at 77 GHz, 64 Hz at 1 GHz).
NATIVE_FREQUENCY_RESOLUTION_LAW = (
    "resolution_hz = ulp_float32(reference_frequency_hz)"
)


def native_frequency_resolution_hz(reference_frequency_hz: float) -> float:
    """Smallest absolute frequency step the native launch grid resolves.

    The value is one float32 unit in the last place at
    ``reference_frequency_hz``. A caller computes the same number the
    wideband refusal uses instead of rederiving it, which is the ADR-036 rule
    that a declared limit is discoverable rather than learned from a rejection.
    """

    value = abs(float(reference_frequency_hz))
    if not math.isfinite(value) or value == 0.0:
        raise ValueError(
            "reference_frequency_hz must be finite and non-zero to have a "
            "native frequency resolution"
        )
    # Round to the float32 the launch actually receives before reading the
    # binade, so a value that rounds up across a power of two reports the
    # resolution of the launch rather than of the request.
    launched = struct.unpack("<f", struct.pack("<f", value))[0]
    _, exponent = math.frexp(launched)
    # float32 carries a 24-bit significand, so one ULP in that binade is
    # 2**(exponent - 24).
    return math.ldexp(1.0, exponent - 24)


def require_frequency_offsets(value: object) -> tuple[float, ...] | None:
    """Structural validation of a wideband offset grid, before any native work.

    The grid is a HOST DECLARATION in the same class as ``slot_count``: it
    names which absolute frequencies the same frozen rows are evaluated at. It
    is deliberately not a tensor and deliberately not differentiable.
    """

    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        raise TypeError(
            "frequency_offsets_hz must be a tuple of host floats, not a "
            "torch.Tensor: the offset grid is a host declaration of which "
            "absolute frequencies to evaluate, not a differentiable input. A "
            "tangent with respect to one grid point is identical to the "
            "reference_frequency_hz tangent evaluated at that point, so seed "
            "reference_frequency_hz instead"
        )
    if not isinstance(value, tuple):
        raise TypeError("frequency_offsets_hz must be a tuple of floats or None")
    if not value:
        raise ValueError(
            "frequency_offsets_hz must be a non-empty tuple; pass None for a "
            "single-frequency request"
        )
    offsets = []
    for entry in value:
        if isinstance(entry, bool) or not isinstance(entry, (int, float)):
            raise TypeError("frequency_offsets_hz entries must be floats")
        offset = float(entry)
        if not math.isfinite(offset):
            raise ValueError(
                f"frequency_offsets_hz entries must be finite, got {entry!r}"
            )
        offsets.append(offset)
    if len(set(offsets)) != len(offsets):
        raise ValueError(
            "frequency_offsets_hz must not repeat an offset; duplicate entries "
            "produce bit-identical columns and hide a caller bug"
        )
    return tuple(offsets)


def require_wideband_payload(
    name: str,
    payload: object,
    offsets: object,
    reference: torch.Tensor,
) -> None:
    """Enforce the ADR-042 paired-presence and shape law on one transport.

    The payload and the grid it was evaluated on are both present or both
    absent. An unpaired payload is a column set nobody can label, and an
    unpaired grid is a promise nobody kept; either one is a contract error
    rather than something a reader should have to guess about.

    ``reference`` is the single-frequency tensor the payload is the band of, so
    its shape defines the payload's: the frequency axis is inserted after the
    row axis and every trailing axis is preserved. Taking the shape from the
    reference rather than restating it keeps one description of what a column
    is.
    """

    if payload is None and offsets is None:
        return
    if payload is None or offsets is None:
        raise ValueError(
            f"{name} and frequency_offsets_hz are paired: publish both or neither"
        )
    if not isinstance(offsets, tuple) or not offsets:
        raise TypeError("frequency_offsets_hz must be a non-empty tuple of floats")
    if not isinstance(payload, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if payload.dtype != reference.dtype:
        raise TypeError(f"{name} must use {reference.dtype}, got {payload.dtype}")
    expected = (int(reference.shape[0]), len(offsets), *reference.shape[1:])
    if tuple(payload.shape) != expected:
        raise ValueError(
            f"{name} must have shape {expected}, got {tuple(payload.shape)}"
        )
    if payload.device != reference.device:
        raise ValueError(f"{name} must be on {reference.device}, got {payload.device}")


# The layout a wideband payload adds on top of the row and slot layouts, which
# it does not redefine. Frequency is orthogonal to both: the same rows are
# evaluated at F frequencies, so the axis is appended and nothing is tiled,
# re-paired, or re-segmented.
WIDEBAND_OFFSET_LAYOUT = (
    "frequency_minor:"
    "payload[row, j] = response(row) at"
    " reference_frequency_hz+frequency_offsets_hz[j];"
    "row axis and pair segmentation unchanged;"
    "row_valid stays [K] and broadcasts over j;"
    "geometry published once from the reference evaluation;"
    "slot composition gives [slot_count*frozen_row_count, F]"
)

# Why an offset grid can be refused as unresolvable. Channel publishes the
# resolution and the resulting phase bound; it does not evaluate the bound,
# because that needs max(delay_s), which is a device reduction plus a host read
# the ADR-032 budget does not have. The caller owns that check.
WIDEBAND_FREQUENCY_QUANTIZATION_LAW = (
    "launch_grid=float32;"
    " resolution_hz=ulp_float32(reference_frequency_hz);"
    " abs_phase_error_rad <= pi*resolution_hz*delay_s"
)


__all__ = [
    "NATIVE_FREQUENCY_RESOLUTION_LAW",
    "WIDEBAND_FREQUENCY_QUANTIZATION_LAW",
    "WIDEBAND_OFFSET_LAYOUT",
    "native_frequency_resolution_hz",
    "require_frequency_offsets",
    "require_wideband_payload",
]
