# Copyright Xingyu Chen.
# Implements bdpt ad oracles.

"""Implements bdpt ad oracles."""

from __future__ import annotations

import torch

from tests.ad._reference_fields import (
    SPEED_OF_LIGHT as _C0,
    _project,
    _transverse_project,
)

_SMALL_EPS = 1.0e-6
_EPS = 1.0e-10
_POL_TE = 0
_POL_TM = 1

# BDPT accumulable component ids (bdpt_connect.cu kComponent*).
# Transmission and scattering are 5 and 6 in the native encoding (3 and 4 are
# unused), so the oracle must use the same ids the accumulate kernels route on.
COMPONENT_LOS = 0
COMPONENT_REFLECTION = 1
COMPONENT_DIFFRACTION = 2
COMPONENT_TRANSMISSION = 5
COMPONENT_SCATTERING = 6
_ACCUMULABLE = (
    COMPONENT_LOS,
    COMPONENT_REFLECTION,
    COMPONENT_DIFFRACTION,
    COMPONENT_TRANSMISSION,
    COMPONENT_SCATTERING,
)
_COMPONENT_NAMES = {
    COMPONENT_LOS: "los",
    COMPONENT_REFLECTION: "reflection",
    COMPONENT_DIFFRACTION: "diffraction",
    COMPONENT_TRANSMISSION: "transmission",
    COMPONENT_SCATTERING: "scattering",
}


def _ez_like(reference: torch.Tensor) -> torch.Tensor:
    ez = torch.zeros_like(reference)
    ez[..., 2] = 1.0
    return ez


# ---------------------------------------------------------------------------
# Endpoint-connection contribution.
# ---------------------------------------------------------------------------


def endpoint_connection_contribution_reference(
    light_field: torch.Tensor, source_power: torch.Tensor, light_origin: torch.Tensor,
    sensor_origin: torch.Tensor, receiver_polarization: torch.Tensor,
    light_path_length: torch.Tensor, frequency: torch.Tensor, samples_per_tx: int,
) -> torch.Tensor:
    """Endpoint-connection derivatives per-row endpoint contribution.

 ``contribution = P_src * |proj(F)|^2 * (1/(2 k L))^2 / N``, ``L`` the unfolded
 length ``|sensor-light| + light_path_length`` (frozen), ``proj`` the frozen
 transverse receiver projection. Live leaves: ``light_field`` (Complex3),
 ``source_power`` (tx_power), ``frequency``. Frozen: both origins, ``L``,
 ``receiver_polarization``, ``samples_per_tx``.
 """

    offset = sensor_origin - light_origin
    distance = torch.linalg.vector_norm(offset, dim=-1).clamp_min(1.0e-6)
    direction = offset / distance.unsqueeze(-1)
    total_distance = (distance + light_path_length.clamp_min(0.0)).clamp_min(1.0e-6)
    wave_number = (2.0 * torch.pi * frequency / _C0).clamp_min(1.0e-12)
    amplitude = 1.0 / (2.0 * wave_number * total_distance)
    rx_axis = _transverse_project(direction, receiver_polarization)
    coefficient = _project(light_field, rx_axis)
    coefficient_power = coefficient.real.square() + coefficient.imag.square()
    return (
        source_power
        * coefficient_power
        * amplitude
        * amplitude
        / float(samples_per_tx)
    )


# ---------------------------------------------------------------------------
# Connection-sample accumulation for power and coherent domains.
# ---------------------------------------------------------------------------


def _bin_index(
    tx_id: torch.Tensor, rx_id: torch.Tensor, component_id: torch.Tensor, valid: torch.Tensor,
    tx_count: int, rx_count: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the flat (tx*rx) bin index and the accumulable-row mask (frozen)."""

    accumulable = torch.zeros_like(valid)
    for cid in _ACCUMULABLE:
        accumulable = accumulable | (component_id == cid)
    row_ok = (
        valid
        & accumulable
        & (tx_id >= 0)
        & (tx_id < tx_count)
        & (rx_id >= 0)
        & (rx_id < rx_count)
    )
    flat = (tx_id.clamp_min(0) * rx_count + rx_id.clamp_min(0)).to(torch.int64)
    return flat, row_ok


def accumulate_power_reference(
    contribution: torch.Tensor, mis_weight: torch.Tensor, tx_id: torch.Tensor, rx_id: torch.Tensor,
    component_id: torch.Tensor, valid: torch.Tensor, tx_count: int, rx_count: int,
) -> dict[str, torch.Tensor]:
    """Connection-accumulation derivatives power-domain accumulate. Live leaf: ``contribution``.

 ``M[b] = sum_r contribution_r * mis_r`` binned by (tx, rx) per component, plus a
 ``path_gain`` over every accumulable component. ``mis_weight`` and the id/valid
 structure are frozen.
 """

    flat, row_ok = _bin_index(
        tx_id, rx_id, component_id, valid, tx_count, rx_count
    )
    weighted = contribution * mis_weight.to(contribution.dtype) * row_ok.to(
        contribution.dtype
    )
    out: dict[str, torch.Tensor] = {}
    path_gain = torch.zeros(
        tx_count * rx_count, dtype=contribution.dtype, device=contribution.device
    )
    path_gain = path_gain.index_add(0, flat, weighted)
    for cid, name in _COMPONENT_NAMES.items():
        mask = (component_id == cid).to(contribution.dtype)
        comp = torch.zeros_like(path_gain)
        comp = comp.index_add(0, flat, weighted * mask)
        out[name] = comp.reshape(tx_count, rx_count)
    out["path_gain"] = path_gain.reshape(tx_count, rx_count)
    return out


def accumulate_coherent_reference(
    coeff_real: torch.Tensor, coeff_imag: torch.Tensor, tx_id: torch.Tensor, rx_id: torch.Tensor,
    component_id: torch.Tensor, valid: torch.Tensor, tx_count: int, rx_count: int,
) -> dict[str, torch.Tensor]:
    """Connection-accumulation derivatives coherent-domain accumulate. Live leaves: coeff_real/imag.

 ``S_b = sum_r c_r`` per (tx, rx, component); ``P_comp = |S_b|^2``;
 ``path_gain = sum_comp P_comp`` (components combine incoherently). Also returns
 the per-component complex bin sums ``S_b`` (``*_sum``), which the native coherent
 backward reads as the forward-retained buffers (retained forward bin sums).
 """

    flat, row_ok = _bin_index(
        tx_id, rx_id, component_id, valid, tx_count, rx_count
    )
    keep = row_ok.to(coeff_real.dtype)
    out: dict[str, torch.Tensor] = {}
    path_gain = torch.zeros(
        tx_count * rx_count, dtype=coeff_real.dtype, device=coeff_real.device
    )
    for cid, name in _COMPONENT_NAMES.items():
        mask = (component_id == cid).to(coeff_real.dtype) * keep
        s_re = torch.zeros_like(path_gain).index_add(0, flat, coeff_real * mask)
        s_im = torch.zeros_like(path_gain).index_add(0, flat, coeff_imag * mask)
        power = s_re * s_re + s_im * s_im
        out[name] = power.reshape(tx_count, rx_count)
        out[f"{name}_sum"] = torch.complex(s_re, s_im).reshape(tx_count, rx_count)
        path_gain = path_gain + power
    out["path_gain"] = path_gain.reshape(tx_count, rx_count)
    return out


# ---------------------------------------------------------------------------
# Linear finalization for point components and component maps.
# ---------------------------------------------------------------------------


def finalize_components_reference(
    los: torch.Tensor, reflection: torch.Tensor, diffraction: torch.Tensor,
    transmission: torch.Tensor, scattering: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Result-finalization derivatives finalize (shape-agnostic linear map).

 ``path_gain = los + reflection + diffraction + transmission + scattering``
 (elementwise) and each ``*_power`` is the scalar sum of that component over the
 whole map. Live leaves: the five component tensors (2-D for point results and 3-D for maps; algebra identical).
 """

    path_gain = los + reflection + diffraction + transmission + scattering
    return {
        "path_gain": path_gain,
        "los_power": los.sum(),
        "reflection_power": reflection.sum(),
        "diffraction_power": diffraction.sum(),
        "transmission_power": transmission.sum(),
        "scattering_power": scattering.sum(),
    }


# Named aliases for the point (2-D) and map (3-D) finalizers; identical algebra.
finalize_point_components_reference = finalize_components_reference
finalize_component_maps_reference = finalize_components_reference