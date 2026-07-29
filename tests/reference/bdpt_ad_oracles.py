# Copyright Xingyu Chen.
# Implements bdpt ad oracles.

"""Implements bdpt ad oracles."""

from __future__ import annotations

import torch

from tests.ad._reference_fields import (
    SPEED_OF_LIGHT as _C0,
    VACUUM_PERMEABILITY as _MU0,
    VACUUM_PERMITTIVITY as _EPS0,
    _dot,
    _project,
    _safe_normalize,
    _stable_perp_basis,
    _transverse_project,
    slab_fresnel_reference,
    stack_rt_reference,
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
# Reflected light-subpath advance.
# ---------------------------------------------------------------------------


def _reflect_complex3(
    field: torch.Tensor, incident_dir: torch.Tensor, normal: torch.Tensor, eps_r: torch.Tensor,
    sigma_e: torch.Tensor, mu_r: torch.Tensor, gain: torch.Tensor, thickness: torch.Tensor,
    frequency: torch.Tensor,
) -> torch.Tensor:
    """One specular reflection Jones update (``transport::reflect_complex3``).

 Identical per-bounce algebra to ``reflection_sequence_reference`` (finite-slab
 ``slab_fresnel`` and the oriented s/p decomposition); ``field`` is the incoming
 Complex3 Jones field, ``incident_dir`` the incoming ray direction (frozen),
 ``normal`` the frozen hit normal.
 """

    ez = _ez_like(incident_dir)
    incident = _safe_normalize(incident_dir, ez)
    normal_unit = _safe_normalize(normal, ez)
    flip = _dot(incident, normal_unit) > 0.0
    oriented = torch.where(flip.unsqueeze(-1), -normal_unit, normal_unit)
    dot_in = _dot(incident, oriented)
    reflected = _safe_normalize(
        incident - oriented * (2.0 * dot_in).unsqueeze(-1), -incident
    )
    s_axis = _safe_normalize(
        torch.cross(oriented, incident, dim=-1), _stable_perp_basis(incident, oriented)
    )
    p_in = _safe_normalize(
        torch.cross(s_axis, incident, dim=-1), _stable_perp_basis(incident, s_axis)
    )
    p_out = _safe_normalize(
        torch.cross(s_axis, reflected, dim=-1), _stable_perp_basis(reflected, s_axis)
    )
    r_te, r_tm = slab_fresnel_reference(
        dot_in.abs(), eps_r, sigma_e, mu_r, gain, thickness, frequency
    )
    e_s = _project(field, s_axis)
    e_p = _project(field, p_in)
    return s_axis.to(field.dtype) * (r_te * e_s).unsqueeze(-1) + p_out.to(
        field.dtype
    ) * (r_tm * e_p).unsqueeze(-1)


def effective_power_reflectance(
    incident_dir: torch.Tensor, normal: torch.Tensor, eps_r: torch.Tensor, sigma_e: torch.Tensor,
    mu_r: torch.Tensor, frequency: torch.Tensor,
) -> torch.Tensor:
    """``effective_power_reflectance`` (bdpt_paths.cu): interface-only reflectance.

 ``|r_te*e_s|^2 + |r_tm*e_p|^2`` with single-INTERFACE Fresnel coefficients (no
 slab phase, no thickness, no gain) and the fixed x-hat transmit polarization
 projected transversely onto the wall s/p basis. This is the amplitude proxy the
 throughput carries; gain multiplies it outside (``sqrt(gain * R)``).
 """

    ez = _ez_like(incident_dir)
    incident = _safe_normalize(incident_dir, ez)
    normal_unit = _safe_normalize(normal, ez)
    facing = _dot(incident, normal_unit) > 0.0
    oriented = torch.where(facing.unsqueeze(-1), -normal_unit, normal_unit)
    dot_in = _dot(incident, oriented)
    cos_theta = (-dot_in).clamp(_SMALL_EPS, 1.0)
    sin2 = (1.0 - cos_theta * cos_theta).clamp_min(0.0)
    omega = (2.0 * torch.pi * frequency).clamp_min(_SMALL_EPS)
    eta = torch.complex(
        eps_r.clamp_min(_SMALL_EPS).to(torch.float64),
        -sigma_e.clamp_min(0.0).to(torch.float64) / (omega * _EPS0),
    )
    mu = mu_r.clamp_min(_SMALL_EPS).to(torch.float64)
    root = torch.sqrt(eta * mu - sin2.to(eta.dtype))
    mu_cos = (mu * cos_theta).to(eta.dtype)
    eta_cos = eta * cos_theta.to(eta.dtype)
    r_te = (mu_cos - root) / (mu_cos + root)
    r_tm = (eta_cos - root) / (eta_cos + root)

    s_raw = torch.cross(oriented, incident, dim=-1)
    s_len = torch.linalg.vector_norm(s_raw, dim=-1, keepdim=True)
    s_axis = s_raw / s_len.clamp_min(_EPS)
    p_axis = torch.cross(s_axis, incident, dim=-1)
    x_hat = torch.zeros_like(incident)
    x_hat[..., 0] = 1.0
    transverse = x_hat - incident * _dot(x_hat, incident).unsqueeze(-1)
    t_len = torch.linalg.vector_norm(transverse, dim=-1, keepdim=True)
    e_s = (_dot(transverse, s_axis) / t_len.squeeze(-1).clamp_min(_EPS))
    e_p = (_dot(transverse, p_axis) / t_len.squeeze(-1).clamp_min(_EPS))
    reflectance = (
        r_te.abs().square() * e_s * e_s + r_tm.abs().square() * e_p * e_p
    )
    # Degenerate s-axis (normal incidence) collapses to |r_te|^2 (r_te == r_tm).
    degenerate = (s_len.squeeze(-1) <= _SMALL_EPS) | (t_len.squeeze(-1) <= _SMALL_EPS)
    return torch.where(degenerate, r_te.abs().square(), reflectance)


def reflected_subpath_advance_reference(
    field_in: torch.Tensor, throughput_in: torch.Tensor, incident_dir: torch.Tensor,
    normal: torch.Tensor, eps_r: torch.Tensor, sigma_e: torch.Tensor, mu_r: torch.Tensor,
    gain: torch.Tensor, thickness: torch.Tensor, frequency: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Reflected-subpath derivatives reflected-subpath advance (field + throughput proxy).

 Live leaves: ``eps_r``/``sigma_e``/``gain``/``thickness`` (per-hit single slab),
 ``frequency``, and the upstream ``field_in``/``throughput_in``. Frozen:
 ``incident_dir``/``normal`` geometry, ``mu_r``. Returns the advanced Complex3
 field and the complex throughput (imag rides through the real amplitude).
 """

    field_out = _reflect_complex3(
        field_in, incident_dir, normal, eps_r, sigma_e, mu_r, gain, thickness, frequency
    )
    reflectance = effective_power_reflectance(
        incident_dir, normal, eps_r, sigma_e, mu_r, frequency
    )
    amplitude = (gain.clamp_min(0.0) * reflectance).clamp_min(0.0).sqrt()
    throughput_out = throughput_in * amplitude.to(throughput_in.dtype)
    return {"field": field_out, "throughput": throughput_out}


# ---------------------------------------------------------------------------
# Transmitted light-subpath advance.
# ---------------------------------------------------------------------------


def _sp_proxy_weights(
    incident: torch.Tensor, normal_in: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``sp_proxy_weights`` (bdpt_paths.cu): x-hat power weights onto the s/p basis."""

    s_raw = torch.cross(normal_in, incident, dim=-1)
    s_len = torch.linalg.vector_norm(s_raw, dim=-1, keepdim=True)
    s_axis = s_raw / s_len.clamp_min(_EPS)
    p_axis = torch.cross(s_axis, incident, dim=-1)
    x_hat = torch.zeros_like(incident)
    x_hat[..., 0] = 1.0
    transverse = x_hat - incident * _dot(x_hat, incident).unsqueeze(-1)
    t_len = torch.linalg.vector_norm(transverse, dim=-1, keepdim=True)
    e_s = _dot(transverse, s_axis) / t_len.squeeze(-1).clamp_min(_EPS)
    e_p = _dot(transverse, p_axis) / t_len.squeeze(-1).clamp_min(_EPS)
    degenerate = (s_len.squeeze(-1) <= _SMALL_EPS) | (t_len.squeeze(-1) <= _SMALL_EPS)
    w_s = torch.where(degenerate, torch.ones_like(e_s), e_s * e_s)
    w_p = torch.where(degenerate, torch.zeros_like(e_p), e_p * e_p)
    return w_s, w_p


def _layer_phase_index(
    eps_r: torch.Tensor, sigma_e: torch.Tensor, mu_r: torch.Tensor, omega: torch.Tensor,
    k0: torch.Tensor,
) -> torch.Tensor:
    """Re(k_layer)/k0 from the passive medium (``em::make_medium`` phase index)."""

    eps_abs = torch.complex(
        (_EPS0 * eps_r.clamp_min(_SMALL_EPS)).to(torch.float64),
        (-sigma_e.clamp_min(0.0) / omega).to(torch.float64),
    )
    mu_abs = (_MU0 * mu_r.clamp_min(_SMALL_EPS)).to(torch.complex128)
    k = torch.sqrt(eps_abs * mu_abs) * omega
    return (k.real / k0.clamp_min(_SMALL_EPS)).clamp_min(_SMALL_EPS)


def transmitted_subpath_advance_reference(
    field_in: torch.Tensor, throughput_in: torch.Tensor, incident_dir: torch.Tensor,
    normal: torch.Tensor, layer_thickness_m: torch.Tensor, layer_eps_r: torch.Tensor,
    layer_sigma_e: torch.Tensor, layer_mu_r: torch.Tensor, frequency: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Transmitted-subpath derivatives transmitted-subpath advance (single wall, per-row layers).

 ``layer_*`` are ``[N, L]`` per-row stacks (entry->exit) of the single wall each
 row crosses. Live leaves: ``layer_thickness_m``/``layer_eps_r``/``layer_sigma_e``,
 ``frequency``, upstream field/throughput. Frozen: geometry, ``layer_mu_r``.
 Returns the advanced field (with the lateral/interior compensation phase) and the
 complex throughput.
 """

    count, depth = layer_thickness_m.shape
    ez = _ez_like(incident_dir)
    incident = _safe_normalize(incident_dir, ez)
    normal_unit = _safe_normalize(normal, ez)
    facing = _dot(incident, normal_unit) > 0.0
    normal_in = torch.where(facing.unsqueeze(-1), -normal_unit, normal_unit)
    cos_theta = (-_dot(incident, normal_in)).clamp(_SMALL_EPS, 1.0)
    sin_theta = (1.0 - cos_theta * cos_theta).clamp_min(0.0).sqrt()
    omega = (2.0 * torch.pi * frequency).clamp_min(_SMALL_EPS)
    k0 = omega / _C0
    k_par = k0 * sin_theta

    field_rows = []
    throughput_rows = []
    for row in range(count):
        _, t_te = stack_rt_reference(
            cos_theta[row],
            layer_thickness_m[row],
            layer_eps_r[row],
            layer_sigma_e[row],
            layer_mu_r[row],
            frequency,
            _POL_TE,
        )
        _, t_tm = stack_rt_reference(
            cos_theta[row],
            layer_thickness_m[row],
            layer_eps_r[row],
            layer_sigma_e[row],
            layer_mu_r[row],
            frequency,
            _POL_TM,
        )
        # Vacuum-bounded stack: cap_t = |t|^2 (Re(Y_exit)/Re(Y_entry) == 1).
        cap_t_te = t_te.abs().square()
        cap_t_tm = t_tm.abs().square()

        total_thickness = torch.zeros((), dtype=torch.float64, device=field_in.device)
        lateral = torch.zeros((), dtype=torch.float64, device=field_in.device)
        for layer in range(depth):
            thickness = layer_thickness_m[row, layer].clamp_min(0.0)
            phase_index = _layer_phase_index(
                layer_eps_r[row, layer],
                layer_sigma_e[row, layer],
                layer_mu_r[row, layer],
                omega,
                k0,
            )
            sin_layer = sin_theta[row] / phase_index
            cos_layer = (1.0 - sin_layer * sin_layer).clamp_min(1.0e-6).sqrt()
            total_thickness = total_thickness + thickness
            lateral = lateral + thickness * (sin_layer / cos_layer)

        u_par = _safe_normalize(
            incident[row] + normal_in[row] * cos_theta[row],
            _stable_perp_basis(normal_in[row], incident[row]),
        )
        step = -normal_in[row] * total_thickness + u_par * lateral
        jump = torch.linalg.vector_norm(step, dim=-1)

        s_axis = _safe_normalize(
            torch.cross(normal_in[row], incident[row], dim=-1),
            _stable_perp_basis(incident[row], normal_in[row]),
        )
        p_axis = _safe_normalize(
            torch.cross(s_axis, incident[row], dim=-1),
            _stable_perp_basis(incident[row], s_axis),
        )
        e_s = _project(field_in[row], s_axis)
        e_p = _project(field_in[row], p_axis)
        updated = s_axis.to(field_in.dtype) * (t_te * e_s) + p_axis.to(
            field_in.dtype
        ) * (t_tm * e_p)
        compensation = torch.exp(
            -1.0j * (k_par[row] * lateral - k0 * jump)
        )
        field_rows.append(updated * compensation)

        w_s, w_p = _sp_proxy_weights(
            incident[row : row + 1], normal_in[row : row + 1]
        )
        amplitude = (cap_t_te * w_s[0] + cap_t_tm * w_p[0]).clamp_min(0.0).sqrt()
        throughput_rows.append(throughput_in[row] * amplitude.to(throughput_in.dtype))

    return {
        "field": torch.stack(field_rows, dim=0),
        "throughput": torch.stack(throughput_rows, dim=0),
    }


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