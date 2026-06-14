from __future__ import annotations

import math

import torch


LIGHT_SPEED_M_PER_S = 299_792_458.0
_EPSILON0 = 8.854187817e-12


def free_space_phase_rad(path_length_m: torch.Tensor, frequency_hz: float) -> torch.Tensor:
    phase = (2.0 * math.pi * float(frequency_hz) / LIGHT_SPEED_M_PER_S) * path_length_m
    return torch.remainder(phase.to(dtype=torch.float32), 2.0 * math.pi)


def free_space_complex_field(
    path_gain: torch.Tensor,
    path_length_m: torch.Tensor,
    frequency_hz: float,
) -> torch.Tensor:
    if path_gain.shape != path_length_m.shape:
        raise ValueError("path_gain and path_length_m must have the same shape")
    amplitude = torch.sqrt(path_gain.to(dtype=torch.float32).clamp_min(0.0))
    phase = -free_space_phase_rad(path_length_m.to(device=path_gain.device), frequency_hz)
    return torch.polar(amplitude, phase).to(dtype=torch.complex64).contiguous()


def equivalent_field_from_power_phase(path_gain: torch.Tensor, phase_rad: torch.Tensor) -> torch.Tensor:
    amplitude = torch.sqrt(path_gain.to(dtype=torch.float32).clamp_min(0.0))
    return torch.polar(amplitude, -phase_rad.to(device=path_gain.device, dtype=torch.float32)).to(
        dtype=torch.complex64
    ).contiguous()


def phase_rad_from_complex_field(path_field: torch.Tensor) -> torch.Tensor:
    phase = -torch.angle(path_field.to(dtype=torch.complex64))
    return torch.remainder(phase.to(dtype=torch.float32), 2.0 * math.pi).contiguous()


def equivalent_field_from_vector_components(
    x_re: torch.Tensor,
    x_im: torch.Tensor,
    y_re: torch.Tensor,
    y_im: torch.Tensor,
    z_re: torch.Tensor,
    z_im: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    components = torch.stack(
        (
            torch.complex(x_re.to(dtype=torch.float32), x_im.to(dtype=torch.float32)),
            torch.complex(y_re.to(dtype=torch.float32), y_im.to(dtype=torch.float32)),
            torch.complex(z_re.to(dtype=torch.float32), z_im.to(dtype=torch.float32)),
        ),
        dim=1,
    ).to(dtype=torch.complex64)
    component_power = components.abs().square().to(dtype=torch.float32)
    path_gain = component_power.sum(dim=1).to(dtype=torch.float32).contiguous()
    dominant = component_power.argmax(dim=1)
    dominant_field = components.gather(1, dominant[:, None]).reshape(-1)
    phase = torch.angle(dominant_field)
    path_field = torch.polar(torch.sqrt(path_gain.clamp_min(0.0)), phase.to(dtype=torch.float32))
    return path_gain, path_field.to(dtype=torch.complex64).contiguous()


def _fallback_transverse(direction: torch.Tensor) -> torch.Tensor:
    fallback = torch.tensor((0.0, 0.0, 1.0), device=direction.device, dtype=torch.float32).expand_as(direction)
    fallback_alt = torch.tensor((0.0, 1.0, 0.0), device=direction.device, dtype=torch.float32).expand_as(direction)
    fallback = torch.where(direction[:, 2:3].abs() < 0.9, fallback, fallback_alt)
    fallback = fallback - (fallback * direction).sum(dim=1, keepdim=True) * direction
    return torch.nn.functional.normalize(fallback, dim=1, eps=1.0e-6)


def _fresnel_scalar_coefficient(
    *,
    incident: torch.Tensor,
    normal: torch.Tensor,
    eps_r: torch.Tensor,
    sigma_e: torch.Tensor,
    mu_r: torch.Tensor,
    gain: torch.Tensor,
    frequency_hz: float,
) -> torch.Tensor:
    incident = torch.nn.functional.normalize(incident, dim=1, eps=1.0e-6)
    n = torch.nn.functional.normalize(normal, dim=1, eps=1.0e-6)
    n = torch.where(((incident * n).sum(dim=1, keepdim=True) > 0.0), -n, n)
    cos_theta = (incident * n).sum(dim=1).abs().clamp(1.0e-6, 1.0)
    sin2 = (1.0 - cos_theta.square()).clamp_min(0.0)

    omega = 2.0 * math.pi * float(frequency_hz)
    eta = torch.complex(
        eps_r.to(dtype=torch.float32).clamp_min(1.0e-6),
        -sigma_e.to(dtype=torch.float32).clamp_min(0.0) / (omega * _EPSILON0),
    )
    mu = torch.complex(mu_r.to(dtype=torch.float32).clamp_min(1.0e-6), torch.zeros_like(mu_r, dtype=torch.float32))
    root = torch.sqrt(mu * eta - torch.complex(sin2, torch.zeros_like(sin2)))
    mu_cos = torch.complex(mu_r.to(dtype=torch.float32).clamp_min(1.0e-6) * cos_theta, torch.zeros_like(cos_theta))
    eta_cos = eta * torch.complex(cos_theta, torch.zeros_like(cos_theta))
    r_te = (mu_cos - root) / (mu_cos + root)
    r_tm = (eta_cos - root) / (eta_cos + root)

    s_hat = torch.cross(n, incident, dim=1)
    s_norm = torch.linalg.vector_norm(s_hat, dim=1, keepdim=True)
    fallback = _fallback_transverse(incident)
    s_hat = torch.where(s_norm > 1.0e-6, s_hat / s_norm.clamp_min(1.0e-6), fallback)
    p_in = torch.nn.functional.normalize(torch.cross(s_hat, incident, dim=1), dim=1, eps=1.0e-6)
    tx_pol = torch.tensor((1.0, 0.0, 0.0), device=incident.device, dtype=torch.float32).expand_as(incident)
    transverse = tx_pol - (tx_pol * incident).sum(dim=1, keepdim=True) * incident
    transverse_norm = torch.linalg.vector_norm(transverse, dim=1, keepdim=True)
    transverse = torch.where(
        transverse_norm > 1.0e-6,
        transverse / transverse_norm.clamp_min(1.0e-6),
        fallback,
    )
    e_s = torch.complex((transverse * s_hat).sum(dim=1), torch.zeros_like(cos_theta))
    e_p = torch.complex((transverse * p_in).sum(dim=1), torch.zeros_like(cos_theta))
    return gain.to(dtype=torch.float32).to(dtype=torch.complex64) * (
        r_te.to(torch.complex64) * e_s + r_tm.to(torch.complex64) * e_p
    )


def reflection_complex_field(
    *,
    tx_position: torch.Tensor,
    rx_position: torch.Tensor,
    hit_position: torch.Tensor,
    normal: torch.Tensor,
    tx_power_w: torch.Tensor,
    eps_r: torch.Tensor,
    sigma_e: torch.Tensor,
    mu_r: torch.Tensor,
    gain: torch.Tensor,
    frequency_hz: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Evaluate a scalar equivalent of the native EPC reflection field.

    The underlying RayDN EPC field kernel carries a vector field. The public
    deterministic path table and accumulator use scalar complex fields, so this
    returns an equivalent scalar whose magnitude squared is the vector power and
    whose phase follows the Fresnel coefficient and propagation phase.
    """

    path_length = (
        torch.linalg.vector_norm(hit_position - tx_position, dim=1)
        + torch.linalg.vector_norm(rx_position - hit_position, dim=1)
    ).clamp_min(1.0e-6)
    coeff = _fresnel_scalar_coefficient(
        incident=hit_position - tx_position,
        normal=normal,
        eps_r=eps_r,
        sigma_e=sigma_e,
        mu_r=mu_r,
        gain=gain,
        frequency_hz=frequency_hz,
    )
    return reflection_field_from_coefficient(
        coeff=coeff,
        path_length=path_length,
        tx_power_w=tx_power_w,
        frequency_hz=frequency_hz,
    )


def reflection_field_from_coefficient(
    *,
    coeff: torch.Tensor,
    path_length: torch.Tensor,
    tx_power_w: torch.Tensor,
    frequency_hz: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    wavelength = LIGHT_SPEED_M_PER_S / float(frequency_hz)
    phase = torch.polar(
        torch.ones_like(path_length, dtype=torch.float32),
        (-(2.0 * math.pi / wavelength) * path_length).to(dtype=torch.float32),
    ).to(torch.complex64)
    amplitude = (
        torch.sqrt(tx_power_w.to(device=path_length.device, dtype=torch.float32).clamp_min(0.0))
        * (wavelength / (4.0 * math.pi))
        / path_length
    ).to(torch.float32)
    field = amplitude.to(torch.complex64) * coeff * phase
    path_gain = field.abs().square().to(dtype=torch.float32).contiguous()
    return path_gain, field.to(dtype=torch.complex64).contiguous()


def reflection_sequence_complex_field(
    *,
    tx_position: torch.Tensor,
    rx_position: torch.Tensor,
    hit_positions: torch.Tensor,
    normals: torch.Tensor,
    tx_power_w: torch.Tensor,
    eps_r: torch.Tensor,
    sigma_e: torch.Tensor,
    mu_r: torch.Tensor,
    gain: torch.Tensor,
    frequency_hz: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    depth = int(hit_positions.shape[1])
    segments = [hit_positions[:, 0, :] - tx_position]
    for index in range(1, depth):
        segments.append(hit_positions[:, index, :] - hit_positions[:, index - 1, :])
    coeff = torch.ones((hit_positions.shape[0],), device=hit_positions.device, dtype=torch.complex64)
    for index, segment in enumerate(segments):
        coeff = coeff * _fresnel_scalar_coefficient(
            incident=segment,
            normal=normals[:, index, :],
            eps_r=eps_r[:, index],
            sigma_e=sigma_e[:, index],
            mu_r=mu_r[:, index],
            gain=gain[:, index],
            frequency_hz=frequency_hz,
        )
    path_length = torch.linalg.vector_norm(segments[0], dim=1)
    for segment in segments[1:]:
        path_length = path_length + torch.linalg.vector_norm(segment, dim=1)
    path_length = (path_length + torch.linalg.vector_norm(rx_position - hit_positions[:, -1, :], dim=1)).clamp_min(
        1.0e-6
    )
    path_gain, path_field = reflection_field_from_coefficient(
        coeff=coeff,
        path_length=path_length,
        tx_power_w=tx_power_w,
        frequency_hz=frequency_hz,
    )
    return path_gain, path_field, path_length.to(dtype=torch.float32).contiguous()
