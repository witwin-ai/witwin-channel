"""Pure-torch complex128 reference implementations of the field kernels.

Test-only mirrors of the exact formulas in ``native/channel_native/
field_transport.cuh`` (free-space carrier, finite-slab Fresnel reflection
chain) and ``native/channel_native/em/*.cuh`` (Rouard transmission stack),
used for (a) forward parity against the native float32 kernels and (b) as a
gradient oracle: torch autograd through these functions defines the exact
Wirtinger-convention derivatives the native backward/jvp companions must
reproduce (plan 07 section 9.1).

Every function is differentiable with respect to the material tensors, the
0-d ``frequency`` tensor and (since plan 07 AD-2) the continuous geometry
arguments (source, target, interaction positions/normals); only the discrete
winner (validity masks, material ids, normal-flip branches) is fixed.
"""

from __future__ import annotations

import torch

SPEED_OF_LIGHT = 299792458.0
VACUUM_PERMITTIVITY = 8.8541878128e-12
VACUUM_PERMEABILITY = 1.25663706212e-6
_EPS = 1.0e-10  # utd::UTD_EPS
_SMALL_EPS = 1.0e-6  # utd::UTD_SMALL_EPS
_POL_TE = 0
_POL_TM = 1


def _safe_normalize(v: torch.Tensor, fallback: torch.Tensor) -> torch.Tensor:
    norm = torch.linalg.vector_norm(v, dim=-1, keepdim=True)
    fallback_norm = torch.linalg.vector_norm(fallback, dim=-1, keepdim=True)
    return torch.where(
        norm > _SMALL_EPS, v / (norm + _EPS), fallback / (fallback_norm + _EPS)
    )


def _stable_perp_basis(ray_dir: torch.Tensor, preferred: torch.Tensor) -> torch.Tensor:
    proj = preferred - ray_dir * (preferred * ray_dir).sum(-1, keepdim=True)
    e_z = torch.tensor([0.0, 0.0, 1.0], dtype=ray_dir.dtype, device=ray_dir.device)
    e_y = torch.tensor([0.0, 1.0, 0.0], dtype=ray_dir.dtype, device=ray_dir.device)
    alt_axis = torch.where(ray_dir[..., 2:3].abs() < 0.9, e_z, e_y)
    alt_proj = alt_axis - ray_dir * (alt_axis * ray_dir).sum(-1, keepdim=True)
    return _safe_normalize(proj, alt_proj)


def _dot(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return (a * b).sum(-1)


def _project(value: torch.Tensor, axis: torch.Tensor) -> torch.Tensor:
    return (value * axis.to(value.dtype)).sum(-1)


def _outputs(
    value: torch.Tensor,
    rx_axis: torch.Tensor,
    tx_power: torch.Tensor,
) -> dict[str, torch.Tensor]:
    coefficient = _project(value, rx_axis)
    amplitude = tx_power.clamp_min(0.0).sqrt()
    path_field = coefficient * amplitude
    return {
        "field_vector": value,
        "coefficient": coefficient,
        "path_field": path_field,
        "path_gain": path_field.real.square() + path_field.imag.square(),
    }


def free_space_reference(
    source: torch.Tensor,
    target: torch.Tensor,
    tx_power: torch.Tensor,
    tx_polarization: torch.Tensor,
    rx_polarization: torch.Tensor,
    frequency: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Mirror of free_space_complex3 + project_receiver (float64/complex128)."""

    offset = target - source
    distance = torch.linalg.vector_norm(offset, dim=-1)
    e_z = torch.tensor([0.0, 0.0, 1.0], dtype=source.dtype, device=source.device)
    direction = _safe_normalize(offset, e_z.expand_as(offset))
    tx_axis = _stable_perp_basis(direction, tx_polarization)
    rx_axis = _stable_perp_basis(direction, rx_polarization)
    wave_number = 2.0 * torch.pi * frequency / SPEED_OF_LIGHT
    amplitude = 1.0 / (
        2.0 * wave_number.clamp_min(_SMALL_EPS) * distance.clamp_min(_EPS)
    )
    carrier = amplitude * torch.exp(-1.0j * wave_number * distance)
    value = tx_axis.to(carrier.dtype) * carrier.unsqueeze(-1)
    return _outputs(value, rx_axis, tx_power)


def slab_fresnel_reference(
    cos_theta: torch.Tensor,
    eps_r: torch.Tensor,
    sigma_e: torch.Tensor,
    mu_r: torch.Tensor,
    gain: torch.Tensor,
    thickness: torch.Tensor,
    frequency: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Mirror of field_transport::slab_fresnel (thin-slab interior Fabry-Perot)."""

    omega = (2.0 * torch.pi * frequency).clamp_min(_SMALL_EPS)
    wavelength = SPEED_OF_LIGHT / frequency
    ct = cos_theta.abs().clamp(_SMALL_EPS, 1.0)
    sin2 = (1.0 - ct * ct).clamp_min(0.0)
    eta = torch.complex(
        eps_r.clamp_min(_SMALL_EPS).to(torch.float64),
        -sigma_e.clamp_min(0.0).to(torch.float64) / (omega * VACUUM_PERMITTIVITY),
    )
    mu = mu_r.clamp_min(_SMALL_EPS)
    root = torch.sqrt(eta * mu - sin2)
    mu_ct = (mu * ct).to(eta.dtype)
    eta_ct = eta * ct
    interface_te = (mu_ct - root) / (mu_ct + root)
    interface_tm = (eta_ct - root) / (eta_ct + root)
    q = root * (
        2.0 * torch.pi * thickness.clamp_min(0.0) / wavelength.clamp_min(_SMALL_EPS)
    )
    phase = torch.exp(-2.0j * q)
    r_te = gain * (interface_te * (1.0 - phase)) / (
        1.0 - interface_te * interface_te * phase
    )
    r_tm = gain * (interface_tm * (1.0 - phase)) / (
        1.0 - interface_tm * interface_tm * phase
    )
    return r_te, r_tm


def reflection_sequence_reference(
    source: torch.Tensor,
    target: torch.Tensor,
    interaction_positions: torch.Tensor,
    interaction_normals: torch.Tensor,
    tx_power: torch.Tensor,
    tx_polarization: torch.Tensor,
    rx_polarization: torch.Tensor,
    eps_r: torch.Tensor,
    sigma_e: torch.Tensor,
    mu_r: torch.Tensor,
    gain: torch.Tensor,
    thickness: torch.Tensor,
    frequency: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Mirror of reflection_sequence_kernel (general depth)."""

    depth = interaction_positions.shape[1]
    e_z = torch.tensor([0.0, 0.0, 1.0], dtype=source.dtype, device=source.device)
    previous = source
    incident = _safe_normalize(
        interaction_positions[:, 0] - previous, e_z.expand_as(source)
    )
    tx_axis = _stable_perp_basis(incident, tx_polarization)
    value = tx_axis.to(torch.complex128)
    total_length = torch.zeros_like(tx_power).to(torch.float64)
    outgoing = incident
    for bounce in range(depth):
        hit = interaction_positions[:, bounce]
        segment = hit - previous
        incident = _safe_normalize(segment, outgoing)
        total_length = total_length + torch.linalg.vector_norm(segment, dim=-1)
        normal = _safe_normalize(
            interaction_normals[:, bounce], e_z.expand_as(source)
        )
        flip = _dot(incident, normal) > 0.0
        oriented = torch.where(flip.unsqueeze(-1), -normal, normal)
        dot_in = _dot(incident, oriented)
        reflected = _safe_normalize(
            incident - oriented * (2.0 * dot_in).unsqueeze(-1), -incident
        )
        s_axis = _safe_normalize(
            torch.cross(oriented, incident, dim=-1),
            _stable_perp_basis(incident, oriented),
        )
        p_in = _safe_normalize(
            torch.cross(s_axis, incident, dim=-1),
            _stable_perp_basis(incident, s_axis),
        )
        p_out = _safe_normalize(
            torch.cross(s_axis, reflected, dim=-1),
            _stable_perp_basis(reflected, s_axis),
        )
        r_te, r_tm = slab_fresnel_reference(
            dot_in.abs(),
            eps_r[:, bounce],
            sigma_e[:, bounce],
            mu_r[:, bounce],
            gain[:, bounce],
            thickness[:, bounce],
            frequency,
        )
        e_s = _project(value, s_axis)
        e_p = _project(value, p_in)
        value = s_axis.to(value.dtype) * (r_te * e_s).unsqueeze(-1) + p_out.to(
            value.dtype
        ) * (r_tm * e_p).unsqueeze(-1)
        outgoing = reflected
        previous = hit
    final_offset = target - previous
    final_direction = _safe_normalize(final_offset, outgoing)
    total_length = total_length + torch.linalg.vector_norm(final_offset, dim=-1)
    wave_number = 2.0 * torch.pi * frequency / SPEED_OF_LIGHT
    amplitude = 1.0 / (2.0 * wave_number * total_length.clamp_min(_EPS))
    propagation = amplitude * torch.exp(-1.0j * wave_number * total_length)
    value = value * propagation.unsqueeze(-1)
    rx_axis = _stable_perp_basis(final_direction, rx_polarization)
    return _outputs(value, rx_axis, tx_power)


def _sqrt_passive(z: torch.Tensor) -> torch.Tensor:
    """em::c_sqrt_passive on the passive half plane (Im(z) <= 0)."""

    assert bool((z.imag <= 1.0e-30).all()), "passive sqrt expects Im(z) <= 0"
    return torch.sqrt(z)


def stack_rt_reference(
    cos_theta: torch.Tensor,
    layer_thickness: torch.Tensor,
    layer_eps_r: torch.Tensor,
    layer_sigma_e: torch.Tensor,
    layer_mu_r: torch.Tensor,
    frequency: torch.Tensor,
    pol: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Mirror of em::stack_rt (backward Rouard recursion, one wall).

    ``layer_*`` are 1-d tensors ordered entry -> exit; returns (r, t).
    """

    omega = (2.0 * torch.pi * frequency).clamp_min(_SMALL_EPS)
    ct = cos_theta.abs().clamp(_SMALL_EPS, 1.0)
    sin2 = (1.0 - ct * ct).clamp_min(0.0)
    k_entry = omega / SPEED_OF_LIGHT
    k_par = k_entry * sin2.sqrt()

    def admittance(eps_abs: torch.Tensor, mu_abs: torch.Tensor, k_z: torch.Tensor):
        if pol == _POL_TE:
            return k_z / (mu_abs * omega)
        return (eps_abs * omega) / k_z

    kz_entry = (k_entry * ct).to(torch.complex128)
    eps_vacuum = torch.tensor(
        VACUUM_PERMITTIVITY, dtype=torch.complex128, device=cos_theta.device
    )
    mu_vacuum = torch.tensor(
        VACUUM_PERMEABILITY, dtype=torch.complex128, device=cos_theta.device
    )
    y_entry = admittance(eps_vacuum, mu_vacuum, kz_entry)
    y_exit = y_entry

    count = int(layer_thickness.shape[0])
    if count == 0:
        one = torch.ones((), dtype=torch.complex128, device=cos_theta.device)
        return torch.zeros_like(one), one

    def medium(index: int):
        eps_abs = torch.complex(
            VACUUM_PERMITTIVITY * layer_eps_r[index].clamp_min(_SMALL_EPS).to(torch.float64),
            -layer_sigma_e[index].clamp_min(0.0).to(torch.float64) / omega,
        )
        mu_abs = (
            VACUUM_PERMEABILITY * layer_mu_r[index].clamp_min(_SMALL_EPS)
        ).to(torch.complex128)
        k = _sqrt_passive(eps_abs * mu_abs) * omega
        k_z = _sqrt_passive(k * k - (k_par * k_par).to(torch.complex128))
        return eps_abs, mu_abs, k_z

    eps_below, mu_below, kz_below = medium(count - 1)
    y_below = admittance(eps_below, mu_below, kz_below)
    r_total = (y_below - y_exit) / (y_below + y_exit)
    t_total = 2.0 * y_below / (y_below + y_exit)

    for layer in range(count - 1, -1, -1):
        phase = torch.exp(-1.0j * kz_below * layer_thickness[layer].clamp_min(0.0))
        phase2 = phase * phase
        if layer > 0:
            eps_above, mu_above, kz_above = medium(layer - 1)
            y_above = admittance(eps_above, mu_above, kz_above)
        else:
            kz_above = kz_entry
            y_above = y_entry
        r_top = (y_above - y_below) / (y_above + y_below)
        t_top = 2.0 * y_above / (y_above + y_below)
        loop = phase2 * r_total
        denominator = 1.0 + r_top * loop
        r_total = (r_top + loop) / denominator
        t_total = t_top * phase * t_total / denominator
        kz_below = kz_above
        y_below = y_above

    return r_total, t_total


def transmission_sequence_reference(
    source: torch.Tensor,
    target: torch.Tensor,
    interaction_normals: torch.Tensor,
    interaction_material_id: torch.Tensor,
    interaction_valid: torch.Tensor,
    tx_power: torch.Tensor,
    tx_polarization: torch.Tensor,
    rx_polarization: torch.Tensor,
    layer_offset: torch.Tensor,
    layer_count: torch.Tensor,
    layer_thickness_m: torch.Tensor,
    layer_eps_r: torch.Tensor,
    layer_sigma_e: torch.Tensor,
    layer_mu_r: torch.Tensor,
    frequency: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Mirror of transmission_sequence_kernel (straight thin-sheet chain)."""

    count, depth = interaction_valid.shape
    e_z = torch.tensor([0.0, 0.0, 1.0], dtype=source.dtype, device=source.device)
    offset = target - source
    total_length = torch.linalg.vector_norm(offset, dim=-1)
    direction = _safe_normalize(offset, e_z.expand_as(source))
    tx_axis = _stable_perp_basis(direction, tx_polarization)
    rx_axis = _stable_perp_basis(direction, rx_polarization)
    rows = []
    for index in range(count):
        value = tx_axis[index].to(torch.complex128)
        carrier_length = total_length[index]
        for wall in range(depth):
            if not bool(interaction_valid[index, wall]):
                continue
            material = int(interaction_material_id[index, wall])
            normal = _safe_normalize(interaction_normals[index, wall], e_z)
            if float(_dot(direction[index], normal)) > 0.0:
                normal = -normal
            cos_theta = _dot(direction[index], normal).abs().clamp(_SMALL_EPS, 1.0)
            first = int(layer_offset[material])
            layers = slice(first, first + int(layer_count[material]))
            _, t_te = stack_rt_reference(
                cos_theta,
                layer_thickness_m[layers],
                layer_eps_r[layers],
                layer_sigma_e[layers],
                layer_mu_r[layers],
                frequency,
                _POL_TE,
            )
            _, t_tm = stack_rt_reference(
                cos_theta,
                layer_thickness_m[layers],
                layer_eps_r[layers],
                layer_sigma_e[layers],
                layer_mu_r[layers],
                frequency,
                _POL_TM,
            )
            s_axis = _safe_normalize(
                torch.cross(normal, direction[index], dim=-1),
                _stable_perp_basis(direction[index], normal),
            )
            p_axis = _safe_normalize(
                torch.cross(s_axis, direction[index], dim=-1),
                _stable_perp_basis(direction[index], s_axis),
            )
            e_s = _project(value, s_axis)
            e_p = _project(value, p_axis)
            value = s_axis.to(value.dtype) * (t_te * e_s) + p_axis.to(value.dtype) * (
                t_tm * e_p
            )
            wall_thickness = layer_thickness_m[layers].clamp_min(0.0).sum()
            carrier_length = carrier_length - wall_thickness * cos_theta
        wave_number = 2.0 * torch.pi * frequency / SPEED_OF_LIGHT
        amplitude = 1.0 / (2.0 * wave_number * total_length[index].clamp_min(_EPS))
        propagation = amplitude * torch.exp(-1.0j * wave_number * carrier_length)
        rows.append(value * propagation)
    value = torch.stack(rows, dim=0)
    return _outputs(value, rx_axis, tx_power)
