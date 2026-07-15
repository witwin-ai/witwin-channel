from __future__ import annotations

import torch

from .kernels import fields as field_kernels


LIGHT_SPEED_M_PER_S = 299_792_458.0
_EPSILON0 = 8.854187817e-12


def free_space_phase_rad(path_length_m: torch.Tensor, frequency_hz: float) -> torch.Tensor:
    return field_kernels.deterministic_phase_from_length(path_length_m.contiguous(), frequency_hz=float(frequency_hz))


def free_space_complex_field(
    path_gain: torch.Tensor,
    path_length_m: torch.Tensor,
    frequency_hz: float,
) -> torch.Tensor:
    exported = field_kernels.deterministic_los_field(
        path_gain.contiguous(),
        path_length_m.contiguous(),
        frequency_hz=float(frequency_hz),
    )
    return field_kernels.deterministic_pack_complex(exported["field_real"], exported["field_imag"])


def equivalent_field_from_power_phase(path_gain: torch.Tensor, phase_rad: torch.Tensor) -> torch.Tensor:
    exported = field_kernels.deterministic_field_from_power_phase(path_gain.contiguous(), phase_rad.contiguous())
    return field_kernels.deterministic_pack_complex(exported["field_real"], exported["field_imag"])


def phase_rad_from_complex_field(path_field: torch.Tensor) -> torch.Tensor:
    field = path_field.contiguous()
    return field_kernels.deterministic_phase_from_field(
        field.real.contiguous(),
        field.imag.contiguous(),
    )


def equivalent_field_from_vector_components(
    x_re: torch.Tensor,
    x_im: torch.Tensor,
    y_re: torch.Tensor,
    y_im: torch.Tensor,
    z_re: torch.Tensor,
    z_im: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    exported = field_kernels.deterministic_diffraction_vector_field(
        x_re.contiguous(),
        x_im.contiguous(),
        y_re.contiguous(),
        y_im.contiguous(),
        z_re.contiguous(),
        z_im.contiguous(),
    )
    field = field_kernels.deterministic_pack_complex(exported["field_real"], exported["field_imag"])
    return exported["path_gain"], field


def _fresnel_scalar_coefficient(**_kwargs: object) -> torch.Tensor:
    raise RuntimeError("deterministic Fresnel coefficient evaluation requires native field kernels")


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
    exported = field_kernels.deterministic_reflection_field(
        tx_position=tx_position.contiguous(),
        rx_position=rx_position.contiguous(),
        hit_position=hit_position.contiguous(),
        normal=normal.contiguous(),
        tx_power=tx_power_w.contiguous(),
        eps_r=eps_r.contiguous(),
        sigma_e=sigma_e.contiguous(),
        mu_r=mu_r.contiguous(),
        gain=gain.contiguous(),
        frequency_hz=float(frequency_hz),
    )
    field = field_kernels.deterministic_pack_complex(exported["field_real"], exported["field_imag"])
    return exported["path_gain"], field


def reflection_field_from_coefficient(
    *,
    coeff: torch.Tensor,
    path_length: torch.Tensor,
    tx_power_w: torch.Tensor,
    frequency_hz: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    del coeff, path_length, tx_power_w, frequency_hz
    raise RuntimeError("reflection field from coefficient requires native field kernels")


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
    exported = field_kernels.deterministic_reflection_sequence_field(
        tx_position=tx_position.contiguous(),
        rx_position=rx_position.contiguous(),
        hit_positions=hit_positions.contiguous(),
        normals=normals.contiguous(),
        tx_power=tx_power_w.contiguous(),
        eps_r=eps_r.contiguous(),
        sigma_e=sigma_e.contiguous(),
        mu_r=mu_r.contiguous(),
        gain=gain.contiguous(),
        frequency_hz=float(frequency_hz),
    )
    field = field_kernels.deterministic_pack_complex(exported["field_real"], exported["field_imag"])
    return exported["path_gain"], field, exported["path_length_m"]
