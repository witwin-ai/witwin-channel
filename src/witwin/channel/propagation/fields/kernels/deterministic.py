from __future__ import annotations

import torch

from witwin.channel.runtime import native_extension, validate_cuda_tensor


def deterministic_los_field(
    path_gain: torch.Tensor,
    path_length_m: torch.Tensor,
    *,
    frequency_hz: float,
) -> dict[str, torch.Tensor]:
    validate_cuda_tensor("path_gain", path_gain, dtype=torch.float32, ndim=1)
    validate_cuda_tensor("path_length_m", path_length_m, dtype=torch.float32, ndim=1)
    if path_length_m.shape != path_gain.shape:
        raise ValueError("path_length_m must match path_gain")
    if frequency_hz <= 0.0:
        raise ValueError("frequency_hz must be positive")

    native = native_extension()
    if native is None or not hasattr(native, "deterministic_los_field"):
        raise RuntimeError(
            "_channel.deterministic_los_field CUDA kernel is required"
        )
    exported = native.deterministic_los_field(
        path_gain, path_length_m, float(frequency_hz)
    )
    if not isinstance(exported, dict):
        raise TypeError("_channel.deterministic_los_field must return a dict")
    validate_cuda_tensor(
        "path_gain", exported["path_gain"], dtype=torch.float32, ndim=1
    )
    validate_cuda_tensor(
        "field_real", exported["field_real"], dtype=torch.float32, ndim=1
    )
    validate_cuda_tensor(
        "field_imag", exported["field_imag"], dtype=torch.float32, ndim=1
    )
    if exported["path_gain"].shape != path_gain.shape:
        raise ValueError("_channel.deterministic_los_field returned bad shape")
    return exported


def deterministic_diffraction_vector_field(
    x_re: torch.Tensor,
    x_im: torch.Tensor,
    y_re: torch.Tensor,
    y_im: torch.Tensor,
    z_re: torch.Tensor,
    z_im: torch.Tensor,
) -> dict[str, torch.Tensor]:
    validate_cuda_tensor("x_re", x_re, dtype=torch.float32, ndim=1)
    for name, tensor in {
        "x_im": x_im,
        "y_re": y_re,
        "y_im": y_im,
        "z_re": z_re,
        "z_im": z_im,
    }.items():
        validate_cuda_tensor(name, tensor, dtype=torch.float32, ndim=1)
        if tensor.shape != x_re.shape:
            raise ValueError(f"{name} must match x_re")

    native = native_extension()
    if native is None or not hasattr(native, "deterministic_diffraction_vector_field"):
        raise RuntimeError(
            "_channel.deterministic_diffraction_vector_field CUDA kernel is required"
        )
    exported = native.deterministic_diffraction_vector_field(
        x_re, x_im, y_re, y_im, z_re, z_im
    )
    if not isinstance(exported, dict):
        raise TypeError(
            "_channel.deterministic_diffraction_vector_field must return a dict"
        )
    validate_cuda_tensor(
        "path_gain", exported["path_gain"], dtype=torch.float32, ndim=1
    )
    validate_cuda_tensor(
        "field_real", exported["field_real"], dtype=torch.float32, ndim=1
    )
    validate_cuda_tensor(
        "field_imag", exported["field_imag"], dtype=torch.float32, ndim=1
    )
    if exported["path_gain"].shape != x_re.shape:
        raise ValueError(
            "_channel.deterministic_diffraction_vector_field returned bad shape"
        )
    return exported


def deterministic_reflection_field(
    tx_position: torch.Tensor,
    rx_position: torch.Tensor,
    hit_position: torch.Tensor,
    normal: torch.Tensor,
    tx_power: torch.Tensor,
    eps_r: torch.Tensor,
    sigma_e: torch.Tensor,
    mu_r: torch.Tensor,
    gain: torch.Tensor,
    *,
    frequency_hz: float,
) -> dict[str, torch.Tensor]:
    validate_cuda_tensor(
        "tx_position", tx_position, dtype=torch.float32, ndim=2, trailing_shape=(3,)
    )
    validate_cuda_tensor(
        "rx_position", rx_position, dtype=torch.float32, ndim=2, trailing_shape=(3,)
    )
    validate_cuda_tensor(
        "hit_position", hit_position, dtype=torch.float32, ndim=2, trailing_shape=(3,)
    )
    validate_cuda_tensor(
        "normal", normal, dtype=torch.float32, ndim=2, trailing_shape=(3,)
    )
    validate_cuda_tensor("tx_power", tx_power, dtype=torch.float32, ndim=1)
    validate_cuda_tensor("eps_r", eps_r, dtype=torch.float32, ndim=1)
    validate_cuda_tensor("sigma_e", sigma_e, dtype=torch.float32, ndim=1)
    validate_cuda_tensor("mu_r", mu_r, dtype=torch.float32, ndim=1)
    validate_cuda_tensor("gain", gain, dtype=torch.float32, ndim=1)
    count = tx_position.shape[0]
    if (
        rx_position.shape != tx_position.shape
        or hit_position.shape != tx_position.shape
        or normal.shape != tx_position.shape
    ):
        raise ValueError("reflection field vec3 tensors must have matching shape")
    for name, tensor in {
        "tx_power": tx_power,
        "eps_r": eps_r,
        "sigma_e": sigma_e,
        "mu_r": mu_r,
        "gain": gain,
    }.items():
        if tensor.shape[0] != count:
            raise ValueError(f"{name} must match path count")
    if frequency_hz <= 0.0:
        raise ValueError("frequency_hz must be positive")

    native = native_extension()
    if native is None or not hasattr(native, "deterministic_reflection_field"):
        raise RuntimeError(
            "_channel.deterministic_reflection_field CUDA kernel is required"
        )
    exported = native.deterministic_reflection_field(
        tx_position,
        rx_position,
        hit_position,
        normal,
        tx_power,
        eps_r,
        sigma_e,
        mu_r,
        gain,
        float(frequency_hz),
    )
    if not isinstance(exported, dict):
        raise TypeError(
            "_channel.deterministic_reflection_field must return a dict"
        )
    validate_cuda_tensor(
        "path_gain", exported["path_gain"], dtype=torch.float32, ndim=1
    )
    validate_cuda_tensor(
        "field_real", exported["field_real"], dtype=torch.float32, ndim=1
    )
    validate_cuda_tensor(
        "field_imag", exported["field_imag"], dtype=torch.float32, ndim=1
    )
    validate_cuda_tensor(
        "path_length_m", exported["path_length_m"], dtype=torch.float32, ndim=1
    )
    validate_cuda_tensor("delay_s", exported["delay_s"], dtype=torch.float32, ndim=1)
    if exported["path_length_m"].shape != (count,) or exported["delay_s"].shape != (
        count,
    ):
        raise ValueError(
            "_channel.deterministic_reflection_field returned bad length shape"
        )
    return exported


def deterministic_reflection_sequence_field(
    tx_position: torch.Tensor,
    rx_position: torch.Tensor,
    hit_positions: torch.Tensor,
    normals: torch.Tensor,
    tx_power: torch.Tensor,
    eps_r: torch.Tensor,
    sigma_e: torch.Tensor,
    mu_r: torch.Tensor,
    gain: torch.Tensor,
    *,
    frequency_hz: float,
) -> dict[str, torch.Tensor]:
    validate_cuda_tensor(
        "tx_position", tx_position, dtype=torch.float32, ndim=2, trailing_shape=(3,)
    )
    validate_cuda_tensor(
        "rx_position", rx_position, dtype=torch.float32, ndim=2, trailing_shape=(3,)
    )
    validate_cuda_tensor("hit_positions", hit_positions, dtype=torch.float32, ndim=3)
    validate_cuda_tensor("normals", normals, dtype=torch.float32, ndim=3)
    validate_cuda_tensor("tx_power", tx_power, dtype=torch.float32, ndim=1)
    validate_cuda_tensor("eps_r", eps_r, dtype=torch.float32, ndim=2)
    validate_cuda_tensor("sigma_e", sigma_e, dtype=torch.float32, ndim=2)
    validate_cuda_tensor("mu_r", mu_r, dtype=torch.float32, ndim=2)
    validate_cuda_tensor("gain", gain, dtype=torch.float32, ndim=2)
    count = tx_position.shape[0]
    if rx_position.shape != tx_position.shape:
        raise ValueError("rx_position must match tx_position")
    if hit_positions.shape[0] != count or hit_positions.shape[2] != 3:
        raise ValueError("hit_positions must have shape (path_count, depth, 3)")
    if normals.shape != hit_positions.shape:
        raise ValueError("normals must match hit_positions")
    depth_shape = hit_positions.shape[:2]
    for name, tensor in {
        "eps_r": eps_r,
        "sigma_e": sigma_e,
        "mu_r": mu_r,
        "gain": gain,
    }.items():
        if tensor.shape != depth_shape:
            raise ValueError(f"{name} must have shape (path_count, depth)")
    if tx_power.shape[0] != count:
        raise ValueError("tx_power must match path count")
    if frequency_hz <= 0.0:
        raise ValueError("frequency_hz must be positive")

    native = native_extension()
    if native is None or not hasattr(native, "deterministic_reflection_sequence_field"):
        raise RuntimeError(
            "_channel.deterministic_reflection_sequence_field CUDA kernel is required"
        )
    exported = native.deterministic_reflection_sequence_field(
        tx_position,
        rx_position,
        hit_positions,
        normals,
        tx_power,
        eps_r,
        sigma_e,
        mu_r,
        gain,
        float(frequency_hz),
    )
    if not isinstance(exported, dict):
        raise TypeError(
            "_channel.deterministic_reflection_sequence_field must return a dict"
        )
    validate_cuda_tensor(
        "path_gain", exported["path_gain"], dtype=torch.float32, ndim=1
    )
    validate_cuda_tensor(
        "field_real", exported["field_real"], dtype=torch.float32, ndim=1
    )
    validate_cuda_tensor(
        "field_imag", exported["field_imag"], dtype=torch.float32, ndim=1
    )
    validate_cuda_tensor(
        "path_length_m", exported["path_length_m"], dtype=torch.float32, ndim=1
    )
    validate_cuda_tensor("delay_s", exported["delay_s"], dtype=torch.float32, ndim=1)
    if exported["path_length_m"].shape != (count,) or exported["delay_s"].shape != (
        count,
    ):
        raise ValueError(
            "_channel.deterministic_reflection_sequence_field returned bad length shape"
        )
    return exported


def deterministic_delay_to_path_length(delay_s: torch.Tensor) -> torch.Tensor:
    validate_cuda_tensor("delay_s", delay_s, dtype=torch.float32, ndim=1)
    native = native_extension()
    if native is None or not hasattr(native, "deterministic_delay_to_path_length"):
        raise RuntimeError(
            "_channel.deterministic_delay_to_path_length CUDA kernel is required"
        )
    path_length = native.deterministic_delay_to_path_length(delay_s)
    validate_cuda_tensor("path_length_m", path_length, dtype=torch.float32, ndim=1)
    if path_length.shape != delay_s.shape:
        raise ValueError(
            "_channel.deterministic_delay_to_path_length returned bad shape"
        )
    return path_length


def deterministic_pack_complex(
    field_real: torch.Tensor, field_imag: torch.Tensor
) -> torch.Tensor:
    validate_cuda_tensor("field_real", field_real, dtype=torch.float32, ndim=1)
    validate_cuda_tensor("field_imag", field_imag, dtype=torch.float32, ndim=1)
    if field_imag.shape != field_real.shape:
        raise ValueError("field_imag must match field_real")
    native = native_extension()
    if native is None or not hasattr(native, "deterministic_pack_complex"):
        raise RuntimeError(
            "_channel.deterministic_pack_complex CUDA kernel is required"
        )
    field = native.deterministic_pack_complex(field_real, field_imag)
    validate_cuda_tensor("field", field, dtype=torch.complex64, ndim=1)
    if field.shape != field_real.shape:
        raise ValueError(
            "_channel.deterministic_pack_complex returned bad shape"
        )
    return field


def deterministic_phase_from_field(
    field_real: torch.Tensor, field_imag: torch.Tensor
) -> torch.Tensor:
    validate_cuda_tensor("field_real", field_real, dtype=torch.float32, ndim=1)
    validate_cuda_tensor("field_imag", field_imag, dtype=torch.float32, ndim=1)
    if field_imag.shape != field_real.shape:
        raise ValueError("field_imag must match field_real")
    native = native_extension()
    if native is None or not hasattr(native, "deterministic_phase_from_field"):
        raise RuntimeError(
            "_channel.deterministic_phase_from_field CUDA kernel is required"
        )
    phase = native.deterministic_phase_from_field(field_real, field_imag)
    validate_cuda_tensor("phase_rad", phase, dtype=torch.float32, ndim=1)
    if phase.shape != field_real.shape:
        raise ValueError(
            "_channel.deterministic_phase_from_field returned bad shape"
        )
    return phase


def deterministic_zero_field_phase(reference: torch.Tensor) -> dict[str, torch.Tensor]:
    validate_cuda_tensor("reference", reference, dtype=torch.float32, ndim=1)
    native = native_extension()
    if native is None or not hasattr(native, "deterministic_zero_field_phase"):
        raise RuntimeError(
            "_channel.deterministic_zero_field_phase CUDA kernel is required"
        )
    exported = native.deterministic_zero_field_phase(reference)
    if not isinstance(exported, dict):
        raise TypeError(
            "_channel.deterministic_zero_field_phase must return a dict"
        )
    validate_cuda_tensor(
        "path_field", exported["path_field"], dtype=torch.complex64, ndim=1
    )
    validate_cuda_tensor(
        "phase_rad", exported["phase_rad"], dtype=torch.float32, ndim=1
    )
    if (
        exported["path_field"].shape != reference.shape
        or exported["phase_rad"].shape != reference.shape
    ):
        raise ValueError(
            "_channel.deterministic_zero_field_phase returned bad shape"
        )
    return exported


def deterministic_phase_from_length(
    path_length_m: torch.Tensor, *, frequency_hz: float
) -> torch.Tensor:
    validate_cuda_tensor("path_length_m", path_length_m, dtype=torch.float32, ndim=1)
    if frequency_hz <= 0.0:
        raise ValueError("frequency_hz must be positive")
    native = native_extension()
    if native is None or not hasattr(native, "deterministic_phase_from_length"):
        raise RuntimeError(
            "_channel.deterministic_phase_from_length CUDA kernel is required"
        )
    phase = native.deterministic_phase_from_length(path_length_m, float(frequency_hz))
    validate_cuda_tensor("phase_rad", phase, dtype=torch.float32, ndim=1)
    if phase.shape != path_length_m.shape:
        raise ValueError(
            "_channel.deterministic_phase_from_length returned bad shape"
        )
    return phase


def deterministic_field_from_power_phase(
    path_gain: torch.Tensor, phase_rad: torch.Tensor
) -> dict[str, torch.Tensor]:
    validate_cuda_tensor("path_gain", path_gain, dtype=torch.float32, ndim=1)
    validate_cuda_tensor("phase_rad", phase_rad, dtype=torch.float32, ndim=1)
    if phase_rad.shape != path_gain.shape:
        raise ValueError("phase_rad must match path_gain")
    native = native_extension()
    if native is None or not hasattr(native, "deterministic_field_from_power_phase"):
        raise RuntimeError(
            "_channel.deterministic_field_from_power_phase CUDA kernel is required"
        )
    exported = native.deterministic_field_from_power_phase(path_gain, phase_rad)
    if not isinstance(exported, dict):
        raise TypeError(
            "_channel.deterministic_field_from_power_phase must return a dict"
        )
    validate_cuda_tensor(
        "field_real", exported["field_real"], dtype=torch.float32, ndim=1
    )
    validate_cuda_tensor(
        "field_imag", exported["field_imag"], dtype=torch.float32, ndim=1
    )
    if (
        exported["field_real"].shape != path_gain.shape
        or exported["field_imag"].shape != path_gain.shape
    ):
        raise ValueError(
            "_channel.deterministic_field_from_power_phase returned bad shape"
        )
    return exported
