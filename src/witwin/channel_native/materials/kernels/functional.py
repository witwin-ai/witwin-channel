from __future__ import annotations

import torch

from witwin.channel_native.runtime.symbols import native_extension
from witwin.channel_native.runtime.symbols import required_symbol as _required_native_op
from witwin.channel_native.runtime.tensor_contracts import validate_cuda_tensor

from .contracts import _validate_layer_csr


def bdpt_face_material_tensors(
    material_eps_r: torch.Tensor,
    material_sigma_e: torch.Tensor,
    material_mu_r: torch.Tensor,
    face_material_id: torch.Tensor,
) -> dict[str, torch.Tensor]:
    validate_cuda_tensor("material_eps_r", material_eps_r, dtype=torch.float32, ndim=1)
    validate_cuda_tensor(
        "material_sigma_e", material_sigma_e, dtype=torch.float32, ndim=1
    )
    validate_cuda_tensor("material_mu_r", material_mu_r, dtype=torch.float32, ndim=1)
    validate_cuda_tensor(
        "face_material_id", face_material_id, dtype=torch.int32, ndim=1
    )
    exported = _required_native_op("bdpt_face_material_tensors")(
        material_eps_r,
        material_sigma_e,
        material_mu_r,
        face_material_id,
    )
    if not isinstance(exported, dict):
        raise TypeError("_channel_native.bdpt_face_material_tensors must return a dict")
    validate_cuda_tensor("eps_r", exported["eps_r"], dtype=torch.float32, ndim=1)
    validate_cuda_tensor("sigma_e", exported["sigma_e"], dtype=torch.float32, ndim=1)
    validate_cuda_tensor("mu_r", exported["mu_r"], dtype=torch.float32, ndim=1)
    validate_cuda_tensor("gain", exported["gain"], dtype=torch.float32, ndim=1)
    validate_cuda_tensor("valid", exported["valid"], dtype=torch.bool, ndim=1)
    return exported


def bdpt_face_material_tensors_from_host(
    material_eps_r: tuple[float, ...],
    material_sigma_e: tuple[float, ...],
    material_mu_r: tuple[float, ...],
    face_material_id: tuple[int, ...],
) -> dict[str, torch.Tensor]:
    if not material_eps_r:
        raise ValueError("material_eps_r must not be empty")
    if len(material_sigma_e) != len(material_eps_r):
        raise ValueError("material_sigma_e must match material_eps_r")
    if len(material_mu_r) != len(material_eps_r):
        raise ValueError("material_mu_r must match material_eps_r")
    if any(index < 0 or index >= len(material_eps_r) for index in face_material_id):
        raise ValueError("face_material_id entries must reference a material")
    exported = _required_native_op("bdpt_face_material_tensors_from_host")(
        material_eps_r,
        material_sigma_e,
        material_mu_r,
        face_material_id,
    )
    if not isinstance(exported, dict):
        raise TypeError(
            "_channel_native.bdpt_face_material_tensors_from_host must return a dict"
        )
    validate_cuda_tensor("eps_r", exported["eps_r"], dtype=torch.float32, ndim=1)
    validate_cuda_tensor("sigma_e", exported["sigma_e"], dtype=torch.float32, ndim=1)
    validate_cuda_tensor("mu_r", exported["mu_r"], dtype=torch.float32, ndim=1)
    validate_cuda_tensor("gain", exported["gain"], dtype=torch.float32, ndim=1)
    validate_cuda_tensor("valid", exported["valid"], dtype=torch.bool, ndim=1)
    return exported


_EM_LAYER_STACK_FIELDS = (
    "r_te_real",
    "r_te_imag",
    "r_tm_real",
    "r_tm_imag",
    "t_te_real",
    "t_te_imag",
    "t_tm_real",
    "t_tm_imag",
    "cap_R_te",
    "cap_R_tm",
    "cap_T_te",
    "cap_T_tm",
)


def em_layer_stack_eval(
    cos_theta: torch.Tensor,
    material_id: torch.Tensor,
    layer_offset: torch.Tensor,
    layer_count: torch.Tensor,
    layer_thickness_m: torch.Tensor,
    layer_eps_r: torch.Tensor,
    layer_sigma_e: torch.Tensor,
    layer_mu_r: torch.Tensor,
    *,
    frequency_hz: float,
) -> dict[str, torch.Tensor]:
    validate_cuda_tensor("cos_theta", cos_theta, dtype=torch.float32, ndim=1)
    validate_cuda_tensor("material_id", material_id, dtype=torch.int32, ndim=1)
    count = int(cos_theta.shape[0])
    if material_id.shape != (count,):
        raise ValueError("material_id must match cos_theta length")
    if material_id.get_device() != cos_theta.get_device():
        raise ValueError("material_id must share cos_theta device")
    _validate_layer_csr(
        layer_offset,
        layer_count,
        layer_thickness_m,
        layer_eps_r,
        layer_sigma_e,
        layer_mu_r,
        cos_theta.get_device(),
    )
    if frequency_hz <= 0.0:
        raise ValueError("frequency_hz must be positive")
    out = _required_native_op("em_layer_stack_eval")(
        cos_theta,
        material_id,
        layer_offset,
        layer_count,
        layer_thickness_m,
        layer_eps_r,
        layer_sigma_e,
        layer_mu_r,
        float(frequency_hz),
    )
    if not isinstance(out, dict):
        raise TypeError("_channel_native.em_layer_stack_eval must return a dict")
    if set(out) != set(_EM_LAYER_STACK_FIELDS):
        raise ValueError("em_layer_stack_eval returned unexpected fields")
    for name in _EM_LAYER_STACK_FIELDS:
        validate_cuda_tensor(name, out[name], dtype=torch.float32, ndim=1)
        if out[name].shape != (count,):
            raise ValueError(f"em_layer_stack_eval returned bad {name} shape")
    return out


def em_layer_stack_backward(
    cos_theta: torch.Tensor,
    material_id: torch.Tensor,
    layer_offset: torch.Tensor,
    layer_count: torch.Tensor,
    layer_thickness_m: torch.Tensor,
    layer_eps_r: torch.Tensor,
    layer_sigma_e: torch.Tensor,
    layer_mu_r: torch.Tensor,
    grad_outputs: tuple[torch.Tensor | None, ...],
    *,
    frequency_hz: float,
    need_cos_theta: bool,
    need_layers: bool,
    need_frequency: bool,
) -> dict[str, torch.Tensor]:
    if len(grad_outputs) != len(_EM_LAYER_STACK_FIELDS):
        raise ValueError(
            "grad_outputs must carry one cotangent slot per stack output"
        )
    out = _required_native_op("em_layer_stack_backward")(
        cos_theta,
        material_id,
        layer_offset,
        layer_count,
        layer_thickness_m,
        layer_eps_r,
        layer_sigma_e,
        layer_mu_r,
        float(frequency_hz),
        list(grad_outputs),
        bool(need_cos_theta),
        bool(need_layers),
        bool(need_frequency),
    )
    if not isinstance(out, dict):
        raise TypeError("_channel_native.em_layer_stack_backward must return a dict")
    return out


def em_layer_stack_jvp(
    cos_theta: torch.Tensor,
    material_id: torch.Tensor,
    layer_offset: torch.Tensor,
    layer_count: torch.Tensor,
    layer_thickness_m: torch.Tensor,
    layer_eps_r: torch.Tensor,
    layer_sigma_e: torch.Tensor,
    layer_mu_r: torch.Tensor,
    *,
    frequency_hz: float,
    tangent_cos_theta: torch.Tensor | None,
    tangent_layer_thickness: torch.Tensor | None,
    tangent_layer_eps_r: torch.Tensor | None,
    tangent_layer_sigma_e: torch.Tensor | None,
    tangent_frequency: float,
) -> dict[str, torch.Tensor]:
    out = _required_native_op("em_layer_stack_jvp")(
        cos_theta,
        material_id,
        layer_offset,
        layer_count,
        layer_thickness_m,
        layer_eps_r,
        layer_sigma_e,
        layer_mu_r,
        float(frequency_hz),
        tangent_cos_theta,
        tangent_layer_thickness,
        tangent_layer_eps_r,
        tangent_layer_sigma_e,
        float(tangent_frequency),
    )
    if not isinstance(out, dict):
        raise TypeError("_channel_native.em_layer_stack_jvp must return a dict")
    return out


def mc_face_material_tensors(
    material_eps_r: torch.Tensor,
    material_sigma_e: torch.Tensor,
    material_mu_r: torch.Tensor,
    face_material_id: torch.Tensor,
) -> dict[str, torch.Tensor]:
    validate_cuda_tensor("material_eps_r", material_eps_r, dtype=torch.float32, ndim=1)
    validate_cuda_tensor(
        "material_sigma_e", material_sigma_e, dtype=torch.float32, ndim=1
    )
    validate_cuda_tensor("material_mu_r", material_mu_r, dtype=torch.float32, ndim=1)
    validate_cuda_tensor(
        "face_material_id", face_material_id, dtype=torch.int32, ndim=1
    )
    if material_sigma_e.shape != material_eps_r.shape:
        raise ValueError("material_sigma_e must match material_eps_r shape")
    if material_mu_r.shape != material_eps_r.shape:
        raise ValueError("material_mu_r must match material_eps_r shape")

    native = native_extension()
    if native is None or not hasattr(native, "mc_face_material_tensors"):
        raise RuntimeError(
            "_channel_native.mc_face_material_tensors CUDA kernel is required"
        )
    exported = native.mc_face_material_tensors(
        material_eps_r,
        material_sigma_e,
        material_mu_r,
        face_material_id,
    )
    if not isinstance(exported, dict):
        raise TypeError("_channel_native.mc_face_material_tensors must return a dict")
    return exported


__all__ = [
    "bdpt_face_material_tensors",
    "bdpt_face_material_tensors_from_host",
    "em_layer_stack_backward",
    "em_layer_stack_eval",
    "em_layer_stack_jvp",
    "mc_face_material_tensors",
]
