"""DrJit-to-PyTorch tensor view conversion and reshape helpers.

Zero-copy via DLPack when possible; falls back to ``torch.as_tensor`` for
non-DrJit inputs.
"""

from __future__ import annotations

import math
from types import MappingProxyType

import drjit as dr
import torch
from witwin.channel import types as wt


FloatTensor = dr.tensor_t(wt.Float)
IntTensor = dr.tensor_t(wt.Int32)
BoolTensor = dr.tensor_t(wt.Bool)


def to_mapping_proxy(mapping):
    return MappingProxyType(dict(mapping or {}))


def shape_tuple(shape) -> tuple[int, ...]:
    return tuple(int(value) for value in shape)


def to_torch_view(
    value,
    *,
    detach: bool = False,
    dtype: torch.dtype | None = None,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Return a torch tensor sharing memory with ``value`` when possible."""
    if isinstance(value, torch.Tensor):
        tensor = value.detach() if detach else value
    elif hasattr(value, "__dlpack__"):
        source = dr.detach(value) if detach else value
        tensor = torch.utils.dlpack.from_dlpack(source)
    elif hasattr(value, "torch"):
        tensor = value.torch()
        if detach:
            tensor = tensor.detach()
    else:
        tensor = torch.as_tensor(value)
    if device is not None or dtype is not None:
        tensor = tensor.to(
            device=device if device is not None else tensor.device,
            dtype=dtype if dtype is not None else tensor.dtype,
        )
    return tensor


def reshape_or_broadcast(tensor: torch.Tensor, *, shape: tuple[int, ...], dtype) -> torch.Tensor:
    if shape_tuple(tensor.shape) == shape:
        return tensor.contiguous()
    if tensor.numel() == 1:
        return torch.full(shape, tensor.reshape(()).item(), dtype=dtype, device=tensor.device).contiguous()
    return tensor.reshape(shape).contiguous()


def to_typed_tensor(value, *, shape: tuple[int, ...], tensor_type, torch_dtype, detach: bool = False):
    source = dr.detach(value) if detach and (dr.is_array_v(type(value)) or dr.is_tensor_v(type(value))) else value
    if detach:
        tensor = to_torch_view(source, detach=True, dtype=torch_dtype).clone()
        return tensor_type(reshape_or_broadcast(tensor, shape=shape, dtype=torch_dtype))
    if dr.is_tensor_v(type(source)) and type(source) is tensor_type and shape_tuple(source.shape) == shape:
        return source
    try:
        return dr.reshape(tensor_type, source, shape)
    except (TypeError, RuntimeError):
        tensor = to_torch_view(source, dtype=torch_dtype)
        return tensor_type(reshape_or_broadcast(tensor, shape=shape, dtype=torch_dtype))


def to_float_tensor(value, *, shape: tuple[int, ...], detach: bool = False) -> FloatTensor:
    return to_typed_tensor(value, shape=shape, tensor_type=FloatTensor, torch_dtype=torch.float32, detach=detach)


def to_int_tensor(value, *, shape: tuple[int, ...], detach: bool = False) -> IntTensor:
    return to_typed_tensor(value, shape=shape, tensor_type=IntTensor, torch_dtype=torch.int32, detach=detach)


def to_bool_tensor(value, *, shape: tuple[int, ...], detach: bool = False) -> BoolTensor:
    return to_typed_tensor(value, shape=shape, tensor_type=BoolTensor, torch_dtype=torch.bool, detach=detach)


def to_vector_tensor(value, *, component_shape: tuple[int, ...]) -> FloatTensor:
    expected_shape = component_shape + (3,)
    if dr.is_tensor_v(type(value)) and type(value) is FloatTensor and shape_tuple(value.shape) == expected_shape:
        return value
    if all(hasattr(value, axis) for axis in ("x", "y", "z")):
        stacked = torch.stack(
            [to_torch_view(getattr(value, axis), dtype=torch.float32) for axis in ("x", "y", "z")],
            dim=-1,
        )
        return FloatTensor(reshape_or_broadcast(stacked, shape=expected_shape, dtype=torch.float32))
    try:
        return dr.reshape(FloatTensor, value, expected_shape)
    except (TypeError, RuntimeError):
        tensor = to_torch_view(value, dtype=torch.float32)
        return FloatTensor(reshape_or_broadcast(tensor, shape=expected_shape, dtype=torch.float32))


def to_complex_array(value, *, shape: tuple[int, ...]):
    """Convert a torch/drjit complex-like value to a drjit ``Complex2f`` array."""
    target_size = math.prod(shape) if shape else 1
    if not isinstance(value, torch.Tensor) and not dr.is_tensor_v(type(value)):
        real = getattr(value, "real", None)
        imag = getattr(value, "imag", None)
        if real is not None and imag is not None and int(dr.width(real)) == target_size:
            return value

    tensor = to_torch_view(value)
    if not torch.is_complex(tensor):
        if tensor.ndim == 0 or tensor.shape[-1] != 2:
            raise ValueError("Complex tensor payloads must provide a trailing real/imag axis of size 2.")
        tensor = torch.view_as_complex(tensor.to(dtype=torch.float32).contiguous())
    tensor = reshape_or_broadcast(tensor, shape=shape, dtype=torch.complex64)
    flat = tensor.reshape(-1).contiguous()
    return wt.Complex2f(wt.Float(flat.real), wt.Float(flat.imag))


__all__ = [
    "BoolTensor",
    "FloatTensor",
    "IntTensor",
    "reshape_or_broadcast",
    "shape_tuple",
    "to_bool_tensor",
    "to_complex_array",
    "to_float_tensor",
    "to_int_tensor",
    "to_mapping_proxy",
    "to_torch_view",
    "to_typed_tensor",
    "to_vector_tensor",
]
