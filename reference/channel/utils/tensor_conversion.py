from __future__ import annotations

import math
from types import MappingProxyType

import drjit as dr
import torch
import witwin as wt

from .torch_bridge import drjit_to_torch_view


FloatTensor = dr.tensor_t(wt.Float)
IntTensor = dr.tensor_t(wt.Int32)
BoolTensor = dr.tensor_t(wt.Bool)


def to_mapping_proxy(mapping):
    return MappingProxyType(dict(mapping or {}))


def shape_tuple(shape) -> tuple[int, ...]:
    return tuple(int(value) for value in shape)


def reshape_or_broadcast_torch_tensor(
    tensor: torch.Tensor,
    *,
    shape: tuple[int, ...],
    dtype,
) -> torch.Tensor:
    if shape_tuple(tensor.shape) != shape:
        if tensor.numel() == 1:
            tensor = torch.full(
                shape,
                tensor.reshape(()).item(),
                dtype=dtype,
                device=tensor.device,
            )
        else:
            tensor = tensor.reshape(shape)
    return tensor.contiguous()


def to_typed_tensor(value, *, shape: tuple[int, ...], tensor_type, torch_dtype):
    if dr.is_tensor_v(type(value)) and type(value) is tensor_type and shape_tuple(value.shape) == shape:
        return value
    try:
        return dr.reshape(tensor_type, value, shape)
    except Exception:
        tensor = drjit_to_torch_view(value, dtype=torch_dtype)
        tensor = reshape_or_broadcast_torch_tensor(tensor, shape=shape, dtype=torch_dtype)
        return tensor_type(tensor)


def to_float_tensor(value, *, shape: tuple[int, ...]):
    return to_typed_tensor(
        value,
        shape=shape,
        tensor_type=FloatTensor,
        torch_dtype=torch.float32,
    )


def to_int_tensor(value, *, shape: tuple[int, ...]):
    return to_typed_tensor(
        value,
        shape=shape,
        tensor_type=IntTensor,
        torch_dtype=torch.int32,
    )


def to_bool_tensor(value, *, shape: tuple[int, ...]):
    return to_typed_tensor(
        value,
        shape=shape,
        tensor_type=BoolTensor,
        torch_dtype=torch.bool,
    )


def to_vector_tensor(value, *, component_shape: tuple[int, ...]):
    expected_shape = component_shape + (3,)
    if dr.is_tensor_v(type(value)) and type(value) is FloatTensor and shape_tuple(value.shape) == expected_shape:
        return value
    try:
        x, y, z = value.x, value.y, value.z
    except Exception:
        pass
    else:
        tensor = torch.stack(
            [
                drjit_to_torch_view(x, dtype=torch.float32),
                drjit_to_torch_view(y, dtype=torch.float32),
                drjit_to_torch_view(z, dtype=torch.float32),
            ],
            dim=-1,
        )
        tensor = reshape_or_broadcast_torch_tensor(
            tensor,
            shape=expected_shape,
            dtype=torch.float32,
        )
        return FloatTensor(tensor)
    try:
        return dr.reshape(FloatTensor, value, expected_shape)
    except Exception:
        tensor = drjit_to_torch_view(value, dtype=torch.float32)
        tensor = reshape_or_broadcast_torch_tensor(
            tensor,
            shape=expected_shape,
            dtype=torch.float32,
        )
        return FloatTensor(tensor)


def to_complex_array(value, *, shape: tuple[int, ...]):
    target_size = math.prod(shape) if len(shape) > 0 else 1
    if not isinstance(value, torch.Tensor) and not dr.is_tensor_v(type(value)):
        try:
            real, imag = value.real, value.imag
        except Exception:
            pass
        else:
            if int(dr.width(real)) == target_size:
                return value

    tensor = drjit_to_torch_view(value)
    if not torch.is_complex(tensor):
        if tensor.ndim == 0 or tensor.shape[-1] != 2:
            raise ValueError("Complex tensor payloads must provide a trailing real/imag axis of size 2.")
        tensor = torch.view_as_complex(tensor.to(dtype=torch.float32).contiguous())
    tensor = reshape_or_broadcast_torch_tensor(
        tensor,
        shape=shape,
        dtype=torch.complex64,
    )
    flat = tensor.reshape(-1).contiguous()
    return wt.Complex2f(wt.Float(flat.real), wt.Float(flat.imag))


__all__ = [
    "BoolTensor",
    "FloatTensor",
    "IntTensor",
    "reshape_or_broadcast_torch_tensor",
    "shape_tuple",
    "to_bool_tensor",
    "to_complex_array",
    "to_float_tensor",
    "to_int_tensor",
    "to_mapping_proxy",
    "to_typed_tensor",
    "to_vector_tensor",
]
