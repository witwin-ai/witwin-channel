from __future__ import annotations

import drjit as dr
import torch
import witwin as wt


def _torch_tensor_to_drjit(tensor):
    return wt.Float(tensor)


def _drjit_value_to_torch(value):
    if hasattr(value, "torch"):
        return value.torch()
    return value


def _wrap_drjit_forward_output(value, has_tangent):
    import torch.autograd.forward_ad as fwAD

    if isinstance(value, tuple):
        return tuple(_wrap_drjit_forward_output(item, has_tangent) for item in value)

    primal = _drjit_value_to_torch(value)
    if not has_tangent:
        return primal

    dr.forward_to(value)
    tangent = _drjit_value_to_torch(dr.grad(value))
    return fwAD.make_dual(primal, tangent)


def drjit_forward_ad(func, *args, **kwargs):
    """Run a DrJit function on PyTorch tensors while preserving forward-mode tangents."""
    import torch.autograd.forward_ad as fwAD

    drjit_args = []
    has_tangent = False

    for arg in args:
        if isinstance(arg, torch.Tensor):
            primal, tangent = fwAD.unpack_dual(arg)
            drjit_arg = _torch_tensor_to_drjit(primal)
            if tangent is not None:
                dr.enable_grad(drjit_arg)
                dr.set_grad(drjit_arg, _torch_tensor_to_drjit(tangent))
                has_tangent = True
            drjit_args.append(drjit_arg)
        else:
            drjit_args.append(arg)

    result = func(*drjit_args, **kwargs)
    return _wrap_drjit_forward_output(result, has_tangent)


def wrap_drjit_forward_ad(func):
    """Decorator form of :func:`drjit_forward_ad`."""

    def wrapped(*args, **kwargs):
        return drjit_forward_ad(func, *args, **kwargs)

    return wrapped


def _is_torch_tensor(value):
    return isinstance(value, torch.Tensor)


def torch_to_drjit(value):
    """Keep PyTorch tensors differentiable when passed into wrapped DrJit ops."""
    if _is_torch_tensor(value):
        return value
    return wt.Float(value)


def drjit_to_torch(value):
    """Convert DrJit values to PyTorch tensors when available."""
    if isinstance(value, tuple):
        return tuple(drjit_to_torch(item) for item in value)
    if _is_torch_tensor(value):
        return value
    if hasattr(value, "torch"):
        return value.torch()
    return value


def to_drjit(value):
    return torch_to_drjit(value)


def to_torch(value):
    return drjit_to_torch(value)


def drjit_to_torch_view(value, *, detach=False, dtype=None, device=None):
    """Convert DrJit arrays to torch tensors while reusing device memory when possible."""
    if _is_torch_tensor(value):
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


def stable_torch_argsort(values):
    try:
        return torch.argsort(values, stable=True)
    except TypeError:
        return torch.argsort(values)


def torch_lexsort(keys):
    if len(keys) == 0:
        return torch.zeros(0, dtype=torch.int64)
    order = torch.arange(keys[0].shape[0], device=keys[0].device, dtype=torch.int64)
    for key in reversed(keys):
        order = order.index_select(0, stable_torch_argsort(key.index_select(0, order)))
    return order


def wrap_drjit(func):
    """Wrap a DrJit function so it can be called directly on PyTorch tensors."""
    if hasattr(dr, "wrap"):
        return dr.wrap(source="torch", target="drjit")(func)
    return func


@wrap_drjit
def _wrapped_mul(a: wt.Float, b: wt.Float) -> wt.Float:
    return a * b


@wrap_drjit
def _wrapped_add(a: wt.Float, b: wt.Float) -> wt.Float:
    return a + b


@wrap_drjit
def _wrapped_sqrt(x: wt.Float) -> wt.Float:
    return dr.sqrt(x)


@wrap_drjit
def _wrapped_norm(x: wt.Float, y: wt.Float, z: wt.Float) -> wt.Float:
    return dr.sqrt(x * x + y * y + z * z)


def drjit_mul(a, b):
    if _is_torch_tensor(a) or _is_torch_tensor(b):
        return _wrapped_mul(torch_to_drjit(a), torch_to_drjit(b))
    return a * b


def drjit_add(a, b):
    if _is_torch_tensor(a) or _is_torch_tensor(b):
        return _wrapped_add(torch_to_drjit(a), torch_to_drjit(b))
    return a + b


def drjit_sqrt(x):
    if _is_torch_tensor(x):
        return _wrapped_sqrt(torch_to_drjit(x))
    return dr.sqrt(x)


def drjit_norm(x, y, z):
    if _is_torch_tensor(x) or _is_torch_tensor(y) or _is_torch_tensor(z):
        return _wrapped_norm(torch_to_drjit(x), torch_to_drjit(y), torch_to_drjit(z))
    return dr.sqrt(x * x + y * y + z * z)


__all__ = [
    "drjit_add",
    "drjit_forward_ad",
    "drjit_mul",
    "drjit_norm",
    "drjit_sqrt",
    "drjit_to_torch",
    "drjit_to_torch_view",
    "stable_torch_argsort",
    "to_drjit",
    "to_torch",
    "torch_lexsort",
    "torch_to_drjit",
    "wrap_drjit",
    "wrap_drjit_forward_ad",
]
