"""Single owner for supported PyTorch private runtime APIs."""

from __future__ import annotations

from contextlib import AbstractContextManager
from typing import Protocol, cast

import torch


class _Interpreter(Protocol):
    def key(self) -> object: ...


def is_transform_wrapped_tensor(value: torch.Tensor) -> bool:
    """Return whether ``value`` still carries a functorch transform wrapper."""

    functorch = torch._C._functorch
    return bool(
        functorch.is_functorch_wrapped_tensor(value)
        or functorch.is_gradtrackingtensor(value)
    )


def transform_level(value: torch.Tensor) -> int:
    """Return the functorch transform level, or ``-1`` for a plain tensor."""

    return int(torch._C._functorch.maybe_get_level(value))


def interpreter_stack() -> tuple[_Interpreter, ...]:
    """Return the active functorch interpreter stack in outer-to-inner order."""

    stack = torch._C._functorch.get_interpreter_stack()
    return () if stack is None else tuple(stack)


def is_jvp_transform(interpreter: _Interpreter) -> bool:
    """Return whether an interpreter stack entry is a JVP transform."""

    return bool(interpreter.key() == torch._C._functorch.TransformType.Jvp)


def unwrap_transform_tensor(value: torch.Tensor) -> torch.Tensor:
    """Remove one functorch wrapper without copying tensor storage."""

    return torch._C._functorch.get_unwrapped(value)


def disable_functorch() -> AbstractContextManager[object]:
    """Disable functorch dispatch inside native/custom-AD bridge code."""

    return cast(AbstractContextManager[object], torch._C._DisableFuncTorch())


def uses_cxx11_abi() -> bool:
    """Return the CXX11 ABI flag compiled into the active PyTorch runtime."""

    return bool(torch._C._GLIBCXX_USE_CXX11_ABI)


__all__ = [
    "disable_functorch",
    "interpreter_stack",
    "is_jvp_transform",
    "is_transform_wrapped_tensor",
    "transform_level",
    "unwrap_transform_tensor",
    "uses_cxx11_abi",
]
