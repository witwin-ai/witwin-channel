# Copyright Xingyu Chen.
# Declarative native-operation specs that drive facade validation, dispatch and autograd plumbing.

"""Declarative native-operation specs: ABI argument lists, checks, dispatch and AD scaffolding."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import torch

from witwin.channel.runtime import (
    _ad_checked_tangent,
    _ad_geometry_tangent,
    _ad_native_tangent_or_none,
    _ad_reject_fixed_inputs,
    _ad_reject_fixed_tangents,
    validate_cuda_tensor,
)

__all__ = [
    "AdInputs",
    "Arg",
    "NativeOp",
    "NativeOps",
    "flag",
    "frequency_meta",
    "integer",
    "real",
    "save",
    "tensor",
    "opaque",
]

Grad = Literal["none", "diff", "fixed"]
Resolver = Callable[[str], Callable[..., Any]]


@dataclass(frozen=True, slots=True)
class Arg:
    """One ABI argument: its name, the facade check it carries, and its AD role.

    ``dtype``/``ndim``/``trailing``/``contiguous`` describe the CUDA tensor check the facade runs
    (no check when ``dtype`` is None). ``convert`` is applied to the value at dispatch (``float``,
    ``int``, ``bool``). ``grad`` is the autograd role: ``"diff"`` inputs are differentiated,
    ``"fixed"`` inputs refuse a requested gradient, ``"none"`` inputs (ids, flags, scalars) are
    never differentiated. ``tangent`` says whether a fixed input also refuses a forward tangent.
    """

    name: str
    dtype: torch.dtype | None = None
    ndim: int | None = None
    trailing: tuple[int, ...] = ()
    contiguous: bool = True
    convert: Callable[[Any], Any] | None = None
    grad: Grad = "none"
    tangent: bool = True
    abi: str | None = None

    @property
    def abi_name(self) -> str:
        """The C++ parameter name, when it differs from the facade name used in messages."""

        return self.name if self.abi is None else self.abi

    def check(self, value: Any) -> None:
        """Run the declared CUDA tensor check (dtype, device, rank, trailing shape, layout)."""

        if self.dtype is None or self.ndim is None:
            return
        validate_cuda_tensor(
            self.name, value, dtype=self.dtype, ndim=self.ndim,
            trailing_shape=self.trailing, require_contiguous=self.contiguous,
        )

    def renamed(self, name: str) -> Arg:
        """The same declaration under another ABI name (for repeated slots)."""

        return Arg(
            name, self.dtype, self.ndim, self.trailing, self.contiguous, self.convert, self.grad,
            self.tangent,
        )

    def with_abi(self, abi: str) -> Arg:
        """The same declaration bound to a differently named C++ parameter."""

        return Arg(
            self.name, self.dtype, self.ndim, self.trailing, self.contiguous, self.convert,
            self.grad, self.tangent, abi,
        )


def tensor(
    name: str, dtype: torch.dtype | None = None, ndim: int | None = None,
    trailing: tuple[int, ...] = (), *, contiguous: bool = True, grad: Grad = "none",
    tangent: bool = True, abi: str | None = None,
) -> Arg:
    """A tensor argument; ``dtype=None`` forwards it unchecked (optional or opaque tensors)."""

    return Arg(name, dtype, ndim, trailing, contiguous, None, grad, tangent, abi)


def real(name: str, *, grad: Grad = "none") -> Arg:
    """A host scalar forwarded as ``float``."""

    return Arg(name, convert=float, grad=grad)


def integer(name: str) -> Arg:
    """A host scalar forwarded as ``int``."""

    return Arg(name, convert=int)


def flag(name: str) -> Arg:
    """A host flag forwarded as ``bool``."""

    return Arg(name, convert=bool)


def opaque(
    name: str, *, convert: Callable[[Any], Any] | None = None, grad: Grad = "none",
    tangent: bool = True, abi: str | None = None,
) -> Arg:
    """An unchecked argument, forwarded unchanged or through ``convert``."""

    return Arg(name, convert=convert, grad=grad, tangent=tangent, abi=abi)


ArgGroup = Arg | Sequence["ArgGroup"]


def _flatten(args: Iterable[ArgGroup]) -> tuple[Arg, ...]:
    """Flatten nested argument groups into one ABI-ordered tuple."""

    flat: list[Arg] = []
    for item in args:
        if isinstance(item, Arg):
            flat.append(item)
        else:
            flat.extend(_flatten(item))
    return tuple(flat)


def _index(args: tuple[Arg, ...], owner: str) -> dict[str, int]:
    index: dict[str, int] = {}
    for position, arg in enumerate(args):
        if arg.name in index:
            raise ValueError(f"{owner} declares argument {arg.name!r} twice")
        index[arg.name] = position
    return index


class NativeOp:
    """One registered ``_channel`` symbol and its argument list in ABI order.

    The declaration order is the ABI order and the order :meth:`check` validates in.
    ``ci/check_native_bindings.py`` checks the names and count against the C++ entry point.
    """

    __slots__ = ("_index", "_resolve", "abi_names", "args", "names", "symbol")

    def __init__(self, resolve: Resolver, symbol: str, args: tuple[Arg, ...]) -> None:
        self._resolve = resolve
        self.symbol = symbol
        self.args = args
        self.names = tuple(arg.name for arg in args)
        self.abi_names = tuple(arg.abi_name for arg in args)
        self._index = _index(args, symbol)

    def __repr__(self) -> str:
        return f"NativeOp({self.symbol!r}, {len(self.args)} args)"

    def arg(self, name: str) -> Arg:
        return self.args[self._index[name]]

    def check(self, values: Sequence[object], *names: str) -> None:
        """Validate ``values`` (one per argument) in declaration order, or ``names`` in order."""

        if names:
            for name in names:
                position = self._index[name]
                self.args[position].check(values[position])
            return
        for arg, item in zip(self.args, values, strict=True):
            arg.check(item)

    def checks(self, mapping: Mapping[str, object], *names: str) -> None:
        """Validate the values of ``mapping`` in declaration order, or ``names`` in order."""

        for name in names or tuple(name for name in self.names if name in mapping):
            self.args[self._index[name]].check(mapping[name])

    def __call__(self, *values: object) -> Any:
        """Resolve the symbol through the owner module's resolver and dispatch in ABI order.

        The symbol resolves before any argument is converted, as in a hand-written call.
        """

        function = self._resolve(self.symbol)
        converted = tuple(
            item if arg.convert is None else arg.convert(item)
            for arg, item in zip(self.args, values)
        )
        # Values past the declaration are forwarded as given; the extension rejects a bad arity.
        return function(*converted, *values[len(converted):])


class NativeOps:
    """The ``NativeOp`` declarations of one facade module and the resolver they dispatch through.

    The resolver is the module's own symbol lookup, read at call time, so a module-level
    resolver replaced in a test reaches every declared operation of that module.
    """

    __slots__ = ("_resolve",)

    def __init__(self, resolve: Resolver) -> None:
        self._resolve = resolve

    def op(self, symbol: str, *args: ArgGroup) -> NativeOp:
        return NativeOp(self._resolve, symbol, _flatten(args))


class AdInputs:
    """The ``apply`` argument list of one autograd Function, by name.

    It replaces positional ``inputs[i]`` / ``needs_input_grad[i]`` / ``(None,) * N`` handling:
    gradients and tangents are addressed by argument name, and the fixed inputs come from the
    declarations, in declaration order.
    """

    __slots__ = ("_fixed_grads", "_fixed_tangents", "_index", "args", "names", "none", "op_name")

    def __init__(self, op_name: str, *args: ArgGroup) -> None:
        self.op_name = op_name
        self.args = _flatten(args)
        self.names = tuple(arg.name for arg in self.args)
        self._index = _index(self.args, op_name)
        self.none: tuple[None, ...] = (None,) * len(self.args)
        self._fixed_grads = tuple(
            (position, arg.name) for position, arg in enumerate(self.args) if arg.grad == "fixed"
        )
        self._fixed_tangents = tuple(
            (position, arg.name)
            for position, arg in enumerate(self.args)
            if arg.grad == "fixed" and arg.tangent
        )

    def __len__(self) -> int:
        return len(self.args)

    def index(self, name: str) -> int:
        return self._index[name]

    def get(self, inputs: Sequence[Any], name: str) -> Any:
        return inputs[self._index[name]]

    def unpack(self, inputs: Sequence[Any]) -> dict[str, Any]:
        """Name every ``apply`` input."""

        return dict(zip(self.names, inputs, strict=True))

    def primals(self, inputs: Sequence[Any], *names: str) -> tuple[Any, ...]:
        """The forward-AD primals of the named inputs, in the order named."""

        return tuple(
            torch.autograd.forward_ad.unpack_dual(inputs[self._index[name]]).primal
            for name in names
        )

    def need(self, ctx: Any, name: str) -> bool:
        return bool(ctx.needs_input_grad[self._index[name]])

    def needs(self, ctx: Any, *names: str) -> bool:
        """True when any named input needs a gradient."""

        return any(bool(ctx.needs_input_grad[self._index[name]]) for name in names)

    def reject_fixed_grads(self, ctx: Any) -> None:
        """Refuse a requested gradient on any fixed input (first fixed input in order)."""

        _ad_reject_fixed_inputs(self.op_name, ctx.needs_input_grad, self._fixed_grads)

    def reject_fixed_tangents(self, tangents: Sequence[Any]) -> None:
        """Refuse a live tangent on any fixed input (first fixed input in order)."""

        _ad_reject_fixed_tangents(
            self.op_name,
            tuple((tangents[position], name) for position, name in self._fixed_tangents),
        )

    def tangent(self, tangents: Sequence[Any], name: str) -> Any:
        return tangents[self._index[name]]

    def geometry_tangent(self, tangents: Sequence[Any], name: str, primal: torch.Tensor) -> Any:
        """The unwrapped tangent of ``name`` checked against its primal shape and dtype."""

        return _ad_geometry_tangent(
            f"{self.op_name} tangent_{name}", tangents[self._index[name]], primal
        )

    def checked_tangent(
        self, tangents: Sequence[Any], name: str, shape: tuple[int, ...], label: str | None = None,
    ) -> torch.Tensor | None:
        """The unwrapped float32 tangent of ``name`` checked against ``shape``."""

        return _ad_checked_tangent(
            f"{self.op_name} tangent_{label or name}",
            _ad_native_tangent_or_none(tangents[self._index[name]]),
            shape,
        )

    def grads(self, ctx: Any, **values: Any) -> tuple[Any, ...]:
        """One gradient slot per input: each named value where that input needs it, else None."""

        result: list[Any] = [None] * len(self.args)
        needs = ctx.needs_input_grad
        for name, item in values.items():
            position = self._index[name]
            if needs[position]:
                result[position] = item
        return tuple(result)

    def output_grads(
        self, ctx: Any, out: Mapping[str, Any], keys: Mapping[str, str], **values: Any,
    ) -> tuple[Any, ...]:
        """Gradient slots read as ``out[keys[name]]`` only for the inputs that need them.

        ``values`` adds slots computed by the caller (also masked by ``needs_input_grad``).
        """

        needs = ctx.needs_input_grad
        read = {name: out[key] for name, key in keys.items() if needs[self._index[name]]}
        return self.grads(ctx, **read, **values)

    def place(self, **values: Any) -> tuple[Any, ...]:
        """One slot per input holding the named values unmasked, None elsewhere."""

        result: list[Any] = [None] * len(self.args)
        for name, item in values.items():
            result[self._index[name]] = item
        return tuple(result)


def save(ctx: Any, *tensors: Any) -> None:
    """Save the same tensors for the reverse and the forward-mode companion."""

    ctx.save_for_backward(*tensors)
    ctx.save_for_forward(*tensors)


def frequency_meta(frequency: object) -> tuple[torch.dtype, torch.device] | None:
    """The dtype/device a frequency gradient is returned in (None for a host float)."""

    if isinstance(frequency, torch.Tensor):
        return (frequency.dtype, frequency.device)
    return None
