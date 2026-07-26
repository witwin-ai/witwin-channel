"""Native endpoint-row gather for frozen LoS consumer topology."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from witwin.channel.runtime import torch_compat
from witwin.channel.runtime.autograd_contracts import (
    _ad_geometry_live,
    _ad_native_tangent_or_none,
    _ad_native_tensor,
)
from witwin.channel.runtime.symbols import required_symbol as _required_native_op

from .contracts import EndpointBatch, PropagationTopology


_ROW_FIELDS = (
    "source",
    "target",
    "tx_power",
    "tx_polarization",
    "rx_polarization",
)
_OUTPUT_FIELDS = (*_ROW_FIELDS, "pair_index", "pair_offsets")
_ENDPOINT_GRAD_FIELDS = (
    "source_positions",
    "sink_positions",
    "source_powers",
    "source_polarizations",
    "sink_polarizations",
)


@dataclass(frozen=True, slots=True)
class FixedLoSRows:
    """Exact frozen LoS rows ready for the RayD-owned free-space field family."""

    source: torch.Tensor
    target: torch.Tensor
    tx_power: torch.Tensor
    tx_polarization: torch.Tensor
    rx_polarization: torch.Tensor
    pair_index: torch.Tensor
    pair_offsets: torch.Tensor
    validation_d2h_copies: int
    validation_d2h_bytes: int
    validation_synchronizations: int

    @property
    def row_count(self) -> int:
        return int(self.pair_index.shape[0])


class _FixedLoSGatherFunction(torch.autograd.Function):
    @staticmethod
    def forward(*inputs):
        raw = _required_native_op("consumer_fixed_los_gather")(*inputs)
        if not isinstance(raw, dict) or set(raw) != set(_OUTPUT_FIELDS):
            raise TypeError("native fixed LoS gather returned bad fields")
        return tuple(raw[name] for name in _OUTPUT_FIELDS)

    @staticmethod
    def setup_context(ctx, inputs, output):
        ctx.set_materialize_grads(False)
        source_index, sink_index = (
            torch.autograd.forward_ad.unpack_dual(value).primal
            for value in inputs[:2]
        )
        ctx.source_count = int(inputs[6].shape[0])
        ctx.sink_count = int(inputs[7].shape[0])
        ctx.save_for_backward(source_index, sink_index)
        ctx.save_for_forward(source_index, sink_index)
        ctx.mark_non_differentiable(*output[5:])

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, *grad_outputs):
        endpoint_grads = grad_outputs[:5]
        if all(value is None for value in endpoint_grads):
            return (None,) * 13
        source_index, sink_index = ctx.saved_tensors
        raw = _required_native_op("consumer_fixed_los_gather_backward")(
            source_index,
            sink_index,
            *endpoint_grads,
            ctx.source_count,
            ctx.sink_count,
        )
        if not isinstance(raw, dict) or set(raw) != set(_ENDPOINT_GRAD_FIELDS):
            raise TypeError("native fixed LoS gather backward returned bad fields")
        return (
            *(None for _ in range(6)),
            *(
                raw[name] if ctx.needs_input_grad[index] else None
                for index, name in enumerate(_ENDPOINT_GRAD_FIELDS, start=6)
            ),
            None,
            None,
        )

    @staticmethod
    def jvp(ctx, *tangents):
        endpoint_tangents = tuple(
            _ad_native_tangent_or_none(value) for value in tangents[6:11]
        )
        if all(value is None for value in endpoint_tangents):
            return (None,) * len(_OUTPUT_FIELDS)
        source_index, sink_index = (
            _ad_native_tensor(value) for value in ctx.saved_tensors
        )
        with torch_compat.disable_functorch():
            raw = _required_native_op("consumer_fixed_los_gather_jvp")(
                source_index,
                sink_index,
                *endpoint_tangents,
                ctx.source_count,
                ctx.sink_count,
            )
        if not isinstance(raw, dict) or set(raw) != set(_ROW_FIELDS):
            raise TypeError("native fixed LoS gather jvp returned bad fields")
        return (*(raw[name] for name in _ROW_FIELDS), None, None)


def fixed_los_gather(
    topology: PropagationTopology,
    sources: EndpointBatch,
    sinks: EndpointBatch,
) -> FixedLoSRows:
    """Validate and gather frozen LoS rows without Python/Torch indexing."""

    if not isinstance(topology, PropagationTopology):
        raise TypeError("topology must be a PropagationTopology")
    if not isinstance(sources, EndpointBatch) or not isinstance(sinks, EndpointBatch):
        raise TypeError("sources and sinks must be EndpointBatch instances")
    if sources.powers_w is None:
        raise ValueError("sources.powers_w is required")
    if sinks.powers_w is not None:
        raise ValueError("sinks.powers_w must be absent")
    if topology.device != sources.device or topology.device != sinks.device:
        raise ValueError("topology and endpoint tensors must share one CUDA device")

    values = _FixedLoSGatherFunction.apply(
        topology.source_index,
        topology.sink_index,
        topology.source_id,
        topology.sink_id,
        topology.depth,
        topology.component_id,
        sources.positions_m,
        sinks.positions_m,
        sources.powers_w,
        sources.polarizations,
        sinks.polarizations,
        sources.stable_ids,
        sinks.stable_ids,
    )
    rows = int(topology.source_index.shape[0])
    return FixedLoSRows(
        **dict(zip(_OUTPUT_FIELDS, values, strict=True)),
        validation_d2h_copies=1 if rows else 0,
        validation_d2h_bytes=4 if rows else 0,
        validation_synchronizations=1 if rows else 0,
    )


def fixed_los_geometry_live(rows: FixedLoSRows) -> bool:
    """ADR-038 liveness for the raw frozen line-of-sight route.

    A zero-interaction row is a function of its two gathered endpoints alone, so
    this is the complete liveness question for that route. It is answered here,
    once, from the gathered rows, above any frequency-column loop that replays
    them.
    """

    return _ad_geometry_live(rows.source, rows.target)


def require_fixed_los_geometry_live(rows: FixedLoSRows, decided: bool) -> None:
    """Fail loudly if one column disagrees with the hoisted decision.

    The field facade keeps deciding liveness for itself - that is its ADR-038
    contract - and this makes "every column decides the same thing" a checked
    invariant instead of an assumption, before the operator runs.
    """

    if fixed_los_geometry_live(rows) != decided:
        raise RuntimeError(
            "fixed LoS geometry liveness disagrees with the decision taken "
            f"above the frequency-column loop (decided {decided}); every "
            "column must answer the same ADR-038 question"
        )


__all__ = [
    "FixedLoSRows",
    "fixed_los_gather",
    "fixed_los_geometry_live",
    "require_fixed_los_geometry_live",
]
