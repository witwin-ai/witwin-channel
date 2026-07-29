# Copyright Xingyu Chen.
# Native Monte Carlo kernel facades.

"""Native Monte Carlo kernel facades.

Thin facades over the ``_channel`` Monte Carlo ABI for both Monte Carlo
solvers: MC Basic (``mc_*``) and BDPT (``bdpt_*``). Every entry validates its
contract, requests the required native symbol through
:mod:`witwin.channel.runtime`, dispatches the native operation, and converts
its result into a named typed contract.

The two prefixes are kept side by side and are deliberately not merged. Nine
operations share a name up to their ``mc_``/``bdpt_`` prefix and have already
drifted apart, so collapsing them is a numerical question that needs its own
change, not a layout move.

basic capacity
--------------
The ADR-032 MC Basic capacity-failure sanitizer and its native
backward/JVP companions: one native identity/zero transaction boundary that
makes every component map inert after a failed transaction.

basic maps
----------
The MC Basic component-map owners: buffer allocation, per-component stores,
LoS grid maps, the slab reflection and UTD diffraction tape accumulators, the
finalize reduction, and the ``torch.autograd.Function`` companions that
dispatch their registered native backward/JVP entries.

basic sampling
--------------
MC Basic launch inputs, the diffraction edge discovery entries, and the
diffraction state pack/incident-direction primitives. ``mc_sample_directions``
is re-exported from the public :mod:`witwin.channel.propagation.topology` seam
rather than redefined; the topology stage stays its single owner.

basic transmission
------------------
The ADR-027 MC straight-penetration wall-product estimator: the fixed-capacity
:class:`McTransmissionWallProduct` contract, its primal facade, and the
``torch.autograd.Function`` that dispatches the registered native VJP/JVP
companions.

bdpt maps
---------
The BDPT component and point-component matrix owners: buffer allocation,
column and matrix stores, the grid expansion, and the finalize reductions.

bdpt paths
----------
The BDPT subpath and connection-sample owners: launch state, endpoint and
reflected/transmitted light subpath states, subpath intersection inputs,
connection-sample export, filtering, concatenation, counting, compaction, the
MIS/PDF measure entries, and the connection variance reduction.

bdpt sampling
-------------
BDPT direction sampling, reflection launch inputs, and vector packing.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch

from witwin.channel.materials import validate_layer_csr as _validate_layer_csr
from witwin.channel.propagation.geometry import (
    BDPT_INTERSECTION_FIELDS as _BDPT_INTERSECTION_FIELDS,
)
from witwin.channel.propagation.topology import (
    mc_sample_directions,  # noqa: F401
    path_los_export,
)
from witwin.channel.runtime import (
    CapacityFailureBit,
    CapacityFailureState,
    _ad_checked_tangent,
    _ad_first_order_only,
    _ad_frequency_grad,
    _ad_frequency_tangent,
    _ad_frequency_value,
    _ad_geometry_tangent,
    _ad_native_tangent_or_none,
    _ad_native_tensor,
    _ad_reject_fixed_inputs,
    _ad_reject_fixed_tangents,
    bdpt_zero_matrix,  # noqa: F401
    disable_functorch,
    mc_pack_vec3,  # noqa: F401
    mc_receiver_grid_points,  # noqa: F401
    mc_transmitter_tensors,  # noqa: F401
    require_capacity_failure_state,
    required_symbol,
    required_symbol as _required_native_op,
    validate_cuda_tensor,
)


# -------------------------------------------------------------------------
# basic capacity
# -------------------------------------------------------------------------
_MC_COMPONENT_MAP_FIELDS = (
    "los",
    "reflection",
    "diffraction",
    "transmission",
    "scattering",
)


def _validate_capacity_component_maps(
    maps: tuple[torch.Tensor, ...],
) -> torch.Tensor:
    if len(maps) != len(_MC_COMPONENT_MAP_FIELDS):
        raise ValueError("MC capacity sanitizer requires five component maps")
    reference = maps[0]
    for name, value in zip(_MC_COMPONENT_MAP_FIELDS, maps, strict=True):
        validate_cuda_tensor(name, value, dtype=torch.float32, ndim=3)
        if value.shape != reference.shape:
            raise ValueError(f"{name} must match los shape")
        if value.device != reference.device:
            raise ValueError(f"{name} must share the los device")
    return reference


def _capacity_component_maps_result(
    exported: object,
    *,
    reference: torch.Tensor,
) -> tuple[torch.Tensor, ...]:
    if not isinstance(exported, dict) or set(exported) != set(_MC_COMPONENT_MAP_FIELDS):
        raise TypeError("native MC capacity map sanitizer returned bad fields")
    values = tuple(exported[name] for name in _MC_COMPONENT_MAP_FIELDS)
    for name, value in zip(_MC_COMPONENT_MAP_FIELDS, values, strict=True):
        validate_cuda_tensor(name, value, dtype=torch.float32, ndim=3)
        if value.shape != reference.shape or value.device != reference.device:
            raise ValueError(f"native MC capacity sanitizer returned bad {name}")
    return values


def _mc_capacity_failure_component_maps_sanitize_native(
    failure_state_bits: torch.Tensor,
    *maps: torch.Tensor,
) -> tuple[torch.Tensor, ...]:
    reference = _validate_capacity_component_maps(maps)
    exported = _required_native_op("mc_capacity_failure_component_maps_sanitize")(
        failure_state_bits, *maps
    )
    return _capacity_component_maps_result(exported, reference=reference)


def _mc_capacity_failure_component_maps_sanitize_backward_native(
    failure_state_bits: torch.Tensor,
    reference: torch.Tensor,
    *gradients: torch.Tensor | None,
) -> tuple[torch.Tensor, ...]:
    exported = _required_native_op(
        "mc_capacity_failure_component_maps_sanitize_backward"
    )(failure_state_bits, reference, *gradients)
    return _capacity_component_maps_result(exported, reference=reference)


def _mc_capacity_failure_component_maps_sanitize_jvp_native(
    failure_state_bits: torch.Tensor,
    reference: torch.Tensor,
    *tangents: torch.Tensor | None,
) -> tuple[torch.Tensor, ...]:
    exported = _required_native_op("mc_capacity_failure_component_maps_sanitize_jvp")(
        failure_state_bits, reference, *tangents
    )
    return _capacity_component_maps_result(exported, reference=reference)


class _McCapacityFailureComponentMapsSanitizeFunction(torch.autograd.Function):
    """Native identity/zero transaction boundary for all MC Basic maps."""

    @staticmethod
    def forward(failure_state_bits, *maps):
        return _mc_capacity_failure_component_maps_sanitize_native(
            failure_state_bits, *maps
        )

    @staticmethod
    def setup_context(ctx, inputs, output):
        ctx.set_materialize_grads(False)
        failure_state_bits = torch.autograd.forward_ad.unpack_dual(inputs[0]).primal
        reference = torch.autograd.forward_ad.unpack_dual(output[0]).primal
        ctx.save_for_backward(failure_state_bits, reference)
        ctx.save_for_forward(failure_state_bits, reference)

    @staticmethod
    @_ad_first_order_only
    def backward(ctx, *gradients):
        if not any(ctx.needs_input_grad[1:]):
            return (None,) * 6
        failure_state_bits, reference = ctx.saved_tensors
        values = _mc_capacity_failure_component_maps_sanitize_backward_native(
            failure_state_bits, reference, *gradients
        )
        return (
            None,
            *(
                value if ctx.needs_input_grad[index] else None
                for index, value in enumerate(values, start=1)
            ),
        )

    @staticmethod
    def jvp(ctx, _failure_state_tangent, *tangents):
        tangents = tuple(_ad_native_tangent_or_none(value) for value in tangents)
        if all(value is None for value in tangents):
            return (None,) * len(_MC_COMPONENT_MAP_FIELDS)
        failure_state_bits, reference = (
            _ad_native_tensor(value) for value in ctx.saved_tensors
        )
        with disable_functorch():
            return _mc_capacity_failure_component_maps_sanitize_jvp_native(
                failure_state_bits, reference, *tangents
            )


def mc_capacity_failure_component_maps_sanitize(
    los: torch.Tensor,
    reflection: torch.Tensor,
    diffraction: torch.Tensor,
    transmission: torch.Tensor,
    scattering: torch.Tensor,
    *,
    failure_state: CapacityFailureState,
) -> dict[str, torch.Tensor]:
    """Make every MC Basic component map inert after transaction failure."""

    maps = (los, reflection, diffraction, transmission, scattering)
    reference = _validate_capacity_component_maps(maps)
    require_capacity_failure_state(failure_state, device=reference.device)
    values = _McCapacityFailureComponentMapsSanitizeFunction.apply(
        failure_state.bits, *maps
    )
    return dict(zip(_MC_COMPONENT_MAP_FIELDS, values, strict=True))


# -------------------------------------------------------------------------
# basic maps
# -------------------------------------------------------------------------
_LIGHT_SPEED_M_PER_S_AD = 299_792_458.0

_MC_FINALIZE_FIELDS = (
    "path_gain",
    "los_power",
    "reflection_power",
    "diffraction_power",
    "transmission_power",
    "scattering_power",
)


def mc_finalize_component_maps(
    los: torch.Tensor,
    reflection: torch.Tensor,
    diffraction: torch.Tensor,
    transmission: torch.Tensor,
    scattering: torch.Tensor,
) -> dict[str, torch.Tensor]:
    validate_cuda_tensor("los", los, dtype=torch.float32, ndim=3)
    validate_cuda_tensor("reflection", reflection, dtype=torch.float32, ndim=3)
    validate_cuda_tensor("diffraction", diffraction, dtype=torch.float32, ndim=3)
    validate_cuda_tensor("transmission", transmission, dtype=torch.float32, ndim=3)
    validate_cuda_tensor("scattering", scattering, dtype=torch.float32, ndim=3)
    if reflection.shape != los.shape:
        raise ValueError("reflection must match los shape")
    if diffraction.shape != los.shape:
        raise ValueError("diffraction must match los shape")
    if transmission.shape != los.shape:
        raise ValueError("transmission must match los shape")
    if scattering.shape != los.shape:
        raise ValueError("scattering must match los shape")

    exported = _required_native_op("mc_finalize_component_maps")(
        los, reflection, diffraction, transmission, scattering
    )
    if not isinstance(exported, dict):
        raise TypeError("_channel.mc_finalize_component_maps must return a dict")
    return exported


class _McFinalizeComponentMapsAdFunction(torch.autograd.Function):
    """Native linear map finalization and power-reduction AD (plan 07 AD-3)."""

    @staticmethod
    def forward(los, reflection, diffraction, transmission, scattering):
        out = mc_finalize_component_maps(
            los, reflection, diffraction, transmission, scattering
        )
        return tuple(out[name] for name in _MC_FINALIZE_FIELDS)

    @staticmethod
    def setup_context(ctx, inputs, output):
        ctx.set_materialize_grads(False)
        ctx.map_shape = tuple(inputs[0].shape)
        primal = torch.autograd.forward_ad.unpack_dual(inputs[0]).primal
        ctx.save_for_forward(primal)

    @staticmethod
    @_ad_first_order_only
    def backward(ctx, grad_path_gain, *grad_powers):
        tx_count, dim0, dim1 = ctx.map_shape
        base = (
            grad_path_gain.unflatten(1, (dim0, dim1))
            if grad_path_gain is not None
            else None
        )
        grads = []
        for index in range(5):
            if not ctx.needs_input_grad[index]:
                grads.append(None)
                continue
            grad_power = grad_powers[index]
            if grad_power is None:
                grads.append(base)
            elif base is None:
                grads.append(grad_power.expand(ctx.map_shape))
            else:
                grads.append(base + grad_power)
        return tuple(grads)

    @staticmethod
    def jvp(ctx, t_los, t_reflection, t_diffraction, t_transmission, t_scattering):
        tangents = [
            _ad_native_tangent_or_none(value)
            for value in (
                t_los,
                t_reflection,
                t_diffraction,
                t_transmission,
                t_scattering,
            )
        ]
        if all(value is None for value in tangents):
            return (None,) * len(_MC_FINALIZE_FIELDS)
        (reference,) = ctx.saved_tensors
        zero = None
        filled = []
        for value in tangents:
            if value is None:
                if zero is None:
                    zero = torch.zeros_like(_ad_native_tensor(reference))
                value = zero
            filled.append(value)
        with disable_functorch():
            out = mc_finalize_component_maps(*filled)
        return tuple(out[name] for name in _MC_FINALIZE_FIELDS)


def mc_finalize_component_maps_ad(
    los: torch.Tensor,
    reflection: torch.Tensor,
    diffraction: torch.Tensor,
    transmission: torch.Tensor,
    scattering: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Differentiable :func:`mc_finalize_component_maps`."""

    values = _McFinalizeComponentMapsAdFunction.apply(
        los, reflection, diffraction, transmission, scattering
    )
    return dict(zip(_MC_FINALIZE_FIELDS, values, strict=True))


def mc_los_component_maps_adjoint(
    grad_maps: torch.Tensor,
    visible: torch.Tensor | None,
) -> torch.Tensor:
    """Adjoint of the (visibility-masked) LoS component-map layout."""

    if not isinstance(grad_maps, torch.Tensor):
        raise TypeError("grad_maps must be a torch.Tensor")
    if grad_maps.dtype != torch.float32 or not grad_maps.is_cuda:
        raise TypeError("grad_maps must be a float32 CUDA tensor")
    if grad_maps.ndim != 3:
        raise ValueError("grad_maps must have 3 dimensions")
    if visible is None:
        visible = grad_maps.new_empty((0, 0), dtype=torch.bool)
    out = _required_native_op("mc_los_component_maps_adjoint")(grad_maps, visible)
    if not isinstance(out, torch.Tensor):
        raise TypeError(
            "_channel.mc_los_component_maps_adjoint must return a tensor"
        )
    return out


class _McLosGridMapsAdFunction(torch.autograd.Function):
    """Native AD for the visibility-masked matrix-to-map layout (plan 07 AD-3)."""

    @staticmethod
    def forward(matrix, visible, rows, cols):
        maps = mc_los_component_maps_from_matrix(matrix, rows=rows, cols=cols)
        if visible is not None:
            for tx_index in range(int(matrix.shape[0])):
                mc_apply_los_visibility(
                    maps, matrix, visible[tx_index], tx_index=tx_index
                )
        return maps

    @staticmethod
    def setup_context(ctx, inputs, output):
        ctx.set_materialize_grads(False)
        matrix, visible, rows, cols = inputs
        ctx.rows = int(rows)
        ctx.cols = int(cols)
        ctx.has_visible = visible is not None
        if visible is not None:
            ctx.save_for_backward(visible)
            ctx.save_for_forward(visible)
        else:
            ctx.save_for_backward()
            ctx.save_for_forward()

    @staticmethod
    @_ad_first_order_only
    def backward(ctx, grad_maps):
        if grad_maps is None or not ctx.needs_input_grad[0]:
            return None, None, None, None
        visible = ctx.saved_tensors[0] if ctx.has_visible else None
        return (
            mc_los_component_maps_adjoint(grad_maps, visible),
            None,
            None,
            None,
        )

    @staticmethod
    def jvp(ctx, t_matrix, _t_visible, _t_rows, _t_cols):
        tangent = _ad_native_tangent_or_none(t_matrix)
        if tangent is None:
            return None
        visible = ctx.saved_tensors[0] if ctx.has_visible else None
        with disable_functorch():
            maps = mc_los_component_maps_from_matrix(
                tangent, rows=ctx.rows, cols=ctx.cols
            )
            if visible is not None:
                visible = _ad_native_tensor(visible)
                for tx_index in range(int(tangent.shape[0])):
                    mc_apply_los_visibility(
                        maps, tangent, visible[tx_index], tx_index=tx_index
                    )
        return maps


def mc_los_grid_maps_ad(
    matrix: torch.Tensor,
    visible: torch.Tensor | None,
    *,
    rows: int,
    cols: int,
) -> torch.Tensor:
    """Differentiable grid component maps from a (tx, cells) matrix."""

    return _McLosGridMapsAdFunction.apply(matrix, visible, int(rows), int(cols))


def mc_zero_matrix(reference: torch.Tensor, *, rows: int, cols: int) -> torch.Tensor:
    validate_cuda_tensor("reference", reference, dtype=torch.float32, ndim=2)
    if rows < 0 or cols < 0:
        raise ValueError("rows and cols must be non-negative")
    out = _required_native_op("mc_zero_matrix")(reference, int(rows), int(cols))
    if not isinstance(out, torch.Tensor):
        raise TypeError("_channel.mc_zero_matrix must return a tensor")
    validate_cuda_tensor("out", out, dtype=torch.float32, ndim=2)
    if out.shape != (int(rows), int(cols)):
        raise ValueError("_channel.mc_zero_matrix returned an unexpected shape")
    return out


def mc_point_component_power(
    path_gain: torch.Tensor, *, include_los: bool
) -> dict[str, torch.Tensor]:
    validate_cuda_tensor("path_gain", path_gain, dtype=torch.float32, ndim=2)
    exported = _required_native_op("mc_point_component_power")(
        path_gain, bool(include_los)
    )
    if not isinstance(exported, dict):
        raise TypeError("_channel.mc_point_component_power must return a dict")
    for name in ("los", "reflection", "diffraction"):
        validate_cuda_tensor(name, exported[name], dtype=torch.float32, ndim=0)
    return exported


def mc_component_map_buffer(
    reference: torch.Tensor,
    *,
    tx_count: int,
    dim0: int,
    dim1: int,
) -> torch.Tensor:
    validate_cuda_tensor("reference", reference, dtype=torch.float32, ndim=2)
    if tx_count < 0 or dim0 < 0 or dim1 < 0:
        raise ValueError("tx_count, dim0, and dim1 must be non-negative")
    maps = _required_native_op("mc_component_map_buffer")(
        reference, int(tx_count), int(dim0), int(dim1)
    )
    if not isinstance(maps, torch.Tensor):
        raise TypeError("_channel.mc_component_map_buffer must return a tensor")
    validate_cuda_tensor("maps", maps, dtype=torch.float32, ndim=3)
    if maps.shape != (tx_count, dim0, dim1):
        raise ValueError(
            "_channel.mc_component_map_buffer returned an unexpected shape"
        )
    return maps


def mc_store_component_map(
    maps: torch.Tensor,
    source: torch.Tensor,
    *,
    tx_index: int,
) -> torch.Tensor:
    validate_cuda_tensor("maps", maps, dtype=torch.float32, ndim=3)
    validate_cuda_tensor("source", source, dtype=torch.float32, ndim=2)
    if source.shape != maps.shape[1:]:
        raise ValueError("source shape must match one maps slot")
    out = _required_native_op("mc_store_component_map")(maps, source, int(tx_index))
    if not isinstance(out, torch.Tensor):
        raise TypeError("_channel.mc_store_component_map must return a tensor")
    validate_cuda_tensor("maps", out, dtype=torch.float32, ndim=3)
    return out


def mc_store_scaled_component_map(
    maps: torch.Tensor,
    source: torch.Tensor,
    scale_values: torch.Tensor,
    *,
    tx_index: int,
    scale_index: int,
) -> torch.Tensor:
    validate_cuda_tensor("maps", maps, dtype=torch.float32, ndim=3)
    validate_cuda_tensor("source", source, dtype=torch.float32, ndim=2)
    validate_cuda_tensor("scale_values", scale_values, dtype=torch.float32, ndim=1)
    if source.shape != maps.shape[1:]:
        raise ValueError("source shape must match one maps slot")
    out = _required_native_op("mc_store_scaled_component_map")(
        maps,
        source,
        scale_values,
        int(tx_index),
        int(scale_index),
    )
    if not isinstance(out, torch.Tensor):
        raise TypeError(
            "_channel.mc_store_scaled_component_map must return a tensor"
        )
    validate_cuda_tensor("maps", out, dtype=torch.float32, ndim=3)
    return out


def mc_los_component_maps(los: torch.Tensor) -> torch.Tensor:
    validate_cuda_tensor("los", los, dtype=torch.float32, ndim=3)
    maps = _required_native_op("mc_los_component_maps")(los)
    if not isinstance(maps, torch.Tensor):
        raise TypeError("_channel.mc_los_component_maps must return a tensor")
    validate_cuda_tensor("maps", maps, dtype=torch.float32, ndim=3)
    return maps


def mc_los_component_maps_from_matrix(
    los: torch.Tensor, *, rows: int, cols: int
) -> torch.Tensor:
    validate_cuda_tensor("los", los, dtype=torch.float32, ndim=2)
    if rows < 0 or cols < 0:
        raise ValueError("rows and cols must be non-negative")
    if los.shape[1] != rows * cols:
        raise ValueError("los columns must match rows * cols")
    maps = _required_native_op("mc_los_component_maps_from_matrix")(
        los, int(rows), int(cols)
    )
    if not isinstance(maps, torch.Tensor):
        raise TypeError(
            "_channel.mc_los_component_maps_from_matrix must return a tensor"
        )
    validate_cuda_tensor("maps", maps, dtype=torch.float32, ndim=3)
    if maps.shape != (los.shape[0], cols, rows):
        raise ValueError(
            "_channel.mc_los_component_maps_from_matrix returned an unexpected shape"
        )
    return maps


def mc_apply_los_visibility(
    maps: torch.Tensor,
    los: torch.Tensor,
    visible: torch.Tensor,
    *,
    tx_index: int,
) -> torch.Tensor:
    validate_cuda_tensor("maps", maps, dtype=torch.float32, ndim=3)
    validate_cuda_tensor("los", los, dtype=torch.float32, ndim=2)
    validate_cuda_tensor("visible", visible, dtype=torch.bool, ndim=1)
    if maps.shape[0] != los.shape[0] or los.shape[1] != maps.shape[1] * maps.shape[2]:
        raise ValueError("los must match maps flattened grid layout")
    out = _required_native_op("mc_apply_los_visibility")(
        maps, los, visible, int(tx_index)
    )
    if not isinstance(out, torch.Tensor):
        raise TypeError("_channel.mc_apply_los_visibility must return a tensor")
    return out


def mc_los_visibility_inputs(
    tx_positions: torch.Tensor,
    *,
    tx_index: int,
    rx_count: int,
) -> dict[str, torch.Tensor]:
    validate_cuda_tensor(
        "tx_positions", tx_positions, dtype=torch.float32, ndim=2, trailing_shape=(3,)
    )
    if rx_count < 0:
        raise ValueError("rx_count must be non-negative")
    exported = _required_native_op("mc_los_visibility_inputs")(
        tx_positions, int(tx_index), int(rx_count)
    )
    if not isinstance(exported, dict):
        raise TypeError("_channel.mc_los_visibility_inputs must return a dict")
    validate_cuda_tensor(
        "start", exported["start"], dtype=torch.float32, ndim=2, trailing_shape=(3,)
    )
    validate_cuda_tensor("active", exported["active"], dtype=torch.bool, ndim=1)
    return exported


def mc_los_path_gain_backward(
    tx_positions: torch.Tensor,
    tx_power: torch.Tensor,
    rx_positions: torch.Tensor,
    grad_output: torch.Tensor,
    tx_polarizations: torch.Tensor,
    *,
    frequency_hz: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    validate_cuda_tensor(
        "tx_positions", tx_positions, dtype=torch.float32, ndim=2, trailing_shape=(3,)
    )
    validate_cuda_tensor("tx_power", tx_power, dtype=torch.float32, ndim=1)
    validate_cuda_tensor(
        "rx_positions", rx_positions, dtype=torch.float32, ndim=2, trailing_shape=(3,)
    )
    validate_cuda_tensor(
        "tx_polarizations",
        tx_polarizations,
        dtype=torch.float32,
        ndim=2,
        trailing_shape=(3,),
    )
    if not isinstance(grad_output, torch.Tensor):
        raise TypeError("grad_output must be a torch.Tensor")
    if grad_output.dtype != torch.float32:
        raise TypeError("grad_output must have dtype torch.float32")
    if not grad_output.is_cuda:
        raise ValueError("grad_output must be a CUDA tensor")
    if grad_output.ndim != 2:
        raise ValueError("grad_output must have 2 dimensions")
    if grad_output.shape != (tx_positions.shape[0], rx_positions.shape[0]):
        raise ValueError("grad_output must match the LoS path-gain matrix shape")
    if tx_power.shape[0] != tx_positions.shape[0]:
        raise ValueError("tx_power must have one value per transmitter")
    if frequency_hz <= 0.0:
        raise ValueError("frequency_hz must be positive")

    gradients = _required_native_op("mc_los_path_gain_backward")(
        tx_positions,
        tx_power,
        rx_positions,
        grad_output,
        float(frequency_hz),
        tx_polarizations,
    )
    if not isinstance(gradients, tuple) or len(gradients) != 4:
        raise TypeError(
            "_channel.mc_los_path_gain_backward must return 4 tensors"
        )
    validate_cuda_tensor(
        "grad_tx", gradients[0], dtype=torch.float32, ndim=2, trailing_shape=(3,)
    )
    validate_cuda_tensor("grad_power", gradients[1], dtype=torch.float32, ndim=1)
    validate_cuda_tensor(
        "grad_rx", gradients[2], dtype=torch.float32, ndim=2, trailing_shape=(3,)
    )
    validate_cuda_tensor("grad_frequency", gradients[3], dtype=torch.float32, ndim=1)
    if gradients[0].shape != tx_positions.shape:
        raise ValueError(
            "_channel.mc_los_path_gain_backward returned bad grad_tx shape"
        )
    if gradients[1].shape != tx_power.shape:
        raise ValueError(
            "_channel.mc_los_path_gain_backward returned bad grad_power shape"
        )
    if gradients[2].shape != rx_positions.shape:
        raise ValueError(
            "_channel.mc_los_path_gain_backward returned bad grad_rx shape"
        )
    if gradients[3].shape != (1,):
        raise ValueError(
            "_channel.mc_los_path_gain_backward returned bad grad_frequency shape"
        )
    return gradients


def mc_los_path_gain_jvp(
    tx_positions: torch.Tensor,
    tx_power: torch.Tensor,
    rx_positions: torch.Tensor,
    tx_tangent: torch.Tensor,
    power_tangent: torch.Tensor,
    rx_tangent: torch.Tensor,
    has_tx_tangent: bool,
    has_power_tangent: bool,
    has_rx_tangent: bool,
    tx_polarizations: torch.Tensor,
    *,
    frequency_hz: float,
    frequency_tangent: float = 0.0,
) -> torch.Tensor:
    validate_cuda_tensor(
        "tx_positions", tx_positions, dtype=torch.float32, ndim=2, trailing_shape=(3,)
    )
    validate_cuda_tensor("tx_power", tx_power, dtype=torch.float32, ndim=1)
    validate_cuda_tensor(
        "rx_positions", rx_positions, dtype=torch.float32, ndim=2, trailing_shape=(3,)
    )
    validate_cuda_tensor(
        "tx_polarizations",
        tx_polarizations,
        dtype=torch.float32,
        ndim=2,
        trailing_shape=(3,),
    )
    if tx_power.shape[0] != tx_positions.shape[0]:
        raise ValueError("tx_power must have one value per transmitter")
    if has_tx_tangent:
        validate_cuda_tensor(
            "tx_tangent", tx_tangent, dtype=torch.float32, ndim=2, trailing_shape=(3,)
        )
        if tx_tangent.shape != tx_positions.shape:
            raise ValueError("tx_tangent must match tx_positions")
    if has_power_tangent:
        validate_cuda_tensor(
            "power_tangent", power_tangent, dtype=torch.float32, ndim=1
        )
        if power_tangent.shape != tx_power.shape:
            raise ValueError("power_tangent must match tx_power")
    if has_rx_tangent:
        validate_cuda_tensor(
            "rx_tangent", rx_tangent, dtype=torch.float32, ndim=2, trailing_shape=(3,)
        )
        if rx_tangent.shape != rx_positions.shape:
            raise ValueError("rx_tangent must match rx_positions")
    if frequency_hz <= 0.0:
        raise ValueError("frequency_hz must be positive")

    out = _required_native_op("mc_los_path_gain_jvp")(
        tx_positions,
        tx_power,
        rx_positions,
        tx_tangent,
        power_tangent,
        rx_tangent,
        bool(has_tx_tangent),
        bool(has_power_tangent),
        bool(has_rx_tangent),
        float(frequency_hz),
        float(frequency_tangent),
        tx_polarizations,
    )
    if not isinstance(out, torch.Tensor):
        raise TypeError("_channel.mc_los_path_gain_jvp must return a tensor")
    validate_cuda_tensor("out", out, dtype=torch.float32, ndim=2)
    if out.shape != (tx_positions.shape[0], rx_positions.shape[0]):
        raise ValueError(
            "_channel.mc_los_path_gain_jvp returned an unexpected shape"
        )
    return out


class _McLosPathGainAdFunction(torch.autograd.Function):
    """Differentiable LoS power-gain matrix (plan 07 AD-3).

    Differentiable inputs: tx_positions, rx_positions and the carrier
    frequency. tx_power stays fixed under the plan 07 contract; requesting
    its gradient fails loudly. The forward is the primal path_los_export
    kernel; only the path_gain_matrix output is exposed.
    """

    @staticmethod
    def forward(
        tx_positions, tx_power, rx_positions, frequency, frequency_value, tx_pol
    ):
        exported = path_los_export(
            tx_positions,
            tx_power,
            rx_positions,
            tx_pol,
            frequency_hz=frequency_value,
        )
        return exported["path_gain_matrix"]

    @staticmethod
    def setup_context(ctx, inputs, output):
        ctx.set_materialize_grads(False)
        tx_positions, tx_power, rx_positions, frequency, frequency_value, tx_pol = (
            inputs
        )
        primals = tuple(
            torch.autograd.forward_ad.unpack_dual(value).primal
            for value in (tx_positions, tx_power, rx_positions, tx_pol)
        )
        ctx.frequency_value = frequency_value
        ctx.frequency_meta = (
            (frequency.dtype, frequency.device)
            if isinstance(frequency, torch.Tensor)
            else None
        )
        ctx.save_for_backward(*primals)
        ctx.save_for_forward(*primals)

    @staticmethod
    @_ad_first_order_only
    def backward(ctx, grad_output):
        _ad_reject_fixed_inputs(
            "mc_los_path_gain_ad",
            ctx.needs_input_grad,
            ((1, "tx_power"),),
        )
        need_tx = bool(ctx.needs_input_grad[0])
        need_rx = bool(ctx.needs_input_grad[2])
        need_frequency = bool(ctx.needs_input_grad[3])
        if grad_output is None or not (need_tx or need_rx or need_frequency):
            return None, None, None, None, None, None
        tx_positions, tx_power, rx_positions, tx_pol = ctx.saved_tensors
        grad_tx, _grad_power, grad_rx, grad_frequency = mc_los_path_gain_backward(
            tx_positions,
            tx_power,
            rx_positions,
            grad_output,
            tx_pol,
            frequency_hz=ctx.frequency_value,
        )
        return (
            grad_tx if need_tx else None,
            None,
            grad_rx if need_rx else None,
            _ad_frequency_grad(grad_frequency, ctx.frequency_meta)
            if need_frequency
            else None,
            None,
            None,
        )

    @staticmethod
    def jvp(ctx, t_tx, t_power, t_rx, t_frequency, _t_frequency_value, _t_tx_pol):
        _ad_reject_fixed_tangents("mc_los_path_gain_ad", ((t_power, "tx_power"),))
        saved = ctx.saved_tensors
        tangent_tx = _ad_geometry_tangent(
            "mc_los_path_gain_ad tangent_tx", t_tx, saved[0]
        )
        tangent_rx = _ad_geometry_tangent(
            "mc_los_path_gain_ad tangent_rx", t_rx, saved[2]
        )
        tangent_frequency = _ad_frequency_tangent(t_frequency)
        if tangent_tx is None and tangent_rx is None and tangent_frequency == 0.0:
            return None
        tx_positions, tx_power, rx_positions, tx_pol = saved
        with disable_functorch():
            return mc_los_path_gain_jvp(
                _ad_native_tensor(tx_positions),
                _ad_native_tensor(tx_power),
                _ad_native_tensor(rx_positions),
                tangent_tx
                if tangent_tx is not None
                else _ad_native_tensor(tx_positions),
                _ad_native_tensor(tx_power),
                tangent_rx
                if tangent_rx is not None
                else _ad_native_tensor(rx_positions),
                tangent_tx is not None,
                False,
                tangent_rx is not None,
                _ad_native_tensor(tx_pol),
                frequency_hz=ctx.frequency_value,
                frequency_tangent=tangent_frequency,
            )


def mc_los_path_gain_ad(
    tx_positions: torch.Tensor,
    tx_power: torch.Tensor,
    rx_positions: torch.Tensor,
    tx_polarizations: torch.Tensor,
    *,
    frequency: torch.Tensor | float,
    frequency_value: float | None = None,
) -> torch.Tensor:
    """Differentiable LoS path-gain matrix (endpoints and frequency).

    ``frequency_value`` optionally carries the precomputed host scalar of
    ``frequency`` (one read per solve at the seam, audit M3); when not
    supplied it is read here, exactly once per apply. ``tx_polarizations`` is
    the fixed per-transmitter polarization (frozen winner) driving the dipole
    sin^2 pattern; its endpoint dependence is differentiated natively.
    """

    if frequency_value is None:
        frequency_value = _ad_frequency_value(frequency)
    return _McLosPathGainAdFunction.apply(
        tx_positions,
        tx_power,
        rx_positions,
        frequency,
        float(frequency_value),
        tx_polarizations,
    )


def mc_slab_reflection_accumulate(
    ray_o: torch.Tensor,
    ray_d: torch.Tensor,
    trace_valid: torch.Tensor,
    trace_t: torch.Tensor,
    trace_prim: torch.Tensor,
    face_normals: torch.Tensor,
    material_eta_r: torch.Tensor,
    material_sigma: torch.Tensor,
    material_gain: torch.Tensor,
    material_valid: torch.Tensor,
    material_thickness: torch.Tensor,
    *,
    tx_pol: torch.Tensor,
    contribution_depth: int,
    grid_axis: int,
    grid_position: float,
    grid_coord0_min: float,
    grid_coord0_max: float,
    grid_coord1_min: float,
    grid_coord1_max: float,
    grid_resolution0: int,
    grid_resolution1: int,
    wavelength: float,
    solid_angle_per_ray: float,
    grid_cell_area: float,
) -> torch.Tensor:
    """Accumulate finite-thickness slab specular reflections into a radiomap.

    Convention. This family reproduces one law only, and that law is a
    material law rather than a sampling rule: the ITU finite-thickness
    single-slab Fresnel TE/TM coefficient pair over ``eta_r`` / ``sigma_e`` /
    ``gain`` / ``thickness`` / ``wavelength``, applied per bounce in the
    ``(s_hat, p_in) -> (s_hat, p_out)`` frame and deposited incoherently as
    ``|E|^2 * solid_angle * (lambda / 4pi)^2 / (cell_area * |cos|)`` at the
    first plane crossing that the next trace hit does not occlude. It
    deliberately does NOT reproduce the hardcoded-vertical radiomap source
    convention, under which every launched ray leaves the transmitter with
    unit magnitude and a fixed z-hat polarization regardless of launch
    direction. Here the launched field is the unnormalized transverse
    projection of the true transmitter polarization instead, so every deposit
    carries the short-dipole ``sin^2(theta)`` pattern (R5 polarization
    consistency with the LoS and diffraction maps). Level parity with a
    unit-magnitude-source radiomap is therefore knowingly abandoned; a
    near-axis level offset is the intended behaviour, not a defect.
    """

    return _required_native_op("mc_slab_reflection_accumulate")(
        ray_o,
        ray_d,
        trace_valid,
        trace_t,
        trace_prim,
        face_normals,
        material_eta_r,
        material_sigma,
        material_gain,
        material_valid,
        material_thickness,
        int(contribution_depth),
        int(grid_axis),
        float(grid_position),
        float(grid_coord0_min),
        float(grid_coord0_max),
        float(grid_coord1_min),
        float(grid_coord1_max),
        int(grid_resolution0),
        int(grid_resolution1),
        float(wavelength),
        float(solid_angle_per_ray),
        float(grid_cell_area),
        tx_pol,
    )


def mc_reflection_ad_max_depth() -> int:
    """Depth cap of the native reflection radiomap AD companions.

    The backward/jvp kernels stage per-bounce state in fixed-size register
    arrays, so ``contribution_depth`` must not exceed this cap. Exposed so the
    solver can reject an over-deep AD configuration before any launch.
    """

    return int(_required_native_op("mc_reflection_ad_max_depth")())


def mc_slab_reflection_accumulate_backward(
    ray_o: torch.Tensor,
    ray_d: torch.Tensor,
    trace_valid: torch.Tensor,
    trace_t: torch.Tensor,
    trace_prim: torch.Tensor,
    face_normals: torch.Tensor,
    material_eta_r: torch.Tensor,
    material_sigma: torch.Tensor,
    material_gain: torch.Tensor,
    material_valid: torch.Tensor,
    material_thickness: torch.Tensor,
    grad_output: torch.Tensor,
    *,
    tx_pol: torch.Tensor,
    need_materials: bool,
    need_frequency: bool,
    contribution_depth: int,
    grid_axis: int,
    grid_position: float,
    grid_coord0_min: float,
    grid_coord0_max: float,
    grid_coord1_min: float,
    grid_coord1_max: float,
    grid_resolution0: int,
    grid_resolution1: int,
    wavelength: float,
    solid_angle_per_ray: float,
    grid_cell_area: float,
    wavelength_dfreq: float,
) -> tuple[torch.Tensor, ...]:
    gradients = _required_native_op("mc_slab_reflection_accumulate_backward")(
        ray_o,
        ray_d,
        trace_valid,
        trace_t,
        trace_prim,
        face_normals,
        material_eta_r,
        material_sigma,
        material_gain,
        material_valid,
        material_thickness,
        grad_output,
        bool(need_materials),
        bool(need_frequency),
        int(contribution_depth),
        int(grid_axis),
        float(grid_position),
        float(grid_coord0_min),
        float(grid_coord0_max),
        float(grid_coord1_min),
        float(grid_coord1_max),
        int(grid_resolution0),
        int(grid_resolution1),
        float(wavelength),
        float(solid_angle_per_ray),
        float(grid_cell_area),
        float(wavelength_dfreq),
        tx_pol,
    )
    if not isinstance(gradients, tuple) or len(gradients) != 5:
        raise TypeError(
            "_channel.mc_slab_reflection_accumulate_backward must "
            "return 5 tensors"
        )
    return gradients


def mc_slab_reflection_accumulate_jvp(
    ray_o: torch.Tensor,
    ray_d: torch.Tensor,
    trace_valid: torch.Tensor,
    trace_t: torch.Tensor,
    trace_prim: torch.Tensor,
    face_normals: torch.Tensor,
    material_eta_r: torch.Tensor,
    material_sigma: torch.Tensor,
    material_gain: torch.Tensor,
    material_valid: torch.Tensor,
    material_thickness: torch.Tensor,
    tangent_eta_r: torch.Tensor | None,
    tangent_sigma: torch.Tensor | None,
    tangent_gain: torch.Tensor | None,
    tangent_thickness: torch.Tensor | None,
    *,
    tx_pol: torch.Tensor,
    contribution_depth: int,
    grid_axis: int,
    grid_position: float,
    grid_coord0_min: float,
    grid_coord0_max: float,
    grid_coord1_min: float,
    grid_coord1_max: float,
    grid_resolution0: int,
    grid_resolution1: int,
    wavelength: float,
    solid_angle_per_ray: float,
    grid_cell_area: float,
    wavelength_tangent: float,
) -> torch.Tensor:
    output = _required_native_op("mc_slab_reflection_accumulate_jvp")(
        ray_o,
        ray_d,
        trace_valid,
        trace_t,
        trace_prim,
        face_normals,
        material_eta_r,
        material_sigma,
        material_gain,
        material_valid,
        material_thickness,
        tangent_eta_r if tangent_eta_r is not None else material_eta_r,
        tangent_sigma if tangent_sigma is not None else material_sigma,
        tangent_gain if tangent_gain is not None else material_gain,
        tangent_thickness if tangent_thickness is not None else material_thickness,
        tangent_eta_r is not None,
        tangent_sigma is not None,
        tangent_gain is not None,
        tangent_thickness is not None,
        int(contribution_depth),
        int(grid_axis),
        float(grid_position),
        float(grid_coord0_min),
        float(grid_coord0_max),
        float(grid_coord1_min),
        float(grid_coord1_max),
        int(grid_resolution0),
        int(grid_resolution1),
        float(wavelength),
        float(solid_angle_per_ray),
        float(grid_cell_area),
        float(wavelength_tangent),
        tx_pol,
    )
    if not isinstance(output, torch.Tensor):
        raise TypeError(
            "_channel.mc_slab_reflection_accumulate_jvp must return a tensor"
        )
    return output


class _McReflectionMapAdFunction(torch.autograd.Function):
    """Differentiable slab reflection radiomap for one transmitter.

    Differentiable inputs: per-face eta_r / sigma_e / gain / thickness, the
    carrier frequency and the transmitter anchor. The trace tape, sampled
    directions, face normals and validity masks are frozen winners and fail
    loudly when a gradient is requested. The deposit weight is independent of
    the ray origin, so the transmitter anchor receives an exact zero gradient
    without a kernel launch (its binning influence is discrete and frozen).
    """

    @staticmethod
    def forward(
        tx_anchor,
        eta_r,
        sigma_e,
        gain,
        thickness,
        frequency,
        ray_o,
        ray_d,
        trace_valid,
        trace_t,
        trace_prim,
        face_normals,
        material_valid,
        params,
    ):
        return mc_slab_reflection_accumulate(
            ray_o,
            ray_d,
            trace_valid,
            trace_t,
            trace_prim,
            face_normals,
            eta_r,
            sigma_e,
            gain,
            material_valid,
            thickness,
            tx_pol=params["tx_pol"],
            contribution_depth=params["contribution_depth"],
            grid_axis=params["grid_axis"],
            grid_position=params["grid_position"],
            grid_coord0_min=params["grid_coord0_min"],
            grid_coord0_max=params["grid_coord0_max"],
            grid_coord1_min=params["grid_coord1_min"],
            grid_coord1_max=params["grid_coord1_max"],
            grid_resolution0=params["grid_resolution0"],
            grid_resolution1=params["grid_resolution1"],
            wavelength=params["wavelength"],
            solid_angle_per_ray=params["solid_angle_per_ray"],
            grid_cell_area=params["grid_cell_area"],
        )

    @staticmethod
    def setup_context(ctx, inputs, output):
        ctx.set_materialize_grads(False)
        primals = tuple(
            torch.autograd.forward_ad.unpack_dual(value).primal for value in inputs[:5]
        )
        frequency = inputs[5]
        ctx.frequency_meta = (
            (frequency.dtype, frequency.device)
            if isinstance(frequency, torch.Tensor)
            else None
        )
        ctx.params = inputs[13]
        ctx.anchor_shape = tuple(inputs[0].shape)
        saved = (*primals[1:], *inputs[6:13])
        ctx.save_for_backward(*saved)
        ctx.save_for_forward(*saved)

    @staticmethod
    @_ad_first_order_only
    def backward(ctx, grad_output):
        none_grads = (None,) * 14
        _ad_reject_fixed_inputs(
            "mc_slab_reflection_accumulate_ad",
            ctx.needs_input_grad,
            (
                (6, "ray_o"),
                (7, "ray_d"),
                (8, "trace_valid"),
                (9, "trace_t"),
                (10, "trace_prim"),
                (11, "face_normals"),
                (12, "material_valid"),
            ),
        )
        if grad_output is None:
            return none_grads
        (
            eta_r,
            sigma_e,
            gain,
            thickness,
            ray_o,
            ray_d,
            trace_valid,
            trace_t,
            trace_prim,
            face_normals,
            material_valid,
        ) = ctx.saved_tensors
        need_anchor = bool(ctx.needs_input_grad[0])
        need_eta = bool(ctx.needs_input_grad[1])
        need_sigma = bool(ctx.needs_input_grad[2])
        need_gain = bool(ctx.needs_input_grad[3])
        need_thickness = bool(ctx.needs_input_grad[4])
        need_materials = need_eta or need_sigma or need_gain or need_thickness
        need_frequency = bool(ctx.needs_input_grad[5])
        grad_anchor = None
        if need_anchor:
            # The deposit weight carries no ray-origin dependence; the frozen
            # binning is the only door and it is discrete. Exact zero.
            grad_anchor = mc_zero_matrix(
                ray_o, rows=ctx.anchor_shape[0], cols=ctx.anchor_shape[1]
            )
        if not (need_materials or need_frequency):
            return (grad_anchor,) + (None,) * 13
        params = ctx.params
        wavelength = float(params["wavelength"])
        gradients = mc_slab_reflection_accumulate_backward(
            ray_o,
            ray_d,
            trace_valid,
            trace_t,
            trace_prim,
            face_normals,
            eta_r,
            sigma_e,
            gain,
            material_valid,
            thickness,
            grad_output,
            tx_pol=params["tx_pol"],
            need_materials=need_materials,
            need_frequency=need_frequency,
            contribution_depth=params["contribution_depth"],
            grid_axis=params["grid_axis"],
            grid_position=params["grid_position"],
            grid_coord0_min=params["grid_coord0_min"],
            grid_coord0_max=params["grid_coord0_max"],
            grid_coord1_min=params["grid_coord1_min"],
            grid_coord1_max=params["grid_coord1_max"],
            grid_resolution0=params["grid_resolution0"],
            grid_resolution1=params["grid_resolution1"],
            wavelength=wavelength,
            solid_angle_per_ray=params["solid_angle_per_ray"],
            grid_cell_area=params["grid_cell_area"],
            wavelength_dfreq=-wavelength * wavelength / _LIGHT_SPEED_M_PER_S_AD,
        )
        grad_frequency = (
            _ad_frequency_grad(gradients[4], ctx.frequency_meta)
            if need_frequency
            else None
        )
        return (
            grad_anchor,
            gradients[0] if need_eta else None,
            gradients[1] if need_sigma else None,
            gradients[2] if need_gain else None,
            gradients[3] if need_thickness else None,
            grad_frequency,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )

    @staticmethod
    def jvp(
        ctx,
        t_anchor,
        t_eta,
        t_sigma,
        t_gain,
        t_thickness,
        t_frequency,
        t_ray_o,
        t_ray_d,
        t_trace_valid,
        t_trace_t,
        t_trace_prim,
        t_face_normals,
        t_material_valid,
        _t_params,
    ):
        _ad_reject_fixed_tangents(
            "mc_slab_reflection_accumulate_ad",
            (
                (t_ray_o, "ray_o"),
                (t_ray_d, "ray_d"),
                (t_face_normals, "face_normals"),
            ),
        )
        saved = ctx.saved_tensors
        face_shape = tuple(saved[0].shape)
        tangent_eta = _ad_checked_tangent(
            "mc_slab_reflection_accumulate_ad tangent_eta_r",
            _ad_native_tangent_or_none(t_eta),
            face_shape,
        )
        tangent_sigma = _ad_checked_tangent(
            "mc_slab_reflection_accumulate_ad tangent_sigma_e",
            _ad_native_tangent_or_none(t_sigma),
            face_shape,
        )
        tangent_gain = _ad_checked_tangent(
            "mc_slab_reflection_accumulate_ad tangent_gain",
            _ad_native_tangent_or_none(t_gain),
            face_shape,
        )
        tangent_thickness = _ad_checked_tangent(
            "mc_slab_reflection_accumulate_ad tangent_thickness",
            _ad_native_tangent_or_none(t_thickness),
            face_shape,
        )
        tangent_frequency = _ad_frequency_tangent(t_frequency)
        params = ctx.params
        if (
            tangent_eta is None
            and tangent_sigma is None
            and tangent_gain is None
            and tangent_thickness is None
            and tangent_frequency == 0.0
        ):
            if (
                _ad_native_tangent_or_none(
                    t_anchor if isinstance(t_anchor, torch.Tensor) else None
                )
                is None
            ):
                return None
            # Transmitter-anchor-only tangent: the deposit weight carries no
            # ray-origin dependence (frozen binning), so the map tangent is
            # exactly zero. A concrete zero tensor is required: this torch
            # build rejects a None jvp when an input tangent is live.
            return mc_zero_matrix(
                _ad_native_tensor(saved[4]),
                rows=params["grid_resolution1"],
                cols=params["grid_resolution0"],
            )
        (
            eta_r,
            sigma_e,
            gain,
            thickness,
            ray_o,
            ray_d,
            trace_valid,
            trace_t,
            trace_prim,
            face_normals,
            material_valid,
        ) = saved
        wavelength = float(params["wavelength"])
        with disable_functorch():
            return mc_slab_reflection_accumulate_jvp(
                _ad_native_tensor(ray_o),
                _ad_native_tensor(ray_d),
                _ad_native_tensor(trace_valid),
                _ad_native_tensor(trace_t),
                _ad_native_tensor(trace_prim),
                _ad_native_tensor(face_normals),
                _ad_native_tensor(eta_r),
                _ad_native_tensor(sigma_e),
                _ad_native_tensor(gain),
                _ad_native_tensor(material_valid),
                _ad_native_tensor(thickness),
                tangent_eta,
                tangent_sigma,
                tangent_gain,
                tangent_thickness,
                tx_pol=params["tx_pol"],
                contribution_depth=params["contribution_depth"],
                grid_axis=params["grid_axis"],
                grid_position=params["grid_position"],
                grid_coord0_min=params["grid_coord0_min"],
                grid_coord0_max=params["grid_coord0_max"],
                grid_coord1_min=params["grid_coord1_min"],
                grid_coord1_max=params["grid_coord1_max"],
                grid_resolution0=params["grid_resolution0"],
                grid_resolution1=params["grid_resolution1"],
                wavelength=wavelength,
                solid_angle_per_ray=params["solid_angle_per_ray"],
                grid_cell_area=params["grid_cell_area"],
                wavelength_tangent=(-wavelength * wavelength / _LIGHT_SPEED_M_PER_S_AD)
                * tangent_frequency,
            )


def mc_slab_reflection_accumulate_ad(
    tx_anchor: torch.Tensor,
    material_eta_r: torch.Tensor,
    material_sigma: torch.Tensor,
    material_gain: torch.Tensor,
    material_thickness: torch.Tensor,
    frequency: torch.Tensor | float,
    ray_o: torch.Tensor,
    ray_d: torch.Tensor,
    trace_valid: torch.Tensor,
    trace_t: torch.Tensor,
    trace_prim: torch.Tensor,
    face_normals: torch.Tensor,
    material_valid: torch.Tensor,
    *,
    tx_pol: torch.Tensor,
    contribution_depth: int,
    grid_axis: int,
    grid_position: float,
    grid_coord0_min: float,
    grid_coord0_max: float,
    grid_coord1_min: float,
    grid_coord1_max: float,
    grid_resolution0: int,
    grid_resolution1: int,
    wavelength: float,
    solid_angle_per_ray: float,
    grid_cell_area: float,
) -> torch.Tensor:
    """Differentiable :func:`mc_slab_reflection_accumulate` (one tx)."""

    params = {
        "tx_pol": tx_pol,
        "contribution_depth": int(contribution_depth),
        "grid_axis": int(grid_axis),
        "grid_position": float(grid_position),
        "grid_coord0_min": float(grid_coord0_min),
        "grid_coord0_max": float(grid_coord0_max),
        "grid_coord1_min": float(grid_coord1_min),
        "grid_coord1_max": float(grid_coord1_max),
        "grid_resolution0": int(grid_resolution0),
        "grid_resolution1": int(grid_resolution1),
        "wavelength": float(wavelength),
        "solid_angle_per_ray": float(solid_angle_per_ray),
        "grid_cell_area": float(grid_cell_area),
    }
    return _McReflectionMapAdFunction.apply(
        tx_anchor,
        material_eta_r,
        material_sigma,
        material_gain,
        material_thickness,
        frequency,
        ray_o,
        ray_d,
        trace_valid,
        trace_t,
        trace_prim,
        face_normals,
        material_valid,
        params,
    )


def mc_utd_diffraction_tape_accumulate(*args: object) -> torch.Tensor:
    """Accumulate UTD diffraction power over the sampled Keller-cone tape.

    Convention. The convention this family adopts is a Monte Carlo per-sample
    acceptance interval, not a material law: RayD proposes the complete Keller
    cone, a lane whose edge azimuth exceeds the wedge exterior angle is
    rejected, and an accepted lane keeps the full-cone ``1 / (2pi)`` proposal
    density, so its weight stays ``2pi`` rather than the accepted interval
    width. Everything else is this package's own UTD: the full
    Kouyoumjian-Pathak pair over the stored finite-thickness slab face
    operators (fixed-point plus stored-ops convention, ``selectStationaryPoint
    = 0`` and ``mat.omega = 0`` at the pair call), pseudo-infinite ``+-1e5``
    edge bounds, and the true per-transmitter polarization (R5) rather than a
    hardcoded vertical source. It deliberately does NOT reproduce the
    diffracted field level or the cell-by-cell lit set of a radiomap built on
    that hardcoded unit-magnitude vertical source; the short-dipole source
    pattern alone separates the two, so a level delta is expected.
    """

    output = _required_native_op("mc_utd_diffraction_tape_accumulate")(*args)
    if not isinstance(output, torch.Tensor):
        raise TypeError(
            "_channel.mc_utd_diffraction_tape_accumulate must return a tensor"
        )
    return output


# ---------------------------------------------------------------------------
# MC basic incoherent power-map AD (plan 07 AD-3). The solver emits a REAL
# power map assembled from per-component maps; the differentiable inputs are
# the compiled material store leaves (per-face eta_r / sigma_e / gain /
# thickness, per-layer CSR parameters), the carrier frequency and the LoS
# endpoint positions. The RayD trace and sampling tapes, the sampled ray
# directions, the deposit binning and the visibility masks are all frozen
# winners (plan 07 section 4): gradients cover the continuous part only. The
# reflection deposit weight is analytically independent of the ray origin
# (the weight is |Gamma|^2 * solid_angle * (lambda/4pi)^2 / (A_cell * |cos|),
# with the 1/d^2 spreading carried by the frozen ray density), so the
# transmitter-position gradient of the reflection map is exactly zero and is
# returned without a kernel launch.
# ---------------------------------------------------------------------------


def mc_utd_diffraction_tape_accumulate_backward(
    tape_tensors: tuple[torch.Tensor, ...],
    state_tensors: tuple[torch.Tensor, ...],
    material_eta_r: torch.Tensor,
    material_sigma: torch.Tensor,
    material_mu_r: torch.Tensor,
    material_gain: torch.Tensor,
    material_valid: torch.Tensor,
    material_thickness: torch.Tensor,
    grad_output: torch.Tensor,
    *,
    tx_pol: torch.Tensor,
    need_materials: bool,
    need_source: bool,
    need_frequency: bool,
    grid_axis: int,
    grid_position: float,
    grid_resolution0: int,
    grid_resolution1: int,
    wavelength: float,
    grid_cell_area: float,
    seed: int,
    total_edge_length: float,
    wavelength_dfreq: float,
) -> tuple[torch.Tensor, ...]:
    gradients = _required_native_op("mc_utd_diffraction_tape_accumulate_backward")(
        *tape_tensors,
        *state_tensors,
        material_eta_r,
        material_sigma,
        material_mu_r,
        material_gain,
        material_valid,
        material_thickness,
        grad_output,
        bool(need_materials),
        bool(need_source),
        bool(need_frequency),
        int(grid_axis),
        float(grid_position),
        int(grid_resolution0),
        int(grid_resolution1),
        float(wavelength),
        float(grid_cell_area),
        int(seed),
        float(total_edge_length),
        float(wavelength_dfreq),
        tx_pol,
    )
    if not isinstance(gradients, tuple) or len(gradients) != 6:
        raise TypeError(
            "_channel.mc_utd_diffraction_tape_accumulate_backward "
            "must return 6 tensors"
        )
    return gradients


def mc_utd_diffraction_tape_accumulate_jvp(
    tape_tensors: tuple[torch.Tensor, ...],
    state_tensors: tuple[torch.Tensor, ...],
    material_eta_r: torch.Tensor,
    material_sigma: torch.Tensor,
    material_mu_r: torch.Tensor,
    material_gain: torch.Tensor,
    material_valid: torch.Tensor,
    material_thickness: torch.Tensor,
    tangent_eta_r: torch.Tensor | None,
    tangent_sigma: torch.Tensor | None,
    tangent_gain: torch.Tensor | None,
    tangent_thickness: torch.Tensor | None,
    tangent_source: torch.Tensor | None,
    *,
    tx_pol: torch.Tensor,
    grid_axis: int,
    grid_position: float,
    grid_resolution0: int,
    grid_resolution1: int,
    wavelength: float,
    grid_cell_area: float,
    seed: int,
    total_edge_length: float,
    wavelength_tangent: float,
) -> torch.Tensor:
    output = _required_native_op("mc_utd_diffraction_tape_accumulate_jvp")(
        *tape_tensors,
        *state_tensors,
        material_eta_r,
        material_sigma,
        material_mu_r,
        material_gain,
        material_valid,
        material_thickness,
        tangent_eta_r if tangent_eta_r is not None else material_eta_r,
        tangent_sigma if tangent_sigma is not None else material_sigma,
        tangent_gain if tangent_gain is not None else material_gain,
        tangent_thickness if tangent_thickness is not None else material_thickness,
        tangent_source
        if tangent_source is not None
        else material_eta_r.new_empty((3,)),
        tangent_eta_r is not None,
        tangent_sigma is not None,
        tangent_gain is not None,
        tangent_thickness is not None,
        tangent_source is not None,
        int(grid_axis),
        float(grid_position),
        int(grid_resolution0),
        int(grid_resolution1),
        float(wavelength),
        float(grid_cell_area),
        int(seed),
        float(total_edge_length),
        float(wavelength_tangent),
        tx_pol,
    )
    if not isinstance(output, torch.Tensor):
        raise TypeError(
            "_channel.mc_utd_diffraction_tape_accumulate_jvp must "
            "return a tensor"
        )
    return output


class _McDiffractionMapAdFunction(torch.autograd.Function):
    """Differentiable UTD diffraction radiomap for one transmitter.

    Differentiable inputs: per-face eta_r / sigma_e / gain / thickness, the
    carrier frequency and the transmitter anchor (the state sources are the
    anchor broadcast per winner edge state, so the anchor gradient is the
    per-state source gradient summed natively). The RayD sampling tape
    (active / state / cell / u), the per-lane Keller-cone azimuth, the edge
    state tables and the deposit binning are frozen winners and fail loudly
    when a gradient is requested; the continuous source dependence (incident
    spherical wave, incidence angles, cone orientation and the recomputed
    plane-crossing Jacobian) flows through the templated dual row.
    """

    @staticmethod
    def forward(
        tx_anchor,
        eta_r,
        sigma_e,
        gain,
        thickness,
        frequency,
        tape_active,
        tape_state,
        tape_cell,
        tape_u,
        state_edge_pos,
        state_edge_dir,
        state_t_min,
        state_t_max,
        state_n0,
        state_n1,
        state_prim0,
        state_prim1,
        state_exterior_angle,
        state_src,
        state_src_power,
        material_mu_r,
        material_valid,
        params,
    ):
        return mc_utd_diffraction_tape_accumulate(
            tape_active,
            tape_state,
            tape_cell,
            tape_u,
            state_edge_pos,
            state_edge_dir,
            state_t_min,
            state_t_max,
            state_n0,
            state_n1,
            state_prim0,
            state_prim1,
            state_exterior_angle,
            state_src,
            state_src_power,
            eta_r,
            sigma_e,
            material_mu_r,
            gain,
            material_valid,
            thickness,
            params["grid_axis"],
            params["grid_position"],
            params["grid_coord0_min"],
            params["grid_coord0_max"],
            params["grid_coord1_min"],
            params["grid_coord1_max"],
            params["grid_resolution0"],
            params["grid_resolution1"],
            params["wavelength"],
            params["grid_cell_area"],
            params["seed"],
            params["total_edge_length"],
            params["tx_pol"],
        )

    @staticmethod
    def setup_context(ctx, inputs, output):
        ctx.set_materialize_grads(False)
        primals = tuple(
            torch.autograd.forward_ad.unpack_dual(value).primal for value in inputs[:5]
        )
        frequency = inputs[5]
        ctx.frequency_meta = (
            (frequency.dtype, frequency.device)
            if isinstance(frequency, torch.Tensor)
            else None
        )
        ctx.params = inputs[23]
        ctx.anchor_shape = tuple(inputs[0].shape)
        saved = (*primals[1:], *inputs[6:23])
        ctx.save_for_backward(*saved)
        ctx.save_for_forward(*saved)

    @staticmethod
    @_ad_first_order_only
    def backward(ctx, grad_output):
        none_grads = (None,) * 24
        _ad_reject_fixed_inputs(
            "mc_utd_diffraction_tape_accumulate_ad",
            ctx.needs_input_grad,
            (
                (6, "tape_active"),
                (7, "tape_state"),
                (8, "tape_cell"),
                (9, "tape_u"),
                (10, "state_edge_pos"),
                (11, "state_edge_dir"),
                (12, "state_t_min"),
                (13, "state_t_max"),
                (14, "state_n0"),
                (15, "state_n1"),
                (18, "state_exterior_angle"),
                (19, "state_src"),
                (20, "state_src_power"),
                (21, "material_mu_r"),
                (22, "material_valid"),
            ),
        )
        if grad_output is None:
            return none_grads
        saved = ctx.saved_tensors
        (eta_r, sigma_e, gain, thickness) = saved[:4]
        tape_tensors = saved[4:8]
        state_tensors = saved[8:19]
        material_mu_r = saved[19]
        material_valid = saved[20]
        need_anchor = bool(ctx.needs_input_grad[0])
        need_eta = bool(ctx.needs_input_grad[1])
        need_sigma = bool(ctx.needs_input_grad[2])
        need_gain = bool(ctx.needs_input_grad[3])
        need_thickness = bool(ctx.needs_input_grad[4])
        need_materials = need_eta or need_sigma or need_gain or need_thickness
        need_frequency = bool(ctx.needs_input_grad[5])
        if not (need_anchor or need_materials or need_frequency):
            return none_grads
        params = ctx.params
        wavelength = float(params["wavelength"])
        gradients = mc_utd_diffraction_tape_accumulate_backward(
            tuple(tape_tensors),
            tuple(state_tensors),
            eta_r,
            sigma_e,
            material_mu_r,
            gain,
            material_valid,
            thickness,
            grad_output,
            tx_pol=params["tx_pol"],
            need_materials=need_materials,
            need_source=need_anchor,
            need_frequency=need_frequency,
            grid_axis=params["grid_axis"],
            grid_position=params["grid_position"],
            grid_resolution0=params["grid_resolution0"],
            grid_resolution1=params["grid_resolution1"],
            wavelength=wavelength,
            grid_cell_area=params["grid_cell_area"],
            seed=params["seed"],
            total_edge_length=params["total_edge_length"],
            wavelength_dfreq=-wavelength * wavelength / _LIGHT_SPEED_M_PER_S_AD,
        )
        grad_frequency = (
            _ad_frequency_grad(gradients[5], ctx.frequency_meta)
            if need_frequency
            else None
        )
        return (
            gradients[4] if need_anchor else None,
            gradients[0] if need_eta else None,
            gradients[1] if need_sigma else None,
            gradients[2] if need_gain else None,
            gradients[3] if need_thickness else None,
            grad_frequency,
        ) + (None,) * 18

    @staticmethod
    def jvp(ctx, t_anchor, t_eta, t_sigma, t_gain, t_thickness, t_frequency, *t_rest):
        _ad_reject_fixed_tangents(
            "mc_utd_diffraction_tape_accumulate_ad",
            (
                (t_rest[4], "state_edge_pos"),
                (t_rest[5], "state_edge_dir"),
                (t_rest[8], "state_n0"),
                (t_rest[9], "state_n1"),
                (t_rest[13], "state_src"),
                (t_rest[14], "state_src_power"),
            ),
        )
        saved = ctx.saved_tensors
        face_shape = tuple(saved[0].shape)
        tangent_eta = _ad_checked_tangent(
            "mc_utd_diffraction_tape_accumulate_ad tangent_eta_r",
            _ad_native_tangent_or_none(t_eta),
            face_shape,
        )
        tangent_sigma = _ad_checked_tangent(
            "mc_utd_diffraction_tape_accumulate_ad tangent_sigma_e",
            _ad_native_tangent_or_none(t_sigma),
            face_shape,
        )
        tangent_gain = _ad_checked_tangent(
            "mc_utd_diffraction_tape_accumulate_ad tangent_gain",
            _ad_native_tangent_or_none(t_gain),
            face_shape,
        )
        tangent_thickness = _ad_checked_tangent(
            "mc_utd_diffraction_tape_accumulate_ad tangent_thickness",
            _ad_native_tangent_or_none(t_thickness),
            face_shape,
        )
        tangent_anchor = _ad_native_tangent_or_none(
            t_anchor if isinstance(t_anchor, torch.Tensor) else None
        )
        tangent_frequency = _ad_frequency_tangent(t_frequency)
        params = ctx.params
        if (
            tangent_eta is None
            and tangent_sigma is None
            and tangent_gain is None
            and tangent_thickness is None
            and tangent_anchor is None
            and tangent_frequency == 0.0
        ):
            return None
        (eta_r, sigma_e, gain, thickness) = saved[:4]
        tape_tensors = tuple(_ad_native_tensor(value) for value in saved[4:8])
        state_tensors = tuple(_ad_native_tensor(value) for value in saved[8:19])
        wavelength = float(params["wavelength"])
        with disable_functorch():
            return mc_utd_diffraction_tape_accumulate_jvp(
                tape_tensors,
                state_tensors,
                _ad_native_tensor(eta_r),
                _ad_native_tensor(sigma_e),
                _ad_native_tensor(saved[19]),
                _ad_native_tensor(gain),
                _ad_native_tensor(saved[20]),
                _ad_native_tensor(thickness),
                tangent_eta,
                tangent_sigma,
                tangent_gain,
                tangent_thickness,
                tangent_anchor,
                tx_pol=params["tx_pol"],
                grid_axis=params["grid_axis"],
                grid_position=params["grid_position"],
                grid_resolution0=params["grid_resolution0"],
                grid_resolution1=params["grid_resolution1"],
                wavelength=wavelength,
                grid_cell_area=params["grid_cell_area"],
                seed=params["seed"],
                total_edge_length=params["total_edge_length"],
                wavelength_tangent=(-wavelength * wavelength / _LIGHT_SPEED_M_PER_S_AD)
                * tangent_frequency,
            )


def mc_utd_diffraction_tape_accumulate_ad(
    tx_anchor: torch.Tensor,
    material_eta_r: torch.Tensor,
    material_sigma: torch.Tensor,
    material_gain: torch.Tensor,
    material_thickness: torch.Tensor,
    frequency: torch.Tensor | float,
    tape_tensors: tuple[torch.Tensor, ...],
    state_tensors: tuple[torch.Tensor, ...],
    material_mu_r: torch.Tensor,
    material_valid: torch.Tensor,
    *,
    tx_pol: torch.Tensor,
    grid_axis: int,
    grid_position: float,
    grid_coord0_min: float,
    grid_coord0_max: float,
    grid_coord1_min: float,
    grid_coord1_max: float,
    grid_resolution0: int,
    grid_resolution1: int,
    wavelength: float,
    grid_cell_area: float,
    seed: int,
    total_edge_length: float,
) -> torch.Tensor:
    """Differentiable :func:`mc_utd_diffraction_tape_accumulate` (one tx)."""

    if len(tape_tensors) != 4:
        raise ValueError("tape_tensors must hold (active, state, cell, u)")
    if len(state_tensors) != 11:
        raise ValueError("state_tensors must hold the 11 packed state tables")
    params = {
        "tx_pol": tx_pol,
        "grid_axis": int(grid_axis),
        "grid_position": float(grid_position),
        "grid_coord0_min": float(grid_coord0_min),
        "grid_coord0_max": float(grid_coord0_max),
        "grid_coord1_min": float(grid_coord1_min),
        "grid_coord1_max": float(grid_coord1_max),
        "grid_resolution0": int(grid_resolution0),
        "grid_resolution1": int(grid_resolution1),
        "wavelength": float(wavelength),
        "grid_cell_area": float(grid_cell_area),
        "seed": int(seed),
        "total_edge_length": float(total_edge_length),
    }
    return _McDiffractionMapAdFunction.apply(
        tx_anchor,
        material_eta_r,
        material_sigma,
        material_gain,
        material_thickness,
        frequency,
        *tape_tensors,
        *state_tensors,
        material_mu_r,
        material_valid,
        params,
    )


# -------------------------------------------------------------------------
# basic sampling
# -------------------------------------------------------------------------
def mc_reflection_launch_inputs(
    tx_positions: torch.Tensor,
    *,
    tx_index: int,
    sample_count: int,
) -> dict[str, torch.Tensor]:
    validate_cuda_tensor(
        "tx_positions", tx_positions, dtype=torch.float32, ndim=2, trailing_shape=(3,)
    )
    if sample_count < 0:
        raise ValueError("sample_count must be non-negative")
    exported = required_symbol("mc_reflection_launch_inputs")(
        tx_positions, int(tx_index), int(sample_count)
    )
    if not isinstance(exported, dict):
        raise TypeError(
            "_channel.mc_reflection_launch_inputs must return a dict"
        )
    validate_cuda_tensor(
        "ray_o", exported["ray_o"], dtype=torch.float32, ndim=2, trailing_shape=(3,)
    )
    validate_cuda_tensor("ray_tmax", exported["ray_tmax"], dtype=torch.float32, ndim=1)
    validate_cuda_tensor("active", exported["active"], dtype=torch.bool, ndim=1)
    validate_cuda_tensor(
        "tx_pol", exported["tx_pol"], dtype=torch.float32, ndim=2, trailing_shape=(3,)
    )
    return exported


def _validate_mc_diffraction_discovery_args(
    args: tuple[torch.Tensor, ...], *, counted: bool
) -> None:
    expected = 16 if counted else 15
    if len(args) != expected:
        raise TypeError(f"MC diffraction discovery expects {expected} tensors")
    hit_count_index = 6 if counted else None
    offset = 1 if counted else 0
    names = (
        "tx_pos", "ray_dir", "prim_index", "hit_p", "hit_n", "hit_geo_n"
    )
    for index, name in enumerate(names):
        trailing_shape = (3,) if name not in {"prim_index"} else None
        validate_cuda_tensor(
            name,
            args[index],
            dtype=torch.int32 if name == "prim_index" else torch.float32,
            ndim=1 if name in {"tx_pos", "prim_index"} else 2,
            trailing_shape=trailing_shape,
        )
    if hit_count_index is not None:
        validate_cuda_tensor("hit_count", args[hit_count_index], dtype=torch.int32, ndim=1)
    table_start = 6 + offset
    validate_cuda_tensor(
        "triangle_edge_count", args[table_start], dtype=torch.int32, ndim=1
    )
    validate_cuda_tensor(
        "triangle_edge_indices", args[table_start + 1], dtype=torch.int32, ndim=2
    )
    for index, name in enumerate(("edge_pos", "edge_dir", "edge_n0", "edge_n1"), start=2):
        validate_cuda_tensor(
            name,
            args[table_start + index],
            dtype=torch.float32,
            ndim=2,
            trailing_shape=(3,),
        )
    for index, name, dtype in (
        (6, "edge_line_min", torch.float32),
        (7, "edge_line_max", torch.float32),
        (8, "edge_adjacent_face1", torch.int32),
    ):
        validate_cuda_tensor(name, args[table_start + index], dtype=dtype, ndim=1)


def mc_diffraction_discover_edges(*args: torch.Tensor) -> torch.Tensor:
    _validate_mc_diffraction_discovery_args(args, counted=False)
    out = required_symbol("mc_diffraction_discover_edges")(*args)
    if not isinstance(out, torch.Tensor):
        raise TypeError(
            "_channel.mc_diffraction_discover_edges must return a tensor"
        )
    return out


def mc_diffraction_discover_edges_counted(*args: torch.Tensor) -> torch.Tensor:
    _validate_mc_diffraction_discovery_args(args, counted=True)
    out = required_symbol("mc_diffraction_discover_edges_counted")(*args)
    if not isinstance(out, torch.Tensor):
        raise TypeError(
            "_channel.mc_diffraction_discover_edges_counted must return a tensor"
        )
    return out


def mc_diffraction_state_wi(
    state_edge_pos: torch.Tensor, state_src: torch.Tensor
) -> torch.Tensor:
    validate_cuda_tensor(
        "state_edge_pos",
        state_edge_pos,
        dtype=torch.float32,
        ndim=2,
        trailing_shape=(3,),
    )
    validate_cuda_tensor(
        "state_src", state_src, dtype=torch.float32, ndim=2, trailing_shape=(3,)
    )
    if state_src.shape != state_edge_pos.shape:
        raise ValueError("state_src must match state_edge_pos shape")
    state_wi = required_symbol("mc_diffraction_state_wi")(state_edge_pos, state_src)
    if not isinstance(state_wi, torch.Tensor):
        raise TypeError("_channel.mc_diffraction_state_wi must return a tensor")
    validate_cuda_tensor(
        "state_wi", state_wi, dtype=torch.float32, ndim=2, trailing_shape=(3,)
    )
    return state_wi


def mc_diffraction_state_pack(
    edge_indices: torch.Tensor,
    edge_pos: torch.Tensor,
    edge_dir: torch.Tensor,
    line_min: torch.Tensor,
    line_max: torch.Tensor,
    n0: torch.Tensor,
    n1: torch.Tensor,
    face0: torch.Tensor,
    face1: torch.Tensor,
    exterior_angle: torch.Tensor,
    tx: torch.Tensor,
    tx_power: torch.Tensor,
) -> tuple[torch.Tensor, ...]:
    validate_cuda_tensor("edge_indices", edge_indices, dtype=torch.int32, ndim=1)
    validate_cuda_tensor(
        "edge_pos", edge_pos, dtype=torch.float32, ndim=2, trailing_shape=(3,)
    )
    validate_cuda_tensor(
        "edge_dir", edge_dir, dtype=torch.float32, ndim=2, trailing_shape=(3,)
    )
    validate_cuda_tensor("line_min", line_min, dtype=torch.float32, ndim=1)
    validate_cuda_tensor("line_max", line_max, dtype=torch.float32, ndim=1)
    validate_cuda_tensor("n0", n0, dtype=torch.float32, ndim=2, trailing_shape=(3,))
    validate_cuda_tensor("n1", n1, dtype=torch.float32, ndim=2, trailing_shape=(3,))
    validate_cuda_tensor("face0", face0, dtype=torch.int32, ndim=1)
    validate_cuda_tensor("face1", face1, dtype=torch.int32, ndim=1)
    validate_cuda_tensor("exterior_angle", exterior_angle, dtype=torch.float32, ndim=1)
    validate_cuda_tensor("tx", tx, dtype=torch.float32, ndim=1)
    validate_cuda_tensor("tx_power", tx_power, dtype=torch.float32, ndim=0)
    if tx.shape[0] != 3:
        raise ValueError("tx must have shape (3,)")
    states = required_symbol("mc_diffraction_state_pack")(
        edge_indices,
        edge_pos,
        edge_dir,
        line_min,
        line_max,
        n0,
        n1,
        face0,
        face1,
        exterior_angle,
        tx,
        tx_power,
    )
    if not isinstance(states, tuple) or len(states) != 12:
        raise TypeError(
            "_channel.mc_diffraction_state_pack must return 12 tensors"
        )
    validate_cuda_tensor("state_edge_index", states[0], dtype=torch.int32, ndim=1)
    validate_cuda_tensor(
        "state_edge_pos", states[1], dtype=torch.float32, ndim=2, trailing_shape=(3,)
    )
    validate_cuda_tensor(
        "state_edge_dir", states[2], dtype=torch.float32, ndim=2, trailing_shape=(3,)
    )
    validate_cuda_tensor("state_line_min", states[3], dtype=torch.float32, ndim=1)
    validate_cuda_tensor("state_line_max", states[4], dtype=torch.float32, ndim=1)
    validate_cuda_tensor(
        "state_n0", states[5], dtype=torch.float32, ndim=2, trailing_shape=(3,)
    )
    validate_cuda_tensor(
        "state_n1", states[6], dtype=torch.float32, ndim=2, trailing_shape=(3,)
    )
    validate_cuda_tensor("state_face0", states[7], dtype=torch.int32, ndim=1)
    validate_cuda_tensor("state_face1", states[8], dtype=torch.int32, ndim=1)
    validate_cuda_tensor("state_exterior_angle", states[9], dtype=torch.float32, ndim=1)
    validate_cuda_tensor(
        "state_src", states[10], dtype=torch.float32, ndim=2, trailing_shape=(3,)
    )
    validate_cuda_tensor("state_src_power", states[11], dtype=torch.float32, ndim=1)
    return states


# -------------------------------------------------------------------------
# basic transmission
# -------------------------------------------------------------------------
_CONTRACT_FAILURE_BIT = int(CapacityFailureBit.PAIR_CONTRACT_ERROR)


@dataclass(frozen=True, slots=True)
class McTransmissionWallProduct:
    """Fixed-capacity resident outputs of the MC wall-product estimator."""

    scaled_power: torch.Tensor
    transmittance: torch.Tensor
    wall_count: torch.Tensor
    penetrated: torch.Tensor


def _validate_inputs(
    valid: torch.Tensor,
    num_hits: torch.Tensor,
    reached_target: torch.Tensor,
    direction: torch.Tensor,
    normal: torch.Tensor,
    global_primitive_id: torch.Tensor,
    face_material_id: torch.Tensor,
    geometry_mode_id: torch.Tensor,
    layer_offset: torch.Tensor,
    layer_count: torch.Tensor,
    layer_thickness_m: torch.Tensor,
    layer_eps_r: torch.Tensor,
    layer_sigma_e: torch.Tensor,
    layer_mu_r: torch.Tensor,
    pair_polarization: torch.Tensor,
    base_power: torch.Tensor,
    failure_state: CapacityFailureState,
    *,
    frequency_hz: float,
) -> tuple[int, int]:
    validate_cuda_tensor("valid", valid, dtype=torch.bool, ndim=2)
    rows, hit_capacity = valid.shape
    validate_cuda_tensor("num_hits", num_hits, dtype=torch.int32, ndim=1)
    validate_cuda_tensor("reached_target", reached_target, dtype=torch.bool, ndim=1)
    validate_cuda_tensor(
        "direction", direction, dtype=torch.float32, ndim=2, trailing_shape=(3,)
    )
    validate_cuda_tensor(
        "normal", normal, dtype=torch.float32, ndim=3, trailing_shape=(3,)
    )
    validate_cuda_tensor(
        "global_primitive_id", global_primitive_id, dtype=torch.int32, ndim=2
    )
    validate_cuda_tensor(
        "face_material_id", face_material_id, dtype=torch.int32, ndim=1
    )
    validate_cuda_tensor(
        "geometry_mode_id", geometry_mode_id, dtype=torch.int32, ndim=1
    )
    validate_cuda_tensor("layer_offset", layer_offset, dtype=torch.int32, ndim=1)
    validate_cuda_tensor("layer_count", layer_count, dtype=torch.int32, ndim=1)
    validate_cuda_tensor(
        "layer_thickness_m", layer_thickness_m, dtype=torch.float32, ndim=1
    )
    validate_cuda_tensor("layer_eps_r", layer_eps_r, dtype=torch.float32, ndim=1)
    validate_cuda_tensor("layer_sigma_e", layer_sigma_e, dtype=torch.float32, ndim=1)
    validate_cuda_tensor("layer_mu_r", layer_mu_r, dtype=torch.float32, ndim=1)
    validate_cuda_tensor(
        "pair_polarization",
        pair_polarization,
        dtype=torch.float32,
        ndim=2,
        trailing_shape=(3,),
    )
    validate_cuda_tensor("base_power", base_power, dtype=torch.float32, ndim=1)
    if num_hits.shape != (rows,):
        raise ValueError("num_hits must match valid rows")
    if reached_target.shape != (rows,):
        raise ValueError("reached_target must match valid rows")
    if direction.shape != (rows, 3):
        raise ValueError("direction must have shape (N, 3)")
    if normal.shape != (rows, hit_capacity, 3):
        raise ValueError("normal must have shape (N, D, 3)")
    if global_primitive_id.shape != valid.shape:
        raise ValueError("global_primitive_id must match valid")
    if pair_polarization.shape != (rows, 3):
        raise ValueError("pair_polarization must have shape (N, 3)")
    if base_power.shape != (rows,):
        raise ValueError("base_power must match valid rows")
    material_count = layer_offset.shape[0]
    if layer_count.shape != (material_count,):
        raise ValueError("layer_count must match layer_offset")
    if geometry_mode_id.shape != (material_count,):
        raise ValueError("geometry_mode_id must match material rows")
    layer_length = layer_thickness_m.shape[0]
    if any(
        tensor.shape != (layer_length,)
        for tensor in (layer_eps_r, layer_sigma_e, layer_mu_r)
    ):
        raise ValueError("layer property tensors must have one shared length")
    device = valid.device
    for name, tensor in (
        ("num_hits", num_hits),
        ("reached_target", reached_target),
        ("direction", direction),
        ("normal", normal),
        ("global_primitive_id", global_primitive_id),
        ("face_material_id", face_material_id),
        ("geometry_mode_id", geometry_mode_id),
        ("layer_offset", layer_offset),
        ("layer_count", layer_count),
        ("layer_thickness_m", layer_thickness_m),
        ("layer_eps_r", layer_eps_r),
        ("layer_sigma_e", layer_sigma_e),
        ("layer_mu_r", layer_mu_r),
        ("pair_polarization", pair_polarization),
        ("base_power", base_power),
    ):
        if tensor.device != device:
            raise ValueError(f"{name} must share the valid device")
    require_capacity_failure_state(failure_state, device=device)
    _validate_frequency_hz(frequency_hz)
    return int(rows), int(hit_capacity)


def _arguments(
    valid: torch.Tensor,
    num_hits: torch.Tensor,
    reached_target: torch.Tensor,
    direction: torch.Tensor,
    normal: torch.Tensor,
    global_primitive_id: torch.Tensor,
    face_material_id: torch.Tensor,
    geometry_mode_id: torch.Tensor,
    layer_offset: torch.Tensor,
    layer_count: torch.Tensor,
    layer_thickness_m: torch.Tensor,
    layer_eps_r: torch.Tensor,
    layer_sigma_e: torch.Tensor,
    layer_mu_r: torch.Tensor,
    pair_polarization: torch.Tensor,
    base_power: torch.Tensor,
    frequency_hz: float,
    failure_state: CapacityFailureState,
) -> tuple[object, ...]:
    return (
        valid,
        num_hits,
        reached_target,
        direction,
        normal,
        global_primitive_id,
        face_material_id,
        geometry_mode_id,
        layer_offset,
        layer_count,
        layer_thickness_m,
        layer_eps_r,
        layer_sigma_e,
        layer_mu_r,
        pair_polarization,
        base_power,
        float(frequency_hz),
        failure_state.bits,
        _CONTRACT_FAILURE_BIT,
    )


def _result(exported: object, *, rows: int) -> McTransmissionWallProduct:
    if not isinstance(exported, dict):
        raise TypeError("native MC transmission wall product must return a dict")
    scaled_power = exported["scaled_power"]
    transmittance = exported["transmittance"]
    wall_count = exported["wall_count"]
    penetrated = exported["penetrated"]
    validate_cuda_tensor("scaled_power", scaled_power, dtype=torch.float32, ndim=1)
    validate_cuda_tensor("transmittance", transmittance, dtype=torch.float32, ndim=1)
    validate_cuda_tensor("wall_count", wall_count, dtype=torch.int32, ndim=1)
    validate_cuda_tensor("penetrated", penetrated, dtype=torch.bool, ndim=1)
    if any(
        tensor.shape != (rows,)
        for tensor in (scaled_power, transmittance, wall_count, penetrated)
    ):
        raise ValueError("native MC transmission wall product returned wrong rows")
    return McTransmissionWallProduct(
        scaled_power=scaled_power,
        transmittance=transmittance,
        wall_count=wall_count,
        penetrated=penetrated,
    )


def mc_transmission_wall_product(
    valid: torch.Tensor,
    num_hits: torch.Tensor,
    reached_target: torch.Tensor,
    direction: torch.Tensor,
    normal: torch.Tensor,
    global_primitive_id: torch.Tensor,
    face_material_id: torch.Tensor,
    geometry_mode_id: torch.Tensor,
    layer_offset: torch.Tensor,
    layer_count: torch.Tensor,
    layer_thickness_m: torch.Tensor,
    layer_eps_r: torch.Tensor,
    layer_sigma_e: torch.Tensor,
    layer_mu_r: torch.Tensor,
    pair_polarization: torch.Tensor,
    base_power: torch.Tensor,
    failure_state: CapacityFailureState,
    *,
    frequency_hz: float,
) -> McTransmissionWallProduct:
    """Evaluate the live ADR-027 fixed-capacity MC estimator."""

    rows, _ = _validate_inputs(
        valid,
        num_hits,
        reached_target,
        direction,
        normal,
        global_primitive_id,
        face_material_id,
        geometry_mode_id,
        layer_offset,
        layer_count,
        layer_thickness_m,
        layer_eps_r,
        layer_sigma_e,
        layer_mu_r,
        pair_polarization,
        base_power,
        failure_state,
        frequency_hz=float(frequency_hz),
    )
    exported = required_symbol("mc_transmission_wall_product")(
        *_arguments(
            valid,
            num_hits,
            reached_target,
            direction,
            normal,
            global_primitive_id,
            face_material_id,
            geometry_mode_id,
            layer_offset,
            layer_count,
            layer_thickness_m,
            layer_eps_r,
            layer_sigma_e,
            layer_mu_r,
            pair_polarization,
            base_power,
            float(frequency_hz),
            failure_state,
        )
    )
    return _result(exported, rows=rows)


def mc_transmission_wall_product_backward(
    *inputs: torch.Tensor,
    frequency_hz: float,
    failure_state: CapacityFailureState,
    grad_scaled_power: torch.Tensor | None,
    grad_transmittance: torch.Tensor | None,
) -> tuple[torch.Tensor, ...]:
    if len(inputs) != 16:
        raise ValueError("MC transmission wall-product backward requires 16 inputs")
    rows, _ = _validate_inputs(*inputs, failure_state, frequency_hz=float(frequency_hz))
    for name, gradient in (
        ("grad_scaled_power", grad_scaled_power),
        ("grad_transmittance", grad_transmittance),
    ):
        if gradient is not None:
            validate_cuda_tensor(
                name,
                gradient,
                dtype=torch.float32,
                ndim=1,
                require_contiguous=False,
            )
            if gradient.shape != (rows,):
                raise ValueError(f"{name} must match the output rows")
    exported = required_symbol("mc_transmission_wall_product_backward")(
        *_arguments(*inputs, float(frequency_hz), failure_state),
        grad_scaled_power,
        grad_transmittance,
    )
    if not isinstance(exported, tuple) or len(exported) != 7:
        raise TypeError(
            "native MC transmission wall-product backward must return 7 tensors"
        )
    expected = (
        (torch.float32, inputs[3].shape),
        (torch.float32, inputs[4].shape),
        (torch.float32, inputs[10].shape),
        (torch.float32, inputs[11].shape),
        (torch.float32, inputs[12].shape),
        (torch.float32, inputs[15].shape),
        (torch.float32, (1,)),
    )
    for index, (tensor, (dtype, shape)) in enumerate(
        zip(exported, expected, strict=True)
    ):
        validate_cuda_tensor(f"gradient[{index}]", tensor, dtype=dtype, ndim=len(shape))
        if tensor.shape != shape:
            raise ValueError(f"gradient[{index}] has the wrong shape")
    return exported


def mc_transmission_wall_product_jvp(
    *inputs: torch.Tensor,
    frequency_hz: float,
    failure_state: CapacityFailureState,
    tangent_direction: torch.Tensor | None,
    tangent_normal: torch.Tensor | None,
    tangent_layer_thickness_m: torch.Tensor | None,
    tangent_layer_eps_r: torch.Tensor | None,
    tangent_layer_sigma_e: torch.Tensor | None,
    tangent_base_power: torch.Tensor | None,
    tangent_frequency: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    if len(inputs) != 16:
        raise ValueError("MC transmission wall-product JVP requires 16 inputs")
    rows, _ = _validate_inputs(*inputs, failure_state, frequency_hz=float(frequency_hz))
    exported = required_symbol("mc_transmission_wall_product_jvp")(
        *_arguments(*inputs, float(frequency_hz), failure_state),
        tangent_direction,
        tangent_normal,
        tangent_layer_thickness_m,
        tangent_layer_eps_r,
        tangent_layer_sigma_e,
        tangent_base_power,
        float(tangent_frequency),
    )
    if not isinstance(exported, dict):
        raise TypeError("native MC transmission wall-product JVP must return a dict")
    scaled_power = exported["scaled_power"]
    transmittance = exported["transmittance"]
    for name, tensor in (
        ("scaled_power", scaled_power),
        ("transmittance", transmittance),
    ):
        validate_cuda_tensor(name, tensor, dtype=torch.float32, ndim=1)
        if tensor.shape != (rows,):
            raise ValueError(
                f"native MC transmission wall-product JVP {name} has wrong rows"
            )
    return scaled_power, transmittance


class _McTransmissionWallProductAd(torch.autograd.Function):
    @staticmethod
    def forward(*inputs):
        frequency_value = inputs[17]
        failure_state = CapacityFailureState(bits=inputs[18])
        result = mc_transmission_wall_product(
            *inputs[:16], failure_state, frequency_hz=frequency_value
        )
        return (
            result.scaled_power,
            result.transmittance,
            result.wall_count,
            result.penetrated,
        )

    @staticmethod
    def setup_context(ctx, inputs, output):
        ctx.set_materialize_grads(False)
        ctx.mark_non_differentiable(output[2], output[3])
        ctx.frequency_value = float(inputs[17])
        ctx.frequency_meta = (
            (inputs[16].dtype, inputs[16].device)
            if isinstance(inputs[16], torch.Tensor)
            else None
        )
        primals = tuple(
            torch.autograd.forward_ad.unpack_dual(value).primal
            for value in (*inputs[:16], inputs[18])
        )
        ctx.save_for_backward(*primals)
        ctx.save_for_forward(*primals)

    @staticmethod
    @_ad_first_order_only
    def backward(ctx, grad_scaled, grad_transmittance, _grad_count, _grad_penetrated):
        _ad_reject_fixed_inputs(
            "mc_transmission_wall_product_ad",
            ctx.needs_input_grad,
            tuple(
                (index, name)
                for index, name in (
                    (0, "valid"),
                    (1, "num_hits"),
                    (2, "reached_target"),
                    (5, "global_primitive_id"),
                    (6, "face_material_id"),
                    (7, "geometry_mode_id"),
                    (8, "layer_offset"),
                    (9, "layer_count"),
                    (13, "layer_mu_r"),
                    (14, "pair_polarization"),
                    (18, "capacity_failure_state"),
                )
            ),
        )
        live = (3, 4, 10, 11, 12, 15, 16)
        if (grad_scaled is None and grad_transmittance is None) or not any(
            ctx.needs_input_grad[index] for index in live
        ):
            return (None,) * 19
        saved = ctx.saved_tensors
        inputs = saved[:16]
        gradients = mc_transmission_wall_product_backward(
            *inputs,
            frequency_hz=ctx.frequency_value,
            failure_state=CapacityFailureState(bits=saved[16]),
            grad_scaled_power=grad_scaled,
            grad_transmittance=grad_transmittance,
        )
        returned: list[torch.Tensor | None] = [None] * 19
        for input_index, gradient_index in zip(live[:-1], range(6), strict=True):
            if ctx.needs_input_grad[input_index]:
                returned[input_index] = gradients[gradient_index]
        if ctx.needs_input_grad[16]:
            returned[16] = _ad_frequency_grad(gradients[6], ctx.frequency_meta)
        return tuple(returned)

    @staticmethod
    def jvp(ctx, *tangents):
        _ad_reject_fixed_tangents(
            "mc_transmission_wall_product_ad",
            tuple(
                (tangents[index], name)
                for index, name in (
                    (0, "valid"),
                    (1, "num_hits"),
                    (2, "reached_target"),
                    (5, "global_primitive_id"),
                    (6, "face_material_id"),
                    (7, "geometry_mode_id"),
                    (8, "layer_offset"),
                    (9, "layer_count"),
                    (13, "layer_mu_r"),
                    (14, "pair_polarization"),
                    (18, "capacity_failure_state"),
                )
            ),
        )
        saved = ctx.saved_tensors
        inputs = tuple(_ad_native_tensor(value) for value in saved[:16])
        tangent_direction = _ad_geometry_tangent(
            "tangent_direction", tangents[3], inputs[3]
        )
        tangent_normal = _ad_geometry_tangent("tangent_normal", tangents[4], inputs[4])
        continuous_tangents = tuple(
            _ad_checked_tangent(
                name, _ad_native_tangent_or_none(tangent), tuple(primal.shape)
            )
            for name, tangent, primal in (
                ("tangent_layer_thickness_m", tangents[10], inputs[10]),
                ("tangent_layer_eps_r", tangents[11], inputs[11]),
                ("tangent_layer_sigma_e", tangents[12], inputs[12]),
                ("tangent_base_power", tangents[15], inputs[15]),
            )
        )
        tangent_frequency = _ad_frequency_tangent(tangents[16])
        if (
            tangent_direction is None
            and tangent_normal is None
            and all(value is None for value in continuous_tangents)
            and tangent_frequency == 0.0
        ):
            return None, None, None, None
        with disable_functorch():
            scaled, transmittance = mc_transmission_wall_product_jvp(
                *inputs,
                frequency_hz=ctx.frequency_value,
                failure_state=CapacityFailureState(bits=_ad_native_tensor(saved[16])),
                tangent_direction=tangent_direction,
                tangent_normal=tangent_normal,
                tangent_layer_thickness_m=continuous_tangents[0],
                tangent_layer_eps_r=continuous_tangents[1],
                tangent_layer_sigma_e=continuous_tangents[2],
                tangent_base_power=continuous_tangents[3],
                tangent_frequency=tangent_frequency,
            )
        return scaled, transmittance, None, None


def mc_transmission_wall_product_ad(
    valid: torch.Tensor,
    num_hits: torch.Tensor,
    reached_target: torch.Tensor,
    direction: torch.Tensor,
    normal: torch.Tensor,
    global_primitive_id: torch.Tensor,
    face_material_id: torch.Tensor,
    geometry_mode_id: torch.Tensor,
    layer_offset: torch.Tensor,
    layer_count: torch.Tensor,
    layer_thickness_m: torch.Tensor,
    layer_eps_r: torch.Tensor,
    layer_sigma_e: torch.Tensor,
    layer_mu_r: torch.Tensor,
    pair_polarization: torch.Tensor,
    base_power: torch.Tensor,
    frequency: torch.Tensor | float,
    failure_state: CapacityFailureState,
    *,
    frequency_value: float | None = None,
) -> McTransmissionWallProduct:
    """Differentiable fixed-topology wall product with native VJP/JVP."""

    if frequency_value is None:
        frequency_value = _ad_frequency_value(frequency)
    values = _McTransmissionWallProductAd.apply(
        valid,
        num_hits,
        reached_target,
        direction,
        normal,
        global_primitive_id,
        face_material_id,
        geometry_mode_id,
        layer_offset,
        layer_count,
        layer_thickness_m,
        layer_eps_r,
        layer_sigma_e,
        layer_mu_r,
        pair_polarization,
        base_power,
        frequency,
        float(frequency_value),
        failure_state.bits,
    )
    return McTransmissionWallProduct(*values)


def _validate_frequency_hz(frequency_hz: float) -> None:
    if not math.isfinite(frequency_hz) or frequency_hz <= 0.0:
        raise ValueError("frequency_hz must be finite and positive")


__all__ = [
    "McTransmissionWallProduct",
    "mc_transmission_wall_product",
    "mc_transmission_wall_product_ad",
    "mc_transmission_wall_product_backward",
    "mc_transmission_wall_product_jvp",
]


# -------------------------------------------------------------------------
# bdpt maps
# -------------------------------------------------------------------------
def bdpt_store_point_component_column(
    target: torch.Tensor,
    source: torch.Tensor,
    *,
    rx_index: int,
) -> torch.Tensor:
    validate_cuda_tensor("target", target, dtype=torch.float32, ndim=2)
    validate_cuda_tensor("source", source, dtype=torch.float32, ndim=3)
    if source.shape[0] != target.shape[0] or source.shape[1:] != (1, 1):
        raise ValueError("source must have shape (tx, 1, 1)")
    if rx_index < 0 or rx_index >= target.shape[1]:
        raise ValueError("rx_index is out of range")
    native = _required_native_op("bdpt_store_point_component_column")
    out = native(target, source, int(rx_index))
    if not isinstance(out, torch.Tensor):
        raise TypeError(
            "_channel.bdpt_store_point_component_column must return a tensor"
        )
    validate_cuda_tensor("out", out, dtype=torch.float32, ndim=2)
    if out.shape != target.shape:
        raise ValueError(
            "_channel.bdpt_store_point_component_column returned an unexpected shape"
        )
    return out


def bdpt_finalize_point_components(
    los: torch.Tensor,
    reflection: torch.Tensor,
    diffraction: torch.Tensor,
    transmission: torch.Tensor,
    scattering: torch.Tensor,
) -> dict[str, torch.Tensor]:
    validate_cuda_tensor("los", los, dtype=torch.float32, ndim=2)
    validate_cuda_tensor("reflection", reflection, dtype=torch.float32, ndim=2)
    validate_cuda_tensor("diffraction", diffraction, dtype=torch.float32, ndim=2)
    validate_cuda_tensor("transmission", transmission, dtype=torch.float32, ndim=2)
    validate_cuda_tensor("scattering", scattering, dtype=torch.float32, ndim=2)
    if (
        reflection.shape != los.shape
        or diffraction.shape != los.shape
        or transmission.shape != los.shape
        or scattering.shape != los.shape
    ):
        raise ValueError("point component matrices must have matching shapes")
    exported = _required_native_op("bdpt_finalize_point_components")(
        los, reflection, diffraction, transmission, scattering
    )
    if not isinstance(exported, dict):
        raise TypeError(
            "_channel.bdpt_finalize_point_components must return a dict"
        )
    validate_cuda_tensor(
        "path_gain", exported["path_gain"], dtype=torch.float32, ndim=2
    )
    if exported["path_gain"].shape != los.shape:
        raise ValueError(
            "_channel.bdpt_finalize_point_components returned bad path_gain shape"
        )
    for name in (
        "los_power",
        "reflection_power",
        "diffraction_power",
        "transmission_power",
        "scattering_power",
    ):
        validate_cuda_tensor(name, exported[name], dtype=torch.float32, ndim=0)
    return exported


def bdpt_point_component_power(
    path_gain: torch.Tensor, *, include_los: bool
) -> dict[str, torch.Tensor]:
    validate_cuda_tensor("path_gain", path_gain, dtype=torch.float32, ndim=2)
    exported = _required_native_op("bdpt_point_component_power")(
        path_gain, bool(include_los)
    )
    if not isinstance(exported, dict):
        raise TypeError("_channel.bdpt_point_component_power must return a dict")
    for name in ("los", "reflection", "diffraction"):
        validate_cuda_tensor(name, exported[name], dtype=torch.float32, ndim=0)
    return exported


def bdpt_transmitter_tensors(
    flat_positions: tuple[float, ...],
    powers: tuple[float, ...],
) -> dict[str, torch.Tensor]:
    if len(flat_positions) % 3 != 0:
        raise ValueError("flat_positions must contain xyz triples")
    if len(flat_positions) // 3 != len(powers):
        raise ValueError("powers must match flat_positions")
    exported = _required_native_op("bdpt_transmitter_tensors")(flat_positions, powers)
    if not isinstance(exported, dict):
        raise TypeError("_channel.bdpt_transmitter_tensors must return a dict")
    validate_cuda_tensor(
        "positions",
        exported["positions"],
        dtype=torch.float32,
        ndim=2,
        trailing_shape=(3,),
    )
    validate_cuda_tensor("power", exported["power"], dtype=torch.float32, ndim=1)
    return exported


def bdpt_host_vec3_tensor(flat_positions: tuple[float, ...]) -> torch.Tensor:
    if len(flat_positions) % 3 != 0:
        raise ValueError("flat_positions must contain xyz triples")
    powers = tuple(1.0 for _ in range(len(flat_positions) // 3))
    return bdpt_transmitter_tensors(flat_positions, powers)["positions"]


def bdpt_receiver_grid_points(
    reference: torch.Tensor,
    *,
    origin: tuple[float, float, float],
    x_axis: tuple[float, float, float],
    y_axis: tuple[float, float, float],
    shape: tuple[int, int],
    spacing: tuple[float, float],
) -> torch.Tensor:
    validate_cuda_tensor("reference", reference, dtype=torch.float32, ndim=2)
    rows, cols = shape
    if rows < 0 or cols < 0:
        raise ValueError("shape entries must be non-negative")
    if spacing[0] <= 0.0 or spacing[1] <= 0.0:
        raise ValueError("spacing entries must be positive")
    points = _required_native_op("bdpt_receiver_grid_points")(
        reference,
        int(rows),
        int(cols),
        float(origin[0]),
        float(origin[1]),
        float(origin[2]),
        float(x_axis[0]),
        float(x_axis[1]),
        float(x_axis[2]),
        float(y_axis[0]),
        float(y_axis[1]),
        float(y_axis[2]),
        float(spacing[0]),
        float(spacing[1]),
    )
    if not isinstance(points, torch.Tensor):
        raise TypeError(
            "_channel.bdpt_receiver_grid_points must return a tensor"
        )
    validate_cuda_tensor(
        "points", points, dtype=torch.float32, ndim=2, trailing_shape=(3,)
    )
    if points.shape[0] != rows * cols:
        raise ValueError(
            "_channel.bdpt_receiver_grid_points returned an unexpected shape"
        )
    return points


def bdpt_los_export(
    tx_positions: torch.Tensor,
    tx_power: torch.Tensor,
    rx_positions: torch.Tensor,
    tx_polarizations: torch.Tensor,
    *,
    frequency_hz: float,
) -> dict[str, torch.Tensor]:
    validate_cuda_tensor(
        "tx_positions", tx_positions, dtype=torch.float32, ndim=2, trailing_shape=(3,)
    )
    validate_cuda_tensor("tx_power", tx_power, dtype=torch.float32, ndim=1)
    validate_cuda_tensor(
        "rx_positions", rx_positions, dtype=torch.float32, ndim=2, trailing_shape=(3,)
    )
    validate_cuda_tensor(
        "tx_polarizations",
        tx_polarizations,
        dtype=torch.float32,
        ndim=2,
        trailing_shape=(3,),
    )
    if tx_power.shape[0] != tx_positions.shape[0]:
        raise ValueError("tx_power must have one value per transmitter")
    if tx_polarizations.shape != tx_positions.shape:
        raise ValueError("tx_polarizations must match tx_positions shape")
    if frequency_hz <= 0.0:
        raise ValueError("frequency_hz must be positive")
    exported = _required_native_op("bdpt_los_export")(
        tx_positions,
        tx_power,
        rx_positions,
        float(frequency_hz),
        tx_polarizations,
    )
    if not isinstance(exported, dict):
        raise TypeError("_channel.bdpt_los_export must return a dict")
    validate_cuda_tensor(
        "path_gain_matrix", exported["path_gain_matrix"], dtype=torch.float32, ndim=2
    )
    return exported


def bdpt_los_component_maps(los: torch.Tensor) -> torch.Tensor:
    validate_cuda_tensor("los", los, dtype=torch.float32, ndim=3)
    maps = _required_native_op("bdpt_los_component_maps")(los)
    if not isinstance(maps, torch.Tensor):
        raise TypeError("_channel.bdpt_los_component_maps must return a tensor")
    validate_cuda_tensor("maps", maps, dtype=torch.float32, ndim=3)
    if maps.shape != los.shape:
        raise ValueError(
            "_channel.bdpt_los_component_maps returned an unexpected shape"
        )
    return maps


def bdpt_los_component_maps_from_matrix(
    los: torch.Tensor, *, rows: int, cols: int
) -> torch.Tensor:
    validate_cuda_tensor("los", los, dtype=torch.float32, ndim=2)
    if rows < 0 or cols < 0:
        raise ValueError("rows and cols must be non-negative")
    if los.shape[1] != int(rows) * int(cols):
        raise ValueError("los columns must match rows * cols")
    maps = _required_native_op("bdpt_los_component_maps_from_matrix")(
        los, int(rows), int(cols)
    )
    if not isinstance(maps, torch.Tensor):
        raise TypeError(
            "_channel.bdpt_los_component_maps_from_matrix must return a tensor"
        )
    validate_cuda_tensor("maps", maps, dtype=torch.float32, ndim=3)
    if maps.shape != (los.shape[0], int(cols), int(rows)):
        raise ValueError(
            "_channel.bdpt_los_component_maps_from_matrix returned an unexpected shape"
        )
    return maps


def bdpt_los_visibility_inputs(
    tx_positions: torch.Tensor,
    *,
    tx_index: int,
    rx_count: int,
) -> dict[str, torch.Tensor]:
    validate_cuda_tensor(
        "tx_positions", tx_positions, dtype=torch.float32, ndim=2, trailing_shape=(3,)
    )
    if rx_count < 0:
        raise ValueError("rx_count must be non-negative")
    exported = _required_native_op("bdpt_los_visibility_inputs")(
        tx_positions, int(tx_index), int(rx_count)
    )
    if not isinstance(exported, dict):
        raise TypeError("_channel.bdpt_los_visibility_inputs must return a dict")
    validate_cuda_tensor(
        "start", exported["start"], dtype=torch.float32, ndim=2, trailing_shape=(3,)
    )
    validate_cuda_tensor("active", exported["active"], dtype=torch.bool, ndim=1)
    return exported


def bdpt_apply_los_visibility(
    maps: torch.Tensor,
    los: torch.Tensor,
    visible: torch.Tensor,
    *,
    tx_index: int,
) -> torch.Tensor:
    validate_cuda_tensor("maps", maps, dtype=torch.float32, ndim=3)
    validate_cuda_tensor("los", los, dtype=torch.float32, ndim=2)
    validate_cuda_tensor("visible", visible, dtype=torch.bool, ndim=1)
    if maps.shape[0] != los.shape[0] or los.shape[1] != maps.shape[1] * maps.shape[2]:
        raise ValueError("los must match maps flattened grid layout")
    out = _required_native_op("bdpt_apply_los_visibility")(
        maps, los, visible, int(tx_index)
    )
    if not isinstance(out, torch.Tensor):
        raise TypeError(
            "_channel.bdpt_apply_los_visibility must return a tensor"
        )
    validate_cuda_tensor("maps", out, dtype=torch.float32, ndim=3)
    return out


def bdpt_component_map_buffer(
    reference: torch.Tensor,
    *,
    tx_count: int,
    dim0: int,
    dim1: int,
) -> torch.Tensor:
    validate_cuda_tensor("reference", reference, dtype=torch.float32, ndim=2)
    if tx_count < 0 or dim0 < 0 or dim1 < 0:
        raise ValueError("tx_count, dim0, and dim1 must be non-negative")
    maps = _required_native_op("bdpt_component_map_buffer")(
        reference, int(tx_count), int(dim0), int(dim1)
    )
    if not isinstance(maps, torch.Tensor):
        raise TypeError(
            "_channel.bdpt_component_map_buffer must return a tensor"
        )
    validate_cuda_tensor("maps", maps, dtype=torch.float32, ndim=3)
    if maps.shape != (tx_count, dim0, dim1):
        raise ValueError(
            "_channel.bdpt_component_map_buffer returned an unexpected shape"
        )
    return maps


def bdpt_store_component_map(
    maps: torch.Tensor,
    source: torch.Tensor,
    *,
    tx_index: int,
) -> torch.Tensor:
    validate_cuda_tensor("maps", maps, dtype=torch.float32, ndim=3)
    validate_cuda_tensor("source", source, dtype=torch.float32, ndim=2)
    out = _required_native_op("bdpt_store_component_map")(maps, source, int(tx_index))
    if not isinstance(out, torch.Tensor):
        raise TypeError("_channel.bdpt_store_component_map must return a tensor")
    validate_cuda_tensor("maps", out, dtype=torch.float32, ndim=3)
    return out


def bdpt_store_scaled_component_map(
    maps: torch.Tensor,
    source: torch.Tensor,
    scale_values: torch.Tensor,
    *,
    tx_index: int,
    scale_index: int,
) -> torch.Tensor:
    validate_cuda_tensor("maps", maps, dtype=torch.float32, ndim=3)
    validate_cuda_tensor("source", source, dtype=torch.float32, ndim=2)
    validate_cuda_tensor("scale_values", scale_values, dtype=torch.float32, ndim=1)
    out = _required_native_op("bdpt_store_scaled_component_map")(
        maps,
        source,
        scale_values,
        int(tx_index),
        int(scale_index),
    )
    if not isinstance(out, torch.Tensor):
        raise TypeError(
            "_channel.bdpt_store_scaled_component_map must return a tensor"
        )
    validate_cuda_tensor("maps", out, dtype=torch.float32, ndim=3)
    return out


def bdpt_finalize_component_maps(
    los: torch.Tensor,
    reflection: torch.Tensor,
    diffraction: torch.Tensor,
    transmission: torch.Tensor,
    scattering: torch.Tensor,
) -> dict[str, torch.Tensor]:
    validate_cuda_tensor("los", los, dtype=torch.float32, ndim=3)
    validate_cuda_tensor("reflection", reflection, dtype=torch.float32, ndim=3)
    validate_cuda_tensor("diffraction", diffraction, dtype=torch.float32, ndim=3)
    validate_cuda_tensor("transmission", transmission, dtype=torch.float32, ndim=3)
    validate_cuda_tensor("scattering", scattering, dtype=torch.float32, ndim=3)
    if (
        reflection.shape != los.shape
        or diffraction.shape != los.shape
        or transmission.shape != los.shape
        or scattering.shape != los.shape
    ):
        raise ValueError("component maps must share shape")
    exported = _required_native_op("bdpt_finalize_component_maps")(
        los, reflection, diffraction, transmission, scattering
    )
    if not isinstance(exported, dict):
        raise TypeError(
            "_channel.bdpt_finalize_component_maps must return a dict"
        )
    validate_cuda_tensor(
        "path_gain", exported["path_gain"], dtype=torch.float32, ndim=3
    )
    if exported["path_gain"].shape != los.shape:
        raise ValueError(
            "_channel.bdpt_finalize_component_maps returned bad path_gain shape"
        )
    for name in (
        "los_power",
        "reflection_power",
        "diffraction_power",
        "transmission_power",
        "scattering_power",
    ):
        validate_cuda_tensor(name, exported[name], dtype=torch.float32, ndim=0)
    return exported


# ---------------------------------------------------------------------------
# ADR-022 finalize AD companion facades (plan 10a section 6.5 / 6.6). The
# finalize map is linear (elementwise sum into path_gain + per-component 0-dim
# power reductions), so the backward is the transpose scaling and the jvp is the
# forward map on the tangents; both are deterministic with no atomics. Pure
# dispatch, no Torch physics.
# ---------------------------------------------------------------------------

_BDPT_FINALIZE_COMPONENT_GRADS = (
    "grad_los",
    "grad_reflection",
    "grad_diffraction",
    "grad_transmission",
    "grad_scattering",
)
_BDPT_FINALIZE_TANGENTS = (
    "tangent_path_gain",
    "tangent_los_power",
    "tangent_reflection_power",
    "tangent_diffraction_power",
    "tangent_transmission_power",
    "tangent_scattering_power",
)


def _bdpt_finalize_backward(
    op_name: str,
    ndim: int,
    los: torch.Tensor,
    reflection: torch.Tensor,
    diffraction: torch.Tensor,
    transmission: torch.Tensor,
    scattering: torch.Tensor,
    *,
    grad_path_gain: torch.Tensor | None,
    grad_los_power: torch.Tensor | None,
    grad_reflection_power: torch.Tensor | None,
    grad_diffraction_power: torch.Tensor | None,
    grad_transmission_power: torch.Tensor | None,
    grad_scattering_power: torch.Tensor | None,
    need_grad_components: bool,
) -> dict[str, torch.Tensor | None]:
    for name, tensor in (
        ("los", los),
        ("reflection", reflection),
        ("diffraction", diffraction),
        ("transmission", transmission),
        ("scattering", scattering),
    ):
        validate_cuda_tensor(name, tensor, dtype=torch.float32, ndim=ndim)
        if tensor.shape != los.shape:
            raise ValueError("component tensors must share shape")
    if grad_path_gain is not None:
        validate_cuda_tensor(
            "grad_path_gain",
            grad_path_gain,
            dtype=torch.float32,
            ndim=ndim,
            require_contiguous=False,
        )
    for name, tensor in (
        ("grad_los_power", grad_los_power),
        ("grad_reflection_power", grad_reflection_power),
        ("grad_diffraction_power", grad_diffraction_power),
        ("grad_transmission_power", grad_transmission_power),
        ("grad_scattering_power", grad_scattering_power),
    ):
        if tensor is not None:
            validate_cuda_tensor(name, tensor, dtype=torch.float32, ndim=0)
    exported = _required_native_op(op_name)(
        los,
        reflection,
        diffraction,
        transmission,
        scattering,
        grad_path_gain,
        grad_los_power,
        grad_reflection_power,
        grad_diffraction_power,
        grad_transmission_power,
        grad_scattering_power,
        bool(need_grad_components),
    )
    if not isinstance(exported, dict) or set(exported) != set(
        _BDPT_FINALIZE_COMPONENT_GRADS
    ):
        raise TypeError(f"_channel.{op_name} returned unexpected fields")
    return exported


def _bdpt_finalize_jvp(
    op_name: str,
    ndim: int,
    los: torch.Tensor,
    reflection: torch.Tensor,
    diffraction: torch.Tensor,
    transmission: torch.Tensor,
    scattering: torch.Tensor,
    *,
    tangent_los: torch.Tensor | None,
    tangent_reflection: torch.Tensor | None,
    tangent_diffraction: torch.Tensor | None,
    tangent_transmission: torch.Tensor | None,
    tangent_scattering: torch.Tensor | None,
) -> dict[str, torch.Tensor]:
    for name, tensor in (
        ("los", los),
        ("reflection", reflection),
        ("diffraction", diffraction),
        ("transmission", transmission),
        ("scattering", scattering),
    ):
        validate_cuda_tensor(name, tensor, dtype=torch.float32, ndim=ndim)
        if tensor.shape != los.shape:
            raise ValueError("component tensors must share shape")
    exported = _required_native_op(op_name)(
        los,
        reflection,
        diffraction,
        transmission,
        scattering,
        tangent_los,
        tangent_reflection,
        tangent_diffraction,
        tangent_transmission,
        tangent_scattering,
    )
    if not isinstance(exported, dict) or set(exported) != set(_BDPT_FINALIZE_TANGENTS):
        raise TypeError(f"_channel.{op_name} returned unexpected fields")
    return exported


def bdpt_finalize_point_components_backward(
    los: torch.Tensor,
    reflection: torch.Tensor,
    diffraction: torch.Tensor,
    transmission: torch.Tensor,
    scattering: torch.Tensor,
    *,
    grad_path_gain: torch.Tensor | None = None,
    grad_los_power: torch.Tensor | None = None,
    grad_reflection_power: torch.Tensor | None = None,
    grad_diffraction_power: torch.Tensor | None = None,
    grad_transmission_power: torch.Tensor | None = None,
    grad_scattering_power: torch.Tensor | None = None,
    need_grad_components: bool = False,
) -> dict[str, torch.Tensor | None]:
    """VJP of :func:`bdpt_finalize_point_components` (spec 6.5)."""

    return _bdpt_finalize_backward(
        "bdpt_finalize_point_components_backward",
        2,
        los,
        reflection,
        diffraction,
        transmission,
        scattering,
        grad_path_gain=grad_path_gain,
        grad_los_power=grad_los_power,
        grad_reflection_power=grad_reflection_power,
        grad_diffraction_power=grad_diffraction_power,
        grad_transmission_power=grad_transmission_power,
        grad_scattering_power=grad_scattering_power,
        need_grad_components=need_grad_components,
    )


def bdpt_finalize_point_components_jvp(
    los: torch.Tensor,
    reflection: torch.Tensor,
    diffraction: torch.Tensor,
    transmission: torch.Tensor,
    scattering: torch.Tensor,
    *,
    tangent_los: torch.Tensor | None = None,
    tangent_reflection: torch.Tensor | None = None,
    tangent_diffraction: torch.Tensor | None = None,
    tangent_transmission: torch.Tensor | None = None,
    tangent_scattering: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """JVP of :func:`bdpt_finalize_point_components` (spec 6.5)."""

    return _bdpt_finalize_jvp(
        "bdpt_finalize_point_components_jvp",
        2,
        los,
        reflection,
        diffraction,
        transmission,
        scattering,
        tangent_los=tangent_los,
        tangent_reflection=tangent_reflection,
        tangent_diffraction=tangent_diffraction,
        tangent_transmission=tangent_transmission,
        tangent_scattering=tangent_scattering,
    )


def bdpt_finalize_component_maps_backward(
    los: torch.Tensor,
    reflection: torch.Tensor,
    diffraction: torch.Tensor,
    transmission: torch.Tensor,
    scattering: torch.Tensor,
    *,
    grad_path_gain: torch.Tensor | None = None,
    grad_los_power: torch.Tensor | None = None,
    grad_reflection_power: torch.Tensor | None = None,
    grad_diffraction_power: torch.Tensor | None = None,
    grad_transmission_power: torch.Tensor | None = None,
    grad_scattering_power: torch.Tensor | None = None,
    need_grad_components: bool = False,
) -> dict[str, torch.Tensor | None]:
    """VJP of :func:`bdpt_finalize_component_maps` (spec 6.6, 3-D maps)."""

    return _bdpt_finalize_backward(
        "bdpt_finalize_component_maps_backward",
        3,
        los,
        reflection,
        diffraction,
        transmission,
        scattering,
        grad_path_gain=grad_path_gain,
        grad_los_power=grad_los_power,
        grad_reflection_power=grad_reflection_power,
        grad_diffraction_power=grad_diffraction_power,
        grad_transmission_power=grad_transmission_power,
        grad_scattering_power=grad_scattering_power,
        need_grad_components=need_grad_components,
    )


def bdpt_finalize_component_maps_jvp(
    los: torch.Tensor,
    reflection: torch.Tensor,
    diffraction: torch.Tensor,
    transmission: torch.Tensor,
    scattering: torch.Tensor,
    *,
    tangent_los: torch.Tensor | None = None,
    tangent_reflection: torch.Tensor | None = None,
    tangent_diffraction: torch.Tensor | None = None,
    tangent_transmission: torch.Tensor | None = None,
    tangent_scattering: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """JVP of :func:`bdpt_finalize_component_maps` (spec 6.6, 3-D maps)."""

    return _bdpt_finalize_jvp(
        "bdpt_finalize_component_maps_jvp",
        3,
        los,
        reflection,
        diffraction,
        transmission,
        scattering,
        tangent_los=tangent_los,
        tangent_reflection=tangent_reflection,
        tangent_diffraction=tangent_diffraction,
        tangent_transmission=tangent_transmission,
        tangent_scattering=tangent_scattering,
    )


# -------------------------------------------------------------------------
# bdpt paths
# -------------------------------------------------------------------------
def bdpt_launch_state(
    reference: torch.Tensor,
    *,
    tx_count: int,
    samples: int,
    sample_streams: int,
    seed: int,
) -> dict[str, torch.Tensor]:
    validate_cuda_tensor("reference", reference, dtype=torch.float32, ndim=2)
    if tx_count < 0:
        raise ValueError("tx_count must be non-negative")
    if samples <= 0:
        raise ValueError("samples must be positive")
    if sample_streams <= 0:
        raise ValueError("sample_streams must be positive")
    if seed < 0:
        raise ValueError("seed must be non-negative")

    exported = _required_native_op("bdpt_launch_state")(
        reference,
        int(tx_count),
        int(samples),
        int(sample_streams),
        int(seed),
    )
    if not isinstance(exported, dict):
        raise TypeError("_channel.bdpt_launch_state must return a dict")
    expected = int(tx_count) * int(samples) * int(sample_streams)
    for name in ("tx_id", "sample_id", "stream_id"):
        validate_cuda_tensor(name, exported[name], dtype=torch.int32, ndim=1)
        if exported[name].shape != (expected,):
            raise ValueError(
                f"_channel.bdpt_launch_state returned bad {name} shape"
            )
    for name in ("light_seed",):
        validate_cuda_tensor(name, exported[name], dtype=torch.int64, ndim=1)
        if exported[name].shape != (expected,):
            raise ValueError(
                f"_channel.bdpt_launch_state returned bad {name} shape"
            )
    return exported


_BDPT_SUBPATH_SCHEMA: dict[str, tuple[torch.dtype, tuple[int | None, ...]]] = {
    "origin": (torch.float32, (None, 3)),
    "direction": (torch.float32, (None, 3)),
    "throughput_real": (torch.float32, (None,)),
    "throughput_imag": (torch.float32, (None,)),
    "pdf_forward": (torch.float32, (None,)),
    "pdf_reverse": (torch.float32, (None,)),
    "depth": (torch.int32, (None,)),
    "component_mask": (torch.int32, (None,)),
    "primitive_id": (torch.int32, (None,)),
    "edge_id": (torch.int32, (None,)),
    "tx_id": (torch.int32, (None,)),
    "rx_id": (torch.int32, (None,)),
    "grid_linear_id": (torch.int32, (None,)),
    "valid": (torch.bool, (None,)),
    "path_length": (torch.float32, (None,)),
    "field_real": (torch.float32, (None, 3)),
    "field_imag": (torch.float32, (None, 3)),
    "source_power": (torch.float32, (None,)),
    "event_type": (torch.int32, (None,)),
}


def _validate_bdpt_subpath_state(
    name: str, exported: dict[str, torch.Tensor], expected_count: int | None
) -> None:
    if not isinstance(exported, dict):
        raise TypeError(f"{name} must be a dict")
    if set(exported) != set(_BDPT_SUBPATH_SCHEMA):
        raise ValueError(f"{name} returned unexpected fields")
    inferred_count: int | None = expected_count
    for field, (dtype, shape_spec) in _BDPT_SUBPATH_SCHEMA.items():
        tensor = exported[field]
        validate_cuda_tensor(field, tensor, dtype=dtype, ndim=len(shape_spec))
        if inferred_count is None:
            inferred_count = int(tensor.shape[0])
        expected_shape = tuple(
            inferred_count if dim is None else dim for dim in shape_spec
        )
        if tuple(tensor.shape) != expected_shape:
            raise ValueError(f"{name} returned bad {field} shape")


_BDPT_CONNECTION_SCHEMA: dict[str, tuple[torch.dtype, tuple[int | None, ...]]] = {
    "topology": (torch.int32, (None, 4)),
    "contribution": (torch.float32, (None,)),
    "pdf": (torch.float32, (None,)),
    "mis_weight": (torch.float32, (None,)),
    "component_id": (torch.int32, (None,)),
    "valid": (torch.bool, (None,)),
    "tx_id": (torch.int32, (None,)),
    "rx_id": (torch.int32, (None,)),
    "grid_linear_id": (torch.int32, (None,)),
    "light_depth": (torch.int32, (None,)),
    "sensor_depth": (torch.int32, (None,)),
    "path_length_m": (torch.float32, (None,)),
}


def _validate_bdpt_connection_samples(
    name: str,
    exported: dict[str, torch.Tensor],
    expected_count: int | None,
) -> None:
    if not isinstance(exported, dict):
        raise TypeError(f"{name} must be a dict")
    if set(exported) != set(_BDPT_CONNECTION_SCHEMA):
        raise ValueError(f"{name} returned unexpected fields")
    inferred_count = expected_count
    for field, (dtype, shape_spec) in _BDPT_CONNECTION_SCHEMA.items():
        tensor = exported[field]
        validate_cuda_tensor(
            f"{name}.{field}", tensor, dtype=dtype, ndim=len(shape_spec)
        )
        if inferred_count is None:
            inferred_count = int(tensor.shape[0])
        expected_shape = tuple(
            inferred_count if dim is None else dim for dim in shape_spec
        )
        if tuple(tensor.shape) != expected_shape:
            raise ValueError(f"{name} returned bad {field} shape")


# The six accumulate outputs, in the order every caller reads them. The set is
# used for the ADR-022 subset key-set check (the coherent forward may add
# bin-sum buffers as extra keys).
_BDPT_COMPONENT_MATRIX_ORDER = (
    "path_gain",
    "los",
    "reflection",
    "diffraction",
    "transmission",
    "scattering",
)
_BDPT_COMPONENT_MATRIX_FIELDS = frozenset(_BDPT_COMPONENT_MATRIX_ORDER)

# ADR-022 spec 6.4: the coherent forward returns the per-component phasor bin
# sums S_b as non-differentiable outputs; the coherent backward reads them as
# explicit args in this order (real/imag per accumulating component). Absent for
# the power domain.
_BDPT_ACCUMULATE_BIN_SUM_ORDER = (
    "los_re",
    "los_im",
    "reflection_re",
    "reflection_im",
    "diffraction_re",
    "diffraction_im",
    "transmission_re",
    "transmission_im",
    "scattering_re",
    "scattering_im",
)


def _bdpt_accumulate_bin_sum_args(
    combine_domain: str, bin_sums: tuple[torch.Tensor, ...]
) -> tuple[torch.Tensor | None, ...]:
    """Expand the coherent forward's phasor bin sums into the ten positional
    ``los_re..scattering_im`` args the native accumulate VJP/JVP consume.

    ADR-022 spec 6.4 (supervisor ruling): the coherent backward/jvp read the
    per-component bin sums ``S_b`` retained by the forward, so no in-backward
    re-reduction and no sample coefficients are needed. The power domain takes
    no bin sums; every slot is ``None``."""

    bins = tuple(bin_sums)
    if combine_domain == "coherent":
        if len(bins) != len(_BDPT_ACCUMULATE_BIN_SUM_ORDER):
            raise ValueError(
                "coherent accumulate backward/jvp requires the ten forward "
                "phasor bin sums"
            )
        return bins
    if bins:
        raise ValueError("power-domain accumulate takes no bin sums")
    return (None,) * len(_BDPT_ACCUMULATE_BIN_SUM_ORDER)


def _bdpt_mis_mode_id(mis: str) -> int:
    if mis == "none":
        return 0
    if mis == "balance":
        return 1
    if mis == "power_heuristic":
        return 2
    raise ValueError("mis is not supported")


def bdpt_empty_subpath_state(reference: torch.Tensor) -> dict[str, torch.Tensor]:
    validate_cuda_tensor("reference", reference, dtype=torch.float32, ndim=2)
    exported = _required_native_op("bdpt_empty_subpath_state")(reference)
    _validate_bdpt_subpath_state(
        "_channel.bdpt_empty_subpath_state", exported, 0
    )
    return exported


def bdpt_endpoint_subpath_state(
    tx_positions: torch.Tensor,
    tx_power: torch.Tensor,
    tx_polarization: torch.Tensor,
    rx_positions: torch.Tensor,
    rx_polarization: torch.Tensor,
    launch_tx_id: torch.Tensor,
    light_seed: torch.Tensor,
) -> dict[str, dict[str, torch.Tensor]]:
    validate_cuda_tensor(
        "tx_positions", tx_positions, dtype=torch.float32, ndim=2, trailing_shape=(3,)
    )
    validate_cuda_tensor("tx_power", tx_power, dtype=torch.float32, ndim=1)
    validate_cuda_tensor(
        "tx_polarization", tx_polarization, dtype=torch.float32, ndim=2, trailing_shape=(3,)
    )
    validate_cuda_tensor(
        "rx_positions", rx_positions, dtype=torch.float32, ndim=2, trailing_shape=(3,)
    )
    validate_cuda_tensor(
        "rx_polarization", rx_polarization, dtype=torch.float32, ndim=2, trailing_shape=(3,)
    )
    validate_cuda_tensor("launch_tx_id", launch_tx_id, dtype=torch.int32, ndim=1)
    validate_cuda_tensor("light_seed", light_seed, dtype=torch.int64, ndim=1)
    if tx_power.shape != (tx_positions.shape[0],):
        raise ValueError("tx_power must match tx_positions")
    if tx_polarization.shape != tx_positions.shape:
        raise ValueError("tx_polarization must match tx_positions")
    if rx_polarization.shape != rx_positions.shape:
        raise ValueError("rx_polarization must match rx_positions")
    if light_seed.shape != launch_tx_id.shape:
        raise ValueError("light_seed must match launch_tx_id")
    device = tx_positions.get_device()
    if (
        tx_power.get_device() != device
        or tx_polarization.get_device() != device
        or rx_positions.get_device() != device
        or rx_polarization.get_device() != device
        or launch_tx_id.get_device() != device
        or light_seed.get_device() != device
    ):
        raise ValueError("BDPT endpoint tensors must share one CUDA device")
    exported = _required_native_op("bdpt_endpoint_subpath_state")(
        tx_positions,
        tx_power,
        tx_polarization,
        rx_positions,
        rx_polarization,
        launch_tx_id,
        light_seed,
    )
    if not isinstance(exported, dict) or set(exported) != {"light", "sensor"}:
        raise TypeError(
            "_channel.bdpt_endpoint_subpath_state must return light/sensor dicts"
        )
    light = exported["light"]
    sensor = exported["sensor"]
    _validate_bdpt_subpath_state(
        "_channel.bdpt_endpoint_subpath_state.light",
        light,
        int(launch_tx_id.shape[0]),
    )
    _validate_bdpt_subpath_state(
        "_channel.bdpt_endpoint_subpath_state.sensor",
        sensor,
        int(rx_positions.shape[0]),
    )
    return {"light": light, "sensor": sensor}


def bdpt_subpath_intersection_inputs(
    subpath: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    _validate_bdpt_subpath_state("subpath", subpath, None)
    exported = _required_native_op("bdpt_subpath_intersection_inputs")(subpath)
    if not isinstance(exported, dict):
        raise TypeError(
            "_channel.bdpt_subpath_intersection_inputs must return a dict"
        )
    if set(exported) != {"ray_o", "ray_d", "ray_tmax", "active"}:
        raise ValueError(
            "_channel.bdpt_subpath_intersection_inputs returned unexpected fields"
        )
    validate_cuda_tensor(
        "ray_o", exported["ray_o"], dtype=torch.float32, ndim=2, trailing_shape=(3,)
    )
    validate_cuda_tensor(
        "ray_d", exported["ray_d"], dtype=torch.float32, ndim=2, trailing_shape=(3,)
    )
    validate_cuda_tensor("ray_tmax", exported["ray_tmax"], dtype=torch.float32, ndim=1)
    validate_cuda_tensor("active", exported["active"], dtype=torch.bool, ndim=1)
    if (
        exported["ray_o"].shape != subpath["origin"].shape
        or exported["ray_d"].shape != subpath["direction"].shape
    ):
        raise ValueError(
            "_channel.bdpt_subpath_intersection_inputs returned bad ray shape"
        )
    if exported["active"].shape != subpath["valid"].shape:
        raise ValueError(
            "_channel.bdpt_subpath_intersection_inputs returned bad active shape"
        )
    if exported["ray_tmax"].shape != (0,):
        raise ValueError(
            "_channel.bdpt_subpath_intersection_inputs returned bad ray_tmax shape"
        )
    return exported


def bdpt_reflected_light_subpath_state(
    light: dict[str, torch.Tensor],
    intersection: dict[str, torch.Tensor],
    *,
    material_gain: torch.Tensor,
    material_valid: torch.Tensor,
    material_eps_r: torch.Tensor,
    material_sigma_e: torch.Tensor,
    material_mu_r: torch.Tensor,
    material_thickness: torch.Tensor,
    frequency_hz: float,
) -> dict[str, torch.Tensor]:
    _validate_bdpt_subpath_state("light", light, None)
    if not isinstance(intersection, dict) or set(intersection) != set(
        _BDPT_INTERSECTION_FIELDS
    ):
        raise ValueError("intersection returned unexpected fields")
    count = int(light["origin"].shape[0])
    validate_cuda_tensor(
        "intersection.t", intersection["t"], dtype=torch.float32, ndim=1
    )
    validate_cuda_tensor(
        "intersection.p",
        intersection["p"],
        dtype=torch.float32,
        ndim=2,
        trailing_shape=(3,),
    )
    validate_cuda_tensor(
        "intersection.n",
        intersection["n"],
        dtype=torch.float32,
        ndim=2,
        trailing_shape=(3,),
    )
    validate_cuda_tensor(
        "intersection.global_prim_id",
        intersection["global_prim_id"],
        dtype=torch.int32,
        ndim=1,
    )
    validate_cuda_tensor("material_gain", material_gain, dtype=torch.float32, ndim=1)
    validate_cuda_tensor("material_valid", material_valid, dtype=torch.bool, ndim=1)
    validate_cuda_tensor("material_eps_r", material_eps_r, dtype=torch.float32, ndim=1)
    validate_cuda_tensor(
        "material_sigma_e", material_sigma_e, dtype=torch.float32, ndim=1
    )
    validate_cuda_tensor("material_mu_r", material_mu_r, dtype=torch.float32, ndim=1)
    validate_cuda_tensor(
        "material_thickness", material_thickness, dtype=torch.float32, ndim=1
    )
    if int(material_gain.shape[0]) != int(material_valid.shape[0]):
        raise ValueError("material_gain and material_valid must have matching length")
    for name, tensor in (
        ("material_eps_r", material_eps_r),
        ("material_sigma_e", material_sigma_e),
        ("material_mu_r", material_mu_r),
        ("material_thickness", material_thickness),
    ):
        if int(tensor.shape[0]) != int(material_gain.shape[0]):
            raise ValueError(f"{name} must match material_gain length")
    if frequency_hz <= 0.0:
        raise ValueError("frequency_hz must be positive")
    if (
        material_gain.get_device() != light["origin"].get_device()
        or material_valid.get_device() != light["origin"].get_device()
        or material_eps_r.get_device() != light["origin"].get_device()
        or material_sigma_e.get_device() != light["origin"].get_device()
        or material_mu_r.get_device() != light["origin"].get_device()
        or material_thickness.get_device() != light["origin"].get_device()
    ):
        raise ValueError("material tensors must share light device")
    for name in ("t", "p", "n", "global_prim_id"):
        if int(intersection[name].shape[0]) != count:
            raise ValueError("intersection must match light subpath count")
        if intersection[name].get_device() != light["origin"].get_device():
            raise ValueError("intersection tensors must share light device")
    exported = _required_native_op("bdpt_reflected_light_subpath_state")(
        light,
        intersection,
        material_gain,
        material_valid,
        material_eps_r,
        material_sigma_e,
        material_mu_r,
        material_thickness,
        float(frequency_hz),
    )
    _validate_bdpt_subpath_state(
        "_channel.bdpt_reflected_light_subpath_state", exported, count
    )
    return exported


def bdpt_transmitted_light_subpath_state(
    light: dict[str, torch.Tensor],
    intersection: dict[str, torch.Tensor],
    *,
    face_material_id: torch.Tensor,
    layer_offset: torch.Tensor,
    layer_count: torch.Tensor,
    layer_thickness_m: torch.Tensor,
    layer_eps_r: torch.Tensor,
    layer_sigma_e: torch.Tensor,
    layer_mu_r: torch.Tensor,
    frequency_hz: float,
) -> dict[str, torch.Tensor]:
    _validate_bdpt_subpath_state("light", light, None)
    if not isinstance(intersection, dict) or set(intersection) != set(
        _BDPT_INTERSECTION_FIELDS
    ):
        raise ValueError("intersection returned unexpected fields")
    count = int(light["origin"].shape[0])
    validate_cuda_tensor(
        "intersection.t", intersection["t"], dtype=torch.float32, ndim=1
    )
    validate_cuda_tensor(
        "intersection.p",
        intersection["p"],
        dtype=torch.float32,
        ndim=2,
        trailing_shape=(3,),
    )
    validate_cuda_tensor(
        "intersection.n",
        intersection["n"],
        dtype=torch.float32,
        ndim=2,
        trailing_shape=(3,),
    )
    validate_cuda_tensor(
        "intersection.global_prim_id",
        intersection["global_prim_id"],
        dtype=torch.int32,
        ndim=1,
    )
    validate_cuda_tensor(
        "face_material_id", face_material_id, dtype=torch.int32, ndim=1
    )
    device = light["origin"].get_device()
    if face_material_id.get_device() != device:
        raise ValueError("face_material_id must share light device")
    _validate_layer_csr(
        layer_offset,
        layer_count,
        layer_thickness_m,
        layer_eps_r,
        layer_sigma_e,
        layer_mu_r,
        device,
    )
    if frequency_hz <= 0.0:
        raise ValueError("frequency_hz must be positive")
    for name in ("t", "p", "n", "global_prim_id"):
        if int(intersection[name].shape[0]) != count:
            raise ValueError("intersection must match light subpath count")
        if intersection[name].get_device() != device:
            raise ValueError("intersection tensors must share light device")
    exported = _required_native_op("bdpt_transmitted_light_subpath_state")(
        light,
        intersection,
        face_material_id,
        layer_offset,
        layer_count,
        layer_thickness_m,
        layer_eps_r,
        layer_sigma_e,
        layer_mu_r,
        float(frequency_hz),
    )
    _validate_bdpt_subpath_state(
        "_channel.bdpt_transmitted_light_subpath_state", exported, count
    )
    return exported


def bdpt_endpoint_connection_samples(
    light: dict[str, torch.Tensor],
    sensor: dict[str, torch.Tensor],
    *,
    frequency_hz: float,
    samples_per_tx: int,
    max_paths: int | None = None,
    mis: str = "power_heuristic",
    beta: float = 2.0,
    strategy_count: int = 1,
) -> dict[str, torch.Tensor]:
    _validate_bdpt_subpath_state("light", light, None)
    _validate_bdpt_subpath_state("sensor", sensor, None)
    if frequency_hz <= 0.0:
        raise ValueError("frequency_hz must be positive")
    if samples_per_tx <= 0:
        raise ValueError("samples_per_tx must be positive")
    if beta <= 0.0:
        raise ValueError("beta must be positive")
    if strategy_count != 1:
        raise ValueError("endpoint connections support exactly one strategy")
    max_paths_value = -1 if max_paths is None else int(max_paths)
    if max_paths_value < -1:
        raise ValueError("max_paths must be non-negative")
    mode_id = _bdpt_mis_mode_id(mis)
    expected_total = int(light["origin"].shape[0]) * int(sensor["origin"].shape[0])
    expected_count = (
        expected_total if max_paths is None else min(int(max_paths), expected_total)
    )
    exported = _required_native_op("bdpt_endpoint_connection_samples")(
        light,
        sensor,
        float(frequency_hz),
        int(samples_per_tx),
        int(mode_id),
        float(beta),
        int(strategy_count),
        int(max_paths_value),
    )
    _validate_bdpt_connection_samples(
        "_channel.bdpt_endpoint_connection_samples", exported, expected_count
    )
    return exported


def bdpt_endpoint_connection_visibility_inputs(
    light: dict[str, torch.Tensor],
    sensor: dict[str, torch.Tensor],
    *,
    sample_count: int,
) -> dict[str, torch.Tensor]:
    _validate_bdpt_subpath_state("light", light, None)
    _validate_bdpt_subpath_state("sensor", sensor, None)
    if sample_count < 0:
        raise ValueError("sample_count must be non-negative")
    expected_total = int(light["origin"].shape[0]) * int(sensor["origin"].shape[0])
    if int(sample_count) > expected_total:
        raise ValueError("sample_count exceeds endpoint pair count")
    exported = _required_native_op("bdpt_endpoint_connection_visibility_inputs")(
        light,
        sensor,
        int(sample_count),
    )
    if not isinstance(exported, dict):
        raise TypeError(
            "_channel.bdpt_endpoint_connection_visibility_inputs must return a dict"
        )
    if set(exported) != {"start", "end", "active"}:
        raise ValueError(
            "_channel.bdpt_endpoint_connection_visibility_inputs returned unexpected fields"
        )
    validate_cuda_tensor(
        "start", exported["start"], dtype=torch.float32, ndim=2, trailing_shape=(3,)
    )
    validate_cuda_tensor(
        "end", exported["end"], dtype=torch.float32, ndim=2, trailing_shape=(3,)
    )
    validate_cuda_tensor("active", exported["active"], dtype=torch.bool, ndim=1)
    if tuple(exported["start"].shape) != (int(sample_count), 3):
        raise ValueError(
            "_channel.bdpt_endpoint_connection_visibility_inputs returned bad start shape"
        )
    if exported["end"].shape != exported["start"].shape or exported["active"].shape != (
        int(sample_count),
    ):
        raise ValueError(
            "_channel.bdpt_endpoint_connection_visibility_inputs returned bad visibility shape"
        )
    return exported


def _resolve_accumulate_coeffs(
    samples: dict[str, torch.Tensor],
    combine_domain: str,
    coeff_real: torch.Tensor | None,
    coeff_imag: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Validate (coherent) or synthesize (power) the phasor coefficient planes.

    The coherent branch requires row-aligned ``coeff_real``/``coeff_imag`` and
    validates them; the power branch ignores any coefficients and returns empty
    placeholders. Split out of the forward to keep its complexity within budget.
    """

    if combine_domain != "coherent":
        empty = torch.empty(
            (0,), device=samples["contribution"].device, dtype=torch.float32
        )
        return empty, empty
    if coeff_real is None or coeff_imag is None:
        raise ValueError("coherent combine requires coeff_real and coeff_imag")
    for name, tensor in (("coeff_real", coeff_real), ("coeff_imag", coeff_imag)):
        # The coefficients arrive as .real/.imag strided views of the
        # natively-computed complex path field; the one-time layout copy
        # happens at the C++ ABI boundary (mc hot-path layout-copy rule),
        # so contiguity is not required here.
        validate_cuda_tensor(
            name, tensor, dtype=torch.float32, ndim=1, require_contiguous=False
        )
        if tensor.shape != samples["contribution"].shape:
            raise ValueError(f"{name} must match connection-sample rows")
        if tensor.get_device() != samples["contribution"].get_device():
            raise ValueError(f"{name} must share the connection-sample device")
    return coeff_real, coeff_imag


def bdpt_accumulate_connection_samples(
    samples: dict[str, torch.Tensor],
    *,
    tx_count: int,
    rx_count: int,
    accumulation_strategy: str = "atomic",
    combine_domain: str = "power",
    coeff_real: torch.Tensor | None = None,
    coeff_imag: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Accumulate connection samples into per-component matrices.

    ``combine_domain='power'`` (default) is the incoherent per-path power
    accumulation, bit-identical to the pre-ADR-019 behaviour; the coefficient
    tensors are ignored. ``combine_domain='coherent'`` (ADR-019, opt-in) sums
    the complex projected field coefficient (``coeff_real``/``coeff_imag``,
    row-aligned to ``samples``) into per-(tx, rx, component) phasor bins and
    finalizes ``|sum|^2``; the ``accumulation_strategy`` perf axis stays
    orthogonal (the coherent phasor sum always uses the atomic-double
    reduction).
    """

    _validate_bdpt_connection_samples("samples", samples, None)
    if tx_count < 0 or rx_count < 0:
        raise ValueError("tx_count and rx_count must be non-negative")
    strategy_ids = {"atomic": 0, "staged": 1, "compact": 2}
    if accumulation_strategy not in strategy_ids:
        raise ValueError(
            "accumulation_strategy must be 'atomic', 'staged', or 'compact'"
        )
    combine_ids = {"power": 0, "coherent": 1}
    if combine_domain not in combine_ids:
        raise ValueError("combine_domain must be 'power' or 'coherent'")
    coeff_real, coeff_imag = _resolve_accumulate_coeffs(
        samples, combine_domain, coeff_real, coeff_imag
    )
    exported = _required_native_op("bdpt_accumulate_connection_samples")(
        samples,
        int(tx_count),
        int(rx_count),
        int(strategy_ids[accumulation_strategy]),
        int(combine_ids[combine_domain]),
        coeff_real,
        coeff_imag,
    )
    if not isinstance(exported, dict):
        raise TypeError(
            "_channel.bdpt_accumulate_connection_samples must return a dict"
        )
    # ADR-022 spec 6.4 supervisor ruling: under combine_domain='coherent' the
    # forward additionally returns its per-component phasor bin-sum buffers
    # (``S_b``) as non-differentiable outputs so the coherent backward can read
    # them without a second atomic-double reduction. The primal component
    # matrices are unchanged bitwise; a subset check accepts the extra keys and
    # keeps the public return the six component matrices only.
    if not _BDPT_COMPONENT_MATRIX_FIELDS.issubset(exported):
        raise ValueError(
            "_channel.bdpt_accumulate_connection_samples returned unexpected fields"
        )
    for name in _BDPT_COMPONENT_MATRIX_ORDER:
        tensor = exported[name]
        validate_cuda_tensor(name, tensor, dtype=torch.float32, ndim=2)
        if tuple(tensor.shape) != (int(tx_count), int(rx_count)):
            raise ValueError(
                f"_channel.bdpt_accumulate_connection_samples returned bad {name} shape"
            )
    return {name: exported[name] for name in _BDPT_COMPONENT_MATRIX_ORDER}


def bdpt_filter_connection_samples(
    samples: dict[str, torch.Tensor],
    visible: torch.Tensor,
) -> dict[str, torch.Tensor]:
    _validate_bdpt_connection_samples("samples", samples, None)
    validate_cuda_tensor("visible", visible, dtype=torch.bool, ndim=1)
    if visible.shape != samples["valid"].shape:
        raise ValueError("visible must match samples")
    if visible.get_device() != samples["valid"].get_device():
        raise ValueError("visible must share samples device")
    exported = _required_native_op("bdpt_filter_connection_samples")(samples, visible)
    _validate_bdpt_connection_samples(
        "_channel.bdpt_filter_connection_samples", exported, None
    )
    return exported


def bdpt_count_valid_connection_samples(samples: dict[str, torch.Tensor]) -> int:
    _validate_bdpt_connection_samples("samples", samples, None)
    count = _required_native_op("bdpt_count_valid_connection_samples")(samples)
    if not isinstance(count, int):
        raise TypeError(
            "_channel.bdpt_count_valid_connection_samples must return an int"
        )
    if count < 0 or count > int(samples["valid"].shape[0]):
        raise ValueError(
            "_channel.bdpt_count_valid_connection_samples returned bad count"
        )
    return count


def bdpt_compact_connection_samples(
    samples: dict[str, torch.Tensor],
    *,
    max_paths: int | None = None,
) -> dict[str, torch.Tensor]:
    _validate_bdpt_connection_samples("samples", samples, None)
    max_paths_value = -1 if max_paths is None else int(max_paths)
    if max_paths_value < -1:
        raise ValueError("max_paths must be non-negative")
    exported = _required_native_op("bdpt_compact_connection_samples")(
        samples, int(max_paths_value)
    )
    _validate_bdpt_connection_samples(
        "_channel.bdpt_compact_connection_samples", exported, None
    )
    if max_paths is not None and int(exported["valid"].shape[0]) > int(max_paths):
        raise ValueError(
            "_channel.bdpt_compact_connection_samples exceeded max_paths"
        )
    return exported


def bdpt_concat_connection_samples(
    samples: tuple[dict[str, torch.Tensor], ...] | list[dict[str, torch.Tensor]],
) -> dict[str, torch.Tensor]:
    if not samples:
        raise ValueError("samples must not be empty")
    expected_count = 0
    for index, block in enumerate(samples):
        _validate_bdpt_connection_samples(f"samples[{index}]", block, None)
        expected_count += int(block["valid"].shape[0])
    exported = _required_native_op("bdpt_concat_connection_samples")(tuple(samples))
    _validate_bdpt_connection_samples(
        "_channel.bdpt_concat_connection_samples", exported, expected_count
    )
    return exported


def bdpt_connection_variance(
    samples: dict[str, torch.Tensor],
    *,
    tx_count: int,
    rx_count: int,
    samples_per_tx: int,
) -> torch.Tensor:
    _validate_bdpt_connection_samples("samples", samples, None)
    if tx_count < 0 or rx_count < 0:
        raise ValueError("tx_count and rx_count must be non-negative")
    if samples_per_tx <= 0:
        raise ValueError("samples_per_tx must be positive")
    variance = _required_native_op("bdpt_connection_variance")(
        samples,
        int(tx_count),
        int(rx_count),
        int(samples_per_tx),
    )
    if not isinstance(variance, torch.Tensor):
        raise TypeError("_channel.bdpt_connection_variance must return a tensor")
    validate_cuda_tensor("variance", variance, dtype=torch.float32, ndim=2)
    if tuple(variance.shape) != (int(tx_count), int(rx_count)):
        raise ValueError("_channel.bdpt_connection_variance returned bad shape")
    return variance


def bdpt_mis_weights(
    pdf: torch.Tensor,
    strategy_pdf_sum: torch.Tensor,
    *,
    mis: str,
    beta: float = 2.0,
) -> torch.Tensor:
    validate_cuda_tensor("pdf", pdf, dtype=torch.float32, ndim=1)
    validate_cuda_tensor(
        "strategy_pdf_sum", strategy_pdf_sum, dtype=torch.float32, ndim=0
    )
    mode_id = _bdpt_mis_mode_id(mis)
    if beta <= 0.0:
        raise ValueError("beta must be positive")

    weights = _required_native_op("bdpt_mis_weights")(
        pdf, strategy_pdf_sum, int(mode_id), float(beta)
    )
    if not isinstance(weights, torch.Tensor):
        raise TypeError("_channel.bdpt_mis_weights must return a tensor")
    validate_cuda_tensor("weights", weights, dtype=torch.float32, ndim=1)
    if weights.shape != pdf.shape:
        raise ValueError(
            "_channel.bdpt_mis_weights returned an unexpected shape"
        )
    return weights


# -------------------------------------------------------------------------
# bdpt sampling
# -------------------------------------------------------------------------
def bdpt_sample_directions(
    count: int, reference: torch.Tensor, *, seed: int
) -> torch.Tensor:
    if count < 0:
        raise ValueError("count must be non-negative")
    if seed < 0:
        raise ValueError("seed must be non-negative")
    validate_cuda_tensor("reference", reference, dtype=torch.float32, ndim=2)
    directions = _required_native_op("bdpt_sample_directions")(
        int(count), reference, int(seed)
    )
    if not isinstance(directions, torch.Tensor):
        raise TypeError("_channel.bdpt_sample_directions must return a tensor")
    validate_cuda_tensor(
        "directions", directions, dtype=torch.float32, ndim=2, trailing_shape=(3,)
    )
    return directions


def bdpt_reflection_launch_inputs(
    tx_positions: torch.Tensor,
    *,
    tx_index: int,
    sample_count: int,
) -> dict[str, torch.Tensor]:
    validate_cuda_tensor(
        "tx_positions", tx_positions, dtype=torch.float32, ndim=2, trailing_shape=(3,)
    )
    if sample_count < 0:
        raise ValueError("sample_count must be non-negative")
    exported = _required_native_op("bdpt_reflection_launch_inputs")(
        tx_positions, int(tx_index), int(sample_count)
    )
    if not isinstance(exported, dict):
        raise TypeError(
            "_channel.bdpt_reflection_launch_inputs must return a dict"
        )
    validate_cuda_tensor(
        "ray_o", exported["ray_o"], dtype=torch.float32, ndim=2, trailing_shape=(3,)
    )
    validate_cuda_tensor("ray_tmax", exported["ray_tmax"], dtype=torch.float32, ndim=1)
    validate_cuda_tensor("active", exported["active"], dtype=torch.bool, ndim=1)
    validate_cuda_tensor(
        "tx_pol", exported["tx_pol"], dtype=torch.float32, ndim=2, trailing_shape=(3,)
    )
    validate_cuda_tensor("tx_id", exported["tx_id"], dtype=torch.int32, ndim=1)
    validate_cuda_tensor(
        "light_seed", exported["light_seed"], dtype=torch.int64, ndim=1
    )
    return exported


def bdpt_pack_vec3(x: torch.Tensor, y: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    validate_cuda_tensor("x", x, dtype=torch.float32, ndim=1)
    validate_cuda_tensor("y", y, dtype=torch.float32, ndim=1)
    validate_cuda_tensor("z", z, dtype=torch.float32, ndim=1)
    packed = _required_native_op("bdpt_pack_vec3")(x, y, z)
    if not isinstance(packed, torch.Tensor):
        raise TypeError("_channel.bdpt_pack_vec3 must return a tensor")
    validate_cuda_tensor(
        "packed", packed, dtype=torch.float32, ndim=2, trailing_shape=(3,)
    )
    return packed