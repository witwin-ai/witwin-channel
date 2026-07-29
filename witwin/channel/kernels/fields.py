# Copyright Xingyu Chen.
# Native RF field kernel facades.

"""Native RF field kernel facades.

Thin facades over the ``_channel`` field ABI: the enumerated field transports
and their native derivative companions, the ``torch.autograd.Function``
wrappers that dispatch those companions, the ADR-039 source-amplitude owner,
and the deterministic single-component field owners. Every entry validates its
contract, requests the required native symbol through
:mod:`witwin.channel.runtime`, dispatches the native operation, and converts
its result into a named typed contract.

functional
----------
The plain forward transports - free space, receiver projection, reflection and
transmission sequences, coupled RD/DD, the pure wedge, and the rough-surface
``C_r`` scale - together with the registered native backward/JVP companions
they publish and the shared output/tangent field tuples every wrapper reuses.

liveness
--------
Which conditional outputs of one field apply carry a derivative: the ADR-038
geometry decision behind ``path_length_m``/``delay_s`` and the ADR-043
direction decision behind ``field_direction``. Both are decided by the wrapper
where forward duals are still visible and read back here, so the field
Functions agree on what a dead output looks like.

autograd
--------
The ``torch.autograd.Function`` companions for free space, the reflection and
transmission sequences, the pure wedge, coupled RD, and the coupled RD
preparation seam.

coupled DD autograd
-------------------
The coupled double-diffraction ``torch.autograd.Function`` and its material
grad-request index groups, kept beside the wedge-face column layout it mirrors.

projection autograd
-------------------
The differentiable receiver projection on a frozen polarization basis.

rough scale
-----------
The differentiable rough-surface ``C_r`` reflection scale (ADR-010 op 3).

source amplitude
----------------
Source-amplitude application onto a transported complex3 field (ADR-039). The
field transport kernels publish a unit-excitation complex3 vector and an
excited scalar ``path_field = coefficient * sqrt(tx_power)``, but no excited
vector; this is the native facade for that missing output and its
differentiable wrapper. The map is linear in the field vector and the
amplitude is a real per-row constant, so the VJP and the JVP are the same
native scale, and ``tx_power`` stays a frozen primal exactly as it is in every
field transport companion.

deterministic
-------------
The deterministic solver's single-component field owners: the LoS,
diffraction-vector, reflection and reflection-sequence field kernels plus the
small delay/phase/pack primitives they share.
"""

from __future__ import annotations

import torch

from witwin.channel.materials import validate_layer_csr as _validate_layer_csr
from witwin.channel.runtime import (
    _ad_checked_tangent,
    _ad_first_order_only,
    _ad_frequency_grad,
    _ad_frequency_tangent,
    _ad_frequency_value,
    _ad_geometry_live,
    _ad_geometry_tangent,
    _ad_native_tangent_or_none,
    _ad_native_tensor,
    _ad_reject_fixed_inputs,
    _ad_reject_fixed_tangents,
    disable_functorch,
    required_symbol as _required_native_op,
    validate_cuda_tensor,
)

__all__ = [
    "_CoupledRdPrepareAdFunction",
    "_FieldCoupledDdAdFunction",
    "_FieldCoupledRdAdFunction",
    "_FieldDiffractionWedgeAdFunction",
    "_FieldFreeSpaceAdFunction",
    "_FieldProjectComplex3AdFunction",
    "_FieldReflectionSequenceAdFunction",
    "_FieldRoughReflectionScaleAdFunction",
    "_FieldSourceAmplitudeScaleAdFunction",
    "_FieldTransmissionSequenceAdFunction",
    "ad_liveness",
    "coupled_rd_prepare_ad",
    "deterministic_delay_to_path_length",
    "deterministic_diffraction_vector_field",
    "deterministic_field_from_power_phase",
    "deterministic_los_field",
    "deterministic_pack_complex",
    "deterministic_phase_from_field",
    "deterministic_phase_from_length",
    "deterministic_reflection_field",
    "deterministic_reflection_sequence_field",
    "deterministic_zero_field_phase",
    "direction_cotangent",
    "direction_tangents",
    "field_coupled_dd",
    "field_coupled_dd_ad",
    "field_coupled_rd",
    "field_coupled_rd_ad",
    "field_diffraction_wedge",
    "field_diffraction_wedge_ad",
    "field_free_space",
    "field_free_space_ad",
    "field_free_space_backward",
    "field_free_space_jvp",
    "field_project_complex3",
    "field_project_complex3_ad",
    "field_reflection_sequence",
    "field_reflection_sequence_ad",
    "field_reflection_sequence_backward",
    "field_reflection_sequence_jvp",
    "field_rough_reflection_scale",
    "field_rough_reflection_scale_ad",
    "field_rough_reflection_scale_backward",
    "field_rough_reflection_scale_jvp",
    "field_source_amplitude_scale",
    "field_source_amplitude_scale_ad",
    "field_source_amplitude_scale_backward",
    "field_source_amplitude_scale_jvp",
    "field_transmission_sequence",
    "field_transmission_sequence_ad",
    "field_transmission_sequence_backward",
    "field_transmission_sequence_jvp",
    "mark_dead_outputs",
]


def _validate_enumerated_field_result(
    out: object,
    count: int,
    *,
    type_error: str,
    field_error: str,
    shape_error_prefix: str,
) -> dict[str, torch.Tensor]:
    """Validate the shared enumerated-field result schema in field order."""

    if not isinstance(out, dict):
        raise TypeError(type_error)
    schema = {
        "field_vector": (torch.complex64, 2, (count, 3)),
        "coefficient": (torch.complex64, 1, (count,)),
        "path_field": (torch.complex64, 1, (count,)),
        "path_gain": (torch.float32, 1, (count,)),
        "path_length_m": (torch.float32, 1, (count,)),
        "delay_s": (torch.float32, 1, (count,)),
        "direction": (torch.float32, 2, (count, 3)),
    }
    if set(out) != set(schema):
        raise ValueError(field_error)
    for name, (dtype, ndim, shape) in schema.items():
        validate_cuda_tensor(name, out[name], dtype=dtype, ndim=ndim)
        if tuple(out[name].shape) != shape:
            raise ValueError(f"{shape_error_prefix} returned bad {name} shape")
    return out


def field_free_space(
    source: torch.Tensor,
    target: torch.Tensor,
    tx_power: torch.Tensor,
    tx_polarization: torch.Tensor,
    rx_polarization: torch.Tensor,
    *,
    frequency_hz: float,
) -> dict[str, torch.Tensor]:
    for name, value in (
        ("source", source),
        ("target", target),
        ("tx_polarization", tx_polarization),
        ("rx_polarization", rx_polarization),
    ):
        validate_cuda_tensor(
            name, value, dtype=torch.float32, ndim=2, trailing_shape=(3,)
        )
    validate_cuda_tensor("tx_power", tx_power, dtype=torch.float32, ndim=1)
    count = int(source.shape[0])
    if any(
        int(value.shape[0]) != count
        for value in (target, tx_power, tx_polarization, rx_polarization)
    ):
        raise ValueError("free-space field tensors must have matching rows")
    if frequency_hz <= 0.0:
        raise ValueError("frequency_hz must be positive")
    out = _required_native_op("field_free_space")(
        source,
        target,
        tx_power,
        tx_polarization,
        rx_polarization,
        float(frequency_hz),
    )
    return _validate_enumerated_field_result(
        out,
        count,
        type_error="_channel.field_free_space must return a dict",
        field_error="_channel.field_free_space returned unexpected fields",
        shape_error_prefix="_channel.field_free_space",
    )


def field_project_complex3(
    field_vector: torch.Tensor,
    direction: torch.Tensor,
    rx_polarization: torch.Tensor,
) -> dict[str, torch.Tensor]:
    validate_cuda_tensor("field_vector", field_vector, dtype=torch.complex64, ndim=2)
    validate_cuda_tensor(
        "direction", direction, dtype=torch.float32, ndim=2, trailing_shape=(3,)
    )
    validate_cuda_tensor(
        "rx_polarization",
        rx_polarization,
        dtype=torch.float32,
        ndim=2,
        trailing_shape=(3,),
    )
    count = int(field_vector.shape[0])
    if field_vector.shape != (count, 3):
        raise ValueError("field_vector must have shape (N, 3)")
    if direction.shape != (count, 3) or rx_polarization.shape != (count, 3):
        raise ValueError("direction and rx_polarization must match field_vector rows")
    out = _required_native_op("field_project_complex3")(
        field_vector, direction, rx_polarization
    )
    if not isinstance(out, dict) or set(out) != {"coefficient", "path_gain"}:
        raise TypeError("_channel.field_project_complex3 returned invalid fields")
    validate_cuda_tensor("coefficient", out["coefficient"], dtype=torch.complex64, ndim=1)
    validate_cuda_tensor("path_gain", out["path_gain"], dtype=torch.float32, ndim=1)
    if out["coefficient"].shape != (count,) or out["path_gain"].shape != (count,):
        raise ValueError("field projection returned invalid shapes")
    return out


def field_reflection_sequence(
    source: torch.Tensor,
    target: torch.Tensor,
    interaction_positions: torch.Tensor,
    interaction_normals: torch.Tensor,
    tx_power: torch.Tensor,
    tx_polarization: torch.Tensor,
    rx_polarization: torch.Tensor,
    eps_r: torch.Tensor,
    sigma_e: torch.Tensor,
    mu_r: torch.Tensor,
    gain: torch.Tensor,
    thickness: torch.Tensor,
    *,
    frequency_hz: float,
) -> dict[str, torch.Tensor]:
    for name, value in (
        ("source", source),
        ("target", target),
        ("tx_polarization", tx_polarization),
        ("rx_polarization", rx_polarization),
    ):
        validate_cuda_tensor(
            name, value, dtype=torch.float32, ndim=2, trailing_shape=(3,)
        )
    for name, value in (
        ("interaction_positions", interaction_positions),
        ("interaction_normals", interaction_normals),
    ):
        validate_cuda_tensor(
            name, value, dtype=torch.float32, ndim=3, trailing_shape=(3,)
        )
    validate_cuda_tensor("tx_power", tx_power, dtype=torch.float32, ndim=1)
    count = int(source.shape[0])
    depth = int(interaction_positions.shape[1])
    if interaction_positions.shape != (count, depth, 3) or depth <= 0:
        raise ValueError("interaction_positions must have shape (N, D, 3), D > 0")
    if interaction_normals.shape != interaction_positions.shape:
        raise ValueError("interaction_normals must match interaction_positions")
    for name, value in (
        ("eps_r", eps_r),
        ("sigma_e", sigma_e),
        ("mu_r", mu_r),
        ("gain", gain),
        ("thickness", thickness),
    ):
        validate_cuda_tensor(name, value, dtype=torch.float32, ndim=2)
        if value.shape != (count, depth):
            raise ValueError(f"{name} must have shape (N, D)")
    if frequency_hz <= 0.0:
        raise ValueError("frequency_hz must be positive")
    out = _required_native_op("field_reflection_sequence")(
        source,
        target,
        interaction_positions,
        interaction_normals,
        tx_power,
        tx_polarization,
        rx_polarization,
        eps_r,
        sigma_e,
        mu_r,
        gain,
        thickness,
        float(frequency_hz),
    )
    return _validate_enumerated_field_result(
        out,
        count,
        type_error="_channel.field_reflection_sequence must return a dict",
        field_error="field_reflection_sequence returned unexpected fields",
        shape_error_prefix="field_reflection_sequence",
    )


def field_transmission_sequence(
    path_valid: torch.Tensor,
    source: torch.Tensor,
    target: torch.Tensor,
    interaction_positions: torch.Tensor,
    interaction_normals: torch.Tensor,
    interaction_material_id: torch.Tensor,
    interaction_valid: torch.Tensor,
    tx_power: torch.Tensor,
    tx_polarization: torch.Tensor,
    rx_polarization: torch.Tensor,
    layer_offset: torch.Tensor,
    layer_count: torch.Tensor,
    layer_thickness_m: torch.Tensor,
    layer_eps_r: torch.Tensor,
    layer_sigma_e: torch.Tensor,
    layer_mu_r: torch.Tensor,
    *,
    frequency_hz: float,
) -> dict[str, torch.Tensor]:
    validate_cuda_tensor("path_valid", path_valid, dtype=torch.bool, ndim=1)
    for name, value in (
        ("source", source),
        ("target", target),
        ("tx_polarization", tx_polarization),
        ("rx_polarization", rx_polarization),
    ):
        validate_cuda_tensor(
            name, value, dtype=torch.float32, ndim=2, trailing_shape=(3,)
        )
    for name, value in (
        ("interaction_positions", interaction_positions),
        ("interaction_normals", interaction_normals),
    ):
        validate_cuda_tensor(
            name, value, dtype=torch.float32, ndim=3, trailing_shape=(3,)
        )
    validate_cuda_tensor(
        "interaction_material_id", interaction_material_id, dtype=torch.int32, ndim=2
    )
    validate_cuda_tensor(
        "interaction_valid", interaction_valid, dtype=torch.bool, ndim=2
    )
    validate_cuda_tensor("tx_power", tx_power, dtype=torch.float32, ndim=1)
    count = int(source.shape[0])
    if path_valid.shape != (count,):
        raise ValueError("path_valid must have shape (N,)")
    depth = int(interaction_positions.shape[1])
    if interaction_positions.shape != (count, depth, 3) or depth <= 0:
        raise ValueError("interaction_positions must have shape (N, D, 3), D > 0")
    if interaction_normals.shape != interaction_positions.shape:
        raise ValueError("interaction_normals must match interaction_positions")
    if interaction_material_id.shape != (count, depth):
        raise ValueError("interaction_material_id must have shape (N, D)")
    if interaction_valid.shape != (count, depth):
        raise ValueError("interaction_valid must have shape (N, D)")
    if any(
        int(value.shape[0]) != count
        for value in (target, tx_power, tx_polarization, rx_polarization)
    ):
        raise ValueError("transmission endpoint tensors must have matching rows")
    _validate_layer_csr(
        layer_offset,
        layer_count,
        layer_thickness_m,
        layer_eps_r,
        layer_sigma_e,
        layer_mu_r,
        source.get_device(),
    )
    if frequency_hz <= 0.0:
        raise ValueError("frequency_hz must be positive")
    out = _required_native_op("field_transmission_sequence")(
        path_valid,
        source,
        target,
        interaction_positions,
        interaction_normals,
        interaction_material_id,
        interaction_valid,
        tx_power,
        tx_polarization,
        rx_polarization,
        layer_offset,
        layer_count,
        layer_thickness_m,
        layer_eps_r,
        layer_sigma_e,
        layer_mu_r,
        float(frequency_hz),
    )
    return _validate_enumerated_field_result(
        out,
        count,
        type_error="_channel.field_transmission_sequence must return a dict",
        field_error="field_transmission_sequence returned unexpected fields",
        shape_error_prefix="field_transmission_sequence",
    )


def field_coupled_rd(
    source: torch.Tensor,
    target: torch.Tensor,
    reflection_position: torch.Tensor,
    reflection_normal: torch.Tensor,
    edge_position: torch.Tensor,
    edge_direction: torch.Tensor,
    edge_n0: torch.Tensor,
    edge_n1: torch.Tensor,
    exterior_angle: torch.Tensor,
    tx_power: torch.Tensor,
    tx_polarization: torch.Tensor,
    rx_polarization: torch.Tensor,
    reflection_material: tuple[torch.Tensor, ...],
    wedge_material0: tuple[torch.Tensor, ...],
    wedge_material1: tuple[torch.Tensor, ...],
    edge_line_min: torch.Tensor,
    edge_line_max: torch.Tensor,
    *,
    frequency_hz: float,
    reverse: bool,
) -> dict[str, torch.Tensor]:
    vectors = (
        source,
        target,
        reflection_position,
        reflection_normal,
        edge_position,
        edge_direction,
        edge_n0,
        edge_n1,
        tx_polarization,
        rx_polarization,
    )
    count = int(source.shape[0])
    for value in vectors:
        validate_cuda_tensor(
            "coupled_vector", value, dtype=torch.float32, ndim=2, trailing_shape=(3,)
        )
        if value.shape != (count, 3):
            raise ValueError("coupled field vector tensors must have shape (N, 3)")
    if any(len(bundle) != 5 for bundle in (reflection_material, wedge_material0, wedge_material1)):
        raise ValueError("coupled material bundles must contain eps/sigma/mu/gain/thickness")
    scalars = (
        exterior_angle,
        tx_power,
        *reflection_material,
        *wedge_material0,
        *wedge_material1,
        edge_line_min,
        edge_line_max,
    )
    for value in scalars:
        validate_cuda_tensor("coupled_scalar", value, dtype=torch.float32, ndim=1)
        if value.shape != (count,):
            raise ValueError("coupled field scalar tensors must have shape (N,)")
    if frequency_hz <= 0.0:
        raise ValueError("frequency_hz must be positive")
    out = _required_native_op("field_coupled_rd")(
        *vectors[:8],
        exterior_angle,
        tx_power,
        tx_polarization,
        rx_polarization,
        *reflection_material,
        *wedge_material0,
        *wedge_material1,
        edge_line_min,
        edge_line_max,
        float(frequency_hz),
        bool(reverse),
    )
    if not isinstance(out, dict):
        raise TypeError("_channel.field_coupled_rd must return a dict")
    schema = {
        "field_vector": (torch.complex64, 2, (count, 3)),
        "coefficient": (torch.complex64, 1, (count,)),
        "path_field": (torch.complex64, 1, (count,)),
        "path_gain": (torch.float32, 1, (count,)),
        "direction": (torch.float32, 2, (count, 3)),
    }
    if set(out) != set(schema):
        raise ValueError("field_coupled_rd returned unexpected fields")
    for name, (dtype, ndim, shape) in schema.items():
        validate_cuda_tensor(name, out[name], dtype=dtype, ndim=ndim)
        if tuple(out[name].shape) != shape:
            raise ValueError(f"field_coupled_rd returned bad {name} shape")
    return out


def field_coupled_dd(
    source: torch.Tensor,
    target: torch.Tensor,
    edge1_position: torch.Tensor,
    edge1_direction: torch.Tensor,
    edge1_n0: torch.Tensor,
    edge1_n1: torch.Tensor,
    edge1_exterior: torch.Tensor,
    edge2_position: torch.Tensor,
    edge2_direction: torch.Tensor,
    edge2_n0: torch.Tensor,
    edge2_n1: torch.Tensor,
    edge2_exterior: torch.Tensor,
    tx_power: torch.Tensor,
    tx_polarization: torch.Tensor,
    rx_polarization: torch.Tensor,
    wedge1_material0: tuple[torch.Tensor, ...],
    wedge1_material1: tuple[torch.Tensor, ...],
    wedge2_material0: tuple[torch.Tensor, ...],
    wedge2_material1: tuple[torch.Tensor, ...],
    edge1_line_min: torch.Tensor,
    edge1_line_max: torch.Tensor,
    edge2_line_min: torch.Tensor,
    edge2_line_max: torch.Tensor,
    *,
    frequency_hz: float,
) -> dict[str, torch.Tensor]:
    """Coupled double-diffraction field (TX->e1->e2->RX), component id 7.

    Two sequential wedge operators in one native launch (ADR-013 D3). Outputs
    are identical in shape to :func:`field_coupled_rd`.
    """

    vectors = (
        source,
        target,
        edge1_position,
        edge1_direction,
        edge1_n0,
        edge1_n1,
        edge2_position,
        edge2_direction,
        edge2_n0,
        edge2_n1,
        tx_polarization,
        rx_polarization,
    )
    count = int(source.shape[0])
    for value in vectors:
        validate_cuda_tensor(
            "coupled_dd_vector", value, dtype=torch.float32, ndim=2, trailing_shape=(3,)
        )
        if value.shape != (count, 3):
            raise ValueError("coupled dd field vector tensors must have shape (N, 3)")
    if any(
        len(bundle) != 5
        for bundle in (
            wedge1_material0,
            wedge1_material1,
            wedge2_material0,
            wedge2_material1,
        )
    ):
        raise ValueError(
            "coupled dd material bundles must contain eps/sigma/mu/gain/thickness"
        )
    scalars = (
        edge1_exterior,
        edge2_exterior,
        tx_power,
        *wedge1_material0,
        *wedge1_material1,
        *wedge2_material0,
        *wedge2_material1,
        edge1_line_min,
        edge1_line_max,
        edge2_line_min,
        edge2_line_max,
    )
    for value in scalars:
        validate_cuda_tensor("coupled_dd_scalar", value, dtype=torch.float32, ndim=1)
        if value.shape != (count,):
            raise ValueError("coupled dd field scalar tensors must have shape (N,)")
    if frequency_hz <= 0.0:
        raise ValueError("frequency_hz must be positive")
    out = _required_native_op("field_coupled_dd")(
        source,
        target,
        edge1_position,
        edge1_direction,
        edge1_n0,
        edge1_n1,
        edge1_exterior,
        edge2_position,
        edge2_direction,
        edge2_n0,
        edge2_n1,
        edge2_exterior,
        tx_power,
        tx_polarization,
        rx_polarization,
        *wedge1_material0,
        *wedge1_material1,
        *wedge2_material0,
        *wedge2_material1,
        edge1_line_min,
        edge1_line_max,
        edge2_line_min,
        edge2_line_max,
        float(frequency_hz),
    )
    if not isinstance(out, dict):
        raise TypeError("_channel.field_coupled_dd must return a dict")
    schema = {
        "field_vector": (torch.complex64, 2, (count, 3)),
        "coefficient": (torch.complex64, 1, (count,)),
        "path_field": (torch.complex64, 1, (count,)),
        "path_gain": (torch.float32, 1, (count,)),
        "direction": (torch.float32, 2, (count, 3)),
    }
    if set(out) != set(schema):
        raise ValueError("field_coupled_dd returned unexpected fields")
    for name, (dtype, ndim, shape) in schema.items():
        validate_cuda_tensor(name, out[name], dtype=dtype, ndim=ndim)
        if tuple(out[name].shape) != shape:
            raise ValueError(f"field_coupled_dd returned bad {name} shape")
    return out


_FIELD_AD_OUTPUT_FIELDS = (
    "field_vector",
    "coefficient",
    "path_field",
    "path_gain",
    "path_length_m",
    "delay_s",
    "direction",
)


_FIELD_AD_TANGENT_FIELDS = (
    "field_vector",
    "coefficient",
    "path_field",
    "path_gain",
    "path_length_m",
    "delay_s",
)


# The two Channel-owned transports additionally publish the arrival-direction
# tangent (ADR-043). Transmission and the wedge/coupled families forward to
# rayd::torch, which owns their direction seam and does not publish it, so the
# shared tuple above stays exactly what those families return.
_FIELD_AD_DIRECTION_TANGENT_FIELDS = (*_FIELD_AD_TANGENT_FIELDS, "direction")


def field_free_space_backward(
    source: torch.Tensor,
    target: torch.Tensor,
    tx_power: torch.Tensor,
    tx_polarization: torch.Tensor,
    rx_polarization: torch.Tensor,
    *,
    frequency_hz: float,
    grad_field_vector: torch.Tensor | None = None,
    grad_coefficient: torch.Tensor | None = None,
    grad_path_field: torch.Tensor | None = None,
    grad_path_gain: torch.Tensor | None = None,
    grad_path_length: torch.Tensor | None = None,
    grad_delay: torch.Tensor | None = None,
    grad_direction: torch.Tensor | None = None,
    need_grad_frequency: bool = True,
    need_grad_geometry: bool = False,
) -> dict[str, torch.Tensor | None]:
    out = _required_native_op("field_free_space_backward")(
        source,
        target,
        tx_power,
        tx_polarization,
        rx_polarization,
        float(frequency_hz),
        grad_field_vector,
        grad_coefficient,
        grad_path_field,
        grad_path_gain,
        grad_path_length,
        grad_delay,
        grad_direction,
        bool(need_grad_frequency),
        bool(need_grad_geometry),
    )
    expected = {"grad_frequency", "grad_source", "grad_target"}
    if not isinstance(out, dict) or set(out) != expected:
        raise TypeError("_channel.field_free_space_backward returned invalid fields")
    return out


def field_free_space_jvp(
    source: torch.Tensor,
    target: torch.Tensor,
    tx_power: torch.Tensor,
    tx_polarization: torch.Tensor,
    rx_polarization: torch.Tensor,
    *,
    frequency_hz: float,
    tangent_frequency: float,
    tangent_source: torch.Tensor | None = None,
    tangent_target: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    out = _required_native_op("field_free_space_jvp")(
        source,
        target,
        tx_power,
        tx_polarization,
        rx_polarization,
        float(frequency_hz),
        float(tangent_frequency),
        tangent_source,
        tangent_target,
    )
    if not isinstance(out, dict) or set(out) != set(
        _FIELD_AD_DIRECTION_TANGENT_FIELDS
    ):
        raise TypeError("_channel.field_free_space_jvp returned invalid fields")
    return out


def field_reflection_sequence_backward(
    source: torch.Tensor,
    target: torch.Tensor,
    interaction_positions: torch.Tensor,
    interaction_normals: torch.Tensor,
    tx_power: torch.Tensor,
    tx_polarization: torch.Tensor,
    rx_polarization: torch.Tensor,
    eps_r: torch.Tensor,
    sigma_e: torch.Tensor,
    mu_r: torch.Tensor,
    gain: torch.Tensor,
    thickness: torch.Tensor,
    *,
    frequency_hz: float,
    grad_field_vector: torch.Tensor | None = None,
    grad_coefficient: torch.Tensor | None = None,
    grad_path_field: torch.Tensor | None = None,
    grad_path_gain: torch.Tensor | None = None,
    grad_path_length: torch.Tensor | None = None,
    grad_delay: torch.Tensor | None = None,
    grad_direction: torch.Tensor | None = None,
    need_grad_eps_r: bool = True,
    need_grad_sigma_e: bool = True,
    need_grad_gain: bool = False,
    need_grad_thickness: bool = True,
    need_grad_frequency: bool = True,
    need_grad_geometry: bool = False,
) -> dict[str, torch.Tensor | None]:
    out = _required_native_op("field_reflection_sequence_backward")(
        source,
        target,
        interaction_positions,
        interaction_normals,
        tx_power,
        tx_polarization,
        rx_polarization,
        eps_r,
        sigma_e,
        mu_r,
        gain,
        thickness,
        float(frequency_hz),
        grad_field_vector,
        grad_coefficient,
        grad_path_field,
        grad_path_gain,
        grad_path_length,
        grad_delay,
        grad_direction,
        bool(need_grad_eps_r),
        bool(need_grad_sigma_e),
        bool(need_grad_gain),
        bool(need_grad_thickness),
        bool(need_grad_frequency),
        bool(need_grad_geometry),
    )
    expected = {
        "grad_eps_r",
        "grad_sigma_e",
        "grad_gain",
        "grad_thickness",
        "grad_frequency",
        "grad_source",
        "grad_target",
        "grad_interaction_positions",
        "grad_interaction_normals",
    }
    if not isinstance(out, dict) or set(out) != expected:
        raise TypeError(
            "_channel.field_reflection_sequence_backward returned invalid fields"
        )
    return out


def field_reflection_sequence_jvp(
    source: torch.Tensor,
    target: torch.Tensor,
    interaction_positions: torch.Tensor,
    interaction_normals: torch.Tensor,
    tx_power: torch.Tensor,
    tx_polarization: torch.Tensor,
    rx_polarization: torch.Tensor,
    eps_r: torch.Tensor,
    sigma_e: torch.Tensor,
    mu_r: torch.Tensor,
    gain: torch.Tensor,
    thickness: torch.Tensor,
    *,
    frequency_hz: float,
    tangent_eps_r: torch.Tensor | None = None,
    tangent_sigma_e: torch.Tensor | None = None,
    tangent_gain: torch.Tensor | None = None,
    tangent_thickness: torch.Tensor | None = None,
    tangent_frequency: float = 0.0,
    tangent_source: torch.Tensor | None = None,
    tangent_target: torch.Tensor | None = None,
    tangent_interaction_positions: torch.Tensor | None = None,
    tangent_interaction_normals: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    out = _required_native_op("field_reflection_sequence_jvp")(
        source,
        target,
        interaction_positions,
        interaction_normals,
        tx_power,
        tx_polarization,
        rx_polarization,
        eps_r,
        sigma_e,
        mu_r,
        gain,
        thickness,
        float(frequency_hz),
        tangent_eps_r,
        tangent_sigma_e,
        tangent_gain,
        tangent_thickness,
        float(tangent_frequency),
        tangent_source,
        tangent_target,
        tangent_interaction_positions,
        tangent_interaction_normals,
    )
    if not isinstance(out, dict) or set(out) != set(
        _FIELD_AD_DIRECTION_TANGENT_FIELDS
    ):
        raise TypeError(
            "_channel.field_reflection_sequence_jvp returned invalid fields"
        )
    return out


def field_transmission_sequence_backward(
    path_valid: torch.Tensor,
    source: torch.Tensor,
    target: torch.Tensor,
    interaction_positions: torch.Tensor,
    interaction_normals: torch.Tensor,
    interaction_material_id: torch.Tensor,
    interaction_valid: torch.Tensor,
    tx_power: torch.Tensor,
    tx_polarization: torch.Tensor,
    rx_polarization: torch.Tensor,
    layer_offset: torch.Tensor,
    layer_count: torch.Tensor,
    layer_thickness_m: torch.Tensor,
    layer_eps_r: torch.Tensor,
    layer_sigma_e: torch.Tensor,
    layer_mu_r: torch.Tensor,
    *,
    frequency_hz: float,
    grad_field_vector: torch.Tensor | None = None,
    grad_coefficient: torch.Tensor | None = None,
    grad_path_field: torch.Tensor | None = None,
    grad_path_gain: torch.Tensor | None = None,
    grad_path_length: torch.Tensor | None = None,
    grad_delay: torch.Tensor | None = None,
    need_grad_layer_thickness: bool = True,
    need_grad_layer_eps_r: bool = True,
    need_grad_layer_sigma_e: bool = True,
    need_grad_frequency: bool = True,
    need_grad_geometry: bool = False,
) -> dict[str, torch.Tensor | None]:
    out = _required_native_op("field_transmission_sequence_backward")(
        path_valid,
        source,
        target,
        interaction_positions,
        interaction_normals,
        interaction_material_id,
        interaction_valid,
        tx_power,
        tx_polarization,
        rx_polarization,
        layer_offset,
        layer_count,
        layer_thickness_m,
        layer_eps_r,
        layer_sigma_e,
        layer_mu_r,
        float(frequency_hz),
        grad_field_vector,
        grad_coefficient,
        grad_path_field,
        grad_path_gain,
        grad_path_length,
        grad_delay,
        bool(need_grad_layer_thickness),
        bool(need_grad_layer_eps_r),
        bool(need_grad_layer_sigma_e),
        bool(need_grad_frequency),
        bool(need_grad_geometry),
    )
    expected = {
        "grad_layer_thickness_m",
        "grad_layer_eps_r",
        "grad_layer_sigma_e",
        "grad_frequency",
        "grad_source",
        "grad_target",
        "grad_interaction_positions",
        "grad_interaction_normals",
    }
    if not isinstance(out, dict) or set(out) != expected:
        raise TypeError(
            "_channel.field_transmission_sequence_backward returned invalid fields"
        )
    return out


def field_transmission_sequence_jvp(
    path_valid: torch.Tensor,
    source: torch.Tensor,
    target: torch.Tensor,
    interaction_positions: torch.Tensor,
    interaction_normals: torch.Tensor,
    interaction_material_id: torch.Tensor,
    interaction_valid: torch.Tensor,
    tx_power: torch.Tensor,
    tx_polarization: torch.Tensor,
    rx_polarization: torch.Tensor,
    layer_offset: torch.Tensor,
    layer_count: torch.Tensor,
    layer_thickness_m: torch.Tensor,
    layer_eps_r: torch.Tensor,
    layer_sigma_e: torch.Tensor,
    layer_mu_r: torch.Tensor,
    *,
    frequency_hz: float,
    tangent_layer_thickness_m: torch.Tensor | None = None,
    tangent_layer_eps_r: torch.Tensor | None = None,
    tangent_layer_sigma_e: torch.Tensor | None = None,
    tangent_frequency: float = 0.0,
    tangent_source: torch.Tensor | None = None,
    tangent_target: torch.Tensor | None = None,
    tangent_interaction_positions: torch.Tensor | None = None,
    tangent_interaction_normals: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    out = _required_native_op("field_transmission_sequence_jvp")(
        path_valid,
        source,
        target,
        interaction_positions,
        interaction_normals,
        interaction_material_id,
        interaction_valid,
        tx_power,
        tx_polarization,
        rx_polarization,
        layer_offset,
        layer_count,
        layer_thickness_m,
        layer_eps_r,
        layer_sigma_e,
        layer_mu_r,
        float(frequency_hz),
        tangent_layer_thickness_m,
        tangent_layer_eps_r,
        tangent_layer_sigma_e,
        float(tangent_frequency),
        tangent_source,
        tangent_target,
        tangent_interaction_positions,
        tangent_interaction_normals,
    )
    if not isinstance(out, dict) or set(out) != set(_FIELD_AD_TANGENT_FIELDS):
        raise TypeError(
            "_channel.field_transmission_sequence_jvp returned invalid fields"
        )
    return out


_WEDGE_OUTPUT_FIELDS = ("field_vector", "direction")


_COUPLED_OUTPUT_FIELDS = (
    "field_vector",
    "coefficient",
    "path_field",
    "path_gain",
    "direction",
)


def _validate_wedge_valid(valid: torch.Tensor, source: torch.Tensor) -> None:
    validate_cuda_tensor("valid", valid, dtype=torch.bool, ndim=1)
    if valid.shape != (int(source.shape[0]),):
        raise ValueError("valid must have shape (N,) matching source rows")


def field_diffraction_wedge(
    valid: torch.Tensor,
    source: torch.Tensor,
    target: torch.Tensor,
    edge_position: torch.Tensor,
    edge_direction: torch.Tensor,
    edge_t_min: torch.Tensor,
    edge_t_max: torch.Tensor,
    edge_n0: torch.Tensor,
    edge_n1: torch.Tensor,
    exterior_angle: torch.Tensor,
    face0_valid: torch.Tensor,
    face0_eps_r: torch.Tensor,
    face0_sigma_e: torch.Tensor,
    face0_mu_r: torch.Tensor,
    face0_gain: torch.Tensor,
    face1_valid: torch.Tensor,
    face1_eps_r: torch.Tensor,
    face1_sigma_e: torch.Tensor,
    face1_mu_r: torch.Tensor,
    face1_gain: torch.Tensor,
    tx_power: torch.Tensor,
    *,
    frequency_hz: float,
) -> dict[str, torch.Tensor]:
    """Re-evaluate RayD's order-1 UTD wedge export from the frozen topology."""

    _validate_wedge_valid(valid, source)
    out = _required_native_op("field_diffraction_wedge")(
        valid,
        source,
        target,
        edge_position,
        edge_direction,
        edge_t_min,
        edge_t_max,
        edge_n0,
        edge_n1,
        exterior_angle,
        face0_valid,
        face0_eps_r,
        face0_sigma_e,
        face0_mu_r,
        face0_gain,
        face1_valid,
        face1_eps_r,
        face1_sigma_e,
        face1_mu_r,
        face1_gain,
        tx_power,
        float(frequency_hz),
        None,
        None,
        None,
        None,
        None,
    )
    if not isinstance(out, dict) or set(out) != set(_WEDGE_OUTPUT_FIELDS):
        raise TypeError(
            "_channel.field_diffraction_wedge returned invalid fields"
        )
    return out


_ROUGH_SCALE_OUTPUT_FIELDS = (
    "field_vector",
    "coefficient",
    "path_field",
    "path_gain",
)
_ROUGH_SCALE_TANGENT_FIELDS = tuple(
    f"tangent_{name}" for name in _ROUGH_SCALE_OUTPUT_FIELDS
)


def field_rough_reflection_scale(
    field_vector: torch.Tensor,
    coefficient: torch.Tensor,
    path_field: torch.Tensor,
    path_gain: torch.Tensor,
    positions: torch.Tensor,
    normals: torch.Tensor,
    source: torch.Tensor,
    sigma_b: torch.Tensor,
    rough_b: torch.Tensor,
    replaced: torch.Tensor,
    *,
    frequency_hz: float,
) -> dict[str, torch.Tensor]:
    """Native rough-surface C_r factor applied to the reflection outputs.

    Computes ``C_r = prod_b exp(-2*(k0*cos_b*sigma_b)^2)`` on rough bounces
    (``1`` otherwise), zeroes rows flagged ``replaced``, and scales the four
    field outputs (``path_gain`` by ``C_r^2``). Returns the scaled outputs plus
    the real ``factor`` per row.
    """

    validate_cuda_tensor("field_vector", field_vector, dtype=torch.complex64, ndim=2)
    validate_cuda_tensor("coefficient", coefficient, dtype=torch.complex64, ndim=1)
    validate_cuda_tensor("path_field", path_field, dtype=torch.complex64, ndim=1)
    validate_cuda_tensor("path_gain", path_gain, dtype=torch.float32, ndim=1)
    validate_cuda_tensor("positions", positions, dtype=torch.float32, ndim=3)
    validate_cuda_tensor("normals", normals, dtype=torch.float32, ndim=3)
    validate_cuda_tensor("source", source, dtype=torch.float32, ndim=2)
    validate_cuda_tensor("sigma_b", sigma_b, dtype=torch.float32, ndim=2)
    validate_cuda_tensor("rough_b", rough_b, dtype=torch.bool, ndim=2)
    validate_cuda_tensor("replaced", replaced, dtype=torch.bool, ndim=1)
    out = _required_native_op("field_rough_reflection_scale")(
        field_vector,
        coefficient,
        path_field,
        path_gain,
        positions,
        normals,
        source,
        sigma_b,
        rough_b,
        replaced,
        float(frequency_hz),
    )
    expected = {*_ROUGH_SCALE_OUTPUT_FIELDS, "factor"}
    if not isinstance(out, dict) or set(out) != expected:
        raise TypeError(
            "_channel.field_rough_reflection_scale returned invalid fields"
        )
    return out


def field_rough_reflection_scale_backward(
    field_vector: torch.Tensor,
    coefficient: torch.Tensor,
    path_field: torch.Tensor,
    path_gain: torch.Tensor,
    positions: torch.Tensor,
    normals: torch.Tensor,
    source: torch.Tensor,
    sigma_b: torch.Tensor,
    rough_b: torch.Tensor,
    replaced: torch.Tensor,
    *,
    frequency_hz: float,
    grad_field_vector: torch.Tensor | None = None,
    grad_coefficient: torch.Tensor | None = None,
    grad_path_field: torch.Tensor | None = None,
    grad_path_gain: torch.Tensor | None = None,
    need_field: bool = True,
    need_geometry: bool = False,
    need_frequency: bool = True,
) -> dict[str, torch.Tensor | None]:
    """VJP of :func:`field_rough_reflection_scale` (frequency and geometry)."""

    out = _required_native_op("field_rough_reflection_scale_backward")(
        field_vector,
        coefficient,
        path_field,
        path_gain,
        positions,
        normals,
        source,
        sigma_b,
        rough_b,
        replaced,
        float(frequency_hz),
        grad_field_vector,
        grad_coefficient,
        grad_path_field,
        grad_path_gain,
        bool(need_field),
        bool(need_geometry),
        bool(need_frequency),
    )
    expected = {
        "grad_field_vector",
        "grad_coefficient",
        "grad_path_field",
        "grad_path_gain",
        "grad_positions",
        "grad_normals",
        "grad_source",
        "grad_frequency",
    }
    if not isinstance(out, dict) or set(out) != expected:
        raise TypeError(
            "_channel.field_rough_reflection_scale_backward returned"
            " invalid fields"
        )
    return out


def field_rough_reflection_scale_jvp(
    field_vector: torch.Tensor,
    coefficient: torch.Tensor,
    path_field: torch.Tensor,
    path_gain: torch.Tensor,
    positions: torch.Tensor,
    normals: torch.Tensor,
    source: torch.Tensor,
    sigma_b: torch.Tensor,
    rough_b: torch.Tensor,
    replaced: torch.Tensor,
    *,
    frequency_hz: float,
    tangent_field_vector: torch.Tensor | None = None,
    tangent_coefficient: torch.Tensor | None = None,
    tangent_path_field: torch.Tensor | None = None,
    tangent_path_gain: torch.Tensor | None = None,
    tangent_positions: torch.Tensor | None = None,
    tangent_normals: torch.Tensor | None = None,
    tangent_source: torch.Tensor | None = None,
    tangent_frequency: float = 0.0,
) -> dict[str, torch.Tensor]:
    """JVP of :func:`field_rough_reflection_scale` (frequency and geometry)."""

    out = _required_native_op("field_rough_reflection_scale_jvp")(
        field_vector,
        coefficient,
        path_field,
        path_gain,
        positions,
        normals,
        source,
        sigma_b,
        rough_b,
        replaced,
        float(frequency_hz),
        tangent_field_vector,
        tangent_coefficient,
        tangent_path_field,
        tangent_path_gain,
        tangent_positions,
        tangent_normals,
        tangent_source,
        float(tangent_frequency),
    )
    if not isinstance(out, dict) or set(out) != set(_ROUGH_SCALE_TANGENT_FIELDS):
        raise TypeError(
            "_channel.field_rough_reflection_scale_jvp returned invalid fields"
        )
    return out


def ad_liveness(direction_live: bool, *geometry: object) -> tuple[bool, bool]:
    """The one liveness record an apply carries, decided by the wrapper.

    A direction derivative is a geometry derivative, so the direction half can
    only be live where the geometry half is. The other half of the direction
    decision is the caller's host-known component set, which is why it arrives
    as a flag rather than being inferred from a tensor.
    """

    geometry_live = _ad_geometry_live(*geometry)
    return (geometry_live, geometry_live and bool(direction_live))


def mark_dead_outputs(ctx, output) -> None:
    """Declare the outputs of one apply that carry no derivative.

    A dead output is marked exactly as it was before the direction seam
    existed, so a caller that does not ask for a live direction sees the same
    object graph it saw at contract version 5.
    """

    dead = []
    if not ctx.geometry_live:
        dead.extend((output[4], output[5]))
    if not ctx.direction_live:
        dead.append(output[6])
    if dead:
        ctx.mark_non_differentiable(*dead)


def direction_cotangent(ctx, grad_direction):
    """The incoming direction cotangent, or ``None`` if it was declared dead.

    Torch does not deliver a cotangent for an output marked non-differentiable,
    so this is belt and braces; it is also the one place that says out loud that
    a dead output's seed never reaches a native companion.
    """

    return grad_direction if ctx.direction_live else None


def direction_tangents(ctx, out: dict[str, torch.Tensor]) -> tuple:
    """Publish one apply's output tangents under the two liveness decisions.

    The native companion always computes the direction tangent - it is the dual
    the transverse projection already needed - so this only decides whether it
    is published. A dead output receives ``None`` rather than a zero tensor,
    because a zero tangent on a declared-dead output is exactly the silent
    answer ADR-043 removes.
    """

    tangents = tuple(out[name] for name in _FIELD_AD_DIRECTION_TANGENT_FIELDS)
    if not ctx.geometry_live:
        return (*tangents[:4], None, None, None)
    if not ctx.direction_live:
        return (*tangents[:6], None)
    return tangents


class _FieldFreeSpaceAdFunction(torch.autograd.Function):
    """Fixed-topology differentiable free-space transport.

    Frequency and endpoints are differentiable; power and polarizations are
    fixed. Float64 inputs use the strict-double companion for gradcheck.
    """

    @staticmethod
    def forward(
        source,
        target,
        tx_power,
        tx_polarization,
        rx_polarization,
        frequency,
        frequency_value,
        liveness,
    ):
        op_name = (
            "field_free_space_fwd64"
            if source.dtype == torch.float64
            else "field_free_space"
        )
        out = _required_native_op(op_name)(
            source,
            target,
            tx_power,
            tx_polarization,
            rx_polarization,
            frequency_value,
        )
        return tuple(out[name] for name in _FIELD_AD_OUTPUT_FIELDS)

    @staticmethod
    def setup_context(ctx, inputs, output):
        ctx.set_materialize_grads(False)
        (
            source,
            target,
            tx_power,
            tx_polarization,
            rx_polarization,
            frequency,
            frequency_value,
            liveness,
        ) = inputs
        primals = tuple(
            torch.autograd.forward_ad.unpack_dual(value).primal
            for value in (source, target, tx_power, tx_polarization, rx_polarization)
        )
        ctx.frequency_value = frequency_value
        ctx.frequency_meta = (
            (frequency.dtype, frequency.device)
            if isinstance(frequency, torch.Tensor)
            else None
        )
        # Both decided by the wrapper, where forward duals are still visible;
        # Function.apply unpacks them before this hook runs.
        ctx.geometry_live, ctx.direction_live = liveness
        ctx.save_for_backward(*primals)
        ctx.save_for_forward(*primals)
        mark_dead_outputs(ctx, output)

    @staticmethod
    @_ad_first_order_only
    def backward(
        ctx,
        grad_field_vector,
        grad_coefficient,
        grad_path_field,
        grad_path_gain,
        grad_path_length,
        grad_delay,
        grad_direction,
    ):
        none_grads = (None,) * 8
        _ad_reject_fixed_inputs(
            "field_free_space_ad",
            ctx.needs_input_grad,
            (
                (2, "tx_power"),
                (3, "tx_polarization"),
                (4, "rx_polarization"),
            ),
        )
        need_geometry = bool(ctx.needs_input_grad[0]) or bool(
            ctx.needs_input_grad[1]
        )
        need_frequency = bool(ctx.needs_input_grad[5])
        grad_direction = direction_cotangent(ctx, grad_direction)
        grads = (
            grad_field_vector,
            grad_coefficient,
            grad_path_field,
            grad_path_gain,
            grad_path_length,
            grad_delay,
            grad_direction,
        )
        if not (need_geometry or need_frequency) or all(
            value is None for value in grads
        ):
            return none_grads
        source, target, tx_power, tx_polarization, rx_polarization = ctx.saved_tensors
        out = field_free_space_backward(
            source,
            target,
            tx_power,
            tx_polarization,
            rx_polarization,
            frequency_hz=ctx.frequency_value,
            grad_field_vector=grad_field_vector,
            grad_coefficient=grad_coefficient,
            grad_path_field=grad_path_field,
            grad_path_gain=grad_path_gain,
            grad_path_length=grad_path_length,
            grad_delay=grad_delay,
            grad_direction=grad_direction,
            need_grad_frequency=need_frequency,
            need_grad_geometry=need_geometry,
        )
        grad_frequency = (
            _ad_frequency_grad(out["grad_frequency"], ctx.frequency_meta)
            if need_frequency
            else None
        )
        return (
            out["grad_source"] if ctx.needs_input_grad[0] else None,
            out["grad_target"] if ctx.needs_input_grad[1] else None,
            None,
            None,
            None,
            grad_frequency,
            None,
            None,
        )

    @staticmethod
    def jvp(
        ctx,
        t_source,
        t_target,
        t_tx_power,
        t_tx_pol,
        t_rx_pol,
        t_frequency,
        _t_frequency_value,
        _t_liveness,
    ):
        _ad_reject_fixed_tangents(
            "field_free_space_ad",
            (
                (t_tx_power, "tx_power"),
                (t_tx_pol, "tx_polarization"),
                (t_rx_pol, "rx_polarization"),
            ),
        )
        saved = ctx.saved_tensors
        tangent_source = _ad_geometry_tangent(
            "field_free_space_ad tangent_source", t_source, saved[0]
        )
        tangent_target = _ad_geometry_tangent(
            "field_free_space_ad tangent_target", t_target, saved[1]
        )
        tangent_frequency = _ad_frequency_tangent(t_frequency)
        if (
            tangent_frequency == 0.0
            and tangent_source is None
            and tangent_target is None
        ):
            return (None,) * 7
        source, target, tx_power, tx_polarization, rx_polarization = saved
        with disable_functorch():
            out = field_free_space_jvp(
                _ad_native_tensor(source),
                _ad_native_tensor(target),
                _ad_native_tensor(tx_power),
                _ad_native_tensor(tx_polarization),
                _ad_native_tensor(rx_polarization),
                frequency_hz=ctx.frequency_value,
                tangent_frequency=tangent_frequency,
                tangent_source=tangent_source,
                tangent_target=tangent_target,
            )
        return direction_tangents(ctx, out)


def field_free_space_ad(
    source: torch.Tensor,
    target: torch.Tensor,
    tx_power: torch.Tensor,
    tx_polarization: torch.Tensor,
    rx_polarization: torch.Tensor,
    *,
    frequency: torch.Tensor | float,
    frequency_value: float | None = None,
    direction_live: bool = False,
) -> dict[str, torch.Tensor]:
    """Differentiable :func:`field_free_space` (frequency only in AD-1).

    ``frequency_value`` optionally carries the precomputed host scalar of
    ``frequency`` (one read per solve at the seam, audit M3); when not
    supplied it is read here, exactly once per apply.
    """

    if frequency_value is None:
        frequency_value = _ad_frequency_value(frequency)
    values = _FieldFreeSpaceAdFunction.apply(
        source,
        target,
        tx_power,
        tx_polarization,
        rx_polarization,
        frequency,
        float(frequency_value),
        ad_liveness(direction_live, source, target),
    )
    return dict(zip(_FIELD_AD_OUTPUT_FIELDS, values, strict=True))


class _FieldReflectionSequenceAdFunction(torch.autograd.Function):
    """Fixed-topology differentiable reflection transport.

    Frequency, hit geometry, and per-bounce material scalars except ``mu_r``
    are differentiable; power, polarizations, and ``mu_r`` stay fixed.
    """

    @staticmethod
    def forward(
        source,
        target,
        interaction_positions,
        interaction_normals,
        tx_power,
        tx_polarization,
        rx_polarization,
        eps_r,
        sigma_e,
        mu_r,
        gain,
        thickness,
        frequency,
        frequency_value,
        liveness,
    ):
        out = _required_native_op("field_reflection_sequence")(
            source,
            target,
            interaction_positions,
            interaction_normals,
            tx_power,
            tx_polarization,
            rx_polarization,
            eps_r,
            sigma_e,
            mu_r,
            gain,
            thickness,
            frequency_value,
        )
        return tuple(out[name] for name in _FIELD_AD_OUTPUT_FIELDS)

    @staticmethod
    def setup_context(ctx, inputs, output):
        ctx.set_materialize_grads(False)
        frequency = inputs[12]
        primals = tuple(
            torch.autograd.forward_ad.unpack_dual(value).primal
            for value in inputs[:12]
        )
        ctx.frequency_value = inputs[13]
        ctx.frequency_meta = (
            (frequency.dtype, frequency.device)
            if isinstance(frequency, torch.Tensor)
            else None
        )
        # Both decided by the wrapper, where forward duals are still visible.
        ctx.geometry_live, ctx.direction_live = inputs[14]
        ctx.save_for_backward(*primals)
        ctx.save_for_forward(*primals)
        mark_dead_outputs(ctx, output)

    @staticmethod
    @_ad_first_order_only
    def backward(
        ctx,
        grad_field_vector,
        grad_coefficient,
        grad_path_field,
        grad_path_gain,
        grad_path_length,
        grad_delay,
        grad_direction,
    ):
        none_grads = (None,) * 15
        _ad_reject_fixed_inputs(
            "field_reflection_sequence_ad",
            ctx.needs_input_grad,
            (
                (4, "tx_power"),
                (5, "tx_polarization"),
                (6, "rx_polarization"),
                (9, "mu_r"),
            ),
        )
        need_geometry = any(bool(ctx.needs_input_grad[i]) for i in range(4))
        need_eps = bool(ctx.needs_input_grad[7])
        need_sigma = bool(ctx.needs_input_grad[8])
        need_gain = bool(ctx.needs_input_grad[10])
        need_thickness = bool(ctx.needs_input_grad[11])
        need_frequency = bool(ctx.needs_input_grad[12])
        grad_direction = direction_cotangent(ctx, grad_direction)
        grads = (
            grad_field_vector,
            grad_coefficient,
            grad_path_field,
            grad_path_gain,
            grad_path_length,
            grad_delay,
            grad_direction,
        )
        if not (
            need_geometry
            or need_eps
            or need_sigma
            or need_gain
            or need_thickness
            or need_frequency
        ) or all(value is None for value in grads):
            return none_grads
        (
            source,
            target,
            interaction_positions,
            interaction_normals,
            tx_power,
            tx_polarization,
            rx_polarization,
            eps_r,
            sigma_e,
            mu_r,
            gain,
            thickness,
        ) = ctx.saved_tensors
        out = field_reflection_sequence_backward(
            source,
            target,
            interaction_positions,
            interaction_normals,
            tx_power,
            tx_polarization,
            rx_polarization,
            eps_r,
            sigma_e,
            mu_r,
            gain,
            thickness,
            frequency_hz=ctx.frequency_value,
            grad_field_vector=grad_field_vector,
            grad_coefficient=grad_coefficient,
            grad_path_field=grad_path_field,
            grad_path_gain=grad_path_gain,
            grad_path_length=grad_path_length,
            grad_delay=grad_delay,
            grad_direction=grad_direction,
            need_grad_eps_r=need_eps,
            need_grad_sigma_e=need_sigma,
            need_grad_gain=need_gain,
            need_grad_thickness=need_thickness,
            need_grad_frequency=need_frequency,
            need_grad_geometry=need_geometry,
        )
        grad_frequency = (
            _ad_frequency_grad(out["grad_frequency"], ctx.frequency_meta)
            if need_frequency
            else None
        )
        return (
            out["grad_source"] if ctx.needs_input_grad[0] else None,
            out["grad_target"] if ctx.needs_input_grad[1] else None,
            out["grad_interaction_positions"] if ctx.needs_input_grad[2] else None,
            out["grad_interaction_normals"] if ctx.needs_input_grad[3] else None,
            None,
            None,
            None,
            out["grad_eps_r"] if need_eps else None,
            out["grad_sigma_e"] if need_sigma else None,
            None,
            out["grad_gain"] if need_gain else None,
            out["grad_thickness"] if need_thickness else None,
            grad_frequency,
            None,
            None,
        )

    @staticmethod
    def jvp(
        ctx,
        t_source,
        t_target,
        t_positions,
        t_normals,
        t_tx_power,
        t_tx_pol,
        t_rx_pol,
        t_eps_r,
        t_sigma_e,
        t_mu_r,
        t_gain,
        t_thickness,
        t_frequency,
        _t_frequency_value,
        _t_liveness,
    ):
        _ad_reject_fixed_tangents(
            "field_reflection_sequence_ad",
            (
                (t_tx_power, "tx_power"),
                (t_tx_pol, "tx_polarization"),
                (t_rx_pol, "rx_polarization"),
                (t_mu_r, "mu_r"),
            ),
        )
        saved = ctx.saved_tensors
        eps_shape = tuple(saved[7].shape)
        tangent_source = _ad_geometry_tangent(
            "field_reflection_sequence_ad tangent_source", t_source, saved[0]
        )
        tangent_target = _ad_geometry_tangent(
            "field_reflection_sequence_ad tangent_target", t_target, saved[1]
        )
        tangent_positions = _ad_geometry_tangent(
            "field_reflection_sequence_ad tangent_interaction_positions",
            t_positions,
            saved[2],
        )
        tangent_normals = _ad_geometry_tangent(
            "field_reflection_sequence_ad tangent_interaction_normals",
            t_normals,
            saved[3],
        )
        tangent_eps = _ad_checked_tangent(
            "field_reflection_sequence_ad tangent_eps_r",
            _ad_native_tangent_or_none(t_eps_r),
            eps_shape,
        )
        tangent_sigma = _ad_checked_tangent(
            "field_reflection_sequence_ad tangent_sigma_e",
            _ad_native_tangent_or_none(t_sigma_e),
            eps_shape,
        )
        tangent_gain = _ad_checked_tangent(
            "field_reflection_sequence_ad tangent_gain",
            _ad_native_tangent_or_none(t_gain),
            eps_shape,
        )
        tangent_thickness = _ad_checked_tangent(
            "field_reflection_sequence_ad tangent_thickness",
            _ad_native_tangent_or_none(t_thickness),
            eps_shape,
        )
        tangent_frequency = _ad_frequency_tangent(t_frequency)
        if (
            tangent_eps is None
            and tangent_sigma is None
            and tangent_gain is None
            and tangent_thickness is None
            and tangent_source is None
            and tangent_target is None
            and tangent_positions is None
            and tangent_normals is None
            and tangent_frequency == 0.0
        ):
            return (None,) * 7
        with disable_functorch():
            out = field_reflection_sequence_jvp(
                *(_ad_native_tensor(value) for value in saved),
                frequency_hz=ctx.frequency_value,
                tangent_eps_r=tangent_eps,
                tangent_sigma_e=tangent_sigma,
                tangent_gain=tangent_gain,
                tangent_thickness=tangent_thickness,
                tangent_frequency=tangent_frequency,
                tangent_source=tangent_source,
                tangent_target=tangent_target,
                tangent_interaction_positions=tangent_positions,
                tangent_interaction_normals=tangent_normals,
            )
        return direction_tangents(ctx, out)


def field_reflection_sequence_ad(
    source: torch.Tensor,
    target: torch.Tensor,
    interaction_positions: torch.Tensor,
    interaction_normals: torch.Tensor,
    tx_power: torch.Tensor,
    tx_polarization: torch.Tensor,
    rx_polarization: torch.Tensor,
    eps_r: torch.Tensor,
    sigma_e: torch.Tensor,
    mu_r: torch.Tensor,
    gain: torch.Tensor,
    thickness: torch.Tensor,
    *,
    frequency: torch.Tensor | float,
    frequency_value: float | None = None,
    direction_live: bool = False,
) -> dict[str, torch.Tensor]:
    """Differentiable :func:`field_reflection_sequence` (materials + frequency).

    ``frequency_value`` optionally carries the precomputed host scalar of
    ``frequency`` (one read per solve at the seam, audit M3); when not
    supplied it is read here, exactly once per apply.
    """

    if frequency_value is None:
        frequency_value = _ad_frequency_value(frequency)
    values = _FieldReflectionSequenceAdFunction.apply(
        source,
        target,
        interaction_positions,
        interaction_normals,
        tx_power,
        tx_polarization,
        rx_polarization,
        eps_r,
        sigma_e,
        mu_r,
        gain,
        thickness,
        frequency,
        float(frequency_value),
        ad_liveness(
            direction_live, source, target, interaction_positions, interaction_normals
        ),
    )
    return dict(zip(_FIELD_AD_OUTPUT_FIELDS, values, strict=True))


class _FieldTransmissionSequenceAdFunction(torch.autograd.Function):
    """Fixed-topology differentiable transmission transport (plan 07 AD-1/AD-2).

    Differentiable inputs: CSR layer thickness / eps_r / sigma_e, frequency,
    and the hit geometry (source, target, interaction_normals). The straight
    transmission field is independent of the crossing points themselves, so
    interaction_positions receives an exact zero gradient (None). tx_power,
    the polarizations, layer_mu_r, material ids and valid masks are fixed;
    requesting their gradient fails loudly. Layer gradients accumulate
    atomically across paths because the CSR store is shared by every wall
    crossing.
    """

    @staticmethod
    def forward(
        path_valid,
        source,
        target,
        interaction_positions,
        interaction_normals,
        interaction_material_id,
        interaction_valid,
        tx_power,
        tx_polarization,
        rx_polarization,
        layer_offset,
        layer_count,
        layer_thickness_m,
        layer_eps_r,
        layer_sigma_e,
        layer_mu_r,
        frequency,
        frequency_value,
    ):
        out = _required_native_op("field_transmission_sequence")(
            path_valid,
            source,
            target,
            interaction_positions,
            interaction_normals,
            interaction_material_id,
            interaction_valid,
            tx_power,
            tx_polarization,
            rx_polarization,
            layer_offset,
            layer_count,
            layer_thickness_m,
            layer_eps_r,
            layer_sigma_e,
            layer_mu_r,
            frequency_value,
        )
        return tuple(out[name] for name in _FIELD_AD_OUTPUT_FIELDS)

    @staticmethod
    def setup_context(ctx, inputs, output):
        ctx.set_materialize_grads(False)
        frequency = inputs[16]
        primals = tuple(
            torch.autograd.forward_ad.unpack_dual(value).primal
            for value in inputs[:16]
        )
        ctx.frequency_value = inputs[17]
        ctx.frequency_meta = (
            (frequency.dtype, frequency.device)
            if isinstance(frequency, torch.Tensor)
            else None
        )
        ctx.geometry_live = _ad_geometry_live(*inputs[1:5])
        ctx.save_for_backward(*primals)
        ctx.save_for_forward(*primals)
        # ADR-043: RayD owns this family's direction seam, so the arrival
        # direction is a declared non-differentiable output on every route.
        ctx.direction_live = False
        mark_dead_outputs(ctx, output)

    @staticmethod
    @_ad_first_order_only
    def backward(
        ctx,
        grad_field_vector,
        grad_coefficient,
        grad_path_field,
        grad_path_gain,
        grad_path_length,
        grad_delay,
        _grad_direction,
    ):
        none_grads = (None,) * 18
        _ad_reject_fixed_inputs(
            "field_transmission_sequence_ad",
            ctx.needs_input_grad,
            (
                (7, "tx_power"),
                (8, "tx_polarization"),
                (9, "rx_polarization"),
                (15, "layer_mu_r"),
            ),
        )
        # interaction_positions (index 3) never enters the straight-path
        # field: its gradient is exactly zero, so it does not drive a launch.
        need_geometry = (
            bool(ctx.needs_input_grad[1])
            or bool(ctx.needs_input_grad[2])
            or bool(ctx.needs_input_grad[4])
        )
        need_thickness = bool(ctx.needs_input_grad[12])
        need_eps = bool(ctx.needs_input_grad[13])
        need_sigma = bool(ctx.needs_input_grad[14])
        need_frequency = bool(ctx.needs_input_grad[16])
        grads = (
            grad_field_vector,
            grad_coefficient,
            grad_path_field,
            grad_path_gain,
            grad_path_length,
            grad_delay,
        )
        if not (
            need_geometry or need_thickness or need_eps or need_sigma
            or need_frequency
        ) or all(value is None for value in grads):
            return none_grads
        saved = ctx.saved_tensors
        out = field_transmission_sequence_backward(
            *saved,
            frequency_hz=ctx.frequency_value,
            grad_field_vector=grad_field_vector,
            grad_coefficient=grad_coefficient,
            grad_path_field=grad_path_field,
            grad_path_gain=grad_path_gain,
            grad_path_length=grad_path_length,
            grad_delay=grad_delay,
            need_grad_layer_thickness=need_thickness,
            need_grad_layer_eps_r=need_eps,
            need_grad_layer_sigma_e=need_sigma,
            need_grad_frequency=need_frequency,
            need_grad_geometry=need_geometry,
        )
        grad_frequency = (
            _ad_frequency_grad(out["grad_frequency"], ctx.frequency_meta)
            if need_frequency
            else None
        )
        return (
            None,
            out["grad_source"] if ctx.needs_input_grad[1] else None,
            out["grad_target"] if ctx.needs_input_grad[2] else None,
            None,
            out["grad_interaction_normals"] if ctx.needs_input_grad[4] else None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            out["grad_layer_thickness_m"] if need_thickness else None,
            out["grad_layer_eps_r"] if need_eps else None,
            out["grad_layer_sigma_e"] if need_sigma else None,
            None,
            grad_frequency,
            None,
        )

    @staticmethod
    def jvp(
        ctx,
        _t_path_valid,
        t_source,
        t_target,
        t_positions,
        t_normals,
        _t_material_id,
        _t_interaction_valid,
        t_tx_power,
        t_tx_pol,
        t_rx_pol,
        _t_layer_offset,
        _t_layer_count,
        t_layer_thickness,
        t_layer_eps_r,
        t_layer_sigma_e,
        t_layer_mu_r,
        t_frequency,
        _t_frequency_value,
    ):
        _ad_reject_fixed_tangents(
            "field_transmission_sequence_ad",
            (
                (t_tx_power, "tx_power"),
                (t_tx_pol, "tx_polarization"),
                (t_rx_pol, "rx_polarization"),
                (t_layer_mu_r, "layer_mu_r"),
            ),
        )
        saved = ctx.saved_tensors
        layer_shape = tuple(saved[12].shape)
        tangent_source = _ad_geometry_tangent(
            "field_transmission_sequence_ad tangent_source", t_source, saved[1]
        )
        tangent_target = _ad_geometry_tangent(
            "field_transmission_sequence_ad tangent_target", t_target, saved[2]
        )
        tangent_positions = _ad_geometry_tangent(
            "field_transmission_sequence_ad tangent_interaction_positions",
            t_positions,
            saved[3],
        )
        tangent_normals = _ad_geometry_tangent(
            "field_transmission_sequence_ad tangent_interaction_normals",
            t_normals,
            saved[4],
        )
        tangent_thickness = _ad_checked_tangent(
            "field_transmission_sequence_ad tangent_layer_thickness_m",
            _ad_native_tangent_or_none(t_layer_thickness),
            layer_shape,
        )
        tangent_eps = _ad_checked_tangent(
            "field_transmission_sequence_ad tangent_layer_eps_r",
            _ad_native_tangent_or_none(t_layer_eps_r),
            layer_shape,
        )
        tangent_sigma = _ad_checked_tangent(
            "field_transmission_sequence_ad tangent_layer_sigma_e",
            _ad_native_tangent_or_none(t_layer_sigma_e),
            layer_shape,
        )
        tangent_frequency = _ad_frequency_tangent(t_frequency)
        if (
            tangent_thickness is None
            and tangent_eps is None
            and tangent_sigma is None
            and tangent_source is None
            and tangent_target is None
            and tangent_positions is None
            and tangent_normals is None
            and tangent_frequency == 0.0
        ):
            return (None,) * 7
        with disable_functorch():
            out = field_transmission_sequence_jvp(
                *(_ad_native_tensor(value) for value in saved),
                frequency_hz=ctx.frequency_value,
                tangent_layer_thickness_m=tangent_thickness,
                tangent_layer_eps_r=tangent_eps,
                tangent_layer_sigma_e=tangent_sigma,
                tangent_frequency=tangent_frequency,
                tangent_source=tangent_source,
                tangent_target=tangent_target,
                tangent_interaction_positions=tangent_positions,
                tangent_interaction_normals=tangent_normals,
            )
        tangents = tuple(out[name] for name in _FIELD_AD_TANGENT_FIELDS)
        if not ctx.geometry_live:
            return (*tangents[:4], None, None, None)
        return (*tangents, None)


def field_transmission_sequence_ad(
    path_valid: torch.Tensor,
    source: torch.Tensor,
    target: torch.Tensor,
    interaction_positions: torch.Tensor,
    interaction_normals: torch.Tensor,
    interaction_material_id: torch.Tensor,
    interaction_valid: torch.Tensor,
    tx_power: torch.Tensor,
    tx_polarization: torch.Tensor,
    rx_polarization: torch.Tensor,
    layer_offset: torch.Tensor,
    layer_count: torch.Tensor,
    layer_thickness_m: torch.Tensor,
    layer_eps_r: torch.Tensor,
    layer_sigma_e: torch.Tensor,
    layer_mu_r: torch.Tensor,
    *,
    frequency: torch.Tensor | float,
    frequency_value: float | None = None,
) -> dict[str, torch.Tensor]:
    """Differentiable :func:`field_transmission_sequence` (layers + frequency).

    ``frequency_value`` optionally carries the precomputed host scalar of
    ``frequency`` (one read per solve at the seam, audit M3); when not
    supplied it is read here, exactly once per apply.
    """

    if frequency_value is None:
        frequency_value = _ad_frequency_value(frequency)
    values = _FieldTransmissionSequenceAdFunction.apply(
        path_valid,
        source,
        target,
        interaction_positions,
        interaction_normals,
        interaction_material_id,
        interaction_valid,
        tx_power,
        tx_polarization,
        rx_polarization,
        layer_offset,
        layer_count,
        layer_thickness_m,
        layer_eps_r,
        layer_sigma_e,
        layer_mu_r,
        frequency,
        float(frequency_value),
    )
    return dict(zip(_FIELD_AD_OUTPUT_FIELDS, values, strict=True))


class _FieldDiffractionWedgeAdFunction(torch.autograd.Function):
    """Fixed-topology differentiable UTD wedge field (plan 07 AD-4).

    Differentiable inputs: both faces' eps_r / sigma_e / gain, frequency,
    endpoints, and optional per-row winner vertices v0/v1 plus each face's
    opposite vertex. The kernel rebuilds edge tables so mesh-vertex gradients
    reach edge geometry. Frozen edge tables, valid masks, mu_r and tx_power
    stay fixed and reject gradients. The stationary edge point is re-solved
    inside the kernel, preserving its endpoint and vertex gradient motion.
    """

    @staticmethod
    def forward(
        valid,
        source,
        target,
        edge_position,
        edge_direction,
        edge_t_min,
        edge_t_max,
        edge_n0,
        edge_n1,
        exterior_angle,
        face0_valid,
        face0_eps_r,
        face0_sigma_e,
        face0_mu_r,
        face0_gain,
        face1_valid,
        face1_eps_r,
        face1_sigma_e,
        face1_mu_r,
        face1_gain,
        tx_power,
        frequency,
        vertex_v0,
        vertex_v1,
        vertex_opp0,
        vertex_opp1,
        edge_boundary,
        frequency_value,
    ):
        out = _required_native_op("field_diffraction_wedge")(
            valid,
            source,
            target,
            edge_position,
            edge_direction,
            edge_t_min,
            edge_t_max,
            edge_n0,
            edge_n1,
            exterior_angle,
            face0_valid,
            face0_eps_r,
            face0_sigma_e,
            face0_mu_r,
            face0_gain,
            face1_valid,
            face1_eps_r,
            face1_sigma_e,
            face1_mu_r,
            face1_gain,
            tx_power,
            frequency_value,
            vertex_v0,
            vertex_v1,
            vertex_opp0,
            vertex_opp1,
            edge_boundary,
            # ISB boundary taper (ADR-017), D member. Always 0.0: taper + AD is
            # refused by the deterministic/path pipelines (gate 3, C1 clearance
            # companion pending), so the differentiable twin never tapers. The
            # argument is threaded for lockstep completeness of the guarded path.
            0.0,
        )
        return tuple(out[name] for name in _WEDGE_OUTPUT_FIELDS)

    @staticmethod
    def setup_context(ctx, inputs, output):
        ctx.set_materialize_grads(False)
        frequency = inputs[21]
        primals = tuple(
            torch.autograd.forward_ad.unpack_dual(value).primal
            for value in inputs[:21]
        )
        vertex_primals = tuple(
            torch.autograd.forward_ad.unpack_dual(value).primal
            if isinstance(value, torch.Tensor)
            else value
            for value in inputs[22:27]
        )
        ctx.has_vertices = vertex_primals[0] is not None
        ctx.frequency_value = inputs[27]
        ctx.frequency_meta = (
            (frequency.dtype, frequency.device)
            if isinstance(frequency, torch.Tensor)
            else None
        )
        ctx.geometry_live = _ad_geometry_live(inputs[1], inputs[2])
        saved = primals + tuple(
            value for value in vertex_primals if value is not None
        )
        ctx.save_for_backward(*saved)
        ctx.save_for_forward(*saved)

    @staticmethod
    def _unpack_saved(ctx):
        saved = ctx.saved_tensors
        primals = saved[:21]
        vertices = saved[21:26] if ctx.has_vertices else (None,) * 5
        return primals, vertices

    @staticmethod
    @_ad_first_order_only
    def backward(ctx, grad_field_vector, grad_direction):
        none_grads = (None,) * 28
        _ad_reject_fixed_inputs(
            "field_diffraction_wedge_ad",
            ctx.needs_input_grad,
            (
                (0, "valid"),
                (3, "edge_position"),
                (4, "edge_direction"),
                (5, "edge_t_min"),
                (6, "edge_t_max"),
                (7, "edge_n0"),
                (8, "edge_n1"),
                (9, "exterior_angle"),
                (10, "face0_valid"),
                (13, "face0_mu_r"),
                (15, "face1_valid"),
                (18, "face1_mu_r"),
                (20, "tx_power"),
                (26, "edge_boundary"),
            ),
        )
        need_geometry = bool(ctx.needs_input_grad[1]) or bool(ctx.needs_input_grad[2])
        need_material = any(
            bool(ctx.needs_input_grad[index]) for index in (11, 12, 14, 16, 17, 19)
        )
        need_frequency = bool(ctx.needs_input_grad[21])
        need_vertices = any(bool(ctx.needs_input_grad[i]) for i in (22, 23, 24, 25))
        if not (need_geometry or need_material or need_frequency or need_vertices) or (
            grad_field_vector is None and grad_direction is None
        ):
            return none_grads
        primals, vertices = _FieldDiffractionWedgeAdFunction._unpack_saved(ctx)
        out = _required_native_op("field_diffraction_wedge_backward")(
            *primals,
            ctx.frequency_value,
            *vertices,
            grad_field_vector,
            grad_direction,
            need_material,
            need_frequency,
            need_geometry,
            need_vertices,
            # ADR-017 D-member width; always 0.0 (taper + AD is guarded off).
            0.0,
        )
        grad_frequency = (
            _ad_frequency_grad(out["grad_frequency"], ctx.frequency_meta)
            if need_frequency
            else None
        )
        return (
            None,
            out["grad_source"] if ctx.needs_input_grad[1] else None,
            out["grad_target"] if ctx.needs_input_grad[2] else None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            out["grad_face0_eps_r"] if ctx.needs_input_grad[11] else None,
            out["grad_face0_sigma_e"] if ctx.needs_input_grad[12] else None,
            None,
            out["grad_face0_gain"] if ctx.needs_input_grad[14] else None,
            None,
            out["grad_face1_eps_r"] if ctx.needs_input_grad[16] else None,
            out["grad_face1_sigma_e"] if ctx.needs_input_grad[17] else None,
            None,
            out["grad_face1_gain"] if ctx.needs_input_grad[19] else None,
            None,
            grad_frequency,
            out["grad_vertex_v0"] if ctx.needs_input_grad[22] else None,
            out["grad_vertex_v1"] if ctx.needs_input_grad[23] else None,
            out["grad_vertex_opp0"] if ctx.needs_input_grad[24] else None,
            out["grad_vertex_opp1"] if ctx.needs_input_grad[25] else None,
            None,
            None,
        )

    @staticmethod
    def jvp(ctx, *tangents):
        _ad_reject_fixed_tangents(
            "field_diffraction_wedge_ad",
            (
                (tangents[0], "valid"),
                (tangents[3], "edge_position"),
                (tangents[4], "edge_direction"),
                (tangents[5], "edge_t_min"),
                (tangents[6], "edge_t_max"),
                (tangents[7], "edge_n0"),
                (tangents[8], "edge_n1"),
                (tangents[9], "exterior_angle"),
                (tangents[13], "face0_mu_r"),
                (tangents[18], "face1_mu_r"),
                (tangents[20], "tx_power"),
            ),
        )
        primals, vertices = _FieldDiffractionWedgeAdFunction._unpack_saved(ctx)
        scalar_shape = tuple(primals[11].shape)
        tangent_source = _ad_geometry_tangent(
            "field_diffraction_wedge_ad tangent_source", tangents[1], primals[1])
        tangent_target = _ad_geometry_tangent(
            "field_diffraction_wedge_ad tangent_target", tangents[2], primals[2])
        material_tangents = {}
        for index, name in (
            (11, "face0_eps_r"),
            (12, "face0_sigma_e"),
            (14, "face0_gain"),
            (16, "face1_eps_r"),
            (17, "face1_sigma_e"),
            (19, "face1_gain"),
        ):
            material_tangents[name] = _ad_checked_tangent(
                f"field_diffraction_wedge_ad tangent_{name}",
                _ad_native_tangent_or_none(tangents[index]),
                scalar_shape,
            )
        vertex_tangents = []
        for index, name in (
            (22, "vertex_v0"),
            (23, "vertex_v1"),
            (24, "vertex_opp0"),
            (25, "vertex_opp1"),
        ):
            tangent = tangents[index] if index < len(tangents) else None
            vertex_tangents.append(
                _ad_native_tangent_or_none(
                    tangent if isinstance(tangent, torch.Tensor) else None
                )
            )
        tangent_frequency = _ad_frequency_tangent(tangents[21])
        if (
            tangent_source is None
            and tangent_target is None
            and tangent_frequency == 0.0
            and all(value is None for value in material_tangents.values())
            and all(value is None for value in vertex_tangents)
        ):
            return (None, None)
        with disable_functorch():
            out = _required_native_op("field_diffraction_wedge_jvp")(
                *(_ad_native_tensor(value) for value in primals),
                ctx.frequency_value,
                *(
                    _ad_native_tensor(value) if isinstance(value, torch.Tensor)
                    else value
                    for value in vertices
                ),
                tangent_source,
                tangent_target,
                material_tangents["face0_eps_r"],
                material_tangents["face0_sigma_e"],
                material_tangents["face0_gain"],
                material_tangents["face1_eps_r"],
                material_tangents["face1_sigma_e"],
                material_tangents["face1_gain"],
                float(tangent_frequency),
                *vertex_tangents,
                # ADR-017 D-member width; always 0.0 (taper + AD is guarded off).
                0.0,
            )
        return (out["tangent_field_vector"], out["tangent_direction"])


def field_diffraction_wedge_ad(
    valid: torch.Tensor,
    source: torch.Tensor,
    target: torch.Tensor,
    edge_position: torch.Tensor,
    edge_direction: torch.Tensor,
    edge_t_min: torch.Tensor,
    edge_t_max: torch.Tensor,
    edge_n0: torch.Tensor,
    edge_n1: torch.Tensor,
    exterior_angle: torch.Tensor,
    face0_valid: torch.Tensor,
    face0_eps_r: torch.Tensor,
    face0_sigma_e: torch.Tensor,
    face0_mu_r: torch.Tensor,
    face0_gain: torch.Tensor,
    face1_valid: torch.Tensor,
    face1_eps_r: torch.Tensor,
    face1_sigma_e: torch.Tensor,
    face1_mu_r: torch.Tensor,
    face1_gain: torch.Tensor,
    tx_power: torch.Tensor,
    *,
    frequency: torch.Tensor | float,
    frequency_value: float | None = None,
    vertices: tuple[torch.Tensor, ...] | None = None,
) -> dict[str, torch.Tensor]:
    """Differentiable :func:`field_diffraction_wedge`.

    ``vertices`` optionally supplies the winner edge vertices as
    ``(v0, v1, opp0, opp1, edge_boundary)`` per row; the kernel then rebuilds
    the edge tables from them so mesh-vertex gradients exist (plan 07 section
    9.3 mesh-vertex x diffraction). ``frequency_value`` optionally carries
    the precomputed host scalar of ``frequency`` (one read per solve at the
    seam, audit M3); when not supplied it is read here, exactly once per
    apply.
    """

    _validate_wedge_valid(valid, source)
    if vertices is not None and len(vertices) != 5:
        raise ValueError(
            "vertices must hold (v0, v1, opp0, opp1, edge_boundary) per row"
        )
    if frequency_value is None:
        frequency_value = _ad_frequency_value(frequency)
    vertex_args = vertices if vertices is not None else (None,) * 5
    values = _FieldDiffractionWedgeAdFunction.apply(
        valid,
        source,
        target,
        edge_position,
        edge_direction,
        edge_t_min,
        edge_t_max,
        edge_n0,
        edge_n1,
        exterior_angle,
        face0_valid,
        face0_eps_r,
        face0_sigma_e,
        face0_mu_r,
        face0_gain,
        face1_valid,
        face1_eps_r,
        face1_sigma_e,
        face1_mu_r,
        face1_gain,
        tx_power,
        frequency,
        *vertex_args,
        float(frequency_value),
    )
    return dict(zip(_WEDGE_OUTPUT_FIELDS, values, strict=True))


class _FieldCoupledRdAdFunction(torch.autograd.Function):
    """Fixed-topology differentiable coupled R-D transport (plan 07 AD-4).

    Differentiable inputs: eps_r / sigma_e / gain / thickness for the
    reflection wall and both wedge faces (12 scalars per path), frequency,
    and the continuous geometry (source, target, reflection_position,
    edge_position). The wedge axis/normals/exterior angle, the reflection
    normal (a frozen wall plane), mu_r, tx_power and the polarizations stay
    fixed. The pseudo-infinite edge truncation factor is a frozen regularizer
    of the differentiation (see kernels/field_wedge_ad.cu).
    """

    @staticmethod
    def forward(
        source,
        target,
        reflection_position,
        reflection_normal,
        edge_position,
        edge_direction,
        edge_n0,
        edge_n1,
        exterior_angle,
        tx_power,
        tx_polarization,
        rx_polarization,
        refl_eps_r,
        refl_sigma_e,
        refl_mu_r,
        refl_gain,
        refl_thickness,
        w0_eps_r,
        w0_sigma_e,
        w0_mu_r,
        w0_gain,
        w0_thickness,
        w1_eps_r,
        w1_sigma_e,
        w1_mu_r,
        w1_gain,
        w1_thickness,
        frequency,
        reverse,
        frequency_value,
        edge_line_min,
        edge_line_max,
    ):
        out = _required_native_op("field_coupled_rd")(
            source,
            target,
            reflection_position,
            reflection_normal,
            edge_position,
            edge_direction,
            edge_n0,
            edge_n1,
            exterior_angle,
            tx_power,
            tx_polarization,
            rx_polarization,
            refl_eps_r,
            refl_sigma_e,
            refl_mu_r,
            refl_gain,
            refl_thickness,
            w0_eps_r,
            w0_sigma_e,
            w0_mu_r,
            w0_gain,
            w0_thickness,
            w1_eps_r,
            w1_sigma_e,
            w1_mu_r,
            w1_gain,
            w1_thickness,
            edge_line_min,
            edge_line_max,
            frequency_value,
            bool(reverse),
        )
        return tuple(out[name] for name in _COUPLED_OUTPUT_FIELDS)

    @staticmethod
    def setup_context(ctx, inputs, output):
        ctx.set_materialize_grads(False)
        frequency = inputs[27]
        primals = tuple(
            torch.autograd.forward_ad.unpack_dual(value).primal
            for value in inputs[:27]
        )
        # edge_line_min / edge_line_max (inputs[30], inputs[31]) are frozen edge
        # bounds (G4): non-differentiable, but saved so the backward/jvp
        # companions can forward them to the native coupled kernels in the same
        # position as the primal.
        bounds = tuple(
            torch.autograd.forward_ad.unpack_dual(value).primal
            for value in (inputs[30], inputs[31])
        )
        ctx.frequency_value = inputs[29]
        ctx.frequency_meta = (
            (frequency.dtype, frequency.device)
            if isinstance(frequency, torch.Tensor)
            else None
        )
        ctx.reverse = bool(inputs[28])
        ctx.save_for_backward(*primals, *bounds)
        ctx.save_for_forward(*primals, *bounds)
        ctx.mark_non_differentiable(output[4])

    @staticmethod
    @_ad_first_order_only
    def backward(
        ctx,
        grad_field_vector,
        grad_coefficient,
        grad_path_field,
        grad_path_gain,
        _grad_direction,
    ):
        none_grads = (None,) * 32
        _ad_reject_fixed_inputs(
            "field_coupled_rd_ad",
            ctx.needs_input_grad,
            (
                (3, "reflection_normal"),
                (5, "edge_direction"),
                (6, "edge_n0"),
                (7, "edge_n1"),
                (8, "exterior_angle"),
                (9, "tx_power"),
                (10, "tx_polarization"),
                (11, "rx_polarization"),
                (14, "reflection_mu_r"),
                (19, "wedge_mu_r0"),
                (24, "wedge_mu_r1"),
                (30, "edge_line_min"),
                (31, "edge_line_max"),
            ),
        )
        need_geometry = any(
            bool(ctx.needs_input_grad[index]) for index in (0, 1, 2, 4)
        )
        need_eps = any(bool(ctx.needs_input_grad[index]) for index in (12, 17, 22))
        need_sigma = any(bool(ctx.needs_input_grad[index]) for index in (13, 18, 23))
        need_gain = any(bool(ctx.needs_input_grad[index]) for index in (15, 20, 25))
        need_thickness = any(
            bool(ctx.needs_input_grad[index]) for index in (16, 21, 26)
        )
        need_frequency = bool(ctx.needs_input_grad[27])
        grads = (grad_field_vector, grad_coefficient, grad_path_field, grad_path_gain)
        if not (
            need_geometry
            or need_eps
            or need_sigma
            or need_gain
            or need_thickness
            or need_frequency
        ) or all(value is None for value in grads):
            return none_grads
        saved = ctx.saved_tensors
        out = _required_native_op("field_coupled_rd_backward")(
            *saved,
            ctx.frequency_value,
            ctx.reverse,
            grad_field_vector,
            grad_coefficient,
            grad_path_field,
            grad_path_gain,
            need_eps,
            need_sigma,
            need_gain,
            need_thickness,
            need_frequency,
            need_geometry,
        )
        grad_frequency = (
            _ad_frequency_grad(out["grad_frequency"], ctx.frequency_meta)
            if need_frequency
            else None
        )

        def material_column(name: str, column: int, index: int):
            if not ctx.needs_input_grad[index]:
                return None
            return out[name][:, column]

        return (
            out["grad_source"] if ctx.needs_input_grad[0] else None,
            out["grad_target"] if ctx.needs_input_grad[1] else None,
            out["grad_reflection_position"] if ctx.needs_input_grad[2] else None,
            None,
            out["grad_edge_position"] if ctx.needs_input_grad[4] else None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            material_column("grad_eps_r", 0, 12),
            material_column("grad_sigma_e", 0, 13),
            None,
            material_column("grad_gain", 0, 15),
            material_column("grad_thickness", 0, 16),
            material_column("grad_eps_r", 1, 17),
            material_column("grad_sigma_e", 1, 18),
            None,
            material_column("grad_gain", 1, 20),
            material_column("grad_thickness", 1, 21),
            material_column("grad_eps_r", 2, 22),
            material_column("grad_sigma_e", 2, 23),
            None,
            material_column("grad_gain", 2, 25),
            material_column("grad_thickness", 2, 26),
            grad_frequency,
            None,
            None,
            None,
            None,
        )

    @staticmethod
    def jvp(ctx, *tangents):
        _ad_reject_fixed_tangents(
            "field_coupled_rd_ad",
            (
                (tangents[3], "reflection_normal"),
                (tangents[5], "edge_direction"),
                (tangents[6], "edge_n0"),
                (tangents[7], "edge_n1"),
                (tangents[8], "exterior_angle"),
                (tangents[9], "tx_power"),
                (tangents[10], "tx_polarization"),
                (tangents[11], "rx_polarization"),
                (tangents[14], "reflection_mu_r"),
                (tangents[19], "wedge_mu_r0"),
                (tangents[24], "wedge_mu_r1"),
                (tangents[30], "edge_line_min"),
                (tangents[31], "edge_line_max"),
            ),
        )
        saved = ctx.saved_tensors
        scalar_shape = tuple(saved[12].shape)
        tangent_source = _ad_geometry_tangent(
            "field_coupled_rd_ad tangent_source", tangents[0], saved[0]
        )
        tangent_target = _ad_geometry_tangent(
            "field_coupled_rd_ad tangent_target", tangents[1], saved[1]
        )
        tangent_hit = _ad_geometry_tangent(
            "field_coupled_rd_ad tangent_reflection_position",
            tangents[2],
            saved[2],
        )
        tangent_edge = _ad_geometry_tangent(
            "field_coupled_rd_ad tangent_edge_position", tangents[4], saved[4]
        )

        def material_pack(indices: tuple[int, int, int], name: str):
            columns = tuple(
                _ad_checked_tangent(
                    f"field_coupled_rd_ad tangent_{name}",
                    _ad_native_tangent_or_none(tangents[index]),
                    scalar_shape,
                )
                for index in indices
            )
            if all(column is None for column in columns):
                return None
            zero = torch.zeros(
                scalar_shape, device=saved[12].device, dtype=torch.float32
            )
            return torch.stack(
                [zero if column is None else column for column in columns], dim=1
            )

        tangent_eps = material_pack((12, 17, 22), "eps_r")
        tangent_sigma = material_pack((13, 18, 23), "sigma_e")
        tangent_gain = material_pack((15, 20, 25), "gain")
        tangent_thickness = material_pack((16, 21, 26), "thickness")
        tangent_frequency = _ad_frequency_tangent(tangents[27])
        if (
            tangent_source is None
            and tangent_target is None
            and tangent_hit is None
            and tangent_edge is None
            and tangent_eps is None
            and tangent_sigma is None
            and tangent_gain is None
            and tangent_thickness is None
            and tangent_frequency == 0.0
        ):
            return (None,) * 5
        with disable_functorch():
            out = _required_native_op("field_coupled_rd_jvp")(
                *(_ad_native_tensor(value) for value in saved),
                ctx.frequency_value,
                ctx.reverse,
                tangent_source,
                tangent_target,
                tangent_hit,
                tangent_edge,
                tangent_eps,
                tangent_sigma,
                tangent_gain,
                tangent_thickness,
                float(tangent_frequency),
            )
        return (
            out["tangent_field_vector"],
            out["tangent_coefficient"],
            out["tangent_path_field"],
            out["tangent_path_gain"],
            None,
        )


def field_coupled_rd_ad(
    source: torch.Tensor,
    target: torch.Tensor,
    reflection_position: torch.Tensor,
    reflection_normal: torch.Tensor,
    edge_position: torch.Tensor,
    edge_direction: torch.Tensor,
    edge_n0: torch.Tensor,
    edge_n1: torch.Tensor,
    exterior_angle: torch.Tensor,
    tx_power: torch.Tensor,
    tx_polarization: torch.Tensor,
    rx_polarization: torch.Tensor,
    reflection_material: tuple[torch.Tensor, ...],
    wedge_material0: tuple[torch.Tensor, ...],
    wedge_material1: tuple[torch.Tensor, ...],
    edge_line_min: torch.Tensor,
    edge_line_max: torch.Tensor,
    *,
    frequency: torch.Tensor | float,
    frequency_value: float | None = None,
    reverse: bool,
) -> dict[str, torch.Tensor]:
    """Differentiable :func:`field_coupled_rd` (12 material scalars + frequency + geometry).

    ``frequency_value`` optionally carries the precomputed host scalar of
    ``frequency`` (one read per solve at the seam, audit M3); when not
    supplied it is read here, exactly once per apply.
    """

    if any(
        len(bundle) != 5
        for bundle in (reflection_material, wedge_material0, wedge_material1)
    ):
        raise ValueError(
            "coupled material bundles must contain eps/sigma/mu/gain/thickness"
        )
    if frequency_value is None:
        frequency_value = _ad_frequency_value(frequency)
    values = _FieldCoupledRdAdFunction.apply(
        source,
        target,
        reflection_position,
        reflection_normal,
        edge_position,
        edge_direction,
        edge_n0,
        edge_n1,
        exterior_angle,
        tx_power,
        tx_polarization,
        rx_polarization,
        *reflection_material,
        *wedge_material0,
        *wedge_material1,
        frequency,
        bool(reverse),
        float(frequency_value),
        edge_line_min,
        edge_line_max,
    )
    return dict(zip(_COUPLED_OUTPUT_FIELDS, values, strict=True))


class _CoupledRdPrepareAdFunction(torch.autograd.Function):
    """Fixed-winner coupled stationary geometry (plan 07 AD-4).

    Re-solves the image source, the stationary diffraction point on the edge
    and the predicted wall crossing for the frozen winner (wall plane + edge
    line), so the coupled interaction points move with the endpoints on the
    autograd graph. Differentiable inputs: source and receiver.
    """

    @staticmethod
    def forward(
        source,
        receiver,
        plane_point,
        plane_normal,
        edge_pos,
        edge_dir,
        edge_t_min,
        edge_t_max,
    ):
        out = _required_native_op("coupled_rd_prepare")(
            source,
            receiver,
            plane_point,
            plane_normal,
            edge_pos,
            edge_dir,
            edge_t_min,
            edge_t_max,
        )
        active, edge_point, _virtual_source, reflection_point = out
        return (edge_point, reflection_point, active)

    @staticmethod
    def setup_context(ctx, inputs, output):
        ctx.set_materialize_grads(False)
        primals = tuple(
            torch.autograd.forward_ad.unpack_dual(value).primal
            for value in inputs[:7]
        )
        ctx.save_for_backward(*primals)
        ctx.save_for_forward(*primals)
        ctx.mark_non_differentiable(output[2])

    @staticmethod
    @_ad_first_order_only
    def backward(ctx, grad_edge_point, grad_reflection_point, _grad_active):
        none_grads = (None,) * 8
        _ad_reject_fixed_inputs(
            "coupled_rd_prepare_ad",
            ctx.needs_input_grad,
            (
                (2, "plane_point"),
                (3, "plane_normal"),
                (4, "edge_pos"),
                (5, "edge_dir"),
                (6, "edge_t_min"),
                (7, "edge_t_max"),
            ),
        )
        need_source = bool(ctx.needs_input_grad[0])
        need_receiver = bool(ctx.needs_input_grad[1])
        if not (need_source or need_receiver) or (
            grad_edge_point is None and grad_reflection_point is None
        ):
            return none_grads
        out = _required_native_op("coupled_rd_prepare_backward")(
            *ctx.saved_tensors,
            grad_edge_point,
            grad_reflection_point,
            need_source,
            need_receiver,
        )
        return (
            out["grad_source"] if need_source else None,
            out["grad_receiver"] if need_receiver else None,
            None,
            None,
            None,
            None,
            None,
            None,
        )

    @staticmethod
    def jvp(ctx, t_source, t_receiver, *t_rest):
        _ad_reject_fixed_tangents(
            "coupled_rd_prepare_ad",
            (
                (t_rest[0], "plane_point"),
                (t_rest[1], "plane_normal"),
                (t_rest[2], "edge_pos"),
                (t_rest[3], "edge_dir"),
                (t_rest[4], "edge_t_min"),
                (t_rest[5], "edge_t_max"),
            ),
        )
        saved = ctx.saved_tensors
        tangent_source = _ad_geometry_tangent(
            "coupled_rd_prepare_ad tangent_source", t_source, saved[0]
        )
        tangent_receiver = _ad_geometry_tangent(
            "coupled_rd_prepare_ad tangent_receiver", t_receiver, saved[1]
        )
        if tangent_source is None and tangent_receiver is None:
            return (None, None, None)
        with disable_functorch():
            out = _required_native_op("coupled_rd_prepare_jvp")(
                *(_ad_native_tensor(value) for value in saved),
                tangent_source,
                tangent_receiver,
            )
        return (
            out["tangent_edge_point"],
            out["tangent_reflection_point"],
            None,
        )


def coupled_rd_prepare_ad(
    source: torch.Tensor,
    receiver: torch.Tensor,
    plane_point: torch.Tensor,
    plane_normal: torch.Tensor,
    edge_pos: torch.Tensor,
    edge_dir: torch.Tensor,
    edge_t_min: torch.Tensor,
    edge_t_max: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Differentiable fixed-winner coupled stationary geometry re-solve."""

    edge_point, reflection_point, active = _CoupledRdPrepareAdFunction.apply(
        source,
        receiver,
        plane_point,
        plane_normal,
        edge_pos,
        edge_dir,
        edge_t_min,
        edge_t_max,
    )
    return {
        "edge_point": edge_point,
        "reflection_point": reflection_point,
        "active": active,
    }


# Material grad-request index groups for the four coupled double-diffraction
# wedge faces (wedge1 face0, wedge1 face1, wedge2 face0, wedge2 face1), mirroring
# the native (N, 4) column layout.
_DD_EPS_INDICES = (15, 20, 25, 30)
_DD_SIGMA_INDICES = (16, 21, 26, 31)
_DD_GAIN_INDICES = (18, 23, 28, 33)
_DD_THICKNESS_INDICES = (19, 24, 29, 34)


def _dd_backward_needs(needs_input_grad) -> dict[str, bool]:
    """Per-family gradient-request flags for the coupled-DD backward launch."""
    return {
        "geometry": any(bool(needs_input_grad[index]) for index in (0, 1)),
        "eps": any(bool(needs_input_grad[index]) for index in _DD_EPS_INDICES),
        "sigma": any(bool(needs_input_grad[index]) for index in _DD_SIGMA_INDICES),
        "gain": any(bool(needs_input_grad[index]) for index in _DD_GAIN_INDICES),
        "thickness": any(
            bool(needs_input_grad[index]) for index in _DD_THICKNESS_INDICES
        ),
        "frequency": bool(needs_input_grad[35]),
    }


def _dd_backward_is_noop(needs: dict[str, bool], grads: tuple) -> bool:
    """True when no requested input needs a gradient or every seed is None."""
    return not any(needs.values()) or all(value is None for value in grads)


class _FieldCoupledDdAdFunction(torch.autograd.Function):
    """Fixed-topology differentiable coupled double-diffraction transport (ADR-013 D4).

    Twin of :class:`_FieldCoupledRdAdFunction` for cid-7 (TX -> e1 -> e2 -> RX)
    rows. Differentiable inputs: eps_r / sigma_e / gain / thickness for the four
    wedge faces (16 scalars per path), frequency, and the tx/rx endpoints
    (source, target), whose gradients flow through the native per-leg
    re-anchoring. The frozen discovery seeds Q1/Q2 (edge1_position /
    edge2_position), the edge axes/normals/exterior angles, the edge bounds,
    mu_r, tx_power and the polarizations stay fixed; requesting their gradient
    fails loudly. Mesh-vertex gradients are refused one layer up (evaluation.py
    coupled block, ADR-013 D4).
    """

    @staticmethod
    def forward(
        source,
        target,
        edge1_position,
        edge1_direction,
        edge1_n0,
        edge1_n1,
        edge1_exterior,
        edge2_position,
        edge2_direction,
        edge2_n0,
        edge2_n1,
        edge2_exterior,
        tx_power,
        tx_polarization,
        rx_polarization,
        w1a_eps_r,
        w1a_sigma_e,
        w1a_mu_r,
        w1a_gain,
        w1a_thickness,
        w1b_eps_r,
        w1b_sigma_e,
        w1b_mu_r,
        w1b_gain,
        w1b_thickness,
        w2a_eps_r,
        w2a_sigma_e,
        w2a_mu_r,
        w2a_gain,
        w2a_thickness,
        w2b_eps_r,
        w2b_sigma_e,
        w2b_mu_r,
        w2b_gain,
        w2b_thickness,
        frequency,
        frequency_value,
        edge1_line_min,
        edge1_line_max,
        edge2_line_min,
        edge2_line_max,
    ):
        out = _required_native_op("field_coupled_dd")(
            source,
            target,
            edge1_position,
            edge1_direction,
            edge1_n0,
            edge1_n1,
            edge1_exterior,
            edge2_position,
            edge2_direction,
            edge2_n0,
            edge2_n1,
            edge2_exterior,
            tx_power,
            tx_polarization,
            rx_polarization,
            w1a_eps_r,
            w1a_sigma_e,
            w1a_mu_r,
            w1a_gain,
            w1a_thickness,
            w1b_eps_r,
            w1b_sigma_e,
            w1b_mu_r,
            w1b_gain,
            w1b_thickness,
            w2a_eps_r,
            w2a_sigma_e,
            w2a_mu_r,
            w2a_gain,
            w2a_thickness,
            w2b_eps_r,
            w2b_sigma_e,
            w2b_mu_r,
            w2b_gain,
            w2b_thickness,
            edge1_line_min,
            edge1_line_max,
            edge2_line_min,
            edge2_line_max,
            frequency_value,
        )
        return tuple(out[name] for name in _COUPLED_OUTPUT_FIELDS)

    @staticmethod
    def setup_context(ctx, inputs, output):
        ctx.set_materialize_grads(False)
        frequency = inputs[35]
        # The native backward/jvp take the 35 primal field tensors (source ..
        # wedge2 face1 thickness) followed by the four frozen edge bounds, then
        # the host frequency scalar. Q1/Q2/bounds are non-differentiable (ADR-013
        # D4) but saved so the companions forward them in the primal position.
        primals = tuple(
            torch.autograd.forward_ad.unpack_dual(value).primal
            for value in inputs[:35]
        )
        bounds = tuple(
            torch.autograd.forward_ad.unpack_dual(value).primal
            for value in (inputs[37], inputs[38], inputs[39], inputs[40])
        )
        ctx.frequency_value = inputs[36]
        ctx.frequency_meta = (
            (frequency.dtype, frequency.device)
            if isinstance(frequency, torch.Tensor)
            else None
        )
        ctx.save_for_backward(*primals, *bounds)
        ctx.save_for_forward(*primals, *bounds)
        ctx.mark_non_differentiable(output[4])

    @staticmethod
    @_ad_first_order_only
    def backward(
        ctx,
        grad_field_vector,
        grad_coefficient,
        grad_path_field,
        grad_path_gain,
        _grad_direction,
    ):
        none_grads = (None,) * 41
        _ad_reject_fixed_inputs(
            "field_coupled_dd_ad",
            ctx.needs_input_grad,
            (
                (2, "edge1_position"),
                (3, "edge1_direction"),
                (4, "edge1_n0"),
                (5, "edge1_n1"),
                (6, "edge1_exterior"),
                (7, "edge2_position"),
                (8, "edge2_direction"),
                (9, "edge2_n0"),
                (10, "edge2_n1"),
                (11, "edge2_exterior"),
                (12, "tx_power"),
                (13, "tx_polarization"),
                (14, "rx_polarization"),
                (17, "wedge1_mu_r0"),
                (22, "wedge1_mu_r1"),
                (27, "wedge2_mu_r0"),
                (32, "wedge2_mu_r1"),
                (37, "edge1_line_min"),
                (38, "edge1_line_max"),
                (39, "edge2_line_min"),
                (40, "edge2_line_max"),
            ),
        )
        needs = _dd_backward_needs(ctx.needs_input_grad)
        grads = (grad_field_vector, grad_coefficient, grad_path_field, grad_path_gain)
        if _dd_backward_is_noop(needs, grads):
            return none_grads
        saved = ctx.saved_tensors
        out = _required_native_op("field_coupled_dd_backward")(
            *saved,
            ctx.frequency_value,
            grad_field_vector,
            grad_coefficient,
            grad_path_field,
            grad_path_gain,
            needs["eps"],
            needs["sigma"],
            needs["gain"],
            needs["thickness"],
            needs["frequency"],
            needs["geometry"],
        )
        grad_frequency = (
            _ad_frequency_grad(out["grad_frequency"], ctx.frequency_meta)
            if needs["frequency"]
            else None
        )

        def material_column(name: str, column: int, index: int):
            if not ctx.needs_input_grad[index]:
                return None
            return out[name][:, column]

        # Material grad columns: wedge1 face0, wedge1 face1, wedge2 face0,
        # wedge2 face1 (mirrors the native (N, 4) slot layout).
        return (
            out["grad_source"] if ctx.needs_input_grad[0] else None,
            out["grad_target"] if ctx.needs_input_grad[1] else None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            material_column("grad_eps_r", 0, 15),
            material_column("grad_sigma_e", 0, 16),
            None,
            material_column("grad_gain", 0, 18),
            material_column("grad_thickness", 0, 19),
            material_column("grad_eps_r", 1, 20),
            material_column("grad_sigma_e", 1, 21),
            None,
            material_column("grad_gain", 1, 23),
            material_column("grad_thickness", 1, 24),
            material_column("grad_eps_r", 2, 25),
            material_column("grad_sigma_e", 2, 26),
            None,
            material_column("grad_gain", 2, 28),
            material_column("grad_thickness", 2, 29),
            material_column("grad_eps_r", 3, 30),
            material_column("grad_sigma_e", 3, 31),
            None,
            material_column("grad_gain", 3, 33),
            material_column("grad_thickness", 3, 34),
            grad_frequency,
            None,
            None,
            None,
            None,
            None,
        )

    @staticmethod
    def jvp(ctx, *tangents):
        _ad_reject_fixed_tangents(
            "field_coupled_dd_ad",
            (
                (tangents[2], "edge1_position"),
                (tangents[3], "edge1_direction"),
                (tangents[4], "edge1_n0"),
                (tangents[5], "edge1_n1"),
                (tangents[6], "edge1_exterior"),
                (tangents[7], "edge2_position"),
                (tangents[8], "edge2_direction"),
                (tangents[9], "edge2_n0"),
                (tangents[10], "edge2_n1"),
                (tangents[11], "edge2_exterior"),
                (tangents[12], "tx_power"),
                (tangents[13], "tx_polarization"),
                (tangents[14], "rx_polarization"),
                (tangents[17], "wedge1_mu_r0"),
                (tangents[22], "wedge1_mu_r1"),
                (tangents[27], "wedge2_mu_r0"),
                (tangents[32], "wedge2_mu_r1"),
                (tangents[37], "edge1_line_min"),
                (tangents[38], "edge1_line_max"),
                (tangents[39], "edge2_line_min"),
                (tangents[40], "edge2_line_max"),
            ),
        )
        saved = ctx.saved_tensors
        scalar_shape = tuple(saved[15].shape)
        tangent_source = _ad_geometry_tangent(
            "field_coupled_dd_ad tangent_source", tangents[0], saved[0]
        )
        tangent_target = _ad_geometry_tangent(
            "field_coupled_dd_ad tangent_target", tangents[1], saved[1]
        )

        def material_pack(indices: tuple[int, int, int, int], name: str):
            columns = tuple(
                _ad_checked_tangent(
                    f"field_coupled_dd_ad tangent_{name}",
                    _ad_native_tangent_or_none(tangents[index]),
                    scalar_shape,
                )
                for index in indices
            )
            if all(column is None for column in columns):
                return None
            zero = torch.zeros(
                scalar_shape, device=saved[15].device, dtype=torch.float32
            )
            return torch.stack(
                [zero if column is None else column for column in columns], dim=1
            )

        tangent_eps = material_pack((15, 20, 25, 30), "eps_r")
        tangent_sigma = material_pack((16, 21, 26, 31), "sigma_e")
        tangent_gain = material_pack((18, 23, 28, 33), "gain")
        tangent_thickness = material_pack((19, 24, 29, 34), "thickness")
        tangent_frequency = _ad_frequency_tangent(tangents[35])
        if (
            tangent_source is None
            and tangent_target is None
            and tangent_eps is None
            and tangent_sigma is None
            and tangent_gain is None
            and tangent_thickness is None
            and tangent_frequency == 0.0
        ):
            return (None,) * 5
        with disable_functorch():
            out = _required_native_op("field_coupled_dd_jvp")(
                *(_ad_native_tensor(value) for value in saved),
                ctx.frequency_value,
                tangent_source,
                tangent_target,
                tangent_eps,
                tangent_sigma,
                tangent_gain,
                tangent_thickness,
                float(tangent_frequency),
            )
        return (
            out["tangent_field_vector"],
            out["tangent_coefficient"],
            out["tangent_path_field"],
            out["tangent_path_gain"],
            None,
        )


def field_coupled_dd_ad(
    source: torch.Tensor,
    target: torch.Tensor,
    edge1_position: torch.Tensor,
    edge1_direction: torch.Tensor,
    edge1_n0: torch.Tensor,
    edge1_n1: torch.Tensor,
    edge1_exterior: torch.Tensor,
    edge2_position: torch.Tensor,
    edge2_direction: torch.Tensor,
    edge2_n0: torch.Tensor,
    edge2_n1: torch.Tensor,
    edge2_exterior: torch.Tensor,
    tx_power: torch.Tensor,
    tx_polarization: torch.Tensor,
    rx_polarization: torch.Tensor,
    wedge1_material0: tuple[torch.Tensor, ...],
    wedge1_material1: tuple[torch.Tensor, ...],
    wedge2_material0: tuple[torch.Tensor, ...],
    wedge2_material1: tuple[torch.Tensor, ...],
    edge1_line_min: torch.Tensor,
    edge1_line_max: torch.Tensor,
    edge2_line_min: torch.Tensor,
    edge2_line_max: torch.Tensor,
    *,
    frequency: torch.Tensor | float,
    frequency_value: float | None = None,
) -> dict[str, torch.Tensor]:
    """Differentiable :func:`field_coupled_dd` (16 material scalars + frequency + tx/rx).

    ``frequency_value`` optionally carries the precomputed host scalar of
    ``frequency`` (one read per solve at the seam); when not supplied it is read
    here, exactly once per apply.
    """

    if any(
        len(bundle) != 5
        for bundle in (
            wedge1_material0,
            wedge1_material1,
            wedge2_material0,
            wedge2_material1,
        )
    ):
        raise ValueError(
            "coupled dd material bundles must contain eps/sigma/mu/gain/thickness"
        )
    if frequency_value is None:
        frequency_value = _ad_frequency_value(frequency)
    values = _FieldCoupledDdAdFunction.apply(
        source,
        target,
        edge1_position,
        edge1_direction,
        edge1_n0,
        edge1_n1,
        edge1_exterior,
        edge2_position,
        edge2_direction,
        edge2_n0,
        edge2_n1,
        edge2_exterior,
        tx_power,
        tx_polarization,
        rx_polarization,
        *wedge1_material0,
        *wedge1_material1,
        *wedge2_material0,
        *wedge2_material1,
        frequency,
        float(frequency_value),
        edge1_line_min,
        edge1_line_max,
        edge2_line_min,
        edge2_line_max,
    )
    return dict(zip(_COUPLED_OUTPUT_FIELDS, values, strict=True))


class _FieldProjectComplex3AdFunction(torch.autograd.Function):
    """Differentiable receiver projection on a frozen polarization basis."""

    @staticmethod
    def forward(field_vector, direction, rx_polarization):
        out = _required_native_op("field_project_complex3")(
            field_vector, direction, rx_polarization
        )
        return (out["coefficient"], out["path_gain"])

    @staticmethod
    def setup_context(ctx, inputs, output):
        ctx.set_materialize_grads(False)
        primals = tuple(
            torch.autograd.forward_ad.unpack_dual(value).primal for value in inputs
        )
        ctx.save_for_backward(*primals)
        ctx.save_for_forward(*primals)

    @staticmethod
    @_ad_first_order_only
    def backward(ctx, grad_coefficient, grad_path_gain):
        _ad_reject_fixed_inputs(
            "field_project_complex3_ad",
            ctx.needs_input_grad,
            ((2, "rx_polarization"),),
        )
        need_field = bool(ctx.needs_input_grad[0])
        need_direction = bool(ctx.needs_input_grad[1])
        if not (need_field or need_direction) or (
            grad_coefficient is None and grad_path_gain is None
        ):
            return (None, None, None)
        field_vector, direction, rx_polarization = ctx.saved_tensors
        out = _required_native_op("field_project_complex3_backward")(
            field_vector,
            direction,
            rx_polarization,
            grad_coefficient,
            grad_path_gain,
            need_field,
            need_direction,
        )
        return (
            out["grad_field_vector"] if need_field else None,
            out["grad_direction"] if need_direction else None,
            None,
        )

    @staticmethod
    def jvp(ctx, t_field_vector, t_direction, t_rx_polarization):
        _ad_reject_fixed_tangents(
            "field_project_complex3_ad",
            ((t_rx_polarization, "rx_polarization"),),
        )
        saved = ctx.saved_tensors
        tangent_field = _ad_native_tangent_or_none(t_field_vector)
        tangent_direction = _ad_geometry_tangent(
            "field_project_complex3_ad tangent_direction", t_direction, saved[1]
        )
        if tangent_field is None and tangent_direction is None:
            return (None, None)
        with disable_functorch():
            out = _required_native_op("field_project_complex3_jvp")(
                *(_ad_native_tensor(value) for value in saved),
                tangent_field,
                tangent_direction,
            )
        return (out["tangent_coefficient"], out["tangent_path_gain"])


def field_project_complex3_ad(
    field_vector: torch.Tensor,
    direction: torch.Tensor,
    rx_polarization: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Differentiable :func:`field_project_complex3` (field vector + direction)."""

    coefficient, path_gain = _FieldProjectComplex3AdFunction.apply(
        field_vector, direction, rx_polarization
    )
    return {"coefficient": coefficient, "path_gain": path_gain}


def _grad_or_none(out: dict, key: str, needed: bool) -> torch.Tensor | None:
    return out[key] if needed else None


class _FieldRoughReflectionScaleAdFunction(torch.autograd.Function):
    """Fixed-topology differentiable rough-surface C_r scale (ADR-010 op 3).

    Differentiable inputs: the four reflection field outputs (field_vector,
    coefficient, path_field, path_gain), frequency, and the hit geometry
    (positions, normals, source). sigma_b, rough_b and the realization
    ``replaced`` mask are fixed; requesting a sigma_b gradient fails loudly.
    Positions/normals/source only carry a gradient when the fixed-winner
    geometry AD path is live (matching the previous Torch factor's reach).
    """

    @staticmethod
    def forward(
        field_vector,
        coefficient,
        path_field,
        path_gain,
        positions,
        normals,
        source,
        sigma_b,
        rough_b,
        replaced,
        frequency,
        frequency_value,
        geometry_live,
    ):
        out = field_rough_reflection_scale(
            field_vector,
            coefficient,
            path_field,
            path_gain,
            positions,
            normals,
            source,
            sigma_b,
            rough_b,
            replaced,
            frequency_hz=frequency_value,
        )
        return tuple(out[name] for name in _ROUGH_SCALE_OUTPUT_FIELDS)

    @staticmethod
    def setup_context(ctx, inputs, output):
        ctx.set_materialize_grads(False)
        frequency = inputs[10]
        primals = tuple(
            torch.autograd.forward_ad.unpack_dual(value).primal
            for value in inputs[:10]
        )
        ctx.frequency_value = inputs[11]
        ctx.frequency_meta = (
            (frequency.dtype, frequency.device)
            if isinstance(frequency, torch.Tensor)
            else None
        )
        # Computed by the wrapper, where forward duals are still visible.
        ctx.geometry_live = inputs[12]
        ctx.save_for_backward(*primals)
        ctx.save_for_forward(*primals)

    @staticmethod
    @_ad_first_order_only
    def backward(
        ctx,
        grad_field_vector,
        grad_coefficient,
        grad_path_field,
        grad_path_gain,
    ):
        none_grads = (None,) * 13
        _ad_reject_fixed_inputs(
            "field_rough_reflection_scale_ad",
            ctx.needs_input_grad,
            ((7, "sigma_b"),),
        )
        needed = tuple(bool(ctx.needs_input_grad[i]) for i in range(11))
        need_field = any(needed[:4])
        need_geometry = any(needed[4:7])
        need_frequency = needed[10]
        grads = (
            grad_field_vector,
            grad_coefficient,
            grad_path_field,
            grad_path_gain,
        )
        if not (need_field or need_geometry or need_frequency) or all(
            value is None for value in grads
        ):
            return none_grads
        out = field_rough_reflection_scale_backward(
            *ctx.saved_tensors,
            frequency_hz=ctx.frequency_value,
            grad_field_vector=grad_field_vector,
            grad_coefficient=grad_coefficient,
            grad_path_field=grad_path_field,
            grad_path_gain=grad_path_gain,
            need_field=need_field,
            need_geometry=need_geometry,
            need_frequency=need_frequency,
        )
        grad_frequency = (
            _ad_frequency_grad(out["grad_frequency"], ctx.frequency_meta)
            if need_frequency
            else None
        )
        return (
            _grad_or_none(out, "grad_field_vector", needed[0]),
            _grad_or_none(out, "grad_coefficient", needed[1]),
            _grad_or_none(out, "grad_path_field", needed[2]),
            _grad_or_none(out, "grad_path_gain", needed[3]),
            _grad_or_none(out, "grad_positions", needed[4]),
            _grad_or_none(out, "grad_normals", needed[5]),
            _grad_or_none(out, "grad_source", needed[6]),
            None,
            None,
            None,
            grad_frequency,
            None,
            None,
        )

    @staticmethod
    def jvp(
        ctx,
        t_field_vector,
        t_coefficient,
        t_path_field,
        t_path_gain,
        t_positions,
        t_normals,
        t_source,
        t_sigma_b,
        _t_rough_b,
        _t_replaced,
        t_frequency,
        _t_frequency_value,
        _t_geometry_live,
    ):
        _ad_reject_fixed_tangents(
            "field_rough_reflection_scale_ad",
            ((t_sigma_b, "sigma_b"),),
        )
        saved = ctx.saved_tensors
        tangent_field = _ad_native_tangent_or_none(t_field_vector)
        tangent_coef = _ad_native_tangent_or_none(t_coefficient)
        tangent_pf = _ad_native_tangent_or_none(t_path_field)
        tangent_pg = _ad_native_tangent_or_none(t_path_gain)
        tangent_positions = _ad_geometry_tangent(
            "field_rough_reflection_scale_ad tangent_positions",
            t_positions,
            saved[4],
        )
        tangent_normals = _ad_geometry_tangent(
            "field_rough_reflection_scale_ad tangent_normals", t_normals, saved[5]
        )
        tangent_source = _ad_geometry_tangent(
            "field_rough_reflection_scale_ad tangent_source", t_source, saved[6]
        )
        tangent_frequency = _ad_frequency_tangent(t_frequency)
        if (
            tangent_field is None
            and tangent_coef is None
            and tangent_pf is None
            and tangent_pg is None
            and tangent_positions is None
            and tangent_normals is None
            and tangent_source is None
            and tangent_frequency == 0.0
        ):
            return (None,) * len(_ROUGH_SCALE_OUTPUT_FIELDS)
        with disable_functorch():
            out = field_rough_reflection_scale_jvp(
                *(_ad_native_tensor(value) for value in saved),
                frequency_hz=ctx.frequency_value,
                tangent_field_vector=tangent_field,
                tangent_coefficient=tangent_coef,
                tangent_path_field=tangent_pf,
                tangent_path_gain=tangent_pg,
                tangent_positions=tangent_positions,
                tangent_normals=tangent_normals,
                tangent_source=tangent_source,
                tangent_frequency=tangent_frequency,
            )
        return tuple(out[name] for name in _ROUGH_SCALE_TANGENT_FIELDS)


def field_rough_reflection_scale_ad(
    field_vector: torch.Tensor,
    coefficient: torch.Tensor,
    path_field: torch.Tensor,
    path_gain: torch.Tensor,
    positions: torch.Tensor,
    normals: torch.Tensor,
    source: torch.Tensor,
    sigma_b: torch.Tensor,
    rough_b: torch.Tensor,
    replaced: torch.Tensor,
    *,
    frequency: torch.Tensor | float,
    frequency_value: float | None = None,
) -> dict[str, torch.Tensor]:
    """Differentiable :func:`field_rough_reflection_scale` (frequency + geometry).

    ``frequency_value`` optionally carries the precomputed host scalar of
    ``frequency`` (one read per solve at the seam, audit M3); when not supplied
    it is read here, exactly once per apply.
    """

    if frequency_value is None:
        frequency_value = _ad_frequency_value(frequency)
    values = _FieldRoughReflectionScaleAdFunction.apply(
        field_vector,
        coefficient,
        path_field,
        path_gain,
        positions,
        normals,
        source,
        sigma_b,
        rough_b,
        replaced,
        frequency,
        float(frequency_value),
        _ad_geometry_live(positions, normals, source),
    )
    return dict(zip(_ROUGH_SCALE_OUTPUT_FIELDS, values, strict=True))


def _validate_source_amplitude_inputs(
    field_vector: torch.Tensor, tx_power: torch.Tensor, *, field_name: str
) -> int:
    # Cotangents and tangents arrive as strided views; the native owner
    # canonicalizes them, so layout is not part of this contract.
    validate_cuda_tensor(
        field_name,
        field_vector,
        dtype=torch.complex64,
        ndim=2,
        trailing_shape=(3,),
        require_contiguous=False,
    )
    validate_cuda_tensor(
        "tx_power",
        tx_power,
        dtype=torch.float32,
        ndim=1,
        require_contiguous=False,
    )
    count = int(field_vector.shape[0])
    if tx_power.shape != (count,):
        raise ValueError("tx_power must have one row per complex3 field row")
    return count


def _validate_source_amplitude_result(
    out: object, name: str, count: int, *, operation: str
) -> dict[str, torch.Tensor]:
    if not isinstance(out, dict) or set(out) != {name}:
        raise TypeError(f"_channel.{operation} returned invalid fields")
    validate_cuda_tensor(name, out[name], dtype=torch.complex64, ndim=2)
    if tuple(out[name].shape) != (count, 3):
        raise ValueError(f"{operation} returned an invalid shape")
    return out


def field_source_amplitude_scale(
    field_vector: torch.Tensor, tx_power: torch.Tensor
) -> dict[str, torch.Tensor]:
    """Apply ``sqrt(max(tx_power, 0))`` to a transported complex3 field.

    The transport kernels publish ``field_vector`` for unit source amplitude
    and ``path_field = coefficient * sqrt(tx_power)`` for the excited scalar,
    but no excited vector. This is that missing output, evaluated natively with
    the same amplitude expression, so its receiver projection reproduces
    ``path_field``.
    """

    count = _validate_source_amplitude_inputs(
        field_vector, tx_power, field_name="field_vector"
    )
    out = _required_native_op("field_source_amplitude_scale")(field_vector, tx_power)
    return _validate_source_amplitude_result(
        out, "path_field_vector", count, operation="field_source_amplitude_scale"
    )


def field_source_amplitude_scale_backward(
    tx_power: torch.Tensor, grad_path_field_vector: torch.Tensor
) -> dict[str, torch.Tensor]:
    """VJP of :func:`field_source_amplitude_scale`; ``tx_power`` is frozen."""

    count = _validate_source_amplitude_inputs(
        grad_path_field_vector, tx_power, field_name="grad_path_field_vector"
    )
    out = _required_native_op("field_source_amplitude_scale_backward")(
        tx_power, grad_path_field_vector
    )
    return _validate_source_amplitude_result(
        out,
        "grad_field_vector",
        count,
        operation="field_source_amplitude_scale_backward",
    )


def field_source_amplitude_scale_jvp(
    tx_power: torch.Tensor, tangent_field_vector: torch.Tensor
) -> dict[str, torch.Tensor]:
    """JVP of :func:`field_source_amplitude_scale`; ``tx_power`` is frozen."""

    count = _validate_source_amplitude_inputs(
        tangent_field_vector, tx_power, field_name="tangent_field_vector"
    )
    out = _required_native_op("field_source_amplitude_scale_jvp")(
        tx_power, tangent_field_vector
    )
    return _validate_source_amplitude_result(
        out,
        "tangent_path_field_vector",
        count,
        operation="field_source_amplitude_scale_jvp",
    )


class _FieldSourceAmplitudeScaleAdFunction(torch.autograd.Function):
    """Differentiable ``field_vector * sqrt(max(tx_power, 0))``."""

    @staticmethod
    def forward(field_vector, tx_power):
        out = _required_native_op("field_source_amplitude_scale")(
            field_vector, tx_power
        )
        return out["path_field_vector"]

    @staticmethod
    def setup_context(ctx, inputs, output):
        ctx.set_materialize_grads(False)
        tx_power = torch.autograd.forward_ad.unpack_dual(inputs[1]).primal
        ctx.save_for_backward(tx_power)
        ctx.save_for_forward(tx_power)

    @staticmethod
    @_ad_first_order_only
    def backward(ctx, grad_path_field_vector):
        _ad_reject_fixed_inputs(
            "field_source_amplitude_scale_ad",
            ctx.needs_input_grad,
            ((1, "tx_power"),),
        )
        if not ctx.needs_input_grad[0] or grad_path_field_vector is None:
            return (None, None)
        (tx_power,) = ctx.saved_tensors
        out = _required_native_op("field_source_amplitude_scale_backward")(
            tx_power, grad_path_field_vector
        )
        return (out["grad_field_vector"], None)

    @staticmethod
    def jvp(ctx, t_field_vector, t_tx_power):
        _ad_reject_fixed_tangents(
            "field_source_amplitude_scale_ad", ((t_tx_power, "tx_power"),)
        )
        tangent = _ad_native_tangent_or_none(t_field_vector)
        if tangent is None:
            return None
        (tx_power,) = ctx.saved_tensors
        with disable_functorch():
            out = _required_native_op("field_source_amplitude_scale_jvp")(
                _ad_native_tensor(tx_power), tangent
            )
        return out["tangent_path_field_vector"]


def field_source_amplitude_scale_ad(
    field_vector: torch.Tensor, tx_power: torch.Tensor
) -> torch.Tensor:
    """Differentiable :func:`field_source_amplitude_scale` (field vector only)."""

    return _FieldSourceAmplitudeScaleAdFunction.apply(field_vector, tx_power)


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

    exported = _required_native_op("deterministic_los_field")(
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

    exported = _required_native_op("deterministic_diffraction_vector_field")(
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

    exported = _required_native_op("deterministic_reflection_field")(
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

    exported = _required_native_op("deterministic_reflection_sequence_field")(
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
    path_length = _required_native_op("deterministic_delay_to_path_length")(delay_s)
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
    field = _required_native_op("deterministic_pack_complex")(field_real, field_imag)
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
    phase = _required_native_op("deterministic_phase_from_field")(field_real, field_imag)
    validate_cuda_tensor("phase_rad", phase, dtype=torch.float32, ndim=1)
    if phase.shape != field_real.shape:
        raise ValueError(
            "_channel.deterministic_phase_from_field returned bad shape"
        )
    return phase


def deterministic_zero_field_phase(reference: torch.Tensor) -> dict[str, torch.Tensor]:
    validate_cuda_tensor("reference", reference, dtype=torch.float32, ndim=1)
    exported = _required_native_op("deterministic_zero_field_phase")(reference)
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
    phase = _required_native_op("deterministic_phase_from_length")(path_length_m, float(frequency_hz))
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
    exported = _required_native_op("deterministic_field_from_power_phase")(path_gain, phase_rad)
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