# Copyright Xingyu Chen.
# Native RF field kernel facades.

"""Native RF field kernel facades."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from witwin.channel.kernels.materials import validate_layer_csr as _validate_layer_csr
from witwin.channel.kernels.spec import (
    AdInputs,
    Arg,
    Grad,
    NativeOps,
    flag,
    frequency_meta,
    real,
    save,
    tensor,
    opaque,
)
from witwin.channel.runtime import (
    _ad_first_order_only,
    _ad_frequency_grad,
    _ad_frequency_tangent,
    _ad_frequency_value,
    _ad_geometry_live,
    _ad_native_tangent_or_none,
    _ad_native_tensor,
    disable_functorch,
    required_symbol as _required_native_op,
    validate_cuda_tensor,
)

__all__ = [
    "ReflectionLayers",
    "reflection_layers",
    "_FieldCoupledChainAdFunction",
    "_FieldCoupledDdAdFunction",
    "_FieldDiffractionWedgeAdFunction",
    "_FieldFreeSpaceAdFunction",
    "_FieldProjectComplex3AdFunction",
    "_FieldReflectionSequenceAdFunction",
    "_FieldRoughReflectionScaleAdFunction",
    "_FieldSourceAmplitudeScaleAdFunction",
    "_FieldTransmissionSequenceAdFunction",
    "ad_liveness",
    "deterministic_pack_complex",
    "deterministic_phase_from_field",
    "deterministic_zero_field_phase",
    "direction_cotangent",
    "direction_tangents",
    "field_coupled_chain",
    "field_coupled_chain_ad",
    "field_coupled_dd",
    "field_coupled_dd_ad",
    "field_diffraction_wedge",
    "field_diffraction_wedge_ad",
    "CanonicalFaceTables",
    "field_canonical_rows",
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
    "field_transport_columns",
    "field_transmission_sequence_ad",
    "field_transmission_sequence_backward",
    "field_transmission_sequence_jvp",
    "mark_dead_outputs",
]

_native = NativeOps(lambda symbol: _required_native_op(symbol))
_F32 = torch.float32
_C64 = torch.complex64


def _rows3(name: str, *, grad: Grad = "none") -> Arg:
    """An ``(N, 3)`` float32 row tensor."""

    return tensor(name, _F32, 2, (3,), grad=grad)


def _slots3(name: str, *, grad: Grad = "none") -> Arg:
    """An ``(N, D, 3)`` float32 per-interaction tensor."""

    return tensor(name, _F32, 3, (3,), grad=grad)


def _values(*names: str) -> tuple[Arg, ...]:
    return tuple(opaque(name) for name in names)


def _flags(*names: str) -> tuple[Arg, ...]:
    return tuple(flag(name) for name in names)


_FREQUENCY = real("frequency_hz")
_FIELD_GRADS = _values(
    "grad_field_vector", "grad_coefficient", "grad_path_field", "grad_path_gain",
    "grad_path_length", "grad_delay",
)
_COUPLED_GRADS = _FIELD_GRADS[:4]
# The per-face layer request of a reflection launch (``ReflectionLayers.native`` order).
_LAYER_REQUEST = tuple(
    tensor(name)
    for name in (
        "bounce_material", "layer_offset", "layer_count", "layer_thickness_m", "layer_eps_r",
        "layer_sigma_e", "layer_mu_r",
    )
)

_FREE_SPACE_PRIMALS = (
    _rows3("source", grad="diff"),
    _rows3("target", grad="diff"),
    tensor("tx_power", _F32, 1, grad="fixed"),
    _rows3("tx_polarization", grad="fixed"),
    _rows3("rx_polarization", grad="fixed"),
)
_FREE_SPACE_CHECKS = ("source", "target", "tx_polarization", "rx_polarization", "tx_power")
_FIELD_FREE_SPACE = _native.op("field_free_space", _FREE_SPACE_PRIMALS, _FREQUENCY)
_FIELD_FREE_SPACE_FWD64 = _native.op("field_free_space_fwd64", _FREE_SPACE_PRIMALS, _FREQUENCY)
_FIELD_FREE_SPACE_COLUMNS = _native.op(
    "field_free_space_columns", _FREE_SPACE_PRIMALS, opaque("frequencies_hz")
)
_FIELD_FREE_SPACE_BACKWARD = _native.op(
    "field_free_space_backward", _FREE_SPACE_PRIMALS, _FREQUENCY, _FIELD_GRADS,
    opaque("grad_direction"), _flags("need_grad_frequency", "need_grad_geometry"),
)
_FIELD_FREE_SPACE_JVP = _native.op(
    "field_free_space_jvp", _FREE_SPACE_PRIMALS, _FREQUENCY, real("tangent_frequency"),
    _values("tangent_source", "tangent_target"),
)

_REFLECTION_PRIMALS = (
    _rows3("source", grad="diff"),
    _rows3("target", grad="diff"),
    _slots3("interaction_positions", grad="diff"),
    _slots3("interaction_normals", grad="diff"),
    tensor("tx_power", _F32, 1, grad="fixed"),
    _rows3("tx_polarization", grad="fixed"),
    _rows3("rx_polarization", grad="fixed"),
    tensor("eps_r", _F32, 2, grad="diff"),
    tensor("sigma_e", _F32, 2, grad="diff"),
    tensor("mu_r", _F32, 2, grad="fixed"),
    tensor("gain", _F32, 2, grad="diff"),
    tensor("thickness", _F32, 2, grad="diff"),
)
_REFLECTION_MATERIALS = ("eps_r", "sigma_e", "mu_r", "gain", "thickness")
_FIELD_REFLECTION_SEQUENCE = _native.op(
    "field_reflection_sequence", _REFLECTION_PRIMALS, _LAYER_REQUEST, _FREQUENCY
)
_FIELD_REFLECTION_SEQUENCE_COLUMNS = _native.op(
    "field_reflection_sequence_columns", _REFLECTION_PRIMALS, _LAYER_REQUEST,
    opaque("frequencies_hz"),
)
_FIELD_REFLECTION_SEQUENCE_BACKWARD = _native.op(
    "field_reflection_sequence_backward", _REFLECTION_PRIMALS, _LAYER_REQUEST, _FREQUENCY,
    _FIELD_GRADS, opaque("grad_direction"),
    _flags(
        "need_grad_eps_r", "need_grad_sigma_e", "need_grad_gain", "need_grad_thickness",
        "need_grad_frequency", "need_grad_geometry",
    ),
)
_FIELD_REFLECTION_SEQUENCE_JVP = _native.op(
    "field_reflection_sequence_jvp", _REFLECTION_PRIMALS, _LAYER_REQUEST, _FREQUENCY,
    _values("tangent_eps_r", "tangent_sigma_e", "tangent_gain", "tangent_thickness"),
    real("tangent_frequency"),
    _values(
        "tangent_source", "tangent_target", "tangent_interaction_positions",
        "tangent_interaction_normals",
    ),
)

# ``interaction_positions`` never enters the straight transmission field; its gradient is None.
_TRANSMISSION_PRIMALS = (
    tensor("path_valid", torch.bool, 1),
    _rows3("source", grad="diff"),
    _rows3("target", grad="diff"),
    _slots3("interaction_positions", grad="diff"),
    _slots3("interaction_normals", grad="diff"),
    tensor("interaction_material_id", torch.int32, 2),
    tensor("interaction_valid", torch.bool, 2),
    tensor("tx_power", _F32, 1, grad="fixed"),
    _rows3("tx_polarization", grad="fixed"),
    _rows3("rx_polarization", grad="fixed"),
    tensor("layer_offset"),
    tensor("layer_count"),
    tensor("layer_thickness_m", grad="diff"),
    tensor("layer_eps_r", grad="diff"),
    tensor("layer_sigma_e", grad="diff"),
    tensor("layer_mu_r", grad="fixed"),
)
_TRANSMISSION_CHECKS = (
    "path_valid", "source", "target", "tx_polarization", "rx_polarization",
    "interaction_positions", "interaction_normals", "interaction_material_id",
    "interaction_valid", "tx_power",
)
_FIELD_TRANSMISSION_SEQUENCE = _native.op(
    "field_transmission_sequence", _TRANSMISSION_PRIMALS, _FREQUENCY
)
_FIELD_TRANSMISSION_SEQUENCE_BACKWARD = _native.op(
    "field_transmission_sequence_backward", _TRANSMISSION_PRIMALS, _FREQUENCY, _FIELD_GRADS,
    _flags(
        "need_grad_layer_thickness", "need_grad_layer_eps_r", "need_grad_layer_sigma_e",
        "need_grad_frequency", "need_grad_geometry",
    ),
)
_FIELD_TRANSMISSION_SEQUENCE_JVP = _native.op(
    "field_transmission_sequence_jvp", _TRANSMISSION_PRIMALS, _FREQUENCY,
    _values("tangent_layer_thickness_m", "tangent_layer_eps_r", "tangent_layer_sigma_e"),
    real("tangent_frequency"),
    _values(
        "tangent_source", "tangent_target", "tangent_interaction_positions",
        "tangent_interaction_normals",
    ),
)

_FIELD_CANONICAL_ROWS = _native.op(
    "field_canonical_rows",
    *(
        tensor(name)
        for name in (
            "order", "component_id", "depth", "tx_id", "rx_id", "primitive_sequence",
            "tx_positions", "tx_power", "tx_polarization", "rx_positions", "rx_polarization",
            "interaction_positions", "interaction_normals", "field_xyz", "coefficient",
            "path_field", "path_gain", "path_length", "delay", "direction", "face_eps_r",
            "face_sigma_e", "face_mu_r", "face_gain", "face_thickness", "face_material",
            "layer_offset", "layer_count", "layer_thickness_m", "layer_eps_r", "layer_sigma_e",
            "layer_mu_r",
        )
    ),
    flag("multilayer"),
    _values("face_rough_sigma", "face_rough", "face_replaced"),
    _flags("rough", "replace_specular", "evaluate_los"),
    _FREQUENCY,
)

# Interaction-chain inputs: reflections, one diffraction, reflections, then the slot materials.
_CHAIN_PRIMALS = (
    opaque("source", grad="diff"),
    opaque("target", grad="diff"),
    opaque("positions", grad="fixed"),
    opaque("normals", grad="fixed"),
    opaque("component_id"),
    opaque("depth"),
    *(
        opaque(name, grad="fixed")
        for name in (
            "edge_direction", "edge_n0", "edge_n1", "exterior_angle", "edge_line_min",
            "edge_line_max", "tx_power", "tx_polarization", "rx_polarization",
        )
    ),
    opaque("eps_r", grad="diff"),
    opaque("sigma_e", grad="diff"),
    opaque("mu_r", grad="fixed"),
    opaque("gain", grad="diff"),
    opaque("thickness", grad="diff"),
)
_CHAIN_MATERIAL_GRADS = ("eps_r", "sigma_e", "gain", "thickness")
_FIELD_COUPLED_CHAIN = _native.op(
    "field_coupled_chain", _CHAIN_PRIMALS, _LAYER_REQUEST, _FREQUENCY
)
_FIELD_COUPLED_CHAIN_BACKWARD = _native.op(
    "field_coupled_chain_backward", _CHAIN_PRIMALS, _LAYER_REQUEST, _FREQUENCY, _COUPLED_GRADS,
    _flags(
        "need_grad_eps_r", "need_grad_sigma_e", "need_grad_gain", "need_grad_thickness",
        "need_grad_frequency", "need_grad_geometry",
    ),
)
_FIELD_COUPLED_CHAIN_JVP = _native.op(
    "field_coupled_chain_jvp", _CHAIN_PRIMALS, _LAYER_REQUEST, _FREQUENCY,
    _values(
        "tangent_source", "tangent_target", "tangent_eps_r", "tangent_sigma_e", "tangent_gain",
        "tangent_thickness",
    ),
    real("tangent_frequency"),
)

# The four coupled double-diffraction wedge faces, in the native (N, 4) column order.
_DD_FACES = ("wedge1_{}0", "wedge1_{}1", "wedge2_{}0", "wedge2_{}1")
_DD_PRIMALS = (
    opaque("source", grad="diff"),
    opaque("target", grad="diff"),
    *(
        opaque(name, grad="fixed")
        for name in (
            "edge1_position", "edge1_direction", "edge1_n0", "edge1_n1", "edge1_exterior",
            "edge2_position", "edge2_direction", "edge2_n0", "edge2_n1", "edge2_exterior",
            "tx_power", "tx_polarization", "rx_polarization",
        )
    ),
    *(
        opaque(face.format(field), grad="fixed" if field == "mu_r" else "diff")
        for face in _DD_FACES
        for field in ("eps_r", "sigma_e", "mu_r", "gain", "thickness")
    ),
)
_DD_BOUNDS = tuple(
    opaque(name, grad="fixed")
    for name in ("edge1_line_min", "edge1_line_max", "edge2_line_min", "edge2_line_max")
)
_FIELD_COUPLED_DD = _native.op("field_coupled_dd", _DD_PRIMALS, _DD_BOUNDS, _FREQUENCY)
_FIELD_COUPLED_DD_BACKWARD = _native.op(
    "field_coupled_dd_backward", _DD_PRIMALS, _DD_BOUNDS, _FREQUENCY, _COUPLED_GRADS,
    _flags(
        "need_grad_eps_r", "need_grad_sigma_e", "need_grad_gain", "need_grad_thickness",
        "need_grad_frequency", "need_grad_geometry",
    ),
)
_FIELD_COUPLED_DD_JVP = _native.op(
    "field_coupled_dd_jvp", _DD_PRIMALS, _DD_BOUNDS, _FREQUENCY,
    _values(
        "tangent_source", "tangent_target", "tangent_eps_r", "tangent_sigma_e", "tangent_gain",
        "tangent_thickness",
    ),
    real("tangent_frequency"),
)

# Order-1 UTD wedge inputs. The face valid masks and the edge boundary refuse a gradient; a
# tangent on them is ignored.
_WEDGE_PRIMALS = (
    opaque("valid", grad="fixed"),
    opaque("source", grad="diff"),
    opaque("target", grad="diff"),
    *(
        opaque(name, grad="fixed")
        for name in (
            "edge_position", "edge_direction", "edge_t_min", "edge_t_max", "edge_n0", "edge_n1",
            "exterior_angle",
        )
    ),
    *(
        arg
        for face in ("face0", "face1")
        for arg in (
            opaque(f"{face}_valid", grad="fixed", tangent=False),
            opaque(f"{face}_eps_r", grad="diff"),
            opaque(f"{face}_sigma_e", grad="diff"),
            opaque(f"{face}_mu_r", grad="fixed"),
            opaque(f"{face}_gain", grad="diff"),
        )
    ),
    opaque("tx_power", grad="fixed"),
)
_WEDGE_VERTICES = (
    *(
        opaque(name, grad="diff")
        for name in ("vertex_v0", "vertex_v1", "vertex_opp0", "vertex_opp1")
    ),
    opaque("edge_boundary", grad="fixed", tangent=False),
)
_WEDGE_MATERIALS = tuple(
    f"{face}_{field}" for face in ("face0", "face1") for field in ("eps_r", "sigma_e", "gain")
)
_ISB_TAPER = real("isb_boundary_taper_width")
# Optional slab / layer-stack tables of both wedge faces (see :func:`wedge_face_layers`); they carry
# no derivative.
_FACE_LAYERS = opaque("face_layers")
_FIELD_DIFFRACTION_WEDGE = _native.op(
    "field_diffraction_wedge", _WEDGE_PRIMALS, _FREQUENCY, _WEDGE_VERTICES, _ISB_TAPER,
    _FACE_LAYERS,
)
_FIELD_DIFFRACTION_WEDGE_BACKWARD = _native.op(
    "field_diffraction_wedge_backward", _WEDGE_PRIMALS, _FREQUENCY, _WEDGE_VERTICES,
    _values("grad_field_vector", "grad_direction"),
    _flags("need_grad_material", "need_grad_frequency", "need_grad_geometry", "need_grad_vertices"),
    _ISB_TAPER, _FACE_LAYERS,
)
_FIELD_DIFFRACTION_WEDGE_JVP = _native.op(
    "field_diffraction_wedge_jvp", _WEDGE_PRIMALS, _FREQUENCY, _WEDGE_VERTICES,
    _values("tangent_source", "tangent_target", *(f"tangent_{name}" for name in _WEDGE_MATERIALS)),
    real("tangent_frequency"),
    _values(
        "tangent_vertex_v0", "tangent_vertex_v1", "tangent_vertex_opp0", "tangent_vertex_opp1"
    ),
    _ISB_TAPER, _FACE_LAYERS,
)

_ROUGH_PRIMALS = (
    tensor("field_vector", _C64, 2, grad="diff"),
    tensor("coefficient", _C64, 1, grad="diff"),
    tensor("path_field", _C64, 1, grad="diff"),
    tensor("path_gain", _F32, 1, grad="diff"),
    tensor("positions", _F32, 3, grad="diff"),
    tensor("normals", _F32, 3, grad="diff"),
    tensor("source", _F32, 2, grad="diff"),
    tensor("sigma_b", _F32, 2, grad="fixed"),
    tensor("rough_b", torch.bool, 2),
    tensor("replaced", torch.bool, 1),
)
_FIELD_ROUGH_REFLECTION_SCALE = _native.op(
    "field_rough_reflection_scale", _ROUGH_PRIMALS, _FREQUENCY
)
_FIELD_ROUGH_REFLECTION_SCALE_BACKWARD = _native.op(
    "field_rough_reflection_scale_backward", _ROUGH_PRIMALS, _FREQUENCY, _COUPLED_GRADS,
    _flags("need_field", "need_geometry", "need_frequency"),
)
_FIELD_ROUGH_REFLECTION_SCALE_JVP = _native.op(
    "field_rough_reflection_scale_jvp", _ROUGH_PRIMALS, _FREQUENCY,
    _values(
        "tangent_field_vector", "tangent_coefficient", "tangent_path_field", "tangent_path_gain",
        "tangent_positions", "tangent_normals", "tangent_source",
    ),
    real("tangent_frequency"),
)

_PROJECT_PRIMALS = (
    tensor("field_vector", _C64, 2, grad="diff"),
    _rows3("direction", grad="diff"),
    _rows3("rx_polarization", grad="fixed"),
)
_FIELD_PROJECT_COMPLEX3 = _native.op("field_project_complex3", _PROJECT_PRIMALS)
_FIELD_PROJECT_COMPLEX3_BACKWARD = _native.op(
    "field_project_complex3_backward", _PROJECT_PRIMALS,
    _values("grad_coefficient", "grad_path_gain"),
    _flags("need_grad_field_vector", "need_grad_direction"),
)
_FIELD_PROJECT_COMPLEX3_JVP = _native.op(
    "field_project_complex3_jvp", _PROJECT_PRIMALS,
    _values("tangent_field_vector", "tangent_direction"),
)

_FIELD_SOURCE_AMPLITUDE_SCALE = _native.op(
    "field_source_amplitude_scale", tensor("field_vector"), tensor("tx_power")
)
_FIELD_SOURCE_AMPLITUDE_SCALE_BACKWARD = _native.op(
    "field_source_amplitude_scale_backward", tensor("tx_power"), tensor("grad_path_field_vector")
)
_FIELD_SOURCE_AMPLITUDE_SCALE_JVP = _native.op(
    "field_source_amplitude_scale_jvp", tensor("tx_power"), tensor("tangent_field_vector")
)

_FIELD_PARTS = (tensor("field_real", _F32, 1), tensor("field_imag", _F32, 1))
_DETERMINISTIC_PACK_COMPLEX = _native.op("deterministic_pack_complex", _FIELD_PARTS)
_DETERMINISTIC_PHASE_FROM_FIELD = _native.op("deterministic_phase_from_field", _FIELD_PARTS)
_DETERMINISTIC_ZERO_FIELD_PHASE = _native.op(
    "deterministic_zero_field_phase", tensor("reference", _F32, 1)
)


def _validate_enumerated_field_result(
    out: object, count: int, *, type_error: str, field_error: str, shape_error_prefix: str,
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
    source: torch.Tensor, target: torch.Tensor, tx_power: torch.Tensor,
    tx_polarization: torch.Tensor, rx_polarization: torch.Tensor, *, frequency_hz: float,
) -> dict[str, torch.Tensor]:
    primals = (source, target, tx_power, tx_polarization, rx_polarization)
    _FIELD_FREE_SPACE.check(primals, *_FREE_SPACE_CHECKS)
    count = int(source.shape[0])
    if any(int(row.shape[0]) != count for row in primals[1:]):
        raise ValueError("free-space field tensors must have matching rows")
    if frequency_hz <= 0.0:
        raise ValueError("frequency_hz must be positive")
    out = _FIELD_FREE_SPACE(*primals, frequency_hz)
    return _validate_enumerated_field_result(
        out,
        count,
        type_error="_channel.field_free_space must return a dict",
        field_error="_channel.field_free_space returned unexpected fields",
        shape_error_prefix="_channel.field_free_space",
    )


def field_project_complex3(
    field_vector: torch.Tensor, direction: torch.Tensor, rx_polarization: torch.Tensor,
) -> dict[str, torch.Tensor]:
    _FIELD_PROJECT_COMPLEX3.check((field_vector, direction, rx_polarization))
    count = int(field_vector.shape[0])
    if field_vector.shape != (count, 3):
        raise ValueError("field_vector must have shape (N, 3)")
    if direction.shape != (count, 3) or rx_polarization.shape != (count, 3):
        raise ValueError("direction and rx_polarization must match field_vector rows")
    out = _FIELD_PROJECT_COMPLEX3(field_vector, direction, rx_polarization)
    if not isinstance(out, dict) or set(out) != {"coefficient", "path_gain"}:
        raise TypeError("_channel.field_project_complex3 returned invalid fields")
    validate_cuda_tensor("coefficient", out["coefficient"], dtype=torch.complex64, ndim=1)
    validate_cuda_tensor("path_gain", out["path_gain"], dtype=torch.float32, ndim=1)
    if out["coefficient"].shape != (count,) or out["path_gain"].shape != (count,):
        raise ValueError("field projection returned invalid shapes")
    return out


@dataclass(frozen=True, slots=True)
class ReflectionLayers:
    """Per-bounce material ids (N, D) and the compiled CSR layer store of a reflection request.

    ``multilayer`` records, from the host-side material store, whether any material has more than
    one layer; such faces reflect as their whole stack.
    """

    bounce_material: torch.Tensor
    layer_offset: torch.Tensor
    layer_count: torch.Tensor
    layer_thickness_m: torch.Tensor
    layer_eps_r: torch.Tensor
    layer_sigma_e: torch.Tensor
    layer_mu_r: torch.Tensor
    multilayer: bool

    def native(self) -> tuple[torch.Tensor, ...]:
        return (
            self.bounce_material, self.layer_offset, self.layer_count, self.layer_thickness_m,
            self.layer_eps_r, self.layer_sigma_e, self.layer_mu_r,
        )


def reflection_layers(
    material: dict[str, torch.Tensor], face_id: torch.Tensor, *, multilayer: bool,
) -> ReflectionLayers:
    """Layer request of reflection rows hitting ``face_id`` (N, D), from the per-face material bundle."""

    return ReflectionLayers(
        bounce_material=material["material_id"][face_id].contiguous(),
        layer_offset=material["layer_offset"],
        layer_count=material["layer_count"],
        layer_thickness_m=material["layer_thickness_m"],
        layer_eps_r=material["layer_eps_r"],
        layer_sigma_e=material["layer_sigma_e"],
        layer_mu_r=material["layer_mu_r"],
        multilayer=bool(multilayer),
    )


def _native_layers(layers: ReflectionLayers | None, like: torch.Tensor) -> tuple[torch.Tensor, ...]:
    """Native layer arguments; without a layer request every face uses its scalar slab view."""

    if layers is not None:
        return layers.native()
    # Never read: an empty layer store marks every face single-layer.
    index = torch.empty(like.shape, device=like.device, dtype=torch.int32)
    empty_i = torch.empty((0,), device=like.device, dtype=torch.int32)
    empty_f = torch.empty((0,), device=like.device, dtype=torch.float32)
    return (index, empty_i, empty_i, empty_f, empty_f, empty_f, empty_f)


_MULTILAYER_MATERIAL_AD_ERROR = (
    "material derivatives (eps_r, sigma_e, thickness) of reflection off a multilayer material are not "
    "supported; geometry, gain and frequency derivatives are"
)


def field_reflection_sequence(
    source: torch.Tensor, target: torch.Tensor, interaction_positions: torch.Tensor,
    interaction_normals: torch.Tensor, tx_power: torch.Tensor, tx_polarization: torch.Tensor,
    rx_polarization: torch.Tensor, eps_r: torch.Tensor, sigma_e: torch.Tensor, mu_r: torch.Tensor,
    gain: torch.Tensor, thickness: torch.Tensor, *, layers: ReflectionLayers | None = None,
    frequency_hz: float,
) -> dict[str, torch.Tensor]:
    primals = (
        source, target, interaction_positions, interaction_normals, tx_power, tx_polarization,
        rx_polarization, eps_r, sigma_e, mu_r, gain, thickness,
    )
    spec = _FIELD_REFLECTION_SEQUENCE
    spec.check(
        primals, "source", "target", "tx_polarization", "rx_polarization",
        "interaction_positions", "interaction_normals", "tx_power",
    )
    count = int(source.shape[0])
    depth = int(interaction_positions.shape[1])
    if interaction_positions.shape != (count, depth, 3) or depth <= 0:
        raise ValueError("interaction_positions must have shape (N, D, 3), D > 0")
    if interaction_normals.shape != interaction_positions.shape:
        raise ValueError("interaction_normals must match interaction_positions")
    for name in _REFLECTION_MATERIALS:
        spec.check(primals, name)
        if primals[spec.names.index(name)].shape != (count, depth):
            raise ValueError(f"{name} must have shape (N, D)")
    if frequency_hz <= 0.0:
        raise ValueError("frequency_hz must be positive")
    out = spec(*primals, *_native_layers(layers, eps_r), frequency_hz)
    return _validate_enumerated_field_result(
        out,
        count,
        type_error="_channel.field_reflection_sequence must return a dict",
        field_error="field_reflection_sequence returned unexpected fields",
        shape_error_prefix="field_reflection_sequence",
    )


def field_transport_columns(
    *arguments: torch.Tensor, depth: int, frequencies_hz: tuple[float, ...],
    layers: ReflectionLayers | None = None,
) -> dict[str, torch.Tensor]:
    """Free-space (``depth == 0``) or reflection field at several carrier frequencies.

    ``arguments`` are the positional tensors of :func:`field_free_space` or
    :func:`field_reflection_sequence`. Each (row, frequency) lane runs that op's
    single-frequency arithmetic, so column ``f`` is bitwise the single op at
    ``frequencies_hz[f]``. Returns ``field_vector`` ``[N, F, 3]`` and ``path_field`` ``[N, F]``.
    """

    frequencies = tuple(float(value) for value in frequencies_hz)
    if not frequencies or any(not value > 0.0 for value in frequencies):
        raise ValueError("frequencies_hz must be a non-empty tuple of positive values")
    spec, expected = (
        (_FIELD_FREE_SPACE_COLUMNS, 5) if depth == 0 else (_FIELD_REFLECTION_SEQUENCE_COLUMNS, 12)
    )
    name = spec.symbol
    if len(arguments) != expected:
        raise TypeError(f"{name} takes {expected} tensors")
    for value in arguments:
        validate_cuda_tensor("field input", value, dtype=torch.float32, ndim=value.ndim)
    count = int(arguments[0].shape[0])
    native_layers = () if depth == 0 else _native_layers(layers, arguments[7])
    out = spec(*arguments, *native_layers, list(frequencies))
    if not isinstance(out, dict) or set(out) != {"field_vector", "path_field"}:
        raise TypeError(f"_channel.{name} returned invalid fields")
    columns = len(frequencies)
    validate_cuda_tensor("field_vector", out["field_vector"], dtype=torch.complex64, ndim=3)
    validate_cuda_tensor("path_field", out["path_field"], dtype=torch.complex64, ndim=2)
    if out["field_vector"].shape != (count, columns, 3) or out["path_field"].shape != (
        count,
        columns,
    ):
        raise ValueError(f"_channel.{name} returned bad shapes")
    return out


_CANONICAL_ROW_COLUMNS = (
    "field_xyz", "coefficient", "path_field", "path_gain", "path_length", "delay", "direction",
)


@dataclass(frozen=True, slots=True)
class CanonicalFaceTables:
    """Per-face slab tables, the layer store and the rough-surface tables of a canonical launch.

    ``rough_sigma``/``rough`` are None on a smooth scene and ``replaced`` is None unless a
    realization phase screen replaces the specular reflection of its faces.
    """

    eps_r: torch.Tensor
    sigma_e: torch.Tensor
    mu_r: torch.Tensor
    gain: torch.Tensor
    thickness: torch.Tensor
    material: torch.Tensor
    layers: tuple[torch.Tensor, ...]
    multilayer: bool
    rough_sigma: torch.Tensor | None = None
    rough: torch.Tensor | None = None
    replaced: torch.Tensor | None = None

    @classmethod
    def empty(cls, device: torch.device) -> CanonicalFaceTables:
        """Tables of a launch without reflection rows; nothing reads them."""

        empty_f = torch.empty((0,), device=device, dtype=torch.float32)
        empty_i = torch.empty((0,), device=device, dtype=torch.int32)
        return cls(
            eps_r=empty_f, sigma_e=empty_f, mu_r=empty_f, gain=empty_f, thickness=empty_f,
            material=empty_i, layers=(empty_i, empty_i, empty_f, empty_f, empty_f, empty_f),
            multilayer=False,
        )


def field_canonical_rows(
    *, order: torch.Tensor | None, component_id: torch.Tensor, depth: torch.Tensor,
    tx_id: torch.Tensor, rx_id: torch.Tensor, primitive_sequence: torch.Tensor,
    tx_positions: torch.Tensor, tx_power: torch.Tensor, tx_polarization: torch.Tensor,
    rx_positions: torch.Tensor, rx_polarization: torch.Tensor, interaction_positions: torch.Tensor,
    interaction_normals: torch.Tensor, fields: dict[str, torch.Tensor], faces: CanonicalFaceTables,
    evaluate_los: bool, frequency_hz: float,
) -> dict[str, torch.Tensor]:
    """Field-stage columns of every canonical row in one native launch.

    Line-of-sight rows (when ``evaluate_los``), reflection rows of depth ``1..W`` and order-1
    diffraction rows are evaluated with the arithmetic of :func:`field_free_space`,
    :func:`field_reflection_sequence` (plus the rough-surface factor) and the diffraction
    receiver projection; every other row keeps its value from ``fields``. Endpoints are read
    from the per-transmitter and per-receiver tables by ``tx_id``/``rx_id``. Returns new
    tensors keyed like ``fields`` (``field_xyz``, ``coefficient``, ``path_field``,
    ``path_gain``, ``path_length``, ``delay``, ``direction``).
    """

    if set(fields) != set(_CANONICAL_ROW_COLUMNS):
        raise ValueError("fields must hold exactly the canonical field-stage columns")
    if not frequency_hz > 0.0:
        raise ValueError("frequency_hz must be positive")
    device = component_id.device
    empty_f = torch.empty((0,), device=device, dtype=torch.float32)
    empty_b = torch.empty((0,), device=device, dtype=torch.bool)
    out = _FIELD_CANONICAL_ROWS(
        torch.empty((0,), device=device, dtype=torch.int64) if order is None else order,
        component_id, depth, tx_id, rx_id, primitive_sequence, tx_positions, tx_power,
        tx_polarization, rx_positions, rx_polarization, interaction_positions,
        interaction_normals, *(fields[name] for name in _CANONICAL_ROW_COLUMNS), faces.eps_r,
        faces.sigma_e, faces.mu_r, faces.gain, faces.thickness, faces.material, *faces.layers,
        faces.multilayer, empty_f if faces.rough_sigma is None else faces.rough_sigma,
        empty_b if faces.rough is None else faces.rough,
        empty_b if faces.replaced is None else faces.replaced, faces.rough is not None,
        faces.replaced is not None, evaluate_los, frequency_hz,
    )
    if not isinstance(out, dict) or set(out) != set(_CANONICAL_ROW_COLUMNS):
        raise TypeError("_channel.field_canonical_rows returned invalid fields")
    for name in _CANONICAL_ROW_COLUMNS:
        if out[name].shape != fields[name].shape or out[name].dtype != fields[name].dtype:
            raise ValueError(f"_channel.field_canonical_rows returned bad {name}")
    return out


def field_transmission_sequence(
    path_valid: torch.Tensor, source: torch.Tensor, target: torch.Tensor,
    interaction_positions: torch.Tensor, interaction_normals: torch.Tensor,
    interaction_material_id: torch.Tensor, interaction_valid: torch.Tensor, tx_power: torch.Tensor,
    tx_polarization: torch.Tensor, rx_polarization: torch.Tensor, layer_offset: torch.Tensor,
    layer_count: torch.Tensor, layer_thickness_m: torch.Tensor, layer_eps_r: torch.Tensor,
    layer_sigma_e: torch.Tensor, layer_mu_r: torch.Tensor, *, frequency_hz: float,
) -> dict[str, torch.Tensor]:
    primals = (
        path_valid, source, target, interaction_positions, interaction_normals,
        interaction_material_id, interaction_valid, tx_power, tx_polarization, rx_polarization,
        layer_offset, layer_count, layer_thickness_m, layer_eps_r, layer_sigma_e, layer_mu_r,
    )
    _FIELD_TRANSMISSION_SEQUENCE.check(primals, *_TRANSMISSION_CHECKS)
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
        int(row.shape[0]) != count
        for row in (target, tx_power, tx_polarization, rx_polarization)
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
    out = _FIELD_TRANSMISSION_SEQUENCE(*primals, frequency_hz)
    return _validate_enumerated_field_result(
        out,
        count,
        type_error="_channel.field_transmission_sequence must return a dict",
        field_error="field_transmission_sequence returned unexpected fields",
        shape_error_prefix="field_transmission_sequence",
    )


_CHAIN_MATERIAL_NAMES = ("eps_r", "sigma_e", "mu_r", "gain", "thickness")


def _validate_chain_inputs(
    source: torch.Tensor, target: torch.Tensor, positions: torch.Tensor, normals: torch.Tensor,
    component_id: torch.Tensor, depth: torch.Tensor, edge_direction: torch.Tensor,
    edge_n0: torch.Tensor, edge_n1: torch.Tensor, exterior_angle: torch.Tensor,
    edge_line_min: torch.Tensor, edge_line_max: torch.Tensor, tx_power: torch.Tensor,
    tx_polarization: torch.Tensor, rx_polarization: torch.Tensor,
    material: tuple[torch.Tensor, ...],
) -> None:
    count = int(source.shape[0])
    for name, value in (
        ("source", source), ("target", target), ("edge_direction", edge_direction),
        ("edge_n0", edge_n0), ("edge_n1", edge_n1), ("tx_polarization", tx_polarization),
        ("rx_polarization", rx_polarization),
    ):
        validate_cuda_tensor(name, value, dtype=torch.float32, ndim=2, trailing_shape=(3,))
        if int(value.shape[0]) != count:
            raise ValueError(f"chain field {name} must have one row per path")
    for name, value in (("positions", positions), ("normals", normals)):
        validate_cuda_tensor(name, value, dtype=torch.float32, ndim=3, trailing_shape=(3,))
        if int(value.shape[0]) != count:
            raise ValueError(f"chain field {name} must have one row per path")
    width = int(positions.shape[1])
    if not 2 <= width <= 8 or int(normals.shape[1]) != width:
        raise ValueError("chain field positions and normals must share a width in [2, 8]")
    for name, value in (("component_id", component_id), ("depth", depth)):
        validate_cuda_tensor(name, value, dtype=torch.int32, ndim=1)
        if int(value.shape[0]) != count:
            raise ValueError(f"chain field {name} must have one row per path")
    for name, value in (
        ("exterior_angle", exterior_angle), ("edge_line_min", edge_line_min),
        ("edge_line_max", edge_line_max), ("tx_power", tx_power),
    ):
        validate_cuda_tensor(name, value, dtype=torch.float32, ndim=1)
        if int(value.shape[0]) != count:
            raise ValueError(f"chain field {name} must have one row per path")
    if len(material) != len(_CHAIN_MATERIAL_NAMES):
        raise ValueError("chain material bundle must contain eps/sigma/mu/gain/thickness")
    for name, value in zip(_CHAIN_MATERIAL_NAMES, material, strict=True):
        validate_cuda_tensor(f"chain {name}", value, dtype=torch.float32, ndim=2)
        if tuple(value.shape) != (count, width + 2):
            raise ValueError(f"chain {name} must have shape (N, W + 2)")


def field_coupled_chain(
    source: torch.Tensor, target: torch.Tensor, positions: torch.Tensor, normals: torch.Tensor,
    component_id: torch.Tensor, depth: torch.Tensor, edge_direction: torch.Tensor,
    edge_n0: torch.Tensor, edge_n1: torch.Tensor, exterior_angle: torch.Tensor,
    edge_line_min: torch.Tensor, edge_line_max: torch.Tensor, tx_power: torch.Tensor,
    tx_polarization: torch.Tensor, rx_polarization: torch.Tensor,
    material: tuple[torch.Tensor, ...], *, frequency_hz: float,
    layers: ReflectionLayers | None = None,
) -> dict[str, torch.Tensor]:
    """Field of interaction-chain rows (reflections, one diffraction, reflections).

    ``positions`` and ``normals`` are ``(N, W, 3)`` interaction points and reflection normals;
    the component id and depth place the diffraction. ``material`` holds ``(N, W + 2)``
    eps/sigma/mu/gain/thickness tables: the reflection slots, then both wedge faces.
    ``layers.bounce_material`` has shape ``(N, W + 2)``: the layer-store rows of the same columns.
    """

    _validate_chain_inputs(
        source, target, positions, normals, component_id, depth, edge_direction, edge_n0,
        edge_n1, exterior_angle, edge_line_min, edge_line_max, tx_power, tx_polarization,
        rx_polarization, material,
    )
    if frequency_hz <= 0.0:
        raise ValueError("frequency_hz must be positive")
    count = int(source.shape[0])
    out = _FIELD_COUPLED_CHAIN(
        source, target, positions, normals, component_id, depth, edge_direction, edge_n0,
        edge_n1, exterior_angle, edge_line_min, edge_line_max, tx_power, tx_polarization,
        rx_polarization, *material, *_native_layers(layers, material[0]), frequency_hz,
    )
    if not isinstance(out, dict):
        raise TypeError("_channel.field_coupled_chain must return a dict")
    schema = {
        "field_vector": (torch.complex64, 2, (count, 3)),
        "coefficient": (torch.complex64, 1, (count,)),
        "path_field": (torch.complex64, 1, (count,)),
        "path_gain": (torch.float32, 1, (count,)),
        "direction": (torch.float32, 2, (count, 3)),
    }
    if set(out) != set(schema):
        raise ValueError("field_coupled_chain returned unexpected fields")
    for name, (dtype, ndim, shape) in schema.items():
        validate_cuda_tensor(name, out[name], dtype=dtype, ndim=ndim)
        if tuple(out[name].shape) != shape:
            raise ValueError(f"field_coupled_chain returned bad {name} shape")
    return out


def field_coupled_dd(
    source: torch.Tensor, target: torch.Tensor, edge1_position: torch.Tensor,
    edge1_direction: torch.Tensor, edge1_n0: torch.Tensor, edge1_n1: torch.Tensor,
    edge1_exterior: torch.Tensor, edge2_position: torch.Tensor, edge2_direction: torch.Tensor,
    edge2_n0: torch.Tensor, edge2_n1: torch.Tensor, edge2_exterior: torch.Tensor,
    tx_power: torch.Tensor, tx_polarization: torch.Tensor, rx_polarization: torch.Tensor,
    wedge1_material0: tuple[torch.Tensor, ...], wedge1_material1: tuple[torch.Tensor, ...],
    wedge2_material0: tuple[torch.Tensor, ...], wedge2_material1: tuple[torch.Tensor, ...],
    edge1_line_min: torch.Tensor, edge1_line_max: torch.Tensor, edge2_line_min: torch.Tensor,
    edge2_line_max: torch.Tensor, *, frequency_hz: float,
) -> dict[str, torch.Tensor]:
    """Coupled double-diffraction field (TX->e1->e2->RX), component id 7.

    Two sequential wedge operators in one native launch (coupled double diffraction). Outputs
    are identical in shape to :func:`field_coupled_chain`.
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
    out = _FIELD_COUPLED_DD(
        source, target, edge1_position, edge1_direction, edge1_n0, edge1_n1, edge1_exterior,
        edge2_position, edge2_direction, edge2_n0, edge2_n1, edge2_exterior, tx_power,
        tx_polarization, rx_polarization, *wedge1_material0, *wedge1_material1,
        *wedge2_material0, *wedge2_material1, edge1_line_min, edge1_line_max, edge2_line_min,
        edge2_line_max, frequency_hz,
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
# tangent (first-order differentiation). Transmission and the wedge/coupled families forward to
# rayd::torch, which owns their direction seam and does not publish it, so the
# shared tuple above stays exactly what those families return.
_FIELD_AD_DIRECTION_TANGENT_FIELDS = (*_FIELD_AD_TANGENT_FIELDS, "direction")


def field_free_space_backward(
    source: torch.Tensor, target: torch.Tensor, tx_power: torch.Tensor,
    tx_polarization: torch.Tensor, rx_polarization: torch.Tensor, *, frequency_hz: float,
    grad_field_vector: torch.Tensor | None = None, grad_coefficient: torch.Tensor | None = None,
    grad_path_field: torch.Tensor | None = None, grad_path_gain: torch.Tensor | None = None,
    grad_path_length: torch.Tensor | None = None, grad_delay: torch.Tensor | None = None,
    grad_direction: torch.Tensor | None = None, need_grad_frequency: bool = True,
    need_grad_geometry: bool = False,
) -> dict[str, torch.Tensor | None]:
    out = _FIELD_FREE_SPACE_BACKWARD(
        source, target, tx_power, tx_polarization, rx_polarization, frequency_hz,
        grad_field_vector, grad_coefficient, grad_path_field, grad_path_gain, grad_path_length,
        grad_delay, grad_direction, need_grad_frequency, need_grad_geometry,
    )
    expected = {"grad_frequency", "grad_source", "grad_target"}
    if not isinstance(out, dict) or set(out) != expected:
        raise TypeError("_channel.field_free_space_backward returned invalid fields")
    return out


def field_free_space_jvp(
    source: torch.Tensor, target: torch.Tensor, tx_power: torch.Tensor,
    tx_polarization: torch.Tensor, rx_polarization: torch.Tensor, *, frequency_hz: float,
    tangent_frequency: float, tangent_source: torch.Tensor | None = None,
    tangent_target: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    out = _FIELD_FREE_SPACE_JVP(
        source, target, tx_power, tx_polarization, rx_polarization, frequency_hz,
        tangent_frequency, tangent_source, tangent_target,
    )
    if not isinstance(out, dict) or set(out) != set(
        _FIELD_AD_DIRECTION_TANGENT_FIELDS
    ):
        raise TypeError("_channel.field_free_space_jvp returned invalid fields")
    return out


def field_reflection_sequence_backward(
    source: torch.Tensor, target: torch.Tensor, interaction_positions: torch.Tensor,
    interaction_normals: torch.Tensor, tx_power: torch.Tensor, tx_polarization: torch.Tensor,
    rx_polarization: torch.Tensor, eps_r: torch.Tensor, sigma_e: torch.Tensor, mu_r: torch.Tensor,
    gain: torch.Tensor, thickness: torch.Tensor, *, layers: ReflectionLayers | None = None,
    frequency_hz: float, grad_field_vector: torch.Tensor | None = None,
    grad_coefficient: torch.Tensor | None = None, grad_path_field: torch.Tensor | None = None,
    grad_path_gain: torch.Tensor | None = None, grad_path_length: torch.Tensor | None = None,
    grad_delay: torch.Tensor | None = None, grad_direction: torch.Tensor | None = None,
    need_grad_eps_r: bool = True, need_grad_sigma_e: bool = True, need_grad_gain: bool = False,
    need_grad_thickness: bool = True, need_grad_frequency: bool = True,
    need_grad_geometry: bool = False,
) -> dict[str, torch.Tensor | None]:
    if layers is not None and layers.multilayer and (need_grad_eps_r or need_grad_sigma_e or need_grad_thickness):
        raise NotImplementedError(_MULTILAYER_MATERIAL_AD_ERROR)
    out = _FIELD_REFLECTION_SEQUENCE_BACKWARD(
        source, target, interaction_positions, interaction_normals, tx_power, tx_polarization,
        rx_polarization, eps_r, sigma_e, mu_r, gain, thickness, *_native_layers(layers, eps_r),
        frequency_hz, grad_field_vector, grad_coefficient, grad_path_field, grad_path_gain,
        grad_path_length, grad_delay, grad_direction, need_grad_eps_r, need_grad_sigma_e,
        need_grad_gain, need_grad_thickness, need_grad_frequency, need_grad_geometry,
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
    source: torch.Tensor, target: torch.Tensor, interaction_positions: torch.Tensor,
    interaction_normals: torch.Tensor, tx_power: torch.Tensor, tx_polarization: torch.Tensor,
    rx_polarization: torch.Tensor, eps_r: torch.Tensor, sigma_e: torch.Tensor, mu_r: torch.Tensor,
    gain: torch.Tensor, thickness: torch.Tensor, *, layers: ReflectionLayers | None = None,
    frequency_hz: float, tangent_eps_r: torch.Tensor | None = None,
    tangent_sigma_e: torch.Tensor | None = None, tangent_gain: torch.Tensor | None = None,
    tangent_thickness: torch.Tensor | None = None, tangent_frequency: float = 0.0,
    tangent_source: torch.Tensor | None = None, tangent_target: torch.Tensor | None = None,
    tangent_interaction_positions: torch.Tensor | None = None,
    tangent_interaction_normals: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    if layers is not None and layers.multilayer and (
        tangent_eps_r is not None or tangent_sigma_e is not None or tangent_thickness is not None
    ):
        raise NotImplementedError(_MULTILAYER_MATERIAL_AD_ERROR)
    out = _FIELD_REFLECTION_SEQUENCE_JVP(
        source, target, interaction_positions, interaction_normals, tx_power, tx_polarization,
        rx_polarization, eps_r, sigma_e, mu_r, gain, thickness, *_native_layers(layers, eps_r),
        frequency_hz, tangent_eps_r, tangent_sigma_e, tangent_gain, tangent_thickness,
        tangent_frequency, tangent_source, tangent_target, tangent_interaction_positions,
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
    path_valid: torch.Tensor, source: torch.Tensor, target: torch.Tensor,
    interaction_positions: torch.Tensor, interaction_normals: torch.Tensor,
    interaction_material_id: torch.Tensor, interaction_valid: torch.Tensor, tx_power: torch.Tensor,
    tx_polarization: torch.Tensor, rx_polarization: torch.Tensor, layer_offset: torch.Tensor,
    layer_count: torch.Tensor, layer_thickness_m: torch.Tensor, layer_eps_r: torch.Tensor,
    layer_sigma_e: torch.Tensor, layer_mu_r: torch.Tensor, *, frequency_hz: float,
    grad_field_vector: torch.Tensor | None = None, grad_coefficient: torch.Tensor | None = None,
    grad_path_field: torch.Tensor | None = None, grad_path_gain: torch.Tensor | None = None,
    grad_path_length: torch.Tensor | None = None, grad_delay: torch.Tensor | None = None,
    need_grad_layer_thickness: bool = True, need_grad_layer_eps_r: bool = True,
    need_grad_layer_sigma_e: bool = True, need_grad_frequency: bool = True,
    need_grad_geometry: bool = False,
) -> dict[str, torch.Tensor | None]:
    out = _FIELD_TRANSMISSION_SEQUENCE_BACKWARD(
        path_valid, source, target, interaction_positions, interaction_normals,
        interaction_material_id, interaction_valid, tx_power, tx_polarization, rx_polarization,
        layer_offset, layer_count, layer_thickness_m, layer_eps_r, layer_sigma_e, layer_mu_r,
        frequency_hz, grad_field_vector, grad_coefficient, grad_path_field, grad_path_gain,
        grad_path_length, grad_delay, need_grad_layer_thickness, need_grad_layer_eps_r,
        need_grad_layer_sigma_e, need_grad_frequency, need_grad_geometry,
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
    path_valid: torch.Tensor, source: torch.Tensor, target: torch.Tensor,
    interaction_positions: torch.Tensor, interaction_normals: torch.Tensor,
    interaction_material_id: torch.Tensor, interaction_valid: torch.Tensor, tx_power: torch.Tensor,
    tx_polarization: torch.Tensor, rx_polarization: torch.Tensor, layer_offset: torch.Tensor,
    layer_count: torch.Tensor, layer_thickness_m: torch.Tensor, layer_eps_r: torch.Tensor,
    layer_sigma_e: torch.Tensor, layer_mu_r: torch.Tensor, *, frequency_hz: float,
    tangent_layer_thickness_m: torch.Tensor | None = None,
    tangent_layer_eps_r: torch.Tensor | None = None,
    tangent_layer_sigma_e: torch.Tensor | None = None, tangent_frequency: float = 0.0,
    tangent_source: torch.Tensor | None = None, tangent_target: torch.Tensor | None = None,
    tangent_interaction_positions: torch.Tensor | None = None,
    tangent_interaction_normals: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    out = _FIELD_TRANSMISSION_SEQUENCE_JVP(
        path_valid, source, target, interaction_positions, interaction_normals,
        interaction_material_id, interaction_valid, tx_power, tx_polarization, rx_polarization,
        layer_offset, layer_count, layer_thickness_m, layer_eps_r, layer_sigma_e, layer_mu_r,
        frequency_hz, tangent_layer_thickness_m, tangent_layer_eps_r, tangent_layer_sigma_e,
        tangent_frequency, tangent_source, tangent_target, tangent_interaction_positions,
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


def wedge_face_layers(
    material: dict[str, torch.Tensor], face0: torch.Tensor, face1: torch.Tensor,
) -> tuple[torch.Tensor, ...]:
    """Slab / layer-stack tables of wedge rows with faces ``face0``/``face1``, from the material bundle.

    The tuple is (face0 thickness, face1 thickness, face0 layer-store row, face1 layer-store row,
    layer offset, count, thickness, eps_r, sigma_e, mu_r), the native ``face_layers`` argument.
    """

    return (
        material["thickness"][face0].contiguous(), material["thickness"][face1].contiguous(),
        material["material_id"][face0].contiguous(), material["material_id"][face1].contiguous(),
        material["layer_offset"], material["layer_count"], material["layer_thickness_m"],
        material["layer_eps_r"], material["layer_sigma_e"], material["layer_mu_r"],
    )


def field_diffraction_wedge(
    valid: torch.Tensor, source: torch.Tensor, target: torch.Tensor, edge_position: torch.Tensor,
    edge_direction: torch.Tensor, edge_t_min: torch.Tensor, edge_t_max: torch.Tensor,
    edge_n0: torch.Tensor, edge_n1: torch.Tensor, exterior_angle: torch.Tensor,
    face0_valid: torch.Tensor, face0_eps_r: torch.Tensor, face0_sigma_e: torch.Tensor,
    face0_mu_r: torch.Tensor, face0_gain: torch.Tensor, face1_valid: torch.Tensor,
    face1_eps_r: torch.Tensor, face1_sigma_e: torch.Tensor, face1_mu_r: torch.Tensor,
    face1_gain: torch.Tensor, tx_power: torch.Tensor, *, frequency_hz: float,
    face_layers: tuple[torch.Tensor, ...] | None = None,
) -> dict[str, torch.Tensor]:
    """Re-evaluate RayD's order-1 UTD wedge export from the frozen topology.

    ``face_layers`` optionally gives the faces their slab / layer-stack reflection: per row the two
    faces' thickness and layer-store row, then the layer store (see :func:`wedge_face_layers`).
    """

    _validate_wedge_valid(valid, source)
    out = _FIELD_DIFFRACTION_WEDGE(
        valid, source, target, edge_position, edge_direction, edge_t_min, edge_t_max, edge_n0,
        edge_n1, exterior_angle, face0_valid, face0_eps_r, face0_sigma_e, face0_mu_r, face0_gain,
        face1_valid, face1_eps_r, face1_sigma_e, face1_mu_r, face1_gain, tx_power, frequency_hz,
        None, None, None, None, None, 0.0, face_layers,
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
    field_vector: torch.Tensor, coefficient: torch.Tensor, path_field: torch.Tensor,
    path_gain: torch.Tensor, positions: torch.Tensor, normals: torch.Tensor, source: torch.Tensor,
    sigma_b: torch.Tensor, rough_b: torch.Tensor, replaced: torch.Tensor, *, frequency_hz: float,
) -> dict[str, torch.Tensor]:
    """Native rough-surface C_r factor applied to the reflection outputs.

    Computes ``C_r = prod_b exp(-2*(k0*cos_b*sigma_b)^2)`` on rough bounces
    (``1`` otherwise), zeroes rows flagged ``replaced``, and scales the four
    field outputs (``path_gain`` by ``C_r^2``). Returns the scaled outputs plus
    the real ``factor`` per row.
    """

    primals = (
        field_vector, coefficient, path_field, path_gain, positions, normals, source, sigma_b,
        rough_b, replaced,
    )
    _FIELD_ROUGH_REFLECTION_SCALE.check(primals + (frequency_hz,))
    out = _FIELD_ROUGH_REFLECTION_SCALE(*primals, frequency_hz)
    expected = {*_ROUGH_SCALE_OUTPUT_FIELDS, "factor"}
    if not isinstance(out, dict) or set(out) != expected:
        raise TypeError(
            "_channel.field_rough_reflection_scale returned invalid fields"
        )
    return out


def field_rough_reflection_scale_backward(
    field_vector: torch.Tensor, coefficient: torch.Tensor, path_field: torch.Tensor,
    path_gain: torch.Tensor, positions: torch.Tensor, normals: torch.Tensor, source: torch.Tensor,
    sigma_b: torch.Tensor, rough_b: torch.Tensor, replaced: torch.Tensor, *, frequency_hz: float,
    grad_field_vector: torch.Tensor | None = None, grad_coefficient: torch.Tensor | None = None,
    grad_path_field: torch.Tensor | None = None, grad_path_gain: torch.Tensor | None = None,
    need_field: bool = True, need_geometry: bool = False, need_frequency: bool = True,
) -> dict[str, torch.Tensor | None]:
    """VJP of :func:`field_rough_reflection_scale` (frequency and geometry)."""

    out = _FIELD_ROUGH_REFLECTION_SCALE_BACKWARD(
        field_vector, coefficient, path_field, path_gain, positions, normals, source, sigma_b,
        rough_b, replaced, frequency_hz, grad_field_vector, grad_coefficient, grad_path_field,
        grad_path_gain, need_field, need_geometry, need_frequency,
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
    field_vector: torch.Tensor, coefficient: torch.Tensor, path_field: torch.Tensor,
    path_gain: torch.Tensor, positions: torch.Tensor, normals: torch.Tensor, source: torch.Tensor,
    sigma_b: torch.Tensor, rough_b: torch.Tensor, replaced: torch.Tensor, *, frequency_hz: float,
    tangent_field_vector: torch.Tensor | None = None,
    tangent_coefficient: torch.Tensor | None = None, tangent_path_field: torch.Tensor | None = None,
    tangent_path_gain: torch.Tensor | None = None, tangent_positions: torch.Tensor | None = None,
    tangent_normals: torch.Tensor | None = None, tangent_source: torch.Tensor | None = None,
    tangent_frequency: float = 0.0,
) -> dict[str, torch.Tensor]:
    """JVP of :func:`field_rough_reflection_scale` (frequency and geometry)."""

    out = _FIELD_ROUGH_REFLECTION_SCALE_JVP(
        field_vector, coefficient, path_field, path_gain, positions, normals, source, sigma_b,
        rough_b, replaced, frequency_hz, tangent_field_vector, tangent_coefficient,
        tangent_path_field, tangent_path_gain, tangent_positions, tangent_normals,
        tangent_source, tangent_frequency,
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


def mark_dead_outputs(ctx: Any, output: Any) -> None:
    """Declare the outputs of one apply that carry no derivative.

    A dead output is marked exactly as it was before the direction seam
    existed, so a caller that does not ask for a live direction sees the same
    object graph it saw at contract version 5.
    """

    dead: list[torch.Tensor] = []
    if not ctx.geometry_live:
        dead.extend((output[4], output[5]))
    if not ctx.direction_live:
        dead.append(output[6])
    if dead:
        ctx.mark_non_differentiable(*dead)


def direction_cotangent(ctx: Any, grad_direction: torch.Tensor | None) -> torch.Tensor | None:
    """The incoming direction cotangent, or ``None`` if it was declared dead.

    Torch does not deliver a cotangent for an output marked non-differentiable,
    so this is belt and braces; it is also the one place that says out loud that
    a dead output's seed never reaches a native companion.
    """

    return grad_direction if ctx.direction_live else None


def direction_tangents(ctx: Any, out: dict[str, torch.Tensor]) -> tuple:
    """Publish one apply's output tangents under the two liveness decisions.

    The native companion always computes the direction tangent - it is the dual
    the transverse projection already needed - so this only decides whether it
    is published. A dead output receives ``None`` rather than a zero tensor,
    because a zero tangent on a declared-dead output is exactly the silent
    answer first-order differentiation
    """

    tangents = tuple(out[name] for name in _FIELD_AD_DIRECTION_TANGENT_FIELDS)
    if not ctx.geometry_live:
        return (*tangents[:4], None, None, None)
    if not ctx.direction_live:
        return (*tangents[:6], None)
    return tangents


def _grad_keys(*names: str) -> dict[str, str]:
    """The native gradient key of each named input (``grad_<name>``)."""

    return {name: f"grad_{name}" for name in names}


_GEOMETRY_GRAD_KEYS = _grad_keys("source", "target")
_FREE_SPACE_AD = AdInputs(
    "field_free_space_ad", _FREE_SPACE_PRIMALS, opaque("frequency", grad="diff"),
    opaque("frequency_value"), opaque("liveness"),
)


class _FieldFreeSpaceAdFunction(torch.autograd.Function):
    """Fixed-topology differentiable free-space transport.

    Frequency and endpoints are differentiable; power and polarizations are
    fixed. Float64 inputs use the strict-double companion for gradcheck.
    """

    @staticmethod
    def forward(
        source: Any, target: Any, tx_power: Any, tx_polarization: Any, rx_polarization: Any,
        frequency: Any, frequency_value: Any, liveness: Any,
    ) -> Any:
        op = _FIELD_FREE_SPACE_FWD64 if source.dtype == torch.float64 else _FIELD_FREE_SPACE
        out = op(source, target, tx_power, tx_polarization, rx_polarization, frequency_value)
        return tuple(out[name] for name in _FIELD_AD_OUTPUT_FIELDS)

    @staticmethod
    def setup_context(ctx: Any, inputs: Any, output: Any) -> Any:
        ctx.set_materialize_grads(False)
        primals = _FREE_SPACE_AD.primals(inputs, *_FIELD_FREE_SPACE.names[:5])
        ctx.frequency_value = _FREE_SPACE_AD.get(inputs, "frequency_value")
        ctx.frequency_meta = frequency_meta(_FREE_SPACE_AD.get(inputs, "frequency"))
        # Both decided by the wrapper, where forward duals are still visible;
        # Function.apply unpacks them before this hook runs.
        ctx.geometry_live, ctx.direction_live = _FREE_SPACE_AD.get(inputs, "liveness")
        save(ctx, *primals)
        mark_dead_outputs(ctx, output)

    @staticmethod
    @_ad_first_order_only
    def backward(
        ctx: Any, grad_field_vector: Any, grad_coefficient: Any, grad_path_field: Any,
        grad_path_gain: Any, grad_path_length: Any, grad_delay: Any, grad_direction: Any,
    ) -> Any:
        _FREE_SPACE_AD.reject_fixed_grads(ctx)
        need_geometry = _FREE_SPACE_AD.needs(ctx, "source", "target")
        need_frequency = _FREE_SPACE_AD.need(ctx, "frequency")
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
            return _FREE_SPACE_AD.none
        out = field_free_space_backward(
            *ctx.saved_tensors,
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
        return _FREE_SPACE_AD.output_grads(
            ctx, out, _GEOMETRY_GRAD_KEYS, frequency=grad_frequency
        )

    @staticmethod
    def jvp(ctx: Any, *tangents: Any) -> Any:
        _FREE_SPACE_AD.reject_fixed_tangents(tangents)
        saved = ctx.saved_tensors
        tangent_source = _FREE_SPACE_AD.geometry_tangent(tangents, "source", saved[0])
        tangent_target = _FREE_SPACE_AD.geometry_tangent(tangents, "target", saved[1])
        tangent_frequency = _ad_frequency_tangent(_FREE_SPACE_AD.tangent(tangents, "frequency"))
        if (
            tangent_frequency == 0.0
            and tangent_source is None
            and tangent_target is None
        ):
            return (None,) * 7
        with disable_functorch():
            out = field_free_space_jvp(
                *(_ad_native_tensor(value) for value in saved),
                frequency_hz=ctx.frequency_value,
                tangent_frequency=tangent_frequency,
                tangent_source=tangent_source,
                tangent_target=tangent_target,
            )
        return direction_tangents(ctx, out)


def field_free_space_ad(
    source: torch.Tensor, target: torch.Tensor, tx_power: torch.Tensor,
    tx_polarization: torch.Tensor, rx_polarization: torch.Tensor, *,
    frequency: torch.Tensor | float, frequency_value: float | None = None,
    direction_live: bool = False,
) -> dict[str, torch.Tensor]:
    """Differentiable :func:`field_free_space` (frequency only in AD).

    ``frequency_value`` optionally carries the precomputed host scalar of
    ``frequency`` (one read per solve at the seam); when not
    supplied it is read here, exactly once per apply.
    """

    if frequency_value is None:
        frequency_value = _ad_frequency_value(frequency)
    values = _FieldFreeSpaceAdFunction.apply(
        source, target, tx_power, tx_polarization, rx_polarization, frequency,
        float(frequency_value), ad_liveness(direction_live, source, target),
    )
    return dict(zip(_FIELD_AD_OUTPUT_FIELDS, values, strict=True))


_REFLECTION_AD = AdInputs(
    "field_reflection_sequence_ad", _REFLECTION_PRIMALS, opaque("frequency", grad="diff"),
    opaque("frequency_value"), opaque("liveness"), opaque("layers"),
)
_REFLECTION_GEOMETRY = ("source", "target", "interaction_positions", "interaction_normals")


class _FieldReflectionSequenceAdFunction(torch.autograd.Function):
    """Fixed-topology differentiable reflection transport.

    Frequency, hit geometry, and per-bounce material scalars except ``mu_r``
    are differentiable; power, polarizations, and ``mu_r`` stay fixed.
    """

    @staticmethod
    def forward(
        source: Any, target: Any, interaction_positions: Any, interaction_normals: Any,
        tx_power: Any, tx_polarization: Any, rx_polarization: Any, eps_r: Any, sigma_e: Any,
        mu_r: Any, gain: Any, thickness: Any, frequency: Any, frequency_value: Any, liveness: Any,
        layers: Any,
    ) -> Any:
        out = _FIELD_REFLECTION_SEQUENCE(
            source, target, interaction_positions, interaction_normals, tx_power, tx_polarization,
            rx_polarization, eps_r, sigma_e, mu_r, gain, thickness,
            *_native_layers(layers, eps_r), frequency_value,
        )
        return tuple(out[name] for name in _FIELD_AD_OUTPUT_FIELDS)

    @staticmethod
    def setup_context(ctx: Any, inputs: Any, output: Any) -> Any:
        ctx.set_materialize_grads(False)
        primals = _REFLECTION_AD.primals(inputs, *_FIELD_REFLECTION_SEQUENCE.names[:12])
        ctx.frequency_value = _REFLECTION_AD.get(inputs, "frequency_value")
        ctx.frequency_meta = frequency_meta(_REFLECTION_AD.get(inputs, "frequency"))
        # Both decided by the wrapper, where forward duals are still visible.
        ctx.geometry_live, ctx.direction_live = _REFLECTION_AD.get(inputs, "liveness")
        ctx.layers = _REFLECTION_AD.get(inputs, "layers")
        save(ctx, *primals)
        mark_dead_outputs(ctx, output)

    @staticmethod
    @_ad_first_order_only
    def backward(
        ctx: Any, grad_field_vector: Any, grad_coefficient: Any, grad_path_field: Any,
        grad_path_gain: Any, grad_path_length: Any, grad_delay: Any, grad_direction: Any,
    ) -> Any:
        _REFLECTION_AD.reject_fixed_grads(ctx)
        need_geometry = _REFLECTION_AD.needs(ctx, *_REFLECTION_GEOMETRY)
        need_eps = _REFLECTION_AD.need(ctx, "eps_r")
        need_sigma = _REFLECTION_AD.need(ctx, "sigma_e")
        need_gain = _REFLECTION_AD.need(ctx, "gain")
        need_thickness = _REFLECTION_AD.need(ctx, "thickness")
        need_frequency = _REFLECTION_AD.need(ctx, "frequency")
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
            return _REFLECTION_AD.none
        out = field_reflection_sequence_backward(
            *ctx.saved_tensors,
            layers=ctx.layers,
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
        return _REFLECTION_AD.output_grads(
            ctx, out, _grad_keys(*_REFLECTION_GEOMETRY, "eps_r", "sigma_e", "gain", "thickness"),
            frequency=grad_frequency,
        )

    @staticmethod
    def jvp(ctx: Any, *tangents: Any) -> Any:
        _REFLECTION_AD.reject_fixed_tangents(tangents)
        saved = ctx.saved_tensors
        eps_shape = tuple(saved[7].shape)
        tangent_source, tangent_target, tangent_positions, tangent_normals = (
            _REFLECTION_AD.geometry_tangent(tangents, name, saved[index])
            for index, name in enumerate(_REFLECTION_GEOMETRY)
        )
        tangent_eps, tangent_sigma, tangent_gain, tangent_thickness = (
            _REFLECTION_AD.checked_tangent(tangents, name, eps_shape)
            for name in ("eps_r", "sigma_e", "gain", "thickness")
        )
        tangent_frequency = _ad_frequency_tangent(_REFLECTION_AD.tangent(tangents, "frequency"))
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
                layers=ctx.layers,
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
    source: torch.Tensor, target: torch.Tensor, interaction_positions: torch.Tensor,
    interaction_normals: torch.Tensor, tx_power: torch.Tensor, tx_polarization: torch.Tensor,
    rx_polarization: torch.Tensor, eps_r: torch.Tensor, sigma_e: torch.Tensor, mu_r: torch.Tensor,
    gain: torch.Tensor, thickness: torch.Tensor, *, layers: ReflectionLayers | None = None,
    frequency: torch.Tensor | float, frequency_value: float | None = None,
    direction_live: bool = False,
) -> dict[str, torch.Tensor]:
    """Differentiable :func:`field_reflection_sequence` (materials + frequency).

    ``frequency_value`` optionally carries the precomputed host scalar of
    ``frequency`` (one read per solve at the seam); when not
    supplied it is read here, exactly once per apply.
    """

    if frequency_value is None:
        frequency_value = _ad_frequency_value(frequency)
    values = _FieldReflectionSequenceAdFunction.apply(
        source, target, interaction_positions, interaction_normals, tx_power, tx_polarization,
        rx_polarization, eps_r, sigma_e, mu_r, gain, thickness, frequency, float(frequency_value),
        ad_liveness(direction_live, source, target, interaction_positions, interaction_normals),
        layers,
    )
    return dict(zip(_FIELD_AD_OUTPUT_FIELDS, values, strict=True))


_TRANSMISSION_AD = AdInputs(
    "field_transmission_sequence_ad", _TRANSMISSION_PRIMALS, opaque("frequency", grad="diff"),
    opaque("frequency_value"),
)
_TRANSMISSION_LAYERS = ("layer_thickness_m", "layer_eps_r", "layer_sigma_e")


class _FieldTransmissionSequenceAdFunction(torch.autograd.Function):
    """Fixed-topology differentiable transmission transport (material and frequency derivatives/AD).

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
        path_valid: Any, source: Any, target: Any, interaction_positions: Any,
        interaction_normals: Any, interaction_material_id: Any, interaction_valid: Any,
        tx_power: Any, tx_polarization: Any, rx_polarization: Any, layer_offset: Any,
        layer_count: Any, layer_thickness_m: Any, layer_eps_r: Any, layer_sigma_e: Any,
        layer_mu_r: Any, frequency: Any, frequency_value: Any,
    ) -> Any:
        out = _FIELD_TRANSMISSION_SEQUENCE(
            path_valid, source, target, interaction_positions, interaction_normals,
            interaction_material_id, interaction_valid, tx_power, tx_polarization,
            rx_polarization, layer_offset, layer_count, layer_thickness_m, layer_eps_r,
            layer_sigma_e, layer_mu_r, frequency_value,
        )
        return tuple(out[name] for name in _FIELD_AD_OUTPUT_FIELDS)

    @staticmethod
    def setup_context(ctx: Any, inputs: Any, output: Any) -> Any:
        ctx.set_materialize_grads(False)
        primals = _TRANSMISSION_AD.primals(inputs, *_FIELD_TRANSMISSION_SEQUENCE.names[:16])
        ctx.frequency_value = _TRANSMISSION_AD.get(inputs, "frequency_value")
        ctx.frequency_meta = frequency_meta(_TRANSMISSION_AD.get(inputs, "frequency"))
        ctx.geometry_live = _ad_geometry_live(
            *(
                _TRANSMISSION_AD.get(inputs, name)
                for name in ("source", "target", "interaction_positions", "interaction_normals")
            )
        )
        save(ctx, *primals)
        # first-order differentiation: RayD owns this family's direction seam, so the arrival
        # direction is a declared non-differentiable output on every route.
        ctx.direction_live = False
        mark_dead_outputs(ctx, output)

    @staticmethod
    @_ad_first_order_only
    def backward(
        ctx: Any, grad_field_vector: Any, grad_coefficient: Any, grad_path_field: Any,
        grad_path_gain: Any, grad_path_length: Any, grad_delay: Any, _grad_direction: Any,
    ) -> Any:
        _TRANSMISSION_AD.reject_fixed_grads(ctx)
        # interaction_positions never enters the straight-path field: its
        # gradient is exactly zero, so it does not drive a launch.
        need_geometry = _TRANSMISSION_AD.needs(ctx, "source", "target", "interaction_normals")
        need_thickness = _TRANSMISSION_AD.need(ctx, "layer_thickness_m")
        need_eps = _TRANSMISSION_AD.need(ctx, "layer_eps_r")
        need_sigma = _TRANSMISSION_AD.need(ctx, "layer_sigma_e")
        need_frequency = _TRANSMISSION_AD.need(ctx, "frequency")
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
            return _TRANSMISSION_AD.none
        out = field_transmission_sequence_backward(
            *ctx.saved_tensors,
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
        return _TRANSMISSION_AD.output_grads(
            ctx, out, _grad_keys("source", "target", "interaction_normals", *_TRANSMISSION_LAYERS),
            frequency=grad_frequency,
        )

    @staticmethod
    def jvp(
        ctx: Any, _t_path_valid: Any, t_source: Any, t_target: Any, t_positions: Any,
        t_normals: Any, _t_material_id: Any, _t_interaction_valid: Any, t_tx_power: Any,
        t_tx_pol: Any, t_rx_pol: Any, _t_layer_offset: Any, _t_layer_count: Any,
        t_layer_thickness: Any, t_layer_eps_r: Any, t_layer_sigma_e: Any, t_layer_mu_r: Any,
        t_frequency: Any, _t_frequency_value: Any,
    ) -> Any:
        tangents = (
            _t_path_valid, t_source, t_target, t_positions, t_normals, _t_material_id,
            _t_interaction_valid, t_tx_power, t_tx_pol, t_rx_pol, _t_layer_offset,
            _t_layer_count, t_layer_thickness, t_layer_eps_r, t_layer_sigma_e, t_layer_mu_r,
            t_frequency, _t_frequency_value,
        )
        _TRANSMISSION_AD.reject_fixed_tangents(tangents)
        saved = ctx.saved_tensors
        layer_shape = tuple(saved[12].shape)
        tangent_source, tangent_target, tangent_positions, tangent_normals = (
            _TRANSMISSION_AD.geometry_tangent(tangents, name, saved[index])
            for index, name in enumerate(
                ("source", "target", "interaction_positions", "interaction_normals"), start=1
            )
        )
        tangent_thickness, tangent_eps, tangent_sigma = (
            _TRANSMISSION_AD.checked_tangent(tangents, name, layer_shape)
            for name in _TRANSMISSION_LAYERS
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
        tangents_out = tuple(out[name] for name in _FIELD_AD_TANGENT_FIELDS)
        if not ctx.geometry_live:
            return (*tangents_out[:4], None, None, None)
        return (*tangents_out, None)


def field_transmission_sequence_ad(
    path_valid: torch.Tensor, source: torch.Tensor, target: torch.Tensor,
    interaction_positions: torch.Tensor, interaction_normals: torch.Tensor,
    interaction_material_id: torch.Tensor, interaction_valid: torch.Tensor, tx_power: torch.Tensor,
    tx_polarization: torch.Tensor, rx_polarization: torch.Tensor, layer_offset: torch.Tensor,
    layer_count: torch.Tensor, layer_thickness_m: torch.Tensor, layer_eps_r: torch.Tensor,
    layer_sigma_e: torch.Tensor, layer_mu_r: torch.Tensor, *, frequency: torch.Tensor | float,
    frequency_value: float | None = None,
) -> dict[str, torch.Tensor]:
    """Differentiable :func:`field_transmission_sequence` (layers + frequency).

    ``frequency_value`` optionally carries the precomputed host scalar of
    ``frequency`` (one read per solve at the seam); when not
    supplied it is read here, exactly once per apply.
    """

    if frequency_value is None:
        frequency_value = _ad_frequency_value(frequency)
    values = _FieldTransmissionSequenceAdFunction.apply(
        path_valid, source, target, interaction_positions, interaction_normals,
        interaction_material_id, interaction_valid, tx_power, tx_polarization, rx_polarization,
        layer_offset, layer_count, layer_thickness_m, layer_eps_r, layer_sigma_e, layer_mu_r,
        frequency, float(frequency_value),
    )
    return dict(zip(_FIELD_AD_OUTPUT_FIELDS, values, strict=True))


_WEDGE_AD = AdInputs(
    "field_diffraction_wedge_ad", _WEDGE_PRIMALS, opaque("frequency", grad="diff"),
    _WEDGE_VERTICES, opaque("frequency_value"), _FACE_LAYERS,
)
_WEDGE_VERTEX_NAMES = ("vertex_v0", "vertex_v1", "vertex_opp0", "vertex_opp1")


class _FieldDiffractionWedgeAdFunction(torch.autograd.Function):
    """Fixed-topology differentiable UTD wedge field (diffraction AD).

    Differentiable inputs: both faces' eps_r / sigma_e / gain, frequency,
    endpoints, and optional per-row winner vertices v0/v1 plus each face's
    opposite vertex. The kernel rebuilds edge tables so mesh-vertex gradients
    reach edge geometry. Frozen edge tables, valid masks, mu_r and tx_power
    stay fixed and reject gradients. The stationary edge point is re-solved
    inside the kernel, preserving its endpoint and vertex gradient motion.
    """

    @staticmethod
    def forward(
        valid: Any, source: Any, target: Any, edge_position: Any, edge_direction: Any,
        edge_t_min: Any, edge_t_max: Any, edge_n0: Any, edge_n1: Any, exterior_angle: Any,
        face0_valid: Any, face0_eps_r: Any, face0_sigma_e: Any, face0_mu_r: Any, face0_gain: Any,
        face1_valid: Any, face1_eps_r: Any, face1_sigma_e: Any, face1_mu_r: Any, face1_gain: Any,
        tx_power: Any, frequency: Any, vertex_v0: Any, vertex_v1: Any, vertex_opp0: Any,
        vertex_opp1: Any, edge_boundary: Any, frequency_value: Any, face_layers: Any,
    ) -> Any:
        out = _FIELD_DIFFRACTION_WEDGE(
            valid, source, target, edge_position, edge_direction, edge_t_min, edge_t_max,
            edge_n0, edge_n1, exterior_angle, face0_valid, face0_eps_r, face0_sigma_e,
            face0_mu_r, face0_gain, face1_valid, face1_eps_r, face1_sigma_e, face1_mu_r,
            face1_gain, tx_power, frequency_value, vertex_v0, vertex_v1, vertex_opp0,
            vertex_opp1, edge_boundary,
            # ISB boundary taper, D member. Always 0.0: taper + AD is
            # refused by the deterministic/path pipelines (gate 3, C1 clearance
            # companion pending), so the differentiable twin never tapers. The
            # argument is threaded for lockstep completeness of the guarded path.
            0.0, face_layers,
        )
        return tuple(out[name] for name in _WEDGE_OUTPUT_FIELDS)

    @staticmethod
    def setup_context(ctx: Any, inputs: Any, output: Any) -> Any:
        ctx.set_materialize_grads(False)
        primals = _WEDGE_AD.primals(inputs, *_FIELD_DIFFRACTION_WEDGE.names[:21])
        vertex_primals = tuple(
            torch.autograd.forward_ad.unpack_dual(value).primal
            if isinstance(value, torch.Tensor)
            else value
            for value in (_WEDGE_AD.get(inputs, arg.name) for arg in _WEDGE_VERTICES)
        )
        ctx.has_vertices = vertex_primals[0] is not None
        ctx.frequency_value = _WEDGE_AD.get(inputs, "frequency_value")
        ctx.face_layers = _WEDGE_AD.get(inputs, "face_layers")
        ctx.frequency_meta = frequency_meta(_WEDGE_AD.get(inputs, "frequency"))
        ctx.geometry_live = _ad_geometry_live(
            _WEDGE_AD.get(inputs, "source"), _WEDGE_AD.get(inputs, "target")
        )
        save(ctx, *primals, *(value for value in vertex_primals if value is not None))

    @staticmethod
    def _unpack_saved(ctx: Any) -> tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor | None, ...]]:
        saved = ctx.saved_tensors
        primals = saved[:21]
        vertices = saved[21:26] if ctx.has_vertices else (None,) * 5
        return primals, vertices

    @staticmethod
    @_ad_first_order_only
    def backward(ctx: Any, grad_field_vector: Any, grad_direction: Any) -> Any:
        _WEDGE_AD.reject_fixed_grads(ctx)
        need_geometry = _WEDGE_AD.needs(ctx, "source", "target")
        need_material = _WEDGE_AD.needs(ctx, *_WEDGE_MATERIALS)
        need_frequency = _WEDGE_AD.need(ctx, "frequency")
        need_vertices = _WEDGE_AD.needs(ctx, *_WEDGE_VERTEX_NAMES)
        if not (need_geometry or need_material or need_frequency or need_vertices) or (
            grad_field_vector is None and grad_direction is None
        ):
            return _WEDGE_AD.none
        primals, vertices = _FieldDiffractionWedgeAdFunction._unpack_saved(ctx)
        out = _FIELD_DIFFRACTION_WEDGE_BACKWARD(
            *primals, ctx.frequency_value, *vertices, grad_field_vector, grad_direction,
            need_material, need_frequency, need_geometry, need_vertices,
            # the boundary taper width; always 0.0 (taper + AD is guarded off).
            0.0, ctx.face_layers,
        )
        grad_frequency = (
            _ad_frequency_grad(out["grad_frequency"], ctx.frequency_meta)
            if need_frequency
            else None
        )
        return _WEDGE_AD.output_grads(
            ctx, out, _grad_keys("source", "target", *_WEDGE_MATERIALS, *_WEDGE_VERTEX_NAMES),
            frequency=grad_frequency,
        )

    @staticmethod
    def jvp(ctx: Any, *tangents: Any) -> Any:
        _WEDGE_AD.reject_fixed_tangents(tangents)
        primals, vertices = _FieldDiffractionWedgeAdFunction._unpack_saved(ctx)
        scalar_shape = tuple(primals[11].shape)
        tangent_source = _WEDGE_AD.geometry_tangent(tangents, "source", primals[1])
        tangent_target = _WEDGE_AD.geometry_tangent(tangents, "target", primals[2])
        material_tangents = tuple(
            _WEDGE_AD.checked_tangent(tangents, name, scalar_shape) for name in _WEDGE_MATERIALS
        )
        vertex_tangents = []
        for name in _WEDGE_VERTEX_NAMES:
            index = _WEDGE_AD.index(name)
            tangent = tangents[index] if index < len(tangents) else None
            vertex_tangents.append(
                _ad_native_tangent_or_none(
                    tangent if isinstance(tangent, torch.Tensor) else None
                )
            )
        tangent_frequency = _ad_frequency_tangent(_WEDGE_AD.tangent(tangents, "frequency"))
        if (
            tangent_source is None
            and tangent_target is None
            and tangent_frequency == 0.0
            and all(value is None for value in material_tangents)
            and all(value is None for value in vertex_tangents)
        ):
            return (None, None)
        with disable_functorch():
            out = _FIELD_DIFFRACTION_WEDGE_JVP(
                *(_ad_native_tensor(value) for value in primals),
                ctx.frequency_value,
                *(
                    _ad_native_tensor(value) if isinstance(value, torch.Tensor)
                    else value
                    for value in vertices
                ),
                tangent_source,
                tangent_target,
                *material_tangents,
                tangent_frequency,
                *vertex_tangents,
                # the boundary taper width; always 0.0 (taper + AD is guarded off).
                0.0, ctx.face_layers,
            )
        return (out["tangent_field_vector"], out["tangent_direction"])


def field_diffraction_wedge_ad(
    valid: torch.Tensor, source: torch.Tensor, target: torch.Tensor, edge_position: torch.Tensor,
    edge_direction: torch.Tensor, edge_t_min: torch.Tensor, edge_t_max: torch.Tensor,
    edge_n0: torch.Tensor, edge_n1: torch.Tensor, exterior_angle: torch.Tensor,
    face0_valid: torch.Tensor, face0_eps_r: torch.Tensor, face0_sigma_e: torch.Tensor,
    face0_mu_r: torch.Tensor, face0_gain: torch.Tensor, face1_valid: torch.Tensor,
    face1_eps_r: torch.Tensor, face1_sigma_e: torch.Tensor, face1_mu_r: torch.Tensor,
    face1_gain: torch.Tensor, tx_power: torch.Tensor, *, frequency: torch.Tensor | float,
    frequency_value: float | None = None, vertices: tuple[torch.Tensor, ...] | None = None,
    face_layers: tuple[torch.Tensor, ...] | None = None,
) -> dict[str, torch.Tensor]:
    """Differentiable :func:`field_diffraction_wedge`.

    ``vertices`` optionally supplies the winner edge vertices as
    ``(v0, v1, opp0, opp1, edge_boundary)`` per row; the kernel then rebuilds
    the edge tables from them so mesh-vertex gradients exist (AD capability matrix cell mesh-vertex x diffraction). ``frequency_value`` optionally carries
    the precomputed host scalar of ``frequency`` (one read per solve at the
    seam); when not supplied it is read here, exactly once per
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
        valid, source, target, edge_position, edge_direction, edge_t_min, edge_t_max, edge_n0,
        edge_n1, exterior_angle, face0_valid, face0_eps_r, face0_sigma_e, face0_mu_r, face0_gain,
        face1_valid, face1_eps_r, face1_sigma_e, face1_mu_r, face1_gain, tx_power, frequency,
        *vertex_args, float(frequency_value), face_layers,
    )
    return dict(zip(_WEDGE_OUTPUT_FIELDS, values, strict=True))


_CHAIN_AD = AdInputs(
    "field_coupled_chain_ad", _CHAIN_PRIMALS, opaque("frequency", grad="diff"),
    opaque("frequency_value"),
)


class _FieldCoupledChainAdFunction(torch.autograd.Function):
    """Fixed-topology differentiable interaction-chain transport.

    Differentiable inputs: the endpoints (the interaction points follow them through the native
    image re-solve of the frozen winner), eps_r / sigma_e / gain / thickness of every reflection
    slot and both wedge faces, and frequency. The discovered interaction points, the normals and
    edge tables, mu_r, tx_power and the polarizations stay fixed.
    """

    @staticmethod
    def forward(
        source: Any, target: Any, positions: Any, normals: Any, component_id: Any, depth: Any,
        edge_direction: Any, edge_n0: Any, edge_n1: Any, exterior_angle: Any, edge_line_min: Any,
        edge_line_max: Any, tx_power: Any, tx_polarization: Any, rx_polarization: Any, eps_r: Any,
        sigma_e: Any, mu_r: Any, gain: Any, thickness: Any, frequency: Any, frequency_value: Any,
    ) -> Any:
        out = _FIELD_COUPLED_CHAIN(
            source, target, positions, normals, component_id, depth, edge_direction, edge_n0,
            edge_n1, exterior_angle, edge_line_min, edge_line_max, tx_power, tx_polarization,
            rx_polarization, eps_r, sigma_e, mu_r, gain, thickness,
            *_native_layers(None, eps_r), frequency_value,
        )
        return tuple(out[name] for name in _COUPLED_OUTPUT_FIELDS)

    @staticmethod
    def setup_context(ctx: Any, inputs: Any, output: Any) -> Any:
        ctx.set_materialize_grads(False)
        primals = _CHAIN_AD.primals(inputs, *(arg.name for arg in _CHAIN_PRIMALS))
        ctx.frequency_value = _CHAIN_AD.get(inputs, "frequency_value")
        ctx.frequency_meta = frequency_meta(_CHAIN_AD.get(inputs, "frequency"))
        save(ctx, *primals)
        ctx.mark_non_differentiable(output[4])

    @staticmethod
    def _native_primals(ctx: Any) -> tuple[torch.Tensor, ...]:
        saved = tuple(_ad_native_tensor(value) for value in ctx.saved_tensors)
        return (*saved, *_native_layers(None, saved[15]), ctx.frequency_value)

    @staticmethod
    @_ad_first_order_only
    def backward(
        ctx: Any, grad_field_vector: Any, grad_coefficient: Any, grad_path_field: Any,
        grad_path_gain: Any, _grad_direction: Any,
    ) -> Any:
        _CHAIN_AD.reject_fixed_grads(ctx)
        need_geometry = _CHAIN_AD.needs(ctx, "source", "target")
        need = {name: _CHAIN_AD.need(ctx, name) for name in (*_CHAIN_MATERIAL_GRADS, "frequency")}
        grads = (grad_field_vector, grad_coefficient, grad_path_field, grad_path_gain)
        if not (need_geometry or any(need.values())) or all(value is None for value in grads):
            return _CHAIN_AD.none
        out = _FIELD_COUPLED_CHAIN_BACKWARD(
            *_FieldCoupledChainAdFunction._native_primals(ctx), *grads,
            *need.values(), need_geometry,
        )
        grad_frequency = (
            _ad_frequency_grad(out["grad_frequency"], ctx.frequency_meta)
            if need["frequency"]
            else None
        )
        return _CHAIN_AD.output_grads(
            ctx, out, _grad_keys("source", "target", *_CHAIN_MATERIAL_GRADS),
            frequency=grad_frequency,
        )

    @staticmethod
    def jvp(ctx: Any, *tangents: Any) -> Any:
        _CHAIN_AD.reject_fixed_tangents(tangents)
        saved = ctx.saved_tensors
        tangent_source = _CHAIN_AD.geometry_tangent(tangents, "source", saved[0])
        tangent_target = _CHAIN_AD.geometry_tangent(tangents, "target", saved[1])
        material_shape = tuple(saved[_CHAIN_AD.index("eps_r")].shape)
        material_tangents = tuple(
            _CHAIN_AD.checked_tangent(tangents, name, material_shape)
            for name in _CHAIN_MATERIAL_GRADS
        )
        tangent_frequency = _ad_frequency_tangent(_CHAIN_AD.tangent(tangents, "frequency"))
        if (
            tangent_source is None
            and tangent_target is None
            and all(value is None for value in material_tangents)
            and tangent_frequency == 0.0
        ):
            return (None,) * 5
        with disable_functorch():
            out = _FIELD_COUPLED_CHAIN_JVP(
                *_FieldCoupledChainAdFunction._native_primals(ctx),
                tangent_source,
                tangent_target,
                *material_tangents,
                tangent_frequency,
            )
        return (
            out["tangent_field_vector"],
            out["tangent_coefficient"],
            out["tangent_path_field"],
            out["tangent_path_gain"],
            None,
        )


def field_coupled_chain_ad(
    source: torch.Tensor, target: torch.Tensor, positions: torch.Tensor, normals: torch.Tensor,
    component_id: torch.Tensor, depth: torch.Tensor, edge_direction: torch.Tensor,
    edge_n0: torch.Tensor, edge_n1: torch.Tensor, exterior_angle: torch.Tensor,
    edge_line_min: torch.Tensor, edge_line_max: torch.Tensor, tx_power: torch.Tensor,
    tx_polarization: torch.Tensor, rx_polarization: torch.Tensor,
    material: tuple[torch.Tensor, ...], *, frequency: torch.Tensor | float,
    frequency_value: float | None = None,
) -> dict[str, torch.Tensor]:
    """Differentiable :func:`field_coupled_chain` (endpoints, slot materials and frequency).

    ``frequency_value`` optionally carries the precomputed host scalar of ``frequency`` (one read
    per solve at the seam); when not supplied it is read here, exactly once per apply.
    """

    _validate_chain_inputs(
        source, target, positions, normals, component_id, depth, edge_direction, edge_n0,
        edge_n1, exterior_angle, edge_line_min, edge_line_max, tx_power, tx_polarization,
        rx_polarization, material,
    )
    if frequency_value is None:
        frequency_value = _ad_frequency_value(frequency)
    values = _FieldCoupledChainAdFunction.apply(
        source, target, positions, normals, component_id, depth, edge_direction, edge_n0, edge_n1,
        exterior_angle, edge_line_min, edge_line_max, tx_power, tx_polarization, rx_polarization,
        *material, frequency, float(frequency_value),
    )
    return dict(zip(_COUPLED_OUTPUT_FIELDS, values, strict=True))


_DD_AD = AdInputs(
    "field_coupled_dd_ad", _DD_PRIMALS, opaque("frequency", grad="diff"), opaque("frequency_value"),
    _DD_BOUNDS,
)
# The four face names of one material family, in the native (N, 4) column order.
_DD_FAMILIES = {
    family: tuple(face.format(family) for face in _DD_FACES)
    for family in ("eps_r", "sigma_e", "gain", "thickness")
}


class _FieldCoupledDdAdFunction(torch.autograd.Function):
    """Fixed-topology differentiable coupled double-diffraction transport (coupled double diffraction).

    Twin of :class:`_FieldCoupledChainAdFunction` for cid-7 (TX -> e1 -> e2 -> RX)
    rows. Differentiable inputs: eps_r / sigma_e / gain / thickness for the four
    wedge faces (16 scalars per path), frequency, and the tx/rx endpoints
    (source, target), whose gradients flow through the native per-leg
    re-anchoring. The frozen discovery seeds Q1/Q2 (edge1_position /
    edge2_position), the edge axes/normals/exterior angles, the edge bounds,
    mu_r, tx_power and the polarizations stay fixed; requesting their gradient
    fails loudly. Mesh-vertex gradients are refused one layer up (evaluation.py
    coupled block, coupled double diffraction).
    """

    @staticmethod
    def forward(
        source: Any, target: Any, edge1_position: Any, edge1_direction: Any, edge1_n0: Any,
        edge1_n1: Any, edge1_exterior: Any, edge2_position: Any, edge2_direction: Any,
        edge2_n0: Any, edge2_n1: Any, edge2_exterior: Any, tx_power: Any, tx_polarization: Any,
        rx_polarization: Any, w1a_eps_r: Any, w1a_sigma_e: Any, w1a_mu_r: Any, w1a_gain: Any,
        w1a_thickness: Any, w1b_eps_r: Any, w1b_sigma_e: Any, w1b_mu_r: Any, w1b_gain: Any,
        w1b_thickness: Any, w2a_eps_r: Any, w2a_sigma_e: Any, w2a_mu_r: Any, w2a_gain: Any,
        w2a_thickness: Any, w2b_eps_r: Any, w2b_sigma_e: Any, w2b_mu_r: Any, w2b_gain: Any,
        w2b_thickness: Any, frequency: Any, frequency_value: Any, edge1_line_min: Any,
        edge1_line_max: Any, edge2_line_min: Any, edge2_line_max: Any,
    ) -> Any:
        out = _FIELD_COUPLED_DD(
            source, target, edge1_position, edge1_direction, edge1_n0, edge1_n1, edge1_exterior,
            edge2_position, edge2_direction, edge2_n0, edge2_n1, edge2_exterior, tx_power,
            tx_polarization, rx_polarization, w1a_eps_r, w1a_sigma_e, w1a_mu_r, w1a_gain,
            w1a_thickness, w1b_eps_r, w1b_sigma_e, w1b_mu_r, w1b_gain, w1b_thickness, w2a_eps_r,
            w2a_sigma_e, w2a_mu_r, w2a_gain, w2a_thickness, w2b_eps_r, w2b_sigma_e, w2b_mu_r,
            w2b_gain, w2b_thickness, edge1_line_min, edge1_line_max, edge2_line_min,
            edge2_line_max, frequency_value,
        )
        return tuple(out[name] for name in _COUPLED_OUTPUT_FIELDS)

    @staticmethod
    def setup_context(ctx: Any, inputs: Any, output: Any) -> Any:
        ctx.set_materialize_grads(False)
        # The native backward/jvp take the 35 primal field tensors (source ..
        # wedge2 face1 thickness) followed by the four frozen edge bounds, then
        # the host frequency scalar. Q1/Q2/bounds are non-differentiable (coupled double diffraction
        # edge bounds) but saved so the companions forward them in the primal position.
        primals = _DD_AD.primals(inputs, *(arg.name for arg in (*_DD_PRIMALS, *_DD_BOUNDS)))
        ctx.frequency_value = _DD_AD.get(inputs, "frequency_value")
        ctx.frequency_meta = frequency_meta(_DD_AD.get(inputs, "frequency"))
        save(ctx, *primals)
        ctx.mark_non_differentiable(output[4])

    @staticmethod
    @_ad_first_order_only
    def backward(
        ctx: Any, grad_field_vector: Any, grad_coefficient: Any, grad_path_field: Any,
        grad_path_gain: Any, _grad_direction: Any,
    ) -> Any:
        _DD_AD.reject_fixed_grads(ctx)
        needs = {family: _DD_AD.needs(ctx, *faces) for family, faces in _DD_FAMILIES.items()}
        need_frequency = _DD_AD.need(ctx, "frequency")
        need_geometry = _DD_AD.needs(ctx, "source", "target")
        grads = (grad_field_vector, grad_coefficient, grad_path_field, grad_path_gain)
        if not (any(needs.values()) or need_frequency or need_geometry) or all(
            value is None for value in grads
        ):
            return _DD_AD.none
        out = _FIELD_COUPLED_DD_BACKWARD(
            *ctx.saved_tensors, ctx.frequency_value, *grads, *needs.values(), need_frequency,
            need_geometry,
        )
        grad_frequency = (
            _ad_frequency_grad(out["grad_frequency"], ctx.frequency_meta)
            if need_frequency
            else None
        )
        # Material grad columns: wedge1 face0, wedge1 face1, wedge2 face0,
        # wedge2 face1 (mirrors the native (N, 4) slot layout).
        columns = {
            name: out[f"grad_{family}"][:, column]
            for family, faces in _DD_FAMILIES.items()
            for column, name in enumerate(faces)
            if _DD_AD.need(ctx, name)
        }
        return _DD_AD.output_grads(
            ctx, out, _GEOMETRY_GRAD_KEYS, **columns, frequency=grad_frequency
        )

    @staticmethod
    def jvp(ctx: Any, *tangents: Any) -> Any:
        _DD_AD.reject_fixed_tangents(tangents)
        saved = ctx.saved_tensors
        scalar_shape = tuple(saved[15].shape)
        tangent_source = _DD_AD.geometry_tangent(tangents, "source", saved[0])
        tangent_target = _DD_AD.geometry_tangent(tangents, "target", saved[1])

        def material_pack(family: str) -> torch.Tensor | None:
            columns = tuple(
                _DD_AD.checked_tangent(tangents, name, scalar_shape, label=family)
                for name in _DD_FAMILIES[family]
            )
            if all(column is None for column in columns):
                return None
            zero = torch.zeros(
                scalar_shape, device=saved[15].device, dtype=torch.float32
            )
            return torch.stack(
                [zero if column is None else column for column in columns], dim=1
            )

        tangent_eps, tangent_sigma, tangent_gain, tangent_thickness = (
            material_pack(family) for family in _DD_FAMILIES
        )
        tangent_frequency = _ad_frequency_tangent(_DD_AD.tangent(tangents, "frequency"))
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
            out = _FIELD_COUPLED_DD_JVP(
                *(_ad_native_tensor(value) for value in saved),
                ctx.frequency_value,
                tangent_source,
                tangent_target,
                tangent_eps,
                tangent_sigma,
                tangent_gain,
                tangent_thickness,
                tangent_frequency,
            )
        return (
            out["tangent_field_vector"],
            out["tangent_coefficient"],
            out["tangent_path_field"],
            out["tangent_path_gain"],
            None,
        )


def field_coupled_dd_ad(
    source: torch.Tensor, target: torch.Tensor, edge1_position: torch.Tensor,
    edge1_direction: torch.Tensor, edge1_n0: torch.Tensor, edge1_n1: torch.Tensor,
    edge1_exterior: torch.Tensor, edge2_position: torch.Tensor, edge2_direction: torch.Tensor,
    edge2_n0: torch.Tensor, edge2_n1: torch.Tensor, edge2_exterior: torch.Tensor,
    tx_power: torch.Tensor, tx_polarization: torch.Tensor, rx_polarization: torch.Tensor,
    wedge1_material0: tuple[torch.Tensor, ...], wedge1_material1: tuple[torch.Tensor, ...],
    wedge2_material0: tuple[torch.Tensor, ...], wedge2_material1: tuple[torch.Tensor, ...],
    edge1_line_min: torch.Tensor, edge1_line_max: torch.Tensor, edge2_line_min: torch.Tensor,
    edge2_line_max: torch.Tensor, *, frequency: torch.Tensor | float,
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
        source, target, edge1_position, edge1_direction, edge1_n0, edge1_n1, edge1_exterior,
        edge2_position, edge2_direction, edge2_n0, edge2_n1, edge2_exterior, tx_power,
        tx_polarization, rx_polarization, *wedge1_material0, *wedge1_material1,
        *wedge2_material0, *wedge2_material1, frequency, float(frequency_value), edge1_line_min,
        edge1_line_max, edge2_line_min, edge2_line_max,
    )
    return dict(zip(_COUPLED_OUTPUT_FIELDS, values, strict=True))


_PROJECT_AD = AdInputs("field_project_complex3_ad", _PROJECT_PRIMALS)


class _FieldProjectComplex3AdFunction(torch.autograd.Function):
    """Differentiable receiver projection on a frozen polarization basis."""

    @staticmethod
    def forward(field_vector: Any, direction: Any, rx_polarization: Any) -> Any:
        out = _FIELD_PROJECT_COMPLEX3(field_vector, direction, rx_polarization)
        return (out["coefficient"], out["path_gain"])

    @staticmethod
    def setup_context(ctx: Any, inputs: Any, output: Any) -> Any:
        ctx.set_materialize_grads(False)
        save(ctx, *_PROJECT_AD.primals(inputs, *_PROJECT_AD.names))

    @staticmethod
    @_ad_first_order_only
    def backward(ctx: Any, grad_coefficient: Any, grad_path_gain: Any) -> Any:
        _PROJECT_AD.reject_fixed_grads(ctx)
        need_field = _PROJECT_AD.need(ctx, "field_vector")
        need_direction = _PROJECT_AD.need(ctx, "direction")
        if not (need_field or need_direction) or (
            grad_coefficient is None and grad_path_gain is None
        ):
            return _PROJECT_AD.none
        out = _FIELD_PROJECT_COMPLEX3_BACKWARD(
            *ctx.saved_tensors, grad_coefficient, grad_path_gain, need_field, need_direction
        )
        return _PROJECT_AD.output_grads(ctx, out, _grad_keys("field_vector", "direction"))

    @staticmethod
    def jvp(ctx: Any, *tangents: Any) -> Any:
        _PROJECT_AD.reject_fixed_tangents(tangents)
        saved = ctx.saved_tensors
        tangent_field = _ad_native_tangent_or_none(_PROJECT_AD.tangent(tangents, "field_vector"))
        tangent_direction = _PROJECT_AD.geometry_tangent(tangents, "direction", saved[1])
        if tangent_field is None and tangent_direction is None:
            return (None, None)
        with disable_functorch():
            out = _FIELD_PROJECT_COMPLEX3_JVP(
                *(_ad_native_tensor(value) for value in saved),
                tangent_field,
                tangent_direction,
            )
        return (out["tangent_coefficient"], out["tangent_path_gain"])


def field_project_complex3_ad(
    field_vector: torch.Tensor, direction: torch.Tensor, rx_polarization: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Differentiable :func:`field_project_complex3` (field vector + direction)."""

    coefficient, path_gain = _FieldProjectComplex3AdFunction.apply(
        field_vector, direction, rx_polarization
    )
    return {"coefficient": coefficient, "path_gain": path_gain}


_ROUGH_AD = AdInputs(
    "field_rough_reflection_scale_ad", _ROUGH_PRIMALS, opaque("frequency", grad="diff"),
    opaque("frequency_value"), opaque("geometry_live"),
)
_ROUGH_FIELD = ("field_vector", "coefficient", "path_field", "path_gain")
_ROUGH_GEOMETRY = ("positions", "normals", "source")


class _FieldRoughReflectionScaleAdFunction(torch.autograd.Function):
    """Fixed-topology differentiable rough-surface C_r scale (rough-surface scattering).

    Differentiable inputs: the four reflection field outputs (field_vector,
    coefficient, path_field, path_gain), frequency, and the hit geometry
    (positions, normals, source). sigma_b, rough_b and the realization
    ``replaced`` mask are fixed; requesting a sigma_b gradient fails loudly.
    Positions/normals/source only carry a gradient when the fixed-winner
    geometry AD path is live (matching the previous Torch factor's reach).
    """

    @staticmethod
    def forward(
        field_vector: Any, coefficient: Any, path_field: Any, path_gain: Any, positions: Any,
        normals: Any, source: Any, sigma_b: Any, rough_b: Any, replaced: Any, frequency: Any,
        frequency_value: Any, geometry_live: Any,
    ) -> Any:
        out = field_rough_reflection_scale(
            field_vector, coefficient, path_field, path_gain, positions, normals, source, sigma_b,
            rough_b, replaced, frequency_hz=frequency_value,
        )
        return tuple(out[name] for name in _ROUGH_SCALE_OUTPUT_FIELDS)

    @staticmethod
    def setup_context(ctx: Any, inputs: Any, output: Any) -> Any:
        ctx.set_materialize_grads(False)
        primals = _ROUGH_AD.primals(inputs, *_FIELD_ROUGH_REFLECTION_SCALE.names[:10])
        ctx.frequency_value = _ROUGH_AD.get(inputs, "frequency_value")
        ctx.frequency_meta = frequency_meta(_ROUGH_AD.get(inputs, "frequency"))
        # Computed by the wrapper, where forward duals are still visible.
        ctx.geometry_live = _ROUGH_AD.get(inputs, "geometry_live")
        save(ctx, *primals)

    @staticmethod
    @_ad_first_order_only
    def backward(
        ctx: Any, grad_field_vector: Any, grad_coefficient: Any, grad_path_field: Any,
        grad_path_gain: Any,
    ) -> Any:
        _ROUGH_AD.reject_fixed_grads(ctx)
        need_field = _ROUGH_AD.needs(ctx, *_ROUGH_FIELD)
        need_geometry = _ROUGH_AD.needs(ctx, *_ROUGH_GEOMETRY)
        need_frequency = _ROUGH_AD.need(ctx, "frequency")
        grads = (
            grad_field_vector,
            grad_coefficient,
            grad_path_field,
            grad_path_gain,
        )
        if not (need_field or need_geometry or need_frequency) or all(
            value is None for value in grads
        ):
            return _ROUGH_AD.none
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
        return _ROUGH_AD.output_grads(
            ctx, out, _grad_keys(*_ROUGH_FIELD, *_ROUGH_GEOMETRY), frequency=grad_frequency
        )

    @staticmethod
    def jvp(
        ctx: Any, t_field_vector: Any, t_coefficient: Any, t_path_field: Any, t_path_gain: Any,
        t_positions: Any, t_normals: Any, t_source: Any, t_sigma_b: Any, _t_rough_b: Any,
        _t_replaced: Any, t_frequency: Any, _t_frequency_value: Any, _t_geometry_live: Any,
    ) -> Any:
        tangents = (
            t_field_vector, t_coefficient, t_path_field, t_path_gain, t_positions, t_normals,
            t_source, t_sigma_b, _t_rough_b, _t_replaced, t_frequency, _t_frequency_value,
            _t_geometry_live,
        )
        _ROUGH_AD.reject_fixed_tangents(tangents)
        saved = ctx.saved_tensors
        tangent_field, tangent_coef, tangent_pf, tangent_pg = (
            _ad_native_tangent_or_none(_ROUGH_AD.tangent(tangents, name)) for name in _ROUGH_FIELD
        )
        tangent_positions, tangent_normals, tangent_source = (
            _ROUGH_AD.geometry_tangent(tangents, name, saved[_ROUGH_AD.index(name)])
            for name in _ROUGH_GEOMETRY
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
    field_vector: torch.Tensor, coefficient: torch.Tensor, path_field: torch.Tensor,
    path_gain: torch.Tensor, positions: torch.Tensor, normals: torch.Tensor, source: torch.Tensor,
    sigma_b: torch.Tensor, rough_b: torch.Tensor, replaced: torch.Tensor, *,
    frequency: torch.Tensor | float, frequency_value: float | None = None,
) -> dict[str, torch.Tensor]:
    """Differentiable :func:`field_rough_reflection_scale` (frequency + geometry).

    ``frequency_value`` optionally carries the precomputed host scalar of
    ``frequency`` (one read per solve at the seam); when not supplied
    it is read here, exactly once per apply.
    """

    if frequency_value is None:
        frequency_value = _ad_frequency_value(frequency)
    values = _FieldRoughReflectionScaleAdFunction.apply(
        field_vector, coefficient, path_field, path_gain, positions, normals, source, sigma_b,
        rough_b, replaced, frequency, float(frequency_value),
        _ad_geometry_live(positions, normals, source),
    )
    return dict(zip(_ROUGH_SCALE_OUTPUT_FIELDS, values, strict=True))


def _validate_source_amplitude_inputs(
    field_vector: torch.Tensor, tx_power: torch.Tensor, *, field_name: str,
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
    out: object, name: str, count: int, *, operation: str,
) -> dict[str, torch.Tensor]:
    if not isinstance(out, dict) or set(out) != {name}:
        raise TypeError(f"_channel.{operation} returned invalid fields")
    validate_cuda_tensor(name, out[name], dtype=torch.complex64, ndim=2)
    if tuple(out[name].shape) != (count, 3):
        raise ValueError(f"{operation} returned an invalid shape")
    return out


def field_source_amplitude_scale(
    field_vector: torch.Tensor, tx_power: torch.Tensor,
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
    out = _FIELD_SOURCE_AMPLITUDE_SCALE(field_vector, tx_power)
    return _validate_source_amplitude_result(
        out, "path_field_vector", count, operation="field_source_amplitude_scale"
    )


def field_source_amplitude_scale_backward(
    tx_power: torch.Tensor, grad_path_field_vector: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """VJP of :func:`field_source_amplitude_scale`; ``tx_power`` is frozen."""

    count = _validate_source_amplitude_inputs(
        grad_path_field_vector, tx_power, field_name="grad_path_field_vector"
    )
    out = _FIELD_SOURCE_AMPLITUDE_SCALE_BACKWARD(tx_power, grad_path_field_vector)
    return _validate_source_amplitude_result(
        out,
        "grad_field_vector",
        count,
        operation="field_source_amplitude_scale_backward",
    )


def field_source_amplitude_scale_jvp(
    tx_power: torch.Tensor, tangent_field_vector: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """JVP of :func:`field_source_amplitude_scale`; ``tx_power`` is frozen."""

    count = _validate_source_amplitude_inputs(
        tangent_field_vector, tx_power, field_name="tangent_field_vector"
    )
    out = _FIELD_SOURCE_AMPLITUDE_SCALE_JVP(tx_power, tangent_field_vector)
    return _validate_source_amplitude_result(
        out,
        "tangent_path_field_vector",
        count,
        operation="field_source_amplitude_scale_jvp",
    )


_SOURCE_AMPLITUDE_AD = AdInputs(
    "field_source_amplitude_scale_ad", opaque("field_vector", grad="diff"),
    opaque("tx_power", grad="fixed"),
)


class _FieldSourceAmplitudeScaleAdFunction(torch.autograd.Function):
    """Differentiable ``field_vector * sqrt(max(tx_power, 0))``."""

    @staticmethod
    def forward(field_vector: Any, tx_power: Any) -> Any:
        out = _FIELD_SOURCE_AMPLITUDE_SCALE(field_vector, tx_power)
        return out["path_field_vector"]

    @staticmethod
    def setup_context(ctx: Any, inputs: Any, output: Any) -> Any:
        ctx.set_materialize_grads(False)
        save(ctx, *_SOURCE_AMPLITUDE_AD.primals(inputs, "tx_power"))

    @staticmethod
    @_ad_first_order_only
    def backward(ctx: Any, grad_path_field_vector: Any) -> Any:
        _SOURCE_AMPLITUDE_AD.reject_fixed_grads(ctx)
        if not _SOURCE_AMPLITUDE_AD.need(ctx, "field_vector") or grad_path_field_vector is None:
            return _SOURCE_AMPLITUDE_AD.none
        (tx_power,) = ctx.saved_tensors
        out = _FIELD_SOURCE_AMPLITUDE_SCALE_BACKWARD(tx_power, grad_path_field_vector)
        return _SOURCE_AMPLITUDE_AD.place(field_vector=out["grad_field_vector"])

    @staticmethod
    def jvp(ctx: Any, *tangents: Any) -> Any:
        _SOURCE_AMPLITUDE_AD.reject_fixed_tangents(tangents)
        tangent = _ad_native_tangent_or_none(_SOURCE_AMPLITUDE_AD.tangent(tangents, "field_vector"))
        if tangent is None:
            return None
        (tx_power,) = ctx.saved_tensors
        with disable_functorch():
            out = _FIELD_SOURCE_AMPLITUDE_SCALE_JVP(_ad_native_tensor(tx_power), tangent)
        return out["tangent_path_field_vector"]


def field_source_amplitude_scale_ad(
    field_vector: torch.Tensor, tx_power: torch.Tensor,
) -> torch.Tensor:
    """Differentiable :func:`field_source_amplitude_scale` (field vector only)."""

    return _FieldSourceAmplitudeScaleAdFunction.apply(field_vector, tx_power)


def deterministic_pack_complex(field_real: torch.Tensor, field_imag: torch.Tensor) -> torch.Tensor:
    _DETERMINISTIC_PACK_COMPLEX.check((field_real, field_imag))
    if field_imag.shape != field_real.shape:
        raise ValueError("field_imag must match field_real")
    field = _DETERMINISTIC_PACK_COMPLEX(field_real, field_imag)
    validate_cuda_tensor("field", field, dtype=torch.complex64, ndim=1)
    if field.shape != field_real.shape:
        raise ValueError(
            "_channel.deterministic_pack_complex returned bad shape"
        )
    return field


def deterministic_phase_from_field(
    field_real: torch.Tensor, field_imag: torch.Tensor,
) -> torch.Tensor:
    _DETERMINISTIC_PHASE_FROM_FIELD.check((field_real, field_imag))
    if field_imag.shape != field_real.shape:
        raise ValueError("field_imag must match field_real")
    phase = _DETERMINISTIC_PHASE_FROM_FIELD(field_real, field_imag)
    validate_cuda_tensor("phase_rad", phase, dtype=torch.float32, ndim=1)
    if phase.shape != field_real.shape:
        raise ValueError(
            "_channel.deterministic_phase_from_field returned bad shape"
        )
    return phase


def deterministic_zero_field_phase(reference: torch.Tensor) -> dict[str, torch.Tensor]:
    _DETERMINISTIC_ZERO_FIELD_PHASE.check((reference,))
    exported = _DETERMINISTIC_ZERO_FIELD_PHASE(reference)
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

