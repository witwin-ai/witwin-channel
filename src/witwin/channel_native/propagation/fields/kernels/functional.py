from __future__ import annotations

import torch

from witwin.channel_native.materials import validate_layer_csr as _validate_layer_csr
from witwin.channel_native.runtime.symbols import required_symbol as _required_native_op
from witwin.channel_native.runtime.tensor_contracts import validate_cuda_tensor


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
    if not isinstance(out, dict):
        raise TypeError("_channel_native.field_free_space must return a dict")
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
        raise ValueError("_channel_native.field_free_space returned unexpected fields")
    for name, (dtype, ndim, shape) in schema.items():
        validate_cuda_tensor(name, out[name], dtype=dtype, ndim=ndim)
        if tuple(out[name].shape) != shape:
            raise ValueError(f"_channel_native.field_free_space returned bad {name} shape")
    return out


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
        raise TypeError("_channel_native.field_project_complex3 returned invalid fields")
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
    if not isinstance(out, dict):
        raise TypeError("_channel_native.field_reflection_sequence must return a dict")
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
        raise ValueError("field_reflection_sequence returned unexpected fields")
    for name, (dtype, ndim, shape) in schema.items():
        validate_cuda_tensor(name, out[name], dtype=dtype, ndim=ndim)
        if tuple(out[name].shape) != shape:
            raise ValueError(f"field_reflection_sequence returned bad {name} shape")
    return out


def field_transmission_sequence(
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
    if not isinstance(out, dict):
        raise TypeError("_channel_native.field_transmission_sequence must return a dict")
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
        raise ValueError("field_transmission_sequence returned unexpected fields")
    for name, (dtype, ndim, shape) in schema.items():
        validate_cuda_tensor(name, out[name], dtype=dtype, ndim=ndim)
        if tuple(out[name].shape) != shape:
            raise ValueError(f"field_transmission_sequence returned bad {name} shape")
    return out


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
        raise TypeError("_channel_native.field_coupled_rd must return a dict")
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
        bool(need_grad_frequency),
        bool(need_grad_geometry),
    )
    expected = {"grad_frequency", "grad_source", "grad_target"}
    if not isinstance(out, dict) or set(out) != expected:
        raise TypeError("_channel_native.field_free_space_backward returned invalid fields")
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
    if not isinstance(out, dict) or set(out) != set(_FIELD_AD_TANGENT_FIELDS):
        raise TypeError("_channel_native.field_free_space_jvp returned invalid fields")
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
            "_channel_native.field_reflection_sequence_backward returned invalid fields"
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
    if not isinstance(out, dict) or set(out) != set(_FIELD_AD_TANGENT_FIELDS):
        raise TypeError(
            "_channel_native.field_reflection_sequence_jvp returned invalid fields"
        )
    return out


def field_transmission_sequence_backward(
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
            "_channel_native.field_transmission_sequence_backward returned invalid fields"
        )
    return out


def field_transmission_sequence_jvp(
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
            "_channel_native.field_transmission_sequence_jvp returned invalid fields"
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


def field_diffraction_wedge(
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

    out = _required_native_op("field_diffraction_wedge")(
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
            "_channel_native.field_diffraction_wedge returned invalid fields"
        )
    return out


__all__ = [
    "field_coupled_rd",
    "field_diffraction_wedge",
    "field_free_space",
    "field_free_space_backward",
    "field_free_space_jvp",
    "field_project_complex3",
    "field_reflection_sequence",
    "field_reflection_sequence_backward",
    "field_reflection_sequence_jvp",
    "field_transmission_sequence",
    "field_transmission_sequence_backward",
    "field_transmission_sequence_jvp",
]
