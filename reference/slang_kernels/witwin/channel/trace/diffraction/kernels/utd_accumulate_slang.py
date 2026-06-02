"""Slang-backed fused accumulation and derivative kernels for diffraction totals."""

from __future__ import annotations

from pathlib import Path

import drjit as dr
import torch
import witwin as wt

from ....polarization import vector_zero
from ....utils import drjit_to_torch_view
from ..common import _cartesian_chunk_size, _ownership_code_from_depths
from witwin.channel.utils.drjit_ops import complex_zero
from ..execution import experimental_slang_enabled
from ..geometry import _segment_visibility_mask
from .slang_runtime import launch_shape_1d, load_slang_module


_BLOCK_SIZE = 256

_STATE_VIEW_KEYS = (
    "edgePosX",
    "edgePosY",
    "edgePosZ",
    "edgeDirX",
    "edgeDirY",
    "edgeDirZ",
    "n0X",
    "n0Y",
    "n0Z",
    "nnX",
    "nnY",
    "nnZ",
    "wedgeN",
    "sourcePosX",
    "sourcePosY",
    "sourcePosZ",
    "incidentFieldReal",
    "incidentFieldImag",
    "incidentNormalDerivativeReal",
    "incidentNormalDerivativeImag",
    "r0Real",
    "r0Imag",
    "rnReal",
    "rnImag",
    "incidentVectorXReal",
    "incidentVectorXImag",
    "incidentVectorYReal",
    "incidentVectorYImag",
    "incidentVectorZReal",
    "incidentVectorZImag",
    "incidentDerivativeVectorXReal",
    "incidentDerivativeVectorXImag",
    "incidentDerivativeVectorYReal",
    "incidentDerivativeVectorYImag",
    "incidentDerivativeVectorZReal",
    "incidentDerivativeVectorZImag",
    "incidentJonesUReal",
    "incidentJonesUImag",
    "incidentJonesVReal",
    "incidentJonesVImag",
    "incidentDerivativeJonesUReal",
    "incidentDerivativeJonesUImag",
    "incidentDerivativeJonesVReal",
    "incidentDerivativeJonesVImag",
    "incidentBasisUX",
    "incidentBasisUY",
    "incidentBasisUZ",
    "incidentBasisVX",
    "incidentBasisVY",
    "incidentBasisVZ",
    "incidentBasisKX",
    "incidentBasisKY",
    "incidentBasisKZ",
    "face0OperatorM00Real",
    "face0OperatorM00Imag",
    "face0OperatorM01Real",
    "face0OperatorM01Imag",
    "face0OperatorM10Real",
    "face0OperatorM10Imag",
    "face0OperatorM11Real",
    "face0OperatorM11Imag",
    "face1OperatorM00Real",
    "face1OperatorM00Imag",
    "face1OperatorM01Real",
    "face1OperatorM01Imag",
    "face1OperatorM10Real",
    "face1OperatorM10Imag",
    "face1OperatorM11Real",
    "face1OperatorM11Imag",
    "face0EtaR",
    "face0Sigma",
    "face0Gain",
    "face0UseFresnel",
    "face0MaterialPresent",
    "face1EtaR",
    "face1Sigma",
    "face1Gain",
    "face1UseFresnel",
    "face1MaterialPresent",
)

_OUTPUT_VIEW_KEYS = (
    "directReal",
    "directImag",
    "multiReal",
    "multiImag",
    "directVectorXReal",
    "directVectorXImag",
    "directVectorYReal",
    "directVectorYImag",
    "directVectorZReal",
    "directVectorZImag",
    "multiVectorXReal",
    "multiVectorXImag",
    "multiVectorYReal",
    "multiVectorYImag",
    "multiVectorZReal",
    "multiVectorZImag",
)


def _get_utd_accumulate_module():
    return load_slang_module(Path(__file__).with_name("utd_accumulate.slang"))


def slang_utd_accumulate_available() -> bool:
    if not experimental_slang_enabled():
        return False
    try:
        return _get_utd_accumulate_module() is not None
    except Exception:
        return False


def _scalar_float(value) -> float:
    if isinstance(value, (float, int)):
        return float(value)
    return float(dr.slice(value))


def _float_tensor_view(value, *, detach: bool) -> torch.Tensor:
    return drjit_to_torch_view(value, detach=detach, dtype=torch.float32).contiguous()


def _int_tensor_view(value) -> torch.Tensor:
    return drjit_to_torch_view(value, detach=True, dtype=torch.int32).contiguous()


def _torch_vector_components(vec, *, detach: bool) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        _float_tensor_view(vec.x, detach=detach),
        _float_tensor_view(vec.y, detach=detach),
        _float_tensor_view(vec.z, detach=detach),
    )


def _torch_complex_components(value, *, detach: bool) -> tuple[torch.Tensor, torch.Tensor]:
    return (
        _float_tensor_view(value.real, detach=detach),
        _float_tensor_view(value.imag, detach=detach),
    )


def _dr_complex_from_torch(real: torch.Tensor, imag: torch.Tensor) -> wt.Complex2f:
    return wt.Complex2f(wt.Float(real), wt.Float(imag))


def _dr_vector_from_torch(real_imag_axes: dict[str, tuple[torch.Tensor, torch.Tensor]]):
    return {
        axis: _dr_complex_from_torch(real, imag)
        for axis, (real, imag) in real_imag_axes.items()
    }


def _empty_like_views(template: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {
        key: torch.zeros_like(value, device=value.device, dtype=torch.float32)
        for key, value in template.items()
    }


def _prefixed_tensor_dict(values: dict[str, torch.Tensor], prefix: str) -> dict[str, torch.Tensor]:
    return {
        f"{prefix}{key[0].upper()}{key[1:]}": value
        for key, value in values.items()
    }


def _prepare_state_tensor_views(state_arrays, *, detach: bool) -> dict[str, torch.Tensor]:
    incident_field_real, incident_field_imag = _torch_complex_components(state_arrays["incident_field"], detach=detach)
    incident_derivative_real, incident_derivative_imag = _torch_complex_components(
        state_arrays["incident_normal_derivative"], detach=detach
    )
    r0_real, r0_imag = _torch_complex_components(state_arrays["r0"], detach=detach)
    rn_real, rn_imag = _torch_complex_components(state_arrays["rn"], detach=detach)
    incident_vector_x_real, incident_vector_x_imag = _torch_complex_components(
        state_arrays["incident_vector_x"], detach=detach
    )
    incident_vector_y_real, incident_vector_y_imag = _torch_complex_components(
        state_arrays["incident_vector_y"], detach=detach
    )
    incident_vector_z_real, incident_vector_z_imag = _torch_complex_components(
        state_arrays["incident_vector_z"], detach=detach
    )
    derivative_vector_x_real, derivative_vector_x_imag = _torch_complex_components(
        state_arrays["incident_normal_derivative_vector_x"], detach=detach
    )
    derivative_vector_y_real, derivative_vector_y_imag = _torch_complex_components(
        state_arrays["incident_normal_derivative_vector_y"], detach=detach
    )
    derivative_vector_z_real, derivative_vector_z_imag = _torch_complex_components(
        state_arrays["incident_normal_derivative_vector_z"], detach=detach
    )
    incident_jones_u_real, incident_jones_u_imag = _torch_complex_components(
        state_arrays["incident_jones_u"], detach=detach
    )
    incident_jones_v_real, incident_jones_v_imag = _torch_complex_components(
        state_arrays["incident_jones_v"], detach=detach
    )
    incident_derivative_jones_u_real, incident_derivative_jones_u_imag = _torch_complex_components(
        state_arrays["incident_derivative_jones_u"], detach=detach
    )
    incident_derivative_jones_v_real, incident_derivative_jones_v_imag = _torch_complex_components(
        state_arrays["incident_derivative_jones_v"], detach=detach
    )
    incident_basis_u_x, incident_basis_u_y, incident_basis_u_z = _torch_vector_components(
        state_arrays["incident_basis_u"], detach=detach
    )
    incident_basis_v_x, incident_basis_v_y, incident_basis_v_z = _torch_vector_components(
        state_arrays["incident_basis_v"], detach=detach
    )
    incident_basis_k_x, incident_basis_k_y, incident_basis_k_z = _torch_vector_components(
        state_arrays["incident_basis_k"], detach=detach
    )
    face0_operator_m00_real, face0_operator_m00_imag = _torch_complex_components(
        state_arrays["face0_operator_m00"], detach=detach
    )
    face0_operator_m01_real, face0_operator_m01_imag = _torch_complex_components(
        state_arrays["face0_operator_m01"], detach=detach
    )
    face0_operator_m10_real, face0_operator_m10_imag = _torch_complex_components(
        state_arrays["face0_operator_m10"], detach=detach
    )
    face0_operator_m11_real, face0_operator_m11_imag = _torch_complex_components(
        state_arrays["face0_operator_m11"], detach=detach
    )
    face1_operator_m00_real, face1_operator_m00_imag = _torch_complex_components(
        state_arrays["face1_operator_m00"], detach=detach
    )
    face1_operator_m01_real, face1_operator_m01_imag = _torch_complex_components(
        state_arrays["face1_operator_m01"], detach=detach
    )
    face1_operator_m10_real, face1_operator_m10_imag = _torch_complex_components(
        state_arrays["face1_operator_m10"], detach=detach
    )
    face1_operator_m11_real, face1_operator_m11_imag = _torch_complex_components(
        state_arrays["face1_operator_m11"], detach=detach
    )
    edge_pos_x, edge_pos_y, edge_pos_z = _torch_vector_components(state_arrays["edge_pos"], detach=detach)
    edge_dir_x, edge_dir_y, edge_dir_z = _torch_vector_components(state_arrays["edge_dir"], detach=detach)
    n0_x, n0_y, n0_z = _torch_vector_components(state_arrays["n0"], detach=detach)
    nn_x, nn_y, nn_z = _torch_vector_components(state_arrays["nn"], detach=detach)
    source_pos_x, source_pos_y, source_pos_z = _torch_vector_components(state_arrays["source_pos"], detach=detach)
    zeros_like_scalar = torch.zeros_like(incident_field_real)
    face0_eta_r = (
        _float_tensor_view(state_arrays["face0_eta_r"], detach=detach)
        if "face0_eta_r" in state_arrays
        else torch.full_like(incident_field_real, 5.0)
    )
    face0_sigma = (
        _float_tensor_view(state_arrays["face0_sigma"], detach=detach)
        if "face0_sigma" in state_arrays
        else zeros_like_scalar.clone()
    )
    face0_gain = (
        _float_tensor_view(state_arrays["face0_gain"], detach=detach)
        if "face0_gain" in state_arrays
        else torch.ones_like(incident_field_real)
    )
    face0_use_fresnel = (
        _float_tensor_view(state_arrays["face0_use_fresnel"], detach=detach)
        if "face0_use_fresnel" in state_arrays
        else zeros_like_scalar.clone()
    )
    face0_material_present = (
        torch.ones_like(incident_field_real)
        if "face0_eta_r" in state_arrays
        else zeros_like_scalar.clone()
    )
    face1_eta_r = (
        _float_tensor_view(state_arrays["face1_eta_r"], detach=detach)
        if "face1_eta_r" in state_arrays
        else torch.full_like(incident_field_real, 5.0)
    )
    face1_sigma = (
        _float_tensor_view(state_arrays["face1_sigma"], detach=detach)
        if "face1_sigma" in state_arrays
        else zeros_like_scalar.clone()
    )
    face1_gain = (
        _float_tensor_view(state_arrays["face1_gain"], detach=detach)
        if "face1_gain" in state_arrays
        else torch.ones_like(incident_field_real)
    )
    face1_use_fresnel = (
        _float_tensor_view(state_arrays["face1_use_fresnel"], detach=detach)
        if "face1_use_fresnel" in state_arrays
        else zeros_like_scalar.clone()
    )
    face1_material_present = (
        torch.ones_like(incident_field_real)
        if "face1_eta_r" in state_arrays
        else zeros_like_scalar.clone()
    )
    return {
        "edgePosX": edge_pos_x,
        "edgePosY": edge_pos_y,
        "edgePosZ": edge_pos_z,
        "edgeDirX": edge_dir_x,
        "edgeDirY": edge_dir_y,
        "edgeDirZ": edge_dir_z,
        "n0X": n0_x,
        "n0Y": n0_y,
        "n0Z": n0_z,
        "nnX": nn_x,
        "nnY": nn_y,
        "nnZ": nn_z,
        "wedgeN": _float_tensor_view(state_arrays["wedge_n"], detach=detach),
        "sourcePosX": source_pos_x,
        "sourcePosY": source_pos_y,
        "sourcePosZ": source_pos_z,
        "incidentFieldReal": incident_field_real,
        "incidentFieldImag": incident_field_imag,
        "incidentNormalDerivativeReal": incident_derivative_real,
        "incidentNormalDerivativeImag": incident_derivative_imag,
        "r0Real": r0_real,
        "r0Imag": r0_imag,
        "rnReal": rn_real,
        "rnImag": rn_imag,
        "incidentVectorXReal": incident_vector_x_real,
        "incidentVectorXImag": incident_vector_x_imag,
        "incidentVectorYReal": incident_vector_y_real,
        "incidentVectorYImag": incident_vector_y_imag,
        "incidentVectorZReal": incident_vector_z_real,
        "incidentVectorZImag": incident_vector_z_imag,
        "incidentDerivativeVectorXReal": derivative_vector_x_real,
        "incidentDerivativeVectorXImag": derivative_vector_x_imag,
        "incidentDerivativeVectorYReal": derivative_vector_y_real,
        "incidentDerivativeVectorYImag": derivative_vector_y_imag,
        "incidentDerivativeVectorZReal": derivative_vector_z_real,
        "incidentDerivativeVectorZImag": derivative_vector_z_imag,
        "incidentJonesUReal": incident_jones_u_real,
        "incidentJonesUImag": incident_jones_u_imag,
        "incidentJonesVReal": incident_jones_v_real,
        "incidentJonesVImag": incident_jones_v_imag,
        "incidentDerivativeJonesUReal": incident_derivative_jones_u_real,
        "incidentDerivativeJonesUImag": incident_derivative_jones_u_imag,
        "incidentDerivativeJonesVReal": incident_derivative_jones_v_real,
        "incidentDerivativeJonesVImag": incident_derivative_jones_v_imag,
        "incidentBasisUX": incident_basis_u_x,
        "incidentBasisUY": incident_basis_u_y,
        "incidentBasisUZ": incident_basis_u_z,
        "incidentBasisVX": incident_basis_v_x,
        "incidentBasisVY": incident_basis_v_y,
        "incidentBasisVZ": incident_basis_v_z,
        "incidentBasisKX": incident_basis_k_x,
        "incidentBasisKY": incident_basis_k_y,
        "incidentBasisKZ": incident_basis_k_z,
        "face0OperatorM00Real": face0_operator_m00_real,
        "face0OperatorM00Imag": face0_operator_m00_imag,
        "face0OperatorM01Real": face0_operator_m01_real,
        "face0OperatorM01Imag": face0_operator_m01_imag,
        "face0OperatorM10Real": face0_operator_m10_real,
        "face0OperatorM10Imag": face0_operator_m10_imag,
        "face0OperatorM11Real": face0_operator_m11_real,
        "face0OperatorM11Imag": face0_operator_m11_imag,
        "face1OperatorM00Real": face1_operator_m00_real,
        "face1OperatorM00Imag": face1_operator_m00_imag,
        "face1OperatorM01Real": face1_operator_m01_real,
        "face1OperatorM01Imag": face1_operator_m01_imag,
        "face1OperatorM10Real": face1_operator_m10_real,
        "face1OperatorM10Imag": face1_operator_m10_imag,
        "face1OperatorM11Real": face1_operator_m11_real,
        "face1OperatorM11Imag": face1_operator_m11_imag,
        "face0EtaR": face0_eta_r,
        "face0Sigma": face0_sigma,
        "face0Gain": face0_gain,
        "face0UseFresnel": face0_use_fresnel,
        "face0MaterialPresent": face0_material_present,
        "face1EtaR": face1_eta_r,
        "face1Sigma": face1_sigma,
        "face1Gain": face1_gain,
        "face1UseFresnel": face1_use_fresnel,
        "face1MaterialPresent": face1_material_present,
    }


def _prepare_rx_tensor_views(rx_pos, *, detach: bool) -> dict[str, torch.Tensor]:
    return {
        "x": _float_tensor_view(rx_pos.x, detach=detach),
        "y": _float_tensor_view(rx_pos.y, detach=detach),
        "z": _float_tensor_view(rx_pos.z, detach=detach),
    }


def _allocate_output_tensors(n_rx: int, device: torch.device) -> dict[str, torch.Tensor]:
    def zeros() -> torch.Tensor:
        return torch.zeros(n_rx, device=device, dtype=torch.float32)

    return {key: zeros() for key in _OUTPUT_VIEW_KEYS}


def _prepare_dispatch_tensors(state_arrays, state_idx, rx_idx) -> dict[str, torch.Tensor]:
    ownership_code = _int_tensor_view(
        _ownership_code_from_depths(
            state_arrays["prefix_reflection_depth"],
            state_arrays["intermediate_reflection_depth"],
            state_arrays["suffix_reflection_depth"],
        )
    )
    return {
        "stateIndex": _int_tensor_view(state_idx),
        "rxIndex": _int_tensor_view(rx_idx),
        "ownershipCode": ownership_code,
    }


def _material_params(material_detail, wavelength) -> dict[str, float | int]:
    return {
        "useFresnel": 0,
        "etaR": 1.0,
        "sigma": 0.0,
        "gain": 1.0,
        "omega": 0.0 if wavelength is None else float(2.0 * 3.141592653589793 * 299792458.0 / wavelength),
    }


def _launch_kernel(module_fn, *, n_pairs: int, **kwargs) -> None:
    if n_pairs == 0:
        return
    block_size = (_BLOCK_SIZE, 1, 1)
    module_fn(**kwargs).launchRaw(
        blockSize=block_size,
        gridSize=launch_shape_1d(n_pairs, block_size[0]),
    )


def _launch_forward_kernel(
    module,
    dispatch_tensors,
    state_tensors,
    rx_tensors,
    output_tensors,
    material_params,
    *,
    n_pairs: int,
    k,
) -> None:
    _launch_kernel(
        module.utdAccumulateForward,
        n_pairs=n_pairs,
        **dispatch_tensors,
        **state_tensors,
        rxX=rx_tensors["x"],
        rxY=rx_tensors["y"],
        rxZ=rx_tensors["z"],
        **output_tensors,
        nPairs=int(n_pairs),
        k=_scalar_float(k),
        material=material_params,
    )


def _launch_forward_jvp_kernel(
    module,
    dispatch_tensors,
    state_tensors,
    state_tangents,
    rx_tensors,
    rx_tangents,
    output_tangents,
    material_params,
    *,
    n_pairs: int,
    k,
) -> None:
    _launch_kernel(
        module.utdAccumulateForwardJvp,
        n_pairs=n_pairs,
        **dispatch_tensors,
        **state_tensors,
        **_prefixed_tensor_dict(state_tangents, "tangent"),
        rxX=rx_tensors["x"],
        rxY=rx_tensors["y"],
        rxZ=rx_tensors["z"],
        tangentRxX=rx_tangents["x"],
        tangentRxY=rx_tangents["y"],
        tangentRxZ=rx_tangents["z"],
        **output_tangents,
        nPairs=int(n_pairs),
        k=_scalar_float(k),
        material=material_params,
    )


def _launch_field_forward_jvp_kernel(
    module,
    dispatch_tensors,
    state_tensors,
    state_tangents,
    rx_tensors,
    rx_tangents,
    output_tangents,
    material_params,
    *,
    n_pairs: int,
    k,
) -> None:
    _launch_kernel(
        module.utdAccumulateFieldForwardJvp,
        n_pairs=n_pairs,
        **dispatch_tensors,
        **state_tensors,
        **_prefixed_tensor_dict(state_tangents, "tangent"),
        rxX=rx_tensors["x"],
        rxY=rx_tensors["y"],
        rxZ=rx_tensors["z"],
        tangentRxX=rx_tangents["x"],
        tangentRxY=rx_tangents["y"],
        tangentRxZ=rx_tangents["z"],
        **output_tangents,
        nPairs=int(n_pairs),
        k=_scalar_float(k),
        material=material_params,
    )


def _launch_backward_full_kernel(
    module,
    dispatch_tensors,
    state_tensors,
    rx_tensors,
    output_grads,
    state_grads,
    rx_grads,
    material_params,
    *,
    n_pairs: int,
    k,
) -> None:
    _launch_kernel(
        module.utdAccumulateBackwardFull,
        n_pairs=n_pairs,
        **dispatch_tensors,
        **state_tensors,
        rxX=rx_tensors["x"],
        rxY=rx_tensors["y"],
        rxZ=rx_tensors["z"],
        **output_grads,
        **_prefixed_tensor_dict(state_grads, "grad"),
        gradRxX=rx_grads["x"],
        gradRxY=rx_grads["y"],
        gradRxZ=rx_grads["z"],
        nPairs=int(n_pairs),
        k=_scalar_float(k),
        material=material_params,
    )


def _launch_backward_scalar_kernel(
    module,
    dispatch_tensors,
    state_tensors,
    rx_tensors,
    output_grads,
    state_grads,
    rx_grads,
    material_params,
    *,
    n_pairs: int,
    k,
) -> None:
    _launch_kernel(
        module.utdAccumulateBackwardScalar,
        n_pairs=n_pairs,
        **dispatch_tensors,
        **state_tensors,
        rxX=rx_tensors["x"],
        rxY=rx_tensors["y"],
        rxZ=rx_tensors["z"],
        directReal=output_grads["directReal"],
        directImag=output_grads["directImag"],
        multiReal=output_grads["multiReal"],
        multiImag=output_grads["multiImag"],
        **_prefixed_tensor_dict(state_grads, "grad"),
        gradRxX=rx_grads["x"],
        gradRxY=rx_grads["y"],
        gradRxZ=rx_grads["z"],
        nPairs=int(n_pairs),
        k=_scalar_float(k),
        material=material_params,
    )


def _launch_backward_vector_kernel(
    module,
    dispatch_tensors,
    state_tensors,
    rx_tensors,
    output_grads,
    state_grads,
    rx_grads,
    material_params,
    *,
    n_pairs: int,
    k,
) -> None:
    _launch_kernel(
        module.utdAccumulateBackwardVector,
        n_pairs=n_pairs,
        **dispatch_tensors,
        **state_tensors,
        rxX=rx_tensors["x"],
        rxY=rx_tensors["y"],
        rxZ=rx_tensors["z"],
        **output_grads,
        **_prefixed_tensor_dict(state_grads, "grad"),
        gradRxX=rx_grads["x"],
        gradRxY=rx_grads["y"],
        gradRxZ=rx_grads["z"],
        nPairs=int(n_pairs),
        k=_scalar_float(k),
        material=material_params,
    )


def _receiver_vector_grads_are_zero(output_grads) -> bool:
    if output_grads is None:
        return True
    for vector in output_grads[2:]:
        for axis in ("x", "y", "z"):
            if dr.any(dr.abs(vector[axis].real) > 0) or dr.any(dr.abs(vector[axis].imag) > 0):
                return False
    return True


def _dr_outputs_from_torch(output_tensors: dict[str, torch.Tensor]):
    direct_total = _dr_complex_from_torch(output_tensors["directReal"], output_tensors["directImag"])
    multi_total = _dr_complex_from_torch(output_tensors["multiReal"], output_tensors["multiImag"])
    direct_vector_total = _dr_vector_from_torch(
        {
            "x": (output_tensors["directVectorXReal"], output_tensors["directVectorXImag"]),
            "y": (output_tensors["directVectorYReal"], output_tensors["directVectorYImag"]),
            "z": (output_tensors["directVectorZReal"], output_tensors["directVectorZImag"]),
        }
    )
    multi_vector_total = _dr_vector_from_torch(
        {
            "x": (output_tensors["multiVectorXReal"], output_tensors["multiVectorXImag"]),
            "y": (output_tensors["multiVectorYReal"], output_tensors["multiVectorYImag"]),
            "z": (output_tensors["multiVectorZReal"], output_tensors["multiVectorZImag"]),
        }
    )
    return direct_total, multi_total, direct_vector_total, multi_vector_total


def _zero_like_state_grad_structure(state_arrays):
    grads = {}
    for key, value in state_arrays.items():
        if key == "n_states":
            grads[key] = value
        elif isinstance(value, (int, float, bool)):
            grads[key] = type(value)()
        else:
            grads[key] = dr.zeros(type(value), dr.width(value))
    return grads


def _coalesce_state_like(state_like, reference_state):
    if state_like is None:
        return _zero_like_state_grad_structure(reference_state)
    return state_like


def _coalesce_rx_like(rx_like, reference_rx):
    if rx_like is None:
        width = dr.width(reference_rx.x)
        zeros = dr.zeros(wt.Float, width)
        return wt.Point3f(zeros, zeros, zeros)
    return rx_like


def _state_grads_from_torch(state_arrays, state_grad_tensors: dict[str, torch.Tensor]):
    grads = _zero_like_state_grad_structure(state_arrays)
    grads["edge_pos"] = wt.Point3f(
        wt.Float(state_grad_tensors["edgePosX"]),
        wt.Float(state_grad_tensors["edgePosY"]),
        wt.Float(state_grad_tensors["edgePosZ"]),
    )
    grads["edge_dir"] = wt.Vector3f(
        wt.Float(state_grad_tensors["edgeDirX"]),
        wt.Float(state_grad_tensors["edgeDirY"]),
        wt.Float(state_grad_tensors["edgeDirZ"]),
    )
    grads["n0"] = wt.Vector3f(
        wt.Float(state_grad_tensors["n0X"]),
        wt.Float(state_grad_tensors["n0Y"]),
        wt.Float(state_grad_tensors["n0Z"]),
    )
    grads["nn"] = wt.Vector3f(
        wt.Float(state_grad_tensors["nnX"]),
        wt.Float(state_grad_tensors["nnY"]),
        wt.Float(state_grad_tensors["nnZ"]),
    )
    grads["wedge_n"] = wt.Float(state_grad_tensors["wedgeN"])
    grads["source_pos"] = wt.Point3f(
        wt.Float(state_grad_tensors["sourcePosX"]),
        wt.Float(state_grad_tensors["sourcePosY"]),
        wt.Float(state_grad_tensors["sourcePosZ"]),
    )
    grads["incident_field"] = _dr_complex_from_torch(
        state_grad_tensors["incidentFieldReal"], state_grad_tensors["incidentFieldImag"]
    )
    grads["incident_normal_derivative"] = _dr_complex_from_torch(
        state_grad_tensors["incidentNormalDerivativeReal"], state_grad_tensors["incidentNormalDerivativeImag"]
    )
    grads["r0"] = _dr_complex_from_torch(
        state_grad_tensors["r0Real"], state_grad_tensors["r0Imag"]
    )
    grads["rn"] = _dr_complex_from_torch(
        state_grad_tensors["rnReal"], state_grad_tensors["rnImag"]
    )
    for axis, real_key, imag_key in (
        ("incident_vector_x", "incidentVectorXReal", "incidentVectorXImag"),
        ("incident_vector_y", "incidentVectorYReal", "incidentVectorYImag"),
        ("incident_vector_z", "incidentVectorZReal", "incidentVectorZImag"),
        (
            "incident_normal_derivative_vector_x",
            "incidentDerivativeVectorXReal",
            "incidentDerivativeVectorXImag",
        ),
        (
            "incident_normal_derivative_vector_y",
            "incidentDerivativeVectorYReal",
            "incidentDerivativeVectorYImag",
        ),
        (
            "incident_normal_derivative_vector_z",
            "incidentDerivativeVectorZReal",
            "incidentDerivativeVectorZImag",
        ),
    ):
        grads[axis] = _dr_complex_from_torch(state_grad_tensors[real_key], state_grad_tensors[imag_key])
    grads["incident_jones_u"] = _dr_complex_from_torch(
        state_grad_tensors["incidentJonesUReal"], state_grad_tensors["incidentJonesUImag"]
    )
    grads["incident_jones_v"] = _dr_complex_from_torch(
        state_grad_tensors["incidentJonesVReal"], state_grad_tensors["incidentJonesVImag"]
    )
    grads["incident_derivative_jones_u"] = _dr_complex_from_torch(
        state_grad_tensors["incidentDerivativeJonesUReal"], state_grad_tensors["incidentDerivativeJonesUImag"]
    )
    grads["incident_derivative_jones_v"] = _dr_complex_from_torch(
        state_grad_tensors["incidentDerivativeJonesVReal"], state_grad_tensors["incidentDerivativeJonesVImag"]
    )
    grads["incident_basis_u"] = wt.Vector3f(
        wt.Float(state_grad_tensors["incidentBasisUX"]),
        wt.Float(state_grad_tensors["incidentBasisUY"]),
        wt.Float(state_grad_tensors["incidentBasisUZ"]),
    )
    grads["incident_basis_v"] = wt.Vector3f(
        wt.Float(state_grad_tensors["incidentBasisVX"]),
        wt.Float(state_grad_tensors["incidentBasisVY"]),
        wt.Float(state_grad_tensors["incidentBasisVZ"]),
    )
    grads["incident_basis_k"] = wt.Vector3f(
        wt.Float(state_grad_tensors["incidentBasisKX"]),
        wt.Float(state_grad_tensors["incidentBasisKY"]),
        wt.Float(state_grad_tensors["incidentBasisKZ"]),
    )
    for prefix in ("face0", "face1"):
        for element in ("m00", "m01", "m10", "m11"):
            grads[f"{prefix}_operator_{element}"] = _dr_complex_from_torch(
                state_grad_tensors[f"{prefix}Operator{element.upper()}Real"],
                state_grad_tensors[f"{prefix}Operator{element.upper()}Imag"],
            )
    grads["face0_eta_r"] = wt.Float(state_grad_tensors["face0EtaR"])
    grads["face0_sigma"] = wt.Float(state_grad_tensors["face0Sigma"])
    grads["face0_gain"] = wt.Float(state_grad_tensors["face0Gain"])
    grads["face1_eta_r"] = wt.Float(state_grad_tensors["face1EtaR"])
    grads["face1_sigma"] = wt.Float(state_grad_tensors["face1Sigma"])
    grads["face1_gain"] = wt.Float(state_grad_tensors["face1Gain"])
    return grads


def _rx_grads_from_torch(rx_grad_tensors: dict[str, torch.Tensor]) -> wt.Point3f:
    return wt.Point3f(
        wt.Float(rx_grad_tensors["x"]),
        wt.Float(rx_grad_tensors["y"]),
        wt.Float(rx_grad_tensors["z"]),
    )


def _visible_state_rx_indices(state_arrays, rx_pos, scene, state_start, chunk_n_states, n_rx):
    n_pairs = chunk_n_states * n_rx
    pair_idx = dr.arange(wt.UInt32, n_pairs)
    state_idx = pair_idx // n_rx + wt.UInt32(state_start)
    rx_idx = pair_idx % n_rx

    if scene is None:
        return state_idx, rx_idx

    state_edge_pos = dr.gather(wt.Point3f, state_arrays["edge_pos"], state_idx)
    batch_rx_all = wt.Point3f(
        dr.gather(wt.Float, rx_pos.x, rx_idx),
        dr.gather(wt.Float, rx_pos.y, rx_idx),
        dr.gather(wt.Float, rx_pos.z, rx_idx),
    )
    visible = _segment_visibility_mask(
        state_edge_pos,
        batch_rx_all,
        scene,
        ignore_prim_idx=(
            dr.gather(wt.Int32, state_arrays["adjacent_face0"], state_idx),
            dr.gather(wt.Int32, state_arrays["adjacent_face1"], state_idx),
        ),
    )
    pair_idx = dr.compress(visible)
    if dr.width(pair_idx) == 0:
        return None, None
    return pair_idx // n_rx + wt.UInt32(state_start), pair_idx % n_rx


def _iter_dispatch_chunks(state_arrays, rx_pos, scene, dispatch_chunks):
    if dispatch_chunks is not None:
        for state_idx, rx_idx in dispatch_chunks:
            if state_idx is None or rx_idx is None or dr.width(state_idx) == 0:
                continue
            yield state_idx, rx_idx
        return

    n_states = int(state_arrays["n_states"])
    n_rx = dr.width(rx_pos.x)
    state_chunk_size = _cartesian_chunk_size(n_states, n_rx)
    for state_start in range(0, n_states, state_chunk_size):
        chunk_n_states = min(state_chunk_size, n_states - state_start)
        state_idx, rx_idx = _visible_state_rx_indices(state_arrays, rx_pos, scene, state_start, chunk_n_states, n_rx)
        if state_idx is None:
            continue
        yield state_idx, rx_idx


def accumulate_edge_state_totals_slang(
    state_arrays,
    rx_pos,
    k,
    scene=None,
    wavelength=None,
    material_detail=None,
    dispatch_chunks=None,
):
    if dr.backend_v(type(rx_pos.x)) != dr.JitBackend.CUDA:
        return None
    if not slang_utd_accumulate_available():
        return None

    module = _get_utd_accumulate_module()
    n_states = int(state_arrays["n_states"])
    n_rx = dr.width(rx_pos.x)
    direct_total = complex_zero(n_rx)
    multi_total = complex_zero(n_rx)
    direct_vector_total = vector_zero(n_rx)
    multi_vector_total = vector_zero(n_rx)
    if n_states == 0 or n_rx == 0:
        return direct_total, multi_total, direct_vector_total, multi_vector_total

    state_tensors = _prepare_state_tensor_views(state_arrays, detach=True)
    rx_tensors = _prepare_rx_tensor_views(rx_pos, detach=True)
    output_tensors = _allocate_output_tensors(n_rx, rx_tensors["x"].device)
    material_params = _material_params(material_detail, wavelength)

    for state_idx, rx_idx in _iter_dispatch_chunks(state_arrays, rx_pos, scene, dispatch_chunks):
        dispatch_tensors = _prepare_dispatch_tensors(state_arrays, state_idx, rx_idx)
        _launch_forward_kernel(
            module,
            dispatch_tensors,
            state_tensors,
            rx_tensors,
            output_tensors,
            material_params,
            n_pairs=int(dispatch_tensors["stateIndex"].shape[0]),
            k=k,
        )

    return _dr_outputs_from_torch(output_tensors)


def accumulate_edge_state_totals_slang_forward_jvp(
    state_arrays,
    rx_pos,
    state_tangents,
    rx_tangents,
    k,
    scene=None,
    wavelength=None,
    material_detail=None,
    dispatch_chunks=None,
):
    if dr.backend_v(type(rx_pos.x)) != dr.JitBackend.CUDA:
        return None
    if not slang_utd_accumulate_available():
        return None

    module = _get_utd_accumulate_module()
    n_states = int(state_arrays["n_states"])
    n_rx = dr.width(rx_pos.x)
    if n_states == 0 or n_rx == 0:
        zero_complex = complex_zero(n_rx)
        zero_vector = vector_zero(n_rx)
        return zero_complex, zero_complex, zero_vector, zero_vector

    state_tangents = _coalesce_state_like(state_tangents, state_arrays)
    rx_tangents = _coalesce_rx_like(rx_tangents, rx_pos)
    state_tensors = _prepare_state_tensor_views(state_arrays, detach=True)
    state_tangent_tensors = _prepare_state_tensor_views(state_tangents, detach=True)
    rx_tensors = _prepare_rx_tensor_views(rx_pos, detach=True)
    rx_tangent_tensors = _prepare_rx_tensor_views(rx_tangents, detach=True)
    output_tensors = _allocate_output_tensors(n_rx, rx_tensors["x"].device)
    material_params = _material_params(material_detail, wavelength)

    for state_idx, rx_idx in _iter_dispatch_chunks(state_arrays, rx_pos, scene, dispatch_chunks):
        dispatch_tensors = _prepare_dispatch_tensors(state_arrays, state_idx, rx_idx)
        _launch_forward_jvp_kernel(
            module,
            dispatch_tensors,
            state_tensors,
            state_tangent_tensors,
            rx_tensors,
            rx_tangent_tensors,
            output_tensors,
            material_params,
            n_pairs=int(dispatch_tensors["stateIndex"].shape[0]),
            k=k,
        )

    return _dr_outputs_from_torch(output_tensors)


def accumulate_edge_state_field_totals_slang_forward_jvp(
    state_arrays,
    rx_pos,
    state_tangents,
    rx_tangents,
    k,
    scene=None,
    wavelength=None,
    material_detail=None,
    dispatch_chunks=None,
):
    if dr.backend_v(type(rx_pos.x)) != dr.JitBackend.CUDA:
        return None
    if not slang_utd_accumulate_available():
        return None

    module = _get_utd_accumulate_module()
    n_states = int(state_arrays["n_states"])
    n_rx = dr.width(rx_pos.x)
    if n_states == 0 or n_rx == 0:
        zero_complex = complex_zero(n_rx)
        zero_vector = vector_zero(n_rx)
        return zero_complex, zero_complex, zero_vector, zero_vector

    state_tangents = _coalesce_state_like(state_tangents, state_arrays)
    rx_tangents = _coalesce_rx_like(rx_tangents, rx_pos)
    state_tensors = _prepare_state_tensor_views(state_arrays, detach=True)
    state_tangent_tensors = _prepare_state_tensor_views(state_tangents, detach=True)
    rx_tensors = _prepare_rx_tensor_views(rx_pos, detach=True)
    rx_tangent_tensors = _prepare_rx_tensor_views(rx_tangents, detach=True)
    output_tensors = _allocate_output_tensors(n_rx, rx_tensors["x"].device)
    material_params = _material_params(material_detail, wavelength)

    for state_idx, rx_idx in _iter_dispatch_chunks(state_arrays, rx_pos, scene, dispatch_chunks):
        dispatch_tensors = _prepare_dispatch_tensors(state_arrays, state_idx, rx_idx)
        _launch_field_forward_jvp_kernel(
            module,
            dispatch_tensors,
            state_tensors,
            state_tangent_tensors,
            rx_tensors,
            rx_tangent_tensors,
            output_tensors,
            material_params,
            n_pairs=int(dispatch_tensors["stateIndex"].shape[0]),
            k=k,
        )

    return _dr_outputs_from_torch(output_tensors)


def accumulate_edge_state_totals_slang_backward(
    state_arrays,
    rx_pos,
    output_grads,
    k,
    scene=None,
    wavelength=None,
    material_detail=None,
    dispatch_chunks=None,
):
    if dr.backend_v(type(rx_pos.x)) != dr.JitBackend.CUDA:
        return None
    if not slang_utd_accumulate_available():
        return None

    module = _get_utd_accumulate_module()
    n_states = int(state_arrays["n_states"])
    n_rx = dr.width(rx_pos.x)
    state_grad_struct = _zero_like_state_grad_structure(state_arrays)
    rx_grad = wt.Point3f(dr.zeros(wt.Float, n_rx), dr.zeros(wt.Float, n_rx), dr.zeros(wt.Float, n_rx))
    if n_states == 0 or n_rx == 0:
        return state_grad_struct, rx_grad

    state_tensors = _prepare_state_tensor_views(state_arrays, detach=True)
    rx_tensors = _prepare_rx_tensor_views(rx_pos, detach=True)
    output_grads = (
        output_grads
        if output_grads is not None
        else (
            complex_zero(n_rx),
            complex_zero(n_rx),
            vector_zero(n_rx),
            vector_zero(n_rx),
        )
    )
    output_grad_tensors = {
        "directReal": _float_tensor_view(output_grads[0].real, detach=True),
        "directImag": _float_tensor_view(output_grads[0].imag, detach=True),
        "multiReal": _float_tensor_view(output_grads[1].real, detach=True),
        "multiImag": _float_tensor_view(output_grads[1].imag, detach=True),
        "directVectorXReal": _float_tensor_view(output_grads[2]["x"].real, detach=True),
        "directVectorXImag": _float_tensor_view(output_grads[2]["x"].imag, detach=True),
        "directVectorYReal": _float_tensor_view(output_grads[2]["y"].real, detach=True),
        "directVectorYImag": _float_tensor_view(output_grads[2]["y"].imag, detach=True),
        "directVectorZReal": _float_tensor_view(output_grads[2]["z"].real, detach=True),
        "directVectorZImag": _float_tensor_view(output_grads[2]["z"].imag, detach=True),
        "multiVectorXReal": _float_tensor_view(output_grads[3]["x"].real, detach=True),
        "multiVectorXImag": _float_tensor_view(output_grads[3]["x"].imag, detach=True),
        "multiVectorYReal": _float_tensor_view(output_grads[3]["y"].real, detach=True),
        "multiVectorYImag": _float_tensor_view(output_grads[3]["y"].imag, detach=True),
        "multiVectorZReal": _float_tensor_view(output_grads[3]["z"].real, detach=True),
        "multiVectorZImag": _float_tensor_view(output_grads[3]["z"].imag, detach=True),
    }
    state_grad_tensors = _empty_like_views(state_tensors)
    rx_grad_tensors = _empty_like_views(rx_tensors)
    material_params = _material_params(material_detail, wavelength)
    use_scalar_kernel = _receiver_vector_grads_are_zero(output_grads)

    for state_idx, rx_idx in _iter_dispatch_chunks(state_arrays, rx_pos, scene, dispatch_chunks):
        dispatch_tensors = _prepare_dispatch_tensors(state_arrays, state_idx, rx_idx)
        if use_scalar_kernel:
            _launch_backward_scalar_kernel(
                module,
                dispatch_tensors,
                state_tensors,
                rx_tensors,
                output_grad_tensors,
                state_grad_tensors,
                rx_grad_tensors,
                material_params,
                n_pairs=int(dispatch_tensors["stateIndex"].shape[0]),
                k=k,
            )
        else:
            _launch_backward_full_kernel(
                module,
                dispatch_tensors,
                state_tensors,
                rx_tensors,
                output_grad_tensors,
                state_grad_tensors,
                rx_grad_tensors,
                material_params,
                n_pairs=int(dispatch_tensors["stateIndex"].shape[0]),
                k=k,
            )

    return _state_grads_from_torch(state_arrays, state_grad_tensors), _rx_grads_from_torch(rx_grad_tensors)


def accumulate_edge_state_totals_slang_vector_backward(
    state_arrays,
    rx_pos,
    output_grads,
    k,
    scene=None,
    wavelength=None,
    material_detail=None,
    dispatch_chunks=None,
):
    if dr.backend_v(type(rx_pos.x)) != dr.JitBackend.CUDA:
        return None
    if not slang_utd_accumulate_available():
        return None

    module = _get_utd_accumulate_module()
    n_states = int(state_arrays["n_states"])
    n_rx = dr.width(rx_pos.x)
    state_grad_struct = _zero_like_state_grad_structure(state_arrays)
    rx_grad = wt.Point3f(dr.zeros(wt.Float, n_rx), dr.zeros(wt.Float, n_rx), dr.zeros(wt.Float, n_rx))
    if n_states == 0 or n_rx == 0:
        return state_grad_struct, rx_grad

    state_tensors = _prepare_state_tensor_views(state_arrays, detach=True)
    rx_tensors = _prepare_rx_tensor_views(rx_pos, detach=True)
    output_grads = (
        output_grads
        if output_grads is not None
        else (
            complex_zero(n_rx),
            complex_zero(n_rx),
            vector_zero(n_rx),
            vector_zero(n_rx),
        )
    )
    output_grad_tensors = {
        "directReal": _float_tensor_view(output_grads[0].real, detach=True),
        "directImag": _float_tensor_view(output_grads[0].imag, detach=True),
        "multiReal": _float_tensor_view(output_grads[1].real, detach=True),
        "multiImag": _float_tensor_view(output_grads[1].imag, detach=True),
        "directVectorXReal": _float_tensor_view(output_grads[2]["x"].real, detach=True),
        "directVectorXImag": _float_tensor_view(output_grads[2]["x"].imag, detach=True),
        "directVectorYReal": _float_tensor_view(output_grads[2]["y"].real, detach=True),
        "directVectorYImag": _float_tensor_view(output_grads[2]["y"].imag, detach=True),
        "directVectorZReal": _float_tensor_view(output_grads[2]["z"].real, detach=True),
        "directVectorZImag": _float_tensor_view(output_grads[2]["z"].imag, detach=True),
        "multiVectorXReal": _float_tensor_view(output_grads[3]["x"].real, detach=True),
        "multiVectorXImag": _float_tensor_view(output_grads[3]["x"].imag, detach=True),
        "multiVectorYReal": _float_tensor_view(output_grads[3]["y"].real, detach=True),
        "multiVectorYImag": _float_tensor_view(output_grads[3]["y"].imag, detach=True),
        "multiVectorZReal": _float_tensor_view(output_grads[3]["z"].real, detach=True),
        "multiVectorZImag": _float_tensor_view(output_grads[3]["z"].imag, detach=True),
    }
    state_grad_tensors = _empty_like_views(state_tensors)
    rx_grad_tensors = _empty_like_views(rx_tensors)
    material_params = _material_params(material_detail, wavelength)

    for state_idx, rx_idx in _iter_dispatch_chunks(state_arrays, rx_pos, scene, dispatch_chunks):
        dispatch_tensors = _prepare_dispatch_tensors(state_arrays, state_idx, rx_idx)
        _launch_backward_vector_kernel(
            module,
            dispatch_tensors,
            state_tensors,
            rx_tensors,
            output_grad_tensors,
            state_grad_tensors,
            rx_grad_tensors,
            material_params,
            n_pairs=int(dispatch_tensors["stateIndex"].shape[0]),
            k=k,
        )

    return _state_grads_from_torch(state_arrays, state_grad_tensors), _rx_grads_from_torch(rx_grad_tensors)


__all__ = [
    "accumulate_edge_state_field_totals_slang_forward_jvp",
    "accumulate_edge_state_totals_slang",
    "accumulate_edge_state_totals_slang_backward",
    "accumulate_edge_state_totals_slang_vector_backward",
    "accumulate_edge_state_totals_slang_forward_jvp",
    "slang_utd_accumulate_available",
]
