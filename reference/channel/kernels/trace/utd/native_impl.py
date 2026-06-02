"""
Native C++/CUDA implementation of the UTD accumulation kernel.

Python does ONLY:
  1. Build (state, rx) pair indices
  2. Visibility filtering (needs BVH; cannot move to CUDA)
  3. Pack SoA pointers
  4. ONE call to C++ mega-kernel (field eval + ownership + atomicAdd scatter)
  5. Unpack result

NO dr.scatter_reduce, dr.select, or per-axis loops on the Python side.
"""

from __future__ import annotations

import drjit as dr

import witwin as wt
from witwin.channel._native import _extension
from witwin.channel.utils.drjit_ops import ArrayInit, complex_abs_sqr, eval_complex
from witwin.channel.utils.polarization import (
    effective_rx_polarization,
    scalarize_tangential_jones,
    tangential_jones,
    vector_zero,
)
from witwin.channel.trace.materials import reflection_material_omega
from witwin.channel.trace.diffraction.constants import (
    _cartesian_chunk_size,
    _ownership_code_from_depths,
)
from witwin.channel.config import coerce_diffraction_execution
from witwin.channel.kernels.trace.cartesian_filter.native_impl import compact_index_pairs
from witwin.channel.kernels.trace.packed_state import gather_state_arrays
from witwin.channel.trace.diffraction.state import (
    gather_path_export_eval_state_fields,
    is_path_export_reduced_state_arrays,
)


_UTD_SCALAR_POWER_PAIR_CHUNK = 1 << 20


def _state_arrays_have_finite_edge_bounds(state_arrays: dict) -> bool:
    return (
        state_arrays is not None
        and state_arrays.get("edge_line_min") is not None
        and state_arrays.get("edge_line_max") is not None
    )


def _require_finite_edge_bounds(state_arrays: dict, *, context: str) -> None:
    if _state_arrays_have_finite_edge_bounds(state_arrays):
        return
    raise RuntimeError(
        f"{context} requires finite-wedge state arrays with edge_line_min and edge_line_max."
    )


def _detach_native_index_array(arr):
    return None if arr is None else dr.cuda.Int(arr)


def _pack_state_soa(s):
    """Extract all SoA raw float pointers from a state dict.

    Returns a flat tuple in the order expected by the C++ kernel.
    """
    dr.eval(
        s["edge_pos"], s["edge_dir"], s["n0"], s["n_face_n"],
        s["wedge_n"], s["edge_line_min"], s["edge_line_max"], s["source_pos"],
        s["incident_field"], s["incident_normal_derivative"],
        s["r_face0"], s["r_face_n"],
        s["incident_vector_x"], s["incident_vector_y"], s["incident_vector_z"],
        s["incident_normal_derivative_vector_x"],
        s["incident_normal_derivative_vector_y"],
        s["incident_normal_derivative_vector_z"],
        s["incident_jones_u"], s["incident_jones_v"],
        s["incident_derivative_jones_u"], s["incident_derivative_jones_v"],
        s["incident_basis_u"], s["incident_basis_v"], s["incident_basis_k"],
        s["face0_operator_m00"], s["face0_operator_m01"],
        s["face0_operator_m10"], s["face0_operator_m11"],
        s["face1_operator_m00"], s["face1_operator_m01"],
        s["face1_operator_m10"], s["face1_operator_m11"],
        s["face0_eta_r"], s["face0_sigma"], s["face0_gain"],
        s["face1_eta_r"], s["face1_sigma"], s["face1_gain"],
    )
    p = s["edge_pos"]
    d = s["edge_dir"]
    n0 = s["n0"]
    nn = s["n_face_n"]
    sp = s["source_pos"]
    # Bool -> Float for use_fresnel / present
    f0uf = wt.Float(s["face0_use_fresnel"])
    f1uf = wt.Float(s["face1_use_fresnel"])
    n_s = int(s["n_states"])
    f0pr = dr.full(wt.Float, 1.0, n_s)
    f1pr = dr.full(wt.Float, 1.0, n_s)
    dr.eval(f0uf, f1uf, f0pr, f1pr)
    return (
        p.x, p.y, p.z,
        d.x, d.y, d.z,
        n0.x, n0.y, n0.z,
        nn.x, nn.y, nn.z,
        s["wedge_n"],
        s["edge_line_min"], s["edge_line_max"],
        sp.x, sp.y, sp.z,
        s["incident_field"].real, s["incident_field"].imag,
        s["incident_normal_derivative"].real, s["incident_normal_derivative"].imag,
        s["r_face0"].real, s["r_face0"].imag,
        s["r_face_n"].real, s["r_face_n"].imag,
        s["incident_vector_x"].real, s["incident_vector_x"].imag,
        s["incident_vector_y"].real, s["incident_vector_y"].imag,
        s["incident_vector_z"].real, s["incident_vector_z"].imag,
        s["incident_normal_derivative_vector_x"].real,
        s["incident_normal_derivative_vector_x"].imag,
        s["incident_normal_derivative_vector_y"].real,
        s["incident_normal_derivative_vector_y"].imag,
        s["incident_normal_derivative_vector_z"].real,
        s["incident_normal_derivative_vector_z"].imag,
        s["incident_jones_u"].real, s["incident_jones_u"].imag,
        s["incident_jones_v"].real, s["incident_jones_v"].imag,
        s["incident_derivative_jones_u"].real,
        s["incident_derivative_jones_u"].imag,
        s["incident_derivative_jones_v"].real,
        s["incident_derivative_jones_v"].imag,
        s["incident_basis_u"].x, s["incident_basis_u"].y,
        s["incident_basis_u"].z,
        s["incident_basis_v"].x, s["incident_basis_v"].y,
        s["incident_basis_v"].z,
        s["incident_basis_k"].x, s["incident_basis_k"].y,
        s["incident_basis_k"].z,
        s["face0_operator_m00"].real, s["face0_operator_m00"].imag,
        s["face0_operator_m01"].real, s["face0_operator_m01"].imag,
        s["face0_operator_m10"].real, s["face0_operator_m10"].imag,
        s["face0_operator_m11"].real, s["face0_operator_m11"].imag,
        s["face1_operator_m00"].real, s["face1_operator_m00"].imag,
        s["face1_operator_m01"].real, s["face1_operator_m01"].imag,
        s["face1_operator_m10"].real, s["face1_operator_m10"].imag,
        s["face1_operator_m11"].real, s["face1_operator_m11"].imag,
        s["face0_eta_r"], s["face0_sigma"],
        s["face0_gain"], f0uf, f0pr,
        s["face1_eta_r"], s["face1_sigma"],
        s["face1_gain"], f1uf, f1pr,
    )


def _zero_pair_output_buffers(n_rx: int):
    out_buffers = {
        "direct_re": dr.zeros(wt.Float, n_rx),
        "direct_im": dr.zeros(wt.Float, n_rx),
        "multi_re": dr.zeros(wt.Float, n_rx),
        "multi_im": dr.zeros(wt.Float, n_rx),
        "direct_vec_x_re": dr.zeros(wt.Float, n_rx),
        "direct_vec_x_im": dr.zeros(wt.Float, n_rx),
        "direct_vec_y_re": dr.zeros(wt.Float, n_rx),
        "direct_vec_y_im": dr.zeros(wt.Float, n_rx),
        "direct_vec_z_re": dr.zeros(wt.Float, n_rx),
        "direct_vec_z_im": dr.zeros(wt.Float, n_rx),
        "multi_vec_x_re": dr.zeros(wt.Float, n_rx),
        "multi_vec_x_im": dr.zeros(wt.Float, n_rx),
        "multi_vec_y_re": dr.zeros(wt.Float, n_rx),
        "multi_vec_y_im": dr.zeros(wt.Float, n_rx),
        "multi_vec_z_re": dr.zeros(wt.Float, n_rx),
        "multi_vec_z_im": dr.zeros(wt.Float, n_rx),
    }
    dr.eval(*out_buffers.values())
    return out_buffers


def _pair_output_arrays(out_buffers: dict):
    return (
        out_buffers["direct_re"],
        out_buffers["direct_im"],
        out_buffers["multi_re"],
        out_buffers["multi_im"],
        out_buffers["direct_vec_x_re"],
        out_buffers["direct_vec_x_im"],
        out_buffers["direct_vec_y_re"],
        out_buffers["direct_vec_y_im"],
        out_buffers["direct_vec_z_re"],
        out_buffers["direct_vec_z_im"],
        out_buffers["multi_vec_x_re"],
        out_buffers["multi_vec_x_im"],
        out_buffers["multi_vec_y_re"],
        out_buffers["multi_vec_y_im"],
        out_buffers["multi_vec_z_re"],
        out_buffers["multi_vec_z_im"],
    )


def _pair_vector_power_output_arrays(out_buffers: dict, matched_power, valid_pair_count):
    return (
        out_buffers["direct_re"],
        out_buffers["direct_im"],
        out_buffers["multi_re"],
        out_buffers["multi_im"],
        out_buffers["direct_vec_x_re"],
        out_buffers["direct_vec_x_im"],
        out_buffers["direct_vec_y_re"],
        out_buffers["direct_vec_y_im"],
        out_buffers["direct_vec_z_re"],
        out_buffers["direct_vec_z_im"],
        out_buffers["multi_vec_x_re"],
        out_buffers["multi_vec_x_im"],
        out_buffers["multi_vec_y_re"],
        out_buffers["multi_vec_y_im"],
        out_buffers["multi_vec_z_re"],
        out_buffers["multi_vec_z_im"],
        matched_power,
        valid_pair_count,
    )


def _build_material_params(ext, material_detail, wavelength=None):
    mat = ext.MaterialParams()
    if wavelength is not None:
        mat.omega = float(reflection_material_omega(wavelength)[0])
    if material_detail is not None:
        mat.use_fresnel = int(material_detail.get("use_fresnel", 0))
        mat.eta_r = float(material_detail.get("eta_r", 5.0))
        mat.sigma = float(material_detail.get("sigma", 0.0))
        mat.gain = float(material_detail.get("gain", 1.0))
        mat.omega = float(material_detail.get("omega", mat.omega))
    return mat


def _ownership_codes(state_arrays):
    ownership = wt.Int32(
        _ownership_code_from_depths(
            state_arrays["prefix_reflection_depth"],
            state_arrays["intermediate_reflection_depth"],
            state_arrays["suffix_reflection_depth"],
        )
    )
    return ownership


def _zero_complex(width: int):
    return wt.Complex2f(dr.zeros(wt.Float, width), dr.zeros(wt.Float, width))


def _zero_vector(width: int):
    return {
        "x": _zero_complex(width),
        "y": _zero_complex(width),
        "z": _zero_complex(width),
    }


def _coerce_complex_grad(grad_value, width: int):
    if grad_value is None:
        return _zero_complex(width)
    return wt.Complex2f(grad_value.real, grad_value.imag)


def _coerce_vector_grad(grad_vector, width: int):
    if grad_vector is None:
        return _zero_vector(width)
    return {
        axis: _coerce_complex_grad(grad_vector.get(axis), width)
        for axis in ("x", "y", "z")
    }


_REPLAY_POINT_KEYS = (
    "edge_pos",
    "edge_dir",
    "n0",
    "n_face_n",
    "source_pos",
    "incident_basis_u",
    "incident_basis_v",
    "incident_basis_k",
)
_REPLAY_POINT_CONSTRUCTORS = {
    "edge_pos": wt.Point3f,
    "edge_dir": wt.Vector3f,
    "n0": wt.Vector3f,
    "n_face_n": wt.Vector3f,
    "source_pos": wt.Point3f,
    "incident_basis_u": wt.Vector3f,
    "incident_basis_v": wt.Vector3f,
    "incident_basis_k": wt.Vector3f,
}
_REPLAY_FLOAT_KEYS = (
    "wedge_n",
    "edge_line_min",
    "edge_line_max",
    "face0_eta_r",
    "face0_sigma",
    "face0_gain",
    "face1_eta_r",
    "face1_sigma",
    "face1_gain",
)
_REPLAY_COMPLEX_KEYS = (
    "incident_field",
    "incident_normal_derivative",
    "r_face0",
    "r_face_n",
    "incident_vector_x",
    "incident_vector_y",
    "incident_vector_z",
    "incident_normal_derivative_vector_x",
    "incident_normal_derivative_vector_y",
    "incident_normal_derivative_vector_z",
    "incident_jones_u",
    "incident_jones_v",
    "incident_derivative_jones_u",
    "incident_derivative_jones_v",
    "face0_operator_m00",
    "face0_operator_m01",
    "face0_operator_m10",
    "face0_operator_m11",
    "face1_operator_m00",
    "face1_operator_m01",
    "face1_operator_m10",
    "face1_operator_m11",
)


def _detach_point_value(value):
    return type(value)(dr.detach(value.x), dr.detach(value.y), dr.detach(value.z))


def _detach_complex_value(value):
    return wt.Complex2f(dr.detach(value.real), dr.detach(value.imag))


def _detach_state_arrays_for_replay(state_arrays: dict) -> dict:
    replay_state = dict(state_arrays)
    for key in _REPLAY_POINT_KEYS:
        replay_state[key] = _detach_point_value(state_arrays[key])
    for key in _REPLAY_FLOAT_KEYS:
        replay_state[key] = dr.detach(state_arrays[key])
    for key in _REPLAY_COMPLEX_KEYS:
        replay_state[key] = _detach_complex_value(state_arrays[key])
    return replay_state


def _enable_grad_state_replay_inputs(state_arrays: dict) -> None:
    for key in _REPLAY_POINT_KEYS:
        value = state_arrays[key]
        dr.enable_grad(value.x, value.y, value.z)
    for key in _REPLAY_FLOAT_KEYS:
        dr.enable_grad(state_arrays[key])
    for key in _REPLAY_COMPLEX_KEYS:
        value = state_arrays[key]
        dr.enable_grad(value.real, value.imag)


def _zero_grad_array(width: int):
    return dr.zeros(wt.Float, width)


def _grad_or_zero(value, width: int):
    grad_value = dr.grad(value)
    if grad_value is None:
        return _zero_grad_array(width)
    return grad_value


def _point_grad_or_zero(value):
    return type(value)(
        _grad_or_zero(value.x, dr.width(value.x)),
        _grad_or_zero(value.y, dr.width(value.y)),
        _grad_or_zero(value.z, dr.width(value.z)),
    )


def _complex_grad_or_zero(value):
    return wt.Complex2f(
        _grad_or_zero(value.real, dr.width(value.real)),
        _grad_or_zero(value.imag, dr.width(value.imag)),
    )


def _zero_state_grads(n_states: int) -> dict:
    zero_state = {"n_states": n_states}
    for key in _REPLAY_POINT_KEYS:
        zero_state[key] = _REPLAY_POINT_CONSTRUCTORS[key](
            dr.zeros(wt.Float, n_states),
            dr.zeros(wt.Float, n_states),
            dr.zeros(wt.Float, n_states),
        )
    for key in _REPLAY_FLOAT_KEYS:
        zero_state[key] = dr.zeros(wt.Float, n_states)
    for key in _REPLAY_COMPLEX_KEYS:
        zero_state[key] = wt.Complex2f(
            dr.zeros(wt.Float, n_states),
            dr.zeros(wt.Float, n_states),
        )
    return zero_state


def _state_grads_from_replay_state(state_arrays: dict) -> dict:
    state_grads = {"n_states": int(state_arrays["n_states"])}
    for key in _REPLAY_POINT_KEYS:
        state_grads[key] = _point_grad_or_zero(state_arrays[key])
    for key in _REPLAY_FLOAT_KEYS:
        value = state_arrays[key]
        state_grads[key] = _grad_or_zero(value, dr.width(value))
    for key in _REPLAY_COMPLEX_KEYS:
        state_grads[key] = _complex_grad_or_zero(state_arrays[key])
    return state_grads


def _utd_vjp_loss(
    outputs,
    grad_direct_total,
    grad_multi_total,
    grad_direct_vector,
    grad_multi_vector,
):
    direct_total, multi_total, direct_vector, multi_vector, _ = outputs
    n_rx = dr.width(direct_total.real)
    grad_direct_total = _coerce_complex_grad(grad_direct_total, n_rx)
    grad_multi_total = _coerce_complex_grad(grad_multi_total, n_rx)
    grad_direct_vector = _coerce_vector_grad(grad_direct_vector, n_rx)
    grad_multi_vector = _coerce_vector_grad(grad_multi_vector, n_rx)

    loss = dr.zeros(wt.Float, 1)
    loss += dr.sum(
        direct_total.real * grad_direct_total.real
        + direct_total.imag * grad_direct_total.imag
    )
    loss += dr.sum(
        multi_total.real * grad_multi_total.real
        + multi_total.imag * grad_multi_total.imag
    )
    for axis in ("x", "y", "z"):
        loss += dr.sum(
            direct_vector[axis].real * grad_direct_vector[axis].real
            + direct_vector[axis].imag * grad_direct_vector[axis].imag
        )
        loss += dr.sum(
            multi_vector[axis].real * grad_multi_vector[axis].real
            + multi_vector[axis].imag * grad_multi_vector[axis].imag
        )
    return loss


def _materialize_receiver_positions(rx_pos):
    n_rx = dr.width(rx_pos.x)
    if n_rx == 0:
        return rx_pos
    rx_idx = dr.arange(wt.UInt32, n_rx)
    rx_x = dr.gather(wt.Float, rx_pos.x, rx_idx)
    rx_y = dr.gather(wt.Float, rx_pos.y, rx_idx)
    rx_z = dr.gather(wt.Float, rx_pos.z, rx_idx)
    dr.eval(rx_x, rx_y, rx_z)
    return wt.Point3f(rx_x, rx_y, rx_z)


def _utd_accumulate_tiled_vector_power_into(
    *,
    state_arrays: dict,
    state_idx,
    rx_pos,
    rx_idx,
    valid_mask,
    out_buffers: dict,
    matched_power,
    k: float,
    wavelength: float | None = None,
    material_detail=None,
    rx_polarization=None,
    ownership_code=None,
):
    _require_finite_edge_bounds(
        state_arrays,
        context="Native tiled UTD vector-power accumulation",
    )
    ext = _extension()
    if not hasattr(ext, "utd_accumulate_tiled_vector_power_into"):
        raise RuntimeError(
            "Native tiled UTD vector-power accumulation requires "
            "utd_accumulate_tiled_vector_power_into. Rebuild the witwin.channel native extension."
        )

    n_local_states = int(dr.width(state_idx))
    n_local_receivers = int(dr.width(rx_idx))
    if n_local_states <= 0 or n_local_receivers <= 0:
        return 0

    state_idx_i32 = wt.Int32(state_idx)
    rx_idx_i32 = wt.Int32(rx_idx)
    valid_mask_i32 = None if valid_mask is None else wt.Int32(valid_mask)
    local_ownership = ownership_code
    if local_ownership is None:
        local_ownership = _ownership_codes(state_arrays)
    local_ownership = wt.Int32(local_ownership)

    native_rx_pos = _materialize_receiver_positions(rx_pos)
    active_rx_pol = effective_rx_polarization(rx_polarization, (1.0, 0.0, 0.0))
    mat = _build_material_params(ext, material_detail, wavelength)
    valid_pair_count = dr.zeros(wt.Float, 1)
    output_arrays = _pair_vector_power_output_arrays(
        out_buffers,
        matched_power,
        valid_pair_count,
    )
    eval_targets = [
        state_idx_i32,
        rx_idx_i32,
        local_ownership,
        native_rx_pos.x,
        native_rx_pos.y,
        native_rx_pos.z,
        *output_arrays,
    ]
    if valid_mask_i32 is not None:
        eval_targets.append(valid_mask_i32)
    dr.eval(*eval_targets)
    ext.utd_accumulate_tiled_vector_power_into(
        _detach_native_index_array(state_idx_i32),
        _detach_native_index_array(rx_idx_i32),
        False if valid_mask_i32 is None else _detach_native_index_array(valid_mask_i32),
        _detach_native_index_array(local_ownership),
        _pack_state_soa(state_arrays),
        (native_rx_pos.x, native_rx_pos.y, native_rx_pos.z),
        output_arrays,
        mat,
        n_local_states,
        n_local_receivers,
        k,
        float(active_rx_pol[0]),
        float(active_rx_pol[1]),
        float(active_rx_pol[2]),
    )
    dr.eval(valid_pair_count)
    return int(float(valid_pair_count[0]))


def _gather_eval_state_fields(state_arrays: dict, indices):
    if is_path_export_reduced_state_arrays(state_arrays):
        return gather_path_export_eval_state_fields(state_arrays, indices)
    return gather_state_arrays(state_arrays, indices)


def utd_accumulate_scalar_power_pairs(
    state_arrays: dict,
    pair_rx_pos,
    output_rx_idx,
    *,
    n_output_rx: int,
    k: float,
    wavelength: float | None = None,
    material_detail=None,
    rx_polarization=None,
):
    _require_finite_edge_bounds(
        state_arrays,
        context="Native UTD scalar-power accumulation",
    )
    ext = _extension()
    n_pairs = int(dr.width(output_rx_idx))
    n_rx = int(n_output_rx)
    coherent = ArrayInit.complex_zero(n_rx)
    power = dr.zeros(wt.Float, n_rx)
    if n_pairs <= 0 or n_rx <= 0:
        dr.eval(coherent.real, coherent.imag, power)
        return eval_complex(coherent), power, 0

    native_pair_rx_pos = _materialize_receiver_positions(pair_rx_pos)
    active_rx_pol = effective_rx_polarization(rx_polarization, (1.0, 0.0, 0.0))
    mat = _build_material_params(ext, material_detail, wavelength)
    coherent_re = coherent.real
    coherent_im = coherent.imag
    valid_pair_count = dr.zeros(wt.Float, 1)
    dr.eval(output_rx_idx, coherent_re, coherent_im, power, valid_pair_count)

    chunk_size = min(_UTD_SCALAR_POWER_PAIR_CHUNK, n_pairs)
    for pair_start in range(0, n_pairs, chunk_size):
        pair_count = min(chunk_size, n_pairs - pair_start)
        pair_idx = dr.arange(wt.UInt32, pair_count) + wt.UInt32(pair_start)
        chunk_output_rx_idx = dr.gather(wt.UInt32, output_rx_idx, pair_idx)
        chunk_state = _gather_eval_state_fields(state_arrays, pair_idx)
        chunk_pair_rx = wt.Point3f(
            dr.gather(wt.Float, native_pair_rx_pos.x, pair_idx),
            dr.gather(wt.Float, native_pair_rx_pos.y, pair_idx),
            dr.gather(wt.Float, native_pair_rx_pos.z, pair_idx),
        )
        chunk_soa = _pack_state_soa(chunk_state)
        (
            chunk_coherent_re,
            chunk_coherent_im,
            chunk_power,
            chunk_valid_pair_count,
        ) = ext.utd_accumulate_scalar_power_arrays(
            _detach_native_index_array(wt.Int32(chunk_output_rx_idx)),
            chunk_soa,
            (chunk_pair_rx.x, chunk_pair_rx.y, chunk_pair_rx.z),
            mat,
            n_rx,
            pair_count,
            k,
            float(active_rx_pol[0]),
            float(active_rx_pol[1]),
            float(active_rx_pol[2]),
        )
        coherent_re = coherent_re + chunk_coherent_re
        coherent_im = coherent_im + chunk_coherent_im
        power = power + chunk_power
        valid_pair_count = valid_pair_count + chunk_valid_pair_count

    dr.eval(coherent_re, coherent_im, power, valid_pair_count)
    return (
        eval_complex(wt.Complex2f(coherent_re, coherent_im)),
        power,
        int(float(valid_pair_count[0])),
    )


def _add_vector_grads(lhs, rhs):
    return {
        axis: wt.Complex2f(
            lhs[axis].real + rhs[axis].real,
            lhs[axis].imag + rhs[axis].imag,
        )
        for axis in ("x", "y", "z")
    }


def _vector_grad_from_scalar_output(primal_vector, grad_scalar, active_rx_pol, receiver_axis):
    grad_scalar = _coerce_complex_grad(grad_scalar, dr.width(primal_vector["x"].real))
    if dr.width(grad_scalar.real) == 0:
        return _zero_vector(0)
    if bool(dr.all((grad_scalar.real == 0) & (grad_scalar.imag == 0))):
        return _zero_vector(dr.width(grad_scalar.real))

    vxr = dr.detach(primal_vector["x"].real)
    vxi = dr.detach(primal_vector["x"].imag)
    vyr = dr.detach(primal_vector["y"].real)
    vyi = dr.detach(primal_vector["y"].imag)
    vzr = dr.detach(primal_vector["z"].real)
    vzi = dr.detach(primal_vector["z"].imag)
    for arr in (vxr, vxi, vyr, vyi, vzr, vzi):
        dr.enable_grad(arr)

    vector_value = {
        "x": wt.Complex2f(vxr, vxi),
        "y": wt.Complex2f(vyr, vyi),
        "z": wt.Complex2f(vzr, vzi),
    }
    scalar_value = scalarize_tangential_jones(
        tangential_jones(vector_value, axis=receiver_axis),
        active_rx_pol,
        axis=receiver_axis,
    )
    loss = dr.sum(
        scalar_value.real * grad_scalar.real
        + scalar_value.imag * grad_scalar.imag
    )
    dr.backward(loss)
    return {
        "x": wt.Complex2f(dr.grad(vxr), dr.grad(vxi)),
        "y": wt.Complex2f(dr.grad(vyr), dr.grad(vyi)),
        "z": wt.Complex2f(dr.grad(vzr), dr.grad(vzi)),
    }


def _utd_accumulate_forward_native_primal(
    state_arrays: dict,
    rx_pos,
    k: float,
    n_edges: int,
    return_per_edge: bool,
    *,
    scene=None,
    wavelength: float | None = None,
    material_detail=None,
    rx_polarization=None,
    receiver_axis: str = "z",
    execution=None,
):
    """
    Native CUDA path for UTD diffraction accumulation.

    Same signature as ``drjit_impl.utd_accumulate_forward``.
    """
    from witwin.channel.trace.diffraction.geometry import (
        _edge_owner_structure_idx,
        _segment_visibility_mask,
    )

    _require_finite_edge_bounds(
        state_arrays,
        context="Native UTD accumulation",
    )
    ext = _extension()
    execution = coerce_diffraction_execution(execution)
    n_states = state_arrays["n_states"]
    n_rx = dr.width(rx_pos.x)
    active_rx_pol = (1.0, 0.0, 0.0) if rx_polarization is None else rx_polarization
    native_rx_pos = _materialize_receiver_positions(rx_pos)

    if n_states == 0 or n_rx == 0:
        zf = ArrayInit.complex_zero(n_rx)
        zv = vector_zero(n_rx)
        pe = [(zf.real, zf.imag) for _ in range(n_edges)] if return_per_edge else []
        return zf, zf, zv, zv, pe

    mat = _build_material_params(ext, material_detail, wavelength)

    out_buffers = _zero_pair_output_buffers(n_rx)
    output_arrays = _pair_output_arrays(out_buffers)

    # Pack state SoA pointers once; chunking only limits temporary visibility masks.
    soa = _pack_state_soa(state_arrays)
    ownership = _ownership_codes(state_arrays)

    chunk_size = _cartesian_chunk_size(n_states, n_rx)
    full_rx_idx = wt.Int32(dr.arange(wt.UInt32, n_rx))

    for state_start in range(0, n_states, chunk_size):
        chunk_n = min(chunk_size, n_states - state_start)
        chunk_state_idx = wt.Int32(dr.arange(wt.UInt32, chunk_n) + wt.UInt32(state_start))
        valid_mask = None
        if scene is not None:
            n_pairs = chunk_n * n_rx
            pair_idx = dr.arange(wt.UInt32, n_pairs)
            state_idx = pair_idx // n_rx + wt.UInt32(state_start)
            rx_idx = pair_idx % n_rx
            edge_pos = dr.gather(wt.Point3f, state_arrays["edge_pos"], state_idx)
            adj0 = dr.gather(wt.Int32, state_arrays["adjacent_face0"], state_idx)
            adj1 = dr.gather(wt.Int32, state_arrays["adjacent_face1"], state_idx)
            owner_structure_idx = _edge_owner_structure_idx(scene, adj0, adj1)
            batch_rx = wt.Point3f(
                dr.gather(wt.Float, native_rx_pos.x, rx_idx),
                dr.gather(wt.Float, native_rx_pos.y, rx_idx),
                dr.gather(wt.Float, native_rx_pos.z, rx_idx),
            )
            visible = _segment_visibility_mask(
                edge_pos, batch_rx, scene,
                ignore_prim_idx=(adj0, adj1),
                ignore_structure_idx=owner_structure_idx,
            )
            valid_mask = dr.select(visible, wt.Int32(1), wt.Int32(0))
            if not bool(dr.any(visible)):
                continue

        chunk_outputs = ext.utd_accumulate_tiled_arrays_v2(
            _detach_native_index_array(chunk_state_idx),
            _detach_native_index_array(full_rx_idx),
            False if valid_mask is None else _detach_native_index_array(valid_mask),
            _detach_native_index_array(ownership),
            soa,
            (native_rx_pos.x, native_rx_pos.y, native_rx_pos.z),
            mat,
            chunk_n,
            n_rx,
            k,
        )
        for key, chunk_value in zip(out_buffers.keys(), chunk_outputs):
            out_buffers[key] = out_buffers[key] + chunk_value

    # Wrap raw buffers into Complex/Vector dicts
    direct_vector_total = {
        "x": wt.Complex2f(out_buffers["direct_vec_x_re"], out_buffers["direct_vec_x_im"]),
        "y": wt.Complex2f(out_buffers["direct_vec_y_re"], out_buffers["direct_vec_y_im"]),
        "z": wt.Complex2f(out_buffers["direct_vec_z_re"], out_buffers["direct_vec_z_im"]),
    }
    multi_vector_total = {
        "x": wt.Complex2f(out_buffers["multi_vec_x_re"], out_buffers["multi_vec_x_im"]),
        "y": wt.Complex2f(out_buffers["multi_vec_y_re"], out_buffers["multi_vec_y_im"]),
        "z": wt.Complex2f(out_buffers["multi_vec_z_re"], out_buffers["multi_vec_z_im"]),
    }
    # Scalarize (pure Python post-processing, no scatter)
    direct_total = (
        scalarize_tangential_jones(
            tangential_jones(direct_vector_total, axis=receiver_axis),
            active_rx_pol, axis=receiver_axis,
        )
    )
    multi_total = (
        scalarize_tangential_jones(
            tangential_jones(multi_vector_total, axis=receiver_axis),
            active_rx_pol, axis=receiver_axis,
        )
    )
    # per_edge not yet supported in native path
    per_edge_list = []
    return direct_total, multi_total, direct_vector_total, multi_vector_total, per_edge_list


def utd_accumulate_forward(
    state_arrays: dict,
    rx_pos,
    k: float,
    n_edges: int,
    return_per_edge: bool,
    *,
    scene=None,
    wavelength: float | None = None,
    material_detail=None,
    rx_polarization=None,
    receiver_axis: str = "z",
    execution=None,
    receiver_tiles=None,
):
    _require_finite_edge_bounds(
        state_arrays,
        context="Native UTD accumulation",
    )
    # Receiver-tiled UTD rollout has been detached from the main execution
    # path. Keep the public forward on the validated Dr.Jit finite-wedge
    # evaluator until the native full-cartesian kernel path is revalidated.
    del receiver_tiles
    from . import drjit_impl

    return drjit_impl.utd_accumulate_forward(
        state_arrays,
        rx_pos,
        k,
        n_edges,
        return_per_edge,
        scene=scene,
        wavelength=wavelength,
        material_detail=material_detail,
        rx_polarization=rx_polarization,
        receiver_axis=receiver_axis,
        execution=execution,
    )


def utd_accumulate_backward(
    state_arrays: dict,
    rx_pos,
    k: float,
    n_edges: int,
    return_per_edge: bool,
    grad_direct_total,
    grad_multi_total,
    grad_direct_vector,
    grad_multi_vector,
    *,
    scene=None,
    wavelength: float | None = None,
    material_detail=None,
    rx_polarization=None,
    receiver_axis: str = "z",
    execution=None,
    receiver_tiles=None,
):
    """Explicit VJP for the native UTD forward path.

    Returns
    -------
    tuple[dict, wt.Point3f]
        ``(state_grads, rx_pos_grads)`` where ``state_grads`` contains the
        differentiable SoA fields supported by the native backward kernel.
    """
    _require_finite_edge_bounds(
        state_arrays,
        context="Native explicit UTD backward",
    )
    del receiver_tiles
    from . import drjit_impl

    n_states = int(state_arrays["n_states"])
    n_rx = dr.width(rx_pos.x)
    if n_states == 0 or n_rx == 0:
        return _zero_state_grads(n_states), wt.Point3f(
            dr.zeros(wt.Float, n_rx),
            dr.zeros(wt.Float, n_rx),
            dr.zeros(wt.Float, n_rx),
        )

    replay_state = _detach_state_arrays_for_replay(state_arrays)
    replay_rx_pos = wt.Point3f(
        dr.detach(rx_pos.x),
        dr.detach(rx_pos.y),
        dr.detach(rx_pos.z),
    )
    _enable_grad_state_replay_inputs(replay_state)
    dr.enable_grad(replay_rx_pos.x, replay_rx_pos.y, replay_rx_pos.z)

    outputs = drjit_impl.utd_accumulate_forward(
        replay_state,
        replay_rx_pos,
        k,
        n_edges,
        return_per_edge,
        scene=scene,
        wavelength=wavelength,
        material_detail=material_detail,
        rx_polarization=rx_polarization,
        receiver_axis=receiver_axis,
        execution=execution,
    )
    dr.backward(
        _utd_vjp_loss(
            outputs,
            grad_direct_total,
            grad_multi_total,
            grad_direct_vector,
            grad_multi_vector,
        ),
        flags=dr.ADFlag.Default | dr.ADFlag.AllowNoGrad,
    )
    return _state_grads_from_replay_state(replay_state), _point_grad_or_zero(replay_rx_pos)
    from witwin.channel.trace.diffraction.geometry import (
        _edge_owner_structure_idx,
        _segment_visibility_mask,
    )

    receiver_tiles = resolve_receiver_tiles(
        grid=None,
        receiver_positions=rx_pos,
        receiver_tiles=receiver_tiles,
    )
    ext = _extension()
    execution = coerce_diffraction_execution(execution)
    del return_per_edge

    n_states = state_arrays["n_states"]
    n_rx = dr.width(rx_pos.x)
    native_rx_pos = _materialize_receiver_positions(rx_pos)
    if n_states == 0 or n_rx == 0:
        zero_state = {
            "n_states": n_states,
            "edge_pos": wt.Point3f(dr.zeros(wt.Float, n_states), dr.zeros(wt.Float, n_states), dr.zeros(wt.Float, n_states)),
            "edge_dir": wt.Vector3f(dr.zeros(wt.Float, n_states), dr.zeros(wt.Float, n_states), dr.zeros(wt.Float, n_states)),
            "n0": wt.Vector3f(dr.zeros(wt.Float, n_states), dr.zeros(wt.Float, n_states), dr.zeros(wt.Float, n_states)),
            "n_face_n": wt.Vector3f(dr.zeros(wt.Float, n_states), dr.zeros(wt.Float, n_states), dr.zeros(wt.Float, n_states)),
            "wedge_n": dr.zeros(wt.Float, n_states),
            "source_pos": wt.Point3f(dr.zeros(wt.Float, n_states), dr.zeros(wt.Float, n_states), dr.zeros(wt.Float, n_states)),
            "incident_field": wt.Complex2f(dr.zeros(wt.Float, n_states), dr.zeros(wt.Float, n_states)),
            "incident_normal_derivative": wt.Complex2f(dr.zeros(wt.Float, n_states), dr.zeros(wt.Float, n_states)),
            "r_face0": wt.Complex2f(dr.zeros(wt.Float, n_states), dr.zeros(wt.Float, n_states)),
            "r_face_n": wt.Complex2f(dr.zeros(wt.Float, n_states), dr.zeros(wt.Float, n_states)),
            "incident_vector_x": wt.Complex2f(dr.zeros(wt.Float, n_states), dr.zeros(wt.Float, n_states)),
            "incident_vector_y": wt.Complex2f(dr.zeros(wt.Float, n_states), dr.zeros(wt.Float, n_states)),
            "incident_vector_z": wt.Complex2f(dr.zeros(wt.Float, n_states), dr.zeros(wt.Float, n_states)),
            "incident_normal_derivative_vector_x": wt.Complex2f(dr.zeros(wt.Float, n_states), dr.zeros(wt.Float, n_states)),
            "incident_normal_derivative_vector_y": wt.Complex2f(dr.zeros(wt.Float, n_states), dr.zeros(wt.Float, n_states)),
            "incident_normal_derivative_vector_z": wt.Complex2f(dr.zeros(wt.Float, n_states), dr.zeros(wt.Float, n_states)),
            "face0_operator_m00": wt.Complex2f(dr.zeros(wt.Float, n_states), dr.zeros(wt.Float, n_states)),
            "face0_operator_m01": wt.Complex2f(dr.zeros(wt.Float, n_states), dr.zeros(wt.Float, n_states)),
            "face0_operator_m10": wt.Complex2f(dr.zeros(wt.Float, n_states), dr.zeros(wt.Float, n_states)),
            "face0_operator_m11": wt.Complex2f(dr.zeros(wt.Float, n_states), dr.zeros(wt.Float, n_states)),
            "face1_operator_m00": wt.Complex2f(dr.zeros(wt.Float, n_states), dr.zeros(wt.Float, n_states)),
            "face1_operator_m01": wt.Complex2f(dr.zeros(wt.Float, n_states), dr.zeros(wt.Float, n_states)),
            "face1_operator_m10": wt.Complex2f(dr.zeros(wt.Float, n_states), dr.zeros(wt.Float, n_states)),
            "face1_operator_m11": wt.Complex2f(dr.zeros(wt.Float, n_states), dr.zeros(wt.Float, n_states)),
            "face0_eta_r": dr.zeros(wt.Float, n_states),
            "face0_sigma": dr.zeros(wt.Float, n_states),
            "face0_gain": dr.zeros(wt.Float, n_states),
            "face1_eta_r": dr.zeros(wt.Float, n_states),
            "face1_sigma": dr.zeros(wt.Float, n_states),
            "face1_gain": dr.zeros(wt.Float, n_states),
        }
        zero_rx = wt.Point3f(dr.zeros(wt.Float, n_rx), dr.zeros(wt.Float, n_rx), dr.zeros(wt.Float, n_rx))
        return zero_state, zero_rx

    active_rx_pol = (1.0, 0.0, 0.0) if rx_polarization is None else rx_polarization
    _, _, direct_vector_total, multi_vector_total, _ = _utd_accumulate_forward_native_primal(
        state_arrays,
        rx_pos,
        k,
        n_edges,
        False,
        scene=scene,
        wavelength=wavelength,
        material_detail=material_detail,
        rx_polarization=rx_polarization,
        receiver_axis=receiver_axis,
        execution=execution,
    )
    direct_vector_grad = _add_vector_grads(
        _coerce_vector_grad(grad_direct_vector, n_rx),
        _vector_grad_from_scalar_output(
            direct_vector_total, grad_direct_total, active_rx_pol, receiver_axis
        ),
    )
    multi_vector_grad = _add_vector_grads(
        _coerce_vector_grad(grad_multi_vector, n_rx),
        _vector_grad_from_scalar_output(
            multi_vector_total, grad_multi_total, active_rx_pol, receiver_axis
        ),
    )

    mat = _build_material_params(ext, material_detail, wavelength)
    soa = _pack_state_soa(state_arrays)
    ownership = _ownership_codes(state_arrays)

    g_epx = dr.zeros(wt.Float, n_states); g_epy = dr.zeros(wt.Float, n_states); g_epz = dr.zeros(wt.Float, n_states)
    g_edx = dr.zeros(wt.Float, n_states); g_edy = dr.zeros(wt.Float, n_states); g_edz = dr.zeros(wt.Float, n_states)
    g_n0x = dr.zeros(wt.Float, n_states); g_n0y = dr.zeros(wt.Float, n_states); g_n0z = dr.zeros(wt.Float, n_states)
    g_nnx = dr.zeros(wt.Float, n_states); g_nny = dr.zeros(wt.Float, n_states); g_nnz = dr.zeros(wt.Float, n_states)
    g_wn = dr.zeros(wt.Float, n_states)
    g_spx = dr.zeros(wt.Float, n_states); g_spy = dr.zeros(wt.Float, n_states); g_spz = dr.zeros(wt.Float, n_states)
    g_ifr = dr.zeros(wt.Float, n_states); g_ifi = dr.zeros(wt.Float, n_states)
    g_inr = dr.zeros(wt.Float, n_states); g_ini = dr.zeros(wt.Float, n_states)
    g_r0r = dr.zeros(wt.Float, n_states); g_r0i = dr.zeros(wt.Float, n_states)
    g_rnr = dr.zeros(wt.Float, n_states); g_rni = dr.zeros(wt.Float, n_states)
    g_vxr = dr.zeros(wt.Float, n_states); g_vxi = dr.zeros(wt.Float, n_states)
    g_vyr = dr.zeros(wt.Float, n_states); g_vyi = dr.zeros(wt.Float, n_states)
    g_vzr = dr.zeros(wt.Float, n_states); g_vzi = dr.zeros(wt.Float, n_states)
    g_dxr = dr.zeros(wt.Float, n_states); g_dxi = dr.zeros(wt.Float, n_states)
    g_dyr = dr.zeros(wt.Float, n_states); g_dyi = dr.zeros(wt.Float, n_states)
    g_dzr = dr.zeros(wt.Float, n_states); g_dzi = dr.zeros(wt.Float, n_states)
    g_f0m00r = dr.zeros(wt.Float, n_states); g_f0m00i = dr.zeros(wt.Float, n_states)
    g_f0m01r = dr.zeros(wt.Float, n_states); g_f0m01i = dr.zeros(wt.Float, n_states)
    g_f0m10r = dr.zeros(wt.Float, n_states); g_f0m10i = dr.zeros(wt.Float, n_states)
    g_f0m11r = dr.zeros(wt.Float, n_states); g_f0m11i = dr.zeros(wt.Float, n_states)
    g_f1m00r = dr.zeros(wt.Float, n_states); g_f1m00i = dr.zeros(wt.Float, n_states)
    g_f1m01r = dr.zeros(wt.Float, n_states); g_f1m01i = dr.zeros(wt.Float, n_states)
    g_f1m10r = dr.zeros(wt.Float, n_states); g_f1m10i = dr.zeros(wt.Float, n_states)
    g_f1m11r = dr.zeros(wt.Float, n_states); g_f1m11i = dr.zeros(wt.Float, n_states)
    g_f0eta = dr.zeros(wt.Float, n_states); g_f0sigma = dr.zeros(wt.Float, n_states); g_f0gain = dr.zeros(wt.Float, n_states)
    g_f1eta = dr.zeros(wt.Float, n_states); g_f1sigma = dr.zeros(wt.Float, n_states); g_f1gain = dr.zeros(wt.Float, n_states)
    g_rxx = dr.zeros(wt.Float, n_rx); g_rxy = dr.zeros(wt.Float, n_rx); g_rxz = dr.zeros(wt.Float, n_rx)
    zero_scalar = dr.zeros(wt.Float, n_rx)
    dr.eval(
        native_rx_pos.x, native_rx_pos.y, native_rx_pos.z,
        direct_vector_grad["x"].real, direct_vector_grad["x"].imag,
        direct_vector_grad["y"].real, direct_vector_grad["y"].imag,
        direct_vector_grad["z"].real, direct_vector_grad["z"].imag,
        multi_vector_grad["x"].real, multi_vector_grad["x"].imag,
        multi_vector_grad["y"].real, multi_vector_grad["y"].imag,
        multi_vector_grad["z"].real, multi_vector_grad["z"].imag,
        g_epx, g_epy, g_epz, g_edx, g_edy, g_edz,
        g_n0x, g_n0y, g_n0z, g_nnx, g_nny, g_nnz, g_wn,
        g_spx, g_spy, g_spz, g_ifr, g_ifi, g_inr, g_ini,
        g_r0r, g_r0i, g_rnr, g_rni,
        g_vxr, g_vxi, g_vyr, g_vyi, g_vzr, g_vzi,
        g_dxr, g_dxi, g_dyr, g_dyi, g_dzr, g_dzi,
        g_f0m00r, g_f0m00i, g_f0m01r, g_f0m01i,
        g_f0m10r, g_f0m10i, g_f0m11r, g_f0m11i,
        g_f1m00r, g_f1m00i, g_f1m01r, g_f1m01i,
        g_f1m10r, g_f1m10i, g_f1m11r, g_f1m11i,
        g_f0eta, g_f0sigma, g_f0gain,
        g_f1eta, g_f1sigma, g_f1gain,
        g_rxx, g_rxy, g_rxz, zero_scalar,
    )
    chunk_size = _cartesian_chunk_size(n_states, n_rx)
    for state_start in range(0, n_states, chunk_size):
        chunk_n = min(chunk_size, n_states - state_start)
        n_pairs = chunk_n * n_rx
        pair_idx = dr.arange(wt.UInt32, n_pairs)
        state_idx = pair_idx // n_rx + wt.UInt32(state_start)
        rx_idx = pair_idx % n_rx

        if scene is not None:
            edge_pos = dr.gather(wt.Point3f, state_arrays["edge_pos"], state_idx)
            adj0 = dr.gather(wt.Int32, state_arrays["adjacent_face0"], state_idx)
            adj1 = dr.gather(wt.Int32, state_arrays["adjacent_face1"], state_idx)
            owner_structure_idx = _edge_owner_structure_idx(scene, adj0, adj1)
            batch_rx = wt.Point3f(
                dr.gather(wt.Float, native_rx_pos.x, rx_idx),
                dr.gather(wt.Float, native_rx_pos.y, rx_idx),
                dr.gather(wt.Float, native_rx_pos.z, rx_idx),
            )
            visible = _segment_visibility_mask(
                edge_pos,
                batch_rx,
                scene,
                ignore_prim_idx=(adj0, adj1),
                ignore_structure_idx=owner_structure_idx,
            )
            state_idx, rx_idx = compact_index_pairs(state_idx, rx_idx, visible)
            if dr.width(state_idx) == 0:
                continue

        n_p = dr.width(state_idx)
        state_idx_i = wt.Int32(state_idx)
        rx_idx_i = wt.Int32(rx_idx)
        chunk_grads = ext.utd_accumulate_backward_arrays(
            state_idx_i,
            rx_idx_i,
            ownership,
            *soa,
            native_rx_pos.x,
            native_rx_pos.y,
            native_rx_pos.z,
            zero_scalar,
            zero_scalar,
            zero_scalar,
            zero_scalar,
            direct_vector_grad["x"].real,
            direct_vector_grad["x"].imag,
            direct_vector_grad["y"].real,
            direct_vector_grad["y"].imag,
            direct_vector_grad["z"].real,
            direct_vector_grad["z"].imag,
            multi_vector_grad["x"].real,
            multi_vector_grad["x"].imag,
            multi_vector_grad["y"].real,
            multi_vector_grad["y"].imag,
            multi_vector_grad["z"].real,
            multi_vector_grad["z"].imag,
            n_p,
            k,
            mat,
        )
        g_epx = g_epx + chunk_grads[0]; g_epy = g_epy + chunk_grads[1]; g_epz = g_epz + chunk_grads[2]
        g_edx = g_edx + chunk_grads[3]; g_edy = g_edy + chunk_grads[4]; g_edz = g_edz + chunk_grads[5]
        g_n0x = g_n0x + chunk_grads[6]; g_n0y = g_n0y + chunk_grads[7]; g_n0z = g_n0z + chunk_grads[8]
        g_nnx = g_nnx + chunk_grads[9]; g_nny = g_nny + chunk_grads[10]; g_nnz = g_nnz + chunk_grads[11]
        g_wn = g_wn + chunk_grads[12]
        g_spx = g_spx + chunk_grads[13]; g_spy = g_spy + chunk_grads[14]; g_spz = g_spz + chunk_grads[15]
        g_ifr = g_ifr + chunk_grads[16]; g_ifi = g_ifi + chunk_grads[17]
        g_inr = g_inr + chunk_grads[18]; g_ini = g_ini + chunk_grads[19]
        g_r0r = g_r0r + chunk_grads[20]; g_r0i = g_r0i + chunk_grads[21]
        g_rnr = g_rnr + chunk_grads[22]; g_rni = g_rni + chunk_grads[23]
        g_vxr = g_vxr + chunk_grads[24]; g_vxi = g_vxi + chunk_grads[25]
        g_vyr = g_vyr + chunk_grads[26]; g_vyi = g_vyi + chunk_grads[27]
        g_vzr = g_vzr + chunk_grads[28]; g_vzi = g_vzi + chunk_grads[29]
        g_dxr = g_dxr + chunk_grads[30]; g_dxi = g_dxi + chunk_grads[31]
        g_dyr = g_dyr + chunk_grads[32]; g_dyi = g_dyi + chunk_grads[33]
        g_dzr = g_dzr + chunk_grads[34]; g_dzi = g_dzi + chunk_grads[35]
        g_f0m00r = g_f0m00r + chunk_grads[36]; g_f0m00i = g_f0m00i + chunk_grads[37]
        g_f0m01r = g_f0m01r + chunk_grads[38]; g_f0m01i = g_f0m01i + chunk_grads[39]
        g_f0m10r = g_f0m10r + chunk_grads[40]; g_f0m10i = g_f0m10i + chunk_grads[41]
        g_f0m11r = g_f0m11r + chunk_grads[42]; g_f0m11i = g_f0m11i + chunk_grads[43]
        g_f1m00r = g_f1m00r + chunk_grads[44]; g_f1m00i = g_f1m00i + chunk_grads[45]
        g_f1m01r = g_f1m01r + chunk_grads[46]; g_f1m01i = g_f1m01i + chunk_grads[47]
        g_f1m10r = g_f1m10r + chunk_grads[48]; g_f1m10i = g_f1m10i + chunk_grads[49]
        g_f1m11r = g_f1m11r + chunk_grads[50]; g_f1m11i = g_f1m11i + chunk_grads[51]
        g_f0eta = g_f0eta + chunk_grads[52]; g_f0sigma = g_f0sigma + chunk_grads[53]; g_f0gain = g_f0gain + chunk_grads[54]
        g_f1eta = g_f1eta + chunk_grads[55]; g_f1sigma = g_f1sigma + chunk_grads[56]; g_f1gain = g_f1gain + chunk_grads[57]
        g_rxx = g_rxx + chunk_grads[58]; g_rxy = g_rxy + chunk_grads[59]; g_rxz = g_rxz + chunk_grads[60]

    state_grads = {
        "n_states": n_states,
        "edge_pos": wt.Point3f(g_epx, g_epy, g_epz),
        "edge_dir": wt.Vector3f(g_edx, g_edy, g_edz),
        "n0": wt.Vector3f(g_n0x, g_n0y, g_n0z),
        "n_face_n": wt.Vector3f(g_nnx, g_nny, g_nnz),
        "wedge_n": g_wn,
        "source_pos": wt.Point3f(g_spx, g_spy, g_spz),
        "incident_field": wt.Complex2f(g_ifr, g_ifi),
        "incident_normal_derivative": wt.Complex2f(g_inr, g_ini),
        "r_face0": wt.Complex2f(g_r0r, g_r0i),
        "r_face_n": wt.Complex2f(g_rnr, g_rni),
        "incident_vector_x": wt.Complex2f(g_vxr, g_vxi),
        "incident_vector_y": wt.Complex2f(g_vyr, g_vyi),
        "incident_vector_z": wt.Complex2f(g_vzr, g_vzi),
        "incident_normal_derivative_vector_x": wt.Complex2f(g_dxr, g_dxi),
        "incident_normal_derivative_vector_y": wt.Complex2f(g_dyr, g_dyi),
        "incident_normal_derivative_vector_z": wt.Complex2f(g_dzr, g_dzi),
        "face0_operator_m00": wt.Complex2f(g_f0m00r, g_f0m00i),
        "face0_operator_m01": wt.Complex2f(g_f0m01r, g_f0m01i),
        "face0_operator_m10": wt.Complex2f(g_f0m10r, g_f0m10i),
        "face0_operator_m11": wt.Complex2f(g_f0m11r, g_f0m11i),
        "face1_operator_m00": wt.Complex2f(g_f1m00r, g_f1m00i),
        "face1_operator_m01": wt.Complex2f(g_f1m01r, g_f1m01i),
        "face1_operator_m10": wt.Complex2f(g_f1m10r, g_f1m10i),
        "face1_operator_m11": wt.Complex2f(g_f1m11r, g_f1m11i),
        "face0_eta_r": g_f0eta,
        "face0_sigma": g_f0sigma,
        "face0_gain": g_f0gain,
        "face1_eta_r": g_f1eta,
        "face1_sigma": g_f1sigma,
        "face1_gain": g_f1gain,
    }
    rx_grads = wt.Point3f(g_rxx, g_rxy, g_rxz)
    return state_grads, rx_grads

