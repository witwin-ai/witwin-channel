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

from witwin.channel.deterministic import types as wt
from witwin.channel._native.deterministic import NativeExtension
from witwin.channel.core.numerics.arrays import (
    complex_zero,
    eval_complex,
)
from witwin.channel.core.physics.polarization import (
    jones_tangential,
    scalarize_tangential_jones,
    vector_select,
    vector_zero,
)
from witwin.channel.core.runtime import Material, Rx, Tx, Wave, material_angular_frequency
from witwin.channel.core.geometry.diffraction import wedge_exterior_mask, wedge_geometry
from witwin.channel.core.physics.wave_math import shadow_support_angle_from_cutoff_db
from witwin.channel.deterministic.diffraction.state import (
    DIFFRACTION_MIN_DISTANCE,
    Geo,
    SOURCE_TYPE_DIRECT_TX,
)
from witwin.channel.deterministic.config import coerce_diffraction_execution


_VALID_PAIR_FLAG = 1


def _state_arrays_have_finite_edge_bounds(state_arrays: dict) -> bool:
    return (
        state_arrays is not None
        and state_arrays.get("edge_line_min") is not None
        and state_arrays.get("edge_line_max") is not None
    )


def _value_has_grad(value) -> bool:
    if value is None:
        return False
    if isinstance(value, dict):
        return any(_value_has_grad(item) for item in value.values())
    try:
        components = tuple(getattr(value, axis) for axis in ("x", "y", "z"))
    except Exception:
        components = None
    if components is not None:
        return any(_value_has_grad(component) for component in components)
    try:
        real = value.real
        imag = value.imag
    except Exception:
        real = imag = None
    if real is not None and imag is not None and real is not value and imag is not value:
        return _value_has_grad(real) or _value_has_grad(imag)
    try:
        return bool(dr.grad_enabled(value))
    except TypeError:
        return False


def _state_arrays_have_material_grad(state_arrays: dict) -> bool:
    material_keys = (
        "face0_eta_r",
        "face0_mu_r",
        "face0_sigma",
        "face0_gain",
        "face1_eta_r",
        "face1_mu_r",
        "face1_sigma",
        "face1_gain",
    )
    return any(_value_has_grad(state_arrays.get(key)) for key in material_keys)


def _require_finite_edge_bounds(state_arrays: dict, *, context: str) -> None:
    if _state_arrays_have_finite_edge_bounds(state_arrays):
        return
    raise RuntimeError(
        f"{context} requires finite-wedge state arrays with edge_line_min and edge_line_max."
    )


def _gather_pair_state(state_arrays: dict, state_idx) -> dict:
    return {
        "edge_idx": dr.gather(wt.UInt32, state_arrays["edge_idx"], state_idx),
        "edge_pos": dr.gather(wt.Point3f, state_arrays["edge_pos"], state_idx),
        "edge_dir": dr.gather(wt.Vector3f, state_arrays["edge_dir"], state_idx),
        "n0": dr.gather(wt.Vector3f, state_arrays["n0"], state_idx),
        "n_face_n": dr.gather(wt.Vector3f, state_arrays["n_face_n"], state_idx),
        "wedge_n": dr.gather(wt.Float, state_arrays["wedge_n"], state_idx),
        "edge_line_min": dr.gather(wt.Float, state_arrays["edge_line_min"], state_idx),
        "edge_line_max": dr.gather(wt.Float, state_arrays["edge_line_max"], state_idx),
        "adjacent_face0": dr.gather(wt.Int32, state_arrays["adjacent_face0"], state_idx),
        "adjacent_face1": dr.gather(wt.Int32, state_arrays["adjacent_face1"], state_idx),
        "source_pos": dr.gather(wt.Point3f, state_arrays["source_pos"], state_idx),
        "incident_field": dr.gather(wt.Complex2f, state_arrays["incident_field"], state_idx),
        "incident_normal_derivative": dr.gather(
            wt.Complex2f,
            state_arrays["incident_normal_derivative"],
            state_idx,
        ),
        "incident_jones_u": dr.gather(wt.Complex2f, state_arrays["incident_jones_u"], state_idx),
        "incident_jones_v": dr.gather(wt.Complex2f, state_arrays["incident_jones_v"], state_idx),
        "incident_derivative_jones_u": dr.gather(
            wt.Complex2f,
            state_arrays["incident_derivative_jones_u"],
            state_idx,
        ),
        "incident_derivative_jones_v": dr.gather(
            wt.Complex2f,
            state_arrays["incident_derivative_jones_v"],
            state_idx,
        ),
        "incident_basis_u": dr.gather(wt.Vector3f, state_arrays["incident_basis_u"], state_idx),
        "incident_basis_v": dr.gather(wt.Vector3f, state_arrays["incident_basis_v"], state_idx),
        "incident_basis_k": dr.gather(wt.Vector3f, state_arrays["incident_basis_k"], state_idx),
        "r_face0": dr.gather(wt.Complex2f, state_arrays["r_face0"], state_idx),
        "r_face_n": dr.gather(wt.Complex2f, state_arrays["r_face_n"], state_idx),
        "face0_operator_m00": dr.gather(
            wt.Complex2f,
            state_arrays["face0_operator_m00"],
            state_idx,
        ),
        "face0_operator_m01": dr.gather(
            wt.Complex2f,
            state_arrays["face0_operator_m01"],
            state_idx,
        ),
        "face0_operator_m10": dr.gather(
            wt.Complex2f,
            state_arrays["face0_operator_m10"],
            state_idx,
        ),
        "face0_operator_m11": dr.gather(
            wt.Complex2f,
            state_arrays["face0_operator_m11"],
            state_idx,
        ),
        "face1_operator_m00": dr.gather(
            wt.Complex2f,
            state_arrays["face1_operator_m00"],
            state_idx,
        ),
        "face1_operator_m01": dr.gather(
            wt.Complex2f,
            state_arrays["face1_operator_m01"],
            state_idx,
        ),
        "face1_operator_m10": dr.gather(
            wt.Complex2f,
            state_arrays["face1_operator_m10"],
            state_idx,
        ),
        "face1_operator_m11": dr.gather(
            wt.Complex2f,
            state_arrays["face1_operator_m11"],
            state_idx,
        ),
        "face0_eta_r": dr.gather(wt.Float, state_arrays["face0_eta_r"], state_idx),
        "face0_mu_r": dr.gather(wt.Float, state_arrays["face0_mu_r"], state_idx),
        "face0_sigma": dr.gather(wt.Float, state_arrays["face0_sigma"], state_idx),
        "face0_gain": dr.gather(wt.Float, state_arrays["face0_gain"], state_idx),
        "face0_use_fresnel": dr.gather(wt.Bool, state_arrays["face0_use_fresnel"], state_idx),
        "face1_eta_r": dr.gather(wt.Float, state_arrays["face1_eta_r"], state_idx),
        "face1_mu_r": dr.gather(wt.Float, state_arrays["face1_mu_r"], state_idx),
        "face1_sigma": dr.gather(wt.Float, state_arrays["face1_sigma"], state_idx),
        "face1_gain": dr.gather(wt.Float, state_arrays["face1_gain"], state_idx),
        "face1_use_fresnel": dr.gather(wt.Bool, state_arrays["face1_use_fresnel"], state_idx),
        "incident_vector_x": dr.gather(wt.Complex2f, state_arrays["incident_vector_x"], state_idx),
        "incident_vector_y": dr.gather(wt.Complex2f, state_arrays["incident_vector_y"], state_idx),
        "incident_vector_z": dr.gather(wt.Complex2f, state_arrays["incident_vector_z"], state_idx),
        "incident_normal_derivative_vector_x": dr.gather(
            wt.Complex2f,
            state_arrays["incident_normal_derivative_vector_x"],
            state_idx,
        ),
        "incident_normal_derivative_vector_y": dr.gather(
            wt.Complex2f,
            state_arrays["incident_normal_derivative_vector_y"],
            state_idx,
        ),
        "incident_normal_derivative_vector_z": dr.gather(
            wt.Complex2f,
            state_arrays["incident_normal_derivative_vector_z"],
            state_idx,
        ),
        "source_type_code": dr.gather(wt.UInt32, state_arrays["source_type_code"], state_idx),
        "order": dr.gather(wt.UInt32, state_arrays["order"], state_idx),
    }


def _scatter_pair_vectors(pair_vector, ownership, rx_idx, *, n_rx: int):
    direct_mask = ownership == wt.Int32(0)
    multi_mask = ownership == wt.Int32(1)
    direct_pair_vector = vector_select(direct_mask, pair_vector, vector_zero(dr.width(rx_idx)))
    multi_pair_vector = vector_select(multi_mask, pair_vector, vector_zero(dr.width(rx_idx)))
    direct_total = vector_zero(n_rx)
    multi_total = vector_zero(n_rx)
    for axis in ("x", "y", "z"):
        dr.scatter_reduce(
            dr.ReduceOp.Add,
            direct_total[axis].real,
            direct_pair_vector[axis].real,
            rx_idx,
        )
        dr.scatter_reduce(
            dr.ReduceOp.Add,
            direct_total[axis].imag,
            direct_pair_vector[axis].imag,
            rx_idx,
        )
        dr.scatter_reduce(
            dr.ReduceOp.Add,
            multi_total[axis].real,
            multi_pair_vector[axis].real,
            rx_idx,
        )
        dr.scatter_reduce(
            dr.ReduceOp.Add,
            multi_total[axis].imag,
            multi_pair_vector[axis].imag,
            rx_idx,
        )
    return direct_total, multi_total


def _utd_accumulate_forward_drjit_ad(
    state_arrays: dict,
    rx_pos,
    n_edges: int,
    return_per_edge: bool,
    *,
    scene=None,
    wave: Wave,
    material: Material | None = None,
    rx_polarization=None,
    receiver_axis: str = "z",
    select_diffraction_point: bool = True,
    prefilter_visibility: bool = False,
    shadow_support_cutoff_db: float | None = None,
    return_scalar: bool = True,
    tx: Tx | None = None,
):
    del n_edges
    from witwin.channel.deterministic.diffraction.forward import ForwardEval

    if return_per_edge:
        raise RuntimeError("DrJit AD UTD accumulation does not support per-edge output.")

    n_states = int(state_arrays["n_states"])
    n_rx = dr.width(rx_pos.x)
    if n_states == 0 or n_rx == 0:
        zf = complex_zero(n_rx)
        zv = vector_zero(n_rx)
        return zf, zf, zv, zv, []

    n_pairs = n_states * n_rx
    pair_idx = dr.arange(wt.UInt32, n_pairs)
    state_idx = pair_idx // wt.UInt32(n_rx)
    rx_idx = pair_idx % wt.UInt32(n_rx)
    pair_state = _gather_pair_state(state_arrays, state_idx)
    pair_rx = wt.Point3f(
        dr.gather(wt.Float, rx_pos.x, rx_idx),
        dr.gather(wt.Float, rx_pos.y, rx_idx),
        dr.gather(wt.Float, rx_pos.z, rx_idx),
    )
    ownership = dr.gather(wt.Int32, _ownership_codes(state_arrays), state_idx)
    _, pair_vector = ForwardEval.to_targets(
        pair_state,
        pair_rx,
        wave,
        return_vector=True,
        material=material,
        scene=scene,
        smooth_exterior_shadow=shadow_support_cutoff_db is not None,
        tx=tx,
        select_diffraction_point=bool(select_diffraction_point),
        enable_segment_visibility=bool(prefilter_visibility),
    )
    direct_vector_total, multi_vector_total = _scatter_pair_vectors(
        pair_vector,
        ownership,
        rx_idx,
        n_rx=n_rx,
    )
    if return_scalar:
        active_rx_pol = (1.0, 0.0, 0.0) if rx_polarization is None else rx_polarization
        direct_total = scalarize_tangential_jones(
            jones_tangential(direct_vector_total, axis=receiver_axis),
            active_rx_pol,
            axis=receiver_axis,
        )
        multi_total = scalarize_tangential_jones(
            jones_tangential(multi_vector_total, axis=receiver_axis),
            active_rx_pol,
            axis=receiver_axis,
        )
    else:
        direct_total = None
        multi_total = None
    return direct_total, multi_total, direct_vector_total, multi_vector_total, []


def _direct_first_order_selection(s: dict, *, enabled: bool):
    n_s = int(s["n_states"])
    if not enabled or "source_type_code" not in s or "order" not in s:
        return dr.zeros(wt.Float, n_s)
    direct_mask = s["source_type_code"] == wt.UInt32(SOURCE_TYPE_DIRECT_TX)
    first_order_mask = s["order"] == wt.UInt32(1)
    return wt.Float(direct_mask & first_order_mask)


def _pack_state_soa(s, *, select_diffraction_point: bool = False):
    """Extract all SoA raw float pointers from a state dict.

    Returns a flat tuple in the order expected by the C++ kernel.
    """
    n_s = int(s["n_states"])
    select_stationary = _direct_first_order_selection(
        s,
        enabled=bool(select_diffraction_point),
    )
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
        s["face0_eta_r"], s["face0_mu_r"], s["face0_sigma"], s["face0_gain"],
        s["face1_eta_r"], s["face1_mu_r"], s["face1_sigma"], s["face1_gain"],
        select_stationary,
    )
    p = s["edge_pos"]
    d = s["edge_dir"]
    n0 = s["n0"]
    nn = s["n_face_n"]
    sp = s["source_pos"]
    # Bool -> Float for use_fresnel / present
    f0uf = wt.Float(s["face0_use_fresnel"])
    f1uf = wt.Float(s["face1_use_fresnel"])
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
        s["face0_eta_r"], s["face0_mu_r"],
        s["face0_sigma"], s["face0_gain"], f0uf, f0pr,
        s["face1_eta_r"], s["face1_mu_r"],
        s["face1_sigma"], s["face1_gain"], f1uf, f1pr,
        select_stationary,
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


def _pair_vector_output_keys():
    return (
        "direct_vec_x_re",
        "direct_vec_x_im",
        "direct_vec_y_re",
        "direct_vec_y_im",
        "direct_vec_z_re",
        "direct_vec_z_im",
        "multi_vec_x_re",
        "multi_vec_x_im",
        "multi_vec_y_re",
        "multi_vec_y_im",
        "multi_vec_z_re",
        "multi_vec_z_im",
    )


def _build_material_params(ext, material: Material | None = None, wavelength=None, tx_polarization=None):
    mat = ext.MaterialParams()
    mat.use_fresnel = 0
    mat.eta_r = 0.0
    mat.mu_r = 1.0
    mat.sigma = 0.0
    mat.gain = 1.0
    mat.omega = 0.0
    def _scalar(value):
        try:
            return float(value)
        except TypeError:
            return float(value[0])

    active_tx_pol = (1.0, 0.0, 0.0) if tx_polarization is None else tx_polarization
    mat.tx_pol_x = _scalar(active_tx_pol.x if hasattr(active_tx_pol, "x") else active_tx_pol[0])
    mat.tx_pol_y = _scalar(active_tx_pol.y if hasattr(active_tx_pol, "y") else active_tx_pol[1])
    mat.tx_pol_z = _scalar(active_tx_pol.z if hasattr(active_tx_pol, "z") else active_tx_pol[2])
    if wavelength is not None:
        mat.omega = float(material_angular_frequency(wavelength)[0])
    if material is not None:
        mat.gain = float(material.gain_scalar)
    return mat


def _ownership_codes(state_arrays):
    ownership = wt.Int32(
        Geo.ownership_code(
            state_arrays["prefix_reflection_depth"],
            state_arrays["intermediate_reflection_depth"],
            state_arrays["suffix_reflection_depth"],
        )
    )
    return ownership


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


def _shadow_support_mask(
    *,
    batch_rx,
    edge_pos,
    edge_dir,
    n0,
    nn,
    source_pos,
    wedge_n,
    visible,
    target_exterior,
    scene,
    shadow_support_cutoff_db,
    finite_selected=None,
):
    edge_geometry = wedge_geometry(source_pos, edge_pos, edge_dir, n0, batch_rx)
    source_exterior = wedge_exterior_mask(source_pos - edge_pos, edge_dir, n0, nn)
    base_valid = (
        visible
        & source_exterior
        & (edge_geometry.s_prime > DIFFRACTION_MIN_DISTANCE)
        & (edge_geometry.s > DIFFRACTION_MIN_DISTANCE)
    )
    shadow_opening_angle = dr.maximum(
        wt.Float(2.0 * dr.pi) - wedge_n * dr.pi,
        wt.Float(2.0e-3),
    )
    shadow_half_angle = wt.Float(0.5) * shadow_opening_angle
    wrap_boundary = edge_geometry.phi >= wt.Float(2.0 * dr.pi) - shadow_half_angle
    shadow_boundary_distance = dr.select(
        wrap_boundary,
        wt.Float(2.0 * dr.pi) - edge_geometry.phi,
        edge_geometry.phi - wedge_n * dr.pi,
    )
    support_angle = shadow_support_angle_from_cutoff_db(
        wedge_n,
        shadow_support_cutoff_db,
    )
    if shadow_support_cutoff_db is None and finite_selected is not None:
        support_angle = dr.select(
            finite_selected,
            shadow_half_angle,
            support_angle,
        )
    shadow_completion = (
        ~target_exterior
        & (shadow_boundary_distance >= wt.Float(0.0))
        & (shadow_boundary_distance < support_angle)
    )
    if scene is not None and hasattr(scene, "point_inside_closed_mesh"):
        interior_shadow = scene.point_inside_closed_mesh(
            batch_rx,
            robust=True,
            active=base_valid & shadow_completion,
        )
        shadow_completion = shadow_completion & ~interior_shadow
    return base_valid & (target_exterior | shadow_completion)


def _gather_direct_first_order_mask(state_arrays: dict, state_idx):
    if "source_type_code" not in state_arrays or "order" not in state_arrays:
        return dr.zeros(wt.Bool, dr.width(state_idx))
    source_type = dr.gather(wt.UInt32, state_arrays["source_type_code"], state_idx)
    order = dr.gather(wt.UInt32, state_arrays["order"], state_idx)
    return (source_type == wt.UInt32(SOURCE_TYPE_DIRECT_TX)) & (order == wt.UInt32(1))


def _encode_pair_validity_mask(visible):
    return dr.select(
        visible,
        wt.Int32(_VALID_PAIR_FLAG),
        wt.Int32(0),
    )


def _utd_accumulate_forward_native_primal(
    state_arrays: dict,
    rx_pos,
    k: float,
    n_edges: int,
    return_per_edge: bool,
    *,
    scene=None,
    wavelength: float | None = None,
    material: Material | None = None,
    rx_polarization=None,
    receiver_axis: str = "z",
    execution=None,
    shadow_support_cutoff_db: float | None = None,
    select_diffraction_point: bool = True,
    tx_polarization=None,
    return_scalar: bool = True,
):
    """
    Native CUDA path for UTD diffraction accumulation.
    """
    _require_finite_edge_bounds(
        state_arrays,
        context="Native UTD accumulation",
    )
    ext = NativeExtension.load()
    execution = coerce_diffraction_execution(execution)
    n_states = state_arrays["n_states"]
    n_rx = dr.width(rx_pos.x)
    active_rx_pol = (1.0, 0.0, 0.0) if rx_polarization is None else rx_polarization
    native_rx_pos = _materialize_receiver_positions(rx_pos)

    if n_states == 0 or n_rx == 0:
        zf = complex_zero(n_rx)
        zv = vector_zero(n_rx)
        pe = [(zf.real, zf.imag) for _ in range(n_edges)] if return_per_edge else []
        return zf, zf, zv, zv, pe
    if return_per_edge:
        raise RuntimeError("Native UTD accumulation does not support per-edge output.")

    mat = _build_material_params(ext, material, wavelength, tx_polarization=tx_polarization)

    out_buffers = _zero_pair_output_buffers(n_rx)
    vector_output_keys = _pair_vector_output_keys()

    # Pack state SoA pointers once; chunking only limits temporary visibility masks.
    soa = _pack_state_soa(
        state_arrays,
        select_diffraction_point=bool(select_diffraction_point),
    )
    ownership = _ownership_codes(state_arrays)

    chunk_size = Geo.cart_chunk(n_states, n_rx)
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
            edge_dir = dr.gather(wt.Vector3f, state_arrays["edge_dir"], state_idx)
            n0 = dr.gather(wt.Vector3f, state_arrays["n0"], state_idx)
            nn = dr.gather(wt.Vector3f, state_arrays["n_face_n"], state_idx)
            source_pos = dr.gather(wt.Point3f, state_arrays["source_pos"], state_idx)
            wedge_n = dr.gather(wt.Float, state_arrays["wedge_n"], state_idx)
            adj0 = dr.gather(wt.Int32, state_arrays["adjacent_face0"], state_idx)
            adj1 = dr.gather(wt.Int32, state_arrays["adjacent_face1"], state_idx)
            adj_group0 = scene.triangle_group_id(adj0)
            adj_group1 = scene.triangle_group_id(adj1)
            batch_rx = wt.Point3f(
                dr.gather(wt.Float, native_rx_pos.x, rx_idx),
                dr.gather(wt.Float, native_rx_pos.y, rx_idx),
                dr.gather(wt.Float, native_rx_pos.z, rx_idx),
            )
            if select_diffraction_point:
                select_mask = _gather_direct_first_order_mask(state_arrays, state_idx)
                pair_state = {
                    "edge_pos": edge_pos,
                    "edge_dir": edge_dir,
                    "source_pos": source_pos,
                    "edge_line_min": dr.gather(wt.Float, state_arrays["edge_line_min"], state_idx),
                    "edge_line_max": dr.gather(wt.Float, state_arrays["edge_line_max"], state_idx),
                }
                selected_point = Geo.finite_edge_diffraction_point(pair_state, batch_rx)
                selected_valid = select_mask & selected_point["valid"]
                visibility_edge_pos = dr.select(
                    selected_valid,
                    selected_point["visibility_point"],
                    edge_pos,
                )
                edge_pos = dr.select(selected_valid, selected_point["point"], edge_pos)
                visible_stationary = ~select_mask | selected_point["valid"]
            else:
                visibility_edge_pos = edge_pos
                visible_stationary = dr.full(wt.Bool, dr.width(state_idx), True)
                selected_valid = None
            visible = scene.segment_visible(
                visibility_edge_pos, batch_rx,
                ignore_surface_group_idx=(adj_group0, adj_group1),
            )
            target_exterior = wedge_exterior_mask(
                batch_rx - edge_pos,
                edge_dir,
                n0,
                nn,
            )
            visible = _shadow_support_mask(
                batch_rx=batch_rx,
                edge_pos=edge_pos,
                edge_dir=edge_dir,
                n0=n0,
                nn=nn,
                source_pos=source_pos,
                wedge_n=wedge_n,
                visible=visible,
                target_exterior=target_exterior,
                scene=scene,
                shadow_support_cutoff_db=shadow_support_cutoff_db,
                finite_selected=selected_valid,
            )
            visible = visible & visible_stationary
            valid_mask = _encode_pair_validity_mask(visible)
            if not bool(dr.any(visible)):
                continue
        else:
            valid_mask = dr.full(wt.Int32, _VALID_PAIR_FLAG, chunk_n * n_rx)

        chunk_outputs = ext.utd_accumulate_tiled_vectors(
            wt.Int32(chunk_state_idx),
            wt.Int32(full_rx_idx),
            wt.Int32(valid_mask),
            wt.Int32(ownership),
            soa,
            (native_rx_pos.x, native_rx_pos.y, native_rx_pos.z),
            mat,
            chunk_n,
            n_rx,
            k,
        )
        for key, chunk_value in zip(vector_output_keys, chunk_outputs):
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
    if return_scalar:
        direct_total = (
            scalarize_tangential_jones(
                jones_tangential(direct_vector_total, axis=receiver_axis),
                active_rx_pol, axis=receiver_axis,
            )
        )
        multi_total = (
            scalarize_tangential_jones(
                jones_tangential(multi_vector_total, axis=receiver_axis),
                active_rx_pol, axis=receiver_axis,
            )
        )
    else:
        direct_total = None
        multi_total = None
    # per_edge not yet supported in native path
    per_edge_list = []
    return direct_total, multi_total, direct_vector_total, multi_vector_total, per_edge_list


def utd_pair_vectors(
    state_arrays: dict,
    target_pos,
    *,
    wave: Wave,
    material: Material | None = None,
    select_diffraction_point: bool = True,
):
    _require_finite_edge_bounds(
        state_arrays,
        context="Native pair UTD vector evaluation",
    )
    ext = NativeExtension.require_functions(
        ("utd_pair_vectors",),
        context="Native pair UTD vector evaluation",
    )

    n_pairs = int(dr.width(target_pos.x))
    if n_pairs <= 0:
        return complex_zero(0), vector_zero(0)

    state_idx = wt.Int32(dr.arange(wt.UInt32, n_pairs))
    rx_idx = wt.Int32(dr.arange(wt.UInt32, n_pairs))
    ownership = dr.zeros(wt.Int32, n_pairs)
    native_target_pos = _materialize_receiver_positions(target_pos)
    mat = _build_material_params(ext, material, wave.wavelength_scalar)
    dr.eval(
        state_idx,
        rx_idx,
        ownership,
        native_target_pos.x,
        native_target_pos.y,
        native_target_pos.z,
    )
    outputs = ext.utd_pair_vectors(
        state_idx,
        rx_idx,
        ownership,
        _pack_state_soa(
            state_arrays,
            select_diffraction_point=bool(select_diffraction_point),
        ),
        (native_target_pos.x, native_target_pos.y, native_target_pos.z),
        mat,
        n_pairs,
        wave.k_scalar,
    )
    direct = wt.Complex2f(outputs[0], outputs[1])
    multi = wt.Complex2f(outputs[2], outputs[3])
    direct_vector = {
        "x": wt.Complex2f(outputs[4], outputs[5]),
        "y": wt.Complex2f(outputs[6], outputs[7]),
        "z": wt.Complex2f(outputs[8], outputs[9]),
    }
    multi_vector = {
        "x": wt.Complex2f(outputs[10], outputs[11]),
        "y": wt.Complex2f(outputs[12], outputs[13]),
        "z": wt.Complex2f(outputs[14], outputs[15]),
    }
    return eval_complex(direct + multi), {
        "x": direct_vector["x"] + multi_vector["x"],
        "y": direct_vector["y"] + multi_vector["y"],
        "z": direct_vector["z"] + multi_vector["z"],
    }


def utd_accumulate_forward(
    state_arrays: dict,
    rx: Rx,
    tx: Tx,
    n_edges: int,
    return_per_edge: bool,
    *,
    scene=None,
    wave: Wave,
    material: Material,
    receiver_axis: str = "z",
    execution=None,
    select_diffraction_point: bool = True,
    prefilter_visibility: bool = False,
    shadow_support_cutoff_db: float | None = None,
    return_scalar: bool = True,
):
    _require_finite_edge_bounds(
        state_arrays,
        context="Native UTD accumulation",
    )
    if _state_arrays_have_material_grad(state_arrays):
        return _utd_accumulate_forward_drjit_ad(
            state_arrays,
            rx.positions,
            n_edges,
            return_per_edge,
            scene=scene,
            wave=wave,
            material=material,
            rx_polarization=rx.effective_polarization(tx),
            receiver_axis=receiver_axis,
            select_diffraction_point=bool(select_diffraction_point),
            prefilter_visibility=bool(prefilter_visibility),
            shadow_support_cutoff_db=shadow_support_cutoff_db,
            return_scalar=bool(return_scalar),
            tx=tx,
        )
    return _utd_accumulate_forward_native_primal(
        state_arrays,
        rx.positions,
        wave.k_scalar,
        n_edges,
        return_per_edge,
        scene=scene,
        wavelength=wave.wavelength_scalar,
        material=material,
        rx_polarization=rx.effective_polarization(tx),
        receiver_axis=receiver_axis,
        execution=execution,
        shadow_support_cutoff_db=shadow_support_cutoff_db,
        select_diffraction_point=bool(select_diffraction_point),
        tx_polarization=tx.polarization,
        return_scalar=bool(return_scalar),
    )
