"""Face-material Jones operator helpers for diffraction accumulation."""

import drjit as dr
import witwin as wt

from ...utils.material import complex_relative_permittivity, fresnel_reflection
from ...utils.polarization import (
    basis_from_first_vector,
    jones_operator_diagonal,
    jones_operator_in_basis,
)
from ..materials import reflection_material_omega


def state_has_face_material_params(edge_state) -> bool:
    required = (
        "face0_eta_r",
        "face0_sigma",
        "face0_gain",
        "face1_eta_r",
        "face1_sigma",
        "face1_gain",
    )
    return all(key in edge_state for key in required)


def face_reflection_operator_from_material_inputs(
    face_material,
    *,
    cos_theta,
    normal,
    incoming_hat,
    outgoing_hat,
    incoming_edge_basis,
    outgoing_edge_basis,
    wavelength,
):
    omega = reflection_material_omega(wavelength)
    eta = complex_relative_permittivity(
        face_material["eta_r"],
        face_material["sigma"],
        omega,
    )
    r_te, r_tm = fresnel_reflection(cos_theta, eta)
    if __debug__ and not bool(dr.flag(dr.JitFlag.Recording)):
        _te_bad = dr.any(~dr.isfinite(r_te.real) | ~dr.isfinite(r_te.imag))
        _tm_bad = dr.any(~dr.isfinite(r_tm.real) | ~dr.isfinite(r_tm.imag))
        if dr.hint(_te_bad | _tm_bad, mode="scalar"):
            import warnings
            warnings.warn("fresnel_reflection: non-finite coefficient detected", stacklevel=2)
    r_te = wt.Complex2f(
        dr.select(dr.isfinite(r_te.real), r_te.real, wt.Float(0.0)),
        dr.select(dr.isfinite(r_te.imag), r_te.imag, wt.Float(0.0)),
    )
    r_tm = wt.Complex2f(
        dr.select(dr.isfinite(r_tm.real), r_tm.real, wt.Float(0.0)),
        dr.select(dr.isfinite(r_tm.imag), r_tm.imag, wt.Float(0.0)),
    )
    gain = wt.Complex2f(face_material["gain"], wt.Float(0.0))
    diag_operator = jones_operator_diagonal(gain * r_te, gain * r_tm)

    face_s = dr.cross(incoming_hat, normal)
    face_in_basis = basis_from_first_vector(incoming_hat, face_s)
    face_out_basis = basis_from_first_vector(outgoing_hat, face_s)
    return jones_operator_in_basis(
        diag_operator,
        face_in_basis,
        face_out_basis,
        incoming_edge_basis,
        outgoing_edge_basis,
    )


def pair_face_material_operators(
    edge_state,
    *,
    width,
    incoming_hat,
    outgoing_hat,
    incoming_edge_basis,
    outgoing_edge_basis,
    cos_theta0,
    cos_theta1,
    wavelength,
):
    if not state_has_face_material_params(edge_state):
        return None, None

    face0 = {
        "eta_r": dr.full(wt.Float, edge_state["face0_eta_r"], width),
        "sigma": dr.full(wt.Float, edge_state["face0_sigma"], width),
        "gain": dr.full(wt.Float, edge_state["face0_gain"], width),
    }
    face1 = {
        "eta_r": dr.full(wt.Float, edge_state["face1_eta_r"], width),
        "sigma": dr.full(wt.Float, edge_state["face1_sigma"], width),
        "gain": dr.full(wt.Float, edge_state["face1_gain"], width),
    }
    face0_operator = face_reflection_operator_from_material_inputs(
        face0,
        cos_theta=cos_theta0,
        normal=edge_state["n0"],
        incoming_hat=incoming_hat,
        outgoing_hat=outgoing_hat,
        incoming_edge_basis=incoming_edge_basis,
        outgoing_edge_basis=outgoing_edge_basis,
        wavelength=wavelength,
    )
    face1_operator = face_reflection_operator_from_material_inputs(
        face1,
        cos_theta=cos_theta1,
        normal=edge_state["n_face_n"],
        incoming_hat=incoming_hat,
        outgoing_hat=outgoing_hat,
        incoming_edge_basis=incoming_edge_basis,
        outgoing_edge_basis=outgoing_edge_basis,
        wavelength=wavelength,
    )
    return face0_operator, face1_operator


__all__ = [
    "face_reflection_operator_from_material_inputs",
    "pair_face_material_operators",
    "state_has_face_material_params",
]
