"""Canonical material-aware wedge diffraction operator assembly."""

import drjit as dr
import witwin as wt

from ...utils.polarization import (
    jones_operator_add,
    jones_operator_identity,
    jones_operator_scale,
)
from .utd import (
    _diffraction_beta_groups,
    _diffraction_beta_groups_3d,
)


def _operator_terms_2d(phi, phi_prime, wedge_n, k, s, s_prime):
    zero = wt.Complex2f(0.0, 0.0)
    one = wt.Complex2f(1.0, 0.0)
    factor, dif_group, dif_group_1, dif_group_2, _, _, _ = _diffraction_beta_groups(
        phi,
        phi_prime,
        wedge_n,
        wt.Float(k),
        s,
        s_prime,
        zero,
        zero,
    )
    _, _, _, _, sum_plus, sum_plus_1, sum_plus_2 = _diffraction_beta_groups(
        phi,
        phi_prime,
        wedge_n,
        wt.Float(k),
        s,
        s_prime,
        one,
        zero,
    )
    _, _, _, _, sum_minus, sum_minus_1, sum_minus_2 = _diffraction_beta_groups(
        phi,
        phi_prime,
        wedge_n,
        wt.Float(k),
        s,
        s_prime,
        zero,
        one,
    )
    # The grouped beta terms expose the reflected branches in the same face
    # ordering consumed by the current wedge-face operators. The only missing
    # Sionna-style correction at assembly time is the direct-term sign:
    # d12 = -(d1 + d2).
    return {
        "direct": factor * dif_group,
        "face0": factor * sum_plus,
        "face1": factor * sum_minus,
        "direct_dphi": factor * dif_group_1,
        "face0_dphi": factor * sum_plus_1,
        "face1_dphi": factor * sum_minus_1,
        "direct_dphi_prime": factor * (-dif_group_1),
        "face0_dphi_prime": factor * sum_plus_1,
        "face1_dphi_prime": factor * sum_minus_1,
        "direct_d2phi_phi_prime": factor * (-dif_group_2),
        "face0_d2phi_phi_prime": factor * sum_plus_2,
        "face1_d2phi_phi_prime": factor * sum_minus_2,
    }


def _operator_terms_3d(phi, phi_prime, wedge_n, k, s, s_prime, sin_beta0):
    zero = wt.Complex2f(0.0, 0.0)
    one = wt.Complex2f(1.0, 0.0)
    factor, dif_group, dif_group_1, dif_group_2, _, _, _ = _diffraction_beta_groups_3d(
        phi,
        phi_prime,
        wedge_n,
        wt.Float(k),
        s,
        s_prime,
        sin_beta0,
        zero,
        zero,
    )
    _, _, _, _, sum_plus, sum_plus_1, sum_plus_2 = _diffraction_beta_groups_3d(
        phi,
        phi_prime,
        wedge_n,
        wt.Float(k),
        s,
        s_prime,
        sin_beta0,
        one,
        zero,
    )
    _, _, _, _, sum_minus, sum_minus_1, sum_minus_2 = _diffraction_beta_groups_3d(
        phi,
        phi_prime,
        wedge_n,
        wt.Float(k),
        s,
        s_prime,
        sin_beta0,
        zero,
        one,
    )
    # Keep the same raw face-term ordering as the 2D helper above.
    return {
        "direct": factor * dif_group,
        "face0": factor * sum_plus,
        "face1": factor * sum_minus,
        "direct_dphi": factor * dif_group_1,
        "face0_dphi": factor * sum_plus_1,
        "face1_dphi": factor * sum_minus_1,
        "direct_dphi_prime": factor * (-dif_group_1),
        "face0_dphi_prime": factor * sum_plus_1,
        "face1_dphi_prime": factor * sum_minus_1,
        "direct_d2phi_phi_prime": factor * (-dif_group_2),
        "face0_d2phi_phi_prime": factor * sum_plus_2,
        "face1_d2phi_phi_prime": factor * sum_minus_2,
    }


def assemble_diffraction_operator(free_term, face0_term, face1_term, face0_operator, face1_operator):
    width = dr.width(free_term.real)
    total = jones_operator_scale(jones_operator_identity(width), free_term)
    total = jones_operator_add(total, jones_operator_scale(face0_operator, face0_term))
    total = jones_operator_add(total, jones_operator_scale(face1_operator, face1_term))
    return total


def assemble_material_diffraction_operators(
    *,
    phi,
    phi_prime,
    wedge_n,
    k,
    s,
    s_prime,
    face0_operator,
    face1_operator,
    sin_beta0=None,
    include_normal_derivative_ops: bool = True,
):
    raw_terms = (
        _operator_terms_3d(phi, phi_prime, wedge_n, k, s, s_prime, sin_beta0)
        if sin_beta0 is not None
        else _operator_terms_2d(phi, phi_prime, wedge_n, k, s, s_prime)
    )
    # The face operators are already built in the repository's face ordering.
    # Only the canonical direct term is missing the Sionna-style d12 sign.
    operator_terms = {
        "direct": -raw_terms["direct"],
        "face0": raw_terms["face0"],
        "face1": raw_terms["face1"],
        "direct_dphi": -raw_terms["direct_dphi"],
        "face0_dphi": raw_terms["face0_dphi"],
        "face1_dphi": raw_terms["face1_dphi"],
        "direct_dphi_prime": -raw_terms["direct_dphi_prime"],
        "face0_dphi_prime": raw_terms["face0_dphi_prime"],
        "face1_dphi_prime": raw_terms["face1_dphi_prime"],
        "direct_d2phi_phi_prime": -raw_terms["direct_d2phi_phi_prime"],
        "face0_d2phi_phi_prime": raw_terms["face0_d2phi_phi_prime"],
        "face1_d2phi_phi_prime": raw_terms["face1_d2phi_phi_prime"],
    }
    slope_factor = wt.Complex2f(0.0, -1.0) * dr.rcp(wt.Float(k))
    result = {
        "field": assemble_diffraction_operator(
            operator_terms["direct"],
            operator_terms["face0"],
            operator_terms["face1"],
            face0_operator,
            face1_operator,
        ),
        "slope": assemble_diffraction_operator(
            slope_factor * operator_terms["direct_dphi_prime"],
            slope_factor * operator_terms["face0_dphi_prime"],
            slope_factor * operator_terms["face1_dphi_prime"],
            face0_operator,
            face1_operator,
        ),
        "terms": operator_terms,
    }
    if include_normal_derivative_ops:
        result["field_dphi"] = assemble_diffraction_operator(
            operator_terms["direct_dphi"],
            operator_terms["face0_dphi"],
            operator_terms["face1_dphi"],
            face0_operator,
            face1_operator,
        )
        result["slope_dphi"] = assemble_diffraction_operator(
            slope_factor * operator_terms["direct_d2phi_phi_prime"],
            slope_factor * operator_terms["face0_d2phi_phi_prime"],
            slope_factor * operator_terms["face1_d2phi_phi_prime"],
            face0_operator,
            face1_operator,
        )
    return result


__all__ = [
    "assemble_diffraction_operator",
    "assemble_material_diffraction_operators",
]
