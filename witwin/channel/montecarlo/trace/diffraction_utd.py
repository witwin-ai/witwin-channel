"""UTD diffraction math for Monte Carlo radiomap."""
from __future__ import annotations

import math
from dataclasses import dataclass

import drjit as dr
from witwin.channel.montecarlo import types as wt
from .. import grid_ops
from witwin.channel.core.numerics.constants import DIFFRACTION_MIN_DISTANCE, EPS, SMALL_EPS
from witwin.channel.core.numerics import arrays
from witwin.channel.core.physics import polarization
from witwin.channel.core.geometry.diffraction import cotangent_pole_safe_mask, wedge_exterior_mask, wedge_geometry
from witwin.channel.core.physics.wave_math import cot, f_utd


@dataclass(slots=True)
class EdgeGeometrySetup:
    """Edge-local geometry frame produced by setup_edge_geometry."""
    width: int
    phi: object
    phi_prime: object
    s: object
    s_prime: object
    sin_beta0: object
    wedge_n_b: object
    n0_b: object
    nn_b: object
    edge_hat: object
    field_valid: object
    incident_hat: object
    outgoing_hat: object
    incident_basis: object
    outgoing_basis: object
    incident_jones: object


@dataclass(slots=True)
class DiffractionCoefficients:
    """Kouyoumjian-Pathak diffraction coefficients and transition data."""
    pole_safe: object
    field_valid: object
    phi_eval: object
    phi_prime_eval: object
    exterior_angle: object
    d12: object
    d3: object
    d4: object
    incident_transition_response: object
    reflection_transition_response: object
    incident_transition_weight: object
    reflection_transition_weight: object
    dif_n_p: object
    dif_n_m: object
    sum_n_p: object
    sum_n_m: object


@dataclass(slots=True)
class DiffractionFieldSupport:
    """Per-sample diffraction-field diagnostic outputs."""
    field_valid: object
    pole_safe: object
    incident_transition_weight: object
    reflection_transition_weight: object
    incident_transition_response: object
    reflection_transition_response: object
    dif_n_p: object
    dif_n_m: object
    sum_n_p: object
    sum_n_m: object


@dataclass(slots=True)
class DiffractionSupportOverride:
    """Locks discrete decisions during AD replay so backward graph matches forward."""
    field_valid: object
    pole_safe: object
    dif_n_p: object
    dif_n_m: object
    sum_n_p: object
    sum_n_m: object


@dataclass(slots=True)
class DiffContribution:
    """eval_diff_contribution output: per-sample diffraction power and metadata."""
    contribution: object
    contribution_active: object
    cell_idx: object
    field_support: DiffractionFieldSupport
    integration_weight: object
    exterior_angle: object


class UTD:
    """UTD diffraction math: transition functions, Kouyoumjian-Pathak diffraction
    coefficients, Jones polarization chain, and face-reflection operators."""

    @staticmethod
    def setup_edge_geometry(source_pos, sampled_edge_pos, edge_dir, n0, nn, target_pos, wedge_n) -> EdgeGeometrySetup:
        width = dr.width(target_pos.x)
        edge_geometry = wedge_geometry(source_pos, sampled_edge_pos, edge_dir, n0, target_pos)
        wedge_n_b = arrays.broadcast(wedge_n, width)
        edge_dir_b = arrays.broadcast(edge_dir, width)
        n0_b = arrays.broadcast(n0, width)
        nn_b = arrays.broadcast(nn, width)
        source_pos_b = arrays.broadcast(source_pos, width)
        edge_hat = polarization.normalize_real_with_fallback(edge_geometry.edge_hat, wt.Vector3f(0.0, 0.0, 1.0))
        source_exterior = wedge_exterior_mask(
            source_pos_b - sampled_edge_pos, edge_dir_b, n0_b, nn_b,
        )
        s = edge_geometry.s
        s_prime = edge_geometry.s_prime
        field_valid = (
            source_exterior & (s_prime > DIFFRACTION_MIN_DISTANCE) & (s > DIFFRACTION_MIN_DISTANCE)
        )
        incident_ray_dir = sampled_edge_pos - source_pos_b
        outgoing_ray_dir = target_pos - sampled_edge_pos
        incident_hat = polarization.normalize_real_with_fallback(incident_ray_dir, wt.Vector3f(0.0, 0.0, -1.0))
        outgoing_hat = polarization.normalize_real_with_fallback(outgoing_ray_dir, wt.Vector3f(0.0, 0.0, 1.0))
        incident_basis = polarization.basis_from_first_vector(incident_hat, polarization.implicit_basis_vector(incident_hat))
        outgoing_basis = polarization.basis_from_first_vector(outgoing_hat, polarization.implicit_basis_vector(outgoing_hat))
        incident_jones = {
            "u": arrays.broadcast(wt.Complex2f(1.0, 0.0), width),
            "v": arrays.complex_zero(width),
        }
        return EdgeGeometrySetup(
            width=width,
            phi=edge_geometry.phi, phi_prime=edge_geometry.phi_prime,
            s=s, s_prime=s_prime,
            sin_beta0=edge_geometry.sin_beta_eff,
            wedge_n_b=wedge_n_b, n0_b=n0_b, nn_b=nn_b,
            edge_hat=edge_hat, field_valid=field_valid,
            incident_hat=incident_hat, outgoing_hat=outgoing_hat,
            incident_basis=incident_basis, outgoing_basis=outgoing_basis,
            incident_jones=incident_jones,
        )

    @staticmethod
    def edge_diffraction_power(*, source_pos, oriented, wedge_n, sampled_edge_pos, target_pos,
                               k, wavelength, support_override: DiffractionSupportOverride | None = None):
        geo = UTD.setup_edge_geometry(
            source_pos, sampled_edge_pos, oriented.edge_dir, oriented.n0, oriented.nn, target_pos, wedge_n,
        )
        coeffs = UTD.diffraction_coefficients(
            geo.phi, geo.phi_prime, geo.wedge_n_b,
            geo.s, geo.s_prime, geo.sin_beta0,
            k, geo.field_valid, support_override, geo.width,
        )
        field_power, support = UTD.apply_jones_chain(
            geo=geo, coeffs=coeffs,
            face0=oriented.face0_material, face1=oriented.face1_material,
            wavelength=wavelength,
        )
        return field_power, coeffs.field_valid, support

    @staticmethod
    def sample_keller_cone(edge_dir, n0, nn, sample, ki, *, lit_region: bool):
        edge_hat = edge_dir / (dr.norm(edge_dir) + wt.Float(EPS))
        n0_hat = n0 / (dr.norm(n0) + wt.Float(EPS))
        nn_hat = nn / (dr.norm(nn) + wt.Float(EPS))
        ki_hat = ki / (dr.norm(ki) + wt.Float(EPS))
        t0 = dr.normalize(dr.cross(n0_hat, edge_hat))
        e_fwd = dr.select(dr.dot(edge_hat, ki_hat) > 0.0, edge_hat, -edge_hat)
        ki_local = wt.Vector3f(
            dr.dot(ki_hat, t0),
            dr.dot(ki_hat, n0_hat),
            dr.dot(ki_hat, e_fwd),
        )
        sin_beta0 = dr.sqrt(
            dr.maximum(wt.Float(0.0), wt.Float(1.0) - ki_local.z * ki_local.z)
        )
        beta0 = dr.atan2(sin_beta0, ki_local.z)
        phi_i = dr.atan2(ki_local.y, ki_local.x)
        phi_i = dr.select(phi_i < 0.0, phi_i + wt.Float(2.0 * math.pi), phi_i)
        wedge_interior = dr.safe_acos(
            dr.clip(-dr.dot(n0_hat, nn_hat), wt.Float(-1.0), wt.Float(1.0))
        )
        exterior_angle = wt.Float(2.0 * math.pi) - wedge_interior
        phi = (
            sample * exterior_angle
            if lit_region
            else phi_i + sample * dr.maximum(exterior_angle - phi_i, wt.Float(0.0))
        )
        sin_beta, cos_beta = dr.sincos(beta0)
        sin_phi, cos_phi = dr.sincos(phi)
        return sin_phi * sin_beta * n0_hat + cos_phi * sin_beta * t0 + cos_beta * e_fwd

    @staticmethod
    def integration_weight(*, edge_origin, edge_dir, n0, source_pos, diff_point, k_world, target_pos, plane_normal):
        width = int(dr.width(target_pos.x))
        edge_origin_b = arrays.broadcast(edge_origin, width)
        edge_hat = arrays.broadcast(edge_dir / (dr.norm(edge_dir) + wt.Float(EPS)), width)
        n0_b = arrays.broadcast(n0 / (dr.norm(n0) + wt.Float(EPS)), width)
        plane_normal_b = arrays.broadcast(plane_normal, width)
        source_pos_b = arrays.broadcast(source_pos, width)
        incident_dir = diff_point - source_pos_b
        e_fwd = dr.select(dr.dot(edge_hat, incident_dir) > 0.0, edge_hat, -edge_hat)
        t0 = dr.normalize(dr.cross(n0_b, edge_hat))
        k_local_x = dr.dot(k_world, t0)
        k_local_y = dr.dot(k_world, n0_b)
        phi = dr.atan2(k_local_y, k_local_x)
        ell = dr.dot(diff_point - edge_origin_b, edge_hat)
        v = source_pos_b - edge_origin_b
        w = dr.dot(v, edge_hat)
        source_proj = edge_origin_b + w * edge_hat
        perp_offset = source_pos_b - source_proj
        perp_norm = dr.norm(perp_offset)
        u = ell - w
        radial_distance = dr.norm(ell * edge_hat - v)
        nrm = radial_distance + wt.Float(EPS)
        sin_phi, cos_phi = dr.sincos(phi)
        tangential_dir = cos_phi * t0 + sin_phi * n0_b
        angular_dir = -sin_phi * t0 + cos_phi * n0_b
        d_world = (perp_norm / nrm) * tangential_dir + (u / nrm) * e_fwd
        dd_dphi = (perp_norm / nrm) * angular_dir
        dd_dell = e_fwd / nrm - d_world * (u / dr.maximum(radial_distance, wt.Float(EPS))) / nrm
        numerator = dr.dot(
            plane_normal_b,
            target_pos - edge_origin_b - ell * edge_hat,
        )
        denominator = dr.dot(plane_normal_b, d_world)
        safe_denominator = denominator + wt.Float(EPS)
        numerator_dell = -dr.dot(plane_normal_b, edge_hat)
        denominator_dell = dr.dot(plane_normal_b, dd_dell)
        denominator_dphi = dr.dot(plane_normal_b, dd_dphi)
        travel = numerator / safe_denominator
        dtravel_dell = (
            numerator_dell * safe_denominator - numerator * denominator_dell
        ) / (safe_denominator * safe_denominator)
        dtravel_dphi = (
            -numerator * denominator_dphi
        ) / (safe_denominator * safe_denominator)
        ds_dell = edge_hat + dtravel_dell * d_world + travel * dd_dell
        ds_dphi = dtravel_dphi * d_world + travel * dd_dphi
        return dr.norm(dr.cross(ds_dphi, ds_dell))

    @staticmethod
    def eval_diff_contribution(*, oriented, batch_states, diff_point, ko, plane_hit, source_visible,
                               visible_target, sample_active, grid, k, wavelength, diff_gain_scale,
                               total_length_weight, plane_normal) -> DiffContribution:
        wedge_interior = dr.safe_acos(
            dr.clip(
                -dr.dot(
                    oriented.n0 / (dr.norm(oriented.n0) + wt.Float(EPS)),
                    oriented.nn / (dr.norm(oriented.nn) + wt.Float(EPS)),
                ),
                wt.Float(-1.0),
                wt.Float(1.0),
            )
        )
        exterior_angle = wt.Float(2.0 * math.pi) - wedge_interior
        iw = UTD.integration_weight(
            edge_origin=batch_states.edge_pos,
            edge_dir=oriented.edge_dir,
            n0=oriented.n0,
            source_pos=batch_states.source_pos,
            diff_point=diff_point,
            k_world=ko,
            target_pos=plane_hit.target_pos,
            plane_normal=plane_normal,
        )
        field_power, field_valid, field_support = UTD.edge_diffraction_power(
            source_pos=batch_states.source_pos,
            oriented=oriented,
            wedge_n=batch_states.wedge_n,
            sampled_edge_pos=diff_point,
            target_pos=plane_hit.target_pos,
            k=k,
            wavelength=wavelength,
        )
        contribution_active = sample_active & source_visible & visible_target & field_valid
        contribution = dr.select(
            contribution_active,
            field_power * diff_gain_scale * iw * total_length_weight * exterior_angle,
            wt.Float(0.0),
        )
        cell_idx = grid_ops.cell_index(
            grid=grid,
            coord_0=plane_hit.coord_0,
            coord_1=plane_hit.coord_1,
        )
        return DiffContribution(
            contribution=contribution,
            contribution_active=contribution_active,
            cell_idx=cell_idx,
            field_support=field_support,
            integration_weight=iw,
            exterior_angle=exterior_angle,
        )

    @staticmethod
    def transition_weight_from_argument(x):
        """Map UTD F(x) magnitude to a phase-free boundary proximity weight."""
        x = dr.maximum(x, wt.Float(1.0e-20))
        magnitude = dr.sqrt(
            dr.maximum(arrays.complex_abs_sqr(f_utd(x)), wt.Float(0.0))
        )
        weight = wt.Float(1.0) - dr.minimum(magnitude, wt.Float(1.0))
        weight = dr.clip(weight, wt.Float(0.0), wt.Float(1.0))
        return dr.select(dr.isfinite(weight), weight, wt.Float(0.0))


    @staticmethod
    def override_mask(mask_override, default_mask, *, width):
        if mask_override is None:
            return default_mask
        if dr.width(mask_override) == width:
            return mask_override
        return dr.repeat(mask_override, width)

    @staticmethod
    def override_value(value_override, default_value, *, width):
        if value_override is None:
            return default_value
        if dr.width(value_override) == width:
            return value_override
        return dr.repeat(value_override, width)

    @staticmethod
    # UTD a+/a- coefficients and associated n+/n- indices.
    def a_coefficients(beta, exterior_angle, support_override, n_p_attr, n_m_attr, width):
        n_p = dr.round((beta + dr.pi) * dr.rcp(2.0 * exterior_angle))
        n_m = dr.round((beta - dr.pi) * dr.rcp(2.0 * exterior_angle))
        if support_override is not None:
            n_p = UTD.override_value(getattr(support_override, n_p_attr), n_p, width=width)
            n_m = UTD.override_value(getattr(support_override, n_m_attr), n_m, width=width)
        a_p = 2.0 * dr.square(dr.cos(exterior_angle * n_p - beta * 0.5))
        a_m = 2.0 * dr.square(dr.cos(exterior_angle * n_m - beta * 0.5))
        return a_p, a_m, n_p, n_m

    @staticmethod
    # Compute UTD diffraction coefficients d12, d3, d4.
    def diffraction_coefficients(phi, phi_prime, wedge_n_b, s, s_prime, sin_beta0, k, field_valid,
                                 support_override: DiffractionSupportOverride | None, width) -> DiffractionCoefficients:
        pole_safe = field_valid & cotangent_pole_safe_mask(
            phi, phi_prime, wedge_n_b, wt.Float(1.0e-6),
        )
        if support_override is not None:
            field_valid = UTD.override_mask(support_override.field_valid, field_valid, width=width)
            pole_safe = UTD.override_mask(support_override.pole_safe, pole_safe, width=width)
        phi_eval = dr.select(pole_safe, phi, 0.5 * wedge_n_b * dr.pi)
        phi_prime_eval = dr.select(pole_safe, phi_prime, 0.5 * wedge_n_b * dr.pi)
        exterior_angle = wedge_n_b * dr.pi
        l = s * s_prime * dr.rcp(s + s_prime + wt.Float(EPS)) * dr.square(sin_beta0)
        n = wedge_n_b
        dif_phi = phi_eval - phi_prime_eval
        sum_phi = phi_eval + phi_prime_eval
        a1, a2, dif_n_p, dif_n_m = UTD.a_coefficients(
            dif_phi, exterior_angle, support_override, "dif_n_p", "dif_n_m", width,
        )
        a3, a4, sum_n_p, sum_n_m = UTD.a_coefficients(
            sum_phi, exterior_angle, support_override, "sum_n_p", "sum_n_m", width,
        )
        transition_active = field_valid & pole_safe
        incident_transition_arg = wt.Float(k) * l * dr.minimum(a1, a2)
        reflection_transition_arg = wt.Float(k) * l * dr.minimum(a3, a4)
        zero_transition = arrays.complex_zero(width)
        incident_transition_response = dr.select(
            transition_active,
            polarization.sanitize_complex(f_utd(incident_transition_arg)),
            zero_transition,
        )
        reflection_transition_response = dr.select(
            transition_active,
            polarization.sanitize_complex(f_utd(reflection_transition_arg)),
            zero_transition,
        )
        incident_transition_weight = dr.select(
            transition_active,
            UTD.transition_weight_from_argument(incident_transition_arg),
            wt.Float(0.0),
        )
        reflection_transition_weight = dr.select(
            transition_active,
            UTD.transition_weight_from_argument(reflection_transition_arg),
            wt.Float(0.0),
        )
        factor = -dr.exp(wt.Complex2f(0.0, -0.25 * dr.pi))
        factor *= dr.rcp(
            2.0 * n
            * dr.safe_sqrt(dr.two_pi * wt.Float(k))
            * dr.maximum(sin_beta0, wt.Float(EPS))
        )
        d1 = cot((dr.pi + dif_phi) * dr.rcp(2.0 * n)) * factor * f_utd(wt.Float(k) * l * a1)
        d2 = cot((dr.pi - dif_phi) * dr.rcp(2.0 * n)) * factor * f_utd(wt.Float(k) * l * a2)
        d3 = cot((dr.pi + sum_phi) * dr.rcp(2.0 * n)) * factor * f_utd(wt.Float(k) * l * a3)
        d4 = cot((dr.pi - sum_phi) * dr.rcp(2.0 * n)) * factor * f_utd(wt.Float(k) * l * a4)
        d12 = -(d1 + d2)
        return DiffractionCoefficients(
            pole_safe=pole_safe, field_valid=field_valid,
            phi_eval=phi_eval, phi_prime_eval=phi_prime_eval,
            exterior_angle=exterior_angle,
            d12=d12, d3=d3, d4=d4,
            incident_transition_response=incident_transition_response,
            reflection_transition_response=reflection_transition_response,
            incident_transition_weight=incident_transition_weight,
            reflection_transition_weight=reflection_transition_weight,
            dif_n_p=dif_n_p, dif_n_m=dif_n_m,
            sum_n_p=sum_n_p, sum_n_m=sum_n_m,
        )

    @staticmethod
    # Assemble Jones operator chain and compute output field power.
    def apply_jones_chain(*, geo: EdgeGeometrySetup, coeffs: DiffractionCoefficients,
                          face0, face1, wavelength):
        width = geo.width
        nrv = polarization.normalize_real_with_fallback
        phi_hat_prime = nrv(dr.cross(geo.incident_hat, geo.edge_hat), geo.incident_basis["u"])
        phi_hat = -nrv(dr.cross(geo.outgoing_hat, geo.edge_hat), geo.outgoing_basis["u"])
        e_i_s_0_hat = nrv(dr.cross(geo.incident_hat, geo.n0_b), phi_hat_prime)
        e_i_s_n_hat = nrv(dr.cross(geo.incident_hat, geo.nn_b), phi_hat_prime)
        jr = polarization.jones_operator_rotator
        w_in = jr(geo.incident_hat, geo.incident_basis["u"], phi_hat_prime)
        w_out = jr(geo.outgoing_hat, phi_hat, geo.outgoing_basis["u"])
        w_0_in = jr(geo.incident_hat, phi_hat_prime, e_i_s_0_hat)
        w_0_out = jr(geo.outgoing_hat, e_i_s_0_hat, phi_hat)
        w_n_in = jr(geo.incident_hat, phi_hat_prime, e_i_s_n_hat)
        w_n_out = jr(geo.outgoing_hat, e_i_s_n_hat, phi_hat)

        cos_theta0 = dr.clip(dr.abs(dr.sin(coeffs.phi_prime_eval)), wt.Float(1.0e-6), wt.Float(1.0))
        cos_theta1 = dr.clip(
            dr.abs(dr.sin(coeffs.exterior_angle - coeffs.phi_eval)),
            wt.Float(1.0e-6),
            wt.Float(1.0),
        )
        frd = polarization.fresnel_diagonal_operator
        face0_gain_b = arrays.broadcast(face0.gain, width)
        face1_gain_b = arrays.broadcast(face1.gain, width)
        face0_diag = frd(
            eta_r=arrays.broadcast(face0.eta_r, width),
            sigma=arrays.broadcast(face0.sigma, width),
            gain=face0_gain_b,
            use_fresnel=arrays.broadcast(face0.use_fresnel, width),
            cos_theta=cos_theta0, wavelength=wavelength,
            mu_r=arrays.broadcast(face0.mu_r, width),
        )
        face1_diag = frd(
            eta_r=arrays.broadcast(face1.eta_r, width),
            sigma=arrays.broadcast(face1.sigma, width),
            gain=face1_gain_b,
            use_fresnel=arrays.broadcast(face1.use_fresnel, width),
            cos_theta=cos_theta1, wavelength=wavelength,
            mu_r=arrays.broadcast(face1.mu_r, width),
        )
        jm = polarization.jones_operator_matmul
        edge_gain = wt.Complex2f(
            wt.Float(0.5) * (face0_gain_b + face1_gain_b),
            wt.Float(0.0),
        )
        direct_operator = polarization.jones_operator_scale(
            polarization.jones_operator_diagonal(coeffs.d12, coeffs.d12),
            edge_gain,
        )
        face0_operator = jm(w_0_out, jm(polarization.jones_operator_scale(face0_diag, coeffs.d4), w_0_in))
        face1_operator = jm(w_n_out, jm(polarization.jones_operator_scale(face1_diag, coeffs.d3), w_n_in))
        total_operator = polarization.jones_operator_add(
            direct_operator,
            polarization.jones_operator_add(face0_operator, face1_operator),
        )
        total_operator = jm(w_out, jm(total_operator, w_in))
        total_operator = polarization.jones_operator_mask_detach(total_operator, coeffs.pole_safe)
        field_jones = polarization.apply_jones_operator(geo.incident_jones, total_operator)
        local_scale = dr.rsqrt(geo.s * geo.s_prime * (geo.s + geo.s_prime) + EPS)
        scaled_field_u = field_jones["u"] * wt.Complex2f(local_scale, wt.Float(0.0))
        scaled_field_v = field_jones["v"] * wt.Complex2f(local_scale, wt.Float(0.0))
        field_power = dr.select(
            coeffs.field_valid,
            arrays.complex_abs_sqr(scaled_field_u) + arrays.complex_abs_sqr(scaled_field_v),
            wt.Float(0.0),
        )
        support = DiffractionFieldSupport(
            field_valid=coeffs.field_valid, pole_safe=coeffs.pole_safe,
            incident_transition_weight=coeffs.incident_transition_weight,
            reflection_transition_weight=coeffs.reflection_transition_weight,
            incident_transition_response=coeffs.incident_transition_response,
            reflection_transition_response=coeffs.reflection_transition_response,
            dif_n_p=coeffs.dif_n_p, dif_n_m=coeffs.dif_n_m,
            sum_n_p=coeffs.sum_n_p, sum_n_m=coeffs.sum_n_m,
        )
        return field_power, support

__all__ = [
    "UTD",
    "EdgeGeometrySetup",
    "DiffractionCoefficients",
    "DiffractionFieldSupport",
    "DiffractionSupportOverride",
    "DiffContribution",
]
