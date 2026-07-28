"""Channel material ABI, per-face encoding, and offline layer-stack evaluation.

Logical material specifications are owned by :mod:`witwin.core`; this module
owns the finite encodings consumed by native Channel kernels, the per-face
tensor export the solvers hand to those kernels, and the compile-time
transfer-matrix precompute that produces planar layer-stack coefficients.

:mod:`witwin.channel.kernels.materials` is the native facade; only
``validate_layer_csr`` is re-exported here, and it is the single public name.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Sequence

import numpy as np
import torch

from witwin.channel.constants import C0, EPS0, MU0
from witwin.channel.kernels.materials import (
    mc_face_material_tensors,
    validate_layer_csr as validate_layer_csr,
)
from witwin.channel.runtime import _frequency_participates_in_ad

if TYPE_CHECKING:
    from witwin.channel.scene.compiler import CompiledScene
    from witwin.channel.scene.endpoints import SolverScene as Scene

__all__ = ["validate_layer_csr"]


MATERIAL_ABI_VERSION = 3
DIELECTRIC_MODEL_ID = 1
PEC_MODEL_ID = 2
PEC_EFFECTIVE_SIGMA_E = 1.0e9


def face_material_tensors(
    scene_or_compiled: "Scene | CompiledScene",
    *,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    # Imported here because ``scene`` reaches materials through the scattering
    # tables; a module-level edge back into the compiled scene would close that
    # import cycle.
    from witwin.channel.scene.endpoints import require_compiled

    compiled = require_compiled(scene_or_compiled)
    materials = compiled.materials
    assignments = compiled.assignments
    material_eps_r = materials.eps_r.to(device=device, dtype=torch.float32).contiguous()
    material_sigma_e = materials.sigma_e.to(device=device, dtype=torch.float32)
    # The Fresnel kernels only see (eps_r, sigma_e, mu_r); realize the PEC
    # limit through an effective conductivity.
    material_sigma_e = torch.where(
        materials.model_id.to(device=material_sigma_e.device) == PEC_MODEL_ID,
        material_sigma_e.clamp_min(PEC_EFFECTIVE_SIGMA_E),
        material_sigma_e,
    ).contiguous()
    material_mu_r = materials.mu_r.to(device=device, dtype=torch.float32).contiguous()
    face_material_id = assignments.face_material_id.to(
        device=device, dtype=torch.int32
    ).contiguous()
    exported = mc_face_material_tensors(
        material_eps_r,
        material_sigma_e,
        material_mu_r,
        face_material_id,
    )
    return (
        exported["eps_r"],
        exported["sigma_e"],
        exported["mu_r"],
        exported["gain"],
        exported["valid"],
    )


def face_material_thickness(
    scene_or_compiled: "Scene | CompiledScene",
    *,
    device: torch.device,
) -> torch.Tensor:
    """Expand ITU slab thickness to the global face layout."""

    from witwin.channel.scene.endpoints import require_compiled

    compiled = require_compiled(scene_or_compiled)
    material_thickness = compiled.materials.thickness_m.to(
        device=device,
        dtype=torch.float32,
    )
    face_material_id = compiled.assignments.face_material_id.to(
        device=device, dtype=torch.int64
    )
    return material_thickness.index_select(0, face_material_id).contiguous()


def face_material_field_bundle(
    scene_or_compiled: "Scene | CompiledScene", *, device: torch.device
) -> dict[str, torch.Tensor]:
    """Return the complete per-face finite-slab field operator inputs."""

    from witwin.channel.scene.endpoints import require_compiled

    compiled = require_compiled(scene_or_compiled)
    materials = compiled.materials
    material_id = compiled.assignments.face_material_id.to(
        device=device, dtype=torch.int64
    )
    sigma_e = materials.sigma_e.to(device=device, dtype=torch.float32)
    sigma_e = torch.where(
        materials.model_id.to(device=device) == PEC_MODEL_ID,
        sigma_e.clamp_min(PEC_EFFECTIVE_SIGMA_E),
        sigma_e,
    )

    def per_face(values: torch.Tensor) -> torch.Tensor:
        return (
            values.to(device=device, dtype=torch.float32)
            .index_select(0, material_id)
            .contiguous()
        )

    def material_level(values: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        return values.to(device=device, dtype=dtype).contiguous()

    return {
        "eps_r": per_face(materials.eps_r),
        "sigma_e": per_face(sigma_e),
        "mu_r": per_face(materials.mu_r),
        "gain": per_face(materials.gain),
        "thickness": per_face(materials.thickness_m),
        "material_id": material_id.to(dtype=torch.int32).contiguous(),
        "model_id": materials.model_id.to(device=device, dtype=torch.int32)
        .index_select(0, material_id)
        .contiguous(),
        "valid": torch.ones((material_id.shape[0],), device=device, dtype=torch.bool),
        # ABI v3 material-level views: faces index the CSR via material_id,
        # layers are never expanded per face.
        "layer_offset": material_level(materials.layer_offset, torch.int32),
        "layer_count": material_level(materials.layer_count, torch.int32),
        "layer_thickness_m": material_level(materials.layer_thickness_m, torch.float32),
        "layer_eps_r": material_level(materials.layer_eps_r, torch.float32),
        "layer_sigma_e": material_level(materials.layer_sigma_e, torch.float32),
        "layer_mu_r": material_level(materials.layer_mu_r, torch.float32),
        "rough_sigma_h_m": material_level(materials.rough_sigma_h_m, torch.float32),
        "rough_corr_x_m": material_level(materials.rough_corr_x_m, torch.float32),
        "rough_corr_y_m": material_level(materials.rough_corr_y_m, torch.float32),
        "rough_axis_rad": material_level(materials.rough_axis_rad, torch.float32),
        "geometry_mode_id": material_level(materials.geometry_mode_id, torch.int32),
        "scatter_model_id": material_level(materials.scatter_model_id, torch.int32),
    }


@dataclass(frozen=True)
class _ProductionMedium:
    frequency_hz: float
    eps: complex
    mu: complex
    k: complex

    @property
    def omega(self) -> float:
        return 2.0 * np.pi * self.frequency_hz


@dataclass(frozen=True)
class _ProductionRTCoefficients:
    """Production complex128 layer-stack amplitudes and power coefficients."""

    r_te: np.ndarray
    r_tm: np.ndarray
    t_te: np.ndarray
    t_tm: np.ndarray
    R_te: np.ndarray
    R_tm: np.ndarray
    T_te: np.ndarray
    T_tm: np.ndarray
    A_te: np.ndarray
    A_tm: np.ndarray


def _complex_sqrt_passive(z):
    w = np.sqrt(np.asarray(z, dtype=np.complex128))
    return np.where(w.imag > 0.0, -w, w)


def _medium_params(eps_r, sigma_e, mu_r, frequency_hz) -> _ProductionMedium:
    omega = 2.0 * np.pi * float(frequency_hz)
    eps_rel = complex(eps_r) - 1j * float(sigma_e) / (omega * EPS0)
    mu_rel = complex(mu_r)
    k0 = omega / C0
    k = k0 * complex(_complex_sqrt_passive(eps_rel * mu_rel))
    return _ProductionMedium(
        frequency_hz=float(frequency_hz),
        eps=eps_rel * EPS0,
        mu=mu_rel * MU0,
        k=k,
    )


def _vacuum_medium(frequency_hz) -> _ProductionMedium:
    return _medium_params(1.0, 0.0, 1.0, frequency_hz)


def _admittances(medium: _ProductionMedium, k_z):
    return k_z / (medium.omega * medium.mu), medium.omega * medium.eps / k_z


def _power_coefficients(r, t, y1, y2):
    big_r = np.abs(r) ** 2
    big_t = (np.real(y2) / np.real(y1)) * np.abs(t) ** 2
    return big_r, big_t, 1.0 - big_r - big_t


def _stack_rt_one_pol(y_out, y_layers, deltas, y_back):
    m11 = np.ones_like(y_out)
    m12 = np.zeros_like(y_out)
    m21 = np.zeros_like(y_out)
    m22 = np.ones_like(y_out)
    log_scale = np.zeros(np.shape(y_out), dtype=np.float64)
    for y, delta in zip(y_layers, deltas):
        a = -np.imag(delta)
        e_plus = np.exp(1j * np.real(delta))
        e_minus = np.exp(-2.0 * a - 1j * np.real(delta))
        cos_s = 0.5 * (e_plus + e_minus)
        sin_s = (e_plus - e_minus) / 2j
        l11 = cos_s
        l12 = 1j * sin_s / y
        l21 = 1j * y * sin_s
        l22 = cos_s
        m11, m12, m21, m22 = (
            m11 * l11 + m12 * l21,
            m11 * l12 + m12 * l22,
            m21 * l11 + m22 * l21,
            m21 * l12 + m22 * l22,
        )
        log_scale = log_scale + a
    b = m11 + m12 * y_back
    c = m21 + m22 * y_back
    denom = y_out * b + c
    r = (y_out * b - c) / denom
    t = (2.0 * y_out / denom) * np.exp(-log_scale)
    return r, t


def layer_stack_rt(
    layers: Sequence[tuple],
    cos_theta_i,
    frequency_hz,
    outside: _ProductionMedium | None = None,
    backing: _ProductionMedium | None = None,
) -> _ProductionRTCoefficients:
    """Production NumPy precompute for a planar material layer stack."""

    if outside is None:
        outside = _vacuum_medium(frequency_hz)
    if backing is None:
        backing = _vacuum_medium(frequency_hz)
    for medium in (outside, backing):
        if medium.frequency_hz != float(frequency_hz):
            raise ValueError("outside/backing media frequency mismatch")
    cos_i = np.asarray(cos_theta_i, dtype=np.float64)
    sin2_i = 1.0 - cos_i * cos_i
    k_par2 = outside.k * outside.k * sin2_i

    def kz(medium: _ProductionMedium):
        return _complex_sqrt_passive(medium.k * medium.k - k_par2)

    kz_out, kz_back = kz(outside), kz(backing)
    y_out = _admittances(outside, kz_out)
    y_back = _admittances(backing, kz_back)
    y_te_layers, y_tm_layers, deltas = [], [], []
    for thickness_m, eps_r, sigma_e, mu_r in layers:
        medium = _medium_params(eps_r, sigma_e, mu_r, frequency_hz)
        kz_l = kz(medium)
        y_te, y_tm = _admittances(medium, kz_l)
        y_te_layers.append(y_te)
        y_tm_layers.append(y_tm)
        deltas.append(kz_l * float(thickness_m))
    ones = np.ones_like(cos_i, dtype=np.complex128)
    r_te, t_te = _stack_rt_one_pol(y_out[0] * ones, y_te_layers, deltas, y_back[0])
    r_tm, t_tm = _stack_rt_one_pol(y_out[1] * ones, y_tm_layers, deltas, y_back[1])
    big_r_te, big_t_te, big_a_te = _power_coefficients(r_te, t_te, y_out[0], y_back[0])
    big_r_tm, big_t_tm, big_a_tm = _power_coefficients(r_tm, t_tm, y_out[1], y_back[1])
    return _ProductionRTCoefficients(
        r_te=r_te,
        r_tm=r_tm,
        t_te=t_te,
        t_tm=t_tm,
        R_te=big_r_te,
        R_tm=big_r_tm,
        T_te=big_t_te,
        T_tm=big_t_tm,
        A_te=big_a_te,
        A_tm=big_a_tm,
    )


def _require_frequency_ad_constant_materials(
    scene: Scene, compiled: object, *, ad_mode: str
) -> None:
    """Explicit-failure contract for frequency AD over dispersive materials.

    Channel ``scene.compile()`` freezes material records at the primal frequency, so
    a frequency gradient through a scene with frequency-dependent material
    laws would silently miss d(material)/d(frequency) (plan 07 section 7:
    never return misleading gradients). Fail before any launch instead.
    """

    dependent = tuple(compiled.materials.frequency_dependent)
    if not dependent or not _frequency_participates_in_ad(scene.frequency):
        return
    raise NotImplementedError(
        f"ad_mode='{ad_mode}' cannot differentiate with respect to frequency "
        "in this scene: material records are frozen at the primal frequency "
        "at compile time, so the gradient would silently miss "
        "d(material)/d(frequency) for the frequency-dependent materials "
        f"{list(dependent)}. Use a constant-material scene for frequency AD, "
        "or drop the frequency requires_grad/tangent for materials-only AD."
    )
