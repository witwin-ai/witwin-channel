from __future__ import annotations

from dataclasses import dataclass
import math


MATERIAL_ABI_VERSION = 2
DIELECTRIC_MODEL_ID = 1
PEC_MODEL_ID = 2
DISPERSIVE_MODEL_ID = 3

# Native Fresnel kernels still consume finite scalar tensors. This value is
# only the ABI-v1 kernel encoding of the explicit PEC model; PEC identity is
# retained separately in model_id and is never inferred from conductivity.
PEC_EFFECTIVE_SIGMA_E = 1.0e9


def _positive(name: str, value: float) -> float:
    value = float(value)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return value


def _nonnegative(name: str, value: float) -> float:
    value = float(value)
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return value


def _unit_interval(name: str, value: float) -> float:
    value = float(value)
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be in [0, 1]")
    return value


def effective_sigma_e(parameters: dict[str, float | int | str]) -> float:
    """Encode an explicit material model for legacy finite-sigma kernels."""

    sigma = float(parameters["sigma_e"])
    if int(parameters.get("model_id", 0)) == PEC_MODEL_ID:
        return max(sigma, PEC_EFFECTIVE_SIGMA_E)
    return sigma


@dataclass(frozen=True, slots=True)
class Dielectric:
    eps_r: float
    mu_r: float = 1.0
    sigma_e: float = 0.0
    gain: float = 1.0
    thickness_m: float = 0.1
    scattering_coefficient: float = 0.0
    xpd_coefficient: float = 0.0
    name: str = "dielectric"

    def parameters(
        self, frequency_hz: float | None = None
    ) -> dict[str, float | int | str]:
        del frequency_hz
        return {
            "eps_r": _positive("eps_r", self.eps_r),
            "mu_r": _positive("mu_r", self.mu_r),
            "sigma_e": _nonnegative("sigma_e", self.sigma_e),
            "gain": _nonnegative("gain", self.gain),
            "thickness_m": _positive("thickness_m", self.thickness_m),
            "scattering_coefficient": _unit_interval(
                "scattering_coefficient", self.scattering_coefficient
            ),
            "xpd_coefficient": _unit_interval("xpd_coefficient", self.xpd_coefficient),
            "model_id": DIELECTRIC_MODEL_ID,
            "name": str(self.name),
        }


@dataclass(frozen=True, slots=True)
class LossyDielectric(Dielectric):
    sigma_e: float = 0.0


@dataclass(frozen=True, slots=True)
class DispersiveMaterial:
    """Power-law dispersive dielectric evaluated at scene frequency.

    ``eps_r = eps_r_ref * (f/f_ref)**eps_r_exponent`` and the same convention
    applies to conductivity. This matches the ITU-R P.2040 material form while
    remaining usable for custom measured power-law fits.
    """

    eps_r_ref: float
    sigma_e_ref: float
    reference_frequency_hz: float = 1.0e9
    eps_r_exponent: float = 0.0
    sigma_e_exponent: float = 0.0
    mu_r: float = 1.0
    gain: float = 1.0
    thickness_m: float = 0.1
    scattering_coefficient: float = 0.0
    xpd_coefficient: float = 0.0
    name: str = "dispersive"

    def parameters(
        self, frequency_hz: float | None = None
    ) -> dict[str, float | int | str]:
        if frequency_hz is None:
            raise ValueError("frequency_hz is required for a dispersive material")
        frequency_hz = _positive("frequency_hz", frequency_hz)
        reference = _positive("reference_frequency_hz", self.reference_frequency_hz)
        ratio = frequency_hz / reference
        return {
            "eps_r": _positive("eps_r", self.eps_r_ref * ratio**self.eps_r_exponent),
            "mu_r": _positive("mu_r", self.mu_r),
            "sigma_e": _nonnegative(
                "sigma_e", self.sigma_e_ref * ratio**self.sigma_e_exponent
            ),
            "gain": _nonnegative("gain", self.gain),
            "thickness_m": _positive("thickness_m", self.thickness_m),
            "scattering_coefficient": _unit_interval(
                "scattering_coefficient", self.scattering_coefficient
            ),
            "xpd_coefficient": _unit_interval("xpd_coefficient", self.xpd_coefficient),
            "model_id": DISPERSIVE_MODEL_ID,
            "name": str(self.name),
        }


@dataclass(frozen=True, slots=True)
class ITUMaterial:
    """Frequency-dependent ITU-R P.2040 radio material."""

    name: str
    thickness_m: float = 0.1
    scattering_coefficient: float = 0.0
    xpd_coefficient: float = 0.0
    gain: float = 1.0

    def parameters(
        self, frequency_hz: float | None = None
    ) -> dict[str, float | int | str]:
        if frequency_hz is None:
            raise ValueError("frequency_hz is required for an ITU material")
        from .scene_loader import itu_material_parameters

        eps_r, sigma_e = itu_material_parameters(self.name, frequency_hz)
        return DispersiveMaterial(
            eps_r_ref=eps_r,
            sigma_e_ref=sigma_e,
            reference_frequency_hz=frequency_hz,
            thickness_m=self.thickness_m,
            scattering_coefficient=self.scattering_coefficient,
            xpd_coefficient=self.xpd_coefficient,
            gain=self.gain,
            name=self.name,
        ).parameters(frequency_hz)


@dataclass(frozen=True, slots=True)
class PerfectConductor:
    gain: float = 1.0
    thickness_m: float = 0.1
    name: str = "perfect_conductor"

    def parameters(
        self, frequency_hz: float | None = None
    ) -> dict[str, float | int | str]:
        del frequency_hz
        return {
            "eps_r": 1.0,
            "mu_r": 1.0,
            "sigma_e": 0.0,
            "gain": _nonnegative("gain", self.gain),
            "thickness_m": _positive("thickness_m", self.thickness_m),
            "scattering_coefficient": 0.0,
            "xpd_coefficient": 0.0,
            "model_id": PEC_MODEL_ID,
            "name": str(self.name),
        }
