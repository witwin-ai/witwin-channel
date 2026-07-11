from __future__ import annotations

from dataclasses import dataclass

# Model id for PerfectConductor materials in Material.parameters().
PEC_MODEL_ID = 2
# Effective conductivity used when evaluating Fresnel coefficients for PEC
# materials: the field kernels only receive (eps_r, sigma_e, mu_r), and the
# PEC limit sigma -> inf is reached to <1e-4 in |r|^2 at this value for the
# supported frequency range.
PEC_EFFECTIVE_SIGMA_E = 1.0e9


def effective_sigma_e(parameters: dict[str, float | int]) -> float:
    """Conductivity to feed the Fresnel kernels for a material parameter dict."""

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

    def parameters(self) -> dict[str, float | int]:
        return {
            "eps_r": float(self.eps_r),
            "mu_r": float(self.mu_r),
            "sigma_e": float(self.sigma_e),
            "gain": float(self.gain),
            "thickness_m": float(self.thickness_m),
            "model_id": 1,
        }


@dataclass(frozen=True, slots=True)
class LossyDielectric(Dielectric):
    sigma_e: float = 0.0


@dataclass(frozen=True, slots=True)
class PerfectConductor:
    gain: float = 1.0
    thickness_m: float = 0.1

    def parameters(self) -> dict[str, float | int]:
        return {
            "eps_r": 1.0,
            "mu_r": 1.0,
            "sigma_e": 0.0,
            "gain": float(self.gain),
            "thickness_m": float(self.thickness_m),
            "model_id": 2,
        }
