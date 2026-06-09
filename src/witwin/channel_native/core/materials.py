from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Dielectric:
    eps_r: float
    mu_r: float = 1.0
    sigma_e: float = 0.0
    gain: float = 1.0

    def parameters(self) -> dict[str, float | int]:
        return {
            "eps_r": float(self.eps_r),
            "mu_r": float(self.mu_r),
            "sigma_e": float(self.sigma_e),
            "gain": float(self.gain),
            "model_id": 1,
        }


@dataclass(frozen=True, slots=True)
class LossyDielectric(Dielectric):
    sigma_e: float = 0.0


@dataclass(frozen=True, slots=True)
class PerfectConductor:
    gain: float = 1.0

    def parameters(self) -> dict[str, float | int]:
        return {
            "eps_r": 1.0,
            "mu_r": 1.0,
            "sigma_e": 0.0,
            "gain": float(self.gain),
            "model_id": 2,
        }
