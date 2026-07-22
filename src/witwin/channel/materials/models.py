from __future__ import annotations

from dataclasses import dataclass
import math


MATERIAL_ABI_VERSION = 3
DIELECTRIC_MODEL_ID = 1
PEC_MODEL_ID = 2
DISPERSIVE_MODEL_ID = 3
PHYSICAL_SURFACE_MODEL_ID = 4

GEOMETRY_MODE_IDS = {"thin_sheet": 0, "closed_volume": 1}

_VACUUM_PERMITTIVITY = 8.8541878128e-12  # F/m

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


def _finite(name: str, value: float) -> float:
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


@dataclass(frozen=True, slots=True)
class DebyeModel:
    """Single-pole Debye dispersion with an optional DC conductivity term.

    ``eps(w) = eps_inf + delta_eps/(1 + j*w*tau_s) - j*sigma_dc/(w*eps0)``
    under the ``e^{+j w t}`` time convention, so ``Im(eps) <= 0`` for a
    passive medium (``delta_eps >= 0``, ``sigma_dc >= 0``).
    """

    eps_inf: float
    delta_eps: float
    tau_s: float
    sigma_dc: float = 0.0

    def __post_init__(self) -> None:
        _positive("eps_inf", self.eps_inf)
        _nonnegative("delta_eps", self.delta_eps)
        _positive("tau_s", self.tau_s)
        _nonnegative("sigma_dc", self.sigma_dc)

    def complex_eps(self, frequency_hz: float) -> complex:
        omega = 2.0 * math.pi * _positive("frequency_hz", frequency_hz)
        return (
            self.eps_inf
            + self.delta_eps / (1.0 + 1j * omega * self.tau_s)
            - 1j * self.sigma_dc / (omega * _VACUUM_PERMITTIVITY)
        )


@dataclass(frozen=True, slots=True)
class TabulatedPermittivity:
    """Measured complex relative permittivity, linearly interpolated in f.

    ``eps_imag`` is the signed imaginary part under the ``e^{+j w t}``
    convention and must be ``<= 0`` for passive media. Evaluation outside
    the tabulated frequency range is an error, never an extrapolation.
    """

    frequency_hz: tuple[float, ...]
    eps_real: tuple[float, ...]
    eps_imag: tuple[float, ...]

    def __post_init__(self) -> None:
        frequency = tuple(float(f) for f in self.frequency_hz)
        eps_real = tuple(float(v) for v in self.eps_real)
        eps_imag = tuple(float(v) for v in self.eps_imag)
        if len(frequency) < 2:
            raise ValueError("frequency_hz must contain at least two samples")
        if not (len(frequency) == len(eps_real) == len(eps_imag)):
            raise ValueError(
                "frequency_hz, eps_real, eps_imag must have the same length"
            )
        for f in frequency:
            _positive("frequency_hz", f)
        if any(b <= a for a, b in zip(frequency, frequency[1:])):
            raise ValueError("frequency_hz must be strictly increasing")
        for v in eps_real:
            _positive("eps_real", v)
        for v in eps_imag:
            if not math.isfinite(v) or v > 0.0:
                raise ValueError("eps_imag must be finite and <= 0 (passive)")
        object.__setattr__(self, "frequency_hz", frequency)
        object.__setattr__(self, "eps_real", eps_real)
        object.__setattr__(self, "eps_imag", eps_imag)

    def complex_eps(self, frequency_hz: float) -> complex:
        f = _positive("frequency_hz", frequency_hz)
        table = self.frequency_hz
        if f < table[0] or f > table[-1]:
            raise ValueError(
                f"frequency_hz {f:g} is outside the tabulated range "
                f"[{table[0]:g}, {table[-1]:g}]"
            )
        hi = next(i for i, fi in enumerate(table) if fi >= f)
        if table[hi] == f:
            return complex(self.eps_real[hi], self.eps_imag[hi])
        lo = hi - 1
        w = (f - table[lo]) / (table[hi] - table[lo])
        return complex(
            self.eps_real[lo] + w * (self.eps_real[hi] - self.eps_real[lo]),
            self.eps_imag[lo] + w * (self.eps_imag[hi] - self.eps_imag[lo]),
        )


@dataclass(frozen=True, slots=True)
class Layer:
    """One homogeneous slab of a layered physical surface.

    Exactly one of ``eps_r`` (constant permittivity, optionally with a
    conduction loss ``sigma_e``) or ``eps_model`` (dispersive model that
    already carries its loss) must be provided.
    """

    thickness_m: float
    eps_r: float | None = None
    sigma_e: float = 0.0
    mu_r: float = 1.0
    eps_model: DebyeModel | TabulatedPermittivity | None = None

    def __post_init__(self) -> None:
        _positive("thickness_m", self.thickness_m)
        _positive("mu_r", self.mu_r)
        if (self.eps_r is None) == (self.eps_model is None):
            raise ValueError("Layer requires exactly one of eps_r or eps_model")
        if self.eps_model is None:
            _positive("eps_r", self.eps_r)
            _nonnegative("sigma_e", self.sigma_e)
        elif float(self.sigma_e) != 0.0:
            raise ValueError(
                "Layer sigma_e must be 0 when eps_model is given; "
                "put the loss in the dispersion model"
            )

    def complex_eps(self, frequency_hz: float) -> complex:
        if self.eps_model is not None:
            return self.eps_model.complex_eps(frequency_hz)
        omega = 2.0 * math.pi * _positive("frequency_hz", frequency_hz)
        return complex(
            self.eps_r, -self.sigma_e / (omega * _VACUUM_PERMITTIVITY)
        )

    def parameters(self, frequency_hz: float) -> tuple[float, float, float]:
        """Return ``(eps_r_real, sigma_e_equiv, mu_r)`` at ``frequency_hz``.

        ``Im(eps)`` is folded into an equivalent conductivity
        ``sigma_e_equiv = -Im(eps)*w*eps0`` at that frequency.
        """

        if self.eps_model is None:
            return (float(self.eps_r), float(self.sigma_e), float(self.mu_r))
        eps = self.eps_model.complex_eps(frequency_hz)
        omega = 2.0 * math.pi * float(frequency_hz)
        return (eps.real, -eps.imag * omega * _VACUUM_PERMITTIVITY, float(self.mu_r))


@dataclass(frozen=True, slots=True)
class Roughness:
    """Gaussian-correlated surface roughness statistics (meters, radians)."""

    rms_height_m: float
    corr_length_x_m: float
    corr_length_y_m: float
    principal_axis_rad: float = 0.0
    correlation: str = "gaussian"

    def __post_init__(self) -> None:
        _nonnegative("rms_height_m", self.rms_height_m)
        _positive("corr_length_x_m", self.corr_length_x_m)
        _positive("corr_length_y_m", self.corr_length_y_m)
        _finite("principal_axis_rad", self.principal_axis_rad)
        if self.correlation != "gaussian":
            raise ValueError(
                f"correlation must be 'gaussian', got {self.correlation!r}"
            )


@dataclass(frozen=True, slots=True)
class PhysicalSurface:
    """Layered, optionally rough surface material (material ABI v3).

    Legacy scalar fields (eps_r, sigma_e, mu_r, thickness_m) are the
    layer-0 view; multilayer surfaces flag ``legacy_scalar_approximation``.
    Missing roughness means smooth; roughness is never guessed.
    """

    layers: tuple[Layer, ...]
    geometry_mode: str = "thin_sheet"
    roughness_front: Roughness | None = None
    roughness_back: Roughness | None = None
    name: str | None = None

    def __post_init__(self) -> None:
        layers = tuple(self.layers)
        if not layers:
            raise ValueError("layers must contain at least one Layer")
        for layer in layers:
            if not isinstance(layer, Layer):
                raise ValueError(f"layers entries must be Layer, got {type(layer).__name__}")
        object.__setattr__(self, "layers", layers)
        if self.geometry_mode not in GEOMETRY_MODE_IDS:
            raise ValueError(
                f"geometry_mode must be one of {sorted(GEOMETRY_MODE_IDS)}, "
                f"got {self.geometry_mode!r}"
            )
        for field_name, roughness in (
            ("roughness_front", self.roughness_front),
            ("roughness_back", self.roughness_back),
        ):
            if roughness is not None and not isinstance(roughness, Roughness):
                raise ValueError(f"{field_name} must be a Roughness or None")

    def layer_parameters(
        self, frequency_hz: float
    ) -> tuple[tuple[float, float, float, float], ...]:
        """Per-layer ``(thickness_m, eps_r_real, sigma_e_equiv, mu_r)``."""

        return tuple(
            (float(layer.thickness_m), *layer.parameters(frequency_hz))
            for layer in self.layers
        )

    def parameters(
        self, frequency_hz: float | None = None
    ) -> dict[str, float | int | str | list | bool | None]:
        if frequency_hz is None:
            raise ValueError("frequency_hz is required for a physical surface")
        layer_rows = self.layer_parameters(frequency_hz)
        eps_r0, sigma_e0, mu_r0 = self.layers[0].parameters(frequency_hz)
        rough = self.roughness_front
        return {
            # Legacy scalar view: layer 0 only.
            "eps_r": _positive("eps_r", eps_r0),
            "mu_r": _positive("mu_r", mu_r0),
            "sigma_e": _nonnegative("sigma_e", sigma_e0),
            "gain": 1.0,
            "thickness_m": float(self.layers[0].thickness_m),
            "scattering_coefficient": 0.0,
            "xpd_coefficient": 0.0,
            "model_id": PHYSICAL_SURFACE_MODEL_ID,
            "name": str(self.name or "physical_surface"),
            # ABI v3 payload.
            "layers": [list(row) for row in layer_rows],
            "geometry_mode": self.geometry_mode,
            "roughness": None
            if rough is None
            else [
                float(rough.rms_height_m),
                float(rough.corr_length_x_m),
                float(rough.corr_length_y_m),
                float(rough.principal_axis_rad),
            ],
            "legacy_scalar_approximation": len(self.layers) > 1,
        }


@dataclass(frozen=True, slots=True)
class PhaseScreen:
    """Per-surface height realization for coherent rough scattering.

    ``height`` is a 2D torch tensor or nested list of rows; it is stored
    as-is and converted to a tensor lazily via :meth:`height_tensor`.
    Heights only enter complex phase; geometry is never displaced.
    """

    height: object
    height_scale_m: float
    height_offset_m: float = 0.0
    realization_id: int = 0
    mode: str = "realization_coherent"
    correlation: Roughness | None = None
    quadrature_tolerance: float = 1e-4

    def __post_init__(self) -> None:
        import torch

        if isinstance(self.height, torch.Tensor):
            if self.height.ndim != 2 or self.height.numel() == 0:
                raise ValueError("height tensor must be 2D and non-empty")
        elif isinstance(self.height, (list, tuple)):
            rows = self.height
            if not rows or not all(isinstance(row, (list, tuple)) for row in rows):
                raise ValueError("height must be a non-empty 2D nested sequence")
            widths = {len(row) for row in rows}
            if widths != {len(rows[0])} or len(rows[0]) == 0:
                raise ValueError("height rows must be non-empty and equal length")
        else:
            raise ValueError(
                f"height must be a torch.Tensor or nested sequence, "
                f"got {type(self.height).__name__}"
            )
        _positive("height_scale_m", self.height_scale_m)
        _finite("height_offset_m", self.height_offset_m)
        if int(self.realization_id) < 0:
            raise ValueError("realization_id must be non-negative")
        if self.mode not in ("realization_coherent", "ensemble_bsdf"):
            raise ValueError(
                "mode must be 'realization_coherent' or 'ensemble_bsdf', "
                f"got {self.mode!r}"
            )
        if self.correlation is not None and not isinstance(self.correlation, Roughness):
            raise ValueError("correlation must be a Roughness or None")
        _positive("quadrature_tolerance", self.quadrature_tolerance)

    def height_tensor(self) -> object:
        """Convert the stored height field to a float32 2D torch tensor."""

        import torch

        if isinstance(self.height, torch.Tensor):
            return self.height.to(dtype=torch.float32)
        return torch.tensor(self.height, dtype=torch.float32)

    def shape(self) -> tuple[int, int]:
        import torch

        if isinstance(self.height, torch.Tensor):
            return (int(self.height.shape[0]), int(self.height.shape[1]))
        return (len(self.height), len(self.height[0]))


@dataclass(frozen=True, slots=True)
class SurfaceAssignment:
    """Bind a material to a structure with an optional phase screen."""

    material: object
    phase_screen: PhaseScreen | None = None

    def __post_init__(self) -> None:
        if not hasattr(self.material, "parameters"):
            raise ValueError("material must expose a parameters(frequency_hz) method")
        if self.phase_screen is not None and not isinstance(
            self.phase_screen, PhaseScreen
        ):
            raise ValueError("phase_screen must be a PhaseScreen or None")


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

    def as_physical_surface(self) -> PhysicalSurface:
        """Single-layer PhysicalSurface view (gain/scattering fields drop)."""

        return PhysicalSurface(
            layers=(
                Layer(
                    thickness_m=self.thickness_m,
                    eps_r=self.eps_r,
                    sigma_e=self.sigma_e,
                    mu_r=self.mu_r,
                ),
            ),
            name=self.name,
        )

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


DebyeModel.__module__ = "witwin.channel.core.materials"
TabulatedPermittivity.__module__ = "witwin.channel.core.materials"
Layer.__module__ = "witwin.channel.core.materials"
Roughness.__module__ = "witwin.channel.core.materials"
PhysicalSurface.__module__ = "witwin.channel.core.materials"
PhaseScreen.__module__ = "witwin.channel.core.materials"
SurfaceAssignment.__module__ = "witwin.channel.core.materials"
Dielectric.__module__ = "witwin.channel.core.materials"
LossyDielectric.__module__ = "witwin.channel.core.materials"
DispersiveMaterial.__module__ = "witwin.channel.core.materials"
ITUMaterial.__module__ = "witwin.channel.core.materials"
PerfectConductor.__module__ = "witwin.channel.core.materials"
