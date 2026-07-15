from __future__ import annotations

import torch

from witwin.channel_native.runtime.symbols import required_symbol as _required_native_op
from witwin.channel_native.runtime.tensor_contracts import validate_cuda_tensor


def scattering_table_eval(
    wi: torch.Tensor,
    wo: torch.Tensor,
    f_te: torch.Tensor,
    f_tm: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Native CUDA multilinear Kirchhoff-table evaluation; required op."""

    validate_cuda_tensor("wi", wi, dtype=torch.float32, ndim=2)
    validate_cuda_tensor("wo", wo, dtype=torch.float32, ndim=2)
    validate_cuda_tensor("f_te", f_te, dtype=torch.float32, ndim=4)
    validate_cuda_tensor("f_tm", f_tm, dtype=torch.float32, ndim=4)
    if wi.shape != wo.shape or wi.shape[1:] != (3,):
        raise ValueError("wi and wo must have matching shape (N, 3)")
    out = _required_native_op("scattering_table_eval")(wi, wo, f_te, f_tm)
    if not isinstance(out, dict) or set(out) != {"f_te", "f_tm"}:
        raise TypeError("_channel_native.scattering_table_eval returned invalid fields")
    return out["f_te"], out["f_tm"]


def scattering_table_pdf(
    wi: torch.Tensor,
    wo: torch.Tensor,
    sample_density: torch.Tensor,
    *,
    reverse: bool = False,
) -> torch.Tensor:
    """Native CUDA piecewise-constant Kirchhoff PDF; required op."""

    validate_cuda_tensor("wi", wi, dtype=torch.float32, ndim=2)
    validate_cuda_tensor("wo", wo, dtype=torch.float32, ndim=2)
    validate_cuda_tensor("sample_density", sample_density, dtype=torch.float32, ndim=4)
    return _required_native_op("scattering_table_pdf")(
        wi, wo, sample_density, bool(reverse)
    )


def scattering_table_sample(
    wi: torch.Tensor,
    uniforms: torch.Tensor,
    marginal_cdf: torch.Tensor,
    conditional_cdf: torch.Tensor,
    sample_density: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Native CUDA CDF inversion plus forward/reverse PDFs; required op."""

    validate_cuda_tensor("wi", wi, dtype=torch.float32, ndim=2)
    validate_cuda_tensor("uniforms", uniforms, dtype=torch.float32, ndim=2)
    validate_cuda_tensor("marginal_cdf", marginal_cdf, dtype=torch.float32, ndim=3)
    validate_cuda_tensor("conditional_cdf", conditional_cdf, dtype=torch.float32, ndim=4)
    validate_cuda_tensor("sample_density", sample_density, dtype=torch.float32, ndim=4)
    out = _required_native_op("scattering_table_sample")(
        wi, uniforms, marginal_cdf, conditional_cdf, sample_density
    )
    expected = {"wo", "pdf_forward", "pdf_reverse"}
    if not isinstance(out, dict) or set(out) != expected:
        raise TypeError("_channel_native.scattering_table_sample returned invalid fields")
    return out


def scattering_event_probabilities(
    cos_theta: torch.Tensor,
    material_id: torch.Tensor,
    cap_r_te: torch.Tensor,
    cap_r_tm: torch.Tensor,
    cap_t_te: torch.Tensor,
    cap_t_tm: torch.Tensor,
    rough_sigma_h_m: torch.Tensor,
    scatter_model_id: torch.Tensor,
    *,
    frequency_hz: float,
    probability_floor: float,
) -> dict[str, torch.Tensor]:
    """Fused native CUDA rough-event budgets and probabilities."""

    validate_cuda_tensor("cos_theta", cos_theta, dtype=torch.float32, ndim=1)
    validate_cuda_tensor("material_id", material_id, dtype=torch.int32, ndim=1)
    out = _required_native_op("scattering_event_probabilities")(
        cos_theta,
        material_id,
        cap_r_te,
        cap_r_tm,
        cap_t_te,
        cap_t_tm,
        rough_sigma_h_m,
        scatter_model_id,
        float(frequency_hz),
        float(probability_floor),
    )
    expected = {"p_scatter", "p_transmit", "r_coh_amplitude", "rough"}
    if not isinstance(out, dict) or set(out) != expected:
        raise TypeError("_channel_native.scattering_event_probabilities returned invalid fields")
    return out


__all__ = [
    "scattering_event_probabilities",
    "scattering_table_eval",
    "scattering_table_pdf",
    "scattering_table_sample",
]
