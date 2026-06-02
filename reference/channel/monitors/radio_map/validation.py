from __future__ import annotations

from . import types as rm_types


def validate_shadow_boundary_contract(
    *,
    shadow_boundary_mode: rm_types.ShadowBoundaryMode,
    combine_mode: rm_types.CombineMode,
    receiver_model: rm_types.ReceiverModel,
    quadrature_mode: str,
) -> None:
    if shadow_boundary_mode == rm_types.ShadowBoundaryMode.UTD_CROSS_TERM_SURROGATE and (
        combine_mode != rm_types.CombineMode.INCOHERENT
        or receiver_model != rm_types.ReceiverModel.MATCHED_ISOTROPIC
    ):
        raise ValueError(
            "shadow_boundary_mode='utd_cross_term_surrogate' requires "
            "combine_mode='incoherent' and receiver_model='matched_isotropic'."
        )
    if shadow_boundary_mode == rm_types.ShadowBoundaryMode.PROJECTED_ISB_COMPLETION and (
        combine_mode != rm_types.CombineMode.COHERENT
        or receiver_model != rm_types.ReceiverModel.PROJECTED_POLARIZED
        or quadrature_mode != "center"
    ):
        raise ValueError(
            "shadow_boundary_mode='projected_isb_completion' requires "
            "combine_mode='coherent', receiver_model='projected_polarized', "
            "and quadrature_mode='center'."
        )
    if shadow_boundary_mode == rm_types.ShadowBoundaryMode.MATCHED_ISB_COMPLETION and (
        combine_mode != rm_types.CombineMode.COHERENT
        or receiver_model != rm_types.ReceiverModel.MATCHED_ISOTROPIC
        or quadrature_mode != "center"
    ):
        raise ValueError(
            "shadow_boundary_mode='matched_isb_completion' requires "
            "combine_mode='coherent', receiver_model='matched_isotropic', "
            "and quadrature_mode='center'."
        )


def validate_monte_carlo_contract(
    *,
    sampling_mode: rm_types.SamplingMode,
    surface_mode: rm_types.SurfaceMode,
    max_diffractions: int | None,
    combine_mode: rm_types.CombineMode,
    receiver_model: rm_types.ReceiverModel,
    accumulation_backend: rm_types.AccumulationBackend,
    shadow_boundary_mode: rm_types.ShadowBoundaryMode,
) -> None:
    if sampling_mode != rm_types.SamplingMode.MONTE_CARLO:
        return
    if surface_mode != rm_types.SurfaceMode.AXIS_ALIGNED:
        raise ValueError(
            "sampling_mode='monte_carlo' currently requires an axis-aligned radio-map surface."
        )
    if max_diffractions not in (None, 0, 1):
        raise ValueError(
            "sampling_mode='monte_carlo' currently supports only max_diffractions <= 1."
        )
    if combine_mode != rm_types.CombineMode.INCOHERENT:
        raise ValueError(
            "sampling_mode='monte_carlo' currently requires combine_mode='incoherent'."
        )
    if receiver_model != rm_types.ReceiverModel.MATCHED_ISOTROPIC:
        raise ValueError(
            "sampling_mode='monte_carlo' currently requires receiver_model='matched_isotropic'."
        )
    if accumulation_backend not in {
        rm_types.AccumulationBackend.AUTO,
        rm_types.AccumulationBackend.NATIVE_MONTE_CARLO,
    }:
        raise ValueError(
            "sampling_mode='monte_carlo' requires accumulation_backend='auto' or "
            "'native_monte_carlo'."
        )
    if shadow_boundary_mode != rm_types.ShadowBoundaryMode.NONE:
        raise ValueError(
            "sampling_mode='monte_carlo' currently requires shadow_boundary_mode='none'."
        )


__all__ = [
    "validate_monte_carlo_contract",
    "validate_shadow_boundary_contract",
]
