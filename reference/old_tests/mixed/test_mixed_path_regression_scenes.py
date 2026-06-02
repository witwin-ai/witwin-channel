"""Regression scenes for mixed-path diffraction family reconstruction."""

import math
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import witwin as wt

import drjit as dr

from witwin.channel import Field
from witwin.channel.trace.diffraction import (
    _accumulate_state_subset_field,
    _build_state_audit,
    compute_diffraction_field,
    _prepare_diffraction_state_arrays,
)
from witwin.channel.kernels.trace.packed_state import subset_state_arrays
from witwin.channel.trace import compute_reflection_field
from witwin.channel.validation import build_inserted_reflection_case, build_mixed_prefix_suffix_case
FREQ = 1e9
WAVELENGTH = 299792458.0 / FREQ
K = 2.0 * math.pi / WAVELENGTH
REFLECTION_COEF = 0.7
REFLECTION_N_RAYS = 1024
GRID_SIZE = 28
MATERIAL_DETAIL = {
    "relative_permittivity": 5.0,
    "conductivity": 0.0,
    "gain": REFLECTION_COEF,
}


def field_power(field):
    return float(dr.sum(field.real * field.real + field.imag * field.imag)[0])


def field_sub(lhs, rhs):
    return wt.Complex2f(lhs.real - rhs.real, lhs.imag - rhs.imag)


def field_add(*fields):
    total = wt.Complex2f(0.0, 0.0)
    for field in fields:
        total = wt.Complex2f(total.real + field.real, total.imag + field.imag)
    return total


def prepare_case(case, max_diffractions):
    field = Field(bounds=(case.range_x, case.range_y), size=(GRID_SIZE, GRID_SIZE))
    coords = field.get_coordinates()
    tx = wt.Point3f(*case.tx_pos)
    _, _, reflection_detail = compute_reflection_field(
        grid=field,
        rx_z=case.calculation_height,
        tx_pos=tx,
        scene=case.scene,
        wavelength=WAVELENGTH,
        k=K,
        n_rays=REFLECTION_N_RAYS,
        max_reflections=1,
        reflection_coef=REFLECTION_COEF,
        grid_data=coords,
    )
    _, edge_data, state_arrays, _ = _prepare_diffraction_state_arrays(
        tx_pos=tx,
        rx_z=case.calculation_height,
        scene=case.scene,
        wavelength=WAVELENGTH,
        k=K,
        reflection_detail=reflection_detail,
        material_detail=MATERIAL_DETAIL,
        reflection_n_rays=REFLECTION_N_RAYS,
        reflection_max_bounces=1,
        reflection_coef=REFLECTION_COEF,
        reflection_mode="2d",
        max_diffractions=max_diffractions,
    )
    rx_pos = wt.Point3f(coords["X"], coords["Y"], wt.Float(case.calculation_height))
    audit = _build_state_audit(state_arrays, edge_data)
    suffix_rays_per_state = max(128, int(REFLECTION_N_RAYS / max(1, int(state_arrays["n_states"]))))
    return {
        "field": field,
        "coords": coords,
        "tx": tx,
        "reflection_detail": reflection_detail,
        "edge_data": edge_data,
        "state_arrays": state_arrays,
        "rx_pos": rx_pos,
        "audit": audit,
        "case": case,
        "suffix_rays_per_state": suffix_rays_per_state,
    }


def accumulate_subset(context, mask, with_suffix):
    subset = subset_state_arrays(context["state_arrays"], mask)
    suffix_n_rays = 0
    if with_suffix:
        suffix_n_rays = int(subset["n_states"]) * int(context["suffix_rays_per_state"])
    from witwin.channel.config import ReflectionSuffixConfig
    return _accumulate_state_subset_field(
        state_arrays=subset,
        rx_pos=context["rx_pos"],
        scene=context["case"].scene,
        wavelength=WAVELENGTH,
        k=K,
        n_edges=context["edge_data"]["n_edges"],
        material_detail=MATERIAL_DETAIL,
        suffix=ReflectionSuffixConfig(
            n_rays=suffix_n_rays,
            max_bounces=1 if with_suffix else 0,
            coef=REFLECTION_COEF,
            mode="2d",
            detail=context["reflection_detail"],
            grid=context["field"] if with_suffix else None,
            grid_data=context["coords"] if with_suffix else None,
            rx_z=context["case"].calculation_height,
        ),
    )


def test_prefix_suffix_scene_reconstructs_first_order_mixed_field():
    case = build_mixed_prefix_suffix_case()
    context = prepare_case(case, max_diffractions=1)
    state_arrays = context["state_arrays"]

    direct_order1_mask = (
        (state_arrays["order"] == wt.UInt32(1))
        & (state_arrays["prefix_reflection_depth"] == wt.UInt32(0))
        & (state_arrays["intermediate_reflection_depth"] == wt.UInt32(0))
    )
    prefix_order1_mask = (
        (state_arrays["order"] == wt.UInt32(1))
        & (state_arrays["prefix_reflection_depth"] > wt.UInt32(0))
        & (state_arrays["intermediate_reflection_depth"] == wt.UInt32(0))
    )

    direct_base = accumulate_subset(context, direct_order1_mask, with_suffix=False)
    direct_with_suffix = accumulate_subset(context, direct_order1_mask, with_suffix=True)
    prefix_base = accumulate_subset(context, prefix_order1_mask, with_suffix=False)
    prefix_with_suffix = accumulate_subset(context, prefix_order1_mask, with_suffix=True)

    d_to_r = field_sub(direct_with_suffix, direct_base)
    r_to_d_to_r = field_sub(prefix_with_suffix, prefix_base)
    reconstructed_mixed = field_add(prefix_base, d_to_r, r_to_d_to_r)

    _, _, _, dif_components = compute_diffraction_field(
        context["coords"]["X"],
        context["coords"]["Y"],
        case.calculation_height,
        context["tx"],
        case.scene,
        WAVELENGTH,
        K,
        reflection_detail=context["reflection_detail"],
        max_diffractions=1,
        reflection_n_rays=REFLECTION_N_RAYS,
        reflection_max_bounces=1,
        reflection_coef=REFLECTION_COEF,
        reflection_mode="2d",
        grid=context["field"],
        grid_data=context["coords"],
        return_components=True,
        return_per_edge=False,
        diffraction_material=MATERIAL_DETAIL,
    )

    mixed_error = field_sub(dif_components["a_multi"], reconstructed_mixed)
    assert field_power(mixed_error) < 1e-12
    assert field_power(prefix_base) > 1e-8
    assert field_power(d_to_r) > 1e-12
    assert field_power(r_to_d_to_r) > 1e-14

    audit = context["audit"]
    prefix_mask_np = np.asarray(prefix_order1_mask)
    assert np.any(prefix_mask_np)
    assert all(audit["path_sequence"][idx] == "S -> R -> D" for idx in np.flatnonzero(prefix_mask_np))


def test_prefix_first_order_states_keep_slope_term_disabled():
    case = build_mixed_prefix_suffix_case()
    context = prepare_case(case, max_diffractions=1)
    state_arrays = context["state_arrays"]

    prefix_order1_mask = (
        (state_arrays["order"] == wt.UInt32(1))
        & (state_arrays["prefix_reflection_depth"] > wt.UInt32(0))
        & (state_arrays["intermediate_reflection_depth"] == wt.UInt32(0))
    )
    prefix_states = subset_state_arrays(state_arrays, prefix_order1_mask)

    assert int(prefix_states["n_states"]) > 0
    assert field_power(prefix_states["incident_normal_derivative"]) < 1e-16


def test_inserted_reflection_scene_reconstructs_second_order_mixed_field():
    case = build_inserted_reflection_case()
    context = prepare_case(case, max_diffractions=2)
    state_arrays = context["state_arrays"]

    order2_prefix_recursive_mask = (
        (state_arrays["order"] == wt.UInt32(2))
        & (state_arrays["prefix_reflection_depth"] > wt.UInt32(0))
        & (state_arrays["intermediate_reflection_depth"] == wt.UInt32(0))
    )
    order2_inserted_direct_mask = (
        (state_arrays["order"] == wt.UInt32(2))
        & (state_arrays["prefix_reflection_depth"] == wt.UInt32(0))
        & (state_arrays["intermediate_reflection_depth"] > wt.UInt32(0))
    )
    order2_inserted_prefix_mask = (
        (state_arrays["order"] == wt.UInt32(2))
        & (state_arrays["prefix_reflection_depth"] > wt.UInt32(0))
        & (state_arrays["intermediate_reflection_depth"] > wt.UInt32(0))
    )
    order2_mixed_mask = (
        (state_arrays["order"] == wt.UInt32(2))
        & (
            (state_arrays["prefix_reflection_depth"] > wt.UInt32(0))
            | (state_arrays["intermediate_reflection_depth"] > wt.UInt32(0))
        )
    )

    prefix_recursive_field = accumulate_subset(context, order2_prefix_recursive_mask, with_suffix=False)
    inserted_direct_field = accumulate_subset(context, order2_inserted_direct_mask, with_suffix=False)
    inserted_prefix_field = accumulate_subset(context, order2_inserted_prefix_mask, with_suffix=False)
    order2_mixed_total = accumulate_subset(context, order2_mixed_mask, with_suffix=False)

    reconstructed_order2_mixed = field_add(
        prefix_recursive_field,
        inserted_direct_field,
        inserted_prefix_field,
    )
    order2_error = field_sub(order2_mixed_total, reconstructed_order2_mixed)

    assert field_power(order2_error) < 1e-12
    assert field_power(inserted_direct_field) > 1e-12

    audit = context["audit"]
    inserted_direct_mask_np = np.asarray(order2_inserted_direct_mask)
    assert np.any(inserted_direct_mask_np)
    assert "S -> D -> R -> D" in {
        audit["path_sequence"][idx] for idx in np.flatnonzero(inserted_direct_mask_np)
    }


