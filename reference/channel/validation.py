"""Validation helpers for canonical multi-diffraction scenes."""

from __future__ import annotations

import math
from dataclasses import dataclass

import drjit as dr
import numpy as np
import witwin as wt

from .config import ReflectionSuffixConfig
from .trace.diffraction import (
    _accumulate_state_subset_field,
    _build_state_audit,
    _build_tx_first_order_state_arrays,
    _edge_state_field_to_targets,
    _empty_state_arrays,
    _finalize_state_lineage,
    _make_state_arrays,
    _segment_visibility_mask,
    _state_ids,
    _state_lineage_store,
    compute_diffraction_field,
    compute_diffraction_order_breakdown,
)
from .kernels.trace.packed_state import gather_state_arrays
from .utils.drjit_ops import eval_complex  # standalone helper
from .monitors import Field, FieldMonitor
from .scene import Scene
from .trace import Tracer
from .trace.diffraction.constants import _state_history_size
from .trace.diffraction.finite_wedge import require_edge_data_line_bounds
from .utils import to_power_db
from witwin.core import Box, Material, Prism, Structure


_LEGACY_PRISM_YAW_OFFSET = -math.pi / 2.0


def _cube_structure(*, name: str, center, size: float, rotation=None):
    return Structure(
        geometry=Box(
            position=center,
            size=(size, size, size),
            rotation=None if rotation is None else (0.0, 0.0, float(rotation)),
            device="cuda",
        ),
        material=Material(),
        name=name,
    )


def _prism_structure(*, name: str, n_sides: int, center, radius: float, height: float, rotation=None):
    yaw = _LEGACY_PRISM_YAW_OFFSET if rotation is None else float(rotation) + _LEGACY_PRISM_YAW_OFFSET
    return Structure(
        geometry=Prism(
            position=center,
            radius=radius,
            height=height,
            num_sides=n_sides,
            rotation=(0.0, 0.0, yaw),
            device="cuda",
        ),
        material=Material(),
        name=name,
    )


@dataclass
class ValidationCase:
    """Canonical wedge-like scene used for solver/reference comparisons."""

    name: str
    description: str
    scene: Scene
    tx_pos: tuple[float, float, float]
    range_x: tuple[float, float]
    range_y: tuple[float, float]
    calculation_height: float
    cut_axis: str
    cut_value: float

def build_double_wedge_case(
    center_gap: float = 3.4,
    radius: float = 0.95,
    height: float = 3.0,
    calculation_height: float = 1.5,
) -> ValidationCase:
    """Create a symmetric two-prism scene for double-diffraction benchmarking."""

    scene = Scene(
        structures=[
            _prism_structure(
                name="left",
                n_sides=3,
                center=(-center_gap / 2.0, 0.0, calculation_height),
                radius=radius,
                height=height,
                rotation=math.pi / 2.0,
            ),
            _prism_structure(
                name="right",
                n_sides=3,
                center=(center_gap / 2.0, 0.0, calculation_height),
                radius=radius,
                height=height,
                rotation=-math.pi / 2.0,
            ),
        ]
    )
    return ValidationCase(
        name="double_wedge",
        description="Two opposing triangular prisms used as a canonical double-diffraction scene.",
        scene=scene,
        tx_pos=(0.0, -6.0, calculation_height),
        range_x=(-7.5, 7.5),
        range_y=(-7.0, 7.0),
        calculation_height=calculation_height,
        cut_axis="y",
        cut_value=2.0,
    )


def build_single_wedge_case(
    radius: float = 0.95,
    height: float = 3.0,
    calculation_height: float = 1.5,
) -> ValidationCase:
    """Create a single-prism scene for first-order overlap comparisons."""

    scene = Scene(
        structures=[
            _prism_structure(
                name="wedge",
                n_sides=3,
                center=(0.0, 0.0, calculation_height),
                radius=radius,
                height=height,
                rotation=math.pi / 2.0,
            )
        ]
    )
    return ValidationCase(
        name="single_wedge",
        description="Single triangular prism used for first-order finite-wedge explicit-reference overlap checks.",
        scene=scene,
        tx_pos=(0.0, -6.0, calculation_height),
        range_x=(-7.0, 7.0),
        range_y=(-7.0, 7.0),
        calculation_height=calculation_height,
        cut_axis="y",
        cut_value=2.0,
    )


def build_triple_wedge_case(
    radius_to_center: float = 2.5,
    prism_radius: float = 0.9,
    height: float = 3.0,
    calculation_height: float = 1.5,
) -> ValidationCase:
    """Create a three-prism scene for third-order diffraction benchmarking."""

    placements = [
        (0.0, radius_to_center, math.pi),
        (-0.8660254 * radius_to_center, -0.5 * radius_to_center, math.pi / 3.0),
        (0.8660254 * radius_to_center, -0.5 * radius_to_center, -math.pi / 3.0),
    ]

    scene = Scene(
        structures=[
            _prism_structure(
                name=f"prism_{index}",
                n_sides=3,
                center=(center_x, center_y, calculation_height),
                radius=prism_radius,
                height=height,
                rotation=rotation,
            )
            for index, (center_x, center_y, rotation) in enumerate(placements)
        ]
    )
    return ValidationCase(
        name="triple_wedge",
        description="Three triangular prisms arranged to expose third-order diffraction paths.",
        scene=scene,
        tx_pos=(0.0, -6.5, calculation_height),
        range_x=(-8.0, 8.0),
        range_y=(-8.0, 8.0),
        calculation_height=calculation_height,
        cut_axis="y",
        cut_value=2.5,
    )


def build_mixed_prefix_suffix_case(
    calculation_height: float = 1.5,
) -> ValidationCase:
    """Create a scene that exposes R->D, D->R, and R->D->R families."""

    scene = Scene(
        structures=[
            _cube_structure(name="left", center=(-2.5, -3.0, calculation_height), size=2.0),
            _cube_structure(name="right", center=(2.5, 1.0, calculation_height), size=2.0),
        ]
    )
    return ValidationCase(
        name="mixed_prefix_suffix",
        description="Two cubes positioned so first-order diffraction mixes with sampled reflection prefixes and one-bounce suffix reflections.",
        scene=scene,
        tx_pos=(0.0, -5.0, calculation_height),
        range_x=(-6.0, 6.0),
        range_y=(-6.0, 6.0),
        calculation_height=calculation_height,
        cut_axis="y",
        cut_value=1.0,
    )


def build_inserted_reflection_case(
    calculation_height: float = 1.5,
) -> ValidationCase:
    """Create a scene that exposes D->R->D inserted-reflection diffraction."""

    scene = Scene(
        structures=[
            _cube_structure(name="left", center=(-2.5, 0.0, calculation_height), size=2.0),
            _prism_structure(
                name="right",
                n_sides=3,
                center=(2.5, 0.0, calculation_height),
                radius=1.0,
                height=3.0,
                rotation=0.0,
            ),
        ]
    )
    return ValidationCase(
        name="inserted_reflection_diffraction",
        description="Cube-prism scene chosen to make inserted-reflection D->R->D diffraction states auditable at second order.",
        scene=scene,
        tx_pos=(0.0, -4.0, calculation_height),
        range_x=(-6.0, 6.0),
        range_y=(-6.0, 6.0),
        calculation_height=calculation_height,
        cut_axis="y",
        cut_value=0.0,
    )


def _extract_line_cut(X, Y, values, grid_size: int, axis: str, value: float):
    """Extract the nearest horizontal or vertical line cut."""

    X2 = np.asarray(X).reshape(grid_size, grid_size)
    Y2 = np.asarray(Y).reshape(grid_size, grid_size)
    V2 = np.asarray(values).reshape(grid_size, grid_size)

    if axis == "y":
        idx = int(np.argmin(np.abs(Y2[:, 0] - value)))
        return {
            "axis": "y",
            "requested_value": float(value),
            "actual_value": float(Y2[idx, 0]),
            "coord": X2[idx, :].copy(),
            "values": V2[idx, :].copy(),
        }
    if axis == "x":
        idx = int(np.argmin(np.abs(X2[0, :] - value)))
        return {
            "axis": "x",
            "requested_value": float(value),
            "actual_value": float(X2[0, idx]),
            "coord": Y2[:, idx].copy(),
            "values": V2[:, idx].copy(),
        }
    raise ValueError(f"Unsupported cut axis: {axis}")


def _to_db(field):
    return np.asarray(to_power_db(field))


def _audit_value_to_numpy(value):
    try:
        return np.stack([np.asarray(value.x), np.asarray(value.y), np.asarray(value.z)], axis=-1)
    except Exception:
        pass
    try:
        return np.asarray(value.real) + 1j * np.asarray(value.imag)
    except Exception:
        pass
    return np.asarray(value)


def _state_audit_to_numpy(state_audit: dict | None) -> dict | None:
    if state_audit is None:
        return None

    numpy_audit = {
        "n_states": int(state_audit["n_states"]),
        "history_size": int(state_audit["history_size"]),
    }
    for key, value in state_audit.items():
        if key in {"n_states", "history_size"}:
            continue
        numpy_audit[key] = _audit_value_to_numpy(value)
    return numpy_audit


def _complex_field_to_numpy(field, grid_size: int | None = None):
    array = np.asarray(field.real) + 1j * np.asarray(field.imag)
    if grid_size is None:
        return array
    return array.reshape(grid_size, grid_size)


def _complex_field_metrics(candidate_field, reference_field) -> dict:
    error = candidate_field - reference_field
    error_abs = np.abs(error)
    return {
        "candidate_complex_field": candidate_field,
        "reference_complex_field": reference_field,
        "complex_error_field": error,
        "max_abs_complex_error": float(np.max(error_abs)) if error_abs.size > 0 else 0.0,
        "rms_complex_error": float(np.sqrt(np.mean(error_abs * error_abs))) if error_abs.size > 0 else 0.0,
    }


def _build_explicit_double_diffraction_states(
    case: ValidationCase,
    frequency: float,
    material_detail: dict | None = None,
):
    return _build_explicit_multi_diffraction_states(
        case=case,
        frequency=frequency,
        order=2,
        material_detail=material_detail,
    )


def _expand_explicit_edge_state_order(
    prev_states,
    scene: Scene,
    edge_data,
    wavelength: float,
    k: float,
    material_detail: dict | None = None,
):
    history_size = max(1, _state_history_size(prev_states))
    if prev_states["n_states"] == 0 or edge_data is None or edge_data["n_edges"] == 0:
        return _empty_state_arrays(history_size=history_size)

    n_edges = int(edge_data["n_edges"])
    n_pairs = int(prev_states["n_states"]) * n_edges
    pair_idx = dr.arange(wt.UInt32, n_pairs)
    prev_state_idx = pair_idx // n_edges
    next_edge_idx = pair_idx % n_edges
    prev_edge_idx = dr.gather(wt.UInt32, prev_states["edge_idx"], prev_state_idx)
    distinct_edge = next_edge_idx != prev_edge_idx

    prev_edge_pos = dr.gather(wt.Point3f, prev_states["edge_pos"], prev_state_idx)
    next_edge_pos = dr.gather(wt.Point3f, edge_data["pos"], next_edge_idx)
    prev_adjacent_face0 = dr.gather(wt.Int32, prev_states["adjacent_face0"], prev_state_idx)
    prev_adjacent_face1 = dr.gather(wt.Int32, prev_states["adjacent_face1"], prev_state_idx)
    next_adjacent_face0 = dr.gather(wt.Int32, edge_data["adjacent_face0"], next_edge_idx)
    next_adjacent_face1 = dr.gather(wt.Int32, edge_data["adjacent_face1"], next_edge_idx)
    visible = _segment_visibility_mask(
        prev_edge_pos,
        next_edge_pos,
        scene,
        ignore_prim_idx=(
            prev_adjacent_face0,
            prev_adjacent_face1,
            next_adjacent_face0,
            next_adjacent_face1,
        ),
    )
    keep_idx = dr.compress(distinct_edge & visible)
    if dr.width(keep_idx) == 0:
        return _empty_state_arrays(history_size=history_size)

    keep_prev_state_idx = dr.gather(wt.UInt32, prev_state_idx, keep_idx)
    keep_edge_idx = dr.gather(wt.UInt32, next_edge_idx, keep_idx)
    keep_prev_states = gather_state_arrays(prev_states, keep_prev_state_idx)
    keep_edge_pos = dr.gather(wt.Point3f, edge_data["pos"], keep_edge_idx)
    keep_edge_dir = dr.gather(wt.Vector3f, edge_data["edge_dir"], keep_edge_idx)
    keep_n0 = dr.gather(wt.Vector3f, edge_data["n0"], keep_edge_idx)
    keep_nn = dr.gather(wt.Vector3f, edge_data["n_face_n"], keep_edge_idx)
    keep_wedge_n = dr.gather(wt.Float, edge_data["wedge_n"], keep_edge_idx)
    keep_adjacent_face0 = dr.gather(wt.Int32, edge_data["adjacent_face0"], keep_edge_idx)
    keep_adjacent_face1 = dr.gather(wt.Int32, edge_data["adjacent_face1"], keep_edge_idx)

    incident_field, incident_normal_derivative, incident_vector, incident_normal_derivative_vector = _edge_state_field_to_targets(
        keep_prev_states,
        keep_edge_pos,
        k,
        return_normal_derivative=True,
        return_vector=True,
        wavelength=wavelength,
        material_detail=material_detail,
    )
    field_power = (
        incident_vector["x"].real * incident_vector["x"].real
        + incident_vector["x"].imag * incident_vector["x"].imag
        + incident_vector["y"].real * incident_vector["y"].real
        + incident_vector["y"].imag * incident_vector["y"].imag
        + incident_vector["z"].real * incident_vector["z"].real
        + incident_vector["z"].imag * incident_vector["z"].imag
        + incident_normal_derivative_vector["x"].real * incident_normal_derivative_vector["x"].real
        + incident_normal_derivative_vector["x"].imag * incident_normal_derivative_vector["x"].imag
        + incident_normal_derivative_vector["y"].real * incident_normal_derivative_vector["y"].real
        + incident_normal_derivative_vector["y"].imag * incident_normal_derivative_vector["y"].imag
        + incident_normal_derivative_vector["z"].real * incident_normal_derivative_vector["z"].real
        + incident_normal_derivative_vector["z"].imag * incident_normal_derivative_vector["z"].imag
    )
    valid_idx = dr.compress(field_power > wt.Float(1e-20))
    if dr.width(valid_idx) == 0:
        return _empty_state_arrays(history_size=history_size)

    keep_edge_idx = dr.gather(wt.UInt32, keep_edge_idx, valid_idx)
    keep_edge_pos = dr.gather(wt.Point3f, keep_edge_pos, valid_idx)
    keep_edge_dir = dr.gather(wt.Vector3f, keep_edge_dir, valid_idx)
    keep_n0 = dr.gather(wt.Vector3f, keep_n0, valid_idx)
    keep_nn = dr.gather(wt.Vector3f, keep_nn, valid_idx)
    keep_wedge_n = dr.gather(wt.Float, keep_wedge_n, valid_idx)
    edge_line_min_all, edge_line_max_all = require_edge_data_line_bounds(
        edge_data,
        context="_build_next_diffraction_states",
    )
    keep_edge_line_min = dr.gather(wt.Float, edge_line_min_all, keep_edge_idx)
    keep_edge_line_max = dr.gather(wt.Float, edge_line_max_all, keep_edge_idx)
    keep_adjacent_face0 = dr.gather(wt.Int32, keep_adjacent_face0, valid_idx)
    keep_adjacent_face1 = dr.gather(wt.Int32, keep_adjacent_face1, valid_idx)
    keep_source_pos = dr.gather(wt.Point3f, keep_prev_states["edge_pos"], valid_idx)
    keep_incident_field = eval_complex(dr.gather(wt.Complex2f, incident_field, valid_idx))
    keep_incident_normal_derivative = eval_complex(
        dr.gather(wt.Complex2f, incident_normal_derivative, valid_idx)
    )
    keep_incident_vector = {
        axis: eval_complex(dr.gather(wt.Complex2f, incident_vector[axis], valid_idx))
        for axis in ("x", "y", "z")
    }
    keep_incident_normal_derivative_vector = {
        axis: eval_complex(dr.gather(wt.Complex2f, incident_normal_derivative_vector[axis], valid_idx))
        for axis in ("x", "y", "z")
    }
    keep_prev_order = dr.gather(wt.UInt32, keep_prev_states["order"], valid_idx)
    keep_source_type_code = dr.gather(wt.UInt32, keep_prev_states["source_type_code"], valid_idx)
    keep_prefix_reflection_depth = dr.gather(
        wt.UInt32, keep_prev_states["prefix_reflection_depth"], valid_idx
    )
    keep_intermediate_reflection_depth = dr.gather(
        wt.UInt32, keep_prev_states["intermediate_reflection_depth"], valid_idx
    )
    keep_suffix_reflection_depth = dr.gather(
        wt.UInt32, keep_prev_states["suffix_reflection_depth"], valid_idx
    )
    keep_order = keep_prev_order + wt.UInt32(1)
    n_valid = dr.width(valid_idx)

    return _make_state_arrays(
        edge_idx=keep_edge_idx,
        edge_pos=keep_edge_pos,
        edge_dir=keep_edge_dir,
        n0=keep_n0,
        nn=keep_nn,
        wedge_n=keep_wedge_n,
        adjacent_face0=keep_adjacent_face0,
        adjacent_face1=keep_adjacent_face1,
        source_pos=keep_source_pos,
        edge_line_min=keep_edge_line_min,
        edge_line_max=keep_edge_line_max,
        incident_field=keep_incident_field,
        incident_normal_derivative=keep_incident_normal_derivative,
        incident_vector=keep_incident_vector,
        incident_normal_derivative_vector=keep_incident_normal_derivative_vector,
        is_direct_tx=dr.full(wt.Bool, False, n_valid),
        source_type_code=keep_source_type_code,
        prefix_reflection_depth=keep_prefix_reflection_depth,
        intermediate_reflection_depth=keep_intermediate_reflection_depth,
        suffix_reflection_depth=keep_suffix_reflection_depth,
        approximation_mode_code=dr.full(wt.UInt32, 1, n_valid),
        order=keep_order,
        lineage_parent_state_id=dr.gather(
            wt.Int32,
            _state_ids(keep_prev_states),
            valid_idx,
        ),
        lineage_last_edge_idx=wt.Int32(keep_edge_idx),
        lineage_last_reflection_depth_delta=dr.zeros(wt.UInt32, n_valid),
        lineage_store=_state_lineage_store(keep_prev_states),
    )


def _build_explicit_multi_diffraction_states(
    case: ValidationCase,
    frequency: float,
    order: int,
    material_detail: dict | None = None,
):
    wavelength = 299792458.0 / frequency
    k = 2.0 * math.pi / wavelength
    history_size = max(1, int(order))
    edge_data = case.scene.get_edge_data(case.calculation_height)["edge_data"]
    if edge_data is None:
        return wavelength, k, None, _empty_state_arrays(history_size=history_size)

    tx = wt.Point3f(*case.tx_pos)
    current_states = _build_tx_first_order_state_arrays(
        tx,
        edge_data,
        wavelength,
        k,
        history_size=history_size,
        scene=case.scene,
    )
    current_states, lineage_store, next_state_id = _finalize_state_lineage(
        current_states,
        lineage_store=None,
        next_state_id=0,
    )
    if current_states["n_states"] == 0:
        return wavelength, k, edge_data, _empty_state_arrays(history_size=history_size)

    if order == 1:
        return wavelength, k, edge_data, current_states

    for _ in range(2, int(order) + 1):
        current_states = _expand_explicit_edge_state_order(
            current_states,
            case.scene,
            edge_data,
            wavelength,
            k,
            material_detail=material_detail,
        )
        current_states, lineage_store, next_state_id = _finalize_state_lineage(
            current_states,
            lineage_store=lineage_store,
            next_state_id=next_state_id,
        )
        if current_states["n_states"] == 0:
            break

    return wavelength, k, edge_data, current_states


def evaluate_closed_form_double_diffraction_reference(
    case: ValidationCase,
    frequency: float,
    grid_size: int = 48,
) -> dict:
    """Evaluate order-2 diffraction against an explicit pair-expansion reference."""

    wavelength, k, edge_data, reference_states = _build_explicit_double_diffraction_states(case, frequency)
    field = Field(bounds=(case.range_x, case.range_y), size=(grid_size, grid_size))
    coords = field.get_coordinates()
    rx_pos = wt.Point3f(coords["X"], coords["Y"], wt.Float(case.calculation_height))

    breakdown = compute_diffraction_order_breakdown(
        coords["X"],
        coords["Y"],
        case.calculation_height,
        wt.Point3f(*case.tx_pos),
        case.scene,
        wavelength,
        k,
        reflection_detail=None,
        max_diffractions=2,
        reflection_n_rays=0,
        reflection_max_bounces=0,
        reflection_coef=1.0,
        reflection_mode="2d",
        grid=field,
        grid_data=coords,
        split_by_edge=False,
    )
    candidate_field = breakdown["order_fields"][1]
    reference_field = _accumulate_state_subset_field(
        state_arrays=reference_states,
        rx_pos=rx_pos,
        scene=case.scene,
        wavelength=wavelength,
        k=k,
        n_edges=0 if edge_data is None else edge_data["n_edges"],
        material_detail=None,
        suffix=ReflectionSuffixConfig(),
    )

    candidate_np = _complex_field_to_numpy(candidate_field, grid_size)
    reference_np = _complex_field_to_numpy(reference_field, grid_size)
    metrics = _complex_field_metrics(candidate_np, reference_np)
    metrics.update(
        {
            "case": case,
            "grid_size": int(grid_size),
            "frequency": float(frequency),
            "reference_kind": "explicit_pair_expansion_finite_wedge_utd",
            "reference_state_audit": _state_audit_to_numpy(
                _build_state_audit(reference_states, edge_data)
            ),
            "candidate_line_cut": _extract_line_cut(
                coords["X"], coords["Y"], candidate_np, grid_size, case.cut_axis, case.cut_value
            ),
            "reference_line_cut": _extract_line_cut(
                coords["X"], coords["Y"], reference_np, grid_size, case.cut_axis, case.cut_value
            ),
        }
    )
    return metrics


def evaluate_closed_form_triple_diffraction_reference(
    case: ValidationCase,
    frequency: float,
    grid_size: int = 48,
) -> dict:
    """Evaluate order-3 diffraction against an explicit triplet-expansion reference."""

    wavelength, k, edge_data, reference_states = _build_explicit_multi_diffraction_states(
        case=case,
        frequency=frequency,
        order=3,
    )
    field = Field(bounds=(case.range_x, case.range_y), size=(grid_size, grid_size))
    coords = field.get_coordinates()
    rx_pos = wt.Point3f(coords["X"], coords["Y"], wt.Float(case.calculation_height))

    breakdown = compute_diffraction_order_breakdown(
        coords["X"],
        coords["Y"],
        case.calculation_height,
        wt.Point3f(*case.tx_pos),
        case.scene,
        wavelength,
        k,
        reflection_detail=None,
        max_diffractions=3,
        reflection_n_rays=0,
        reflection_max_bounces=0,
        reflection_coef=1.0,
        reflection_mode="2d",
        grid=field,
        grid_data=coords,
        split_by_edge=False,
    )
    candidate_field = breakdown["order_fields"][2]
    reference_field = _accumulate_state_subset_field(
        state_arrays=reference_states,
        rx_pos=rx_pos,
        scene=case.scene,
        wavelength=wavelength,
        k=k,
        n_edges=0 if edge_data is None else edge_data["n_edges"],
        material_detail=None,
        suffix=ReflectionSuffixConfig(),
    )

    candidate_np = _complex_field_to_numpy(candidate_field, grid_size)
    reference_np = _complex_field_to_numpy(reference_field, grid_size)
    metrics = _complex_field_metrics(candidate_np, reference_np)
    metrics.update(
        {
            "case": case,
            "grid_size": int(grid_size),
            "frequency": float(frequency),
            "reference_kind": "explicit_triplet_expansion_finite_wedge_utd",
            "reference_state_audit": _state_audit_to_numpy(
                _build_state_audit(reference_states, edge_data)
            ),
            "candidate_line_cut": _extract_line_cut(
                coords["X"], coords["Y"], candidate_np, grid_size, case.cut_axis, case.cut_value
            ),
            "reference_line_cut": _extract_line_cut(
                coords["X"], coords["Y"], reference_np, grid_size, case.cut_axis, case.cut_value
            ),
        }
    )
    return metrics


def compare_first_order_overlap_against_sionna(
    case: ValidationCase,
    frequency: float,
    grid_size: int = 48,
) -> dict:
    """Compare first-order direct diffraction against an explicit finite-wedge reference."""

    wavelength = 299792458.0 / frequency
    k = 2.0 * math.pi / wavelength
    field = Field(bounds=(case.range_x, case.range_y), size=(grid_size, grid_size))
    coords = field.get_coordinates()
    tx = wt.Point3f(*case.tx_pos)

    _, _, _, dif_components = compute_diffraction_field(
        coords["X"],
        coords["Y"],
        case.calculation_height,
        tx,
        case.scene,
        wavelength,
        k,
        reflection_detail=None,
        max_diffractions=1,
        reflection_n_rays=0,
        reflection_max_bounces=0,
        reflection_coef=1.0,
        reflection_mode="2d",
        grid=field,
        grid_data=coords,
        return_components=True,
        return_per_edge=False,
        return_state_audit=True,
        diffraction_material=None,
    )

    edge_data = case.scene.get_edge_data(case.calculation_height)["edge_data"]
    reference_states = _build_tx_first_order_state_arrays(
        tx,
        edge_data,
        wavelength,
        k,
        history_size=1,
        scene=case.scene,
        reflection_coef=1.0,
    )
    reference_field = _accumulate_state_subset_field(
        state_arrays=reference_states,
        rx_pos=wt.Point3f(coords["X"], coords["Y"], wt.Float(case.calculation_height)),
        scene=case.scene,
        wavelength=wavelength,
        k=k,
        n_edges=edge_data["n_edges"],
        material_detail=None,
        suffix=ReflectionSuffixConfig(),
    )

    candidate_np = _complex_field_to_numpy(dif_components["a_direct"], grid_size)
    reference_np = _complex_field_to_numpy(reference_field, grid_size)
    metrics = _complex_field_metrics(candidate_np, reference_np)
    metrics.update(
        {
            "case": case,
            "grid_size": int(grid_size),
            "frequency": float(frequency),
            "reference_kind": "explicit_first_order_finite_wedge_utd",
            "candidate_state_audit": _state_audit_to_numpy(dif_components["state_audit"]),
            "candidate_line_cut": _extract_line_cut(
                coords["X"], coords["Y"], candidate_np, grid_size, case.cut_axis, case.cut_value
            ),
            "reference_line_cut": _extract_line_cut(
                coords["X"], coords["Y"], reference_np, grid_size, case.cut_axis, case.cut_value
            ),
        }
    )
    return metrics


def run_diffraction_order_sweep(
    case: ValidationCase,
    frequency: float,
    grid_size: int = 96,
    max_order: int = 3,
) -> dict:
    """
    Run solver-side order sweep for a canonical validation case.

    This is intentionally solver-only. Closed-form reference coefficients can be
    attached later without changing the surrounding data export format.
    """

    orders = list(range(1, int(max_order) + 1))
    order_results = []

    for order in orders:
        tracer = Tracer(
            frequency=frequency,
            scene=case.scene,
            reflection_n_rays=0,
            reflection_max_bounces=0,
            enable_rd_diffraction=False,
            max_diffractions=order,
        )
        monitor = FieldMonitor(
            "validation_plane",
            axis="z",
            position=case.calculation_height,
            bounds=(case.range_x, case.range_y),
            grid_size=grid_size,
        )
        result = tracer.trace(
            tx_pos=wt.Point3f(*case.tx_pos),
            monitor=monitor,
            verbose=False,
            return_diffraction_audit=True,
        )
        payload = result

        dif_db = _to_db(payload.field.diffraction).reshape(grid_size, grid_size)
        tot_db = _to_db(payload.field.total).reshape(grid_size, grid_size)
        line_cut_dif = _extract_line_cut(
            payload.coords.grid_x,
            payload.coords.grid_y,
            dif_db,
            grid_size,
            case.cut_axis,
            case.cut_value,
        )
        line_cut_tot = _extract_line_cut(
            payload.coords.grid_x,
            payload.coords.grid_y,
            tot_db,
            grid_size,
            case.cut_axis,
            case.cut_value,
        )
        state_audit = _state_audit_to_numpy(payload.diffraction_detail["state_audit"])

        order_results.append({
            "order": order,
            "grid_size": grid_size,
            "X": np.asarray(payload.coords.grid_x).reshape(grid_size, grid_size),
            "Y": np.asarray(payload.coords.grid_y).reshape(grid_size, grid_size),
            "dif_db": dif_db,
            "tot_db": tot_db,
            "line_cut_dif": line_cut_dif,
            "line_cut_tot": line_cut_tot,
            "state_audit": state_audit,
        })

    return {
        "case": case,
        "frequency": float(frequency),
        "orders": order_results,
    }


def sweep_to_npz_payload(sweep_result: dict) -> dict:
    """Flatten a sweep result into a format accepted by numpy.savez."""

    case = sweep_result["case"]
    payload = {
        "case_name": np.array(case.name),
        "case_description": np.array(case.description),
        "tx_pos": np.array(case.tx_pos, dtype=np.float64),
        "range_x": np.array(case.range_x, dtype=np.float64),
        "range_y": np.array(case.range_y, dtype=np.float64),
        "calculation_height": np.array(case.calculation_height, dtype=np.float64),
        "cut_axis": np.array(case.cut_axis),
        "cut_value": np.array(case.cut_value, dtype=np.float64),
        "frequency": np.array(sweep_result["frequency"], dtype=np.float64),
        "n_orders": np.array(len(sweep_result["orders"]), dtype=np.int32),
    }

    for item in sweep_result["orders"]:
        order = item["order"]
        payload[f"order_{order}_X"] = item["X"]
        payload[f"order_{order}_Y"] = item["Y"]
        payload[f"order_{order}_dif_db"] = item["dif_db"]
        payload[f"order_{order}_tot_db"] = item["tot_db"]
        payload[f"order_{order}_cut_coord"] = item["line_cut_dif"]["coord"]
        payload[f"order_{order}_cut_dif_db"] = item["line_cut_dif"]["values"]
        payload[f"order_{order}_cut_tot_db"] = item["line_cut_tot"]["values"]
        payload[f"order_{order}_cut_actual_value"] = np.array(
            item["line_cut_dif"]["actual_value"], dtype=np.float64
        )
        state_audit = item.get("state_audit")
        if state_audit is not None:
            payload[f"order_{order}_audit_n_states"] = np.array(state_audit["n_states"], dtype=np.int32)
            payload[f"order_{order}_audit_history_size"] = np.array(state_audit["history_size"], dtype=np.int32)
            for key, value in state_audit.items():
                if key in {"n_states", "history_size"}:
                    continue
                payload[f"order_{order}_audit_{key}"] = value
    return payload

