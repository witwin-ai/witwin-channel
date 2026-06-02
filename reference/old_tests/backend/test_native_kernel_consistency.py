from __future__ import annotations

import math
from types import SimpleNamespace

import drjit as dr
import numpy as np
import pytest
import torch
from tests._scene_helpers import box_geometry, build_scene as build_test_scene
from witwin.channel import Field, FieldMonitor, Tracer, native_extension_available
import witwin as wt
from witwin.channel.config import DiffractionExecutionConfig
from witwin.channel.kernels.trace.cartesian_filter import drjit_impl as cartesian_filter_drjit
from witwin.channel.kernels.trace.cartesian_filter import native_impl as cartesian_filter_native
from witwin.channel.kernels.scene_build.coplanarity import drjit_impl as coplanarity_drjit
from witwin.channel.kernels.scene_build.coplanarity import native_impl as coplanarity_native
from witwin.channel.kernels.scene_build.edge_geometry import drjit_impl as edge_geometry_drjit
from witwin.channel.kernels.scene_build.edge_geometry import native_impl as edge_geometry_native
from witwin.channel.kernels.trace.packed_state import drjit_impl as packed_state_drjit
from witwin.channel.kernels.trace.packed_state import native_impl as packed_state_native
from witwin.channel.kernels.trace.pruning_sort import drjit_impl as pruning_sort_drjit
from witwin.channel.kernels.trace.pruning_sort import native_impl as pruning_sort_native
from witwin.channel.kernels.trace.reflection import drjit_impl as reflection_drjit
from witwin.channel.kernels.trace.reflection import native_impl as reflection_native
from witwin.channel.kernels.monitors.field.reflection_grid import drjit_impl as reflection_grid_drjit
from witwin.channel.kernels.monitors.field.reflection_grid import native_impl as reflection_grid_native
from witwin.channel.kernels.monitors.field.radio_map_accumulate import (
    native_impl as radio_map_accumulate_native,
)
from witwin.channel.kernels.monitors.common.suffix_grid import drjit_impl as suffix_grid_drjit
from witwin.channel.kernels.monitors.common.suffix_grid import native_impl as suffix_grid_native
from witwin.channel.kernels.trace.utd import drjit_impl as utd_drjit
from witwin.channel.kernels.trace.utd import native_impl as utd_native
from witwin.channel.trace.diffraction import _build_tx_first_order_state_arrays
from witwin.channel.trace.diffraction.constants import STATE_STATIC_KEYS, _state_history_size
from witwin.channel.trace.diffraction.state import reduce_state_arrays_for_path_export
import witwin.channel.trace.diffraction.suffix as suffix_module
from witwin.channel.trace.diffraction.state.arrays import (
    _finalize_state_lineage,
    _make_state_arrays,
    _materialize_state_history,
    _state_ids,
)
import witwin.channel._native as internal_native_module
import witwin.channel.trace.reflection.api as reflection_api_module
import witwin.channel.trace.reflection.paths as reflection_paths_module
import witwin.channel.trace.reflection.epc as reflection_epc_module
from witwin.channel.trace.reflection.api import compute_reflection_field
from witwin.channel.trace.materials import coerce_reflection_trace_detail
from witwin.channel.utils import drjit_to_torch_view
FLAGS = dr.ADFlag.Default | dr.ADFlag.AllowNoGrad


def _max_abs_diff(lhs, rhs) -> float:
    return float(dr.max(dr.abs(lhs - rhs))[0])


def _torch_max_abs_diff(lhs, rhs) -> float:
    lhs_t = drjit_to_torch_view(lhs, detach=True)
    rhs_t = drjit_to_torch_view(rhs, detach=True)
    if lhs_t.numel() == 0:
        return 0.0
    if lhs_t.dtype == torch.bool:
        return float((lhs_t != rhs_t).any().item())
    if lhs_t.dtype in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8, torch.uint32):
        diff = (lhs_t.to(torch.int64) - rhs_t.to(torch.int64)).abs()
        return float(diff.max().item())
    diff = (lhs_t.to(torch.float32) - rhs_t.to(torch.float32)).abs()
    return float(diff.max().item())


def _try_component(value, name: str):
    try:
        return getattr(value, name)
    except Exception:
        return None


def _state_value_max_abs_diff(lhs, rhs) -> float:
    x = _try_component(lhs, "x")
    y = _try_component(lhs, "y")
    if x is not None and y is not None:
        diffs = [
            _state_value_max_abs_diff(lhs.x, rhs.x),
            _state_value_max_abs_diff(lhs.y, rhs.y),
        ]
        z = _try_component(lhs, "z")
        if z is not None:
            diffs.append(_state_value_max_abs_diff(lhs.z, rhs.z))
        return max(diffs)
    real = _try_component(lhs, "real")
    imag = _try_component(lhs, "imag")
    if real is not None and imag is not None and real is not lhs and imag is not lhs:
        return max(
            _state_value_max_abs_diff(lhs.real, rhs.real),
            _state_value_max_abs_diff(lhs.imag, rhs.imag),
        )
    return _torch_max_abs_diff(lhs, rhs)


def _detach_float(value):
    return dr.detach(value)


def _detach_point3f(value):
    return wt.Point3f(_detach_float(value.x), _detach_float(value.y), _detach_float(value.z))


def _detach_vector3f(value):
    return wt.Vector3f(_detach_float(value.x), _detach_float(value.y), _detach_float(value.z))


def _detach_complex(value):
    return wt.Complex2f(_detach_float(value.real), _detach_float(value.imag))


def _detach_vector_complex(value):
    return {
        "x": _detach_complex(value["x"]),
        "y": _detach_complex(value["y"]),
        "z": _detach_complex(value["z"]),
    }


def _enable_grad_point3f(value):
    dr.enable_grad(value.x, value.y, value.z)


def _enable_grad_vector3f(value):
    dr.enable_grad(value.x, value.y, value.z)


def _enable_grad_complex(value):
    dr.enable_grad(value.real, value.imag)


def _enable_grad_vector_complex(value):
    _enable_grad_complex(value["x"])
    _enable_grad_complex(value["y"])
    _enable_grad_complex(value["z"])


def _set_grad_point3f(value, grad_value):
    dr.set_grad(value.x, grad_value.x)
    dr.set_grad(value.y, grad_value.y)
    dr.set_grad(value.z, grad_value.z)


def _set_grad_vector3f(value, grad_value):
    dr.set_grad(value.x, grad_value.x)
    dr.set_grad(value.y, grad_value.y)
    dr.set_grad(value.z, grad_value.z)


def _set_grad_complex(value, grad_value):
    dr.set_grad(value.real, grad_value.real)
    dr.set_grad(value.imag, grad_value.imag)


def _clone_packed_state_ad_inputs(state_arrays: dict) -> dict:
    cloned = dict(state_arrays)
    cloned["edge_pos"] = _detach_point3f(state_arrays["edge_pos"])
    cloned["source_pos"] = _detach_point3f(state_arrays["source_pos"])
    cloned["wedge_n"] = _detach_float(state_arrays["wedge_n"])
    cloned["incident_field"] = _detach_complex(state_arrays["incident_field"])
    cloned["incident_vector_x"] = _detach_complex(state_arrays["incident_vector_x"])
    return cloned


def _enable_grad_packed_state_inputs(state_arrays: dict) -> None:
    _enable_grad_point3f(state_arrays["edge_pos"])
    _enable_grad_point3f(state_arrays["source_pos"])
    dr.enable_grad(state_arrays["wedge_n"])
    _enable_grad_complex(state_arrays["incident_field"])
    _enable_grad_complex(state_arrays["incident_vector_x"])


def _set_grad_packed_state_inputs(state_arrays: dict, tangent: dict) -> None:
    _set_grad_point3f(state_arrays["edge_pos"], tangent["edge_pos"])
    _set_grad_point3f(state_arrays["source_pos"], tangent["source_pos"])
    dr.set_grad(state_arrays["wedge_n"], tangent["wedge_n"])
    _set_grad_complex(state_arrays["incident_field"], tangent["incident_field"])
    _set_grad_complex(state_arrays["incident_vector_x"], tangent["incident_vector_x"])


def _packed_state_tangent(n_states: int) -> dict:
    return {
        "edge_pos": wt.Point3f(
            dr.linspace(wt.Float, 0.01, 0.03, n_states),
            dr.linspace(wt.Float, -0.02, 0.01, n_states),
            dr.linspace(wt.Float, 0.02, -0.01, n_states),
        ),
        "source_pos": wt.Point3f(
            dr.linspace(wt.Float, -0.03, 0.02, n_states),
            dr.linspace(wt.Float, 0.01, -0.02, n_states),
            dr.linspace(wt.Float, 0.00, 0.02, n_states),
        ),
        "wedge_n": dr.linspace(wt.Float, -0.04, 0.05, n_states),
        "incident_field": wt.Complex2f(
            dr.linspace(wt.Float, 0.03, -0.02, n_states),
            dr.linspace(wt.Float, -0.01, 0.04, n_states),
        ),
        "incident_vector_x": wt.Complex2f(
            dr.linspace(wt.Float, 0.02, 0.05, n_states),
            dr.linspace(wt.Float, -0.03, 0.01, n_states),
        ),
    }


def _packed_state_loss(state_arrays: dict):
    n_states = int(state_arrays["n_states"])
    weights0 = dr.linspace(wt.Float, 0.1, 0.3, n_states)
    weights1 = dr.linspace(wt.Float, -0.2, 0.05, n_states)
    weights2 = dr.linspace(wt.Float, 0.07, 0.19, n_states)
    loss = dr.zeros(wt.Float, 1)
    loss += dr.sum(state_arrays["edge_pos"].x * weights0)
    loss += dr.sum(state_arrays["source_pos"].y * weights1)
    loss += dr.sum(state_arrays["wedge_n"] * weights2)
    loss += dr.sum(state_arrays["incident_field"].real * weights0)
    loss += dr.sum(state_arrays["incident_field"].imag * weights1)
    loss += dr.sum(state_arrays["incident_vector_x"].real * weights2)
    loss += dr.sum(state_arrays["incident_vector_x"].imag * weights0)
    return loss


def _packed_state_input_grads(state_arrays: dict) -> dict:
    return {
        "edge_pos": _detach_point3f(dr.grad(state_arrays["edge_pos"])),
        "source_pos": _detach_point3f(dr.grad(state_arrays["source_pos"])),
        "wedge_n": _detach_float(dr.grad(state_arrays["wedge_n"])),
        "incident_field": _detach_complex(dr.grad(state_arrays["incident_field"])),
        "incident_vector_x": _detach_complex(dr.grad(state_arrays["incident_vector_x"])),
    }


def _vector_field_max_abs_diff(ref, got) -> float:
    max_diff = 0.0
    for axis in ("x", "y", "z"):
        max_diff = max(max_diff, _max_abs_diff(ref[axis].real, got[axis].real))
        max_diff = max(max_diff, _max_abs_diff(ref[axis].imag, got[axis].imag))
    return max_diff


def _reflection_output_components(outputs):
    components = []
    for polarization_field in outputs:
        for axis in ("x", "y", "z"):
            components.append(polarization_field[axis].real)
            components.append(polarization_field[axis].imag)
    return tuple(components)


def _reflection_replay_output_components(vector, endpoints):
    return (
        vector["x"].real,
        vector["x"].imag,
        vector["y"].real,
        vector["y"].imag,
        vector["z"].real,
        vector["z"].imag,
        endpoints["tx_pos"].x,
        endpoints["tx_pos"].y,
        endpoints["tx_pos"].z,
        endpoints["first_hit"].x,
        endpoints["first_hit"].y,
        endpoints["first_hit"].z,
        endpoints["last_hit"].x,
        endpoints["last_hit"].y,
        endpoints["last_hit"].z,
    )


def _reflection_replay_loss(vector, endpoints):
    components = _reflection_replay_output_components(vector, endpoints)
    weights = [
        0.11, -0.07, 0.09, -0.05, 0.13, -0.03,
        0.04, -0.06, 0.08,
        -0.02, 0.05, -0.04,
        0.03, -0.01, 0.02,
    ]
    loss = dr.zeros(wt.Float, 1)
    for component, weight in zip(components, weights):
        loss += dr.sum(component * wt.Float(weight))
    return loss


def _clone_reflection_grid_inputs(case: dict):
    prev_refl_p = _detach_point3f(case["prev_refl_p"])
    prev_refl_n = _detach_vector3f(case["prev_refl_n"])
    prev_tx = _detach_point3f(case["prev_tx"])
    prev_weight = _detach_complex(case["prev_weight"])
    prev_polarization = _detach_vector_complex(case["prev_polarization"])
    return {
        "prev_refl_p": prev_refl_p,
        "prev_refl_n": prev_refl_n,
        "prev_tx": prev_tx,
        "prev_weight": prev_weight,
        "prev_polarization": prev_polarization,
    }


def _clone_suffix_grid_inputs(case: dict):
    seg_origin = _detach_point3f(case["seg_origin"])
    seg_field = _detach_complex(case["seg_field"])
    seg_vector = _detach_vector_complex(case["seg_vector"])
    return {
        "seg_origin": seg_origin,
        "seg_field": seg_field,
        "seg_vector": seg_vector,
    }


def _reflection_grid_jvp_components(outputs):
    return (
        outputs[0],
        outputs[1],
        outputs[3],
        outputs[4],
        outputs[5],
        outputs[6],
        outputs[7],
        outputs[8],
    )


def _reflection_grid_loss(outputs):
    n_cells = dr.width(outputs[0])
    weights = [
        dr.linspace(wt.Float, 0.1, 0.4, n_cells),
        dr.linspace(wt.Float, -0.3, 0.2, n_cells),
        dr.linspace(wt.Float, 0.05, 0.15, n_cells),
        dr.linspace(wt.Float, -0.12, 0.08, n_cells),
        dr.linspace(wt.Float, 0.07, 0.17, n_cells),
        dr.linspace(wt.Float, -0.15, 0.05, n_cells),
        dr.linspace(wt.Float, 0.09, 0.19, n_cells),
        dr.linspace(wt.Float, -0.18, 0.02, n_cells),
    ]
    loss = dr.zeros(wt.Float, 1)
    for component, weight in zip(_reflection_grid_jvp_components(outputs), weights):
        loss += dr.sum(component * weight)
    return loss


def _suffix_grid_loss(field, vector):
    n_cells = dr.width(field.real)
    weights = {
        "field_real": dr.linspace(wt.Float, 0.11, 0.29, n_cells),
        "field_imag": dr.linspace(wt.Float, -0.21, 0.07, n_cells),
        "x_real": dr.linspace(wt.Float, 0.03, 0.17, n_cells),
        "x_imag": dr.linspace(wt.Float, -0.19, 0.05, n_cells),
        "y_real": dr.linspace(wt.Float, 0.09, 0.23, n_cells),
        "y_imag": dr.linspace(wt.Float, -0.13, 0.11, n_cells),
        "z_real": dr.linspace(wt.Float, 0.02, 0.14, n_cells),
        "z_imag": dr.linspace(wt.Float, -0.07, 0.09, n_cells),
    }
    loss = dr.zeros(wt.Float, 1)
    loss += dr.sum(field.real * weights["field_real"])
    loss += dr.sum(field.imag * weights["field_imag"])
    loss += dr.sum(vector["x"].real * weights["x_real"])
    loss += dr.sum(vector["x"].imag * weights["x_imag"])
    loss += dr.sum(vector["y"].real * weights["y_real"])
    loss += dr.sum(vector["y"].imag * weights["y_imag"])
    loss += dr.sum(vector["z"].real * weights["z_real"])
    loss += dr.sum(vector["z"].imag * weights["z_imag"])
    return loss


def _suffix_grid_tangent() -> dict:
    return {
        "seg_origin": wt.Point3f(
            [0.03, -0.02, 0.01, -0.01, 0.02, 0.00],
            [0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
            [-0.01, 0.02, -0.02, 0.03, -0.03, 0.00],
        ),
        "seg_field": wt.Complex2f(
            [0.04, -0.03, 0.02, -0.01, 0.03, 0.00],
            [-0.02, 0.01, -0.01, 0.02, -0.03, 0.00],
        ),
        "seg_vector": {
            "x": wt.Complex2f(
                [0.02, -0.01, 0.01, -0.02, 0.02, 0.00],
                [0.00, 0.01, -0.01, 0.01, -0.01, 0.00],
            ),
            "y": wt.Complex2f(
                [-0.01, 0.02, -0.02, 0.01, -0.01, 0.00],
                [0.01, -0.01, 0.02, -0.02, 0.01, 0.00],
            ),
            "z": wt.Complex2f(
                [0.01, 0.00, -0.01, 0.02, -0.02, 0.00],
                [-0.01, 0.02, 0.00, -0.01, 0.01, 0.00],
            ),
        },
    }


def _suffix_grid_grad_dot_tangent(grads: dict, tangent: dict) -> float:
    total = dr.zeros(wt.Float, 1)
    total += dr.sum(
        grads["seg_origin"].x * tangent["seg_origin"].x
        + grads["seg_origin"].y * tangent["seg_origin"].y
        + grads["seg_origin"].z * tangent["seg_origin"].z
    )
    total += dr.sum(
        grads["seg_field"].real * tangent["seg_field"].real
        + grads["seg_field"].imag * tangent["seg_field"].imag
    )
    for axis in ("x", "y", "z"):
        total += dr.sum(
            grads["seg_vector"][axis].real * tangent["seg_vector"][axis].real
            + grads["seg_vector"][axis].imag * tangent["seg_vector"][axis].imag
    )
    return float(total[0])


def _radiomap_vector_power_case(n_rx: int = 9) -> dict:
    return {
        "field_vector": {
            "x": wt.Complex2f(
                dr.linspace(wt.Float, -0.31, 0.27, n_rx),
                dr.linspace(wt.Float, 0.18, -0.22, n_rx),
            ),
            "y": wt.Complex2f(
                dr.linspace(wt.Float, 0.21, -0.17, n_rx),
                dr.linspace(wt.Float, -0.09, 0.16, n_rx),
            ),
            "z": wt.Complex2f(
                dr.linspace(wt.Float, -0.14, 0.19, n_rx),
                dr.linspace(wt.Float, 0.11, -0.13, n_rx),
            ),
        },
    }


def _radiomap_vector_power_tangent(n_rx: int = 9) -> dict:
    return {
        "field_vector": {
            "x": wt.Complex2f(
                dr.linspace(wt.Float, 0.03, -0.02, n_rx),
                dr.linspace(wt.Float, -0.01, 0.04, n_rx),
            ),
            "y": wt.Complex2f(
                dr.linspace(wt.Float, -0.02, 0.03, n_rx),
                dr.linspace(wt.Float, 0.04, -0.01, n_rx),
            ),
            "z": wt.Complex2f(
                dr.linspace(wt.Float, 0.01, 0.02, n_rx),
                dr.linspace(wt.Float, -0.03, 0.02, n_rx),
            ),
        },
    }


def _clone_radiomap_vector_power_inputs(case: dict) -> dict:
    return {"field_vector": _detach_vector_complex(case["field_vector"])}


def _radiomap_vector_power_loss(power):
    n_rx = dr.width(power)
    weights = dr.linspace(wt.Float, -0.25, 0.35, n_rx)
    return dr.sum(power * weights)


def _radiomap_matched_isb_case(n_rx: int = 7) -> dict:
    t = np.linspace(-1.0, 1.0, n_rx, dtype=np.float32)

    def _normalized_vectors(phase: float, z_bias: float):
        vec = np.stack(
            [
                np.cos(phase + 0.45 * np.pi * t),
                0.35 * np.sin((phase + 0.3) + 0.7 * np.pi * t),
                z_bias + 0.2 * t,
            ],
            axis=0,
        )
        norm = np.maximum(np.linalg.norm(vec, axis=0, keepdims=True), 1.0e-6)
        vec = vec / norm
        return wt.Vector3f(vec[0], vec[1], vec[2])

    return {
        "continued_direct": wt.Complex2f(
            np.linspace(0.05, 0.17, n_rx, dtype=np.float32),
            np.linspace(-0.08, 0.06, n_rx, dtype=np.float32),
        ),
        "tx_basis": _normalized_vectors(0.1, 0.45),
        "rx_basis": _normalized_vectors(0.6, 0.30),
        "hard_visibility": wt.Float(np.array([1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0], dtype=np.float32)),
        "interior_mask": wt.Int32(np.array([0, 0, 1, 0, 0, 0, 1], dtype=np.int32)),
        "incident_weight": wt.Float(np.linspace(0.05, 0.55, n_rx, dtype=np.float32)),
        "incident_response": wt.Complex2f(
            np.linspace(0.7, 1.1, n_rx, dtype=np.float32),
            np.linspace(-0.15, 0.20, n_rx, dtype=np.float32),
        ),
        "raw_transition_vector": {
            "x": wt.Complex2f(
                np.linspace(0.12, -0.06, n_rx, dtype=np.float32),
                np.linspace(-0.04, 0.07, n_rx, dtype=np.float32),
            ),
            "y": wt.Complex2f(
                np.linspace(-0.09, 0.11, n_rx, dtype=np.float32),
                np.linspace(0.05, -0.03, n_rx, dtype=np.float32),
            ),
            "z": wt.Complex2f(
                np.linspace(0.04, 0.10, n_rx, dtype=np.float32),
                np.linspace(-0.02, 0.06, n_rx, dtype=np.float32),
            ),
        },
    }


def _radiomap_matched_isb_tangent(n_rx: int = 7) -> dict:
    return {
        "continued_direct": wt.Complex2f(
            dr.linspace(wt.Float, 0.03, -0.01, n_rx),
            dr.linspace(wt.Float, -0.02, 0.04, n_rx),
        ),
        "tx_basis": wt.Vector3f(
            dr.linspace(wt.Float, 0.01, -0.02, n_rx),
            dr.linspace(wt.Float, -0.03, 0.01, n_rx),
            dr.linspace(wt.Float, 0.02, 0.00, n_rx),
        ),
        "rx_basis": wt.Vector3f(
            dr.linspace(wt.Float, -0.01, 0.03, n_rx),
            dr.linspace(wt.Float, 0.02, -0.02, n_rx),
            dr.linspace(wt.Float, -0.02, 0.01, n_rx),
        ),
        "incident_weight": dr.linspace(wt.Float, 0.02, -0.01, n_rx),
        "incident_response": wt.Complex2f(
            dr.linspace(wt.Float, -0.02, 0.03, n_rx),
            dr.linspace(wt.Float, 0.01, -0.03, n_rx),
        ),
        "raw_transition_vector": {
            "x": wt.Complex2f(
                dr.linspace(wt.Float, 0.01, -0.02, n_rx),
                dr.linspace(wt.Float, -0.03, 0.01, n_rx),
            ),
            "y": wt.Complex2f(
                dr.linspace(wt.Float, -0.02, 0.02, n_rx),
                dr.linspace(wt.Float, 0.03, -0.01, n_rx),
            ),
            "z": wt.Complex2f(
                dr.linspace(wt.Float, 0.02, 0.01, n_rx),
                dr.linspace(wt.Float, -0.01, 0.02, n_rx),
            ),
        },
    }


def _clone_radiomap_matched_isb_inputs(case: dict) -> dict:
    return {
        "continued_direct": _detach_complex(case["continued_direct"]),
        "tx_basis": _detach_vector3f(case["tx_basis"]),
        "rx_basis": _detach_vector3f(case["rx_basis"]),
        "hard_visibility": _detach_float(case["hard_visibility"]),
        "interior_mask": wt.Int32(case["interior_mask"]),
        "incident_weight": _detach_float(case["incident_weight"]),
        "incident_response": _detach_complex(case["incident_response"]),
        "raw_transition_vector": _detach_vector_complex(case["raw_transition_vector"]),
    }


def _radiomap_matched_isb_output_components(outputs):
    return (
        outputs["coherent"].real,
        outputs["coherent"].imag,
        outputs["power"],
        outputs["vector_coherent"]["x"].real,
        outputs["vector_coherent"]["x"].imag,
        outputs["vector_coherent"]["y"].real,
        outputs["vector_coherent"]["y"].imag,
        outputs["vector_coherent"]["z"].real,
        outputs["vector_coherent"]["z"].imag,
        outputs["continued_direct_power"],
        outputs["transition_magnitude"],
        outputs["transition_phase"],
    )


def _radiomap_matched_isb_loss(outputs):
    n_rx = dr.width(outputs["power"])
    weights = [
        dr.linspace(wt.Float, 0.11, -0.07, n_rx),
        dr.linspace(wt.Float, -0.13, 0.09, n_rx),
        dr.linspace(wt.Float, 0.05, 0.19, n_rx),
        dr.linspace(wt.Float, -0.17, 0.03, n_rx),
        dr.linspace(wt.Float, 0.07, -0.05, n_rx),
        dr.linspace(wt.Float, 0.02, 0.14, n_rx),
        dr.linspace(wt.Float, -0.09, 0.06, n_rx),
        dr.linspace(wt.Float, 0.08, -0.04, n_rx),
        dr.linspace(wt.Float, -0.03, 0.12, n_rx),
        dr.linspace(wt.Float, 0.04, 0.10, n_rx),
        dr.linspace(wt.Float, -0.06, 0.08, n_rx),
        dr.linspace(wt.Float, 0.09, -0.02, n_rx),
    ]
    loss = dr.zeros(wt.Float, 1)
    for component, weight in zip(_radiomap_matched_isb_output_components(outputs), weights):
        loss += dr.sum(component * weight)
    return loss


def _radiomap_matched_isb_input_grads(inputs: dict) -> dict:
    return {
        "continued_direct": _detach_complex(dr.grad(inputs["continued_direct"])),
        "tx_basis": _detach_vector3f(dr.grad(inputs["tx_basis"])),
        "rx_basis": _detach_vector3f(dr.grad(inputs["rx_basis"])),
        "incident_weight": _detach_float(dr.grad(inputs["incident_weight"])),
        "incident_response": _detach_complex(dr.grad(inputs["incident_response"])),
        "raw_transition_vector": {
            "x": _detach_complex(dr.grad(inputs["raw_transition_vector"]["x"])),
            "y": _detach_complex(dr.grad(inputs["raw_transition_vector"]["y"])),
            "z": _detach_complex(dr.grad(inputs["raw_transition_vector"]["z"])),
        },
    }


def _radiomap_shadow_boundary_incident_stats_case(n_rx: int = 5, n_edges: int = 4) -> dict:
    rx_t = np.linspace(-1.0, 1.0, n_rx, dtype=np.float32)
    edge_t = np.linspace(-1.0, 1.0, n_edges, dtype=np.float32)
    edge_dir = np.stack(
        [
            np.zeros(n_edges, dtype=np.float32),
            np.zeros(n_edges, dtype=np.float32),
            np.ones(n_edges, dtype=np.float32),
        ],
        axis=0,
    )
    return {
        "tx_pos": wt.Point3f(1.8, 1.4, 0.2),
        "rx_pos": wt.Point3f(
            1.1 + 0.35 * rx_t,
            0.9 + 0.25 * np.cos(0.8 * np.pi * rx_t),
            0.15 + 0.20 * rx_t,
        ),
        "edge_pos": wt.Point3f(
            0.15 + 0.45 * edge_t,
            0.10 + 0.30 * np.sin(0.5 * np.pi * edge_t),
            np.linspace(-0.35, 0.35, n_edges, dtype=np.float32),
        ),
        "edge_dir": wt.Vector3f(edge_dir[0], edge_dir[1], edge_dir[2]),
        "n0": wt.Vector3f(
            np.ones(n_edges, dtype=np.float32),
            np.zeros(n_edges, dtype=np.float32),
            np.zeros(n_edges, dtype=np.float32),
        ),
        "n_face_n": wt.Vector3f(
            np.zeros(n_edges, dtype=np.float32),
            np.ones(n_edges, dtype=np.float32),
            np.zeros(n_edges, dtype=np.float32),
        ),
        "wedge_n": wt.Float(np.array([1.6, 1.35, 0.95, 1.8], dtype=np.float32)[:n_edges]),
        "edge_line_min": wt.Float(np.linspace(-0.8, -0.4, n_edges, dtype=np.float32)),
        "edge_line_max": wt.Float(np.linspace(0.6, 1.0, n_edges, dtype=np.float32)),
        "source_visible": wt.Bool(np.array([True, False, True, True], dtype=np.bool_)[:n_edges]),
        "k": float(2.0 * np.pi / 0.31),
    }


def _radiomap_shadow_boundary_incident_stats_tangent(n_rx: int = 5, n_edges: int = 4) -> dict:
    return {
        "tx_pos": wt.Point3f(0.07, -0.04, 0.03),
        "rx_pos": wt.Point3f(
            dr.linspace(wt.Float, -0.03, 0.05, n_rx),
            dr.linspace(wt.Float, 0.02, -0.01, n_rx),
            dr.linspace(wt.Float, -0.02, 0.04, n_rx),
        ),
        "edge_pos": wt.Point3f(
            dr.linspace(wt.Float, 0.04, -0.03, n_edges),
            dr.linspace(wt.Float, -0.02, 0.01, n_edges),
            dr.linspace(wt.Float, 0.03, 0.02, n_edges),
        ),
    }


def _radiomap_shadow_boundary_inner_taper_case() -> dict:
    return {
        "tx_pos": wt.Point3f(1.0, 1.0, 0.0),
        "rx_pos": wt.Point3f(
            [1.0, -0.02, -1.0],
            [-1.0, -1.0, -1.0],
            [0.0, 0.0, 0.0],
        ),
        "edge_pos": wt.Point3f([0.0], [0.0], [0.0]),
        "edge_dir": wt.Vector3f([0.0], [0.0], [1.0]),
        "n0": wt.Vector3f([1.0], [0.0], [0.0]),
        "n_face_n": wt.Vector3f([0.0], [1.0], [0.0]),
        "wedge_n": wt.Float([1.5]),
        "edge_line_min": wt.Float([-10.0]),
        "edge_line_max": wt.Float([10.0]),
        "source_visible": wt.Bool([True]),
        "k": float(2.0 * np.pi / 0.31),
    }


def _clone_radiomap_shadow_boundary_incident_stats_inputs(case: dict) -> dict:
    return {
        "tx_pos": _detach_point3f(case["tx_pos"]),
        "rx_pos": _detach_point3f(case["rx_pos"]),
        "edge_pos": _detach_point3f(case["edge_pos"]),
        "edge_dir": _detach_vector3f(case["edge_dir"]),
        "n0": _detach_vector3f(case["n0"]),
        "n_face_n": _detach_vector3f(case["n_face_n"]),
        "wedge_n": _detach_float(case["wedge_n"]),
        "edge_line_min": _detach_float(case["edge_line_min"]),
        "edge_line_max": _detach_float(case["edge_line_max"]),
        "source_visible": wt.Bool(case["source_visible"]),
        "k": float(case["k"]),
    }


def _radiomap_shadow_boundary_incident_stats_loss(outputs: dict) -> wt.Float:
    n_rx = dr.width(outputs["sum_incident_weight"])
    weights = [
        dr.linspace(wt.Float, 0.08, -0.03, n_rx),
        dr.linspace(wt.Float, -0.05, 0.09, n_rx),
        dr.linspace(wt.Float, 0.11, 0.02, n_rx),
        dr.linspace(wt.Float, -0.07, 0.06, n_rx),
    ]
    components = (
        outputs["sum_incident_weight"],
        outputs["max_incident_weight"],
        outputs["weighted_incident_response_real"],
        outputs["weighted_incident_response_imag"],
    )
    loss = dr.zeros(wt.Float, 1)
    for component, weight in zip(components, weights):
        loss += dr.sum(component * weight)
    return loss


def _radiomap_shadow_boundary_incident_stats_input_grads(inputs: dict) -> dict:
    return {
        "tx_pos": _detach_point3f(dr.grad(inputs["tx_pos"])),
        "rx_pos": _detach_point3f(dr.grad(inputs["rx_pos"])),
        "edge_pos": _detach_point3f(dr.grad(inputs["edge_pos"])),
    }


def _assert_state_arrays_match(ref: dict, got: dict, tol: float = 1.0e-6) -> None:
    assert ref["n_states"] == got["n_states"]
    for key in STATE_STATIC_KEYS:
        if key not in ref and key not in got:
            continue
        assert _state_value_max_abs_diff(ref[key], got[key]) <= tol, key
    history_size = _state_history_size(ref)
    assert history_size == _state_history_size(got)
    ref_edge_history, ref_reflection_history = _materialize_state_history(ref)
    got_edge_history, got_reflection_history = _materialize_state_history(got)
    for slot in range(history_size):
        assert _state_value_max_abs_diff(
            ref_edge_history[slot],
            got_edge_history[slot],
        ) == 0.0
        assert _state_value_max_abs_diff(
            ref_reflection_history[slot],
            got_reflection_history[slot],
        ) == 0.0


def _assert_inserted_reflection_fields_match(ref: dict, got: dict, tol: float = 1.0e-6) -> None:
    assert ref["n_states"] == got["n_states"]
    for key in (
        "edge_pos",
        "path_length_prefix",
        "first_interaction_pos",
        "source_type_code",
        "prefix_reflection_depth",
        "intermediate_reflection_depth",
        "suffix_reflection_depth",
        "order",
    ):
        assert _state_value_max_abs_diff(ref[key], got[key]) <= tol, key
    history_size = _state_history_size(ref)
    assert history_size == _state_history_size(got)
    ref_edge_history, ref_reflection_history = _materialize_state_history(ref)
    got_edge_history, got_reflection_history = _materialize_state_history(got)
    for slot in range(history_size):
        assert _state_value_max_abs_diff(
            ref_edge_history[slot],
            got_edge_history[slot],
        ) == 0.0
        assert _state_value_max_abs_diff(
            ref_reflection_history[slot],
            got_reflection_history[slot],
        ) == 0.0


def _assert_field_eval_state_fields_match(ref: dict, got: dict, tol: float = 1.0e-6) -> None:
    assert ref["n_states"] == got["n_states"]
    for key in (
        "edge_pos",
        "edge_dir",
        "n0",
        "n_face_n",
        "wedge_n",
        "source_pos",
        "edge_line_min",
        "edge_line_max",
        "incident_jones_u",
        "incident_jones_v",
        "incident_derivative_jones_u",
        "incident_derivative_jones_v",
        "incident_basis_u",
        "incident_basis_v",
        "incident_basis_k",
        "face0_eta_r",
        "face0_sigma",
        "face0_gain",
        "face1_eta_r",
        "face1_sigma",
        "face1_gain",
        "face0_operator_m00",
        "face0_operator_m01",
        "face0_operator_m10",
        "face0_operator_m11",
        "face1_operator_m00",
        "face1_operator_m01",
        "face1_operator_m10",
        "face1_operator_m11",
    ):
        if key not in ref or key not in got:
            continue
        assert _state_value_max_abs_diff(ref[key], got[key]) <= tol, key


def _assert_path_slot_build_match(ref, got, tol: float = 1.0e-6) -> None:
    ref_type, ref_vertex, ref_normal, ref_object, ref_max_depth = ref
    got_type, got_vertex, got_normal, got_object, got_max_depth = got
    assert ref_max_depth == got_max_depth
    assert len(ref_type) == len(got_type)
    for ref_slot, got_slot in zip(ref_type, got_type):
        assert _state_value_max_abs_diff(ref_slot, got_slot) == 0.0
    if ref_vertex is None:
        assert got_vertex is None
        assert got_normal is None
        assert got_object is None
        return
    assert got_vertex is not None
    assert got_normal is not None
    assert got_object is not None
    for ref_slot, got_slot in zip(ref_vertex, got_vertex):
        assert _state_value_max_abs_diff(ref_slot, got_slot) <= tol
    for ref_slot, got_slot in zip(ref_normal, got_normal):
        assert _state_value_max_abs_diff(ref_slot, got_slot) <= tol
    for ref_slot, got_slot in zip(ref_object, got_object):
        assert _state_value_max_abs_diff(ref_slot, got_slot) == 0.0


def _build_reflection_case():
    wavelength = 299792458.0 / 1.0e9
    k = float(2.0 * dr.pi / wavelength)
    scene = build_test_scene(
        box_geometry(center=(-2.0, -2.0, 1.5), size=2.0),
        box_geometry(center=(2.0, 1.5, 1.5), size=2.0),
    )
    field = Field(bounds=((-3.0, 3.0), (-3.0, 3.0)), size=(5, 5))
    tx = wt.Point3f(0.0, -4.0, 1.5)
    _, _, reflection_detail = compute_reflection_field(
        grid=field,
        rx_z=1.5,
        tx_pos=tx,
        scene=scene,
        wavelength=wavelength,
        k=k,
        n_rays=64,
        max_reflections=1,
        mode="3d",
        reflection_coef=1.0,
        tx_polarization=(1.0, 0.0, 0.0),
        reflection_material=None,
        use_scene_materials=False,
        return_per_bounce=False,
    )
    detail = coerce_reflection_trace_detail(reflection_detail)
    return {
        "rx_pos": field.receiver_positions_3d(position=1.5),
        "scene": scene,
        "wavelength": wavelength,
        "k": k,
        "source_paths_per_bounce": list(detail.source_paths_per_bounce),
        "reflection_detail": detail,
        "tx_polarization": (1.0, 0.0, 0.0),
    }


def _build_utd_case():
    wavelength = 299792458.0 / 1.0e9
    k_dr = 2.0 * dr.pi / wavelength
    k = float(k_dr)
    scene = build_test_scene(
        box_geometry(center=(-1.8, -1.2, 1.5), size=2.0),
        box_geometry(center=(1.8, 1.2, 1.5), size=2.0),
    )
    edge_data = scene.get_edge_data(1.5)["edge_data"]
    tx = wt.Point3f(0.0, -4.0, 1.5)
    state_arrays = _build_tx_first_order_state_arrays(tx, edge_data, wavelength, k_dr)
    field = Field(bounds=((-2.0, 2.0), (-2.0, 2.0)), size=(4, 4))
    coords = field.get_coordinates()
    rx_pos = wt.Point3f(coords["X"], coords["Y"], wt.Float(1.5))
    return {
        "state_arrays": state_arrays,
        "rx_pos": rx_pos,
        "k": k,
        "wavelength": wavelength,
        "n_edges": edge_data["n_edges"],
        "scene": scene,
    }


def _build_mesh_kernel_case():
    scene = build_test_scene(
        box_geometry(center=(-1.8, -1.2, 1.5), size=2.0),
        box_geometry(center=(1.8, 1.2, 1.5), size=2.0),
    )
    interior_edges = [
        (edge_vertices, adjacent_faces)
        for edge_vertices, adjacent_faces in scene._edge_topology
        if len(adjacent_faces) == 2
    ]
    return {
        "vertices": scene.vertices,
        "faces": scene.faces,
        "face_normals": scene._face_normals,
        "edge_v0": wt.UInt32([int(edge_vertices[0]) for edge_vertices, _ in interior_edges]),
        "edge_v1": wt.UInt32([int(edge_vertices[1]) for edge_vertices, _ in interior_edges]),
        "edge_face0": wt.Int32([int(adjacent_faces[0]) for _, adjacent_faces in interior_edges]),
        "edge_face1": wt.Int32([int(adjacent_faces[1]) for _, adjacent_faces in interior_edges]),
        "n_edges": len(interior_edges),
    }


def _build_cartesian_filter_case():
    prev_edge_idx = wt.UInt32([3, 1, 4, 0])
    prev_edge_history = [
        wt.Int32([-1, 0, 2, -1]),
        wt.Int32([-1, -1, 0, 3]),
    ]
    prev_power = wt.Float([1.0, 1.0e-30, 2.0, 0.25])
    return {
        "prev_edge_idx": prev_edge_idx,
        "prev_edge_history": prev_edge_history,
        "prev_power": prev_power,
        "n_prev": 4,
        "n_edges": 6,
        "min_power": 1.0e-20,
    }


def _build_cartesian_filter_multiblock_case():
    n_prev = 384
    n_edges = 96
    prev_idx = dr.arange(wt.UInt32, n_prev)
    prev_edge_idx = prev_idx % wt.UInt32(n_edges)
    hist0 = dr.select(
        prev_idx > wt.UInt32(0),
        wt.Int32((prev_idx - wt.UInt32(1)) % wt.UInt32(n_edges)),
        wt.Int32(-1),
    )
    hist1 = dr.select(
        prev_idx > wt.UInt32(1),
        wt.Int32((prev_idx - wt.UInt32(2)) % wt.UInt32(n_edges)),
        wt.Int32(-1),
    )
    hist2 = dr.select(
        prev_idx > wt.UInt32(2),
        wt.Int32((prev_idx - wt.UInt32(3)) % wt.UInt32(n_edges)),
        wt.Int32(-1),
    )
    prev_power = dr.select(
        (prev_idx % wt.UInt32(11)) == wt.UInt32(0),
        wt.Float(1.0e-30),
        wt.Float(1.0),
    )
    return {
        "prev_edge_idx": prev_edge_idx,
        "prev_edge_history": [hist0, hist1, hist2],
        "prev_power": prev_power,
        "n_prev": n_prev,
        "n_edges": n_edges,
        "min_power": 1.0e-20,
    }


def _build_compact_index_pairs_case():
    count = 513
    idx = dr.arange(wt.UInt32, count)
    return {
        "lhs_idx": (idx * wt.UInt32(7)) % wt.UInt32(103),
        "rhs_idx": (idx * wt.UInt32(11)) % wt.UInt32(97),
        "active_mask": (((idx * wt.UInt32(5)) % wt.UInt32(17)) < wt.UInt32(6)) | ((idx % wt.UInt32(29)) == 0),
    }


def _build_pruning_tie_case():
    n_states = 4

    def _float(values):
        return wt.Float(values)

    def _int(values):
        return wt.Int32(values)

    def _uint(values):
        return wt.UInt32(values)

    def _complex(values):
        return wt.Complex2f(_float(values), _float([0.0] * n_states))

    parent_states = _make_state_arrays(
        edge_idx=_uint([0, 1, 2, 3]),
        edge_pos=wt.Point3f(
            _float([0.0, 1.0, 2.0, 3.0]),
            _float([0.0, 0.0, 0.0, 0.0]),
            _float([0.0, 0.0, 0.0, 0.0]),
        ),
        edge_dir=wt.Vector3f(
            _float([1.0, 1.0, 1.0, 1.0]),
            _float([0.0, 0.0, 0.0, 0.0]),
            _float([0.0, 0.0, 0.0, 0.0]),
        ),
        n0=wt.Vector3f(
            _float([0.0, 0.0, 0.0, 0.0]),
            _float([1.0, 1.0, 1.0, 1.0]),
            _float([0.0, 0.0, 0.0, 0.0]),
        ),
        nn=wt.Vector3f(
            _float([0.0, 0.0, 0.0, 0.0]),
            _float([0.0, 0.0, 0.0, 0.0]),
            _float([1.0, 1.0, 1.0, 1.0]),
        ),
        wedge_n=_float([1.0, 1.0, 1.0, 1.0]),
        adjacent_face0=_int([0, 0, 0, 0]),
        adjacent_face1=_int([1, 1, 1, 1]),
        source_pos=wt.Point3f(
            _float([0.0, 0.0, 0.0, 0.0]),
            _float([0.0, 0.0, 0.0, 0.0]),
            _float([-1.0, -1.0, -1.0, -1.0]),
        ),
        edge_line_min=_float([-0.5, -0.5, -0.5, -0.5]),
        edge_line_max=_float([0.5, 0.5, 0.5, 0.5]),
        incident_field=_complex([0.0, 0.0, 0.0, 0.0]),
        incident_normal_derivative=_complex([0.0, 0.0, 0.0, 0.0]),
        incident_jones={
            "u": _complex([1.0, 1.0, 1.0, 1.0]),
            "v": _complex([0.0, 0.0, 0.0, 0.0]),
        },
        incident_derivative_jones={
            "u": _complex([0.0, 0.0, 0.0, 0.0]),
            "v": _complex([0.0, 0.0, 0.0, 0.0]),
        },
        order=_uint([1, 1, 1, 1]),
        prefix_reflection_depth=_uint([0, 0, 0, 0]),
        intermediate_reflection_depth=_uint([0, 0, 0, 0]),
        suffix_reflection_depth=_uint([0, 0, 0, 0]),
        lineage_parent_state_id=_int([-1, -1, -1, -1]),
        lineage_last_edge_idx=_int([0, 1, 2, 3]),
        lineage_last_reflection_depth_delta=_uint([0, 0, 0, 0]),
    )
    parent_states, lineage_store, next_state_id = _finalize_state_lineage(
        parent_states,
        lineage_store=None,
        next_state_id=0,
    )

    state_arrays = _make_state_arrays(
        edge_idx=_uint([100, 1, 50, 2]),
        edge_pos=wt.Point3f(
            _float([0.0, 1.0, 2.0, 3.0]),
            _float([0.0, 0.0, 0.0, 0.0]),
            _float([0.0, 0.0, 0.0, 0.0]),
        ),
        edge_dir=wt.Vector3f(
            _float([1.0, 1.0, 1.0, 1.0]),
            _float([0.0, 0.0, 0.0, 0.0]),
            _float([0.0, 0.0, 0.0, 0.0]),
        ),
        n0=wt.Vector3f(
            _float([0.0, 0.0, 0.0, 0.0]),
            _float([1.0, 1.0, 1.0, 1.0]),
            _float([0.0, 0.0, 0.0, 0.0]),
        ),
        nn=wt.Vector3f(
            _float([0.0, 0.0, 0.0, 0.0]),
            _float([0.0, 0.0, 0.0, 0.0]),
            _float([1.0, 1.0, 1.0, 1.0]),
        ),
        wedge_n=_float([1.0, 1.0, 1.0, 1.0]),
        adjacent_face0=_int([0, 0, 0, 0]),
        adjacent_face1=_int([1, 1, 1, 1]),
        source_pos=wt.Point3f(
            _float([0.0, 0.0, 0.0, 0.0]),
            _float([0.0, 0.0, 0.0, 0.0]),
            _float([-1.0, -1.0, -1.0, -1.0]),
        ),
        edge_line_min=_float([-0.5, -0.5, -0.5, -0.5]),
        edge_line_max=_float([0.5, 0.5, 0.5, 0.5]),
        incident_field=_complex([0.0, 0.0, 0.0, 0.0]),
        incident_normal_derivative=_complex([0.0, 0.0, 0.0, 0.0]),
        incident_jones={
            "u": _complex([1.0, 1.0, 1.0, 1.0]),
            "v": _complex([0.0, 0.0, 0.0, 0.0]),
        },
        incident_derivative_jones={
            "u": _complex([0.0, 0.0, 0.0, 0.0]),
            "v": _complex([0.0, 0.0, 0.0, 0.0]),
        },
        order=_uint([2, 2, 2, 2]),
        prefix_reflection_depth=_uint([0, 0, 0, 0]),
        intermediate_reflection_depth=_uint([0, 0, 0, 0]),
        suffix_reflection_depth=_uint([0, 0, 0, 0]),
        lineage_parent_state_id=_state_ids(parent_states),
        lineage_last_edge_idx=_int([9, 0, 0, 0]),
        lineage_last_reflection_depth_delta=_uint([0, 0, 0, 0]),
    )
    state_arrays, _, _ = _finalize_state_lineage(
        state_arrays,
        lineage_store=lineage_store,
        next_state_id=next_state_id,
    )
    return state_arrays


def _utd_loss(outputs, grad_direct_total, grad_multi_total, grad_direct_vector, grad_multi_vector):
    loss = dr.zeros(wt.Float, 1)
    direct_total, multi_total, direct_vector, multi_vector, _ = outputs
    loss += dr.sum(
        direct_total.real * grad_direct_total.real
        + direct_total.imag * grad_direct_total.imag
    )
    loss += dr.sum(
        multi_total.real * grad_multi_total.real
        + multi_total.imag * grad_multi_total.imag
    )
    for axis in ("x", "y", "z"):
        loss += dr.sum(
            direct_vector[axis].real * grad_direct_vector[axis].real
            + direct_vector[axis].imag * grad_direct_vector[axis].imag
        )
        loss += dr.sum(
            multi_vector[axis].real * grad_multi_vector[axis].real
            + multi_vector[axis].imag * grad_multi_vector[axis].imag
        )
    return loss


def _build_reflection_grid_case(axis: str):
    wavelength = 299792458.0 / 1.0e9
    k = float(2.0 * dr.pi / wavelength)

    if axis == "z":
        grid = Field(bounds=((-1.6, 1.6), (-1.2, 1.2)), size=(5, 4), axis="z", position=1.5)
        ray_origin = wt.Point3f(
            [-1.35, -0.55, 0.15, 0.85, 1.10, -0.15],
            [-0.95, -0.15, 0.30, 0.75, -0.60, 0.95],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        )
        ray_dir = wt.Vector3f(
            [0.55, 0.45, -0.35, -0.25, -0.45, 0.20],
            [0.35, 0.20, 0.40, -0.30, 0.10, -0.55],
            [0.15, 0.10, 0.20, 0.05, 0.12, 0.18],
        )
        blocker_dist = wt.Float([4.5, 3.2, 2.8, 2.4, 1.9, 3.8])
    elif axis == "x":
        grid = Field(bounds=((-1.5, 1.5), (0.4, 2.6)), size=(5, 4), axis="x", position=0.35)
        ray_origin = wt.Point3f(
            [-1.20, -0.90, -0.60, -0.35, -0.80, -0.50],
            [-1.10, -0.25, 0.40, 0.95, 1.30, -0.60],
            [0.75, 1.15, 1.80, 2.30, 0.55, 2.45],
        )
        ray_dir = wt.Vector3f(
            [0.80, 0.65, 0.55, 0.45, -0.35, 0.02],
            [0.25, 0.10, -0.20, 0.15, 0.05, 0.10],
            [0.12, -0.08, 0.10, -0.05, 0.02, -0.03],
        )
        blocker_dist = wt.Float([2.2, 2.4, 2.0, 1.8, 2.5, 1.0])
    else:
        raise ValueError(f"Unsupported reflection grid axis: {axis}")

    active = wt.Bool([True, True, True, True, True, False])
    prev_refl_p = wt.Point3f(
        [0.10, 0.05, -0.15, 0.08, -0.05, 0.12],
        [-0.20, 0.15, 0.05, -0.12, 0.18, -0.08],
        [1.25, 1.30, 1.28, 1.22, 1.35, 1.18],
    )
    prev_refl_n = wt.Vector3f(
        [0.10, -0.08, 0.06, -0.04, 0.02, 0.03],
        [0.02, 0.05, -0.03, 0.04, -0.01, 0.02],
        [0.99, 0.98, 0.995, 0.985, 0.99, 0.97],
    )
    prev_tx = wt.Point3f(
        [-0.30, -0.15, 0.10, 0.25, 0.35, -0.05],
        [-3.20, -3.00, -2.85, -3.10, -2.95, -3.05],
        [1.70, 1.68, 1.72, 1.66, 1.75, 1.69],
    )
    prev_weight = wt.Complex2f(
        [0.80, 0.65, 0.55, 0.72, 0.48, 0.60],
        [0.10, -0.08, 0.04, -0.12, 0.06, 0.02],
    )
    prev_polarization = {
        "x": wt.Complex2f(
            [0.30, 0.22, 0.14, 0.18, 0.11, 0.09],
            [0.05, -0.03, 0.02, -0.04, 0.01, 0.00],
        ),
        "y": wt.Complex2f(
            [0.12, 0.18, 0.16, 0.09, 0.14, 0.07],
            [-0.02, 0.03, -0.01, 0.02, -0.03, 0.01],
        ),
        "z": wt.Complex2f(
            [0.06, 0.04, 0.08, 0.05, 0.03, 0.02],
            [0.01, -0.02, 0.00, 0.01, -0.01, 0.00],
        ),
    }
    prev_prim_idx = wt.UInt32([0, 1, 2, 3, 4, 5])

    return {
        "grid": grid,
        "grid_data": grid.get_coordinates(),
        "ray_origin": ray_origin,
        "ray_dir": ray_dir,
        "active": active,
        "blocker_dist": blocker_dist,
        "prev_refl_p": prev_refl_p,
        "prev_refl_n": prev_refl_n,
        "prev_tx": prev_tx,
        "prev_weight": prev_weight,
        "prev_polarization": prev_polarization,
        "prev_prim_idx": prev_prim_idx,
        "wavelength": wavelength,
        "k": k,
    }


def _run_reflection_grid_reference(case: dict):
    grid = case["grid"]
    n_cells = grid.n_cells
    result_real = [dr.zeros(wt.Float, n_cells)]
    result_imag = [dr.zeros(wt.Float, n_cells)]
    result_count = [dr.zeros(wt.Float, n_cells)]
    result_pol_real_x = [dr.zeros(wt.Float, n_cells)]
    result_pol_imag_x = [dr.zeros(wt.Float, n_cells)]
    result_pol_real_y = [dr.zeros(wt.Float, n_cells)]
    result_pol_imag_y = [dr.zeros(wt.Float, n_cells)]
    result_pol_real_z = [dr.zeros(wt.Float, n_cells)]
    result_pol_imag_z = [dr.zeros(wt.Float, n_cells)]
    (coord_0_min, coord_0_max), (coord_1_min, coord_1_max) = grid.bounds
    n_coord_0, n_coord_1 = grid.size
    max_steps = 2 * (n_coord_0 + n_coord_1)

    if grid.axis == "z":
        reflection_grid_drjit.run_dda_traversal(
            grid=grid,
            ray_origin=case["ray_origin"],
            ray_dir=case["ray_dir"],
            active=case["active"],
            blocker_dist=case["blocker_dist"],
            prev_refl_p=case["prev_refl_p"],
            prev_refl_n=case["prev_refl_n"],
            prev_tx=case["prev_tx"],
            prev_weight=case["prev_weight"],
            prev_polarization=case["prev_polarization"],
            prev_prim_idx=case["prev_prim_idx"],
            x_min=coord_0_min,
            x_max=coord_0_max,
            y_min=coord_1_min,
            y_max=coord_1_max,
            cell_size_x=grid.cell_size[0],
            cell_size_y=grid.cell_size[1],
            nx=n_coord_0,
            max_steps=max_steps,
            x_coords_dr=case["grid_data"]["x_coords"],
            y_coords_dr=case["grid_data"]["y_coords"],
            rx_z=grid.position,
            wavelength=case["wavelength"],
            k=case["k"],
            validate_paths=False,
            has_mesh_data=False,
            tri_v0=None,
            tri_v1=None,
            tri_v2=None,
            tri_surface_data=None,
            result_real=result_real,
            result_imag=result_imag,
            result_count=result_count,
            result_pol_real_x=result_pol_real_x,
            result_pol_imag_x=result_pol_imag_x,
            result_pol_real_y=result_pol_real_y,
            result_pol_imag_y=result_pol_imag_y,
            result_pol_real_z=result_pol_real_z,
            result_pol_imag_z=result_pol_imag_z,
            bounce_idx=0,
        )
    else:
        intersections = reflection_grid_drjit.prepare_plane_intersections(
            grid=grid,
            ray_origin=case["ray_origin"],
            ray_dir=case["ray_dir"],
            active=case["active"],
            blocker_dist=case["blocker_dist"],
            plane_position=grid.position,
        )
        reflection_grid_drjit.intersect_and_scatter(
            grid=grid,
            plane_position=grid.position,
            intersections=intersections,
            prev_refl_p=case["prev_refl_p"],
            prev_refl_n=case["prev_refl_n"],
            prev_tx=case["prev_tx"],
            prev_weight=case["prev_weight"],
            prev_polarization=case["prev_polarization"],
            prev_prim_idx=case["prev_prim_idx"],
            nx=n_coord_0,
            x_coords_dr=case["grid_data"]["x_coords"],
            y_coords_dr=case["grid_data"]["y_coords"],
            wavelength=case["wavelength"],
            k=case["k"],
            validate_paths=False,
            has_mesh_data=False,
            tri_v0=None,
            tri_v1=None,
            tri_v2=None,
            tri_surface_data=None,
            result_real=result_real,
            result_imag=result_imag,
            result_count=result_count,
            result_pol_real_x=result_pol_real_x,
            result_pol_imag_x=result_pol_imag_x,
            result_pol_real_y=result_pol_real_y,
            result_pol_imag_y=result_pol_imag_y,
            result_pol_real_z=result_pol_real_z,
            result_pol_imag_z=result_pol_imag_z,
            bounce_idx=0,
        )

    dr.eval(
        result_real[0],
        result_imag[0],
        result_count[0],
        result_pol_real_x[0],
        result_pol_imag_x[0],
        result_pol_real_y[0],
        result_pol_imag_y[0],
        result_pol_real_z[0],
        result_pol_imag_z[0],
    )
    return (
        result_real[0],
        result_imag[0],
        result_count[0],
        result_pol_real_x[0],
        result_pol_imag_x[0],
        result_pol_real_y[0],
        result_pol_imag_y[0],
        result_pol_real_z[0],
        result_pol_imag_z[0],
    )


def _build_suffix_grid_case():
    wavelength = 299792458.0 / 1.0e9
    k = float(2.0 * dr.pi / wavelength)
    grid = Field(bounds=((-1.5, 1.5), (0.6, 2.4)), size=(5, 4), axis="y", position=0.35)
    seg_origin = wt.Point3f(
        [-1.10, -0.55, 0.10, 0.55, 1.05, -0.25],
        [0.35, 0.35, 0.35, 0.35, 0.35, 0.35],
        [0.75, 1.05, 1.35, 1.85, 2.15, 1.55],
    )
    seg_dir = wt.Vector3f(
        [0.50, 0.45, -0.35, -0.25, -0.40, 0.20],
        [0.60, 0.55, 0.50, 0.45, 0.40, 0.35],
        [0.20, 0.10, 0.25, -0.15, 0.05, -0.30],
    )
    blocker_dist = wt.Float([4.0, 3.2, 2.7, 2.4, 1.8, 3.0])
    seg_field = wt.Complex2f(
        [0.75, 0.60, 0.42, 0.35, 0.55, 0.28],
        [0.08, -0.04, 0.05, -0.03, 0.02, 0.01],
    )
    seg_vector = {
        "x": wt.Complex2f(
            [0.22, 0.18, 0.12, 0.08, 0.16, 0.06],
            [0.03, -0.02, 0.01, -0.01, 0.02, 0.00],
        ),
        "y": wt.Complex2f(
            [0.10, 0.12, 0.15, 0.09, 0.11, 0.07],
            [-0.02, 0.03, -0.01, 0.02, -0.03, 0.01],
        ),
        "z": wt.Complex2f(
            [0.05, 0.06, 0.04, 0.07, 0.03, 0.02],
            [0.01, -0.01, 0.00, 0.01, -0.01, 0.00],
        ),
    }
    active = wt.Bool([True, True, True, True, True, False])
    state_idx = wt.UInt32([0, 0, 1, 1, 2, 2])
    return {
        "grid": grid,
        "grid_data": grid.get_coordinates(),
        "seg_origin": seg_origin,
        "seg_dir": seg_dir,
        "blocker_dist": blocker_dist,
        "seg_field": seg_field,
        "seg_vector": seg_vector,
        "state_idx": state_idx,
        "n_states": 3,
        "active": active,
        "wavelength": wavelength,
        "k": k,
    }


@pytest.mark.gpu
def test_native_reflection_forward_matches_drjit_reference():
    if not native_extension_available():
        pytest.skip("native extension is not built in the current environment")

    case = _build_reflection_case()
    ref = reflection_drjit.reflection_accumulate_forward(**case)
    got = reflection_native.reflection_accumulate_forward(**case)

    for axis in ("x", "y", "z"):
        assert _max_abs_diff(ref[0][axis].real, got[0][axis].real) < 1.0e-6
        assert _max_abs_diff(ref[0][axis].imag, got[0][axis].imag) < 1.0e-6


@pytest.mark.gpu
def test_native_reflection_jvp_matches_drjit_reference():
    if not native_extension_available():
        pytest.skip("native extension is not built in the current environment")

    case = _build_reflection_case()
    n_rx = dr.width(case["rx_pos"].x)
    tangent_rx = wt.Point3f(
        dr.linspace(wt.Float, -0.03, 0.04, n_rx),
        dr.linspace(wt.Float, 0.02, -0.01, n_rx),
        dr.linspace(wt.Float, 0.01, 0.05, n_rx),
    )

    ref_rx = _detach_point3f(case["rx_pos"])
    _enable_grad_point3f(ref_rx)
    _set_grad_point3f(ref_rx, tangent_rx)
    ref_outputs = reflection_drjit.reflection_accumulate_forward(
        **{**case, "rx_pos": ref_rx}
    )
    dr.forward_to(_reflection_output_components(ref_outputs), flags=FLAGS)
    ref_jvp = [dr.grad(component) for component in _reflection_output_components(ref_outputs)]

    native_rx = _detach_point3f(case["rx_pos"])
    _enable_grad_point3f(native_rx)
    _set_grad_point3f(native_rx, tangent_rx)
    native_outputs = reflection_native.reflection_accumulate_forward(
        **{**case, "rx_pos": native_rx}
    )
    dr.forward_to(_reflection_output_components(native_outputs), flags=FLAGS)
    native_jvp = [dr.grad(component) for component in _reflection_output_components(native_outputs)]

    for ref_component, got_component in zip(ref_jvp, native_jvp):
        assert _max_abs_diff(ref_component, got_component) < 2.0e-5


@pytest.mark.gpu
def test_native_reflection_replay_to_target_matches_python_reference():
    if not native_extension_available():
        pytest.skip("native extension is not built in the current environment")

    case = _build_reflection_case()
    scene = case["scene"]
    detail = case["reflection_detail"]
    paths = case["source_paths_per_bounce"][0]
    edge_data = scene.get_edge_data(1.5)["edge_data"]
    n_paths = int(paths["n_paths"])
    n_edges = int(edge_data["n_edges"])
    assert n_paths > 0
    assert n_edges > 0

    pair_count = min(12, n_paths * n_edges)
    pair_idx = dr.arange(wt.UInt32, pair_count)
    path_idx = pair_idx % n_paths
    edge_idx = (pair_idx * 3) % n_edges
    target_pos = dr.gather(wt.Point3f, edge_data["pos"], edge_idx)
    target_faces = (
        dr.gather(wt.Int32, edge_data["adjacent_face0"], edge_idx),
        dr.gather(wt.Int32, edge_data["adjacent_face1"], edge_idx),
    )

    ref_valid, ref_vector, ref_endpoints = reflection_epc_module._epc_reflection_chain_to_target_python(
        paths=paths,
        path_idx=path_idx,
        target_pos=target_pos,
        scene=scene,
        target_adjacent_faces=target_faces,
        reflection_detail=detail,
        wavelength=case["wavelength"],
        tx_polarization=case["tx_polarization"],
        return_endpoints=True,
    )
    got_valid, got_vector, got_endpoints = reflection_epc_module.epc_reflection_chain_to_target(
        paths=paths,
        path_idx=path_idx,
        target_pos=target_pos,
        scene=scene,
        target_adjacent_faces=target_faces,
        reflection_detail=detail,
        wavelength=case["wavelength"],
        tx_polarization=case["tx_polarization"],
        return_endpoints=True,
    )

    assert _state_value_max_abs_diff(ref_valid, got_valid) == 0.0
    for axis in ("x", "y", "z"):
        assert _state_value_max_abs_diff(ref_vector[axis].real, got_vector[axis].real) < 1.0e-5
        assert _state_value_max_abs_diff(ref_vector[axis].imag, got_vector[axis].imag) < 1.0e-5
    assert _state_value_max_abs_diff(ref_endpoints["tx_pos"], got_endpoints["tx_pos"]) < 1.0e-5
    assert _state_value_max_abs_diff(ref_endpoints["first_hit"], got_endpoints["first_hit"]) < 1.0e-5
    assert _state_value_max_abs_diff(ref_endpoints["last_hit"], got_endpoints["last_hit"]) < 1.0e-5


@pytest.mark.gpu
def test_native_reflection_replay_to_target_matches_python_reference_for_path_subset():
    if not native_extension_available():
        pytest.skip("native extension is not built in the current environment")

    case = _build_reflection_case()
    scene = case["scene"]
    detail = case["reflection_detail"]
    paths = case["source_paths_per_bounce"][0]
    edge_data = scene.get_edge_data(1.5)["edge_data"]
    n_paths = int(paths["n_paths"])
    n_edges = int(edge_data["n_edges"])
    if n_paths <= 1:
        pytest.skip("reflection case does not expose a non-zero-based path subset")
    pair_count = min(4, n_paths - 1)

    pair_idx = dr.arange(wt.UInt32, pair_count)
    path_idx = pair_idx + wt.UInt32(n_paths - pair_count)
    edge_idx = (pair_idx * 3) % n_edges
    target_pos = dr.gather(wt.Point3f, edge_data["pos"], edge_idx)
    target_faces = (
        dr.gather(wt.Int32, edge_data["adjacent_face0"], edge_idx),
        dr.gather(wt.Int32, edge_data["adjacent_face1"], edge_idx),
    )

    ref_valid, ref_vector, ref_endpoints = reflection_epc_module._epc_reflection_chain_to_target_python(
        paths=paths,
        path_idx=path_idx,
        target_pos=target_pos,
        scene=scene,
        target_adjacent_faces=target_faces,
        reflection_detail=detail,
        wavelength=case["wavelength"],
        tx_polarization=case["tx_polarization"],
        return_endpoints=True,
    )
    got_valid, got_vector, got_endpoints = reflection_epc_module.epc_reflection_chain_to_target(
        paths=paths,
        path_idx=path_idx,
        target_pos=target_pos,
        scene=scene,
        target_adjacent_faces=target_faces,
        reflection_detail=detail,
        wavelength=case["wavelength"],
        tx_polarization=case["tx_polarization"],
        return_endpoints=True,
    )

    assert _state_value_max_abs_diff(ref_valid, got_valid) == 0.0
    for axis in ("x", "y", "z"):
        assert _state_value_max_abs_diff(ref_vector[axis].real, got_vector[axis].real) < 1.0e-5
        assert _state_value_max_abs_diff(ref_vector[axis].imag, got_vector[axis].imag) < 1.0e-5
    assert _state_value_max_abs_diff(ref_endpoints["tx_pos"], got_endpoints["tx_pos"]) < 1.0e-5
    assert _state_value_max_abs_diff(ref_endpoints["first_hit"], got_endpoints["first_hit"]) < 1.0e-5
    assert _state_value_max_abs_diff(ref_endpoints["last_hit"], got_endpoints["last_hit"]) < 1.0e-5


@pytest.mark.gpu
def test_native_reflection_replay_to_target_jvp_matches_python_reference():
    if not native_extension_available():
        pytest.skip("native extension is not built in the current environment")

    case = _build_reflection_case()
    scene = case["scene"]
    detail = case["reflection_detail"]
    paths = case["source_paths_per_bounce"][0]
    edge_data = scene.get_edge_data(1.5)["edge_data"]
    n_paths = int(paths["n_paths"])
    n_edges = int(edge_data["n_edges"])
    pair_count = min(10, n_paths * n_edges)
    pair_idx = dr.arange(wt.UInt32, pair_count)
    path_idx = pair_idx % n_paths
    edge_idx = (pair_idx * 5) % n_edges
    target_faces = (
        dr.gather(wt.Int32, edge_data["adjacent_face0"], edge_idx),
        dr.gather(wt.Int32, edge_data["adjacent_face1"], edge_idx),
    )
    tangent = wt.Point3f(
        dr.linspace(wt.Float, 0.01, 0.03, pair_count),
        dr.linspace(wt.Float, -0.02, 0.01, pair_count),
        dr.linspace(wt.Float, 0.02, -0.01, pair_count),
    )

    ref_target = _detach_point3f(dr.gather(wt.Point3f, edge_data["pos"], edge_idx))
    _enable_grad_point3f(ref_target)
    _set_grad_point3f(ref_target, tangent)
    ref_valid, ref_vector, ref_endpoints = reflection_epc_module._epc_reflection_chain_to_target_python(
        paths=paths,
        path_idx=path_idx,
        target_pos=ref_target,
        scene=scene,
        target_adjacent_faces=target_faces,
        reflection_detail=detail,
        wavelength=case["wavelength"],
        tx_polarization=case["tx_polarization"],
        return_endpoints=True,
    )
    dr.forward_to(_reflection_replay_output_components(ref_vector, ref_endpoints), flags=FLAGS)
    ref_jvp = [dr.grad(component) for component in _reflection_replay_output_components(ref_vector, ref_endpoints)]

    got_target = _detach_point3f(dr.gather(wt.Point3f, edge_data["pos"], edge_idx))
    _enable_grad_point3f(got_target)
    _set_grad_point3f(got_target, tangent)
    got_valid, got_vector, got_endpoints = reflection_epc_module.epc_reflection_chain_to_target(
        paths=paths,
        path_idx=path_idx,
        target_pos=got_target,
        scene=scene,
        target_adjacent_faces=target_faces,
        reflection_detail=detail,
        wavelength=case["wavelength"],
        tx_polarization=case["tx_polarization"],
        return_endpoints=True,
    )
    dr.forward_to(_reflection_replay_output_components(got_vector, got_endpoints), flags=FLAGS)
    got_jvp = [dr.grad(component) for component in _reflection_replay_output_components(got_vector, got_endpoints)]

    assert _state_value_max_abs_diff(ref_valid, got_valid) == 0.0
    for ref_component, got_component in zip(ref_jvp, got_jvp):
        assert _max_abs_diff(ref_component, got_component) < 2.0e-5


@pytest.mark.gpu
def test_native_reflection_replay_to_target_backward_matches_python_reference():
    if not native_extension_available():
        pytest.skip("native extension is not built in the current environment")

    case = _build_reflection_case()
    scene = case["scene"]
    detail = case["reflection_detail"]
    paths = case["source_paths_per_bounce"][0]
    edge_data = scene.get_edge_data(1.5)["edge_data"]
    n_paths = int(paths["n_paths"])
    n_edges = int(edge_data["n_edges"])
    pair_count = min(10, n_paths * n_edges)
    pair_idx = dr.arange(wt.UInt32, pair_count)
    path_idx = pair_idx % n_paths
    edge_idx = (pair_idx * 7) % n_edges
    target_faces = (
        dr.gather(wt.Int32, edge_data["adjacent_face0"], edge_idx),
        dr.gather(wt.Int32, edge_data["adjacent_face1"], edge_idx),
    )

    ref_target = _detach_point3f(dr.gather(wt.Point3f, edge_data["pos"], edge_idx))
    _enable_grad_point3f(ref_target)
    _, ref_vector, ref_endpoints = reflection_epc_module._epc_reflection_chain_to_target_python(
        paths=paths,
        path_idx=path_idx,
        target_pos=ref_target,
        scene=scene,
        target_adjacent_faces=target_faces,
        reflection_detail=detail,
        wavelength=case["wavelength"],
        tx_polarization=case["tx_polarization"],
        return_endpoints=True,
    )
    dr.backward(_reflection_replay_loss(ref_vector, ref_endpoints), flags=FLAGS)
    ref_grad = _detach_point3f(dr.grad(ref_target))

    got_target = _detach_point3f(dr.gather(wt.Point3f, edge_data["pos"], edge_idx))
    _enable_grad_point3f(got_target)
    _, got_vector, got_endpoints = reflection_epc_module.epc_reflection_chain_to_target(
        paths=paths,
        path_idx=path_idx,
        target_pos=got_target,
        scene=scene,
        target_adjacent_faces=target_faces,
        reflection_detail=detail,
        wavelength=case["wavelength"],
        tx_polarization=case["tx_polarization"],
        return_endpoints=True,
    )
    dr.backward(_reflection_replay_loss(got_vector, got_endpoints), flags=FLAGS)
    got_grad = _detach_point3f(dr.grad(got_target))

    assert _max_abs_diff(ref_grad.x, got_grad.x) < 2.0e-5
    assert _max_abs_diff(ref_grad.y, got_grad.y) < 2.0e-5
    assert _max_abs_diff(ref_grad.z, got_grad.z) < 2.0e-5


@pytest.mark.gpu
def test_native_reflection_tiled_rollout_skips_ad_sensitive_inputs(monkeypatch):
    if not native_extension_available():
        pytest.skip("native extension is not built in the current environment")

    case = _build_reflection_case()
    paths = case["source_paths_per_bounce"][0]
    n_paths = int(paths.n_paths if hasattr(paths, "n_paths") else paths.get("n_paths", 0))
    if n_paths <= 0:
        pytest.skip("reflection case did not produce any replayable paths")

    rx_pos = _detach_point3f(case["rx_pos"])
    _enable_grad_point3f(rx_pos)

    def _unexpected_tile_plan(**kwargs):
        raise AssertionError("AD-sensitive reflection workloads must bypass the tiled primal rollout")

    monkeypatch.setattr(reflection_native, "build_reflection_family_tile_plan", _unexpected_tile_plan)

    tiled = reflection_native._accumulate_reflection_paths_tiled(
        paths=paths,
        rx_pos=rx_pos,
        scene=case["scene"],
        wavelength=case["wavelength"],
        k=case["k"],
        reflection_detail=case["reflection_detail"],
        tx_polarization=case["tx_polarization"],
        receiver_tiles=SimpleNamespace(n_tiles=2),
        default_eta_r=5.0,
        default_sigma=0.0,
        default_gain=float(case["reflection_detail"].reflection_gain),
    )

    assert tiled is None


@pytest.mark.gpu
def test_rayd_reflection_prefix_paths_preserve_geometry_descriptor_grads():
    chain = SimpleNamespace(
        ray_count=2,
        max_bounces=2,
        bounce_count=wt.UInt32([2, 2]),
        discovery_count=wt.UInt32([1, 1]),
        representative_ray_index=wt.UInt32([0, 1]),
        prim_ids=wt.Int32([10, 11, 20, 21]),
        image_sources=wt.Point3f(
            wt.Float([1.0, 2.0, 3.0, 4.0]),
            wt.Float([1.5, 2.5, 3.5, 4.5]),
            wt.Float([2.0, 3.0, 4.0, 5.0]),
        ),
        plane_points=wt.Point3f(
            wt.Float([10.0, 11.0, 12.0, 13.0]),
            wt.Float([20.0, 21.0, 22.0, 23.0]),
            wt.Float([30.0, 31.0, 32.0, 33.0]),
        ),
        plane_normals=wt.Vector3f(
            wt.Float([0.1, 0.2, 0.3, 0.4]),
            wt.Float([0.5, 0.6, 0.7, 0.8]),
            wt.Float([0.9, 1.0, 1.1, 1.2]),
        ),
        hit_points=wt.Point3f(
            wt.Float([40.0, 41.0, 42.0, 43.0]),
            wt.Float([50.0, 51.0, 52.0, 53.0]),
            wt.Float([60.0, 61.0, 62.0, 63.0]),
        ),
    )
    for array in (
        chain.image_sources.x, chain.image_sources.y, chain.image_sources.z,
        chain.plane_points.x, chain.plane_points.y, chain.plane_points.z,
        chain.plane_normals.x, chain.plane_normals.y, chain.plane_normals.z,
        chain.hit_points.x, chain.hit_points.y, chain.hit_points.z,
    ):
        dr.enable_grad(array)

    source_paths = reflection_paths_module._collect_reflection_prefix_paths_from_rayd_chain(
        chain,
        chain_depth=2,
    )
    depth1 = source_paths[0]
    depth2 = source_paths[1]

    loss = dr.zeros(wt.Float, 1)
    loss += dr.sum(depth1.image_source.x)
    loss += dr.sum(depth1.plane_point(0).y)
    loss += dr.sum(depth1.plane_normal(0).z)
    loss += dr.sum(depth1.hit_point(0).x)
    loss += dr.sum(depth2.image_source.z)
    loss += dr.sum(depth2.plane_point(1).x)
    loss += dr.sum(depth2.plane_normal(1).y)
    loss += dr.sum(depth2.hit_point(1).z)
    dr.backward(loss, flags=FLAGS)

    image_source_x_grad = dr.grad(chain.image_sources.x)
    image_source_z_grad = dr.grad(chain.image_sources.z)
    plane_point_y_grad = dr.grad(chain.plane_points.y)
    plane_point_x_grad = dr.grad(chain.plane_points.x)
    plane_normal_z_grad = dr.grad(chain.plane_normals.z)
    plane_normal_y_grad = dr.grad(chain.plane_normals.y)
    hit_point_x_grad = dr.grad(chain.hit_points.x)
    hit_point_z_grad = dr.grad(chain.hit_points.z)

    assert float(image_source_x_grad[0]) == pytest.approx(1.0)
    assert float(image_source_x_grad[2]) == pytest.approx(1.0)
    assert float(image_source_z_grad[1]) == pytest.approx(1.0)
    assert float(image_source_z_grad[3]) == pytest.approx(1.0)
    assert float(plane_point_y_grad[0]) == pytest.approx(1.0)
    assert float(plane_point_y_grad[2]) == pytest.approx(1.0)
    assert float(plane_point_x_grad[1]) == pytest.approx(1.0)
    assert float(plane_point_x_grad[3]) == pytest.approx(1.0)
    assert float(plane_normal_z_grad[0]) == pytest.approx(1.0)
    assert float(plane_normal_z_grad[2]) == pytest.approx(1.0)
    assert float(plane_normal_y_grad[1]) == pytest.approx(1.0)
    assert float(plane_normal_y_grad[3]) == pytest.approx(1.0)
    assert float(hit_point_x_grad[0]) == pytest.approx(1.0)
    assert float(hit_point_x_grad[2]) == pytest.approx(1.0)
    assert float(hit_point_z_grad[1]) == pytest.approx(1.0)
    assert float(hit_point_z_grad[3]) == pytest.approx(1.0)


def test_internal_native_module_does_not_export_pointer_bridge_helpers():
    assert not hasattr(internal_native_module, "device_ptr")
    assert not hasattr(internal_native_module, "float_ptr")
    assert not hasattr(internal_native_module, "int_ptr")
    assert "device_ptr" not in internal_native_module.__all__
    assert "float_ptr" not in internal_native_module.__all__
    assert "int_ptr" not in internal_native_module.__all__


@pytest.mark.gpu
def test_native_extension_does_not_expose_removed_pointer_bridge_entries():
    if not native_extension_available():
        pytest.skip("native extension is not built in the current environment")

    ext = internal_native_module._extension()
    assert not hasattr(ext, "batch_coplanarity_check_raw")
    assert not hasattr(ext, "batch_edge_geometry_raw")
    assert not hasattr(ext, "prune_state_arrays_by_budget_raw")
    assert not hasattr(ext, "prune_state_arrays_by_budget_pair_raw")
    assert hasattr(ext, "batch_coplanarity_check_arrays")
    assert hasattr(ext, "batch_edge_geometry_arrays")
    assert hasattr(ext, "prune_state_arrays_by_budget_arrays")
    assert hasattr(ext, "prune_state_arrays_by_budget_pair_arrays")


@pytest.mark.gpu
def test_native_utd_forward_matches_drjit_reference():
    if not native_extension_available():
        pytest.skip("native extension is not built in the current environment")

    case = _build_utd_case()
    ref = utd_drjit.utd_accumulate_forward(
        case["state_arrays"],
        case["rx_pos"],
        case["k"],
        case["n_edges"],
        False,
        wavelength=case["wavelength"],
    )
    got = utd_native.utd_accumulate_forward(
        case["state_arrays"],
        case["rx_pos"],
        case["k"],
        case["n_edges"],
        False,
        wavelength=case["wavelength"],
    )

    assert _max_abs_diff(ref[0].real, got[0].real) < 5.0e-5
    assert _max_abs_diff(ref[0].imag, got[0].imag) < 5.0e-5
    assert _max_abs_diff(ref[2]["x"].real, got[2]["x"].real) < 5.0e-5
    assert _max_abs_diff(ref[2]["x"].imag, got[2]["x"].imag) < 5.0e-5
    assert _max_abs_diff(ref[2]["y"].real, got[2]["y"].real) < 5.0e-5
    assert _max_abs_diff(ref[2]["y"].imag, got[2]["y"].imag) < 5.0e-5


@pytest.mark.gpu
def test_native_utd_requires_finite_edge_bounds():
    if not native_extension_available():
        pytest.skip("native extension is not built in the current environment")

    case = _build_utd_case()
    state = dict(case["state_arrays"])
    state.pop("edge_line_min", None)
    state.pop("edge_line_max", None)

    with pytest.raises(RuntimeError, match="edge_line_min and edge_line_max"):
        utd_native.utd_accumulate_forward(
            state,
            case["rx_pos"],
            case["k"],
            case["n_edges"],
            False,
            wavelength=case["wavelength"],
        )


@pytest.mark.gpu
def test_native_utd_jvp_geometry_matches_drjit_reference():
    if not native_extension_available():
        pytest.skip("native extension is not built in the current environment")

    case = _build_utd_case()
    state = case["state_arrays"]

    def _clone_state():
        cloned = dict(state)
        cloned["edge_pos"] = wt.Point3f(
            dr.detach(state["edge_pos"].x),
            dr.detach(state["edge_pos"].y),
            dr.detach(state["edge_pos"].z),
        )
        cloned["n0"] = wt.Vector3f(
            dr.detach(state["n0"].x),
            dr.detach(state["n0"].y),
            dr.detach(state["n0"].z),
        )
        cloned["n_face_n"] = wt.Vector3f(
            dr.detach(state["n_face_n"].x),
            dr.detach(state["n_face_n"].y),
            dr.detach(state["n_face_n"].z),
        )
        cloned["wedge_n"] = dr.detach(state["wedge_n"])
        cloned["source_pos"] = wt.Point3f(
            dr.detach(state["source_pos"].x),
            dr.detach(state["source_pos"].y),
            dr.detach(state["source_pos"].z),
        )
        cloned["r_face0"] = wt.Complex2f(
            dr.detach(state["r_face0"].real),
            dr.detach(state["r_face0"].imag),
        )
        cloned["r_face_n"] = wt.Complex2f(
            dr.detach(state["r_face_n"].real),
            dr.detach(state["r_face_n"].imag),
        )
        return cloned

    def _seed_tangent(test_state, rx_pos):
        n_states = int(test_state["n_states"])
        n_rx = dr.width(rx_pos.x)
        for arr in (
            test_state["edge_pos"].x, test_state["edge_pos"].y, test_state["edge_pos"].z,
            test_state["n0"].x, test_state["n0"].y, test_state["n0"].z,
            test_state["n_face_n"].x, test_state["n_face_n"].y, test_state["n_face_n"].z,
            test_state["wedge_n"],
            test_state["source_pos"].x, test_state["source_pos"].y, test_state["source_pos"].z,
            test_state["r_face0"].real, test_state["r_face0"].imag,
            test_state["r_face_n"].real, test_state["r_face_n"].imag,
            rx_pos.x, rx_pos.y, rx_pos.z,
        ):
            dr.enable_grad(arr)
        dr.set_grad(test_state["edge_pos"].x, dr.linspace(wt.Float, -0.05, 0.07, n_states))
        dr.set_grad(test_state["edge_pos"].y, dr.linspace(wt.Float, 0.02, -0.03, n_states))
        dr.set_grad(test_state["edge_pos"].z, dr.linspace(wt.Float, 0.01, 0.04, n_states))
        dr.set_grad(test_state["n0"].x, dr.linspace(wt.Float, -0.02, 0.03, n_states))
        dr.set_grad(test_state["n0"].y, dr.linspace(wt.Float, 0.01, -0.02, n_states))
        dr.set_grad(test_state["n0"].z, dr.linspace(wt.Float, 0.03, 0.00, n_states))
        dr.set_grad(test_state["n_face_n"].x, dr.linspace(wt.Float, 0.04, -0.01, n_states))
        dr.set_grad(test_state["n_face_n"].y, dr.linspace(wt.Float, -0.03, 0.02, n_states))
        dr.set_grad(test_state["n_face_n"].z, dr.linspace(wt.Float, 0.02, 0.05, n_states))
        dr.set_grad(test_state["wedge_n"], dr.linspace(wt.Float, -0.04, 0.03, n_states))
        dr.set_grad(test_state["source_pos"].x, dr.linspace(wt.Float, 0.06, -0.02, n_states))
        dr.set_grad(test_state["source_pos"].y, dr.linspace(wt.Float, -0.01, 0.05, n_states))
        dr.set_grad(test_state["source_pos"].z, dr.linspace(wt.Float, 0.02, 0.06, n_states))
        dr.set_grad(test_state["r_face0"].real, dr.linspace(wt.Float, -0.03, 0.02, n_states))
        dr.set_grad(test_state["r_face0"].imag, dr.linspace(wt.Float, 0.01, -0.02, n_states))
        dr.set_grad(test_state["r_face_n"].real, dr.linspace(wt.Float, 0.02, -0.01, n_states))
        dr.set_grad(test_state["r_face_n"].imag, dr.linspace(wt.Float, -0.02, 0.03, n_states))
        dr.set_grad(rx_pos.x, dr.linspace(wt.Float, 0.03, -0.01, n_rx))
        dr.set_grad(rx_pos.y, dr.linspace(wt.Float, -0.02, 0.04, n_rx))
        dr.set_grad(rx_pos.z, dr.linspace(wt.Float, 0.01, 0.05, n_rx))

    n_rx = dr.width(case["rx_pos"].x)
    grad_direct_total = wt.Complex2f(
        dr.linspace(wt.Float, 0.1, 0.4, n_rx),
        dr.linspace(wt.Float, -0.2, 0.2, n_rx),
    )
    grad_multi_total = wt.Complex2f(dr.zeros(wt.Float, n_rx), dr.zeros(wt.Float, n_rx))
    grad_direct_vector = {
        "x": wt.Complex2f(dr.linspace(wt.Float, 0.3, 0.6, n_rx), dr.linspace(wt.Float, -0.1, 0.1, n_rx)),
        "y": wt.Complex2f(dr.linspace(wt.Float, -0.4, 0.2, n_rx), dr.linspace(wt.Float, 0.2, -0.2, n_rx)),
        "z": wt.Complex2f(dr.linspace(wt.Float, 0.05, -0.05, n_rx), dr.linspace(wt.Float, 0.07, -0.03, n_rx)),
    }
    grad_multi_vector = {
        axis: wt.Complex2f(dr.zeros(wt.Float, n_rx), dr.zeros(wt.Float, n_rx))
        for axis in ("x", "y", "z")
    }

    ref_state = _clone_state()
    ref_rx = wt.Point3f(
        dr.detach(case["rx_pos"].x),
        dr.detach(case["rx_pos"].y),
        dr.detach(case["rx_pos"].z),
    )
    _seed_tangent(ref_state, ref_rx)
    ref_outputs = utd_drjit.utd_accumulate_forward(
        ref_state,
        ref_rx,
        case["k"],
        case["n_edges"],
        False,
        wavelength=case["wavelength"],
    )
    ref_loss = _utd_loss(
        ref_outputs,
        grad_direct_total,
        grad_multi_total,
        grad_direct_vector,
        grad_multi_vector,
    )
    dr.forward_to(ref_loss, flags=FLAGS)
    ref_jvp = float(dr.grad(ref_loss)[0])

    native_state = _clone_state()
    native_rx = wt.Point3f(
        dr.detach(case["rx_pos"].x),
        dr.detach(case["rx_pos"].y),
        dr.detach(case["rx_pos"].z),
    )
    _seed_tangent(native_state, native_rx)
    native_outputs = utd_native.utd_accumulate_forward(
        native_state,
        native_rx,
        case["k"],
        case["n_edges"],
        False,
        wavelength=case["wavelength"],
    )
    native_loss = _utd_loss(
        native_outputs,
        grad_direct_total,
        grad_multi_total,
        grad_direct_vector,
        grad_multi_vector,
    )
    dr.forward_to(native_loss, flags=FLAGS)
    native_jvp = float(dr.grad(native_loss)[0])

    assert math.isfinite(ref_jvp)
    assert math.isfinite(native_jvp)
    assert abs(ref_jvp - native_jvp) < 2.0e-5


@pytest.mark.gpu
def test_native_utd_finite_wedge_backward_accumulates_input_grads():
    if not native_extension_available():
        pytest.skip("native extension is not built in the current environment")

    case = _build_utd_case()
    state = dict(case["state_arrays"])
    state["edge_pos"] = wt.Point3f(
        dr.detach(state["edge_pos"].x),
        dr.detach(state["edge_pos"].y),
        dr.detach(state["edge_pos"].z),
    )
    state["n0"] = wt.Vector3f(
        dr.detach(state["n0"].x),
        dr.detach(state["n0"].y),
        dr.detach(state["n0"].z),
    )
    state["n_face_n"] = wt.Vector3f(
        dr.detach(state["n_face_n"].x),
        dr.detach(state["n_face_n"].y),
        dr.detach(state["n_face_n"].z),
    )
    state["wedge_n"] = dr.detach(state["wedge_n"])
    state["source_pos"] = wt.Point3f(
        dr.detach(state["source_pos"].x),
        dr.detach(state["source_pos"].y),
        dr.detach(state["source_pos"].z),
    )
    rx_pos = wt.Point3f(
        dr.detach(case["rx_pos"].x),
        dr.detach(case["rx_pos"].y),
        dr.detach(case["rx_pos"].z),
    )
    for arr in (
        state["edge_pos"].x, state["edge_pos"].y, state["edge_pos"].z,
        state["n0"].x, state["n0"].y, state["n0"].z,
        state["n_face_n"].x, state["n_face_n"].y, state["n_face_n"].z,
        state["wedge_n"],
        state["source_pos"].x, state["source_pos"].y, state["source_pos"].z,
        rx_pos.x, rx_pos.y, rx_pos.z,
    ):
        dr.enable_grad(arr)

    outputs = utd_native.utd_accumulate_forward(
        state,
        rx_pos,
        case["k"],
        case["n_edges"],
        False,
        wavelength=case["wavelength"],
    )
    n_rx = dr.width(rx_pos.x)
    weights0 = dr.linspace(wt.Float, 0.1, 0.3, n_rx)
    weights1 = dr.linspace(wt.Float, -0.2, 0.2, n_rx)
    weights2 = dr.linspace(wt.Float, 0.05, 0.15, n_rx)
    loss = dr.sum(outputs[0].real * weights0)
    loss += dr.sum(outputs[2]["x"].real * weights1)
    loss += dr.sum(outputs[3]["y"].real * weights2)
    dr.backward(loss, flags=FLAGS)

    edge_grad = dr.slice(dr.sum(dr.abs(dr.grad(state["edge_pos"].x))))
    rx_grad = dr.slice(dr.sum(dr.abs(dr.grad(rx_pos.x))))
    assert math.isfinite(edge_grad)
    assert math.isfinite(rx_grad)
    assert edge_grad > 0.0
    assert rx_grad > 0.0


@pytest.mark.gpu
def test_native_packed_state_gather_matches_drjit_reference():
    if not native_extension_available():
        pytest.skip("native extension is not built in the current environment")

    case = _build_utd_case()
    state_arrays = case["state_arrays"]
    n_states = state_arrays["n_states"]
    indices = wt.UInt32([n_states - 1, 0, min(3, n_states - 1), 0])

    ref = packed_state_drjit.gather_state_arrays(state_arrays, indices)
    got = packed_state_native.gather_state_arrays(state_arrays, indices)

    _assert_state_arrays_match(ref, got)


@pytest.mark.gpu
def test_native_packed_state_concat_and_subset_match_drjit_reference():
    if not native_extension_available():
        pytest.skip("native extension is not built in the current environment")

    case = _build_utd_case()
    state_arrays = case["state_arrays"]
    n_states = state_arrays["n_states"]

    left_idx = wt.UInt32([0, min(2, n_states - 1), min(4, n_states - 1)])
    right_idx = wt.UInt32([n_states - 1, min(1, n_states - 1)])
    left = packed_state_drjit.gather_state_arrays(state_arrays, left_idx)
    right = packed_state_drjit.gather_state_arrays(state_arrays, right_idx)

    ref_concat = packed_state_drjit.concat_state_arrays([left, right])
    got_concat = packed_state_native.concat_state_arrays([left, right])
    _assert_state_arrays_match(ref_concat, got_concat)

    mask = (dr.arange(wt.UInt32, n_states) % 3) != 1
    ref_subset = packed_state_drjit.subset_state_arrays(state_arrays, mask)
    got_subset = packed_state_native.subset_state_arrays(state_arrays, mask)
    _assert_state_arrays_match(ref_subset, got_subset)


@pytest.mark.gpu
def test_native_packed_state_inserted_reflection_field_gather_matches_reference():
    if not native_extension_available():
        pytest.skip("native extension is not built in the current environment")

    case = _build_utd_case()
    state_arrays = packed_state_drjit.concat_state_arrays([case["state_arrays"]] * 2)
    n_states = int(state_arrays["n_states"])
    indices = wt.UInt32([n_states - 1, 0, min(3, n_states - 1), min(5, n_states - 1), 1])

    ref_fields = packed_state_drjit.gather_inserted_reflection_state_fields(state_arrays, indices)
    got_fields = packed_state_native.gather_inserted_reflection_state_fields(state_arrays, indices)
    _assert_inserted_reflection_fields_match(ref_fields, got_fields)


@pytest.mark.gpu
def test_native_packed_state_field_eval_gather_matches_reference():
    if not native_extension_available():
        pytest.skip("native extension is not built in the current environment")

    case = _build_utd_case()
    state_arrays = packed_state_drjit.concat_state_arrays([case["state_arrays"]] * 2)
    n_states = int(state_arrays["n_states"])
    indices = wt.UInt32([n_states - 1, 0, min(3, n_states - 1), min(5, n_states - 1), 1])

    full_fields = packed_state_drjit.gather_state_arrays(state_arrays, indices)
    ref_fields = packed_state_drjit.gather_field_evaluation_state_fields(state_arrays, indices)
    got_fields = packed_state_native.gather_field_evaluation_state_fields(state_arrays, indices)

    _assert_field_eval_state_fields_match(full_fields, ref_fields)
    _assert_field_eval_state_fields_match(ref_fields, got_fields)


def test_suffix_field_state_gather_uses_hot_field_eval_fields_for_full_state(monkeypatch):
    case = _build_utd_case()
    indices = wt.UInt32([0, 1])
    sentinel = {"n_states": 2, "sentinel": True}

    monkeypatch.setattr(
        suffix_module,
        "gather_field_evaluation_state_fields",
        lambda *args, **kwargs: sentinel,
    )

    got = suffix_module._gather_suffix_field_state_fields(case["state_arrays"], indices)

    assert got is sentinel


def test_suffix_field_state_gather_uses_reduced_hot_fields_for_path_export_state(monkeypatch):
    case = _build_utd_case()
    reduced = reduce_state_arrays_for_path_export(case["state_arrays"])
    indices = wt.UInt32([0, 1])
    sentinel = {"n_states": 2, "sentinel": "reduced"}

    monkeypatch.setattr(
        suffix_module,
        "gather_path_export_field_state_fields",
        lambda *args, **kwargs: sentinel,
    )

    got = suffix_module._gather_suffix_field_state_fields(reduced, indices)

    assert got is sentinel


@pytest.mark.gpu
def test_native_packed_state_inserted_reflection_field_gather_ad_avoids_drjit_fallback(monkeypatch):
    if not native_extension_available():
        pytest.skip("native extension is not built in the current environment")

    case = _build_utd_case()
    state_arrays = case["state_arrays"]
    n_states = int(state_arrays["n_states"])
    indices = wt.UInt32([n_states - 1, 0, min(3, n_states - 1), 0])
    tangent = _packed_state_tangent(n_states)

    def _fail_fallback(*args, **kwargs):
        raise AssertionError("AD inserted-field gather unexpectedly used the Dr.Jit fallback")

    monkeypatch.setattr(packed_state_native, "_drjit_gather_inserted_fields", _fail_fallback)

    ref_state = _clone_packed_state_ad_inputs(state_arrays)
    _enable_grad_packed_state_inputs(ref_state)
    _set_grad_packed_state_inputs(ref_state, tangent)
    ref_fields = packed_state_drjit.gather_inserted_reflection_state_fields(ref_state, indices)
    ref_loss = (
        dr.sum(ref_fields["edge_pos"].x)
        + dr.sum(ref_fields["first_interaction_pos"].x)
        + dr.sum(ref_fields["path_length_prefix"])
    )
    dr.forward_to(ref_loss, flags=FLAGS)
    ref_directional = float(dr.grad(ref_loss)[0])

    native_state = _clone_packed_state_ad_inputs(state_arrays)
    _enable_grad_packed_state_inputs(native_state)
    _set_grad_packed_state_inputs(native_state, tangent)
    native_fields = packed_state_native.gather_inserted_reflection_state_fields(
        native_state,
        indices,
    )
    native_loss = (
        dr.sum(native_fields["edge_pos"].x)
        + dr.sum(native_fields["first_interaction_pos"].x)
        + dr.sum(native_fields["path_length_prefix"])
    )
    dr.forward_to(native_loss, flags=FLAGS)
    native_directional = float(dr.grad(native_loss)[0])

    assert math.isfinite(ref_directional)
    assert math.isfinite(native_directional)
    assert abs(ref_directional - native_directional) < 1.0e-5


@pytest.mark.gpu
def test_native_diffraction_path_slot_builder_matches_drjit_reference():
    if not native_extension_available():
        pytest.skip("native extension is not built in the current environment")

    keep_states = _build_pruning_tie_case()
    edge_count = 12
    edge_data = {
        "n_edges": edge_count,
        "pos": wt.Point3f(
            wt.Float([float(idx) for idx in range(edge_count)]),
            wt.Float([float(idx) * 0.1 for idx in range(edge_count)]),
            wt.Float([1.0 + float(idx) * 0.05 for idx in range(edge_count)]),
        ),
        "n0": wt.Vector3f(
            wt.Float([0.0] * edge_count),
            wt.Float([1.0] * edge_count),
            wt.Float([0.0] * edge_count),
        ),
    }
    edge_object_idx = wt.Int32([idx + 10 for idx in range(edge_count)])

    ref = packed_state_drjit.build_diffraction_path_slots(
        keep_states=keep_states,
        edge_data=edge_data,
        edge_object_idx=edge_object_idx,
        return_geometry=True,
    )
    got = packed_state_native.build_diffraction_path_slots(
        keep_states=keep_states,
        edge_data=edge_data,
        edge_object_idx=edge_object_idx,
        return_geometry=True,
    )

    _assert_path_slot_build_match(ref, got)


@pytest.mark.gpu
def test_native_packed_state_gather_jvp_matches_drjit_reference():
    if not native_extension_available():
        pytest.skip("native extension is not built in the current environment")

    case = _build_utd_case()
    state_arrays = case["state_arrays"]
    n_states = int(state_arrays["n_states"])
    indices = wt.UInt32([n_states - 1, 0, min(3, n_states - 1), 0])
    tangent = _packed_state_tangent(n_states)

    ref_state = _clone_packed_state_ad_inputs(state_arrays)
    _enable_grad_packed_state_inputs(ref_state)
    _set_grad_packed_state_inputs(ref_state, tangent)
    ref_gathered = packed_state_drjit.gather_state_arrays(ref_state, indices)
    ref_loss = _packed_state_loss(ref_gathered)
    dr.forward_to(ref_loss, flags=FLAGS)
    ref_directional = float(dr.grad(ref_loss)[0])

    native_state = _clone_packed_state_ad_inputs(state_arrays)
    _enable_grad_packed_state_inputs(native_state)
    _set_grad_packed_state_inputs(native_state, tangent)
    native_gathered = packed_state_native.gather_state_arrays(native_state, indices)
    native_loss = _packed_state_loss(native_gathered)
    dr.forward_to(native_loss, flags=FLAGS)
    native_directional = float(dr.grad(native_loss)[0])

    assert math.isfinite(ref_directional)
    assert math.isfinite(native_directional)
    assert abs(ref_directional - native_directional) < 1.0e-5


@pytest.mark.gpu
def test_native_packed_state_gather_ad_avoids_drjit_fallback(monkeypatch):
    if not native_extension_available():
        pytest.skip("native extension is not built in the current environment")

    case = _build_utd_case()
    state_arrays = case["state_arrays"]
    n_states = int(state_arrays["n_states"])
    indices = wt.UInt32([n_states - 1, 0, min(3, n_states - 1), 0])
    tangent = _packed_state_tangent(n_states)

    def _fail_fallback(*args, **kwargs):
        raise AssertionError("AD gather unexpectedly used the Dr.Jit fallback")

    monkeypatch.setattr(packed_state_native, "_drjit_gather", _fail_fallback)

    native_state = _clone_packed_state_ad_inputs(state_arrays)
    _enable_grad_packed_state_inputs(native_state)
    _set_grad_packed_state_inputs(native_state, tangent)
    native_gathered = packed_state_native.gather_state_arrays(native_state, indices)
    native_loss = _packed_state_loss(native_gathered)
    dr.forward_to(native_loss, flags=FLAGS)
    native_directional = float(dr.grad(native_loss)[0])

    assert math.isfinite(native_directional)


@pytest.mark.gpu
def test_native_packed_state_subset_backward_matches_drjit_reference():
    if not native_extension_available():
        pytest.skip("native extension is not built in the current environment")

    case = _build_utd_case()
    state_arrays = case["state_arrays"]
    n_states = int(state_arrays["n_states"])
    mask = (dr.arange(wt.UInt32, n_states) % 3) != 1

    ref_state = _clone_packed_state_ad_inputs(state_arrays)
    _enable_grad_packed_state_inputs(ref_state)
    ref_subset = packed_state_drjit.subset_state_arrays(ref_state, mask)
    dr.backward(_packed_state_loss(ref_subset), flags=FLAGS)
    ref_grads = _packed_state_input_grads(ref_state)

    native_state = _clone_packed_state_ad_inputs(state_arrays)
    _enable_grad_packed_state_inputs(native_state)
    native_subset = packed_state_native.subset_state_arrays(native_state, mask)
    dr.backward(_packed_state_loss(native_subset), flags=FLAGS)
    native_grads = _packed_state_input_grads(native_state)

    for key in ref_grads:
        assert _state_value_max_abs_diff(ref_grads[key], native_grads[key]) < 1.0e-5, key


@pytest.mark.gpu
def test_native_packed_state_subset_ad_avoids_drjit_fallback(monkeypatch):
    if not native_extension_available():
        pytest.skip("native extension is not built in the current environment")

    case = _build_utd_case()
    state_arrays = case["state_arrays"]
    n_states = int(state_arrays["n_states"])
    mask = (dr.arange(wt.UInt32, n_states) % 3) != 1

    def _fail_fallback(*args, **kwargs):
        raise AssertionError("AD subset unexpectedly used the Dr.Jit fallback")

    monkeypatch.setattr(packed_state_native, "_drjit_subset", _fail_fallback)

    native_state = _clone_packed_state_ad_inputs(state_arrays)
    _enable_grad_packed_state_inputs(native_state)
    native_subset = packed_state_native.subset_state_arrays(native_state, mask)
    dr.backward(_packed_state_loss(native_subset), flags=FLAGS)
    native_grads = _packed_state_input_grads(native_state)

    assert bool(dr.all(dr.isfinite(native_grads["wedge_n"])))


@pytest.mark.gpu
def test_native_edge_geometry_matches_drjit_reference():
    if not native_extension_available():
        pytest.skip("native extension is not built in the current environment")

    case = _build_mesh_kernel_case()
    ref = edge_geometry_drjit.batch_edge_geometry(
        case["vertices"],
        case["face_normals"],
        case["edge_v0"],
        case["edge_v1"],
        case["edge_face0"],
        case["edge_face1"],
        case["n_edges"],
    )
    got = edge_geometry_native.batch_edge_geometry(
        case["vertices"],
        case["face_normals"],
        case["edge_v0"],
        case["edge_v1"],
        case["edge_face0"],
        case["edge_face1"],
        case["n_edges"],
    )

    for key in ("pos", "edge_dir", "n0", "nn", "wedge_n", "length"):
        assert _state_value_max_abs_diff(ref[key], got[key]) < 1.0e-6, key


@pytest.mark.gpu
def test_native_coplanarity_matches_drjit_reference():
    if not native_extension_available():
        pytest.skip("native extension is not built in the current environment")

    case = _build_mesh_kernel_case()
    ref = coplanarity_drjit.batch_coplanarity_check(
        case["face_normals"],
        case["edge_face0"],
        case["edge_face1"],
        case["vertices"],
        case["faces"],
        case["n_edges"],
    )
    got = coplanarity_native.batch_coplanarity_check(
        case["face_normals"],
        case["edge_face0"],
        case["edge_face1"],
        case["vertices"],
        case["faces"],
        case["n_edges"],
    )

    assert _torch_max_abs_diff(ref, got) == 0.0


@pytest.mark.gpu
def test_native_cartesian_filter_matches_drjit_reference():
    if not native_extension_available():
        pytest.skip("native extension is not built in the current environment")

    case = _build_cartesian_filter_case()
    ref = cartesian_filter_drjit.cartesian_filter_bruteforce(**case)
    got = cartesian_filter_native.cartesian_filter_bruteforce(**case)

    assert _torch_max_abs_diff(ref[0], got[0]) == 0.0
    assert _torch_max_abs_diff(ref[1], got[1]) == 0.0


@pytest.mark.gpu
def test_native_cartesian_filter_multiblock_order_matches_drjit_reference():
    if not native_extension_available():
        pytest.skip("native extension is not built in the current environment")

    case = _build_cartesian_filter_multiblock_case()
    ref = cartesian_filter_drjit.cartesian_filter_bruteforce(**case)
    got = cartesian_filter_native.cartesian_filter_bruteforce(**case)

    assert _torch_max_abs_diff(ref[0], got[0]) == 0.0
    assert _torch_max_abs_diff(ref[1], got[1]) == 0.0


@pytest.mark.gpu
def test_native_cartesian_pair_dedup_restores_canonical_order():
    if not native_extension_available():
        pytest.skip("native extension is not built in the current environment")

    prev_idx = wt.UInt32(torch.tensor([3, 1, 3, 0, 1, 0, 3, 1], device="cuda", dtype=torch.int32))
    edge_idx = wt.UInt32(torch.tensor([2, 4, 2, 1, 2, 1, 0, 2], device="cuda", dtype=torch.int32))

    got_prev, got_edge = cartesian_filter_native.deduplicate_cartesian_pairs(
        prev_idx,
        edge_idx,
        n_edges=8,
    )

    ref_prev = wt.UInt32(torch.tensor([0, 1, 1, 3, 3], device="cuda", dtype=torch.int32))
    ref_edge = wt.UInt32(torch.tensor([1, 2, 4, 0, 2], device="cuda", dtype=torch.int32))

    assert _torch_max_abs_diff(got_prev, ref_prev) == 0.0
    assert _torch_max_abs_diff(got_edge, ref_edge) == 0.0


@pytest.mark.gpu
def test_native_compact_index_pairs_matches_drjit_reference():
    if not native_extension_available():
        pytest.skip("native extension is not built in the current environment")

    case = _build_compact_index_pairs_case()
    ref_lhs, ref_rhs = cartesian_filter_drjit.compact_index_pairs(**case)
    got_lhs, got_rhs = cartesian_filter_native.compact_index_pairs(**case)

    assert _torch_max_abs_diff(ref_lhs, got_lhs) == 0.0
    assert _torch_max_abs_diff(ref_rhs, got_rhs) == 0.0


@pytest.mark.gpu
def test_native_pruning_sort_matches_drjit_reference():
    if not native_extension_available():
        pytest.skip("native extension is not built in the current environment")

    case = _build_utd_case()
    state_arrays = packed_state_drjit.concat_state_arrays([case["state_arrays"]] * 3)
    budget = max(1, int(state_arrays["n_states"]) // 2)

    ref_state, ref_report = pruning_sort_drjit.prune_state_arrays_by_budget(
        state_arrays,
        budget,
        "test_budget",
    )
    got_state, got_report = pruning_sort_native.prune_state_arrays_by_budget(
        state_arrays,
        budget,
        "test_budget",
    )

    _assert_state_arrays_match(ref_state, got_state)
    assert ref_report == got_report


@pytest.mark.gpu
def test_native_pruning_sort_history_tiebreak_matches_drjit_reference():
    if not native_extension_available():
        pytest.skip("native extension is not built in the current environment")

    state_arrays = _build_pruning_tie_case()
    ref_state, _ = pruning_sort_drjit.prune_state_arrays_by_budget(state_arrays, 1, "tie_case")
    got_state, _ = pruning_sort_native.prune_state_arrays_by_budget(state_arrays, 1, "tie_case")

    _assert_state_arrays_match(ref_state, got_state)
    assert _torch_max_abs_diff(ref_state["edge_idx"], wt.UInt32([100])) == 0.0


@pytest.mark.gpu
def test_native_pruning_sort_pair_matches_drjit_reference():
    if not native_extension_available():
        pytest.skip("native extension is not built in the current environment")

    case = _build_utd_case()
    state_arrays = packed_state_drjit.concat_state_arrays([case["state_arrays"]] * 3)
    n_states = int(state_arrays["n_states"])
    higher_budget = max(1, n_states // 3)
    inserted_budget = max(1, n_states // 2)

    (
        ref_higher_state,
        ref_higher_report,
        ref_inserted_state,
        ref_inserted_report,
    ) = pruning_sort_drjit.prune_state_arrays_by_budget_pair(
        state_arrays,
        higher_budget,
        inserted_budget,
        higher_budget_name="higher_budget",
        inserted_budget_name="inserted_budget",
    )
    (
        got_higher_state,
        got_higher_report,
        got_inserted_state,
        got_inserted_report,
    ) = pruning_sort_native.prune_state_arrays_by_budget_pair(
        state_arrays,
        higher_budget,
        inserted_budget,
        higher_budget_name="higher_budget",
        inserted_budget_name="inserted_budget",
    )

    _assert_state_arrays_match(ref_higher_state, got_higher_state)
    _assert_state_arrays_match(ref_inserted_state, got_inserted_state)
    assert ref_higher_report == got_higher_report
    assert ref_inserted_report == got_inserted_report


@pytest.mark.gpu
def test_native_utd_backward_geometry_matches_drjit_reference():
    if not native_extension_available():
        pytest.skip("native extension is not built in the current environment")

    case = _build_utd_case()
    state = case["state_arrays"]
    edge_pos = wt.Point3f(
        dr.detach(state["edge_pos"].x),
        dr.detach(state["edge_pos"].y),
        dr.detach(state["edge_pos"].z),
    )
    n0 = wt.Vector3f(
        dr.detach(state["n0"].x),
        dr.detach(state["n0"].y),
        dr.detach(state["n0"].z),
    )
    nn = wt.Vector3f(
        dr.detach(state["n_face_n"].x),
        dr.detach(state["n_face_n"].y),
        dr.detach(state["n_face_n"].z),
    )
    wedge_n = dr.detach(state["wedge_n"])
    source_pos = wt.Point3f(
        dr.detach(state["source_pos"].x),
        dr.detach(state["source_pos"].y),
        dr.detach(state["source_pos"].z),
    )
    rx_pos = wt.Point3f(
        dr.detach(case["rx_pos"].x),
        dr.detach(case["rx_pos"].y),
        dr.detach(case["rx_pos"].z),
    )
    for arr in (
        edge_pos.x, edge_pos.y, edge_pos.z,
        n0.x, n0.y, n0.z,
        nn.x, nn.y, nn.z,
        wedge_n,
        source_pos.x, source_pos.y, source_pos.z,
        rx_pos.x, rx_pos.y, rx_pos.z,
    ):
        dr.enable_grad(arr)

    ref_state = dict(state)
    ref_state["edge_pos"] = edge_pos
    ref_state["n0"] = n0
    ref_state["n_face_n"] = nn
    ref_state["wedge_n"] = wedge_n
    ref_state["source_pos"] = source_pos

    n_rx = dr.width(case["rx_pos"].x)
    grad_direct_total = wt.Complex2f(
        dr.linspace(wt.Float, 0.1, 0.4, n_rx),
        dr.linspace(wt.Float, -0.2, 0.2, n_rx),
    )
    grad_multi_total = wt.Complex2f(dr.zeros(wt.Float, n_rx), dr.zeros(wt.Float, n_rx))
    grad_direct_vector = {
        "x": wt.Complex2f(dr.linspace(wt.Float, 0.3, 0.6, n_rx), dr.linspace(wt.Float, -0.1, 0.1, n_rx)),
        "y": wt.Complex2f(dr.linspace(wt.Float, -0.4, 0.2, n_rx), dr.linspace(wt.Float, 0.2, -0.2, n_rx)),
        "z": wt.Complex2f(dr.linspace(wt.Float, 0.05, -0.05, n_rx), dr.linspace(wt.Float, 0.07, -0.03, n_rx)),
    }
    grad_multi_vector = {
        axis: wt.Complex2f(dr.zeros(wt.Float, n_rx), dr.zeros(wt.Float, n_rx))
        for axis in ("x", "y", "z")
    }

    ref_outputs = utd_drjit.utd_accumulate_forward(
        ref_state,
        rx_pos,
        case["k"],
        case["n_edges"],
        False,
        wavelength=case["wavelength"],
    )
    dr.backward(
        _utd_loss(
            ref_outputs,
            grad_direct_total,
            grad_multi_total,
            grad_direct_vector,
            grad_multi_vector,
        )
    )

    state_grads, rx_grads = utd_native.utd_accumulate_backward(
        case["state_arrays"],
        case["rx_pos"],
        case["k"],
        case["n_edges"],
        False,
        grad_direct_total,
        grad_multi_total,
        grad_direct_vector,
        grad_multi_vector,
        wavelength=case["wavelength"],
    )

    assert state_grads["n_states"] == case["state_arrays"]["n_states"]
    assert _max_abs_diff(state_grads["edge_pos"].x, dr.grad(edge_pos.x)) < 1.0e-6
    assert _max_abs_diff(state_grads["edge_pos"].y, dr.grad(edge_pos.y)) < 1.0e-6
    assert _max_abs_diff(state_grads["n0"].x, dr.grad(n0.x)) < 1.0e-6
    assert _max_abs_diff(state_grads["n0"].y, dr.grad(n0.y)) < 1.0e-6
    assert _max_abs_diff(state_grads["n0"].z, dr.grad(n0.z)) < 1.0e-6
    assert _max_abs_diff(state_grads["n_face_n"].x, dr.grad(nn.x)) < 1.0e-6
    assert _max_abs_diff(state_grads["n_face_n"].y, dr.grad(nn.y)) < 1.0e-6
    assert _max_abs_diff(state_grads["n_face_n"].z, dr.grad(nn.z)) < 1.0e-6
    assert _max_abs_diff(state_grads["wedge_n"], dr.grad(wedge_n)) < 1.0e-6
    assert _max_abs_diff(state_grads["source_pos"].x, dr.grad(source_pos.x)) < 1.0e-6
    assert _max_abs_diff(rx_grads.x, dr.grad(rx_pos.x)) < 1.0e-6
    assert _max_abs_diff(rx_grads.y, dr.grad(rx_pos.y)) < 1.0e-6


@pytest.mark.gpu
def test_native_utd_backward_face_material_matches_drjit_reference():
    if not native_extension_available():
        pytest.skip("native extension is not built in the current environment")

    case = _build_utd_case()
    state = case["state_arrays"]
    rx_pos = wt.Point3f(
        dr.detach(case["rx_pos"].x),
        dr.detach(case["rx_pos"].y),
        dr.detach(case["rx_pos"].z),
    )
    ref_state = dict(state)
    ref_state["face0_eta_r"] = dr.detach(state["face0_eta_r"])
    ref_state["face0_sigma"] = dr.detach(state["face0_sigma"])
    ref_state["face0_gain"] = dr.detach(state["face0_gain"])
    ref_state["face1_eta_r"] = dr.detach(state["face1_eta_r"])
    ref_state["face1_sigma"] = dr.detach(state["face1_sigma"])
    ref_state["face1_gain"] = dr.detach(state["face1_gain"])
    for arr in (
        ref_state["face0_eta_r"],
        ref_state["face0_sigma"],
        ref_state["face0_gain"],
        ref_state["face1_eta_r"],
        ref_state["face1_sigma"],
        ref_state["face1_gain"],
    ):
        dr.enable_grad(arr)

    n_rx = dr.width(case["rx_pos"].x)
    grad_direct_total = wt.Complex2f(
        dr.linspace(wt.Float, 0.1, 0.4, n_rx),
        dr.linspace(wt.Float, -0.2, 0.2, n_rx),
    )
    grad_multi_total = wt.Complex2f(dr.zeros(wt.Float, n_rx), dr.zeros(wt.Float, n_rx))
    grad_direct_vector = {
        "x": wt.Complex2f(dr.linspace(wt.Float, 0.3, 0.6, n_rx), dr.linspace(wt.Float, -0.1, 0.1, n_rx)),
        "y": wt.Complex2f(dr.linspace(wt.Float, -0.4, 0.2, n_rx), dr.linspace(wt.Float, 0.2, -0.2, n_rx)),
        "z": wt.Complex2f(dr.linspace(wt.Float, 0.05, -0.05, n_rx), dr.linspace(wt.Float, 0.07, -0.03, n_rx)),
    }
    grad_multi_vector = {
        axis: wt.Complex2f(dr.zeros(wt.Float, n_rx), dr.zeros(wt.Float, n_rx))
        for axis in ("x", "y", "z")
    }

    ref_outputs = utd_drjit.utd_accumulate_forward(
        ref_state,
        rx_pos,
        case["k"],
        case["n_edges"],
        False,
        wavelength=case["wavelength"],
    )
    dr.backward(
        _utd_loss(
            ref_outputs,
            grad_direct_total,
            grad_multi_total,
            grad_direct_vector,
            grad_multi_vector,
        )
    )

    state_grads, _ = utd_native.utd_accumulate_backward(
        state,
        case["rx_pos"],
        case["k"],
        case["n_edges"],
        False,
        grad_direct_total,
        grad_multi_total,
        grad_direct_vector,
        grad_multi_vector,
        wavelength=case["wavelength"],
    )

    assert _max_abs_diff(state_grads["face0_eta_r"], dr.grad(ref_state["face0_eta_r"])) < 1.0e-6
    assert _max_abs_diff(state_grads["face0_sigma"], dr.grad(ref_state["face0_sigma"])) < 1.0e-6
    assert _max_abs_diff(state_grads["face0_gain"], dr.grad(ref_state["face0_gain"])) < 1.0e-6
    assert _max_abs_diff(state_grads["face1_eta_r"], dr.grad(ref_state["face1_eta_r"])) < 1.0e-6
    assert _max_abs_diff(state_grads["face1_sigma"], dr.grad(ref_state["face1_sigma"])) < 1.0e-6
    assert _max_abs_diff(state_grads["face1_gain"], dr.grad(ref_state["face1_gain"])) < 1.0e-6


@pytest.mark.gpu
def test_native_utd_forward_stored_face_operator_matches_drjit_reference():
    if not native_extension_available():
        pytest.skip("native extension is not built in the current environment")

    case = _build_utd_case()
    state = dict(case["state_arrays"])
    n_states = int(state["n_states"])

    def _complex_linspace(re0, re1, im0, im1):
        return wt.Complex2f(
            dr.linspace(wt.Float, re0, re1, n_states),
            dr.linspace(wt.Float, im0, im1, n_states),
        )

    state["face0_operator_m00"] = _complex_linspace(0.30, 0.60, -0.20, 0.10)
    state["face0_operator_m01"] = _complex_linspace(-0.40, 0.20, 0.10, -0.10)
    state["face0_operator_m10"] = _complex_linspace(0.05, -0.15, -0.05, 0.12)
    state["face0_operator_m11"] = _complex_linspace(0.70, 0.90, 0.20, -0.30)
    state["face1_operator_m00"] = _complex_linspace(-0.20, 0.40, 0.30, -0.20)
    state["face1_operator_m01"] = _complex_linspace(0.10, -0.30, -0.20, 0.20)
    state["face1_operator_m10"] = _complex_linspace(0.25, 0.45, 0.05, -0.15)
    state["face1_operator_m11"] = _complex_linspace(-0.60, -0.20, 0.15, 0.35)

    ref = utd_drjit.utd_accumulate_forward(
        state,
        case["rx_pos"],
        case["k"],
        case["n_edges"],
        False,
        wavelength=None,
    )
    got = utd_native.utd_accumulate_forward(
        state,
        case["rx_pos"],
        case["k"],
        case["n_edges"],
        False,
        wavelength=None,
    )

    assert _max_abs_diff(ref[0].real, got[0].real) < 1.0e-6
    assert _max_abs_diff(ref[0].imag, got[0].imag) < 1.0e-6
    assert _max_abs_diff(ref[2]["x"].real, got[2]["x"].real) < 1.0e-6
    assert _max_abs_diff(ref[2]["x"].imag, got[2]["x"].imag) < 1.0e-6
    assert _max_abs_diff(ref[2]["y"].real, got[2]["y"].real) < 1.0e-6
    assert _max_abs_diff(ref[2]["y"].imag, got[2]["y"].imag) < 1.0e-6
    assert _max_abs_diff(ref[2]["z"].real, got[2]["z"].real) < 1.0e-6
    assert _max_abs_diff(ref[2]["z"].imag, got[2]["z"].imag) < 1.0e-6


@pytest.mark.gpu
def test_native_utd_backward_stored_face_operator_matches_drjit_reference():
    if not native_extension_available():
        pytest.skip("native extension is not built in the current environment")

    case = _build_utd_case()
    state = case["state_arrays"]
    n_states = int(state["n_states"])
    rx_pos = wt.Point3f(
        dr.detach(case["rx_pos"].x),
        dr.detach(case["rx_pos"].y),
        dr.detach(case["rx_pos"].z),
    )

    def _complex_linspace(re0, re1, im0, im1):
        return wt.Complex2f(
            dr.linspace(wt.Float, re0, re1, n_states),
            dr.linspace(wt.Float, im0, im1, n_states),
        )

    ref_state = dict(state)
    ref_state["face0_operator_m00"] = _complex_linspace(0.30, 0.60, -0.20, 0.10)
    ref_state["face0_operator_m01"] = _complex_linspace(-0.40, 0.20, 0.10, -0.10)
    ref_state["face0_operator_m10"] = _complex_linspace(0.05, -0.15, -0.05, 0.12)
    ref_state["face0_operator_m11"] = _complex_linspace(0.70, 0.90, 0.20, -0.30)
    ref_state["face1_operator_m00"] = _complex_linspace(-0.20, 0.40, 0.30, -0.20)
    ref_state["face1_operator_m01"] = _complex_linspace(0.10, -0.30, -0.20, 0.20)
    ref_state["face1_operator_m10"] = _complex_linspace(0.25, 0.45, 0.05, -0.15)
    ref_state["face1_operator_m11"] = _complex_linspace(-0.60, -0.20, 0.15, 0.35)
    for key in (
        "face0_operator_m00",
        "face0_operator_m01",
        "face0_operator_m10",
        "face0_operator_m11",
        "face1_operator_m00",
        "face1_operator_m01",
        "face1_operator_m10",
        "face1_operator_m11",
    ):
        dr.enable_grad(ref_state[key].real, ref_state[key].imag)

    native_state = dict(state)
    for key in (
        "face0_operator_m00",
        "face0_operator_m01",
        "face0_operator_m10",
        "face0_operator_m11",
        "face1_operator_m00",
        "face1_operator_m01",
        "face1_operator_m10",
        "face1_operator_m11",
    ):
        native_state[key] = wt.Complex2f(dr.detach(ref_state[key].real), dr.detach(ref_state[key].imag))

    n_rx = dr.width(case["rx_pos"].x)
    grad_direct_total = wt.Complex2f(
        dr.linspace(wt.Float, 0.1, 0.4, n_rx),
        dr.linspace(wt.Float, -0.2, 0.2, n_rx),
    )
    grad_multi_total = wt.Complex2f(dr.zeros(wt.Float, n_rx), dr.zeros(wt.Float, n_rx))
    grad_direct_vector = {
        "x": wt.Complex2f(dr.linspace(wt.Float, 0.3, 0.6, n_rx), dr.linspace(wt.Float, -0.1, 0.1, n_rx)),
        "y": wt.Complex2f(dr.linspace(wt.Float, -0.4, 0.2, n_rx), dr.linspace(wt.Float, 0.2, -0.2, n_rx)),
        "z": wt.Complex2f(dr.linspace(wt.Float, 0.05, -0.05, n_rx), dr.linspace(wt.Float, 0.07, -0.03, n_rx)),
    }
    grad_multi_vector = {
        axis: wt.Complex2f(dr.zeros(wt.Float, n_rx), dr.zeros(wt.Float, n_rx))
        for axis in ("x", "y", "z")
    }

    ref_outputs = utd_drjit.utd_accumulate_forward(
        ref_state,
        rx_pos,
        case["k"],
        case["n_edges"],
        False,
        wavelength=None,
    )
    dr.backward(
        _utd_loss(
            ref_outputs,
            grad_direct_total,
            grad_multi_total,
            grad_direct_vector,
            grad_multi_vector,
        )
    )

    state_grads, _ = utd_native.utd_accumulate_backward(
        native_state,
        case["rx_pos"],
        case["k"],
        case["n_edges"],
        False,
        grad_direct_total,
        grad_multi_total,
        grad_direct_vector,
        grad_multi_vector,
        wavelength=None,
    )

    for key in (
        "face0_operator_m00",
        "face0_operator_m01",
        "face0_operator_m10",
        "face0_operator_m11",
        "face1_operator_m00",
        "face1_operator_m01",
        "face1_operator_m10",
        "face1_operator_m11",
    ):
        assert _max_abs_diff(state_grads[key].real, dr.grad(ref_state[key].real)) < 1.0e-6, key
        assert _max_abs_diff(state_grads[key].imag, dr.grad(ref_state[key].imag)) < 1.0e-6, key


@pytest.mark.gpu
def test_native_reflection_grid_z_dda_matches_drjit_reference():
    if not native_extension_available():
        pytest.skip("native extension is not built in the current environment")

    case = _build_reflection_grid_case("z")
    ref = _run_reflection_grid_reference(case)
    got = reflection_grid_native.accumulate_reflection_grid(
        grid=case["grid"],
        plane_position=case["grid"].position,
        grid_data=case["grid_data"],
        ray_origin=case["ray_origin"],
        ray_dir=case["ray_dir"],
        active=case["active"],
        blocker_dist=case["blocker_dist"],
        prev_refl_p=case["prev_refl_p"],
        prev_refl_n=case["prev_refl_n"],
        prev_tx=case["prev_tx"],
        prev_weight=case["prev_weight"],
        prev_polarization=case["prev_polarization"],
        prev_prim_idx=case["prev_prim_idx"],
        wavelength=case["wavelength"],
        k=case["k"],
        validate_paths=False,
        tri_data=None,
    )

    for ref_component, got_component in zip(ref, got):
        assert _max_abs_diff(ref_component, got_component) < 1.0e-6


@pytest.mark.gpu
def test_native_reflection_grid_x_scatter_matches_drjit_reference():
    if not native_extension_available():
        pytest.skip("native extension is not built in the current environment")

    case = _build_reflection_grid_case("x")
    ref = _run_reflection_grid_reference(case)
    got = reflection_grid_native.accumulate_reflection_grid(
        grid=case["grid"],
        plane_position=case["grid"].position,
        grid_data=case["grid_data"],
        ray_origin=case["ray_origin"],
        ray_dir=case["ray_dir"],
        active=case["active"],
        blocker_dist=case["blocker_dist"],
        prev_refl_p=case["prev_refl_p"],
        prev_refl_n=case["prev_refl_n"],
        prev_tx=case["prev_tx"],
        prev_weight=case["prev_weight"],
        prev_polarization=case["prev_polarization"],
        prev_prim_idx=case["prev_prim_idx"],
        wavelength=case["wavelength"],
        k=case["k"],
        validate_paths=False,
        tri_data=None,
    )

    for ref_component, got_component in zip(ref, got):
        assert _max_abs_diff(ref_component, got_component) < 1.0e-6


@pytest.mark.gpu
def test_native_reflection_grid_jvp_matches_drjit_reference():
    if not native_extension_available():
        pytest.skip("native extension is not built in the current environment")

    case = _build_reflection_grid_case("x")
    tangent = {
        "prev_refl_p": wt.Point3f(
            [0.02, -0.01, 0.03, -0.02, 0.01, 0.00],
            [-0.01, 0.02, -0.02, 0.01, 0.03, 0.00],
            [0.03, 0.02, -0.01, 0.01, -0.02, 0.00],
        ),
        "prev_refl_n": wt.Vector3f(
            [0.01, -0.02, 0.01, -0.01, 0.02, 0.00],
            [0.00, 0.01, -0.01, 0.02, -0.02, 0.00],
            [0.02, -0.01, 0.01, 0.00, -0.01, 0.00],
        ),
        "prev_tx": wt.Point3f(
            [0.04, -0.03, 0.02, -0.01, 0.03, 0.00],
            [-0.02, 0.01, -0.03, 0.02, -0.01, 0.00],
            [0.01, 0.02, -0.01, 0.03, -0.02, 0.00],
        ),
        "prev_weight": wt.Complex2f(
            [0.05, -0.03, 0.02, -0.01, 0.04, 0.00],
            [-0.02, 0.01, 0.03, -0.04, 0.02, 0.00],
        ),
        "prev_polarization": {
            "x": wt.Complex2f(
                [0.01, -0.01, 0.02, -0.02, 0.01, 0.00],
                [0.00, 0.01, -0.01, 0.01, -0.01, 0.00],
            ),
            "y": wt.Complex2f(
                [-0.02, 0.01, -0.01, 0.02, -0.01, 0.00],
                [0.01, -0.01, 0.01, -0.01, 0.02, 0.00],
            ),
            "z": wt.Complex2f(
                [0.02, 0.00, -0.01, 0.01, -0.02, 0.00],
                [-0.01, 0.02, 0.00, -0.02, 0.01, 0.00],
            ),
        },
    }

    ref_inputs = _clone_reflection_grid_inputs(case)
    _enable_grad_point3f(ref_inputs["prev_refl_p"])
    _enable_grad_vector3f(ref_inputs["prev_refl_n"])
    _enable_grad_point3f(ref_inputs["prev_tx"])
    _enable_grad_complex(ref_inputs["prev_weight"])
    _enable_grad_vector_complex(ref_inputs["prev_polarization"])
    _set_grad_point3f(ref_inputs["prev_refl_p"], tangent["prev_refl_p"])
    _set_grad_vector3f(ref_inputs["prev_refl_n"], tangent["prev_refl_n"])
    _set_grad_point3f(ref_inputs["prev_tx"], tangent["prev_tx"])
    _set_grad_complex(ref_inputs["prev_weight"], tangent["prev_weight"])
    _set_grad_complex(ref_inputs["prev_polarization"]["x"], tangent["prev_polarization"]["x"])
    _set_grad_complex(ref_inputs["prev_polarization"]["y"], tangent["prev_polarization"]["y"])
    _set_grad_complex(ref_inputs["prev_polarization"]["z"], tangent["prev_polarization"]["z"])
    ref_outputs = _run_reflection_grid_reference({
        **case,
        **ref_inputs,
    })
    dr.forward_to(_reflection_grid_jvp_components(ref_outputs), flags=FLAGS)
    ref_jvp = [dr.grad(component) for component in _reflection_grid_jvp_components(ref_outputs)]

    native_inputs = _clone_reflection_grid_inputs(case)
    _enable_grad_point3f(native_inputs["prev_refl_p"])
    _enable_grad_vector3f(native_inputs["prev_refl_n"])
    _enable_grad_point3f(native_inputs["prev_tx"])
    _enable_grad_complex(native_inputs["prev_weight"])
    _enable_grad_vector_complex(native_inputs["prev_polarization"])
    _set_grad_point3f(native_inputs["prev_refl_p"], tangent["prev_refl_p"])
    _set_grad_vector3f(native_inputs["prev_refl_n"], tangent["prev_refl_n"])
    _set_grad_point3f(native_inputs["prev_tx"], tangent["prev_tx"])
    _set_grad_complex(native_inputs["prev_weight"], tangent["prev_weight"])
    _set_grad_complex(native_inputs["prev_polarization"]["x"], tangent["prev_polarization"]["x"])
    _set_grad_complex(native_inputs["prev_polarization"]["y"], tangent["prev_polarization"]["y"])
    _set_grad_complex(native_inputs["prev_polarization"]["z"], tangent["prev_polarization"]["z"])
    native_outputs = reflection_grid_native.accumulate_reflection_grid(
        grid=case["grid"],
        plane_position=case["grid"].position,
        grid_data=case["grid_data"],
        ray_origin=case["ray_origin"],
        ray_dir=case["ray_dir"],
        active=case["active"],
        blocker_dist=case["blocker_dist"],
        prev_refl_p=native_inputs["prev_refl_p"],
        prev_refl_n=native_inputs["prev_refl_n"],
        prev_tx=native_inputs["prev_tx"],
        prev_weight=native_inputs["prev_weight"],
        prev_polarization=native_inputs["prev_polarization"],
        prev_prim_idx=case["prev_prim_idx"],
        wavelength=case["wavelength"],
        k=case["k"],
        validate_paths=False,
        tri_data=None,
    )
    dr.forward_to(_reflection_grid_jvp_components(native_outputs), flags=FLAGS)
    native_jvp = [dr.grad(component) for component in _reflection_grid_jvp_components(native_outputs)]

    for ref_component, got_component in zip(ref_jvp, native_jvp):
        assert _max_abs_diff(ref_component, got_component) < 5.0e-6


@pytest.mark.gpu
def test_native_reflection_grid_backward_matches_drjit_reference():
    if not native_extension_available():
        pytest.skip("native extension is not built in the current environment")

    case = _build_reflection_grid_case("x")

    ref_inputs = _clone_reflection_grid_inputs(case)
    _enable_grad_point3f(ref_inputs["prev_refl_p"])
    _enable_grad_vector3f(ref_inputs["prev_refl_n"])
    _enable_grad_point3f(ref_inputs["prev_tx"])
    _enable_grad_complex(ref_inputs["prev_weight"])
    _enable_grad_vector_complex(ref_inputs["prev_polarization"])
    ref_outputs = _run_reflection_grid_reference({**case, **ref_inputs})
    dr.backward(_reflection_grid_loss(ref_outputs), flags=FLAGS)
    ref_grads = {
        "prev_refl_p": _detach_point3f(dr.grad(ref_inputs["prev_refl_p"])),
        "prev_refl_n": _detach_vector3f(dr.grad(ref_inputs["prev_refl_n"])),
        "prev_tx": _detach_point3f(dr.grad(ref_inputs["prev_tx"])),
        "prev_weight": _detach_complex(dr.grad(ref_inputs["prev_weight"])),
        "prev_polarization": {
            "x": _detach_complex(dr.grad(ref_inputs["prev_polarization"]["x"])),
            "y": _detach_complex(dr.grad(ref_inputs["prev_polarization"]["y"])),
            "z": _detach_complex(dr.grad(ref_inputs["prev_polarization"]["z"])),
        },
    }

    native_inputs = _clone_reflection_grid_inputs(case)
    _enable_grad_point3f(native_inputs["prev_refl_p"])
    _enable_grad_vector3f(native_inputs["prev_refl_n"])
    _enable_grad_point3f(native_inputs["prev_tx"])
    _enable_grad_complex(native_inputs["prev_weight"])
    _enable_grad_vector_complex(native_inputs["prev_polarization"])
    native_outputs = reflection_grid_native.accumulate_reflection_grid(
        grid=case["grid"],
        plane_position=case["grid"].position,
        grid_data=case["grid_data"],
        ray_origin=case["ray_origin"],
        ray_dir=case["ray_dir"],
        active=case["active"],
        blocker_dist=case["blocker_dist"],
        prev_refl_p=native_inputs["prev_refl_p"],
        prev_refl_n=native_inputs["prev_refl_n"],
        prev_tx=native_inputs["prev_tx"],
        prev_weight=native_inputs["prev_weight"],
        prev_polarization=native_inputs["prev_polarization"],
        prev_prim_idx=case["prev_prim_idx"],
        wavelength=case["wavelength"],
        k=case["k"],
        validate_paths=False,
        tri_data=None,
    )
    dr.backward(_reflection_grid_loss(native_outputs), flags=FLAGS)
    native_grads = {
        "prev_refl_p": _detach_point3f(dr.grad(native_inputs["prev_refl_p"])),
        "prev_refl_n": _detach_vector3f(dr.grad(native_inputs["prev_refl_n"])),
        "prev_tx": _detach_point3f(dr.grad(native_inputs["prev_tx"])),
        "prev_weight": _detach_complex(dr.grad(native_inputs["prev_weight"])),
        "prev_polarization": {
            "x": _detach_complex(dr.grad(native_inputs["prev_polarization"]["x"])),
            "y": _detach_complex(dr.grad(native_inputs["prev_polarization"]["y"])),
            "z": _detach_complex(dr.grad(native_inputs["prev_polarization"]["z"])),
        },
    }

    assert _max_abs_diff(ref_grads["prev_refl_p"].x, native_grads["prev_refl_p"].x) < 5.0e-6
    assert _max_abs_diff(ref_grads["prev_refl_p"].y, native_grads["prev_refl_p"].y) < 5.0e-6
    assert _max_abs_diff(ref_grads["prev_refl_p"].z, native_grads["prev_refl_p"].z) < 5.0e-6
    assert _max_abs_diff(ref_grads["prev_refl_n"].x, native_grads["prev_refl_n"].x) < 5.0e-6
    assert _max_abs_diff(ref_grads["prev_refl_n"].y, native_grads["prev_refl_n"].y) < 5.0e-6
    assert _max_abs_diff(ref_grads["prev_refl_n"].z, native_grads["prev_refl_n"].z) < 5.0e-6
    assert _max_abs_diff(ref_grads["prev_tx"].x, native_grads["prev_tx"].x) < 5.0e-6
    assert _max_abs_diff(ref_grads["prev_tx"].y, native_grads["prev_tx"].y) < 5.0e-6
    assert _max_abs_diff(ref_grads["prev_tx"].z, native_grads["prev_tx"].z) < 5.0e-6
    assert _max_abs_diff(ref_grads["prev_weight"].real, native_grads["prev_weight"].real) < 5.0e-6
    assert _max_abs_diff(ref_grads["prev_weight"].imag, native_grads["prev_weight"].imag) < 5.0e-6
    assert _vector_field_max_abs_diff(
        ref_grads["prev_polarization"],
        native_grads["prev_polarization"],
    ) < 5.0e-6


@pytest.mark.gpu
def test_suffix_grid_forward_backends_produce_finite_outputs():
    case = _build_suffix_grid_case()
    ref_execution = DiffractionExecutionConfig(
        suffix_backend="drjit",
        suffix_dda="symbolic",
    )
    native_execution = DiffractionExecutionConfig(
        suffix_backend="native",
        suffix_dda="symbolic",
    )
    ref_field, ref_vector = suffix_grid_drjit.accumulate_reflected_segment_fields_batched(
        grid=case["grid"],
        grid_data=case["grid_data"],
        seg_origin=case["seg_origin"],
        seg_dir=case["seg_dir"],
        blocker_dist=case["blocker_dist"],
        seg_field=case["seg_field"],
        seg_vector=case["seg_vector"],
        state_idx=case["state_idx"],
        n_states=case["n_states"],
        wavelength=case["wavelength"],
        k=case["k"],
        active=case["active"],
        execution=ref_execution,
    )
    got_field, got_vector = suffix_grid_native.accumulate_reflected_segment_fields_batched(
        grid=case["grid"],
        grid_data=case["grid_data"],
        seg_origin=case["seg_origin"],
        seg_dir=case["seg_dir"],
        blocker_dist=case["blocker_dist"],
        seg_field=case["seg_field"],
        seg_vector=case["seg_vector"],
        state_idx=case["state_idx"],
        n_states=case["n_states"],
        wavelength=case["wavelength"],
        k=case["k"],
        active=case["active"],
        execution=native_execution,
    )

    for value in (
        ref_field.real,
        ref_field.imag,
        got_field.real,
        got_field.imag,
        ref_vector["x"].real,
        ref_vector["x"].imag,
        ref_vector["y"].real,
        ref_vector["y"].imag,
        ref_vector["z"].real,
        ref_vector["z"].imag,
        got_vector["x"].real,
        got_vector["x"].imag,
        got_vector["y"].real,
        got_vector["y"].imag,
        got_vector["z"].real,
        got_vector["z"].imag,
    ):
        assert dr.all(dr.isfinite(value))
    assert float(dr.max(dr.abs(got_field.real))[0]) > 1.0e-6


@pytest.mark.gpu
def test_drjit_symbolic_suffix_grid_rejects_ad_sensitive_inputs():
    case = _build_suffix_grid_case()
    ref_execution = DiffractionExecutionConfig(suffix_backend="drjit", suffix_dda="symbolic")
    tangent = _suffix_grid_tangent()

    ref_inputs = _clone_suffix_grid_inputs(case)
    _enable_grad_point3f(ref_inputs["seg_origin"])
    _enable_grad_complex(ref_inputs["seg_field"])
    _enable_grad_vector_complex(ref_inputs["seg_vector"])
    _set_grad_point3f(ref_inputs["seg_origin"], tangent["seg_origin"])
    _set_grad_complex(ref_inputs["seg_field"], tangent["seg_field"])
    _set_grad_complex(ref_inputs["seg_vector"]["x"], tangent["seg_vector"]["x"])
    _set_grad_complex(ref_inputs["seg_vector"]["y"], tangent["seg_vector"]["y"])
    _set_grad_complex(ref_inputs["seg_vector"]["z"], tangent["seg_vector"]["z"])

    with pytest.raises(RuntimeError, match="Use suffix_backend='native' for AD-sensitive suffix workloads"):
        suffix_grid_drjit.accumulate_reflected_segment_fields_chunk(
            grid=case["grid"],
            grid_data=case["grid_data"],
            seg_origin=ref_inputs["seg_origin"],
            seg_dir=case["seg_dir"],
            blocker_dist=case["blocker_dist"],
            seg_field=ref_inputs["seg_field"],
            seg_vector=ref_inputs["seg_vector"],
            wavelength=case["wavelength"],
            k=case["k"],
            active=case["active"],
            execution=ref_execution,
        )


@pytest.mark.gpu
def test_native_suffix_grid_jvp_matches_directional_fd():
    if not native_extension_available():
        pytest.skip("native extension is not built in the current environment")

    case = _build_suffix_grid_case()
    tangent = _suffix_grid_tangent()
    native_execution = DiffractionExecutionConfig(suffix_backend="native", suffix_dda="symbolic")
    native_inputs = _clone_suffix_grid_inputs(case)
    _enable_grad_point3f(native_inputs["seg_origin"])
    _enable_grad_complex(native_inputs["seg_field"])
    _enable_grad_vector_complex(native_inputs["seg_vector"])
    _set_grad_point3f(native_inputs["seg_origin"], tangent["seg_origin"])
    _set_grad_complex(native_inputs["seg_field"], tangent["seg_field"])
    _set_grad_complex(native_inputs["seg_vector"]["x"], tangent["seg_vector"]["x"])
    _set_grad_complex(native_inputs["seg_vector"]["y"], tangent["seg_vector"]["y"])
    _set_grad_complex(native_inputs["seg_vector"]["z"], tangent["seg_vector"]["z"])
    native_field, native_vector = suffix_grid_native.accumulate_reflected_segment_fields_chunk(
        grid=case["grid"],
        grid_data=case["grid_data"],
        seg_origin=native_inputs["seg_origin"],
        seg_dir=case["seg_dir"],
        blocker_dist=case["blocker_dist"],
        seg_field=native_inputs["seg_field"],
        seg_vector=native_inputs["seg_vector"],
        wavelength=case["wavelength"],
        k=case["k"],
        active=case["active"],
        execution=native_execution,
    )
    native_loss = _suffix_grid_loss(native_field, native_vector)
    dr.forward_to(native_loss, flags=FLAGS)
    native_directional = float(dr.grad(native_loss)[0])
    assert math.isfinite(native_directional)
    assert abs(native_directional) > 1.0e-6


@pytest.mark.gpu
def test_native_suffix_grid_backward_matches_directional_fd():
    if not native_extension_available():
        pytest.skip("native extension is not built in the current environment")

    case = _build_suffix_grid_case()
    tangent = _suffix_grid_tangent()
    native_execution = DiffractionExecutionConfig(suffix_backend="native", suffix_dda="symbolic")
    native_inputs = _clone_suffix_grid_inputs(case)
    _enable_grad_point3f(native_inputs["seg_origin"])
    _enable_grad_complex(native_inputs["seg_field"])
    _enable_grad_vector_complex(native_inputs["seg_vector"])
    native_field, native_vector = suffix_grid_native.accumulate_reflected_segment_fields_chunk(
        grid=case["grid"],
        grid_data=case["grid_data"],
        seg_origin=native_inputs["seg_origin"],
        seg_dir=case["seg_dir"],
        blocker_dist=case["blocker_dist"],
        seg_field=native_inputs["seg_field"],
        seg_vector=native_inputs["seg_vector"],
        wavelength=case["wavelength"],
        k=case["k"],
        active=case["active"],
        execution=native_execution,
    )
    dr.backward(_suffix_grid_loss(native_field, native_vector), flags=FLAGS)
    native_grads = {
        "seg_origin": _detach_point3f(dr.grad(native_inputs["seg_origin"])),
        "seg_field": _detach_complex(dr.grad(native_inputs["seg_field"])),
        "seg_vector": {
            "x": _detach_complex(dr.grad(native_inputs["seg_vector"]["x"])),
            "y": _detach_complex(dr.grad(native_inputs["seg_vector"]["y"])),
            "z": _detach_complex(dr.grad(native_inputs["seg_vector"]["z"])),
        },
    }
    native_directional = _suffix_grid_grad_dot_tangent(native_grads, tangent)
    assert math.isfinite(native_directional)
    assert abs(native_directional) > 1.0e-6


@pytest.mark.gpu
def test_native_radiomap_vector_power_matches_reference():
    if not native_extension_available():
        pytest.skip("native extension is not built in the current environment")

    case = _radiomap_vector_power_case()
    ref = radio_map_accumulate_native._reference_vector_power(case["field_vector"])
    got = radio_map_accumulate_native.vector_power(case["field_vector"])
    assert _max_abs_diff(ref, got) < 1.0e-6


@pytest.mark.gpu
def test_native_radiomap_vector_power_jvp_matches_reference():
    if not native_extension_available():
        pytest.skip("native extension is not built in the current environment")

    case = _radiomap_vector_power_case()
    tangent = _radiomap_vector_power_tangent()

    ref_inputs = _clone_radiomap_vector_power_inputs(case)
    _enable_grad_vector_complex(ref_inputs["field_vector"])
    _set_grad_complex(ref_inputs["field_vector"]["x"], tangent["field_vector"]["x"])
    _set_grad_complex(ref_inputs["field_vector"]["y"], tangent["field_vector"]["y"])
    _set_grad_complex(ref_inputs["field_vector"]["z"], tangent["field_vector"]["z"])
    ref = radio_map_accumulate_native._reference_vector_power(ref_inputs["field_vector"])
    dr.forward_to(ref, flags=FLAGS)
    ref_jvp = _detach_float(dr.grad(ref))

    native_inputs = _clone_radiomap_vector_power_inputs(case)
    _enable_grad_vector_complex(native_inputs["field_vector"])
    _set_grad_complex(native_inputs["field_vector"]["x"], tangent["field_vector"]["x"])
    _set_grad_complex(native_inputs["field_vector"]["y"], tangent["field_vector"]["y"])
    _set_grad_complex(native_inputs["field_vector"]["z"], tangent["field_vector"]["z"])
    got = radio_map_accumulate_native.vector_power(native_inputs["field_vector"])
    dr.forward_to(got, flags=FLAGS)
    got_jvp = _detach_float(dr.grad(got))

    assert _max_abs_diff(ref_jvp, got_jvp) < 1.0e-6


@pytest.mark.gpu
def test_native_radiomap_vector_power_backward_matches_reference():
    if not native_extension_available():
        pytest.skip("native extension is not built in the current environment")

    case = _radiomap_vector_power_case()

    ref_inputs = _clone_radiomap_vector_power_inputs(case)
    _enable_grad_vector_complex(ref_inputs["field_vector"])
    ref = radio_map_accumulate_native._reference_vector_power(ref_inputs["field_vector"])
    dr.backward(_radiomap_vector_power_loss(ref), flags=FLAGS)
    ref_grads = {
        "x": _detach_complex(dr.grad(ref_inputs["field_vector"]["x"])),
        "y": _detach_complex(dr.grad(ref_inputs["field_vector"]["y"])),
        "z": _detach_complex(dr.grad(ref_inputs["field_vector"]["z"])),
    }

    native_inputs = _clone_radiomap_vector_power_inputs(case)
    _enable_grad_vector_complex(native_inputs["field_vector"])
    got = radio_map_accumulate_native.vector_power(native_inputs["field_vector"])
    dr.backward(_radiomap_vector_power_loss(got), flags=FLAGS)
    native_grads = {
        "x": _detach_complex(dr.grad(native_inputs["field_vector"]["x"])),
        "y": _detach_complex(dr.grad(native_inputs["field_vector"]["y"])),
        "z": _detach_complex(dr.grad(native_inputs["field_vector"]["z"])),
    }

    assert _vector_field_max_abs_diff(ref_grads, native_grads) < 1.0e-6


@pytest.mark.gpu
def test_native_radiomap_shadow_boundary_incident_stats_match_reference():
    if not native_extension_available():
        pytest.skip("native extension is not built in the current environment")

    case = _radiomap_shadow_boundary_incident_stats_case()
    ref = radio_map_accumulate_native._reference_radiomap_shadow_boundary_incident_statistics(
        **case,
        include_diagnostics=True,
    )
    got = radio_map_accumulate_native.radiomap_shadow_boundary_incident_statistics(
        **case,
        include_diagnostics=True,
    )
    assert _max_abs_diff(ref["sum_incident_weight"], got["sum_incident_weight"]) < 2.0e-5
    assert _max_abs_diff(ref["max_incident_weight"], got["max_incident_weight"]) < 2.0e-5
    assert _max_abs_diff(ref["weighted_incident_response_real"], got["weighted_incident_response_real"]) < 2.0e-5
    assert _max_abs_diff(ref["weighted_incident_response_imag"], got["weighted_incident_response_imag"]) < 2.0e-5
    assert _max_abs_diff(ref["second_max_incident_weight"], got["second_max_incident_weight"]) < 2.0e-5
    assert _max_abs_diff(ref["argmax_margin"], got["argmax_margin"]) < 2.0e-5
    np.testing.assert_array_equal(
        np.asarray(ref["argmax_edge_idx"], dtype=np.int32),
        np.asarray(got["argmax_edge_idx"], dtype=np.int32),
    )
    np.testing.assert_array_equal(
        np.asarray(ref["support_edge_count"], dtype=np.int32),
        np.asarray(got["support_edge_count"], dtype=np.int32),
    )


def test_reference_shadow_boundary_incident_stats_taper_target_inner_support():
    case = _radiomap_shadow_boundary_inner_taper_case()
    ref = radio_map_accumulate_native._reference_radiomap_shadow_boundary_incident_statistics(**case)

    weights = np.asarray(ref["sum_incident_weight"], dtype=np.float32)
    response_real = np.asarray(ref["weighted_incident_response_real"], dtype=np.float32)
    response_imag = np.asarray(ref["weighted_incident_response_imag"], dtype=np.float32)

    assert weights[0] > 0.0
    assert weights[1] > 0.0
    assert weights[2] == 0.0
    assert abs(float(response_real[1])) + abs(float(response_imag[1])) > 0.0


@pytest.mark.gpu
def test_native_radiomap_shadow_boundary_incident_stats_taper_target_inner_support_matches_reference():
    if not native_extension_available():
        pytest.skip("native extension is not built in the current environment")

    case = _radiomap_shadow_boundary_inner_taper_case()
    ref = radio_map_accumulate_native._reference_radiomap_shadow_boundary_incident_statistics(
        **case,
        include_diagnostics=True,
    )
    got = radio_map_accumulate_native.radiomap_shadow_boundary_incident_statistics(
        **case,
        include_diagnostics=True,
    )

    assert _max_abs_diff(ref["sum_incident_weight"], got["sum_incident_weight"]) < 2.0e-5
    assert _max_abs_diff(ref["max_incident_weight"], got["max_incident_weight"]) < 2.0e-5
    assert _max_abs_diff(ref["weighted_incident_response_real"], got["weighted_incident_response_real"]) < 2.0e-5
    assert _max_abs_diff(ref["weighted_incident_response_imag"], got["weighted_incident_response_imag"]) < 2.0e-5
    assert float(np.asarray(got["sum_incident_weight"], dtype=np.float32)[1]) > 0.0


@pytest.mark.gpu
def test_native_radiomap_shadow_boundary_incident_stats_jvp_matches_reference():
    if not native_extension_available():
        pytest.skip("native extension is not built in the current environment")

    case = _radiomap_shadow_boundary_incident_stats_case()
    tangent = _radiomap_shadow_boundary_incident_stats_tangent()

    ref_inputs = _clone_radiomap_shadow_boundary_incident_stats_inputs(case)
    _enable_grad_point3f(ref_inputs["tx_pos"])
    _enable_grad_point3f(ref_inputs["rx_pos"])
    _enable_grad_point3f(ref_inputs["edge_pos"])
    _set_grad_point3f(ref_inputs["tx_pos"], tangent["tx_pos"])
    _set_grad_point3f(ref_inputs["rx_pos"], tangent["rx_pos"])
    _set_grad_point3f(ref_inputs["edge_pos"], tangent["edge_pos"])
    ref = radio_map_accumulate_native._reference_radiomap_shadow_boundary_incident_statistics(**ref_inputs)
    ref_loss = _radiomap_shadow_boundary_incident_stats_loss(ref)
    dr.forward_to(ref_loss, flags=FLAGS)
    ref_jvp = float(dr.grad(ref_loss)[0])

    native_inputs = _clone_radiomap_shadow_boundary_incident_stats_inputs(case)
    _enable_grad_point3f(native_inputs["tx_pos"])
    _enable_grad_point3f(native_inputs["rx_pos"])
    _enable_grad_point3f(native_inputs["edge_pos"])
    _set_grad_point3f(native_inputs["tx_pos"], tangent["tx_pos"])
    _set_grad_point3f(native_inputs["rx_pos"], tangent["rx_pos"])
    _set_grad_point3f(native_inputs["edge_pos"], tangent["edge_pos"])
    got = radio_map_accumulate_native.radiomap_shadow_boundary_incident_statistics(**native_inputs)
    got_loss = _radiomap_shadow_boundary_incident_stats_loss(got)
    dr.forward_to(got_loss, flags=FLAGS)
    got_jvp = float(dr.grad(got_loss)[0])

    assert abs(ref_jvp - got_jvp) < 2.0e-5


@pytest.mark.gpu
def test_native_radiomap_shadow_boundary_incident_stats_backward_matches_reference():
    if not native_extension_available():
        pytest.skip("native extension is not built in the current environment")

    case = _radiomap_shadow_boundary_incident_stats_case()

    ref_inputs = _clone_radiomap_shadow_boundary_incident_stats_inputs(case)
    _enable_grad_point3f(ref_inputs["tx_pos"])
    _enable_grad_point3f(ref_inputs["rx_pos"])
    _enable_grad_point3f(ref_inputs["edge_pos"])
    ref = radio_map_accumulate_native._reference_radiomap_shadow_boundary_incident_statistics(**ref_inputs)
    dr.backward(_radiomap_shadow_boundary_incident_stats_loss(ref), flags=FLAGS)
    ref_grads = _radiomap_shadow_boundary_incident_stats_input_grads(ref_inputs)

    native_inputs = _clone_radiomap_shadow_boundary_incident_stats_inputs(case)
    _enable_grad_point3f(native_inputs["tx_pos"])
    _enable_grad_point3f(native_inputs["rx_pos"])
    _enable_grad_point3f(native_inputs["edge_pos"])
    got = radio_map_accumulate_native.radiomap_shadow_boundary_incident_statistics(**native_inputs)
    dr.backward(_radiomap_shadow_boundary_incident_stats_loss(got), flags=FLAGS)
    native_grads = _radiomap_shadow_boundary_incident_stats_input_grads(native_inputs)

    assert _max_abs_diff(ref_grads["tx_pos"].x, native_grads["tx_pos"].x) < 3.0e-5
    assert _max_abs_diff(ref_grads["tx_pos"].y, native_grads["tx_pos"].y) < 3.0e-5
    assert _max_abs_diff(ref_grads["tx_pos"].z, native_grads["tx_pos"].z) < 3.0e-5
    assert _max_abs_diff(ref_grads["rx_pos"].x, native_grads["rx_pos"].x) < 3.0e-5
    assert _max_abs_diff(ref_grads["rx_pos"].y, native_grads["rx_pos"].y) < 3.0e-5
    assert _max_abs_diff(ref_grads["rx_pos"].z, native_grads["rx_pos"].z) < 3.0e-5
    assert _max_abs_diff(ref_grads["edge_pos"].x, native_grads["edge_pos"].x) < 3.0e-5
    assert _max_abs_diff(ref_grads["edge_pos"].y, native_grads["edge_pos"].y) < 3.0e-5
    assert _max_abs_diff(ref_grads["edge_pos"].z, native_grads["edge_pos"].z) < 3.0e-5


@pytest.mark.gpu
def test_native_radiomap_matched_isb_completion_matches_reference():
    if not native_extension_available():
        pytest.skip("native extension is not built in the current environment")

    case = _radiomap_matched_isb_case()
    ref = radio_map_accumulate_native._reference_matched_isb_completion(**case)
    got = radio_map_accumulate_native.matched_isb_completion(**case)

    assert _max_abs_diff(ref["coherent"].real, got["coherent"].real) < 1.0e-6
    assert _max_abs_diff(ref["coherent"].imag, got["coherent"].imag) < 1.0e-6
    assert _max_abs_diff(ref["power"], got["power"]) < 1.0e-6
    assert _vector_field_max_abs_diff(ref["vector_coherent"], got["vector_coherent"]) < 1.0e-6
    assert _max_abs_diff(ref["continued_direct_power"], got["continued_direct_power"]) < 1.0e-6
    assert _max_abs_diff(ref["transition_magnitude"], got["transition_magnitude"]) < 1.0e-6
    assert _max_abs_diff(ref["transition_phase"], got["transition_phase"]) < 1.0e-6


@pytest.mark.gpu
def test_native_radiomap_matched_isb_completion_jvp_matches_reference():
    if not native_extension_available():
        pytest.skip("native extension is not built in the current environment")

    case = _radiomap_matched_isb_case()
    tangent = _radiomap_matched_isb_tangent()

    ref_inputs = _clone_radiomap_matched_isb_inputs(case)
    _enable_grad_complex(ref_inputs["continued_direct"])
    _enable_grad_vector3f(ref_inputs["tx_basis"])
    _enable_grad_vector3f(ref_inputs["rx_basis"])
    dr.enable_grad(ref_inputs["incident_weight"])
    _enable_grad_complex(ref_inputs["incident_response"])
    _enable_grad_vector_complex(ref_inputs["raw_transition_vector"])
    _set_grad_complex(ref_inputs["continued_direct"], tangent["continued_direct"])
    _set_grad_vector3f(ref_inputs["tx_basis"], tangent["tx_basis"])
    _set_grad_vector3f(ref_inputs["rx_basis"], tangent["rx_basis"])
    dr.set_grad(ref_inputs["incident_weight"], tangent["incident_weight"])
    _set_grad_complex(ref_inputs["incident_response"], tangent["incident_response"])
    _set_grad_complex(ref_inputs["raw_transition_vector"]["x"], tangent["raw_transition_vector"]["x"])
    _set_grad_complex(ref_inputs["raw_transition_vector"]["y"], tangent["raw_transition_vector"]["y"])
    _set_grad_complex(ref_inputs["raw_transition_vector"]["z"], tangent["raw_transition_vector"]["z"])
    ref = radio_map_accumulate_native._reference_matched_isb_completion(**ref_inputs)
    ref_loss = _radiomap_matched_isb_loss(ref)
    dr.forward_to(ref_loss, flags=FLAGS)
    ref_jvp = float(dr.grad(ref_loss)[0])

    native_inputs = _clone_radiomap_matched_isb_inputs(case)
    _enable_grad_complex(native_inputs["continued_direct"])
    _enable_grad_vector3f(native_inputs["tx_basis"])
    _enable_grad_vector3f(native_inputs["rx_basis"])
    dr.enable_grad(native_inputs["incident_weight"])
    _enable_grad_complex(native_inputs["incident_response"])
    _enable_grad_vector_complex(native_inputs["raw_transition_vector"])
    _set_grad_complex(native_inputs["continued_direct"], tangent["continued_direct"])
    _set_grad_vector3f(native_inputs["tx_basis"], tangent["tx_basis"])
    _set_grad_vector3f(native_inputs["rx_basis"], tangent["rx_basis"])
    dr.set_grad(native_inputs["incident_weight"], tangent["incident_weight"])
    _set_grad_complex(native_inputs["incident_response"], tangent["incident_response"])
    _set_grad_complex(native_inputs["raw_transition_vector"]["x"], tangent["raw_transition_vector"]["x"])
    _set_grad_complex(native_inputs["raw_transition_vector"]["y"], tangent["raw_transition_vector"]["y"])
    _set_grad_complex(native_inputs["raw_transition_vector"]["z"], tangent["raw_transition_vector"]["z"])
    got = radio_map_accumulate_native.matched_isb_completion(**native_inputs)
    got_loss = _radiomap_matched_isb_loss(got)
    dr.forward_to(got_loss, flags=FLAGS)
    got_jvp = float(dr.grad(got_loss)[0])

    assert abs(ref_jvp - got_jvp) < 5.0e-6


@pytest.mark.gpu
def test_native_radiomap_matched_isb_completion_backward_matches_reference():
    if not native_extension_available():
        pytest.skip("native extension is not built in the current environment")

    case = _radiomap_matched_isb_case()

    ref_inputs = _clone_radiomap_matched_isb_inputs(case)
    _enable_grad_complex(ref_inputs["continued_direct"])
    _enable_grad_vector3f(ref_inputs["tx_basis"])
    _enable_grad_vector3f(ref_inputs["rx_basis"])
    dr.enable_grad(ref_inputs["incident_weight"])
    _enable_grad_complex(ref_inputs["incident_response"])
    _enable_grad_vector_complex(ref_inputs["raw_transition_vector"])
    ref = radio_map_accumulate_native._reference_matched_isb_completion(**ref_inputs)
    dr.backward(_radiomap_matched_isb_loss(ref), flags=FLAGS)
    ref_grads = _radiomap_matched_isb_input_grads(ref_inputs)

    native_inputs = _clone_radiomap_matched_isb_inputs(case)
    _enable_grad_complex(native_inputs["continued_direct"])
    _enable_grad_vector3f(native_inputs["tx_basis"])
    _enable_grad_vector3f(native_inputs["rx_basis"])
    dr.enable_grad(native_inputs["incident_weight"])
    _enable_grad_complex(native_inputs["incident_response"])
    _enable_grad_vector_complex(native_inputs["raw_transition_vector"])
    got = radio_map_accumulate_native.matched_isb_completion(**native_inputs)
    dr.backward(_radiomap_matched_isb_loss(got), flags=FLAGS)
    native_grads = _radiomap_matched_isb_input_grads(native_inputs)

    assert _max_abs_diff(ref_grads["continued_direct"].real, native_grads["continued_direct"].real) < 5.0e-6
    assert _max_abs_diff(ref_grads["continued_direct"].imag, native_grads["continued_direct"].imag) < 5.0e-6
    assert _max_abs_diff(ref_grads["tx_basis"].x, native_grads["tx_basis"].x) < 5.0e-6
    assert _max_abs_diff(ref_grads["tx_basis"].y, native_grads["tx_basis"].y) < 5.0e-6
    assert _max_abs_diff(ref_grads["tx_basis"].z, native_grads["tx_basis"].z) < 5.0e-6
    assert _max_abs_diff(ref_grads["rx_basis"].x, native_grads["rx_basis"].x) < 5.0e-6
    assert _max_abs_diff(ref_grads["rx_basis"].y, native_grads["rx_basis"].y) < 5.0e-6
    assert _max_abs_diff(ref_grads["rx_basis"].z, native_grads["rx_basis"].z) < 5.0e-6
    assert _max_abs_diff(ref_grads["incident_weight"], native_grads["incident_weight"]) < 5.0e-6
    assert _max_abs_diff(ref_grads["incident_response"].real, native_grads["incident_response"].real) < 5.0e-6
    assert _max_abs_diff(ref_grads["incident_response"].imag, native_grads["incident_response"].imag) < 5.0e-6
    assert _vector_field_max_abs_diff(
        ref_grads["raw_transition_vector"],
        native_grads["raw_transition_vector"],
    ) < 5.0e-6


@pytest.mark.gpu
def test_compute_reflection_field_native_backend_requires_extension(monkeypatch):
    field = Field(bounds=((-1.0, 1.0), (0.5, 2.5)), size=(4, 4), axis="x", position=0.25)
    scene = build_test_scene(box_geometry(center=(0.0, 0.0, 1.5), size=2.0))
    monkeypatch.setattr(reflection_api_module, "native_extension_available", lambda: False)

    with pytest.raises(RuntimeError, match="reflection_field_backend='native'"):
        reflection_api_module.compute_reflection_field(
            grid=field,
            rx_z=0.25,
            tx_pos=wt.Point3f(0.0, -3.0, 1.5),
            scene=scene,
            wavelength=299792458.0 / 1.0e9,
            k=float(2.0 * dr.pi / (299792458.0 / 1.0e9)),
            n_rays=16,
            max_reflections=1,
            mode="3d",
            reflection_coef=1.0,
            reflection_field_backend="native",
            return_per_bounce=False,
        )


def test_native_grid_kernels_require_raw_launchers(monkeypatch):
    class MissingLaunchers:
        pass

    monkeypatch.setattr(reflection_grid_native, "_extension", lambda: MissingLaunchers())
    monkeypatch.setattr(suffix_grid_native, "_extension", lambda: MissingLaunchers())

    with pytest.raises(RuntimeError, match="Native reflection grid backend requires array launchers"):
        reflection_grid_native._require_native_reflection_grid_kernel()
    with pytest.raises(RuntimeError, match="Native suffix grid backend requires raw launchers"):
        suffix_grid_native._require_native_suffix_kernel()


@pytest.mark.gpu
def test_tracer_default_native_backends_match_explicit_drjit():
    if not native_extension_available():
        pytest.skip("native extension is not built in the current environment")

    scene = build_test_scene(
        box_geometry(center=(-1.8, -1.2, 1.5), size=2.0),
        box_geometry(center=(1.8, 1.2, 1.5), size=2.0),
    )
    tx = wt.Point3f(0.0, -4.0, 1.5)
    monitor = FieldMonitor(
        "mixed_plane",
        axis="x",
        position=0.25,
        bounds=((-3.0, 3.0), (0.5, 2.5)),
        grid_size=8,
    )

    tracer_native = Tracer(
        frequency=1.0e9,
        scene=scene,
        reflection_n_rays=64,
        reflection_max_bounces=1,
        enable_rd_diffraction=True,
        max_diffractions=1,
    )
    tracer_drjit = Tracer(
        frequency=1.0e9,
        scene=scene,
        config={
            "trace": {
                "reflection_n_rays": 64,
                "reflection_max_bounces": 1,
                "enable_rd_diffraction": True,
                "max_diffractions": 1,
                "reflection_field_backend": "drjit",
                "diffraction_execution": {
                    "suffix_backend": "drjit",
                    "suffix_dda": "symbolic",
                },
            },
        },
    )

    native_result = tracer_native.trace(tx, monitor=monitor, verbose=False)
    drjit_result = tracer_drjit.trace(tx, monitor=monitor, verbose=False)

    assert _max_abs_diff(
        native_result.primary.field.total.real,
        drjit_result.primary.field.total.real,
    ) < 1.0e-4
    assert _max_abs_diff(
        native_result.primary.field.total.imag,
        drjit_result.primary.field.total.imag,
    ) < 1.0e-4

    suffix_metadata = native_result.primary.metadata["reflection_suffix_backend"]
    assert suffix_metadata["requested_backend"] == "native"
    assert suffix_metadata["resolved_backend"] == "native"
    assert suffix_metadata["implementation"] == "native_cuda_custom_op"

