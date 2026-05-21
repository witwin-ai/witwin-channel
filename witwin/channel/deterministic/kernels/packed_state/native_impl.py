"""Native CUDA implementation of packed-state gather/concat/subset."""

from __future__ import annotations

import drjit as dr

from witwin.channel.deterministic import types as wt
from witwin.channel._native.deterministic import NativeExtension
from witwin.channel.deterministic.kernels.packed_state.drjit_impl import (
    build_diffraction_path_slots as _drjit_build_diffraction_path_slots,
    concat_state_arrays as _drjit_concat,
    gather_field_evaluation_state_fields as _drjit_gather_field_eval,
    gather_state_arrays as _drjit_gather,
    gather_inserted_reflection_state_fields as _drjit_gather_inserted_fields,
    subset_state_arrays as _drjit_subset,
)
from witwin.channel.deterministic.diffraction.state import STATE_STATIC_KEYS, SK_ADJACENT_FACE0, SK_ADJACENT_FACE1, SK_APPROXIMATION_MODE_CODE, SK_EDGE_DIR, SK_EDGE_LINE_MAX, SK_EDGE_LINE_MIN, SK_EDGE_IDX, SK_EDGE_POS, SK_FACE0_ETA_R, SK_FACE0_GAIN, SK_FACE0_OPERATOR_M00, SK_FACE0_OPERATOR_M01, SK_FACE0_OPERATOR_M10, SK_FACE0_OPERATOR_M11, SK_FACE0_SIGMA, SK_FACE0_USE_FRESNEL, SK_FACE1_ETA_R, SK_FACE1_GAIN, SK_FACE1_OPERATOR_M00, SK_FACE1_OPERATOR_M01, SK_FACE1_OPERATOR_M10, SK_FACE1_OPERATOR_M11, SK_FACE1_SIGMA, SK_FACE1_USE_FRESNEL, SK_FIRST_INTERACTION_POS, SK_INCIDENT_BASIS_K, SK_INCIDENT_BASIS_U, SK_INCIDENT_BASIS_V, SK_INCIDENT_DERIVATIVE_JONES_U, SK_INCIDENT_DERIVATIVE_JONES_V, SK_INCIDENT_FIELD, SK_INCIDENT_JONES_U, SK_INCIDENT_JONES_V, SK_INCIDENT_NORMAL_DERIVATIVE, SK_INCIDENT_NORMAL_DERIVATIVE_VECTOR_X, SK_INCIDENT_NORMAL_DERIVATIVE_VECTOR_Y, SK_INCIDENT_NORMAL_DERIVATIVE_VECTOR_Z, SK_INCIDENT_VECTOR_X, SK_INCIDENT_VECTOR_Y, SK_INCIDENT_VECTOR_Z, SK_INTERMEDIATE_REFLECTION_DEPTH, SK_IS_DIRECT_TX, SK_N0, SK_NN, SK_N_STATES, SK_ORDER, SK_PATH_LENGTH_PREFIX, SK_PREFIX_REFLECTION_DEPTH, SK_R0, SK_RN, SK_SOURCE_POS, SK_SOURCE_TYPE_CODE, SK_SUFFIX_REFLECTION_DEPTH, SK_WEDGE_N, Geo
from witwin.channel.deterministic.diffraction.state import SK_FACE0_MU_R, SK_FACE1_MU_R
from witwin.channel.deterministic.diffraction.state import State
from witwin.channel.deterministic.diffraction.state import State
from witwin.channel.deterministic.types import InteractionType
from witwin.channel.core.numerics.arrays import scalar
from witwin.channel.core.physics.polarization import vector_from_jones

_PACKED_BUFFER_KEY = "__native_packed_state_buffer__"
_PACKED_HISTORY_SIZE_KEY = "__native_packed_state_history_size__"
_PACKED_STRIDE_KEY = "__native_packed_state_stride__"
_OPTIONAL_COLD_SPECS = (
    (SK_PATH_LENGTH_PREFIX, wt.Float),
    (SK_FIRST_INTERACTION_POS, wt.Point3f),
    (SK_IS_DIRECT_TX, wt.Bool),
    (SK_SOURCE_TYPE_CODE, wt.UInt32),
    (SK_APPROXIMATION_MODE_CODE, wt.UInt32),
    (SK_EDGE_LINE_MIN, wt.Float),
    (SK_EDGE_LINE_MAX, wt.Float),
)
_NON_DIFFERENTIABLE_STATE_KEYS = {
    SK_EDGE_IDX,
    SK_ADJACENT_FACE0,
    SK_ADJACENT_FACE1,
    SK_FACE0_USE_FRESNEL,
    SK_FACE1_USE_FRESNEL,
    SK_IS_DIRECT_TX,
    SK_SOURCE_TYPE_CODE,
    SK_PREFIX_REFLECTION_DEPTH,
    SK_INTERMEDIATE_REFLECTION_DEPTH,
    SK_SUFFIX_REFLECTION_DEPTH,
    SK_APPROXIMATION_MODE_CODE,
    SK_ORDER,
    SK_N_STATES,
}
_INSERTED_REFLECTION_GATHER_GRAD_KEYS = (
    SK_EDGE_POS,
    SK_PATH_LENGTH_PREFIX,
    SK_FIRST_INTERACTION_POS,
)


def _strip_packed_lineage(state_arrays: dict | None) -> dict | None:
    if state_arrays is None:
        return None
    return {
        key: value
        for key, value in state_arrays.items()
        if key not in {_PACKED_BUFFER_KEY, _PACKED_HISTORY_SIZE_KEY, _PACKED_STRIDE_KEY}
    }


def _empty_inserted_reflection_state_fields(
    history_size: int,
    *,
    retain_lineage_state: bool,
) -> dict:
    selected = {
        SK_EDGE_POS: wt.Point3f(
            dr.zeros(wt.Float, 0),
            dr.zeros(wt.Float, 0),
            dr.zeros(wt.Float, 0),
        ),
        SK_PREFIX_REFLECTION_DEPTH: dr.zeros(wt.UInt32, 0),
        SK_INTERMEDIATE_REFLECTION_DEPTH: dr.zeros(wt.UInt32, 0),
        SK_SUFFIX_REFLECTION_DEPTH: dr.zeros(wt.UInt32, 0),
        SK_ORDER: dr.zeros(wt.UInt32, 0),
        SK_N_STATES: 0,
    }
    if retain_lineage_state:
        selected[SK_PATH_LENGTH_PREFIX] = dr.zeros(wt.Float, 0)
        selected[SK_FIRST_INTERACTION_POS] = wt.Point3f(
            dr.zeros(wt.Float, 0),
            dr.zeros(wt.Float, 0),
            dr.zeros(wt.Float, 0),
        )
        selected[SK_SOURCE_TYPE_CODE] = dr.zeros(wt.UInt32, 0)
    return State.attach_lineage(selected, None, history_size)


def _native_extension_or_none():
    try:
        return NativeExtension.load()
    except ImportError:
        return None

def _packed_stride(ext, history_size: int) -> int:
    del history_size
    raw = int(ext.PACKED_CORE_FLOATS)
    return (raw + 3) & ~3


def _array_has_grad(value) -> bool:
    try:
        return bool(dr.grad_enabled(value))
    except TypeError:
        return False


def _iter_leaf_arrays(value):
    try:
        x = value.x
        y = value.y
    except Exception:
        x = None
        y = None
    if x is not None and y is not None:
        yield from _iter_leaf_arrays(x)
        yield from _iter_leaf_arrays(y)
        try:
            z = value.z
        except Exception:
            z = None
        if z is not None:
            yield from _iter_leaf_arrays(z)
        return
    try:
        real = value.real
        imag = value.imag
    except Exception:
        real = None
        imag = None
    if real is not None and imag is not None and real is not value and imag is not value:
        yield from _iter_leaf_arrays(real)
        yield from _iter_leaf_arrays(imag)
        return
    yield value


def _value_has_grad(value) -> bool:
    if value is None:
        return False
    for leaf in _iter_leaf_arrays(value):
        if _array_has_grad(leaf):
            return True
    return False


def _state_arrays_has_grad(state_arrays: dict) -> bool:
    for key in STATE_STATIC_KEYS:
        if key not in state_arrays:
            continue
        if _value_has_grad(state_arrays[key]):
            return True
    return False


def _is_vector_like(value) -> bool:
    try:
        value.x
        value.y
        value.z
        return True
    except Exception:
        return False


def _is_complex_like(value) -> bool:
    try:
        real = value.real
        imag = value.imag
    except Exception:
        return False
    return real is not value and imag is not value


def _gather_grad_value(value, indices):
    if value is None:
        return None
    if _is_vector_like(value):
        return type(value)(
            _gather_grad_value(value.x, indices),
            _gather_grad_value(value.y, indices),
            _gather_grad_value(value.z, indices),
        )
    if _is_complex_like(value):
        return wt.Complex2f(
            _gather_grad_value(value.real, indices),
            _gather_grad_value(value.imag, indices),
        )
    return dr.gather(type(value), value, indices)


def _zero_like_state_value(template_value):
    if template_value is None:
        return None
    if isinstance(template_value, dict):
        return {
            key: _zero_like_state_value(value)
            for key, value in template_value.items()
        }
    if _is_vector_like(template_value):
        return type(template_value)(
            _zero_like_state_value(template_value.x),
            _zero_like_state_value(template_value.y),
            _zero_like_state_value(template_value.z),
        )
    if _is_complex_like(template_value):
        return wt.Complex2f(
            _zero_like_state_value(template_value.real),
            _zero_like_state_value(template_value.imag),
        )
    if isinstance(template_value, bool):
        return False
    if isinstance(template_value, int):
        return 0
    if isinstance(template_value, float):
        return 0.0
    try:
        return dr.zeros(type(template_value), dr.width(template_value))
    except Exception:
        return 0


def _scatter_grad_value(template_value, grad_value, indices, width: int):
    if grad_value is None:
        return None
    if _is_vector_like(template_value):
        return type(template_value)(
            _scatter_grad_value(template_value.x, grad_value.x, indices, width),
            _scatter_grad_value(template_value.y, grad_value.y, indices, width),
            _scatter_grad_value(template_value.z, grad_value.z, indices, width),
        )
    if _is_complex_like(template_value):
        return wt.Complex2f(
            _scatter_grad_value(template_value.real, grad_value.real, indices, width),
            _scatter_grad_value(template_value.imag, grad_value.imag, indices, width),
        )
    out = dr.zeros(type(template_value), width)
    dr.scatter_reduce(dr.ReduceOp.Add, out, grad_value, indices)
    return out


def _zero_like_grad_state_dict(template_state_arrays: dict):
    if template_state_arrays is None:
        return None
    return {
        key: _zero_like_state_value(template_state_arrays[key])
        for key in STATE_STATIC_KEYS
        if key in template_state_arrays
        and key not in _NON_DIFFERENTIABLE_STATE_KEYS
        and not key.startswith("path_")
    }


def _gather_state_grad_dict(template_state_arrays: dict, grad_state_arrays, indices):
    if template_state_arrays is None:
        return None
    gathered = _zero_like_grad_state_dict(template_state_arrays)
    if grad_state_arrays is None:
        return gathered
    for key in STATE_STATIC_KEYS:
        if key not in template_state_arrays:
            continue
        if key in _NON_DIFFERENTIABLE_STATE_KEYS or key.startswith("path_"):
            continue
        value = grad_state_arrays.get(key)
        gathered_value = _gather_grad_value(value, indices)
        if gathered_value is not None:
            gathered[key] = gathered_value
    return gathered


def _scatter_state_grad_dict(template_state_arrays: dict, grad_state_arrays, indices):
    if template_state_arrays is None:
        return None
    n_states = int(template_state_arrays[SK_N_STATES])
    scattered = _zero_like_grad_state_dict(template_state_arrays)
    if grad_state_arrays is None:
        return scattered
    for key in STATE_STATIC_KEYS:
        if key not in template_state_arrays:
            continue
        if key in _NON_DIFFERENTIABLE_STATE_KEYS or key.startswith("path_"):
            continue
        template_value = template_state_arrays.get(key)
        value = grad_state_arrays.get(key)
        scattered_value = _scatter_grad_value(template_value, value, indices, n_states)
        if scattered_value is not None:
            scattered[key] = scattered_value
    return scattered


class _PackedStateGatherLeafOp(dr.CustomOp):
    def eval(self, source_value, indices, primal_value):
        self._source_type = type(source_value)
        self._source_width = dr.width(source_value)
        self._indices = indices
        return primal_value

    def forward(self):
        grad_source = self.grad_in("source_value")
        if grad_source is None:
            return
        self.set_grad_out(dr.gather(self._source_type, grad_source, self._indices))

    def backward(self):
        grad_out = self.grad_out()
        if grad_out is None:
            return
        grad_source = dr.zeros(self._source_type, self._source_width)
        dr.scatter_reduce(dr.ReduceOp.Add, grad_source, grad_out, self._indices)
        self.set_grad_in("source_value", grad_source)


def _attach_native_gather_grad(primal_value, source_value, indices):
    if primal_value is None or source_value is None:
        return primal_value
    if _is_vector_like(primal_value):
        return type(primal_value)(
            _attach_native_gather_grad(primal_value.x, source_value.x, indices),
            _attach_native_gather_grad(primal_value.y, source_value.y, indices),
            _attach_native_gather_grad(primal_value.z, source_value.z, indices),
        )
    if _is_complex_like(primal_value):
        return wt.Complex2f(
            _attach_native_gather_grad(primal_value.real, source_value.real, indices),
            _attach_native_gather_grad(primal_value.imag, source_value.imag, indices),
        )
    return dr.custom(_PackedStateGatherLeafOp, source_value, indices, primal_value)


def _attach_native_gather_grad_state(state_arrays: dict, gathered: dict, indices) -> dict:
    for key in STATE_STATIC_KEYS:
        if (
            key not in gathered
            or key not in state_arrays
            or key in _NON_DIFFERENTIABLE_STATE_KEYS
            or key.startswith("path_")
        ):
            continue
        gathered[key] = _attach_native_gather_grad(gathered[key], state_arrays[key], indices)
    return gathered


def _attach_native_inserted_reflection_gather_grad(
    state_arrays: dict,
    gathered: dict,
    indices,
) -> dict:
    for key in _INSERTED_REFLECTION_GATHER_GRAD_KEYS:
        if key not in gathered or key not in state_arrays:
            continue
        if not _value_has_grad(state_arrays[key]):
            continue
        gathered[key] = _attach_native_gather_grad(gathered[key], state_arrays[key], indices)
    return gathered


def _gather_optional_cold(state_arrays: dict, source_state_arrays: dict, indices) -> None:
    for key, dtype in _OPTIONAL_COLD_SPECS:
        if key in source_state_arrays:
            state_arrays[key] = dr.gather(dtype, source_state_arrays[key], indices)


def _concat_optional_cold(state_arrays: dict, non_empty: list[dict]) -> None:
    from witwin.channel.core.numerics.arrays import concat_arrays

    for key, dtype in _OPTIONAL_COLD_SPECS:
        if not any(key in source for source in non_empty):
            continue
        parts = []
        for source in non_empty:
            width = int(source[SK_N_STATES])
            if key in source:
                parts.append(source[key])
            elif dtype is wt.Float:
                parts.append(dr.zeros(wt.Float, width))
            elif dtype is wt.Point3f:
                zeros = dr.zeros(wt.Float, width)
                parts.append(wt.Point3f(zeros, zeros, zeros))
            elif dtype is wt.Bool:
                parts.append(dr.zeros(wt.Bool, width))
            else:
                parts.append(dr.zeros(dtype, width))
        if dtype is wt.Point3f:
            state_arrays[key] = wt.Point3f(
                concat_arrays(wt.Float, [part.x for part in parts]),
                concat_arrays(wt.Float, [part.y for part in parts]),
                concat_arrays(wt.Float, [part.z for part in parts]),
            )
        else:
            state_arrays[key] = concat_arrays(dtype, parts)


def _cached_packed_buffer(state_arrays: dict, ext, history_size: int):
    packed = state_arrays.get(_PACKED_BUFFER_KEY)
    if packed is None:
        return None
    stride = _packed_stride(ext, history_size)
    if state_arrays.get(_PACKED_HISTORY_SIZE_KEY) != history_size:
        return None
    if state_arrays.get(_PACKED_STRIDE_KEY) != stride:
        return None
    if dr.width(packed) != int(state_arrays[SK_N_STATES]) * stride:
        return None
    return packed


def _attach_packed_buffer(state_arrays: dict, packed, history_size: int, ext) -> dict:
    state_arrays[_PACKED_BUFFER_KEY] = packed
    state_arrays[_PACKED_HISTORY_SIZE_KEY] = history_size
    state_arrays[_PACKED_STRIDE_KEY] = _packed_stride(ext, history_size)
    return state_arrays


def _inserted_reflection_state_fields_from_tuple(
    raw_outputs,
    n_states: int,
    history_size: int,
    *,
    source_state_arrays: dict | None = None,
    indices=None,
):
    (
        edge_pos_x,
        edge_pos_y,
        edge_pos_z,
        prefix_refl_depth,
        inter_refl_depth,
        suffix_refl_depth,
        order,
    ) = raw_outputs
    selected = {
        SK_EDGE_POS: wt.Point3f(edge_pos_x, edge_pos_y, edge_pos_z),
        SK_PREFIX_REFLECTION_DEPTH: prefix_refl_depth,
        SK_INTERMEDIATE_REFLECTION_DEPTH: inter_refl_depth,
        SK_SUFFIX_REFLECTION_DEPTH: suffix_refl_depth,
        SK_ORDER: order,
        SK_N_STATES: n_states,
    }
    lineage = None
    if source_state_arrays is not None and indices is not None:
        if SK_PATH_LENGTH_PREFIX in source_state_arrays:
            selected[SK_PATH_LENGTH_PREFIX] = dr.gather(
                wt.Float,
                source_state_arrays[SK_PATH_LENGTH_PREFIX],
                indices,
            )
        if SK_FIRST_INTERACTION_POS in source_state_arrays:
            selected[SK_FIRST_INTERACTION_POS] = dr.gather(
                wt.Point3f,
                source_state_arrays[SK_FIRST_INTERACTION_POS],
                indices,
            )
        if SK_SOURCE_TYPE_CODE in source_state_arrays:
            selected[SK_SOURCE_TYPE_CODE] = dr.gather(
                wt.UInt32,
                source_state_arrays[SK_SOURCE_TYPE_CODE],
                indices,
            )
        lineage = State.gather_lineage(source_state_arrays, indices)
    return State.attach_lineage(selected, lineage, history_size)


def _pack_eval_arrays(state_arrays: dict):
    return _pack_core_arrays(state_arrays)


def _pack_core_arrays(state_arrays: dict) -> list:
    return [
        state_arrays[SK_EDGE_IDX],
        state_arrays[SK_EDGE_POS].x, state_arrays[SK_EDGE_POS].y, state_arrays[SK_EDGE_POS].z,
        state_arrays[SK_EDGE_DIR].x, state_arrays[SK_EDGE_DIR].y, state_arrays[SK_EDGE_DIR].z,
        state_arrays[SK_N0].x, state_arrays[SK_N0].y, state_arrays[SK_N0].z,
        state_arrays[SK_NN].x, state_arrays[SK_NN].y, state_arrays[SK_NN].z,
        state_arrays[SK_WEDGE_N],
        state_arrays[SK_ADJACENT_FACE0], state_arrays[SK_ADJACENT_FACE1],
        state_arrays[SK_SOURCE_POS].x, state_arrays[SK_SOURCE_POS].y, state_arrays[SK_SOURCE_POS].z,
        state_arrays[SK_INCIDENT_FIELD].real, state_arrays[SK_INCIDENT_FIELD].imag,
        state_arrays[SK_INCIDENT_NORMAL_DERIVATIVE].real,
        state_arrays[SK_INCIDENT_NORMAL_DERIVATIVE].imag,
        state_arrays[SK_INCIDENT_JONES_U].real, state_arrays[SK_INCIDENT_JONES_U].imag,
        state_arrays[SK_INCIDENT_JONES_V].real, state_arrays[SK_INCIDENT_JONES_V].imag,
        state_arrays[SK_INCIDENT_DERIVATIVE_JONES_U].real,
        state_arrays[SK_INCIDENT_DERIVATIVE_JONES_U].imag,
        state_arrays[SK_INCIDENT_DERIVATIVE_JONES_V].real,
        state_arrays[SK_INCIDENT_DERIVATIVE_JONES_V].imag,
        state_arrays[SK_R0].real, state_arrays[SK_R0].imag,
        state_arrays[SK_RN].real, state_arrays[SK_RN].imag,
        state_arrays[SK_INCIDENT_BASIS_U].x, state_arrays[SK_INCIDENT_BASIS_U].y, state_arrays[SK_INCIDENT_BASIS_U].z,
        state_arrays[SK_INCIDENT_BASIS_V].x, state_arrays[SK_INCIDENT_BASIS_V].y, state_arrays[SK_INCIDENT_BASIS_V].z,
        state_arrays[SK_INCIDENT_BASIS_K].x, state_arrays[SK_INCIDENT_BASIS_K].y, state_arrays[SK_INCIDENT_BASIS_K].z,
        state_arrays[SK_FACE0_OPERATOR_M00].real, state_arrays[SK_FACE0_OPERATOR_M00].imag,
        state_arrays[SK_FACE0_OPERATOR_M01].real, state_arrays[SK_FACE0_OPERATOR_M01].imag,
        state_arrays[SK_FACE0_OPERATOR_M10].real, state_arrays[SK_FACE0_OPERATOR_M10].imag,
        state_arrays[SK_FACE0_OPERATOR_M11].real, state_arrays[SK_FACE0_OPERATOR_M11].imag,
        state_arrays[SK_FACE1_OPERATOR_M00].real, state_arrays[SK_FACE1_OPERATOR_M00].imag,
        state_arrays[SK_FACE1_OPERATOR_M01].real, state_arrays[SK_FACE1_OPERATOR_M01].imag,
        state_arrays[SK_FACE1_OPERATOR_M10].real, state_arrays[SK_FACE1_OPERATOR_M10].imag,
        state_arrays[SK_FACE1_OPERATOR_M11].real, state_arrays[SK_FACE1_OPERATOR_M11].imag,
        state_arrays[SK_FACE0_ETA_R], state_arrays[SK_FACE0_MU_R],
        state_arrays[SK_FACE0_SIGMA], state_arrays[SK_FACE0_GAIN],
        state_arrays[SK_FACE0_USE_FRESNEL],
        state_arrays[SK_FACE1_ETA_R], state_arrays[SK_FACE1_MU_R],
        state_arrays[SK_FACE1_SIGMA], state_arrays[SK_FACE1_GAIN],
        state_arrays[SK_FACE1_USE_FRESNEL],
        state_arrays[SK_PREFIX_REFLECTION_DEPTH],
        state_arrays[SK_INTERMEDIATE_REFLECTION_DEPTH],
        state_arrays[SK_SUFFIX_REFLECTION_DEPTH],
        state_arrays[SK_ORDER],
    ]


def _state_arrays_from_unpacked_lists(
    core_arrays,
    n_states: int,
    packed,
    history_size: int,
    ext,
    *,
    cache_packed: bool = False,
) -> dict:
    core = list(core_arrays)
    index = 0

    def take():
        nonlocal index
        value = core[index]
        index += 1
        return value

    state_arrays = {
        SK_EDGE_IDX: take(),
        SK_EDGE_POS: wt.Point3f(take(), take(), take()),
        SK_EDGE_DIR: wt.Vector3f(take(), take(), take()),
        SK_N0: wt.Vector3f(take(), take(), take()),
        SK_NN: wt.Vector3f(take(), take(), take()),
        SK_WEDGE_N: take(),
        SK_ADJACENT_FACE0: take(),
        SK_ADJACENT_FACE1: take(),
        SK_SOURCE_POS: wt.Point3f(take(), take(), take()),
        SK_INCIDENT_FIELD: wt.Complex2f(take(), take()),
        SK_INCIDENT_NORMAL_DERIVATIVE: wt.Complex2f(take(), take()),
        SK_INCIDENT_JONES_U: wt.Complex2f(take(), take()),
        SK_INCIDENT_JONES_V: wt.Complex2f(take(), take()),
        SK_INCIDENT_DERIVATIVE_JONES_U: wt.Complex2f(take(), take()),
        SK_INCIDENT_DERIVATIVE_JONES_V: wt.Complex2f(take(), take()),
        SK_R0: wt.Complex2f(take(), take()),
        SK_RN: wt.Complex2f(take(), take()),
        SK_INCIDENT_BASIS_U: wt.Vector3f(take(), take(), take()),
        SK_INCIDENT_BASIS_V: wt.Vector3f(take(), take(), take()),
        SK_INCIDENT_BASIS_K: wt.Vector3f(take(), take(), take()),
        SK_FACE0_OPERATOR_M00: wt.Complex2f(take(), take()),
        SK_FACE0_OPERATOR_M01: wt.Complex2f(take(), take()),
        SK_FACE0_OPERATOR_M10: wt.Complex2f(take(), take()),
        SK_FACE0_OPERATOR_M11: wt.Complex2f(take(), take()),
        SK_FACE1_OPERATOR_M00: wt.Complex2f(take(), take()),
        SK_FACE1_OPERATOR_M01: wt.Complex2f(take(), take()),
        SK_FACE1_OPERATOR_M10: wt.Complex2f(take(), take()),
        SK_FACE1_OPERATOR_M11: wt.Complex2f(take(), take()),
        SK_FACE0_ETA_R: take(),
        SK_FACE0_MU_R: take(),
        SK_FACE0_SIGMA: take(),
        SK_FACE0_GAIN: take(),
        SK_FACE0_USE_FRESNEL: take() != 0.0,
        SK_FACE1_ETA_R: take(),
        SK_FACE1_MU_R: take(),
        SK_FACE1_SIGMA: take(),
        SK_FACE1_GAIN: take(),
        SK_FACE1_USE_FRESNEL: take() != 0.0,
        SK_PREFIX_REFLECTION_DEPTH: take(),
        SK_INTERMEDIATE_REFLECTION_DEPTH: take(),
        SK_SUFFIX_REFLECTION_DEPTH: take(),
        SK_ORDER: take(),
        SK_N_STATES: n_states,
    }
    incident_basis = {
        "u": state_arrays[SK_INCIDENT_BASIS_U],
        "v": state_arrays[SK_INCIDENT_BASIS_V],
        "k": state_arrays[SK_INCIDENT_BASIS_K],
    }
    incident_jones = {
        "u": state_arrays[SK_INCIDENT_JONES_U],
        "v": state_arrays[SK_INCIDENT_JONES_V],
    }
    incident_derivative_jones = {
        "u": state_arrays[SK_INCIDENT_DERIVATIVE_JONES_U],
        "v": state_arrays[SK_INCIDENT_DERIVATIVE_JONES_V],
    }
    incident_vector = vector_from_jones(incident_jones, incident_basis)
    incident_derivative_vector = vector_from_jones(
        incident_derivative_jones,
        incident_basis,
    )
    state_arrays[SK_INCIDENT_VECTOR_X] = incident_vector["x"]
    state_arrays[SK_INCIDENT_VECTOR_Y] = incident_vector["y"]
    state_arrays[SK_INCIDENT_VECTOR_Z] = incident_vector["z"]
    state_arrays[SK_INCIDENT_NORMAL_DERIVATIVE_VECTOR_X] = incident_derivative_vector["x"]
    state_arrays[SK_INCIDENT_NORMAL_DERIVATIVE_VECTOR_Y] = incident_derivative_vector["y"]
    state_arrays[SK_INCIDENT_NORMAL_DERIVATIVE_VECTOR_Z] = incident_derivative_vector["z"]
    if index != len(core):
        raise RuntimeError("packed_state unpack returned an unexpected core array count")
    State.attach_lineage(state_arrays, None, history_size)
    if cache_packed:
        return _attach_packed_buffer(state_arrays, packed, history_size, ext)
    return state_arrays


def _unpack_state_arrays(
    packed,
    history_size: int,
    ext,
    *,
    retain_lineage_state: bool = False,
    cache_packed: bool = False,
) -> dict:
    stride = _packed_stride(ext, history_size)
    n_states = dr.width(packed) // stride
    if n_states == 0:
        return State.empty(
            history_size=history_size,
            retain_lineage_state=retain_lineage_state,
        )

    core_arrays = ext.unpack_state_arrays_raw(
        packed,
        n_states,
        stride,
    )
    dr.eval(*core_arrays)
    return _state_arrays_from_unpacked_lists(
        core_arrays,
        n_states,
        packed,
        history_size,
        ext,
        cache_packed=cache_packed,
    )


def _pack_state_arrays(state_arrays: dict, ext, history_size: int | None = None):
    history_size_in = Geo.history_size(state_arrays)
    if history_size is None:
        history_size = history_size_in
    cached = _cached_packed_buffer(state_arrays, ext, history_size)
    if cached is not None:
        return cached, history_size, _packed_stride(ext, history_size)

    n_states = int(state_arrays[SK_N_STATES])
    stride = _packed_stride(ext, history_size)
    dr.eval(*_pack_eval_arrays(state_arrays))
    packed = ext.pack_state_arrays_raw(
        _pack_core_arrays(state_arrays),
        n_states,
        stride,
    )
    dr.eval(packed)
    _attach_packed_buffer(state_arrays, packed, history_size, ext)
    return packed, history_size, stride


def _can_use_native_primal(state_arrays: dict, history_size: int) -> tuple[bool, object | None]:
    ext = _native_extension_or_none()
    if ext is None:
        return False, None
    if int(state_arrays[SK_N_STATES]) == 0:
        return True, ext
    return True, ext


def _can_use_native(state_arrays: dict, history_size: int) -> tuple[bool, object | None]:
    can_use_primal, ext = _can_use_native_primal(state_arrays, history_size)
    if not can_use_primal:
        return False, ext
    if _state_arrays_has_grad(state_arrays):
        return False, ext
    return True, ext


def _native_gather_state_arrays_primal(state_arrays: dict, indices, ext) -> dict:
    history_size = Geo.history_size(state_arrays)
    n_out = dr.width(indices)
    if n_out == 0:
        return State.empty(
            history_size=history_size,
            retain_lineage_state=State.has_lineage_state(state_arrays),
        )

    packed_src, history_size, stride = _pack_state_arrays(state_arrays, ext, history_size)
    indices_i32 = dr.detach(wt.Int32(indices))
    dr.eval(indices_i32)
    packed_dst = ext.gather_packed_states_raw(packed_src, indices_i32, n_out, stride)
    dr.eval(packed_dst)
    gathered = _unpack_state_arrays(packed_dst, history_size, ext)
    _gather_optional_cold(gathered, state_arrays, indices)
    return State.attach_lineage(
        gathered,
        State.gather_lineage(state_arrays, indices),
        history_size,
    )


class _PackedStateGatherOp(dr.CustomOp):
    def eval(self, state_arrays: dict, indices):
        self._state_arrays = state_arrays
        self._input_state_arrays = _strip_packed_lineage(state_arrays)
        self._indices = indices
        history_size = Geo.history_size(state_arrays)
        can_use_native, ext = _can_use_native_primal(state_arrays, history_size)
        if not can_use_native:
            self._ext = None
            self._output_state_arrays = _strip_packed_lineage(_drjit_gather(state_arrays, indices))
            return self._output_state_arrays
        self._ext = ext
        self._output_state_arrays = _strip_packed_lineage(
            _native_gather_state_arrays_primal(state_arrays, indices, ext)
        )
        return self._output_state_arrays

    def forward(self):
        self.set_grad_out(
            _gather_state_grad_dict(
                self._output_state_arrays,
                self.grad_in("state_arrays"),
                self._indices,
            )
        )

    def backward(self):
        self.set_grad_in(
            "state_arrays",
            _scatter_state_grad_dict(self._input_state_arrays, self.grad_out(), self._indices),
        )


def gather_state_arrays(state_arrays: dict, indices) -> dict:
    history_size = 0 if state_arrays is None else Geo.history_size(state_arrays)
    if state_arrays is None or int(state_arrays[SK_N_STATES]) == 0:
        return State.empty(
            history_size=history_size,
            retain_lineage_state=State.has_lineage_state(state_arrays),
        )

    can_use_native_primal, ext = _can_use_native_primal(state_arrays, history_size)
    if not can_use_native_primal:
        return _drjit_gather(state_arrays, indices)

    n_out = dr.width(indices)
    if n_out == 0:
        return State.empty(
            history_size=history_size,
            retain_lineage_state=State.has_lineage_state(state_arrays),
        )

    if _state_arrays_has_grad(state_arrays):
        gathered = _native_gather_state_arrays_primal(state_arrays, indices, ext)
        return _attach_native_gather_grad_state(state_arrays, gathered, indices)
    return _native_gather_state_arrays_primal(state_arrays, indices, ext)


def gather_inserted_reflection_state_fields(state_arrays: dict, indices) -> dict:
    history_size = 0 if state_arrays is None else Geo.history_size(state_arrays)
    if state_arrays is None or int(state_arrays[SK_N_STATES]) == 0:
        return _empty_inserted_reflection_state_fields(
            history_size,
            retain_lineage_state=State.has_lineage_state(state_arrays),
        )

    can_use_native_primal, ext = _can_use_native_primal(state_arrays, history_size)
    if not can_use_native_primal:
        return _drjit_gather_inserted_fields(state_arrays, indices)

    n_out = dr.width(indices)
    if n_out == 0:
        return _empty_inserted_reflection_state_fields(
            history_size,
            retain_lineage_state=State.has_lineage_state(state_arrays),
        )

    packed_src, history_size, stride = _pack_state_arrays(state_arrays, ext, history_size)
    indices_i32 = dr.detach(wt.Int32(indices))
    dr.eval(indices_i32)
    raw_outputs = ext.gather_inserted_reflection_state_fields_raw(
        packed_src,
        indices_i32,
        n_out,
        stride,
    )
    dr.eval(*raw_outputs)
    gathered = _inserted_reflection_state_fields_from_tuple(
        raw_outputs,
        n_out,
        history_size,
        source_state_arrays=state_arrays,
        indices=indices,
    )
    has_inserted_grad = _state_arrays_has_grad(state_arrays) or any(
        _value_has_grad(state_arrays.get(key))
        for key in (SK_PATH_LENGTH_PREFIX, SK_FIRST_INTERACTION_POS)
    )
    if has_inserted_grad:
        return _attach_native_inserted_reflection_gather_grad(
            state_arrays,
            gathered,
            indices,
        )
    return gathered


def gather_field_evaluation_state_fields(state_arrays: dict, indices, *, include_stored_operators: bool = False) -> dict:
    return _drjit_gather_field_eval(
        state_arrays,
        indices,
        include_stored_operators=include_stored_operators,
    )


def _path_slot_geometry_has_grad(keep_states: dict, edge_data) -> bool:
    if SK_FIRST_INTERACTION_POS in keep_states and _value_has_grad(
        keep_states[SK_FIRST_INTERACTION_POS]
    ):
        return True
    if edge_data is None:
        return False
    if "pos" in edge_data and _value_has_grad(edge_data["pos"]):
        return True
    if "n0" in edge_data and _value_has_grad(edge_data["n0"]):
        return True
    return False


def build_diffraction_path_slots(
    *,
    keep_states,
    edge_data,
    edge_object_idx,
    return_geometry: bool,
):
    history_size = Geo.history_size(keep_states)
    count = int(keep_states["n_states"])
    if count <= 0:
        return _drjit_build_diffraction_path_slots(
            keep_states=keep_states,
            edge_data=edge_data,
            edge_object_idx=edge_object_idx,
            return_geometry=return_geometry,
        )

    ext = _native_extension_or_none()
    if ext is None:
        return _drjit_build_diffraction_path_slots(
            keep_states=keep_states,
            edge_data=edge_data,
            edge_object_idx=edge_object_idx,
            return_geometry=return_geometry,
        )

    materialized_edge_slots, materialized_reflection_depth_slots = State.history(
        keep_states
    )
    order = wt.Int32(keep_states[SK_ORDER])
    prefix_depth = (
        wt.Int32(materialized_reflection_depth_slots[0])
        if history_size > 0 and len(materialized_reflection_depth_slots) > 0
        else dr.zeros(wt.Int32, count)
    )
    inserted_depth_slots = [
        wt.Int32(materialized_reflection_depth_slots[slot])
        for slot in range(1, min(history_size, len(materialized_reflection_depth_slots)))
    ]
    path_edge_slots = [
        wt.Int32(materialized_edge_slots[slot])
        for slot in range(min(history_size, len(materialized_edge_slots)))
    ]
    total_depth = prefix_depth + order
    for slot, inserted_depth in enumerate(inserted_depth_slots):
        active = order > wt.Int32(slot + 1)
        total_depth = total_depth + dr.select(active, inserted_depth, wt.Int32(0))
    max_depth = max(1, int(scalar(dr.max(total_depth))))

    if (
        max_depth > int(getattr(ext, "DIFFRACTION_PATH_SLOT_MAX_DEPTH", 0))
        or (return_geometry and _path_slot_geometry_has_grad(keep_states, edge_data))
    ):
        return _drjit_build_diffraction_path_slots(
            keep_states=keep_states,
            edge_data=edge_data,
            edge_object_idx=edge_object_idx,
            return_geometry=return_geometry,
        )

    first_interaction_pos = keep_states.get(SK_FIRST_INTERACTION_POS)
    edge_pos = None if edge_data is None else edge_data.get("pos")
    edge_normal = None if edge_data is None else edge_data.get("n0")
    n_edges = 0 if edge_data is None else int(edge_data["n_edges"])

    dr.eval(
        prefix_depth,
        order,
        *path_edge_slots,
        *inserted_depth_slots,
    )
    raw_outputs = ext.build_diffraction_path_slots_raw(
        prefix_depth,
        order,
        path_edge_slots,
        inserted_depth_slots,
        history_size,
        count,
        max_depth,
        bool(return_geometry),
        None if first_interaction_pos is None else first_interaction_pos.x,
        None if first_interaction_pos is None else first_interaction_pos.y,
        None if first_interaction_pos is None else first_interaction_pos.z,
        None if edge_pos is None else edge_pos.x,
        None if edge_pos is None else edge_pos.y,
        None if edge_pos is None else edge_pos.z,
        None if edge_normal is None else edge_normal.x,
        None if edge_normal is None else edge_normal.y,
        None if edge_normal is None else edge_normal.z,
        edge_object_idx,
        n_edges,
        int(InteractionType.REFLECTION),
        int(InteractionType.DIFFRACTION),
    )
    (
        type_slots,
        vertex_x_slots,
        vertex_y_slots,
        vertex_z_slots,
        normal_x_slots,
        normal_y_slots,
        normal_z_slots,
        object_slots,
    ) = raw_outputs
    vertex_slots = None
    normal_slots = None
    if return_geometry:
        vertex_slots = [
            wt.Point3f(vertex_x_slots[depth], vertex_y_slots[depth], vertex_z_slots[depth])
            for depth in range(max_depth)
        ]
        normal_slots = [
            wt.Vector3f(normal_x_slots[depth], normal_y_slots[depth], normal_z_slots[depth])
            for depth in range(max_depth)
        ]
        if history_size > 0 and first_interaction_pos is not None:
            prefix_mask = prefix_depth > 0
            vertex_slots[0] = dr.select(prefix_mask, first_interaction_pos, vertex_slots[0])
    return type_slots, vertex_slots, normal_slots, object_slots, max_depth


def concat_state_arrays(state_arrays_list: list[dict]) -> dict:
    non_empty = [
        state_arrays
        for state_arrays in state_arrays_list
        if state_arrays is not None and int(state_arrays[SK_N_STATES]) > 0
    ]
    if len(non_empty) == 0:
        history_size = 0
        retain_cold = False
        for state_arrays in state_arrays_list:
            if state_arrays is not None:
                history_size = max(history_size, Geo.history_size(state_arrays))
                retain_cold = retain_cold or State.has_lineage_state(state_arrays)
        return State.empty(
            history_size=history_size,
            retain_lineage_state=retain_cold,
        )
    if len(non_empty) == 1:
        return non_empty[0]

    history_size = max(Geo.history_size(state_arrays) for state_arrays in non_empty)
    ext = None
    for state_arrays in non_empty:
        can_use_native, ext = _can_use_native(state_arrays, history_size)
        if not can_use_native:
            return _drjit_concat(state_arrays_list)

    total_states = sum(int(state_arrays[SK_N_STATES]) for state_arrays in non_empty)
    packed_inputs = [_pack_state_arrays(state_arrays, ext, history_size)[0] for state_arrays in non_empty]
    stride = _packed_stride(ext, history_size)
    dr.eval(*packed_inputs)
    packed_dst = ext.concat_packed_states_raw(
        packed_inputs,
        [int(state_arrays[SK_N_STATES]) for state_arrays in non_empty],
        stride,
    )
    dr.eval(packed_dst)
    state_arrays = _unpack_state_arrays(packed_dst, history_size, ext)
    _concat_optional_cold(state_arrays, non_empty)
    return State.attach_lineage(
        state_arrays,
        State.concat_lineage(non_empty),
        history_size,
    )


def subset_state_arrays(state_arrays: dict, mask) -> dict:
    history_size = 0 if state_arrays is None else Geo.history_size(state_arrays)
    if state_arrays is None or int(state_arrays[SK_N_STATES]) == 0:
        return State.empty(
            history_size=history_size,
            retain_lineage_state=State.has_lineage_state(state_arrays),
        )

    can_use_native_primal, _ = _can_use_native_primal(state_arrays, history_size)
    if not can_use_native_primal:
        return _drjit_subset(state_arrays, mask)

    keep_idx = dr.compress(mask)
    if dr.width(keep_idx) == 0:
        return State.empty(
            history_size=history_size,
            retain_lineage_state=State.has_lineage_state(state_arrays),
        )
    gathered = gather_state_arrays(state_arrays, keep_idx)
    gathered[SK_N_STATES] = dr.width(keep_idx)
    return gathered
