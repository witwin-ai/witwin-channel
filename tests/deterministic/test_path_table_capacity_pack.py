from __future__ import annotations

import inspect
from dataclasses import fields
from pathlib import Path

import pytest
import torch

from witwin.channel_native.deterministic.accumulation import build_path_table
from witwin.channel_native.deterministic.capacity import (
    deterministic_path_table_capacity_pack,
)
from witwin.channel_native.deterministic import capacity as capacity_module
from witwin.channel_native.deterministic.result import PathTable
from witwin.channel_native.propagation.models.capacity import (
    CapacityEvaluatedPaths,
    CapacityPathLayout,
    CapacityPathSelection,
)
from witwin.channel_native.propagation.models.evaluated import EvaluatedPaths
from witwin.channel_native.propagation.models.fields import PathFields
from witwin.channel_native.propagation.models.geometry import PathGeometry
from witwin.channel_native.propagation.models.topology import PathTopology
from witwin.channel_native.runtime import symbols
from witwin.channel_native.runtime.capacity import create_capacity_failure_state
from witwin.channel_native.runtime.symbols import required_symbol


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is required"
)

_CONTINUOUS = (
    "path_length_m",
    "delay_s",
    "field_direction",
    "interaction_position",
    "interaction_normal",
    "interaction_positions",
    "interaction_normals",
    "path_gain",
    "path_field",
    "field_xyz",
    "coefficient",
)

_ROOT = Path(__file__).resolve().parents[2]


def test_path_table_capacity_pack_function_input_arity_and_indices_are_frozen() -> None:
    names = capacity_module._FUNCTION_INPUT_FIELDS
    assert len(names) == 27
    assert names[0] == "failure_state"
    assert names[1:11] == capacity_module._TABLE_INPUT_FIELDS
    assert names[11:22] == capacity_module._CONTINUOUS_INPUT_FIELDS
    assert names[22:] == (
        "num_paths",
        "overflow",
        "include_fields",
        "pair_count",
        "path_capacity_per_pair",
    )
    assert capacity_module._CONTINUOUS_INPUT_SLICE == slice(11, 22)
    function = capacity_module._DeterministicPathTableCapacityPackFunction
    assert capacity_module._required_native_op is symbols.required_symbol
    assert function.forward.__globals__ is capacity_module.__dict__
    assert inspect.unwrap(function.backward).__globals__ is capacity_module.__dict__
    assert function.jvp.__globals__ is capacity_module.__dict__


def test_path_table_capacity_phase_duplicate_is_explicitly_locked() -> None:
    live = (_ROOT / "native/channel_native/kernels/deterministic_field.cu").read_text(
        encoding="utf-8"
    )
    capacity = (
        _ROOT / "native/channel_native/kernels/deterministic_path_table_capacity_pack.cu"
    ).read_text(encoding="utf-8")
    assert "float phase = -atan2f(field_imag[index], field_real[index]);" in live
    assert "phase = fmodf(phase, 2.0f * kPi);" in live
    assert "float phase = -atan2f(value.imag(), value.real());" in capacity
    assert "phase = fmodf(phase, 2.0f * kPi);" in capacity
    assert "Numerical deduplication requires a separate change." in capacity


def _capacity(*, differentiable: bool = False) -> tuple[CapacityEvaluatedPaths, dict[str, torch.Tensor]]:
    rows, width = 6, 2
    valid = torch.tensor([True, False, True, True, False, False], device="cuda")
    invalid = ~valid

    def discrete(start: int) -> torch.Tensor:
        value = torch.arange(start, start + rows, device="cuda", dtype=torch.int32)
        value[invalid] = 777
        return value.contiguous()

    def real(shape: tuple[int, ...], start: float) -> torch.Tensor:
        count = 1
        for extent in shape:
            count *= extent
        value = (torch.arange(count, device="cuda", dtype=torch.float32) + start).reshape(shape)
        value[invalid] = torch.nan
        return value.contiguous()

    def complex_value(shape: tuple[int, ...], start: float) -> torch.Tensor:
        re = real(shape, start)
        im = real(shape, -start - 3.0)
        return torch.complex(re, im).contiguous()

    primitive_sequence = torch.arange(rows * width, device="cuda", dtype=torch.int32).reshape(rows, width)
    material_sequence = primitive_sequence + 100
    interaction_type = primitive_sequence + 200
    primitive_sequence[invalid] = 777
    material_sequence[invalid] = 777
    interaction_type[invalid] = 777
    topology = PathTopology(
        valid=valid,
        tx_id=discrete(0),
        rx_id=discrete(10),
        depth=discrete(1),
        component_id=discrete(20),
        primitive_id=discrete(30),
        edge_id=discrete(40),
        material_id=discrete(50),
        primitive_sequence=primitive_sequence.contiguous(),
        material_sequence=material_sequence.contiguous(),
        interaction_type=interaction_type.contiguous(),
    )
    values = {
        "path_length_m": real((rows,), 0.25),
        "delay_s": real((rows,), 1.25),
        "field_direction": real((rows, 3), 2.25),
        "interaction_position": real((rows, 3), 3.25),
        "interaction_normal": real((rows, 3), 4.25),
        "interaction_positions": real((rows, width, 3), 5.25),
        "interaction_normals": real((rows, width, 3), 6.25),
        "path_gain": real((rows,), 7.25),
        "path_field": complex_value((rows,), 8.25),
        "field_xyz": complex_value((rows, 3), 9.25),
        "coefficient": complex_value((rows,), 10.25),
    }
    path_field_parts = torch.view_as_real(values["path_field"])
    path_field_parts[0] = torch.tensor([0.0, 0.0], device="cuda")
    path_field_parts[2] = torch.tensor([-0.0, 0.0], device="cuda")
    path_field_parts[3] = torch.tensor([0.0, -0.0], device="cuda")
    if differentiable:
        values = {
            name: value.detach().clone().requires_grad_(True)
            for name, value in values.items()
        }
    geometry = PathGeometry(
        row_identity=topology.row_identity,
        **{name: values[name] for name in _CONTINUOUS[:7]},
    )
    path_fields = PathFields(
        row_identity=topology.row_identity,
        **{name: values[name] for name in _CONTINUOUS[7:]},
    )
    failure_state = create_capacity_failure_state(valid)
    layout = CapacityPathLayout(
        pair_count=2,
        path_capacity_per_pair=3,
        failure_state=failure_state,
        valid=valid,
        num_paths=torch.tensor([2, 1], device="cuda", dtype=torch.int32),
        overflow=torch.zeros(1, device="cuda", dtype=torch.bool),
    )
    selection = CapacityPathSelection(
        selected_row_index=torch.where(
            valid,
            torch.arange(rows, device="cuda", dtype=torch.int64),
            torch.full((rows,), -1, device="cuda", dtype=torch.int64),
        ),
        layout=layout,
    )
    evaluated = EvaluatedPaths(topology=topology, geometry=geometry, fields=path_fields)
    return CapacityEvaluatedPaths(selection=selection, evaluated=evaluated), values


def _assert_bits_equal(actual: torch.Tensor, expected: torch.Tensor) -> None:
    assert actual.dtype == expected.dtype
    assert actual.shape == expected.shape
    assert torch.equal(
        actual.contiguous().view(torch.uint8), expected.contiguous().view(torch.uint8)
    )


def _assert_positive_zero_bits(value: torch.Tensor) -> None:
    assert torch.count_nonzero(value.contiguous().view(torch.uint8)).item() == 0


def _direct_outputs(
    capacity: CapacityEvaluatedPaths, *, include_fields: bool = True
) -> dict[str, torch.Tensor]:
    layout = capacity.selection.layout
    raw = required_symbol("deterministic_path_table_capacity_pack")(
        layout.failure_state.bits,
        *capacity_module._path_table_inputs(capacity),
        layout.num_paths,
        layout.overflow,
        include_fields,
        layout.pair_count,
        layout.path_capacity_per_pair,
    )
    assert isinstance(raw, dict)
    return raw


def _assert_canonical_inert(packed) -> None:
    table = packed.table
    assert torch.count_nonzero(packed.valid).item() == 0
    assert torch.count_nonzero(packed.num_paths).item() == 0
    for name in (
        "tx_id",
        "rx_id",
        "component_id",
        "primitive_id",
        "edge_id",
        "material_id",
        "primitive_sequence",
        "material_sequence",
    ):
        assert bool(torch.all(getattr(table, name) == -1)), name
    for name in ("depth", "interaction_count"):
        assert torch.count_nonzero(getattr(table, name)).item() == 0, name
    for name in ("path_length_m", "delay_s"):
        value = getattr(table, name)
        assert bool(torch.all(torch.isfinite(value))), name
        assert bool(torch.all(value == -1.0)), name
    for name in (
        "path_gain",
        "interaction_position",
        "interaction_normal",
        "interaction_positions",
        "interaction_normals",
        "field_real",
        "field_imag",
        "coefficient",
        "field_xyz",
        "field_direction",
        "phase_rad",
    ):
        value = getattr(table, name)
        assert bool(torch.all(torch.isfinite(value))), name
        _assert_positive_zero_bits(value)


@pytest.mark.parametrize("include_fields", (True, False))
def test_path_table_capacity_pack_matches_live_export_bitwise(include_fields: bool) -> None:
    capacity, _ = _capacity()
    packed = deterministic_path_table_capacity_pack(capacity, include_fields=include_fields)
    baseline = build_path_table(
        capacity.evaluated, frequency_hz=28.0e9, include_fields=include_fields
    )
    valid = capacity.selection.valid
    for descriptor in fields(PathTable):
        name = descriptor.name
        _assert_bits_equal(getattr(packed.table, name)[valid], getattr(baseline, name)[valid])

    invalid = ~packed.valid
    assert packed.pair_count == 2
    assert packed.path_capacity_per_pair == 3
    assert packed.row_capacity == 6
    assert packed.layout.failure_state is capacity.selection.layout.failure_state
    assert packed.layout.failure_state.bits is capacity.selection.layout.failure_state.bits
    assert packed.layout.overflow is capacity.selection.layout.overflow
    assert torch.equal(packed.num_paths, torch.tensor([2, 1], device="cuda", dtype=torch.int32))
    assert not bool(torch.any(packed.valid[invalid]))
    for name in ("tx_id", "rx_id", "component_id", "primitive_id", "edge_id", "material_id"):
        assert bool(torch.all(getattr(packed.table, name)[invalid] == -1))
    for name in ("primitive_sequence", "material_sequence"):
        assert bool(torch.all(getattr(packed.table, name)[invalid] == -1))
    assert bool(torch.all(packed.table.path_length_m[invalid] == -1.0))
    assert bool(torch.all(packed.table.delay_s[invalid] == -1.0))
    for name in (
        "path_gain", "interaction_position", "interaction_normal",
        "interaction_positions", "interaction_normals", "field_real", "field_imag",
        "coefficient", "field_xyz", "field_direction", "phase_rad", "interaction_count",
    ):
        assert torch.count_nonzero(getattr(packed.table, name)[invalid]).item() == 0
    assert not packed.table.phase_rad.requires_grad


def test_path_table_capacity_pack_include_fields_false_ignores_valid_path_field_poison() -> None:
    capacity, values = _capacity()
    path_field_parts = torch.view_as_real(values["path_field"])
    path_field_parts[0] = torch.tensor([torch.nan, torch.inf], device="cuda")
    path_field_parts[2] = torch.tensor([-torch.inf, torch.nan], device="cuda")
    path_field_parts[3] = torch.tensor([torch.inf, -torch.inf], device="cuda")

    packed = deterministic_path_table_capacity_pack(capacity, include_fields=False)
    baseline = build_path_table(
        capacity.evaluated, frequency_hz=28.0e9, include_fields=False
    )
    valid = capacity.selection.valid
    for name in ("field_real", "field_imag", "phase_rad"):
        actual = getattr(packed.table, name)
        expected = getattr(baseline, name)
        assert bool(torch.all(torch.isfinite(actual)))
        _assert_positive_zero_bits(actual)
        _assert_bits_equal(actual[valid], expected[valid])

    source = (
        _ROOT / "native/channel_native/kernels/deterministic_path_table_capacity_pack.cu"
    ).read_text(encoding="utf-8")
    assert """if (include_fields) {
            const Complex path_field = input.path_field[row];""" in source


def test_path_table_capacity_pack_valid_nan_phase_matches_live_export_bitwise() -> None:
    capacity, values = _capacity()
    path_field_parts = torch.view_as_real(values["path_field"])
    path_field_parts[0] = torch.tensor([torch.nan, 1.0], device="cuda")
    path_field_parts[2] = torch.tensor([1.0, torch.nan], device="cuda")
    path_field_parts[3] = torch.tensor([torch.nan, torch.nan], device="cuda")

    packed = deterministic_path_table_capacity_pack(capacity, include_fields=True)
    baseline = build_path_table(
        capacity.evaluated, frequency_hz=28.0e9, include_fields=True
    )
    valid = capacity.selection.valid
    actual = packed.table.phase_rad[valid]
    expected = baseline.phase_rad[valid]
    _assert_bits_equal(actual, expected)
    assert torch.equal(torch.isnan(actual), torch.isnan(expected))
    assert torch.equal(torch.isinf(actual), torch.isinf(expected))
    assert torch.equal(torch.isfinite(actual), torch.isfinite(expected))


def test_path_table_capacity_pack_shared_failure_is_inert_before_poison_reads() -> None:
    capacity, _ = _capacity()
    capacity.selection.layout.failure_state.bits.fill_(1)
    packed = deterministic_path_table_capacity_pack(capacity)
    _assert_canonical_inert(packed)
    assert packed.layout.failure_state is capacity.selection.layout.failure_state


def test_path_table_capacity_pack_local_overflow_is_inert_without_trap() -> None:
    capacity, _ = _capacity()
    capacity.selection.layout.overflow.fill_(True)
    packed = deterministic_path_table_capacity_pack(capacity)
    _assert_canonical_inert(packed)


def test_path_table_capacity_pack_direct_rejects_failure_dtype_and_capacity_overflow() -> None:
    capacity, _ = _capacity()
    tensors = capacity_module._path_table_inputs(capacity)
    op = required_symbol("deterministic_path_table_capacity_pack")
    with pytest.raises(RuntimeError, match="failure_state"):
        op(
            capacity.selection.layout.overflow,
            *tensors,
            capacity.selection.num_paths,
            capacity.selection.overflow,
            True,
            2,
            3,
        )
    with pytest.raises(RuntimeError, match="overflows int64"):
        op(
            capacity.selection.layout.failure_state.bits,
            *tensors,
            capacity.selection.num_paths,
            capacity.selection.overflow,
            True,
            2**62,
            8,
        )


def test_path_table_capacity_pack_missing_native_symbol_has_no_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    capacity, _ = _capacity()

    def missing(name: str):
        raise RuntimeError(f"required native symbol missing: {name}")

    monkeypatch.setattr(capacity_module, "_required_native_op", missing)
    with pytest.raises(
        RuntimeError, match="deterministic_path_table_capacity_pack"
    ):
        deterministic_path_table_capacity_pack(capacity)


def test_path_table_capacity_pack_backward_preserves_all_continuous_inputs() -> None:
    capacity, values = _capacity(differentiable=True)
    table = deterministic_path_table_capacity_pack(capacity).table
    loss = (
        table.path_length_m.sum() + table.delay_s.sum() + table.field_direction.sum()
        + table.interaction_position.sum() + table.interaction_normal.sum()
        + table.interaction_positions.sum() + table.interaction_normals.sum()
        + table.path_gain.sum() + table.field_real.sum() + 2.0 * table.field_imag.sum()
        + table.field_xyz.real.sum() + table.field_xyz.imag.sum()
        + table.coefficient.real.sum() + table.coefficient.imag.sum()
    )
    loss.backward()
    valid = capacity.selection.valid
    for name, value in values.items():
        assert value.grad is not None, name
        if value.is_complex():
            expected = torch.complex(torch.ones_like(value.real), torch.ones_like(value.imag))
            if name == "path_field":
                expected = torch.complex(torch.ones_like(value.real), 2.0 * torch.ones_like(value.imag))
        else:
            expected = torch.ones_like(value)
        expanded_valid = valid.reshape((valid.shape[0],) + (1,) * (value.ndim - 1)).expand_as(value)
        _assert_bits_equal(value.grad[expanded_valid], expected[expanded_valid])
        _assert_positive_zero_bits(value.grad[~expanded_valid])
    assert not table.phase_rad.requires_grad


def test_path_table_capacity_pack_shared_failure_backward_is_exact_finite_zero() -> None:
    capacity, values = _capacity(differentiable=True)
    capacity.selection.layout.failure_state.bits.fill_(1)
    table = deterministic_path_table_capacity_pack(capacity).table
    loss = (
        table.path_length_m.sum()
        + table.delay_s.sum()
        + table.field_direction.sum()
        + table.interaction_position.sum()
        + table.interaction_normal.sum()
        + table.interaction_positions.sum()
        + table.interaction_normals.sum()
        + table.path_gain.sum()
        + table.field_real.sum()
        + table.field_imag.sum()
        + table.field_xyz.real.sum()
        + table.field_xyz.imag.sum()
        + table.coefficient.real.sum()
        + table.coefficient.imag.sum()
    )
    loss.backward()
    for name, value in values.items():
        assert value.grad is not None, name
        assert bool(torch.all(torch.isfinite(value.grad))), name
        _assert_positive_zero_bits(value.grad)
    assert not table.phase_rad.requires_grad


def test_path_table_capacity_pack_jvp_preserves_all_continuous_inputs() -> None:
    capacity, values = _capacity()
    tangents = {name: torch.full_like(value, 0.375 + 0.125j if value.is_complex() else 0.375) for name, value in values.items()}
    with torch.autograd.forward_ad.dual_level():
        dual_values = {
            name: torch.autograd.forward_ad.make_dual(values[name], tangents[name])
            for name in _CONTINUOUS
        }
        topology = capacity.evaluated.topology
        dual_geometry = PathGeometry(
            row_identity=topology.row_identity,
            **{name: dual_values[name] for name in _CONTINUOUS[:7]},
        )
        dual_fields = PathFields(
            row_identity=topology.row_identity,
            **{name: dual_values[name] for name in _CONTINUOUS[7:]},
        )
        dual_capacity = CapacityEvaluatedPaths(
            selection=capacity.selection,
            evaluated=EvaluatedPaths(topology=topology, geometry=dual_geometry, fields=dual_fields),
        )
        table = deterministic_path_table_capacity_pack(dual_capacity).table
        expected_by_output = {
            "path_length_m": tangents["path_length_m"], "delay_s": tangents["delay_s"],
            "field_direction": tangents["field_direction"],
            "interaction_position": tangents["interaction_position"],
            "interaction_normal": tangents["interaction_normal"],
            "interaction_positions": tangents["interaction_positions"],
            "interaction_normals": tangents["interaction_normals"],
            "path_gain": tangents["path_gain"], "field_real": tangents["path_field"].real,
            "field_imag": tangents["path_field"].imag, "field_xyz": tangents["field_xyz"],
            "coefficient": tangents["coefficient"],
        }
        for name, expected in expected_by_output.items():
            _, tangent = torch.autograd.forward_ad.unpack_dual(getattr(table, name))
            assert tangent is not None, name
            expanded_valid = capacity.selection.valid.reshape(
                (6,) + (1,) * (expected.ndim - 1)
            ).expand_as(expected)
            _assert_bits_equal(tangent[expanded_valid], expected[expanded_valid])
            _assert_positive_zero_bits(tangent[~expanded_valid])
        _, phase_tangent = torch.autograd.forward_ad.unpack_dual(table.phase_rad)
        assert phase_tangent is None


def test_path_table_capacity_pack_shared_failure_jvp_is_exact_finite_zero() -> None:
    capacity, values = _capacity()
    capacity.selection.layout.failure_state.bits.fill_(1)
    tangents = {
        name: torch.full_like(
            value, 0.375 + 0.125j if value.is_complex() else 0.375
        )
        for name, value in values.items()
    }
    with torch.autograd.forward_ad.dual_level():
        dual_values = {
            name: torch.autograd.forward_ad.make_dual(values[name], tangents[name])
            for name in _CONTINUOUS
        }
        topology = capacity.evaluated.topology
        dual_capacity = CapacityEvaluatedPaths(
            selection=capacity.selection,
            evaluated=EvaluatedPaths(
                topology=topology,
                geometry=PathGeometry(
                    row_identity=topology.row_identity,
                    **{name: dual_values[name] for name in _CONTINUOUS[:7]},
                ),
                fields=PathFields(
                    row_identity=topology.row_identity,
                    **{name: dual_values[name] for name in _CONTINUOUS[7:]},
                ),
            ),
        )
        table = deterministic_path_table_capacity_pack(dual_capacity).table
        for name in capacity_module._DIFFERENTIABLE_OUTPUT_FIELDS:
            _, tangent = torch.autograd.forward_ad.unpack_dual(getattr(table, name))
            assert tangent is not None, name
            assert bool(torch.all(torch.isfinite(tangent))), name
            _assert_positive_zero_bits(tangent)
        _, phase_tangent = torch.autograd.forward_ad.unpack_dual(table.phase_rad)
        assert phase_tangent is None


def test_path_table_capacity_pack_current_stream() -> None:
    capacity, _ = _capacity()
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        raw = _direct_outputs(capacity)
        marker = raw["path_gain"] + 1.0
    torch.cuda.current_stream().wait_stream(stream)
    assert bool(torch.all(marker[raw["valid"]] > 1.0))


def test_path_table_capacity_pack_zero_capacity() -> None:
    source, _ = _capacity()
    old = source.evaluated
    topology = PathTopology(
        **{
            name: getattr(old.topology, name)[:0].contiguous()
            for name in (
                "valid", "tx_id", "rx_id", "depth", "component_id", "primitive_id",
                "edge_id", "material_id", "primitive_sequence", "material_sequence",
                "interaction_type",
            )
        }
    )
    geometry = PathGeometry(
        row_identity=topology.row_identity,
        **{name: getattr(old.geometry, name)[:0].contiguous() for name in _CONTINUOUS[:7]},
    )
    path_fields = PathFields(
        row_identity=topology.row_identity,
        **{name: getattr(old.fields, name)[:0].contiguous() for name in _CONTINUOUS[7:]},
    )
    failure_state = create_capacity_failure_state(topology.valid)
    layout = CapacityPathLayout(
        pair_count=2,
        path_capacity_per_pair=0,
        failure_state=failure_state,
        valid=topology.valid,
        num_paths=torch.zeros(2, device="cuda", dtype=torch.int32),
        overflow=torch.zeros(1, device="cuda", dtype=torch.bool),
    )
    capacity = CapacityEvaluatedPaths(
        selection=CapacityPathSelection(
            selected_row_index=torch.empty(0, device="cuda", dtype=torch.int64),
            layout=layout,
        ),
        evaluated=EvaluatedPaths(topology=topology, geometry=geometry, fields=path_fields),
    )
    packed = deterministic_path_table_capacity_pack(capacity)
    assert packed.row_capacity == 0
    assert packed.valid.numel() == 0
    assert packed.num_paths.shape == (2,)
    assert torch.count_nonzero(packed.num_paths).item() == 0
    for descriptor in fields(PathTable):
        assert getattr(packed.table, descriptor.name).shape[0] == 0
