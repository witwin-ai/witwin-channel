from __future__ import annotations

import inspect
import json
from pathlib import Path

import pytest
import torch

from witwin.channel.deterministic.kernels import diffraction_pair
from witwin.channel.deterministic.kernels.diffraction_pair import (
    deterministic_diffraction_pair_reduce,
    deterministic_diffraction_pair_reduce_ad,
    deterministic_diffraction_pair_reduce_backward,
    deterministic_diffraction_pair_reduce_jvp,
)
from witwin.channel.runtime import symbols
from witwin.channel.runtime.capacity import (
    CapacityFailureBit,
    create_capacity_failure_state,
)


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is required"
)

_ROOT = Path(__file__).resolve().parents[2]


def _case(
    pair_count: int,
    state_capacity: int,
    *,
    valid: torch.Tensor | None = None,
    differentiable: bool = False,
) -> tuple[object, torch.Tensor, tuple[torch.Tensor, ...]]:
    rows = pair_count * state_capacity
    if valid is None:
        valid = torch.ones(rows, device="cuda", dtype=torch.bool)
    fields = tuple(
        torch.zeros(rows, device="cuda", dtype=torch.float32).requires_grad_(
            differentiable
        )
        for _ in range(6)
    )
    return create_capacity_failure_state(valid), valid, fields


def _reported_count(valid: torch.Tensor) -> torch.Tensor:
    return torch.sum(valid, dtype=torch.int32).reshape(1)


def _reduce(
    failure_state,
    valid,
    fields,
    pair_count,
    state_capacity,
    *,
    reported_count: torch.Tensor | None = None,
):
    if reported_count is None:
        reported_count = _reported_count(valid)
    return deterministic_diffraction_pair_reduce(
        failure_state,
        reported_count,
        valid,
        *fields,
        pair_count=pair_count,
        state_capacity=state_capacity,
    )


def _assert_positive_zero(value: torch.Tensor) -> None:
    real_storage = value.view(torch.float32) if value.is_complex() else value
    assert torch.equal(
        real_storage.view(torch.int32),
        torch.zeros_like(real_storage, dtype=torch.int32),
    )


def _assert_bits_equal(actual: torch.Tensor, expected: torch.Tensor) -> None:
    actual_real = actual.view(torch.float32) if actual.is_complex() else actual
    expected_real = expected.view(torch.float32) if expected.is_complex() else expected
    assert torch.equal(actual_real.view(torch.int32), expected_real.view(torch.int32))


def test_diffraction_pair_reduce_facade_owns_all_three_native_symbols() -> None:
    assert diffraction_pair._required_native_op is symbols.required_symbol
    for name in (
        "deterministic_diffraction_pair_reduce",
        "deterministic_diffraction_pair_reduce_backward",
        "deterministic_diffraction_pair_reduce_jvp",
    ):
        owner = getattr(diffraction_pair, name)
        assert inspect.unwrap(owner).__globals__ is diffraction_pair.__dict__


def test_diffraction_pair_reduce_source_contract_is_frozen() -> None:
    source = (
        _ROOT
        / "native/channel/kernels/deterministic_diffraction_pair_reduce.cu"
    ).read_text(encoding="utf-8")
    cmake = (_ROOT / "CMakeLists.txt").read_text(encoding="utf-8")
    assert "__shfl_sync(kFullWarpMask" in source
    assert "if (valid[row])" in source
    assert "power[pair] = (px + py) + pz;" in source
    assert "const float dot = (((((t0 + t1) + t2) + t3) + t4) + t5);" in source
    assert "kDiffractionPathContractError" in source
    assert "reported_count[0]" in source
    assert "atomicAdd" not in source
    assert "__shared__" not in source
    assert "cooperative_groups" not in source
    assert "cudaStreamSynchronize" not in source
    assert "deterministic_diffraction_pair_reduce.cu" in cmake
    flag_block = cmake.split("set_source_files_properties(", maxsplit=3)
    assert any(
        "deterministic_diffraction_pair_reduce.cu" in block
        and 'COMPILE_OPTIONS "--fmad=false"' in block
        for block in flag_block
    )


def test_diffraction_pair_reduce_owner_and_dormant_ledgers_are_complete() -> None:
    inventory = json.loads(
        (
            _ROOT
            / "docs/dev/audit/phase13-current-native-owner-inventory.json"
        ).read_text(encoding="utf-8")
    )
    family = inventory["adr030_dormant_diffraction_pair_reducer"]
    assert family["symbols"] == [
        "deterministic_diffraction_pair_reduce",
        "deterministic_diffraction_pair_reduce_backward",
        "deterministic_diffraction_pair_reduce_jvp",
    ]
    assert family["numerical_owner"] == "Channel"
    assert "live ReceiverGrid Torch index_add route unchanged" in family["status"]
    rows = {row["symbol"]: row for row in inventory["symbols"]}
    for symbol in family["symbols"]:
        assert rows[symbol]["numerical_owner"] == "Channel"
        assert rows[symbol]["production_callers"] == []
        assert rows[symbol]["liveness"] == "dormant-native-producer"

    ledger = json.loads(
        (
            _ROOT
            / "docs/dev/audit/adr-030-diffraction-pair-reducer-launch-resource-ledger.json"
        ).read_text(encoding="utf-8")
    )
    assert ledger["status"] == "dormant native family; no live solver caller"
    assert ledger["live_switch"]["performed"] is False
    assert ledger["launch_contract"]["floating_point_atomics"] is False
    assert ledger["resource_contract"]["full_lane_workspace"] is False


@pytest.mark.parametrize("state_capacity", [0, 1, 31, 32, 33])
def test_diffraction_pair_reduce_capacity_boundaries(state_capacity: int) -> None:
    pair_count = 3
    failure_state, valid, fields = _case(pair_count, state_capacity)
    if state_capacity:
        fields[0].fill_(1.0)
    result = _reduce(failure_state, valid, fields, pair_count, state_capacity)
    expected_field = torch.zeros(
        pair_count, 3, device="cuda", dtype=torch.complex64
    )
    expected_field[:, 0] = complex(float(state_capacity), 0.0)
    expected_power = torch.full(
        (pair_count,),
        float(state_capacity * state_capacity),
        device="cuda",
        dtype=torch.float32,
    )
    _assert_bits_equal(result.field_xyz, expected_field)
    _assert_bits_equal(result.power, expected_power)


def test_diffraction_pair_reduce_zero_pairs_has_typed_empty_outputs() -> None:
    failure_state, valid, fields = _case(0, 33)
    result = _reduce(failure_state, valid, fields, 0, 33)
    assert result.field_xyz.shape == (0, 3)
    assert result.field_xyz.dtype == torch.complex64
    assert result.power.shape == (0,)
    assert result.power.dtype == torch.float32


def test_diffraction_pair_reduce_pins_non_associative_order_across_warp() -> None:
    pair_count, state_capacity = 1, 33
    failure_state, valid, fields = _case(pair_count, state_capacity)
    fields[0][30] = float(2**24)
    fields[0][31] = 1.0
    fields[0][32] = -float(2**24)
    result = _reduce(failure_state, valid, fields, pair_count, state_capacity)
    _assert_positive_zero(result.field_xyz)
    _assert_positive_zero(result.power)


def test_diffraction_pair_reduce_multi_pair_sparse_valid_skips_poison() -> None:
    pair_count, state_capacity = 3, 5
    valid = torch.tensor(
        [
            True,
            False,
            True,
            False,
            False,
            False,
            True,
            False,
            True,
            False,
            False,
            False,
            False,
            False,
            False,
        ],
        device="cuda",
    )
    failure_state, valid, fields = _case(
        pair_count, state_capacity, valid=valid
    )
    for index, field in enumerate(fields):
        field.fill_(float("nan") if index % 2 == 0 else float("inf"))
        field[valid] = float(index + 1)
    result = _reduce(failure_state, valid, fields, pair_count, state_capacity)
    expected = torch.tensor(
        [
            [complex(2.0, 4.0), complex(6.0, 8.0), complex(10.0, 12.0)],
            [complex(2.0, 4.0), complex(6.0, 8.0), complex(10.0, 12.0)],
            [0j, 0j, 0j],
        ],
        device="cuda",
        dtype=torch.complex64,
    )
    expected_power = torch.tensor(
        [364.0, 364.0, 0.0], device="cuda", dtype=torch.float32
    )
    _assert_bits_equal(result.field_xyz, expected)
    _assert_bits_equal(result.power, expected_power)
    _assert_positive_zero(result.field_xyz[2])
    _assert_positive_zero(result.power[2:])
    tangent = deterministic_diffraction_pair_reduce_jvp(
        failure_state,
        valid,
        result.field_xyz,
        tangent_x_re=fields[0],
        tangent_x_im=fields[1],
        tangent_y_re=fields[2],
        tangent_y_im=fields[3],
        tangent_z_re=fields[4],
        tangent_z_im=fields[5],
        pair_count=pair_count,
        state_capacity=state_capacity,
    )
    _assert_positive_zero(tangent.field_xyz[2])
    _assert_positive_zero(tangent.power[2:])
    gradients = deterministic_diffraction_pair_reduce_backward(
        failure_state,
        valid,
        result.field_xyz,
        grad_field_xyz=torch.ones_like(result.field_xyz),
        grad_power=torch.ones_like(result.power),
        pair_count=pair_count,
        state_capacity=state_capacity,
    )
    for gradient in gradients.as_tuple():
        _assert_positive_zero(gradient[~valid])


def test_diffraction_pair_reduce_failure_is_wholly_inert_before_poison() -> None:
    pair_count, state_capacity = 2, 33
    failure_state, valid, fields = _case(pair_count, state_capacity)
    failure_state.bits.fill_(1)
    for index, field in enumerate(fields):
        field.fill_(float("nan") if index % 2 == 0 else float("inf"))
    result = _reduce(failure_state, valid, fields, pair_count, state_capacity)
    _assert_positive_zero(result.field_xyz)
    _assert_positive_zero(result.power)

    gradients = deterministic_diffraction_pair_reduce_backward(
        failure_state,
        valid,
        torch.full(
            (pair_count, 3), complex(float("nan"), float("inf")), device="cuda"
        ),
        grad_field_xyz=torch.full(
            (pair_count, 3), complex(float("nan"), float("inf")), device="cuda"
        ),
        grad_power=torch.full((pair_count,), float("nan"), device="cuda"),
        pair_count=pair_count,
        state_capacity=state_capacity,
    )
    for gradient in gradients.as_tuple():
        _assert_positive_zero(gradient)

    tangent = deterministic_diffraction_pair_reduce_jvp(
        failure_state,
        valid,
        torch.full(
            (pair_count, 3), complex(float("nan"), float("inf")), device="cuda"
        ),
        tangent_x_re=fields[0],
        tangent_x_im=fields[1],
        tangent_y_re=fields[2],
        tangent_y_im=fields[3],
        tangent_z_re=fields[4],
        tangent_z_im=fields[5],
        pair_count=pair_count,
        state_capacity=state_capacity,
    )
    _assert_positive_zero(tangent.field_xyz)
    _assert_positive_zero(tangent.power)


def test_diffraction_pair_reduce_count_mismatch_sets_failure_and_is_wholly_inert(
) -> None:
    pair_count, state_capacity = 2, 3
    failure_state, valid, fields = _case(pair_count, state_capacity)
    for index, field in enumerate(fields):
        field.fill_(float("nan") if index % 2 == 0 else float("inf"))
    reported_count = torch.tensor([5], device="cuda", dtype=torch.int32)
    result = _reduce(
        failure_state,
        valid,
        fields,
        pair_count,
        state_capacity,
        reported_count=reported_count,
    )
    _assert_positive_zero(result.field_xyz)
    _assert_positive_zero(result.power)
    assert int(failure_state.bits.item()) & int(
        CapacityFailureBit.DIFFRACTION_PATH_CONTRACT_ERROR
    )

    gradients = deterministic_diffraction_pair_reduce_backward(
        failure_state,
        valid,
        torch.full(
            (pair_count, 3), complex(float("nan"), float("inf")), device="cuda"
        ),
        grad_field_xyz=torch.full(
            (pair_count, 3), complex(float("nan"), float("inf")), device="cuda"
        ),
        grad_power=torch.full((pair_count,), float("nan"), device="cuda"),
        pair_count=pair_count,
        state_capacity=state_capacity,
    )
    for gradient in gradients.as_tuple():
        _assert_positive_zero(gradient)
    tangent = deterministic_diffraction_pair_reduce_jvp(
        failure_state,
        valid,
        torch.full(
            (pair_count, 3), complex(float("nan"), float("inf")), device="cuda"
        ),
        tangent_x_re=fields[0],
        tangent_x_im=fields[1],
        tangent_y_re=fields[2],
        tangent_y_im=fields[3],
        tangent_z_re=fields[4],
        tangent_z_im=fields[5],
        pair_count=pair_count,
        state_capacity=state_capacity,
    )
    _assert_positive_zero(tangent.field_xyz)
    _assert_positive_zero(tangent.power)


def test_diffraction_pair_reduce_backward_formula_and_missing_cotangents() -> None:
    pair_count, state_capacity = 2, 3
    valid = torch.tensor([True, False, True, True, True, False], device="cuda")
    failure_state, valid, fields = _case(
        pair_count, state_capacity, valid=valid
    )
    for index, field in enumerate(fields):
        field.copy_(
            torch.arange(1, 7, device="cuda", dtype=torch.float32) + index
        )
    primal = _reduce(failure_state, valid, fields, pair_count, state_capacity)
    grad_field = torch.tensor(
        [
            [complex(1.0, 2.0), complex(3.0, 4.0), complex(5.0, 6.0)],
            [complex(-1.0, -2.0), complex(-3.0, -4.0), complex(-5.0, -6.0)],
        ],
        device="cuda",
        dtype=torch.complex64,
    )
    grad_power = torch.tensor([0.25, -0.5], device="cuda")
    gradients = deterministic_diffraction_pair_reduce_backward(
        failure_state,
        valid,
        primal.field_xyz,
        grad_field_xyz=grad_field,
        grad_power=grad_power,
        pair_count=pair_count,
        state_capacity=state_capacity,
    )
    sums = primal.field_xyz.view(torch.float32).reshape(pair_count, 6)
    field_cotangent = grad_field.view(torch.float32).reshape(pair_count, 6)
    for component, actual in enumerate(gradients.as_tuple()):
        expected = torch.zeros_like(actual)
        for row in range(pair_count * state_capacity):
            if bool(valid[row]):
                pair = row // state_capacity
                expected[row] = field_cotangent[pair, component] + (
                    2.0 * sums[pair, component]
                ) * grad_power[pair]
        _assert_bits_equal(actual, expected)

    missing = deterministic_diffraction_pair_reduce_backward(
        failure_state,
        valid,
        primal.field_xyz,
        pair_count=pair_count,
        state_capacity=state_capacity,
    )
    for value in missing.as_tuple():
        _assert_positive_zero(value)
    poison_field = torch.full(
        (pair_count, 3),
        complex(float("nan"), float("inf")),
        device="cuda",
    )
    poison_missing = deterministic_diffraction_pair_reduce_backward(
        failure_state,
        valid,
        poison_field,
        pair_count=pair_count,
        state_capacity=state_capacity,
    )
    for value in poison_missing.as_tuple():
        _assert_positive_zero(value)
    missing_jvp = deterministic_diffraction_pair_reduce_jvp(
        failure_state,
        valid,
        poison_field,
        pair_count=pair_count,
        state_capacity=state_capacity,
    )
    _assert_positive_zero(missing_jvp.field_xyz)
    _assert_positive_zero(missing_jvp.power)


def test_diffraction_pair_reduce_jvp_vjp_duality_and_poison_gating() -> None:
    torch.manual_seed(34017)
    pair_count, state_capacity = 3, 33
    valid = torch.rand(pair_count * state_capacity, device="cuda") > 0.45
    failure_state, valid, fields = _case(
        pair_count, state_capacity, valid=valid
    )
    for field in fields:
        field.copy_(torch.randn_like(field))
    primal = _reduce(failure_state, valid, fields, pair_count, state_capacity)
    tangents = tuple(torch.randn_like(field) for field in fields)
    poisoned_tangents = tuple(value.clone() for value in tangents)
    for index, value in enumerate(poisoned_tangents):
        value[~valid] = float("nan") if index % 2 == 0 else float("inf")
    jvp = deterministic_diffraction_pair_reduce_jvp(
        failure_state,
        valid,
        primal.field_xyz,
        tangent_x_re=poisoned_tangents[0],
        tangent_x_im=poisoned_tangents[1],
        tangent_y_re=poisoned_tangents[2],
        tangent_y_im=poisoned_tangents[3],
        tangent_z_re=poisoned_tangents[4],
        tangent_z_im=poisoned_tangents[5],
        pair_count=pair_count,
        state_capacity=state_capacity,
    )
    assert bool(torch.all(torch.isfinite(jvp.field_xyz)))
    assert bool(torch.all(torch.isfinite(jvp.power)))

    grad_field = torch.randn(
        pair_count, 3, device="cuda", dtype=torch.complex64
    )
    grad_power = torch.randn(pair_count, device="cuda")
    vjp = deterministic_diffraction_pair_reduce_backward(
        failure_state,
        valid,
        primal.field_xyz,
        grad_field_xyz=grad_field,
        grad_power=grad_power,
        pair_count=pair_count,
        state_capacity=state_capacity,
    )
    lhs = (
        (jvp.field_xyz.conj() * grad_field).real.sum()
        + (jvp.power * grad_power).sum()
    )
    rhs = sum(
        (tangent * gradient).sum()
        for tangent, gradient in zip(tangents, vjp.as_tuple(), strict=True)
    )
    torch.testing.assert_close(lhs, rhs, rtol=2e-5, atol=2e-5)


def test_diffraction_pair_reduce_ad_companions_accept_strided_values() -> None:
    pair_count, state_capacity = 2, 5
    failure_state, valid, fields = _case(pair_count, state_capacity)
    for index, field in enumerate(fields):
        field.fill_(float(index + 1))
    primal = _reduce(failure_state, valid, fields, pair_count, state_capacity)
    grad_field = torch.randn(
        pair_count, 6, device="cuda", dtype=torch.complex64
    )[:, ::2]
    grad_power = torch.randn(pair_count * 2, device="cuda")[::2]
    assert not grad_field.is_contiguous()
    assert not grad_power.is_contiguous()
    strided_vjp = deterministic_diffraction_pair_reduce_backward(
        failure_state,
        valid,
        primal.field_xyz,
        grad_field_xyz=grad_field,
        grad_power=grad_power,
        pair_count=pair_count,
        state_capacity=state_capacity,
    )
    contiguous_vjp = deterministic_diffraction_pair_reduce_backward(
        failure_state,
        valid,
        primal.field_xyz,
        grad_field_xyz=grad_field.contiguous(),
        grad_power=grad_power.contiguous(),
        pair_count=pair_count,
        state_capacity=state_capacity,
    )
    for actual, expected in zip(
        strided_vjp.as_tuple(), contiguous_vjp.as_tuple(), strict=True
    ):
        _assert_bits_equal(actual, expected)

    tangents = tuple(
        torch.randn(valid.numel() * 2, device="cuda")[::2] for _ in range(6)
    )
    assert all(not tangent.is_contiguous() for tangent in tangents)
    strided_jvp = deterministic_diffraction_pair_reduce_jvp(
        failure_state,
        valid,
        primal.field_xyz,
        tangent_x_re=tangents[0],
        tangent_x_im=tangents[1],
        tangent_y_re=tangents[2],
        tangent_y_im=tangents[3],
        tangent_z_re=tangents[4],
        tangent_z_im=tangents[5],
        pair_count=pair_count,
        state_capacity=state_capacity,
    )
    contiguous_jvp = deterministic_diffraction_pair_reduce_jvp(
        failure_state,
        valid,
        primal.field_xyz,
        tangent_x_re=tangents[0].contiguous(),
        tangent_x_im=tangents[1].contiguous(),
        tangent_y_re=tangents[2].contiguous(),
        tangent_y_im=tangents[3].contiguous(),
        tangent_z_re=tangents[4].contiguous(),
        tangent_z_im=tangents[5].contiguous(),
        pair_count=pair_count,
        state_capacity=state_capacity,
    )
    _assert_bits_equal(strided_jvp.field_xyz, contiguous_jvp.field_xyz)
    _assert_bits_equal(strided_jvp.power, contiguous_jvp.power)


def test_diffraction_pair_reduce_autograd_and_forward_ad_use_native_companions() -> None:
    torch.manual_seed(34018)
    pair_count, state_capacity = 2, 5
    valid = torch.tensor(
        [True, False, True, True, False, True, True, False, True, True],
        device="cuda",
    )
    failure_state, valid, fields = _case(
        pair_count, state_capacity, valid=valid, differentiable=True
    )
    with torch.no_grad():
        for field in fields:
            field.copy_(torch.randn_like(field))
    plain = _reduce(failure_state, valid, fields, pair_count, state_capacity)
    result = deterministic_diffraction_pair_reduce_ad(
        failure_state,
        _reported_count(valid),
        valid,
        *fields,
        pair_count=pair_count,
        state_capacity=state_capacity,
    )
    _assert_bits_equal(result.field_xyz, plain.field_xyz)
    _assert_bits_equal(result.power, plain.power)

    grad_field = torch.randn(
        pair_count, 3, device="cuda", dtype=torch.complex64
    )
    grad_power = torch.randn(pair_count, device="cuda")
    loss = (result.field_xyz.conj() * grad_field).real.sum() + (
        result.power * grad_power
    ).sum()
    loss.backward()
    direct = deterministic_diffraction_pair_reduce_backward(
        failure_state,
        valid,
        plain.field_xyz,
        grad_field_xyz=grad_field,
        grad_power=grad_power,
        pair_count=pair_count,
        state_capacity=state_capacity,
    )
    for field, expected in zip(fields, direct.as_tuple(), strict=True):
        assert field.grad is not None
        _assert_bits_equal(field.grad, expected)

    tangents = tuple(torch.randn_like(field) for field in fields)
    with torch.autograd.forward_ad.dual_level():
        dual_fields = tuple(
            torch.autograd.forward_ad.make_dual(field.detach(), tangent)
            for field, tangent in zip(fields, tangents, strict=True)
        )
        dual_result = deterministic_diffraction_pair_reduce_ad(
            failure_state,
            _reported_count(valid),
            valid,
            *dual_fields,
            pair_count=pair_count,
            state_capacity=state_capacity,
        )
        primal_field, tangent_field = torch.autograd.forward_ad.unpack_dual(
            dual_result.field_xyz
        )
        primal_power, tangent_power = torch.autograd.forward_ad.unpack_dual(
            dual_result.power
        )
        _assert_bits_equal(primal_field, plain.field_xyz)
        _assert_bits_equal(primal_power, plain.power)
        direct_jvp = deterministic_diffraction_pair_reduce_jvp(
            failure_state,
            valid,
            plain.field_xyz,
            tangent_x_re=tangents[0],
            tangent_x_im=tangents[1],
            tangent_y_re=tangents[2],
            tangent_y_im=tangents[3],
            tangent_z_re=tangents[4],
            tangent_z_im=tangents[5],
            pair_count=pair_count,
            state_capacity=state_capacity,
        )
        _assert_bits_equal(tangent_field, direct_jvp.field_xyz)
        _assert_bits_equal(tangent_power, direct_jvp.power)


def test_diffraction_pair_reduce_current_stream() -> None:
    pair_count, state_capacity = 2, 33
    failure_state, valid, fields = _case(pair_count, state_capacity)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        fields[0].fill_(1.0)
        result = _reduce(failure_state, valid, fields, pair_count, state_capacity)
        marker = result.power + 1.0
    torch.cuda.current_stream().wait_stream(stream)
    expected = torch.full((pair_count,), 1090.0, device="cuda")
    _assert_bits_equal(marker, expected)


def test_diffraction_pair_reduce_missing_native_symbol_has_no_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    failure_state, valid, fields = _case(1, 1)

    def missing(name: str):
        raise RuntimeError(f"required native symbol missing: {name}")

    monkeypatch.setattr(diffraction_pair, "_required_native_op", missing)
    with pytest.raises(
        RuntimeError, match="deterministic_diffraction_pair_reduce"
    ):
        _reduce(failure_state, valid, fields, 1, 1)


def test_diffraction_pair_reduce_rejects_bad_capacity_without_native_work() -> None:
    failure_state, valid, fields = _case(1, 1)
    with pytest.raises(ValueError, match="valid must have shape"):
        _reduce(failure_state, valid, fields, 2, 1)
    with pytest.raises(ValueError, match="state_capacity must be non-negative"):
        _reduce(failure_state, valid, fields, 1, -1)
    with pytest.raises(ValueError, match="reported_count must have shape"):
        _reduce(
            failure_state,
            valid,
            fields,
            1,
            1,
            reported_count=torch.zeros(2, device="cuda", dtype=torch.int32),
        )
