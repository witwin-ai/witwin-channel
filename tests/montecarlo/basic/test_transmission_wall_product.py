from __future__ import annotations

import inspect
import json
from pathlib import Path

import pytest
import torch

from witwin.channel.kernels import montecarlo as transmission
from witwin.channel.kernels.montecarlo import (
    mc_transmission_wall_product,
    mc_transmission_wall_product_ad,
    mc_transmission_wall_product_backward,
    mc_transmission_wall_product_jvp,
)
from witwin.channel.runtime import CapacityFailureBit, create_capacity_failure_state


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is required"
)

_ROOT = Path(__file__).resolve().parents[3]


def _case(*, differentiable: bool = False) -> tuple[tuple[torch.Tensor, ...], object]:
    valid = torch.tensor(
        [[True, False], [True, True], [False, False]],
        device="cuda",
        dtype=torch.bool,
    )
    num_hits = torch.tensor([1, 2, 0], device="cuda", dtype=torch.int32)
    reached = torch.ones(3, device="cuda", dtype=torch.bool)
    direction = torch.tensor(
        [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        device="cuda",
    ).requires_grad_(differentiable)
    normal = torch.tensor(
        [
            [[-1.0, 0.0, 0.0], [float("nan"), float("nan"), float("nan")]],
            [[-1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]],
            [[float("nan"), float("nan"), float("nan")]] * 2,
        ],
        device="cuda",
    ).requires_grad_(differentiable)
    primitive = torch.tensor(
        [[0, 2_000_000_000], [0, 1], [2_000_000_000, 2_000_000_000]],
        device="cuda",
        dtype=torch.int32,
    )
    face_material = torch.tensor([0, 0], device="cuda", dtype=torch.int32)
    geometry_mode = torch.tensor([0], device="cuda", dtype=torch.int32)
    layer_offset = torch.tensor([0], device="cuda", dtype=torch.int32)
    layer_count = torch.tensor([1], device="cuda", dtype=torch.int32)
    layer_thickness = torch.tensor(
        [0.08], device="cuda", dtype=torch.float32, requires_grad=differentiable
    )
    layer_eps = torch.tensor(
        [3.2], device="cuda", dtype=torch.float32, requires_grad=differentiable
    )
    layer_sigma = torch.tensor(
        [0.015], device="cuda", dtype=torch.float32, requires_grad=differentiable
    )
    layer_mu = torch.ones(1, device="cuda")
    polarization = torch.tensor(
        [[0.0, 1.0, 0.0]] * 3, device="cuda", dtype=torch.float32
    )
    base_power = torch.tensor(
        [2.0, 3.0, 4.0],
        device="cuda",
        dtype=torch.float32,
        requires_grad=differentiable,
    )
    inputs = (
        valid,
        num_hits,
        reached,
        direction,
        normal,
        primitive,
        face_material,
        geometry_mode,
        layer_offset,
        layer_count,
        layer_thickness,
        layer_eps,
        layer_sigma,
        layer_mu,
        polarization,
        base_power,
    )
    return inputs, create_capacity_failure_state(valid)


def _assert_tensors_are_zero(values: tuple[torch.Tensor, ...]) -> None:
    for value in values:
        assert torch.count_nonzero(value).item() == 0


def _call_zero_jvp(
    inputs: tuple[torch.Tensor, ...], state: object
) -> tuple[torch.Tensor, torch.Tensor]:
    return mc_transmission_wall_product_jvp(
        *inputs,
        frequency_hz=3.5e9,
        failure_state=state,
        tangent_direction=None,
        tangent_normal=None,
        tangent_layer_thickness_m=None,
        tangent_layer_eps_r=None,
        tangent_layer_sigma_e=None,
        tangent_base_power=None,
        tangent_frequency=0.0,
    )


def test_transmission_wall_product_capacity_poison_and_product_order() -> None:
    inputs, state = _case()
    result = mc_transmission_wall_product(*inputs, state, frequency_hz=3.5e9)

    assert result.scaled_power.shape == (3,)
    assert result.transmittance.shape == (3,)
    assert result.wall_count.tolist() == [1, 2, 0]
    assert result.penetrated.tolist() == [True, True, False]
    first = result.transmittance[0]
    assert torch.equal(result.transmittance[1], first * first)
    assert torch.equal(
        result.scaled_power[:2], result.transmittance[:2] * inputs[15][:2]
    )
    assert result.transmittance[2].item() == 1.0
    assert result.scaled_power[2].item() == 0.0
    assert state.bits.tolist() == [0]


def test_transmission_wall_product_zero_rows_preserve_all_family_shapes() -> None:
    inputs, _ = _case()
    row_inputs = {0, 1, 2, 3, 4, 5, 14, 15}
    empty_inputs = tuple(
        value[:0] if index in row_inputs else value
        for index, value in enumerate(inputs)
    )
    state = create_capacity_failure_state(empty_inputs[0])

    result = mc_transmission_wall_product(*empty_inputs, state, frequency_hz=3.5e9)
    gradients = mc_transmission_wall_product_backward(
        *empty_inputs,
        frequency_hz=3.5e9,
        failure_state=state,
        grad_scaled_power=None,
        grad_transmittance=None,
    )
    tangents = _call_zero_jvp(empty_inputs, state)

    assert result.scaled_power.shape == (0,)
    assert result.transmittance.shape == (0,)
    assert result.wall_count.shape == (0,)
    assert result.penetrated.shape == (0,)
    assert gradients[0].shape == (0, 3)
    assert gradients[1].shape == (0, 2, 3)
    assert gradients[5].shape == (0,)
    assert tangents[0].shape == (0,)
    assert tangents[1].shape == (0,)
    assert state.bits.tolist() == [0]


def test_transmission_wall_product_zero_hit_capacity_is_transparent() -> None:
    inputs, _ = _case()
    zero_depth_inputs = (
        inputs[0][:, :0],
        torch.zeros_like(inputs[1]),
        inputs[2],
        inputs[3],
        inputs[4][:, :0, :],
        inputs[5][:, :0],
        *inputs[6:],
    )
    state = create_capacity_failure_state(zero_depth_inputs[0])
    result = mc_transmission_wall_product(*zero_depth_inputs, state, frequency_hz=3.5e9)

    assert result.transmittance.tolist() == [1.0, 1.0, 1.0]
    assert result.scaled_power.tolist() == [0.0, 0.0, 0.0]
    assert result.wall_count.tolist() == [0, 0, 0]
    assert result.penetrated.tolist() == [False, False, False]
    assert state.bits.tolist() == [0]


def test_transmission_wall_product_preexisting_failure_is_preserved_and_inert() -> None:
    inputs, state = _case()
    preexisting = int(CapacityFailureBit.REFLECTION_CANDIDATE_OVERFLOW)
    state.bits.fill_(preexisting)
    result = mc_transmission_wall_product(*inputs, state, frequency_hz=3.5e9)
    gradients = mc_transmission_wall_product_backward(
        *inputs,
        frequency_hz=3.5e9,
        failure_state=state,
        grad_scaled_power=torch.ones(3, device="cuda"),
        grad_transmittance=torch.ones(3, device="cuda"),
    )
    tangents = mc_transmission_wall_product_jvp(
        *inputs,
        frequency_hz=3.5e9,
        failure_state=state,
        tangent_direction=torch.ones_like(inputs[3]),
        tangent_normal=torch.ones_like(inputs[4]),
        tangent_layer_thickness_m=torch.ones_like(inputs[10]),
        tangent_layer_eps_r=torch.ones_like(inputs[11]),
        tangent_layer_sigma_e=torch.ones_like(inputs[12]),
        tangent_base_power=torch.ones_like(inputs[15]),
        tangent_frequency=1.0,
    )

    _assert_tensors_are_zero(
        (
            result.scaled_power,
            result.transmittance,
            result.wall_count,
            result.penetrated,
        )
    )
    _assert_tensors_are_zero(gradients)
    _assert_tensors_are_zero(tangents)
    assert state.bits.tolist() == [preexisting]


def test_transmission_wall_product_uses_non_default_current_stream() -> None:
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        inputs, state = _case()
        result = mc_transmission_wall_product(*inputs, state, frequency_hz=3.5e9)
        gradients = mc_transmission_wall_product_backward(
            *inputs,
            frequency_hz=3.5e9,
            failure_state=state,
            grad_scaled_power=torch.ones(3, device="cuda"),
            grad_transmittance=None,
        )
        tangents = _call_zero_jvp(inputs, state)
        completed = stream.record_event()
    torch.cuda.current_stream().wait_event(completed)

    assert result.penetrated.tolist() == [True, True, False]
    assert torch.isfinite(gradients[0]).all()
    _assert_tensors_are_zero(tangents)
    assert state.bits.tolist() == [0]


@pytest.mark.parametrize("corruption", ["canonical", "primitive", "csr"])
def test_transmission_wall_product_contract_error_makes_entire_result_inert(
    corruption: str,
) -> None:
    inputs, state = _case()
    if corruption == "canonical":
        inputs[1][0] = 2
    elif corruption == "primitive":
        inputs[5][0, 0] = 2_000_000_000
    else:
        inputs[9][0] = 2
    result = mc_transmission_wall_product(*inputs, state, frequency_hz=3.5e9)

    assert state.bits.tolist() == [int(CapacityFailureBit.PAIR_CONTRACT_ERROR)]
    assert torch.count_nonzero(result.scaled_power).item() == 0
    assert torch.count_nonzero(result.transmittance).item() == 0
    assert torch.count_nonzero(result.wall_count).item() == 0
    assert torch.count_nonzero(result.penetrated).item() == 0


@pytest.mark.parametrize("corruption", ["canonical", "primitive", "csr"])
def test_transmission_wall_product_contract_error_makes_ad_results_inert(
    corruption: str,
) -> None:
    inputs, state = _case()
    if corruption == "canonical":
        inputs[1][0] = 2
    elif corruption == "primitive":
        inputs[5][0, 0] = 2_000_000_000
    else:
        inputs[9][0] = 2
    gradients = mc_transmission_wall_product_backward(
        *inputs,
        frequency_hz=3.5e9,
        failure_state=state,
        grad_scaled_power=torch.ones(3, device="cuda"),
        grad_transmittance=torch.ones(3, device="cuda"),
    )
    _assert_tensors_are_zero(gradients)
    assert state.bits.tolist() == [int(CapacityFailureBit.PAIR_CONTRACT_ERROR)]

    jvp_state = create_capacity_failure_state(inputs[0])
    tangents = mc_transmission_wall_product_jvp(
        *inputs,
        frequency_hz=3.5e9,
        failure_state=jvp_state,
        tangent_direction=torch.ones_like(inputs[3]),
        tangent_normal=torch.ones_like(inputs[4]),
        tangent_layer_thickness_m=torch.ones_like(inputs[10]),
        tangent_layer_eps_r=torch.ones_like(inputs[11]),
        tangent_layer_sigma_e=torch.ones_like(inputs[12]),
        tangent_base_power=torch.ones_like(inputs[15]),
        tangent_frequency=1.0,
    )
    _assert_tensors_are_zero(tangents)
    assert jvp_state.bits.tolist() == [int(CapacityFailureBit.PAIR_CONTRACT_ERROR)]


@pytest.mark.parametrize("later_corruption", ["primitive", "csr"])
def test_transmission_wall_product_blocked_slot_does_not_hide_later_contract_error(
    later_corruption: str,
) -> None:
    inputs, primal_state = _case()
    inputs[6][0] = -1
    inputs[6][1] = 0
    if later_corruption == "primitive":
        inputs[5][1, 1] = 2_000_000_000
    else:
        inputs[9][0] = 2

    result = mc_transmission_wall_product(*inputs, primal_state, frequency_hz=3.5e9)
    _assert_tensors_are_zero(
        (
            result.scaled_power,
            result.transmittance,
            result.wall_count,
            result.penetrated,
        )
    )
    assert primal_state.bits.tolist() == [int(CapacityFailureBit.PAIR_CONTRACT_ERROR)]

    backward_state = create_capacity_failure_state(inputs[0])
    gradients = mc_transmission_wall_product_backward(
        *inputs,
        frequency_hz=3.5e9,
        failure_state=backward_state,
        grad_scaled_power=torch.ones(3, device="cuda"),
        grad_transmittance=torch.ones(3, device="cuda"),
    )
    _assert_tensors_are_zero(gradients)
    assert backward_state.bits.tolist() == [int(CapacityFailureBit.PAIR_CONTRACT_ERROR)]

    jvp_state = create_capacity_failure_state(inputs[0])
    tangents = mc_transmission_wall_product_jvp(
        *inputs,
        frequency_hz=3.5e9,
        failure_state=jvp_state,
        tangent_direction=torch.ones_like(inputs[3]),
        tangent_normal=torch.ones_like(inputs[4]),
        tangent_layer_thickness_m=torch.ones_like(inputs[10]),
        tangent_layer_eps_r=torch.ones_like(inputs[11]),
        tangent_layer_sigma_e=torch.ones_like(inputs[12]),
        tangent_base_power=torch.ones_like(inputs[15]),
        tangent_frequency=1.0,
    )
    _assert_tensors_are_zero(tangents)
    assert jvp_state.bits.tolist() == [int(CapacityFailureBit.PAIR_CONTRACT_ERROR)]


def test_transmission_wall_product_all_optional_ad_inputs_absent_are_zero() -> None:
    inputs, state = _case()
    gradients = mc_transmission_wall_product_backward(
        *inputs,
        frequency_hz=3.5e9,
        failure_state=state,
        grad_scaled_power=None,
        grad_transmittance=None,
    )
    tangents = _call_zero_jvp(inputs, state)

    _assert_tensors_are_zero(gradients)
    _assert_tensors_are_zero(tangents)
    assert state.bits.tolist() == [0]


def test_transmission_wall_product_unreached_rows_do_not_read_poison_payload() -> None:
    inputs, state = _case()
    inputs[2].fill_(False)
    inputs[4].fill_(float("nan"))
    inputs[5].fill_(2_000_000_000)
    result = mc_transmission_wall_product(*inputs, state, frequency_hz=3.5e9)
    gradients = mc_transmission_wall_product_backward(
        *inputs,
        frequency_hz=3.5e9,
        failure_state=state,
        grad_scaled_power=torch.ones(3, device="cuda"),
        grad_transmittance=torch.ones(3, device="cuda"),
    )
    tangents = mc_transmission_wall_product_jvp(
        *inputs,
        frequency_hz=3.5e9,
        failure_state=state,
        tangent_direction=torch.full_like(inputs[3], float("nan")),
        tangent_normal=torch.full_like(inputs[4], float("nan")),
        tangent_layer_thickness_m=torch.ones_like(inputs[10]),
        tangent_layer_eps_r=torch.ones_like(inputs[11]),
        tangent_layer_sigma_e=torch.ones_like(inputs[12]),
        tangent_base_power=torch.ones_like(inputs[15]),
        tangent_frequency=float("nan"),
    )

    assert result.wall_count.tolist() == [1, 2, 0]
    _assert_tensors_are_zero(
        (result.scaled_power, result.transmittance, result.penetrated)
    )
    _assert_tensors_are_zero(gradients)
    _assert_tensors_are_zero(tangents)
    assert state.bits.tolist() == [0]


def test_transmission_wall_product_invalid_material_is_ordinary_blocking() -> None:
    inputs, state = _case()
    inputs[6].fill_(-1)
    result = mc_transmission_wall_product(*inputs, state, frequency_hz=3.5e9)

    assert state.bits.tolist() == [0]
    assert result.transmittance.tolist() == [0.0, 0.0, 1.0]
    assert result.scaled_power.tolist() == [0.0, 0.0, 0.0]
    assert result.wall_count.tolist() == [1, 1, 0]
    assert result.penetrated.tolist() == [False, False, False]


def test_transmission_wall_product_jvp_vjp_duality_and_determinism() -> None:
    inputs, state = _case()
    tangent_direction = torch.tensor(
        [[0.0, 0.03, 0.0], [0.0, -0.02, 0.0], [0.0, 0.0, 0.0]],
        device="cuda",
    )
    tangent_normal = torch.zeros_like(inputs[4])
    tangent_normal[0, 0, 1] = 0.01
    tangent_normal[1, 0, 1] = -0.03
    tangent_normal[1, 1, 1] = 0.02
    tangent_thickness = torch.tensor([0.002], device="cuda")
    tangent_eps = torch.tensor([0.04], device="cuda")
    tangent_sigma = torch.tensor([-0.001], device="cuda")
    tangent_base = torch.tensor([0.1, -0.2, 0.3], device="cuda")
    tangent_frequency = 2.0e6
    tangent_scaled, tangent_trans = mc_transmission_wall_product_jvp(
        *inputs,
        frequency_hz=3.5e9,
        failure_state=state,
        tangent_direction=tangent_direction,
        tangent_normal=tangent_normal,
        tangent_layer_thickness_m=tangent_thickness,
        tangent_layer_eps_r=tangent_eps,
        tangent_layer_sigma_e=tangent_sigma,
        tangent_base_power=tangent_base,
        tangent_frequency=tangent_frequency,
    )
    grad_scaled = torch.tensor([0.7, -0.3, 0.5], device="cuda")
    grad_trans = torch.tensor([-0.2, 0.4, 0.9], device="cuda")
    gradients = mc_transmission_wall_product_backward(
        *inputs,
        frequency_hz=3.5e9,
        failure_state=state,
        grad_scaled_power=grad_scaled,
        grad_transmittance=grad_trans,
    )
    repeated = mc_transmission_wall_product_backward(
        *inputs,
        frequency_hz=3.5e9,
        failure_state=state,
        grad_scaled_power=grad_scaled,
        grad_transmittance=grad_trans,
    )
    lhs = (tangent_scaled * grad_scaled).sum() + (tangent_trans * grad_trans).sum()
    rhs = (
        (gradients[0] * tangent_direction).sum()
        + (gradients[1] * tangent_normal).sum()
        + (gradients[2] * tangent_thickness).sum()
        + (gradients[3] * tangent_eps).sum()
        + (gradients[4] * tangent_sigma).sum()
        + (gradients[5] * tangent_base).sum()
        + gradients[6][0] * tangent_frequency
    )
    torch.testing.assert_close(lhs, rhs, rtol=2.0e-5, atol=2.0e-6)
    for first, second in zip(gradients, repeated, strict=True):
        assert torch.equal(first, second)


def test_transmission_wall_product_nonzero_csr_offset_uses_global_layer_seed() -> None:
    valid = torch.tensor([[True]], device="cuda")
    inputs = (
        valid,
        torch.tensor([1], device="cuda", dtype=torch.int32),
        torch.tensor([True], device="cuda"),
        torch.tensor([[1.0, 0.0, 0.0]], device="cuda"),
        torch.tensor([[[-0.8, -0.6, 0.0]]], device="cuda"),
        torch.tensor([[0]], device="cuda", dtype=torch.int32),
        torch.tensor([1], device="cuda", dtype=torch.int32),
        torch.tensor([0, 0], device="cuda", dtype=torch.int32),
        torch.tensor([0, 1], device="cuda", dtype=torch.int32),
        torch.tensor([1, 1], device="cuda", dtype=torch.int32),
        torch.tensor([0.02, 0.09], device="cuda"),
        torch.tensor([1.8, 4.7], device="cuda"),
        torch.tensor([0.002, 0.035], device="cuda"),
        torch.ones(2, device="cuda"),
        torch.tensor([[0.0, 1.0, 0.0]], device="cuda"),
        torch.tensor([2.0], device="cuda"),
    )
    state = create_capacity_failure_state(valid)
    gradients = mc_transmission_wall_product_backward(
        *inputs,
        frequency_hz=3.5e9,
        failure_state=state,
        grad_scaled_power=None,
        grad_transmittance=torch.ones(1, device="cuda"),
    )
    thickness_tangent = torch.tensor([0.0, 0.003], device="cuda")
    tangent_scaled, tangent_trans = mc_transmission_wall_product_jvp(
        *inputs,
        frequency_hz=3.5e9,
        failure_state=state,
        tangent_direction=None,
        tangent_normal=None,
        tangent_layer_thickness_m=thickness_tangent,
        tangent_layer_eps_r=None,
        tangent_layer_sigma_e=None,
        tangent_base_power=None,
        tangent_frequency=0.0,
    )

    assert gradients[2][0].item() == 0.0
    assert gradients[3][0].item() == 0.0
    assert gradients[4][0].item() == 0.0
    assert gradients[2][1].item() != 0.0
    torch.testing.assert_close(
        tangent_trans[0],
        gradients[2][1] * thickness_tangent[1],
        rtol=2.0e-5,
        atol=2.0e-6,
    )
    torch.testing.assert_close(
        tangent_scaled[0],
        2.0 * tangent_trans[0],
        rtol=2.0e-5,
        atol=2.0e-6,
    )
    assert state.bits.tolist() == [0]


def test_transmission_wall_product_pins_left_to_right_three_wall_product() -> None:
    layer_thickness = torch.tensor([0.24, 0.07, 0.13], device="cuda")
    layer_eps = torch.tensor([2.4, 5.7, 8.2], device="cuda")
    layer_sigma = torch.tensor([0.08, 0.012, 0.045], device="cuda")
    normals = torch.tensor(
        [
            [-1.0, 0.0, 0.0],
            [-0.8, -0.6, 0.0],
            [-0.35, -0.9367497, 0.0],
        ],
        device="cuda",
    )

    def evaluate(order: list[int]) -> torch.Tensor:
        depth = len(order)
        valid = torch.ones((1, depth), device="cuda", dtype=torch.bool)
        inputs = (
            valid,
            torch.tensor([depth], device="cuda", dtype=torch.int32),
            torch.tensor([True], device="cuda"),
            torch.tensor([[1.0, 0.0, 0.0]], device="cuda"),
            normals[order].reshape(1, depth, 3),
            torch.arange(depth, device="cuda", dtype=torch.int32).reshape(1, depth),
            torch.tensor(order, device="cuda", dtype=torch.int32),
            torch.zeros(3, device="cuda", dtype=torch.int32),
            torch.arange(3, device="cuda", dtype=torch.int32),
            torch.ones(3, device="cuda", dtype=torch.int32),
            layer_thickness,
            layer_eps,
            layer_sigma,
            torch.ones(3, device="cuda"),
            torch.tensor([[0.0, 1.0, 0.0]], device="cuda"),
            torch.ones(1, device="cuda"),
        )
        state = create_capacity_failure_state(valid)
        result = mc_transmission_wall_product(*inputs, state, frequency_hz=3.5e9)
        assert state.bits.tolist() == [0]
        return result.transmittance[0]

    factors = [evaluate([material]) for material in (1, 1, 2)]
    combined = evaluate([1, 1, 2])
    left_associated = (factors[0] * factors[1]) * factors[2]
    right_associated = factors[0] * (factors[1] * factors[2])

    assert torch.equal(combined, left_associated)
    assert left_associated.item() != right_associated.item(), [
        factor.item() for factor in factors
    ]


@pytest.mark.parametrize(
    "frequency_hz", [0.0, float("nan"), float("inf"), -float("inf")]
)
def test_transmission_wall_product_rejects_nonfinite_or_zero_frequency(
    frequency_hz: float,
) -> None:
    inputs, state = _case()
    with pytest.raises(ValueError, match="frequency_hz must be finite and positive"):
        mc_transmission_wall_product(*inputs, state, frequency_hz=frequency_hz)


def test_transmission_wall_product_autograd_uses_native_family() -> None:
    inputs, state = _case(differentiable=True)
    frequency = torch.tensor(3.5e9, device="cuda", requires_grad=True)
    result = mc_transmission_wall_product_ad(
        *inputs, frequency, state, frequency_value=3.5e9
    )
    loss = result.scaled_power.sum() + result.transmittance.sum()
    loss.backward()

    for tensor in (inputs[3], inputs[4], *inputs[10:13], inputs[15], frequency):
        assert tensor.grad is not None
        assert torch.isfinite(tensor.grad).all()


def test_transmission_wall_product_ad_views_are_zero_copy_stride_aware() -> None:
    inputs, state = _case()
    expanded_scaled = torch.tensor([0.7], device="cuda").expand(3)
    expanded_trans = torch.tensor([-0.2], device="cuda").expand(3)
    assert expanded_scaled.stride() == (0,)
    strided_vjp = mc_transmission_wall_product_backward(
        *inputs,
        frequency_hz=3.5e9,
        failure_state=state,
        grad_scaled_power=expanded_scaled,
        grad_transmittance=expanded_trans,
    )
    contiguous_vjp = mc_transmission_wall_product_backward(
        *inputs,
        frequency_hz=3.5e9,
        failure_state=state,
        grad_scaled_power=expanded_scaled.clone(),
        grad_transmittance=expanded_trans.clone(),
    )
    for strided, contiguous in zip(strided_vjp, contiguous_vjp, strict=True):
        assert torch.equal(strided, contiguous)

    direction_storage = torch.zeros((3, 6), device="cuda")
    tangent_direction = direction_storage[:, ::2]
    tangent_direction.copy_(
        torch.tensor(
            [[0.0, 0.03, 0.0], [0.0, -0.02, 0.0], [0.0, 0.0, 0.0]],
            device="cuda",
        )
    )
    normal_storage = torch.zeros((3, 2, 6), device="cuda")
    tangent_normal = normal_storage[:, :, ::2]
    tangent_normal[0, 0, 1] = 0.01
    tangent_normal[1, 0, 1] = -0.03
    tangent_normal[1, 1, 1] = 0.02
    tangent_thickness = torch.tensor([0.002, 9.0], device="cuda")[::2]
    tangent_eps = torch.tensor([0.04, 9.0], device="cuda")[::2]
    tangent_sigma = torch.tensor([-0.001, 9.0], device="cuda")[::2]
    tangent_base = torch.tensor([0.1, 9.0, -0.2, 9.0, 0.3, 9.0], device="cuda")[::2]
    assert not tangent_direction.is_contiguous()
    assert not tangent_normal.is_contiguous()
    assert not tangent_base.is_contiguous()

    strided_jvp = mc_transmission_wall_product_jvp(
        *inputs,
        frequency_hz=3.5e9,
        failure_state=state,
        tangent_direction=tangent_direction,
        tangent_normal=tangent_normal,
        tangent_layer_thickness_m=tangent_thickness,
        tangent_layer_eps_r=tangent_eps,
        tangent_layer_sigma_e=tangent_sigma,
        tangent_base_power=tangent_base,
        tangent_frequency=2.0e6,
    )
    contiguous_jvp = mc_transmission_wall_product_jvp(
        *inputs,
        frequency_hz=3.5e9,
        failure_state=state,
        tangent_direction=tangent_direction.clone(),
        tangent_normal=tangent_normal.clone(),
        tangent_layer_thickness_m=tangent_thickness.clone(),
        tangent_layer_eps_r=tangent_eps.clone(),
        tangent_layer_sigma_e=tangent_sigma.clone(),
        tangent_base_power=tangent_base.clone(),
        tangent_frequency=2.0e6,
    )
    assert torch.equal(strided_jvp[0], contiguous_jvp[0])
    assert torch.equal(strided_jvp[1], contiguous_jvp[1])


def test_transmission_wall_product_has_one_owner_and_no_fallback(monkeypatch) -> None:
    for name in (
        "mc_transmission_wall_product",
        "mc_transmission_wall_product_backward",
        "mc_transmission_wall_product_jvp",
    ):
        assert (
            inspect.unwrap(getattr(transmission, name)).__globals__
            is transmission.__dict__
        )

    inputs, state = _case()
    requested: list[str] = []

    def missing(name: str):
        requested.append(name)
        raise RuntimeError("native wall-product symbol is required")

    monkeypatch.setattr(transmission, "required_symbol", missing)
    with pytest.raises(RuntimeError, match="native wall-product symbol is required"):
        mc_transmission_wall_product(*inputs, state, frequency_hz=3.5e9)
    assert requested == ["mc_transmission_wall_product"]


def test_transmission_wall_product_source_freezes_residency_and_reduction() -> None:
    source = (
        _ROOT / "native/channel/kernels/mc_transmission_wall_product.cu"
    ).read_text(encoding="utf-8")
    live_route = (
        _ROOT / "witwin/channel/montecarlo/basic.py"
    ).read_text(encoding="utf-8")
    assert "wall_product_shared_backward_kernel" in source
    assert "frequency_owner" in source
    assert "atomicAdd" not in source
    assert "atomicOr" in source
    assert "cudaStreamSynchronize" not in source
    assert "cudaMemcpy" not in source
    assert ".contiguous()" not in source
    assert "kTransmissionAdMaxDepth" not in source
    assert "mc_transmission_wall_product" in live_route


def test_transmission_wall_product_live_owner_ledgers_are_complete() -> None:
    inventory = json.loads(
        (
            _ROOT / "docs/dev/audit/phase13-current-native-owner-inventory.json"
        ).read_text(encoding="utf-8")
    )
    symbols = [
        "mc_transmission_wall_product",
        "mc_transmission_wall_product_backward",
        "mc_transmission_wall_product_jvp",
    ]
    rows = {row["symbol"]: row for row in inventory["symbols"]}
    for symbol in symbols:
        assert rows[symbol]["production_callers"]
        assert rows[symbol]["liveness"] == "live-static-production-consumer"
        historical_owner = "Channel" + " Native"
        assert rows[symbol]["numerical_owner"] == historical_owner

    ledger = json.loads(
        (
            _ROOT
            / "docs/dev/audit/adr-027-mc-transmission-wall-product-resource-ledger.json"
        ).read_text(encoding="utf-8")
    )
    assert ledger["status"] == (
        "live Phase M native family; flattened MC transmission route active"
    )
    assert ledger["ownership"]["live_callers"]
    assert ledger["numerical_contract"]["floating_point_atomics"] is False
    assert ledger["capacity_contract"]["ad_hit_capacity_limit"] is None
    assert ledger["acceptance"]["live_switch_performed"] is True
