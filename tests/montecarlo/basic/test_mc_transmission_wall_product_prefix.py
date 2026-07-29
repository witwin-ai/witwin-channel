# Copyright Xingyu Chen.
# Tests mc transmission wall product prefix.

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from witwin.channel.kernels.montecarlo import (
    mc_transmission_wall_product,
    mc_transmission_wall_product_backward,
    mc_transmission_wall_product_jvp,
)
from witwin.channel.runtime import CapacityFailureBit, create_capacity_failure_state


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is required"
)

_ROOT = Path(__file__).resolve().parents[3]
_FREQUENCY_HZ = 3.5e9


def _inputs(face_material: list[int]) -> tuple[torch.Tensor, ...]:
    valid = torch.ones((1, 3), device="cuda", dtype=torch.bool)
    return (
        valid,
        torch.tensor([3], device="cuda", dtype=torch.int32),
        torch.ones(1, device="cuda", dtype=torch.bool),
        torch.tensor([[1.0, 0.0, 0.0]], device="cuda"),
        torch.tensor([[[-1.0, 0.0, 0.0]] * 3], device="cuda"),
        torch.tensor([[0, 1, 2]], device="cuda", dtype=torch.int32),
        torch.tensor(face_material, device="cuda", dtype=torch.int32),
        torch.zeros(3, device="cuda", dtype=torch.int32),
        torch.tensor([0, 1, 2], device="cuda", dtype=torch.int32),
        torch.ones(3, device="cuda", dtype=torch.int32),
        # The middle wall is deliberately opaque. The stable RayD layer-stack
        # contract guarantees clean float32 underflow for this supported
        # conductivity/thickness pair.
        torch.tensor([0.08, 10.0, 0.12], device="cuda"),
        torch.tensor([3.2, 4.0, 5.1], device="cuda"),
        torch.tensor([0.015, 1.0e9, 0.025], device="cuda"),
        torch.ones(3, device="cuda"),
        torch.tensor([[0.0, 1.0, 0.0]], device="cuda"),
        torch.tensor([2.0], device="cuda"),
    )


def _backward(
    inputs: tuple[torch.Tensor, ...], failure_state: object
) -> tuple[torch.Tensor, ...]:
    return mc_transmission_wall_product_backward(
        *inputs,
        frequency_hz=_FREQUENCY_HZ,
        failure_state=failure_state,
        grad_scaled_power=torch.ones(1, device="cuda"),
        grad_transmittance=torch.ones(1, device="cuda"),
    )


def _jvp(
    inputs: tuple[torch.Tensor, ...], failure_state: object
) -> tuple[torch.Tensor, torch.Tensor]:
    return mc_transmission_wall_product_jvp(
        *inputs,
        frequency_hz=_FREQUENCY_HZ,
        failure_state=failure_state,
        tangent_direction=torch.ones_like(inputs[3]),
        tangent_normal=torch.ones_like(inputs[4]),
        tangent_layer_thickness_m=torch.ones_like(inputs[10]),
        tangent_layer_eps_r=torch.ones_like(inputs[11]),
        tangent_layer_sigma_e=torch.ones_like(inputs[12]),
        tangent_base_power=torch.ones_like(inputs[15]),
        tangent_frequency=1.0,
    )


def _assert_all_zero(values: tuple[torch.Tensor, ...]) -> None:
    for value in values:
        assert torch.count_nonzero(value).item() == 0


def test_zero_transmission_stops_before_later_ordinary_blocker() -> None:
    inputs = _inputs([0, 1, -1])
    state = create_capacity_failure_state(inputs[0])

    result = mc_transmission_wall_product(
        *inputs, state, frequency_hz=_FREQUENCY_HZ
    )

    assert state.bits.tolist() == [0]
    assert result.wall_count.tolist() == [2]
    assert result.transmittance.tolist() == [0.0]
    assert result.scaled_power.tolist() == [0.0]
    # The later blocker was preflighted but was never part of the effective
    # numerical prefix, matching the former depth march.
    assert result.penetrated.tolist() == [True]

    backward_state = create_capacity_failure_state(inputs[0])
    _assert_all_zero(_backward(inputs, backward_state))
    assert backward_state.bits.tolist() == [0]

    jvp_state = create_capacity_failure_state(inputs[0])
    _assert_all_zero(_jvp(inputs, jvp_state))
    assert jvp_state.bits.tolist() == [0]


def test_ordinary_blocker_stops_prefix_and_makes_all_ad_inert() -> None:
    inputs_list = list(_inputs([0, -1, 2]))
    inputs_list[4][0, 2].fill_(float("nan"))
    inputs_list[10][2] = float("nan")
    inputs_list[11][2] = float("nan")
    inputs_list[12][2] = float("nan")
    inputs = tuple(inputs_list)

    primal_state = create_capacity_failure_state(inputs[0])
    result = mc_transmission_wall_product(
        *inputs, primal_state, frequency_hz=_FREQUENCY_HZ
    )
    assert primal_state.bits.tolist() == [0]
    assert result.wall_count.tolist() == [2]
    assert result.penetrated.tolist() == [False]
    _assert_all_zero((result.scaled_power, result.transmittance))

    backward_state = create_capacity_failure_state(inputs[0])
    _assert_all_zero(_backward(inputs, backward_state))
    assert backward_state.bits.tolist() == [0]

    jvp_state = create_capacity_failure_state(inputs[0])
    _assert_all_zero(_jvp(inputs, jvp_state))
    assert jvp_state.bits.tolist() == [0]


def test_nan_transmission_stops_before_later_ordinary_blocker() -> None:
    inputs_list = list(_inputs([0, -1, 2]))
    inputs_list[4][0, 0].fill_(float("nan"))
    inputs = tuple(inputs_list)
    state = create_capacity_failure_state(inputs[0])

    result = mc_transmission_wall_product(
        *inputs, state, frequency_hz=_FREQUENCY_HZ
    )

    assert state.bits.tolist() == [0]
    assert result.wall_count.tolist() == [1]
    assert result.penetrated.tolist() == [True]
    assert torch.isnan(result.transmittance).all()
    assert torch.isnan(result.scaled_power).all()


def test_nan_transmission_excludes_later_eligible_walls_from_ad() -> None:
    inputs_list = list(_inputs([0, 1, 2]))
    inputs_list[4][0, 0].fill_(float("nan"))
    inputs = tuple(inputs_list)

    primal_state = create_capacity_failure_state(inputs[0])
    result = mc_transmission_wall_product(
        *inputs, primal_state, frequency_hz=_FREQUENCY_HZ
    )
    assert primal_state.bits.tolist() == [0]
    assert result.wall_count.tolist() == [1]
    assert result.penetrated.tolist() == [True]

    backward_state = create_capacity_failure_state(inputs[0])
    gradients = _backward(inputs, backward_state)
    assert backward_state.bits.tolist() == [0]
    assert torch.count_nonzero(gradients[1][:, 1:]).item() == 0
    for shared_gradient in gradients[2:5]:
        assert shared_gradient[1:].tolist() == [0.0, 0.0]

    zero_tangent_state = create_capacity_failure_state(inputs[0])
    zero_tangents = mc_transmission_wall_product_jvp(
        *inputs,
        frequency_hz=_FREQUENCY_HZ,
        failure_state=zero_tangent_state,
        tangent_direction=None,
        tangent_normal=None,
        tangent_layer_thickness_m=None,
        tangent_layer_eps_r=None,
        tangent_layer_sigma_e=None,
        tangent_base_power=None,
        tangent_frequency=0.0,
    )
    later_tangent_state = create_capacity_failure_state(inputs[0])
    later_eps_tangent = torch.tensor([0.0, 1.0, 1.0], device="cuda")
    later_tangents = mc_transmission_wall_product_jvp(
        *inputs,
        frequency_hz=_FREQUENCY_HZ,
        failure_state=later_tangent_state,
        tangent_direction=None,
        tangent_normal=None,
        tangent_layer_thickness_m=None,
        tangent_layer_eps_r=later_eps_tangent,
        tangent_layer_sigma_e=None,
        tangent_base_power=None,
        tangent_frequency=0.0,
    )
    assert zero_tangent_state.bits.tolist() == [0]
    assert later_tangent_state.bits.tolist() == [0]
    for baseline, candidate in zip(zero_tangents, later_tangents, strict=True):
        assert torch.equal(baseline.view(torch.int32), candidate.view(torch.int32))


@pytest.mark.parametrize("termination", ["zero", "blocker"])
def test_later_discrete_corruption_is_preflighted_before_prefix_evaluation(
    termination: str,
) -> None:
    face_material = [0, 1, 2] if termination == "zero" else [-1, 1, 2]
    inputs_list = list(_inputs(face_material))
    inputs_list[5][0, 2] = 2_000_000_000
    inputs = tuple(inputs_list)
    expected_bit = int(CapacityFailureBit.PAIR_CONTRACT_ERROR)

    primal_state = create_capacity_failure_state(inputs[0])
    result = mc_transmission_wall_product(
        *inputs, primal_state, frequency_hz=_FREQUENCY_HZ
    )
    _assert_all_zero(
        (
            result.scaled_power,
            result.transmittance,
            result.wall_count,
            result.penetrated,
        )
    )
    assert primal_state.bits.tolist() == [expected_bit]

    backward_state = create_capacity_failure_state(inputs[0])
    _assert_all_zero(_backward(inputs, backward_state))
    assert backward_state.bits.tolist() == [expected_bit]

    jvp_state = create_capacity_failure_state(inputs[0])
    _assert_all_zero(_jvp(inputs, jvp_state))
    assert jvp_state.bits.tolist() == [expected_bit]


def test_wall_product_source_uses_full_preflight_and_effective_prefix() -> None:
    source = (
        _ROOT / "native/channel/kernels/mc_transmission_wall_product.cu"
    ).read_text(encoding="utf-8")

    assert source.count("for (int slot = 0; slot < count; ++slot)") == 1
    assert source.count(
        "for (int slot = 0; slot < preflight.first_blocker; ++slot)"
    ) == 4
    assert source.count(
        "for (int other = 0; other < effective_count; ++other)"
    ) == 2
    assert "wall_count[row] = preflight.first_blocker + 1" in source
    assert source.count("stopped_before_next_wall") == 12
    assert source.count("if (!(wall_value > 0.0f))") == 2
    assert "if (!(value.value > 0.0f))" in source
    assert "if (!(wall.value > 0.0f))" in source