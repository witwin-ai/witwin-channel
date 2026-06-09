import pytest
import torch

from witwin.channel_native import ReceiverPoint, Scene, Transmitter
from witwin.channel_native.montecarlo.basic import Config, solve


def _differentiable_los_scene() -> tuple[Scene, torch.Tensor]:
    rx_position = torch.tensor([3.0, 4.0, 0.0], device="cuda", requires_grad=True)
    scene = Scene(
        structures=[],
        transmitters=[Transmitter(position=torch.tensor([0.0, 0.0, 0.0]))],
        receivers=[ReceiverPoint(position=rx_position)],
        frequency=3.0e9,
    )
    return scene, rx_position


def test_basic_fixed_topology_vjp_propagates_receiver_position_gradients():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for MC basic fixed-topology AD")

    scene, rx_position = _differentiable_los_scene()
    result = solve(scene, Config(samples=128, components={"los"}, ad_mode="vjp"))

    loss = result.path_gain.sum()
    loss.backward()

    assert rx_position.grad is not None
    assert rx_position.grad.is_cuda
    assert torch.isfinite(rx_position.grad).all()
    assert result.metadata["kernel"]["ad_status"] == "vjp"
    assert result.metadata["kernel"]["backward_launch_count"] == 1
    assert result.metadata["kernel"]["tape_bytes"] > 0


def test_basic_fixed_topology_jvp_preserves_forward_ad_tangent():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for MC basic fixed-topology AD")

    primal = torch.tensor([3.0, 4.0, 0.0], device="cuda", requires_grad=True)
    tangent = torch.tensor([1.0, 0.0, 0.0], device="cuda")
    with torch.autograd.forward_ad.dual_level():
        dual_position = torch.autograd.forward_ad.make_dual(primal, tangent)
        scene = Scene(
            structures=[],
            transmitters=[Transmitter(position=torch.tensor([0.0, 0.0, 0.0]))],
            receivers=[ReceiverPoint(position=dual_position)],
            frequency=3.0e9,
        )
        result = solve(scene, Config(samples=128, components={"los"}, ad_mode="jvp"))
        value, jvp = torch.autograd.forward_ad.unpack_dual(result.path_gain)

    assert value.is_cuda
    assert jvp is not None
    assert torch.isfinite(jvp).all()
    assert result.metadata["kernel"]["ad_status"] == "jvp"
    assert result.metadata["kernel"]["jvp_launch_count"] == 1
    assert result.metadata["kernel"]["tape_bytes"] > 0
