# Copyright Xingyu Chen.
# Tests basic parity small scene.

import pytest
import torch

from tests.support.reference_channel import los_path_gain_reference
from tests.support.scenes import empty_space_los_scene, single_wall_reflection_scene
from witwin.channel.montecarlo.basic import Config, solve


def test_basic_solver_matches_empty_space_los_reference():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for MC basic parity")

    scene = empty_space_los_scene()
    result = solve(
        scene,
        Config(samples=256, seed=11, components={"los"}),
        reference_frequency_hz=3.0e9,
    )
    expected = los_path_gain_reference(
        scene,
        device=torch.device("cuda"),
        reference_frequency_hz=3.0e9,
    )

    torch.testing.assert_close(result.path_gain, expected, rtol=1e-5, atol=1e-8)


def test_single_wall_reflection_on_point_receivers_is_refused():
    # MC basic accumulates reflection on a ReceiverGrid only; with point
    # receivers it refuses instead of reporting an "enabled" zero.
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for MC basic parity")

    with pytest.raises(ValueError, match=r"computes \['reflection'\] only on a ReceiverGrid"):
        solve(
            single_wall_reflection_scene(),
            Config(samples=256, seed=11, components={"reflection"}),
            reference_frequency_hz=3.0e9,
        )
