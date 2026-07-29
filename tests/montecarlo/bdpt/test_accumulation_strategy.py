# Copyright Xingyu Chen.
# Tests accumulation strategy.

import pytest
import torch

from tests.support.scenes import empty_space_los_scene
from witwin.channel.montecarlo.bdpt import Config, solve


@pytest.mark.parametrize("strategy", ["atomic", "staged", "compact", "auto"])
def test_bdpt_accumulation_strategy_is_reported(strategy):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for BDPT accumulation")

    result = solve(
        empty_space_los_scene(),
        Config(samples=64, components={"los"}, accumulation_strategy=strategy),
        reference_frequency_hz=3.0e9,
    )

    expected = "atomic" if strategy == "auto" else strategy
    assert result.metadata["accumulation_strategy"] == expected
    assert result.metadata["workspace_bytes"] >= 0


def test_bdpt_explicit_accumulation_strategies_match_atomic_result():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for BDPT accumulation")

    scene = empty_space_los_scene()
    atomic = solve(
        scene,
        Config(samples=64, components={"los"}, accumulation_strategy="atomic"),
        reference_frequency_hz=3.0e9,
    )

    for strategy in ("staged", "compact"):
        result = solve(
            scene,
            Config(samples=64, components={"los"}, accumulation_strategy=strategy),
            reference_frequency_hz=3.0e9,
        )

        assert result.metadata["accumulation_strategy"] == strategy
        torch.testing.assert_close(
            result.path_gain, atomic.path_gain, rtol=2.0e-6, atol=1.0e-12
        )
        torch.testing.assert_close(
            result.component_power["los"],
            atomic.component_power["los"],
            rtol=2.0e-6,
            atol=1.0e-12,
        )
        torch.testing.assert_close(
            result.component_power["reflection"], atomic.component_power["reflection"]
        )
        torch.testing.assert_close(
            result.component_power["diffraction"], atomic.component_power["diffraction"]
        )


def test_bdpt_accumulation_strategy_metadata_reports_native_kernel_variant():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for BDPT accumulation")

    scene = empty_space_los_scene()
    expected = {
        "atomic": "atomic_add",
        "staged": "cell_reduce",
        "compact": "compact_atomic_add",
    }

    for strategy, kernel_strategy in expected.items():
        result = solve(
            scene,
            Config(samples=64, components={"los"}, accumulation_strategy=strategy),
            reference_frequency_hz=3.0e9,
        )

        assert result.metadata["kernel"]["accumulation_strategy"] == kernel_strategy