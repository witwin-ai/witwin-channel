import math
import statistics

import pytest
import torch

from tests.support.scenes import wedge_diffraction_scene
from witwin.channel_native.core.kernels.extension import build_info
from witwin.channel_native.montecarlo.bdpt import Config, solve


def _estimates(*, samples: int, mis: str) -> list[float]:
    return [
        float(
            solve(
                wedge_diffraction_scene(),
                Config(
                    samples=samples,
                    seed=seed,
                    components={"diffraction"},
                    receiver_strategy="point_sphere",
                    mis=mis,
                ),
            ).path_gain.sum()
        )
        for seed in range(6)
    ]


def _ci_half_width(values: list[float]) -> float:
    return 1.96 * statistics.stdev(values) / math.sqrt(len(values))


def test_bdpt_diffraction_four_x_samples_shrinks_confidence_interval():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for BDPT statistics")
    if not build_info()["uses_raydn_native"]:
        pytest.skip("RayDN native diffraction is not built")

    small = _estimates(samples=256, mis="power_heuristic")
    large = _estimates(samples=1024, mis="power_heuristic")

    assert _ci_half_width(large) < _ci_half_width(small)


def test_bdpt_diffraction_mis_on_off_means_agree_across_seeds():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for BDPT statistics")
    if not build_info()["uses_raydn_native"]:
        pytest.skip("RayDN native diffraction is not built")

    enabled = statistics.mean(_estimates(samples=1024, mis="power_heuristic"))
    disabled = statistics.mean(_estimates(samples=1024, mis="none"))

    assert abs(enabled - disabled) / max(abs(enabled), abs(disabled)) < 0.1
