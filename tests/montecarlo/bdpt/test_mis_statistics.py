import statistics

import pytest
import torch

from tests.support.scenes import wedge_diffraction_scene
from witwin.channel.deployment import build_info
from witwin.channel.montecarlo.bdpt import Config, solve


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
                reference_frequency_hz=3.0e9,
            ).path_gain.sum()
        )
        for seed in range(6)
    ]


def test_bdpt_diffraction_estimate_is_sample_count_and_seed_invariant():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for BDPT statistics")
    if not build_info()["uses_rayd_native"]:
        pytest.skip("RayD native diffraction is not built")

    # ADR-018: standalone diffraction is now a deterministic enumerated estimate,
    # so it no longer depends on the Monte Carlo sample budget or seed. Distinct
    # sample counts and seeds collapse to the identical value, replacing the
    # retired variance-shrinks-with-samples convergence check on the stochastic
    # Keller sampler.
    small = _estimates(samples=256, mis="power_heuristic")
    large = _estimates(samples=1024, mis="power_heuristic")

    reference = small[0]
    assert reference > 0.0
    for value in small + large:
        assert value == pytest.approx(reference, rel=1e-6)


def test_bdpt_diffraction_mis_on_off_means_agree_across_seeds():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for BDPT statistics")
    if not build_info()["uses_rayd_native"]:
        pytest.skip("RayD native diffraction is not built")

    enabled = statistics.mean(_estimates(samples=1024, mis="power_heuristic"))
    disabled = statistics.mean(_estimates(samples=1024, mis="none"))

    assert abs(enabled - disabled) / max(abs(enabled), abs(disabled)) < 0.1
