import pytest
import torch

from tests.support.scenes import coupled_wall_wedge_scene
from witwin.channel import capabilities
from witwin.channel.deployment import build_info
from witwin.channel.montecarlo.bdpt import Config, solve


def test_bdpt_exports_coupled_paths_with_bidirectional_discrete_mass():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for coupled BDPT")
    if not build_info()["uses_rayd_native"]:
        pytest.skip("RayD native coupling is not built")

    result = solve(
        coupled_wall_wedge_scene(),
        Config(
            samples=64,
            max_depth=2,
            coupled_paths=True,
            export_paths=True,
            components={"reflection", "diffraction"},
        ),
        reference_frequency_hz=3.0e9,
    )

    assert result.path_samples is not None
    samples = result.path_samples
    coupled = samples.valid & (samples.component_id == 2) & (samples.light_depth == 2)
    assert bool(coupled.any())
    assert bool(torch.isfinite(samples.contribution[coupled]).all())
    torch.testing.assert_close(
        samples.pdf[coupled], torch.ones_like(samples.pdf[coupled])
    )
    torch.testing.assert_close(
        samples.mis_weight[coupled], torch.ones_like(samples.mis_weight[coupled])
    )
    assert capabilities()["solvers"]["montecarlo_bdpt"][
        "supports_reflection_diffraction_coupling"
    ]
