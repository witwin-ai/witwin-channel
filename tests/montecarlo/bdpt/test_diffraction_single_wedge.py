import pytest
import torch

from tests.support.scenes import wedge_diffraction_scene
from witwin.channel_native import ReceiverGrid
from witwin.channel_native.core.kernels.extension import build_info
from witwin.channel_native.montecarlo.bdpt import Config, solve
from witwin.channel_native.montecarlo.bdpt import solver as bdpt_solver


def _grid() -> ReceiverGrid:
    return ReceiverGrid(
        origin=torch.tensor([3.0, -1.0, -0.5]),
        x_axis=torch.tensor([0.0, 1.0, 0.0]),
        y_axis=torch.tensor([0.0, 0.0, 1.0]),
        shape=(4, 4),
        spacing=(2.0 / 3.0, 1.0 / 3.0),
    )


def test_bdpt_single_wedge_diffraction_returns_finite_native_component_when_available():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for BDPT diffraction")
    if not build_info()["uses_raydn_native"]:
        pytest.skip("RayDN native diffraction is not built")

    result = solve(wedge_diffraction_scene().add(_grid()), Config(samples=512, seed=7, components={"diffraction"}))

    assert result.component_maps is not None
    diffraction = result.component_maps["diffraction"]
    assert diffraction.shape == (1, 4, 4)
    assert torch.isfinite(diffraction).all()
    assert torch.count_nonzero(diffraction > 0.0).item() > 0
    assert result.component_power["diffraction"].item() > 0.0
    torch.testing.assert_close(result.component_power["diffraction"], diffraction.sum(), rtol=1e-5, atol=1e-8)
    assert result.metadata["components"]["diffraction"] == "enabled"


def test_bdpt_single_wedge_diffraction_does_not_use_path_block_sampler():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for BDPT diffraction")
    if not build_info()["uses_raydn_native"]:
        pytest.skip("RayDN native diffraction is not built")

    assert not hasattr(bdpt_solver, "bdpt_sample_path_block")

    result = solve(wedge_diffraction_scene().add(_grid()), Config(samples=512, seed=7, components={"diffraction"}))

    assert result.component_power["diffraction"].item() > 0.0


def test_bdpt_single_wedge_diffraction_fixed_seed_is_stable():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for BDPT diffraction reproducibility")
    if not build_info()["uses_raydn_native"]:
        pytest.skip("RayDN native diffraction is not built")

    scene = wedge_diffraction_scene().add(_grid())
    first = solve(scene, Config(samples=512, seed=7, components={"diffraction"}))
    second = solve(scene, Config(samples=512, seed=7, components={"diffraction"}))
    changed = solve(scene, Config(samples=512, seed=8, components={"diffraction"}))

    torch.testing.assert_close(first.component_maps["diffraction"], second.component_maps["diffraction"])
    assert not torch.equal(first.component_maps["diffraction"], changed.component_maps["diffraction"])


def test_bdpt_single_wedge_diffraction_uses_original_direct_keller_split():
    assert bdpt_solver._diffraction_sample_split(16) == (6, 5, 0)
    assert bdpt_solver._diffraction_sample_split(17) == (6, 6, 0)
    assert bdpt_solver._diffraction_sample_split(512) == (171, 171, 0)


@pytest.mark.parametrize(
    ("samples", "relative_tolerance"),
    [
        (4096, 0.35),
        (16384, 0.20),
    ],
)
def test_bdpt_single_wedge_point_diffraction_converges_to_maintained_reference(samples, relative_tolerance):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for BDPT diffraction convergence")
    if not build_info()["uses_raydn_native"]:
        pytest.skip("RayDN native diffraction is not built")

    result = solve(
        wedge_diffraction_scene(),
        Config(samples=samples, seed=7, components={"diffraction"}, receiver_strategy="point_sphere"),
    )

    observed = result.component_power["diffraction"].detach().cpu()
    reference = torch.tensor(1.25e-04, dtype=observed.dtype)
    torch.testing.assert_close(observed, reference, rtol=relative_tolerance, atol=1.0e-8)


def test_bdpt_diffraction_point_receiver_returns_native_component_without_path_block_fallback():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for BDPT diffraction")
    if not build_info()["uses_raydn_native"]:
        pytest.skip("RayDN native diffraction is not built")

    assert not hasattr(bdpt_solver, "bdpt_sample_path_block")

    result = solve(
        wedge_diffraction_scene(),
        Config(samples=256, seed=7, components={"diffraction"}, receiver_strategy="point_sphere"),
    )

    assert result.component_maps is None
    assert result.path_gain.shape == (1, 1)
    assert torch.isfinite(result.path_gain).all()
    assert result.component_power["diffraction"].item() > 0.0
    torch.testing.assert_close(
        result.path_gain,
        result.component_power["diffraction"].reshape(1, 1),
        rtol=1e-5,
        atol=1e-8,
    )
