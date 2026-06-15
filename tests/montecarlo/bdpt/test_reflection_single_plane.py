import pytest
import torch

from tests.support.scenes import same_side_wall_reflection_scene
from witwin.channel_native import ReceiverGrid
from witwin.channel_native.core.kernels.extension import build_info
from witwin.channel_native.montecarlo.bdpt import Config, solve
from witwin.channel_native.montecarlo.bdpt import solver as bdpt_solver


def _grid() -> ReceiverGrid:
    return ReceiverGrid(
        origin=torch.tensor([0.0, -1.0, 0.0]),
        x_axis=torch.tensor([0.0, 1.0, 0.0]),
        y_axis=torch.tensor([0.0, 0.0, 1.0]),
        shape=(4, 4),
        spacing=(2.0 / 3.0, 1.0 / 3.0),
    )


def test_bdpt_single_plane_reflection_returns_nonzero_native_component_when_available():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for BDPT reflection")
    if not build_info()["uses_raydn_native"]:
        pytest.skip("RayDN native reflection is not built")

    result = solve(same_side_wall_reflection_scene().add(_grid()), Config(samples=2048, seed=5, components={"reflection"}))

    assert result.component_maps is not None
    reflection = result.component_maps["reflection"]
    assert reflection.shape == (1, 4, 4)
    assert torch.isfinite(reflection).all()
    assert torch.count_nonzero(reflection > 0.0).item() > 0
    assert result.component_power["reflection"].item() > 0.0
    torch.testing.assert_close(result.component_power["reflection"], reflection.sum(), rtol=1e-5, atol=1e-8)
    assert result.metadata["components"]["reflection"] == "enabled"


def test_bdpt_point_receiver_reflection_returns_native_component_when_available():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for BDPT point reflection")
    if not build_info()["uses_raydn_native"]:
        pytest.skip("RayDN native reflection is not built")

    result = solve(
        same_side_wall_reflection_scene(),
        Config(samples=2048, seed=5, components={"reflection"}, receiver_strategy="point_sphere"),
    )

    assert result.path_gain.shape == (1, 1)
    assert result.component_maps is None
    assert torch.isfinite(result.path_gain).all()
    assert result.component_power["reflection"].item() > 0.0
    torch.testing.assert_close(result.path_gain.sum(), result.component_power["reflection"], rtol=1e-5, atol=1e-8)
    assert result.metadata["components"]["reflection"] == "enabled"


def test_bdpt_point_reflection_solver_does_not_use_image_source_path_export():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for BDPT reflection")
    if not build_info()["uses_raydn_native"]:
        pytest.skip("RayDN native reflection is not built")

    assert not hasattr(bdpt_solver, "reflection_paths_order1")

    solve(
        same_side_wall_reflection_scene(),
        Config(samples=64, seed=5, components={"reflection"}, receiver_strategy="point_sphere"),
    )


def test_bdpt_grid_reflection_solver_does_not_use_image_source_path_export():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for BDPT reflection")
    if not build_info()["uses_raydn_native"]:
        pytest.skip("RayDN native reflection is not built")

    assert not hasattr(bdpt_solver, "reflection_paths_order1")

    result = solve(
        same_side_wall_reflection_scene().add(_grid()),
        Config(samples=512, seed=5, components={"reflection"}),
    )

    assert result.component_power["reflection"].item() > 0.0


@pytest.mark.parametrize(
    ("samples", "relative_tolerance"),
    [
        (4096, 0.20),
        (16384, 0.10),
    ],
)
def test_bdpt_single_plane_reflection_converges_to_maintained_reference(samples, relative_tolerance):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for BDPT reflection convergence")
    if not build_info()["uses_raydn_native"]:
        pytest.skip("RayDN native reflection is not built")

    result = solve(
        same_side_wall_reflection_scene().add(_grid()),
        Config(samples=samples, seed=5, components={"reflection"}),
    )

    observed = result.component_power["reflection"].detach().cpu()
    reference = torch.tensor(3.98e-06, dtype=observed.dtype)
    torch.testing.assert_close(observed, reference, rtol=relative_tolerance, atol=1.0e-10)
