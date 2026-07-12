import pytest
import torch

from tests.support.scenes import single_wall_reflection_scene
from witwin.channel_native import ReceiverGrid
from witwin.channel_native.core.kernels.extension import build_info
from witwin.channel_native.montecarlo.bdpt import Config, solve


def _grid() -> ReceiverGrid:
    return ReceiverGrid(
        origin=torch.tensor([5.0, -1.0, -0.5]),
        x_axis=torch.tensor([0.0, 1.0, 0.0]),
        y_axis=torch.tensor([0.0, 0.0, 1.0]),
        shape=(4, 4),
        spacing=(2.0 / 3.0, 1.0 / 3.0),
    )


def test_bdpt_component_maps_include_all_components_and_total():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for BDPT component maps")

    scene = single_wall_reflection_scene().add(_grid())
    result = solve(scene, Config(samples=512, seed=5, components={"los", "reflection", "diffraction"}))

    assert result.component_maps is not None
    assert set(result.component_maps) == {"los", "reflection", "diffraction"}
    total = result.component_maps["los"] + result.component_maps["reflection"] + result.component_maps["diffraction"]
    assert result.path_gain.shape == (1, 4, 4)
    torch.testing.assert_close(result.path_gain, total, rtol=1.0e-5, atol=1.0e-8)
    if build_info()["uses_raydn_native"]:
        assert torch.isfinite(result.component_maps["reflection"]).all()


def test_bdpt_component_maps_include_transmission_and_zero_scattering():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for BDPT component maps")
    if not build_info()["uses_raydn_native"]:
        pytest.skip("RayDN native transmission is not built")

    scene = single_wall_reflection_scene().add(_grid())
    components = {"los", "reflection", "diffraction", "transmission", "scattering"}
    result = solve(scene, Config(samples=512, seed=5, components=components))

    # (a) validates and every requested component is present in the maps.
    assert result.component_maps is not None
    assert set(result.component_maps) == components
    # (b) the wall sits between the transmitter and the grid, so the
    # transmission component carries real power now; scattering remains a
    # structurally valid zero until its wave lands.
    assert result.component_power["transmission"].item() > 0.0
    assert torch.count_nonzero(result.component_maps["transmission"]) > 0
    assert torch.count_nonzero(result.component_maps["scattering"]) == 0
    assert torch.count_nonzero(result.component_power["scattering"]) == 0
    total = (
        result.component_maps["los"]
        + result.component_maps["reflection"]
        + result.component_maps["diffraction"]
        + result.component_maps["transmission"]
        + result.component_maps["scattering"]
    )
    torch.testing.assert_close(result.path_gain, total, rtol=1.0e-5, atol=1.0e-8)
    # (c) truthful metadata status: transmission is live, scattering is not.
    assert result.metadata["components"]["transmission"] == "enabled"
    assert result.metadata["components"]["scattering"] == "enabled_no_paths"
    assert result.metadata["transmission"]["component_mask_bit"] == 8
    assert result.metadata["transmission"]["straight_chain_paths"] > 0


def test_bdpt_config_rejects_unknown_component():
    with pytest.raises(ValueError, match="components"):
        Config(components={"los", "teleportation"})
