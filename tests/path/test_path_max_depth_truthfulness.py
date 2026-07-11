import pytest
from dataclasses import fields

from witwin.channel_native.path import Config
from witwin.channel_native.path.solver import _metadata


@pytest.mark.parametrize("component", ["reflection", "diffraction"])
def test_path_rejects_unimplemented_high_order_scattering_at_config_time(component):
    with pytest.raises(RuntimeError, match="max_depth <= 1"):
        Config(max_depth=2, components={component})


def test_path_los_only_reports_requested_and_effective_depth():
    config = Config(max_depth=4, components={"los"})
    metadata = _metadata(
        config=config,
        path_count=1,
        valid_contribution_count=1,
        reflection_available=True,
        diffraction_available=True,
        path_native_available=True,
    )

    assert metadata["requested_max_depth"] == 4
    assert metadata["effective_max_depth"] == 0
    assert metadata["component_max_depth"] == {"los": 0, "reflection": -1, "diffraction": -1}
    assert metadata["requested_config"]["components"] == ["los"]
    assert metadata["effective_config"]["max_depth"] == 0
    assert set(metadata["requested_config"]) == {field.name for field in fields(Config)}
