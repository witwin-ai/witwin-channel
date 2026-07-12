import pytest
from dataclasses import fields

from witwin.channel_native.path import Config
from witwin.channel_native.path.solver import _metadata


def test_path_accepts_supported_reflection_depths_and_rejects_above_capability():
    for depth in range(1, 6):
        assert Config(max_depth=depth, components={"reflection"}).max_depth == depth
    with pytest.raises(RuntimeError, match="max_depth <= 5"):
        Config(max_depth=6, components={"reflection"})


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
    assert metadata["component_max_depth"] == {
        "los": 0,
        "reflection": -1,
        "diffraction": -1,
        "transmission": -1,
        "scattering": -1,
    }
    assert metadata["requested_config"]["components"] == ["los"]
    assert metadata["effective_config"]["max_depth"] == 0
    assert set(metadata["requested_config"]) == {field.name for field in fields(Config)}
