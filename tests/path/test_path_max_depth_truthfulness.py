# Copyright Xingyu Chen.
# Tests path max depth truthfulness.

import pytest
from witwin.channel.path import Config
from witwin.channel.path import _metadata


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
        reflection_available=True,
        diffraction_available=True,
        path_native_available=True,
    )

    assert metadata["effective_max_depth"] == 0
    assert metadata["component_max_depth"] == {
        "los": 0,
        "reflection": -1,
        "diffraction": -1,
        "transmission": -1,
        "scattering": -1,
    }