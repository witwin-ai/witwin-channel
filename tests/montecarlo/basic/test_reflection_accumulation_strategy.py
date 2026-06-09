import pytest

from witwin.channel_native.montecarlo.basic import Config
from witwin.channel_native.montecarlo.basic.metadata import make_solver_metadata


def test_reflection_accumulation_strategy_accepts_known_values():
    for strategy in ("auto", "atomic", "staged", "compact", "streaming_planar"):
        config = Config(reflection_accumulation_strategy=strategy)

        assert config.reflection_accumulation_strategy == strategy


def test_reflection_accumulation_strategy_rejects_unknown_value():
    with pytest.raises(ValueError, match="reflection_accumulation_strategy"):
        Config(reflection_accumulation_strategy="bad")


def test_reflection_accumulation_thresholds_must_be_non_negative():
    with pytest.raises(ValueError, match="reflection_compact_min_samples"):
        Config(reflection_compact_min_samples=-1)
    with pytest.raises(ValueError, match="reflection_staged_min_samples_per_cell"):
        Config(reflection_staged_min_samples_per_cell=-1)


def test_reflection_accumulation_settings_are_reported_in_metadata():
    config = Config(
        reflection_accumulation_strategy="atomic",
        reflection_compact_min_samples=123,
        reflection_staged_min_samples_per_cell=7,
    )

    metadata = make_solver_metadata(
        config=config,
        path_count=0,
        valid_contribution_count=0,
        reflection_available=True,
        diffraction_available=True,
    )

    assert metadata["reflection_accumulation_strategy"] == "atomic"
    assert metadata["reflection_compact_min_samples"] == 123
    assert metadata["reflection_staged_min_samples_per_cell"] == 7
