import pytest

from witwin.channel_native.core.kernels.metadata import (
    REQUIRED_METADATA_FIELDS,
    make_metadata,
    validate_metadata,
)


def test_make_metadata_includes_required_fields():
    metadata = make_metadata(
        primitive="trace_los_field",
        forward_launch_count=1,
        rayd_native=False,
        accumulation_strategy="atomic_add",
        ad_status="none",
    )

    assert set(REQUIRED_METADATA_FIELDS).issubset(metadata)
    assert metadata["primitive"] == "trace_los_field"
    assert metadata["forward_launch_count"] == 1
    assert metadata["rayd_native"] is False
    assert "fusion_debt" not in metadata
    validate_metadata(metadata)


def test_validate_metadata_rejects_missing_field():
    metadata = make_metadata(primitive="trace_los_field")
    metadata.pop("launch_count")

    with pytest.raises(ValueError, match="launch_count"):
        validate_metadata(metadata)


def test_validate_metadata_rejects_invalid_accumulation_strategy():
    metadata = make_metadata(primitive="trace_los_field")
    metadata["accumulation_strategy"] = "python_loop"

    with pytest.raises(ValueError, match="accumulation_strategy"):
        validate_metadata(metadata)


def test_validate_metadata_rejects_negative_counts():
    metadata = make_metadata(primitive="trace_los_field")
    metadata["forward_launch_count"] = -1

    with pytest.raises(ValueError, match="forward_launch_count"):
        validate_metadata(metadata)


def test_validate_metadata_rejects_legacy_unsupported_ad_status():
    with pytest.raises(ValueError, match="ad_status"):
        make_metadata(primitive="trace_los_field", ad_status="unsupported")
