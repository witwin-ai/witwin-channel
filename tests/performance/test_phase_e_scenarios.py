# Copyright Xingyu Chen.
# Tests performance scenario definitions.

from __future__ import annotations

import json

import pytest

from benchmarks.phase_e_scenarios import (
    FullScenarioAssetError,
    SCENARIO_MANIFEST_PATH,
    SCENE_ROOT_ENV,
    build_scenario,
    load_manifest,
    scenario_names,
)


def test_committed_manifest_has_five_truthful_scenario_identities():
    manifest = load_manifest()

    assert SCENARIO_MANIFEST_PATH.is_file()
    assert manifest["schema"] == {
        "name": "witwin.channel.phase_e_scenarios",
        "version": "1.0.0",
    }
    assert scenario_names() == (
        "analytic",
        "three_cube",
        "terrain",
        "munich_full",
        "sf_full",
    )
    assert manifest["asset_policy"] == {
        "external_root_env": SCENE_ROOT_ENV,
        "full_asset_missing": "fail",
        "reduced_fallback": False,
    }
    assert [row["id"] for row in manifest["endpoint_profiles"]] == [
        "point-1x1",
        "points-8x1k",
        "points-16x1k",
    ]
    assert manifest["grid_profiles"] == [[128, 128], [512, 512]]
    assert all(
        len(row["source_sha256"]) == 64 for row in manifest["scenarios"]
    )


@pytest.mark.parametrize(
    ("name", "triangles", "receiver_count", "grid_cells"),
    [
        ("analytic", 2, 3, 0),
        ("three_cube", 36, 3, 0),
    ],
)
def test_point_generators_record_hash_geometry_and_endpoint_scale(
    name: str, triangles: int, receiver_count: int, grid_cells: int
):
    first = build_scenario(name, tx_count=2, receiver_count=receiver_count)
    second = build_scenario(name, tx_count=2, receiver_count=receiver_count)

    assert first.record == second.record
    assert first.record.mode == "generated"
    assert first.record.triangle_count == triangles
    assert first.record.transmitter_count == 2
    assert first.record.receiver_count == receiver_count
    assert first.record.receiver_grid_cells == grid_cells
    assert first.record.source_path is None
    assert len(first.record.source_sha256) == 64
    assert len(first.record.scene_sha256) == 64


def test_terrain_generator_is_self_contained_and_records_grid_scale():
    bundle = build_scenario("terrain", tx_count=2, grid_shape=(4, 5))

    assert bundle.record.mode == "generated"
    assert bundle.record.triangle_count == 128
    assert bundle.record.transmitter_count == 2
    assert bundle.record.receiver_container_count == 1
    assert bundle.record.receiver_count == 20
    assert bundle.record.receiver_grid_cells == 20
    assert bundle.scene.metadata["phase_e_scenario"] == "terrain"


def test_full_city_missing_asset_root_fails_without_reduced_fallback(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.delenv(SCENE_ROOT_ENV, raising=False)

    with pytest.raises(FullScenarioAssetError, match="no reduced fallback"):
        build_scenario("munich_full", grid_shape=(2, 2))


def test_full_city_bad_asset_root_reports_searched_paths():
    with pytest.raises(
        FullScenarioAssetError,
        match="No reduced or synthetic city scene is substituted",
    ):
        build_scenario(
            "sf_full", asset_root=SCENARIO_MANIFEST_PATH.parent, grid_shape=(2, 2)
        )


def test_generated_scenario_rejects_external_asset_root():
    with pytest.raises(ValueError, match="asset_root is not accepted"):
        build_scenario("analytic", asset_root=SCENARIO_MANIFEST_PATH.parent)


def test_manifest_is_stable_json_and_generated_counts_match_declarations():
    payload = json.loads(SCENARIO_MANIFEST_PATH.read_text(encoding="utf-8"))
    generated = {
        row["name"]: row for row in payload["scenarios"] if row["mode"] == "generated"
    }

    for name, spec in generated.items():
        kwargs = {"grid_shape": (2, 3)} if name == "terrain" else {}
        bundle = build_scenario(name, **kwargs)
        assert bundle.record.triangle_count == spec["expected_triangle_count"]
        assert bundle.record.source_sha256 == spec["source_sha256"]