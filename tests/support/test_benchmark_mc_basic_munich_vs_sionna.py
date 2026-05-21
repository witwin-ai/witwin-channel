from __future__ import annotations

from pathlib import Path

from tests.support.bin import benchmark_mc_basic_munich_vs_sionna as bench


def test_default_args_match_munich_protocol():
    args = bench.build_parser().parse_args([])

    assert args.grid_size == 256
    assert args.samples_per_tx == 1_000_000
    assert args.max_depth == 5
    assert args.frequency_hz == 2.4e9
    assert args.seeds == "11,17,23"
    assert args.witwin_accumulation_backend == "rayd_reflection_accumulation"
    assert args.sionna_source_root == bench.DEFAULT_SIONNA_SOURCE_ROOT
    assert args.munich_xml == bench.DEFAULT_MUNICH_XML
    assert args.output == bench.DEFAULT_OUTPUT_JSON


def test_parse_seeds_rejects_empty_tokens():
    assert bench.parse_seeds("11, 17,23") == (11, 17, 23)

    try:
        bench.parse_seeds("11,,23")
    except ValueError as exc:
        assert "empty seed" in str(exc)
    else:
        raise AssertionError("parse_seeds should reject empty seed tokens")


def test_default_paths_stay_inside_expected_trees():
    assert bench.DEFAULT_SIONNA_SOURCE_ROOT == Path(
        r"E:\Code\witwin-platform\channel\reference\sionna-rt-reference-2.0.1\src"
    )
    assert bench.DEFAULT_MUNICH_XML == (
        bench.DEFAULT_SIONNA_SOURCE_ROOT
        / "sionna"
        / "rt"
        / "scenes"
        / "munich"
        / "munich.xml"
    )
    assert bench.DEFAULT_OUTPUT_JSON == (
        bench.CHANNEL_ROOT
        / "docs"
        / "dev"
        / "optimization"
        / "mc_basic_munich_vs_sionna.json"
    )


def test_witwin_munich_edge_policy_matches_sionna_edge_diffraction():
    policy = bench._witwin_edge_policy()

    assert policy.edge_selection_mode == "all_edges"
    assert policy.edge_diffraction is True
    assert policy.boundary_edge_policy == "half_plane"
