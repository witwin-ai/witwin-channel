import pytest
from types import MappingProxyType

from tests.support.bin import profile_deterministic_munich as profiler


def test_parse_args_accepts_munich_scaling_controls():
    args = profiler.parse_args(
        [
            "--grid-size",
            "64",
            "--max-diffractions",
            "2",
            "--shadow-boundary-correction",
            "--edge-selection-mode",
            "vertical_only",
            "--reflection-n-rays",
            "128",
            "--shadow-boundary-backend",
            "native_candidate",
            "--shadow-boundary-tile-shape",
            "16",
            "8",
            "--shadow-boundary-band-width-wavelengths",
            "4.0",
            "--shadow-boundary-max-candidate-factor",
            "128",
            "--solver-mode",
            "fast_approximate",
            "--memory-profile",
            "memory_safe",
            "--munich-xml",
            "E:/data/munich.xml",
            "--assert-peak-used-mib-below",
            "11000",
            "--assert-finite",
            "--json",
        ]
    )

    assert args.grid_size == 64
    assert args.max_diffractions == 2
    assert args.shadow_boundary_correction is True
    assert args.edge_selection_mode == "vertical_only"
    assert args.reflection_n_rays == 128
    assert args.shadow_boundary_backend == "native_candidate"
    assert tuple(args.shadow_boundary_tile_shape) == (16, 8)
    assert args.shadow_boundary_band_width_wavelengths == 4.0
    assert args.shadow_boundary_max_candidate_factor == 128
    assert args.solver_mode == "fast_approximate"
    assert args.memory_profile == "memory_safe"
    assert args.munich_xml.as_posix() == "E:/data/munich.xml"
    assert args.assert_peak_used_mib_below == 11000
    assert args.assert_finite is True
    assert args.json is True


def test_build_summary_keeps_profile_output_jsonable():
    summary = profiler.build_summary(
        environment={"native_extension_available": True},
        scenario={"grid_shape": (64, 64), "shadow_boundary_correction": False},
        memory={
            "before_scene_load": {"used_mib": 1000},
            "after_result_materialization": {"used_mib": 1500},
        },
        timing={
            "scene_load_seconds": 1.0,
            "solve_seconds": 2.0,
            "result_materialization_seconds": 0.5,
            "total_seconds": 3.5,
        },
        result_summary={
            "path_gain": {
                "shape": (64, 64),
                "finite": True,
            },
            "metadata": MappingProxyType(
                {
                    "performance_timing": MappingProxyType(
                        {"total_solve_seconds": 2.0}
                    )
                }
            ),
        },
        kernel_history=None,
    )

    assert summary["scenario"]["grid_shape"] == [64, 64]
    assert summary["timing"]["total_seconds"] == 3.5
    assert summary["memory"]["peak_delta_mib"] == 500
    assert summary["result"]["path_gain"]["shape"] == [64, 64]
    assert summary["result"]["metadata"]["performance_timing"]["total_solve_seconds"] == 2.0


def test_default_output_json_lives_under_tests_output():
    parts = profiler.DEFAULT_OUTPUT_JSON.parts

    assert parts[-2:] == ("output", "deterministic_munich_profile.json")
    assert parts[-3] == "tests"


def test_sionna_source_root_is_derived_from_munich_xml():
    xml = profiler.Path("root/src/sionna/rt/scenes/munich/munich.xml")

    assert profiler._sionna_source_root_from_xml(xml) == profiler.Path("root/src")


def test_validate_summary_enforces_finite_and_memory_gates():
    summary = {
        "memory": {"peak_used_mib": 12001},
        "result": {"path_gain": {"finite": True}},
    }

    with pytest.raises(AssertionError, match="peak GPU memory"):
        profiler.validate_summary(
            summary,
            assert_peak_used_mib_below=11000,
            assert_finite=True,
        )

    summary["memory"]["peak_used_mib"] = 9000
    summary["result"]["path_gain"]["finite"] = False

    with pytest.raises(AssertionError, match="path_gain contains non-finite"):
        profiler.validate_summary(
            summary,
            assert_peak_used_mib_below=11000,
            assert_finite=True,
        )
