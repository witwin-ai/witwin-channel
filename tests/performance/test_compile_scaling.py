# Copyright Xingyu Chen.
# Gate `scene.compile` cost against scene size.

"""Gate `scene.compile` cost against scene size.

`compile` reads all four Core version properties before it consults its cache,
so a cache hit still walks the whole world model, and every `solve` calls
`compile`. Nothing else covers this: the solver and consumer benchmarks compile
one scene once, outside their timing loops, and the Stage-I Phase-3 evidence
compiled an empty `Scene()`.

The regression this gate exists for moved compile from an O(1) integer read to
an O(N) walk. It was linear before and after, so the shape check alone would
have missed it; `calibrated_per_structure` is the budget that catches it.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest
import torch

from benchmarks import bench_compile_scaling as scaling

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
GATE_PATH = REPOSITORY_ROOT / "benchmarks" / "gates" / "compile_scaling.v1.json"

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="compile requires the CUDA runtime"
)


def _gate() -> dict:
    return json.loads(GATE_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def report() -> dict:
    gate = _gate()
    protocol = gate["protocol"]
    return scaling.run(
        argparse.Namespace(
            sizes=",".join(str(size) for size in protocol["sizes"]),
            warmup=protocol["warmup"],
            repeats=protocol["repeats"],
            frequency_hz=2.4e9,
            output=None,
        )
    )


def test_gate_declares_a_reproducible_protocol():
    gate = _gate()
    protocol = gate["protocol"]

    assert len(protocol["sizes"]) >= 2
    assert protocol["sizes"] == sorted(protocol["sizes"])
    assert protocol["statistic"] == "minimum sample"
    assert protocol["warmup"] >= 1
    assert protocol["repeats"] >= 5
    for name, budget in gate["budgets"].items():
        assert budget["gating"] in {"always", "never"}, name
        assert "meaning" in budget, name
        if budget["gating"] == "always":
            assert budget["max"] > budget["reference"], name


def test_compile_stays_linear_in_structure_count(report: dict):
    gate = _gate()["budgets"]["scaling_ratio"]
    summary = report["summary"]

    assert summary["size_ratio"] == pytest.approx(4.0)
    assert summary["scaling_ratio"] <= gate["max"], (
        f"compile scaling {summary['scaling_ratio']:.2f} exceeds {gate['max']}; "
        f"linear is {summary['size_ratio']}, quadratic is "
        f"{summary['size_ratio'] ** 2}"
    )


def test_compile_per_structure_constant_factor_is_bounded(report: dict):
    gate = _gate()["budgets"]["calibrated_per_structure"]
    summary = report["summary"]

    assert summary["calibration_us"] > 0
    assert summary["calibrated_per_structure"] <= gate["max"], (
        f"calibrated per-structure compile cost "
        f"{summary['calibrated_per_structure']:.4f} exceeds {gate['max']}; "
        f"raw {summary['max_per_structure_us']:.1f} us per structure against a "
        f"{summary['calibration_us']:.0f} us calibration"
    )


def test_compile_adds_no_second_pass_over_the_scene(report: dict):
    gate = _gate()["budgets"]["version_share"]
    summary = report["summary"]

    assert summary["version_share"] <= gate["max"], (
        f"warm compile is {summary['version_share']:.2f}x the cost of reading "
        f"the four version properties; a value near 2 means a second O(N) pass "
        f"over the scene was added"
    )


def test_report_records_every_measured_point(report: dict):
    gate_sizes = _gate()["protocol"]["sizes"]

    assert [point["structures"] for point in report["points"]] == gate_sizes
    for point in report["points"]:
        assert point["warm_min_ms"] > 0
        assert point["warm_min_ms"] <= point["warm_median_ms"]
        assert point["version_only_min_ms"] > 0
        assert point["per_structure_us"] > 0
    assert report["schema"] == scaling.SCHEMA_NAME
    assert report["environment"]["torch"] == torch.__version__


def test_scene_builder_varies_only_structure_count():
    small = scaling.build_scene(4)
    large = scaling.build_scene(16)

    assert len(small.structures) == 4
    assert len(large.structures) == 16
    assert len(small.endpoints) == len(large.endpoints) == 1
    faces = {
        tuple(structure.geometry.faces.shape) for structure in large.structures
    }
    assert faces == {(12, 3)}