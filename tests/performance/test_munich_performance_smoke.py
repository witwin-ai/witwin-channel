from __future__ import annotations

from pathlib import Path

import pytest

from tests.support.bin import benchmark_munich_performance as bench


pytestmark = [pytest.mark.gpu, pytest.mark.optimize]


def test_munich_path_basic_bdpt_performance_smoke(tmp_path: Path):
    if not bench.munich_base.DEFAULT_MUNICH_XML.exists():
        pytest.skip("Bundled Munich scene is not available.")

    args = bench.build_parser().parse_args(
        [
            "--cases",
            "path_order1,mc_basic_order1,mc_bdpt_order1",
            "--path-samples",
            "4096",
            "--path-max-num-paths",
            "16",
            "--mc-grid-size",
            "16",
            "--mc-samples-per-tx",
            "4096",
            "--mc-max-bounces",
            "1",
            "--warmup",
            "0",
            "--repeats",
            "1",
            "--output",
            str(tmp_path / "munich_performance_smoke.json"),
        ]
    )

    result = bench.run_benchmark(args)

    assert result["gates"]["status"] == "not_configured"
    assert [case["case_id"] for case in result["cases"]] == [
        "path_order1",
        "mc_basic_order1",
        "mc_bdpt_order1",
    ]
    for case in result["cases"]:
        assert case["ok"], case["error"]
        assert case["profile"]["median_ms"] > 0.0
        if case["solver"] == "path":
            assert case["stats"]["finite_tau"]
            assert case["stats"]["finite_field"]
        else:
            assert case["stats"]["finite"]
